import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

import xlg14_luck_family as u3  # noqa: E402

RAW = ROOT / "data" / "external" / "coaches"
OUT = ROOT / "artifacts" / "coach_familiarity"
TEAM_INFO = ROOT / "data" / "cfb" / "team_info" / "raw"
LANE = "docs/lanes/coach-familiarity-2026-11-16.md"
POWER_LANE = u3.POWER_LANE
FAMILY = "coach_familiarity_family_v1"
STREAM = 14
ELIGIBILITY_SEASONS = 4
FIRST_YEAR = 2000
LAST_YEAR = 2025
API = "https://api.collegefootballdata.com/coaches"
CANDIDATES = {
    "hc_former_employer_gap": {
        "what": (
            "home flag minus away flag, flag = the team's head coach this season was head coach "
            f"of the opponent in any of the previous {ELIGIBILITY_SEASONS} seasons."
        ),
        "plain": "are coached by a man who ran the opponent's program in the last few years",
    },
    "hc_former_employer_resid": {
        "what": (
            "the same former-employer gap residualised on home minus away own cover margin "
            "against the opener over earlier games this season."
        ),
        "plain": (
            "are coached by a man who ran the opponent's program recently, after allowing "
            "for how both teams have done against the spread so far"
        ),
    },
}
NAMES = list(CANDIDATES)


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / ".gitignore").write_text("*\n", encoding="utf-8")
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        raise SystemExit("CFBD_API_KEY missing")
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        path = RAW / f"coaches_{year}.json"
        if path.exists():
            continue
        url = API + "?" + urllib.parse.urlencode({"year": year})
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "research-client/1.0",
                "Accept": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
        path.write_bytes(body)
        print(year, len(json.loads(body)), flush=True)
        time.sleep(1.0)


def team_ids():
    frames = [pd.read_parquet(p) for p in sorted(TEAM_INFO.glob("*/season=*/team_info.parquet"))]
    t = pd.concat(frames)[["team_id", "school"]].drop_duplicates()
    return dict(zip(t["school"], t["team_id"].astype("int64"), strict=False))


def head_coaches():
    ids = team_ids()
    rows, unmatched = [], set()
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        for c in json.loads((RAW / f"coaches_{year}.json").read_text(encoding="utf-8")):
            name = f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()
            for s in c.get("seasons", []):
                if int(s["year"]) != year:
                    continue
                tid = ids.get(s["school"])
                if tid is None:
                    unmatched.add(s["school"])
                    continue
                rows.append({"coach": name, "team": tid, "season": year})
    hc = pd.DataFrame(rows).drop_duplicates()
    return hc, sorted(unmatched)


def flags(hc):
    count = hc.groupby(["team", "season"])["coach"].transform("nunique")
    single = hc[count.eq(1)]
    past = {}
    for coach, team, season in hc[["coach", "team", "season"]].itertuples(index=False):
        past.setdefault(coach, []).append((season, team))
    out = {}
    for coach, team, season in single[["coach", "team", "season"]].itertuples(index=False):
        former = {
            t
            for y, t in past.get(coach, [])
            if season - ELIGIBILITY_SEASONS <= y <= season - 1 and t != team
        }
        out[(team, season)] = former
    return out


def ats_to_date(d, cutoff=None):
    g = d[["game_id", "season", "week", "home_id", "away_id", "result", "spread_open"]].copy()
    g["week"] = pd.to_numeric(g["week"]).astype(int)
    m = (g["result"] - g["spread_open"]).to_numpy(float)
    long = pd.concat(
        [
            pd.DataFrame({"game_id": g["game_id"], "season": g["season"], "week": g["week"],
                          "team": g["home_id"].astype("int64"), "cover": m}),
            pd.DataFrame({"game_id": g["game_id"], "season": g["season"], "week": g["week"],
                          "team": g["away_id"].astype("int64"), "cover": -m}),
        ]
    )
    if cutoff is not None:
        season, week = cutoff
        later = long["season"].eq(season) & long["week"].ge(week)
        long.loc[later & long["week"].gt(week), "cover"] = np.nan
        long = long[~(later & long["week"].gt(week))]
    long = long.sort_values(["team", "season", "week"])
    grp = long.groupby(["team", "season"])["cover"]
    long["prior"] = (grp.cumsum() - long["cover"]) / grp.cumcount().replace(0, np.nan)
    long["prior"] = long["prior"].fillna(0.0)
    return long.set_index(["game_id", "team"])["prior"]


def terms(d, fam):
    gid = d["game_id"]
    home = d["home_id"].astype("int64").to_numpy()
    away = d["away_id"].astype("int64").to_numpy()
    season = d["season"].astype(int).to_numpy()
    hf = np.array([a in fam.get((h, s), set()) for h, a, s in zip(home, away, season, strict=True)],
                  float)
    af = np.array([h in fam.get((a, s), set()) for h, a, s in zip(home, away, season, strict=True)],
                  float)
    gap = hf - af
    prior = ats_to_date(d)
    ats = (
        prior.reindex(pd.MultiIndex.from_arrays([gid, home])).to_numpy()
        - prior.reindex(pd.MultiIndex.from_arrays([gid, away])).to_numpy()
    )
    x = np.column_stack([np.ones(len(ats)), ats])
    beta = np.linalg.lstsq(x, gap, rcond=None)[0]
    resid = gap - x @ beta
    return gap, resid, ats, hf, af, beta


def asof_check(d):
    rng = np.random.default_rng([36, STREAM, 1])
    full = ats_to_date(d)
    keys = d[["season", "week"]].drop_duplicates().to_numpy()
    worst, rows = 0.0, 0
    for i in rng.choice(len(keys), 5, replace=False):
        s, w = int(keys[i][0]), int(keys[i][1])
        cut = ats_to_date(d, (s, w))
        sel = d[(d["season"] == s) & (pd.to_numeric(d["week"]).astype(int) == w)]
        for side in ("home_id", "away_id"):
            k = pd.MultiIndex.from_arrays([sel["game_id"], sel[side].astype("int64")])
            diff = np.abs(full.reindex(k).to_numpy() - cut.reindex(k).to_numpy())
            worst = max(worst, float(np.nanmax(diff)) if len(diff) else 0.0)
            rows += len(k)
    return {"cutoffs": 5, "team_games": rows, "max_diff": worst}


def cells(m36, name, res, seasons, draws):
    spec = CANDIDATES[name]
    held = res["held_out"]
    early = res["halves"]["earlier"]["coefficient_per_sd"]
    late = res["halves"]["later"]["coefficient_per_sd"]
    verdict = (
        "It stood out from chance and earns one declared pro-football check."
        if res["survivor"]
        else "No effect large enough for this test to see showed up."
    )
    out = []
    for k, units in zip(m36.SHORT, m36.METRICS, strict=True):
        out.append(
            {
                "name": f"coach_familiarity_cfb_{name}_{units}",
                "description": (
                    f"College coach-familiarity family, {name}: {spec['what']} Season-level, "
                    "known preseason; school-seasons with more than one listed head coach set "
                    "to 0. One offset-logistic coefficient on a leave-one-season-out ridge cover "
                    f"probability with the opener as input and grade, {len(seasons)} seasons; "
                    f"declared sign positive. Nonzero games {res['nonzero_games']}; changed "
                    f"picks {held['changed']}, record {held['arm_right']}-{held['ref_right']}; "
                    f"coefficient per SD {res['coefficient_per_sd']:+.4f}; season-half "
                    f"coefficients {early:+.4f} and {late:+.4f}; this metric sits at percentile "
                    f"{res['null_percentile'][k]:.1f} of {draws} within-season shuffles; "
                    f"in-sample minus held-out {res['in_sample_minus_held_out'][k]:+.6f}."
                ),
                "source": LANE,
                "effect": round(held[k]["effect"], 6),
                "effect_units": units,
                "classification": "unresolved_below_power",
                "league": "cfb",
                "season_start": int(seasons[0]),
                "season_end": int(seasons[-1]),
                "family": FAMILY,
                "category": "market",
                "classification_evidence": (
                    "No resolved wrong sign for the declared familiarity mechanism, split-half "
                    "reliability not measured, and no positive control bounds an effect this "
                    "small on a sparse flag: the college add-one harness shows about .08 to .11 "
                    f"logit per SD four times in five ({POWER_LANE})."
                ),
                "notes": (
                    f"Family {FAMILY}: 2 candidates x 3 metrics = 6 recorded cells; 2 gating "
                    "reads (log-loss percentile in the matched null), 4 season-half "
                    "coefficients and control reads; survivor rule declared in the lane before "
                    f"any outcome was joined; survivor: {'yes' if res['survivor'] else 'no'}."
                ),
                "interval_low": round(held[k]["low"], 6),
                "interval_high": round(held[k]["high"], 6),
                "probability_positive": round(held[k]["pp"], 4),
                "sample_games": int(res["games"]),
                "sample_blocks": len(seasons),
                "plain_summary": (
                    f"Checks whether college teams that {spec['plain']} beat the opening "
                    f"spread more often. {verdict}"
                ),
            }
        )
    return out


def screen():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    u3.STREAM_LUCK = STREAM
    hp = u3.load_power()
    m36 = hp.M36
    d = hp.x14.cfb_frame()
    hc, unmatched = head_coaches()
    fam = flags(hc)
    gap, resid, ats, hf, af, beta = terms(d, fam)
    build = {
        "head_coach_rows": len(hc),
        "unmatched_schools": len(unmatched),
        "unmatched_sample": unmatched[:15],
        "team_seasons_flaggable": len(fam),
        "team_seasons_with_former_employer": int(sum(bool(v) for v in fam.values())),
        "games": len(d),
        "home_flag": int(hf.sum()),
        "away_flag": int(af.sum()),
        "both_flag": int((hf * af).sum()),
        "gap_nonzero": int((gap != 0).sum()),
        "gap_sd": float(gap.std()),
        "corr_gap_ats_to_date": float(np.corrcoef(gap, ats)[0, 1]),
        "resid_beta": [float(b) for b in beta],
        "nonzero_by_season": {
            int(s): int(v) for s, v in pd.Series(gap != 0).groupby(d["season"].to_numpy()).sum().items()
        },
        "asof": asof_check(d),
    }
    print(json.dumps(build), flush=True)
    frame = pd.DataFrame(
        {
            "game_id": d["game_id"].to_numpy(),
            "season": d["season"].to_numpy(int),
            "home_covered": d["y"].to_numpy(float),
            m36.SERVED_COLUMN: expit(d["logit_p"].to_numpy(float)),
            NAMES[0]: gap,
            NAMES[1]: resid,
        }
    )
    seasons = sorted(int(s) for s in frame["season"].unique())
    out = {"family": FAMILY, "build": build, "games": len(frame), "seasons": seasons,
           "draws": hp.DRAWS, "seed": [hp.SEED, STREAM], "candidates": {}}
    batch, lines = [], []
    for index, n in enumerate(NAMES):
        res, st = u3.read_candidate(hp, frame, n, index)
        if index == 0:
            out["controls"] = u3.controls(m36, st)
        del st
        out["candidates"][n] = res
        batch.extend(cells(m36, n, res, seasons, hp.DRAWS))
        h = res["held_out"]
        lines.append(
            f"{n}: games {res['games']} nonzero {res['nonzero_games']}; changed {h['changed']} "
            f"record {h['arm_right']}-{h['ref_right']}; acc {h['acc']['effect']:+.3f} "
            f"[{h['acc']['low']:+.3f},{h['acc']['high']:+.3f}] pp {h['acc']['pp']:.3f} "
            f"pct {res['null_percentile']['acc']:.1f}; ll {h['ll']['effect']:+.6f} "
            f"[{h['ll']['low']:+.6f},{h['ll']['high']:+.6f}] pp {h['ll']['pp']:.3f} "
            f"pct {res['null_percentile']['ll']:.1f}; brier pct "
            f"{res['null_percentile']['brier']:.1f}; null ll [{res['null']['ll']['q05']:+.6f},"
            f"{res['null']['ll']['q95']:+.6f}]; gap {res['in_sample_minus_held_out']['ll']:+.6f}; "
            f"coef {res['coefficient_per_sd']:+.4f} folds {res['folds_with_full_sign']}/"
            f"{len(seasons)}; halves {res['halves']['earlier']['coefficient_per_sd']:+.4f} / "
            f"{res['halves']['later']['coefficient_per_sd']:+.4f}; survivor {res['survivor']}"
        )
        print(lines[-1], f"{time.time() - started:.0f}s", flush=True)
    out["survivors"] = [n for n in NAMES if out["candidates"][n]["survivor"]]
    lines.append(
        "controls: "
        + "; ".join(f"{k} acc {v['accuracy']:.3f} ll {v['log_loss']:.6f}"
                    for k, v in out["controls"].items())
    )
    lines.append(f"survivors: {out['survivors']}")
    print(lines[-2], lines[-1], sep="\n", flush=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": batch}, indent=1),
                                                 encoding="utf-8")


if __name__ == "__main__":
    {"fetch": fetch, "screen": screen}[sys.argv[1]]()
