import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "prediction_tracker"
OUT = ROOT / "artifacts" / "rating_dispersion"
BASE = "https://www.thepredictiontracker.com/"
AGENT = "Mozilla/5.0 (compatible; research-client)"
PAUSE_SECONDS = 2.0
NFL_FILES = {1999: "nfl99.csv", **{2000 + k: f"nfl{k:02d}.csv" for k in range(27)}}
CFB_FILES = {2000: "ncaa20000.csv", **{s: f"ncaa{s}.csv" for s in range(2001, 2027)}}
OUTCOME_COLUMNS = {
    "hscore",
    "rscore",
    "vscore",
    "actual",
    "total",
    "difference",
    "phcover",
    "phwin",
    "hcover",
    "hwin",
}
MARKET_OR_AGGREGATE = {
    "line",
    "lineopen",
    "linemidweek",
    "lineavg",
    "linemed",
    "linemedian",
    "linestd",
    "linesystemavg",
    "linecons",
}
MIN_SYSTEMS_FOR_SD = 2
NFL_NAMES = {
    "arizona": "ARI",
    "atlanta": "ATL",
    "baltimore": "BAL",
    "buffalo": "BUF",
    "carolina": "CAR",
    "chicago": "CHI",
    "cincinnati": "CIN",
    "cleveland": "CLE",
    "dallas": "DAL",
    "denver": "DEN",
    "detroit": "DET",
    "green bay": "GB",
    "houston": "HOU",
    "indianapolis": "IND",
    "jacksonville": "JAX",
    "kansas city": "KC",
    "la chargers": "LAC",
    "l.a. chargers": "LAC",
    "san diego": "LAC",
    "la rams": "LA",
    "l.a. rams": "LA",
    "los angeles": "LA",
    "st. louis": "LA",
    "st louis": "LA",
    "las vegas": "LV",
    "oakland": "LV",
    "miami": "MIA",
    "minnesota": "MIN",
    "new england": "NE",
    "new orleans": "NO",
    "n.y. giants": "NYG",
    "ny giants": "NYG",
    "n.y. jets": "NYJ",
    "ny jets": "NYJ",
    "philadelphia": "PHI",
    "pittsburgh": "PIT",
    "seattle": "SEA",
    "san francisco": "SF",
    "tampa bay": "TB",
    "tennessee": "TEN",
    "washington": "WAS",
}


def fetch():
    for league, files in (("nfl", NFL_FILES), ("cfb", CFB_FILES)):
        d = RAW / league
        d.mkdir(parents=True, exist_ok=True)
        for season, name in files.items():
            target = d / f"{season}.csv"
            req = urllib.request.Request(BASE + name, headers={"User-Agent": AGENT})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = r.read()
                    kind = r.headers.get("Content-Type", "")
            except Exception as e:
                print(league, season, "error", e, flush=True)
                time.sleep(PAUSE_SECONDS)
                continue
            if "csv" in kind:
                target.write_bytes(body)
                print(league, season, len(body), flush=True)
            else:
                print(league, season, "not csv", kind, flush=True)
            time.sleep(PAUSE_SECONDS)
    meta = {
        "source": BASE,
        "nfl_archive": BASE + "nflarchive.html",
        "cfb_archive": BASE + "ncaaarchive.html",
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (RAW / "fetch_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def read_pt(path):
    head = pd.read_csv(path, nrows=0, encoding="latin-1").columns
    keep = [c for c in head if c.strip().lower() not in OUTCOME_COLUMNS]
    t = pd.read_csv(path, usecols=keep, encoding="latin-1")
    t.columns = [c.strip().lower() for c in t.columns]
    return t


def system_columns(cols):
    return [c for c in cols if c.startswith("line") and c not in MARKET_OR_AGGREGATE]


def summarize(t):
    sys_cols = system_columns(t.columns)
    v = t[sys_cols].apply(pd.to_numeric, errors="coerce")
    n = v.notna().sum(axis=1)
    out = pd.DataFrame(
        {
            "pt_home": t["home"].astype(str).str.strip(),
            "pt_road": t["road"].astype(str).str.strip(),
            "pt_week": pd.to_numeric(t["week"], errors="coerce") if "week" in t else np.nan,
            "systems": n,
            "dispersion": v.std(axis=1, ddof=1).where(n >= MIN_SYSTEMS_FOR_SD),
            "system_median": v.median(axis=1),
            "pt_open": pd.to_numeric(t["lineopen"], errors="coerce")
            if "lineopen" in t
            else np.nan,
        }
    )
    return out, sys_cols


def norm(s):
    s = s.lower().replace("&", "and")
    s = re.sub(r"\bst\.?(\s|$)", r"state\1", s)
    return re.sub(r"[^a-z0-9]", "", s)


def cfb_map(pt, repo):
    names = pd.unique(pd.concat([repo["home_team"], repo["away_team"]]))
    direct = {norm(n): n for n in names}
    mapping = {}
    for n in pd.unique(pd.concat([pt["pt_home"], pt["pt_road"]])):
        if norm(n) in direct:
            mapping[n] = direct[norm(n)]
    pairs = repo[["season", "home_team", "away_team"]]
    for _ in range(4):
        votes = {}
        for side, other in (("pt_home", "pt_road"), ("pt_road", "pt_home")):
            known = pt[pt[other].isin(mapping) & ~pt[side].isin(mapping)]
            rside = "home_team" if side == "pt_home" else "away_team"
            rother = "away_team" if side == "pt_home" else "home_team"
            k = known.assign(r_other=known[other].map(mapping)).merge(
                pairs.rename(columns={rother: "r_other", rside: "cand"})[
                    ["season", "r_other", "cand"]
                ],
                on=["season", "r_other"],
            )
            for (a, b), c in k.groupby([side, "cand"]).size().items():
                votes[(a, b)] = votes.get((a, b), 0) + c
        if not votes:
            break
        v = pd.Series(votes).rename("n").reset_index()
        v.columns = ["pt", "cand", "n"]
        v = v[~v["cand"].isin(set(mapping.values()))]
        best = v.sort_values("n", ascending=False).groupby("pt").head(2)
        for name, g in best.groupby("pt"):
            top = g["n"].to_numpy()
            if len(top) == 1 or top[0] > 2 * top[1]:
                mapping[name] = g["cand"].iloc[0]
    return mapping


def join():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    nfl = pd.read_parquet(
        ROOT / "data" / "processed" / "game_features.parquet",
        columns=["game_id", "season", "week", "home_team", "away_team"],
    )
    cfb = pd.read_parquet(
        ROOT / "data" / "processed" / "cfb_game_features.parquet",
        columns=["game_id", "season", "week", "home_team", "away_team"],
    )
    for league, repo in (("nfl", nfl), ("cfb", cfb)):
        frames, cols_by_season = [], {}
        for path in sorted((RAW / league).glob("*.csv")):
            season = int(path.stem)
            s, sys_cols = summarize(read_pt(path))
            s["season"] = season
            frames.append(s)
            cols_by_season[season] = sys_cols
        pt = pd.concat(frames, ignore_index=True)
        if league == "nfl":
            mapping = {n: NFL_NAMES.get(n.lower()) for n in pd.unique(pd.concat([pt["pt_home"], pt["pt_road"]]))}
            mapping = {k: v for k, v in mapping.items() if v}
        else:
            mapping = cfb_map(pt, repo)
        pt["home_team"] = pt["pt_home"].map(mapping)
        pt["away_team"] = pt["pt_road"].map(mapping)
        keys = ["season", "home_team", "away_team"]
        if league == "nfl":
            pt["week"] = pt["pt_week"]
            repo = repo.assign(week=pd.to_numeric(repo["week"], errors="coerce"))
            keys = [*keys, "week"]
        else:
            repo = repo.drop(columns=["week"])
        pt["row"] = np.arange(len(pt))
        r = repo.drop_duplicates(keys, keep=False)
        p1 = pt.drop_duplicates(keys, keep=False)
        m = p1.merge(r, on=keys, how="inner")
        rest = p1[~p1["row"].isin(m["row"])]
        flip = rest.merge(
            r[~r["game_id"].isin(m["game_id"])].rename(
                columns={"home_team": "away_team", "away_team": "home_team"}
            ),
            on=keys,
            how="inner",
        )
        flip["system_median"] = -flip["system_median"]
        flip["pt_open"] = -flip["pt_open"]
        flip["pt_home"], flip["pt_road"] = flip["pt_road"], flip["pt_home"]
        both = pd.concat([m.assign(orientation="same"), flip.assign(orientation="swapped")])
        both = both.drop_duplicates(["game_id"], keep=False)
        both.to_parquet(OUT / f"{league}_dispersion.parquet", index=False)
        unmapped = sorted(set(pd.concat([pt["pt_home"], pt["pt_road"]])) - set(mapping))
        report[league] = {
            "pt_rows": len(pt),
            "seasons": sorted(cols_by_season),
            "joined_games": len(both),
            "swapped_orientation": int((both["orientation"] == "swapped").sum()),
            "joined_by_season": {int(k): int(v) for k, v in both.groupby("season").size().items()},
            "repo_games_by_season": {
                int(k): int(v)
                for k, v in repo[repo["season"].isin(cols_by_season)].groupby("season").size().items()
            },
            "median_systems": float(both["systems"].median()),
            "systems_by_season": {int(k): len(v) for k, v in cols_by_season.items()},
            "unmapped_names": len(unmapped),
            "unmapped_sample": unmapped[:30],
        }
    (OUT / "join_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: {a: b for a, b in v.items() if a != "unmapped_sample"} for k, v in report.items()}))
    print({k: v["unmapped_sample"] for k, v in report.items()})


LANE = "docs/lanes/rating-dispersion-2026-11-16.md"
FAMILY = "rating_dispersion_cfb_u1"
STREAM = 41
SCREEN = {
    "dispersion_temper": {
        "hypothesis": "negative",
        "what": (
            "the served-equivalent logit times standardized D, D the per-game standard "
            "deviation of predicted margins across Prediction Tracker computer systems "
            "(market and aggregate columns excluded), so one coefficient shrinks or stretches "
            "confidence"
        ),
        "plain": "the computer ratings disagree more than usual about",
    },
    "median_open_gap": {
        "hypothesis": "positive",
        "what": (
            "the Prediction Tracker systems' median predicted home margin minus the opening "
            "spread (control, expected to be priced)"
        ),
        "plain": "the computer ratings' middle pick sits away from the opening line for",
    },
}


def cells(m36, name, res, seasons, draws, power_lane):
    spec = SCREEN[name]
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
                "name": f"rating_dispersion_cfb_{name}_{units}",
                "description": (
                    f"College-first rating-dispersion family, {name}: {spec['what']}. One "
                    "offset-logistic coefficient on the leave-one-season-out ridge cover "
                    f"probability with the opener as input and grade, {len(seasons)} seasons; "
                    f"declared sign {spec['hypothesis']}. Changed picks {held['changed']}, "
                    f"record {held['arm_right']}-{held['ref_right']}; coefficient per SD "
                    f"{res['coefficient_per_sd']:+.4f}; season halves {early:+.4f} and "
                    f"{late:+.4f}; percentile {res['null_percentile'][k]:.1f} of {draws} "
                    "within-season shuffles of D; in-sample minus held-out "
                    f"{res['in_sample_minus_held_out'][k]:+.6f}."
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
                    "No resolved wrong sign for the declared mechanism, split-half reliability "
                    "not measured, and no positive control bounds an effect this small "
                    f"({power_lane})."
                ),
                "notes": (
                    f"Family {FAMILY}: 2 candidates x 3 metrics = 6 recorded cells; 2 gating "
                    "log-loss reads in the matched null, 4 season-half coefficients; declared "
                    "in artifacts/rating_dispersion/declaration.json before the outcome join; "
                    f"survivor: {'yes' if res['survivor'] else 'no'}."
                ),
                "interval_low": round(held[k]["low"], 6),
                "interval_high": round(held[k]["high"], 6),
                "probability_positive": round(held[k]["pp"], 4),
                "sample_games": int(res["games"]),
                "sample_blocks": len(seasons),
                "plain_summary": (
                    f"Checks whether college games {spec['plain']} beat the opening spread "
                    f"more or less often than the model expects. {verdict}"
                ),
            }
        )
    return out


def screen():
    from scipy.special import expit

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import xlg14_luck_family as u3

    u3.STREAM_LUCK = STREAM
    hp = u3.load_power()
    m36 = hp.M36
    d = hp.x14.cfb_frame()
    pt = pd.read_parquet(OUT / "cfb_dispersion.parquet")
    pt = pt[["game_id", "dispersion", "system_median"]]
    d = d.assign(game_id=pd.to_numeric(d["game_id"]).astype("int64"))
    pt = pt.assign(game_id=pd.to_numeric(pt["game_id"]).astype("int64"))
    d = d.merge(pt, on="game_id", how="inner")
    d = d[d["dispersion"].notna()].sort_values(["season", "game_id"]).reset_index(drop=True)
    dz = ((d["dispersion"] - d["dispersion"].mean()) / d["dispersion"].std(ddof=1)).to_numpy()
    z = d["logit_p"].to_numpy(float)
    frame = pd.DataFrame(
        {
            "game_id": d["game_id"].to_numpy(),
            "season": d["season"].to_numpy(int),
            "home_covered": d["y"].to_numpy(float),
            m36.SERVED_COLUMN: expit(z),
            "dispersion_temper": z * dz,
            "median_open_gap": (d["system_median"] - d["spread_open"]).to_numpy(float),
        }
    )
    seasons = sorted(int(s) for s in frame["season"].unique())
    original = m36.permuted

    def shuffle_d(st, rng):
        out = dz.copy()
        for k in st["groups"]:
            out[k] = dz[rng.permutation(k)]
        return st["z"] * out / st["sd"]

    out = {"family": FAMILY, "games": len(frame), "seasons": seasons, "draws": hp.DRAWS}
    out["candidates"], batch, lines = {}, [], []
    for index, n in enumerate(SCREEN):
        m36.permuted = shuffle_d if n == "dispersion_temper" else original
        try:
            res, st = u3.read_candidate(hp, frame, n, index)
        finally:
            m36.permuted = original
        del st
        bar = res["survivor_bar_percentile"]
        res["survivor"] = bool(res["null_percentile"]["ll"] > bar and res["half_signs_agree"])
        out["candidates"][n] = res
        batch.extend(cells(m36, n, res, seasons, hp.DRAWS, u3.POWER_LANE))
        h = res["held_out"]
        lines.append(
            f"{n}: games {res['games']} changed {h['changed']} record {h['arm_right']}-"
            f"{h['ref_right']}; acc {h['acc']['effect']:+.3f} [{h['acc']['low']:+.3f},"
            f"{h['acc']['high']:+.3f}] pp {h['acc']['pp']:.3f}; ll {h['ll']['effect']:+.6f} "
            f"[{h['ll']['low']:+.6f},{h['ll']['high']:+.6f}] pp {h['ll']['pp']:.3f} pct "
            f"{res['null_percentile']['ll']:.1f}; brier {h['brier']['effect']:+.6f} pct "
            f"{res['null_percentile']['brier']:.1f}; in-sample ll gap "
            f"{res['in_sample_minus_held_out']['ll']:+.6f}; coef {res['coefficient_per_sd']:+.4f} "
            f"folds {res['folds_with_full_sign']}/{len(seasons)}; halves "
            f"{res['halves']['earlier']['coefficient_per_sd']:+.4f} / "
            f"{res['halves']['later']['coefficient_per_sd']:+.4f}; survivor {res['survivor']}"
        )
        print(lines[-1], flush=True)
    out["survivors"] = [n for n in SCREEN if out["candidates"][n]["survivor"]]
    lines.append(f"survivors: {out['survivors']}")
    (OUT / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": batch}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"fetch": fetch, "join": join, "screen": screen}[sys.argv[1]]()
