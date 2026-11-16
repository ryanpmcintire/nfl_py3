import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

import xlg14_luck_family as u3  # noqa: E402

from nfl_ats.cfb_features import cfb_season_partitions  # noqa: E402

x14 = u3.x14
OUT = ROOT / "artifacts" / "xlg14_family_u5"
LANE = "docs/lanes/xlg-14-u5-2026-11-16.md"
POWER_LANE = u3.POWER_LANE
FAMILY = "xlg14_composition_family_u5"
STREAM = 5
PLAY_COLUMNS = [
    "game_id",
    "week",
    "seasonType",
    "pos_team_id",
    "def_pos_team_id",
    "homeTeamId",
    "rush",
    "pass",
    "kneel_down",
    "EPA",
    "EPA_success",
    "home_wp_before",
    "away_wp_before",
]
SUMS = ["n", "epa", "sr", "w", "wepa"]
SIDES = ["o", "d"]
GAME_COLUMNS = [f"{s}_{k}" for s in SIDES for k in SUMS]
CANDIDATES = {
    "off_bigplay_residual": {
        "hypothesis": "negative",
        "what": (
            "offensive EPA per play minus the EPA per play its success rate predicts, slope and "
            "intercept fitted on team-games in strictly earlier seasons."
        ),
        "plain": (
            "have gained more of their offense from big plays "
            "than their down-to-down success explains"
        ),
    },
    "def_bigplay_residual": {
        "hypothesis": "positive",
        "what": (
            "allowed EPA per play minus the EPA per play the allowed success rate predicts, "
            "same earlier-season fit."
        ),
        "plain": "have allowed more big plays than their down-to-down defense explains",
    },
    "off_leverage_gap": {
        "hypothesis": "negative",
        "what": (
            "offensive EPA per play unweighted minus EPA per play weighted by wp(1 - wp) of the "
            "offence before the snap (production piled up in lopsided game states)."
        ),
        "plain": "have piled up offense mostly when the game was already decided",
    },
    "def_leverage_gap": {
        "hypothesis": "positive",
        "what": (
            "allowed EPA per play unweighted minus allowed EPA per play weighted by wp(1 - wp) "
            "before the snap."
        ),
        "plain": "have given up yardage mostly when the game was already decided",
    },
}
NAMES = list(CANDIDATES)


def season_tables(season, path):
    f = pd.read_parquet(path, columns=PLAY_COLUMNS)
    f = f[pd.to_numeric(f["seasonType"], errors="coerce").eq(u3.REGULAR_SEASON)].copy()
    f["week"] = pd.to_numeric(f["week"], errors="coerce")
    f = f[f["pos_team_id"].notna() & f["def_pos_team_id"].notna() & f["week"].notna()]
    f = f.reset_index(drop=True)
    f["game_id"] = pd.to_numeric(f["game_id"]).astype("int64")
    f["pos"] = pd.to_numeric(f["pos_team_id"]).astype("int64")
    f["dfn"] = pd.to_numeric(f["def_pos_team_id"]).astype("int64")
    pairs = f[["game_id", "pos", "dfn"]].drop_duplicates()
    pairs = pairs[pairs["pos"].ne(pairs["dfn"])]
    teams = pd.concat(
        [
            pairs.rename(columns={"pos": "team", "dfn": "opp"}),
            pairs.rename(columns={"dfn": "team", "pos": "opp"}),
        ]
    ).drop_duplicates()
    count = teams.groupby("game_id")["team"].transform("nunique")
    teams = teams[count.eq(2)].drop_duplicates(["game_id", "team"]).reset_index(drop=True)
    week = f.groupby("game_id")["week"].first()
    scrim = pd.to_numeric(f["rush"], errors="coerce").eq(1) | pd.to_numeric(
        f["pass"], errors="coerce"
    ).eq(1)
    kneel = pd.to_numeric(f["kneel_down"], errors="coerce").eq(1)
    f["epa"] = pd.to_numeric(f["EPA"], errors="coerce")
    p = f[scrim & ~kneel & f["epa"].notna() & f["pos"].ne(f["dfn"])].copy()
    home = pd.to_numeric(p["homeTeamId"], errors="coerce").eq(p["pos"])
    wp = np.where(
        home,
        pd.to_numeric(p["home_wp_before"], errors="coerce"),
        pd.to_numeric(p["away_wp_before"], errors="coerce"),
    )
    lev = wp * (1.0 - wp)
    ok = np.isfinite(lev)
    p["n"] = 1.0
    p["sr"] = pd.to_numeric(p["EPA_success"], errors="coerce").fillna(0.0)
    p["w"] = np.where(ok, lev, 0.0)
    p["wepa"] = p["w"] * p["epa"]
    agg = {}
    for side, key in (("o", "pos"), ("d", "dfn")):
        a = p.groupby(["game_id", key])[SUMS].sum()
        a.index = a.index.set_names(["game_id", "team"])
        agg[side] = a.add_prefix(f"{side}_")
    t = pd.concat([agg["o"], agg["d"]], axis=1).reset_index()
    m = teams.merge(t, on=["game_id", "team"], how="left")
    raw = pd.DataFrame(
        {
            "game_id": m["game_id"].astype("int64"),
            "team": m["team"].astype("int64"),
            "season": season,
        }
    )
    raw["week"] = m["game_id"].map(week).to_numpy(float)
    for c in GAME_COLUMNS:
        raw[c] = m[c].fillna(0.0).to_numpy(float)
    diag = {
        "plays": len(f),
        "scrimmage_plays": len(p),
        "team_games": len(raw),
        "share_wp_nonnull": float(ok.mean()) if len(ok) else 0.0,
        "team_games_without_offense": int(raw["o_n"].eq(0).sum()),
    }
    return raw, diag


def season_fits(raw):
    g = raw[raw["o_n"] > 0]
    x = g["o_sr"] / g["o_n"]
    y = g["o_epa"] / g["o_n"]
    fits = {}
    for s in sorted(raw["season"].unique()):
        e = g["season"] < s
        if e.sum() < 2 or x[e].var() == 0:
            continue
        b = float(np.cov(x[e], y[e], ddof=1)[0, 1] / x[e].var(ddof=1))
        fits[int(s)] = (float(y[e].mean() - b * x[e].mean()), b)
    return fits


def to_date(raw, fits=None):
    fits = season_fits(raw) if fits is None else fits
    r = raw.reset_index(drop=True).copy()
    r["games"] = 1.0
    cols = [*GAME_COLUMNS, "games"]
    tw = r.groupby(["team", "season", "week"])[cols].sum().reset_index()
    prior = tw.groupby(["team", "season"])[cols].cumsum() - tw[cols]
    out = tw[["team", "season", "week"]].copy()
    a = out["season"].map({s: v[0] for s, v in fits.items()})
    b = out["season"].map({s: v[1] for s, v in fits.items()})
    for side, name in (("o", "off"), ("d", "def")):
        n = prior[f"{side}_n"].where(prior[f"{side}_n"] > 0)
        w = prior[f"{side}_w"].where(prior[f"{side}_w"] > 0)
        rate = prior[f"{side}_epa"] / n
        out[f"{name}_bigplay_residual"] = (rate - (a + b * prior[f"{side}_sr"] / n)).fillna(0.0)
        out[f"{name}_leverage_gap"] = (rate - prior[f"{side}_wepa"] / w).fillna(0.0)
    out["prior_games"] = prior["games"]
    return r[["game_id", "team", "season", "week"]].merge(
        out, on=["team", "season", "week"], how="left"
    )


def asof_check(raw, values):
    rng = np.random.default_rng([u3.ASOF_SEED, STREAM, u3.ASOF_STREAM])
    pool = values[values["season"].isin(x14.u1.SEASONS)][["season", "week"]].drop_duplicates()
    picks = pool.sample(u3.ASOF_CUTOFFS, random_state=rng)
    worst, rows, cutoffs = 0.0, 0, []
    for s, w in picks.itertuples(index=False):
        r2 = raw[(raw["season"] < s) | ((raw["season"] == s) & (raw["week"] <= w))].copy()
        r2.loc[(r2["season"] == s) & (r2["week"] == w), GAME_COLUMNS] = 0.0
        v2 = to_date(r2)
        key = ["game_id", "team"]
        a = values[(values["season"] == s) & (values["week"] == w)].set_index(key)[NAMES]
        b = v2[(v2["season"] == s) & (v2["week"] == w)].set_index(key)[NAMES].reindex(a.index)
        gap = (a - b).abs().to_numpy()
        worst = max(worst, float(np.nanmax(gap)) if gap.size else 0.0)
        worst = float("inf") if np.isnan(gap).any() else worst
        rows += len(a)
        cutoffs.append([int(s), int(w), len(a), int((a != 0).any(axis=1).sum())])
    return {
        "cutoffs_season_week_teamgames_nonzero": cutoffs,
        "team_games": rows,
        "max_abs_difference": worst,
        "tolerance": u3.TOLERANCE,
    }


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = cfb_season_partitions(ROOT / "data" / "cfb", "pbp")
    last = max(x14.u1.SEASONS)
    raws, diag, skipped = [], {}, []
    for season in sorted(s for s in parts if s <= last):
        have = set(pq.ParquetFile(parts[season]).schema_arrow.names)
        if not set(PLAY_COLUMNS) <= have:
            skipped.append(int(season))
            continue
        r, d = season_tables(int(season), parts[season])
        raws.append(r)
        diag[str(season)] = d
    raw = pd.concat(raws, ignore_index=True)
    fits = season_fits(raw)
    values = to_date(raw, fits)
    check = asof_check(raw, values)
    pop = values[values["season"].isin(x14.u1.SEASONS)]
    corr = pop[NAMES].corr().round(3)
    report = {
        "family": FAMILY,
        "skipped_seasons_missing_columns": skipped,
        "fits_intercept_slope_by_season": {
            str(s): [round(a, 4), round(b, 4)] for s, (a, b) in fits.items()
        },
        "population_team_games": len(pop),
        "population_nonzero_share": {n: float(pop[n].ne(0).mean()) for n in NAMES},
        "population_sd": {n: float(pop[n].std(ddof=1)) for n in NAMES},
        "population_correlation": {n: corr[n].to_dict() for n in NAMES},
        "asof_check": check,
        "per_season": diag,
    }
    (OUT / "build.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("skipped", skipped)
    print("fits", report["fits_intercept_slope_by_season"])
    print("wp share", {s: round(d["share_wp_nonnull"], 3) for s, d in diag.items()})
    print("nonzero", report["population_nonzero_share"])
    print("sd", {k: round(v, 4) for k, v in report["population_sd"].items()})
    print("corr", corr.to_string())
    print("asof", check, flush=True)
    if not check["max_abs_difference"] <= u3.TOLERANCE:
        raise SystemExit("as-of check failed; screen not run")
    values.to_parquet(OUT / "team_values.parquet", index=False)


def joined(d, values):
    look = values.set_index(["game_id", "team"])
    gid = pd.to_numeric(d["game_id"]).astype("int64")
    side = {}
    for s in ("home", "away"):
        key = pd.MultiIndex.from_arrays([gid, pd.to_numeric(d[f"{s}_id"]).astype("int64")])
        side[s] = look[[*NAMES, "prior_games"]].reindex(key).reset_index(drop=True)
    both = side["home"]["prior_games"].notna() & side["away"]["prior_games"].notna()
    out = pd.DataFrame(index=d.index)
    for n in NAMES:
        diff = side["home"][n].to_numpy() - side["away"][n].to_numpy()
        out[n] = np.where(both.to_numpy(), diff, 0.0)
    return out, int(both.sum())


def cells(m36, name, res, seasons, draws):
    spec = CANDIDATES[name]
    held = res["held_out"]
    early = res["halves"]["earlier"]["coefficient_per_sd"]
    late = res["halves"]["later"]["coefficient_per_sd"]
    survived = "yes" if res["survivor"] else "no"
    verdict = (
        "It stood out from chance and earns one declared pro-football check."
        if res["survivor"]
        else "No effect large enough for this test to see showed up."
    )
    out = []
    for k, units in zip(m36.SHORT, m36.METRICS, strict=True):
        out.append(
            {
                "name": f"xlg14_composition_cfb_{name}_{units}",
                "description": (
                    f"College-first production-composition family, {name}: {spec['what']} "
                    "Team value pooled per play over strictly earlier weeks of the same season, "
                    "home minus away; one offset-logistic coefficient on a leave-one-season-out "
                    f"ridge cover probability with the opener as input and grade, {len(seasons)} "
                    f"seasons; declared sign {spec['hypothesis']}. Changed picks "
                    f"{held['changed']}, record {held['arm_right']}-{held['ref_right']}; "
                    f"coefficient per SD {res['coefficient_per_sd']:+.4f}; earlier and later "
                    f"season-half coefficients {early:+.4f} and {late:+.4f}; this metric sits at "
                    f"percentile {res['null_percentile'][k]:.1f} of {draws} within-season "
                    "shuffles of the same column; in-sample minus held-out "
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
                    "No resolved wrong sign for the declared persistence mechanism, split-half "
                    "reliability of the trait not measured, and no positive control bounds an "
                    "effect this small: the college add-one harness shows about .08 to .11 "
                    f"logit per SD four times in five ({POWER_LANE})."
                ),
                "notes": (
                    f"Family {FAMILY}: 4 candidates x 3 metrics = 12 recorded cells; 4 gating "
                    "reads (log-loss percentile in the matched null), 8 season-half "
                    "coefficients and 2 control reads; survivor rule declared in the lane "
                    f"before any outcome was joined; survivor: {survived}."
                ),
                "interval_low": round(held[k]["low"], 6),
                "interval_high": round(held[k]["high"], 6),
                "probability_positive": round(held[k]["pp"], 4),
                "sample_games": int(res["games"]),
                "sample_blocks": len(seasons),
                "plain_summary": (
                    f"Checks whether college teams that {spec['plain']} go on to beat the "
                    f"opening spread more or less often. {verdict}"
                ),
            }
        )
    return out


def screen():
    started = time.time()
    u3.STREAM_LUCK = STREAM
    hp = u3.load_power()
    m36 = hp.M36
    d = hp.x14.cfb_frame()
    values = pd.read_parquet(OUT / "team_values.parquet")
    terms, both = joined(d, values)
    frame = pd.DataFrame(
        {
            "game_id": d["game_id"].to_numpy(),
            "season": d["season"].to_numpy(int),
            "home_covered": d["y"].to_numpy(float),
            m36.SERVED_COLUMN: expit(d["logit_p"].to_numpy(float)),
        }
    )
    for n in NAMES:
        frame[n] = terms[n].to_numpy(float)
    seasons = sorted(int(s) for s in frame["season"].unique())
    out = {
        "family": FAMILY,
        "games": len(frame),
        "games_with_both_teams_in_play_by_play": both,
        "seasons": seasons,
        "draws": hp.DRAWS,
        "seed": [hp.SEED, STREAM],
        "candidates": {},
    }
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
            f"pct {res['null_percentile']['ll']:.1f}; brier {h['brier']['effect']:+.6f} "
            f"pct {res['null_percentile']['brier']:.1f}; null ll mean "
            f"{res['null']['ll']['mean']:+.6f} [{res['null']['ll']['q05']:+.6f},"
            f"{res['null']['ll']['q95']:+.6f}]; in-sample ll {res['in_sample']['ll']:+.6f} gap "
            f"{res['in_sample_minus_held_out']['ll']:+.6f}; coef {res['coefficient_per_sd']:+.4f} "
            f"folds {res['folds_with_full_sign']}/{len(seasons)}; halves "
            f"{res['halves']['earlier']['coefficient_per_sd']:+.4f} / "
            f"{res['halves']['later']['coefficient_per_sd']:+.4f}; survivor {res['survivor']}"
        )
        print(lines[-1], f"{time.time() - started:.0f}s", flush=True)
    out["survivors"] = [n for n in NAMES if out["candidates"][n]["survivor"]]
    c = out["controls"]
    lines.append(
        "controls: "
        + "; ".join(f"{k} acc {v['accuracy']:.3f} ll {v['log_loss']:.6f}" for k, v in c.items())
    )
    lines.append(f"survivors: {out['survivors']}; both-team games {both} of {len(frame)}")
    print(lines[-2], lines[-1], sep="\n", flush=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "weak_signals_batch.json").write_text(
        json.dumps({"cells": batch}, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    {"build": build, "screen": screen}[sys.argv[1]]()
