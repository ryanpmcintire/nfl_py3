from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mod57_p2c_uncorrected_error_model import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED
from mod57_p2o_how_wrong_is_the_line import (
    design,
    fit_coef,
    record,
    slope_boot,
    team_index,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/mod57_p2s"
RATINGS = ROOT / "artifacts/mod57_p2o/per_game_ratings_LEAK_NOT_SERVABLE.parquet"
SERVED = ROOT / "artifacts/pick_probability/20261111T011859Z/per_game.parquet"
SHARD = ROOT / "registry/weak_signals/mod57_how_wrong_is_the_line.json"
FAMILY = "mod57_how_wrong_is_the_line"
FIRST_SEASON = 2010
FIRST_TEST = 2015
LAST_SEASON = 2025
OPENER_FIRST = 2020
WINDOWS = {"weeks_1_4": (1, 4), "weeks_1_6": (1, 6)}
WEEK_GROUPS = {
    "week_1": (1, 1),
    "weeks_2_6": (2, 6),
    "weeks_7_12": (7, 12),
    "weeks_13_18": (13, 18),
}
CHECK_GAMES = 10
CHECK_SEED = 20261112
QUANTITIES = ("A0", "A1")


def quantities(reg, index):
    n = len(index)
    final = {}
    for s, sdf in reg.groupby("season"):
        coef = fit_coef(design(sdf, index), sdf["close_line"].to_numpy(float))
        final[s] = (coef[:n], float(coef[n]))
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "season": reg["season"],
                    "team": reg[f"{side}_team"],
                    "res": sign * (reg["margin"] - reg["close_line"]),
                }
            )
            for side, sign in (("home", 1.0), ("away", -1.0))
        ]
    )
    prior_mean = long.groupby(["season", "team"])["res"].mean()
    hi = reg["home_team"].map(index).to_numpy()
    ai = reg["away_team"].map(index).to_numpy()
    out = reg[["game_id", "season", "week"]].copy()
    a0c = np.full(len(reg), np.nan)
    a0o = np.full(len(reg), np.nan)
    a1 = np.full(len(reg), np.nan)
    for s in sorted(reg["season"].unique()):
        if s - 1 not in final:
            continue
        r, h = final[s - 1]
        ids = np.flatnonzero((reg["season"] == s).to_numpy())
        diff = r[hi[ids]] - r[ai[ids]]
        a0c[ids] = reg["close_line"].to_numpy(float)[ids] - h - diff
        a0o[ids] = reg["open_line"].to_numpy(float)[ids] - h - diff
        pm = prior_mean.loc[s - 1]
        home = reg["home_team"].to_numpy()[ids]
        away = reg["away_team"].to_numpy()[ids]
        a1[ids] = pm.reindex(home).to_numpy() - pm.reindex(away).to_numpy()
    out["A0"] = a0c
    out["A0_open"] = a0o
    out["A1"] = a1
    return out


def truncation_check(reg, index, full):
    pool = reg.loc[reg["season"] > reg["season"].min()]
    sample = pool.sample(CHECK_GAMES, random_state=CHECK_SEED)
    rows = []
    for _, g in sample.iterrows():
        cut = reg.copy()
        late = (cut["gameday"] >= g["gameday"]) & (cut["game_id"] != g["game_id"])
        cut.loc[late, ["margin", "close_line", "open_line"]] = 0.0
        cut.loc[cut["game_id"] == g["game_id"], "margin"] = 0.0
        again = quantities(cut, index).set_index("game_id").loc[g["game_id"]]
        f = full.set_index("game_id").loc[g["game_id"]]
        rows.append(
            {
                "game_id": g["game_id"],
                "a0_diff": float(abs(again["A0"] - f["A0"])),
                "a1_diff": float(abs(again["A1"] - f["A1"])),
            }
        )
    return rows


def block_ci(hit, block, rng):
    n_blocks = int(block.max()) + 1
    sums = np.bincount(block, weights=hit - 0.5, minlength=n_blocks)
    sizes = np.bincount(block, minlength=n_blocks)
    idx = rng.integers(0, n_blocks, (BOOTSTRAP_SAMPLES, n_blocks))
    draws = 100.0 * sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return {
        "points_over_50": float(100.0 * (hit.mean() - 0.5)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_positive": float((draws > 0).mean()),
    }


def read_hits(hit, season, block, rng):
    entry = record(hit.astype(bool))
    entry.update(block_ci(hit, block, rng))
    return entry


def part1(rng):
    d = pd.read_parquet(SERVED)
    d = d.loc[d["out_of_season_home_probability"].notna()].copy()
    cover = d["margin_vs_open"]
    decisive = cover != 0
    served = ((d["out_of_season_home_probability"] > 0.5) == (cover > 0)).astype(float)
    model = d["model_correct"].astype(float)
    moved = d["open_move"] != 0
    move = ((d["open_move"] > 0) == (cover > 0)).astype(float)
    arms = {
        "served": served.where(decisive),
        "model_alone": model.where(decisive),
        "line_move_side": move.where(decisive & moved),
    }
    d["block"] = d["season"].astype(str) + "_" + d["week"].astype(str)
    out = {"by_week": {}, "by_group": {}, "weeks_2_6_by_season": {}}

    def cell(mask):
        res = {}
        for arm, y in arms.items():
            sub = mask & y.notna()
            hit = y[sub].to_numpy()
            block = pd.factorize(d.loc[sub, "block"])[0]
            res[arm] = read_hits(hit, None, block, rng) if len(hit) else None
        return res

    for w in range(1, 19):
        out["by_week"][w] = cell(d["week"] == w)
    for name, (lo, hi) in WEEK_GROUPS.items():
        out["by_group"][name] = cell(d["week"].between(lo, hi))
    early = d["week"].between(2, 6)
    for s in sorted(d["season"].unique()):
        out["weeks_2_6_by_season"][int(s)] = cell(early & (d["season"] == s))
    out["games_total"] = len(d)
    return out


def slope_cells(frame, rng):
    out = {}
    outcomes = {
        "hindsight_gap": frame["R_H"] - frame["close_line"],
        "real_residual": frame["margin"] - frame["close_line"],
    }
    for q in QUANTITIES:
        for wname, (lo, hi) in WINDOWS.items():
            for oname, y in outcomes.items():
                mask = (frame[q].notna() & y.notna() & frame["week"].between(lo, hi)).to_numpy()
                x = frame.loc[mask, q].to_numpy(float)
                block = pd.factorize(frame.loc[mask, "season"])[0]
                s = slope_boot(x, y[mask].to_numpy(float), block, rng)
                s["games"] = int(mask.sum())
                out[f"{q}_{oname}_{wname}"] = s
    return out


def walk_forward(frame, q, line, lo, hi, rng):
    qcol = "A0_open" if (q == "A0" and line == "open_line") else q
    real = frame["margin"] - frame["close_line"]
    in_window = frame["week"].between(lo, hi)
    rows = []
    signs = {}
    first = OPENER_FIRST if line == "open_line" else FIRST_TEST
    for test in range(first, LAST_SEASON + 1):
        train = frame.loc[(frame["season"] < test) & frame[q].notna() & real.notna() & in_window]
        sign = float(
            np.sign(np.cov(train[q].to_numpy(float), real.loc[train.index].to_numpy(float))[0, 1])
        )
        signs[test] = sign
        te = frame.loc[(frame["season"] == test) & in_window]
        te = te.loc[te[qcol].notna() & te[line].notna()]
        te = te.loc[(te["margin"] != te[line]) & (te[qcol] != 0)]
        pick_home = (sign * te[qcol]) > 0
        hit = pick_home == (te["margin"] > te[line])
        rows.append(pd.DataFrame({"season": test, "hit": hit.astype(float)}))
    allrows = pd.concat(rows, ignore_index=True)
    hit = allrows["hit"].to_numpy()
    block = pd.factorize(allrows["season"])[0]
    result = {"signs_by_test_season": signs, "overall": read_hits(hit, None, block, rng)}
    result["by_season"] = {
        int(s): record(g["hit"].to_numpy().astype(bool)) for s, g in allrows.groupby("season")
    }
    return result


def cells(slopes, walk):
    out = []
    for name, s in slopes.items():
        out.append(
            {
                "name": f"mod57_p2s_{name}_slope",
                "description": (
                    f"Slope of {name.replace('_', ' ')} on the market offseason quantity, points "
                    "per point; hindsight gap is a leak used only as a descriptive outcome; "
                    "season-block bootstrap"
                ),
                "effect": s["mean"],
                "effect_units": "calibration_slope",
                "interval_low": s["ci95"][0],
                "interval_high": s["ci95"][1],
                "probability_positive": s["probability_positive"],
                "season_start": FIRST_SEASON,
                "season_end": LAST_SEASON,
            }
        )
    for name, w in walk.items():
        e = w["overall"]
        out.append(
            {
                "name": f"mod57_p2s_walkforward_{name}_accuracy_points",
                "description": (
                    f"Walk-forward sign-following pick accuracy minus 50, {name}, decisive games, "
                    "season-block bootstrap"
                ),
                "effect": e["points_over_50"],
                "effect_units": "accuracy_points",
                "interval_low": e["ci95"][0],
                "interval_high": e["ci95"][1],
                "probability_positive": e["probability_positive"],
                "season_start": OPENER_FIRST if "open" in name else FIRST_TEST,
                "season_end": LAST_SEASON,
            }
        )
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = len(json.loads(SHARD.read_text(encoding="utf-8"))["signals"])
    reg = pd.read_parquet(RATINGS)
    reg["gameday"] = pd.to_datetime(reg["gameday"])
    reg = reg.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    index = team_index(reg)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    p1 = part1(rng)
    q = quantities(reg, index)
    trunc = truncation_check(reg, index, q)
    frame = reg.merge(q[["game_id", "A0", "A0_open", "A1"]], on="game_id")
    frame = frame.loc[frame["season"] >= FIRST_SEASON]
    slopes = slope_cells(frame, rng)
    walk = {}
    for qn in QUANTITIES:
        for wname, (lo, hi) in WINDOWS.items():
            for line in ("close_line", "open_line"):
                walk[f"{qn}_{wname}_{line.split('_')[0]}"] = walk_forward(
                    frame, qn, line, lo, hi, rng
                )
    results = {
        "prior_shard_cells": prior,
        "truncation_check": trunc,
        "truncation_max_abs_diff": max(max(r["a0_diff"], r["a1_diff"]) for r in trunc),
        "part1": p1,
        "slopes": slopes,
        "walk_forward": walk,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    batch = {
        "league": "nfl",
        "classification": "unresolved_below_power",
        "source": "artifacts/mod57_p2s/results.json",
        "category": "market",
        "family": FAMILY,
        "notes": (
            "Market offseason re-rating and prior-season surprise against early-season results "
            "and the hindsight gap (leak, outcome only); walk-forward sign for picks."
        ),
        "classification_evidence": (
            "Season-block bootstrap, probability_positive reported; no refuted mechanism and "
            "no closing ground."
        ),
        "plain_summary": "Whether the line's offseason re-rating of a team is off.",
        "cells": cells(slopes, walk),
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print("prior shard cells", prior, "truncation max diff", results["truncation_max_abs_diff"])
    print("PART1 games", p1["games_total"])
    for sect in ("by_group", "by_week", "weeks_2_6_by_season"):
        for k, v in p1[sect].items():
            for arm, e in v.items():
                if e:
                    print(
                        f"{sect} {k} {arm} {e['record']} {e['accuracy']:.3f} "
                        f"[{e['ci95'][0]:+.1f},{e['ci95'][1]:+.1f}]pts "
                        f"P+ {e['probability_positive']:.2f}"
                    )
    for k, s in slopes.items():
        print(
            f"{k} n {s['games']} {s['mean']:+.4f} [{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}] "
            f"P+ {s['probability_positive']:.3f}"
        )
    for k, w in walk.items():
        e = w["overall"]
        print(
            k,
            e["record"],
            f"{e['accuracy']:.4f} {e['points_over_50']:+.2f} "
            f"[{e['ci95'][0]:+.2f},{e['ci95'][1]:+.2f}] P+ {e['probability_positive']:.3f}",
            "signs",
            set(w["signs_by_test_season"].values()),
        )
        print("   by season", {s: r["record"] for s, r in w["by_season"].items()})


if __name__ == "__main__":
    main()
