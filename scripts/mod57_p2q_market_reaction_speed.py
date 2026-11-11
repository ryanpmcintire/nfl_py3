from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mod57_p2c_uncorrected_error_model import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED
from mod57_p2o_how_wrong_is_the_line import (
    MIN_PRIOR_GAMES,
    THIRD_EDGES,
    THIRD_LABELS,
    design,
    fit_coef,
    record,
    slope_boot,
    team_index,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/mod57_p2q"
RATINGS = ROOT / "artifacts/mod57_p2o/per_game_ratings_LEAK_NOT_SERVABLE.parquet"
FAMILY = "mod57_how_wrong_is_the_line"
FIRST_TEST = 2015
LAST_SEASON = 2025
OPENER_FIRST = 2020
EARLY = (2, 6)
CHECK_GAMES = 10
CHECK_SEED = 20261112
QUANTITIES = ("D1", "D2", "D3")


def pregame_values(reg, index):
    n = len(index)
    d1 = pd.Series(np.nan, index=reg["game_id"].to_numpy())
    fallback = 0
    team_games = 0
    seasons = sorted(reg["season"].unique())
    final = {}
    for s in seasons:
        sdf = reg.loc[reg["season"] == s]
        if (sdf["close_line"].notna()).all() and len(sdf):
            final[s] = fit_coef(design(sdf, index), sdf["close_line"].to_numpy(float))[:n]
    for s, sdf in reg.groupby("season"):
        start = final.get(s - 1)
        for day, target in sdf.groupby("gameday"):
            train = sdf.loc[sdf["gameday"] < day]
            counts = np.zeros(n)
            now = None
            if len(train):
                counts = np.bincount(train["home_team"].map(index), minlength=n) + np.bincount(
                    train["away_team"].map(index), minlength=n
                )
                if (counts >= MIN_PRIOR_GAMES).any():
                    now = fit_coef(design(train, index), train["close_line"].to_numpy(float))[:n]
            for _, row in target.iterrows():
                drift = []
                for team in (row["home_team"], row["away_team"]):
                    i = index[team]
                    team_games += 1
                    if start is None:
                        drift.append(np.nan)
                    elif counts[i] < MIN_PRIOR_GAMES or now is None:
                        fallback += 1
                        drift.append(0.0)
                    else:
                        drift.append(float(now[i] - start[i]))
                d1[row["game_id"]] = drift[0] - drift[1]
    return d1, fallback, team_games


def surprise_values(reg):
    parts = []
    for side, sign in (("home", 1.0), ("away", -1.0)):
        part = reg[["game_id", "season", "gameday", f"{side}_team"]].rename(
            columns={f"{side}_team": "team"}
        )
        part["side"] = side
        part["res"] = sign * (reg["margin"] - reg["close_line"])
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    long = long.sort_values(["season", "team", "gameday", "game_id"]).reset_index(drop=True)
    group = long.groupby(["season", "team"])
    prior_sum = group["res"].cumsum() - long["res"]
    prior_n = group.cumcount()
    long["mean_to_date"] = (prior_sum / prior_n.replace(0, np.nan)).to_numpy()
    home = long.loc[long["side"] == "home"].set_index("game_id")["mean_to_date"]
    away = long.loc[long["side"] == "away"].set_index("game_id")["mean_to_date"]
    return home - away.reindex(home.index)


def quantities(reg, index):
    d1, fallback, team_games = pregame_values(reg, index)
    d2 = surprise_values(reg)
    out = reg[["game_id", "season", "week"]].copy()
    out["D1"] = d1.reindex(out["game_id"]).to_numpy()
    out["D2"] = d2.reindex(out["game_id"]).to_numpy()
    early = out["week"].between(*EARLY)
    out["D3"] = out["D2"].where(early, np.where(out["D2"].notna(), 0.0, np.nan))
    return out, fallback, team_games


def truncation_check(reg, index, full):
    pool = reg.loc[reg["season"] > reg["season"].min()]
    sample = pool.loc[pool["week"] > 1].sample(CHECK_GAMES, random_state=CHECK_SEED)
    rows = []
    for _, g in sample.iterrows():
        keep = reg["gameday"] < g["gameday"]
        same = (reg["gameday"] == g["gameday"]) & (reg["season"] == g["season"])
        cut = pd.concat(
            [reg.loc[keep], reg.loc[same].assign(margin=0.0, close_line=0.0)]
        ).reset_index(drop=True)
        again, _, _ = quantities(cut, index)
        a = again.set_index("game_id").loc[g["game_id"]]
        f = full.set_index("game_id").loc[g["game_id"]]
        rows.append(
            {
                "game_id": g["game_id"],
                "d1_diff": float(abs(a["D1"] - f["D1"])),
                "d2_diff": float(abs(a["D2"] - f["D2"])),
            }
        )
    return rows


def thirds(frame):
    cut = pd.cut(frame["week"], THIRD_EDGES, labels=THIRD_LABELS).astype(str)
    return cut.to_numpy()


def slope_cells(frame, rng):
    out = {}
    label = frame["R_H"] - frame["close_line"]
    real = frame["margin"] - frame["close_line"]
    third = thirds(frame)
    for q in QUANTITIES:
        for outcome, y in (("hindsight_gap", label), ("real_residual", real)):
            for scope in ("overall", *THIRD_LABELS):
                mask = (frame[q].notna() & y.notna()).to_numpy()
                if scope != "overall":
                    mask = mask & (third == scope)
                x = frame.loc[mask, q].to_numpy(float)
                if len(x) < 3 or np.ptp(x) == 0 or x.std() == 0:
                    out[f"{q}_{outcome}_{scope}"] = None
                    continue
                block = pd.factorize(frame.loc[mask, "season"])[0]
                s = slope_boot(x, y[mask].to_numpy(float), block, rng)
                s["games"] = int(mask.sum())
                out[f"{q}_{outcome}_{scope}"] = s
    return out


def walk_forward(frame, q, line, rng):
    real = frame["margin"] - frame["close_line"]
    rows = []
    for test in range(FIRST_TEST, LAST_SEASON + 1):
        train = frame.loc[(frame["season"] < test) & frame[q].notna() & real.notna()]
        x = train[q].to_numpy(float)
        y = real.loc[train.index].to_numpy(float)
        sign = float(np.sign(np.cov(x, y)[0, 1]))
        te = frame.loc[(frame["season"] == test) & frame[q].notna() & frame[line].notna()]
        te = te.loc[(te["margin"] != te[line]) & (te[q] != 0)]
        pick_home = (sign * te[q]) > 0
        hit = pick_home == (te["margin"] > te[line])
        rows.append(
            pd.DataFrame(
                {"season": test, "week": te["week"], "hit": hit.astype(float), "sign": sign}
            )
        )
    allrows = pd.concat(rows, ignore_index=True)
    result = {"signs_by_test_season": {int(r.season): r.sign for r in rows_unique(allrows)}}
    for scope, mask in (
        ("overall", np.ones(len(allrows), bool)),
        ("weeks_2_6", allrows["week"].between(*EARLY).to_numpy()),
    ):
        sub = allrows.loc[mask]
        hit = sub["hit"].to_numpy()
        if len(hit) == 0:
            result[scope] = None
            continue
        block = pd.factorize(sub["season"])[0]
        n_blocks = int(block.max()) + 1
        sums = np.bincount(block, weights=hit - 0.5, minlength=n_blocks)
        sizes = np.bincount(block, minlength=n_blocks)
        idx = rng.integers(0, n_blocks, (BOOTSTRAP_SAMPLES, n_blocks))
        draws = 100.0 * sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
        entry = record(hit.astype(bool))
        entry.update(
            {
                "points_over_50": float(100.0 * (hit.mean() - 0.5)),
                "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
                "probability_positive": float((draws > 0).mean()),
            }
        )
        result[scope] = entry
    return result


def rows_unique(allrows):
    return list(allrows.drop_duplicates("season").itertuples())


def cells(slopes, walk):
    out = []
    for name, s in slopes.items():
        if s is None:
            continue
        out.append(
            {
                "name": f"mod57_p2q_{name}_slope",
                "description": (
                    f"Slope of {name.replace('_', ' ')} on the pregame market quantity, points per "
                    "point; hindsight gap is a leak used only as a descriptive outcome; "
                    "season-block bootstrap"
                ),
                "effect": s["mean"],
                "effect_units": "calibration_slope",
                "interval_low": s["ci95"][0],
                "interval_high": s["ci95"][1],
                "probability_positive": s["probability_positive"],
                "season_start": 2009,
                "season_end": LAST_SEASON,
            }
        )
    for name, w in walk.items():
        for scope in ("overall", "weeks_2_6"):
            e = w[scope]
            if e is None:
                continue
            opener = "open" in name
            out.append(
                {
                    "name": f"mod57_p2q_walkforward_{name}_{scope}_accuracy_points",
                    "description": (
                        f"Walk-forward sign-following pick accuracy minus 50, {name}, {scope}, "
                        "decisive games, season-block bootstrap"
                    ),
                    "effect": e["points_over_50"],
                    "effect_units": "accuracy_points",
                    "interval_low": e["ci95"][0],
                    "interval_high": e["ci95"][1],
                    "probability_positive": e["probability_positive"],
                    "season_start": OPENER_FIRST if opener else FIRST_TEST,
                    "season_end": LAST_SEASON,
                }
            )
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reg = pd.read_parquet(RATINGS)
    reg["gameday"] = pd.to_datetime(reg["gameday"])
    reg = reg.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    index = team_index(reg)
    q, fallback, team_games = quantities(reg, index)
    trunc = truncation_check(reg, index, q)
    frame = reg.merge(q[["game_id", *QUANTITIES]], on="game_id")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    slopes = slope_cells(frame, rng)
    walk = {}
    for qn in QUANTITIES:
        for line in ("close_line", "open_line"):
            walk[f"{qn}_{line.split('_')[0]}"] = walk_forward(frame, qn, line, rng)
    results = {
        "label": "hindsight gap is a LEAK (future games), descriptive outcome only",
        "team_games": team_games,
        "team_games_using_prior_season_rating": fallback,
        "truncation_check": trunc,
        "truncation_max_abs_diff": max(max(r["d1_diff"], r["d2_diff"]) for r in trunc),
        "games_with_D1": int(frame["D1"].notna().sum()),
        "games_with_D2": int(frame["D2"].notna().sum()),
        "slopes": slopes,
        "walk_forward": walk,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    batch = {
        "league": "nfl",
        "classification": "unresolved_below_power",
        "source": "artifacts/mod57_p2q/results.json",
        "category": "market",
        "family": FAMILY,
        "notes": (
            "Market rating drift and surprise to date against the hindsight gap (leak, outcome "
            "only) and against real results; walk-forward sign for picks."
        ),
        "classification_evidence": (
            "Season-block bootstrap, probability_positive reported; no refuted mechanism and "
            "no closing ground."
        ),
        "plain_summary": (
            "Whether the betting line reacts too slowly or too fast to what a season has shown."
        ),
        "cells": cells(slopes, walk),
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print("team games", team_games, "prior-season fallback", fallback)
    print("truncation max abs diff", results["truncation_max_abs_diff"], len(trunc))
    print("games D1", results["games_with_D1"], "D2", results["games_with_D2"])
    for k, s in slopes.items():
        if s is None:
            print(k, "undefined")
        else:
            print(
                f"{k} n {s['games']} {s['mean']:+.4f} "
                f"[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}] P+ {s['probability_positive']:.3f}"
            )
    for k, w in walk.items():
        print(k, "signs", w["signs_by_test_season"])
        for scope in ("overall", "weeks_2_6"):
            e = w[scope]
            if e:
                print(
                    f"  {scope} {e['record']} {e['accuracy']:.4f} "
                    f"{e['points_over_50']:+.2f} [{e['ci95'][0]:+.2f},{e['ci95'][1]:+.2f}] "
                    f"P+ {e['probability_positive']:.3f}"
                )


if __name__ == "__main__":
    main()
