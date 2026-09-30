import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats import clv
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.margin import (
    _MARGIN_PROFILE_FEATURE_SETS,
    _PROFILE_SUPPRESSED_MISSING_INDICATORS,
    make_margin_estimator,
    margin_feature_columns,
    resolve_feature_groups,
)
from nfl_ats.modeling import regular_season_rows

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "game_features_weak_stack.parquet"
MARKET_ROOT = ROOT / "data" / "market" / "raw"
BASELINE_DIR = ROOT / "artifacts" / "opener_evaluation" / "20260929T192743Z"
OUT_DIR = ROOT / "artifacts" / "mod23_unit1"
PROFILE = "weak_stack"
SET_NAME = _MARGIN_PROFILE_FEATURE_SETS[PROFILE][1]
BASE_ALPHA = 10.0
ALPHA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)
SCORE_SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
INNER_SEASONS = 4
MIN_RELATIVE_GAIN = 0.0005
PINNED_FAMILY = "market"
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 20260930
BASE_CONFIG = {
    "feature_profile": PROFILE,
    "regressor": "ridge",
    "target": "market_residual",
    "probability_method": "gaussian_median",
    "calibration_method": "none",
}
FULL_COLUMNS = tuple(margin_feature_columns("market_residual", PROFILE))
COLUMN_FAMILY = dict(zip(FULL_COLUMNS, resolve_feature_groups(FULL_COLUMNS), strict=True))
FAMILIES = sorted({f for f in COLUMN_FAMILY.values() if f != PINNED_FAMILY})


def columns_for(dropped):
    return tuple(c for c in FULL_COLUMNS if COLUMN_FAMILY[c] not in dropped)


def config_key(alpha, dropped):
    return json.dumps({"alpha": alpha, "dropped": sorted(dropped)}, sort_keys=True)


def inner_mse(frame, columns, alpha, seasons_scored):
    total = 0.0
    count = 0
    for season in seasons_scored:
        train = frame.loc[frame["season"].lt(season)]
        test = frame.loc[frame["season"].eq(season)]
        estimator = make_margin_estimator(
            "ridge",
            42,
            ridge_alpha=alpha,
            suppressed_indicator_columns=_PROFILE_SUPPRESSED_MISSING_INDICATORS.get(PROFILE, ()),
        )
        estimator.fit(train.loc[:, list(columns)], train["ats_margin"])
        pred = np.asarray(estimator.predict(test.loc[:, list(columns)]), dtype=float)
        err = test["ats_margin"].to_numpy(dtype=float) - pred
        total += float(np.sum(err**2))
        count += len(err)
    return total / count


def inner_accuracy(frame, columns, alpha, seasons_scored):
    hits = 0
    count = 0
    for season in seasons_scored:
        train = frame.loc[frame["season"].lt(season)]
        test = frame.loc[frame["season"].eq(season)]
        estimator = make_margin_estimator(
            "ridge",
            42,
            ridge_alpha=alpha,
            suppressed_indicator_columns=_PROFILE_SUPPRESSED_MISSING_INDICATORS.get(PROFILE, ()),
        )
        estimator.fit(train.loc[:, list(columns)], train["ats_margin"])
        pred = np.asarray(estimator.predict(test.loc[:, list(columns)]), dtype=float)
        actual = test["ats_margin"].to_numpy(dtype=float)
        keep = actual != 0
        hits += int(np.sum((pred[keep] > 0) == (actual[keep] > 0)))
        count += int(keep.sum())
    return hits / count


def select_alpha_only(frame, season):
    inner = list(range(season - INNER_SEASONS, season))
    scores = {a: inner_mse(frame, FULL_COLUMNS, a, inner) for a in ALPHA_GRID}
    best = min(ALPHA_GRID, key=lambda a: (scores[a], abs(a - BASE_ALPHA)))
    return {"alpha": best, "dropped": [], "inner_mse": scores[best], "looks": len(ALPHA_GRID)}


def select_greedy(frame, season):
    inner = list(range(season - INNER_SEASONS, season))
    looks = 0

    def best_alpha(dropped):
        nonlocal looks
        cols = columns_for(dropped)
        scores = {a: inner_mse(frame, cols, a, inner) for a in ALPHA_GRID}
        looks += len(ALPHA_GRID)
        a = min(ALPHA_GRID, key=lambda x: (scores[x], abs(x - BASE_ALPHA)))
        return a, scores[a]

    dropped = frozenset()
    alpha, current = best_alpha(dropped)
    while True:
        candidates = []
        for fam in FAMILIES:
            if fam in dropped:
                continue
            trial = dropped | {fam}
            if not columns_for(trial):
                continue
            a, s = best_alpha(trial)
            candidates.append((s, fam, a))
        if not candidates:
            break
        s, fam, a = min(candidates)
        if s < current * (1.0 - MIN_RELATIVE_GAIN):
            dropped = dropped | {fam}
            alpha, current = a, s
        else:
            break
    return {"alpha": alpha, "dropped": sorted(dropped), "inner_mse": current, "looks": looks}


def run_config(features, alpha, dropped):
    columns = columns_for(dropped)
    original = FEATURE_SETS[SET_NAME]
    FEATURE_SETS[SET_NAME] = columns
    try:
        config = {**BASE_CONFIG, "ridge_alpha": alpha, "mod23_variant": config_key(alpha, dropped)}
        return clv.opener_pick_evaluation(
            MARKET_ROOT,
            features,
            active_model_config=config,
            min_train_games=DEFAULT_MIN_TRAIN_GAMES,
        )
    finally:
        FEATURE_SETS[SET_NAME] = original


def decisive(scored):
    frame = scored.loc[scored["correct_at_open_probability_rule"].notna()].copy()
    frame["y"] = (frame["margin_vs_open"] > 0).astype(float)
    frame["p"] = frame["home_cover_probability_at_open"].clip(1e-6, 1 - 1e-6)
    frame["ll"] = -(frame["y"] * np.log(frame["p"]) + (1 - frame["y"]) * np.log(1 - frame["p"]))
    frame["brier"] = (frame["p"] - frame["y"]) ** 2
    return frame.set_index("game_id")


def bootstrap_diff(diff_frame):
    blocks = [g["d"].to_numpy() for _, g in diff_frame.groupby(["season", "week"])]
    sums = np.array([b.sum() for b in blocks])
    sizes = np.array([len(b) for b in blocks])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(blocks), len(blocks))
        draws.append(100.0 * sums[idx].sum() / sizes[idx].sum())
    draws = np.array(draws)
    return {
        "low": float(np.percentile(draws, 2.5)),
        "high": float(np.percentile(draws, 97.5)),
        "probability_positive": float((draws > 0).mean()),
    }


def summarize(name, arm, base):
    joined = arm.join(base, lsuffix="", rsuffix="_b", how="inner")
    joined["d"] = (
        joined["correct_at_open_probability_rule"] - joined["correct_at_open_probability_rule_b"]
    )
    wins = int(arm["correct_at_open_probability_rule"].sum())
    n = len(arm)
    result = {
        "arm": name,
        "record": f"{wins}-{n - wins}",
        "accuracy": wins / n,
        "diff_points": 100.0 * float(joined["d"].mean()),
        "bootstrap": bootstrap_diff(joined),
        "log_loss": float(arm["ll"].mean()),
        "log_loss_base": float(base["ll"].mean()),
        "log_loss_even": float(np.log(2)),
        "brier": float(arm["brier"].mean()),
        "brier_base": float(base["brier"].mean()),
        "brier_even": 0.25,
        "per_season": {},
    }
    for season, g in joined.groupby("season"):
        result["per_season"][int(season)] = {
            "n": len(g),
            "acc": float(g["correct_at_open_probability_rule"].mean()),
            "acc_base": float(g["correct_at_open_probability_rule_b"].mean()),
            "diff_points": 100.0 * float(g["d"].mean()),
        }
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    features = pd.read_parquet(FEATURES)
    sha = hashlib.sha256(FEATURES.read_bytes()).hexdigest()
    print("feature sha", sha)
    frame = regular_season_rows(features)
    frame = frame.loc[frame["ats_margin"].notna() & frame["result"].notna()].copy()
    selection_path = OUT_DIR / "selection.json"
    if selection_path.exists() and stage != "select":
        selection = json.loads(selection_path.read_text())
    else:
        selection = {"greedy": {}, "alpha_only": {}}
        for season in SCORE_SEASONS:
            selection["greedy"][str(season)] = select_greedy(frame, season)
            selection["alpha_only"][str(season)] = select_alpha_only(frame, season)
            print(
                season,
                selection["greedy"][str(season)],
                selection["alpha_only"][str(season)],
                flush=True,
            )
        selection_path.write_text(json.dumps(selection, indent=2))
    if stage == "select":
        return

    baseline_scored = run_config(features, BASE_ALPHA, frozenset())
    base = decisive(baseline_scored)
    artifact = pd.read_parquet(BASELINE_DIR / "per_game.parquet")
    art = artifact.set_index("game_id")["correct_at_open_probability_rule"].dropna()
    mine = base["correct_at_open_probability_rule"]
    common = art.index.intersection(mine.index)
    print(
        "baseline reproduction: mine",
        int(mine.sum()),
        len(mine),
        "artifact",
        int(art.sum()),
        len(art),
        "identical rows",
        bool((art.loc[common] == mine.loc[common]).all()),
        "common",
        len(common),
        flush=True,
    )
    if int(mine.sum()) != int(art.sum()) or len(mine) != len(art):
        print("REPRODUCTION FAILED")
        return

    cache = {config_key(BASE_ALPHA, frozenset()): base}
    outputs = {}
    for arm_name in ("greedy", "alpha_only"):
        pieces = []
        for season in SCORE_SEASONS:
            choice = selection[arm_name][str(season)]
            key = config_key(choice["alpha"], frozenset(choice["dropped"]))
            if key not in cache:
                cache[key] = decisive(
                    run_config(features, choice["alpha"], frozenset(choice["dropped"]))
                )
                print("ran", key, flush=True)
            pieces.append(cache[key].loc[cache[key]["season"].eq(season)])
        arm = pd.concat(pieces)
        outputs[arm_name] = (arm, summarize(arm_name, arm, base))
        arm.to_parquet(OUT_DIR / f"{arm_name}_per_game.parquet")

    gaps = {}
    for arm_name in ("greedy", "alpha_only"):
        arm = outputs[arm_name][0]
        rows = {}
        for season in SCORE_SEASONS:
            c = selection[arm_name][str(season)]
            sub = arm.loc[arm["season"].eq(season)]
            b = base.loc[base["season"].eq(season)]
            inner = list(range(season - INNER_SEASONS, season))
            base_inner = inner_mse(frame, FULL_COLUMNS, BASE_ALPHA, inner)
            outer_mse = float(np.mean((sub["margin_vs_open"] - sub["residual_at_open"]) ** 2))
            outer_base = float(np.mean((b["margin_vs_open"] - b["residual_at_open"]) ** 2))
            rows[season] = {
                "inner_mse_chosen": c["inner_mse"],
                "inner_mse_base": base_inner,
                "inner_gain": base_inner - c["inner_mse"],
                "outer_mse_chosen": outer_mse,
                "outer_mse_base": outer_base,
                "outer_gain": outer_base - outer_mse,
            }
        gaps[arm_name] = rows
    report = {
        "selection": selection,
        "summaries": {k: v[1] for k, v in outputs.items()},
        "gaps": gaps,
        "distinct_configs_run": len(cache),
        "looks": {
            "inner_greedy": sum(v["looks"] for v in selection["greedy"].values()),
            "inner_alpha_only": sum(v["looks"] for v in selection["alpha_only"].values()),
            "outer_arms": 2,
        },
        "declared": {
            "alpha_grid": ALPHA_GRID,
            "inner_seasons": INNER_SEASONS,
            "min_relative_gain": MIN_RELATIVE_GAIN,
            "families": FAMILIES,
            "pinned": PINNED_FAMILY,
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


main()
