import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _repo_src() -> None:
    path = str(REPO / "src")
    if path not in sys.path:
        sys.path.insert(0, path)


BASE_FEATURES = (
    "model_logit",
    "composition_flag_sum",
    "market_move_toward_home",
    "market_move_available",
)
EXTRA_FEATURES = (
    "ats_streak_regress_flag",
    "post_ot_fatigue_flag",
    "division_dog_flag",
    "low_total_div_home_dog_flag",
    "sept_heat_home_flag",
    "den_home_flag",
    "spread_size_signed",
    "key7_distance_signed",
)
FULL_FEATURES = BASE_FEATURES + EXTRA_FEATURES
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260821
EPSILON = 1e-6


def fit_logit(design_matrix, target, ridge, iterations):
    beta = np.zeros(design_matrix.shape[1])
    eye = np.eye(design_matrix.shape[1])
    for _ in range(iterations):
        linear = np.clip(design_matrix @ beta, -35.0, 35.0)
        prob = 1.0 / (1.0 + np.exp(-linear))
        weight = np.clip(prob * (1.0 - prob), 1e-6, None)
        gradient = design_matrix.T @ (target - prob) - ridge * beta
        hessian = (design_matrix.T * weight) @ design_matrix + ridge * eye
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta = beta + step
    return beta


def design_matrix(frame, features, means, stds):
    columns = [np.ones(len(frame))]
    for name in features:
        columns.append(((frame[name].astype(float) - means[name]) / stds[name]).to_numpy())
    return np.column_stack(columns)


def standardisers(train, features):
    means = {name: float(train[name].mean()) for name in features}
    stds = {name: float(train[name].std(ddof=0)) or 1.0 for name in features}
    return means, stds


def natural_coefficients(beta, features, means, stds):
    natural = {name: float(beta[index + 1] / stds[name]) for index, name in enumerate(features)}
    intercept = float(beta[0])
    for index, name in enumerate(features):
        intercept -= float(beta[index + 1]) * means[name] / stds[name]
    natural["intercept"] = intercept
    return natural


def predict_proba(frame, features, beta, means, stds):
    linear = np.clip(design_matrix(frame, features, means, stds) @ beta, -35.0, 35.0)
    return np.asarray(1.0 / (1.0 + np.exp(-linear)), dtype=float)


def loso_fit(population, features, ridge, iterations):
    out = pd.Series(np.nan, index=population.index, dtype=float)
    fold_coefficients = {}
    seasons = sorted(int(value) for value in population["season"].unique())
    for held in seasons:
        train = population.loc[population["season"].ne(held)]
        test = population.loc[population["season"].eq(held)]
        if train.empty or test.empty:
            continue
        fold_means, fold_stds = standardisers(train, features)
        fold_beta = fit_logit(
            design_matrix(train, features, fold_means, fold_stds),
            train["home_covered"].astype(float).to_numpy(),
            ridge,
            iterations,
        )
        out.loc[test.index] = predict_proba(test, features, fold_beta, fold_means, fold_stds)
        fold_coefficients[str(held)] = natural_coefficients(
            fold_beta, features, fold_means, fold_stds
        )
    full_means, full_stds = standardisers(population, features)
    full_beta = fit_logit(
        design_matrix(population, features, full_means, full_stds),
        population["home_covered"].astype(float).to_numpy(),
        ridge,
        iterations,
    )
    in_sample = predict_proba(population, features, full_beta, full_means, full_stds)
    full_natural = natural_coefficients(full_beta, features, full_means, full_stds)
    return out, fold_coefficients, in_sample, full_natural


def week_block_bootstrap(frame, draws, seed):
    rng = np.random.default_rng(seed)
    blocks = sorted(frame["block"].unique())
    effects = np.empty(draws)
    for draw in range(draws):
        picked = rng.choice(len(blocks), size=len(blocks), replace=True)
        pooled = pd.concat([frame.loc[frame["block"].eq(blocks[index])] for index in picked])
        effects[draw] = pooled["paired_diff"].mean()
    low, high = np.quantile(effects, [0.025, 0.975])
    return float(low), float(high), float((effects > 0).mean())


def accuracy(probs, truth):
    picked_home = pd.Series(np.asarray(probs)).ge(0.5).to_numpy()
    return float((picked_home.astype(float) == truth.to_numpy(dtype=float)).mean())


def brier(probs, truth):
    return float(((np.asarray(probs) - truth.to_numpy(dtype=float)) ** 2).mean())


def logloss(probs, truth):
    clipped = np.clip(np.asarray(probs, dtype=float), EPSILON, 1.0 - EPSILON)
    truth_values = truth.to_numpy(dtype=float)
    return float(
        -(truth_values * np.log(clipped) + (1.0 - truth_values) * np.log(1.0 - clipped)).mean()
    )


def paired_summary(scored, column):
    frame = scored[["season", "week", column]].rename(columns={column: "paired_diff"}).copy()
    effect_points = float(frame["paired_diff"].mean() * 100.0)
    low, high, probability_positive = week_block_bootstrap(
        frame.assign(block=frame["season"].astype(str) + "-w" + frame["week"].astype(str)),
        BOOTSTRAP_DRAWS,
        BOOTSTRAP_SEED,
    )
    decisive = frame.loc[frame["paired_diff"].ne(0.0)]
    wins = int((decisive["paired_diff"] > 0).sum())
    losses = int((decisive["paired_diff"] < 0).sum())
    per_season = (
        frame.groupby("season")["paired_diff"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "diff_fraction", "size": "games"})
    )
    per_season_detail = {
        str(season): {
            "diff_points": float(row["diff_fraction"] * 100.0),
            "games": int(row["games"]),
        }
        for season, row in per_season.iterrows()
    }
    return {
        "effect_accuracy_points": effect_points,
        "interval_low": low * 100.0,
        "interval_high": high * 100.0,
        "probability_positive": probability_positive,
        "decisive_games": len(decisive),
        "decisive_wins": wins,
        "decisive_losses": losses,
        "per_season": per_season_detail,
    }


def build_extra_columns(population, schedules, builders):
    opener_lines = builders["default_opener_lines"](schedules)
    tables = [
        builders["derive_ats_streak_regress_features"](schedules),
        builders["derive_post_ot_fatigue_features"](schedules),
        builders["derive_division_dog_features"](schedules, opener_lines),
        builders["derive_low_total_div_home_dog_features"](schedules, opener_lines),
        builders["derive_sept_heat_home_features"](schedules),
    ]
    enriched = population.copy()
    for table in tables:
        key = table.columns[1]
        lookup = table.drop_duplicates("game_id").set_index("game_id")[key]
        enriched[key] = enriched["game_id"].astype(str).map(lookup)
    regime_input = pd.DataFrame(
        {"spread_line": pd.to_numeric(enriched["tue_open_home_spread"], errors="coerce")}
    )
    regime = builders["attach_spread_regime"](regime_input)
    enriched["spread_size_signed"] = pd.to_numeric(
        regime["regime_signed_absolute"], errors="coerce"
    )
    enriched["key7_distance_signed"] = pd.to_numeric(regime["regime_distance_7"], errors="coerce")
    enriched["den_home_flag"] = (
        enriched["home_team"].astype(str).eq("DEN") & enriched["game_type"].astype(str).eq("REG")
    ).astype(float)
    coverage = {}
    for name in EXTRA_FEATURES:
        coverage[name] = {
            "non_null": int(enriched[name].notna().sum()),
            "null_filled_zero": int(enriched[name].isna().sum()),
            "nonzero": int(enriched[name].fillna(0.0).ne(0.0).sum()),
        }
        enriched[name] = enriched[name].fillna(0.0).astype(float)
    total_lookup = opener_lines.drop_duplicates("game_id").set_index("game_id")[
        "tue_open_total_line"
    ]
    coverage["opener_total_join"] = {
        "graded_games": len(enriched),
        "with_opener_total": int(enriched["game_id"].astype(str).map(total_lookup).notna().sum()),
    }
    return enriched, coverage


def main(argv=None):
    global BOOTSTRAP_DRAWS, BOOTSTRAP_SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--out", type=str, default="tests/scratch/pooled_signal_second_fit.json")
    args = parser.parse_args(argv)
    BOOTSTRAP_DRAWS = int(args.draws)
    BOOTSTRAP_SEED = int(args.seed)

    _repo_src()
    from nfl_ats.pick_probability_fit import FIT_ITERATIONS, FIT_RIDGE, build_fit_population
    from nfl_ats.schedule_flag_features import (
        default_opener_lines,
        derive_ats_streak_regress_features,
        derive_division_dog_features,
        derive_low_total_div_home_dog_features,
        derive_post_ot_fatigue_features,
        derive_sept_heat_home_features,
    )
    from nfl_ats.snapshots import latest_snapshot, load_snapshot
    from nfl_ats.spread_regime import attach_spread_regime

    builders = {
        "derive_ats_streak_regress_features": derive_ats_streak_regress_features,
        "derive_division_dog_features": derive_division_dog_features,
        "derive_low_total_div_home_dog_features": derive_low_total_div_home_dog_features,
        "derive_post_ot_fatigue_features": derive_post_ot_fatigue_features,
        "derive_sept_heat_home_features": derive_sept_heat_home_features,
        "default_opener_lines": default_opener_lines,
        "latest_snapshot": latest_snapshot,
        "load_snapshot": load_snapshot,
        "attach_spread_regime": attach_spread_regime,
    }

    population, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    schedules, _team_stats = load_snapshot(latest_snapshot(REPO / "data" / "raw"))
    enriched, coverage = build_extra_columns(population, schedules, builders)

    base_oos, base_folds, base_insample, _base_full = loso_fit(
        enriched, list(BASE_FEATURES), FIT_RIDGE, FIT_ITERATIONS
    )
    full_oos, full_folds, full_insample, full_natural = loso_fit(
        enriched, list(FULL_FEATURES), FIT_RIDGE, FIT_ITERATIONS
    )
    scored = enriched.copy()
    scored["base_oos_prob"] = base_oos
    scored["full_oos_prob"] = full_oos
    scored = scored.loc[scored["full_oos_prob"].notna()].copy()
    scored["full_correct"] = (
        scored["full_oos_prob"].ge(0.5).astype(float).eq(scored["home_covered"]).astype(float)
    )
    scored["base_correct"] = (
        scored["base_oos_prob"].ge(0.5).astype(float).eq(scored["home_covered"]).astype(float)
    )
    scored["diff_vs_model_only"] = scored["full_correct"] - scored["model_correct"]
    scored["diff_vs_four_term"] = scored["full_correct"] - scored["base_correct"]

    vs_model = paired_summary(scored, "diff_vs_model_only")
    vs_four = paired_summary(scored, "diff_vs_four_term")

    truth = scored["home_covered"]
    metrics = {
        "full_oos_accuracy": accuracy(scored["full_oos_prob"], truth),
        "full_insample_accuracy": accuracy(full_insample, enriched["home_covered"]),
        "base_oos_accuracy": accuracy(scored["base_oos_prob"], truth),
        "base_insample_accuracy": accuracy(base_insample, enriched["home_covered"]),
        "model_only_accuracy": float(scored["model_correct"].mean()),
        "full_oos_brier": brier(scored["full_oos_prob"], truth),
        "base_oos_brier": brier(scored["base_oos_prob"], truth),
        "model_oos_brier": brier(scored["model_probability"], truth),
        "full_oos_logloss": logloss(scored["full_oos_prob"], truth),
        "base_oos_logloss": logloss(scored["base_oos_prob"], truth),
        "model_oos_logloss": logloss(scored["model_probability"], truth),
    }
    metrics["full_insample_minus_oos_gap"] = (
        metrics["full_insample_accuracy"] - metrics["full_oos_accuracy"]
    )
    metrics["base_insample_minus_oos_gap"] = (
        metrics["base_insample_accuracy"] - metrics["base_oos_accuracy"]
    )

    fold_frame = pd.DataFrame(full_folds).T
    stability = {
        name: {
            "mean": float(fold_frame[name].mean()),
            "sd": float(fold_frame[name].std(ddof=0)),
            "min": float(fold_frame[name].min()),
            "max": float(fold_frame[name].max()),
            "full_sample": float(full_natural[name]),
        }
        for name in fold_frame.columns
    }

    results = {
        "command": "python scripts/pooled_signal_second_fit.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_features": list(BASE_FEATURES),
        "extra_features": list(EXTRA_FEATURES),
        "n_extra_columns": len(EXTRA_FEATURES),
        "dropped_candidates": [
            "roof_state: family is a walk-forward pregame roof-state predictor "
            "built from forecast plus venue history with no derive_* builder; "
            "needs new snapshot history, dropped with no substitute"
        ],
        "looks_count": 9,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "blocking": "season-week",
        "sample_games": len(scored),
        "sample_blocks": int(
            (scored["season"].astype(str) + "-w" + scored["week"].astype(str)).nunique()
        ),
        "vs_model_only": vs_model,
        "vs_four_term": vs_four,
        "metrics": metrics,
        "fold_coefficients": full_folds,
        "base_fold_coefficients": base_folds,
        "coefficient_stability": stability,
        "extra_column_coverage": coverage,
        "non_commensurable_columns": [
            "low_total_div_home_dog_flag: unsigned 0/1, fires home-side only",
            "sept_heat_home_flag: unsigned 0/1, tiny population",
            "den_home_flag: unsigned 0/1, population restricted to Denver home games",
            "spread_size_signed: continuous opener-spread points, residual to model_logit",
            "key7_distance_signed: continuous opener-spread points, residual to model_logit",
        ],
        "population_provenance": provenance,
    }
    out_path = (REPO / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(out_path.relative_to(REPO)).replace("\\", "/"),
                "n_extra_columns": len(EXTRA_FEATURES),
                "vs_model_only": vs_model,
                "vs_four_term": vs_four,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
