"""Run the MOD-17 research-half screen: one joint residual model vs two.

Executes the frozen predeclaration in ``docs/mod17_joint_residual_model.md``.
Every number this script prints is measured; nothing here decides anything
that document did not already commit to deciding before this ran.

This is a promotion-style look on the SAME Tuesday-opener archive several
other confirmations have already graded arms against (disclosed, per
``docs/player_arrests_policy_eval.md`` precedent) -- it spends no rotation
window, and records through ``nfl-ats weak-signals record`` only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.clv import opener_evaluation_metrics, opener_pick_evaluation  # noqa: E402
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES  # noqa: E402
from nfl_ats.joint_residual_model import (  # noqa: E402
    JOINT_RIDGE_ALPHA,
    MARGIN_BASELINE_FEATURES,
    POSITIVE_CONTROL_COLUMN,
    UNION_FEATURES,
    blocked_correlation,
    joint_opener_pick_evaluation,
    leak_target_into_feature,
    opener_accuracy_bootstrap,
    out_of_sample_r2,
    paired_opener_accuracy,
    pearson_correlation,
    per_season_correlation,
    realised_residual_frame,
    second_stage_predictions,
    totals_shaped_predictions,
    walk_forward_joint_predictions,
)
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.totals import TOTALS_FEATURES, blend_sweep, choose_weight  # noqa: E402
from nfl_ats.totals_wave2 import bootstrap_wave_vs_wave, wave_vs_wave_paired_frame  # noqa: E402

DEFAULT_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack.parquet"
DEFAULT_MARKET_ROOT = REPO_ROOT / "data/market/raw"
MARGIN_BASELINE_PROFILE = "weak_stack"
BASE_TOTAL_K = 0.1  # docs/totals_model.md's frozen served weight.
OPENER_BOOTSTRAP_SAMPLES = 20_000
OPENER_SEED = 20260905
TOTAL_BOOTSTRAP_SAMPLES = 2_000
TOTAL_SEED = 20260901  # matches nfl_ats.totals's own regime seed.
CORRELATION_SAMPLES = 2_000
CORRELATION_SEED = 20260905
STAGE2_MIN_TRAIN_GAMES = 200


def margin_model_config() -> dict[str, Any]:
    return {
        "feature_profile": MARGIN_BASELINE_PROFILE,
        "regressor": "ridge",
        "ridge_alpha": JOINT_RIDGE_ALPHA,
        "target": "market_residual",
    }


def margin_side(
    features: pd.DataFrame, *, market_root: Path, min_train_games: int
) -> dict[str, Any]:
    baseline = opener_pick_evaluation(
        market_root,
        features,
        active_model_config=margin_model_config(),
        min_train_games=min_train_games,
    )
    joint = joint_opener_pick_evaluation(
        baseline, features, feature_columns=UNION_FEATURES, min_train_games=min_train_games
    )
    paired = paired_opener_accuracy(baseline, joint)
    bootstrap = opener_accuracy_bootstrap(
        paired, samples=OPENER_BOOTSTRAP_SAMPLES, seed=OPENER_SEED
    )
    per_season = (
        paired.groupby("season")
        .agg(
            games=("game_id", "size"),
            baseline_accuracy=("baseline_correct_open", "mean"),
            candidate_accuracy=("candidate_correct_open", "mean"),
        )
        .reset_index()
    )
    per_season["delta_accuracy_points"] = (
        per_season["candidate_accuracy"] - per_season["baseline_accuracy"]
    ) * 100.0
    return {
        "baseline": baseline,
        "joint": joint,
        "paired": paired,
        "bootstrap": bootstrap,
        "per_season": per_season,
        "baseline_metrics": opener_evaluation_metrics(baseline),
    }


def total_side(population: pd.DataFrame, *, min_train_games: int) -> dict[str, Any]:
    base = totals_shaped_predictions(
        walk_forward_joint_predictions(
            population,
            feature_columns=TOTALS_FEATURES,
            target_columns=("total_residual",),
            min_train_games=min_train_games,
        ),
        target_column="total_residual",
    )
    joint_full = walk_forward_joint_predictions(
        population,
        feature_columns=UNION_FEATURES,
        target_columns=("margin_residual", "total_residual"),
        min_train_games=min_train_games,
    )
    joint = totals_shaped_predictions(joint_full, target_column="total_residual")
    joint_sweep = blend_sweep(joint)
    joint_k = choose_weight(joint_sweep)
    paired = wave_vs_wave_paired_frame(base, BASE_TOTAL_K, joint, joint_k)
    bootstrap = bootstrap_wave_vs_wave(paired, samples=TOTAL_BOOTSTRAP_SAMPLES, seed=TOTAL_SEED)
    per_season = (
        paired.groupby("season")
        .agg(
            games=("game_id", "size"),
            base_mae=("wave1_abs_error", "mean"),
            joint_mae=("wave2_abs_error", "mean"),
        )
        .reset_index()
    )
    per_season["mae_improvement"] = per_season["base_mae"] - per_season["joint_mae"]
    return {
        "base_predictions": base,
        "joint_predictions": joint_full,
        "joint_k": joint_k,
        "base_k": BASE_TOTAL_K,
        "paired": paired,
        "bootstrap": bootstrap,
        "per_season": per_season,
    }


def r2_report(
    population: pd.DataFrame, joint_full: pd.DataFrame, min_train_games: int
) -> dict[str, Any]:
    base_margin = walk_forward_joint_predictions(
        population,
        feature_columns=MARGIN_BASELINE_FEATURES,
        target_columns=("margin_residual",),
        min_train_games=min_train_games,
    )
    base_total = walk_forward_joint_predictions(
        population,
        feature_columns=TOTALS_FEATURES,
        target_columns=("total_residual",),
        min_train_games=min_train_games,
    )
    return {
        "population_games": len(joint_full),
        "margin_baseline_r2_vs_market": out_of_sample_r2(
            base_margin["margin_residual"], base_margin["predicted_margin_residual"]
        ),
        "margin_union_r2_vs_market": out_of_sample_r2(
            joint_full["margin_residual"], joint_full["predicted_margin_residual"]
        ),
        "total_baseline_r2_vs_market": out_of_sample_r2(
            base_total["total_residual"], base_total["predicted_total_residual"]
        ),
        "total_union_r2_vs_market": out_of_sample_r2(
            joint_full["total_residual"], joint_full["predicted_total_residual"]
        ),
    }


def coupling_report(joint_full: pd.DataFrame) -> dict[str, Any]:
    """Part 3's cross-informing test: does the OTHER target's prediction help
    beyond simple own-shrinkage of a noisy raw ridge output?"""

    coupled = second_stage_predictions(
        joint_full,
        target_columns=("margin_residual", "total_residual"),
        min_train_games=STAGE2_MIN_TRAIN_GAMES,
    )
    solo_margin = second_stage_predictions(
        joint_full,
        target_columns=("margin_residual",),
        predictor_columns=("predicted_margin_residual",),
        min_train_games=STAGE2_MIN_TRAIN_GAMES,
    )
    solo_total = second_stage_predictions(
        joint_full,
        target_columns=("total_residual",),
        predictor_columns=("predicted_total_residual",),
        min_train_games=STAGE2_MIN_TRAIN_GAMES,
    )
    common_ids = sorted(
        set(coupled["game_id"]) & set(solo_margin["game_id"]) & set(solo_total["game_id"])
    )
    coupled_c = coupled.set_index("game_id").loc[common_ids]
    solo_margin_c = solo_margin.set_index("game_id").loc[common_ids]
    solo_total_c = solo_total.set_index("game_id").loc[common_ids]
    return {
        "games": len(common_ids),
        "stage1_margin_r2": out_of_sample_r2(
            coupled_c["margin_residual"], coupled_c["predicted_margin_residual"]
        ),
        "stage1_total_r2": out_of_sample_r2(
            coupled_c["total_residual"], coupled_c["predicted_total_residual"]
        ),
        "stage2_coupled_margin_r2": out_of_sample_r2(
            coupled_c["margin_residual"], coupled_c["predicted_margin_residual_stage2"]
        ),
        "stage2_coupled_total_r2": out_of_sample_r2(
            coupled_c["total_residual"], coupled_c["predicted_total_residual_stage2"]
        ),
        "stage2_solo_margin_r2": out_of_sample_r2(
            solo_margin_c["margin_residual"], solo_margin_c["predicted_margin_residual_stage2"]
        ),
        "stage2_solo_total_r2": out_of_sample_r2(
            solo_total_c["total_residual"], solo_total_c["predicted_total_residual_stage2"]
        ),
        "interpretation": (
            "stage2_solo_* isolates pure shrinkage/recalibration of the noisy stage-1 raw "
            "prediction using ONLY its own history; stage2_coupled_* adds the OTHER target's "
            "stage-1 prediction as a second input. Coupled beating solo would be evidence of "
            "real cross-target information; coupled failing to beat solo means the observed "
            "stage-2 gain over stage-1 is recalibration, not coupling."
        ),
    }


def positive_control(
    features: pd.DataFrame,
    population: pd.DataFrame,
    baseline_opener: pd.DataFrame,
    base_total_predictions: pd.DataFrame,
    *,
    market_root: Path,
    min_train_games: int,
) -> dict[str, Any]:
    contaminated_features = features.copy()
    contaminated_features[POSITIVE_CONTROL_COLUMN] = pd.to_numeric(
        contaminated_features["ats_margin"], errors="coerce"
    )
    joint_leak_opener = joint_opener_pick_evaluation(
        baseline_opener,
        contaminated_features,
        feature_columns=UNION_FEATURES,
        min_train_games=min_train_games,
    )
    paired_leak_opener = paired_opener_accuracy(baseline_opener, joint_leak_opener)
    opener_bootstrap = opener_accuracy_bootstrap(
        paired_leak_opener, samples=OPENER_BOOTSTRAP_SAMPLES, seed=OPENER_SEED
    )

    contaminated_population = leak_target_into_feature(
        population, feature_column=POSITIVE_CONTROL_COLUMN, target_column="margin_residual"
    )
    joint_leak_full = walk_forward_joint_predictions(
        contaminated_population,
        feature_columns=UNION_FEATURES,
        target_columns=("margin_residual", "total_residual"),
        min_train_games=min_train_games,
    )
    margin_r2 = out_of_sample_r2(
        joint_leak_full["margin_residual"], joint_leak_full["predicted_margin_residual"]
    )
    total_r2 = out_of_sample_r2(
        joint_leak_full["total_residual"], joint_leak_full["predicted_total_residual"]
    )
    joint_leak_total = totals_shaped_predictions(joint_leak_full, target_column="total_residual")
    leak_sweep = blend_sweep(joint_leak_total)
    leak_k = choose_weight(leak_sweep)
    paired_leak_total = wave_vs_wave_paired_frame(
        base_total_predictions, BASE_TOTAL_K, joint_leak_total, leak_k
    )
    total_bootstrap = bootstrap_wave_vs_wave(
        paired_leak_total, samples=TOTAL_BOOTSTRAP_SAMPLES, seed=TOTAL_SEED
    )

    return {
        "leaked_column": POSITIVE_CONTROL_COLUMN,
        "method": "column replaced by the row's own margin_residual (unit slope, zero noise)",
        "margin_opener_bootstrap": opener_bootstrap,
        "margin_full_population_r2": margin_r2,
        "total_full_population_r2": total_r2,
        "total_bootstrap": total_bootstrap,
        "expected_shape": (
            "margin-side accuracy delta and R2 both read hugely positive (the harness can "
            "plainly detect an effect this large); total-side is NOT predeclared to move much, "
            "because margin truth is leaked, not total truth, and the two residuals are only "
            "weakly correlated"
        ),
        "shape_matches_expectation": bool(
            opener_bootstrap["probability_positive"] >= 0.95 and margin_r2 >= 0.9
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    args = parser.parse_args()

    started = time.time()
    features = pd.read_parquet(args.features)
    population = realised_residual_frame(features)

    margin = margin_side(
        features, market_root=args.market_root, min_train_games=args.min_train_games
    )
    total = total_side(population, min_train_games=args.min_train_games)
    r2 = r2_report(population, total["joint_predictions"], args.min_train_games)
    coupling = coupling_report(total["joint_predictions"])

    realised_correlation = {
        "overall": pearson_correlation(population["margin_residual"], population["total_residual"]),
        "per_season": per_season_correlation(
            population, "margin_residual", "total_residual"
        ).to_dict(orient="records"),
        "season_blocked_bootstrap": blocked_correlation(
            population,
            "margin_residual",
            "total_residual",
            block="season",
            samples=CORRELATION_SAMPLES,
            seed=CORRELATION_SEED,
        ),
    }
    predicted_correlation = {
        "overall": pearson_correlation(
            total["joint_predictions"]["predicted_margin_residual"],
            total["joint_predictions"]["predicted_total_residual"],
        ),
        "per_season": per_season_correlation(
            total["joint_predictions"], "predicted_margin_residual", "predicted_total_residual"
        ).to_dict(orient="records"),
        "season_blocked_bootstrap": blocked_correlation(
            total["joint_predictions"],
            "predicted_margin_residual",
            "predicted_total_residual",
            block="season",
            samples=CORRELATION_SAMPLES,
            seed=CORRELATION_SEED,
        ),
    }

    control = positive_control(
        features,
        population,
        margin["baseline"],
        total["base_predictions"],
        market_root=args.market_root,
        min_train_games=args.min_train_games,
    )

    result: dict[str, Any] = {
        "status": "scored",
        "population_games": len(population),
        "union_feature_count": len(UNION_FEATURES),
        "margin_baseline_feature_count": len(MARGIN_BASELINE_FEATURES),
        "totals_baseline_feature_count": len(TOTALS_FEATURES),
        "margin_side": {
            "opener_games": len(margin["paired"]),
            "opener_weeks": int(margin["paired"].groupby(["season", "week"]).ngroups),
            "baseline_metrics": margin["baseline_metrics"],
            "bootstrap": margin["bootstrap"],
            "per_season": margin["per_season"].to_dict(orient="records"),
        },
        "total_side": {
            "games": len(total["paired"]),
            "base_k": total["base_k"],
            "joint_k": total["joint_k"],
            "bootstrap": total["bootstrap"],
            "per_season": total["per_season"].to_dict(orient="records"),
        },
        "out_of_sample_r2": r2,
        "coupling_stage2": coupling,
        "realised_residual_correlation": realised_correlation,
        "predicted_residual_correlation": predicted_correlation,
        "positive_control": control,
        "decision_rule": (
            "promote joint margin output iff opener probability_positive > 0.5 and the point "
            "estimate is non-negative; promote joint total output iff its probability_positive "
            "(vs the served k=0.1 blend) > 0.5 and the point estimate is non-negative"
        ),
    }
    result["decision"] = {
        "promote_joint_margin": bool(
            margin["bootstrap"]["probability_positive"] > 0.5
            and margin["bootstrap"]["estimate"] >= 0.0
        ),
        "promote_joint_total": bool(
            total["bootstrap"]["probability_positive"] > 0.5
            and total["bootstrap"]["estimate"] >= 0.0
        ),
    }

    configuration = {
        "features_path": str(args.features),
        "market_root": str(args.market_root),
        "min_train_games": args.min_train_games,
        "ridge_alpha": JOINT_RIDGE_ALPHA,
        "opener_bootstrap_samples": OPENER_BOOTSTRAP_SAMPLES,
        "opener_seed": OPENER_SEED,
        "total_bootstrap_samples": TOTAL_BOOTSTRAP_SAMPLES,
        "total_seed": TOTAL_SEED,
        "correlation_samples": CORRELATION_SAMPLES,
        "correlation_seed": CORRELATION_SEED,
        "stage2_min_train_games": STAGE2_MIN_TRAIN_GAMES,
        "union_features": list(UNION_FEATURES),
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        "predeclaration": "docs/mod17_joint_residual_model.md",
        **configuration,
        "result": result,
        "provenance": artifact_provenance(configuration, args.features, project_root=REPO_ROOT),
    }

    output = (
        REPO_ROOT
        / "artifacts"
        / "mod17_joint_residual"
        / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    write_experiment_artifact(
        output,
        "results.json",
        payload,
        command="mod17-joint-residual-screen",
        metrics={
            "margin_opener_delta_accuracy_points": margin["bootstrap"]["estimate"],
            "margin_opener_probability_positive": margin["bootstrap"]["probability_positive"],
            "total_mae_improvement": total["bootstrap"]["estimate"],
            "total_probability_positive": total["bootstrap"]["probability_positive"],
            "realised_residual_correlation": realised_correlation["overall"],
        },
        notes=(
            "MOD-17 research half: joint vs marginal residual models. Promotion-style look on "
            "the reused Tuesday-opener archive; no rotation window spent."
        ),
    )
    margin["paired"].to_csv(output / "paired_margin_opener.csv", index=False)
    total["paired"].to_csv(output / "paired_total.csv", index=False)
    population[["game_id", "season", "week", "margin_residual", "total_residual"]].to_csv(
        output / "realised_residuals.csv", index=False
    )
    print(f"wrote {output / 'results.json'}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
