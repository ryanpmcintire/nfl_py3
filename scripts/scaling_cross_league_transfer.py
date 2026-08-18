"""Part 2 -- does borrowing strength from CFB help, hurt, or nothing?

Runs the machinery in ``nfl_ats.cross_league_transfer`` twice:

1. **Mismatch measurement** (``measure_league_mismatch``) on full-history
   Power-Five and Group-of-Five CFB frames -- a static diagnostic, never
   scored against held-out outcomes.
2. **The transfer benchmark** (``cross_league_transfer_benchmark``), with
   Power-Five CFB as the large "pretraining" auxiliary league and
   Group-of-Five CFB as the smaller "target" league -- a real, measured
   talent/market-depth distribution shift, entirely free under
   rotation-registry rule 8. This validates whether joint fitting,
   hierarchical shrinkage, and prior-mean ridge can EVER help a smaller,
   differently-distributed target before any NFL confirmation is proposed.
   It is NOT an NFL result and spends no NFL rotation-registry window --
   see ``docs/scaling_and_transfer.md``.

Both P5-only and G5-only pools are large, real CFB samples (roughly 5,700
and 4,900 games respectively over 2006-2025); nothing here touches NFL data
or ``registry/rotation_registry.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl_ats.cfb_opponent_adjustment import paired_margin_error_comparison
from nfl_ats.cross_league_transfer import (
    ALIGNED_TRANSFER_FEATURE_COLUMNS,
    TRANSFER_ARMS,
    cross_league_transfer_benchmark,
    measure_league_mismatch,
)
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.outcomes import outcome_bootstrap_intervals

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFB_FEATURES = REPO / "data/processed/cfb_game_features.parquet"

TEST_START_SEASON = 2018
TEST_END_SEASON = 2025
MIN_TRAIN_GAMES = 50
SHRINKAGE_SAMPLES = 150
SHRINKAGE_SEED = 20260818
PAIRED_BOOTSTRAP_SAMPLES = 2_000
PAIRED_BOOTSTRAP_SEED = 20260818

_REQUIRED = tuple(
    dict.fromkeys(
        (
            "game_id",
            "season",
            "week",
            "gameday",
            "spread_line",
            "home_spread_odds",
            "away_spread_odds",
            "result",
            "ats_margin",
            "home_cover",
            *ALIGNED_TRANSFER_FEATURE_COLUMNS,
        )
    )
)


def split_power5_group5(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition completed CFB games into Group-of-Five-only and Power-Five-only pools.

    Cross-tier games (one Power Five side, one Group of Five side) are
    dropped from BOTH pools -- they are neither a clean target-league
    observation nor a clean auxiliary-league one, and keeping them out
    avoids a games-counted-twice bookkeeping question.
    """

    completed = features.loc[
        pd.to_numeric(features["result"], errors="coerce").notna()
        & pd.to_numeric(features["ats_margin"], errors="coerce").notna()
    ].copy()
    group5 = completed.loc[completed["home_power5"].eq(0) & completed["away_power5"].eq(0)]
    power5 = completed.loc[completed["home_power5"].eq(1) & completed["away_power5"].eq(1)]
    return group5.loc[:, list(_REQUIRED)].copy(), power5.loc[:, list(_REQUIRED)].copy()


def paired_arm_evidence(predictions: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    """Margin (MAE/RMSE) + probability-metric (accuracy/Brier/log-loss) paired evidence.

    Reuses the project's own paired-bootstrap machinery instead of a third
    hand-rolled implementation: ``paired_margin_error_comparison`` (built for
    the CFB opponent-adjustment screen, margin-metric only) and
    ``paired_feature_comparisons`` (built for feature-family screens,
    probability-metric only, keyed on a ``feature_set`` column -- populated
    here from ``method``).
    """

    frames = []
    for block in ("week", "season"):
        margin = paired_margin_error_comparison(
            predictions,
            baseline_method=baseline,
            candidate_method=candidate,
            samples=PAIRED_BOOTSTRAP_SAMPLES,
            block=block,
            seed=PAIRED_BOOTSTRAP_SEED,
        )
        frames.append(margin.assign(block=block))

        prob_frame = predictions.loc[predictions["method"].isin([baseline, candidate])].copy()
        prob_frame["feature_set"] = prob_frame["method"]
        prob = paired_feature_comparisons(
            prob_frame,
            baseline_feature_set=baseline,
            samples=PAIRED_BOOTSTRAP_SAMPLES,
            block=block,
            seed=PAIRED_BOOTSTRAP_SEED,
        )
        prob = prob.drop(columns=["baseline_feature_set", "candidate_feature_set"])
        prob["metric"] = prob["metric"].str.replace("_improvement", "", regex=False)
        frames.append(prob.assign(block=block))
    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, "baseline", baseline)
    combined.insert(1, "candidate", candidate)
    return combined


def main() -> None:
    output = Path(
        "C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/"
        "56edf890-1650-456a-b560-8d8b00b374b6/scratchpad/scaling/cross_league_transfer"
    )
    output.mkdir(parents=True, exist_ok=True)

    print("Loading CFB features and splitting Power-Five / Group-of-Five pools")
    features = pd.read_parquet(DEFAULT_CFB_FEATURES)
    group5, power5 = split_power5_group5(features)
    print(f"group5 (target): {len(group5)} games; power5 (auxiliary): {len(power5)} games")

    print("\n=== Mismatch measurement (full history, static diagnostic) ===")
    mismatch = measure_league_mismatch(group5, power5, label_a="group5", label_b="power5")
    mismatch.per_feature.to_csv(output / "mismatch_per_feature.csv", index=False)
    print(mismatch.per_feature.to_string(index=False))
    print(
        f"cosine_similarity={mismatch.cosine_similarity:.4f}  "
        f"pearson_r={mismatch.pearson_correlation:.4f}"
    )
    print(
        f"residual_std: group5={mismatch.fit_a.residual_std:.4f}  "
        f"power5={mismatch.fit_b.residual_std:.4f}  ratio={mismatch.residual_std_ratio:.4f}"
    )

    print("\n=== Transfer benchmark walk-forward (2018-2025 Group-of-Five test window) ===")
    result = cross_league_transfer_benchmark(
        group5,
        power5,
        start_season=TEST_START_SEASON,
        end_season=TEST_END_SEASON,
        min_train_games=MIN_TRAIN_GAMES,
        shrinkage_samples=SHRINKAGE_SAMPLES,
        shrinkage_seed=SHRINKAGE_SEED,
    )
    result.predictions.to_parquet(output / "predictions.parquet", index=False)
    (output / "diagnostics.json").write_text(
        json.dumps(result.diagnostics, indent=2, sort_keys=True, default=float)
    )
    print(json.dumps(result.diagnostics, indent=2, sort_keys=True, default=float))
    print(
        f"shrinkage weights (mean/min/max over {len(result.shrinkage.weights)} components): "
        f"{result.shrinkage.weights.mean():.3f} / {result.shrinkage.weights.min():.3f} / "
        f"{result.shrinkage.weights.max():.3f}"
    )

    print("\n=== Delta vs market baseline, by arm (week-blocked, clean 2018-2025) ===")
    market_intervals = outcome_bootstrap_intervals(
        result.predictions,
        samples=PAIRED_BOOTSTRAP_SAMPLES,
        block="week",
        seed=PAIRED_BOOTSTRAP_SEED,
    )
    market_intervals.to_csv(output / "delta_vs_market.csv", index=False)
    headline_cols = [
        "method",
        "metric",
        "estimate",
        "delta_vs_market",
        "delta_lower",
        "delta_upper",
    ]
    print(
        market_intervals.loc[
            market_intervals["metric"].isin(["cover_accuracy", "cover_brier_score", "margin_mae"]),
            headline_cols,
        ].to_string(index=False)
    )

    print("\n=== Each transfer arm vs the target-only (current NFL-only-style) fit ===")
    all_evidence: list[pd.DataFrame] = []
    for candidate in [arm for arm in TRANSFER_ARMS if arm != "target_only"]:
        evidence = paired_arm_evidence(result.predictions, "target_only", candidate)
        all_evidence.append(evidence)
        print(f"\n--- target_only -> {candidate} ---")
        print(
            evidence.loc[
                :, ["metric", "block", "estimate", "lower", "upper", "probability_positive"]
            ].to_string(index=False)
        )
    evidence_table = pd.concat(all_evidence, ignore_index=True)
    evidence_table.to_csv(output / "arm_vs_target_only.csv", index=False)

    print(f"\nartifacts written to {output}")


if __name__ == "__main__":
    main()
