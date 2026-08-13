"""Comparable feature-set experiments over the walk-forward evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from nfl_ats.backtest import summarize_predictions, walk_forward_backtest
from nfl_ats.constants import FEATURE_SETS
from nfl_ats.margin import MARGIN_FEATURE_PROFILES, MarginFeatureProfile
from nfl_ats.outcomes import summarize_outcome_method, walk_forward_outcomes

DEFAULT_EXPERIMENT_SETS = (
    "market",
    "market_context",
    "market_elo",
    "football",
    "full_without_ats",
    "full",
)

DEFAULT_PLAYER_PROFILE_SETS = (
    "base",
    "player_qb",
    "player_injuries",
    "player_continuity",
    "player_qb_injuries",
    "player_qb_continuity",
    "player_injuries_continuity",
    "player",
    "player_injury_value",
    "player_value",
)

PairedBlock = Literal["week", "season"]


@dataclass(frozen=True)
class ExperimentResult:
    summary: pd.DataFrame
    season_summary: pd.DataFrame
    predictions: pd.DataFrame


@dataclass(frozen=True)
class OutcomeProfileExperimentResult:
    summary: pd.DataFrame
    season_summary: pd.DataFrame
    predictions: pd.DataFrame


@dataclass(frozen=True)
class OutcomeProfileSelectionResult:
    candidate_validation: pd.DataFrame
    fold_summary: pd.DataFrame
    predictions: pd.DataFrame
    summary: pd.DataFrame
    season_summary: pd.DataFrame


def _paired_row_improvements(paired: pd.DataFrame) -> pd.DataFrame:
    actual = paired["home_cover_baseline"].to_numpy(dtype=float)
    baseline = paired["home_cover_probability_baseline"].to_numpy(dtype=float)
    candidate = paired["home_cover_probability_candidate"].to_numpy(dtype=float)
    baseline_clipped = np.clip(baseline, 1e-15, 1.0 - 1e-15)
    candidate_clipped = np.clip(candidate, 1e-15, 1.0 - 1e-15)
    return pd.DataFrame(
        {
            "accuracy_improvement": ((candidate >= 0.5) == actual).astype(float)
            - ((baseline >= 0.5) == actual).astype(float),
            "brier_improvement": np.square(baseline - actual) - np.square(candidate - actual),
            "log_loss_improvement": -(
                actual * np.log(baseline_clipped) + (1.0 - actual) * np.log(1.0 - baseline_clipped)
            )
            + (
                actual * np.log(candidate_clipped)
                + (1.0 - actual) * np.log(1.0 - candidate_clipped)
            ),
        },
        index=paired.index,
    )


def paired_feature_comparisons(
    predictions: pd.DataFrame,
    *,
    baseline_feature_set: str,
    samples: int = 2_000,
    confidence: float = 0.95,
    block: PairedBlock = "week",
    seed: int = 20260812,
) -> pd.DataFrame:
    """Block-bootstrap paired per-game improvements over a feature baseline.

    Positive estimates mean the candidate is better. Pairing keeps the exact
    same game outcomes in both arms and resamples whole weeks or seasons.
    """

    if samples < 10:
        raise ValueError("samples must be at least 10")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if block not in ("week", "season"):
        raise ValueError("block must be 'week' or 'season'")
    required = {
        "feature_set",
        "game_id",
        "season",
        "week",
        "home_cover",
        "home_cover_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing paired columns: {', '.join(missing)}")
    feature_sets = set(predictions["feature_set"].astype(str))
    if baseline_feature_set not in feature_sets:
        raise ValueError(f"Unknown paired baseline feature set: {baseline_feature_set}")

    columns = [
        "game_id",
        "season",
        "week",
        "home_cover",
        "home_cover_probability",
    ]
    baseline = predictions.loc[predictions["feature_set"].eq(baseline_feature_set), columns]
    rows: list[dict[str, Any]] = []
    tail = (1.0 - confidence) / 2.0
    for candidate_name in sorted(feature_sets.difference((baseline_feature_set,))):
        candidate = predictions.loc[predictions["feature_set"].eq(candidate_name), columns]
        paired = baseline.merge(
            candidate,
            on="game_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_baseline", "_candidate"),
        )
        paired = paired.loc[
            paired["home_cover_baseline"].notna() & paired["home_cover_candidate"].notna()
        ].copy()
        if paired.empty:
            raise ValueError(f"No paired completed games for {candidate_name}")
        for column in ("season", "week", "home_cover"):
            if not paired[f"{column}_baseline"].equals(paired[f"{column}_candidate"]):
                raise ValueError(f"Paired {column} values differ for {candidate_name}")

        improvements = _paired_row_improvements(paired)
        group_columns = ["season_baseline", "week_baseline"]
        if block == "season":
            group_columns = ["season_baseline"]
        grouped_indices = list(paired.groupby(group_columns, sort=False).indices.values())
        generator = np.random.default_rng(seed)
        draws = np.empty((samples, len(improvements.columns)), dtype=float)
        for sample_index in range(samples):
            selected = generator.integers(0, len(grouped_indices), size=len(grouped_indices))
            positions = np.concatenate([grouped_indices[index] for index in selected])
            draws[sample_index] = improvements.iloc[positions].mean(axis=0).to_numpy()

        for metric_index, metric in enumerate(improvements.columns):
            rows.append(
                {
                    "baseline_feature_set": baseline_feature_set,
                    "candidate_feature_set": candidate_name,
                    "metric": metric,
                    "estimate": float(improvements[metric].mean()),
                    "lower": float(np.quantile(draws[:, metric_index], tail)),
                    "upper": float(np.quantile(draws[:, metric_index], 1.0 - tail)),
                    "confidence": confidence,
                    "block": block,
                    "samples": samples,
                    "paired_games": len(paired),
                }
            )
    return pd.DataFrame(rows)


def run_outcome_profile_experiment(
    features: pd.DataFrame,
    *,
    start_season: int,
    profiles: tuple[str, ...] = DEFAULT_PLAYER_PROFILE_SETS,
    regressor: str = "ridge",
    min_edge: float = 0.02,
    min_train_games: int = 500,
) -> OutcomeProfileExperimentResult:
    """Evaluate only the residual-margin method across matched player profiles.

    This intentionally avoids fitting fair-margin, straight-up, and direct-ATS
    estimators that cannot answer the player-family ablation question.
    """

    if not profiles:
        raise ValueError("At least one player profile is required")
    if len(profiles) != len(set(profiles)):
        raise ValueError("Player profiles must be unique")
    unknown = sorted(set(profiles).difference(MARGIN_FEATURE_PROFILES))
    if unknown:
        raise ValueError(f"Unknown player profiles: {', '.join(unknown)}")

    summaries: list[pd.DataFrame] = []
    season_summaries: list[pd.DataFrame] = []
    prediction_batches: list[pd.DataFrame] = []
    for profile_name in profiles:
        profile: MarginFeatureProfile = profile_name  # type: ignore[assignment]
        result = walk_forward_outcomes(
            features,
            start_season=start_season,
            regressor=regressor,
            min_edge=min_edge,
            min_train_games=min_train_games,
            feature_profile=profile,
            methods=("market_residual",),
        )
        summary = result.summary.copy()
        summary.insert(0, "feature_profile", profile_name)
        summaries.append(summary)
        season_summary = result.season_summary.copy()
        season_summary.insert(0, "feature_profile", profile_name)
        season_summaries.append(season_summary)
        predictions = result.predictions.copy()
        predictions.insert(0, "feature_profile", profile_name)
        prediction_batches.append(predictions)

    return OutcomeProfileExperimentResult(
        summary=pd.concat(summaries, ignore_index=True),
        season_summary=pd.concat(season_summaries, ignore_index=True),
        predictions=pd.concat(prediction_batches, ignore_index=True),
    )


def nested_outcome_profile_selection(
    predictions: pd.DataFrame,
    *,
    first_test_season: int,
    validation_seasons: int = 2,
) -> OutcomeProfileSelectionResult:
    """Select a player profile on prior seasons, then score the next season once."""

    if validation_seasons < 1:
        raise ValueError("validation_seasons must be positive")
    required = {
        "feature_profile",
        "season",
        "game_id",
        "home_cover",
        "home_cover_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Profile predictions are missing columns: {', '.join(missing)}")
    available_seasons = sorted(predictions["season"].astype(int).unique())
    test_seasons = [season for season in available_seasons if season >= first_test_season]
    if not test_seasons:
        raise ValueError(f"No profile predictions found from season {first_test_season}")
    profiles = sorted(predictions["feature_profile"].astype(str).unique())

    candidate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selected_batches: list[pd.DataFrame] = []
    for test_season in test_seasons:
        validation_range = tuple(range(test_season - validation_seasons, test_season))
        missing_seasons = sorted(set(validation_range).difference(available_seasons))
        if missing_seasons:
            raise ValueError(
                f"Test season {test_season} lacks validation seasons: {missing_seasons}"
            )
        validation = predictions.loc[predictions["season"].isin(validation_range)]
        candidates: list[dict[str, Any]] = []
        for profile in profiles:
            rows = validation.loc[validation["feature_profile"].eq(profile)]
            cover_rows = rows.loc[
                rows["home_cover"].notna() & rows["home_cover_probability"].notna()
            ]
            actual = cover_rows["home_cover"].to_numpy(dtype=float)
            probability = cover_rows["home_cover_probability"].to_numpy(dtype=float)
            candidate = {
                "test_season": test_season,
                "validation_start_season": validation_range[0],
                "validation_end_season": validation_range[-1],
                "feature_profile": profile,
                "validation_games": len(cover_rows),
                "validation_accuracy": float(((probability >= 0.5) == actual).mean()),
                "validation_brier_score": float(np.square(probability - actual).mean()),
            }
            candidates.append(candidate)
            candidate_rows.append(candidate)
        selected = sorted(
            candidates,
            key=lambda row: (
                -float(row["validation_accuracy"]),
                float(row["validation_brier_score"]),
                str(row["feature_profile"]),
            ),
        )[0]
        selected_profile = str(selected["feature_profile"])
        test_rows = predictions.loc[
            predictions["season"].eq(test_season)
            & predictions["feature_profile"].eq(selected_profile)
        ].copy()
        if test_rows.empty:
            raise ValueError(
                f"Selected profile {selected_profile} has no rows for season {test_season}"
            )
        test_summary = summarize_outcome_method(test_rows)
        test_rows["selected_feature_profile"] = selected_profile
        test_rows["validation_start_season"] = validation_range[0]
        test_rows["validation_end_season"] = validation_range[-1]
        selected_batches.append(test_rows)
        fold_rows.append(
            {
                **selected,
                "test_games": test_summary["cover_games"],
                "test_accuracy": test_summary["cover_accuracy"],
                "test_brier_score": test_summary["cover_brier_score"],
            }
        )

    selected_predictions = pd.concat(selected_batches, ignore_index=True)
    summary = pd.DataFrame([summarize_outcome_method(selected_predictions)])
    season_summary = pd.DataFrame(
        [
            {"season": int(str(season)), **summarize_outcome_method(rows)}
            for season, rows in selected_predictions.groupby("season", sort=True)
        ]
    )
    return OutcomeProfileSelectionResult(
        candidate_validation=pd.DataFrame(candidate_rows),
        fold_summary=pd.DataFrame(fold_rows),
        predictions=selected_predictions,
        summary=summary,
        season_summary=season_summary,
    )


def run_feature_set_experiment(
    features: pd.DataFrame,
    start_season: int,
    model_name: str = "logistic",
    feature_sets: tuple[str, ...] = DEFAULT_EXPERIMENT_SETS,
    min_edge: float = 0.02,
    min_train_games: int = 500,
) -> ExperimentResult:
    unknown = sorted(set(feature_sets).difference(FEATURE_SETS))
    if unknown:
        raise ValueError(f"Unknown feature sets: {', '.join(unknown)}")
    if len(set(feature_sets)) != len(feature_sets):
        raise ValueError("Feature sets must be unique")

    metrics: list[dict[str, Any]] = []
    prediction_batches: list[pd.DataFrame] = []
    for feature_set in feature_sets:
        result = walk_forward_backtest(
            features,
            start_season=start_season,
            model_name=model_name,
            min_edge=min_edge,
            min_train_games=min_train_games,
            feature_set=feature_set,
        )
        metric_row = {**result.metrics, "feature_count": len(FEATURE_SETS[feature_set])}
        metrics.append(metric_row)
        batch = result.predictions.copy()
        batch["feature_set"] = feature_set
        prediction_batches.append(batch)

    summary = pd.DataFrame(metrics).sort_values(
        ["brier_score", "log_loss"], ascending=True, ignore_index=True
    )
    predictions = pd.concat(prediction_batches, ignore_index=True)
    season_rows: list[dict[str, Any]] = []
    for (group_feature_set, season), season_predictions in predictions.groupby(
        ["feature_set", "season"], sort=True
    ):
        season_rows.append(
            {
                "feature_set": str(group_feature_set),
                "season": int(str(season)),
                **summarize_predictions(season_predictions),
            }
        )
    season_summary = pd.DataFrame(season_rows)
    return ExperimentResult(
        summary=summary,
        season_summary=season_summary,
        predictions=predictions,
    )
