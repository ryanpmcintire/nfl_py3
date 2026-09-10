from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from nfl_ats.clv import pick_correct, week_blocked_bootstrap
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.margin import margin_feature_columns
from nfl_ats.modeling import regular_season_rows
from nfl_ats.totals import (
    chronological_blocks,
    design_matrix,
    make_totals_estimator,
)
from nfl_ats.totals_wave2 import WAVE2_DRIVE_FEATURES

BootstrapBlock = Literal["week", "season"]

MARGIN_BASELINE_FEATURES: tuple[str, ...] = margin_feature_columns("market_residual", "weak_stack")

_union_overlap = set(MARGIN_BASELINE_FEATURES) & set(WAVE2_DRIVE_FEATURES)
if _union_overlap:
    raise RuntimeError(f"MOD-17 union feature set has unexpected overlap: {sorted(_union_overlap)}")
UNION_FEATURES: tuple[str, ...] = MARGIN_BASELINE_FEATURES + WAVE2_DRIVE_FEATURES

JOINT_RIDGE_ALPHA = 10.0

POSITIVE_CONTROL_COLUMN = "home_point_diff"

_TARGET_COLUMNS: tuple[str, str] = ("margin_residual", "total_residual")


def make_joint_estimator(*, ridge_alpha: float = JOINT_RIDGE_ALPHA) -> BaseEstimator:

    return make_totals_estimator(ridge_alpha=ridge_alpha)


def realised_residual_frame(
    features: pd.DataFrame, *, feature_columns: Sequence[str] = UNION_FEATURES
) -> pd.DataFrame:

    required = {
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "result",
        "ats_margin",
        "spread_line",
        "total_line",
        "home_score",
        "away_score",
        *feature_columns,
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"MOD-17 population is missing columns: {', '.join(missing)}")

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    frame["market_total"] = pd.to_numeric(frame["total_line"], errors="coerce")
    frame["actual_total"] = pd.to_numeric(frame["home_score"], errors="coerce") + pd.to_numeric(
        frame["away_score"], errors="coerce"
    )
    frame["margin_residual"] = pd.to_numeric(frame["ats_margin"], errors="coerce")
    frame["total_residual"] = frame["actual_total"] - frame["market_total"]
    frame["market_error"] = frame["market_total"] - frame["actual_total"]

    finite = frame["margin_residual"].notna() & frame["total_residual"].notna()
    frame = frame.loc[finite].copy()
    if frame.empty:
        raise ValueError("MOD-17 population has no rows with both residual targets defined")
    return frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def walk_forward_joint_predictions(
    population: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = UNION_FEATURES,
    target_columns: Sequence[str] = _TARGET_COLUMNS,
    ridge_alpha: float = JOINT_RIDGE_ALPHA,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> pd.DataFrame:

    if not target_columns:
        raise ValueError("walk_forward_joint_predictions requires at least one target column")
    missing_targets = sorted(set(target_columns).difference(population.columns))
    if missing_targets:
        raise DataContractError(f"Population is missing targets: {', '.join(missing_targets)}")

    blocks = chronological_blocks(population)
    keys = list(zip(population["season"], population["week"], strict=True))
    order = np.array([blocks.index((int(season), int(week))) for season, week in keys])
    design = design_matrix(population, feature_columns)
    targets = population.loc[:, list(target_columns)].astype(float).to_numpy()

    chunks: list[pd.DataFrame] = []
    for position, (block_season, block_week) in enumerate(blocks):
        train_mask = order < position
        train_count = int(train_mask.sum())
        if train_count < min_train_games:
            continue
        test_mask = order == position
        estimator = make_joint_estimator(ridge_alpha=ridge_alpha)
        estimator.fit(design.loc[train_mask], targets[train_mask])
        predicted = np.atleast_2d(np.asarray(estimator.predict(design.loc[test_mask]), dtype=float))
        if predicted.shape[0] != int(test_mask.sum()):
            predicted = predicted.reshape(int(test_mask.sum()), -1)
        block = population.loc[test_mask, :].copy()
        for index, name in enumerate(target_columns):
            block[f"predicted_{name}"] = predicted[:, index]
        block["train_games"] = train_count
        block["block_season"] = block_season
        block["block_week"] = block_week
        chunks.append(block)

    if not chunks:
        raise ValueError(
            f"no block reached min_train_games={min_train_games}; population has {len(population)}"
        )
    return pd.concat(chunks, ignore_index=True)


def totals_shaped_predictions(
    predictions: pd.DataFrame, *, target_column: str = "total_residual"
) -> pd.DataFrame:

    column = f"predicted_{target_column}"
    if column not in predictions.columns:
        raise DataContractError(f"predictions is missing {column!r}")
    return predictions.assign(predicted_residual=predictions[column].astype(float))


def out_of_sample_r2(actual: pd.Series, predicted: pd.Series) -> float:

    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(actual_values) & np.isfinite(predicted_values)
    actual_values = actual_values[finite]
    predicted_values = predicted_values[finite]
    if len(actual_values) == 0:
        raise ValueError("out_of_sample_r2 requires at least one finite paired observation")
    total = float(np.sum(actual_values**2))
    if total == 0.0:
        raise ValueError("out_of_sample_r2 is undefined when every actual value is exactly 0")
    residual = float(np.sum((actual_values - predicted_values) ** 2))
    return 1.0 - residual / total


def pearson_correlation(a: pd.Series, b: pd.Series) -> float:

    left = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) < 2:
        raise ValueError("pearson_correlation requires at least two finite paired observations")
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def per_season_correlation(frame: pd.DataFrame, column_a: str, column_b: str) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []
    for season, group in frame.groupby("season", sort=True):
        try:
            correlation = pearson_correlation(group[column_a], group[column_b])
        except ValueError:
            correlation = float("nan")
        rows.append({"season": int(str(season)), "games": len(group), "correlation": correlation})
    return pd.DataFrame(rows)


def blocked_correlation(
    frame: pd.DataFrame,
    column_a: str,
    column_b: str,
    *,
    block: BootstrapBlock = "season",
    samples: int = 2_000,
    seed: int = 20260905,
) -> dict[str, Any]:

    def metric(sample: pd.DataFrame) -> dict[str, float]:
        try:
            return {"correlation": pearson_correlation(sample[column_a], sample[column_b])}
        except ValueError:
            return {"correlation": float("nan")}

    result = week_blocked_bootstrap(frame, metric, block=block, samples=samples, seed=seed)
    row = result.iloc[0]
    group_columns = ["season"] if block == "season" else ["season", "week"]
    return {
        "estimate": float(row["estimate"]),
        "lower": float(row["lower"]),
        "upper": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
        "samples": int(row["samples"]),
        "block": block,
        "blocks": int(frame.groupby(group_columns).ngroups),
        "games": len(frame),
    }


def second_stage_predictions(
    stage1_predictions: pd.DataFrame,
    *,
    target_columns: Sequence[str] = _TARGET_COLUMNS,
    predictor_columns: Sequence[str] | None = None,
    ridge_alpha: float = JOINT_RIDGE_ALPHA,
    min_train_games: int = 200,
) -> pd.DataFrame:

    predictor_columns = tuple(predictor_columns or (f"predicted_{name}" for name in target_columns))
    missing = sorted(set(predictor_columns).difference(stage1_predictions.columns))
    if missing:
        raise DataContractError(f"stage1_predictions is missing columns: {', '.join(missing)}")

    ordered = stage1_predictions.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    blocks = chronological_blocks(ordered)
    keys = list(zip(ordered["season"], ordered["week"], strict=True))
    order = np.array([blocks.index((int(season), int(week))) for season, week in keys])
    design = ordered.loc[:, list(predictor_columns)].astype(float)
    targets = ordered.loc[:, list(target_columns)].astype(float).to_numpy()

    chunks: list[pd.DataFrame] = []
    for position in range(len(blocks)):
        train_mask = order < position
        train_count = int(train_mask.sum())
        if train_count < min_train_games:
            continue
        test_mask = order == position
        estimator = make_joint_estimator(ridge_alpha=ridge_alpha)
        estimator.fit(design.loc[train_mask], targets[train_mask])
        predicted = np.asarray(estimator.predict(design.loc[test_mask]), dtype=float)
        predicted = predicted.reshape(int(test_mask.sum()), -1)
        block = ordered.loc[test_mask, :].copy()
        for index, name in enumerate(target_columns):
            block[f"predicted_{name}_stage2"] = predicted[:, index]
        block["stage2_train_games"] = train_count
        chunks.append(block)

    if not chunks:
        raise ValueError(
            f"no block reached min_train_games={min_train_games} for the stage-2 fit; "
            f"stage-1 output has {len(ordered)} rows"
        )
    return pd.concat(chunks, ignore_index=True)


def leak_target_into_feature(
    frame: pd.DataFrame, *, feature_column: str = POSITIVE_CONTROL_COLUMN, target_column: str
) -> pd.DataFrame:

    if feature_column not in frame.columns:
        raise DataContractError(f"{feature_column!r} is not a column of the given frame")
    if target_column not in frame.columns:
        raise DataContractError(f"{target_column!r} is not a column of the given frame")
    contaminated = frame.copy()
    contaminated[feature_column] = contaminated[target_column].astype(float)
    return contaminated


def joint_opener_pick_evaluation(
    baseline: pd.DataFrame,
    features: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = UNION_FEATURES,
    ridge_alpha: float = JOINT_RIDGE_ALPHA,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> pd.DataFrame:

    required_baseline = {
        "game_id",
        "season",
        "week",
        "tue_open_home_spread",
        "close_home_spread",
        "margin_vs_open",
        "margin_vs_close",
    }
    missing_baseline = sorted(required_baseline.difference(baseline.columns))
    if missing_baseline:
        raise DataContractError(f"baseline is missing columns: {', '.join(missing_baseline)}")

    population = realised_residual_frame(features, feature_columns=feature_columns)

    scored_weeks: list[pd.DataFrame] = []
    for (season, week), group in baseline.groupby(["season", "week"], sort=True):
        week_ids = set(group["game_id"].astype(str))
        week_rows = population.loc[population["game_id"].astype(str).isin(week_ids)]
        if week_rows.empty:
            continue
        cutoff = week_rows["gameday"].min()
        training = population.loc[population["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        estimator = make_joint_estimator(ridge_alpha=ridge_alpha)
        estimator.fit(
            design_matrix(training, feature_columns),
            training.loc[:, list(_TARGET_COLUMNS)].astype(float).to_numpy(),
        )
        scoring = group[["game_id", "tue_open_home_spread", "close_home_spread"]].merge(
            week_rows.loc[:, ["game_id", *feature_columns]], on="game_id", how="inner"
        )
        if len(scoring) != len(group):
            raise ValueError(
                f"season {season} week {week}: {len(group) - len(scoring)} archived games "
                "were not found in the feature table"
            )
        at_open = scoring.copy()
        at_open["spread_line"] = at_open["tue_open_home_spread"]
        at_close = scoring.copy()
        at_close["spread_line"] = at_close["close_home_spread"]
        predicted_open = np.asarray(
            estimator.predict(design_matrix(at_open, feature_columns)), dtype=float
        ).reshape(len(scoring), -1)
        predicted_close = np.asarray(
            estimator.predict(design_matrix(at_close, feature_columns)), dtype=float
        ).reshape(len(scoring), -1)
        scored = scoring[["game_id"]].copy()
        scored["season"] = int(str(season))
        scored["week"] = int(str(week))
        scored["predicted_margin_residual_open"] = predicted_open[:, 0]
        scored["predicted_total_residual_open"] = predicted_open[:, 1]
        scored["predicted_margin_residual_close"] = predicted_close[:, 0]
        scored["predicted_total_residual_close"] = predicted_close[:, 1]
        scored["train_games"] = len(training)
        scored_weeks.append(scored)

    if not scored_weeks:
        raise ValueError("No archived week had at least min_train_games completed training rows")
    joint = pd.concat(scored_weeks, ignore_index=True)
    joint["pick_home_at_open"] = joint["predicted_margin_residual_open"].gt(0.0)
    joint["pick_home_at_close"] = joint["predicted_margin_residual_close"].gt(0.0)
    return joint.merge(
        baseline[["game_id", "margin_vs_open", "margin_vs_close"]], on="game_id", how="inner"
    )


def paired_opener_accuracy(baseline: pd.DataFrame, joint: pd.DataFrame) -> pd.DataFrame:

    required = {"game_id", "season", "week", "correct_at_open", "margin_vs_open"}
    missing = sorted(required.difference(baseline.columns))
    if missing:
        raise DataContractError(f"baseline is missing columns: {', '.join(missing)}")
    if "pick_home_at_open" not in joint.columns:
        raise DataContractError("joint is missing pick_home_at_open")

    left = baseline[["game_id", "season", "week", "correct_at_open", "margin_vs_open"]].copy()
    right = joint[["game_id", "pick_home_at_open"]].copy()
    paired = left.merge(right, on="game_id", how="inner", validate="one_to_one")
    paired["candidate_correct_open"] = pick_correct(
        paired["pick_home_at_open"], paired["margin_vs_open"]
    )
    paired["baseline_correct_open"] = pd.to_numeric(paired["correct_at_open"], errors="coerce")
    paired["delta"] = paired["candidate_correct_open"] - paired["baseline_correct_open"]
    return paired.dropna(subset=["baseline_correct_open", "candidate_correct_open"]).reset_index(
        drop=True
    )


def opener_accuracy_bootstrap(
    paired: pd.DataFrame, *, samples: int = 20_000, seed: int = 20260905
) -> dict[str, float]:

    def metric(sample: pd.DataFrame) -> dict[str, float]:
        return {"accuracy_points": float(sample["delta"].mean()) * 100.0}

    result = week_blocked_bootstrap(paired, metric, block="week", samples=samples, seed=seed)
    row = result.iloc[0]
    return {
        "estimate": float(row["estimate"]),
        "lower": float(row["lower"]),
        "upper": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
        "samples": int(row["samples"]),
        "games": len(paired),
        "weeks": int(paired.groupby(["season", "week"]).ngroups),
    }


__all__ = [
    "JOINT_RIDGE_ALPHA",
    "MARGIN_BASELINE_FEATURES",
    "POSITIVE_CONTROL_COLUMN",
    "UNION_FEATURES",
    "blocked_correlation",
    "joint_opener_pick_evaluation",
    "leak_target_into_feature",
    "make_joint_estimator",
    "opener_accuracy_bootstrap",
    "out_of_sample_r2",
    "paired_opener_accuracy",
    "pearson_correlation",
    "per_season_correlation",
    "realised_residual_frame",
    "second_stage_predictions",
    "totals_shaped_predictions",
    "walk_forward_joint_predictions",
]
