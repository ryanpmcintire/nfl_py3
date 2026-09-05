"""MOD-17 research half: does one joint margin/total residual model beat two.

Executes the frozen predeclaration in ``docs/mod17_joint_residual_model.md``
(written 2026-09-05, before any number below was computed). Pure functions
only -- no filesystem I/O -- so every step here is independently testable;
``scripts/mod17_joint_residual_screen.py`` is the thin I/O layer that loads
the production feature table, calls these functions, and writes the artifact.

The predeclared, load-bearing fact this module is built around: an ordinary
:class:`sklearn.linear_model.Ridge` fit against a two-column target is
mathematically IDENTICAL, column by column, to fitting two independent
single-target ridges on the same design matrix -- ridge regression's
closed-form solution for output column ``j`` is
``beta_j = (X'X + alpha*I)^-1 X'y_j``, which never references any other
column of ``y``. ``tests/test_joint_residual_model.py`` pins this identity.
So the "joint model" (:func:`walk_forward_joint_predictions`) measures the
effect of the WIDER union feature set, not of joint estimation as such; the
one arm that can show real cross-target coupling is the second-stage
regression in :func:`second_stage_predictions`, which regresses each target
on the OTHER target's stage-1 out-of-fold prediction.
"""

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

#: Production's margin feature contract for the ``market_residual`` target
#: under the ``weak_stack`` profile -- the "base marginal margin model" arm,
#: reused unmodified rather than re-derived.
MARGIN_BASELINE_FEATURES: tuple[str, ...] = margin_feature_columns("market_residual", "weak_stack")

#: The frozen union: production margin features plus the totals wave-2
#: drive-pace family. ``docs/mod17_joint_residual_model.md`` verifies the
#: totals wave-1 allowlist is already a strict subset of
#: ``MARGIN_BASELINE_FEATURES`` (so it needs no separate listing here) and
#: that this union has zero internal overlap.
_union_overlap = set(MARGIN_BASELINE_FEATURES) & set(WAVE2_DRIVE_FEATURES)
if _union_overlap:
    raise RuntimeError(f"MOD-17 union feature set has unexpected overlap: {sorted(_union_overlap)}")
UNION_FEATURES: tuple[str, ...] = MARGIN_BASELINE_FEATURES + WAVE2_DRIVE_FEATURES

#: Production's ridge penalty, unchanged throughout this module.
JOINT_RIDGE_ALPHA = 10.0

#: The positive-control column: an arbitrary, pre-chosen, already-present
#: member of the union design matrix -- same convention
#: ``nfl_ats.totals_wave2.POSITIVE_CONTROL_COLUMN`` uses.
POSITIVE_CONTROL_COLUMN = "home_point_diff"

_TARGET_COLUMNS: tuple[str, str] = ("margin_residual", "total_residual")


def make_joint_estimator(*, ridge_alpha: float = JOINT_RIDGE_ALPHA) -> BaseEstimator:
    """Production's exact pipeline, reused so no arm can win or lose on plumbing.

    Identical to ``nfl_ats.totals.make_totals_estimator`` -- re-exported under
    this module's own name because the caller here fits it against a 2-D
    target, and ``SimpleImputer``/``StandardScaler`` are agnostic to ``y``
    shape so nothing about the recipe needs to change.
    """

    return make_totals_estimator(ridge_alpha=ridge_alpha)


def realised_residual_frame(
    features: pd.DataFrame, *, feature_columns: Sequence[str] = UNION_FEATURES
) -> pd.DataFrame:
    """Regular-season games with both residual targets defined, one row each.

    ``margin_residual`` is ``ats_margin`` (production's own margin target);
    ``total_residual`` is ``(home_score + away_score) - total_line``,
    identical to ``nfl_ats.totals``'s target. Also carries ``market_total``
    and ``actual_total`` under those names so the resulting frame can feed
    ``nfl_ats.totals``/``nfl_ats.totals_wave2``'s blend/pairing helpers
    without renaming at each call site.
    """

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
    """Expanding-window walk-forward, one fit per ``(season, week)`` block.

    Generalizes ``nfl_ats.totals.walk_forward_predictions`` to an arbitrary
    number of target columns (one or many): a single-column ``target_columns``
    reproduces a single-target ridge bit-for-bit; a two-column one is the
    "joint model" this module exists to measure. Reuses
    ``nfl_ats.totals.chronological_blocks``/``design_matrix`` unmodified so
    the block calendar and column-selection contract are identical to the
    already-run totals regime.
    """

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
            # sklearn's Ridge returns a 1-D array for a single target column;
            # atleast_2d on a 1-D array of length n makes a (1, n) row rather
            # than the (n, 1) column this loop needs -- reshape explicitly
            # rather than relying on atleast_2d's orientation guess.
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
    """Alias ``predicted_<target_column>`` to ``predicted_residual``.

    ``nfl_ats.totals``/``nfl_ats.totals_wave2``'s blend and pairing helpers
    (``blend_sweep``, ``paired_error_frame``, ``per_season_deltas``,
    ``wave_vs_wave_paired_frame``) all read a fixed ``predicted_residual``
    column name. Rather than reimplementing that math, this renames in place
    so the joint model's total-side output can be handed to those functions
    unmodified.
    """

    column = f"predicted_{target_column}"
    if column not in predictions.columns:
        raise DataContractError(f"predictions is missing {column!r}")
    return predictions.assign(predicted_residual=predictions[column].astype(float))


def out_of_sample_r2(actual: pd.Series, predicted: pd.Series) -> float:
    """R-squared against the "trust the market fully" (predict-zero) baseline.

    Both ``actual`` and ``predicted`` are already residuals against the
    market's own number, so predicting exactly 0 for every game IS the
    market baseline. ``1 - SS_res / SS_tot`` under that baseline reduces to
    ``1 - sum((actual - predicted)**2) / sum(actual**2)``, which is negative
    whenever the model's residual predictions are worse than trusting the
    market outright -- the same shape the margin and totals sides have both
    already measured (production margin MAE 10.00 vs market 9.91; total
    wave-1 raw-model MAE 10.5495 vs market 10.4249).
    """

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
    """Pearson correlation with a guard for degenerate (zero-variance) input."""

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
    """Season-by-season Pearson correlation of two columns, plus game counts."""

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
    """Season- (or week-) blocked bootstrap of the Pearson correlation of two columns.

    Reuses ``nfl_ats.clv.week_blocked_bootstrap`` unmodified -- the same
    resampling machinery every arm of this project already reports through --
    with a metric function that recomputes the correlation on each blocked
    resample rather than a mean.
    """

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
    """The cheap SUR-lite variant: regress each target on BOTH stage-1 predictions.

    Walk-forward over stage 1's own ``(season, week)`` block calendar: a
    block's stage-2 fit uses only stage-1 predictions from STRICTLY EARLIER
    blocks, so a stage-1 prediction is never used to help predict the very
    residual it was itself trained to predict. This is the one arm in this
    module that can show a real cross-target coupling effect, because
    ordinary multi-output ridge (:func:`walk_forward_joint_predictions`)
    cannot -- see the module docstring.
    """

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
    """Positive-control contamination: replace one feature with its row's own target value.

    Unit slope, zero noise -- identical method to
    ``nfl_ats.totals_wave2.run_positive_control``. Returns a copy; the input
    frame is never mutated.
    """

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
    """The joint model's opener/close margin (and total) prediction, on
    EXACTLY the game/week set ``nfl_ats.clv.opener_pick_evaluation`` already
    produced for ``baseline``.

    Reuses ``baseline``'s own ``(season, week)`` grouping and its already
    -computed ``tue_open_home_spread``/``close_home_spread`` columns rather
    than re-deriving the Tuesday-opener archive a second time from the odds
    snapshots -- ``opener_pick_evaluation`` already did that work once. Only
    ``spread_line`` is swapped between the open and close scoring passes,
    the same declared approximation ``opener_pick_evaluation`` itself uses
    (every other feature, including ``total_line``, stays at its close-era
    value).
    """

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
    """Per-game paired sign-rule correctness, baseline vs the joint model's margin output.

    Both arms are graded by the SAME sign rule (``predicted_residual > 0``,
    ``docs/opener_evaluation.md``'s predeclared historical record) so the
    comparison isolates the feature-set/estimator change, not a rule choice.
    Positive ``delta`` = the joint arm is correct where the baseline is not
    (net of the reverse), matching this project's "positive favours the
    candidate" convention.
    """

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
    """Week-blocked bootstrap of the paired opener sign-rule accuracy delta.

    Reports the delta in ACCURACY POINTS (percentage points, matching
    ``weak_signals.py``'s ``accuracy_points`` unit convention) rather than a
    bare fraction.
    """

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
