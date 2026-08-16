"""Positive-control sensitivity audit for the CFB market-residual benchmark.

This ports the NFL evaluator sensitivity audit (``scripts/sensitivity_audit.py``)
to the CFB-only benchmark: it measures whether the exact chronological weekly
Ridge path used by ``cfb_benchmark`` recovers deliberately injected signal.
The synthetic features are generated independently of real outcomes, then
given a known counterfactual effect on the ATS margin (0.5, 1, and 2 points
per feature standard deviation), repeated across independent signal draws,
with permuted copies as negative controls. Before any effect is scored the
audit must reproduce the benchmark's own market-residual predictions exactly;
otherwise it measures a different evaluator and fails.

Detection power is reported on the clean-core window (2012-2019, 2021-2025),
the same sample the benchmark headline uses, with week- and season-blocked
bootstrap intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_DISTRIBUTION_FRACTION,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    cfb_evaluation_window,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.margin import make_margin_estimator

CFB_AUDIT_EFFECTS = (0.0, 0.5, 1.0, 2.0)
CFB_AUDIT_REPLICAS = 8
CFB_AUDIT_BOOTSTRAP_SAMPLES = 2_000
CFB_AUDIT_SEED = 20260816

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CfbSensitivityAuditResult:
    details: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, Any]


def _blocked_interval(
    frame: pd.DataFrame,
    *,
    candidate: str,
    block: str,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    candidate_correct = frame[f"{candidate}_correct"].to_numpy(dtype=float)
    baseline_correct = frame["baseline_correct"].to_numpy(dtype=float)
    differences = candidate_correct - baseline_correct
    point = float(differences.mean())
    keys = (
        frame["season"].astype(str) + "-" + frame["week"].astype(str)
        if block == "week"
        else frame["season"].astype(str)
    )
    grouped = (
        pd.DataFrame({"key": keys, "difference": differences})
        .groupby("key", sort=True)["difference"]
        .agg(["sum", "size"])
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(samples, len(grouped)))
    draws = grouped["sum"].to_numpy()[indices].sum(axis=1) / grouped["size"].to_numpy()[
        indices
    ].sum(axis=1)
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _evaluate_replica(
    predictions: pd.DataFrame,
    *,
    replica: int,
    effect: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    actual_margin = predictions["ats_margin"] + effect * predictions[f"signal_{replica}"]
    keep = actual_margin.ne(0.0)
    frame = predictions.loc[keep].copy()
    actual_home = actual_margin.loc[keep].gt(0.0)
    effect_index = CFB_AUDIT_EFFECTS.index(effect)
    frame["baseline_correct"] = (
        frame[f"baseline_probability_{effect_index}_{replica}"].ge(0.5).eq(actual_home)
    )
    frame["signal_correct"] = (
        frame[f"signal_probability_{effect_index}_{replica}"].ge(0.5).eq(actual_home)
    )
    frame["permuted_correct"] = (
        frame[f"permuted_probability_{effect_index}_{replica}"].ge(0.5).eq(actual_home)
    )
    row: dict[str, Any] = {
        "replica": replica,
        "effect_points_per_sd": effect,
        "games": len(frame),
        "baseline_accuracy": float(frame["baseline_correct"].mean()),
        "signal_accuracy": float(frame["signal_correct"].mean()),
        "permuted_accuracy": float(frame["permuted_correct"].mean()),
    }
    for candidate in ("signal", "permuted"):
        for block in ("week", "season"):
            point, lower, upper = _blocked_interval(
                frame,
                candidate=candidate,
                block=block,
                samples=samples,
                seed=seed + replica * 101 + (0 if block == "week" else 10_000),
            )
            row[f"{candidate}_lift"] = point
            row[f"{candidate}_{block}_lower"] = lower
            row[f"{candidate}_{block}_upper"] = upper
    return row


def _fit_components(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    feature_columns: list[str],
    target: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """One weekly fit reproducing ``fit_cfb_residual_model`` numerically.

    Ridge is linear, so fitting the multi-output target ``[ats_margin,
    signal]`` yields, for any effect ``e``, the prediction and residual
    distribution of the counterfactual target ``ats_margin + e * signal`` as
    the matching linear combination of columns.
    """

    distribution_rows = int(len(training) * CFB_BENCHMARK_DISTRIBUTION_FRACTION)
    split = len(training) - distribution_rows
    temporary = make_margin_estimator("ridge", ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA)
    temporary.fit(training.iloc[:split][feature_columns], target[:split])
    residuals = target[split:] - np.asarray(
        temporary.predict(training.iloc[split:][feature_columns]), dtype=float
    )
    final = make_margin_estimator("ridge", ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA)
    final.fit(training[feature_columns], target)
    prediction = np.asarray(final.predict(scoring[feature_columns]), dtype=float)
    return prediction, residuals


def _smoothed_probabilities(center: FloatArray, residuals: FloatArray) -> FloatArray:
    successes = (residuals[None, :] > -center[:, None]).sum(axis=1)
    return np.asarray((successes + 0.5) / (len(residuals) + 1.0), dtype=np.float64)


def _validate_benchmark_reproduction(
    reproduction: pd.DataFrame, *, prediction_rows: int
) -> tuple[float, float, int]:
    """Fail unless the audit exactly reconstructs the benchmark evaluation."""

    prediction_error = float(
        np.abs(reproduction["baseline_yhat"] - reproduction["predicted_market_residual"]).max()
    )
    probability_error = float(
        np.abs(
            reproduction["baseline_probability_0_0"] - reproduction["home_cover_probability"]
        ).max()
    )
    nonpush = reproduction.loc[reproduction["ats_margin"].ne(0.0)]
    correct = int(
        nonpush["baseline_probability_0_0"].ge(0.5).eq(nonpush["ats_margin"].gt(0.0)).sum()
    )
    expected_correct = int(
        nonpush["home_cover_probability"].ge(0.5).eq(nonpush["ats_margin"].gt(0.0)).sum()
    )
    if (
        len(reproduction) != prediction_rows
        or prediction_error > 1e-9
        or probability_error > 1e-12
        or correct != expected_correct
    ):
        raise RuntimeError(
            "CFB positive-control evaluator did not reproduce the benchmark evaluation: "
            f"rows={len(reproduction)}/{prediction_rows}, "
            f"prediction_error={prediction_error}, probability_error={probability_error}, "
            f"correct={correct}/{expected_correct}"
        )
    return prediction_error, probability_error, correct


def run_cfb_sensitivity_audit(
    features: pd.DataFrame,
    benchmark_predictions: pd.DataFrame,
    *,
    replicas: int = CFB_AUDIT_REPLICAS,
    bootstrap_samples: int = CFB_AUDIT_BOOTSTRAP_SAMPLES,
    seed: int = CFB_AUDIT_SEED,
    start_season: int = CFB_BENCHMARK_START_SEASON,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
) -> CfbSensitivityAuditResult:
    """Measure the CFB benchmark's power to detect known injected effects."""

    if replicas < 1 or bootstrap_samples < 100:
        raise ValueError("replicas must be positive and bootstrap samples must be at least 100")
    completed = features.loc[
        pd.to_numeric(features["result"], errors="coerce").notna()
        & pd.to_numeric(features["ats_margin"], errors="coerce").notna()
    ].copy()
    completed["gameday"] = pd.to_datetime(completed["gameday"], errors="raise")
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    completed["evaluation_window"] = completed["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    rng = np.random.default_rng(seed)
    signals = rng.standard_normal((len(completed), replicas))
    permutations = np.column_stack(
        [rng.permutation(signals[:, replica]) for replica in range(replicas)]
    )
    for replica in range(replicas):
        completed[f"signal_{replica}"] = signals[:, replica]
        completed[f"permuted_{replica}"] = permutations[:, replica]

    base_columns = list(CFB_MODEL_FEATURE_COLUMNS)
    target_columns = ["ats_margin", *[f"signal_{i}" for i in range(replicas)]]
    test = completed.loc[completed["season"].ge(start_season)]
    batches: list[pd.DataFrame] = []
    for (_, _), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        target = training[target_columns].to_numpy(dtype=float)
        baseline_prediction, baseline_residuals = _fit_components(
            training, weekly_games, base_columns, target
        )
        batch = weekly_games[
            [
                "game_id",
                "season",
                "week",
                "gameday",
                "evaluation_window",
                "ats_margin",
                *target_columns[1:],
            ]
        ].copy()
        batch["baseline_yhat"] = baseline_prediction[:, 0]
        for replica in range(replicas):
            for effect_index, effect in enumerate(CFB_AUDIT_EFFECTS):
                baseline_center = (
                    baseline_prediction[:, 0] + effect * baseline_prediction[:, replica + 1]
                )
                baseline_residual = (
                    baseline_residuals[:, 0] + effect * baseline_residuals[:, replica + 1]
                )
                batch[f"baseline_probability_{effect_index}_{replica}"] = _smoothed_probabilities(
                    baseline_center, baseline_residual
                )
            for kind in ("signal", "permuted"):
                synthetic_column = f"{kind}_{replica}"
                prediction, residuals = _fit_components(
                    training,
                    weekly_games,
                    [*base_columns, synthetic_column],
                    target[:, [0, replica + 1]],
                )
                for effect_index, effect in enumerate(CFB_AUDIT_EFFECTS):
                    center = prediction[:, 0] + effect * prediction[:, 1]
                    residual = residuals[:, 0] + effect * residuals[:, 1]
                    batch[f"{kind}_probability_{effect_index}_{replica}"] = _smoothed_probabilities(
                        center, residual
                    )
        batches.append(batch)
    if not batches:
        raise ValueError("No CFB audit week had enough prior training games")
    predictions = pd.concat(batches, ignore_index=True)

    active = benchmark_predictions.loc[benchmark_predictions["method"].eq("market_residual")]
    reproduction = predictions.merge(
        active[["game_id", "predicted_market_residual", "home_cover_probability"]],
        on="game_id",
        validate="one_to_one",
    )
    max_prediction_error, max_probability_error, reproduced_correct = (
        _validate_benchmark_reproduction(reproduction, prediction_rows=len(predictions))
    )

    clean_core = predictions.loc[predictions["evaluation_window"].eq("clean_core")]
    if clean_core.empty:
        raise ValueError("No clean-core predictions available for the sensitivity audit")
    rows = [
        _evaluate_replica(
            clean_core,
            replica=replica,
            effect=effect,
            samples=bootstrap_samples,
            seed=seed,
        )
        for effect in CFB_AUDIT_EFFECTS
        for replica in range(replicas)
    ]
    details = pd.DataFrame(rows)
    summary = summarize_cfb_sensitivity_details(details)
    metadata = {
        "purpose": ("positive control only; synthetic target signal is not a football feature"),
        "target": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": CFB_BENCHMARK_RIDGE_ALPHA,
        "start_season": start_season,
        "min_train_games": min_train_games,
        "replicas": replicas,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "synthetic_effects_points_per_feature_sd": list(CFB_AUDIT_EFFECTS),
        "detection_window": "clean_core",
        "benchmark_prediction_reproduction_max_absolute_error": max_prediction_error,
        "benchmark_probability_reproduction_max_absolute_error": max_probability_error,
        "benchmark_classification_reproduction_correct": reproduced_correct,
        "prediction_rows": len(predictions),
        "clean_core_rows": len(clean_core),
        "nonpush_clean_core_games_at_zero_effect": int(clean_core["ats_margin"].ne(0).sum()),
    }
    return CfbSensitivityAuditResult(details=details, summary=summary, metadata=metadata)


def summarize_cfb_sensitivity_details(details: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replica rows into the detection-power table."""

    return (
        details.groupby("effect_points_per_sd", as_index=False)
        .agg(
            replicas=("replica", "size"),
            mean_baseline_accuracy=("baseline_accuracy", "mean"),
            mean_signal_accuracy=("signal_accuracy", "mean"),
            mean_signal_lift=("signal_lift", "mean"),
            min_signal_lift=("signal_lift", "min"),
            max_signal_lift=("signal_lift", "max"),
            positive_signal_replicas=("signal_lift", lambda value: int(value.gt(0).sum())),
            week_detected_replicas=("signal_week_lower", lambda value: int(value.gt(0).sum())),
            season_detected_replicas=(
                "signal_season_lower",
                lambda value: int(value.gt(0).sum()),
            ),
            mean_permuted_lift=("permuted_lift", "mean"),
            permuted_week_false_positives=(
                "permuted_week_lower",
                lambda value: int(value.gt(0).sum()),
            ),
            permuted_season_false_positives=(
                "permuted_season_lower",
                lambda value: int(value.gt(0).sum()),
            ),
        )
        .sort_values("effect_points_per_sd", ignore_index=True)
    )
