from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.cfb_audit import (
    _blocked_interval,
    _evaluate_replica,
    _smoothed_probabilities,
    _validate_benchmark_reproduction,
    run_cfb_sensitivity_audit,
    summarize_cfb_sensitivity_details,
)
from nfl_ats.cfb_benchmark import cfb_walk_forward_benchmark


def test_smoothed_probabilities_use_empirical_residual_distribution() -> None:
    probabilities = _smoothed_probabilities(
        np.array([0.0, 1.0, -1.0]),
        np.array([-2.0, -0.5, 0.5, 2.0]),
    )
    np.testing.assert_allclose(probabilities, [0.5, 0.7, 0.3])


def test_replica_scoring_uses_cover_probability_not_raw_center() -> None:
    predictions = pd.DataFrame(
        {
            "season": [2022, 2022, 2023, 2023],
            "week": [1, 2, 1, 2],
            "ats_margin": [1.0, -1.0, 2.0, -2.0],
            "signal_0": [0.0, 0.0, 0.0, 0.0],
            "baseline_probability_0_0": [0.6, 0.4, 0.6, 0.4],
            "signal_probability_0_0": [0.4, 0.6, 0.6, 0.4],
            "permuted_probability_0_0": [0.6, 0.4, 0.6, 0.4],
        }
    )

    result = _evaluate_replica(
        predictions,
        replica=0,
        effect=0.0,
        samples=100,
        seed=7,
    )

    assert result["games"] == 4
    assert result["baseline_accuracy"] == 1.0
    assert result["signal_accuracy"] == 0.5
    assert result["signal_lift"] == -0.5
    assert result["permuted_lift"] == 0.0


def test_blocked_interval_preserves_a_constant_paired_improvement() -> None:
    frame = pd.DataFrame(
        {
            "season": [2022, 2022, 2023, 2023],
            "week": [1, 2, 1, 2],
            "baseline_correct": [False, False, False, False],
            "candidate_correct": [True, True, True, True],
        }
    )

    for block in ("week", "season"):
        estimate, lower, upper = _blocked_interval(
            frame,
            candidate="candidate",
            block=block,
            samples=100,
            seed=13,
        )
        assert estimate == pytest.approx(1.0)
        assert lower == pytest.approx(1.0)
        assert upper == pytest.approx(1.0)


def test_benchmark_reproduction_canary_checks_all_invariants() -> None:
    rows = 200
    frame = pd.DataFrame(
        {
            "ats_margin": np.r_[np.ones(120), -np.ones(rows - 120)],
            "baseline_yhat": np.zeros(rows),
            "predicted_market_residual": np.zeros(rows),
            "baseline_probability_0_0": np.full(rows, 0.6),
            "home_cover_probability": np.full(rows, 0.6),
        }
    )
    assert _validate_benchmark_reproduction(frame, prediction_rows=rows) == (0.0, 0.0, 120)

    broken = frame.copy()
    broken.loc[0, "home_cover_probability"] = 0.59
    with pytest.raises(RuntimeError, match="probability_error"):
        _validate_benchmark_reproduction(broken, prediction_rows=rows)

    shifted = frame.copy()
    shifted.loc[0, "predicted_market_residual"] = 1.0
    with pytest.raises(RuntimeError, match="prediction_error"):
        _validate_benchmark_reproduction(shifted, prediction_rows=rows)

    with pytest.raises(RuntimeError, match="rows"):
        _validate_benchmark_reproduction(frame, prediction_rows=rows + 1)


@pytest.mark.full  # ENG-11: asserts exact reproduction of a full benchmark build
def test_audit_reproduces_the_benchmark_exactly(cfb_features_frame: pd.DataFrame) -> None:
    benchmark = cfb_walk_forward_benchmark(
        cfb_features_frame, start_season=2014, end_season=2014, min_train_games=50
    )
    result = run_cfb_sensitivity_audit(
        cfb_features_frame,
        benchmark.predictions,
        replicas=2,
        bootstrap_samples=100,
        seed=11,
        start_season=2014,
        min_train_games=50,
    )
    metadata = result.metadata
    assert metadata["benchmark_prediction_reproduction_max_absolute_error"] <= 1e-9
    assert metadata["benchmark_probability_reproduction_max_absolute_error"] <= 1e-12
    assert metadata["detection_window"] == "clean_core"
    assert metadata["prediction_rows"] == metadata["clean_core_rows"]

    assert len(result.details) == 4 * 2
    assert len(result.summary) == 4
    assert result.summary["replicas"].eq(2).all()
    zero_effect = result.summary.loc[result.summary["effect_points_per_sd"].eq(0.0)].iloc[0]
    assert zero_effect["mean_signal_lift"] == pytest.approx(0.0, abs=0.05)


@pytest.mark.full  # ENG-11: dominates --durations; full benchmark determinism audit
def test_audit_fails_when_benchmark_predictions_differ(
    cfb_features_frame: pd.DataFrame,
) -> None:
    benchmark = cfb_walk_forward_benchmark(
        cfb_features_frame, start_season=2014, end_season=2014, min_train_games=50
    )
    corrupted = benchmark.predictions.copy()
    residual_rows = corrupted["method"].eq("market_residual")
    first_index = corrupted.loc[residual_rows].index[0]
    corrupted.loc[first_index, "home_cover_probability"] = 0.999
    with pytest.raises(RuntimeError, match="did not reproduce"):
        run_cfb_sensitivity_audit(
            cfb_features_frame,
            corrupted,
            replicas=1,
            bootstrap_samples=100,
            seed=11,
            start_season=2014,
            min_train_games=50,
        )


def test_audit_input_contracts(cfb_features_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="replicas"):
        run_cfb_sensitivity_audit(
            cfb_features_frame, cfb_features_frame, replicas=0, bootstrap_samples=100
        )
    with pytest.raises(ValueError, match="bootstrap samples"):
        run_cfb_sensitivity_audit(
            cfb_features_frame, cfb_features_frame, replicas=1, bootstrap_samples=10
        )


def test_summarize_details_counts_detections() -> None:
    details = pd.DataFrame(
        {
            "replica": [0, 1, 0, 1],
            "effect_points_per_sd": [0.0, 0.0, 1.0, 1.0],
            "baseline_accuracy": [0.5, 0.5, 0.5, 0.5],
            "signal_accuracy": [0.5, 0.5, 0.6, 0.7],
            "signal_lift": [0.0, -0.01, 0.1, 0.2],
            "signal_week_lower": [-0.02, -0.03, 0.05, 0.1],
            "signal_season_lower": [-0.02, -0.03, -0.01, 0.1],
            "permuted_lift": [0.0, 0.0, 0.0, 0.0],
            "permuted_week_lower": [-0.02, -0.02, -0.02, -0.02],
            "permuted_season_lower": [-0.02, -0.02, -0.02, -0.02],
        }
    )
    summary = summarize_cfb_sensitivity_details(details)
    row = summary.loc[summary["effect_points_per_sd"].eq(1.0)].iloc[0]
    assert row["week_detected_replicas"] == 2
    assert row["season_detected_replicas"] == 1
    assert row["positive_signal_replicas"] == 2
    assert row["permuted_week_false_positives"] == 0
    zero = summary.loc[summary["effect_points_per_sd"].eq(0.0)].iloc[0]
    assert zero["week_detected_replicas"] == 0
