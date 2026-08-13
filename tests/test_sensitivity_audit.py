from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


def _load_audit_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "sensitivity_audit.py"
    spec = importlib.util.spec_from_file_location("sensitivity_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sensitivity audit from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_audit = _load_audit_script()
_blocked_interval = _audit._blocked_interval
_evaluate_replica = _audit._evaluate_replica
_smoothed_probabilities = _audit._smoothed_probabilities
_validate_active_reproduction = _audit._validate_active_reproduction


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


def test_active_reproduction_canary_checks_all_three_invariants() -> None:
    rows = 2075
    frame = pd.DataFrame(
        {
            "ats_margin": np.r_[np.ones(1080), -np.ones(rows - 1080)],
            "baseline_yhat": np.zeros(rows),
            "predicted_market_residual": np.zeros(rows),
            "baseline_probability_0_0": np.full(rows, 0.6),
            "home_cover_probability": np.full(rows, 0.6),
        }
    )
    assert _validate_active_reproduction(frame, prediction_rows=rows) == (0.0, 0.0, 1080)

    broken = frame.copy()
    broken.loc[0, "home_cover_probability"] = 0.59
    with pytest.raises(RuntimeError, match="probability_error"):
        _validate_active_reproduction(broken, prediction_rows=rows)
