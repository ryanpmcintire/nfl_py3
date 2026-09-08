import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from home_side_prior_opener_eval import ARMS, decision, offsets, replay_gate

from nfl_ats.home_side_location import prior_rows_before


def test_side_cells_and_prior():
    prior = pd.DataFrame(
        {
            "spread_line": [11.0, -11.0, 3.0],
            "result": [21.0, 9.0, 100.0],
            "point_incumbent": [11.0, -11.0, 0.0],
        }
    )
    lines = pd.Series([11.0, -11.0, 3.0, -3.0, 7.0])
    np.testing.assert_allclose(offsets(prior, lines, 100), [30 / 102, 30 / 102, 0, 0, 0])
    np.testing.assert_allclose(offsets(prior, lines, 50), [30 / 52, 30 / 52, 0, 0, 0])


def test_future_same_week_and_old_results_cannot_leak():
    stream = pd.DataFrame(
        {
            "season": [2019, 2020, 2025, 2025, 2026],
            "week": [1, 1, 1, 2, 1],
            "spread_line": [11.0] * 5,
            "point_incumbent": [0.0] * 5,
            "result": [999.0, 10.0, 20.0, 999.0, 999.0],
        }
    )
    before = prior_rows_before(stream, 2025, 2)
    np.testing.assert_allclose(offsets(before, pd.Series([11.0]), 100), [30 / 102])
    stream.loc[[0, 3, 4], "result"] = -999999.0
    np.testing.assert_allclose(
        offsets(prior_rows_before(stream, 2025, 2), pd.Series([11.0]), 100), [30 / 102]
    )


def test_missing_results_and_points_do_not_count():
    frame = pd.DataFrame(
        {
            "spread_line": [-8.0, -8.0, -8.0],
            "result": [10.0, np.nan, 99.0],
            "point_incumbent": [0.0, 0.0, np.nan],
        }
    )
    np.testing.assert_allclose(offsets(frame, pd.Series([-8.0, 8.0]), 100), [10 / 101, 10 / 101])


def test_prior_100_exact_production_parity_and_frozen_arms():
    from nfl_ats.home_side_location import fit_home_side_offsets

    assert ARMS == {"S5a": 50.0, "S5b": 25.0, "S5c": 200.0}
    prior = pd.DataFrame(
        {
            "spread_line": [3, 5, 7, -7, 8, -11],
            "result": [2, 8, 14, 3, 4, 7],
            "point_incumbent": [1, 2, 3, 4, 5, 6],
        }
    )
    lines = pd.Series([1, 5, 7, -7, 8, -8, 11, -11])
    np.testing.assert_array_equal(
        offsets(prior, lines, 100), fit_home_side_offsets(prior).offset_for(lines)
    )
    for weight in ARMS.values():
        expected = offsets(prior, lines, 100)
        for i, line in enumerate(lines):
            n = 2 if abs(line) == 7 else 1
            expected[i] *= (n + 100) / (n + weight)
        np.testing.assert_allclose(offsets(prior, lines, weight), expected)


def test_replay_fails_closed():
    import pytest

    assert replay_gate(np.array([0.5]), np.array([0.5])) == 0
    for bad in [0.50000001, np.nan]:
        with pytest.raises(ValueError, match="STOP"):
            replay_gate(np.array([bad]), np.array([0.5]))


def test_decision_uses_top_card_and_brier_guard():
    cells = {}
    for arm in ARMS:
        cells[f"s5_{arm.lower()}_overall_card"] = {
            "candidate_accuracy": 0.55,
            "probability_positive": 0.6,
        }
        cells[f"s5_{arm.lower()}_overall_brier"] = {"delta": 0.001, "probability_positive": 0.8}
    assert decision(cells) == "S5a"
    cells["s5_s5a_overall_brier"].update(delta=-0.001, probability_positive=0.29)
    assert decision(cells) == "S3"
    cells["s5_s5a_overall_brier"]["probability_positive"] = 0.3
    assert decision(cells) == "S5a"
    cells["s5_s5a_overall_card"]["probability_positive"] = 0.5
    assert decision(cells) == "S3"
