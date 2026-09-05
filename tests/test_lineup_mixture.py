"""Lineup mixture contracts: timing, outcome isolation and Gaussian noise."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from nfl_ats.data import DataContractError
from nfl_ats.lineup_mixture import lineup_draws, mixture_probability, paired_summary


def players():
    return pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "team": ["BUF", "BUF"],
            "gsis_id": ["a", "b"],
            "position": ["QB", "WR"],
            "position_group": ["skill", "skill"],
            "depth_rank": [1, 1],
            "decision_at": ["2024-09-08T20:00Z"] * 2,
            "depth_observed_at": ["2024-09-07T20:00Z"] * 2,
            "source_schema": ["daily"] * 2,
            "play_probability": [0.7, 0.4],
            "trailing4_snap_share": [1.0, 0.8],
            "played": [1, 0],
        }
    )


def test_post_decision_depth_does_not_change_scenarios():
    frame = players()
    baseline = lineup_draws(frame)[(2024, 1, "BUF")]
    later = frame.copy()
    later["depth_observed_at"] = "2024-09-08T20:01Z"
    later["play_probability"] = 0.0
    candidate = lineup_draws(pd.concat([frame, later]))[(2024, 1, "BUF")]
    for arm in baseline:
        np.testing.assert_array_equal(baseline[arm], candidate[arm])


def test_target_participation_only_changes_oracle():
    frame = players()
    baseline = lineup_draws(frame)[(2024, 1, "BUF")]
    frame["played"] = 1 - frame.played
    changed = lineup_draws(frame)[(2024, 1, "BUF")]
    for arm in ("mixture", "expected", "permutation"):
        np.testing.assert_array_equal(baseline[arm], changed[arm])
    assert not np.array_equal(baseline["oracle"], changed["oracle"])


def test_known_availability_and_bench_weights():
    frame = players()
    frame["play_probability"] = 0.0
    frame.loc[1, "depth_rank"] = 2
    draws = lineup_draws(frame)[(2024, 1, "BUF")]
    np.testing.assert_array_equal(draws["expected"], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(draws["mixture"], np.tile([0.0, 0.0, 1.0], (200, 1)))


def test_missing_snap_history_is_zero_and_order_is_stable():
    frame = players()
    frame.loc[0, "trailing4_snap_share"] = np.nan
    a = lineup_draws(frame)[(2024, 1, "BUF")]
    b = lineup_draws(frame.iloc[::-1])[(2024, 1, "BUF")]
    np.testing.assert_array_equal(a["mixture"], b["mixture"])
    assert np.all(a["mixture"][:, 2] == 0)


def test_zero_uncertainty_preserves_gaussian_residual_mean():
    result = mixture_probability(3.0, 2.0, -1.5, 13.0, np.zeros(200))
    assert result["probability"] == pytest.approx(ndtr(-0.5 / 13))
    assert result["scenario_sd"] == 0
    assert result["total_sd_increase"] == 0


def test_mixture_integrates_noise_and_widens_distribution():
    shifts = np.array([-4.0, 4.0])
    result = mixture_probability(1.0, 0.0, 0.0, 10.0, shifts)
    assert result["probability"] == pytest.approx((ndtr(-0.3) + ndtr(0.5)) / 2)
    assert result["scenario_sd"] == 4
    assert result["total_sd_increase"] == pytest.approx(np.hypot(10, 4) - 10)


def test_invalid_probability_fails_closed():
    frame = players()
    frame.loc[0, "play_probability"] = np.nan
    with pytest.raises(DataContractError):
        lineup_draws(frame)


def test_block_bootstrap_reports_positive_improvement():
    frame = pd.DataFrame({"season": [2020] * 3, "week": [1, 1, 2], "delta": [1.0, 1.0, 1.0]})
    result = paired_summary(frame, "delta", draws=200)
    assert result["effect"] == 1
    assert result["sample_blocks"] == 2
    assert result["probability_positive"] == 1
