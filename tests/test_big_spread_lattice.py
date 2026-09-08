import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from big_spread_lattice_opener_eval import ARMS, hybrid, verify_replay

from nfl_ats.conditional_margin import predict_conditional_margin


def fixtures():
    history = pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(200)],
            "season": 2020,
            "week": 1,
            "gameday": pd.Timestamp("2020-09-10"),
            "predicted_margin": np.tile([3.0, 7.0, 10.0, 14.0], 50),
            "spread_line": np.tile([3.0, 7.0, 10.0, 14.0], 50),
            "result": np.tile([3.0, 7.0, 10.0, 14.0], 50),
        }
    )
    target = pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d", "e"],
            "season": 2020,
            "week": 2,
            "gameday": pd.Timestamp("2020-09-17"),
            "predicted_margin": [3.0, 5.0, 7.0, 9.0, 12.0],
            "spread_line": [3.0, 5.0, 7.0, 9.0, 12.0],
        }
    )
    return history, target


@pytest.mark.parametrize("method", ARMS.values())
def test_small_exact_big_matches_frozen_k(method):
    history, target = fixtures()
    served = np.array([0.51, 0.49, 0.6, 0.4, 0.6])
    pushes = np.array([0.08, 0.04, 0.02, 0.0, 0.02])
    mapped = hybrid(history, target, served, method, pushes)
    np.testing.assert_array_equal(mapped.home_cover_probability[:2], served[:2])
    np.testing.assert_array_equal(mapped.push_probability[:2], pushes[:2])
    expected = predict_conditional_margin(history, target.iloc[2:], method=method)
    np.testing.assert_array_equal(
        mapped.home_cover_probability[2:], expected.home_cover_probability
    )
    np.testing.assert_array_equal(mapped.push_probability[2:], expected.push_probability)
    assert mapped.push_probability.iloc[2] > 0


@pytest.mark.parametrize("method", ARMS.values())
def test_future_same_week_old_history_cannot_leak(method):
    history, target = fixtures()
    expected = hybrid(history, target, np.full(5, 0.51), method)
    extra = []
    for season, week, date in [
        (2020, 2, "2020-09-16"),
        (2020, 3, "2020-09-24"),
        (2015, 1, "2015-09-10"),
    ]:
        rows = history.copy()
        rows["game_id"] += f"{season}_{week}"
        rows["season"], rows["week"], rows["gameday"], rows["result"] = (
            season,
            week,
            pd.Timestamp(date),
            -99.0,
        )
        extra.append(rows)
    actual = hybrid(pd.concat([history, *extra]), target, np.full(5, 0.51), method)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("method", ARMS.values())
def test_corrected_center_changes_weights_not_atoms(method):
    history, target = fixtures()
    before = hybrid(history, target, np.full(5, 0.51), method)
    target["predicted_margin"] += 3
    after = hybrid(history, target, np.full(5, 0.51), method)
    assert after.home_cover_probability.iloc[3] > before.home_cover_probability.iloc[3]
    np.testing.assert_array_equal(
        after.home_cover_probability[:2], before.home_cover_probability[:2]
    )


def test_small_buckets_do_not_require_lattice_history():
    history, target = fixtures()
    actual = hybrid(history.iloc[:0], target.iloc[:2], np.array([0.51, 0.49]), ARMS["M1"])
    np.testing.assert_array_equal(actual.home_cover_probability, [0.51, 0.49])


@pytest.mark.parametrize("candidate", [np.array([0.500001]), np.array([np.nan])])
def test_replay_gate_stops_drift_or_missing_probability(candidate):
    with pytest.raises(ValueError, match="STOP"):
        verify_replay(candidate, np.array([0.5]))


def test_replay_gate_allows_roundoff():
    assert verify_replay(np.array([0.5 + 1e-15]), np.array([0.5])) < 1e-9
