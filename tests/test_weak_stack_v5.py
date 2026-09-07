"""V5 one-family contract, point-in-time regression and evaluation safeguards."""

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.fluview_production_feature import (
    FLUVIEW_AWAY_ASOF_COLUMN as COLUMN,
)
from nfl_ats.fluview_production_feature import (
    attach_fluview_away_asof_features,
)
from nfl_ats.margin import margin_feature_columns
from scripts.weak_stack_v5_opener_eval import assert_baseline, frozen_pick_null


def games():
    return pd.DataFrame(
        {
            "game_id": ["a", "b", "c"],
            "season": [2020] * 3,
            "week": [1, 2, 3],
            "gameday": ["2020-09-13", "2020-09-20", "2020-09-27"],
            "home_team": ["SEA"] * 3,
            "away_team": ["NE"] * 3,
            "location": ["Home", "Home", "Neutral"],
            "untouched": [3.0, 4.0, 5.0],
        }
    )


def raw():
    return pd.DataFrame(
        {
            "region": ["ma"] * 4,
            "epiweek": [202035, 202036, 202037, 202034],
            "release_date": pd.to_datetime(
                ["2020-09-04", "2020-09-08", "2020-09-14", "2020-09-14"]
            ),
            "ili": [1.0, 90.0, 3.0, 1000.0],
        }
    )


@pytest.mark.parametrize("target", ["margin", "market_residual"])
def test_one_family_only(target):
    assert margin_feature_columns(target, "weak_stack_v5") == (
        *margin_feature_columns(target, "weak_stack"),
        COLUMN,
    )


def test_release_cutoff_and_older_revision():
    base = games()
    enriched = attach_fluview_away_asof_features(base, fluview_raw=raw())
    pd.testing.assert_frame_equal(enriched[base.columns], base, check_exact=True)
    assert enriched[COLUMN].iloc[0] == 1.0  # same Tuesday release unavailable
    assert enriched[COLUMN].iloc[1] == 3.0  # old epiweek revision cannot overwrite newer week
    assert np.isnan(enriched[COLUMN].iloc[2])


def test_future_mutation_cannot_change_past_feature():
    before = raw().iloc[:1]
    original = attach_fluview_away_asof_features(games().iloc[:1], fluview_raw=before)
    future = raw().copy()
    future.loc[1:, "ili"] = 1e9
    future["issue"] = [202036, 209901, 209902, 209903]
    after = attach_fluview_away_asof_features(games().iloc[:1], fluview_raw=future)
    pd.testing.assert_frame_equal(original, after)


def test_absent_release_stays_missing():
    source = raw()
    source["release_date"] = pd.NaT
    result = attach_fluview_away_asof_features(games(), fluview_raw=source)
    assert result[COLUMN].isna().all()


def test_nondefault_index_preserved():
    base = games().set_axis([8, 2, 7])
    result = attach_fluview_away_asof_features(base, fluview_raw=raw())
    assert list(result.index) == [8, 2, 7]
    assert result[COLUMN].iloc[:2].tolist() == [1.0, 3.0]


def test_duplicate_key_fails():
    with pytest.raises(DataContractError, match="unique"):
        attach_fluview_away_asof_features(pd.concat([games(), games()]), fluview_raw=raw())


def test_null_freezes_picks_and_is_deterministic():
    frame = pd.DataFrame(
        {
            "season": [2020] * 4,
            "week": [1, 1, 2, 2],
            "actual_home_cover_open": [1.0, 0.0, 1.0, 0.0],
            "baseline_pick_home_pr": [True, False, True, False],
            "candidate_pick_home_pr": [True, False, True, False],
            "baseline_correct_open_pr": [1.0] * 4,
            "candidate_correct_open_pr": [1.0] * 4,
        }
    )
    copy = frame.copy(deep=True)
    result = frozen_pick_null(frame)
    assert result["mean"] == result["lower"] == result["upper"] == 0.0
    assert frozen_pick_null(frame) == result
    pd.testing.assert_frame_equal(frame, copy)


def test_baseline_guard_rejects_changed_pick():
    base = pd.DataFrame(
        {
            "game_id": ["a"],
            "tue_open_home_spread": [3.0],
            "home_cover_probability_at_open": [0.6],
            "pick_home_at_open_probability_rule": [True],
            "correct_at_open_probability_rule": [True],
        }
    )
    assert_baseline(base, base.copy())
    changed = base.assign(pick_home_at_open_probability_rule=False)
    with pytest.raises(AssertionError):
        assert_baseline(base, changed)
