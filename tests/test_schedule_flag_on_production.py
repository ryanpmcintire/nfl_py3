from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.schedule_flag_features import (
    HOME_THURSDAY_COLUMN,
    MNF_ROAD_SHORT_WEEK_COLUMN,
    POST_OT_FATIGUE_COLUMN,
    attach_home_thursday_features,
    attach_mnf_road_short_week_features,
    attach_post_ot_fatigue_features,
    derive_home_thursday_features,
    derive_mnf_road_short_week_features,
    derive_post_ot_fatigue_features,
)


def _game(
    game_id: str, season: int, gameday: str, weekday: str, home: str, away: str, ot: float
) -> dict:
    return {
        "game_id": game_id,
        "season": season,
        "gameday": gameday,
        "weekday": weekday,
        "home_team": home,
        "away_team": away,
        "overtime": ot,
    }


def _schedule(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _post_ot_schedule() -> pd.DataFrame:
    return _schedule(
        [
            _game("g0", 2020, "2020-09-13", "Sunday", "DDD", "FFF", 1.0),
            _game("g1", 2020, "2020-09-10", "Thursday", "AAA", "BBB", 0.0),
            _game("g3", 2020, "2020-09-27", "Sunday", "AAA", "DDD", 0.0),
            _game("gX", 2020, "2020-09-13", "Sunday", "CCC", "ZZZ", 1.0),
            _game("g4", 2020, "2020-09-27", "Sunday", "CCC", "EEE", 0.0),
            _game("gY", 2020, "2020-09-20", "Sunday", "QQQ", "RRR", 1.0),
            _game("gZ", 2020, "2020-09-20", "Sunday", "SSS", "TTT", 1.0),
            _game("g5", 2020, "2020-09-27", "Sunday", "QQQ", "SSS", 0.0),
        ]
    )


def test_post_ot_sign_convention_covers_all_states() -> None:
    derived = derive_post_ot_fatigue_features(_post_ot_schedule()).set_index("game_id")
    assert derived.loc["g3", POST_OT_FATIGUE_COLUMN] == 1.0
    assert derived.loc["g4", POST_OT_FATIGUE_COLUMN] == -1.0
    assert derived.loc["g5", POST_OT_FATIGUE_COLUMN] == 0.0
    assert derived.loc["g1", POST_OT_FATIGUE_COLUMN] == 0.0


def test_post_ot_week_one_has_no_prior_game_and_is_zero_not_nan() -> None:

    derived = derive_post_ot_fatigue_features(_post_ot_schedule()).set_index("game_id")
    assert derived.loc["g0", POST_OT_FATIGUE_COLUMN] == 0.0
    assert not pd.isna(derived.loc["g0", POST_OT_FATIGUE_COLUMN])


def test_post_ot_never_crosses_a_season_boundary() -> None:

    schedule = _schedule(
        [
            _game("s1", 2020, "2020-12-20", "Sunday", "AAA", "BBB", 1.0),
            _game("s2", 2021, "2021-09-12", "Sunday", "CCC", "AAA", 0.0),
        ]
    )
    derived = derive_post_ot_fatigue_features(schedule).set_index("game_id")
    assert derived.loc["s2", POST_OT_FATIGUE_COLUMN] == 0.0


def _mnf_road_schedule() -> pd.DataFrame:
    return _schedule(
        [
            _game("m1", 2020, "2020-09-21", "Monday", "CCC", "AAA", 0.0),
            _game("m2", 2020, "2020-09-27", "Sunday", "AAA", "DDD", 0.0),
            _game("m3", 2020, "2020-09-21", "Monday", "EEE", "BBB", 0.0),
            _game("m4", 2020, "2020-09-27", "Sunday", "FFF", "BBB", 0.0),
            _game("m5", 2020, "2020-09-21", "Monday", "GGG", "HHH", 0.0),
            _game("m6", 2020, "2020-09-27", "Sunday", "GGG", "III", 0.0),
            _game("m7", 2020, "2020-09-21", "Monday", "KKK", "JJJ", 0.0),
            _game("m8", 2020, "2020-10-11", "Sunday", "JJJ", "LLL", 0.0),
        ]
    )


def test_mnf_road_home_qualifies_is_negative() -> None:
    derived = derive_mnf_road_short_week_features(_mnf_road_schedule()).set_index("game_id")
    assert derived.loc["m2", MNF_ROAD_SHORT_WEEK_COLUMN] == -1.0


def test_mnf_road_away_qualifies_is_positive() -> None:
    derived = derive_mnf_road_short_week_features(_mnf_road_schedule()).set_index("game_id")
    assert derived.loc["m4", MNF_ROAD_SHORT_WEEK_COLUMN] == 1.0


def test_mnf_road_home_game_after_monday_does_not_qualify() -> None:

    derived = derive_mnf_road_short_week_features(_mnf_road_schedule()).set_index("game_id")
    assert derived.loc["m6", MNF_ROAD_SHORT_WEEK_COLUMN] == 0.0


def test_mnf_road_requires_exactly_six_days_not_just_monday_then_sunday() -> None:

    derived = derive_mnf_road_short_week_features(_mnf_road_schedule()).set_index("game_id")
    assert derived.loc["m8", MNF_ROAD_SHORT_WEEK_COLUMN] == 0.0


def test_home_thursday_flags_every_thursday_game_unsigned() -> None:
    schedule = _schedule(
        [
            _game("t1", 2020, "2020-09-10", "Thursday", "AAA", "BBB", 0.0),
            _game("t2", 2020, "2020-09-13", "Sunday", "CCC", "DDD", 0.0),
        ]
    )
    derived = derive_home_thursday_features(schedule).set_index("game_id")
    assert derived.loc["t1", HOME_THURSDAY_COLUMN] == 1.0
    assert derived.loc["t2", HOME_THURSDAY_COLUMN] == 0.0


def test_flags_are_invariant_to_a_games_own_outcome() -> None:

    schedule = _post_ot_schedule()
    baseline = {
        "post_ot": derive_post_ot_fatigue_features(schedule).set_index("game_id"),
    }
    mutated = schedule.copy()
    target = "g3"
    mutated.loc[mutated["game_id"] == target, "overtime"] = 1.0
    after = derive_post_ot_fatigue_features(mutated).set_index("game_id")
    assert (
        after.loc[target, POST_OT_FATIGUE_COLUMN]
        == baseline["post_ot"].loc[target, POST_OT_FATIGUE_COLUMN]
    )

    mnf_schedule = _mnf_road_schedule()
    mnf_before = derive_mnf_road_short_week_features(mnf_schedule).set_index("game_id")
    mnf_mutated = mnf_schedule.copy()
    mnf_target = "m2"
    mnf_mutated.loc[mnf_mutated["game_id"] == mnf_target, "overtime"] = 1.0
    mnf_after = derive_mnf_road_short_week_features(mnf_mutated).set_index("game_id")
    assert (
        mnf_after.loc[mnf_target, MNF_ROAD_SHORT_WEEK_COLUMN]
        == mnf_before.loc[mnf_target, MNF_ROAD_SHORT_WEEK_COLUMN]
    )

    thu_schedule = _schedule([_game("t1", 2020, "2020-09-10", "Thursday", "AAA", "BBB", 0.0)])
    thu_before = derive_home_thursday_features(thu_schedule).set_index("game_id")
    thu_mutated = thu_schedule.copy()
    thu_mutated.loc[thu_mutated["game_id"] == "t1", "overtime"] = 1.0
    thu_after = derive_home_thursday_features(thu_mutated).set_index("game_id")
    assert thu_after.loc["t1", HOME_THURSDAY_COLUMN] == thu_before.loc["t1", HOME_THURSDAY_COLUMN]


def test_a_later_games_flag_may_legitimately_depend_on_an_earlier_result() -> None:

    schedule = _post_ot_schedule()
    before = derive_post_ot_fatigue_features(schedule).set_index("game_id")
    assert before.loc["g3", POST_OT_FATIGUE_COLUMN] == 1.0

    mutated = schedule.copy()
    mutated.loc[mutated["game_id"] == "g0", "overtime"] = 0.0
    after = derive_post_ot_fatigue_features(mutated).set_index("game_id")
    assert after.loc["g3", POST_OT_FATIGUE_COLUMN] == 0.0


def test_attach_is_purely_additive_for_all_three() -> None:
    schedule = _post_ot_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], "some_existing_feature": 1.0})

    widened = attach_post_ot_fatigue_features(features, schedule=schedule)
    new_columns = sorted(set(widened.columns) - set(features.columns))
    assert new_columns == [POST_OT_FATIGUE_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)
    assert list(widened.index) == list(features.index)

    mnf_schedule = _mnf_road_schedule()
    mnf_features = pd.DataFrame({"game_id": mnf_schedule["game_id"]})
    mnf_widened = attach_mnf_road_short_week_features(mnf_features, schedule=mnf_schedule)
    assert sorted(set(mnf_widened.columns) - set(mnf_features.columns)) == [
        MNF_ROAD_SHORT_WEEK_COLUMN
    ]

    thu_schedule = _schedule([_game("t1", 2020, "2020-09-10", "Thursday", "AAA", "BBB", 0.0)])
    thu_features = pd.DataFrame({"game_id": thu_schedule["game_id"]})
    thu_widened = attach_home_thursday_features(thu_features, schedule=thu_schedule)
    assert sorted(set(thu_widened.columns) - set(thu_features.columns)) == [HOME_THURSDAY_COLUMN]


def test_attach_requires_the_join_key() -> None:
    schedule = _post_ot_schedule()
    features = pd.DataFrame({"not_game_id": schedule["game_id"]})
    with pytest.raises(DataContractError, match="game_id"):
        attach_post_ot_fatigue_features(features, schedule=schedule)


def test_attach_refuses_to_overwrite_an_existing_column() -> None:
    schedule = _post_ot_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], POST_OT_FATIGUE_COLUMN: 0.0})
    with pytest.raises(DataContractError, match=POST_OT_FATIGUE_COLUMN):
        attach_post_ot_fatigue_features(features, schedule=schedule)


def test_derive_requires_every_schedule_column() -> None:
    schedule = _post_ot_schedule().drop(columns=["overtime"])
    with pytest.raises(DataContractError, match="overtime"):
        derive_post_ot_fatigue_features(schedule)
