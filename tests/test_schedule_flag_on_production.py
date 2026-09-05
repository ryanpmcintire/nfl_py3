"""Construction, sign-convention and leakage contracts for the three
LEAD-21/22/40 schedule flags, plus the on-production confirmation wrapper's
duck-typed reuse of ``scripts/on_production_opener_confirmation.py``.

Predeclared in ``docs/schedule_flag_battery.md``. Every fixture is built in
memory: these tests must pass in a fresh clone with no local data snapshots
(no schedules.parquet snapshot is ever read).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import schedule_flag_on_production as sfop  # noqa: E402

from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.margin import margin_feature_columns  # noqa: E402
from nfl_ats.schedule_flag_features import (  # noqa: E402
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


# ---------------------------------------------------------------------------
# LEAD-21: post-overtime fatigue
# ---------------------------------------------------------------------------


def _post_ot_schedule() -> pd.DataFrame:
    return _schedule(
        [
            # DDD's week-1 game goes to OT.
            _game("g0", 2020, "2020-09-13", "Sunday", "DDD", "FFF", 1.0),
            # AAA's week-1 game does NOT go to OT.
            _game("g1", 2020, "2020-09-10", "Thursday", "AAA", "BBB", 0.0),
            # g3: AAA (home, not post-OT) hosts DDD (away, post-OT from g0) -> +1.
            _game("g3", 2020, "2020-09-27", "Sunday", "AAA", "DDD", 0.0),
            # g4: CCC (home, post-OT via a separate OT game) hosts EEE (away, not) -> -1.
            _game("gX", 2020, "2020-09-13", "Sunday", "CCC", "ZZZ", 1.0),
            _game("g4", 2020, "2020-09-27", "Sunday", "CCC", "EEE", 0.0),
            # g5: both sides post-OT (two independent OT games in week 2) -> 0.
            _game("gY", 2020, "2020-09-20", "Sunday", "QQQ", "RRR", 1.0),
            _game("gZ", 2020, "2020-09-20", "Sunday", "SSS", "TTT", 1.0),
            _game("g5", 2020, "2020-09-27", "Sunday", "QQQ", "SSS", 0.0),
        ]
    )


def test_post_ot_sign_convention_covers_all_states() -> None:
    derived = derive_post_ot_fatigue_features(_post_ot_schedule()).set_index("game_id")
    assert derived.loc["g3", POST_OT_FATIGUE_COLUMN] == 1.0  # away (DDD) post-OT
    assert derived.loc["g4", POST_OT_FATIGUE_COLUMN] == -1.0  # home (CCC) post-OT
    assert derived.loc["g5", POST_OT_FATIGUE_COLUMN] == 0.0  # both post-OT
    assert derived.loc["g1", POST_OT_FATIGUE_COLUMN] == 0.0  # neither; also no prior game


def test_post_ot_week_one_has_no_prior_game_and_is_zero_not_nan() -> None:
    """A team's first in-season game cannot follow an in-season OT game."""

    derived = derive_post_ot_fatigue_features(_post_ot_schedule()).set_index("game_id")
    assert derived.loc["g0", POST_OT_FATIGUE_COLUMN] == 0.0
    assert not pd.isna(derived.loc["g0", POST_OT_FATIGUE_COLUMN])


def test_post_ot_never_crosses_a_season_boundary() -> None:
    """A team's week-1 game next season is not "post-OT" from last season's finale."""

    schedule = _schedule(
        [
            _game("s1", 2020, "2020-12-20", "Sunday", "AAA", "BBB", 1.0),
            _game("s2", 2021, "2021-09-12", "Sunday", "CCC", "AAA", 0.0),
        ]
    )
    derived = derive_post_ot_fatigue_features(schedule).set_index("game_id")
    assert derived.loc["s2", POST_OT_FATIGUE_COLUMN] == 0.0


# ---------------------------------------------------------------------------
# LEAD-22: Monday-night-road short week
# ---------------------------------------------------------------------------


def _mnf_road_schedule() -> pd.DataFrame:
    return _schedule(
        [
            # AAA plays away on Monday at CCC ...
            _game("m1", 2020, "2020-09-21", "Monday", "CCC", "AAA", 0.0),
            # ... then hosts DDD the following Sunday (6 days later) -> home (AAA) qualifies -> -1.
            _game("m2", 2020, "2020-09-27", "Sunday", "AAA", "DDD", 0.0),
            # BBB plays away on Monday at EEE ...
            _game("m3", 2020, "2020-09-21", "Monday", "EEE", "BBB", 0.0),
            # ... then plays AWAY again the following Sunday at FFF -> away (BBB) qualifies -> +1.
            _game("m4", 2020, "2020-09-27", "Sunday", "FFF", "BBB", 0.0),
            # GGG played HOME on Monday (not on the road) then plays Sunday -> does not qualify.
            _game("m5", 2020, "2020-09-21", "Monday", "GGG", "HHH", 0.0),
            _game("m6", 2020, "2020-09-27", "Sunday", "GGG", "III", 0.0),
            # JJJ played away on Monday but the next game is NOT the following Sunday (bye first).
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
    """Playing at HOME on Monday is not "on the road" -- must not qualify."""

    derived = derive_mnf_road_short_week_features(_mnf_road_schedule()).set_index("game_id")
    assert derived.loc["m6", MNF_ROAD_SHORT_WEEK_COLUMN] == 0.0


def test_mnf_road_requires_exactly_six_days_not_just_monday_then_sunday() -> None:
    """A Monday road game followed by a much-later Sunday (bye in between) must not qualify."""

    derived = derive_mnf_road_short_week_features(_mnf_road_schedule()).set_index("game_id")
    assert derived.loc["m8", MNF_ROAD_SHORT_WEEK_COLUMN] == 0.0


# ---------------------------------------------------------------------------
# LEAD-40: home-Thursday rest compound
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Leakage: a game's OWN outcome never changes its OWN flag
# ---------------------------------------------------------------------------


def test_flags_are_invariant_to_a_games_own_outcome() -> None:
    """Mutating game X's own ``overtime`` value (its own outcome) must never
    change game X's own flag, for all three constructs -- no flag reads
    anything about the CURRENT game other than its schedule facts (weekday,
    home/away, gameday). It legitimately MAY change a LATER game's flag
    (that is pregame-known history, not leakage); this test checks only the
    mutated game's own value.
    """

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
    """The converse of the leakage test: mutating an EARLIER game's overtime
    outcome legitimately changes a LATER game's post-OT flag, since that is
    pregame-known history for the later game, not leakage."""

    schedule = _post_ot_schedule()
    before = derive_post_ot_fatigue_features(schedule).set_index("game_id")
    assert before.loc["g3", POST_OT_FATIGUE_COLUMN] == 1.0

    mutated = schedule.copy()
    mutated.loc[mutated["game_id"] == "g0", "overtime"] = 0.0  # DDD's prior game no longer OT
    after = derive_post_ot_fatigue_features(mutated).set_index("game_id")
    assert after.loc["g3", POST_OT_FATIGUE_COLUMN] == 0.0


# ---------------------------------------------------------------------------
# Additivity / join contracts (mirrors every sibling *_production_feature module)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Registered candidate profiles: production plus exactly the one column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(sfop.CANDIDATES))
def test_registered_profile_is_production_plus_the_declared_one_column(key: str) -> None:
    candidate = sfop.CANDIDATES[key]
    baseline = set(margin_feature_columns("market_residual", sfop.BASELINE_PROFILE))
    treatment = set(margin_feature_columns("market_residual", candidate.profile))
    assert treatment - baseline == {candidate.column}
    assert baseline - treatment == set()


@pytest.mark.parametrize("key", sorted(sfop.CANDIDATES))
def test_candidate_duck_types_with_the_template_profile_identity(key: str) -> None:
    """``on_production_opener_confirmation.profile_identity`` is reused
    unmodified: our ``ScheduleCandidate`` need only carry the same
    ``profile``/``column`` attribute names."""

    candidate = sfop.CANDIDATES[key]
    columns = margin_feature_columns("market_residual", candidate.profile)
    frame = pd.DataFrame({column: [0.0] for column in columns})
    observed = sfop.confirmation.profile_identity(candidate, frame)
    assert observed["only_added_column"] == candidate.column
