"""Construction, opener-conditioning, streak-reset, and leakage contracts for
the four Wave 3 (docs/schedule_flag_battery.md "Wave 3", LEAD-57 leads on
production) public-claim flags: ``road_fav_big_fade_flag``,
``division_dog_flag``, ``week1_dog_flag``, ``ats_streak_regress_flag``.

Every fixture is built in memory: these tests must pass in a fresh clone
with no local data snapshots (no schedules.parquet or market archive is
ever read).
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
    ATS_STREAK_REGRESS_COLUMN,
    DIVISION_DOG_COLUMN,
    ROAD_FAV_BIG_FADE_COLUMN,
    WEEK1_DOG_COLUMN,
    attach_ats_streak_regress_features,
    attach_division_dog_features,
    attach_road_fav_big_fade_features,
    attach_week1_dog_features,
    derive_ats_streak_regress_features,
    derive_division_dog_features,
    derive_road_fav_big_fade_features,
    derive_week1_dog_features,
)


def _game(
    game_id: str,
    season: int,
    gameday: str,
    home: str,
    away: str,
    *,
    game_type: str = "REG",
    week: int = 1,
    div_game: int = 0,
    result: float | None = None,
    spread_line: float | None = None,
) -> dict:
    return {
        "game_id": game_id,
        "season": season,
        "gameday": gameday,
        "game_type": game_type,
        "week": week,
        "div_game": div_game,
        "home_team": home,
        "away_team": away,
        "result": result,
        "spread_line": spread_line,
    }


def _schedule(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _opener_lines(rows: dict[str, tuple[float | None, float | None]]) -> pd.DataFrame:
    """``{game_id: (home_spread, total_line)}`` -> the opener_lines frame shape."""

    return pd.DataFrame(
        {
            "game_id": list(rows),
            "tue_open_home_spread": [v[0] for v in rows.values()],
            "tue_open_total_line": [v[1] for v in rows.values()],
        }
    )


# ---------------------------------------------------------------------------
# road_fav_big_fade
# ---------------------------------------------------------------------------


def test_road_fav_big_fade_away_favorite_is_positive() -> None:
    """Away favored by 7+ at the opener (home_spread <= -7) -> +1 (back home)."""

    schedule = _schedule([_game("g1", 2020, "2020-09-13", "AAA", "BBB")])
    lines = _opener_lines({"g1": (-7.0, 45.0)})
    derived = derive_road_fav_big_fade_features(schedule, lines).set_index("game_id")
    assert derived.loc["g1", ROAD_FAV_BIG_FADE_COLUMN] == 1.0


def test_road_fav_big_fade_home_favorite_is_negative() -> None:
    """Home favored by 7+ at the opener -> -1, the task-instructed mirror case."""

    schedule = _schedule([_game("g2", 2020, "2020-09-13", "AAA", "BBB")])
    lines = _opener_lines({"g2": (7.5, 45.0)})
    derived = derive_road_fav_big_fade_features(schedule, lines).set_index("game_id")
    assert derived.loc["g2", ROAD_FAV_BIG_FADE_COLUMN] == -1.0


def test_road_fav_big_fade_below_threshold_is_zero() -> None:
    schedule = _schedule([_game("g3", 2020, "2020-09-13", "AAA", "BBB")])
    lines = _opener_lines({"g3": (-6.5, 45.0)})
    derived = derive_road_fav_big_fade_features(schedule, lines).set_index("game_id")
    assert derived.loc["g3", ROAD_FAV_BIG_FADE_COLUMN] == 0.0


def test_road_fav_big_fade_missing_opener_spread_is_zero_not_nan() -> None:
    schedule = _schedule([_game("g4", 2020, "2020-09-13", "AAA", "BBB")])
    lines = _opener_lines({"g4": (None, None)})
    derived = derive_road_fav_big_fade_features(schedule, lines).set_index("game_id")
    assert derived.loc["g4", ROAD_FAV_BIG_FADE_COLUMN] == 0.0
    assert not pd.isna(derived.loc["g4", ROAD_FAV_BIG_FADE_COLUMN])


def test_road_fav_big_fade_non_reg_game_is_zero_even_if_qualifying() -> None:
    """A postseason big road favorite is never flagged (REG-only population)."""

    schedule = _schedule([_game("g5", 2020, "2021-01-10", "AAA", "BBB", game_type="WC", week=18)])
    lines = _opener_lines({"g5": (-10.0, 45.0)})
    derived = derive_road_fav_big_fade_features(schedule, lines).set_index("game_id")
    assert derived.loc["g5", ROAD_FAV_BIG_FADE_COLUMN] == 0.0


def test_road_fav_big_fade_uses_the_opener_not_the_schedules_own_close() -> None:
    """Opener conditioning: the schedule's own close spread_line disagrees
    with the opener, and the flag follows the OPENER."""

    schedule = _schedule([_game("g6", 2020, "2020-09-13", "AAA", "BBB", spread_line=-2.0)])
    lines = _opener_lines({"g6": (-9.0, 45.0)})  # opener: away favored by 9
    derived = derive_road_fav_big_fade_features(schedule, lines).set_index("game_id")
    assert derived.loc["g6", ROAD_FAV_BIG_FADE_COLUMN] == 1.0


def test_attach_road_fav_big_fade_is_purely_additive() -> None:
    schedule = _schedule([_game("g7", 2020, "2020-09-13", "AAA", "BBB")])
    lines = _opener_lines({"g7": (-8.0, 45.0)})
    features = pd.DataFrame({"game_id": schedule["game_id"], "existing": 1.0})
    widened = attach_road_fav_big_fade_features(features, schedule=schedule, opener_lines=lines)
    assert sorted(set(widened.columns) - set(features.columns)) == [ROAD_FAV_BIG_FADE_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)


# ---------------------------------------------------------------------------
# division_dog / week1_dog (shared shape)
# ---------------------------------------------------------------------------


def test_division_dog_home_underdog_is_positive() -> None:
    schedule = _schedule([_game("d1", 2020, "2020-09-13", "AAA", "BBB", div_game=1)])
    lines = _opener_lines({"d1": (-3.0, 45.0)})  # home underdog
    derived = derive_division_dog_features(schedule, lines).set_index("game_id")
    assert derived.loc["d1", DIVISION_DOG_COLUMN] == 1.0


def test_division_dog_away_underdog_is_negative() -> None:
    schedule = _schedule([_game("d2", 2020, "2020-09-13", "AAA", "BBB", div_game=1)])
    lines = _opener_lines({"d2": (3.0, 45.0)})  # home favored, away is the dog
    derived = derive_division_dog_features(schedule, lines).set_index("game_id")
    assert derived.loc["d2", DIVISION_DOG_COLUMN] == -1.0


def test_division_dog_non_divisional_is_zero() -> None:
    schedule = _schedule([_game("d3", 2020, "2020-09-13", "AAA", "BBB", div_game=0)])
    lines = _opener_lines({"d3": (-3.0, 45.0)})
    derived = derive_division_dog_features(schedule, lines).set_index("game_id")
    assert derived.loc["d3", DIVISION_DOG_COLUMN] == 0.0


def test_division_dog_pickem_is_zero() -> None:
    schedule = _schedule([_game("d4", 2020, "2020-09-13", "AAA", "BBB", div_game=1)])
    lines = _opener_lines({"d4": (0.0, 45.0)})
    derived = derive_division_dog_features(schedule, lines).set_index("game_id")
    assert derived.loc["d4", DIVISION_DOG_COLUMN] == 0.0


def test_division_dog_excludes_postseason_divisional_rematches() -> None:
    """A divisional playoff game (div_game can be 1 in the WC/DIV round) is
    never flagged -- REG-only population, matching lane G's own claim."""

    schedule = _schedule(
        [_game("d5", 2020, "2021-01-10", "AAA", "BBB", game_type="DIV", week=19, div_game=1)]
    )
    lines = _opener_lines({"d5": (-4.0, 45.0)})
    derived = derive_division_dog_features(schedule, lines).set_index("game_id")
    assert derived.loc["d5", DIVISION_DOG_COLUMN] == 0.0


def test_division_dog_missing_opener_spread_is_zero() -> None:
    schedule = _schedule([_game("d6", 2020, "2020-09-13", "AAA", "BBB", div_game=1)])
    lines = _opener_lines({"d6": (None, None)})
    derived = derive_division_dog_features(schedule, lines).set_index("game_id")
    assert derived.loc["d6", DIVISION_DOG_COLUMN] == 0.0


def test_week1_dog_home_underdog_is_positive() -> None:
    schedule = _schedule([_game("w1", 2020, "2020-09-10", "AAA", "BBB", week=1)])
    lines = _opener_lines({"w1": (-2.5, 45.0)})
    derived = derive_week1_dog_features(schedule, lines).set_index("game_id")
    assert derived.loc["w1", WEEK1_DOG_COLUMN] == 1.0


def test_week1_dog_away_underdog_is_negative() -> None:
    schedule = _schedule([_game("w2", 2020, "2020-09-10", "AAA", "BBB", week=1)])
    lines = _opener_lines({"w2": (2.5, 45.0)})
    derived = derive_week1_dog_features(schedule, lines).set_index("game_id")
    assert derived.loc["w2", WEEK1_DOG_COLUMN] == -1.0


def test_week1_dog_week_two_is_zero_even_if_it_would_otherwise_qualify() -> None:
    schedule = _schedule([_game("w3", 2020, "2020-09-20", "AAA", "BBB", week=2)])
    lines = _opener_lines({"w3": (-2.5, 45.0)})
    derived = derive_week1_dog_features(schedule, lines).set_index("game_id")
    assert derived.loc["w3", WEEK1_DOG_COLUMN] == 0.0


def test_week1_dog_uses_the_opener_not_the_schedules_own_close() -> None:
    schedule = _schedule([_game("w4", 2020, "2020-09-10", "AAA", "BBB", week=1, spread_line=3.0)])
    lines = _opener_lines({"w4": (-1.0, 45.0)})  # opener disagrees with the close
    derived = derive_week1_dog_features(schedule, lines).set_index("game_id")
    assert derived.loc["w4", WEEK1_DOG_COLUMN] == 1.0


def test_attach_division_dog_and_week1_dog_are_purely_additive() -> None:
    schedule = _schedule([_game("a1", 2020, "2020-09-10", "AAA", "BBB", week=1, div_game=1)])
    lines = _opener_lines({"a1": (-2.0, 45.0)})
    features = pd.DataFrame({"game_id": schedule["game_id"]})
    widened = attach_division_dog_features(features, schedule=schedule, opener_lines=lines)
    assert sorted(set(widened.columns) - set(features.columns)) == [DIVISION_DOG_COLUMN]
    widened2 = attach_week1_dog_features(features, schedule=schedule, opener_lines=lines)
    assert sorted(set(widened2.columns) - set(features.columns)) == [WEEK1_DOG_COLUMN]


# ---------------------------------------------------------------------------
# ats_streak_regress
# ---------------------------------------------------------------------------


def _streak_schedule() -> pd.DataFrame:
    # AAA loses ATS three straight weeks (result < spread_line each time),
    # then meets BBB (no streak) in week 4 -> home (AAA) qualifies -> +1.
    return _schedule(
        [
            _game("s1", 2020, "2020-09-10", "AAA", "ZZZ", week=1, result=-10.0, spread_line=-3.0),
            _game("s2", 2020, "2020-09-17", "YYY", "AAA", week=2, result=3.0, spread_line=-1.0),
            _game("s3", 2020, "2020-09-24", "AAA", "XXX", week=3, result=0.0, spread_line=3.0),
            _game("s4", 2020, "2020-10-01", "AAA", "BBB", week=4, result=None, spread_line=None),
        ]
    )


def test_ats_streak_regress_three_straight_losses_qualifies_home() -> None:
    derived = derive_ats_streak_regress_features(_streak_schedule()).set_index("game_id")
    assert derived.loc["s4", ATS_STREAK_REGRESS_COLUMN] == 1.0


def test_ats_streak_regress_a_cover_resets_the_streak() -> None:
    schedule = _schedule(
        [
            _game("r1", 2020, "2020-09-10", "AAA", "ZZZ", week=1, result=-10.0, spread_line=-3.0),
            _game("r2", 2020, "2020-09-17", "YYY", "AAA", week=2, result=3.0, spread_line=-1.0),
            # AAA COVERS here (result 5 > spread -2 from AAA's home perspective).
            _game("r3", 2020, "2020-09-24", "AAA", "XXX", week=3, result=5.0, spread_line=-2.0),
            _game("r4", 2020, "2020-10-01", "AAA", "BBB", week=4, result=None, spread_line=None),
        ]
    )
    derived = derive_ats_streak_regress_features(schedule).set_index("game_id")
    assert derived.loc["r4", ATS_STREAK_REGRESS_COLUMN] == 0.0


def test_ats_streak_regress_a_push_neither_extends_nor_resets() -> None:
    schedule = _schedule(
        [
            _game("p1", 2020, "2020-09-10", "AAA", "ZZZ", week=1, result=-10.0, spread_line=-3.0),
            _game("p2", 2020, "2020-09-17", "YYY", "AAA", week=2, result=3.0, spread_line=-1.0),
            # Push: AAA's home game, result exactly equals spread_line.
            _game("p3", 2020, "2020-09-24", "AAA", "XXX", week=3, result=-2.0, spread_line=-2.0),
            # Third genuine loss.
            _game("p4", 2020, "2020-10-01", "YYY", "AAA", week=4, result=3.0, spread_line=-1.0),
            _game("p5", 2020, "2020-10-08", "AAA", "BBB", week=5, result=None, spread_line=None),
        ]
    )
    derived = derive_ats_streak_regress_features(schedule).set_index("game_id")
    # AAA's streak: loss, loss, PUSH (skipped), loss => 3 losses entering p5.
    assert derived.loc["p5", ATS_STREAK_REGRESS_COLUMN] == 1.0


def test_ats_streak_regress_resets_at_season_boundary() -> None:
    schedule = _schedule(
        [
            _game("b1", 2020, "2020-09-10", "AAA", "ZZZ", week=1, result=-10.0, spread_line=-3.0),
            _game("b2", 2020, "2020-09-17", "YYY", "AAA", week=2, result=3.0, spread_line=-1.0),
            _game("b3", 2020, "2020-09-24", "AAA", "XXX", week=3, result=0.0, spread_line=3.0),
            # Next SEASON's week 1: the 2020 streak does not carry over.
            _game("b4", 2021, "2021-09-12", "AAA", "BBB", week=1, result=None, spread_line=None),
        ]
    )
    derived = derive_ats_streak_regress_features(schedule).set_index("game_id")
    assert derived.loc["b4", ATS_STREAK_REGRESS_COLUMN] == 0.0


def test_ats_streak_regress_both_qualifying_is_zero() -> None:
    schedule = _schedule(
        [
            _game("q1", 2020, "2020-09-10", "AAA", "ZZZ", week=1, result=-10.0, spread_line=-3.0),
            _game("q2", 2020, "2020-09-17", "YYY", "AAA", week=2, result=3.0, spread_line=-1.0),
            _game("q3", 2020, "2020-09-24", "AAA", "XXX", week=3, result=0.0, spread_line=3.0),
            _game("q4", 2020, "2020-09-10", "BBB", "WWW", week=1, result=-10.0, spread_line=-3.0),
            _game("q5", 2020, "2020-09-17", "VVV", "BBB", week=2, result=3.0, spread_line=-1.0),
            _game("q6", 2020, "2020-09-24", "BBB", "UUU", week=3, result=0.0, spread_line=3.0),
            _game("q7", 2020, "2020-10-01", "AAA", "BBB", week=4, result=None, spread_line=None),
        ]
    )
    derived = derive_ats_streak_regress_features(schedule).set_index("game_id")
    assert derived.loc["q7", ATS_STREAK_REGRESS_COLUMN] == 0.0


def test_ats_streak_regress_non_reg_game_is_zero() -> None:
    schedule = _streak_schedule()
    playoff_row = _game(
        "s5",
        2020,
        "2021-01-10",
        "AAA",
        "BBB",
        game_type="WC",
        week=18,
        result=None,
        spread_line=None,
    )
    schedule = pd.concat([schedule, pd.DataFrame([playoff_row])], ignore_index=True)
    derived = derive_ats_streak_regress_features(schedule).set_index("game_id")
    assert derived.loc["s5", ATS_STREAK_REGRESS_COLUMN] == 0.0


def test_ats_streak_regress_is_invariant_to_this_games_own_outcome() -> None:
    """Mutating game s4's own (currently missing) result must not change its
    own flag -- only STRICTLY PRIOR games feed the streak entering it."""

    schedule = _streak_schedule()
    before = derive_ats_streak_regress_features(schedule).set_index("game_id")
    mutated = schedule.copy()
    mutated.loc[mutated["game_id"] == "s4", "result"] = 20.0
    mutated.loc[mutated["game_id"] == "s4", "spread_line"] = -3.0
    after = derive_ats_streak_regress_features(mutated).set_index("game_id")
    assert after.loc["s4", ATS_STREAK_REGRESS_COLUMN] == before.loc["s4", ATS_STREAK_REGRESS_COLUMN]


def test_ats_streak_regress_a_later_games_flag_may_depend_on_an_earlier_result() -> None:
    """The converse: mutating an EARLIER game's own result legitimately
    changes a LATER game's streak flag -- pregame-known history, not leakage."""

    schedule = _streak_schedule()
    before = derive_ats_streak_regress_features(schedule).set_index("game_id")
    assert before.loc["s4", ATS_STREAK_REGRESS_COLUMN] == 1.0

    mutated = schedule.copy()
    # AAA now COVERS in week 3 instead of losing -> streak breaks before s4.
    mutated.loc[mutated["game_id"] == "s3", "result"] = 10.0
    after = derive_ats_streak_regress_features(mutated).set_index("game_id")
    assert after.loc["s4", ATS_STREAK_REGRESS_COLUMN] == 0.0


def test_attach_ats_streak_regress_is_purely_additive() -> None:
    schedule = _streak_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], "existing": 1.0})
    widened = attach_ats_streak_regress_features(features, schedule=schedule)
    assert sorted(set(widened.columns) - set(features.columns)) == [ATS_STREAK_REGRESS_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)


def test_derive_ats_streak_regress_requires_every_schedule_column() -> None:
    schedule = _streak_schedule().drop(columns=["spread_line"])
    with pytest.raises(DataContractError, match="spread_line"):
        derive_ats_streak_regress_features(schedule)


# ---------------------------------------------------------------------------
# Join contracts (mirrors every sibling *_production_feature module)
# ---------------------------------------------------------------------------


def test_attach_requires_the_join_key() -> None:
    schedule = _schedule([_game("z1", 2020, "2020-09-13", "AAA", "BBB", div_game=1)])
    lines = _opener_lines({"z1": (-3.0, 45.0)})
    features = pd.DataFrame({"not_game_id": schedule["game_id"]})
    with pytest.raises(DataContractError, match="game_id"):
        attach_division_dog_features(features, schedule=schedule, opener_lines=lines)


def test_attach_refuses_to_overwrite_an_existing_column() -> None:
    schedule = _schedule([_game("z2", 2020, "2020-09-13", "AAA", "BBB", div_game=1)])
    lines = _opener_lines({"z2": (-3.0, 45.0)})
    features = pd.DataFrame({"game_id": schedule["game_id"], DIVISION_DOG_COLUMN: 0.0})
    with pytest.raises(DataContractError, match=DIVISION_DOG_COLUMN):
        attach_division_dog_features(features, schedule=schedule, opener_lines=lines)


# ---------------------------------------------------------------------------
# Registered candidate profiles: production plus exactly the one column
# ---------------------------------------------------------------------------

WAVE_3_CANDIDATES = ("road_fav_big_fade", "division_dog", "week1_dog", "ats_streak_regress")


@pytest.mark.parametrize("key", WAVE_3_CANDIDATES)
def test_registered_profile_is_production_plus_the_declared_one_column(key: str) -> None:
    candidate = sfop.CANDIDATES[key]
    baseline = set(margin_feature_columns("market_residual", sfop.BASELINE_PROFILE))
    treatment = set(margin_feature_columns("market_residual", candidate.profile))
    assert treatment - baseline == {candidate.column}
    assert baseline - treatment == set()


@pytest.mark.parametrize("key", WAVE_3_CANDIDATES)
def test_candidate_duck_types_with_the_template_profile_identity(key: str) -> None:
    candidate = sfop.CANDIDATES[key]
    columns = margin_feature_columns("market_residual", candidate.profile)
    frame = pd.DataFrame({column: [0.0] for column in columns})
    observed = sfop.confirmation.profile_identity(candidate, frame)
    assert observed["only_added_column"] == candidate.column
