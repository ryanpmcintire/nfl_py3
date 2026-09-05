"""Construction, sign-convention, phrase-discipline, and leakage/timing
contracts for the Wave 8 leads on production (LEAD-24 stage 2, LEAD-16).

Every test injects an already-built table directly (``dependence_table``,
``transactions_index``/``snap_counts``, or ``table``), so no real local
snapshot is read -- matching every sibling on-production test module's own
convention (``tests/test_officials_flag_features.py``,
``tests/test_schedule_flag_on_production.py``).
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.rookie_kicker_flag_features import (
    KICKER_ACQUIRE_RE,
    KICKER_ACQUIRE_SPECULATIVE_RE,
    KICKER_CHANGE_COLUMN,
    KICKER_CHANGE_GAMES,
    KICKER_POSITION,
    ROOKIE_WALL_DEPENDENCE_COLUMN,
    confirmed_kicker_change_transactions,
    derive_kicker_change_underdog_features,
    derive_rookie_wall_dependence_fade_features,
    describe_kicker_change_population,
    kicker_player_slugs,
)

# ---------------------------------------------------------------------------
# LEAD-24 stage 2: rookie-wall dependence fade
# ---------------------------------------------------------------------------


def _schedule_row(game_id: str, season: int, week: int, home: str, away: str) -> dict:
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "home_team": home,
        "away_team": away,
    }


def _dependence_row(team: str, season: int, week: int, dependent: bool) -> dict:
    return {"team": team, "season": season, "week": week, "late_season_high_dependence": dependent}


def test_rookie_wall_dependence_sign_convention() -> None:
    schedule = pd.DataFrame(
        [
            _schedule_row("g_away_dep", 2020, 12, "HOME_A", "AWAY_A"),  # away dependent -> +1
            _schedule_row("g_home_dep", 2020, 12, "HOME_B", "AWAY_B"),  # home dependent -> -1
            _schedule_row("g_both_dep", 2020, 12, "HOME_C", "AWAY_C"),  # both dependent -> 0
            _schedule_row("g_neither", 2020, 12, "HOME_D", "AWAY_D"),  # neither -> 0
            _schedule_row("g_missing", 2020, 12, "HOME_E", "AWAY_E"),  # no dependence rows -> 0
        ]
    )
    dependence = pd.DataFrame(
        [
            _dependence_row("HOME_A", 2020, 12, False),
            _dependence_row("AWAY_A", 2020, 12, True),
            _dependence_row("HOME_B", 2020, 12, True),
            _dependence_row("AWAY_B", 2020, 12, False),
            _dependence_row("HOME_C", 2020, 12, True),
            _dependence_row("AWAY_C", 2020, 12, True),
            _dependence_row("HOME_D", 2020, 12, False),
            _dependence_row("AWAY_D", 2020, 12, False),
        ]
    )
    flags = derive_rookie_wall_dependence_fade_features(schedule, dependence).set_index("game_id")[
        ROOKIE_WALL_DEPENDENCE_COLUMN
    ]
    assert flags["g_away_dep"] == 1.0
    assert flags["g_home_dep"] == -1.0
    assert flags["g_both_dep"] == 0.0
    assert flags["g_neither"] == 0.0
    assert flags["g_missing"] == 0.0


def test_rookie_wall_dependence_joins_the_correct_season_week_row() -> None:
    """A dependence row for the SAME team but a DIFFERENT week must never be
    read for this game -- pins the join key, not just the sign."""

    schedule = pd.DataFrame([_schedule_row("g1", 2021, 14, "HOME_X", "AWAY_X")])
    dependence = pd.DataFrame(
        [
            # Wrong week for AWAY_X: must NOT make g1 fire +1.
            _dependence_row("AWAY_X", 2021, 13, True),
            _dependence_row("AWAY_X", 2021, 14, False),
            _dependence_row("HOME_X", 2021, 14, False),
        ]
    )
    flags = derive_rookie_wall_dependence_fade_features(schedule, dependence).set_index("game_id")[
        ROOKIE_WALL_DEPENDENCE_COLUMN
    ]
    assert flags["g1"] == 0.0


def test_rookie_wall_dependence_missing_schedule_columns_raises() -> None:
    with pytest.raises(DataContractError):
        derive_rookie_wall_dependence_fade_features(
            pd.DataFrame({"game_id": ["g1"]}),
            pd.DataFrame(columns=["team", "season", "week", "late_season_high_dependence"]),
        )


def test_rookie_wall_dependence_missing_table_columns_raises() -> None:
    schedule = pd.DataFrame([_schedule_row("g1", 2020, 12, "H", "A")])
    with pytest.raises(DataContractError):
        derive_rookie_wall_dependence_fade_features(schedule, pd.DataFrame({"team": ["H"]}))


# ---------------------------------------------------------------------------
# LEAD-16: kicker-acquisition phrase discipline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slug",
    [
        "buccaneers-sign-kicker-roberto-aguayo",
        "rams-re-sign-kicker-greg-zuerlein",
        "jets-claim-kicker-kaare-vedvik",
        "with-k-harrison-butker-ailing-chiefs-sign-kicker-to-practice-squad",
        "browns-activate-kicker-nick-folk-from-ir",
        "saints-to-swap-kickers-by-signing-cade-york-waiving-k-blake-grupe",
    ],
)
def test_kicker_acquire_re_matches_confirmed_language(slug: str) -> None:
    assert KICKER_ACQUIRE_RE.search(slug) is not None


@pytest.mark.parametrize(
    "slug",
    [
        "cowboys-wont-sign-kicker-this-week",
        "cowboys-not-signing-kicker",
        "lions-expected-to-sign-ufl-kicker-jake-bates",
        "giants-ben-mcadoo-on-signing-another-kicker-never-say-never",
    ],
)
def test_kicker_acquire_speculative_re_excludes_negated_and_predicted_language(slug: str) -> None:
    assert KICKER_ACQUIRE_RE.search(slug) is not None  # would match the base verb pattern...
    assert KICKER_ACQUIRE_SPECULATIVE_RE.search(slug) is not None  # ...but is excluded


def _transactions_index(rows: list[dict]) -> pd.DataFrame:
    columns = ["slug", "category", "url_year", "url_month"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def test_confirmed_kicker_change_transactions_excludes_speculative_and_wrong_category() -> None:
    rows = _transactions_index(
        [
            {
                "slug": "buccaneers-sign-kicker-roberto-aguayo",
                "category": "signing",
                "url_year": 2016,
                "url_month": 9,
            },
            {
                "slug": "cowboys-wont-sign-kicker-this-week",
                "category": "signing",
                "url_year": 2016,
                "url_month": 9,
            },
            {
                "slug": "vikings-release-kicker-blair-walsh",
                "category": "release",  # not an acquisition-direction category
                "url_year": 2017,
                "url_month": 1,
            },
        ]
    )
    confirmed = confirmed_kicker_change_transactions(rows)
    assert list(confirmed["slug"]) == ["buccaneers-sign-kicker-roberto-aguayo"]


def test_confirmed_kicker_change_transactions_requires_slug_and_category_columns() -> None:
    with pytest.raises(DataContractError):
        confirmed_kicker_change_transactions(pd.DataFrame({"slug": ["x-sign-y"]}))


# ---------------------------------------------------------------------------
# LEAD-16: player-identity resolution restricted to a confirmed kicker
# ---------------------------------------------------------------------------


def _snap_counts(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["player", "team", "position"])


def test_kicker_player_slugs_excludes_non_kicker_positions() -> None:
    snaps = _snap_counts(
        [
            ("Roberto Aguayo", "TB", "K"),
            ("Roberto Anderson", "TB", "WR"),  # similarly-named non-kicker
        ]
    )
    slugs = kicker_player_slugs(snaps)
    assert "Roberto Aguayo" in set(slugs["player"])
    assert "Roberto Anderson" not in set(slugs["player"])


def test_describe_kicker_change_population_never_guesses_an_unresolvable_row() -> None:
    transactions = _transactions_index(
        [
            {
                "slug": "buccaneers-sign-kicker-roberto-aguayo",
                "category": "signing",
                "url_year": 2016,
                "url_month": 9,
            },
            # Team unresolvable (no recognizable team nickname token).
            {
                "slug": "sign-kicker-someone-somewhere",
                "category": "signing",
                "url_year": 2018,
                "url_month": 10,
            },
        ]
    )
    snaps = _snap_counts([("Roberto Aguayo", "TB", "K")])
    stats = describe_kicker_change_population(transactions, snaps)
    assert stats["n_candidate_slugs"] == 2
    assert stats["n_resolved_kicker_and_team"] == 1
    assert stats["resolved_slugs"] == ["buccaneers-sign-kicker-roberto-aguayo"]


# ---------------------------------------------------------------------------
# LEAD-16: sign convention, timing (first-qualifying-game-only), and
# pregame safety
# ---------------------------------------------------------------------------


def _reg_schedule_row(
    game_id: str, season: int, week: int, gameday: str, home: str, away: str
) -> dict:
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "game_type": "REG",
        "gameday": gameday,
        "home_team": home,
        "away_team": away,
    }


def _lines(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["game_id", "tue_open_home_spread"])


def test_kicker_change_underdog_sign_convention() -> None:
    schedule = pd.DataFrame(
        [
            _reg_schedule_row("g1", 2016, 2, "2016-09-15", "TB", "OPP1"),  # TB changed, home dog
            _reg_schedule_row("g2", 2016, 2, "2016-09-15", "OPP2", "TB"),  # TB changed, away dog
            _reg_schedule_row("g3", 2016, 2, "2016-09-15", "TB", "OPP3"),  # TB changed, home fav
        ]
    )
    transactions = _transactions_index(
        [
            {
                "slug": "buccaneers-sign-kicker-roberto-aguayo",
                "category": "signing",
                "url_year": 2016,
                "url_month": 8,  # report predates every Week 2 kickoff above
            }
        ]
    )
    snaps = _snap_counts([("Roberto Aguayo", "TB", "K")])
    lines = _lines([("g1", -3.0), ("g2", 3.0), ("g3", 3.0)])
    flags = derive_kicker_change_underdog_features(schedule, transactions, snaps, lines).set_index(
        "game_id"
    )[KICKER_CHANGE_COLUMN]
    assert flags["g1"] == 1.0  # home underdog, TB (home) changed kicker
    assert flags["g2"] == -1.0  # away underdog, TB (away) changed kicker
    # g3: TB (home) changed its kicker and is FAVORED (not the dog) -- the
    # construct is NOT side-specific (predeclared: signed only by which team
    # is the underdog, not by which team changed its kicker), so eligibility
    # is TRUE (a change happened) and the flag still follows the opener
    # underdog -- here the AWAY team (OPP3) -> -1.
    assert flags["g3"] == -1.0


def test_kicker_change_underdog_requires_eligibility_not_just_underdog() -> None:
    schedule = pd.DataFrame([_reg_schedule_row("g1", 2018, 3, "2018-09-20", "H", "A")])
    transactions = _transactions_index([])
    snaps = _snap_counts([])
    lines = _lines([("g1", -3.0)])  # home underdog but no kicker change anywhere
    flags = derive_kicker_change_underdog_features(schedule, transactions, snaps, lines).set_index(
        "game_id"
    )[KICKER_CHANGE_COLUMN]
    assert flags["g1"] == 0.0


def test_kicker_change_underdog_missing_opener_spread_is_zero() -> None:
    schedule = pd.DataFrame([_reg_schedule_row("g1", 2016, 2, "2016-09-15", "TB", "OPP1")])
    transactions = _transactions_index(
        [
            {
                "slug": "buccaneers-sign-kicker-roberto-aguayo",
                "category": "signing",
                "url_year": 2016,
                "url_month": 8,
            }
        ]
    )
    snaps = _snap_counts([("Roberto Aguayo", "TB", "K")])
    lines = _lines([])  # no opener line row for g1 at all
    flags = derive_kicker_change_underdog_features(schedule, transactions, snaps, lines).set_index(
        "game_id"
    )[KICKER_CHANGE_COLUMN]
    assert flags["g1"] == 0.0


def test_kicker_change_only_the_first_qualifying_game_flags() -> None:
    """KICKER_CHANGE_GAMES == 1: the week immediately after the report
    flags; a LATER week for the same team must not."""

    assert KICKER_CHANGE_GAMES == 1
    schedule = pd.DataFrame(
        [
            _reg_schedule_row("g_week2", 2016, 2, "2016-09-15", "TB", "OPP1"),
            _reg_schedule_row("g_week3", 2016, 3, "2016-09-22", "TB", "OPP2"),
        ]
    )
    transactions = _transactions_index(
        [
            {
                "slug": "buccaneers-sign-kicker-roberto-aguayo",
                "category": "signing",
                "url_year": 2016,
                "url_month": 8,
            }
        ]
    )
    snaps = _snap_counts([("Roberto Aguayo", "TB", "K")])
    lines = _lines([("g_week2", -3.0), ("g_week3", -3.0)])
    flags = derive_kicker_change_underdog_features(schedule, transactions, snaps, lines).set_index(
        "game_id"
    )[KICKER_CHANGE_COLUMN]
    assert flags["g_week2"] == 1.0
    assert flags["g_week3"] == 0.0


def test_kicker_change_report_after_kickoff_never_qualifies_that_game() -> None:
    """Pregame safety: a report whose latest-possible date (month-end)
    falls AFTER this game's own kickoff must never flag this game."""

    schedule = pd.DataFrame([_reg_schedule_row("g_early", 2016, 1, "2016-09-08", "TB", "OPP1")])
    transactions = _transactions_index(
        [
            {
                "slug": "buccaneers-sign-kicker-roberto-aguayo",
                "category": "signing",
                "url_year": 2016,
                "url_month": 9,  # month-end (2016-09-30) is AFTER g_early's kickoff
            }
        ]
    )
    snaps = _snap_counts([("Roberto Aguayo", "TB", "K")])
    lines = _lines([("g_early", -3.0)])
    flags = derive_kicker_change_underdog_features(schedule, transactions, snaps, lines).set_index(
        "game_id"
    )[KICKER_CHANGE_COLUMN]
    assert flags["g_early"] == 0.0


def test_kicker_change_missing_schedule_columns_raises() -> None:
    with pytest.raises(DataContractError):
        derive_kicker_change_underdog_features(
            pd.DataFrame({"game_id": ["g1"]}),
            _transactions_index([]),
            _snap_counts([]),
            _lines([]),
        )


def test_kicker_change_missing_opener_lines_game_id_raises() -> None:
    schedule = pd.DataFrame([_reg_schedule_row("g1", 2016, 2, "2016-09-15", "TB", "OPP1")])
    with pytest.raises(DataContractError):
        derive_kicker_change_underdog_features(
            schedule,
            _transactions_index([]),
            _snap_counts([]),
            pd.DataFrame({"tue_open_home_spread": [1.0]}),
        )


def test_kicker_position_constant() -> None:
    assert KICKER_POSITION == "K"
