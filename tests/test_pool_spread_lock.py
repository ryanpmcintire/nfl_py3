"""The pool's spread lock is declared once, and the live opener honours it.

OPS-05 follow-up (2026-09-08, lock day). The pool fixes its spreads at
12:00 ET on Tuesday; the scheduler captures at 12:05. But a legacy Windows
Task Scheduler entry fired a 09:00 ET capture that morning, and the opener
rule was "each book's EARLIEST Tuesday quote", so that stray capture
silently became the opener. These tests pin the replacement rule
(``nfl_ats.market_data.tuesday_opener_quotes``):

* the lock time lives in exactly one place (``POOL_SPREAD_LOCK_ET``) and is
  a wall-clock time in the pool's zone, so its UTC instant follows DST;
* per book, the opener is the earliest Tuesday quote AT OR AFTER the lock;
  a book with no post-lock quote falls back to its earliest pre-lock quote;
* per game, the median is over post-lock books only whenever any exist,
  and ``opener_basis`` says which rule produced the line;
* days are Eastern calendar days (``pool_calendar_day``), so a game's own
  Tuesday is keyed to its kickoff's ET date (``own_week_tuesday``): Monday
  night's 00:15Z-Tuesday kickoff belongs to the Tuesday before it, a Monday
  21:00 ET capture is Monday (never an opener) and a Tuesday 20:30 ET capture
  is a post-lock Tuesday quote -- lane AF, 2026-09-08, after the UTC-day
  keying dropped DEN at KC from the live openers on lock day (15 of 16);
* ``tuesday_opener_quotes`` and ``live_tuesday_openers`` share that one
  filter (``own_week_tuesday_quotes``) and agree game-for-game;
* the historical ``tue_open`` archive (``nfl_ats.clv.build_pairing_table``)
  never routes through the live rule and is unchanged bit-for-bit;
* the one-click refresh guard and the line-gap report read the shared
  constant instead of restating a clock time.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import nfl_ats.clv as clv_module
import nfl_ats.market_data as market_data
from nfl_ats.clv import build_pairing_table, live_tuesday_openers
from nfl_ats.market_data import (
    OPENER_BASIS_POST_LOCK,
    OPENER_BASIS_PRE_LOCK_FALLBACK,
    POOL_SPREAD_LOCK_ET,
    POOL_TIMEZONE,
    attach_nflverse_game_ids,
    load_quote_history,
    own_week_tuesday,
    own_week_tuesday_quotes,
    parse_odds_api_response,
    pool_calendar_day,
    pool_spread_lock_utc,
    tuesday_opener_quotes,
    write_market_snapshot,
)
from nfl_ats.market_observation import (
    MARKET_OBSERVED_AT_COLUMN,
    MARKET_OPENER_BASIS_COLUMN,
    attach_market_observed_at,
)
from nfl_ats.odds_backfill import (
    HISTORICAL_CAPTURE_KIND,
    BackfillTarget,
    parse_historical_odds_response,
    store_historical_snapshot,
)

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.capture_scheduler as capture_scheduler  # noqa: E402
import scripts.refresh_now as refresh_now  # noqa: E402
import scripts.tuesday_line_gap as tuesday_line_gap  # noqa: E402

TUESDAY = date(2026, 9, 8)
GAME_ID = "2026_01_NE_SEA"
KICKOFF = pd.Timestamp("2026-09-10T00:20:00Z")
SCHEDULE = pd.DataFrame(
    {"game_id": [GAME_ID], "home_team": ["SEA"], "away_team": ["NE"], "kickoff": [KICKOFF]}
)


def _et(hour: int, minute: int = 0, day: date = TUESDAY) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=POOL_TIMEZONE)


EARLY = _et(9, 0)
LOCK_CAPTURE = _et(12, 5)
LATER = _et(15, 0)
MONDAY_EVENING = _et(21, 0, date(2026, 9, 7))
TUESDAY_LATE_EVENING = _et(20, 30)

WEEK_1_KICKOFFS = {
    "2026_01_NE_SEA": "2026-09-10T00:20:00Z",
    "2026_01_SF_LA": "2026-09-11T00:35:00Z",
    "2026_01_ATL_PIT": "2026-09-13T17:00:00Z",
    "2026_01_BAL_IND": "2026-09-13T17:00:00Z",
    "2026_01_BUF_HOU": "2026-09-13T17:00:00Z",
    "2026_01_CHI_CAR": "2026-09-13T17:00:00Z",
    "2026_01_CLE_JAX": "2026-09-13T17:00:00Z",
    "2026_01_NO_DET": "2026-09-13T17:00:00Z",
    "2026_01_NYJ_TEN": "2026-09-13T17:00:00Z",
    "2026_01_TB_CIN": "2026-09-13T17:00:00Z",
    "2026_01_ARI_LAC": "2026-09-13T20:25:00Z",
    "2026_01_GB_MIN": "2026-09-13T20:25:00Z",
    "2026_01_MIA_LV": "2026-09-13T20:25:00Z",
    "2026_01_WAS_PHI": "2026-09-13T20:25:00Z",
    "2026_01_DAL_NYG": "2026-09-14T00:20:00Z",
    "2026_01_DEN_KC": "2026-09-15T00:15:00Z",
}
MONDAY_NIGHT_GAME = "2026_01_DEN_KC"
MONDAY_NIGHT_KICKOFF = pd.Timestamp(WEEK_1_KICKOFFS[MONDAY_NIGHT_GAME])


def _quotes(
    observed_at: datetime,
    books: dict[str, float],
    *,
    game_id: str = GAME_ID,
    kickoff: pd.Timestamp = KICKOFF,
) -> pd.DataFrame:
    """Minimal in-memory home-spread rows, one per book, standardized home line."""

    return pd.DataFrame(
        {
            "nflverse_game_id": game_id,
            "bookmaker_key": list(books),
            "market": "spreads",
            "outcome_side": "HOME",
            "home_spread_line": list(books.values()),
            "observed_at_utc": pd.Timestamp(observed_at).tz_convert(UTC),
            "commence_time_utc": kickoff,
        }
    )


def _history(*captures: tuple[datetime, dict[str, float]]) -> pd.DataFrame:
    return pd.concat([_quotes(at, books) for at, books in captures], ignore_index=True)


def _payload(books: dict[str, float]) -> bytes:
    """A real Odds API event so the store tests go through the parser."""

    return json.dumps(
        [
            {
                "id": "event-ne-sea",
                "sport_key": "americanfootball_nfl",
                "commence_time": "2026-09-10T00:20:00Z",
                "home_team": "Seattle Seahawks",
                "away_team": "New England Patriots",
                "bookmakers": [
                    {
                        "key": key,
                        "title": key,
                        "last_update": "2026-09-08T12:00:00Z",
                        "markets": [
                            {
                                "key": "spreads",
                                "last_update": "2026-09-08T12:00:00Z",
                                "outcomes": [
                                    {"name": "Seattle Seahawks", "price": -110, "point": -line},
                                    {"name": "New England Patriots", "price": -110, "point": line},
                                ],
                            }
                        ],
                    }
                    for key, line in books.items()
                ],
            }
        ]
    ).encode()


def _write_live_capture(root: Path, observed_at: datetime, books: dict[str, float]) -> None:
    payload = _payload(books)
    quotes = attach_nflverse_game_ids(
        parse_odds_api_response(payload, observed_at=observed_at), SCHEDULE
    )
    write_market_snapshot(
        payload, quotes, root, observed_at=observed_at, request_metadata={"regions": "us"}
    )


def test_the_pool_lock_is_declared_once_and_every_reader_uses_that_name() -> None:
    assert time(12, 0) == POOL_SPREAD_LOCK_ET
    assert POOL_TIMEZONE.key == "America/New_York"
    assert capture_scheduler.ET.key == POOL_TIMEZONE.key
    assert refresh_now.POOL_SPREAD_LOCK_ET is POOL_SPREAD_LOCK_ET
    assert POOL_SPREAD_LOCK_ET.strftime("%H:%M") == tuesday_line_gap.DEFAULT_LOCK
    assert not hasattr(refresh_now, "OPENER_HOUR")


@pytest.mark.parametrize(
    ("day", "expected_utc"),
    [
        (date(2026, 9, 8), "2026-09-08T16:00:00Z"),
        (date(2026, 11, 10), "2026-11-10T17:00:00Z"),
        (date(2027, 3, 9), "2027-03-09T17:00:00Z"),
        (date(2027, 3, 16), "2027-03-16T16:00:00Z"),
    ],
)
def test_lock_instant_follows_daylight_saving(day: date, expected_utc: str) -> None:
    assert day.weekday() == 1
    assert pool_spread_lock_utc(day) == pd.Timestamp(expected_utc)


def test_scheduler_opener_capture_lands_after_the_pool_lock() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    start = capture_scheduler.occurrence(schedule["odds_tue_open"], _et(15, 0))
    assert pd.Timestamp(start) > pool_spread_lock_utc(TUESDAY)


def test_opener_is_the_first_capture_at_or_after_the_lock_per_book() -> None:
    history = _history(
        (EARLY, {"book_a": 3.0, "book_b": 3.0}),
        (LOCK_CAPTURE, {"book_a": 3.5, "book_b": 4.0}),
        (LATER, {"book_a": 4.5, "book_b": 4.5}),
    )
    opener = tuesday_opener_quotes(history)
    assert len(opener) == 1
    row = opener.iloc[0]
    assert row["nflverse_game_id"] == GAME_ID
    assert row["opener_home_spread"] == pytest.approx(3.75)
    assert (row["opener_min"], row["opener_max"]) == (3.5, 4.0)
    assert row["bookmakers"] == 2
    assert row["observed_at_utc"] == pd.Timestamp("2026-09-08T16:05:00Z")
    assert row["opener_basis"] == OPENER_BASIS_POST_LOCK
    assert list(opener.columns)[-1] == "opener_basis"


def test_a_quote_exactly_at_the_lock_counts_as_post_lock() -> None:
    history = _history((EARLY, {"book_a": 3.0}), (_et(12, 0), {"book_a": 3.5}))
    row = tuesday_opener_quotes(history).iloc[0]
    assert row["opener_home_spread"] == 3.5
    assert row["opener_basis"] == OPENER_BASIS_POST_LOCK


def test_only_pre_lock_captures_fall_back_to_the_earliest_and_say_so() -> None:
    history = _history((EARLY, {"book_a": 3.0, "book_b": 3.0}), (_et(11, 0), {"book_a": 3.5}))
    row = tuesday_opener_quotes(history).iloc[0]
    assert row["opener_home_spread"] == 3.0
    assert row["observed_at_utc"] == pd.Timestamp("2026-09-08T13:00:00Z")
    assert row["opener_basis"] == OPENER_BASIS_PRE_LOCK_FALLBACK
    assert row["bookmakers"] == 2


def test_monday_evening_capture_is_monday_and_never_an_opener() -> None:
    """Tuesday 01:00Z is Monday 21:00 in the pool's zone; the opener rule
    keys on that zone's calendar day, so the quote is not a Tuesday quote at
    all -- not the opener beside a post-lock capture, and not the fallback
    when it stands alone."""

    assert MONDAY_EVENING.astimezone(UTC).weekday() == 1
    with_lock = _history((MONDAY_EVENING, {"book_a": 2.5}), (LOCK_CAPTURE, {"book_a": 3.5}))
    row = tuesday_opener_quotes(with_lock).iloc[0]
    assert row["opener_home_spread"] == 3.5
    assert row["opener_basis"] == OPENER_BASIS_POST_LOCK
    assert row["bookmakers"] == 1
    assert tuesday_opener_quotes(_history((MONDAY_EVENING, {"book_a": 2.5}))).empty


def test_tuesday_evening_capture_after_20_et_is_still_tuesday_and_post_lock() -> None:
    """Wednesday 00:30Z is Tuesday 20:30 in the pool's zone: a Tuesday quote,
    measured against Tuesday's lock (post-lock), so with nothing earlier it
    is the opener and with a 12:05 capture present it loses to the earlier
    post-lock quote."""

    assert TUESDAY_LATE_EVENING.astimezone(UTC).weekday() == 2
    alone = tuesday_opener_quotes(_history((TUESDAY_LATE_EVENING, {"book_a": 3.0}))).iloc[0]
    assert alone["opener_home_spread"] == 3.0
    assert alone["opener_basis"] == OPENER_BASIS_POST_LOCK
    both = _history((LOCK_CAPTURE, {"book_a": 3.5}), (TUESDAY_LATE_EVENING, {"book_a": 3.0}))
    assert tuesday_opener_quotes(both).iloc[0]["opener_home_spread"] == 3.5


def test_own_week_tuesday_is_the_kickoffs_eastern_date_not_its_utc_day() -> None:
    kickoffs = pd.Series(
        [
            pd.Timestamp("2026-09-10T00:20:00Z"),
            pd.Timestamp("2026-09-13T17:00:00Z"),
            pd.Timestamp("2026-09-14T00:20:00Z"),
            pd.Timestamp("2026-09-15T00:15:00Z"),
            pd.Timestamp("2026-09-15T16:00:00Z"),
        ]
    )
    tuesdays = own_week_tuesday(kickoffs)
    assert tuesdays.tolist() == [pd.Timestamp("2026-09-08")] * 4 + [pd.Timestamp("2026-09-15")]
    assert tuesdays.dt.tz is None
    days = pool_calendar_day(
        pd.Series([pd.Timestamp(MONDAY_EVENING), pd.Timestamp(TUESDAY_LATE_EVENING)])
    )
    assert days.tolist() == [pd.Timestamp("2026-09-07"), pd.Timestamp("2026-09-08")]


def test_monday_night_game_gets_its_own_tuesdays_post_lock_opener() -> None:
    """DEN at KC, Monday 2026-09-14 20:15 ET (2026-09-15T00:15Z). Keyed on the
    UTC day its 'own Tuesday' was its kickoff day and the 09-08 quotes were
    excluded; keyed on the Eastern date it is 09-08 like the rest of Week 1."""

    history = _quotes(
        LOCK_CAPTURE,
        {"book_a": -3.0, "book_b": -3.5},
        game_id=MONDAY_NIGHT_GAME,
        kickoff=MONDAY_NIGHT_KICKOFF,
    )
    opener = tuesday_opener_quotes(history)
    assert len(opener) == 1
    row = opener.iloc[0]
    assert row["nflverse_game_id"] == MONDAY_NIGHT_GAME
    assert row["opener_home_spread"] == pytest.approx(-3.25)
    assert row["observed_at_utc"] == pd.Timestamp("2026-09-08T16:05:00Z")
    assert row["opener_basis"] == OPENER_BASIS_POST_LOCK


def test_sunday_and_thursday_games_key_to_the_same_tuesday() -> None:
    sunday = _quotes(
        LOCK_CAPTURE,
        {"book_a": 1.5},
        game_id="2026_01_GB_MIN",
        kickoff=pd.Timestamp(WEEK_1_KICKOFFS["2026_01_GB_MIN"]),
    )
    thursday = _quotes(LOCK_CAPTURE, {"book_a": 3.0})
    by_game = tuesday_opener_quotes(pd.concat([sunday, thursday], ignore_index=True)).set_index(
        "nflverse_game_id"
    )
    assert by_game.loc["2026_01_GB_MIN", "opener_basis"] == OPENER_BASIS_POST_LOCK
    assert by_game.loc[GAME_ID, "opener_basis"] == OPENER_BASIS_POST_LOCK
    assert by_game["observed_at_utc"].eq(pd.Timestamp("2026-09-08T16:05:00Z")).all()
    previous_week = _quotes(_et(12, 5, date(2026, 9, 1)), {"book_a": 9.0})
    assert tuesday_opener_quotes(previous_week).empty


@pytest.mark.parametrize(
    ("kickoff_utc", "tuesday", "pre_lock_utc", "post_lock_utc"),
    [
        ("2026-11-03T01:20:00Z", "2026-10-27", "2026-10-27T15:30:00Z", "2026-10-27T16:05:00Z"),
        ("2026-11-10T01:15:00Z", "2026-11-03", "2026-11-03T16:30:00Z", "2026-11-03T17:05:00Z"),
        ("2027-03-16T00:15:00Z", "2027-03-09", "2027-03-09T16:30:00Z", "2027-03-09T17:05:00Z"),
    ],
)
def test_own_week_across_daylight_saving(
    kickoff_utc: str, tuesday: str, pre_lock_utc: str, post_lock_utc: str
) -> None:
    kickoff = pd.Timestamp(kickoff_utc)
    assert kickoff.tz_convert(POOL_TIMEZONE).weekday() == 0
    assert own_week_tuesday(pd.Series([kickoff])).iloc[0] == pd.Timestamp(tuesday)
    pre = pd.Timestamp(pre_lock_utc).to_pydatetime()
    post = pd.Timestamp(post_lock_utc).to_pydatetime()
    history = pd.concat(
        [
            _quotes(pre, {"book_a": 2.0}, game_id="game", kickoff=kickoff),
            _quotes(post, {"book_a": 2.5}, game_id="game", kickoff=kickoff),
        ],
        ignore_index=True,
    )
    row = tuesday_opener_quotes(history).iloc[0]
    assert row["opener_home_spread"] == 2.5
    assert row["observed_at_utc"] == pd.Timestamp(post_lock_utc)
    assert row["opener_basis"] == OPENER_BASIS_POST_LOCK
    only_pre = tuesday_opener_quotes(_quotes(pre, {"book_a": 2.0}, game_id="game", kickoff=kickoff))
    assert only_pre.iloc[0]["opener_basis"] == OPENER_BASIS_PRE_LOCK_FALLBACK


def _week_1_history(observed_at: datetime, line: float) -> pd.DataFrame:
    return pd.concat(
        [
            _quotes(
                observed_at,
                {"book_a": line, "book_b": line + 0.5},
                game_id=game_id,
                kickoff=pd.Timestamp(kickoff),
            )
            for game_id, kickoff in WEEK_1_KICKOFFS.items()
        ],
        ignore_index=True,
    )


def test_lock_day_capture_yields_all_sixteen_openers_including_monday_night() -> None:
    """Lock day 2026-09-08, measured at 12:10 ET: the 16:05:46Z capture quoted
    all 16 Week 1 games, yet ``live_tuesday_openers`` returned 15 -- DEN at
    KC missing. Reproduced here in memory with the same kickoffs and one
    capture at the same instant, and pinned fixed: 16 games, every one
    ``post_lock`` at that instant."""

    capture = datetime(2026, 9, 8, 16, 5, 46, tzinfo=UTC)
    history = _week_1_history(capture, 3.0)
    assert history["nflverse_game_id"].nunique() == 16
    assert len(own_week_tuesday_quotes(history)) == len(history)
    opener = tuesday_opener_quotes(history)
    assert len(opener) == 16
    assert set(opener["nflverse_game_id"]) == set(WEEK_1_KICKOFFS)
    assert opener["opener_basis"].eq(OPENER_BASIS_POST_LOCK).all()
    assert opener["observed_at_utc"].eq(pd.Timestamp(capture)).all()
    monday_night = opener.set_index("nflverse_game_id").loc[MONDAY_NIGHT_GAME]
    assert monday_night["opener_home_spread"] == pytest.approx(3.25)


def _write_week_1_capture(
    root: Path, schedule: pd.DataFrame, observed_at: datetime, line: float
) -> None:
    """A real Odds API payload for all sixteen Week 1 games, through the parser."""

    names = {code: name for name, code in market_data.NFL_TEAM_NAMES.items()}
    events = []
    for game_id, kickoff in WEEK_1_KICKOFFS.items():
        away, home = game_id.split("_")[2:4]
        events.append(
            {
                "id": f"event-{game_id}",
                "sport_key": "americanfootball_nfl",
                "commence_time": kickoff,
                "home_team": names[home],
                "away_team": names[away],
                "bookmakers": [
                    {
                        "key": key,
                        "title": key,
                        "last_update": "2026-09-08T12:00:00Z",
                        "markets": [
                            {
                                "key": "spreads",
                                "last_update": "2026-09-08T12:00:00Z",
                                "outcomes": [
                                    {"name": names[home], "price": -110, "point": -book_line},
                                    {"name": names[away], "price": -110, "point": book_line},
                                ],
                            }
                        ],
                    }
                    for key, book_line in (("book_a", line), ("book_b", line + 0.5))
                ],
            }
        )
    payload = json.dumps(events).encode()
    quotes = attach_nflverse_game_ids(
        parse_odds_api_response(payload, observed_at=observed_at), schedule
    )
    assert quotes["nflverse_game_id"].notna().all()
    write_market_snapshot(
        payload, quotes, root, observed_at=observed_at, request_metadata={"regions": "us"}
    )


def test_both_live_readers_agree_game_for_game(tmp_path: Path) -> None:
    """``live_tuesday_openers`` (the CLV / predicted-close reader, via the
    manifest index) and ``tuesday_opener_quotes`` on the free-form quote
    history (Best Pick nomination, the board's observation column) share
    one own-week filter and return the same line, instant and basis for
    every game -- the Monday-night game included."""

    root = tmp_path / "raw"
    schedule = pd.DataFrame(
        {
            "game_id": list(WEEK_1_KICKOFFS),
            "home_team": [game_id.split("_")[3] for game_id in WEEK_1_KICKOFFS],
            "away_team": [game_id.split("_")[2] for game_id in WEEK_1_KICKOFFS],
            "kickoff": [pd.Timestamp(kickoff) for kickoff in WEEK_1_KICKOFFS.values()],
        }
    )
    for observed_at, line in ((EARLY, 2.5), (LOCK_CAPTURE, 3.0), (LATER, 4.0)):
        _write_week_1_capture(root, schedule, observed_at, line)

    free_form = (
        tuesday_opener_quotes(load_quote_history(root))
        .rename(
            columns={
                "nflverse_game_id": "game_id",
                "opener_home_spread": "tue_open_home_spread",
                "bookmakers": "opener_books",
                "observed_at_utc": "opener_observed_at_utc",
            }
        )
        .set_index("game_id")
        .sort_index()
    )
    live = live_tuesday_openers(root).set_index("game_id").sort_index()
    assert len(live) == 16
    assert MONDAY_NIGHT_GAME in live.index
    assert live["opener_basis"].eq(OPENER_BASIS_POST_LOCK).all()
    assert live["opener_observed_at_utc"].eq(pd.Timestamp("2026-09-08T16:05:00Z")).all()
    assert live["tue_open_home_spread"].eq(3.25).all()
    pd.testing.assert_frame_equal(live, free_form[live.columns], check_dtype=False)


def test_books_quoting_only_before_the_lock_are_excluded_from_a_post_lock_median() -> None:
    """Rule: when ANY book has a post-lock Tuesday quote, the game's median is
    over post-lock books only. book_a quoted 3.0 at 09:00 and nothing after
    the lock; its line could have moved before the lock, so it does not
    dilute the locked median (4.5 over book_b and book_c), and the book
    count and dispersion describe the post-lock books alone."""

    history = _history(
        (EARLY, {"book_a": 3.0, "book_b": 3.0, "book_c": 3.0}),
        (LOCK_CAPTURE, {"book_b": 4.0, "book_c": 5.0}),
    )
    row = tuesday_opener_quotes(history).iloc[0]
    assert row["opener_home_spread"] == pytest.approx(4.5)
    assert row["bookmakers"] == 2
    assert row["opener_std"] == pytest.approx(pd.Series([4.0, 5.0]).std())
    assert row["opener_basis"] == OPENER_BASIS_POST_LOCK


def test_games_are_judged_independently() -> None:
    other = _quotes(LOCK_CAPTURE, {"book_a": -1.0}).assign(
        nflverse_game_id="2026_01_KC_LAC", commence_time_utc=pd.Timestamp("2026-09-13T20:25:00Z")
    )
    history = pd.concat([_quotes(EARLY, {"book_a": 3.0}), other], ignore_index=True)
    by_game = tuesday_opener_quotes(history).set_index("nflverse_game_id")
    assert by_game.loc[GAME_ID, "opener_basis"] == OPENER_BASIS_PRE_LOCK_FALLBACK
    assert by_game.loc["2026_01_KC_LAC", "opener_basis"] == OPENER_BASIS_POST_LOCK


def test_restored_early_snapshot_cannot_displace_the_locked_line(tmp_path: Path) -> None:
    """Lock day's quarantined 09:00 ET capture (snapshot 20260908T130004Z,
    moved to data/market/raw_early_tuesday/) may be restored under
    data/market/raw once this rule is in: with all three captures present,
    both the free-form store read and ``live_tuesday_openers`` still pick
    the 12:05 quotes."""

    root = tmp_path / "raw"
    _write_live_capture(root, datetime(2026, 9, 8, 13, 0, 4, tzinfo=UTC), {"a": 3.0, "b": 3.0})
    _write_live_capture(root, LOCK_CAPTURE, {"a": 3.5, "b": 4.0})
    _write_live_capture(root, LATER, {"a": 4.5, "b": 4.5})

    opener = tuesday_opener_quotes(load_quote_history(root)).iloc[0]
    assert opener["opener_home_spread"] == pytest.approx(3.75)
    assert opener["observed_at_utc"] == pd.Timestamp("2026-09-08T16:05:00Z")
    assert opener["opener_basis"] == OPENER_BASIS_POST_LOCK

    live = live_tuesday_openers(root)
    assert list(live.columns) == [
        "game_id",
        "tue_open_home_spread",
        "opener_books",
        "opener_observed_at_utc",
        "opener_basis",
    ]
    row = live.set_index("game_id").loc[GAME_ID]
    assert row["tue_open_home_spread"] == pytest.approx(3.75)
    assert row["opener_books"] == 2
    assert row["opener_observed_at_utc"] == pd.Timestamp("2026-09-08T16:05:00Z")
    assert row["opener_basis"] == OPENER_BASIS_POST_LOCK


def test_live_store_with_only_the_early_capture_reports_the_fallback(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _write_live_capture(root, EARLY, {"a": 3.0, "b": 3.0})
    row = live_tuesday_openers(root).set_index("game_id").loc[GAME_ID]
    assert row["tue_open_home_spread"] == 3.0
    assert row["opener_basis"] == OPENER_BASIS_PRE_LOCK_FALLBACK


def test_forecast_frame_carries_the_basis_beside_the_observation_instant(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _write_live_capture(root, EARLY, {"a": 3.0})
    _write_live_capture(root, LOCK_CAPTURE, {"a": 3.5})
    frame = pd.DataFrame({"game_id": [GAME_ID, "2013_01_AAA_BBB"], "spread_line": [3.5, 1.5]})

    result = attach_market_observed_at(frame, market_raw_root=root)

    assert result.loc[0, MARKET_OBSERVED_AT_COLUMN] == pd.Timestamp("2026-09-08T16:05:00Z")
    assert result.loc[0, MARKET_OPENER_BASIS_COLUMN] == OPENER_BASIS_POST_LOCK
    assert pd.isna(result.loc[1, MARKET_OPENER_BASIS_COLUMN])
    assert result["spread_line"].tolist() == [3.5, 1.5]
    bare = attach_market_observed_at(frame)
    assert bare[MARKET_OPENER_BASIS_COLUMN].isna().all()


def _historical_two_game_store(root: Path) -> pd.DataFrame:
    """A ``tue_open`` backfill snapshot at 09:00 ET (13:00Z), as the archive is."""

    schedule = pd.DataFrame(
        {
            "game_id": ["2024_02_CIN_KC", "2024_02_NE_SEA"],
            "home_team": ["KC", "SEA"],
            "away_team": ["CIN", "NE"],
            "kickoff": [
                pd.Timestamp("2024-09-13T00:20:00Z"),
                pd.Timestamp("2024-09-15T17:00:00Z"),
            ],
        }
    )

    def book(key: str, line: float) -> dict[str, Any]:
        return {
            "key": key,
            "title": key,
            "last_update": "2024-09-10T12:00:00Z",
            "markets": [
                {
                    "key": "spreads",
                    "last_update": "2024-09-10T12:00:00Z",
                    "outcomes": [
                        {"name": "__HOME__", "price": -110, "point": -line},
                        {"name": "__AWAY__", "price": -110, "point": line},
                    ],
                }
            ],
        }

    def event(event_id: str, home: str, away: str, commence: str, books: list[dict]) -> dict:
        resolved = []
        for entry in books:
            outcomes = [
                {**o, "name": home if o["name"] == "__HOME__" else away}
                for o in entry["markets"][0]["outcomes"]
            ]
            resolved.append({**entry, "markets": [{**entry["markets"][0], "outcomes": outcomes}]})
        return {
            "id": event_id,
            "sport_key": "americanfootball_nfl",
            "commence_time": commence,
            "home_team": home,
            "away_team": away,
            "bookmakers": resolved,
        }

    events = [
        event(
            "kc-cin",
            "Kansas City Chiefs",
            "Cincinnati Bengals",
            "2024-09-13T00:20:00Z",
            [book("book_a", 1.5), book("book_b", 1.5), book("book_c", 1.5)],
        ),
        event(
            "sea-ne",
            "Seattle Seahawks",
            "New England Patriots",
            "2024-09-15T17:00:00Z",
            [book("book_a", 2.0), book("book_b", 2.5), book("book_c", 3.0)],
        ),
    ]
    snapshot_time = "2024-09-10T13:00:00Z"
    payload = json.dumps(
        {
            "timestamp": snapshot_time,
            "previous_timestamp": None,
            "next_timestamp": None,
            "data": events,
        },
        separators=(",", ":"),
    ).encode()
    capture = parse_historical_odds_response(payload)
    capture = type(capture)(
        snapshot_at_utc=capture.snapshot_at_utc,
        previous_snapshot_at_utc=capture.previous_snapshot_at_utc,
        next_snapshot_at_utc=capture.next_snapshot_at_utc,
        quotes=attach_nflverse_game_ids(capture.quotes, schedule),
    )
    target = BackfillTarget(
        season=2024,
        week=2,
        label="tue_open",
        requested_at_utc=datetime(2024, 9, 10, 13, 0, tzinfo=UTC),
        markets="spreads,totals,h2h",
        regions="us",
        credits=10,
    )
    store_historical_snapshot(payload, capture, root, target=target)
    return schedule


def test_historical_tue_open_archive_never_routes_through_the_live_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``build_pairing_table``'s ``tue_open`` label is the archive the model
    is graded on (09:00 ET backfill snapshots). It is unchanged bit-for-bit:
    the same frame comes back with the lock moved to an absurd hour, and
    with the live opener function replaced by one that raises."""

    root = tmp_path / "raw"
    _historical_two_game_store(root)
    baseline = build_pairing_table(root, capture_kind=HISTORICAL_CAPTURE_KIND)
    tue = baseline.loc[baseline["decision_label"].eq("tue_open")].set_index("game_id")
    assert tue.loc["2024_02_NE_SEA", "home_spread"] == pytest.approx(2.5)
    assert tue.loc["2024_02_NE_SEA", "spread_books"] == 3
    assert tue.loc["2024_02_CIN_KC", "home_spread"] == pytest.approx(1.5)
    assert tue.loc["2024_02_NE_SEA", "snapshot_timestamp_utc"] < pool_spread_lock_utc(
        date(2024, 9, 10)
    )

    monkeypatch.setattr(market_data, "POOL_SPREAD_LOCK_ET", time(23, 59))
    with_moved_lock = build_pairing_table(root, capture_kind=HISTORICAL_CAPTURE_KIND)
    pd.testing.assert_frame_equal(with_moved_lock, baseline)

    def explode(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise AssertionError("the historical archive must not call the live opener rule")

    monkeypatch.setattr(clv_module, "tuesday_opener_quotes", explode)
    monkeypatch.setattr(market_data, "tuesday_opener_quotes", explode)
    without_live_rule = build_pairing_table(root, capture_kind=HISTORICAL_CAPTURE_KIND)
    pd.testing.assert_frame_equal(without_live_rule, baseline)


@pytest.mark.parametrize(
    ("now", "skipped"),
    [
        (_et(11, 59), True),
        (_et(12, 0), False),
        (_et(12, 4), False),
        (MONDAY_EVENING, False),
        (TUESDAY_LATE_EVENING, False),
        (_et(8, 30, date(2026, 9, 9)), False),
    ],
)
def test_refresh_guard_skips_a_spread_capture_only_before_the_pool_lock(
    now: datetime, skipped: bool
) -> None:
    steps = {step.name: step for step in refresh_now.plan(now)}
    assert (steps["spreads"].skip_reason is not None) is skipped
    if skipped:
        assert "12:00 ET spread lock" in str(steps["spreads"].skip_reason)


def test_refresh_guard_reads_the_shared_lock_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Move the (imported) constant and the guard moves with it -- proof it is
    not a restated local clock time."""

    monkeypatch.setattr(refresh_now, "POOL_SPREAD_LOCK_ET", time(14, 0))
    before = {step.name: step for step in refresh_now.plan(_et(12, 30))}
    after = {step.name: step for step in refresh_now.plan(_et(14, 0))}
    assert before["spreads"].skip_reason is not None
    assert "14:00 ET spread lock" in str(before["spreads"].skip_reason)
    assert after["spreads"].skip_reason is None


def test_line_gap_report_compares_earliest_pre_lock_with_first_post_lock() -> None:
    """OPS-05's question: how far did the line move before the pool locked
    it? Per game the earliest PRE-lock capture (the 09:00 legacy capture)
    against the FIRST post-lock capture (12:05), never the post-lock opener
    against itself; the served opener and its basis sit beside them."""

    history = _history(
        (EARLY, {"book_a": 3.0, "book_b": 3.0}),
        (_et(11, 0), {"book_a": 3.5, "book_b": 3.5}),
        (LOCK_CAPTURE, {"book_a": 3.5, "book_b": 4.0}),
        (LATER, {"book_a": 4.5, "book_b": 4.5}),
    )
    table = tuesday_line_gap.tuesday_gap(history, TUESDAY, POOL_SPREAD_LOCK_ET)
    assert list(table.columns) == tuesday_line_gap.COLUMNS
    row = table.iloc[0]
    assert row["game_id"] == GAME_ID
    assert row["pre_lock"] == pytest.approx(3.0)
    assert row["pre_lock_at"] == pd.Timestamp("2026-09-08T13:00:00Z")
    assert row["post_lock"] == pytest.approx(3.75)
    assert row["post_lock_at"] == pd.Timestamp("2026-09-08T16:05:00Z")
    assert row["move"] == pytest.approx(0.75)
    assert row["opener"] == pytest.approx(3.75)
    assert row["opener_basis"] == OPENER_BASIS_POST_LOCK


def test_line_gap_report_says_when_there_is_nothing_to_compare(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    post_only = tuesday_line_gap.tuesday_gap(
        _history((LOCK_CAPTURE, {"book_a": 3.5})), TUESDAY, POOL_SPREAD_LOCK_ET
    )
    row = post_only.iloc[0]
    assert pd.isna(row["pre_lock"]) and pd.isna(row["move"])
    assert row["post_lock"] == 3.5
    assert row["opener_basis"] == OPENER_BASIS_POST_LOCK
    pre_only = tuesday_line_gap.tuesday_gap(
        _history((EARLY, {"book_a": 3.0})), TUESDAY, POOL_SPREAD_LOCK_ET
    )
    row = pre_only.iloc[0]
    assert row["pre_lock"] == 3.0
    assert pd.isna(row["post_lock"]) and pd.isna(row["move"])
    assert row["opener_basis"] == OPENER_BASIS_PRE_LOCK_FALLBACK

    monkeypatch.setattr(
        tuesday_line_gap,
        "load_decision_quotes",
        lambda *_args, **_kwargs: _history((LOCK_CAPTURE, {"a": 3.5})),
    )
    assert tuesday_line_gap.main(["--date", TUESDAY.isoformat()]) == 0
    out = capsys.readouterr().out
    assert tuesday_line_gap.NO_PRE_LOCK_MESSAGE in out
    summary = json.loads(out.strip().splitlines()[-1])
    assert summary["with_pre_lock_capture"] == 0
    assert summary["with_post_lock_capture"] == 1
    assert summary["comparable"] == 0
    assert "mean_abs_move" not in summary


def test_line_gap_report_lists_the_monday_night_game_on_its_own_tuesday() -> None:
    table = tuesday_line_gap.tuesday_gap(
        _week_1_history(datetime(2026, 9, 8, 16, 5, 46, tzinfo=UTC), 3.0),
        TUESDAY,
        POOL_SPREAD_LOCK_ET,
    )
    assert len(table) == 16
    assert MONDAY_NIGHT_GAME in set(table["game_id"])
