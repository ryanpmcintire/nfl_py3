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
* "Tuesday" stays the UTC calendar day, made explicit at both edges;
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
    parse_odds_api_response,
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

# ---------------------------------------------------------------------------
# Fixtures: one Week 1 game, Tuesday 2026-09-08 (EDT, so the lock is 16:00Z)
# ---------------------------------------------------------------------------

TUESDAY = date(2026, 9, 8)
GAME_ID = "2026_01_NE_SEA"
KICKOFF = pd.Timestamp("2026-09-10T00:20:00Z")  # Thursday night, own-week Tuesday = 09-08
SCHEDULE = pd.DataFrame(
    {"game_id": [GAME_ID], "home_team": ["SEA"], "away_team": ["NE"], "kickoff": [KICKOFF]}
)


def _et(hour: int, minute: int = 0, day: date = TUESDAY) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=POOL_TIMEZONE)


EARLY = _et(9, 0)  # the legacy task's capture (13:00Z), quarantined on lock day
LOCK_CAPTURE = _et(12, 5)  # the scheduler's odds_tue_open (16:05Z)
LATER = _et(15, 0)  # an afternoon capture (19:00Z)
MONDAY_EVENING = _et(21, 0, date(2026, 9, 7))  # Tuesday 01:00Z: Tuesday to the UTC day
TUESDAY_LATE_EVENING = _et(20, 30)  # Wednesday 00:30Z: no longer Tuesday to the UTC day


def _quotes(observed_at: datetime, books: dict[str, float]) -> pd.DataFrame:
    """Minimal in-memory home-spread rows, one per book, standardized home line."""

    return pd.DataFrame(
        {
            "nflverse_game_id": GAME_ID,
            "bookmaker_key": list(books),
            "market": "spreads",
            "outcome_side": "HOME",
            "home_spread_line": list(books.values()),
            "observed_at_utc": pd.Timestamp(observed_at).tz_convert(UTC),
            "commence_time_utc": KICKOFF,
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
                                    # The parser standardizes home_spread_line = -point.
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


# ---------------------------------------------------------------------------
# 1. One declared lock, in the pool's zone, DST-correct
# ---------------------------------------------------------------------------


def test_the_pool_lock_is_declared_once_and_every_reader_uses_that_name() -> None:
    assert time(12, 0) == POOL_SPREAD_LOCK_ET
    assert POOL_TIMEZONE.key == "America/New_York"
    # The scheduler's clock and the pool's clock are the same zone.
    assert capture_scheduler.ET.key == POOL_TIMEZONE.key
    # The one-click refresh guard and the line-gap report import the constant
    # rather than restating "12:00" locally.
    assert refresh_now.POOL_SPREAD_LOCK_ET is POOL_SPREAD_LOCK_ET
    assert POOL_SPREAD_LOCK_ET.strftime("%H:%M") == tuesday_line_gap.DEFAULT_LOCK
    assert not hasattr(refresh_now, "OPENER_HOUR")


@pytest.mark.parametrize(
    ("day", "expected_utc"),
    [
        (date(2026, 9, 8), "2026-09-08T16:00:00Z"),  # EDT: UTC-4
        (date(2026, 11, 10), "2026-11-10T17:00:00Z"),  # EST after the 2026-11-01 fall-back
        (date(2027, 3, 9), "2027-03-09T17:00:00Z"),  # EST, the Tuesday before spring-forward
        (date(2027, 3, 16), "2027-03-16T16:00:00Z"),  # EDT again after 2027-03-14
    ],
)
def test_lock_instant_follows_daylight_saving(day: date, expected_utc: str) -> None:
    assert day.weekday() == 1  # every case is a real Tuesday
    assert pool_spread_lock_utc(day) == pd.Timestamp(expected_utc)


def test_scheduler_opener_capture_lands_after_the_pool_lock() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    start = capture_scheduler.occurrence(schedule["odds_tue_open"], _et(15, 0))
    assert pd.Timestamp(start) > pool_spread_lock_utc(TUESDAY)


# ---------------------------------------------------------------------------
# 2. The live opener rule
# ---------------------------------------------------------------------------


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
    assert row["opener_home_spread"] == pytest.approx(3.75)  # median of the 12:05 lines
    assert (row["opener_min"], row["opener_max"]) == (3.5, 4.0)
    assert row["bookmakers"] == 2
    assert row["observed_at_utc"] == pd.Timestamp("2026-09-08T16:05:00Z")
    assert row["opener_basis"] == OPENER_BASIS_POST_LOCK
    assert list(opener.columns)[-1] == "opener_basis"  # additive: appended last


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


def test_monday_evening_capture_is_tuesday_utc_but_never_beats_a_post_lock_quote() -> None:
    assert MONDAY_EVENING.astimezone(UTC).weekday() == 1  # 01:00Z Tuesday
    with_lock = _history((MONDAY_EVENING, {"book_a": 2.5}), (LOCK_CAPTURE, {"book_a": 3.5}))
    row = tuesday_opener_quotes(with_lock).iloc[0]
    assert row["opener_home_spread"] == 3.5
    assert row["opener_basis"] == OPENER_BASIS_POST_LOCK
    # Alone, it is still a Tuesday (UTC) quote and stands in as the fallback.
    alone = tuesday_opener_quotes(_history((MONDAY_EVENING, {"book_a": 2.5}))).iloc[0]
    assert alone["opener_home_spread"] == 2.5
    assert alone["opener_basis"] == OPENER_BASIS_PRE_LOCK_FALLBACK


def test_tuesday_evening_capture_after_20_et_is_wednesday_utc_and_not_an_opener() -> None:
    assert TUESDAY_LATE_EVENING.astimezone(UTC).weekday() == 2
    assert tuesday_opener_quotes(_history((TUESDAY_LATE_EVENING, {"book_a": 3.0}))).empty


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


# ---------------------------------------------------------------------------
# 3. The store: a restored early snapshot cannot displace the locked line
# ---------------------------------------------------------------------------


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
    assert result["spread_line"].tolist() == [3.5, 1.5]  # provenance only, never the line
    # Null-safe like the instant beside it.
    bare = attach_market_observed_at(frame)
    assert bare[MARKET_OPENER_BASIS_COLUMN].isna().all()


# ---------------------------------------------------------------------------
# 4. The historical archive is a different path and is unchanged
# ---------------------------------------------------------------------------


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
    # The 09:00 ET snapshot is BEFORE the pool lock; the archive keeps it regardless.
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


# ---------------------------------------------------------------------------
# 5. The one-click refresh guard and the line-gap report read the constant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("now", "skipped"),
    [
        (_et(11, 59), True),  # Tuesday, pool not yet locked
        (_et(12, 0), False),  # the lock instant itself: a capture now IS the locked line
        (_et(12, 4), False),  # between the lock and the 12:05 job: still the locked line
        (MONDAY_EVENING, True),  # Tuesday to the UTC day, hours before the lock
        (_et(8, 30, date(2026, 9, 9)), False),  # Wednesday
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


def test_line_gap_report_carries_the_opener_basis(capsys: pytest.CaptureFixture[str]) -> None:
    history = _history(
        (EARLY, {"book_a": 3.0, "book_b": 3.0}),
        (LOCK_CAPTURE, {"book_a": 3.5, "book_b": 4.0}),
        (LATER, {"book_a": 4.5, "book_b": 4.5}),
    )
    table = tuesday_line_gap.tuesday_gap(history, TUESDAY, POOL_SPREAD_LOCK_ET)
    assert "first capture after the lock" in capsys.readouterr().out
    row = table.iloc[0]
    assert row["game_id"] == GAME_ID
    assert row["opener"] == pytest.approx(3.75)
    assert row["opener_basis"] == OPENER_BASIS_POST_LOCK
    assert row["at_lock"] == pytest.approx(3.75)
    assert row["move"] == pytest.approx(0.0)

    early_only = tuesday_line_gap.tuesday_gap(
        _history((EARLY, {"book_a": 3.0})), TUESDAY, POOL_SPREAD_LOCK_ET
    )
    assert early_only.iloc[0]["opener_basis"] == OPENER_BASIS_PRE_LOCK_FALLBACK
