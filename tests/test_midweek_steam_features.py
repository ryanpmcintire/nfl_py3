"""Point-in-time and direction contracts for LEAD-04."""

import pandas as pd
import pytest

from nfl_ats.midweek_steam_features import (
    attach_midweek_steam,
    decision_cutoff,
    spread_move_events,
)


def quotes(books: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nflverse_game_id": "2024_01_A_B",
                "bookmaker_key": str(book),
                "observed_at_utc": observed,
                "bookmaker_last_update_utc": observed,
                "market": "spreads",
                "home_spread_line": line,
            }
            for book in range(books)
            for observed, line in [("2024-09-04T15:00:00Z", 3.0), ("2024-09-04T16:00:00Z", 3.5)]
        ]
    )


def games(kickoff: str = "2024-09-08T20:25:00Z") -> pd.DataFrame:
    return pd.DataFrame([{"game_id": "2024_01_A_B", "kickoff": kickoff}])


def test_follow_standardized_home_margin_and_distinct_books() -> None:
    event = spread_move_events(pd.concat([quotes(), quotes()], ignore_index=True))
    assert len(event) == 1
    assert event.iloc[0]["steam_side"] == "HOME"
    assert spread_move_events(quotes(2)).empty
    inverse = quotes()
    inverse["home_spread_line"] *= -1
    assert spread_move_events(inverse).iloc[0]["steam_side"] == "AWAY"


def test_after_cutoff_and_future_refresh_event_never_flags() -> None:
    event = spread_move_events(quotes())
    before = attach_midweek_steam(games(), event, as_of="2024-09-04T15:59:59Z")
    assert not before.iloc[0]["midweek_steam_exposure"]
    assert attach_midweek_steam(games(), event).iloc[0]["midweek_steam_exposure"]
    for observed in ["2024-09-06T01:00:00Z", "2024-09-06T01:00:01Z"]:
        later = event.copy()
        later["observed_at_utc"] = pd.to_datetime(observed, utc=True)
        assert not attach_midweek_steam(games("2024-09-06T01:00:00Z"), later).iloc[0][
            "midweek_steam_exposure"
        ]


@pytest.mark.parametrize(
    "kickoff,expected",
    [
        ("2024-09-08T17:00:00Z", "2024-09-08T17:00:00Z"),
        ("2024-09-08T20:25:00Z", "2024-09-08T20:00:00Z"),
        ("2024-09-10T00:15:00Z", "2024-09-08T20:00:00Z"),
        ("2024-11-04T01:20:00Z", "2024-11-03T21:00:00Z"),
    ],
)
def test_deadline_eastern_and_dst(kickoff: str, expected: str) -> None:
    assert decision_cutoff(kickoff) == pd.Timestamp(expected)


def test_old_week_and_weekend_and_conflict_are_noop() -> None:
    event = spread_move_events(quotes())
    assert not attach_midweek_steam(games("2024-09-15T17:00:00Z"), event).iloc[0][
        "midweek_steam_exposure"
    ]
    opposite = event.assign(steam_side="AWAY")
    assert not attach_midweek_steam(games(), pd.concat([event, opposite], ignore_index=True)).iloc[
        0
    ]["midweek_steam_exposure"]
    sunday = event.assign(observed_at_utc=pd.Timestamp("2024-09-08T15:00:00Z"))
    assert not attach_midweek_steam(games(), sunday).iloc[0]["midweek_steam_exposure"]


def test_provider_updates_and_one_hour_window() -> None:
    q = quotes()
    q.loc[q["bookmaker_key"].eq("2"), "bookmaker_last_update_utc"] = "2024-09-04T17:00:00Z"
    assert spread_move_events(q).empty
    q = quotes()
    q.loc[q["bookmaker_key"].eq("2"), "bookmaker_last_update_utc"] = "2024-09-04T14:59:59Z"
    assert spread_move_events(q).empty
    q = quotes()
    q.loc[
        q["bookmaker_key"].eq("2") & q["home_spread_line"].eq(3.5),
        ["observed_at_utc", "bookmaker_last_update_utc"],
    ] = "2024-09-04T17:00:01Z"
    assert spread_move_events(q).empty


def test_first_snapshot_is_not_a_move() -> None:
    assert spread_move_events(quotes().query("home_spread_line == 3.5")).empty


def test_latest_pre_refresh_direction_replaces_earlier_direction() -> None:
    event = spread_move_events(quotes())
    later = event.assign(observed_at_utc=pd.Timestamp("2024-09-05T16:00:00Z"), steam_side="AWAY")
    events = pd.concat([event, later], ignore_index=True)
    assert attach_midweek_steam(games(), events).iloc[0]["midweek_steam_side"] == "AWAY"
    assert (
        attach_midweek_steam(games(), events, as_of="2024-09-04T16:00:00Z").iloc[0][
            "midweek_steam_side"
        ]
        == "HOME"
    )


def test_inclusive_hour_across_snapshots_and_spread_market_only() -> None:
    q = quotes()
    q.loc[
        q["bookmaker_key"].eq("2") & q["home_spread_line"].eq(3.5),
        ["observed_at_utc", "bookmaker_last_update_utc"],
    ] = "2024-09-04T17:00:00Z"
    assert len(spread_move_events(q)) == 1
    q.loc[q["bookmaker_key"].eq("2"), "market"] = "totals"
    assert spread_move_events(q).empty


def test_conflicting_duplicate_line_is_rejected() -> None:
    q = quotes()
    duplicate = q.iloc[[0]].assign(home_spread_line=8.0)
    with pytest.raises(ValueError, match="Conflicting"):
        spread_move_events(pd.concat([q, duplicate], ignore_index=True))
