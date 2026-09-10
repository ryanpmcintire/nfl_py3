import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.sharp_book_movement_features import (
    LEADERSHIP_WEIGHTS,
    refresh_pick,
    sharp_book_movement_features,
)


def games(cutoff: str = "2025-09-07T20:00:00Z") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "g",
                "commence_time_utc": "2025-09-08T00:20:00Z",
                "week_first_commence_utc": "2025-09-05T00:20:00Z",
                "cutoff_utc": cutoff,
            }
        ]
    )


def quotes(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nflverse_game_id": "g",
                "bookmaker_key": book,
                "market": "spreads",
                "home_spread_line": line,
                "observed_at_utc": time,
                "bookmaker_last_update_utc": time,
            }
            for book, time, line in rows
        ]
    )


def test_follow_standardized_home_movement_and_equal_comparison() -> None:
    q = quotes(
        [
            ("bovada", "2025-09-02T16:00Z", 3),
            ("bovada", "2025-09-05T16:00Z", 5),
            ("fanduel", "2025-09-02T16:00Z", 3),
            ("fanduel", "2025-09-05T16:00Z", 1),
        ]
    )
    row = sharp_book_movement_features(q, games()).iloc[0]
    assert row.leader_net_move == pytest.approx(
        (2 * LEADERSHIP_WEIGHTS["bovada"] - 2 * LEADERSHIP_WEIGHTS["fanduel"])
        / (LEADERSHIP_WEIGHTS["bovada"] + LEADERSHIP_WEIGHTS["fanduel"])
    )
    assert row.leader_flag and not row.equal_flag
    assert refresh_pick(pd.Series([False]), pd.Series([row.leader_net_move])).iloc[0]


@pytest.mark.parametrize("late", ["2025-09-05T16:00Z", "2025-09-06T16:00Z"])
def test_move_at_or_after_refresh_cutoff_never_flags(late: str) -> None:
    q = quotes([("bovada", "2025-09-02T16:00Z", 3), ("bovada", late, 8)])
    row = sharp_book_movement_features(q, games("2025-09-05T16:00Z")).iloc[0]
    assert not row.leader_flag and not row.equal_flag
    assert not row.leader_move_observed


def test_thursday_kickoff_caps_supplied_cutoff() -> None:
    g = games()
    g["commence_time_utc"] = "2025-09-05T00:20Z"
    q = quotes([("bovada", "2025-09-02T16:00Z", 3), ("bovada", "2025-09-05T01:00Z", 8)])
    assert not sharp_book_movement_features(q, g).iloc[0].leader_flag


def test_sunday_move_is_not_late_week_and_pool_deadline_caps_monday() -> None:
    q = quotes([("bovada", "2025-09-02T16:00Z", 3), ("bovada", "2025-09-07T21:00Z", 8)])
    row = sharp_book_movement_features(q, games("2025-09-08T23:00Z")).iloc[0]
    assert row.cutoff_utc == pd.Timestamp("2025-09-07T20:00Z")
    assert not row.leader_flag


def test_dst_deadline_uses_local_sunday_clock() -> None:
    g = games().drop(columns="cutoff_utc")
    g["commence_time_utc"] = "2025-11-03T01:20Z"
    g["week_first_commence_utc"] = "2025-10-31T00:20Z"
    row = sharp_book_movement_features(pd.DataFrame(), g).iloc[0]
    assert row.cutoff_utc == pd.Timestamp("2025-11-02T21:00Z")


def test_duplicates_reversions_and_future_provider_timestamp() -> None:
    q = quotes(
        [
            ("bovada", "2025-09-02T16:00Z", 3),
            ("bovada", "2025-09-04T16:00Z", 5),
            ("bovada", "2025-09-05T16:00Z", 3),
            ("bovada", "2025-09-06T16:00Z", 8),
        ]
    )
    q.loc[3, "bookmaker_last_update_utc"] = "2025-09-08T16:00Z"
    row = sharp_book_movement_features(pd.concat([q, q]), games()).iloc[0]
    assert row.leader_move_observed
    assert row.leader_net_move == 0 and not row.leader_flag


def test_first_quote_prior_week_and_unknown_books_cannot_create_move() -> None:
    q = quotes(
        [
            ("bovada", "2025-08-29T16:00Z", 3),
            ("bovada", "2025-09-05T16:00Z", 8),
            ("unknown", "2025-09-02T16:00Z", 3),
            ("unknown", "2025-09-05T16:00Z", 9),
        ]
    )
    row = sharp_book_movement_features(q, games()).iloc[0]
    assert row.eligible_books == 0 and not row.leader_flag


def test_conflicting_duplicate_fails_closed() -> None:
    q = quotes([("bovada", "2025-09-05T16:00Z", 3), ("bovada", "2025-09-05T16:00Z", 8)])
    with pytest.raises(DataContractError, match="Conflicting"):
        sharp_book_movement_features(q, games())


def test_threshold_missing_and_negative_direction() -> None:
    pick = refresh_pick(
        pd.Series([False, True, True, False]), pd.Series([0.5, -0.5, float("nan"), 0.49])
    )
    assert pick.tolist() == [True, False, True, False]
