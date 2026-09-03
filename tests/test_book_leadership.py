"""Release-blocking tests for the SKY-04 book-leadership descriptive."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_ats.data import DataContractError
from scripts.book_leadership import score_leadership


def _quotes() -> pd.DataFrame:
    # Two snapshots; game G: book A moves first (earlier provider stamp),
    # book B follows in the same capture window; book C never moves.
    return pd.DataFrame(
        [
            {
                "observed_at_utc": pd.Timestamp("2024-09-10T12:00:00Z"),
                "nflverse_game_id": "2024_01_X_Y",
                "bookmaker_key": book,
                "home_spread_line": -3.0,
                "bookmaker_last_update_utc": pd.Timestamp("2024-09-10T11:00:00Z"),
            }
            for book in ("a_book", "b_book", "c_book")
        ]
        + [
            {
                "observed_at_utc": pd.Timestamp("2024-09-12T12:00:00Z"),
                "nflverse_game_id": "2024_01_X_Y",
                "bookmaker_key": "a_book",
                "home_spread_line": -3.5,
                "bookmaker_last_update_utc": pd.Timestamp("2024-09-11T09:00:00Z"),
            },
            {
                "observed_at_utc": pd.Timestamp("2024-09-12T12:00:00Z"),
                "nflverse_game_id": "2024_01_X_Y",
                "bookmaker_key": "b_book",
                "home_spread_line": -3.5,
                "bookmaker_last_update_utc": pd.Timestamp("2024-09-11T15:00:00Z"),
            },
            {
                "observed_at_utc": pd.Timestamp("2024-09-12T12:00:00Z"),
                "nflverse_game_id": "2024_01_X_Y",
                "bookmaker_key": "c_book",
                "home_spread_line": -3.0,
                "bookmaker_last_update_utc": pd.Timestamp("2024-09-10T11:00:00Z"),
            },
        ]
    )


def test_first_mover_splits_credit_and_opening_snapshot_never_counts() -> None:
    result = score_leadership(_quotes())
    assert result["games"] == 1
    assert result["move_events"] == 1
    table = {row["book"]: row for row in result["books"]}
    assert table["a_book"]["first_move_credits"] == pytest.approx(1.0)
    assert table["a_book"]["move_participations"] == 1
    assert table["a_book"]["leadership_share"] == pytest.approx(1.0)
    assert table["b_book"]["first_move_credits"] == pytest.approx(0.0)
    assert table["b_book"]["move_participations"] == 1
    assert table["b_book"]["leadership_share"] == pytest.approx(0.0)
    assert table["c_book"]["move_participations"] == 0
    assert table["c_book"]["leadership_share"] is None


def test_missing_columns_fail_closed() -> None:
    with pytest.raises(DataContractError, match="missing columns"):
        score_leadership(pd.DataFrame({"bookmaker_key": ["a"]}))
