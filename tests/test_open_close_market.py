from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.open_close_market import (
    BOOK_PREFIXES,
    normalize_open_close_sample,
    parse_open_close_archive,
    summarize_open_close_games,
    write_open_close_snapshot,
)


def _sample_table() -> pd.DataFrame:
    row: dict[str, object] = {
        "Season": 2025,
        "Date": "2025-09-04",
        "Away_Team": "Dallas",
        "Home_Team": "Philadelphia",
        "Away_Score": 20,
        "Home_Score": 24,
        "ML_Away_Bet%": 86,
        "ML_Home_Bet%": 14,
        "Spread_Away_Bet%": 58,
        "Spread_Home_Bet%": 42,
        "Total_Over_Bet%": 68,
        "Total_Under_Bet%": 32,
    }
    for index, prefix in enumerate(BOOK_PREFIXES):
        home_line = -7.0 if prefix == "Opener" else -7.5 - (index % 2) * 0.5
        row.update(
            {
                f"{prefix}_ML_Away_US": 300,
                f"{prefix}_ML_Home_US": -400,
                f"{prefix}_Spread_Away_Line": -home_line,
                f"{prefix}_Spread_Away_US": -110,
                f"{prefix}_Spread_Home_Line": home_line,
                f"{prefix}_Spread_Home_US": -110,
                f"{prefix}_Total_Line": 47.5,
                f"{prefix}_Total_Over_US": -110,
                f"{prefix}_Total_Under_US": -110,
            }
        )
    return pd.DataFrame([row])


def _archive(table: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("sample.csv", table.to_csv(index=False, sep=";"))
    return output.getvalue()


def test_parse_normalize_and_movement_contract() -> None:
    table = _sample_table()
    parsed = parse_open_close_archive(_archive(table))
    quotes = normalize_open_close_sample(parsed, source_version=1, raw_archive_sha256="abc")
    assert len(quotes) == 60
    assert set(quotes["quote_stage"]) == {"opening", "reported_closing"}
    assert not quotes["is_timestamped_quote"].any()
    assert quotes["observed_at_utc"].isna().all()
    opener_home = quotes.loc[
        quotes["bookmaker_key"].eq("opener")
        & quotes["market"].eq("spreads")
        & quotes["outcome_side"].eq("HOME")
    ].iloc[0]
    assert opener_home["line"] == -7.0
    assert opener_home["home_spread_line"] == 7.0

    reference = pd.DataFrame(
        {
            "game_id": ["2025_01_DAL_PHI"],
            "season": [2025],
            "gameday": ["2025-09-04"],
            "home_team": ["PHI"],
            "away_team": ["DAL"],
            "spread_line": [8.0],
        }
    )
    games = summarize_open_close_games(quotes, reference)
    assert games.iloc[0]["nflverse_game_id"] == "2025_01_DAL_PHI"
    assert games.iloc[0]["opening_home_spread"] == 7.0
    assert games.iloc[0]["open_to_close_movement"] == pytest.approx(1.0)


def test_open_close_snapshot_and_contract_guards(tmp_path: Path) -> None:
    table = _sample_table()
    payload = _archive(table)
    quotes = normalize_open_close_sample(table)
    games = summarize_open_close_games(quotes)
    snapshot = write_open_close_snapshot(
        payload,
        table,
        quotes,
        games,
        tmp_path,
        source_metadata={"version": 1, "license": "CC BY-NC 4.0"},
        now=datetime(2026, 8, 12, 12, tzinfo=UTC),
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    assert snapshot.snapshot_id == "20260812T120000Z"
    assert manifest["rows"] == {"source_games": 1, "normalized_quotes": 60}
    assert manifest["semantics"]["timestamped_quotes"] is False

    malformed = io.BytesIO()
    with zipfile.ZipFile(malformed, "w") as archive:
        archive.writestr("one.csv", "a\n1\n")
        archive.writestr("two.csv", "a\n2\n")
    with pytest.raises(DataContractError, match="one CSV"):
        parse_open_close_archive(malformed.getvalue())


def test_opposite_spread_and_moneyline_favorites_are_flagged() -> None:
    table = _sample_table()
    table.loc[0, "Opener_Spread_Away_Line"] = -7.0
    table.loc[0, "Opener_Spread_Home_Line"] = 7.0
    quotes = normalize_open_close_sample(table)
    opener = quotes.loc[quotes["bookmaker_key"].eq("opener")]
    assert not opener["spread_moneyline_direction_consistent"].any()
    games = summarize_open_close_games(quotes)
    assert games.iloc[0]["reported_opening_home_spread"] == -7.0
    assert pd.isna(games.iloc[0]["opening_home_spread"])
    assert pd.isna(games.iloc[0]["open_to_close_movement"])
