from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.historical_market import (
    attach_nflverse_games,
    audit_spread_agreement,
    normalize_spreadspoke,
    parse_spreadspoke_archive,
    write_historical_market_snapshot,
)


def _archive(scores: pd.DataFrame, teams: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("spreadspoke_scores.csv", scores.to_csv(index=False))
        archive.writestr("nfl_teams.csv", teams.to_csv(index=False))
    return output.getvalue()


@pytest.fixture
def spreadspoke_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.DataFrame(
        {
            "schedule_date": ["9/8/2024", "9/9/2024", "9/10/2024"],
            "schedule_season": [2024, 2024, 2024],
            "schedule_week": [1, 1, 1],
            "schedule_playoff": [False, False, False],
            "team_home": ["Detroit Lions", "San Francisco 49ers", "Miami Dolphins"],
            "score_home": [20, 30, 17],
            "score_away": [17, 20, 17],
            "team_away": ["Los Angeles Rams", "New York Jets", "Buffalo Bills"],
            "team_favorite_id": ["DET", "SF", "PICK"],
            "spread_favorite": [-3.0, -4.0, 0.0],
            "over_under_line": [52.0, 43.5, 45.0],
        }
    )
    teams = pd.DataFrame(
        {
            "team_name": [
                "Detroit Lions",
                "Los Angeles Rams",
                "San Francisco 49ers",
                "New York Jets",
                "Miami Dolphins",
                "Buffalo Bills",
            ],
            "team_id": ["DET", "LAR", "SF", "NYJ", "MIA", "BUF"],
        }
    )
    return scores, teams


def test_parse_and_normalize_spreadspoke(
    spreadspoke_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    scores, teams = spreadspoke_data
    parsed_scores, parsed_teams = parse_spreadspoke_archive(_archive(scores, teams))
    market = normalize_spreadspoke(parsed_scores, parsed_teams, source_version=82)

    assert market["spread_line"].tolist() == [3.0, 4.0, 0.0]
    assert market.loc[0, "away_team_id"] == "LA"
    assert market["source_line_type"].eq("reported_closing").all()
    assert not market["is_timestamped_quote"].any()
    assert market["observed_at_utc"].isna().all()


def test_attach_and_audit_nflverse_games(
    spreadspoke_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    scores, teams = spreadspoke_data
    market = normalize_spreadspoke(scores, teams)
    reference = pd.DataFrame(
        {
            "game_id": ["2024_01_LA_DET", "2024_01_NYJ_SF", "2024_01_BUF_MIA"],
            "season": [2024, 2024, 2024],
            "gameday": ["2024-09-08", "2024-09-09", "2024-09-10"],
            "home_team": ["DET", "SF", "MIA"],
            "away_team": ["LA", "NYJ", "BUF"],
            "spread_line": [3.0, 3.5, 0.0],
        }
    )
    attached = attach_nflverse_games(market, reference)
    metrics, discrepancies = audit_spread_agreement(attached)

    assert attached["nflverse_game_id"].notna().all()
    assert metrics["compared_spreads"] == 3
    assert metrics["exact_agreement_rate"] == pytest.approx(2 / 3)
    assert metrics["within_half_point_rate"] == 1.0
    assert discrepancies.iloc[0]["absolute_spread_difference"] == 0.5


def test_historical_market_contract_and_snapshot(
    tmp_path: Path,
    spreadspoke_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    scores, teams = spreadspoke_data
    payload = _archive(scores, teams)
    market = normalize_spreadspoke(scores, teams)
    snapshot = write_historical_market_snapshot(
        payload,
        scores,
        teams,
        market,
        tmp_path,
        source_metadata={"version": 82},
        now=datetime(2026, 8, 12, 12, tzinfo=UTC),
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    assert snapshot.snapshot_id == "20260812T120000Z"
    assert snapshot.archive_path.is_file()
    assert manifest["semantics"]["timestamped_quotes"] is False
    assert manifest["rows"]["normalized_market"] == 3

    with pytest.raises(DataContractError, match="missing files"):
        parse_spreadspoke_archive(b"PK\x05\x06" + b"\0" * 18)
