from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from nfl_ats.market_data import (
    attach_nflverse_game_ids,
    closing_line_value,
    latest_book_quotes,
    load_quote_history,
    parse_odds_api_response,
    spread_consensus,
    write_market_snapshot,
)


def _payload(home_point: float = -3.5) -> bytes:
    data = [
        {
            "id": "event-1",
            "sport_key": "americanfootball_nfl",
            "commence_time": "2026-09-10T00:20:00Z",
            "home_team": "Seattle Seahawks",
            "away_team": "New England Patriots",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": "2026-08-12T12:00:00Z",
                    "markets": [
                        {
                            "key": "spreads",
                            "last_update": "2026-08-12T12:00:00Z",
                            "outcomes": [
                                {
                                    "name": "Seattle Seahawks",
                                    "price": -110,
                                    "point": home_point,
                                },
                                {
                                    "name": "New England Patriots",
                                    "price": -110,
                                    "point": -home_point,
                                },
                            ],
                        },
                        {
                            "key": "h2h",
                            "last_update": "2026-08-12T12:00:00Z",
                            "outcomes": [
                                {"name": "Seattle Seahawks", "price": -170},
                                {"name": "New England Patriots", "price": 145},
                            ],
                        },
                    ],
                }
            ],
        }
    ]
    return json.dumps(data, separators=(",", ":")).encode()


def test_parse_match_and_immutable_market_history(tmp_path) -> None:
    first_time = datetime(2026, 8, 12, 12, tzinfo=UTC)
    first_payload = _payload(-3.5)
    first = parse_odds_api_response(first_payload, observed_at=first_time)
    assert len(first) == 4
    assert first.loc[first["market"].eq("spreads"), "home_spread_line"].eq(3.5).all()
    assert set(first["outcome_side"]) == {"HOME", "AWAY"}

    schedules = pd.DataFrame(
        {
            "game_id": ["2026_01_NE_SEA"],
            "home_team": ["SEA"],
            "away_team": ["NE"],
            "kickoff": [pd.Timestamp("2026-09-10T00:20:00Z")],
        }
    )
    first = attach_nflverse_game_ids(first, schedules)
    assert first["nflverse_game_id"].eq("2026_01_NE_SEA").all()
    snapshot = write_market_snapshot(
        first_payload,
        first,
        tmp_path,
        observed_at=first_time,
        request_metadata={"regions": "us", "markets": "spreads,h2h"},
        quota={"requests_remaining": "499"},
    )
    assert snapshot.manifest_path.is_file()

    second_time = first_time + timedelta(hours=1)
    second_payload = _payload(-4.0)
    second = attach_nflverse_game_ids(
        parse_odds_api_response(second_payload, observed_at=second_time), schedules
    )
    write_market_snapshot(
        second_payload,
        second,
        tmp_path,
        observed_at=second_time,
        request_metadata={"regions": "us", "markets": "spreads,h2h"},
    )
    history = load_quote_history(tmp_path)
    assert len(history) == 8
    latest = latest_book_quotes(history)
    assert len(latest) == 4
    consensus = spread_consensus(history)
    assert consensus.iloc[0]["consensus_home_spread"] == 4.0

    decisions = pd.DataFrame(
        {
            "game_id": ["2026_01_NE_SEA"],
            "bookmaker_key": ["draftkings"],
            "bet_side": ["HOME"],
            "spread_line": [3.5],
            "bet_odds": [-110],
            "decision_time_utc": [first_time],
        }
    )
    clv = closing_line_value(decisions, history)
    assert clv.iloc[0]["clv_points"] == pytest.approx(0.5)
    assert clv.iloc[0]["closing_home_spread"] == 4.0


def test_market_contract_guards(tmp_path) -> None:
    with pytest.raises(ValueError, match="JSON array"):
        parse_odds_api_response(b"{}")
    quotes = parse_odds_api_response(_payload(), observed_at=datetime(2026, 8, 12, tzinfo=UTC))
    with pytest.raises(ValueError, match="Schedule matching"):
        attach_nflverse_game_ids(quotes, pd.DataFrame({"game_id": []}))
    with pytest.raises(ValueError, match="CLV decisions"):
        closing_line_value(pd.DataFrame({"game_id": []}), quotes)
    assert load_quote_history(tmp_path).empty
