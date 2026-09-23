from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
from nfl_ats.clv import (
    opener_evaluation_metrics,
    score_clv,
)
from nfl_ats.clv import (
    record_paper_decisions as _record_paper_decisions,
)
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_json
from nfl_ats.margin import margin_feature_columns
from nfl_ats.market_data import (
    attach_nflverse_game_ids,
    parse_odds_api_response,
    write_market_snapshot,
)
from nfl_ats.odds_backfill import (
    BackfillTarget,
    parse_historical_odds_response,
    store_historical_snapshot,
)


def record_paper_decisions(artifacts_root: Path, *, now: datetime | None = None) -> dict[str, Any]:

    return _record_paper_decisions(
        artifacts_root,
        now=now,
        require_fresh_arrest_overlay=False,
    )


def _spread_book(key: str, standardized_home_spread: float, *, price: int = -110) -> dict[str, Any]:
    home_raw = -standardized_home_spread
    return {
        "key": key,
        "title": key,
        "last_update": "2024-09-10T12:00:00Z",
        "markets": [
            {
                "key": "spreads",
                "last_update": "2024-09-10T12:00:00Z",
                "outcomes": [
                    {"name": "__HOME__", "price": price, "point": home_raw},
                    {"name": "__AWAY__", "price": price, "point": standardized_home_spread},
                ],
            }
        ],
    }


def _asymmetric_spread_book(
    key: str, standardized_home_spread: float, *, home_price: int, away_price: int
) -> dict[str, Any]:

    home_raw = -standardized_home_spread
    return {
        "key": key,
        "title": key,
        "last_update": "2024-09-10T12:00:00Z",
        "markets": [
            {
                "key": "spreads",
                "last_update": "2024-09-10T12:00:00Z",
                "outcomes": [
                    {"name": "__HOME__", "price": home_price, "point": home_raw},
                    {"name": "__AWAY__", "price": away_price, "point": standardized_home_spread},
                ],
            }
        ],
    }


def _event(
    event_id: str,
    home_name: str,
    away_name: str,
    commence_time: str,
    books: list[dict[str, Any]],
) -> dict[str, Any]:
    resolved_books = []
    for book in books:
        markets = []
        for market in book["markets"]:
            outcomes = []
            for outcome in market["outcomes"]:
                name = home_name if outcome["name"] == "__HOME__" else away_name
                outcomes.append({**outcome, "name": name})
            markets.append({**market, "outcomes": outcomes})
        resolved_books.append({**book, "markets": markets})
    return {
        "id": event_id,
        "sport_key": "americanfootball_nfl",
        "commence_time": commence_time,
        "home_team": home_name,
        "away_team": away_name,
        "bookmakers": resolved_books,
    }


def _historical_target(
    season: int,
    week: int,
    label: str,
    requested_at: datetime,
    *,
    markets: str = "spreads,totals,h2h",
) -> BackfillTarget:
    return BackfillTarget(
        season=season,
        week=week,
        label=label,
        requested_at_utc=requested_at,
        markets=markets,
        regions="us",
        credits=10,
    )


def _store_snapshot(
    root: Path,
    schedule: pd.DataFrame,
    *,
    season: int,
    week: int,
    label: str,
    snapshot_time: str,
    events: list[dict[str, Any]],
) -> None:
    wrapper = {
        "timestamp": snapshot_time,
        "previous_timestamp": None,
        "next_timestamp": None,
        "data": events,
    }
    payload = json.dumps(wrapper, separators=(",", ":")).encode()
    capture = parse_historical_odds_response(payload)
    matched = attach_nflverse_game_ids(capture.quotes, schedule)
    capture = type(capture)(
        snapshot_at_utc=capture.snapshot_at_utc,
        previous_snapshot_at_utc=capture.previous_snapshot_at_utc,
        next_snapshot_at_utc=capture.next_snapshot_at_utc,
        quotes=matched,
    )
    target = _historical_target(
        season, week, label, datetime.fromisoformat(snapshot_time.replace("Z", "+00:00"))
    )
    store_historical_snapshot(payload, capture, root, target=target)


@pytest.fixture
def two_game_schedule() -> pd.DataFrame:
    return pd.DataFrame(
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


@pytest.fixture
def two_game_store(tmp_path: Path, two_game_schedule: pd.DataFrame) -> Path:
    root = tmp_path / "raw"

    tue_events = [
        _event(
            "kc-cin",
            "Kansas City Chiefs",
            "Cincinnati Bengals",
            "2024-09-13T00:20:00Z",
            [_spread_book("book_a", 1.5), _spread_book("book_b", 1.5), _spread_book("book_c", 1.5)],
        ),
        _event(
            "sea-ne",
            "Seattle Seahawks",
            "New England Patriots",
            "2024-09-15T17:00:00Z",
            [_spread_book("book_a", 2.0), _spread_book("book_b", 2.5), _spread_book("book_c", 3.0)],
        ),
    ]
    _store_snapshot(
        root,
        two_game_schedule,
        season=2024,
        week=2,
        label="tue_open",
        snapshot_time="2024-09-10T13:00:00Z",
        events=tue_events,
    )

    thu_events = [
        _event(
            "kc-cin",
            "Kansas City Chiefs",
            "Cincinnati Bengals",
            "2024-09-13T00:20:00Z",
            [_spread_book("book_a", 1.0), _spread_book("book_b", 1.0)],
        ),
        _event(
            "sea-ne",
            "Seattle Seahawks",
            "New England Patriots",
            "2024-09-15T17:00:00Z",
            [_spread_book("book_a", 2.5), _spread_book("book_b", 2.5)],
        ),
    ]
    _store_snapshot(
        root,
        two_game_schedule,
        season=2024,
        week=2,
        label="thu_pre_tnf",
        snapshot_time="2024-09-12T22:00:00Z",
        events=thu_events,
    )

    sun_early_events = [
        _event(
            "kc-cin",
            "Kansas City Chiefs",
            "Cincinnati Bengals",
            "2024-09-13T00:20:00Z",
            [_spread_book("book_a", -1.0)],
        ),
        _event(
            "sea-ne",
            "Seattle Seahawks",
            "New England Patriots",
            "2024-09-15T17:00:00Z",
            [_spread_book("book_a", 3.5), _spread_book("book_b", 3.5)],
        ),
    ]
    _store_snapshot(
        root,
        two_game_schedule,
        season=2024,
        week=2,
        label="sun_early_close",
        snapshot_time="2024-09-15T16:00:00Z",
        events=sun_early_events,
    )

    sun_late_events = [
        _event(
            "sea-ne",
            "Seattle Seahawks",
            "New England Patriots",
            "2024-09-15T17:00:00Z",
            [_spread_book("book_a", 4.0)],
        ),
    ]
    _store_snapshot(
        root,
        two_game_schedule,
        season=2024,
        week=2,
        label="sun_late_close",
        snapshot_time="2024-09-15T20:15:00Z",
        events=sun_late_events,
    )
    return root


def test_score_clv_hand_computed_including_key_number_crossing() -> None:
    pairing = pd.DataFrame(
        {
            "game_id": ["G1", "G1", "G2"],
            "decision_label": ["tue_open", "tue_open", "tue_open"],
            "home_spread": [2.5, 2.5, -1.0],
            "spread_books": [4, 4, 5],
        }
    ).drop_duplicates(["game_id", "decision_label"])
    close_reference = pd.DataFrame(
        {
            "game_id": ["G1", "G2"],
            "close_home_spread": [3.5, -2.0],
            "close_source": ["sun_late_close", "sun_late_close"],
            "close_books": [6, 6],
        }
    )
    picks = pd.DataFrame(
        {
            "game_id": ["G1", "G2"],
            "side": ["HOME", "AWAY"],
            "decision_label": ["tue_open", "tue_open"],
            "season": [2024, 2024],
            "week": [2, 2],
        }
    )
    scored = score_clv(picks, pairing, close_reference)
    assert scored.loc[scored["game_id"].eq("G1"), "clv_points"].iloc[0] == pytest.approx(1.0)
    assert scored.loc[scored["game_id"].eq("G2"), "clv_points"].iloc[0] == pytest.approx(1.0)


def test_score_clv_contract_guards() -> None:
    pairing = pd.DataFrame(columns=["game_id", "decision_label", "home_spread", "spread_books"])
    close_reference = pd.DataFrame(
        columns=["game_id", "close_home_spread", "close_source", "close_books"]
    )
    with pytest.raises(DataContractError, match="missing columns"):
        score_clv(pd.DataFrame({"game_id": ["G1"]}), pairing, close_reference)
    with pytest.raises(ValueError, match="unsupported sides"):
        score_clv(
            pd.DataFrame({"game_id": ["G1"], "side": ["PUSH"], "decision_label": ["tue_open"]}),
            pairing,
            close_reference,
        )


def _pilot_features_frame(n_games: int = 70) -> pd.DataFrame:
    index = np.arange(n_games)
    seasons = np.where(index < 50, 2020, 2021)
    weeks = np.where(index < 50, (index % 17) + 1, ((index - 50) % 17) + 1)
    start = pd.Timestamp("2020-09-10", tz="UTC")
    gamedays = [start + pd.Timedelta(days=7 * int(value)) for value in index]
    frame = pd.DataFrame(
        {
            "game_id": [f"G{value:03d}" for value in index],
            "season": seasons,
            "week": weeks,
            "gameday": gamedays,
            "home_team": "SEA",
            "away_team": "NE",
            "rest_diff": ((index % 5) - 2).astype(float),
        }
    )
    for feature_index, column in enumerate(
        margin_feature_columns("market_residual", "base"), start=1
    ):
        frame[column] = np.sin(index / feature_index) + (index % 5) / 10.0
    frame["spread_line"] = np.where(index % 2 == 0, 2.5, -1.5)
    frame["ats_margin"] = np.where(index % 3 == 0, 3.0, -3.0)
    frame["result"] = frame["spread_line"] + frame["ats_margin"]
    return frame


def _store_tue_and_close_for_game(
    root: Path, features: pd.DataFrame, game_row: pd.Series, *, tue_open: float, close: float
) -> None:
    schedule = features.loc[features["game_id"].eq(game_row["game_id"])][
        ["game_id", "home_team", "away_team", "gameday"]
    ].rename(columns={"gameday": "kickoff"})
    commence = game_row["gameday"].strftime("%Y-%m-%dT%H:%M:%SZ")
    event_id = f"evt-{game_row['game_id']}"
    tue_time = (game_row["gameday"] - pd.Timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    close_time = (game_row["gameday"] - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _store_snapshot(
        root,
        schedule,
        season=int(game_row["season"]),
        week=int(game_row["week"]),
        label="tue_open",
        snapshot_time=tue_time,
        events=[
            _event(
                event_id,
                "Seattle Seahawks",
                "New England Patriots",
                commence,
                [_spread_book("book_a", tue_open)],
            )
        ],
    )
    _store_snapshot(
        root,
        schedule,
        season=int(game_row["season"]),
        week=int(game_row["week"]),
        label="sun_late_close",
        snapshot_time=close_time,
        events=[
            _event(
                event_id,
                "Seattle Seahawks",
                "New England Patriots",
                commence,
                [_spread_book("book_a", close)],
            )
        ],
    )


@pytest.fixture
def pilot_setup(tmp_path: Path) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    features = _pilot_features_frame()
    root = tmp_path / "raw"
    for idx, (tue_open, close) in zip((55, 65), ((2.5, 3.5), (-1.0, -2.5)), strict=True):
        _store_tue_and_close_for_game(
            root, features, features.iloc[idx], tue_open=tue_open, close=close
        )
    config = {
        "feature_profile": "base",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "target": "market_residual",
        "probability_method": "ecdf",
    }
    return root, features, config


def _live_features_frame() -> pd.DataFrame:

    features = _pilot_features_frame(n_games=71)
    features.loc[features.index[-1], ["result", "ats_margin"]] = np.nan
    return features


def _store_live_tuesday_snapshot(
    root: Path, features: pd.DataFrame, game_row: pd.Series, *, home_spread: float
) -> None:

    schedule = features.loc[features["game_id"].eq(game_row["game_id"])][
        ["game_id", "home_team", "away_team", "gameday"]
    ].rename(columns={"gameday": "kickoff"})
    commence = game_row["gameday"].strftime("%Y-%m-%dT%H:%M:%SZ")
    observed = (game_row["gameday"] - pd.Timedelta(days=2) + pd.Timedelta(hours=13)).to_pydatetime()
    events = [
        _event(
            f"live-{game_row['game_id']}",
            "Seattle Seahawks",
            "New England Patriots",
            commence,
            [_spread_book("book_a", home_spread), _spread_book("book_b", home_spread)],
        )
    ]
    payload = json.dumps(events).encode()
    quotes = parse_odds_api_response(payload, observed_at=observed)
    quotes = attach_nflverse_game_ids(quotes, schedule)
    write_market_snapshot(
        payload,
        quotes,
        root,
        observed_at=observed,
        request_metadata={"sport": "americanfootball_nfl"},
    )


def _published_card_artifacts(
    tmp_path: Path,
    *,
    kickoffs: list[str],
    spread_lines: list[float],
    probabilities: list[float],
    bet_sides: list[str],
    method: str = "market_residual",
    card_method: str | None = None,
    sweep_widths: list[float] | None = None,
    game_type: str | None = None,
) -> Path:
    from test_cli import _seed_pick_probability

    artifacts = tmp_path / "artifacts"
    forecast_relative = "margin_predictions/2026-week-01-test"
    forecast_dir = artifacts / forecast_relative
    count = len(kickoffs)
    card = pd.DataFrame(
        {
            "game_id": [f"2026_01_A{index}_H{index}" for index in range(count)],
            "season": 2026,
            "week": 1,
            "kickoff": kickoffs,
            "away_team": [f"A{index}" for index in range(count)],
            "home_team": [f"H{index}" for index in range(count)],
            "spread_line": spread_lines,
            "home_cover_probability": probabilities,
            "bet_side": bet_sides,
            "edge": 0.05,
            "method": card_method or method,
        }
    )
    if game_type is not None:
        card["game_type"] = game_type
    forecast_dir.mkdir(parents=True, exist_ok=True)
    card.to_csv(forecast_dir / "recommendations.csv", index=False)
    if sweep_widths is not None:
        offsets = np.arange(-4.0, 4.5, 0.5)
        frames = []
        for index, width in enumerate(sweep_widths):
            holds = np.abs(offsets) <= width
            home = probabilities[index] >= 0.5
            probability = np.where(holds, 0.6, 0.4) if home else np.where(holds, 0.4, 0.6)
            frames.append(
                pd.DataFrame(
                    {
                        "game_id": f"2026_01_A{index}_H{index}",
                        "line_offset": offsets,
                        "home_cover_probability": probability,
                        "method": card_method or method,
                    }
                )
            )
        pd.concat(frames, ignore_index=True).to_parquet(forecast_dir / "line_sweep.parquet")
    atomic_json(
        {
            "active_model_id": "model-1",
            "synchronization_status": "SYNCHRONIZED",
            "created_at_utc": "2026-08-16T00:00:00+00:00",
        },
        forecast_dir / "metadata.json",
    )
    atomic_json(
        {
            "version": ACTIVE_ATS_MODEL_VERSION,
            "status": "SYNCHRONIZED",
            "method": method,
            "model_id": "model-1",
            "weekly_forecast": {"artifact": forecast_relative, "season": 2026, "week": 1},
        },
        artifacts / "active_ats_model.json",
    )
    _seed_pick_probability(artifacts, model_logit=1.0, flag_sum=1.0)
    return artifacts


def _store_live_capture(
    root: Path,
    schedule: pd.DataFrame,
    game_row: pd.Series,
    *,
    home_spread: float,
    observed_at: datetime,
) -> None:
    events = [
        _event(
            f"live-{game_row['game_id']}",
            str(game_row["home_name"]),
            str(game_row["away_name"]),
            pd.Timestamp(game_row["kickoff"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            [_spread_book("book_a", home_spread), _spread_book("book_b", home_spread)],
        )
    ]
    payload = json.dumps(events).encode()
    quotes = parse_odds_api_response(payload, observed_at=observed_at)
    quotes = attach_nflverse_game_ids(quotes, schedule)
    write_market_snapshot(
        payload,
        quotes,
        root,
        observed_at=observed_at,
        request_metadata={"sport": "americanfootball_nfl"},
    )


_ALL_PRE_KICKOFF = ["2026-09-13T17:00:00+00:00", "2026-09-13T20:25:00+00:00"]


def test_opener_evaluation_metrics_omits_probability_rule_keys_when_columns_absent() -> None:

    scored = pd.DataFrame(
        {
            "correct_at_open": [1.0, 0.0, 1.0],
            "correct_at_close": [1.0, 1.0, 0.0],
            "oracle_correct_at_open": [1.0, np.nan, 0.0],
        }
    )
    metrics = opener_evaluation_metrics(scored)
    for name in (
        "opener_accuracy",
        "close_accuracy",
        "opener_minus_close",
        "opener_vs_coin_flip",
        "movement_oracle_accuracy",
    ):
        assert name in metrics
    for name in (
        "opener_accuracy_probability_rule",
        "close_accuracy_probability_rule",
        "opener_minus_close_probability_rule",
        "opener_vs_coin_flip_probability_rule",
    ):
        assert name not in metrics
