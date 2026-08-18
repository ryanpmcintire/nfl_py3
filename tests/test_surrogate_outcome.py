from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.margin import margin_feature_columns
from nfl_ats.market_data import attach_nflverse_game_ids
from nfl_ats.odds_backfill import (
    BackfillTarget,
    parse_historical_odds_response,
    store_historical_snapshot,
)
from nfl_ats.surrogate_outcome import (
    DEFAULT_MOVEMENT_MIN_TRAIN_GAMES,
    fit_movement_target_model,
    movement_agreement,
    movement_agreement_rate,
)

# ---------------------------------------------------------------------------
# movement_agreement / movement_agreement_rate: pure functions, no fixtures.
# ---------------------------------------------------------------------------


def test_movement_agreement_matches_expected_directional_convention() -> None:
    scored = pd.DataFrame(
        {
            "pick_home_at_open": [True, True, False, False],
            "open_move": [0.5, -0.5, -0.5, 0.5],
        }
    )
    result = movement_agreement(scored)
    # Row 0: picked home, line moved toward home (positive) -> agree.
    # Row 1: picked home, line moved away from home -> disagree.
    # Row 2: picked away, line moved toward away -> agree.
    # Row 3: picked away, line moved toward home -> disagree.
    assert result.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_movement_agreement_is_nan_when_the_line_never_moved() -> None:
    scored = pd.DataFrame({"pick_home_at_open": [True, False], "open_move": [0.0, 0.0]})
    result = movement_agreement(scored)
    assert result.isna().all()


def test_movement_agreement_requires_its_columns() -> None:
    with pytest.raises(DataContractError, match="missing columns"):
        movement_agreement(pd.DataFrame({"pick_home_at_open": [True]}))


def test_movement_agreement_rate_excludes_pushes_from_the_denominator() -> None:
    scored = pd.DataFrame(
        {
            "pick_home_at_open": [True, True, False],
            "open_move": [0.5, -0.5, 0.0],
        }
    )
    summary = movement_agreement_rate(scored)
    assert summary["movement_agreement_games"] == 2.0
    assert summary["movement_agreement_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# fit_movement_target_model: leak-safety + shape, via a synthetic snapshot store.
# ---------------------------------------------------------------------------


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
def movement_fit_setup(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    features = _pilot_features_frame()
    root = tmp_path / "raw"
    # Several paired games spread across both seasons so at least one week
    # clears a small min_train_games threshold.
    pairs = {
        30: (2.5, 3.5),
        35: (1.0, 0.5),
        40: (-1.0, -2.5),
        45: (0.5, 1.5),
        55: (2.5, 3.5),
        65: (-1.0, -2.5),
    }
    for idx, (tue_open, close) in pairs.items():
        _store_tue_and_close_for_game(
            root, features, features.iloc[idx], tue_open=tue_open, close=close
        )
    return root, features


def test_fit_movement_target_model_returns_expected_shape(
    movement_fit_setup: tuple[Path, pd.DataFrame],
) -> None:
    root, features = movement_fit_setup
    scored = fit_movement_target_model(root, features, feature_profile="base", min_train_games=2)
    assert not scored.empty
    for column in (
        "game_id",
        "season",
        "week",
        "result",
        "tue_open_home_spread",
        "open_move",
        "predicted_open_move",
        "pick_home_at_open",
        "correct_at_open",
    ):
        assert column in scored.columns
    # The known-paired games' open_move must match tue_open/close exactly.
    row = scored.set_index("game_id").loc["G045"]
    assert row["open_move"] == pytest.approx(1.5 - 0.5)


def test_fit_movement_target_model_never_trains_on_same_or_later_games(
    movement_fit_setup: tuple[Path, pd.DataFrame],
) -> None:
    """Leak-safety regression: swapping a later paired game's movement label
    for an absurd outlier must not change any prediction for an EARLIER
    scored week. If it did, the model would have trained on future data.
    """

    root, features = movement_fit_setup
    baseline = fit_movement_target_model(root, features, feature_profile="base", min_train_games=2)

    poisoned_root = root.parent / "poisoned"
    poisoned_features = _pilot_features_frame()
    pairs = {
        30: (2.5, 3.5),
        35: (1.0, 0.5),
        40: (-1.0, -2.5),
        45: (0.5, 1.5),
        55: (2.5, 3.5),
        65: (-1.0, 999.0),  # last (latest) paired game's movement, poisoned
    }
    for idx, (tue_open, close) in pairs.items():
        _store_tue_and_close_for_game(
            poisoned_root,
            poisoned_features,
            poisoned_features.iloc[idx],
            tue_open=tue_open,
            close=close,
        )
    poisoned = fit_movement_target_model(
        poisoned_root, poisoned_features, feature_profile="base", min_train_games=2
    )

    baseline_indexed = baseline.set_index("game_id")
    poisoned_indexed = poisoned.set_index("game_id")
    earlier_games = [
        game_id
        for game_id in baseline_indexed.index
        if game_id != "G065" and game_id in poisoned_indexed.index
    ]
    assert earlier_games, "fixture must score at least one earlier game"
    for game_id in earlier_games:
        assert baseline_indexed.loc[game_id, "predicted_open_move"] == pytest.approx(
            poisoned_indexed.loc[game_id, "predicted_open_move"]
        ), f"{game_id} prediction changed when a LATER game's outcome was poisoned (leak)"


def test_fit_movement_target_model_requires_paired_snapshots(tmp_path: Path) -> None:
    features = _pilot_features_frame()
    with pytest.raises(ValueError, match="snapshots with decision quotes"):
        fit_movement_target_model(tmp_path / "raw", features, feature_profile="base")


def test_movement_agreement_applies_directly_to_fit_movement_target_model_output(
    movement_fit_setup: tuple[Path, pd.DataFrame],
) -> None:
    root, features = movement_fit_setup
    scored = fit_movement_target_model(root, features, feature_profile="base", min_train_games=2)
    summary = movement_agreement_rate(scored)
    assert 0.0 <= summary["movement_agreement_rate"] <= 1.0
    assert summary["movement_agreement_games"] <= len(scored)


def test_default_movement_min_train_games_is_lower_than_the_real_model_default() -> None:
    # The adversarial control only ever trains on paired (2020+) games, so it
    # needs a lower floor than the real market-residual model's 500-game
    # default (which draws on the full pre-2020 archive) or it would be
    # starved out of most of the window it is meant to test against.
    assert DEFAULT_MOVEMENT_MIN_TRAIN_GAMES < 500
