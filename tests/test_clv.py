from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
from nfl_ats.clv import (
    FROZEN_PILOT_FEATURES,
    FROZEN_PILOT_PROTOCOL,
    ClosePredictionUnavailable,
    PilotProtocolBlocked,
    _validate_close_predictions,
    active_model_residual_at_opener,
    assert_monotone_decision_timeline,
    build_pairing_table,
    build_pilot_frame,
    close_reference_table,
    clv_summary,
    decision_market_consensus,
    evaluate_pilot,
    fit_pilot_model,
    key_number_distance,
    live_tuesday_openers,
    load_decision_quotes,
    load_snapshot_manifest_index,
    predict_close_for_week,
    resolve_active_model_config,
    run_predeclared_pilot,
    score_clv,
    sign_test_pilot_b,
    threshold_policy_clv,
    upcoming_week,
    week_blocked_bootstrap,
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
    HISTORICAL_CAPTURE_KIND,
    BackfillTarget,
    parse_historical_odds_response,
    store_historical_snapshot,
)

# ---------------------------------------------------------------------------
# Fixture helpers: synthetic historical-backfill snapshot directories
# ---------------------------------------------------------------------------


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


@pytest.fixture
def two_game_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2024_02_CIN_KC", "2024_02_NE_SEA"],
            "home_team": ["KC", "SEA"],
            "away_team": ["CIN", "NE"],
            "kickoff": [
                pd.Timestamp("2024-09-13T00:20:00Z"),  # Thursday night game
                pd.Timestamp("2024-09-15T17:00:00Z"),  # Sunday early game
            ],
        }
    )


@pytest.fixture
def two_game_store(tmp_path: Path, two_game_schedule: pd.DataFrame) -> Path:
    root = tmp_path / "raw"

    # tue_open: both games, 3 books each.
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

    # thu_pre_tnf: both games still upcoming.
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

    # sun_early_close: KC-CIN has already kicked off (excluded per-game); SEA-NE still upcoming.
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

    # sun_late_close: captured after SEA-NE has already kicked off; must be excluded entirely.
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


# ---------------------------------------------------------------------------
# 1. Snapshot pairing contracts
# ---------------------------------------------------------------------------


def test_manifest_index_skips_in_progress_directory(
    tmp_path: Path, two_game_schedule: pd.DataFrame
) -> None:
    root = tmp_path / "raw"
    _store_snapshot(
        root,
        two_game_schedule,
        season=2024,
        week=2,
        label="tue_open",
        snapshot_time="2024-09-10T13:00:00Z",
        events=[
            _event(
                "kc-cin",
                "Kansas City Chiefs",
                "Cincinnati Bengals",
                "2024-09-13T00:20:00Z",
                [_spread_book("book_a", 1.5)],
            )
        ],
    )
    # An in-progress directory: files present, manifest not yet written.
    partial = root / "20990101T000000Z"
    partial.mkdir(parents=True)
    (partial / "response.json").write_text("{}", encoding="utf-8")

    index = load_snapshot_manifest_index(root)
    assert len(index) == 1
    assert index.iloc[0]["decision_label"] == "tue_open"
    assert index.iloc[0]["capture_kind"] == HISTORICAL_CAPTURE_KIND


def test_pairing_table_consensus_dispersion_and_post_kickoff_exclusion(
    two_game_store: Path,
) -> None:
    pairing = build_pairing_table(two_game_store, capture_kind=HISTORICAL_CAPTURE_KIND)
    assert set(pairing["capture_kind"]) == {HISTORICAL_CAPTURE_KIND}

    sea_ne_tue = pairing.loc[
        pairing["game_id"].eq("2024_02_NE_SEA") & pairing["decision_label"].eq("tue_open")
    ].iloc[0]
    assert sea_ne_tue["spread_books"] == 3
    assert sea_ne_tue["home_spread"] == pytest.approx(2.5)
    assert sea_ne_tue["spread_min"] == pytest.approx(2.0)
    assert sea_ne_tue["spread_max"] == pytest.approx(3.0)
    assert sea_ne_tue["spread_std"] == pytest.approx(0.5)

    # A snapshot captured after a game's own kickoff must never appear as one
    # of that game's pregame rows, even though it is pregame for the week.
    assert pairing.loc[
        pairing["game_id"].eq("2024_02_NE_SEA") & pairing["decision_label"].eq("sun_late_close")
    ].empty
    # The Thursday game has no pregame Sunday rows at all.
    assert pairing.loc[
        pairing["game_id"].eq("2024_02_CIN_KC")
        & pairing["decision_label"].isin(["sun_early_close", "sun_late_close"])
    ].empty


def test_close_reference_prefers_store_then_falls_back_to_schedule(two_game_store: Path) -> None:
    pairing = build_pairing_table(two_game_store, capture_kind=HISTORICAL_CAPTURE_KIND)
    schedule = pd.DataFrame(
        {"game_id": ["2024_02_CIN_KC", "2024_02_NE_SEA"], "spread_line": [1.0, 3.75]}
    )
    close = close_reference_table(pairing, schedule)
    by_game = close.set_index("game_id")

    # SEA-NE has a pregame sun_early_close row (sun_late_close was excluded as
    # post-kickoff), so the store close is used, not the schedule fallback.
    assert by_game.loc["2024_02_NE_SEA", "close_source"] == "sun_early_close"
    assert by_game.loc["2024_02_NE_SEA", "close_home_spread"] == pytest.approx(3.5)
    assert by_game.loc["2024_02_NE_SEA", "close_books"] == 2

    # KC-CIN (Thursday game) has no store close label at all -> schedule fallback.
    assert by_game.loc["2024_02_CIN_KC", "close_source"] == "schedule_close"
    assert by_game.loc["2024_02_CIN_KC", "close_home_spread"] == pytest.approx(1.0)
    assert by_game.loc["2024_02_CIN_KC", "close_books"] == 0


def test_decision_market_consensus_requires_columns() -> None:
    with pytest.raises(DataContractError, match="missing columns"):
        decision_market_consensus(pd.DataFrame({"game_id": []}))


def test_monotone_decision_timeline_detects_reordering() -> None:
    consensus = pd.DataFrame(
        {
            "nflverse_game_id": ["G1", "G1"],
            "decision_label": ["tue_open", "sat_midday"],
            "snapshot_timestamp_utc": pd.to_datetime(
                ["2024-09-14T16:00:00Z", "2024-09-10T13:00:00Z"], utc=True
            ),
        }
    )
    with pytest.raises(DataContractError, match="Non-monotone"):
        assert_monotone_decision_timeline(consensus)


def test_load_decision_quotes_tags_live_capture_kind(
    tmp_path: Path, two_game_schedule: pd.DataFrame
) -> None:
    root = tmp_path / "raw"
    payload_events = [
        _event(
            "kc-cin",
            "Kansas City Chiefs",
            "Cincinnati Bengals",
            "2024-09-13T00:20:00Z",
            [_spread_book("book_a", 1.0)],
        )
    ]
    payload = json.dumps(payload_events, separators=(",", ":")).encode()
    observed_at = datetime(2024, 9, 10, 13, 0, tzinfo=UTC)
    quotes = parse_odds_api_response(payload, observed_at=observed_at)
    assert "capture_kind" not in quotes.columns  # live schema predates this column
    quotes = attach_nflverse_game_ids(quotes, two_game_schedule)
    write_market_snapshot(
        payload, quotes, root, observed_at=observed_at, request_metadata={"regions": "us"}
    )
    loaded = load_decision_quotes(root, capture_kind="live")
    assert not loaded.empty
    assert set(loaded["capture_kind"]) == {"live"}


# ---------------------------------------------------------------------------
# 2. CLV metric math (hand-computed, including a key-number crossing)
# ---------------------------------------------------------------------------


def test_key_number_distance_folds_by_magnitude() -> None:
    distances = key_number_distance(pd.Series([3.0, -3.0, 0.0, 5.0, 10.0, 2.5]))
    assert distances.tolist() == pytest.approx([0.0, 0.0, 3.0, 2.0, 3.0, 0.5])


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
            # G1's line crosses the key number 3 between open (2.5) and close (3.5).
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
    # G1: HOME at 2.5, close 3.5 -> line moved toward home favor after the bet: +1.0 CLV.
    assert scored.loc[scored["game_id"].eq("G1"), "clv_points"].iloc[0] == pytest.approx(1.0)
    # G2: AWAY at home_spread -1.0 (home a 1-point underdog), close -2.0 (home
    # a bigger, 2-point underdog): the market moved further away from home, which
    # is favorable to an AWAY bettor who is already locked in at the smaller
    # number -- by hand: clv = -1 * (close - decision) = -1 * (-2.0 - (-1.0)) = +1.0.
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


def test_clv_summary_and_week_blocked_bootstrap_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "season": [2024] * 6,
            "week": [1, 1, 2, 2, 3, 3],
            "clv_points": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    summary = clv_summary(frame)
    assert summary == {
        "n": 6.0,
        "mean_clv_points": 1.0,
        "median_clv_points": 1.0,
        "positive_clv_rate": 1.0,
    }
    bootstrap = week_blocked_bootstrap(frame, clv_summary, block="week", samples=50, seed=1)
    row = bootstrap.set_index("metric").loc["mean_clv_points"]
    # Every block has an identical value, so every resample gives the same estimate.
    assert row["estimate"] == pytest.approx(1.0)
    assert row["lower"] == pytest.approx(1.0)
    assert row["upper"] == pytest.approx(1.0)


def test_week_blocked_bootstrap_guards() -> None:
    frame = pd.DataFrame({"season": [2024], "week": [1], "clv_points": [1.0]})
    with pytest.raises(ValueError, match="samples must be"):
        week_blocked_bootstrap(frame, clv_summary, samples=1)
    with pytest.raises(ValueError, match="block must be"):
        week_blocked_bootstrap(frame, clv_summary, block="day")  # type: ignore[arg-type]
    with pytest.raises(DataContractError, match="missing columns"):
        week_blocked_bootstrap(frame.drop(columns=["week"]), clv_summary, block="week")


# ---------------------------------------------------------------------------
# 3. Frozen feature construction (MKT-06 pilot)
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
    # Two target games late enough to have >= 50 prior completed training rows
    # (fit_margin_model requires at least 50 completed games).
    for idx, (tue_open, close) in zip((55, 65), ((2.5, 3.5), (-1.0, -2.5)), strict=True):
        _store_tue_and_close_for_game(
            root, features, features.iloc[idx], tue_open=tue_open, close=close
        )
    config = {
        "feature_profile": "base",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "target": "market_residual",
    }
    return root, features, config


def test_build_pilot_frame_assembles_frozen_features(
    pilot_setup: tuple[Path, pd.DataFrame, dict[str, Any]],
) -> None:
    root, features, config = pilot_setup
    frame = build_pilot_frame(root, features, active_model_config=config, min_train_games=50)
    assert len(frame) == 2
    for column in (*FROZEN_PILOT_FEATURES, "target_close_minus_open", "game_id", "season"):
        assert column in frame.columns
    row = frame.set_index("game_id").loc["G055"]
    assert row["tue_open_home_spread"] == pytest.approx(2.5)
    assert row["target_close_minus_open"] == pytest.approx(3.5 - 2.5)
    assert row["tue_open_key_number_distance"] == pytest.approx(
        key_number_distance(pd.Series([2.5])).iloc[0]
    )
    assert np.isfinite(row["active_model_residual_at_opener"])


def test_active_model_residual_matches_close_when_opener_equals_close() -> None:
    features = _pilot_features_frame()
    target_row = features.iloc[55]
    targets = pd.DataFrame(
        {
            "game_id": [target_row["game_id"]],
            "season": [target_row["season"]],
            "week": [target_row["week"]],
            "tue_open_home_spread": [target_row["spread_line"]],
        }
    )
    residual = active_model_residual_at_opener(
        features, targets, feature_profile="base", ridge_alpha=10.0, min_train_games=50
    )
    assert len(residual) == 1
    assert np.isfinite(residual.iloc[0]["active_model_residual_at_opener"])


def test_active_model_residual_drops_weeks_without_enough_history() -> None:
    features = _pilot_features_frame()
    early_row = features.iloc[2]
    targets = pd.DataFrame(
        {
            "game_id": [early_row["game_id"]],
            "season": [early_row["season"]],
            "week": [early_row["week"]],
            "tue_open_home_spread": [early_row["spread_line"]],
        }
    )
    with pytest.raises(ValueError, match="No target week"):
        active_model_residual_at_opener(
            features, targets, feature_profile="base", ridge_alpha=10.0, min_train_games=500
        )


def test_fit_evaluate_and_threshold_policy() -> None:
    rng = np.random.default_rng(0)
    n = 60
    tue_open = rng.uniform(-7, 7, size=n)
    noise = rng.normal(0, 0.25, size=n)
    target = 0.4 * tue_open + noise  # movement correlated with opener level
    frame = pd.DataFrame(
        {
            "game_id": [f"P{i:03d}" for i in range(n)],
            "season": 2024,
            "week": (np.arange(n) % 17) + 1,
            "tue_open_home_spread": tue_open,
            "tue_open_key_number_distance": key_number_distance(pd.Series(tue_open)),
            "active_model_residual_at_opener": 0.4 * tue_open + rng.normal(0, 0.1, size=n),
            "rest_diff": rng.integers(-3, 4, size=n).astype(float),
            "target_close_minus_open": target,
        }
    )
    model = fit_pilot_model(frame)
    metrics = evaluate_pilot(model, frame)
    assert 0.0 <= metrics["direction_accuracy"] <= 1.0
    assert metrics["mae_model"] < metrics["mae_no_movement_baseline"]

    predicted = np.asarray(model.predict(frame.loc[:, list(FROZEN_PILOT_FEATURES)]))
    policy = threshold_policy_clv(frame, predicted, threshold=0.5)
    assert set(policy["side"]) <= {"HOME", "AWAY"}
    assert (policy["predicted_move"].abs() >= 0.5).all()


def test_fit_pilot_model_requires_frozen_columns() -> None:
    with pytest.raises(DataContractError, match="missing columns"):
        fit_pilot_model(pd.DataFrame({"tue_open_home_spread": [1.0]}))


def test_run_predeclared_pilot_reports_blocker_for_missing_seasons(
    pilot_setup: tuple[Path, pd.DataFrame, dict[str, Any]],
) -> None:
    root, features, config = pilot_setup
    with pytest.raises(PilotProtocolBlocked, match="2024") as excinfo:
        run_predeclared_pilot(root, features, active_model_config=config, min_train_games=50)
    assert str(FROZEN_PILOT_PROTOCOL.test_season) in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. The sign test (report Pilot B)
# ---------------------------------------------------------------------------


def test_sign_test_pilot_b_reports_binomial_intervals(
    pilot_setup: tuple[Path, pd.DataFrame, dict[str, Any]],
) -> None:
    root, features, config = pilot_setup
    result = sign_test_pilot_b(root, features, active_model_config=config, min_train_games=50)
    assert result["overall"]["games"] == 2
    assert 0.0 <= result["overall"]["accuracy"] <= 1.0
    assert result["overall"]["confidence_interval"] is not None
    assert set(result["per_season"]) <= {2020, 2021}


def test_sign_test_requires_data() -> None:
    features = _pilot_features_frame()
    with pytest.raises(ValueError, match="No paired games"):
        sign_test_pilot_b(Path("does-not-exist"), features)


# ---------------------------------------------------------------------------
# 4b. Production wiring: predicted close for an upcoming week (MKT-06)
# ---------------------------------------------------------------------------


def _live_features_frame() -> pd.DataFrame:
    """The pilot fixture frame plus one final, still-unplayed target game."""

    features = _pilot_features_frame(n_games=71)
    features.loc[features.index[-1], ["result", "ats_margin"]] = np.nan
    return features


def _store_live_tuesday_snapshot(
    root: Path, features: pd.DataFrame, game_row: pd.Series, *, home_spread: float
) -> None:
    """A live capture observed on the game's own-week Tuesday at 13:00 UTC."""

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


def test_upcoming_week_picks_earliest_unplayed() -> None:
    features = _live_features_frame()
    target_row = features.iloc[-1]
    assert upcoming_week(features) == (int(target_row["season"]), int(target_row["week"]))


def test_upcoming_week_requires_an_unplayed_game() -> None:
    with pytest.raises(ValueError, match="no unplayed games"):
        upcoming_week(_pilot_features_frame())


def test_live_tuesday_openers_empty_store(tmp_path: Path) -> None:
    assert live_tuesday_openers(tmp_path / "raw").empty


def test_predict_close_for_week_end_to_end(
    pilot_setup: tuple[Path, pd.DataFrame, dict[str, Any]],
) -> None:
    root, _, config = pilot_setup
    features = _live_features_frame()
    target_row = features.iloc[-1]
    _store_live_tuesday_snapshot(root, features, target_row, home_spread=3.0)

    result = predict_close_for_week(
        root,
        features,
        season=int(target_row["season"]),
        week=int(target_row["week"]),
        active_model_config=config,
        min_train_games=50,
    )
    predictions = result["predictions"]
    assert len(predictions) == 1
    row = predictions.iloc[0]
    assert row["game_id"] == target_row["game_id"]
    assert row["tue_open_home_spread"] == pytest.approx(3.0)
    assert row["opener_books"] == 2
    assert np.isfinite(row["predicted_close_minus_open"])
    assert row["predicted_close_home_spread"] == pytest.approx(
        3.0 + row["predicted_close_minus_open"]
    )
    assert result["train_games"] == 2
    assert result["train_start_season"] == FROZEN_PILOT_PROTOCOL.train_start_season


def test_predict_close_unavailable_without_live_opener(
    pilot_setup: tuple[Path, pd.DataFrame, dict[str, Any]],
) -> None:
    root, _, config = pilot_setup
    features = _live_features_frame()
    target_row = features.iloc[-1]
    with pytest.raises(ClosePredictionUnavailable, match="Tuesday"):
        predict_close_for_week(
            root,
            features,
            season=int(target_row["season"]),
            week=int(target_row["week"]),
            active_model_config=config,
            min_train_games=50,
        )


def test_predict_close_blocked_without_training_archive(tmp_path: Path) -> None:
    features = _live_features_frame()
    target_row = features.iloc[-1]
    # A live opener exists, so the (cheaper, checked-first) target step passes
    # and the missing historical training archive is what blocks.
    _store_live_tuesday_snapshot(tmp_path / "raw", features, target_row, home_spread=3.0)
    config = {
        "feature_profile": "base",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "target": "market_residual",
    }
    with pytest.raises(PilotProtocolBlocked, match="training data"):
        predict_close_for_week(
            tmp_path / "raw",
            features,
            season=int(target_row["season"]),
            week=int(target_row["week"]),
            active_model_config=config,
            min_train_games=50,
        )


def test_close_prediction_output_contract_rejects_bad_frames() -> None:
    good = pd.DataFrame(
        {
            "game_id": ["A", "B"],
            "tue_open_home_spread": [2.5, -3.0],
            "predicted_close_minus_open": [0.2, -0.1],
            "predicted_close_home_spread": [2.7, -3.1],
        }
    )
    _validate_close_predictions(good)

    duplicated = good.copy()
    duplicated.loc[1, "game_id"] = "A"
    with pytest.raises(DataContractError, match="duplicate"):
        _validate_close_predictions(duplicated)

    non_finite = good.copy()
    non_finite.loc[0, "predicted_close_home_spread"] = np.nan
    with pytest.raises(DataContractError, match="non-finite"):
        _validate_close_predictions(non_finite)

    implausible = good.copy()
    implausible.loc[0, "predicted_close_home_spread"] = 45.0
    with pytest.raises(DataContractError, match="plausible"):
        _validate_close_predictions(implausible)


# ---------------------------------------------------------------------------
# 5. Active-model configuration resolution
# ---------------------------------------------------------------------------


def test_resolve_active_model_config_falls_back_when_manifest_absent(tmp_path: Path) -> None:
    config = resolve_active_model_config(tmp_path)
    assert config == {
        "feature_profile": "player",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "target": "market_residual",
    }


def test_resolve_active_model_config_reads_manifest_when_present(tmp_path: Path) -> None:
    manifest = {
        "version": ACTIVE_ATS_MODEL_VERSION,
        "status": "SYNCHRONIZED",
        "method": "market_residual",
        "feature_profile": "base",
        "regressor": "ridge",
        "target": "ats_classification",
    }
    atomic_json(manifest, tmp_path / "active_ats_model.json")
    config = resolve_active_model_config(tmp_path)
    assert config["feature_profile"] == "base"
    assert config["ridge_alpha"] == 10.0
