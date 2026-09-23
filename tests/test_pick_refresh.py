from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
from nfl_ats.clv import PAPER_DECISION_COLUMNS, paper_decision_ledger_path
from nfl_ats.io import atomic_json, atomic_parquet
from nfl_ats.lines import apply_external_lines
from nfl_ats.market_data import QUOTE_COLUMNS
from nfl_ats.outcomes import fit_margin_models_for_week
from nfl_ats.pick_refresh import (
    LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY,
    append_refresh_to_card,
    final_pick_per_game,
    load_pick_revisions,
    original_card,
    plan_refresh,
    record_refresh,
    sunday_pick_lock,
)

MIN_TRAIN_GAMES = 50
_NON_FEATURE_COLUMNS = {
    "game_id",
    "season",
    "week",
    "gameday",
    "away_team",
    "home_team",
    "home_spread_odds",
    "away_spread_odds",
    "spread_line",
    "home_cover",
    "ats_margin",
    "result",
}

TNF_KICKOFF = pd.Timestamp("2026-09-18T00:15:00+00:00")
SUN_EARLY_KICKOFF = pd.Timestamp("2026-09-20T17:00:00+00:00")
SNF_KICKOFF = pd.Timestamp("2026-09-21T00:20:00+00:00")
MNF_KICKOFF = pd.Timestamp("2026-09-22T00:15:00+00:00")
SUNDAY_LOCK = pd.Timestamp("2026-09-20T20:00:00+00:00")


def _target_frame(model_frame: pd.DataFrame, games: list[dict]) -> pd.DataFrame:

    feature_columns = [c for c in model_frame.columns if c not in _NON_FEATURE_COLUMNS]
    template = model_frame.iloc[0]
    rows = []
    for game in games:
        row = {column: template[column] for column in feature_columns}
        row.update(
            {
                "home_spread_odds": -110.0,
                "away_spread_odds": -110.0,
                "home_cover": np.nan,
                "ats_margin": np.nan,
                "result": np.nan,
            }
        )
        row.update(game)
        rows.append(row)
    return pd.concat([model_frame, pd.DataFrame(rows)], ignore_index=True, sort=False)


def _write_active_manifest(
    artifacts_root: Path,
    *,
    model_id: str = "model-1",
    feature_profile: str = "base",
    method: str = "market_residual",
    regressor: str = "ridge",
    ridge_alpha: float = 10.0,
    probability_method: str = "ecdf",
) -> None:
    atomic_json(
        {
            "version": ACTIVE_ATS_MODEL_VERSION,
            "status": "SYNCHRONIZED",
            "method": method,
            "feature_profile": feature_profile,
            "regressor": regressor,
            "ridge_alpha": ridge_alpha,
            "probability_method": probability_method,
            "model_id": model_id,
        },
        artifacts_root / "active_ats_model.json",
    )


def _write_original_card(artifacts_root: Path, rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    defaults = {
        "forecast_artifact": "margin_predictions/test",
        "forecast_created_at_utc": pd.Timestamp("2026-09-15T13:00:00+00:00"),
        "method": "market_residual",
        "decision_policy_id": ("overlay_union_coach_division_revenge_player_arrests_spread_gap_v1"),
        "decision_policy_fingerprint": "test-policy-fingerprint",
        "coach_fade_flip": False,
        "division_revenge_flip": False,
        "player_arrests_flip": False,
        "spread_gap_zone_flip": False,
        "composed_overlay_flip": False,
        "player_arrests_home_flag": False,
        "player_arrests_away_flag": False,
        "player_arrests_snapshot_id": "snapshot-tuesday",
        "player_arrests_snapshot_fetched_at_utc": pd.Timestamp("2026-09-15T12:00:00+00:00"),
        "player_arrests_safe_index_sha256": "safe-index-hash",
        "schedule_snapshot_id": "schedule-tuesday",
        "schedule_parquet_sha256": "schedule-hash",
        "is_best_pick": False,
    }
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value
    if "model_pick_side" not in frame:
        frame["model_pick_side"] = frame["pick_side"]
    else:
        frame["model_pick_side"] = frame["model_pick_side"].fillna(frame["pick_side"])
    if "pre_arrest_pick_side" not in frame:
        frame["pre_arrest_pick_side"] = frame["pick_side"]
    if "former_policy_pick_side" not in frame:
        frame["former_policy_pick_side"] = frame["pick_side"]
    frame["recorded_at_utc"] = pd.to_datetime(frame["recorded_at_utc"], utc=True)
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True)
    frame["forecast_created_at_utc"] = pd.to_datetime(frame["forecast_created_at_utc"], utc=True)
    atomic_parquet(frame[list(PAPER_DECISION_COLUMNS)], paper_decision_ledger_path(artifacts_root))
    return frame


def _write_live_quote(
    data_root: Path,
    *,
    snapshot_id: str,
    game_id: str,
    home_spread_line: float,
    observed_at: pd.Timestamp,
    commence_time: pd.Timestamp,
    bookmaker_key: str = "draftkings",
) -> None:

    row = {
        "observed_at_utc": observed_at,
        "provider": "the-odds-api",
        "provider_event_id": f"evt-{game_id}",
        "sport_key": "americanfootball_nfl",
        "commence_time_utc": commence_time,
        "home_team_name": "Home Team",
        "away_team_name": "Away Team",
        "home_team": "HME",
        "away_team": "AWY",
        "nflverse_game_id": game_id,
        "bookmaker_key": bookmaker_key,
        "bookmaker_title": bookmaker_key,
        "bookmaker_last_update_utc": observed_at,
        "market": "spreads",
        "market_last_update_utc": observed_at,
        "outcome_name": "home",
        "outcome_side": "HOME",
        "line": home_spread_line,
        "price": -110.0,
        "home_spread_line": home_spread_line,
        "raw_response_sha256": "deadbeef",
    }
    quotes = pd.DataFrame([row], columns=list(QUOTE_COLUMNS))
    directory = data_root / "market" / "raw" / snapshot_id
    atomic_parquet(quotes, directory / "quotes.parquet")
    atomic_json(
        {"provider": "the-odds-api", "publication_scope": "derived_allowed"},
        directory / "manifest.json",
    )


def _reference_probability(
    model_frame: pd.DataFrame,
    games: list[dict],
    original_lines: dict[str, float],
    *,
    season: int,
    week: int,
) -> dict[str, float]:

    features = _target_frame(model_frame, games)
    target, margin_models = fit_margin_models_for_week(
        features,
        season=season,
        week=week,
        regressor="ridge",
        min_train_games=MIN_TRAIN_GAMES,
        feature_profile="base",
        ridge_alpha=10.0,
        methods=("market_residual",),
    )
    lines = pd.DataFrame(
        {"game_id": list(original_lines), "home_spread": list(original_lines.values())}
    )
    overridden = apply_external_lines(target, lines)
    forecasts = margin_models["market_residual"].predict(overridden, probability_method="ecdf")
    return dict(
        zip(overridden["game_id"].astype(str), forecasts["home_cover_probability"], strict=True)
    )


@pytest.fixture
def refresh_env(tmp_path: Path, model_frame: pd.DataFrame) -> tuple[Path, Path, pd.DataFrame]:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_active_manifest(artifacts_root)
    return artifacts_root, data_root, model_frame


SEASON, WEEK = 2026, 2

GAMES = [
    {
        "game_id": "2026_02_AAA_BBB",
        "season": SEASON,
        "week": WEEK,
        "gameday": pd.Timestamp("2026-09-17"),
        "away_team": "AAA",
        "home_team": "BBB",
        "spread_line": 6.5,
        "kickoff": TNF_KICKOFF,
    },
    {
        "game_id": "2026_02_CCC_DDD",
        "season": SEASON,
        "week": WEEK,
        "gameday": pd.Timestamp("2026-09-20"),
        "away_team": "CCC",
        "home_team": "DDD",
        "spread_line": 4.0,
        "kickoff": SUN_EARLY_KICKOFF,
    },
    {
        "game_id": "2026_02_EEE_FFF",
        "season": SEASON,
        "week": WEEK,
        "gameday": pd.Timestamp("2026-09-20"),
        "away_team": "EEE",
        "home_team": "FFF",
        "spread_line": 2.5,
        "kickoff": SNF_KICKOFF,
    },
    {
        "game_id": "2026_02_GGG_HHH",
        "season": SEASON,
        "week": WEEK,
        "gameday": pd.Timestamp("2026-09-21"),
        "away_team": "GGG",
        "home_team": "HHH",
        "spread_line": -3.5,
        "kickoff": MNF_KICKOFF,
    },
]

ORIGINAL_LINES = {
    "2026_02_AAA_BBB": -1.5,
    "2026_02_CCC_DDD": -1.0,
    "2026_02_EEE_FFF": 0.5,
    "2026_02_GGG_HHH": 3.0,
}


def _original_rows(reference: dict[str, float], *, flip: bool = True) -> list[dict]:

    rows = []
    for game in GAMES:
        game_id = game["game_id"]
        true_side = "HOME" if reference[game_id] >= 0.5 else "AWAY"
        pick_side = ("AWAY" if true_side == "HOME" else "HOME") if flip else true_side
        rows.append(
            {
                "recorded_at_utc": pd.Timestamp("2026-09-15T14:00:00+00:00"),
                "model_id": "model-1",
                "game_id": game_id,
                "season": SEASON,
                "week": WEEK,
                "kickoff": game["kickoff"],
                "away_team": game["away_team"],
                "home_team": game["home_team"],
                "pick_side": pick_side,
                "bet_side": pick_side,
                "decision_home_spread": ORIGINAL_LINES[game_id],
                "edge": 0.05,
            }
        )
    return rows


def test_refreshed_probability_uses_the_original_frozen_line_not_current_features(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=SEASON, week=WEEK)
    _write_original_card(artifacts_root, _original_rows(reference, flip=True))
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)

    plan = plan_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=datetime(2026, 9, 16, tzinfo=UTC),
    )

    by_id = {game.game_id: game for game in plan.games}
    assert len(plan.games) == len(GAMES)
    for game_id, original_line in ORIGINAL_LINES.items():
        game = by_id[game_id]
        assert game.decision_home_spread == pytest.approx(original_line)
        assert game.new_home_cover_probability == pytest.approx(reference[game_id])
        current_line = next(g["spread_line"] for g in GAMES if g["game_id"] == game_id)
        assert current_line != original_line


def test_kickoff_guard_never_revises_a_started_game(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=SEASON, week=WEEK)
    _write_original_card(artifacts_root, _original_rows(reference, flip=True))
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)

    now = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    result = record_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=now,
        record_decisions=True,
    )

    assert "2026_02_AAA_BBB" in result["post_kickoff_skipped"]
    assert "2026_02_AAA_BBB" not in result["changed_game_ids"]
    assert set(result["changed_game_ids"]) == {
        "2026_02_CCC_DDD",
        "2026_02_EEE_FFF",
        "2026_02_GGG_HHH",
    }

    revisions = load_pick_revisions(artifacts_root)
    assert "2026_02_AAA_BBB" not in set(revisions["game_id"])


def test_sunday_pick_lock_is_four_pm_eastern_on_the_weeks_anchor_sunday() -> None:
    kickoffs = pd.Series([TNF_KICKOFF, SUN_EARLY_KICKOFF, SNF_KICKOFF, MNF_KICKOFF])
    assert sunday_pick_lock(kickoffs) == SUNDAY_LOCK


def test_plan_refresh_fails_closed_with_no_recorded_original_card(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)

    with pytest.raises(ValueError, match="record-decisions"):
        plan_refresh(
            artifacts_root,
            data_root,
            season=SEASON,
            week=WEEK,
            features_path=features_path,
            min_train_games=MIN_TRAIN_GAMES,
        )


def test_plan_refresh_rejects_a_model_identity_that_has_since_changed(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=SEASON, week=WEEK)
    rows = _original_rows(reference, flip=True)
    for row in rows:
        row["model_id"] = "a-different-model"
    _write_original_card(artifacts_root, rows)
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)

    with pytest.raises(ValueError, match="different model identity"):
        plan_refresh(
            artifacts_root,
            data_root,
            season=SEASON,
            week=WEEK,
            features_path=features_path,
            min_train_games=MIN_TRAIN_GAMES,
        )


def test_final_pick_per_game_reflects_the_latest_revision_while_original_stays_fixed(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=SEASON, week=WEEK)
    _write_original_card(artifacts_root, _original_rows(reference, flip=True))
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)

    record_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=datetime(2026, 9, 16, tzinfo=UTC),
        record_decisions=True,
    )

    original = original_card(artifacts_root, season=SEASON, week=WEEK)
    final = final_pick_per_game(artifacts_root, season=SEASON, week=WEEK)
    changed_ids = set(load_pick_revisions(artifacts_root)["game_id"])
    assert changed_ids

    final_by_id = final.set_index("game_id")
    original_by_id = original.set_index("game_id")
    for game_id in changed_ids:
        assert final_by_id.loc[game_id, "revised"] == np.True_
        assert (
            final_by_id.loc[game_id, "final_pick_side"] != original_by_id.loc[game_id, "pick_side"]
        )
        assert (
            final_by_id.loc[game_id, "tuesday_pick_side"]
            == original_by_id.loc[game_id, "pick_side"]
        )
        assert original_by_id.loc[game_id, "pick_side"] == original_by_id.loc[game_id, "pick_side"]


def test_append_refresh_to_card_fails_closed_without_a_published_card(tmp_path: Path) -> None:
    from nfl_ats.pick_refresh import RefreshResult

    empty_plan = RefreshResult(
        season=SEASON,
        week=WEEK,
        refresh_run_id="20260916T000000Z",
        computed_at_utc=pd.Timestamp("2026-09-16T00:00:00Z"),
        model_id="model-1",
        feature_table_path="unused",
        feature_table_sha256="unused",
        games=(),
        unrefreshable_game_ids=(),
        missing_from_features_game_ids=(),
    )
    with pytest.raises(ValueError, match="publish-predictions"):
        append_refresh_to_card(tmp_path / "CURRENT_PREDICTIONS.md", empty_plan)


MOVEMENT_GAME = {
    "game_id": "2026_02_MOV_TST",
    "season": SEASON,
    "week": WEEK,
    "gameday": pd.Timestamp("2026-09-17"),
    "away_team": "MOV",
    "home_team": "TST",
    "spread_line": 6.5,
    "kickoff": TNF_KICKOFF,
}
MOVEMENT_ORIGINAL_LINE = -1.5


def _write_movement_original_card(artifacts_root: Path, *, pick_side: str) -> None:
    _write_original_card(
        artifacts_root,
        [
            {
                "recorded_at_utc": pd.Timestamp("2026-09-15T14:00:00+00:00"),
                "model_id": "model-1",
                "game_id": MOVEMENT_GAME["game_id"],
                "season": SEASON,
                "week": WEEK,
                "kickoff": MOVEMENT_GAME["kickoff"],
                "away_team": MOVEMENT_GAME["away_team"],
                "home_team": MOVEMENT_GAME["home_team"],
                "pick_side": pick_side,
                "bet_side": pick_side,
                "decision_home_spread": MOVEMENT_ORIGINAL_LINE,
                "edge": 0.05,
            }
        ],
    )


def test_movement_policy_never_bypasses_the_kickoff_deadline_guard(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:

    deadline_game = {
        "game_id": "2026_02_SUN_TST",
        "season": SEASON,
        "week": WEEK,
        "gameday": pd.Timestamp("2026-09-20"),
        "away_team": "SUN",
        "home_team": "TST",
        "spread_line": 6.5,
        "kickoff": SUN_EARLY_KICKOFF,
    }
    original_line = -1.5

    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(
        model_frame,
        [deadline_game],
        {deadline_game["game_id"]: original_line},
        season=SEASON,
        week=WEEK,
    )
    model_only_side = "HOME" if reference[deadline_game["game_id"]] >= 0.5 else "AWAY"
    _write_original_card(
        artifacts_root,
        [
            {
                "recorded_at_utc": pd.Timestamp("2026-09-15T14:00:00+00:00"),
                "model_id": "model-1",
                "game_id": deadline_game["game_id"],
                "season": SEASON,
                "week": WEEK,
                "kickoff": deadline_game["kickoff"],
                "away_team": deadline_game["away_team"],
                "home_team": deadline_game["home_team"],
                "pick_side": model_only_side,
                "bet_side": model_only_side,
                "decision_home_spread": original_line,
                "edge": 0.05,
            }
        ],
    )
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, [deadline_game]), features_path)

    now = datetime(2026, 9, 20, 19, 0, tzinfo=UTC)
    observed_at = pd.Timestamp("2026-09-20T13:00:00+00:00")
    _write_live_quote(
        data_root,
        snapshot_id="live-1",
        game_id=deadline_game["game_id"],
        home_spread_line=original_line + 3.0,
        observed_at=observed_at,
        commence_time=deadline_game["kickoff"],
    )

    plan = plan_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=now,
    )
    assert plan.current_line_metadata["fresh"] is True
    game = plan.games[0]
    assert game.consensus_delta == pytest.approx(3.0)
    assert game.eligible is False
    assert game.ineligible_reason == "kickoff_passed"
    assert game.changed is False

    result = record_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=now,
        record_decisions=True,
    )
    assert deadline_game["game_id"] in result["post_kickoff_skipped"]
    assert deadline_game["game_id"] not in result["changed_game_ids"]
    assert load_pick_revisions(artifacts_root).empty


def _legacy_trigger_row() -> pd.DataFrame:
    from nfl_ats.pick_refresh import PICK_REVISION_COLUMNS

    row = dict.fromkeys(PICK_REVISION_COLUMNS)
    row.update(
        {
            "revision_recorded_at_utc": pd.Timestamp("2026-09-15T14:00:00+00:00"),
            "game_id": "2026_02_AAA_BBB",
            "season": SEASON,
            "week": WEEK,
        }
    )
    legacy = pd.DataFrame([row])
    return legacy.drop(columns=["trigger_type", "trigger_source", "trigger_observed_at_utc"])


def _revision_frame(rows: list[dict]) -> pd.DataFrame:
    from nfl_ats.pick_refresh import PICK_REVISION_COLUMNS

    base: dict = dict.fromkeys(PICK_REVISION_COLUMNS)
    full = [{**base, **row} for row in rows]
    return pd.DataFrame(full, columns=list(PICK_REVISION_COLUMNS))


def _revision_row(**overrides) -> dict:
    row: dict = {
        "revision_recorded_at_utc": "2026-09-12T14:00:00+00:00",
        "refresh_run_id": "refresh_sat",
        "season": 2026,
        "week": 1,
        "game_id": "2026_01_MIA_LV",
        "home_team": "LV",
        "away_team": "MIA",
        "decision_home_spread": 3.5,
        "previous_pick_side": "LV",
        "new_pick_side": "MIA",
        "movement_delta": 1.5,
        "trigger_type": "clock_dispatch",
    }
    row.update(overrides)
    return row


_REFRESH_GAMES = (("2026_01_MIA_LV", "MIA", "LV"), ("2026_01_DEN_KC", "DEN", "KC"))


LATE_WEEK_SEASON, LATE_WEEK_WEEK = 2025, 2
LATE_WEEK_NOW = datetime(2025, 9, 20, 15, tzinfo=UTC)
LATE_WEEK_ANCHOR_AT = pd.Timestamp("2025-09-16T18:00:00+00:00")
LATE_WEEK_MOVE_AT = pd.Timestamp("2025-09-19T18:00:00+00:00")
LATE_WEEK_KICKOFF = pd.Timestamp("2025-09-21T17:00:00+00:00")
LATE_WEEK_BOOKS = ("bovada", "fanduel")
LATE_WEEK_GAME = {
    "game_id": "2025_02_LWW_MMV",
    "season": LATE_WEEK_SEASON,
    "week": LATE_WEEK_WEEK,
    "gameday": pd.Timestamp("2025-09-21"),
    "away_team": "LWW",
    "home_team": "MMV",
    "spread_line": 6.5,
    "kickoff": LATE_WEEK_KICKOFF,
}
LATE_WEEK_ORIGINAL_LINE = -1.5


def _write_late_week_original_card(artifacts_root: Path, *, pick_side: str) -> None:
    _write_original_card(
        artifacts_root,
        [
            {
                "recorded_at_utc": pd.Timestamp("2025-09-16T16:00:00+00:00"),
                "model_id": "model-1",
                "game_id": LATE_WEEK_GAME["game_id"],
                "season": LATE_WEEK_SEASON,
                "week": LATE_WEEK_WEEK,
                "kickoff": LATE_WEEK_GAME["kickoff"],
                "away_team": LATE_WEEK_GAME["away_team"],
                "home_team": LATE_WEEK_GAME["home_team"],
                "pick_side": pick_side,
                "bet_side": pick_side,
                "decision_home_spread": LATE_WEEK_ORIGINAL_LINE,
                "edge": 0.05,
            }
        ],
    )


def _write_live_intraday_archive(
    data_root: Path,
    *,
    game_id: str,
    kickoff: pd.Timestamp,
    anchor_line: float,
    move: float,
    books: tuple[str, ...] = LATE_WEEK_BOOKS,
) -> None:

    for index, (at, line) in enumerate(
        ((LATE_WEEK_ANCHOR_AT, anchor_line), (LATE_WEEK_MOVE_AT, anchor_line + move))
    ):
        rows = [
            {
                "observed_at_utc": at,
                "provider": "the-odds-api",
                "provider_event_id": f"evt-{game_id}",
                "sport_key": "americanfootball_nfl",
                "commence_time_utc": kickoff,
                "home_team_name": "Home Team",
                "away_team_name": "Away Team",
                "home_team": "HME",
                "away_team": "AWY",
                "nflverse_game_id": game_id,
                "bookmaker_key": book,
                "bookmaker_title": book,
                "bookmaker_last_update_utc": at,
                "market": "spreads",
                "market_last_update_utc": at,
                "outcome_name": "home",
                "outcome_side": "HOME",
                "line": line,
                "price": -110.0,
                "home_spread_line": line,
                "raw_response_sha256": "deadbeef",
            }
            for book in books
        ]
        directory = data_root / "market" / "raw" / f"intraday-{index}"
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=list(QUOTE_COLUMNS)).to_parquet(
            directory / "quotes.parquet", index=False
        )
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "capture_kind": "live",
                    "observed_at_utc": pd.Timestamp(at).isoformat(),
                    "request": {"season": LATE_WEEK_SEASON, "week": LATE_WEEK_WEEK},
                }
            ),
            encoding="utf-8",
        )


def _write_late_week_consensus_quote(data_root: Path, *, line: float) -> None:

    _write_live_quote(
        data_root,
        snapshot_id="live-1",
        game_id=LATE_WEEK_GAME["game_id"],
        home_spread_line=line,
        observed_at=pd.Timestamp(LATE_WEEK_NOW),
        commence_time=LATE_WEEK_GAME["kickoff"],
        bookmaker_key="draftkings",
    )


def _late_week_setup(refresh_env):

    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(
        model_frame,
        [LATE_WEEK_GAME],
        {LATE_WEEK_GAME["game_id"]: LATE_WEEK_ORIGINAL_LINE},
        season=LATE_WEEK_SEASON,
        week=LATE_WEEK_WEEK,
    )
    model_only_side = "HOME" if reference[LATE_WEEK_GAME["game_id"]] >= 0.5 else "AWAY"
    _write_late_week_original_card(artifacts_root, pick_side=model_only_side)
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, [LATE_WEEK_GAME]), features_path)
    return artifacts_root, data_root, features_path, model_only_side


def _late_week_plan(artifacts_root, data_root, features_path):
    plan = plan_refresh(
        artifacts_root,
        data_root,
        season=LATE_WEEK_SEASON,
        week=LATE_WEEK_WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=LATE_WEEK_NOW,
    )
    assert len(plan.games) == 1
    return plan


def test_late_week_follow_governs_the_served_pick(
    refresh_env: tuple[Path, Path, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    import nfl_ats.pick_refresh as pick_refresh

    monkeypatch.setattr(pick_refresh, "LATE_WEEK_FOLLOW_SERVED", True)
    artifacts_root, data_root, features_path, model_only_side = _late_week_setup(refresh_env)
    move = -1.25 if model_only_side == "HOME" else 1.25
    expected_side = "AWAY" if model_only_side == "HOME" else "HOME"
    _write_live_intraday_archive(
        data_root,
        game_id=LATE_WEEK_GAME["game_id"],
        kickoff=LATE_WEEK_GAME["kickoff"],
        anchor_line=LATE_WEEK_ORIGINAL_LINE,
        move=move,
    )
    _write_late_week_consensus_quote(data_root, line=LATE_WEEK_ORIGINAL_LINE + move)
    plan = _late_week_plan(artifacts_root, data_root, features_path)
    game = plan.games[0]
    assert game.model_only_pick_side == model_only_side
    assert game.movement_policy == LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY
    assert game.new_pick_side == expected_side != model_only_side
    assert game.movement_delta == pytest.approx(move)
    assert game.movement_pick_side == expected_side
    assert game.late_week_net_move == pytest.approx(move)
    assert game.late_week_pick_side == expected_side
    assert game.late_week_eligible_books == 1
    assert game.consensus_delta == pytest.approx(move)
    assert game.consensus_pick_side == expected_side
    assert plan.late_week_metadata["available"] is True
    assert plan.late_week_metadata["games_with_exposure"] == 1
    assert plan.late_week_metadata["games_followed"] == 1
