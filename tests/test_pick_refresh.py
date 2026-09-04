from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
from nfl_ats.clv import PAPER_DECISION_COLUMNS, paper_decision_ledger_path
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_json, atomic_parquet
from nfl_ats.lines import apply_external_lines
from nfl_ats.market_data import QUOTE_COLUMNS
from nfl_ats.outcomes import fit_margin_models_for_week
from nfl_ats.pick_refresh import (
    LATE_WEEK_REFRESH_END,
    LATE_WEEK_REFRESH_START,
    MOVEMENT_POLICY_MODEL_ONLY,
    MOVEMENT_POLICY_MOVEMENT,
    MOVEMENT_POLICY_THRESHOLD,
    _movement_side,
    append_refresh_to_card,
    current_captured_home_spread,
    final_pick_per_game,
    load_pick_revisions,
    original_card,
    pick_revision_ledger_path,
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

# A Tue..Mon NFL week used across tests: Tuesday 2026-09-15 through Monday
# 2026-09-21, in UTC (September is EDT, UTC-4).
TNF_KICKOFF = pd.Timestamp("2026-09-18T00:15:00+00:00")  # Thu 8:15pm ET
SUN_EARLY_KICKOFF = pd.Timestamp("2026-09-20T17:00:00+00:00")  # Sun 1:00pm ET
SNF_KICKOFF = pd.Timestamp("2026-09-21T00:20:00+00:00")  # Sun 8:20pm ET
MNF_KICKOFF = pd.Timestamp("2026-09-22T00:15:00+00:00")  # Mon 8:15pm ET
SUNDAY_LOCK = pd.Timestamp("2026-09-20T20:00:00+00:00")  # Sun 4:00pm ET


def _target_frame(model_frame: pd.DataFrame, games: list[dict]) -> pd.DataFrame:
    """Append unplayed target-week rows onto ``model_frame``'s training history."""

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
    """One single-book HOME spread quote, matching a scheduled `odds-ingest`
    capture's `quotes.parquet` shape closely enough for
    `nfl_ats.market_data.load_quote_history` / `spread_consensus` to read it.
    A single book means the cross-book median equals `home_spread_line`
    exactly, keeping assertions exact."""

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


def _reference_probability(
    model_frame: pd.DataFrame,
    games: list[dict],
    original_lines: dict[str, float],
    *,
    season: int,
    week: int,
) -> dict[str, float]:
    """Independently reproduce the frozen-line prediction, bypassing pick_refresh
    entirely, so tests assert against a ground truth computed a different way."""

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
        "spread_line": 6.5,  # CURRENT feature table's line -- must never be used
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

# The FROZEN Tuesday lines -- deliberately different from GAMES' own
# "current" spread_line above, so any test that used the current line by
# mistake would compute a different (and therefore caught) probability.
ORIGINAL_LINES = {
    "2026_02_AAA_BBB": -1.5,
    "2026_02_CCC_DDD": -1.0,
    "2026_02_EEE_FFF": 0.5,
    "2026_02_GGG_HHH": 3.0,
}


def _original_rows(reference: dict[str, float], *, flip: bool = True) -> list[dict]:
    """One paper-decision row per game. ``flip=True`` sets pick_side OPPOSITE
    the reference probability's side, so a refresh is guaranteed to change
    it; ``flip=False`` matches it, guaranteeing a no-op."""

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


# ---------------------------------------------------------------------------
# 1. Lines frozen from the original card
# ---------------------------------------------------------------------------


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
        # The CURRENT feature table's own spread_line differs from the
        # frozen original on every game above -- confirms the reference
        # (independently computed at the frozen line) is not simply
        # reproducing whatever the current table happens to carry.
        current_line = next(g["spread_line"] for g in GAMES if g["game_id"] == game_id)
        assert current_line != original_line


def test_refresh_reuses_frozen_arrest_flags_and_never_reads_a_newer_snapshot(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=SEASON, week=WEEK)
    rows = _original_rows(reference, flip=False)
    first = rows[0]
    model_side = first["pick_side"]
    first["model_pick_side"] = model_side
    first["player_arrests_home_flag"] = model_side == "AWAY"
    first["player_arrests_away_flag"] = model_side == "HOME"
    first["pick_side"] = "AWAY" if model_side == "HOME" else "HOME"
    first["player_arrests_flip"] = True
    first["composed_overlay_flip"] = True
    _write_original_card(artifacts_root, rows)
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)

    # This intentionally unusable newer source would fail a fresh-source load.
    # Refresh must not consult it: Tuesday's ledger flags are the frozen input.
    newest = data_root / "raw" / "player_arrests" / "20260916T120000Z"
    newest.mkdir(parents=True)
    (newest / "manifest.json").write_text('{"complete": false}', encoding="utf-8")

    plan = plan_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=datetime(2026, 9, 16, tzinfo=UTC),
    )
    refreshed = {game.game_id: game for game in plan.games}[first["game_id"]]

    assert refreshed.player_arrests_flip is True
    assert refreshed.new_pick_side == first["pick_side"]
    assert refreshed.player_arrests_snapshot_id == "snapshot-tuesday"
    assert refreshed.player_arrests_safe_index_sha256 == "safe-index-hash"


# ---------------------------------------------------------------------------
# 2. Kickoff guard: a started game is never revised
# ---------------------------------------------------------------------------


def test_kickoff_guard_never_revises_a_started_game(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=SEASON, week=WEEK)
    _write_original_card(artifacts_root, _original_rows(reference, flip=True))
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)

    # Friday noon UTC: after TNF's own kickoff, well before the Sunday lock.
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
    # Every OTHER game (kickoff still ahead, Sunday lock still ahead) did change.
    assert set(result["changed_game_ids"]) == {
        "2026_02_CCC_DDD",
        "2026_02_EEE_FFF",
        "2026_02_GGG_HHH",
    }

    revisions = load_pick_revisions(artifacts_root)
    assert "2026_02_AAA_BBB" not in set(revisions["game_id"])


# ---------------------------------------------------------------------------
# 3. Sunday 4:00 PM ET pick lock: SNF/MNF lock early
# ---------------------------------------------------------------------------


def test_sunday_pick_lock_blocks_monday_night_game_before_its_own_kickoff(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=SEASON, week=WEEK)
    _write_original_card(artifacts_root, _original_rows(reference, flip=True))
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)

    # Sunday 5:00pm ET (21:00 UTC): one hour after the 4:00pm ET lock, over a
    # full day before the MNF game's own kickoff (Monday 8:15pm ET).
    after_lock = datetime(2026, 9, 20, 21, 0, tzinfo=UTC)
    plan_after = plan_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=after_lock,
    )
    mnf_after = next(g for g in plan_after.games if g.game_id == "2026_02_GGG_HHH")
    assert mnf_after.eligible is False
    assert mnf_after.ineligible_reason == "sunday_pick_lock_passed"
    assert mnf_after.changed is False
    # SNF is bound by the same week-wide cap, even though its own kickoff was
    # also still ahead at this instant.
    snf_after = next(g for g in plan_after.games if g.game_id == "2026_02_EEE_FFF")
    assert snf_after.eligible is False
    assert snf_after.ineligible_reason == "sunday_pick_lock_passed"

    # One hour BEFORE the lock, the same MNF game is still eligible.
    before_lock = datetime(2026, 9, 20, 19, 0, tzinfo=UTC)
    plan_before = plan_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=before_lock,
    )
    mnf_before = next(g for g in plan_before.games if g.game_id == "2026_02_GGG_HHH")
    assert mnf_before.eligible is True
    assert mnf_before.ineligible_reason == ""


def test_sunday_pick_lock_is_four_pm_eastern_on_the_weeks_anchor_sunday() -> None:
    kickoffs = pd.Series([TNF_KICKOFF, SUN_EARLY_KICKOFF, SNF_KICKOFF, MNF_KICKOFF])
    assert sunday_pick_lock(kickoffs) == SUNDAY_LOCK


# ---------------------------------------------------------------------------
# 4. Append-only revisions, opt-in recording, no-op refresh
# ---------------------------------------------------------------------------


def test_record_refresh_is_opt_in_and_writes_nothing_by_default(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=SEASON, week=WEEK)
    _write_original_card(artifacts_root, _original_rows(reference, flip=True))
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)

    result = record_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=datetime(2026, 9, 16, tzinfo=UTC),
        record_decisions=False,
    )
    assert result["ledger"]["recorded"] == 0
    assert result["ledger"]["skipped"] is True
    assert len(result["changed_game_ids"]) > 0  # there WAS something to record
    assert load_pick_revisions(artifacts_root).empty
    assert not pick_revision_ledger_path(artifacts_root).is_file()


def test_record_refresh_appends_only_new_rows_and_a_second_identical_pass_is_a_no_op(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=SEASON, week=WEEK)
    _write_original_card(artifacts_root, _original_rows(reference, flip=True))
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)
    now = datetime(2026, 9, 16, tzinfo=UTC)

    first = record_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=now,
        note="thursday_afternoon",
        record_decisions=True,
    )
    changed = set(first["changed_game_ids"])
    assert changed  # every game was flipped relative to the (opposite) original
    assert first["ledger"]["recorded"] == len(changed)
    revisions = load_pick_revisions(artifacts_root)
    assert set(revisions["game_id"]) == changed
    assert revisions["reason"].eq("pick_refresh recompute (thursday_afternoon)").all()
    for game_id in changed:
        row = revisions.loc[revisions["game_id"].eq(game_id)].iloc[0]
        assert row["decision_home_spread"] == pytest.approx(ORIGINAL_LINES[game_id])
        assert row["new_home_cover_probability"] == pytest.approx(reference[game_id])

    # The original Tuesday ledger is untouched -- append-only, never rewritten.
    original = original_card(artifacts_root, season=SEASON, week=WEEK)
    assert len(original) == len(GAMES)

    # A second pass with IDENTICAL inputs finds nothing new to change (the
    # chain-latest pick now already matches the model's own read).
    second = record_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=now,
        record_decisions=True,
    )
    assert second["changed_game_ids"] == []
    assert second["ledger"]["recorded"] == 0
    assert second["ledger"]["ledger_rows"] == len(revisions)
    assert len(load_pick_revisions(artifacts_root)) == len(revisions)


def test_record_refresh_no_change_writes_zero_rows_and_exits_clean(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=SEASON, week=WEEK)
    # flip=False: the original card ALREADY matches what the model would say.
    _write_original_card(artifacts_root, _original_rows(reference, flip=False))
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)

    result = record_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=datetime(2026, 9, 16, tzinfo=UTC),
        record_decisions=True,
    )
    assert result["changed_game_ids"] == []
    assert result["ledger"] == {"recorded": 0, "ledger_rows": 0}
    assert not pick_revision_ledger_path(artifacts_root).is_file()


# ---------------------------------------------------------------------------
# 5. Fail-closed: no original card, model-identity drift
# ---------------------------------------------------------------------------


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


def test_plan_refresh_fails_closed_on_a_game_missing_its_original_line(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=SEASON, week=WEEK)
    # Only record 3 of the 4 games -- the 4th has no frozen line.
    rows = _original_rows(reference, flip=True)
    _write_original_card(artifacts_root, rows[:3])
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
    assert plan.unrefreshable_game_ids == ("2026_02_GGG_HHH",)
    assert "2026_02_GGG_HHH" not in {game.game_id for game in plan.games}
    assert len(plan.games) == 3


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


def test_record_plan_refuses_a_rehearsal_recording_weeks_before_kickoff(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=SEASON, week=WEEK)
    _write_original_card(artifacts_root, _original_rows(reference, flip=True))
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)

    # Three weeks before the week's earliest kickoff -- the exact shape of
    # the 2026-08-18 incident this guard exists to prevent from recurring.
    far_before = datetime(2026, 8, 25, tzinfo=UTC)
    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_refresh(
            artifacts_root,
            data_root,
            season=SEASON,
            week=WEEK,
            features_path=features_path,
            min_train_games=MIN_TRAIN_GAMES,
            now=far_before,
            record_decisions=True,
        )
    assert load_pick_revisions(artifacts_root).empty


# ---------------------------------------------------------------------------
# 6. Ledger contract and final-pick chain resolution
# ---------------------------------------------------------------------------


def test_load_pick_revisions_rejects_a_ledger_missing_columns(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    atomic_parquet(pd.DataFrame({"game_id": ["g1"]}), pick_revision_ledger_path(artifacts_root))
    with pytest.raises(DataContractError, match="missing columns"):
        load_pick_revisions(artifacts_root)


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
    assert changed_ids  # the fixture guarantees at least one flip

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
        # Tuesday's own recorded pick is never rewritten by a later revision.
        assert original_by_id.loc[game_id, "pick_side"] == original_by_id.loc[game_id, "pick_side"]


# ---------------------------------------------------------------------------
# 7. CURRENT_PREDICTIONS.md append (additive, idempotent, never touches Tuesday)
# ---------------------------------------------------------------------------


def test_append_refresh_to_card_is_additive_and_idempotent(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=SEASON, week=WEEK)
    _write_original_card(artifacts_root, _original_rows(reference, flip=True))
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)

    destination = data_root.parent / "CURRENT_PREDICTIONS.md"
    tuesday_text = "# NFL ATS predictions: 2026 Week 2\n\nTuesday content here.\n"
    destination.write_text(tuesday_text, encoding="utf-8")

    plan = plan_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=datetime(2026, 9, 16, tzinfo=UTC),
    )
    append_refresh_to_card(destination, plan, note="thursday_afternoon")
    first_text = destination.read_text(encoding="utf-8")
    assert tuesday_text.strip() in first_text
    assert first_text.count(LATE_WEEK_REFRESH_START) == 1
    assert first_text.count(LATE_WEEK_REFRESH_END) == 1
    assert "Late-week refresh" in first_text
    assert "thursday_afternoon" in first_text

    # Re-running (e.g. the Saturday pass) replaces the section instead of
    # duplicating it, and never disturbs the Tuesday content above it.
    later_plan = plan_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=datetime(2026, 9, 19, 10, tzinfo=UTC),
    )
    append_refresh_to_card(destination, later_plan, note="saturday_pass")
    second_text = destination.read_text(encoding="utf-8")
    assert tuesday_text.strip() in second_text
    assert second_text.count(LATE_WEEK_REFRESH_START) == 1
    assert second_text.count(LATE_WEEK_REFRESH_END) == 1
    assert "saturday_pass" in second_text
    assert "thursday_afternoon" not in second_text


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


# ---------------------------------------------------------------------------
# 8. Observed-movement pick policy (POL-11 addendum, 2026-08-20)
# ---------------------------------------------------------------------------

# A dedicated single-game fixture (kept separate from GAMES/ORIGINAL_LINES
# above, which the earlier sections' helpers -- _original_rows in particular
# -- iterate as a fixed 4-game list) so these tests can control the movement
# delta precisely without disturbing any other game's picks.
MOVEMENT_GAME = {
    "game_id": "2026_02_MOV_TST",
    "season": SEASON,
    "week": WEEK,
    "gameday": pd.Timestamp("2026-09-17"),
    "away_team": "MOV",
    "home_team": "TST",
    "spread_line": 6.5,  # current feature table's line -- must never be used
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


def test_movement_side_sign_convention_matches_the_measurement_script() -> None:
    """`_movement_side` reuses `scripts/observed_movement_channel.py`'s
    `_threshold_pick` sign logic verbatim: positive delta (home spread rose,
    market moved toward home) picks HOME; negative, or an exact tie, picks
    AWAY (the tie case is never actually selected by `plan_refresh` -- it is
    only reached below the 1.0 threshold, where the model pick governs)."""

    assert _movement_side(2.0) == "HOME"
    assert _movement_side(0.5) == "HOME"
    assert _movement_side(-0.5) == "AWAY"
    assert _movement_side(-2.0) == "AWAY"
    assert _movement_side(0.0) == "AWAY"


@pytest.mark.parametrize("delta_sign", [1, -1], ids=["toward_home", "toward_away"])
def test_movement_policy_overrides_the_pick_when_the_market_moves_at_least_one_point(
    refresh_env: tuple[Path, Path, pd.DataFrame], delta_sign: int
) -> None:
    """A >=1.0 point move in EITHER direction overrides the played pick to
    the side the market moved toward, regardless of what the model's own
    recompute says -- both signs, as required, in one fixture."""

    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(
        model_frame,
        [MOVEMENT_GAME],
        {MOVEMENT_GAME["game_id"]: MOVEMENT_ORIGINAL_LINE},
        season=SEASON,
        week=WEEK,
    )
    model_only_side = "HOME" if reference[MOVEMENT_GAME["game_id"]] >= 0.5 else "AWAY"
    _write_movement_original_card(artifacts_root, pick_side=model_only_side)
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, [MOVEMENT_GAME]), features_path)

    now = datetime(2026, 9, 16, tzinfo=UTC)
    delta = delta_sign * 2.0  # well clear of the 1.0 threshold
    current_line = MOVEMENT_ORIGINAL_LINE + delta
    _write_live_quote(
        data_root,
        snapshot_id="live-1",
        game_id=MOVEMENT_GAME["game_id"],
        home_spread_line=current_line,
        observed_at=pd.Timestamp(now),
        commence_time=MOVEMENT_GAME["kickoff"],
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
    assert len(plan.games) == 1
    game = plan.games[0]
    expected_side = "HOME" if delta_sign > 0 else "AWAY"
    assert game.movement_policy == MOVEMENT_POLICY_MOVEMENT
    assert game.movement_delta == pytest.approx(delta)
    assert game.movement_pick_side == expected_side
    assert game.new_pick_side == expected_side
    assert game.model_only_pick_side == model_only_side
    assert plan.current_line_metadata["fresh"] is True


def test_movement_policy_overrides_a_disagreeing_model_pick(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    """A genuine override, not a coincidence: the movement side is
    deliberately forced OPPOSITE the model's own recompute, and the market
    still wins the played pick."""

    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(
        model_frame,
        [MOVEMENT_GAME],
        {MOVEMENT_GAME["game_id"]: MOVEMENT_ORIGINAL_LINE},
        season=SEASON,
        week=WEEK,
    )
    model_only_side = "HOME" if reference[MOVEMENT_GAME["game_id"]] >= 0.5 else "AWAY"
    _write_movement_original_card(artifacts_root, pick_side=model_only_side)
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, [MOVEMENT_GAME]), features_path)

    now = datetime(2026, 9, 16, tzinfo=UTC)
    # Move the market AWAY from whatever the model itself picked.
    delta = -2.0 if model_only_side == "HOME" else 2.0
    current_line = MOVEMENT_ORIGINAL_LINE + delta
    _write_live_quote(
        data_root,
        snapshot_id="live-1",
        game_id=MOVEMENT_GAME["game_id"],
        home_spread_line=current_line,
        observed_at=pd.Timestamp(now),
        commence_time=MOVEMENT_GAME["kickoff"],
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
    game = plan.games[0]
    assert game.model_only_pick_side == model_only_side
    assert game.new_pick_side != model_only_side
    assert game.movement_policy == MOVEMENT_POLICY_MOVEMENT
    assert game.new_pick_side == game.movement_pick_side


def test_movement_policy_keeps_the_model_pick_below_threshold(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(
        model_frame,
        [MOVEMENT_GAME],
        {MOVEMENT_GAME["game_id"]: MOVEMENT_ORIGINAL_LINE},
        season=SEASON,
        week=WEEK,
    )
    model_only_side = "HOME" if reference[MOVEMENT_GAME["game_id"]] >= 0.5 else "AWAY"
    _write_movement_original_card(artifacts_root, pick_side=model_only_side)
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, [MOVEMENT_GAME]), features_path)

    now = datetime(2026, 9, 16, tzinfo=UTC)
    delta = 0.4  # inside the 1.0 threshold
    assert abs(delta) < MOVEMENT_POLICY_THRESHOLD
    current_line = MOVEMENT_ORIGINAL_LINE + delta
    _write_live_quote(
        data_root,
        snapshot_id="live-1",
        game_id=MOVEMENT_GAME["game_id"],
        home_spread_line=current_line,
        observed_at=pd.Timestamp(now),
        commence_time=MOVEMENT_GAME["kickoff"],
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
    game = plan.games[0]
    assert game.movement_policy == MOVEMENT_POLICY_MODEL_ONLY
    assert game.new_pick_side == model_only_side
    assert game.movement_delta == pytest.approx(delta)
    # The candidate side is still computed for transparency, just not applied.
    assert game.movement_pick_side == "HOME"


def test_movement_policy_is_a_no_op_with_no_market_snapshots(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    """Fail-open: no `data/market/raw` store at all -- the model-pick refresh
    still proceeds, unaffected, exactly like before this feature existed."""

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
    assert plan.current_line_metadata["fresh"] is False
    assert plan.current_line_metadata["reason"] == "no_market_snapshots"
    for game in plan.games:
        assert game.movement_policy == MOVEMENT_POLICY_MODEL_ONLY
        assert game.movement_delta is None
        assert game.movement_pick_side == ""
    # Every game still changed relative to the (deliberately flipped)
    # Tuesday pick, exactly as the pre-existing (non-movement) test above
    # already pins -- the movement policy did not disturb that behavior.
    assert len(plan.changed_games) == len(GAMES)


def test_movement_policy_is_a_no_op_when_the_latest_capture_is_stale(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(
        model_frame,
        [MOVEMENT_GAME],
        {MOVEMENT_GAME["game_id"]: MOVEMENT_ORIGINAL_LINE},
        season=SEASON,
        week=WEEK,
    )
    model_only_side = "HOME" if reference[MOVEMENT_GAME["game_id"]] >= 0.5 else "AWAY"
    _write_movement_original_card(artifacts_root, pick_side=model_only_side)
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, [MOVEMENT_GAME]), features_path)

    now = datetime(2026, 9, 16, tzinfo=UTC)
    stale_observed_at = pd.Timestamp(now) - pd.Timedelta(days=3)
    # A stale quote that WOULD clear the threshold if it were fresh -- proves
    # the no-op is driven by staleness, not by the delta being too small.
    _write_live_quote(
        data_root,
        snapshot_id="live-stale",
        game_id=MOVEMENT_GAME["game_id"],
        home_spread_line=MOVEMENT_ORIGINAL_LINE + 3.0,
        observed_at=stale_observed_at,
        commence_time=MOVEMENT_GAME["kickoff"],
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
    assert plan.current_line_metadata["fresh"] is False
    assert plan.current_line_metadata["reason"] == "latest_capture_not_from_today"
    game = plan.games[0]
    assert game.movement_policy == MOVEMENT_POLICY_MODEL_ONLY
    assert game.movement_delta is None
    assert game.new_pick_side == model_only_side


def test_movement_policy_is_a_no_op_per_game_when_that_games_line_is_not_captured(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    """The overall capture can be fresh (today) while a SPECIFIC game's line
    was never matched/captured this pass -- that one game still fails open,
    even though `current_line_metadata["fresh"]` is True for the run."""

    artifacts_root, data_root, model_frame = refresh_env
    games = [
        MOVEMENT_GAME,
        {
            "game_id": "2026_02_UNC_APT",
            "season": SEASON,
            "week": WEEK,
            "gameday": pd.Timestamp("2026-09-17"),
            "away_team": "UNC",
            "home_team": "APT",
            "spread_line": 6.5,
            "kickoff": TNF_KICKOFF,
        },
    ]
    original_lines = {MOVEMENT_GAME["game_id"]: MOVEMENT_ORIGINAL_LINE, "2026_02_UNC_APT": 2.0}
    reference = _reference_probability(model_frame, games, original_lines, season=SEASON, week=WEEK)
    rows = []
    for game in games:
        side = "HOME" if reference[game["game_id"]] >= 0.5 else "AWAY"
        rows.append(
            {
                "recorded_at_utc": pd.Timestamp("2026-09-15T14:00:00+00:00"),
                "model_id": "model-1",
                "game_id": game["game_id"],
                "season": SEASON,
                "week": WEEK,
                "kickoff": game["kickoff"],
                "away_team": game["away_team"],
                "home_team": game["home_team"],
                "pick_side": side,
                "bet_side": side,
                "decision_home_spread": original_lines[game["game_id"]],
                "edge": 0.05,
            }
        )
    _write_original_card(artifacts_root, rows)
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, games), features_path)

    now = datetime(2026, 9, 16, tzinfo=UTC)
    # Only MOVEMENT_GAME gets a captured quote; "2026_02_UNC_APT" gets none.
    _write_live_quote(
        data_root,
        snapshot_id="live-1",
        game_id=MOVEMENT_GAME["game_id"],
        home_spread_line=MOVEMENT_ORIGINAL_LINE + 2.0,
        observed_at=pd.Timestamp(now),
        commence_time=MOVEMENT_GAME["kickoff"],
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
    by_id = {game.game_id: game for game in plan.games}
    assert by_id[MOVEMENT_GAME["game_id"]].movement_policy == MOVEMENT_POLICY_MOVEMENT
    uncaptured = by_id["2026_02_UNC_APT"]
    assert uncaptured.movement_policy == MOVEMENT_POLICY_MODEL_ONLY
    assert uncaptured.movement_delta is None


def test_ledger_records_movement_policy_delta_and_both_candidate_picks(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(
        model_frame,
        [MOVEMENT_GAME],
        {MOVEMENT_GAME["game_id"]: MOVEMENT_ORIGINAL_LINE},
        season=SEASON,
        week=WEEK,
    )
    model_only_side = "HOME" if reference[MOVEMENT_GAME["game_id"]] >= 0.5 else "AWAY"
    _write_movement_original_card(artifacts_root, pick_side=model_only_side)
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, [MOVEMENT_GAME]), features_path)

    now = datetime(2026, 9, 16, tzinfo=UTC)
    delta = -2.0 if model_only_side == "HOME" else 2.0
    _write_live_quote(
        data_root,
        snapshot_id="live-1",
        game_id=MOVEMENT_GAME["game_id"],
        home_spread_line=MOVEMENT_ORIGINAL_LINE + delta,
        observed_at=pd.Timestamp(now),
        commence_time=MOVEMENT_GAME["kickoff"],
    )

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
    assert result["ledger"]["recorded"] == 1
    assert result["movement_policy"]["current_line_fresh"] is True
    assert result["movement_policy"]["games_movement_applied"] == [MOVEMENT_GAME["game_id"]]

    revisions = load_pick_revisions(artifacts_root)
    row = revisions.loc[revisions["game_id"].eq(MOVEMENT_GAME["game_id"])].iloc[0]
    assert row["movement_policy"] == MOVEMENT_POLICY_MOVEMENT
    assert row["movement_delta"] == pytest.approx(delta)
    assert row["model_only_pick_side"] == model_only_side
    assert row["new_pick_side"] != model_only_side
    assert row["movement_pick_side"] == row["new_pick_side"]


def test_movement_policy_never_bypasses_the_kickoff_deadline_guard(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    """A threshold-clearing captured line for a game whose OWN kickoff has
    already passed must not revive it -- the deadline guard runs
    independently of, and after, the movement-policy computation.

    Needs its OWN game (not `MOVEMENT_GAME`, whose TNF kickoff is a full
    calendar day before any `now` that could follow it): a captured quote
    must be BOTH pregame (before that game's own kickoff, the market
    store's own pregame filter) AND from the same America/New_York calendar
    date as `now` (the movement policy's freshness gate) to exist at all --
    which requires a game whose kickoff and the test's "now" fall on the
    same day. A Sunday-early game with "now" later that same Sunday
    (after kickoff, so ineligible) is the realistic shape: e.g. a Sunday
    9am ET capture, still valid for a Sunday-afternoon "now"."""

    deadline_game = {
        "game_id": "2026_02_SUN_TST",
        "season": SEASON,
        "week": WEEK,
        "gameday": pd.Timestamp("2026-09-20"),
        "away_team": "SUN",
        "home_team": "TST",
        "spread_line": 6.5,
        "kickoff": SUN_EARLY_KICKOFF,  # Sun 1:00pm ET
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

    # Sunday 3:00pm ET: after this game's own 1:00pm ET kickoff (and before
    # the week-wide 4:00pm ET lock, so this specifically exercises the
    # KICKOFF guard, not the Sunday-lock guard already covered elsewhere).
    now = datetime(2026, 9, 20, 19, 0, tzinfo=UTC)
    # A pregame capture from earlier the SAME Sunday (9:00am ET) -- fresh by
    # the movement policy's same-day rule, and strictly before this game's
    # own kickoff, so the market store's own pregame filter keeps it.
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
    # The movement policy still COMPUTES a movement-side override...
    assert game.movement_policy == MOVEMENT_POLICY_MOVEMENT
    # ...but eligibility and `changed` are governed solely by the deadline.
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


def test_recorded_revisions_carry_trigger_provenance(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, data_root, model_frame = refresh_env
    reference = _reference_probability(model_frame, GAMES, ORIGINAL_LINES, season=SEASON, week=WEEK)
    _write_original_card(artifacts_root, _original_rows(reference, flip=True))
    features_path = data_root / "processed" / "game_features.parquet"
    atomic_parquet(_target_frame(model_frame, GAMES), features_path)

    result = record_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=datetime(2026, 9, 16, tzinfo=UTC),
        record_decisions=True,
        trigger_type="clock_dispatch",
        trigger_source="refresh_thu",
    )
    assert result["ledger"]["recorded"] == len(GAMES)
    revisions = load_pick_revisions(artifacts_root)
    assert (revisions["trigger_type"] == "clock_dispatch").all()
    assert (revisions["trigger_source"] == "refresh_thu").all()
    assert revisions["trigger_observed_at_utc"].notna().all()


def test_legacy_revision_rows_read_back_with_unknown_trigger(
    refresh_env: tuple[Path, Path, pd.DataFrame],
) -> None:
    artifacts_root, _, _ = refresh_env
    atomic_parquet(_legacy_trigger_row(), pick_revision_ledger_path(artifacts_root))
    revisions = load_pick_revisions(artifacts_root)
    assert (revisions["trigger_type"] == "unknown").all()
    assert revisions["trigger_observed_at_utc"].isna().all()


def test_current_captured_home_spread_reads_the_local_store_only(tmp_path: Path) -> None:
    """Direct unit coverage of the read-only adapter, independent of
    `plan_refresh`: fresh data returns a populated mapping, and an empty
    store fails open with a named reason -- never raises."""

    data_root = tmp_path / "data"
    now = pd.Timestamp("2026-09-16T00:00:00Z")

    empty_lines, empty_meta = current_captured_home_spread(data_root, now=now)
    assert empty_lines == {}
    assert empty_meta["fresh"] is False
    assert empty_meta["reason"] == "no_market_snapshots"

    _write_live_quote(
        data_root,
        snapshot_id="live-1",
        game_id="2026_02_MOV_TST",
        home_spread_line=1.5,
        observed_at=now,
        commence_time=MOVEMENT_GAME["kickoff"],
    )
    lines, meta = current_captured_home_spread(data_root, now=now)
    assert lines == {"2026_02_MOV_TST": 1.5}
    assert meta["fresh"] is True
    assert meta["games_with_current_line"] == 1


# ---------------------------------------------------------------------------
# UI-17 refresh-diff sentences.
# ---------------------------------------------------------------------------


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


def test_describe_week_revisions_reports_changed_and_confirmed() -> None:
    from nfl_ats.pick_refresh import describe_week_revisions

    frame = _revision_frame(
        [
            _revision_row(),
            _revision_row(
                game_id="2026_01_DEN_KC",
                home_team="KC",
                away_team="DEN",
                previous_pick_side="KC",
                new_pick_side="KC",
                movement_delta=0.0,
                refresh_run_id="refresh_sun",
            ),
        ]
    )
    lines = describe_week_revisions(frame, _REFRESH_GAMES, season=2026, week=1)
    assert len(lines) == 2
    assert lines[0] == (
        "MIA at LV refresh (refresh_sat): pick now MIA (Tuesday card: LV); "
        "frozen Tuesday line (home +3.5); line moved +1.5 points."
    )
    assert "refresh confirmed KC, no change from Tuesday" in lines[1]


def test_describe_week_revisions_latest_wins_and_scope_filters() -> None:
    from nfl_ats.pick_refresh import describe_week_revisions

    frame = _revision_frame(
        [
            _revision_row(
                revision_recorded_at_utc="2026-09-12T10:00:00+00:00",
                new_pick_side="LV",
                previous_pick_side="LV",
            ),
            _revision_row(),  # later stamp supersedes the earlier row
            _revision_row(game_id="2026_01_NO_DET", week=2),  # wrong week
            _revision_row(game_id="2026_01_ARI_LAC"),  # game not on this card
        ]
    )
    lines = describe_week_revisions(frame, _REFRESH_GAMES, season=2026, week=1)
    assert len(lines) == 1
    assert "pick now MIA (Tuesday card: LV)" in lines[0]


def test_describe_week_revisions_empty_without_rows() -> None:
    from nfl_ats.pick_refresh import describe_week_revisions

    assert describe_week_revisions(pd.DataFrame(), _REFRESH_GAMES, season=2026, week=1) == ()
    assert (
        describe_week_revisions(
            _revision_frame([_revision_row()]), _REFRESH_GAMES, season=None, week=1
        )
        == ()
    )
