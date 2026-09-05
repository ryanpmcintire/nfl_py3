"""ENG-23: a real market/injury observation instant on the card, not a snapshot
fallback -- and the leakage regression that instant must never violate.

Covers, per ``docs/feature_lineage.md`` gap items 2 and 3:

* ``nfl_ats.market_observation.attach_market_observed_at`` joining the
  point-in-time odds capture's ``observed_at_utc`` onto a forecast frame by
  ``game_id`` -- synthetic snapshots under ``tmp_path``, and the null-safe
  path for games (most of history) with no matching capture.
* ``nfl_ats.players.enrich_with_player_features``'s ``injury_snapshot_captured_at``
  fallback populating ``{side}_injury_observed_at`` for a team with no
  visible injury revision, guarded so it can never fire after that game's own
  decision cutoff.
* ``nfl_ats.lineage.build_card_lineage`` preferring those frame-level columns
  over the whole-table manifest fallback for the ``market_line`` and
  ``model_input:player_injuries`` records, with legacy frames (no such
  columns) unaffected.
* The leakage invariant itself: a synthetic row whose observed-at is AFTER
  the prediction timestamp fails the existing ``market_timing``
  (``prediction_safety``) and ``lineage_effective_timestamp`` (``lineage``)
  checks -- these are not new checks, only new columns proven to trip them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nfl_ats.backtest import score_week
from nfl_ats.lineage import (
    BASE_SNAPSHOT_UNRECORDED,
    FIELD_MARKET_LINE,
    LINEAGE_CHECKS,
    LineageError,
    build_card_lineage,
    validate_card_lineage,
)
from nfl_ats.market_data import QUOTE_COLUMNS, load_quote_history, write_market_snapshot
from nfl_ats.market_observation import MARKET_OBSERVED_AT_COLUMN, attach_market_observed_at
from nfl_ats.pbp import PBP_SNAPSHOT_COLUMNS
from nfl_ats.players import enrich_with_player_features
from nfl_ats.prediction_safety import PredictionSafetyError, validate_prediction_card

# ---------------------------------------------------------------------------
# attach_market_observed_at
# ---------------------------------------------------------------------------

_TUESDAY_OPENER = pd.Timestamp("2026-09-01T09:05:00Z")  # a real Tuesday


def _quote_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "observed_at_utc": _TUESDAY_OPENER,
        "provider": "the-odds-api",
        "provider_event_id": "evt-1",
        "sport_key": "americanfootball_nfl",
        "commence_time_utc": pd.Timestamp("2026-09-06T17:00:00Z"),
        "home_team_name": "Kansas City Chiefs",
        "away_team_name": "Baltimore Ravens",
        "home_team": "KC",
        "away_team": "BAL",
        "nflverse_game_id": "2026_01_BAL_KC",
        "bookmaker_key": "draftkings",
        "bookmaker_title": "DraftKings",
        "bookmaker_last_update_utc": _TUESDAY_OPENER,
        "market": "spreads",
        "market_last_update_utc": _TUESDAY_OPENER,
        "outcome_name": "Kansas City Chiefs",
        "outcome_side": "HOME",
        "line": -3.0,
        "price": -110,
        "home_spread_line": -3.0,
        "raw_response_sha256": "deadbeef",
    }
    row.update(overrides)
    return row


def _write_quote_snapshot(root: Path) -> None:
    quotes = pd.DataFrame([_quote_row()], columns=QUOTE_COLUMNS)
    write_market_snapshot(
        payload=b"{}",
        quotes=quotes,
        root=root,
        observed_at=_TUESDAY_OPENER,
        request_metadata={"sport": "americanfootball_nfl"},
    )


def test_attach_market_observed_at_joins_the_tuesday_opener_by_game_id(tmp_path: Path) -> None:
    _write_quote_snapshot(tmp_path)
    frame = pd.DataFrame(
        {
            "game_id": ["2026_01_BAL_KC", "2013_01_AAA_BBB"],
            "spread_line": [-3.0, 1.5],
        }
    )

    result = attach_market_observed_at(frame, market_raw_root=tmp_path)

    assert result.loc[0, MARKET_OBSERVED_AT_COLUMN] == _TUESDAY_OPENER
    # A historical row with no matching capture (most of the archive predates
    # the-odds-api ingestion) is left null, not an error.
    assert pd.isna(result.loc[1, MARKET_OBSERVED_AT_COLUMN])
    # spread_line -- or any other column -- is never read or modified.
    assert result["spread_line"].tolist() == [-3.0, 1.5]


def test_attach_market_observed_at_reuses_an_already_loaded_quote_history(tmp_path: Path) -> None:
    _write_quote_snapshot(tmp_path)
    history = load_quote_history(tmp_path)
    frame = pd.DataFrame({"game_id": ["2026_01_BAL_KC"]})

    result = attach_market_observed_at(frame, quote_history=history)

    assert result.loc[0, MARKET_OBSERVED_AT_COLUMN] == _TUESDAY_OPENER


@pytest.mark.parametrize(
    "build_frame",
    [
        lambda: pd.DataFrame({"game_id": ["2013_01_AAA_BBB"]}),
        lambda: pd.DataFrame({"not_game_id": ["2013_01_AAA_BBB"]}),
    ],
)
def test_attach_market_observed_at_is_null_safe_with_no_capture_available(
    build_frame: Any, tmp_path: Path
) -> None:
    frame = build_frame()

    no_root = attach_market_observed_at(frame)
    missing_dir = attach_market_observed_at(frame, market_raw_root=tmp_path / "does_not_exist")
    empty_history = attach_market_observed_at(
        frame, quote_history=pd.DataFrame(columns=QUOTE_COLUMNS)
    )

    for result in (no_root, missing_dir, empty_history):
        assert MARKET_OBSERVED_AT_COLUMN in result.columns
        assert result[MARKET_OBSERVED_AT_COLUMN].isna().all()


# ---------------------------------------------------------------------------
# enrich_with_player_features: injury_snapshot_captured_at fallback
# ---------------------------------------------------------------------------


def _games() -> pd.DataFrame:
    dates = pd.date_range("2022-09-11", periods=4, freq="7D")
    return pd.DataFrame(
        {
            "game_id": [f"2022_{week:02d}_B_A" for week in range(1, 5)],
            "season": 2022,
            "week": range(1, 5),
            "gameday": dates,
            "kickoff": pd.date_range("2022-09-11 17:00Z", periods=4, freq="7D"),
            "away_team": "B",
            "home_team": "A",
        }
    )


def _rosters() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for week in range(1, 5):
        for team, player_id, pfr_id, position, experience in (
            ("A", "QB-A", "PFR-A", "QB", 5),
            ("A", "WR-A", "PFR-WR-A", "WR", 4),
            ("B", "QB-B", "PFR-B", "QB", 3),
            ("B", "WR-B", "PFR-WR-B", "WR", 2),
        ):
            rows.append(
                {
                    "season": 2022,
                    "team": team,
                    "position": position,
                    "status": "ACT",
                    "full_name": f"{position} {team}",
                    "gsis_id": player_id,
                    "pfr_id": pfr_id,
                    "years_exp": experience,
                    "week": week,
                    "game_type": "REG",
                }
            )
    return pd.DataFrame(rows)


def _injuries() -> pd.DataFrame:
    # Team A only -- team B never files a report, exercising the "no
    # revision was ever visible" branch every week, on both sides in turn.
    return pd.DataFrame(
        {
            "season": [2022, 2022],
            "game_type": ["REG", "REG"],
            "team": ["A", "A"],
            "week": [2, 2],
            "gsis_id": ["QB-A", "QB-A"],
            "position": ["QB", "QB"],
            "report_status": ["Questionable", "Out"],
            "practice_status": ["Limited Participation in Practice", "Did Not Participate"],
            "date_modified": ["2022-09-16T12:00:00Z", "2022-09-18T16:30:00Z"],
        }
    )


def _snaps() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for week in range(1, 5):
        for team, opponent, pfr_id, position, offense_pct in (
            ("A", "B", "PFR-A", "QB", 1.0),
            ("A", "B", "PFR-WR-A", "WR", 0.8),
            ("B", "A", "PFR-B", "QB", 1.0),
            ("B", "A", "PFR-WR-B", "WR", 0.8),
        ):
            rows.append(
                {
                    "game_id": f"2022_{week:02d}_B_A",
                    "season": 2022,
                    "game_type": "REG",
                    "week": week,
                    "player": f"{position} {team}",
                    "pfr_player_id": pfr_id,
                    "position": position,
                    "team": team,
                    "opponent": opponent,
                    "offense_snaps": 60 * offense_pct,
                    "offense_pct": offense_pct,
                    "defense_snaps": 0,
                    "defense_pct": 0.0,
                    "st_snaps": 0,
                    "st_pct": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _pbp() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for week in range(1, 5):
        for team, opponent, quarterback, direction in (
            ("A", "B", "QB-A", 1.0),
            ("B", "A", "QB-B", -1.0),
        ):
            for play_id in range(1, 7):
                row = dict.fromkeys(PBP_SNAPSHOT_COLUMNS, np.nan)
                row.update(
                    {
                        "play_id": play_id + (100 if team == "B" else 0),
                        "game_id": f"2022_{week:02d}_B_A",
                        "season": 2022,
                        "season_type": "REG",
                        "week": week,
                        "home_team": "A",
                        "away_team": "B",
                        "posteam": team,
                        "defteam": opponent,
                        "fixed_drive": 1 if team == "A" else 2,
                        "down": 1,
                        "play_type": "pass",
                        "yards_gained": 8,
                        "pass_attempt": 1,
                        "rush_attempt": 0,
                        "qb_dropback": 1,
                        "qb_kneel": 0,
                        "qb_spike": 0,
                        "aborted_play": 0,
                        "sack": 0,
                        "qb_hit": 0,
                        "interception": 0,
                        "epa": direction * week + play_id / 100,
                        "success": int(direction > 0),
                        "wp": 0.5,
                        "passer_player_id": quarterback,
                        "passer_player_name": f"QB {team}",
                        "cpoe": direction * 2,
                        "pass_oe": 0.1,
                        "yardline_100": 60,
                        "play": 1,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_injury_observed_at_falls_back_to_the_snapshot_capture_instant() -> None:
    # Before every game's own decision cutoff (kickoff - 24h, earliest is
    # 2022-09-10T17:00Z), so the leakage guard never blocks it here.
    captured_at = "2022-09-09T12:00:00Z"

    enriched = enrich_with_player_features(
        _games(),
        _injuries(),
        _rosters(),
        _snaps(),
        _pbp(),
        qb_min_dropbacks=1,
        injury_snapshot_captured_at=captured_at,
    )

    # Team B (away) never files a report in any week: every row falls back
    # to the snapshot capture instant instead of staying null.
    assert (enriched["away_injury_observed_at"] == pd.Timestamp(captured_at)).all()
    # Team A (home), week 1: also no revision on record yet -- same fallback.
    assert enriched.loc[0, "home_injury_observed_at"] == pd.Timestamp(captured_at)
    # Team A (home), week 2: a real revision IS visible -- unaffected by the
    # fallback, byte-identical to the pre-ENG-23 behaviour.
    assert enriched.loc[1, "home_injury_observed_at"] == pd.Timestamp("2022-09-16T12:00:00Z")


def test_injury_observed_at_fallback_omitted_reproduces_the_previous_null_behaviour() -> None:
    enriched = enrich_with_player_features(
        _games(), _injuries(), _rosters(), _snaps(), _pbp(), qb_min_dropbacks=1
    )

    assert enriched["away_injury_observed_at"].isna().all()
    assert pd.isna(enriched.loc[0, "home_injury_observed_at"])


def test_injury_observed_at_fallback_never_fires_after_its_own_decision_cutoff() -> None:
    # After every game's kickoff in this fixture (latest is 2022-10-02T17:00Z)
    # -- the fallback must never claim an as-of it cannot prove.
    future_capture = "2022-10-05T00:00:00Z"

    enriched = enrich_with_player_features(
        _games(),
        _injuries(),
        _rosters(),
        _snaps(),
        _pbp(),
        qb_min_dropbacks=1,
        injury_snapshot_captured_at=future_capture,
    )

    assert enriched["away_injury_observed_at"].isna().all()
    assert pd.isna(enriched.loc[0, "home_injury_observed_at"])
    # The one row with a REAL visible revision is untouched by any of this.
    assert enriched.loc[1, "home_injury_observed_at"] == pd.Timestamp("2022-09-16T12:00:00Z")


# ---------------------------------------------------------------------------
# lineage: market_line / model_input:player_injuries prefer the frame columns
# ---------------------------------------------------------------------------

PREDICTION_TIMESTAMP = "2026-09-03T14:32:53+00:00"
FEATURE_BUILD = "2026-09-03T14:31:38+00:00"
PLAYER_SNAPSHOT = "20260817T184901Z"
PLAYER_SNAPSHOT_CAPTURED = "2026-08-17T18:49:01+00:00"


def _forecast(**observed_at_columns: list[Any]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "game_id": ["2026_01_AAA_BBB", "2026_01_CCC_DDD"],
            "season": [2026, 2026],
            "week": [1, 1],
            "gameday": ["2026-09-13", "2026-09-13"],
            "home_team": ["BBB", "DDD"],
            "away_team": ["AAA", "CCC"],
            "spread_line": [-3.5, 2.5],
            "home_cover_probability": [0.55, 0.44],
            "method": ["market_residual", "market_residual"],
            "train_max_gameday": ["2026-01-04", "2026-01-04"],
        }
    )
    for column, values in observed_at_columns.items():
        frame[column] = values
    return frame


def _metadata() -> dict[str, Any]:
    return {
        "created_at_utc": PREDICTION_TIMESTAMP,
        "season": 2026,
        "week": 1,
        "feature_profile": "weak_stack",
        "active_model_id": "123d60be8c80a35d",
        "provenance": {
            "feature_table": {
                "path": "data/processed/game_features_weak_stack.parquet",
                "sha256": "a" * 64,
                "manifest": {
                    "built_at_utc": FEATURE_BUILD,
                    "source_player_snapshot": PLAYER_SNAPSHOT,
                    "player_feature_version": "v3-availability-v1",
                },
            }
        },
    }


_FEATURE_COLUMNS = ("spread_line", "elo_diff", "diff_injury_offense_unavailability")
_DISPLAY_FIELDS = {"Matchup": "formatted from team columns"}


def test_lineage_prefers_frame_level_observed_at_over_the_manifest_fallback() -> None:
    market_captured = "2026-09-02T09:00:00+00:00"
    injury_captured = "2026-08-30T10:00:00+00:00"
    forecast = _forecast(
        market_observed_at_utc=[market_captured, market_captured],
        home_injury_observed_at=[injury_captured, pd.NaT],
        away_injury_observed_at=[pd.NaT, injury_captured],
    )

    lineage = build_card_lineage(
        forecast,
        _metadata(),
        feature_columns=_FEATURE_COLUMNS,
        display_fields=_DISPLAY_FIELDS,
    )

    market = lineage.field(FIELD_MARKET_LINE)
    assert market is not None and market.lineage is not None
    assert market.lineage.source_captured_at == market_captured
    assert market.lineage.effective_timestamp == market_captured
    assert market.lineage.effective_timestamp_basis == "source_capture"
    # source_snapshot (WHICH snapshot) is untouched -- the metadata's
    # manifest names none, so it is still the digest fallback.
    assert market.lineage.source_snapshot == f"feature_table:sha256:{'a' * 64}"
    assert market.lineage.unknown_source_reason == BASE_SNAPSHOT_UNRECORDED

    injuries = lineage.field("model_input:player_injuries")
    assert injuries is not None and injuries.lineage is not None
    assert injuries.lineage.source_captured_at == injury_captured
    assert injuries.lineage.effective_timestamp == injury_captured
    assert injuries.lineage.effective_timestamp_basis == "source_capture"
    # source_snapshot is still the real manifest snapshot -- only
    # captured_at/effective_timestamp moved to the tighter frame value.
    assert injuries.lineage.source_snapshot == PLAYER_SNAPSHOT

    assert validate_card_lineage(lineage) == LINEAGE_CHECKS


def test_lineage_legacy_frame_without_the_new_columns_is_unaffected() -> None:
    lineage = build_card_lineage(
        _forecast(),
        _metadata(),
        feature_columns=_FEATURE_COLUMNS,
        display_fields=_DISPLAY_FIELDS,
    )

    market = lineage.field(FIELD_MARKET_LINE)
    assert market is not None and market.lineage is not None
    assert market.lineage.effective_timestamp_basis == "feature_table_build"

    injuries = lineage.field("model_input:player_injuries")
    assert injuries is not None and injuries.lineage is not None
    assert injuries.lineage.source_captured_at == PLAYER_SNAPSHOT_CAPTURED
    assert injuries.lineage.effective_timestamp_basis == "source_capture"


# ---------------------------------------------------------------------------
# Leakage regression: an observed-at after the prediction timestamp fails
# closed. Not a new check -- proof that the new columns actually trip the
# existing ones (prediction_safety.market_timing, lineage_effective_timestamp).
# ---------------------------------------------------------------------------


def test_lineage_effective_timestamp_check_fails_when_injury_observed_at_leaks() -> None:
    leaking_instant = "2026-09-04T00:00:00+00:00"  # after PREDICTION_TIMESTAMP
    forecast = _forecast(
        home_injury_observed_at=[leaking_instant, pd.NaT],
        away_injury_observed_at=[pd.NaT, pd.NaT],
    )
    lineage = build_card_lineage(
        forecast,
        _metadata(),
        feature_columns=_FEATURE_COLUMNS,
        display_fields=_DISPLAY_FIELDS,
    )

    with pytest.raises(LineageError, match="model_input:player_injuries") as error:
        validate_card_lineage(lineage)
    assert "after the prediction timestamp" in str(error.value)


def test_market_timing_check_passes_then_fails_on_a_future_dated_observation(
    model_frame: pd.DataFrame,
) -> None:
    predictions, _ = score_week(model_frame, season=2020, week=1, min_train_games=80)
    predictions = predictions.copy()
    predictions["kickoff"] = pd.Timestamp("2026-09-14T17:00:00+00:00")
    # A real prospective card has no outcome yet -- model_frame's synthetic
    # home_cover/result/ats_margin (built for backtest scoring) would
    # otherwise trip the unrelated outcome_embargo check before market_timing
    # is ever reached.
    for column in ("home_cover", "result", "ats_margin"):
        predictions[column] = np.nan
    created_at = pd.Timestamp("2026-09-10T12:00:00+00:00")

    valid = predictions.copy()
    valid[MARKET_OBSERVED_AT_COLUMN] = pd.Timestamp("2026-09-09T09:00:00+00:00")
    audit = validate_prediction_card(valid, min_edge=0.02, prospective=True, created_at=created_at)
    assert "market_timing" in audit.checks_passed

    leaking = predictions.copy()
    leaking[MARKET_OBSERVED_AT_COLUMN] = pd.Timestamp("2026-09-14T18:00:00+00:00")
    with pytest.raises(PredictionSafetyError, match="market_timing"):
        validate_prediction_card(leaking, min_edge=0.02, prospective=True, created_at=created_at)


# ---------------------------------------------------------------------------
# players.py contract guard: the new keyword-only parameter cannot corrupt an
# unrelated, already-covered contract check (tests/test_players.py's own
# ``test_player_contract_guards`` pattern, plus the new kwarg alongside it).
# ---------------------------------------------------------------------------


def test_enrich_with_player_features_still_rejects_bad_arguments_with_the_new_kwarg() -> None:
    with pytest.raises(ValueError, match="decision_hours"):
        enrich_with_player_features(
            _games(),
            _injuries(),
            _rosters(),
            _snaps(),
            _pbp(),
            decision_hours_before_kickoff=-1,
            injury_snapshot_captured_at="2022-09-09T12:00:00Z",
        )
