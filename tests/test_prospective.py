from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats import cli
from nfl_ats.clv import PAPER_DECISION_COLUMNS
from nfl_ats.prospective import (
    FROZEN_PREDICTION_COLUMNS,
    MOVEMENT_RULE_COMPOSED_CHALLENGER_ID,
    NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID,
    freeze_forecast,
    movement_rule_pick,
    nflcom_out2_starters_flip,
    nflcom_team_starter_out_counts,
    record_movement_rule_composed_challenger_decisions,
    record_nflcom_refresh_out2_starters_challenger_decisions,
    verify_frozen_forecast,
)
from nfl_ats.prospective_scoring import config_fingerprint
from nfl_ats.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_CHALLENGER_IDS = (
    MOVEMENT_RULE_COMPOSED_CHALLENGER_ID,
    NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID,
)
EXPECTED_REGISTRY_SOURCES = {
    MOVEMENT_RULE_COMPOSED_CHALLENGER_ID: "registry/weak_signals.json:movement_rule_composed_chain",
    NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID: (
        "registry/weak_signals.json:nflcom_refresh_out2_starters_on_chain"
    ),
}
NOW = datetime(2026, 9, 8, 15, 0, tzinfo=UTC)
KICKOFF = pd.Timestamp("2026-09-12T17:00:00Z")


def _predictions(created_at: datetime) -> pd.DataFrame:
    rows = 2
    frame = pd.DataFrame({column: [None] * rows for column in FROZEN_PREDICTION_COLUMNS})
    frame["game_id"] = ["2026_01_A_B", "2026_01_C_D"]
    frame["season"] = 2026
    frame["week"] = 1
    frame["gameday"] = [created_at.date() + timedelta(days=2)] * rows
    frame["kickoff"] = [created_at + timedelta(days=2), created_at + timedelta(days=3)]
    frame["away_team"] = ["A", "C"]
    frame["home_team"] = ["B", "D"]
    frame["spread_line"] = [2.5, -1.5]
    frame["away_spread_odds"] = -110.0
    frame["home_spread_odds"] = -110.0
    frame["home_cover_probability"] = [0.55, 0.48]
    frame["pick"] = ["HOME", "AWAY"]
    frame["bet_side"] = ["HOME", "PASS"]
    break_even = 110 / 210
    frame["edge"] = [0.55 - break_even, 0.52 - break_even]
    frame["bet_odds"] = [-110.0, float("nan")]
    frame["break_even_probability"] = [break_even, float("nan")]
    frame["market_home_no_vig_probability"] = 0.5
    frame["market_hold"] = (2 * break_even) - 1
    frame["train_rows"] = 4000
    frame["train_max_gameday"] = created_at.date() - timedelta(days=1)
    frame["home_cover"] = pd.NA
    return frame


def test_freeze_forecast_writes_immutable_record(tmp_path) -> None:
    created_at = datetime(2026, 8, 12, 12, tzinfo=UTC)
    frozen = freeze_forecast(
        _predictions(created_at),
        {"model_name": "logistic", "feature_set": "market_context"},
        tmp_path,
        created_at=created_at,
    )
    assert frozen.games == 2
    manifest_path = frozen.directory / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["forecast_id"] == frozen.forecast_id
    assert manifest["schema_version"] == 2
    assert manifest["prediction_safety"]["status"] == "PASS_WITH_WARNINGS"
    assert len(manifest["predictions_sha256"]) == 64
    stored = pd.read_parquet(frozen.directory / "predictions.parquet")
    assert stored["forecast_id"].eq(frozen.forecast_id).all()
    assert str(stored["kickoff"].dt.tz) == "UTC"
    assert verify_frozen_forecast(frozen.directory)["games"] == 2

    # Revalidation catches internally corrupt data even if an attacker or bug
    # also recomputes the file digest in the manifest.
    stored.loc[0, "edge"] = 0.99
    prediction_path = frozen.directory / "predictions.parquet"
    stored.to_parquet(prediction_path, index=False)
    manifest["predictions_sha256"] = sha256_file(prediction_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="decision_policy"):
        verify_frozen_forecast(frozen.directory)

    with pytest.raises(ValueError, match="already exists"):
        freeze_forecast(_predictions(created_at), {}, tmp_path, created_at=created_at)


def test_freeze_forecast_rejects_unverifiable_or_retrospective_rows(tmp_path) -> None:
    created_at = datetime(2026, 8, 12, 12, tzinfo=UTC)
    predictions = _predictions(created_at)
    predictions.loc[0, "kickoff"] = pd.NaT
    with pytest.raises(ValueError, match="kickoff is missing"):
        freeze_forecast(predictions, {}, tmp_path, created_at=created_at)

    predictions = _predictions(created_at)
    predictions.loc[0, "kickoff"] = created_at
    with pytest.raises(ValueError, match="at or after kickoff"):
        freeze_forecast(predictions, {}, tmp_path, created_at=created_at)

    predictions = _predictions(created_at)
    predictions.loc[0, "home_cover"] = 1.0
    with pytest.raises(ValueError, match="before outcomes"):
        freeze_forecast(predictions, {}, tmp_path, created_at=created_at)

    predictions = _predictions(created_at)
    predictions.loc[0, "home_spread_odds"] = pd.NA
    with pytest.raises(ValueError, match="missing lines or prices"):
        freeze_forecast(predictions, {}, tmp_path, created_at=created_at)


def test_verify_frozen_forecast_rejects_manifest_corruption(tmp_path) -> None:
    created_at = datetime(2026, 8, 12, 12, tzinfo=UTC)
    frozen = freeze_forecast(
        _predictions(created_at),
        {"model_name": "logistic", "min_edge": 0.02},
        tmp_path,
        created_at=created_at,
    )
    manifest_path = frozen.directory / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    corruptions = (
        ({**original, "predictions_sha256": "0" * 64}, "digest mismatch"),
        ({**original, "games": 999}, "row-count mismatch"),
        ({**original, "forecast_id": "wrong"}, "identity mismatch"),
        ({key: value for key, value in original.items() if key != "prediction_safety"}, "safety"),
    )
    for corrupted, message in corruptions:
        manifest_path.write_text(json.dumps(corrupted), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            verify_frozen_forecast(frozen.directory)

    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    assert verify_frozen_forecast(frozen.directory)["forecast_id"] == frozen.forecast_id

    empty = tmp_path / "incomplete"
    empty.mkdir()
    with pytest.raises(ValueError, match="Incomplete"):
        verify_frozen_forecast(empty)


def _repo_challenger_entries() -> dict[str, dict[str, object]]:
    payload = json.loads(
        (REPO_ROOT / "artifacts" / "prospective" / "challengers.json").read_text(encoding="utf-8")
    )
    return {entry["challenger_id"]: entry for entry in payload["challengers"]}


def test_2026_challenger_registrations_present_and_schema_valid() -> None:
    entries = _repo_challenger_entries()
    for challenger_id in NEW_CHALLENGER_IDS:
        assert challenger_id in entries
        entry = entries[challenger_id]
        assert entry["status"] == "ACTIVE_PROSPECTIVE"
        for field in (
            "registered_at_utc",
            "registered_by",
            "status_reason",
            "config_fingerprint",
            "model",
            "model_note",
            "evidence",
            "prospective_protocol",
            "weekly_recording_command",
        ):
            assert field in entry
        assert config_fingerprint(entry["model"]) == entry["config_fingerprint"]
        command = entry["weekly_recording_command"]
        assert "nfl-ats publish-predictions --record-decisions" in command
        evidence_sources = entry["evidence"]["registry_source"]
        if isinstance(evidence_sources, str):
            evidence_sources = [evidence_sources]
        assert EXPECTED_REGISTRY_SOURCES[challenger_id] in evidence_sources


def test_nflcom_registration_carries_the_verbatim_frozen_rule_text() -> None:
    entry = _repo_challenger_entries()[NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID]
    rule_text = str(entry["rule_frozen_text"])
    doc = (REPO_ROOT / "docs" / "nflcom_friday_refresh.md").read_text(encoding="utf-8")
    for required_line in (
        "Rule (frozen): For each REG game on the weekly card",
        "starter-caliber = played >=50% of offensive or defensive snaps",
        "both-flagged keeps; page absent or failing",
        "Fails open: any ingest or freshness failure",
    ):
        assert required_line in rule_text
        assert required_line in doc


def test_publish_time_map_includes_both_new_challengers() -> None:
    keys = cli.PUBLISH_CHALLENGER_RESULT_KEYS
    assert keys[MOVEMENT_RULE_COMPOSED_CHALLENGER_ID] == "movement_rule_composed_challenger_ledger"
    assert (
        keys[NFLCOM_REFRESH_OUT2_STARTERS_CHALLENGER_ID]
        == "nflcom_refresh_out2_starters_challenger_ledger"
    )
    assert len(set(keys.values())) == len(keys)


def test_movement_rule_pick_follows_market_only_above_threshold() -> None:
    assert movement_rule_pick("AWAY", 1.0) == "HOME"
    assert movement_rule_pick("AWAY", 2.5) == "HOME"
    assert movement_rule_pick("HOME", -1.5) == "AWAY"
    assert movement_rule_pick("AWAY", 0.99) == "AWAY"
    assert movement_rule_pick("HOME", -0.5) == "HOME"
    assert movement_rule_pick("HOME", None) == "HOME"


def test_nflcom_out2_starters_flip_matches_the_frozen_rule() -> None:
    assert nflcom_out2_starters_flip(True, 2, 0) is False
    assert nflcom_out2_starters_flip(False, 3, 1) is True
    assert nflcom_out2_starters_flip(True, 2, 2) is True
    assert nflcom_out2_starters_flip(False, 1, 0) is False
    assert nflcom_out2_starters_flip(True, 0, 5) is True


def test_nflcom_team_starter_out_counts_aggregates_per_canonical_team_week(
    tmp_path: Path,
) -> None:
    """Regression pin: the shared counter the publish-time recorder AND the
    refresh-path overlay (nfl_ats.nflcom_refresh_overlay) both consume. The
    pre-2026-08-24 inline version crashed on a pandas as_index=False quirk
    before this extraction; teams with zero flagged Outs are simply absent."""

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    pd.DataFrame(
        [
            {"player": "Alpha One", "game_status": "Out", "season": 2026, "week": 5, "team": "BBB"},
            {"player": "Beta Two", "game_status": "Out", "season": 2026, "week": 5, "team": "BBB"},
            {
                "player": "Gamma Three",
                "game_status": "Out",
                "season": 2026,
                "week": 5,
                "team": "AAA",
            },
            # A non-starter-caliber Out (no prior-week snap share >=50%) must not
            # count -- but its team-week still appears in the mapping, at 0.
            {
                "player": "Nobody Four",
                "game_status": "Out",
                "season": 2026,
                "week": 5,
                "team": "CCC",
            },
        ]
    ).to_parquet(snapshot / "injuries.parquet", index=False)
    players_raw = tmp_path / "players" / "raw" / "snap"
    players_raw.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "season": 2026,
                "game_type": "REG",
                "week": 4,
                "team": "BBB",
                "player": "Alpha One",
                "offense_pct": 0.9,
                "defense_pct": 0.0,
            },
            {
                "season": 2026,
                "game_type": "REG",
                "week": 4,
                "team": "BBB",
                "player": "Beta Two",
                "offense_pct": 0.0,
                "defense_pct": 0.7,
            },
            {
                "season": 2026,
                "game_type": "REG",
                "week": 4,
                "team": "AAA",
                "player": "Gamma Three",
                "offense_pct": 0.8,
                "defense_pct": 0.0,
            },
        ]
    ).to_parquet(players_raw / "snap_counts.parquet", index=False)

    counts = nflcom_team_starter_out_counts(snapshot, players_raw / "snap_counts.parquet")

    assert counts == {
        (2026, 5, "BBB"): 2,
        (2026, 5, "AAA"): 1,
        (2026, 5, "CCC"): 0,
    }


def _write_registry(artifacts: Path) -> None:
    entries = _repo_challenger_entries()
    payload = {
        "ledger": "prospective_challengers",
        "schema_version": 1,
        "challengers": [
            {
                "challenger_id": challenger_id,
                "status": entries[challenger_id]["status"],
                "model": entries[challenger_id]["model"],
            }
            for challenger_id in NEW_CHALLENGER_IDS
        ],
    }
    path = artifacts / "prospective" / "challengers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_active_model_card_and_paper_ledger(artifacts: Path) -> None:
    forecast = artifacts / "margin_predictions" / "2026-week-01-forecast"
    forecast.mkdir(parents=True, exist_ok=True)
    metadata = {
        "active_model_id": "model-xyz",
        "synchronization_status": "SYNCHRONIZED",
        "season": 2026,
        "week": 1,
        "created_at_utc": "2026-09-08T15:00:00+00:00",
        "ats_method": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "calibration_method": "none",
        "feature_profile": "weak_stack",
        "min_edge": 0.02,
        "min_train_games": 500,
        "provenance": {
            "feature_table": {
                "path": "data/processed/game_features_weak_stack.parquet",
                "sha256": "abc123",
            }
        },
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame(
        {
            "game_id": ["2026_01_AAA_BBB", "2026_01_CCC_DDD"],
            "pick_side": ["HOME", "AWAY"],
        }
    ).to_csv(forecast / "recommendations.csv", index=False)
    active = {
        "version": 1,
        "status": "SYNCHRONIZED",
        "model_id": "model-xyz",
        "method": "market_residual",
        "feature_profile": "weak_stack",
        "historical_evaluation": {"accuracy": 0.52, "correct": 1, "games": 1, "intervals": {}},
        "weekly_forecast": {
            "artifact": "margin_predictions/2026-week-01-forecast",
            "season": 2026,
            "week": 1,
        },
    }
    (artifacts / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")

    ledger = pd.DataFrame(
        {
            "recorded_at_utc": [pd.Timestamp(NOW)] * 2,
            "forecast_artifact": ["2026-week-01-forecast"] * 2,
            "forecast_created_at_utc": [pd.Timestamp(NOW)] * 2,
            "model_id": ["model-xyz"] * 2,
            "method": ["market_residual"] * 2,
            "decision_policy_id": [
                "overlay_union_coach_division_revenge_player_arrests_spread_gap_v1"
            ]
            * 2,
            "decision_policy_fingerprint": ["fingerprint"] * 2,
            "game_id": ["2026_01_AAA_BBB", "2026_01_CCC_DDD"],
            "season": [2026, 2026],
            "week": [1, 1],
            "kickoff": [KICKOFF, KICKOFF + pd.Timedelta(hours=1)],
            "away_team": ["AAA", "CCC"],
            "home_team": ["BBB", "DDD"],
            "model_pick_side": ["HOME", "AWAY"],
            "pre_arrest_pick_side": ["HOME", "AWAY"],
            "former_policy_pick_side": ["HOME", "AWAY"],
            "pick_side": ["HOME", "AWAY"],
            "coach_fade_flip": [False] * 2,
            "division_revenge_flip": [False] * 2,
            "player_arrests_flip": [False] * 2,
            "spread_gap_zone_flip": [False] * 2,
            "composed_overlay_flip": [False] * 2,
            "player_arrests_home_flag": [False] * 2,
            "player_arrests_away_flag": [False] * 2,
            "player_arrests_snapshot_id": ["snap"] * 2,
            "player_arrests_snapshot_fetched_at_utc": [pd.Timestamp(NOW)] * 2,
            "player_arrests_safe_index_sha256": ["sha"] * 2,
            "schedule_snapshot_id": ["sched"] * 2,
            "schedule_parquet_sha256": ["parquet-sha"] * 2,
            "bet_side": ["PASS"] * 2,
            "decision_home_spread": [-2.5, 3.0],
            "edge": [float("nan"), float("nan")],
            "is_best_pick": [False, False],
        }
    )
    ledger_path = artifacts / "clv_ledger" / "decisions.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger[list(PAPER_DECISION_COLUMNS)].to_parquet(ledger_path, index=False)


def test_movement_rule_recorder_skips_week_without_fresh_capture(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_card_and_paper_ledger(artifacts)

    result = record_movement_rule_composed_challenger_decisions(
        artifacts, tmp_path / "data", now=NOW
    )

    assert result["recorded"] == 0
    assert result["skipped"] is True
    assert "no fresh captured market line" in result["reason"]


def test_movement_rule_recorder_refuses_without_a_recorded_chain_card(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_card_and_paper_ledger(artifacts)
    (artifacts / "clv_ledger" / "decisions.parquet").unlink()

    with pytest.raises(ValueError, match="No recorded original card"):
        record_movement_rule_composed_challenger_decisions(artifacts, tmp_path / "data", now=NOW)


def test_nflcom_recorder_skips_week_without_an_injuries_snapshot(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_card_and_paper_ledger(artifacts)

    result = record_nflcom_refresh_out2_starters_challenger_decisions(
        artifacts, tmp_path / "data", now=NOW
    )

    assert result["recorded"] == 0
    assert result["skipped"] is True
    assert result["reason"] == "no_nflcom_injuries_snapshot"


def test_nflcom_recorder_skips_week_without_snap_counts(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    data = tmp_path / "data"
    _write_registry(artifacts)
    _write_active_model_card_and_paper_ledger(artifacts)
    snapshot = data / "raw" / "nflcom_injuries" / "20260901T000000Z"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(json.dumps({"pages": []}), encoding="utf-8")

    result = record_nflcom_refresh_out2_starters_challenger_decisions(artifacts, data, now=NOW)

    assert result["recorded"] == 0
    assert result["skipped"] is True
    assert result["reason"] == "no_snap_counts_snapshot"


def test_nflcom_recorder_skips_week_when_page_fails_the_freshness_gate(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    data = tmp_path / "data"
    _write_registry(artifacts)
    _write_active_model_card_and_paper_ledger(artifacts)
    snapshot = data / "raw" / "nflcom_injuries" / "20260901T000000Z"
    pages_dir = snapshot / "pages"
    pages_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "player": ["Player One"],
            "game_status": ["Out"],
            "season": [2026],
            "week": [1],
            "team": ["BBB"],
        }
    ).to_parquet(snapshot / "injuries.parquet", index=False)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "season": 2026,
                        "week": 1,
                        # Tuesday of game week: before the Friday 16:00 ET gate.
                        "fetched_at_utc": "2026-09-08T12:00:00Z",
                        "http_status": 200,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    players_raw = data / "players" / "raw" / "20260901T000000Z"
    players_raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "season": [2026],
            "game_type": ["REG"],
            "week": [0],
            "team": ["BBB"],
            "player": ["Player One"],
            "offense_pct": [0.9],
            "defense_pct": [0.0],
        }
    ).to_parquet(players_raw / "snap_counts.parquet", index=False)

    result = record_nflcom_refresh_out2_starters_challenger_decisions(artifacts, data, now=NOW)

    assert result["recorded"] == 0
    assert result["skipped"] is True
    assert "freshness gate failed" in result["reason"]
