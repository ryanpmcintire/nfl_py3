from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
from nfl_ats.clv import PAPER_DECISION_COLUMNS, paper_decision_ledger_path
from nfl_ats.data import DataContractError
from nfl_ats.four_overlay_composition import INCUMBENT_CHALLENGER_ID, POLICY_ID
from nfl_ats.four_overlay_incumbent import record_former_production_incumbent_decisions
from nfl_ats.io import atomic_parquet
from nfl_ats.prospective_scoring import (
    artifact_model_config,
    load_challenger_decisions,
)


def _write_fixture(root: Path, *, complete_primary: bool = True) -> tuple[Path, Path]:
    artifacts = root / "artifacts"
    forecast = artifacts / "margin_predictions" / "2026-week-01-test"
    forecast.mkdir(parents=True)
    metadata = {
        "season": 2026,
        "week": 1,
        "created_at_utc": "2026-09-08T16:00:00+00:00",
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
                "sha256": "feature-hash",
            }
        },
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    card = pd.DataFrame(
        {
            "game_id": ["2026_01_A_B", "2026_01_C_D"],
            "season": [2026, 2026],
            "week": [1, 1],
            "kickoff": ["2026-09-10T00:20:00Z", "2026-09-13T17:00:00Z"],
            "away_team": ["A", "C"],
            "home_team": ["B", "D"],
        }
    )
    card.to_csv(forecast / "recommendations.csv", index=False)
    active = {
        "version": ACTIVE_ATS_MODEL_VERSION,
        "status": "SYNCHRONIZED",
        "model_id": "model-test",
        "weekly_forecast": {"artifact": "margin_predictions/2026-week-01-test"},
    }
    (artifacts / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")
    registry = {
        "challengers": [
            {
                "challenger_id": INCUMBENT_CHALLENGER_ID,
                "status": "ACTIVE_PROSPECTIVE",
                "model": artifact_model_config(metadata),
            }
        ]
    }
    registry_path = artifacts / "prospective" / "challengers.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    rows = []
    for index, game in card.iterrows():
        rows.append(
            {
                "recorded_at_utc": pd.Timestamp("2026-09-08T16:00:00Z"),
                "forecast_artifact": forecast.name,
                "forecast_created_at_utc": pd.Timestamp(metadata["created_at_utc"]),
                "model_id": "model-test",
                "method": "market_residual",
                "decision_policy_id": POLICY_ID,
                "decision_policy_fingerprint": "policy-fingerprint",
                "game_id": game["game_id"],
                "season": 2026,
                "week": 1,
                "kickoff": pd.Timestamp(game["kickoff"]),
                "away_team": game["away_team"],
                "home_team": game["home_team"],
                "model_pick_side": "HOME",
                "pre_arrest_pick_side": "HOME",
                "former_policy_pick_side": "AWAY" if index == 0 else "HOME",
                "pick_side": "AWAY" if index == 0 else "HOME",
                "coach_fade_flip": index == 0,
                "division_revenge_flip": False,
                "player_arrests_flip": False,
                "spread_gap_zone_flip": False,
                "composed_overlay_flip": index == 0,
                "player_arrests_home_flag": False,
                "player_arrests_away_flag": False,
                "player_arrests_snapshot_id": "arrests-tuesday",
                "player_arrests_snapshot_fetched_at_utc": pd.Timestamp("2026-09-08T15:00:00Z"),
                "player_arrests_safe_index_sha256": "arrest-hash",
                "schedule_snapshot_id": "schedule-tuesday",
                "schedule_parquet_sha256": "schedule-hash",
                "bet_side": "PASS",
                "decision_home_spread": -2.5,
                "edge": float("nan"),
                "is_best_pick": index == 0,
            }
        )
    if not complete_primary:
        rows.pop()
    atomic_parquet(
        pd.DataFrame(rows)[list(PAPER_DECISION_COLUMNS)], paper_decision_ledger_path(artifacts)
    )
    return artifacts, root / "data"


def test_records_exact_frozen_former_policy_side_and_is_idempotent(tmp_path: Path) -> None:
    artifacts, data_root = _write_fixture(tmp_path)
    now = datetime(2026, 9, 8, 16, 30, tzinfo=UTC)

    result = record_former_production_incumbent_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert ledger.loc["2026_01_A_B", "pick_side"] == "AWAY"
    assert ledger.loc["2026_01_C_D", "pick_side"] == "HOME"
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    again = record_former_production_incumbent_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_refuses_an_incomplete_primary_card(tmp_path: Path) -> None:
    artifacts, data_root = _write_fixture(tmp_path, complete_primary=False)

    with pytest.raises(DataContractError, match="complete current four-overlay card"):
        record_former_production_incumbent_decisions(
            artifacts,
            data_root,
            now=datetime(2026, 9, 8, 16, 30, tzinfo=UTC),
        )
    assert load_challenger_decisions(artifacts).empty
