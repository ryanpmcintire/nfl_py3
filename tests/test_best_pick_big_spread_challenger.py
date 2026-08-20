"""Prospective-only 10+ point spread Best-Pick eligibility challenger."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nfl_ats.best_pick_big_spread_challenger as big
from nfl_ats.best_pick_big_spread_challenger import (
    BIG_SPREAD_THRESHOLD,
    CHALLENGER_ID,
    BigSpreadNominationResult,
    apply_big_spread_eligibility,
    record_big_spread_nomination_challenger_decisions,
)
from nfl_ats.best_pick_nomination import DispersionPool, NominationV2Result
from nfl_ats.data import DataContractError
from nfl_ats.prospective_scoring import CHALLENGER_DECISION_COLUMNS, load_challenger_decisions

KICKOFF = pd.Timestamp("2026-09-12T17:00:00Z")
_MODEL_CONFIG = {
    "method": "market_residual",
    "target": "market_residual",
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "calibration_method": "none",
    "feature_profile": "base",
    "min_edge": 0.02,
    "min_train_games": 100,
    "feature_table": "features.parquet",
}


def _base_result(rows: list[dict[str, object]], *, game_id: str) -> NominationV2Result:
    table = pd.DataFrame(rows)
    dispersion = DispersionPool(
        frame=table[["game_id", "spread_std", "pool_pass"]].copy(),
        fallback=False,
        fallback_reason=None,
        n_games=len(table),
        n_missing=0,
        n_pool_pass=int(table["pool_pass"].sum()),
    )
    return NominationV2Result(
        game_id=game_id,
        n_tied_at_max=1,
        tie_break="none",
        probability_table=table,
        dispersion=dispersion,
    )


def test_excludes_both_signs_at_the_ten_point_boundary() -> None:
    predictions = pd.DataFrame(
        {
            "game_id": ["big_favorite", "big_dog", "eligible"],
            "spread_line": [-10.0, 10.5, -9.5],
        }
    )
    base = _base_result(
        [
            {
                "game_id": "big_favorite",
                "candidate_dist": 0.30,
                "spread_std": 0.2,
                "pool_pass": True,
            },
            {"game_id": "big_dog", "candidate_dist": 0.25, "spread_std": 0.3, "pool_pass": True},
            {"game_id": "eligible", "candidate_dist": 0.10, "spread_std": 0.4, "pool_pass": True},
        ],
        game_id="big_favorite",
    )

    result = apply_big_spread_eligibility(predictions, base)

    assert BIG_SPREAD_THRESHOLD == 10.0
    assert result.game_id == "eligible"
    assert result.excluded_game_ids == ("big_dog", "big_favorite")
    assert result.fallback_to_v2 is False
    assert result.base_v2_game_id == "big_favorite"


def test_never_resurrects_a_game_rejected_by_the_v2_pool() -> None:
    predictions = pd.DataFrame(
        {"game_id": ["outside_v2", "inside_v2"], "spread_line": [-2.5, -3.5]}
    )
    base = _base_result(
        [
            {
                "game_id": "outside_v2",
                "candidate_dist": 0.40,
                "spread_std": 0.1,
                "pool_pass": False,
            },
            {"game_id": "inside_v2", "candidate_dist": 0.05, "spread_std": 0.2, "pool_pass": True},
        ],
        game_id="inside_v2",
    )

    result = apply_big_spread_eligibility(predictions, base)

    assert result.game_id == "inside_v2"
    assert result.excluded_game_ids == ()


def test_falls_back_to_v2_when_every_v2_candidate_is_a_big_spread() -> None:
    predictions = pd.DataFrame({"game_id": ["top", "other"], "spread_line": [-10.0, 13.5]})
    base = _base_result(
        [
            {"game_id": "top", "candidate_dist": 0.20, "spread_std": 0.4, "pool_pass": True},
            {"game_id": "other", "candidate_dist": 0.10, "spread_std": 0.3, "pool_pass": True},
        ],
        game_id="top",
    )

    result = apply_big_spread_eligibility(predictions, base)

    assert result.game_id == "top"
    assert result.fallback_to_v2 is True
    assert result.excluded_game_ids == ("other", "top")


def test_postgame_columns_cannot_change_nomination_and_inputs_are_unchanged() -> None:
    predictions = pd.DataFrame(
        {
            "game_id": ["big", "small"],
            "spread_line": [-10.5, -3.5],
            "result": [999.0, -999.0],
            "home_cover": [True, False],
        }
    )
    base = _base_result(
        [
            {"game_id": "big", "candidate_dist": 0.20, "spread_std": 0.1, "pool_pass": True},
            {"game_id": "small", "candidate_dist": 0.10, "spread_std": 0.2, "pool_pass": True},
        ],
        game_id="big",
    )
    original_predictions = predictions.copy(deep=True)
    original_table = base.probability_table.copy(deep=True)

    first = apply_big_spread_eligibility(predictions, base)
    predictions.loc[:, "result"] *= -1
    predictions.loc[:, "home_cover"] = ~predictions["home_cover"]
    second = apply_big_spread_eligibility(predictions, base)

    assert first.game_id == second.game_id == "small"
    pd.testing.assert_frame_equal(original_table, base.probability_table)
    pd.testing.assert_series_equal(original_predictions["game_id"], predictions["game_id"])
    pd.testing.assert_series_equal(original_predictions["spread_line"], predictions["spread_line"])


def test_rejects_nonfinite_decision_spreads() -> None:
    predictions = pd.DataFrame({"game_id": ["g"], "spread_line": [np.nan]})
    base = _base_result(
        [{"game_id": "g", "candidate_dist": 0.1, "spread_std": 0.2, "pool_pass": True}],
        game_id="g",
    )
    with pytest.raises(DataContractError, match="non-finite decision spread"):
        apply_big_spread_eligibility(predictions, base)


def _write_registry(artifacts: Path, *, ridge_alpha: float = 10.0) -> None:
    config = dict(_MODEL_CONFIG, ridge_alpha=ridge_alpha)
    payload = {
        "ledger": "prospective_challengers",
        "schema_version": 1,
        "challengers": [
            {"challenger_id": CHALLENGER_ID, "status": "ACTIVE_PROSPECTIVE", "model": config}
        ],
    }
    path = artifacts / "prospective" / "challengers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_active_model_and_card(
    artifacts: Path, tmp_path: Path, *, ridge_alpha: float = 10.0
) -> None:
    forecast = artifacts / "margin_predictions" / "2026-week-01-forecast"
    forecast.mkdir(parents=True, exist_ok=True)
    features_path = tmp_path / "features.parquet"
    pd.DataFrame({"game_id": ["placeholder"]}).to_parquet(features_path)
    metadata = {
        "active_model_id": "model-xyz",
        "synchronization_status": "SYNCHRONIZED",
        "season": 2026,
        "week": 1,
        "created_at_utc": "2026-09-08T15:00:00+00:00",
        "ats_method": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": ridge_alpha,
        "calibration_method": "none",
        "feature_profile": "base",
        "min_edge": 0.02,
        "min_train_games": 100,
        "provenance": {"feature_table": {"path": str(features_path), "sha256": "abc123"}},
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame(
        {
            "game_id": ["big", "small"],
            "season": [2026, 2026],
            "week": [1, 1],
            "kickoff": [KICKOFF.isoformat(), (KICKOFF + pd.Timedelta(hours=1)).isoformat()],
            "away_team": ["AAA", "CCC"],
            "home_team": ["BBB", "DDD"],
            "spread_line": [-10.5, -3.5],
            "home_cover_probability": [0.60, 0.45],
        }
    ).to_csv(forecast / "recommendations.csv", index=False)
    active = {
        "version": 1,
        "status": "SYNCHRONIZED",
        "model_id": "model-xyz",
        "method": "market_residual",
        "feature_profile": "base",
        "historical_evaluation": {"accuracy": 0.52, "correct": 1, "games": 1, "intervals": {}},
        "weekly_forecast": {
            "artifact": "margin_predictions/2026-week-01-forecast",
            "season": 2026,
            "week": 1,
        },
    }
    (artifacts / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")


def _fake_nomination() -> BigSpreadNominationResult:
    return BigSpreadNominationResult(
        game_id="small",
        n_tied_at_max=1,
        tie_break="none",
        probability_table=pd.DataFrame(),
        excluded_game_ids=("big",),
        fallback_to_v2=False,
        base_v2_game_id="big",
    )


def test_recorder_writes_only_the_alternative_nominee_and_never_rewrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts, tmp_path)
    monkeypatch.setattr(big, "nominate_big_spread_challenger", lambda *a, **k: _fake_nomination())
    now = KICKOFF - pd.Timedelta(days=3)

    result = record_big_spread_nomination_challenger_decisions(artifacts, tmp_path, now=now)

    assert result["recorded"] == 1
    assert result["nominated_game_id"] == "small"
    assert result["base_v2_game_id"] == "big"
    assert result["excluded_game_ids"] == ["big"]
    ledger = load_challenger_decisions(artifacts)
    assert list(ledger.columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert len(ledger) == 1
    row = ledger.iloc[0]
    assert row["challenger_id"] == CHALLENGER_ID
    assert row["game_id"] == "small"
    assert row["pick_side"] == "AWAY"
    assert row["decision_home_spread"] == -3.5
    assert row["bet_side"] == "PASS"
    assert pd.isna(row["edge"])

    again = record_big_spread_nomination_challenger_decisions(artifacts, tmp_path, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 1
    assert len(load_challenger_decisions(artifacts)) == 1


def test_recorder_refuses_backdated_week(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts, tmp_path)
    monkeypatch.setattr(big, "nominate_big_spread_challenger", lambda *a, **k: _fake_nomination())

    result = record_big_spread_nomination_challenger_decisions(
        artifacts, tmp_path, now=KICKOFF + pd.Timedelta(minutes=1)
    )

    assert result["recorded"] == 0
    assert result["post_kickoff_skipped"] == 1
    assert load_challenger_decisions(artifacts).empty


def test_recorder_refuses_outside_lock_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts, tmp_path)
    monkeypatch.setattr(big, "nominate_big_spread_challenger", lambda *a, **k: _fake_nomination())

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_big_spread_nomination_challenger_decisions(
            artifacts, tmp_path, now=datetime(2026, 8, 1, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_recorder_refuses_configuration_fingerprint_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, ridge_alpha=10.0)
    _write_active_model_and_card(artifacts, tmp_path, ridge_alpha=1.0)
    monkeypatch.setattr(big, "nominate_big_spread_challenger", lambda *a, **k: _fake_nomination())

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_big_spread_nomination_challenger_decisions(
            artifacts, tmp_path, now=KICKOFF - pd.Timedelta(days=3)
        )
    assert load_challenger_decisions(artifacts).empty
