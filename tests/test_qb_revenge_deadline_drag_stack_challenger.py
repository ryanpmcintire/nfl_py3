"""``weak_stack_qb_revenge_deadline_drag`` prospective challenger recording.

Mirrors ``tests/test_best_pick_nomination.py`` section 6's registry/active-
model/card fixtures and its strategy of monkeypatching the expensive
step (here ``fit_margin_models_for_week`` plus ``build_stacked_features``)
rather than building a real fittable feature table: this pins the
RECORDER's own plumbing (registration, fingerprint pinning, anti-
backdating, ledger append, the declared ``feature_profile`` deviation) --
the ridge fit itself is already covered by ``tests/test_margin.py``,
``tests/test_promotion_eval_profiles.py``,
``tests/test_qb_identity_features.py``, and
``tests/test_transaction_flag_features.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

import nfl_ats.qb_revenge_deadline_drag_stack_challenger as challenger_module
from nfl_ats.data import DataContractError
from nfl_ats.prospective_scoring import (
    CHALLENGER_DECISION_COLUMNS,
    config_fingerprint,
    load_challenger_decisions,
)
from nfl_ats.qb_revenge_deadline_drag_stack_challenger import (
    CANDIDATE_FEATURE_PROFILE,
    CHALLENGER_ID,
    record_qb_revenge_deadline_drag_stack_challenger_decisions,
)

TUESDAY = pd.Timestamp("2026-09-08T13:00:00Z")  # a real Tuesday
KICKOFF = TUESDAY + pd.Timedelta(days=4)

#: A SNAPSHOT of the active model's own configuration, matching the module
#: docstring's "Declared deviation" note -- this is NOT this challenger's
#: own fit recipe, only what its recording guard pins against.
_MODEL_CONFIG = {
    "method": "market_residual",
    "target": "market_residual",
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "calibration_method": "none",
    "feature_profile": "weak_stack",
    "feature_set": "full_weak_stack",
    "min_edge": 0.02,
    "min_train_games": 500,
    "feature_table": "features.parquet",
}


def _write_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    payload = {
        "challengers": [
            {"challenger_id": CHALLENGER_ID, "status": status, "model": dict(_MODEL_CONFIG)}
        ]
    }
    path = artifacts / "prospective" / "challengers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_active_model_and_card(
    artifacts: Path, tmp_path: Path, *, ridge_alpha: float = 10.0, n_games: int = 2
) -> tuple[Path, pd.DataFrame]:
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
        "ridge_alpha": ridge_alpha,
        "calibration_method": "none",
        "feature_profile": "weak_stack",
        "min_edge": 0.02,
        "min_train_games": 500,
        "provenance": {"feature_table": {"path": "features.parquet", "sha256": "abc123"}},
    }

    kickoffs = [KICKOFF + pd.Timedelta(hours=i) for i in range(n_games)]
    # Active picks AWAY, HOME, ... (0.40, 0.60, 0.40, 0.60, ...) so the fake
    # candidate probabilities below can deliberately disagree on every game.
    card = pd.DataFrame(
        {
            "game_id": [f"2026_01_G{i}" for i in range(n_games)],
            "season": 2026,
            "week": 1,
            "kickoff": [ts.isoformat() for ts in kickoffs],
            "away_team": [f"AWY{i}" for i in range(n_games)],
            "home_team": [f"HME{i}" for i in range(n_games)],
            "spread_line": [1.5 + i for i in range(n_games)],
            "home_cover_probability": [0.40 if i % 2 == 0 else 0.60 for i in range(n_games)],
        }
    )
    card.to_csv(forecast / "recommendations.csv", index=False)

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

    features_path = tmp_path / "features.parquet"
    metadata["provenance"]["feature_table"]["path"] = str(features_path)
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame({"game_id": ["placeholder"]}).to_parquet(features_path)
    return forecast, card


class _FakeModel:
    """Stands in for the real fitted ``MarginModel``: returns a caller-chosen
    ``home_cover_probability`` per ``game_id`` and asserts it was asked for
    the gaussian mapping, matching production's own weekly-forecast call."""

    def __init__(self, probabilities: dict[str, float]) -> None:
        self._probabilities = probabilities

    def predict(self, target: pd.DataFrame, *, probability_method: str) -> pd.DataFrame:
        assert probability_method == "gaussian"
        return pd.DataFrame(
            {
                "home_cover_probability": [
                    self._probabilities[game_id] for game_id in target["game_id"]
                ]
            },
            index=target.index,
        )


def _patch_fit(
    monkeypatch: pytest.MonkeyPatch, card: pd.DataFrame, probabilities: dict[str, float]
) -> None:
    """Bypass the real attach-features/ridge-refit machinery so this test
    pins the RECORDER's own plumbing, not the fit (see module docstring)."""

    monkeypatch.setattr(
        challenger_module, "build_stacked_features", lambda base_features: base_features
    )

    def fake_fit(
        features: pd.DataFrame,
        *,
        season: int,
        week: int,
        regressor: str,
        min_train_games: int,
        feature_profile: str,
        ridge_alpha: float,
        methods: tuple[str, ...],
    ) -> tuple[pd.DataFrame, dict[str, _FakeModel]]:
        assert feature_profile == CANDIDATE_FEATURE_PROFILE
        assert methods == ("market_residual",)
        target = card[["game_id", "season", "week"]].copy()
        return target, {"market_residual": _FakeModel(probabilities)}

    monkeypatch.setattr(challenger_module, "fit_margin_models_for_week", fake_fit)


def test_record_challenger_decisions_records_every_game_and_flags_diffs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _forecast, card = _write_active_model_and_card(artifacts, tmp_path, n_games=2)
    # Active picks AWAY (0.40) then HOME (0.60); the candidate disagrees on
    # both, so picks_differing_from_active must read 2.
    _patch_fit(monkeypatch, card, {"2026_01_G0": 0.60, "2026_01_G1": 0.40})
    now = KICKOFF - pd.Timedelta(days=3)

    result = record_qb_revenge_deadline_drag_stack_challenger_decisions(
        artifacts, tmp_path, now=now
    )

    assert result["challenger_id"] == CHALLENGER_ID
    assert result["recorded"] == 2
    assert result["already_recorded"] == 0
    assert result["picks_differing_from_active"] == 2

    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()
    assert ledger.loc["2026_01_G0", "pick_side"] == "HOME"  # candidate 0.60
    assert ledger.loc["2026_01_G1", "pick_side"] == "AWAY"  # candidate 0.40
    # Declared deviation from the tilt-overlay/nomination convention: this
    # ledger's own feature_profile column is the CANDIDATE's profile, not a
    # literal copy of the active model's "weak_stack".
    assert (ledger["feature_profile"] == CANDIDATE_FEATURE_PROFILE).all()
    assert (ledger["decision_home_spread"] == card["spread_line"].to_numpy()).all()

    # Re-running is a no-op: append-only, never rewrites.
    again = record_qb_revenge_deadline_drag_stack_challenger_decisions(artifacts, tmp_path, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2
    assert len(load_challenger_decisions(artifacts)) == 2


def test_record_challenger_refuses_outside_recording_lock_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _forecast, card = _write_active_model_and_card(artifacts, tmp_path)
    _patch_fit(monkeypatch, card, dict.fromkeys(card["game_id"], 0.5))

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_qb_revenge_deadline_drag_stack_challenger_decisions(
            artifacts, tmp_path, now=datetime(2026, 8, 18, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_challenger_refuses_a_fingerprint_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    # The active model's OWN configuration moved (a promotion) since this
    # challenger was pinned -- recording must refuse, not silently switch
    # base models under the same challenger id.
    _forecast, card = _write_active_model_and_card(artifacts, tmp_path, ridge_alpha=1.0)
    _patch_fit(monkeypatch, card, dict.fromkeys(card["game_id"], 0.5))

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_qb_revenge_deadline_drag_stack_challenger_decisions(
            artifacts, tmp_path, now=KICKOFF - pd.Timedelta(days=3)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_challenger_refuses_an_inactive_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _forecast, card = _write_active_model_and_card(artifacts, tmp_path)
    _patch_fit(monkeypatch, card, dict.fromkeys(card["game_id"], 0.5))

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_qb_revenge_deadline_drag_stack_challenger_decisions(
            artifacts, tmp_path, now=KICKOFF - pd.Timedelta(days=3)
        )


def test_fingerprint_helper_agrees_with_the_registered_model_block() -> None:
    """Sanity check that the fixture's config really matches CONFIG_FINGERPRINT_KEYS."""

    metadata = {
        "ats_method": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "calibration_method": "none",
        "feature_profile": "weak_stack",
        "min_edge": 0.02,
        "min_train_games": 500,
        "provenance": {"feature_table": {"path": "features.parquet"}},
    }
    from nfl_ats.prospective_scoring import artifact_model_config

    assert config_fingerprint(_MODEL_CONFIG) == config_fingerprint(artifact_model_config(metadata))


def test_real_registry_entry_fingerprint_is_internally_consistent() -> None:
    """The TRACKED ``artifacts/prospective/challengers.json`` entry for this
    challenger must declare a ``config_fingerprint`` matching its own
    ``model`` block -- the same self-consistency every sibling entry keeps."""

    registry_path = (
        Path(__file__).resolve().parents[1] / "artifacts" / "prospective" / "challengers.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = [
        entry for entry in registry["challengers"] if entry.get("challenger_id") == CHALLENGER_ID
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] == "ACTIVE_PROSPECTIVE"
    assert entry["config_fingerprint"] == config_fingerprint(entry["model"])
    assert "nfl-ats publish-predictions --record-decisions" in entry["weekly_recording_command"]
