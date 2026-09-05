"""``weak_stack_expected_lineup_loss`` prospective challenger recording.

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

import nfl_ats.expected_lineup_loss_challenger as challenger_module
from nfl_ats.data import DataContractError
from nfl_ats.expected_lineup_loss_challenger import (
    CANDIDATE_FEATURE_PROFILE,
    CHALLENGER_ID,
    record_expected_lineup_loss_challenger_decisions,
)
from nfl_ats.prospective_scoring import (
    CHALLENGER_DECISION_COLUMNS,
    config_fingerprint,
    load_challenger_decisions,
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
        challenger_module, "build_stacked_features", lambda base_features, *args: base_features
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

    result = record_expected_lineup_loss_challenger_decisions(artifacts, tmp_path, now=now)

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
    again = record_expected_lineup_loss_challenger_decisions(artifacts, tmp_path, now=now)
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
        record_expected_lineup_loss_challenger_decisions(
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
        record_expected_lineup_loss_challenger_decisions(
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
        record_expected_lineup_loss_challenger_decisions(
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


def test_both_arms_are_frozen_on_retry(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _, card = _write_active_model_and_card(artifacts, tmp_path)
    _patch_fit(monkeypatch, card, {"2026_01_G0": 0.6, "2026_01_G1": 0.4})
    now = KICKOFF - pd.Timedelta(days=3)
    record_expected_lineup_loss_challenger_decisions(artifacts, tmp_path, now=now)
    path = artifacts / "prospective" / f"{CHALLENGER_ID}_paired_decisions.parquet"
    before = pd.read_parquet(path)
    assert before.baseline_pick_side.tolist() == ["AWAY", "HOME"]
    assert before.pick_side.tolist() == ["HOME", "AWAY"]
    _patch_fit(monkeypatch, card, dict.fromkeys(card.game_id, 0.9))
    record_expected_lineup_loss_challenger_decisions(artifacts, tmp_path, now=now)
    pd.testing.assert_frame_equal(before, pd.read_parquet(path))


def test_missing_lineup_source_skips_cleanly(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts, tmp_path)
    result = record_expected_lineup_loss_challenger_decisions(
        artifacts, tmp_path, now=KICKOFF - pd.Timedelta(days=3)
    )
    assert result["skipped"] and result["recorded"] == 0
    assert load_challenger_decisions(artifacts).empty


def test_future_forecast_refused(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _, card = _write_active_model_and_card(artifacts, tmp_path)
    _patch_fit(monkeypatch, card, dict.fromkeys(card.game_id, 0.6))
    with pytest.raises(DataContractError, match="future-dated"):
        record_expected_lineup_loss_challenger_decisions(artifacts, tmp_path, now=TUESDAY)
    assert load_challenger_decisions(artifacts).empty


def test_runtime_profile_adds_exactly_three_columns_and_restores_after_error():
    from nfl_ats import margin, outcomes
    from nfl_ats.constants import FEATURE_SETS
    from nfl_ats.expected_lineup_loss_features import EXPECTED_LINEUP_LOSS_COLUMNS

    profiles = margin.MARGIN_FEATURE_PROFILES
    baseline = margin.margin_feature_columns("market_residual", "weak_stack")
    before = dict(FEATURE_SETS)
    with pytest.raises(RuntimeError), challenger_module.candidate_profile():
        assert (
            margin.margin_feature_columns("market_residual", CANDIDATE_FEATURE_PROFILE)
            == baseline + EXPECTED_LINEUP_LOSS_COLUMNS
        )
        assert (
            outcomes.margin_feature_set("market_residual", CANDIDATE_FEATURE_PROFILE)
            == "full_lead62"
        )
        assert CANDIDATE_FEATURE_PROFILE in outcomes.MARGIN_FEATURE_PROFILES
        assert profiles == margin.MARGIN_FEATURE_PROFILES
        raise RuntimeError("fit failed")
    assert before == FEATURE_SETS
    assert profiles == margin.MARGIN_FEATURE_PROFILES
    assert CANDIDATE_FEATURE_PROFILE not in outcomes.MARGIN_FEATURE_PROFILES


@pytest.mark.parametrize("publication", ["2026-09-08T13:00:00Z", "2026-09-13T21:00:00Z"])
def test_lineups_and_injuries_observed_after_publication_or_pool_cutoff_are_invisible(publication):
    from nfl_ats.expected_lineup_loss_features import visible_injury_lookup

    now = pd.Timestamp(publication)
    cutoff = pd.Timestamp("2026-09-13T20:00:00Z")
    panel = pd.DataFrame(
        {
            "gsis_id": ["visible", "after_publication", "after_cutoff", "unknown"],
            "source_schema": "daily_dt",
            "decision_at": cutoff,
            "depth_observed_at": [
                TUESDAY - pd.Timedelta(seconds=1),
                TUESDAY + pd.Timedelta(hours=2),
                cutoff + pd.Timedelta(seconds=1),
                pd.NaT,
            ],
        }
    )
    injuries = pd.DataFrame(
        {
            "season": 2026,
            "week": 1,
            "team": "H",
            "gsis_id": "visible",
            "report_status": ["Questionable", "Out", "Out"],
            "practice_status": "DNP",
            "effective_observed_at": [
                TUESDAY - pd.Timedelta(seconds=1),
                TUESDAY + pd.Timedelta(hours=2),
                cutoff + pd.Timedelta(seconds=1),
            ],
        }
    )
    visible, injury_rows = challenger_module.visible_sources(panel, injuries, now)
    assert "after_cutoff" not in set(visible.gsis_id)
    assert "unknown" not in set(visible.gsis_id)
    if now < cutoff:
        assert visible.gsis_id.tolist() == ["visible"]
    decisions = pd.DataFrame(
        {"season": [2026], "week": [1], "team": ["H"], "decision_at": [cutoff]}
    )
    lookup = visible_injury_lookup(injury_rows, decisions)
    assert lookup.report_status.tolist() == (["Questionable"] if now < cutoff else ["Out"])


def test_runtime_profile_fits_real_model(model_frame):
    from nfl_ats.constants import FEATURE_SETS
    from nfl_ats.expected_lineup_loss_features import EXPECTED_LINEUP_LOSS_COLUMNS

    frame = model_frame.copy()
    for column in (*FEATURE_SETS["full_weak_stack"], *EXPECTED_LINEUP_LOSS_COLUMNS):
        if column not in frame:
            frame[column] = 0.0
    with challenger_module.candidate_profile():
        target, models = challenger_module.fit_margin_models_for_week(
            frame,
            season=2020,
            week=4,
            regressor="ridge",
            min_train_games=20,
            feature_profile=CANDIDATE_FEATURE_PROFILE,
            ridge_alpha=10.0,
            methods=("market_residual",),
        )
        got = models["market_residual"].predict(target, probability_method="gaussian")
    assert len(got) == len(target) > 0
    assert got.home_cover_probability.between(0, 1).all()


def test_build_reads_only_visible_current_lineup(tmp_path, monkeypatch):
    from nfl_ats.expected_lineup_loss_features import EXPECTED_LINEUP_LOSS_COLUMNS

    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(
        {
            "decision_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "source_schema": pd.Series(dtype=str),
        }
    ).to_parquet(processed / "play_probability_panel.parquet")
    card = pd.DataFrame(
        {
            "game_id": ["g"],
            "season": [2026],
            "week": [1],
            "home_team": ["SEA"],
            "away_team": ["NE"],
            "kickoff": ["2026-09-14T00:20:00Z"],
        }
    )
    depth = pd.DataFrame(
        {
            "dt": ["2026-09-08T12:00:00Z", "2026-09-08T14:00:00Z", "2026-09-13T20:01:00Z"],
            "team": "SEA",
            "player_name": ["Early", "Later", "Post cutoff"],
            "gsis_id": ["early", "later", "post"],
            "pos_abb": "QB",
            "pos_rank": 1,
        }
    )
    injuries = pd.DataFrame(
        {
            "effective_observed_at": pd.to_datetime(
                ["2026-09-08T12:00:00Z", "2026-09-08T14:00:00Z"], utc=True
            )
        }
    )
    monkeypatch.setattr(challenger_module, "latest_player_snapshot", lambda root: None)
    monkeypatch.setattr(challenger_module, "latest_depth_snapshot", lambda root: None)
    monkeypatch.setattr(
        challenger_module,
        "load_player_snapshot",
        lambda *args, **kwargs: (injuries, pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(challenger_module, "load_depth_snapshot", lambda snapshot: depth)
    monkeypatch.setattr(
        challenger_module,
        "attach_history_features",
        lambda current, *args: current.assign(trailing4_snap_share=0.8, weeks_since_last_snap=1.0),
    )

    def attach(base, *, panel, injuries):
        assert panel.gsis_id.tolist() == ["early"]
        assert len(injuries) == 1
        assert panel.decision_at.iloc[0] == pd.Timestamp("2026-09-13T20:00:00Z")
        return base.assign(**dict.fromkeys(EXPECTED_LINEUP_LOSS_COLUMNS, 0.1))

    monkeypatch.setattr(challenger_module, "attach_expected_lineup_loss_features", attach)
    got = challenger_module.build_stacked_features(card, tmp_path, TUESDAY, card)
    assert len(got) == 1
