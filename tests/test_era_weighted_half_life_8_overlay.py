"""Era-weighted (half-life 8) challenger (docs/era_weighting_screen.md, MOD-14).

Mirrors ``tests/test_ecdf_mapping_incumbent_overlay.py``'s structure closely:
this challenger's "control" is a UNIFORM-weight refit (rather than a
different probability-smoothing method), so the fixture card here is built
the ordinary (unweighted) way via ``fit_margin_models_for_week`` +
``model.predict(..., probability_method="gaussian")`` -- exactly what the
promoted production default (``score_outcome_week``) produces.

1. :func:`apply_era_weighted_half_life_8_overlay` reproduces the (uniform-
   weight) Gaussian probability from a refit before trusting anything --
   proving it reads the SAME leak-safe training rows the active card was
   built from -- then replaces every game's ``home_cover_probability`` with
   the half-life-8-weighted refit's Gaussian read, touching no other column.
2. Flip detection is self-consistent.
3. :func:`record_era_weighted_half_life_8_challenger_decisions` writes the
   refit's own picks to the prospective challenger ledger, dual-tracked and
   at no rotation-registry window cost, with the same anti-backdating and
   fingerprint-pin guarantees every other overlay challenger has.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _overlay_test_kit import write_active_model_and_card, write_challenger_registry

from nfl_ats.data import DataContractError
from nfl_ats.era_weighted_half_life_8_overlay import (
    CHALLENGER_ID,
    HALF_LIFE_SEASONS,
    EraWeightedFlip,
    EraWeightedResult,
    apply_era_weighted_half_life_8_overlay,
    half_life_weights,
    overlay_disclosure_note,
    record_era_weighted_half_life_8_challenger_decisions,
)
from nfl_ats.outcomes import fit_margin_models_for_week
from nfl_ats.prospective_scoring import CHALLENGER_DECISION_COLUMNS, load_challenger_decisions

_FEATURE_PROFILE = "base"
_RIDGE_ALPHA = 10.0
_MIN_TRAIN_GAMES = 100
_SEASON = 2020
_WEEK = 4
_FEATURE_TABLE_NAME = "era_weighted_test_features.parquet"


def _week_card(
    model_frame: pd.DataFrame,
    *,
    season: int = _SEASON,
    week: int = _WEEK,
    ridge_alpha: float = _RIDGE_ALPHA,
) -> pd.DataFrame:
    """Build a real card the ordinary (unweighted) way, matching the
    promoted production default: via ``fit_margin_models_for_week`` +
    ``model.predict(..., probability_method="gaussian")``, never hand-typed."""

    target, margin_models = fit_margin_models_for_week(
        model_frame,
        season=season,
        week=week,
        regressor="ridge",
        min_train_games=_MIN_TRAIN_GAMES,
        feature_profile=_FEATURE_PROFILE,  # type: ignore[arg-type]
        ridge_alpha=ridge_alpha,
        methods=("market_residual",),
    )
    model = margin_models["market_residual"]
    predicted = model.predict(target, probability_method="gaussian")
    card = target.copy()
    card["home_cover_probability"] = predicted["home_cover_probability"].to_numpy()
    assert not card.empty
    return card


# ---------------------------------------------------------------------------
# 0. half_life_weights: pure arithmetic, ported from scripts/era_weighting_lib.py
# ---------------------------------------------------------------------------


def test_half_life_weights_is_one_for_the_predicted_season() -> None:
    weights = half_life_weights(np.array([2020.0, 2019.0]), predict_season=2020, half_life=8.0)
    assert weights[0] == pytest.approx(1.0)


def test_half_life_weights_decays_by_half_every_half_life() -> None:
    weights = half_life_weights(np.array([2012.0]), predict_season=2020, half_life=8.0)
    assert weights[0] == pytest.approx(0.5)


def test_half_life_weights_never_looks_forward() -> None:
    """A training row can share the predicted season but never postdate it --
    elapsed is clamped at zero, not negative."""

    weights = half_life_weights(np.array([2025.0]), predict_season=2020, half_life=8.0)
    assert weights[0] == pytest.approx(1.0)


def test_half_life_weights_rejects_a_nonpositive_half_life() -> None:
    with pytest.raises(ValueError, match="half_life must be positive"):
        half_life_weights(np.array([2020.0]), predict_season=2020, half_life=0.0)


# ---------------------------------------------------------------------------
# 1. apply_era_weighted_half_life_8_overlay
# ---------------------------------------------------------------------------


def test_overlay_reproduces_the_uniform_control_before_reweighting(
    model_frame: pd.DataFrame,
) -> None:
    """The load-bearing proof: the refit uniform-weight Gaussian check passes
    silently -- this really is fitting the SAME leak-safe training rows the
    active card was built from."""

    card = _week_card(model_frame)
    result = apply_era_weighted_half_life_8_overlay(
        card,
        model_frame,
        regressor="ridge",
        ridge_alpha=_RIDGE_ALPHA,
        feature_profile=_FEATURE_PROFILE,
        min_train_games=_MIN_TRAIN_GAMES,
    )
    assert len(result.overlaid_predictions) == len(card)


def test_overlay_changes_every_probability_and_only_that_column(model_frame: pd.DataFrame) -> None:
    card = _week_card(model_frame)
    result = apply_era_weighted_half_life_8_overlay(
        card,
        model_frame,
        regressor="ridge",
        ridge_alpha=_RIDGE_ALPHA,
        feature_profile=_FEATURE_PROFILE,
        min_train_games=_MIN_TRAIN_GAMES,
    )
    overlaid = result.overlaid_predictions
    assert list(overlaid.columns) == list(card.columns)
    other_columns = [c for c in card.columns if c != "home_cover_probability"]
    pd.testing.assert_frame_equal(
        overlaid[other_columns].reset_index(drop=True),
        card[other_columns].reset_index(drop=True),
        check_exact=True,
    )
    original = card["home_cover_probability"].to_numpy(dtype=float)
    reweighted = overlaid["home_cover_probability"].to_numpy(dtype=float)
    # model_frame spans seasons 2019/2020 (conftest.py), so half-life-8
    # weighting genuinely changes the training row weights for a 2020 week 4
    # target (2019 rows get weight 0.5**(1/8) < 1) -- the refit coefficients,
    # and therefore the probabilities, must differ from the uniform control.
    assert not np.allclose(original, reweighted)


def test_overlay_flip_consistency(model_frame: pd.DataFrame) -> None:
    """Every reported flip is a genuine side-crossing and every non-flip
    stayed on the same side."""

    card = _week_card(model_frame)
    result = apply_era_weighted_half_life_8_overlay(
        card,
        model_frame,
        regressor="ridge",
        ridge_alpha=_RIDGE_ALPHA,
        feature_profile=_FEATURE_PROFILE,
        min_train_games=_MIN_TRAIN_GAMES,
    )
    overlaid = result.overlaid_predictions.set_index("game_id")
    original = card.set_index("game_id")
    flipped_ids = {flip.game_id for flip in result.flips}
    for game_id in original.index:
        original_side = original.loc[game_id, "home_cover_probability"] >= 0.5
        mapped_side = overlaid.loc[game_id, "home_cover_probability"] >= 0.5
        if game_id in flipped_ids:
            assert original_side != mapped_side
        else:
            assert original_side == mapped_side
    assert len(flipped_ids) == result.flip_count


def test_overlay_disabled_is_a_no_op(model_frame: pd.DataFrame) -> None:
    card = _week_card(model_frame)
    result = apply_era_weighted_half_life_8_overlay(card, model_frame, enabled=False)
    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        card.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_requires_its_prediction_columns(model_frame: pd.DataFrame) -> None:
    with pytest.raises(DataContractError, match="overlay columns"):
        apply_era_weighted_half_life_8_overlay(pd.DataFrame({"game_id": ["G1"]}), model_frame)


def test_overlay_refuses_a_card_that_is_not_the_uniform_control(model_frame: pd.DataFrame) -> None:
    """If the supplied card's probability is NOT the uniform-weight Gaussian
    read (e.g. built with a drifted configuration or a different smoothing
    method), the refit reproduction check fails and the overlay refuses
    rather than silently comparing against a moved target."""

    target, margin_models = fit_margin_models_for_week(
        model_frame,
        season=_SEASON,
        week=_WEEK,
        regressor="ridge",
        min_train_games=_MIN_TRAIN_GAMES,
        feature_profile=_FEATURE_PROFILE,  # type: ignore[arg-type]
        ridge_alpha=_RIDGE_ALPHA,
        methods=("market_residual",),
    )
    model = margin_models["market_residual"]
    predicted_ecdf = model.predict(target, probability_method="ecdf")
    ecdf_card = target.copy()
    ecdf_card["home_cover_probability"] = predicted_ecdf["home_cover_probability"].to_numpy()

    with pytest.raises(DataContractError, match="does not"):
        apply_era_weighted_half_life_8_overlay(
            ecdf_card,
            model_frame,
            regressor="ridge",
            ridge_alpha=_RIDGE_ALPHA,
            feature_profile=_FEATURE_PROFILE,
            min_train_games=_MIN_TRAIN_GAMES,
        )


def test_overlay_refuses_a_game_missing_from_the_refit_universe(model_frame: pd.DataFrame) -> None:
    card = _week_card(model_frame).copy()
    extra = card.iloc[[0]].copy()
    extra["game_id"] = "not_a_real_game"
    card = pd.concat([card, extra], ignore_index=True)
    with pytest.raises(DataContractError, match="missing games"):
        apply_era_weighted_half_life_8_overlay(
            card,
            model_frame,
            regressor="ridge",
            ridge_alpha=_RIDGE_ALPHA,
            feature_profile=_FEATURE_PROFILE,
            min_train_games=_MIN_TRAIN_GAMES,
        )


def test_overlay_refuses_when_training_pool_is_below_min_train_games(
    model_frame: pd.DataFrame,
) -> None:
    card = _week_card(model_frame)
    with pytest.raises(DataContractError, match="need 100000"):
        apply_era_weighted_half_life_8_overlay(
            card,
            model_frame,
            regressor="ridge",
            ridge_alpha=_RIDGE_ALPHA,
            feature_profile=_FEATURE_PROFILE,
            min_train_games=100_000,
        )


def test_half_life_constant_matches_mod14_selected_arm() -> None:
    """Pins the frozen selection (docs/era_weighting_screen.md) against a
    silent constant drift."""

    assert HALF_LIFE_SEASONS == 8.0


# ---------------------------------------------------------------------------
# 2. overlay_disclosure_note
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_disabled(model_frame: pd.DataFrame) -> None:
    card = _week_card(model_frame)
    disabled = apply_era_weighted_half_life_8_overlay(card, model_frame, enabled=False)
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_formats_a_flip() -> None:
    """A pure formatting check on a hand-built result, independent of whether
    the real fixture happens to produce a flip this run."""

    result = EraWeightedResult(
        overlaid_predictions=pd.DataFrame({"game_id": ["G1"]}),
        flips=(
            EraWeightedFlip(
                game_id="G1",
                matchup="AW1 at HM1",
                from_side="HOME",
                to_side="AWAY",
                baseline_probability=0.51,
                era_weighted_probability=0.49,
            ),
        ),
        enabled=True,
    )
    note = overlay_disclosure_note(result)
    assert "1 pick flipped" in note
    assert "HOME -> AWAY" in note
    assert "not applied to the published card" in note


# ---------------------------------------------------------------------------
# 3. record_era_weighted_half_life_8_challenger_decisions
# ---------------------------------------------------------------------------

_FORECAST_DIR = "2020-week-04-forecast"


def _write_challenger_registry(
    artifacts: Path,
    *,
    status: str = "ACTIVE_PROSPECTIVE",
    ridge_alpha: float = _RIDGE_ALPHA,
    feature_table_name: str = _FEATURE_TABLE_NAME,
) -> None:
    model_config = {
        "method": "market_residual",
        "target": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": ridge_alpha,
        "calibration_method": "none",
        "feature_profile": _FEATURE_PROFILE,
        "min_edge": 0.02,
        "min_train_games": _MIN_TRAIN_GAMES,
        "feature_table": feature_table_name,
    }
    write_challenger_registry(
        artifacts,
        challenger_id=CHALLENGER_ID,
        model_config=model_config,
        status=status,
    )


def _write_active_model_and_card(
    artifacts: Path,
    data_root: Path,
    model_frame: pd.DataFrame,
    *,
    ridge_alpha: float = _RIDGE_ALPHA,
    write_features: bool = True,
) -> Path:
    processed = data_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    feature_path = processed / _FEATURE_TABLE_NAME
    if write_features:
        model_frame.to_parquet(feature_path)

    card = _week_card(model_frame, ridge_alpha=ridge_alpha).copy()
    card["kickoff"] = "2026-09-10T17:00:00+00:00"

    write_active_model_and_card(
        artifacts,
        season=_SEASON,
        week=_WEEK,
        created_at_utc="2026-09-08T15:00:00+00:00",
        forecast_dir=_FORECAST_DIR,
        ridge_alpha=ridge_alpha,
        feature_profile=_FEATURE_PROFILE,
        probability_method="gaussian",
        min_train_games=_MIN_TRAIN_GAMES,
        feature_table_path=str(feature_path),
        recommendations=card,
    )
    return feature_path


def test_record_challenger_decisions_records_the_reweighted_arm(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    artifacts = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_challenger_registry(artifacts)
    _write_active_model_and_card(artifacts, data_root, model_frame)
    now = datetime(2026, 9, 8, 16, 0, tzinfo=UTC)

    result = record_era_weighted_half_life_8_challenger_decisions(artifacts, data_root, now=now)

    expected_games = len(_week_card(model_frame))
    assert result["recorded"] == expected_games
    assert result["challenger_id"] == CHALLENGER_ID

    ledger = load_challenger_decisions(artifacts)
    assert list(ledger.columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()
    assert result["flip_count"] == len(result["flipped_game_ids"])

    # Re-running is a no-op: append-only, never rewrites.
    again = record_era_weighted_half_life_8_challenger_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == expected_games


def test_record_challenger_refuses_outside_recording_lock_window(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    artifacts = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_challenger_registry(artifacts)
    _write_active_model_and_card(artifacts, data_root, model_frame)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_era_weighted_half_life_8_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 18, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_challenger_refuses_a_fingerprint_mismatch(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    artifacts = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_challenger_registry(artifacts, ridge_alpha=_RIDGE_ALPHA)
    # The active model's OWN configuration moved (a promotion) since this
    # challenger was pinned -- recording must refuse, not silently switch
    # base models under the same challenger id.
    _write_active_model_and_card(artifacts, data_root, model_frame, ridge_alpha=1.0)

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_era_weighted_half_life_8_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_challenger_refuses_an_inactive_registration(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    artifacts = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_challenger_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts, data_root, model_frame)

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_era_weighted_half_life_8_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
        )


def test_record_challenger_refuses_a_missing_feature_table(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    artifacts = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_challenger_registry(artifacts)
    _write_active_model_and_card(artifacts, data_root, model_frame, write_features=False)

    with pytest.raises(ValueError, match="not built yet"):
        record_era_weighted_half_life_8_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
        )
