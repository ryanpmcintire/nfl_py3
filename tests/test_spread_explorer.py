"""Tests for ``nfl_ats.spread_explorer`` -- the spread-explorer library.

Owner request, 2026-08-20: "pick a spread for a game and see the odds of
covering." Three things are load-bearing here, mirroring
``tests/test_smooth_cdf_mapping_overlay.py``'s structure since this module
follows the exact same refit-and-verify discipline:

1. :func:`compute_spread_explorer_params` reproduces the production Gaussian
   probability from a refit before trusting anything -- proving it reads the
   SAME out-of-time residual sample the card was built from -- and refuses
   (``DataContractError``) rather than silently comparing against a moved
   target when the supplied probability does not reproduce.
2. :func:`widget_home_cover_probability` (the Abramowitz-Stegun erf
   approximation the browser widget also evaluates) tracks the
   production-precision scipy-based formula
   (``nfl_ats.calibration.smoothed_home_cover_probability``) tightly, and is
   monotonic in the expected direction as the hypothetical line moves.
3. :func:`load_feature_table_for_forecast` resolves the recorded absolute
   path first, falls back to ``data_root/processed/<name>``, and raises a
   clear error when neither exists.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.outcomes import fit_margin_models_for_week
from nfl_ats.spread_explorer import (
    SPREAD_EXPLORER_MAX_LINE,
    SPREAD_EXPLORER_MIN_LINE,
    SPREAD_EXPLORER_STEP,
    compute_spread_explorer_params,
    load_feature_table_for_forecast,
    spread_explorer_payload,
    widget_home_cover_probability,
)

_FEATURE_PROFILE = "base"
_RIDGE_ALPHA = 10.0
_MIN_TRAIN_GAMES = 100
_SEASON = 2020
_WEEK = 4


def _week_card(model_frame: pd.DataFrame) -> pd.DataFrame:
    """A real card, built the same way ``compute_spread_explorer_params``
    itself refits -- via ``fit_margin_models_for_week``, never a hand-typed
    probability."""

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
    predicted = model.predict(target, probability_method="gaussian")
    card = target.copy()
    card["home_cover_probability"] = predicted["home_cover_probability"].to_numpy()
    assert not card.empty
    return card


# ---------------------------------------------------------------------------
# 1. compute_spread_explorer_params
# ---------------------------------------------------------------------------


def test_compute_params_reproduces_the_gaussian_control(model_frame: pd.DataFrame) -> None:
    card = _week_card(model_frame)
    params = compute_spread_explorer_params(
        card,
        model_frame,
        regressor="ridge",
        ridge_alpha=_RIDGE_ALPHA,
        feature_profile=_FEATURE_PROFILE,
        min_train_games=_MIN_TRAIN_GAMES,
    )
    assert set(params) == set(card["game_id"].astype(str))
    for _, row in card.iterrows():
        p = params[str(row["game_id"])]
        assert p.home_team == row["home_team"]
        assert p.away_team == row["away_team"]
        assert p.card_line == pytest.approx(float(row["spread_line"]))
        assert p.card_home_cover_probability == pytest.approx(
            float(row["home_cover_probability"]), abs=1e-9
        )
        # compute_spread_explorer_params already raises if a refit Gaussian
        # probability fails to reproduce the supplied card (see the "refuses
        # a drifted probability" test below); reaching this point without an
        # exception IS that proof. The next test additionally re-derives the
        # scipy-precision probability from the returned params directly.


def test_compute_params_matches_scipy_precision_gaussian(model_frame: pd.DataFrame) -> None:
    """The returned (center, mean, std) reproduce the card's own probability
    through the EXACT production formula (scipy-based), not an approximation."""

    card = _week_card(model_frame)
    params = compute_spread_explorer_params(
        card,
        model_frame,
        regressor="ridge",
        ridge_alpha=_RIDGE_ALPHA,
        feature_profile=_FEATURE_PROFILE,
        min_train_games=_MIN_TRAIN_GAMES,
    )
    from scipy import stats

    for _, row in card.iterrows():
        p = params[str(row["game_id"])]
        threshold = p.card_line - p.center
        production = float(stats.norm.sf(threshold, loc=p.residual_mean, scale=p.residual_std))
        assert production == pytest.approx(p.card_home_cover_probability, abs=1e-9)


def test_compute_params_empty_predictions_returns_empty(model_frame: pd.DataFrame) -> None:
    empty = pd.DataFrame(
        columns=[
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "spread_line",
            "home_cover_probability",
        ]
    )
    assert (
        compute_spread_explorer_params(
            empty,
            model_frame,
            regressor="ridge",
            ridge_alpha=_RIDGE_ALPHA,
            feature_profile=_FEATURE_PROFILE,
            min_train_games=_MIN_TRAIN_GAMES,
        )
        == {}
    )


def test_compute_params_requires_its_columns(model_frame: pd.DataFrame) -> None:
    with pytest.raises(DataContractError, match="spread-explorer columns"):
        compute_spread_explorer_params(
            pd.DataFrame({"game_id": ["G1"]}),
            model_frame,
            regressor="ridge",
            ridge_alpha=_RIDGE_ALPHA,
            feature_profile=_FEATURE_PROFILE,
            min_train_games=_MIN_TRAIN_GAMES,
        )


def test_compute_params_refuses_a_drifted_probability(model_frame: pd.DataFrame) -> None:
    card = _week_card(model_frame).copy()
    card.iloc[0, card.columns.get_loc("home_cover_probability")] = 0.999999
    with pytest.raises(DataContractError, match="do not"):
        compute_spread_explorer_params(
            card,
            model_frame,
            regressor="ridge",
            ridge_alpha=_RIDGE_ALPHA,
            feature_profile=_FEATURE_PROFILE,
            min_train_games=_MIN_TRAIN_GAMES,
        )


def test_compute_params_refuses_a_game_missing_from_the_refit_universe(
    model_frame: pd.DataFrame,
) -> None:
    card = _week_card(model_frame).copy()
    extra = card.iloc[[0]].copy()
    extra["game_id"] = "not_a_real_game"
    card = pd.concat([card, extra], ignore_index=True)
    with pytest.raises(DataContractError, match="missing games"):
        compute_spread_explorer_params(
            card,
            model_frame,
            regressor="ridge",
            ridge_alpha=_RIDGE_ALPHA,
            feature_profile=_FEATURE_PROFILE,
            min_train_games=_MIN_TRAIN_GAMES,
        )


# ---------------------------------------------------------------------------
# 2. widget_home_cover_probability (the browser-mirrored erf approximation)
# ---------------------------------------------------------------------------


def test_widget_formula_tracks_scipy_closely(model_frame: pd.DataFrame) -> None:
    from scipy import stats

    card = _week_card(model_frame)
    params = compute_spread_explorer_params(
        card,
        model_frame,
        regressor="ridge",
        ridge_alpha=_RIDGE_ALPHA,
        feature_profile=_FEATURE_PROFILE,
        min_train_games=_MIN_TRAIN_GAMES,
    )
    for p in params.values():
        for offset in (-6.0, -1.0, 0.0, 1.0, 6.0):
            line = p.card_line + offset
            widget = widget_home_cover_probability(line, p.center, p.residual_mean, p.residual_std)
            production = float(
                stats.norm.sf(line - p.center, loc=p.residual_mean, scale=p.residual_std)
            )
            assert widget == pytest.approx(production, abs=1e-5)


def test_widget_formula_is_monotonically_decreasing_in_the_line() -> None:
    """Per this codebase's spread_line convention (a MORE positive home
    spread means the home team is a BIGGER favorite -- see
    ``public_board.spread_words``: ``home_spread > 0`` -> home favored),
    a higher hypothetical line makes it harder, not easier, for home to
    cover, so probability strictly decreases as the line increases."""

    center, mean, std = 1.5, 0.5, 12.0
    lines = np.arange(-20.0, 20.01, 0.5)
    probabilities = [widget_home_cover_probability(line, center, mean, std) for line in lines]
    assert all(a > b for a, b in itertools.pairwise(probabilities))


def test_widget_formula_pick_em_is_near_half_when_residual_mean_is_near_zero() -> None:
    p = widget_home_cover_probability(0.0, 0.0, 0.0, 12.0)
    assert p == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. spread_explorer_payload
# ---------------------------------------------------------------------------


def test_payload_is_json_serializable_and_rounded(model_frame: pd.DataFrame) -> None:
    card = _week_card(model_frame)
    params = compute_spread_explorer_params(
        card,
        model_frame,
        regressor="ridge",
        ridge_alpha=_RIDGE_ALPHA,
        feature_profile=_FEATURE_PROFILE,
        min_train_games=_MIN_TRAIN_GAMES,
    )
    payload = spread_explorer_payload(params)
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert set(decoded) == set(params)
    for game_id, p in params.items():
        row = decoded[game_id]
        assert row["home"] == p.home_team
        assert row["away"] == p.away_team
        assert row["center"] == pytest.approx(p.center, abs=1e-6)
        assert row["mean"] == pytest.approx(p.residual_mean, abs=1e-6)
        assert row["std"] == pytest.approx(p.residual_std, abs=1e-6)
        assert row["line"] == pytest.approx(p.card_line, abs=1e-3)


def test_payload_of_empty_params_is_empty() -> None:
    assert spread_explorer_payload({}) == {}


# ---------------------------------------------------------------------------
# 4. load_feature_table_for_forecast
# ---------------------------------------------------------------------------


def test_load_feature_table_prefers_the_recorded_absolute_path(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    feature_path = tmp_path / "features.parquet"
    model_frame.to_parquet(feature_path)
    metadata = {"provenance": {"feature_table": {"path": str(feature_path)}}}
    loaded = load_feature_table_for_forecast(metadata, tmp_path / "unused_data_root")
    assert len(loaded) == len(model_frame)


def test_load_feature_table_falls_back_to_data_root_processed(
    tmp_path: Path, model_frame: pd.DataFrame
) -> None:
    data_root = tmp_path / "data"
    processed = data_root / "processed"
    processed.mkdir(parents=True)
    feature_path = processed / "game_features_weak_stack.parquet"
    model_frame.to_parquet(feature_path)
    # The recorded path is from a DIFFERENT machine and does not exist here.
    metadata = {
        "provenance": {
            "feature_table": {"path": "C:\\some\\other\\machine\\game_features_weak_stack.parquet"}
        }
    }
    loaded = load_feature_table_for_forecast(metadata, data_root)
    assert len(loaded) == len(model_frame)


def test_load_feature_table_raises_when_neither_path_exists(tmp_path: Path) -> None:
    metadata = {"provenance": {"feature_table": {"path": "nowhere.parquet"}}}
    with pytest.raises(DataContractError, match="not available locally"):
        load_feature_table_for_forecast(metadata, tmp_path / "data")


def test_load_feature_table_raises_without_a_recorded_path(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="no feature table path"):
        load_feature_table_for_forecast({}, tmp_path / "data")


# ---------------------------------------------------------------------------
# 5. Constants (declared range/step)
# ---------------------------------------------------------------------------


def test_slider_constants_match_the_declared_spec() -> None:
    assert SPREAD_EXPLORER_MIN_LINE == -20.0
    assert SPREAD_EXPLORER_MAX_LINE == 20.0
    assert SPREAD_EXPLORER_STEP == 0.5
    assert math.isclose(
        (SPREAD_EXPLORER_MAX_LINE - SPREAD_EXPLORER_MIN_LINE) % SPREAD_EXPLORER_STEP, 0.0
    )
