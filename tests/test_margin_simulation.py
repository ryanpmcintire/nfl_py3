from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.margin import MarginModel
from nfl_ats.margin_simulation import simulate_margin_distribution
from nfl_ats.outcomes import fit_margin_models_for_week


def _model(residuals: list[float]) -> MarginModel:
    return MarginModel(
        estimator=None,
        residuals=np.asarray(residuals, dtype=np.float64),
        model_name="market",
        ridge_alpha=None,
        target="market",
        feature_columns=(),
        training_rows=100,
        distribution_rows=len(residuals),
        training_max_gameday="2025-12-31",
    )


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["integer", "half"],
            "gameday": ["2026-09-10", "2026-09-13"],
            "spread_line": [3.0, 3.5],
        }
    )


def test_simulation_is_reproducible_and_preserves_auditable_draws() -> None:
    model = _model([-2.0, 0.0, 1.0, 4.0])
    first = simulate_margin_distribution(model, _games(), samples=500, seed=20260902)
    second = simulate_margin_distribution(model, _games(), samples=500, seed=20260902)

    assert np.array_equal(first.latent_margins, second.latent_margins)
    assert np.array_equal(first.settled_margins, second.settled_margins)
    assert first.probabilities.equals(second.probabilities)
    samples = first.sample_frame()
    assert len(samples) == 1_000
    assert samples.groupby("game_id")["simulation_id"].nunique().eq(500).all()
    assert np.array_equal(samples["settled_margin"], np.rint(samples["latent_margin"]))


def test_simulation_derives_three_way_ats_probabilities_from_integer_margins() -> None:
    # The market model centers each game on its quoted line. These four
    # residuals therefore imply one cover, two pushes and one loss at line 3.
    model = _model([-1.0, 0.0, 0.0, 1.0])
    game = _games().iloc[[0]]
    result = simulate_margin_distribution(model, game, samples=80_000, seed=7)
    row = result.probabilities.iloc[0]

    assert row["home_cover_probability"] == pytest.approx(0.25, abs=0.01)
    assert row["home_cover_probability_excluding_push"] == pytest.approx(0.25, abs=0.01)
    assert row["push_probability"] == pytest.approx(0.50, abs=0.01)
    assert row["home_loss_probability"] == pytest.approx(0.25, abs=0.01)
    assert row["home_cover_probability_conditional_on_no_push"] == pytest.approx(0.5, abs=0.02)
    assert (
        row["home_cover_probability_excluding_push"]
        + row["push_probability"]
        + row["home_loss_probability"]
    ) == pytest.approx(1.0)


def test_half_point_line_has_zero_push_probability() -> None:
    result = simulate_margin_distribution(
        _model([-1.0, 0.0, 1.0]), _games().iloc[[1]], samples=5_000, seed=9
    )
    assert result.probabilities.iloc[0]["push_probability"] == 0.0


@pytest.mark.parametrize("samples", [0, -1, True, 1.5])
def test_simulation_rejects_invalid_sample_counts(samples: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        simulate_margin_distribution(_model([0.0]), _games(), samples=samples)  # type: ignore[arg-type]


def test_simulation_refuses_targets_at_or_before_training_cutoff() -> None:
    games = _games()
    games.loc[0, "gameday"] = "2025-12-31"
    with pytest.raises(DataContractError, match="strictly after"):
        simulate_margin_distribution(_model([0.0]), games, samples=10)


def test_simulation_requires_unique_games_and_finite_residuals() -> None:
    duplicate = _games().copy()
    duplicate.loc[1, "game_id"] = duplicate.loc[0, "game_id"]
    with pytest.raises(DataContractError, match="unique"):
        simulate_margin_distribution(_model([0.0]), duplicate, samples=10)
    with pytest.raises(ValueError, match="finite"):
        simulate_margin_distribution(_model([float("nan")]), _games(), samples=10)

    non_finite_center = _games()
    non_finite_center.loc[0, "spread_line"] = float("inf")
    with pytest.raises((DataContractError, ValueError), match="finite"):
        simulate_margin_distribution(_model([0.0]), non_finite_center, samples=10)


def test_simulation_integrates_with_leak_safe_weekly_fitter(model_frame: pd.DataFrame) -> None:
    target, models = fit_margin_models_for_week(
        model_frame,
        season=2020,
        week=1,
        min_train_games=80,
        methods=("market_residual",),
    )
    result = simulate_margin_distribution(models["market_residual"], target, samples=100, seed=11)

    assert set(result.probabilities["game_id"]) == set(target["game_id"])
    assert pd.to_datetime(result.probabilities["gameday"]).min() > pd.Timestamp(
        result.probabilities["training_max_gameday"].iloc[0]
    )
    assert np.allclose(
        result.probabilities[
            [
                "home_cover_probability_excluding_push",
                "push_probability",
                "home_loss_probability",
            ]
        ].sum(axis=1),
        1.0,
    )
