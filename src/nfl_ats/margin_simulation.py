"""Monte Carlo sampling from a fitted margin model's predictive distribution."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.margin import MarginModel


@dataclass(frozen=True)
class MarginMonteCarloResult:
    """Per-game ATS probabilities and the samples that produced them."""

    probabilities: pd.DataFrame
    latent_margins: npt.NDArray[np.float64]
    settled_margins: npt.NDArray[np.int64]
    samples: int
    seed: int

    def sample_frame(self) -> pd.DataFrame:
        """Return one audit row per game and simulation draw."""

        game_ids = self.probabilities["game_id"].to_numpy()
        return pd.DataFrame(
            {
                "game_id": np.repeat(game_ids, self.samples),
                "simulation_id": np.tile(np.arange(self.samples, dtype=np.int64), len(game_ids)),
                "latent_margin": self.latent_margins.ravel(),
                "settled_margin": self.settled_margins.ravel(),
            }
        )


def _validate_future_frame(model: MarginModel, frame: pd.DataFrame) -> None:
    required = {"game_id", "gameday", "spread_line"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"Margin simulation is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Margin simulation requires at least one game")
    if frame["game_id"].isna().any() or frame["game_id"].duplicated().any():
        raise DataContractError("Margin simulation requires unique, non-null game_id values")

    gamedays = pd.to_datetime(frame["gameday"], errors="coerce")
    if gamedays.isna().any():
        raise DataContractError("Margin simulation requires a valid gameday for every game")
    training_cutoff = pd.Timestamp(model.training_max_gameday)
    if gamedays.min().normalize() <= training_cutoff.normalize():
        raise DataContractError(
            "Margin simulation is leak-safe only when every target gameday is strictly "
            "after the model training cutoff"
        )


def simulate_margin_distribution(
    model: MarginModel,
    frame: pd.DataFrame,
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> MarginMonteCarloResult:
    """Sample predictive margins and derive three-way ATS probabilities.

    Each game independently resamples the fitted model's stored residual
    distribution with replacement and adds its predicted margin center. For
    fair-margin and market-residual models,
    :func:`nfl_ats.margin.fit_margin_model` builds that distribution on an
    out-of-time calibration partition. Integer lines use integer-rounded
    margins for the three-way cover/push/loss split, matching
    :meth:`MarginModel.predict`; the smoothed two-way probability retains that
    method's continuous ECDF convention. The target frame must occur strictly
    after ``model.training_max_gameday``; callers that need a weekly fitted
    model should use
    :func:`nfl_ats.outcomes.fit_margin_models_for_week`.

    ``home_cover_probability`` follows :meth:`MarginModel.predict`'s smoothed
    two-way ECDF convention. The separate
    ``home_cover_probability_excluding_push``, ``push_probability`` and
    ``home_loss_probability`` columns follow its integer-line three-way
    settlement convention and sum to one.
    """

    if isinstance(samples, bool):
        raise ValueError("samples must be a positive integer")
    try:
        sample_count = index(samples)
    except TypeError as error:
        raise ValueError("samples must be a positive integer") from error
    if sample_count <= 0:
        raise ValueError("samples must be a positive integer")
    _validate_future_frame(model, frame)

    residuals = np.asarray(model.residuals, dtype=np.float64)
    if residuals.ndim != 1 or residuals.size == 0 or not np.isfinite(residuals).all():
        raise ValueError("Margin simulation requires a finite, one-dimensional residual sample")

    predictive_distribution = model.distribution(frame)
    if not np.isfinite(predictive_distribution).all():
        raise ValueError("Margin simulation produced a non-finite predictive distribution")
    centers = predictive_distribution[:, 0] - residuals[0]
    rng = np.random.default_rng(seed)
    sampled_indexes = rng.integers(
        0,
        residuals.size,
        size=(len(frame), sample_count),
        dtype=np.int64,
    )
    latent_margins = centers[:, np.newaxis] + residuals[sampled_indexes]
    settled_margins = np.rint(latent_margins).astype(np.int64)

    lines = pd.to_numeric(frame["spread_line"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(lines).all():
        raise DataContractError("Margin simulation requires a finite spread_line for every game")
    line_grid = lines[:, np.newaxis]
    integer_line = np.isclose(np.mod(lines, 1.0), 0.0, atol=1e-9)
    two_way_home_cover = latent_margins > line_grid
    home_cover = np.where(
        integer_line[:, np.newaxis], settled_margins > line_grid, two_way_home_cover
    )
    push = integer_line[:, np.newaxis] & (settled_margins == line_grid)
    away_cover = ~(home_cover | push)

    two_way_successes = two_way_home_cover.sum(axis=1)
    home_probability = (two_way_successes + 0.5) / (sample_count + 1.0)
    home_probability_excluding_push = home_cover.mean(axis=1)
    push_probability = push.mean(axis=1)
    away_probability = away_cover.mean(axis=1)
    non_push_probability = home_probability_excluding_push + away_probability
    conditional_home = np.divide(
        home_probability_excluding_push,
        non_push_probability,
        out=np.full_like(home_probability_excluding_push, 0.5),
        where=non_push_probability > 0.0,
    )
    pick_home = home_probability >= 0.5

    probabilities = pd.DataFrame(
        {
            "game_id": frame["game_id"].to_numpy(),
            "gameday": pd.to_datetime(frame["gameday"], errors="raise").to_numpy(),
            "spread_line": lines,
            "predicted_margin": centers,
            "home_cover_probability": home_probability,
            "home_cover_probability_excluding_push": home_probability_excluding_push,
            "push_probability": push_probability,
            "home_loss_probability": away_probability,
            "home_cover_probability_conditional_on_no_push": conditional_home,
            "pick_side": np.where(pick_home, "HOME", "AWAY"),
            "pick_probability": np.where(pick_home, home_probability, 1.0 - home_probability),
            "monte_carlo_standard_error": np.sqrt(
                home_probability * (1.0 - home_probability) / sample_count
            ),
            "samples": sample_count,
            "seed": seed,
            "training_max_gameday": model.training_max_gameday,
        },
        index=frame.index,
    )
    return MarginMonteCarloResult(
        probabilities=probabilities,
        latent_margins=np.asarray(latent_margins, dtype=np.float64),
        settled_margins=settled_margins,
        samples=sample_count,
        seed=seed,
    )
