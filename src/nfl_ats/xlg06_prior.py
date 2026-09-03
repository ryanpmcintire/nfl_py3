"""Frozen rookie-prior spec for XLG-06 Stage 3.

Maps a drafted skill player's pre-draft recruiting rating to an expected
rookie production rate, with a fixed exposure-decay schedule blending the
prior with observed NFL production as snaps accumulate::

    mu(r)    = a + b * r
    w(s)     = N0 / (N0 + s)
    prior(s) = w(s) * mu(r) + (1 - w(s)) * observed_avg

``(a, b)`` are OLS coefficients fit once on the Stage-2 per-player table;
``N0`` is a fixed placeholder constant (see
``docs/xlg06_stage3_prior_spec.md``). This module fits parameters and
evaluates the formula. It wires no feature, scores no ATS outcome, and
spends no registry window.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError, require_columns

RATING_COLUMN = "rating_num"
EPA_COLUMN = "rookie_epa"
WEEKS_COLUMN = "rookie_reg_weeks"
YEAR_COLUMN = "recruit_year_num"

#: Columns the fitter is allowed to read. A synthetic post-dated predictor
#: can never silently substitute for the pre-draft rating.
FIT_ALLOWLIST = frozenset(
    {
        RATING_COLUMN,
        EPA_COLUMN,
        WEEKS_COLUMN,
        YEAR_COLUMN,
        "gsis_id",
        "position",
        "year",
        "rookie_season",
        "rookie_epa_per_game",
    }
)


def prepare_fit_frame(rookie_epa: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and build the per-game rate; fail closed on gaps."""

    unknown = set(rookie_epa.columns) - FIT_ALLOWLIST
    if unknown:
        raise DataContractError(f"prior fit refuses unexpected columns: {sorted(unknown)}")
    require_columns(rookie_epa, (RATING_COLUMN, EPA_COLUMN, WEEKS_COLUMN), "XLG-06 prior fit")
    frame = rookie_epa.copy()
    for column in (RATING_COLUMN, EPA_COLUMN, WEEKS_COLUMN):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame[RATING_COLUMN].notna() & frame[EPA_COLUMN].notna() & frame[WEEKS_COLUMN].gt(0)
    ].copy()
    if frame.empty:
        raise DataContractError("prior fit has no usable rows after eligibility")
    frame["rookie_epa_per_game"] = frame[EPA_COLUMN] / frame[WEEKS_COLUMN]
    return frame.reset_index(drop=True)


def fit_rating_map(frame: pd.DataFrame) -> dict[str, float]:
    """OLS coefficients of rookie EPA/game on recruiting rating."""

    prepared = prepare_fit_frame(frame)
    x = prepared[RATING_COLUMN].to_numpy(dtype=float)
    y = prepared["rookie_epa_per_game"].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    r_squared = 1.0 - float(np.sum((y - fitted) ** 2)) / float(np.sum((y - np.mean(y)) ** 2))
    return {
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "r_squared": float(r_squared),
        "n": len(prepared),
    }


def bootstrap_slope_ci(
    frame: pd.DataFrame, *, seed: int, samples: int, block_column: str = "year"
) -> dict[str, Any]:
    """Cohort-blocked percentile intervals for the OLS slope/intercept."""

    prepared = prepare_fit_frame(frame)
    if block_column not in prepared.columns:
        raise DataContractError(f"prior bootstrap needs block column {block_column!r}")
    x = prepared[RATING_COLUMN].to_numpy(dtype=float)
    y = prepared["rookie_epa_per_game"].to_numpy(dtype=float)
    blocks = prepared[block_column].astype(str).to_numpy()
    unique_blocks = np.unique(blocks)
    block_rows = {block: np.flatnonzero(blocks == block) for block in unique_blocks}
    rng = np.random.default_rng(seed)
    slopes = np.empty(samples)
    intercepts = np.empty(samples)
    for draw in range(samples):
        chosen = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        sample = np.concatenate([block_rows[block] for block in chosen])
        design = np.column_stack([np.ones_like(x[sample]), x[sample]])
        coefficients, _, _, _ = np.linalg.lstsq(design, y[sample], rcond=None)
        intercepts[draw], slopes[draw] = float(coefficients[0]), float(coefficients[1])
    return {
        "slope_ci95": [float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))],
        "intercept_ci95": [
            float(np.quantile(intercepts, 0.025)),
            float(np.quantile(intercepts, 0.975)),
        ],
        "samples": samples,
        "seed": seed,
        "blocks": len(unique_blocks),
    }


def prior_mean(rating: float, *, intercept: float, slope: float) -> float:
    """Expected rookie EPA/game for a rating under frozen parameters."""

    return intercept + slope * rating


def prior_weight(snaps: float, *, n0: float) -> float:
    """Prior weight w(s) = N0 / (N0 + s); s = 0 recovers the pure prior."""

    if n0 <= 0:
        raise DataContractError(f"decay constant N0 must be positive, got {n0}")
    if snaps < 0:
        raise DataContractError(f"accumulated snaps cannot be negative, got {snaps}")
    return n0 / (n0 + snaps)


def blend_prior(
    rating: float,
    observed_avg: float,
    snaps: float,
    *,
    intercept: float,
    slope: float,
    n0: float,
) -> float:
    """Full prior: weight the rating-implied mean against observed average."""

    weight = prior_weight(snaps, n0=n0)
    return (
        weight * prior_mean(rating, intercept=intercept, slope=slope)
        + (1.0 - weight) * observed_avg
    )


def weight_curve(n0_values: list[float], snaps_grid: list[float]) -> dict[str, list[float]]:
    """Sensitivity appendix: weight curves for several N0; selects nothing."""

    return {f"N0={n0:g}": [prior_weight(snaps, n0=n0) for snaps in snaps_grid] for n0 in n0_values}
