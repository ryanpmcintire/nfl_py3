"""SIM-03: centered lineup uncertainty with production Gaussian residual noise."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import ndtr

from nfl_ats.data import DataContractError
from nfl_ats.expected_lineup_loss_features import (
    EXPECTED_LINEUP_LOSS_COLUMNS,
    _lineup_group,
    _visible_panel,
)

SEED = 2026090521
SCENARIOS = 200
GROUPS = ("offense", "defense", "qb")
Array = npt.NDArray[np.float64]


def lineup_draws(
    players: pd.DataFrame, *, seed: int = SEED, scenarios: int = SCENARIOS
) -> dict[tuple[int, int, str], dict[str, Array]]:
    """Sample all visible players, aggregate LEAD-62 starters; outcomes only in oracle.

    Input probabilities/history must be produced by the existing as-of helpers.
    Precise depth observations after the decision are rejected by starter selection.
    Stable player ordering gives reproducibility independent of input row order.
    """
    if scenarios < 1:
        raise ValueError("scenarios must be positive")
    # Select visibility using the existing LEAD-62 contract even for bench rows.
    visible = _visible_panel(players)
    visible["lineup_group"] = _lineup_group(visible.position, visible.position_group)
    output = {}
    for key, rows in visible.groupby(["season", "week", "team"], sort=True):
        rows = rows.sort_values("gsis_id")
        p = rows.play_probability.to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise DataContractError("Every sampled player requires a finite probability in [0,1]")
        # Team-specific seed avoids changing draws when unrelated teams are added.
        digest = hashlib.sha256(str(key).encode()).digest()
        rng = np.random.default_rng(seed + int.from_bytes(digest[:4], "little"))
        uniforms = rng.random((scenarios, len(rows)))
        weights = np.column_stack(
            [
                rows.trailing4_snap_share.fillna(0).to_numpy(dtype=float)
                * rows.depth_rank.eq(1).to_numpy()
                * rows.lineup_group.eq(group).to_numpy()
                for group in GROUPS
            ]
        )
        expected = (1 - p) @ weights
        shuffled = rng.permutation(p)
        values = {
            "expected": expected,
            "mixture": (uniforms >= p) @ weights,
            "permutation": (uniforms >= shuffled) @ weights,
        }
        if "played" in rows:
            actual = rows.played.to_numpy(dtype=float)
            if not np.isfinite(actual).all() or not np.isin(actual, [0, 1]).all():
                raise DataContractError("Oracle requires observed binary participation labels")
            values["oracle"] = np.broadcast_to((1 - actual) @ weights, (scenarios, 3)).copy()
        output[(int(str(key[0])), int(str(key[1])), str(key[2]))] = values
    return output


def loss_coefficients(model: Any, row: pd.DataFrame) -> Array:
    """Recover original-unit coefficients through the fitted ridge pipeline."""
    base = row.iloc[:1].copy()
    base[list(EXPECTED_LINEUP_LOSS_COLUMNS)] = 0.0
    center = model._predicted_margin(base, model._spread(base))[0][0]
    values = []
    for column in EXPECTED_LINEUP_LOSS_COLUMNS:
        bump = base.copy()
        bump[column] = 1.0
        values.append(model._predicted_margin(bump, model._spread(bump))[0][0] - center)
    return np.asarray(values, dtype=float)


def mixture_probability(
    center: float, line: float, residual_mean: float, sigma: float, deltas: Array
) -> dict[str, float]:
    """Integrate game noise conditional on each sampled lineup margin shift."""
    if sigma <= 0 or not np.isfinite([center, line, residual_mean, sigma]).all():
        raise ValueError("Finite centers and positive finite residual sigma required")
    if deltas.size == 0 or not np.isfinite(deltas).all():
        raise ValueError("Finite nonempty scenario deltas required")
    sd = float(np.std(deltas))
    return {
        "probability": float(ndtr((center + deltas + residual_mean - line) / sigma).mean()),
        "center_cover_share": float(np.mean(center + deltas > line)),
        "scenario_sd": sd,
        "total_sd_increase": float(np.hypot(sigma, sd) - sigma),
    }


def paired_summary(frame: pd.DataFrame, column: str, *, draws: int = 20_000) -> dict[str, Any]:
    """Positive differences favour the mixture; resample whole season-week blocks."""
    blocks = frame.groupby(["season", "week"])[column].agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    samples = []
    for start in range(0, draws, 1000):
        indices = rng.integers(0, len(blocks), size=(min(1000, draws - start), len(blocks)))
        samples.extend(
            (
                blocks["sum"].to_numpy()[indices].sum(axis=1)
                / blocks["count"].to_numpy()[indices].sum(axis=1)
            ).tolist()
        )
    return {
        "effect": float(frame[column].mean()),
        "interval_low": float(np.quantile(samples, 0.025)),
        "interval_high": float(np.quantile(samples, 0.975)),
        "standard_error": float(np.std(samples, ddof=1)),
        "probability_positive": float(np.mean(np.asarray(samples) > 0)),
        "sample_games": len(frame),
        "sample_blocks": len(blocks),
        "draws": draws,
    }
