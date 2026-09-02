"""Candidate-specific probability uncertainty for paper sizing.

This module transforms a caller's uncertainty statement into a conservative
probability for the side already selected. It never chooses or flips a side.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

AUDIT_COLUMNS = (
    "raw_bet_probability",
    "conservative_bet_probability",
    "probability_uncertainty_method",
    "supplied_probability_lower_bound",
    "supplied_posterior_sd",
    "posterior_z",
    "effective_probability_haircut",
)


def conservative_probability_audit(
    candidates: pd.DataFrame,
    *,
    probability_haircut: float = 0.0,
    probability_uncertainty: pd.DataFrame | None = None,
    posterior_z: float = 1.645,
) -> pd.DataFrame:
    """Return one sizing-probability audit row per candidate.

    ``probability_uncertainty`` is indexed by active ``game_id`` and carries a
    matching ``bet_side``. Each row supplies exactly one of
    ``probability_lower_bound`` (already expressed for that selected side) or
    ``posterior_sd`` (converted as point probability minus ``posterior_z``
    standard deviations). Values are floored at 0.5 for sizing, which produces
    zero Kelly exposure when uncertainty removes the estimated edge.
    """

    required = {"bet_side", "home_cover_probability"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Candidates are missing probability columns: {', '.join(missing)}")
    if not math.isfinite(probability_haircut) or not 0.0 <= probability_haircut < 0.5:
        raise ValueError("probability_haircut must be finite and in [0, 0.5)")
    if not math.isfinite(posterior_z) or posterior_z <= 0.0:
        raise ValueError("posterior_z must be finite and positive")
    if probability_uncertainty is not None and probability_haircut != 0.0:
        raise ValueError(
            "probability_haircut and probability_uncertainty cannot be combined; "
            "provide one auditable uncertainty rule"
        )

    sides = candidates["bet_side"].astype(str)
    invalid_sides = sorted(set(sides).difference({"HOME", "AWAY", "PASS"}))
    if invalid_sides:
        raise ValueError("bet_side must be HOME, AWAY, or PASS")
    active = sides.ne("PASS")
    home_probability = pd.to_numeric(candidates["home_cover_probability"], errors="coerce")
    active_home = home_probability.loc[active]
    if active_home.isna().any() or not np.isfinite(active_home).all():
        raise ValueError("Active home_cover_probability values must be finite numbers")
    if ((active_home < 0.0) | (active_home > 1.0)).any():
        raise ValueError("Active home_cover_probability values must lie in [0, 1]")
    raw = pd.Series(0.0, index=candidates.index, dtype=float)
    raw.loc[active] = np.where(sides.loc[active].eq("HOME"), active_home, 1.0 - active_home)
    conservative = raw.copy()
    method = pd.Series("PASS", index=candidates.index, dtype="object")
    lower_audit = pd.Series(np.nan, index=candidates.index, dtype=float)
    sd_audit = pd.Series(np.nan, index=candidates.index, dtype=float)
    z_audit = pd.Series(np.nan, index=candidates.index, dtype=float)

    if probability_uncertainty is None:
        conservative.loc[active] = np.maximum(0.5, raw.loc[active] - probability_haircut)
        method.loc[active] = "fixed_haircut" if probability_haircut > 0.0 else "point_probability"
    else:
        if "game_id" not in candidates:
            raise ValueError("game_id is required with candidate-specific probability uncertainty")
        if not isinstance(probability_uncertainty, pd.DataFrame):
            raise TypeError("probability_uncertainty must be a labeled pandas DataFrame")
        if probability_uncertainty.index.has_duplicates:
            raise ValueError("probability_uncertainty game_id index must be unique")
        active_ids = candidates.loc[active, "game_id"].astype(str)
        if (
            candidates.loc[active, "game_id"].isna().any()
            or active_ids.str.strip().eq("").any()
            or active_ids.duplicated().any()
        ):
            raise ValueError("active candidate game_id values must be non-empty and unique")
        uncertainty = probability_uncertainty.copy()
        uncertainty.index = uncertainty.index.astype(str)
        if uncertainty.index.has_duplicates:
            raise ValueError("probability_uncertainty game_id index must be unique")
        if set(uncertainty.index) != set(active_ids):
            raise ValueError(
                "probability_uncertainty index must exactly match active candidate game_id values"
            )
        required_uncertainty = {"bet_side"}
        missing_uncertainty = sorted(required_uncertainty.difference(uncertainty.columns))
        if missing_uncertainty:
            raise ValueError("probability_uncertainty must include bet_side")
        if not {"probability_lower_bound", "posterior_sd"}.intersection(uncertainty.columns):
            raise ValueError(
                "probability_uncertainty must include probability_lower_bound or posterior_sd"
            )
        if "probability_lower_bound" not in uncertainty:
            uncertainty["probability_lower_bound"] = np.nan
        if "posterior_sd" not in uncertainty:
            uncertainty["posterior_sd"] = np.nan
        uncertainty = uncertainty.loc[active_ids.tolist()]
        uncertainty.index = candidates.index[active]
        supplied_sides = uncertainty["bet_side"].astype(str)
        if not supplied_sides.eq(sides.loc[active]).all():
            raise ValueError("probability_uncertainty bet_side must match each active candidate")
        lower = pd.to_numeric(uncertainty["probability_lower_bound"], errors="coerce")
        posterior_sd = pd.to_numeric(uncertainty["posterior_sd"], errors="coerce")
        lower_present = lower.notna()
        sd_present = posterior_sd.notna()
        if not (lower_present ^ sd_present).all():
            raise ValueError(
                "Each probability_uncertainty row must supply exactly one lower bound "
                "or posterior_sd"
            )
        if ((lower.loc[lower_present] < 0.0) | (lower.loc[lower_present] > 1.0)).any():
            raise ValueError("probability lower bounds must lie in [0, 1]")
        if (lower.loc[lower_present] > raw.loc[active].loc[lower_present] + 1e-12).any():
            raise ValueError("probability lower bounds cannot exceed the point probability")
        if (
            not np.isfinite(posterior_sd.loc[sd_present]).all()
            or (posterior_sd.loc[sd_present] < 0.0).any()
        ):
            raise ValueError("posterior_sd values must be finite and non-negative")

        active_conservative = raw.loc[active].copy()
        active_conservative.loc[lower_present] = lower.loc[lower_present]
        active_conservative.loc[sd_present] = (
            raw.loc[active].loc[sd_present] - posterior_z * posterior_sd.loc[sd_present]
        )
        conservative.loc[active] = np.maximum(0.5, active_conservative)
        active_method = pd.Series("posterior_sd", index=uncertainty.index, dtype="object")
        active_method.loc[lower_present] = "supplied_lower_bound"
        method.loc[active] = active_method
        lower_audit.loc[active] = lower
        sd_audit.loc[active] = posterior_sd
        z_audit.loc[active & method.eq("posterior_sd")] = posterior_z

    audit = pd.DataFrame(index=candidates.index)
    audit["raw_bet_probability"] = raw
    audit["conservative_bet_probability"] = conservative
    audit["probability_uncertainty_method"] = method
    audit["supplied_probability_lower_bound"] = lower_audit
    audit["supplied_posterior_sd"] = sd_audit
    audit["posterior_z"] = z_audit
    audit["effective_probability_haircut"] = np.maximum(0.0, raw - conservative)
    return audit.loc[:, list(AUDIT_COLUMNS)]
