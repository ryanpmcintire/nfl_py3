"""No-vig market probability diagnostics (MKT-03).

Turns the point-in-time consensus PRICE this project's own purchased market
archive already captures into two-way-normalized, vig-removed probabilities
and holds, plus a favourite-longshot calibration diagnostic against realized
outcomes:

- spreads: :func:`spread_novig_probabilities` consumes
  ``nfl_ats.clv.spread_price_consensus_table``'s consensus HOME/AWAY spread
  PRICE (a sibling extraction added alongside ``nfl_ats.clv.build_pairing_table``
  -- ``build_pairing_table`` itself is untouched).
- moneyline: :func:`moneyline_novig_probabilities` consumes the
  ``home_moneyline``/``away_moneyline`` columns :func:`nfl_ats.clv.build_pairing_table`
  already produces -- no new extraction needed there.
- calibration: :func:`favourite_longshot_calibration` buckets either
  probability against a realized 0/1 outcome and reports the gap, with
  :func:`bootstrap_calibration_gaps` reusing
  :func:`nfl_ats.clv.week_blocked_bootstrap` (no new bootstrap code).

No new probability math is introduced anywhere in this module: every
probability/hold figure is produced by calling ``nfl_ats.odds``'s already
audited, unit-tested ``no_vig_probabilities``/``market_hold``
(``tests/test_odds.py``) -- the same functions ``nfl_ats.margin``,
``nfl_ats.backtest``, and ``nfl_ats.prediction_safety`` already call
elsewhere in this codebase, there against a single, already-existing
per-game ``home_spread_odds``/``away_spread_odds`` feature column (sourced
upstream of ``nfl_ats.clv``, not from the point-in-time multi-book market
archive). What is new here is applying that same math to the richer,
per-book, per-decision-label consensus price this project's ``clv`` module
manages -- nothing in the pipeline had surfaced that price before (see
``docs/novig_diagnostics.md``).

This is a DIAGNOSTIC module, not a candidate-feature pipeline. Per rule 8 of
``docs/rotation_registry.md`` ("CFB and non-reserved seasons stay free...
The registry governs NFL confirmation looks only"), descriptive measurement
that produces no pick and gates no modeling decision is not itself an NFL
confirmation look, so nothing here predeclares a window or calls
``nfl_ats.rotation``. **Consuming any output of this module as a model
feature is explicitly out of scope** -- that is a new candidate and must go
through the project's normal look-spending process (rotation registry /
``registry/weak_signals.json``), not be treated as validated merely by
having appeared here. See ``docs/novig_diagnostics.md`` for the full
reasoning and the measured results.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral

import numpy as np
import pandas as pd

from nfl_ats.clv import BootstrapBlock, week_blocked_bootstrap
from nfl_ats.data import DataContractError
from nfl_ats.odds import market_hold, no_vig_probabilities

DEFAULT_CALIBRATION_BUCKETS = 5


def _validated_bucket_count(buckets: object) -> int:
    if isinstance(buckets, bool) or not isinstance(buckets, Integral) or buckets < 1:
        raise ValueError("buckets must be a positive integer")
    return int(buckets)


def _validated_probability(values: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    malformed = values.notna() & numeric.isna()
    if malformed.any():
        raise DataContractError(f"{name} contains non-numeric values")
    resolved = numeric.notna()
    finite = np.isfinite(numeric.loc[resolved].to_numpy(dtype=float))
    if not finite.all() or not numeric.loc[resolved].between(0.0, 1.0).all():
        raise DataContractError(f"{name} must contain probabilities in [0, 1] or null")
    return numeric


def _validated_outcome(values: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    malformed = values.notna() & numeric.isna()
    if malformed.any():
        raise DataContractError(f"{name} contains non-numeric values")
    resolved = numeric.notna()
    if (
        not np.isfinite(numeric.loc[resolved].to_numpy(dtype=float)).all()
        or not numeric.loc[resolved].isin([0.0, 1.0]).all()
    ):
        raise DataContractError(f"{name} must contain binary 0/1 outcomes or null")
    return numeric


def _validated_edges(edges: np.ndarray) -> np.ndarray:
    try:
        resolved = np.asarray(edges, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("calibration edges must be numeric") from error
    if resolved.ndim != 1 or len(resolved) < 2:
        raise ValueError("calibration edges must be a one-dimensional array with at least 2 values")
    if np.isnan(resolved).any() or not np.all(np.diff(resolved) > 0.0):
        raise ValueError("calibration edges must be strictly increasing and cannot contain NaN")
    if not np.isneginf(resolved[0]) or not np.isposinf(resolved[-1]):
        raise ValueError("calibration edges must start at -inf and end at +inf")
    if len(resolved) > 2 and not np.all((resolved[1:-1] >= 0.0) & (resolved[1:-1] <= 1.0)):
        raise ValueError("interior calibration edges must lie in [0, 1]")
    return resolved.copy()


def _apply_no_vig_row_by_row(home: pd.Series, away: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Reuse ``nfl_ats.odds``'s audited scalar functions row-by-row.

    Deliberately not re-derived as a vectorized formula: this project has one
    audited, tested definition of the American-odds vig-removal math
    (``tests/test_odds.py``), and every existing caller of it elsewhere in
    the codebase (``nfl_ats.margin``, ``nfl_ats.backtest``,
    ``nfl_ats.prediction_safety``) also calls it per-row rather than
    re-deriving the arithmetic. Row counts here are one row per (game,
    decision label) -- thousands, not the archive's millions of raw quote
    rows -- so a Python-level loop is not a performance concern.

    A row with a missing or exactly-zero price on either side gets NaN in
    both outputs rather than a silent ``-110`` fallback: ``nfl_ats.odds``'s
    internal ``_resolved_odds`` would substitute the default price for
    ``None``, which is exactly the assumption this diagnostic exists to
    test (52.8% of archived spread quotes are not ``-110`` -- see
    ``docs/novig_diagnostics.md`` section 3a).
    """

    home_values = pd.to_numeric(home, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    away_values = pd.to_numeric(away, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    malformed_home = home.notna().to_numpy() & ~np.isfinite(home_values)
    malformed_away = away.notna().to_numpy() & ~np.isfinite(away_values)
    if malformed_home.any() or malformed_away.any():
        raise DataContractError("No-vig price columns must contain finite numeric odds or null")
    valid = (
        np.isfinite(home_values)
        & np.isfinite(away_values)
        & (home_values != 0.0)
        & (away_values != 0.0)
    )
    home_probability = np.full(len(home_values), np.nan, dtype=float)
    hold = np.full(len(home_values), np.nan, dtype=float)
    for position in np.flatnonzero(valid):
        home_p, _away_p = no_vig_probabilities(home_values[position], away_values[position])
        home_probability[position] = home_p
        hold[position] = market_hold(home_values[position], away_values[position])
    return home_probability, hold


def spread_novig_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    """No-vig ATS (home-cover) probability and hold from consensus spread PRICES.

    ``frame`` needs ``home_spread_price``/``away_spread_price`` -- e.g. the
    output of :func:`nfl_ats.clv.spread_price_consensus_table`. Adds
    ``no_vig_home_cover_probability`` and ``spread_hold`` columns; every
    other column passes through unchanged.
    """

    required = {"home_spread_price", "away_spread_price"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"Spread price frame is missing columns: {', '.join(missing)}")
    result = frame.copy()
    probability, hold = _apply_no_vig_row_by_row(
        result["home_spread_price"], result["away_spread_price"]
    )
    result["no_vig_home_cover_probability"] = probability
    result["spread_hold"] = hold
    return result


def moneyline_novig_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    """No-vig SU (straight-up home win) probability and hold from moneyline prices.

    ``frame`` needs ``home_moneyline``/``away_moneyline`` -- already columns
    on :func:`nfl_ats.clv.build_pairing_table`'s output, so this needs no new
    extraction. Secondary-goal-only (``docs/novig_diagnostics.md`` section
    3b): answers "who wins straight up," not "who covers," and the pool
    grades ATS.
    """

    required = {"home_moneyline", "away_moneyline"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"Moneyline frame is missing columns: {', '.join(missing)}")
    result = frame.copy()
    probability, hold = _apply_no_vig_row_by_row(result["home_moneyline"], result["away_moneyline"])
    result["no_vig_home_win_probability"] = probability
    result["moneyline_hold"] = hold
    return result


def calibration_bucket_edges(
    probability: pd.Series, buckets: int = DEFAULT_CALIBRATION_BUCKETS
) -> np.ndarray:
    """Fixed quantile bucket edges for a probability column.

    Computed once (on the full available sample) and then reused identically
    for both the point-estimate table and every bootstrap resample --
    recomputing quantile edges per resample would make the buckets
    themselves a moving target and the resulting interval uninterpretable.
    The outer edges are widened to +/-inf so every real-valued probability
    (including exactly 0.0 or 1.0) always falls inside the outer buckets.
    Duplicate quantiles (heavily repeated probability values) collapse to
    fewer, wider buckets rather than raising.
    """

    resolved_buckets = _validated_bucket_count(buckets)
    values = _validated_probability(probability, name="probability").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("No non-null probabilities to compute bucket edges from")
    quantiles = np.linspace(0.0, 1.0, resolved_buckets + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 2:
        raise ValueError("Probability column has too little variation to bucket")
    edges[0] = -np.inf
    edges[-1] = np.inf
    return _validated_edges(edges)


def favourite_longshot_calibration(
    frame: pd.DataFrame,
    probability_col: str,
    outcome_col: str,
    *,
    edges: np.ndarray | None = None,
    buckets: int = DEFAULT_CALIBRATION_BUCKETS,
) -> pd.DataFrame:
    """Bucket table: per-bucket n, mean predicted probability, mean observed frequency, gap.

    Rows with a null probability or a null outcome (e.g. an ATS push or a
    straight-up tie, excluded the same way :func:`nfl_ats.clv.pick_correct`
    excludes pushes) are dropped before bucketing rather than scored.
    ``edges`` lets a caller pass fixed bucket boundaries (e.g. to keep them
    identical across a bootstrap's resamples via
    :func:`calibration_bucket_edges`); when omitted, edges are computed
    fresh from this call's own data.

    ``calibration_gap`` is ``mean_observed_frequency - mean_predicted_probability``
    (positive = market underpriced this bucket, i.e. realized frequency beat
    the no-vig probability). ``brier_component`` retains the original
    unweighted squared gap for compatibility; ``brier_reliability_contribution``
    is the standard bucket-share-weighted contribution to Brier reliability.
    """

    missing = sorted({probability_col, outcome_col} - set(frame.columns))
    if missing:
        raise DataContractError(f"Calibration frame is missing columns: {', '.join(missing)}")
    _validated_bucket_count(buckets)
    probability = _validated_probability(frame[probability_col], name=probability_col)
    outcome = _validated_outcome(frame[outcome_col], name=outcome_col)
    valid = probability.notna() & outcome.notna()
    probability = probability.loc[valid]
    outcome = outcome.loc[valid]
    if probability.empty:
        raise ValueError("No rows with both a probability and a resolved outcome")
    resolved_edges = (
        _validated_edges(edges)
        if edges is not None
        else calibration_bucket_edges(probability, buckets)
    )
    bucket_index = pd.cut(probability, bins=list(resolved_edges), include_lowest=True, labels=False)
    working = pd.DataFrame(
        {
            "probability": probability.to_numpy(),
            "outcome": outcome.to_numpy(),
            "bucket": bucket_index.to_numpy(),
        }
    )
    grouped = (
        working.groupby("bucket", observed=True)
        .agg(
            n=("probability", "size"),
            mean_predicted_probability=("probability", "mean"),
            mean_observed_frequency=("outcome", "mean"),
        )
        .reset_index()
    )
    grouped["calibration_gap"] = (
        grouped["mean_observed_frequency"] - grouped["mean_predicted_probability"]
    )
    grouped["bucket_weight"] = grouped["n"] / grouped["n"].sum()
    grouped["brier_component"] = grouped["calibration_gap"] ** 2
    grouped["brier_reliability_contribution"] = (
        grouped["bucket_weight"] * grouped["brier_component"]
    )
    bucket_positions = grouped["bucket"].to_numpy(dtype=int)
    grouped["bucket_lower"] = resolved_edges[bucket_positions]
    grouped["bucket_upper"] = resolved_edges[bucket_positions + 1]
    return (
        grouped[
            [
                "bucket",
                "bucket_lower",
                "bucket_upper",
                "n",
                "bucket_weight",
                "mean_predicted_probability",
                "mean_observed_frequency",
                "calibration_gap",
                "brier_component",
                "brier_reliability_contribution",
            ]
        ]
        .sort_values("bucket")
        .reset_index(drop=True)
    )


def calibration_gap_metric_fn(
    probability_col: str, outcome_col: str, edges: np.ndarray
) -> Callable[[pd.DataFrame], dict[str, float]]:
    """Build a ``week_blocked_bootstrap``-shaped ``metric_fn`` for fixed bucket edges.

    Returns one named metric per bucket (``bucket_{i}_gap``, ``i`` counting
    from 0) plus ``mean_abs_calibration_gap`` averaged over whichever
    buckets have at least one row in the resampled frame -- exactly the
    ``metric_fn`` shape ``nfl_ats.clv.clv_summary``/``opener_evaluation_metrics``
    already use, so :func:`nfl_ats.clv.week_blocked_bootstrap` needs no new
    bootstrap code. A bucket with zero rows in a given resample reports
    ``nan`` for that draw rather than a fabricated value; with block
    resampling over many weeks this is expected to be rare for the outer
    buckets and essentially never for the middle ones.
    """

    resolved_edges = _validated_edges(edges)
    bucket_count = len(resolved_edges) - 1

    def _metric(rows: pd.DataFrame) -> dict[str, float]:
        missing = sorted({probability_col, outcome_col}.difference(rows.columns))
        if missing:
            raise DataContractError(f"Calibration frame is missing columns: {', '.join(missing)}")
        probability = _validated_probability(rows[probability_col], name=probability_col)
        outcome = _validated_outcome(rows[outcome_col], name=outcome_col)
        valid = probability.notna() & outcome.notna()
        probability = probability.loc[valid]
        outcome = outcome.loc[valid]
        bucket_index = pd.cut(
            probability, bins=list(resolved_edges), include_lowest=True, labels=False
        )
        result: dict[str, float] = {}
        counts: list[int] = []
        gaps: list[float] = []
        for index in range(bucket_count):
            in_bucket = bucket_index == index
            if in_bucket.any():
                gap = float(outcome.loc[in_bucket].mean() - probability.loc[in_bucket].mean())
                result[f"bucket_{index}_gap"] = gap
                counts.append(int(in_bucket.sum()))
                gaps.append(gap)
            else:
                result[f"bucket_{index}_gap"] = float("nan")
        finite = [value for value in result.values() if np.isfinite(value)]
        result["mean_abs_calibration_gap"] = (
            float(np.mean(np.abs(finite))) if finite else float("nan")
        )
        total = sum(counts)
        result["expected_calibration_error"] = (
            float(np.average(np.abs(gaps), weights=counts)) if total else float("nan")
        )
        result["brier_reliability"] = (
            float(np.average(np.square(gaps), weights=counts)) if total else float("nan")
        )
        return result

    return _metric


def bootstrap_calibration_gaps(
    frame: pd.DataFrame,
    probability_col: str,
    outcome_col: str,
    edges: np.ndarray,
    *,
    block: BootstrapBlock = "week",
    samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 20260818,
) -> pd.DataFrame:
    """Week/season-blocked bootstrap CIs for each fixed calibration bucket's gap.

    Thin convenience wrapper around :func:`nfl_ats.clv.week_blocked_bootstrap`
    with :func:`calibration_gap_metric_fn` -- no new bootstrap logic. ``frame``
    needs ``season``/``week`` columns (``week_blocked_bootstrap``'s own
    requirement).
    """

    missing = sorted({probability_col, outcome_col}.difference(frame.columns))
    if missing:
        raise DataContractError(f"Calibration frame is missing columns: {', '.join(missing)}")
    probability = _validated_probability(frame[probability_col], name=probability_col)
    outcome = _validated_outcome(frame[outcome_col], name=outcome_col)
    valid_mask = probability.notna() & outcome.notna()
    valid = frame.loc[valid_mask].copy()
    valid[probability_col] = probability.loc[valid_mask]
    valid[outcome_col] = outcome.loc[valid_mask]
    if valid.empty:
        raise ValueError("No rows with both a probability and a resolved outcome to bootstrap")
    metric_fn = calibration_gap_metric_fn(probability_col, outcome_col, _validated_edges(edges))
    return week_blocked_bootstrap(
        valid, metric_fn, block=block, samples=samples, confidence=confidence, seed=seed
    )
