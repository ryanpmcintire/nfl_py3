"""Anytime-valid inference for the paired feature-set comparison.

``experiments.paired_feature_comparisons`` answers "is the candidate better,
given the games we happened to score so far" with a FIXED-SAMPLE block
bootstrap: its 95% interval is only guaranteed to have 95% coverage if you
compute it once, at a sample size chosen before looking at the data. This
project cannot honour that discipline -- ``docs/rotation_registry.md``
documents ~130-150 looks already taken against 2018-2025, and the whole
reason windows are rationed is that repeated peeking at a fixed-sample
interval inflates the false-positive rate every time you check it. Peeking
after every week for a season and stopping the first time an interval
excludes zero is a classic multiple-testing violation dressed up as
patience.

This module answers the same question -- is the candidate's paired per-game
improvement positive -- with an estimator that stays valid under CONTINUOUS
MONITORING: you may look after every week, every season, or every game, and
stop whenever you like, without inflating Type-I error. Concretely it is a
**normal-mixture test martingale / confidence sequence** (Robbins 1970;
formalised as the "Normal mixture" nonnegative supermartingale in Howard,
Ramdas, McAuliffe & Sesia 2021, *Time-uniform, nonparametric, nonasymptotic
confidence sequences*, Ann. Statist.; the same construction backs the
"always-valid p-values" used for continuously-monitored A/B tests in Johari,
Pekelis & Walsh 2015). It was chosen over the two other suitable families
named in this module's brief -- a WSR betting martingale or an
empirical-Bernstein sequence -- because it has a single closed form with no
online optimisation loop to get subtly wrong, which matters here: an
under-covering interval would be worse than the status quo, and a closed
form is the easiest kind of formula to unit-test against algebra instead of
against its own code.

The assumption doing the work
------------------------------
A test martingale needs increments that are, conditional on everything seen
so far, mean-``mu0`` under the null and bounded (hence sub-Gaussian by
Hoeffding's lemma). Individual GAMES do not satisfy this -- games in the same
week share injury news, weather, and market over/under-reactions, which is
exactly why ``paired_feature_comparisons`` resamples whole weeks or seasons
instead of rows. This module makes the same choice at the martingale level:
**the unit fed to the martingale is one whole week (or season) block, never
an individual game.** Concretely, each block's contribution is its raw SUM
of per-game improvements, treated as ONE bounded random variable on
``[-k, k]`` for a ``k``-game block (Hoeffding's lemma again, now applied at
the block-sum scale), which is deliberately the most conservative treatment
available: it assigns a block of games the same worst-case variance a
single perfectly-correlated swing of that size would have, so it needs no
assumption whatsoever about what happens *within* a block. What the method
DOES assume is that different blocks are conditionally mean-``mu0`` given
the past -- i.e. no drift in the true effect across the peeking horizon,
and no dependence of a block's expected outcome on how earlier blocks
came out beyond being bounded. That is weaker than the exchangeability the
existing bootstrap already assumes (exchangeability implies it), so nothing
here is being asked of the data that the status quo was not already asking.

Two direct consequences worth stating plainly:

- Because a block contributes exactly ONE number regardless of its game
  count, the effective sample size this method resolves against is the
  BLOCK COUNT, never the game count. A three-season NFL window is ~150-160
  weekly looks, not ~800 independent games -- this is intentional, and is
  what keeps the interval from being narrower than the status quo.
- ``log_loss_improvement`` is excluded on purpose. It is unbounded (a
  single confidently-wrong prediction can blow it out arbitrarily), so it
  has no fixed sub-Gaussian proxy and would need a materially different
  (heavier-tailed) construction. Only ``accuracy_improvement`` and
  ``brier_improvement`` are supported, both provably bounded in ``[-1, 1]``
  as differences of quantities already bounded in ``[0, 1]``.

A zero-assumption worst case has a real cost: at the fully conservative
setting (every game in a block moving in lockstep, ``intraclass_
correlation=1.0``, and a single game's own variance at Hoeffding's worst
case, ``per_game_variance_proxy=1.0``), the method needs on the order of a
million games to resolve a 1-2 accuracy-point effect -- unusable at this
project's scale (``docs/anytime_valid.md``). ``per_game_variance_proxy``
defaults to that worst case and is overridden with a measured value
deliberately, per call, as a stated assumption.

``intraclass_correlation`` is different: it does **not** default to the
worst case. **This project's standing decision, made explicitly and not
re-litigated per comparison, is independence** -- ``intraclass_correlation
=0.0``. Games within a week involve disjoint teams playing separate
contests with no shared outcome mechanism; the design effect this
parameter controls (Kish: ``DEFF=1+(k-1)*icc``) should be exactly 1 for
that reason, not merely "measured small." An earlier version of this module
padded the assumption to 0.10 anyway, "to be safe" -- ``docs/anytime_
valid.md`` records the correction: at the real block sizes here that pad
cost 2.4x the effective NFL sample and 5.5x the effective CFB sample for a
correlation the data does not support (four independent measurements,
NFL and CFB, all landing within a hair of zero and straddling it -- the
signature of noise around exactly zero, not of a small positive value). The
worst case remains available as ``WORST_CASE_INTRACLASS_CORRELATION`` for
an explicit stress test; it must never be silently reintroduced as a
default, and no per-comparison auto-estimator sets the operating value --
`anova_intraclass_correlation`/`bootstrap_intraclass_correlation` exist only
as read-only diagnostics a caller may report alongside the confidence
sequence, never as an input to it.

Reading the output
-------------------
At every look ``t`` this module reports both faces of the same evidence:
an e-value (``e_value`` / ``log_e_value``, a test martingale for the null
"no improvement") and a confidence sequence (``lower``/``upper``, the set
of null means not yet rejected). They are dual by construction -- the
interval excludes zero exactly when the e-value clears ``1/alpha`` -- and
``tests/test_anytime.py`` pins that identity so it can never silently drift.
Per ``AGENTS.md``, an interval containing zero is never a rejection; it is
simply "not yet resolved," and unlike the fixed-sample interval, THIS
interval may be checked again next week without penalty.

What this replaces, and what it does not
------------------------------------------
It replaces the STOPPING RULE, not the rationing rule: a family may still
draw at most one rotation window, but within that window every week may be
inspected as it arrives instead of waiting for the window to close, and a
window that resolves early no longer has to sit idle to "look predeclared."
It does not relax ``rotation_registry.md``'s window-assignment or
contamination-inheritance rules, does not touch the 2018-2025 multiplicity
discount, and does not license re-scoring an already-spent window under a
new name. See ``docs/anytime_valid.md`` for the calibration-under-peeking
proof, the power table on planted effects, and the answer for the
1.3-point injury-value lead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.experiments import PairedBlock

DEFAULT_ALPHA = 0.05

# accuracy_improvement and brier_improvement are both differences of two
# quantities already bounded in [0, 1] (a 0/1 correctness indicator, or a
# Brier score), so both are bounded in [-1, 1]. log_loss_improvement is not
# bounded -- see the module docstring -- and is deliberately unsupported.
ANYTIME_METRICS: tuple[str, ...] = ("accuracy_improvement", "brier_improvement")

# The rotation registry's OWN default confirmation-window size (three
# seasons, ~800 games; docs/rotation_registry.md, "Window mechanics") is
# already the project's established scale for "enough evidence to decide
# something." Reusing it as the default target horizon for tuning the
# martingale's prior variance means the default is derived from an existing,
# documented project constant rather than invented for this module.
DEFAULT_TARGET_GAMES = 800

# Standing project decision (2026-08-18, docs/anytime_valid.md): games within
# a week involve disjoint teams playing separate contests with no shared
# outcome mechanism, so independence is the modelling decision for the
# within-block correlation, not a quantity to estimate per comparison. This
# is the default everywhere below; ``WORST_CASE_INTRACLASS_CORRELATION`` is
# kept as an explicit, named override for stress-testing only.
DEFAULT_INTRACLASS_CORRELATION = 0.0
WORST_CASE_INTRACLASS_CORRELATION = 1.0

_REQUIRED_PREDICTION_COLUMNS = frozenset(
    {"feature_set", "game_id", "season", "week", "home_cover", "home_cover_probability"}
)


# ---------------------------------------------------------------------------
# Pairing (mirrors experiments.paired_feature_comparisons' contract exactly,
# duplicated rather than imported so this module owns its correctness
# independently of any future change to that file, which this brief forbids
# touching).
# ---------------------------------------------------------------------------


def _validate_predictions_columns(predictions: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing paired columns: {', '.join(missing)}")


def _paired_rows(
    predictions: pd.DataFrame, baseline_feature_set: str, candidate_feature_set: str
) -> pd.DataFrame:
    columns = ["game_id", "season", "week", "home_cover", "home_cover_probability"]
    baseline = predictions.loc[predictions["feature_set"].eq(baseline_feature_set), columns]
    candidate = predictions.loc[predictions["feature_set"].eq(candidate_feature_set), columns]
    paired = baseline.merge(
        candidate,
        on="game_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_candidate"),
    )
    paired = paired.loc[
        paired["home_cover_baseline"].notna() & paired["home_cover_candidate"].notna()
    ].copy()
    if paired.empty:
        raise ValueError(f"No paired completed games for {candidate_feature_set}")
    for column in ("season", "week", "home_cover"):
        if not paired[f"{column}_baseline"].equals(paired[f"{column}_candidate"]):
            raise ValueError(f"Paired {column} values differ for {candidate_feature_set}")
    return paired


def _row_improvements(paired: pd.DataFrame) -> pd.DataFrame:
    actual = paired["home_cover_baseline"].to_numpy(dtype=float)
    baseline_p = paired["home_cover_probability_baseline"].to_numpy(dtype=float)
    candidate_p = paired["home_cover_probability_candidate"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "accuracy_improvement": ((candidate_p >= 0.5) == actual).astype(float)
            - ((baseline_p >= 0.5) == actual).astype(float),
            "brier_improvement": np.square(baseline_p - actual) - np.square(candidate_p - actual),
        },
        index=paired.index,
    )


def _ordered_blocks(
    paired: pd.DataFrame, values: pd.Series, block: PairedBlock
) -> tuple[pd.DataFrame, list[npt.NDArray[np.float64]]]:
    """Chronologically ordered (season[, week]) identity rows plus each block's raw values.

    Returns the raw per-block arrays (not just their sizes/sums) because the
    empirical intraclass-correlation estimate (``predictable_intraclass_
    correlation``) needs the individual observations, not just their
    aggregates.
    """

    frame = paired[["season_baseline", "week_baseline"]].copy()
    frame.columns = pd.Index(["season", "week"])
    frame["value"] = values.to_numpy(dtype=float)
    group_columns = ["season", "week"] if block == "week" else ["season"]
    grouped = frame.groupby(group_columns, sort=True)
    identity = grouped.size().reset_index()[group_columns]
    block_arrays = [group["value"].to_numpy(dtype=np.float64) for _, group in grouped]
    return identity, block_arrays


# ---------------------------------------------------------------------------
# The martingale / confidence sequence engine, pure-numeric so it is reusable
# by the calibration and power simulations without any DataFrame overhead.
# ---------------------------------------------------------------------------


def _per_block_variance(
    block_size: npt.NDArray[np.float64] | float,
    per_game_variance_proxy: float,
    intraclass_correlation: float,
) -> Any:
    """Kish's cluster design-effect formula: Var(sum of k) = k * s^2 * DEFF.

    ``DEFF = 1 + (k - 1) * icc``. At ``icc=0`` (the project's standing
    decision -- see the module docstring) this is exactly ``k * s^2``: games
    within a block are independent events, so a block's variance is just the
    sum of its games' own variances, no inflation. At ``icc=1`` it is
    ``k^2 * s^2`` -- every game in the block swings together, the fully
    conservative worst case kept available for stress-testing via an
    explicit override, never the default.
    """

    return (
        block_size * per_game_variance_proxy * (1.0 + (block_size - 1.0) * intraclass_correlation)
    )


def default_prior_variance(
    average_block_size: float,
    *,
    target_games: int = DEFAULT_TARGET_GAMES,
    per_game_variance_proxy: float = 1.0,
    intraclass_correlation: float = 0.0,
) -> float:
    """A GROW-flavoured default for the martingale's mixing (prior) variance.

    This tunes POWER only -- validity (Type-I error control) holds for any
    ``prior_variance > 0``; ``tests/test_anytime.py`` and the calibration
    simulation in ``scripts/anytime_validate.py`` confirm coverage holds
    across several values, including ones far from this default. The
    heuristic: the mixture's finite-sample behaviour is governed by the
    product ``prior_variance * cumulative_variance_process``; setting that
    product to 1 (the point where the mixing prior stops dominating and the
    data starts to) at the point where ``target_games`` worth of evidence
    has accrued gives a single closed form with no free parameters beyond
    the target itself. ``per_game_variance_proxy``/``intraclass_correlation``
    must match whatever will be passed to
    ``confidence_sequence_from_block_stats`` for the reference point to be
    meaningful; see that function for what they mean.
    """

    if average_block_size <= 0.0:
        raise ValueError("average_block_size must be positive")
    if target_games <= 0:
        raise ValueError("target_games must be positive")
    reference_variance_process = float(target_games) * float(
        _per_block_variance(average_block_size, per_game_variance_proxy, intraclass_correlation)
    )
    return 1.0 / reference_variance_process


def confidence_sequence_from_block_stats(
    block_sizes: npt.NDArray[np.float64],
    block_sums: npt.NDArray[np.float64],
    *,
    alpha: float = DEFAULT_ALPHA,
    prior_variance: float,
    per_game_variance_proxy: float = 1.0,
    intraclass_correlation: float = 0.0,
) -> pd.DataFrame:
    """The normal-mixture e-process and confidence sequence, one row per look.

    ``block_sizes``/``block_sums`` must already be in chronological order,
    one entry per block (week or season). Each block is treated as a single
    bounded observation of the cumulative paired-improvement SUM (see module
    docstring for why). Two knobs set the per-block variance:

    - ``per_game_variance_proxy`` (default ``1.0``, Hoeffding's worst case
      for a single game bounded in ``[-1, 1]``): an upper bound on one
      game's own variance. The true value is normally much smaller (this
      project's measured ``accuracy_improvement`` variance on real CFB data
      is ~0.55; see ``docs/anytime_valid.md``), so supplying a measured
      value here is a genuine, clearly-flagged additional assumption -- that
      the measured typical variance is representative going forward --
      traded for materially more power.
    - ``intraclass_correlation`` (default ``0.0``): Kish's cluster design
      effect, ``DEFF = 1 + (k - 1) * icc``, turning a block's variance into
      ``k * per_game_variance_proxy * DEFF`` (see ``_per_block_variance``).
      **This project's standing decision is independence** (``icc=0``):
      games within a week involve disjoint teams and separate contests, with
      no shared outcome mechanism, so a block's variance is just the sum of
      its games' own variances -- see the module docstring and
      ``docs/anytime_valid.md`` for the full rationale and the calibration
      check that confirms it. ``icc=1`` (every game in a block swinging
      together, the fully conservative Hoeffding worst case) is available
      as an explicit override for stress-testing, never the default.

    The two headline columns are dual by construction: ``excludes_zero`` at
    look ``t`` is true if and only if ``e_value`` at look ``t`` is at least
    ``1 / alpha`` (Ville's inequality applied to the same martingale from
    two directions). ``tests/test_anytime.py`` checks this algebraically.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if prior_variance <= 0.0:
        raise ValueError("prior_variance must be positive")
    if len(block_sizes) == 0:
        raise ValueError("At least one block is required")
    if len(block_sizes) != len(block_sums):
        raise ValueError("block_sizes and block_sums must have the same length")
    if np.any(block_sizes <= 0):
        raise ValueError("Every block must contain at least one game")
    if per_game_variance_proxy <= 0.0:
        raise ValueError("per_game_variance_proxy must be positive")
    if not 0.0 <= intraclass_correlation <= 1.0:
        raise ValueError("intraclass_correlation must be between 0 and 1")

    cumulative_games = np.cumsum(block_sizes)
    cumulative_sum = np.cumsum(block_sums)
    # The predictable variance process (see docstring): k_i is known from the
    # schedule before block i's outcomes are revealed, so it is predictable,
    # and per_game_variance_proxy/intraclass_correlation are fixed BEFORE any
    # monitoring begins -- neither is re-estimated online, which is what
    # keeps this a legitimate, non-adaptive variance process.
    cumulative_variance_process = np.cumsum(
        _per_block_variance(block_sizes, per_game_variance_proxy, intraclass_correlation)
    )

    denominator = 1.0 + prior_variance * cumulative_variance_process
    log_e_value = -0.5 * np.log(denominator) + (prior_variance * np.square(cumulative_sum)) / (
        2.0 * denominator
    )
    with np.errstate(over="ignore"):
        e_value = np.exp(np.clip(log_e_value, a_min=None, a_max=700.0))
    radius = (1.0 / cumulative_games) * np.sqrt(
        (2.0 * denominator / prior_variance) * np.log((1.0 / alpha) * np.sqrt(denominator))
    )
    mean = cumulative_sum / cumulative_games
    lower = mean - radius
    upper = mean + radius
    excludes_zero = (lower > 0.0) | (upper < 0.0)

    return pd.DataFrame(
        {
            "look": np.arange(1, len(block_sizes) + 1, dtype=int),
            "block_games": block_sizes.astype(int),
            "cumulative_games": cumulative_games.astype(int),
            "cumulative_variance_process": cumulative_variance_process,
            "cumulative_mean": mean,
            "log_e_value": log_e_value,
            "e_value": e_value,
            "lower": lower,
            "upper": upper,
            "excludes_zero": excludes_zero,
        }
    )


# ---------------------------------------------------------------------------
# The DataFrame-facing surface, wired alongside paired_feature_comparisons.
# ---------------------------------------------------------------------------


def paired_anytime_comparisons(
    predictions: pd.DataFrame,
    *,
    baseline_feature_set: str,
    metric: str = "accuracy_improvement",
    block: PairedBlock = "week",
    alpha: float = DEFAULT_ALPHA,
    prior_variance: float | None = None,
    target_games: int = DEFAULT_TARGET_GAMES,
    per_game_variance_proxy: float = 1.0,
    intraclass_correlation: float = 0.0,
) -> pd.DataFrame:
    """Anytime-valid paired per-game improvement, one row per candidate per look.

    Same input contract as ``experiments.paired_feature_comparisons`` --
    ``feature_set``/``game_id``/``season``/``week``/``home_cover``/
    ``home_cover_probability`` -- so it runs on any predictions frame already
    produced across this repo. Unlike that function it returns a full
    TRAJECTORY (one row per block seen so far), because the point of an
    anytime-valid method is that every prefix of the trajectory is a valid
    inference on its own; ``anytime_summary`` condenses it to one row per
    candidate for a quick headline read.

    ``per_game_variance_proxy`` defaults to the fully conservative worst
    case (see ``confidence_sequence_from_block_stats``); passing a measured,
    smaller value is a genuine additional assumption, not a free lunch --
    state it when you do.

    ``intraclass_correlation`` defaults to **``0.0``, a standing project
    decision, not an estimate**: games within a week involve disjoint teams
    and separate contests with no shared outcome mechanism, so independence
    is the modelling decision, not something re-derived per comparison (see
    the module docstring and ``docs/anytime_valid.md``). Pass an explicit
    float to override -- e.g. ``1.0`` for the assumption-free worst case --
    for a stress test; this parameter must never be used to plug in a
    per-comparison estimate.

    Each candidate's trace also carries ``measured_icc_diagnostic``: the
    plain ANOVA estimate (``anova_intraclass_correlation``) on this
    comparison's own blocks, reported so a reader can sanity-check the
    independence decision against whatever data is at hand. It is
    informational only and never feeds the variance process above.
    """

    if metric not in ANYTIME_METRICS:
        raise ValueError(
            f"Unsupported anytime metric {metric!r}; expected one of {ANYTIME_METRICS}. "
            "log_loss_improvement is unbounded and has no fixed sub-Gaussian proxy, so it "
            "cannot back this confidence sequence without a different (heavier-tailed) "
            "construction -- see the module docstring."
        )
    if block not in ("week", "season"):
        raise ValueError("block must be 'week' or 'season'")
    _validate_predictions_columns(predictions)
    feature_sets = set(predictions["feature_set"].astype(str))
    if baseline_feature_set not in feature_sets:
        raise ValueError(f"Unknown paired baseline feature set: {baseline_feature_set}")

    traces: list[pd.DataFrame] = []
    for candidate_name in sorted(feature_sets.difference((baseline_feature_set,))):
        paired = _paired_rows(predictions, baseline_feature_set, candidate_name)
        values = _row_improvements(paired)[metric]
        identity, block_arrays = _ordered_blocks(paired, values, block)
        block_sizes = np.array([len(b) for b in block_arrays], dtype=np.float64)
        block_sums = np.array([float(b.sum()) for b in block_arrays], dtype=np.float64)
        average_block_size = float(np.mean(block_sizes))
        # Diagnostic only -- see the docstring -- never fed back into rho or
        # the variance process, both of which use the fixed operating value.
        measured_icc_diagnostic = (
            anova_intraclass_correlation(block_arrays) if len(block_arrays) >= 2 else float("nan")
        )

        rho = (
            prior_variance
            if prior_variance is not None
            else default_prior_variance(
                average_block_size,
                target_games=target_games,
                per_game_variance_proxy=per_game_variance_proxy,
                intraclass_correlation=intraclass_correlation,
            )
        )
        trace = confidence_sequence_from_block_stats(
            block_sizes,
            block_sums,
            alpha=alpha,
            prior_variance=rho,
            per_game_variance_proxy=per_game_variance_proxy,
            intraclass_correlation=intraclass_correlation,
        )
        trace.insert(0, "candidate_feature_set", candidate_name)
        trace.insert(0, "baseline_feature_set", baseline_feature_set)
        trace["metric"] = metric
        trace["block"] = block
        trace["prior_variance"] = rho
        trace["per_game_variance_proxy"] = per_game_variance_proxy
        trace["intraclass_correlation"] = intraclass_correlation
        trace["measured_icc_diagnostic"] = measured_icc_diagnostic
        trace["season"] = identity["season"].to_numpy()
        trace["week"] = identity["week"].to_numpy() if block == "week" else None
        traces.append(trace)
    return pd.concat(traces, ignore_index=True)


def anytime_summary(trace: pd.DataFrame) -> pd.DataFrame:
    """Condense a ``paired_anytime_comparisons`` trace to one row per candidate."""

    group_columns = ["baseline_feature_set", "candidate_feature_set", "metric", "block"]
    rows: list[dict[str, Any]] = []
    for keys, group in trace.groupby(group_columns, sort=True):
        ordered = group.sort_values("look")
        final = ordered.iloc[-1]
        hits = ordered.loc[ordered["excludes_zero"]]
        identity = dict(zip(group_columns, keys, strict=True))
        rows.append(
            {
                **identity,
                "alpha": DEFAULT_ALPHA,
                "prior_variance": float(final["prior_variance"]),
                "intraclass_correlation": float(final["intraclass_correlation"]),
                "measured_icc_diagnostic": float(final["measured_icc_diagnostic"]),
                "looks": int(final["look"]),
                "games": int(final["cumulative_games"]),
                "final_estimate": float(final["cumulative_mean"]),
                "final_lower": float(final["lower"]),
                "final_upper": float(final["upper"]),
                "final_e_value": float(final["e_value"]),
                "final_log_e_value": float(final["log_e_value"]),
                "final_excludes_zero": bool(final["excludes_zero"]),
                "first_excluding_zero_look": (
                    int(hits.iloc[0]["look"]) if not hits.empty else None
                ),
                "first_excluding_zero_games": (
                    int(hits.iloc[0]["cumulative_games"]) if not hits.empty else None
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulation building blocks shared by tests/test_anytime.py (small-scale
# correctness checks) and scripts/anytime_validate.py (the full calibration
# and power study on real CFB block sizes).
# ---------------------------------------------------------------------------


def simulate_block_sequence(
    rng: np.random.Generator,
    block_sizes: Sequence[int],
    *,
    true_mean: float,
    total_variance: float = 0.545,
    intraclass_correlation: float = 0.10,
) -> list[npt.NDArray[np.float64]]:
    """One synthetic universe of bounded per-game improvement values, by block.

    Every block gets its own zero-mean shared shock on top of per-game
    idiosyncratic noise, decomposed so the SIMULATED per-game variance is
    exactly ``total_variance`` and exactly ``intraclass_correlation`` of it
    is shared within a block (a standard equal-correlation cluster model:
    ``shock ~ N(0, icc * total_variance)``, ``noise ~ N(0, (1-icc) *
    total_variance)``). Both defaults are the values measured on real CFB
    ``market`` vs ``market_residual`` predictions in
    ``docs/anytime_valid.md`` (variance ~0.545, a near-zero measured ICC
    padded to 0.10 for safety margin) -- this is the REALISTIC regime the
    power table uses; the calibration study additionally sweeps ICC up to
    1.0 as a deliberate stress test.

    Since the shock is zero-mean and drawn independently per block, every
    game's conditional mean is exactly ``true_mean`` regardless of the
    realized correlation, satisfying the martingale-difference condition no
    matter how ``intraclass_correlation`` is set. Values are clipped to
    ``[-1, 1]`` to match ``accuracy_improvement``/``brier_improvement``'s
    true range; at these variance scales and ``|true_mean| <= 0.02`` (i.e.
    <= 2 accuracy points) clipping is rare enough not to measurably bias the
    simulated mean away from ``true_mean``.
    """

    if total_variance <= 0.0:
        raise ValueError("total_variance must be positive")
    if not 0.0 <= intraclass_correlation <= 1.0:
        raise ValueError("intraclass_correlation must be between 0 and 1")
    shock_scale = float(np.sqrt(intraclass_correlation * total_variance))
    noise_scale = float(np.sqrt((1.0 - intraclass_correlation) * total_variance))
    blocks: list[npt.NDArray[np.float64]] = []
    for size in block_sizes:
        if size <= 0:
            raise ValueError("block sizes must be positive")
        shock = rng.normal(0.0, shock_scale) if shock_scale > 0.0 else 0.0
        noise = rng.normal(0.0, noise_scale, size=size) if noise_scale > 0.0 else np.zeros(size)
        blocks.append(np.clip(true_mean + shock + noise, -1.0, 1.0))
    return blocks


def block_bootstrap_ci_fast(
    block_sizes: npt.NDArray[np.float64],
    block_sums: npt.NDArray[np.float64],
    *,
    samples: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """The exact ``paired_feature_comparisons`` block-bootstrap CI, vectorized.

    ``paired_feature_comparisons`` resamples block INDICES with replacement,
    concatenates the resampled blocks' row positions (repeated rows for a
    block drawn more than once), and takes the mean over that pooled set of
    rows. That pooled mean equals the sum of the resampled blocks' own sums
    divided by the sum of their own sizes -- a block contributes its total
    exactly once per time it is drawn, whether or not you materialise its
    individual rows. Precomputing each block's ``(size, sum)`` turns every
    resample into two gather-and-sum operations instead of a row
    concatenation, which is what makes a many-universe, many-peek-point
    simulation of the FIXED-sample method tractable enough to run at all.
    """

    if samples < 1:
        raise ValueError("samples must be at least 1")
    n_blocks = len(block_sizes)
    if n_blocks == 0:
        raise ValueError("At least one block is required")
    draws = rng.integers(0, n_blocks, size=(samples, n_blocks))
    pooled_sum = block_sums[draws].sum(axis=1)
    pooled_count = block_sizes[draws].sum(axis=1)
    means = pooled_sum / pooled_count
    tail = alpha / 2.0
    lower = float(np.quantile(means, tail))
    upper = float(np.quantile(means, 1.0 - tail))
    return lower, upper


@dataclass(frozen=True)
class PeekingTrialResult:
    """One simulated universe's outcome under repeated weekly peeking.

    ``*_first_look`` is the 1-indexed block at which the interval FIRST
    excluded zero (``None`` if it never did). Under a true null, any
    non-``None`` value is a false alarm; under a planted effect, it is the
    detection time.
    """

    cs_excluded: bool
    cs_first_look: int | None
    cs_first_games: int | None
    fixed_sample_excluded: bool
    fixed_sample_first_look: int | None
    fixed_sample_first_games: int | None


def run_peeking_trial(
    rng: np.random.Generator,
    block_sizes: Sequence[int],
    *,
    true_mean: float,
    alpha: float = DEFAULT_ALPHA,
    prior_variance: float,
    fixed_sample_bootstrap_samples: int = 500,
    simulated_total_variance: float = 0.545,
    simulated_intraclass_correlation: float = 0.10,
    assumed_per_game_variance_proxy: float = 1.0,
    assumed_intraclass_correlation: float = 0.0,
    check_fixed_sample: bool = True,
    min_blocks_before_fixed_sample_check: int = 4,
) -> PeekingTrialResult:
    """Simulate one universe and peek after every block with both methods.

    This is the single building block behind both validation questions: with
    ``true_mean=0`` it measures a false-alarm rate (calibration); with
    ``true_mean=delta`` it measures a detection time (power). Both methods
    see the IDENTICAL simulated data for this universe, so the comparison
    isolates the estimator, not the noise draw.

    The ``simulated_*`` parameters control the DATA-GENERATING process (what
    is actually true in this universe); ``assumed_*`` control what the
    confidence sequence is TOLD to assume. ``assumed_intraclass_correlation``
    defaults to ``0.0``, matching ``paired_anytime_comparisons``'s standing
    project decision (independence -- disjoint teams, no shared outcome
    mechanism; see the module docstring and ``docs/anytime_valid.md``), and
    should stay there for the calibration study this function backs. Setting
    ``simulated_intraclass_correlation`` above ``assumed_intraclass_
    correlation`` is the deliberate stress test: does calibration survive
    the true correlation being higher than assumed. The fixed-sample
    comparator has no such assumption to violate, since it estimates
    variance from the resampled data directly, so this knob only affects
    the anytime method.

    ``min_blocks_before_fixed_sample_check`` skips the fixed-sample check
    until at least this many blocks exist: with fewer than ~4 blocks a block
    bootstrap has almost no distinct resamples to draw from (with 1 block it
    has exactly one, a point mass), so its "interval" degenerates to a
    single point that excludes zero unless the realized mean lands exactly
    on it -- a guaranteed false alarm that reflects a bootstrap run on too
    little data to mean anything, not the peeking problem this comparison is
    meant to isolate. No such floor exists for, or is applied to, the CS.
    ``check_fixed_sample=False`` skips the fixed-sample loop entirely, for
    when only the (much cheaper) CS side is needed at a large horizon.
    """

    blocks = simulate_block_sequence(
        rng,
        block_sizes,
        true_mean=true_mean,
        total_variance=simulated_total_variance,
        intraclass_correlation=simulated_intraclass_correlation,
    )
    sizes = np.array([len(b) for b in blocks], dtype=np.float64)
    sums = np.array([float(b.sum()) for b in blocks], dtype=np.float64)

    cs_trace = confidence_sequence_from_block_stats(
        sizes,
        sums,
        alpha=alpha,
        prior_variance=prior_variance,
        per_game_variance_proxy=assumed_per_game_variance_proxy,
        intraclass_correlation=assumed_intraclass_correlation,
    )
    cs_hits = cs_trace.loc[cs_trace["excludes_zero"]]
    cs_excluded = not cs_hits.empty
    cs_first_look = int(cs_hits.iloc[0]["look"]) if cs_excluded else None
    cs_first_games = int(cs_hits.iloc[0]["cumulative_games"]) if cs_excluded else None

    fixed_first_look: int | None = None
    fixed_first_games: int | None = None
    if check_fixed_sample:
        start = max(1, min_blocks_before_fixed_sample_check)
        for t in range(start, len(block_sizes) + 1):
            lower, upper = block_bootstrap_ci_fast(
                sizes[:t], sums[:t], samples=fixed_sample_bootstrap_samples, alpha=alpha, rng=rng
            )
            if lower > 0.0 or upper < 0.0:
                fixed_first_look = t
                fixed_first_games = int(np.cumsum(sizes[:t])[-1])
                break
    return PeekingTrialResult(
        cs_excluded=cs_excluded,
        cs_first_look=cs_first_look,
        cs_first_games=cs_first_games,
        fixed_sample_excluded=fixed_first_look is not None,
        fixed_sample_first_look=fixed_first_look,
        fixed_sample_first_games=fixed_first_games,
    )


# ---------------------------------------------------------------------------
# Measuring the intraclass correlation -- as a DIAGNOSTIC, not an input.
#
# ``intraclass_correlation`` is the single biggest lever on power (Kish's
# design effect is linear in it). This project's operating value is a
# standing decision, not an estimate: independence (icc=0), because games
# within a week involve disjoint teams with no shared outcome mechanism --
# see the module docstring and docs/anytime_valid.md. The two functions
# below exist so a caller CAN check what a specific comparison's own data
# says, as a sanity check against that decision; neither one feeds
# ``paired_anytime_comparisons``'s or ``confidence_sequence_from_block_
# stats``'s operating value, by design. Four independent checks so far (this
# module's own tests plus three real CFB comparisons and one NFL comparison
# in docs/anytime_valid.md) all land within a hair of zero.
# ---------------------------------------------------------------------------


def anova_intraclass_correlation(
    block_values: Sequence[npt.NDArray[np.float64]],
) -> float:
    """One-way random-effects ICC(1) for unbalanced blocks (Fisher's ANOVA estimator).

    Standard construction (e.g. Donner & Koval 1980; Shrout & Fleiss 1979's
    ICC(1,1) generalised to unequal group sizes): with block sizes
    :math:`n_i`, grand mean :math:`\\bar y`, block means :math:`\\bar y_i`,

    .. math::
        MSB=\\tfrac{1}{k-1}\\sum_i n_i(\\bar y_i-\\bar y)^2,\\quad
        MSW=\\tfrac{1}{N-k}\\sum_i\\sum_j(y_{ij}-\\bar y_i)^2,\\quad
        n_0=\\tfrac{1}{k-1}\\Big(N-\\tfrac{\\sum_i n_i^2}{N}\\Big)

        \\widehat{\\text{ICC}}=\\frac{MSB-MSW}{MSB+(n_0-1)MSW}

    Can be negative (block means vary LESS than independence would predict);
    that is a legitimate estimate, not an error -- callers wanting a
    conservative assumption should floor it at 0 explicitly, not have this
    function hide the sign.
    """

    blocks = [np.asarray(b, dtype=np.float64) for b in block_values]
    if len(blocks) < 2:
        raise ValueError("At least two blocks are required to estimate ICC")
    sizes = np.array([len(b) for b in blocks], dtype=np.float64)
    if np.any(sizes < 1):
        raise ValueError("Every block must contain at least one observation")
    k = len(blocks)
    n_total = float(sizes.sum())
    grand_mean = float(np.concatenate(blocks).mean())
    block_means = np.array([float(b.mean()) for b in blocks], dtype=np.float64)

    ssb = float(np.sum(sizes * (block_means - grand_mean) ** 2))
    msb = ssb / (k - 1)
    ssw = float(sum(np.sum((b - m) ** 2) for b, m in zip(blocks, block_means, strict=True)))
    degrees_within = n_total - k
    if degrees_within <= 0:
        raise ValueError("Not enough within-block degrees of freedom to estimate ICC")
    msw = ssw / degrees_within

    n0 = (n_total - float(np.sum(sizes**2)) / n_total) / (k - 1)
    denominator = msb + (n0 - 1.0) * msw
    if denominator == 0.0:
        return 0.0
    return (msb - msw) / denominator


def bootstrap_intraclass_correlation(
    block_values: Sequence[npt.NDArray[np.float64]],
    *,
    samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 20260818,
) -> dict[str, float]:
    """Block-bootstrap the ICC estimate: resample whole blocks, recompute.

    Consistent with this project's existing block-bootstrap uncertainty
    style (``reporting.block_bootstrap_intervals``,
    ``experiments.paired_feature_comparisons``) rather than an asymptotic
    F-distribution approximation, which is unreliable at ~200 blocks.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    point_estimate = anova_intraclass_correlation(block_values)
    blocks = list(block_values)
    n_blocks = len(blocks)
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected = generator.integers(0, n_blocks, size=n_blocks)
        draws[sample_index] = anova_intraclass_correlation([blocks[i] for i in selected])
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": point_estimate,
        "lower": float(np.quantile(draws, tail)),
        "upper": float(np.quantile(draws, 1.0 - tail)),
        "confidence": confidence,
        "samples": samples,
        "n_blocks": n_blocks,
    }
