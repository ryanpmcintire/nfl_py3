"""Validate the anytime-valid confidence sequence in ``nfl_ats.anytime``.

Three things, in order of how much they matter to the 2026 season:

1. **Calibration under peeking.** Simulate many null universes (candidate
   identical to baseline in expectation), peek after every week for a full
   season, and confirm the false-alarm rate of the anytime confidence
   sequence (CS) stays at or below the nominal alpha -- and confirm the
   FIXED-sample block bootstrap (``paired_feature_comparisons``'s own
   method), peeked the same way, does NOT. This is the headline result:
   2026 will be monitored every week, which is exactly the repeated-peeking
   regime the fixed-sample method is not built for.
2. **The intraclass correlation: independence is a standing project
   decision, not an estimate.** Games within a week involve disjoint teams
   playing separate contests with no shared outcome mechanism, so
   ``intraclass_correlation=0.0`` is hardcoded -- see ``docs/anytime_valid.
   md`` for the full rationale, including why an earlier padded (0.10) and
   even an auto-estimated version of this parameter were both considered
   and rejected. This script still MEASURES the ICC on three independent
   real CFB paired comparisons (a proper ANOVA, unbalanced-design
   estimator with a block-bootstrap confidence interval) purely as a
   diagnostic that confirms the decision -- all three, and an
   independently-supplied NFL measurement
   (``artifacts/availability_experiments/20260813T133345Z``, 2,075 games,
   ICC(week) = -0.0054), land within a hair of zero, straddling it, which
   is the signature of sampling noise around exactly zero rather than a
   small positive correlation. The measurement never sets the operating
   value.
3. **A short instrument-property appendix.** How many games the sequence
   needs to formally resolve a handful of effect sizes, at the operating
   (independence) assumption. This is NOT a request for more data and is
   not optimised further -- NFL produces ~285 games a year and CFB's usable
   archive is finite, so "how many games would resolve this" answers a
   different question than "what should the pool do this season." See
   ``docs/anytime_valid.md`` for that question; this script only measures
   the instrument.

A fourth question -- whether a betting-style (WSR) or game-level
cluster-robust construction recovers meaningfully more power than the
normal-mixture default -- was investigated and deliberately dropped
(2026-08-18): the project owner ruled that chasing sample-size reductions is
moot when every reachable number is still unreachable in a season. See
``docs/anytime_valid.md``, "Why the tightness investigation was dropped."

Everything empirical runs on CFB (rule 8: free, unlimited, no rotation
window) plus one independently-supplied NFL measurement cited, not
re-derived, here. This script does not touch
``registry/rotation_registry.json`` and reads no NFL outcome data itself.

Usage::

    .tools/uv.exe run --no-sync python scripts/anytime_validate.py \
        --output <scratch>/anytime

Runtime is well under a minute; pass ``--fast`` for an even quicker smoke run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.anytime import (
    DEFAULT_ALPHA,
    bootstrap_intraclass_correlation,
    confidence_sequence_from_block_stats,
    default_prior_variance,
    run_peeking_trial,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFB_PREDICTIONS = REPO / "artifacts/cfb_benchmark/20260818T115149Z/predictions.parquet"

# Measured on real CFB market vs market_residual predictions, clean_core
# window (below, ``measure_real_variance``): ~0.545. Rounded up slightly for
# a small safety margin.
MEASURED_PER_GAME_VARIANCE_PROXY = 0.55

# Standing project decision, not a measurement: games within a week are
# independent events (disjoint teams, no shared outcome mechanism), so the
# Kish design effect is exactly 1. measure_cfb_intraclass_correlations()
# below confirms this as a DIAGNOSTIC (three independent CFB comparisons,
# point estimates -0.0007/+0.0021/+0.0030, all 95% CIs inside
# [-0.005, +0.010], agreeing with an independently-supplied NFL measurement
# of -0.0054) -- it does not set this value.
OPERATING_INTRACLASS_CORRELATION = 0.0
# The fully conservative, assumption-free worst case (Hoeffding, every game
# in a block moving in lockstep) -- kept as a labelled floor for stress
# tests, never the operating value.
WORST_CASE_INTRACLASS_CORRELATION = 1.0
# Used only as a calibration STRESS test: does validity survive the
# operating assumption being wrong in the dangerous direction (true
# correlation the full worst case while the CS is told to assume
# independence)?
STRESS_INTRACLASS_CORRELATION = 1.0

APPENDIX_EFFECTS_ACCURACY_POINTS = (0.5, 1.0, 1.3, 2.0, 3.0)

# Real second and third CFB paired comparisons already on disk (independent
# candidate models, same clean_core population), used only to confirm the
# ICC measurement is not an artifact of one specific comparison.
ADDITIONAL_CFB_COMPARISONS: tuple[tuple[str, Path, str, str], ...] = (
    (
        "cfb_role_experiments",
        REPO / "artifacts/cfb_role_experiments/20260817T110541Z/predictions.parquet",
        "market_residual",
        "market_residual_roles",
    ),
    (
        "cfb_variance_experiments",
        REPO / "artifacts/cfb_variance_experiments/20260817T112146Z/predictions.parquet",
        "market_residual",
        "market_residual_variance",
    ),
)


# ---------------------------------------------------------------------------
# Real CFB inputs
# ---------------------------------------------------------------------------


def paired_accuracy_difference(
    predictions_path: Path, baseline_method: str, candidate_method: str, *, window: str
) -> pd.DataFrame:
    """Paired per-game ``accuracy_improvement`` for one CFB ``method``-column arm pair."""

    predictions = pd.read_parquet(predictions_path)
    clean = predictions.loc[predictions["evaluation_window"].eq(window)]
    columns = ["game_id", "season", "week", "home_cover", "home_cover_probability"]
    baseline = clean.loc[clean["method"].eq(baseline_method), columns]
    candidate = clean.loc[clean["method"].eq(candidate_method), columns]
    paired = baseline.merge(candidate, on="game_id", suffixes=("_baseline", "_candidate"))
    actual = paired["home_cover_baseline"].to_numpy(dtype=float)
    baseline_p = paired["home_cover_probability_baseline"].to_numpy(dtype=float)
    candidate_p = paired["home_cover_probability_candidate"].to_numpy(dtype=float)
    paired["accuracy_improvement"] = ((candidate_p >= 0.5) == actual).astype(float) - (
        (baseline_p >= 0.5) == actual
    ).astype(float)
    return paired.rename(columns={"season_baseline": "season", "week_baseline": "week"})


def load_real_cfb_comparison(predictions_path: Path) -> pd.DataFrame:
    """Paired per-game accuracy_improvement, market vs market_residual, clean_core."""

    return paired_accuracy_difference(
        predictions_path, "market", "market_residual", window="clean_core"
    )


def measure_real_variance(paired: pd.DataFrame) -> dict[str, float]:
    """Per-game variance of accuracy_improvement -- the other assumption this feeds."""

    values = paired["accuracy_improvement"].to_numpy(dtype=float)
    grouped = paired.groupby(["season", "week"])["accuracy_improvement"].agg(["size"])
    return {
        "games": len(paired),
        "blocks": len(grouped),
        "average_block_size": float(grouped["size"].mean()),
        "per_game_variance": float(values.var()),
        "mean_effect": float(values.mean()),
    }


def measure_cfb_intraclass_correlations(default_predictions_path: Path) -> pd.DataFrame:
    """ICC(week), ANOVA estimator + block-bootstrap CI, on three independent CFB comparisons.

    Three, not one: a single comparison's estimate could be a quirk of that
    particular candidate model. Agreement across three independently-fit
    candidates (a market-residual model, a role-continuity variant, and a
    heteroskedastic-variance variant) is what makes "ICC is indistinguishable
    from zero on CFB" a claim about the DATA rather than about one model.
    """

    sources = (
        (
            "cfb_benchmark (market vs market_residual)",
            default_predictions_path,
            "market",
            "market_residual",
        ),
        *(
            (f"{name} ({baseline} vs {candidate})", path, baseline, candidate)
            for name, path, baseline, candidate in ADDITIONAL_CFB_COMPARISONS
        ),
    )
    rows: list[dict[str, Any]] = []
    for label, path, baseline_method, candidate_method in sources:
        paired = paired_accuracy_difference(
            path, baseline_method, candidate_method, window="clean_core"
        )
        grouped = paired.groupby(["season", "week"])["accuracy_improvement"]
        blocks = [group.to_numpy(dtype=np.float64) for _, group in grouped]
        result = bootstrap_intraclass_correlation(blocks, samples=3000, seed=1)
        rows.append(
            {
                "comparison": label,
                "games": len(paired),
                "blocks": len(blocks),
                "average_block_size": float(np.mean([len(b) for b in blocks])),
                "icc_estimate": result["estimate"],
                "icc_lower_95": result["lower"],
                "icc_upper_95": result["upper"],
            }
        )
    return pd.DataFrame(rows)


def real_block_sizes(paired: pd.DataFrame, seasons: list[int]) -> list[int]:
    subset = paired.loc[paired["season"].isin(seasons)]
    grouped = subset.groupby(["season", "week"]).size().sort_index()
    return [int(size) for size in grouped.to_numpy()]


def nfl_scale_block_sizes(n_seasons: int, games_per_week: int = 16, weeks: int = 18) -> list[int]:
    """A schedule shape matching 2026's ~285-game NFL season (18 weeks, byes)."""

    one_season = [games_per_week] * (weeks - 1) + [games_per_week - 3]
    return one_season * n_seasons


# ---------------------------------------------------------------------------
# 1. Calibration under peeking -- the deliverable that matters this season
# ---------------------------------------------------------------------------


def run_calibration_study(
    block_sizes: list[int],
    *,
    label: str,
    universes: int,
    fixed_sample_bootstrap_samples: int,
    alpha: float,
    assumed_variance_proxy: float,
    assumed_icc: float,
    simulated_variance: float,
    simulated_icc: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    average_block_size = float(np.mean(block_sizes))
    prior_variance = default_prior_variance(
        average_block_size,
        target_games=int(sum(block_sizes)),
        per_game_variance_proxy=assumed_variance_proxy,
        intraclass_correlation=assumed_icc,
    )
    cs_false_alarms = 0
    fixed_false_alarms = 0
    started = time.perf_counter()
    for _ in range(universes):
        result = run_peeking_trial(
            rng,
            block_sizes,
            true_mean=0.0,
            alpha=alpha,
            prior_variance=prior_variance,
            fixed_sample_bootstrap_samples=fixed_sample_bootstrap_samples,
            simulated_total_variance=simulated_variance,
            simulated_intraclass_correlation=simulated_icc,
            assumed_per_game_variance_proxy=assumed_variance_proxy,
            assumed_intraclass_correlation=assumed_icc,
        )
        cs_false_alarms += int(result.cs_excluded)
        fixed_false_alarms += int(result.fixed_sample_excluded)
    elapsed = time.perf_counter() - started
    return {
        "label": label,
        "n_blocks": len(block_sizes),
        "n_games": int(sum(block_sizes)),
        "universes": universes,
        "alpha": alpha,
        "assumed_per_game_variance_proxy": assumed_variance_proxy,
        "assumed_intraclass_correlation": assumed_icc,
        "simulated_intraclass_correlation": simulated_icc,
        "prior_variance": prior_variance,
        "cs_false_alarm_rate": cs_false_alarms / universes,
        "fixed_sample_false_alarm_rate": fixed_false_alarms / universes,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# 2. Short instrument-property appendix -- NOT optimised, NOT a data request
# ---------------------------------------------------------------------------


def deterministic_games_to_detect(
    delta: float,
    block_size: float,
    prior_variance: float,
    *,
    proxy: float,
    icc: float,
    alpha: float,
    max_games: int,
) -> int | None:
    """First cumulative-games look where a noiseless run of ``delta`` excludes zero."""

    n_blocks = int(max_games / block_size) + 1
    sizes = np.full(n_blocks, block_size, dtype=np.float64)
    sums = sizes * delta
    trace = confidence_sequence_from_block_stats(
        sizes,
        sums,
        alpha=alpha,
        prior_variance=prior_variance,
        per_game_variance_proxy=proxy,
        intraclass_correlation=icc,
    )
    hits = trace.loc[trace["excludes_zero"]]
    return None if hits.empty else int(hits.iloc[0]["cumulative_games"])


def best_tuned_games_to_detect(
    delta: float, block_size: float, *, proxy: float, icc: float, alpha: float, max_games: int
) -> int | None:
    """The fewest games ANY mixing-variance choice needs -- the instrument's ceiling."""

    best_games: int | None = None
    for rho in np.logspace(-9, -1, 80):
        games = deterministic_games_to_detect(
            delta, block_size, float(rho), proxy=proxy, icc=icc, alpha=alpha, max_games=max_games
        )
        if games is not None and (best_games is None or games < best_games):
            best_games = games
    return best_games


def instrument_property_table(
    deltas_accuracy_points: tuple[float, ...],
    *,
    nfl_block_size: float,
    cfb_block_size: float,
    proxy: float,
    icc: float,
    alpha: float,
    max_games: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for delta_points in deltas_accuracy_points:
        delta = delta_points / 100.0
        rows.append(
            {
                "delta_accuracy_points": delta_points,
                "games_needed_nfl_scale": best_tuned_games_to_detect(
                    delta, nfl_block_size, proxy=proxy, icc=icc, alpha=alpha, max_games=max_games
                ),
                "games_needed_cfb_scale": best_tuned_games_to_detect(
                    delta, cfb_block_size, proxy=proxy, icc=icc, alpha=alpha, max_games=max_games
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. A single real-data confirmatory trajectory (self-comparison null)
# ---------------------------------------------------------------------------


def real_data_self_comparison_trace(
    paired: pd.DataFrame, *, assumed_variance_proxy: float, assumed_icc: float, alpha: float
) -> pd.DataFrame:
    """market vs a relabeled copy of itself: the true effect is exactly zero.

    Unlike the synthetic calibration study this uses REAL within-week/season
    correlation as it actually occurs in CFB data, at the cost of being a
    single trajectory rather than a rate.
    """

    grouped = paired.groupby(["season", "week"]).size().sort_index()
    block_sizes = np.array(grouped.to_numpy(), dtype=np.float64)
    block_sums = np.zeros_like(block_sizes)  # baseline minus itself is exactly 0 every game
    average_block_size = float(block_sizes.mean())
    prior_variance = default_prior_variance(
        average_block_size,
        target_games=int(block_sizes.sum()),
        per_game_variance_proxy=assumed_variance_proxy,
        intraclass_correlation=assumed_icc,
    )
    return confidence_sequence_from_block_stats(
        block_sizes,
        block_sums,
        alpha=alpha,
        prior_variance=prior_variance,
        per_game_variance_proxy=assumed_variance_proxy,
        intraclass_correlation=assumed_icc,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_CFB_PREDICTIONS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--fast", action="store_true", help="Small sample sizes for a quick smoke run."
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    calibration_universes = 300 if args.fast else 3000
    bootstrap_samples = 150 if args.fast else 400

    print(f"loading real CFB predictions: {args.predictions}")
    paired = load_real_cfb_comparison(args.predictions)
    measured = measure_real_variance(paired)
    print("measured per-game variance (market vs market_residual, clean_core):")
    print(json.dumps(measured, indent=2))

    # -- 0. Measure the intraclass correlation instead of assuming it -----
    print("\n=== intraclass correlation: measured, not assumed ===")
    icc_table = measure_cfb_intraclass_correlations(args.predictions)
    icc_table.to_csv(args.output / "intraclass_correlation_measured.csv", index=False)
    print(icc_table.to_string(index=False))
    print(
        "independently-supplied NFL measurement (not re-derived here): "
        "ICC(week) = -0.0054, 2075 games, 141 weeks, "
        "artifacts/availability_experiments/20260813T133345Z"
    )

    cfb_one_season = real_block_sizes(paired, [2024])
    cfb_three_seasons = real_block_sizes(paired, [2023, 2024, 2025])
    nfl_one_season = nfl_scale_block_sizes(1)

    print(f"\ncfb one season (2024): {len(cfb_one_season)} weeks, {sum(cfb_one_season)} games")
    print(
        f"cfb three seasons (2023-2025): {len(cfb_three_seasons)} weeks, "
        f"{sum(cfb_three_seasons)} games"
    )
    print(f"nfl one season (synthetic): {len(nfl_one_season)} weeks, {sum(nfl_one_season)} games")

    # -- 1. Calibration under peeking, at the OPERATING (independence) ICC --
    print("\n=== calibration under peeking (true effect = 0), operating ICC = 0.0 ===")
    calibration_rows = []
    calibration_configs: list[dict[str, Any]] = [
        {
            "block_sizes": cfb_one_season,
            "label": "cfb_one_season/worst_case_floor",
            "assumed_variance_proxy": 1.0,
            "assumed_icc": WORST_CASE_INTRACLASS_CORRELATION,
            "simulated_icc": OPERATING_INTRACLASS_CORRELATION,
        },
        {
            "block_sizes": cfb_one_season,
            "label": "cfb_one_season/operating (headline)",
            "assumed_variance_proxy": MEASURED_PER_GAME_VARIANCE_PROXY,
            "assumed_icc": OPERATING_INTRACLASS_CORRELATION,
            "simulated_icc": OPERATING_INTRACLASS_CORRELATION,
        },
        {
            "block_sizes": cfb_one_season,
            "label": "cfb_one_season/operating_stress_tested (sim icc=1.0)",
            "assumed_variance_proxy": MEASURED_PER_GAME_VARIANCE_PROXY,
            "assumed_icc": OPERATING_INTRACLASS_CORRELATION,
            "simulated_icc": STRESS_INTRACLASS_CORRELATION,
        },
        {
            "block_sizes": cfb_three_seasons,
            "label": "cfb_three_seasons/operating",
            "assumed_variance_proxy": MEASURED_PER_GAME_VARIANCE_PROXY,
            "assumed_icc": OPERATING_INTRACLASS_CORRELATION,
            "simulated_icc": OPERATING_INTRACLASS_CORRELATION,
        },
        {
            "block_sizes": nfl_one_season,
            "label": "nfl_one_season/operating (2026 monitoring cadence)",
            "assumed_variance_proxy": MEASURED_PER_GAME_VARIANCE_PROXY,
            "assumed_icc": OPERATING_INTRACLASS_CORRELATION,
            "simulated_icc": OPERATING_INTRACLASS_CORRELATION,
        },
    ]
    for config in calibration_configs:
        result = run_calibration_study(
            config["block_sizes"],
            label=config["label"],
            universes=calibration_universes,
            fixed_sample_bootstrap_samples=bootstrap_samples,
            alpha=args.alpha,
            assumed_variance_proxy=config["assumed_variance_proxy"],
            assumed_icc=config["assumed_icc"],
            simulated_variance=measured["per_game_variance"],
            simulated_icc=config["simulated_icc"],
            seed=args.seed,
        )
        calibration_rows.append(result)
        print(
            f"{config['label']:52s} CS false-alarm={result['cs_false_alarm_rate']:.4f}  "
            f"fixed-sample-peeked false-alarm={result['fixed_sample_false_alarm_rate']:.4f}  "
            f"({result['elapsed_seconds']:.1f}s, {result['universes']} universes, "
            f"{result['n_blocks']} weeks)"
        )
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(args.output / "calibration_under_peeking.csv", index=False)

    # -- 2. Short instrument-property appendix -----------------------------
    print("\n=== instrument property: games needed at the operating (independence) ICC ===")
    appendix = instrument_property_table(
        APPENDIX_EFFECTS_ACCURACY_POINTS,
        nfl_block_size=14.7,  # matches the independently-supplied NFL measurement's scale
        cfb_block_size=measured["average_block_size"],
        proxy=MEASURED_PER_GAME_VARIANCE_PROXY,
        icc=OPERATING_INTRACLASS_CORRELATION,
        alpha=args.alpha,
        max_games=3_000_000,
    )
    appendix.to_csv(args.output / "instrument_property_appendix.csv", index=False)
    print(appendix.to_string(index=False))

    # -- 3. Real-data confirmatory trajectory -------------------------------
    print("\n=== real-data self-comparison null (market vs itself, all clean_core weeks) ===")
    self_trace = real_data_self_comparison_trace(
        paired,
        assumed_variance_proxy=MEASURED_PER_GAME_VARIANCE_PROXY,
        assumed_icc=OPERATING_INTRACLASS_CORRELATION,
        alpha=args.alpha,
    )
    self_trace.to_csv(args.output / "real_data_self_comparison_trace.csv", index=False)
    never_excluded = not bool(self_trace["excludes_zero"].any())
    print(
        f"weeks={len(self_trace)} games={int(self_trace['cumulative_games'].iloc[-1])} "
        f"never_excludes_zero={never_excluded} "
        f"final_interval=[{self_trace['lower'].iloc[-1]:+.4f}, {self_trace['upper'].iloc[-1]:+.4f}]"
    )

    manifest = {
        "measured_real_variance": measured,
        "measured_per_game_variance_proxy_used": MEASURED_PER_GAME_VARIANCE_PROXY,
        "operating_intraclass_correlation_used": OPERATING_INTRACLASS_CORRELATION,
        "worst_case_intraclass_correlation": WORST_CASE_INTRACLASS_CORRELATION,
        "alpha": args.alpha,
        "calibration_universes": calibration_universes,
        "fixed_sample_bootstrap_samples": bootstrap_samples,
        "self_comparison_never_excludes_zero": never_excluded,
        "fast_mode": args.fast,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    print(f"\nartifacts written to {args.output}")


if __name__ == "__main__":
    main()
