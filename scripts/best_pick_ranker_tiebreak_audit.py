"""Tier-2 re-read of ``best_pick_ranker_opener`` (Docs revisit list, 2026-08-18).

This is a RE-READ, not a re-run: it recomputes statistics from the already
-stored per-pick parquet artifacts under ``artifacts/best_pick_ranker/``
(``opener_2020_2021.picks.parquet`` and ``screen_2013_2015.picks.parquet``).
No model is refit, no rotation window is touched, no registry file is
written -- consistent with ``docs/revisit_list.md`` Tier 2's constraint.

It answers the four questions the re-read was scoped to:

1. How many of the confirmation/screen weeks were ``sweep_robustness`` ties,
   and what is top-1 accuracy under the recorded alphabetical tie-break vs. a
   tie-break-free (average-over-tied-candidates) estimate?
2. A D2-style sensitivity check: how does ``probability_positive`` move if
   the naive week-blocked bootstrap's width is widened by the empirical
   inflation factors that were on record in ``docs/estimation_variance.md``
   when this audit was written (1.037x-1.575x, the "17-58% too narrow"
   headline)? **That headline is retracted.** Part II of that document found
   it double-counted a training-by-game interaction the game bootstrap
   already carries; the honest refit factor is **1.003x, one-sided 95% upper
   bound 1.099x** -- below every value swept here, including the low end
   (1.037x). The real defect behind the original under-coverage measurement
   was **D4 (too few blocks)**, not D2 width. The sweep below is left
   UNCHANGED as a sensitivity bound (it still shows ``probability_positive``
   barely moves even at inflation factors far more generous than the honest
   one), and its conclusion is, if anything, strengthened by the correction:
   D2 was never going to rescue or sink the tie-break-agnostic delta, and now
   we know by how much it doesn't matter. This mirrors
   ``clv.week_blocked_bootstrap``'s own resampling algorithm (blocks = weeks,
   metric recomputed on resampled weeks) but does not refit any model --
   refitting would require re-running the walk-forward ridge fit, which
   Tier 2 explicitly disallows.
3. Paired power arithmetic: the SE of a top-1 rate over ``W`` weeks
   (``sqrt(0.25 / W)``), how many SE the recorded delta represents, and what
   a small pick-level swing does to the headline number.

Run::

    .\\.tools\\uv.exe run --no-sync python scripts/best_pick_ranker_tiebreak_audit.py

Write-up: ``docs/best_pick_ranker.md`` (see the "Tier-2 re-read" section).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "best_pick_ranker"

# Empirical width-inflation factors as first measured in
# docs/estimation_variance.md Part I (naive-vs-honest, refit-aware bootstrap):
# two synthetic ground-truth DGPs (null: 1.575x, real effect: 1.176x) and two
# real CFB comparisons (A, large disagreement fraction f: 1.037x; B, small f:
# 1.330x). That measured range, 1.037x-1.575x ("17-58% too narrow"), is RETRACTED
# by Part II of the same document (2026-08-18): it double-counted a
# training-by-game interaction the game bootstrap already carries. The honest
# refit factor is 1.003x, one-sided 95% upper bound 1.099x -- the real defect
# was D4 (too few blocks), not D2. These four factors are kept exactly as
# measured, unchanged, so the sweep below still functions as a sensitivity
# bound: even the most generous of them (1.575x, the retracted ceiling) barely
# moves probability_positive, so the corrected, much smaller 1.003x-1.099x
# factor moves it even less.
D2_INFLATION_FACTORS = {
    "min_measured_1.037x_comparisonA_large_f": 1.037,
    "headline_floor_1.17x_17pct": 1.17,
    "comparisonB_1.330x": 1.330,
    "headline_ceiling_1.575x_58pct_null_dgp": 1.575,
}


def _week_blocked_bootstrap_of_mean(
    weekly_values: np.ndarray, samples: int, seed: int
) -> np.ndarray:
    """Resample weeks with replacement; return the distribution of the mean.

    Mirrors ``nfl_ats.clv.week_blocked_bootstrap``'s block-by-week resampling
    algorithm (draw block indices with a seeded RNG, recompute the metric on
    the resampled weeks), specialised to a single one-number-per-week series.
    """

    n = len(weekly_values)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        draws[i] = weekly_values[idx].mean()
    return draws


def analyse(
    path: Path, label: str, *, samples: int, seed: int, mc_draws: int = 5000
) -> dict[str, Any]:
    picks = pd.read_parquet(path)
    all_pick_acc = float(picks["correct"].mean())

    weekly_rows = []
    tied_correct_arrays: list[np.ndarray] = []
    for (season, week), grp in picks.groupby(["season", "week"], sort=True):
        top_score = grp["sweep_robustness"].max()
        tied = grp.loc[grp["sweep_robustness"] == top_score]
        alpha_pick = tied.sort_values("game_id").iloc[0]
        tied_correct_arrays.append(tied["correct"].to_numpy())
        weekly_rows.append(
            {
                "season": int(str(season)),
                "week": int(str(week)),
                "n_candidates": len(grp),
                "n_tied_at_max": len(tied),
                "max_score": float(top_score),
                "at_sweep_ceiling_8pt": bool(top_score >= 8.0),
                "alpha_correct": float(alpha_pick["correct"]),
                "tie_agnostic_mean_correct": float(tied["correct"].mean()),
            }
        )
    weekly_df = pd.DataFrame(weekly_rows)
    n_weeks = len(weekly_df)
    n_ties = int((weekly_df["n_tied_at_max"] > 1).sum())

    # Monte Carlo over uniformly-random tie-breaks: for each of `mc_draws`
    # replicate "seasons", draw one nomination per week uniformly among that
    # week's tied maximum, and record the resulting season top-1 accuracy.
    # This places the recorded alphabetical result within the distribution of
    # accuracies an arbitrary (but not alphabetical) tie-break could have
    # produced -- how much of the recorded number is tie-break luck.
    mc_rng = np.random.default_rng(seed + 1)
    mc_accs = np.empty(mc_draws)
    for d in range(mc_draws):
        total = 0.0
        for arr in tied_correct_arrays:
            total += arr[mc_rng.integers(0, len(arr))]
        mc_accs[d] = total / n_weeks

    alpha_acc = float(weekly_df["alpha_correct"].mean())
    tie_agnostic_acc = float(weekly_df["tie_agnostic_mean_correct"].mean())

    alpha_draws = _week_blocked_bootstrap_of_mean(
        weekly_df["alpha_correct"].to_numpy(), samples, seed
    )
    agn_draws = _week_blocked_bootstrap_of_mean(
        weekly_df["tie_agnostic_mean_correct"].to_numpy(), samples, seed
    )

    def summarize(draws: np.ndarray) -> dict[str, float]:
        delta_draws = draws - all_pick_acc
        lo, hi = np.quantile(delta_draws, [0.025, 0.975])
        return {
            "lower": float(lo),
            "upper": float(hi),
            "probability_positive": float((delta_draws > 0).mean()),
        }

    n_correct = round(alpha_acc * n_weeks)
    se_top1 = float(np.sqrt(0.25 / n_weeks))

    return {
        "label": label,
        "source": str(path.relative_to(REPO)),
        "n_weeks": n_weeks,
        "n_resolved_picks": len(picks),
        "n_ties": n_ties,
        "tie_fraction": n_ties / n_weeks,
        "n_at_sweep_ceiling_8pt": int(weekly_df["at_sweep_ceiling_8pt"].sum()),
        "all_pick_acc": all_pick_acc,
        "recorded_alphabetical_top1_acc": alpha_acc,
        "recorded_delta": alpha_acc - all_pick_acc,
        "tie_break_agnostic_top1_acc": tie_agnostic_acc,
        "tie_agnostic_delta": tie_agnostic_acc - all_pick_acc,
        "week_blocked_bootstrap_recompute": {
            "alphabetical": summarize(alpha_draws),
            "tie_agnostic": summarize(agn_draws),
        },
        "monte_carlo_random_tiebreak": {
            "draws": mc_draws,
            "mean_acc": float(mc_accs.mean()),
            "p05_acc": float(np.percentile(mc_accs, 5)),
            "p95_acc": float(np.percentile(mc_accs, 95)),
            "recorded_alphabetical_percentile": float((mc_accs <= alpha_acc).mean()),
        },
        "power": {
            "se_top1_over_n_weeks": se_top1,
            "recorded_delta_in_se_units": (alpha_acc - all_pick_acc) / se_top1,
            "n_correct_top1_recorded": n_correct,
            "3_pick_swing_new_top1_acc": (n_correct - 3) / n_weeks,
            "3_pick_swing_new_delta": (n_correct - 3) / n_weeks - all_pick_acc,
        },
    }


def d2_sensitivity(naive_lower: float, naive_upper: float, estimate: float) -> dict[str, Any]:
    """D2 width-inflation sensitivity under a normal approximation.

    Backs out an implied SE from the naive interval (assuming approximate
    normality of the bootstrap distribution), then scales that SE by each of
    ``D2_INFLATION_FACTORS`` and recomputes ``probability_positive``. This is
    a sensitivity check, not a measurement: a real refit-aware bootstrap can
    also move the point estimate (docs/estimation_variance.md section 3 shows
    this happening on its comparison B), which holding the estimate fixed
    here cannot capture.
    """

    naive_se = (naive_upper - naive_lower) / (2 * 1.959964)
    out: dict[str, Any] = {
        "estimate": estimate,
        "naive_se_implied": naive_se,
        "naive_probability_positive_recomputed": float(norm.cdf(estimate / naive_se)),
    }
    for name, factor in D2_INFLATION_FACTORS.items():
        honest_se = naive_se * factor
        out[name] = {
            "honest_se": honest_se,
            "honest_probability_positive": float(norm.cdf(estimate / honest_se)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--out", type=Path, default=ART / "tiebreak_reread_20260818.json")
    args = parser.parse_args()

    opener = analyse(
        ART / "opener_2020_2021.picks.parquet",
        "opener confirmation ([2020, 2021])",
        samples=args.samples,
        seed=args.seed,
    )
    screen = analyse(
        ART / "screen_2013_2015.picks.parquet",
        "nflverse-spread screen ([2013, 2015])",
        samples=args.samples,
        seed=args.seed,
    )

    # D2 sensitivity applied to the RECORDED opener-confirmation numbers, read
    # from artifacts/best_pick_ranker/opener_2020_2021.json: estimate
    # 0.08684210526315783, bootstrap_lower -0.06997804357245584,
    # bootstrap_upper 0.2288232557466831.
    d2_recorded = d2_sensitivity(
        naive_lower=-0.06997804357245584,
        naive_upper=0.2288232557466831,
        estimate=0.08684210526315783,
    )
    # Same sensitivity applied to the honest tie-break-agnostic delta
    # recomputed above -- the actually-defensible estimate once both defects
    # (D2 narrowness and the alphabetical artifact) are corrected together.
    tba = opener["week_blocked_bootstrap_recompute"]["tie_agnostic"]
    d2_tie_agnostic = d2_sensitivity(
        naive_lower=tba["lower"], naive_upper=tba["upper"], estimate=opener["tie_agnostic_delta"]
    )

    results = {
        "opener_2020_2021": opener,
        "screen_2013_2015": screen,
        "d2_sensitivity_recorded_alphabetical": d2_recorded,
        "d2_sensitivity_tie_break_agnostic": d2_tie_agnostic,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
