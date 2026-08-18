"""Out-of-screen confirmation of the frozen tau=0.05 f-lever disagreement gate (estvar).

Research item: ``docs/estimation_variance.md`` sec 5 reports that gating the CFB
"full" feature profile's influence to games where its predicted cover
probability disagrees with the "thin" profile by ``tau >= 0.05`` turns
comparison B (full vs. thin, CFB clean-core, 8,933 games) from losing
(``probability_positive`` 0.264) to strongly resolving (``P+ 0.9505``). The
doc itself flags that ``tau=0.05`` was "chosen by looking at a 9-point sweep
after seeing the pooled result" and that the qualitative pattern, not the
specific ``tau`` or ``P+`` figure, is what should be trusted without a
predeclared confirmation design. The registry entry
``estimation_variance_disagreement_gate_full_vs_thin_cfb`` records exactly
that gap: *"a predeclared tau on a train/validation split, confirmed once on
a held-out test split, is required before this could be called resolved."*

This script is that confirmation, executed per the design at
``flever_scope/design.md`` (frozen 2026-08-18) and the orchestrator's
adjudication of its five open decisions:

1. **Population is CFB-only, out-of-screen.** ``thin_2006_2011`` (2006-2011)
   + ``regime_2020`` (2020) -- the CFB seasons neither the original 9-point
   sweep (``estvar_real_cfb_audit.py::f_lever_report``) nor the split-half
   check (``estvar_f_lever_splithalf.py``) ever touched; both operate only on
   ``CFB_CLEAN_CORE_SEASONS``. Free per ``rotation_registry.md`` rule 8 (CFB
   is unreserved). NFL is explicitly NOT admitted here -- no window drawn, no
   rotation-family declared.
2. The reliability-only NFL diagnostic is deferred (out of scope for this run).
3. No ``P+`` promotion bar governs anything in this script or its output; it
   reports continuous evidence (effect, interval, ``probability_positive``,
   ``f``, ``MDE80``, reliability) for a human to read descriptively.
4. The doc/script tau-grid mismatch (9 grid values in ``f_lever_report`` vs. 8
   in ``estvar_f_lever_splithalf.py``, both excluding this script) is not
   touched here; this script only consumes the single frozen ``tau=0.05``.
5. The placebo arm (random gate, size-matched to the real gate's membership
   count) is REQUIRED, not optional -- it is the only arm that can tell "the
   disagreement mechanism is informative" apart from "any small, arbitrary
   subset looks better because a smaller ``f`` shrinks ``MDE80`` and the
   game-sampling variance of the paired estimate."

Frozen configuration (not re-swept on this population):

- ``tau = 0.05``.
- ``THIN_COLUMNS`` (11 cols: market + context + experience) vs.
  ``FULL_COLUMNS`` (35 cols: ``CFB_MODEL_FEATURE_COLUMNS``), unchanged from
  the existing screens.
- Estimator: ``refit_aware_interval`` (Part II's corrected, durable
  estimator), ``samples=20_000``, ``on_degenerate="raise"`` -- NOT the naive
  ``2_000``/``1_500``-sample estimator the exploratory scripts used.
- ``n_boot=120`` paired refit draws (``paired_refit_predicted_values``, both
  arms refit on the SAME resampled training rows each draw -- structural
  pairing, not an incidental seed match).
- The gate itself is applied INSIDE the honest interval: each refit draw's
  gated candidate probabilities are computed from that draw's own baseline
  and candidate refit probabilities, not just the point fit.

Order matters: the gate-membership reliability check (are the tau=0.05
"in"/"out" calls stable under training-row resampling?) is computed and
printed BEFORE the accuracy screen, per the design's binding order -- an
unstable gate would mean the accuracy result measures noise in WHICH games
got selected, not whether disagreement is informative.

Per this project's binding rule (AGENTS.md), an interval containing zero is
never grounds to reject; this population's own MDE80 arithmetic (~0.69
accuracy points at the screen's realized f and n, against the original
screen's own +0.246-point effect estimate) means the honest expectation
going in is a directional read, not a resolution.

Does not modify ``src/`` or any project doc. Writes JSON under
``artifacts/estvar_f_lever_confirmation/<UTC timestamp>/``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    cfb_evaluation_window,
    fit_cfb_residual_model,
)
from nfl_ats.cfb_features import (
    CFB_CONTEXT_FEATURES,
    CFB_EXPERIENCE_FEATURES,
    CFB_MARKET_FEATURES,
    CFB_MODEL_FEATURE_COLUMNS,
)
from nfl_ats.estimation_variance import (
    gate_by_disagreement,
    home_cover_probability_from_center,
    mde80,
    paired_refit_predicted_values,
    picks_differ_fraction,
    refit_aware_interval,
)
from nfl_ats.margin import MarginModel

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFB_FEATURES = REPO / "data/processed/cfb_game_features.parquet"
ARTIFACT_ROOT = REPO / "artifacts" / "estvar_f_lever_confirmation"

THIN_COLUMNS: tuple[str, ...] = (
    *CFB_MARKET_FEATURES,
    *CFB_CONTEXT_FEATURES,
    *CFB_EXPERIENCE_FEATURES,
)
FULL_COLUMNS: tuple[str, ...] = CFB_MODEL_FEATURE_COLUMNS

# --- Frozen configuration (design.md sec 4a). Declared before this script is
# --- ever run against the out-of-screen population; not re-swept afterward.
TAU = 0.05
N_BOOT = 120
RIDGE_ALPHA = CFB_BENCHMARK_RIDGE_ALPHA
MIN_TRAIN_GAMES = CFB_BENCHMARK_MIN_TRAIN_GAMES
INTERVAL_SAMPLES = 20_000
INTERVAL_SEED = 20260818
REFIT_SEED_BASE = 3  # distinct from estvar_real_cfb_audit.py's A=1, B=2
# Placebo-gate random seed, declared before results are seen (design.md sec 7
# flags this as a fresh underived choice that must be declared up front, same
# discipline as estvar_f_lever_splithalf.py's existing 20260818 seeds).
PLACEBO_SEED = 20260818

# Population: out-of-screen only. Neither the 9-point sweep
# (estvar_real_cfb_audit.py::f_lever_report) nor the split-half check
# (estvar_f_lever_splithalf.py) ever touched these seasons -- both operate on
# CFB_CLEAN_CORE_SEASONS (2012-2019, 2021-2025) exclusively. NFL is not
# admitted (orchestrator adjudication, decision 1).
OUT_OF_SCREEN_SEASONS: tuple[int, ...] = (*tuple(range(2006, 2012)), 2020)


def _load_completed(features_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(features_path)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
        & pd.to_numeric(frame["home_cover"], errors="coerce").notna()
    ].copy()
    return completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def _refit_probabilities(
    raw_refits: np.ndarray, spread: np.ndarray, residuals: np.ndarray
) -> np.ndarray:
    """Cover probability per (draw, game), looping over draws to bound memory."""

    n_boot = raw_refits.shape[0]
    out = np.empty_like(raw_refits)
    for draw in range(n_boot):
        centers = spread + raw_refits[draw]
        out[draw] = home_cover_probability_from_center(centers, spread, residuals)
    return out


@dataclass
class SeasonFit:
    season: int
    window: str
    test: pd.DataFrame
    block_ids: np.ndarray
    actual: np.ndarray
    baseline_model: MarginModel
    candidate_model: MarginModel
    baseline_prob_point: np.ndarray
    candidate_prob_point: np.ndarray
    baseline_refit_prob: np.ndarray  # (n_boot, n_test)
    candidate_refit_prob: np.ndarray  # (n_boot, n_test)


def fit_seasons(
    completed: pd.DataFrame, seasons: tuple[int, ...]
) -> tuple[list[SeasonFit], list[dict[str, Any]]]:
    """Annual-refit cadence, mirroring estvar_real_cfb_audit.py::fit_seasons.

    A season is dropped entirely (not just its early weeks) when strictly
    earlier training rows fall below ``CFB_BENCHMARK_MIN_TRAIN_GAMES`` -- the
    same rule the point-model benchmark pipeline applies. Reported explicitly
    in ``seasons_skipped`` rather than assumed away.
    """

    fits: list[SeasonFit] = []
    skipped: list[dict[str, Any]] = []
    for season in seasons:
        test_all = completed.loc[completed["season"].eq(season)]
        if test_all.empty:
            skipped.append({"season": season, "reason": "no rows"})
            continue
        cutoff = test_all["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < MIN_TRAIN_GAMES:
            skipped.append(
                {
                    "season": season,
                    "reason": "below training floor",
                    "train_rows": len(training),
                    "min_train_games": MIN_TRAIN_GAMES,
                }
            )
            continue

        test = test_all.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        keep = pd.to_numeric(test["home_cover"], errors="coerce").notna()
        test = test.loc[keep].reset_index(drop=True)

        spread = pd.to_numeric(test["spread_line"], errors="coerce").to_numpy(dtype=float)
        actual = pd.to_numeric(test["home_cover"], errors="raise").to_numpy(dtype=float)

        baseline_model = fit_cfb_residual_model(
            training, ridge_alpha=RIDGE_ALPHA, feature_columns=THIN_COLUMNS
        )
        candidate_model = fit_cfb_residual_model(
            training, ridge_alpha=RIDGE_ALPHA, feature_columns=FULL_COLUMNS
        )
        baseline_prob_point = baseline_model.predict(test)["home_cover_probability"].to_numpy(
            dtype=float
        )
        candidate_prob_point = candidate_model.predict(test)["home_cover_probability"].to_numpy(
            dtype=float
        )

        refits = paired_refit_predicted_values(
            training,
            test,
            baseline_feature_columns=THIN_COLUMNS,
            candidate_feature_columns=FULL_COLUMNS,
            target_column="ats_margin",
            ridge_alpha=RIDGE_ALPHA,
            n_boot=N_BOOT,
            seed=REFIT_SEED_BASE + season,
            paired=True,
        )
        baseline_refit_prob = _refit_probabilities(
            refits.baseline, spread, baseline_model.residuals
        )
        candidate_refit_prob = _refit_probabilities(
            refits.candidate, spread, candidate_model.residuals
        )
        block_ids = season * 100 + test["week"].to_numpy(dtype=np.int64)
        window = cfb_evaluation_window(season)

        fits.append(
            SeasonFit(
                season=season,
                window=window,
                test=test,
                block_ids=block_ids,
                actual=actual,
                baseline_model=baseline_model,
                candidate_model=candidate_model,
                baseline_prob_point=baseline_prob_point,
                candidate_prob_point=candidate_prob_point,
                baseline_refit_prob=baseline_refit_prob,
                candidate_refit_prob=candidate_refit_prob,
            )
        )
        print(
            f"  season {season} [{window}]: train={baseline_model.training_rows} test={len(test)}",
            flush=True,
        )
    return fits, skipped


def pooled_arrays(fits: list[SeasonFit]) -> dict[str, np.ndarray]:
    return {
        "actual": np.concatenate([f.actual for f in fits]),
        "block_ids": np.concatenate([f.block_ids for f in fits]),
        "baseline_prob": np.concatenate([f.baseline_prob_point for f in fits]),
        "candidate_prob": np.concatenate([f.candidate_prob_point for f in fits]),
        "baseline_refit_prob": np.concatenate([f.baseline_refit_prob for f in fits], axis=1),
        "candidate_refit_prob": np.concatenate([f.candidate_refit_prob for f in fits], axis=1),
    }


def gate_membership_reliability(arrays: dict[str, np.ndarray], *, tau: float) -> dict[str, Any]:
    """Gate-membership stability under refit resampling (design.md sec 3).

    Not a standard split-half correlation (the gate is a per-game binary
    label, not a repeated measurement of one persistent trait): for each of
    the ``n_boot`` paired refit draws, recompute which games clear the
    ``tau`` disagreement boundary and compare that draw's membership mask to
    the POINT fit's membership mask. Reported BEFORE the accuracy screen.
    """

    baseline_point = arrays["baseline_prob"]
    candidate_point = arrays["candidate_prob"]
    point_membership = np.abs(candidate_point - baseline_point) >= tau

    baseline_refit = arrays["baseline_refit_prob"]
    candidate_refit = arrays["candidate_refit_prob"]
    draw_membership = np.abs(candidate_refit - baseline_refit) >= tau

    flips = draw_membership != point_membership[np.newaxis, :]
    in_mask = point_membership
    out_mask = ~point_membership
    return {
        "tau": tau,
        "n_games": len(point_membership),
        "n_boot": int(baseline_refit.shape[0]),
        "point_membership_count": int(point_membership.sum()),
        "point_membership_rate": float(np.mean(point_membership)),
        "overall_flip_rate": float(np.mean(flips)),
        "in_gate_flip_rate": (float(np.mean(flips[:, in_mask])) if in_mask.any() else None),
        "out_gate_flip_rate": (float(np.mean(flips[:, out_mask])) if out_mask.any() else None),
    }


def _gate_refits_by_disagreement(
    baseline_refit: np.ndarray, candidate_refit: np.ndarray, *, threshold: float
) -> np.ndarray:
    """Apply the real disagreement gate to EVERY refit draw, not just the point fit."""

    out = np.empty_like(candidate_refit)
    for draw in range(candidate_refit.shape[0]):
        out[draw] = gate_by_disagreement(
            baseline_refit[draw], candidate_refit[draw], threshold=threshold
        )
    return out


def _placebo_mask(n_games: int, *, count: int, seed: int) -> np.ndarray:
    """A single fixed-seed random subset of games, size-matched to the real gate.

    Chosen ONCE (not per refit draw) and applied identically to the point fit
    and to every refit draw, mirroring the real gate's own game-level
    deferral structure but with membership decided by a random draw instead
    of disagreement magnitude.
    """

    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_games, size=count, replace=False)
    mask = np.zeros(n_games, dtype=bool)
    mask[chosen] = True
    return mask


def _gate_refits_by_mask(
    baseline_refit: np.ndarray, candidate_refit: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    return np.where(mask[np.newaxis, :], candidate_refit, baseline_refit)


def accuracy_screen(
    arrays: dict[str, np.ndarray], *, tau: float, placebo_seed: int
) -> dict[str, Any]:
    """The frozen confirmatory screen: real gate vs. its size-matched placebo control."""

    actual = arrays["actual"]
    block_ids = arrays["block_ids"]
    baseline_point = arrays["baseline_prob"]
    candidate_point = arrays["candidate_prob"]
    baseline_refit = arrays["baseline_refit_prob"]
    candidate_refit = arrays["candidate_refit_prob"]
    n = len(actual)
    n_blocks = len(np.unique(block_ids))

    membership_count = int(np.sum(np.abs(candidate_point - baseline_point) >= tau))

    real_gate_point = gate_by_disagreement(baseline_point, candidate_point, threshold=tau)
    real_gate_refit = _gate_refits_by_disagreement(baseline_refit, candidate_refit, threshold=tau)
    f_real = picks_differ_fraction(baseline_point, real_gate_point)
    real_result = refit_aware_interval(
        actual,
        baseline_refit,
        real_gate_refit,
        block_ids,
        point_baseline_prob=baseline_point,
        point_candidate_prob=real_gate_point,
        samples=INTERVAL_SAMPLES,
        seed=INTERVAL_SEED,
        on_degenerate="raise",
    )

    mask = _placebo_mask(n, count=membership_count, seed=placebo_seed)
    placebo_point = np.where(mask, candidate_point, baseline_point)
    placebo_refit = _gate_refits_by_mask(baseline_refit, candidate_refit, mask)
    f_placebo = picks_differ_fraction(baseline_point, placebo_point)
    placebo_result = refit_aware_interval(
        actual,
        baseline_refit,
        placebo_refit,
        block_ids,
        point_baseline_prob=baseline_point,
        point_candidate_prob=placebo_point,
        samples=INTERVAL_SAMPLES,
        seed=INTERVAL_SEED,
        on_degenerate="raise",
    )

    return {
        "tau": tau,
        "n_games": n,
        "n_blocks": n_blocks,
        "membership_count": membership_count,
        "membership_rate": membership_count / n,
        "real_gate": {
            "f": f_real,
            "mde80": mde80(max(f_real, 1e-9), n),
            "honest": vars(real_result.honest),
            "naive": vars(real_result.naive),
            "decomposition": vars(real_result.decomposition),
        },
        "placebo_gate": {
            "seed": placebo_seed,
            "f": f_placebo,
            "mde80": mde80(max(f_placebo, 1e-9), n),
            "honest": vars(placebo_result.honest),
            "naive": vars(placebo_result.naive),
            "decomposition": vars(placebo_result.decomposition),
        },
    }


def main() -> None:
    started = time.time()
    completed = _load_completed(DEFAULT_CFB_FEATURES)
    print(
        f"Loaded {len(completed)} completed CFB games (result/ats_margin/home_cover all non-null)",
        flush=True,
    )

    fits, skipped = fit_seasons(completed, OUT_OF_SCREEN_SEASONS)
    for entry in skipped:
        print(f"  season {entry['season']}: SKIPPED ({entry['reason']})", flush=True)

    arrays = pooled_arrays(fits)
    n_games = len(arrays["actual"])
    n_blocks = len(np.unique(arrays["block_ids"]))
    print(f"Realized out-of-screen population: {n_games} games, {n_blocks} week-blocks", flush=True)

    results: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "design": "flever_scope/design.md (2026-08-18 f-lever confirmation)",
        "tau": TAU,
        "n_boot": N_BOOT,
        "interval_samples": INTERVAL_SAMPLES,
        "seasons_requested": list(OUT_OF_SCREEN_SEASONS),
        "seasons_skipped": skipped,
        "seasons_kept": [f.season for f in fits],
        "windows_kept": sorted({f.window for f in fits}),
        "n_games_realized": n_games,
        "n_blocks_realized": n_blocks,
    }

    # --- PHASE 1: reliability, computed and reported BEFORE the accuracy screen.
    print(
        "\n=== Gate-membership reliability (read this BEFORE the accuracy screen) ===", flush=True
    )
    reliability = gate_membership_reliability(arrays, tau=TAU)
    results["reliability"] = reliability
    print(json.dumps(reliability, indent=2), flush=True)

    # --- PHASE 2: frozen accuracy screen (real gate + required placebo control).
    print("\n=== Accuracy screen: real gate vs. placebo (tau frozen, no re-sweep) ===", flush=True)
    screen = accuracy_screen(arrays, tau=TAU, placebo_seed=PLACEBO_SEED)
    results["accuracy_screen"] = screen
    print(json.dumps(screen, indent=2, default=str), flush=True)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path} in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
