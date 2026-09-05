"""Validate the GATING RECIPE (not the injury candidate) on CFB, free and unlimited.

Research item: ``docs/surgical_injury.md``. ``docs/injury_value_lost.md``
section 5 already establishes, by direct data audit, that CFB carries no
pregame injury/availability signal at all -- so the injury candidate itself
cannot be screened there, not weakly, not as a proxy. What CAN be tested on
CFB is the RECIPE: "gate a sparse/lumpy candidate's influence to the games
where its own construct fired materially, using a threshold derived from the
construct's distribution alone." That is a general claim about HOW to use a
sparse feature, independent of which league or which sparse feature it is
applied to, and CFB has a real, already-scored, already-negative comparison
built on a genuinely sparse/lumpy construct to test it against:
``cfb_role_continuity`` (``docs/cfb_role_features.md``, status
``closed_negative`` in ``registry/rotation_registry.json`` -- closed at the
CFB benchmark, no NFL window ever drawn, so re-reading it spends nothing and
touches no NFL ledger).

The analog construct: role-continuity ``absent_mass`` (share-weighted mass of
previously-established role holders who did not show up this game;
``src/nfl_ats/cfb_role_features.py:462-472``) -- sparse (55.7% of games
exactly zero) and lumpy (long right tail) in the same shape as the NFL
injury-value-lost magnitude this recipe was built for. The per-game gating
covariate here is the SUM of ``absent_mass`` across both teams and both
action types (dropback + carry), read straight from the already-saved
``role_continuity.parquet`` -- deliberately the total, not a home-minus-away
differential, because reconstructing the home/away split requires a
play-by-play team-id join (``attach_role_continuity``) not available from the
saved artifact alone, and the total is the natural single-scalar analog of
"how much established role mass is absent from this game" the way the NFL
side reads "how much value is absent", not "which side has more absence".

The baseline/candidate arms are the ALREADY-SCORED, ALREADY-SAVED predictions
from ``artifacts/cfb_role_experiments/20260817T110541Z/predictions.parquet``
(``market_residual`` vs ``market_residual_roles``, clean_core evaluation
window, walk-forward, exactly what produced the closed_negative verdict). No
model is refit here. This script only re-derives the gating threshold from
``role_continuity.parquet``'s own full distribution (the same free,
descriptive-statistic category as the NFL derivation) and re-scores the
ALREADY-COMPUTED probabilities under the gate.

Because CFB is free and unlimited (rotation_registry.md rule 8: "CFB and
non-reserved seasons stay free"), this script DOES compute and report
accuracy for both the ungated and gated comparisons -- unlike the NFL side,
where a gated variant's accuracy may not be measured on the one window
available. This is the intended asymmetry: CFB is where the recipe itself
gets to be tested against real outcomes; NFL is where the specific candidate
is predeclared but not yet scored.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/surgical_cfb_recipe_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.estimation_variance import mde80, naive_block_bootstrap_interval, picks_differ_fraction
from nfl_ats.surgical_gating import (
    derive_conditional_median_threshold,
    gate_active_fraction,
    gate_by_value_lost_magnitude,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO / "artifacts/cfb_role_experiments/20260817T110541Z"
BASELINE_METHOD = "market_residual"
CANDIDATE_METHOD = "market_residual_roles"
SEED = 20260818
SAMPLES = 4000


def derive_cfb_threshold(role_continuity: pd.DataFrame) -> dict[str, Any]:
    """Same rule as the NFL side: conditional median of nonzero per-game magnitude.

    Computed on role_continuity.parquet's OWN full population (8,951 games,
    2013-2025 -- the whole span the feature is ever defined for), not
    restricted to the accuracy-evaluation slice below, mirroring how the NFL
    threshold was derived from that feature's own full table rather than
    from the confirmation window.
    """

    per_game = role_continuity.groupby("game_id")["absent_mass"].sum()
    unconditional_median = float(per_game.median())
    threshold = derive_conditional_median_threshold(per_game)
    return {
        "games": len(per_game),
        "fraction_exactly_zero": float((per_game == 0.0).mean()),
        "unconditional_median": unconditional_median,
        "unconditional_median_collides_with_zero_mass": unconditional_median == 0.0,
        "conditional_median_threshold": threshold,
        "per_game_series": per_game,
    }


READ_ONLY_SCRIPT = True
# ENG-29: read-only; the ENG-29 scanner confirms zero write sites -- it re-derives a gating
# threshold from an already-saved feature table and re-scores already-computed predictions,
# printing text to stdout only.


def main() -> None:
    predictions = pd.read_parquet(ARTIFACT_DIR / "predictions.parquet")
    role_continuity = pd.read_parquet(ARTIFACT_DIR / "role_continuity.parquet")

    derivation = derive_cfb_threshold(role_continuity)
    threshold = derivation["conditional_median_threshold"]
    per_game_magnitude = derivation.pop("per_game_series")

    clean_core = predictions.loc[predictions["evaluation_window"] == "clean_core"].copy()
    wide = clean_core.pivot_table(
        index=["game_id", "season", "week"],
        columns="method",
        values="home_cover_probability",
    ).reset_index()
    actual = (
        clean_core.drop_duplicates("game_id")
        .set_index("game_id")["home_cover"]
        .reindex(wide["game_id"])
    )
    wide["actual"] = actual.to_numpy()
    wide = wide.dropna(subset=["actual", BASELINE_METHOD, CANDIDATE_METHOD]).reset_index(drop=True)

    wide["magnitude"] = (
        wide["game_id"].map(per_game_magnitude).fillna(0.0).to_numpy()
    )  # left-join semantics: no role data (e.g. 2012) == no disruption == 0, matching
    # the CONTINUITY_NEUTRAL imputation the underlying model run already used.

    actual_arr = wide["actual"].to_numpy(dtype=float)
    baseline_prob = wide[BASELINE_METHOD].to_numpy(dtype=float)
    candidate_prob = wide[CANDIDATE_METHOD].to_numpy(dtype=float)
    magnitude = wide["magnitude"].to_numpy(dtype=float)
    block_ids = (wide["season"].astype(int) * 100 + wide["week"].astype(int)).to_numpy()
    n = len(wide)

    gated_prob = gate_by_value_lost_magnitude(
        baseline_prob, candidate_prob, magnitude, threshold=threshold
    )

    f_ungated = picks_differ_fraction(baseline_prob, candidate_prob)
    f_gated = picks_differ_fraction(baseline_prob, gated_prob)

    ungated_interval = naive_block_bootstrap_interval(
        actual_arr, baseline_prob, candidate_prob, block_ids, samples=SAMPLES, seed=SEED
    )
    gated_interval = naive_block_bootstrap_interval(
        actual_arr, baseline_prob, gated_prob, block_ids, samples=SAMPLES, seed=SEED
    )
    # The sharper question: does gating improve on the ungated candidate
    # directly (not "is the gated form good", but "did gating help at all")?
    # Both arms still crossed the SAME baseline, so this is a fair paired
    # comparison, not a derived difference-of-differences.
    gate_benefit_interval = naive_block_bootstrap_interval(
        actual_arr, candidate_prob, gated_prob, block_ids, samples=SAMPLES, seed=SEED
    )

    baseline_acc = float(((baseline_prob >= 0.5) == actual_arr).mean())
    candidate_acc = float(((candidate_prob >= 0.5) == actual_arr).mean())
    gated_acc = float(((gated_prob >= 0.5) == actual_arr).mean())

    disagree_mask = (baseline_prob >= 0.5) != (candidate_prob >= 0.5)
    retained = int(np.sum(disagree_mask & (magnitude >= threshold)))
    dropped = int(np.sum(disagree_mask & (magnitude < threshold)))

    report: dict[str, Any] = {
        "note": (
            "Recipe validation only. This is NOT a measurement of the NFL injury "
            "candidate -- CFB carries no pregame injury signal (docs/injury_value_lost.md "
            "section 5). It tests whether magnitude-gating a sparse/lumpy candidate against "
            "an ALREADY-CLOSED-NEGATIVE comparison changes its resolvability, using the "
            "already-saved market_residual vs market_residual_roles predictions. Free per "
            "rotation_registry.md rule 8 (CFB needs no registry entry) and cfb_role_continuity's "
            "own closed_negative status (no NFL window ever drawn on this family)."
        ),
        "baseline_method": BASELINE_METHOD,
        "candidate_method": CANDIDATE_METHOD,
        "games": n,
        "threshold_derivation": derivation,
        "gate_active_fraction_on_evaluation_set": gate_active_fraction(
            magnitude, threshold=threshold
        ),
        "f_ungated": f_ungated,
        "f_gated": f_gated,
        "f_reduction_factor": (f_ungated / f_gated) if f_gated > 0 else float("inf"),
        "mde80_ungated": mde80(f_ungated, n),
        "mde80_gated": mde80(f_gated, n) if f_gated > 0 else 0.0,
        "disagreements_total": int(disagree_mask.sum()),
        "disagreements_retained_by_gate": retained,
        "disagreements_dropped_by_gate": dropped,
        "accuracy": {
            "baseline": baseline_acc,
            "candidate_ungated": candidate_acc,
            "candidate_gated": gated_acc,
            "delta_ungated_pts": (candidate_acc - baseline_acc) * 100.0,
            "delta_gated_pts": (gated_acc - baseline_acc) * 100.0,
        },
        "ungated_interval": {
            "estimate_pts": ungated_interval.estimate * 100.0,
            "lower_pts": ungated_interval.lower * 100.0,
            "upper_pts": ungated_interval.upper * 100.0,
            "probability_positive": ungated_interval.probability_positive,
        },
        "gated_interval": {
            "estimate_pts": gated_interval.estimate * 100.0,
            "lower_pts": gated_interval.lower * 100.0,
            "upper_pts": gated_interval.upper * 100.0,
            "probability_positive": gated_interval.probability_positive,
        },
        "gate_benefit_interval_vs_ungated_candidate": {
            "estimate_pts": gate_benefit_interval.estimate * 100.0,
            "lower_pts": gate_benefit_interval.lower * 100.0,
            "upper_pts": gate_benefit_interval.upper * 100.0,
            "probability_positive": gate_benefit_interval.probability_positive,
        },
        "bootstrap_samples": SAMPLES,
        "bootstrap_seed": SEED,
    }
    text = json.dumps(report, indent=2)
    print(text)


if __name__ == "__main__":
    main()
