"""Record the two head-to-head-vs-live-v2 comparisons from
``scripts/best_pick_nomination_v3_audit.py`` into ``registry/weak_signals.json``
via ``nfl-ats weak-signals record``.

Every numeric field (effect, interval, probability_positive, sample_games,
sample_blocks) is read directly from the audit script's ``summary.json`` and
passed through as a CLI argument -- no hand-typed numbers. Same pattern as
``scripts/record_weather_followup.py``.

Per AGENTS.md ("An interval crossing zero is NOT grounds for rejection"),
both comparisons record ``unresolved_below_power`` regardless of interval
shape: neither interval sits ENTIRELY on the wrong side of zero (the
``dispersion_filtered_candidate`` comparison's interval is entirely
non-negative but there is no admissible "confirmed positive" classification
in this registry's schema -- only ``unresolved_below_power``,
``refuted_mechanism``, or ``bounded_by_control`` -- and refuted_mechanism's
own ``wrong_sign_resolved`` ground requires the interval on the WRONG,
i.e. negative, side, which is inapplicable to a positive lean), and no
positive-control bound was run for either.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/record_best_pick_nomination_v3_audit.py \\
        --artifact artifacts/best_pick_nomination_v3_audit/<run_id>/summary.json [--replace]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _latest_artifact() -> Path:
    root = REPO / "artifacts" / "best_pick_nomination_v3_audit"
    runs = sorted(p for p in root.iterdir() if p.is_dir())
    if not runs:
        raise SystemExit(f"No runs found under {root}; run best_pick_nomination_v3_audit.py first")
    return runs[-1] / "summary.json"


_SOURCE = (
    "scripts/best_pick_nomination_v3_audit.py; artifacts/best_pick_nomination_v3_audit/"
    "{timestamp}/summary.json; docs/best_pick_ranker.md 'v3 audit'; registry/weak_signals.json:"
    "best_pick_opener_ranker_dispersion_filtered_candidate_vs_unfiltered, "
    "best_pick_opener_ranker_candidate_prob_distance_vs_status_quo"
)

_CLASSIFICATION_EVIDENCE = {
    "dispersion_filtered_candidate_vs_live_v2_production": (
        "Head-to-head audit of the open registry lead "
        "(best_pick_opener_ranker_dispersion_filtered_candidate_vs_unfiltered, "
        "+3.92 pts vs its UNFILTERED parent, P+ 0.813) against the ACTUAL live "
        "production v2 nomination rule (nfl_ats.best_pick_nomination.select_nominee: "
        "same filter, dispersion tie-break within the pool) rather than that "
        "unfiltered baseline. Survives the tie-break audit (tie-agnostic recompute "
        "moves the original lead from +3.92 to +3.43 pts, P+ 0.813 to 0.798 -- not "
        "a majority-tie artifact like sweep_robustness' collapsed +8.68). The "
        "head-to-head interval's lower bound is exactly 0.0, but per this registry's "
        "schema there is no 'confirmed positive' classification (only "
        "unresolved_below_power / refuted_mechanism / bounded_by_control), and "
        "wrong_sign_resolved requires the interval on the NEGATIVE side, which does "
        "not apply to a non-negative lean -- unresolved_below_power is the only "
        "admissible classification. IMPORTANT CAVEAT the notes field expands on: "
        "the entire +0.97-point estimate traces to exactly ONE of 103 paired weeks "
        "(2023 week 15) where the two rules' nominees differ AND settle differently; "
        "a second nominally-diverging week (2020 week 8) produced identical lift for "
        "both rules and contributes zero net difference. This is an N=1 result "
        "wearing a P+ of 0.631 -- an EV-positive forced-pick lean per AGENTS.md, not "
        "a resolution, and reported as such."
    ),
    "candidate_prob_distance_vs_live_v2_production": (
        "Head-to-head audit of the SECOND open registry lead "
        "(best_pick_opener_ranker_candidate_prob_distance_vs_status_quo, +1.24 pts "
        "vs the abs-residual status-quo chooser, P+ 0.587) against the ACTUAL live "
        "production v2 nomination rule. Removing the dispersion filter entirely "
        "(this chooser's own construction) leans NEGATIVE against live v2 (P+ "
        "0.196, i.e. live v2 is favoured ~80% of the time on this population), "
        "consistent with the filter itself being where essentially all of the "
        "measured edge over the unfiltered chooser lives. The interval's upper "
        "bound is positive (+4.90 pts), so wrong_sign_resolved does NOT apply (the "
        "whole interval is not on the negative side) -- unresolved_below_power is "
        "the only admissible classification, but the point estimate and P+ argue "
        "against ever promoting this specific candidate over what is already live."
    ),
}


def _notes_for(entry: dict[str, Any], key: str, artifact: dict[str, Any]) -> str:
    pop = artifact["population"]
    common = (
        f"Fourth reuse of the same 107 opener weeks this session-family "
        f"(ridge_alpha promotion look, odds-microstructure battery, the original "
        f"ranker screen, now this audit) -- compounding look-reuse discount. "
        f"Population: {pop['games']:,} games, {pop['weeks']} weeks, seasons "
        f"{pop['seasons'][0]}-{pop['seasons'][-1]}. Bootstrap: seed "
        f"{artifact['bootstrap_seed']}, {artifact['bootstrap_samples']:,} draws, "
        f"week-blocked, reused deliberately from the audited artifact rather than "
        f"re-chosen after seeing results. Reproduction check: recomputed frame "
        f"matched the stored artifact (artifacts/best_pick_opener_ranker/"
        f"20260818T230550Z/summary.json) exactly on top1_accuracy, n_tie_weeks, and "
        f"mean_weekly_lift for all three audited choosers before anything new was "
        f"computed. Not a rotation-registry window; no window is spent or implied."
    )
    n_diverge = entry["n_weeks_nominee_diverges"]
    if key == "dispersion_filtered_candidate_vs_live_v2_production":
        common += (
            f" {n_diverge} of {entry['n_weeks_paired']} paired weeks have a "
            "different nominee between the two rules (2020 week 8: DAL_PHI vs "
            "TEN_CIN, both nominees LOST that week -- zero net lift difference; "
            "2023 week 15: DAL_BUF (candidate, WON) vs MIN_CIN (live v2, LOST) -- "
            "the entire measured advantage)."
        )
    else:
        common += (
            f" {n_diverge} of {entry['n_weeks_paired']} paired weeks have a different nominee."
        )
    return common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=None, help="path to summary.json")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--recorded-at", default=None)
    args = parser.parse_args()

    artifact_path = args.artifact or _latest_artifact()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    timestamp = artifact_path.parent.name
    source = _SOURCE.format(timestamp=timestamp)
    pop = artifact["population"]

    descriptions = {
        "dispersion_filtered_candidate_vs_live_v2_production": (
            "dispersion_filtered_candidate Best-Pick chooser (candidate_prob_distance "
            "restricted to below-median cross-book opener spread_std, ties broken by "
            "ascending game_id -- registry lead "
            "best_pick_opener_ranker_dispersion_filtered_candidate_vs_unfiltered) vs "
            "the LIVE production v2 nomination rule (nfl_ats.best_pick_nomination."
            "select_nominee: identical filter, ties broken by lower cross-book "
            "dispersion then game_id) on the 1537-game/107-week paired opener archive"
        ),
        "candidate_prob_distance_vs_live_v2_production": (
            "candidate_prob_distance Best-Pick chooser (alpha=2000 calibrated "
            "probability distance from 0.5, UNFILTERED -- registry lead "
            "best_pick_opener_ranker_candidate_prob_distance_vs_status_quo) vs the "
            "LIVE production v2 nomination rule (same primary signal, but restricted "
            "to the below-median dispersion pool with a dispersion tie-break) on the "
            "1537-game/107-week paired opener archive"
        ),
    }
    names = {
        "dispersion_filtered_candidate_vs_live_v2_production": (
            "best_pick_opener_ranker_dispersion_filtered_candidate_vs_live_v2"
        ),
        "candidate_prob_distance_vs_live_v2_production": (
            "best_pick_opener_ranker_candidate_prob_distance_vs_live_v2"
        ),
    }

    for key, entry in artifact["head_to_head_vs_live_v2"].items():
        cmd = [
            sys.executable,
            "-m",
            "nfl_ats.cli",
            "weak-signals",
            "record",
            "--name",
            names[key],
            "--description",
            descriptions[key],
            "--source",
            source,
            "--effect",
            f"{entry['estimate'] * 100:.10f}",
            "--effect-units",
            "accuracy_points",
            "--classification",
            "unresolved_below_power",
            "--league",
            "nfl",
            "--season-start",
            str(pop["seasons"][0]),
            "--season-end",
            str(pop["seasons"][-1]),
            "--interval-low",
            f"{entry['lower'] * 100:.10f}",
            "--interval-high",
            f"{entry['upper'] * 100:.10f}",
            "--probability-positive",
            f"{entry['probability_positive']:.10f}",
            "--sample-games",
            str(pop["games"]),
            "--sample-blocks",
            str(entry["n_weeks_paired"]),
            "--classification-evidence",
            _CLASSIFICATION_EVIDENCE[key],
            "--notes",
            _notes_for(entry, key, artifact),
        ]
        if args.recorded_at:
            cmd += ["--recorded-at", args.recorded_at]
        if args.replace:
            cmd.append("--replace")

        print(f"=== recording {names[key]} ===")
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(
                f"weak-signals record failed for {names[key]} (exit "
                f"{result.returncode}); per AGENTS.md 'if a record command errors, "
                "the verdict is wrong, not the validator' -- fix the invocation, "
                "do not weaken the classification to force it through."
            )


if __name__ == "__main__":
    main()
