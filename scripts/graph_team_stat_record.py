"""Record the graph `team_stat` screen's results to the weak-signal registry.

Reads the screen's own artifact rather than console output, on purpose: the
input screen's docstring records that hand-copying console numbers into the
registry produced a 100x scaling bug, a sign bug and a corrupted source path in
one session. Recording goes through the same validated ``record_signal`` the
CLI calls, so the registry's admissibility checks apply unchanged.

Closing-grounds taxonomy (binding, restated per AGENTS.md so this file stands
on its own): an interval containing zero is NEVER grounds to reject, fail or
close an experiment. Only a RESOLVED wrong sign (whole interval on the wrong
side of zero), zero split-half reliability, or a positive control proven able
to detect an effect that size closes a line of work. Everything else is
``unresolved_below_power``: record it and report ``probability_positive``,
never the binary "contains zero".

The screen is CLOSE-graded, so nothing it produces may settle a play/no-play
decision; that is reserved for the opener grade.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from graph_input_screen import FAMILY_CATEGORY  # noqa: E402

from nfl_ats.weak_signals import (  # noqa: E402
    WeakSignal,
    default_registry_path,
    load_registry,
    record_signal,
    save_registry,
)

SIGNAL_PREFIX = "graph_team_stat_"
SIGNAL_FAMILY = "graph_ratings_v2_team_stat"


def classify(summary: dict[str, Any], grade: str) -> tuple[str, str | None, str]:
    """Terminal only on a RESOLVED wrong sign AT THE GRADE THAT DECIDES.

    The predeclaration (docs/graph_ratings_v2_screen.md section 6) reserves any
    terminal classification for the OPENER grade, because the pool settles at
    the opener and a close-graded screen may not settle a play/no-play
    decision. A close-graded run therefore records every family as
    ``unresolved_below_power`` even when its own interval is resolved -- the
    resolved shape is reported as continuous evidence in the notes instead.

    An interval that merely CONTAINS zero is never a closure at any grade: it
    is the expected shape for a real small signal at this evaluator's ~2-point
    resolution.
    """

    upper = summary["week_blocked_ci95"][1]
    resolved_below = upper < 0.0
    if grade != "opener":
        evidence = ""
        return "unresolved_below_power", None, evidence
    if resolved_below:
        return (
            "refuted_mechanism",
            "wrong_sign_resolved",
            (
                "Opener-graded: the week-blocked 95% CI "
                f"{summary['week_blocked_ci95']} for the paired "
                "treatment-minus-control accuracy delta sits entirely below zero, so "
                "the graph-propagated form of this statistic is resolved WORSE than "
                "its own raw differential."
            ),
        )
    return "unresolved_below_power", None, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="path to the screen's results.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be recorded without writing the registry",
    )
    args = parser.parse_args()

    artifact = Path(args.artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("mode") != "screen":
        raise SystemExit(f"refusing to record a '{payload.get('mode')}' run; screen mode only")

    seasons = payload["seasons"]
    source = artifact.resolve().relative_to(REPO_ROOT).as_posix()
    recorded_at = time.strftime("%Y-%m-%d", time.gmtime())

    registry_path = default_registry_path()
    registry = load_registry(registry_path)
    recorded = 0

    for name, entry in sorted(payload["results"].items()):
        if entry.get("status") != "scored" or not entry.get("treatment_vs_control"):
            print(f"skipped {name}: {entry.get('status')}")
            continue
        primary = entry["treatment_vs_control"]
        null = entry.get("permutation_null_treatment_vs_control") or {}
        baseline_arm = entry.get("treatment_vs_baseline") or {}
        control_arm = entry.get("control_vs_baseline") or {}
        classification, closing_ground, evidence = classify(primary, payload["grade"])

        notes = (
            f"Paired treatment-minus-control on {primary['n_games']} games, "
            f"{primary['n_weeks']} weeks. Graph arm accuracy "
            f"{primary['candidate_accuracy'] * 100:.2f}% vs raw-differential control "
            f"{primary['reference_accuracy'] * 100:.2f}%. Season-blocked P+ "
            f"{primary['season_blocked_probability_positive']:.3f}. "
            f"Against the market baseline the graph arm is "
            f"{baseline_arm.get('delta_accuracy', float('nan')) * 100:+.3f} pts and the raw "
            f"control {control_arm.get('delta_accuracy', float('nan')) * 100:+.3f} pts. "
            "Conservative reference: within-week permutation null "
            f"({null.get('permutations', 0)} permutations) centres at "
            f"{null.get('null_mean_delta', float('nan')) * 100:+.3f} pts with SD "
            f"{null.get('null_sd_delta', float('nan')) * 100:.3f}, so the observed delta sits "
            f"at the {null.get('fraction_of_null_below_observed', float('nan')) * 100:.1f}th "
            "percentile of its own null; that null is not zero-centred by design "
            "(it preserves each week's home-cover rate) -- see "
            "docs/graph_ratings_v2_screen.md section 6. "
            + (
                "Interval resolved entirely below zero, reported as continuous evidence "
                "only: this run is CLOSE-graded and the predeclaration reserves any "
                "terminal classification for the opener grade. "
                if primary["week_blocked_ci95"][1] < 0.0
                else ""
            )
            + "One of 38 uncorrected cells screened together; the run used 200 "
            "within-week permutations (each family's permutation block records the "
            "count; the artifact's top-level config field was written null by a bug "
            "fixed the same session). CLOSE-graded: may not settle a play/no-play "
            "decision."
        )

        signal = WeakSignal(
            name=f"{SIGNAL_PREFIX}{name}",
            recorded_at=recorded_at,
            description=(
                f"Graph ratings v2 team_stat arm: signed-Katz opponent-adjusted "
                f"{name} (frozen CFB structure, alpha 0.85, half-life 8 weeks) versus the "
                f"raw home-minus-away {name} differential as the paired control, "
                "single-feature market-residual ridge, weekly-refit walk-forward, "
                "close-graded."
            ),
            source=source,
            effect=float(primary["delta_accuracy"] * 100.0),
            effect_units="accuracy_points",
            classification=classification,
            league="nfl",
            seasons=(seasons[0], seasons[-1]),
            interval=(
                primary["week_blocked_ci95"][0] * 100.0,
                primary["week_blocked_ci95"][1] * 100.0,
            ),
            probability_positive=primary["week_blocked_probability_positive"],
            sample_games=primary["n_games"],
            sample_blocks=primary["n_weeks"],
            family=SIGNAL_FAMILY,
            classification_evidence=evidence,
            closing_ground=closing_ground,
            notes=notes,
            plain_summary=f"Tests whether schedule-adjusting {name} beats using it raw.",
            category=FAMILY_CATEGORY[name],
        )
        if args.dry_run:
            print(
                f"{signal.name:<52}{signal.effect:>9.3f} pts  P+ "
                f"{signal.probability_positive:.3f}  {signal.classification}"
            )
        else:
            registry = record_signal(registry, signal, replace=True)
        recorded += 1

    if args.dry_run:
        print(f"dry run: {recorded} signals would be recorded")
        return 0
    save_registry(registry, registry_path)
    print(f"recorded {recorded} signals to {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
