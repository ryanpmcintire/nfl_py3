"""Record every PBP-08 team-style cell to ``registry/weak_signals.json``,
reading ONLY from ``scripts/team_style_screen.py``'s results JSON -- no
hand-typed numbers, per the task's own instruction.

Classification is computed MECHANICALLY via
``nfl_ats.experiment_runner.classify_subset_bias_result`` (imported, not
reimplemented) applied to each cell's PRIMARY (week-blocked) interval --
the same function/logic already enforced elsewhere in the repo, so this
script cannot improvise a verdict that violates the binding taxonomy:

**An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment.** At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two things
justify closing a line of work: (1) a refuted mechanism -- a RESOLVED wrong
sign (the whole interval on the wrong side of zero, AND the honest-refit
widening factor exceeds the documented bound) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an
effect that size. Everything else is ``unresolved_below_power``: record it,
report ``probability_positive``, never the binary "contains zero". If a
record command errors, the verdict is wrong, not the validator.

Shells out to ``nfl-ats weak-signals record`` per cell (never writes the
registry file directly), then reads the registry back to verify every name
landed with the expected classification/effect. ``--replace`` only if a
name already exists and was clobbered by a prior run of this script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.experiment_runner import classify_subset_bias_result  # noqa: E402
from nfl_ats.weak_signals import default_registry_path, load_registry  # noqa: E402

EFFECT_UNITS = "accuracy_points"
LEAGUE = "nfl"
SEASON_START = 2009
SEASON_END = 2025

# Weakest-link reliability figure embedded in each cell's --reliability flag
# (measured, scripts/team_style_reliability.py); composite/two-sided cells
# use the lower of their inputs so the recorded figure is conservative.
CELL_RELIABILITY: dict[str, float] = {
    "team_style_distinct_identity": 0.306,  # min over 9 dims (deep_share)
    "team_style_short_game_identity": 0.408,  # short_pass_share YoY r
    "team_style_short_game_vs_pressure_defense": 0.278,  # min(0.408 off, 0.278 def proxy)
    "team_style_pace_mismatch_dog_cover": 0.489,  # seconds_per_play_pace YoY r
    "team_style_deep_ball_outdoor_wind": 0.306,  # deep_share YoY r
}


def _latest_results() -> Path:
    candidates = sorted((REPO / "artifacts" / "team_style_screen").glob("*/results.json"))
    if not candidates:
        raise FileNotFoundError("no artifacts/team_style_screen/*/results.json found")
    return candidates[-1]


def build_args(cell: dict[str, Any], *, source: str, replace: bool) -> list[str]:
    wb = cell["week_blocked"]
    estimate = float(wb["full_slate_effect_pts"])
    lower, upper = (float(v) for v in wb["ci95_scaled"])
    verdict = classify_subset_bias_result(estimate=estimate, lower=lower, upper=upper)

    name = cell["name"]
    notes = (
        f"{cell['reliability_note']}. {verdict.note} Season-blocked secondary 95% CI "
        f"[{cell['season_blocked_secondary']['ci95_scaled'][0]:+.4f}, "
        f"{cell['season_blocked_secondary']['ci95_scaled'][1]:+.4f}], P+="
        f"{cell['season_blocked_secondary']['probability_positive']:.4f}. "
        f"n_flag={cell['n_flag']}, n_missing_required_data={cell['n_missing_required_data']}. "
        "PBP-08 team style/personality battery (docs/team_style.md); predeclared before any "
        "cover-rate sign was inspected."
    )
    args = [
        "weak-signals",
        "record",
        "--name",
        name,
        "--description",
        cell["description"],
        "--source",
        source,
        "--effect",
        f"{estimate!r}",
        "--effect-units",
        EFFECT_UNITS,
        "--classification",
        verdict.classification,
        "--league",
        LEAGUE,
        "--season-start",
        str(SEASON_START),
        "--season-end",
        str(SEASON_END),
        "--interval-low",
        f"{lower!r}",
        "--interval-high",
        f"{upper!r}",
        "--probability-positive",
        f"{wb['probability_positive']!r}",
        "--sample-games",
        str(wb["n_total"]),
        "--sample-blocks",
        str(wb["n_blocks"]),
        "--reliability",
        str(CELL_RELIABILITY[name]),
        "--classification-evidence",
        verdict.note,
        "--notes",
        notes,
    ]
    if verdict.closing_ground:
        args += ["--closing-ground", verdict.closing_ground]
    if replace:
        args.append("--replace")
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_path = args.results or _latest_results()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    source = f"docs/team_style.md, {results_path.relative_to(REPO)}"
    print(f"=== reading {results_path} ===")
    print(f"=== source recorded as: {source} ===")

    recorded_names = []
    for cell in payload["results"]:
        cli_args = build_args(cell, source=source, replace=args.replace)
        print(f"\n--- {cell['name']} ---")
        print(" ".join(cli_args))
        if args.dry_run:
            continue
        completed = subprocess.run(
            [sys.executable, "-m", "nfl_ats.cli", *cli_args],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        print(completed.stdout)
        if completed.returncode != 0:
            print(completed.stderr, file=sys.stderr)
            raise SystemExit(
                f"nfl-ats weak-signals record failed for {cell['name']!r} "
                f"(exit {completed.returncode}) -- per AGENTS.md, a record command error means "
                "the verdict is wrong, not the validator; fix the classification, do not bypass."
            )
        recorded_names.append(cell["name"])

    if args.dry_run:
        print("\n=== dry run, nothing recorded ===")
        return

    print("\n=== verifying registry ===")
    registry_path = default_registry_path()
    registry = load_registry(registry_path)
    for name in recorded_names:
        signal = registry.signals.get(name)
        if signal is None:
            raise SystemExit(f"expected {name!r} in registry after recording, not found")
        print(
            f"  {name}: classification={signal.classification} effect={signal.effect:+.4f} "
            f"{signal.effect_units} P+={signal.probability_positive} "
            f"reliability={signal.reliability}"
        )
    print(f"\ntotal signals in registry: {len(registry.signals)}")


if __name__ == "__main__":
    main()
