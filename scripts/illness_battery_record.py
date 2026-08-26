"""Record every illness-designation battery cell to
``registry/weak_signals.json``, reading ONLY from
``scripts/illness_battery_screen.py``'s results JSON -- no hand-typed
numbers.

Classification is computed MECHANICALLY via
``nfl_ats.experiment_runner.classify_subset_bias_result`` (imported, not
reimplemented) applied to each cell's PRIMARY (week-blocked, 2010-2024
excluding the COVID-era 2020 stratum) interval:

**An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment.** At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two things
justify closing a line of work: (1) a refuted mechanism -- a RESOLVED wrong
sign (the whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an
effect that size. Everything else is ``unresolved_below_power``: record it,
report ``probability_positive``, never the binary "contains zero". If a
record command errors, the verdict is wrong, not the validator.

Shells out to ``nfl-ats weak-signals record`` per cell (never writes the
registry file directly), then reads the registry back to verify every name
landed with the expected classification/effect.

Only the 5 PRIMARY cells are recorded. The 2020 (COVID-era) stratum in the
results JSON (``stratum_2020_covid_era_supplementary_not_pooled``) is
deliberately NOT recorded as separate registry entries -- per
``docs/illness_battery.md`` section 4 it is a supplementary diagnostic
(regime-change check), not a pooled or gating measurement.
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
SEASON_START = 2010
SEASON_END = 2024


def _latest_results() -> Path:
    candidates = sorted((REPO / "artifacts" / "illness_battery").glob("*/results.json"))
    if not candidates:
        raise FileNotFoundError("no artifacts/illness_battery/*/results.json found")
    return candidates[-1]


def build_args(
    cell: dict[str, Any], *, source: str, reliability: float, replace: bool
) -> list[str]:
    wb = cell["week_blocked"]
    estimate = float(wb["full_slate_effect_pts"])
    lower, upper = (float(v) for v in wb["ci95_scaled"])
    verdict = classify_subset_bias_result(estimate=estimate, lower=lower, upper=upper)

    name = cell["name"]
    sb = cell["season_blocked_secondary"]
    sb_note = (
        f"Season-blocked secondary 95% CI [{sb['ci95_scaled'][0]:+.4f}, "
        f"{sb['ci95_scaled'][1]:+.4f}], P+={sb['probability_positive']:.4f}."
        if not sb.get("insufficient_data")
        else "Season-blocked secondary: insufficient data."
    )
    notes = (
        f"{verdict.note} {sb_note} n_population={cell['n_population']}, "
        f"n_excluded_missing={cell['n_excluded_missing']}, n_flag={cell['n_flag']}. "
        "Illness-designation battery (docs/illness_battery.md); predeclared before any "
        "cover-rate sign was inspected. Point-in-time-safe as-of construction (per-entity "
        "(season,week,team,gsis_id) checkpoint, date_modified <= "
        "nfl_ats.pick_refresh.pick_deadline(kickoff, sunday_lock), docs/illness_battery.md "
        "section 3). Population is REG 2010-2024 EXCLUDING the COVID-era 2020 stratum "
        "(scored separately, not pooled, not recorded as its own registry entries -- "
        "docs/illness_battery.md section 4). Reconciled against the existing NFL.com "
        "injury scrape on 2022-2024: 97.13% illness-designation agreement, 2.87% "
        "disagreement (scripts/nflverse_injuries_reconcile.py). Caveat: illness is a "
        "self-reported team designation with known between-club reporting variance, "
        "biasing toward attenuation, not a false positive (docs/illness_battery.md "
        "section 7)."
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
        str(reliability),
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
    source = f"docs/illness_battery.md, {results_path.relative_to(REPO)}"
    reliability = float(payload["reliability"]["spearman_brown_full_length_reliability"])
    print(f"=== reading {results_path} ===")
    print(f"=== source recorded as: {source} ===")
    print(f"=== shared reliability figure (section 6): {reliability:.4f} ===")

    recorded_names = []
    for cell in payload["results"]:
        if cell["week_blocked"].get("insufficient_data"):
            print(f"\n--- {cell['name']}: SKIPPED (insufficient data) ---")
            continue
        cli_args = build_args(cell, source=source, reliability=reliability, replace=args.replace)
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
