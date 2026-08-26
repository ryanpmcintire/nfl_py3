"""Record every transaction-wire battery cell to ``registry/weak_signals.json``,
reading ONLY from ``scripts/transaction_wire_battery_screen.py``'s results
JSON -- no hand-typed numbers.

Classification is computed MECHANICALLY via
``nfl_ats.experiment_runner.classify_subset_bias_result`` (imported, not
reimplemented) applied to each cell's PRIMARY (week-blocked) interval:

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


def _latest_results() -> Path:
    candidates = sorted((REPO / "artifacts" / "transaction_wire_battery").glob("*/results.json"))
    if not candidates:
        raise FileNotFoundError("no artifacts/transaction_wire_battery/*/results.json found")
    return candidates[-1]


def build_args(
    cell: dict[str, Any],
    *,
    source: str,
    reliability: float,
    season_start: int,
    season_end: int,
    replace: bool,
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
        f"n_excluded={cell['n_excluded']}, n_flag={cell['n_flag']}. "
        "Transaction-wire battery (docs/transaction_wire_battery.md); predeclared before "
        "any cover-rate sign was inspected. Point-in-time-safe team-week window counts "
        "(nfl_ats.transaction_wire_features.attach_transaction_counts, leakage-tested); "
        "scored population restricted to seasons meeting the predeclared per-article "
        "date-fetch completeness rule (section 1) -- see the results artifact's "
        "included_seasons/excluded_seasons for exactly which."
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
        str(season_start),
        "--season-end",
        str(season_end),
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
    source = f"docs/transaction_wire_battery.md, {results_path.relative_to(REPO)}"
    reliability = float(payload["reliability"]["spearman_brown_full_length_reliability"])
    season_start = int(payload["season_start"])
    season_end = int(payload["season_end"])
    print(f"=== reading {results_path} ===")
    print(f"=== source recorded as: {source} ===")
    print(f"=== included seasons: {payload['included_seasons']} ===")
    print(f"=== excluded seasons: {payload['excluded_seasons']} ===")
    print(f"=== shared reliability figure (section 5): {reliability:.4f} ===")

    recorded_names = []
    for cell in payload["results"]:
        if cell["week_blocked"].get("insufficient_data"):
            print(f"\n--- {cell['name']}: SKIPPED (insufficient data) ---")
            continue
        cli_args = build_args(
            cell,
            source=source,
            reliability=reliability,
            season_start=season_start,
            season_end=season_end,
            replace=args.replace,
        )
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
