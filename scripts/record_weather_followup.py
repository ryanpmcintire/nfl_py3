"""Record the 5 ``weather_followup_*`` cells from
``scripts/nfl_weather_followup_screen.py`` into ``registry/weak_signals.json``
via ``nfl-ats weak-signals record``.

Every numeric field (effect, interval, probability_positive, sample_games,
sample_blocks) is read directly from the run's ``results.json`` and passed
through as a CLI argument -- no hand-typed numbers. Per AGENTS.md ("An
interval crossing zero is NOT grounds for rejection"), every cell in this
mined/predeclared battery records ``unresolved_below_power`` regardless of
interval shape; none of these cells' intervals sit ENTIRELY on the wrong
side of the predicted (positive) direction, so ``wrong_sign_resolved`` does
not apply, and no positive-control bound was run, so
``positive_control_bound`` does not apply either -- the only admissible
classification is ``unresolved_below_power``, carrying no ``--closing-ground``.

Usage::

    uv run python scripts/record_weather_followup.py --results <path> [--replace]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

SOURCE = (
    "scripts/nfl_weather_followup_screen.py; artifacts/nfl_weather_followup/"
    "{timestamp}/results.json; docs/weather_followup.md; "
    "scratchpad/agent_weather_followup/predeclaration.json"
)

CLASSIFICATION_EVIDENCE = (
    "Mined 5-cell second-generation weather/environment follow-up battery "
    "(uncorrected multiplicity), predeclared in docs/weather_followup.md before "
    "scoring. Week-blocked 95% CI crosses zero or is only narrowly resolved-shaped "
    "-- per AGENTS.md that is the EXPECTED shape for a real small signal, not "
    "grounds to close. Neither admissible closing ground applies: no resolved "
    "wrong sign (no interval sits entirely on the wrong side of the predicted "
    "positive direction) and no positive-control bound was run."
)


def notes_for(cell: dict[str, Any], payload: dict[str, Any]) -> str:
    wb = cell["week_blocked"]
    sb = cell["season_blocked_secondary"]
    return (
        "Battery multiplicity: one of 5 predeclared cells in the NFL weather "
        "follow-up battery (scripts/nfl_weather_followup_screen.py), scored on "
        f"the same {wb['n_total']:,}-row REG {payload['season_start']}-"
        f"{payload['season_end']} population "
        f"({payload['n_pushes_or_missing_dropped']} pushes/rows with missing "
        f"result or spread_line dropped from {payload['n_reg_games_before_push_drop']:,} "
        "REG games); mined/exploratory; do not sign-test-pool battery cells as "
        f"independent. Seed {payload['bootstrap_seed']}, "
        f"{payload['bootstrap_samples']:,} bootstrap draws. "
        f"n_missing_required_data={cell['n_missing_required_data']} rows had a "
        "required input missing for this cell -- flag forced False, included in "
        "complement. Season-blocked secondary bootstrap "
        f"(block=season, n={sb['n_blocks']} seasons): 95% "
        f"[{sb['ci95_scaled'][0]:+.4f}, {sb['ci95_scaled'][1]:+.4f}] "
        f"P+={sb['probability_positive']:.4f} (robustness check, not the registry "
        "interval)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--recorded-at", default=None)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    # Timestamp directory name is the results.json's parent directory name.
    timestamp = args.results.parent.name
    source = SOURCE.format(timestamp=timestamp)

    for cell in payload["results"]:
        wb = cell["week_blocked"]
        if wb.get("insufficient_data"):
            print(f"skip {cell['name']}: insufficient_data")
            continue

        cmd = [
            sys.executable,
            "-m",
            "nfl_ats.cli",
            "weak-signals",
            "record",
            "--name",
            cell["name"],
            "--description",
            cell["description"],
            "--source",
            source,
            "--effect",
            f"{wb['full_slate_effect_pts']:.10f}",
            "--effect-units",
            "accuracy_points",
            "--classification",
            "unresolved_below_power",
            "--league",
            "nfl",
            "--season-start",
            str(payload["season_start"]),
            "--season-end",
            str(payload["season_end"]),
            "--interval-low",
            f"{wb['ci95_scaled'][0]:.10f}",
            "--interval-high",
            f"{wb['ci95_scaled'][1]:.10f}",
            "--probability-positive",
            f"{wb['probability_positive']:.10f}",
            "--sample-games",
            str(wb["n_total"]),
            "--sample-blocks",
            str(wb["n_blocks"]),
            "--classification-evidence",
            CLASSIFICATION_EVIDENCE,
            "--notes",
            notes_for(cell, payload),
        ]
        if args.recorded_at:
            cmd += ["--recorded-at", args.recorded_at]
        if args.replace:
            cmd.append("--replace")

        print(f"=== recording {cell['name']} ===")
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(
                f"weak-signals record failed for {cell['name']} (exit "
                f"{result.returncode}); per AGENTS.md 'if a record command errors, "
                "the verdict is wrong, not the validator' -- fix the invocation, "
                "do not weaken the classification to force it through."
            )


if __name__ == "__main__":
    main()
