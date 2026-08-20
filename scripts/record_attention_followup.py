"""Record the 4 ``attention_followup_*`` cells from
``scripts/attention_followup_screen.py`` into ``registry/weak_signals.json``
via ``nfl-ats weak-signals record``.

Every numeric field (effect, interval, probability_positive, sample_games,
sample_blocks) is read directly from the run's ``results.json`` and passed
through as a CLI argument -- no hand-typed numbers.

Closing-grounds taxonomy (restated verbatim, AGENTS.md): An interval or CI
that contains zero is NEVER grounds to reject, fail, or close an experiment.
At this evaluator's ~2-point resolution, "contains zero" is the EXPECTED
outcome for a real small signal. Only two grounds ever close a line of work:
(1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the wrong
side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
``unresolved_below_power``: record it with ``nfl-ats weak-signals record``,
report ``probability_positive``, never the binary "contains zero." The
registry code hard-rejects inadmissible closures; if a record command
errors, the verdict is wrong, not the validator.

None of the 4 cells here has a whole-interval-wrong-side-of-zero result (all
4 week-blocked CIs contain zero) and no positive-control bound was run, so
the only admissible classification for all 4 is ``unresolved_below_power``,
carrying no ``--closing-ground``.

Usage::

    uv run python scripts/record_attention_followup.py --results <path> [--replace]
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
    "scripts/attention_followup_screen.py; artifacts/attention_followup/"
    "{timestamp}/results.json; docs/attention_followup.md"
)

RELIABILITY = 0.13153959916293442  # read from attention_battery_both_cold registry entry;
# same underlying attention_z trait, not recomputed here.

CLASSIFICATION_EVIDENCE = (
    "Mined 4-cell follow-up battery (uncorrected multiplicity) deepening "
    "attention_battery_both_cold (+0.52pt, P+0.857), predeclared in "
    "docs/attention_followup.md before scoring. Week-blocked 95% CI contains "
    "zero for all 4 cells -- per AGENTS.md that is the EXPECTED shape for a "
    "real small signal, not grounds to close. Neither admissible closing "
    "ground applies: no interval sits entirely on the wrong side of zero "
    "(no resolved wrong sign) and no positive-control bound was run."
)


def notes_for(cell: dict[str, Any], payload: dict[str, Any]) -> str:
    wb = cell["week_blocked_primary"]
    sb = cell["season_blocked_secondary"]
    units_note = wb.get("units_note", "")
    extra = f" {units_note}" if units_note else ""
    return (
        "Battery multiplicity: one of 4 predeclared cells in the attention "
        "both_cold follow-up battery (scripts/attention_followup_screen.py), "
        f"deepening attention_battery_both_cold; scored on REG "
        f"{payload['season_start']}-{payload['season_end']}. "
        f"n_total={wb['n_total']}, n_blocks={wb.get('n_blocks')}. "
        "Same underlying Wikipedia-pageview attention_z construct as the "
        "parent battery (window ends Tuesday of game week, point-in-time "
        "safe); split-half reliability of that trait (read from the parent "
        f"registry entry, not recomputed): r={RELIABILITY:.4f}. "
        "Season-blocked secondary bootstrap "
        f"(block=season, n={sb.get('n_blocks')} seasons): 95% "
        f"[{sb['week_blocked_ci95_scaled'][0]:+.4f}, "
        f"{sb['week_blocked_ci95_scaled'][1]:+.4f}] "
        f"P+={sb['probability_positive']:.4f} (robustness check, not the "
        f"registry interval). mined/exploratory; do not sign-test-pool "
        "battery cells as independent." + extra
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--recorded-at", default=None)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    timestamp = args.results.parent.name
    source = SOURCE.format(timestamp=timestamp)

    for cell in payload["results"]:
        wb = cell["week_blocked_primary"]
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
            f"{wb['week_blocked_ci95_scaled'][0]:.10f}",
            "--interval-high",
            f"{wb['week_blocked_ci95_scaled'][1]:.10f}",
            "--probability-positive",
            f"{wb['probability_positive']:.10f}",
            "--sample-games",
            str(wb["n_total"]),
            "--sample-blocks",
            str(wb.get("n_blocks")),
            "--reliability",
            f"{RELIABILITY:.10f}",
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
