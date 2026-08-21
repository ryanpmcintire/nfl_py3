"""Record the 7 ``sagarin_battery_*`` cells from
``scripts/sagarin_divergence_battery.py`` into ``registry/weak_signals.json``
via ``nfl-ats weak-signals record``.

Every numeric field (effect, interval, probability_positive, sample_games,
sample_blocks) is read directly from the run's ``results.json`` and passed
through as a CLI argument -- no hand-typed numbers.

## Binding closing-grounds taxonomy (verbatim, per AGENTS.md / this task's brief)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is ``unresolved_below_power``: record it with
``nfl-ats weak-signals record``, report ``probability_positive``, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Every cell is recorded regardless of sign.

None of the 7 predeclared cells carries a predicted direction ("does the
Sagarin side cover" / "does agreement predict anything" are two-sided
questions a priori, per
``<scratchpad>/sagarin_divergence/predeclaration.json``), so
``wrong_sign_resolved`` can never apply to any of them; no positive control
was run, so ``positive_control_bound`` can never apply either. Every cell
therefore records ``unresolved_below_power`` -- decided before scoring, not
after seeing the numbers.

Usage::

    uv run python scripts/record_sagarin_divergence_battery.py --results <path> [--replace]
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
    "scripts/sagarin_divergence_battery.py; artifacts/sagarin_divergence_battery/"
    "{timestamp}/results.json; docs/sagarin_backfill.md section 8/9 (measured)"
)

CLASSIFICATION_EVIDENCE = (
    "Predeclared Sagarin-divergence screen (docs/sagarin_backfill.md section 6, frozen in "
    "<scratchpad>/sagarin_divergence/predeclaration.json before scoring). "
    "team-quality-is-already-priced ceiling applies (project MEMORY.md): Sagarin divergence "
    "from the market line is a power-rating-vs-market-consensus disagreement, exactly the "
    "family already measured bounded near zero. None of the 7 cells carries a predeclared "
    "sign, so wrong_sign_resolved cannot apply; no positive control was run, so "
    "positive_control_bound cannot apply. The complete-source replication preserves that "
    "frozen family and classification; unresolved_below_power per AGENTS.md's binding "
    "taxonomy."
)


def notes_for(cell: dict[str, Any], payload: dict[str, Any]) -> str:
    kind = cell["kind"]
    if kind == "single":
        rate = cell.get("sagarin_side_cover_rate")
        rate_str = f"{rate:.4f}" if rate is not None else "n/a"
        extra = f"sagarin_side_cover_rate={rate_str}. "
    else:
        extra = (
            f"agree_accuracy={cell.get('agree_accuracy'):.4f} "
            f"disagree_accuracy={cell.get('disagree_accuracy'):.4f} "
            f"n_flag(agree)={cell.get('n_flag')} "
            f"n_complement(disagree)={cell.get('n_complement')}. "
        )
    source = payload.get("sagarin_source", {})
    source_note = (
        f"Sagarin snapshot={source.get('snapshot_id', 'unrecorded')} "
        f"attempted={source.get('captures_attempted', 'unrecorded')} "
        f"fetch_ok={source.get('captures_fetch_ok', 'unrecorded')} "
        f"fetch_failed={source.get('captures_fetch_failed', 'unrecorded')} "
        f"parse_ok={source.get('captures_parse_ok', 'unrecorded')} "
        f"index_rows={source.get('index_rows', 'unrecorded')}. "
    )
    return (
        f"{cell['description']}. {extra}"
        f"n={cell['n_total']}, n_week_blocks={cell['n_blocks']}, "
        f"bootstrap_samples={cell['bootstrap_samples']}, dropped_draws={cell['dropped_draws']}, "
        f"seed={payload['bootstrap_seed']}. "
        f"{source_note}Close population: {payload['close_population_n']} REG games "
        "2010-2025 with a "
        "Tuesday-asof Sagarin RATING+home_edge_rating snapshot (2012 entirely unusable -- "
        "every 2012 capture's home-edge line failed to parse, measured this session -- and "
        "2013 usable only from week 13 onward, same reason; both are real archive-format "
        "gaps, not a join bug, see docs/sagarin_backfill.md's coverage table). "
        f"Opener population: {payload['open_population_n']} games -- "
        f"{payload['open_population_note']}. "
        f"Model-agreement population: {payload['model_agreement_population_n']} games -- "
        f"{payload['model_agreement_note']}. "
        f"{payload['ceiling_caveat']}."
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
        if cell.get("insufficient_data"):
            print(f"skip {cell['name']}: insufficient_data")
            continue

        classification = "unresolved_below_power"
        closing_ground = None

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
            f"{cell['effect_pts']:.10f}",
            "--effect-units",
            "accuracy_points",
            "--classification",
            classification,
            "--league",
            "nfl",
            "--season-start",
            str(cell["season_start"]),
            "--season-end",
            str(cell["season_end"]),
            "--interval-low",
            f"{cell['ci95'][0]:.10f}",
            "--interval-high",
            f"{cell['ci95'][1]:.10f}",
            "--probability-positive",
            f"{cell['probability_positive']:.10f}",
            "--sample-games",
            str(cell["n_total"]),
            "--sample-blocks",
            str(cell["n_blocks"]),
            "--classification-evidence",
            CLASSIFICATION_EVIDENCE,
            "--notes",
            notes_for(cell, payload),
        ]
        if closing_ground:
            cmd += ["--closing-ground", closing_ground]
        if args.recorded_at:
            cmd += ["--recorded-at", args.recorded_at]
        if args.replace:
            cmd.append("--replace")

        print(f"=== recording {cell['name']} (classification={classification}) ===")
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
