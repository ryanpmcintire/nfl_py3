"""Record the 5 ``dst_*`` cells from ``scripts/dst_transition_battery_screen.py``
into ``registry/weak_signals.json`` via ``nfl-ats weak-signals record``.

Every numeric field (effect, interval, probability_positive, sample_games,
sample_blocks) is read directly from the run's ``results.json`` and passed
through as a CLI argument -- no hand-typed numbers.

## Binding closing-grounds taxonomy (verbatim, per AGENTS.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that
size. Everything else is ``unresolved_below_power``: record it with
``nfl-ats weak-signals record``, report ``probability_positive``, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator.

Every cell in this 5-cell predeclared battery (``docs/dst_transition_battery.md``)
records ``unresolved_below_power``: none of the 5 week-blocked intervals sits
ENTIRELY outside zero (checked programmatically below, regardless of which
direction was predicted), so ``wrong_sign_resolved`` does not apply, and no
positive-control bound was run, so ``positive_control_bound`` does not apply
either.

Two of the five cells (``dst_arizona_home_shield``, ``dst_arizona_away_shield``)
have ``n_flag_blocks`` below ``estimation_variance.MIN_BLOCKS_FOR_INTERVAL``
(10) under BOTH blockings -- the recorded interval is still passed through
(the registry schema has nowhere else to put it and the number is not
wrong, just not a trustworthy 95% coverage claim), but the ``notes`` field
says so explicitly per AGENTS.md's below-power reporting rule, and the
recorded standard_error/interval must never be read as resolving those two
cells.

Usage::

    uv run python scripts/record_dst_transition_battery.py --results <path> [--replace]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

MIN_BLOCKS_FOR_INTERVAL = 10  # nfl_ats.estimation_variance.MIN_BLOCKS_FOR_INTERVAL

SOURCE = (
    "scripts/dst_transition_battery_screen.py; artifacts/dst_transition_battery/"
    "{timestamp}/results.json; docs/dst_transition_battery.md"
)

CLASSIFICATION_EVIDENCE = (
    "Predeclared 5-cell DST-transition-shock battery (docs/dst_transition_battery.md), "
    "frozen before scoring. Week-blocked 95% CI crosses zero -- per AGENTS.md that is "
    "the EXPECTED shape for a real small signal, not grounds to close. Neither "
    "admissible closing ground applies: no resolved wrong sign (interval does not sit "
    "entirely on the wrong side of the predeclared predicted direction) and no "
    "positive-control bound was run."
)


def _block_tier_label(n_flag_blocks: int) -> str:
    if n_flag_blocks >= 50:
        return "at or above the 50-block reliable tier"
    return "marginal tier, 10-49 blocks: usable but measurably narrower than nominal"


def notes_for(cell: dict[str, Any], payload: dict[str, Any]) -> str:
    wb = cell["week_blocked"]
    sb = cell["season_blocked_secondary"]
    below_floor = wb["n_flag_blocks"] < MIN_BLOCKS_FOR_INTERVAL
    floor_note = (
        f"n_flag_blocks={wb['n_flag_blocks']} (week-blocked) / {sb['n_flag_blocks']} "
        f"(season-blocked) is BELOW MIN_BLOCKS_FOR_INTERVAL=10 -- measured coverage at "
        "this block count is well under the nominal 95% (0.47 at 2 blocks, 0.76 at 4, "
        "0.90 at 10). The recorded interval below is NOT a trustworthy 95% interval; "
        "read only the point estimate (effect) and probability_positive, per AGENTS.md "
        "('report the point estimate and probability_positive... do NOT treat it as a "
        "failure')."
        if below_floor
        else (
            f"n_flag_blocks={wb['n_flag_blocks']} (week-blocked) / {sb['n_flag_blocks']} "
            "(season-blocked) meets the MIN_BLOCKS_FOR_INTERVAL=10 floor "
            f"({_block_tier_label(wb['n_flag_blocks'])})."
        )
    )
    return (
        "Battery multiplicity: one of 5 predeclared cells in the NFL DST-transition "
        "battery (scripts/dst_transition_battery_screen.py), scored on the same "
        f"{wb['n_total']:,}-row restricted population "
        f"({payload['season_start']}-{payload['season_end']}); mined/exploratory; do "
        "not sign-test-pool battery cells as independent. "
        f"predicted_sign={cell['predicted_sign']:+d}. Seed {payload['bootstrap_seed']}, "
        f"{payload['bootstrap_samples']:,} bootstrap draws. n_blocks (whole restricted "
        f"population, not the governing figure) = {wb['n_blocks']} week-blocks / "
        f"{sb['n_blocks']} season-blocks. {floor_note} Season-blocked secondary "
        f"bootstrap: 95% [{sb['ci95_scaled'][0]:+.4f}, {sb['ci95_scaled'][1]:+.4f}] "
        f"P+={sb['probability_positive']:.4f} effect={sb['full_slate_effect_pts']:+.4f}pts "
        "(robustness check, not the registry interval -- for this battery's once/"
        "twice-per-season cells, week- and season-blocking are structurally similar, "
        "not independent checks). Transition dates (docs/dst_transition_battery.md sec "
        "1) were measured via zoneinfo, not hardcoded; the spring-transition/postseason "
        f"overlap candidate cell was dropped ({payload['spring_postseason_overlap_seasons']} "
        "of 17 seasons measured to overlap)."
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
        wb = cell["week_blocked"]
        if wb.get("insufficient_data"):
            print(f"skip {cell['name']}: insufficient_data")
            continue

        lo, hi = wb["ci95_scaled"]
        if lo > 0 or hi < 0:
            raise SystemExit(
                f"{cell['name']}: week-blocked interval [{lo:+.4f}, {hi:+.4f}] does NOT "
                "cross zero -- CLASSIFICATION_EVIDENCE above asserts every cell crosses "
                "zero. Stop and re-check by hand whether this is actually a resolved "
                "wrong sign (predicted_sign vs. the interval's side) before recording -- "
                "would need --classification refuted_mechanism --closing-ground "
                "wrong_sign_resolved instead, and ONLY if the interval sits entirely on "
                "the WRONG side of predicted_sign."
            )

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
            "--family",
            "dst_transition_battery",
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
