"""Record MOD-14 era-weighting screen arms into ``registry/weak_signals.json``
via ``nfl-ats weak-signals record`` -- every numeric field read directly from
a run's ``paired_comparisons.csv``/``diagnostics.json``, no hand-typed
numbers. Works for both the CFB screen
(``scripts/era_weighting_cfb_screen.py``) and the NFL close-grade analog
(``scripts/era_weighting_nfl_screen.py``); pass ``--league`` to select.

Binding classification taxonomy (restated verbatim, per AGENTS.md/CLAUDE.md
-- this script is not a subagent, but the rule applies uniformly): An
interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is
the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is `unresolved_below_power`: record it, report `probability_positive`, never
the binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator.

Classification is decided MECHANICALLY per entry (whole week-blocked
interval below zero -> wrong_sign_resolved; otherwise
unresolved_below_power), never asserted in prose first, by reading each
arm's own week-blocked accuracy interval from paired_comparisons.csv.

Usage::

    uv run --no-sync python scripts/era_weighting_record.py \\
        --league cfb \\
        --paired artifacts/era_weighting_cfb_screen/<run-id>/paired_comparisons.csv \\
        --diagnostics artifacts/era_weighting_cfb_screen/<run-id>/diagnostics.json \\
        --season-start 2012 --season-end 2025 [--replace]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DOC = "docs/era_weighting_screen.md"

CLOSING_RULE_NOTE = (
    "Binding taxonomy: an interval containing zero is NEVER grounds to reject/close "
    "(expected shape for a real small signal at this evaluator's resolution). Only a "
    "RESOLVED wrong sign (whole interval below zero) or a positive-control bound may "
    "close a line of work; no positive control was run in this document, so only "
    "wrong_sign_resolved is ever available here, and only when the whole week-blocked "
    "interval sits below zero. Otherwise: unresolved_below_power, reported with "
    "probability_positive, never as a binary pass/fail."
)

ARM_LABELS = {
    "half_life_2": "exponential season-decay sample weight, half-life 2 seasons",
    "half_life_4": "exponential season-decay sample weight, half-life 4 seasons",
    "half_life_8": "exponential season-decay sample weight, half-life 8 seasons",
    "half_life_16": "exponential season-decay sample weight, half-life 16 seasons",
    "rolling_6": "training truncated to a rolling 6-season window (unweighted within window)",
    "rolling_10": "training truncated to a rolling 10-season window (unweighted within window)",
}


def pct(fraction: float) -> float:
    """Registry convention: accuracy_points are percentage points (fraction * 100)."""

    return fraction * 100.0


def classify(lower_pts: float, upper_pts: float) -> tuple[str, str | None, str]:
    """Mechanical classification from the week-blocked interval alone."""

    if upper_pts < 0.0:
        return (
            "refuted_mechanism",
            "wrong_sign_resolved",
            f"Whole week-blocked interval [{lower_pts:+.4f}, {upper_pts:+.4f}] sits below "
            "zero -- the only admissible closing ground this screen can produce (no "
            "positive control was run). " + CLOSING_RULE_NOTE,
        )
    return (
        "unresolved_below_power",
        None,
        f"Week-blocked interval [{lower_pts:+.4f}, {upper_pts:+.4f}] does not sit entirely "
        "below zero (crosses zero or sits entirely above it) -- neither admissible closing "
        "ground applies (no positive control was run in this screen). " + CLOSING_RULE_NOTE,
    )


def build_entry(
    arm: str,
    row: pd.Series,
    *,
    league: str,
    name_prefix: str,
    season_start: int,
    season_end: int,
    paired_path: Path,
    diagnostics_path: Path,
    window_note: str,
) -> dict[str, Any]:
    lower_pts = pct(float(row["lower"]))
    upper_pts = pct(float(row["upper"]))
    estimate_pts = pct(float(row["estimate"]))
    classification, closing_ground, evidence = classify(lower_pts, upper_pts)
    grade_label = (
        "CFB XLG-03 free screen" if league == "cfb" else "NFL close-grade below-power screen"
    )

    description = (
        f"MOD-14 era-weighting screen ({grade_label}): {ARM_LABELS[arm]}, vs. a uniform-weight "
        f"all-history baseline arm, on the frozen market-residual Ridge recipe "
        f"(ridge_alpha=10.0 unchanged, no bundled tuning). {window_note}"
    )
    notes = (
        f"Week-blocked paired accuracy improvement vs baseline: {estimate_pts:+.4f} pts, 95% "
        f"[{lower_pts:+.4f}, {upper_pts:+.4f}] pts, P+={float(row['probability_positive']):.4f}, "
        f"paired games={int(row.get('paired_games', row.get('games', 0)))}, "
        f"blocks={int(row['blocks'])}. spec/script=scripts/era_weighting_{league}_screen.py; "
        f"predeclaration={DOC}; source={paired_path}; diagnostics={diagnostics_path}. "
        + CLOSING_RULE_NOTE
    )

    record: dict[str, Any] = {
        "name": f"{name_prefix}_{arm}",
        "description": description,
        "source": f"scripts/era_weighting_{league}_screen.py; {paired_path}; {DOC}",
        "effect": f"{estimate_pts:.10f}",
        "effect_units": "accuracy_points",
        "classification": classification,
        "league": league,
        "season_start": str(season_start),
        "season_end": str(season_end),
        "interval_low": f"{lower_pts:.10f}",
        "interval_high": f"{upper_pts:.10f}",
        "probability_positive": f"{float(row['probability_positive']):.10f}",
        "sample_games": str(int(row.get("paired_games", row.get("games", 0)))),
        "sample_blocks": str(int(row["blocks"])),
        "classification_evidence": evidence,
        "notes": notes,
    }
    if closing_ground is not None:
        record["closing_ground"] = closing_ground
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=["cfb", "nfl"], required=True)
    parser.add_argument("--paired", type=Path, required=True, help="paired_comparisons.csv")
    parser.add_argument("--diagnostics", type=Path, required=True, help="diagnostics.json")
    parser.add_argument("--season-start", type=int, required=True)
    parser.add_argument("--season-end", type=int, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--recorded-at", default=None)
    args = parser.parse_args()

    paired = pd.read_csv(args.paired)
    diagnostics = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    arms = [name for name in diagnostics["arms"] if name != "baseline"]
    name_prefix = f"era_weighting_{args.league}"

    subset = paired.loc[paired["metric"].eq("accuracy_improvement") & paired["block"].eq("week")]
    if "family" in subset.columns:
        subset = subset.loc[subset["family"].eq("probability")]
    if args.league == "cfb":
        subset = subset.loc[subset["evaluation_window"].eq("clean_core")]
        window_note = (
            "Effect is the clean-core (2012-2019, 2021-2025; 2020's sparse-provider regime and "
            "2006-2011's thin-line regime excluded, matching nfl_ats.cfb_benchmark's own "
            "convention) week-blocked paired accuracy improvement."
        )
    else:
        window_note = (
            "Effect is the close-grade, week-blocked paired accuracy improvement over seasons "
            "no rotation-registry family has reserved (rule 8); this run is gated on a positive "
            "CFB lean, so it is a second, disclosed look at the same direction, not an "
            "independent blind draw."
        )

    for arm in arms:
        rows = subset.loc[subset["candidate_feature_set"].eq(arm)]
        if rows.empty:
            raise SystemExit(f"No week-blocked accuracy row found for arm {arm!r} in {args.paired}")
        row = rows.iloc[0]
        record = build_entry(
            arm,
            row,
            league=args.league,
            name_prefix=name_prefix,
            season_start=args.season_start,
            season_end=args.season_end,
            paired_path=args.paired,
            diagnostics_path=args.diagnostics,
            window_note=window_note,
        )
        cmd = [sys.executable, "-m", "nfl_ats.cli", "weak-signals", "record"]
        for key, value in record.items():
            flag = "--" + key.replace("_", "-")
            cmd += [flag, value]
        if args.recorded_at:
            cmd += ["--recorded-at", args.recorded_at]
        if args.replace:
            cmd.append("--replace")

        print(f"=== recording {record['name']} ===")
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(
                f"weak-signals record failed for {record['name']} (exit "
                f"{result.returncode}); per AGENTS.md 'if a record command errors, the "
                "verdict is wrong, not the validator' -- fix the invocation, do not "
                "weaken the classification to force it through."
            )


if __name__ == "__main__":
    main()
