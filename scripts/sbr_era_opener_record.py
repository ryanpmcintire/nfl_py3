"""Record the four ``sbr_opener_*`` entries from
``scripts/sbr_era_opener_eval.py``'s run into ``registry/weak_signals.json``
via ``nfl-ats weak-signals record`` -- every numeric field read directly from
the run's artifact JSON, no hand-typed numbers.

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
era's own week-blocked interval from the artifact.

Usage::

    uv run --no-sync python scripts/sbr_era_opener_record.py \
        --summary artifacts/sbr_era_opener_eval/<run-id>/summary.json [--replace]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

DOC = "docs/sbr_opener_evaluation.md"

CLOSING_RULE_NOTE = (
    "Binding taxonomy: an interval containing zero is NEVER grounds to reject/close "
    "(expected shape for a real small signal at this evaluator's resolution). Only a "
    "RESOLVED wrong sign (whole interval below zero) or a positive-control bound may "
    "close a line of work; no positive control was run in this document, so only "
    "wrong_sign_resolved is ever available here, and only when the whole week-blocked "
    "interval sits below zero. Otherwise: unresolved_below_power, reported with "
    "probability_positive, never as a binary pass/fail."
)

ENTRY_DESCRIPTIONS = {
    "era_2011_2014": (
        "Era A of the SBR-derived-opener era-stratified evaluation: frozen production "
        "model (market_residual/weak_stack/ridge alpha=10.0), SBR Open substituted as the "
        "settlement line, REG games 2011-2014 (earliest era the 500-game warm-up floor "
        "allows). Re-slice of the same scored games underlying the already-recorded "
        "proxy_opener_production_rule_2009_2019 entry (2011-2019 pooled) -- see notes for "
        "the overlap disclosure."
    ),
    "era_2015_2019": (
        "Era B of the SBR-derived-opener era-stratified evaluation: frozen production "
        "model, SBR Open substituted as the settlement line, REG games 2015-2019 (identical "
        "boundary to docs/era_magnitude_profile.md's own 2015-2019 bucket). Re-slice of the "
        "same scored games underlying the already-recorded "
        "proxy_opener_production_rule_2009_2019 entry -- see notes for the overlap "
        "disclosure."
    ),
    "era_2020_2021": (
        "Era C of the SBR-derived-opener era-stratified evaluation: frozen production "
        "model, SBR Open substituted as the settlement line, REG games 2020-2021 -- the "
        "FULL SBR-matched population (not restricted to the tue_open-paired subset used by "
        "docs/proxy_opener_replication.md's calibration arm). Genuinely new SBR-graded "
        "population, never previously scored in full. SBR's archive ceiling (2021-22 is "
        "the last populated season) bounds this era on the late end."
    ),
    "pooled_2011_2021": (
        "Pooled 2011-2021 headline read of the SBR-derived-opener era-stratified "
        "evaluation: frozen production model, SBR Open substituted as the settlement line, "
        "all 11 scorable seasons (2011-2021) combined. Answers 'where does the model's "
        "SBR-graded opener edge live' at the pooled level; per-era magnitudes are the "
        "separate sbr_opener_era_* entries."
    ),
}


def pct(fraction: float) -> float:
    """Registry convention: accuracy_points are percentage points (fraction * 100),
    reported here as points ABOVE 50 (the metric this run's bootstrap already computes
    as accuracy - 0.5)."""

    return fraction * 100.0


def _fmt_rule(rule: dict[str, Any], label: str) -> str:
    wb = rule["week_blocked"]
    sb = rule["season_blocked"]
    parts = [
        f"{label}: absolute={pct(rule['absolute_accuracy'] - 0.5) + 50:.2f}%, "
        f"week-blocked estimate={pct(wb['estimate']):+.4f} pts, 95% "
        f"[{pct(wb['lower']):+.4f}, {pct(wb['upper']):+.4f}] pts, "
        f"P+={wb['probability_positive']:.4f}"
    ]
    if sb is not None:
        parts.append(
            f"season-blocked estimate={pct(sb['estimate']):+.4f} pts, 95% "
            f"[{pct(sb['lower']):+.4f}, {pct(sb['upper']):+.4f}] pts, "
            f"P+={sb['probability_positive']:.4f} (coverage caveat: <10 season-blocks, "
            "see docs/sbr_opener_evaluation.md section 5)"
        )
    else:
        parts.append("season-blocked: not computed (< 2 season-blocks)")
    return "; ".join(parts)


def classify(rule: dict[str, Any]) -> tuple[str, str | None, str]:
    """Mechanical classification from the week-blocked interval alone.

    Whole interval below zero -> wrong_sign_resolved (the only admissible
    closing ground available in this document; no positive control was run).
    Otherwise unresolved_below_power, regardless of how close to or far from
    zero the interval sits -- per the binding taxonomy, crossing zero (or
    even sitting entirely above it) is never itself a closing ground.
    """

    wb = rule["week_blocked"]
    upper = pct(wb["upper"])
    lower = pct(wb["lower"])
    if upper < 0.0:
        return (
            "refuted_mechanism",
            "wrong_sign_resolved",
            f"Whole week-blocked interval [{lower:+.4f}, {upper:+.4f}] sits below zero -- "
            "the only admissible closing ground this document can produce (no positive "
            "control was run). " + CLOSING_RULE_NOTE,
        )
    return (
        "unresolved_below_power",
        None,
        f"Week-blocked interval [{lower:+.4f}, {upper:+.4f}] does not sit entirely below "
        "zero (crosses zero or sits entirely above it) -- neither admissible closing "
        "ground applies (no positive control was run in this document). " + CLOSING_RULE_NOTE,
    )


def build_entry(name: str, slice_result: dict[str, Any], timestamp: str) -> dict[str, Any]:
    pr = slice_result["probability_rule"]
    sign = slice_result["sign_rule"]
    classification, closing_ground, evidence = classify(pr)

    wb = pr["week_blocked"]
    notes = (
        f"Seasons {slice_result['season_start']}-{slice_result['season_end']} "
        f"({slice_result['seasons_present']}), {slice_result['games']} games, "
        f"{slice_result['n_weeks']} weeks. PRODUCTION RULE (primary): "
        f"{_fmt_rule(pr, 'production')}. SIGN RULE (secondary): {_fmt_rule(sign, 'sign')}. "
        f"spec/script=scripts/sbr_era_opener_eval.py; predeclaration={DOC}; "
        f"artifact=artifacts/sbr_era_opener_eval/{timestamp}/summary.json. " + CLOSING_RULE_NOTE
    )

    record: dict[str, Any] = {
        "name": f"sbr_opener_{name}",
        "description": ENTRY_DESCRIPTIONS[name],
        "source": (
            f"scripts/sbr_era_opener_eval.py; "
            f"artifacts/sbr_era_opener_eval/{timestamp}/summary.json; {DOC}"
        ),
        "effect": f"{pct(wb['estimate']):.10f}",
        "effect_units": "accuracy_points",
        "classification": classification,
        "league": "nfl",
        "season_start": str(slice_result["season_start"]),
        "season_end": str(slice_result["season_end"]),
        "interval_low": f"{pct(wb['lower']):.10f}",
        "interval_high": f"{pct(wb['upper']):.10f}",
        "probability_positive": f"{wb['probability_positive']:.10f}",
        "sample_games": str(slice_result["games"]),
        "sample_blocks": str(slice_result["n_weeks"]),
        "classification_evidence": evidence,
        "notes": notes,
    }
    if closing_ground is not None:
        record["closing_ground"] = closing_ground
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--recorded-at", default=None)
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    timestamp = args.summary.parent.name

    entries = {
        "era_2011_2014": payload["eras"]["era_2011_2014"],
        "era_2015_2019": payload["eras"]["era_2015_2019"],
        "era_2020_2021": payload["eras"]["era_2020_2021"],
        "pooled_2011_2021": payload["pooled_2011_2021"],
    }

    for name, slice_result in entries.items():
        record = build_entry(name, slice_result, timestamp)
        cmd = [sys.executable, "-m", "nfl_ats.cli", "weak-signals", "record"]
        for key, value in record.items():
            flag = "--" + key.replace("_", "-")
            cmd += [flag, value]
        if args.recorded_at:
            cmd += ["--recorded-at", args.recorded_at]
        if args.replace:
            cmd.append("--replace")

        print(f"=== recording sbr_opener_{name} ===")
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(
                f"weak-signals record failed for sbr_opener_{name} (exit "
                f"{result.returncode}); per AGENTS.md 'if a record command errors, the "
                "verdict is wrong, not the validator' -- fix the invocation, do not "
                "weaken the classification to force it through."
            )


if __name__ == "__main__":
    main()
