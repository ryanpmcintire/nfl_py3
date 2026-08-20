"""Record ``surface_profile_opener_head_to_head`` into ``registry/weak_signals.json``
via ``nfl-ats weak-signals record``, reading every numeric field directly from
``scripts/surface_profile_opener_eval.py``'s artifact JSON -- no hand-typed
numbers.

Binding classification taxonomy (restated verbatim, per AGENTS.md /
CLAUDE.md -- this script is not a subagent, but the rule applies uniformly):
An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that
size. Everything else is `unresolved_below_power`: record it, report
`probability_positive`, never the binary "contains zero". The registry
code hard-rejects inadmissible closures; if a record command errors, the
verdict is wrong, not the validator.

This run's primary (week-blocked, production probability-rule, opener
grade) interval is [-2.108, +1.307] accuracy points -- crosses zero, upper
bound positive, so `wrong_sign_resolved` does NOT apply (that requires the
WHOLE interval below zero). No positive control was run. The only
admissible classification is `unresolved_below_power`.

Usage::

    uv run python scripts/surface_profile_opener_record.py --summary <path> [--replace]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

NAME = "surface_profile_opener_head_to_head"

DESCRIPTION = (
    "Opener-graded, forced-pick, paired head-to-head of MOD-08 candidate "
    "weak_stack_surface (weak_stack plus the single surface_switch_flag ridge "
    "feature) vs production weak_stack, both ridge_alpha=10.0, market_residual, "
    "on the paired Tuesday-opener archive (build_pairing_table / "
    "opener_pick_evaluation), scored under BOTH production's probability pick "
    "rule (home_cover_probability_at_open >= 0.5, headline -- what pool.py/"
    "backtest.py actually play) and the historical sign rule (residual_at_open "
    "> 0). This is the OPENER-graded confirmation of the same pairing "
    "docs/surface_switch_feature_arm.md already scored at the CLOSE grade "
    "(registry entry surface_switch_feature_arm, +0.2410 pts, P+ 0.6181)."
)

CLASSIFICATION_EVIDENCE = (
    "Primary (week-blocked, production probability-rule, opener grade) paired "
    "accuracy delta crosses zero with a positive upper bound -- per AGENTS.md "
    "this is the EXPECTED shape for a real small signal, not grounds to close. "
    "wrong_sign_resolved does not apply (requires the WHOLE interval below "
    "zero; this interval's upper bound is positive). No positive-control bound "
    "was run, so positive_control_bound does not apply either. The only "
    "admissible classification is unresolved_below_power."
)


def pct(fraction: float) -> float:
    """Registry convention (matches surface_switch_feature_arm's stored effect):
    accuracy_points are percentage points, i.e. fraction * 100."""

    return fraction * 100.0


def build_notes(d: dict[str, Any], summary_path: Path) -> str:
    pr = d["open_accuracy_delta_probability_rule"]
    sign = d["open_accuracy_delta_sign_rule"]
    close_pr = d["close_accuracy_delta_probability_rule"]
    close_sign = d["close_accuracy_delta_sign_rule"]
    bm = d["baseline_metrics"]
    cm = d["candidate_metrics"]
    brier_wb = d["brier_delta_open"]["week_blocked"]
    brier_by_metric = {row["metric"]: row for row in brier_wb}

    return (
        f"seed={d.get('bootstrap_seed', 'unset-in-artifact')}; samples=20000; "
        "baseline=weak_stack(alpha=10) candidate=weak_stack_surface(alpha=10); "
        f"paired_games={d['paired_games']} weeks={d['weeks']} seasons="
        f"{min(d['seasons'])}-{max(d['seasons'])}. "
        "Absolute opener accuracy -- baseline (production rule) "
        f"{pct(bm['opener_accuracy_probability_rule']):.4f}%, candidate "
        f"{pct(cm['opener_accuracy_probability_rule']):.4f}%; baseline (sign "
        f"rule) {pct(bm['opener_accuracy']):.4f}%, candidate "
        f"{pct(cm['opener_accuracy']):.4f}%. Sanity anchor: incumbent "
        f"production-rule opener accuracy measured here "
        f"({pct(bm['opener_accuracy_probability_rule']):.4f}%) matches the "
        "tracked 53.36% from docs/opener_evaluation.md's 2026-08-19 addendum "
        f"(sign-rule 52.83% match: {pct(bm['opener_accuracy']):.4f}%). "
        "Season-blocked secondary (registered effect/interval/P+ above is "
        f"week-blocked, primary): estimate={pct(pr['season_blocked']['estimate']):+.4f} pts, "
        f"95% [{pct(pr['season_blocked']['lower']):+.4f}, "
        f"{pct(pr['season_blocked']['upper']):+.4f}] pts, "
        f"P+={pr['season_blocked']['probability_positive']:.4f}. "
        "Sign-rule opener delta (secondary comparability read, historical "
        f"protocol): week-blocked estimate={pct(sign['week_blocked']['estimate']):+.4f} pts, "
        f"95% [{pct(sign['week_blocked']['lower']):+.4f}, "
        f"{pct(sign['week_blocked']['upper']):+.4f}] pts, "
        f"P+={sign['week_blocked']['probability_positive']:.4f}; season-blocked "
        f"estimate={pct(sign['season_blocked']['estimate']):+.4f} pts, "
        f"95% [{pct(sign['season_blocked']['lower']):+.4f}, "
        f"{pct(sign['season_blocked']['upper']):+.4f}] pts, "
        f"P+={sign['season_blocked']['probability_positive']:.4f}. "
        "Close-graded reads on the SAME paired games (secondary; per AGENTS.md "
        "'grade the decision at the OPENER... a close-graded number may never "
        "veto a play' -- reported, never a gate): production rule week-blocked "
        f"estimate={pct(close_pr['week_blocked']['estimate']):+.4f} pts, 95% "
        f"[{pct(close_pr['week_blocked']['lower']):+.4f}, "
        f"{pct(close_pr['week_blocked']['upper']):+.4f}] pts, "
        f"P+={close_pr['week_blocked']['probability_positive']:.4f}; sign rule "
        f"week-blocked estimate={pct(close_sign['week_blocked']['estimate']):+.4f} pts, "
        f"P+={close_sign['week_blocked']['probability_positive']:.4f}. "
        "Brier/log-loss (week-blocked, probabilities are rule-agnostic): "
        f"brier_improvement={brier_by_metric['brier_improvement']['estimate']:+.6f} "
        f"P+={brier_by_metric['brier_improvement']['probability_positive']:.4f}; "
        f"log_loss_improvement={brier_by_metric['log_loss_improvement']['estimate']:+.6f} "
        f"P+={brier_by_metric['log_loss_improvement']['probability_positive']:.4f} "
        "(both near a coin flip, not resolved either direction). "
        f"Pick flips vs baseline: {d['picks_disagreeing_probability_rule']} of "
        f"{d['paired_games']} games (production rule), "
        f"{d['picks_disagreeing_sign_rule']} (sign rule). "
        f"surface_switch_flag is SET on {d['surface_switch_flag_games_in_archive']} "
        f"of {d['paired_games']} paired opener-archive games "
        f"({100 * d['surface_switch_flag_rate_in_archive']:.2f}%, 0 missing) -- "
        "DOUBLE-COUNT FLAG for the orchestrator: the live "
        "surface_switch_tilt_overlay pick-level challenger already flips a "
        "subset of these same flagged games on the published card; if "
        "weak_stack_surface were ever promoted into production, the overlay "
        "would be re-applying the identical construct on top of a model that "
        "already trained on it -- the two must not both be live at once. "
        "REUSED-WINDOW ACKNOWLEDGMENT (rotation_registry.md rule 6, not a "
        "fresh independent look): this opener read is correlated with the "
        "surface family's prior looks this week -- the mined weather-battery "
        "cell (weather_battery_surface_switch_grass_to_turf), the "
        "venue-controlled follow-up (surface_familiarity_r1_turf_venue_visitor_split), "
        "the era splits (surface_familiarity_r3_era_2009_2017 / _2018_2025), the "
        "grass-venue mirror (surface_familiarity_r2_grass_venue_mirror), the CFB "
        "replications, and the CLOSE-graded feature_arm look "
        "(surface_switch_feature_arm, +0.2410 pts, P+ 0.6181) on this exact same "
        "weak_stack_surface vs weak_stack pairing. Per AGENTS.md, a reused "
        "window 'carries a stated discount, not a ban' -- this result should be "
        "read as one more correlated measurement of the same underlying "
        "construct, not independent confirmation. "
        f"spec/script=scripts/surface_profile_opener_eval.py; "
        f"artifact={summary_path.as_posix()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to surface_profile_opener_eval.py's opener_summary.json",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--recorded-at", default=None)
    args = parser.parse_args()

    d = json.loads(args.summary.read_text(encoding="utf-8"))

    pr = d["open_accuracy_delta_probability_rule"]["week_blocked"]

    source = (
        f"scripts/surface_profile_opener_eval.py; {args.summary.as_posix()}; "
        "docs/surface_switch_feature_arm.md; docs/opener_evaluation.md; "
        "docs/rotation_registry.md"
    )

    cmd = [
        sys.executable,
        "-m",
        "nfl_ats.cli",
        "weak-signals",
        "record",
        "--name",
        NAME,
        "--description",
        DESCRIPTION,
        "--source",
        source,
        "--effect",
        f"{pct(pr['estimate']):.10f}",
        "--effect-units",
        "accuracy_points",
        "--classification",
        "unresolved_below_power",
        "--league",
        "nfl",
        "--season-start",
        str(min(d["seasons"])),
        "--season-end",
        str(max(d["seasons"])),
        "--standard-error",
        f"{(pct(pr['upper']) - pct(pr['lower'])) / (2 * 1.959963984540054):.10f}",
        "--interval-low",
        f"{pct(pr['lower']):.10f}",
        "--interval-high",
        f"{pct(pr['upper']):.10f}",
        "--probability-positive",
        f"{pr['probability_positive']:.10f}",
        "--sample-games",
        str(d["paired_games"]),
        "--sample-blocks",
        str(d["weeks"]),
        "--classification-evidence",
        CLASSIFICATION_EVIDENCE,
        "--notes",
        build_notes(d, args.summary),
    ]
    if args.recorded_at:
        cmd += ["--recorded-at", args.recorded_at]
    if args.replace:
        cmd.append("--replace")

    print(f"=== recording {NAME} ===")
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(
            f"weak-signals record failed for {NAME} (exit {result.returncode}); "
            "per AGENTS.md 'if a record command errors, the verdict is wrong, "
            "not the validator' -- fix the invocation, do not weaken the "
            "classification to force it through."
        )


if __name__ == "__main__":
    main()
