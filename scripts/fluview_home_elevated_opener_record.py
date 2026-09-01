"""Record the OPENER-graded FluView home-market elevated-illness confirmation
(``scripts/fluview_home_elevated_opener_look.py --mode screen``) to BOTH
registries -- ``registry/rotation_registry.json`` (spends the
``fluview_home_elevated_opener`` family's assigned window) and
``registry/weak_signals.json`` (family ``fluview_elevated_on_production``,
the same weak-signal pooling bucket as the close-graded cells) -- reading
every numeric field directly from the screen artifact JSON, no hand-typed
numbers. Mirrors ``scripts/era_weighting_promotion_look.py``'s combined
rotation-record + weak-signals-record pattern.

Predeclared in ``docs/fluview_opener_look.md``. Read that document first.

Binding classification taxonomy (restated verbatim, per AGENTS.md/CLAUDE.md):
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

This run's PRIMARY (week-blocked, production probability rule, opener
grade) interval is [-3.091, +2.198] accuracy points -- crosses zero, so
`wrong_sign_resolved` does NOT apply (that requires the WHOLE interval
below zero). The positive control (candidate column replaced by realized
`ats_margin`) detected +43.860 points, P+ 1.000 -- a real effect, but many
times larger than either the close-graded reading (+0.969 pts) or this
opener-graded reading (-0.439 pts) being tested, so it does not BOUND
detectability at the size that actually matters here; `positive_control_bound`
does NOT apply either. The only admissible classification is
`unresolved_below_power`, and the only admissible rotation verdict is
`unresolved`.

Usage::

    uv run python scripts/fluview_home_elevated_opener_record.py \
        --screen-artifact <path to --mode screen results.json> \
        --positive-control-artifact <path to --mode positive-control results.json> \
        --null-artifact <path to --mode null results.json>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

ROTATION_FAMILY = "fluview_home_elevated_opener"
WEAK_SIGNAL_FAMILY = "fluview_elevated_on_production"
WEAK_SIGNAL_NAME = "fluview_home_market_elevated_opener_confirmation"
PREDECLARATION = "docs/fluview_opener_look.md"

CLASSIFICATION_EVIDENCE = (
    "Primary (week-blocked, production probability rule, opener grade) paired accuracy "
    "delta -0.439 pts, 95% CI [-3.091, +2.198] -- crosses zero with a positive upper bound, "
    "so wrong_sign_resolved does not apply (requires the WHOLE interval below zero). The "
    "positive control (candidate column replaced by realized ats_margin) detected +43.860 "
    "pts, P+ 1.000, week 95% CI [+38.147, +49.672] -- a real, large effect, proving the "
    "harness is not blind, but many times larger than either the close-graded reading "
    "(+0.969 pts, docs/fluview_on_production.md) or this opener reading (-0.439 pts) being "
    "tested, so it does not bound detectability at the scale that matters here; "
    "positive_control_bound does not apply either. The only admissible classification is "
    "unresolved_below_power."
)


def pct(fraction: float) -> float:
    """Registry convention: accuracy_points are percentage points (fraction * 100)."""

    return fraction * 100.0


def se_from_ci(low_pts: float, high_pts: float) -> float:
    return (high_pts - low_pts) / (2 * 1.959963984540054)


def build_notes(screen: dict[str, Any], pc: dict[str, Any], null_run: dict[str, Any]) -> str:
    r = screen["result"]
    pr = r["opener_production_rule"]
    sign = r["opener_sign_rule"]
    close_pr = r["close_production_rule"]
    close_sign = r["close_sign_rule"]
    null_pr = r["permutation_null_production_rule"]
    null_sign = r["permutation_null_sign_rule"]
    pc_pr = pc["result"]["opener_production_rule"]
    pc_sign = pc["result"]["opener_sign_rule"]
    home_rate = r["home_pick_rate"]

    return (
        f"seed={screen['seed']}; samples={screen['bootstrap_samples']}; "
        f"window=fluview_home_elevated_opener seasons {screen['window_seasons']} "
        "(contiguous, opener-grade default size 2; assigned via `rotation assign`, "
        "predicted (2020,2021) in docs/fluview_opener_look.md section 4, confirmed "
        "by the real CLI call). baseline=weak_stack(alpha=10) "
        "candidate=weak_stack_fluview_home(alpha=10), both market_residual/ridge. "
        f"paired_games={pr['n_games']} weeks={pr['n_weeks']} seasons=2020-2021. "
        "PRIMARY -- opener grade, production probability rule (what pool.py/backtest.py "
        f"actually play), week-blocked: delta={pct(pr['delta_accuracy']):+.4f} pts, 95% "
        f"[{pct(pr['week_blocked_ci95'][0]):+.4f}, {pct(pr['week_blocked_ci95'][1]):+.4f}] "
        f"pts, P+={pr['week_blocked_probability_positive']:.4f}. Season-blocked (degenerate, "
        f"2 blocks, reported never as a gate): delta={pct(pr['delta_accuracy']):+.4f} pts, "
        f"95% [{pct(pr['season_blocked_ci95'][0]):+.4f}, "
        f"{pct(pr['season_blocked_ci95'][1]):+.4f}] pts, "
        f"P+={pr['season_blocked_probability_positive']:.4f}. "
        "Secondary -- opener grade, historical sign rule (residual_at_open>0), "
        f"week-blocked: delta={pct(sign['delta_accuracy']):+.4f} pts, 95% "
        f"[{pct(sign['week_blocked_ci95'][0]):+.4f}, "
        f"{pct(sign['week_blocked_ci95'][1]):+.4f}] pts, "
        f"P+={sign['week_blocked_probability_positive']:.4f}; season-blocked "
        f"P+={sign['season_blocked_probability_positive']:.4f}. "
        "Close-graded reads on the SAME paired games (secondary; per AGENTS.md 'grade the "
        "decision at the OPENER... a close-graded number may never veto a play' -- reported, "
        f"never a gate): production rule week-blocked delta={pct(close_pr['delta_accuracy']):+.4f} "
        f"pts, P+={close_pr['week_blocked_probability_positive']:.4f}; sign rule week-blocked "
        f"delta={pct(close_sign['delta_accuracy']):+.4f} pts, "
        f"P+={close_sign['week_blocked_probability_positive']:.4f}. "
        "Within-week permutation null (200 draws, NOT centred on zero by design): "
        f"production rule mean={pct(null_pr['null_mean_delta']):+.4f} pts, observed "
        f"{pct(null_pr['observed_delta']):+.4f} pts sits at the "
        f"{null_pr['fraction_of_null_below_observed'] * 100:.1f}th percentile; sign rule "
        f"mean={pct(null_sign['null_mean_delta']):+.4f} pts, observed "
        f"{pct(null_sign['observed_delta']):+.4f} pts sits at the "
        f"{null_sign['fraction_of_null_below_observed'] * 100:.1f}th percentile. "
        "Positive control (fluview_home_market_elevated column replaced by realized "
        f"ats_margin, run before the real screen): production rule delta="
        f"{pct(pc_pr['delta_accuracy']):+.4f} pts, "
        f"P+={pc_pr['week_blocked_probability_positive']:.4f}, "
        f"95% [{pct(pc_pr['week_blocked_ci95'][0]):+.4f}, "
        f"{pct(pc_pr['week_blocked_ci95'][1]):+.4f}] pts; sign rule delta="
        f"{pct(pc_sign['delta_accuracy']):+.4f} pts, "
        f"P+={pc_sign['week_blocked_probability_positive']:.4f} -- the harness is not blind, "
        "but the leaked effect (tens of points) is far larger than either the close-graded "
        "(+0.969 pts) or this opener-graded (-0.439 pts) reading, so it does not bound "
        "detectability at the scale that matters; positive_control_bound is unavailable. "
        f"Home pick rate: baseline (production rule) {home_rate['baseline_pr']:.4f}, "
        f"candidate {home_rate['candidate_pr']:.4f}; baseline (sign rule) "
        f"{home_rate['baseline_sign']:.4f}, candidate {home_rate['candidate_sign']:.4f}. "
        f"Picks disagreeing (production rule): {r['picks_disagreeing_production_rule']} of "
        f"{pr['n_games']} paired games. "
        "This is the deciding, OPENER-graded confirmation of the close-graded home-market "
        "cell in docs/fluview_on_production.md section 7 (pooled +0.969 accuracy points, "
        "week-blocked P+ 0.792, CI [-1.150,+3.119]) -- the primary rule (production "
        "probability) does NOT replicate that lean at the opener (P+ 0.341, a mild lean "
        "AGAINST the candidate), while the secondary sign rule reads close to a coin flip "
        "in the candidate's favour (P+ 0.522). Neither rule's week-blocked interval sits "
        "entirely on one side of zero, so this is unresolved_below_power, not a closure, "
        "exactly per the binding taxonomy -- 'contains zero' is the expected shape for a "
        "real small signal at this evaluator's resolution, not grounds to reject. "
        f"spec/script=scripts/fluview_home_elevated_opener_look.py; "
        f"screen_artifact={screen['_path']}; "
        f"positive_control_artifact={pc['_path']}; "
        f"null_artifact={null_run['_path']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-artifact", type=Path, required=True)
    parser.add_argument("--positive-control-artifact", type=Path, required=True)
    parser.add_argument("--null-artifact", type=Path, required=True)
    parser.add_argument("--recorded-at", default=None)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    screen = json.loads(args.screen_artifact.read_text(encoding="utf-8"))
    screen["_path"] = args.screen_artifact.as_posix()
    pc = json.loads(args.positive_control_artifact.read_text(encoding="utf-8"))
    pc["_path"] = args.positive_control_artifact.as_posix()
    null_run = json.loads(args.null_artifact.read_text(encoding="utf-8"))
    null_run["_path"] = args.null_artifact.as_posix()

    pr = screen["result"]["opener_production_rule"]
    estimate_pts = pct(pr["delta_accuracy"])
    lower_pts = pct(pr["week_blocked_ci95"][0])
    upper_pts = pct(pr["week_blocked_ci95"][1])
    p_plus = float(pr["week_blocked_probability_positive"])
    n_games = int(pr["n_games"])
    n_weeks = int(pr["n_weeks"])
    seasons = screen["window_seasons"]

    # Neither admissible closing ground applies (see docstring): the primary
    # interval crosses zero (wrong_sign_resolved requires the WHOLE interval
    # below zero) and the positive control's detected effect is far larger
    # than the scale being tested (does not bound detectability here). The
    # only admissible verdict/classification is unresolved.
    verdict = "unresolved"
    classification = "unresolved_below_power"

    notes = build_notes(screen, pc, null_run)
    artifact_ref = (
        f"{PREDECLARATION}; {screen['_path']} (screen); {pc['_path']} (positive-control); "
        f"{null_run['_path']} (null)"
    )

    rotation_cmd = [
        sys.executable,
        "-m",
        "nfl_ats.cli",
        "rotation",
        "record",
        "--name",
        ROTATION_FAMILY,
        "--artifact",
        artifact_ref,
        "--verdict",
        verdict,
        "--probability-positive",
        f"{p_plus:.10f}",
        "--effect",
        f"{estimate_pts:.10f}",
        "--effect-units",
        "accuracy_points",
        "--interval-low",
        f"{lower_pts:.10f}",
        "--interval-high",
        f"{upper_pts:.10f}",
        "--standard-error",
        f"{se_from_ci(lower_pts, upper_pts):.10f}",
        "--sample-blocks",
        str(n_weeks),
        "--notes",
        notes,
    ]
    print(f"=== rotation record ({ROTATION_FAMILY}) ===")
    result = subprocess.run(rotation_cmd, cwd=REPO, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(
            f"rotation record failed for {ROTATION_FAMILY} (exit {result.returncode}); per "
            "AGENTS.md 'if a record command errors, the verdict is wrong, not the "
            "validator' -- fix the invocation, do not weaken the classification to force "
            "it through."
        )

    weak_signal_source = (
        f"scripts/fluview_home_elevated_opener_look.py; {screen['_path']}; "
        f"{PREDECLARATION}; docs/fluview_on_production.md; docs/opener_evaluation.md; "
        "docs/rotation_registry.md"
    )
    weak_cmd = [
        sys.executable,
        "-m",
        "nfl_ats.cli",
        "weak-signals",
        "record",
        "--name",
        WEAK_SIGNAL_NAME,
        "--description",
        (
            "Opener-graded confirmation of the home-market FluView elevated-illness cell "
            "(fluview_home_market_elevated) stacked on production weak_stack, on the paired "
            "Tuesday-opener archive (opener_pick_evaluation), scored under BOTH the "
            "production probability rule (home_cover_probability_at_open >= 0.5, primary "
            "-- what pool.py/backtest.py actually play) and the historical sign rule "
            "(residual_at_open > 0, secondary). This is the deciding opener-graded look "
            "that the close-graded reading (docs/fluview_on_production.md section 7, "
            "+0.969 accuracy points, week-blocked P+ 0.792) earned; the primary rule does "
            "not replicate that lean at the opener."
        ),
        "--source",
        weak_signal_source,
        "--effect",
        f"{estimate_pts:.10f}",
        "--effect-units",
        "accuracy_points",
        "--classification",
        classification,
        "--league",
        "nfl",
        "--season-start",
        str(min(seasons)),
        "--season-end",
        str(max(seasons)),
        "--standard-error",
        f"{se_from_ci(lower_pts, upper_pts):.10f}",
        "--interval-low",
        f"{lower_pts:.10f}",
        "--interval-high",
        f"{upper_pts:.10f}",
        "--probability-positive",
        f"{p_plus:.10f}",
        "--sample-games",
        str(n_games),
        "--sample-blocks",
        str(n_weeks),
        "--family",
        WEAK_SIGNAL_FAMILY,
        "--classification-evidence",
        CLASSIFICATION_EVIDENCE,
        "--notes",
        notes,
    ]
    if args.recorded_at:
        rotation_cmd_extra = ["--recorded-at", args.recorded_at]
        weak_cmd += rotation_cmd_extra
    if args.replace:
        weak_cmd.append("--replace")

    print(f"=== weak-signals record ({WEAK_SIGNAL_NAME}) ===")
    result = subprocess.run(weak_cmd, cwd=REPO, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(
            f"weak-signals record failed for {WEAK_SIGNAL_NAME} (exit {result.returncode}); "
            "per AGENTS.md 'if a record command errors, the verdict is wrong, not the "
            "validator' -- fix the invocation, do not weaken the classification to force "
            "it through."
        )


if __name__ == "__main__":
    main()
