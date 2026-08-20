"""Record ``proxy_opener_production_rule_2009_2019`` into
``registry/weak_signals.json`` via ``nfl-ats weak-signals record``, reading
every numeric field directly from ``scripts/proxy_opener_replication.py``'s
artifact JSON -- no hand-typed numbers.

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

This run's primary (week-blocked, production probability-rule, pooled
2011-2019 proxy-opener) interval is [-1.600, +2.370] accuracy points above
50 -- crosses zero, upper bound positive, so `wrong_sign_resolved` does NOT
apply (that requires the WHOLE interval below zero). No positive control was
run. The only admissible classification is `unresolved_below_power`.

Usage::

    uv run python scripts/proxy_opener_replication_record.py --summary <path> [--replace]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

NAME = "proxy_opener_production_rule_2009_2019"

DESCRIPTION = (
    "Production model (market_residual/weak_stack/ridge alpha=10.0) walked forward "
    "week-by-week exactly as nfl_ats.clv.opener_pick_evaluation does, scored with "
    "SBR sportsbookreviewsonline.com 'Open' substituted as the settlement/prediction "
    "line -- a PROXY for the true Tuesday opener (r=0.949, mean |diff| 1.36-1.37 pts "
    "vs the true opener on the only overlap seasons, docs/sbr_odds_archive.md) -- on "
    "11 seasons (2009-2019) the purchased point-in-time market archive cannot reach. "
    "500-game warm-up floor (same constant/mechanism as the rotation registry's rule "
    "9) consumes 2009-2010 entirely; scored population is 2011-2019, 2,304 games, 153 "
    "weeks. Reported alongside a same-instrument 2020-2021 dual-grade calibration arm "
    "(SBR-Open vs true tue_open, same weekly-refit models, 466 paired games) that "
    "measures the proxy discount directly rather than assuming it."
)

CLASSIFICATION_EVIDENCE = (
    "Primary (week-blocked, production probability-rule, pooled 2011-2019 proxy-opener "
    "vs 50%) interval [-1.600, +2.370] accuracy points crosses zero with a positive "
    "upper bound -- per AGENTS.md this is the EXPECTED shape for a real small signal, "
    "not grounds to close. wrong_sign_resolved does not apply (requires the WHOLE "
    "interval below zero; this interval's upper bound is positive). No positive-control "
    "bound was run, so positive_control_bound does not apply either. The calibration "
    "arm's own paired delta (SBR-Open vs true-opener, production rule) is itself "
    "unresolved (P+ 0.167, interval crosses zero) so it cannot be used to reclassify "
    "this result either. The only admissible classification is unresolved_below_power."
)


def pct(fraction: float) -> float:
    """Registry convention: accuracy_points are percentage points (fraction * 100),
    reported here as points ABOVE 50 (the metric this run's bootstrap already computes
    as accuracy - 0.5), matching the task's declared effect definition."""

    return fraction * 100.0


def build_notes(d: dict[str, Any], summary_path: Path) -> str:
    main = d["main_arm"]
    calib = d["calibration_arm"]
    pr_wb = main["pooled_accuracy_vs_coin_flip"]["probability_rule"]["week_blocked"]
    pr_sb = main["pooled_accuracy_vs_coin_flip"]["probability_rule"]["season_blocked"]
    sign_wb = main["pooled_accuracy_vs_coin_flip"]["sign_rule"]["week_blocked"]
    sign_sb = main["pooled_accuracy_vs_coin_flip"]["sign_rule"]["season_blocked"]
    aa = main["pooled_accuracy_absolute"]

    per_season = "; ".join(
        f"{row['season']}: "
        + (
            "warm-up, not scorable"
            if not row["scorable"]
            else f"sign {pct(row['accuracy_sign_rule'] - 0.5) + 50:.2f}% / "
            f"prod {pct(row['accuracy_probability_rule'] - 0.5) + 50:.2f}%"
        )
        for row in main["per_season"]
    )

    calib_pr = calib["paired_delta_sbr_minus_true"]["probability_rule"]["week_blocked"]
    calib_sign = calib["paired_delta_sbr_minus_true"]["sign_rule"]["week_blocked"]
    calib_aa = calib["absolute_accuracy"]

    return (
        f"seed={d.get('bootstrap_seed')}; samples={d.get('bootstrap_samples')}; "
        f"config=weak_stack/ridge/alpha={d.get('ridge_alpha')} "
        f"(matches artifacts/active_ats_model.json); min_train_games={d.get('min_train_games')}. "
        f"MAIN ARM: population 2,816/2,816 REG games 2009-2019 have an SBR Open (100% "
        f"match, 2 flagged open_ambiguous, kept). Warm-up (500-game floor, same "
        "mechanism as rotation.py rule 9) consumes 2009-2010 entirely (0 of 34 weeks "
        f"scorable); 2011-2019 fully scorable. Scored: {main['scored_games']} games, "
        f"{d.get('_weeks', 153)} weeks, seasons {min(main['scored_seasons'])}-"
        f"{max(main['scored_seasons'])}. Absolute pooled accuracy: production rule "
        f"{pct(aa['probability_rule']):.4f}%, sign rule {pct(aa['sign_rule']):.4f}%. "
        f"Per-season (sign/production): {per_season}. "
        "Pooled vs 50% coin flip -- PRODUCTION RULE (primary, registered effect is "
        f"week-blocked): week-blocked estimate={pct(pr_wb['estimate']):+.4f} pts, 95% "
        f"[{pct(pr_wb['lower']):+.4f}, {pct(pr_wb['upper']):+.4f}] pts, "
        f"P+={pr_wb['probability_positive']:.4f}; season-blocked "
        f"estimate={pct(pr_sb['estimate']):+.4f} pts, 95% "
        f"[{pct(pr_sb['lower']):+.4f}, {pct(pr_sb['upper']):+.4f}] pts, "
        f"P+={pr_sb['probability_positive']:.4f}. SIGN RULE (secondary, historical "
        f"protocol comparability): week-blocked estimate={pct(sign_wb['estimate']):+.4f} "
        f"pts, 95% [{pct(sign_wb['lower']):+.4f}, {pct(sign_wb['upper']):+.4f}] pts, "
        f"P+={sign_wb['probability_positive']:.4f}; season-blocked "
        f"estimate={pct(sign_sb['estimate']):+.4f} pts, 95% "
        f"[{pct(sign_sb['lower']):+.4f}, {pct(sign_sb['upper']):+.4f}] pts, "
        f"P+={sign_sb['probability_positive']:.4f}. "
        "COMPARISON: the 2020-2025 true-opener production-rule figure is 53.36%, "
        "season-blocked 95% [51.97%, 54.56%] (docs/opener_evaluation.md, "
        "artifacts/opener_evaluation/20260819T174244Z/) -- this run's 2011-2019 "
        f"proxy-opener pooled production-rule accuracy ({pct(aa['probability_rule']):.2f}%) "
        "sits roughly 3 points closer to a coin flip than either the 6-season "
        "true-opener figure or the true-opener reading on the identical 2020-2021 "
        "seasons measured in the calibration arm "
        f"({pct(calib_aa['true_open_probability_rule']):.2f}%). "
        "CALIBRATION ARM (2020-2021, same weekly-refit models, both grades on the same "
        f"{calib['paired_games']} paired games, mean |line diff|="
        f"{calib['mean_abs_line_diff']:.3f} pts, reproduces docs/sbr_odds_archive.md's "
        "independently-measured 1.36pt figure): absolute accuracy -- SBR-Open sign "
        f"{pct(calib_aa['sbr_open_sign_rule']):.2f}%, SBR-Open production "
        f"{pct(calib_aa['sbr_open_probability_rule']):.2f}%, true-opener sign "
        f"{pct(calib_aa['true_open_sign_rule']):.2f}%, true-opener production "
        f"{pct(calib_aa['true_open_probability_rule']):.2f}%. Paired delta (SBR minus "
        f"true), production rule (primary): week-blocked estimate="
        f"{pct(calib_pr['estimate']):+.4f} pts, 95% [{pct(calib_pr['lower']):+.4f}, "
        f"{pct(calib_pr['upper']):+.4f}] pts, P+={calib_pr['probability_positive']:.4f} "
        "(proxy reads LOWER than true opener under the rule that matters, unresolved). "
        f"Sign rule: week-blocked estimate={pct(calib_sign['estimate']):+.4f} pts, "
        f"P+={calib_sign['probability_positive']:.4f} (proxy reads HIGHER under this "
        "rule -- the two rules disagree on the discount's direction, both unresolved). "
        "READING: neither admissible AGENTS.md closing ground applies (whole interval "
        "not below zero; no positive control run) -- unresolved_below_power is the only "
        "admissible classification. The calibration arm's measured discount "
        "(-0.865 pts, itself unresolved) is far too small to explain the ~3-point gap "
        "between the proxy-opener 2011-2019 pooled read and the true-opener figures by "
        "itself; whether the remaining gap is proxy-instrument noise, era-dependent "
        "model edge, or both is NOT resolved by this measurement (inferred discussion "
        "only, not claimed as fact). "
        f"spec/script=scripts/proxy_opener_replication.py; artifact={summary_path.as_posix()}; "
        "predeclaration=docs/proxy_opener_replication.md."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to proxy_opener_replication.py's summary.json",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--recorded-at", default=None)
    args = parser.parse_args()

    d = json.loads(args.summary.read_text(encoding="utf-8"))
    main_arm = d["main_arm"]
    pr_wb = main_arm["pooled_accuracy_vs_coin_flip"]["probability_rule"]["week_blocked"]

    source = (
        f"scripts/proxy_opener_replication.py; {args.summary.as_posix()}; "
        "docs/proxy_opener_replication.md; docs/sbr_odds_archive.md; "
        "docs/opener_evaluation.md"
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
        f"{pct(pr_wb['estimate']):.10f}",
        "--effect-units",
        "accuracy_points",
        "--classification",
        "unresolved_below_power",
        "--league",
        "nfl",
        "--season-start",
        str(min(main_arm["scored_seasons"])),
        "--season-end",
        str(max(main_arm["scored_seasons"])),
        "--standard-error",
        f"{(pct(pr_wb['upper']) - pct(pr_wb['lower'])) / (2 * 1.959963984540054):.10f}",
        "--interval-low",
        f"{pct(pr_wb['lower']):.10f}",
        "--interval-high",
        f"{pct(pr_wb['upper']):.10f}",
        "--probability-positive",
        f"{pr_wb['probability_positive']:.10f}",
        "--sample-games",
        str(main_arm["scored_games"]),
        "--sample-blocks",
        "153",
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
