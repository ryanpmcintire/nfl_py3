"""Rotation and weak-signal registry commands."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Any

from nfl_ats import registry_explorer
from nfl_ats.cli_common import _print_json
from nfl_ats.rotation import (
    GRADE_POOLS,
    MAX_WINDOW_SIZE,
    MIN_WINDOW_SIZE,
    MINED_SEASONS,
    STRATIFIED_GRADE,
    VERDICTS,
    Registry,
    assign_stratified_window,
    assign_window,
    declare_coverage_stub,
    declare_family,
    default_registry_path,
    load_registry,
    record_look,
    record_no_rotation_needed,
    registry_status,
    save_registry,
    set_plain_summary,
    validate_registry,
)
from nfl_ats.weak_signals import CATEGORIES as WEAK_SIGNAL_CATEGORIES
from nfl_ats.weak_signals import (
    CLASSIFICATIONS,
    EFFECT_UNITS,
    LEAGUES,
    WeakSignal,
    combination_report,
    family_overlap_warnings,
    invalidate_signal,
    record_signal,
    retag_effect_units,
    set_reliability,
)
from nfl_ats.weak_signals import CLOSING_GROUNDS as WEAK_SIGNAL_CLOSING_GROUNDS
from nfl_ats.weak_signals import coherence_problems as weak_signal_coherence_problems
from nfl_ats.weak_signals import default_registry_path as weak_signal_registry_path
from nfl_ats.weak_signals import load_registry as load_weak_signals
from nfl_ats.weak_signals import save_registry as save_weak_signals


def _rotation_family_payload(registry: Registry, name: str) -> dict[str, Any]:
    status = registry_status(registry)
    families = [family for family in status["families"] if family["name"] == name]
    return {
        "registry": str(default_registry_path()),
        "family": families[0],
        "grade_pools": status["grade_pools"],
        "season_usage": status["season_usage"],
    }


def _cmd_rotation_status(_: argparse.Namespace) -> None:
    registry = load_registry()
    _print_json({"registry": str(default_registry_path()), **registry_status(registry)})


def _cmd_weak_signals_status(args: argparse.Namespace) -> None:
    path = weak_signal_registry_path()
    registry = load_weak_signals(path)
    signals = sorted(registry.signals.values(), key=lambda signal: signal.name)
    filtered = [
        signal
        for signal in signals
        if args.classification in (None, signal.classification)
        and (args.include_invalidated or signal.status != "invalidated")
    ]
    families = family_overlap_warnings(filtered)
    _print_json(
        {
            "registry": str(path),
            "recorded": len(signals),
            "excluded_invalidated": sum(
                s.status == "invalidated" and args.classification in (None, s.classification)
                for s in signals
            )
            if not args.include_invalidated
            else 0,
            "families": families["families"],
            "overlap_warnings": families,
            "measurement_coherence_problems": weak_signal_coherence_problems(filtered),
            "signals": [
                {
                    "name": signal.name,
                    "status": signal.status,
                    "invalidated_reason": signal.invalidated_reason,
                    "superseded_by": signal.superseded_by,
                    "classification": signal.classification,
                    "league": signal.league,
                    "seasons": list(signal.seasons),
                    "effect": signal.effect,
                    "effect_units": signal.effect_units,
                    "standard_error": signal.resolved_standard_error(),
                    "favours_candidate": signal.favours_candidate,
                    "family": signal.family,
                    "source": signal.source,
                }
                for signal in filtered
            ],
        }
    )


def _cmd_weak_signals_invalidate(args: argparse.Namespace) -> None:
    path = weak_signal_registry_path()
    registry = invalidate_signal(
        load_weak_signals(path),
        name=args.name,
        reason=args.reason,
        superseded_by=args.superseded_by,
    )
    save_weak_signals(registry, path)
    signal = registry.signals[args.name]
    _print_json(
        {
            "registry": str(path),
            "name": signal.name,
            "status": signal.status,
            "invalidated_reason": signal.invalidated_reason,
            "superseded_by": signal.superseded_by,
        }
    )


def _cmd_weak_signals_record(args: argparse.Namespace) -> None:
    """Record one below-power result so it stops being re-litigated in prose.

    This command exists because its absence was the actual defect. The registry
    had ``status`` and ``pool`` but no way in, so recording a signal meant
    hand-writing Python against the internal API -- and every session took the
    cheaper path of writing a prose verdict instead. A standing rule with no
    ergonomic path is a rule that silently stops being followed: the ledger sat
    at three entries while a documented 13 of 27 discarded families belonged in
    it.
    """

    path = weak_signal_registry_path()
    registry = load_weak_signals(path)
    interval = None
    if args.interval_low is not None and args.interval_high is not None:
        interval = (float(args.interval_low), float(args.interval_high))
    signal = WeakSignal(
        name=args.name,
        recorded_at=args.recorded_at or datetime.now(UTC).date().isoformat(),
        description=args.description,
        source=args.source,
        effect=float(args.effect),
        effect_units=args.effect_units,
        classification=args.classification,
        league=args.league,
        seasons=(int(args.season_start), int(args.season_end)),
        standard_error=args.standard_error,
        interval=interval,
        probability_positive=args.probability_positive,
        sample_games=args.sample_games,
        sample_blocks=args.sample_blocks,
        reliability=args.reliability,
        family=args.family,
        classification_evidence=args.classification_evidence,
        closing_ground=args.closing_ground,
        notes=args.notes,
        plain_summary=args.plain_summary,
        category=args.category,
    )
    registry = record_signal(registry, signal, replace=args.replace)
    save_weak_signals(registry, path)
    # Both fields are optional (475 pre-existing rows carry neither), but a
    # NEW record that skips them is the ledger's raw-description/Uncategorised
    # fallback silently choosing itself -- warn out loud on stderr so this
    # never gets buried in the JSON stdout a caller might be parsing.
    if not args.plain_summary:
        print(
            f"warning: {signal.name!r} recorded with no --plain-summary; the public "
            "Signal Ledger page will show its raw description instead of plain "
            "English for this row",
            file=sys.stderr,
        )
    if not args.category:
        print(
            f"warning: {signal.name!r} recorded with no --category; it will render "
            "under 'Uncategorised' on the public Signal Ledger page",
            file=sys.stderr,
        )
    _print_json(
        {
            "registry": str(path),
            "recorded": signal.name,
            "classification": signal.classification,
            "effect": signal.effect,
            "effect_units": signal.effect_units,
            "favours_candidate": signal.favours_candidate,
            "total_signals": len(registry.signals),
        }
    )


def _cmd_weak_signals_pool(args: argparse.Namespace) -> None:
    """Ask whether the accumulated below-power pile is worth one combined look."""

    path = weak_signal_registry_path()
    registry = load_weak_signals(path)
    report = combination_report(
        registry,
        league=args.league,
        effect_units=args.effect_units,
        method=args.method,
    )
    _print_json({"registry": str(path), **report})


def _cmd_weak_signals_retag_units(args: argparse.Namespace) -> None:
    """Correct a mis-tagged ``effect_units`` on one entry without touching anything else.

    Exists because some entries were forced into a unit that did not match
    what was measured (a correlation coefficient, an MAE/Brier/log-loss
    *improvement*), with the true sign convention explained only in prose
    inside ``notes`` -- exactly the note a pooler will not read. This changes
    only the unit and appends one audit line; effect, interval,
    classification, and closing_ground are untouched (AGENTS.md forbids
    silently rewriting a recorded measurement, and a unit correction is not a
    new one).
    """

    path = weak_signal_registry_path()
    registry = load_weak_signals(path)
    previous_units = (
        registry.signals[args.name].effect_units if args.name in registry.signals else None
    )
    registry = retag_effect_units(
        registry,
        args.name,
        effect_units=args.effect_units,
        reason=args.reason,
    )
    save_weak_signals(registry, path)
    signal = registry.signals[args.name]
    _print_json(
        {
            "registry": str(path),
            "retagged": signal.name,
            "previous_effect_units": previous_units,
            "effect_units": signal.effect_units,
            "notes": signal.notes,
        }
    )


def _cmd_weak_signals_set_reliability(args: argparse.Namespace) -> None:
    """Attach a measured split-half reliability to one entry, touching nothing else.

    Most entries carry ``reliability: null``, which leaves one of only two
    admissible closing grounds neither usable nor rulable-out. This writes the
    measured number (plus its interval, method and artifact path, as one audit
    line in ``notes`` -- the schema has no interval field) and leaves effect,
    interval, classification, closing_ground and source byte-identical. It
    does NOT reclassify: a low reliability is a candidate for the
    ``no_split_half_reliability`` ground, and acting on it stays a separate,
    explicit decision.
    """

    path = weak_signal_registry_path()
    registry = load_weak_signals(path)
    previous = registry.signals[args.name].reliability if args.name in registry.signals else None
    registry = set_reliability(
        registry,
        args.name,
        reliability=args.reliability,
        reliability_low=args.reliability_low,
        reliability_high=args.reliability_high,
        method=args.method,
        source=args.source,
        reason=args.reason,
    )
    save_weak_signals(registry, path)
    signal = registry.signals[args.name]
    _print_json(
        {
            "registry": str(path),
            "name": signal.name,
            "previous_reliability": previous,
            "reliability": signal.reliability,
            "reliability_interval": [args.reliability_low, args.reliability_high],
            "method": args.method,
            "measured_from": args.source,
            "classification": signal.classification,
            "closing_ground": signal.closing_ground,
            "notes": signal.notes,
        }
    )


def _cmd_rotation_declare(args: argparse.Namespace) -> None:
    path = default_registry_path()
    inherits = tuple(part.strip() for part in str(args.inherits or "").split(",") if part.strip())
    registry = declare_family(
        load_registry(path),
        args.name,
        description=args.description,
        grade=args.grade,
        inherits=inherits,
        acknowledges_mined_2018_2025=args.acknowledge_mined,
        plain_summary=args.plain_summary,
    )
    save_registry(registry, path)
    _print_json({"declared": args.name, **_rotation_family_payload(registry, args.name)})


def _cmd_rotation_set_plain_summary(args: argparse.Namespace) -> None:
    """Attach (or correct) one family's reader-facing plain-English summary.

    Additive to the CLI surface, not a new registry concept: it changes
    ONLY ``plain_summary`` on one already-declared family, leaving grade,
    status, windows and every recorded verdict byte-identical -- see
    ``nfl_ats.rotation.set_plain_summary``.
    """

    path = default_registry_path()
    registry = set_plain_summary(load_registry(path), args.name, plain_summary=args.plain_summary)
    save_registry(registry, path)
    _print_json({"updated": args.name, **_rotation_family_payload(registry, args.name)})


def _cmd_rotation_assign(args: argparse.Namespace) -> None:
    path = default_registry_path()
    if args.stratified:
        if args.size is not None:
            raise ValueError(
                "--size does not apply to --stratified windows; a stratified "
                "window is always a two-leg pair (docs/era_stratified_windows_proposal.md)"
            )
        registry = assign_stratified_window(load_registry(path), args.name)
    else:
        registry = assign_window(load_registry(path), args.name, size=args.size)
    save_registry(registry, path)
    _print_json({"assigned": args.name, **_rotation_family_payload(registry, args.name)})


def _cmd_rotation_record(args: argparse.Namespace) -> None:
    path = default_registry_path()
    interval = None
    if args.interval_low is not None and args.interval_high is not None:
        interval = (float(args.interval_low), float(args.interval_high))
    leg_effects = None if args.leg_effects is None else json.loads(args.leg_effects)
    registry = record_look(
        load_registry(path),
        args.name,
        artifact=args.artifact,
        verdict=args.verdict,
        probability_positive=args.probability_positive,
        closing_ground=args.closing_ground,
        effect=args.effect,
        effect_units=args.effect_units,
        interval=interval,
        standard_error=args.standard_error,
        sample_blocks=args.sample_blocks,
        leg_effects=leg_effects,
        notes=args.notes,
        replace_existing=args.replace,
    )
    save_registry(registry, path)
    _print_json({"recorded": args.name, **_rotation_family_payload(registry, args.name)})


def _render_rotation_validate_text(payload: dict[str, Any]) -> str:
    lines = [
        f"{payload['families']} families; {payload['error_count']} error(s), "
        f"{payload['warning_count']} warning(s)",
        "",
    ]
    if not payload["issues"]:
        lines.append("no issues found")
        return "\n".join(lines)
    header = f"{'severity':<8} {'code':<32} {'family':<40} message"
    lines.append(header)
    lines.append("-" * len(header))
    for issue in payload["issues"]:
        lines.append(
            f"{issue['severity']:<8} {issue['code']:<32} {(issue['family'] or ''):<40.40} "
            f"{issue['message']}"
        )
    return "\n".join(lines)


def _cmd_rotation_validate(args: argparse.Namespace) -> None:
    """ENG-27: full-audit pass; never modifies the ledger.

    Exits non-zero on any error-severity issue -- see
    ``nfl_ats.rotation.validate_registry`` for what each check means and why
    it never blocks a ``save_registry`` write (existing tracked data, e.g.
    ``fluview_elevated_on_production``'s ``[2011, 2025]`` window, predates
    several of these checks and must keep loading).
    """

    path = default_registry_path()
    registry = load_registry(path)
    issues = validate_registry(registry)
    errors = [issue for issue in issues if issue.severity == "error"]
    payload = {
        "registry": str(path),
        "families": len(registry.families),
        "issues": [
            {
                "severity": issue.severity,
                "code": issue.code,
                "family": issue.family,
                "message": issue.message,
            }
            for issue in issues
        ],
        "error_count": len(errors),
        "warning_count": len(issues) - len(errors),
    }
    if args.json:
        _print_json(payload)
    else:
        print(_render_rotation_validate_text(payload))
    if errors:
        raise SystemExit(1)


def _cmd_rotation_declare_coverage(args: argparse.Namespace) -> None:
    """ENG-27: cover every weak-signal family lacking a rotation family.

    Read-only unless ``--apply``: computes the plan
    (``registry_explorer.coverage_plan``), and only writes when told to.
    Additive only -- every action either reserves a brand-new rotation
    -family name (a ``declared_for_coverage`` stub with no window) or
    records a brand-new ``no_rotation_needed`` entry; no existing family or
    look is ever touched.
    """

    weak_path = weak_signal_registry_path()
    rotation_path = default_registry_path()
    weak_registry = load_weak_signals(weak_path)
    registry = load_registry(rotation_path)

    plan = registry_explorer.coverage_plan(weak_registry, registry)
    counts = {
        registry_explorer.COVERAGE_ACTION_STUB: 0,
        registry_explorer.COVERAGE_ACTION_NO_ROTATION_NEEDED: 0,
    }
    for row in plan:
        counts[row["action"]] += 1

    if not args.apply:
        _print_json(
            {
                "mode": "dry_run",
                "weak_signals_registry": str(weak_path),
                "rotation_registry": str(rotation_path),
                "plan_rows": len(plan),
                "counts": counts,
                "plan": plan,
            }
        )
        return

    for row in plan:
        if row["action"] == registry_explorer.COVERAGE_ACTION_STUB:
            registry = declare_coverage_stub(
                registry,
                row["stub_name"],
                weak_signal_family=row["weak_signal_family"],
                league=row["league"],
                effect_units=tuple(row["effect_units"]),
            )
        else:
            registry = record_no_rotation_needed(
                registry,
                row["weak_signal_family"],
                league=row["league"],
                reason=row["reason"],
                effect_units=tuple(row["effect_units"]),
                notes=f"nfl-ats rotation declare-coverage: category={row['category']!r}",
            )
    save_registry(registry, rotation_path)
    _print_json(
        {
            "mode": "apply",
            "weak_signals_registry": str(weak_path),
            "rotation_registry": str(rotation_path),
            "applied_rows": len(plan),
            "counts": counts,
            "families_total": len(registry.families),
            "no_rotation_needed_total": len(registry.no_rotation_needed),
        }
    )


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the ``rotation`` and ``weak-signals`` command groups."""

    rotation = subparsers.add_parser(
        "rotation",
        help="manage the per-family confirmation-window registry "
        "(docs/rotation_registry.md); a look is one look and is always recorded",
    )
    rotation_commands = rotation.add_subparsers(dest="rotation_command", required=True)

    weak_signals = subparsers.add_parser(
        "weak-signals",
        help="track effects too small for their own test to resolve, and pool them "
        "(docs/pool_edge_plan.md); a below-power result is kept, never deleted",
    )
    weak_signal_commands = weak_signals.add_subparsers(dest="weak_signal_command", required=True)

    weak_signals_status = weak_signal_commands.add_parser(
        "status",
        aliases=["list"],
        help="list active recorded signals, their effects, directions and classifications",
    )
    weak_signals_status.add_argument(
        "--classification",
        choices=tuple(CLASSIFICATIONS),
        default=None,
        help="show only signals of one kind (default: all)",
    )
    weak_signals_status.set_defaults(handler=_cmd_weak_signals_status)
    weak_signals_status.add_argument(
        "--include-invalidated",
        action="store_true",
        help="include invalidated measurements retained for audit",
    )
    weak_signals_invalidate = weak_signal_commands.add_parser(
        "invalidate",
        help="exclude an invalid measurement while retaining its history; not a closure",
    )
    weak_signals_invalidate.add_argument("--name", required=True)
    weak_signals_invalidate.add_argument("--reason", required=True)
    weak_signals_invalidate.add_argument("--superseded-by", default=None)
    weak_signals_invalidate.set_defaults(handler=_cmd_weak_signals_invalidate)

    weak_signals_record = weak_signal_commands.add_parser(
        "record",
        help="record one below-power result so it is kept instead of re-litigated; "
        "an interval containing zero is NOT a negative and belongs here",
    )
    weak_signals_record.add_argument("--name", required=True)
    weak_signals_record.add_argument("--description", required=True)
    weak_signals_record.add_argument(
        "--source", required=True, help="artifact path or doc that records the measurement"
    )
    weak_signals_record.add_argument("--effect", type=float, required=True)
    weak_signals_record.add_argument("--effect-units", choices=tuple(EFFECT_UNITS), required=True)
    weak_signals_record.add_argument(
        "--classification", choices=tuple(CLASSIFICATIONS), required=True
    )
    weak_signals_record.add_argument("--league", choices=tuple(LEAGUES), required=True)
    weak_signals_record.add_argument("--season-start", type=int, required=True)
    weak_signals_record.add_argument("--season-end", type=int, required=True)
    weak_signals_record.add_argument("--standard-error", type=float, default=None)
    weak_signals_record.add_argument("--interval-low", type=float, default=None)
    weak_signals_record.add_argument("--interval-high", type=float, default=None)
    weak_signals_record.add_argument("--probability-positive", type=float, default=None)
    weak_signals_record.add_argument("--sample-games", type=int, default=None)
    weak_signals_record.add_argument("--sample-blocks", type=int, default=None)
    weak_signals_record.add_argument(
        "--reliability",
        type=float,
        default=None,
        help=(
            "split-half reliability of the underlying trait. AGENTS.md makes "
            "this the decisive field: an unreliable trait is refuted because no "
            "sample size rescues it, so a signal recorded without it cannot be "
            "adjudicated later"
        ),
    )
    weak_signals_record.add_argument(
        "--family",
        default=None,
        help=(
            "measurement family this cell belongs to (e.g. its screening battery). "
            "Family members share windows and are correlated, not independent votes; "
            "omit to have one inferred from the name"
        ),
    )
    weak_signals_record.add_argument(
        "--classification-evidence",
        default="",
        help="why this classification and not one of the other two",
    )
    weak_signals_record.add_argument(
        "--closing-ground",
        choices=tuple(
            ground for grounds in WEAK_SIGNAL_CLOSING_GROUNDS.values() for ground in grounds
        ),
        default=None,
        help="required for a terminal classification: the admissible AGENTS.md "
        "ground the closure stands on. An interval containing zero is NOT one; "
        "that outcome is unresolved_below_power",
    )
    weak_signals_record.add_argument(
        "--plain-summary",
        default=None,
        help=(
            "one or two sentences a football fan with no statistics background "
            "can read on their own, naming the situation AND what the rule does "
            "about it. Optional but recorded rows without one render their raw "
            "--description on the public Signal Ledger page instead"
        ),
    )
    weak_signals_record.add_argument(
        "--category",
        choices=tuple(WEAK_SIGNAL_CATEGORIES),
        default=None,
        help=(
            "one reader-facing bucket for the public Signal Ledger page. "
            "Optional but an omitted category renders under 'Uncategorised'"
        ),
    )
    weak_signals_record.add_argument("--notes", default="")
    weak_signals_record.add_argument("--recorded-at", default=None, help="default: today")
    weak_signals_record.add_argument(
        "--replace", action="store_true", help="overwrite an existing signal of this name"
    )
    weak_signals_record.set_defaults(handler=_cmd_weak_signals_record)

    weak_signals_pool = weak_signal_commands.add_parser(
        "pool",
        help="sign test plus inverse-variance pooling across the unresolved pile, "
        "with shared-season warnings; says whether a combined look is worth a window",
    )
    weak_signals_pool.add_argument("--league", choices=tuple(LEAGUES), default=None)
    weak_signals_pool.add_argument("--effect-units", choices=tuple(EFFECT_UNITS), default=None)
    weak_signals_pool.add_argument(
        "--method",
        choices=("random", "fixed"),
        default="random",
        help="random effects (default) inflates the variance by observed heterogeneity",
    )
    weak_signals_pool.set_defaults(handler=_cmd_weak_signals_pool)

    weak_signals_retag_units = weak_signal_commands.add_parser(
        "retag-units",
        help="correct a mis-tagged effect_units on one existing entry; changes ONLY "
        "the unit and appends an audit note to it, nothing else",
    )
    weak_signals_retag_units.add_argument(
        "--name", required=True, help="the recorded signal's name"
    )
    weak_signals_retag_units.add_argument(
        "--effect-units", choices=tuple(EFFECT_UNITS), required=True
    )
    weak_signals_retag_units.add_argument(
        "--reason", required=True, help="why the original unit was wrong"
    )
    weak_signals_retag_units.set_defaults(handler=_cmd_weak_signals_retag_units)

    weak_signals_set_reliability = weak_signal_commands.add_parser(
        "set-reliability",
        help="attach a measured split-half reliability (plus its interval, method and "
        "artifact) to one existing entry; changes ONLY the reliability field and "
        "appends an audit note, and never reclassifies the entry",
    )
    weak_signals_set_reliability.add_argument(
        "--name", required=True, help="the recorded signal's name"
    )
    weak_signals_set_reliability.add_argument(
        "--reliability",
        type=float,
        required=True,
        help="point estimate, a correlation in [-1, 1]; an unmeasurable reliability is "
        "reported as unmeasured, never written here as a number",
    )
    weak_signals_set_reliability.add_argument(
        "--reliability-low", type=float, required=True, help="95%% interval lower bound"
    )
    weak_signals_set_reliability.add_argument(
        "--reliability-high", type=float, required=True, help="95%% interval upper bound"
    )
    weak_signals_set_reliability.add_argument(
        "--method",
        required=True,
        help="which quantity was measured (e.g. 'team-season odd/even-week split-half of "
        "<trait>, Spearman-Brown corrected'); a trait's reliability and a flag's exposure "
        "reliability are different quantities and must not be compared",
    )
    weak_signals_set_reliability.add_argument(
        "--source", required=True, help="artifact path holding the measurement"
    )
    weak_signals_set_reliability.add_argument(
        "--reason", required=True, help="why this reliability applies to this entry"
    )
    weak_signals_set_reliability.set_defaults(handler=_cmd_weak_signals_set_reliability)

    rotation_status = rotation_commands.add_parser(
        "status", help="print every family, its windows, remaining pool capacity, and usage"
    )
    rotation_status.set_defaults(handler=_cmd_rotation_status)

    rotation_declare = rotation_commands.add_parser(
        "declare", help="declare a family BEFORE any confirmation run"
    )
    rotation_declare.add_argument("--name", required=True)
    rotation_declare.add_argument("--description", required=True)
    rotation_declare.add_argument("--grade", choices=tuple(GRADE_POOLS), required=True)
    rotation_declare.add_argument(
        "--inherits", help="comma-separated families whose spent windows this family inherits"
    )
    rotation_declare.add_argument(
        "--acknowledge-mined",
        action="store_true",
        help=f"acknowledge the {MINED_SEASONS[0]}-{MINED_SEASONS[1]} multiplicity ledger; "
        "required for any window intersecting those seasons",
    )
    rotation_declare.add_argument(
        "--plain-summary",
        default=None,
        help=(
            "one or two sentences a football fan with no statistics background can read "
            "on their own, naming the situation AND what the rule does about it. Optional; "
            "add or correct one later with `nfl-ats rotation set-plain-summary`"
        ),
    )
    rotation_declare.set_defaults(handler=_cmd_rotation_declare)

    rotation_assign = rotation_commands.add_parser(
        "assign", help="assign the earliest eligible window block to a family"
    )
    rotation_assign.add_argument("--name", required=True)
    rotation_assign.add_argument(
        "--size",
        type=int,
        help=f"window size in seasons ({MIN_WINDOW_SIZE}-{MAX_WINDOW_SIZE}); "
        "defaults to the grade's default; not valid with --stratified",
    )
    rotation_assign.add_argument(
        "--stratified",
        action="store_true",
        help="assign a two-leg era-stratified window instead of a contiguous block "
        f"({STRATIFIED_GRADE}-graded families only; "
        "docs/era_stratified_windows_proposal.md)",
    )
    rotation_assign.set_defaults(handler=_cmd_rotation_assign)

    rotation_record = rotation_commands.add_parser(
        "record", help="record the look and spend the family's assigned window"
    )
    rotation_record.add_argument("--name", required=True)
    rotation_record.add_argument("--artifact", required=True)
    rotation_record.add_argument("--verdict", choices=VERDICTS, required=True)
    rotation_record.add_argument(
        "--probability-positive",
        type=float,
        required=True,
        help="fraction of blocked resamples favouring the candidate",
    )
    rotation_record.add_argument(
        "--closing-ground",
        choices=tuple(
            ground for grounds in WEAK_SIGNAL_CLOSING_GROUNDS.values() for ground in grounds
        ),
        default=None,
        help="required for closed_negative: the admissible AGENTS.md ground the "
        "closure stands on; an interval containing zero is NOT one and that "
        "verdict is 'unresolved'",
    )
    rotation_record.add_argument(
        "--effect",
        type=float,
        default=None,
        help="point estimate, positive favours the candidate (requires --effect-units)",
    )
    rotation_record.add_argument("--effect-units", choices=tuple(EFFECT_UNITS), default=None)
    rotation_record.add_argument("--interval-low", type=float, default=None)
    rotation_record.add_argument("--interval-high", type=float, default=None)
    rotation_record.add_argument("--standard-error", type=float, default=None)
    rotation_record.add_argument("--sample-blocks", type=int, default=None)
    rotation_record.add_argument(
        "--leg-effects",
        default=None,
        help="JSON list of per-leg magnitudes, required for a stratified window: "
        '\'[{"season": 2013, "effect": 1.2, "probability_positive": 0.7, '
        '"sample_blocks": 12}, ...]\' -- one entry per leg, sharing --effect-units '
        "(owner's binding refinement: era variation is a change in magnitude, "
        "never collapsed into the pooled read alone)",
    )
    rotation_record.add_argument("--notes", default="")
    rotation_record.add_argument(
        "--replace",
        action="store_true",
        help=(
            "correct the latest spent window only; requires its exact existing artifact and "
            "preserves assignment/spend provenance"
        ),
    )
    rotation_record.set_defaults(handler=_cmd_rotation_record)

    rotation_validate = rotation_commands.add_parser(
        "validate",
        help="ENG-27: audit every family's windows for width/overlap/mined-ack/status "
        "issues; read-only, exits non-zero on any error-severity issue",
    )
    rotation_validate.add_argument(
        "--json", action="store_true", help="emit JSON instead of a table"
    )
    rotation_validate.set_defaults(handler=_cmd_rotation_validate)

    rotation_declare_coverage = rotation_commands.add_parser(
        "declare-coverage",
        help="ENG-27: for every weak-signal family lacking a rotation family, plan (or "
        "write) a no-window coverage stub or an explicit no_rotation_needed reason",
    )
    coverage_mode = rotation_declare_coverage.add_mutually_exclusive_group()
    coverage_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan only and write nothing (default)",
    )
    coverage_mode.add_argument(
        "--apply", action="store_true", help="write the plan to the rotation registry"
    )
    rotation_declare_coverage.set_defaults(handler=_cmd_rotation_declare_coverage)

    rotation_set_plain_summary = rotation_commands.add_parser(
        "set-plain-summary",
        help="attach or correct one family's reader-facing plain-English summary; "
        "changes ONLY plain_summary, nothing else about the family",
    )
    rotation_set_plain_summary.add_argument("--name", required=True)
    rotation_set_plain_summary.add_argument(
        "--plain-summary",
        required=True,
        help=(
            "one or two sentences a football fan with no statistics background can read "
            "on their own, naming the situation AND what the rule does about it"
        ),
    )
    rotation_set_plain_summary.set_defaults(handler=_cmd_rotation_set_plain_summary)
