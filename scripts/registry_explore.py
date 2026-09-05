"""Read-only registry and overlap explorer CLI (ROADMAP.md Phase 13, ENG-07).

Thin CLI wrapper around ``nfl_ats.registry_explorer`` -- see that module's
docstring for what each view means and how it was built. This lives under
``scripts/`` rather than as a new ``nfl-ats registry explore`` subcommand in
``src/nfl_ats/cli.py``: that file is large (6000+ lines) and was being edited
concurrently by other sessions' work on other ROADMAP Phase 13 items during
this one, so a new top-level subparser risked a merge conflict for no
functional benefit -- ENG-07's own task description names this script path
as the explicit fallback when "cli registration proves conflict-prone."
Nothing stops a future session from also wiring a thin ``nfl-ats registry
explore`` subcommand that imports and calls this same module.

**This script is READ-ONLY.** It calls only ``weak_signals.load_registry``
and ``rotation.load_registry`` plus the pure-function views in
``registry_explorer`` -- it never calls either registry's ``save_registry``/
``record_*`` writer, and never mutates ``registry/weak_signals.json`` or
``registry/rotation_registry.json``.

**Binding taxonomy (verbatim, since a script has no access to
AGENTS.md/CLAUDE.md's session context injection):**

    An interval or CI that contains zero is NEVER grounds to reject, fail,
    or close an experiment. At this evaluator's ~2-point resolution,
    "contains zero" is the EXPECTED outcome for a real small signal. Only
    two grounds ever close a line of work: (1) refuted mechanism -- a
    RESOLVED wrong sign (whole interval on the wrong side of zero) or zero
    split-half reliability; (2) bounded by a positive control proven able
    to detect an effect that size. Everything else is
    `unresolved_below_power`: record it with `nfl-ats weak-signals record`,
    report `probability_positive`, never the binary "contains zero". If a
    record command errors, the verdict is wrong, not the validator.

This script reports ``probability_positive`` on every row that has one and
never prints a binary "contains zero" verdict; it never computes a "games
needed" figure (banned project-wide -- within-week correlation is fixed at
zero by owner mandate).

Usage (``uv run`` prefix omitted for width; run each with
``./.tools/uv.exe run --no-sync python scripts/registry_explore.py <rest>``)::

    registry_explore.py unresolved [--league nfl] [--units accuracy_points] [--family NAME] [--json]
    registry_explore.py repeated-windows [--json]
    registry_explore.py shared-populations [--league nfl] [--units accuracy_points] [--json]
    registry_explore.py source-availability [--league nfl] [--json]
    registry_explore.py next-shots [--league nfl] [--units accuracy_points] [--top 15] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats import registry_explorer, rotation, weak_signals  # noqa: E402


def _pp(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_interval(interval: list[float] | None) -> str:
    if interval is None:
        return "n/a"
    return f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"


def _render_unresolved(rows: list[dict[str, Any]]) -> str:
    lines = [f"{len(rows)} unresolved_below_power signal(s)", ""]
    header = f"{'name':<45} {'family':<28} {'PP':>6} {'effect':>9} interval"
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        lines.append(
            f"{row['name']:<45.45} {row['family']:<28.28} {_pp(row['probability_positive']):>6} "
            f"{row['effect']:>9.4f} {_fmt_interval(row['interval'])}"
        )
    return "\n".join(lines)


def _render_repeated_windows(report: dict[str, Any]) -> str:
    lines = [report["reuse_discount_rule"], ""]
    lines.append(
        f"Cross-family reuse: {len(report['multi_family_seasons'])} season(s) touched by >1 family"
    )
    header = f"{'season':>6} {'families (count)':<10} {'families'}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in report["multi_family_seasons"]:
        lines.append(f"{row['season']:>6} {row['family_count']:<10} {', '.join(row['families'])}")
    lines.append("")
    mined_count = len(report["mined_era_windows"])
    lines.append(f"Mined-2018-2025 windows (disclosed-discount rule applies): {mined_count}")
    header2 = f"{'family':<28} {'seasons':<14} {'state':<8} {'verdict':<12} acknowledged"
    lines.append(header2)
    lines.append("-" * len(header2))
    for row in report["mined_era_windows"]:
        lines.append(
            f"{row['family']:<28.28} {row['seasons']!s:<14} {row['state']:<8} "
            f"{row['verdict']!s:<12} {row['acknowledges_mined_2018_2025']}"
        )
    return "\n".join(lines)


def _render_shared_populations(report: dict[str, Any]) -> str:
    lines = [
        report["note"],
        "",
        f"{len(report['groups'])} shared-population group(s) "
        f"(pool_summary: {report['pool_summary']['families']} families total, "
        f"{report['pool_summary']['families_with_internal_overlap']} with internal overlap)",
        "",
    ]
    header = f"{'group_id':<45} {'members':>7} {'seasons':<12} eff_N games (sum/max)"
    lines.append(header)
    lines.append("-" * len(header))
    for group in report["groups"]:
        ess = group["effective_sample_size_games"]
        lines.append(
            f"{group['group_id']:<45.45} {group['member_count']:>7} "
            f"{group['shared_seasons']!s:<12} "
            f"{ess['naive_sum_upper_bound']}/{ess['max_single_member_lower_bound']}"
        )
    return "\n".join(lines)


def _render_source_availability(rows: list[dict[str, Any]]) -> str:
    lines = [f"{len(rows)} (league, family) row(s)", ""]
    header = f"{'league':<5} {'family':<30} {'status':<26} category"
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        lines.append(
            f"{row['league']:<5} {row['family']:<30.30} {row['status']:<26} {row['category']}"
        )
    return "\n".join(lines)


def _render_next_shots(rows: list[dict[str, Any]]) -> str:
    lines = [
        f"Top {len(rows)} next shot(s), ranked by probability_positive then rotation capacity",
        "",
    ]
    header = f"{'#':>3} {'name':<42} {'family':<24} {'PP':>6} {'unspent_window':<15} group_id"
    lines.append(header)
    lines.append("-" * len(header))
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index:>3} {row['name']:<42.42} {row['family']:<24.24} "
            f"{_pp(row['probability_positive']):>6} {row['unspent_rotation_window']!s:<15} "
            f"{row['overlap_group_id']}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("view", choices=registry_explorer.VIEWS)
    parser.add_argument("--league", choices=weak_signals.LEAGUES, default=None)
    parser.add_argument(
        "--units", dest="effect_units", choices=weak_signals.EFFECT_UNITS, default=None
    )
    parser.add_argument("--family", default=None, help="only for the 'unresolved' view")
    parser.add_argument("--top", type=int, default=None, help="only for the 'next-shots' view")
    parser.add_argument(
        "--json", action="store_true", help="emit the raw JSON payload instead of a table"
    )
    parser.add_argument(
        "--weak-signals-path", type=Path, default=None, help="override registry/weak_signals.json"
    )
    parser.add_argument(
        "--rotation-path", type=Path, default=None, help="override registry/rotation_registry.json"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    weak_path = args.weak_signals_path or weak_signals.default_registry_path()
    rotation_path = args.rotation_path or rotation.default_registry_path()

    weak_registry = weak_signals.load_registry(weak_path)
    rot_registry = rotation.load_registry(rotation_path) if rotation_path.is_file() else None

    payload: Any
    if args.view == "unresolved":
        payload = registry_explorer.unresolved_signals(
            weak_registry, league=args.league, effect_units=args.effect_units, family=args.family
        )
        text = _render_unresolved(payload)
    elif args.view == "repeated-windows":
        if rot_registry is None:
            raise SystemExit(f"Rotation registry not found: {rotation_path}")
        payload = registry_explorer.repeated_windows(rot_registry)
        text = _render_repeated_windows(payload)
    elif args.view == "shared-populations":
        payload = registry_explorer.shared_population_groups(
            weak_registry, league=args.league, effect_units=args.effect_units
        )
        text = _render_shared_populations(payload)
    elif args.view == "source-availability":
        payload = registry_explorer.source_availability(weak_registry, league=args.league)
        text = _render_source_availability(payload)
    else:
        assert args.view == "next-shots"
        if rot_registry is None:
            raise SystemExit(f"Rotation registry not found: {rotation_path}")
        payload = registry_explorer.next_shots(
            weak_registry,
            rot_registry,
            league=args.league,
            effect_units=args.effect_units,
            top=args.top,
        )
        text = _render_next_shots(payload)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
