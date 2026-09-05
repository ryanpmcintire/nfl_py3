"""Operational commands: health checks, handoff and the weekly run."""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nfl_ats import __version__
from nfl_ats.cli_common import (
    _add_season_week_args,
    _artifacts_root,
    _data_root,
    _print_json,
    _registry_root,
)
from nfl_ats.handoff import check_session_handoff, write_session_handoff
from nfl_ats.lockday_package import capture_ledger_state, write_decision_package
from nfl_ats.preflight import preflight_exit_code, run_preflight
from nfl_ats.snapshots import describe_snapshot, latest_snapshot
from nfl_ats.weekly import run_weekly


def _cmd_doctor(_: argparse.Namespace) -> None:
    import nflreadpy
    import sklearn

    data_root = _data_root()
    payload: dict[str, Any] = {
        "nfl_ats_version": __version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "nflreadpy": getattr(nflreadpy, "__version__", "unknown"),
        "scikit_learn": sklearn.__version__,
        "data_root": str(data_root.resolve()),
        "artifacts_root": str(_artifacts_root().resolve()),
    }
    try:
        payload["latest_snapshot"] = describe_snapshot(latest_snapshot(data_root / "raw"))
    except FileNotFoundError:
        payload["latest_snapshot"] = None
    _print_json(payload)


def _cmd_handoff(args: argparse.Namespace) -> None:
    if args.check:
        result = check_session_handoff(
            Path.cwd(), _artifacts_root(), args.destination, registry_root=_registry_root()
        )
    else:
        result = write_session_handoff(
            Path.cwd(),
            _artifacts_root(),
            args.destination,
            registry_root=_registry_root(),
        )
    _print_json(result)


def _cmd_preflight(args: argparse.Namespace) -> None:
    report = run_preflight(
        Path.cwd(),
        data_root=_data_root(),
        artifacts_root=_artifacts_root(),
        registry_root=_registry_root(),
    )
    if args.json:
        _print_json(report.to_dict())
    else:
        for check in report.checks:
            marker = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}[check.status]
            print(f"[{marker}] ({check.category}) {check.name}: {check.detail}")
            if check.remedy:
                print(f"    remedy: {check.remedy}")
        counts = report.counts()
        print(f"\n{counts['ok']} ok, {counts['warn']} warn, {counts['fail']} fail")
    code = preflight_exit_code(report, strict=args.strict)
    if code != 0:
        raise SystemExit(code)


@dataclass(frozen=True)
class WeeklyRunRequest:
    """Everything ``nfl-ats weekly-run`` needs from the command line.

    Built by :func:`parse_weekly_run_request` and consumed by
    :func:`orchestrate_weekly_run`. It holds no environment lookups, so a
    test can assert on it directly."""

    season: int
    week: int
    refresh_player_data: bool
    skip_ingest: bool
    skip_prospective: bool
    skip_drift: bool
    record_decisions: bool
    dry_run: bool
    no_package: bool


def parse_weekly_run_request(args: argparse.Namespace) -> WeeklyRunRequest:
    """Validate the parsed namespace into a WeeklyRunRequest.

    Pure: reads only ``args`` and raises exactly what reading a missing or
    ill-typed attribute raises today."""

    return WeeklyRunRequest(
        season=args.season,
        week=args.week,
        refresh_player_data=bool(args.refresh_player_data),
        skip_ingest=bool(args.skip_ingest),
        skip_prospective=bool(args.skip_prospective),
        skip_drift=bool(args.skip_drift),
        record_decisions=bool(args.record_decisions),
        dry_run=bool(args.dry_run),
        no_package=bool(getattr(args, "no_package", False)),
    )


def orchestrate_weekly_run(request: WeeklyRunRequest) -> dict[str, Any]:
    """Run the weekly sequence and return its JSON summary.

    Writes the lock-day decision package as a side effect in exactly the
    place the handler used to (a ``finally`` block, fail-open). The caller
    is responsible only for printing the returned summary."""

    # ENG-01 (docs/lockday_package.md): the real lock (--record-decisions)
    # additionally writes one immutable decision package linking inputs, model
    # identity, outputs, recorder results, ledger writes and lockday_verify.
    # Two contracts. The "before" ledger state must be read BEFORE anything
    # runs -- that is the only moment it exists. And the package write is
    # fail-safe: by the time it runs the rows are appended and the card is
    # published, so it sits in a finally, behind a never-raising writer AND a
    # local guard, and can never abort or roll back a lock that already
    # happened.
    artifacts_root = _artifacts_root()
    write_package = bool(request.record_decisions) and not bool(request.no_package)
    ledgers_before = capture_ledger_state(artifacts_root) if write_package else None
    summary: dict[str, Any] = {}
    try:
        summary = run_weekly(
            season=request.season,
            week=request.week,
            data_root=_data_root(),
            artifacts_root=artifacts_root,
            refresh_player_data=request.refresh_player_data,
            skip_ingest=request.skip_ingest,
            skip_prospective=request.skip_prospective,
            skip_drift=request.skip_drift,
            record_decisions=request.record_decisions,
            dry_run=request.dry_run,
        )
    finally:
        if write_package and not request.dry_run:
            try:
                package = write_decision_package(
                    season=request.season,
                    week=request.week,
                    artifacts_root=artifacts_root,
                    data_root=_data_root(),
                    repo_root=Path.cwd(),
                    run_summary=summary,
                    ledger_state_before=ledgers_before,
                )
                summary["decision_package"] = package
                print(
                    f"lock-day decision package: {package.get('package_directory')}",
                    file=sys.stderr,
                )
            except Exception as package_error:  # deliberately broad; see above
                summary["decision_package"] = {
                    "written": False,
                    "ok": False,
                    "package_directory": None,
                    "errors": [{"component": "_cmd_weekly_run", "error": str(package_error)}],
                }
                print(
                    f"lock-day decision package FAILED to write: {package_error}. "
                    "The lock itself is unaffected -- the ledger rows and the card "
                    "were written before this step.",
                    file=sys.stderr,
                )
    return summary


def _cmd_weekly_run(args: argparse.Namespace) -> None:
    result = orchestrate_weekly_run(parse_weekly_run_request(args))
    _print_json(result)


def register_health(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the runtime health commands."""

    doctor = subparsers.add_parser("doctor", help="show runtime and data health")
    doctor.set_defaults(handler=_cmd_doctor)

    preflight = subparsers.add_parser(
        "preflight",
        help=(
            "read-only environment/configuration check (Python, uv, hooks, writable "
            "destinations, source policy, local research-data inventory)"
        ),
    )
    preflight.add_argument("--json", action="store_true", help="emit a machine-readable report")
    preflight.add_argument(
        "--strict",
        action="store_true",
        help=(
            "also exit nonzero on any non-ok row, including missing research data "
            "(default: only environment/configuration fail rows are fatal)"
        ),
    )
    preflight.set_defaults(handler=_cmd_preflight)


def register_handoff(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the session-handoff command."""

    handoff = subparsers.add_parser(
        "handoff",
        help="refresh the tracked new-session handoff from Git and local model state",
    )
    handoff.add_argument("--destination", type=Path, default=Path("HANDOFF.md"))
    handoff.add_argument(
        "--check",
        action="store_true",
        help="verify tracked handoff freshness without changing files",
    )
    handoff.set_defaults(handler=_cmd_handoff)


def register_weekly(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the weekly-run orchestrator command."""

    weekly = subparsers.add_parser(
        "weekly-run",
        help="run the whole Tuesday sequence in order, fail-closed, and publish",
    )
    _add_season_week_args(weekly, required=True)
    weekly.add_argument(
        "--refresh-player-data",
        action="store_true",
        help="build the enriched tables from the latest player/PBP snapshots instead of "
        "the snapshot ids pinned in the current production manifests",
    )
    weekly.add_argument(
        "--skip-ingest",
        action="store_true",
        help="reuse the latest nflverse snapshot instead of downloading a fresh one",
    )
    weekly.add_argument(
        "--skip-prospective",
        action="store_true",
        help="skip steps 8-11, which produce, record and settle the prospective 2026 "
        "challenger evidence; they run after the publish and never block the card",
    )
    weekly.add_argument(
        "--skip-drift",
        action="store_true",
        help="skip step 13 (drift-report), which writes a read-only drift-monitoring "
        "telemetry artifact after the publish; it never blocks the card",
    )
    weekly.add_argument(
        "--record-decisions",
        action="store_true",
        help=(
            "the real weekly lock: append this card's picks to the paper-decision ledger "
            "(step 7) and the challenger's picks to the prospective ledger (step 10). Off "
            "by default so an ordinary/rehearsal weekly-run does not touch either ledger; "
            "pass this only for the actual Tuesday lock. Both underlying recorders also "
            "refuse to write when this week's earliest kickoff is more than "
            "RECORDING_LOCK_WINDOW away, so this flag alone cannot reach the ledger outside "
            "the real lock week either."
        ),
    )
    weekly.add_argument(
        "--no-package",
        action="store_true",
        help=(
            "skip the immutable lock-day decision package (ENG-01, "
            "docs/lockday_package.md) that --record-decisions otherwise writes to "
            "artifacts/lockday_packages/ as the run's last step. The package is "
            "read-only evidence and never changes what the run decides, so this is "
            "an escape hatch for a constrained disk, not a normal flag"
        ),
    )
    weekly.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved plan and run nothing",
    )
    weekly.set_defaults(handler=_cmd_weekly_run)
