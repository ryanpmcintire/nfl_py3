from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from nfl_ats.refresh_triggers import (  # noqa: E402
    MKT08_DISPATCHABLE_SOURCES,
    TriggerScanRoots,
    append_triggers_to_evidence_log,
    archive_lineup_snapshot,
    detect_all_triggers,
    evidence_log_path,
)
from scripts.ingest_nflcom_injuries import resolve_current_reg_week  # noqa: E402


def _load_scheduler_state(repo: Path) -> dict[str, object]:
    path = repo / "data" / "scheduler_state.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct and optionally dispatch MKT-08 refresh triggers."
    )
    parser.add_argument("--scan", action="store_true", required=True, help="run the trigger scan")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument(
        "--current",
        action="store_true",
        help="resolve the live (season, REG week) from the newest schedule snapshot",
    )
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--artifacts-root", type=Path, default=REPO / "artifacts")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trigger-source", type=str, default="")
    args = parser.parse_args(argv)

    if args.dry_run and not args.dispatch:
        parser.error("--dry-run requires --dispatch")
    if args.dispatch and not args.trigger_source:
        parser.error("--dispatch requires --trigger-source")

    if not args.current and (args.season is None or args.week is None):
        raise SystemExit("pass --current, or both --season and --week")

    if args.current:
        season, week = resolve_current_reg_week(REPO, pd.Timestamp.now(tz="UTC"))
    else:
        season, week = int(args.season), int(args.week)

    observed = pd.Timestamp.now(tz="UTC")
    scheduler_state = _load_scheduler_state(REPO)

    dispatch_path = args.artifacts_root / "refresh_triggers" / "dispatch_state.json"
    if dispatch_path.is_file():
        dispatch_state = json.loads(dispatch_path.read_text(encoding="utf-8"))
        if not isinstance(dispatch_state, dict) or not isinstance(
            dispatch_state.get("completed"), dict
        ):
            raise SystemExit(f"invalid dispatch state: {dispatch_path}")
    else:
        dispatch_state = {
            "version": 1,
            "activated_at_utc": observed.isoformat(),
            "completed": {},
        }
        if args.dispatch and not args.dry_run:
            dispatch_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = dispatch_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(dispatch_state, indent=2) + "\n", encoding="utf-8")
            temporary.replace(dispatch_path)

    lineup_archive_dir = args.artifacts_root / "refresh_triggers" / "_lineup_archive"
    stable_lineup = args.artifacts_root / "lineups" / "current" / "lineups.json"
    if not args.dry_run:
        archive_lineup_snapshot(stable_lineup, lineup_archive_dir)

    roots = TriggerScanRoots(
        repo_root=REPO,
        data_root=args.data_root,
        artifacts_root=args.artifacts_root,
        lineup_archive_dir=lineup_archive_dir,
        scheduler_state=scheduler_state,
    )
    triggers = detect_all_triggers(roots, season=season, week=week, now=observed)

    log_path = evidence_log_path(args.artifacts_root, season=season, week=week)
    if args.dry_run:
        written, skipped = 0, len(triggers)
    else:
        written, skipped = append_triggers_to_evidence_log(log_path, triggers)

    records: dict[str, dict[str, object]] = {}
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if isinstance(record, dict):
                key = json.dumps(
                    [
                        record.get("trigger_source"),
                        record.get("source_capture_time"),
                        record.get("game_id"),
                    ],
                    separators=(",", ":"),
                )
                records[key] = record
    for trigger in triggers:
        record = trigger.to_record()
        records[json.dumps(list(trigger.dedupe_key), separators=(",", ":"))] = record

    activated = pd.Timestamp(str(dispatch_state["activated_at_utc"]))
    if activated.tzinfo is None:
        activated = activated.tz_localize("UTC")
    else:
        activated = activated.tz_convert("UTC")
    completed = dispatch_state["completed"]
    candidates: dict[str, dict[str, object]] = {}
    expired = 0
    for key, record in records.items():
        source = str(record.get("trigger_source", ""))
        capture = pd.Timestamp(str(record.get("source_capture_time", "")))
        deadline = pd.Timestamp(str(record.get("deadline", "")))
        capture = (
            capture.tz_localize("UTC") if capture.tzinfo is None else capture.tz_convert("UTC")
        )
        deadline = (
            deadline.tz_localize("UTC") if deadline.tzinfo is None else deadline.tz_convert("UTC")
        )
        if (
            source not in MKT08_DISPATCHABLE_SOURCES
            or not bool(record.get("deadline_valid"))
            or capture <= activated
            or capture > observed
            or key in completed
        ):
            continue
        if observed >= deadline:
            expired += 1
            continue
        candidates[key] = record

    dispatched = 0
    if args.dispatch and candidates and not args.dry_run:
        command = [
            sys.executable,
            "-m",
            "nfl_ats.cli",
            "refresh-picks",
            "--season",
            str(season),
            "--week",
            str(week),
            "--record-decisions",
            "--note",
            "news_trigger_dispatch",
            "--trigger-type",
            "news_event",
            "--trigger-source",
            args.trigger_source,
            "--trigger-observed-at-utc",
            observed.to_pydatetime().isoformat(),
        ]
        result = subprocess.run(command, cwd=REPO, check=False, text=True)
        if result.returncode:
            return result.returncode
        completed_at = datetime.now(UTC).isoformat()
        for key, record in candidates.items():
            completed[key] = {
                "dispatched_at_utc": completed_at,
                "game_id": record.get("game_id"),
                "trigger_source": record.get("trigger_source"),
            }
        temporary = dispatch_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(dispatch_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(dispatch_path)
        dispatched = len(candidates)

    by_source: dict[str, int] = {}
    for trigger in triggers:
        by_source[trigger.trigger_source] = by_source.get(trigger.trigger_source, 0) + 1
    invalid = sum(1 for trigger in triggers if not trigger.deadline_valid)

    print(f"season={season} week={week}")
    print(f"evidence log: {log_path}")
    print(
        f"reconstructed {len(triggers)} trigger(s); appended {written} new, "
        f"skipped {skipped} already-logged duplicate(s)"
    )
    print(f"deadline_valid=False (excluded from any future comparison): {invalid}")
    print(
        f"dispatch candidates={len(candidates)} dispatched={dispatched} expired={expired} "
        f"dry_run={args.dry_run}"
    )
    for source in sorted(by_source):
        print(f"  {source}: {by_source[source]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
