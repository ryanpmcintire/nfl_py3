"""Read-only coverage report over every officials Wayback sweep run.

LEAD-59 (``ROADMAP.md``): ``scripts/officials_wayback_sweep.py`` fetches
2009-2014 Pro-Football-Reference officiating-crew boxscores via
``web.archive.org`` into ``data/raw/officials_pfr_wayback/<run-id>/``, one
run directory per invocation, each holding a ``manifest.json`` (schema
``officials_pfr_wayback_manifest/1``, one row per game under its ``games``
key) and an ``html/`` directory of fetched pages. This script never fetches
anything and never writes into ``data/`` -- it only reads what is already on
disk across every run directory and reports:

- per (run, season): games attempted / captured (a page is on disk,
  regardless of parse outcome) / parsed with a full crew row / failed
  outright (no page ever landed) / pending (in the season's schedule but not
  yet attempted by this run).
- referee-name coverage: distinct referees and games officiated per referee.
- how many games have a COMPLETE seven-person crew (all of
  :data:`CORE_CREW_POSITIONS`, the seven position labels actually observed
  in the archive -- see that constant's docstring for the measurement).
- duplicate/conflicting captures of the same ``game_id`` across run
  directories, and which one this report treats as canonical (newest
  ``wayback_capture_timestamp`` among captured records -- an extension of
  the sweep's own intra-game "newest post-game capture" preference
  (``capture_policy`` in the manifest) to the cross-run case the sweep
  itself never has to resolve, since no two real runs to date share a
  season window).
- whether the crew-tilt feature consumers (``nfl_ats.officials_flag_features``,
  ``nfl_ats.crew_tilt_refresh_overlay``) read this archive at all, and what
  completeness bar (if any) is stated for using it -- see
  :func:`crew_tilt_consumer_status`.

Safe to run while a sweep is writing the SAME manifest concurrently
(``docs/officials_archive_probe.md`` records one prior incident where a
concurrent status read collided with the sweep's atomic
``manifest.json`` replace and drew a transient Windows ``WinError 5``):
every manifest read is wrapped in a short bounded retry
(:func:`_read_json_with_retry`), and a run whose manifest still cannot be
read after those retries is reported as ``busy``, never raised as an
exception. Each manifest is read exactly ONCE per report run (a single
:func:`load_run_manifest` call per directory); a manifest row naming an
``html_file`` that has not yet been fully written by a live sweep is
handled by checking the file's existence on disk before reading it, not by
trusting the manifest row alone -- see :func:`_parse_game_html`.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/officials_coverage_report.py

    .\\.tools\\uv.exe run --no-sync python scripts/officials_coverage_report.py `
        --out artifacts/research/laneN/officials_coverage_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import officials_wayback_sweep as sweep  # noqa: E402

from nfl_ats.provenance import utc_now, write_stamped_artifact  # noqa: E402

MANIFEST_SCHEMA = "officials_pfr_wayback_manifest/1"
DEFAULT_RAW_ROOT = REPO / "data" / "raw" / "officials_pfr_wayback"
DEFAULT_OUTPUT = REPO / "artifacts" / "research" / "laneN" / "officials_coverage_report.json"

_MANIFEST_READ_ATTEMPTS = 5
_MANIFEST_READ_RETRY_SECONDS = 0.05

CORE_CREW_POSITIONS = frozenset(
    {
        "Referee",
        "Umpire",
        "Head Linesman",
        "Line Judge",
        "Field Judge",
        "Side Judge",
        "Back Judge",
    }
)

_POSITION_ALIASES: dict[str, str] = {"Down Judge": "Head Linesman"}

_FAILURE_OUTCOMES = frozenset({"cdx_fetch_failed", "replay_fetch_failed", "no_capture_found"})


def normalize_position(position: str) -> str:
    return _POSITION_ALIASES.get(position, position)


@dataclass(frozen=True)
class ManifestLoad:
    """The outcome of trying to read one run directory's ``manifest.json``."""

    run_dir: Path
    status: str
    payload: dict[str, Any] | None
    detail: str | None = None


def discover_run_dirs(raw_root: Path) -> list[Path]:
    if not raw_root.is_dir():
        return []
    return sorted(p for p in raw_root.iterdir() if p.is_dir())


def _read_json_with_retry(
    path: Path,
    *,
    attempts: int = _MANIFEST_READ_ATTEMPTS,
    retry_seconds: float = _MANIFEST_READ_RETRY_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any] | None, str | None]:
    """Best-effort read of a JSON file, tolerant of a concurrent atomic replace.

    Returns ``(payload, None)`` on success or ``(None, last_error)`` if every
    attempt failed (a real OS error, or a body that never parsed as JSON --
    both are treated as "this file is mid-write right now," not a defect).
    """

    last_error: str | None = None
    for attempt in range(attempts):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < attempts:
                sleep_fn(retry_seconds)
            continue
        try:
            return json.loads(text), None
        except json.JSONDecodeError as exc:
            last_error = f"JSONDecodeError: {exc}"
            if attempt + 1 < attempts:
                sleep_fn(retry_seconds)
            continue
    return None, last_error


def load_run_manifest(
    run_dir: Path, *, sleep_fn: Callable[[float], None] = time.sleep
) -> ManifestLoad:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return ManifestLoad(run_dir, "missing", None, "no manifest.json in this directory")
    payload, error = _read_json_with_retry(manifest_path, sleep_fn=sleep_fn)
    if payload is None:
        return ManifestLoad(run_dir, "busy", None, error)
    if payload.get("schema") != MANIFEST_SCHEMA or "games" not in payload:
        return ManifestLoad(
            run_dir,
            "not_a_sweep_manifest",
            payload,
            f"schema={payload.get('schema')!r} (expected {MANIFEST_SCHEMA!r}) or no 'games' key "
            "-- e.g. one of this run's own ad hoc diagnostic-probe manifests, a different shape",
        )
    return ManifestLoad(run_dir, "ok", payload, None)


@dataclass(frozen=True)
class GameRecord:
    run_id: str
    season: int
    week: int
    game_id: str
    pfr_id: str
    outcome: str | None
    html_file: str | None
    html_on_disk: bool
    wayback_capture_timestamp: str | None
    positions: tuple[str, ...]
    referees: tuple[str, ...]
    n_crew_rows: int
    complete_crew: bool
    parse_warnings: tuple[str, ...]


def _parse_game_html(
    run_dir: Path, html_file: str | None
) -> tuple[list[tuple[str, str]], list[str]]:
    if not html_file:
        return [], ["no html_file recorded for this game"]
    path = run_dir / html_file
    if not path.exists():
        return [], ["html_file recorded but missing on disk (partial write or in-flight sweep)"]
    try:
        html_text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [], [f"could not read html file: {type(exc).__name__}: {exc}"]
    rows, warnings = sweep.parse_officials_block(html_text)
    return rows, list(warnings)


def build_game_record(run_dir: Path, run_id: str, row: dict[str, Any]) -> GameRecord:
    html_file = row.get("html_file")
    html_on_disk = bool(html_file) and (run_dir / str(html_file)).exists()
    rows, warnings = _parse_game_html(run_dir, html_file)
    normalized_positions = [normalize_position(position) for position, _name in rows]
    referees = tuple(name for position, name in rows if normalize_position(position) == "Referee")
    complete_crew = CORE_CREW_POSITIONS.issubset(set(normalized_positions))
    season_raw = row.get("season")
    week_raw = row.get("week")
    return GameRecord(
        run_id=run_id,
        season=int(season_raw) if season_raw is not None else -1,
        week=int(week_raw) if week_raw is not None else -1,
        game_id=str(row.get("game_id")),
        pfr_id=str(row.get("pfr_id")),
        outcome=row.get("outcome"),
        html_file=str(html_file) if html_file else None,
        html_on_disk=html_on_disk,
        wayback_capture_timestamp=row.get("wayback_capture_timestamp"),
        positions=tuple(sorted(set(normalized_positions))),
        referees=referees,
        n_crew_rows=len(rows),
        complete_crew=complete_crew,
        parse_warnings=tuple(warnings),
    )


def is_captured(record: GameRecord) -> bool:
    return record.html_on_disk


def is_failed(record: GameRecord) -> bool:
    """A manifest row is a true failure only when NO page ever landed on disk.

    A retried row can carry a ``*_failed`` outcome label (e.g.
    ``cdx_fetch_failed``) while still holding ``html_file`` populated from a
    PREVIOUS successful fetch -- ``scripts/officials_wayback_sweep.py``'s
    ``_keep_existing_page`` helper (read: lines 587-601 and 673-693) sets the
    outcome to the current attempt's failure string even when it kept an
    older page. So this classifies on disk truth (``html_on_disk``), not the
    outcome string alone; a row is a failure here iff nothing is on disk to
    show for it, regardless of what its ``outcome`` field says.
    """

    return not record.html_on_disk


def failure_kind(record: GameRecord) -> str | None:
    if is_captured(record):
        return None
    if record.outcome in _FAILURE_OUTCOMES:
        return record.outcome
    return "unknown_failure" if record.outcome is not None else "never_attempted"


@dataclass
class SeasonBucket:
    run_id: str
    season: int
    expected_games: int | None
    attempted: int = 0
    captured: int = 0
    parsed_with_crew: int = 0
    complete_crew: int = 0
    failed: int = 0
    failed_fetch: int = 0
    no_capture_found: int = 0
    pending: int | None = None
    distinct_referees: int = 0


def expected_games_by_season(
    season_start: int,
    season_end: int,
    *,
    schedule_path: Path | None = None,
) -> dict[int, int]:
    """One local, offline read of the newest schedule snapshot -- no network.

    Reuses ``officials_wayback_sweep.load_games`` verbatim so "pending" is
    counted against the EXACT same population the sweep itself iterates
    over (same REG-season filter, same drop of rows with no ``pfr`` id).
    """

    path = schedule_path or sweep.newest_schedule_snapshot()
    games = sweep.load_games(path, season_start=season_start, season_end=season_end)
    counts = games.groupby("season").size()
    return {season: int(counts.get(season, 0)) for season in range(season_start, season_end + 1)}


def build_season_buckets(
    run_id: str,
    records: list[GameRecord],
    *,
    season_start: int | None,
    season_end: int | None,
    schedule_path: Path | None = None,
    compute_pending: bool = True,
) -> list[SeasonBucket]:
    by_season: dict[int, list[GameRecord]] = defaultdict(list)
    for record in records:
        by_season[record.season].append(record)

    expected: dict[int, int] = {}
    if compute_pending and season_start is not None and season_end is not None:
        try:
            expected = expected_games_by_season(
                season_start, season_end, schedule_path=schedule_path
            )
        except SystemExit:
            expected = {}

    buckets: list[SeasonBucket] = []
    for season in sorted(by_season):
        season_records = by_season[season]
        bucket = SeasonBucket(run_id=run_id, season=season, expected_games=expected.get(season))
        bucket.attempted = len(season_records)
        referee_names: set[str] = set()
        for record in season_records:
            if is_captured(record):
                bucket.captured += 1
                if record.n_crew_rows > 0:
                    bucket.parsed_with_crew += 1
                if record.complete_crew:
                    bucket.complete_crew += 1
                referee_names.update(record.referees)
            else:
                bucket.failed += 1
                kind = failure_kind(record)
                if kind == "no_capture_found":
                    bucket.no_capture_found += 1
                else:
                    bucket.failed_fetch += 1
        bucket.distinct_referees = len(referee_names)
        if bucket.expected_games is not None:
            bucket.pending = max(bucket.expected_games - bucket.attempted, 0)
        buckets.append(bucket)
    return buckets


@dataclass(frozen=True)
class DuplicateConflict:
    game_id: str
    run_ids: tuple[str, ...]
    candidate_capture_timestamps: tuple[str | None, ...]
    winning_run_id: str
    winning_capture_timestamp: str | None


def resolve_canonical_games(
    records: list[GameRecord],
) -> tuple[dict[str, GameRecord], list[DuplicateConflict]]:
    """One record per ``game_id``, plus every cross-run duplicate found.

    On a conflict: a CAPTURED record always beats an uncaptured one, and
    among captured records the NEWEST ``wayback_capture_timestamp`` wins
    (ties broken by earlier discovery order). This mirrors the sweep's own
    intra-game "newest post-game capture" preference
    (``capture_policy: newest_post_game_capture_with_fallback`` in every
    manifest read here) extended to the cross-run case, which the sweep
    itself never has to resolve (no two real run directories share a season
    window as of this report). No games currently have more than one run's
    worth of rows -- see the report's ``duplicate_games`` list, expected
    empty today.
    """

    by_game: dict[str, list[GameRecord]] = defaultdict(list)
    for record in records:
        by_game[record.game_id].append(record)

    canonical: dict[str, GameRecord] = {}
    conflicts: list[DuplicateConflict] = []
    for game_id, group in by_game.items():
        if len(group) == 1:
            canonical[game_id] = group[0]
            continue

        def rank_key(rec: GameRecord) -> tuple[int, str]:
            captured_rank = 1 if rec.html_on_disk else 0
            timestamp = rec.wayback_capture_timestamp or ""
            return (captured_rank, timestamp)

        winner = max(group, key=rank_key)
        canonical[game_id] = winner
        conflicts.append(
            DuplicateConflict(
                game_id=game_id,
                run_ids=tuple(rec.run_id for rec in group),
                candidate_capture_timestamps=tuple(rec.wayback_capture_timestamp for rec in group),
                winning_run_id=winner.run_id,
                winning_capture_timestamp=winner.wayback_capture_timestamp,
            )
        )
    return canonical, conflicts


def referee_coverage(canonical: dict[str, GameRecord]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for record in canonical.values():
        for name in record.referees:
            counts[name] += 1
    return {
        "n_distinct_referees": len(counts),
        "games_per_referee": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def canonical_coverage_by_season(
    canonical: dict[str, GameRecord],
    *,
    expected_by_season: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    by_season: dict[int, list[GameRecord]] = defaultdict(list)
    for record in canonical.values():
        by_season[record.season].append(record)

    rows: list[dict[str, Any]] = []
    for season in sorted(by_season):
        season_records = by_season[season]
        captured = [r for r in season_records if is_captured(r)]
        parsed = [r for r in captured if r.n_crew_rows > 0]
        complete = [r for r in captured if r.complete_crew]
        referees = {name for r in captured for name in r.referees}
        expected = (expected_by_season or {}).get(season)
        rows.append(
            {
                "season": season,
                "expected_games": expected,
                "games_present_any_run": len(season_records),
                "captured": len(captured),
                "parsed_with_crew": len(parsed),
                "complete_crew": len(complete),
                "complete_crew_pct_of_expected": (
                    round(len(complete) / expected, 4) if expected else None
                ),
                "distinct_referees": len(referees),
            }
        )
    return rows


def crew_tilt_consumer_status() -> dict[str, Any]:
    """What the crew-tilt feature builders actually read, read from source.

    Read: ``src/nfl_ats/officials_flag_features.py`` and
    ``src/nfl_ats/crew_tilt_refresh_overlay.py`` both import
    ``nfl_ats.experiment_runner._latest_officials_snapshot``, which globs
    ``data/raw/officials/*/officials.parquet`` only (the nflverse
    ``load_officials()`` feed, measured 2015-2025 per
    ``docs/referee_battery.md``). Neither module, nor any other file under
    ``src/nfl_ats`` or ``scripts/`` (grepped for
    ``officials_pfr_wayback``/``officials_2009_2014``), reads
    ``data/processed/officials_pfr_wayback/*/officials_2009_2014.parquet`` --
    this archive has no consumer today. ``ROADMAP.md``'s LEAD-59 row states
    the plan ("extend the crew battery back to 2009 with the full
    seven-person crews and re-run the officials flags on production") but
    neither that row nor ``docs/referee_battery.md`` states a numeric
    completeness bar (e.g. "season X is usable once Y% of its games have a
    complete crew"). This function reports that absence rather than
    inventing one; :func:`canonical_coverage_by_season`'s
    ``complete_crew_pct_of_expected`` gives the raw number so a bar can be
    set later.
    """

    return {
        "consumer_reads_wayback_archive": False,
        "consumer_modules_checked": [
            "src/nfl_ats/officials_flag_features.py",
            "src/nfl_ats/crew_tilt_refresh_overlay.py",
            "src/nfl_ats/experiment_runner.py (_latest_officials_snapshot)",
        ],
        "actual_consumer_source": (
            "data/raw/officials/*/officials.parquet (nflverse load_officials(), measured 2015-2025)"
        ),
        "stated_completeness_threshold": None,
        "roadmap_lead59_plan": (
            "extend the crew battery (docs/referee_battery.md) back to 2009 with the full "
            "seven-person crews and re-run the officials flags on production"
        ),
        "note": (
            "no numeric completeness bar is stated anywhere for when a season built from this "
            "archive becomes usable; report the raw per-season coverage and let that decision "
            "be made explicitly"
        ),
    }


def build_summary(
    raw_root: Path = DEFAULT_RAW_ROOT,
    *,
    schedule_path: Path | None = None,
    compute_pending: bool = True,
) -> dict[str, Any]:
    run_dirs = discover_run_dirs(raw_root)
    runs_discovered: list[dict[str, Any]] = []
    all_records: list[GameRecord] = []
    per_run_season: list[SeasonBucket] = []

    for run_dir in run_dirs:
        loaded = load_run_manifest(run_dir)
        entry: dict[str, Any] = {
            "run_dir": run_dir.name,
            "status": loaded.status,
            "detail": loaded.detail,
        }
        if loaded.status != "ok" or loaded.payload is None:
            runs_discovered.append(entry)
            continue

        payload = loaded.payload
        run_id = str(payload.get("run_id") or run_dir.name)
        season_start = payload.get("season_start")
        season_end = payload.get("season_end")
        entry.update(
            {
                "run_id": run_id,
                "season_start": season_start,
                "season_end": season_end,
                "capture_policy": payload.get(
                    "capture_policy", "unknown (pre-2026-09-07 manifest schema)"
                ),
                "n_manifest_rows": len(payload.get("games") or []),
            }
        )
        runs_discovered.append(entry)

        records = [build_game_record(run_dir, run_id, row) for row in (payload.get("games") or [])]
        all_records.extend(records)

        buckets = build_season_buckets(
            run_id,
            records,
            season_start=season_start,
            season_end=season_end,
            schedule_path=schedule_path,
            compute_pending=compute_pending,
        )
        per_run_season.extend(buckets)

    canonical, conflicts = resolve_canonical_games(all_records)

    expected_by_season: dict[int, int] = {}
    ok_runs = [r for r in runs_discovered if r.get("status") == "ok"]
    if compute_pending and ok_runs:
        starts = [r["season_start"] for r in ok_runs if r.get("season_start") is not None]
        ends = [r["season_end"] for r in ok_runs if r.get("season_end") is not None]
        if starts and ends:
            try:
                expected_by_season = expected_games_by_season(
                    min(starts), max(ends), schedule_path=schedule_path
                )
            except SystemExit:
                expected_by_season = {}

    summary: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "raw_root": str(raw_root),
        "runs_discovered": runs_discovered,
        "per_run_season": [asdict(bucket) for bucket in per_run_season],
        "duplicate_games": [asdict(conflict) for conflict in conflicts],
        "n_duplicate_games": len(conflicts),
        "n_canonical_games": len(canonical),
        "canonical_coverage_by_season": canonical_coverage_by_season(
            canonical, expected_by_season=expected_by_season
        ),
        "referee_coverage": referee_coverage(canonical),
        "crew_tilt_consumer": crew_tilt_consumer_status(),
        "core_crew_positions": sorted(CORE_CREW_POSITIONS),
        "position_aliases": dict(_POSITION_ALIASES),
    }
    return summary


def render_text_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"officials coverage report -- generated {summary['generated_at_utc']}")
    lines.append(f"raw root: {summary['raw_root']}")
    lines.append("")
    lines.append("Run directories discovered:")
    for run in summary["runs_discovered"]:
        if run["status"] != "ok":
            lines.append(f"  {run['run_dir']}: SKIPPED ({run['status']}: {run['detail']})")
            continue
        lines.append(
            f"  {run['run_dir']}: run_id={run['run_id']} "
            f"seasons={run['season_start']}-{run['season_end']} "
            f"capture_policy={run['capture_policy']} rows={run['n_manifest_rows']}"
        )
    lines.append("")
    lines.append(
        "Per (run, season): attempted / captured / parsed_with_crew / complete_crew / "
        "failed (fetch/no_capture) / pending / distinct_referees"
    )
    for bucket in summary["per_run_season"]:
        lines.append(
            f"  {bucket['run_id']} {bucket['season']}: "
            f"attempted={bucket['attempted']} captured={bucket['captured']} "
            f"parsed_with_crew={bucket['parsed_with_crew']} "
            f"complete_crew={bucket['complete_crew']} "
            f"failed={bucket['failed']} (fetch={bucket['failed_fetch']} "
            f"no_capture={bucket['no_capture_found']}) "
            f"pending={bucket['pending']} distinct_referees={bucket['distinct_referees']}"
        )
    lines.append("")
    lines.append(f"Cross-run duplicate games: {summary['n_duplicate_games']}")
    for conflict in summary["duplicate_games"]:
        lines.append(
            f"  {conflict['game_id']}: runs={conflict['run_ids']} "
            f"candidates={conflict['candidate_capture_timestamps']} "
            f"winner={conflict['winning_run_id']}@{conflict['winning_capture_timestamp']}"
        )
    lines.append("")
    lines.append(f"Canonical games across all runs: {summary['n_canonical_games']}")
    lines.append("Canonical coverage by season (expected from local schedule snapshot):")
    for row in summary["canonical_coverage_by_season"]:
        pct = row["complete_crew_pct_of_expected"]
        pct_text = f"{pct:.1%}" if pct is not None else "n/a"
        lines.append(
            f"  {row['season']}: expected={row['expected_games']} "
            f"present={row['games_present_any_run']} captured={row['captured']} "
            f"parsed_with_crew={row['parsed_with_crew']} complete_crew={row['complete_crew']} "
            f"({pct_text} of expected) distinct_referees={row['distinct_referees']}"
        )
    lines.append("")
    ref_cov = summary["referee_coverage"]
    lines.append(
        f"Referee-name coverage overall: {ref_cov['n_distinct_referees']} distinct referees"
    )
    for name, count in list(ref_cov["games_per_referee"].items())[:10]:
        lines.append(f"  {name}: {count} games")
    if len(ref_cov["games_per_referee"]) > 10:
        lines.append(f"  ... and {len(ref_cov['games_per_referee']) - 10} more")
    lines.append("")
    consumer = summary["crew_tilt_consumer"]
    lines.append("Crew-tilt feature consumer status:")
    lines.append(f"  reads this archive: {consumer['consumer_reads_wayback_archive']}")
    lines.append(f"  actual source: {consumer['actual_consumer_source']}")
    lines.append(f"  stated completeness threshold: {consumer['stated_completeness_threshold']}")
    lines.append(f"  note: {consumer['note']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--schedule-snapshot", type=Path, default=None)
    parser.add_argument(
        "--no-pending",
        action="store_true",
        help="skip the local schedule read used to compute expected/pending game counts",
    )
    parser.add_argument("--quiet", action="store_true", help="skip the printed text report")
    args = parser.parse_args(argv)

    summary = build_summary(
        args.raw_root,
        schedule_path=args.schedule_snapshot,
        compute_pending=not args.no_pending,
    )
    if not args.quiet:
        print(render_text_report(summary))
    write_stamped_artifact(summary, args.out, project_root=REPO)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
