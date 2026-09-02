"""Capture Sportradar's credentialed NFL Weekly Injuries feed immutably.

The provider documents this endpoint as a weekly team/player injury report
including practice and game statuses, with a four-hour cache TTL:
https://developer.sportradar.com/football/reference/nfl-weekly-injuries

Every successful run writes the verbatim response and a canonical parquet to
a new UTC-stamped private snapshot. ``manifest.json`` is written last and pins
the capture time, source URL, response SHA-256, requested week, and coverage.
Missing credentials fail before I/O; stale, malformed, wrong-week, or
incomplete responses leave only a failed manifest and cannot be consumed.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from nfl_ats.io import atomic_bytes, atomic_json, atomic_parquet, run_id
from nfl_ats.provenance import sha256_file
from nfl_ats.source_policy import require_acquisition, require_private_raw_destination

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "data/raw/sportradar_injuries"
API_KEY_ENV = "SPORTRADAR_API_KEY"
BASE_URL = (
    "https://api.sportradar.com/nfl/official/{access_level}/v7/en/"
    "seasons/{season}/{season_type}/{week}/injuries.json"
)
MAX_SOURCE_AGE = timedelta(hours=8)
VALID_ACCESS_LEVELS = frozenset({"trial", "production"})


class SportradarInjuryCaptureError(RuntimeError):
    """A response cannot be retained as a safe injury-report snapshot."""


def source_url(season: int, week: int, season_type: str, access_level: str) -> str:
    if access_level not in VALID_ACCESS_LEVELS:
        raise SportradarInjuryCaptureError(f"Invalid Sportradar access level: {access_level}")
    return BASE_URL.format(
        access_level=access_level,
        season=season,
        season_type=season_type,
        week=week,
    )


def _schedule_context(
    repo: Path, now: datetime, schedule_path: Path | None = None
) -> tuple[int, int, set[str]]:
    path = schedule_path
    if path is None:
        candidates = sorted((repo / "data/raw").glob("*/schedules.parquet"))
        if not candidates:
            raise SportradarInjuryCaptureError("No schedule snapshot can resolve the live REG week")
        path = candidates[-1]
    schedule = pd.read_parquet(path)
    required = {"season", "week", "game_type", "gameday", "gametime", "away_team", "home_team"}
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise SportradarInjuryCaptureError(f"Schedule is missing columns: {missing}")
    regular = schedule.loc[schedule["game_type"].astype(str).eq("REG")].copy()
    local = pd.to_datetime(
        regular["gameday"].astype(str).str[:10] + " " + regular["gametime"].astype(str),
        errors="coerce",
    )
    regular["kickoff_utc"] = local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")
    ahead = regular.loc[regular["kickoff_utc"] > pd.Timestamp(now.astimezone(UTC))]
    if ahead.empty:
        raise SportradarInjuryCaptureError("Schedule has no future REG kickoff")
    first = ahead.sort_values("kickoff_utc").iloc[0]
    season, week = int(first["season"]), int(first["week"])
    slate = regular.loc[(regular["season"] == season) & (regular["week"] == week)]
    teams = set(slate["away_team"].astype(str)) | set(slate["home_team"].astype(str))
    if not teams:
        raise SportradarInjuryCaptureError("Resolved REG slate has no teams")
    return season, week, teams


def _fetch(url: str, api_key: str) -> bytes:
    response = requests.get(url, headers={"x-api-key": api_key}, timeout=60)
    response.raise_for_status()
    return response.content


def parse_response(
    payload: bytes,
    *,
    expected_season: int,
    expected_week: int,
    expected_season_type: str,
    required_teams: set[str],
    captured_at: datetime,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        raw = json.loads(payload)
        season = raw["season"]
        week = raw["week"]
        teams = raw["injuries"]
        generated_at = pd.Timestamp(raw["generated_at"])
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SportradarInjuryCaptureError(f"Malformed Weekly Injuries response: {exc}") from exc
    if not isinstance(teams, list):
        raise SportradarInjuryCaptureError("Weekly Injuries 'injuries' must be a list")
    observed = (int(season.get("year", -1)), str(season.get("type")), int(week.get("sequence", -1)))
    expected = (expected_season, expected_season_type, expected_week)
    if observed != expected:
        raise SportradarInjuryCaptureError(
            f"Wrong season/week response: {observed}, expected {expected}"
        )
    if generated_at.tzinfo is None:
        raise SportradarInjuryCaptureError("Sportradar generated_at lacks a timezone")
    age = pd.Timestamp(captured_at.astimezone(UTC)) - generated_at.tz_convert("UTC")
    if age < pd.Timedelta(0) or age > pd.Timedelta(MAX_SOURCE_AGE):
        raise SportradarInjuryCaptureError(f"Sportradar response is stale/future: age={age}")

    reported_teams: set[str] = set()
    rows: list[dict[str, Any]] = []
    for team in teams:
        if (
            not isinstance(team, dict)
            or not team.get("alias")
            or not isinstance(team.get("players"), list)
        ):
            raise SportradarInjuryCaptureError("Malformed team block in Weekly Injuries response")
        alias = str(team["alias"])
        reported_teams.add(alias)
        for player in team["players"]:
            injury = player.get("injury") if isinstance(player, dict) else None
            if not isinstance(injury, dict) or not player.get("id") or not player.get("name"):
                raise SportradarInjuryCaptureError(f"Malformed player/injury block for {alias}")
            practice = injury.get("practice") or {}
            if not isinstance(practice, dict):
                raise SportradarInjuryCaptureError(f"Malformed practice block for {alias}")
            rows.append(
                {
                    "season": expected_season,
                    "week": expected_week,
                    "season_type": expected_season_type,
                    "team": alias,
                    "player": str(player["name"]),
                    "player_id": str(player["id"]),
                    "sr_id": player.get("sr_id"),
                    "position": player.get("position"),
                    "injury": injury.get("primary"),
                    "practice_status": practice.get("status"),
                    "game_status": injury.get("status"),
                    "status_date": injury.get("status_date"),
                    "source_generated_at_utc": generated_at.tz_convert("UTC"),
                    "available_at_utc": pd.Timestamp(captured_at.astimezone(UTC)),
                }
            )
    missing_teams = sorted(required_teams - reported_teams)
    unexpected_teams = sorted(reported_teams - required_teams)
    if missing_teams:
        raise SportradarInjuryCaptureError(
            "Incomplete slate coverage; "
            f"missing={missing_teams}, extra_non_slate_teams={unexpected_teams}"
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SportradarInjuryCaptureError("Weekly Injuries response has no player rows")
    if frame[["player_id", "team"]].duplicated().any():
        raise SportradarInjuryCaptureError("Duplicate player/team rows in Weekly Injuries response")
    return frame, {
        "reported_teams": sorted(reported_teams),
        "required_teams": sorted(required_teams),
        "extra_non_slate_teams": unexpected_teams,
        "rows": len(frame),
        "source_generated_at_utc": generated_at.tz_convert("UTC").isoformat(),
    }


def capture(
    out_root: Path = DEFAULT_ROOT,
    *,
    now: datetime | None = None,
    api_key: str | None = None,
    access_level: str = "trial",
    schedule_path: Path | None = None,
    fetcher: Callable[[str, str], bytes] = _fetch,
) -> Path:
    require_acquisition("sportradar_nfl_injuries")
    require_private_raw_destination("sportradar_nfl_injuries", out_root)
    secret = api_key or os.environ.get(API_KEY_ENV)
    if not secret:
        raise SportradarInjuryCaptureError(f"{API_KEY_ENV} is required; no capture was attempted")
    captured_at = (now or datetime.now(UTC)).astimezone(UTC)
    season, week, required_teams = _schedule_context(REPO, captured_at, schedule_path)
    url = source_url(season, week, "REG", access_level)
    snapshot: Path = out_root / str(run_id(captured_at))
    snapshot.mkdir(parents=True, exist_ok=False)
    raw_path = snapshot / "source.json"
    try:
        payload = fetcher(url, secret)
        if not payload:
            raise SportradarInjuryCaptureError("Sportradar returned an empty response")
        atomic_bytes(payload, raw_path)
        frame, audit = parse_response(
            payload,
            expected_season=season,
            expected_week=week,
            expected_season_type="REG",
            required_teams=required_teams,
            captured_at=captured_at,
        )
        data_path = snapshot / "injuries.parquet"
        atomic_parquet(frame, data_path)
        atomic_json(
            {
                "status": "complete",
                "schema": "sportradar_nfl_injuries_snapshot/1",
                "snapshot_id": snapshot.name,
                "captured_at_utc": captured_at.isoformat(),
                "available_at_policy": "capture_time_not_provider_status_date",
                "source_url": url,
                "access_level": access_level,
                "season": season,
                "week": week,
                "season_type": "REG",
                "coverage": audit,
                "files": [
                    {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                    for path in (raw_path, data_path)
                ],
            },
            snapshot / "manifest.json",
        )
        return snapshot
    except Exception as exc:
        failed_manifest: dict[str, Any] = {
            "status": "failed",
            "schema": "sportradar_nfl_injuries_snapshot/1",
            "snapshot_id": snapshot.name,
            "captured_at_utc": captured_at.isoformat(),
            "source_url": url,
            "season": season,
            "week": week,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if raw_path.is_file():
            failed_manifest["files"] = [
                {
                    "path": raw_path.name,
                    "bytes": raw_path.stat().st_size,
                    "sha256": sha256_file(raw_path),
                }
            ]
        atomic_json(failed_manifest, snapshot / "manifest.json")
        if isinstance(exc, SportradarInjuryCaptureError):
            raise
        raise SportradarInjuryCaptureError(str(exc)) from exc


def load_for_decision(root: Path, decision_at: datetime) -> tuple[Path, pd.DataFrame]:
    """Load the newest verified snapshot that existed by ``decision_at``."""

    cutoff = pd.Timestamp(decision_at)
    if cutoff.tzinfo is None:
        raise SportradarInjuryCaptureError("Decision time must carry a timezone")
    eligible: list[tuple[pd.Timestamp, Path, dict[str, Any]]] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        if manifest.get("schema") != "sportradar_nfl_injuries_snapshot/1":
            raise SportradarInjuryCaptureError(f"Wrong snapshot schema in {manifest_path}")
        captured_at = pd.Timestamp(manifest.get("captured_at_utc"))
        if captured_at.tzinfo is None:
            raise SportradarInjuryCaptureError(f"Naive capture time in {manifest_path}")
        if captured_at <= cutoff.tz_convert("UTC"):
            eligible.append((captured_at, manifest_path.parent, manifest))
    if not eligible:
        raise SportradarInjuryCaptureError(f"No complete injury snapshot existed by {cutoff}")
    captured_at, snapshot, manifest = max(eligible, key=lambda item: item[0])
    entries = manifest.get("files")
    if not isinstance(entries, list) or {entry.get("path") for entry in entries} != {
        "source.json",
        "injuries.parquet",
    }:
        raise SportradarInjuryCaptureError(f"Incomplete file manifest in {snapshot}")
    for entry in entries:
        path = snapshot / str(entry["path"])
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise SportradarInjuryCaptureError(f"Snapshot file failed SHA-256 verification: {path}")
    frame = pd.read_parquet(snapshot / "injuries.parquet")
    available = pd.to_datetime(frame["available_at_utc"], utc=True, errors="coerce")
    if frame.empty or available.isna().any() or not available.eq(captured_at).all():
        raise SportradarInjuryCaptureError(
            "Snapshot availability does not match immutable capture time"
        )
    if (available > cutoff.tz_convert("UTC")).any():
        raise SportradarInjuryCaptureError(
            "Post-decision injury rows crossed the availability boundary"
        )
    return snapshot, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--access-level", choices=sorted(VALID_ACCESS_LEVELS), default="trial")
    args = parser.parse_args()
    snapshot = capture(args.out, access_level=args.access_level)
    print((snapshot / "manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
