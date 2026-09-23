from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.capture_scheduler as capture_scheduler
from nfl_ats import inactives_capture as ic

FIXTURES = Path(__file__).resolve().parent / "fixtures"
POPULATED_HTML = (FIXTURES / "nflcom_inactives_populated.html").read_text(encoding="utf-8")
GARBAGE_HTML = "<html><body><p>Some unrelated page with no game markup.</p></body></html>"

FIXED_NOW = datetime(2026, 9, 7, 18, 30, 0, tzinfo=UTC)


def make_fetch(
    responses: dict[str, tuple[str | None, int | None, str | None, bool]],
) -> tuple[ic.FetchFn, list[str]]:

    calls: list[str] = []

    def fetch(url: str, robots_url: str) -> tuple[str | None, int | None, str | None, bool]:
        calls.append(url)
        return responses[url]

    return fetch, calls


def test_parse_populated_fixture_maps_team_codes_and_rows() -> None:
    rows, warnings = ic._parse_shared_design_system(
        POPULATED_HTML,
        season=2026,
        week=1,
        source_url=ic.PRIMARY_URL,
        fetched_at_utc="2026-09-07T18:30:00Z",
    )

    assert warnings == []
    assert len(rows) == 6
    teams = {row["team"] for row in rows}
    assert teams == {"KC", "DEN", "SF", "SEA"}

    by_name = {row["player_name"]: row for row in rows}
    assert by_name["Dominic Fairweather"]["status"] == "Inactive - Injury (Hamstring)"
    assert by_name["Dominic Fairweather"]["team"] == "DEN"
    assert by_name["Elijah Sandoval"]["position"] == "WR"
    assert all(row["status"] for row in rows)


def test_garbage_html_parses_to_zero_rows_no_crash() -> None:
    rows, warnings = ic._parse_shared_design_system(
        GARBAGE_HTML,
        season=2026,
        week=1,
        source_url=ic.PRIMARY_URL,
        fetched_at_utc="2026-09-07T18:30:00Z",
    )
    assert rows == []
    assert warnings == []


def test_run_capture_primary_success_writes_snapshot_and_skips_fallback(tmp_path: Path) -> None:
    fetch, calls = make_fetch(
        {
            ic.PRIMARY_URL: (POPULATED_HTML, 200, None, True),
        }
    )
    out_root = tmp_path / "data" / "players" / "inactives"

    snapshot, ok = ic.run_capture(
        season=2026,
        week=1,
        slot="sun_early",
        out_root=out_root,
        repo=tmp_path,
        fetch=fetch,
        now=FIXED_NOW,
    )

    assert ok is True
    assert calls == [ic.PRIMARY_URL]
    assert (snapshot / "primary.html").read_text(encoding="utf-8") == POPULATED_HTML
    assert not (snapshot / "fallback.html").exists()

    frame = pd.read_parquet(snapshot / "inactives.parquet")
    assert len(frame) == 6
    assert set(frame["team"]) == {"KC", "DEN", "SF", "SEA"}
    assert (frame["captured_at_utc"] == "2026-09-07T18:30:00Z").all()

    manifest = _read_manifest(snapshot)
    assert manifest["schema"] == "nflcom_inactives_snapshot/1"
    assert manifest["source_used"] == "primary"
    assert manifest["row_count"] == 6
    assert manifest["slot"] == "sun_early"
    assert manifest["season"] == 2026
    assert manifest["week"] == 1
    assert manifest["ok"] is True
    assert manifest["empty_reason"] is None
    assert manifest["teams_seen"] == ["DEN", "KC", "SEA", "SF"]
    assert manifest["primary"]["sha256"] == ic.sha256_bytes(POPULATED_HTML.encode("utf-8"))
    assert manifest["primary"]["http_status"] == 200
    assert manifest["primary"]["showed_known_placeholder"] is False
    assert manifest["fallback"] is None


def test_run_capture_both_fetches_failing_exits_non_zero(tmp_path: Path) -> None:
    fetch, calls = make_fetch(
        {
            ic.PRIMARY_URL: (None, 503, "http_503", True),
            ic.FALLBACK_URL: (None, None, "robots_disallowed", False),
        }
    )
    out_root = tmp_path / "data" / "players" / "inactives"

    snapshot, ok = ic.run_capture(
        season=2026,
        week=1,
        slot="thu_primetime",
        out_root=out_root,
        repo=tmp_path,
        fetch=fetch,
        now=FIXED_NOW,
    )

    assert ok is False
    assert calls == [ic.PRIMARY_URL, ic.FALLBACK_URL]

    manifest = _read_manifest(snapshot)
    assert manifest["empty_reason"] == ic.EMPTY_REASON_FETCH_FAILED
    assert manifest["primary"]["error"] == "http_503"
    assert manifest["fallback"]["error"] == "robots_disallowed"
    assert not (snapshot / "primary.html").exists()
    assert not (snapshot / "fallback.html").exists()


def test_run_capture_no_schedule_snapshot_is_zero_row_ok_and_never_fetches(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(url: str, robots_url: str) -> tuple[str | None, int | None, str | None, bool]:
        calls.append(url)
        raise AssertionError("must not fetch when the schedule cannot even be resolved")

    out_root = tmp_path / "data" / "players" / "inactives"
    snapshot, ok = ic.run_capture(
        season=None,
        week=None,
        slot="sun_early",
        out_root=out_root,
        repo=tmp_path,
        fetch=fetch,
        now=FIXED_NOW,
    )

    assert ok is True
    assert calls == []
    manifest = _read_manifest(snapshot)
    assert manifest["empty_reason"] == ic.EMPTY_REASON_NO_SCHEDULE
    assert manifest["schedule_error"] is not None


def test_main_requires_current_or_explicit_season_week() -> None:
    with pytest.raises(SystemExit):
        ic.main(["--slot", "sun_early"])


def test_scheduler_dedupe_recognizes_a_fresh_inactives_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    fetch, _ = make_fetch({ic.PRIMARY_URL: (POPULATED_HTML, 200, None, True)})
    ic.run_capture(
        season=2026,
        week=1,
        slot="sun_early",
        out_root=tmp_path / "data" / "players" / "inactives",
        repo=tmp_path,
        fetch=fetch,
        now=FIXED_NOW,
    )

    monkeypatch.setattr(capture_scheduler, "REPO", tmp_path)
    ten_minutes_later = FIXED_NOW.astimezone(capture_scheduler.ET) + pd.Timedelta(minutes=10)
    age = capture_scheduler.newest_snapshot_age_minutes("data/players/inactives", ten_minutes_later)

    assert age is not None
    assert age < 60
    job = capture_scheduler.Job(
        name="inactives_sun_early",
        day="sun",
        at="11:35",
        grace_minutes=15,
        command=["cmd.exe", "/c", "echo"],
        enabled=True,
        why="test",
        dedupe_dir="data/players/inactives",
        dedupe_minutes=60,
    )
    satisfied, reported_age = capture_scheduler.already_captured(job, ten_minutes_later)
    assert satisfied is True
    assert reported_age is not None and reported_age < 60


def _read_manifest(snapshot: Path) -> dict[str, Any]:
    import json

    return json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
