from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.capture_scheduler as capture_scheduler
from nfl_ats import referee_assignments_capture as rac

FIXTURES = Path(__file__).resolve().parent / "fixtures"
WEEK10_HTML = (FIXTURES / "footballzebras_week10_2025_referee_assignments.html").read_text(
    encoding="utf-8"
)
CATEGORY_INDEX_HTML = (FIXTURES / "footballzebras_category_assignments_index.html").read_text(
    encoding="utf-8"
)
GARBAGE_HTML = "<html><body><p>Some unrelated page with no assignment markup.</p></body></html>"

FIXED_NOW = datetime(2025, 11, 5, 18, 0, 0, tzinfo=UTC)

WEEK10_URL = "https://www.footballzebras.com/2025/11/week-10-referee-assignments-2025/"
WEEK18_URL = "https://www.footballzebras.com/2025/12/week-18-referee-assignments-2025/"


def make_fetch(
    responses: dict[str, tuple[str | None, int | None, str | None, bool]],
) -> tuple[rac.FetchFn, list[str]]:
    calls: list[str] = []

    def fetch(url: str, robots_url: str) -> tuple[str | None, int | None, str | None, bool]:
        calls.append(url)
        return responses[url]

    return fetch, calls


def write_schedule(repo: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    out_dir = repo / "data" / "raw" / "20251101T000000Z"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / "schedules.parquet", index=False)


def _read_manifest(snapshot: Path) -> dict[str, Any]:
    return json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))


WEEK10_SCHEDULE_ROWS = [
    {
        "season": 2025,
        "week": 10,
        "game_type": "REG",
        "game_id": "2025_10_LV_DEN",
        "home_team": "DEN",
        "away_team": "LV",
        "gameday": "2025-11-06",
        "gametime": "20:15:00",
    },
    {
        "season": 2025,
        "week": 10,
        "game_type": "REG",
        "game_id": "2025_10_ATL_IND",
        "home_team": "IND",
        "away_team": "ATL",
        "gameday": "2025-11-09",
        "gametime": "09:30:00",
    },
    {
        "season": 2025,
        "week": 10,
        "game_type": "REG",
        "game_id": "2025_10_NO_CAR",
        "home_team": "CAR",
        "away_team": "NO",
        "gameday": "2025-11-09",
        "gametime": "13:00:00",
    },
]


def test_parse_week10_fixture_maps_teams_and_both_matchup_forms() -> None:
    rows, warnings = rac.parse_assignment_page(
        WEEK10_HTML,
        season=2025,
        week=10,
        source_url=WEEK10_URL,
        fetched_at_utc="2025-11-05T18:00:00Z",
    )

    assert warnings == []
    assert len(rows) == 14

    by_source_name = {row["referee_source_name"]: row for row in rows}
    assert by_source_name["Bill Vinovich"]["team_code_a"] == "LV"
    assert by_source_name["Bill Vinovich"]["team_code_b"] == "DEN"
    assert by_source_name["Bill Vinovich"]["game_day_label"] == "Thursday, Nov. 6"
    assert by_source_name["Clete Blakeman"]["team_code_a"] == "ATL"
    assert by_source_name["Clete Blakeman"]["team_code_b"] == "IND"
    assert by_source_name["Clete Blakeman"]["game_day_label"] == "Sunday, Nov. 9"

    day_labels = {row["game_day_label"] for row in rows}
    assert day_labels == {"Thursday, Nov. 6", "Sunday, Nov. 9", "Monday, Nov. 10"}


def test_garbage_html_parses_to_zero_rows_no_crash() -> None:
    rows, warnings = rac.parse_assignment_page(
        GARBAGE_HTML,
        season=2025,
        week=10,
        source_url=WEEK10_URL,
        fetched_at_utc="2025-11-05T18:00:00Z",
    )
    assert rows == []
    assert warnings == []


def test_run_capture_category_index_success_resolves_schedule_join(tmp_path: Path) -> None:
    write_schedule(tmp_path, WEEK10_SCHEDULE_ROWS)
    fetch, calls = make_fetch(
        {
            rac.CATEGORY_URL: (CATEGORY_INDEX_HTML.replace("week-18", "week-10"), 200, None, True),
            "https://www.footballzebras.com/2025/12/week-10-referee-assignments-2025/": (
                WEEK10_HTML,
                200,
                None,
                True,
            ),
        }
    )
    out_root = tmp_path / "data" / "players" / "referee_assignments"

    snapshot, ok = rac.run_capture(
        season=2025, week=10, out_root=out_root, repo=tmp_path, fetch=fetch, now=FIXED_NOW
    )

    assert ok is True
    assert calls == [
        rac.CATEGORY_URL,
        "https://www.footballzebras.com/2025/12/week-10-referee-assignments-2025/",
    ]
    assert (snapshot / "category_index.html").exists()
    assert (snapshot / "post.html").exists()

    frame = pd.read_parquet(snapshot / "assignments.parquet")
    assert len(frame) == 14
    assert (frame["captured_at_utc"] == "2025-11-05T18:00:00Z").all()
    assert list(frame.columns) == rac.PARQUET_COLUMNS

    matched = frame.set_index("referee_source_name")
    assert matched.loc["Bill Vinovich", "game_id"] == "2025_10_LV_DEN"
    assert matched.loc["Bill Vinovich", "home_team"] == "DEN"
    assert matched.loc["Bill Vinovich", "away_team"] == "LV"
    assert matched.loc["Clete Blakeman", "game_id"] == "2025_10_ATL_IND"
    assert matched.loc["Clete Blakeman", "home_team"] == "IND"
    assert pd.isna(matched.loc["Ron Torbert", "game_id"])
    assert matched.loc["Ron Torbert", "referee"] == "Ronald Torbert"
    assert (frame["crew_number"].isna()).all()

    manifest = _read_manifest(snapshot)
    assert manifest["schema"] == "referee_assignments_snapshot/1"
    assert manifest["source_used"] == "category_index"
    assert manifest["row_count"] == 14
    assert manifest["ok"] is True
    assert manifest["empty_reason"] is None
    assert "DEN" in manifest["teams_seen"]
    assert "Ronald Torbert" in manifest["referees_seen"]
    assert any("no schedule match" in w for w in manifest["warnings"])


def test_run_capture_not_yet_published_is_expected_zero_row_ok(tmp_path: Path) -> None:
    fetch, calls = make_fetch({rac.CATEGORY_URL: (CATEGORY_INDEX_HTML, 200, None, True)})
    out_root = tmp_path / "data" / "players" / "referee_assignments"

    snapshot, ok = rac.run_capture(
        season=2026, week=1, out_root=out_root, repo=tmp_path, fetch=fetch, now=FIXED_NOW
    )

    assert ok is True
    assert calls == [rac.CATEGORY_URL]

    frame = pd.read_parquet(snapshot / "assignments.parquet")
    assert len(frame) == 0
    assert list(frame.columns) == rac.PARQUET_COLUMNS

    manifest = _read_manifest(snapshot)
    assert manifest["empty_reason"] == rac.EMPTY_REASON_NOT_YET_PUBLISHED
    assert manifest["ok"] is True
    assert manifest["source_used"] == "none"


def test_run_capture_unrecognized_structure_when_listed_url_parses_zero(tmp_path: Path) -> None:
    fetch, calls = make_fetch(
        {
            rac.CATEGORY_URL: (CATEGORY_INDEX_HTML, 200, None, True),
            WEEK18_URL: (GARBAGE_HTML, 200, None, True),
        }
    )
    out_root = tmp_path / "data" / "players" / "referee_assignments"

    snapshot, ok = rac.run_capture(
        season=2025, week=18, out_root=out_root, repo=tmp_path, fetch=fetch, now=FIXED_NOW
    )

    assert ok is False
    assert calls == [rac.CATEGORY_URL, WEEK18_URL]
    manifest = _read_manifest(snapshot)
    assert manifest["empty_reason"] == rac.EMPTY_REASON_UNRECOGNIZED_STRUCTURE
    assert manifest["ok"] is False
    assert manifest["row_count"] == 0


def test_run_capture_category_index_fetch_failed_exits_non_zero(tmp_path: Path) -> None:
    fetch, calls = make_fetch({rac.CATEGORY_URL: (None, 503, "http_503", True)})
    out_root = tmp_path / "data" / "players" / "referee_assignments"

    snapshot, ok = rac.run_capture(
        season=2025, week=10, out_root=out_root, repo=tmp_path, fetch=fetch, now=FIXED_NOW
    )

    assert ok is False
    assert calls == [rac.CATEGORY_URL]
    manifest = _read_manifest(snapshot)
    assert manifest["empty_reason"] == rac.EMPTY_REASON_FETCH_FAILED
    assert manifest["category_index"]["error"] == "http_503"
    assert not (snapshot / "category_index.html").exists()
    assert not (snapshot / "post.html").exists()


def test_run_capture_no_schedule_snapshot_is_zero_row_ok_and_never_fetches(tmp_path: Path) -> None:
    def fetch(url: str, robots_url: str) -> tuple[str | None, int | None, str | None, bool]:
        raise AssertionError("must not fetch when the schedule cannot even be resolved")

    out_root = tmp_path / "data" / "players" / "referee_assignments"
    snapshot, ok = rac.run_capture(
        season=None, week=None, out_root=out_root, repo=tmp_path, fetch=fetch, now=FIXED_NOW
    )

    assert ok is True
    manifest = _read_manifest(snapshot)
    assert manifest["empty_reason"] == rac.EMPTY_REASON_NO_SCHEDULE
    assert manifest["schedule_error"] is not None


def test_main_requires_current_or_explicit_season_week() -> None:
    with pytest.raises(SystemExit):
        rac.main([])


def test_scheduler_dedupe_recognizes_a_fresh_referee_assignments_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetch, _ = make_fetch({rac.CATEGORY_URL: (CATEGORY_INDEX_HTML, 200, None, True)})
    rac.run_capture(
        season=2026,
        week=1,
        out_root=tmp_path / "data" / "players" / "referee_assignments",
        repo=tmp_path,
        fetch=fetch,
        now=FIXED_NOW,
    )

    monkeypatch.setattr(capture_scheduler, "REPO", tmp_path)
    ten_minutes_later = FIXED_NOW.astimezone(capture_scheduler.ET) + pd.Timedelta(minutes=10)
    age = capture_scheduler.newest_snapshot_age_minutes(
        "data/players/referee_assignments", ten_minutes_later
    )

    assert age is not None
    assert age < 240
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    job = schedule["referee_assignments_wed"]
    satisfied, reported_age = capture_scheduler.already_captured(job, ten_minutes_later)
    assert satisfied is True
    assert reported_age is not None and reported_age < 240
