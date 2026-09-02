"""Tests for the official-inactives capture (WP17).

Covers parse correctness and team-code mapping against the fixtures in
``tests/fixtures/`` (one trimmed-but-verbatim real fetch of each source's
current preseason placeholder state, one CONSTRUCTED populated page --  see
that fixture's own header comment and ``src/nfl_ats/inactives_capture.py``'s
module docstring for why the populated structure could not be measured this
session), the primary/fallback source-selection logic, every ``empty_reason``
branch (including which ones must still exit non-zero), manifest field
presence, schedule-derived game_id/home_team/away_team resolution, and the
scheduler-naming contract the dedupe mechanism (this project's substitute for
in-script idempotence, matching how ``injuries_*`` and ``player_arrests_tue``
already dedupe) depends on.
"""

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
PRIMARY_PLACEHOLDER_HTML = (FIXTURES / "nflcom_inactives_placeholder.html").read_text(
    encoding="utf-8"
)
FALLBACK_PLACEHOLDER_HTML = (FIXTURES / "rotowire_inactives_placeholder.html").read_text(
    encoding="utf-8"
)
GARBAGE_HTML = "<html><body><p>Some unrelated page with no game markup.</p></body></html>"

FIXED_NOW = datetime(2026, 9, 7, 18, 30, 0, tzinfo=UTC)


def make_fetch(
    responses: dict[str, tuple[str | None, int | None, str | None, bool]],
) -> tuple[ic.FetchFn, list[str]]:
    """A fake ``FetchFn`` keyed by url, recording call order for assertions."""

    calls: list[str] = []

    def fetch(url: str, robots_url: str) -> tuple[str | None, int | None, str | None, bool]:
        calls.append(url)
        return responses[url]

    return fetch, calls


def write_schedule(repo: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    out_dir = repo / "data" / "raw" / "20260901T000000Z"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / "schedules.parquet", index=False)


# --------------------------------------------------------------------------
# Parse correctness + team-code mapping
# --------------------------------------------------------------------------


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
    # The "Reason"-headed column on the Broncos table must resolve the same
    # as a "Status"-headed one.
    assert by_name["Dominic Fairweather"]["status"] == "Inactive - Injury (Hamstring)"
    assert by_name["Dominic Fairweather"]["team"] == "DEN"
    # The "Pos"-headed column on the 49ers table must resolve like "Position".
    assert by_name["Elijah Sandoval"]["position"] == "WR"
    assert all(row["status"] for row in rows)


def test_placeholder_markers_present_in_their_own_fixtures() -> None:
    assert ic.PRIMARY_PLACEHOLDER_TEXT in PRIMARY_PLACEHOLDER_HTML
    assert ic.FALLBACK_PLACEHOLDER_TEXT in FALLBACK_PLACEHOLDER_HTML
    # And absent from the populated / garbage fixtures, or the placeholder
    # branch would wrongly short-circuit real content.
    assert ic.PRIMARY_PLACEHOLDER_TEXT not in POPULATED_HTML
    assert ic.PRIMARY_PLACEHOLDER_TEXT not in GARBAGE_HTML


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


# --------------------------------------------------------------------------
# run_capture: source selection + empty_reason branches
# --------------------------------------------------------------------------


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
    assert calls == [ic.PRIMARY_URL]  # fallback never fetched: primary already had rows
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


def test_run_capture_offseason_placeholder_is_expected_zero_row_ok(tmp_path: Path) -> None:
    fetch, calls = make_fetch(
        {
            ic.PRIMARY_URL: (PRIMARY_PLACEHOLDER_HTML, 200, None, True),
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
    assert calls == [ic.PRIMARY_URL]  # no point trying the fallback: both would agree

    frame = pd.read_parquet(snapshot / "inactives.parquet")
    assert len(frame) == 0
    assert list(frame.columns) == ic.PARQUET_COLUMNS

    manifest = _read_manifest(snapshot)
    assert manifest["empty_reason"] == ic.EMPTY_REASON_OFFSEASON_PLACEHOLDER
    assert manifest["ok"] is True
    assert manifest["source_used"] == "none"
    assert manifest["primary"]["showed_known_placeholder"] is True


def test_run_capture_falls_back_when_primary_parses_zero_without_placeholder(
    tmp_path: Path,
) -> None:
    fetch, calls = make_fetch(
        {
            ic.PRIMARY_URL: (GARBAGE_HTML, 200, None, True),
            ic.FALLBACK_URL: (POPULATED_HTML, 200, None, True),
        }
    )
    out_root = tmp_path / "data" / "players" / "inactives"

    snapshot, ok = ic.run_capture(
        season=2026,
        week=1,
        slot="sun_late",
        out_root=out_root,
        repo=tmp_path,
        fetch=fetch,
        now=FIXED_NOW,
    )

    assert ok is True
    assert calls == [ic.PRIMARY_URL, ic.FALLBACK_URL]

    manifest = _read_manifest(snapshot)
    assert manifest["source_used"] == "fallback"
    assert manifest["row_count"] == 6
    assert manifest["empty_reason"] is None
    assert any("primary source parsed 0 rows" in w for w in manifest["warnings"])
    assert manifest["fallback"]["url"] == ic.FALLBACK_URL


def test_run_capture_unrecognized_structure_both_sources_exits_non_zero(tmp_path: Path) -> None:
    fetch, calls = make_fetch(
        {
            ic.PRIMARY_URL: (GARBAGE_HTML, 200, None, True),
            ic.FALLBACK_URL: (GARBAGE_HTML, 200, None, True),
        }
    )
    out_root = tmp_path / "data" / "players" / "inactives"

    snapshot, ok = ic.run_capture(
        season=2026,
        week=1,
        slot="sat_early",
        out_root=out_root,
        repo=tmp_path,
        fetch=fetch,
        now=FIXED_NOW,
    )

    assert ok is False  # a real page with unparseable structure is a bug to fix, not a success
    assert calls == [ic.PRIMARY_URL, ic.FALLBACK_URL]

    manifest = _read_manifest(snapshot)
    assert manifest["empty_reason"] == ic.EMPTY_REASON_UNRECOGNIZED_STRUCTURE
    assert manifest["ok"] is False
    assert manifest["row_count"] == 0


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


def test_run_capture_unknown_slot_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ic.run_capture(
            season=2026,
            week=1,
            slot="not_a_real_slot",
            out_root=tmp_path,
            repo=tmp_path,
            fetch=lambda *_a: (None, None, None, True),
            now=FIXED_NOW,
        )


# --------------------------------------------------------------------------
# Off-season / no-schedule zero-row behaviour (resolved via --current)
# --------------------------------------------------------------------------


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
        repo=tmp_path,  # no data/raw/*/schedules.parquet under here
        fetch=fetch,
        now=FIXED_NOW,
    )

    assert ok is True
    assert calls == []
    manifest = _read_manifest(snapshot)
    assert manifest["empty_reason"] == ic.EMPTY_REASON_NO_SCHEDULE
    assert manifest["schedule_error"] is not None


def test_run_capture_season_complete_is_zero_row_ok(tmp_path: Path) -> None:
    write_schedule(
        tmp_path,
        [
            {
                "season": 2025,
                "week": 1,
                "game_type": "REG",
                "game_id": "2025_01_DEN_KC",
                "home_team": "KC",
                "away_team": "DEN",
                "gameday": "2025-09-07",
                "gametime": "13:00:00",
            }
        ],
    )
    calls: list[str] = []

    def fetch(url: str, robots_url: str) -> tuple[str | None, int | None, str | None, bool]:
        calls.append(url)
        raise AssertionError("must not fetch past the end of the schedule")

    out_root = tmp_path / "data" / "players" / "inactives"
    snapshot, ok = ic.run_capture(
        season=None,
        week=None,
        slot="sun_early",
        out_root=out_root,
        repo=tmp_path,
        fetch=fetch,
        now=FIXED_NOW,  # 2026-09-07, long after the only game in this fake schedule
    )

    assert ok is True
    assert calls == []
    manifest = _read_manifest(snapshot)
    assert manifest["empty_reason"] == ic.EMPTY_REASON_SEASON_COMPLETE


# --------------------------------------------------------------------------
# Schedule-derived game_id / home_team / away_team
# --------------------------------------------------------------------------


def test_run_capture_resolves_game_id_and_home_away_from_schedule(tmp_path: Path) -> None:
    write_schedule(
        tmp_path,
        [
            {
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "game_id": "2026_01_DEN_KC",
                "home_team": "KC",
                "away_team": "DEN",
                "gameday": "2026-09-13",
                "gametime": "13:00:00",
            }
            # SF/SEA deliberately absent: those rows must resolve to None,
            # not crash or fabricate a game.
        ],
    )
    fetch, _ = make_fetch({ic.PRIMARY_URL: (POPULATED_HTML, 200, None, True)})
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

    frame = pd.read_parquet(snapshot / "inactives.parquet").set_index("team")
    assert (frame.loc[["KC"], "game_id"] == "2026_01_DEN_KC").all()
    assert (frame.loc[["KC"], "home_team"] == "KC").all()
    assert (frame.loc[["KC"], "away_team"] == "DEN").all()
    assert (frame.loc[["DEN"], "game_id"] == "2026_01_DEN_KC").all()
    assert frame.loc[["SF"], "game_id"].isna().all()


# --------------------------------------------------------------------------
# main(): argument validation + exit-code passthrough
# --------------------------------------------------------------------------


def test_main_requires_current_or_explicit_season_week() -> None:
    with pytest.raises(SystemExit):
        ic.main(["--slot", "sun_early"])


def test_main_rejects_unknown_slot() -> None:
    with pytest.raises(SystemExit):
        ic.main(["--current", "--slot", "not_a_real_slot"])


def test_main_rejects_short_delay() -> None:
    with pytest.raises(SystemExit):
        ic.main(["--current", "--slot", "sun_early", "--delay", "0.5"])


def test_main_returns_zero_or_one_from_run_capture_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    def fake_run_capture(**kwargs: Any) -> tuple[Path, bool]:
        calls.update(kwargs)
        return tmp_path / "20260907T183000Z", kwargs["season"] == 2026

    monkeypatch.setattr(ic, "run_capture", fake_run_capture)

    exit_ok = ic.main(["--season", "2026", "--week", "1", "--slot", "sun_early"])
    exit_bad = ic.main(["--season", "2099", "--week", "1", "--slot", "sun_early"])

    assert exit_ok == 0
    assert exit_bad == 1
    assert calls["slot"] == "sun_early"


# --------------------------------------------------------------------------
# Idempotence: the scheduler-level dedupe this capture relies on (mirrors
# injuries_*/player_arrests_tue -- there is no in-script "skip if already
# captured", the SCHEDULER checks the newest snapshot's age before ever
# invoking the job; see scripts/capture_scheduler.py's already_captured()).
# --------------------------------------------------------------------------


def test_snapshot_directory_name_matches_scheduler_naming_convention(tmp_path: Path) -> None:
    fetch, _ = make_fetch({ic.PRIMARY_URL: (POPULATED_HTML, 200, None, True)})
    snapshot, ok = ic.run_capture(
        season=2026,
        week=1,
        slot="sun_early",
        out_root=tmp_path / "data" / "players" / "inactives",
        repo=tmp_path,
        fetch=fetch,
        now=FIXED_NOW,
    )
    assert ok is True
    assert capture_scheduler.SNAPSHOT_NAME.match(snapshot.name)
    assert snapshot.name == "20260907T183000Z"


def test_scheduler_dedupe_recognizes_a_fresh_inactives_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A second run inside the dedupe window must find this snapshot recent
    enough that the scheduler would skip re-fetching -- the actual mechanism
    that makes this capture safe to double-schedule/re-run, matching how
    injuries_*/player_arrests_tue dedupe (see docs/capture_scheduling.md).
    """

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
    assert age < 60  # the dedupe_minutes used for every inactives_* SCHEDULE row
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
