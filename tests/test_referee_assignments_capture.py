"""Tests for the weekly officiating-crew-assignments capture (WP22).

Covers parse correctness against the two real, trimmed fixtures in
``tests/fixtures/`` (double- and single-quoted markup variants; "at" and
"vs." matchup forms; ``<sup>`` seed annotations and "*" footnote markers),
team-code mapping, referee-name normalisation and its measured join rate
against the historical crew traits (``officials.parquet``), the
category-index/direct-guess source-selection logic, every ``empty_reason``
branch (including which ones must still exit non-zero), manifest field
presence, schedule-derived game_id/home_team/away_team resolution by
unordered team pair, and the scheduler-naming/dedupe contract the
``referee_assignments_wed`` job depends on (matching how ``player_arrests_tue``
and the ``inactives_*`` jobs already dedupe).
"""

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
WEEK18_EXCERPT_HTML = (
    FIXTURES / "footballzebras_week18_2025_referee_assignments_excerpt.html"
).read_text(encoding="utf-8")
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
    # BUF/MIA (the Ron Torbert row) deliberately absent: exercises the
    # unmatched-schedule-pair warning path without a crash.
]


# --------------------------------------------------------------------------
# Parse correctness + team-code mapping + referee-name normalisation
# --------------------------------------------------------------------------


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
    # "Team at Team" form.
    assert by_source_name["Bill Vinovich"]["team_code_a"] == "LV"
    assert by_source_name["Bill Vinovich"]["team_code_b"] == "DEN"
    assert by_source_name["Bill Vinovich"]["game_day_label"] == "Thursday, Nov. 6"
    # "Team vs. Team" form (international, designated-home game).
    assert by_source_name["Clete Blakeman"]["team_code_a"] == "ATL"
    assert by_source_name["Clete Blakeman"]["team_code_b"] == "IND"
    assert by_source_name["Clete Blakeman"]["game_day_label"] == "Sunday, Nov. 9"

    day_labels = {row["game_day_label"] for row in rows}
    assert day_labels == {"Thursday, Nov. 6", "Sunday, Nov. 9", "Monday, Nov. 10"}


def test_referee_name_alias_resolves_the_one_measured_mismatch() -> None:
    """Football Zebras prints "Ron Torbert"; officials.parquet's own
    official_name has always recorded him as "Ronald Torbert" (measured,
    src/nfl_ats/referee_assignments_capture.py module docstring)."""
    rows, _ = rac.parse_assignment_page(
        WEEK10_HTML,
        season=2025,
        week=10,
        source_url=WEEK10_URL,
        fetched_at_utc="2025-11-05T18:00:00Z",
    )
    row = next(r for r in rows if r["referee_source_name"] == "Ron Torbert")
    assert row["referee"] == "Ronald Torbert"
    # Every other referee on this page is unchanged by the alias table.
    unchanged = [r for r in rows if r["referee_source_name"] != "Ron Torbert"]
    assert all(r["referee"] == r["referee_source_name"] for r in unchanged)


def test_parse_week18_excerpt_strips_sup_seed_tags_and_asterisk_footnote() -> None:
    """Single-quoted markup variant; exercises <sup>seed</sup> annotations
    and the "*" footnote WordPress attaches to "Buccaneers*"."""
    rows, warnings = rac.parse_assignment_page(
        WEEK18_EXCERPT_HTML,
        season=2025,
        week=18,
        source_url=WEEK18_URL,
        fetched_at_utc="2025-12-29T18:00:00Z",
    )
    assert warnings == []
    assert len(rows) == 3
    by_source_name = {row["referee_source_name"]: row for row in rows}
    assert by_source_name["Bill Vinovich"]["team_code_a"] == "SEA"
    assert by_source_name["Bill Vinovich"]["team_code_b"] == "SF"
    # "Buccaneers*" must resolve to TB, not fail to map because of the "*".
    assert by_source_name["Brad Allen"]["team_code_a"] == "CAR"
    assert by_source_name["Brad Allen"]["team_code_b"] == "TB"


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


def test_empty_referee_cell_and_unparseable_matchup_are_skipped_with_warnings() -> None:
    html = (
        "<div class='b_post'><div class='b_post-game'>Bears vs Packers</div>"
        "<div class='b_post-referee'></div></div>"
        "<div class='b_post'><div class='b_post-game'>NotAMatchup</div>"
        "<div class='b_post-referee'>Bill Vinovich</div></div>"
        "<div class='b_post'><div class='b_post-game'>Nickname at Nowhere</div>"
        "<div class='b_post-referee'>Bill Vinovich</div></div>"
    )
    rows, warnings = rac.parse_assignment_page(
        html, season=2025, week=1, source_url="https://example/x", fetched_at_utc="x"
    )
    assert rows == []
    assert len(warnings) == 3
    assert "empty referee cell" in warnings[0]
    assert "unparseable matchup text" in warnings[1]
    assert "unresolved team nickname" in warnings[2]


# --------------------------------------------------------------------------
# Measured join rate against the historical officials.parquet crew traits
# --------------------------------------------------------------------------


def test_current_crew_names_join_the_historical_officials_snapshot() -> None:
    """MEASURED (this session): the 2026-season Football Zebras crew roster
    lists 17 referees; 16 match officials.parquet's official_name verbatim,
    and the alias table resolves the lone remaining mismatch (Ron Torbert).
    This pins that measurement so a future officials.parquet refresh or a
    roster change cannot silently break the join without a test failing.
    """
    officials_snapshot = sorted(
        (Path(__file__).resolve().parents[1] / "data" / "raw" / "officials").glob(
            "*/officials.parquet"
        )
    )
    if not officials_snapshot:
        pytest.skip("no local data/raw/officials/*/officials.parquet snapshot to join against")
    officials = pd.read_parquet(officials_snapshot[-1])
    historical_names = set(
        officials.loc[officials["position"] == "Referee", "official_name"].unique()
    )

    current_2026_crew_referees = {
        "Brad Allen",
        "Clete Blakeman",
        "Carl Cheffers",
        "Land Clark",
        "Alan Eck",
        "Adrian Hill",
        "Shawn Hochuli",
        "John Hussey",
        "Alex Kemp",
        "Clay Martin",
        "Alex Moore",
        "Scott Novak",
        "Brad Rogers",
        "Shawn Smith",
        "Ron Torbert",
        "Bill Vinovich",
        "Craig Wrolstad",
    }
    exact_matches = current_2026_crew_referees & historical_names
    assert len(exact_matches) == 16

    aliased = {rac.normalize_referee_name(name) for name in current_2026_crew_referees}
    assert aliased <= historical_names, aliased - historical_names


# --------------------------------------------------------------------------
# find_week_url: category-index discovery
# --------------------------------------------------------------------------


def test_find_week_url_locates_a_listed_week() -> None:
    assert (
        rac.find_week_url(CATEGORY_INDEX_HTML, season=2025, week=18)
        == "https://www.footballzebras.com/2025/12/week-18-referee-assignments-2025/"
    )
    assert (
        rac.find_week_url(CATEGORY_INDEX_HTML, season=2025, week=17)
        == "https://www.footballzebras.com/2025/12/week-17-referee-assignments-2025/"
    )


def test_find_week_url_returns_none_for_a_week_not_yet_listed() -> None:
    """MEASURED 2026-09-01: this real category-index fetch does not (yet)
    list a 2026 Week 1 post -- the genuine current not_yet_published state.
    """
    assert rac.find_week_url(CATEGORY_INDEX_HTML, season=2026, week=1) is None


def test_find_week_url_does_not_confuse_week_1_with_week_18() -> None:
    assert rac.find_week_url(CATEGORY_INDEX_HTML, season=2025, week=1) is None


# --------------------------------------------------------------------------
# run_capture: source selection + empty_reason branches
# --------------------------------------------------------------------------


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
    # BUF/MIA deliberately not in the fake schedule: must resolve to None,
    # not crash or fabricate a game.
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
    """The real, current (2026-09-01) state for 2026 Week 1: the index loads
    fine and simply does not list it yet, and no schedule exists to build a
    direct-guess URL from either."""
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


def test_run_capture_direct_guess_succeeds_when_index_has_not_caught_up(tmp_path: Path) -> None:
    write_schedule(tmp_path, WEEK10_SCHEDULE_ROWS)
    guess_url = "https://www.footballzebras.com/2025/11/week-10-referee-assignments-2025/"
    fetch, calls = make_fetch(
        {
            rac.CATEGORY_URL: (CATEGORY_INDEX_HTML, 200, None, True),  # does not list week 10
            guess_url: (WEEK10_HTML, 200, None, True),
        }
    )
    out_root = tmp_path / "data" / "players" / "referee_assignments"

    snapshot, ok = rac.run_capture(
        season=2025, week=10, out_root=out_root, repo=tmp_path, fetch=fetch, now=FIXED_NOW
    )

    assert ok is True
    assert calls == [rac.CATEGORY_URL, guess_url]
    manifest = _read_manifest(snapshot)
    assert manifest["source_used"] == "direct_guess"
    assert manifest["row_count"] == 14
    assert any("category index did not list this week yet" in w for w in manifest["warnings"])


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

    assert ok is False  # the index says it exists; failing to read it is a bug to fix
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


# --------------------------------------------------------------------------
# Off-season / no-schedule zero-row behaviour (resolved via --current)
# --------------------------------------------------------------------------


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


def test_run_capture_season_complete_is_zero_row_ok(tmp_path: Path) -> None:
    write_schedule(
        tmp_path,
        [
            {
                "season": 2024,
                "week": 1,
                "game_type": "REG",
                "game_id": "2024_01_DEN_KC",
                "home_team": "KC",
                "away_team": "DEN",
                "gameday": "2024-09-05",
                "gametime": "20:20:00",
            }
        ],
    )

    def fetch(url: str, robots_url: str) -> tuple[str | None, int | None, str | None, bool]:
        raise AssertionError("must not fetch past the end of the schedule")

    out_root = tmp_path / "data" / "players" / "referee_assignments"
    snapshot, ok = rac.run_capture(
        season=None,
        week=None,
        out_root=out_root,
        repo=tmp_path,
        fetch=fetch,
        now=FIXED_NOW,  # 2025-11-05, long after the only game in this fake schedule
    )

    assert ok is True
    manifest = _read_manifest(snapshot)
    assert manifest["empty_reason"] == rac.EMPTY_REASON_SEASON_COMPLETE


# --------------------------------------------------------------------------
# main(): argument validation
# --------------------------------------------------------------------------


def test_main_requires_current_or_explicit_season_week() -> None:
    with pytest.raises(SystemExit):
        rac.main([])


def test_main_rejects_short_delay() -> None:
    with pytest.raises(SystemExit):
        rac.main(["--current", "--delay", "0.5"])


def test_main_returns_zero_or_one_from_run_capture_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    def fake_run_capture(**kwargs: Any) -> tuple[Path, bool]:
        calls.update(kwargs)
        return tmp_path / "20251105T180000Z", kwargs["season"] == 2025

    monkeypatch.setattr(rac, "run_capture", fake_run_capture)

    exit_ok = rac.main(["--season", "2025", "--week", "10"])
    exit_bad = rac.main(["--season", "2099", "--week", "1"])

    assert exit_ok == 0
    assert exit_bad == 1
    assert calls["week"] == 1


# --------------------------------------------------------------------------
# Idempotence: the scheduler-level dedupe referee_assignments_wed relies on
# (mirrors injuries_*/player_arrests_tue/inactives_* -- there is no in-script
# "skip if already captured"; the SCHEDULER checks the newest snapshot's age
# before ever invoking the job; see scripts/capture_scheduler.py's
# already_captured()).
# --------------------------------------------------------------------------


def test_snapshot_directory_name_matches_scheduler_naming_convention(tmp_path: Path) -> None:
    fetch, _ = make_fetch({rac.CATEGORY_URL: (CATEGORY_INDEX_HTML, 200, None, True)})
    snapshot, ok = rac.run_capture(
        season=2026,
        week=1,
        out_root=tmp_path / "data" / "players" / "referee_assignments",
        repo=tmp_path,
        fetch=fetch,
        now=FIXED_NOW,
    )
    assert ok is True
    assert capture_scheduler.SNAPSHOT_NAME.match(snapshot.name)
    assert snapshot.name == "20251105T180000Z"


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
    assert age < 240  # the dedupe_minutes used by referee_assignments_wed
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    job = schedule["referee_assignments_wed"]
    satisfied, reported_age = capture_scheduler.already_captured(job, ten_minutes_later)
    assert satisfied is True
    assert reported_age is not None and reported_age < 240
