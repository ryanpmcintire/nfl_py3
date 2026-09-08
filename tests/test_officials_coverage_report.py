"""Tests for scripts/officials_coverage_report.py (LEAD-59, lane N).

Fully offline and read-only: every fixture is a synthetic run directory
built under ``tmp_path`` (never the real ``data/raw/officials_pfr_wayback/``
archive, which a live sweep may be writing to concurrently), and the
"pending" calculation is driven by a tiny synthetic schedule parquet, not
the repository's real snapshot. Covers the four scenarios named in the
task: a complete seven-person crew, a captured page missing the referee
position, a failed fetch (no page ever landed), and a duplicate/conflicting
capture of the same ``game_id`` across two run directories.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import officials_coverage_report as ocr  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

FULL_CREW = [
    ("Referee", "John Smith"),
    ("Umpire", "Ann Ump"),
    ("Head Linesman", "Hal Lines"),
    ("Line Judge", "Lee Judge"),
    ("Field Judge", "Fiona Field"),
    ("Side Judge", "Sid Judge"),
    ("Back Judge", "Bea Judge"),
]

MISSING_REFEREE_CREW = [row for row in FULL_CREW if row[0] != "Referee"]


def _officials_html(rows: list[tuple[str, str]]) -> str:
    body = "\n".join(
        f'<tr><th data-stat="position">{position}</th><td data-stat="official">{name}</td></tr>'
        for position, name in rows
    )
    return f'<div id="content"><table id="officials"><tbody>{body}</tbody></table></div>'


def _manifest_row(
    *,
    game_id: str,
    pfr_id: str,
    season: int,
    week: int = 1,
    gameday: str = "2009-09-13",
    home_team: str = "AAA",
    away_team: str = "BBB",
    html_file: str | None = None,
    outcome: str = "fetched",
    capture_ts: str | None = None,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "pfr_id": pfr_id,
        "gameday": gameday,
        "home_team": home_team,
        "away_team": away_team,
        "wayback_capture_timestamp": capture_ts,
        "wayback_url": None,
        "html_file": html_file,
        "outcome": outcome,
        "officials_parsed": 0,
        "parse_warnings": [],
        "attempted_captures": [],
    }


def _write_manifest(
    run_dir: Path,
    *,
    run_id: str,
    season_start: int | None,
    season_end: int | None,
    games: list[dict[str, object]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": ocr.MANIFEST_SCHEMA,
        "run_id": run_id,
        "source": "internet_archive_pfr_boxscores",
        "capture_policy": "newest_post_game_capture_with_fallback",
        "season_start": season_start,
        "season_end": season_end,
        "games": games,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_html(run_dir: Path, relative_path: str, html: str) -> None:
    path = run_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _write_schedule(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "season",
            "week",
            "game_type",
            "gameday",
            "home_team",
            "away_team",
            "pfr",
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


# ---------------------------------------------------------------------------
# Unit-level: position normalization, failure classification
# ---------------------------------------------------------------------------


def test_normalize_position_aliases_down_judge_to_head_linesman() -> None:
    assert ocr.normalize_position("Down Judge") == "Head Linesman"
    assert ocr.normalize_position("Referee") == "Referee"


def test_core_crew_positions_has_exactly_seven_entries() -> None:
    assert len(ocr.CORE_CREW_POSITIONS) == 7


# ---------------------------------------------------------------------------
# Manifest loading, defensively
# ---------------------------------------------------------------------------


def test_load_run_manifest_missing_directory_has_no_manifest(tmp_path: Path) -> None:
    empty_dir = tmp_path / "no_manifest_here"
    empty_dir.mkdir()
    loaded = ocr.load_run_manifest(empty_dir)
    assert loaded.status == "missing"


def test_load_run_manifest_rejects_a_different_schema(tmp_path: Path) -> None:
    probe_dir = tmp_path / "laneN_probe_x"
    probe_dir.mkdir()
    (probe_dir / "manifest.json").write_text(
        json.dumps({"schema": "officials_pfr_wayback_probe/1", "cdx": [], "fetches": []}),
        encoding="utf-8",
    )
    loaded = ocr.load_run_manifest(probe_dir)
    assert loaded.status == "not_a_sweep_manifest"


def test_load_run_manifest_ok(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_ok"
    _write_manifest(run_dir, run_id="run_ok", season_start=2009, season_end=2009, games=[])
    loaded = ocr.load_run_manifest(run_dir)
    assert loaded.status == "ok"
    assert loaded.payload is not None
    assert loaded.payload["run_id"] == "run_ok"


def test_read_json_with_retry_recovers_from_a_transient_error() -> None:
    class FlakyPath:
        def __init__(self, good_text: str, n_failures: int) -> None:
            self.good_text = good_text
            self.n_failures = n_failures
            self.calls = 0

        def read_text(self, encoding: str = "utf-8") -> str:
            self.calls += 1
            if self.calls <= self.n_failures:
                raise OSError("simulated transient lock (mid atomic-replace)")
            return self.good_text

    flaky = FlakyPath('{"ok": true}', n_failures=2)
    payload, error = ocr._read_json_with_retry(
        flaky, attempts=5, retry_seconds=0.0, sleep_fn=lambda _seconds: None
    )
    assert error is None
    assert payload == {"ok": True}
    assert flaky.calls == 3


def test_read_json_with_retry_gives_up_after_exhausting_attempts() -> None:
    class AlwaysFlakyPath:
        def read_text(self, encoding: str = "utf-8") -> str:
            raise OSError("permanently locked")

    payload, error = ocr._read_json_with_retry(
        AlwaysFlakyPath(), attempts=3, retry_seconds=0.0, sleep_fn=lambda _seconds: None
    )
    assert payload is None
    assert error is not None and "OSError" in error


# ---------------------------------------------------------------------------
# Per-game record building: the four required scenarios
# ---------------------------------------------------------------------------


def test_build_game_record_complete_crew(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    _write_html(run_dir, "html/aaa1.html", _officials_html(FULL_CREW))
    row = _manifest_row(
        game_id="2009_01_AAA_BBB",
        pfr_id="aaa1",
        season=2009,
        html_file="html/aaa1.html",
        capture_ts="20090901000000",
    )
    record = ocr.build_game_record(run_dir, "run_a", row)
    assert record.html_on_disk is True
    assert record.complete_crew is True
    assert record.referees == ("John Smith",)
    assert record.n_crew_rows == 7


def test_build_game_record_missing_referee(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    _write_html(run_dir, "html/ccc1.html", _officials_html(MISSING_REFEREE_CREW))
    row = _manifest_row(
        game_id="2009_01_CCC_DDD",
        pfr_id="ccc1",
        season=2009,
        html_file="html/ccc1.html",
        capture_ts="20090901000000",
    )
    record = ocr.build_game_record(run_dir, "run_a", row)
    assert record.html_on_disk is True
    assert record.complete_crew is False
    assert record.referees == ()
    assert record.n_crew_rows == 6


def test_build_game_record_failed_fetch_has_no_page_on_disk(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir(parents=True)
    row = _manifest_row(
        game_id="2009_01_EEE_FFF",
        pfr_id="eee1",
        season=2009,
        html_file=None,
        outcome="cdx_fetch_failed",
    )
    record = ocr.build_game_record(run_dir, "run_a", row)
    assert record.html_on_disk is False
    assert ocr.is_captured(record) is False
    assert ocr.is_failed(record) is True
    assert ocr.failure_kind(record) == "cdx_fetch_failed"


def test_build_game_record_tolerates_html_file_named_but_not_yet_on_disk(tmp_path: Path) -> None:
    """A manifest row can name an html_file mid-write by a concurrent sweep."""

    run_dir = tmp_path / "run_a"
    run_dir.mkdir(parents=True)
    row = _manifest_row(
        game_id="2009_01_ZZZ_YYY",
        pfr_id="zzz1",
        season=2009,
        html_file="html/zzz1.html",  # never actually written in this test
        outcome="fetched",
    )
    record = ocr.build_game_record(run_dir, "run_a", row)
    assert record.html_on_disk is False
    assert any("missing on disk" in warning for warning in record.parse_warnings)


def test_failed_outcome_label_with_a_kept_page_still_counts_as_captured(tmp_path: Path) -> None:
    """``_keep_existing_page`` in the sweep can leave outcome="*_failed" while
    ``html_file`` still points at a page kept from a prior successful fetch;
    bucket classification here must trust disk truth, not the outcome label.
    """

    run_dir = tmp_path / "run_a"
    _write_html(run_dir, "html/kept.html", _officials_html(FULL_CREW))
    row = _manifest_row(
        game_id="2009_01_KEPT_GAME",
        pfr_id="kept1",
        season=2009,
        html_file="html/kept.html",
        outcome="cdx_fetch_failed",
        capture_ts="20090901000000",
    )
    record = ocr.build_game_record(run_dir, "run_a", row)
    assert ocr.is_captured(record) is True
    assert ocr.is_failed(record) is False


# ---------------------------------------------------------------------------
# Cross-run duplicate resolution
# ---------------------------------------------------------------------------


def test_resolve_canonical_games_prefers_the_newest_captured_duplicate() -> None:
    older = ocr.GameRecord(
        run_id="run_a",
        season=2009,
        week=1,
        game_id="2009_01_GGG_HHH",
        pfr_id="ggg1",
        outcome="fetched",
        html_file="html/ggg1.html",
        html_on_disk=True,
        wayback_capture_timestamp="20090901000000",
        positions=tuple(sorted(p for p, _ in FULL_CREW)),
        referees=("Old Ref",),
        n_crew_rows=7,
        complete_crew=True,
        parse_warnings=(),
    )
    newer = ocr.GameRecord(
        run_id="run_b",
        season=2009,
        week=1,
        game_id="2009_01_GGG_HHH",
        pfr_id="ggg1",
        outcome="fetched",
        html_file="html/ggg1.html",
        html_on_disk=True,
        wayback_capture_timestamp="20260101000000",
        positions=tuple(sorted(p for p, _ in FULL_CREW)),
        referees=("New Ref",),
        n_crew_rows=7,
        complete_crew=True,
        parse_warnings=(),
    )
    canonical, conflicts = ocr.resolve_canonical_games([older, newer])

    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.game_id == "2009_01_GGG_HHH"
    assert conflict.run_ids == ("run_a", "run_b")
    assert conflict.winning_run_id == "run_b"
    assert conflict.winning_capture_timestamp == "20260101000000"
    assert canonical["2009_01_GGG_HHH"].referees == ("New Ref",)


def test_resolve_canonical_games_no_conflict_for_a_single_run() -> None:
    solo = ocr.GameRecord(
        run_id="run_a",
        season=2009,
        week=1,
        game_id="2009_01_AAA_BBB",
        pfr_id="aaa1",
        outcome="fetched",
        html_file="html/aaa1.html",
        html_on_disk=True,
        wayback_capture_timestamp="20090901000000",
        positions=(),
        referees=("John Smith",),
        n_crew_rows=7,
        complete_crew=True,
        parse_warnings=(),
    )
    canonical, conflicts = ocr.resolve_canonical_games([solo])
    assert conflicts == []
    assert set(canonical) == {"2009_01_AAA_BBB"}


# ---------------------------------------------------------------------------
# End-to-end: build_summary over two synthetic run directories
# ---------------------------------------------------------------------------


@pytest.fixture
def two_run_archive(tmp_path: Path) -> tuple[Path, Path]:
    """Two run directories sharing one duplicate game_id, plus a probe-schema
    directory and a manifest-mid-write ("busy") directory, matching the four
    required scenarios (complete crew, missing referee, failed fetch,
    cross-run duplicate) plus the two tolerances the report must not choke
    on (an unrelated manifest schema, a transiently unreadable manifest).
    """

    raw_root = tmp_path / "officials_pfr_wayback"

    run_a = raw_root / "run_a"
    _write_html(run_a, "html/aaa1.html", _officials_html(FULL_CREW))
    _write_html(run_a, "html/ccc1.html", _officials_html(MISSING_REFEREE_CREW))
    _write_html(
        run_a, "html/ggg1.html", _officials_html(FULL_CREW).replace("John Smith", "Old Ref")
    )
    _write_manifest(
        run_a,
        run_id="run_a",
        season_start=2009,
        season_end=2009,
        games=[
            _manifest_row(
                game_id="2009_01_AAA_BBB",
                pfr_id="aaa1",
                season=2009,
                html_file="html/aaa1.html",
                capture_ts="20090902000000",
            ),
            _manifest_row(
                game_id="2009_01_CCC_DDD",
                pfr_id="ccc1",
                season=2009,
                html_file="html/ccc1.html",
                capture_ts="20090902000000",
            ),
            _manifest_row(
                game_id="2009_01_EEE_FFF",
                pfr_id="eee1",
                season=2009,
                html_file=None,
                outcome="cdx_fetch_failed",
            ),
            _manifest_row(
                game_id="2009_01_GGG_HHH",
                pfr_id="ggg1",
                season=2009,
                html_file="html/ggg1.html",
                capture_ts="20090901000000",
            ),
        ],
    )

    run_b = raw_root / "run_b"
    _write_html(
        run_b, "html/ggg1.html", _officials_html(FULL_CREW).replace("John Smith", "New Ref")
    )
    _write_manifest(
        run_b,
        run_id="run_b",
        season_start=2009,
        season_end=2009,
        games=[
            _manifest_row(
                game_id="2009_01_GGG_HHH",
                pfr_id="ggg1",
                season=2009,
                html_file="html/ggg1.html",
                capture_ts="20260101000000",
            ),
        ],
    )

    probe_dir = raw_root / "laneN_probe_x"
    probe_dir.mkdir(parents=True)
    (probe_dir / "manifest.json").write_text(
        json.dumps({"schema": "officials_pfr_wayback_probe/1", "cdx": [], "fetches": []}),
        encoding="utf-8",
    )

    busy_dir = raw_root / "run_busy"
    busy_dir.mkdir(parents=True)
    (busy_dir / "manifest.json").write_text("{not valid json, mid atomic-replace", encoding="utf-8")

    schedule_path = tmp_path / "schedule" / "schedules.parquet"
    _write_schedule(
        schedule_path,
        [
            {
                "game_id": pfr_id,
                "season": 2009,
                "week": 1,
                "game_type": "REG",
                "gameday": "2009-09-13",
                "home_team": "AAA",
                "away_team": "BBB",
                "pfr": pfr_id,
            }
            for pfr_id in ("aaa1", "ccc1", "eee1", "ggg1", "hhh1")
        ],
    )

    return raw_root, schedule_path


def test_build_summary_discovers_all_run_directories(two_run_archive: tuple[Path, Path]) -> None:
    raw_root, schedule_path = two_run_archive
    summary = ocr.build_summary(raw_root, schedule_path=schedule_path)

    statuses = {row["run_dir"]: row["status"] for row in summary["runs_discovered"]}
    assert statuses["run_a"] == "ok"
    assert statuses["run_b"] == "ok"
    assert statuses["laneN_probe_x"] == "not_a_sweep_manifest"
    assert statuses["run_busy"] == "busy"


def test_build_summary_per_run_season_buckets(two_run_archive: tuple[Path, Path]) -> None:
    raw_root, schedule_path = two_run_archive
    summary = ocr.build_summary(raw_root, schedule_path=schedule_path)

    buckets = {(row["run_id"], row["season"]): row for row in summary["per_run_season"]}
    run_a_2009 = buckets[("run_a", 2009)]
    assert run_a_2009["attempted"] == 4
    assert run_a_2009["captured"] == 3
    assert run_a_2009["parsed_with_crew"] == 3
    assert run_a_2009["complete_crew"] == 2
    assert run_a_2009["failed"] == 1
    assert run_a_2009["failed_fetch"] == 1
    assert run_a_2009["no_capture_found"] == 0
    assert run_a_2009["pending"] == 1  # 5 expected - 4 attempted
    assert run_a_2009["distinct_referees"] == 2  # John Smith, Old Ref

    run_b_2009 = buckets[("run_b", 2009)]
    assert run_b_2009["attempted"] == 1
    assert run_b_2009["captured"] == 1
    assert run_b_2009["complete_crew"] == 1
    assert run_b_2009["pending"] == 4  # 5 expected - 1 attempted
    assert run_b_2009["distinct_referees"] == 1


def test_build_summary_finds_the_one_cross_run_duplicate(
    two_run_archive: tuple[Path, Path],
) -> None:
    raw_root, schedule_path = two_run_archive
    summary = ocr.build_summary(raw_root, schedule_path=schedule_path)

    assert summary["n_duplicate_games"] == 1
    conflict = summary["duplicate_games"][0]
    assert conflict["game_id"] == "2009_01_GGG_HHH"
    assert conflict["winning_run_id"] == "run_b"
    assert conflict["winning_capture_timestamp"] == "20260101000000"


def test_build_summary_canonical_coverage_collapses_the_duplicate(
    two_run_archive: tuple[Path, Path],
) -> None:
    raw_root, schedule_path = two_run_archive
    summary = ocr.build_summary(raw_root, schedule_path=schedule_path)

    # 4 distinct game_ids total (aaa1, ccc1, eee1, ggg1); the duplicate
    # collapses to one canonical row, so 4, not 5.
    assert summary["n_canonical_games"] == 4

    season_row = next(
        row for row in summary["canonical_coverage_by_season"] if row["season"] == 2009
    )
    assert season_row["expected_games"] == 5
    assert season_row["games_present_any_run"] == 4
    assert season_row["captured"] == 3  # aaa1, ccc1 captured, ggg1(canonical) captured; eee1 failed
    assert season_row["complete_crew"] == 2  # aaa1 and ggg1(canonical); ccc1 missing referee
    assert season_row["complete_crew_pct_of_expected"] == pytest.approx(2 / 5)
    assert season_row["distinct_referees"] == 2  # John Smith, New Ref (canonical winner)


def test_build_summary_referee_coverage_uses_the_canonical_winner(
    two_run_archive: tuple[Path, Path],
) -> None:
    raw_root, schedule_path = two_run_archive
    summary = ocr.build_summary(raw_root, schedule_path=schedule_path)

    coverage = summary["referee_coverage"]
    assert coverage["n_distinct_referees"] == 2
    assert coverage["games_per_referee"] == {"John Smith": 1, "New Ref": 1}
    assert "Old Ref" not in coverage["games_per_referee"]  # superseded duplicate, not canonical


def test_crew_tilt_consumer_status_reports_no_consumer_and_no_stated_bar() -> None:
    status = ocr.crew_tilt_consumer_status()
    assert status["consumer_reads_wayback_archive"] is False
    assert status["stated_completeness_threshold"] is None


def test_render_text_report_smoke(two_run_archive: tuple[Path, Path]) -> None:
    raw_root, schedule_path = two_run_archive
    summary = ocr.build_summary(raw_root, schedule_path=schedule_path)
    text = ocr.render_text_report(summary)
    assert "Cross-run duplicate games: 1" in text
    assert "Referee-name coverage overall" in text
    assert "Crew-tilt feature consumer status" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_writes_a_stamped_json_artifact(
    two_run_archive: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_root, schedule_path = two_run_archive
    out_path = tmp_path / "out" / "officials_coverage_report.json"

    exit_code = ocr.main(
        [
            "--raw-root",
            str(raw_root),
            "--schedule-snapshot",
            str(schedule_path),
            "--out",
            str(out_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "_provenance_stamp" in payload
    assert payload["n_duplicate_games"] == 1

    captured = capsys.readouterr()
    assert "wrote" in captured.out
    assert "Run directories discovered" not in captured.out  # --quiet suppressed the text report


def test_main_never_writes_outside_the_requested_out_path(
    two_run_archive: tuple[Path, Path], tmp_path: Path
) -> None:
    """Regression guard: this script must never touch data/ -- only the one
    JSON path it was told to write."""

    raw_root, schedule_path = two_run_archive
    before = {p: p.stat().st_mtime for p in raw_root.rglob("*") if p.is_file()}

    ocr.main(
        [
            "--raw-root",
            str(raw_root),
            "--schedule-snapshot",
            str(schedule_path),
            "--out",
            str(tmp_path / "out.json"),
            "--quiet",
        ]
    )

    after = {p: p.stat().st_mtime for p in raw_root.rglob("*") if p.is_file()}
    assert before == after
