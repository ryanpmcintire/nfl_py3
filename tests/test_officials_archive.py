"""Contracts for the canonical officials archive loader (LEAD-59).

Every test builds a SYNTHETIC sweep-run tree under ``tmp_path`` and a
synthetic nflverse feed frame; no test reads ``data/raw/officials`` or
``data/raw/officials_pfr_wayback``. The one real thing borrowed from the
repository is ``scripts/officials_wayback_sweep.py``'s own
``parse_officials_block`` -- reused, never reimplemented -- so the fixtures
below are written in the same HTML shape the sweep actually captures
(``<table id="ref_info">`` with a ``<th>`` position label and a ``<td>``
name, the 2014-era layout measured in ``docs/officials_archive_probe.md``).

Two of these are the family's LEAKAGE regression tests, required by
AGENTS.md for every new feature family:

- ``test_a_capture_at_or_before_kickoff_fails_closed``
- ``test_archive_rows_are_refused_by_the_prospective_channel``

See ``docs/officials_archive.md`` for the timing contract they enforce.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_ats.officials_archive import (
    ARCHIVE_SOURCE,
    ARCHIVE_TIMING_CLASS,
    CANONICAL_CREW_COLUMNS,
    CORE_CREW_POSITIONS,
    CREW_COLUMNS,
    INCLUDE_ARCHIVE_DEFAULT,
    MANIFEST_SCHEMA,
    NFLVERSE_OFFICIALS_COLUMNS,
    NFLVERSE_SOURCE,
    OfficialsArchiveError,
    _load_sweep_module,
    archive_officials_long,
    assert_captures_are_post_game,
    canonical_crew_table,
    clear_cache,
    describe_archive_coverage,
    discover_sweep_runs,
    load_archive_crew_rows,
    load_officials,
    load_officials_for_prospective_channel,
    normalize_position,
    refuse_archive_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

FULL_CREW = (
    ("Referee", "Ed Hochuli"),
    ("Umpire", "Bruce Stritesky"),
    ("Head Linesman", "Kent Payne"),
    ("Line Judge", "Julian Mapp"),
    ("Field Judge", "Steve Zimmer"),
    ("Side Judge", "Joe Larrew"),
    ("Back Judge", "Perry Paganelli"),
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _crew_html(pairs: tuple[tuple[str, str], ...]) -> str:
    rows = "".join(f"<tr><th>{position}</th><td>{name}</td></tr>" for position, name in pairs)
    return f'<html><body><table id="ref_info"><tbody>{rows}</tbody></table></body></html>'


def _manifest_game(
    game_id: str,
    *,
    season: int,
    week: int,
    gameday: str,
    pfr_id: str,
    capture_ts: str,
    html_file: str | None,
    home_team: str = "SEA",
    away_team: str = "GB",
) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "gameday": gameday,
        "pfr_id": pfr_id,
        "home_team": home_team,
        "away_team": away_team,
        "outcome": "fetched",
        "html_file": html_file,
        "officials_parsed": 7,
        "wayback_capture_timestamp": capture_ts,
        "wayback_url": (
            f"https://web.archive.org/web/{capture_ts}id_/"
            f"https://www.pro-football-reference.com/boxscores/{pfr_id}.htm"
        ),
        "fetch_instant_utc": "2026-09-07T14:03:10Z",
    }


def _write_run(
    raw_root: Path,
    run_id: str,
    games: list[tuple[dict[str, Any], tuple[tuple[str, str], ...] | None]],
    *,
    schema: str = MANIFEST_SCHEMA,
    season_start: int = 2014,
    season_end: int = 2014,
) -> Path:
    run_dir = raw_root / run_id
    (run_dir / "html").mkdir(parents=True, exist_ok=True)
    for game, crew in games:
        html_file = game.get("html_file")
        if html_file and crew is not None:
            (run_dir / str(html_file)).write_text(_crew_html(crew), encoding="utf-8")
    payload = {
        "schema": schema,
        "run_id": run_id,
        "capture_policy": "newest_post_game_capture_with_fallback",
        "season_start": season_start,
        "season_end": season_end,
        "games": [game for game, _crew in games],
    }
    (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


def _one_game_archive(
    tmp_path: Path,
    *,
    crew: tuple[tuple[str, str], ...] = FULL_CREW,
    game_id: str = "2014_01_GB_SEA",
    capture_ts: str = "20141006175522",
    gameday: str = "2014-09-04",
) -> Path:
    raw_root = tmp_path / "officials_pfr_wayback"
    pfr_id = "201409040sea"
    game = _manifest_game(
        game_id,
        season=2014,
        week=1,
        gameday=gameday,
        pfr_id=pfr_id,
        capture_ts=capture_ts,
        html_file=f"html/{pfr_id}__{capture_ts}.html",
    )
    _write_run(raw_root, "20260907T140309Z", [(game, crew)])
    return raw_root


def _schedules(rows: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    rows = rows or [
        {
            "game_id": "2014_01_GB_SEA",
            "old_game_id": "2014090400",
            "season": 2014,
            "week": 1,
            "gameday": "2014-09-04",
        }
    ]
    return pd.DataFrame(rows)


def _feed(seasons: tuple[int, ...] = (2015, 2016)) -> pd.DataFrame:
    """A synthetic nflverse ``officials.parquet``, in its measured schema."""

    records = []
    for season in seasons:
        for index, (position, name) in enumerate(FULL_CREW):
            records.append(
                {
                    "game_id": f"{season}091000",
                    "game_key": f"key_{season}",
                    "official_name": name,
                    "position": position,
                    "jersey_number": 10 + index,
                    "official_id": str(100 + index),
                    "season": season,
                    "season_type": "REG",
                    "week": 1,
                }
            )
    frame = pd.DataFrame(records)
    return frame.astype(
        {
            "game_id": "str",
            "game_key": "str",
            "official_name": "str",
            "position": "str",
            "jersey_number": "int32",
            "official_id": "str",
            "season": "int32",
            "season_type": "str",
            "week": "int32",
        }
    )


@pytest.fixture(autouse=True)
def _no_cross_test_cache() -> Any:
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Run discovery and parsing
# ---------------------------------------------------------------------------


def test_the_retyped_source_id_matches_the_sweep_scripts_own() -> None:
    """``ARCHIVE_SOURCE`` is retyped rather than imported; pin it equal.

    This also exercises the by-path import of the sweep whose
    ``parse_officials_block`` every archive read reuses.
    """

    sweep = _load_sweep_module(REPO_ROOT)
    assert ARCHIVE_SOURCE == sweep.SOURCE_ID
    assert MANIFEST_SCHEMA == "officials_pfr_wayback_manifest/1"
    assert sweep.parse_officials_block(_crew_html(FULL_CREW))[0] == list(FULL_CREW)


def test_discover_sweep_runs_skips_directories_that_are_not_sweep_runs(tmp_path: Path) -> None:
    raw_root = _one_game_archive(tmp_path)
    probe = raw_root / "laneN_probe_20260907T233340Z"
    probe.mkdir()
    (probe / "manifest.json").write_text(
        json.dumps({"schema": "officials_pfr_wayback_probe/1", "cdx": [], "fetches": []}),
        encoding="utf-8",
    )
    (raw_root / "no_manifest_at_all").mkdir()

    runs = discover_sweep_runs(raw_root)
    assert [run.run_id for run in runs] == ["20260907T140309Z"]
    assert runs[0].capture_policy == "newest_post_game_capture_with_fallback"
    assert runs[0].n_manifest_rows == 1


def test_discover_sweep_runs_on_a_missing_root_is_empty(tmp_path: Path) -> None:
    assert discover_sweep_runs(tmp_path / "nothing_here") == []


def test_a_complete_crew_parses_to_seven_positions(tmp_path: Path) -> None:
    raw_root = _one_game_archive(tmp_path)
    rows = load_archive_crew_rows(repo_root=REPO_ROOT, raw_root=raw_root)

    assert len(rows) == 7
    assert set(rows["position"]) == set(CORE_CREW_POSITIONS)
    assert set(rows["source"]) == {ARCHIVE_SOURCE}
    assert set(rows["source_run_id"]) == {"20260907T140309Z"}
    assert list(rows["row_order"]) == [0, 1, 2, 3, 4, 5, 6]


def test_a_manifest_row_whose_page_is_not_on_disk_is_skipped(tmp_path: Path) -> None:
    """An in-flight sweep names a page before it finishes writing it, and a
    retried row can carry a ``*_failed`` label with no page at all. Disk is
    the only truth."""

    raw_root = tmp_path / "officials_pfr_wayback"
    present = _manifest_game(
        "2014_01_GB_SEA",
        season=2014,
        week=1,
        gameday="2014-09-04",
        pfr_id="201409040sea",
        capture_ts="20141006175522",
        html_file="html/201409040sea__20141006175522.html",
    )
    absent = _manifest_game(
        "2014_01_NO_ATL",
        season=2014,
        week=1,
        gameday="2014-09-07",
        pfr_id="201409070atl",
        capture_ts="20141008120000",
        html_file="html/201409070atl__20141008120000.html",
    )
    no_file = _manifest_game(
        "2014_01_MIN_STL",
        season=2014,
        week=1,
        gameday="2014-09-07",
        pfr_id="201409070ram",
        capture_ts="20141008130000",
        html_file=None,
    )
    _write_run(
        raw_root, "20260907T140309Z", [(present, FULL_CREW), (absent, None), (no_file, None)]
    )

    rows = load_archive_crew_rows(repo_root=REPO_ROOT, raw_root=raw_root)
    assert set(rows["game_id"]) == {"2014_01_GB_SEA"}


def test_a_missing_position_is_left_missing_not_invented(tmp_path: Path) -> None:
    crew = tuple(pair for pair in FULL_CREW if pair[0] != "Side Judge")
    raw_root = _one_game_archive(tmp_path, crew=crew)

    table = canonical_crew_table(
        repo_root=REPO_ROOT,
        raw_root=raw_root,
        schedules=_schedules(),
    )
    assert len(table) == 1
    row = table.iloc[0]
    assert pd.isna(row["side_judge"])
    assert row["referee"] == "Ed Hochuli"
    assert bool(row["complete_crew"]) is False
    assert int(row["n_positions"]) == 6


def test_down_judge_is_folded_into_head_linesman() -> None:
    assert normalize_position("Down Judge") == "Head Linesman"
    assert normalize_position("Head Linesman") == "Head Linesman"
    assert normalize_position("Referee") == "Referee"


# ---------------------------------------------------------------------------
# Canonical table
# ---------------------------------------------------------------------------


def test_canonical_table_has_the_frozen_columns_and_schedule_derived_fields(
    tmp_path: Path,
) -> None:
    raw_root = _one_game_archive(tmp_path)
    table = canonical_crew_table(repo_root=REPO_ROOT, raw_root=raw_root, schedules=_schedules())

    assert list(table.columns) == list(CANONICAL_CREW_COLUMNS)
    row = table.iloc[0]
    assert row["game_id"] == "2014_01_GB_SEA"
    # season/week/gameday/old_game_id come from the SCHEDULE, the authority.
    assert row["old_game_id"] == "2014090400"
    assert int(row["season"]) == 2014
    assert int(row["week"]) == 1
    assert row["gameday"] == "2014-09-04"
    assert bool(row["complete_crew"]) is True
    assert int(row["n_crew_rows"]) == 7
    assert int(row["discarded_duplicate_position_rows"]) == 0
    assert row["source"] == ARCHIVE_SOURCE
    assert row["source_run_id"] == "20260907T140309Z"
    assert row["wayback_capture_timestamp"] == "20141006175522"
    assert row["wayback_url"].startswith("https://web.archive.org/web/20141006175522id_/")
    assert [row[column] for column in CREW_COLUMNS] == [name for _p, name in FULL_CREW]


def test_a_duplicate_position_on_one_page_keeps_the_first_and_counts_the_discard(
    tmp_path: Path,
) -> None:
    """Measured on the real archive (2026-09-08): ``2014_06_DET_MIN`` lists
    "John Parry" and "John Perry" both as Referee, and ``2014_09_PHI_HOU``
    lists two Back Judges. First in page order wins, so a duplicated Referee
    can never double-count a game downstream."""

    crew = (("Referee", "John Parry"), ("Referee", "John Perry"), *FULL_CREW[1:])
    raw_root = _one_game_archive(tmp_path, crew=crew)

    table = canonical_crew_table(repo_root=REPO_ROOT, raw_root=raw_root, schedules=_schedules())
    row = table.iloc[0]
    assert row["referee"] == "John Parry"
    assert int(row["n_crew_rows"]) == 8
    assert int(row["n_positions"]) == 7
    assert int(row["discarded_duplicate_position_rows"]) == 1

    long = archive_officials_long(table=table)
    assert int((long["position"] == "Referee").sum()) == 1


def test_a_game_captured_by_two_runs_resolves_to_the_newest_capture(tmp_path: Path) -> None:
    raw_root = tmp_path / "officials_pfr_wayback"
    older = _manifest_game(
        "2014_01_GB_SEA",
        season=2014,
        week=1,
        gameday="2014-09-04",
        pfr_id="201409040sea",
        capture_ts="20141006175522",
        html_file="html/201409040sea__20141006175522.html",
    )
    newer = _manifest_game(
        "2014_01_GB_SEA",
        season=2014,
        week=1,
        gameday="2014-09-04",
        pfr_id="201409040sea",
        capture_ts="20151201000000",
        html_file="html/201409040sea__20151201000000.html",
    )
    old_crew = (("Referee", "Stale Name"), *FULL_CREW[1:])
    _write_run(raw_root, "20260907T140309Z", [(older, old_crew)])
    _write_run(raw_root, "20260908T090000Z", [(newer, FULL_CREW)])

    rows = load_archive_crew_rows(repo_root=REPO_ROOT, raw_root=raw_root)
    assert len(rows) == 14  # both runs' rows are read

    table = canonical_crew_table(repo_root=REPO_ROOT, raw_root=raw_root, schedules=_schedules())
    assert len(table) == 1
    row = table.iloc[0]
    assert row["source_run_id"] == "20260908T090000Z"
    assert row["wayback_capture_timestamp"] == "20151201000000"
    assert row["referee"] == "Ed Hochuli"
    assert int(row["n_crew_rows"]) == 7  # only the winning run's rows survive


def test_a_crew_row_with_no_schedule_game_fails_closed(tmp_path: Path) -> None:
    raw_root = _one_game_archive(tmp_path, game_id="2014_01_NOT_A_GAME")
    with pytest.raises(OfficialsArchiveError, match="do not join to any schedule game"):
        canonical_crew_table(repo_root=REPO_ROOT, raw_root=raw_root, schedules=_schedules())


def test_a_crew_row_matching_two_schedule_games_fails_closed(tmp_path: Path) -> None:
    raw_root = _one_game_archive(tmp_path)
    duplicated = pd.concat([_schedules(), _schedules()], ignore_index=True)
    with pytest.raises(OfficialsArchiveError, match="more than one schedule game"):
        canonical_crew_table(repo_root=REPO_ROOT, raw_root=raw_root, schedules=duplicated)


def test_a_schedule_missing_the_crosswalk_column_fails_closed(tmp_path: Path) -> None:
    raw_root = _one_game_archive(tmp_path)
    schedules = _schedules().drop(columns=["old_game_id"])
    with pytest.raises(OfficialsArchiveError, match="missing columns"):
        canonical_crew_table(repo_root=REPO_ROOT, raw_root=raw_root, schedules=schedules)


def test_an_empty_archive_yields_an_empty_canonical_table(tmp_path: Path) -> None:
    empty = tmp_path / "officials_pfr_wayback"
    empty.mkdir()
    table = canonical_crew_table(repo_root=REPO_ROOT, raw_root=empty, schedules=_schedules())
    assert table.empty
    assert list(table.columns) == list(CANONICAL_CREW_COLUMNS)


# ---------------------------------------------------------------------------
# LEAKAGE regression tests (AGENTS.md: one per new feature family)
# ---------------------------------------------------------------------------


def test_a_capture_at_or_before_kickoff_fails_closed(tmp_path: Path) -> None:
    """LEAKAGE. The archive's whole timing claim is that its captures are
    strictly POST-game, so it can only ever support historical crew
    identity. The sweep cannot produce a pre-game capture (CDX
    ``from=<gameday + 1>``), and a pre-game PFR boxscore is a placeholder
    with no officials block -- so a row claiming one is fabricated crew
    data, not early knowledge, and must fail closed rather than be trusted
    as a pregame capture."""

    raw_root = _one_game_archive(tmp_path, capture_ts="20140903120000")  # game is 2014-09-04
    with pytest.raises(OfficialsArchiveError, match="after the game's own day"):
        canonical_crew_table(repo_root=REPO_ROOT, raw_root=raw_root, schedules=_schedules())

    # Same calendar day is refused too: nothing in the timestamp separates a
    # 10am pre-game placeholder from an evening capture, and the sweep's own
    # CDX bound is gameday + 1 day.
    same_day = _one_game_archive(tmp_path / "sameday", capture_ts="20140904235959")
    with pytest.raises(OfficialsArchiveError, match=ARCHIVE_TIMING_CLASS):
        canonical_crew_table(repo_root=REPO_ROOT, raw_root=same_day, schedules=_schedules())

    # The day after is the first admissible capture.
    ok = _one_game_archive(tmp_path / "nextday", capture_ts="20140905000000")
    table = canonical_crew_table(repo_root=REPO_ROOT, raw_root=ok, schedules=_schedules())
    assert len(table) == 1


def test_a_missing_or_malformed_capture_timestamp_fails_closed() -> None:
    rows = pd.DataFrame(
        [
            {
                "game_id": "2014_01_GB_SEA",
                "gameday": "2014-09-04",
                "wayback_capture_timestamp": None,
            }
        ]
    )
    with pytest.raises(OfficialsArchiveError):
        assert_captures_are_post_game(rows)

    rows.loc[0, "wayback_capture_timestamp"] = "not-a-timestamp"
    with pytest.raises(OfficialsArchiveError):
        assert_captures_are_post_game(rows)


def test_archive_rows_are_refused_by_the_prospective_channel() -> None:
    """LEAKAGE. The refresh path requires a snapshot captured strictly
    before each game's own ``min(kickoff, Sunday 16:00 ET)`` deadline. Every
    Wayback capture is after kickoff, so archive rows can never satisfy it
    and are refused rather than silently accepted."""

    merged = load_officials(feed=_feed(), include_archive=False)
    refuse_archive_rows(merged, channel="crew_tilt_refresh_v1")  # no source column: the raw feed

    with_archive = pd.DataFrame({"official_name": ["Ed Hochuli"], "source": [ARCHIVE_SOURCE]})
    with pytest.raises(OfficialsArchiveError, match=ARCHIVE_TIMING_CLASS):
        refuse_archive_rows(with_archive, channel="crew_tilt_refresh_v1")

    nflverse_only = pd.DataFrame({"official_name": ["Ed Hochuli"], "source": [NFLVERSE_SOURCE]})
    refuse_archive_rows(nflverse_only, channel="crew_tilt_refresh_v1")


def test_the_prospective_loader_never_includes_the_archive(tmp_path: Path) -> None:
    feed = _feed()
    frame = load_officials_for_prospective_channel(tmp_path, feed=feed)
    pd.testing.assert_frame_equal(frame, feed)
    assert "source" not in frame.columns


# ---------------------------------------------------------------------------
# The merged table: nflverse wins, the archive fills earlier seasons
# ---------------------------------------------------------------------------


def test_the_shipped_default_returns_the_feed_bit_for_bit(tmp_path: Path) -> None:
    """Routing a consumer through ``load_officials`` must change nothing."""

    assert INCLUDE_ARCHIVE_DEFAULT is False
    feed = _feed()
    path = tmp_path / "officials.parquet"
    feed.to_parquet(path)
    on_disk = pd.read_parquet(path)

    pd.testing.assert_frame_equal(load_officials(tmp_path, officials_path=path), on_disk)
    assert list(load_officials(tmp_path, officials_path=path).columns) == list(
        NFLVERSE_OFFICIALS_COLUMNS
    )


def test_every_2015_2025_row_survives_the_merge_bit_for_bit(tmp_path: Path) -> None:
    """The pin the whole extension rests on: turning the archive ON must not
    change a single value the nflverse feed already carries. The one declared
    schema difference is ``jersey_number`` widening ``int32`` -> nullable
    ``Int32`` (a PFR boxscore carries no jersey numbers); every 2015-2025
    jersey number stays present and identical."""

    feed = _feed(seasons=(2015, 2016, 2017))
    raw_root = _one_game_archive(tmp_path)
    merged = load_officials(
        REPO_ROOT,
        feed=feed,
        include_archive=True,
        raw_root=raw_root,
        schedules=_schedules(),
    )

    nflverse_slice = merged.loc[
        merged["source"] == NFLVERSE_SOURCE, list(NFLVERSE_OFFICIALS_COLUMNS)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(nflverse_slice, feed.astype({"jersey_number": "Int32"}))

    # Same statement addressed by SEASON rather than by provenance label.
    modern = merged.loc[merged["season"].between(2015, 2025), list(NFLVERSE_OFFICIALS_COLUMNS)]
    pd.testing.assert_frame_equal(
        modern.reset_index(drop=True), feed.astype({"jersey_number": "Int32"})
    )
    assert modern["jersey_number"].notna().all()
    assert list(modern["jersey_number"].astype("int32")) == list(feed["jersey_number"])

    # ... and the archive really did add its season.
    assert set(merged.loc[merged["source"] == ARCHIVE_SOURCE, "season"]) == {2014}
    assert len(merged) == len(feed) + 7


def test_nflverse_wins_on_overlap(tmp_path: Path) -> None:
    """An archive game whose legacy id is already in the feed is dropped
    WHOLE -- never merged position-by-position."""

    feed = _feed(seasons=(2014,))
    assert set(feed["game_id"]) == {"2014091000"}
    raw_root = _one_game_archive(tmp_path)
    schedules = _schedules(
        [
            {
                "game_id": "2014_01_GB_SEA",
                "old_game_id": "2014091000",  # collides with the feed
                "season": 2014,
                "week": 1,
                "gameday": "2014-09-04",
            }
        ]
    )

    merged = load_officials(
        REPO_ROOT, feed=feed, include_archive=True, raw_root=raw_root, schedules=schedules
    )
    assert set(merged["source"]) == {NFLVERSE_SOURCE}
    assert len(merged) == len(feed)


def test_the_archive_uses_the_legacy_game_id_the_feed_joins_on(tmp_path: Path) -> None:
    raw_root = _one_game_archive(tmp_path)
    long = archive_officials_long(repo_root=REPO_ROOT, raw_root=raw_root, schedules=_schedules())

    assert list(long.columns) == [*NFLVERSE_OFFICIALS_COLUMNS, "source"]
    assert set(long["game_id"]) == {"2014090400"}
    assert set(long["season_type"]) == {"REG"}
    assert set(long["source"]) == {ARCHIVE_SOURCE}
    assert long["jersey_number"].isna().all()
    assert long["official_id"].isna().all()
    assert long["game_key"].isna().all()
    assert list(long["position"]) == list(CORE_CREW_POSITIONS)


def test_merging_rejects_a_feed_missing_its_own_columns(tmp_path: Path) -> None:
    """The merge needs the feed's full schema. The default path does NOT
    validate -- it is a pure pass-through, and several consumer tests hand it
    a minimal officials fixture carrying only the columns they use."""

    broken = _feed().drop(columns=["jersey_number"])
    empty = tmp_path / "officials_pfr_wayback"
    empty.mkdir()
    with pytest.raises(OfficialsArchiveError, match="missing columns"):
        load_officials(tmp_path, feed=broken, include_archive=True, raw_root=empty)

    pd.testing.assert_frame_equal(load_officials(tmp_path, feed=broken), broken)


def test_a_feed_with_no_archive_available_is_returned_with_provenance(tmp_path: Path) -> None:
    empty = tmp_path / "officials_pfr_wayback"
    empty.mkdir()
    feed = _feed()
    merged = load_officials(REPO_ROOT, feed=feed, include_archive=True, raw_root=empty)
    assert set(merged["source"]) == {NFLVERSE_SOURCE}
    assert len(merged) == len(feed)


# ---------------------------------------------------------------------------
# Coverage description and caching
# ---------------------------------------------------------------------------


def test_describe_archive_coverage_reports_per_season(tmp_path: Path) -> None:
    raw_root = _one_game_archive(tmp_path)
    summary = describe_archive_coverage(
        table=canonical_crew_table(repo_root=REPO_ROOT, raw_root=raw_root, schedules=_schedules())
    )
    assert summary["n_games"] == 1
    assert summary["n_complete_crews"] == 1
    assert summary["timing_class"] == ARCHIVE_TIMING_CLASS
    assert summary["seasons"]["2014"]["n_distinct_referees"] == 1


def test_the_row_cache_invalidates_when_a_sweep_writes(tmp_path: Path) -> None:
    """A sweep is often still running; the memo must not pin a stale read."""

    raw_root = _one_game_archive(tmp_path)
    first = load_archive_crew_rows(repo_root=REPO_ROOT, raw_root=raw_root)
    assert len(first) == 7

    run_dir = raw_root / "20260907T140309Z"
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    second_game = _manifest_game(
        "2014_01_NO_ATL",
        season=2014,
        week=1,
        gameday="2014-09-07",
        pfr_id="201409070atl",
        capture_ts="20141008120000",
        html_file="html/201409070atl__20141008120000.html",
    )
    (run_dir / str(second_game["html_file"])).write_text(_crew_html(FULL_CREW), encoding="utf-8")
    payload["games"].append(second_game)
    (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    second = load_archive_crew_rows(repo_root=REPO_ROOT, raw_root=raw_root)
    assert len(second) == 14
