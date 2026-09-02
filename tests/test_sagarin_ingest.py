"""Tests for scripts/ingest_sagarin_ratings.py, focused on the WP19 fix
(docs/sagarin_backfill.md section 9): a transitional home-advantage line
format, measured across roughly Nov 2011 - Sep 2013 (both the sagarin.com
and usatoday.com domains), that the original 4-bracket/1-bracket parser pair
never matched -- leaving every 2012 capture (and most of 2013) with a null
`home_edge_rating`, so 2012 contributed 0/256 games to the Sagarin-divergence
join even though its team rows parsed fine.

Fixtures under tests/fixtures/sagarin/*.html are REAL, trimmed excerpts (the
header line + the home-advantage/home-edge line + 3 team rows) taken
verbatim from genuine Wayback captures cached this session
(data/raw/sagarin/20260820T112501Z/, gitignored/local-only) -- the long
boilerplate explanatory paragraph between the header and the ratings table
is dropped since it carries no information the parser reads, but every byte
that IS kept (including the surrounding <font color=...> tags) is copied
unmodified from the real page.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ingest_sagarin_ratings import (
    ERA_SAGARIN_COM,
    ERA_USATODAY,
    enumerate_cached_captures,
    parse_capture_html,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sagarin"


def _load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


# ---------------------------------------------------------------------------
# Pre-existing formats: must still parse exactly as before (regression guard)
# ---------------------------------------------------------------------------


def test_era_a_4bracket_still_parses_all_four_methods() -> None:
    parsed = parse_capture_html(_load("era_a_4bracket_snippet.html"))

    assert parsed.season == 2014
    assert parsed.era_format == ERA_SAGARIN_COM
    assert parsed.home_edge_rating == 2.74
    assert parsed.home_edge_methods == [2.73, 2.74, 2.74]
    # Fixtures are trimmed to 3 of 32 real team rows, so parse_error is the
    # expected "too few rows" flag -- not one of the genuine failure modes
    # (no_header_match / no_team_rows) that would indicate a real bug.
    assert parsed.parse_error == "only_3_teams"

    teams = {row["team_name_raw"]: row for row in parsed.team_rows}
    assert set(teams) == {"Seattle Seahawks", "Denver Broncos", "San Francisco 49ers"}
    seahawks = teams["Seattle Seahawks"]
    assert seahawks["team_code"] == "SEA"
    assert seahawks["rating"] == 28.33
    assert seahawks["golden_mean_value"] == 27.54
    assert seahawks["pure_points_value"] == 28.18
    assert seahawks["elo_score_value"] == 27.39


def test_era_b_1bracket_still_parses_single_home_edge() -> None:
    parsed = parse_capture_html(_load("era_b_1bracket_snippet.html"))

    assert parsed.season == 2010
    assert parsed.era_format == ERA_USATODAY
    assert parsed.home_edge_rating == 2.52
    assert parsed.home_edge_methods == []
    # Fixtures are trimmed to 3 of 32 real team rows, so parse_error is the
    # expected "too few rows" flag -- not one of the genuine failure modes
    # (no_header_match / no_team_rows) that would indicate a real bug.
    assert parsed.parse_error == "only_3_teams"

    teams = {row["team_name_raw"]: row for row in parsed.team_rows}
    steelers = teams["Pittsburgh Steelers"]
    assert steelers["team_code"] == "PIT"
    assert steelers["elo_chess_value"] == 28.40
    assert steelers["pure_points_value"] == 28.88


# ---------------------------------------------------------------------------
# WP19 fix: the transitional 3-bracket and comma-separated formats
# ---------------------------------------------------------------------------


def test_transitional_3bracket_home_advantage_now_parses(_load=_load) -> None:
    """The exact shape that made every 2012 capture null before the fix:
    'HOME ADVANTAGE=[  2.02]  ...  [  0.91]  ...  [  1.33]' (three brackets,
    not four) -- measured on a genuine 2012-week-8 sagarin.com capture."""

    parsed = parse_capture_html(_load("era_transitional_3bracket_snippet.html"))

    assert parsed.season == 2012
    assert parsed.header_week_number == 8
    assert parsed.era_format == ERA_USATODAY  # 2-method column shape, not 3
    assert parsed.home_edge_rating == 2.02
    # Per-method values are deliberately NOT written into the
    # golden_mean/pure_points/elo_score slots (those assume a fixed
    # GOLDEN_MEAN/PURE_POINTS/ELO_SCORE order); this era's real methods are
    # ELO_CHESS/PURE POINTS, so home_edge_methods stays empty rather than
    # mislabeling data -- only home_edge_rating (what the join needs) is
    # recovered.
    assert parsed.home_edge_methods == []
    # Fixtures are trimmed to 3 of 32 real team rows, so parse_error is the
    # expected "too few rows" flag -- not one of the genuine failure modes
    # (no_header_match / no_team_rows) that would indicate a real bug.
    assert parsed.parse_error == "only_3_teams"

    teams = {row["team_name_raw"]: row for row in parsed.team_rows}
    niners = teams["San Francisco 49ers"]
    assert niners["team_code"] == "SF"
    assert niners["rating"] == 30.74
    assert niners["elo_chess_value"] == 29.11
    assert niners["pure_points_value"] == 31.24


def test_transitional_3bracket_preseason_starting_ratings_variant() -> None:
    """Same 3-bracket home-advantage shape, but on a pre-Week-1 'Starting
    Ratings' snapshot (every team 0-0-0) -- a second, header-distinct
    real-world instance of the same format."""

    parsed = parse_capture_html(_load("era_preseason_3bracket_snippet.html"))

    assert parsed.season == 2013
    assert parsed.era_format == ERA_USATODAY
    assert parsed.home_edge_rating == 2.53
    assert parsed.home_edge_methods == []
    # Fixtures are trimmed to 3 of 32 real team rows, so parse_error is the
    # expected "too few rows" flag -- not one of the genuine failure modes
    # (no_header_match / no_team_rows) that would indicate a real bug.
    assert parsed.parse_error == "only_3_teams"
    assert len(parsed.team_rows) == 3


def test_transitional_comma_home_edge_now_parses() -> None:
    """A third, one-off layout measured in the same window: 'HOME EDGE=
    3.04,  2.38,  2.74' -- comma-separated, unbracketed, and a different
    label ('HOME EDGE=' not 'HOME ADVANTAGE=') -- measured on a genuine
    usatoday.com 2012-01-09 (season 2011, Wild Card Weekend) capture."""

    parsed = parse_capture_html(_load("era_transitional_comma_snippet.html"))

    assert parsed.season == 2011
    assert parsed.era_format == ERA_USATODAY
    assert parsed.home_edge_rating == 3.04
    assert parsed.home_edge_methods == []
    # Fixtures are trimmed to 3 of 32 real team rows, so parse_error is the
    # expected "too few rows" flag -- not one of the genuine failure modes
    # (no_header_match / no_team_rows) that would indicate a real bug.
    assert parsed.parse_error == "only_3_teams"

    teams = {row["team_name_raw"]: row for row in parsed.team_rows}
    packers = teams["Green Bay Packers"]
    assert packers["team_code"] == "GB"
    assert packers["elo_chess_value"] == 34.43
    assert packers["pure_points_value"] == 29.34


def test_transitional_formats_never_leave_home_edge_rating_null() -> None:
    for name in (
        "era_transitional_3bracket_snippet.html",
        "era_transitional_comma_snippet.html",
        "era_preseason_3bracket_snippet.html",
    ):
        parsed = parse_capture_html(_load(name))
        assert parsed.home_edge_rating is not None, name


# ---------------------------------------------------------------------------
# enumerate_cached_captures: reparse-only mode used to rebuild the alignment
# view from disk without any new network calls (WP19 addition).
# ---------------------------------------------------------------------------


def test_enumerate_cached_captures_walks_pages_dir(tmp_path: Path) -> None:
    pages_dir = tmp_path / "pages"
    (pages_dir / "sagarin_com" / "nflsend").mkdir(parents=True)
    (pages_dir / "usatoday" / "nfl11").mkdir(parents=True)
    (pages_dir / "sagarin_com" / "nflsend" / "20121031035824.html").write_bytes(b"<html></html>")
    (pages_dir / "usatoday" / "nfl11" / "20120109071948.html").write_bytes(b"<html></html>")

    rows = enumerate_cached_captures(pages_dir)

    by_key = {(r["era"], r["url_key"], r["timestamp"]) for r in rows}
    assert by_key == {
        ("sagarin_com", "nflsend", "20121031035824"),
        ("usatoday", "nfl11", "20120109071948"),
    }
    sagarin_row = next(r for r in rows if r["era"] == "sagarin_com")
    assert sagarin_row["original"] == "sagarin.com/sports/nflsend.htm"
    usatoday_row = next(r for r in rows if r["era"] == "usatoday")
    assert usatoday_row["original"] == "www.usatoday.com/sports/sagarin/nfl11.htm"
    assert all(r["digest"] is None for r in rows)


def test_enumerate_cached_captures_empty_dir_returns_empty(tmp_path: Path) -> None:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    assert enumerate_cached_captures(pages_dir) == []
