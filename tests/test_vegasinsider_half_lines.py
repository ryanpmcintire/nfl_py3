"""Half-line archive point-in-time regressions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.backfill_vegasinsider as biv


def test_offline_rebuild_refuses_network_and_preserves_evidence(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    (cache / "line_movement").mkdir(parents=True)
    manifest = cache / "manifest_2006.json"
    manifest.write_text(
        json.dumps({"snapshots": [{"capture_timestamp": "20061005120000", "file": "cached.html"}]})
    )
    html = _lm_page(
        book_sections=_book_section(
            "A",
            "CAESARS",
            _data_row(
                "10/08", "12:00PM", "", "", "", "", "", "", "IND-3", "TEN+3", "IND-1", "TEN+1"
            ),
        )
    )
    page = cache / "line_movement" / "20061005120000_abcd1234.html"
    page.write_text(html)

    def no_network(*args, **kwargs):
        raise AssertionError("offline rebuild attempted network")

    monkeypatch.setattr(biv, "fetch_via_curl", no_network)
    monkeypatch.setattr(biv, "stamp_sidecar", lambda *args, **kwargs: None)
    out = tmp_path / "new"
    summary = biv.rebuild_half_lines_from_cache(cache, out, [2006])
    frame = pd.read_parquet(out / "half_lines_2006.parquet")
    assert frame.spread_line.isna().all()
    assert summary["seasons"][0]["dropped_movement_rows"] == {"movement_after_observation": 1}
    assert page.read_text() == html
    with pytest.raises(FileExistsError):
        biv.rebuild_half_lines_from_cache(cache, out, [2006])


def test_screen_excludes_in_play_and_unknown_observations(tmp_path):
    from scripts.lead02_half_line_script_screen import load_half_lines

    pd.DataFrame({"spread_line": [-1.0, -2.0, -3.0], "in_play": [False, True, None]}).to_parquet(
        tmp_path / "half_lines_2009.parquet"
    )
    assert load_half_lines(tmp_path).spread_line.tolist() == [-1.0]


def _lm_page(
    *,
    away: str = "Tennessee Titans",
    home: str = "Indianapolis Colts",
    game_date: str = "Sunday, October 08, 2006",
    game_time: str = "1:00PM ET",
    book_sections: str = "",
) -> str:
    return f"""<html><body>
<table><tbody><tr>
<td align=center class=page_title>
<font size=4>{away} @ {home}</font>
</td>
</tr></tbody></table>
<table>
<tr><TD vAlign=top><B>&nbsp;Game Date:</B>&nbsp;&nbsp;&nbsp;{game_date}</TD></tr>
<tr><TD vAlign=top><B>&nbsp;Game Time:</B>&nbsp;&nbsp;&nbsp;{game_time}</TD></tr>
</table>
{book_sections}
</body></html>"""


def _book_section(anchor: str, book: str, rows_html: str) -> str:
    return f"""
<a name="{anchor}"></a>
{book} LINE MOVEMENTS
<table><tbody>
<TR class=bg0_sub vAlign=center align=right height=17>
<TD align=left colspan=2 NOWRAP width="15%"></TD>
<TD align=center NOWRAP colspan=2 width="17%">Money Line</TD>
<TD align=center NOWRAP colspan=2 width="17%">Spread</TD>
<TD align=center NOWRAP colspan=2 width="17%">Total</TD>
<TD align=center NOWRAP colspan=2 width="17%">1st Half</TD>
<TD align=center NOWRAP colspan=2 width="17%">2nd Half</TD>
</TR>
<TR class=bg0_sub vAlign=center align=right height=17>
<TD align=center>Date</TD><TD align=middle NOWRAP>Time</TD>
<TD align=middle NOWRAP>Fav</TD><TD align=center NOWRAP>Dog</TD>
<TD align=center NOWRAP>Fav</TD><TD align=center NOWRAP>Dog</TD>
<TD align=middle NOWRAP>Over</TD><TD align=center NOWRAP>Under</TD>
<TD align=middle NOWRAP>Fav</TD><TD align=center NOWRAP>Dog</TD>
<TD align=middle NOWRAP>Fav</TD><TD align=center NOWRAP>Dog</TD>
</TR>
{rows_html}
</tbody></table>
"""


def _data_row(
    date: str,
    time: str,
    ml_fav: str,
    ml_dog: str,
    spread_fav: str,
    spread_dog: str,
    total_over: str,
    total_under: str,
    h1_fav: str,
    h1_dog: str,
    h2_fav: str,
    h2_dog: str,
) -> str:
    cells = [
        date,
        time,
        ml_fav,
        ml_dog,
        spread_fav,
        spread_dog,
        total_over,
        total_under,
        h1_fav,
        h1_dog,
        h2_fav,
        h2_dog,
    ]
    tds = "".join(f'<TD class="bg2" nowrap>{c}</TD>' for c in cells)
    return f"<TR>{tds}</TR>"


# ---------------------------------------------------------------------------
# parse_half_cell_value: token-level parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token,expected",
    [
        ("IND-11.5", -11.5),
        ("TEN+11.5", 11.5),
        ("GNB -11", -11.0),
        ("PHI +.5", 0.5),
        ("TEN-.5", -0.5),
        ("CIN PK", 0.0),
        ("INDXX", None),
        ("IND XX", None),
        ("", None),
        ("CIN", None),  # malformed/truncated cell: no usable number
    ],
)
def test_parse_half_cell_value(token: str, expected: float | None) -> None:
    assert biv.parse_half_cell_value(token) == expected


def test_normalize_full_team_name_handles_the_giants_title_artifact() -> None:
    # VegasInsider's page title literally renders "N.Y. Giants Giants" (a
    # city-alias + mascot concatenation bug on their end); measured across
    # all 165 cached line_movement files, so it must resolve like the clean
    # "New York Giants" string does.
    assert biv.normalize_full_team_name("New York Giants") == "NYG"
    assert biv.normalize_full_team_name("N.Y. Giants Giants") == "NYG"
    assert biv.normalize_full_team_name("St. Louis Rams") == "LAR"
    assert biv.normalize_full_team_name("Los Angeles Rams") == "LAR"
    assert biv.normalize_full_team_name("Oakland Raiders") == "LV"
    assert biv.normalize_full_team_name("Not A Real Team") is None


# ---------------------------------------------------------------------------
# Fixture 1: full game + 1H + 2H columns, two rows (last row wins per half)
# ---------------------------------------------------------------------------


def test_parse_page_with_full_and_half_columns() -> None:
    rows = _data_row(
        "10/05",
        "4:37pm",
        "IND XX",
        "TEN XX",
        "IND-18.0",
        "TEN+18.0",
        "48",
        "48",
        "IND-11.5",
        "TEN+11.5",
        "",
        "",
    ) + _data_row(
        "10/08",
        "2:32pm",
        "IND XX",
        "TEN XX",
        "IND-17.5",
        "TEN+17.5",
        "48",
        "48",
        "IND-12.5",
        "TEN+12.5",
        "IND-13.5",
        "TEN+13.5",
    )
    html = _lm_page(book_sections=_book_section("E", "CAESARS", rows))

    parsed = biv.parse_line_movement_page(html)

    assert parsed is not None
    meta, books = parsed
    assert meta.away == "TEN"
    assert meta.home == "IND"
    assert meta.game_date_iso == "2006-10-08"
    assert meta.kickoff_time == "1:00 PM"
    assert books == [("CAESARS", -12.5, -13.5)]


def test_last_row_wins_when_half_becomes_unavailable_again() -> None:
    # 1H posted early, then withdrawn ("XX") on the final row: the LAST
    # *usable* reading should still win, not a stale earlier one silently
    # dropped, and not a None overwrite of a real number.
    rows = _data_row(
        "10/05",
        "9:00am",
        "",
        "",
        "IND-17.0",
        "TEN+17.0",
        "47",
        "47",
        "IND-11.0",
        "TEN+11.0",
        "",
        "",
    ) + _data_row(
        "10/08", "9:00am", "", "", "IND-17.5", "TEN+17.5", "47.5", "47.5", "INDXX", "TENXX", "", ""
    )
    html = _lm_page(book_sections=_book_section("E", "CAESARS", rows))

    _, books = biv.parse_line_movement_page(html)  # type: ignore[misc]

    assert books == [("CAESARS", -11.0, None)]


# ---------------------------------------------------------------------------
# Fixture 2: a page with no half section at all
# ---------------------------------------------------------------------------


def test_book_section_without_half_columns_yields_none_for_both_halves() -> None:
    # Full game columns only (8 cells, no 1H/2H) -- a genuinely narrower
    # table than the standard 12-column layout.
    narrow_row = (
        '<TR><TD class="bg2">10/08</TD><TD class="bg2">9:00am</TD>'
        '<TD class="bg2">IND XX</TD><TD class="bg2">TEN XX</TD>'
        '<TD class="bg2">IND-17.5</TD><TD class="bg2">TEN+17.5</TD>'
        '<TD class="bg2">48</TD><TD class="bg2">48</TD></TR>'
    )
    half1, half2 = biv.extract_book_half_lines(narrow_row)
    assert half1 is None
    assert half2 is None


def test_page_missing_page_title_is_not_a_line_movement_page() -> None:
    # Measured: 5/165 cached "line_movement" files are actually a mis-fetched
    # VegasInsider homepage (wrong wayback redirect target), not a real
    # movement page. They must be recognised and skipped, not mis-parsed.
    homepage_html = (
        "<html><body><div class='viHeaderNorm'>TODAY'S TOP BETTING TRENDS</div></body></html>"
    )
    assert biv.parse_line_movement_page(homepage_html) is None


# ---------------------------------------------------------------------------
# build_half_lines: end-to-end over cached files, season-scoping, and the
# point-in-time / leakage discipline.
# ---------------------------------------------------------------------------


def _write_lm_file(lm_dir: Path, capture_ts: str, html: str, suffix: str = "aaaa1111") -> Path:
    lm_dir.mkdir(parents=True, exist_ok=True)
    path = lm_dir / f"{capture_ts}_{suffix}.html"
    path.write_text(html, encoding="utf-8")
    return path


def test_build_half_lines_end_to_end(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "20260101T000000Z"
    lm_dir = snapshot_dir / "line_movement"
    rows = _data_row(
        "10/08",
        "2:32pm",
        "IND XX",
        "TEN XX",
        "IND-17.5",
        "TEN+17.5",
        "48",
        "48",
        "IND-12.5",
        "TEN+12.5",
        "IND-13.5",
        "TEN+13.5",
    )
    html = _lm_page(book_sections=_book_section("E", "CAESARS", rows))
    capture_ts = "20061008190000"
    _write_lm_file(lm_dir, capture_ts, html)

    frame = biv.build_half_lines(snapshot_dir, {capture_ts})

    assert frame["in_play"].all()
    assert list(frame.columns) == biv.HALF_LINES_COLUMNS
    assert len(frame) == 2  # one row per half for the one book
    assert set(frame["half"]) == {1, 2}
    assert frame["total_line"].isna().all()  # measured: no half total exists in this source
    assert frame["spread_price"].isna().all()  # measured: no half price exists in this source
    half1_row = frame.loc[frame["half"] == 1].iloc[0]
    half2_row = frame.loc[frame["half"] == 2].iloc[0]
    assert half1_row["spread_line"] == pytest.approx(-12.5)
    assert half2_row["spread_line"] == pytest.approx(-13.5)
    assert half1_row["book"] == "CAESARS"
    assert half1_row["away"] == "TEN"
    assert half1_row["home"] == "IND"
    assert half1_row["game_date"] == "2006-10-08"


def test_build_half_lines_filters_by_capture_ts_values(tmp_path: Path) -> None:
    # Two seasons' worth of files can share one line_movement/ cache dir;
    # build_half_lines for one season must not pull in the other season's
    # captures.
    snapshot_dir = tmp_path / "run"
    lm_dir = snapshot_dir / "line_movement"
    rows = _data_row(
        "10/08",
        "1:00pm",
        "",
        "",
        "IND-17.5",
        "TEN+17.5",
        "48",
        "48",
        "IND-12.5",
        "TEN+12.5",
        "",
        "",
    )
    html = _lm_page(book_sections=_book_section("E", "CAESARS", rows))
    _write_lm_file(lm_dir, "20061005080503", html, suffix="aaaa1111")
    _write_lm_file(lm_dir, "20071005080503", html, suffix="bbbb2222")

    frame_2006 = biv.build_half_lines(snapshot_dir, {"20061005080503"})

    assert set(frame_2006["capture_ts"]) == {"20061005080503"}
    assert len(frame_2006) == 2  # half 1 + half 2 rows for the single in-scope file


def test_future_movement_is_rejected_not_backdated(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "run"
    lm_dir = snapshot_dir / "line_movement"
    rows = (
        _data_row("09/24", "9:00am", "", "", "IND-14.0", "TEN+14.0", "45", "45", "", "", "", "")
        + _data_row(
            "10/01",
            "9:00am",
            "",
            "",
            "IND-16.0",
            "TEN+16.0",
            "46",
            "46",
            "IND-10.0",
            "TEN+10.0",
            "",
            "",
        )
        + _data_row(
            "10/08",
            "9:00am",
            "",
            "",
            "IND-17.5",
            "TEN+17.5",
            "48",
            "48",
            "IND-12.5",
            "TEN+12.5",
            "IND-13.5",
            "TEN+13.5",
        )
    )
    html = _lm_page(book_sections=_book_section("E", "CAESARS", rows))
    capture_ts = "20061005080503"  # earlier than the last row's own 10/08 date on purpose
    _write_lm_file(lm_dir, capture_ts, html)

    frame = biv.build_half_lines(snapshot_dir, {capture_ts})

    assert frame.loc[frame["half"].eq(1), "spread_line"].iloc[0] == -10.0
    assert frame.loc[frame["half"].eq(2), "spread_line"].isna().all()
    assert frame.attrs["dropped_movement_rows"] == {"movement_after_observation": 1}
    assert not frame["in_play"].any()
    assert set(frame["capture_ts"].unique()) == {capture_ts}
    for value in frame["capture_ts"]:
        assert value == capture_ts
        assert "09/24" not in value
        assert "10/01" not in value
        assert "10/08" not in value


def test_unparsed_line_movement_files_are_counted_not_silently_dropped(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "run"
    lm_dir = snapshot_dir / "line_movement"
    homepage_html = "<html><body>not a movement page</body></html>"
    capture_ts = "20111213131021"
    _write_lm_file(lm_dir, capture_ts, homepage_html)

    frame = biv.build_half_lines(snapshot_dir, {capture_ts})

    assert frame.empty
    assert frame.attrs["unparsed_line_movement_files"] == 1


# ---------------------------------------------------------------------------
# classify_missing_half_nav_boards
# ---------------------------------------------------------------------------


def _snapshot_record(capture_ts: str, file: str) -> biv.SnapshotRecord:
    return biv.SnapshotRecord(
        capture_ts=capture_ts,
        original_url="http://example.test",
        wayback_url="http://example.test",
        file=file,
        sha256=None,
        size_bytes=0,
        cdx_digest="digest",
        cdx_length=0,
        http_status="cached",
        error=None,
    )


def test_classify_missing_half_nav_boards(tmp_path: Path) -> None:
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir(parents=True)
    (snapshots_dir / "20051001023412.html").write_text(
        "<html>legacy board, no half nav link, plain table markup only</html>", encoding="utf-8"
    )
    (snapshots_dir / "20061005080503.html").write_text(
        "<html>modern board with an oddsText grid but no half nav link</html>",
        encoding="utf-8",
    )
    (snapshots_dir / "20071009061103.html").write_text(
        "<html>modern board <a href='/nfl/odds/las-vegas/first-half/'>1st Half</a> oddsText</html>",
        encoding="utf-8",
    )
    records = [
        _snapshot_record("20051001023412", "snapshots/20051001023412.html"),
        _snapshot_record("20061005080503", "snapshots/20061005080503.html"),
        _snapshot_record("20071009061103", "snapshots/20071009061103.html"),
    ]

    result = biv.classify_missing_half_nav_boards(tmp_path, records)

    assert result["board_snapshots_checked"] == 3
    assert result["board_snapshots_missing_1st_half_nav_text"] == 2
    classifications = {
        row["capture_ts"]: row["classification"] for row in result["board_snapshots_missing_detail"]
    }
    assert classifications["20051001023412"] == "layout_variant_legacy_board"
    assert classifications["20061005080503"] == "genuinely_absent"


# ---------------------------------------------------------------------------
# compute_half_line_coverage: join rate against the full-game tidy rows
# ---------------------------------------------------------------------------


def test_compute_half_line_coverage_join_rate() -> None:
    half_lines = pd.DataFrame(
        [
            {
                "capture_ts": "20061005080503",
                "game_date": "2006-10-08",
                "away": "TEN",
                "home": "IND",
                "kickoff_time": "1:00 PM",
                "book": "CAESARS",
                "half": 1,
                "spread_line": -12.5,
                "total_line": None,
                "spread_price": None,
                "total_price": None,
            },
            {
                "capture_ts": "20061005080503",
                "game_date": "2006-10-08",
                "away": "TEN",
                "home": "IND",
                "kickoff_time": "1:00 PM",
                "book": "CAESARS",
                "half": 2,
                "spread_line": -13.5,
                "total_line": None,
                "spread_price": None,
                "total_price": None,
            },
            {
                "capture_ts": "20061005080503",
                "game_date": "2006-10-08",
                "away": "TEN",
                "home": "IND",
                "kickoff_time": "1:00 PM",
                "book": "UNMATCHEDBOOK",
                "half": 1,
                "spread_line": -12.0,
                "total_line": None,
                "spread_price": None,
                "total_price": None,
            },
        ]
    )
    tidy = pd.DataFrame(
        [
            {
                "capture_ts": "20061005080503",
                "game_date": "2006-10-08",
                "away": "TEN",
                "home": "IND",
                "kickoff_time": "1:00 PM",
                "book": "CAESARS",
                "spread_line": -18.5,
                "total_line": 48.0,
            }
        ]
    )

    stats = biv.compute_half_line_coverage(half_lines, tidy, {"board_snapshots_checked": 0})

    assert stats["rows"] == 3
    assert stats["rows_half1"] == 2
    assert stats["rows_half2"] == 1
    assert stats["rows_with_half1_spread"] == 2
    assert stats["rows_with_half2_spread"] == 1
    assert stats["rows_with_half1_total"] == 0
    assert stats["rows_with_half2_total"] == 0
    # 2 of the 3 (capture, matchup, book) keys in half_lines also exist in tidy
    assert stats["distinct_capture_matchup_book_keys_half_lines"] == 2
    assert stats["half_line_keys_present_in_full_game_tidy"] == 1
    assert stats["half_line_key_join_rate_against_full_game_tidy"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "actual,day,time,expected_in_play",
    [(None, "10/05", "9:00am", False), ("20061008190000", "10/08", "2:00pm", True)],
)
def test_observation_boundary_and_actual_capture(
    tmp_path: Path, actual: str | None, day: str, time: str, expected_in_play: bool
) -> None:
    rows = _data_row(day, time, "", "", "", "", "", "", "IND-10", "TEN+10", "IND-3", "TEN+3")
    html = _lm_page(book_sections=_book_section("E", "CAESARS", rows))
    requested = "20061005130000"
    if actual:
        html += f"<!-- archive_capture_ts={actual} -->"
    _write_lm_file(tmp_path / "line_movement", requested, html)
    frame = biv.build_half_lines(tmp_path, {requested})
    assert set(frame["capture_ts"]) == {actual or requested}
    assert frame["spread_line"].tolist() == [-10.0, -3.0]
    assert frame.attrs["dropped_movement_rows"] == {}
    assert frame["in_play"].eq(expected_in_play).all()


def test_fetch_preserves_redirected_archive_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    response = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=b"<html>cached</html>\n__CURL_HTTP_CODE__200\n__CURL_EFFECTIVE_URL__https://web.archive.org/web/20061008190000id_/http://www.vegasinsider.com/nfl/",
        stderr=b"",
    )
    monkeypatch.setattr(biv.subprocess, "run", lambda *args, **kwargs: response)
    monkeypatch.setattr(biv.RateLimiter, "wait", lambda self: None)
    body, status = biv.fetch_via_curl("https://example.test", biv.RateLimiter(delay_seconds=0))
    assert status == "200"
    assert b"archive_capture_ts=20061008190000" in body
