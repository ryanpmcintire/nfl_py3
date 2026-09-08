"""Tests for scripts/officials_wayback_sweep.py (LEAD-59).

No network anywhere in this file: every fetch is a fake ``FetchFn`` (a plain
callable stub) and every backoff sleep is a fake ``sleep_fn`` that records
its argument instead of blocking. Covers: the officials-block parser against
two constructed fixture pages (table strategy, inline-line fallback
strategy), the backoff/retry schedule, the resume-skip logic, the hard-stop
consecutive-failure counter, the "no capture found" non-failure path, the
manifest shape, season/game-type filtering of the schedule snapshot, and the
leakage-safety invariant (``effective_time`` == the game's own date, never a
later timestamp). 2026-09-07 (lane N) additions at the bottom: newest-capture
selection, the per-game fallback walk, ``--retry-unparsed``, one immutable
HTML file per fetch, and manifest-row upserts on re-attempts.
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

import scripts.officials_wayback_sweep as sweep  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parses_the_table_strategy_fixture() -> None:
    html = (FIXTURES / "pfr_boxscore_officials_table.html").read_text(encoding="utf-8")
    rows, warnings = sweep.parse_officials_block(html)

    assert warnings == []
    assert rows == [
        ("Referee", "Ed Hochuli"),
        ("Umpire", "Fred Bryan"),
        ("Head Linesman", "Kent Payne"),
        ("Line Judge", "Rusty Baynes"),
        ("Field Judge", "Doug Rosenbaum"),
        ("Side Judge", "Boris Cheek"),
        ("Back Judge", "Terrence Miles"),
    ]


def test_parses_the_inline_fallback_fixture_and_warns_it_used_the_fallback() -> None:
    html = (FIXTURES / "pfr_boxscore_officials_inline.html").read_text(encoding="utf-8")
    rows, warnings = sweep.parse_officials_block(html)

    assert any("inline" in w for w in warnings)
    assert rows == [
        ("Referee", "Walt Anderson"),
        ("Umpire", "Carl Paganelli"),
        ("Head Linesman", "Jerry Bergman"),
        ("Line Judge", "Julian Mapp"),
        ("Field Judge", "Tom Hill"),
        ("Side Judge", "Anthony Jeffries"),
        ("Back Judge", "Perry Paganelli"),
    ]


def test_neither_strategy_matches_returns_empty_with_a_warning() -> None:
    rows, warnings = sweep.parse_officials_block("<html><body>no officials here</body></html>")

    assert rows == []
    assert len(warnings) == 1
    assert "no officials block found" in warnings[0]


def test_table_strategy_skips_a_literal_header_row() -> None:
    html = (
        '<table id="officials"><tbody>'
        "<tr><th>Position</th><td>Official</td></tr>"
        "<tr><th>Referee</th><td>Jane Doe</td></tr>"
        "</tbody></table>"
    )
    rows, warnings = sweep.parse_officials_block(html)

    assert warnings == []
    assert rows == [("Referee", "Jane Doe")]


# ---------------------------------------------------------------------------
# Backoff / retry schedule (fetch_with_backoff)
# ---------------------------------------------------------------------------


class _ScriptedFetch:
    """Returns one canned FetchResult per call, in order."""

    def __init__(self, results: list[sweep.FetchResult]) -> None:
        self._results = list(results)
        self.calls: list[str] = []

    def __call__(self, url: str) -> sweep.FetchResult:
        self.calls.append(url)
        return self._results.pop(0)


def test_backoff_doubles_starting_at_the_initial_value_and_succeeds_on_retry() -> None:
    fetch = _ScriptedFetch(
        [
            sweep.FetchResult(None, 429, "http_429"),
            sweep.FetchResult(None, 503, "http_503"),
            sweep.FetchResult(b"<html>ok</html>", 200, None),
        ]
    )
    sleeps: list[float] = []
    limiter = sweep.RateLimiter(8.0, sleep_fn=lambda _s: None)

    outcome = sweep.fetch_with_backoff(
        "https://web.archive.org/x",
        fetch,
        limiter,
        initial_backoff_seconds=60.0,
        max_attempts=5,
        sleep_fn=sleeps.append,
    )

    assert outcome.status_code == 200
    assert outcome.content == b"<html>ok</html>"
    assert outcome.attempts == 3
    assert outcome.backoff_schedule_seconds == (60.0, 120.0)
    assert sleeps == [60.0, 120.0]
    assert len(fetch.calls) == 3


def test_backoff_gives_up_after_max_attempts_on_persistent_429() -> None:
    fetch = _ScriptedFetch([sweep.FetchResult(None, 429, "http_429")] * 3)
    limiter = sweep.RateLimiter(8.0, sleep_fn=lambda _s: None)
    sleeps: list[float] = []

    outcome = sweep.fetch_with_backoff(
        "https://web.archive.org/x",
        fetch,
        limiter,
        initial_backoff_seconds=60.0,
        max_attempts=3,
        sleep_fn=sleeps.append,
    )

    assert outcome.content is None
    assert outcome.status_code == 429
    assert outcome.attempts == 3
    assert outcome.gave_up_after_retries is True
    # Backoff happens BETWEEN attempts only: 2 sleeps for 3 attempts.
    assert sleeps == [60.0, 120.0]


def test_a_non_retryable_status_fails_immediately_without_backoff() -> None:
    fetch = _ScriptedFetch([sweep.FetchResult(None, 404, "http_404")])
    limiter = sweep.RateLimiter(8.0, sleep_fn=lambda _s: None)
    sleeps: list[float] = []

    outcome = sweep.fetch_with_backoff(
        "https://web.archive.org/x",
        fetch,
        limiter,
        initial_backoff_seconds=60.0,
        max_attempts=5,
        sleep_fn=sleeps.append,
    )

    assert outcome.attempts == 1
    assert outcome.gave_up_after_retries is False
    assert sleeps == []


def test_rate_limiter_enforces_the_delay_between_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    waited: list[float] = []
    limiter = sweep.RateLimiter(8.0, sleep_fn=waited.append)
    # wait() reads monotonic() once to compute `elapsed` (when there is a
    # prior call) and always once more to record the new `_last_call`, so
    # the first wait() consumes one value and the second consumes two.
    times = iter([0.0, 1.0, 1.0])

    monkeypatch.setattr(sweep.time, "monotonic", lambda: next(times))

    limiter.wait()  # baseline, no sleep
    limiter.wait()  # elapsed=1s, remaining=7s

    assert waited == [7.0]


# ---------------------------------------------------------------------------
# Schedule loading / season & game_type filtering
# ---------------------------------------------------------------------------


def _write_schedule_fixture(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "game_id": "2014_01_A_B",
                "season": 2014,
                "week": 1,
                "game_type": "REG",
                "gameday": "2014-09-07",
                "home_team": "B",
                "away_team": "A",
                "pfr": "201409070xyz",
            },
            {
                "game_id": "2013_01_C_D",
                "season": 2013,
                "week": 1,
                "game_type": "REG",
                "gameday": "2013-09-08",
                "home_team": "D",
                "away_team": "C",
                "pfr": "201309080abc",
            },
            {
                # Out of the requested season window -- must be excluded.
                "game_id": "2015_01_E_F",
                "season": 2015,
                "week": 1,
                "game_type": "REG",
                "gameday": "2015-09-06",
                "home_team": "F",
                "away_team": "E",
                "pfr": "201509060def",
            },
            {
                # Postseason -- must be excluded (REG only).
                "game_id": "2014_21_G_H",
                "season": 2014,
                "week": 21,
                "game_type": "WC",
                "gameday": "2015-01-04",
                "home_team": "H",
                "away_team": "G",
                "pfr": "201501040ghi",
            },
            {
                # No pfr id -- must be dropped rather than crash the sweep.
                "game_id": "2014_02_I_J",
                "season": 2014,
                "week": 2,
                "game_type": "REG",
                "gameday": "2014-09-14",
                "home_team": "J",
                "away_team": "I",
                "pfr": None,
            },
        ]
    )
    frame.to_parquet(path, index=False)


def test_load_games_filters_to_reg_season_window_and_drops_missing_pfr(tmp_path: Path) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _write_schedule_fixture(schedule_path)

    games = sweep.load_games(schedule_path, season_start=2013, season_end=2014)

    assert list(games["game_id"]) == ["2013_01_C_D", "2014_01_A_B"]


# ---------------------------------------------------------------------------
# run_sweep: resume-skip, hard-stop, no-capture-found, manifest shape,
# leakage safety
# ---------------------------------------------------------------------------


def _cdx_json(timestamp: str) -> bytes:
    payload = [
        ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"],
        [
            "x",
            timestamp,
            "https://www.pro-football-reference.com/boxscores/x.htm",
            "text/html",
            "200",
            "abc",
            "111",
        ],
    ]
    return json.dumps(payload).encode("utf-8")


_EMPTY_CDX_JSON = json.dumps(
    [["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"]]
).encode("utf-8")


def _table_html() -> str:
    return (FIXTURES / "pfr_boxscore_officials_table.html").read_text(encoding="utf-8")


def _config(tmp_path: Path, schedule_path: Path, **overrides: object) -> sweep.SweepConfig:
    defaults: dict[str, object] = {
        "season_start": 2014,
        "season_end": 2014,
        "run_id": "20260905T000000Z",
        "raw_root": tmp_path / "raw",
        "processed_root": tmp_path / "processed",
        "schedule_path": schedule_path,
        "delay_seconds": 8.0,
        "initial_backoff_seconds": 1.0,
        "max_request_retries": 2,
        "max_consecutive_failures": 2,
    }
    defaults.update(overrides)
    return sweep.SweepConfig(**defaults)  # type: ignore[arg-type]


def _one_game_schedule(path: Path, *, n: int = 1) -> None:
    rows = [
        {
            "game_id": f"2014_0{i}_A{i}_B{i}",
            "season": 2014,
            "week": i,
            "game_type": "REG",
            "gameday": f"2014-09-0{i}",
            "home_team": f"B{i}",
            "away_team": f"A{i}",
            "pfr": f"20140907{i:03d}xyz",
        }
        for i in range(1, n + 1)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_run_sweep_happy_path_writes_manifest_and_parquet_with_leakage_safe_effective_time(
    tmp_path: Path,
) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=1)
    config = _config(tmp_path, schedule_path)

    fetch = _ScriptedFetch(
        [
            sweep.FetchResult(_cdx_json("20141001000000"), 200, None),
            sweep.FetchResult(_table_html().encode("utf-8"), 200, None),
        ]
    )
    summary = sweep.run_sweep(config, fetch_fn=fetch, sleep_fn=lambda _s: None)

    assert summary["games_fetched_ok"] == 1
    assert summary["officials_rows_parsed"] == 7
    assert summary["stopped_early"] is False
    assert summary["total_http_requests"] == 2

    manifest = json.loads((config.raw_root / config.run_id / "manifest.json").read_text())
    assert manifest["schema"] == "officials_pfr_wayback_manifest/1"
    assert len(manifest["games"]) == 1
    row = manifest["games"][0]
    for key in (
        "original_url",
        "wayback_url",
        "wayback_capture_timestamp",
        "fetch_instant_utc",
        "cdx_status_code",
        "replay_status_code",
        "html_file",
        "outcome",
    ):
        assert key in row
    assert row["outcome"] == "fetched"
    assert row["wayback_capture_timestamp"] == "20141001000000"
    assert (config.raw_root / config.run_id / row["html_file"]).exists()

    parquet_path = config.processed_root / config.run_id / "officials_2009_2014.parquet"
    assert parquet_path.exists()
    frame = pd.read_parquet(parquet_path)
    assert len(frame) == 7
    # Leakage safety: the effective time is exactly the game's own date --
    # never the (necessarily later) fetch/capture instant.
    assert (frame["effective_time"] == frame["game_date"]).all()
    assert (frame["effective_time"] == "2014-09-01").all()
    assert set(frame["position"]) == {
        "Referee",
        "Umpire",
        "Head Linesman",
        "Line Judge",
        "Field Judge",
        "Side Judge",
        "Back Judge",
    }

    sidecar = parquet_path.with_name(parquet_path.name + ".provenance.json")
    assert sidecar.exists()


def test_run_sweep_resume_skips_games_already_on_disk_with_zero_new_requests(
    tmp_path: Path,
) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=1)
    config = _config(tmp_path, schedule_path)

    # First run: fetch the one game for real (via the fake fetch_fn).
    first_fetch = _ScriptedFetch(
        [
            sweep.FetchResult(_cdx_json("20141001000000"), 200, None),
            sweep.FetchResult(_table_html().encode("utf-8"), 200, None),
        ]
    )
    sweep.run_sweep(config, fetch_fn=first_fetch, sleep_fn=lambda _s: None)

    # Second run, same run_id: must skip the network entirely for that game.
    def _explode(_url: str) -> sweep.FetchResult:
        raise AssertionError("resume must not re-fetch a game already on disk")

    summary = sweep.run_sweep(config, fetch_fn=_explode, sleep_fn=lambda _s: None)

    assert summary["games_already_on_disk"] == 1
    assert summary["new_fetch_attempts"] == 0
    assert summary["total_http_requests"] == 0
    assert summary["officials_rows_parsed"] == 7


def test_run_sweep_hard_stops_after_consecutive_failures_and_issues_no_further_requests(
    tmp_path: Path,
) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=5)  # more games than the failure budget
    config = _config(tmp_path, schedule_path, max_consecutive_failures=2, max_request_retries=1)

    # Every CDX call 429s; max_request_retries=1 means no backoff sleep is
    # actually needed for the retry loop itself to finish quickly.
    fetch = _ScriptedFetch([sweep.FetchResult(None, 429, "http_429")] * 10)
    summary = sweep.run_sweep(config, fetch_fn=fetch, sleep_fn=lambda _s: None)

    assert summary["stopped_early"] is True
    assert summary["stop_reason"] == "hard_stop_after_2_consecutive_failures"
    assert summary["new_fetch_attempts"] == 2
    assert summary["games_fetched_ok"] == 0
    # Exactly 2 games attempted, 1 CDX call each (max_request_retries=1) = 2
    # total requests -- proof the sweep really stopped rather than grinding
    # through the remaining 3 games in the window.
    assert summary["total_http_requests"] == 2
    assert len(fetch.calls) == 2


def test_no_capture_found_is_not_a_failure_and_does_not_trip_the_hard_stop(
    tmp_path: Path,
) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=3)
    config = _config(tmp_path, schedule_path, max_consecutive_failures=2)

    # Every CDX call succeeds (200) but finds zero captures -- a content
    # result, not a fetch failure, so all 3 games should be attempted.
    fetch = _ScriptedFetch([sweep.FetchResult(_EMPTY_CDX_JSON, 200, None)] * 3)
    summary = sweep.run_sweep(config, fetch_fn=fetch, sleep_fn=lambda _s: None)

    assert summary["stopped_early"] is False
    assert summary["games_no_capture_found"] == 3
    assert summary["new_fetch_attempts"] == 3
    assert summary["total_http_requests"] == 3  # one CDX call per game, no replay call
    assert len(fetch.calls) == 3


def test_delay_floor_is_enforced(tmp_path: Path) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=1)
    config = _config(tmp_path, schedule_path, delay_seconds=1.0)

    with pytest.raises(SystemExit, match="must be >= 8"):
        sweep.run_sweep(config, fetch_fn=_ScriptedFetch([]), sleep_fn=lambda _s: None)


def test_select_capture_timestamp_picks_the_first_data_row_and_handles_empty() -> None:
    assert sweep._select_capture_timestamp(_cdx_json("20141001000000")) == "20141001000000"
    assert sweep._select_capture_timestamp(_EMPTY_CDX_JSON) is None
    assert sweep._select_capture_timestamp(b"not json") is None


def test_pre_game_captures_are_never_selected() -> None:
    """2026-09-07, measured on the first live fetch: the earliest capture of
    2014_01_GB_SEA (played 2014-09-04) was dated 2014-05-30 -- a placeholder
    page with no officials block. The CDX query now starts the day after the
    game and the selector re-applies the bound client-side."""
    assert sweep.capture_not_before("2014-09-04") == "20140905"
    assert sweep.capture_not_before(pd.Timestamp("2014-09-04 20:30")) == "20140905"
    rows = [["urlkey", "timestamp"], ["k", "20140530011957"], ["k", "20140905101010"]]
    payload = json.dumps(rows).encode("utf-8")
    assert sweep._select_capture_timestamp(payload, not_before="20140905") == "20140905101010"
    only_pre = json.dumps([["urlkey", "timestamp"], ["k", "20140530011957"]]).encode("utf-8")
    assert sweep._select_capture_timestamp(only_pre, not_before="20140905") is None
    assert "from={not_before}" in sweep.CDX_URL_TEMPLATE


def test_parses_the_2014_era_ref_info_table_from_a_real_capture() -> None:
    """2026-09-07: the first five live post-game captures (2014 season) all
    parsed zero officials because that era's boxscore names the table
    id="ref_info" (not "officials") and bolds each label. Fixture extracted
    verbatim from the 2014_01_GB_SEA capture."""
    html = (FIXTURES / "pfr_boxscore_officials_ref_info_2014.html").read_text(encoding="utf-8")
    rows, warnings = sweep.parse_officials_block(html)
    assert warnings == []
    assert rows[0] == ("Referee", "John Parry")
    assert dict(rows)["Umpire"] == "Mark Pellis"
    assert dict(rows)["Head Linesman"] == "Derick Bowers"
    assert len(rows) == 7


# ---------------------------------------------------------------------------
# 2026-09-07 (lane N): newest-capture selection, per-game fallback,
# --retry-unparsed, one immutable file per fetch, no duplicate manifest rows.
#
# Measured on live Wayback captures this day (docs/officials_archive_probe.md):
# the 2009-2013 run parsed 0 officials on 418/418 pages because the earliest
# post-game capture of a 2009-2011 boxscore predates PFR's officials block
# (absent 2012-08-19, present 2012-10-25 on the same URL), while the newest
# capture of the same URL parses all seven positions.
# ---------------------------------------------------------------------------


def _cdx_json_multi(*timestamps: str) -> bytes:
    header = ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"]
    rows = [
        [
            "x",
            ts,
            "https://www.pro-football-reference.com/boxscores/x.htm",
            "text/html",
            "200",
            "d",
            "1",
        ]
        for ts in timestamps
    ]
    return json.dumps([header, *rows]).encode("utf-8")


_NO_OFFICIALS_HTML = (
    "<html><body><div id='page_content'><h2>Scoring</h2><h2>Team Stats</h2>"
    "<p>2009-era layout: no game_info, no ref_info, no officials table.</p>"
    "</div></body></html>"
)

# Verbatim structure of the 2026-09-02 capture of 200909100pit (the officials
# table sits inside an HTML comment in the 2016+ layout).
_COMMENTED_OFFICIALS_2026_HTML = """
<div class="placeholder"></div>
<!--
<div class="table_container" id="div_officials">
    <table class="suppress_all sortable stats_table" id="officials" data-cols-to-freeze="0">
    <caption>Officials Table</caption>
    <tr class="thead onecell" ><td class="right center" data-stat="onecell" colspan="2" >Officials</td></tr>
<tr ><th scope="row" class="center " data-stat="ref_pos" >Referee</th><td class="center " data-stat="name" ><a href="/officials/LeavBi0r.htm">Bill Leavy</a></td></tr>
<tr ><th scope="row" class="center " data-stat="ref_pos" >Umpire</th><td class="center " data-stat="name" ><a href="/officials/JenkDa0r.htm">Darrell Jenkins</a></td></tr>
<tr ><th scope="row" class="center " data-stat="ref_pos" >Head Linesman</th><td class="center " data-stat="name" ><a href="/officials/BaltMa0r.htm">Mark Baltz</a></td></tr>
<tr ><th scope="row" class="center " data-stat="ref_pos" >Line Judge</th><td class="center " data-stat="name" ><a href="/officials/PerlMa0r.htm">Mark Perlman</a></td></tr>
<tr ><th scope="row" class="center " data-stat="ref_pos" >Back Judge</th><td class="center " data-stat="name" ><a href="/officials/FergKe0r.htm">Keith Ferguson</a></td></tr>
<tr ><th scope="row" class="center " data-stat="ref_pos" >Side Judge</th><td class="center " data-stat="name" ><a href="/officials/BradGr0r.htm">Greg Bradley</a></td></tr>
<tr ><th scope="row" class="center " data-stat="ref_pos" >Field Judge</th><td class="center " data-stat="name" ><a href="/officials/BlakCl0r.htm">Clete Blakeman</a></td></tr>
</table>
</div>
-->
"""  # noqa: E501


def test_parses_the_2016_plus_layout_with_the_officials_table_inside_a_comment() -> None:
    rows, warnings = sweep.parse_officials_block(_COMMENTED_OFFICIALS_2026_HTML)
    assert warnings == []
    assert rows[0] == ("Referee", "Bill Leavy")
    assert dict(rows)["Field Judge"] == "Clete Blakeman"
    assert len(rows) == 7


def test_rank_capture_timestamps_is_newest_first_deduplicated_and_bounded() -> None:
    payload = _cdx_json_multi(
        "20091002171923", "20130914063806", "20260902021727", "20130914063806", "20090801000000"
    )
    ranked = sweep._rank_capture_timestamps(payload, not_before="20090911")
    assert ranked == ["20260902021727", "20130914063806", "20091002171923"]
    assert sweep._select_capture_timestamp(payload, not_before="20090911") == "20260902021727"
    assert sweep._rank_capture_timestamps(_EMPTY_CDX_JSON) == []
    assert sweep._rank_capture_timestamps(b"not json") == []
    assert sweep._rank_capture_timestamps(b'{"unexpected": 1}') == []


def test_cdx_query_keeps_the_post_game_bound_and_asks_for_every_capture() -> None:
    """2026-09-07 (lane N, measured): ``limit=-3`` on the CDX server drew an
    HTTP 504 (a negative limit forces a full index scan), while unbounded
    queries for the same URLs returned 200 with 61-74 rows. The query
    therefore carries no ``limit`` and the selector ranks client-side."""

    url = sweep.CDX_URL_TEMPLATE.format(original="https://x/y.htm", not_before="20090911")
    assert "from=20090911" in url
    assert "limit=" not in url


def test_run_sweep_fetches_the_newest_post_game_capture_first(tmp_path: Path) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=1)
    config = _config(tmp_path, schedule_path)
    fetch = _ScriptedFetch(
        [
            sweep.FetchResult(
                _cdx_json_multi("20140905101010", "20260902021727", "20160629163948"), 200, None
            ),
            sweep.FetchResult(_table_html().encode("utf-8"), 200, None),
        ]
    )
    summary = sweep.run_sweep(config, fetch_fn=fetch, sleep_fn=lambda _s: None)

    assert summary["officials_rows_parsed"] == 7
    assert summary["total_http_requests"] == 2
    assert summary["fallback_replay_fetches"] == 0
    assert "/web/20260902021727id_/" in fetch.calls[1]
    manifest = json.loads((config.raw_root / config.run_id / "manifest.json").read_text())
    assert manifest["capture_policy"] == "newest_post_game_capture_with_fallback"
    assert manifest["fallback_captures"] == 2
    row = manifest["games"][0]
    assert row["wayback_capture_timestamp"] == "20260902021727"
    assert row["html_file"] == "html/20140907001xyz__20260902021727.html"
    assert (config.raw_root / config.run_id / row["html_file"]).exists()
    assert len(row["attempted_captures"]) == 1
    assert row["fallback_fetches"] == 0
    assert row["retried_unparsed"] is False


def test_fallback_walks_newest_first_and_stops_at_the_first_capture_that_parses(
    tmp_path: Path,
) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=1)
    config = _config(tmp_path, schedule_path, fallback_captures=2)
    fetch = _ScriptedFetch(
        [
            sweep.FetchResult(
                _cdx_json_multi("20241124171706", "20160629163948", "20260902021727"), 200, None
            ),
            sweep.FetchResult(_NO_OFFICIALS_HTML.encode("utf-8"), 200, None),  # 2026: parses 0
            sweep.FetchResult(_table_html().encode("utf-8"), 200, None),  # 2024: parses 7
        ]
    )
    summary = sweep.run_sweep(config, fetch_fn=fetch, sleep_fn=lambda _s: None)

    assert summary["officials_rows_parsed"] == 7
    assert summary["total_http_requests"] == 3  # CDX + 2 replays; the third capture never fetched
    assert summary["fallback_replay_fetches"] == 1
    assert summary["games_parsed_zero"] == 0
    assert "/web/20260902021727id_/" in fetch.calls[1]
    assert "/web/20241124171706id_/" in fetch.calls[2]
    row = json.loads((config.raw_root / config.run_id / "manifest.json").read_text())["games"][0]
    assert row["outcome"] == "fetched"
    assert row["wayback_capture_timestamp"] == "20241124171706"
    assert row["html_file"] == "html/20140907001xyz__20241124171706.html"
    assert row["officials_parsed"] == 7
    assert row["fallback_fetches"] == 1
    assert [a["wayback_capture_timestamp"] for a in row["attempted_captures"]] == [
        "20260902021727",
        "20241124171706",
    ]
    assert row["attempted_captures"][0]["officials_parsed"] == 0
    # Both fetched pages are kept on disk, immutable, one file per fetch.
    html_dir = config.raw_root / config.run_id / "html"
    assert sorted(p.name for p in html_dir.iterdir()) == [
        "20140907001xyz__20241124171706.html",
        "20140907001xyz__20260902021727.html",
    ]


def test_fallback_budget_is_respected_and_zero_fallbacks_means_one_replay(tmp_path: Path) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=1)
    cdx = _cdx_json_multi("20140905101010", "20141006175522", "20241124171706", "20150926165118")

    # fallback_captures=2 -> at most 3 replays even though 4 captures exist.
    config = _config(tmp_path, schedule_path, fallback_captures=2)
    fetch = _ScriptedFetch(
        [sweep.FetchResult(cdx, 200, None)]
        + [sweep.FetchResult(_NO_OFFICIALS_HTML.encode("utf-8"), 200, None)] * 3
    )
    summary = sweep.run_sweep(config, fetch_fn=fetch, sleep_fn=lambda _s: None)
    assert summary["total_http_requests"] == 4
    assert summary["fallback_replay_fetches"] == 2
    assert summary["games_parsed_zero"] == 1
    assert summary["officials_rows_parsed"] == 0
    row = json.loads((config.raw_root / config.run_id / "manifest.json").read_text())["games"][0]
    assert row["outcome"] == "fetched"
    assert row["officials_parsed"] == 0
    assert [a["wayback_capture_timestamp"] for a in row["attempted_captures"]] == [
        "20241124171706",
        "20150926165118",
        "20141006175522",
    ]
    assert row["html_file"] is not None
    assert (config.raw_root / config.run_id / row["html_file"]).exists()

    # fallback_captures=0 -> exactly one replay, the newest.
    config0 = _config(tmp_path / "zero", schedule_path, fallback_captures=0)
    fetch0 = _ScriptedFetch(
        [
            sweep.FetchResult(cdx, 200, None),
            sweep.FetchResult(_NO_OFFICIALS_HTML.encode("utf-8"), 200, None),
        ]
    )
    summary0 = sweep.run_sweep(config0, fetch_fn=fetch0, sleep_fn=lambda _s: None)
    assert summary0["total_http_requests"] == 2
    assert summary0["fallback_replay_fetches"] == 0
    assert "/web/20241124171706id_/" in fetch0.calls[1]


def test_a_fallback_replay_failure_keeps_the_page_already_fetched_and_counts_a_failure(
    tmp_path: Path,
) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=1)
    config = _config(tmp_path, schedule_path, fallback_captures=2, max_request_retries=1)
    fetch = _ScriptedFetch(
        [
            sweep.FetchResult(_cdx_json_multi("20241124171706", "20260902021727"), 200, None),
            sweep.FetchResult(_NO_OFFICIALS_HTML.encode("utf-8"), 200, None),
            sweep.FetchResult(None, 429, "http_429"),
        ]
    )
    summary = sweep.run_sweep(config, fetch_fn=fetch, sleep_fn=lambda _s: None)

    assert summary["stopped_early"] is False
    assert summary["games_fetched_ok"] == 1
    row = json.loads((config.raw_root / config.run_id / "manifest.json").read_text())["games"][0]
    assert row["outcome"] == "fetched"
    assert row["wayback_capture_timestamp"] == "20260902021727"
    assert row["officials_parsed"] == 0
    assert row["attempted_captures"][1]["replay_status_code"] == 429
    assert row["attempted_captures"][1]["html_file"] is None


def _seed_old_policy_run(config: sweep.SweepConfig, html: str, *, capture_ts: str) -> Path:
    """Write a manifest + page the way the pre-2026-09-07 sweep did (html/<pfr>.html)."""

    snapshot_dir = config.raw_root / config.run_id
    (snapshot_dir / "html").mkdir(parents=True)
    old_html = snapshot_dir / "html" / "20140907001xyz.html"
    old_html.write_text(html, encoding="utf-8")
    parsed = len(sweep.parse_officials_block(html)[0])
    original = sweep.ORIGINAL_URL_TEMPLATE.format(pfr_id="20140907001xyz")
    row = {
        "game_id": "2014_01_A1_B1",
        "season": 2014,
        "week": 1,
        "pfr_id": "20140907001xyz",
        "gameday": "2014-09-01",
        "home_team": "B1",
        "away_team": "A1",
        "original_url": original,
        "cdx_url": "old",
        "fetch_instant_utc": "2026-09-07T18:00:00Z",
        "cdx_status_code": 200,
        "cdx_attempts": 1,
        "cdx_backoff_schedule_seconds": [],
        "cdx_error": None,
        "wayback_capture_timestamp": capture_ts,
        "wayback_url": sweep.REPLAY_URL_TEMPLATE.format(ts=capture_ts, original=original),
        "replay_status_code": 200,
        "replay_attempts": 1,
        "replay_backoff_schedule_seconds": [],
        "replay_error": None,
        "html_file": "html/20140907001xyz.html",
        "outcome": "fetched",
        "officials_parsed": parsed,
        "parse_warnings": [],
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps({"schema": "officials_pfr_wayback_manifest/1", "games": [row]}),
        encoding="utf-8",
    )
    return old_html


def test_retry_unparsed_off_never_refetches_a_zero_parse_page(tmp_path: Path) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=1)
    config = _config(tmp_path, schedule_path)  # retry_unparsed defaults to False
    _seed_old_policy_run(config, _NO_OFFICIALS_HTML, capture_ts="20140905101010")

    def _explode(_url: str) -> sweep.FetchResult:
        raise AssertionError("without --retry-unparsed a page on disk is never refetched")

    summary = sweep.run_sweep(config, fetch_fn=_explode, sleep_fn=lambda _s: None)
    assert summary["games_already_on_disk"] == 1
    assert summary["total_http_requests"] == 0
    assert summary["games_retried_unparsed"] == 0
    assert summary["officials_rows_parsed"] == 0
    assert config.retry_unparsed is False


def test_retry_unparsed_refetches_newer_captures_excluding_the_one_already_on_disk(
    tmp_path: Path,
) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=1)
    config = _config(tmp_path, schedule_path, retry_unparsed=True)
    old_html = _seed_old_policy_run(config, _NO_OFFICIALS_HTML, capture_ts="20140905101010")
    old_bytes = old_html.read_bytes()

    fetch = _ScriptedFetch(
        [
            sweep.FetchResult(
                _cdx_json_multi("20140905101010", "20160629163948", "20260902021727"), 200, None
            ),
            sweep.FetchResult(_table_html().encode("utf-8"), 200, None),
        ]
    )
    summary = sweep.run_sweep(config, fetch_fn=fetch, sleep_fn=lambda _s: None)

    assert summary["games_retried_unparsed"] == 1
    assert summary["total_http_requests"] == 2
    assert summary["officials_rows_parsed"] == 7
    assert "/web/20260902021727id_/" in fetch.calls[1]
    manifest = json.loads((config.raw_root / config.run_id / "manifest.json").read_text())
    assert len(manifest["games"]) == 1  # upserted, not duplicated
    row = manifest["games"][0]
    assert row["retried_unparsed"] is True
    assert row["outcome"] == "fetched"
    assert row["officials_parsed"] == 7
    assert row["wayback_capture_timestamp"] == "20260902021727"
    assert row["html_file"] == "html/20140907001xyz__20260902021727.html"
    assert [a["wayback_capture_timestamp"] for a in row["attempted_captures"]] == [
        "20140905101010",
        "20260902021727",
    ]
    assert row["attempted_captures"][0]["from_previous_run"] is True
    assert row["attempted_captures"][0]["html_file"] == "html/20140907001xyz.html"
    # The old capture is immutable: still on disk, byte-identical.
    assert old_html.read_bytes() == old_bytes
    frame = pd.read_parquet(config.processed_root / config.run_id / "officials_2009_2014.parquet")
    assert (frame["wayback_capture_timestamp"] == "20260902021727").all()
    assert (frame["effective_time"] == "2014-09-01").all()


def test_retry_unparsed_skips_a_page_that_now_parses_under_the_current_parser(
    tmp_path: Path,
) -> None:
    """The 2014 run's first five rows say officials_parsed=0 because the
    ref_info parser fix landed after they were fetched and the resume path
    never wrote back; a retry must re-parse from disk, update the row, and
    spend zero requests."""

    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=1)
    config = _config(tmp_path, schedule_path, retry_unparsed=True)
    _seed_old_policy_run(config, _table_html(), capture_ts="20141006175522")
    snapshot_dir = config.raw_root / config.run_id
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    manifest["games"][0]["officials_parsed"] = 0  # stale, as written by the old parser
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _explode(_url: str) -> sweep.FetchResult:
        raise AssertionError("a page that parses under the current parser is never refetched")

    summary = sweep.run_sweep(config, fetch_fn=_explode, sleep_fn=lambda _s: None)
    assert summary["total_http_requests"] == 0
    assert summary["games_retried_unparsed"] == 0
    assert summary["officials_rows_parsed"] == 7
    row = json.loads((snapshot_dir / "manifest.json").read_text())["games"][0]
    assert row["officials_parsed"] == 7


def test_retry_unparsed_with_no_newer_capture_keeps_the_old_page_without_duplicating_the_row(
    tmp_path: Path,
) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=1)
    config = _config(tmp_path, schedule_path, retry_unparsed=True)
    _seed_old_policy_run(config, _NO_OFFICIALS_HTML, capture_ts="20140905101010")

    fetch = _ScriptedFetch([sweep.FetchResult(_cdx_json_multi("20140905101010"), 200, None)])
    summary = sweep.run_sweep(config, fetch_fn=fetch, sleep_fn=lambda _s: None)

    assert summary["total_http_requests"] == 1
    assert summary["games_retried_unparsed"] == 1
    assert summary["games_retry_no_new_capture"] == 1
    assert summary["games_no_capture_found"] == 0
    manifest_path = config.raw_root / config.run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["games"]) == 1
    row = manifest["games"][0]
    assert row["outcome"] == "fetched"
    assert row["html_file"] == "html/20140907001xyz.html"
    assert row["officials_parsed"] == 0
    assert "no post-game capture beyond" in row["retry_note"]
    # A further resume with the same flag re-issues the single CDX call and
    # still leaves exactly one row for the game.
    fetch2 = _ScriptedFetch([sweep.FetchResult(_cdx_json_multi("20140905101010"), 200, None)])
    summary2 = sweep.run_sweep(config, fetch_fn=fetch2, sleep_fn=lambda _s: None)
    assert summary2["total_http_requests"] == 1
    assert len(json.loads(manifest_path.read_text())["games"]) == 1


def test_limit_caps_new_fetches_but_still_reparses_every_page_on_disk(tmp_path: Path) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=3)
    config = _config(tmp_path, schedule_path)

    # Fetch game 1 and game 3 only (game 2 is throttled and left unfetched).
    first = _ScriptedFetch(
        [
            sweep.FetchResult(_cdx_json_multi("20260902021727"), 200, None),
            sweep.FetchResult(_table_html().encode("utf-8"), 200, None),
            sweep.FetchResult(None, 404, "http_404"),
            sweep.FetchResult(_cdx_json_multi("20260902021727"), 200, None),
            sweep.FetchResult(_table_html().encode("utf-8"), 200, None),
        ]
    )
    sweep.run_sweep(config, fetch_fn=first, sleep_fn=lambda _s: None)

    # Resume with --limit 0: no network at all, yet BOTH on-disk games are
    # re-parsed into the parquet (game 3 sits after the unfetched game 2).
    def _explode(_url: str) -> sweep.FetchResult:
        raise AssertionError("--limit 0 must issue no requests")

    config0 = _config(tmp_path, schedule_path, limit=0)
    summary0 = sweep.run_sweep(config0, fetch_fn=_explode, sleep_fn=lambda _s: None)
    assert summary0["total_http_requests"] == 0
    assert summary0["stop_reason"] == "limit_reached"
    assert summary0["officials_rows_parsed"] == 14


def test_failed_games_are_re_attempted_on_resume_without_duplicate_rows(tmp_path: Path) -> None:
    schedule_path = tmp_path / "schedules.parquet"
    _one_game_schedule(schedule_path, n=1)
    config = _config(tmp_path, schedule_path, max_request_retries=1)

    first = _ScriptedFetch([sweep.FetchResult(None, 429, "http_429")])
    sweep.run_sweep(config, fetch_fn=first, sleep_fn=lambda _s: None)
    second = _ScriptedFetch(
        [
            sweep.FetchResult(_cdx_json_multi("20260902021727"), 200, None),
            sweep.FetchResult(_table_html().encode("utf-8"), 200, None),
        ]
    )
    summary = sweep.run_sweep(config, fetch_fn=second, sleep_fn=lambda _s: None)

    assert summary["officials_rows_parsed"] == 7
    manifest = json.loads((config.raw_root / config.run_id / "manifest.json").read_text())
    assert len(manifest["games"]) == 1
    assert manifest["games"][0]["outcome"] == "fetched"
    assert summary["games_cdx_or_replay_failed"] == 0


def test_cli_exposes_fallback_captures_and_retry_unparsed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[sweep.SweepConfig] = []

    def _fake_run_sweep(config: sweep.SweepConfig, **_kwargs: object) -> dict[str, object]:
        captured.append(config)
        return {
            "run_id": config.run_id,
            "season_start": config.season_start,
            "season_end": config.season_end,
            "games_in_window": 0,
            "games_already_on_disk": 0,
            "new_fetch_attempts": 0,
            "games_fetched_ok": 0,
            "games_cdx_or_replay_failed": 0,
            "games_no_capture_found": 0,
            "total_http_requests": 0,
            "officials_rows_parsed": 0,
            "games_parsed_zero": 0,
            "fallback_replay_fetches": 0,
            "games_retried_unparsed": 0,
            "stopped_early": False,
            "stop_reason": None,
        }

    monkeypatch.setattr(sweep, "run_sweep", _fake_run_sweep)

    assert sweep.main(["--season-start", "2009", "--season-end", "2013"]) == 0
    assert captured[-1].fallback_captures == 2
    assert captured[-1].retry_unparsed is False

    argv = [
        "--season-start",
        "2009",
        "--season-end",
        "2013",
        "--run-id",
        "20260907T175420Z",
        "--retry-unparsed",
        "--fallback-captures",
        "3",
    ]
    assert sweep.main(argv) == 0
    assert captured[-1].run_id == "20260907T175420Z"
    assert captured[-1].fallback_captures == 3
    assert captured[-1].retry_unparsed is True

    with pytest.raises(SystemExit):
        sweep.main(["--fallback-captures", "-1"])
