"""Tests for scripts/officials_wayback_sweep.py (LEAD-59).

No network anywhere in this file: every fetch is a fake ``FetchFn`` (a plain
callable stub) and every backoff sleep is a fake ``sleep_fn`` that records
its argument instead of blocking. Covers: the officials-block parser against
two constructed fixture pages (table strategy, inline-line fallback
strategy), the backoff/retry schedule, the resume-skip logic, the hard-stop
consecutive-failure counter, the "no capture found" non-failure path, the
manifest shape, season/game-type filtering of the schedule snapshot, and the
leakage-safety invariant (``effective_time`` == the game's own date, never a
later timestamp).
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
