"""Tests for ENG-03 capture/scheduler observability.

Covers three new surfaces, all additive to the existing scheduler:

* ``data/scheduler_heartbeat.json`` -- written by the daemon loop on every
  poll, read back by ``capture_scheduler.build_health_report``.
* ``state["job_health"]`` -- a NEW sibling key next to the pre-existing
  ``state["runs"]``, recording per-job last_success_at / last_failure_at /
  last_error / consecutive_failures / missed_window_count.
* ``nfl_ats.capture_freshness`` -- per-source on-disk freshness derived from
  the SCHEDULE's own day/at/grace fields, plus ``capture_scheduler.py``'s
  ``--health`` fail-visible summary built on top of it.

``tests/test_capture_scheduler.py`` (the pre-existing scheduler test suite)
must keep passing unchanged; nothing here modifies existing state["runs"]
semantics, only adds alongside them.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import scripts.capture_scheduler as capture_scheduler
from nfl_ats import capture_freshness as cf
from scripts.capture_scheduler import ET, Job


def make_job(**overrides: Any) -> Job:
    defaults: dict[str, Any] = {
        "name": "demo_job",
        "day": "sun",
        "at": "22:00",
        "grace_minutes": 300,
        "command": ["cmd.exe", "/c", "echo"],
        "enabled": True,
        "why": "demo",
        "season_guarded": False,
    }
    defaults.update(overrides)
    return Job(**defaults)


def stamp_dir(root: Path, when: datetime) -> Path:
    """Create an empty UTC-stamped snapshot directory, project convention."""

    name = when.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


NOW = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# nfl_ats.capture_freshness: cadence derivation
# ---------------------------------------------------------------------------


def test_single_weekly_job_gets_a_week_plus_grace_budget() -> None:
    job = make_job(day="tue", at="09:00", grace_minutes=180)
    budget = cf.derive_budget_minutes([job])
    assert budget == 7 * 24 * 60 + 180


def test_two_jobs_budget_is_the_larger_weekly_gap_plus_its_grace() -> None:
    tue = make_job(name="a", day="tue", at="09:00", grace_minutes=180)
    thu = make_job(name="b", day="thu", at="18:00", grace_minutes=90)

    tue_minutes = 1 * 1440 + 9 * 60
    thu_minutes = 3 * 1440 + 18 * 60
    gap_forward = thu_minutes - tue_minutes
    gap_wrap = tue_minutes + 7 * 24 * 60 - thu_minutes
    expected_gap = max(gap_forward, gap_wrap)

    budget = cf.derive_budget_minutes([tue, thu])

    assert budget == expected_gap + 180  # largest grace among the two jobs


def test_a_fully_disabled_job_group_has_no_budget() -> None:
    job = make_job(enabled=False)
    assert cf.derive_budget_minutes([job]) is None


def test_disabled_jobs_are_excluded_from_the_gap_computation() -> None:
    """A disabled sibling must not change the budget of the enabled ones."""

    enabled_job = make_job(name="a", day="tue", at="09:00", grace_minutes=180)
    disabled_job = make_job(name="b", day="wed", at="09:00", grace_minutes=10, enabled=False)

    only_enabled = cf.derive_budget_minutes([enabled_job])
    with_disabled_sibling = cf.derive_budget_minutes([enabled_job, disabled_job])

    assert only_enabled == with_disabled_sibling


def test_group_by_source_ignores_jobs_with_no_dedupe_dir() -> None:
    orchestration_job = make_job(dedupe_dir="")
    capture_job = make_job(name="capture", dedupe_dir="data/market/raw")

    groups = cf.group_by_source([orchestration_job, capture_job])

    assert list(groups) == ["data/market/raw"]


# ---------------------------------------------------------------------------
# nfl_ats.capture_freshness: locators
# ---------------------------------------------------------------------------


def test_newest_snapshot_instant_reads_the_newest_directory_name(tmp_path: Path) -> None:
    older = NOW - timedelta(days=2)
    newer = NOW - timedelta(hours=1)
    stamp_dir(tmp_path, older)
    stamp_dir(tmp_path, newer)
    (tmp_path / "not_a_snapshot").mkdir()

    found = cf.newest_snapshot_instant(tmp_path)

    assert found == newer.replace(microsecond=0)


def test_newest_snapshot_instant_ignores_filesystem_mtime(tmp_path: Path) -> None:
    """The directory NAME is authoritative even when a stale-named directory
    has the newest mtime on disk -- matches the same rule
    ``scripts/capture_scheduler.newest_snapshot_age_minutes`` documents."""

    newer_named = stamp_dir(tmp_path, NOW - timedelta(days=1))
    older_named = stamp_dir(tmp_path, NOW - timedelta(days=5))
    old_time = (NOW - timedelta(days=30)).timestamp()
    new_time = NOW.timestamp()
    os.utime(newer_named, (old_time, old_time))
    os.utime(older_named, (new_time, new_time))

    found = cf.newest_snapshot_instant(tmp_path)

    assert found == (NOW - timedelta(days=1)).replace(microsecond=0)


def test_newest_snapshot_instant_missing_directory_is_none(tmp_path: Path) -> None:
    assert cf.newest_snapshot_instant(tmp_path / "absent") is None


def test_newest_json_field_instant_reads_the_compact_utc_stamp(tmp_path: Path) -> None:
    path = tmp_path / "lineups.json"
    path.write_text(json.dumps({"generated_at": "20260910T113000Z"}), encoding="utf-8")

    found = cf.newest_json_field_instant(path, "generated_at")

    assert found == datetime(2026, 9, 10, 11, 30, 0, tzinfo=UTC)


def test_newest_json_field_instant_missing_file_is_none(tmp_path: Path) -> None:
    assert cf.newest_json_field_instant(tmp_path / "absent.json", "generated_at") is None


def test_newest_json_field_instant_malformed_json_is_none(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert cf.newest_json_field_instant(path, "generated_at") is None


# ---------------------------------------------------------------------------
# nfl_ats.capture_freshness: compute_freshness status paths (fresh/stale/
# missing/disabled) -- the DoD's required coverage.
# ---------------------------------------------------------------------------


def test_compute_freshness_fresh_source(tmp_path: Path) -> None:
    job = make_job(name="odds_x", day="tue", at="09:00", grace_minutes=180, dedupe_dir="market/raw")
    stamp_dir(tmp_path / "market/raw", NOW - timedelta(minutes=30))

    [source] = cf.compute_freshness([job], repo_root=tmp_path, now=NOW)

    assert source.status == "fresh"
    assert source.age_minutes == pytest.approx(30.0, abs=0.1)
    assert source.expected_active is True


def test_compute_freshness_stale_source(tmp_path: Path) -> None:
    job = make_job(
        name="referee_x", day="wed", at="15:00", grace_minutes=240, dedupe_dir="players/referee"
    )
    stamp_dir(tmp_path / "players/referee", NOW - timedelta(days=10))

    [source] = cf.compute_freshness([job], repo_root=tmp_path, now=NOW)

    assert source.status == "stale"
    assert source.age_minutes is not None
    assert source.budget_minutes is not None
    assert source.age_minutes > source.budget_minutes


def test_compute_freshness_missing_source_with_no_snapshot_at_all(tmp_path: Path) -> None:
    job = make_job(
        name="arrests_x", day="tue", at="07:00", grace_minutes=90, dedupe_dir="raw/arrests"
    )

    [source] = cf.compute_freshness([job], repo_root=tmp_path, now=NOW)

    assert source.status == "missing"
    assert source.newest_artifact_at is None


def test_compute_freshness_disabled_when_every_job_for_the_source_is_disabled(
    tmp_path: Path,
) -> None:
    job = make_job(name="nflcom_x", enabled=False, dedupe_dir="raw/nflcom_injuries")

    [source] = cf.compute_freshness([job], repo_root=tmp_path, now=NOW)

    assert source.status == "disabled"
    assert source.expected_active is False


def test_compute_freshness_missing_offseason_source_is_not_unexpected(tmp_path: Path) -> None:
    job = make_job(name="inactives_x", dedupe_dir="players/inactives", season_guarded=True)

    [source] = cf.compute_freshness(
        [job], repo_root=tmp_path, now=NOW, season_active=lambda _now: False
    )

    assert source.status == "missing"
    assert source.expected_active is False
    assert not cf.any_unexpected_missing([source])


def test_compute_freshness_missing_in_season_source_is_unexpected(tmp_path: Path) -> None:
    job = make_job(name="inactives_x", dedupe_dir="players/inactives", season_guarded=True)

    [source] = cf.compute_freshness(
        [job], repo_root=tmp_path, now=NOW, season_active=lambda _now: True
    )

    assert source.status == "missing"
    assert source.expected_active is True
    assert cf.any_unexpected_missing([source])


def test_compute_freshness_uses_the_json_field_locator_for_lineups(tmp_path: Path) -> None:
    job = make_job(name="lineups_x", dedupe_dir="artifacts/lineups")
    payload_dir = tmp_path / "artifacts/lineups/current"
    payload_dir.mkdir(parents=True)
    stamp = (NOW - timedelta(minutes=45)).strftime("%Y%m%dT%H%M%SZ")
    (payload_dir / "lineups.json").write_text(json.dumps({"generated_at": stamp}), encoding="utf-8")

    [source] = cf.compute_freshness([job], repo_root=tmp_path, now=NOW)

    assert source.status == "fresh"
    assert source.age_minutes == pytest.approx(45.0, abs=0.1)


def test_compute_freshness_groups_multiple_jobs_sharing_a_dedupe_dir(tmp_path: Path) -> None:
    a = make_job(name="odds_a", day="tue", at="09:00", grace_minutes=180, dedupe_dir="market/raw")
    b = make_job(name="odds_b", day="thu", at="18:00", grace_minutes=90, dedupe_dir="market/raw")
    stamp_dir(tmp_path / "market/raw", NOW - timedelta(minutes=10))

    sources = cf.compute_freshness([a, b], repo_root=tmp_path, now=NOW)

    assert len(sources) == 1
    assert sources[0].job_names == ("odds_a", "odds_b")
    assert sources[0].enabled_job_count == 2


def test_render_table_and_as_dict_cover_every_status(tmp_path: Path) -> None:
    fresh_job = make_job(name="fresh_job", day="tue", at="09:00", grace_minutes=180, dedupe_dir="a")
    stamp_dir(tmp_path / "a", NOW - timedelta(minutes=1))
    missing_job = make_job(
        name="missing_job", day="wed", at="09:00", grace_minutes=180, dedupe_dir="b"
    )
    disabled_job = make_job(name="disabled_job", enabled=False, dedupe_dir="c")

    sources = cf.compute_freshness(
        [fresh_job, missing_job, disabled_job], repo_root=tmp_path, now=NOW
    )
    table = cf.render_table(sources)

    assert "[fresh]" in table
    assert "[missing]" in table
    assert "[disabled]" in table
    json.dumps([source.as_dict() for source in sources])  # must be JSON-serializable


# ---------------------------------------------------------------------------
# capture_scheduler.py: heartbeat (item 1)
# ---------------------------------------------------------------------------


def test_write_heartbeat_writes_a_file_separate_from_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", state_path)
    started = datetime(2026, 9, 10, 8, 0, tzinfo=ET)
    now = datetime(2026, 9, 10, 8, 1, tzinfo=ET)

    capture_scheduler.write_heartbeat(started_at=started, now=now)

    assert heartbeat_path.is_file()
    assert not state_path.exists()
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload["pid"] > 0
    assert payload["poll_seconds"] == capture_scheduler.POLL_SECONDS
    assert payload["enabled_job_count"] == sum(j.enabled for j in capture_scheduler.SCHEDULE)
    assert payload["started_at"] == started.isoformat(timespec="seconds")
    assert payload["last_poll_at"] == now.isoformat(timespec="seconds")


def test_read_heartbeat_missing_file_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", tmp_path / "absent.json")
    assert capture_scheduler.read_heartbeat() is None


def test_read_heartbeat_malformed_file_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "heartbeat.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", path)
    assert capture_scheduler.read_heartbeat() is None


# ---------------------------------------------------------------------------
# capture_scheduler.py: build_health_report / render_health (item 4)
# ---------------------------------------------------------------------------


def _heartbeat_payload(
    now: datetime,
    *,
    age: timedelta,
    code_sha256: str | None = None,
    schedule_digest: str | None = None,
) -> str:
    """ENG-26: ``code_sha256``/``schedule_digest`` default to omitted (a
    pre-ENG-26-style heartbeat, deliberately reported STALE by
    ``build_health_report`` -- see that function's docstring), not to a
    freshly-computed value, so a test asserting a DEAD/FAIL outcome for an
    unrelated reason does not need to know or care about code identity."""

    payload: dict[str, Any] = {
        "pid": 4242,
        "started_at": (now - timedelta(hours=1)).isoformat(timespec="seconds"),
        "last_poll_at": (now - age).isoformat(timespec="seconds"),
        "poll_seconds": 60,
        "enabled_job_count": 10,
    }
    if code_sha256 is not None:
        payload["code_sha256"] = code_sha256
    if schedule_digest is not None:
        payload["schedule_digest"] = schedule_digest
    return json.dumps(payload)


def test_health_report_alive_when_heartbeat_is_recent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", ())
    # code/schedule must be CURRENT for this test's "everything is fine"
    # scenario -- compute them after the SCHEDULE monkeypatch above so the
    # schedule digest matches what build_health_report will recompute.
    heartbeat_path.write_text(
        _heartbeat_payload(
            now,
            age=timedelta(seconds=30),
            code_sha256=capture_scheduler.compute_code_sha256(),
            schedule_digest=capture_scheduler.compute_schedule_digest(),
        ),
        encoding="utf-8",
    )

    report = capture_scheduler.build_health_report(now, {"runs": {}})

    assert report["heartbeat"]["daemon_alive"] is True
    assert report["ok"] is True
    assert "ALIVE" in capture_scheduler.render_health(report)


def test_health_report_dead_when_heartbeat_is_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)
    heartbeat_path.write_text(_heartbeat_payload(now, age=timedelta(minutes=10)), encoding="utf-8")
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", ())

    report = capture_scheduler.build_health_report(now, {"runs": {}})

    assert report["heartbeat"]["daemon_alive"] is False
    assert report["ok"] is False
    assert "DEAD" in capture_scheduler.render_health(report)


def test_health_report_dead_when_heartbeat_file_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", ())
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)

    report = capture_scheduler.build_health_report(now, {"runs": {}})

    assert report["heartbeat"]["exists"] is False
    assert report["ok"] is False
    assert "MISSING" in capture_scheduler.render_health(report)


def test_health_report_lists_missed_rows_and_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)
    heartbeat_path.write_text(_heartbeat_payload(now, age=timedelta(seconds=1)), encoding="utf-8")
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", ())
    state = {
        "runs": {
            "backup_data@2026-08-30": {
                "status": "MISSED",
                "window_start": "2026-08-30T22:00:00-04:00",
            }
        }
    }

    report = capture_scheduler.build_health_report(now, state)

    assert report["ok"] is False
    assert len(report["missed"]) == 1
    assert "backup_data@2026-08-30" in capture_scheduler.render_health(report)


def test_health_report_ok_end_to_end_with_a_real_fresh_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)
    job = make_job(
        name="odds_ok",
        day="tue",
        at="09:00",
        grace_minutes=180,
        dedupe_dir="market/raw",
        season_guarded=False,
    )
    stamp_dir(tmp_path / "market/raw", now.astimezone(UTC) - timedelta(minutes=5))
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "REPO", tmp_path)
    # Compute AFTER the SCHEDULE monkeypatch above, same reasoning as
    # test_health_report_alive_when_heartbeat_is_recent.
    heartbeat_path.write_text(
        _heartbeat_payload(
            now,
            age=timedelta(seconds=1),
            code_sha256=capture_scheduler.compute_code_sha256(),
            schedule_digest=capture_scheduler.compute_schedule_digest(),
        ),
        encoding="utf-8",
    )

    report = capture_scheduler.build_health_report(now, {"runs": {}})

    assert report["ok"] is True
    assert "OVERALL: OK" in capture_scheduler.render_health(report)


def test_health_report_fails_when_an_expected_active_source_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)
    heartbeat_path.write_text(_heartbeat_payload(now, age=timedelta(seconds=1)), encoding="utf-8")
    job = make_job(
        name="odds_missing",
        day="tue",
        at="09:00",
        grace_minutes=180,
        dedupe_dir="market/raw",
        season_guarded=False,
    )
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "REPO", tmp_path)  # empty: no market/raw dir at all

    report = capture_scheduler.build_health_report(now, {"runs": {}})

    assert report["ok"] is False
    assert "OVERALL: FAIL" in capture_scheduler.render_health(report)


def test_health_report_json_is_serializable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", ())
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)

    report = capture_scheduler.build_health_report(now, {"runs": {}})
    text = json.dumps(capture_scheduler._health_report_json(report), sort_keys=True)

    assert '"ok": false' in text


# ---------------------------------------------------------------------------
# capture_scheduler.py: per-job persisted health, state["job_health"] (item 2)
# ---------------------------------------------------------------------------


def test_run_job_success_updates_job_health_and_resets_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job(command=["cmd.exe", "/c", "echo", "ok"])
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    state: dict[str, Any] = {
        "runs": {},
        "job_health": {
            job.name: {**capture_scheduler._DEFAULT_JOB_HEALTH, "consecutive_failures": 3}
        },
    }

    capture_scheduler.run_job(job, datetime(2026, 9, 10, 9, 0, tzinfo=ET), state)

    entry = state["job_health"][job.name]
    assert entry["last_success_at"] is not None
    assert entry["consecutive_failures"] == 0


def test_run_job_failure_records_error_and_increments_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job(command=["cmd.exe", "/c", "exit", "1"])
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    state: dict[str, Any] = {"runs": {}}

    capture_scheduler.run_job(job, datetime(2026, 9, 10, 9, 0, tzinfo=ET), state)

    entry = state["job_health"][job.name]
    assert entry["last_failure_at"] is not None
    assert entry["consecutive_failures"] == 1
    assert isinstance(entry["last_error"], str)


def test_consecutive_failures_accumulate_across_repeated_run_job_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job(name="flaky", command=["cmd.exe", "/c", "exit", "1"])
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    state: dict[str, Any] = {"runs": {}}

    for _ in range(3):
        capture_scheduler.run_job(job, datetime(2026, 9, 10, 9, 0, tzinfo=ET), state)

    assert state["job_health"][job.name]["consecutive_failures"] == 3


def test_record_already_captured_counts_as_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job()
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    start = datetime(2026, 9, 10, 9, 0, tzinfo=ET)
    state: dict[str, Any] = {"runs": {}}

    capture_scheduler.record_already_captured(job, start, 12.3, state)

    entry = state["job_health"][job.name]
    assert entry["last_success_at"] is not None
    assert entry["consecutive_failures"] == 0


def test_sweep_missed_increments_missed_window_count_once_per_occurrence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job(catch_up=False)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    thursday = datetime(2026, 9, 10, 23, 0, tzinfo=ET)
    state: dict[str, Any] = {"runs": {}}

    capture_scheduler.sweep_missed(thursday, state)
    capture_scheduler.sweep_missed(thursday, state)  # same occurrence: must be a no-op

    assert state["job_health"][job.name]["missed_window_count"] == 1


def test_job_health_key_does_not_disturb_state_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """show_status and every pre-existing scheduler test read only
    state["runs"]; the new sibling key must leave it exactly as before."""

    job = make_job(catch_up=True, command=["cmd.exe", "/c", "echo", "caught-up-ok"])
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    thursday = datetime(2026, 8, 27, 17, 0, tzinfo=ET)
    state: dict[str, Any] = {"runs": {}}

    capture_scheduler.sweep_missed(thursday, state)

    assert "job_health" in state
    start = capture_scheduler.occurrence(job, thursday)
    key = f"{job.name}@{start.date().isoformat()}"
    assert state["runs"][key]["status"] == "CAUGHT_UP"
    assert set(state["runs"][key]) == {"status", "window_start", "ran_at", "caught_up"}


# ---------------------------------------------------------------------------
# ENG-26: code-version guard -- heartbeat identity, --health STALE flag,
# --is-running double-start guard, --acknowledge-missed.
# ---------------------------------------------------------------------------


def test_write_heartbeat_records_code_and_schedule_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    started = datetime(2026, 9, 10, 8, 0, tzinfo=ET)
    now = datetime(2026, 9, 10, 8, 1, tzinfo=ET)

    capture_scheduler.write_heartbeat(
        started_at=started,
        now=now,
        code_sha256="deadbeef" * 8,
        schedule_digest="cafef00d" * 8,
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload["code_sha256"] == "deadbeef" * 8
    assert payload["schedule_digest"] == "cafef00d" * 8


def test_write_heartbeat_defaults_code_and_schedule_when_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Back-compat: a caller that does not pass the new kwargs (the
    pre-ENG-26 call shape) still gets a valid pair matching disk right now,
    rather than a missing field or KeyError."""
    heartbeat_path = tmp_path / "heartbeat.json"
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    started = datetime(2026, 9, 10, 8, 0, tzinfo=ET)
    now = datetime(2026, 9, 10, 8, 1, tzinfo=ET)

    capture_scheduler.write_heartbeat(started_at=started, now=now)

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload["code_sha256"] == capture_scheduler.compute_code_sha256()
    assert payload["schedule_digest"] == capture_scheduler.compute_schedule_digest()


def test_daemon_loop_freezes_code_identity_at_startup_not_per_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins that main()'s daemon branch computes code_sha256/schedule_digest
    ONCE before the loop and passes the SAME values into every write_heartbeat
    call -- the source of truth for "what the daemon started with"."""
    calls: list[dict[str, Any]] = []

    def fake_write_heartbeat(**kwargs: Any) -> None:
        calls.append(kwargs)
        if len(calls) >= 2:
            raise SystemExit(0)  # stop the infinite loop after two polls

    monkeypatch.setattr(capture_scheduler, "write_heartbeat", fake_write_heartbeat)
    monkeypatch.setattr(capture_scheduler, "load_state", lambda: {"runs": {}})
    monkeypatch.setattr(capture_scheduler, "due_jobs", lambda now, state: [])
    monkeypatch.setattr(capture_scheduler, "sweep_missed", lambda now, state: None)
    monkeypatch.setattr(capture_scheduler, "prune", lambda state: None)
    monkeypatch.setattr(capture_scheduler, "save_state", lambda state: None)
    monkeypatch.setattr(capture_scheduler, "log", lambda message: None)
    monkeypatch.setattr(capture_scheduler.time, "sleep", lambda seconds: None)

    with pytest.raises(SystemExit):
        capture_scheduler.main([])

    assert len(calls) == 2
    assert calls[0]["code_sha256"] == calls[1]["code_sha256"]
    assert calls[0]["schedule_digest"] == calls[1]["schedule_digest"]
    assert calls[0]["code_sha256"] == capture_scheduler.compute_code_sha256()


def test_health_report_flags_stale_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", ())
    heartbeat_path.write_text(
        _heartbeat_payload(
            now,
            age=timedelta(seconds=1),
            code_sha256="0" * 64,  # deliberately wrong
            schedule_digest=capture_scheduler.compute_schedule_digest(),
        ),
        encoding="utf-8",
    )

    report = capture_scheduler.build_health_report(now, {"runs": {}})
    rendered = capture_scheduler.render_health(report)

    assert report["code_version"]["code_current"] is False
    assert report["code_version"]["schedule_current"] is True
    assert report["ok"] is False
    assert "STALE" in rendered
    assert "scripts/stop_capture_scheduler.cmd" in rendered
    assert "scripts/start_capture_scheduler.cmd" in rendered


def test_health_report_flags_stale_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", ())
    heartbeat_path.write_text(
        _heartbeat_payload(
            now,
            age=timedelta(seconds=1),
            code_sha256=capture_scheduler.compute_code_sha256(),
            schedule_digest="0" * 64,  # deliberately wrong
        ),
        encoding="utf-8",
    )

    report = capture_scheduler.build_health_report(now, {"runs": {}})

    assert report["code_version"]["code_current"] is True
    assert report["code_version"]["schedule_current"] is False
    assert report["ok"] is False
    assert "STALE" in capture_scheduler.render_health(report)


def test_health_report_treats_a_legacy_heartbeat_without_hashes_as_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The exact 2026-09-04 situation this item was written for: a daemon
    that started before ENG-26 existed writes a heartbeat with neither
    field, and --health must fail closed instead of silently skipping the
    check because it cannot prove currency."""
    heartbeat_path = tmp_path / "heartbeat.json"
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)
    heartbeat_path.write_text(_heartbeat_payload(now, age=timedelta(seconds=1)), encoding="utf-8")
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", ())

    report = capture_scheduler.build_health_report(now, {"runs": {}})

    assert report["heartbeat"]["daemon_alive"] is True  # the daemon itself IS alive
    assert report["code_version"]["code_current"] is False
    assert report["code_version"]["schedule_current"] is False
    assert report["ok"] is False


def test_short_hash_handles_none_and_truncates() -> None:
    assert capture_scheduler._short_hash(None) == "unknown"
    assert capture_scheduler._short_hash("abcdef0123456789") == "abcdef012345"


def test_acknowledge_missed_adds_reason_without_deleting_the_missed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    state: dict[str, Any] = {
        "runs": {
            "backup_data@2026-08-30": {
                "status": "MISSED",
                "window_start": "2026-08-30T22:00:00-04:00",
            }
        }
    }

    code = capture_scheduler.acknowledge_missed(
        "backup_data@2026-08-30", "backup run manually 2026-09-04", state
    )

    assert code == 0
    record = state["runs"]["backup_data@2026-08-30"]
    assert record["status"] == "MISSED"
    assert record["acknowledged"]["reason"] == "backup run manually 2026-09-04"
    assert "at" in record["acknowledged"]
    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert saved["runs"]["backup_data@2026-08-30"]["acknowledged"]["reason"] == (
        "backup run manually 2026-09-04"
    )


def test_acknowledge_missed_rejects_an_unknown_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    state: dict[str, Any] = {"runs": {}}

    code = capture_scheduler.acknowledge_missed("nonexistent@2026-01-01", "why", state)

    assert code == 1
    assert not (tmp_path / "state.json").exists()


def test_acknowledge_missed_rejects_a_row_that_is_not_missed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    state: dict[str, Any] = {
        "runs": {"odds_tue_open@2026-09-01": {"status": "OK", "window_start": "x"}}
    }

    code = capture_scheduler.acknowledge_missed("odds_tue_open@2026-09-01", "why", state)

    assert code == 1
    assert "acknowledged" not in state["runs"]["odds_tue_open@2026-09-01"]


def test_health_report_stops_counting_an_acknowledged_missed_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", ())
    heartbeat_path.write_text(
        _heartbeat_payload(
            now,
            age=timedelta(seconds=1),
            code_sha256=capture_scheduler.compute_code_sha256(),
            schedule_digest=capture_scheduler.compute_schedule_digest(),
        ),
        encoding="utf-8",
    )
    state = {
        "runs": {
            "backup_data@2026-08-30": {
                "status": "MISSED",
                "window_start": "2026-08-30T22:00:00-04:00",
                "acknowledged": {
                    "reason": "backup run manually",
                    "at": "2026-09-04T17:00:00-04:00",
                },
            }
        }
    }

    report = capture_scheduler.build_health_report(now, state)
    rendered = capture_scheduler.render_health(report)

    assert len(report["missed"]) == 1
    assert len(report["missed_unacknowledged"]) == 0
    assert report["ok"] is True
    assert "MISSED (acknowledged: backup run manually)" in rendered


def test_daemon_is_running_true_when_heartbeat_fresh_and_pid_alive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)
    heartbeat_path.write_text(_heartbeat_payload(now, age=timedelta(seconds=5)), encoding="utf-8")
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "pid_is_alive", lambda pid: True)

    alive, pid = capture_scheduler.daemon_is_running(now)

    assert alive is True
    assert pid == 4242


def test_daemon_is_running_false_when_pid_is_dead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)
    heartbeat_path.write_text(_heartbeat_payload(now, age=timedelta(seconds=5)), encoding="utf-8")
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "pid_is_alive", lambda pid: False)

    alive, pid = capture_scheduler.daemon_is_running(now)

    assert alive is False
    assert pid == 4242


def test_daemon_is_running_false_when_heartbeat_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)
    heartbeat_path.write_text(_heartbeat_payload(now, age=timedelta(minutes=10)), encoding="utf-8")
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "pid_is_alive", lambda pid: True)

    alive, pid = capture_scheduler.daemon_is_running(now)

    assert alive is False
    assert pid == 4242  # still reported, so a caller can print it either way


def test_daemon_is_running_false_when_heartbeat_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", tmp_path / "absent.json")
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)

    alive, pid = capture_scheduler.daemon_is_running(now)

    assert alive is False
    assert pid is None


def test_is_running_cli_flag_exits_zero_and_prints_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    now = datetime.now(tz=ET)
    heartbeat_path.write_text(_heartbeat_payload(now, age=timedelta(seconds=1)), encoding="utf-8")
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(capture_scheduler, "pid_is_alive", lambda pid: True)

    code = capture_scheduler.main(["--is-running"])

    assert code == 0
    assert "4242" in capsys.readouterr().out


def test_is_running_cli_flag_exits_one_when_no_heartbeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")

    code = capture_scheduler.main(["--is-running"])

    assert code == 1
    assert "not running" in capsys.readouterr().out


def test_acknowledge_missed_cli_flag_writes_state_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "runs": {
                    "backup_data@2026-08-30": {
                        "status": "MISSED",
                        "window_start": "2026-08-30T22:00:00-04:00",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", state_path)

    code = capture_scheduler.main(
        [
            "--acknowledge-missed",
            "backup_data@2026-08-30",
            "--reason",
            "backup run manually 2026-09-04",
        ]
    )

    assert code == 0
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["runs"]["backup_data@2026-08-30"]["status"] == "MISSED"
    assert (
        saved["runs"]["backup_data@2026-08-30"]["acknowledged"]["reason"]
        == "backup run manually 2026-09-04"
    )


def test_acknowledge_missed_cli_flag_requires_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")

    code = capture_scheduler.main(["--acknowledge-missed", "backup_data@2026-08-30"])

    assert code == 2
