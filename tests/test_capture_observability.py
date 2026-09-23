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

    name = when.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


NOW = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)


def test_single_weekly_job_gets_a_week_plus_grace_budget() -> None:
    job = make_job(day="tue", at="09:00", grace_minutes=180)
    budget = cf.derive_budget_minutes([job])
    assert budget == 7 * 24 * 60 + 180


def test_newest_snapshot_instant_reads_the_newest_directory_name(tmp_path: Path) -> None:
    older = NOW - timedelta(days=2)
    newer = NOW - timedelta(hours=1)
    stamp_dir(tmp_path, older)
    stamp_dir(tmp_path, newer)
    (tmp_path / "not_a_snapshot").mkdir()

    found = cf.newest_snapshot_instant(tmp_path)

    assert found == newer.replace(microsecond=0)


def test_newest_snapshot_instant_ignores_filesystem_mtime(tmp_path: Path) -> None:

    newer_named = stamp_dir(tmp_path, NOW - timedelta(days=1))
    older_named = stamp_dir(tmp_path, NOW - timedelta(days=5))
    old_time = (NOW - timedelta(days=30)).timestamp()
    new_time = NOW.timestamp()
    os.utime(newer_named, (old_time, old_time))
    os.utime(older_named, (new_time, new_time))

    found = cf.newest_snapshot_instant(tmp_path)

    assert found == (NOW - timedelta(days=1)).replace(microsecond=0)


def test_newest_json_field_instant_reads_the_compact_utc_stamp(tmp_path: Path) -> None:
    path = tmp_path / "lineups.json"
    path.write_text(json.dumps({"generated_at": "20260910T113000Z"}), encoding="utf-8")

    found = cf.newest_json_field_instant(path, "generated_at")

    assert found == datetime(2026, 9, 10, 11, 30, 0, tzinfo=UTC)


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


def test_read_heartbeat_malformed_file_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "heartbeat.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", path)
    assert capture_scheduler.read_heartbeat() is None


def _heartbeat_payload(
    now: datetime,
    *,
    age: timedelta,
    code_sha256: str | None = None,
    schedule_digest: str | None = None,
) -> str:

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
    monkeypatch.setattr(capture_scheduler, "REPO", tmp_path)

    report = capture_scheduler.build_health_report(now, {"runs": {}})

    assert report["ok"] is False
    assert "OVERALL: FAIL" in capture_scheduler.render_health(report)


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


def test_health_report_flags_stale_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    heartbeat_path = tmp_path / "heartbeat.json"
    now = datetime(2026, 9, 10, 8, 5, tzinfo=ET)
    monkeypatch.setattr(capture_scheduler, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", ())
    heartbeat_path.write_text(
        _heartbeat_payload(
            now,
            age=timedelta(seconds=1),
            code_sha256="0" * 64,
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
