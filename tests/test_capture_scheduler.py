from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.capture_scheduler as capture_scheduler
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


def empty_state() -> dict[str, Any]:
    return {"runs": {}}


THURSDAY = datetime(2026, 8, 27, 17, 0, tzinfo=ET)


def test_window_that_closed_before_the_job_existed_is_not_missed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job(added_on="2026-08-27")
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    state = empty_state()

    capture_scheduler.sweep_missed(THURSDAY, state)

    assert state["runs"] == {}


def test_the_same_window_is_missed_without_the_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job()
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    state = empty_state()

    capture_scheduler.sweep_missed(THURSDAY, state)

    assert [record["status"] for record in state["runs"].values()] == ["MISSED"]


def test_a_window_on_the_day_the_job_was_added_still_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job(added_on="2026-08-23")
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    state = empty_state()

    capture_scheduler.sweep_missed(THURSDAY, state)

    assert [record["status"] for record in state["runs"].values()] == ["MISSED"]


def test_predates_job_is_inert_for_every_pre_existing_job() -> None:
    for job in capture_scheduler.SCHEDULE:
        if job.added_on:
            continue
        start = capture_scheduler.occurrence(job, THURSDAY)
        assert not capture_scheduler.predates_job(job, start)


def test_a_predating_window_is_never_due_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    job = make_job(added_on="2026-08-27", day="thu", at="16:00", grace_minutes=300)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    inside_window = datetime(2026, 8, 27, 17, 0, tzinfo=ET)
    assert capture_scheduler.due_jobs(inside_window, empty_state())

    later = make_job(added_on="2026-08-28", day="thu", at="16:00", grace_minutes=300)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (later,))
    assert not capture_scheduler.due_jobs(inside_window, empty_state())


def test_status_labels_a_predating_window_instead_of_calling_it_unrun(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    job = make_job(added_on="2026-08-27")
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))

    capture_scheduler.show_status(THURSDAY, empty_state())

    out = capsys.readouterr().out
    assert "window predates job" in out
    assert "not run" not in out


def test_the_real_backup_job_is_guarded_and_runs_after_the_weeks_last_capture() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    backup = schedule["backup_data"]

    assert backup.added_on == "2026-08-27"
    assert not backup.season_guarded
    sunday = datetime(2026, 8, 30, 23, 0, tzinfo=ET)
    latest_capture = max(
        capture_scheduler.occurrence(job, sunday)
        for job in capture_scheduler.SCHEDULE
        if not job.name.startswith(("backup_", "sync_captures_"))
        and capture_scheduler.occurrence(job, sunday).date() == sunday.date()
    )
    assert capture_scheduler.occurrence(backup, sunday) > latest_capture


def test_backup_job_finishes_well_inside_the_subprocess_timeout() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    command = schedule["backup_data"].command

    assert any(part.endswith("backup_data.py") for part in command)
    assert "--include-artifacts" in command
    assert "--verify-all" not in command


def test_a_missed_catch_up_job_runs_once_on_the_next_tick_and_shows_caught_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job(catch_up=True, command=["cmd.exe", "/c", "echo", "caught-up-ok"])
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    state = empty_state()

    capture_scheduler.sweep_missed(THURSDAY, state)

    key = f"{job.name}@{capture_scheduler.occurrence(job, THURSDAY).date().isoformat()}"
    record = state["runs"][key]
    assert record["status"] == "CAUGHT_UP"
    assert record["caught_up"] is True
    assert (tmp_path / "state.json").is_file()


def test_a_non_catch_up_job_still_shows_missed_and_does_not_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job(catch_up=False, command=["cmd.exe", "/c", "echo", "should-not-run"])
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    state = empty_state()

    capture_scheduler.sweep_missed(THURSDAY, state)

    key = f"{job.name}@{capture_scheduler.occurrence(job, THURSDAY).date().isoformat()}"
    assert state["runs"][key]["status"] == "MISSED"
    assert not (tmp_path / "state.json").exists()


def test_a_catch_up_job_never_runs_twice_for_one_occurrence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = make_job(catch_up=True, command=["cmd.exe", "/c", "echo", "once-only"])
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    state = empty_state()

    capture_scheduler.sweep_missed(THURSDAY, state)
    key = f"{job.name}@{capture_scheduler.occurrence(job, THURSDAY).date().isoformat()}"
    first_ran_at = state["runs"][key]["ran_at"]

    later = THURSDAY + timedelta(hours=6)
    capture_scheduler.sweep_missed(later, state)

    assert state["runs"][key]["ran_at"] == first_ran_at
    assert state["runs"][key]["status"] == "CAUGHT_UP"


def test_status_renders_caught_up_distinctly_from_ok_and_missed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    job = make_job(catch_up=True)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    start = capture_scheduler.occurrence(job, THURSDAY)
    key = f"{job.name}@{start.date().isoformat()}"
    state = {
        "runs": {
            key: {
                "status": "CAUGHT_UP",
                "window_start": start.isoformat(),
                "ran_at": "2026-08-27T12:00:00-04:00",
                "caught_up": True,
            }
        }
    }

    capture_scheduler.show_status(THURSDAY, state)

    out = capsys.readouterr().out
    assert "CAUGHT_UP" in out
    assert "not run" not in out


def test_player_arrests_tue_is_idempotent_catch_up_and_added_on_guarded() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    arrests = schedule["player_arrests_tue"]

    assert arrests.catch_up is True
    assert arrests.added_on == "2026-09-01"
    assert arrests.command[-1] == "ingest-player-arrests"


def test_player_arrests_tue_window_closes_before_the_tuesday_opener() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    arrests = schedule["player_arrests_tue"]
    opener = schedule["odds_tue_open"]
    tuesday = datetime(2026, 9, 1, 15, 0, tzinfo=ET)

    arrests_close = capture_scheduler.occurrence(arrests, tuesday) + timedelta(
        minutes=arrests.grace_minutes
    )
    opener_start = capture_scheduler.occurrence(opener, tuesday)
    assert arrests_close < opener_start


def test_grace_window_does_not_collide_with_the_next_job(monkeypatch: pytest.MonkeyPatch) -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    backup = schedule["backup_data"]
    sunday = datetime(2026, 8, 30, 23, 0, tzinfo=ET)
    close = capture_scheduler.occurrence(backup, sunday) + timedelta(minutes=backup.grace_minutes)

    monday_mnf = capture_scheduler.occurrence(schedule["odds_mon_mnf"], sunday + timedelta(days=2))
    assert close < monday_mnf


_INACTIVES_JOB_NAMES = (
    "inactives_sun_early",
    "inactives_sun_late",
    "inactives_thu_afternoon_early",
    "inactives_thu_afternoon_late",
    "inactives_thu_primetime",
    "inactives_wed_primetime",
    "inactives_sat_early",
    "inactives_sat_late",
)


def test_inactives_jobs_are_point_in_time_and_added_this_session() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    for name in _INACTIVES_JOB_NAMES:
        job = schedule[name]
        assert job.catch_up is False
        assert job.added_on == ("2026-09-09" if name == "inactives_wed_primetime" else "2026-09-01")
        assert job.season_guarded is True
        assert job.dedupe_dir == "data/players/inactives"
        assert job.dedupe_minutes == 60
        slot = name.removeprefix("inactives_")
        assert job.command[-2:] == ["--slot", slot]


def test_inactives_job_names_are_unique_across_the_whole_schedule() -> None:
    names = [job.name for job in capture_scheduler.SCHEDULE]
    assert len(names) == len(set(names))


def test_inactives_sun_early_closes_well_before_the_thirteen_hundred_slate() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    job = schedule["inactives_sun_early"]
    sunday = datetime(2026, 9, 13, 20, 0, tzinfo=ET)
    close = capture_scheduler.occurrence(job, sunday) + timedelta(minutes=job.grace_minutes)
    assert close.strftime("%H:%M") == "11:50"
    kickoff = datetime(2026, 9, 13, 13, 0, tzinfo=ET)
    assert close < kickoff


def test_inactives_sun_late_closes_before_the_sunday_pick_lock() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    job = schedule["inactives_sun_late"]
    sunday = datetime(2026, 9, 13, 20, 0, tzinfo=ET)
    close = capture_scheduler.occurrence(job, sunday) + timedelta(minutes=job.grace_minutes)
    sunday_lock = datetime(2026, 9, 13, 16, 0, tzinfo=ET)
    assert close < sunday_lock
    assert (sunday_lock - close) == timedelta(minutes=65)


def test_inactives_thu_cluster_has_three_non_colliding_occurrences() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    thursday = datetime(2026, 9, 10, 23, 0, tzinfo=ET)
    early = schedule["inactives_thu_afternoon_early"]
    late = schedule["inactives_thu_afternoon_late"]
    primetime = schedule["inactives_thu_primetime"]

    early_close = capture_scheduler.occurrence(early, thursday) + timedelta(
        minutes=early.grace_minutes
    )
    late_start = capture_scheduler.occurrence(late, thursday)
    late_close = capture_scheduler.occurrence(late, thursday) + timedelta(
        minutes=late.grace_minutes
    )
    primetime_start = capture_scheduler.occurrence(primetime, thursday)

    assert early_close < late_start
    assert late_close < primetime_start


def test_inactives_sat_cluster_has_two_non_colliding_occurrences() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    saturday = datetime(2026, 9, 12, 23, 0, tzinfo=ET)
    early = schedule["inactives_sat_early"]
    late = schedule["inactives_sat_late"]

    early_close = capture_scheduler.occurrence(early, saturday) + timedelta(
        minutes=early.grace_minutes
    )
    late_start = capture_scheduler.occurrence(late, saturday)
    assert early_close < late_start


def test_no_inactives_row_targets_snf_or_mnf() -> None:
    for job in capture_scheduler.SCHEDULE:
        if not job.name.startswith("inactives_"):
            continue
        assert job.day != "mon"
        if job.day == "sun":
            hour, _minute = (int(part) for part in job.at.split(":"))
            assert hour < 16


_INACTIVES_REFRESH_WINDOWS = {
    "refresh_thu_inactives_early": ("inactives_thu_afternoon_early", "11:55", 55),
    "refresh_thu_inactives_late": ("inactives_thu_afternoon_late", "15:25", 55),
    "refresh_thu_inactives_primetime": ("inactives_thu_primetime", "19:15", 50),
    "refresh_wed_inactives_primetime": ("inactives_wed_primetime", "19:15", 50),
    "refresh_sat_inactives_early": ("inactives_sat_early", "15:50", 60),
    "refresh_sat_inactives_late": ("inactives_sat_late", "19:15", 55),
    "refresh_sun_inactives_early": ("inactives_sun_early", "11:55", 55),
    "refresh_sun_inactives_late": ("inactives_sun_late", "15:00", 50),
}


def test_inactives_refreshes_begin_after_capture_and_stay_before_their_deadline() -> None:

    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    anchors = {
        "thu": datetime(2026, 11, 26, 23, 0, tzinfo=ET),
        "sat": datetime(2026, 12, 19, 23, 0, tzinfo=ET),
        "sun": datetime(2026, 9, 20, 23, 0, tzinfo=ET),
        "wed": datetime(2026, 9, 9, 23, 0, tzinfo=ET),
    }
    deadline_by_refresh = {
        "refresh_wed_inactives_primetime": datetime(2026, 9, 9, 20, 15, tzinfo=ET),
        "refresh_thu_inactives_early": datetime(2026, 11, 26, 13, 0, tzinfo=ET),
        "refresh_thu_inactives_late": datetime(2026, 11, 26, 16, 30, tzinfo=ET),
        "refresh_thu_inactives_primetime": datetime(2026, 11, 26, 20, 15, tzinfo=ET),
        "refresh_sat_inactives_early": datetime(2026, 12, 19, 17, 0, tzinfo=ET),
        "refresh_sat_inactives_late": datetime(2026, 12, 19, 20, 20, tzinfo=ET),
        "refresh_sun_inactives_early": datetime(2026, 9, 20, 13, 0, tzinfo=ET),
        "refresh_sun_inactives_late": datetime(2026, 9, 20, 16, 0, tzinfo=ET),
    }
    for refresh_name, (capture_name, target, grace) in _INACTIVES_REFRESH_WINDOWS.items():
        refresh = schedule[refresh_name]
        capture = schedule[capture_name]
        anchor = anchors[refresh.day]
        capture_close = capture_scheduler.occurrence(capture, anchor) + timedelta(
            minutes=capture.grace_minutes
        )
        refresh_start = capture_scheduler.occurrence(refresh, anchor)
        refresh_close = refresh_start + timedelta(minutes=refresh.grace_minutes)

        assert refresh.at == target
        assert refresh.grace_minutes == grace
        assert refresh_start == capture_close + timedelta(minutes=5)
        assert refresh_close == deadline_by_refresh[refresh_name] - timedelta(minutes=10)
        assert refresh.season_guarded is True
        assert refresh.catch_up is False
        assert refresh.added_on == (
            "2026-09-09" if refresh_name == "refresh_wed_inactives_primetime" else "2026-09-02"
        )
        assert refresh.command[-4:] == [
            "refresh-picks",
            "--record-decisions",
            "--note",
            refresh.command[-1],
        ]
        assert "--publish-card" not in refresh.command


def test_referee_assignments_wed_catch_up_and_added_on_guarded() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    job = schedule["referee_assignments_wed"]

    assert job.catch_up is True
    assert job.added_on == "2026-09-01"
    assert job.command[-1] == "--current"
    assert "capture_referee_assignments.py" in job.command[-2]
    assert job.dedupe_dir == "data/players/referee_assignments"
    assert job.dedupe_minutes == 240


def test_referee_assignments_wed_target_clears_the_latest_measured_publish_time() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    job = schedule["referee_assignments_wed"]

    assert job.day == "wed"
    target_hour, target_minute = (int(part) for part in job.at.split(":"))
    latest_measured_publish_minutes = 12 * 60 + 42
    target_minutes = target_hour * 60 + target_minute
    assert target_minutes - latest_measured_publish_minutes >= 120


def test_pfr_transaction_jobs_are_resume_safe_and_well_spaced() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    wed = schedule["pfr_transactions_wed"]
    sat = schedule["pfr_transactions_sat"]

    for job in (wed, sat):
        assert job.enabled is True
        assert job.season_guarded is True
        assert job.catch_up is True
        assert job.added_on == "2026-09-03"
        assert any("ingest_transaction_news.py" in part for part in job.command)
        assert job.command[-1] == "--fresh-snapshot"
        assert job.dedupe_dir == "data/raw/pfr_transactions"
        assert 0 < job.dedupe_minutes < 3 * 24 * 60
    assert (wed.day, wed.at) == ("wed", "07:00")
    assert (sat.day, sat.at) == ("sat", "07:00")


def test_pfr_transactions_argv_resolves_and_dry_run_exits_0_with_no_network() -> None:

    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    for job_name in ("pfr_transactions_wed", "pfr_transactions_sat"):
        job = schedule[job_name]
        script_path = Path(job.command[-2])
        assert script_path.is_file(), f"{job_name}: script not found at {script_path}"

        result = subprocess.run(
            [*job.command, "--dry-run"],
            cwd=capture_scheduler.REPO,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"{job_name} --dry-run exited {result.returncode}: {result.stderr}"
        )
        payload = json.loads(result.stdout)
        assert payload["dry_run"] is True
        assert payload["network_requests_made"] == 0
        assert payload["mode"] == "fresh_snapshot"


def test_odds_halves_jobs_ride_their_paired_bulk_capture_window() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    tue_halves = schedule["odds_tue_open_halves"]
    tue_open = schedule["odds_tue_open"]
    sat_halves = schedule["odds_sat_halves"]
    sat = schedule["odds_sat"]

    assert (tue_halves.day, tue_halves.at, tue_halves.grace_minutes) == (
        tue_open.day,
        tue_open.at,
        tue_open.grace_minutes,
    )
    assert tue_halves.requires == ("odds_tue_open",)
    assert (sat_halves.day, sat_halves.at, sat_halves.grace_minutes) == (
        sat.day,
        sat.at,
        sat.grace_minutes,
    )
    assert sat_halves.requires == ("odds_sat",)

    for job in (tue_halves, sat_halves):
        assert job.enabled is True
        assert job.season_guarded is False
        assert job.added_on == "2026-09-05"
        assert job.dedupe_dir == ""
        assert job.dedupe_minutes == 0
        assert job.command[-1] == "odds-ingest-halves"


def test_odds_halves_jobs_wait_for_their_paired_bulk_capture_to_succeed() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    job = schedule["odds_tue_open_halves"]
    tuesday = datetime(2026, 9, 8, 9, 30, tzinfo=ET)
    start = capture_scheduler.occurrence(job, tuesday)
    date_key = f"odds_tue_open@{start.date().isoformat()}"

    not_yet_run = {"runs": {}}
    assert capture_scheduler.prerequisites_satisfied(job, start, not_yet_run) is False

    failed = {"runs": {date_key: {"status": "FAIL(1)"}}}
    assert capture_scheduler.prerequisites_satisfied(job, start, failed) is False

    succeeded = {"runs": {date_key: {"status": "OK"}}}
    assert capture_scheduler.prerequisites_satisfied(job, start, succeeded) is True

    already_captured = {"runs": {date_key: {"status": "ALREADY-CAPTURED"}}}
    assert capture_scheduler.prerequisites_satisfied(job, start, already_captured) is True


def test_odds_sat_halves_waits_for_odds_sat_to_succeed() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    job = schedule["odds_sat_halves"]
    saturday = datetime(2026, 9, 12, 12, 30, tzinfo=ET)
    start = capture_scheduler.occurrence(job, saturday)
    date_key = f"odds_sat@{start.date().isoformat()}"

    not_yet_run = {"runs": {}}
    assert capture_scheduler.prerequisites_satisfied(job, start, not_yet_run) is False

    succeeded = {"runs": {date_key: {"status": "OK"}}}
    assert capture_scheduler.prerequisites_satisfied(job, start, succeeded) is True


def _nfl_ats_argv(job: Job) -> list[str] | None:
    command = list(job.command)
    if "nfl-ats" not in command:
        return None
    return command[command.index("nfl-ats") + 1 :]


def test_every_scheduled_nfl_ats_command_parses_against_the_real_parser() -> None:
    from nfl_ats.cli import build_parser

    parser = build_parser()
    checked: list[str] = []
    for job in capture_scheduler.SCHEDULE:
        argv = _nfl_ats_argv(job)
        if argv is None:
            continue
        try:
            parser.parse_args(argv)
        except SystemExit as error:
            pytest.fail(f"{job.name}: nfl-ats {' '.join(argv)} does not parse ({error})")
        checked.append(job.name)
    assert {"refresh_thu", "refresh_sat", "refresh_sun"} <= set(checked)


def test_refresh_jobs_pass_no_season_or_week_and_the_parser_defaults_them() -> None:
    from nfl_ats.cli import build_parser

    parser = build_parser()
    for job in capture_scheduler.SCHEDULE:
        if not job.name.startswith("refresh_") or job.name.startswith("refresh_trigger"):
            continue
        argv = _nfl_ats_argv(job)
        assert argv is not None and argv[0] == "refresh-picks", job.name
        assert "--season" not in argv and "--week" not in argv, job.name
        args = parser.parse_args(argv)
        assert args.season is None and args.week is None, job.name


def test_failure_detail_keeps_the_end_of_stderr_not_the_start() -> None:
    banners = "".join(f"weekly-run step {n} something ...\n" for n in range(2, 9))
    stderr = banners + "Traceback (most recent call last):\n  ...\nValueError: the real reason\n"

    detail = capture_scheduler.failure_detail(stderr, "last stdout line")

    assert detail.endswith("ValueError: the real reason")
    assert len(detail) <= 300
    assert capture_scheduler.failure_detail("", "last stdout line") == "last stdout line"
    assert capture_scheduler.failure_detail(None, "") == ""


def test_wednesday_opener_pair_runs_between_the_tuesday_lock_and_the_wednesday_kickoff() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    capture = schedule["odds_wed_opener"]
    refresh = schedule["refresh_wed"]
    wednesday = datetime(2026, 9, 9, 12, 0, tzinfo=ET)
    kickoff = datetime(2026, 9, 9, 20, 20, tzinfo=ET)

    capture_start = capture_scheduler.occurrence(capture, wednesday)
    refresh_start = capture_scheduler.occurrence(refresh, wednesday)
    refresh_close = refresh_start + timedelta(minutes=refresh.grace_minutes)

    assert capture.day == refresh.day == "wed"
    assert capture_start < refresh_start < refresh_close < kickoff
    assert refresh_close <= kickoff - timedelta(minutes=30)
    assert capture.added_on == refresh.added_on == "2026-09-07"
    assert capture.dedupe_dir == "data/market/raw"
    assert refresh.season_guarded is True
    assert refresh.command[-4:] == [
        "refresh-picks",
        "--record-decisions",
        "--note",
        "wednesday_opener",
    ]
    assert "--publish-card" not in refresh.command


def _isolate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(capture_scheduler, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(capture_scheduler, "LOG_PATH", tmp_path / "log.txt")


def test_run_job_executes_the_exact_argv_and_records_a_manual_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _isolate_state(monkeypatch, tmp_path)
    job = make_job(
        name="demo_ok",
        command=[sys.executable, "-c", "print('captured 3 rows')"],
        season_guarded=True,
    )
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    state = empty_state()
    assert capture_scheduler.has_ever_executed(state, "demo_ok") is False

    assert capture_scheduler.run_job_manually(job, state) == 0

    assert capture_scheduler.has_ever_executed(state, "demo_ok") is True
    health = state["job_health"]["demo_ok"]
    assert health["last_manual_status"] == "OK"
    assert health["last_manual_run_at"]
    assert state["runs"] == {}
    out = capsys.readouterr().out
    assert "MANUAL-RUN demo_ok: OK" in out and "captured 3 rows" in out
    log_text = (tmp_path / "log.txt").read_text(encoding="utf-8")
    assert "MANUAL-RUN demo_ok:" in log_text and "MANUAL-RUN OK demo_ok" in log_text


def test_run_job_failure_keeps_the_end_of_stderr_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_state(monkeypatch, tmp_path)
    script = "import sys; [print(f'step {n} ...', file=sys.stderr) for n in range(40)]; " + (
        "print('ValueError: the real reason', file=sys.stderr); sys.exit(2)"
    )
    job = make_job(name="demo_fail", command=[sys.executable, "-c", script])
    state = empty_state()

    assert capture_scheduler.run_job_manually(job, state) == 1

    health = state["job_health"]["demo_fail"]
    assert health["last_manual_status"] == "FAIL(2)"
    assert health["last_error"].endswith("ValueError: the real reason")


def test_dry_run_strips_only_the_recording_flags() -> None:
    argv = [
        "uv",
        "run",
        "--no-sync",
        "nfl-ats",
        "refresh-picks",
        "--record-decisions",
        "--note",
        "thursday_afternoon",
        "--publish-card",
    ]
    assert capture_scheduler.dry_command(argv) == [
        "uv",
        "run",
        "--no-sync",
        "nfl-ats",
        "refresh-picks",
        "--note",
        "thursday_afternoon",
    ]
    assert capture_scheduler.dry_command(["x.ps1", "--current"]) == ["x.ps1", "--current"]


def test_dry_manual_run_is_labelled_dry_in_state_and_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_state(monkeypatch, tmp_path)
    marker = tmp_path / "wrote.txt"
    script = (
        "import sys, pathlib; "
        f"pathlib.Path(r'{marker}').write_text('x') if '--record-decisions' in sys.argv else None"
    )
    job = make_job(name="demo_dry", command=[sys.executable, "-c", script, "--record-decisions"])
    state = empty_state()

    assert capture_scheduler.run_job_manually(job, state, dry=True) == 0

    assert not marker.exists()
    assert state["job_health"]["demo_dry"]["last_manual_status"] == "OK (dry)"
    assert "MANUAL-DRY-RUN demo_dry:" in (tmp_path / "log.txt").read_text(encoding="utf-8")


def test_status_marks_enabled_jobs_that_have_never_executed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _isolate_state(monkeypatch, tmp_path)
    fresh = make_job(name="never_ran", added_on="2026-09-07")
    exercised = make_job(name="ran_by_hand", added_on="2026-09-07")
    disabled = make_job(name="off", enabled=False)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (fresh, exercised, disabled))
    state = empty_state()
    state["job_health"] = {"ran_by_hand": {"last_manual_run_at": "2026-09-07T09:00:00-04:00"}}
    state["runs"]["never_ran@2026-09-06"] = {"status": "MISSED"}
    state["runs"]["never_ran@2026-08-30"] = {"status": "ALREADY-CAPTURED"}
    assert capture_scheduler.has_ever_executed(state, "never_ran") is False
    assert capture_scheduler.has_ever_executed(
        {"runs": {"legacy@2026-08-30": {"status": "OK"}}}, "legacy"
    )

    capture_scheduler.show_status(datetime(2026, 9, 7, 9, 0, tzinfo=ET), state)

    lines = capsys.readouterr().out.splitlines()
    row = {
        line.split()[0]: line
        for line in lines
        if line and line.split()[0] in {"never_ran", "ran_by_hand", "off"}
    }
    assert "NEVER RUN (exercise: --run-job never_ran)" in row["never_ran"]
    assert "NEVER RUN" not in row["ran_by_hand"]
    assert "NEVER RUN" not in row["off"]


def test_main_run_job_rejects_unknown_names_and_bare_dry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_state(monkeypatch, tmp_path)
    assert capture_scheduler.main(["--run-job", "no_such_job"]) == 2
    assert capture_scheduler.main(["--dry"]) == 2


def test_pid_is_alive_is_false_for_an_exited_process_whose_handle_is_still_open() -> None:
    import subprocess

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=60)
    try:
        assert capture_scheduler.pid_is_alive(child.pid) is False
        assert capture_scheduler.pid_is_alive(__import__("os").getpid()) is True
    finally:
        del child


def test_tuesday_opener_is_captured_after_the_pool_locks_at_noon() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    opener = schedule["odds_tue_open"]
    halves = schedule["odds_tue_open_halves"]
    lock = schedule["weekly_lock"]
    lineups = schedule["lineups_tue"]
    tuesday = datetime(2026, 9, 8, 15, 0, tzinfo=ET)
    pool_lock = datetime(2026, 9, 8, 12, 0, tzinfo=ET)

    opener_start = capture_scheduler.occurrence(opener, tuesday)
    lock_start = capture_scheduler.occurrence(lock, tuesday)
    lineups_start = capture_scheduler.occurrence(lineups, tuesday)

    assert opener_start == datetime(2026, 9, 8, 12, 5, tzinfo=ET)
    assert opener_start > pool_lock
    assert lock_start == datetime(2026, 9, 8, 12, 20, tzinfo=ET)
    assert lock.requires == ("odds_tue_open", "splash_board_tue")
    assert (halves.day, halves.at) == ("tue", "12:05")
    assert lineups_start >= lock_start + timedelta(minutes=lock.grace_minutes)
    assert "odds_tue_noon" not in schedule
    for job in capture_scheduler.SCHEDULE:
        if job.day == "tue" and "odds" in job.name:
            assert capture_scheduler.occurrence(job, tuesday) > pool_lock, job.name


def test_retry_is_opt_in_and_every_pre_existing_job_defaults_off() -> None:
    opted_in = {"player_arrests_tue", "splash_board_tue"}
    for job in capture_scheduler.SCHEDULE:
        if job.name in opted_in:
            continue
        assert job.retry_backoff_minutes == 0
        assert job.max_retries == 0


def test_a_failed_occurrence_without_retry_policy_is_never_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job(day="tue", at="07:00", grace_minutes=90)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    start = datetime(2026, 9, 8, 7, 0, tzinfo=ET)
    key = f"{job.name}@{start.date().isoformat()}"
    state = {
        "runs": {
            key: {
                "status": "FAIL(2)",
                "window_start": start.isoformat(),
                "ran_at": start.isoformat(),
            }
        }
    }

    assert capture_scheduler.due_jobs(start + timedelta(hours=1), state) == []


def test_a_failed_occurrence_is_retried_once_backoff_elapses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job(day="tue", at="07:00", grace_minutes=90, retry_backoff_minutes=15, max_retries=2)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    first_attempt = datetime(2026, 9, 8, 7, 0, tzinfo=ET)
    key = f"{job.name}@{first_attempt.date().isoformat()}"
    state = {
        "runs": {
            key: {
                "status": "FAIL(2)",
                "window_start": first_attempt.isoformat(),
                "ran_at": first_attempt.isoformat(),
            }
        }
    }

    too_soon = first_attempt + timedelta(minutes=10)
    assert capture_scheduler.due_jobs(too_soon, state) == []

    ready = first_attempt + timedelta(minutes=15)
    due = capture_scheduler.due_jobs(ready, state)
    assert [j.name for j, _ in due] == [job.name]


def test_retry_stops_after_max_retries_even_inside_the_grace_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job(day="tue", at="07:00", grace_minutes=90, retry_backoff_minutes=15, max_retries=2)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    first_attempt = datetime(2026, 9, 8, 7, 0, tzinfo=ET)
    key = f"{job.name}@{first_attempt.date().isoformat()}"
    state = {
        "runs": {
            key: {
                "status": "FAIL(2)",
                "window_start": first_attempt.isoformat(),
                "ran_at": first_attempt.isoformat(),
                "retries": 2,
            }
        }
    }

    later = first_attempt + timedelta(minutes=40)
    assert capture_scheduler.due_jobs(later, state) == []


def test_a_successful_retry_is_never_retried_again(monkeypatch: pytest.MonkeyPatch) -> None:
    job = make_job(day="tue", at="07:00", grace_minutes=90, retry_backoff_minutes=15, max_retries=2)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    first_attempt = datetime(2026, 9, 8, 7, 0, tzinfo=ET)
    key = f"{job.name}@{first_attempt.date().isoformat()}"
    state = {
        "runs": {
            key: {
                "status": "OK",
                "window_start": first_attempt.isoformat(),
                "ran_at": first_attempt.isoformat(),
                "retries": 1,
            }
        }
    }

    assert capture_scheduler.due_jobs(first_attempt + timedelta(minutes=20), state) == []


def test_run_job_labels_a_retry_and_increments_the_counter_until_it_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_state(monkeypatch, tmp_path)
    marker = tmp_path / "attempts.txt"
    script = (
        "import pathlib, sys; "
        f"p = pathlib.Path(r'{marker}'); "
        "n = int(p.read_text()) if p.exists() else 0; "
        "p.write_text(str(n + 1)); "
        "sys.exit(1 if n < 1 else 0)"
    )
    job = make_job(
        name="demo_retry",
        command=[sys.executable, "-c", script],
        day="tue",
        at="07:00",
        grace_minutes=90,
        retry_backoff_minutes=15,
        max_retries=2,
    )
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    state = empty_state()
    start = datetime(2026, 9, 8, 7, 0, tzinfo=ET)
    key = f"{job.name}@{start.date().isoformat()}"

    capture_scheduler.run_job(job, start, state)
    assert state["runs"][key]["status"] == "FAIL(1)"
    assert "retries" not in state["runs"][key]

    capture_scheduler.run_job(job, start, state)
    assert state["runs"][key]["status"] == "OK"
    assert state["runs"][key]["retries"] == 1

    log_text = (tmp_path / "log.txt").read_text(encoding="utf-8")
    assert "RUN demo_retry" in log_text
    assert "RETRY demo_retry" in log_text


def test_status_shows_the_retry_count_on_an_eventually_ok_row(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    job = make_job(retry_backoff_minutes=15, max_retries=2)
    monkeypatch.setattr(capture_scheduler, "SCHEDULE", (job,))
    start = capture_scheduler.occurrence(job, THURSDAY)
    key = f"{job.name}@{start.date().isoformat()}"
    state = {
        "runs": {
            key: {
                "status": "OK",
                "window_start": start.isoformat(),
                "ran_at": THURSDAY.isoformat(),
                "retries": 1,
            }
        }
    }

    capture_scheduler.show_status(THURSDAY, state)

    out = capsys.readouterr().out
    assert "OK" in out
    assert "after 1 retry" in out


def test_player_arrests_tue_retry_policy_fits_inside_its_own_grace_window() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    arrests = schedule["player_arrests_tue"]

    assert arrests.retry_backoff_minutes == 15
    assert arrests.max_retries == 2
    last_retry_offset = arrests.retry_backoff_minutes * arrests.max_retries
    assert last_retry_offset < arrests.grace_minutes


def test_player_arrests_tue_diagnoses_winerror_10013() -> None:
    import scripts.ingest_player_arrests as ingest_player_arrests

    class _FakeReason:
        winerror = 10013

    class _FakeURLError(Exception):
        reason = _FakeReason()

    message = ingest_player_arrests._diagnose(_FakeURLError())
    assert "WinError 10013" in message
    assert "transient" in message

    assert ingest_player_arrests._diagnose(TimeoutError("timed out")) == ""


_INJURY_PIPELINE_DAYS = ("wed", "thu", "fri", "sat", "sun")


def test_injury_pipeline_jobs_exist_for_every_late_week_day() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    for day in _INJURY_PIPELINE_DAYS:
        assert f"nflverse_injuries_{day}" in schedule
        assert f"player_snapshot_{day}" in schedule


def test_injury_pipeline_jobs_are_enabled_catch_up_and_dated_today() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    for day in _INJURY_PIPELINE_DAYS:
        for prefix in ("nflverse_injuries_", "player_snapshot_"):
            job = schedule[f"{prefix}{day}"]
            assert job.enabled is True
            assert job.catch_up is True, job.name
            assert job.added_on == "2026-09-08", job.name


def test_nflverse_injuries_job_command_and_dedupe() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    job = schedule["nflverse_injuries_wed"]
    assert job.command[-1].endswith("nflverse_injuries_ingest.py")
    assert job.dedupe_dir == "data/raw/nflverse_injuries"
    assert job.dedupe_minutes > 0


def test_player_snapshot_argv_matches_the_known_good_recipe() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    job = schedule["player_snapshot_wed"]
    argv = _nfl_ats_argv(job)
    assert argv is not None
    assert argv[0] == "player-ingest"
    assert argv[1:] == [
        "--injury-start-season",
        "2009",
        "--injury-end-season",
        "2026",
        "--roster-start-season",
        "2009",
        "--roster-end-season",
        "2026",
        "--snap-start-season",
        "2013",
        "--snap-end-season",
        "2025",
        "--include-postseason",
        "--timestamp-fallback",
        "week_proxy",
    ]
    assert job.dedupe_dir == "data/players/raw"


def test_player_snapshot_snap_end_season_stays_below_the_2026_404() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    for day in _INJURY_PIPELINE_DAYS:
        argv = _nfl_ats_argv(schedule[f"player_snapshot_{day}"])
        assert argv is not None
        snap_end = argv[argv.index("--snap-end-season") + 1]
        assert int(snap_end) <= 2025


def test_injury_pipeline_jobs_run_before_every_same_day_consumer() -> None:
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    anchor = datetime(2026, 9, 8, 12, 0, tzinfo=ET)
    consumers_by_day = {
        "wed": ("lineups_wed", "refresh_wed"),
        "thu": ("lineups_thu", "refresh_thu"),
        "fri": ("lineups_fri",),
        "sat": ("lineups_sat", "refresh_sat"),
        "sun": ("lineups_sun", "refresh_sun"),
    }
    for day, consumer_names in consumers_by_day.items():
        injuries_start = capture_scheduler.occurrence(schedule[f"nflverse_injuries_{day}"], anchor)
        snapshot_start = capture_scheduler.occurrence(schedule[f"player_snapshot_{day}"], anchor)
        for consumer_name in consumer_names:
            consumer_start = capture_scheduler.occurrence(schedule[consumer_name], anchor)
            assert injuries_start < consumer_start, (day, consumer_name)
            assert snapshot_start < consumer_start, (day, consumer_name)
