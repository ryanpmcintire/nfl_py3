"""The one-click refresh (scripts/refresh_now.py): plan, guard, argv parity."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

from nfl_ats.cli import build_parser

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.capture_scheduler as capture_scheduler  # noqa: E402
import scripts.refresh_now as refresh_now  # noqa: E402
from scripts.capture_scheduler import ET  # noqa: E402


def _by_name(now: datetime) -> dict[str, refresh_now.Step]:
    return {step.name: step for step in refresh_now.plan(now)}


def test_plan_runs_the_schedulers_own_commands_in_order() -> None:
    steps = refresh_now.plan(datetime(2026, 9, 10, 15, 0, tzinfo=ET))
    assert [step.name for step in steps] == ["spreads", "lineups and injuries", "picks", "board"]
    assert all(step.skip_reason is None for step in steps)
    schedule = {job.name: job for job in capture_scheduler.SCHEDULE}
    by_name = {step.name: step for step in steps}
    # Spreads: the identical capture argv the Tuesday opener job runs.
    assert list(by_name["spreads"].argv) == schedule["odds_tue_open"].command
    # Lineups + injuries: the identical argv the daily lineups jobs run
    # (depth charts, then weekly-run --refresh-player-data).
    assert list(by_name["lineups and injuries"].argv) == schedule["lineups_tue"].command
    # Picks: the served late-week rule, recorded and labelled on the card.
    picks = list(by_name["picks"].argv)
    assert picks[-5:] == [
        "refresh-picks",
        "--record-decisions",
        "--publish-card",
        "--note",
        refresh_now.MANUAL_NOTE,
    ]
    assert list(by_name["board"].argv)[-1] == "publish-board"


def test_every_nfl_ats_argv_parses_against_the_real_parser() -> None:
    parser = build_parser()
    for step in refresh_now.plan(datetime(2026, 9, 10, 15, 0, tzinfo=ET)):
        argv = list(step.argv)
        if "nfl-ats" not in argv:
            continue
        parser.parse_args(argv[argv.index("nfl-ats") + 1 :])


@pytest.mark.parametrize(
    ("now", "skipped"),
    [
        (datetime(2026, 9, 8, 8, 30, tzinfo=ET), True),  # Tuesday before the opener
        (datetime(2026, 9, 8, 12, 4, tzinfo=ET), True),  # pool locked, capture not yet
        (datetime(2026, 9, 8, 12, 5, tzinfo=ET), False),  # opener window open
        (datetime(2026, 9, 8, 12, 30, tzinfo=ET), False),
        (datetime(2026, 9, 7, 8, 30, tzinfo=ET), False),  # Monday morning
        (datetime(2026, 9, 7, 21, 0, tzinfo=ET), True),  # Monday 21:00 ET is Tuesday UTC
        (datetime(2026, 9, 8, 20, 30, tzinfo=ET), False),  # Tuesday evening (after the opener)
        (datetime(2026, 9, 9, 8, 30, tzinfo=ET), False),  # Wednesday
    ],
)
def test_spreads_capture_is_skipped_on_tuesday_before_the_opener(
    now: datetime, skipped: bool
) -> None:
    """A capture before 12:05 ET on a Tuesday would BE the week's opener
    (``tuesday_opener_quotes`` takes the earliest Tuesday quote per book)."""
    spreads = _by_name(now)["spreads"]
    assert (spreads.skip_reason is not None) is skipped
    others = [step for step in refresh_now.plan(now) if step.name != "spreads"]
    assert all(step.skip_reason is None for step in others)


def test_dry_run_prints_every_command_and_runs_nothing(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry run must not spawn anything")

    monkeypatch.setattr(refresh_now.subprocess, "run", explode)
    code = refresh_now.run(refresh_now.plan(datetime(2026, 9, 8, 8, 30, tzinfo=ET)), dry=True)
    out = capsys.readouterr().out
    assert code == 0
    assert "[skip] spreads" in out
    assert "refresh-picks" in out
    assert "publish-board" in out


def test_failures_are_reported_and_do_not_stop_later_steps(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class Completed:
        def __init__(self, code: int) -> None:
            self.returncode = code

    def fake_run(argv: list[str], **_kwargs: object) -> Completed:
        calls.append(argv[-1])
        return Completed(2 if argv[-1].endswith("odds_capture.ps1") else 0)

    monkeypatch.setattr(refresh_now.subprocess, "run", fake_run)
    code = refresh_now.run(refresh_now.plan(datetime(2026, 9, 10, 15, 0, tzinfo=ET)))
    out = capsys.readouterr().out
    assert code == 1
    assert len(calls) == 4
    assert "[FAIL] spreads" in out
    assert "[ ok ] board" in out
    assert "1 step(s) failed" in out


def test_cmd_wrapper_calls_the_script_and_waits() -> None:
    text = (REPO / "scripts" / "refresh_now.cmd").read_text(encoding="utf-8")
    assert "scripts\\refresh_now.py" in text
    assert "pause" in text
    assert "--no-sync" in text
