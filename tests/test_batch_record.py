from __future__ import annotations

import json
import re
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.batch_record as batch_record


def make_task(name: str = "batch_record_demo_signal", **overrides: Any) -> dict[str, Any]:
    task: dict[str, Any] = {
        "name": name,
        "description": "demo signal",
        "source": "docs/demo.md",
        "effect": -0.12,
        "effect_units": "accuracy_points",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": 2009,
        "season_end": 2025,
        "interval_low": -0.55,
        "interval_high": 0.31,
        "probability_positive": 0.62,
        "sample_games": 1200,
        "sample_blocks": 17,
    }
    task.update(overrides)
    return task


def write_payload(tmp_path: Path, payload: Any, filename: str = "task.json") -> Path:
    path = tmp_path / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def enqueue(tmp_path: Path, payload: Any, queue_dir: Path) -> int:
    file_path = write_payload(tmp_path, payload)
    return batch_record.main(["enqueue", "--file", str(file_path), "--queue-dir", str(queue_dir)])


@pytest.fixture()
def queue_dir(tmp_path: Path) -> Path:
    return tmp_path / "queue"


def fake_run(
    results: dict[str, tuple[int, str]] | None = None,
    processed: list[str] | None = None,
    guard: threading.Lock | None = None,
    delay: float = 0.0,
) -> Any:
    def _run(cmd: list[str], **kwargs: Any) -> types.SimpleNamespace:
        name = cmd[cmd.index("--name") + 1]
        if delay:
            time.sleep(delay)
        if processed is not None:
            if guard is not None:
                with guard:
                    processed.append(name)
            else:
                processed.append(name)
        returncode, stderr = (results or {}).get(name, (0, ""))
        return types.SimpleNamespace(returncode=returncode, stdout="recorded ok", stderr=stderr)

    return _run


def test_enqueue_accepts_valid_object_and_prints_queue_id(
    tmp_path: Path, queue_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert enqueue(tmp_path, make_task(), queue_dir) == 0
    captured = capsys.readouterr()
    queued = batch_record.queued_files(queue_dir)
    assert len(queued) == 1
    assert re.fullmatch(r"\d{2}-\d{8}T\d{12}Z-[0-9a-f]{8}", queued[0].stem)
    assert f"enqueued {queued[0].stem}" in captured.out
    stored = json.loads(queued[0].read_text(encoding="utf-8"))
    assert stored["name"] == make_task()["name"]


def test_enqueue_rejects_unknown_field_with_actionable_error(
    tmp_path: Path, queue_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = enqueue(tmp_path, make_task(effect_unit="accuracy_points"), queue_dir)
    captured = capsys.readouterr()
    assert code == 2
    assert "unknown field(s)" in captured.err
    assert "'effect_unit'" in captured.err
    assert "allowed fields" in captured.err
    assert batch_record.queued_files(queue_dir) == []


def test_enqueue_rejects_bad_effect_units(
    tmp_path: Path, queue_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = enqueue(tmp_path, make_task(effect_units="touchdowns"), queue_dir)
    captured = capsys.readouterr()
    assert code == 2
    assert "'effect_units'" in captured.err
    assert batch_record.queued_files(queue_dir) == []


def test_enqueue_rejects_bad_classification(tmp_path: Path, queue_dir: Path) -> None:
    code = enqueue(tmp_path, make_task(classification="contains_zero"), queue_dir)
    assert code == 2
    assert batch_record.queued_files(queue_dir) == []


def test_enqueue_rejects_missing_required_field(
    tmp_path: Path, queue_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    task = make_task()
    del task["probability_positive"]
    code = enqueue(tmp_path, task, queue_dir)
    captured = capsys.readouterr()
    assert code == 2
    assert "missing required field(s)" in captured.err
    assert "'probability_positive'" in captured.err


def test_enqueue_rejects_terminal_classification_without_closing_ground(
    tmp_path: Path, queue_dir: Path
) -> None:
    code = enqueue(tmp_path, make_task(classification="refuted_mechanism"), queue_dir)
    assert code == 2
    assert batch_record.queued_files(queue_dir) == []


def test_enqueue_tolerates_utf8_bom_payload(tmp_path: Path, queue_dir: Path) -> None:
    path = tmp_path / "bom_task.json"
    path.write_text(json.dumps(make_task("sig_bom")), encoding="utf-8-sig")
    code = batch_record.main(["enqueue", "--file", str(path), "--queue-dir", str(queue_dir)])
    assert code == 0
    assert len(batch_record.queued_files(queue_dir)) == 1


def test_enqueue_accepts_list_payload(tmp_path: Path, queue_dir: Path) -> None:
    payload = [make_task("sig_a"), make_task("sig_b")]
    assert enqueue(tmp_path, payload, queue_dir) == 0
    assert len(batch_record.queued_files(queue_dir)) == 2


def test_build_command_matches_cli_flags() -> None:
    task = make_task(reliability=0.41, closing_ground=None, replace=True)
    command = batch_record.build_command(task)
    joined = " ".join(command)
    assert command[:5] == [sys.executable, "-m", "nfl_ats.cli", "weak-signals", "record"]
    assert "--effect-units accuracy_points" in joined
    assert "--classification unresolved_below_power" in joined
    assert "--interval-low -0.55" in joined
    assert "--probability-positive 0.62" in joined
    assert "--reliability 0.41" in joined
    assert command[-1] == "--replace"
    assert "--closing-ground" not in joined


def test_drain_dry_run_is_default_and_runs_no_subprocess(
    tmp_path: Path,
    queue_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert enqueue(tmp_path, make_task(), queue_dir) == 0

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("subprocess must not run during a dry-run")

    monkeypatch.setattr(batch_record.subprocess, "run", explode)
    code = batch_record.main(["drain", "--queue-dir", str(queue_dir)])
    captured = capsys.readouterr()
    assert code == 0
    assert "[dry-run]" in captured.out
    assert "weak-signals record" in captured.out
    assert "pass --execute to record" in captured.out
    assert len(batch_record.queued_files(queue_dir)) == 1
    assert not (queue_dir / batch_record.DONE_DIR_NAME).exists()


def test_drain_processes_fifo_oldest_first(
    tmp_path: Path, queue_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = ["sig_c_first", "sig_a_second", "sig_b_third"]
    for name in names:
        assert enqueue(tmp_path, make_task(name), queue_dir) == 0
    files = batch_record.queued_files(queue_dir)
    for index, file_path in enumerate(files):
        stamp = f"20260101T00000{index}000000Z"
        file_path.replace(file_path.with_name(f"05-{stamp}-deadbeef.json"))

    seen: list[str] = []
    monkeypatch.setattr(batch_record.subprocess, "run", fake_run(processed=seen))
    code = batch_record.main(["drain", "--execute", "--queue-dir", str(queue_dir)])
    assert code == 0
    assert seen == names
    assert batch_record.queued_files(queue_dir) == []
    done = sorted((queue_dir / batch_record.DONE_DIR_NAME).glob("*.json"))
    assert len(done) == 3


def test_drain_failure_routes_to_failed_with_stderr(
    tmp_path: Path, queue_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert enqueue(tmp_path, make_task("sig_boom"), queue_dir) == 0
    queued = batch_record.queued_files(queue_dir)[0]
    monkeypatch.setattr(
        batch_record.subprocess,
        "run",
        fake_run(results={"sig_boom": (1, "registry exploded")}),
    )
    code = batch_record.main(["drain", "--execute", "--queue-dir", str(queue_dir)])
    assert code == 0
    failed_dir = queue_dir / batch_record.FAILED_DIR_NAME
    failed_files = sorted(failed_dir.glob("*.json"))
    assert len(failed_files) == 1
    stderr_text = (failed_dir / (queued.stem + ".stderr.txt")).read_text(encoding="utf-8")
    assert "registry exploded" in stderr_text
    assert batch_record.queued_files(queue_dir) == []


def test_drain_moves_unparseable_file_to_failed(
    tmp_path: Path, queue_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_dir.mkdir(parents=True)
    bad = queue_dir / "05-20260101T000000000000Z-badbad01.json"
    bad.write_text("{not json", encoding="utf-8")

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("subprocess must not run for an unparseable file")

    monkeypatch.setattr(batch_record.subprocess, "run", explode)
    code = batch_record.main(["drain", "--execute", "--queue-dir", str(queue_dir)])
    assert code == 0
    failed = sorted((queue_dir / batch_record.FAILED_DIR_NAME).glob("*.json"))
    assert len(failed) == 1
    stderr_text = (failed[0].parent / (failed[0].stem + ".stderr.txt")).read_text(encoding="utf-8")
    assert "validation failed" in stderr_text


def test_draining_twice_never_double_records(
    tmp_path: Path, queue_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert enqueue(tmp_path, make_task("sig_once"), queue_dir) == 0
    calls: list[str] = []
    monkeypatch.setattr(batch_record.subprocess, "run", fake_run(processed=calls))
    assert batch_record.main(["drain", "--execute", "--queue-dir", str(queue_dir)]) == 0
    assert batch_record.main(["drain", "--execute", "--queue-dir", str(queue_dir)]) == 0
    assert calls == ["sig_once"]


def test_lock_blocks_second_drainer_until_timeout(
    tmp_path: Path, queue_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("sig_x", "sig_y"):
        assert enqueue(tmp_path, make_task(name), queue_dir) == 0

    processed: list[str] = []
    errors: list[BaseException] = []
    monkeypatch.setattr(batch_record.subprocess, "run", fake_run(processed=processed, delay=0.02))

    acquired = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with batch_record.drain_lock(queue_dir, timeout=10.0):
            acquired.set()
            release.wait(timeout=10.0)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    try:
        assert acquired.wait(timeout=10.0)

        def second_drain() -> None:
            try:
                batch_record.run_drain(queue_dir, execute=True, lock_timeout=0.25)
            except BaseException as exc:
                errors.append(exc)

        racer = threading.Thread(target=second_drain)
        racer.start()
        racer.join(timeout=10.0)
        assert not racer.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], TimeoutError)
        assert processed == []
        assert len(batch_record.queued_files(queue_dir)) == 2
    finally:
        release.set()
        holder.join(timeout=10.0)


def test_two_concurrent_drains_serialize_each_task_exactly_once(
    tmp_path: Path, queue_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(4):
        assert enqueue(tmp_path, make_task(f"sig_{index}"), queue_dir) == 0

    processed: list[str] = []
    guard = threading.Lock()
    monkeypatch.setattr(
        batch_record.subprocess,
        "run",
        fake_run(processed=processed, guard=guard, delay=0.02),
    )

    errors: list[BaseException] = []

    def drain_worker() -> None:
        try:
            batch_record.run_drain(queue_dir, execute=True, lock_timeout=30.0)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=drain_worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60.0)
    assert errors == []
    assert sorted(processed) == sorted(f"sig_{index}" for index in range(4))
    assert len(processed) == len(set(processed))
    assert batch_record.queued_files(queue_dir) == []


def test_status_reports_counts(
    tmp_path: Path,
    queue_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert enqueue(tmp_path, make_task("sig_status"), queue_dir) == 0
    monkeypatch.setattr(batch_record.subprocess, "run", fake_run())
    assert batch_record.main(["drain", "--execute", "--queue-dir", str(queue_dir)]) == 0
    code = batch_record.main(["status", "--queue-dir", str(queue_dir)])
    captured = capsys.readouterr()
    assert code == 0
    assert "queued=0" in captured.out
    assert "done=1" in captured.out
    assert "failed=0" in captured.out
