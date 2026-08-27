from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.backup_data as backup_data


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A miniature repo with one mirrorable tree and one machine-local file."""
    repo = tmp_path / "repo"
    (repo / "data" / "raw" / "injury_news").mkdir(parents=True)
    (repo / "data" / "raw" / "injury_news" / "week1.json").write_text("snapshot", encoding="utf-8")
    (repo / "data" / "market" / "raw" / "20260101T000000Z").mkdir(parents=True)
    (repo / "data" / "market" / "raw" / "20260101T000000Z" / "odds.json").write_text(
        "quotes", encoding="utf-8"
    )
    (repo / "data" / "scheduler_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(backup_data, "REPO", repo)
    return repo


def test_status_reports_nothing_covered_before_first_run(fake_repo: Path, tmp_path: Path) -> None:
    report = backup_data.process_tree("data", tmp_path / "mirror", apply=False, verify_all=False)

    assert report.source_files == 2
    assert report.missing == 2
    assert report.up_to_date == 0
    assert report.copied == 0
    assert not (tmp_path / "mirror").exists()


def test_copies_missing_files_and_verifies_them(fake_repo: Path, tmp_path: Path) -> None:
    mirror = tmp_path / "mirror"

    report = backup_data.process_tree("data", mirror, apply=True, verify_all=False)

    assert report.copied == 2
    assert report.verified == 2
    assert not report.failures
    assert (mirror / "data" / "raw" / "injury_news" / "week1.json").read_text(
        encoding="utf-8"
    ) == "snapshot"


def test_machine_local_state_is_never_mirrored(fake_repo: Path, tmp_path: Path) -> None:
    mirror = tmp_path / "mirror"

    backup_data.process_tree("data", mirror, apply=True, verify_all=False)

    assert not (mirror / "data" / "scheduler_state.json").exists()


def test_second_run_copies_nothing(fake_repo: Path, tmp_path: Path) -> None:
    mirror = tmp_path / "mirror"
    backup_data.process_tree("data", mirror, apply=True, verify_all=False)

    report = backup_data.process_tree("data", mirror, apply=True, verify_all=False)

    assert report.copied == 0
    assert report.up_to_date == 2
    assert report.pending == 0


def test_changed_source_is_detected_as_stale_and_recopied(fake_repo: Path, tmp_path: Path) -> None:
    mirror = tmp_path / "mirror"
    backup_data.process_tree("data", mirror, apply=True, verify_all=False)
    source = fake_repo / "data" / "raw" / "injury_news" / "week1.json"
    source.write_text("snapshot revised later", encoding="utf-8")

    report = backup_data.process_tree("data", mirror, apply=True, verify_all=False)

    assert report.stale == 1
    assert report.copied == 1
    assert (mirror / "data" / "raw" / "injury_news" / "week1.json").read_text(
        encoding="utf-8"
    ) == "snapshot revised later"


def test_verify_all_catches_corruption_that_size_and_mtime_miss(
    fake_repo: Path, tmp_path: Path
) -> None:
    mirror = tmp_path / "mirror"
    backup_data.process_tree("data", mirror, apply=True, verify_all=False)
    copied = mirror / "data" / "raw" / "injury_news" / "week1.json"
    original = copied.stat()
    # Same byte count, same mtime -- invisible to the skip check by design.
    copied.write_text("SNAPSHOT", encoding="utf-8")
    os.utime(copied, (original.st_atime, original.st_mtime))

    quiet = backup_data.process_tree("data", mirror, apply=True, verify_all=False)
    assert not quiet.failures

    loud = backup_data.process_tree("data", mirror, apply=True, verify_all=True)
    assert any("content mismatch" in failure for failure in loud.failures)


def test_main_writes_a_manifest_and_reports_success(
    fake_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mirror = tmp_path / "mirror"

    exit_code = backup_data.main(["--dest", str(mirror)])

    assert exit_code == 0
    manifest = json.loads((mirror / backup_data.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["sources"] == ["data"]
    assert manifest["trees"][0]["copied"] == 2
    assert "verified 2 by sha256" in capsys.readouterr().out


def test_status_mode_leaves_the_mirror_untouched(
    fake_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mirror = tmp_path / "mirror"

    exit_code = backup_data.main(["--dest", str(mirror), "--status"])

    assert exit_code == 0
    assert not mirror.exists()
    assert "not yet mirrored" in capsys.readouterr().out
