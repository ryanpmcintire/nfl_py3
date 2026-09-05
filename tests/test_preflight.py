"""Tests for the ENG-02 read-only environment/configuration preflight."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from nfl_ats import preflight
from nfl_ats.handoff import _local_inventory


def _completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# Pure version-spec parsing
# ---------------------------------------------------------------------------


def test_parse_version_extracts_leading_digits() -> None:
    assert preflight._parse_version("3.12") == (3, 12)
    assert preflight._parse_version("3.12.4") == (3, 12, 4)


def test_python_satisfies_true_for_matching_spec() -> None:
    satisfied, unparsed = preflight._python_satisfies((3, 12, 4), ">=3.12,<3.14")
    assert satisfied is True
    assert unparsed == []


def test_python_satisfies_false_for_out_of_range() -> None:
    satisfied, unparsed = preflight._python_satisfies((3, 9, 0), ">=3.12,<3.14")
    assert satisfied is False
    assert unparsed == []

    satisfied, unparsed = preflight._python_satisfies((3, 14, 0), ">=3.12,<3.14")
    assert satisfied is False


def test_python_satisfies_reports_unparsed_clause() -> None:
    satisfied, unparsed = preflight._python_satisfies((3, 12, 0), ">=3.12,~=3.12")
    assert unparsed == ["~=3.12"]
    # the parseable clause still governs "satisfied" for clauses that DID parse
    assert satisfied is True


# ---------------------------------------------------------------------------
# check_python_version
# ---------------------------------------------------------------------------


def test_check_python_version_reads_pyproject_requires_python(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.12,<3.14"\n', encoding="utf-8"
    )
    check = preflight.check_python_version(tmp_path, running_version=(3, 12, 4))
    assert check.category == "environment"
    assert check.status == "ok"
    assert "pyproject.toml" in check.detail


def test_check_python_version_fails_when_running_version_too_old(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.12,<3.14"\n', encoding="utf-8"
    )
    check = preflight.check_python_version(tmp_path, running_version=(3, 9, 0))
    assert check.status == "fail"
    assert check.remedy is not None


def test_check_python_version_falls_back_when_pyproject_missing(tmp_path: Path) -> None:
    check = preflight.check_python_version(tmp_path, running_version=(3, 12, 4))
    assert check.status == "ok"
    assert "fallback" in check.detail


# ---------------------------------------------------------------------------
# check_uv_available
# ---------------------------------------------------------------------------


def test_check_uv_available_ok_when_executable_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools_dir = tmp_path / ".tools"
    tools_dir.mkdir()
    uv_path = tools_dir / "uv.exe"
    uv_path.write_text("not a real binary", encoding="utf-8")

    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *a, **k: _completed(0, stdout="uv 0.12.3\n"),
    )
    check, resolved = preflight.check_uv_available(tmp_path)
    assert check.category == "environment"
    assert check.status == "ok"
    assert resolved == uv_path


def test_check_uv_available_fails_when_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda _name: None)
    check, resolved = preflight.check_uv_available(tmp_path)
    assert check.status == "fail"
    assert check.category == "environment"
    assert resolved is None
    assert check.remedy is not None


def test_check_uv_available_fails_when_execution_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools_dir = tmp_path / ".tools"
    tools_dir.mkdir()
    (tools_dir / "uv.exe").write_text("stub", encoding="utf-8")

    def _raise(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("permission denied")

    monkeypatch.setattr(preflight.subprocess, "run", _raise)
    check, resolved = preflight.check_uv_available(tmp_path)
    assert check.status == "fail"
    assert resolved is None


def test_check_uv_available_fails_when_version_command_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools_dir = tmp_path / ".tools"
    tools_dir.mkdir()
    (tools_dir / "uv.exe").write_text("stub", encoding="utf-8")
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: _completed(1, stderr="broken"))
    check, resolved = preflight.check_uv_available(tmp_path)
    assert check.status == "fail"
    assert resolved is None


# ---------------------------------------------------------------------------
# check_uv_cache
# ---------------------------------------------------------------------------


def test_check_uv_cache_fail_when_uv_path_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def _run(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        return _completed(0)

    monkeypatch.setattr(preflight.subprocess, "run", _run)
    check = preflight.check_uv_cache(None)
    assert check.status == "fail"
    assert check.category == "environment"
    assert called is False  # never shells out when there is no known uv path


def test_check_uv_cache_ok_when_directory_exists_and_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "uv-cache"
    cache_dir.mkdir()
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *a, **k: _completed(0, stdout=str(cache_dir) + "\n"),
    )
    check = preflight.check_uv_cache(tmp_path / ".tools" / "uv.exe")
    assert check.status == "ok"
    assert not any(cache_dir.glob(".nfl_ats_preflight_*"))  # probe file removed


def test_check_uv_cache_warn_when_directory_missing_but_ancestor_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_cache = tmp_path / "not-yet" / "uv-cache"
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *a, **k: _completed(0, stdout=str(missing_cache) + "\n"),
    )
    check = preflight.check_uv_cache(tmp_path / ".tools" / "uv.exe")
    assert check.status == "warn"
    assert "does not exist yet" in check.detail


def test_check_uv_cache_fail_when_command_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: _completed(1, stderr="boom"))
    check = preflight.check_uv_cache(tmp_path / ".tools" / "uv.exe")
    assert check.status == "fail"


# ---------------------------------------------------------------------------
# check_git_available / check_hooks_path
# ---------------------------------------------------------------------------


def test_check_git_available_ok_and_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda _name: "C:/Git/git.exe")
    check, path = preflight.check_git_available()
    assert check.status == "ok"
    assert path == "C:/Git/git.exe"

    monkeypatch.setattr(preflight.shutil, "which", lambda _name: None)
    check, path = preflight.check_git_available()
    assert check.status == "fail"
    assert path is None


def test_check_hooks_path_fail_when_git_unavailable(tmp_path: Path) -> None:
    check = preflight.check_hooks_path(tmp_path, None)
    assert check.status == "fail"
    assert check.category == "environment"


def test_check_hooks_path_ok_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        preflight.subprocess, "run", lambda *a, **k: _completed(0, stdout=".githooks\n")
    )
    check = preflight.check_hooks_path(tmp_path, "git")
    assert check.status == "ok"


def test_check_hooks_path_fail_when_wrong_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        preflight.subprocess, "run", lambda *a, **k: _completed(0, stdout="hooks\n")
    )
    check = preflight.check_hooks_path(tmp_path, "git")
    assert check.status == "fail"
    assert "hooks" in check.detail
    assert check.remedy is not None and ".githooks" in check.remedy


def test_check_hooks_path_fail_when_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: _completed(1, stdout=""))
    check = preflight.check_hooks_path(tmp_path, "git")
    assert check.status == "fail"
    assert "not configured" in check.detail


# ---------------------------------------------------------------------------
# check_writable_directory
# ---------------------------------------------------------------------------


def test_check_writable_directory_ok_for_existing_writable_dir(tmp_path: Path) -> None:
    target = tmp_path / "artifacts"
    target.mkdir()
    check = preflight.check_writable_directory("artifacts directory writable", target)
    assert check.status == "ok"
    assert check.category == "environment"
    assert not any(target.glob(".nfl_ats_preflight_*"))


def test_check_writable_directory_warn_for_missing_dir_with_writable_ancestor(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifacts"  # deliberately not created
    check = preflight.check_writable_directory("artifacts directory writable", target)
    assert check.status == "warn"
    assert "does not exist yet" in check.detail


def test_check_writable_directory_fail_when_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "artifacts"
    target.mkdir()
    monkeypatch.setattr(preflight, "_probe_writable", lambda _directory: (False, "denied"))
    check = preflight.check_writable_directory("artifacts directory writable", target)
    assert check.status == "fail"
    assert check.remedy is not None


# ---------------------------------------------------------------------------
# Configuration category: source policy and directory overrides
# ---------------------------------------------------------------------------


def test_check_source_policy_reports_presence_without_leaking_value() -> None:
    secret = "sk-totally-secret-value-12345"
    checks = preflight.check_source_policy({"THE_ODDS_API_KEY": secret})
    by_name = {check.name: check for check in checks}
    odds_check = by_name["source policy: THE_ODDS_API_KEY"]
    assert odds_check.category == "configuration"
    assert odds_check.status == "ok"
    assert secret not in odds_check.detail
    assert secret not in json.dumps(odds_check.to_dict())

    cfbd_check = by_name["source policy: CFBD_API_KEY"]
    assert cfbd_check.status == "warn"


def test_check_source_policy_warns_when_absent() -> None:
    checks = preflight.check_source_policy({})
    assert all(check.status == "warn" for check in checks)
    assert all(check.category == "configuration" for check in checks)


def test_check_directory_overrides_reports_default_and_override() -> None:
    checks = preflight.check_directory_overrides({})
    by_name = {check.name: check for check in checks}
    assert "default" in by_name["directory override: NFL_ATS_DATA_DIR"].detail
    assert all(check.status == "ok" for check in checks)

    overridden = preflight.check_directory_overrides({"NFL_ATS_DATA_DIR": "D:/custom-data"})
    by_name = {check.name: check for check in overridden}
    assert "D:/custom-data" in by_name["directory override: NFL_ATS_DATA_DIR"].detail


# ---------------------------------------------------------------------------
# Research-data category: absence is legitimate, never a failure
# ---------------------------------------------------------------------------


def test_check_research_artifacts_present_vs_absent(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "game_features.parquet").write_bytes(b"not really parquet")
    artifacts_root = tmp_path / "artifacts"

    checks = preflight.check_research_artifacts(tmp_path, artifacts_root)
    assert checks  # the inventory is non-empty
    assert all(check.category == "research_data" for check in checks)

    present = [check for check in checks if check.status == "ok"]
    absent = [check for check in checks if check.status == "warn"]
    assert present  # the one file we created shows up as present
    assert absent  # everything else is legitimately missing on a fresh clone
    assert any("game_features.parquet" in check.detail for check in present)


def test_check_research_artifacts_never_fails(tmp_path: Path) -> None:
    # Completely empty repo/artifacts roots: every row must still be ok/warn.
    checks = preflight.check_research_artifacts(tmp_path, tmp_path / "artifacts")
    assert all(check.status in ("ok", "warn") for check in checks)
    assert len(checks) == len(_local_inventory(tmp_path, tmp_path / "artifacts"))


# ---------------------------------------------------------------------------
# preflight_exit_code rule (pure, no I/O)
# ---------------------------------------------------------------------------


def _check(
    category: preflight.Category, status: preflight.Status, name: str = "x"
) -> preflight.PreflightCheck:
    return preflight.PreflightCheck(name=name, category=category, status=status, detail="d")


def test_exit_code_zero_when_everything_ok() -> None:
    report = preflight.PreflightReport(
        generated_at_utc="2026-09-04T00:00:00+00:00",
        checks=(_check("environment", "ok"), _check("configuration", "ok")),
    )
    assert preflight.preflight_exit_code(report) == 0
    assert preflight.preflight_exit_code(report, strict=True) == 0


def test_exit_code_nonzero_for_environment_failure() -> None:
    report = preflight.PreflightReport(
        generated_at_utc="2026-09-04T00:00:00+00:00",
        checks=(_check("environment", "fail"),),
    )
    assert preflight.preflight_exit_code(report) == 1
    assert preflight.preflight_exit_code(report, strict=True) == 1


def test_exit_code_nonzero_for_configuration_failure() -> None:
    report = preflight.PreflightReport(
        generated_at_utc="2026-09-04T00:00:00+00:00",
        checks=(_check("configuration", "fail"),),
    )
    assert preflight.preflight_exit_code(report) == 1


def test_exit_code_zero_by_default_for_missing_research_data_but_nonzero_strict() -> None:
    report = preflight.PreflightReport(
        generated_at_utc="2026-09-04T00:00:00+00:00",
        checks=(_check("environment", "ok"), _check("research_data", "warn")),
    )
    assert preflight.preflight_exit_code(report) == 0
    assert preflight.preflight_exit_code(report, strict=True) == 1


def test_exit_code_zero_by_default_for_configuration_warn_but_nonzero_strict() -> None:
    report = preflight.PreflightReport(
        generated_at_utc="2026-09-04T00:00:00+00:00",
        checks=(_check("configuration", "warn"),),
    )
    assert preflight.preflight_exit_code(report) == 0
    assert preflight.preflight_exit_code(report, strict=True) == 1


# ---------------------------------------------------------------------------
# run_preflight integration: all three categories, fully mocked subprocess
# ---------------------------------------------------------------------------


def test_run_preflight_aggregates_all_categories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path
    (repo_root / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.0"\n', encoding="utf-8"
    )
    tools_dir = repo_root / ".tools"
    tools_dir.mkdir()
    (tools_dir / "uv.exe").write_text("stub", encoding="utf-8")

    data_root = repo_root / "data"
    artifacts_root = repo_root / "artifacts"
    registry_root = repo_root / "registry"
    for directory in (data_root, artifacts_root, registry_root):
        directory.mkdir()

    uv_cache_dir = repo_root / "uv-cache"
    uv_cache_dir.mkdir()

    monkeypatch.setattr(preflight.shutil, "which", lambda name: "git" if name == "git" else None)

    def _fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if args[-1] == "--version":
            return _completed(0, stdout="uv 0.12.3\n")
        if args[-2:] == ["cache", "dir"]:
            return _completed(0, stdout=str(uv_cache_dir) + "\n")
        if args[-3:] == ["config", "--get", "core.hooksPath"]:
            return _completed(0, stdout=".githooks\n")
        raise AssertionError(f"unexpected subprocess call: {args}")

    monkeypatch.setattr(preflight.subprocess, "run", _fake_run)

    report = preflight.run_preflight(
        repo_root,
        data_root=data_root,
        artifacts_root=artifacts_root,
        registry_root=registry_root,
        env={"THE_ODDS_API_KEY": "present-value"},
    )

    categories = {check.category for check in report.checks}
    assert categories == {"environment", "research_data", "configuration"}

    expected_research_count = len(_local_inventory(repo_root, artifacts_root))
    research_checks = [c for c in report.checks if c.category == "research_data"]
    assert len(research_checks) == expected_research_count
    assert all(c.status in ("ok", "warn") for c in research_checks)

    # Nothing in this fully-configured happy path should fail.
    assert report.has_environment_or_configuration_failure() is False
    assert preflight.preflight_exit_code(report) == 0
    # But the research-data inventory is entirely absent in this bare tmp_path
    # repo, so --strict must catch it.
    assert preflight.preflight_exit_code(report, strict=True) == 1

    # The report round-trips through JSON (the --json CLI path).
    payload = report.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["version"] == preflight.PREFLIGHT_VERSION
    assert payload["summary"]["fail"] == 0

    # Secrets passed in via env never appear in the serialized report.
    assert "present-value" not in json.dumps(payload)


# ---------------------------------------------------------------------------
# CLI wiring (registration only; the handler's own logic is covered above)
# ---------------------------------------------------------------------------


def test_cli_registers_preflight_subcommand() -> None:
    from nfl_ats.cli import build_parser
    from nfl_ats.cli_commands.operations import _cmd_preflight

    parser = build_parser()
    args = parser.parse_args(["preflight", "--json", "--strict"])
    assert args.handler is _cmd_preflight
    assert args.json is True
    assert args.strict is True


def test_cli_preflight_defaults_are_off() -> None:
    from nfl_ats.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["preflight"])
    assert args.json is False
    assert args.strict is False
