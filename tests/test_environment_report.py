from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from nfl_ats import environment_report as env_report_module
from nfl_ats.environment_report import (
    classify_field,
    compare_environment,
    environment_report,
)

# ---------------------------------------------------------------------------
# environment_report(): shape
# ---------------------------------------------------------------------------


def test_report_has_the_documented_top_level_sections() -> None:
    report = environment_report()
    for key in (
        "generated_at_utc",
        "python",
        "uv",
        "platform",
        "packages",
        "blas",
        "thread_counts",
        "git",
        "uv_lock",
        "environment_variables",
        "secrets_detected",
    ):
        assert key in report, f"missing section {key!r}"


def test_report_lists_the_numerically_relevant_packages() -> None:
    report = environment_report()
    packages = report["packages"]
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow"):
        assert name in packages
        # Installed in the locked dev env this test runs under.
        assert packages[name] is not None


def test_report_python_and_platform_fields_are_populated() -> None:
    report = environment_report()
    assert report["python"]["major"] == 3
    assert report["python"]["implementation"] == "CPython"
    assert report["platform"]["system"]
    assert report["platform"]["machine"]


def test_report_is_json_serializable() -> None:
    # Would raise TypeError on any non-JSON-native value (e.g. a Path).
    json.dumps(environment_report())


# ---------------------------------------------------------------------------
# secrets: presence booleans only, values never leak
# ---------------------------------------------------------------------------


def test_fake_secret_value_never_appears_in_the_report(monkeypatch: Any) -> None:
    secret_value = "sekrit-value-should-never-leak-9f3c1a"
    monkeypatch.setenv("FAKE_API_KEY", secret_value)

    report = environment_report()
    serialized = json.dumps(report)

    assert secret_value not in serialized
    assert report["secrets_detected"]["FAKE_API_KEY"] is True


def test_known_project_secrets_are_always_reported_as_booleans(monkeypatch: Any) -> None:
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    monkeypatch.setenv("CFBD_API_KEY", "another-secret-value-should-not-leak")

    report = environment_report()
    serialized = json.dumps(report)

    assert "another-secret-value-should-not-leak" not in serialized
    assert report["secrets_detected"]["THE_ODDS_API_KEY"] is False
    assert report["secrets_detected"]["CFBD_API_KEY"] is True


def test_secret_shaped_allowlisted_prefix_is_redacted_to_a_boolean(monkeypatch: Any) -> None:
    """Defense in depth: an NFL_ATS_*-prefixed var named like a secret must
    still never have its value included, even though the prefix is
    allow-listed for the ordinary environment_variables dump.
    """

    monkeypatch.setenv("NFL_ATS_API_KEY", "should-never-appear-anywhere")

    report = environment_report()
    serialized = json.dumps(report)

    assert "should-never-appear-anywhere" not in serialized
    assert report["environment_variables"]["NFL_ATS_API_KEY"] is True


def test_ordinary_env_vars_outside_the_allowlist_are_excluded(monkeypatch: Any) -> None:
    monkeypatch.setenv("SOME_RANDOM_TEST_VAR_XYZ", "not-a-secret-not-allowlisted")

    report = environment_report()

    assert "SOME_RANDOM_TEST_VAR_XYZ" not in report["environment_variables"]


def test_allowlisted_env_vars_are_included_with_their_real_values(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", "registry_test")

    report = environment_report()
    env_vars = report["environment_variables"]

    assert env_vars["PYTHONHASHSEED"] == "0"
    assert env_vars["OMP_NUM_THREADS"] == "4"
    assert env_vars["TZ"] == "UTC"
    assert env_vars["NFL_ATS_REGISTRY_DIR"] == "registry_test"


# ---------------------------------------------------------------------------
# fail-safe wrapper
# ---------------------------------------------------------------------------


def test_environment_report_never_raises_even_when_assembly_breaks(monkeypatch: Any) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated assembly failure")

    monkeypatch.setattr(env_report_module, "_python_info", _boom)

    report = environment_report()

    assert set(report) == {"error"}
    assert "simulated assembly failure" in report["error"]


def test_uv_absence_is_tolerated_not_raised(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(env_report_module.shutil, "which", lambda _name: None)
    env_report_module._cached_uv_version.cache_clear()

    report = environment_report(project_root=tmp_path)

    assert report["uv"]["available"] is False
    assert report["uv"]["version"] is None


def test_report_tolerates_a_non_git_project_root() -> None:
    # Deliberately NOT pytest's tmp_path: this suite is run with
    # --basetemp .agent_tmp/<name> INSIDE this repository, so a plain
    # tmp_path is still under nfl_py3's own .git tree and `git rev-parse`
    # would resolve there instead of failing. tempfile's default location is
    # the OS temp directory, genuinely outside any git repository.
    with tempfile.TemporaryDirectory() as raw_directory:
        report = environment_report(project_root=Path(raw_directory))

    assert report["git"] == {"revision": None, "dirty": None}
    assert report["uv_lock"] == {"present": False, "sha256": None}


def test_precomputed_git_and_lock_info_are_reused_verbatim(tmp_path: Path) -> None:
    git_info = {"revision": "deadbeef", "dirty": True}
    report = environment_report(project_root=tmp_path, git_info=git_info, uv_lock_sha256="abc123")

    assert report["git"] == git_info
    assert report["uv_lock"] == {"present": True, "sha256": "abc123"}


# ---------------------------------------------------------------------------
# compare_environment(): reproducibility-affecting vs cosmetic
# ---------------------------------------------------------------------------


def _synthetic_report(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "generated_at_utc": "2026-09-04T12:00:00Z",
        "python": {"version": "3.12.5", "major": 3, "minor": 12, "micro": 5},
        "uv": {"available": True, "version": "0.4.18", "raw": "uv 0.4.18 (abc 2026-08-01)"},
        "platform": {
            "system": "Windows",
            "release": "10",
            "version": "10.0.19045",
            "machine": "AMD64",
        },
        "packages": {"numpy": "2.1.0", "pandas": "2.2.3"},
        "thread_counts": {"OMP_NUM_THREADS": None},
        "git": {"revision": "aaaa", "dirty": False},
        "uv_lock": {"present": True, "sha256": "hash-a"},
        "environment_variables": {"PYTHONHASHSEED": None},
        "secrets_detected": {"THE_ODDS_API_KEY": True},
    }
    for path, value in overrides.items():
        cursor = base
        keys = path.split(".")
        for key in keys[:-1]:
            cursor = cursor[key]
        cursor[keys[-1]] = value
    return base


def test_identical_reports_have_no_differences() -> None:
    a = _synthetic_report()
    b = _synthetic_report()
    comparison = compare_environment(a, b)

    assert comparison == {
        "differs": False,
        "reproducibility_affecting": False,
        "fields": {},
        "reproducibility_affecting_fields": [],
        "cosmetic_fields": [],
    }


def test_python_minor_version_difference_is_reproducibility_affecting() -> None:
    a = _synthetic_report()
    b = _synthetic_report(**{"python.minor": 13, "python.version": "3.13.0"})
    comparison = compare_environment(a, b)

    assert "python.minor" in comparison["reproducibility_affecting_fields"]
    assert comparison["reproducibility_affecting"] is True
    # The full version string differs on patch too, but is itself cosmetic.
    assert "python.version" in comparison["cosmetic_fields"]


def test_package_version_difference_is_reproducibility_affecting() -> None:
    a = _synthetic_report()
    b = _synthetic_report(**{"packages.numpy": "2.2.0"})
    comparison = compare_environment(a, b)

    assert comparison["fields"]["packages.numpy"]["classification"] == "reproducibility_affecting"


def test_uv_lock_hash_difference_is_reproducibility_affecting() -> None:
    a = _synthetic_report()
    b = _synthetic_report(**{"uv_lock.sha256": "hash-b"})
    comparison = compare_environment(a, b)

    assert comparison["fields"]["uv_lock.sha256"]["classification"] == "reproducibility_affecting"


def test_thread_count_difference_is_reproducibility_affecting() -> None:
    a = _synthetic_report()
    b = _synthetic_report(**{"thread_counts.OMP_NUM_THREADS": "8"})
    comparison = compare_environment(a, b)

    assert (
        comparison["fields"]["thread_counts.OMP_NUM_THREADS"]["classification"]
        == "reproducibility_affecting"
    )


def test_pythonhashseed_difference_is_reproducibility_affecting() -> None:
    a = _synthetic_report()
    b = _synthetic_report(**{"environment_variables.PYTHONHASHSEED": "42"})
    comparison = compare_environment(a, b)

    assert (
        comparison["fields"]["environment_variables.PYTHONHASHSEED"]["classification"]
        == "reproducibility_affecting"
    )


def test_platform_machine_difference_is_reproducibility_affecting() -> None:
    a = _synthetic_report()
    b = _synthetic_report(**{"platform.machine": "arm64"})
    comparison = compare_environment(a, b)

    assert comparison["fields"]["platform.machine"]["classification"] == "reproducibility_affecting"


def test_uv_patch_version_difference_is_cosmetic() -> None:
    a = _synthetic_report()
    b = _synthetic_report(**{"uv.version": "0.4.19"})
    comparison = compare_environment(a, b)

    assert comparison["fields"]["uv.version"]["classification"] == "cosmetic"
    assert comparison["reproducibility_affecting"] is False


def test_os_build_number_difference_is_cosmetic() -> None:
    a = _synthetic_report()
    b = _synthetic_report(**{"platform.version": "10.0.19046"})
    comparison = compare_environment(a, b)

    assert comparison["fields"]["platform.version"]["classification"] == "cosmetic"


def test_git_dirty_flag_difference_is_cosmetic() -> None:
    a = _synthetic_report()
    b = _synthetic_report(**{"git.dirty": True})
    comparison = compare_environment(a, b)

    assert comparison["fields"]["git.dirty"]["classification"] == "cosmetic"


def test_git_revision_difference_is_reproducibility_affecting() -> None:
    a = _synthetic_report()
    b = _synthetic_report(**{"git.revision": "bbbb"})
    comparison = compare_environment(a, b)

    assert comparison["fields"]["git.revision"]["classification"] == "reproducibility_affecting"


def test_timestamp_difference_is_cosmetic() -> None:
    a = _synthetic_report()
    b = _synthetic_report(**{"generated_at_utc": "2026-09-05T00:00:00Z"})
    comparison = compare_environment(a, b)

    assert comparison["fields"]["generated_at_utc"]["classification"] == "cosmetic"
    assert comparison["reproducibility_affecting"] is False


def test_unmatched_field_defaults_to_reproducibility_affecting() -> None:
    assert classify_field("some.brand.new.field.nobody.declared") == "reproducibility_affecting"


def test_an_error_report_compares_without_raising() -> None:
    comparison = compare_environment({"error": "boom"}, _synthetic_report())
    assert comparison["differs"] is True


# ---------------------------------------------------------------------------
# wiring: artifact_provenance() (the shared metadata writer both experiment
# metadata and forecast/weekly-run metadata already go through) carries this
# report additively.
# ---------------------------------------------------------------------------


def test_artifact_provenance_carries_an_environment_section(tmp_path: Path) -> None:
    from nfl_ats.provenance import artifact_provenance

    feature_path = tmp_path / "game_features.parquet"
    feature_path.write_bytes(b"features")

    payload = artifact_provenance({"model": "test"}, feature_path, project_root=tmp_path)

    assert "environment" in payload
    assert "python" in payload["environment"]
    # Reused, not recomputed, from the same git_state()/uv.lock work
    # artifact_provenance() already does for "code"/"uv_lock_sha256".
    assert payload["environment"]["git"] == payload["code"]
    assert payload["environment"]["uv_lock"]["sha256"] == payload["uv_lock_sha256"]
