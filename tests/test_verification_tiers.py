"""Contract tests for the fast/full verification tiers (ENG-11).

Why this exists
----------------
``scripts/verify_fast.py`` (PR-speed: safety/typing/lint + ``-m "not full"``)
and ``scripts/verify_full.py`` (the unchanged AGENTS.md release gate) only do
their job if the ``full`` marker is actually declared (``--strict-markers`` is
on, so an undeclared marker is a hard collection error, not a silent no-op)
and if ``-m "not full"`` really does deselect the slow/model-fitting/
real-data/determinism tests tagged with it. These tests pin both, plus that
the two scripts are present and syntactically valid, without re-running either
tier's (multi-second to multi-minute) actual pytest step.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.verify_fast as verify_fast
import scripts.verify_full as verify_full

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pytest_ini_options() -> dict[str, object]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    result = data["tool"]["pytest"]["ini_options"]
    assert isinstance(result, dict)
    return result


def _collected_count(output: str) -> int:
    """Parse pytest's ``--collect-only -q`` trailer.

    E.g. "31/3765 tests collected (3734 deselected)".
    """

    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped.endswith("collected") or " collected " in stripped:
            head = stripped.split()[0]
            if "/" in head:
                return int(head.split("/")[0])
            return int(head)
    raise AssertionError(f"could not find a 'collected' summary line in:\n{output}")


# ---------------------------------------------------------------------------
# 1. The `full` marker is declared (--strict-markers would otherwise reject it)
# ---------------------------------------------------------------------------


def test_full_marker_is_declared_in_pyproject() -> None:
    options = _pytest_ini_options()
    markers = options.get("markers")
    assert isinstance(markers, list) and markers, (
        "expected a non-empty [tool.pytest.ini_options] markers list"
    )
    assert any(str(marker).startswith("full:") for marker in markers), markers
    assert "--strict-markers" in str(options.get("addopts", "")), (
        "the whole point of declaring `full` is that --strict-markers would "
        "otherwise reject an undeclared marker outright"
    )


# ---------------------------------------------------------------------------
# 2. `-m full` collects a non-empty, and `-m "not full"` deselects it
# ---------------------------------------------------------------------------


def test_full_marker_selects_a_nonempty_tagged_set(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "full",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(tmp_path / "collect_full"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _collected_count(completed.stdout) > 0, completed.stdout


def test_not_full_deselects_a_known_tagged_test(tmp_path: Path) -> None:
    """One representative ENG-11-tagged node (a real HGB model fit) must be
    absent from the ``-m "not full"`` collection -- the fast tier's whole
    reason to exist."""

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "not full",
            "tests/test_margin.py",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(tmp_path / "collect_not_full"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "test_margin_hgb_and_guards" not in completed.stdout, completed.stdout


# ---------------------------------------------------------------------------
# 3. The two tier scripts exist and are syntactically valid
# ---------------------------------------------------------------------------


def _flat_commands(steps: list[tuple[str, list[str]]]) -> list[str]:
    return [token for _, cmd in steps for token in cmd]


def test_verify_fast_script_exists_and_parses() -> None:
    path = REPO_ROOT / "scripts" / "verify_fast.py"
    assert path.is_file()
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    tokens = _flat_commands(verify_fast.STEPS)
    assert "ruff" in tokens and "mypy" in tokens and "pytest" in tokens
    # The fast tier's whole reason to exist: it must actually filter pytest
    # by the `full` marker, not just resemble the full tier's command list.
    assert "not full" in tokens


def test_verify_full_script_exists_and_parses() -> None:
    path = REPO_ROOT / "scripts" / "verify_full.py"
    assert path.is_file()
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    tokens = _flat_commands(verify_full.STEPS)
    assert "ruff" in tokens and "mypy" in tokens and "pytest" in tokens
    # The full tier is the AGENTS.md gate unchanged -- its actual pytest
    # invocation (not the docstring, which discusses the fast tier's marker
    # filter in prose) must not filter by marker the way the fast tier does.
    assert "not full" not in tokens
    assert "-m" not in tokens
