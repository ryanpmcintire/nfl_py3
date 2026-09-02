"""Contracts for the millisecond-scale lock-day wiring audit."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import lockday_contract


def _repo(tmp_path: Path, challengers: list[dict[str, str]]) -> Path:
    (tmp_path / "artifacts" / "prospective").mkdir(parents=True)
    (tmp_path / "src" / "nfl_ats").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "artifacts" / "prospective" / "challengers.json").write_text(
        json.dumps({"challengers": challengers}), encoding="utf-8"
    )
    (tmp_path / "src" / "nfl_ats" / "cli.py").write_text(
        'PUBLISH_CHALLENGER_RESULT_KEYS: dict[str, str] = {"arm": "arm_ledger"}\n'
        'result["arm_ledger"] = recorder()\n'
        'result["prospective_record"] = recorder()\n'
        "def _cmd_prospective_record(): pass\n"
        'command = "prospective-record"\n',
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "lockday_verify.py").write_text(
        "active = active_challenger_ids(artifacts_root)\n"
        "for challenger_id in active:\n"
        "    count = shared_counts.get(challenger_id, 0)\n",
        encoding="utf-8",
    )
    return tmp_path


def test_audit_accepts_a_wired_publish_arm(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        [
            {
                "challenger_id": "arm",
                "status": "ACTIVE_PROSPECTIVE",
                "weekly_recording_command": "nfl-ats publish-predictions --record-decisions",
            }
        ],
    )

    report = lockday_contract.audit(root)

    assert report["ok"] is True
    assert report["active_registered"] == 1
    assert report["paths"] == {"publish": 1, "refresh": 0, "weekly_run": 0}
    assert report["contract"] == "static-only; no recorder, model data, or ledger access"


def test_audit_fails_when_an_active_publish_arm_has_no_cli_result_key(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        [
            {
                "challenger_id": "missing",
                "status": "ACTIVE_PROSPECTIVE",
                "weekly_recording_command": "nfl-ats publish-predictions --record-decisions",
            }
        ],
    )

    report = lockday_contract.audit(root)

    assert report["ok"] is False
    assert "missing: no publish result key" in report["errors"]
    assert "arm: stale publish result-key entry" in report["errors"]


def test_fast_audit_module_has_no_heavy_imports() -> None:
    source = Path(lockday_contract.__file__).read_text(encoding="utf-8")
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "pandas" not in imported
    assert "numpy" not in imported
    assert "nfl_ats" not in imported
