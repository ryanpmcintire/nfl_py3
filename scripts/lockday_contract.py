"""Millisecond-scale static audit of lock-day challenger wiring.

This command deliberately does not import pandas or ``nfl_ats``, execute a
recorder, read model data, or write a ledger.  It checks the tracked registry,
the literal CLI result-key map, and verifier coverage as source contracts.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Refresh-time arms do not share the Tuesday publish result-key map.  Keeping
# this tiny contract next to the audit makes additions fail loudly until both
# the production CLI and this readiness check are updated together.
REFRESH_RESULT_KEYS: dict[str, str] = {
    "model_only_refresh_incumbent": "ledger",
    "injury_signal_refresh_tilt": "injury_signal_refresh_tilt",
    "nflcom_friday_refresh_out2_starters_v1": "nflcom_refresh_out2_starters_overlay",
    "inactives_refresh_v1": "inactives_refresh_overlay",
    "crew_tilt_refresh_v1": "crew_tilt_refresh_overlay",
}

WEEKLY_RESULT_KEYS: dict[str, str] = {
    "mod07_weak_signal_stack": "prospective_record",
}


def _literal_assignment(source: str, name: str) -> dict[str, str]:
    """Read one literal mapping without parsing the entire 6k-line CLI."""

    assignment = source.find(name)
    start = source.find("{", assignment)
    if assignment < 0 or start < 0:
        raise ValueError(f"literal assignment {name} not found")
    depth = 0
    end = -1
    for index in range(start, len(source)):
        token = source[index]
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end < 0:
        raise ValueError(f"literal assignment {name} is unterminated")
    parsed = ast.literal_eval(source[start:end])
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} is not a literal mapping")
    return {str(key): str(item) for key, item in parsed.items()}


def audit(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    registry_path = repo_root / "artifacts" / "prospective" / "challengers.json"
    cli_path = repo_root / "src" / "nfl_ats" / "cli.py"
    verifier_path = repo_root / "scripts" / "lockday_verify.py"

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    cli_source = cli_path.read_text(encoding="utf-8")
    verifier_source = verifier_path.read_text(encoding="utf-8")
    publish_keys = _literal_assignment(cli_source, "PUBLISH_CHALLENGER_RESULT_KEYS")

    active = [
        entry
        for entry in registry.get("challengers", [])
        if isinstance(entry, dict) and entry.get("status") == "ACTIVE_PROSPECTIVE"
    ]
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    for entry in active:
        challenger_id = str(entry.get("challenger_id", ""))
        command = str(entry.get("weekly_recording_command", ""))
        if challenger_id in REFRESH_RESULT_KEYS:
            path = "refresh"
            result_key = REFRESH_RESULT_KEYS[challenger_id]
        elif challenger_id in WEEKLY_RESULT_KEYS:
            path = "weekly-run"
            result_key = WEEKLY_RESULT_KEYS[challenger_id]
        else:
            path = "publish"
            result_key = publish_keys.get(challenger_id, "")

        if not result_key:
            errors.append(f"{challenger_id}: no {path} result key")
        elif path != "weekly-run" and f'result["{result_key}"]' not in cli_source:
            errors.append(f"{challenger_id}: CLI never assigns result[{result_key!r}]")
        if path == "publish" and "publish-predictions --record-decisions" not in command:
            errors.append(f"{challenger_id}: registry command is not the publish path")
        rows.append(
            {
                "challenger_id": challenger_id,
                "path": path,
                "result_key": result_key,
            }
        )

    active_ids = {row["challenger_id"] for row in rows}
    stale_publish = sorted(set(publish_keys) - active_ids)
    errors.extend(f"{item}: stale publish result-key entry" for item in stale_publish)
    verifier_contracts = (
        "active_challenger_ids(artifacts_root)",
        "for challenger_id in active:",
        "shared_counts.get(challenger_id, 0)",
    )
    for contract in verifier_contracts:
        if contract not in verifier_source:
            errors.append(f"lockday verifier lost dynamic coverage contract: {contract}")
    if "prospective-record" not in cli_source or "_cmd_prospective_record" not in cli_source:
        errors.append("CLI lost prospective-record command dispatch")
    elapsed_ns = time.perf_counter_ns() - started_ns
    return {
        "ok": not errors,
        "active_registered": len(active),
        "paths": {
            "publish": sum(row["path"] == "publish" for row in rows),
            "refresh": sum(row["path"] == "refresh" for row in rows),
            "weekly_run": sum(row["path"] == "weekly-run" for row in rows),
        },
        "errors": errors,
        "dispatch": rows,
        "audit_elapsed_ns": elapsed_ns,
        "audit_elapsed_ms": elapsed_ns / 1_000_000,
        "contract": "static-only; no recorder, model data, or ledger access",
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    report = audit(REPO_ROOT)
    status = "PASS" if report["ok"] else "FAIL"
    paths = report["paths"]
    print(
        f"lock-day wiring audit: {status} in {report['audit_elapsed_ms']:.3f} ms; "
        f"{report['active_registered']} active "
        f"({paths['publish']} publish, {paths['refresh']} refresh, "
        f"{paths['weekly_run']} weekly-run); {len(report['errors'])} errors"
    )
    for error in report["errors"]:
        print(f"  {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
