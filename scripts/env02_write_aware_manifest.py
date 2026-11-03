import hashlib
import json
from pathlib import Path

SOURCE = Path("data/raw/forecast_archive/pool_decision_2009_2025")
TARGET = Path("data/raw/forecast_archive/pool_decision_2009_2025_aware")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    localization = json.loads((TARGET / "localization.json").read_text(encoding="utf-8"))
    target_sha = sha(TARGET / "forecasts.parquet")
    if target_sha != localization["target_sha256"]:
        raise SystemExit("aware parquet does not match localization.json")
    if sha(SOURCE / "forecasts.parquet") != localization["source_sha256"]:
        raise SystemExit("source parquet does not match localization.json")
    if localization["issuance_equals_requested_utc_runtime"] != localization["ok_rows"]:
        raise SystemExit("localization identity is incomplete")
    manifest["files"] = {
        "forecasts.parquet": {
            "rows": manifest["files"]["forecasts.parquet"]["rows"],
            "sha256": target_sha,
        }
    }
    recovered = Path(".tmp/env02-recovery/game_features.parquet").resolve()
    if sha(recovered) != manifest["inputs"]["game_features"]["sha256"]:
        raise SystemExit("recovered game_features does not match the pinned hash")
    manifest["inputs"]["game_features"] = {
        "path": str(recovered),
        "sha256": manifest["inputs"]["game_features"]["sha256"],
        "original_path": manifest["inputs"]["game_features"]["path"],
    }
    manifest["source_snapshot"] = {
        "path": str(SOURCE / "forecasts.parquet"),
        "sha256": localization["source_sha256"],
        "manifest_sha256": sha(SOURCE / "manifest.json"),
    }
    manifest["localization"] = {
        "script": "scripts/env02_localize_issuance.py",
        "derivation": "naive issuance_runtime_utc equals floor_12h(UTC decision_cutoff_utc) minus 12h x lookback_steps_used for every ok row, the UTC runtime the ingest requested; naive time columns localized to UTC",  # noqa: E501
        "columns_localized": [
            "issuance_runtime_utc",
            "forecast_valid_utc",
            "forecast_precip_prob_valid_utc",
            "decision_cutoff_utc",
            "kickoff_utc",
        ],
        "evidence": localization,
        "provenance_basis": "exact identity with requested UTC runtimes accepted in place of a hashed provider contract",  # noqa: E501
    }
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["files"]))


if __name__ == "__main__":
    main()
