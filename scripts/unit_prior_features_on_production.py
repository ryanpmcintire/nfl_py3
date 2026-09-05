"""PER-14 missing-source inventory; no ATS scores without annual unit ratings.

The local unit APM producer writes reliability summaries only. This executable
implements the predeclared missing-artifact fallback, not a fake screen.
"""

from __future__ import annotations

import json
from pathlib import Path

from nfl_ats.provenance import sha256_file, write_stamped_artifact

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/experiments/unit_prior_features"


def main() -> None:
    files = sorted((ROOT / "artifacts/unit_apm").rglob("*"))
    inventory = []
    for path in files:
        if not path.is_file():
            continue
        entry: dict[str, object] = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            entry["keys"] = sorted(data)
            entry["source_seasons"] = (
                data.get("provenance", {}).get("configuration", {}).get("seasons", [])
            )
            entry["unit_summary_keys"] = {
                unit: sorted(summary) for unit, summary in data.get("units", {}).items()
            }
        inventory.append(entry)
    # Fail closed if a producer adds a new layout: this audited fallback must
    # not silently label a newly available coefficient table as missing.
    expected_keys = {"elapsed_seconds", "provenance", "units", "unmapped_positions"}
    expected_summary = {
        "min_plays_per_half",
        "n",
        "spearman_brown_pearson",
        "split_half_pearson",
        "split_half_spearman",
        "unit",
    }
    for entry in inventory:
        if set(entry.get("keys", [])) != expected_keys:
            raise ValueError(f"New unit artifact layout; inspect before screening: {entry['path']}")
        for keys in entry["unit_summary_keys"].values():
            if set(keys) != expected_summary:
                raise ValueError("New unit summary fields; inspect for annual ratings")
    payload = {
        "status": "missing_season_final_unit_ratings",
        "inventory": inventory,
        "files": len(inventory),
        "annual_rating_rows": 0,
        "annual_rating_seasons": [],
        "membership_rows_in_unit_artifacts": 0,
        "paired_games_scored": 0,
        "effect_accuracy_points": None,
        "probability_positive": None,
        "missing": [
            "season/team/OFF_OL final rating and availability timestamp",
            "season/team/OFF_SKILL final rating and availability timestamp",
            "annual odd/even coefficient estimates for team-unit reliability",
            "unit personnel membership and dated offseason continuity in these artifacts",
        ],
        "scope": "artifacts/unit_apm; raw roster/participation reconstruction not attempted",
        "predeclaration": "docs/unit_prior_features.md",
        "registry_measurement": "none: no measured accuracy-point effect",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_stamped_artifact(payload, OUTPUT / "inventory.json", project_root=ROOT)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
