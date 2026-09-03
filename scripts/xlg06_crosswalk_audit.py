"""Run the XLG-06 Stage-2 identity coverage audit against local snapshots.

The audit retains the latest local CFBD recruiting and draft snapshots, loads
the current nflverse player identity table, and writes the joined crosswalk
plus its measured coverage and provenance.  It does not read outcomes or
produce a model feature.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import pandas as pd

from nfl_ats.cfb import latest_cfb_snapshot, load_cfb_snapshot
from nfl_ats.io import atomic_parquet, run_id
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact
from nfl_ats.xlg06_crosswalk import (
    build_recruit_to_nfl_crosswalk,
    summarize_crosswalk_cohorts,
)

REPO = Path(__file__).resolve().parents[1]
CFB_ROOT = REPO / "data" / "cfb"
ARTIFACT_ROOT = REPO / "artifacts" / "xlg06_crosswalk"


def _manifest_summary(snapshot: Any) -> dict[str, Any]:
    payload = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    return {
        "source": snapshot.source,
        "snapshot_id": snapshot.snapshot_id,
        "path": str(snapshot.root.relative_to(REPO)),
        "sha256": payload.get("sha256"),
        "seasons": list(snapshot.seasons),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ARTIFACT_ROOT,
        help="artifact root (default: artifacts/xlg06_crosswalk)",
    )
    args = parser.parse_args()

    recruiting_snapshot = latest_cfb_snapshot(CFB_ROOT, "recruiting_players")
    draft_snapshot = latest_cfb_snapshot(CFB_ROOT, "draft_picks")
    recruiting = load_cfb_snapshot(recruiting_snapshot)
    draft = load_cfb_snapshot(draft_snapshot)

    players_polars = nfl.load_players()
    players = (
        players_polars.to_pandas()
        if hasattr(players_polars, "to_pandas")
        else pd.DataFrame(players_polars)
    )
    crosswalk, audit = build_recruit_to_nfl_crosswalk(recruiting, draft, players)
    audit["cohorts"] = summarize_crosswalk_cohorts(crosswalk)

    stamp = run_id(datetime.now(UTC))
    output_dir = args.output_root / stamp
    crosswalk_path = output_dir / "recruit_to_nfl_crosswalk.parquet"
    atomic_parquet(crosswalk, crosswalk_path)

    configuration = {
        "command": "xlg06-crosswalk-audit",
        "allowed_join": (
            "recruiting.athleteId -> draft_picks.collegeAthleteId -> "
            "nfl_players.espn_id -> nfl_players.gsis_id"
        ),
        "name_join_used": False,
        "cfbd_nflAthleteId_used": False,
        "recruiting_snapshot": _manifest_summary(recruiting_snapshot),
        "draft_snapshot": _manifest_summary(draft_snapshot),
        "nflreadpy_loader": "nflreadpy.load_players",
    }
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "audit": audit,
        "sources": configuration,
        "provenance": artifact_provenance(configuration, crosswalk_path, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "audit.json",
        metadata,
        command="xlg06-crosswalk-audit",
        metrics=audit,
        notes="Identity coverage only; no outcomes, model feature, or wagering decision.",
        project_root=REPO,
        registry_root=output_dir / "experiment_registry",
    )
    print(
        json.dumps({"artifact": str(output_dir.relative_to(REPO)), "audit": audit}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
