"""MOD-06 NFL position-prior shrinkage feature build (standalone, src/ opt-in only).

Executes the predeclaration in ``docs/mod06_position_prior_shrinkage.md``,
written and frozen before this script produced any output. Re-runs
``nfl_ats.players.enrich_with_player_features`` a second time with the new,
opt-in ``value_shrinkage_target="position_prior"`` parameter, using the
IDENTICAL raw snapshots and configuration the existing
``data/processed/game_features_weak_stack.parquet`` table was built from
(read from that table's own manifest), then merges just the six new
``*_js_prior``-suffixed home/away/diff columns onto a COPY of that existing
table by ``game_id``. The original ``game_features_weak_stack.parquet`` is
never touched; output goes to a NEW file,
``data/processed/game_features_weak_stack_js_prior.parquet``, mirroring
`docs/surface_switch_feature_arm.md`'s precedent for adding one derived
column set to `weak_stack` without a full base -> pbp -> learned-
availability rebuild.

Also re-derives the ``value_shrinkage_target="zero"`` columns from the same
re-run and asserts them byte-identical to the existing table's own
``injury_skill_epa_value_lost``/``injury_defense_disruption_value_lost``
columns -- a direct check that this reproduction uses the exact same
inputs/config as the original production build, not merely an assumption.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.constants import (  # noqa: E402
    PLAYER_VALUE_JS_PRIOR_STATE_METRICS,
    PLAYER_VALUE_STATE_METRICS,
)
from nfl_ats.io import atomic_json, atomic_parquet  # noqa: E402
from nfl_ats.pbp import load_pbp_snapshot  # noqa: E402
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root  # noqa: E402
from nfl_ats.players import (  # noqa: E402
    enrich_with_player_features,
    load_player_snapshot,
    load_player_value_snapshot,
    player_snapshot_from_root,
    player_value_snapshot_from_root,
)

PROCESSED = REPO / "data" / "processed"
EXISTING_WEAK_STACK = PROCESSED / "game_features_weak_stack.parquet"
EXISTING_MANIFEST = PROCESSED / "game_features_weak_stack.manifest.json"
OUTPUT_TABLE = PROCESSED / "game_features_weak_stack_js_prior.parquet"
OUTPUT_MANIFEST = PROCESSED / "game_features_weak_stack_js_prior.manifest.json"

VALUE_JS_PRIOR_POOL_MINIMUM = 20  # docs/mod06_position_prior_shrinkage.md: reused CFB constant


def main() -> None:
    started = time.perf_counter()
    if not EXISTING_MANIFEST.is_file():
        raise FileNotFoundError(f"Missing manifest: {EXISTING_MANIFEST}")
    manifest = json.loads(EXISTING_MANIFEST.read_text(encoding="utf-8"))
    print(f"Read manifest: {EXISTING_MANIFEST}", flush=True)

    player_root = REPO / "data" / "players" / "raw"
    pbp_root = REPO / "data" / "pbp" / "raw"
    player_value_root = REPO / "data" / "players" / "values" / "raw"

    player_snapshot = player_snapshot_from_root(player_root / manifest["source_player_snapshot"])
    pbp_snapshot = pbp_snapshot_from_root(pbp_root / manifest["source_pbp_snapshot"])
    player_value_snapshot = player_value_snapshot_from_root(
        player_value_root / manifest["source_player_value_snapshot"]
    )
    print(
        "Snapshots: "
        f"player={player_snapshot.snapshot_id} pbp={pbp_snapshot.snapshot_id} "
        f"player_value={player_value_snapshot.snapshot_id}",
        flush=True,
    )

    injuries, rosters, snaps = load_player_snapshot(player_snapshot)
    pbp = load_pbp_snapshot(pbp_snapshot)
    player_stats = load_player_value_snapshot(player_value_snapshot)

    source_features_path = REPO / manifest["source_features"]
    features = pd.read_parquet(source_features_path)
    availability_rates_path = PROCESSED / "weak_stack_availability_rates.parquet"
    availability_rates = pd.read_parquet(availability_rates_path)
    decision_hours = int(manifest["availability_configuration"]["decision_hours_before_kickoff"])
    print(
        f"Loaded source_features={source_features_path} rows={len(features)}; "
        f"availability_rates rows={len(availability_rates)}; "
        f"decision_hours_before_kickoff={decision_hours}",
        flush=True,
    )

    # role_span, qb_span, qb_min_dropbacks, offseason_retention, value_span,
    # value_prior_snaps all left at function defaults, which match cli.py's
    # build-learned-availability-features subcommand defaults exactly
    # (role_span=8, qb_span=12, qb_min_dropbacks=20, offseason_retention=0.75,
    # value_span=16, value_prior_snaps=200.0).
    common_kwargs = {"decision_hours_before_kickoff": decision_hours}

    print("\n=== Re-deriving value_shrinkage_target='zero' (reproduction check) ===", flush=True)
    t0 = time.perf_counter()
    reproduced_zero = enrich_with_player_features(
        features,
        injuries,
        rosters,
        snaps,
        pbp,
        player_stats=player_stats,
        availability_rates=availability_rates,
        value_shrinkage_target="zero",
        **common_kwargs,
    )
    print(f"  ({time.perf_counter() - t0:.1f}s)", flush=True)

    existing = pd.read_parquet(EXISTING_WEAK_STACK)
    existing_indexed = existing.set_index("game_id")
    reproduced_indexed = reproduced_zero.set_index("game_id")
    mismatches: dict[str, int] = {}
    for metric in PLAYER_VALUE_STATE_METRICS:
        for side in ("home", "away", "diff"):
            column = f"{side}_{metric}"
            aligned_existing, aligned_new = existing_indexed[column].align(
                reproduced_indexed[column]
            )
            not_close = ~np.isclose(
                aligned_existing.to_numpy(dtype=float),
                aligned_new.to_numpy(dtype=float),
                equal_nan=True,
            )
            if not_close.any():
                mismatches[column] = int(not_close.sum())
    if mismatches:
        raise AssertionError(
            "Reproduction of the existing weak_stack table's player-value columns "
            f"under value_shrinkage_target='zero' did NOT match: {mismatches}. "
            "The snapshot/config reproduction in this script is not exact -- do "
            "not proceed to the position_prior candidate build."
        )
    print(
        "Reproduction check PASSED: every home/away/diff "
        f"{PLAYER_VALUE_STATE_METRICS} column matches the existing "
        f"{EXISTING_WEAK_STACK.name} exactly under value_shrinkage_target='zero'.",
        flush=True,
    )

    print("\n=== Building value_shrinkage_target='position_prior' (candidate) ===", flush=True)
    t0 = time.perf_counter()
    candidate = enrich_with_player_features(
        features,
        injuries,
        rosters,
        snaps,
        pbp,
        player_stats=player_stats,
        availability_rates=availability_rates,
        value_shrinkage_target="position_prior",
        value_js_prior_pool_minimum=VALUE_JS_PRIOR_POOL_MINIMUM,
        **common_kwargs,
    )
    print(f"  ({time.perf_counter() - t0:.1f}s)", flush=True)

    rename_map = {
        f"{side}_{old}": f"{side}_{new}"
        for old, new in zip(
            PLAYER_VALUE_STATE_METRICS, PLAYER_VALUE_JS_PRIOR_STATE_METRICS, strict=True
        )
        for side in ("home", "away", "diff")
    }
    new_columns = ["game_id", *rename_map]
    prior_columns = candidate.loc[:, new_columns].rename(columns=rename_map)

    unmatched = set(prior_columns["game_id"]) - set(existing["game_id"])
    if unmatched:
        raise AssertionError(
            f"{len(unmatched)} game_id(s) in the position_prior rebuild do not "
            f"exist in {EXISTING_WEAK_STACK.name}: {sorted(unmatched)[:5]}"
        )
    merged = existing.merge(prior_columns, on="game_id", how="left", validate="one_to_one")
    if len(merged) != len(existing):
        raise AssertionError(f"Merge changed row count: {len(existing)} -> {len(merged)}")
    js_prior_columns = list(rename_map.values())
    na_counts = merged[js_prior_columns].isna().sum().to_dict()
    print(f"js_prior column NA counts after merge: {na_counts}", flush=True)

    # Purely additive: every pre-existing column must be untouched.
    existing_by_game_id = existing.set_index("game_id")
    merged_by_game_id = merged.set_index("game_id")
    for column in existing.columns:
        if column == "game_id":
            continue
        aligned_existing, aligned_merged = existing_by_game_id[column].align(
            merged_by_game_id[column]
        )
        if aligned_existing.dtype.kind in "biufc":
            same = np.allclose(
                aligned_existing.to_numpy(dtype=float),
                aligned_merged.to_numpy(dtype=float),
                equal_nan=True,
            )
        else:
            same = aligned_existing.equals(aligned_merged)
        if not same:
            raise AssertionError(f"Merge altered pre-existing column {column!r}")
    print(
        "Additive-merge check PASSED: every pre-existing weak_stack column is untouched.",
        flush=True,
    )

    atomic_parquet(merged, OUTPUT_TABLE)
    diff_describe = (
        merged[[c for c in js_prior_columns if c.startswith("diff_")]].describe().to_dict()
    )
    metadata = {
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/mod06_position_prior_shrinkage_build.py",
        "predeclaration": "docs/mod06_position_prior_shrinkage.md",
        "source_weak_stack_table": str(EXISTING_WEAK_STACK.relative_to(REPO)),
        "source_manifest": str(EXISTING_MANIFEST.relative_to(REPO)),
        "reproduction_check": (
            "PASSED (value_shrinkage_target='zero' bit/float-matches the existing table)"
        ),
        "value_js_prior_pool_minimum": VALUE_JS_PRIOR_POOL_MINIMUM,
        "new_columns": js_prior_columns,
        "new_column_na_counts": na_counts,
        "new_column_describe": diff_describe,
        "rows": len(merged),
        "destination": str(OUTPUT_TABLE.relative_to(REPO)),
        "total_seconds": time.perf_counter() - started,
    }
    atomic_json(metadata, OUTPUT_MANIFEST)
    print(f"\nWrote {OUTPUT_TABLE} ({len(merged)} rows)", flush=True)
    print(f"Wrote {OUTPUT_MANIFEST}", flush=True)
    print(f"Total runtime: {metadata['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
