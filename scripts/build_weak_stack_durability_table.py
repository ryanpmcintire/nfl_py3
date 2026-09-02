"""Build ``data/processed/game_features_weak_stack_durability.parquet``.

Pure additive merge-by-``game_id`` enrichment of the PRODUCTION
``game_features_weak_stack.parquet`` with the nine `_durability` columns of
``docs/per13_durability_stage2_on_production.md`` sec 4: production's own nine
availability-derived injury columns, rebuilt through production's own
aggregation code on a durability-augmented P(plays).

Built on the production table on purpose, NOT on
``game_features_weak_stack_surface/_v3/_v4/_graph_*/_fluview/_illness.parquet``:
the question is whether the better P(plays) adds to PRODUCTION, and stacking it
onto a profile already refused or still undecided would confound the answer --
the same reason every sibling builder gives for its own choice.

Everything the rebuild needs is PINNED to the production table's own manifest
(player, player-value and PBP snapshots, and the source feature table), so the
zero-offset rebuild is a like-for-like reproduction of production rather than a
re-derivation on newer data. The reproduction result is reported either way; the
candidate column is formed as ``production + (rebuilt_offset -
rebuilt_baseline)``, which is additive against production by construction.

Never touches ``game_features_weak_stack.parquet`` or any other existing file.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/build_weak_stack_durability_table.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from per13_durability_stage1 import build_history  # noqa: E402

from nfl_ats.availability import (  # noqa: E402
    availability_rate_lookup,
    build_availability_outcomes,
    build_season_lagged_availability_rates,
    fixed_unavailability,
    learned_unavailability,
    position_group,
    score_availability_rates,
)
from nfl_ats.constants import (  # noqa: E402
    PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS,
    PER13_DURABILITY_SWAPPED_BASE_COLUMNS,
    TEAM_ABBREVIATION_ALIASES,
)
from nfl_ats.pbp import load_pbp_snapshot  # noqa: E402
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root  # noqa: E402
from nfl_ats.per13_durability_production_feature import (  # noqa: E402
    attach_durability_injury_columns,
    build_durability_offsets,
    durability_severity,
    offset_lookup,
    reproduction_report,
)
from nfl_ats.players import (  # noqa: E402
    attach_snap_player_ids,
    canonicalize_injuries,
    canonicalize_rosters,
    canonicalize_snaps,
    enrich_with_player_features,
    load_player_snapshot,
    load_player_value_snapshot,
    player_snapshot_from_root,
    player_value_snapshot_from_root,
)
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

SOURCE = REPO / "data/processed/game_features_weak_stack.parquet"
MANIFEST = REPO / "data/processed/game_features_weak_stack.manifest.json"
DEST = REPO / "data/processed/game_features_weak_stack_durability.parquet"

DECISION_HOURS = 24


def _snapshot_ids() -> dict[str, str]:
    """Pin every input to the production table's own manifest, never to 'latest'."""

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        "player": str(manifest["source_player_snapshot"]),
        "player_value": str(manifest["source_player_value_snapshot"]),
        "pbp": str(manifest["source_pbp_snapshot"]),
        "features": str(manifest["source_features"]).replace("\\", "/"),
    }


def _team_games(features: pd.DataFrame) -> pd.DataFrame:
    rows = features.loc[
        features["game_id"].notna(),
        ["game_id", "season", "week", "home_team", "away_team", "kickoff"],
    ].copy()
    rows["kickoff"] = pd.to_datetime(rows["kickoff"], errors="coerce", utc=True)
    sides = [
        rows.rename(columns={side: "team"})[["game_id", "season", "week", "team", "kickoff"]]
        for side in ("home_team", "away_team")
    ]
    stacked = pd.concat(sides, ignore_index=True)
    stacked["team"] = stacked["team"].replace(TEAM_ABBREVIATION_ALIASES).astype(str)
    return stacked.drop_duplicates(["season", "week", "team"])


def _base_probability(frame: pd.DataFrame, lookup: dict[Any, float]) -> pd.Series:
    """Production's own severity for a row: learned cell where one exists, else fixed.

    Mirrors ``players._injury_unavailability`` exactly -- the learned lookup is
    consulted first and ``fixed_unavailability`` is the fallback -- but does it
    on a frame rather than a row, so the whole training/target panel can be
    built at once.
    """

    learned = [
        learned_unavailability(
            lookup,
            target_season=int(season),
            report_status=report,
            practice_status=practice,
            position=position,
        )
        for season, report, practice, position in zip(
            frame["season"],
            frame["report_status"],
            frame["practice_status"],
            frame["position"],
            strict=True,
        )
    ]
    fallback = [
        fixed_unavailability(report, practice)
        for report, practice in zip(frame["report_status"], frame["practice_status"], strict=True)
    ]
    return pd.Series(
        [
            float(value) if value is not None else float(other)
            for value, other in zip(learned, fallback, strict=True)
        ],
        index=frame.index,
        dtype=float,
    )


def _training_rows(
    outcomes: pd.DataFrame, kickoffs: pd.DataFrame, lookup: dict[Any, float]
) -> pd.DataFrame:
    frame = outcomes.merge(kickoffs, on="game_id", how="left", validate="many_to_one")
    if frame["kickoff"].isna().any():
        raise RuntimeError("availability outcomes carry a game without a kickoff timestamp")
    frame = frame.rename(
        columns={"report_category": "report_status", "practice_category": "practice_status"}
    )
    frame["base_probability"] = _base_probability(frame, lookup)
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], errors="coerce", utc=True)
    frame["decision_cutoff"] = frame["kickoff"] - pd.Timedelta(hours=DECISION_HOURS)
    frame["gsis_id"] = frame["gsis_id"].astype(str)
    frame["position_group"] = frame["position_group"].astype(str)
    frame["unavailable"] = frame["unavailable"].astype(float)
    return frame.sort_values(["season", "week", "game_id", "gsis_id"]).reset_index(drop=True)


def _target_rows(
    injuries: pd.DataFrame, features: pd.DataFrame, lookup: dict[Any, float]
) -> pd.DataFrame:
    """Every ``(season, week, gsis_id)`` the enrichment loop could ever score.

    Built from the canonical injury table rather than from the availability
    outcome frame, because the outcome frame exists only where participation
    labels do (2013+) while the enrichment loop reads injury reports back to
    2009. Rows the loop never sees simply carry an unused offset.
    """

    frame = injuries.loc[injuries["gsis_id"].notna()].copy()
    frame["team"] = frame["team"].replace(TEAM_ABBREVIATION_ALIASES).astype(str)
    frame = frame.merge(
        _team_games(features), on=["season", "week", "team"], how="inner", validate="many_to_one"
    )
    frame["decision_cutoff"] = frame["kickoff"] - pd.Timedelta(hours=DECISION_HOURS)
    frame["gsis_id"] = frame["gsis_id"].astype(str)
    frame["position_group"] = frame["position"].map(position_group).astype(str)
    frame["base_probability"] = _base_probability(frame, lookup)
    frame = frame.sort_values(["season", "week", "gsis_id", "date_modified"])
    return frame.drop_duplicates(["season", "week", "gsis_id"], keep="last").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO / "artifacts")
    args = parser.parse_args()

    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    destination = args.output_root / "per13_durability_on_production" / stamp

    ids = _snapshot_ids()
    print(f"pinned snapshots: {ids}")
    features = pd.read_parquet(REPO / ids["features"])
    player_snapshot = player_snapshot_from_root(REPO / "data/players/raw" / ids["player"])
    injuries, rosters, snaps = load_player_snapshot(player_snapshot)
    canonical_injuries = canonicalize_injuries(injuries)
    canonical_rosters = canonicalize_rosters(rosters)
    canonical_snaps = attach_snap_player_ids(canonicalize_snaps(snaps), canonical_rosters)

    clock = time.time()
    outcomes = build_availability_outcomes(
        canonical_injuries, canonical_snaps, features, decision_hours_before_kickoff=DECISION_HOURS
    )
    rates = build_season_lagged_availability_rates(
        outcomes, target_seasons=sorted(features["season"].astype(int).unique())
    )
    scored = score_availability_rates(outcomes, rates)
    lookup = availability_rate_lookup(rates)
    print(
        f"availability pipeline: {len(outcomes)} outcomes, {len(scored)} scored, "
        f"{time.time() - clock:.1f}s"
    )

    history = build_history(
        {
            "features": features,
            "outcomes": outcomes,
            "scored": scored,
            "snaps": canonical_snaps,
            "rosters": canonical_rosters,
        }
    )
    kickoffs = features.loc[features["game_id"].notna(), ["game_id", "kickoff"]].drop_duplicates(
        "game_id"
    )
    training = _training_rows(outcomes, kickoffs, lookup)
    targets = _target_rows(canonical_injuries, features, lookup)
    print(f"offset panel: {len(training)} training rows, {len(targets)} target player-games")

    clock = time.time()
    offsets = build_durability_offsets(history, training, targets)
    lookup_offsets = offset_lookup(offsets)
    print(
        f"offsets: {len(lookup_offsets)} non-zero of {len(offsets)} "
        f"({offsets.attrs.get('fitted_weeks', 0)} weeks fitted) in {time.time() - clock:.1f}s"
    )

    support = (
        offsets.groupby("season")
        .agg(
            rows=("offset", "size"),
            with_history=("has_history", "mean"),
            with_offset=("has_offset", "mean"),
        )
        .reset_index()
    )
    support["season"] = support["season"].astype(int)
    print(support.to_string(index=False))

    pbp = load_pbp_snapshot(pbp_snapshot_from_root(REPO / "data/pbp/raw" / ids["pbp"]))
    player_stats = load_player_value_snapshot(
        player_value_snapshot_from_root(REPO / "data/players/values/raw" / ids["player_value"])
    )

    def enrich() -> pd.DataFrame:
        return enrich_with_player_features(
            features,
            injuries,
            rosters,
            snaps,
            pbp,
            player_stats=player_stats,
            availability_rates=rates,
            decision_hours_before_kickoff=DECISION_HOURS,
        )

    clock = time.time()
    rebuilt_baseline = enrich()
    print(f"zero-offset rebuild: {len(rebuilt_baseline)} rows in {time.time() - clock:.1f}s")
    clock = time.time()
    with durability_severity(lookup_offsets):
        rebuilt_offset = enrich()
    print(f"durability rebuild: {len(rebuilt_offset)} rows in {time.time() - clock:.1f}s")

    production = pd.read_parquet(SOURCE)
    print(f"source: {SOURCE} rows={len(production)} cols={len(production.columns)}")
    reproduction = reproduction_report(production, rebuilt_baseline)
    print("reproduction of production's nine columns by the zero-offset rebuild:")
    print(json.dumps(reproduction, indent=2, sort_keys=True))

    widened = attach_durability_injury_columns(production, rebuilt_offset, rebuilt_baseline)
    new_columns = sorted(set(widened.columns) - set(production.columns))
    assert new_columns == sorted(PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS), new_columns
    pre_existing = [column for column in production.columns if column in widened.columns]
    pd.testing.assert_frame_equal(production[pre_existing], widened[pre_existing], check_exact=True)
    print("additivity check passed: every pre-existing column is bit-identical")

    moved = {
        column: int(
            (widened[f"{column}_durability"].fillna(0.0) - widened[column].fillna(0.0))
            .abs()
            .gt(0.0)
            .sum()
        )
        for column in PER13_DURABILITY_SWAPPED_BASE_COLUMNS
    }
    print(f"games whose candidate column differs from production: {moved}")

    widened.to_parquet(DEST)
    print(f"wrote {DEST} rows={len(widened)} cols={len(widened.columns)}")

    configuration = {
        "command": "build-weak-stack-durability-table",
        "snapshots": ids,
        "decision_hours_before_kickoff": DECISION_HOURS,
        "predeclaration": "docs/per13_durability_stage2_on_production.md",
        "source": str(SOURCE),
        "destination": str(DEST),
    }
    payload = {
        "created_at_utc": started.isoformat(),
        **configuration,
        "training_rows": len(training),
        "target_player_games": len(targets),
        "non_zero_offsets": len(lookup_offsets),
        "fitted_weeks": int(offsets.attrs.get("fitted_weeks", 0)),
        "support_by_season": support.to_dict(orient="records"),
        "reproduction": reproduction,
        "games_with_moved_column": moved,
        "provenance": artifact_provenance(configuration, SOURCE, project_root=REPO),
    }
    write_experiment_artifact(
        destination,
        "build.json",
        payload,
        command="build-weak-stack-durability-table",
        metrics={"non_zero_offsets": len(lookup_offsets)},
        notes=(
            "PER-13 Stage 2 candidate table: production weak_stack with its nine "
            "availability-derived injury columns rebuilt on a durability-augmented "
            "P(plays); see docs/per13_durability_stage2_on_production.md."
        ),
    )
    print(f"artifacts: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
