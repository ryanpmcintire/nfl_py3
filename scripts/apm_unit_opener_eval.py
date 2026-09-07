"""PER-09 assigned-window opener evaluation; frozen before scoring in its doc."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from nfl_ats import clv
from nfl_ats.apm_unit_feature import APM_UNIT_COLUMNS, attach_apm_unit_features
from nfl_ats.participation import (
    build_participation_play_table,
    latest_participation_snapshot,
    load_participation_snapshot,
)
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact  # ENG-38
from nfl_ats.rotation import load_registry
from scripts.weak_stack_v4_opener_eval import paired_frame
from scripts.weak_stack_v5_opener_eval import assert_baseline, frozen_pick_null

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "artifacts/opener_evaluation/20260907T122841Z"
FAMILY = "apm_unit_on_production"
SEED = 20260817


def summary(frame: pd.DataFrame, samples: int) -> dict:
    valid = frame.dropna(subset=["candidate_correct_open_pr", "baseline_correct_open_pr"])
    delta = valid.candidate_correct_open_pr - valid.baseline_correct_open_pr
    block = (
        valid.assign(delta=delta)
        .groupby(["season", "week"], sort=False)
        .delta.agg(["sum", "count"])
    )
    rng = np.random.default_rng(SEED)
    selected = rng.integers(0, len(block), size=(samples, len(block)))
    draws = (
        100
        * block["sum"].to_numpy()[selected].sum(axis=1)
        / block["count"].to_numpy()[selected].sum(axis=1)
    )
    return {
        "games": len(frame),
        "nonpush_games": len(valid),
        "weeks": len(block),
        "baseline_accuracy": float(valid.baseline_correct_open_pr.mean()),
        "candidate_accuracy": float(valid.candidate_correct_open_pr.mean()),
        "delta": 100 * float(delta.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "probability_positive": float((draws > 0).mean()),
        "flips": int(valid.baseline_pick_home_pr.ne(valid.candidate_pick_home_pr).sum()),
    }


def cli(out: Path, *args: str) -> None:
    command = [str(REPO / ".tools/uv.exe"), "run", "--no-sync", "nfl-ats", *args]
    run = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    with (out / "registry_commands.log").open("a", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n" + run.stdout + run.stderr + "\n")
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)


def record(out: Path, result: dict, start: int, end: int) -> None:
    source = str((out / f"window_{start}_{end}.json").relative_to(REPO))
    measured = result["primary"]
    fields = [
        "--effect",
        str(measured["delta"]),
        "--effect-units",
        "accuracy_points",
        "--interval-low",
        str(measured["lower"]),
        "--interval-high",
        str(measured["upper"]),
        "--probability-positive",
        str(measured["probability_positive"]),
        "--sample-blocks",
        str(measured["weeks"]),
        "--notes",
        "Predeclared hierarchical APM; mined-era and prior APM/EPA selection discount. "
        "No admissible closing ground established. Covered games only; full policy in artifact.",
    ]
    cli(
        out,
        "weak-signals",
        "record",
        "--name",
        f"{FAMILY}_{start}_{end}",
        "--family",
        FAMILY,
        "--description",
        "Hierarchical team-unit APM versus active opener picks",
        "--source",
        source,
        "--league",
        "nfl",
        "--season-start",
        str(start),
        "--season-end",
        str(end),
        "--classification",
        "unresolved_below_power",
        "--sample-games",
        str(measured["nonpush_games"]),
        "--category",
        "onfield",
        *fields,
    )
    cli(
        out,
        "rotation",
        "record",
        "--name",
        FAMILY,
        "--artifact",
        source,
        "--verdict",
        "unresolved",
        *fields,
    )


def build_features(base: pd.DataFrame, out: Path) -> pd.DataFrame:
    participation = latest_participation_snapshot(REPO / "data/players/participation/raw")
    pbp = latest_pbp_snapshot(REPO / "data/pbp/raw")
    plays = build_participation_play_table(
        load_participation_snapshot(participation), load_pbp_snapshot(pbp)
    )
    dates = pd.to_datetime(base.gameday).dt.normalize()
    completed = (
        (dates + pd.Timedelta(days=1)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    completed = completed.where(base.result.notna())
    schedule = base[["game_id", "week"]].assign(completed_at=completed)
    plays = plays.merge(schedule, on="game_id", how="left", validate="many_to_one")
    rosters_path = REPO / "data/players/raw/20260817T184901Z/weekly_rosters.parquet"
    rosters = pd.read_parquet(rosters_path)
    decisions = dates - pd.to_timedelta((dates.dt.weekday - 1) % 7, unit="D")
    decisions = decisions.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    augmented = attach_apm_unit_features(
        base.assign(decision_timestamp=decisions), plays, rosters
    ).drop(columns="decision_timestamp")
    pd.testing.assert_frame_equal(augmented[base.columns], base, check_exact=True)
    augmented.to_parquet(out / "features.parquet")
    stamp_sidecar(out / "features.parquet")  # ENG-38 provenance
    write_stamped_artifact(  # ENG-38 provenance
        {
            "participation": participation.snapshot_id,
            "pbp": pbp.snapshot_id,
            "rosters": str(rosters_path),
            "play_rows": len(plays),
            "play_seasons": sorted(plays.season.unique().tolist()),
            "predeclaration_sha256": hashlib.sha256(
                (REPO / "docs/apm_unit_feature_on_production.md").read_bytes()
            ).hexdigest(),
        },
        out / "sources.json",
    )
    return augmented


def evaluate_summary(paired: pd.DataFrame, samples: int) -> dict:
    covered = paired[list(APM_UNIT_COLUMNS[:4])].notna().all(axis=1)
    policy = paired.copy()
    for suffix in ("correct_open_pr", "pick_home_pr"):
        policy.loc[~covered, f"candidate_{suffix}"] = policy.loc[~covered, f"baseline_{suffix}"]
    eligible = paired.loc[covered]
    return {
        "archive_games": len(paired),
        "covered_games": int(covered.sum()),
        "coverage": float(covered.mean()),
        "primary": summary(eligible, samples),
        "full_archive_fallback_policy": summary(policy, samples),
        "frozen_pick_null": frozen_pick_null(eligible),
        "seasons": [
            dict(season=int(y), **summary(g, samples)) for y, g in eligible.groupby("season")
        ],
        "seed": SEED,
        "samples": samples,
        "within_week_correlation": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()
    out = (
        args.out_dir
        or REPO / "artifacts/apm_unit_opener_eval" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    ).resolve()
    out.mkdir(parents=True, exist_ok=False)
    active = json.loads((REPO / "artifacts/active_ats_model.json").read_text())
    metadata = json.loads((BASELINE / "metadata.json").read_text())
    source = REPO / "data/processed/game_features_weak_stack.parquet"
    assert active["model_id"] == metadata["active_model_id"] == "2ceaf63b56b7ce25"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == active["feature_table_sha256"]
    config = metadata["active_model_config"]
    for key in (
        "feature_profile",
        "ridge_alpha",
        "regressor",
        "probability_method",
        "calibration_method",
    ):
        assert config[key] == active[key]
    base = pd.read_parquet(source)
    saved = pd.read_parquet(BASELINE / "per_game.parquet")
    replay = clv.opener_pick_evaluation(
        REPO / "data/market/raw",
        base,
        active_model_config=config,
        min_train_games=metadata["min_train_games"],
    )
    assert len(saved) == 1537
    assert_baseline(saved, replay)
    print("Baseline replay exactly matches 1537 saved games.", flush=True)
    features = build_features(base, out)
    print("Built strictly as-of features.", flush=True)
    all_paired = []
    original_pairing = clv.build_pairing_table
    for look in range(3):
        registry = load_registry()
        window = registry.families[FAMILY].assigned_window
        if window is None:
            raise RuntimeError("No assigned APM window; refusing scoring")
        start, end = window.seasons

        def assigned_pairing(*args, _start=start, _end=end, **kwargs):
            frame = original_pairing(*args, **kwargs)
            return frame.loc[frame.season.between(_start, _end)].copy()

        with patch.object(clv, "build_pairing_table", assigned_pairing):
            candidate = clv.opener_pick_evaluation(
                REPO / "data/market/raw",
                features,
                active_model_config={**config, "feature_profile": "weak_stack_apm_unit"},
                min_train_games=metadata["min_train_games"],
            )
        baseline = saved.loc[saved.season.between(start, end)]
        assert set(candidate.game_id) == set(baseline.game_id)
        paired = paired_frame(baseline, candidate).merge(
            features[["game_id", *APM_UNIT_COLUMNS]], on="game_id", validate="one_to_one"
        )
        paired.to_parquet(out / f"paired_{start}_{end}.parquet")
        stamp_sidecar(out / f"paired_{start}_{end}.parquet")  # ENG-38 provenance
        candidate.to_parquet(out / f"candidate_{start}_{end}.parquet")
        stamp_sidecar(out / f"candidate_{start}_{end}.parquet")  # ENG-38 provenance
        result = evaluate_summary(paired, args.samples)
        write_stamped_artifact(result, out / f"window_{start}_{end}.json")  # ENG-38
        record(out, result, start, end)
        all_paired.append(paired)
        print(f"Recorded {start}-{end}: {result['primary']}", flush=True)
        if look < 2:
            cli(out, "rotation", "assign", "--name", FAMILY)
    full = pd.concat(all_paired, ignore_index=True)
    assert len(full) == 1537 and full.game_id.is_unique
    full.to_parquet(out / "opener_paired.parquet")
    stamp_sidecar(out / "opener_paired.parquet")  # ENG-38 provenance
    result = evaluate_summary(full, args.samples)
    result.update(
        active_model_id=active["model_id"],
        candidate_profile="weak_stack_apm_unit",
        baseline_artifact=str(BASELINE),
        feature_table_sha256=active["feature_table_sha256"],
    )
    write_stamped_artifact(result, out / "opener_summary.json")  # ENG-38 provenance
    print(json.dumps(result, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
