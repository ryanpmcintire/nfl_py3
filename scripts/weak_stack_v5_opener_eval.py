"""Predeclared away-only FluView stack; docs/weak_stack_v5.md. Measure only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.clv import opener_pick_evaluation, week_blocked_bootstrap
from nfl_ats.fluview_production_feature import (
    FLUVIEW_AWAY_ASOF_COLUMN,
    attach_fluview_away_asof_features,
    default_fluview_raw_path,
)
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact  # ENG-38
from scripts.weak_stack_v4_opener_eval import paired_frame

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "artifacts/opener_evaluation/20260907T122841Z"
SEED = 20260817


def metric(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "accuracy_points": 100
        * float((frame.candidate_correct_open_pr - frame.baseline_correct_open_pr).mean())
    }


def summarize(frame: pd.DataFrame, samples: int) -> dict:
    valid = frame.dropna(subset=["candidate_correct_open_pr", "baseline_correct_open_pr"])
    return {
        "games": len(frame),
        "nonpush_games": len(valid),
        "weeks": valid.groupby(["season", "week"]).ngroups,
        "baseline_accuracy": float(valid.baseline_correct_open_pr.mean()),
        "candidate_accuracy": float(valid.candidate_correct_open_pr.mean()),
        "week": week_blocked_bootstrap(valid, metric, samples=samples, seed=SEED).iloc[0].to_dict(),
        "season": week_blocked_bootstrap(valid, metric, block="season", samples=samples, seed=SEED)
        .iloc[0]
        .to_dict(),
        "flips": int(valid.baseline_pick_home_pr.ne(valid.candidate_pick_home_pr).sum()),
    }


def frozen_pick_null(frame: pd.DataFrame, draws: int = 200) -> dict:
    valid = frame.dropna(subset=["actual_home_cover_open"]).reset_index(drop=True)
    actual = valid.actual_home_cover_open.to_numpy()
    base = valid.baseline_pick_home_pr.to_numpy()
    candidate = valid.candidate_pick_home_pr.to_numpy()
    groups = list(valid.groupby(["season", "week"]).indices.values())
    rng = np.random.default_rng(SEED)
    effects = []
    for _ in range(draws):
        shuffled = actual.copy()
        for positions in groups:
            shuffled[positions] = rng.permutation(actual[positions])
        effects.append(
            100
            * float(
                ((candidate == shuffled).astype(float) - (base == shuffled).astype(float)).mean()
            )
        )
    return {
        "draws": draws,
        "mean": float(np.mean(effects)),
        "lower": float(np.quantile(effects, 0.025)),
        "upper": float(np.quantile(effects, 0.975)),
        "observed_percentile": float(
            np.mean(np.asarray(effects) <= metric(valid)["accuracy_points"])
        ),
    }


def assert_baseline(saved: pd.DataFrame, replay: pd.DataFrame) -> None:
    columns = [
        "game_id",
        "tue_open_home_spread",
        "home_cover_probability_at_open",
        "pick_home_at_open_probability_rule",
        "correct_at_open_probability_rule",
    ]
    pd.testing.assert_frame_equal(
        saved[columns].sort_values("game_id").reset_index(drop=True),
        replay[columns].sort_values("game_id").reset_index(drop=True),
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()
    out = args.out_dir or REPO / "artifacts/weak_stack_v5_opener_eval" / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((BASELINE / "metadata.json").read_text())
    active = json.loads((REPO / "artifacts/active_ats_model.json").read_text())
    source = REPO / "data/processed/game_features_weak_stack.parquet"
    assert active["model_id"] == metadata["active_model_id"]
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == active["feature_table_sha256"]
        == metadata["feature_table_sha256"]
    )
    config = metadata["active_model_config"]
    for key in [
        "feature_profile",
        "ridge_alpha",
        "regressor",
        "probability_method",
        "calibration_method",
    ]:
        assert config[key] == active[key]
    base_features = pd.read_parquet(source)
    features = attach_fluview_away_asof_features(base_features)
    pd.testing.assert_frame_equal(features[base_features.columns], base_features, check_exact=True)
    features.to_parquet(out / "features.parquet")
    stamp_sidecar(out / "features.parquet")  # ENG-38 provenance
    baseline = pd.read_parquet(BASELINE / "per_game.parquet")
    assert len(baseline) == 1537
    replay = opener_pick_evaluation(
        REPO / "data/market/raw",
        base_features,
        active_model_config=config,
        min_train_games=metadata["min_train_games"],
    )
    assert_baseline(baseline, replay)
    print("Baseline replay matches all 1537 saved opener picks and probabilities.", flush=True)
    candidate = opener_pick_evaluation(
        REPO / "data/market/raw",
        features,
        active_model_config={**config, "feature_profile": "weak_stack_v5"},
        min_train_games=metadata["min_train_games"],
    )
    assert set(candidate.game_id) == set(baseline.game_id)
    paired = paired_frame(baseline, candidate).merge(
        features[["game_id", FLUVIEW_AWAY_ASOF_COLUMN]], on="game_id", validate="one_to_one"
    )
    eligible = paired.loc[paired[FLUVIEW_AWAY_ASOF_COLUMN].notna()].copy()
    policy = paired.copy()
    missing = policy[FLUVIEW_AWAY_ASOF_COLUMN].isna()
    for suffix in ["correct_open_pr", "pick_home_pr"]:
        policy.loc[missing, f"candidate_{suffix}"] = policy.loc[missing, f"baseline_{suffix}"]
    result = {
        "active_model_id": active["model_id"],
        "feature_table_sha256": active["feature_table_sha256"],
        "raw_source": str(default_fluview_raw_path()),
        "baseline_artifact": str(BASELINE),
        "candidate_profile": "weak_stack_v5",
        "feature_columns": [FLUVIEW_AWAY_ASOF_COLUMN],
        "samples": args.samples,
        "seed": SEED,
        "archive_games": len(paired),
        "coverage": len(eligible) / len(paired),
        "primary": summarize(eligible, args.samples),
        "assigned_window": summarize(
            eligible.loc[eligible.season.between(2020, 2021)], args.samples
        ),
        "full_archive_fallback_policy": summarize(policy, args.samples),
        "frozen_pick_null": frozen_pick_null(eligible),
        "seasons": [
            {
                "season": int(season),
                "archive_games": len(group),
                "eligible_games": int(group[FLUVIEW_AWAY_ASOF_COLUMN].notna().sum()),
                **metric(group.loc[group[FLUVIEW_AWAY_ASOF_COLUMN].notna()]),
            }
            for season, group in paired.groupby("season")
        ],
        "secondary": {
            "sign_delta_points": 100
            * float((eligible.candidate_correct_open - eligible.baseline_correct_open).mean()),
            "close_delta_points": 100
            * float(
                (eligible.candidate_correct_close_pr - eligible.baseline_correct_close_pr).mean()
            ),
        },
    }
    for name, frame in [
        ("opener_paired", paired),
        ("opener_eligible", eligible),
        ("opener_baseline", baseline),
        ("opener_candidate", candidate),
    ]:
        frame.to_parquet(out / f"{name}.parquet")
        stamp_sidecar(out / f"{name}.parquet")  # ENG-38 provenance
    write_stamped_artifact(result, out / "opener_summary.json")  # ENG-38 provenance
    print(json.dumps(result, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
