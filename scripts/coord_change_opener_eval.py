"""One predeclared PER-07 opener look; docs/coordinator_change_on_production.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from nfl_ats.clv import decision_market_consensus, load_decision_quotes, opener_pick_evaluation
from nfl_ats.coordinator_changes import (
    COORDINATOR_SEASON_COLUMNS,
    build_coordinator_season_features,
)
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact
from nfl_ats.public_board import find_matching_opener_evaluation
from scripts.weak_stack_v4_opener_eval import paired_frame
from scripts.weak_stack_v5_opener_eval import assert_baseline, frozen_pick_null, metric, summarize

REPO = Path(__file__).resolve().parents[1]


def attach_features(
    base: pd.DataFrame, history: pd.DataFrame, decisions: pd.Series
) -> pd.DataFrame:
    games = base[["game_id", "season", "week", "home_team", "away_team", "kickoff"]].copy()
    games["kickoff"] = pd.to_datetime(games.kickoff, utc=True)
    games["decision_at"] = pd.to_datetime(games.season.astype(str) + "-09-01", utc=True)
    games["decision_at"] = games.decision_at.where(
        games.decision_at.lt(games.kickoff), games.kickoff - pd.Timedelta(days=1)
    )
    games["decision_at"] = games.game_id.map(decisions).fillna(games.decision_at)
    extra = build_coordinator_season_features(games, history)
    result = base.merge(extra, on="game_id", how="left", validate="one_to_one")
    pd.testing.assert_frame_equal(
        result[base.columns], base.reset_index(drop=True), check_exact=True
    )
    return result


def fallback(paired: pd.DataFrame) -> pd.DataFrame:
    result = paired.copy()
    missing = ~result[list(COORDINATOR_SEASON_COLUMNS)].notna().all(axis=1)
    for suffix in ("correct_open_pr", "pick_home_pr"):
        result.loc[missing, f"candidate_{suffix}"] = result.loc[missing, f"baseline_{suffix}"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("NFL_ATS_ARTIFACTS_DIR") or not os.environ.get("NFL_ATS_REGISTRY_DIR"):
        raise ValueError("Set isolated NFL_ATS_ARTIFACTS_DIR and NFL_ATS_REGISTRY_DIR")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    active = json.loads((REPO / "artifacts/active_ats_model.json").read_bytes())
    matched = find_matching_opener_evaluation(REPO / "artifacts", active)
    assert matched is not None
    metadata, baseline_dir = matched
    assert active["model_id"] == "a4c757efd2525da6"
    source = REPO / "data/processed/game_features_weak_stack.parquet"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == active["feature_table_sha256"]
    base = pd.read_parquet(source)
    baseline = pd.read_parquet(baseline_dir / "per_game.parquet")
    assert len(baseline) == 1537
    consensus = decision_market_consensus(
        load_decision_quotes(REPO / "data/market/raw", labels=["tue_open"])
    )
    decisions = consensus.groupby("nflverse_game_id").snapshot_timestamp_utc.max()
    assert baseline.game_id.isin(decisions.index).all()
    features = attach_features(base, pd.read_parquet(args.history), decisions)
    config = metadata["active_model_config"]
    replay = opener_pick_evaluation(
        REPO / "data/market/raw",
        base,
        active_model_config=config,
        min_train_games=metadata["min_train_games"],
    )
    assert_baseline(baseline, replay)
    print("Baseline replay matched 1537 picks and probabilities.", flush=True)
    candidate = opener_pick_evaluation(
        REPO / "data/market/raw",
        features,
        active_model_config={**config, "feature_profile": "weak_stack_coord_change"},
        min_train_games=metadata["min_train_games"],
    )
    assert set(candidate.game_id) == set(baseline.game_id)
    paired = paired_frame(baseline, candidate).merge(
        features[["game_id", *COORDINATOR_SEASON_COLUMNS]], on="game_id", validate="one_to_one"
    )
    eligible = paired.loc[paired[list(COORDINATOR_SEASON_COLUMNS)].notna().all(axis=1)]
    policy = fallback(paired)
    samples = 20000
    result = {
        "active_model_id": active["model_id"],
        "baseline_artifact": str(baseline_dir),
        "source": str(args.history),
        "source_sha256": hashlib.sha256(args.history.read_bytes()).hexdigest(),
        "candidate_profile": "weak_stack_coord_change",
        "seed": 20260817,
        "samples": samples,
        "within_week_correlation": 0,
        "archive_games": len(paired),
        "covered_games": len(eligible),
        "full_archive_fallback_policy": summarize(policy, samples),
        "covered_only": summarize(eligible, samples),
        "assigned_window": summarize(policy.loc[policy.season.between(2020, 2021)], samples),
        "frozen_pick_null": frozen_pick_null(policy),
        "seasons": [
            dict(
                season=int(season),
                games=len(group),
                covered=int(group[list(COORDINATOR_SEASON_COLUMNS)].notna().all(axis=1).sum()),
                **metric(
                    group.dropna(subset=["candidate_correct_open_pr", "baseline_correct_open_pr"])
                ),
            )
            for season, group in policy.groupby("season")
        ],
    }
    for name, frame in (
        ("features", features),
        ("opener_paired", paired),
        ("opener_policy", policy),
        ("opener_candidate", candidate),
        ("opener_baseline", baseline),
    ):
        frame.to_parquet(out / f"{name}.parquet", index=False)
        stamp_sidecar(out / f"{name}.parquet")
    write_stamped_artifact(result, out / "opener_summary.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
