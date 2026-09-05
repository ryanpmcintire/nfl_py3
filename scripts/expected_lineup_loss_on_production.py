"""LEAD-62: frozen three-column loss construct on the production opener harness.

Run null, positive-control, then screen on the CLI-assigned window. Profiles
are registered only within this process because fleet lanes own shared modules.
All derived outputs and the experiment provenance mirror live under artifacts;
curated weak-signal and rotation records are written separately by their CLIs.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import on_production_opener_confirmation as confirmation  # noqa: E402

from nfl_ats import margin  # noqa: E402
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS  # noqa: E402
from nfl_ats.expected_lineup_loss_features import (  # noqa: E402
    EXPECTED_LINEUP_LOSS_COLUMNS,
    LINEUP_GROUPS,
    attach_expected_lineup_loss_features,
    team_season_split_half_reliability,
)
from nfl_ats.io import atomic_json, atomic_parquet  # noqa: E402
from nfl_ats.provenance import (  # noqa: E402
    artifact_provenance,
    sha256_file,
    stamp_sidecar,
    write_experiment_artifact,
)
from nfl_ats.rotation import load_registry  # noqa: E402

OUTPUT = REPO_ROOT / "artifacts/experiments/expected_lineup_loss_cx5"
BASE_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack.parquet"
PANEL = REPO_ROOT / "data/processed/play_probability_panel.parquet"
PROFILE = "weak_stack_expected_lineup_loss"
FAMILY = "expected_lineup_loss_on_production"
CANDIDATE = confirmation.Candidate(
    FAMILY,
    "",
    PROFILE,
    EXPECTED_LINEUP_LOSS_COLUMNS[0],
    OUTPUT / "features.parquet",
    "docs/expected_lineup_loss.md",
    str(OUTPUT),
)


@contextmanager
def candidate_profile():
    """Production columns plus exactly the frozen three; restore after each arm."""
    names = ("football_" + PROFILE, "full_" + PROFILE)
    extra = tuple(EXPECTED_LINEUP_LOSS_COLUMNS)
    additions = {
        names[0]: FEATURE_SETS["football_weak_stack"] + extra,
        names[1]: FEATURE_SETS["full_weak_stack"] + extra,
    }
    with (
        patch.dict(FEATURE_SETS, additions),
        patch.dict(margin._MARGIN_PROFILE_FEATURE_SETS, {PROFILE: names}),
        patch.object(margin, "MARGIN_FEATURE_PROFILES", (*margin.MARGIN_FEATURE_PROFILES, PROFILE)),
    ):
        baseline = margin.margin_feature_columns("market_residual", "weak_stack")
        candidate = margin.margin_feature_columns("market_residual", PROFILE)
        assert tuple(candidate) == (*baseline, *extra)
        yield {
            "baseline_columns": len(baseline),
            "candidate_columns": len(candidate),
            "added_columns": list(extra),
        }


def build_features(output: Path) -> Path:
    base = pd.read_parquet(BASE_FEATURES)
    manifest = json.loads(BASE_FEATURES.with_suffix(".manifest.json").read_text())
    snapshot = manifest["source_player_snapshot"]
    injuries_path = REPO_ROOT / "data/players/raw" / snapshot / "injuries.parquet"
    panel = pd.read_parquet(PANEL)
    features = attach_expected_lineup_loss_features(
        base, panel=panel, injuries=pd.read_parquet(injuries_path)
    )
    teams = (
        pd.concat(
            [
                features[
                    [
                        "season",
                        "week",
                        f"{side}_team",
                        *[f"{side}_expected_lineup_loss_{group}" for group in LINEUP_GROUPS],
                    ]
                ].rename(
                    columns={
                        f"{side}_team": "team",
                        **{
                            f"{side}_expected_lineup_loss_{group}": f"expected_lineup_loss_{group}"
                            for group in LINEUP_GROUPS
                        },
                    }
                )
                for side in ("home", "away")
            ],
            ignore_index=True,
        )
        .dropna()
        .drop_duplicates(["season", "week", "team"])
    )
    reliability = team_season_split_half_reliability(teams)
    coverage = {}
    for season, rows in features.groupby("season"):
        valid = rows[list(EXPECTED_LINEUP_LOSS_COLUMNS)].notna().all(axis=1)
        coverage[str(season)] = {"games": len(rows), "covered_games": int(valid.sum())}
    path = output / "features.parquet"
    atomic_parquet(features, path)
    atomic_parquet(teams, output / "team_week_loss.parquet")
    metadata = {
        "reliability": reliability,
        "coverage": coverage,
        "panel_rows": len(panel),
        "panel_seasons": sorted(map(int, panel.season.unique())),
        "base_sha256": sha256_file(BASE_FEATURES),
        "panel_sha256": sha256_file(PANEL),
        "injuries_path": str(injuries_path),
        "injuries_sha256": sha256_file(injuries_path),
        "decision": "min(kickoff, Sunday 16:00 America/New_York)",
        "legacy_depth_assumption": "Pregame week proxy; exact observation times unavailable",
    }
    atomic_json(metadata, output / "build.json")
    stamp_sidecar(path, extra=metadata)
    print(json.dumps(metadata), flush=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("null", "positive-control", "screen"), required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=confirmation.BOOTSTRAP_SAMPLES)
    parser.add_argument("--permutations", type=int, default=confirmation.NULL_PERMUTATIONS)
    args = parser.parse_args()
    output = args.output / args.mode
    if (output / "results.json").exists():
        raise ValueError(
            "This mode already has a recorded artifact; do not repeat the outcome look"
        )
    features_path = args.features or build_features(args.output)
    features = pd.read_parquet(features_path)
    # Retain ALL production training rows. Missing early candidate history uses
    # the production imputer, never a shortened baseline training window.
    eligible_games = set(features.dropna(subset=list(EXPECTED_LINEUP_LOSS_COLUMNS))["game_id"])
    scoped, seasons = confirmation.scoped_window_frame(features, load_registry(), FAMILY)
    # Compute and persist reliability before fitting either decision arm.
    build = json.loads((features_path.parent / "build.json").read_text())
    with candidate_profile() as identity:
        baseline = confirmation.run_arm(
            scoped,
            CANDIDATE,
            market_root=confirmation.DEFAULT_MARKET_ROOT,
            profile="weak_stack",
            seasons=seasons,
            min_train_games=DEFAULT_MIN_TRAIN_GAMES,
            leak=False,
        )
        candidate = confirmation.run_arm(
            scoped,
            CANDIDATE,
            market_root=confirmation.DEFAULT_MARKET_ROOT,
            profile=PROFILE,
            seasons=seasons,
            min_train_games=DEFAULT_MIN_TRAIN_GAMES,
            leak=args.mode == "positive-control",
        )
    paired = confirmation.paired_frame(baseline, candidate)
    paired = paired.loc[paired.game_id.isin(eligible_games)].reset_index(drop=True)
    if paired.empty:
        raise ValueError("No paired opener games")
    result = {
        "paired_games": len(paired),
        "profile_identity": identity,
        "reliability": build["reliability"],
        "coverage": build["coverage"],
    }
    if args.mode == "null":
        result["null_production_rule"] = confirmation.null_distribution(
            paired,
            probability_rule=True,
            permutations=args.permutations,
            seed=confirmation.BOOTSTRAP_SEED,
        )
    else:
        for name, reference, treatment in (
            ("opener_production_rule", "baseline_correct_open_pr", "candidate_correct_open_pr"),
            ("opener_sign_rule", "baseline_correct_open", "candidate_correct_open"),
            ("close_production_rule", "baseline_correct_close_pr", "candidate_correct_close_pr"),
        ):
            result[name] = confirmation.summarize(
                paired, reference, treatment, args.bootstrap_samples, confirmation.BOOTSTRAP_SEED
            )
        result["per_season"] = {
            str(season): confirmation.summarize(
                rows,
                "baseline_correct_open_pr",
                "candidate_correct_open_pr",
                args.bootstrap_samples,
                confirmation.BOOTSTRAP_SEED,
            )
            for season, rows in paired.groupby("season")
        }
        result["picks_disagreeing"] = int(
            paired.baseline_pick_home_pr.ne(paired.candidate_pick_home_pr).sum()
        )
    configuration = {
        "mode": args.mode,
        "family": FAMILY,
        "window_seasons": list(seasons),
        "grade": "opener",
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": confirmation.BOOTSTRAP_SEED,
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "target": "market_residual",
        "build": build,
    }
    payload = {
        "result": result,
        **configuration,
        "provenance": artifact_provenance(configuration, features_path),
    }
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "paired_predictions.csv", index=False)
    baseline.to_parquet(output / "baseline_predictions.parquet", index=False)
    candidate.to_parquet(output / "candidate_predictions.parquet", index=False)
    write_experiment_artifact(
        output,
        "results.json",
        payload,
        command="expected-lineup-loss-on-production",
        metrics={"mode": args.mode, "paired_games": len(paired)},
        registry_root=args.output / "experiment_registry",
        rotation_family=FAMILY,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
