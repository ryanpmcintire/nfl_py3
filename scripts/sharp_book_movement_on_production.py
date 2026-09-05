"""MKT-15 coverage and frozen refresh screen; docs/sharp_book_movement_lead.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from observed_movement_channel import _load_intraday_with_kickoff  # noqa: E402

from nfl_ats.clv import (  # noqa: E402
    opener_pick_evaluation,
    pick_correct,
    resolve_active_model_config,
)
from nfl_ats.io import atomic_csv, atomic_parquet, run_id  # noqa: E402
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.provenance import (  # noqa: E402
    artifact_provenance,
    sha256_file,
    write_experiment_artifact,
)
from nfl_ats.sharp_book_movement_features import (  # noqa: E402
    LEADERSHIP_WEIGHTS,
    THRESHOLD,
    refresh_pick,
    sharp_book_movement_features,
)

OUTPUT = REPO / "artifacts/experiments/sharp_book_movement"
FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"
SAMPLES = 20_000
SEED = 2026090518


def summarize(frame: pd.DataFrame, candidate: str, baseline: str) -> dict[str, Any]:
    """Whole-week bootstrap via sufficient statistics, same game-weighted mean."""
    paired = frame.dropna(subset=[candidate, baseline]).copy()
    paired["delta"] = (paired[candidate] - paired[baseline]) * 100.0
    blocks = paired.groupby(["season", "week"], sort=False).delta.agg(["sum", "count"])
    if blocks.empty:
        raise ValueError("No paired non-push games")
    selected = np.random.default_rng(SEED).integers(0, len(blocks), (SAMPLES, len(blocks)))
    draws = blocks["sum"].to_numpy()[selected].sum(axis=1) / blocks["count"].to_numpy()[
        selected
    ].sum(axis=1)
    return {
        "games": len(paired),
        "weeks": len(blocks),
        "candidate_accuracy": float(paired[candidate].mean()),
        "baseline_accuracy": float(paired[baseline].mean()),
        "effect": float(paired.delta.mean()),
        "interval_low": float(np.quantile(draws, 0.025)),
        "interval_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float((draws > 0).mean()),
        "season_effects": {
            str(s): float(v) for s, v in paired.groupby("season").delta.mean().items()
        },
    }


def reliability(frame: pd.DataFrame, flag: str) -> dict[str, Any]:
    teams = pd.concat(
        [
            frame[["season", "week", side, flag]].rename(columns={side: "team"})
            for side in ("home_team", "away_team")
        ]
    )
    teams["parity"] = teams.week % 2
    halves = teams.groupby(["season", "team", "parity"])[flag].mean().unstack("parity")
    halves = halves.dropna()
    value = halves[0].corr(halves[1]) if len(halves) > 1 else float("nan")
    return {"team_seasons": len(halves), "correlation": float(value) if pd.notna(value) else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--quotes-cache", type=Path)
    parser.add_argument("--kickoff-cache", type=Path)
    args = parser.parse_args()
    output = OUTPUT / run_id()
    features = regular_season_rows(pd.read_parquet(FEATURES))
    if args.quotes_cache is not None:
        if args.kickoff_cache is None:
            parser.error("--quotes-cache requires --kickoff-cache")
        quotes, kickoff = pd.read_parquet(args.quotes_cache), pd.read_parquet(args.kickoff_cache)
    else:
        quotes, kickoff = _load_intraday_with_kickoff(REPO / "data/market/raw", features)
    games = kickoff.rename(columns={"nflverse_game_id": "game_id"}).merge(
        features[["game_id", "home_team", "away_team"]], on="game_id", validate="one_to_one"
    )
    exposure = sharp_book_movement_features(quotes, games)
    coverage = (
        exposure.groupby("season")
        .agg(
            archive_games=("game_id", "size"),
            games_with_leader_move=("leader_move_observed", "sum"),
            leader_flags=("leader_flag", "sum"),
            equal_flags=("equal_flag", "sum"),
        )
        .reset_index()
    )
    atomic_csv(coverage, output / "coverage.csv")
    atomic_parquet(exposure, output / "exposure.parquet")
    config: dict[str, Any] = {
        "weights": LEADERSHIP_WEIGHTS,
        "threshold": THRESHOLD,
        "bootstrap_samples": SAMPLES,
        "seed": SEED,
        "predeclaration_sha256": sha256_file(REPO / "docs/sharp_book_movement_lead.md"),
        "feature_module_sha256": sha256_file(REPO / "src/nfl_ats/sharp_book_movement_features.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "quotes_cache_sha256": sha256_file(args.quotes_cache) if args.quotes_cache else None,
        "kickoff_cache_sha256": sha256_file(args.kickoff_cache) if args.kickoff_cache else None,
    }
    metadata: dict[str, Any] = {"coverage": coverage.to_dict("records"), "configuration": config}
    print(coverage.to_string(index=False), flush=True)
    if not args.coverage_only:
        active = resolve_active_model_config(REPO / "artifacts")
        if active.get("feature_table_sha256") != sha256_file(FEATURES):
            raise ValueError("Feature table differs from active model manifest")
        print(f"Recomputing chronological production probability picks: {active}", flush=True)
        scored = opener_pick_evaluation(
            REPO / "data/market/raw", features, active_model_config=active
        )
        frame = scored.merge(
            exposure.drop(columns=["season", "week"]), on="game_id", validate="one_to_one"
        )
        frame = frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
        production = "pick_home_at_open_probability_rule"
        baseline = "correct_at_open_probability_rule"
        cells: dict[str, Any] = {}
        for arm in ("leader", "equal", "closing_control"):
            if arm == "closing_control":
                movement = frame.close_home_spread - frame.tue_open_home_spread
                frame[f"{arm}_pick"] = frame[production].where(movement.eq(0), movement.gt(0))
                flag = movement.ne(0)
            else:
                frame[f"{arm}_pick"] = refresh_pick(frame[production], frame[f"{arm}_net_move"])
                flag = frame[f"{arm}_flag"]
            frame[f"{arm}_correct"] = pick_correct(frame[f"{arm}_pick"], frame.margin_vs_open)
            cell = summarize(frame, f"{arm}_correct", baseline)
            cell["flags"] = int(flag.sum())
            cell["flips"] = int(frame[f"{arm}_pick"].ne(frame[production]).sum())
            cell["exposure_reliability"] = (
                reliability(exposure, f"{arm}_flag") if arm != "closing_control" else None
            )
            cell["population_reliability"] = (
                reliability(frame, f"{arm}_flag") if arm != "closing_control" else None
            )
            cells[arm] = cell
        cells["leader_minus_equal"] = summarize(frame, "leader_correct", "equal_correct")
        atomic_parquet(frame, output / "per_game.parquet")
        atomic_csv(
            pd.json_normalize([{"arm": name, **cell} for name, cell in cells.items()]),
            output / "cells.csv",
        )
        metadata.update(
            {
                "cells": cells,
                "active_model_config": active,
                "population": len(frame),
                "pushes": int(frame[baseline].isna().sum()),
            }
        )
        print(json.dumps(cells, indent=2), flush=True)
    metadata["provenance"] = artifact_provenance(config, FEATURES, project_root=REPO)
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="sharp-book-movement-on-production",
        metrics=metadata,
        registry_root=OUTPUT / "local_registry",
        notes=(
            "Frozen MKT-15 paired refresh screen. "
            "Shared registry writes only via separate record CLI."
        ),
    )
    print(f"artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
