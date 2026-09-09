"""Isolate the wiring fix from the rule change for the rookie-crew reconciliation."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rookie_crew_reconciliation_eval import SAMPLES, SEED, load_script, served_model_config
from threadpoolctl import threadpool_limits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--stamp", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    bat = load_script(repo, "officials_archive_battery_eval")

    from nfl_ats.clv import opener_pick_evaluation
    from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
    from nfl_ats.officials_flag_features import ROOKIE_CREW_UNDERDOG_COLUMN
    from nfl_ats.provenance import write_stamped_artifact

    out = repo / "artifacts/rookie_crew_reconciliation" / args.stamp / "wiring_isolation"
    out.mkdir(parents=True, exist_ok=True)

    artifact = bat.served_artifact_id()
    served_dir = repo / "artifacts/opener_evaluation" / artifact
    served = pd.read_parquet(served_dir / "per_game.parquet")
    metadata = json.loads((served_dir / "metadata.json").read_text(encoding="utf-8"))

    schedule = bat.schedules()
    base_features = pd.read_parquet(bat.FEATURES)
    feed = bat.flag_columns(schedule, close_proxy=False)["rookie_feed"]

    def score(features: pd.DataFrame, profile: str, method: str | None) -> pd.DataFrame:
        config = served_model_config(metadata, profile)
        if method is not None:
            config["probability_method"] = method
        scored = opener_pick_evaluation(
            bat.MARKET_ROOT,
            features,
            active_model_config=config,
            min_train_games=DEFAULT_MIN_TRAIN_GAMES,
        )
        return scored.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    with threadpool_limits(limits=1):
        baseline = score(base_features, "weak_stack", None)
        gate = bat.replay_gate(baseline, served, artifact)
        shipped = score(
            bat.attach(base_features, ROOKIE_CREW_UNDERDOG_COLUMN, feed),
            "weak_stack_rookie_crew_underdog",
            None,
        )
        ecdf_baseline = score(base_features, "weak_stack", "ecdf")
        ecdf_shipped = score(
            bat.attach(base_features, ROOKIE_CREW_UNDERDOG_COLUMN, feed),
            "weak_stack_rookie_crew_underdog",
            "ecdf",
        )

    truth = baseline.loc[:, ["game_id", "season", "week", "margin_vs_open"]]
    cells: dict[str, Any] = {}
    for label, seasons in (("2020_2025", tuple(range(2020, 2026))), ("2020_2021", (2020, 2021))):
        mask = truth["season"].isin(list(seasons)).to_numpy()
        window = truth.loc[mask].reset_index(drop=True)
        cells[label] = {
            "shipped_feed_flag_fixed_wiring_vs_production": bat.paired_accuracy(
                window,
                shipped.loc[mask, "pick_home_at_open_probability_rule"].to_numpy(),
                baseline.loc[mask, "pick_home_at_open_probability_rule"].to_numpy(),
                samples=SAMPLES,
                seed=SEED,
            ),
            "shipped_feed_flag_ecdf_vs_production": bat.paired_accuracy(
                window,
                ecdf_shipped.loc[mask, "pick_home_at_open_probability_rule"].to_numpy(),
                ecdf_baseline.loc[mask, "pick_home_at_open_probability_rule"].to_numpy(),
                samples=SAMPLES,
                seed=SEED,
            ),
        }

    result = {"replay_gate": gate, "served_artifact": artifact, "cells": cells}
    write_stamped_artifact(bat.jsonable(result), out / "results.json")
    print(json.dumps(bat.jsonable(result), indent=1, default=str))


if __name__ == "__main__":
    main()
