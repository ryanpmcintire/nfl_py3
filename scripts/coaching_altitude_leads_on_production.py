"""CX15 fixed Denver-home overlay screen; docs/coaching_leads.md predeclares it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.altitude_split_features import denver_home_flag, quarter_margins
from nfl_ats.pbp import latest_pbp_snapshot
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact
from nfl_ats.public_board import find_matching_opener_evaluation

SAMPLES = 20_000
SEED = 20260905
OUTPUT = Path("artifacts/experiments/coaching_altitude_leads")


def blocked_means(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """Joint week resampling; nullable columns retain their own populations."""
    grouped = frame.groupby(["season", "week"])[columns]
    sums = grouped.sum().to_numpy(float)
    counts = grouped.count().to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = np.empty((SAMPLES, len(columns)))
    for start in range(0, SAMPLES, 250):
        indices = rng.integers(0, len(sums), size=(min(250, SAMPLES - start), len(sums)))
        denominator = counts[indices].sum(axis=1)
        draws[start : start + len(indices)] = np.divide(
            sums[indices].sum(axis=1),
            denominator,
            out=np.full_like(denominator, np.nan),
            where=denominator > 0,
        )
    result: dict[str, Any] = {}
    for i, column in enumerate(columns):
        values = draws[:, i]
        values = values[np.isfinite(values)]
        result[column] = {
            "estimate": float(frame[column].mean()),
            "interval95": np.quantile(values, [0.025, 0.975]).tolist(),
            "probability_positive": float((values > 0).mean()),
            "standard_error": float(values.std(ddof=1)),
            "games": int(frame[column].count()),
        }
    if "denver_fourth" in columns:
        delta = draws[:, columns.index("denver_fourth")] - draws[:, columns.index("fourth_margin")]
        delta = delta[np.isfinite(delta)]
        result["denver_minus_league_fourth"] = {
            "estimate": float(frame.denver_fourth.mean() - frame.fourth_margin.mean()),
            "interval95": np.quantile(delta, [0.025, 0.975]).tolist(),
            "probability_positive": float((delta > 0).mean()),
        }
    return result


def split_half(frame: pd.DataFrame) -> dict[str, Any]:
    denver = frame.loc[denver_home_flag(frame)].copy()
    denver["half"] = denver.groupby("season").cumcount() % 2
    pivot = denver.pivot_table(index="season", columns="half", values="late_minus_early_rate")
    values = pivot.dropna().to_numpy(float)
    rng = np.random.default_rng(SEED)
    correlations = []
    for _ in range(SAMPLES):
        sample = values[rng.integers(0, len(values), len(values))]
        if np.all(sample.std(axis=0) > 0):
            correlations.append(float(np.corrcoef(sample.T)[0, 1]))
    return {
        "estimate": float(np.corrcoef(values.T)[0, 1]),
        "interval95": np.quantile(correlations, [0.025, 0.975]).tolist(),
        "probability_positive": float((np.array(correlations) > 0).mean()),
        "seasons": len(values),
        "halves": {
            str(h): blocked_means(g, ["late_minus_early_rate"]) for h, g in denver.groupby("half")
        },
    }


def paired_overlay(predictions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """All-slate paired override, preserving the production probability rule."""
    result = predictions.merge(
        games[["game_id", "home_team"]], on="game_id", validate="one_to_one"
    ).sort_values(["season", "week", "game_id"])
    if len(result) != len(predictions):
        raise ValueError("Schedule does not cover every prediction")
    result = result.loc[result.margin_vs_open.notna() & result.margin_vs_open.ne(0)].copy()
    result["denver_home"] = denver_home_flag(result)
    base_pick = result.pick_home_at_open_probability_rule.astype(bool)
    covered = result.margin_vs_open.gt(0)
    result["candidate_pick_home"] = base_pick | result.denver_home
    result["oracle_pick_home"] = base_pick | (result.denver_home & covered)
    result["baseline"] = base_pick.eq(covered).astype(float) * 100
    result["candidate"] = result.candidate_pick_home.eq(covered).astype(float) * 100
    result["oracle"] = result.oracle_pick_home.eq(covered).astype(float) * 100
    result["delta"] = result.candidate - result.baseline
    result["oracle_delta"] = result.oracle - result.baseline
    return result


def main() -> None:
    match = find_matching_opener_evaluation(Path("artifacts"))
    if match is None:
        raise ValueError("No active-model-matching opener evaluation")
    metadata, source = match
    schedules_path = sorted(Path("data/raw").glob("*/schedules.parquet"))[-1]
    games = pd.read_parquet(schedules_path)
    snapshot = latest_pbp_snapshot(Path("data/pbp/raw"))
    columns = [
        "game_id",
        "play_id",
        "season_type",
        "qtr",
        "posteam",
        "home_team",
        "away_team",
        "score_differential",
        "game_seconds_remaining",
    ]
    panels = [pd.read_parquet(snapshot.season_path(s), columns=columns) for s in range(2013, 2026)]
    quarters = quarter_margins(pd.concat(panels, ignore_index=True), games)
    quarters["denver_fourth"] = quarters.fourth_margin.where(denver_home_flag(quarters))
    descriptive = blocked_means(quarters, ["fourth_margin", "denver_fourth"])
    descriptive["denver"] = blocked_means(
        quarters.loc[denver_home_flag(quarters)],
        ["first_three_margin", "fourth_margin", "late_minus_early", "late_minus_early_rate"],
    )
    descriptive["split_half"] = split_half(quarters)
    descriptive["excluded_games"] = int(
        ((games.season.between(2013, 2025)) & games.game_type.eq("REG")).sum() - len(quarters)
    )
    predictions = pd.read_parquet(source / "per_game.parquet")
    predictions = predictions.loc[predictions.season.between(2020, 2025)]
    paired = paired_overlay(predictions, games)
    metrics = blocked_means(paired, ["baseline", "candidate", "oracle", "delta", "oracle_delta"])
    config = {
        "samples": SAMPLES,
        "seed": SEED,
        "predeclaration": "docs/coaching_leads.md",
        "opener_source": str(source),
        "pbp_source": str(snapshot.root),
        "schedules_source": str(schedules_path),
        "active_model_id": metadata["active_model_id"],
        "feature_table_sha256": metadata["feature_table_sha256"],
        "grade": "opener_probability_rule",
        "screen_seasons": [2020, 2025],
        "descriptive_seasons": [2013, 2025],
    }
    payload = {
        "configuration": config,
        "descriptive": descriptive,
        "metrics": metrics,
        "paired_games": len(paired),
        "paired_weeks": paired.groupby(["season", "week"]).ngroups,
        "denver_games": int(paired.denver_home.sum()),
        "changed_picks": int(
            paired.candidate_pick_home.ne(paired.pick_home_at_open_probability_rule).sum()
        ),
        "per_season": {
            str(s): blocked_means(g, ["baseline", "candidate", "delta"])
            for s, g in paired.groupby("season")
        },
        "provenance": artifact_provenance(
            config, Path("data/processed/game_features_weak_stack.parquet")
        ),
    }
    write_experiment_artifact(
        OUTPUT,
        "results.json",
        payload,
        command="coaching-altitude-leads-on-production",
        metrics=metrics["delta"],
        registry_root=OUTPUT / "experiment_registry",
    )
    paired.to_csv(OUTPUT / "paired_predictions.csv", index=False)
    quarters.to_csv(OUTPUT / "quarter_margins.csv", index=False)
    print(json.dumps({k: v for k, v in payload.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
