"""Hierarchical APM vs flat on held-out EPA (PER-09 slice, frozen screen).

Predeclared in ``docs/hierarchical_apm.md`` before any comparison number
was computed. Expanding-origin held-out seasons {2022, 2023, 2024}, fits on
[Y-3, Y-1]: flat pooled APM vs flat-plus-unit-mean shrinkage (k=500).
Metric: pooled MSE difference with a play-level paired bootstrap.
Measure-only: no ATS screen, no window, no wiring, no registry entry (no
honest effect unit exists for an MSE delta).

Writes ``artifacts/hierarchical_apm/<stamp>/results.json`` via
``write_experiment_artifact``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[1]
for _path in (str(REPO), str(REPO / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.io import run_id  # noqa: E402
from nfl_ats.participation import (  # noqa: E402
    PARTICIPATION_RATING_EPA_CLIP,
    PARTICIPATION_RATING_RIDGE_ALPHA,
    PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
    _player_ids,
    build_participation_play_table,
    latest_participation_snapshot,
    load_participation_snapshot,
)
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot  # noqa: E402
from nfl_ats.provenance import (  # noqa: E402
    configuration_hash,
    git_state,
    write_experiment_artifact,
)

OUT_ROOT = REPO / "artifacts" / "hierarchical_apm"
HELDOUT_SEASONS = (2022, 2023, 2024)
LOOKBACK_SEASONS = 3
SHRINKAGE_K_PLAYS = 500.0
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260903


def fit_flat_coefficients(
    table: pd.DataFrame,
    *,
    ridge_alpha: float = PARTICIPATION_RATING_RIDGE_ALPHA,
    team_feature_scale: float = PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
    epa_clip: float = PARTICIPATION_RATING_EPA_CLIP,
) -> dict[str, float]:
    """Pooled APM fit (the frozen recipe, no units)."""

    rows: list[dict[str, float]] = []
    for row in table.itertuples(index=False):
        values = {
            f"offense_player::{player_id}": 1.0 for player_id in _player_ids(row.offense_players)
        }
        values.update(
            {f"defense_player::{player_id}": -1.0 for player_id in _player_ids(row.defense_players)}
        )
        values[f"offense_team::{row.posteam}"] = team_feature_scale
        values[f"defense_team::{row.defteam}"] = -team_feature_scale
        rows.append(values)
    vectorizer = DictVectorizer(sparse=True, sort=True)
    matrix = vectorizer.fit_transform(rows)
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    target = (
        pd.to_numeric(table["epa"], errors="raise").clip(-epa_clip, epa_clip).to_numpy(dtype=float)
    )
    estimator = Ridge(alpha=ridge_alpha, solver="lsqr", fit_intercept=True, tol=1e-6)
    estimator.fit(matrix, target)
    return {
        "intercept": float(estimator.intercept_),
        **dict(zip(feature_names, np.asarray(estimator.coef_), strict=True)),
    }


def apply_hierarchical_shrinkage(
    flat_coefficients: dict[str, float],
    unit_lookup: dict[tuple[str, int], str],
    play_counts: Counter[str],
    season: int,
    *,
    shrinkage_k: float = SHRINKAGE_K_PLAYS,
) -> dict[str, float]:
    """Shrink each player's coefficient toward their unit mean.

    Weight n/(n+k) on the player, k/(n+k) on the unit mean; players without
    a unit pass through untouched. Team effects and intercept pass through
    untouched. Deterministic.
    """

    if shrinkage_k < 0:
        raise DataContractError(f"shrinkage k must be nonnegative, got {shrinkage_k}")
    unit_members: dict[str, list[str]] = {}
    for (player_id, player_season), unit in unit_lookup.items():
        if player_season == season:
            unit_members.setdefault(unit, []).append(player_id)
    shrunk = dict(flat_coefficients)
    for members in unit_members.values():
        # Offense and defense coefficients pool separately: a two-way
        # player's tackle-avoidance and coverage effects must never average
        # into one number.
        for prefix in ("offense_player", "defense_player"):
            keys = [f"{prefix}::{player_id}" for player_id in members]
            present = [key for key in keys if key in flat_coefficients]
            if not present:
                continue
            mean = float(np.mean([flat_coefficients[key] for key in present]))
            for key in present:
                player_id = key.split("::", 1)[1]
                plays = play_counts.get(player_id, 0)
                weight = plays / (plays + shrinkage_k) if (plays + shrinkage_k) > 0 else 1.0
                shrunk[key] = weight * flat_coefficients[key] + (1.0 - weight) * mean
    return shrunk


def predict_epa(
    table: pd.DataFrame, coefficients: dict[str, float], *, team_feature_scale: float
) -> np.ndarray:
    """Score held-out plays under one coefficient set."""

    predictions = np.full(len(table), coefficients.get("intercept", 0.0))
    for row_index, row in enumerate(table.itertuples(index=False)):
        total = 0.0
        for player_id in _player_ids(row.offense_players):
            total += coefficients.get(f"offense_player::{player_id}", 0.0)
        for player_id in _player_ids(row.defense_players):
            total -= coefficients.get(f"defense_player::{player_id}", 0.0)
        total += team_feature_scale * (
            coefficients.get(f"offense_team::{row.posteam}", 0.0)
            - coefficients.get(f"defense_team::{row.defteam}", 0.0)
        )
        predictions[row_index] += total
    return predictions


def paired_mse_comparison(
    actual: np.ndarray, flat_pred: np.ndarray, hier_pred: np.ndarray, *, seed: int, samples: int
) -> dict[str, Any]:
    """Pooled MSE difference (flat minus hierarchical) with paired bootstrap."""

    flat_se = (actual - flat_pred) ** 2
    hier_se = (actual - hier_pred) ** 2
    delta = float(np.mean(flat_se) - np.mean(hier_se))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for draw in range(samples):
        idx = rng.integers(0, len(actual), size=len(actual))
        draws[draw] = float(np.mean(flat_se[idx]) - np.mean(hier_se[idx]))
    return {
        "mse_flat": float(np.mean(flat_se)),
        "mse_hierarchical": float(np.mean(hier_se)),
        "mse_delta_flat_minus_hier": delta,
        "mse_delta_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_hierarchical_better": float(np.mean(draws > 0.0)),
        "samples": samples,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    started = time.time()
    snapshot = latest_participation_snapshot(REPO / "data" / "players" / "participation" / "raw")
    participation = load_participation_snapshot(snapshot)
    pbp_snapshot = latest_pbp_snapshot(REPO / "data" / "pbp" / "raw")
    pbp = load_pbp_snapshot(pbp_snapshot)
    full_table = build_participation_play_table(participation, pbp)
    rosters = pd.read_parquet(
        REPO / "data" / "players" / "raw" / "20260817T184901Z" / "weekly_rosters.parquet"
    )
    from scripts.unit_apm_screen import UNIT_BY_POSITION

    rosters = rosters.loc[rosters["gsis_id"].notna()].copy()
    rosters["unit"] = rosters["position"].astype(str).str.strip().str.upper().map(UNIT_BY_POSITION)
    seasons: dict[str, Any] = {}
    for heldout in HELDOUT_SEASONS:
        train = full_table.loc[
            full_table["season"].astype(int).ge(heldout - LOOKBACK_SEASONS)
            & full_table["season"].astype(int).le(heldout - 1)
        ].copy()
        test = full_table.loc[full_table["season"].astype(int).eq(heldout)].copy()
        if train.empty or test.empty:
            raise DataContractError(f"empty train/test split for heldout {heldout}")
        flat = fit_flat_coefficients(train)
        leaky = fit_flat_coefficients(pd.concat([train, test], ignore_index=True))
        leaky_pred = predict_epa(
            test, leaky, team_feature_scale=PARTICIPATION_RATING_TEAM_FEATURE_SCALE
        )
        leaky_actual = (
            pd.to_numeric(test["epa"], errors="raise")
            .clip(-PARTICIPATION_RATING_EPA_CLIP, PARTICIPATION_RATING_EPA_CLIP)
            .to_numpy(dtype=float)
        )
        leaky_mse = float(np.mean((leaky_actual - leaky_pred) ** 2))
        counts: Counter[str] = Counter()
        for row in train.itertuples(index=False):
            counts.update(_player_ids(row.offense_players))
            counts.update(_player_ids(row.defense_players))
        lookup = {
            (str(gsis_id), int(season)): str(unit)
            for gsis_id, season, unit in zip(
                rosters["gsis_id"], rosters["season"], rosters["unit"], strict=True
            )
            if pd.notna(unit)
        }
        hier = apply_hierarchical_shrinkage(flat, lookup, counts, heldout)
        actual = (
            pd.to_numeric(test["epa"], errors="raise")
            .clip(-PARTICIPATION_RATING_EPA_CLIP, PARTICIPATION_RATING_EPA_CLIP)
            .to_numpy(dtype=float)
        )
        team_scale = PARTICIPATION_RATING_TEAM_FEATURE_SCALE
        flat_pred = predict_epa(test, flat, team_feature_scale=team_scale)
        hier_pred = predict_epa(test, hier, team_feature_scale=team_scale)
        seasons[str(heldout)] = {
            "train_rows": len(train),
            "test_rows": len(test),
            "leaky_control_mse": leaky_mse,
            **paired_mse_comparison(
                actual,
                flat_pred,
                hier_pred,
                seed=BOOTSTRAP_SEED,
                samples=BOOTSTRAP_SAMPLES,
            ),
        }
    pooled_delta = float(
        np.mean([seasons[str(year)]["mse_delta_flat_minus_hier"] for year in HELDOUT_SEASONS])
    )
    configuration = {
        "command": "hierarchical-apm-screen",
        "participation_snapshot": snapshot.snapshot_id,
        "pbp_snapshot": pbp_snapshot.snapshot_id,
        "heldout_seasons": list(HELDOUT_SEASONS),
        "lookback_seasons": LOOKBACK_SEASONS,
        "shrinkage_k_plays": SHRINKAGE_K_PLAYS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "predeclaration": "docs/hierarchical_apm.md (frozen before scoring)",
    }
    payload = {
        "seasons": seasons,
        "pooled_mse_delta_flat_minus_hier": pooled_delta,
        "elapsed_seconds": time.time() - started,
        "provenance": {
            "configuration": configuration,
            "configuration_sha256": configuration_hash(configuration),
            "code": git_state(REPO),
        },
    }
    output_dir = args.output or (OUT_ROOT / run_id())
    output_dir.mkdir(parents=True, exist_ok=False)
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="hierarchical-apm-screen",
        metrics=payload,
        notes=(
            "Measure-only hierarchical-vs-flat EPA comparison (PER-09 "
            "slice); no registry entry exists for an MSE delta and none "
            "was forced (AGENTS.md unit-honesty rule)."
        ),
    )
    print(f"pooled_mse_delta={pooled_delta:.6f}")
    print(f"wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
