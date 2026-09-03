"""Special-teams player-rating reliability gate (PER-09 slice, frozen screen).

Predeclared in ``docs/st_player_ratings.md`` before any ST rating or
reliability number was computed. Population: participation-snapshot plays
2019-2024 whose personnel carries a K/P/LS token. Model: the frozen APM
recipe (Ridge alpha 1000, team effects at scale 11, EPA clip 5.0). Gate:
odd/even-week split-half correlation of per-player coefficients.
Measure-only: no ATS screen, no window, no wiring.

Writes ``artifacts/st_player_ratings/<stamp>/results.json`` via
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
from scipy.stats import spearmanr
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.io import run_id  # noqa: E402
from nfl_ats.participation import (  # noqa: E402
    PARTICIPATION_RATING_EPA_CLIP,
    PARTICIPATION_RATING_RIDGE_ALPHA,
    PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
    _player_ids,
    canonicalize_participation,
    latest_participation_snapshot,
    load_participation_snapshot,
)
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "st_player_ratings"

SOURCE_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)
MIN_PLAYS_PER_HALF = 50


def _personnel_tokens(personnel: Any) -> set[str]:
    """Split "1 C, 1 G, ..." into position tokens (regex-free: a global
    pattern match eats the ", " separator its neighbor needs to match)."""

    if pd.isna(personnel):
        return set()
    tokens: set[str] = set()
    for piece in str(personnel).split(","):
        parts = piece.strip().split()
        if len(parts) == 2 and parts[0] == "1":
            tokens.add(parts[1])
    return tokens


def classify_st_unit(offense_personnel: Any, defense_personnel: Any) -> str | None:
    """FG_XP / PUNT / KICKOFF / None from personnel tokens (frozen rule)."""

    tokens = _personnel_tokens(offense_personnel) | _personnel_tokens(defense_personnel)
    if not ({"K", "P", "LS"} & tokens):
        return None
    if "K" in tokens and "LS" in tokens:
        return "FG_XP"
    if "P" in tokens and "LS" in tokens:
        return "PUNT"
    if "K" in tokens:
        return "KICKOFF"
    return "OTHER_ST"


ST_PLAY_TYPES = frozenset({"kickoff", "punt", "extra_point", "field_goal"})


def build_st_play_table(participation: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Join ST-token plays to raw PBP EPA with 22-unique-ids validity.

    Reads the raw snapshot directly: ``analysis_plays`` keeps only
    pass/rush attempts and would drop every kick. The personnel-token
    classifier is primary; PBP ``play_type`` agreement is reported as a
    diagnostic cross-check, never a filter.
    """

    canonical = canonicalize_participation(participation)
    missing = sorted(
        {"game_id", "play_id", "season", "week", "posteam", "epa", "play_type"}.difference(
            pbp.columns
        )
    )
    if missing:
        raise DataContractError(f"PBP snapshot is missing columns: {', '.join(missing)}")
    plays = pbp.loc[
        :, ["game_id", "play_id", "season", "week", "posteam", "epa", "play_type"]
    ].copy()
    plays["play_id"] = pd.to_numeric(plays["play_id"], errors="raise").astype(int)
    plays["week"] = pd.to_numeric(plays["week"], errors="raise").astype(int)
    plays["epa"] = pd.to_numeric(plays["epa"], errors="coerce")
    plays = plays.loc[plays["epa"].notna()].copy()
    joined = canonical.merge(
        plays,
        on=["game_id", "play_id", "season"],
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise DataContractError("No participation plays match the PBP snapshot")
    joined["st_unit"] = [
        classify_st_unit(offense, defense)
        for offense, defense in zip(
            joined["offense_personnel"], joined["defense_personnel"], strict=True
        )
    ]
    joined = joined.loc[joined["st_unit"].notna()].copy()
    joined["side_a_ids"] = joined["offense_players"].map(_player_ids)
    joined["side_b_ids"] = joined["defense_players"].map(_player_ids)
    valid = (
        joined["side_a_ids"].map(len).eq(11)
        & joined["side_b_ids"].map(len).eq(11)
        & joined["side_a_ids"].map(lambda values: len(set(values))).eq(11)
        & joined["side_b_ids"].map(lambda values: len(set(values))).eq(11)
    )
    result = joined.loc[
        valid,
        [
            "season",
            "week",
            "game_id",
            "play_id",
            "posteam",
            "possession_team",
            "epa",
            "st_unit",
            "play_type",
            "side_a_ids",
            "side_b_ids",
        ],
    ].copy()
    if result.empty:
        raise DataContractError("No valid special-teams plays remain")
    return result.reset_index(drop=True)


def play_type_agreement(table: pd.DataFrame) -> dict[str, Any]:
    """Diagnostic cross-check: personnel unit vs PBP play_type (not a filter)."""

    expected = {"FG_XP": {"extra_point", "field_goal"}, "PUNT": {"punt"}, "KICKOFF": {"kickoff"}}
    rows = 0
    agree = 0
    by_unit: dict[str, dict[str, int]] = {}
    for row in table.itertuples(index=False):
        if row.st_unit == "OTHER_ST":
            continue
        rows += 1
        play_type = str(row.play_type)
        match = play_type in expected.get(row.st_unit, set())
        agree += int(match)
        cell = by_unit.setdefault(row.st_unit, {"agree": 0, "total": 0})
        cell["total"] += 1
        cell["agree"] += int(match)
    return {
        "rows": rows,
        "agreement_rate": (agree / rows) if rows else float("nan"),
        "by_unit": by_unit,
    }


def _design_rows(
    table: pd.DataFrame, *, team_feature_scale: float
) -> tuple[Any, np.ndarray, list[str], dict[str, int]]:
    """Sparse ST design: one coefficient per participant + team effects."""

    counts: Counter[str] = Counter()
    rows: list[dict[str, float]] = []
    for row in table.itertuples(index=False):
        side_a = tuple(row.side_a_ids)
        side_b = tuple(row.side_b_ids)
        counts.update(side_a)
        counts.update(side_b)
        values = {f"st_player::{player_id}": 1.0 for player_id in side_a}
        values.update({f"st_player::{player_id}": -1.0 for player_id in side_b})
        values[f"st_team::{row.posteam}"] = team_feature_scale
        rows.append(values)
    vectorizer = DictVectorizer(sparse=True, sort=True)
    matrix = vectorizer.fit_transform(rows)
    return matrix, np.asarray(vectorizer.get_feature_names_out()), dict(counts)


def fit_st_coefficients(
    table: pd.DataFrame,
    *,
    ridge_alpha: float = PARTICIPATION_RATING_RIDGE_ALPHA,
    team_feature_scale: float = PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
    epa_clip: float = PARTICIPATION_RATING_EPA_CLIP,
) -> tuple[dict[str, float], dict[str, int]]:
    """Ridge ST coefficients on one play table (deterministic, lsqr)."""

    matrix, feature_names, counts = _design_rows(table, team_feature_scale=team_feature_scale)
    target = (
        pd.to_numeric(table["epa"], errors="raise").clip(-epa_clip, epa_clip).to_numpy(dtype=float)
    )
    estimator = Ridge(alpha=ridge_alpha, solver="lsqr", fit_intercept=True, tol=1e-6)
    estimator.fit(matrix, target)
    return dict(zip(feature_names, np.asarray(estimator.coef_), strict=True)), counts


def split_half_reliability(
    table: pd.DataFrame, *, min_plays_per_half: int = MIN_PLAYS_PER_HALF, **fit_kwargs: Any
) -> dict[str, Any]:
    """Odd/even-week split fits correlated over well-observed players."""

    if "week" not in table.columns:
        raise DataContractError("ST play table needs a week column for split-half")
    odd = table.loc[table["week"].astype(int).mod(2).eq(1)].copy()
    even = table.loc[table["week"].astype(int).mod(2).eq(0)].copy()
    if odd.empty or even.empty:
        raise DataContractError("ST split-half needs nonempty odd and even halves")
    odd_coef, odd_counts = fit_st_coefficients(odd, **fit_kwargs)
    even_coef, even_counts = fit_st_coefficients(even, **fit_kwargs)
    players = sorted(
        player_id
        for player_id in set(odd_counts).intersection(even_counts)
        if odd_counts[player_id] >= min_plays_per_half
        and even_counts[player_id] >= min_plays_per_half
    )
    if len(players) < 3:
        return {"n": len(players), "insufficient": True}
    x = np.array([odd_coef[f"st_player::{player_id}"] for player_id in players])
    y = np.array([even_coef[f"st_player::{player_id}"] for player_id in players])
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else float("nan")
    spearman = float(spearmanr(x, y).statistic)
    spearman_brown_pearson = 2.0 * pearson / (1.0 + pearson) if pearson > 0 else pearson
    return {
        "n": len(players),
        "split_half_pearson": pearson,
        "split_half_spearman": spearman,
        "spearman_brown_pearson": float(spearman_brown_pearson),
        "min_plays_per_half": min_plays_per_half,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(SOURCE_SEASONS))
    args = parser.parse_args()
    started = time.time()
    requested = sorted({int(season) for season in args.seasons})
    if max(requested) > max(SOURCE_SEASONS):
        raise DataContractError(
            f"source seasons {requested} exceed the predeclared {list(SOURCE_SEASONS)}; "
            "extending the window needs a new predeclaration"
        )
    snapshot = latest_participation_snapshot(REPO / "data" / "players" / "participation" / "raw")
    participation = load_participation_snapshot(snapshot)
    participation = participation.loc[participation["season"].astype(int).isin(requested)].copy()
    pbp_snapshot = latest_pbp_snapshot(REPO / "data" / "pbp" / "raw")
    pbp = load_pbp_snapshot(pbp_snapshot)
    table = build_st_play_table(participation, pbp)
    _full_coef, full_counts = fit_st_coefficients(table)
    reliability = split_half_reliability(table)
    agreement = play_type_agreement(table)
    units = table["st_unit"].value_counts().to_dict()
    configuration = {
        "command": "st-player-ratings-screen",
        "participation_snapshot": snapshot.snapshot_id,
        "pbp_snapshot": pbp_snapshot.snapshot_id,
        "seasons": requested,
        "ridge_alpha": PARTICIPATION_RATING_RIDGE_ALPHA,
        "team_feature_scale": PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
        "epa_clip": PARTICIPATION_RATING_EPA_CLIP,
        "min_plays_per_half": MIN_PLAYS_PER_HALF,
        "predeclaration": "docs/st_player_ratings.md (frozen before scoring)",
    }
    payload = {
        "st_plays": len(table),
        "unit_counts": {str(key): int(value) for key, value in units.items()},
        "players": len(full_counts),
        "reliability": reliability,
        "play_type_agreement": agreement,
        "elapsed_seconds": time.time() - started,
        "provenance": artifact_provenance(configuration, snapshot.manifest_path, project_root=REPO),
    }
    output_dir = args.output or (OUT_ROOT / run_id())
    output_dir.mkdir(parents=True, exist_ok=False)
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="st-player-ratings-screen",
        metrics=payload,
        notes=(
            "Measure-only ST-APM reliability gate (PER-09 slice); predeclared "
            "cells record via separate nfl-ats weak-signals record calls "
            "regardless of outcome shape (AGENTS.md)."
        ),
    )
    print(f"st_plays={len(table)} reliability={reliability}")
    print(f"wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
