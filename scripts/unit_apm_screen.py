"""Unit-level APM reliability gates (PER-09 slice, frozen screen).

Predeclared in ``docs/unit_apm_ratings.md`` before any unit-level rating or
reliability number was computed. Population: the SAME valid competitive
11-on-11 scrimmage table the pooled screen uses
(``build_participation_play_table``), seasons 2019-2024. Four separate Ridge
fits (alpha 1000, team effects at scale 11, EPA clip 5.0), one per roster
unit, each seeing only its unit's players. Gate: odd/even-week split-half
correlation per unit. Measure-only: no ATS screen, no window, no wiring.

Writes ``artifacts/unit_apm/<stamp>/results.json`` via
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

OUT_ROOT = REPO / "artifacts" / "unit_apm"
SOURCE_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)
MIN_PLAYS_PER_HALF = 50

#: Frozen roster-position to unit mapping (docs/unit_apm_ratings.md §2).
UNIT_BY_POSITION = {
    **dict.fromkeys(["C", "G", "T", "OT", "OG", "OC", "OL"], "OFF_OL"),
    **dict.fromkeys(["QB", "RB", "FB", "HB", "WR", "TE"], "OFF_SKILL"),
    **dict.fromkeys(
        ["DE", "DT", "NT", "LB", "OLB", "ILB", "MLB", "DL", "EDGE", "LDE", "RDE", "LDT", "RDT"],
        "DEF_FRONT",
    ),
    **dict.fromkeys(["CB", "S", "SS", "FS", "DB", "NB", "LCB", "RCB", "SAF"], "DEF_SECONDARY"),
}
UNITS = ("OFF_OL", "OFF_SKILL", "DEF_FRONT", "DEF_SECONDARY")
UNIT_SIDE = {
    "OFF_OL": "offense",
    "OFF_SKILL": "offense",
    "DEF_FRONT": "defense",
    "DEF_SECONDARY": "defense",
}


def modal_unit_by_player_season(rosters: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """One unit per (player, season) by modal roster position.

    Position switches across seasons are real: the unit is resolved per
    season, so a convert belongs to each season's unit for that season's
    plays. Unmapped positions are excluded and counted, never guessed.
    """

    missing = sorted({"gsis_id", "season", "position"}.difference(rosters.columns))
    if missing:
        raise DataContractError(f"rosters are missing columns: {', '.join(missing)}")
    frame = rosters.loc[rosters["gsis_id"].notna()].copy()
    frame["gsis_id"] = frame["gsis_id"].astype(str)
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame = frame.loc[frame["season"].notna()].copy()
    frame["season"] = frame["season"].astype(int)
    frame["position"] = frame["position"].astype(str).str.strip().str.upper()
    frame["unit"] = frame["position"].map(UNIT_BY_POSITION)
    unmapped = frame.loc[frame["unit"].isna(), "position"].value_counts().to_dict()
    frame = frame.loc[frame["unit"].notna()].copy()
    modal = (
        frame.groupby(["gsis_id", "season"], sort=False)["unit"]
        .agg(lambda values: values.mode().iloc[0])
        .reset_index()
    )
    return modal, {str(position): int(count) for position, count in unmapped.items()}


UnitLookup = dict[tuple[str, int], str]


def build_unit_lookup(units: pd.DataFrame) -> UnitLookup:
    """(gsis_id, season) -> unit, from the modal roster mapping."""

    return {
        (str(gsis_id), int(season)): str(unit)
        for gsis_id, season, unit in zip(
            units["gsis_id"], units["season"], units["unit"], strict=True
        )
    }


def fit_unit_coefficients(
    table: pd.DataFrame,
    unit: str,
    unit_lookup: UnitLookup,
    *,
    ridge_alpha: float = PARTICIPATION_RATING_RIDGE_ALPHA,
    team_feature_scale: float = PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
    epa_clip: float = PARTICIPATION_RATING_EPA_CLIP,
) -> tuple[dict[str, float], dict[str, int]]:
    """Ridge unit-APM coefficients on one play table (deterministic, lsqr).

    Only side-players whose modal roster unit for that season equals
    ``unit`` enter the design; everyone else on the play is context the
    team effects absorb.
    """

    side = UNIT_SIDE[unit]
    player_column = "offense_players" if side == "offense" else "defense_players"
    rows: list[dict[str, float]] = []
    counts: Counter[str] = Counter()
    for row in table.itertuples(index=False):
        season = int(row.season)
        members = [
            player
            for player in _player_ids(getattr(row, player_column))
            if unit_lookup.get((player, season)) == unit
        ]
        counts.update(members)
        values = {f"unit_player::{player_id}": 1.0 for player_id in members}
        if side == "offense":
            values[f"unit_team::{row.posteam}"] = team_feature_scale
        else:
            values[f"unit_team::{row.defteam}"] = -team_feature_scale
        rows.append(values)
    vectorizer = DictVectorizer(sparse=True, sort=True)
    matrix = vectorizer.fit_transform(rows)
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    target = (
        pd.to_numeric(table["epa"], errors="raise").clip(-epa_clip, epa_clip).to_numpy(dtype=float)
    )
    if side == "defense":
        target = -target
    estimator = Ridge(alpha=ridge_alpha, solver="lsqr", fit_intercept=True, tol=1e-6)
    estimator.fit(matrix, target)
    return dict(zip(feature_names, np.asarray(estimator.coef_), strict=True)), dict(counts)


def split_half_reliability(
    table: pd.DataFrame,
    unit: str,
    unit_lookup: UnitLookup,
    *,
    min_plays_per_half: int = MIN_PLAYS_PER_HALF,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    """Odd/even-week split fits correlated over well-observed unit members."""

    if "week" not in table.columns:
        raise DataContractError("unit play table needs a week column for split-half")
    odd = table.loc[table["week"].astype(int).mod(2).eq(1)].copy()
    even = table.loc[table["week"].astype(int).mod(2).eq(0)].copy()
    if odd.empty or even.empty:
        raise DataContractError("unit split-half needs nonempty odd and even halves")
    odd_coef, odd_counts = fit_unit_coefficients(odd, unit, unit_lookup, **fit_kwargs)
    even_coef, even_counts = fit_unit_coefficients(even, unit, unit_lookup, **fit_kwargs)
    players = sorted(
        player_id
        for player_id in set(odd_counts).intersection(even_counts)
        if odd_counts[player_id] >= min_plays_per_half
        and even_counts[player_id] >= min_plays_per_half
    )
    if len(players) < 3:
        return {"unit": unit, "n": len(players), "insufficient": True}
    x = np.array([odd_coef[f"unit_player::{player_id}"] for player_id in players])
    y = np.array([even_coef[f"unit_player::{player_id}"] for player_id in players])
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else float("nan")
    spearman = float(spearmanr(x, y).statistic)
    return {
        "unit": unit,
        "n": len(players),
        "split_half_pearson": pearson,
        "split_half_spearman": spearman,
        "spearman_brown_pearson": float(2.0 * pearson / (1.0 + pearson))
        if pearson > 0
        else pearson,
        "min_plays_per_half": min_plays_per_half,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(SOURCE_SEASONS))
    parser.add_argument(
        "--units",
        nargs="+",
        default=list(UNITS),
        choices=list(UNITS),
        help="subset of units to score (default: all four)",
    )
    args = parser.parse_args()
    for unit in args.units:
        if unit not in UNITS:
            raise DataContractError(f"unknown unit {unit!r}; expected one of {list(UNITS)}")
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
    table = build_participation_play_table(participation, pbp)
    week_map = pbp.loc[:, ["game_id", "play_id", "season", "week"]].copy()
    week_map["play_id"] = pd.to_numeric(week_map["play_id"], errors="coerce")
    table = table.merge(week_map, on=["game_id", "play_id", "season"], how="left")
    if table["week"].isna().any():
        raise DataContractError("unit play table has plays without a PBP week")
    rosters = pd.read_parquet(
        REPO / "data" / "players" / "raw" / "20260817T184901Z" / "weekly_rosters.parquet"
    )
    units, unmapped = modal_unit_by_player_season(rosters)
    lookup = build_unit_lookup(units)
    selected = [unit for unit in UNITS if unit in set(args.units)]
    results: dict[str, Any] = {}
    for unit in selected:
        results[unit] = split_half_reliability(table, unit, lookup)
    configuration = {
        "command": "unit-apm-screen",
        "participation_snapshot": snapshot.snapshot_id,
        "pbp_snapshot": pbp_snapshot.snapshot_id,
        "seasons": requested,
        "units": selected,
        "ridge_alpha": PARTICIPATION_RATING_RIDGE_ALPHA,
        "team_feature_scale": PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
        "epa_clip": PARTICIPATION_RATING_EPA_CLIP,
        "min_plays_per_half": MIN_PLAYS_PER_HALF,
        "predeclaration": "docs/unit_apm_ratings.md (frozen before scoring)",
    }
    payload = {
        "unmapped_positions": unmapped,
        "units": results,
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
        command="unit-apm-screen",
        metrics=payload,
        notes=(
            "Measure-only unit-APM reliability gates (PER-09 slice); "
            "predeclared cells record via separate nfl-ats weak-signals "
            "record calls regardless of outcome shape (AGENTS.md)."
        ),
    )
    print(f"units={list(results)}")
    print(f"wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
