"""Joint-absence dependence by unit (PER-10/SIM-03 slice, frozen screen).

Predeclared in ``docs/absence_dependence.md`` before any joint-absence
number was computed. Measures per-unit excess multi-player non-participation
vs the independence-implied rate, with a team-season block bootstrap and a
permutation diagnostic. Descriptive: no ATS screen, no window, no registry
verdict.

Writes ``artifacts/absence_dependence/<stamp>/results.json`` via
``write_experiment_artifact``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _path in (str(REPO), str(REPO / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.io import run_id  # noqa: E402
from nfl_ats.players import (  # noqa: E402
    attach_snap_player_ids,
    canonicalize_rosters,
    canonicalize_snaps,
)
from nfl_ats.provenance import (  # noqa: E402
    configuration_hash,
    git_state,
    write_experiment_artifact,
)
from scripts.unit_apm_screen import UNIT_BY_POSITION  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "absence_dependence"
SEASONS = tuple(range(2016, 2026))
TRAILING_GAMES = 4
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260903
PERMUTATIONS = 200
PERMUTATION_SEED = 20260904

UNIT_SIDE_SNAPS = {
    "OFF_OL": "offense_snaps",
    "OFF_SKILL": "offense_snaps",
    "DEF_FRONT": "defense_snaps",
    "DEF_SECONDARY": "defense_snaps",
}


def load_unit_rosters(player_snapshot: Path) -> pd.DataFrame:
    """(gsis_id, season, week, team, unit) roster rows with mapped units."""

    rosters = canonicalize_rosters(pd.read_parquet(player_snapshot / "weekly_rosters.parquet"))
    frame = rosters.loc[rosters["gsis_id"].notna()].copy()
    frame["gsis_id"] = frame["gsis_id"].astype(str)
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype(int)
    frame["unit"] = frame["position"].astype(str).str.strip().str.upper().map(UNIT_BY_POSITION)
    return frame.loc[frame["unit"].notna(), ["gsis_id", "season", "week", "team", "unit"]].copy()


def build_contributor_games(
    snaps: pd.DataFrame, rosters: pd.DataFrame, *, trailing_games: int = TRAILING_GAMES
) -> pd.DataFrame:
    """One row per rostered player-game with trailing activity and absence.

    Established contributor = side-appropriate snaps in ≥1 of the trailing 4
    team games (same season). Absence = zero side snaps this game. REG only.
    """

    required = {"gsis_id", "season", "week", "team", "offense_snaps", "defense_snaps", "game_type"}
    missing = sorted(required.difference(snaps.columns))
    if missing:
        raise DataContractError(f"snaps are missing columns: {', '.join(missing)}")
    reg = snaps.loc[snaps["game_type"].astype(str).eq("REG")].copy()
    reg["season"] = pd.to_numeric(reg["season"], errors="coerce").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="coerce").astype(int)
    for column in ("offense_snaps", "defense_snaps"):
        reg[column] = pd.to_numeric(reg[column], errors="coerce").fillna(0.0)
    base = rosters.merge(
        reg.loc[:, ["gsis_id", "season", "week", "team", "offense_snaps", "defense_snaps"]],
        on=["gsis_id", "season", "week", "team"],
        how="left",
    )
    base[["offense_snaps", "defense_snaps"]] = base[["offense_snaps", "defense_snaps"]].fillna(0.0)
    base = base.sort_values(["gsis_id", "season", "team", "week"]).reset_index(drop=True)
    base["side_snaps"] = np.where(
        base["unit"].str.startswith("OFF"), base["offense_snaps"], base["defense_snaps"]
    )
    # Window covers the current row plus predecessors; subtracting the
    # current row leaves exactly the trailing ``trailing_games`` predecessors.
    prior = (
        base.groupby(["gsis_id", "season", "team", "unit"], sort=False)["side_snaps"]
        .rolling(trailing_games + 1, min_periods=1)
        .sum()
        .reset_index(level=[0, 1, 2, 3], drop=True)
    )
    base["trailing_snaps"] = prior - base["side_snaps"]
    established = base.loc[base["trailing_snaps"].gt(0)].copy()
    established["absent"] = established["side_snaps"].le(0.0)
    return established.reset_index(drop=True)


def unit_game_table(contributors: pd.DataFrame, unit: str) -> pd.DataFrame:
    """One row per team-game: season, team, week, unit size, absence count."""

    frame = contributors.loc[contributors["unit"].eq(unit)].copy()
    if frame.empty:
        raise DataContractError(f"no contributor games for unit {unit}")
    grouped = frame.groupby(["season", "team", "week"], sort=False)
    table = pd.DataFrame(
        {
            "block": grouped["season"].first().astype(str)
            + "_"
            + grouped["team"].first().astype(str),
            "size": grouped.size(),
            "absent": grouped["absent"].sum(),
        }
    ).reset_index(drop=True)
    table = table.loc[table["size"].ge(2)].copy()
    if table.empty:
        raise DataContractError(f"unit {unit} has no team-games with 2+ contributors")
    return table.reset_index(drop=True)


def full_sit_stats(sizes: np.ndarray, absence_counts: np.ndarray, q: float) -> dict[str, float]:
    """Observed shares plus the full-sit excess from game-level arrays.

    The full-sit rate (every contributor absent) is the primary estimand:
    it is monotone in coupling strength and is the whole-lineup state the
    mixture kernel scores. A P(2+) ratio is also reported descriptively but
    points the wrong way under strong coupling (see the predeclaration
    amendment note) and carries no claim.
    """

    k = sizes.astype(float)
    full_sits = absence_counts == sizes
    implied_full = float(np.mean(q**k))
    return {
        "team_games": len(sizes),
        "p0": float(np.mean(absence_counts == 0)),
        "p1": float(np.mean(absence_counts == 1)),
        "p2plus": float(np.mean(absence_counts >= 2)),
        "marginal_q": float(q),
        "implied_p2plus": float(
            np.mean(1.0 - (1.0 - q) ** k - k * q * (1.0 - q) ** np.maximum(k - 1, 0))
        ),
        "p_full_sit": float(np.mean(full_sits)),
        "implied_p_full_sit": implied_full,
        "excess_ratio": (float(np.mean(full_sits)) / implied_full)
        if implied_full > 0
        else float("nan"),
    }


def unit_excess(contributors: pd.DataFrame, unit: str) -> dict[str, Any]:
    """Observed shares plus the full-sit excess for one unit (see
    :func:`full_sit_stats`)."""

    table = unit_game_table(contributors, unit)
    q = float(contributors.loc[contributors["unit"].eq(unit), "absent"].mean())
    return full_sit_stats(table["size"].to_numpy(), table["absent"].to_numpy(), q)


def bootstrap_excess_ratio(
    contributors: pd.DataFrame,
    unit: str,
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Team-season block bootstrap CI for the excess ratio.

    Resamples whole team-season blocks of the compact game table (not
    player rows), so 2,000 draws stay in numpy-land.
    """

    table = unit_game_table(contributors, unit)
    q = float(contributors.loc[contributors["unit"].eq(unit), "absent"].mean())
    blocks = table["block"].to_numpy()
    sizes = table["size"].to_numpy()
    absent = table["absent"].to_numpy()
    unique_blocks = np.unique(blocks)
    block_rows = {block: np.flatnonzero(blocks == block) for block in unique_blocks}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for draw in range(samples):
        chosen = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([block_rows[block] for block in chosen])
        draws[draw] = full_sit_stats(sizes[idx], absent[idx], q)["excess_ratio"]
    draws = draws[np.isfinite(draws)]
    point = full_sit_stats(sizes, absent, q)["excess_ratio"]
    return {
        "excess_ratio": float(point),
        "excess_ratio_ci95": (
            [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
            if len(draws)
            else [float("nan"), float("nan")]
        ),
        "samples": samples,
        "seed": seed,
        "blocks": len(unique_blocks),
    }


def permutation_diagnostic(
    contributors: pd.DataFrame,
    unit: str,
    *,
    seed: int = PERMUTATION_SEED,
    permutations: int = PERMUTATIONS,
) -> dict[str, Any]:
    """Shuffle absence labels within team-seasons: heterogeneous marginals
    alone must not reproduce the observed excess.

    Permutes player-game flags with numpy inside each team-season group and
    rebuilds game counts vectorized — no per-draw frame copies.
    """

    frame = contributors.loc[contributors["unit"].eq(unit)].copy()
    frame["block"] = frame["season"].astype(str) + "_" + frame["team"].astype(str)
    frame = frame.sort_values(["block", "season", "week"]).reset_index(drop=True)
    flags = frame["absent"].to_numpy(dtype=bool)
    game_index = frame.groupby(["season", "team", "week"], sort=False).ngroup().to_numpy()
    block_of_row = frame["block"].to_numpy()
    unique_blocks = np.unique(block_of_row)
    block_masks = {block: np.flatnonzero(block_of_row == block) for block in unique_blocks}
    q = float(flags.mean())
    observed = full_sit_stats(
        *_game_counts(flags, game_index),
        q,
    )["excess_ratio"]
    rng = np.random.default_rng(seed)
    nulls = np.empty(permutations)
    for draw in range(permutations):
        shuffled = flags.copy()
        for rows in block_masks.values():
            shuffled[rows] = rng.permutation(shuffled[rows])
        nulls[draw] = full_sit_stats(*_game_counts(shuffled, game_index), q)["excess_ratio"]
    return {
        "null_center": float(np.mean(nulls)),
        "observed_percentile": float(np.mean(nulls <= observed)),
        "permutations": permutations,
        "seed": seed,
    }


def _game_counts(flags: np.ndarray, game_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sizes, absence counts) per team-game for eligible (2+) games."""

    order = np.argsort(game_index, kind="stable")
    sorted_flags = flags[order]
    sorted_games = game_index[order]
    boundaries = np.flatnonzero(np.diff(sorted_games, prepend=sorted_games[0] - 1))
    sizes = np.diff(np.append(boundaries, len(sorted_games)))
    counts = np.add.reduceat(sorted_flags.astype(int), boundaries)
    eligible = sizes >= 2
    return sizes[eligible], counts[eligible]


def load_schedule_games() -> pd.DataFrame:
    """(season, week, team) team-games from the newest schedules snapshot."""

    candidates = sorted((REPO / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise DataContractError("no schedules.parquet found under data/raw")
    schedules = pd.read_parquet(candidates[-1])
    missing = sorted(
        {"season", "week", "home_team", "away_team", "game_type"}.difference(schedules.columns)
    )
    if missing:
        raise DataContractError(f"schedules are missing columns: {', '.join(missing)}")
    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    home = reg.loc[:, ["season", "week", "home_team"]].rename(columns={"home_team": "team"})
    away = reg.loc[:, ["season", "week", "away_team"]].rename(columns={"away_team": "team"})
    games = pd.concat([home, away], ignore_index=True)
    games["season"] = pd.to_numeric(games["season"], errors="coerce").astype(int)
    games["week"] = pd.to_numeric(games["week"], errors="coerce").astype(int)
    return games.drop_duplicates().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--player-snapshot",
        type=Path,
        default=REPO / "data" / "players" / "raw" / "20260817T184901Z",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    started = time.time()
    snaps = canonicalize_snaps(pd.read_parquet(args.player_snapshot / "snap_counts.parquet"))
    rosters_raw = pd.read_parquet(args.player_snapshot / "weekly_rosters.parquet")
    snaps = attach_snap_player_ids(snaps, canonicalize_rosters(rosters_raw))
    rosters = load_unit_rosters(args.player_snapshot)
    contributors = build_contributor_games(snaps, rosters)
    schedule = load_schedule_games()
    contributors = contributors.merge(
        schedule.assign(on_schedule=True), on=["season", "week", "team"], how="inner"
    )
    team_snaps = (
        contributors.groupby(["season", "team", "week"], sort=False)["side_snaps"]
        .sum()
        .rename("team_side_snaps")
    )
    contributors = contributors.merge(team_snaps, on=["season", "team", "week"], how="left")
    contributors = contributors.loc[contributors["team_side_snaps"].ge(100)].copy()
    contributors = contributors.loc[contributors["season"].astype(int).isin(SEASONS)].copy()
    if contributors.empty:
        raise DataContractError("no contributor games in the requested seasons")
    units: dict[str, Any] = {}
    for unit in ("OFF_OL", "OFF_SKILL", "DEF_FRONT", "DEF_SECONDARY"):
        sub = contributors.loc[contributors["unit"].eq(unit)]
        if sub.empty:
            units[unit] = {"empty": True}
            continue
        gate = bootstrap_excess_ratio(contributors, unit)
        null = permutation_diagnostic(contributors, unit)
        base = unit_excess(contributors, unit)
        units[unit] = {**base, "bootstrap": gate, "permutation_null": null}
    configuration = {
        "command": "absence-dependence-screen",
        "player_snapshot": args.player_snapshot.name,
        "seasons": list(SEASONS),
        "trailing_games": TRAILING_GAMES,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "permutations": PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "predeclaration": "docs/absence_dependence.md (frozen before scoring)",
    }
    payload = {
        "contributor_games": len(contributors),
        "units": units,
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
        command="absence-dependence-screen",
        metrics=payload,
        notes=(
            "Descriptive PER-10/SIM-03 joint-absence measurement; no ATS "
            "outcome, no registry verdict, no window (AGENTS.md)."
        ),
    )
    for unit, result in units.items():
        print(
            unit,
            {k: v for k, v in result.items() if k in ("p2plus", "implied_p2plus", "excess_ratio")},
        )
    print(f"wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
