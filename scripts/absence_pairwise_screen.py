"""Pairwise co-absence excess by unit (PER-10 slice, frozen screen).

Predeclared in ``docs/absence_pairwise_dependence.md`` before any pairwise
number was computed. Motivation disclosed: the full-sit estimand degenerated
(0 whole-unit sits), so coupling is re-asked at the pair level the mixture
kernel actually scores. Descriptive: no ATS screen, no window, no registry
verdict.

Writes ``artifacts/absence_pairwise/<stamp>/results.json`` via
``write_experiment_artifact``.
"""

from __future__ import annotations

import argparse
import itertools
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
from scripts.absence_dependence_screen import (  # noqa: E402
    build_contributor_games,
    load_schedule_games,
    load_unit_rosters,
)

OUT_ROOT = REPO / "artifacts" / "absence_pairwise"
MIN_OVERLAP_GAMES = 10
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260906
PERMUTATIONS = 200
PERMUTATION_SEED = 20260907
MIN_TEAM_SNAPS = 100


def load_gated_contributors(player_snapshot: Path) -> pd.DataFrame:
    """Contributor frame with the frozen schedule + data-presence gates."""

    snaps = canonicalize_snaps(pd.read_parquet(player_snapshot / "snap_counts.parquet"))
    rosters_raw = pd.read_parquet(player_snapshot / "weekly_rosters.parquet")
    snaps = attach_snap_player_ids(snaps, canonicalize_rosters(rosters_raw))
    rosters = load_unit_rosters(player_snapshot)
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
    return contributors.loc[contributors["team_side_snaps"].ge(MIN_TEAM_SNAPS)].copy()


def enumerate_pairs(contributors: pd.DataFrame, unit: str) -> pd.DataFrame:
    """One row per eligible unordered pair: together games, joint absences,
    and each member's own absence rate on overlapping games."""

    frame = contributors.loc[contributors["unit"].eq(unit)].copy()
    if frame.empty:
        raise DataContractError(f"no contributor games for unit {unit}")
    frame["block"] = frame["season"].astype(str) + "_" + frame["team"].astype(str)
    frame["game"] = (
        frame["season"].astype(str)
        + "_"
        + frame["team"].astype(str)
        + "_"
        + frame["week"].astype(str)
    )
    records: list[dict[str, Any]] = []
    for (_, _game), group in frame.groupby(["block", "game"], sort=False):
        players = group["gsis_id"].astype(str).tolist()
        absent = {gsis: bool(flag) for gsis, flag in zip(players, group["absent"], strict=True)}
        for first, second in itertools.combinations(sorted(set(players)), 2):
            records.append(
                {
                    "block": group["block"].iloc[0],
                    "pair": (first, second),
                    "together": 1,
                    "both_absent": int(absent[first] and absent[second]),
                    "first_absent": int(absent[first]),
                    "second_absent": int(absent[second]),
                }
            )
    pairs = pd.DataFrame(records)
    if pairs.empty:
        raise DataContractError(f"unit {unit} produced no pairs")
    overlap = pairs.groupby("pair", sort=False)["together"].sum()
    eligible = overlap.loc[overlap.ge(MIN_OVERLAP_GAMES)].index
    pairs = pairs.loc[pairs["pair"].isin(set(eligible))].copy()
    if pairs.empty:
        raise DataContractError(
            f"unit {unit} has no pairs with {MIN_OVERLAP_GAMES}+ overlapping games"
        )
    return pairs.reset_index(drop=True)


def pooled_excess(pairs: pd.DataFrame) -> dict[str, float]:
    """Observed vs expected joint-absence rates over eligible pairs."""

    grouped = pairs.groupby("pair", sort=False)
    together = grouped["together"].sum()
    both = grouped["both_absent"].sum()
    first_rate = grouped["first_absent"].sum() / together
    second_rate = grouped["second_absent"].sum() / together
    expected = first_rate * second_rate
    observed = float(both.sum() / together.sum())
    expected_mean = float(expected.mean())
    return {
        "pairs": len(together),
        "together_games": int(together.sum()),
        "observed_joint_rate": observed,
        "expected_joint_rate": expected_mean,
        "excess_ratio": (observed / expected_mean) if expected_mean > 0 else float("nan"),
    }


def _pair_block_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Collapse pair-instances to (pair, block) rows for fast resampling."""

    grouped = pairs.groupby(["pair", "block"], sort=False)
    table = pd.DataFrame(
        {
            "together": grouped["together"].sum(),
            "both_absent": grouped["both_absent"].sum(),
            "first_absent": grouped["first_absent"].sum(),
            "second_absent": grouped["second_absent"].sum(),
        }
    ).reset_index()
    return table


def _excess_from_pair_blocks(table: pd.DataFrame) -> float:
    """Pooled excess from a (pair, block) table (nan when undefined)."""

    grouped = table.groupby("pair", sort=False)
    together = grouped["together"].sum()
    eligible = together[together.ge(1)].index
    if len(eligible) == 0:
        return float("nan")
    both = grouped["both_absent"].sum().loc[eligible]
    first_rate = (grouped["first_absent"].sum() / together).loc[eligible]
    second_rate = (grouped["second_absent"].sum() / together).loc[eligible]
    observed = float(both.sum() / together.loc[eligible].sum())
    expected_mean = float((first_rate * second_rate).mean())
    return (observed / expected_mean) if expected_mean > 0 else float("nan")


def bootstrap_excess(
    pairs: pd.DataFrame, *, seed: int = BOOTSTRAP_SEED, samples: int = BOOTSTRAP_SAMPLES
) -> dict[str, Any]:
    """Team-season block bootstrap CI over the collapsed pair-block table."""

    table = _pair_block_table(pairs)
    blocks = table["block"].unique()
    block_index: dict[str, np.ndarray] = {
        block: np.flatnonzero(table["block"].to_numpy() == block) for block in blocks
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for draw in range(samples):
        chosen = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([block_index[block] for block in chosen])
        draws[draw] = _excess_from_pair_blocks(table.iloc[idx])
    draws = draws[np.isfinite(draws)]
    point = pooled_excess(pairs)["excess_ratio"]
    return {
        "excess_ratio": float(point),
        "excess_ratio_ci95": (
            [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
            if len(draws)
            else [float("nan"), float("nan")]
        ),
        "samples": samples,
        "seed": seed,
        "blocks": len(blocks),
    }


def _block_player_game_matrices(contributors: pd.DataFrame, unit: str) -> dict[str, dict[str, Any]]:
    """Per team-season block: player list, game list, and established and
    absence indicator matrices (players x games)."""

    frame = contributors.loc[contributors["unit"].eq(unit)].copy()
    frame["block"] = frame["season"].astype(str) + "_" + frame["team"].astype(str)
    matrices: dict[str, dict[str, Any]] = {}
    for block, group in frame.groupby("block", sort=False):
        players = sorted(group["gsis_id"].astype(str).unique())
        games = sorted(
            group[["season", "week"]].drop_duplicates().itertuples(index=False, name=None)
        )
        player_index = {player: position for position, player in enumerate(players)}
        game_index = {game: position for position, game in enumerate(games)}
        established = np.zeros((len(players), len(games)), dtype=bool)
        absent = np.zeros((len(players), len(games)), dtype=bool)
        for row in group.itertuples(index=False):
            game = (int(row.season), int(row.week))
            if game not in game_index:
                continue
            position = player_index[str(row.gsis_id)]
            established[position, game_index[game]] = True
            absent[position, game_index[game]] = bool(row.absent)
        matrices[str(block)] = {
            "players": players,
            "established": established,
            "absent": absent,
        }
    if not matrices:
        raise DataContractError(f"no contributor games for unit {unit}")
    return matrices


def permutation_diagnostic(
    pairs: pd.DataFrame,
    contributors: pd.DataFrame,
    unit: str,
    *,
    seed: int = PERMUTATION_SEED,
    permutations: int = PERMUTATIONS,
) -> dict[str, Any]:
    """Shuffle absence labels within team-seasons over playerxgame matrices.

    Each player's own total is held fixed; only who-sat-with-whom varies.
    Eligibility (≥10 overlapping games on the full frame) is frozen, so a
    permuted draw cannot shop for pairs. Pooled across blocks exactly like
    the point estimate: joint absences over together-games, expected from
    the full-frame member rates.
    """

    grouped = pairs.groupby("pair", sort=False)
    together_full = grouped["together"].sum()
    eligible_pairs = set(together_full.loc[together_full.ge(MIN_OVERLAP_GAMES)].index)
    if not eligible_pairs:
        raise DataContractError(f"unit {unit} has no pairs clearing the overlap floor")
    # Member rates from the full frame (fixed across draws, as predeclared).
    member_rates: dict[str, float] = {}
    first_absent: dict[str, int] = {}
    together_count: dict[str, int] = {}
    for _, row in pairs.iterrows():
        first, second = row["pair"]
        for member, flag in ((first, row["first_absent"]), (second, row["second_absent"])):
            together_count[member] = together_count.get(member, 0) + int(row["together"])
            first_absent[member] = first_absent.get(member, 0) + int(flag)
    for member in together_count:
        member_rates[member] = first_absent[member] / together_count[member]
    matrices = _block_player_game_matrices(contributors, unit)
    rng = np.random.default_rng(seed)
    nulls = np.empty(permutations)
    for draw in range(permutations):
        joint_total = 0
        together_total = 0
        expected_weighted = 0.0
        for block in matrices.values():
            players = block["players"]
            shuffled = block["absent"].copy()
            for position in range(shuffled.shape[0]):
                shuffled[position] = rng.permutation(shuffled[position])
            established = block["established"]
            for first_pos in range(len(players)):
                for second_pos in range(first_pos + 1, len(players)):
                    first, second = players[first_pos], players[second_pos]
                    key = (
                        (first, second)
                        if (first, second) in eligible_pairs
                        else ((second, first) if (second, first) in eligible_pairs else None)
                    )
                    if key is None:
                        continue
                    together_mask = established[first_pos] & established[second_pos]
                    together_n = int(together_mask.sum())
                    if together_n == 0:
                        continue
                    joint_n = int(
                        (shuffled[first_pos] & shuffled[second_pos] & together_mask).sum()
                    )
                    joint_total += joint_n
                    together_total += together_n
                    expected_weighted += (
                        member_rates.get(first, 0.0) * member_rates.get(second, 0.0) * together_n
                    )
        if together_total == 0 or expected_weighted <= 0:
            nulls[draw] = np.nan
        else:
            nulls[draw] = (joint_total / together_total) / (expected_weighted / together_total)
    nulls = nulls[np.isfinite(nulls)]
    observed = pooled_excess(pairs)["excess_ratio"]
    return {
        "null_center": float(np.mean(nulls)) if len(nulls) else float("nan"),
        "observed_percentile": float(np.mean(nulls <= observed)) if len(nulls) else float("nan"),
        "permutations": permutations,
        "seed": seed,
    }


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
    contributors = load_gated_contributors(args.player_snapshot)
    contributors = contributors.loc[
        contributors["season"].astype(int).isin(range(2016, 2026))
    ].copy()
    if contributors.empty:
        raise DataContractError("no contributor games in the requested seasons")
    units: dict[str, Any] = {}
    for unit in ("OFF_OL", "OFF_SKILL", "DEF_FRONT", "DEF_SECONDARY"):
        try:
            pairs = enumerate_pairs(contributors, unit)
        except DataContractError as error:
            units[unit] = {"error": str(error)}
            continue
        gate = bootstrap_excess(pairs)
        null = permutation_diagnostic(pairs, contributors, unit)
        base = pooled_excess(pairs)
        units[unit] = {**base, "bootstrap": gate, "permutation_null": null}
    configuration = {
        "command": "absence-pairwise-screen",
        "player_snapshot": args.player_snapshot.name,
        "seasons": list(range(2016, 2026)),
        "min_overlap_games": MIN_OVERLAP_GAMES,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "permutations": PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "predeclaration": "docs/absence_pairwise_dependence.md (frozen before scoring)",
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
        command="absence-pairwise-screen",
        metrics=payload,
        notes=(
            "Descriptive PER-10 pairwise co-absence measurement; no ATS "
            "outcome, no registry verdict, no window (AGENTS.md)."
        ),
    )
    for unit, result in units.items():
        print(unit, {k: v for k, v in result.items() if k in ("excess_ratio", "pairs", "error")})
    print(f"wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
