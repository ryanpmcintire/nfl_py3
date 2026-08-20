"""Environmental-exposure battery: the two predeclared mined cells from
``docs/environmental_exposures.md`` section 6 -- high-AQI outdoor games (A)
and severe-drought grass-surface games (B) -- screened against ATS on
2009-2025 (close grade) with an OPENER-grade re-screen on the 2020-2025
paired archive where population allows.

**Both directions are NOT predeclared.** Section 6.A's mechanism paragraph
hedges ("...or vice versa if the home team is *also* unacclimated") and
section 6.B's never names a team side at all. Per this project's own
precedent for a mined cell with no predeclared direction
(``roof_battery_closed_benign_forecast_vs_open``), both cells here are scored
**exploratory, sign fixed at +1, raw gap reported as-is**.

The full predeclaration (population definitions reproduced to match the
doc's headline 44/222 exactly, the whole-league-complement method choice,
era-split boundary, opener re-screen plan) is frozen in
``<scratchpad>/environmental_battery/predeclaration.md`` and was written
before this script scored anything.

**Method**, reused verbatim from ``scripts/nfl_weather_battery_screen.py``
(game-level population, ``home_cover`` value column, week-blocked joint
block bootstrap, full-slate-scaled effect) and
``scripts/roof_decision_screen.py`` (whole-REG-population complement,
opener-grade re-screen via ``nfl_ats.experiment_runner._opener_graded_features``,
imported read-only).

Data: ``data/processed/game_features.parquet`` inner-joined on ``game_id``
with ``data/processed/environmental_exposures/game_join.parquet`` (AQI +
drought + roof/surface flags, built by ``scripts/build_environmental_exposure_join.py``).
REG season only, 2009-2025, for the close-grade population; the opener
re-screen further restricts to the paired 2020-2025 Tuesday-opener/close
archive.

**Measure-only.** This script never writes to the weak-signal registry; a
separate, explicit ``nfl-ats weak-signals record`` invocation per cell
records each finding as ``environmental_battery_<family>[_opener]``. It DOES
write an automatic, low-stakes experiment-provenance stamp to
``registry/experiments/`` via ``write_experiment_artifact`` (RWB-09).

Writes JSON to
``artifacts/environmental_exposure_battery/<UTC timestamp>/results.json``.
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
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.experiment_runner import _opener_graded_features  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

DEFAULT_FEATURES = REPO / "data/processed/game_features.parquet"
DEFAULT_JOIN = REPO / "data/processed/environmental_exposures/game_join.parquet"

SEASON_START = 2009
SEASON_END = 2025
AQI_THRESHOLD = 101.0
DROUGHT_THRESHOLD = 50.0
ERA_SPLITS: tuple[tuple[str, int, int], ...] = (
    ("2009_2016", 2009, 2016),
    ("2017_2025", 2017, 2025),
)
MIN_ERA_FLAG = 5
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260820

ENV_COLUMNS = [
    "game_id",
    "roof",
    "surface",
    "is_outdoor_exposed",
    "is_dome_or_closed",
    "aqi",
    "aqi_category",
    "aqi_is_stale_carryforward",
    "drought_primary_category",
    "drought_d2",
    "drought_is_stale_carryforward",
]


def load_population(features_path: Path, join_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per REG-season game, 2009-2025, environmental columns joined,
    ``home_cover`` pushes dropped. Returns (df, counters) for the honesty
    accounting printed/recorded alongside the cells.
    """

    features = pd.read_parquet(features_path)
    join = pd.read_parquet(join_path).loc[:, ENV_COLUMNS]

    merged = features.merge(join, on="game_id", how="inner", validate="one_to_one")
    merged = merged.loc[merged["game_type"] == "REG"].copy()
    merged["season"] = merged["season"].astype(int)
    merged["week"] = merged["week"].astype(int)
    n_reg_joined = len(merged)

    merged = merged.loc[merged["season"].between(SEASON_START, SEASON_END)].copy()
    n_before_push_drop = len(merged)
    merged = merged.loc[merged["home_cover"].notna()].reset_index(drop=True)
    pushes_dropped = n_before_push_drop - len(merged)

    merged["is_outdoor_exposed"] = merged["is_outdoor_exposed"].fillna(False).astype(bool)
    merged["surface_is_grass"] = (
        merged["surface"].astype(str).str.contains("grass", case=False, na=False)
    )
    merged["week_block"] = merged["season"] * 100 + merged["week"]

    counters = {
        "n_reg_joined": n_reg_joined,
        "n_before_push_drop": n_before_push_drop,
        "pushes_dropped": pushes_dropped,
        "n_scored_population": len(merged),
    }
    return merged, counters


def cell_a_flags(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(population, subset) boolean masks for the high-AQI outdoor cell."""

    population = ~df["aqi_is_stale_carryforward"]
    subset = population & df["is_outdoor_exposed"] & (df["aqi"] >= AQI_THRESHOLD)
    return population, subset


def cell_b_flags(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(population, subset) boolean masks for the severe-drought grass cell."""

    population = ~df["drought_is_stale_carryforward"]
    subset = (
        population
        & df["is_outdoor_exposed"]
        & df["surface_is_grass"]
        & (df["drought_d2"] >= DROUGHT_THRESHOLD)
    )
    return population, subset


def block_bootstrap_two_group(
    df: pd.DataFrame,
    *,
    flag_col: str,
    value_col: str,
    block_col: str,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Vectorized joint block bootstrap of ``100*(subset_mean-complement_mean)``.

    Reused verbatim (same algorithm) from
    ``scripts/nfl_weather_battery_screen.py::block_bootstrap_two_group``.
    """

    blocks, block_index = np.unique(df[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    values = df[value_col].to_numpy(dtype=np.float64)
    flag = df[flag_col].to_numpy(dtype=bool)

    sums: dict[bool, np.ndarray] = {}
    counts: dict[bool, np.ndarray] = {}
    for group in (True, False):
        mask = flag == group
        sums[group] = np.bincount(
            block_index[mask], weights=values[mask], minlength=block_count
        ).astype(np.float64)
        counts[group] = np.bincount(block_index[mask], minlength=block_count).astype(np.float64)

    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    subset_count = drawn @ counts[True]
    complement_count = drawn @ counts[False]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_subset = (drawn @ sums[True]) / subset_count
        mean_complement = (drawn @ sums[False]) / complement_count
    gap = (mean_subset - mean_complement) * 100.0
    valid = (subset_count > 0) & (complement_count > 0)
    return gap[valid]


def summarize(
    df: pd.DataFrame,
    *,
    population: pd.Series,
    subset: pd.Series,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """subset vs. (population - subset) complement, sign fixed at +1 (both
    cells here are exploratory / no predeclared direction -- see
    predeclaration.md).
    """

    scored = df.loc[population].reset_index(drop=True)
    flag = subset.loc[population].reset_index(drop=True)
    n_total = len(scored)
    n_flag = int(flag.sum())
    n_complement = n_total - n_flag
    if n_flag == 0 or n_complement == 0:
        return {
            "n_total": n_total,
            "n_flag": n_flag,
            "n_complement": n_complement,
            "insufficient_data": True,
        }

    work = scored.copy()
    work["_flag"] = flag.to_numpy()
    subset_cover = float(work.loc[work["_flag"], "home_cover"].mean())
    complement_cover = float(work.loc[~work["_flag"], "home_cover"].mean())
    raw_gap_pts = (subset_cover - complement_cover) * 100.0
    fraction_of_slate = n_flag / n_total
    full_slate_effect_pts = raw_gap_pts * fraction_of_slate

    draws = block_bootstrap_two_group(
        work,
        flag_col="_flag",
        value_col="home_cover",
        block_col="week_block",
        samples=samples,
        seed=seed,
    )
    dropped = samples - len(draws)
    scaled_draws = draws * fraction_of_slate
    lower, upper = (
        np.quantile(scaled_draws, [0.025, 0.975]) if len(scaled_draws) else (np.nan, np.nan)
    )

    return {
        "n_total": n_total,
        "n_flag": n_flag,
        "n_complement": n_complement,
        "n_week_blocks": int(work["week_block"].nunique()),
        "subset_cover": subset_cover,
        "complement_cover": complement_cover,
        "raw_gap_pts": raw_gap_pts,
        "fraction_of_slate": fraction_of_slate,
        "full_slate_effect_pts": full_slate_effect_pts,
        "week_blocked_ci95_scaled": [float(lower), float(upper)],
        "probability_positive": float(np.mean(draws > 0)) if len(draws) else np.nan,
        "bootstrap_samples": samples,
        "dropped_draws": int(dropped),
        "insufficient_data": False,
    }


def score_family(
    df: pd.DataFrame,
    *,
    name: str,
    flags_fn: Any,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    population, subset = flags_fn(df)
    full = summarize(df, population=population, subset=subset, samples=samples, seed=seed)

    era_results: dict[str, dict[str, Any]] = {}
    for era_label, start, end in ERA_SPLITS:
        era_mask = df["season"].between(start, end)
        era_pop = population & era_mask
        era_sub = subset & era_mask
        n_flag_era = int((era_sub & era_pop).sum())
        n_complement_era = int((era_pop & ~era_sub).sum())
        if n_flag_era < MIN_ERA_FLAG or n_complement_era < MIN_ERA_FLAG:
            era_results[era_label] = {
                "insufficient_data": True,
                "n_flag": n_flag_era,
                "n_complement": n_complement_era,
                "reason": f"below predeclared MIN_ERA_FLAG={MIN_ERA_FLAG}",
            }
            continue
        era_results[era_label] = summarize(
            df, population=era_pop, subset=era_sub, samples=samples, seed=seed
        )

    return {"name": name, "full_period": full, "era_split": era_results}


def _print_cell(cell: dict[str, Any]) -> None:
    print(f"\n=== {cell['name']} ===")
    full = cell["full_period"]
    if full.get("insufficient_data"):
        print(f"  insufficient data: n_flag={full['n_flag']} n_complement={full['n_complement']}")
        return
    print(
        f"  n_flag={full['n_flag']} n_total={full['n_total']} "
        f"subset_cover={full['subset_cover']:.4f} complement_cover={full['complement_cover']:.4f}"
    )
    print(
        f"  raw_gap={full['raw_gap_pts']:+.3f}pts frac_of_slate={full['fraction_of_slate']:.4f} "
        f"full_slate_effect={full['full_slate_effect_pts']:+.4f}pts"
    )
    print(
        f"  week-blocked 95% scaled [{full['week_blocked_ci95_scaled'][0]:+.4f}, "
        f"{full['week_blocked_ci95_scaled'][1]:+.4f}] P+={full['probability_positive']:.4f} "
        f"n_week_blocks={full['n_week_blocks']}"
    )
    for era_label, era in cell.get("era_split", {}).items():
        if era.get("insufficient_data"):
            print(
                f"  [{era_label}] insufficient data n_flag={era.get('n_flag')} "
                f"n_complement={era.get('n_complement')}"
            )
            continue
        print(
            f"  [{era_label}] n_flag={era['n_flag']} cover={era['subset_cover']:.4f} "
            f"vs {era['complement_cover']:.4f} "
            f"full_slate_effect={era['full_slate_effect_pts']:+.4f}pts "
            f"P+={era['probability_positive']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--join", type=Path, default=DEFAULT_JOIN)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (
        REPO / "artifacts" / "environmental_exposure_battery" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.features} + {args.join} ===")
    df, counters = load_population(args.features, args.join)
    print(
        f"REG {SEASON_START}-{SEASON_END} games joined: {counters['n_reg_joined']}, "
        f"pushes dropped: {counters['pushes_dropped']}, "
        f"scored population: {counters['n_scored_population']}"
    )

    pop_a, sub_a = cell_a_flags(df)
    pop_b, sub_b = cell_b_flags(df)
    print(f"cell A (AQI): population(fresh)={int(pop_a.sum())} subset(flag)={int(sub_a.sum())}")
    print(f"cell B (drought): population(fresh)={int(pop_b.sum())} subset(flag)={int(sub_b.sum())}")

    results: list[dict[str, Any]] = []

    cell_a = score_family(
        df,
        name="environmental_battery_aqi_high_outdoor",
        flags_fn=cell_a_flags,
        samples=args.samples,
        seed=args.seed,
    )
    results.append(cell_a)
    _print_cell(cell_a)

    cell_b = score_family(
        df,
        name="environmental_battery_drought_severe_grass",
        flags_fn=cell_b_flags,
        samples=args.samples,
        seed=args.seed,
    )
    results.append(cell_b)
    _print_cell(cell_b)

    print("\n=== opener-grade re-screen (2020-2025 Tuesday-open/close-paired archive) ===")
    opener_df, opener_note = _opener_graded_features(df, repo_root=REPO, market_root=None)
    print(f"  {opener_note}")
    n_opener_before_push_drop = len(opener_df)
    # Opener regrading recomputes home_cover off the OPENER line, which can push
    # a game that did not push at the close (and vice versa) -- re-drop pushes
    # here rather than trusting the close-grade push-drop from load_population,
    # matching roof_decision_screen.py's build_long_table doing the same after
    # its own opener regrade.
    opener_df = opener_df.loc[opener_df["home_cover"].notna()].reset_index(drop=True)
    opener_pushes_dropped = n_opener_before_push_drop - len(opener_df)
    opener_df["week_block"] = opener_df["season"] * 100 + opener_df["week"]
    opener_df["is_outdoor_exposed"] = opener_df["is_outdoor_exposed"].fillna(False).astype(bool)
    opener_df["surface_is_grass"] = (
        opener_df["surface"].astype(str).str.contains("grass", case=False, na=False)
    )
    print(
        f"  opener-grade pushes dropped (post-regrade): {opener_pushes_dropped}, "
        f"scored population: {len(opener_df)}"
    )

    pop_a_op, sub_a_op = cell_a_flags(opener_df)
    pop_b_op, sub_b_op = cell_b_flags(opener_df)
    print(
        f"  opener cell A: population(fresh)={int(pop_a_op.sum())} "
        f"subset(flag)={int(sub_a_op.sum())}"
    )
    print(
        f"  opener cell B: population(fresh)={int(pop_b_op.sum())} "
        f"subset(flag)={int(sub_b_op.sum())}"
    )

    cell_a_opener_full = summarize(
        opener_df, population=pop_a_op, subset=sub_a_op, samples=args.samples, seed=args.seed
    )
    cell_a_opener = {
        "name": "environmental_battery_aqi_high_outdoor_opener",
        "full_period": cell_a_opener_full,
        "era_split": {},
    }
    results.append(cell_a_opener)
    _print_cell(cell_a_opener)

    cell_b_opener_full = summarize(
        opener_df, population=pop_b_op, subset=sub_b_op, samples=args.samples, seed=args.seed
    )
    cell_b_opener = {
        "name": "environmental_battery_drought_severe_grass_opener",
        "full_period": cell_b_opener_full,
        "era_split": {},
    }
    results.append(cell_b_opener)
    _print_cell(cell_b_opener)

    configuration = {
        "command": "environmental-exposure-battery",
        "features": str(args.features),
        "join": str(args.join),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "aqi_threshold": AQI_THRESHOLD,
        "drought_threshold": DROUGHT_THRESHOLD,
        "era_splits": ERA_SPLITS,
        "min_era_flag": MIN_ERA_FLAG,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": SEASON_START,
        "season_end": SEASON_END,
        "counters": counters,
        "opener_population_note": opener_note,
        "opener_pushes_dropped": opener_pushes_dropped,
        "predeclaration": (
            "<scratchpad>/environmental_battery/predeclaration.md (frozen before scoring)"
        ),
        "results": results,
        "provenance": artifact_provenance(configuration, args.join, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="environmental-exposure-battery",
        metrics=payload,
        notes=(
            "Measure-only battery, 2 predeclared mined cells (AQI, drought) x 2 grades "
            "(close full-period + era split, opener re-screen); both directions NOT "
            "predeclared in docs/environmental_exposures.md sec 6, scored exploratory "
            "sign=+1 per that section's own hedged mechanism text. Every cell predeclared "
            "to record unresolved_below_power via a separate nfl-ats weak-signals record "
            "call regardless of interval shape (AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
