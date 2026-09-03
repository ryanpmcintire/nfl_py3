"""XLG-06 Stage 2: recruiting rating vs NFL rookie production (frozen screen).

Predeclared in ``docs/xlg06_stage2_nfl_screen.md`` before any outcome number
for this comparison was computed. Population: crosswalk-linked recruits at
recruiting positions WR/RB/TE with a completed rookie season (<= 2024).
Predictor: recruiting ``rating``. Outcome: rookie-season REG
``rushing_epa + receiving_epa`` total. Statistic: Pearson r (Spearman
alongside) with a recruiting-cohort-blocked bootstrap.

Writes ``artifacts/xlg06_stage2_nfl/<stamp>/results.json`` via
``write_experiment_artifact`` plus a per-player audit table. Measure-only:
no card, model, or registry decision follows from any outcome.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.io import atomic_parquet, run_id  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "xlg06_stage2_nfl"
DEFAULT_CROSSWALK = (
    REPO / "artifacts" / "xlg06_crosswalk" / "20260903T104848Z" / "recruit_to_nfl_crosswalk.parquet"
)
DEFAULT_PLAYER_STATS = (
    REPO / "data" / "players" / "values" / "raw" / "20260817T184911Z" / "player_stats.parquet"
)

SKILL_POSITIONS = ("WR", "RB", "TE")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260903
NULL_SHUFFLES = 200
NULL_SEED = 20260904
MIN_ROOKIE_REG_WEEKS_FOR_RELIABILITY = 4


def build_stage2_population(
    crosswalk: pd.DataFrame, player_stats: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the frozen eligibility rules; fail closed on chronology.

    Returns the per-player frame (one row per linked recruit) and exclusion
    counts. Raises :class:`DataContractError` if any included row's rookie
    season does not strictly postdate its recruiting year.
    """

    excluded: dict[str, int] = {}
    linked = crosswalk.loc[
        crosswalk["gsis_id"].notna() & crosswalk["position"].isin(SKILL_POSITIONS)
    ].copy()
    linked["gsis_id"] = linked["gsis_id"].astype(str)
    linked["rating_num"] = pd.to_numeric(linked["rating"], errors="coerce")
    linked["recruit_year_num"] = pd.to_numeric(linked["year"], errors="coerce")
    usable = linked.loc[linked["rating_num"].notna()].copy()
    excluded["null_rating"] = int(len(linked) - len(usable))
    # One row per NFL identity; keep the best-rated recruiting row on ties.
    usable = (
        usable.sort_values("rating_num", ascending=False)
        .drop_duplicates(subset="gsis_id", keep="first")
        .reset_index(drop=True)
    )
    stats = player_stats.loc[
        player_stats["player_id"].astype(str).isin(set(usable["gsis_id"]))
    ].copy()
    stats["gsis_id"] = stats["player_id"].astype(str)
    stats["season"] = pd.to_numeric(stats["season"], errors="coerce")
    rookie_season = stats.groupby("gsis_id")["season"].min().rename("rookie_season")
    frame = usable.merge(rookie_season, on="gsis_id", how="left")
    no_stats = frame.loc[frame["rookie_season"].isna()].copy()
    excluded["no_production_rows"] = len(no_stats)
    frame = frame.loc[frame["rookie_season"].notna()].copy()
    incomplete = frame.loc[frame["rookie_season"].gt(2024)].copy()
    excluded["incomplete_rookie_season_2025"] = len(incomplete)
    frame = frame.loc[frame["rookie_season"].le(2024)].copy()
    rookie_reg = stats.loc[
        stats["season_type"].eq("REG") & stats["gsis_id"].isin(set(frame["gsis_id"]))
    ].copy()
    rookie_reg = rookie_reg.merge(
        frame.loc[:, ["gsis_id", "rookie_season"]], on="gsis_id", how="inner"
    )
    rookie_reg = rookie_reg.loc[rookie_reg["season"].eq(rookie_reg["rookie_season"])].copy()
    with_reg = set(rookie_reg["gsis_id"].unique())
    excluded["postseason_only_debut"] = int(len(frame) - len(set(frame["gsis_id"]) & with_reg))
    frame = frame.loc[frame["gsis_id"].isin(with_reg)].copy()
    violated = frame.loc[
        frame["recruit_year_num"].notna() & frame["rookie_season"].le(frame["recruit_year_num"])
    ]
    if len(violated):
        raise DataContractError(
            "recruiting rating does not strictly predate the rookie season for "
            f"{len(violated)} rows (e.g. {violated['gsis_id'].iloc[:5].tolist()})"
        )
    frame = frame.reset_index(drop=True)
    return frame, excluded


def rookie_epa_totals(frame: pd.DataFrame, rookie_reg: pd.DataFrame) -> pd.DataFrame:
    """Sum REG rushing+receiving EPA over each player's rookie season."""

    epa = rookie_reg.copy()
    epa["rushing_epa"] = pd.to_numeric(epa["rushing_epa"], errors="coerce").fillna(0.0)
    epa["receiving_epa"] = pd.to_numeric(epa["receiving_epa"], errors="coerce").fillna(0.0)
    epa["weekly_epa"] = epa["rushing_epa"] + epa["receiving_epa"]
    totals = (
        epa.groupby("gsis_id", sort=False)
        .agg(rookie_epa=("weekly_epa", "sum"), rookie_reg_weeks=("week", "nunique"))
        .reset_index()
    )
    return frame.merge(totals, on="gsis_id", how="inner")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)


def blocked_bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    """Cohort-blocked percentile bootstrap for Pearson and Spearman."""

    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(blocks)
    block_rows = {block: np.flatnonzero(blocks == block) for block in unique_blocks}
    pearson_draws = np.empty(samples)
    spearman_draws = np.empty(samples)
    for draw in range(samples):
        # Cluster bootstrap: resample whole cohorts with replacement, taking
        # every row of each drawn cohort (drawn cohorts repeat).
        chosen = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        sample = np.concatenate([block_rows[block] for block in chosen])
        pearson_draws[draw] = _pearson(x[sample], y[sample])
        spearman_draws[draw] = _spearman(x[sample], y[sample])
    return {
        "pearson_r": _pearson(x, y),
        "pearson_r_ci95": [
            float(np.nanquantile(pearson_draws, 0.025)),
            float(np.nanquantile(pearson_draws, 0.975)),
        ],
        "pearson_probability_positive": float(np.mean(pearson_draws > 0.0)),
        "spearman_rho": _spearman(x, y),
        "spearman_rho_ci95": [
            float(np.nanquantile(spearman_draws, 0.025)),
            float(np.nanquantile(spearman_draws, 0.975)),
        ],
        "spearman_probability_positive": float(np.mean(spearman_draws > 0.0)),
        "samples": samples,
        "seed": seed,
        "blocks": len(unique_blocks),
    }


def split_half_reliability(
    rookie_reg: pd.DataFrame, *, min_weeks: int = MIN_ROOKIE_REG_WEEKS_FOR_RELIABILITY
) -> dict[str, Any]:
    """Odd/even-week split-half reliability of the rookie EPA total."""

    ordered = rookie_reg.sort_values(["gsis_id", "week"]).copy()
    ordered["half"] = ordered.groupby("gsis_id").cumcount().mod(2).map({0: "odd", 1: "even"})
    halves = (
        ordered.groupby(["gsis_id", "half"], sort=False)["weekly_epa"]
        .agg(["sum", "size"])
        .reset_index()
    )
    wide = halves.pivot(index="gsis_id", columns="half")
    wide.columns = [f"{stat}_{half}" for stat, half in wide.columns]
    eligible = wide.loc[wide["size_odd"].add(wide["size_even"], fill_value=0).ge(min_weeks)].copy()
    if len(eligible) < 3:
        return {"n": len(eligible), "insufficient": True}
    r = _pearson(
        eligible["sum_odd"].to_numpy(dtype=float), eligible["sum_even"].to_numpy(dtype=float)
    )
    reliability = 2.0 * r / (1.0 + r) if np.isfinite(r) and r > 0 else float(r)
    return {"n": len(eligible), "split_half_r": float(r), "spearman_brown": float(reliability)}


def shuffle_null(
    x: np.ndarray, y: np.ndarray, blocks: np.ndarray, *, seed: int, shuffles: int
) -> dict[str, Any]:
    """Within-cohort predictor shuffles: machinery check, not a control."""

    rng = np.random.default_rng(seed)
    draws = np.empty(shuffles)
    for draw in range(shuffles):
        permuted = x.copy()
        for block in np.unique(blocks):
            idx = np.flatnonzero(blocks == block)
            permuted[idx] = rng.permutation(permuted[idx])
        draws[draw] = _pearson(permuted, y)
    observed = _pearson(x, y)
    return {
        "null_center": float(np.mean(draws)),
        "observed_percentile": float(np.mean(draws <= observed)),
        "shuffles": shuffles,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--player-stats", type=Path, default=DEFAULT_PLAYER_STATS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    started = time.time()
    crosswalk = pd.read_parquet(args.crosswalk)
    player_stats = pd.read_parquet(args.player_stats)
    frame, excluded = build_stage2_population(crosswalk, player_stats)
    rookie_reg = player_stats.loc[
        player_stats["season_type"].eq("REG")
        & player_stats["player_id"].astype(str).isin(set(frame["gsis_id"]))
    ].copy()
    rookie_reg["gsis_id"] = rookie_reg["player_id"].astype(str)
    rookie_reg = rookie_reg.merge(frame.loc[:, ["gsis_id", "rookie_season"]], on="gsis_id")
    rookie_reg = rookie_reg.loc[rookie_reg["season"].eq(rookie_reg["rookie_season"])].copy()
    rookie_reg["rushing_epa"] = pd.to_numeric(rookie_reg["rushing_epa"], errors="coerce").fillna(
        0.0
    )
    rookie_reg["receiving_epa"] = pd.to_numeric(
        rookie_reg["receiving_epa"], errors="coerce"
    ).fillna(0.0)
    rookie_reg["weekly_epa"] = rookie_reg["rushing_epa"] + rookie_reg["receiving_epa"]
    scored = rookie_epa_totals(frame, rookie_reg)
    x = scored["rating_num"].to_numpy(dtype=float)
    y = scored["rookie_epa"].to_numpy(dtype=float)
    blocks = scored["recruit_year_num"].astype(str).to_numpy()
    correlation = blocked_bootstrap_correlation(
        x, y, blocks, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES
    )
    reliability = split_half_reliability(rookie_reg)
    null = shuffle_null(x, y, blocks, seed=NULL_SEED, shuffles=NULL_SHUFFLES)
    configuration = {
        "command": "xlg06-stage2-nfl-screen",
        "crosswalk": str(args.crosswalk),
        "player_stats": str(args.player_stats),
        "positions": list(SKILL_POSITIONS),
        "predictor": "recruiting rating",
        "outcome": "rookie-season REG rushing_epa + receiving_epa total",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "null_shuffles": NULL_SHUFFLES,
        "null_seed": NULL_SEED,
        "predeclaration": "docs/xlg06_stage2_nfl_screen.md (frozen before scoring)",
    }
    payload = {
        "n": len(scored),
        "excluded": excluded,
        "correlation": correlation,
        "reliability": reliability,
        "shuffle_null": null,
        "elapsed_seconds": time.time() - started,
        "provenance": artifact_provenance(configuration, args.player_stats, project_root=REPO),
    }
    output_dir = args.output or (OUT_ROOT / run_id())
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_parquet(
        scored.loc[
            :,
            [
                "gsis_id",
                "position",
                "year",
                "rating_num",
                "rookie_season",
                "rookie_epa",
                "rookie_reg_weeks",
            ],
        ],
        output_dir / "rookie_epa.parquet",
    )
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="xlg06-stage2-nfl-screen",
        metrics=payload,
        notes=(
            "Measure-only Stage-2 correlation gate (recruiting rating vs NFL "
            "rookie production); predeclared cells record unresolved_below_power "
            "via separate nfl-ats weak-signals record calls regardless of "
            "interval shape (AGENTS.md)."
        ),
    )
    print(
        f"n={len(scored)} pearson_r={correlation['pearson_r']:.4f} "
        f"P+={correlation['pearson_probability_positive']:.4f}"
    )
    print(f"wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
