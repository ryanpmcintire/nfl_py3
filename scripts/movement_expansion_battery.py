"""Movement expansion battery: predeclared measurement (measure-only).

Predeclaration: `docs/movement_expansion_battery.md` (frozen before any
accuracy sign below was computed). Reuses the frozen opener archive and the
observed-movement threshold-overlay construction from
`scripts/observed_movement_channel.py`, extended along two axes the existing
family never tested: MAGNITUDE (a 2.0-point threshold, above the frozen
0.5/1.0 grid) and TIMING (the purchased archive's `thu_pre_tnf`/`sat_midday`
decision-label checkpoints, full 2020-2025 coverage, not just the
2023-2025-only `intraday_hourly` Sunday-realism arm).

This is a NEW rotation-registry family (`movement_expansion_v1`, grade
`opener`, `--acknowledge-mined`), declared and assigned to `[2020, 2021]`
before this script ran (see the predeclaration doc). Every cell below is
therefore scored on that drawn window's 466 games (456 non-push), not the
full archive -- a smaller, genuinely fresh-to-this-family look, honestly
sized rather than silently widened.

Five predeclared cells (`docs/movement_expansion_battery.md`):
  1. movement_expansion_window_close_threshold_1_0  (reproduction/consistency check)
  2. movement_expansion_thu_oracle_full_slate         (timing, ceiling)
  3. movement_expansion_thu_threshold_1_0             (timing, playable rule)
  4. movement_expansion_sat_threshold_1_0             (timing, playable rule)
  5. movement_expansion_close_threshold_2_0           (magnitude, untested tier)

Plus one instrument diagnostic, NOT recorded to the weak-signal registry:
a perfect-foresight positive control on the identical population.

Every cell's paired flip-value = candidate pick minus production pick
(`pick_home_at_open_probability_rule`), both graded at `margin_vs_open` (the
frozen Tuesday line), reported regardless of sign (AGENTS.md). Week-blocked
bootstrap primary, season-blocked secondary
(`nfl_ats.clv.week_blocked_bootstrap`, seed 20260831, 20,000 samples), plus a
200-draw within-week permutation null of the settlement margin (not centred
on zero by design -- the production pick carries its own home/away tilt).

Writes `artifacts/movement_expansion_battery/<run_id>/` (`per_game.parquet`,
`cells_summary.csv`, `metadata.json` via `write_experiment_artifact`, which
also stamps `registry/experiments/`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.clv import (  # noqa: E402
    build_pairing_table,
    pick_correct,
    week_blocked_bootstrap,
)
from nfl_ats.odds_backfill import DECISION_LABELS, HISTORICAL_CAPTURE_KIND  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.rotation import load_registry  # noqa: E402

DEFAULT_ARCHIVE = REPO_ROOT / "artifacts/opener_evaluation/20260819T174244Z/per_game.parquet"
DEFAULT_MARKET_ROOT = REPO_ROOT / "data/market/raw"
DEFAULT_GAME_FEATURES = REPO_ROOT / "data/processed/game_features.parquet"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/movement_expansion_battery"
DEFAULT_REGISTRY_ROOT = REPO_ROOT / "registry"
ROTATION_FAMILY = "movement_expansion_v1"

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260831
NULL_PERMUTATIONS = 200
PRODUCTION_PICK_COL = "pick_home_at_open_probability_rule"
PRODUCTION_CORRECT_COL = "correct_at_open_probability_rule"


# ---------------------------------------------------------------------------
# Population construction
# ---------------------------------------------------------------------------


def load_window_population(archive_path: Path, seasons: tuple[int, int]) -> pd.DataFrame:
    archive = pd.read_parquet(archive_path)
    window = archive.loc[archive["season"].between(seasons[0], seasons[1])].reset_index(drop=True)
    return window


def load_timing_checkpoints(
    market_root: Path, game_features_path: Path, seasons: tuple[int, int]
) -> pd.DataFrame:
    """Wide (game_id -> {label: home_spread}) table for the drawn window's seasons."""

    features = pd.read_parquet(game_features_path)
    schedule = features.loc[features["game_type"].eq("REG"), ["game_id", "season", "week"]].dropna()
    schedule = schedule.assign(
        season=schedule["season"].astype(int), week=schedule["week"].astype(int)
    )
    season_list = list(range(seasons[0], seasons[1] + 1))
    pairing = build_pairing_table(
        market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=DECISION_LABELS,
        seasons=season_list,
        schedule=schedule,
    )
    wide = pairing.pivot_table(
        index="game_id", columns="decision_label", values="home_spread", aggfunc="first"
    )
    return wide.reset_index()


# ---------------------------------------------------------------------------
# Candidate pick construction (frozen, docs/movement_expansion_battery.md)
# ---------------------------------------------------------------------------


def oracle_pick(cur: pd.Series, tue_open: pd.Series, production_home: pd.Series) -> pd.Series:
    move = cur - tue_open
    return pd.Series(
        np.where(move.gt(0.0), True, np.where(move.lt(0.0), False, production_home.astype(bool))),
        index=cur.index,
    )


def threshold_pick(
    cur: pd.Series, tue_open: pd.Series, production_home: pd.Series, threshold: float
) -> pd.Series:
    move = cur - tue_open
    movement_home = np.where(move.gt(0.0), True, np.where(move.lt(0.0), False, np.nan))
    eligible = move.abs().ge(threshold)
    return pd.Series(
        np.where(eligible, movement_home, production_home.astype(bool)),
        index=cur.index,
    ).astype(bool)


# ---------------------------------------------------------------------------
# Scoring: paired week/season-blocked bootstrap + within-week permutation null
# ---------------------------------------------------------------------------


def _paired_metric_fn(candidate_col: str, production_col: str):
    def _metric(rows: pd.DataFrame) -> dict[str, float]:
        cand = pd.to_numeric(rows[candidate_col], errors="coerce")
        prod = pd.to_numeric(rows[production_col], errors="coerce")
        both = rows.loc[cand.notna() & prod.notna()]
        if both.empty:
            return {
                "candidate_accuracy": float("nan"),
                "production_accuracy": float("nan"),
                "paired_delta": float("nan"),
            }
        cand_v = pd.to_numeric(both[candidate_col], errors="coerce")
        prod_v = pd.to_numeric(both[production_col], errors="coerce")
        return {
            "candidate_accuracy": float(cand_v.mean()),
            "production_accuracy": float(prod_v.mean()),
            "paired_delta": float((cand_v - prod_v).mean()),
        }

    return _metric


def week_positions(frame: pd.DataFrame) -> list[np.ndarray]:
    return [
        np.asarray(positions, dtype=np.intp)
        for positions in frame.groupby(["season", "week"], sort=False).indices.values()
    ]


def permuted_margins(
    frame: pd.DataFrame, column: str, rng: np.random.Generator, groups: list[np.ndarray]
) -> pd.Series:
    values = frame[column].to_numpy(dtype=float, copy=True)
    for positions in groups:
        values[positions] = rng.permutation(values[positions])
    return pd.Series(values, index=frame.index)


def pick_correct_flag(pick_home: pd.Series, margin: pd.Series) -> pd.Series:
    covered_home = margin.gt(0.0)
    correct = np.where(pick_home.astype(bool), covered_home, ~covered_home).astype(float)
    return pd.Series(np.where(margin.eq(0.0), np.nan, correct), index=margin.index)


def null_distribution(
    frame: pd.DataFrame, *, candidate_col: str, permutations: int, seed: int
) -> dict[str, Any]:
    working = frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    groups = week_positions(working)
    deltas: list[float] = []
    production_home = working[PRODUCTION_PICK_COL]
    candidate_home = working[candidate_col]
    for _ in range(permutations):
        margin = permuted_margins(working, "margin_vs_open", rng, groups)
        base_correct = pick_correct_flag(production_home, margin)
        cand_correct = pick_correct_flag(candidate_home, margin)
        valid = pd.DataFrame({"b": base_correct, "c": cand_correct}).dropna()
        deltas.append(float((valid["c"] - valid["b"]).mean()) if len(valid) else float("nan"))
    values = np.asarray(deltas, dtype=float)
    finite = values[np.isfinite(values)]
    observed_margin = working["margin_vs_open"]
    observed_base = pick_correct_flag(production_home, observed_margin)
    observed_cand = pick_correct_flag(candidate_home, observed_margin)
    observed_valid = pd.DataFrame({"b": observed_base, "c": observed_cand}).dropna()
    observed = float((observed_valid["c"] - observed_valid["b"]).mean())
    return {
        "permutations": len(finite),
        "null_mean_delta_points": float(finite.mean() * 100.0) if len(finite) else float("nan"),
        "null_q025_points": float(np.quantile(finite, 0.025) * 100.0)
        if len(finite)
        else float("nan"),
        "null_q975_points": float(np.quantile(finite, 0.975) * 100.0)
        if len(finite)
        else float("nan"),
        "observed_delta_points": float(observed * 100.0),
        "fraction_of_null_below_observed": float((finite < observed).mean())
        if len(finite)
        else float("nan"),
    }


def score_cell(
    frame: pd.DataFrame, *, name: str, candidate_col: str, samples: int, seed: int
) -> dict[str, Any]:
    working = frame.copy()
    working["_candidate_correct"] = pick_correct(
        working[candidate_col].astype(bool), working["margin_vs_open"]
    )
    metric = _paired_metric_fn("_candidate_correct", PRODUCTION_CORRECT_COL)
    week_bs = week_blocked_bootstrap(working, metric, block="week", samples=samples, seed=seed)
    season_bs = week_blocked_bootstrap(working, metric, block="season", samples=samples, seed=seed)
    week_row = week_bs.loc[week_bs["metric"].eq("paired_delta")].iloc[0]
    season_row = season_bs.loc[season_bs["metric"].eq("paired_delta")].iloc[0]
    both = working.dropna(subset=["_candidate_correct", PRODUCTION_CORRECT_COL])
    n_weeks = int(both[["season", "week"]].drop_duplicates().shape[0])
    permutation = null_distribution(
        working, candidate_col=candidate_col, permutations=NULL_PERMUTATIONS, seed=seed
    )
    return {
        "cell": name,
        "n_population": len(frame),
        "n_graded": len(both),
        "n_weeks": n_weeks,
        "candidate_accuracy": float(both["_candidate_correct"].mean()),
        "production_accuracy": float(both[PRODUCTION_CORRECT_COL].mean()),
        "paired_delta_points": float(row_or_nan(week_row, "estimate") * 100.0),
        "week_blocked_ci95_points": [
            float(week_row["lower"] * 100.0),
            float(week_row["upper"] * 100.0),
        ],
        "week_blocked_probability_positive": float(week_row["probability_positive"]),
        "season_blocked_ci95_points": [
            float(season_row["lower"] * 100.0),
            float(season_row["upper"] * 100.0),
        ],
        "season_blocked_probability_positive": float(season_row["probability_positive"]),
        "permutation_null": permutation,
    }


def row_or_nan(row: pd.Series, key: str) -> float:
    value = row[key]
    return float(value) if pd.notna(value) else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--game-features", type=Path, default=DEFAULT_GAME_FEATURES)
    parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY_ROOT / "rotation_registry.json"
    )
    parser.add_argument("--out", default="", help="artifact directory name override")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    family = registry.families[ROTATION_FAMILY]
    window = family.assigned_window
    if window is None:
        raise SystemExit(f"{ROTATION_FAMILY} has no assigned window; run `rotation assign` first")
    seasons = (window.seasons[0], window.seasons[1])
    print(f"family={ROTATION_FAMILY} window seasons={seasons}")

    started = time.time()
    base = load_window_population(args.archive, seasons)
    print(f"base window population: {len(base)} games")

    checkpoints = load_timing_checkpoints(args.market_root, args.game_features, seasons)
    merged = base.merge(
        checkpoints[["game_id", "thu_pre_tnf", "sat_midday"]],
        on="game_id",
        how="left",
    ).rename(
        columns={"thu_pre_tnf": "thu_pre_tnf_home_spread", "sat_midday": "sat_midday_home_spread"}
    )

    production_home = merged[PRODUCTION_PICK_COL].astype(bool)
    tue_open = merged["tue_open_home_spread"]

    # Cell 1: reproduction/consistency check, close timing, 1.0 threshold, window-only.
    merged["_pick_close_thr_1_0"] = threshold_pick(
        merged["close_home_spread"], tue_open, production_home, 1.0
    )
    cell_1_pop = merged

    # Cells 2-3: Thursday-pre-TNF timing checkpoint (oracle + 1.0 threshold).
    thu_pop = merged.loc[merged["thu_pre_tnf_home_spread"].notna()].reset_index(drop=True)
    thu_pop["_pick_thu_oracle"] = oracle_pick(
        thu_pop["thu_pre_tnf_home_spread"],
        thu_pop["tue_open_home_spread"],
        thu_pop[PRODUCTION_PICK_COL],
    )
    thu_pop["_pick_thu_thr_1_0"] = threshold_pick(
        thu_pop["thu_pre_tnf_home_spread"],
        thu_pop["tue_open_home_spread"],
        thu_pop[PRODUCTION_PICK_COL],
        1.0,
    )

    # Cell 4: Saturday-midday timing checkpoint, 1.0 threshold.
    sat_pop = merged.loc[merged["sat_midday_home_spread"].notna()].reset_index(drop=True)
    sat_pop["_pick_sat_thr_1_0"] = threshold_pick(
        sat_pop["sat_midday_home_spread"],
        sat_pop["tue_open_home_spread"],
        sat_pop[PRODUCTION_PICK_COL],
        1.0,
    )

    # Cell 5: close timing, 2.0 threshold (untested magnitude tier).
    merged["_pick_close_thr_2_0"] = threshold_pick(
        merged["close_home_spread"], tue_open, production_home, 2.0
    )

    cells = [
        score_cell(
            cell_1_pop,
            name="movement_expansion_window_close_threshold_1_0",
            candidate_col="_pick_close_thr_1_0",
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
        score_cell(
            thu_pop,
            name="movement_expansion_thu_oracle_full_slate",
            candidate_col="_pick_thu_oracle",
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
        score_cell(
            thu_pop,
            name="movement_expansion_thu_threshold_1_0",
            candidate_col="_pick_thu_thr_1_0",
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
        score_cell(
            sat_pop,
            name="movement_expansion_sat_threshold_1_0",
            candidate_col="_pick_sat_thr_1_0",
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
        score_cell(
            merged,
            name="movement_expansion_close_threshold_2_0",
            candidate_col="_pick_close_thr_2_0",
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
    ]

    # Positive control: perfect-foresight pick (deliberate leak of the
    # settlement outcome), NOT recorded to weak_signals -- instrument
    # sensitivity diagnostic only.
    control_frame = merged.copy()
    control_frame["_pick_perfect_foresight"] = control_frame["margin_vs_open"].gt(0.0)
    control = score_cell(
        control_frame,
        name="movement_expansion_positive_control_perfect_foresight",
        candidate_col="_pick_perfect_foresight",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    for cell in [*cells, control]:
        print(
            f"{cell['cell']}: n={cell['n_graded']} delta={cell['paired_delta_points']:+.4f}pts "
            f"weekCI={cell['week_blocked_ci95_points']} "
            f"P+={cell['week_blocked_probability_positive']:.4f} "
            f"seasonP+={cell['season_blocked_probability_positive']:.4f}"
        )

    run_dir_name = args.out or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = DEFAULT_OUTPUT_ROOT / run_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    merged.to_parquet(output_dir / "per_game.parquet", index=False)
    cells_df = pd.DataFrame(
        [
            {
                "cell": c["cell"],
                "n_population": c["n_population"],
                "n_graded": c["n_graded"],
                "n_weeks": c["n_weeks"],
                "candidate_accuracy": c["candidate_accuracy"],
                "production_accuracy": c["production_accuracy"],
                "paired_delta_points": c["paired_delta_points"],
                "week_ci_low": c["week_blocked_ci95_points"][0],
                "week_ci_high": c["week_blocked_ci95_points"][1],
                "week_probability_positive": c["week_blocked_probability_positive"],
                "season_ci_low": c["season_blocked_ci95_points"][0],
                "season_ci_high": c["season_blocked_ci95_points"][1],
                "season_probability_positive": c["season_blocked_probability_positive"],
                "null_mean_delta_points": c["permutation_null"]["null_mean_delta_points"],
                "null_q025_points": c["permutation_null"]["null_q025_points"],
                "null_q975_points": c["permutation_null"]["null_q975_points"],
                "null_fraction_below_observed": c["permutation_null"][
                    "fraction_of_null_below_observed"
                ],
            }
            for c in [*cells, control]
        ]
    )
    cells_df.to_csv(output_dir / "cells_summary.csv", index=False)

    configuration = {
        "rotation_family": ROTATION_FAMILY,
        "window_seasons": list(seasons),
        "archive": str(args.archive),
        "market_root": str(args.market_root),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "null_permutations": NULL_PERMUTATIONS,
        "cells": [c["cell"] for c in cells],
    }
    metadata = {
        "command": "movement-expansion-battery",
        "predeclaration": "docs/movement_expansion_battery.md",
        "cells": cells,
        "positive_control": control,
        "elapsed_seconds": time.time() - started,
        "provenance": artifact_provenance(configuration, args.archive, project_root=REPO_ROOT),
    }
    write_experiment_artifact(
        output_dir,
        "metadata.json",
        metadata,
        command="movement-expansion-battery",
        metrics={"cells": cells, "positive_control": control},
        notes="Predeclared movement magnitude/timing expansion battery; "
        "docs/movement_expansion_battery.md",
        source="docs/movement_expansion_battery.md",
        rotation_family=ROTATION_FAMILY,
        project_root=REPO_ROOT,
        registry_root=DEFAULT_REGISTRY_ROOT,
    )
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
