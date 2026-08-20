"""Follow-up battery (4 predeclared cells) deepening the strongest cell from
the 2026-08-19 Wikipedia-pageview attention pilot
(``attention_battery_both_cold``, +0.52 full-slate accuracy points, P+
0.857 -- read from ``artifacts/attention_battery/20260819T155949Z/results.json``).

**Pilot / measure-only**, same posture as the parent battery: this script
never writes to ``registry/weak_signals.json``; recording happens via
``scripts/record_attention_followup.py`` against this script's output. It
writes an automatic experiment-provenance stamp to ``registry/experiments/``
via ``write_experiment_artifact`` -- a run log, not a verdict.

The full predeclaration (mechanism, exact cell definitions, predicted sign
per cell, units caveats) is frozen in ``docs/attention_followup.md`` and
``<scratchpad>/agent_attention_followup/predeclaration.md``, written before
this script scored anything. This module implements exactly what that
document specifies. Per AGENTS.md, every cell here is predeclared to record
``unresolved_below_power`` regardless of shape (mined, uncorrected
multiplicity, 4 cells).

**Reuses ``scripts/attention_battery_screen.py`` by import** (data loading,
attention_z construction, ``block_bootstrap_two_group``,
``summarize_population``) -- not copy-pasted, not edited -- to guarantee
bit-identical attention_z values to the parent battery.

Data sources: identical to the parent battery (see its docstring) -- same
Wikimedia pageviews raw JSON, same schedules.parquet snapshot.

Writes JSON to ``artifacts/attention_followup/<UTC timestamp>/results.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402


def _load_parent_module():
    spec = importlib.util.spec_from_file_location(
        "attention_battery_screen", REPO / "scripts" / "attention_battery_screen.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_parent_module()

BOOTSTRAP_SAMPLES = base.BOOTSTRAP_SAMPLES
BOOTSTRAP_SEED = base.BOOTSTRAP_SEED

DESCRIPTION_SUFFIX = (
    " mined follow-up battery, uncorrected multiplicity; deepens "
    "attention_battery_both_cold (+0.52pt, P+0.857); Wikipedia pageview "
    "attention proxy, window ends Tuesday of game week (point-in-time safe)."
)


# --------------------------------------------------------------------------
# Extra game-level fields the parent battery doesn't compute
# --------------------------------------------------------------------------


def attach_followup_fields(
    games: pd.DataFrame, long_df: pd.DataFrame, game_df: pd.DataFrame
) -> pd.DataFrame:
    home_side = long_df.loc[long_df["is_home"]].set_index("game_id")
    away_side = long_df.loc[~long_df["is_home"]].set_index("game_id")

    out = game_df.set_index("game_id").copy()
    out["home_trailing_mean"] = home_side["trailing_mean"]
    out["away_trailing_mean"] = away_side["trailing_mean"]

    weekday = out["weekday"].astype(str)
    gametime = out["gametime"].astype(str)
    out["is_primetime"] = weekday.isin(["Thursday", "Saturday", "Monday"]) | (
        (weekday == "Sunday") & (gametime >= "20:00")
    )

    out["combined_z"] = out["home_z"] + out["away_z"]

    return out.reset_index()


# --------------------------------------------------------------------------
# Cells 1-3: subset-vs-complement, reuse base.summarize_population verbatim
# --------------------------------------------------------------------------


def build_subset_cells(game_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}

    both_baseline = (
        game_df["home_has_baseline"].fillna(False) & game_df["away_has_baseline"].fillna(False)
    ).to_numpy(dtype=bool)
    both_cold = ((game_df["home_z"] <= -0.5) & (game_df["away_z"] <= -0.5)).to_numpy(dtype=bool)

    market_size_proxy = (
        game_df["home_trailing_mean"].to_numpy(dtype=float)
        + game_df["away_trailing_mean"].to_numpy(dtype=float)
    ) / 2.0
    eligible_market = both_baseline & np.isfinite(market_size_proxy)
    median_market = float(np.median(market_size_proxy[eligible_market]))
    small_market = market_size_proxy <= median_market
    cells["both_cold_small_market"] = {
        "eligible": eligible_market,
        "flag": both_cold & small_market,
        "value_col": "home_cover",
        "sign": -1,
        "description": (
            "both_cold (home_z<=-0.5 AND away_z<=-0.5) AND market_size_proxy "
            f"(mean of both teams' trailing raw pageview baseline) <= its "
            f"eligible-population median ({median_market:.1f} views); "
            "response home_cover; predicted SAME sign, LARGER magnitude than "
            "the parent both_cold cell (persistently small audience, not a "
            "one-week dip)." + DESCRIPTION_SUFFIX
        ),
        "median_market_size_proxy": median_market,
    }

    weekday_gametime_notna = game_df["weekday"].notna() & game_df["gametime"].notna()
    eligible_primetime = both_baseline & weekday_gametime_notna.to_numpy(dtype=bool)
    non_primetime = (~game_df["is_primetime"]).to_numpy(dtype=bool)
    cells["both_cold_non_primetime"] = {
        "eligible": eligible_primetime,
        "flag": both_cold & non_primetime,
        "value_col": "home_cover",
        "sign": -1,
        "description": (
            "both_cold AND NOT primetime (weekday in "
            "{Thursday,Saturday,Monday} OR Sunday with gametime>=20:00); "
            "response home_cover; predicted SAME sign, LARGER magnitude than "
            "the parent both_cold cell (schedule-driven exposure should "
            "dilute a low-attention mechanism)." + DESCRIPTION_SUFFIX
        ),
    }

    eligible_visitor = game_df["away_has_baseline"].fillna(False).to_numpy(dtype=bool)
    cold_visitor_only = (game_df["away_z"] <= -0.5).to_numpy(dtype=bool)
    cells["cold_visitor_only"] = {
        "eligible": eligible_visitor,
        "flag": cold_visitor_only,
        "value_col": "home_cover",
        "sign": -1,
        "description": (
            "away_z <= -0.5 only (home team's z unconstrained; eligibility "
            "relaxed to away_has_baseline only -- a broader population than "
            "the parent both_cold cell, by design, to test which side drives "
            "the effect); response home_cover; predicted SAME sign, SMALLER "
            "magnitude than the parent both_cold cell." + DESCRIPTION_SUFFIX
        ),
    }

    return cells


# --------------------------------------------------------------------------
# Cell 4: continuous tilt via block-bootstrapped OLS slope
# --------------------------------------------------------------------------


def block_bootstrap_slope(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    block_col: str,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Block-bootstrapped OLS slope of y on x, resampling whole blocks with
    replacement (same joint multinomial scheme as
    ``base.block_bootstrap_two_group``), recombining per-block sufficient
    statistics (n, Sx, Sy, Sxy, Sxx) into a slope each draw."""

    blocks, block_index = np.unique(df[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    x = df[x_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)

    n_b = np.bincount(block_index, minlength=block_count).astype(np.float64)
    sx_b = np.bincount(block_index, weights=x, minlength=block_count).astype(np.float64)
    sy_b = np.bincount(block_index, weights=y, minlength=block_count).astype(np.float64)
    sxy_b = np.bincount(block_index, weights=x * y, minlength=block_count).astype(np.float64)
    sxx_b = np.bincount(block_index, weights=x * x, minlength=block_count).astype(np.float64)

    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)

    n = drawn @ n_b
    sx = drawn @ sx_b
    sy = drawn @ sy_b
    sxy = drawn @ sxy_b
    sxx = drawn @ sxx_b

    denom = n * sxx - sx * sx
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (n * sxy - sx * sy) / denom
    valid = (n > 1) & np.isfinite(slope) & (np.abs(denom) > 1e-9)
    return slope[valid]


def summarize_slope_cell(
    game_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    both_cold_flag: np.ndarray,
    samples: int,
    seed: int,
    block_col: str,
) -> dict[str, Any]:
    both_baseline = (
        game_df["home_has_baseline"].fillna(False) & game_df["away_has_baseline"].fillna(False)
    ).to_numpy(dtype=bool)
    eligible = both_baseline & game_df[y_col].notna().to_numpy(dtype=bool)
    work = game_df.loc[eligible].copy()
    x = work[x_col].to_numpy(dtype=np.float64)
    y = work[y_col].to_numpy(dtype=np.float64)

    n = len(work)
    sx, sy = x.sum(), y.sum()
    sxy = float((x * y).sum())
    sxx = float((x * x).sum())
    denom = n * sxx - sx * sx
    point_slope = (n * sxy - sx * sy) / denom if denom else float("nan")

    mean_combined_z_subset = float(x[both_cold_flag[eligible]].mean())
    mean_combined_z_population = float(x.mean())
    anchor_gap = mean_combined_z_subset - mean_combined_z_population
    point_anchored_effect_pts = point_slope * 100.0 * anchor_gap

    draws = block_bootstrap_slope(
        work, x_col=x_col, y_col=y_col, block_col=block_col, samples=samples, seed=seed
    )
    dropped = samples - len(draws)
    anchored_draws = draws * 100.0 * anchor_gap
    slope_pts_draws = draws * 100.0
    lower_anchored, upper_anchored = (
        np.quantile(anchored_draws, [0.025, 0.975]) if len(anchored_draws) else (np.nan, np.nan)
    )
    lower_slope, upper_slope = (
        np.quantile(slope_pts_draws, [0.025, 0.975]) if len(slope_pts_draws) else (np.nan, np.nan)
    )
    # Predicted sign is positive (higher combined_z -> higher home_cover).
    prob_positive_anchored = float(np.mean(anchored_draws > 0)) if len(anchored_draws) else np.nan

    return {
        "n_total": n,
        "n_blocks": int(work[block_col].nunique()),
        "mean_combined_z_subset_both_cold": mean_combined_z_subset,
        "mean_combined_z_population": mean_combined_z_population,
        "anchor_gap": anchor_gap,
        "raw_slope_pts_per_z": point_slope * 100.0,
        "raw_slope_ci95_pts_per_z": [float(lower_slope), float(upper_slope)],
        "full_slate_effect_pts": point_anchored_effect_pts,
        "week_blocked_ci95_scaled": [float(lower_anchored), float(upper_anchored)],
        "probability_positive": prob_positive_anchored,
        "bootstrap_samples": samples,
        "dropped_draws": int(dropped),
        "insufficient_data": False,
        "units_note": (
            "NOT a raw subset-vs-complement full-slate gap like cells 1-3 or "
            "the parent battery -- this is a regression-slope-implied point "
            "estimate anchored at the both_cold subset's own empirical "
            "combined_z depth. See docs/attention_followup.md cell 4."
        ),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=base.DEFAULT_SCRATCH)
    parser.add_argument("--schedules", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    schedules_path = args.schedules or base._latest_schedules()
    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "attention_followup" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading pageviews from {args.scratch} + {schedules_path} ===")
    team_views = base.load_team_daily_views(args.scratch)
    games = base.load_games(schedules_path)
    print(f"REG {base.SEASON_START}-{base.SEASON_END} games with spread_line: {len(games)}")

    long_df = base.build_team_game_long(games, team_views)
    game_df = base.attach_game_level(games, long_df)
    # game_df already carries weekday/gametime -- attach_game_level's `out`
    # starts as a copy of `games`, which includes every schedules.parquet
    # column (verified: no merge needed).
    game_df = attach_followup_fields(games, long_df, game_df)

    results = []

    subset_cells = build_subset_cells(game_df)
    for name, spec in subset_cells.items():
        print(f"\n=== {name} ===")
        value_col = spec["value_col"]
        full_eligible_df = game_df.loc[spec["eligible"]].copy()
        full_eligible_df["_flag"] = spec["flag"][spec["eligible"]]
        full_eligible_df = full_eligible_df.loc[full_eligible_df[value_col].notna()].reset_index(
            drop=True
        )
        flag = full_eligible_df["_flag"]

        primary = base.summarize_population(
            full_eligible_df,
            flag=flag,
            value_col=value_col,
            block_col="week_block",
            sign=spec["sign"],
            samples=args.samples,
            seed=args.seed,
        )
        secondary = base.summarize_population(
            full_eligible_df,
            flag=flag,
            value_col=value_col,
            block_col="season",
            sign=spec["sign"],
            samples=args.samples,
            seed=args.seed,
        )
        results.append(
            {
                "name": f"attention_followup_{name}",
                "description": spec["description"],
                "sign_dir": spec["sign"],
                "value_col": value_col,
                "week_blocked_primary": primary,
                "season_blocked_secondary": secondary,
            }
        )
        if primary.get("insufficient_data"):
            print("  insufficient data (empty subset or complement)")
            continue
        print(
            f"  n_flag={primary['n_flag']} n_total={primary['n_total']} "
            f"full_slate_effect={primary['full_slate_effect_pts']:+.4f}pts "
            f"P+={primary['probability_positive']:.4f} "
            f"CI=[{primary['week_blocked_ci95_scaled'][0]:+.4f}, "
            f"{primary['week_blocked_ci95_scaled'][1]:+.4f}]"
        )

    print("\n=== deep_cold_tilt (continuous) ===")
    both_cold_flag = ((game_df["home_z"] <= -0.5) & (game_df["away_z"] <= -0.5)).to_numpy(
        dtype=bool
    )
    slope_primary = summarize_slope_cell(
        game_df,
        x_col="combined_z",
        y_col="home_cover",
        both_cold_flag=both_cold_flag,
        samples=args.samples,
        seed=args.seed,
        block_col="week_block",
    )
    slope_secondary = summarize_slope_cell(
        game_df,
        x_col="combined_z",
        y_col="home_cover",
        both_cold_flag=both_cold_flag,
        samples=args.samples,
        seed=args.seed,
        block_col="season",
    )
    results.append(
        {
            "name": "attention_followup_deep_cold_tilt",
            "description": (
                "Continuous version: block-bootstrapped OLS slope of home_cover "
                "on combined_z (=home_z+away_z) over the full both_baseline-"
                "eligible population; effect anchored at the both_cold subset's "
                "own empirical combined_z depth (see units_note). predicted sign "
                "positive (higher combined_z -> higher home_cover)." + DESCRIPTION_SUFFIX
            ),
            "sign_dir": 1,
            "value_col": "home_cover",
            "week_blocked_primary": slope_primary,
            "season_blocked_secondary": slope_secondary,
        }
    )
    print(
        f"  n_total={slope_primary['n_total']} "
        f"raw_slope={slope_primary['raw_slope_pts_per_z']:+.4f}pts/z "
        f"anchor_gap={slope_primary['anchor_gap']:+.4f} "
        f"anchored_effect={slope_primary['full_slate_effect_pts']:+.4f}pts "
        f"P+={slope_primary['probability_positive']:.4f} "
        f"CI=[{slope_primary['week_blocked_ci95_scaled'][0]:+.4f}, "
        f"{slope_primary['week_blocked_ci95_scaled'][1]:+.4f}]"
    )

    configuration = {
        "command": "attention-followup-screen",
        "schedules": str(schedules_path),
        "scratch": str(args.scratch),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": base.SEASON_START,
        "season_end": base.SEASON_END,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "season_start": base.SEASON_START,
        "season_end": base.SEASON_END,
        "n_cells": len(results),
        "n_reg_games": len(games),
        "parent_cell": "attention_battery_both_cold",
        "parent_artifact": "artifacts/attention_battery/20260819T155949Z/results.json",
        "predeclaration": "docs/attention_followup.md (frozen before scoring)",
        "results": results,
        "provenance": artifact_provenance(configuration, schedules_path, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="attention-followup-screen",
        metrics=payload,
        notes=(
            "Follow-up battery (4 predeclared cells) deepening "
            "attention_battery_both_cold; mined family, every cell predeclared "
            "to record unresolved_below_power via a separate nfl-ats "
            "weak-signals record call regardless of interval shape (AGENTS.md)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
