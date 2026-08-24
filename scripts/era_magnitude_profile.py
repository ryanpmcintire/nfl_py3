"""Era-magnitude profile: MEASURE-ONLY re-slice of seven already-built signals
by calendar era, per the owner's standing hypothesis that effects vary in
MAGNITUDE across eras (not existence) -- predeclared in full in
``docs/era_magnitude_profile.md`` before this script scored anything.

Every construct is reused by IMPORT from its established source, never
re-derived:

- ``surface_switch``            -- ``scripts/nfl_weather_battery_screen.py``
- ``division_revenge_game``     -- ``nfl_ats.experiment_runner.FLAG_BUILDERS``
- ``home_underdog``             -- ``nfl_ats.experiment_runner.FLAG_BUILDERS``
- ``extra_rest_edge``           -- ``nfl_ats.experiment_runner.FLAG_BUILDERS``
- ``penalty_rate_quartile``     -- ``nfl_ats.experiment_runner.FLAG_BUILDERS``
- ``hc_year_one_fade``          -- ``scripts/hc_year_one_fade.py`` (ported onto
  the same generic subset-vs-complement pipeline the other five use, for
  era-slicing consistency; the ORIGINAL cluster-bootstrap script is untouched)
- production model's own opener-proxy edge -- re-sliced from
  ``artifacts/proxy_opener_replication/20260819T194330Z/main_scored.parquet``
  (proxy-opener, 2011-2019) and
  ``artifacts/opener_evaluation/20260819T174244Z/per_game.parquet``
  (true-opener, 2020-2025). No walk-forward is re-run.

For each of the six subset-vs-complement signals, this script computes, using
the EXACT ``scale_subset_effect`` convention and the EXACT vectorized joint
week-block bootstrap (``_block_bootstrap_subset_gap``) already used by
``nfl_ats.experiment_runner``:

1. Three fixed-era slices (2009-2014 / 2015-2019 / 2020-2025) -- a reporting
   convenience, not a mechanistic claim.
2. A continuous per-season point series, plus an OLS season-trend slope with
   its own week-block bootstrap interval and ``probability_positive``.
3. A free-break single-changepoint search on the real per-season series, with
   a bootstrap DISTRIBUTION of the break season (reusing the same draws as
   #2 -- no second resample), reported as a spread when unstable rather than
   forced to one answer.
4. A weighted-OLS regression of the per-season effect on ONE declared,
   mechanistic, league-level modulator series (see the doc's Stage 2 table),
   weights = inverse variance of that season's own bootstrap draws.

The production model's own edge (signal 7) gets the same four treatments via
``nfl_ats.clv.week_blocked_bootstrap`` on its own already-scored per-game
frame (pooled accuracy vs. 50% coin flip, in accuracy points), since it is
not a subset-vs-complement construct.

Every number is reported with an interval regardless of sign. Per AGENTS.md,
an interval crossing zero is NEVER grounds to reject a shape; a weaker- or
wrong-signed era reading is reported as exactly that, never as absence.

Writes ``artifacts/era_magnitude_profile/<UTC timestamp>/results.json``.
Registry recording happens in a SEPARATE script
(``scripts/era_magnitude_profile_record.py``), reading this artifact, run
only after this script's output has been read back and reviewed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import hc_year_one_fade as hc_module  # noqa: E402
import nfl_weather_battery_screen as weather_battery  # noqa: E402

from nfl_ats.clv import week_blocked_bootstrap  # noqa: E402
from nfl_ats.experiment_runner import (  # noqa: E402
    FLAG_BUILDERS,
    _block_bootstrap_subset_gap,
    _team_season_penalty_rate,
    classify_subset_bias_result,
    scale_subset_effect,
)
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot  # noqa: E402

FEATURES_PATH = REPO / "data" / "processed" / "game_features.parquet"
PROXY_OPENER_ARTIFACT = (
    REPO / "artifacts" / "proxy_opener_replication" / "20260819T194330Z" / "main_scored.parquet"
)
TRUE_OPENER_ARTIFACT = (
    REPO / "artifacts" / "opener_evaluation" / "20260819T174244Z" / "per_game.parquet"
)

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819
POPULATION_START = 2009
POPULATION_END = 2025
FIXED_ERAS = (
    ((2009, 2014), "era_2009_2014"),
    ((2015, 2019), "era_2015_2019"),
    (
        (2020, 2025),
        "era_2020_2025",
    ),
)
MIN_SEGMENT_SEASONS = 3


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Construct wrapper: a common shape every subset-vs-complement signal fits
# ---------------------------------------------------------------------------


@dataclass
class Construct:
    name: str
    table: pd.DataFrame
    flag: pd.Series
    eligible: pd.Series | None
    sign: int
    response_col: str
    description: str
    week_block_col: str = "week_block"
    season_col: str = "season"
    reliability: float | None = None


def _restrict_seasons(construct: Construct, lo: int, hi: int) -> Construct:
    mask = construct.table[construct.season_col].between(lo, hi)
    return Construct(
        name=construct.name,
        table=construct.table.loc[mask],
        flag=construct.flag.loc[mask],
        eligible=None if construct.eligible is None else construct.eligible.loc[mask],
        sign=construct.sign,
        response_col=construct.response_col,
        description=construct.description,
        week_block_col=construct.week_block_col,
        season_col=construct.season_col,
        reliability=construct.reliability,
    )


# ---------------------------------------------------------------------------
# Building the six subset-vs-complement constructs
# ---------------------------------------------------------------------------


def build_surface_switch() -> Construct:
    df = weather_battery.load_population(weather_battery.default_schedules())
    cells = weather_battery.build_cells(df)
    cell = cells["weather_battery_surface_switch_grass_to_turf"]
    df = df.loc[df["season"].between(POPULATION_START, POPULATION_END)]
    flag = cell["flag"].loc[df.index]
    return Construct(
        name="surface_switch",
        table=df,
        flag=flag,
        eligible=None,
        sign=1,
        response_col="home_cover",
        description=cell["description"],
    )


def build_flag_builder_construct(name: str, key: str) -> Construct:
    features = pd.read_parquet(FEATURES_PATH)
    builder = FLAG_BUILDERS[key]
    raw = builder.build(features, (POPULATION_START, POPULATION_END), {}, REPO)
    mask = raw.table["season"].between(POPULATION_START, POPULATION_END)
    return Construct(
        name=name,
        table=raw.table.loc[mask],
        flag=raw.flag.loc[mask],
        eligible=None if raw.eligible is None else raw.eligible.loc[mask],
        sign=raw.sign,
        response_col="team_covered",
        description=builder.description,
        reliability=raw.reliability,
    )


def build_hc_year_one_fade() -> Construct:
    schedules = pd.read_parquet(hc_module.default_schedules())
    features = pd.read_parquet(hc_module.DEFAULT_FEATURES)
    long = hc_module.build_team_game_table(schedules, features)
    primary = hc_module.team_season_primary_coach(long)
    tenure = hc_module.flag_year_one(primary)

    weeks = long.loc[long["game_type"].eq("REG") & long["week"].le(8)].copy()
    weeks = weeks.merge(tenure, on=["team", "season"], how="inner")
    weeks = weeks.loc[weeks["team_covered"].notna()].copy()
    weeks["week_block"] = weeks["season"].astype(int) * 100 + weeks["week"].astype(int)
    weeks = weeks.loc[weeks["season"].between(POPULATION_START, POPULATION_END)].reset_index(
        drop=True
    )

    flag = weeks["year_one"].fillna(False).astype(bool)
    eligible = flag | weeks["kept_coach"].fillna(False).astype(bool)
    return Construct(
        name="hc_year_one_fade",
        table=weeks,
        flag=flag,
        eligible=eligible,
        sign=-1,  # raw gap (year_one - kept_coach) is negative; sign flips to favour the hypothesis
        response_col="team_covered",
        description=(
            "First-year HC (weeks 1-8, known tenure) vs kept-coach complement, two-sided design "
            "(ported from scripts/hc_year_one_fade.py onto the generic subset-vs-complement "
            "pipeline for era-slicing consistency)."
        ),
    )


def build_all_constructs() -> dict[str, Construct]:
    _log("=== building constructs ===")
    constructs = {}
    constructs["surface_switch"] = build_surface_switch()
    _log(f"  surface_switch: {len(constructs['surface_switch'].table)} rows")
    constructs["division_revenge_game"] = build_flag_builder_construct(
        "division_revenge_game", "division_revenge_game"
    )
    _log(f"  division_revenge_game: {len(constructs['division_revenge_game'].table)} rows")
    constructs["home_underdog"] = build_flag_builder_construct("home_underdog", "home_underdog")
    _log(f"  home_underdog: {len(constructs['home_underdog'].table)} rows")
    constructs["extra_rest_edge"] = build_flag_builder_construct(
        "extra_rest_edge", "extra_rest_edge"
    )
    _log(f"  extra_rest_edge: {len(constructs['extra_rest_edge'].table)} rows")
    constructs["penalty_rate_quartile"] = build_flag_builder_construct(
        "penalty_rate_quartile", "penalty_rate_quartile"
    )
    _log(f"  penalty_rate_quartile: {len(constructs['penalty_rate_quartile'].table)} rows")
    constructs["hc_year_one_fade"] = build_hc_year_one_fade()
    _log(f"  hc_year_one_fade: {len(constructs['hc_year_one_fade'].table)} rows")
    return constructs


# ---------------------------------------------------------------------------
# Stage 1a: three fixed-era slices
# ---------------------------------------------------------------------------


def era_summary(
    construct: Construct, season_lo: int, season_hi: int, *, samples: int, seed: int
) -> dict[str, Any]:
    restricted = _restrict_seasons(construct, season_lo, season_hi)
    table, flag, eligible = restricted.table, restricted.flag, restricted.eligible
    n_total = len(table)
    comparison = table if eligible is None else table.loc[eligible]
    comparison_flag = flag if eligible is None else flag.loc[eligible]
    n_flag = int(comparison_flag.sum())
    n_complement = len(comparison) - n_flag
    if n_total == 0 or n_flag == 0 or n_complement == 0:
        return {
            "season_lo": season_lo,
            "season_hi": season_hi,
            "insufficient_data": True,
            "n_total": n_total,
            "n_flag": n_flag,
            "n_complement": n_complement,
        }

    subset_cover = float(comparison.loc[comparison_flag, construct.response_col].mean())
    complement_cover = float(comparison.loc[~comparison_flag, construct.response_col].mean())
    raw_gap_fraction = subset_cover - complement_cover
    slate_numerator = n_flag if eligible is None else len(comparison)
    fraction_of_slate = slate_numerator / n_total
    effect = scale_subset_effect(
        raw_gap_fraction, sign=construct.sign, fraction_of_slate=fraction_of_slate
    )

    def block_result(block_col: str) -> dict[str, Any] | None:
        if comparison[block_col].nunique() < 2:
            return None
        draws = _block_bootstrap_subset_gap(
            comparison,
            flag=comparison_flag,
            value_col=construct.response_col,
            block_col=block_col,
            samples=samples,
            seed=seed,
        )
        if len(draws) == 0:
            return None
        scaled = construct.sign * draws * fraction_of_slate
        lower, upper = np.quantile(scaled, [0.025, 0.975])
        return {
            "block_count": int(comparison[block_col].nunique()),
            "estimate": float(np.mean(scaled)),
            "lower": float(lower),
            "upper": float(upper),
            "probability_positive": float(np.mean(scaled > 0.0)),
            "samples_used": len(scaled),
            "samples_dropped": int(samples - len(scaled)),
        }

    week_blocked = block_result(construct.week_block_col)
    season_blocked = block_result(construct.season_col)
    classification = None
    if week_blocked is not None:
        verdict = classify_subset_bias_result(
            estimate=week_blocked["estimate"],
            lower=week_blocked["lower"],
            upper=week_blocked["upper"],
        )
        classification = {
            "classification": verdict.classification,
            "closing_ground": verdict.closing_ground,
            "note": verdict.note,
            "widening_factor": verdict.widening_factor,
        }

    return {
        "season_lo": season_lo,
        "season_hi": season_hi,
        "insufficient_data": False,
        "n_total": n_total,
        "n_flag": n_flag,
        "n_complement": n_complement,
        "subset_cover": subset_cover,
        "complement_cover": complement_cover,
        "fraction_of_slate": fraction_of_slate,
        "effect": effect,
        "week_blocked": week_blocked,
        "season_blocked": season_blocked,
        "mechanical_classification": classification,
    }


# ---------------------------------------------------------------------------
# Stage 1b/2a/2b: joint per-season bootstrap -> slope, changepoint, modulator
# ---------------------------------------------------------------------------


def _real_per_season(construct: Construct) -> tuple[list[int], list[dict[str, Any]]]:
    """Per-season point effect on the REAL (non-resampled) data."""

    table, flag, eligible = construct.table, construct.flag, construct.eligible
    comparison = table if eligible is None else table.loc[eligible]
    comparison_flag = (flag if eligible is None else flag.loc[eligible]).to_numpy(dtype=bool)
    seasons_all = sorted(int(s) for s in table[construct.season_col].unique())

    rows = []
    for season in seasons_all:
        season_table_mask = table[construct.season_col] == season
        n_total_s = int(season_table_mask.sum())
        season_comparison_mask = (comparison[construct.season_col] == season).to_numpy()
        season_comparison = comparison.loc[season_comparison_mask]
        season_flag = comparison_flag[season_comparison_mask]
        n_flag_s = int(season_flag.sum())
        n_complement_s = len(season_comparison) - n_flag_s
        if n_total_s == 0 or n_flag_s == 0 or n_complement_s == 0:
            rows.append({"season": season, "insufficient_data": True})
            continue
        subset_cover_s = float(season_comparison.loc[season_flag, construct.response_col].mean())
        complement_cover_s = float(
            season_comparison.loc[~season_flag, construct.response_col].mean()
        )
        raw_gap_s = subset_cover_s - complement_cover_s
        slate_numerator_s = n_flag_s if eligible is None else len(season_comparison)
        fraction_s = slate_numerator_s / n_total_s
        effect_s = scale_subset_effect(raw_gap_s, sign=construct.sign, fraction_of_slate=fraction_s)
        rows.append(
            {
                "season": season,
                "insufficient_data": False,
                "n_total": n_total_s,
                "n_flag": n_flag_s,
                "n_complement": n_complement_s,
                "fraction_of_slate": fraction_s,
                "effect": effect_s,
            }
        )
    valid_seasons = [r["season"] for r in rows if not r["insufficient_data"]]
    return valid_seasons, rows


def _joint_season_draws(
    construct: Construct, valid_seasons: list[int], *, samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Vectorized joint week-block bootstrap of the per-season scaled gap.

    Returns (Y, fraction_of_slate_by_season, valid_seasons) where Y has shape
    (n_valid_draws, n_valid_seasons): only draws with a non-empty subset AND
    complement in EVERY listed season are kept (mirrors
    ``_block_bootstrap_subset_gap``'s own "drop degenerate draws" convention,
    generalized from one comparison to many simultaneous ones).
    """

    table, flag, eligible = construct.table, construct.flag, construct.eligible
    comparison = table if eligible is None else table.loc[eligible]
    comparison_flag_full = (flag if eligible is None else flag.loc[eligible]).to_numpy(dtype=bool)
    season_mask = comparison[construct.season_col].isin(valid_seasons).to_numpy()
    comparison = comparison.loc[season_mask]
    comparison_flag = comparison_flag_full[season_mask]

    real_fraction = {}
    table_season_mask_cache = {
        s: int((table[construct.season_col] == s).sum()) for s in valid_seasons
    }
    for s in valid_seasons:
        s_comp_mask = (comparison[construct.season_col] == s).to_numpy()
        n_flag_s = int(comparison_flag[s_comp_mask].sum())
        n_comp_rows_s = int(s_comp_mask.sum())
        slate_numerator_s = n_flag_s if eligible is None else n_comp_rows_s
        real_fraction[s] = slate_numerator_s / table_season_mask_cache[s]

    block_col = construct.week_block_col
    blocks, block_index = np.unique(comparison[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    values = comparison[construct.response_col].to_numpy(dtype=np.float64)

    sums: dict[bool, np.ndarray] = {}
    counts: dict[bool, np.ndarray] = {}
    for group in (True, False):
        m = comparison_flag == group
        sums[group] = np.bincount(block_index[m], weights=values[m], minlength=block_count).astype(
            np.float64
        )
        counts[group] = np.bincount(block_index[m], minlength=block_count).astype(np.float64)

    block_season = (blocks // 100).astype(int)
    season_ids = np.array(sorted(valid_seasons), dtype=int)
    n_seasons = len(season_ids)
    season_index_of_block = np.searchsorted(season_ids, block_season)

    onehot = np.zeros((block_count, n_seasons), dtype=np.float64)
    onehot[np.arange(block_count), season_index_of_block] = 1.0

    m_true_sum = sums[True][:, None] * onehot
    m_true_cnt = counts[True][:, None] * onehot
    m_false_sum = sums[False][:, None] * onehot
    m_false_cnt = counts[False][:, None] * onehot

    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)

    season_true_sum = drawn @ m_true_sum
    season_true_cnt = drawn @ m_true_cnt
    season_false_sum = drawn @ m_false_sum
    season_false_cnt = drawn @ m_false_cnt

    valid_draw_mask = (season_true_cnt > 0).all(axis=1) & (season_false_cnt > 0).all(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_true = season_true_sum / season_true_cnt
        mean_false = season_false_sum / season_false_cnt

    fraction_arr = np.array([real_fraction[s] for s in season_ids])
    season_gap = construct.sign * (mean_true - mean_false) * 100.0 * fraction_arr[None, :]
    y = season_gap[valid_draw_mask]
    return y, fraction_arr, list(season_ids)


def _ols_slope_closed_form(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xbar = float(x.mean())
    denom = float(np.sum((x - xbar) ** 2))
    if denom <= 0:
        return float("nan"), float("nan")
    slope = float(np.sum((x - xbar) * (y - y.mean())) / denom)
    intercept = float(y.mean() - slope * xbar)
    return slope, intercept


def _changepoint_grid(y: np.ndarray, min_seg: int) -> tuple[int, float]:
    """Best break INDEX (segment-2 start) minimizing total SSE, single series."""

    n = len(y)
    best_k, best_sse = -1, np.inf
    for k in range(min_seg, n - min_seg + 1):
        pre, post = y[:k], y[k:]
        sse = float(np.sum((pre - pre.mean()) ** 2) + np.sum((post - post.mean()) ** 2))
        if sse < best_sse:
            best_sse, best_k = sse, k
    return best_k, best_sse


def _changepoint_grid_vectorized(Y: np.ndarray, min_seg: int) -> np.ndarray:
    """Best break INDEX per draw, vectorized over draws. Y: (n_draws, n_seasons)."""

    n_draws, n_seasons = Y.shape
    cumsum = np.cumsum(Y, axis=1)
    cumsum_sq = np.cumsum(Y**2, axis=1)
    total = cumsum[:, -1]
    total_sq = cumsum_sq[:, -1]

    candidates = list(range(min_seg, n_seasons - min_seg + 1))
    sse_matrix = np.empty((n_draws, len(candidates)), dtype=np.float64)
    for idx, k in enumerate(candidates):
        sum1 = cumsum[:, k - 1]
        sq1 = cumsum_sq[:, k - 1]
        sse1 = sq1 - sum1**2 / k
        sum2 = total - sum1
        sq2 = total_sq - sq1
        n2 = n_seasons - k
        sse2 = sq2 - sum2**2 / n2
        sse_matrix[:, idx] = sse1 + sse2
    best_idx = np.argmin(sse_matrix, axis=1)
    return np.array([candidates[i] for i in best_idx])


def season_trend_and_changepoint(
    construct: Construct,
    *,
    samples: int,
    seed: int,
    modulator: dict[int, float] | None,
    modulator_name: str,
    modulator_seasons_only: list[int] | None = None,
) -> dict[str, Any]:
    valid_seasons, real_rows = _real_per_season(construct)
    if len(valid_seasons) < 2:
        return {"insufficient_seasons": True, "valid_seasons": valid_seasons}

    x_real = np.array(valid_seasons, dtype=float)
    y_real = np.array([r["effect"] for r in real_rows if not r["insufficient_data"]], dtype=float)
    slope_point, intercept_point = _ols_slope_closed_form(x_real, y_real)

    Y, _fraction_arr, season_ids = _joint_season_draws(
        construct, valid_seasons, samples=samples, seed=seed
    )
    n_valid_draws = Y.shape[0]
    se_by_season = (
        np.std(Y, axis=0, ddof=1) if n_valid_draws > 1 else np.full(len(season_ids), np.nan)
    )

    xbar = x_real.mean()
    denom = float(np.sum((x_real - xbar) ** 2))
    slope_draws = Y @ (x_real - xbar) / denom if denom > 0 else np.full(n_valid_draws, np.nan)
    slope_lower, slope_upper = (
        np.quantile(slope_draws, [0.025, 0.975]) if n_valid_draws else (np.nan, np.nan)
    )

    result: dict[str, Any] = {
        "insufficient_seasons": False,
        "valid_seasons": valid_seasons,
        "real_per_season": real_rows,
        "n_valid_bootstrap_draws": int(n_valid_draws),
        "n_dropped_bootstrap_draws": int(samples - n_valid_draws),
        "slope": {
            "point_estimate": slope_point,
            "intercept_point_estimate": intercept_point,
            "bootstrap_lower": float(slope_lower),
            "bootstrap_upper": float(slope_upper),
            "probability_positive": float(np.mean(slope_draws > 0.0))
            if n_valid_draws
            else float("nan"),
            "units": "accuracy_points_per_season",
        },
    }

    # --- 2a: free-break changepoint ---
    n_seasons = len(season_ids)
    if n_seasons >= 2 * MIN_SEGMENT_SEASONS:
        best_k, _best_sse = _changepoint_grid(y_real, MIN_SEGMENT_SEASONS)
        break_season_point = season_ids[best_k]
        pre_mean_point = float(y_real[:best_k].mean())
        post_mean_point = float(y_real[best_k:].mean())

        best_k_draws = _changepoint_grid_vectorized(Y, MIN_SEGMENT_SEASONS)
        break_season_draws = np.array([season_ids[k] for k in best_k_draws])
        values_, counts_ = np.unique(break_season_draws, return_counts=True)
        modal_idx = int(np.argmax(counts_))
        pre_draws = Y[:, :best_k].mean(axis=1)
        post_draws = Y[:, best_k:].mean(axis=1)
        result["changepoint"] = {
            "min_segment_seasons": MIN_SEGMENT_SEASONS,
            "break_season_point_estimate": int(break_season_point),
            "pre_break_mean_point": pre_mean_point,
            "post_break_mean_point": post_mean_point,
            "break_season_bootstrap_median": float(np.median(break_season_draws)),
            "break_season_bootstrap_p2_5": float(np.quantile(break_season_draws, 0.025)),
            "break_season_bootstrap_p97_5": float(np.quantile(break_season_draws, 0.975)),
            "break_season_bootstrap_modal": int(values_[modal_idx]),
            "break_season_bootstrap_modal_share": float(
                counts_[modal_idx] / len(break_season_draws)
            ),
            "break_season_bootstrap_distribution": {
                str(int(v)): int(c) for v, c in zip(values_, counts_, strict=True)
            },
            "pre_break_mean_bootstrap_lower": float(np.quantile(pre_draws, 0.025)),
            "pre_break_mean_bootstrap_upper": float(np.quantile(pre_draws, 0.975)),
            "post_break_mean_bootstrap_lower": float(np.quantile(post_draws, 0.025)),
            "post_break_mean_bootstrap_upper": float(np.quantile(post_draws, 0.975)),
            "stable": bool(counts_[modal_idx] / len(break_season_draws) >= 0.25),
        }
    else:
        result["changepoint"] = {
            "insufficient_seasons_for_changepoint": True,
            "n_seasons": n_seasons,
            "min_required": 2 * MIN_SEGMENT_SEASONS,
        }

    # --- 2b: mechanistic modulator regression ---
    if modulator is not None:
        mod_seasons = modulator_seasons_only or valid_seasons
        mod_mask_idx = [i for i, s in enumerate(season_ids) if s in mod_seasons and s in modulator]
        if len(mod_mask_idx) >= 3:
            mod_season_ids = [season_ids[i] for i in mod_mask_idx]
            mod_values = np.array([modulator[s] for s in mod_season_ids])
            y_mod_real = np.array(
                [
                    r["effect"]
                    for r in real_rows
                    if not r["insufficient_data"] and r["season"] in mod_season_ids
                ]
            )
            se_mod = se_by_season[mod_mask_idx]
            weights = 1.0 / np.maximum(se_mod, 1e-9) ** 2
            wbar = float(np.average(mod_values, weights=weights))
            denom_w = float(np.sum(weights * (mod_values - wbar) ** 2))
            if denom_w > 0:
                slope_mod_point = float(
                    np.sum(weights * (mod_values - wbar) * y_mod_real) / denom_w
                )
                coef_vec = weights * (mod_values - wbar) / denom_w
                y_mod_draws = Y[:, mod_mask_idx]
                slope_mod_draws = y_mod_draws @ coef_vec
                lower_m, upper_m = np.quantile(slope_mod_draws, [0.025, 0.975])
                result["modulator"] = {
                    "name": modulator_name,
                    "n_seasons": len(mod_season_ids),
                    "seasons": mod_season_ids,
                    "modulator_values": mod_values.tolist(),
                    "slope_point_estimate": slope_mod_point,
                    "bootstrap_lower": float(lower_m),
                    "bootstrap_upper": float(upper_m),
                    "probability_positive": float(np.mean(slope_mod_draws > 0.0)),
                }
            else:
                result["modulator"] = {"name": modulator_name, "degenerate": True}
        else:
            result["modulator"] = {
                "name": modulator_name,
                "insufficient_overlap": True,
                "n_overlap_seasons": len(mod_mask_idx),
            }
    return result


# ---------------------------------------------------------------------------
# Declared league-level modulator series ("what makes an era")
# ---------------------------------------------------------------------------


def compute_league_series(
    constructs: dict[str, Construct], features: pd.DataFrame, pbp: pd.DataFrame
) -> dict[str, dict[int, float]]:
    reg_features = features.loc[
        (features["game_type"] == "REG")
        & features["season"].between(POPULATION_START, POPULATION_END)
    ].copy()
    reg_features["result"] = pd.to_numeric(reg_features["result"], errors="coerce")

    # 1. league turf share (surface_switch modulator)
    weather_df = constructs["surface_switch"].table
    turf_share = (
        weather_df.assign(is_turf=weather_df["surface_norm"] == "turf")
        .loc[weather_df["surface_norm"].notna()]
        .groupby("season")["is_turf"]
        .mean()
        * 100.0
    )

    # 2. league mean |scoring margin| (division_revenge modulator)
    margin_abs = (
        reg_features.dropna(subset=["result"])
        .groupby("season")["result"]
        .apply(lambda s: s.abs().mean())
    )

    # 3. league mean raw home-field advantage: mean(home - away) (home_underdog modulator)
    home_field_adv = reg_features.dropna(subset=["result"]).groupby("season")["result"].mean()

    # 4. league mean |own_rest - opp_rest| (extra_rest_edge modulator)
    rest_table = constructs["extra_rest_edge"].table
    rest_gap = (
        (rest_table["own_rest"] - rest_table["opp_rest"]).abs().groupby(rest_table["season"]).mean()
    )

    # 5. league mean team-season penalty rate (penalty_rate_quartile modulator)
    penalty_rate = _team_season_penalty_rate(pbp)
    penalty_rate_by_season = penalty_rate.groupby("season")["rate"].mean() * 100.0

    # 6. count of year-one team-seasons per season (hc_year_one_fade modulator)
    schedules = pd.read_parquet(hc_module.default_schedules())
    all_features = pd.read_parquet(hc_module.DEFAULT_FEATURES)
    long = hc_module.build_team_game_table(schedules, all_features)
    primary = hc_module.team_season_primary_coach(long)
    tenure = hc_module.flag_year_one(primary)
    year_one_count = tenure.loc[tenure["year_one"]].groupby("season").size()

    def _to_dict(series: pd.Series) -> dict[int, float]:
        return {int(k): float(v) for k, v in series.items() if int(k) <= POPULATION_END}

    return {
        "league_turf_share_pct": _to_dict(turf_share),
        "league_mean_abs_scoring_margin": _to_dict(margin_abs),
        "league_home_field_advantage": _to_dict(home_field_adv),
        "league_mean_rest_gap": _to_dict(rest_gap),
        "league_mean_penalty_rate_pct": _to_dict(penalty_rate_by_season),
        "league_year_one_hc_count": _to_dict(year_one_count),
    }


MODULATOR_ASSIGNMENT = {
    "surface_switch": "league_turf_share_pct",
    "division_revenge_game": "league_mean_abs_scoring_margin",
    "home_underdog": "league_home_field_advantage",
    "extra_rest_edge": "league_mean_rest_gap",
    "penalty_rate_quartile": "league_mean_penalty_rate_pct",
    "hc_year_one_fade": "league_year_one_hc_count",
}


# ---------------------------------------------------------------------------
# Signal 7: production model's own opener-proxy edge
# ---------------------------------------------------------------------------


def load_signal7_frame() -> pd.DataFrame:
    proxy = pd.read_parquet(PROXY_OPENER_ARTIFACT)[
        ["game_id", "season", "week", "correct_at_open_proxy_pr"]
    ].rename(columns={"correct_at_open_proxy_pr": "correct"})
    proxy["grade"] = "proxy_open"
    proxy["close_books"] = np.nan

    true_ = pd.read_parquet(TRUE_OPENER_ARTIFACT)[
        ["game_id", "season", "week", "correct_at_open_probability_rule", "close_books"]
    ].rename(columns={"correct_at_open_probability_rule": "correct"})
    true_["grade"] = "true_open"

    combined = pd.concat([proxy, true_], ignore_index=True)
    combined["correct"] = pd.to_numeric(combined["correct"], errors="coerce")
    combined = combined.dropna(subset=["correct"]).reset_index(drop=True)
    return combined


def _accuracy_points_metric(frame: pd.DataFrame) -> dict[str, float]:
    return {"accuracy_points": float(frame["correct"].mean()) * 100.0 - 50.0}


def signal7_era_summary(
    combined: pd.DataFrame, season_lo: int, season_hi: int, *, samples: int, seed: int
) -> dict[str, Any]:
    subset = combined.loc[combined["season"].between(season_lo, season_hi)]
    if subset.empty:
        return {"season_lo": season_lo, "season_hi": season_hi, "insufficient_data": True}
    result = week_blocked_bootstrap(
        subset, _accuracy_points_metric, block="week", samples=samples, seed=seed
    ).iloc[0]
    grades_present = sorted(subset["grade"].unique().tolist())
    verdict = classify_subset_bias_result(
        estimate=float(result["estimate"]),
        lower=float(result["lower"]),
        upper=float(result["upper"]),
    )
    return {
        "season_lo": season_lo,
        "season_hi": season_hi,
        "insufficient_data": False,
        "n_games": len(subset),
        "grades_present": grades_present,
        "estimate": float(result["estimate"]),
        "lower": float(result["lower"]),
        "upper": float(result["upper"]),
        "probability_positive": float(result["probability_positive"]),
        "mechanical_classification": {
            "classification": verdict.classification,
            "closing_ground": verdict.closing_ground,
            "note": verdict.note,
            "widening_factor": verdict.widening_factor,
        },
    }


def signal7_real_per_season(combined: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for season, group in combined.groupby("season"):
        rows.append(
            {
                "season": int(season),
                "n_games": len(group),
                "grade": sorted(group["grade"].unique().tolist()),
                "effect": float(group["correct"].mean()) * 100.0 - 50.0,
            }
        )
    return sorted(rows, key=lambda r: r["season"])


def signal7_slope_and_changepoint(
    combined: pd.DataFrame, *, samples: int, seed: int
) -> dict[str, Any]:
    seasons_sorted = sorted(int(s) for s in combined["season"].unique())
    x_real = np.array(seasons_sorted, dtype=float)
    real_rows = signal7_real_per_season(combined)
    y_real = np.array([r["effect"] for r in real_rows], dtype=float)
    slope_point, intercept_point = _ols_slope_closed_form(x_real, y_real)

    def metric_fn(frame: pd.DataFrame) -> dict[str, float]:
        by_season = frame.groupby("season")["correct"].mean() * 100.0 - 50.0
        by_season = by_season.reindex(seasons_sorted)
        if by_season.isna().any():
            # Season entirely missing from this resample (should be rare given
            # ~15-20 week-blocks per season); fall back to the real value so the
            # draw is not silently dropped and does not bias the location.
            by_season = by_season.fillna(pd.Series(y_real, index=seasons_sorted))
        y = by_season.to_numpy(dtype=float)
        xbar = x_real.mean()
        denom = float(np.sum((x_real - xbar) ** 2))
        slope = (
            float(np.sum((x_real - xbar) * (y - y.mean())) / denom) if denom > 0 else float("nan")
        )
        out = {"slope": slope}
        n_seasons = len(seasons_sorted)
        if n_seasons >= 2 * MIN_SEGMENT_SEASONS:
            best_k, _ = _changepoint_grid(y, MIN_SEGMENT_SEASONS)
            out["break_season"] = float(seasons_sorted[best_k])
            out["pre_break_mean"] = float(y[:best_k].mean())
            out["post_break_mean"] = float(y[best_k:].mean())
        return out

    boot = week_blocked_bootstrap(
        combined, metric_fn, block="week", samples=samples, seed=seed
    ).set_index("metric")

    result: dict[str, Any] = {
        "valid_seasons": seasons_sorted,
        "real_per_season": real_rows,
        "slope": {
            "point_estimate": slope_point,
            "intercept_point_estimate": intercept_point,
            "bootstrap_lower": float(boot.loc["slope", "lower"]),
            "bootstrap_upper": float(boot.loc["slope", "upper"]),
            "probability_positive": float(boot.loc["slope", "probability_positive"]),
            "units": "accuracy_points_per_season",
        },
    }
    if "break_season" in boot.index:
        best_k_point, _ = _changepoint_grid(y_real, MIN_SEGMENT_SEASONS)
        result["changepoint"] = {
            "min_segment_seasons": MIN_SEGMENT_SEASONS,
            "break_season_point_estimate": int(seasons_sorted[best_k_point]),
            "pre_break_mean_point": float(y_real[:best_k_point].mean()),
            "post_break_mean_point": float(y_real[best_k_point:].mean()),
            "break_season_bootstrap_estimate": float(boot.loc["break_season", "estimate"]),
            "break_season_bootstrap_lower": float(boot.loc["break_season", "lower"]),
            "break_season_bootstrap_upper": float(boot.loc["break_season", "upper"]),
            "pre_break_mean_bootstrap_lower": float(boot.loc["pre_break_mean", "lower"]),
            "pre_break_mean_bootstrap_upper": float(boot.loc["pre_break_mean", "upper"]),
            "post_break_mean_bootstrap_lower": float(boot.loc["post_break_mean", "lower"]),
            "post_break_mean_bootstrap_upper": float(boot.loc["post_break_mean", "upper"]),
            "note": (
                "Bootstrap CI on break_season here is the plain percentile interval of the "
                "per-draw argmin break season (not a mode/spread report like the six "
                "subset-vs-complement signals); read alongside the point estimate, not instead "
                "of it."
            ),
        }
    else:
        result["changepoint"] = {"insufficient_seasons_for_changepoint": True}
    return result


def signal7_modulator(combined: pd.DataFrame, *, samples: int, seed: int) -> dict[str, Any]:
    true_only = combined.loc[combined["grade"] == "true_open"].dropna(subset=["close_books"])
    if true_only.empty:
        return {"insufficient_data": True}
    seasons_sorted = sorted(int(s) for s in true_only["season"].unique())
    mod_by_season = true_only.groupby("season")["close_books"].mean()
    mod_values = np.array([mod_by_season[s] for s in seasons_sorted])

    real_effect = (
        (true_only.groupby("season")["correct"].mean() * 100.0 - 50.0)
        .reindex(seasons_sorted)
        .to_numpy()
    )

    # week_blocked_bootstrap only returns summary stats, not raw per-draw vectors,
    # and the weighted-slope bootstrap below needs the raw per-season draws (same
    # joint-resample requirement as season_trend_and_changepoint's Y matrix) --
    # so this reimplements its exact resampling loop (same algorithm: draw whole
    # (season, week) blocks with replacement, same RNG convention) restricted to
    # this small (6-season) true-opener-only population, rather than discarding a
    # second, wasted 20,000-sample run just to get the interval-only summary.
    group_columns = ["season", "week"]
    grouped_indices = list(
        true_only.groupby(group_columns, sort=False, dropna=False).indices.values()
    )
    rng = np.random.default_rng(seed)
    n_seasons = len(seasons_sorted)
    y_draws = np.empty((samples, n_seasons), dtype=float)
    for i in range(samples):
        selected = rng.integers(0, len(grouped_indices), size=len(grouped_indices))
        positions = np.concatenate([grouped_indices[j] for j in selected])
        sampled = true_only.iloc[positions]
        by_season = (sampled.groupby("season")["correct"].mean() * 100.0 - 50.0).reindex(
            seasons_sorted
        )
        if by_season.isna().any():
            by_season = by_season.fillna(pd.Series(real_effect, index=seasons_sorted))
        y_draws[i] = by_season.to_numpy(dtype=float)

    se_by_season = np.std(y_draws, axis=0, ddof=1)
    weights = 1.0 / np.maximum(se_by_season, 1e-9) ** 2
    wbar = float(np.average(mod_values, weights=weights))
    denom_w = float(np.sum(weights * (mod_values - wbar) ** 2))
    slope_point = (
        float(np.sum(weights * (mod_values - wbar) * real_effect) / denom_w)
        if denom_w > 0
        else float("nan")
    )
    coef_vec = weights * (mod_values - wbar) / denom_w if denom_w > 0 else np.zeros(n_seasons)
    slope_draws = y_draws @ coef_vec
    lower, upper = np.quantile(slope_draws, [0.025, 0.975])
    return {
        "name": "league_mean_close_books",
        "n_seasons": n_seasons,
        "seasons": seasons_sorted,
        "modulator_values": mod_values.tolist(),
        "slope_point_estimate": slope_point,
        "bootstrap_lower": float(lower),
        "bootstrap_upper": float(upper),
        "probability_positive": float(np.mean(slope_draws > 0.0)),
        "caveat": (
            "6 seasons only (2020-2025) -- the SBR proxy leg (2011-2019) carries no book-count "
            "field. Likely underpowered; reported per the owner's instruction to show the spread "
            "rather than force a conclusion."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "era_magnitude_profile" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    constructs = build_all_constructs()

    _log("\n=== loading league-level modulator series ===")
    features = pd.read_parquet(FEATURES_PATH)
    snapshot = latest_pbp_snapshot(REPO / "data" / "pbp" / "raw")
    pbp = load_pbp_snapshot(snapshot, include_postseason=False)
    league_series = compute_league_series(constructs, features, pbp)
    for name, series in league_series.items():
        _log(f"  {name}: {len(series)} seasons")

    signals: dict[str, Any] = {}
    for name, construct in constructs.items():
        _log(f"\n=== {name} ===")
        era_results = {}
        for (lo, hi), key in FIXED_ERAS:
            era_results[key] = era_summary(construct, lo, hi, samples=args.samples, seed=args.seed)
            r = era_results[key]
            if not r["insufficient_data"]:
                wb = r["week_blocked"]
                _log(
                    f"  {key}: effect={r['effect']:+.4f}pts week-blocked "
                    f"[{wb['lower']:+.4f},{wb['upper']:+.4f}] P+={wb['probability_positive']:.4f} "
                    f"n={r['n_total']}"
                )
            else:
                _log(f"  {key}: insufficient data")

        modulator_name = MODULATOR_ASSIGNMENT[name]
        trend = season_trend_and_changepoint(
            construct,
            samples=args.samples,
            seed=args.seed,
            modulator=league_series[modulator_name],
            modulator_name=modulator_name,
        )
        if not trend.get("insufficient_seasons"):
            slope = trend["slope"]
            _log(
                f"  season-trend slope={slope['point_estimate']:+.4f} pts/season "
                f"[{slope['bootstrap_lower']:+.4f},{slope['bootstrap_upper']:+.4f}] "
                f"P+={slope['probability_positive']:.4f}"
            )
            cp = trend.get("changepoint", {})
            if not cp.get("insufficient_seasons_for_changepoint"):
                _log(
                    f"  changepoint: break={cp['break_season_point_estimate']} "
                    f"modal_bootstrap={cp['break_season_bootstrap_modal']} "
                    f"share={cp['break_season_bootstrap_modal_share']:.3f} "
                    f"pre={cp['pre_break_mean_point']:+.3f} post={cp['post_break_mean_point']:+.3f}"
                )
            mod = trend.get("modulator", {})
            if "slope_point_estimate" in mod:
                _log(
                    f"  modulator({mod['name']}): slope={mod['slope_point_estimate']:+.5f} "
                    f"[{mod['bootstrap_lower']:+.5f},{mod['bootstrap_upper']:+.5f}] "
                    f"P+={mod['probability_positive']:.4f}"
                )

        signals[name] = {
            "description": construct.description,
            "response_col": construct.response_col,
            "sign": construct.sign,
            "reliability": construct.reliability,
            "era_results": era_results,
            "season_trend": trend,
        }

    _log("\n=== signal 7: production model's own opener-proxy edge ===")
    combined7 = load_signal7_frame()
    era7 = {}
    for (lo, hi), key in FIXED_ERAS:
        era7[key] = signal7_era_summary(combined7, lo, hi, samples=args.samples, seed=args.seed)
        r = era7[key]
        if not r["insufficient_data"]:
            _log(
                f"  {key}: estimate={r['estimate']:+.4f}pts [{r['lower']:+.4f},{r['upper']:+.4f}] "
                f"P+={r['probability_positive']:.4f} n={r['n_games']} grades={r['grades_present']}"
            )
        else:
            _log(f"  {key}: insufficient data")

    trend7 = signal7_slope_and_changepoint(combined7, samples=args.samples, seed=args.seed)
    _log(
        f"  season-trend slope={trend7['slope']['point_estimate']:+.4f} pts/season "
        f"[{trend7['slope']['bootstrap_lower']:+.4f},{trend7['slope']['bootstrap_upper']:+.4f}] "
        f"P+={trend7['slope']['probability_positive']:.4f}"
    )
    if not trend7["changepoint"].get("insufficient_seasons_for_changepoint"):
        cp7 = trend7["changepoint"]
        _log(
            f"  changepoint: break={cp7['break_season_point_estimate']} "
            f"pre={cp7['pre_break_mean_point']:+.3f} post={cp7['post_break_mean_point']:+.3f}"
        )

    mod7 = signal7_modulator(combined7, samples=min(args.samples, 5000), seed=args.seed)
    if "slope_point_estimate" in mod7:
        _log(
            f"  modulator(close_books): slope={mod7['slope_point_estimate']:+.5f} "
            f"[{mod7['bootstrap_lower']:+.5f},{mod7['bootstrap_upper']:+.5f}] "
            f"P+={mod7['probability_positive']:.4f} (n_seasons={mod7['n_seasons']})"
        )

    signals["production_model_opener_proxy_edge"] = {
        "description": (
            "Frozen production model (market_residual/weak_stack/ridge alpha=10.0), probability "
            "rule, pooled accuracy vs. 50% coin flip. 2011-2019 graded at the SBR-Open PROXY "
            "opener; 2020-2025 graded at the true Tuesday opener. Re-sliced from already-scored "
            "per-game artifacts; no walk-forward re-run."
        ),
        "era_results": era7,
        "season_trend": trend7,
        "modulator": mod7,
    }

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.time() - started,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "population_start": POPULATION_START,
        "population_end": POPULATION_END,
        "fixed_eras": [
            {"key": key, "season_lo": lo, "season_hi": hi} for (lo, hi), key in FIXED_ERAS
        ],
        "min_segment_seasons": MIN_SEGMENT_SEASONS,
        "predeclaration": "docs/era_magnitude_profile.md (frozen before this script scored anything)",  # noqa: E501
        "league_series": league_series,
        "modulator_assignment": MODULATOR_ASSIGNMENT,
        "signals": signals,
    }
    output_path = output_dir / "results.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, default=float), encoding="utf-8"
    )
    _log(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
