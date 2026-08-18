"""Screen a per-metric ``offseason_retention`` vector against the shipped scalar.

Predeclared BEFORE any arm was scored: see the scratch predeclaration this
measure-only task wrote (not committed to ``docs/`` — this task's brief is
measure-only, no repo doc edits). Rotation registry: untouched -- rule 8
(``docs/rotation_registry.md``) makes CFB unconditionally free, and no NFL
confirmation window is spent or declared here.

``docs/offseason_retention.md`` already re-derived three NFL-side estimates
of a *single scalar* retention (0.337-0.400) and ran a scalar-grid CFB screen
(``scripts/offseason_retention_cfb_screen.py``) that found accuracy prefers
0.67 or higher, not lower. That document's own recommendation named "directly
testing per-metric retention values ... inside the full Ridge pipeline" as
the next, out-of-scope experiment. This script is that experiment: does
letting each of the 8 ``CFB_STATE_METRICS`` decay at its own rate (estimated
from CFB's own team-game history) beat the single shipped 0.67 applied to all
of them.

Two arms only, no gate-shopping:

* ``baseline_0_670`` -- the shipped scalar, 0.67 for every metric.
* ``per_metric_vector`` -- one retention per metric, estimated via a CFB
  re-implementation of the NFL "Route 1" methodology
  (``scripts/offseason_retention_routes.py``): team-season means centered
  within season, OLS slope of the next season's first-4-games centered mean
  on the prior season's full-season centered mean, clipped to [0, 1]. This
  estimation step is run once, deterministically, before either arm is
  scored on the benchmark -- it never looks at forced-pick accuracy, Brier,
  or log-loss, only at the metrics' own season-to-season persistence.
  Disclosed caveat: because the estimation and the scoring benchmark share
  the same 2006-2025 CFB corpus, this is a weaker independence guarantee
  than the NFL-routes-vs-CFB-screen structure had (different statistic, same
  corpus, not a held-out split).

Both arms are built and scored through the exact frozen XLG-03 harness
(``nfl_ats.cfb_benchmark``: Ridge alpha 10, no calibration, 500-game training
floor, out-of-time residual distribution, full 2006-2025 window). The
per-metric arm reuses the unmodified production ``build_cfb_team_states`` /
``attach_cfb_team_states`` functions, called once per distinct retention
value in the vector, and splices each metric's ``home_*``/``away_*``/
``diff_*`` columns onto a copy of the production ``build_cfb_game_features``
output at 0.67 -- every non-state column (market, context, experience) is
retention-independent and therefore identical across arms by construction.

Usage::

    .\\.tools\\uv.exe run --no-sync python \\
        scripts/offseason_retention_cfb_permetric_screen.py --output <scratch dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_START_SEASON,
    cfb_walk_forward_benchmark,
)
from nfl_ats.cfb_features import (
    CFB_MODEL_FEATURE_COLUMNS,
    CFB_STATE_METRICS,
    _filtered_schedule,  # private helper; precedented reuse, see cfb_opponent_adjustment_screen.py
    attach_cfb_team_states,
    build_cfb_game_features,
    build_cfb_team_game_metrics,
    build_cfb_team_states,
    load_cfb_benchmark_inputs,
)
from nfl_ats.experiments import paired_feature_comparisons

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFB_ROOT = REPO / "data" / "cfb"

BASELINE_RETENTION = 0.67
BASELINE_METHOD = "baseline_0_670"
PER_METRIC_METHOD = "per_metric_vector"
SPAN = 8
MIN_PERIODS = 3

ROUTE_HORIZON_GAMES = 4
ROUTE_BOOTSTRAP_SAMPLES = 1_000
ROUTE_BOOTSTRAP_SEED = 20260817
ROUTE_MIN_TRANSITIONS = 20

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260817
EVALUATION_WINDOWS = ("clean_core", "thin_2006_2011", "regime_2020", "all")


# ---------------------------------------------------------------------------
# Step 1: estimate a per-metric retention vector from CFB's own history
# (the NFL "Route 1" methodology, re-implemented for CFB team-game metrics).
# ---------------------------------------------------------------------------


def _season_team_means(
    team_games: pd.DataFrame, metric: str, first_n_games: int | None
) -> pd.DataFrame:
    """Mean of ``metric`` per team-season, optionally over only the first N games."""

    frame = team_games[["team_id", "season", "week", "game_id", metric]].dropna(subset=[metric])
    frame = frame.sort_values(["team_id", "season", "week", "game_id"]).copy()
    if first_n_games is not None:
        frame["game_rank"] = frame.groupby(["team_id", "season"]).cumcount() + 1
        frame = frame.loc[frame["game_rank"] <= first_n_games]
    grouped = frame.groupby(["team_id", "season"])[metric].mean().reset_index()
    return grouped.rename(columns={metric: "value"})


def _centered(frame: pd.DataFrame) -> pd.DataFrame:
    league = frame.groupby("season")["value"].mean().rename("league_mean").reset_index()
    merged = frame.merge(league, on="season")
    merged["centered"] = merged["value"] - merged["league_mean"]
    return merged


def _transitions(team_games: pd.DataFrame, metric: str, next_horizon: int | None) -> pd.DataFrame:
    """One row per (team, consecutive season boundary) in the CFB window."""

    prior = _centered(_season_team_means(team_games, metric, first_n_games=None))
    nxt = _centered(_season_team_means(team_games, metric, first_n_games=next_horizon))
    prior = prior[["team_id", "season", "centered"]].rename(
        columns={"season": "prior_season", "centered": "prior_centered"}
    )
    nxt = nxt[["team_id", "season", "centered"]].rename(
        columns={"season": "next_season", "centered": "next_centered"}
    )
    prior = prior.copy()
    prior["next_season"] = prior["prior_season"] + 1
    return prior.merge(nxt, on=["team_id", "next_season"], how="inner")


def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def _season_block_bootstrap(
    merged: pd.DataFrame, samples: int = ROUTE_BOOTSTRAP_SAMPLES, seed: int = ROUTE_BOOTSTRAP_SEED
) -> np.ndarray:
    seasons = merged["prior_season"].unique()
    rng = np.random.default_rng(seed)
    by_season = dict(iter(merged.groupby("prior_season")))
    slopes = np.empty(samples, dtype=float)
    for index in range(samples):
        chosen = rng.choice(seasons, size=len(seasons), replace=True)
        resampled = pd.concat([by_season[season] for season in chosen], ignore_index=True)
        slope, _ = _ols_slope(
            resampled["prior_centered"].to_numpy(), resampled["next_centered"].to_numpy()
        )
        slopes[index] = slope
    return slopes


def estimate_per_metric_retention(
    team_games: pd.DataFrame, horizon: int = ROUTE_HORIZON_GAMES
) -> pd.DataFrame:
    """CFB Route-1 estimate: one season-transition OLS slope per metric, clipped to [0, 1]."""

    rows: list[dict[str, Any]] = []
    for metric in CFB_STATE_METRICS:
        merged = _transitions(team_games, metric, horizon)
        n_transitions = len(merged)
        if n_transitions < ROUTE_MIN_TRANSITIONS:
            raise ValueError(
                f"metric {metric!r} has only {n_transitions} season transitions "
                f"(< {ROUTE_MIN_TRANSITIONS}); cannot estimate a retention value for it"
            )
        slope, intercept = _ols_slope(
            merged["prior_centered"].to_numpy(), merged["next_centered"].to_numpy()
        )
        clipped = float(np.clip(slope, 0.0, 1.0))
        slopes = _season_block_bootstrap(merged)
        lower, upper = np.quantile(slopes, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "horizon_games": horizon,
                "slope": slope,
                "retention": clipped,
                "clipped": bool(clipped != slope),
                "intercept": intercept,
                "n_transitions": n_transitions,
                "n_seasons": int(merged["prior_season"].nunique()),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: build the per-metric feature table by splicing single-metric state
# columns (each built at its own scalar retention via the unmodified
# production functions) onto a copy of the baseline (0.67) feature table.
# ---------------------------------------------------------------------------


def build_state_input(
    schedules: pd.DataFrame, pbp: pd.DataFrame, start_season: int, end_season: int
) -> pd.DataFrame:
    all_schedule, _schedule_audit = _filtered_schedule(schedules, start_season, end_season)
    team_games, _metric_audit = build_cfb_team_game_metrics(pbp)
    state_input = team_games.loc[team_games["game_id"].isin(all_schedule["game_id"])].copy()
    game_dates = all_schedule[["game_id", "gameday"]]
    return state_input.merge(game_dates, on="game_id", how="left", validate="many_to_one")


def build_per_metric_features(
    baseline_games: pd.DataFrame,
    state_input: pd.DataFrame,
    retention_vector: dict[str, float],
    *,
    span: int,
    min_periods: int,
) -> pd.DataFrame:
    result = baseline_games.copy()
    attached_by_retention: dict[float, pd.DataFrame] = {}
    for retention in sorted(set(retention_vector.values())):
        states = build_cfb_team_states(
            state_input, span=span, min_periods=min_periods, offseason_retention=retention
        )
        attached_by_retention[retention] = attach_cfb_team_states(
            baseline_games.copy(), states, offseason_retention=retention
        )
    for metric in CFB_STATE_METRICS:
        source = attached_by_retention[retention_vector[metric]]
        for side in ("home", "away"):
            result[f"{side}_{metric}"] = source[f"{side}_{metric}"]
        result[f"diff_{metric}"] = source[f"diff_{metric}"]
    for column in CFB_MODEL_FEATURE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfb-root", type=Path, default=DEFAULT_CFB_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-season", type=int, default=CFB_BENCHMARK_START_SEASON)
    parser.add_argument("--end-season", type=int, default=CFB_BENCHMARK_END_SEASON)
    parser.add_argument("--min-train-games", type=int, default=CFB_BENCHMARK_MIN_TRAIN_GAMES)
    parser.add_argument("--route-horizon", type=int, default=ROUTE_HORIZON_GAMES)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("loading CFB schedules/lines/pbp ...")
    schedules, lines, pbp = load_cfb_benchmark_inputs(
        args.cfb_root, args.start_season, args.end_season
    )

    print(f"\n=== estimating per-metric retention (CFB Route 1, horizon={args.route_horizon}) ===")
    team_games, team_games_audit = build_cfb_team_game_metrics(pbp)
    print(f"team_games: {len(team_games)} rows ({team_games_audit})")
    route1 = estimate_per_metric_retention(team_games, horizon=args.route_horizon)
    route1.to_csv(args.output / "per_metric_route1_cells.csv", index=False)
    print(route1.to_string(index=False))
    retention_vector = dict(zip(route1["metric"], route1["retention"], strict=True))
    (args.output / "per_metric_vector.json").write_text(
        json.dumps(retention_vector, indent=2), encoding="utf-8"
    )
    print(f"\nfrozen per-metric vector: {retention_vector}")

    print("\n=== building baseline (0.67) CFB features ===")
    baseline_games, baseline_audit = build_cfb_game_features(
        schedules,
        lines,
        pbp,
        start_season=args.start_season,
        end_season=args.end_season,
        span=SPAN,
        min_periods=MIN_PERIODS,
        offseason_retention=BASELINE_RETENTION,
    )
    print(f"canonical games: {baseline_audit['canonical_games']}")

    print("\n=== splicing per-metric CFB features ===")
    state_input = build_state_input(schedules, pbp, args.start_season, args.end_season)
    per_metric_games = build_per_metric_features(
        baseline_games, state_input, retention_vector, span=SPAN, min_periods=MIN_PERIODS
    )

    print("\n=== scoring baseline arm ===")
    baseline_result = cfb_walk_forward_benchmark(
        baseline_games,
        start_season=args.start_season,
        end_season=args.end_season,
        min_train_games=args.min_train_games,
    )
    print("\n=== scoring per-metric arm ===")
    per_metric_result = cfb_walk_forward_benchmark(
        per_metric_games,
        start_season=args.start_season,
        end_season=args.end_season,
        min_train_games=args.min_train_games,
    )

    market_predictions = baseline_result.predictions.loc[
        baseline_result.predictions["method"].eq("market")
    ].copy()
    market_predictions["feature_set"] = "market"

    baseline_candidate = baseline_result.predictions.loc[
        baseline_result.predictions["method"].eq("market_residual")
    ].copy()
    baseline_candidate["method"] = BASELINE_METHOD
    baseline_candidate["feature_set"] = BASELINE_METHOD

    per_metric_candidate = per_metric_result.predictions.loc[
        per_metric_result.predictions["method"].eq("market_residual")
    ].copy()
    per_metric_candidate["method"] = PER_METRIC_METHOD
    per_metric_candidate["feature_set"] = PER_METRIC_METHOD

    predictions = pd.concat(
        [market_predictions, baseline_candidate, per_metric_candidate], ignore_index=True
    )
    predictions.to_parquet(args.output / "predictions.parquet", index=False)

    baseline_summary = baseline_result.summary.copy()
    baseline_summary["arm"] = BASELINE_METHOD
    per_metric_summary = per_metric_result.summary.copy()
    per_metric_summary["arm"] = PER_METRIC_METHOD
    summary_all = pd.concat([baseline_summary, per_metric_summary], ignore_index=True)
    summary_all.to_csv(args.output / "summary_by_arm.csv", index=False)
    print("\n=== raw per-arm summary (descriptive, not a bootstrap endpoint) ===")
    print(
        summary_all.loc[
            :,
            [
                "arm",
                "evaluation_window",
                "method",
                "games",
                "cover_accuracy",
                "cover_brier_score",
                "margin_mae",
            ],
        ]
        .loc[summary_all["method"].eq("market_residual")]
        .to_string(index=False)
    )

    accuracy_rows: list[pd.DataFrame] = []
    for window in EVALUATION_WINDOWS:
        subset = (
            predictions
            if window == "all"
            else predictions.loc[predictions["evaluation_window"].eq(window)]
        )
        candidate_subset = subset.loc[subset["method"].isin((BASELINE_METHOD, PER_METRIC_METHOD))]
        if candidate_subset.empty:
            continue
        for block in ("week", "season"):
            if block == "season" and candidate_subset["season"].nunique() < 2:
                continue
            comparison = paired_feature_comparisons(
                candidate_subset,
                baseline_feature_set=BASELINE_METHOD,
                samples=args.bootstrap_samples,
                block=block,
                seed=args.bootstrap_seed,
            )
            comparison["window"] = window
            comparison["block"] = block
            accuracy_rows.append(comparison)

    accuracy_all = pd.concat(accuracy_rows, ignore_index=True)
    accuracy_all.to_csv(args.output / "paired_accuracy_brier_logloss.csv", index=False)

    print("\n=== headline: clean_core, week-blocked, per_metric_vector vs baseline_0_670 ===")
    headline = accuracy_all.loc[
        accuracy_all["window"].eq("clean_core") & accuracy_all["block"].eq("week")
    ]
    print(
        headline.loc[
            :,
            [
                "metric",
                "estimate",
                "lower",
                "upper",
                "probability_positive",
                "paired_games",
                "blocks",
            ],
        ].to_string(index=False)
    )

    diagnostics: dict[str, Any] = {
        "predeclaration": "scratch predeclaration.md (measure-only task; no docs/ edit this round)",
        "rotation_registry_touched": False,
        "arms": [BASELINE_METHOD, PER_METRIC_METHOD],
        "baseline_retention": BASELINE_RETENTION,
        "per_metric_vector": retention_vector,
        "route_horizon_games": args.route_horizon,
        "route_bootstrap_samples": ROUTE_BOOTSTRAP_SAMPLES,
        "route_bootstrap_seed": ROUTE_BOOTSTRAP_SEED,
        "start_season": args.start_season,
        "end_season": args.end_season,
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "baseline_canonical_games": baseline_audit["canonical_games"],
    }
    (args.output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(f"\nartifacts written to {args.output}")


if __name__ == "__main__":
    main()
