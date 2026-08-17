"""Run the CFB dimension-neutral opponent-adjustment SUBSTITUTION screen.

Predeclaration: ``docs/cfb_opponent_adjustment.md`` (frozen before any arm was
fit). Rotation registry: untouched — rule 8 makes CFB free, and no NFL
confirmation window is spent here.

The screen runs three arms on identical weeks and identical games through the
frozen XLG-03 harness: the market baseline, the frozen 35-column CFB
market-residual model, and the same 35-column model with the six raw per-play
EPA columns replaced in place by their opponent-adjusted equivalents.

Stages, each cached so a rerun is cheap:

1. ``team_games`` — the canonical CFB team-game metric table, built one season
   at a time from local play-by-play.
2. ``adjusted`` — the benchmark feature table plus the six opponent-adjusted
   columns, fit weekly from strictly earlier weeks and earlier dates.
3. the walk-forward screen, its paired margin-error evidence (primary), and
   its paired probability evidence (secondary).

``--leak-probe`` additionally scores a deliberately leaked arm whose
adjustment is fit once on the entire history, including the future. It is a
calibration instrument for the honesty guardrail — "how big does a win look
when the columns really can see the answer?" — and is never part of the
screen's result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb_features import (
    _PBP_LOAD_COLUMNS,
    build_cfb_team_game_metrics,
    load_cfb_seasons,
)
from nfl_ats.cfb_opponent_adjustment import (
    CFB_OPPONENT_ADJUSTED_FEATURE_COLUMNS,
    CFB_OPPONENT_BOOTSTRAP_SAMPLES,
    CFB_OPPONENT_BOOTSTRAP_SEED,
    CFB_OPPONENT_HALF_LIFE_WEEKS,
    CFB_OPPONENT_MIN_TEAM_GAMES,
    CFB_OPPONENT_RIDGE_ALPHA,
    CFB_OPPONENT_SOURCE_METRIC,
    CFB_TIME_DECAYED_MODEL_FEATURE_COLUMNS,
    OPPONENT_BENCHMARK_BASELINE_METHOD,
    OPPONENT_BENCHMARK_CANDIDATE_METHOD,
    OPPONENT_BENCHMARK_CONTROL_METHOD,
    add_cfb_opponent_adjusted_features,
    build_cfb_opponent_history,
    cfb_opponent_adjustment_benchmark,
    paired_margin_error_comparison,
)
from nfl_ats.opponent_adjustment import fit_opponent_effects

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO / "data/processed/cfb_game_features.parquet"
DEFAULT_CFB_ROOT = REPO / "data/cfb"


def build_team_games(cfb_root: Path, seasons: list[int]) -> pd.DataFrame:
    """Build the CFB team-game metric table one season at a time."""

    frames: list[pd.DataFrame] = []
    for season in seasons:
        pbp = load_cfb_seasons(cfb_root, "pbp", [season], columns=list(_PBP_LOAD_COLUMNS))
        metrics, audit = build_cfb_team_game_metrics(pbp)
        print(
            f"  {season}: {audit['team_games']:>6} team-games "
            f"({audit['unpaired_games_dropped']} unpaired games dropped)"
        )
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def leaked_adjusted_features(games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """DIAGNOSTIC ONLY: adjusted columns fit once on the entire history.

    This arm can see every game it scores, including the future. It exists to
    measure what a leak looks like on this benchmark, so the real screen's
    effect size can be read against a known-contaminated reference instead of
    against intuition.
    """

    result = games.copy()
    result["gameday"] = pd.to_datetime(result["gameday"], errors="raise")
    history = build_cfb_opponent_history(team_games, result)
    teams = tuple(sorted(set(history["team"]) | set(history["opponent"])))
    effects = fit_opponent_effects(
        history,
        metric=CFB_OPPONENT_SOURCE_METRIC,
        teams=teams,
        cutoff=history["gameday"].max() + pd.Timedelta(days=1),
        half_life_weeks=CFB_OPPONENT_HALF_LIFE_WEEKS,
        ridge_alpha=CFB_OPPONENT_RIDGE_ALPHA,
        min_team_games=CFB_OPPONENT_MIN_TEAM_GAMES,
    )
    if effects is None:
        raise ValueError("The leak probe could not fit the full history")
    home = result["home_id"].astype("int64").astype(str)
    away = result["away_id"].astype("int64").astype(str)
    offense = {team: effects.offense_rating(team) for team in teams}
    defense = {team: effects.defense_rating(team) for team in teams}
    for metric, ratings in (("off_epa_per_play", offense), ("def_epa_per_play", defense)):
        home_values = home.map(ratings).astype(float)
        away_values = away.map(ratings).astype(float)
        result[f"home_{metric}_adjusted"] = home_values
        result[f"away_{metric}_adjusted"] = away_values
        result[f"diff_{metric}_adjusted"] = home_values - away_values
    return result


def split_margin_evidence(predictions: pd.DataFrame, *, samples: int, seed: int) -> pd.DataFrame:
    """Paired margin-error evidence on every reported evaluation split."""

    rows: list[pd.DataFrame] = []
    for window in ("clean_core", "thin_2006_2011", "regime_2020", "all"):
        subset = (
            predictions
            if window == "all"
            else predictions.loc[predictions["evaluation_window"].eq(window)]
        )
        if subset.empty:
            continue
        for block in ("week", "season"):
            if block == "season" and subset["season"].nunique() < 2:
                continue
            rows.append(
                paired_margin_error_comparison(
                    subset,
                    baseline_method=OPPONENT_BENCHMARK_BASELINE_METHOD,
                    candidate_method=OPPONENT_BENCHMARK_CANDIDATE_METHOD,
                    samples=samples,
                    block=block,
                    seed=seed,
                ).assign(evaluation_window=window)
            )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--cfb-root", type=Path, default=DEFAULT_CFB_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-season", type=int, default=2006)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--min-train-games", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=CFB_OPPONENT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=CFB_OPPONENT_BOOTSTRAP_SEED)
    parser.add_argument("--leak-probe", action="store_true")
    parser.add_argument("--control-arm", action="store_true")
    arguments = parser.parse_args()

    output: Path = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(arguments.features)
    seasons = sorted(
        int(season)
        for season in features["season"].unique()
        if arguments.start_season <= int(season) <= arguments.end_season
    )
    print(f"features: {len(features)} games, seasons {seasons[0]}-{seasons[-1]}")

    team_games_path = output / "cfb_team_games.parquet"
    if team_games_path.exists():
        team_games = pd.read_parquet(team_games_path)
        print(f"team games: reused cache ({len(team_games)} rows)")
    else:
        print("team games: building from play-by-play")
        team_games = build_team_games(arguments.cfb_root, seasons)
        team_games.to_parquet(team_games_path, index=False)
    print(f"team games: {len(team_games)} rows, {team_games['team_id'].nunique()} teams")

    adjusted_path = output / "cfb_features_opponent_adjusted.parquet"
    if adjusted_path.exists():
        adjusted = pd.read_parquet(adjusted_path)
        print("adjusted features: reused cache")
    else:
        print("adjusted features: fitting weekly opponent adjustments")
        adjusted = add_cfb_opponent_adjusted_features(features, team_games)
        adjusted.to_parquet(adjusted_path, index=False)
    coverage = {
        column: float(adjusted[column].notna().mean())
        for column in CFB_OPPONENT_ADJUSTED_FEATURE_COLUMNS
    }
    print(f"adjusted coverage: {json.dumps(coverage, indent=None)}")

    print("screen: walking forward three arms on identical weeks")
    result = cfb_opponent_adjustment_benchmark(
        adjusted,
        start_season=arguments.start_season,
        end_season=arguments.end_season,
        min_train_games=arguments.min_train_games,
        bootstrap_samples=arguments.bootstrap_samples,
        bootstrap_seed=arguments.bootstrap_seed,
    )
    result.predictions.to_parquet(output / "predictions.parquet", index=False)
    result.summary.to_csv(output / "summary.csv", index=False)
    result.season_summary.to_csv(output / "season_summary.csv", index=False)
    result.paired_probability.to_csv(output / "paired_probability.csv", index=False)

    margin = split_margin_evidence(
        result.predictions, samples=arguments.bootstrap_samples, seed=arguments.bootstrap_seed
    )
    margin.to_csv(output / "paired_margin.csv", index=False)

    diagnostics: dict[str, Any] = {
        "predeclaration": "docs/cfb_opponent_adjustment.md",
        "hypothesis_frozen_before_scoring": True,
        "rotation_registry_touched": False,
        "half_life_weeks": CFB_OPPONENT_HALF_LIFE_WEEKS,
        "ridge_alpha": CFB_OPPONENT_RIDGE_ALPHA,
        "min_team_games": CFB_OPPONENT_MIN_TEAM_GAMES,
        "min_train_games": arguments.min_train_games,
        "bootstrap_samples": arguments.bootstrap_samples,
        "bootstrap_seed": arguments.bootstrap_seed,
        "adjusted_column_coverage": coverage,
        **result.diagnostics,
    }

    if arguments.control_arm:
        print("control arm: time decay without the opponent block (POST HOC)")
        control_path = output / "cfb_features_time_decayed.parquet"
        if control_path.exists():
            control_features = pd.read_parquet(control_path)
        else:
            control_features = add_cfb_opponent_adjusted_features(
                features, team_games, include_opponent=False
            )
            control_features.to_parquet(control_path, index=False)
        control_result = cfb_opponent_adjustment_benchmark(
            control_features,
            start_season=arguments.start_season,
            end_season=arguments.end_season,
            min_train_games=arguments.min_train_games,
            bootstrap_samples=arguments.bootstrap_samples,
            bootstrap_seed=arguments.bootstrap_seed,
            candidate_columns=CFB_TIME_DECAYED_MODEL_FEATURE_COLUMNS,
            candidate_method=OPPONENT_BENCHMARK_CONTROL_METHOD,
        )
        # The two runs share a baseline arm by construction; assert it rather
        # than assume it, then stitch so the control and the frozen candidate
        # can be compared to each other on the same games.
        for method in ("market", OPPONENT_BENCHMARK_BASELINE_METHOD):
            left = result.predictions.loc[result.predictions["method"].eq(method)]
            right = control_result.predictions.loc[control_result.predictions["method"].eq(method)]
            pd.testing.assert_series_equal(
                left.sort_values("game_id")["predicted_margin"].reset_index(drop=True),
                right.sort_values("game_id")["predicted_margin"].reset_index(drop=True),
            )
        stitched = pd.concat(
            [
                result.predictions,
                control_result.predictions.loc[
                    control_result.predictions["method"].eq(OPPONENT_BENCHMARK_CONTROL_METHOD)
                ],
            ],
            ignore_index=True,
        )
        decomposition = pd.concat(
            [
                paired_margin_error_comparison(
                    stitched.loc[stitched["evaluation_window"].eq("clean_core")],
                    baseline_method=baseline,
                    candidate_method=candidate,
                    samples=arguments.bootstrap_samples,
                    block="week",
                    seed=arguments.bootstrap_seed,
                ).assign(comparison=label)
                for baseline, candidate, label in (
                    (
                        OPPONENT_BENCHMARK_BASELINE_METHOD,
                        OPPONENT_BENCHMARK_CONTROL_METHOD,
                        "time_weighting_only",
                    ),
                    (
                        OPPONENT_BENCHMARK_CONTROL_METHOD,
                        OPPONENT_BENCHMARK_CANDIDATE_METHOD,
                        "opponent_block_only",
                    ),
                    (
                        OPPONENT_BENCHMARK_BASELINE_METHOD,
                        OPPONENT_BENCHMARK_CANDIDATE_METHOD,
                        "both_together",
                    ),
                )
            ],
            ignore_index=True,
        )
        decomposition.to_csv(output / "control_decomposition.csv", index=False)
        print("\n=== post-hoc decomposition: clean core, week-blocked ===")
        print(
            decomposition.loc[
                :, ["comparison", "metric", "estimate", "lower", "upper", "probability_positive"]
            ].to_string(index=False)
        )
        diagnostics["control_decomposition"] = decomposition.to_dict(orient="records")

    if arguments.leak_probe:
        print("leak probe: scoring a deliberately contaminated arm")
        leaked = leaked_adjusted_features(features, team_games)
        leaked_result = cfb_opponent_adjustment_benchmark(
            leaked,
            start_season=arguments.start_season,
            end_season=arguments.end_season,
            min_train_games=arguments.min_train_games,
            bootstrap_samples=arguments.bootstrap_samples,
            bootstrap_seed=arguments.bootstrap_seed,
        )
        leaked_margin = split_margin_evidence(
            leaked_result.predictions,
            samples=arguments.bootstrap_samples,
            seed=arguments.bootstrap_seed,
        )
        leaked_margin.to_csv(output / "leak_probe_paired_margin.csv", index=False)
        leaked_result.summary.to_csv(output / "leak_probe_summary.csv", index=False)
        clean = leaked_margin.loc[
            leaked_margin["evaluation_window"].eq("clean_core") & leaked_margin["block"].eq("week")
        ]
        diagnostics["leak_probe_clean_core_week"] = clean.to_dict(orient="records")

    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )

    print("\n=== headline: clean core, week-blocked, paired ===")
    headline = margin.loc[margin["evaluation_window"].eq("clean_core") & margin["block"].eq("week")]
    for row in headline.itertuples(index=False):
        print(
            f"{row.metric:>24}: {row.estimate:+.4f} points "
            f"[{row.lower:+.4f}, {row.upper:+.4f}] "
            f"P(candidate better)={row.probability_positive:.3f} on {row.games} games"
        )
    print("\n=== per-split margin error by arm ===")
    columns = ["evaluation_window", "method", "games", "margin_mae", "margin_rmse"]
    available = [column for column in columns if column in result.summary.columns]
    print(result.summary.loc[:, available].to_string(index=False))
    print("\n=== paired probability evidence (clean core) ===")
    print(
        result.paired_probability.loc[
            :, ["metric", "block", "estimate", "lower", "upper", "probability_positive"]
        ].to_string(index=False)
    )
    print(f"\nartifacts: {output}")
    print(
        "design width: "
        f"baseline {diagnostics['baseline_effective_design_width']} effective "
        f"vs candidate {diagnostics['candidate_effective_design_width']} "
        f"(inputs {diagnostics['baseline_input_columns']} vs "
        f"{diagnostics['candidate_input_columns']})"
    )
    if np.isfinite(list(diagnostics["adjusted_column_correlation"].values())).all():
        print(f"raw/adjusted correlations: {diagnostics['adjusted_column_correlation']}")


if __name__ == "__main__":
    main()
