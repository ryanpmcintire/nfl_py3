from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.pbp import latest_pbp_snapshot
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact
from nfl_ats.public_board import find_matching_opener_evaluation

SAMPLES = 20_000
SEED = 20260910
OUTPUT = Path("artifacts/experiments/altitude_fourth_quarter")
SEASON_START = 2009
SEASON_END = 2025
SCREEN_SEASON_START = 2020
SCREEN_SEASON_END = 2025


def denver_home_flag(games: pd.DataFrame) -> pd.Series:
    if games["home_team"].isna().any():
        raise ValueError("Missing home team")
    return games["home_team"].eq("DEN").rename("denver_home")


def mexico_city_flag(games: pd.DataFrame) -> pd.Series:
    return games["stadium_id"].eq("MEX00").rename("mexico_city")


def _summarize(values: np.ndarray, point: pd.Series) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "estimate": float(point.mean()) if point.count() else None,
            "interval95": [None, None],
            "probability_positive": None,
            "standard_error": None,
            "games": int(point.count()),
        }
    return {
        "estimate": float(point.mean()),
        "interval95": np.quantile(finite, [0.025, 0.975]).tolist(),
        "probability_positive": float((finite > 0).mean()),
        "standard_error": float(finite.std(ddof=1)) if len(finite) > 1 else None,
        "games": int(point.count()),
    }


def blocked_bootstrap(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    block_columns: tuple[str, str] = ("season", "week"),
    seed: int = SEED,
    samples: int = SAMPLES,
    contrasts: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    grouped = frame.groupby(list(block_columns))[columns]
    sums = grouped.sum().to_numpy(float)
    counts = grouped.count().to_numpy(float)
    n_blocks = len(sums)
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(columns)))
    for start in range(0, samples, 250):
        batch = min(250, samples - start)
        indices = rng.integers(0, n_blocks, size=(batch, n_blocks))
        denom = counts[indices].sum(axis=1)
        draws[start : start + batch] = np.divide(
            sums[indices].sum(axis=1),
            denom,
            out=np.full_like(denom, np.nan),
            where=denom > 0,
        )
    result: dict[str, Any] = {"blocks": n_blocks}
    for i, column in enumerate(columns):
        result[column] = _summarize(draws[:, i], frame[column])
    if contrasts:
        for candidate, baseline, label in contrasts:
            i, j = columns.index(candidate), columns.index(baseline)
            delta = draws[:, i] - draws[:, j]
            delta = delta[np.isfinite(delta)]
            result[label] = {
                "estimate": float(frame[candidate].mean() - frame[baseline].mean()),
                "interval95": np.quantile(delta, [0.025, 0.975]).tolist()
                if len(delta)
                else [None, None],
                "probability_positive": float((delta > 0).mean()) if len(delta) else None,
                "standard_error": float(delta.std(ddof=1)) if len(delta) > 1 else None,
            }
    return result


def quarter_table(pbp: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    plays = pbp.loc[pbp["season_type"].eq("REG")].copy()
    plays = plays.sort_values(["game_id", "play_id"])
    oriented = plays.loc[
        plays.posteam.eq(plays.home_team) | plays.posteam.eq(plays.away_team)
    ].copy()

    scoring = oriented.loc[oriented["score_differential"].notna()].copy()
    scoring["home_margin"] = np.where(
        scoring.posteam.eq(scoring.home_team),
        scoring.score_differential,
        -scoring.score_differential,
    )
    q4_start = scoring.loc[scoring.qtr.eq(4)].drop_duplicates("game_id")
    q4_start = q4_start.loc[
        q4_start.game_seconds_remaining.eq(900), ["game_id", "home_margin"]
    ].rename(columns={"home_margin": "first_three_margin"})
    overtime = scoring.loc[scoring.qtr.ge(5)].drop_duplicates("game_id")
    overtime = overtime[["game_id", "home_margin"]].rename(
        columns={"home_margin": "regulation_margin"}
    )
    ot_ids = set(scoring.loc[scoring.qtr.ge(5), "game_id"])

    epa_plays = oriented.loc[oriented["epa"].notna() & oriented.qtr.between(1, 4)].copy()
    epa_plays["home_net_epa"] = np.where(
        epa_plays.posteam.eq(epa_plays.home_team), epa_plays.epa, -epa_plays.epa
    )
    epa_plays["segment"] = np.where(epa_plays.qtr.eq(4), "fourth_epa", "first_three_epa")
    epa_wide = (
        epa_plays.groupby(["game_id", "segment"])["home_net_epa"]
        .sum()
        .unstack(fill_value=np.nan)
        .reset_index()
    )
    for column in ("first_three_epa", "fourth_epa"):
        if column not in epa_wide:
            epa_wide[column] = np.nan

    scope = games.loc[games.season.between(SEASON_START, SEASON_END) & games.game_type.eq("REG")]
    merged = scope.merge(q4_start, on="game_id", how="left")
    excluded = merged.loc[merged.first_three_margin.isna()].copy()
    result = merged.loc[merged.first_three_margin.notna()].copy()
    result = result.merge(overtime, on="game_id", how="left")
    result = result.loc[~result.game_id.isin(ot_ids) | result.regulation_margin.notna()].copy()
    result["regulation_margin"] = result.regulation_margin.fillna(result["result"])
    result["fourth_margin"] = result.regulation_margin - result.first_three_margin
    result["late_minus_early_rate"] = result.fourth_margin - result.first_three_margin / 3
    result["margin_q4_share"] = np.where(
        result.regulation_margin.ne(0),
        result.fourth_margin / result.regulation_margin,
        np.nan,
    )

    result = result.merge(epa_wide, on="game_id", how="left")
    result["epa_late_minus_early_rate"] = result.fourth_epa - result.first_three_epa / 3
    total_epa = result.first_three_epa + result.fourth_epa
    result["epa_q4_share"] = np.where(total_epa.ne(0), result.fourth_epa / total_epa, np.nan)

    return result.sort_values(["season", "week", "game_id"]).reset_index(drop=True), excluded


def next_game_index(games: pd.DataFrame) -> pd.DataFrame:
    reg = games.loc[games.game_type.eq("REG")].copy()
    home = reg[["season", "week", "game_id", "home_team"]].rename(columns={"home_team": "team"})
    home["is_home"] = True
    away = reg[["season", "week", "game_id", "away_team"]].rename(columns={"away_team": "team"})
    away["is_home"] = False
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "season", "week"])
    long["next_game_id"] = long.groupby(["team", "season"])["game_id"].shift(-1)
    long["next_is_home"] = long.groupby(["team", "season"])["is_home"].shift(-1)
    long["next_week"] = long.groupby(["team", "season"])["week"].shift(-1)
    return long


def visitor_next_game_effect(
    denver_games: pd.DataFrame,
    quarters: pd.DataFrame,
    games: pd.DataFrame,
    league_home_rate: float,
    league_home_epa_rate: float,
) -> pd.DataFrame:
    idx = next_game_index(games)
    rows = []
    quarters_by_id = quarters.set_index("game_id")
    idx_by_team_game = idx.set_index(["team", "game_id"])
    for _, den in denver_games.iterrows():
        team = den["away_team"]
        key = (team, den["game_id"])
        if key not in idx_by_team_game.index:
            continue
        entry = idx_by_team_game.loc[key]
        if isinstance(entry, pd.DataFrame):
            entry = entry.iloc[0]
        next_id = entry["next_game_id"]
        if pd.isna(next_id):
            continue
        if next_id not in quarters_by_id.index:
            continue
        next_row = quarters_by_id.loc[next_id]
        next_is_home = bool(next_row["home_team"] == team)
        sign = 1.0 if next_is_home else -1.0
        team_rate = sign * next_row["late_minus_early_rate"]
        team_epa_rate = sign * next_row["epa_late_minus_early_rate"]
        expected_rate = league_home_rate if next_is_home else -league_home_rate
        expected_epa_rate = league_home_epa_rate if next_is_home else -league_home_epa_rate
        rows.append(
            {
                "season": den["season"],
                "week": den["week"],
                "denver_game_id": den["game_id"],
                "visitor": team,
                "next_game_id": next_id,
                "next_is_home": next_is_home,
                "team_rate": team_rate,
                "team_epa_rate": team_epa_rate,
                "expected_rate": expected_rate,
                "expected_epa_rate": expected_epa_rate,
                "excess_rate": team_rate - expected_rate,
                "excess_epa_rate": team_epa_rate - expected_epa_rate,
            }
        )
    return pd.DataFrame(rows)


def paired_overlay(
    predictions: pd.DataFrame,
    games: pd.DataFrame,
    flag_fn: Callable[[pd.DataFrame], pd.Series],
    flag_name: str,
) -> pd.DataFrame:
    result = predictions.merge(
        games[["game_id", "home_team", "stadium_id"]], on="game_id", validate="one_to_one"
    ).sort_values(["season", "week", "game_id"])
    if len(result) != len(predictions):
        raise ValueError("Schedule does not cover every prediction")
    result = result.loc[result.margin_vs_open.notna() & result.margin_vs_open.ne(0)].copy()
    result[flag_name] = flag_fn(result)
    base_pick = result.pick_home_at_open_probability_rule.astype(bool)
    covered = result.margin_vs_open.gt(0)
    result["candidate_pick_home"] = base_pick | result[flag_name]
    result["oracle_pick_home"] = base_pick | (result[flag_name] & covered)
    result["baseline"] = base_pick.eq(covered).astype(float) * 100
    result["candidate"] = result.candidate_pick_home.eq(covered).astype(float) * 100
    result["oracle"] = result.oracle_pick_home.eq(covered).astype(float) * 100
    result["delta"] = result.candidate - result.baseline
    result["oracle_delta"] = result.oracle - result.baseline
    return result


def ols_slope_intercept(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = x.notna() & y.notna()
    xv = x[valid].to_numpy(float)
    yv = y[valid].to_numpy(float)
    b = float(np.cov(xv, yv, ddof=1)[0, 1] / np.var(xv, ddof=1))
    a = float(yv.mean() - b * xv.mean())
    return a, b


def season_pair_reliability(
    denver: pd.DataFrame, value_column: str, seed: int = SEED, samples: int = SAMPLES
) -> dict[str, Any]:
    season_means = denver.groupby("season")[value_column].mean()
    seasons = sorted(season_means.index)
    pairs = []
    for odd_season in seasons:
        if odd_season % 2 == 1 and (odd_season + 1) in season_means.index:
            pairs.append((odd_season, odd_season + 1))
    odd_vals = np.array([season_means[a] for a, _ in pairs])
    even_vals = np.array([season_means[b] for _, b in pairs])
    rng = np.random.default_rng(seed)
    n = len(pairs)
    correlations = []
    for _ in range(samples):
        pick = rng.integers(0, n, n)
        o, e = odd_vals[pick], even_vals[pick]
        if o.std() > 0 and e.std() > 0:
            correlations.append(float(np.corrcoef(o, e)[0, 1]))
    point = float(np.corrcoef(odd_vals, even_vals)[0, 1]) if n > 1 else None
    return {
        "pairs": pairs,
        "n_pairs": n,
        "estimate": point,
        "interval95": np.quantile(correlations, [0.025, 0.975]).tolist()
        if correlations
        else [None, None],
        "probability_positive": float((np.array(correlations) > 0).mean())
        if correlations
        else None,
        "odd_season_values": odd_vals.tolist(),
        "even_season_values": even_vals.tolist(),
    }


def alternating_game_reliability(
    quarters: pd.DataFrame, value_column: str, seed: int = SEED, samples: int = SAMPLES
) -> dict[str, Any]:
    denver = quarters.loc[denver_home_flag(quarters)].copy()
    denver["half"] = denver.groupby("season").cumcount() % 2
    pivot = denver.pivot_table(index="season", columns="half", values=value_column)
    values = pivot.dropna().to_numpy(float)
    rng = np.random.default_rng(seed)
    correlations = []
    for _ in range(samples):
        sample = values[rng.integers(0, len(values), len(values))]
        if np.all(sample.std(axis=0) > 0):
            correlations.append(float(np.corrcoef(sample.T)[0, 1]))
    return {
        "estimate": float(np.corrcoef(values.T)[0, 1]),
        "interval95": np.quantile(correlations, [0.025, 0.975]).tolist(),
        "probability_positive": float((np.array(correlations) > 0).mean()),
        "seasons": len(values),
    }


def main() -> None:
    schedules_path = sorted(Path("data/raw").glob("*/schedules.parquet"))[-1]
    games = pd.read_parquet(schedules_path)
    snapshot = latest_pbp_snapshot(Path("data/pbp/raw"))
    columns = [
        "game_id",
        "play_id",
        "season_type",
        "qtr",
        "posteam",
        "home_team",
        "away_team",
        "score_differential",
        "game_seconds_remaining",
        "epa",
    ]
    panels = [
        pd.read_parquet(snapshot.season_path(s), columns=columns)
        for s in range(SEASON_START, SEASON_END + 1)
    ]
    pbp = pd.concat(panels, ignore_index=True)
    quarters, excluded = quarter_table(pbp, games)

    quarters["denver_fourth"] = quarters.fourth_margin.where(denver_home_flag(quarters))
    quarters["denver_first_three"] = quarters.first_three_margin.where(denver_home_flag(quarters))
    quarters["denver_late_rate"] = quarters.late_minus_early_rate.where(denver_home_flag(quarters))
    quarters["denver_q4_share"] = quarters.margin_q4_share.where(denver_home_flag(quarters))
    quarters["denver_fourth_epa"] = quarters.fourth_epa.where(denver_home_flag(quarters))
    quarters["denver_first_three_epa"] = quarters.first_three_epa.where(denver_home_flag(quarters))
    quarters["denver_epa_late_rate"] = quarters.epa_late_minus_early_rate.where(
        denver_home_flag(quarters)
    )
    quarters["denver_epa_q4_share"] = quarters.epa_q4_share.where(denver_home_flag(quarters))

    descriptive_scoring = blocked_bootstrap(
        quarters,
        [
            "fourth_margin",
            "denver_fourth",
            "first_three_margin",
            "denver_first_three",
            "late_minus_early_rate",
            "denver_late_rate",
            "margin_q4_share",
            "denver_q4_share",
        ],
        contrasts=[
            ("denver_fourth", "fourth_margin", "denver_minus_league_fourth"),
            ("denver_late_rate", "late_minus_early_rate", "denver_minus_league_late_rate"),
            ("denver_q4_share", "margin_q4_share", "denver_minus_league_q4_share"),
        ],
    )
    descriptive_epa = blocked_bootstrap(
        quarters,
        [
            "fourth_epa",
            "denver_fourth_epa",
            "first_three_epa",
            "denver_first_three_epa",
            "epa_late_minus_early_rate",
            "denver_epa_late_rate",
            "epa_q4_share",
            "denver_epa_q4_share",
        ],
        contrasts=[
            ("denver_fourth_epa", "fourth_epa", "denver_minus_league_fourth_epa"),
            (
                "denver_epa_late_rate",
                "epa_late_minus_early_rate",
                "denver_minus_league_epa_late_rate",
            ),
            ("denver_epa_q4_share", "epa_q4_share", "denver_minus_league_epa_q4_share"),
        ],
    )

    denver_games = games.loc[
        denver_home_flag(games)
        & games.season.between(SEASON_START, SEASON_END)
        & games.game_type.eq("REG")
    ]
    league_home_rate = float(quarters.late_minus_early_rate.mean())
    league_home_epa_rate = float(quarters.epa_late_minus_early_rate.mean())
    visitor_effect = visitor_next_game_effect(
        denver_games, quarters, games, league_home_rate, league_home_epa_rate
    )
    visitor_effect_missing = int(len(denver_games) - len(visitor_effect))
    visitor_bootstrap = (
        blocked_bootstrap(visitor_effect, ["excess_rate", "excess_epa_rate"])
        if len(visitor_effect)
        else {}
    )

    league_spread_close = quarters.copy()
    non_denver_close = league_spread_close.loc[~denver_home_flag(league_spread_close)]
    a_close, b_close = ols_slope_intercept(
        non_denver_close["spread_line"], non_denver_close["fourth_margin"]
    )
    league_spread_close["predicted_fourth_close"] = (
        a_close + b_close * league_spread_close["spread_line"]
    )
    league_spread_close["residual_fourth_close"] = (
        league_spread_close["fourth_margin"] - league_spread_close["predicted_fourth_close"]
    )
    league_spread_close["denver_residual_fourth_close"] = league_spread_close[
        "residual_fourth_close"
    ].where(denver_home_flag(league_spread_close))
    line_conditioned_close = blocked_bootstrap(
        league_spread_close,
        ["residual_fourth_close", "denver_residual_fourth_close"],
        contrasts=[
            (
                "denver_residual_fourth_close",
                "residual_fourth_close",
                "denver_minus_league_fourth_line_conditioned",
            )
        ],
    )
    close_slope = {
        "intercept": a_close,
        "slope": b_close,
        "conditioned_on": "schedule spread_line (closing-line proxy, full 2009-2025 range)",
    }

    match = find_matching_opener_evaluation(Path("artifacts"))
    if match is None:
        raise ValueError("No active-model-matching opener evaluation")
    metadata, source = match
    predictions_full = pd.read_parquet(source / "per_game.parquet")
    predictions = predictions_full.loc[
        predictions_full.season.between(SCREEN_SEASON_START, SCREEN_SEASON_END)
    ]

    opener_quarters = quarters.merge(
        predictions[["game_id", "tue_open_home_spread"]], on="game_id", how="inner"
    )
    non_denver_open = opener_quarters.loc[~denver_home_flag(opener_quarters)]
    a_open, b_open = ols_slope_intercept(
        non_denver_open["tue_open_home_spread"], non_denver_open["fourth_margin"]
    )
    opener_quarters["predicted_fourth_open"] = (
        a_open + b_open * opener_quarters["tue_open_home_spread"]
    )
    opener_quarters["residual_fourth_open"] = (
        opener_quarters["fourth_margin"] - opener_quarters["predicted_fourth_open"]
    )
    opener_quarters["denver_residual_fourth_open"] = opener_quarters["residual_fourth_open"].where(
        denver_home_flag(opener_quarters)
    )
    line_conditioned_opener = blocked_bootstrap(
        opener_quarters,
        ["residual_fourth_open", "denver_residual_fourth_open"],
        contrasts=[
            (
                "denver_residual_fourth_open",
                "residual_fourth_open",
                "denver_minus_league_fourth_line_conditioned_opener",
            )
        ],
    )
    open_slope = {
        "intercept": a_open,
        "slope": b_open,
        "conditioned_on": "true Tuesday opener tue_open_home_spread, 2020-2025 subset",
    }

    denver_paired = paired_overlay(predictions, games, denver_home_flag, "denver_home")
    denver_only = denver_paired.loc[denver_paired.denver_home]
    league_cover_rate = float(denver_paired.margin_vs_open.gt(0).mean())
    denver_cover_rate = float(denver_only.margin_vs_open.gt(0).mean())
    denver_paired["home_covered"] = denver_paired.margin_vs_open.gt(0).astype(float) * 100
    denver_paired["denver_home_covered"] = denver_paired.home_covered.where(
        denver_paired.denver_home
    )
    cover_rate_contrast = blocked_bootstrap(
        denver_paired,
        ["home_covered", "denver_home_covered"],
        contrasts=[("denver_home_covered", "home_covered", "denver_minus_league_cover_rate")],
    )

    denver_metrics = blocked_bootstrap(
        denver_paired, ["baseline", "candidate", "oracle", "delta", "oracle_delta"]
    )
    denver_changed = int(
        denver_paired.candidate_pick_home.ne(denver_paired.pick_home_at_open_probability_rule).sum()
    )

    mexico_paired = paired_overlay(predictions, games, mexico_city_flag, "mexico_city")
    mexico_only = mexico_paired.loc[mexico_paired.mexico_city]
    mexico_metrics = (
        blocked_bootstrap(
            mexico_paired, ["baseline", "candidate", "oracle", "delta", "oracle_delta"]
        )
        if mexico_only["mexico_city"].any()
        else {}
    )
    mexico_changed = int(
        mexico_paired.candidate_pick_home.ne(mexico_paired.pick_home_at_open_probability_rule).sum()
    )
    mexico_all_games = games.loc[mexico_city_flag(games) & games.game_type.eq("REG")]

    season_pairs_scoring = season_pair_reliability(
        quarters.loc[denver_home_flag(quarters)], "late_minus_early_rate"
    )
    season_pairs_epa = season_pair_reliability(
        quarters.loc[denver_home_flag(quarters)], "epa_late_minus_early_rate"
    )
    alternating_scoring = alternating_game_reliability(quarters, "late_minus_early_rate")

    config = {
        "samples": SAMPLES,
        "seed": SEED,
        "predeclaration": "ROADMAP LEAD-43, extending docs/coaching_leads.md's CX15 predeclaration",
        "opener_source": str(source),
        "pbp_source": str(snapshot.root),
        "schedules_source": str(schedules_path),
        "active_model_id": metadata["active_model_id"],
        "feature_table_sha256": metadata["feature_table_sha256"],
        "grade": "opener_probability_rule",
        "screen_seasons": [SCREEN_SEASON_START, SCREEN_SEASON_END],
        "descriptive_seasons": [SEASON_START, SEASON_END],
    }
    payload = {
        "configuration": config,
        "descriptive_scoring": descriptive_scoring,
        "descriptive_epa": descriptive_epa,
        "excluded_games": excluded.game_id.tolist(),
        "visitor_next_game": {
            "n_denver_games": len(denver_games),
            "n_matched": len(visitor_effect),
            "n_missing_next_game": visitor_effect_missing,
            "league_home_late_rate": league_home_rate,
            "league_home_epa_late_rate": league_home_epa_rate,
            "bootstrap": visitor_bootstrap,
        },
        "market_pricing": {
            "denver_cover_rate_opener": denver_cover_rate,
            "league_cover_rate_opener": league_cover_rate,
            "cover_rate_contrast": cover_rate_contrast,
            "line_conditioned_close_full_range": {
                "regression": close_slope,
                "bootstrap": line_conditioned_close,
            },
            "line_conditioned_opener_2020_2025": {
                "regression": open_slope,
                "bootstrap": line_conditioned_opener,
            },
        },
        "production_screen_denver": {
            "paired_games": len(denver_paired),
            "paired_weeks": denver_paired.groupby(["season", "week"]).ngroups,
            "denver_games": int(denver_paired.denver_home.sum()),
            "changed_picks": denver_changed,
            "metrics": denver_metrics,
        },
        "production_screen_mexico_city": {
            "paired_games": len(mexico_paired),
            "mexico_city_games_in_screen_window": int(mexico_only["mexico_city"].sum()),
            "mexico_city_games_all_time": len(mexico_all_games),
            "mexico_city_game_ids_all_time": mexico_all_games.game_id.tolist(),
            "changed_picks": mexico_changed,
            "metrics": mexico_metrics,
        },
        "split_half_reliability": {
            "season_pairs_scoring": season_pairs_scoring,
            "season_pairs_epa": season_pairs_epa,
            "alternating_games_scoring_2009_2025": alternating_scoring,
        },
        "provenance": artifact_provenance(
            config, Path("data/processed/game_features_weak_stack.parquet")
        ),
    }
    write_experiment_artifact(
        OUTPUT,
        "results.json",
        payload,
        command="altitude-fourth-quarter-screen",
        metrics=denver_metrics.get("delta", {}),
        registry_root=OUTPUT / "experiment_registry",
    )
    quarters.to_csv(OUTPUT / "quarter_table.csv", index=False)
    visitor_effect.to_csv(OUTPUT / "visitor_next_game.csv", index=False)
    denver_paired.to_csv(OUTPUT / "denver_paired_predictions.csv", index=False)
    mexico_paired.loc[mexico_paired.mexico_city].to_csv(
        OUTPUT / "mexico_city_games.csv", index=False
    )
    print(
        json.dumps({k: v for k, v in payload.items() if k != "provenance"}, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
