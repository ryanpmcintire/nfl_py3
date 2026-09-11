from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import (  # noqa: E402
    HISTORICAL_CAPTURE_KIND,
    cached_pairing_table,
    opener_pick_evaluation,
    pick_correct,
    resolve_active_model_config,
    week_blocked_bootstrap,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES  # noqa: E402
from nfl_ats.evidence_conventions import probability_positive_from_draws  # noqa: E402
from nfl_ats.rotation import confirmation_split, load_registry  # noqa: E402

RETRACT_TEAMS = ("ARI", "ATL", "DAL", "HOU", "IND")
GAME_FEATURES = REPO / "data" / "processed" / "game_features.parquet"
PRODUCTION_FEATURES = REPO / "data" / "processed" / "game_features_weak_stack.parquet"
POOL_DECISION_ARCHIVE = (
    REPO / "data" / "raw" / "forecast_archive" / "pool_decision_2009_2025" / "forecasts.parquet"
)
MARKET_ROOT = REPO / "data" / "market" / "raw"
ARTIFACTS_ROOT = REPO / "artifacts"
OUT_DIR = REPO / "artifacts" / "roof_state_screen"

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260910
FAMILY = "roof_state_predicted_open_fade_on_production"


def _latest_schedules() -> Path:
    candidates = sorted((REPO / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise SystemExit("no data/raw/*/schedules.parquet snapshot found")
    return candidates[-1]


def load_population() -> pd.DataFrame:
    feat = pd.read_parquet(
        GAME_FEATURES,
        columns=[
            "game_id",
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            "home_cover",
            "spread_line",
            "result",
        ],
    )
    reg = feat.loc[feat["game_type"].eq("REG")].copy()
    reg["game_id"] = reg["game_id"].astype(str)
    sched = pd.read_parquet(_latest_schedules(), columns=["game_id", "roof", "stadium"])
    sched["game_id"] = sched["game_id"].astype(str)
    reg = reg.merge(sched, on="game_id", how="left", validate="one_to_one")
    pop = reg.loc[
        reg["home_team"].isin(RETRACT_TEAMS) & reg["roof"].isin(["open", "closed"])
    ].reset_index(drop=True)
    pop["is_open"] = pop["roof"].eq("open")
    pop["week_block"] = pop["season"] * 100 + pop["week"]
    return pop


def _block_bootstrap_gap(
    frame: pd.DataFrame,
    outcome_column: str,
    group_a_mask: pd.Series,
    *,
    block_column: str,
    samples: int,
    seed: int,
) -> dict:
    valid = frame[outcome_column].notna()
    working = frame.loc[valid].reset_index(drop=True)
    a_mask = group_a_mask.loc[valid].reset_index(drop=True).to_numpy(dtype=bool)
    outcome = working[outcome_column].to_numpy(dtype=float)
    blocks = list(working.groupby(block_column, sort=False).indices.values())

    def _metric(positions: np.ndarray) -> float:
        sub_outcome = outcome[positions]
        sub_a = a_mask[positions]
        if sub_a.sum() == 0 or (~sub_a).sum() == 0:
            return float("nan")
        return float(sub_outcome[sub_a].mean() - sub_outcome[~sub_a].mean())

    point = _metric(np.arange(len(working)))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for i in range(samples):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        positions = np.concatenate([blocks[j] for j in selected])
        draws[i] = _metric(positions)
    finite = draws[~np.isnan(draws)]
    if finite.size == 0:
        lower = upper = prob_pos = float("nan")
    else:
        lower, upper = (float(x) for x in np.quantile(finite, [0.025, 0.975]))
        prob_pos = float(probability_positive_from_draws(finite))
    return {
        "point_estimate": point,
        "n_group_a": int(a_mask.sum()),
        "n_group_b": int((~a_mask).sum()),
        "ci95": [lower, upper],
        "probability_positive": prob_pos,
        "samples": samples,
        "nan_draws": int(np.isnan(draws).sum()),
        "block_column": block_column,
    }


def opener_population(pop: pd.DataFrame) -> pd.DataFrame:
    sched_full = pd.read_parquet(_latest_schedules(), columns=["game_id", "season", "week"])
    sched_full["game_id"] = sched_full["game_id"].astype(str)
    pairing = cached_pairing_table(
        MARKET_ROOT,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open",),
        schedule=sched_full,
    )
    pairing["game_id"] = pairing["game_id"].astype(str)
    merged = pop.merge(
        pairing[["game_id", "home_spread"]].rename(columns={"home_spread": "tue_open_home_spread"}),
        on="game_id",
        how="inner",
    )
    merged["margin_vs_open"] = merged["result"] - merged["tue_open_home_spread"]
    merged["home_cover_at_open"] = merged["margin_vs_open"].gt(0.0).astype(float)
    merged.loc[merged["margin_vs_open"].eq(0.0), "home_cover_at_open"] = np.nan
    return merged


def coverage_by_season(pop: pd.DataFrame) -> pd.DataFrame:
    table = pop.groupby(["home_team", "season"])["roof"].value_counts().unstack(fill_value=0)
    return table.reset_index()


def market_pricing_check(pop: pd.DataFrame) -> dict:
    rows = []
    open_games = pop.loc[pop["is_open"]]
    for row in open_games.itertuples(index=False):
        controls = pop.loc[
            (pop["home_team"] == row.home_team)
            & (pop["season"] == row.season)
            & (~pop["is_open"])
            & (pop["game_id"] != row.game_id)
        ]
        if controls.empty:
            continue
        rows.append(
            {
                "game_id": row.game_id,
                "season": int(row.season),
                "home_team": row.home_team,
                "open_spread_line": float(row.spread_line),
                "control_mean_spread_line": float(controls["spread_line"].mean()),
                "n_controls": len(controls),
                "diff": float(row.spread_line) - float(controls["spread_line"].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"n_open_games_with_controls": 0, "note": "no open game had an eligible control"}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks = list(frame.groupby("season", sort=False).indices.values())
    point = float(frame["diff"].mean())
    draws = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for i in range(BOOTSTRAP_SAMPLES):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        positions = np.concatenate([blocks[j] for j in selected])
        draws[i] = float(frame["diff"].to_numpy()[positions].mean())
    lower, upper = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
    return {
        "n_open_games_with_controls": len(frame),
        "n_open_games_no_controls": int(len(open_games) - len(frame)),
        "mean_diff_spread_points": point,
        "ci95": [lower, upper],
        "probability_market_already_shades_home_when_open": float(
            probability_positive_from_draws(-draws)
        ),
        "samples": BOOTSTRAP_SAMPLES,
        "sample_blocks": len(blocks),
        "sign_convention": (
            "home-perspective spread_line; negative diff means the open game's line "
            "favoured the home team MORE than that same team's other (closed) home games "
            "that season, i.e. the market already shaded toward home when the roof opens"
        ),
    }


def cmd_screen(_args: argparse.Namespace) -> None:
    pop = load_population()
    opener = opener_population(pop)

    close_gap_week = _block_bootstrap_gap(
        pop,
        "home_cover",
        pop["is_open"],
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    close_gap_season = _block_bootstrap_gap(
        pop,
        "home_cover",
        pop["is_open"],
        block_column="season",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    opener_gap_week = _block_bootstrap_gap(
        opener,
        "home_cover_at_open",
        opener["is_open"],
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    opener_gap_season = _block_bootstrap_gap(
        opener,
        "home_cover_at_open",
        opener["is_open"],
        block_column="season",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    odd = pop.loc[pop["season"] % 2 == 1].reset_index(drop=True)
    even = pop.loc[pop["season"] % 2 == 0].reset_index(drop=True)
    odd_gap = _block_bootstrap_gap(
        odd,
        "home_cover",
        odd["is_open"],
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    even_gap = _block_bootstrap_gap(
        even,
        "home_cover",
        even["is_open"],
        block_column="week_block",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    venue_open_rate = (
        pop.groupby("home_team")["is_open"].agg(["mean", "sum", "count"]).reset_index()
    )

    pricing = market_pricing_check(pop)

    result = {
        "population": {
            "n_games_reg_2009_2025": len(pop),
            "n_open": int(pop["is_open"].sum()),
            "n_closed": int((~pop["is_open"]).sum()),
            "retract_teams": list(RETRACT_TEAMS),
        },
        "coverage_by_season_venue": coverage_by_season(pop).to_dict(orient="records"),
        "venue_lifetime_open_rate": venue_open_rate.to_dict(orient="records"),
        "cover_rates": {
            "close_grade": {
                "open": float(pop.loc[pop["is_open"], "home_cover"].mean()),
                "closed": float(pop.loc[~pop["is_open"], "home_cover"].mean()),
            },
            "opener_grade": {
                "n_paired_2020_2025": len(opener),
                "open": float(opener.loc[opener["is_open"], "home_cover_at_open"].mean()),
                "closed": float(opener.loc[~opener["is_open"], "home_cover_at_open"].mean()),
            },
        },
        "home_cover_gap_open_vs_closed": {
            "close_grade_2009_2025": {
                "week_blocked_primary": close_gap_week,
                "season_blocked_secondary": close_gap_season,
            },
            "opener_grade_2020_2025_paired": {
                "week_blocked_primary": opener_gap_week,
                "season_blocked_secondary": opener_gap_season,
            },
        },
        "market_pricing_check": pricing,
        "split_half_reliability_odd_even_seasons_close_grade": {
            "odd_seasons": odd_gap,
            "even_seasons": even_gap,
            "note": (
                "is_open is a per-game situational condition (this week's roof decision), "
                "not a persistent per-team trait with a year-over-year value; a formal "
                "split-half correlation coefficient does not apply the way it would to a "
                "team trait. Reported instead: the home-cover gap measured independently "
                "on odd-numbered and even-numbered seasons."
            ),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pop.to_parquet(OUT_DIR / "roof_state_population.parquet", index=False)
    (OUT_DIR / "screen_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'screen_results.json'}")


def _venue_features(frame: pd.DataFrame) -> np.ndarray:
    temp = frame["forecast_temp_f"].to_numpy(dtype=float)
    precip = frame["forecast_precip_prob_pct"].to_numpy(dtype=float)
    home = frame["home_team"].to_numpy()
    cols = [temp, precip]
    for venue in RETRACT_TEAMS:
        cols.append(np.where(home == venue, temp, 0.0))
    for venue in RETRACT_TEAMS:
        cols.append((home == venue).astype(float))
    return np.column_stack(cols)


def build_prediction_table() -> pd.DataFrame:
    pop = load_population()
    archive = pd.read_parquet(
        POOL_DECISION_ARCHIVE,
        columns=["game_id", "fetch_status", "forecast_temp_f", "forecast_precip_prob_pct"],
    )
    archive["game_id"] = archive["game_id"].astype(str)
    archive = archive.loc[archive["fetch_status"].eq("ok")]
    merged = pop.merge(
        archive[["game_id", "forecast_temp_f", "forecast_precip_prob_pct"]],
        on="game_id",
        how="inner",
    )
    merged = merged.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    merged["predicted_open"] = False
    merged["prediction_basis"] = "cold_start_no_prior_season_default_closed"

    for season in sorted(merged["season"].unique()):
        train = merged.loc[merged["season"] < season]
        test_mask = merged["season"].eq(season)
        if train.empty or train["is_open"].nunique() < 2:
            continue
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(_venue_features(train), train["is_open"].astype(int).to_numpy())
        test_rows = merged.loc[test_mask]
        predicted = model.predict(_venue_features(test_rows))
        merged.loc[test_mask, "predicted_open"] = predicted.astype(bool)
        merged.loc[test_mask, "prediction_basis"] = "logistic_walk_forward_prior_seasons_only"
    return merged


def cmd_predict_roof(_args: argparse.Namespace) -> None:
    table = build_prediction_table()
    walk_forward = table.loc[
        table["prediction_basis"].eq("logistic_walk_forward_prior_seasons_only")
    ]
    cold_start = table.loc[
        ~table["prediction_basis"].eq("logistic_walk_forward_prior_seasons_only")
    ]

    def _accuracy(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {"n": 0}
        correct = (frame["predicted_open"] == frame["is_open"]).to_numpy()
        baseline_correct = (~frame["is_open"]).to_numpy()
        predicted_open = frame["predicted_open"].to_numpy()
        actual_open = frame["is_open"].to_numpy()
        precision = (
            float((predicted_open & actual_open).sum() / predicted_open.sum())
            if predicted_open.sum() > 0
            else float("nan")
        )
        recall = (
            float((predicted_open & actual_open).sum() / actual_open.sum())
            if actual_open.sum() > 0
            else float("nan")
        )
        return {
            "n": len(frame),
            "n_actual_open": int(actual_open.sum()),
            "n_predicted_open": int(predicted_open.sum()),
            "accuracy": float(correct.mean()),
            "baseline_always_closed_accuracy": float(baseline_correct.mean()),
            "precision_on_open_calls": precision,
            "recall_of_open_games": recall,
        }

    full_accuracy = _accuracy(table)
    walk_forward_accuracy = _accuracy(walk_forward)
    cold_start_accuracy = _accuracy(cold_start)

    corr_point = float(
        np.corrcoef(table["predicted_open"].astype(float), table["is_open"].astype(float))[0, 1]
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks = list(table.groupby("week_block", sort=False).indices.values())
    pred = table["predicted_open"].astype(float).to_numpy()
    actual = table["is_open"].astype(float).to_numpy()
    draws = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for i in range(BOOTSTRAP_SAMPLES):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        positions = np.concatenate([blocks[j] for j in selected])
        p = pred[positions]
        a = actual[positions]
        if np.std(p) == 0 or np.std(a) == 0:
            draws[i] = np.nan
            continue
        draws[i] = float(np.corrcoef(p, a)[0, 1])
    finite = draws[~np.isnan(draws)]
    corr_lower, corr_upper = (float(x) for x in np.quantile(finite, [0.025, 0.975]))
    corr_prob_positive = float(probability_positive_from_draws(finite))

    by_venue = table.groupby("home_team").apply(lambda g: pd.Series(_accuracy(g))).reset_index()

    result = {
        "n_games_with_forecast_and_roof": len(table),
        "full_population_accuracy": full_accuracy,
        "walk_forward_learned_accuracy": walk_forward_accuracy,
        "cold_start_default_accuracy": cold_start_accuracy,
        "accuracy_by_venue": by_venue.to_dict(orient="records"),
        "prediction_outcome_correlation": {
            "point_estimate": corr_point,
            "week_blocked_ci95": [corr_lower, corr_upper],
            "probability_positive": corr_prob_positive,
            "samples": BOOTSTRAP_SAMPLES,
            "sample_blocks": len(blocks),
        },
        "method": (
            "Per-season walk-forward logistic regression: features are the pool_decision-cutoff "
            "forecast temperature and precipitation probability, each venue's own temperature "
            "interacted separately, and five venue dummy intercepts. Refit every season using "
            "strictly PRIOR seasons only (expanding window); season 2009 has no prior season and "
            "falls back to a fixed 'always closed' default (disclosed as cold_start rows)."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_DIR / "roof_prediction_table.parquet", index=False)
    (OUT_DIR / "predict_roof_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'predict_roof_results.json'}")


def _accuracy_metric(frame: pd.DataFrame) -> dict:
    valid = frame.dropna(subset=["baseline_correct", "candidate_correct"])
    return {
        "accuracy_points": 100.0
        * float((valid["candidate_correct"] - valid["baseline_correct"]).mean()),
        "candidate_accuracy": 100.0 * float(valid["candidate_correct"].mean()),
        "baseline_accuracy": 100.0 * float(valid["baseline_correct"].mean()),
    }


def _summarize(frame: pd.DataFrame, samples: int, seed: int) -> dict:
    point = _accuracy_metric(frame)
    week = week_blocked_bootstrap(frame, _accuracy_metric, block="week", samples=samples, seed=seed)
    season = week_blocked_bootstrap(
        frame, _accuracy_metric, block="season", samples=samples, seed=seed
    )
    w = week.loc[week["metric"].eq("accuracy_points")].iloc[0]
    s = season.loc[season["metric"].eq("accuracy_points")].iloc[0]
    return {
        **point,
        "week_blocked_ci95": [float(w["lower"]), float(w["upper"])],
        "week_blocked_probability_positive": float(w["probability_positive"]),
        "season_blocked_ci95": [float(s["lower"]), float(s["upper"])],
        "season_blocked_probability_positive": float(s["probability_positive"]),
        "n_games": len(frame),
        "n_weeks": int(frame[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(frame["season"].nunique()),
    }


def cmd_production(_args: argparse.Namespace) -> None:
    registry = load_registry()
    features = pd.read_parquet(PRODUCTION_FEATURES)
    training, window = confirmation_split(features, registry, FAMILY)
    if pd.to_datetime(training["gameday"]).max() >= pd.to_datetime(window["gameday"]).min():
        raise SystemExit("confirmation split leaked a training row into the assigned window")
    seasons = tuple(sorted(int(s) for s in window["season"].unique()))
    scoped = pd.concat([training, window], ignore_index=True)

    config = resolve_active_model_config(ARTIFACTS_ROOT)
    scored = opener_pick_evaluation(
        MARKET_ROOT,
        scoped,
        active_model_config=config,
        min_train_games=DEFAULT_MIN_TRAIN_GAMES,
    )
    scored = scored.loc[scored["season"].astype(int).isin(seasons)].reset_index(drop=True)

    predicted = build_prediction_table()[["game_id", "predicted_open"]]
    predicted["game_id"] = predicted["game_id"].astype(str)
    scored["game_id"] = scored["game_id"].astype(str)
    scored = scored.merge(predicted, on="game_id", how="left")
    scored["predicted_open"] = scored["predicted_open"].fillna(False).astype(bool)

    scored["baseline_pick_home"] = scored["home_cover_probability_at_open"].ge(0.5)
    scored["candidate_pick_home"] = scored["baseline_pick_home"] & ~scored["predicted_open"]
    scored["baseline_correct"] = pick_correct(
        scored["baseline_pick_home"], scored["margin_vs_open"]
    )
    scored["candidate_correct"] = pick_correct(
        pd.Series(scored["candidate_pick_home"], index=scored.index), scored["margin_vs_open"]
    )

    graded = scored.dropna(subset=["baseline_correct", "candidate_correct"])
    picks_changed = int((scored["baseline_pick_home"] != scored["candidate_pick_home"]).sum())
    picks_changed_graded = int(
        (graded["baseline_pick_home"] != graded["candidate_pick_home"]).sum()
    )

    summary = _summarize(graded, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    result = {
        "family": FAMILY,
        "window_seasons": list(seasons),
        "grade": "opener",
        "active_model_id": config.get("model_id"),
        "predeclared_tilt": (
            "fade the home team when the roof is predicted open at a venue that usually "
            "closes; flips a home pick to away, never flips an away pick to home"
        ),
        "n_predicted_open_in_window": int(scored["predicted_open"].sum()),
        "picks_changed": picks_changed,
        "picks_changed_graded": picks_changed_graded,
        "summary": summary,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUT_DIR / "production_paired.parquet", index=False)
    (OUT_DIR / "production_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'production_results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ENV-02 roof-state (open/closed) screen")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("screen", help="coverage, cover rates, market pricing, split-half reliability")
    sub.add_parser("predict-roof", help="deadline-visible walk-forward roof-state predictor")
    sub.add_parser("production", help="production stack: predicted-open fade on the opener grade")
    args = parser.parse_args()
    if args.command == "screen":
        cmd_screen(args)
    elif args.command == "predict-roof":
        cmd_predict_roof(args)
    elif args.command == "production":
        cmd_production(args)


if __name__ == "__main__":
    main()
