from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.pick_probability_fit import (
    FIT_FEATURES,
    FIT_RIDGE,
    _design,
    _fit_logit,
    _predict,
    _standardisers,
    build_fit_population,
)

REPO = Path(__file__).resolve().parents[1]

FEATURES = ["home_favorite", "spread_size", "key_number_distance", "prior_move_diff", "rest_diff"]
KEY_NUMBERS = (3.0, 7.0, 10.0, 14.0, 17.0)
RIDGE = FIT_RIDGE
SEED = 20260923
SAMPLES = 4000
CFB_TRAIN_SEASONS_ALL = (
    2012,
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2021,
    2022,
    2023,
    2024,
    2025,
)
CFB_BOOK_BY_SEASON = dict.fromkeys(range(2012, 2020), "5Dimes & sportbet")
CFB_BOOK_BY_SEASON.update(dict.fromkeys((2021, 2022, 2023, 2024, 2025), "Bovada"))
CFB_LINES_ROOT = REPO / "data/cfb/lines/raw/20260816T143907Z"
CFB_SCHEDULES_ROOT = REPO / "data/cfb/schedules/raw/20260816T162105Z"
EXTENDED_POPULATION_PATH = (
    REPO / "artifacts/extended_fit_population/20260923T205910Z/population.parquet"
)
SBR_SCORED_ARTIFACT = REPO / "artifacts/sbr_era_opener_eval/20260819T233013Z/scored.parquet"
FEATURES_PATH = REPO / "data/processed/game_features_weak_stack.parquet"
LOOKS: list[str] = []


def record_look(label: str) -> None:
    LOOKS.append(label)


def key_number_distance(spread: pd.Series) -> pd.Series:
    magnitude = spread.abs()
    distances = pd.concat([(magnitude - k).abs() for k in KEY_NUMBERS], axis=1)
    return distances.min(axis=1)


def attach_transfer_features(frame: pd.DataFrame, *, date_col: str, move_col: str) -> pd.DataFrame:
    frame = frame.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    home_long = frame[["game_id", "season", date_col, "home_team", move_col]].rename(
        columns={"home_team": "team"}
    )
    home_long["signed_move"] = home_long[move_col]
    away_long = frame[["game_id", "season", date_col, "away_team", move_col]].rename(
        columns={"away_team": "team"}
    )
    away_long["signed_move"] = -away_long[move_col]
    long = pd.concat(
        [
            home_long[["game_id", "season", date_col, "team", "signed_move"]],
            away_long[["game_id", "season", date_col, "team", "signed_move"]],
        ],
        ignore_index=True,
    ).sort_values(["team", "season", date_col])
    long["prior_move"] = long.groupby(["team", "season"])["signed_move"].shift(1)
    long["prev_date"] = long.groupby(["team", "season"])[date_col].shift(1)
    long["rest_days"] = (long[date_col] - long["prev_date"]).dt.days.astype(float)
    home_join = long.rename(
        columns={"team": "home_team", "prior_move": "home_prior_move", "rest_days": "home_rest"}
    )[["game_id", "home_team", "home_prior_move", "home_rest"]]
    away_join = long.rename(
        columns={"team": "away_team", "prior_move": "away_prior_move", "rest_days": "away_rest"}
    )[["game_id", "away_team", "away_prior_move", "away_rest"]]
    frame = frame.merge(home_join, on=["game_id", "home_team"], how="left")
    frame = frame.merge(away_join, on=["game_id", "away_team"], how="left")
    frame["prior_move_diff"] = frame["home_prior_move"].fillna(0.0) - frame[
        "away_prior_move"
    ].fillna(0.0)
    median_rest = pd.concat([frame["home_rest"], frame["away_rest"]]).median()
    frame["rest_diff"] = frame["home_rest"].fillna(median_rest) - frame["away_rest"].fillna(
        median_rest
    )
    return frame


def load_cfb_population() -> pd.DataFrame:
    frames = []
    for season in CFB_TRAIN_SEASONS_ALL:
        book = CFB_BOOK_BY_SEASON[season]
        lines = pd.read_parquet(CFB_LINES_ROOT / f"season={season}" / "lines.parquet")
        sched = pd.read_parquet(CFB_SCHEDULES_ROOT / f"season={season}" / "schedules.parquet")
        spread = lines.loc[
            lines.market_type.eq("spread") & lines.book.eq(book) & lines.opening_lines.notna()
        ].copy()
        base = sched.loc[
            sched.completed.fillna(False) & ~sched.neutral_site.fillna(False),
            [
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "home_points",
                "away_points",
                "start_date",
            ],
        ]
        matched = spread.merge(
            base[["game_id", "home_team", "away_team"]], on="game_id", how="inner"
        )
        home_rows = matched.loc[
            matched.abbr.eq(matched.home_team), ["game_id", "lines", "opening_lines"]
        ]
        home_rows = home_rows.rename(columns={"lines": "home_close", "opening_lines": "home_open"})
        home_rows = home_rows.drop_duplicates("game_id")
        game = base.merge(home_rows, on="game_id", how="inner")
        frames.append(game)
    games = pd.concat(frames, ignore_index=True)
    games["move"] = games["home_close"] - games["home_open"]
    games["home_margin"] = games["home_points"] - games["away_points"]
    games["margin_vs_open"] = games["home_margin"] + games["home_open"]
    games = games.loc[games["margin_vs_open"].ne(0.0)].reset_index(drop=True)
    games["home_covered"] = games["margin_vs_open"].gt(0.0).astype(float)
    games["home_favorite"] = games["home_open"].lt(0.0).astype(float)
    games["spread_size"] = games["home_open"].abs()
    games["key_number_distance"] = key_number_distance(games["home_open"])
    games = attach_transfer_features(games, date_col="start_date", move_col="move")
    games = games.dropna(subset=[*FEATURES, "move", "home_covered"]).reset_index(drop=True)
    return games


def load_extended_nfl_population() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    extended = pd.read_parquet(EXTENDED_POPULATION_PATH)
    extended["game_id"] = extended["game_id"].astype(str)

    teams = pd.read_parquet(FEATURES_PATH, columns=["game_id", "home_team", "away_team"])
    teams["game_id"] = teams["game_id"].astype(str)
    teams = teams.drop_duplicates("game_id")
    extended = extended.merge(teams, on="game_id", how="left")

    true_pop, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    true_pop = true_pop.copy()
    true_pop["game_id"] = true_pop["game_id"].astype(str)
    true_join = true_pop[["game_id", "gameday", "tue_open_home_spread", "open_move"]]

    sbr = pd.read_parquet(SBR_SCORED_ARTIFACT)
    sbr = sbr.loc[sbr["season"].le(2019)].copy()
    sbr["game_id"] = sbr["game_id"].astype(str)
    sbr_join = sbr[["game_id", "gameday", "proxy_open_home_spread"]]

    extended = extended.merge(true_join, on="game_id", how="left")
    extended = extended.merge(sbr_join, on="game_id", how="left", suffixes=("", "_sbr"))
    extended["gameday"] = extended["gameday"].fillna(extended["gameday_sbr"])
    extended["home_open"] = extended["tue_open_home_spread"].fillna(
        extended["proxy_open_home_spread"]
    )
    extended["open_move"] = extended["open_move"].fillna(0.0)
    extended = extended.drop(
        columns=["gameday_sbr", "tue_open_home_spread", "proxy_open_home_spread"]
    )

    extended["home_favorite"] = extended["home_open"].lt(0.0).astype(float)
    extended["spread_size"] = extended["home_open"].abs()
    extended["key_number_distance"] = key_number_distance(extended["home_open"])
    extended = attach_transfer_features(extended, date_col="gameday", move_col="open_move")

    cfb_seasons_available = set(CFB_TRAIN_SEASONS_ALL)
    graded_seasons = sorted(
        s for s in extended["season"].unique() if any(c < s for c in cfb_seasons_available)
    )
    excluded_seasons = sorted(set(extended["season"].unique()) - set(graded_seasons))
    graded_meta = {
        "graded_seasons": [int(s) for s in graded_seasons],
        "excluded_seasons_no_preceding_cfb": [int(s) for s in excluded_seasons],
    }

    extended = extended.loc[extended["season"].isin(graded_seasons)].reset_index(drop=True)
    extended = extended.dropna(subset=[*FEATURES, "home_covered"]).reset_index(drop=True)
    return extended, provenance, graded_meta


def design(
    df: pd.DataFrame, feature_cols: list[str], means: dict[str, float], stds: dict[str, float]
) -> np.ndarray:
    cols = [np.ones(len(df))]
    for c in feature_cols:
        cols.append(((df[c].astype(float) - means[c]) / stds[c]).to_numpy())
    return np.column_stack(cols)


def standardize(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    means = {c: float(df[c].mean()) for c in feature_cols}
    stds = {c: float(df[c].std(ddof=0)) or 1.0 for c in feature_cols}
    return means, stds


def fit_logit_beta(x: np.ndarray, y: np.ndarray, l2: float = RIDGE, iters: int = 50) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        z = np.clip(x @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        grad = x.T @ (y - p) - l2 * beta
        hessian = (x.T * w) @ x + l2 * np.eye(x.shape[1])
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, grad, rcond=None)[0]
        beta = beta + step
    return beta


def predict_logit(
    df: pd.DataFrame, feature_cols: list[str], beta: np.ndarray, means: dict, stds: dict
) -> np.ndarray:
    z = np.clip(design(df, feature_cols, means, stds) @ beta, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def predict_linear(
    df: pd.DataFrame, feature_cols: list[str], beta: np.ndarray, means: dict, stds: dict
) -> np.ndarray:
    return design(df, feature_cols, means, stds) @ beta


def fit_ridge_beta(x: np.ndarray, y: np.ndarray, l2: float = RIDGE) -> np.ndarray:
    return np.linalg.solve(x.T @ x + l2 * np.eye(x.shape[1]), x.T @ y)


def loso_logit(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, arm_name: str
) -> tuple[pd.Series, dict[str, Any]]:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    fold_betas: dict[str, Any] = {}
    for held in sorted(int(v) for v in df["season"].unique()):
        train = df.loc[df["season"].ne(held)]
        test = df.loc[df["season"].eq(held)]
        if train.empty or test.empty:
            continue
        means, stds = standardize(train, feature_cols)
        beta = fit_logit_beta(
            design(train, feature_cols, means, stds), train[target_col].astype(float).to_numpy()
        )
        out.loc[test.index] = predict_logit(test, feature_cols, beta, means, stds)
        fold_betas[str(held)] = {
            "intercept": float(beta[0]),
            **{name: float(beta[idx + 1]) for idx, name in enumerate(feature_cols)},
            "n_train": len(train),
            "n_test": len(test),
        }
    record_look(f"loso_{arm_name}")
    return out, fold_betas


def games_record(correct: pd.Series) -> str:
    wins = int(correct.sum())
    return f"{wins}-{len(correct) - wins}"


def diff_report(
    df: pd.DataFrame, a_pick: str, a_correct: str, b_pick: str, b_correct: str
) -> dict[str, Any]:
    diff = df[a_pick].ne(df[b_pick])
    return {
        "n_diff": int(diff.sum()),
        "a_record_on_diff": games_record(df.loc[diff, a_correct]),
        "b_record_on_diff": games_record(df.loc[diff, b_correct]),
    }


def paired_accuracy_effect(
    df: pd.DataFrame, candidate_correct: str, baseline_correct: str
) -> dict[str, float]:
    def metric_fn(sub: pd.DataFrame) -> dict[str, float]:
        return {
            "effect": float(
                (
                    sub[candidate_correct].astype(float).mean()
                    - sub[baseline_correct].astype(float).mean()
                )
                * 100.0
            )
        }

    result = week_blocked_bootstrap(df, metric_fn, block="season", samples=SAMPLES, seed=SEED)
    row = result.iloc[0]
    return {
        "effect_accuracy_points": float(row["estimate"]),
        "interval_low": float(row["lower"]),
        "interval_high": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
    }


def prob_metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pc = np.clip(p, 1e-9, 1.0 - 1e-9)
    return {
        "accuracy": float(np.mean((pc >= 0.5) == y)),
        "brier": float(np.mean((pc - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log1p(-pc))),
    }


def main() -> None:
    nfl, nfl_provenance, graded_meta = load_extended_nfl_population()
    cfb_all = load_cfb_population()

    per_fold_cfb_meta: dict[str, Any] = {}
    cfb_pred_move = pd.Series(np.nan, index=nfl.index, dtype=float)
    cfb_pred_p = pd.Series(np.nan, index=nfl.index, dtype=float)
    for held in sorted(int(v) for v in nfl["season"].unique()):
        cfb_train = cfb_all.loc[cfb_all["season"].lt(held)]
        test = nfl.loc[nfl["season"].eq(held)]
        if cfb_train.empty or test.empty:
            continue
        means_cfb, stds_cfb = standardize(cfb_train, FEATURES)
        beta_lin = fit_ridge_beta(
            design(cfb_train, FEATURES, means_cfb, stds_cfb), cfb_train["move"].to_numpy()
        )
        beta_logit = fit_logit_beta(
            design(cfb_train, FEATURES, means_cfb, stds_cfb), cfb_train["home_covered"].to_numpy()
        )
        cfb_pred_move.loc[test.index] = predict_linear(
            test, FEATURES, beta_lin, means_cfb, stds_cfb
        )
        cfb_pred_p.loc[test.index] = predict_logit(test, FEATURES, beta_logit, means_cfb, stds_cfb)
        per_fold_cfb_meta[str(held)] = {
            "cfb_train_seasons": sorted(int(s) for s in cfb_train["season"].unique()),
            "cfb_train_games": len(cfb_train),
        }
    record_look("per_season_cfb_train_predict")

    nfl["cfb_pred_move"] = cfb_pred_move
    nfl["cfb_pred_p"] = cfb_pred_p
    nfl = nfl.dropna(subset=["cfb_pred_p"]).reset_index(drop=True)
    nfl["cfb_transfer_logit"] = np.log(
        np.clip(nfl["cfb_pred_p"], 1e-6, 1 - 1e-6)
        / (1 - np.clip(nfl["cfb_pred_p"], 1e-6, 1 - 1e-6))
    )

    base_oos = pd.Series(np.nan, index=nfl.index, dtype=float)
    base_fold_betas: dict[str, Any] = {}
    for held in sorted(int(v) for v in nfl["season"].unique()):
        train = nfl.loc[nfl["season"].ne(held)]
        test = nfl.loc[nfl["season"].eq(held)]
        if train.empty or test.empty:
            continue
        fold_means, fold_stds = _standardisers(train)
        fold_beta = _fit_logit(
            _design(train, fold_means, fold_stds), train["home_covered"].to_numpy(), FIT_RIDGE
        )
        base_oos.loc[test.index] = _predict(test, fold_beta, fold_means, fold_stds)
        base_fold_betas[str(held)] = {
            "intercept": float(fold_beta[0]),
            **{name: float(fold_beta[idx + 1]) for idx, name in enumerate(FIT_FEATURES)},
            "n_train": len(train),
            "n_test": len(test),
        }
    record_look("loso_base_four_term")
    nfl["base_oos_p"] = base_oos

    plus_features = [*FIT_FEATURES, "cfb_transfer_logit"]
    nfl["plus_oos_p"], plus_fold_betas = loso_logit(
        nfl, plus_features, "home_covered", "base_plus_cfb_transfer_v3"
    )

    nfl["base_pick"] = nfl["base_oos_p"].ge(0.5)
    nfl["base_correct"] = nfl["base_pick"].astype(float).eq(nfl["home_covered"]).astype(float)
    nfl["plus_pick"] = nfl["plus_oos_p"].ge(0.5)
    nfl["plus_correct"] = nfl["plus_pick"].astype(float).eq(nfl["home_covered"]).astype(float)
    nfl["cfb_pick"] = nfl["cfb_pred_p"].ge(0.5)
    nfl["cfb_correct"] = nfl["cfb_pick"].astype(float).eq(nfl["home_covered"]).astype(float)

    nfl_2020_2025 = nfl.loc[nfl["season"].ge(2020)].reset_index(drop=True)

    cell_all_graded = {
        "diff": diff_report(nfl, "plus_pick", "plus_correct", "base_pick", "base_correct"),
        "effect": paired_accuracy_effect(nfl, "plus_correct", "base_correct"),
        "n_games": len(nfl),
        "n_seasons": int(nfl["season"].nunique()),
    }
    record_look("added_term_vs_base_all_graded")

    cell_2020_2025 = {
        "diff": diff_report(
            nfl_2020_2025, "plus_pick", "plus_correct", "base_pick", "base_correct"
        ),
        "effect": paired_accuracy_effect(nfl_2020_2025, "plus_correct", "base_correct"),
        "n_games": len(nfl_2020_2025),
        "n_seasons": int(nfl_2020_2025["season"].nunique()),
    }
    record_look("added_term_vs_base_2020_2025_subset")

    cfb_direct_all = {
        "diff": diff_report(nfl, "cfb_pick", "cfb_correct", "base_pick", "base_correct"),
        "effect": paired_accuracy_effect(nfl, "cfb_correct", "base_correct"),
    }
    record_look("cfb_direct_vs_base_all_graded_context")

    records = {
        "base_4term": games_record(nfl["base_correct"]),
        "base_plus_cfb_transfer": games_record(nfl["plus_correct"]),
        "cfb_direct": games_record(nfl["cfb_correct"]),
    }

    plus_oos_metrics = prob_metrics(nfl["plus_oos_p"].to_numpy(), nfl["home_covered"].to_numpy())
    base_oos_metrics = prob_metrics(nfl["base_oos_p"].to_numpy(), nfl["home_covered"].to_numpy())

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts/opener_error_transfer_unit3" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "unit": 3,
        "graded_meta": graded_meta,
        "cfb_train_seasons_available": list(CFB_TRAIN_SEASONS_ALL),
        "cfb_book_by_season": CFB_BOOK_BY_SEASON,
        "per_fold_cfb_meta": per_fold_cfb_meta,
        "features": FEATURES,
        "ridge": RIDGE,
        "seed": SEED,
        "samples": SAMPLES,
        "nfl_games_all_graded": len(nfl),
        "nfl_games_2020_2025_subset": len(nfl_2020_2025),
        "records": records,
        "plus_oos_metrics_all_graded": plus_oos_metrics,
        "base_oos_metrics_all_graded": base_oos_metrics,
        "added_term_vs_base_all_graded": cell_all_graded,
        "added_term_vs_base_2020_2025_subset": cell_2020_2025,
        "cfb_direct_vs_base_all_graded_context": cfb_direct_all,
        "plus_loso_fold_betas": plus_fold_betas,
        "base_four_term_loso_fold_betas": base_fold_betas,
        "look_count": len(LOOKS),
        "look_log": LOOKS,
        "nfl_provenance": nfl_provenance,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    export_cols = [
        c
        for c in [
            "game_id",
            "season",
            "week",
            "opener_source",
            "home_covered",
            "cfb_pred_move",
            "cfb_pred_p",
            "base_oos_p",
            "plus_oos_p",
        ]
        if c in nfl.columns
    ]
    nfl[export_cols].to_csv(out_dir / "per_game.csv", index=False)

    print(f"artifact_dir={out_dir}")
    print("graded_meta", json.dumps(graded_meta))
    print("nfl_games_all_graded", len(nfl), "nfl_games_2020_2025", len(nfl_2020_2025))
    print("records", json.dumps(records))
    print("added_term_vs_base_all_graded", json.dumps(cell_all_graded))
    print("added_term_vs_base_2020_2025_subset", json.dumps(cell_2020_2025))
    print("cfb_direct_vs_base_all_graded_context", json.dumps(cfb_direct_all))
    print("look_count", len(LOOKS))


if __name__ == "__main__":
    main()
