from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

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

FEATURES = ["home_favorite", "spread_size", "key_number_distance", "prior_move_diff", "rest_diff"]
KEY_NUMBERS = (3.0, 7.0, 10.0, 14.0, 17.0)
RIDGE = FIT_RIDGE
SEED = 20260923
SAMPLES = 4000
CFB_TRAIN_SEASONS = (2021, 2022, 2023, 2024)
CFB_BOOK = "Bovada"
CFB_LINES_ROOT = REPO / "data/cfb/lines/raw/20260816T143907Z"
CFB_SCHEDULES_ROOT = REPO / "data/cfb/schedules/raw/20260816T162105Z"
LOOKS: list[str] = []


def record_look(label: str) -> None:
    LOOKS.append(label)


def key_number_distance(spread: pd.Series) -> pd.Series:
    magnitude = spread.abs()
    distances = pd.concat([(magnitude - k).abs() for k in KEY_NUMBERS], axis=1)
    return distances.min(axis=1)


def attach_transfer_features(
    frame: pd.DataFrame, *, date_col: str, move_col: str
) -> pd.DataFrame:
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
        [home_long[["game_id", "season", date_col, "team", "signed_move"]],
         away_long[["game_id", "season", date_col, "team", "signed_move"]]],
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
    frame["prior_move_diff"] = frame["home_prior_move"].fillna(0.0) - frame["away_prior_move"].fillna(0.0)
    median_rest = pd.concat([frame["home_rest"], frame["away_rest"]]).median()
    frame["rest_diff"] = frame["home_rest"].fillna(median_rest) - frame["away_rest"].fillna(median_rest)
    return frame


def load_cfb_population() -> pd.DataFrame:
    frames = []
    for season in CFB_TRAIN_SEASONS:
        lines = pd.read_parquet(CFB_LINES_ROOT / f"season={season}" / "lines.parquet")
        sched = pd.read_parquet(CFB_SCHEDULES_ROOT / f"season={season}" / "schedules.parquet")
        spread = lines.loc[
            lines.market_type.eq("spread") & lines.book.eq(CFB_BOOK) & lines.opening_lines.notna()
        ].copy()
        base = sched.loc[
            sched.completed.fillna(False) & ~sched.neutral_site.fillna(False),
            ["game_id", "season", "week", "home_team", "away_team", "home_points",
             "away_points", "start_date"],
        ]
        matched = spread.merge(base[["game_id", "home_team", "away_team"]], on="game_id", how="inner")
        home_rows = matched.loc[matched.abbr.eq(matched.home_team), ["game_id", "lines", "opening_lines"]]
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


def load_nfl_population() -> tuple[pd.DataFrame, dict[str, Any]]:
    population, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    population["home_favorite"] = population["tue_open_home_spread"].lt(0.0).astype(float)
    population["spread_size"] = population["tue_open_home_spread"].abs()
    population["key_number_distance"] = key_number_distance(population["tue_open_home_spread"])
    population = attach_transfer_features(population, date_col="gameday", move_col="open_move")
    population = population.dropna(subset=[*FEATURES, "open_move", "home_covered"]).reset_index(drop=True)
    return population, provenance


def design(df: pd.DataFrame, feature_cols: list[str], means: dict[str, float], stds: dict[str, float]) -> np.ndarray:
    cols = [np.ones(len(df))]
    for c in feature_cols:
        cols.append(((df[c].astype(float) - means[c]) / stds[c]).to_numpy())
    return np.column_stack(cols)


def standardize(df: pd.DataFrame, feature_cols: list[str]) -> tuple[dict[str, float], dict[str, float]]:
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


def fit_ridge_beta(x: np.ndarray, y: np.ndarray, l2: float = RIDGE) -> np.ndarray:
    return np.linalg.solve(x.T @ x + l2 * np.eye(x.shape[1]), x.T @ y)


def predict_logit(df: pd.DataFrame, feature_cols: list[str], beta: np.ndarray, means: dict, stds: dict) -> np.ndarray:
    z = np.clip(design(df, feature_cols, means, stds) @ beta, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def predict_linear(df: pd.DataFrame, feature_cols: list[str], beta: np.ndarray, means: dict, stds: dict) -> np.ndarray:
    return design(df, feature_cols, means, stds) @ beta


def loso_logit(df: pd.DataFrame, feature_cols: list[str], target_col: str, arm_name: str) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for held in sorted(int(v) for v in df["season"].unique()):
        train = df.loc[df["season"].ne(held)]
        test = df.loc[df["season"].eq(held)]
        if train.empty or test.empty:
            continue
        means, stds = standardize(train, feature_cols)
        beta = fit_logit_beta(design(train, feature_cols, means, stds), train[target_col].astype(float).to_numpy())
        out.loc[test.index] = predict_logit(test, feature_cols, beta, means, stds)
    record_look(f"loso_{arm_name}")
    return out


def games_record(correct: pd.Series) -> str:
    wins = int(correct.sum())
    return f"{wins}-{len(correct) - wins}"


def diff_report(df: pd.DataFrame, a_pick: str, a_correct: str, b_pick: str, b_correct: str) -> dict[str, Any]:
    diff = df[a_pick].ne(df[b_pick])
    return {
        "n_diff": int(diff.sum()),
        "a_record_on_diff": games_record(df.loc[diff, a_correct]),
        "b_record_on_diff": games_record(df.loc[diff, b_correct]),
    }


def paired_accuracy_effect(df: pd.DataFrame, candidate_correct: str, baseline_correct: str) -> dict[str, float]:
    def metric_fn(sub: pd.DataFrame) -> dict[str, float]:
        return {"effect": float((sub[candidate_correct].astype(float).mean() - sub[baseline_correct].astype(float).mean()) * 100.0)}

    result = week_blocked_bootstrap(df, metric_fn, block="season", samples=SAMPLES, seed=SEED)
    row = result.iloc[0]
    return {
        "effect_accuracy_points": float(row["estimate"]),
        "interval_low": float(row["lower"]),
        "interval_high": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
    }


def paired_mae_effect(df: pd.DataFrame, candidate_abs_err: str, baseline_abs_err: str) -> dict[str, float]:
    def metric_fn(sub: pd.DataFrame) -> dict[str, float]:
        return {"effect": float(sub[baseline_abs_err].mean() - sub[candidate_abs_err].mean())}

    result = week_blocked_bootstrap(df, metric_fn, block="season", samples=SAMPLES, seed=SEED)
    row = result.iloc[0]
    return {
        "effect_mae_improvement": float(row["estimate"]),
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


def calibration_table(df: pd.DataFrame, p_col: str, target_col: str) -> list[dict[str, Any]]:
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for lo, hi in pairwise(edges):
        mask = df[p_col].ge(lo) & df[p_col].lt(hi if hi < 1.0 else 1.0 + 1e-9)
        n = int(mask.sum())
        rows.append({
            "bin": f"{lo:.1f}_to_{hi:.1f}",
            "n": n,
            "mean_predicted": float(df.loc[mask, p_col].mean()) if n else None,
            "actual_rate": float(df.loc[mask, target_col].mean()) if n else None,
        })
    record_look(f"calibration_table_{p_col}")
    return rows


def split_half_reliability(cfb: pd.DataFrame, nfl: pd.DataFrame) -> float:
    seasons = sorted(cfb["season"].unique())
    half_a = cfb.loc[cfb["season"].isin(seasons[0::2])]
    half_b = cfb.loc[cfb["season"].isin(seasons[1::2])]
    means_a, stds_a = standardize(half_a, FEATURES)
    beta_a = fit_logit_beta(design(half_a, FEATURES, means_a, stds_a), half_a["home_covered"].to_numpy())
    means_b, stds_b = standardize(half_b, FEATURES)
    beta_b = fit_logit_beta(design(half_b, FEATURES, means_b, stds_b), half_b["home_covered"].to_numpy())
    pred_a = predict_logit(nfl, FEATURES, beta_a, means_a, stds_a)
    pred_b = predict_logit(nfl, FEATURES, beta_b, means_b, stds_b)
    record_look("split_half_reliability_cfb_season_parity")
    return float(np.corrcoef(pred_a, pred_b)[0, 1])


def main() -> None:
    cfb = load_cfb_population()
    nfl, nfl_provenance = load_nfl_population()

    means_cfb, stds_cfb = standardize(cfb, FEATURES)
    beta_linear = fit_ridge_beta(design(cfb, FEATURES, means_cfb, stds_cfb), cfb["move"].to_numpy())
    record_look("cfb_fit_ridge_linear_move")
    beta_logit = fit_logit_beta(design(cfb, FEATURES, means_cfb, stds_cfb), cfb["home_covered"].to_numpy())
    record_look("cfb_fit_logistic_cover")

    nfl["cfb_pred_move"] = predict_linear(nfl, FEATURES, beta_linear, means_cfb, stds_cfb)
    nfl["cfb_pred_p"] = predict_logit(nfl, FEATURES, beta_logit, means_cfb, stds_cfb)
    nfl["cfb_transfer_logit"] = np.log(np.clip(nfl["cfb_pred_p"], 1e-6, 1 - 1e-6) / (1 - np.clip(nfl["cfb_pred_p"], 1e-6, 1 - 1e-6)))
    reliability = split_half_reliability(cfb, nfl)

    nfl["abs_err_cfb"] = (nfl["open_move"] - nfl["cfb_pred_move"]).abs()
    nfl["abs_err_zero"] = nfl["open_move"].abs()
    record_look("target_a_cfb_vs_zero_mae")
    move_effect = paired_mae_effect(nfl, "abs_err_cfb", "abs_err_zero")
    move_by_season = nfl.groupby("season").apply(
        lambda s: pd.Series({
            "n": len(s),
            "mae_cfb": s["abs_err_cfb"].mean(),
            "mae_zero": s["abs_err_zero"].mean(),
        }),
        include_groups=False,
    ).reset_index()

    nfl["base_oos_p"] = None
    base_oos = pd.Series(np.nan, index=nfl.index, dtype=float)
    for held in sorted(int(v) for v in nfl["season"].unique()):
        train = nfl.loc[nfl["season"].ne(held)]
        test = nfl.loc[nfl["season"].eq(held)]
        if train.empty or test.empty:
            continue
        fold_means, fold_stds = _standardisers(train)
        fold_beta = _fit_logit(_design(train, fold_means, fold_stds), train["home_covered"].to_numpy(), FIT_RIDGE)
        base_oos.loc[test.index] = _predict(test, fold_beta, fold_means, fold_stds)
    record_look("loso_base_four_term")
    nfl["base_oos_p"] = base_oos

    plus_features = list(FIT_FEATURES) + ["cfb_transfer_logit"]
    nfl["plus_oos_p"] = loso_logit(nfl, plus_features, "home_covered", "base_plus_cfb_transfer")
    nfl["nfl_only_oos_p"] = loso_logit(nfl, FEATURES, "home_covered", "nfl_only_twin")

    nfl["base_pick"] = nfl["base_oos_p"].ge(0.5)
    nfl["base_correct"] = nfl["base_pick"].astype(float).eq(nfl["home_covered"]).astype(float)
    nfl["plus_pick"] = nfl["plus_oos_p"].ge(0.5)
    nfl["plus_correct"] = nfl["plus_pick"].astype(float).eq(nfl["home_covered"]).astype(float)
    nfl["nfl_only_pick"] = nfl["nfl_only_oos_p"].ge(0.5)
    nfl["nfl_only_correct"] = nfl["nfl_only_pick"].astype(float).eq(nfl["home_covered"]).astype(float)
    nfl["cfb_pick"] = nfl["cfb_pred_p"].ge(0.5)
    nfl["cfb_correct"] = nfl["cfb_pick"].astype(float).eq(nfl["home_covered"]).astype(float)

    records = {
        "base_4term": games_record(nfl["base_correct"]),
        "base_plus_cfb_transfer": games_record(nfl["plus_correct"]),
        "nfl_only_twin": games_record(nfl["nfl_only_correct"]),
        "cfb_direct": games_record(nfl["cfb_correct"]),
    }

    cell_cfb_direct_vs_base = {
        "diff": diff_report(nfl, "cfb_pick", "cfb_correct", "base_pick", "base_correct"),
        "effect": paired_accuracy_effect(nfl, "cfb_correct", "base_correct"),
        "candidate_metrics": prob_metrics(nfl["cfb_pred_p"].to_numpy(), nfl["home_covered"].to_numpy()),
        "baseline_metrics": prob_metrics(nfl["base_oos_p"].to_numpy(), nfl["home_covered"].to_numpy()),
    }
    record_look("cfb_direct_vs_base_4term")

    cell_plus_vs_base = {
        "diff": diff_report(nfl, "plus_pick", "plus_correct", "base_pick", "base_correct"),
        "effect": paired_accuracy_effect(nfl, "plus_correct", "base_correct"),
        "candidate_metrics": prob_metrics(nfl["plus_oos_p"].to_numpy(), nfl["home_covered"].to_numpy()),
        "baseline_metrics": prob_metrics(nfl["base_oos_p"].to_numpy(), nfl["home_covered"].to_numpy()),
        "calibration": calibration_table(nfl, "plus_oos_p", "home_covered"),
    }
    record_look("base_plus_cfb_transfer_vs_base_4term")

    cell_nflonly_vs_base = {
        "diff": diff_report(nfl, "nfl_only_pick", "nfl_only_correct", "base_pick", "base_correct"),
        "effect": paired_accuracy_effect(nfl, "nfl_only_correct", "base_correct"),
        "candidate_metrics": prob_metrics(nfl["nfl_only_oos_p"].to_numpy(), nfl["home_covered"].to_numpy()),
        "baseline_metrics": prob_metrics(nfl["base_oos_p"].to_numpy(), nfl["home_covered"].to_numpy()),
    }
    record_look("nfl_only_twin_vs_base_4term")

    cell_nflonly_vs_cfbdirect = {
        "diff": diff_report(nfl, "nfl_only_pick", "nfl_only_correct", "cfb_pick", "cfb_correct"),
        "effect": paired_accuracy_effect(nfl, "nfl_only_correct", "cfb_correct"),
    }
    record_look("nfl_only_twin_vs_cfb_direct")

    base_calibration = calibration_table(nfl, "base_oos_p", "home_covered")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts/opener_error_transfer" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "cfb_train_seasons": list(CFB_TRAIN_SEASONS),
        "cfb_book": CFB_BOOK,
        "cfb_games": len(cfb),
        "nfl_games": len(nfl),
        "features": FEATURES,
        "ridge": RIDGE,
        "seed": SEED,
        "samples": SAMPLES,
        "split_half_reliability_cfb_to_nfl": reliability,
        "records": records,
        "target_a_move_vs_zero": {
            "effect": move_effect,
            "per_season": move_by_season.to_dict(orient="records"),
        },
        "target_b_cfb_direct_vs_base": cell_cfb_direct_vs_base,
        "target_b_base_plus_cfb_transfer_vs_base": cell_plus_vs_base,
        "target_b_nfl_only_twin_vs_base": cell_nflonly_vs_base,
        "target_b_nfl_only_twin_vs_cfb_direct": cell_nflonly_vs_cfbdirect,
        "base_calibration": base_calibration,
        "look_count": len(LOOKS),
        "look_log": LOOKS,
        "nfl_provenance": nfl_provenance,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    export_cols = [c for c in [
        "game_id", "season", "week", "home_team", "away_team", "tue_open_home_spread",
        "open_move", "home_covered", "cfb_pred_move", "cfb_pred_p", "base_oos_p", "plus_oos_p",
        "nfl_only_oos_p",
    ] if c in nfl.columns]
    nfl[export_cols].to_csv(out_dir / "per_game.csv", index=False)

    print(f"artifact_dir={out_dir}")
    print("cfb_games", len(cfb), "nfl_games", len(nfl))
    print("reliability", reliability)
    print("records", json.dumps(records))
    print("move_effect", json.dumps(move_effect))
    print("cfb_direct_vs_base", json.dumps(cell_cfb_direct_vs_base["effect"]), cell_cfb_direct_vs_base["diff"])
    print("plus_vs_base", json.dumps(cell_plus_vs_base["effect"]), cell_plus_vs_base["diff"])
    print("nflonly_vs_base", json.dumps(cell_nflonly_vs_base["effect"]), cell_nflonly_vs_base["diff"])
    print("nflonly_vs_cfbdirect", json.dumps(cell_nflonly_vs_cfbdirect["effect"]), cell_nflonly_vs_cfbdirect["diff"])
    print("look_count", len(LOOKS))


if __name__ == "__main__":
    main()
