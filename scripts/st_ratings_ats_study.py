from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "2"

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import four_term_probability_eval as ft
import joint_probability_model_eval as jpm
from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.special_teams import canonicalize_special_teams_ratings, classify_special_teams_unit

PARTICIPATION_ROOT = Path("data/players/participation/raw/20260813T131635Z")
RATINGS_PATH = Path("artifacts/st_player_ratings/season_lagged_2019_2024/ratings.parquet")
DECLARATION_PATH = Path("docs/st_ratings_ats_study.md")
BASE = ["x0_model_logit", "flag_sum", "move", "move_available"]
CAND = [*BASE, "st_diff"]
METRICS = ["log_loss", "brier", "accuracy"]
SAMPLES = 20000
SEED = 20260914
COVER_FLOOR = 0.5


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def st_participants(seasons: list[int]) -> pd.DataFrame:
    rows: set[tuple[str, str, str]] = set()
    for season in seasons:
        path = PARTICIPATION_ROOT / f"season={season}" / "participation.parquet"
        part = pd.read_parquet(
            path,
            columns=["game_id", "possession_team", "offense_personnel",
                     "defense_personnel", "offense_players", "defense_players"],
        )
        for game_id, poss, off_personnel, def_personnel, off, dfn in part.itertuples(
            index=False, name=None
        ):
            if classify_special_teams_unit(off_personnel, def_personnel) is None:
                continue
            away, home = str(game_id).split("_")[2:4]
            if pd.isna(poss) or poss not in (away, home):
                continue
            other = home if poss == away else away
            for team, ids in ((poss, off), (other, dfn)):
                if pd.isna(ids):
                    continue
                for player_id in str(ids).split(";"):
                    player_id = player_id.strip()
                    if player_id:
                        rows.add((str(game_id), str(team), player_id))
    return pd.DataFrame(sorted(rows), columns=["game_id", "team", "player_id"])


def team_ratings(schedules: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    reg = schedules.loc[
        schedules["game_type"].eq("REG") & schedules["season"].between(2020, 2025)
    ].copy()
    reg["gameday"] = pd.to_datetime(reg["gameday"], errors="raise")
    columns = ["game_id", "season", "week", "gameday"]
    tg = pd.concat(
        [reg[[*columns, f"{side}_team"]].rename(columns={f"{side}_team": "team"})
         for side in ("home", "away")],
        ignore_index=True,
    ).sort_values(["team", "season", "gameday", "game_id"])
    groups = tg.groupby(["team", "season"])
    tg["prior_game_id"] = groups["game_id"].shift(1)
    tg["prior_gameday"] = groups["gameday"].shift(1)
    has_prior = tg["prior_game_id"].notna()
    if not tg.loc[has_prior, "prior_gameday"].lt(tg.loc[has_prior, "gameday"]).all():
        raise ValueError("ST participants must come from a strictly earlier game day")
    part = st_participants(sorted(reg["season"].unique().tolist()))
    part = part.merge(reg[["game_id", "season"]], on="game_id", validate="many_to_one")
    rate = ratings.rename(columns={"target_season": "season"})[
        ["season", "player_id", "special_teams_rating"]
    ]
    part = part.merge(rate, on=["season", "player_id"], how="left", validate="many_to_one")
    part["rated"] = part["special_teams_rating"].notna().astype(float)
    part["rating0"] = part["special_teams_rating"].fillna(0.0)
    agg = (
        part.groupby(["game_id", "team"])
        .agg(st_n=("player_id", "size"), st_rated=("rated", "mean"), st_mean=("rating0", "mean"))
        .reset_index()
        .rename(columns={"game_id": "prior_game_id"})
    )
    out = tg.merge(agg, on=["prior_game_id", "team"], how="left", validate="many_to_one")
    return out[["game_id", "team", "prior_game_id", "prior_gameday", "st_n", "st_rated", "st_mean"]]


def loss_values(frame: pd.DataFrame, p_col: str) -> np.ndarray:
    p = frame[p_col].to_numpy(dtype=float).clip(1e-6, 1 - 1e-6)
    y = frame["home_covered"].to_numpy(dtype=float)
    return np.column_stack((
        -(y * np.log(p) + (1 - y) * np.log(1 - p)),
        (p - y) ** 2,
        (p >= 0.5) == y,
    ))


def metrics(frame: pd.DataFrame, p_col: str) -> dict[str, float]:
    return dict(zip(METRICS, loss_values(frame, p_col).mean(axis=0).tolist(), strict=True))


def paired(frame: pd.DataFrame) -> dict[str, Any]:
    work = frame[["season", "week"]].reset_index(drop=True).copy()
    names = []
    for scope, suffix in (("oos", ""), ("in", "_in")):
        effects = loss_values(frame, f"p_base{suffix}") - loss_values(frame, f"p_cand{suffix}")
        effects[:, 2] *= -100.0
        for i, metric in enumerate(METRICS):
            name = f"{scope}_{metric}"
            names.append(name)
            work[name] = effects[:, i]

    def metric_fn(sub: pd.DataFrame) -> dict[str, float]:
        return sub[names].mean().to_dict()

    def draw_factory(sub: pd.DataFrame) -> Any:
        values = sub[names].to_numpy(dtype=float)

        def draw(positions: np.ndarray) -> dict[str, float]:
            return dict(zip(names, values[positions].mean(axis=0), strict=True))

        return draw

    intervals = week_blocked_bootstrap(
        work, metric_fn, block="week", samples=SAMPLES, seed=SEED,
        metric_draw_factory=draw_factory,
    ).set_index("metric")
    return {
        scope: {
            metric: {
                "effect": float(intervals.loc[f"{scope}_{metric}", "estimate"]),
                "interval_low": float(intervals.loc[f"{scope}_{metric}", "lower"]),
                "interval_high": float(intervals.loc[f"{scope}_{metric}", "upper"]),
                "probability_positive": float(intervals.loc[f"{scope}_{metric}", "probability_positive"]),
            }
            for metric in METRICS
        }
        for scope in ("oos", "in")
    }


def insample_fit(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    beta, means, stds, hessian = jpm.standardize_fit(df, cols, "home_covered", 1e-3)
    return (
        jpm.predict_p(df, cols, beta, means, stds),
        jpm.natural_coefficients(beta, hessian, cols, means, stds),
    )


def decisive(frame: pd.DataFrame) -> dict[str, Any]:
    changed = frame["p_base"].ge(0.5).ne(frame["p_cand"].ge(0.5))
    sub = frame.loc[changed]
    return {
        "n_decisive": len(sub),
        **{f"{arm}_record": jpm.games_record(sub[f"p_{arm}"].ge(0.5).eq(sub["home_covered"]))
           for arm in ("base", "cand")},
    }


def reliability(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for arm in ("market", "model", "base", "cand"):
        p = frame[f"p_{arm}"]
        bins = np.minimum((p * 5).astype(int), 4)
        for bucket in range(5):
            selected = bins.eq(bucket)
            rows.append({
                "arm": arm, "lower": bucket / 5, "upper": (bucket + 1) / 5,
                "n_games": int(selected.sum()),
                "mean_probability": float(p.loc[selected].mean()) if selected.any() else None,
                "home_cover_rate": float(frame.loc[selected, "home_covered"].mean()) if selected.any() else None,
            })
    return rows


def main() -> None:
    declaration_hash = sha256(DECLARATION_PATH)
    ratings = canonicalize_special_teams_ratings(pd.read_parquet(RATINGS_PATH))
    schedules = jpm.load_schedules()
    tr = team_ratings(schedules, ratings)
    df, provenance = ft.load_population()
    for side, prefix in (("home", "h"), ("away", "a")):
        team_frame = tr.rename(columns={
            "team": f"{side}_team", "st_n": f"{prefix}_n",
            "st_rated": f"{prefix}_rated", "st_mean": f"{prefix}_mean",
            "prior_game_id": f"{prefix}_prior", "prior_gameday": f"{prefix}_prior_gameday",
        })
        df = df.merge(team_frame, on=["game_id", f"{side}_team"], how="left", validate="one_to_one")
    df["has_prior"] = df["h_n"].notna() & df["a_n"].notna()
    df["season_ratings_available"] = df["season"].isin(ratings["target_season"].unique())
    df["st_diff"] = (df["h_mean"] - df["a_mean"]).fillna(0.0)
    df["covered"] = df["has_prior"] & df["h_rated"].ge(COVER_FLOOR) & df["a_rated"].ge(COVER_FLOOR)
    df["both_any_rated"] = df["h_rated"].gt(0) & df["a_rated"].gt(0)
    df["nonzero"] = df["st_diff"].ne(0.0)
    df["mean_share_rated"] = df[["h_rated", "a_rated"]].fillna(0.0).mean(axis=1)
    coverage = []
    for season, sub in df.groupby("season"):
        coverage.append({
            "season": int(season), "games": len(sub),
            "with_prior_st_game": int(sub["has_prior"].sum()),
            "missing_prior_st_game": int((~sub["has_prior"]).sum()),
            "missing_season_ratings": int((~sub["season_ratings_available"]).sum()),
            "both_any_rated": int(sub["both_any_rated"].sum()),
            "mean_share_rated": float(sub["mean_share_rated"].mean()),
            "covered": int(sub["covered"].sum()),
            "below_coverage_floor": int((sub["has_prior"] & ~sub["covered"]).sum()),
            "nonzero_st_diff": int(sub["nonzero"].sum()),
            "share_nonzero_st_diff": float(sub["nonzero"].mean()),
        })
    if not np.isfinite(df[[*CAND, "home_covered"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite model input or target")
    if sorted(df["season"].unique()) != list(range(2020, 2026)):
        raise ValueError("The declared six seasons are required")
    base = jpm.loso_arm(df, BASE, "st_base", select_penalty=False)
    cand = jpm.loso_arm(df, CAND, "st_cand", select_penalty=False)
    df["p_base"] = base["oos_p"]
    df["p_cand"] = cand["oos_p"]
    df["p_base_in"], base_in = insample_fit(df, BASE)
    df["p_cand_in"], cand_in = insample_fit(df, CAND)
    df["p_market"] = 0.5
    df["p_model"] = df["home_cover_probability_at_open_raw"].astype(float)
    results = {}
    seasonal = []
    for label, sub in (("all", df), ("covered", df.loc[df["covered"]])):
        if sub.empty:
            results[label] = {"n_games": 0, "status": "unavailable_no_covered_games"}
            continue
        block = {"n_games": len(sub), "n_blocks": len(sub[["season", "week"]].drop_duplicates()),
                 "decisive": decisive(sub)}
        for arm in ("base", "cand"):
            block[f"{arm}_oos"] = metrics(sub, f"p_{arm}")
            block[f"{arm}_in"] = metrics(sub, f"p_{arm}_in")
            block[f"{arm}_gap_in_minus_oos"] = {
                metric: block[f"{arm}_in"][metric] - block[f"{arm}_oos"][metric]
                for metric in METRICS
            }
        block["market"] = metrics(sub, "p_market")
        block["model_only"] = metrics(sub, "p_model")
        block["paired"] = paired(sub)
        results[label] = block
        for season, season_frame in sub.groupby("season"):
            seasonal.append({
                "population": label, "season": int(season), "n_games": len(season_frame),
                "decisive": decisive(season_frame),
                **{arm: metrics(season_frame, f"p_{arm}") for arm in ("market", "model", "base", "cand")},
            })
    coefficients = [fold["st_diff"]["coef"] for fold in cand["fold_coefficients"].values()]
    provenance.update({
        "declaration_sha256_before_run": declaration_hash,
        "script_sha256": sha256(Path(__file__)), "ratings_sha256": sha256(RATINGS_PATH),
        "ratings_path": str(RATINGS_PATH),
        "participation_partitions": {
            str(season): sha256(PARTICIPATION_ROOT / f"season={season}" / "participation.parquet")
            for season in range(2020, 2026)
        },
        "bootstrap_samples": SAMPLES, "bootstrap_seed": SEED,
        "bootstrap_unit": "season-week", "numeric_threads": 2,
    })
    result = {
        "decisive": decisive(df), "coverage": coverage, "results": results,
        "fold_coefficients": cand["fold_coefficients"],
        "base_fold_coefficients": base["fold_coefficients"],
        "pooled_coefficients": {"base": base_in, "cand": cand_in},
        "st_sign_stability": {"positive": sum(c > 0 for c in coefficients), "folds": len(coefficients)},
        "reliability": reliability(df), "seasonal": seasonal,
        "looks": {"family": "st_rating_ats_study", "oos_effect_cells": 6,
                  "in_sample_effect_cells": 6, "season_metric_cells": 36,
                  "arms": 4, "fits": 14, "calibration_bins": 20, "total": 86},
        "provenance": provenance,
    }
    print("RESULT " + json.dumps(result, allow_nan=False, default=str))
    prediction_columns = [
        "game_id", "season", "week", "home_team", "away_team", "home_covered",
        *CAND, "h_prior", "a_prior", "h_prior_gameday", "a_prior_gameday",
        "h_n", "a_n", "h_rated", "a_rated", "has_prior", "covered", "season_ratings_available",
        "p_base", "p_cand", "p_base_in", "p_cand_in", "p_market", "p_model",
    ]
    predictions = df[prediction_columns].to_json(orient="records", lines=True, date_format="iso")
    print("PREDICTIONS_BEGIN")
    print(predictions)
    print("PREDICTIONS_END")


if __name__ == "__main__":
    main()
