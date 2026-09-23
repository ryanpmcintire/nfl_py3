from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from line_move_yardstick_paired_eval import cell_stats  # noqa: E402
from opener_error_transfer_unit3 import (  # noqa: E402
    CFB_TRAIN_SEASONS_ALL,
    attach_transfer_features,
    design,
    diff_report,
    fit_logit_beta,
    games_record,
    key_number_distance,
    load_cfb_population,
    loso_logit,
    predict_logit,
    standardize,
)
from opener_error_transfer_unit3 import FEATURES as CFB_FEATURES  # noqa: E402

from nfl_ats.pick_probability_fit import build_fit_population  # noqa: E402

EXTENDED_POPULATION_PATH = (
    REPO / "artifacts/extended_fit_population/20260923T205910Z/population.parquet"
)
FEATURES_PATH = REPO / "data/processed/game_features_weak_stack.parquet"
SBR_ODDS_PATH = REPO / "data/processed/sbr_odds.parquet"
SBR_SEASON_START = 2013
SBR_SEASON_END = 2019
BASE_FEATURES = ("model_logit", "composition_flag_sum")
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923
LOOKS: list[str] = []


def record_look(label: str) -> None:
    LOOKS.append(label)


def load_line_move_population() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
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

    sbr = pd.read_parquet(
        SBR_ODDS_PATH,
        columns=["game_id", "season", "game_date", "open_home_spread", "close_home_spread"],
    )
    sbr = sbr.loc[sbr["game_id"].notna()].copy()
    sbr["game_id"] = sbr["game_id"].astype(str)
    sbr = sbr.dropna(subset=["open_home_spread", "close_home_spread"])
    sbr = sbr.loc[sbr["season"].between(SBR_SEASON_START, SBR_SEASON_END)]
    sbr = sbr.drop_duplicates(subset="game_id")
    sbr["sbr_open_move"] = sbr["close_home_spread"] - sbr["open_home_spread"]
    sbr_join = sbr.rename(columns={"game_date": "gameday_sbr"})[
        ["game_id", "gameday_sbr", "open_home_spread", "sbr_open_move"]
    ]

    extended = extended.merge(true_join, on="game_id", how="left")
    extended = extended.merge(sbr_join, on="game_id", how="left")

    extended["gameday"] = extended["gameday"].fillna(extended["gameday_sbr"])
    extended["home_open"] = extended["tue_open_home_spread"].fillna(extended["open_home_spread"])
    extended["open_move"] = extended["open_move"].fillna(extended["sbr_open_move"])
    extended = extended.drop(
        columns=["gameday_sbr", "tue_open_home_spread", "open_home_spread", "sbr_open_move"]
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
    extended = extended.dropna(
        subset=[*CFB_FEATURES, "open_move", "home_covered"]
    ).reset_index(drop=True)
    return extended, provenance, graded_meta


def attach_cfb_transfer_logit(
    nfl: pd.DataFrame, cfb_all: pd.DataFrame
) -> tuple[pd.Series, dict[str, Any]]:
    per_fold_meta: dict[str, Any] = {}
    pred_p = pd.Series(np.nan, index=nfl.index, dtype=float)
    for held in sorted(int(v) for v in nfl["season"].unique()):
        cfb_train = cfb_all.loc[cfb_all["season"].lt(held)]
        test = nfl.loc[nfl["season"].eq(held)]
        if cfb_train.empty or test.empty:
            continue
        means_cfb, stds_cfb = standardize(cfb_train, CFB_FEATURES)
        beta_logit = fit_logit_beta(
            design(cfb_train, CFB_FEATURES, means_cfb, stds_cfb),
            cfb_train["home_covered"].to_numpy(),
        )
        pred_p.loc[test.index] = predict_logit(
            test, CFB_FEATURES, beta_logit, means_cfb, stds_cfb
        )
        per_fold_meta[str(held)] = {
            "cfb_train_seasons": sorted(int(s) for s in cfb_train["season"].unique()),
            "cfb_train_games": len(cfb_train),
        }
    record_look("per_season_cfb_train_predict_line_move")
    return pred_p, per_fold_meta


def build_report(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    line_move_cell = cell_stats(
        f"{label}_line_move_toward_pick", frame, "diff_line_move", BOOTSTRAP_SEED, BOOTSTRAP_DRAWS
    )
    record_look(f"{label}_line_move")
    accuracy_cell = cell_stats(
        f"{label}_accuracy_companion",
        frame,
        "diff_accuracy",
        BOOTSTRAP_SEED + 500,
        BOOTSTRAP_DRAWS,
    )
    record_look(f"{label}_accuracy_companion")
    diff = diff_report(frame, "plus_pick_home", "plus_correct", "base_pick_home", "base_correct")
    return {
        "label": label,
        "n_games": len(frame),
        "n_seasons": int(frame["season"].nunique()),
        "diff": diff,
        "line_move_toward_pick_cell": line_move_cell,
        "accuracy_companion_cell": accuracy_cell,
    }


def main() -> None:
    nfl, nfl_provenance, graded_meta = load_line_move_population()
    cfb_all = load_cfb_population()

    cfb_pred_p, per_fold_cfb_meta = attach_cfb_transfer_logit(nfl, cfb_all)
    nfl["cfb_pred_p"] = cfb_pred_p
    nfl = nfl.dropna(subset=["cfb_pred_p"]).reset_index(drop=True)
    clipped = np.clip(nfl["cfb_pred_p"], 1e-6, 1.0 - 1e-6)
    nfl["cfb_transfer_logit"] = np.log(clipped / (1.0 - clipped))

    nfl["base_oos_p"], base_fold_betas = loso_logit(
        nfl, list(BASE_FEATURES), "home_covered", "tuesday_base_line_move"
    )
    record_look("loso_base_tuesday_only")

    plus_features = [*BASE_FEATURES, "cfb_transfer_logit"]
    nfl["plus_oos_p"], plus_fold_betas = loso_logit(
        nfl, plus_features, "home_covered", "tuesday_base_plus_cfb_transfer_line_move"
    )
    record_look("loso_base_plus_cfb_transfer")

    nfl = nfl.loc[nfl["base_oos_p"].notna() & nfl["plus_oos_p"].notna()].reset_index(drop=True)

    nfl["base_pick_home"] = nfl["base_oos_p"].ge(0.5)
    nfl["plus_pick_home"] = nfl["plus_oos_p"].ge(0.5)
    nfl["lm_base"] = np.where(nfl["base_pick_home"], 1.0, -1.0) * nfl["open_move"]
    nfl["lm_plus"] = np.where(nfl["plus_pick_home"], 1.0, -1.0) * nfl["open_move"]
    nfl["diff_line_move"] = nfl["lm_plus"] - nfl["lm_base"]

    nfl["base_correct"] = (
        nfl["base_pick_home"].astype(float).eq(nfl["home_covered"]).astype(float)
    )
    nfl["plus_correct"] = (
        nfl["plus_pick_home"].astype(float).eq(nfl["home_covered"]).astype(float)
    )
    nfl["diff_accuracy"] = nfl["plus_correct"] - nfl["base_correct"]

    nfl_2020_2025 = nfl.loc[nfl["season"].ge(2020)].reset_index(drop=True)

    report_all = build_report(nfl, "cfb_transfer_logit_line_move_v4_all_graded")
    report_2020_2025 = build_report(
        nfl_2020_2025, "cfb_transfer_logit_line_move_v4_2020_2025_subset"
    )

    records = {
        "base_tuesday_only_all_graded": games_record(nfl["base_correct"]),
        "base_plus_cfb_transfer_all_graded": games_record(nfl["plus_correct"]),
        "base_tuesday_only_2020_2025": games_record(nfl_2020_2025["base_correct"]),
        "base_plus_cfb_transfer_2020_2025": games_record(nfl_2020_2025["plus_correct"]),
    }

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts/opener_error_transfer_unit4" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "unit": 4,
        "graded_meta": graded_meta,
        "cfb_train_seasons_available": list(CFB_TRAIN_SEASONS_ALL),
        "base_features": list(BASE_FEATURES),
        "plus_features": plus_features,
        "seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "nfl_games_all_graded": len(nfl),
        "nfl_games_2020_2025_subset": len(nfl_2020_2025),
        "records": records,
        "per_fold_cfb_meta": per_fold_cfb_meta,
        "base_loso_fold_betas": base_fold_betas,
        "plus_loso_fold_betas": plus_fold_betas,
        "cfb_transfer_logit_line_move_v4_all_graded": report_all,
        "cfb_transfer_logit_line_move_v4_2020_2025_subset": report_2020_2025,
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
            "open_move",
            "cfb_pred_p",
            "base_oos_p",
            "plus_oos_p",
            "diff_line_move",
            "diff_accuracy",
        ]
        if c in nfl.columns
    ]
    nfl[export_cols].to_csv(out_dir / "per_game.csv", index=False)

    print(f"artifact_dir={out_dir}")
    print("graded_meta", json.dumps(graded_meta))
    print("nfl_games_all_graded", len(nfl), "nfl_games_2020_2025", len(nfl_2020_2025))
    print("records", json.dumps(records))
    print(
        "line_move_all_graded",
        json.dumps(report_all["line_move_toward_pick_cell"]),
    )
    print(
        "line_move_2020_2025_subset",
        json.dumps(report_2020_2025["line_move_toward_pick_cell"]),
    )
    print("look_count", len(LOOKS))


if __name__ == "__main__":
    main()
