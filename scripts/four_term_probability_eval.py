from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import joint_probability_model_eval as jpm  # noqa: E402

BEST_PICK_WEEKLY_PATH = Path("artifacts/best_pick_served_score_ranker/20260914T163425Z/weekly.csv")

M5_FEATURES = ["x0_model_logit", "flag_sum", "move", "move_available"]
M5A_FEATURES = ["x0_model_logit", "flag_sum"]
M5B_FEATURES = ["flag_sum", "move", "move_available"]
M5C_FEATURES = ["x0_model_logit", "flag_sum_capped", "move", "move_available"]


def load_population() -> tuple[pd.DataFrame, dict[str, Any]]:
    per_game, opener_dir, matches_active = jpm.load_opener_population()
    schedules = jpm.load_schedules()
    incidents, arrest_snapshot_id = jpm.load_incidents()
    forecasts_full = pd.read_parquet("data/raw/forecast_archive/full_2020_2025/forecasts.parquet")[
        ["game_id", "forecast_temp_f"]
    ].copy()
    forecasts_kn = pd.read_parquet(
        "data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet"
    )[["game_id", "forecast_precip_prob_pct"]].copy()
    pbp_snapshot = jpm.latest_pbp_snapshot(Path("data"))
    if pbp_snapshot is None:
        raise FileNotFoundError("No PBP snapshot found under data/pbp/raw")

    base = jpm.build_base(per_game, schedules)
    push = base["margin_vs_open"].astype(float).eq(0.0)
    n_pushes = int(push.sum())
    df = base.loc[~push].reset_index(drop=True)

    p_raw = df["home_cover_probability_at_open_raw"].astype(float)
    df["x_p_raw"] = p_raw
    p_clipped = p_raw.clip(1e-6, 1.0 - 1e-6)
    df["x0_model_logit"] = np.log(p_clipped / (1.0 - p_clipped))
    df["home_covered"] = np.where(df["margin_vs_open"].astype(float).gt(0.0), 1.0, 0.0)
    df["m0_pick_home"] = p_raw.ge(0.5)
    df["m0_correct"] = df["m0_pick_home"].astype(float).eq(df["home_covered"]).astype(float)

    df, flag_coverage = jpm.add_flags(
        df, schedules, incidents, forecasts_full, forecasts_kn, pbp_snapshot
    )

    market = pd.read_parquet(jpm.MARKET_POPULATION_PATH)[["game_id", "leader_median_net"]]
    df = df.merge(market, on="game_id", how="left")
    df["move_raw"] = df["leader_median_net"].astype(float)
    df["move_available"] = df["move_raw"].notna().astype(float)
    df["move"] = df["move_raw"].fillna(0.0)

    p1, m1_detail = jpm.build_m1(
        df, schedules, incidents, forecasts_full, forecasts_kn, pbp_snapshot
    )
    df["m1_p"] = p1
    df["m1_pick_home"] = df["m1_p"].ge(0.5)
    df["m1_correct"] = df["m1_pick_home"].astype(float).eq(df["home_covered"]).astype(float)

    df["flag_sum"] = df[jpm.FLAG_COLUMNS].sum(axis=1).astype(float)
    df["flag_sum_capped"] = df["flag_sum"].clip(-1.0, 1.0)

    provenance = {
        "opener_evaluation_directory": str(opener_dir),
        "opener_evaluation_matches_active_model": matches_active,
        "arrest_snapshot_id": arrest_snapshot_id,
        "pbp_snapshot": str(pbp_snapshot),
        "n_games_2020_2025_before_push_drop": len(base),
        "n_pushes_dropped": n_pushes,
        "n_games_graded": len(df),
        "n_games_move_available_2023_2025": int(df["move_available"].sum()),
        "flag_coverage": flag_coverage,
        "m1_reproduction_detail": m1_detail,
    }
    return df, provenance


def add_loss_cols(frame: pd.DataFrame, p_col: str, prefix: str) -> None:
    pc = frame[p_col].astype(float).clip(1e-6, 1.0 - 1e-6)
    y = frame["home_covered"].astype(float)
    frame[f"{prefix}_logloss"] = -(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))
    frame[f"{prefix}_brier"] = (pc - y) ** 2


def fit_arm(df: pd.DataFrame, features: list[str], name: str, correct_col: str) -> dict[str, Any]:
    fit = jpm.loso_arm(df, features, name, select_penalty=False)
    df[f"{name}_oos_p"] = fit["oos_p"]
    df[f"{name}_pick_home"] = df[f"{name}_oos_p"].ge(0.5)
    df[correct_col] = df[f"{name}_pick_home"].astype(float).eq(df["home_covered"]).astype(float)
    return fit


def best_pick_star(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[["season", "week", "game_id"]].copy()
    tmp["confidence"] = np.maximum(df["m5_oos_p"].to_numpy(), 1.0 - df["m5_oos_p"].to_numpy())
    tmp["m5_star_correct"] = df["m5_correct"].to_numpy()
    tmp = tmp.sort_values(
        ["season", "week", "confidence", "game_id"], ascending=[True, True, False, True]
    )
    star = tmp.groupby(["season", "week"], as_index=False).first()
    return star.rename(columns={"game_id": "m5_star_game_id", "confidence": "m5_star_confidence"})[
        ["season", "week", "m5_star_game_id", "m5_star_confidence", "m5_star_correct"]
    ]


def decisive_week_record(
    pair: pd.DataFrame,
    star_game_col: str,
    other_game_col: str,
    star_correct_col: str,
    other_correct_col: str,
) -> dict[str, Any]:
    diff = pair[star_game_col].ne(pair[other_game_col])
    sub = pair.loc[diff]
    return {
        "n_decisive_weeks": int(diff.sum()),
        "n_paired_weeks": len(pair),
        "m5_star_record_on_decisive": jpm.games_record(sub[star_correct_col]),
        "other_record_on_decisive": jpm.games_record(sub[other_correct_col]),
    }


def main() -> None:
    df, provenance = load_population()

    m3 = jpm.loso_arm(df, jpm.M3_FEATURES, "m3", select_penalty=True)
    df["m3_oos_p"] = m3["oos_p"]
    df["m3_pick_home"] = df["m3_oos_p"].ge(0.5)
    df["m3_correct"] = df["m3_pick_home"].astype(float).eq(df["home_covered"]).astype(float)

    m5 = fit_arm(df, M5_FEATURES, "m5", "m5_correct")
    m5a = fit_arm(df, M5A_FEATURES, "m5a", "m5a_correct")
    m5b = fit_arm(df, M5B_FEATURES, "m5b", "m5b_correct")
    m5c = fit_arm(df, M5C_FEATURES, "m5c", "m5c_correct")

    add_loss_cols(df, "x_p_raw", "m0")
    add_loss_cols(df, "m1_p", "m1")
    add_loss_cols(df, "m3_oos_p", "m3")
    add_loss_cols(df, "m5_oos_p", "m5")
    add_loss_cols(df, "m5a_oos_p", "m5a")
    add_loss_cols(df, "m5b_oos_p", "m5b")
    add_loss_cols(df, "m5c_oos_p", "m5c")

    head_to_head: dict[str, Any] = {}

    def hth(
        name: str, cand_prefix: str, base_prefix: str, cand_correct: str, base_correct: str
    ) -> None:
        jpm.record_look(f"hth_logloss_{name}")
        ll = jpm.paired_effect_loss(df, f"{cand_prefix}_logloss", f"{base_prefix}_logloss")
        jpm.record_look(f"hth_brier_{name}")
        br = jpm.paired_effect_loss(df, f"{cand_prefix}_brier", f"{base_prefix}_brier")
        jpm.record_look(f"hth_accuracy_{name}")
        acc = jpm.paired_effect_accuracy(df, cand_correct, base_correct)
        head_to_head[name] = {"log_loss": ll, "brier": br, "accuracy_points": acc}

    for cand_name, cand_correct in (
        ("m5", "m5_correct"),
        ("m5a", "m5a_correct"),
        ("m5b", "m5b_correct"),
        ("m5c", "m5c_correct"),
    ):
        hth(f"{cand_name}_vs_m0", cand_name, "m0", cand_correct, "m0_correct")
        hth(f"{cand_name}_vs_m1", cand_name, "m1", cand_correct, "m1_correct")
        hth(f"{cand_name}_vs_m3", cand_name, "m3", cand_correct, "m3_correct")

    calibration = {"m5": jpm.calibration_table(df, "m5_oos_p")}

    decisive = {
        "m5_vs_m0": jpm.decisive_report(
            df, "m5_pick_home", "m5_correct", "m0_pick_home", "m0_correct"
        ),
        "m5_vs_m1": jpm.decisive_report(
            df, "m5_pick_home", "m5_correct", "m1_pick_home", "m1_correct"
        ),
        "m5a_vs_m0": jpm.decisive_report(
            df, "m5a_pick_home", "m5a_correct", "m0_pick_home", "m0_correct"
        ),
        "m5b_vs_m0": jpm.decisive_report(
            df, "m5b_pick_home", "m5b_correct", "m0_pick_home", "m0_correct"
        ),
        "m5c_vs_m0": jpm.decisive_report(
            df, "m5c_pick_home", "m5c_correct", "m0_pick_home", "m0_correct"
        ),
    }

    records = {
        "m0": jpm.games_record(df["m0_correct"]),
        "m1": jpm.games_record(df["m1_correct"]),
        "m3_oos": jpm.games_record(df["m3_correct"]),
        "m5_oos": jpm.games_record(df["m5_correct"]),
        "m5a_oos": jpm.games_record(df["m5a_correct"]),
        "m5b_oos": jpm.games_record(df["m5b_correct"]),
        "m5c_oos": jpm.games_record(df["m5c_correct"]),
    }

    diff_mask = df["m5_pick_home"].ne(df["m1_pick_home"])
    df["foresight_on_m5_vs_m1_diff"] = np.where(
        diff_mask & df["m1_correct"].eq(0.0), 1.0, df["m1_correct"]
    )
    jpm.record_look("positive_control_foresight_on_m5_vs_m1_diff_games")
    positive_control = jpm.paired_effect_accuracy(df, "foresight_on_m5_vs_m1_diff", "m1_correct")
    positive_control_detail = {
        "n_m5_vs_m1_diff_games": int(diff_mask.sum()),
        "m1_record_on_diff": jpm.games_record(df.loc[diff_mask, "m1_correct"]),
        "foresight_record_on_diff": jpm.games_record(
            df.loc[diff_mask, "foresight_on_m5_vs_m1_diff"]
        ),
        "effect_vs_m1": positive_control,
    }

    season_split = jpm.season_split(
        df,
        {
            "m0_accuracy": "m0_correct",
            "m1_accuracy": "m1_correct",
            "m3_oos_accuracy": "m3_correct",
            "m5_oos_accuracy": "m5_correct",
            "m5a_oos_accuracy": "m5a_correct",
            "m5b_oos_accuracy": "m5b_correct",
            "m5c_oos_accuracy": "m5c_correct",
        },
    )

    star = best_pick_star(df)
    weekly = pd.read_csv(BEST_PICK_WEEKLY_PATH)
    merged = star.merge(
        weekly[
            [
                "season",
                "week",
                "a0_incumbent_game_id",
                "a0_incumbent_correct",
                "a1_served_score_game_id",
                "a1_served_score_correct",
            ]
        ],
        on=["season", "week"],
        how="left",
    )

    pair_a0 = merged.dropna(subset=["a0_incumbent_correct"]).copy()
    jpm.record_look("best_pick_m5_star_vs_a0_incumbent")
    bp_vs_a0 = jpm.paired_effect_accuracy(pair_a0, "m5_star_correct", "a0_incumbent_correct")

    pair_a1 = merged.dropna(subset=["a1_served_score_correct"]).copy()
    jpm.record_look("best_pick_m5_star_vs_a1_served_score")
    bp_vs_a1 = jpm.paired_effect_accuracy(pair_a1, "m5_star_correct", "a1_served_score_correct")

    decisive_a0 = decisive_week_record(
        pair_a0,
        "m5_star_game_id",
        "a0_incumbent_game_id",
        "m5_star_correct",
        "a0_incumbent_correct",
    )
    decisive_a1 = decisive_week_record(
        pair_a1,
        "m5_star_game_id",
        "a1_served_score_game_id",
        "m5_star_correct",
        "a1_served_score_correct",
    )

    best_pick_summary = {
        "n_weeks_total": len(merged),
        "n_weeks_unmatched_to_weekly_csv": int(
            merged["a1_served_score_game_id"].isna().sum()
            + merged["a0_incumbent_game_id"].isna().sum()
        ),
        "m5_star_hit_rate_all_weeks": float(star["m5_star_correct"].mean()),
        "vs_a0_incumbent": {
            "n_paired_weeks": len(pair_a0),
            "m5_star_hit_rate": float(pair_a0["m5_star_correct"].mean()),
            "a0_hit_rate": float(pair_a0["a0_incumbent_correct"].mean()),
            "effect_accuracy_points": bp_vs_a0,
            "decisive_weeks": decisive_a0,
        },
        "vs_a1_served_score": {
            "n_paired_weeks": len(pair_a1),
            "m5_star_hit_rate": float(pair_a1["m5_star_correct"].mean()),
            "a1_hit_rate": float(pair_a1["a1_served_score_correct"].mean()),
            "effect_accuracy_points": bp_vs_a1,
            "decisive_weeks": decisive_a1,
        },
    }

    registry_wrong_sign: list[str] = []
    for name, cell in head_to_head.items():
        for metric_name, effect in cell.items():
            if jpm.resolved_wrong_sign(effect):
                registry_wrong_sign.append(f"{name}:{metric_name}")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("artifacts/four_term_probability") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        **provenance,
        "seed": jpm.SEED,
        "samples": jpm.SAMPLES,
        "fixed_l2": 1e-3,
        "m5_features": M5_FEATURES,
        "m5a_features": M5A_FEATURES,
        "m5b_features": M5B_FEATURES,
        "m5c_features": M5C_FEATURES,
        "records": records,
        "head_to_head": head_to_head,
        "calibration": calibration,
        "decisive": decisive,
        "positive_control_foresight_on_m5_vs_m1_diff": positive_control_detail,
        "season_split": season_split,
        "best_pick_weekly_source": str(BEST_PICK_WEEKLY_PATH),
        "best_pick": best_pick_summary,
        "m3_fold_coefficients": m3["fold_coefficients"],
        "m5_fold_coefficients": m5["fold_coefficients"],
        "m5a_fold_coefficients": m5a["fold_coefficients"],
        "m5b_fold_coefficients": m5b["fold_coefficients"],
        "m5c_fold_coefficients": m5c["fold_coefficients"],
        "registry_resolved_wrong_sign_cells": registry_wrong_sign,
        "look_count": len(jpm.LOOKS),
        "look_log": jpm.LOOKS,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    coefficient_rows = []
    for arm_name, fit in (("m3", m3), ("m5", m5), ("m5a", m5a), ("m5b", m5b), ("m5c", m5c)):
        for season_key, coefs in fit["fold_coefficients"].items():
            for covariate, value in coefs.items():
                if not isinstance(value, dict):
                    continue
                coefficient_rows.append(
                    {
                        "arm": arm_name,
                        "held_out_season": season_key,
                        "covariate": covariate,
                        "coef": value["coef"],
                        "se": value["se"],
                        "ci_low": value["ci_low"],
                        "ci_high": value["ci_high"],
                    }
                )
    pd.DataFrame(coefficient_rows).to_csv(out_dir / "coefficients.csv", index=False)

    export_cols = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "x_p_raw",
        "home_covered",
        "flag_sum",
        "flag_sum_capped",
        "move",
        "move_available",
        "m0_pick_home",
        "m0_correct",
        "m1_pick_home",
        "m1_correct",
        "m3_pick_home",
        "m3_correct",
        "m5_oos_p",
        "m5_pick_home",
        "m5_correct",
        "m5a_pick_home",
        "m5a_correct",
        "m5b_pick_home",
        "m5b_correct",
        "m5c_pick_home",
        "m5c_correct",
    ]
    df[export_cols].to_csv(out_dir / "per_game.csv", index=False)
    merged.to_csv(out_dir / "best_pick_weekly.csv", index=False)

    print(f"artifact_dir={out_dir}")
    print("records", json.dumps(records))
    print(
        "n_pushes_dropped",
        provenance["n_pushes_dropped"],
        "n_games_graded",
        provenance["n_games_graded"],
    )
    print("look_count", len(jpm.LOOKS))
    print("head_to_head", json.dumps(head_to_head, indent=2))
    print("decisive", json.dumps(decisive, indent=2))
    print("positive_control", json.dumps(positive_control_detail, indent=2))
    print("best_pick", json.dumps(best_pick_summary, indent=2))
    print(
        "m5_fold_coefficients",
        json.dumps(
            {
                k: {
                    kk: v.get(kk) for kk in ("x0_model_logit", "flag_sum", "move", "move_available")
                }
                for k, v in m5["fold_coefficients"].items()
            },
            indent=2,
        ),
    )
    print("registry_resolved_wrong_sign_cells", registry_wrong_sign)


if __name__ == "__main__":
    main()
