from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.bye_edge_fade_overlay import apply_bye_edge_fade_overlay, bye_edge_flag_by_game
from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.coach_fade_overlay import OVERLAY_WEEK_MAX as COACH_WEEK_MAX
from nfl_ats.coach_fade_overlay import apply_coach_fade_overlay, year_one_by_game
from nfl_ats.division_revenge_tilt_overlay import (
    apply_division_revenge_tilt_overlay,
    division_revenge_side_by_game,
)
from nfl_ats.forecast_cold_visitor_tilt_overlay import (
    apply_forecast_cold_visitor_tilt_overlay,
    forecast_cold_visitor_flag_by_game,
)
from nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay import (
    precip_high_total_flag_by_game,
)
from nfl_ats.four_overlay_composition import OWNER_HELD_MEMBERS
from nfl_ats.interim_hc_first_game_tilt_overlay import interim_first_game_flag_by_game_fail_open
from nfl_ats.pbp08_matchup_flags import build_flag_table
from nfl_ats.pbp08_protection_mismatch_tilt_overlay import (
    apply_pbp08_protection_mismatch_tilt,
    latest_pbp_snapshot,
)
from nfl_ats.player_arrests_back_side_overlay import (
    _broad_side_flags,
    apply_player_arrests_back_side_overlay,
)
from nfl_ats.snapshots import latest_snapshot, load_snapshot
from nfl_ats.tank_zone_fade_tilt_overlay import OVERLAY_WEEK_MAX as TANK_WEEK_MAX
from nfl_ats.tank_zone_fade_tilt_overlay import OVERLAY_WEEK_MIN as TANK_WEEK_MIN
from nfl_ats.tank_zone_fade_tilt_overlay import (
    apply_tank_zone_fade_tilt_overlay,
    tank_zone_flag_by_game,
)

REPO_ROOT = Path(".")
SEASON_START = 2020
SEASON_END = 2025
MARKET_POPULATION_PATH = Path("artifacts/sharp_weighted_follow/20260909T233606Z/per_game.parquet")
SEED = 20260914
SAMPLES = 20000
CANDIDATE_L2 = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
FIXED_L2_M4 = 1e-3
CALIBRATION_BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
Z95 = 1.959963985

FLAG_COLUMNS = [
    "flag_coach",
    "flag_division",
    "flag_arrests",
    "flag_bye",
    "flag_cold_visitor",
    "flag_protection",
    "flag_interim_hc",
    "flag_tank_zone",
    "flag_precip",
]
M4_FEATURES = ["x0_model_logit"]
M3_FEATURES = ["x0_model_logit", *FLAG_COLUMNS]
M2_FEATURES = ["x0_model_logit", *FLAG_COLUMNS, "move"]

LOOKS: list[str] = []


def record_look(label: str) -> None:
    LOOKS.append(label)


def newest_opener_evaluation_dir() -> Path:
    root = Path("artifacts/opener_evaluation")
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "per_game.parquet").exists()),
        reverse=True,
    )
    return candidates[0]


def load_active_model_feature_sha() -> str:
    manifest = load_active_ats_model(Path("artifacts"))
    return str(manifest.get("feature_table_sha256")) if manifest else ""


def load_opener_population() -> tuple[pd.DataFrame, Path, bool]:
    directory = newest_opener_evaluation_dir()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    matches_active = str(metadata.get("feature_table_sha256")) == load_active_model_feature_sha()
    per_game = pd.read_parquet(directory / "per_game.parquet")
    return per_game, directory, matches_active


def load_schedules() -> pd.DataFrame:
    snapshot = latest_snapshot(Path("data/raw"))
    schedules, _team_stats = load_snapshot(snapshot)
    return schedules


def load_incidents() -> tuple[pd.DataFrame, str]:
    root = Path("data/raw/player_arrests")
    directories = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    newest = directories[0]
    incidents = pd.read_parquet(newest / "incidents_point_in_time.parquet")
    return incidents, newest.name


def build_base(per_game: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    graded = per_game.loc[per_game.season.between(SEASON_START, SEASON_END)].reset_index(drop=True)
    sched_cols = schedules[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "gameday",
            "home_team",
            "away_team",
            "total_line",
        ]
    ].drop_duplicates("game_id")
    base = graded.merge(
        sched_cols,
        on=["game_id", "season"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_sched"),
    )
    if base["home_team"].isna().any():
        missing = base.loc[base["home_team"].isna(), "game_id"].tolist()
        raise ValueError(f"{len(missing)} opener-evaluation games have no matching schedule row")
    if (base["game_type"] != "REG").any():
        raise ValueError("opener-evaluation population contains non-REG games")
    if (base["week"] != base["week_sched"]).any():
        raise ValueError("opener-evaluation week disagrees with the schedules snapshot")
    base = base.drop(columns=["week_sched"])
    return base


def add_flags(
    base: pd.DataFrame,
    schedules: pd.DataFrame,
    incidents: pd.DataFrame,
    forecasts_full: pd.DataFrame,
    forecasts_kn: pd.DataFrame,
    pbp_snapshot: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    coverage: dict[str, Any] = {}
    frame = base.copy()

    coach = year_one_by_game(schedules)
    merged = frame.merge(coach, on=["game_id", "season"], how="left")
    year_one_home = merged["year_one_home"].fillna(False)
    year_one_away = merged["year_one_away"].fillna(False)
    eligible_coach = merged["week"].astype(int).le(COACH_WEEK_MAX) & merged["game_type"].eq("REG")
    frame["flag_coach"] = np.where(
        eligible_coach & year_one_away & ~year_one_home,
        1,
        np.where(eligible_coach & year_one_home & ~year_one_away, -1, 0),
    )
    coverage["coach"] = {
        "eligible_games": int(eligible_coach.sum()),
        "both_flagged_games": int((eligible_coach & year_one_home & year_one_away).sum()),
    }

    revenge = division_revenge_side_by_game(schedules)
    merged = frame.merge(revenge, on=["game_id", "season"], how="left")
    revenge_home = merged["revenge_home"].fillna(False)
    revenge_away = merged["revenge_away"].fillna(False)
    frame["flag_division"] = np.where(
        revenge_home & ~revenge_away, 1, np.where(revenge_away & ~revenge_home, -1, 0)
    )
    coverage["division_revenge"] = {
        "both_flagged_games": int((revenge_home & revenge_away).sum()),
        "flagged_games": int((revenge_home | revenge_away).sum()),
    }

    predictions_for_arrest = frame[["game_id", "gameday", "home_team", "away_team"]].copy()
    home_flags, away_flags = _broad_side_flags(predictions_for_arrest, incidents)
    frame["flag_arrests"] = np.where(
        home_flags & ~away_flags, 1, np.where(away_flags & ~home_flags, -1, 0)
    )
    coverage["arrests"] = {
        "incidents_available": len(incidents),
        "flagged_games": int((home_flags | away_flags).sum()),
    }

    bye = bye_edge_flag_by_game(schedules)
    merged = frame.merge(bye, on=["game_id", "season"], how="left")
    home_off_bye = merged["home_off_bye"].fillna(False)
    away_off_bye = merged["away_off_bye"].fillna(False)
    frame["flag_bye"] = np.where(
        away_off_bye & ~home_off_bye, 1, np.where(home_off_bye & ~away_off_bye, -1, 0)
    )
    coverage["bye_edge"] = {"flagged_games": int((home_off_bye | away_off_bye).sum())}

    cold = forecast_cold_visitor_flag_by_game(schedules, forecasts_full)
    merged = frame.merge(cold, on="game_id", how="left")
    cold_flag = merged["forecast_cold_visitor_flag"].fillna(False)
    frame["flag_cold_visitor"] = np.where(cold_flag, 1, 0)
    coverage["cold_visitor"] = {
        "climate_missing": int(merged["climate_temp"].isna().sum()),
        "forecast_missing": int(merged["forecast_temp_f"].isna().sum()),
        "flagged_games": int(cold_flag.sum()),
    }

    sched_for_flags = schedules.loc[
        schedules["game_type"].eq("REG") & schedules["season"].between(2009, SEASON_END)
    ].copy()
    flag_table = build_flag_table(sched_for_flags, pbp_snapshot)
    merged = frame.merge(flag_table[["game_id", "back_side"]], on="game_id", how="left")
    back_side = merged["back_side"].fillna("")
    frame["flag_protection"] = np.where(
        back_side == "HOME", 1, np.where(back_side == "AWAY", -1, 0)
    )
    coverage["protection_mismatch"] = {
        "flagged_games": int(back_side.ne("").sum()),
        "pbp_snapshot": str(pbp_snapshot),
    }

    interim = interim_first_game_flag_by_game_fail_open(REPO_ROOT)
    home_interim = interim.rename(columns={"team": "home_team"})[["game_id", "home_team"]].copy()
    home_interim["home_first_game"] = True
    away_interim = interim.rename(columns={"team": "away_team"})[["game_id", "away_team"]].copy()
    away_interim["away_first_game"] = True
    merged = frame.merge(home_interim, on=["game_id", "home_team"], how="left")
    merged = merged.merge(away_interim, on=["game_id", "away_team"], how="left")
    home_first = merged["home_first_game"].fillna(False)
    away_first = merged["away_first_game"].fillna(False)
    frame["flag_interim_hc"] = np.where(
        home_first & ~away_first, 1, np.where(away_first & ~home_first, -1, 0)
    )
    coverage["interim_hc"] = {
        "entries_total": len(interim),
        "flagged_games": int((home_first | away_first).sum()),
    }

    tank = tank_zone_flag_by_game(schedules)
    merged = frame.merge(tank, on=["game_id", "season"], how="left")
    tank_home = merged["tank_zone_home"].fillna(False)
    tank_away = merged["tank_zone_away"].fillna(False)
    eligible_tank = merged["week"].astype(int).between(TANK_WEEK_MIN, TANK_WEEK_MAX) & merged[
        "game_type"
    ].eq("REG")
    frame["flag_tank_zone"] = np.where(
        eligible_tank & tank_away & ~tank_home,
        1,
        np.where(eligible_tank & tank_home & ~tank_away, -1, 0),
    )
    coverage["tank_zone"] = {"eligible_games": int(eligible_tank.sum())}

    total_lines = frame[["game_id", "total_line"]].copy()
    precip = precip_high_total_flag_by_game(schedules, forecasts_kn, total_lines)
    merged = frame.merge(precip, on="game_id", how="left")
    precip_flag = merged["precip_high_total_flag"].fillna(False)
    frame["flag_precip"] = np.where(precip_flag, 1, 0)
    coverage["precip_high_total"] = {"flagged_games": int(precip_flag.sum())}

    return frame, coverage


def build_apply_frame(frame: pd.DataFrame, probability: np.ndarray) -> pd.DataFrame:
    apply_frame = frame[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "gameday",
            "home_team",
            "away_team",
            "total_line",
        ]
    ].copy()
    apply_frame["home_cover_probability"] = probability
    return apply_frame


def build_m1(
    frame: pd.DataFrame,
    schedules: pd.DataFrame,
    incidents: pd.DataFrame,
    forecasts_full: pd.DataFrame,
    forecasts_kn: pd.DataFrame,
    pbp_snapshot: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    p_raw = frame["x_p_raw"].to_numpy(dtype=float)
    apply_frame = build_apply_frame(frame, p_raw)

    assert "interim_hc_first_game_tilt" in OWNER_HELD_MEMBERS
    assert "precip_high_total_tilt" in OWNER_HELD_MEMBERS

    coach_res = apply_coach_fade_overlay(apply_frame, schedules)
    division_res = apply_division_revenge_tilt_overlay(apply_frame, schedules)
    arrest_res = apply_player_arrests_back_side_overlay(apply_frame, incidents)
    bye_res = apply_bye_edge_fade_overlay(apply_frame, schedules)
    cold_res = apply_forecast_cold_visitor_tilt_overlay(apply_frame, schedules, forecasts_full)
    sched_for_flags = schedules.loc[
        schedules["game_type"].eq("REG") & schedules["season"].between(2009, SEASON_END)
    ].copy()
    flag_table = build_flag_table(sched_for_flags, pbp_snapshot)
    protection_res = apply_pbp08_protection_mismatch_tilt(apply_frame, flag_table)
    tank_res = apply_tank_zone_fade_tilt_overlay(apply_frame, schedules)

    member_flip_ids = {
        "coach_fade": {flip.game_id for flip in coach_res.flips},
        "division_revenge_tilt": {flip.game_id for flip in division_res.flips},
        "player_arrests_back_side_policy": {flip.game_id for flip in arrest_res.flips},
        "bye_edge_fade": {flip.game_id for flip in bye_res.flips},
        "forecast_cold_visitor_tilt": {flip.game_id for flip in cold_res.flips},
        "pbp08_protection_mismatch_tilt": {flip.game_id for flip in protection_res.flips},
        "tank_zone_fade_tilt": {flip.game_id for flip in tank_res.flips},
    }
    union_ids: set[str] = set()
    for ids in member_flip_ids.values():
        union_ids |= ids
    union_mask = frame["game_id"].astype(str).isin(union_ids)

    raw_pick_dir = np.where(p_raw >= 0.5, 1, -1)
    non_held_flags = [c for c in FLAG_COLUMNS if c not in ("flag_interim_hc", "flag_precip")]
    signal = frame[non_held_flags].to_numpy(dtype=int)
    disagreement = (signal * raw_pick_dir[:, None]) < 0
    shortcut_union_mask = disagreement.any(axis=1)

    mismatch = int((union_mask.to_numpy() != shortcut_union_mask).sum())

    p1 = np.where(union_mask.to_numpy(), 1.0 - p_raw, p_raw)
    detail = {
        "member_flip_counts": {name: len(ids) for name, ids in member_flip_ids.items()},
        "union_flip_count": len(union_ids),
        "signed_flag_shortcut_flip_count": int(shortcut_union_mask.sum()),
        "mismatch_between_apply_functions_and_signed_flag_shortcut": mismatch,
        "owner_held_members_excluded": sorted(OWNER_HELD_MEMBERS),
    }
    return p1, detail


def fit_logit(
    x: np.ndarray, y: np.ndarray, l2: float, iters: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    beta = np.zeros(x.shape[1])
    hessian = np.eye(x.shape[1])
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
    return beta, hessian


def design_matrix(
    frame: pd.DataFrame, feature_cols: list[str], means: dict[str, float], stds: dict[str, float]
) -> np.ndarray:
    cols = [np.ones(len(frame))]
    for c in feature_cols:
        cols.append(((frame[c].astype(float) - means[c]) / stds[c]).to_numpy())
    return np.column_stack(cols)


def standardize_fit(
    train: pd.DataFrame, feature_cols: list[str], target_col: str, l2: float
) -> tuple[np.ndarray, dict[str, float], dict[str, float], np.ndarray]:
    means = {c: float(train[c].mean()) for c in feature_cols}
    stds = {c: float(train[c].std(ddof=0)) or 1.0 for c in feature_cols}
    x = design_matrix(train, feature_cols, means, stds)
    y = train[target_col].astype(float).to_numpy()
    beta, hessian = fit_logit(x, y, l2)
    return beta, means, stds, hessian


def predict_p(
    frame: pd.DataFrame,
    feature_cols: list[str],
    beta: np.ndarray,
    means: dict[str, float],
    stds: dict[str, float],
) -> np.ndarray:
    x = design_matrix(frame, feature_cols, means, stds)
    z = np.clip(x @ beta, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def natural_coefficients(
    beta: np.ndarray,
    hessian: np.ndarray,
    feature_cols: list[str],
    means: dict[str, float],
    stds: dict[str, float],
) -> dict[str, Any]:
    cov = np.linalg.inv(hessian)
    se_std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    out: dict[str, Any] = {}
    for i, c in enumerate(feature_cols):
        coef = float(beta[i + 1] / stds[c])
        se = float(se_std[i + 1] / stds[c])
        out[c] = {
            "coef": coef,
            "se": se,
            "ci_low": coef - Z95 * se,
            "ci_high": coef + Z95 * se,
        }
    intercept = float(beta[0])
    for i, c in enumerate(feature_cols):
        intercept -= float(beta[i + 1]) * means[c] / stds[c]
    intercept_se = float(se_std[0])
    out["intercept"] = {
        "coef": intercept,
        "se": intercept_se,
        "ci_low": intercept - Z95 * intercept_se,
        "ci_high": intercept + Z95 * intercept_se,
    }
    return out


def inner_loso_log_loss(train: pd.DataFrame, feature_cols: list[str], l2: float) -> float:
    seasons = sorted(train.season.unique().tolist())
    total_loss = 0.0
    total_n = 0
    for held in seasons:
        inner_train = train.loc[train.season.ne(held)]
        inner_test = train.loc[train.season.eq(held)]
        if inner_train.empty or inner_test.empty:
            continue
        beta, means, stds, _hessian = standardize_fit(inner_train, feature_cols, "home_covered", l2)
        p = predict_p(inner_test, feature_cols, beta, means, stds)
        y = inner_test["home_covered"].to_numpy(dtype=float)
        pc = np.clip(p, 1e-6, 1.0 - 1e-6)
        total_loss += float(-(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)).sum())
        total_n += len(inner_test)
    return total_loss / total_n if total_n else float("inf")


def select_l2(train: pd.DataFrame, feature_cols: list[str]) -> tuple[float, float]:
    best_l2 = CANDIDATE_L2[0]
    best_loss = float("inf")
    for l2 in CANDIDATE_L2:
        loss = inner_loso_log_loss(train, feature_cols, l2)
        if loss < best_loss:
            best_loss = loss
            best_l2 = l2
    return best_l2, best_loss


def loso_arm(
    df: pd.DataFrame,
    feature_cols: list[str],
    arm_name: str,
    *,
    select_penalty: bool,
    fixed_l2: float = 1e-3,
) -> dict[str, Any]:
    seasons = sorted(df.season.unique().tolist())
    fold_coefficients: dict[str, Any] = {}
    oos_p = pd.Series(index=df.index, dtype=float)
    for held in seasons:
        train = df.loc[df.season.ne(held)]
        test = df.loc[df.season.eq(held)]
        if select_penalty:
            chosen_l2, inner_loss = select_l2(train, feature_cols)
        else:
            chosen_l2, inner_loss = fixed_l2, None
        beta, means, stds, hessian = standardize_fit(train, feature_cols, "home_covered", chosen_l2)
        oos_p.loc[test.index] = predict_p(test, feature_cols, beta, means, stds)
        coefs = natural_coefficients(beta, hessian, feature_cols, means, stds)
        coefs["n_train"] = len(train)
        coefs["n_test"] = len(test)
        coefs["chosen_l2"] = chosen_l2
        if inner_loss is not None:
            coefs["inner_loso_log_loss"] = inner_loss
        fold_coefficients[str(held)] = coefs
    record_look(f"loso_{arm_name}_fit_{len(seasons)}_folds")
    return {
        "feature_cols": feature_cols,
        "fold_coefficients": fold_coefficients,
        "oos_p": oos_p,
    }


def games_record(correct: pd.Series) -> str:
    wins = int(correct.sum())
    losses = int(len(correct) - wins)
    return f"{wins}-{losses}"


def paired_effect_accuracy(
    df: pd.DataFrame, candidate_col: str, baseline_col: str
) -> dict[str, float]:
    def metric_fn(sub: pd.DataFrame) -> dict[str, float]:
        return {
            "effect": float(
                (sub[candidate_col].astype(float).mean() - sub[baseline_col].astype(float).mean())
                * 100.0
            )
        }

    result = week_blocked_bootstrap(df, metric_fn, block="week", samples=SAMPLES, seed=SEED)
    row = result.iloc[0]
    return {
        "effect": float(row["estimate"]),
        "interval_low": float(row["lower"]),
        "interval_high": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
    }


def paired_effect_loss(df: pd.DataFrame, candidate_col: str, baseline_col: str) -> dict[str, float]:
    def metric_fn(sub: pd.DataFrame) -> dict[str, float]:
        return {
            "effect": float(
                sub[baseline_col].astype(float).mean() - sub[candidate_col].astype(float).mean()
            )
        }

    result = week_blocked_bootstrap(df, metric_fn, block="week", samples=SAMPLES, seed=SEED)
    row = result.iloc[0]
    return {
        "effect": float(row["estimate"]),
        "interval_low": float(row["lower"]),
        "interval_high": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
    }


def resolved_wrong_sign(effect: dict[str, float]) -> bool:
    return effect["interval_high"] < 0.0


def calibration_table(
    df: pd.DataFrame, p_col: str, outcome_col: str = "home_covered"
) -> list[dict[str, Any]]:
    rows = []
    for lo, hi in pairwise(CALIBRATION_BIN_EDGES):
        mask = df[p_col].ge(lo) & df[p_col].lt(hi if hi < 1.0 else 1.0 + 1e-9)
        n = int(mask.sum())
        mean_p = float(df.loc[mask, p_col].mean()) if n else float("nan")
        actual_rate = float(df.loc[mask, outcome_col].mean()) if n else float("nan")
        rows.append(
            {
                "bin": f"{lo:.1f}_to_{hi:.1f}",
                "n": n,
                "mean_predicted_p_home": mean_p,
                "actual_home_cover_rate": actual_rate,
                "gap": (actual_rate - mean_p) if n else float("nan"),
            }
        )
    record_look(f"calibration_table_{p_col}")
    return rows


def decisive_report(
    df: pd.DataFrame,
    arm_pick_col: str,
    arm_correct_col: str,
    base_pick_col: str,
    base_correct_col: str,
) -> dict[str, Any]:
    diff = df[arm_pick_col].ne(df[base_pick_col])
    n = int(diff.sum())
    base_rec = games_record(df.loc[diff, base_correct_col])
    arm_rec = games_record(df.loc[diff, arm_correct_col])
    record_look(f"decisive_report_{arm_pick_col}_vs_{base_pick_col}")
    return {"n_diff": n, "base_record_on_diff": base_rec, "arm_record_on_diff": arm_rec}


def season_split(df: pd.DataFrame, cols: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for season, sub in df.groupby("season"):
        row: dict[str, Any] = {"season": int(season), "n_games": len(sub)}
        for label, col in cols.items():
            row[label] = float(sub[col].mean())
        rows.append(row)
    record_look("season_split_descriptive")
    return rows


def main() -> None:
    per_game, opener_dir, matches_active = load_opener_population()
    schedules = load_schedules()
    incidents, arrest_snapshot_id = load_incidents()
    forecasts_full = pd.read_parquet("data/raw/forecast_archive/full_2020_2025/forecasts.parquet")[
        ["game_id", "forecast_temp_f"]
    ].copy()
    forecasts_kn = pd.read_parquet(
        "data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet"
    )[["game_id", "forecast_precip_prob_pct"]].copy()
    pbp_snapshot = latest_pbp_snapshot(Path("data"))
    if pbp_snapshot is None:
        raise FileNotFoundError("No PBP snapshot found under data/pbp/raw")

    base = build_base(per_game, schedules)

    push = base["margin_vs_open"].astype(float).eq(0.0)
    n_pushes = int(push.sum())
    df = base.loc[~push].reset_index(drop=True)

    p_raw = df["home_cover_probability_at_open_raw"].astype(float)
    p_served = df["home_cover_probability_at_open"].astype(float)
    df["x_p_raw"] = p_raw
    p_clipped = p_raw.clip(1e-6, 1.0 - 1e-6)
    df["x0_model_logit"] = np.log(p_clipped / (1.0 - p_clipped))
    home_covered = np.where(df["margin_vs_open"].astype(float).gt(0.0), 1.0, 0.0)
    df["home_covered"] = home_covered
    df["m0_pick_home"] = p_raw.ge(0.5)
    df["m0_correct"] = df["m0_pick_home"].astype(float).eq(df["home_covered"]).astype(float)
    n_raw_vs_served_diff = int(p_raw.ne(p_served).sum())

    df, flag_coverage = add_flags(
        df, schedules, incidents, forecasts_full, forecasts_kn, pbp_snapshot
    )

    market = pd.read_parquet(MARKET_POPULATION_PATH)[["game_id", "leader_median_net"]]
    df = df.merge(market, on="game_id", how="left")
    df["move"] = df["leader_median_net"].astype(float)
    df["move_available"] = df["move"].notna()

    p1, m1_detail = build_m1(df, schedules, incidents, forecasts_full, forecasts_kn, pbp_snapshot)
    df["m1_p"] = p1
    df["m1_pick_home"] = df["m1_p"].ge(0.5)
    df["m1_correct"] = df["m1_pick_home"].astype(float).eq(df["home_covered"]).astype(float)

    m4 = loso_arm(df, M4_FEATURES, "m4", select_penalty=False, fixed_l2=FIXED_L2_M4)
    df["m4_oos_p"] = m4["oos_p"]
    df["m4_pick_home"] = df["m4_oos_p"].ge(0.5)
    df["m4_correct"] = df["m4_pick_home"].astype(float).eq(df["home_covered"]).astype(float)

    m3 = loso_arm(df, M3_FEATURES, "m3", select_penalty=True)
    df["m3_oos_p"] = m3["oos_p"]
    df["m3_pick_home"] = df["m3_oos_p"].ge(0.5)
    df["m3_correct"] = df["m3_pick_home"].astype(float).eq(df["home_covered"]).astype(float)

    pop2 = df.loc[df["move_available"]].copy()
    m2 = loso_arm(pop2, M2_FEATURES, "m2", select_penalty=True)
    pop2["m2_oos_p"] = m2["oos_p"]
    pop2["m2_pick_home"] = pop2["m2_oos_p"].ge(0.5)
    pop2["m2_correct"] = pop2["m2_pick_home"].astype(float).eq(pop2["home_covered"]).astype(float)

    def add_loss_cols(frame: pd.DataFrame, p_col: str, prefix: str) -> None:
        pc = frame[p_col].astype(float).clip(1e-6, 1.0 - 1e-6)
        y = frame["home_covered"].astype(float)
        frame[f"{prefix}_logloss"] = -(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))
        frame[f"{prefix}_brier"] = (pc - y) ** 2

    add_loss_cols(df, "x_p_raw", "m0")
    add_loss_cols(df, "m1_p", "m1")
    add_loss_cols(df, "m3_oos_p", "m3")
    add_loss_cols(df, "m4_oos_p", "m4")
    add_loss_cols(pop2, "x_p_raw", "m0")
    add_loss_cols(pop2, "m1_p", "m1")
    add_loss_cols(pop2, "m2_oos_p", "m2")

    head_to_head: dict[str, Any] = {}

    def hth(
        name: str,
        pop: pd.DataFrame,
        cand_prefix: str,
        base_prefix: str,
        cand_correct: str,
        base_correct: str,
    ) -> None:
        record_look(f"hth_logloss_{name}")
        ll = paired_effect_loss(pop, f"{cand_prefix}_logloss", f"{base_prefix}_logloss")
        record_look(f"hth_brier_{name}")
        br = paired_effect_loss(pop, f"{cand_prefix}_brier", f"{base_prefix}_brier")
        record_look(f"hth_accuracy_{name}")
        acc = paired_effect_accuracy(pop, cand_correct, base_correct)
        head_to_head[name] = {"log_loss": ll, "brier": br, "accuracy_points": acc}

    hth("m1_vs_m0", df, "m1", "m0", "m1_correct", "m0_correct")
    hth("m3_vs_m0", df, "m3", "m0", "m3_correct", "m0_correct")
    hth("m3_vs_m1", df, "m3", "m1", "m3_correct", "m1_correct")
    hth("m4_vs_m0", df, "m4", "m0", "m4_correct", "m0_correct")
    hth("m2_vs_m0", pop2, "m2", "m0", "m2_correct", "m0_correct")
    hth("m2_vs_m1", pop2, "m2", "m1", "m2_correct", "m1_correct")

    calibration = {
        "m0": calibration_table(df, "x_p_raw"),
        "m1": calibration_table(df, "m1_p"),
        "m3": calibration_table(df, "m3_oos_p"),
        "m4": calibration_table(df, "m4_oos_p"),
        "m2": calibration_table(pop2, "m2_oos_p"),
    }

    decisive = {
        "m1_vs_m0": decisive_report(df, "m1_pick_home", "m1_correct", "m0_pick_home", "m0_correct"),
        "m3_vs_m0": decisive_report(df, "m3_pick_home", "m3_correct", "m0_pick_home", "m0_correct"),
        "m4_vs_m0": decisive_report(df, "m4_pick_home", "m4_correct", "m0_pick_home", "m0_correct"),
        "m2_vs_m0": decisive_report(
            pop2, "m2_pick_home", "m2_correct", "m0_pick_home", "m0_correct"
        ),
    }

    records = {
        "m0": games_record(df["m0_correct"]),
        "m1": games_record(df["m1_correct"]),
        "m3_oos": games_record(df["m3_correct"]),
        "m4_oos": games_record(df["m4_correct"]),
        "m2_oos": games_record(pop2["m2_correct"]),
    }

    m2_diff_mask = pop2["m2_pick_home"].ne(pop2["m0_pick_home"])
    pop2_pc = pop2.copy()
    pop2_pc["foresight_on_m2_diff"] = np.where(
        m2_diff_mask & pop2["m0_correct"].eq(0.0), 1.0, pop2["m0_correct"]
    )
    record_look("positive_control_foresight_on_m2_diff_games")
    positive_control = paired_effect_accuracy(pop2_pc, "foresight_on_m2_diff", "m0_correct")
    positive_control_detail = {
        "n_m2_diff_games": int(m2_diff_mask.sum()),
        "card_record_on_diff": games_record(pop2.loc[m2_diff_mask, "m0_correct"]),
        "foresight_record_on_diff": games_record(pop2_pc.loc[m2_diff_mask, "foresight_on_m2_diff"]),
        "effect_vs_m0": positive_control,
    }

    seasons_full = season_split(
        df,
        {
            "m0_accuracy": "m0_correct",
            "m1_accuracy": "m1_correct",
            "m3_oos_accuracy": "m3_correct",
            "m4_oos_accuracy": "m4_correct",
        },
    )
    seasons_m2 = season_split(
        pop2,
        {"m0_accuracy": "m0_correct", "m1_accuracy": "m1_correct", "m2_oos_accuracy": "m2_correct"},
    )

    registry_wrong_sign: list[str] = []
    for name, cell in head_to_head.items():
        for metric_name, effect in cell.items():
            if resolved_wrong_sign(effect):
                registry_wrong_sign.append(f"{name}:{metric_name}")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("artifacts/joint_probability_model") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "opener_evaluation_directory": str(opener_dir),
        "opener_evaluation_matches_active_model": matches_active,
        "market_population": str(MARKET_POPULATION_PATH),
        "arrest_snapshot_id": arrest_snapshot_id,
        "pbp_snapshot": str(pbp_snapshot),
        "seed": SEED,
        "samples": SAMPLES,
        "candidate_l2": CANDIDATE_L2,
        "fixed_l2_m4": FIXED_L2_M4,
        "n_games_2020_2025_before_push_drop": len(base),
        "n_pushes_dropped": n_pushes,
        "n_games_graded": len(df),
        "n_games_m2_population_2023_2025": len(pop2),
        "n_raw_vs_served_open_probability_differ": n_raw_vs_served_diff,
        "flag_coverage": flag_coverage,
        "m1_reproduction_detail": m1_detail,
        "records": records,
        "head_to_head": head_to_head,
        "calibration": calibration,
        "decisive": decisive,
        "positive_control_foresight_on_m2_diff": positive_control_detail,
        "season_split_full_population": seasons_full,
        "season_split_m2_population": seasons_m2,
        "m2_fold_coefficients": m2["fold_coefficients"],
        "m3_fold_coefficients": m3["fold_coefficients"],
        "m4_fold_coefficients": m4["fold_coefficients"],
        "registry_resolved_wrong_sign_cells": registry_wrong_sign,
        "look_count": len(LOOKS),
        "look_log": LOOKS,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    coefficient_rows = []
    for arm_name, fit in (("m2", m2), ("m3", m3), ("m4", m4)):
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
        "tue_open_home_spread",
        "x_p_raw",
        "home_covered",
        "flag_coach",
        "flag_division",
        "flag_arrests",
        "flag_bye",
        "flag_cold_visitor",
        "flag_protection",
        "flag_interim_hc",
        "flag_tank_zone",
        "flag_precip",
        "move",
        "m0_pick_home",
        "m0_correct",
        "m1_pick_home",
        "m1_correct",
        "m3_pick_home",
        "m3_correct",
        "m4_pick_home",
        "m4_correct",
    ]
    df_export = df[export_cols].merge(
        pop2[["game_id", "m2_pick_home", "m2_correct", "m2_oos_p"]], on="game_id", how="left"
    )
    df_export.to_csv(out_dir / "per_game.csv", index=False)

    print(f"artifact_dir={out_dir}")
    print("records", json.dumps(records))
    print("n_pushes_dropped", n_pushes, "n_games_graded", len(df), "n_m2_population", len(pop2))
    print("n_raw_vs_served_open_probability_differ", n_raw_vs_served_diff)
    print("look_count", len(LOOKS))
    print("head_to_head", json.dumps(head_to_head, indent=2))
    print("decisive", json.dumps(decisive, indent=2))
    print("positive_control", json.dumps(positive_control_detail, indent=2))
    print("m4_fold_coefficients", json.dumps(m4["fold_coefficients"], indent=2))
    print(
        "m3_fold_coefficients_x0_only",
        json.dumps({k: v.get("x0_model_logit") for k, v in m3["fold_coefficients"].items()}),
    )
    print(
        "m2_fold_coefficients_x0_and_move",
        json.dumps(
            {
                k: {"x0_model_logit": v.get("x0_model_logit"), "move": v.get("move")}
                for k, v in m2["fold_coefficients"].items()
            }
        ),
    )
    print("registry_resolved_wrong_sign_cells", registry_wrong_sign)


if __name__ == "__main__":
    main()
