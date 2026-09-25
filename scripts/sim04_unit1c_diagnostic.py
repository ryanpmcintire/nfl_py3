from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

PBP_SNAPSHOT_DIR = REPO / "data" / "pbp" / "raw" / "20260817T184927Z"
GAME_FEATURES_PATH = REPO / "data" / "processed" / "game_features_pbp.parquet"
ARTIFACT_ROOT = REPO / "artifacts" / "sim04_unit1c"

VALIDATION_TRAIN_SEASONS = tuple(range(2009, 2015))
VALIDATION_EVAL_SEASONS = tuple(range(2015, 2018))
TEST_TRAIN_SEASONS = tuple(range(2009, 2018))
TEST_EVAL_SEASONS = tuple(range(2018, 2026))

KEY_NUMBERS = (3, 7, 10, 14, 17)
FP_BIN_EDGES = np.array(list(range(0, 101, 10)), dtype=float)
N_FP_BUCKETS = len(FP_BIN_EDGES) - 1
N_GAMES = 10_000
RNG_SEED = 20260925
N_BOOT = 2000
OT_SECONDS = 600.0
MAX_OT_POSSESSIONS = 12

CONFIGS = {
    "A": {"fine_score": False, "state_transition": False, "min_cell_n": 25},
    "B": {"fine_score": False, "state_transition": True, "min_cell_n": 25},
    "C": {"fine_score": True, "state_transition": False, "min_cell_n": 25},
    "D": {"fine_score": True, "state_transition": True, "min_cell_n": 25},
    "E": {"fine_score": False, "state_transition": False, "min_cell_n": 15},
    "F": {"fine_score": True, "state_transition": True, "min_cell_n": 15},
}


def fp_bucket(fp: float) -> int:
    idx = int(np.digitize([min(max(fp, 0.0), 99.999)], FP_BIN_EDGES)[0]) - 1
    return min(max(idx, 0), N_FP_BUCKETS - 1)


def score_bucket_fine(diff: float, fine_split: bool) -> int:
    if not fine_split:
        if diff <= -17:
            return 0
        if diff <= -9:
            return 1
        if diff <= -4:
            return 2
        if diff <= -1:
            return 3
        if diff == 0:
            return 4
        if diff <= 3:
            return 5
        if diff <= 8:
            return 6
        if diff <= 16:
            return 7
        return 8
    if diff <= -17:
        return 0
    if diff <= -9:
        return 1
    if diff <= -7:
        return 2
    if diff <= -4:
        return 3
    if diff <= -1:
        return 4
    if diff == 0:
        return 5
    if diff <= 3:
        return 6
    if diff <= 6:
        return 7
    if diff <= 8:
        return 8
    if diff <= 16:
        return 9
    return 10


def score_bucket_coarse(diff: float) -> int:
    if diff <= -9:
        return 0
    if diff < 0:
        return 1
    if diff == 0:
        return 2
    if diff <= 8:
        return 3
    return 4


def seconds_left_in_period(qtr: int, gsr: float) -> float:
    return gsr - (4 - qtr) * 900.0


def time_bucket_fine(qtr: int, gsr: float) -> int:
    if qtr >= 5:
        return 7
    left = seconds_left_in_period(qtr, gsr)
    if qtr == 1:
        return 0
    if qtr == 2:
        return 2 if left <= 120 else 1
    if qtr == 3:
        return 3
    if left <= 120:
        return 6
    if left <= 300:
        return 5
    return 4


def time_bucket_coarse(tb_fine: int) -> int:
    if tb_fine == 7:
        return 2
    if tb_fine in (2, 5, 6):
        return 1
    return 0


def load_seasons(seasons: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = PBP_SNAPSHOT_DIR / f"season={season}" / "plays.parquet"
        frame = pd.read_parquet(path)
        frame = frame[frame["season_type"] == "REG"].copy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _emit_drive(
    rows: list,
    game_id: str,
    season: int,
    running: dict[str, float],
    drive_start_score: dict[str, float] | None,
    drive_offense: str | None,
    drive_defense: str | None,
    drive_start_fp: float | None,
    drive_category: str | None,
    drive_start_diff: float | None,
    drive_start_qtr: float | None,
    drive_start_gsr: float | None,
    drive_last_gsr: float | None,
) -> bool:
    if (
        drive_start_fp is None
        or drive_offense is None
        or drive_defense is None
        or drive_start_score is None
    ):
        return True
    if drive_start_diff is None or drive_start_qtr is None or drive_start_gsr is None:
        return False
    end_gsr = drive_last_gsr if drive_last_gsr is not None else drive_start_gsr
    duration = max(0.0, min(1800.0, drive_start_gsr - end_gsr))
    rows.append(
        (
            game_id,
            season,
            drive_offense,
            drive_defense,
            drive_start_fp,
            drive_category,
            running[drive_offense] - drive_start_score[drive_offense],
            running[drive_defense] - drive_start_score[drive_defense],
            drive_start_diff,
            int(drive_start_qtr),
            drive_start_gsr,
            duration,
        )
    )
    return True


def reconstruct_drives(pbp: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    pbp = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    rows = []
    dropped_missing_state = 0
    for game_id, game in pbp.groupby("game_id", sort=False):
        home_team = game["home_team"].iloc[0]
        away_team = game["away_team"].iloc[0]
        season = game["season"].iloc[0]
        running = {home_team: 0.0, away_team: 0.0}
        drive_start_score: dict[str, float] | None = None
        current_drive = None
        drive_offense = None
        drive_defense = None
        drive_start_fp = None
        drive_start_diff = None
        drive_start_qtr = None
        drive_start_gsr = None
        drive_last_gsr = None
        drive_category = None

        for play in game.itertuples(index=False):
            drive_id = play.fixed_drive
            if pd.isna(drive_id):
                continue
            if drive_id != current_drive:
                if current_drive is not None and not _emit_drive(
                    rows,
                    game_id,
                    season,
                    running,
                    drive_start_score,
                    drive_offense,
                    drive_defense,
                    drive_start_fp,
                    drive_category,
                    drive_start_diff,
                    drive_start_qtr,
                    drive_start_gsr,
                    drive_last_gsr,
                ):
                    dropped_missing_state += 1
                current_drive = drive_id
                drive_offense = None
                drive_defense = None
                drive_start_fp = None
                drive_start_diff = None
                drive_start_qtr = None
                drive_start_gsr = None
                drive_last_gsr = None
                drive_category = play.fixed_drive_result
                drive_start_score = None
            if drive_offense is None and pd.notna(play.posteam) and pd.notna(play.defteam):
                drive_offense = play.posteam
                drive_defense = play.defteam
                drive_start_score = dict(running)
            if (
                drive_start_fp is None
                and play.play_type != "kickoff"
                and pd.notna(play.yardline_100)
            ):
                drive_start_fp = play.yardline_100
                if (
                    pd.notna(play.score_differential)
                    and pd.notna(play.qtr)
                    and pd.notna(play.game_seconds_remaining)
                ):
                    drive_start_diff = float(play.score_differential)
                    drive_start_qtr = play.qtr
                    drive_start_gsr = float(play.game_seconds_remaining)
            if pd.notna(play.game_seconds_remaining):
                drive_last_gsr = float(play.game_seconds_remaining)
            posteam = play.posteam
            score_post = play.posteam_score_post
            if pd.notna(posteam) and pd.notna(score_post) and posteam in running:
                running[posteam] = score_post
        if current_drive is not None and not _emit_drive(
            rows,
            game_id,
            season,
            running,
            drive_start_score,
            drive_offense,
            drive_defense,
            drive_start_fp,
            drive_category,
            drive_start_diff,
            drive_start_qtr,
            drive_start_gsr,
            drive_last_gsr,
        ):
            dropped_missing_state += 1
    drives = pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "season",
            "offense",
            "defense",
            "start_fp",
            "category",
            "points_off",
            "points_def",
            "start_diff",
            "start_qtr",
            "start_gsr",
            "duration",
        ],
    )
    return drives, dropped_missing_state


def build_opening_fp(drives: pd.DataFrame) -> np.ndarray:
    firsts = (
        drives.sort_values(["game_id", "start_gsr"], ascending=[True, False])
        .groupby("game_id", sort=False)
        .first()
    )
    return firsts["start_fp"].to_numpy()


def build_transition_pools(
    drives: pd.DataFrame,
) -> tuple[dict[tuple[str, int, int], np.ndarray], dict[str, np.ndarray]]:
    fine_raw: dict[tuple[str, int, int], list[float]] = {}
    coarse_raw: dict[str, list[float]] = {}
    for _, game in drives.sort_values("start_gsr", ascending=False).groupby("game_id", sort=False):
        game = game.reset_index(drop=True)
        for i in range(len(game) - 1):
            category = game.loc[i, "category"]
            next_fp = float(game.loc[i + 1, "start_fp"])
            coarse_raw.setdefault(category, []).append(next_fp)
            sb_c = score_bucket_coarse(float(game.loc[i + 1, "start_diff"]))
            tb_c = time_bucket_coarse(
                time_bucket_fine(
                    int(game.loc[i + 1, "start_qtr"]), float(game.loc[i + 1, "start_gsr"])
                )
            )
            fine_raw.setdefault((category, sb_c, tb_c), []).append(next_fp)
    fine = {key: np.array(values, dtype=float) for key, values in fine_raw.items()}
    coarse = {key: np.array(values, dtype=float) for key, values in coarse_raw.items()}
    return fine, coarse


def build_state_cells(
    drives: pd.DataFrame,
    fine_split: bool,
) -> tuple[dict, dict, dict, list]:
    sb_fine = drives["start_diff"].map(lambda d: score_bucket_fine(d, fine_split)).to_numpy()
    sb_coarse = drives["start_diff"].map(score_bucket_coarse).to_numpy()
    tb_fine = np.array(
        [
            time_bucket_fine(int(q), g)
            for q, g in zip(drives["start_qtr"], drives["start_gsr"], strict=True)
        ]
    )
    tb_coarse = np.array([time_bucket_coarse(int(t)) for t in tb_fine])
    fb = drives["start_fp"].map(fp_bucket).to_numpy()

    cat = drives["category"].to_numpy()
    poff = drives["points_off"].to_numpy(dtype=float)
    pdef = drives["points_def"].to_numpy(dtype=float)
    dur = drives["duration"].to_numpy(dtype=float)

    def make_pool(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return cat[mask], poff[mask], pdef[mask], dur[mask]

    level0: dict[tuple[int, int, int], tuple] = {}
    for key in set(zip(sb_fine.tolist(), tb_fine.tolist(), fb.tolist(), strict=True)):
        mask = (sb_fine == key[0]) & (tb_fine == key[1]) & (fb == key[2])
        level0[key] = make_pool(mask)

    level1: dict[tuple[int, int, int], tuple] = {}
    for key in set(zip(sb_fine.tolist(), tb_coarse.tolist(), fb.tolist(), strict=True)):
        mask = (sb_fine == key[0]) & (tb_coarse == key[1]) & (fb == key[2])
        level1[key] = make_pool(mask)

    level2: dict[tuple[int, int, int], tuple] = {}
    for key in set(zip(sb_coarse.tolist(), tb_coarse.tolist(), fb.tolist(), strict=True)):
        mask = (sb_coarse == key[0]) & (tb_coarse == key[1]) & (fb == key[2])
        level2[key] = make_pool(mask)

    level3: list[tuple] = []
    for b in range(N_FP_BUCKETS):
        mask = fb == b
        level3.append(make_pool(mask))

    return level0, level1, level2, level3


def draw_cell(
    diff: float,
    qtr: int,
    gsr: float,
    fp: float,
    level0: dict,
    level1: dict,
    level2: dict,
    level3: list,
    min_cell_n: int,
    fine_split: bool,
) -> tuple:
    sb_f = score_bucket_fine(diff, fine_split)
    sb_c = score_bucket_coarse(diff)
    tb_f = time_bucket_fine(qtr, gsr)
    tb_c = time_bucket_coarse(tb_f)
    fb = fp_bucket(fp)

    pool = level0.get((sb_f, tb_f, fb))
    if pool is not None and len(pool[0]) >= min_cell_n:
        return pool
    pool = level1.get((sb_f, tb_c, fb))
    if pool is not None and len(pool[0]) >= min_cell_n:
        return pool
    pool = level2.get((sb_c, tb_c, fb))
    if pool is not None and len(pool[0]) >= min_cell_n:
        return pool
    return level3[fb]


def draw_next_fp(
    rng: np.random.Generator,
    category: str,
    diff: float,
    qtr: int,
    gsr: float,
    transition_fine: dict,
    transition_coarse: dict,
    opening_fp_pool: np.ndarray,
    min_cell_n: int,
    state_conditioned: bool,
) -> float:
    if state_conditioned:
        sb_c = score_bucket_coarse(diff)
        tb_c = time_bucket_coarse(time_bucket_fine(qtr, gsr))
        pool = transition_fine.get((category, sb_c, tb_c))
        if pool is not None and len(pool) >= min_cell_n:
            return float(rng.choice(pool))
    pool = transition_coarse.get(category)
    if pool is None or len(pool) == 0:
        pool = opening_fp_pool
    return float(rng.choice(pool))


def simulate_games(
    n_games: int,
    rng: np.random.Generator,
    opening_fp_pool: np.ndarray,
    transition_fine: dict,
    transition_coarse: dict,
    level0: dict,
    level1: dict,
    level2: dict,
    level3: list,
    min_cell_n: int,
    fine_split: bool,
    state_conditioned: bool,
) -> tuple[np.ndarray, np.ndarray]:
    margins = np.empty(n_games, dtype=float)
    went_to_ot = np.zeros(n_games, dtype=bool)

    for game_index in range(n_games):
        offense = "home" if rng.random() < 0.5 else "away"
        score = {"home": 0.0, "away": 0.0}
        start_fp = float(rng.choice(opening_fp_pool))
        gsr = 3600.0

        while gsr > 0:
            defense = "away" if offense == "home" else "home"
            elapsed = 3600.0 - gsr
            qtr = min(4, int(elapsed // 900) + 1)
            diff = score[offense] - score[defense]
            categories, points_off, points_def, durations = draw_cell(
                diff, qtr, gsr, start_fp, level0, level1, level2, level3, min_cell_n, fine_split
            )
            draw = rng.integers(0, len(categories))
            score[offense] += float(points_off[draw])
            score[defense] += float(points_def[draw])
            duration = float(durations[draw])
            category = categories[draw]
            gsr = max(0.0, gsr - duration)
            new_elapsed = 3600.0 - gsr
            new_qtr = min(4, int(new_elapsed // 900) + 1)
            next_diff = score[defense] - score[offense]
            start_fp = draw_next_fp(
                rng,
                category,
                next_diff,
                new_qtr,
                gsr,
                transition_fine,
                transition_coarse,
                opening_fp_pool,
                min_cell_n,
                state_conditioned,
            )
            offense = defense

        if score["home"] == score["away"]:
            went_to_ot[game_index] = True
            offense = "home" if rng.random() < 0.5 else "away"
            start_fp = float(rng.choice(opening_fp_pool))
            ot_clock = OT_SECONDS
            possessions = {"home": 0, "away": 0}
            settled = False
            for _ in range(MAX_OT_POSSESSIONS):
                if ot_clock <= 0 or settled:
                    break
                defense = "away" if offense == "home" else "home"
                diff = score[offense] - score[defense]
                off_before = score[offense]
                def_before = score[defense]
                categories, points_off, points_def, durations = draw_cell(
                    diff, 5, 0.0, start_fp, level0, level1, level2, level3, min_cell_n, fine_split
                )
                draw = rng.integers(0, len(categories))
                score[offense] += float(points_off[draw])
                score[defense] += float(points_def[draw])
                duration = float(durations[draw])
                category = categories[draw]
                off_gain = score[offense] - off_before
                def_gain = score[defense] - def_before
                ot_clock = max(0.0, ot_clock - duration)
                possessions[offense] += 1
                next_diff = score[defense] - score[offense]
                start_fp = draw_next_fp(
                    rng,
                    category,
                    next_diff,
                    5,
                    0.0,
                    transition_fine,
                    transition_coarse,
                    opening_fp_pool,
                    min_cell_n,
                    state_conditioned,
                )
                if (
                    def_gain > 0
                    or off_gain >= 6
                    or (
                        possessions["home"] >= 1
                        and possessions["away"] >= 1
                        and score["home"] != score["away"]
                    )
                ):
                    settled = True
                offense = defense
        margins[game_index] = score["home"] - score["away"]
    return margins, went_to_ot


def key_number_mass(margins: np.ndarray, keys: tuple[int, ...]) -> dict[int, float]:
    abs_margins = np.abs(margins)
    return {k: float(np.mean(abs_margins == k)) for k in keys}


def bootstrap_season_ci(
    actual_by_season: dict[int, np.ndarray],
    keys: tuple[int, ...],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[int, tuple[float, float, float]]:
    seasons = list(actual_by_season.keys())
    boot_masses = {k: [] for k in keys}
    for _ in range(n_boot):
        sampled_seasons = rng.choice(seasons, size=len(seasons), replace=True)
        pooled = np.concatenate([actual_by_season[s] for s in sampled_seasons])
        mass = key_number_mass(pooled, keys)
        for k in keys:
            boot_masses[k].append(mass[k])
    result = {}
    for k in keys:
        arr = np.array(boot_masses[k])
        result[k] = (
            float(np.mean(arr)),
            float(np.percentile(arr, 5)),
            float(np.percentile(arr, 95)),
        )
    return result


def discrete_log_loss(
    actual_margins: np.ndarray, reference_margins: np.ndarray, max_abs: int = 60
) -> float:
    grid = np.arange(-max_abs, max_abs + 1)
    counts = np.array([(reference_margins == g).sum() for g in grid], dtype=float)
    counts += 1.0
    probs = counts / counts.sum()
    clipped_actual = np.clip(actual_margins, -max_abs, max_abs).round().astype(int)
    idx = clipped_actual + max_abs
    p = probs[idx]
    return float(-np.mean(np.log(p)))


def run_config(
    config_name: str,
    config: dict,
    train_drives: pd.DataFrame,
    train_margins_actual: np.ndarray,
    eval_margins_actual: np.ndarray,
    actual_by_season: dict[int, np.ndarray],
    boot_ci: dict,
    rng: np.random.Generator,
) -> dict:
    opening_fp_pool = build_opening_fp(train_drives)
    transition_fine, transition_coarse = build_transition_pools(train_drives)
    level0, level1, level2, level3 = build_state_cells(train_drives, config["fine_score"])

    sim_margins, went_to_ot = simulate_games(
        N_GAMES,
        rng,
        opening_fp_pool,
        transition_fine,
        transition_coarse,
        level0,
        level1,
        level2,
        level3,
        config["min_cell_n"],
        config["fine_score"],
        config["state_transition"],
    )

    sim_mass = key_number_mass(sim_margins, KEY_NUMBERS)
    hits = sum(1 for k in KEY_NUMBERS if boot_ci[k][1] <= sim_mass[k] <= boot_ci[k][2])
    sim_log_loss = discrete_log_loss(eval_margins_actual, sim_margins)
    naive_log_loss = discrete_log_loss(eval_margins_actual, train_margins_actual)
    log_loss_delta = sim_log_loss - naive_log_loss
    go_decision = bool(hits >= 4 and log_loss_delta <= 0.02)

    return {
        "config_name": config_name,
        "config": config,
        "key_number_mass": {
            str(k): {
                "simulated": sim_mass[k],
                "actual_eval_bootstrap_p05": boot_ci[k][1],
                "actual_eval_bootstrap_p95": boot_ci[k][2],
                "within_ci": bool(boot_ci[k][1] <= sim_mass[k] <= boot_ci[k][2]),
            }
            for k in KEY_NUMBERS
        },
        "key_number_hits_of_5": hits,
        "discrete_log_loss": {
            "simulator_vs_actual_eval": sim_log_loss,
            "naive_train_histogram_vs_actual_eval": naive_log_loss,
            "delta_sim_minus_naive": log_loss_delta,
        },
        "go_no_go": "GO" if go_decision else "NO_GO",
        "sim_margin_mean": float(np.mean(sim_margins)),
        "sim_margin_std": float(np.std(sim_margins)),
        "sim_ot_share_of_games": float(np.mean(went_to_ot)),
    }


def decompose_margin3(
    eval_pbp: pd.DataFrame, eval_seasons: tuple[int, ...], game_features: pd.DataFrame
) -> dict:
    eval_drives, _ = reconstruct_drives(eval_pbp)
    last_drive = (
        eval_drives.sort_values(["game_id", "start_gsr"], ascending=[True, False])
        .groupby("game_id", sort=False)
        .last()
    )
    gf = game_features[game_features["season"].isin(eval_seasons)].copy()
    gf["margin"] = gf["home_score"] - gf["away_score"]
    ot_game_ids = set(eval_pbp.loc[eval_pbp["qtr"] >= 5, "game_id"].unique().tolist())
    gf["went_to_ot"] = gf["game_id"].isin(ot_game_ids)

    m3 = gf[gf["margin"].abs() == 3]
    m3_ot = m3[m3["went_to_ot"]]
    m3_reg = m3[~m3["went_to_ot"]]
    reg_last_cat = last_drive["category"].reindex(m3_reg["game_id"]).fillna("unknown")
    cat_counts = reg_last_cat.value_counts().to_dict()

    n_m3 = len(m3)
    return {
        "n_eval_games": len(gf),
        "n_margin3_games": n_m3,
        "n_margin3_ot_games": len(m3_ot),
        "margin3_ot_share": len(m3_ot) / n_m3 if n_m3 else 0.0,
        "n_margin3_nonot_games": len(m3_reg),
        "margin3_nonot_last_drive_category_counts": {str(k): int(v) for k, v in cat_counts.items()},
        "margin3_nonot_last_drive_field_goal_share": (
            cat_counts.get("Field goal", 0) / len(m3_reg) if len(m3_reg) else 0.0
        ),
        "margin3_nonot_last_drive_touchdown_share": (
            cat_counts.get("Touchdown", 0) / len(m3_reg) if len(m3_reg) else 0.0
        ),
    }


def run_validation() -> dict:
    rng = np.random.default_rng(RNG_SEED)

    train_pbp = load_seasons(VALIDATION_TRAIN_SEASONS)
    train_drives, dropped_missing_state = reconstruct_drives(train_pbp)

    eval_pbp = load_seasons(VALIDATION_EVAL_SEASONS)

    game_features = pd.read_parquet(GAME_FEATURES_PATH)
    game_features = game_features[
        game_features["season"].isin(VALIDATION_TRAIN_SEASONS + VALIDATION_EVAL_SEASONS)
    ].copy()
    game_features["margin"] = game_features["home_score"] - game_features["away_score"]

    train_margins_actual = game_features[game_features["season"].isin(VALIDATION_TRAIN_SEASONS)][
        "margin"
    ].to_numpy()
    eval_margins_actual = game_features[game_features["season"].isin(VALIDATION_EVAL_SEASONS)][
        "margin"
    ].to_numpy()

    actual_by_season = {
        int(season): group["margin"].to_numpy()
        for season, group in game_features[
            game_features["season"].isin(VALIDATION_EVAL_SEASONS)
        ].groupby("season")
    }
    boot_ci = bootstrap_season_ci(actual_by_season, KEY_NUMBERS, n_boot=N_BOOT, rng=rng)

    config_results = {}
    for name, config in CONFIGS.items():
        config_results[name] = run_config(
            name,
            config,
            train_drives,
            train_margins_actual,
            eval_margins_actual,
            actual_by_season,
            boot_ci,
            rng,
        )

    decomposition = decompose_margin3(eval_pbp, VALIDATION_EVAL_SEASONS, game_features)

    return {
        "mode": "validation",
        "train_seasons": list(VALIDATION_TRAIN_SEASONS),
        "eval_seasons": list(VALIDATION_EVAL_SEASONS),
        "n_train_drives": len(train_drives),
        "n_train_drives_dropped_missing_state": dropped_missing_state,
        "configs": config_results,
        "margin3_decomposition": decomposition,
    }


def run_test(config_name: str) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    config = CONFIGS[config_name]

    train_pbp = load_seasons(TEST_TRAIN_SEASONS)
    train_drives, dropped_missing_state = reconstruct_drives(train_pbp)

    game_features = pd.read_parquet(GAME_FEATURES_PATH)
    game_features = game_features[
        game_features["season"].isin(TEST_TRAIN_SEASONS + TEST_EVAL_SEASONS)
    ].copy()
    game_features["margin"] = game_features["home_score"] - game_features["away_score"]

    train_margins_actual = game_features[game_features["season"].isin(TEST_TRAIN_SEASONS)][
        "margin"
    ].to_numpy()
    eval_margins_actual = game_features[game_features["season"].isin(TEST_EVAL_SEASONS)][
        "margin"
    ].to_numpy()

    actual_by_season = {
        int(season): group["margin"].to_numpy()
        for season, group in game_features[game_features["season"].isin(TEST_EVAL_SEASONS)].groupby(
            "season"
        )
    }
    boot_ci = bootstrap_season_ci(actual_by_season, KEY_NUMBERS, n_boot=N_BOOT, rng=rng)

    result = run_config(
        config_name,
        config,
        train_drives,
        train_margins_actual,
        eval_margins_actual,
        actual_by_season,
        boot_ci,
        rng,
    )
    result["mode"] = "test"
    result["train_seasons"] = list(TEST_TRAIN_SEASONS)
    result["eval_seasons"] = list(TEST_EVAL_SEASONS)
    result["n_train_drives"] = len(train_drives)
    result["n_train_drives_dropped_missing_state"] = dropped_missing_state
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validation", "test"], default="validation")
    parser.add_argument("--config", choices=list(CONFIGS), default="A")
    args = parser.parse_args()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "validation":
        report = run_validation()
        out_path = out_dir / "validation_report.json"
    else:
        report = run_test(args.config)
        out_path = out_dir / "test_report.json"

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Artifact directory: {out_dir}")


if __name__ == "__main__":
    main()
