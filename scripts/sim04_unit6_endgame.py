from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import sim04_unit1b_state_chain as u1b  # noqa: E402

PBP_SNAPSHOT_DIR = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
GAME_FEATURES_PATH = u1b.GAME_FEATURES_PATH
ARTIFACT_ROOT = REPO / "artifacts" / "sim04_unit6"

VALIDATION_TRAIN_SEASONS = tuple(range(2009, 2015))
VALIDATION_EVAL_SEASONS = tuple(range(2015, 2018))
TEST_TRAIN_SEASONS = tuple(range(2009, 2018))
TEST_EVAL_SEASONS = tuple(range(2018, 2026))

KEY_NUMBERS = u1b.KEY_NUMBERS
N_GAMES = u1b.N_GAMES
RNG_SEED = u1b.RNG_SEED
N_BOOT = u1b.N_BOOT

CONFIGS = {
    "A": {"name": "A", "use_late": False, "include_off_to": False, "min_cell_n_late": None},
    "B": {"name": "B", "use_late": True, "include_off_to": False, "min_cell_n_late": 15},
    "C": {"name": "C", "use_late": True, "include_off_to": True, "min_cell_n_late": 15},
    "D": {"name": "D", "use_late": True, "include_off_to": False, "min_cell_n_late": 25},
}


def to_bucket(value: float) -> int:
    v = 3.0 if pd.isna(value) else float(value)
    v = round(min(max(v, 0.0), 3.0))
    if v <= 0:
        return 0
    if v == 1:
        return 1
    return 2


def late_time_bucket(qtr: int, gsr: float) -> int | None:
    if qtr >= 5:
        return 2
    if qtr == 4:
        if gsr <= 120.0:
            return 1
        if gsr <= 300.0:
            return 0
    return None


def _emit_drive_to(
    rows: list,
    game_id: str,
    season: int,
    running: dict[str, float],
    drive_start_score,
    drive_offense,
    drive_defense,
    drive_start_fp,
    drive_category,
    drive_start_diff,
    drive_start_qtr,
    drive_start_gsr,
    drive_last_gsr,
    drive_start_off_to,
    drive_start_def_to,
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
            3.0 if drive_start_off_to is None else drive_start_off_to,
            3.0 if drive_start_def_to is None else drive_start_def_to,
        )
    )
    return True


def reconstruct_drives_to(pbp: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    pbp = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    rows = []
    dropped_missing_state = 0
    for game_id, game in pbp.groupby("game_id", sort=False):
        home_team = game["home_team"].iloc[0]
        away_team = game["away_team"].iloc[0]
        season = game["season"].iloc[0]
        running = {home_team: 0.0, away_team: 0.0}
        drive_start_score = None
        current_drive = None
        drive_offense = None
        drive_defense = None
        drive_start_fp = None
        drive_start_diff = None
        drive_start_qtr = None
        drive_start_gsr = None
        drive_last_gsr = None
        drive_category = None
        drive_start_off_to = None
        drive_start_def_to = None

        for play in game.itertuples(index=False):
            drive_id = play.fixed_drive
            if pd.isna(drive_id):
                continue
            if drive_id != current_drive:
                if current_drive is not None and not _emit_drive_to(
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
                    drive_start_off_to,
                    drive_start_def_to,
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
                drive_start_off_to = None
                drive_start_def_to = None
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
                    drive_start_off_to = (
                        float(play.posteam_timeouts_remaining)
                        if pd.notna(play.posteam_timeouts_remaining)
                        else None
                    )
                    drive_start_def_to = (
                        float(play.defteam_timeouts_remaining)
                        if pd.notna(play.defteam_timeouts_remaining)
                        else None
                    )
            if pd.notna(play.game_seconds_remaining):
                drive_last_gsr = float(play.game_seconds_remaining)
            posteam = play.posteam
            score_post = play.posteam_score_post
            if pd.notna(posteam) and pd.notna(score_post) and posteam in running:
                running[posteam] = score_post
        if current_drive is not None and not _emit_drive_to(
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
            drive_start_off_to,
            drive_start_def_to,
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
            "start_off_to",
            "start_def_to",
        ],
    )
    return drives, dropped_missing_state


def build_late_pools(drives: pd.DataFrame, include_off_to: bool) -> tuple[dict, dict]:
    ltb = np.array(
        [
            late_time_bucket(int(q), g)
            for q, g in zip(drives["start_qtr"], drives["start_gsr"], strict=True)
        ]
    )
    late_mask = ltb != None  # noqa: E711
    late = drives[late_mask].copy()
    late_ltb = ltb[late_mask].astype(int)
    sb_f = late["start_diff"].map(u1b.score_bucket_fine).to_numpy()
    fb = late["start_fp"].map(u1b.fp_bucket).to_numpy()
    dto = late["start_def_to"].map(to_bucket).to_numpy()
    oto = late["start_off_to"].map(to_bucket).to_numpy()

    cat = late["category"].to_numpy()
    poff = late["points_off"].to_numpy(dtype=float)
    pdef = late["points_def"].to_numpy(dtype=float)
    dur = late["duration"].to_numpy(dtype=float)

    def make_pool(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return cat[mask], poff[mask], pdef[mask], dur[mask]

    fine: dict[tuple, tuple] = {}
    coarse: dict[tuple, tuple] = {}
    if include_off_to:
        fine_keys = set(
            zip(
                sb_f.tolist(),
                late_ltb.tolist(),
                fb.tolist(),
                dto.tolist(),
                oto.tolist(),
                strict=True,
            )
        )
        for key in fine_keys:
            mask = (
                (sb_f == key[0])
                & (late_ltb == key[1])
                & (fb == key[2])
                & (dto == key[3])
                & (oto == key[4])
            )
            fine[key] = make_pool(mask)
        coarse_keys = set(
            zip(sb_f.tolist(), late_ltb.tolist(), dto.tolist(), oto.tolist(), strict=True)
        )
        for key in coarse_keys:
            mask = (sb_f == key[0]) & (late_ltb == key[1]) & (dto == key[2]) & (oto == key[3])
            coarse[key] = make_pool(mask)
    else:
        fine_keys = set(
            zip(sb_f.tolist(), late_ltb.tolist(), fb.tolist(), dto.tolist(), strict=True)
        )
        for key in fine_keys:
            mask = (sb_f == key[0]) & (late_ltb == key[1]) & (fb == key[2]) & (dto == key[3])
            fine[key] = make_pool(mask)
        coarse_keys = set(zip(sb_f.tolist(), late_ltb.tolist(), dto.tolist(), strict=True))
        for key in coarse_keys:
            mask = (sb_f == key[0]) & (late_ltb == key[1]) & (dto == key[2])
            coarse[key] = make_pool(mask)
    return fine, coarse


def build_to_joint(drives: pd.DataFrame) -> dict[tuple[int, int], list[tuple[int, int]]]:
    ltb = np.array(
        [
            late_time_bucket(int(q), g)
            for q, g in zip(drives["start_qtr"], drives["start_gsr"], strict=True)
        ]
    )
    late_mask = ltb != None  # noqa: E711
    late = drives[late_mask].copy()
    late_ltb = ltb[late_mask].astype(int)
    sb_f = late["start_diff"].map(u1b.score_bucket_fine).to_numpy()
    dto = late["start_def_to"].map(to_bucket).to_numpy()
    oto = late["start_off_to"].map(to_bucket).to_numpy()
    joint: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for s, t, d, o in zip(
        sb_f.tolist(), late_ltb.tolist(), dto.tolist(), oto.tolist(), strict=True
    ):
        joint.setdefault((s, t), []).append((d, o))
    return joint


def draw_cell_endgame(
    rng: np.random.Generator,
    diff: float,
    qtr: int,
    gsr: float,
    fp: float,
    config: dict,
    late_fine: dict,
    late_coarse: dict,
    to_joint: dict,
    level0: dict,
    level1: dict,
    level2: dict,
    level3: list,
) -> tuple:
    if config["use_late"]:
        ltb = late_time_bucket(qtr, gsr)
        if ltb is not None:
            sb_f = u1b.score_bucket_fine(diff)
            pairs = to_joint.get((sb_f, ltb))
            if pairs:
                dto, oto = pairs[rng.integers(0, len(pairs))]
            else:
                dto, oto = 2, 2
            fb = u1b.fp_bucket(fp)
            min_n = config["min_cell_n_late"]
            if config["include_off_to"]:
                key_fine = (sb_f, ltb, fb, dto, oto)
                key_coarse = (sb_f, ltb, dto, oto)
            else:
                key_fine = (sb_f, ltb, fb, dto)
                key_coarse = (sb_f, ltb, dto)
            pool = late_fine.get(key_fine)
            if pool is not None and len(pool[0]) >= min_n:
                return pool
            pool = late_coarse.get(key_coarse)
            if pool is not None and len(pool[0]) >= min_n:
                return pool
    return u1b.draw_cell(diff, qtr, gsr, fp, level0, level1, level2, level3)


def simulate_games_endgame(
    n_games: int,
    rng: np.random.Generator,
    opening_fp_pool: np.ndarray,
    transition_pools: dict[str, np.ndarray],
    config: dict,
    late_fine: dict,
    late_coarse: dict,
    to_joint: dict,
    level0: dict,
    level1: dict,
    level2: dict,
    level3: list,
) -> tuple[np.ndarray, np.ndarray]:
    margins = np.empty(n_games, dtype=float)
    went_to_ot = np.zeros(n_games, dtype=bool)

    def draw_next_fp(category: str) -> float:
        next_pool = transition_pools.get(category)
        if next_pool is None or len(next_pool) == 0:
            next_pool = opening_fp_pool
        return float(rng.choice(next_pool))

    def apply_drive(offense, defense, diff, qtr, gsr, fp, score):
        categories, points_off, points_def, durations = draw_cell_endgame(
            rng,
            diff,
            qtr,
            gsr,
            fp,
            config,
            late_fine,
            late_coarse,
            to_joint,
            level0,
            level1,
            level2,
            level3,
        )
        draw = rng.integers(0, len(categories))
        score[offense] += float(points_off[draw])
        score[defense] += float(points_def[draw])
        duration = float(durations[draw])
        next_fp = draw_next_fp(categories[draw])
        return categories[draw], duration, next_fp

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
            _category, duration, next_fp = apply_drive(
                offense, defense, diff, qtr, gsr, start_fp, score
            )
            gsr = max(0.0, gsr - duration)
            start_fp = next_fp
            offense = defense

        if score["home"] == score["away"]:
            went_to_ot[game_index] = True
            offense = "home" if rng.random() < 0.5 else "away"
            start_fp = float(rng.choice(opening_fp_pool))
            ot_clock = u1b.OT_SECONDS
            possessions = {"home": 0, "away": 0}
            settled = False
            for _ in range(u1b.MAX_OT_POSSESSIONS):
                if ot_clock <= 0 or settled:
                    break
                defense = "away" if offense == "home" else "home"
                diff = score[offense] - score[defense]
                off_before = score[offense]
                def_before = score[defense]
                _category, duration, next_fp = apply_drive(
                    offense, defense, diff, 5, 0.0, start_fp, score
                )
                off_gain = score[offense] - off_before
                def_gain = score[defense] - def_before
                ot_clock = max(0.0, ot_clock - duration)
                possessions[offense] += 1
                start_fp = next_fp
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


def run_config(
    config: dict,
    train_drives_to: pd.DataFrame,
    opening_fp_pool: np.ndarray,
    transition_pools: dict,
    level0: dict,
    level1: dict,
    level2: dict,
    level3: list,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if config["use_late"]:
        late_fine, late_coarse = build_late_pools(train_drives_to, config["include_off_to"])
        to_joint = build_to_joint(train_drives_to)
    else:
        late_fine, late_coarse, to_joint = {}, {}, {}
    return simulate_games_endgame(
        N_GAMES,
        rng,
        opening_fp_pool,
        transition_pools,
        config,
        late_fine,
        late_coarse,
        to_joint,
        level0,
        level1,
        level2,
        level3,
    )


def evaluate(
    sim_margins: np.ndarray,
    train_margins_actual: np.ndarray,
    eval_margins_actual: np.ndarray,
    actual_by_season: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> dict:
    sim_mass = u1b.key_number_mass(sim_margins, KEY_NUMBERS)
    boot_ci = u1b.bootstrap_season_ci(actual_by_season, KEY_NUMBERS, n_boot=N_BOOT, rng=rng)
    actual_mass = u1b.key_number_mass(eval_margins_actual, KEY_NUMBERS)
    hits = sum(1 for k in KEY_NUMBERS if boot_ci[k][1] <= sim_mass[k] <= boot_ci[k][2])
    sim_log_loss = u1b.discrete_log_loss(eval_margins_actual, sim_margins)
    naive_log_loss = u1b.discrete_log_loss(eval_margins_actual, train_margins_actual)
    log_loss_delta = sim_log_loss - naive_log_loss
    return {
        "key_number_mass": {
            str(k): {
                "simulated": sim_mass[k],
                "actual_pooled": actual_mass[k],
                "actual_bootstrap_mean": boot_ci[k][0],
                "actual_bootstrap_p05": boot_ci[k][1],
                "actual_bootstrap_p95": boot_ci[k][2],
                "within_ci": bool(boot_ci[k][1] <= sim_mass[k] <= boot_ci[k][2]),
            }
            for k in KEY_NUMBERS
        },
        "key_number_hits_of_5": hits,
        "discrete_log_loss": {
            "simulator": sim_log_loss,
            "naive_train_histogram": naive_log_loss,
            "delta_sim_minus_naive": log_loss_delta,
        },
        "go_no_go": "GO" if bool(hits >= 4 and log_loss_delta <= 0.02) else "NO_GO",
    }


def run_validation() -> dict:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    orig_dir = u1b.PBP_SNAPSHOT_DIR
    u1b.PBP_SNAPSHOT_DIR = PBP_SNAPSHOT_DIR
    train_pbp = u1b.load_seasons(VALIDATION_TRAIN_SEASONS)
    u1b.PBP_SNAPSHOT_DIR = orig_dir

    train_drives_to, dropped = reconstruct_drives_to(train_pbp)

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

    opening_fp_pool = u1b.build_opening_fp(train_drives_to)
    transition_pools = u1b.build_transition_pools(train_drives_to)
    level0, level1, level2, level3 = u1b.build_state_cells(train_drives_to)

    results = {}
    for name, config in CONFIGS.items():
        sim_margins, _went_to_ot = run_config(
            config,
            train_drives_to,
            opening_fp_pool,
            transition_pools,
            level0,
            level1,
            level2,
            level3,
            rng,
        )
        metrics = evaluate(
            sim_margins, train_margins_actual, eval_margins_actual, actual_by_season, rng
        )
        results[name] = metrics

    report = {
        "generated_at": timestamp,
        "mode": "validation",
        "train_seasons": list(VALIDATION_TRAIN_SEASONS),
        "eval_seasons": list(VALIDATION_EVAL_SEASONS),
        "n_train_drives": len(train_drives_to),
        "n_train_drives_dropped_missing_state": dropped,
        "n_eval_games_actual": len(eval_margins_actual),
        "configs": {
            name: {
                "use_late": c["use_late"],
                "include_off_to": c["include_off_to"],
                "min_cell_n_late": c["min_cell_n_late"],
            }
            for name, c in CONFIGS.items()
        },
        "results": results,
    }
    with (out_dir / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Artifact directory: {out_dir}")
    return report


def run_test(config_name: str) -> dict:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    orig_dir = u1b.PBP_SNAPSHOT_DIR
    u1b.PBP_SNAPSHOT_DIR = PBP_SNAPSHOT_DIR
    train_pbp = u1b.load_seasons(TEST_TRAIN_SEASONS)
    u1b.PBP_SNAPSHOT_DIR = orig_dir

    train_drives_to, dropped = reconstruct_drives_to(train_pbp)

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

    opening_fp_pool = u1b.build_opening_fp(train_drives_to)
    transition_pools = u1b.build_transition_pools(train_drives_to)
    level0, level1, level2, level3 = u1b.build_state_cells(train_drives_to)

    config = CONFIGS[config_name]
    sim_margins, went_to_ot = run_config(
        config,
        train_drives_to,
        opening_fp_pool,
        transition_pools,
        level0,
        level1,
        level2,
        level3,
        rng,
    )
    metrics = evaluate(
        sim_margins, train_margins_actual, eval_margins_actual, actual_by_season, rng
    )

    report = {
        "generated_at": timestamp,
        "mode": "test",
        "config": config_name,
        "train_seasons": list(TEST_TRAIN_SEASONS),
        "eval_seasons": list(TEST_EVAL_SEASONS),
        "n_train_drives": len(train_drives_to),
        "n_train_drives_dropped_missing_state": dropped,
        "n_eval_games_actual": len(eval_margins_actual),
        "sim_ot_share_of_games": float(np.mean(went_to_ot)),
        "look_note": "4th look at 2018-2025 across the sim04 unit family (Units 1, 1b, 1c, unit 6)",
        "result": metrics,
    }
    with (out_dir / "test_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    np.save(out_dir / "sim_margins.npy", sim_margins)
    print(json.dumps(report, indent=2))
    print(f"Artifact directory: {out_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validation", "test"], default="validation")
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="A")
    args = parser.parse_args()
    if args.mode == "validation":
        run_validation()
    else:
        run_test(args.config)


if __name__ == "__main__":
    main()
