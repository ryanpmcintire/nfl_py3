from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sim04_unit1b_state_chain as u1b

REPO = Path(__file__).resolve().parents[1]
NEW_SNAPSHOT_DIR = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
GAME_FEATURES_PATH = REPO / "data" / "processed" / "game_features_pbp.parquet"
ARTIFACT_ROOT = REPO / "artifacts" / "sim04_unit7"

VALID_TRAIN_SEASONS = tuple(range(2009, 2015))
VALID_EVAL_SEASONS = (2015, 2016, 2017)
FULL_TRAIN_SEASONS = tuple(range(2009, 2018))
TEST_SEASONS = tuple(range(2018, 2026))

KEY_NUMBERS = u1b.KEY_NUMBERS
N_GAMES = u1b.N_GAMES
RNG_SEED = u1b.RNG_SEED
N_BOOT = u1b.N_BOOT
OT_SECONDS = u1b.OT_SECONDS
MAX_OT_POSSESSIONS = u1b.MAX_OT_POSSESSIONS

WINDOW_SECONDS = 300.0
N_SL_BUCKETS = 5
MAX_RACE_PLAYS = 30

CONFIGS = {
    "A": {"use_timeouts": True, "min_cell_n": 25},
    "B": {"use_timeouts": False, "min_cell_n": 25},
    "C": {"use_timeouts": True, "min_cell_n": 15},
}


def to_bucket(value: float) -> int:
    return int(min(3, max(0, value)))


def sl_bucket(seconds_left: float) -> int:
    clipped = min(WINDOW_SECONDS, max(0.0, seconds_left))
    return int(min(N_SL_BUCKETS - 1, clipped // 60.0))


def load_reg_pbp(seasons: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = NEW_SNAPSHOT_DIR / f"season={season}" / "plays.parquet"
        frame = pd.read_parquet(path)
        frame = frame[frame["season_type"] == "REG"].copy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_play_rows(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    rows = []
    for _game_id, game in pbp.groupby("game_id", sort=False):
        for _drive_id, drive in game.groupby("fixed_drive", sort=False):
            if pd.isna(drive["fixed_drive"].iloc[0]):
                continue
            drive = drive.reset_index(drop=True)
            selected = []
            for i in range(len(drive)):
                play = drive.iloc[i]
                if play["play_type"] in ("kickoff", "extra_point"):
                    continue
                if pd.isna(play["qtr"]) or pd.isna(play["game_seconds_remaining"]):
                    continue
                qtr = int(play["qtr"])
                gsr = float(play["game_seconds_remaining"])
                if qtr not in (2, 4):
                    continue
                seconds_left = u1b.seconds_left_in_period(qtr, gsr)
                if seconds_left > WINDOW_SECONDS or seconds_left < 0:
                    continue
                if pd.isna(play["score_differential"]):
                    continue
                if pd.isna(play["posteam_timeouts_remaining"]) or pd.isna(
                    play["defteam_timeouts_remaining"]
                ):
                    continue
                selected.append((i, play, qtr, gsr, seconds_left))
            for j, (i, play, qtr, gsr, seconds_left) in enumerate(selected):
                if play["qb_kneel"] == 1:
                    label = "kneel"
                elif play["qb_spike"] == 1:
                    label = "spike"
                elif play["play_type"] == "field_goal":
                    label = "field_goal"
                elif play["play_type"] == "punt":
                    label = "punt"
                elif play["play_type"] == "run":
                    label = "run"
                elif play["play_type"] == "pass":
                    label = "pass"
                elif play["play_type"] == "no_play":
                    label = "no_play"
                else:
                    continue
                if j + 1 < len(selected):
                    next_gsr = float(selected[j + 1][1]["game_seconds_remaining"])
                    elapsed = gsr - next_gsr
                else:
                    boundary = 1800.0 if qtr == 2 else 0.0
                    elapsed = gsr - boundary
                elapsed = float(min(WINDOW_SECONDS, max(0.0, elapsed)))
                is_last_of_drive = i == len(drive) - 1
                is_terminal = bool(
                    label in ("punt", "field_goal")
                    or play["touchdown"] == 1
                    or play["interception"] == 1
                    or play["fumble_lost"] == 1
                    or is_last_of_drive
                )
                own_to_before = to_bucket(play["posteam_timeouts_remaining"])
                def_to_before = to_bucket(play["defteam_timeouts_remaining"])
                if j + 1 < len(selected):
                    nxt = selected[j + 1][1]
                    own_to_after = to_bucket(nxt["posteam_timeouts_remaining"])
                    def_to_after = to_bucket(nxt["defteam_timeouts_remaining"])
                else:
                    own_to_after = own_to_before
                    def_to_after = def_to_before
                own_delta = max(0, own_to_before - own_to_after)
                def_delta = max(0, def_to_before - def_to_after)
                rows.append(
                    (
                        label,
                        elapsed,
                        is_terminal,
                        own_delta,
                        def_delta,
                        u1b.score_bucket_fine(float(play["score_differential"])),
                        u1b.score_bucket_coarse(float(play["score_differential"])),
                        sl_bucket(seconds_left),
                        own_to_before,
                        def_to_before,
                    )
                )
    return pd.DataFrame(
        rows,
        columns=[
            "label",
            "elapsed",
            "is_terminal",
            "own_delta",
            "def_delta",
            "sb_fine",
            "sb_coarse",
            "sl_bucket",
            "own_to",
            "def_to",
        ],
    )


def build_race_pools(play_rows: pd.DataFrame, use_timeouts: bool) -> tuple[dict, dict, dict, tuple]:
    sb_fine = play_rows["sb_fine"].to_numpy()
    sb_coarse = play_rows["sb_coarse"].to_numpy()
    sl = play_rows["sl_bucket"].to_numpy()
    own_to = play_rows["own_to"].to_numpy()
    def_to = play_rows["def_to"].to_numpy()

    fields = ("label", "elapsed", "is_terminal", "own_delta", "def_delta")
    arrays = {f: play_rows[f].to_numpy() for f in fields}

    def make_pool(mask: np.ndarray) -> tuple:
        return tuple(arrays[f][mask] for f in fields)

    l0: dict = {}
    l1: dict = {}
    if use_timeouts:
        for key in set(
            zip(sb_fine.tolist(), sl.tolist(), own_to.tolist(), def_to.tolist(), strict=True)
        ):
            mask = (sb_fine == key[0]) & (sl == key[1]) & (own_to == key[2]) & (def_to == key[3])
            l0[key] = make_pool(mask)
        for key in set(zip(sb_fine.tolist(), sl.tolist(), own_to.tolist(), strict=True)):
            mask = (sb_fine == key[0]) & (sl == key[1]) & (own_to == key[2])
            l1[key] = make_pool(mask)
    else:
        for key in set(zip(sb_fine.tolist(), sl.tolist(), strict=True)):
            mask = (sb_fine == key[0]) & (sl == key[1])
            l0[key] = make_pool(mask)
        l1 = l0

    l2: dict = {}
    for key in set(zip(sb_coarse.tolist(), sl.tolist(), strict=True)):
        mask = (sb_coarse == key[0]) & (sl == key[1])
        l2[key] = make_pool(mask)

    floor = make_pool(np.ones(len(play_rows), dtype=bool))
    return l0, l1, l2, floor


def draw_race_row(
    sb_f: int,
    sb_c: int,
    sl_b: int,
    own_b: int,
    def_b: int,
    l0: dict,
    l1: dict,
    l2: dict,
    floor: tuple,
    min_cell_n: int,
    use_timeouts: bool,
    rng: np.random.Generator,
) -> tuple[str, float, bool, int, int]:
    if use_timeouts:
        pool = l0.get((sb_f, sl_b, own_b, def_b))
        if pool is None or len(pool[0]) < min_cell_n:
            pool = l1.get((sb_f, sl_b, own_b))
        if pool is None or len(pool[0]) < min_cell_n:
            pool = l2.get((sb_c, sl_b))
    else:
        pool = l0.get((sb_f, sl_b))
        if pool is None or len(pool[0]) < min_cell_n:
            pool = l2.get((sb_c, sl_b))
    if pool is None or len(pool[0]) == 0:
        pool = floor
    idx = int(rng.integers(0, len(pool[0])))
    return (
        pool[0][idx],
        float(pool[1][idx]),
        bool(pool[2][idx]),
        int(pool[3][idx]),
        int(pool[4][idx]),
    )


def run_race(
    diff: float,
    qtr: int,
    gsr: float,
    own_timeouts: int,
    def_timeouts: int,
    l0: dict,
    l1: dict,
    l2: dict,
    floor: tuple,
    min_cell_n: int,
    use_timeouts: bool,
    rng: np.random.Generator,
) -> tuple[bool, float, int, int]:
    remaining = u1b.seconds_left_in_period(qtr, gsr)
    sb_f = u1b.score_bucket_fine(diff)
    sb_c = u1b.score_bucket_coarse(diff)
    own_b = to_bucket(own_timeouts)
    def_b = to_bucket(def_timeouts)
    consumed = 0.0
    own_used = 0
    def_used = 0
    for _ in range(MAX_RACE_PLAYS):
        sl_b = sl_bucket(remaining - consumed)
        _label, elapsed, is_terminal, own_delta, def_delta = draw_race_row(
            sb_f, sb_c, sl_b, own_b, def_b, l0, l1, l2, floor, min_cell_n, use_timeouts, rng
        )
        if consumed + elapsed >= remaining:
            return True, remaining, own_used, def_used
        consumed += elapsed
        own_used += own_delta
        def_used += def_delta
        if is_terminal:
            return False, consumed, own_used, def_used
    return True, remaining, own_used, def_used


def simulate_games_u7(
    n_games: int,
    rng: np.random.Generator,
    opening_fp_pool: np.ndarray,
    transition_pools: dict,
    level0: dict,
    level1: dict,
    level2: dict,
    level3: list,
    race_l0: dict,
    race_l1: dict,
    race_l2: dict,
    race_floor: tuple,
    min_cell_n: int,
    use_timeouts: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    margins = np.empty(n_games, dtype=float)
    went_to_ot = np.zeros(n_games, dtype=bool)
    clock_expired_count = np.zeros(n_games, dtype=int)
    final_drive_expired = np.zeros(n_games, dtype=bool)

    def draw_next_fp(category: str) -> float:
        next_pool = transition_pools.get(category)
        if next_pool is None or len(next_pool) == 0:
            next_pool = opening_fp_pool
        return float(rng.choice(next_pool))

    for game_index in range(n_games):
        offense = "home" if rng.random() < 0.5 else "away"
        score = {"home": 0.0, "away": 0.0}
        timeouts = {"home": 3, "away": 3}
        start_fp = float(rng.choice(opening_fp_pool))
        gsr = 3600.0
        second_half_reset_done = False
        n_expired = 0
        last_drive_expired = False

        while gsr > 0:
            defense = "away" if offense == "home" else "home"
            elapsed_game = 3600.0 - gsr
            qtr = min(4, int(elapsed_game // 900) + 1)
            if qtr >= 3 and not second_half_reset_done:
                timeouts = {"home": 3, "away": 3}
                second_half_reset_done = True
            diff = score[offense] - score[defense]
            seconds_left = u1b.seconds_left_in_period(qtr, gsr)
            triggered = qtr in (2, 4) and seconds_left <= WINDOW_SECONDS
            if triggered:
                clock_expired, duration, own_used, def_used = run_race(
                    diff,
                    qtr,
                    gsr,
                    timeouts[offense],
                    timeouts[defense],
                    race_l0,
                    race_l1,
                    race_l2,
                    race_floor,
                    min_cell_n,
                    use_timeouts,
                    rng,
                )
                timeouts[offense] = max(0, timeouts[offense] - own_used)
                timeouts[defense] = max(0, timeouts[defense] - def_used)
                last_drive_expired = clock_expired
                if clock_expired:
                    n_expired += 1
                    next_fp = start_fp
                else:
                    categories, points_off, points_def, _durations = u1b.draw_cell(
                        diff, qtr, gsr, start_fp, level0, level1, level2, level3
                    )
                    draw = rng.integers(0, len(categories))
                    score[offense] += float(points_off[draw])
                    score[defense] += float(points_def[draw])
                    next_fp = draw_next_fp(categories[draw])
            else:
                last_drive_expired = False
                categories, points_off, points_def, durations = u1b.draw_cell(
                    diff, qtr, gsr, start_fp, level0, level1, level2, level3
                )
                draw = rng.integers(0, len(categories))
                score[offense] += float(points_off[draw])
                score[defense] += float(points_def[draw])
                duration = float(durations[draw])
                next_fp = draw_next_fp(categories[draw])
            gsr = max(0.0, gsr - duration)
            start_fp = next_fp
            offense = defense

        clock_expired_count[game_index] = n_expired
        final_drive_expired[game_index] = last_drive_expired

        if score["home"] == score["away"]:
            went_to_ot[game_index] = True
            final_drive_expired[game_index] = False
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
                categories, points_off, points_def, durations = u1b.draw_cell(
                    diff, 5, 0.0, start_fp, level0, level1, level2, level3
                )
                draw = rng.integers(0, len(categories))
                score[offense] += float(points_off[draw])
                score[defense] += float(points_def[draw])
                duration = float(durations[draw])
                next_fp = draw_next_fp(categories[draw])
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
    return margins, went_to_ot, clock_expired_count, final_drive_expired


def actual_margins(seasons: tuple[int, ...], reg_only: bool) -> np.ndarray:
    game_features = pd.read_parquet(GAME_FEATURES_PATH)
    subset = game_features[game_features["season"].isin(seasons)].copy()
    if reg_only:
        subset = subset[subset["game_type"] == "REG"]
    subset["margin"] = subset["home_score"] - subset["away_score"]
    return subset["margin"].to_numpy()


def actual_by_season(seasons: tuple[int, ...], reg_only: bool) -> dict[int, np.ndarray]:
    game_features = pd.read_parquet(GAME_FEATURES_PATH)
    subset = game_features[game_features["season"].isin(seasons)].copy()
    if reg_only:
        subset = subset[subset["game_type"] == "REG"]
    subset["margin"] = subset["home_score"] - subset["away_score"]
    return {int(s): g["margin"].to_numpy() for s, g in subset.groupby("season")}


def final_drive_share(final_drive_expired: np.ndarray) -> float:
    return float(np.mean(final_drive_expired))


def run_config(
    config_name: str,
    train_seasons: tuple[int, ...],
    eval_seasons: tuple[int, ...],
    rng_seed: int,
) -> dict:
    cfg = CONFIGS[config_name]
    rng = np.random.default_rng(rng_seed)

    train_pbp = load_reg_pbp(train_seasons)
    train_drives, dropped = u1b.reconstruct_drives(train_pbp)
    opening_fp_pool = u1b.build_opening_fp(train_drives)
    transition_pools = u1b.build_transition_pools(train_drives)
    level0, level1, level2, level3 = u1b.build_state_cells(train_drives)

    play_rows = build_play_rows(train_pbp)
    race_l0, race_l1, race_l2, race_floor = build_race_pools(play_rows, cfg["use_timeouts"])

    sim_margins, went_to_ot, clock_expired_count, final_drive_expired = simulate_games_u7(
        N_GAMES,
        rng,
        opening_fp_pool,
        transition_pools,
        level0,
        level1,
        level2,
        level3,
        race_l0,
        race_l1,
        race_l2,
        race_floor,
        cfg["min_cell_n"],
        cfg["use_timeouts"],
    )

    eval_margins = actual_margins(eval_seasons, reg_only=True)
    train_margins = actual_margins(train_seasons, reg_only=True)
    eval_by_season = actual_by_season(eval_seasons, reg_only=True)

    sim_mass = u1b.key_number_mass(sim_margins, KEY_NUMBERS)
    boot_ci = u1b.bootstrap_season_ci(eval_by_season, KEY_NUMBERS, n_boot=N_BOOT, rng=rng)
    actual_mass = u1b.key_number_mass(eval_margins, KEY_NUMBERS)

    hits = sum(1 for k in KEY_NUMBERS if boot_ci[k][1] <= sim_mass[k] <= boot_ci[k][2])
    sim_log_loss = u1b.discrete_log_loss(eval_margins, sim_margins)
    naive_log_loss = u1b.discrete_log_loss(eval_margins, train_margins)
    log_loss_delta = sim_log_loss - naive_log_loss

    return {
        "config": config_name,
        "n_train_drives": len(train_drives),
        "n_train_drives_dropped_missing_state": dropped,
        "n_play_rows": len(play_rows),
        "n_race_l0_cells": len(race_l0),
        "n_race_l1_cells": len(race_l1),
        "n_race_l2_cells": len(race_l2),
        "n_eval_games_actual": len(eval_margins),
        "key_number_mass": {
            str(k): {
                "simulated": sim_mass[k],
                "actual_pooled": actual_mass[k],
                "actual_bootstrap_p05": boot_ci[k][1],
                "actual_bootstrap_p95": boot_ci[k][2],
                "in_ci": bool(boot_ci[k][1] <= sim_mass[k] <= boot_ci[k][2]),
            }
            for k in KEY_NUMBERS
        },
        "key_number_hits_of_5": hits,
        "discrete_log_loss_delta": log_loss_delta,
        "sim_log_loss": sim_log_loss,
        "naive_log_loss": naive_log_loss,
        "sim_margin_mean": float(np.mean(sim_margins)),
        "sim_margin_std": float(np.std(sim_margins)),
        "actual_margin_mean": float(np.mean(eval_margins)),
        "actual_margin_std": float(np.std(eval_margins)),
        "sim_ot_share": float(np.mean(went_to_ot)),
        "final_drive_clock_expired_share": final_drive_share(final_drive_expired),
        "mean_clock_expirations_per_game": float(np.mean(clock_expired_count)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Unit 7 play-level clock model")
    parser.add_argument("--mode", choices=["validation", "test"], required=True)
    parser.add_argument("--config", choices=list(CONFIGS), default=None)
    args = parser.parse_args()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "validation":
        results = {
            name: run_config(name, VALID_TRAIN_SEASONS, VALID_EVAL_SEASONS, RNG_SEED)
            for name in CONFIGS
        }
        report = {
            "generated_at": timestamp,
            "mode": "validation",
            "train_seasons": list(VALID_TRAIN_SEASONS),
            "eval_seasons": list(VALID_EVAL_SEASONS),
            "results": results,
        }
    else:
        config_name = args.config or "A"
        result = run_config(config_name, FULL_TRAIN_SEASONS, TEST_SEASONS, RNG_SEED)
        go_decision = bool(
            result["key_number_hits_of_5"] >= 4 and result["discrete_log_loss_delta"] <= 0.02
        )
        old_actual = actual_margins(TEST_SEASONS, reg_only=False)
        old_mass = u1b.key_number_mass(old_actual, KEY_NUMBERS)
        new_actual = actual_margins(TEST_SEASONS, reg_only=True)
        new_mass = u1b.key_number_mass(new_actual, KEY_NUMBERS)
        report = {
            "generated_at": timestamp,
            "mode": "test",
            "config": config_name,
            "train_seasons": list(FULL_TRAIN_SEASONS),
            "test_seasons": list(TEST_SEASONS),
            "result": result,
            "go_no_go": "GO" if go_decision else "NO_GO",
            "reg_only_fix_effect": {
                "n_games_season_only": len(old_actual),
                "n_games_reg_only": len(new_actual),
                "key_number_mass_season_only": {str(k): old_mass[k] for k in KEY_NUMBERS},
                "key_number_mass_reg_only": {str(k): new_mass[k] for k in KEY_NUMBERS},
            },
        }

    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Artifact directory: {out_dir}")


if __name__ == "__main__":
    main()
