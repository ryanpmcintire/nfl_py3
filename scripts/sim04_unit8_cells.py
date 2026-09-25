from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sim04_unit1b_state_chain as u1b
import sim04_unit7_clock as u7
import sim04_unit7b_scoring_mix as u7b

REPO = Path(__file__).resolve().parents[1]
GAME_FEATURES_PATH = REPO / "data" / "processed" / "game_features_pbp.parquet"
ARTIFACT_ROOT = REPO / "artifacts" / "sim04_unit8"

TRAIN_SEASONS = tuple(range(2009, 2015))
VALIDATE_SEASONS = (2015, 2016, 2017)
N_GAMES = 10_000
RNG_SEED = 20260925
RACE_CONFIG_B = {"use_timeouts": False, "min_cell_n": 25}
KEY_NUMBERS = u1b.KEY_NUMBERS
N_BOOT = u1b.N_BOOT
MAX_OT_POSSESSIONS = u1b.MAX_OT_POSSESSIONS

Q4_ENDGAME_FINE = (5, 6)

CONFIGS = {
    "A": {"min_cell_n": 25, "use_q4_level": True, "use_score_level": True},
    "B": {"min_cell_n": 15, "use_q4_level": True, "use_score_level": True},
    "C": {"min_cell_n": 25, "use_q4_level": False, "use_score_level": False},
}


def time_bucket_coarse2(tb_fine: int) -> int:
    if tb_fine == 7:
        return 3
    if tb_fine == 2:
        return 1
    if tb_fine in Q4_ENDGAME_FINE:
        return 2
    return 0


def build_state_cells2(drives: pd.DataFrame) -> tuple[dict, dict, dict, dict, dict, tuple]:
    sb_fine = drives["start_diff"].map(u1b.score_bucket_fine).to_numpy()
    sb_coarse = drives["start_diff"].map(u1b.score_bucket_coarse).to_numpy()
    tb_fine = np.array(
        [
            u1b.time_bucket_fine(int(q), g)
            for q, g in zip(drives["start_qtr"], drives["start_gsr"], strict=True)
        ]
    )
    tb_coarse = np.array([time_bucket_coarse2(int(t)) for t in tb_fine])
    fb = drives["start_fp"].map(u1b.fp_bucket).to_numpy()

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

    q4_mask_all = np.isin(tb_fine, Q4_ENDGAME_FINE)
    level_q4: dict[tuple[int, int], tuple] = {}
    for key in set(zip(sb_fine[q4_mask_all].tolist(), tb_fine[q4_mask_all].tolist(), strict=True)):
        mask = (sb_fine == key[0]) & (tb_fine == key[1]) & q4_mask_all
        level_q4[key] = make_pool(mask)

    level1: dict[tuple[int, int, int], tuple] = {}
    for key in set(zip(sb_fine.tolist(), tb_coarse.tolist(), fb.tolist(), strict=True)):
        mask = (sb_fine == key[0]) & (tb_coarse == key[1]) & (fb == key[2])
        level1[key] = make_pool(mask)

    level2: dict[tuple[int, int, int], tuple] = {}
    for key in set(zip(sb_coarse.tolist(), tb_coarse.tolist(), fb.tolist(), strict=True)):
        mask = (sb_coarse == key[0]) & (tb_coarse == key[1]) & (fb == key[2])
        level2[key] = make_pool(mask)

    level_score: dict[int, tuple] = {}
    for key in set(sb_fine.tolist()):
        mask = sb_fine == key
        level_score[key] = make_pool(mask)

    level3 = make_pool(np.ones(len(drives), dtype=bool))

    return level0, level_q4, level1, level2, level_score, level3


def draw_cell2(
    diff: float,
    qtr: int,
    gsr: float,
    fp: float,
    level0: dict,
    level_q4: dict,
    level1: dict,
    level2: dict,
    level_score: dict,
    level3: tuple,
    min_cell_n: int,
    use_q4_level: bool,
    use_score_level: bool,
) -> tuple:
    sb_f = u1b.score_bucket_fine(diff)
    sb_c = u1b.score_bucket_coarse(diff)
    tb_f = u1b.time_bucket_fine(qtr, gsr)
    tb_c = time_bucket_coarse2(tb_f)
    fb = u1b.fp_bucket(fp)

    pool = level0.get((sb_f, tb_f, fb))
    if pool is not None and len(pool[0]) >= min_cell_n:
        return pool
    if use_q4_level and tb_f in Q4_ENDGAME_FINE:
        pool = level_q4.get((sb_f, tb_f))
        if pool is not None and len(pool[0]) >= min_cell_n:
            return pool
    pool = level1.get((sb_f, tb_c, fb))
    if pool is not None and len(pool[0]) >= min_cell_n:
        return pool
    pool = level2.get((sb_c, tb_c, fb))
    if pool is not None and len(pool[0]) >= min_cell_n:
        return pool
    if use_score_level:
        pool = level_score.get(sb_f)
        if pool is not None and len(pool[0]) >= min_cell_n:
            return pool
    return level3


def ot_seconds_options(eval_seasons: tuple[int, ...]) -> np.ndarray:
    return np.array([900.0 if s <= 2016 else 600.0 for s in eval_seasons], dtype=float)


def simulate_diagnostics_u8(
    n_games: int,
    rng: np.random.Generator,
    opening_fp_pool: np.ndarray,
    transition_pools: dict,
    level0: dict,
    level_q4: dict,
    level1: dict,
    level2: dict,
    level_score: dict,
    level3: tuple,
    race_l0: dict,
    race_l1: dict,
    race_l2: dict,
    race_floor: tuple,
    ot_options: np.ndarray,
    min_cell_n: int,
    use_q4_level: bool,
    use_score_level: bool,
) -> list[dict]:
    def draw_next_fp(category: str) -> float:
        pool = transition_pools.get(category)
        if pool is None or len(pool) == 0:
            pool = opening_fp_pool
        return float(rng.choice(pool))

    def draw_cell_here(diff: float, qtr: int, gsr: float, fp: float) -> tuple:
        return draw_cell2(
            diff,
            qtr,
            gsr,
            fp,
            level0,
            level_q4,
            level1,
            level2,
            level_score,
            level3,
            min_cell_n,
            use_q4_level,
            use_score_level,
        )

    games: list[dict] = []
    for _ in range(n_games):
        offense = "home" if rng.random() < 0.5 else "away"
        score = {"home": 0.0, "away": 0.0}
        timeouts = {"home": 3, "away": 3}
        start_fp = float(rng.choice(opening_fp_pool))
        gsr = 3600.0
        second_half_reset_done = False
        checkpoint_diff = None
        post_drives: list[tuple] = []

        while gsr > 0:
            defense = "away" if offense == "home" else "home"
            elapsed_game = 3600.0 - gsr
            qtr = min(4, int(elapsed_game // 900) + 1)
            if qtr >= 3 and not second_half_reset_done:
                timeouts = {"home": 3, "away": 3}
                second_half_reset_done = True
            diff = score[offense] - score[defense]
            seconds_left = u1b.seconds_left_in_period(qtr, gsr)
            triggered = qtr in (2, 4) and seconds_left <= u7.WINDOW_SECONDS
            if qtr == 4 and seconds_left <= u7.WINDOW_SECONDS and checkpoint_diff is None:
                checkpoint_diff = score["home"] - score["away"]
            if triggered:
                clock_expired, duration, own_used, def_used = u7.run_race(
                    diff,
                    qtr,
                    gsr,
                    timeouts[offense],
                    timeouts[defense],
                    race_l0,
                    race_l1,
                    race_l2,
                    race_floor,
                    RACE_CONFIG_B["min_cell_n"],
                    RACE_CONFIG_B["use_timeouts"],
                    rng,
                )
                timeouts[offense] = max(0, timeouts[offense] - own_used)
                timeouts[defense] = max(0, timeouts[defense] - def_used)
                if clock_expired:
                    category = "clock_expired"
                    points_off = 0.0
                    points_def = 0.0
                    next_fp = start_fp
                else:
                    categories, poff, pdef, _durs = draw_cell_here(diff, qtr, gsr, start_fp)
                    draw = rng.integers(0, len(categories))
                    category = categories[draw]
                    points_off = float(poff[draw])
                    points_def = float(pdef[draw])
                    score[offense] += points_off
                    score[defense] += points_def
                    next_fp = draw_next_fp(category)
                if qtr == 4:
                    post_drives.append((category, points_off, points_def, start_fp))
            else:
                categories, poff, pdef, durs = draw_cell_here(diff, qtr, gsr, start_fp)
                draw = rng.integers(0, len(categories))
                category = categories[draw]
                points_off = float(poff[draw])
                points_def = float(pdef[draw])
                score[offense] += points_off
                score[defense] += points_def
                duration = float(durs[draw])
                next_fp = draw_next_fp(category)
            gsr = max(0.0, gsr - duration)
            start_fp = next_fp
            offense = defense

        went_to_ot = False
        ot_ending = None
        if score["home"] == score["away"]:
            went_to_ot = True
            offense = "home" if rng.random() < 0.5 else "away"
            start_fp = float(rng.choice(opening_fp_pool))
            ot_clock = float(rng.choice(ot_options))
            possessions = {"home": 0, "away": 0}
            settled = False
            last_cat = None
            for _ in range(MAX_OT_POSSESSIONS):
                if ot_clock <= 0 or settled:
                    break
                defense = "away" if offense == "home" else "home"
                diff = score[offense] - score[defense]
                off_before = score[offense]
                def_before = score[defense]
                categories, poff, pdef, durs = draw_cell_here(diff, 5, 0.0, start_fp)
                draw = rng.integers(0, len(categories))
                category = categories[draw]
                points_off = float(poff[draw])
                points_def = float(pdef[draw])
                score[offense] += points_off
                score[defense] += points_def
                duration = float(durs[draw])
                next_fp = draw_next_fp(category)
                off_gain = score[offense] - off_before
                def_gain = score[defense] - def_before
                ot_clock = max(0.0, ot_clock - duration)
                possessions[offense] += 1
                start_fp = next_fp
                post_drives.append((category, points_off, points_def, start_fp))
                last_cat = category
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
            if score["home"] == score["away"]:
                ot_ending = "tie"
            elif last_cat == "Field goal":
                ot_ending = "FG"
            elif last_cat in ("Touchdown", "Opp touchdown"):
                ot_ending = "TD"
            elif last_cat == "Safety":
                ot_ending = "safety"
            else:
                ot_ending = "other"

        games.append(
            {
                "checkpoint_diff": checkpoint_diff,
                "final_margin": score["home"] - score["away"],
                "went_to_ot": went_to_ot,
                "ot_ending": ot_ending,
                "post_drives": post_drives,
            }
        )
    return games


def run_config(config_name: str, rng_seed: int) -> dict:
    cfg = CONFIGS[config_name]
    rng = np.random.default_rng(rng_seed)

    train_pbp = u7.load_reg_pbp(TRAIN_SEASONS)
    train_drives, dropped = u1b.reconstruct_drives(train_pbp)
    opening_fp_pool = u1b.build_opening_fp(train_drives)
    transition_pools = u1b.build_transition_pools(train_drives)
    level0, level_q4, level1, level2, level_score, level3 = build_state_cells2(train_drives)

    play_rows = u7.build_play_rows(train_pbp)
    race_l0, race_l1, race_l2, race_floor = u7.build_race_pools(
        play_rows, RACE_CONFIG_B["use_timeouts"]
    )

    ot_options = ot_seconds_options(VALIDATE_SEASONS)

    sim_games = simulate_diagnostics_u8(
        N_GAMES,
        rng,
        opening_fp_pool,
        transition_pools,
        level0,
        level_q4,
        level1,
        level2,
        level_score,
        level3,
        race_l0,
        race_l1,
        race_l2,
        race_floor,
        ot_options,
        cfg["min_cell_n"],
        cfg["use_q4_level"],
        cfg["use_score_level"],
    )

    validate_pbp = u7.load_reg_pbp(VALIDATE_SEASONS)
    validate_drives, validate_dropped = u1b.reconstruct_drives(validate_pbp)
    game_features = pd.read_parquet(GAME_FEATURES_PATH)
    game_features_eval = game_features[
        game_features["season"].isin(VALIDATE_SEASONS) & (game_features["game_type"] == "REG")
    ].copy()
    actual_games, _actual_skipped = u7b.actual_diagnostics(validate_drives, game_features_eval)

    sim_agg = u7b.aggregate(sim_games)
    actual_agg = u7b.aggregate(actual_games)

    sim_margins = np.array([g["final_margin"] for g in sim_games], dtype=float)
    sim_went_to_ot = np.array([g["went_to_ot"] for g in sim_games], dtype=bool)
    sim_ot_ties = sum(1 for g in sim_games if g["went_to_ot"] and g["ot_ending"] == "tie")
    sim_ot_n = int(sim_went_to_ot.sum())

    train_margins = u7.actual_margins(TRAIN_SEASONS, reg_only=True)
    eval_margins = u7.actual_margins(VALIDATE_SEASONS, reg_only=True)
    eval_by_season = u7.actual_by_season(VALIDATE_SEASONS, reg_only=True)

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
        "n_train_drives_dropped": dropped,
        "n_validate_drives": len(validate_drives),
        "n_validate_drives_dropped": validate_dropped,
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
        "sim_ot_share_of_games": float(np.mean(sim_went_to_ot)),
        "sim_ot_tie_rate_overall": sim_ot_ties / sim_ot_n if sim_ot_n else None,
        "n_sim_ot_games": sim_ot_n,
        "n_sim_ot_ties": sim_ot_ties,
        "tied_at_5min_bucket": {
            "actual": actual_agg["tied"],
            "simulated": sim_agg["tied"],
        },
    }


def main() -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {name: run_config(name, RNG_SEED) for name in CONFIGS}

    ranked = sorted(
        results.values(),
        key=lambda r: (-r["key_number_hits_of_5"], r["discrete_log_loss_delta"]),
    )
    baseline_delta = results["A"]["discrete_log_loss_delta"]
    eligible = [r for r in ranked if r["discrete_log_loss_delta"] <= baseline_delta + 0.02]
    selected = eligible[0]["config"] if eligible else ranked[0]["config"]

    report = {
        "generated_at": timestamp,
        "mode": "validation",
        "train_seasons": list(TRAIN_SEASONS),
        "eval_seasons": list(VALIDATE_SEASONS),
        "results": results,
        "selected_config": selected,
    }

    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Artifact directory: {out_dir}")


if __name__ == "__main__":
    main()
