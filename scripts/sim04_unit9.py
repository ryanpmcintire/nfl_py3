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
import sim04_unit8_cells as u8

REPO = Path(__file__).resolve().parents[1]
GAME_FEATURES_PATH = REPO / "data" / "processed" / "game_features_pbp.parquet"
ARTIFACT_ROOT = REPO / "artifacts" / "sim04_unit9"

TRAIN_SEASONS = tuple(range(2009, 2015))
VALIDATE_SEASONS = (2015, 2016, 2017)
N_GAMES = 10_000
RNG_SEED = 20260925

RACE_CONFIG_B = {"use_timeouts": False, "min_cell_n": 25}
CONFIG_C = u8.CONFIGS["C"]


def build_play_rows_qtr_safe(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    rows = []
    for _game_id, game in pbp.groupby("game_id", sort=False):
        game_selected: list[tuple] = []
        for _drive_id, drive in game.groupby("fixed_drive", sort=False):
            if pd.isna(drive["fixed_drive"].iloc[0]):
                continue
            drive = drive.reset_index(drop=True)
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
                if seconds_left > u7.WINDOW_SECONDS or seconds_left < 0:
                    continue
                if pd.isna(play["score_differential"]):
                    continue
                if pd.isna(play["posteam_timeouts_remaining"]) or pd.isna(
                    play["defteam_timeouts_remaining"]
                ):
                    continue
                is_last_of_drive = i == len(drive) - 1
                game_selected.append((play, qtr, gsr, seconds_left, is_last_of_drive))
        n = len(game_selected)
        for j in range(n):
            play, qtr, gsr, seconds_left, is_last_of_drive = game_selected[j]
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
            next_same_qtr = j + 1 < n and game_selected[j + 1][1] == qtr
            if next_same_qtr:
                next_gsr = float(game_selected[j + 1][2])
                elapsed = gsr - next_gsr
            else:
                boundary = 1800.0 if qtr == 2 else 0.0
                elapsed = gsr - boundary
            elapsed = float(min(u7.WINDOW_SECONDS, max(0.0, elapsed)))
            is_terminal = bool(
                label in ("punt", "field_goal")
                or play["touchdown"] == 1
                or play["interception"] == 1
                or play["fumble_lost"] == 1
                or is_last_of_drive
            )
            own_to_before = u7.to_bucket(play["posteam_timeouts_remaining"])
            def_to_before = u7.to_bucket(play["defteam_timeouts_remaining"])
            if next_same_qtr:
                nxt = game_selected[j + 1][0]
                own_to_after = u7.to_bucket(nxt["posteam_timeouts_remaining"])
                def_to_after = u7.to_bucket(nxt["defteam_timeouts_remaining"])
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
                    u7.sl_bucket(seconds_left),
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


def run_race_fixed(
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
    own_b = u7.to_bucket(own_timeouts)
    def_b = u7.to_bucket(def_timeouts)
    consumed = 0.0
    own_used = 0
    def_used = 0
    for _ in range(u7.MAX_RACE_PLAYS):
        sl_b = u7.sl_bucket(remaining - consumed)
        _label, elapsed, is_terminal, own_delta, def_delta = u7.draw_race_row(
            sb_f, sb_c, sl_b, own_b, def_b, l0, l1, l2, floor, min_cell_n, use_timeouts, rng
        )
        if is_terminal:
            consumed = min(consumed + elapsed, remaining)
            own_used += own_delta
            def_used += def_delta
            return False, consumed, own_used, def_used
        if consumed + elapsed >= remaining:
            return True, remaining, own_used, def_used
        consumed += elapsed
        own_used += own_delta
        def_used += def_delta
    return True, remaining, own_used, def_used


def simulate_games_u9(
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
) -> tuple[list[dict], list[dict]]:
    def draw_next_fp(category: str) -> float:
        pool = transition_pools.get(category)
        if pool is None or len(pool) == 0:
            pool = opening_fp_pool
        return float(rng.choice(pool))

    def draw_cell_here(diff: float, qtr: int, gsr: float, fp: float) -> tuple:
        return u8.draw_cell2(
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
            CONFIG_C["min_cell_n"],
            CONFIG_C["use_q4_level"],
            CONFIG_C["use_score_level"],
        )

    sim_games: list[dict] = []
    tied_possessions: list[dict] = []

    for game_idx in range(n_games):
        offense = "home" if rng.random() < 0.5 else "away"
        score = {"home": 0.0, "away": 0.0}
        timeouts = {"home": 3, "away": 3}
        start_fp = float(rng.choice(opening_fp_pool))
        gsr = 3600.0
        second_half_reset_done = False
        checkpoint_diff = None
        post_drives: list[tuple] = []
        q4_trace: list[dict] = []
        possession_seq = 0

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
                start_clock = seconds_left
                clock_expired, duration, own_used, def_used = run_race_fixed(
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
                    q4_trace.append(
                        {
                            "game_idx": game_idx,
                            "seq": possession_seq,
                            "offense": offense,
                            "start_clock_s": start_clock,
                            "race_clock_expired": bool(clock_expired),
                            "duration_s": duration,
                            "category": category,
                            "points_off": points_off,
                            "points_def": points_def,
                        }
                    )
                    possession_seq += 1
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
            for _ in range(u1b.MAX_OT_POSSESSIONS):
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

        final_margin = score["home"] - score["away"]
        sim_games.append(
            {
                "checkpoint_diff": checkpoint_diff,
                "final_margin": final_margin,
                "went_to_ot": went_to_ot,
                "ot_ending": ot_ending,
                "post_drives": post_drives,
            }
        )
        if checkpoint_diff == 0.0:
            tied_possessions.extend(q4_trace)

    return sim_games, tied_possessions


def summarize_possessions(possessions: list[dict], n_tied_games: int) -> dict:
    n = len(possessions)
    if n == 0:
        return {"n_possessions": 0, "n_tied_games": n_tied_games}
    durations = np.array([p["duration_s"] for p in possessions], dtype=float)
    scoring = np.array([(p["points_off"] > 0 or p["points_def"] > 0) for p in possessions])
    expired = np.array([p["category"] == "clock_expired" for p in possessions])
    counts: dict = {}
    for p in possessions:
        counts[p["game_idx"]] = counts.get(p["game_idx"], 0) + 1
    per_game = np.array(list(counts.values()), dtype=float)
    n_zero = n_tied_games - len(counts)
    per_game_full = np.concatenate([per_game, np.zeros(max(0, n_zero))])
    return {
        "n_tied_games": n_tied_games,
        "n_possessions": n,
        "possessions_per_game_mean": float(per_game_full.mean()),
        "mean_duration_s": float(durations.mean()),
        "scoring_rate_per_possession": float(scoring.mean()),
        "clock_expired_share": float(expired.mean()),
    }


def actual_tied_possessions(
    drives: pd.DataFrame, game_features: pd.DataFrame
) -> tuple[list[dict], int]:
    gf = game_features.set_index("game_id")
    possessions: list[dict] = []
    n_tied_games = 0
    for game_id, group in drives.groupby("game_id", sort=False):
        if game_id not in gf.index:
            continue
        row = gf.loc[game_id]
        home_team = row["home_team"]
        away_team = row["away_team"]
        running: dict[str, float] = {}
        checkpoint_diff = None
        found = False
        game_possessions: list[dict] = []
        idx = 0
        for r in group.itertuples(index=False):
            running.setdefault(r.offense, 0.0)
            running.setdefault(r.defense, 0.0)
            if not found and int(r.start_qtr) == 4 and float(r.start_gsr) <= 300.0:
                found = True
                checkpoint_diff = running.get(home_team, 0.0) - running.get(away_team, 0.0)
            if found and int(r.start_qtr) == 4:
                game_possessions.append(
                    {
                        "game_idx": game_id,
                        "duration_s": float(r.duration),
                        "category": r.category,
                        "points_off": float(r.points_off),
                        "points_def": float(r.points_def),
                    }
                )
                idx += 1
            running[r.offense] = running[r.offense] + float(r.points_off)
            running[r.defense] = running[r.defense] + float(r.points_def)
        if not found:
            continue
        if checkpoint_diff == 0.0:
            n_tied_games += 1
            possessions.extend(game_possessions)
    return possessions, n_tied_games


def audit_drive_duration(drives: pd.DataFrame) -> dict:
    out: dict = {}
    for qtr in (2, 4):
        period_ending = []
        other = []
        for _game_id, game in drives.sort_values("start_gsr", ascending=False).groupby(
            "game_id", sort=False
        ):
            qtr_rows = game.index[game["start_qtr"] == qtr].tolist()
            if not qtr_rows:
                continue
            last_idx = qtr_rows[-1]
            for i in qtr_rows:
                dur = float(game.loc[i, "duration"])
                if i == last_idx:
                    period_ending.append(dur)
                else:
                    other.append(dur)
        out[f"qtr{qtr}"] = {
            "n_period_ending": len(period_ending),
            "mean_duration_period_ending": float(np.mean(period_ending)) if period_ending else None,
            "n_other": len(other),
            "mean_duration_other": float(np.mean(other)) if other else None,
        }
    return out


def main() -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    train_pbp = u7.load_reg_pbp(TRAIN_SEASONS)
    train_drives, _dropped = u1b.reconstruct_drives(train_pbp)
    opening_fp_pool = u1b.build_opening_fp(train_drives)
    transition_pools = u1b.build_transition_pools(train_drives)
    level0, level_q4, level1, level2, level_score, level3 = u8.build_state_cells2(train_drives)

    duration_audit = audit_drive_duration(train_drives)

    game_features_all = pd.read_parquet(GAME_FEATURES_PATH)
    gf_train = game_features_all[
        game_features_all["season"].isin(TRAIN_SEASONS) & (game_features_all["game_type"] == "REG")
    ]
    actual_seconds_per_drive = float(
        pd.concat(
            [gf_train["home_drive_seconds_per_drive"], gf_train["away_drive_seconds_per_drive"]]
        ).mean()
    )

    play_rows = build_play_rows_qtr_safe(train_pbp)
    race_l0, race_l1, race_l2, race_floor = u7.build_race_pools(
        play_rows, RACE_CONFIG_B["use_timeouts"]
    )

    ot_options = u8.ot_seconds_options(VALIDATE_SEASONS)

    sim_games, tied_possessions = simulate_games_u9(
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
    )

    validate_pbp = u7.load_reg_pbp(VALIDATE_SEASONS)
    validate_drives, _validate_dropped = u1b.reconstruct_drives(validate_pbp)
    game_features_eval = game_features_all[
        game_features_all["season"].isin(VALIDATE_SEASONS)
        & (game_features_all["game_type"] == "REG")
    ].copy()

    actual_possessions, n_tied_games_actual = actual_tied_possessions(
        validate_drives, game_features_eval
    )
    n_tied_games_sim = sum(1 for g in sim_games if g["checkpoint_diff"] == 0.0)

    sim_summary = summarize_possessions(tied_possessions, n_tied_games_sim)
    actual_summary = summarize_possessions(actual_possessions, n_tied_games_actual)

    sim_agg = u7b.aggregate(sim_games)
    actual_games_full, _skipped = u7b.actual_diagnostics(validate_drives, game_features_eval)
    actual_agg = u7b.aggregate(actual_games_full)

    train_margins = u7.actual_margins(TRAIN_SEASONS, reg_only=True)
    eval_margins = u7.actual_margins(VALIDATE_SEASONS, reg_only=True)
    eval_by_season = u7.actual_by_season(VALIDATE_SEASONS, reg_only=True)
    sim_margins = np.array([g["final_margin"] for g in sim_games], dtype=float)

    sim_mass = u1b.key_number_mass(sim_margins, u1b.KEY_NUMBERS)
    boot_ci = u1b.bootstrap_season_ci(eval_by_season, u1b.KEY_NUMBERS, n_boot=u1b.N_BOOT, rng=rng)
    actual_mass = u1b.key_number_mass(eval_margins, u1b.KEY_NUMBERS)
    hits = sum(1 for k in u1b.KEY_NUMBERS if boot_ci[k][1] <= sim_mass[k] <= boot_ci[k][2])

    sim_log_loss = u1b.discrete_log_loss(eval_margins, sim_margins)
    naive_log_loss = u1b.discrete_log_loss(eval_margins, train_margins)

    go_decision = bool(hits >= 4 and (sim_log_loss - naive_log_loss) <= 0.02)

    report = {
        "generated_at": timestamp,
        "unit": "sim04_unit9",
        "train_seasons": list(TRAIN_SEASONS),
        "validate_seasons": list(VALIDATE_SEASONS),
        "n_games": N_GAMES,
        "config": {"race": RACE_CONFIG_B, "cells": CONFIG_C},
        "drive_duration_audit": duration_audit,
        "actual_mean_drive_seconds_per_drive_train": actual_seconds_per_drive,
        "tied_at_5min_possession_summary": {
            "sim": sim_summary,
            "actual": actual_summary,
        },
        "tied_bucket_ot_and_margin3": {
            "sim_ot_rate": sim_agg["tied"]["ot_rate"],
            "actual_ot_rate": actual_agg["tied"]["ot_rate"],
            "sim_p_margin3_via_regulation": sim_agg["tied"]["p_final_margin_eq3_via_regulation"],
            "actual_p_margin3_via_regulation": actual_agg["tied"][
                "p_final_margin_eq3_via_regulation"
            ],
            "sim_p_margin3": sim_agg["tied"]["p_final_margin_eq3"],
            "actual_p_margin3": actual_agg["tied"]["p_final_margin_eq3"],
        },
        "key_number_mass": {
            str(k): {
                "simulated": sim_mass[k],
                "actual_pooled": actual_mass[k],
                "actual_bootstrap_p05": boot_ci[k][1],
                "actual_bootstrap_p95": boot_ci[k][2],
                "in_ci": bool(boot_ci[k][1] <= sim_mass[k] <= boot_ci[k][2]),
            }
            for k in u1b.KEY_NUMBERS
        },
        "key_number_hits_of_5": hits,
        "sim_log_loss": sim_log_loss,
        "naive_log_loss": naive_log_loss,
        "discrete_log_loss_delta": sim_log_loss - naive_log_loss,
        "sim_margin_mean": float(np.mean(sim_margins)),
        "sim_margin_std": float(np.std(sim_margins)),
        "actual_margin_mean": float(np.mean(eval_margins)),
        "actual_margin_std": float(np.std(eval_margins)),
        "go_no_go": "GO" if go_decision else "NO_GO",
    }

    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Artifact directory: {out_dir}")


if __name__ == "__main__":
    main()
