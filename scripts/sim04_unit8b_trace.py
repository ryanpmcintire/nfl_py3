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
import sim04_unit7_clock as u7
import sim04_unit7b_scoring_mix as u7b
import sim04_unit8_cells as u8

REPO = Path(__file__).resolve().parents[1]
GAME_FEATURES_PATH = REPO / "data" / "processed" / "game_features_pbp.parquet"
ARTIFACT_ROOT = REPO / "artifacts" / "sim04_unit8b_trace"

TRAIN_SEASONS = tuple(range(2009, 2015))
VALIDATE_SEASONS = (2015, 2016, 2017)
N_GAMES = 10_000
RNG_SEED = 20260925
RACE_CONFIG_B = {"use_timeouts": False, "min_cell_n": 25}
CONFIG_C = {"min_cell_n": 25, "use_q4_level": False, "use_score_level": False}
N_EXAMPLE_TRACES = 3


def draw_cell2_traced(
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
) -> tuple[tuple, str, tuple | None]:
    sb_f = u1b.score_bucket_fine(diff)
    sb_c = u1b.score_bucket_coarse(diff)
    tb_f = u1b.time_bucket_fine(qtr, gsr)
    tb_c = u8.time_bucket_coarse2(tb_f)
    fb = u1b.fp_bucket(fp)

    pool = level0.get((sb_f, tb_f, fb))
    if pool is not None and len(pool[0]) >= min_cell_n:
        return pool, "level0", (sb_f, tb_f, fb)
    if use_q4_level and tb_f in u8.Q4_ENDGAME_FINE:
        pool = level_q4.get((sb_f, tb_f))
        if pool is not None and len(pool[0]) >= min_cell_n:
            return pool, "level_q4", (sb_f, tb_f)
    pool = level1.get((sb_f, tb_c, fb))
    if pool is not None and len(pool[0]) >= min_cell_n:
        return pool, "level1", (sb_f, tb_c, fb)
    pool = level2.get((sb_c, tb_c, fb))
    if pool is not None and len(pool[0]) >= min_cell_n:
        return pool, "level2", (sb_c, tb_c, fb)
    if use_score_level:
        pool = level_score.get(sb_f)
        if pool is not None and len(pool[0]) >= min_cell_n:
            return pool, "level_score", (sb_f,)
    return level3, "level3", None


def build_play_rows_fixed(pbp: pd.DataFrame) -> pd.DataFrame:
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
        for j in range(len(game_selected)):
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
            if j + 1 < len(game_selected):
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


def simulate_traced(
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
    n_example_traces: int,
) -> tuple[list[dict], list[dict], int, list[dict]]:
    def draw_next_fp(category: str) -> float:
        pool = transition_pools.get(category)
        if pool is None or len(pool) == 0:
            pool = opening_fp_pool
        return float(rng.choice(pool))

    def draw_cell_here(
        diff: float, qtr: int, gsr: float, fp: float
    ) -> tuple[tuple, str, tuple | None]:
        return draw_cell2_traced(
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

    sim_games: list[dict] = []
    tied_possessions: list[dict] = []
    n_tied_games = 0
    example_traces: list[dict] = []

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

            trace_row = None
            if triggered:
                start_clock = seconds_left
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
                    cell_level = None
                    cell_key = None
                else:
                    (categories, poff, pdef, _durs), cell_level, cell_key = draw_cell_here(
                        diff, qtr, gsr, start_fp
                    )
                    draw = rng.integers(0, len(categories))
                    category = categories[draw]
                    points_off = float(poff[draw])
                    points_def = float(pdef[draw])
                    score[offense] += points_off
                    score[defense] += points_def
                    next_fp = draw_next_fp(category)
                if qtr == 4:
                    post_drives.append((category, points_off, points_def, start_fp))
                    trace_row = {
                        "seq": possession_seq,
                        "offense": offense,
                        "start_clock_s": start_clock,
                        "path": "race",
                        "race_clock_expired": bool(clock_expired),
                        "cell_level": cell_level,
                        "cell_key": list(cell_key) if cell_key is not None else None,
                        "duration_s": duration,
                        "category": category,
                        "points_off": points_off,
                        "points_def": points_def,
                    }
                    possession_seq += 1
            else:
                (categories, poff, pdef, durs), _lvl, _key = draw_cell_here(
                    diff, qtr, gsr, start_fp
                )
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

            if trace_row is not None:
                q4_trace.append(trace_row)

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
                (categories, poff, pdef, durs), _lvl, _key = draw_cell_here(diff, 5, 0.0, start_fp)
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
            n_tied_games += 1
            for row in q4_trace:
                row["game_idx"] = game_idx
            tied_possessions.extend(q4_trace)
            if len(example_traces) < n_example_traces:
                example_traces.append(
                    {
                        "game_idx": game_idx,
                        "checkpoint_diff": checkpoint_diff,
                        "final_margin": final_margin,
                        "went_to_ot": went_to_ot,
                        "possessions": q4_trace,
                    }
                )

    return sim_games, tied_possessions, n_tied_games, example_traces


def actual_tied_possessions(
    drives: pd.DataFrame, game_features: pd.DataFrame
) -> tuple[list[dict], int, int]:
    gf = game_features.set_index("game_id")
    possessions: list[dict] = []
    n_tied_games = 0
    n_total_games = 0
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
        for r in group.itertuples(index=False):
            running.setdefault(r.offense, 0.0)
            running.setdefault(r.defense, 0.0)
            if not found and int(r.start_qtr) == 4 and float(r.start_gsr) <= 300.0:
                found = True
                checkpoint_diff = running.get(home_team, 0.0) - running.get(away_team, 0.0)
            if found and int(r.start_qtr) == 4:
                game_possessions.append(
                    {
                        "game_id": game_id,
                        "offense": r.offense,
                        "start_clock_s": u1b.seconds_left_in_period(
                            int(r.start_qtr), float(r.start_gsr)
                        ),
                        "duration_s": float(r.duration),
                        "category": r.category,
                        "points_off": float(r.points_off),
                        "points_def": float(r.points_def),
                    }
                )
            running[r.offense] = running[r.offense] + float(r.points_off)
            running[r.defense] = running[r.defense] + float(r.points_def)
        if not found:
            continue
        n_total_games += 1
        if checkpoint_diff == 0.0:
            n_tied_games += 1
            possessions.extend(game_possessions)
    return possessions, n_tied_games, n_total_games


def summarize(possessions: list[dict], n_games: int) -> dict:
    n = len(possessions)
    if n == 0:
        return {"n_possessions": 0, "n_games": n_games}
    durations = np.array([p["duration_s"] for p in possessions], dtype=float)
    scoring = np.array([(p["points_off"] > 0 or p["points_def"] > 0) for p in possessions])
    cats: dict[str, int] = {}
    for p in possessions:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    key = "game_idx" if "game_idx" in possessions[0] else "game_id"
    counts: dict = {}
    for p in possessions:
        counts[p[key]] = counts.get(p[key], 0) + 1
    per_game = np.array(list(counts.values()), dtype=float)
    n_zero_possession_games = n_games - len(counts)
    per_game_full = np.concatenate([per_game, np.zeros(n_zero_possession_games)])
    return {
        "n_games": n_games,
        "n_possessions": n,
        "possessions_per_game_mean": float(per_game_full.mean()),
        "possessions_per_game_hist": {
            str(k): int(v)
            for k, v in zip(*np.unique(per_game_full.astype(int), return_counts=True), strict=True)
        },
        "mean_duration_s": float(durations.mean()),
        "median_duration_s": float(np.median(durations)),
        "scoring_rate_per_possession": float(scoring.mean()),
        "category_counts": cats,
    }


def run(apply_fix: bool) -> dict:
    rng = np.random.default_rng(RNG_SEED)

    train_pbp = u7.load_reg_pbp(TRAIN_SEASONS)
    train_drives, _dropped = u1b.reconstruct_drives(train_pbp)
    opening_fp_pool = u1b.build_opening_fp(train_drives)
    transition_pools = u1b.build_transition_pools(train_drives)
    level0, level_q4, level1, level2, level_score, level3 = u8.build_state_cells2(train_drives)

    play_rows_fn = build_play_rows_fixed if apply_fix else u7.build_play_rows
    play_rows = play_rows_fn(train_pbp)
    race_l0, race_l1, race_l2, race_floor = u7.build_race_pools(
        play_rows, RACE_CONFIG_B["use_timeouts"]
    )

    ot_options = u8.ot_seconds_options(VALIDATE_SEASONS)

    sim_games, tied_possessions, n_tied_games_sim, example_traces = simulate_traced(
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
        CONFIG_C["min_cell_n"],
        CONFIG_C["use_q4_level"],
        CONFIG_C["use_score_level"],
        N_EXAMPLE_TRACES,
    )

    validate_pbp = u7.load_reg_pbp(VALIDATE_SEASONS)
    validate_drives, _validate_dropped = u1b.reconstruct_drives(validate_pbp)
    game_features = pd.read_parquet(GAME_FEATURES_PATH)
    game_features_eval = game_features[
        game_features["season"].isin(VALIDATE_SEASONS) & (game_features["game_type"] == "REG")
    ].copy()
    actual_possessions, n_tied_games_actual, n_total_games_actual = actual_tied_possessions(
        validate_drives, game_features_eval
    )

    sim_summary = summarize(tied_possessions, n_tied_games_sim)
    actual_summary = summarize(actual_possessions, n_tied_games_actual)

    sim_agg = u7b.aggregate(sim_games)
    actual_games_full, _skipped = u7b.actual_diagnostics(validate_drives, game_features_eval)
    actual_agg = u7b.aggregate(actual_games_full)

    train_margins = u7.actual_margins(TRAIN_SEASONS, reg_only=True)
    eval_margins = u7.actual_margins(VALIDATE_SEASONS, reg_only=True)
    eval_by_season = u7.actual_by_season(VALIDATE_SEASONS, reg_only=True)
    sim_margins = np.array([g["final_margin"] for g in sim_games], dtype=float)
    sim_mass = u1b.key_number_mass(sim_margins, u1b.KEY_NUMBERS)
    boot_ci = u1b.bootstrap_season_ci(eval_by_season, u1b.KEY_NUMBERS, n_boot=u1b.N_BOOT, rng=rng)
    hits = sum(1 for k in u1b.KEY_NUMBERS if boot_ci[k][1] <= sim_mass[k] <= boot_ci[k][2])
    sim_log_loss = u1b.discrete_log_loss(eval_margins, sim_margins)
    naive_log_loss = u1b.discrete_log_loss(eval_margins, train_margins)

    return {
        "apply_fix": apply_fix,
        "n_tied_at_5min_games_sim": n_tied_games_sim,
        "n_tied_at_5min_games_actual": n_tied_games_actual,
        "n_total_games_actual": n_total_games_actual,
        "sim_possession_summary": sim_summary,
        "actual_possession_summary": actual_summary,
        "sim_ot_rate_tied_bucket": sim_agg["tied"]["ot_rate"],
        "actual_ot_rate_tied_bucket": actual_agg["tied"]["ot_rate"],
        "sim_p_margin3_via_regulation_tied_bucket": sim_agg["tied"][
            "p_final_margin_eq3_via_regulation"
        ],
        "actual_p_margin3_via_regulation_tied_bucket": actual_agg["tied"][
            "p_final_margin_eq3_via_regulation"
        ],
        "key_number_hits_of_5": hits,
        "sim_log_loss": sim_log_loss,
        "naive_log_loss": naive_log_loss,
        "discrete_log_loss_delta": sim_log_loss - naive_log_loss,
        "example_traces": example_traces,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.both:
        report = {"before_fix": run(False), "after_fix": run(True)}
    else:
        report = {"result": run(args.fix)}

    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Artifact directory: {out_dir}")


if __name__ == "__main__":
    main()
