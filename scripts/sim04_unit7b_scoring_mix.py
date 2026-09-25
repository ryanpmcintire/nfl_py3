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

REPO = Path(__file__).resolve().parents[1]
GAME_FEATURES_PATH = REPO / "data" / "processed" / "game_features_pbp.parquet"
ARTIFACT_ROOT = REPO / "artifacts" / "sim04_unit7b"

TRAIN_SEASONS = tuple(range(2009, 2015))
VALIDATE_SEASONS = (2015, 2016, 2017)
N_GAMES = 10_000
RNG_SEED = 20260925
CONFIG_B = {"use_timeouts": False, "min_cell_n": 25}

BUCKETS = ["tied", "1-3", "4-7", "8+"]
SCORE_TYPES = ["FG", "TD+PAT", "TD+2pt", "safety", "defensive_or_return_TD"]


def bucket_of(diff: float) -> str:
    a = abs(diff)
    if a == 0:
        return "tied"
    if a <= 3:
        return "1-3"
    if a <= 7:
        return "4-7"
    return "8+"


def classify_score(category: str, points_off: float, points_def: float) -> str | None:
    if category == "Field goal":
        return "FG"
    if category == "Touchdown":
        return "TD+2pt" if round(points_off) == 8 else "TD+PAT"
    if category == "Opp touchdown":
        return "defensive_or_return_TD"
    if category == "Safety":
        return "safety"
    return None


def simulate_diagnostics(
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
) -> list[dict]:
    def draw_next_fp(category: str) -> float:
        pool = transition_pools.get(category)
        if pool is None or len(pool) == 0:
            pool = opening_fp_pool
        return float(rng.choice(pool))

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
                    min_cell_n,
                    use_timeouts,
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
                    categories, poff, pdef, _durs = u1b.draw_cell(
                        diff, qtr, gsr, start_fp, level0, level1, level2, level3
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
            else:
                categories, poff, pdef, durs = u1b.draw_cell(
                    diff, qtr, gsr, start_fp, level0, level1, level2, level3
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

        went_to_ot = False
        ot_ending = None
        if score["home"] == score["away"]:
            went_to_ot = True
            offense = "home" if rng.random() < 0.5 else "away"
            start_fp = float(rng.choice(opening_fp_pool))
            ot_clock = u1b.OT_SECONDS
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
                categories, poff, pdef, durs = u1b.draw_cell(
                    diff, 5, 0.0, start_fp, level0, level1, level2, level3
                )
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


def actual_diagnostics(drives: pd.DataFrame, game_features: pd.DataFrame) -> tuple[list[dict], int]:
    gf = game_features.set_index("game_id")
    games: list[dict] = []
    skipped = 0
    for game_id, group in drives.groupby("game_id", sort=False):
        if game_id not in gf.index:
            skipped += 1
            continue
        row = gf.loc[game_id]
        home_team = row["home_team"]
        away_team = row["away_team"]
        final_margin = float(row["home_score"] - row["away_score"])
        running: dict[str, float] = {}
        checkpoint_diff = None
        post_drives: list[tuple] = []
        found = False
        for r in group.itertuples(index=False):
            running.setdefault(r.offense, 0.0)
            running.setdefault(r.defense, 0.0)
            if not found and int(r.start_qtr) == 4 and float(r.start_gsr) <= 300.0:
                found = True
                checkpoint_diff = running.get(home_team, 0.0) - running.get(away_team, 0.0)
            if found:
                post_drives.append(
                    (r.category, float(r.points_off), float(r.points_def), float(r.start_fp))
                )
            running[r.offense] = running[r.offense] + float(r.points_off)
            running[r.defense] = running[r.defense] + float(r.points_def)
        if not found:
            skipped += 1
            continue
        went_to_ot = bool((group["start_qtr"] >= 5).any())
        ot_ending = None
        if went_to_ot:
            last = post_drives[-1] if post_drives else None
            if final_margin == 0:
                ot_ending = "tie"
            elif last is not None:
                cat = last[0]
                if cat == "Field goal":
                    ot_ending = "FG"
                elif cat in ("Touchdown", "Opp touchdown"):
                    ot_ending = "TD"
                elif cat == "Safety":
                    ot_ending = "safety"
                else:
                    ot_ending = "other"
        games.append(
            {
                "checkpoint_diff": checkpoint_diff,
                "final_margin": final_margin,
                "went_to_ot": went_to_ot,
                "ot_ending": ot_ending,
                "post_drives": post_drives,
            }
        )
    return games, skipped


def pmf_dict(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {}
    vals = np.clip(np.round(values), -30, 40)
    uniq, counts = np.unique(vals, return_counts=True)
    total = counts.sum()
    return {str(int(u)): float(c / total) for u, c in zip(uniq, counts, strict=True)}


def aggregate(games: list[dict]) -> dict:
    out = {}
    for b in BUCKETS:
        sub = [
            g
            for g in games
            if g["checkpoint_diff"] is not None and bucket_of(g["checkpoint_diff"]) == b
        ]
        n = len(sub)
        margin_changes = np.array(
            [g["final_margin"] - g["checkpoint_diff"] for g in sub], dtype=float
        )
        score_counts = dict.fromkeys(SCORE_TYPES, 0)
        score_counts["other_nonscoring"] = 0
        per_game_score_n = []
        for g in sub:
            cnt = 0
            for cat, poff, pdef, _fp in g["post_drives"]:
                label = classify_score(cat, poff, pdef)
                if label is not None:
                    score_counts[label] += 1
                    cnt += 1
                else:
                    score_counts["other_nonscoring"] += 1
            per_game_score_n.append(cnt)
        is_m3 = [abs(g["final_margin"]) == 3 for g in sub]
        is_ot = [g["went_to_ot"] for g in sub]
        n_ot = sum(is_ot)
        ot_end_counts: dict[str, int] = {}
        for g in sub:
            if g["went_to_ot"]:
                k = g["ot_ending"] or "unknown"
                ot_end_counts[k] = ot_end_counts.get(k, 0) + 1
        total_scores = sum(score_counts.values())
        out[b] = {
            "n_games": n,
            "mean_scoring_plays_after_5min": float(np.mean(per_game_score_n)) if n else None,
            "scoring_type_counts": score_counts,
            "scoring_type_share": {
                k: v / total_scores if total_scores else None for k, v in score_counts.items()
            },
            "margin_change_mean": float(margin_changes.mean()) if n else None,
            "margin_change_sd": float(margin_changes.std()) if n else None,
            "margin_change_pmf": pmf_dict(margin_changes),
            "ot_rate": n_ot / n if n else None,
            "ot_ending_counts": ot_end_counts,
            "p_fg_given_ot": ot_end_counts.get("FG", 0) / n_ot if n_ot else None,
            "p_final_margin_eq3": float(np.mean(is_m3)) if n else None,
            "p_final_margin_eq3_via_ot": (
                sum(1 for m, o in zip(is_m3, is_ot, strict=True) if m and o) / n if n else None
            ),
            "p_final_margin_eq3_via_regulation": (
                sum(1 for m, o in zip(is_m3, is_ot, strict=True) if m and not o) / n if n else None
            ),
        }
    return out


def fp_table(games: list[dict]) -> dict:
    buckets = {
        i: {"n": 0, "fg_attempt": 0, "go_for_it_failed": 0, "td": 0, "other": 0}
        for i in range(u1b.N_FP_BUCKETS)
    }
    for g in games:
        for cat, _poff, _pdef, fp in g["post_drives"]:
            b = u1b.fp_bucket(fp)
            d = buckets[b]
            d["n"] += 1
            if cat in ("Field goal", "Missed field goal"):
                d["fg_attempt"] += 1
            elif cat == "Turnover on downs":
                d["go_for_it_failed"] += 1
            elif cat == "Touchdown":
                d["td"] += 1
            else:
                d["other"] += 1
    out = {}
    for k, v in buckets.items():
        n = v["n"]
        out[str(k)] = {
            **v,
            "fg_attempt_rate": v["fg_attempt"] / n if n else None,
            "go_for_it_failed_rate": v["go_for_it_failed"] / n if n else None,
            "td_rate": v["td"] / n if n else None,
        }
    return out


def main() -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    train_pbp = u7.load_reg_pbp(TRAIN_SEASONS)
    train_drives, dropped = u1b.reconstruct_drives(train_pbp)
    opening_fp_pool = u1b.build_opening_fp(train_drives)
    transition_pools = u1b.build_transition_pools(train_drives)
    level0, level1, level2, level3 = u1b.build_state_cells(train_drives)

    play_rows = u7.build_play_rows(train_pbp)
    race_l0, race_l1, race_l2, race_floor = u7.build_race_pools(play_rows, CONFIG_B["use_timeouts"])

    sim_games = simulate_diagnostics(
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
        CONFIG_B["min_cell_n"],
        CONFIG_B["use_timeouts"],
    )

    validate_pbp = u7.load_reg_pbp(VALIDATE_SEASONS)
    validate_drives, validate_dropped = u1b.reconstruct_drives(validate_pbp)
    game_features = pd.read_parquet(GAME_FEATURES_PATH)
    game_features = game_features[
        game_features["season"].isin(VALIDATE_SEASONS) & (game_features["game_type"] == "REG")
    ].copy()
    actual_games, actual_skipped = actual_diagnostics(validate_drives, game_features)

    sim_agg = aggregate(sim_games)
    actual_agg = aggregate(actual_games)
    sim_fp = fp_table(sim_games)
    actual_fp = fp_table(actual_games)

    tied_gap = actual_agg["tied"]["p_final_margin_eq3"] - sim_agg["tied"]["p_final_margin_eq3"]
    tied_gap_ot = (
        actual_agg["tied"]["p_final_margin_eq3_via_ot"]
        - sim_agg["tied"]["p_final_margin_eq3_via_ot"]
    )
    tied_gap_reg = (
        actual_agg["tied"]["p_final_margin_eq3_via_regulation"]
        - sim_agg["tied"]["p_final_margin_eq3_via_regulation"]
    )

    report = {
        "generated_at": timestamp,
        "train_seasons": list(TRAIN_SEASONS),
        "validate_seasons": list(VALIDATE_SEASONS),
        "config": "B",
        "n_simulated_games": N_GAMES,
        "n_train_drives": len(train_drives),
        "n_train_drives_dropped": dropped,
        "n_validate_drives": len(validate_drives),
        "n_validate_drives_dropped": validate_dropped,
        "n_actual_validate_games": len(actual_games),
        "actual_games_skipped_no_checkpoint": actual_skipped,
        "by_state_bucket": {"simulated": sim_agg, "actual": actual_agg},
        "field_position_table_post_5min_drives": {"simulated": sim_fp, "actual": actual_fp},
        "tied_bucket_decomposition": {
            "p_margin3_actual": actual_agg["tied"]["p_final_margin_eq3"],
            "p_margin3_sim": sim_agg["tied"]["p_final_margin_eq3"],
            "gap_total": tied_gap,
            "gap_from_ot_path": tied_gap_ot,
            "gap_from_regulation_path": tied_gap_reg,
            "ot_rate_actual": actual_agg["tied"]["ot_rate"],
            "ot_rate_sim": sim_agg["tied"]["ot_rate"],
            "p_fg_given_ot_actual": actual_agg["tied"]["p_fg_given_ot"],
            "p_fg_given_ot_sim": sim_agg["tied"]["p_fg_given_ot"],
        },
    }

    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Artifact directory: {out_dir}")


if __name__ == "__main__":
    main()
