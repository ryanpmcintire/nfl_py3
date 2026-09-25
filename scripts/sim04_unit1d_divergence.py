from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "sim04_unit1b_state_chain", REPO / "scripts" / "sim04_unit1b_state_chain.py"
)
unit1b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(unit1b)

GAME_FEATURES_PATH = REPO / "data" / "processed" / "game_features_pbp.parquet"
ARTIFACT_ROOT = REPO / "artifacts" / "sim04_unit1d"

TRAIN_SEASONS = tuple(range(2009, 2015))
VALIDATE_SEASONS = tuple(range(2015, 2018))

N_GAMES = 10_000
RNG_SEED = 20260925
N_BOOT = 1000

CHECKPOINTS = [
    ("end_q1", 2700.0),
    ("half", 1800.0),
    ("end_q3", 900.0),
    ("q4_5min", 300.0),
    ("q4_2min", 120.0),
]

BUCKET_LABELS = {
    0: "home_trail_ge9",
    1: "home_trail_1to8",
    2: "tied",
    3: "home_lead_1to8",
    4: "home_lead_ge9",
}


def simulate_with_diagnostics(
    n_games: int,
    rng: np.random.Generator,
    opening_fp_pool: np.ndarray,
    transition_pools: dict[str, np.ndarray],
    level0: dict,
    level1: dict,
    level2: dict,
    level3: list,
    checkpoints: list[tuple[str, float]],
) -> dict:
    margins = np.empty(n_games, dtype=float)
    total_points = np.empty(n_games, dtype=float)
    n_drives_total = np.empty(n_games, dtype=int)
    went_to_ot = np.zeros(n_games, dtype=bool)
    checkpoint_diffs = {name: np.empty(n_games, dtype=float) for name, _ in checkpoints}
    last_category = [None] * n_games
    drive_bucket_all: list[int] = []
    drive_points_all: list[float] = []

    def draw_next_fp(category: str) -> float:
        pool = transition_pools.get(category)
        if pool is None or len(pool) == 0:
            pool = opening_fp_pool
        return float(rng.choice(pool))

    def apply_drive(
        offense: str, defense: str, diff: float, qtr: int, gsr: float, fp: float, score: dict
    ) -> tuple[str, float, float]:
        categories, points_off, points_def, durations = unit1b.draw_cell(
            diff, qtr, gsr, fp, level0, level1, level2, level3
        )
        draw = rng.integers(0, len(categories))
        score[offense] += float(points_off[draw])
        score[defense] += float(points_def[draw])
        duration = float(durations[draw])
        next_fp = draw_next_fp(categories[draw])
        return categories[draw], duration, next_fp

    for g in range(n_games):
        offense = "home" if rng.random() < 0.5 else "away"
        score = {"home": 0.0, "away": 0.0}
        start_fp = float(rng.choice(opening_fp_pool))
        gsr = 3600.0
        remaining = list(checkpoints)
        n_drives = 0
        last_cat = None

        while gsr > 0:
            defense = "away" if offense == "home" else "home"
            elapsed = 3600.0 - gsr
            qtr = min(4, int(elapsed // 900) + 1)
            diff = score[offense] - score[defense]
            gsr_start = gsr
            score_before = dict(score)
            category, duration, next_fp = apply_drive(
                offense, defense, diff, qtr, gsr, start_fp, score
            )
            gsr_end = max(0.0, gsr_start - duration)
            while remaining and gsr_end < remaining[0][1] <= gsr_start:
                name, _ = remaining.pop(0)
                checkpoint_diffs[name][g] = score_before["home"] - score_before["away"]
            drive_bucket_all.append(unit1b.score_bucket_coarse(diff))
            drive_points_all.append(score[offense] - score_before[offense])
            n_drives += 1
            last_cat = category
            gsr = gsr_end
            start_fp = next_fp
            offense = defense

        for name, _ in remaining:
            checkpoint_diffs[name][g] = score["home"] - score["away"]

        if score["home"] == score["away"]:
            went_to_ot[g] = True
            offense = "home" if rng.random() < 0.5 else "away"
            start_fp = float(rng.choice(opening_fp_pool))
            ot_clock = unit1b.OT_SECONDS
            possessions = {"home": 0, "away": 0}
            settled = False
            for _ in range(unit1b.MAX_OT_POSSESSIONS):
                if ot_clock <= 0 or settled:
                    break
                defense = "away" if offense == "home" else "home"
                diff = score[offense] - score[defense]
                off_before = score[offense]
                def_before = score[defense]
                category, duration, next_fp = apply_drive(
                    offense, defense, diff, 5, 0.0, start_fp, score
                )
                off_gain = score[offense] - off_before
                def_gain = score[defense] - def_before
                ot_clock = max(0.0, ot_clock - duration)
                possessions[offense] += 1
                n_drives += 1
                last_cat = category
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

        margins[g] = score["home"] - score["away"]
        total_points[g] = score["home"] + score["away"]
        n_drives_total[g] = n_drives
        last_category[g] = last_cat

    return {
        "margins": margins,
        "total_points": total_points,
        "n_drives_total": n_drives_total,
        "went_to_ot": went_to_ot,
        "checkpoint_diffs": checkpoint_diffs,
        "last_category": np.array(last_category, dtype=object),
        "drive_bucket": np.array(drive_bucket_all, dtype=int),
        "drive_points": np.array(drive_points_all, dtype=float),
    }


def compute_actual_checkpoints(
    drives: pd.DataFrame, checkpoints: list[tuple[str, float]]
) -> pd.DataFrame:
    rows = []
    for game_id, game in drives.sort_values(
        ["game_id", "start_gsr"], ascending=[True, False]
    ).groupby("game_id", sort=False):
        game = game.reset_index(drop=True)
        n = len(game)
        running: dict[str, float] = {}
        remaining = list(checkpoints)
        checkpoint_scores: dict[str, dict[str, float]] = {}
        for i in range(n):
            row = game.iloc[i]
            running.setdefault(row["offense"], 0.0)
            running.setdefault(row["defense"], 0.0)
            gsr_start = float(row["start_gsr"])
            if i + 1 < n:
                next_start = float(game.iloc[i + 1]["start_gsr"])
                gsr_end = next_start if next_start < gsr_start else 0.0
            else:
                gsr_end = 0.0
            score_before = dict(running)
            running[row["offense"]] = running[row["offense"]] + float(row["points_off"])
            running[row["defense"]] = running[row["defense"]] + float(row["points_def"])
            while remaining and gsr_end < remaining[0][1] <= gsr_start:
                name, _ = remaining.pop(0)
                checkpoint_scores[name] = score_before
        for name, _ in remaining:
            checkpoint_scores[name] = running
        record = {"game_id": game_id, "n_drives": n, "last_category": game.iloc[-1]["category"]}
        for name, _ in checkpoints:
            record[f"cp_scores__{name}"] = checkpoint_scores.get(name, running)
        rows.append(record)
    return pd.DataFrame(rows)


def bucket_shares(bucket_ids: np.ndarray, n_buckets: int = 5) -> np.ndarray:
    counts = np.bincount(bucket_ids, minlength=n_buckets)
    return counts / counts.sum()


def conditional_margin3_rate(
    bucket_ids: np.ndarray, margins: np.ndarray, n_buckets: int = 5
) -> np.ndarray:
    rates = np.zeros(n_buckets, dtype=float)
    is3 = (np.abs(margins) == 3).astype(float)
    for b in range(n_buckets):
        mask = bucket_ids == b
        rates[b] = float(is3[mask].mean()) if mask.sum() > 0 else 0.0
    return rates


def decomposition(
    actual_ck_bucket: np.ndarray,
    actual_margins: np.ndarray,
    sim_ck_bucket: np.ndarray,
    sim_margins: np.ndarray,
) -> dict:
    a_ck = bucket_shares(actual_ck_bucket)
    s_ck = bucket_shares(sim_ck_bucket)
    a_tm = conditional_margin3_rate(actual_ck_bucket, actual_margins)
    s_tm = conditional_margin3_rate(sim_ck_bucket, sim_margins)
    return {
        "actual_checkpoint_shares": a_ck,
        "sim_checkpoint_shares": s_ck,
        "actual_transition_p3": a_tm,
        "sim_transition_p3": s_tm,
        "p_actual_direct": float(np.dot(a_ck, a_tm)),
        "p_sim_direct": float(np.dot(s_ck, s_tm)),
        "hybrid_actualCk_simTm": float(np.dot(a_ck, s_tm)),
        "hybrid_simCk_actualTm": float(np.dot(s_ck, a_tm)),
    }


def bootstrap_decomposition(
    actual_ck_bucket: np.ndarray,
    actual_margins: np.ndarray,
    sim_ck_bucket: np.ndarray,
    sim_margins: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    keys = ["p_actual_direct", "p_sim_direct", "hybrid_actualCk_simTm", "hybrid_simCk_actualTm"]
    draws = {k: [] for k in keys}
    n_a = len(actual_ck_bucket)
    n_s = len(sim_ck_bucket)
    for _ in range(n_boot):
        idx_a = rng.integers(0, n_a, n_a)
        idx_s = rng.integers(0, n_s, n_s)
        d = decomposition(
            actual_ck_bucket[idx_a], actual_margins[idx_a], sim_ck_bucket[idx_s], sim_margins[idx_s]
        )
        for k in keys:
            draws[k].append(d[k])
    out = {}
    for k in keys:
        arr = np.array(draws[k])
        out[k] = {
            "mean": float(np.mean(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }
    return out


def checkpoint_summary(values: np.ndarray) -> dict:
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "share_abs_le3": float(np.mean(np.abs(values) <= 3)),
        "share_abs_le8": float(np.mean(np.abs(values) <= 8)),
    }


def category_shares(categories: np.ndarray, mask: np.ndarray) -> dict:
    sub = categories[mask]
    n = len(sub)
    if n == 0:
        return {"n": 0}
    counts: dict[str, int] = {}
    for c in sub:
        key = c if isinstance(c, str) else "missing"
        counts[key] = counts.get(key, 0) + 1
    return {"n": n, "shares": {k: v / n for k, v in counts.items()}}


def drive_scoring_rate_by_bucket(
    bucket_ids: np.ndarray, points: np.ndarray, n_buckets: int = 5
) -> dict:
    out = {}
    for b in range(n_buckets):
        mask = bucket_ids == b
        n = int(mask.sum())
        out[BUCKET_LABELS[b]] = {
            "n_drives": n,
            "mean_points_off": float(points[mask].mean()) if n else None,
            "p_scored": float((points[mask] > 0).mean()) if n else None,
        }
    return out


def main() -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    train_pbp = unit1b.load_seasons(TRAIN_SEASONS)
    train_drives, train_dropped = unit1b.reconstruct_drives(train_pbp)

    validate_pbp = unit1b.load_seasons(VALIDATE_SEASONS)
    validate_drives, validate_dropped = unit1b.reconstruct_drives(validate_pbp)

    opening_fp_pool = unit1b.build_opening_fp(train_drives)
    transition_pools = unit1b.build_transition_pools(train_drives)
    level0, level1, level2, level3 = unit1b.build_state_cells(train_drives)

    sim = simulate_with_diagnostics(
        N_GAMES, rng, opening_fp_pool, transition_pools, level0, level1, level2, level3, CHECKPOINTS
    )

    game_features = pd.read_parquet(GAME_FEATURES_PATH)
    game_features = game_features[game_features["season"].isin(VALIDATE_SEASONS)].copy()
    game_features["margin"] = game_features["home_score"] - game_features["away_score"]
    game_features["total_points"] = game_features["home_score"] + game_features["away_score"]
    home_away = game_features.set_index("game_id")[
        ["home_team", "away_team", "margin", "total_points"]
    ]

    actual_cp = compute_actual_checkpoints(validate_drives, CHECKPOINTS)
    actual_cp = actual_cp.merge(home_away, left_on="game_id", right_index=True, how="inner")

    actual_checkpoint_diffs = {}
    for name, _ in CHECKPOINTS:
        col = f"cp_scores__{name}"
        diffs = []
        for _, row in actual_cp.iterrows():
            scores = row[col]
            diffs.append(scores.get(row["home_team"], 0.0) - scores.get(row["away_team"], 0.0))
        actual_checkpoint_diffs[name] = np.array(diffs, dtype=float)

    actual_margins = actual_cp["margin"].to_numpy(dtype=float)
    actual_total_points = actual_cp["total_points"].to_numpy(dtype=float)
    actual_n_drives = actual_cp["n_drives"].to_numpy(dtype=int)
    actual_last_category = actual_cp["last_category"].to_numpy(dtype=object)

    checkpoint_report = {}
    for name, _ in CHECKPOINTS:
        checkpoint_report[name] = {
            "simulated": checkpoint_summary(sim["checkpoint_diffs"][name]),
            "actual": checkpoint_summary(actual_checkpoint_diffs[name]),
        }

    actual_ck_bucket = np.array(
        [unit1b.score_bucket_coarse(d) for d in actual_checkpoint_diffs["q4_5min"]], dtype=int
    )
    sim_ck_bucket = np.array(
        [unit1b.score_bucket_coarse(d) for d in sim["checkpoint_diffs"]["q4_5min"]], dtype=int
    )

    decomp_point = decomposition(actual_ck_bucket, actual_margins, sim_ck_bucket, sim["margins"])
    decomp_boot = bootstrap_decomposition(
        actual_ck_bucket, actual_margins, sim_ck_bucket, sim["margins"], N_BOOT, rng
    )

    transition_table = {}
    for b in range(5):
        label = BUCKET_LABELS[b]
        a_mask = actual_ck_bucket == b
        s_mask = sim_ck_bucket == b
        transition_table[label] = {
            "actual": {
                "n": int(a_mask.sum()),
                "mean_final_margin": float(actual_margins[a_mask].mean()) if a_mask.sum() else None,
                "sd_final_margin": float(actual_margins[a_mask].std()) if a_mask.sum() else None,
                "p_margin3": float((np.abs(actual_margins[a_mask]) == 3).mean())
                if a_mask.sum()
                else None,
            },
            "sim": {
                "n": int(s_mask.sum()),
                "mean_final_margin": float(sim["margins"][s_mask].mean()) if s_mask.sum() else None,
                "sd_final_margin": float(sim["margins"][s_mask].std()) if s_mask.sum() else None,
                "p_margin3": float((np.abs(sim["margins"][s_mask]) == 3).mean())
                if s_mask.sum()
                else None,
            },
        }

    actual_margin3_mask = np.abs(actual_margins) == 3
    sim_margin3_mask = np.abs(sim["margins"]) == 3

    last_play_shares = {
        "actual_all_games": category_shares(
            actual_last_category, np.ones(len(actual_last_category), dtype=bool)
        ),
        "actual_margin3_games": category_shares(actual_last_category, actual_margin3_mask),
        "sim_all_games": category_shares(
            sim["last_category"], np.ones(len(sim["last_category"]), dtype=bool)
        ),
        "sim_margin3_games": category_shares(sim["last_category"], sim_margin3_mask),
    }

    scoring_rate_report = {
        "actual": drive_scoring_rate_by_bucket(
            validate_drives["start_diff"].map(unit1b.score_bucket_coarse).to_numpy(),
            validate_drives["points_off"].to_numpy(dtype=float),
        ),
        "sim": drive_scoring_rate_by_bucket(sim["drive_bucket"], sim["drive_points"]),
    }

    ppg_dpg_report = {
        "actual": {
            "points_per_game": float(actual_total_points.mean()),
            "drives_per_game": float(actual_n_drives.mean()),
        },
        "sim": {
            "points_per_game": float(sim["total_points"].mean()),
            "drives_per_game": float(sim["n_drives_total"].mean()),
        },
    }

    p_actual = decomp_point["p_actual_direct"]
    p_sim = decomp_point["p_sim_direct"]
    hyb_a = decomp_point["hybrid_actualCk_simTm"]
    hyb_b = decomp_point["hybrid_simCk_actualTm"]
    gap = p_actual - p_sim
    reach_explained = (hyb_b - p_sim) / gap if gap != 0 else None
    finish_explained = (p_actual - hyb_a) / gap if gap != 0 else None

    report = {
        "generated_at": timestamp,
        "train_seasons": list(TRAIN_SEASONS),
        "validate_seasons": list(VALIDATE_SEASONS),
        "n_simulated_games": N_GAMES,
        "n_train_drives": len(train_drives),
        "n_train_drives_dropped": train_dropped,
        "n_validate_drives": len(validate_drives),
        "n_validate_drives_dropped": validate_dropped,
        "n_actual_validate_games": len(actual_cp),
        "checkpoint_score_diff_distribution": checkpoint_report,
        "transition_matrix_at_q4_5min": transition_table,
        "decomposition": {
            "point_estimate": {
                k: v for k, v in decomp_point.items() if not isinstance(v, np.ndarray)
            },
            "bootstrap_90pct_ci": decomp_boot,
            "gap_actual_minus_sim": gap,
            "share_of_gap_from_reaching_checkpoint_dist": reach_explained,
            "share_of_gap_from_finishing_transition_matrix": finish_explained,
        },
        "ppg_drives_per_game": ppg_dpg_report,
        "per_drive_scoring_rate_by_score_state": scoring_rate_report,
        "last_scoring_drive_category_shares": last_play_shares,
        "sim_ot_share": float(sim["went_to_ot"].mean()),
        "sim_margin_mean": float(sim["margins"].mean()),
        "sim_margin_sd": float(sim["margins"].std()),
        "actual_margin_mean": float(actual_margins.mean()),
        "actual_margin_sd": float(actual_margins.std()),
    }

    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            indent=2,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o,
        )

    print(
        json.dumps(
            report, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o
        )
    )
    print(f"Artifact directory: {out_dir}")


if __name__ == "__main__":
    main()
