from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

PBP_SNAPSHOT_DIR = REPO / "data" / "pbp" / "raw" / "20260817T184927Z"
GAME_FEATURES_PATH = REPO / "data" / "processed" / "game_features_pbp.parquet"
ARTIFACT_ROOT = REPO / "artifacts" / "sim04_unit1"

TRAIN_SEASONS = tuple(range(2009, 2018))
TEST_SEASONS = tuple(range(2018, 2026))

KEY_NUMBERS = (3, 7, 10, 14, 17)
FP_BIN_EDGES = np.array(list(range(0, 101, 10)), dtype=float)
N_BUCKETS = len(FP_BIN_EDGES) - 1
N_GAMES = 10_000
MAX_OT_DRIVES = 12
RNG_SEED = 20260925
N_BOOT = 2000


def bucket_index(fp: float) -> int:
    idx = int(np.digitize([min(max(fp, 0.0), 99.999)], FP_BIN_EDGES)[0]) - 1
    return min(max(idx, 0), N_BUCKETS - 1)


def load_seasons(seasons: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = PBP_SNAPSHOT_DIR / f"season={season}" / "plays.parquet"
        frame = pd.read_parquet(path)
        frame = frame[frame["season_type"] == "REG"].copy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def reconstruct_drives(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    rows = []
    for game_id, game in pbp.groupby("game_id", sort=False):
        home_team = game["home_team"].iloc[0]
        away_team = game["away_team"].iloc[0]
        running = {home_team: 0.0, away_team: 0.0}
        drive_start_score: dict[str, float] | None = None
        current_drive = None
        drive_offense = None
        drive_defense = None
        drive_start_fp = None
        drive_category = None
        for play in game.itertuples(index=False):
            drive_id = play.fixed_drive
            if pd.isna(drive_id):
                continue
            if drive_id != current_drive:
                if (
                    current_drive is not None
                    and drive_start_fp is not None
                    and drive_offense is not None
                    and drive_defense is not None
                ):
                    rows.append(
                        (
                            game_id,
                            play.season,
                            drive_offense,
                            drive_defense,
                            drive_start_fp,
                            drive_category,
                            running[drive_offense] - drive_start_score[drive_offense],
                            running[drive_defense] - drive_start_score[drive_defense],
                        )
                    )
                current_drive = drive_id
                drive_offense = None
                drive_defense = None
                drive_start_fp = None
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
            posteam = play.posteam
            score_post = play.posteam_score_post
            if pd.notna(posteam) and pd.notna(score_post) and posteam in running:
                running[posteam] = score_post
        if (
            current_drive is not None
            and drive_start_fp is not None
            and drive_offense is not None
            and drive_defense is not None
        ):
            rows.append(
                (
                    game_id,
                    play.season,
                    drive_offense,
                    drive_defense,
                    drive_start_fp,
                    drive_category,
                    running[drive_offense] - drive_start_score[drive_offense],
                    running[drive_defense] - drive_start_score[drive_defense],
                )
            )
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
        ],
    )
    return drives


def build_opening_fp(drives: pd.DataFrame) -> np.ndarray:
    firsts = drives.groupby("game_id", sort=False).first()
    return firsts["start_fp"].to_numpy()


def build_transition_pools(drives: pd.DataFrame) -> dict[str, np.ndarray]:
    pools: dict[str, list[float]] = {}
    for _, game in drives.groupby("game_id", sort=False):
        game = game.reset_index(drop=True)
        for i in range(len(game) - 1):
            category = game.loc[i, "category"]
            pools.setdefault(category, []).append(game.loc[i + 1, "start_fp"])
    return {key: np.array(values, dtype=float) for key, values in pools.items()}


def build_outcome_pools(drives: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    buckets = [bucket_index(v) for v in drives["start_fp"].to_numpy()]
    drives = drives.assign(bucket=buckets)
    pools = []
    for b in range(N_BUCKETS):
        group = drives[drives["bucket"] == b]
        pools.append(
            (
                group["category"].to_numpy(),
                group["points_off"].to_numpy(dtype=float),
                group["points_def"].to_numpy(dtype=float),
            )
        )
    return pools


def drives_per_game_counts(drives: pd.DataFrame) -> np.ndarray:
    return drives.groupby("game_id", sort=False).size().to_numpy()


def simulate_games(
    n_games: int,
    rng: np.random.Generator,
    opening_fp_pool: np.ndarray,
    transition_pools: dict[str, np.ndarray],
    outcome_pools: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    game_drive_counts: np.ndarray,
) -> np.ndarray:
    margins = np.empty(n_games, dtype=float)
    for game_index in range(n_games):
        total_drives = int(rng.choice(game_drive_counts))
        offense = "home" if rng.random() < 0.5 else "away"
        score = {"home": 0.0, "away": 0.0}
        start_fp = float(rng.choice(opening_fp_pool))
        drives_played = 0
        while True:
            regulation_done = drives_played >= total_drives
            tied = score["home"] == score["away"]
            if regulation_done and not tied:
                break
            if drives_played >= total_drives + MAX_OT_DRIVES:
                break
            categories, points_off, points_def = outcome_pools[bucket_index(start_fp)]
            draw = rng.integers(0, len(categories))
            category = categories[draw]
            defense = "away" if offense == "home" else "home"
            score[offense] += points_off[draw]
            score[defense] += points_def[draw]
            next_pool = transition_pools.get(category)
            if next_pool is None or len(next_pool) == 0:
                next_pool = opening_fp_pool
            start_fp = float(rng.choice(next_pool))
            offense = defense
            drives_played += 1
        margins[game_index] = score["home"] - score["away"]
    return margins


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


def main() -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    train_pbp = load_seasons(TRAIN_SEASONS)
    train_drives = reconstruct_drives(train_pbp)

    game_features = pd.read_parquet(GAME_FEATURES_PATH)
    game_features = game_features[game_features["season"].isin(TRAIN_SEASONS + TEST_SEASONS)].copy()
    game_features["margin"] = game_features["home_score"] - game_features["away_score"]

    train_margins_actual = game_features[game_features["season"].isin(TRAIN_SEASONS)][
        "margin"
    ].to_numpy()
    test_margins_actual = game_features[game_features["season"].isin(TEST_SEASONS)][
        "margin"
    ].to_numpy()

    opening_fp_pool = build_opening_fp(train_drives)
    transition_pools = build_transition_pools(train_drives)
    outcome_pools = build_outcome_pools(train_drives)
    game_drive_counts = drives_per_game_counts(train_drives)

    sim_margins = simulate_games(
        N_GAMES, rng, opening_fp_pool, transition_pools, outcome_pools, game_drive_counts
    )

    sim_mass = key_number_mass(sim_margins, KEY_NUMBERS)
    actual_by_season = {
        int(season): group["margin"].to_numpy()
        for season, group in game_features[game_features["season"].isin(TEST_SEASONS)].groupby(
            "season"
        )
    }
    boot_ci = bootstrap_season_ci(actual_by_season, KEY_NUMBERS, n_boot=N_BOOT, rng=rng)
    actual_test_mass = key_number_mass(test_margins_actual, KEY_NUMBERS)

    hits = sum(1 for k in KEY_NUMBERS if boot_ci[k][1] <= sim_mass[k] <= boot_ci[k][2])

    sim_log_loss = discrete_log_loss(test_margins_actual, sim_margins)
    naive_log_loss = discrete_log_loss(test_margins_actual, train_margins_actual)
    log_loss_delta = sim_log_loss - naive_log_loss

    go_decision = bool(hits >= 4 and log_loss_delta <= 0.02)

    report = {
        "generated_at": timestamp,
        "train_seasons": list(TRAIN_SEASONS),
        "test_seasons": list(TEST_SEASONS),
        "n_simulated_games": N_GAMES,
        "n_train_drives": len(train_drives),
        "n_train_games_actual": len(train_margins_actual),
        "n_test_games_actual": len(test_margins_actual),
        "key_number_mass": {
            str(k): {
                "simulated": sim_mass[k],
                "actual_test_pooled": actual_test_mass[k],
                "actual_test_bootstrap_mean": boot_ci[k][0],
                "actual_test_bootstrap_p05": boot_ci[k][1],
                "actual_test_bootstrap_p95": boot_ci[k][2],
                "simulated_within_ci": bool(boot_ci[k][1] <= sim_mass[k] <= boot_ci[k][2]),
            }
            for k in KEY_NUMBERS
        },
        "key_number_hits_of_5": hits,
        "discrete_log_loss": {
            "simulator_vs_actual_test": sim_log_loss,
            "naive_train_histogram_vs_actual_test": naive_log_loss,
            "delta_sim_minus_naive": log_loss_delta,
        },
        "predeclared_criterion": {
            "min_key_number_hits": 4,
            "max_log_loss_delta": 0.02,
        },
        "go_no_go": "GO" if go_decision else "NO_GO",
        "sim_margin_mean": float(np.mean(sim_margins)),
        "sim_margin_std": float(np.std(sim_margins)),
        "actual_test_margin_mean": float(np.mean(test_margins_actual)),
        "actual_test_margin_std": float(np.std(test_margins_actual)),
    }

    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    np.save(out_dir / "sim_margins.npy", sim_margins)

    print(json.dumps(report, indent=2))
    print(f"Artifact directory: {out_dir}")


if __name__ == "__main__":
    main()
