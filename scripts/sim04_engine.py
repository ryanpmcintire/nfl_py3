from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.common import X_DTYPE
from sklearn.neighbors import KDTree

REPO = Path(__file__).resolve().parents[1]
PBP_SNAPSHOT_DIR = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
GAME_FEATURES_PATH = REPO / "data" / "processed" / "game_features_pbp.parquet"
ARTIFACT_ROOT = REPO / "artifacts" / "sim04_engine"

TRAIN_SEASONS = tuple(range(2009, 2015))
VALID_SEASONS = (2015, 2016, 2017)

KEY_NUMBERS = (3, 7, 10, 14, 17)
RNG_SEED = 20260925
N_BOOT = 2000
MIN_CELL_N = 25
MAX_PLAYS_PER_GAME = 400

K_NEIGHBORS = 40
K_STATE = 200
TEAM_KERNEL_LAMBDA = 1_000_000.0
TEAM_KERNEL_H_SCALE = 0.5
TEAM_RATING_YARD_GAIN = 0.6425
SCALE_YDSTOGO = 5.0
SCALE_FP = 2.5
SCALE_TIME = 300.0
SCALE_TIMEOUTS = 1.0
SCORE_CLIP = 21.0
SCORE_INNER = 8.0
SCORE_INNER_SCALE = 2.0
SCORE_OUTER_SCALE = 8.0
LATE_PHASES = (1, 3, 4)
ROUND_DIST = 2.0
ROUND_FP = 5.0
ROUND_SCORE = 2.0
ROUND_TIME = 30.0

OT_SECONDS_BY_SEASON = {2015: 900.0, 2016: 900.0, 2017: 600.0}

LIVE_TYPES = ("run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike", "no_play")
PLAY_TYPE_CODES = {"run": 0, "pass": 1, "punt": 2, "field_goal": 3, "qb_kneel": 4, "qb_spike": 5, "no_play": 6}


def load_reg_seasons(seasons: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = PBP_SNAPSHOT_DIR / f"season={season}" / "plays.parquet"
        frame = pd.read_parquet(path)
        frame = frame[frame["season_type"] == "REG"].copy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def dist_bucket_fine(y: float) -> int:
    if y <= 2:
        return 0
    if y <= 4:
        return 1
    if y <= 7:
        return 2
    if y <= 10:
        return 3
    if y <= 15:
        return 4
    return 5


def dist_bucket_coarse(y: float) -> int:
    if y <= 3:
        return 0
    if y <= 7:
        return 1
    return 2


def fp_bucket_fine(y: float) -> int:
    idx = int(min(max(y, 0.0), 99.999) // 10.0)
    return min(max(idx, 0), 9)


def fp_bucket_coarse(y: float) -> int:
    if y <= 20:
        return 0
    if y <= 50:
        return 1
    if y <= 80:
        return 2
    return 3


def score_bucket_fine(d: float) -> int:
    if d <= -17:
        return 0
    if d <= -9:
        return 1
    if d <= -4:
        return 2
    if d <= -1:
        return 3
    if d == 0:
        return 4
    if d <= 3:
        return 5
    if d <= 8:
        return 6
    if d <= 16:
        return 7
    return 8


def score_bucket_coarse(d: float) -> int:
    if d <= -9:
        return 0
    if d < 0:
        return 1
    if d == 0:
        return 2
    if d <= 8:
        return 3
    return 4


def time_bucket_fine(qtr: int, gsr: float) -> int:
    if qtr >= 5:
        return 7
    left = gsr - (4 - qtr) * 900.0
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


TIME_COARSE_MAP = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 2, 6: 3, 7: 4}


def time_bucket_coarse(tb_fine: int) -> int:
    return TIME_COARSE_MAP[tb_fine]


def vectorized_time_bucket_fine(qtr: np.ndarray, gsr: np.ndarray) -> np.ndarray:
    left = gsr - (4 - qtr) * 900.0
    tb = np.zeros(len(qtr), dtype=np.int8)
    tb[qtr >= 5] = 7
    m1 = qtr == 1
    tb[m1] = 0
    m2 = qtr == 2
    tb[m2 & (left > 120)] = 1
    tb[m2 & (left <= 120)] = 2
    m3 = qtr == 3
    tb[m3] = 3
    m4 = qtr == 4
    tb[m4 & (left > 300)] = 4
    tb[m4 & (left <= 300) & (left > 120)] = 5
    tb[m4 & (left <= 120)] = 6
    return tb


def compute_phase(qtr: int, gsr: float) -> int:
    if qtr >= 5:
        return 4
    if qtr == 4:
        return 3 if gsr <= 300.0 else 2
    if qtr == 2 and (gsr - 1800.0) <= 120.0:
        return 1
    return 0


def vectorized_phase(qtr: np.ndarray, gsr: np.ndarray) -> np.ndarray:
    phase = np.zeros(len(qtr), dtype=np.int8)
    is_q4 = qtr == 4
    phase[is_q4 & (gsr <= 300.0)] = 3
    phase[is_q4 & (gsr > 300.0)] = 2
    is_q2 = qtr == 2
    phase[is_q2 & ((gsr - 1800.0) <= 120.0)] = 1
    phase[qtr >= 5] = 4
    return phase


def continuous_time_feature(qtr: int, gsr: float) -> float:
    if qtr >= 5:
        return gsr
    return gsr - 1800.0 if gsr > 1800.0 else gsr


def vectorized_time_raw(qtr: np.ndarray, gsr: np.ndarray) -> np.ndarray:
    return np.where(qtr >= 5, gsr, np.where(gsr > 1800.0, gsr - 1800.0, gsr))


def scaled_score_diff(d):
    d = np.clip(d, -SCORE_CLIP, SCORE_CLIP)
    abs_d = np.abs(d)
    inner = np.minimum(abs_d, SCORE_INNER)
    outer = np.maximum(abs_d - SCORE_INNER, 0.0)
    mag = inner / SCORE_INNER_SCALE + outer / SCORE_OUTER_SCALE
    return np.sign(d) * mag


def feature_matrix(dist, fp, score, time_raw, off_to, def_to, phase_arr):
    to_weight = np.isin(phase_arr, LATE_PHASES).astype(float)
    return np.column_stack(
        [
            np.asarray(dist) / SCALE_YDSTOGO,
            np.asarray(fp) / SCALE_FP,
            scaled_score_diff(np.asarray(score)),
            np.asarray(time_raw) / SCALE_TIME,
            (np.asarray(off_to) / SCALE_TIMEOUTS) * to_weight,
            (np.asarray(def_to) / SCALE_TIMEOUTS) * to_weight,
        ]
    )


def round_state_key(down: int, phase: int, dist: float, fp: float, score: float, time_raw: float, off_to, def_to):
    r_dist = int(round(min(max(dist, 0.0), 30.0) / ROUND_DIST))
    r_fp = int(round(fp / ROUND_FP))
    r_score = int(round(min(max(score, -SCORE_CLIP), SCORE_CLIP) / ROUND_SCORE))
    r_time = int(round(time_raw / ROUND_TIME))
    if phase in LATE_PHASES:
        r_off = int(off_to)
        r_def = int(def_to)
    else:
        r_off = 0
        r_def = 0
    return (down, phase, r_dist, r_fp, r_score, r_time, r_off, r_def)


def phase_pool_mask(phase_arr: np.ndarray, phase: int) -> np.ndarray:
    if phase == 4:
        return np.isin(phase_arr, (3, 4))
    return phase_arr == phase


def build_neighbor_index(trans: pd.DataFrame) -> dict:
    down_arr = trans["down_i"].to_numpy()
    phase_arr = trans["phase"].to_numpy()
    feats = feature_matrix(
        trans["dist_raw"].to_numpy(),
        trans["fp_raw"].to_numpy(),
        trans["sc_raw"].to_numpy(),
        trans["time_raw"].to_numpy(),
        trans["off_to_raw"].to_numpy(),
        trans["def_to_raw"].to_numpy(),
        phase_arr,
    )
    trees = {}
    for down in (1, 2, 3, 4):
        for phase in range(5):
            mask = (down_arr == down) & phase_pool_mask(phase_arr, phase)
            sub_idx = np.flatnonzero(mask)
            if len(sub_idx) == 0:
                continue
            trees[(down, phase)] = (KDTree(feats[sub_idx]), sub_idx)
    return trees


def pick_index_nn(
    rng: np.random.Generator,
    tables: dict,
    down: int,
    phase: int,
    dist: float,
    fp: float,
    score: float,
    time_raw: float,
    off_to,
    def_to,
    k: int,
) -> int:
    down_key = down if down in (1, 2, 3, 4) else 4
    entry = tables["nn_trees"].get((down_key, phase))
    if entry is None:
        entry = tables["nn_trees"][(down_key, 0)]
    tree, sub_idx = entry
    cache = tables["nn_cache"]
    key = round_state_key(down_key, phase, dist, fp, score, time_raw, off_to, def_to)
    neighbors = cache.get(key)
    if neighbors is None:
        feat = feature_matrix(
            np.array([dist]),
            np.array([fp]),
            np.array([score]),
            np.array([time_raw]),
            np.array([off_to]),
            np.array([def_to]),
            np.array([phase]),
        )
        k_eff = min(k, len(sub_idx))
        _, ind = tree.query(feat, k=k_eff)
        neighbors = sub_idx[ind[0]]
        cache[key] = neighbors
    return int(neighbors[rng.integers(len(neighbors))])


FOURTH_DOWN_TYPE_CODES = (0, 1, 2, 3)


def fourth_down_label(play_type_code: np.ndarray) -> np.ndarray:
    return np.select([play_type_code == 3, play_type_code == 2], [1, 2], default=0)


def fit_fourth_down_policy(trans: pd.DataFrame) -> HistGradientBoostingClassifier:
    down_arr = trans["down_i"].to_numpy()
    ptc = trans["play_type_code"].to_numpy()
    mask = (down_arr == 4) & np.isin(ptc, FOURTH_DOWN_TYPE_CODES)
    sub = trans.loc[mask]
    label = fourth_down_label(sub["play_type_code"].to_numpy())
    kick_dist = sub["fp_raw"].to_numpy() + 17.0
    features = np.column_stack(
        [sub["sc_raw"].to_numpy(), sub["time_raw"].to_numpy(), sub["dist_raw"].to_numpy(), kick_dist]
    )
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=150, random_state=RNG_SEED)
    clf.fit(features, label)
    return clf


def fourth_down_clf_features(score: float, time_raw: float, dist: float, fp: float) -> np.ndarray:
    return np.array([[score, time_raw, dist, fp + 17.0]], dtype=X_DTYPE)


def make_fourth_down_bitsets(clf: HistGradientBoostingClassifier):
    return clf._bin_mapper.make_known_categories_bitsets()


def fast_gbm_predict_label(clf: HistGradientBoostingClassifier, bitsets, feat: np.ndarray) -> int:
    known_cat_bitsets, f_idx_map = bitsets
    raw = np.zeros((1, clf.n_trees_per_iteration_), dtype=clf._baseline_prediction.dtype, order="F")
    raw += clf._baseline_prediction
    for predictors_of_ith_iteration in clf._predictors:
        for k, predictor in enumerate(predictors_of_ith_iteration):
            raw[:, k] += predictor.predict(
                feat, known_cat_bitsets=known_cat_bitsets, f_idx_map=f_idx_map, n_threads=1
            )
    if raw.shape[1] == 1:
        encoded = int(raw.ravel()[0] > 0)
    else:
        encoded = int(np.argmax(raw[0]))
    return int(clf.classes_[encoded])


def build_fourth_down_group_index(trans: pd.DataFrame) -> dict:
    down_arr = trans["down_i"].to_numpy()
    phase_arr = trans["phase"].to_numpy()
    ptc = trans["play_type_code"].to_numpy()
    valid = np.isin(ptc, FOURTH_DOWN_TYPE_CODES)
    label_all = fourth_down_label(ptc)
    feats = feature_matrix(
        trans["dist_raw"].to_numpy(),
        trans["fp_raw"].to_numpy(),
        trans["sc_raw"].to_numpy(),
        trans["time_raw"].to_numpy(),
        trans["off_to_raw"].to_numpy(),
        trans["def_to_raw"].to_numpy(),
        phase_arr,
    )
    trees = {}
    for phase in LATE_PHASES:
        for label in (0, 1, 2):
            mask = valid & (down_arr == 4) & phase_pool_mask(phase_arr, phase) & (label_all == label)
            sub_idx = np.flatnonzero(mask)
            if len(sub_idx) == 0:
                continue
            trees[(phase, label)] = (KDTree(feats[sub_idx]), sub_idx)
    return trees


def pick_index_nn_fourth(
    rng: np.random.Generator,
    tables: dict,
    phase: int,
    label: int,
    dist: float,
    fp: float,
    score: float,
    time_raw: float,
    off_to,
    def_to,
    k: int,
) -> int | None:
    entry = tables["nn_trees_4th"].get((phase, label))
    if entry is None:
        return None
    tree, sub_idx = entry
    cache = tables["nn_cache_4th"]
    key = round_state_key(4, phase, dist, fp, score, time_raw, off_to, def_to) + (label,)
    neighbors = cache.get(key)
    if neighbors is None:
        feat = feature_matrix(
            np.array([dist]),
            np.array([fp]),
            np.array([score]),
            np.array([time_raw]),
            np.array([off_to]),
            np.array([def_to]),
            np.array([phase]),
        )
        k_eff = min(k, len(sub_idx))
        _, ind = tree.query(feat, k=k_eff)
        neighbors = sub_idx[ind[0]]
        cache[key] = neighbors
    return int(neighbors[rng.integers(len(neighbors))])


@lru_cache(maxsize=1)
def load_team_ratings() -> pd.DataFrame:
    df = pd.read_parquet(GAME_FEATURES_PATH)
    df = df[df["game_type"] == "REG"]
    cols = [
        "game_id",
        "home_off_epa_per_play",
        "away_off_epa_per_play",
        "home_def_epa_per_play",
        "away_def_epa_per_play",
    ]
    return df[cols].copy()


def build_neighbor_index_scipy(trans: pd.DataFrame) -> dict:
    down_arr = trans["down_i"].to_numpy()
    phase_arr = trans["phase"].to_numpy()
    feats = feature_matrix(
        trans["dist_raw"].to_numpy(),
        trans["fp_raw"].to_numpy(),
        trans["sc_raw"].to_numpy(),
        trans["time_raw"].to_numpy(),
        trans["off_to_raw"].to_numpy(),
        trans["def_to_raw"].to_numpy(),
        phase_arr,
    )
    trees = {}
    for down in (1, 2, 3, 4):
        for phase in range(5):
            mask = (down_arr == down) & phase_pool_mask(phase_arr, phase)
            sub_idx = np.flatnonzero(mask)
            if len(sub_idx) == 0:
                continue
            trees[(down, phase)] = (cKDTree(feats[sub_idx]), sub_idx)
    return trees


def pick_index_nn_conditioned(
    rng: np.random.Generator,
    tables: dict,
    down: int,
    phase: int,
    dist: float,
    fp: float,
    score: float,
    time_raw: float,
    off_to,
    def_to,
    k_state: int,
    off_sim: float,
    def_sim: float,
    is_home_sim: int,
) -> int:
    down_key = down if down in (1, 2, 3, 4) else 4
    entry = tables["nn_trees_cond"].get((down_key, phase))
    if entry is None:
        entry = tables["nn_trees_cond"][(down_key, 0)]
    tree, sub_idx = entry
    cache = tables["nn_cache_cond"]
    key = round_state_key(down_key, phase, dist, fp, score, time_raw, off_to, def_to)
    neighbors = cache.get(key)
    if neighbors is None:
        feat = feature_matrix(
            np.array([dist]),
            np.array([fp]),
            np.array([score]),
            np.array([time_raw]),
            np.array([off_to]),
            np.array([def_to]),
            np.array([phase]),
        )[0]
        k_eff = min(k_state, len(sub_idx))
        _, ind = tree.query(feat, k=k_eff)
        ind = np.atleast_1d(ind)
        neighbors = sub_idx[ind]
        cache[key] = neighbors
    wcache = tables["nn_weight_cache_cond"]
    wkey = (key, off_sim, def_sim, is_home_sim)
    cached_w = wcache.get(wkey)
    if cached_w is None:
        off_row = tables["arrays"]["off_row"][neighbors]
        def_row = tables["arrays"]["def_row"][neighbors]
        is_home_row = tables["arrays"]["is_home_off"][neighbors]
        h = tables["team_kernel_h"]
        dist_sq = (off_row - off_sim) ** 2 + (def_row - def_sim) ** 2
        weights = np.exp(-dist_sq / (2.0 * h * h))
        weights = weights * np.where(is_home_row == is_home_sim, TEAM_KERNEL_LAMBDA, 1.0)
        weight_sum = weights.sum()
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            weights = np.ones_like(weights)
            weight_sum = weights.sum()
        cdf = np.cumsum(weights)
        cached_w = (cdf, weight_sum)
        wcache[wkey] = cached_w
    cdf, weight_sum = cached_w
    pick = int(np.searchsorted(cdf, rng.random() * weight_sum, side="right"))
    pick = min(pick, len(neighbors) - 1)
    return int(neighbors[pick])


def build_pat_bonus(pbp: pd.DataFrame) -> pd.DataFrame:
    full = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    grp = full.groupby("game_id", sort=False)
    next_play_type = grp["play_type"].shift(-1)
    next_play_type_nfl = grp["play_type_nfl"].shift(-1)
    next_posteam_score = grp["posteam_score"].shift(-1)
    next_posteam_score_post = grp["posteam_score_post"].shift(-1)
    is_pat_next = (next_play_type == "extra_point") | (next_play_type_nfl == "PAT2")
    bonus = np.where(is_pat_next, next_posteam_score_post - next_posteam_score, 0.0)
    full["pat_bonus"] = np.nan_to_num(bonus, nan=0.0)
    return full[["game_id", "play_id", "pat_bonus"]]


def build_transition_frame(pbp: pd.DataFrame, team_ratings: pd.DataFrame | None = None) -> pd.DataFrame:
    pat_bonus = build_pat_bonus(pbp)
    df = pbp[pbp["play_type"].isin(LIVE_TYPES)].copy()
    required = ["down", "ydstogo", "yardline_100", "score_differential", "qtr", "game_seconds_remaining"]
    df = df.dropna(subset=required)
    df = df.merge(pat_bonus, on=["game_id", "play_id"], how="left")
    df["pat_bonus"] = df["pat_bonus"].fillna(0.0)
    if team_ratings is not None:
        rating_cols = [
            "game_id",
            "home_off_epa_per_play",
            "away_off_epa_per_play",
            "home_def_epa_per_play",
            "away_def_epa_per_play",
        ]
        df = df.merge(team_ratings[rating_cols], on="game_id", how="left")
    df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    grp = df.groupby("game_id", sort=False)
    df["next_down"] = grp["down"].shift(-1)
    df["next_distance"] = grp["ydstogo"].shift(-1)
    df["next_yardline"] = grp["yardline_100"].shift(-1)
    df["next_posteam"] = grp["posteam"].shift(-1)
    df["next_gsr"] = grp["game_seconds_remaining"].shift(-1)
    df["next_off_to_same"] = grp["posteam_timeouts_remaining"].shift(-1)
    df["next_def_to_same"] = grp["defteam_timeouts_remaining"].shift(-1)
    game_last = df["next_down"].isna() & (grp.cumcount(ascending=False) == 0)
    df.loc[game_last, "next_down"] = 1.0
    df.loc[game_last, "next_distance"] = 10.0
    df.loc[game_last, "next_yardline"] = 75.0
    df.loc[game_last, "next_posteam"] = df.loc[game_last, "defteam"]
    df.loc[game_last, "next_gsr"] = np.where(
        df.loc[game_last, "qtr"] >= 5, df.loc[game_last, "game_seconds_remaining"] - 6.0, 0.0
    )
    df = df[
        df["next_down"].notna() & df["next_distance"].notna() & df["next_yardline"].notna()
    ].reset_index(drop=True)

    flipped = (df["posteam"] != df["next_posteam"]).to_numpy()
    df["possession_flip"] = flipped

    next_off_to = np.where(flipped, df["next_def_to_same"], df["next_off_to_same"])
    next_def_to = np.where(flipped, df["next_off_to_same"], df["next_def_to_same"])
    off_to_used = np.clip(df["posteam_timeouts_remaining"].to_numpy() - next_off_to, 0, None)
    def_to_used = np.clip(df["defteam_timeouts_remaining"].to_numpy() - next_def_to, 0, None)
    df["off_to_used"] = np.nan_to_num(off_to_used, nan=0.0)
    df["def_to_used"] = np.nan_to_num(def_to_used, nan=0.0)

    turnover_score = (df["touchdown"] == 1) & ((df["interception"] == 1) | (df["fumble_lost"] == 1))
    points_off_raw_pre = (df["posteam_score_post"] - df["posteam_score"]).fillna(0.0).to_numpy()
    return_score = (df["touchdown"].to_numpy() == 1) & ~turnover_score.to_numpy() & (points_off_raw_pre == 0.0)
    def_score = turnover_score.to_numpy() | return_score

    drive_last = df.sort_values(["game_id", "fixed_drive", "play_id"]).groupby(
        ["game_id", "fixed_drive"], sort=False
    ).tail(1)
    safety_pairs = set(
        zip(
            drive_last.loc[drive_last["fixed_drive_result"] == "Safety", "game_id"],
            drive_last.loc[drive_last["fixed_drive_result"] == "Safety", "fixed_drive"],
        )
    )
    is_last_of_drive = df.index.isin(drive_last.index)
    row_keys = list(zip(df["game_id"], df["fixed_drive"]))
    is_safety_row = is_last_of_drive & pd.Series(row_keys, index=df.index).isin(safety_pairs).to_numpy()

    pat_bonus_arr = df["pat_bonus"].to_numpy()
    points_def = np.where(def_score, 6.0 + pat_bonus_arr, np.where(is_safety_row, 2.0, 0.0))
    points_off_raw = np.where(
        (df["touchdown"].to_numpy() == 1) & ~def_score,
        points_off_raw_pre + pat_bonus_arr,
        points_off_raw_pre,
    )
    points_off = np.where(points_def > 0, 0.0, points_off_raw)

    gsr_now = df["game_seconds_remaining"].to_numpy()
    raw_elapsed = gsr_now - df["next_gsr"].to_numpy()
    clock_elapsed = np.where(raw_elapsed < 0, gsr_now, np.clip(raw_elapsed, 1.0, None))

    yards_gained = df["yardline_100"].to_numpy() - df["next_yardline"].to_numpy()
    dist_gained = df["ydstogo"].to_numpy() - df["next_distance"].to_numpy()
    next_down_arr = df["next_down"].to_numpy()
    row_down = df["down"].to_numpy()
    same_side = ~flipped
    short = yards_gained < df["ydstogo"].to_numpy()
    spot_distance = df["ydstogo"].to_numpy() - yards_gained
    repeat_down = same_side & short & (next_down_arr == row_down) & (
        np.abs(df["next_distance"].to_numpy() - spot_distance) <= 0.5
    )
    auto_first = same_side & short & (next_down_arr == 1) & ~repeat_down

    down_i = df["down"].to_numpy().astype(int)
    dist = df["ydstogo"].to_numpy()
    fp = df["yardline_100"].to_numpy()
    sc = df["score_differential"].to_numpy()
    qtr = df["qtr"].to_numpy()
    gsr = df["game_seconds_remaining"].to_numpy()

    dist_f = np.array([dist_bucket_fine(v) for v in dist], dtype=np.int8)
    dist_c = np.array([dist_bucket_coarse(v) for v in dist], dtype=np.int8)
    fp_f = np.array([fp_bucket_fine(v) for v in fp], dtype=np.int8)
    fp_c = np.array([fp_bucket_coarse(v) for v in fp], dtype=np.int8)
    sc_f = np.array([score_bucket_fine(v) for v in sc], dtype=np.int8)
    sc_c = np.array([score_bucket_coarse(v) for v in sc], dtype=np.int8)
    tb_f = vectorized_time_bucket_fine(qtr, gsr)
    tb_c = np.array([time_bucket_coarse(int(v)) for v in tb_f], dtype=np.int8)
    off_to01 = (df["posteam_timeouts_remaining"].fillna(0).to_numpy() > 0).astype(np.int8)
    def_to01 = (df["defteam_timeouts_remaining"].fillna(0).to_numpy() > 0).astype(np.int8)
    phase = vectorized_phase(qtr, gsr)
    time_raw = vectorized_time_raw(qtr, gsr)
    off_to_raw = df["posteam_timeouts_remaining"].fillna(3.0).to_numpy()
    def_to_raw = df["defteam_timeouts_remaining"].fillna(3.0).to_numpy()

    if team_ratings is not None:
        is_home_off = (df["posteam"] == df["home_team"]).to_numpy()
        off_row = np.where(
            is_home_off, df["home_off_epa_per_play"].to_numpy(), df["away_off_epa_per_play"].to_numpy()
        )
        def_row = np.where(
            is_home_off, df["away_def_epa_per_play"].to_numpy(), df["home_def_epa_per_play"].to_numpy()
        )
        off_row = np.where(np.isnan(off_row), np.nanmean(off_row), off_row)
        def_row = np.where(np.isnan(def_row), np.nanmean(def_row), def_row)
        is_home_off_i8 = is_home_off.astype(np.int8)
    else:
        off_row = np.full(len(df), np.nan)
        def_row = np.full(len(df), np.nan)
        is_home_off_i8 = np.full(len(df), -1, dtype=np.int8)

    play_type_code = df["play_type"].map(PLAY_TYPE_CODES).fillna(-1).to_numpy().astype(np.int8)

    out = pd.DataFrame(
        {
            "down_i": down_i,
            "play_type_code": play_type_code,
            "dist_f": dist_f,
            "dist_c": dist_c,
            "fp_f": fp_f,
            "fp_c": fp_c,
            "sc_f": sc_f,
            "sc_c": sc_c,
            "tb_f": tb_f,
            "tb_c": tb_c,
            "off_to01": off_to01,
            "def_to01": def_to01,
            "phase": phase,
            "auto_first": auto_first,
            "repeat_down": repeat_down,
            "dist_raw": dist,
            "fp_raw": fp,
            "sc_raw": sc,
            "time_raw": time_raw,
            "off_to_raw": off_to_raw,
            "def_to_raw": def_to_raw,
            "next_down": df["next_down"].to_numpy(),
            "next_distance": df["next_distance"].to_numpy(),
            "next_yardline": df["next_yardline"].to_numpy(),
            "yards_gained": yards_gained,
            "dist_gained": dist_gained,
            "possession_flip": flipped,
            "points_off": points_off,
            "points_def": points_def,
            "clock_elapsed": clock_elapsed,
            "off_to_used": df["off_to_used"].to_numpy(),
            "def_to_used": df["def_to_used"].to_numpy(),
            "off_row": off_row,
            "def_row": def_row,
            "is_home_off": is_home_off_i8,
        }
    )
    return out


def build_opening_pool(pbp: pd.DataFrame) -> np.ndarray:
    df = pbp[pbp["play_type"].isin(LIVE_TYPES) & pbp["yardline_100"].notna()].sort_values(
        ["game_id", "play_id"]
    )
    first = df.groupby("game_id", sort=False).head(1)
    return first["yardline_100"].to_numpy()


def build_tables(seasons: tuple[int, ...], condition_on_team: bool = False) -> dict:
    pbp = load_reg_seasons(seasons)
    team_ratings = load_team_ratings() if condition_on_team else None
    trans = build_transition_frame(pbp, team_ratings=team_ratings)
    trans = trans.reset_index(drop=True)
    nn_trees_cond = build_neighbor_index_scipy(trans) if condition_on_team else None
    if condition_on_team:
        window_ids = set(pbp["game_id"].unique())
        window_ratings = team_ratings[team_ratings["game_id"].isin(window_ids)]
        off_spread = pd.concat(
            [window_ratings["home_off_epa_per_play"], window_ratings["away_off_epa_per_play"]]
        )
        def_spread = pd.concat(
            [window_ratings["home_def_epa_per_play"], window_ratings["away_def_epa_per_play"]]
        )
        team_kernel_h = TEAM_KERNEL_H_SCALE * float(np.nanstd(off_spread.to_numpy()))
        league_off_mean = float(np.nanmean(off_spread.to_numpy()))
        league_def_mean = float(np.nanmean(def_spread.to_numpy()))
    else:
        team_kernel_h = None
        league_off_mean = None
        league_def_mean = None
    nn_trees = build_neighbor_index(trans)
    fourth_down_clf = fit_fourth_down_policy(trans)
    fourth_down_clf_bitsets = make_fourth_down_bitsets(fourth_down_clf)
    nn_trees_4th = build_fourth_down_group_index(trans)
    opening_pool = build_opening_pool(pbp)
    off_td_mask = trans["points_off"].to_numpy() >= 6.0
    def_td_mask = trans["points_def"].to_numpy() >= 6.0
    pat_bonus_pool = np.concatenate(
        [
            trans.loc[off_td_mask, "points_off"].to_numpy() - 6.0,
            trans.loc[def_td_mask, "points_def"].to_numpy() - 6.0,
        ]
    )
    if len(pat_bonus_pool) == 0:
        pat_bonus_pool = np.array([0.0])
    scored_mask = (
        (trans["points_off"].to_numpy() > 0.0) | (trans["points_def"].to_numpy() > 0.0)
    ) & trans["possession_flip"].to_numpy()
    post_score_pool = trans.loc[scored_mask, "next_yardline"].to_numpy()
    if len(post_score_pool) == 0:
        post_score_pool = opening_pool
    arrays = {
        "down_i": trans["down_i"].to_numpy(),
        "dist_raw": trans["dist_raw"].to_numpy(),
        "fp_raw": trans["fp_raw"].to_numpy(),
        "play_type_code": trans["play_type_code"].to_numpy(),
        "points_off": trans["points_off"].to_numpy(),
        "points_def": trans["points_def"].to_numpy(),
        "clock_elapsed": trans["clock_elapsed"].to_numpy(),
        "possession_flip": trans["possession_flip"].to_numpy(),
        "next_down": trans["next_down"].to_numpy(),
        "next_distance": trans["next_distance"].to_numpy(),
        "next_yardline": trans["next_yardline"].to_numpy(),
        "yards_gained": trans["yards_gained"].to_numpy(),
        "dist_gained": trans["dist_gained"].to_numpy(),
        "auto_first": trans["auto_first"].to_numpy(),
        "repeat_down": trans["repeat_down"].to_numpy(),
        "off_to_used": trans["off_to_used"].to_numpy(),
        "def_to_used": trans["def_to_used"].to_numpy(),
        "off_row": trans["off_row"].to_numpy(),
        "def_row": trans["def_row"].to_numpy(),
        "is_home_off": trans["is_home_off"].to_numpy(),
    }
    return {
        "arrays": arrays,
        "nn_trees": nn_trees,
        "nn_trees_cond": nn_trees_cond,
        "nn_cache": {},
        "nn_cache_cond": {},
        "nn_weight_cache_cond": {},
        "fourth_down_clf": fourth_down_clf,
        "fourth_down_clf_bitsets": fourth_down_clf_bitsets,
        "nn_trees_4th": nn_trees_4th,
        "nn_cache_4th": {},
        "opening_pool": opening_pool,
        "pat_bonus_pool": pat_bonus_pool,
        "post_score_pool": post_score_pool,
        "n_rows": len(trans),
        "seasons": seasons,
        "team_kernel_h": team_kernel_h,
        "league_off_mean": league_off_mean,
        "league_def_mean": league_def_mean,
    }


def default_policy(down, distance, yardline, score_diff, qtr, clock_left, drawn):
    return drawn


def initial_kickoff_state(rng: np.random.Generator, opening_pool: np.ndarray) -> dict:
    offense = "home" if rng.random() < 0.5 else "away"
    return {
        "home_score": 0.0,
        "away_score": 0.0,
        "home_to": 3,
        "away_to": 3,
        "offense": offense,
        "second_half_receiver": "away" if offense == "home" else "home",
        "down": 1,
        "distance": 10.0,
        "yardline": float(rng.choice(opening_pool)),
        "gsr": 3600.0,
        "in_ot": False,
        "ot_clock": 0.0,
        "ot_possession_index": 0,
        "went_ot": False,
        "drive_start_qtr": 1,
        "drive_start_gsr": 3600.0,
        "drive_scored": False,
        "tied_at_5_diff": None,
    }


def run_one_game(
    state: dict,
    tables: dict,
    rng: np.random.Generator,
    ot_seconds: float,
    policy,
    min_cell_n: int,
    max_plays: int,
    home_ratings: dict | None = None,
    away_ratings: dict | None = None,
) -> tuple[dict, bool]:
    arrays = tables["arrays"]
    conditioned = home_ratings is not None and away_ratings is not None and tables["team_kernel_h"] is not None
    opening_pool = tables["opening_pool"]
    pat_bonus_pool = tables["pat_bonus_pool"]
    post_score_pool = tables["post_score_pool"]

    home_score = state["home_score"]
    away_score = state["away_score"]
    home_to = state["home_to"]
    away_to = state["away_to"]
    offense = state["offense"]
    second_half_receiver = state["second_half_receiver"]
    down = state["down"]
    distance = state["distance"]
    yardline = state["yardline"]
    gsr = state["gsr"]
    in_ot = state["in_ot"]
    ot_clock = state["ot_clock"]
    ot_possession_index = state["ot_possession_index"]
    went_ot = state["went_ot"]
    drive_start_qtr = state["drive_start_qtr"]
    drive_start_gsr = state["drive_start_gsr"]
    drive_scored = state["drive_scored"]
    tied_at_5_diff = state["tied_at_5_diff"]

    plays = 0
    possessions = 1
    late_q4_log: list = []
    settled = False
    final_margin = None
    cap_hit = False

    for _step in range(max_plays):
        plays += 1
        if in_ot:
            qtr = 5
        else:
            qtr = 4 - int(np.floor(max(gsr, 0.0) / 900.0))
            qtr = min(max(qtr, 1), 4)
        clock_val = ot_clock if in_ot else gsr

        off_to = home_to if offense == "home" else away_to
        def_to = away_to if offense == "home" else home_to
        off_score = home_score if offense == "home" else away_score
        def_score = away_score if offense == "home" else home_score
        score_diff = off_score - def_score

        phase = 4 if in_ot else compute_phase(qtr, gsr)
        time_feat = ot_clock if in_ot else continuous_time_feature(qtr, gsr)

        if conditioned:
            if offense == "home":
                off_sim = home_ratings["off"]
                def_sim = away_ratings["def"]
                is_home_sim = 1
            else:
                off_sim = away_ratings["off"]
                def_sim = home_ratings["def"]
                is_home_sim = 0
            idx = pick_index_nn_conditioned(
                rng,
                tables,
                down,
                phase,
                distance,
                yardline,
                score_diff,
                time_feat,
                off_to,
                def_to,
                K_STATE,
                off_sim,
                def_sim,
                is_home_sim,
            )
        else:
            idx = pick_index_nn(
                rng, tables, down, phase, distance, yardline, score_diff, time_feat, off_to, def_to, min_cell_n
            )

        down_key = down if down in (1, 2, 3, 4) else 4
        fourth_clf = tables.get("fourth_down_clf")
        if down_key == 4 and phase in LATE_PHASES and fourth_clf is not None:
            feat = fourth_down_clf_features(score_diff, time_feat, distance, yardline)
            label = fast_gbm_predict_label(fourth_clf, tables["fourth_down_clf_bitsets"], feat)
            alt_idx = pick_index_nn_fourth(
                rng, tables, phase, label, distance, yardline, score_diff, time_feat, off_to, def_to, min_cell_n
            )
            if alt_idx is not None:
                idx = alt_idx

        if conditioned:
            drawn_net = float(arrays["off_row"][idx] - arrays["def_row"][idx])
            yard_shift = TEAM_RATING_YARD_GAIN * ((off_sim - def_sim) - drawn_net)
        else:
            yard_shift = 0.0

        drawn = {
            "points_off": arrays["points_off"][idx],
            "points_def": arrays["points_def"][idx],
            "clock_elapsed": arrays["clock_elapsed"][idx],
            "flip": bool(arrays["possession_flip"][idx]),
            "next_down": arrays["next_down"][idx],
            "next_distance": arrays["next_distance"][idx],
            "next_yardline": arrays["next_yardline"][idx],
            "yards_gained": arrays["yards_gained"][idx] + yard_shift,
            "dist_gained": arrays["dist_gained"][idx],
            "auto_first": bool(arrays["auto_first"][idx]),
            "repeat_down": bool(arrays["repeat_down"][idx]),
            "off_to_used": arrays["off_to_used"][idx],
            "def_to_used": arrays["def_to_used"][idx],
            "play_type_code": arrays["play_type_code"][idx],
        }
        drawn = policy(down, distance, yardline, score_diff, qtr, clock_val, drawn)

        points_off = drawn["points_off"]
        points_def = drawn["points_def"]
        clock_elapsed = drawn["clock_elapsed"]
        flip = drawn["flip"]

        if offense == "home":
            home_score += points_off
            away_score += points_def
            home_to = max(0, home_to - drawn["off_to_used"])
            away_to = max(0, away_to - drawn["def_to_used"])
        else:
            away_score += points_off
            home_score += points_def
            away_to = max(0, away_to - drawn["off_to_used"])
            home_to = max(0, home_to - drawn["def_to_used"])

        if points_off > 0 or points_def > 0:
            drive_scored = True

        if in_ot:
            ot_clock = max(0.0, ot_clock - clock_elapsed)
            if (points_def > 0 or points_off >= 6) and home_score != away_score:
                settled = True
            elif points_off > 0 and ot_possession_index >= 1 and home_score != away_score:
                settled = True
            if settled:
                final_margin = home_score - away_score
                break
            if ot_clock <= 0.0:
                final_margin = home_score - away_score
                if drive_start_qtr == 4 and drive_start_gsr <= 300.0:
                    late_q4_log.append(drive_scored)
                break
        else:
            new_gsr = max(0.0, gsr - clock_elapsed)
            if qtr == 4 and gsr > 300.0 and new_gsr <= 300.0 and tied_at_5_diff is None:
                tied_at_5_diff = home_score - away_score
            crossed_half = gsr > 1800.0 and new_gsr <= 1800.0
            if crossed_half:
                home_to, away_to = 3, 3
                if drive_start_qtr == 4 and drive_start_gsr <= 300.0:
                    late_q4_log.append(drive_scored)
                gsr = 1800.0
                possessions += 1
                offense = second_half_receiver
                down, distance = 1, 10
                yardline = float(rng.choice(opening_pool))
                drive_start_qtr = 3
                drive_start_gsr = gsr
                drive_scored = False
                continue
            gsr = new_gsr
            if gsr <= 0.0:
                if drive_start_qtr == 4 and drive_start_gsr <= 300.0:
                    late_q4_log.append(drive_scored)
                if home_score != away_score:
                    final_margin = home_score - away_score
                    break
                went_ot = True
                in_ot = True
                ot_clock = ot_seconds
                ot_possession_index = 0
                offense = "home" if rng.random() < 0.5 else "away"
                down, distance = 1, 10
                yardline = float(rng.choice(opening_pool))
                home_to, away_to = 3, 3
                possessions += 1
                drive_start_qtr = 5
                drive_start_gsr = ot_clock
                drive_scored = False
                continue

        if flip:
            if drive_start_qtr == 4 and drive_start_gsr <= 300.0:
                late_q4_log.append(drive_scored)
            possessions += 1
            if in_ot:
                ot_possession_index += 1
            offense = "away" if offense == "home" else "home"
            down = int(drawn["next_down"])
            distance = float(drawn["next_distance"])
            yardline = float(drawn["next_yardline"])
            drive_start_qtr = qtr
            drive_start_gsr = clock_val if not in_ot else ot_clock
            drive_scored = False
        else:
            raw_next_yardline = yardline - drawn["yards_gained"]
            if raw_next_yardline <= 0.0:
                bonus = float(rng.choice(pat_bonus_pool))
                td_points = 6.0 + bonus
                if offense == "home":
                    home_score += td_points
                else:
                    away_score += td_points
                drive_scored = True
                if in_ot:
                    final_margin = home_score - away_score
                    break
                if drive_start_qtr == 4 and drive_start_gsr <= 300.0:
                    late_q4_log.append(drive_scored)
                possessions += 1
                offense = "away" if offense == "home" else "home"
                down, distance = 1, 10.0
                yardline = float(rng.choice(post_score_pool))
                drive_start_qtr = qtr
                drive_start_gsr = clock_val if not in_ot else ot_clock
                drive_scored = False
            else:
                new_yardline = min(max(raw_next_yardline, 1.0), 99.0)
                gained = yardline - new_yardline
                if drawn["auto_first"] or (not drawn["repeat_down"] and gained >= distance):
                    down = 1
                    distance = min(10.0, new_yardline)
                    yardline = new_yardline
                elif drawn["repeat_down"]:
                    distance = min(max(distance - gained, 1.0), new_yardline)
                    yardline = new_yardline
                elif down >= 4:
                    if drive_start_qtr == 4 and drive_start_gsr <= 300.0:
                        late_q4_log.append(drive_scored)
                    possessions += 1
                    if in_ot:
                        ot_possession_index += 1
                    offense = "away" if offense == "home" else "home"
                    down, distance = 1, 10.0
                    yardline = min(max(100.0 - new_yardline, 1.0), 99.0)
                    distance = min(10.0, yardline)
                    drive_start_qtr = qtr
                    drive_start_gsr = clock_val if not in_ot else ot_clock
                    drive_scored = False
                else:
                    down = down + 1
                    distance = min(max(distance - gained, 1.0), new_yardline)
                    yardline = new_yardline
    else:
        cap_hit = True
        final_margin = home_score - away_score

    record = {
        "margin": final_margin,
        "total": home_score + away_score,
        "went_ot": went_ot,
        "tied": final_margin == 0,
        "plays": plays,
        "possessions": possessions,
        "tied_at_5_diff": tied_at_5_diff,
        "late_q4_possessions_scored": late_q4_log,
    }
    return record, cap_hit


def simulate(
    n_games: int,
    rng: np.random.Generator,
    tables: dict,
    ot_seconds: float = 600.0,
    policy=None,
    min_cell_n: int = K_NEIGHBORS,
    max_plays: int = MAX_PLAYS_PER_GAME,
    home_ratings: dict | None = None,
    away_ratings: dict | None = None,
) -> pd.DataFrame:
    policy = policy or default_policy
    opening_pool = tables["opening_pool"]

    records = []
    max_play_cap_hits = 0

    for _ in range(n_games):
        state = initial_kickoff_state(rng, opening_pool)
        record, cap_hit = run_one_game(
            state, tables, rng, ot_seconds, policy, min_cell_n, max_plays, home_ratings, away_ratings
        )
        max_play_cap_hits += int(cap_hit)
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    frame.attrs["max_play_cap_hits"] = max_play_cap_hits
    return frame


def simulate_from_states(
    states: list[dict],
    tables: dict,
    rng: np.random.Generator,
    ot_seconds: float,
    reps: int,
    policy=None,
    min_cell_n: int = K_NEIGHBORS,
    max_plays: int = MAX_PLAYS_PER_GAME,
) -> pd.DataFrame:
    policy = policy or default_policy
    records = []
    max_play_cap_hits = 0

    for base_state in states:
        for _ in range(reps):
            state = dict(base_state)
            record, cap_hit = run_one_game(state, tables, rng, ot_seconds, policy, min_cell_n, max_plays)
            record["source_game_id"] = base_state.get("source_game_id")
            max_play_cap_hits += int(cap_hit)
            records.append(record)

    frame = pd.DataFrame.from_records(records)
    frame.attrs["max_play_cap_hits"] = max_play_cap_hits
    return frame


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
        result[k] = (float(np.mean(arr)), float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))
    return result


def discrete_log_loss(actual_margins: np.ndarray, reference_margins: np.ndarray, max_abs: int = 60) -> float:
    grid = np.arange(-max_abs, max_abs + 1)
    counts = np.array([(reference_margins == g).sum() for g in grid], dtype=float)
    counts += 1.0
    probs = counts / counts.sum()
    clipped_actual = np.clip(actual_margins, -max_abs, max_abs).round().astype(int)
    idx = clipped_actual + max_abs
    p = probs[idx]
    return float(-np.mean(np.log(p)))


def measure_actual_diagnostics(seasons: tuple[int, ...]) -> dict:
    pbp = load_reg_seasons(seasons)
    live = pbp[pbp["play_type"].isin(LIVE_TYPES)].copy()
    n_games = live["game_id"].nunique()
    plays_per_game = len(live) / n_games
    possessions_per_game = live.groupby("game_id")["fixed_drive"].nunique().mean()

    drive_first = live.sort_values(["game_id", "fixed_drive", "play_id"]).groupby(
        ["game_id", "fixed_drive"], sort=False
    ).first()
    drive_scored = live.groupby(["game_id", "fixed_drive"])["fixed_drive_result"].last().isin(
        ["Touchdown", "Field goal", "Safety", "Opp touchdown"]
    )
    drive_info = drive_first.join(drive_scored.rename("scored"))
    late_mask = (drive_info["qtr"] == 4) & (drive_info["game_seconds_remaining"] <= 300.0)
    late_q4_scoring_rate = float(drive_info.loc[late_mask, "scored"].mean())

    q4 = live[(live["qtr"] == 4) & (live["game_seconds_remaining"] >= 300.0)].sort_values(
        ["game_id", "game_seconds_remaining"], ascending=[True, True]
    )
    at_5 = q4.groupby("game_id").head(1)[["game_id", "score_differential", "posteam", "home_team"]].copy()
    at_5["home_diff"] = np.where(
        at_5["posteam"] == at_5["home_team"], at_5["score_differential"], -at_5["score_differential"]
    )
    tied_games = set(at_5.loc[at_5["home_diff"] == 0, "game_id"])
    ot_games = set(live.loc[live["qtr"] >= 5, "game_id"].unique())
    n_tied_at_5 = len(tied_games)
    n_tied_at_5_ot = len(tied_games & ot_games)
    tied_at_5_ot_rate = n_tied_at_5_ot / n_tied_at_5 if n_tied_at_5 else float("nan")

    n_ot_games = len(ot_games)
    ot_rate = n_ot_games / n_games

    return {
        "n_games": int(n_games),
        "plays_per_game": float(plays_per_game),
        "possessions_per_game": float(possessions_per_game),
        "late_q4_possession_scoring_rate": late_q4_scoring_rate,
        "n_tied_at_5min_q4_games": n_tied_at_5,
        "tied_at_5min_q4_ot_rate": tied_at_5_ot_rate,
        "ot_rate_all_games": ot_rate,
    }


def summarize_sim(frame: pd.DataFrame) -> dict:
    late_all = [s for row in frame["late_q4_possessions_scored"] for s in row]
    tied5 = frame["tied_at_5_diff"].dropna()
    tied5_games = frame.loc[frame["tied_at_5_diff"] == 0]
    return {
        "n_games": int(len(frame)),
        "plays_per_game": float(frame["plays"].mean()),
        "possessions_per_game": float(frame["possessions"].mean()),
        "late_q4_possession_scoring_rate": float(np.mean(late_all)) if late_all else float("nan"),
        "n_late_q4_possessions": len(late_all),
        "n_tied_at_5min_q4_games": int(len(tied5_games)),
        "tied_at_5min_q4_ot_rate": float(tied5_games["went_ot"].mean()) if len(tied5_games) else float("nan"),
        "ot_rate_all_games": float(frame["went_ot"].mean()),
        "tie_rate": float(frame["tied"].mean()),
        "max_play_cap_hits": int(frame.attrs.get("max_play_cap_hits", 0)),
    }


def run_validation(n_games_per_season: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    tables = build_tables(TRAIN_SEASONS)

    sim_frames = []
    for season in VALID_SEASONS:
        frame = simulate(
            n_games_per_season, rng, tables, ot_seconds=OT_SECONDS_BY_SEASON[season]
        )
        frame["season"] = season
        sim_frames.append(frame)
    sim_all = pd.concat(sim_frames, ignore_index=True)
    sim_all.attrs["max_play_cap_hits"] = sum(f.attrs.get("max_play_cap_hits", 0) for f in sim_frames)

    game_features = pd.read_parquet(GAME_FEATURES_PATH)
    game_features = game_features[
        game_features["season"].isin(TRAIN_SEASONS + VALID_SEASONS) & (game_features["game_type"] == "REG")
    ].copy()
    game_features["margin"] = game_features["home_score"] - game_features["away_score"]

    train_margins_actual = game_features.loc[game_features["season"].isin(TRAIN_SEASONS), "margin"].to_numpy()
    valid_margins_actual = game_features.loc[game_features["season"].isin(VALID_SEASONS), "margin"].to_numpy()
    actual_by_season = {
        int(season): group["margin"].to_numpy()
        for season, group in game_features[game_features["season"].isin(VALID_SEASONS)].groupby("season")
    }

    sim_margins = sim_all["margin"].to_numpy()

    sim_mass = key_number_mass(sim_margins, KEY_NUMBERS)
    boot_ci = bootstrap_season_ci(actual_by_season, KEY_NUMBERS, N_BOOT, rng)
    actual_mass = key_number_mass(valid_margins_actual, KEY_NUMBERS)
    hits = sum(1 for k in KEY_NUMBERS if boot_ci[k][1] <= sim_mass[k] <= boot_ci[k][2])

    sim_log_loss = discrete_log_loss(valid_margins_actual, sim_margins)
    naive_log_loss = discrete_log_loss(valid_margins_actual, train_margins_actual)
    log_loss_delta = sim_log_loss - naive_log_loss

    go_decision = bool(hits >= 4 and log_loss_delta <= 0.02)

    sim_diag = summarize_sim(sim_all)
    actual_diag = measure_actual_diagnostics(VALID_SEASONS)

    report = {
        "generated_at": out_dir.name,
        "engine": "scripts/sim04_engine.py",
        "train_seasons": list(TRAIN_SEASONS),
        "valid_seasons": list(VALID_SEASONS),
        "n_simulated_games_per_season": n_games_per_season,
        "n_train_transition_rows": tables["n_rows"],
        "min_cell_n": K_NEIGHBORS,
        "key_number_mass": {
            str(k): {
                "simulated": sim_mass[k],
                "actual_valid_pooled": actual_mass[k],
                "actual_valid_bootstrap_mean": boot_ci[k][0],
                "actual_valid_bootstrap_p05": boot_ci[k][1],
                "actual_valid_bootstrap_p95": boot_ci[k][2],
                "simulated_within_ci": bool(boot_ci[k][1] <= sim_mass[k] <= boot_ci[k][2]),
            }
            for k in KEY_NUMBERS
        },
        "key_number_hits_of_5": hits,
        "discrete_log_loss": {
            "simulator_vs_actual_valid": sim_log_loss,
            "naive_train_histogram_vs_actual_valid": naive_log_loss,
            "delta_sim_minus_naive": log_loss_delta,
        },
        "predeclared_criterion": {"min_key_number_hits": 4, "max_log_loss_delta": 0.02},
        "go_no_go": "GO" if go_decision else "NO_GO",
        "sim_margin_mean": float(np.mean(sim_margins)),
        "sim_margin_std": float(np.std(sim_margins)),
        "actual_valid_margin_mean": float(np.mean(valid_margins_actual)),
        "actual_valid_margin_std": float(np.std(valid_margins_actual)),
        "margin_sd_ratio_sim_over_actual": float(np.std(sim_margins) / np.std(valid_margins_actual)),
        "sim_points_per_game_mean": float(sim_all["total"].mean()),
        "simulated_diagnostics": sim_diag,
        "actual_diagnostics": actual_diag,
    }
    return report, sim_all


def _mp_process_batch(args):
    batch, seasons, condition_on_team, n_reps, ot_seconds = args
    import threadpoolctl

    with threadpoolctl.threadpool_limits(limits=1):
        tables = build_tables(tuple(seasons), condition_on_team=condition_on_team)
    out = []
    for game_index, seed, home_ratings, away_ratings in batch:
        rng = np.random.default_rng(seed)
        sim = simulate(
            n_reps,
            rng,
            tables,
            ot_seconds=ot_seconds,
            home_ratings=home_ratings,
            away_ratings=away_ratings,
        )
        out.append((game_index, sim["margin"].to_numpy(dtype=float)))
    return out


def simulate_games_multiprocess(
    games: list[dict],
    seasons: tuple[int, ...],
    condition_on_team: bool,
    n_reps: int,
    base_seed: int,
    ot_seconds: float = 600.0,
    max_workers: int | None = None,
) -> list[np.ndarray]:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    if not games:
        return []
    ctx = mp.get_context("spawn")
    n_workers = max(1, min(max_workers or mp.cpu_count(), len(games)))
    tasks = [(i, base_seed + i, g.get("home_ratings"), g.get("away_ratings")) for i, g in enumerate(games)]
    batches = [tasks[i::n_workers] for i in range(n_workers)]
    batch_args = [(batch, seasons, condition_on_team, n_reps, ot_seconds) for batch in batches if batch]
    results: list[np.ndarray | None] = [None] * len(games)
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        for batch_result in executor.map(_mp_process_batch, batch_args):
            for game_index, margins in batch_result:
                results[game_index] = margins
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games-per-season", type=int, default=3334)
    args = parser.parse_args()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    report, sim_all = run_validation(args.n_games_per_season, out_dir)

    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    np.save(out_dir / "sim_margins.npy", sim_all["margin"].to_numpy())

    print(json.dumps(report, indent=2))
    print(f"Artifact directory: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
