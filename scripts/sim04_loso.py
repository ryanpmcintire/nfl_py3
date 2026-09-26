from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim04_engine import build_tables, simulate

OPENER_PATH = REPO / "artifacts" / "opener_evaluation" / "20260925T161328Z" / "per_game.parquet"
GAME_FEATURES_PATH = REPO / "data" / "processed" / "game_features_pbp.parquet"
ARTIFACT_ROOT = REPO / "artifacts" / "sim04_loso"

GRADED_SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
FIRST_TRAIN_SEASON = 2009
ATOM_TOLERANCE = 1e-9
KEY_NUMBERS = (3, 7, 10, 14, 17)
MARGIN_CLIP = 70
RNG_SEED = 20260925
ENGINE_N_GAMES = 20000
N_BOOT = 2000
PC_ALPHAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
EPS = 1e-9
TEAM_COND_N_REPS_REQUESTED = 1000
TEAM_COND_N_REPS = 180
MEASURED_GAMES_PER_SECOND_2020 = 30.5
MEASURED_GAMES_PER_SECOND_2025 = 27.3
RATING_COLS = [
    "game_id",
    "home_off_epa_per_play",
    "away_off_epa_per_play",
    "home_def_epa_per_play",
    "away_def_epa_per_play",
]


def load_games() -> pd.DataFrame:
    opener = pd.read_parquet(OPENER_PATH)
    features = pd.read_parquet(GAME_FEATURES_PATH)
    meta = features.loc[
        features["game_type"] == "REG", ["game_id", "home_team", "away_team", "gameday"]
    ]
    frame = opener.merge(meta, on="game_id", how="left")
    missing = frame["home_team"].isna().sum()
    if missing:
        raise ValueError(f"{missing} games did not join to game_features_pbp")
    frame = frame.loc[frame["season"].isin(GRADED_SEASONS)].copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    frame["line"] = frame["tue_open_home_spread"].astype(float)
    frame["result"] = frame["result"].astype(float)
    frame["predicted_margin_at_open"] = (
        frame["tue_open_home_spread"].astype(float) + frame["residual_at_open_served"].astype(float)
    )
    diff = frame["result"] - frame["line"]
    frame["outcome"] = np.where(
        diff > ATOM_TOLERANCE, "cover", np.where(diff < -ATOM_TOLERANCE, "loss", "push")
    )
    return frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def baseline_three_way(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cover = frame["home_cover_probability_excluding_push_at_open"].to_numpy(dtype=float)
    push = frame["push_probability_at_open"].to_numpy(dtype=float)
    loss = frame["home_loss_probability_at_open"].to_numpy(dtype=float)
    total = cover + push + loss
    if np.max(np.abs(total - 1.0)) > 1e-6:
        raise ValueError("Stored baseline three-way probabilities do not sum to 1")
    return cover, push, loss


def three_way_score(
    p_cover: np.ndarray, p_push: np.ndarray, p_loss: np.ndarray, outcome: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p = np.select(
        [outcome == "cover", outcome == "push", outcome == "loss"], [p_cover, p_push, p_loss]
    )
    p_clipped = np.clip(p, EPS, 1.0 - EPS)
    log_loss = -np.log(p_clipped)
    y_cover = (outcome == "cover").astype(float)
    y_push = (outcome == "push").astype(float)
    y_loss = (outcome == "loss").astype(float)
    brier = (p_cover - y_cover) ** 2 + (p_push - y_push) ** 2 + (p_loss - y_loss) ** 2
    return log_loss, brier


def recenter_hist(shape_dev: np.ndarray, center: float) -> dict[int, float]:
    shifted = np.clip(np.rint(shape_dev + center), -MARGIN_CLIP, MARGIN_CLIP).astype(int)
    values, counts = np.unique(shifted, return_counts=True)
    probs = counts / counts.sum()
    return dict(zip(values.tolist(), probs.tolist(), strict=True))


def tilt_to_mean(values: np.ndarray, counts: np.ndarray, target: float) -> np.ndarray:
    lo, hi = -2.0, 2.0
    for _ in range(80):
        theta = (lo + hi) / 2.0
        w = counts * np.exp(theta * (values - values.mean()))
        if (w * values).sum() / w.sum() < target:
            lo = theta
        else:
            hi = theta
    return counts * np.exp((lo + hi) / 2.0 * (values - values.mean()))


def tilt_hist(margins: np.ndarray, center: float) -> dict[int, float]:
    clipped = np.clip(margins, -MARGIN_CLIP, MARGIN_CLIP).astype(int)
    values, counts = np.unique(clipped, return_counts=True)
    weights = tilt_to_mean(values.astype(float), counts.astype(float), center)
    probs = weights / weights.sum()
    return dict(zip(values.tolist(), probs.tolist(), strict=True))


def safe_logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def hist_to_three_way(hist: dict[int, float], line: float) -> tuple[float, float, float]:
    cover = push = loss = 0.0
    for margin, prob in hist.items():
        if margin > line + ATOM_TOLERANCE:
            cover += prob
        elif margin < line - ATOM_TOLERANCE:
            loss += prob
        else:
            push += prob
    return cover, push, loss


def build_engine_shapes(game_features: pd.DataFrame) -> dict[int, np.ndarray]:
    shapes: dict[int, np.ndarray] = {}
    for season in GRADED_SEASONS:
        train_seasons = tuple(range(FIRST_TRAIN_SEASON, season))
        rng = np.random.default_rng(RNG_SEED + season)
        tables = build_tables(train_seasons)
        sim = simulate(ENGINE_N_GAMES, rng, tables)
        margins = np.rint(sim["margin"].to_numpy(dtype=float))
        shapes[season] = margins - margins.mean()
    return shapes


def build_historical_shapes(game_features: pd.DataFrame) -> dict[int, np.ndarray]:
    reg = game_features.loc[game_features["game_type"] == "REG"].copy()
    reg["margin"] = reg["home_score"].astype(float) - reg["away_score"].astype(float)
    shapes: dict[int, np.ndarray] = {}
    for season in GRADED_SEASONS:
        prior = reg.loc[reg["season"] < season, "margin"].to_numpy(dtype=float)
        prior = np.rint(prior)
        shapes[season] = prior - prior.mean()
    return shapes


def provisional_candidate_hists(frame: pd.DataFrame, engine_shapes: dict[int, np.ndarray]) -> list[dict[int, float]]:
    hists = []
    for season, center in zip(frame["season"], frame["predicted_margin_at_open"], strict=True):
        hists.append(recenter_hist(engine_shapes[int(season)], float(center)))
    return hists


def build_team_conditioned_hists(
    frame: pd.DataFrame, game_features: pd.DataFrame, n_reps: int
) -> tuple[list[dict[int, float]], list[dict[int, float]], list[dict[int, float]]]:
    ratings = game_features.loc[game_features["game_type"] == "REG", RATING_COLS].set_index("game_id")
    primary: list[dict[int, float]] = []
    shift: list[dict[int, float]] = []
    tilt: list[dict[int, float]] = []
    order: list[int] = []
    for season in GRADED_SEASONS:
        train_seasons = tuple(range(FIRST_TRAIN_SEASON, season))
        tables = build_tables(train_seasons, condition_on_team=True)
        rng = np.random.default_rng(RNG_SEED + season)
        league_off = tables["league_off_mean"]
        league_def = tables["league_def_mean"]
        season_rows = frame.loc[frame["season"] == season]
        for idx, row in season_rows.iterrows():
            r = ratings.loc[row["game_id"]]
            home_off = float(r["home_off_epa_per_play"]) if pd.notna(r["home_off_epa_per_play"]) else league_off
            home_def = float(r["home_def_epa_per_play"]) if pd.notna(r["home_def_epa_per_play"]) else league_def
            away_off = float(r["away_off_epa_per_play"]) if pd.notna(r["away_off_epa_per_play"]) else league_off
            away_def = float(r["away_def_epa_per_play"]) if pd.notna(r["away_def_epa_per_play"]) else league_def
            home_ratings = {"off": home_off, "def": home_def}
            away_ratings = {"off": away_off, "def": away_def}
            sim = simulate(n_reps, rng, tables, home_ratings=home_ratings, away_ratings=away_ratings)
            margins = np.rint(sim["margin"].to_numpy(dtype=float))
            values, counts = np.unique(margins, return_counts=True)
            primary.append(dict(zip(values.astype(int).tolist(), (counts / counts.sum()).tolist(), strict=True)))
            shape_dev = margins - margins.mean()
            shift.append(recenter_hist(shape_dev, float(row["predicted_margin_at_open"])))
            tilt.append(tilt_hist(margins, float(row["predicted_margin_at_open"])))
            order.append(idx)
    if order != list(frame.index):
        raise ValueError("team-conditioned candidate row order does not match frame order")
    return primary, shift, tilt


def positive_control_hists(
    frame: pd.DataFrame, historical_shapes: dict[int, np.ndarray], alpha: float
) -> list[dict[int, float]]:
    hists = []
    for season, predicted, result in zip(
        frame["season"], frame["predicted_margin_at_open"], frame["result"], strict=True
    ):
        center = (1.0 - alpha) * float(predicted) + alpha * float(result)
        hists.append(recenter_hist(historical_shapes[int(season)], center))
    return hists


def hists_to_three_way(hists: list[dict[int, float]], lines: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cover = np.empty(len(hists))
    push = np.empty(len(hists))
    loss = np.empty(len(hists))
    for i, (hist, line) in enumerate(zip(hists, lines, strict=True)):
        cover[i], push[i], loss[i] = hist_to_three_way(hist, float(line))
    return cover, push, loss


def week_blocked_bootstrap(
    frame: pd.DataFrame, delta_log_loss: np.ndarray, delta_brier: np.ndarray, n_boot: int, rng: np.random.Generator
) -> dict:
    block_key = frame["season"].astype(str) + "_" + frame["week"].astype(str)
    blocks = block_key.to_numpy()
    unique_blocks = np.unique(blocks)
    block_index = {b: np.where(blocks == b)[0] for b in unique_blocks}
    n_blocks = len(unique_blocks)
    boot_log_loss = np.empty(n_boot)
    boot_brier = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_index[p] for p in picked])
        boot_log_loss[b] = delta_log_loss[idx].mean()
        boot_brier[b] = delta_brier[idx].mean()
    return {
        "log_loss_delta_mean": float(boot_log_loss.mean()),
        "log_loss_delta_ci95": [float(np.percentile(boot_log_loss, 2.5)), float(np.percentile(boot_log_loss, 97.5))],
        "log_loss_probability_positive": float(np.mean(boot_log_loss > 0.0)),
        "brier_delta_mean": float(boot_brier.mean()),
        "brier_delta_ci95": [float(np.percentile(boot_brier, 2.5)), float(np.percentile(boot_brier, 97.5))],
        "brier_probability_positive": float(np.mean(boot_brier > 0.0)),
        "n_blocks": int(n_blocks),
        "n_boot": int(n_boot),
    }


def reliability_table(p_cover: np.ndarray, outcome: np.ndarray) -> list[dict]:
    order = np.argsort(p_cover)
    n = len(p_cover)
    deciles = np.array_split(order, 10)
    rows = []
    is_cover = (outcome == "cover").astype(float)
    for i, idx in enumerate(deciles):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "decile": i + 1,
                "n_games": int(len(idx)),
                "mean_predicted_cover_probability": float(p_cover[idx].mean()),
                "observed_cover_rate": float(is_cover[idx].mean()),
            }
        )
    return rows


def push_calibration(frame: pd.DataFrame, p_push: np.ndarray, label: str) -> list[dict]:
    rows = []
    lines = frame["line"].to_numpy(dtype=float)
    outcome = frame["outcome"].to_numpy()
    for key in KEY_NUMBERS:
        mask = np.abs(np.abs(lines) - key) < ATOM_TOLERANCE
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "candidate": label,
                "key_number": int(key),
                "n_games": n,
                "mean_predicted_push_probability": float(p_push[mask].mean()),
                "observed_push_rate": float((outcome[mask] == "push").mean()),
            }
        )
    return rows


def summarize_candidate(
    frame: pd.DataFrame,
    label: str,
    p_cover: np.ndarray,
    p_push: np.ndarray,
    p_loss: np.ndarray,
    base_log_loss: np.ndarray,
    base_brier: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    outcome = frame["outcome"].to_numpy()
    log_loss, brier = three_way_score(p_cover, p_push, p_loss, outcome)
    delta_log_loss = base_log_loss - log_loss
    delta_brier = base_brier - brier
    per_season = []
    for season, group_idx in frame.groupby("season").groups.items():
        idx = frame.index.get_indexer(group_idx)
        per_season.append(
            {
                "season": int(season),
                "n_games": int(len(idx)),
                "log_loss_delta_mean": float(delta_log_loss[idx].mean()),
                "brier_delta_mean": float(delta_brier[idx].mean()),
            }
        )
    boot = week_blocked_bootstrap(frame, delta_log_loss, delta_brier, N_BOOT, rng)
    return {
        "candidate": label,
        "n_games": int(len(frame)),
        "pooled_log_loss_mean": float(log_loss.mean()),
        "pooled_brier_mean": float(brier.mean()),
        "pooled_log_loss_delta_vs_baseline": float(delta_log_loss.mean()),
        "pooled_brier_delta_vs_baseline": float(delta_brier.mean()),
        "per_season": per_season,
        "week_blocked_bootstrap": boot,
        "reliability_table_cover_deciles": reliability_table(p_cover, outcome),
    }, log_loss, brier


def fit_blend(
    frame: pd.DataFrame,
    baseline_cover: np.ndarray,
    baseline_loss: np.ndarray,
    baseline_push: np.ndarray,
    tilt_cover: np.ndarray,
    tilt_loss: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    served_cond = baseline_cover / np.clip(baseline_cover + baseline_loss, EPS, None)
    tilt_cond = tilt_cover / np.clip(tilt_cover + tilt_loss, EPS, None)
    x_served = safe_logit(served_cond)
    x_tilt = safe_logit(tilt_cond)
    outcome = frame["outcome"].to_numpy()
    seasons = frame["season"].to_numpy()
    nonpush = outcome != "push"
    y_cover = (outcome == "cover").astype(int)
    blend_cond = np.empty(len(frame))
    fold_coeffs: list[dict] = []
    for season in GRADED_SEASONS:
        train_mask = (seasons != season) & nonpush
        test_mask = seasons == season
        x_train = np.column_stack([x_served[train_mask], x_tilt[train_mask]])
        y_train = y_cover[train_mask]
        model = LogisticRegression(C=1e10, max_iter=2000)
        model.fit(x_train, y_train)
        x_test = np.column_stack([x_served[test_mask], x_tilt[test_mask]])
        blend_cond[test_mask] = model.predict_proba(x_test)[:, 1]
        fold_coeffs.append(
            {
                "held_out_season": int(season),
                "n_train": int(train_mask.sum()),
                "intercept": float(model.intercept_[0]),
                "coef_served_logit": float(model.coef_[0][0]),
                "coef_tilt_logit": float(model.coef_[0][1]),
            }
        )
    p_push_blend = baseline_push.copy()
    p_cover_blend = (1.0 - p_push_blend) * blend_cond
    p_loss_blend = (1.0 - p_push_blend) * (1.0 - blend_cond)
    return p_cover_blend, p_push_blend, p_loss_blend, fold_coeffs


def push_split(label: str, log_loss: np.ndarray, frame: pd.DataFrame) -> dict:
    outcome = frame["outcome"].to_numpy()
    push_mask = outcome == "push"
    return {
        "candidate": label,
        "push_log_loss_mean": float(log_loss[push_mask].mean()),
        "n_push": int(push_mask.sum()),
        "nonpush_log_loss_mean": float(log_loss[~push_mask].mean()),
        "n_nonpush": int((~push_mask).sum()),
    }


def cover_miss_delta(
    label: str, base_log_loss: np.ndarray, cand_log_loss: np.ndarray, frame: pd.DataFrame, rng: np.random.Generator
) -> dict:
    outcome = frame["outcome"].to_numpy()
    nonpush = outcome != "push"
    sub_frame = frame.loc[nonpush]
    delta = (base_log_loss - cand_log_loss)[nonpush]
    boot = week_blocked_bootstrap(sub_frame, delta, delta, N_BOOT, rng)
    return {
        "candidate": label,
        "n_games": int(nonpush.sum()),
        "log_loss_delta_mean": float(delta.mean()),
        "ci95": boot["log_loss_delta_ci95"],
        "probability_positive": boot["log_loss_probability_positive"],
    }


def disagreement_report(
    label: str,
    cand_cover: np.ndarray,
    cand_loss: np.ndarray,
    base_cover: np.ndarray,
    base_loss: np.ndarray,
    frame: pd.DataFrame,
) -> dict:
    outcome = frame["outcome"].to_numpy()
    nonpush = outcome != "push"
    cand_pick = (cand_cover[nonpush] > cand_loss[nonpush]).astype(int)
    base_pick = (base_cover[nonpush] > base_loss[nonpush]).astype(int)
    actual = (outcome[nonpush] == "cover").astype(int)
    agree = cand_pick == base_pick
    disagree = ~agree
    n_dis = int(disagree.sum())
    cand_acc = float((cand_pick[disagree] == actual[disagree]).mean()) if n_dis else float("nan")
    base_acc = float((base_pick[disagree] == actual[disagree]).mean()) if n_dis else float("nan")
    return {
        "candidate": label,
        "n_nonpush": int(nonpush.sum()),
        "side_agreement_rate": float(agree.mean()),
        "n_disagreements": n_dis,
        "candidate_forced_pick_accuracy_on_disagreements": cand_acc,
        "served_forced_pick_accuracy_on_disagreements": base_acc,
    }


def main() -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_games()
    game_features = pd.read_parquet(GAME_FEATURES_PATH)

    baseline_cover, baseline_push, baseline_loss = baseline_three_way(frame)
    outcome = frame["outcome"].to_numpy()
    base_log_loss, base_brier = three_way_score(baseline_cover, baseline_push, baseline_loss, outcome)

    rng = np.random.default_rng(RNG_SEED)

    engine_shapes = build_engine_shapes(game_features)
    provisional_hists = provisional_candidate_hists(frame, engine_shapes)
    lines = frame["line"].to_numpy(dtype=float)
    prov_cover, prov_push, prov_loss = hists_to_three_way(provisional_hists, lines)
    prov_summary, prov_log_loss, prov_brier = summarize_candidate(
        frame, "provisional_engine_recentered", prov_cover, prov_push, prov_loss, base_log_loss, base_brier, rng
    )

    team_cond_primary_hists, team_cond_shift_hists, team_cond_tilt_hists = build_team_conditioned_hists(
        frame, game_features, TEAM_COND_N_REPS
    )
    tc_primary_cover, tc_primary_push, tc_primary_loss = hists_to_three_way(team_cond_primary_hists, lines)
    tc_primary_summary, tc_primary_log_loss, tc_primary_brier = summarize_candidate(
        frame, "team_conditioned_raw", tc_primary_cover, tc_primary_push, tc_primary_loss,
        base_log_loss, base_brier, rng,
    )
    tc_shift_cover, tc_shift_push, tc_shift_loss = hists_to_three_way(team_cond_shift_hists, lines)
    tc_shift_summary, tc_shift_log_loss, tc_shift_brier = summarize_candidate(
        frame, "team_conditioned_shift", tc_shift_cover, tc_shift_push, tc_shift_loss,
        base_log_loss, base_brier, rng,
    )
    tc_tilt_cover, tc_tilt_push, tc_tilt_loss = hists_to_three_way(team_cond_tilt_hists, lines)
    tc_tilt_summary, tc_tilt_log_loss, tc_tilt_brier = summarize_candidate(
        frame, "team_conditioned_tilt", tc_tilt_cover, tc_tilt_push, tc_tilt_loss,
        base_log_loss, base_brier, rng,
    )
    blend_cover, blend_push, blend_loss, blend_fold_coeffs = fit_blend(
        frame, baseline_cover, baseline_loss, baseline_push, tc_tilt_cover, tc_tilt_loss
    )
    blend_summary, blend_log_loss, blend_brier = summarize_candidate(
        frame, "team_conditioned_tilt_served_blend", blend_cover, blend_push, blend_loss,
        base_log_loss, base_brier, rng,
    )

    candidates = [
        ("team_conditioned_raw", tc_primary_cover, tc_primary_push, tc_primary_loss, tc_primary_log_loss, tc_primary_summary),
        ("team_conditioned_shift", tc_shift_cover, tc_shift_push, tc_shift_loss, tc_shift_log_loss, tc_shift_summary),
        ("team_conditioned_tilt", tc_tilt_cover, tc_tilt_push, tc_tilt_loss, tc_tilt_log_loss, tc_tilt_summary),
        ("team_conditioned_tilt_served_blend", blend_cover, blend_push, blend_loss, blend_log_loss, blend_summary),
    ]
    push_split_report = [push_split(label, ll, frame) for label, _, _, _, ll, _ in candidates]
    cover_miss_report = [
        cover_miss_delta(label, base_log_loss, ll, frame, rng) for label, _, _, _, ll, _ in candidates
    ]
    disagreement_reports = [
        disagreement_report(label, cov, los, baseline_cover, baseline_loss, frame)
        for label, cov, _, los, _, _ in candidates
    ]
    best_label, _, _, _, _, best_summary = min(candidates, key=lambda c: c[5]["pooled_log_loss_mean"])
    best_read = {
        "candidate": best_label,
        "reliability_table_cover_deciles": best_summary["reliability_table_cover_deciles"],
    }

    historical_shapes = build_historical_shapes(game_features)
    pc_results = []
    for alpha in PC_ALPHAS:
        pc_hists = positive_control_hists(frame, historical_shapes, alpha)
        pc_cover, pc_push, pc_loss = hists_to_three_way(pc_hists, lines)
        pc_summary, pc_log_loss, pc_brier = summarize_candidate(
            frame, f"positive_control_alpha_{alpha}", pc_cover, pc_push, pc_loss, base_log_loss, base_brier, rng
        )
        pc_results.append(pc_summary)

    push_cal = (
        push_calibration(frame, baseline_push, "baseline")
        + push_calibration(frame, prov_push, "provisional_engine_recentered")
        + push_calibration(frame, tc_primary_push, "team_conditioned_raw")
        + push_calibration(frame, tc_shift_push, "team_conditioned_shift")
        + push_calibration(frame, tc_tilt_push, "team_conditioned_tilt")
        + push_calibration(frame, blend_push, "team_conditioned_tilt_served_blend")
    )

    per_game = frame[
        ["game_id", "season", "week", "gameday", "home_team", "away_team", "line", "result", "outcome",
         "predicted_margin_at_open"]
    ].copy()
    per_game["baseline_cover"] = baseline_cover
    per_game["baseline_push"] = baseline_push
    per_game["baseline_loss"] = baseline_loss
    per_game["baseline_log_loss"] = base_log_loss
    per_game["baseline_brier"] = base_brier
    per_game["provisional_cover"] = prov_cover
    per_game["provisional_push"] = prov_push
    per_game["provisional_loss"] = prov_loss
    per_game["provisional_log_loss"] = prov_log_loss
    per_game["provisional_brier"] = prov_brier
    per_game["team_conditioned_raw_cover"] = tc_primary_cover
    per_game["team_conditioned_raw_push"] = tc_primary_push
    per_game["team_conditioned_raw_loss"] = tc_primary_loss
    per_game["team_conditioned_raw_log_loss"] = tc_primary_log_loss
    per_game["team_conditioned_raw_brier"] = tc_primary_brier
    per_game["team_conditioned_shift_cover"] = tc_shift_cover
    per_game["team_conditioned_shift_push"] = tc_shift_push
    per_game["team_conditioned_shift_loss"] = tc_shift_loss
    per_game["team_conditioned_shift_log_loss"] = tc_shift_log_loss
    per_game["team_conditioned_shift_brier"] = tc_shift_brier
    per_game["team_conditioned_tilt_cover"] = tc_tilt_cover
    per_game["team_conditioned_tilt_push"] = tc_tilt_push
    per_game["team_conditioned_tilt_loss"] = tc_tilt_loss
    per_game["team_conditioned_tilt_log_loss"] = tc_tilt_log_loss
    per_game["team_conditioned_tilt_brier"] = tc_tilt_brier
    per_game["blend_cover"] = blend_cover
    per_game["blend_push"] = blend_push
    per_game["blend_loss"] = blend_loss
    per_game["blend_log_loss"] = blend_log_loss
    per_game["blend_brier"] = blend_brier
    per_game.to_parquet(out_dir / "per_game.parquet", index=False)

    report = {
        "generated_at": timestamp,
        "harness": "scripts/sim04_loso.py",
        "opener_source": str(OPENER_PATH.relative_to(REPO)),
        "n_games": int(len(frame)),
        "graded_seasons": list(GRADED_SEASONS),
        "sign_convention": {
            "measured": (
                "home covers when result (home_score - away_score) > tue_open_home_spread, "
                "push when equal, loss when less; verified by reading clv.py prior_pool (line = "
                "tue_open_home_spread with no negation, margin.py:722-724 predicted_margin = spread "
                "+ raw with no negation) and by data: frac(result>line)=0.4854 close to 0.5 versus "
                "frac(result>-line)=0.5511 (measured on this per_game.parquet); "
                "mean(tue_open_home_spread)=1.4115 approx mean(result)=1.6396 (measured)."
            )
        },
        "baseline": {
            "reused_columns": [
                "home_cover_probability_excluding_push_at_open",
                "push_probability_at_open",
                "home_loss_probability_at_open",
            ],
            "reproduction_check": (
                "read: clv.py:2154-2169 calls serve_discrete_three_way(served_at_open, at_open, "
                "reader, ...) where reader = DiscretePushReader.for_week(discrete_pool, ...) built "
                "per week at clv.py:2105, and the three stored columns are assigned directly from "
                "that discrete result at clv.py:2168-2169. This is the exact production opener-grading "
                "code path (mass_preserving_lattice.py:216 for_week, :394 serve_discrete_three_way), "
                "not a re-derivation, so the stored columns are reused as-is."
            ),
            "pooled_log_loss_mean": float(base_log_loss.mean()),
            "pooled_brier_mean": float(base_brier.mean()),
        },
        "predicted_margin_at_open": {
            "measured": (
                "not stored as its own column; reconstructed as tue_open_home_spread + "
                "residual_at_open_served per margin.py:722-724 (predicted_margin = spread + raw "
                "residual) and clv.py:2184 (residual_at_open_served = residual_at_open + "
                "home_side_offset_at_open). This omits the small per-week 'location' shift that "
                "serve_discrete_three_way adds on top (clv.py mass_preserving_lattice.py:407-409); "
                "that shift is not stored per game."
            )
        },
        "provisional_candidate": prov_summary,
        "team_conditioning": {
            "measured": (
                "engine.py build_tables(seasons, condition_on_team=True) joins each training "
                "transition row's offense/defense to its pregame home_/away_off_epa_per_play and "
                "home_/away_def_epa_per_play from game_features_pbp.parquet (posteam==home_team "
                "selects the home columns), missing (early 2009) rows filled with the training "
                "window's own league mean. Draw: k_state=200 state neighbours via a dedicated scipy "
                "cKDTree per (down,phase) (same features/scales as the existing k=40 pool), resampled "
                "with kernel weight w=exp(-((off_row-off_sim)^2+(def_row-def_sim)^2)/(2h^2)) * "
                "(1.5 if home flag matches else 1). h=0.5*cross-team SD of off EPA in the training "
                "window (predeclared formula, computed fresh per fold, not tuned on graded results). "
                "Unconditioned mode (home_ratings/away_ratings=None) is byte-for-byte the prior code "
                "path (verified: engine sanity re-run reproduced unit 3d exactly: hits=2/5, log-loss "
                "delta +0.004451, SD ratio 1.0862, points/game 41.77)."
            ),
            "conditioned_sanity_2015_2017": {
                "n_games": 768,
                "reps_per_game": 60,
                "correlation_sim_mean_margin_vs_actual_margin": 0.288,
                "implied_home_field_edge_equal_rating_teams": 0.5615,
                "actual_2009_2014_mean_margin_home_field_reference": 2.5658,
            },
            "n_reps_per_game_requested": TEAM_COND_N_REPS_REQUESTED,
            "n_reps_per_game_used": TEAM_COND_N_REPS,
            "n_reps_reduction_reason": (
                "SIM-08 unit 7 re-benchmark on the current engine (post OT-pool/4th-down-layer/"
                "team-yard-shift): 30.5 games/sec on 2020's table (11-season train window), "
                "27.3 games/sec on 2025's table (16-season train window), both measured on 15 games "
                "x 40 reps, tests/scratch/sim08_unit7_timing.py. 1537 games x 1000 reps would run "
                "~14 hours at this throughput, far over the ~3 hour budget; 180 reps/game estimates "
                "to ~2.7-2.8 hours using the slower 2025-table rate, applied uniformly to every game "
                "and every candidate (raw, shift and tilt share the same draws). Monte Carlo cost at "
                "180 draws, (K-1)/2N with K=3 outcomes: 2/360 = 0.00556 in three-way log loss."
            ),
        },
        "team_conditioned_raw": tc_primary_summary,
        "team_conditioned_shift": tc_shift_summary,
        "team_conditioned_tilt": tc_tilt_summary,
        "team_conditioned_tilt_served_blend": blend_summary,
        "blend_fold_coefficients": blend_fold_coeffs,
        "push_vs_nonpush_log_loss": push_split_report,
        "cover_vs_miss_delta_pushes_excluded": cover_miss_report,
        "disagreement_report": disagreement_reports,
        "best_read_reliability_table": best_read,
        "positive_control": {
            "mechanism": (
                "historical pooled margin shape (real REG results, seasons strictly before the "
                "graded season) recentered at (1-alpha)*predicted_margin_at_open + alpha*result; "
                "alpha=0 reduces to the same center as the provisional candidate's engine shape "
                "and is expected to show ~no improvement over baseline, alpha>0 leaks a known small "
                "amount of the realized outcome"
            ),
            "alpha_sweep": pc_results,
        },
        "push_probability_calibration_at_key_numbers": push_cal,
        "engine_training_windows": {
            str(season): {"train_seasons": [FIRST_TRAIN_SEASON, season - 1], "n_simulated_games": ENGINE_N_GAMES}
            for season in GRADED_SEASONS
        },
    }

    descriptions = {
        "team_conditioned_raw": "Play-level simulator margin histogram, team-conditioned on pregame EPA with exact home/away play matching, graded leave-one-season-out at the Tuesday opener against the served discrete three-way read; three-way log loss, week-blocked bootstrap",
        "team_conditioned_shift": "Same simulator histogram shape re-centred (rounded shift) on the served predicted margin, graded at the Tuesday opener against the served discrete three-way read; three-way log loss, week-blocked bootstrap",
        "team_conditioned_tilt": "Same simulator histogram shape exponentially tilted (integer-support preserving, as in sim04_week.py) to the served predicted margin, graded at the Tuesday opener against the served discrete three-way read; three-way log loss, week-blocked bootstrap",
        "team_conditioned_tilt_served_blend": "Logistic blend of the served cover logit and the tilted-sim cover logit (weights fit leave-one-season-out on 2020-2025), push probability kept at the served read, graded at the Tuesday opener against the served discrete three-way read; three-way log loss, week-blocked bootstrap",
    }
    record_drafts = {}
    for label, _, _, _, _, summary in candidates:
        ci_low, ci_high = summary["week_blocked_bootstrap"]["log_loss_delta_ci95"]
        if ci_high < 0.0:
            classification_args = (
                "--classification refuted_mechanism --closing-ground wrong_sign_resolved "
                "--classification-evidence \"whole week-blocked bootstrap CI for the log-loss delta vs the served read is below zero\" "
            )
        else:
            classification_args = "--classification unresolved_below_power "
        record_drafts[label] = (
            "F:/Repos/nfl_py3/.venv/Scripts/python -m nfl_ats weak-signals record "
            f"--name sim04_engine_{label}_vs_discrete_read --league nfl --effect-units log_loss_improvement "
            f"--effect {summary['pooled_log_loss_delta_vs_baseline']} "
            f"--interval-low {ci_low} --interval-high {ci_high} "
            f"--probability-positive {summary['week_blocked_bootstrap']['log_loss_probability_positive']} "
            f"--sample-games {summary['n_games']} "
            f"--sample-blocks {summary['week_blocked_bootstrap']['n_blocks']} "
            "--season-start 2020 --season-end 2025 --family sim04_play_simulator "
            + classification_args
            + f"--source artifacts/sim04_loso/{timestamp}/report.json "
            f"--description \"{descriptions[label]}\" "
        )
    report["record_command_drafts"] = record_drafts

    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Artifact directory: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
