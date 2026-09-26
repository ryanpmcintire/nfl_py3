from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

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

    historical_shapes = build_historical_shapes(game_features)
    pc_results = []
    for alpha in PC_ALPHAS:
        pc_hists = positive_control_hists(frame, historical_shapes, alpha)
        pc_cover, pc_push, pc_loss = hists_to_three_way(pc_hists, lines)
        pc_summary, pc_log_loss, pc_brier = summarize_candidate(
            frame, f"positive_control_alpha_{alpha}", pc_cover, pc_push, pc_loss, base_log_loss, base_brier, rng
        )
        pc_results.append(pc_summary)

    push_cal = push_calibration(frame, baseline_push, "baseline") + push_calibration(
        frame, prov_push, "provisional_engine_recentered"
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
        "record_command_draft": (
            "F:/Repos/nfl_py3/.venv/Scripts/python -m nfl_ats weak-signals record "
            "--effect-units log_loss_improvement --probability-positive <fill from final candidate run> "
            "--signal sim04_engine_conditioned_margin_vs_discrete_read "
            "--artifact artifacts/sim04_loso/<timestamp>/report.json "
            "--note 'not run yet: this LOSO grade is provisional (league-average engine shape, no team "
            "conditioning); do not record until team-conditioned candidate replaces it'"
        ),
    }

    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Artifact directory: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
