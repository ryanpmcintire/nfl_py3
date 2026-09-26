from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim04_engine import build_tables, simulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nfl_ats.board_content import load_board_content, load_public_board_artifacts
from nfl_ats.cli_common import _artifacts_root, _data_root

SEASONS = tuple(range(2009, 2026))
N_GAMES = 2000
MARGIN_CLIP = 60
RATING_COLS = (
    "home_off_epa_per_play",
    "home_def_epa_per_play",
    "away_off_epa_per_play",
    "away_def_epa_per_play",
)


def _ratings_for_game(
    ratings: pd.DataFrame, game_id: str, league_off: float, league_def: float
) -> tuple[dict[str, float], dict[str, float]] | None:
    if game_id not in ratings.index:
        return None
    row = ratings.loc[game_id]

    def pick(column: str, default: float) -> float:
        value = row[column]
        return float(value) if pd.notna(value) else default

    home_ratings = {
        "off": pick("home_off_epa_per_play", league_off),
        "def": pick("home_def_epa_per_play", league_def),
    }
    away_ratings = {
        "off": pick("away_off_epa_per_play", league_off),
        "def": pick("away_def_epa_per_play", league_def),
    }
    return home_ratings, away_ratings


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


def build_week_margins(artifacts_root: Path, data_root: Path, rng: np.random.Generator) -> dict:
    board = load_board_content(artifacts_root, data_root=data_root)
    artifacts = load_public_board_artifacts(artifacts_root)
    predicted_margin_by_id = (
        artifacts.predictions.set_index("game_id")["predicted_margin"].to_dict()
        if "predicted_margin" in artifacts.predictions.columns
        else {}
    )
    features = pd.read_parquet(data_root / "processed" / "game_features_pbp.parquet")
    ratings = features.loc[features["game_type"] == "REG", ["game_id", *RATING_COLS]].set_index(
        "game_id"
    )
    tables = build_tables(SEASONS, condition_on_team=True)
    league_off = float(tables["league_off_mean"])
    league_def = float(tables["league_def_mean"])

    games_out: dict[str, dict] = {}
    for game in board.games:
        center = predicted_margin_by_id.get(game.game_id)
        if center is None or pd.isna(center):
            continue
        center = float(center)
        pair = _ratings_for_game(ratings, game.game_id, league_off, league_def)
        if pair is None:
            continue
        home_ratings, away_ratings = pair
        sim = simulate(N_GAMES, rng, tables, home_ratings=home_ratings, away_ratings=away_ratings)
        margins = np.clip(np.rint(sim["margin"].to_numpy(dtype=float)), -MARGIN_CLIP, MARGIN_CLIP)
        values, raw_counts = np.unique(margins.astype(int), return_counts=True)
        weights = tilt_to_mean(values.astype(float), raw_counts.astype(float), center)
        counts = np.rint(weights / weights.sum() * N_GAMES).astype(int)
        histogram = {str(int(v)): int(c) for v, c in zip(values, counts, strict=True) if c > 0}
        games_out[game.game_id] = {
            "home": game.home,
            "away": game.away,
            "pick_team": game.pick_team,
            "market_spread": game.market_spread,
            "recentred_home_margin": center,
            "home_margin_histogram": histogram,
        }

    generated = datetime.now(UTC)
    return {
        "generated_at_utc": generated.isoformat(),
        "lineage": {
            "active_model_id": board.headline.model_id,
            "seasons": list(SEASONS),
            "n": N_GAMES,
        },
        "games": games_out,
    }


def main() -> None:
    artifacts_root = _artifacts_root()
    data_root = _data_root()
    rng = np.random.default_rng(404)
    payload = build_week_margins(artifacts_root, data_root, rng)
    generated = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = artifacts_root / "sim04_week" / generated
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "margins.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
