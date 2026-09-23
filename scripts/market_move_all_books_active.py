from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import market_move_decomposition as mmd
import numpy as np
import pandas as pd

from nfl_ats.pick_probability import (
    FLAG_SUM_COLUMN,
    MARKET_MOVE_FEATURE_LEGACY,
    MARKET_MOVE_FEATURE_SUNDAY,
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
)
from nfl_ats.pick_probability_fit import build_fit_population
from nfl_ats.sharp_book_movement_features import (
    LEADER_BOOKS,
    LEADERSHIP_WEIGHTS,
    sharp_book_movement_features,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO / "artifacts"
DATA_ROOT = REPO / "data"
QUOTE_CACHE = ARTIFACTS_ROOT / "sharp_book_weighted_movement" / "spread_quotes.parquet"
ALL_BOOKS = tuple(sorted(LEADERSHIP_WEIGHTS))
ARCHIVE_SEASONS = (2023, 2024, 2025)
FOUR_TERM_FEATURES = ("model_logit", FLAG_SUM_COLUMN, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN)


def _sunday_style_bounds(games: pd.DataFrame) -> pd.DataFrame:
    local = pd.to_datetime(games.week_first_commence_utc, utc=True).dt.tz_convert(
        "America/New_York"
    )
    sunday = local.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(
        (6 - local.dt.weekday) % 7, unit="D"
    )
    monday = (sunday - pd.Timedelta(days=6)).dt.tz_localize("America/New_York").dt.tz_convert(
        "UTC"
    )
    wednesday = (
        (sunday - pd.Timedelta(days=4)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    as_of = (
        (sunday + pd.Timedelta(hours=12, minutes=45))
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    bounds = games[["game_id", "commence_time_utc"]].copy()
    bounds["monday"] = monday.to_numpy()
    bounds["wednesday"] = wednesday.to_numpy()
    bounds["cutoff"] = (
        pd.concat([pd.to_datetime(games.commence_time_utc, utc=True), as_of], axis=1)
        .min(axis=1)
        .to_numpy()
    )
    return bounds


def _sunday_style_move(
    quotes: pd.DataFrame, games: pd.DataFrame, books: tuple[str, ...]
) -> pd.DataFrame:
    bounds = _sunday_style_bounds(games)
    q = quotes.loc[
        quotes.bookmaker_key.isin(books),
        [
            "nflverse_game_id",
            "bookmaker_key",
            "home_spread_line",
            "observed_at_utc",
            "bookmaker_last_update_utc",
        ],
    ].rename(columns={"nflverse_game_id": "game_id"})
    q["game_id"] = q["game_id"].astype(str)
    q = q.merge(bounds, on="game_id", validate="many_to_one")
    q = q.loc[
        q.observed_at_utc.ge(q.monday)
        & q.observed_at_utc.lt(q.cutoff)
        & q.bookmaker_last_update_utc.le(q.observed_at_utc)
        & np.isfinite(q.home_spread_line)
    ].copy()
    keys = ["game_id", "bookmaker_key", "observed_at_utc"]
    if q.groupby(keys).home_spread_line.nunique().gt(1).any():
        raise ValueError("Conflicting book lines at one observed timestamp")
    q = q.sort_values(keys).drop_duplicates(keys)
    q["move"] = q.groupby(["game_id", "bookmaker_key"]).home_spread_line.diff()
    q = q.loc[q.observed_at_utc.ge(q.wednesday) & q.move.notna()]
    per_book = q.groupby(["game_id", "bookmaker_key"], as_index=False).move.sum()
    return (
        per_book.groupby("game_id")
        .agg(move=("move", "median"), books=("move", "size"))
        .reset_index()
    )


def main() -> None:
    population_active, provenance_active = build_fit_population(
        ARTIFACTS_ROOT, DATA_ROOT, market_move_feature_version=MARKET_MOVE_FEATURE_SUNDAY
    )
    population_active = population_active.copy()
    population_active["game_id"] = population_active["game_id"].astype(str)

    population_legacy, _provenance_legacy = build_fit_population(
        ARTIFACTS_ROOT, DATA_ROOT, market_move_feature_version=MARKET_MOVE_FEATURE_LEGACY
    )
    population_legacy = population_legacy.copy()
    population_legacy["game_id"] = population_legacy["game_id"].astype(str)

    quotes = pd.read_parquet(QUOTE_CACHE)
    quotes["nflverse_game_id"] = quotes["nflverse_game_id"].astype(str)
    quotes = quotes.loc[
        quotes.decision_label.eq("intraday_hourly")
        & quotes.archive_season.isin(ARCHIVE_SEASONS)
        & quotes.nflverse_game_id.isin(set(population_legacy.game_id))
    ].copy()
    games = quotes.groupby("nflverse_game_id", as_index=False).agg(
        commence_time_utc=("commence_time_utc", "min")
    )
    games = games.rename(columns={"nflverse_game_id": "game_id"}).merge(
        population_legacy[["game_id", "season", "week"]], on="game_id", validate="one_to_one"
    )
    games["week_first_commence_utc"] = games.groupby(
        ["season", "week"]
    ).commence_time_utc.transform("min")

    baseline = sharp_book_movement_features(quotes, games.copy())[
        ["game_id", "leader_median_net_move", "leader_books"]
    ]
    joined = population_legacy.merge(baseline, on="game_id", how="left", validate="one_to_one")
    exposed = joined[MOVE_AVAILABLE_COLUMN].eq(1.0)
    parity_error = (joined["leader_median_net_move"] - joined[MOVE_COLUMN]).abs()
    legacy_baseline_parity_max = float(parity_error.loc[exposed].max()) if exposed.any() else None
    eligible = exposed & parity_error.le(1e-9)

    leader_active = _sunday_style_move(quotes, games, LEADER_BOOKS).rename(
        columns={"move": "sunday_move", "books": "sunday_books"}
    )
    joined = joined.merge(leader_active, on="game_id", how="left")
    joined["reconstructed_active_move"] = joined[MOVE_COLUMN].where(~eligible, joined.sunday_move)

    reconstructed = joined.loc[exposed, ["game_id", "reconstructed_active_move"]]
    check = population_active[["game_id", MOVE_COLUMN]].merge(
        reconstructed, on="game_id", how="inner"
    )
    diff = (check[MOVE_COLUMN] - check["reconstructed_active_move"]).abs()
    parity = {
        "n_games_compared": len(check),
        "max_abs_diff_points": float(diff.max()) if len(diff) else None,
        "mean_abs_diff_points": float(diff.mean()) if len(diff) else None,
        "legacy_baseline_parity_max_abs": legacy_baseline_parity_max,
        "n_exposed_source_legacy": int(exposed.sum()),
        "n_eligible_recomputed_via_sunday_window": int(eligible.sum()),
        "n_active_move_available_current": int(population_active[MOVE_AVAILABLE_COLUMN].sum()),
    }

    all_books_active = _sunday_style_move(quotes, games, ALL_BOOKS).rename(
        columns={"move": "all_books_move", "books": "all_books_count"}
    )
    all_books_series = all_books_active.set_index("game_id")["all_books_move"]

    def _with_move(series: pd.Series) -> pd.DataFrame:
        frame = population_active.copy()
        mapped = frame.game_id.map(series)
        frame[MOVE_AVAILABLE_COLUMN] = mapped.notna().astype(float)
        frame[MOVE_COLUMN] = mapped.fillna(0.0)
        return frame

    served_p = mmd._loso_predict(population_active, FOUR_TERM_FEATURES)
    arm = mmd._score_arm(
        "e_all_books_median_active_window",
        _with_move(all_books_series),
        FOUR_TERM_FEATURES,
        served_p,
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACTS_ROOT / "market_move_decomposition" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "unit": "unit_3_all_books_active_window",
        "active_feature_version": MARKET_MOVE_FEATURE_SUNDAY,
        "quotes_cache": str(QUOTE_CACHE),
        "archive_seasons_used": list(ARCHIVE_SEASONS),
        "fit_population_provenance": {k: str(v) for k, v in provenance_active.items()},
        "reconstruction_parity_check": parity,
        "outer_seasons": mmd.OUTER_SEASONS,
        "n_fit_population": len(population_active),
        "arm": arm,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(out_dir)}, indent=2))
    print(
        json.dumps(
            {
                "parity": parity,
                "arm_summary": {k: v for k, v in arm.items() if k != "fold_coefficients"},
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
