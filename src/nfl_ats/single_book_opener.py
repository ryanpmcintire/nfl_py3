from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.clv import DataContractError, load_decision_quotes
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND

CANDIDATE_BOOKS: tuple[str, ...] = (
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "williamhill_us",
    "pointsbetus",
    "betrivers",
    "espnbet",
    "barstool",
    "unibet_us",
    "wynnbet",
    "superbook",
    "betonlineag",
    "bovada",
    "mybookieag",
    "lowvig",
    "betus",
    "betfair",
    "foxbet",
    "twinspires",
    "sugarhouse",
    "circasports",
    "hardrockbet",
    "fanatics",
    "ballybet",
)

OPENER_LABEL = "tue_open"
HALF_POINT_TOLERANCE = 1e-9

_QUOTE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "bookmaker_key",
    "home_spread_line",
    "observed_at_utc",
    "snapshot_timestamp_utc",
)


def is_half_point(lines: pd.Series | np.ndarray) -> np.ndarray:
    size = np.abs(pd.to_numeric(pd.Series(lines), errors="coerce").to_numpy(dtype=float))
    return np.asarray(np.abs((size % 1.0) - 0.5) < HALF_POINT_TOLERANCE, dtype=bool)


def opener_book_quotes(
    root: Path,
    *,
    schedule: pd.DataFrame,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    label: str = OPENER_LABEL,
    seasons: Iterable[int] | None = None,
) -> pd.DataFrame:
    required = {"game_id", "season", "week"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise DataContractError(f"Opener line schedule is missing columns: {', '.join(missing)}")
    quotes = load_decision_quotes(root, capture_kind=capture_kind, labels=(label,), seasons=seasons)
    if quotes.empty:
        return pd.DataFrame(columns=list(_QUOTE_COLUMNS))
    quotes = quotes.loc[
        quotes["market"].eq("spreads")
        & quotes["outcome_side"].eq("HOME")
        & quotes["nflverse_game_id"].notna()
        & quotes["observed_at_utc"].lt(quotes["commence_time_utc"])
    ].copy()
    truth = (
        schedule[["game_id", "season", "week"]]
        .drop_duplicates("game_id")
        .rename(columns={"game_id": "nflverse_game_id", "season": "_season", "week": "_week"})
    )
    quotes = quotes.merge(truth, on="nflverse_game_id", how="inner")
    quotes = quotes.loc[quotes["season"].eq(quotes["_season"]) & quotes["week"].eq(quotes["_week"])]
    quotes = quotes.loc[pd.to_numeric(quotes["home_spread_line"], errors="coerce").notna()]
    if quotes.empty:
        return pd.DataFrame(columns=list(_QUOTE_COLUMNS))
    deduped = (
        quotes.sort_values(["observed_at_utc", "snapshot_timestamp_utc"])
        .groupby(["nflverse_game_id", "bookmaker_key"], as_index=False, dropna=False)
        .tail(1)
        .rename(columns={"nflverse_game_id": "game_id"})
    )
    deduped["game_id"] = deduped["game_id"].astype(str)
    deduped["season"] = pd.to_numeric(deduped["season"], errors="coerce").astype(int)
    deduped["week"] = pd.to_numeric(deduped["week"], errors="coerce").astype(int)
    deduped["home_spread_line"] = pd.to_numeric(deduped["home_spread_line"], errors="coerce")
    return (
        deduped[list(_QUOTE_COLUMNS)]
        .sort_values(["season", "week", "game_id", "bookmaker_key"])
        .reset_index(drop=True)
    )


def book_coverage(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame(columns=["bookmaker_key", "games", "rows", "seasons", "half_points"])
    grouped = quotes.groupby("bookmaker_key", as_index=False).agg(
        games=("game_id", "nunique"),
        rows=("game_id", "size"),
        seasons=("season", "nunique"),
        first_season=("season", "min"),
        last_season=("season", "max"),
    )
    half = quotes.assign(half=is_half_point(quotes["home_spread_line"]))
    share = half.groupby("bookmaker_key", as_index=False).agg(half_point_share=("half", "mean"))
    merged = grouped.merge(share, on="bookmaker_key", how="left")
    merged["candidate"] = merged["bookmaker_key"].isin(set(CANDIDATE_BOOKS))
    return merged.sort_values(["games", "rows", "bookmaker_key"], ascending=[False, False, True])[
        [
            "bookmaker_key",
            "candidate",
            "games",
            "rows",
            "seasons",
            "first_season",
            "last_season",
            "half_point_share",
        ]
    ].reset_index(drop=True)


def choose_book(coverage: pd.DataFrame, candidates: Iterable[str] = CANDIDATE_BOOKS) -> str:
    pool = coverage.loc[coverage["bookmaker_key"].isin(set(candidates))]
    if pool.empty:
        raise ValueError("No candidate book appears in the opener archive")
    ranked = pool.sort_values(
        ["games", "rows", "bookmaker_key"], ascending=[False, False, True]
    ).reset_index(drop=True)
    return str(ranked.loc[0, "bookmaker_key"])


def single_book_series(quotes: pd.DataFrame, book: str) -> pd.DataFrame:
    selected = quotes.loc[quotes["bookmaker_key"].eq(book)]
    if selected.empty:
        raise ValueError(f"Book {book!r} has no opener quotes")
    series = selected[["game_id", "season", "week", "home_spread_line"]].rename(
        columns={"home_spread_line": "home_spread"}
    )
    series = series.drop_duplicates("game_id").copy()
    series["books"] = 1
    series["line_source"] = f"book:{book}"
    return series.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def half_point_median_series(quotes: pd.DataFrame) -> pd.DataFrame:
    half = quotes.loc[is_half_point(quotes["home_spread_line"])]
    if half.empty:
        raise ValueError("No half-point opener quotes in the archive")
    grouped = half.groupby(["game_id", "season", "week"], as_index=False).agg(
        home_spread=("home_spread_line", "median"), books=("bookmaker_key", "nunique")
    )
    grouped["line_source"] = "halfpoint_median"
    return grouped.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def series_agreement(series: pd.DataFrame, consensus: pd.DataFrame) -> dict[str, object]:
    reference = consensus[["game_id", "tue_open_home_spread"]].drop_duplicates("game_id")
    merged = series.merge(reference, on="game_id", how="inner")
    gap = (merged["home_spread"] - merged["tue_open_home_spread"]).abs()
    per_season = {
        str(season): {
            "games": len(group),
            "equals_consensus": int((group["_gap"] < HALF_POINT_TOLERANCE).sum()),
            "half_point": int(is_half_point(group["home_spread"]).sum()),
            "mean_absolute_gap": float(group["_gap"].mean()),
        }
        for season, group in merged.assign(_gap=gap).groupby("season", sort=True)
    }
    return {
        "series_games": len(series),
        "paired_games": len(merged),
        "equals_consensus": int((gap < HALF_POINT_TOLERANCE).sum()),
        "equals_consensus_share": float((gap < HALF_POINT_TOLERANCE).mean())
        if len(merged)
        else 0.0,
        "mean_absolute_gap": float(gap.mean()) if len(merged) else float("nan"),
        "max_absolute_gap": float(gap.max()) if len(merged) else float("nan"),
        "half_point_games": int(is_half_point(series["home_spread"]).sum()),
        "half_point_share": (
            float(is_half_point(series["home_spread"]).mean()) if len(series) else 0.0
        ),
        "per_season": per_season,
    }
