"""Book leadership: who moves the spread first (SKY-04 descriptive).

Frozen design in ``docs/book_leadership.md``. For each game, consecutive
quote snapshots by observed time define move events (a book's spread line
changed since its own previous snapshot); the earliest provider timestamp
among changers splits first-mover credit. Descriptive only: no ATS outcome,
no registry verdict, no window.

Writes ``artifacts/book_leadership/<stamp>/results.json`` via
``write_experiment_artifact``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.io import run_id  # noqa: E402
from nfl_ats.provenance import (  # noqa: E402
    configuration_hash,
    git_state,
    write_experiment_artifact,
)

OUT_ROOT = REPO / "artifacts" / "book_leadership"
SEASONS = (2023, 2024, 2025)


def load_spread_quotes(raw_root: Path, *, seasons: tuple[int, ...] = SEASONS) -> pd.DataFrame:
    """All spreads-market quote rows for the given seasons."""

    frames: list[pd.DataFrame] = []
    for snapshot in sorted(raw_root.glob("*/quotes.parquet")):
        try:
            frame = pd.read_parquet(
                snapshot,
                columns=[
                    "observed_at_utc",
                    "nflverse_game_id",
                    "bookmaker_key",
                    "market",
                    "home_spread_line",
                    "bookmaker_last_update_utc",
                ],
            )
        except (OSError, ValueError):
            continue
        frames.append(frame)
    if not frames:
        raise DataContractError(f"No quotes.parquet found under {raw_root}")
    quotes = pd.concat(frames, ignore_index=True)
    quotes = quotes.loc[quotes["market"].astype(str).eq("spreads")].copy()
    quotes["nflverse_game_id"] = quotes["nflverse_game_id"].astype(str)
    quotes = quotes.loc[quotes["nflverse_game_id"].str.match(r"^20(2[3-5])_").fillna(False)].copy()
    if quotes.empty:
        raise DataContractError("No spreads quotes for the requested seasons")
    quotes["observed_at_utc"] = pd.to_datetime(quotes["observed_at_utc"], utc=True)
    quotes["bookmaker_last_update_utc"] = pd.to_datetime(
        quotes["bookmaker_last_update_utc"], utc=True
    )
    quotes["home_spread_line"] = pd.to_numeric(quotes["home_spread_line"], errors="coerce")
    quotes = quotes.loc[quotes["home_spread_line"].notna()].copy()
    keep_seasons = {str(season) for season in seasons}
    quotes = quotes.loc[quotes["nflverse_game_id"].str[:4].isin(keep_seasons)].copy()
    if quotes.empty:
        raise DataContractError("No spreads quotes survived eligibility")
    return quotes.reset_index(drop=True)


def score_leadership(quotes: pd.DataFrame) -> dict[str, Any]:
    """First-mover credits and staleness profile per book."""

    required = {
        "observed_at_utc",
        "nflverse_game_id",
        "bookmaker_key",
        "home_spread_line",
        "bookmaker_last_update_utc",
    }
    missing = sorted(required.difference(quotes.columns))
    if missing:
        raise DataContractError(f"quotes are missing columns: {', '.join(missing)}")
    ordered = quotes.sort_values(["nflverse_game_id", "bookmaker_key", "observed_at_utc"])
    ordered["prev_line"] = ordered.groupby(["nflverse_game_id", "bookmaker_key"])[
        "home_spread_line"
    ].shift(1)
    moved = ordered.loc[
        ordered["prev_line"].notna() & ordered["home_spread_line"].ne(ordered["prev_line"])
    ].copy()
    participations: dict[str, int] = {}
    credits: dict[str, float] = {}
    for _, event in moved.groupby(["nflverse_game_id", "observed_at_utc"], sort=False):
        changers = event.loc[event["bookmaker_last_update_utc"].notna()]
        if changers.empty:
            continue
        earliest = changers["bookmaker_last_update_utc"].min()
        first = changers.loc[changers["bookmaker_last_update_utc"].eq(earliest)]
        share = 1.0 / len(first)
        for book in event["bookmaker_key"].astype(str).unique():
            participations[book] = participations.get(book, 0) + 1
        for book in first["bookmaker_key"].astype(str).unique():
            credits[book] = credits.get(book, 0.0) + share
    staleness: dict[str, float] = {}
    quotes["lag_seconds"] = (
        quotes["observed_at_utc"] - quotes["bookmaker_last_update_utc"]
    ).dt.total_seconds()
    for book, group in quotes.loc[quotes["lag_seconds"].ge(0)].groupby("bookmaker_key"):
        staleness[str(book)] = float(group["lag_seconds"].median())
    books = sorted(set(participations) | set(credits) | set(staleness))
    table = [
        {
            "book": book,
            "move_participations": participations.get(book, 0),
            "first_move_credits": round(credits.get(book, 0.0), 3),
            "leadership_share": (
                round(credits.get(book, 0.0) / participations[book], 4)
                if participations.get(book, 0)
                else None
            ),
            "median_staleness_seconds": staleness.get(book),
        }
        for book in books
    ]
    table.sort(key=lambda row: row["first_move_credits"] or 0.0, reverse=True)
    return {
        "games": int(quotes["nflverse_game_id"].nunique()),
        "snapshots": int(quotes["observed_at_utc"].nunique()),
        "move_events": int(
            moved.groupby(["nflverse_game_id", "observed_at_utc"], sort=False).ngroup().nunique()
        ),
        "books": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=REPO / "data" / "market" / "raw")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    started = time.time()
    quotes = load_spread_quotes(args.raw_root)
    result = score_leadership(quotes)
    configuration = {
        "command": "book-leadership",
        "seasons": list(SEASONS),
        "market": "spreads",
        "predeclaration": "docs/book_leadership.md (frozen design)",
    }
    snapshots = sorted(p.name for p in args.raw_root.glob("*/quotes.parquet"))
    payload = {
        **result,
        "elapsed_seconds": time.time() - started,
        "provenance": {
            # Directory-scanned input: artifact_provenance needs a single
            # feature-table file, so provenance here is the configuration
            # hash plus the explicit snapshot inventory below.
            "configuration": configuration,
            "configuration_sha256": configuration_hash(configuration),
            "quote_snapshots": len(snapshots),
            "quote_rows": len(quotes),
            "code": git_state(REPO),
        },
    }
    output_dir = args.output or (OUT_ROOT / run_id())
    output_dir.mkdir(parents=True, exist_ok=False)
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="book-leadership",
        metrics=payload,
        notes=(
            "Descriptive SKY-04 book-leadership measurement; first is not "
            "right, no ATS outcome, no registry verdict, no window (AGENTS.md)."
        ),
    )
    for row in payload["books"][:8]:
        print(row)
    print(f"wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
