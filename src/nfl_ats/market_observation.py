from __future__ import annotations

from pathlib import Path

import pandas as pd

from nfl_ats.market_data import QUOTE_COLUMNS, load_quote_history, tuesday_opener_quotes

MARKET_OBSERVED_AT_COLUMN = "market_observed_at_utc"

MARKET_OPENER_BASIS_COLUMN = "market_opener_basis"


def _empty_quote_history() -> pd.DataFrame:
    return pd.DataFrame(columns=QUOTE_COLUMNS)


def _null_column(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")


def attach_market_observed_at(
    frame: pd.DataFrame,
    *,
    market_raw_root: Path | None = None,
    quote_history: pd.DataFrame | None = None,
) -> pd.DataFrame:

    result = frame.copy()
    if "game_id" not in result.columns:
        result[MARKET_OBSERVED_AT_COLUMN] = _null_column(result.index)
        result[MARKET_OPENER_BASIS_COLUMN] = pd.Series(pd.NA, index=result.index, dtype="string")
        return result

    if quote_history is None:
        quote_history = (
            load_quote_history(market_raw_root)
            if market_raw_root is not None
            else _empty_quote_history()
        )
    if quote_history.empty:
        result[MARKET_OBSERVED_AT_COLUMN] = _null_column(result.index)
        result[MARKET_OPENER_BASIS_COLUMN] = pd.Series(pd.NA, index=result.index, dtype="string")
        return result

    opener = (
        tuesday_opener_quotes(quote_history)
        .dropna(subset=["nflverse_game_id"])
        .drop_duplicates("nflverse_game_id", keep="last")
        .set_index("nflverse_game_id")
    )
    game_ids = result["game_id"].astype(str)
    observed = game_ids.map(opener["observed_at_utc"])
    result[MARKET_OBSERVED_AT_COLUMN] = pd.to_datetime(observed, utc=True, errors="coerce")
    result[MARKET_OPENER_BASIS_COLUMN] = game_ids.map(opener["opener_basis"]).astype("string")
    return result


__all__ = ["MARKET_OBSERVED_AT_COLUMN", "MARKET_OPENER_BASIS_COLUMN", "attach_market_observed_at"]
