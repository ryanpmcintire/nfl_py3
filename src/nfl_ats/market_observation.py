"""Join the point-in-time odds capture's observation instant onto a forecast
frame (ENG-23, closing ``docs/feature_lineage.md`` gap item 2).

The card's market line (``spread_line``) arrives from the nflverse schedule
table, which carries no observation timestamp of its own -- the freshest
defensible bound was, until now, the whole feature table's ``built_at_utc``.
Separately, this project's own point-in-time odds capture
(``nfl-ats odds-ingest`` / ``odds-backfill``, written under
``data/market/raw/<stamp>/``, read by :mod:`nfl_ats.market_data`) DOES carry a
real ``observed_at_utc`` per quote, but was never joined onto the forecast.

Both ``AGENTS.md`` ("Grade the decision at the OPENER") and
:data:`nfl_ats.source_freshness_policy.SOURCE_FRESHNESS_POLICIES`
(``"odds_opener"``: *"The Odds API Tuesday opener (the grade the pool settles
on)"*) already treat the Tuesday-opener capture as the line this project
grades against. This module joins THAT capture's ``observed_at_utc`` onto the
frame by ``game_id``, using :func:`nfl_ats.market_data.tuesday_opener_quotes`
unchanged -- it does not read, choose, or alter ``spread_line`` or any other
column, only attaches when the market was last observed for the OPENER quote
already treated as authoritative elsewhere in this codebase.

Most historical rows predate the-odds-api ingestion (which only began
capturing partway through this project's history) and are left with a null
``market_observed_at_utc`` -- this never raises; a missing capture is an
unanswerable "when", not an error.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nfl_ats.market_data import QUOTE_COLUMNS, load_quote_history, tuesday_opener_quotes

#: Column name :mod:`nfl_ats.prediction_safety` already looks for first (see
#: ``_prospective_checks``'s ``market_observed_at_utc`` / ``line_observed_at_utc``
#: / ``observed_at_utc`` fallback chain) and :mod:`nfl_ats.lineage` now prefers
#: for the ``market_line`` record when present.
MARKET_OBSERVED_AT_COLUMN = "market_observed_at_utc"


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
    """Return a copy of ``frame`` with :data:`MARKET_OBSERVED_AT_COLUMN` attached.

    ``quote_history`` -- an already-loaded :func:`nfl_ats.market_data.load_quote_history`
    result -- wins over ``market_raw_root`` when both are supplied, so a
    caller (or a test) that already holds the quotes in memory never re-reads
    the raw capture tree. Neither argument is required: with both absent,
    every row gets a null timestamp rather than a crash, matching every other
    source in this project that degrades to "unobserved" instead of failing
    closed here -- the release-blocking check lives in
    :mod:`nfl_ats.prediction_safety`, not here.

    The join key is ``frame["game_id"]`` against
    :func:`nfl_ats.market_data.tuesday_opener_quotes`'s ``nflverse_game_id``;
    the attached value is that function's own ``observed_at_utc`` (the
    earliest Tuesday-captured quote across books). ``spread_line`` -- or any
    other column -- is never read or modified, so this can never change which
    line the card plays.
    """

    result = frame.copy()
    if "game_id" not in result.columns:
        result[MARKET_OBSERVED_AT_COLUMN] = _null_column(result.index)
        return result

    if quote_history is None:
        quote_history = (
            load_quote_history(market_raw_root)
            if market_raw_root is not None
            else _empty_quote_history()
        )
    if quote_history.empty:
        result[MARKET_OBSERVED_AT_COLUMN] = _null_column(result.index)
        return result

    opener = tuesday_opener_quotes(quote_history)
    lookup = (
        opener.dropna(subset=["nflverse_game_id"])
        .drop_duplicates("nflverse_game_id", keep="last")
        .set_index("nflverse_game_id")["observed_at_utc"]
    )
    observed = result["game_id"].astype(str).map(lookup)
    result[MARKET_OBSERVED_AT_COLUMN] = pd.to_datetime(observed, utc=True, errors="coerce")
    return result


__all__ = ["MARKET_OBSERVED_AT_COLUMN", "attach_market_observed_at"]
