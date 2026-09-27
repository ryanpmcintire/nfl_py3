from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError

PriceMethod = Literal["fanduel", "consensus"]


@dataclass(frozen=True)
class PairedPriceExtraction:
    prices: pd.DataFrame
    book_prices: pd.DataFrame
    exclusions: pd.DataFrame
    provenance: dict[str, Any]


def _require_unique_columns(frame: pd.DataFrame, name: str) -> None:
    duplicates = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicates:
        raise DataContractError(f"{name} has duplicate columns: {sorted(set(duplicates))}")


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise DataContractError(f"{name} missing required columns: {missing}")


def _numeric(series: pd.Series, name: str, *, nullable: bool = False) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    malformed = series.notna() & converted.isna()
    if bool(malformed.any()):
        raise DataContractError(f"{name} contains nonnumeric values")
    finite = converted.notna() & ~np.isfinite(converted.astype(float))
    if bool(finite.any()):
        raise DataContractError(f"{name} contains nonfinite values")
    if not nullable and bool(converted.isna().any()):
        raise DataContractError(f"{name} contains missing values")
    return converted


def _datetime(series: pd.Series, name: str) -> pd.Series:
    converted = pd.to_datetime(series, utc=True, errors="coerce")
    malformed = series.notna() & converted.isna()
    if bool(malformed.any()):
        raise DataContractError(f"{name} contains invalid timestamps")
    aware = series.map(_timestamp_is_aware)
    if bool((series.notna() & ~aware).any()):
        raise DataContractError(f"{name} contains timezone-naive timestamps")
    return converted


def _timestamp_is_aware(value: Any) -> bool:
    if pd.isna(value):
        return True
    parsed = pd.to_datetime(value, errors="coerce")
    return not pd.isna(parsed) and getattr(parsed, "tzinfo", None) is not None


def _normalize_lines(lines: pd.DataFrame) -> pd.DataFrame:
    _require_unique_columns(lines, "lines")
    required = {"game_id", "season", "week", "home_spread"}
    _require(lines, required, "lines")
    out = lines.loc[:, sorted(required)].copy()
    out["game_id"] = out["game_id"].astype("string").str.strip()
    if bool(out["game_id"].isna().any()) or bool(out["game_id"].eq("").any()):
        raise DataContractError("lines game_id contains missing values")
    if bool(out["game_id"].duplicated().any()):
        duplicate_values = sorted(
            out.loc[out["game_id"].duplicated(False), "game_id"].astype(str).unique()
        )
        raise DataContractError(f"lines has duplicate game_id values: {duplicate_values[:5]}")
    for column in ("season", "week"):
        values = _numeric(out[column], f"lines {column}")
        if bool((values != np.floor(values)).any()):
            raise DataContractError(f"lines {column} contains nonintegral values")
        out[column] = values.astype(int)
    out["home_spread"] = _numeric(out["home_spread"], "lines home_spread").astype(float)
    return out.sort_values(["season", "week", "game_id"], kind="stable").reset_index(drop=True)


def _normalize_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    _require_unique_columns(quotes, "quotes")
    required = {
        "nflverse_game_id",
        "season",
        "week",
        "capture_kind",
        "market",
        "outcome_side",
        "line",
        "price",
        "home_spread_line",
        "bookmaker_key",
        "provider_event_id",
        "observed_at_utc",
        "commence_time_utc",
        "snapshot_timestamp_utc",
    }
    _require(quotes, required, "quotes")
    out = quotes.copy()
    for column in ("nflverse_game_id", "bookmaker_key", "provider_event_id"):
        out[column] = out[column].astype("string").str.strip()
    out["bookmaker_key"] = out["bookmaker_key"].str.lower()
    for column in ("capture_kind", "market", "outcome_side"):
        out[column] = out[column].astype("string").str.strip().str.lower()
    for column in ("observed_at_utc", "commence_time_utc", "snapshot_timestamp_utc"):
        out[column] = _datetime(out[column], f"quotes {column}")
    for column in ("season", "week"):
        values = _numeric(out[column], f"quotes {column}")
        if bool((values != np.floor(values)).any()):
            raise DataContractError(f"quotes {column} contains nonintegral values")
        out[column] = values.astype(int)
    out["line"] = _numeric(out["line"], "quotes line", nullable=True)
    out["price"] = _numeric(out["price"], "quotes price", nullable=True)
    out["home_spread_line"] = _numeric(
        out["home_spread_line"], "quotes home_spread_line", nullable=True
    )
    return out


def _same_line(values: pd.Series, target: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return pd.Series(
        np.isclose(numeric, target, rtol=0.0, atol=1e-9, equal_nan=False),
        index=values.index,
    )


def _valid_price(value: Any) -> bool:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(price) and abs(price) >= 100.0


def _implied_probability(price: float) -> float:
    if price < 0:
        return -price / (-price + 100.0)
    return 100.0 / (price + 100.0)


def _no_vig_home(home_price: float, away_price: float) -> float:
    home = _implied_probability(home_price)
    away = _implied_probability(away_price)
    return home / (home + away)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or bool(pd.isna(value)):
        return None
    return value


def _frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.copy()
    ordered.columns = ordered.columns.astype(str)
    ordered = ordered.reindex(sorted(ordered.columns), axis=1)
    records = [
        {column: _canonical_value(value) for column, value in row.items()}
        for row in ordered.to_dict(orient="records")
    ]
    rows = sorted(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for record in records
    )
    payload = json.dumps(rows, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _latest_home_quotes(quotes: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    candidates = quotes.loc[
        quotes["market"].eq("spreads")
        & quotes["outcome_side"].eq("home")
        & quotes["nflverse_game_id"].notna()
        & quotes["bookmaker_key"].notna()
        & quotes["observed_at_utc"].notna()
        & quotes["commence_time_utc"].notna()
        & (quotes["observed_at_utc"] < quotes["commence_time_utc"])
        & quotes["home_spread_line"].notna()
    ].copy()
    selected: list[pd.DataFrame] = []
    exclusions: list[dict[str, Any]] = []
    for (game_id, book), group in candidates.groupby(
        ["nflverse_game_id", "bookmaker_key"], sort=True, dropna=False
    ):
        observed = group["observed_at_utc"].max()
        finalists = group.loc[group["observed_at_utc"].eq(observed)]
        snapshots = finalists["snapshot_timestamp_utc"].dropna()
        if not snapshots.empty:
            finalists = finalists.loc[finalists["snapshot_timestamp_utc"].eq(snapshots.max())]
        if len(finalists) != 1:
            exclusions.append(
                {
                    "game_id": str(game_id),
                    "bookmaker_key": str(book),
                    "scope": "book",
                    "reason": "ambiguous_latest_home_quote",
                }
            )
            continue
        selected.append(finalists)
    if not selected:
        return candidates.iloc[0:0].copy(), exclusions
    return pd.concat(selected, ignore_index=True), exclusions


def _pair_groups(quotes: pd.DataFrame) -> dict[tuple[str, str, Any], pd.DataFrame]:
    paired = quotes.loc[
        quotes["market"].eq("spreads")
        & quotes["nflverse_game_id"].notna()
        & quotes["bookmaker_key"].notna()
        & quotes["snapshot_timestamp_utc"].notna()
    ]
    return {
        (str(game_id), str(book), snapshot): group
        for (game_id, book, snapshot), group in paired.groupby(
            ["nflverse_game_id", "bookmaker_key", "snapshot_timestamp_utc"],
            sort=False,
            dropna=False,
        )
    }


def _exact_pair(
    home_quote: pd.Series,
    selected_line: float,
    groups: Mapping[tuple[str, str, Any], pd.DataFrame],
) -> tuple[dict[str, Any] | None, str | None]:
    game_id = str(home_quote["nflverse_game_id"])
    book = str(home_quote["bookmaker_key"])
    if not book:
        return None, "missing_bookmaker_key"
    snapshot = home_quote["snapshot_timestamp_utc"]
    if pd.isna(snapshot):
        return None, "missing_snapshot"
    group = groups.get((game_id, book, snapshot))
    if group is None:
        return None, "missing_snapshot_pair"
    provider_event_id = home_quote["provider_event_id"]
    if pd.isna(provider_event_id) or not str(provider_event_id).strip():
        return None, "missing_provider_event_id"
    event = group.loc[group["provider_event_id"].eq(provider_event_id)]
    exact_home = event.loc[
        event["outcome_side"].eq("home")
        & _same_line(event["line"], -selected_line)
        & _same_line(event["home_spread_line"], selected_line)
    ]
    exact_away = event.loc[
        event["outcome_side"].eq("away")
        & _same_line(event["line"], selected_line)
        & _same_line(event["home_spread_line"], selected_line)
    ]
    if exact_home.empty or exact_away.empty:
        return None, "missing_exact_pair"
    if len(exact_home) != 1 or len(exact_away) != 1:
        return None, "ambiguous_pair_identity"
    home_row = exact_home.iloc[0]
    away_row = exact_away.iloc[0]
    timestamp_values = (
        home_row["observed_at_utc"],
        away_row["observed_at_utc"],
        home_row["commence_time_utc"],
        away_row["commence_time_utc"],
        home_row["snapshot_timestamp_utc"],
        away_row["snapshot_timestamp_utc"],
    )
    if any(pd.isna(value) for value in timestamp_values):
        return None, "missing_pair_timestamp"
    if not all(_timestamp_is_aware(value) for value in timestamp_values):
        return None, "timezone_naive_pair_timestamp"
    home_kickoff = home_row["commence_time_utc"]
    away_kickoff = away_row["commence_time_utc"]
    if home_kickoff != away_kickoff:
        return None, "pair_kickoff_mismatch"
    if (
        home_row["snapshot_timestamp_utc"] != away_row["snapshot_timestamp_utc"]
        or home_row["snapshot_timestamp_utc"] != snapshot
    ):
        return None, "pair_snapshot_mismatch"
    if (
        home_row["observed_at_utc"] >= home_kickoff
        or away_row["observed_at_utc"] >= home_kickoff
        or snapshot >= home_kickoff
    ):
        return None, "pair_not_pregame"
    home_price = home_row["price"]
    away_price = away_row["price"]
    if not _valid_price(home_price) or not _valid_price(away_price):
        return None, "invalid_american_price"
    home_value = float(home_price)
    away_value = float(away_price)
    return (
        {
            "game_id": game_id,
            "bookmaker_key": book,
            "snapshot_timestamp_utc": snapshot,
            "observed_at_utc": home_row["observed_at_utc"],
            "provider_event_id": None if pd.isna(provider_event_id) else str(provider_event_id),
            "home_price": home_value,
            "away_price": away_value,
            "home_probability": _no_vig_home(home_value, away_value),
        },
        None,
    )


def _empty_prices() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "game_id",
            "season",
            "week",
            "line",
            "market_probability",
            "book_count",
            "method",
        ]
    )


def _empty_book_prices() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "game_id",
            "season",
            "week",
            "line",
            "bookmaker_key",
            "snapshot_timestamp_utc",
            "observed_at_utc",
            "provider_event_id",
            "home_price",
            "away_price",
            "home_probability",
        ]
    )


def _empty_exclusions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "game_id",
            "season",
            "week",
            "line",
            "bookmaker_key",
            "scope",
            "reason",
        ]
    )


def extract_paired_opener_prices(
    lines: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    method: PriceMethod,
    bookmaker_key: str | None = None,
    line_provenance: Mapping[str, Any] | None = None,
    quote_provenance: Mapping[str, Any] | None = None,
) -> PairedPriceExtraction:
    if method not in ("fanduel", "consensus"):
        raise DataContractError(f"unsupported paired price method: {method}")
    normalized_book = None
    if bookmaker_key is not None:
        normalized_book = bookmaker_key.strip().lower()
        if not normalized_book:
            raise DataContractError("bookmaker_key is empty")
    if method == "fanduel" and normalized_book != "fanduel":
        raise DataContractError("fanduel method requires bookmaker_key='fanduel'")
    if method == "consensus" and normalized_book is not None:
        raise DataContractError("consensus method does not accept bookmaker_key")

    normalized_lines = _normalize_lines(lines)
    normalized_quotes = _normalize_quotes(quotes)
    if (
        normalized_quotes["capture_kind"].isna().any()
        or (normalized_quotes["capture_kind"] == "").any()
    ):
        raise DataContractError("quotes contain missing capture_kind values")
    capture_kinds = sorted(normalized_quotes["capture_kind"].unique().tolist())
    if len(capture_kinds) != 1:
        raise DataContractError("quotes must contain exactly one capture_kind")

    latest, ambiguous = _latest_home_quotes(normalized_quotes)
    groups = _pair_groups(normalized_quotes)
    ambiguous_by_game: dict[str, list[dict[str, Any]]] = {}
    for item in ambiguous:
        ambiguous_by_game.setdefault(str(item["game_id"]), []).append(item)

    price_rows: list[dict[str, Any]] = []
    book_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    for line_row in normalized_lines.itertuples(index=False):
        game_id = str(line_row.game_id)
        season = int(cast(Any, line_row.season))
        week = int(cast(Any, line_row.week))
        selected_line = float(cast(Any, line_row.home_spread))
        game_latest = latest.loc[latest["nflverse_game_id"].eq(game_id)]
        game_latest = game_latest.loc[_same_line(game_latest["home_spread_line"], selected_line)]
        game_ambiguous = ambiguous_by_game.get(game_id, [])
        if method == "fanduel":
            requested_book = cast(str, normalized_book)
            game_latest = game_latest.loc[game_latest["bookmaker_key"].eq(requested_book)]
            game_ambiguous = [
                item for item in game_ambiguous if item["bookmaker_key"] == requested_book
            ]
        for item in game_ambiguous:
            exclusion_rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "line": selected_line,
                    "bookmaker_key": item["bookmaker_key"],
                    "scope": "book",
                    "reason": item["reason"],
                }
            )

        valid_rows: list[dict[str, Any]] = []
        for _, home_quote in game_latest.iterrows():
            book = str(home_quote["bookmaker_key"])
            pair: dict[str, Any] | None
            reason: str | None
            if (
                int(cast(Any, home_quote["season"])) != season
                or int(cast(Any, home_quote["week"])) != week
            ):
                reason = "season_week_mismatch"
                pair = None
            else:
                pair, reason = _exact_pair(home_quote, selected_line, groups)
                if pair is not None:
                    snapshot = pair["snapshot_timestamp_utc"]
                    group = groups[(game_id, book, snapshot)]
                    event_id = pair["provider_event_id"]
                    if event_id is None:
                        event = group.loc[
                            group["provider_event_id"].isna() | group["provider_event_id"].eq("")
                        ]
                    else:
                        event = group.loc[group["provider_event_id"].eq(event_id)]
                    exact = event.loc[
                        _same_line(event["home_spread_line"], selected_line)
                        & (
                            (
                                event["outcome_side"].eq("home")
                                & _same_line(event["line"], -selected_line)
                            )
                            | (
                                event["outcome_side"].eq("away")
                                & _same_line(event["line"], selected_line)
                            )
                        )
                    ]
                    if bool(exact["season"].ne(season).any() or exact["week"].ne(week).any()):
                        pair = None
                        reason = "season_week_mismatch"
            if pair is None:
                exclusion_rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "line": selected_line,
                        "bookmaker_key": book,
                        "scope": "book",
                        "reason": reason,
                    }
                )
                continue
            row = {
                "game_id": game_id,
                "season": season,
                "week": week,
                "line": selected_line,
                **{key: value for key, value in pair.items() if key != "game_id"},
            }
            valid_rows.append(row)
            book_rows.append(row)

        if not valid_rows:
            reason = (
                "missing_book_at_selected_line"
                if method == "fanduel"
                else "no_valid_book_at_selected_line"
            )
            exclusion_rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "line": selected_line,
                    "bookmaker_key": normalized_book,
                    "scope": "game",
                    "reason": reason,
                }
            )
            continue
        probabilities = np.asarray(
            [float(item["home_probability"]) for item in valid_rows], dtype=float
        )
        price_rows.append(
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "line": selected_line,
                "market_probability": float(np.median(probabilities)),
                "book_count": len(valid_rows),
                "method": method,
            }
        )

    prices = pd.DataFrame(price_rows) if price_rows else _empty_prices()
    books = pd.DataFrame(book_rows) if book_rows else _empty_book_prices()
    exclusions = pd.DataFrame(exclusion_rows) if exclusion_rows else _empty_exclusions()
    prices = prices.sort_values(["season", "week", "game_id"], kind="stable").reset_index(drop=True)
    books = books.sort_values(
        ["season", "week", "game_id", "bookmaker_key"], kind="stable"
    ).reset_index(drop=True)
    exclusions = exclusions.sort_values(
        ["season", "week", "game_id", "scope", "bookmaker_key", "reason"],
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)
    provenance = {
        "schema": "paired_opener_prices.v1",
        "method": method,
        "bookmaker_key": normalized_book,
        "capture_kind": capture_kinds[0],
        "input_line_count": len(normalized_lines),
        "input_quote_count": len(normalized_quotes),
        "price_count": len(prices),
        "book_price_count": len(books),
        "exclusion_count": len(exclusions),
        "selected_bookmakers": (
            [normalized_book]
            if normalized_book is not None
            else sorted(books["bookmaker_key"].unique().tolist())
        ),
        "input_lines_sha256": _frame_hash(normalized_lines),
        "input_quotes_sha256": _frame_hash(normalized_quotes),
        "prices_sha256": _frame_hash(prices),
        "book_prices_sha256": _frame_hash(books),
        "exclusions_sha256": _frame_hash(exclusions),
        "line_provenance": dict(line_provenance or {}),
        "quote_provenance": dict(quote_provenance or {}),
    }
    return PairedPriceExtraction(
        prices=prices,
        book_prices=books,
        exclusions=exclusions,
        provenance=provenance,
    )
