from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import (
    DataContractError,
    decision_quote_source_records,
    load_decision_quotes,
    load_decision_quotes_with_sources,
)
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND
from nfl_ats.provenance import sha256_file

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
INPUT_MANIFEST_SCHEMA = "nfl_ats.opener_line_input_manifest"
INPUT_MANIFEST_VERSION = 1

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


def _opener_book_quotes_from_raw(
    schedule: pd.DataFrame,
    quotes: pd.DataFrame,
) -> pd.DataFrame:
    required = {"game_id", "season", "week"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise DataContractError(f"Opener line schedule is missing columns: {', '.join(missing)}")
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


def opener_book_quotes(
    root: Path,
    *,
    schedule: pd.DataFrame,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    label: str = OPENER_LABEL,
    seasons: Iterable[int] | None = None,
) -> pd.DataFrame:
    quotes = load_decision_quotes(root, capture_kind=capture_kind, labels=(label,), seasons=seasons)
    return _opener_book_quotes_from_raw(schedule, quotes)


def opener_book_quotes_with_sources(
    root: Path,
    *,
    schedule: pd.DataFrame,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    label: str = OPENER_LABEL,
    seasons: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    quotes, sources = load_decision_quotes_with_sources(
        root,
        capture_kind=capture_kind,
        labels=(label,),
        seasons=seasons,
    )
    return _opener_book_quotes_from_raw(schedule, quotes), sources


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


_MANIFEST_KEYS = {"schema", "schema_version", "feature_file", "market_selection", "line_series"}
_FEATURE_KEYS = {"path", "sha256"}
_MARKET_KEYS = {"market_root", "capture_kind", "labels", "seasons", "raw_sources"}
_SOURCE_KEYS = {
    "snapshot_id",
    "capture_kind",
    "season",
    "week",
    "decision_label",
    "manifest_path",
    "manifest_sha256",
    "quotes_path",
    "quotes_sha256",
}
_LINE_KEYS = {
    "series",
    "requested_book",
    "resolved_book",
    "book_selection",
    "artifact_path",
    "artifact_sha256",
    "rows",
    "columns",
    "line_sources",
}


def _manifest_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataContractError(f"{label} must be an object")
    if set(value) != keys:
        missing = sorted(keys.difference(value))
        extra = sorted(set(value).difference(keys))
        raise DataContractError(f"{label} keys differ: missing={missing}, extra={extra}")
    return value


def _manifest_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DataContractError(f"{label} must be a lowercase SHA-256")
    return value


def _manifest_path(value: Any, label: str, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise DataContractError(f"{label} must be a nonempty absolute path")
    path = Path(value)
    resolved = path.resolve()
    if not path.is_absolute() or str(resolved) != value:
        raise DataContractError(f"{label} must be a canonical absolute path")
    if directory:
        if not resolved.is_dir():
            raise DataContractError(f"{label} directory does not exist: {resolved}")
    elif not resolved.is_file():
        raise DataContractError(f"{label} file does not exist: {resolved}")
    return resolved


def _manifest_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DataContractError(f"{label} must be a nonempty string")
    return value


def _manifest_nullable_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _manifest_string(value, label)


def _manifest_relative_path(root: Path, value: Any, label: str) -> Path:
    text = _manifest_string(value, label)
    if "\\" in text:
        raise DataContractError(f"{label} must use POSIX separators")
    path = Path(*text.split("/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in text.split("/")):
        raise DataContractError(f"{label} must be a canonical relative path")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise DataContractError(f"{label} escapes market_root")
    return resolved


def _validate_opener_line_input_payload(
    payload: Any,
    *,
    feature_path: Path,
    market_root: Path,
    line_series_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _manifest_object(payload, _MANIFEST_KEYS, "input manifest")
    if manifest["schema"] != INPUT_MANIFEST_SCHEMA:
        raise DataContractError(f"Unexpected input manifest schema: {manifest['schema']!r}")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != INPUT_MANIFEST_VERSION
    ):
        raise DataContractError(
            f"Unexpected input manifest version: {manifest['schema_version']!r}"
        )

    feature = _manifest_object(manifest["feature_file"], _FEATURE_KEYS, "feature_file")
    recorded_feature = _manifest_path(feature["path"], "feature_file.path")
    supplied_feature = feature_path.resolve()
    if recorded_feature != supplied_feature:
        raise DataContractError(
            f"Feature path differs from manifest: {supplied_feature} != {recorded_feature}"
        )
    feature_sha = _manifest_sha(feature["sha256"], "feature_file.sha256")
    if sha256_file(recorded_feature) != feature_sha:
        raise DataContractError("Feature file SHA-256 differs from input manifest")

    market = _manifest_object(manifest["market_selection"], _MARKET_KEYS, "market_selection")
    recorded_root = _manifest_path(
        market["market_root"], "market_selection.market_root", directory=True
    )
    supplied_root = market_root.resolve()
    if recorded_root != supplied_root:
        raise DataContractError(
            f"Market root differs from manifest: {supplied_root} != {recorded_root}"
        )
    capture_kind = _manifest_string(market["capture_kind"], "market_selection.capture_kind")
    labels_value = market["labels"]
    if (
        not isinstance(labels_value, list)
        or not labels_value
        or any(not isinstance(value, str) or not value for value in labels_value)
        or len(labels_value) != len(set(labels_value))
    ):
        raise DataContractError("market_selection.labels must be a nonempty unique string list")
    labels = list(labels_value)
    seasons_value = market["seasons"]
    seasons: list[int] | None
    if seasons_value is None:
        seasons = None
    elif (
        not isinstance(seasons_value, list)
        or any(type(value) is not int for value in seasons_value)
        or seasons_value != sorted(set(seasons_value))
    ):
        raise DataContractError("market_selection.seasons must be null or sorted unique integers")
    else:
        seasons = list(seasons_value)

    sources_value = market["raw_sources"]
    if not isinstance(sources_value, list) or not sources_value:
        raise DataContractError("market_selection.raw_sources must be a nonempty list")
    sources: list[dict[str, Any]] = []
    for index, value in enumerate(sources_value):
        record = _manifest_object(value, _SOURCE_KEYS, f"raw_sources[{index}]")
        for key in ("snapshot_id", "capture_kind", "decision_label"):
            _manifest_string(record[key], f"raw_sources[{index}].{key}")
        for key in ("season", "week"):
            if type(record[key]) is not int:
                raise DataContractError(f"raw_sources[{index}].{key} must be an integer")
        _manifest_relative_path(
            recorded_root, record["manifest_path"], f"raw_sources[{index}].manifest_path"
        )
        _manifest_relative_path(
            recorded_root, record["quotes_path"], f"raw_sources[{index}].quotes_path"
        )
        _manifest_sha(record["manifest_sha256"], f"raw_sources[{index}].manifest_sha256")
        _manifest_sha(record["quotes_sha256"], f"raw_sources[{index}].quotes_sha256")
        sources.append(record)
    expected_sources = decision_quote_source_records(
        recorded_root,
        capture_kind=capture_kind,
        labels=labels,
        seasons=seasons,
    )
    if sources != expected_sources:
        raise DataContractError("Raw market source set or file hashes differ from input manifest")

    line = _manifest_object(manifest["line_series"], _LINE_KEYS, "line_series")
    series_name = _manifest_string(line["series"], "line_series.series")
    requested_book = _manifest_nullable_string(line["requested_book"], "line_series.requested_book")
    resolved_book = _manifest_nullable_string(line["resolved_book"], "line_series.resolved_book")
    book_selection = _manifest_string(line["book_selection"], "line_series.book_selection")
    recorded_series = _manifest_path(line["artifact_path"], "line_series.artifact_path")
    supplied_series = line_series_path.resolve()
    if recorded_series != supplied_series:
        raise DataContractError(
            f"Line-series path differs from manifest: {supplied_series} != {recorded_series}"
        )
    series_sha = _manifest_sha(line["artifact_sha256"], "line_series.artifact_sha256")
    before_sha = sha256_file(recorded_series)
    if before_sha != series_sha:
        raise DataContractError("Line-series SHA-256 differs from input manifest")
    frame = pd.read_parquet(recorded_series)
    if sha256_file(recorded_series) != before_sha:
        raise DataContractError("Line-series artifact changed while it was being loaded")
    if type(line["rows"]) is not int or line["rows"] < 0 or line["rows"] != len(frame):
        raise DataContractError("Line-series row count differs from input manifest")
    columns = line["columns"]
    if (
        not isinstance(columns, list)
        or any(not isinstance(value, str) for value in columns)
        or columns != list(frame.columns)
    ):
        raise DataContractError("Line-series columns differ from input manifest")
    required_columns = {"game_id", "season", "week", "home_spread", "books", "line_source"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise DataContractError(f"Line-series artifact is missing columns: {', '.join(missing)}")
    if (
        frame["line_source"].isna().any()
        or not frame["line_source"].map(lambda value: isinstance(value, str) and bool(value)).all()
    ):
        raise DataContractError("Line-series line_source values must be nonempty strings")
    actual_sources = sorted(frame["line_source"].unique().tolist())
    recorded_sources = line["line_sources"]
    if (
        not isinstance(recorded_sources, list)
        or any(not isinstance(value, str) or not value for value in recorded_sources)
        or recorded_sources != sorted(set(recorded_sources))
        or recorded_sources != actual_sources
    ):
        raise DataContractError("Line-series source values differ from input manifest")

    if series_name == "book":
        if resolved_book is None:
            raise DataContractError("Book series requires line_series.resolved_book")
        expected_selection = "explicit" if requested_book is not None else "coverage_rank"
        if requested_book is not None and requested_book != resolved_book:
            raise DataContractError("Requested and resolved book differ")
        if book_selection != expected_selection:
            raise DataContractError("Book selection mode differs from requested book")
        expected_line_sources = [f"book:{resolved_book}"]
    elif series_name == "halfpoint_median":
        if requested_book is not None or resolved_book is not None:
            raise DataContractError("Half-point median series cannot name a book")
        if book_selection != "not_applicable":
            raise DataContractError(
                "Half-point median series requires not_applicable book selection"
            )
        expected_line_sources = ["halfpoint_median"]
    else:
        raise DataContractError(f"Unsupported line-series kind: {series_name!r}")
    if actual_sources != expected_line_sources:
        raise DataContractError(
            f"Line-series source does not match recorded series/book: {actual_sources}"
        )
    return frame, manifest


def build_opener_line_input_manifest(
    *,
    feature_path: Path,
    feature_sha256: str,
    market_root: Path,
    capture_kind: str,
    labels: Iterable[str],
    seasons: Iterable[int] | None,
    raw_sources: list[dict[str, Any]],
    line_series_path: Path,
    line_series: pd.DataFrame,
    series: str,
    requested_book: str | None,
    resolved_book: str | None,
    book_selection: str,
) -> dict[str, Any]:
    resolved_feature = feature_path.resolve()
    resolved_root = market_root.resolve()
    resolved_series = line_series_path.resolve()
    normalized_labels = [str(value) for value in labels]
    normalized_seasons = sorted({int(value) for value in seasons}) if seasons is not None else None
    payload: dict[str, Any] = {
        "schema": INPUT_MANIFEST_SCHEMA,
        "schema_version": INPUT_MANIFEST_VERSION,
        "feature_file": {
            "path": str(resolved_feature),
            "sha256": feature_sha256,
        },
        "market_selection": {
            "market_root": str(resolved_root),
            "capture_kind": capture_kind,
            "labels": normalized_labels,
            "seasons": normalized_seasons,
            "raw_sources": raw_sources,
        },
        "line_series": {
            "series": series,
            "requested_book": requested_book,
            "resolved_book": resolved_book,
            "book_selection": book_selection,
            "artifact_path": str(resolved_series),
            "artifact_sha256": sha256_file(resolved_series),
            "rows": len(line_series),
            "columns": list(line_series.columns),
            "line_sources": sorted(line_series["line_source"].unique().tolist()),
        },
    }
    _validate_opener_line_input_payload(
        payload,
        feature_path=resolved_feature,
        market_root=resolved_root,
        line_series_path=resolved_series,
    )
    return payload


def validate_opener_line_input_manifest(
    *,
    manifest_path: Path,
    feature_path: Path,
    market_root: Path,
    line_series_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    resolved_manifest = manifest_path.resolve()
    if not resolved_manifest.is_file():
        raise DataContractError(f"Input manifest does not exist: {resolved_manifest}")
    before_sha = sha256_file(resolved_manifest)
    try:
        payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"Input manifest is unreadable: {resolved_manifest}") from exc
    frame, validated = _validate_opener_line_input_payload(
        payload,
        feature_path=feature_path,
        market_root=market_root,
        line_series_path=line_series_path,
    )
    if sha256_file(resolved_manifest) != before_sha:
        raise DataContractError("Input manifest changed while it was being validated")
    return frame, validated, before_sha
