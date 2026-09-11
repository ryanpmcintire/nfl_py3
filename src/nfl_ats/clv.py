from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import binomtest
from sklearn.pipeline import Pipeline

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.calibration import ResidualSmoothingMethod
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.home_side_location import (
    HOME_SIDE_OFFSET_POLICY,
    HOME_SIDE_OFFSET_SERVED,
    PRIOR_WEIGHT_GAMES,
    TRAILING_SEASONS,
    fit_home_side_offsets,
    prior_rows_before,
)
from nfl_ats.io import atomic_parquet
from nfl_ats.margin import (
    MarginFeatureProfile,
    fit_margin_model,
    make_margin_estimator,
    margin_feature_columns,
)
from nfl_ats.market_data import own_week_tuesday_quotes, tuesday_opener_quotes
from nfl_ats.modeling import regular_season_rows
from nfl_ats.odds_backfill import DECISION_LABELS, HISTORICAL_CAPTURE_KIND
from nfl_ats.provenance import sha256_file
from nfl_ats.recorder_override import replace_week_rows
from nfl_ats.snapshots import latest_snapshot

LIVE_CAPTURE_KIND = "live"
BootstrapBlock = Literal["week", "season"]

CACHE_DISABLED_ENV = "NFL_ATS_DISABLE_EVAL_CACHE"
_PAIRING_CACHE_VERSION = "1"
_OPENER_EVAL_CACHE_VERSION = "1"


def evaluation_cache_root() -> Path | None:
    if os.environ.get(CACHE_DISABLED_ENV, "").strip():
        return None
    return Path(os.environ.get("NFL_ATS_ARTIFACTS_DIR", "artifacts")) / "cache"


def _digest_text(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


_ESTIMATOR_SOURCE_MODULES = (
    "clv.py",
    "margin.py",
    "modeling.py",
    "home_side_location.py",
    "calibration.py",
)


def _estimator_source_digest() -> str:
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _ESTIMATOR_SOURCE_MODULES:
        path = here / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes() if path.is_file() else b"missing")
    return digest.hexdigest()


def market_archive_inventory_digest(root: Path) -> str:
    if not root.is_dir():
        return _digest_text("absent", str(root))
    rows: list[str] = []
    with os.scandir(root) as entries:
        snapshots = sorted((entry for entry in entries if entry.is_dir()), key=lambda e: e.name)
    for snapshot in snapshots:
        parts = [snapshot.name]
        try:
            with os.scandir(snapshot.path) as files:
                stats = {
                    item.name: item.stat()
                    for item in files
                    if item.name in ("manifest.json", "quotes.parquet")
                }
        except OSError:
            continue
        if "manifest.json" not in stats:
            continue
        for name in ("manifest.json", "quotes.parquet"):
            info = stats.get(name)
            parts.append("-" if info is None else f"{name}:{info.st_size}:{info.st_mtime_ns}")
        rows.append("|".join(parts))
    return _digest_text(str(root), *rows)


def _frame_content_digest(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    digest = hashlib.sha256(hashed.tobytes())
    digest.update(",".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(",".join(str(dtype) for dtype in frame.dtypes).encode("utf-8"))
    return digest.hexdigest()


def _read_cached_frame(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return None


def _write_cached_frame(frame: pd.DataFrame, path: Path) -> None:
    try:
        atomic_parquet(frame, path)
    except (OSError, ValueError):
        return


DECISION_LABEL_ORDER: dict[str, int] = {label: index for index, label in enumerate(DECISION_LABELS)}

CLOSE_LABEL_PRIORITY: tuple[str, ...] = ("sun_late_close", "sun_early_close")

KEY_NUMBERS: tuple[float, ...] = (3.0, 7.0)

_ACTIVE_MODEL_FALLBACK_CONFIG: dict[str, Any] = {
    "feature_profile": "player",
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "target": "market_residual",
}

FROZEN_PILOT_FEATURES: tuple[str, ...] = (
    "tue_open_home_spread",
    "tue_open_key_number_distance",
    "active_model_residual_at_opener",
    "week",
    "rest_diff",
)
FROZEN_PILOT_RIDGE_ALPHA = 10.0


class PilotProtocolBlocked(ValueError):
    pass


@dataclass(frozen=True)
class PilotSplit:
    train_start_season: int
    train_end_season: int
    validate_season: int
    test_season: int


FROZEN_PILOT_PROTOCOL = PilotSplit(
    train_start_season=2020, train_end_season=2023, validate_season=2024, test_season=2025
)


def load_snapshot_manifest_index(root: Path) -> pd.DataFrame:

    columns = [
        "dir",
        "snapshot_id",
        "capture_kind",
        "season",
        "week",
        "decision_label",
        "snapshot_timestamp_utc",
        "requested_at_utc",
    ]
    if not root.is_dir():
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        request = manifest.get("request") or {}
        rows.append(
            {
                "dir": str(entry),
                "snapshot_id": manifest.get("snapshot_id", entry.name),
                "capture_kind": manifest.get("capture_kind", LIVE_CAPTURE_KIND),
                "season": request.get("season"),
                "week": request.get("week"),
                "decision_label": request.get("decision_label"),
                "snapshot_timestamp_utc": (
                    manifest.get("snapshot_timestamp_utc") or manifest.get("observed_at_utc")
                ),
                "requested_at_utc": manifest.get("requested_at_utc"),
            }
        )
    index = pd.DataFrame(rows, columns=columns)
    if index.empty:
        return index
    index["season"] = pd.to_numeric(index["season"], errors="coerce").astype("Int64")
    index["week"] = pd.to_numeric(index["week"], errors="coerce").astype("Int64")
    index["snapshot_timestamp_utc"] = pd.to_datetime(
        index["snapshot_timestamp_utc"], utc=True, format="ISO8601"
    )
    index["requested_at_utc"] = pd.to_datetime(
        index["requested_at_utc"], utc=True, format="ISO8601"
    )
    return index.sort_values("snapshot_timestamp_utc", kind="stable").reset_index(drop=True)


def load_decision_quotes(
    root: Path,
    *,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    labels: Iterable[str] | None = None,
    seasons: Iterable[int] | None = None,
) -> pd.DataFrame:

    tag_columns = [
        "capture_kind",
        "season",
        "week",
        "decision_label",
        "snapshot_timestamp_utc",
    ]
    index = load_snapshot_manifest_index(root)
    selected = index.loc[index["capture_kind"].eq(capture_kind)]
    if labels is not None:
        selected = selected.loc[selected["decision_label"].isin(set(labels))]
    if seasons is not None:
        selected = selected.loc[selected["season"].isin(set(seasons))]
    if selected.empty:
        return pd.DataFrame(columns=[*tag_columns])
    frames: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        quotes_path = Path(str(row.dir)) / "quotes.parquet"
        if not quotes_path.is_file():
            continue
        quotes = pd.read_parquet(quotes_path)
        if "capture_kind" not in quotes.columns:
            quotes["capture_kind"] = row.capture_kind
        else:
            quotes["capture_kind"] = quotes["capture_kind"].fillna(row.capture_kind)
        quotes["season"] = row.season
        quotes["week"] = row.week
        quotes["decision_label"] = row.decision_label
        quotes["snapshot_timestamp_utc"] = row.snapshot_timestamp_utc
        frames.append(quotes)
    if not frames:
        return pd.DataFrame(columns=[*tag_columns])
    combined = pd.concat(frames, ignore_index=True)
    combined["commence_time_utc"] = pd.to_datetime(combined["commence_time_utc"], utc=True)
    combined["observed_at_utc"] = pd.to_datetime(combined["observed_at_utc"], utc=True)
    return combined


_CONSENSUS_REQUIRED_COLUMNS = (
    "nflverse_game_id",
    "season",
    "week",
    "decision_label",
    "capture_kind",
    "market",
    "outcome_side",
    "line",
    "price",
    "home_spread_line",
    "bookmaker_key",
    "observed_at_utc",
    "commence_time_utc",
    "snapshot_timestamp_utc",
)


def decision_market_consensus(quotes: pd.DataFrame) -> pd.DataFrame:

    missing = sorted(set(_CONSENSUS_REQUIRED_COLUMNS).difference(quotes.columns))
    if missing:
        raise DataContractError(f"Decision quotes are missing columns: {', '.join(missing)}")
    if quotes.empty:
        return pd.DataFrame(
            columns=[
                "nflverse_game_id",
                "season",
                "week",
                "decision_label",
                "market",
                "outcome_side",
                "books",
                "consensus_line",
                "line_min",
                "line_max",
                "line_std",
                "consensus_price",
                "snapshot_timestamp_utc",
            ]
        )
    working = quotes.copy()
    working = working.loc[working["nflverse_game_id"].notna()]
    working = working.loc[working["observed_at_utc"].lt(working["commence_time_utc"])]
    if working.empty:
        return decision_market_consensus(quotes.iloc[0:0])
    working["line_value"] = np.where(
        working["market"].eq("spreads"),
        working["home_spread_line"],
        working["line"],
    )
    deduped = (
        working.sort_values("observed_at_utc")
        .groupby(
            ["nflverse_game_id", "decision_label", "market", "outcome_side", "bookmaker_key"],
            as_index=False,
            dropna=False,
        )
        .tail(1)
    )
    grouped = deduped.groupby(
        ["nflverse_game_id", "season", "week", "decision_label", "market", "outcome_side"],
        as_index=False,
        dropna=False,
    ).agg(
        books=("bookmaker_key", "nunique"),
        consensus_line=("line_value", "median"),
        line_min=("line_value", "min"),
        line_max=("line_value", "max"),
        line_std=("line_value", "std"),
        consensus_price=("price", "median"),
        snapshot_timestamp_utc=("snapshot_timestamp_utc", "max"),
    )
    grouped = (
        grouped.sort_values(["season", "week"])
        .groupby(
            ["nflverse_game_id", "decision_label", "market", "outcome_side"],
            as_index=False,
            dropna=False,
        )
        .tail(1)
    )
    return grouped


def assert_monotone_decision_timeline(consensus: pd.DataFrame) -> None:

    if consensus.empty:
        return
    unknown = sorted(set(consensus["decision_label"].dropna()) - set(DECISION_LABEL_ORDER))
    if unknown:
        raise DataContractError(f"Unknown decision labels in consensus table: {', '.join(unknown)}")
    working = consensus[["nflverse_game_id", "decision_label", "snapshot_timestamp_utc"]].copy()
    working["_order"] = working["decision_label"].map(DECISION_LABEL_ORDER)
    bad_games: list[str] = []
    for game_id, group in working.groupby("nflverse_game_id", dropna=True):
        ordered = group.sort_values("_order")
        timestamps = ordered["snapshot_timestamp_utc"].to_numpy()
        if len(timestamps) > 1 and not pd.Series(timestamps).is_monotonic_increasing:
            bad_games.append(str(game_id))
    if bad_games:
        raise DataContractError(
            f"Non-monotone decision-label timeline for games: {', '.join(bad_games[:5])}"
        )


def build_pairing_table(
    root: Path,
    *,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    labels: Iterable[str] | None = None,
    seasons: Iterable[int] | None = None,
    schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:

    quotes = load_decision_quotes(root, capture_kind=capture_kind, labels=labels, seasons=seasons)
    if schedule is not None and not quotes.empty:
        required = {"game_id", "season", "week"}
        missing = sorted(required.difference(schedule.columns))
        if missing:
            raise DataContractError(f"Pairing schedule is missing columns: {', '.join(missing)}")
        true_week = (
            schedule[["game_id", "season", "week"]]
            .drop_duplicates("game_id")
            .rename(
                columns={
                    "game_id": "nflverse_game_id",
                    "season": "_true_season",
                    "week": "_true_week",
                }
            )
        )
        quotes = quotes.merge(true_week, on="nflverse_game_id", how="left")
        quotes = quotes.loc[
            quotes["season"].eq(quotes["_true_season"]) & quotes["week"].eq(quotes["_true_week"])
        ].drop(columns=["_true_season", "_true_week"])
    empty_columns = [
        "game_id",
        "season",
        "week",
        "decision_label",
        "capture_kind",
        "snapshot_timestamp_utc",
        "home_spread",
        "spread_books",
        "spread_min",
        "spread_max",
        "spread_std",
        "total_line",
        "total_books",
        "home_moneyline",
        "away_moneyline",
        "moneyline_books",
    ]
    if quotes.empty:
        return pd.DataFrame(columns=empty_columns)
    consensus = decision_market_consensus(quotes)
    assert_monotone_decision_timeline(consensus)

    spreads = consensus.loc[
        consensus["market"].eq("spreads") & consensus["outcome_side"].eq("HOME")
    ][
        [
            "nflverse_game_id",
            "season",
            "week",
            "decision_label",
            "books",
            "consensus_line",
            "line_min",
            "line_max",
            "line_std",
            "snapshot_timestamp_utc",
        ]
    ].rename(
        columns={
            "books": "spread_books",
            "consensus_line": "home_spread",
            "line_min": "spread_min",
            "line_max": "spread_max",
            "line_std": "spread_std",
        }
    )
    keys = ["nflverse_game_id", "season", "week", "decision_label"]
    totals = consensus.loc[consensus["market"].eq("totals") & consensus["outcome_side"].eq("OVER")][
        [*keys, "books", "consensus_line"]
    ].rename(columns={"books": "total_books", "consensus_line": "total_line"})
    home_ml = consensus.loc[consensus["market"].eq("h2h") & consensus["outcome_side"].eq("HOME")][
        [*keys, "books", "consensus_price"]
    ].rename(columns={"books": "moneyline_books", "consensus_price": "home_moneyline"})
    away_ml = consensus.loc[consensus["market"].eq("h2h") & consensus["outcome_side"].eq("AWAY")][
        [*keys, "consensus_price"]
    ].rename(columns={"consensus_price": "away_moneyline"})

    pairing = spreads.merge(totals, on=keys, how="left")
    pairing = pairing.merge(home_ml, on=keys, how="left")
    pairing = pairing.merge(away_ml, on=keys, how="left")
    pairing = pairing.rename(columns={"nflverse_game_id": "game_id"})
    pairing["capture_kind"] = capture_kind
    return (
        pairing[empty_columns]
        .sort_values(
            ["game_id", "decision_label"],
            key=lambda s: s.map(DECISION_LABEL_ORDER) if s.name == "decision_label" else s,
        )
        .reset_index(drop=True)
    )


def cached_pairing_table(
    root: Path,
    *,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    labels: Iterable[str] | None = None,
    seasons: Iterable[int] | None = None,
    schedule: pd.DataFrame | None = None,
    inventory_digest: str | None = None,
) -> pd.DataFrame:
    label_key = "*" if labels is None else ",".join(sorted(labels))
    season_key = "*" if seasons is None else ",".join(str(int(x)) for x in sorted(seasons))
    if schedule is None:
        schedule_key = "none"
    else:
        schedule_key = _frame_content_digest(
            schedule[["game_id", "season", "week"]]
            .drop_duplicates("game_id")
            .sort_values("game_id")
            .reset_index(drop=True)
        )
    cache_root = evaluation_cache_root()
    key = None
    if cache_root is not None:
        key = _digest_text(
            _PAIRING_CACHE_VERSION,
            _estimator_source_digest(),
            inventory_digest or market_archive_inventory_digest(root),
            capture_kind,
            label_key,
            season_key,
            schedule_key,
        )
        cached = _read_cached_frame(cache_root / "pairing_table" / f"{key}.parquet")
        if cached is not None:
            return cached
    pairing = build_pairing_table(
        root, capture_kind=capture_kind, labels=labels, seasons=seasons, schedule=schedule
    )
    if cache_root is not None and key is not None:
        _write_cached_frame(pairing, cache_root / "pairing_table" / f"{key}.parquet")
    return pairing


def close_reference_table(pairing: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:

    required = {"game_id", "spread_line"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise DataContractError(f"Schedule close is missing columns: {', '.join(missing)}")
    candidates = pairing.loc[pairing["decision_label"].isin(CLOSE_LABEL_PRIORITY)][
        ["game_id", "decision_label", "home_spread", "spread_books"]
    ].copy()
    candidates["_priority"] = candidates["decision_label"].map(
        {label: index for index, label in enumerate(CLOSE_LABEL_PRIORITY)}
    )
    best = (
        candidates.sort_values("_priority")
        .groupby("game_id", as_index=False)
        .first()
        .rename(
            columns={
                "home_spread": "close_home_spread",
                "decision_label": "close_source",
                "spread_books": "close_books",
            }
        )
    )[["game_id", "close_home_spread", "close_source", "close_books"]]

    schedule_close = schedule[["game_id", "spread_line"]].drop_duplicates("game_id")
    merged = schedule_close.merge(best, on="game_id", how="left")
    missing_store = merged["close_home_spread"].isna()
    merged.loc[missing_store, "close_home_spread"] = merged.loc[missing_store, "spread_line"]
    merged.loc[missing_store, "close_source"] = "schedule_close"
    merged.loc[missing_store, "close_books"] = 0
    merged["close_books"] = merged["close_books"].astype(int)
    return merged[["game_id", "close_home_spread", "close_source", "close_books"]]


_SPREAD_PRICE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "decision_label",
    "capture_kind",
    "snapshot_timestamp_utc",
    "home_spread_price",
    "home_spread_price_books",
    "away_spread_price",
    "away_spread_price_books",
)


def spread_price_consensus_table(
    root: Path,
    *,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    labels: Iterable[str] | None = None,
    seasons: Iterable[int] | None = None,
    schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:

    quotes = load_decision_quotes(root, capture_kind=capture_kind, labels=labels, seasons=seasons)
    if schedule is not None and not quotes.empty:
        required = {"game_id", "season", "week"}
        missing = sorted(required.difference(schedule.columns))
        if missing:
            raise DataContractError(f"Pairing schedule is missing columns: {', '.join(missing)}")
        true_week = (
            schedule[["game_id", "season", "week"]]
            .drop_duplicates("game_id")
            .rename(
                columns={
                    "game_id": "nflverse_game_id",
                    "season": "_true_season",
                    "week": "_true_week",
                }
            )
        )
        quotes = quotes.merge(true_week, on="nflverse_game_id", how="left")
        quotes = quotes.loc[
            quotes["season"].eq(quotes["_true_season"]) & quotes["week"].eq(quotes["_true_week"])
        ].drop(columns=["_true_season", "_true_week"])
    if quotes.empty:
        return pd.DataFrame(columns=list(_SPREAD_PRICE_COLUMNS))
    consensus = decision_market_consensus(quotes)
    assert_monotone_decision_timeline(consensus)

    keys = ["nflverse_game_id", "season", "week", "decision_label"]
    home = consensus.loc[consensus["market"].eq("spreads") & consensus["outcome_side"].eq("HOME")][
        [*keys, "books", "consensus_price", "snapshot_timestamp_utc"]
    ].rename(columns={"books": "home_spread_price_books", "consensus_price": "home_spread_price"})
    away = consensus.loc[consensus["market"].eq("spreads") & consensus["outcome_side"].eq("AWAY")][
        [*keys, "books", "consensus_price"]
    ].rename(columns={"books": "away_spread_price_books", "consensus_price": "away_spread_price"})

    table = home.merge(away, on=keys, how="outer")
    table["capture_kind"] = capture_kind
    table = table.rename(columns={"nflverse_game_id": "game_id"})
    return (
        table[list(_SPREAD_PRICE_COLUMNS)]
        .sort_values(
            ["game_id", "decision_label"],
            key=lambda s: s.map(DECISION_LABEL_ORDER) if s.name == "decision_label" else s,
        )
        .reset_index(drop=True)
    )


_PICK_REQUIRED_COLUMNS = ("game_id", "side", "decision_label")


def score_clv(
    picks: pd.DataFrame,
    pairing: pd.DataFrame,
    close_reference: pd.DataFrame,
) -> pd.DataFrame:

    missing = sorted(set(_PICK_REQUIRED_COLUMNS).difference(picks.columns))
    if missing:
        raise DataContractError(f"CLV picks are missing columns: {', '.join(missing)}")
    unknown_sides = sorted(set(picks["side"]) - {"HOME", "AWAY"})
    if unknown_sides:
        raise ValueError(f"CLV picks contain unsupported sides: {', '.join(unknown_sides)}")

    decision_spreads = pairing[["game_id", "decision_label", "home_spread", "spread_books"]].rename(
        columns={"home_spread": "decision_home_spread", "spread_books": "decision_books"}
    )
    scored = picks.merge(decision_spreads, on=["game_id", "decision_label"], how="left")
    scored = scored.merge(close_reference, on="game_id", how="left")
    direction = scored["side"].map({"HOME": 1.0, "AWAY": -1.0})
    scored["clv_points"] = direction * (
        scored["close_home_spread"] - scored["decision_home_spread"]
    )
    if not {"season", "week"}.issubset(scored.columns):
        week_lookup = pairing[["game_id", "season", "week"]].drop_duplicates("game_id")
        scored = scored.merge(week_lookup, on="game_id", how="left")
    return scored


def clv_summary(scored: pd.DataFrame) -> dict[str, float]:

    values = pd.to_numeric(scored["clv_points"], errors="coerce").dropna()
    n = float(len(values))
    return {
        "n": n,
        "mean_clv_points": float(values.mean()) if n else float("nan"),
        "median_clv_points": float(values.median()) if n else float("nan"),
        "positive_clv_rate": float((values > 0.0).mean()) if n else float("nan"),
    }


def week_blocked_bootstrap(
    frame: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], dict[str, float]],
    *,
    block: BootstrapBlock = "week",
    samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 20260816,
    metric_columns: Sequence[str] | None = None,
    metric_draw_factory: (
        Callable[[pd.DataFrame], Callable[[Any], dict[str, float]] | None] | None
    ) = None,
) -> pd.DataFrame:

    if samples < 10:
        raise ValueError("samples must be at least 10")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if block not in ("week", "season"):
        raise ValueError("block must be 'week' or 'season'")
    group_columns = ["season", "week"] if block == "week" else ["season"]
    missing = sorted(set(group_columns).difference(frame.columns))
    if missing:
        raise DataContractError(f"Bootstrap frame is missing columns: {', '.join(missing)}")
    valid = frame.dropna(subset=["clv_points"]) if "clv_points" in frame.columns else frame
    if valid.empty:
        raise ValueError("No rows available to bootstrap")
    grouped_indices = list(valid.groupby(group_columns, sort=False, dropna=False).indices.values())
    if not grouped_indices:
        raise ValueError("Bootstrap frame contains no blocks")

    estimate = metric_fn(valid)
    resampled_from = valid
    if metric_columns is not None:
        wanted = set(metric_columns)
        resampled_from = valid.loc[:, [column for column in valid.columns if column in wanted]]
        resampled_from.attrs = {}
    metric_names = list(estimate)
    draw_metric = None if metric_draw_factory is None else metric_draw_factory(valid)
    if draw_metric is not None and set(draw_metric(np.arange(len(valid)))) != set(metric_names):
        draw_metric = None
    draws = np.empty((samples, len(metric_names)), dtype=float)
    generator = np.random.default_rng(seed)
    for sample_index in range(samples):
        selected = generator.integers(0, len(grouped_indices), size=len(grouped_indices))
        positions = np.concatenate([grouped_indices[index] for index in selected])
        sampled = (
            draw_metric(positions)
            if draw_metric is not None
            else metric_fn(resampled_from.take(positions))
        )
        draws[sample_index] = [sampled[name] for name in metric_names]

    tail = (1.0 - confidence) / 2.0
    lower = np.quantile(draws, tail, axis=0)
    upper = np.quantile(draws, 1.0 - tail, axis=0)
    return pd.DataFrame(
        {
            "metric": metric_names,
            "estimate": [estimate[name] for name in metric_names],
            "lower": lower,
            "upper": upper,
            "probability_positive": probability_positive_from_draws(draws, axis=0),
            "confidence": confidence,
            "block": block,
            "samples": samples,
        }
    )


def key_number_distance(
    spread: pd.Series, key_numbers: tuple[float, ...] = KEY_NUMBERS
) -> pd.Series:

    magnitude = spread.abs()
    distances = pd.concat([(magnitude - k).abs() for k in key_numbers], axis=1)
    return distances.min(axis=1)


def resolve_active_model_config(artifacts_root: Path) -> dict[str, Any]:

    manifest = load_active_ats_model(artifacts_root)
    if manifest is None or manifest.get("method") != "market_residual":
        return dict(_ACTIVE_MODEL_FALLBACK_CONFIG)
    ridge_alpha = manifest.get("ridge_alpha")
    return {
        "feature_profile": manifest.get(
            "feature_profile", _ACTIVE_MODEL_FALLBACK_CONFIG["feature_profile"]
        ),
        "regressor": manifest.get("regressor", _ACTIVE_MODEL_FALLBACK_CONFIG["regressor"]),
        "ridge_alpha": float(ridge_alpha) if ridge_alpha is not None else 10.0,
        "target": "market_residual",
        "model_id": manifest.get("model_id"),
        **(
            {"probability_method": manifest["probability_method"]}
            if "probability_method" in manifest
            else {}
        ),
        "calibration_method": manifest.get("calibration_method", "none"),
        "feature_table_sha256": manifest.get("feature_table_sha256"),
    }


def resolve_active_probability_method(
    artifacts_root: Path | None = None,
) -> ResidualSmoothingMethod:
    if artifacts_root is None:
        artifacts_root = Path(__file__).resolve().parents[2] / "artifacts"
    message = (
        "Cannot resolve probability_method from active model manifest at "
        f"{artifacts_root / 'active_ats_model.json'}; "
        "pass probability_method explicitly or provide a readable synchronized "
        "market-residual manifest containing probability_method"
    )
    try:
        config = resolve_active_model_config(artifacts_root)
    except (OSError, ValueError) as exc:
        raise ValueError(message) from exc
    method = config.get("probability_method")
    if not isinstance(method, str) or not method:
        raise ValueError(message)
    return cast(ResidualSmoothingMethod, method)


def active_model_residual_at_opener(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    feature_profile: MarginFeatureProfile = "player",
    model_name: str = "ridge",
    ridge_alpha: float = 10.0,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> pd.DataFrame:

    feature_columns = margin_feature_columns("market_residual", feature_profile)
    required = {"game_id", "gameday", "result", "ats_margin", "spread_line", *feature_columns}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"Active-model features are missing columns: {', '.join(missing)}")
    target_required = {"game_id", "season", "week", "tue_open_home_spread"}
    missing_targets = sorted(target_required.difference(targets.columns))
    if missing_targets:
        raise DataContractError(
            f"Active-model targets are missing columns: {', '.join(missing_targets)}"
        )

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna()].copy()

    rows: list[pd.DataFrame] = []
    for (season, week), group in targets.groupby(["season", "week"], sort=True):
        week_game_ids = set(group["game_id"])
        week_rows = frame.loc[frame["game_id"].isin(week_game_ids)]
        if week_rows.empty:
            continue
        cutoff = week_rows["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name=model_name,
            feature_profile=feature_profile,
            ridge_alpha=ridge_alpha,
        )
        scoring = week_rows.merge(
            group[["game_id", "tue_open_home_spread"]], on="game_id", how="inner"
        ).copy()
        scoring["spread_line"] = scoring["tue_open_home_spread"]
        predicted = model.predict(scoring)
        rows.append(
            pd.DataFrame(
                {
                    "game_id": scoring["game_id"].to_numpy(),
                    "season": season,
                    "week": week,
                    "active_model_residual_at_opener": predicted[
                        "predicted_market_residual"
                    ].to_numpy(),
                }
            )
        )
    if not rows:
        raise ValueError(
            "No target week had at least min_train_games completed training rows before it"
        )
    return pd.concat(rows, ignore_index=True)


def build_pilot_frame(
    root: Path,
    features: pd.DataFrame,
    *,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    active_model_config: dict[str, Any] | None = None,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> pd.DataFrame:

    required = {"game_id", "season", "week", "rest_diff", "spread_line", "gameday"}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"Pilot feature table is missing columns: {', '.join(missing)}")

    pairing = build_pairing_table(
        root,
        capture_kind=capture_kind,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=features,
    )
    if pairing.empty:
        return pd.DataFrame(
            columns=[*FROZEN_PILOT_FEATURES, "game_id", "season", "target_close_minus_open"]
        )
    close = close_reference_table(pairing, features)

    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "season", "week", "home_spread"]
    ].rename(columns={"home_spread": "tue_open_home_spread"})
    frame = tue_open.merge(close, on="game_id", how="inner")
    frame["target_close_minus_open"] = frame["close_home_spread"] - frame["tue_open_home_spread"]
    frame["tue_open_key_number_distance"] = key_number_distance(frame["tue_open_home_spread"])

    rest = features[["game_id", "rest_diff"]].drop_duplicates("game_id")
    frame = frame.merge(rest, on="game_id", how="left")

    config = active_model_config or dict(_ACTIVE_MODEL_FALLBACK_CONFIG)
    residual = active_model_residual_at_opener(
        features,
        frame[["game_id", "season", "week", "tue_open_home_spread"]],
        feature_profile=config["feature_profile"],
        model_name=config["regressor"],
        ridge_alpha=config["ridge_alpha"],
        min_train_games=min_train_games,
    )
    frame = frame.merge(
        residual[["game_id", "active_model_residual_at_opener"]], on="game_id", how="inner"
    )
    return frame.reset_index(drop=True)


def fit_pilot_model(train_frame: pd.DataFrame) -> Pipeline:

    missing = sorted(
        set(FROZEN_PILOT_FEATURES)
        .union({"target_close_minus_open"})
        .difference(train_frame.columns)
    )
    if missing:
        raise DataContractError(f"Pilot training frame is missing columns: {', '.join(missing)}")
    estimator = make_margin_estimator("ridge", ridge_alpha=FROZEN_PILOT_RIDGE_ALPHA)
    estimator.fit(
        train_frame.loc[:, list(FROZEN_PILOT_FEATURES)],
        train_frame["target_close_minus_open"],
    )
    return estimator


def evaluate_pilot(estimator: Pipeline, frame: pd.DataFrame) -> dict[str, Any]:

    if frame.empty:
        raise ValueError("Cannot evaluate the pilot model on an empty frame")
    actual = frame["target_close_minus_open"].to_numpy(dtype=float)
    predicted = np.asarray(
        estimator.predict(frame.loc[:, list(FROZEN_PILOT_FEATURES)]), dtype=float
    )

    movers = actual != 0.0
    direction_matches = np.sign(predicted[movers]) == np.sign(actual[movers])
    direction_accuracy = float(direction_matches.mean()) if movers.any() else float("nan")
    direction_n = int(movers.sum())

    baseline_predicted = np.zeros_like(actual)
    mae_model = float(np.mean(np.abs(actual - predicted)))
    mae_baseline = float(np.mean(np.abs(actual - baseline_predicted)))
    no_movement_direction_accuracy = float((~movers).mean())

    return {
        "n_games": len(frame),
        "direction_accuracy": direction_accuracy,
        "direction_n_movers": direction_n,
        "direction_accuracy_vs_50pct": direction_accuracy - 0.5 if movers.any() else float("nan"),
        "no_movement_baseline_direction_accuracy": no_movement_direction_accuracy,
        "mae_model": mae_model,
        "mae_no_movement_baseline": mae_baseline,
        "mae_improvement_vs_baseline": mae_baseline - mae_model,
    }


def threshold_policy_clv(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:

    if len(predicted) != len(frame):
        raise ValueError("predicted must have one value per row of frame")
    side = np.where(
        predicted >= threshold, "HOME", np.where(predicted <= -threshold, "AWAY", "PASS")
    )
    actual = frame["target_close_minus_open"].to_numpy(dtype=float)
    clv_points = np.where(side == "HOME", actual, np.where(side == "AWAY", -actual, np.nan))
    result = frame[["game_id", "season", "week"]].copy()
    result["side"] = side
    result["predicted_move"] = predicted
    result["clv_points"] = clv_points
    return result.loc[result["side"].ne("PASS")].reset_index(drop=True)


def run_predeclared_pilot(
    root: Path,
    features: pd.DataFrame,
    *,
    protocol: PilotSplit = FROZEN_PILOT_PROTOCOL,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    active_model_config: dict[str, Any] | None = None,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260816,
    threshold: float = 0.5,
) -> dict[str, Any]:

    pilot_frame = build_pilot_frame(
        root,
        features,
        capture_kind=capture_kind,
        active_model_config=active_model_config,
        min_train_games=min_train_games,
    )
    available_seasons = (
        sorted(pilot_frame["season"].unique().tolist()) if not pilot_frame.empty else []
    )
    coverage = pilot_frame.groupby("season").size().to_dict() if not pilot_frame.empty else {}

    train = pilot_frame.loc[
        pilot_frame["season"].between(protocol.train_start_season, protocol.train_end_season)
    ]
    validate = pilot_frame.loc[pilot_frame["season"].eq(protocol.validate_season)]
    test = pilot_frame.loc[pilot_frame["season"].eq(protocol.test_season)]

    if train.empty or validate.empty or test.empty:
        raise PilotProtocolBlocked(
            "Frozen protocol requires paired tue_open+close data for train seasons "
            f"{protocol.train_start_season}-{protocol.train_end_season}, validate season "
            f"{protocol.validate_season}, and test season {protocol.test_season}; the archive "
            f"currently has paired coverage for seasons {available_seasons} "
            f"(games per season: {coverage}). "
            f"train_games={len(train)} validate_games={len(validate)} test_games={len(test)}."
        )

    estimator = fit_pilot_model(train)
    validate_metrics = evaluate_pilot(estimator, validate)
    test_metrics = evaluate_pilot(estimator, test)

    test_predicted = np.asarray(
        estimator.predict(test.loc[:, list(FROZEN_PILOT_FEATURES)]), dtype=float
    )
    policy = threshold_policy_clv(test, test_predicted, threshold=threshold)
    if policy.empty:
        clv_bootstrap = pd.DataFrame(
            columns=["metric", "estimate", "lower", "upper", "confidence", "block", "samples"]
        )
    else:
        clv_bootstrap = week_blocked_bootstrap(
            policy, clv_summary, block="week", samples=bootstrap_samples, seed=bootstrap_seed
        )

    return {
        "protocol": {
            "train_start_season": protocol.train_start_season,
            "train_end_season": protocol.train_end_season,
            "validate_season": protocol.validate_season,
            "test_season": protocol.test_season,
            "frozen_features": list(FROZEN_PILOT_FEATURES),
            "ridge_alpha": FROZEN_PILOT_RIDGE_ALPHA,
            "threshold_points": threshold,
        },
        "coverage": {"available_seasons": available_seasons, "games_per_season": coverage},
        "train_games": len(train),
        "validate": validate_metrics,
        "test": test_metrics,
        "test_threshold_policy_bets": len(policy),
        "test_threshold_policy_clv_bootstrap": clv_bootstrap.to_dict(orient="records"),
    }


def sign_test_pilot_b(
    root: Path,
    features: pd.DataFrame,
    *,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    active_model_config: dict[str, Any] | None = None,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    confidence: float = 0.95,
) -> dict[str, Any]:

    pilot_frame = build_pilot_frame(
        root,
        features,
        capture_kind=capture_kind,
        active_model_config=active_model_config,
        min_train_games=min_train_games,
    )
    if pilot_frame.empty:
        raise ValueError("No paired games with a resolvable active-model residual are available")

    working = pilot_frame.loc[
        pilot_frame["active_model_residual_at_opener"].ne(0.0)
        & pilot_frame["target_close_minus_open"].ne(0.0)
    ].copy()
    excluded = len(pilot_frame) - len(working)
    working["correct"] = np.sign(working["active_model_residual_at_opener"]) == np.sign(
        working["target_close_minus_open"]
    )

    def _season_result(rows: pd.DataFrame) -> dict[str, Any]:
        successes = int(rows["correct"].sum())
        trials = len(rows)
        test = binomtest(successes, trials, p=0.5) if trials else None
        interval = test.proportion_ci(confidence_level=confidence) if test is not None else None
        return {
            "games": trials,
            "correct": successes,
            "accuracy": successes / trials if trials else float("nan"),
            "p_value_vs_50pct": test.pvalue if test is not None else float("nan"),
            "confidence_interval": (
                {"lower": interval.low, "upper": interval.high} if interval is not None else None
            ),
        }

    overall = _season_result(working)
    season_values = sorted(int(value) for value in working["season"].unique())
    per_season = {
        season: _season_result(working.loc[working["season"].eq(season)])
        for season in season_values
    }
    return {
        "games_excluded_zero_move_or_zero_signal": excluded,
        "overall": overall,
        "per_season": per_season,
        "confidence": confidence,
    }


class ClosePredictionUnavailable(ValueError):
    pass


def upcoming_week(features: pd.DataFrame) -> tuple[int, int]:

    required = {"season", "week", "result"}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"Feature table is missing columns: {', '.join(missing)}")
    unplayed = regular_season_rows(features)
    unplayed = unplayed.loc[unplayed["result"].isna()]
    if unplayed.empty:
        raise ValueError("Feature table has no unplayed games to predict a close for")
    first = unplayed.sort_values(["season", "week"]).iloc[0]
    return int(first["season"]), int(first["week"])


def live_tuesday_openers(root: Path) -> pd.DataFrame:

    columns = [
        "game_id",
        "tue_open_home_spread",
        "opener_books",
        "opener_observed_at_utc",
        "opener_basis",
    ]
    quotes = load_decision_quotes(root, capture_kind=LIVE_CAPTURE_KIND)
    if quotes.empty:
        return pd.DataFrame(columns=columns)
    spreads = quotes.loc[
        quotes["market"].eq("spreads")
        & quotes["outcome_side"].eq("HOME")
        & quotes["nflverse_game_id"].notna()
        & quotes["observed_at_utc"].lt(quotes["commence_time_utc"])
    ].copy()
    if spreads.empty:
        return pd.DataFrame(columns=columns)
    own_week = own_week_tuesday_quotes(spreads)
    if own_week.empty:
        return pd.DataFrame(columns=columns)
    opener = tuesday_opener_quotes(own_week)
    return opener.rename(
        columns={
            "nflverse_game_id": "game_id",
            "opener_home_spread": "tue_open_home_spread",
            "bookmakers": "opener_books",
            "observed_at_utc": "opener_observed_at_utc",
        }
    )[columns]


def _validate_close_predictions(predictions: pd.DataFrame) -> None:

    if predictions["game_id"].duplicated().any():
        raise DataContractError("Close predictions contain duplicate game_id rows")
    values = predictions[
        ["tue_open_home_spread", "predicted_close_minus_open", "predicted_close_home_spread"]
    ].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataContractError("Close predictions contain non-finite values")
    if float(np.abs(predictions["predicted_close_home_spread"]).max()) > 30.0:
        raise DataContractError("Predicted close outside the plausible NFL spread range (|x| > 30)")


def predict_close_for_week(
    root: Path,
    features: pd.DataFrame,
    *,
    season: int,
    week: int,
    protocol: PilotSplit = FROZEN_PILOT_PROTOCOL,
    train_capture_kind: str = HISTORICAL_CAPTURE_KIND,
    active_model_config: dict[str, Any] | None = None,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> dict[str, Any]:

    required = {"game_id", "season", "week", "rest_diff"}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"Feature table is missing columns: {', '.join(missing)}")

    schedule = features[["game_id", "season", "week", "rest_diff"]].drop_duplicates("game_id")
    target_games = schedule.loc[schedule["season"].eq(season) & schedule["week"].eq(week)]
    target = target_games.merge(live_tuesday_openers(root), on="game_id", how="inner")
    if target.empty:
        raise ClosePredictionUnavailable(
            f"No live Tuesday-opener quotes for season {season} week {week} are in the store; "
            "the predicted close becomes available after that week's Tuesday capture."
        )
    target["tue_open_key_number_distance"] = key_number_distance(target["tue_open_home_spread"])

    train_frame = build_pilot_frame(
        root,
        features,
        capture_kind=train_capture_kind,
        active_model_config=active_model_config,
        min_train_games=min_train_games,
    )
    train = (
        train_frame.loc[
            train_frame["season"].between(protocol.train_start_season, protocol.train_end_season)
        ]
        if not train_frame.empty
        else train_frame
    )
    if train.empty:
        raise PilotProtocolBlocked(
            "Frozen protocol requires paired tue_open+close training data for seasons "
            f"{protocol.train_start_season}-{protocol.train_end_season}; none is under {root}"
        )
    estimator = fit_pilot_model(train)

    config = active_model_config or dict(_ACTIVE_MODEL_FALLBACK_CONFIG)
    try:
        residual = active_model_residual_at_opener(
            features,
            target[["game_id", "season", "week", "tue_open_home_spread"]],
            feature_profile=config["feature_profile"],
            model_name=config["regressor"],
            ridge_alpha=config["ridge_alpha"],
            min_train_games=min_train_games,
        )
    except ValueError as error:
        raise ClosePredictionUnavailable(
            f"Cannot rebuild the opener-time residual for season {season} week {week}: {error}"
        ) from error
    target = target.merge(
        residual[["game_id", "active_model_residual_at_opener"]], on="game_id", how="inner"
    )

    predicted = np.asarray(
        estimator.predict(target.loc[:, list(FROZEN_PILOT_FEATURES)]), dtype=float
    )
    predictions = target[
        [
            "game_id",
            "season",
            "week",
            "tue_open_home_spread",
            "opener_books",
            "opener_observed_at_utc",
        ]
    ].copy()
    predictions["predicted_close_minus_open"] = predicted
    predictions["predicted_close_home_spread"] = (
        predictions["tue_open_home_spread"] + predictions["predicted_close_minus_open"]
    )
    _validate_close_predictions(predictions)
    return {
        "predictions": predictions.reset_index(drop=True),
        "train_games": len(train),
        "train_start_season": protocol.train_start_season,
        "train_end_season": protocol.train_end_season,
    }


PAPER_DECISION_COLUMNS: tuple[str, ...] = (
    "recorded_at_utc",
    "forecast_artifact",
    "forecast_created_at_utc",
    "model_id",
    "method",
    "decision_policy_id",
    "decision_policy_fingerprint",
    "game_id",
    "season",
    "week",
    "kickoff",
    "away_team",
    "home_team",
    "model_pick_side",
    "pre_arrest_pick_side",
    "former_policy_pick_side",
    "pick_side",
    "coach_fade_flip",
    "division_revenge_flip",
    "player_arrests_flip",
    "spread_gap_zone_flip",
    "composed_overlay_flip",
    "player_arrests_home_flag",
    "player_arrests_away_flag",
    "player_arrests_snapshot_id",
    "player_arrests_snapshot_fetched_at_utc",
    "player_arrests_safe_index_sha256",
    "schedule_snapshot_id",
    "schedule_parquet_sha256",
    "bet_side",
    "decision_home_spread",
    "edge",
    "is_best_pick",
)

_LEGACY_PAPER_DECISION_DEFAULTS: dict[str, Any] = {
    "is_best_pick": False,
    "decision_policy_id": "legacy_model_only",
    "decision_policy_fingerprint": "",
    "coach_fade_flip": False,
    "division_revenge_flip": False,
    "player_arrests_flip": False,
    "spread_gap_zone_flip": False,
    "composed_overlay_flip": False,
    "player_arrests_home_flag": False,
    "player_arrests_away_flag": False,
    "player_arrests_snapshot_id": "",
    "player_arrests_snapshot_fetched_at_utc": pd.NaT,
    "player_arrests_safe_index_sha256": "",
    "schedule_snapshot_id": "",
    "schedule_parquet_sha256": "",
}

_FOUR_OVERLAY_POLICY_ID = "overlay_union_coach_division_revenge_player_arrests_spread_gap_v1"
_THREE_OVERLAY_POLICY_ID = "overlay_union_coach_division_revenge_player_arrests_v2"
_NINE_OVERLAY_POLICY_ID = (
    "overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3"
)
_COMPOSITION_POLICY_MEMBER_FLIPS: dict[str, tuple[str, ...]] = {
    _FOUR_OVERLAY_POLICY_ID: (
        "coach_fade_flip",
        "division_revenge_flip",
        "player_arrests_flip",
        "spread_gap_zone_flip",
    ),
    _THREE_OVERLAY_POLICY_ID: (
        "coach_fade_flip",
        "division_revenge_flip",
        "player_arrests_flip",
    ),
}

_PARTIAL_COMPOSITION_POLICY_MEMBER_FLIPS: dict[str, tuple[str, ...]] = {
    _NINE_OVERLAY_POLICY_ID: (
        "coach_fade_flip",
        "division_revenge_flip",
        "player_arrests_flip",
    ),
}

COMPOSITION_POLICY_IDS: frozenset[str] = frozenset(_COMPOSITION_POLICY_MEMBER_FLIPS) | frozenset(
    _PARTIAL_COMPOSITION_POLICY_MEMBER_FLIPS
)

_CLOSE_REFERENCE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "close_home_spread",
    "close_source",
    "close_books",
    "close_observed_at_utc",
)

VALID_PICK_SIDES = frozenset({"HOME", "AWAY"})
VALID_BET_SIDES = frozenset({"HOME", "AWAY", "PASS"})


def paper_decision_ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "clv_ledger" / "decisions.parquet"


def load_paper_decisions(artifacts_root: Path) -> pd.DataFrame:

    path = paper_decision_ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(PAPER_DECISION_COLUMNS))
    ledger = pd.read_parquet(path)
    for column, default in _LEGACY_PAPER_DECISION_DEFAULTS.items():
        if column not in ledger.columns:
            ledger[column] = default
    if "model_pick_side" not in ledger.columns and "pick_side" in ledger.columns:
        ledger["model_pick_side"] = ledger["pick_side"]
    if "pre_arrest_pick_side" not in ledger.columns and "pick_side" in ledger.columns:
        ledger["pre_arrest_pick_side"] = ledger["pick_side"]
    if "former_policy_pick_side" not in ledger.columns and "pick_side" in ledger.columns:
        ledger["former_policy_pick_side"] = ledger["pick_side"]
    missing = sorted(set(PAPER_DECISION_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(f"Paper-decision ledger is missing columns: {', '.join(missing)}")
    if ledger["game_id"].duplicated().any():
        raise DataContractError(f"Paper-decision ledger contains duplicate game rows: {path}")
    ledger["is_best_pick"] = ledger["is_best_pick"].fillna(False).astype(bool)
    flagged = ledger.loc[ledger["is_best_pick"]]
    if flagged.duplicated(subset=["season", "week"]).any():
        raise DataContractError(
            f"Paper-decision ledger marks more than one Best Pick in a week: {path}"
        )
    for policy_id, member_columns in _COMPOSITION_POLICY_MEMBER_FLIPS.items():
        composed = ledger["decision_policy_id"].astype(str).eq(policy_id)
        if not composed.any():
            continue
        member_flip = ledger.loc[composed, list(member_columns)].astype(bool).any(axis=1)
        declared_flip = ledger.loc[composed, "composed_overlay_flip"].astype(bool)
        observed_flip = (
            ledger.loc[composed, "model_pick_side"]
            .astype(str)
            .ne(ledger.loc[composed, "pick_side"].astype(str))
        )
        if not member_flip.equals(declared_flip) or not observed_flip.equals(declared_flip):
            raise DataContractError(
                f"Composition rows for {policy_id} violate the raw-card OR-union invariant"
            )
        if policy_id == _THREE_OVERLAY_POLICY_ID and (
            ledger.loc[composed, "spread_gap_zone_flip"].astype(bool).any()
        ):
            raise DataContractError(
                "Three-member composition rows must never carry a spread-gap zone flip "
                "(retired from the played card 2026-09-07)"
            )
    for policy_id, member_columns in _PARTIAL_COMPOSITION_POLICY_MEMBER_FLIPS.items():
        composed = ledger["decision_policy_id"].astype(str).eq(policy_id)
        if not composed.any():
            continue
        member_flip = ledger.loc[composed, list(member_columns)].astype(bool).any(axis=1)
        declared_flip = ledger.loc[composed, "composed_overlay_flip"].astype(bool)
        observed_flip = (
            ledger.loc[composed, "model_pick_side"]
            .astype(str)
            .ne(ledger.loc[composed, "pick_side"].astype(str))
        )
        if not observed_flip.equals(declared_flip) or bool((member_flip & ~declared_flip).any()):
            raise DataContractError(
                f"Composition rows for {policy_id} violate the raw-card OR-union invariant"
            )
    return ledger[list(PAPER_DECISION_COLUMNS)]


RECORDING_LOCK_WINDOW = timedelta(days=7)


def refuse_if_outside_recording_lock_window(
    kickoffs: pd.Series, recorded_at: pd.Timestamp, *, ledger: str
) -> None:

    earliest_kickoff = kickoffs.min()
    if pd.isna(earliest_kickoff):
        return
    gap = earliest_kickoff - recorded_at
    if gap > RECORDING_LOCK_WINDOW:
        raise ValueError(
            f"Refusing to record to the {ledger} ledger: this week's earliest kickoff "
            f"({earliest_kickoff.isoformat()}) is {gap.days} days after the recording "
            f"instant ({recorded_at.isoformat()}), more than RECORDING_LOCK_WINDOW "
            f"({RECORDING_LOCK_WINDOW.days} days). This looks like a rehearsal or a "
            "test run made weeks before the week's real Tuesday lock, not the "
            "deliberate recording itself -- the exact shape of the 2026-08-18 incident "
            "(docs/prospective_evidence.md, 'Known divergence'). If this really is the "
            "week's deliberate lock-day recording, the schedule/kickoff data is wrong; "
            "fix that, don't widen this window for one call."
        )


@dataclass(frozen=True)
class PlayedCardView:
    season: int
    week: int
    model_id: str
    method: str
    forecast_artifact: str
    forecast_created_at_utc: pd.Timestamp
    recorded_at: pd.Timestamp
    card: pd.DataFrame
    kickoffs: pd.Series
    played_card: pd.DataFrame
    model_pick_side: pd.Series
    pre_arrest_pick_side: pd.Series
    former_policy_pick_side: pd.Series
    final_pick_side: pd.Series
    coach_flip_ids: frozenset[str]
    division_flip_ids: frozenset[str]
    arrest_flip_ids: frozenset[str]
    spread_gap_flip_ids: frozenset[str]
    composed_flip_ids: frozenset[str]
    decision_policy_id: str
    decision_policy_fingerprint: str
    view: Any


def current_played_card_view(
    artifacts_root: Path,
    *,
    data_root: Path | None = None,
    now: datetime | None = None,
    require_fresh_arrest_overlay: bool = True,
    forecast_artifact: str | None = None,
) -> PlayedCardView:

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError("No synchronized active ATS model is available to record decisions from")
    if forecast_artifact is not None:
        forecast = (artifacts_root / forecast_artifact).resolve()
        if not forecast.is_dir() or artifacts_root.resolve() not in forecast.parents:
            raise ValueError(f"Forecast artifact is not a directory under artifacts: {forecast}")
        recorded_artifact = forecast.relative_to(artifacts_root.resolve()).as_posix()
    else:
        linked = active_artifact_path(artifacts_root, active, "weekly_forecast")
        if linked is None:
            raise ValueError("Active ATS model has no linked weekly forecast")
        forecast = linked
        recorded_artifact = str(active["weekly_forecast"]["artifact"])
    recommendations_path = forecast / "recommendations.csv"
    metadata_path = forecast / "metadata.json"
    if not recommendations_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Linked weekly forecast is incomplete: {forecast}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if forecast_artifact is None and metadata.get("active_model_id") != active.get("model_id"):
        raise ValueError("Weekly forecast model ID does not match the active model")
    if not metadata.get("active_model_id"):
        raise ValueError("Weekly forecast metadata names no model id")
    recorded_model_id = str(metadata.get("active_model_id"))
    if metadata.get("synchronization_status") != "SYNCHRONIZED":
        raise ValueError("Weekly forecast is not synchronized with an evaluation")

    card = pd.read_csv(recommendations_path)
    required = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
        "bet_side",
        "edge",
        "method",
    }
    missing = sorted(required.difference(card.columns))
    if missing:
        raise DataContractError(f"Weekly forecast card is missing columns: {', '.join(missing)}")
    method = str(metadata.get("ats_method") or active.get("method"))
    if not card["method"].eq(method).all():
        raise DataContractError("Weekly forecast card contains a method other than the active one")
    if card["game_id"].duplicated().any():
        raise DataContractError("Weekly forecast card contains duplicate games")
    unknown_bets = sorted(set(card["bet_side"].astype(str)) - VALID_BET_SIDES)
    if unknown_bets:
        raise DataContractError(
            f"Weekly card contains invalid bet sides: {', '.join(unknown_bets)}"
        )
    spreads = pd.to_numeric(card["spread_line"], errors="coerce")
    if not np.isfinite(spreads.to_numpy(dtype=float)).all():
        raise DataContractError("Weekly forecast card has games without a decision spread")
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Weekly forecast card has games without a kickoff timestamp")

    recorded_at = _record_instant(now)
    sweep_path = forecast / "line_sweep.parquet"
    sweep = pd.read_parquet(sweep_path) if sweep_path.is_file() else pd.DataFrame()
    from nfl_ats.card_view import resolve_card_view

    view = resolve_card_view(
        card,
        sweep,
        metadata,
        data_root=data_root,
        now=recorded_at.to_pydatetime(),
        require_fresh_arrest_overlay=require_fresh_arrest_overlay,
    )
    raw_card = card.reset_index(drop=True)
    coach_card = view.overlay.overlaid_predictions.reset_index(drop=True)
    former_policy_card = view.arrest_overlay.overlaid_predictions.reset_index(drop=True)
    played_card = view.predictions.reset_index(drop=True)
    model_pick_side = pd.Series(
        np.where(raw_card["home_cover_probability"].ge(0.5), "HOME", "AWAY"),
        index=raw_card.index,
    )
    pre_arrest_pick_side = pd.Series(
        np.where(coach_card["home_cover_probability"].ge(0.5), "HOME", "AWAY"),
        index=raw_card.index,
    )
    former_policy_pick_side = pd.Series(
        np.where(former_policy_card["home_cover_probability"].ge(0.5), "HOME", "AWAY"),
        index=raw_card.index,
    )
    final_pick_side = pd.Series(
        np.where(played_card["home_cover_probability"].ge(0.5), "HOME", "AWAY"),
        index=raw_card.index,
    )
    composition = view.production_overlay
    if require_fresh_arrest_overlay and composition is None:
        raise DataContractError("Four-overlay production policy was not resolved")
    member_flip_ids: dict[str, set[str]] = {}
    if composition is not None:
        member_flip_ids = {
            member.member_id: set(member.flipped_game_ids) for member in composition.members
        }
    coach_flip_ids = member_flip_ids.get(
        "coach_fade", {flip.game_id for flip in view.overlay.flips}
    )
    division_flip_ids = member_flip_ids.get("division_revenge_tilt", set())
    arrest_flip_ids = member_flip_ids.get("player_arrests_back_side_policy", set())
    spread_gap_flip_ids = member_flip_ids.get("spread_gap_zone_fade", set())
    composed_flip_ids = (
        set(composition.union_flipped_game_ids) if composition is not None else arrest_flip_ids
    )
    decision_policy_id = (
        composition.policy_id if composition is not None else "coach_fade_then_player_arrests_v1"
    )
    decision_policy_fingerprint = composition.policy_fingerprint if composition is not None else ""

    return PlayedCardView(
        season=int(card["season"].iloc[0]),
        week=int(card["week"].iloc[0]),
        model_id=recorded_model_id,
        method=method,
        forecast_artifact=recorded_artifact,
        forecast_created_at_utc=pd.to_datetime(
            metadata.get("created_at_utc"), utc=True, errors="coerce"
        ),
        recorded_at=recorded_at,
        card=raw_card,
        kickoffs=kickoffs,
        played_card=played_card,
        model_pick_side=model_pick_side,
        pre_arrest_pick_side=pre_arrest_pick_side,
        former_policy_pick_side=former_policy_pick_side,
        final_pick_side=final_pick_side,
        coach_flip_ids=frozenset(coach_flip_ids),
        division_flip_ids=frozenset(division_flip_ids),
        arrest_flip_ids=frozenset(arrest_flip_ids),
        spread_gap_flip_ids=frozenset(spread_gap_flip_ids),
        composed_flip_ids=frozenset(composed_flip_ids),
        decision_policy_id=decision_policy_id,
        decision_policy_fingerprint=decision_policy_fingerprint,
        view=view,
    )


def record_paper_decisions(
    artifacts_root: Path,
    *,
    data_root: Path | None = None,
    now: datetime | None = None,
    require_fresh_arrest_overlay: bool = True,
    forecast_artifact: str | None = None,
    replace_week: bool = False,
) -> dict[str, Any]:

    played = current_played_card_view(
        artifacts_root,
        data_root=data_root,
        now=now,
        require_fresh_arrest_overlay=require_fresh_arrest_overlay,
        forecast_artifact=forecast_artifact,
    )
    card = played.card
    recorded_artifact = played.forecast_artifact
    recorded_model_id = played.model_id
    method = played.method
    kickoffs = played.kickoffs
    spreads = pd.to_numeric(card["spread_line"], errors="coerce")
    forecast_created_at_utc = played.forecast_created_at_utc
    view = played.view
    model_pick_side = played.model_pick_side
    pre_arrest_pick_side = played.pre_arrest_pick_side
    former_policy_pick_side = played.former_policy_pick_side
    final_pick_side = played.final_pick_side
    coach_flip_ids = played.coach_flip_ids
    division_flip_ids = played.division_flip_ids
    arrest_flip_ids = played.arrest_flip_ids
    spread_gap_flip_ids = played.spread_gap_flip_ids
    composed_flip_ids = played.composed_flip_ids
    decision_policy_id = played.decision_policy_id
    decision_policy_fingerprint = played.decision_policy_fingerprint
    recorded_at = played.recorded_at
    schedule_snapshot_id = ""
    schedule_parquet_sha256 = ""
    if data_root is not None:
        schedule_snapshot = latest_snapshot(data_root / "raw")
        schedule_path = schedule_snapshot.schedules_path
        schedule_snapshot_id = schedule_snapshot.snapshot_id
        schedule_parquet_sha256 = sha256_file(schedule_path)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="paper-decision")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_paper_decisions(artifacts_root)
    season = played.season
    week = played.week
    replaced_rows = 0
    left_post_kickoff = 0
    dropped_best_pick_id: str | None = None
    if replace_week and not existing.empty:
        flagged = existing.loc[
            existing["season"].astype(int).eq(season)
            & existing["week"].astype(int).eq(week)
            & existing["is_best_pick"].fillna(False).astype(bool),
            "game_id",
        ]
        dropped_best_pick_id = str(flagged.iloc[0]) if len(flagged) else None
        existing, replaced_rows, left_post_kickoff = replace_week_rows(
            existing,
            paper_decision_ledger_path(artifacts_root),
            season=season,
            week=week,
            recorded_at=recorded_at,
            columns=PAPER_DECISION_COLUMNS,
        )
    already = card["game_id"].isin(set(existing["game_id"].astype(str)))
    fresh = card.loc[pre_kickoff & ~already]
    week_already_flagged = bool(
        existing.loc[
            existing["season"].astype(int).eq(season)
            & existing["week"].astype(int).eq(week)
            & existing["is_best_pick"].astype(bool)
        ].shape[0]
    )
    best_pick_id: str | None = None
    if not week_already_flagged and bool(pre_kickoff.all()):
        best_pick_id = view.nomination.active_game_id
    elif not week_already_flagged and dropped_best_pick_id is not None:
        best_pick_id = dropped_best_pick_id

    decisions = pd.DataFrame(
        {
            "recorded_at_utc": recorded_at,
            "forecast_artifact": recorded_artifact,
            "forecast_created_at_utc": forecast_created_at_utc,
            "model_id": recorded_model_id,
            "method": method,
            "decision_policy_id": decision_policy_id,
            "decision_policy_fingerprint": decision_policy_fingerprint,
            "game_id": fresh["game_id"].astype(str),
            "season": fresh["season"].astype(int),
            "week": fresh["week"].astype(int),
            "kickoff": kickoffs.loc[fresh.index],
            "away_team": fresh["away_team"].astype(str),
            "home_team": fresh["home_team"].astype(str),
            "model_pick_side": model_pick_side.loc[fresh.index].astype(str),
            "pre_arrest_pick_side": pre_arrest_pick_side.loc[fresh.index].astype(str),
            "former_policy_pick_side": former_policy_pick_side.loc[fresh.index].astype(str),
            "pick_side": final_pick_side.loc[fresh.index].astype(str),
            "coach_fade_flip": fresh["game_id"].astype(str).isin(coach_flip_ids),
            "division_revenge_flip": fresh["game_id"].astype(str).isin(division_flip_ids),
            "player_arrests_flip": fresh["game_id"].astype(str).isin(arrest_flip_ids),
            "spread_gap_zone_flip": fresh["game_id"].astype(str).isin(spread_gap_flip_ids),
            "composed_overlay_flip": fresh["game_id"].astype(str).isin(composed_flip_ids),
            "player_arrests_home_flag": view.arrest_overlay.home_flags.loc[fresh.index].astype(
                bool
            ),
            "player_arrests_away_flag": view.arrest_overlay.away_flags.loc[fresh.index].astype(
                bool
            ),
            "player_arrests_snapshot_id": str(view.arrest_overlay.snapshot_id or ""),
            "player_arrests_snapshot_fetched_at_utc": view.arrest_overlay.snapshot_fetched_at_utc,
            "player_arrests_safe_index_sha256": str(view.arrest_overlay.safe_index_sha256 or ""),
            "schedule_snapshot_id": schedule_snapshot_id,
            "schedule_parquet_sha256": schedule_parquet_sha256,
            "bet_side": np.where(
                final_pick_side.loc[fresh.index].eq(model_pick_side.loc[fresh.index]),
                fresh["bet_side"].astype(str),
                "PASS",
            ),
            "decision_home_spread": spreads.loc[fresh.index].astype(float),
            "edge": pd.to_numeric(fresh["edge"], errors="coerce").where(
                final_pick_side.loc[fresh.index].eq(model_pick_side.loc[fresh.index])
            ),
            "is_best_pick": fresh["game_id"].astype(str).eq(str(best_pick_id)),
        }
    )
    if decisions.empty:
        combined = existing.copy()
    elif existing.empty:
        combined = decisions
    else:
        combined = pd.concat([existing, decisions], ignore_index=True)
    best_pick_recorded = False
    flag_written = False
    if best_pick_id is not None and not combined.empty:
        target = combined["game_id"].astype(str).eq(best_pick_id)
        if target.any():
            best_pick_recorded = True
            if not combined.loc[target, "is_best_pick"].fillna(False).astype(bool).any():
                combined.loc[target, "is_best_pick"] = True
                flag_written = True
    if not decisions.empty or flag_written:
        combined["is_best_pick"] = combined["is_best_pick"].fillna(False).astype(bool)
        atomic_parquet(
            combined[list(PAPER_DECISION_COLUMNS)], paper_decision_ledger_path(artifacts_root)
        )
        ledger_rows = len(combined)
    else:
        ledger_rows = len(existing)
    return {
        "season": season,
        "week": week,
        "recorded": len(decisions),
        "replaced_rows": replaced_rows,
        "left_post_kickoff": left_post_kickoff,
        "forecast_artifact": recorded_artifact,
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "ledger_rows": int(ledger_rows),
        "best_pick_game_id": best_pick_id if best_pick_recorded else None,
        "best_pick_recorded": best_pick_recorded,
        "best_pick_already_recorded": week_already_flagged,
        "decision_policy_id": decision_policy_id,
        "decision_policy_fingerprint": decision_policy_fingerprint,
        "coach_flip_count": view.overlay.flip_count,
        "player_arrests_flip_count": len(arrest_flip_ids),
        "division_revenge_flip_count": len(division_flip_ids),
        "spread_gap_zone_flip_count": len(spread_gap_flip_ids),
        "composed_overlay_flip_count": len(composed_flip_ids),
        "player_arrests_snapshot_id": view.arrest_overlay.snapshot_id,
        "player_arrests_safe_index_sha256": view.arrest_overlay.safe_index_sha256,
    }


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def live_close_reference(root: Path, schedule: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:

    required = {"game_id", "spread_line", "result"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise DataContractError(f"Close schedule is missing columns: {', '.join(missing)}")
    cutoff = _record_instant(as_of)

    live = pd.DataFrame(columns=list(_CLOSE_REFERENCE_COLUMNS))
    quotes = load_decision_quotes(root, capture_kind=LIVE_CAPTURE_KIND)
    if not quotes.empty:
        spreads = quotes.loc[
            quotes["market"].eq("spreads")
            & quotes["outcome_side"].eq("HOME")
            & quotes["nflverse_game_id"].notna()
            & quotes["observed_at_utc"].lt(quotes["commence_time_utc"])
            & quotes["commence_time_utc"].le(cutoff)
        ]
        if not spreads.empty:
            last_per_book = (
                spreads.sort_values("observed_at_utc")
                .groupby(["nflverse_game_id", "bookmaker_key"], as_index=False)
                .tail(1)
            )
            live = (
                last_per_book.groupby("nflverse_game_id", as_index=False)
                .agg(
                    close_home_spread=("home_spread_line", "median"),
                    close_books=("bookmaker_key", "nunique"),
                    close_observed_at_utc=("observed_at_utc", "max"),
                )
                .rename(columns={"nflverse_game_id": "game_id"})
            )
            live["close_source"] = "live_store_close"
            live = live[list(_CLOSE_REFERENCE_COLUMNS)]

    completed = schedule.loc[
        schedule["result"].notna(), ["game_id", "spread_line"]
    ].drop_duplicates("game_id")
    fallback_rows = completed.loc[~completed["game_id"].isin(set(live["game_id"].astype(str)))]
    fallback = pd.DataFrame(
        {
            "game_id": fallback_rows["game_id"].astype(str),
            "close_home_spread": pd.to_numeric(fallback_rows["spread_line"], errors="coerce"),
            "close_source": "schedule_close",
            "close_books": 0,
            "close_observed_at_utc": pd.NaT,
        }
    )
    frames = [frame for frame in (live, fallback) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=list(_CLOSE_REFERENCE_COLUMNS))
    combined = pd.concat(frames, ignore_index=True)
    return combined[list(_CLOSE_REFERENCE_COLUMNS)].reset_index(drop=True)


def score_paper_ledger(decisions: pd.DataFrame, close_reference: pd.DataFrame) -> pd.DataFrame:

    required = {"game_id", "season", "week", "pick_side", "bet_side", "decision_home_spread"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise DataContractError(f"Ledger decisions are missing columns: {', '.join(missing)}")
    if decisions["game_id"].duplicated().any():
        raise DataContractError("Ledger decisions contain duplicate game rows")
    unknown_picks = sorted(set(decisions["pick_side"].astype(str)) - VALID_PICK_SIDES)
    if unknown_picks:
        raise ValueError(f"Ledger contains unsupported pick sides: {', '.join(unknown_picks)}")
    unknown_bets = sorted(set(decisions["bet_side"].astype(str)) - VALID_BET_SIDES)
    if unknown_bets:
        raise ValueError(f"Ledger contains unsupported bet sides: {', '.join(unknown_bets)}")

    reference = (
        close_reference
        if not close_reference.empty
        else pd.DataFrame(columns=list(_CLOSE_REFERENCE_COLUMNS))
    )
    scored = decisions.merge(reference, on="game_id", how="left")
    move = scored["close_home_spread"] - scored["decision_home_spread"]
    pick_direction = scored["pick_side"].map({"HOME": 1.0, "AWAY": -1.0})
    scored["clv_points"] = pick_direction * move
    bet_direction = scored["bet_side"].map({"HOME": 1.0, "AWAY": -1.0})
    scored["bet_clv_points"] = bet_direction * move
    scored["clv_status"] = np.where(scored["clv_points"].notna(), "scored", "pending")
    return scored


def pick_correct(pick_home: pd.Series, settle_margin: pd.Series) -> pd.Series:

    covered_home = settle_margin.gt(0.0)
    correct = np.where(pick_home.astype(bool), covered_home, ~covered_home).astype(float)
    return pd.Series(np.where(settle_margin.eq(0.0), np.nan, correct), index=settle_margin.index)


def normalise_opener_line_override(override: pd.DataFrame | None) -> pd.DataFrame | None:
    if override is None:
        return None
    required = {"game_id", "home_spread"}
    missing = sorted(required.difference(override.columns))
    if missing:
        raise DataContractError(f"Opener line override is missing columns: {', '.join(missing)}")
    frame = override.loc[
        :, [c for c in ("game_id", "home_spread", "books") if c in override]
    ].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["home_spread"] = pd.to_numeric(frame["home_spread"], errors="coerce")
    if frame["home_spread"].isna().any():
        raise DataContractError("Opener line override contains a non-numeric home spread")
    if frame["game_id"].duplicated().any():
        raise DataContractError("Opener line override contains duplicate game rows")
    if "books" not in frame.columns:
        frame["books"] = 1
    frame["books"] = pd.to_numeric(frame["books"], errors="coerce").fillna(1).astype(int)
    frame.attrs = {}
    return frame.sort_values("game_id").reset_index(drop=True)


def opener_pick_evaluation(
    root: Path,
    features: pd.DataFrame,
    *,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
    active_model_config: dict[str, Any] | None = None,
    artifacts_root: Path | None = None,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    home_side_offset: bool | None = None,
    opener_line_override: pd.DataFrame | None = None,
) -> pd.DataFrame:

    apply_offset = HOME_SIDE_OFFSET_SERVED if home_side_offset is None else bool(home_side_offset)
    config = active_model_config or dict(_ACTIVE_MODEL_FALLBACK_CONFIG)
    if config.get("calibration_method", "none") != "none":
        raise ValueError("Opener evaluation supports only uncalibrated margin probabilities")
    probability_method = (
        config["probability_method"]
        if "probability_method" in config
        else resolve_active_probability_method(artifacts_root)
    )
    profile: MarginFeatureProfile = config["feature_profile"]
    feature_columns = margin_feature_columns("market_residual", profile)
    required = {
        "game_id",
        "season",
        "week",
        "gameday",
        "result",
        "ats_margin",
        "spread_line",
        *feature_columns,
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"Opener evaluation is missing columns: {', '.join(missing)}")
    features = regular_season_rows(features)
    features = features.loc[:, [c for c in features.columns if c in required]]
    features.attrs = {}

    override = normalise_opener_line_override(opener_line_override)

    cache_root = evaluation_cache_root()
    result_cache: Path | None = None
    inventory_digest: str | None = None
    if cache_root is not None:
        inventory_digest = market_archive_inventory_digest(root)
        result_cache = (
            cache_root
            / "opener_pick_evaluation"
            / (
                _digest_text(
                    _OPENER_EVAL_CACHE_VERSION,
                    _estimator_source_digest(),
                    inventory_digest,
                    capture_kind,
                    json.dumps(config, sort_keys=True, default=str),
                    str(probability_method),
                    str(min_train_games),
                    str(apply_offset),
                    _frame_content_digest(features.reset_index(drop=True)),
                    "none" if override is None else _frame_content_digest(override),
                )
                + ".parquet"
            )
        )
        cached_result = _read_cached_frame(result_cache)
        if cached_result is not None:
            return cached_result

    pairing = cached_pairing_table(
        root,
        capture_kind=capture_kind,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=features,
        inventory_digest=inventory_digest,
    )
    if pairing.empty:
        raise ValueError(f"No {capture_kind!r} snapshots with decision quotes under {root}")
    close = close_reference_table(pairing, features)
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "season", "week", "home_spread", "spread_books"]
    ].rename(columns={"home_spread": "tue_open_home_spread", "spread_books": "opener_books"})
    if override is not None:
        tue_open = tue_open.drop(columns=["tue_open_home_spread", "opener_books"]).merge(
            override.rename(
                columns={"home_spread": "tue_open_home_spread", "books": "opener_books"}
            ),
            on="game_id",
            how="inner",
        )
        if tue_open.empty:
            raise ValueError("The opener line override covers none of the archived opener games")
    paired = tue_open.merge(close, on="game_id", how="inner")

    outcomes = features[["game_id", "result"]].drop_duplicates("game_id")
    paired = paired.merge(outcomes, on="game_id", how="inner")
    paired = paired.loc[pd.to_numeric(paired["result"], errors="coerce").notna()].copy()
    if paired.empty:
        raise ValueError("No completed games have both a Tuesday opener and a close")

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna()].copy()

    scored_weeks: list[pd.DataFrame] = []
    stream_columns = ["game_id", "season", "week", "spread_line", "point_incumbent", "result"]
    archive_stream = pd.DataFrame(columns=stream_columns)
    for (season, week), group in paired.groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"]))]
        if week_rows.empty:
            continue
        cutoff = week_rows["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name=config["regressor"],
            feature_profile=profile,
            ridge_alpha=config["ridge_alpha"],
        )
        scoring = week_rows.merge(
            group[["game_id", "tue_open_home_spread", "close_home_spread"]],
            on="game_id",
            how="inner",
        ).copy()
        at_open = scoring.copy()
        at_open["spread_line"] = at_open["tue_open_home_spread"]
        at_close = scoring.copy()
        at_close["spread_line"] = at_close["close_home_spread"]
        scored = scoring[["game_id"]].copy()
        scored["probability_method"] = probability_method
        scored["season"] = int(str(season))
        scored["week"] = int(str(week))
        predicted_at_open = model.predict(at_open, probability_method=probability_method)
        predicted_at_close = model.predict(at_close, probability_method=probability_method)
        scored["residual_at_open"] = predicted_at_open["predicted_market_residual"].to_numpy()
        scored["residual_at_close"] = predicted_at_close["predicted_market_residual"].to_numpy()
        scored["home_cover_probability_at_open_raw"] = predicted_at_open[
            "home_cover_probability"
        ].to_numpy()
        scored["home_cover_probability_at_close_raw"] = predicted_at_close[
            "home_cover_probability"
        ].to_numpy()
        if apply_offset and not archive_stream.empty:
            fitted = fit_home_side_offsets(
                prior_rows_before(archive_stream, int(str(season)), int(str(week)))
            )
            offsets = fitted.offset_for(at_open["spread_line"]).fillna(0.0).to_numpy(dtype=float)
        else:
            offsets = np.zeros(len(at_open), dtype=float)
        scored["home_side_offset_at_open"] = offsets
        if apply_offset and np.any(offsets != 0.0):
            served_at_open = model.predict(
                at_open, probability_method=probability_method, center_offset=offsets
            )
            served_at_close = model.predict(
                at_close, probability_method=probability_method, center_offset=offsets
            )
            scored["home_cover_probability_at_open"] = served_at_open[
                "home_cover_probability"
            ].to_numpy()
            scored["home_cover_probability_at_close"] = served_at_close[
                "home_cover_probability"
            ].to_numpy()
        else:
            scored["home_cover_probability_at_open"] = scored["home_cover_probability_at_open_raw"]
            scored["home_cover_probability_at_close"] = scored[
                "home_cover_probability_at_close_raw"
            ]
        scored["residual_at_open_served"] = scored["residual_at_open"] + offsets
        scored["residual_at_close_served"] = scored["residual_at_close"] + offsets
        scored_weeks.append(scored)
        week_stream = pd.DataFrame(
            {
                "game_id": scoring["game_id"].astype(str).to_numpy(),
                "season": int(str(season)),
                "week": int(str(week)),
                "spread_line": pd.to_numeric(
                    scoring["tue_open_home_spread"], errors="coerce"
                ).to_numpy(),
                "point_incumbent": (
                    pd.to_numeric(scoring["tue_open_home_spread"], errors="coerce").to_numpy()
                    + scored["residual_at_open"].to_numpy()
                ),
                "result": pd.to_numeric(scoring["result"], errors="coerce").to_numpy(),
            }
        )
        archive_stream = (
            week_stream
            if archive_stream.empty
            else pd.concat([archive_stream, week_stream], ignore_index=True)
        )
    if not scored_weeks:
        raise ValueError("No paired week had at least min_train_games completed training rows")
    residuals = pd.concat(scored_weeks, ignore_index=True)

    result = paired.merge(residuals.drop(columns=["season", "week"]), on="game_id", how="inner")
    result["margin_vs_open"] = result["result"] - result["tue_open_home_spread"]
    result["margin_vs_close"] = result["result"] - result["close_home_spread"]
    result["open_move"] = result["close_home_spread"] - result["tue_open_home_spread"]

    result["pick_home_at_open"] = result["residual_at_open"].gt(0.0)
    result["pick_home_at_close"] = result["residual_at_close"].gt(0.0)
    result["correct_at_open"] = pick_correct(result["pick_home_at_open"], result["margin_vs_open"])
    result["correct_at_close"] = pick_correct(
        result["pick_home_at_close"], result["margin_vs_close"]
    )
    result["pick_home_at_open_probability_rule"] = result["home_cover_probability_at_open"].ge(0.5)
    result["pick_home_at_close_probability_rule"] = result["home_cover_probability_at_close"].ge(
        0.5
    )
    result["correct_at_open_probability_rule"] = pick_correct(
        result["pick_home_at_open_probability_rule"], result["margin_vs_open"]
    )
    result["correct_at_close_probability_rule"] = pick_correct(
        result["pick_home_at_close_probability_rule"], result["margin_vs_close"]
    )
    result["pick_home_at_open_probability_rule_raw"] = result[
        "home_cover_probability_at_open_raw"
    ].ge(0.5)
    result["pick_home_at_close_probability_rule_raw"] = result[
        "home_cover_probability_at_close_raw"
    ].ge(0.5)
    result["correct_at_open_probability_rule_raw"] = pick_correct(
        result["pick_home_at_open_probability_rule_raw"], result["margin_vs_open"]
    )
    result["correct_at_close_probability_rule_raw"] = pick_correct(
        result["pick_home_at_close_probability_rule_raw"], result["margin_vs_close"]
    )
    oracle_correct = pick_correct(result["open_move"].gt(0.0), result["margin_vs_open"])
    result["oracle_correct_at_open"] = oracle_correct.where(result["open_move"].ne(0.0))
    scored_result = result.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    if result_cache is not None:
        _write_cached_frame(scored_result, result_cache)
    return scored_result


OPENER_EVALUATION_METRIC_COLUMNS: tuple[str, ...] = (
    "correct_at_open",
    "correct_at_close",
    "oracle_correct_at_open",
    "correct_at_open_probability_rule",
    "correct_at_close_probability_rule",
    "correct_at_open_probability_rule_raw",
    "correct_at_close_probability_rule_raw",
)


def opener_evaluation_metrics(scored: pd.DataFrame) -> dict[str, float]:

    at_open = pd.to_numeric(scored["correct_at_open"], errors="coerce").dropna()
    at_close = pd.to_numeric(scored["correct_at_close"], errors="coerce").dropna()
    both = scored.dropna(subset=["correct_at_open", "correct_at_close"])
    oracle = pd.to_numeric(scored["oracle_correct_at_open"], errors="coerce").dropna()
    metrics = {
        "opener_accuracy": float(at_open.mean()) if len(at_open) else float("nan"),
        "close_accuracy": float(at_close.mean()) if len(at_close) else float("nan"),
        "opener_minus_close": (
            float(both["correct_at_open"].mean() - both["correct_at_close"].mean())
            if len(both)
            else float("nan")
        ),
        "opener_vs_coin_flip": (float(at_open.mean()) - 0.5) if len(at_open) else float("nan"),
        "movement_oracle_accuracy": float(oracle.mean()) if len(oracle) else float("nan"),
    }
    probability_rule_columns = {
        "correct_at_open_probability_rule",
        "correct_at_close_probability_rule",
    }
    if probability_rule_columns.issubset(scored.columns):
        at_open_pr = pd.to_numeric(scored["correct_at_open_probability_rule"], errors="coerce")
        at_open_pr = at_open_pr.dropna()
        at_close_pr = pd.to_numeric(scored["correct_at_close_probability_rule"], errors="coerce")
        at_close_pr = at_close_pr.dropna()
        both_pr = scored.dropna(
            subset=["correct_at_open_probability_rule", "correct_at_close_probability_rule"]
        )
        metrics.update(
            {
                "opener_accuracy_probability_rule": (
                    float(at_open_pr.mean()) if len(at_open_pr) else float("nan")
                ),
                "close_accuracy_probability_rule": (
                    float(at_close_pr.mean()) if len(at_close_pr) else float("nan")
                ),
                "opener_minus_close_probability_rule": (
                    float(
                        both_pr["correct_at_open_probability_rule"].mean()
                        - both_pr["correct_at_close_probability_rule"].mean()
                    )
                    if len(both_pr)
                    else float("nan")
                ),
                "opener_vs_coin_flip_probability_rule": (
                    (float(at_open_pr.mean()) - 0.5) if len(at_open_pr) else float("nan")
                ),
            }
        )
    raw_columns = {
        "correct_at_open_probability_rule_raw",
        "correct_at_close_probability_rule_raw",
    }
    if raw_columns.issubset(scored.columns):
        at_open_raw = pd.to_numeric(
            scored["correct_at_open_probability_rule_raw"], errors="coerce"
        ).dropna()
        at_close_raw = pd.to_numeric(
            scored["correct_at_close_probability_rule_raw"], errors="coerce"
        ).dropna()
        metrics.update(
            {
                "opener_accuracy_probability_rule_raw": (
                    float(at_open_raw.mean()) if len(at_open_raw) else float("nan")
                ),
                "close_accuracy_probability_rule_raw": (
                    float(at_close_raw.mean()) if len(at_close_raw) else float("nan")
                ),
            }
        )
    return metrics


def _compacted_mean(values: npt.NDArray[np.float64]) -> float:
    if values.size == 0:
        return float("nan")
    return float(values.sum() / values.size)


def opener_evaluation_metric_draws(
    scored: pd.DataFrame,
) -> Callable[[Any], dict[str, float]] | None:
    names = ("correct_at_open", "correct_at_close", "oracle_correct_at_open")
    probability_names = ("correct_at_open_probability_rule", "correct_at_close_probability_rule")
    raw_names = ("correct_at_open_probability_rule_raw", "correct_at_close_probability_rule_raw")
    available = set(scored.columns)
    if not set(names).issubset(available):
        return None
    wanted = [*names]
    with_probability = set(probability_names).issubset(available)
    with_raw = set(raw_names).issubset(available)
    if with_probability:
        wanted.extend(probability_names)
    if with_raw:
        wanted.extend(raw_names)
    columns: dict[str, npt.NDArray[np.float64]] = {}
    for name in wanted:
        series = scored[name]
        if not pd.api.types.is_float_dtype(series.dtype):
            return None
        columns[name] = series.to_numpy(dtype=float, copy=True)

    def draw(positions: Any) -> dict[str, float]:
        at_open = columns["correct_at_open"][positions]
        at_close = columns["correct_at_close"][positions]
        oracle = columns["oracle_correct_at_open"][positions]
        open_valid = ~np.isnan(at_open)
        close_valid = ~np.isnan(at_close)
        both = open_valid & close_valid
        both_count = int(both.sum())
        open_mean = _compacted_mean(at_open[open_valid])
        metrics = {
            "opener_accuracy": open_mean,
            "close_accuracy": _compacted_mean(at_close[close_valid]),
            "opener_minus_close": (
                float(at_open[both].sum() / both_count - at_close[both].sum() / both_count)
                if both_count
                else float("nan")
            ),
            "opener_vs_coin_flip": open_mean - 0.5 if open_valid.any() else float("nan"),
            "movement_oracle_accuracy": _compacted_mean(oracle[~np.isnan(oracle)]),
        }
        if with_probability:
            open_pr = columns["correct_at_open_probability_rule"][positions]
            close_pr = columns["correct_at_close_probability_rule"][positions]
            open_pr_valid = ~np.isnan(open_pr)
            close_pr_valid = ~np.isnan(close_pr)
            both_pr = open_pr_valid & close_pr_valid
            both_pr_count = int(both_pr.sum())
            open_pr_mean = _compacted_mean(open_pr[open_pr_valid])
            metrics.update(
                {
                    "opener_accuracy_probability_rule": open_pr_mean,
                    "close_accuracy_probability_rule": _compacted_mean(close_pr[close_pr_valid]),
                    "opener_minus_close_probability_rule": (
                        float(
                            open_pr[both_pr].sum() / both_pr_count
                            - close_pr[both_pr].sum() / both_pr_count
                        )
                        if both_pr_count
                        else float("nan")
                    ),
                    "opener_vs_coin_flip_probability_rule": (
                        open_pr_mean - 0.5 if open_pr_valid.any() else float("nan")
                    ),
                }
            )
        if with_raw:
            open_raw = columns["correct_at_open_probability_rule_raw"][positions]
            close_raw = columns["correct_at_close_probability_rule_raw"][positions]
            metrics.update(
                {
                    "opener_accuracy_probability_rule_raw": _compacted_mean(
                        open_raw[~np.isnan(open_raw)]
                    ),
                    "close_accuracy_probability_rule_raw": _compacted_mean(
                        close_raw[~np.isnan(close_raw)]
                    ),
                }
            )
        return metrics

    return draw


def opener_evaluation_home_side_offset_summary(
    scored: pd.DataFrame, *, served: bool
) -> dict[str, Any]:

    offsets = pd.to_numeric(
        scored.get("home_side_offset_at_open", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0)
    changed = 0
    if {"pick_home_at_open_probability_rule", "pick_home_at_open_probability_rule_raw"}.issubset(
        scored.columns
    ):
        changed = int(
            scored["pick_home_at_open_probability_rule"]
            .astype(bool)
            .ne(scored["pick_home_at_open_probability_rule_raw"].astype(bool))
            .sum()
        )
    return {
        "policy": HOME_SIDE_OFFSET_POLICY,
        "served": bool(served),
        "prior_weight_games": PRIOR_WEIGHT_GAMES,
        "trailing_seasons": TRAILING_SEASONS,
        "fitted_from": "walk-forward on this evaluation's own raw out-of-time opener points",
        "games_with_nonzero_offset": int(offsets.ne(0.0).sum()),
        "mean_absolute_offset": float(offsets.abs().mean()) if len(offsets) else 0.0,
        "opener_picks_changed_by_offset": changed,
    }
