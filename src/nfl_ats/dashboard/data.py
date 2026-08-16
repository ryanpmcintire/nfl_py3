"""Read-only, cached data access for the dashboard.

Every loader here is defensive: artifacts may not exist yet (a fresh clone,
preseason, or an experiment nobody has run), and artifact *schemas* may gain
columns from other in-flight work (for example, cover/push/loss splits or a
line-sweep curve). Loaders never raise on a missing file or a missing
optional column -- they return ``None``/empty results instead, and pages
render an explicit "not available yet" state. The dashboard never writes
through these paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.pool import build_ats_pool_card
from nfl_ats.reporting import artifact_directories, read_json
from nfl_ats.snapshots import describe_snapshot, latest_snapshot

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------


def data_root() -> Path:
    return Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))


def artifacts_root() -> Path:
    return Path(os.environ.get("NFL_ATS_ARTIFACTS_DIR", "artifacts"))


def read_json_safe(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (ValueError, OSError):
        return None


def artifact_time(path: Path) -> str:
    """Best-effort human timestamp parsed from a run-id-suffixed directory name."""

    token = path.name.rsplit("-", maxsplit=1)[-1]
    try:
        instant = datetime.strptime(token, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return path.name
    return instant.strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Active model + weekly forecast
# ---------------------------------------------------------------------------


def load_active_model(root: Path) -> dict[str, Any] | None:
    """Load the active-model manifest, treating any invalid manifest as absent."""

    try:
        return load_active_ats_model(root)
    except ValueError:
        return None


@dataclass(frozen=True)
class WeeklyForecast:
    directory: Path
    recommendations: pd.DataFrame
    metadata: dict[str, Any]
    is_active: bool


def load_weekly_ats_forecast(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the named ATS method from either outcome or legacy direct cards."""

    metadata = read_json_safe(path / "metadata.json") or {}
    recommendations_path = path / "recommendations.csv"
    if recommendations_path.is_file():
        recommendations = pd.read_csv(recommendations_path)
        metadata.setdefault("ats_method", "direct_ats")
        return recommendations, metadata

    predictions_path = path / "predictions.csv"
    if not predictions_path.is_file():
        raise ValueError(f"Forecast {path.name} has no ATS prediction table")
    predictions = pd.read_csv(predictions_path)
    ats_method = str(metadata.get("ats_method", "market_residual"))
    recommendations = predictions.loc[predictions["method"].eq(ats_method)].copy()
    if recommendations.empty:
        raise ValueError(f"Forecast {path.name} has no rows for ATS method {ats_method!r}")
    metadata["ats_method"] = ats_method
    return recommendations, metadata


def latest_weekly_forecast(root: Path) -> WeeklyForecast | None:
    """The forecast the home page should show: the active one, else the newest saved card."""

    active = load_active_model(root)
    directories = artifact_directories(root / "margin_predictions", "predictions.csv")
    if not directories:
        directories = artifact_directories(root / "predictions", "recommendations.csv")
    if not directories:
        return None
    active_path = active_artifact_path(root, active, "weekly_forecast") if active else None
    if active_path is not None:
        directories = sorted(directories, key=lambda path: path.resolve() != active_path.resolve())
    chosen = directories[0]
    try:
        recommendations, metadata = load_weekly_ats_forecast(chosen)
    except ValueError:
        return None
    is_active = active_path is not None and chosen.resolve() == active_path.resolve()
    return WeeklyForecast(chosen, recommendations, metadata, is_active)


def list_weekly_forecasts(root: Path) -> list[Path]:
    directories = artifact_directories(root / "margin_predictions", "predictions.csv")
    if not directories:
        directories = artifact_directories(root / "predictions", "recommendations.csv")
    return directories


def load_named_weekly_forecast(path: Path) -> WeeklyForecast | None:
    try:
        recommendations, metadata = load_weekly_ats_forecast(path)
    except ValueError:
        return None
    return WeeklyForecast(path, recommendations, metadata, False)


def pool_card(recommendations: pd.DataFrame, directory: Path) -> pd.DataFrame:
    saved = directory / "pool_card.csv"
    if saved.is_file():
        try:
            return pd.read_csv(saved)
        except (ValueError, OSError):
            pass
    try:
        return build_ats_pool_card(recommendations)
    except ValueError:
        return pd.DataFrame()


# --- Optional / newer-schema detection (feature-detect, never assume) ------

OPTIONAL_SPLIT_COLUMNS: dict[str, tuple[str, ...]] = {
    "cover": ("home_cover_probability_at_line", "cover_probability"),
    "push": ("home_push_probability", "push_probability"),
    "loss": ("home_loss_probability", "loss_probability"),
}


def detect_cover_push_loss(recommendations: pd.DataFrame) -> dict[str, str]:
    """Return {kind: column_name} for any cover/push/loss split columns present."""

    found: dict[str, str] = {}
    for kind, candidates in OPTIONAL_SPLIT_COLUMNS.items():
        for candidate in candidates:
            if candidate in recommendations.columns:
                found[kind] = candidate
                break
    return found


def find_line_sweep_file(directory: Path) -> Path | None:
    """A per-game line-sweep curve file, if a parallel build has started writing one."""

    for pattern in ("*line_sweep*.csv", "*line_sweep*.parquet"):
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


@st.cache_data(show_spinner=False, ttl="10m")
def load_line_sweep(path: Path) -> pd.DataFrame:
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except (ValueError, OSError):
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Line journey: opening/latest live quotes + predicted close (all optional)
# ---------------------------------------------------------------------------


def game_opening_lines(quotes: pd.DataFrame) -> pd.DataFrame:
    """The earliest captured consensus home spread per game.

    Deliberately distinct from ``market_data.tuesday_opener_quotes``, which
    only counts a quote observed on a Tuesday -- too narrow here, since a
    delayed or resumed capture archive may have no Tuesday quote at all and
    this only needs "whatever was captured first." Takes literally the
    earliest ``observed_at_utc`` snapshot for each game and reports its
    cross-book median home spread, mirroring how
    ``market_data.spread_consensus`` reports the latest one.
    """

    required = {"nflverse_game_id", "observed_at_utc", "market", "outcome_side", "home_spread_line"}
    missing = sorted(required.difference(quotes.columns))
    if missing:
        raise ValueError(f"Quote history is missing columns: {', '.join(missing)}")
    spreads = quotes.loc[quotes["market"].eq("spreads") & quotes["outcome_side"].eq("HOME")].copy()
    if spreads.empty:
        return pd.DataFrame(columns=["nflverse_game_id", "opener_home_spread"])
    spreads["observed_at_utc"] = pd.to_datetime(spreads["observed_at_utc"], utc=True)
    earliest_per_game = spreads.groupby("nflverse_game_id")["observed_at_utc"].transform("min")
    opener_quotes = spreads.loc[spreads["observed_at_utc"].eq(earliest_per_game)]
    return opener_quotes.groupby("nflverse_game_id", as_index=False).agg(
        opener_home_spread=("home_spread_line", "median")
    )


def find_latest_close_predictions(root: Path) -> Path | None:
    """The most recent predicted-close artifact, if that pilot model has run.

    Expected to always be absent until the frozen MKT-06 close-prediction
    pilot has a trained model (blocked on the 2020-2022 line archive
    re-fetch) -- feature-detected so the page never assumes it exists.
    """

    directories = artifact_directories(root / "close_predictions", "predictions.parquet")
    return directories[0] if directories else None


@st.cache_data(show_spinner=False, ttl="10m")
def load_close_predictions(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path / "predictions.parquet")
    except (ValueError, OSError):
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Historical evaluation (track record)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl="10m")
def load_evaluation_predictions(evaluation_directory: Path) -> pd.DataFrame:
    path = evaluation_directory / "predictions.parquet"
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except (ValueError, OSError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl="10m")
def load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (ValueError, OSError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl="10m")
def load_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except (ValueError, OSError):
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Market / live-line archive
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketSnapshotInfo:
    snapshot_id: str
    directory: Path
    observed_at: pd.Timestamp


def _parse_snapshot_timestamp(name: str) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(datetime.strptime(name, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC))
    except ValueError:
        return None


def list_recent_market_snapshots(root: Path, *, since_days: int = 10) -> list[MarketSnapshotInfo]:
    """Cheaply find recent snapshot directories by parsing their name -- no file reads.

    The live capture archive and the 2020-2025 historical-backfill archive share
    one directory layout, so this filters by recency first (directory names are
    UTC timestamps) to avoid touching hundreds of historical-backfill snapshots
    on every page load.
    """

    if not root.is_dir():
        return []
    cutoff = pd.Timestamp.now(tz=UTC) - pd.Timedelta(days=since_days)
    found = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        observed_at = _parse_snapshot_timestamp(entry.name)
        if observed_at is None or observed_at < cutoff:
            continue
        found.append(MarketSnapshotInfo(entry.name, entry, observed_at))
    return sorted(found, key=lambda info: info.observed_at)


@st.cache_data(show_spinner=False, ttl="5m")
def load_live_quotes(snapshot_directories: tuple[Path, ...]) -> pd.DataFrame:
    """Load quotes for the given snapshot directories, dropping historical-backfill rows.

    ``capture_kind`` may be absent entirely (older live captures never carried
    the column) or present with values other than ``historical_backfill``;
    both cases are treated as live.
    """

    frames = []
    for directory in snapshot_directories:
        path = directory / "quotes.parquet"
        if not path.is_file():
            continue
        try:
            quotes = pd.read_parquet(path)
        except (ValueError, OSError):
            continue
        if "capture_kind" in quotes.columns:
            quotes = quotes.loc[quotes["capture_kind"].ne("historical_backfill")]
        frames.append(quotes)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Data / feature-table health
# ---------------------------------------------------------------------------


def data_summary(root: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    snapshot_metadata: dict[str, Any] | None = None
    feature_metadata: dict[str, Any] | None = None
    try:
        snapshot = latest_snapshot(root / "raw")
        snapshot_metadata = describe_snapshot(snapshot)
    except FileNotFoundError:
        pass
    feature_manifest = root / "processed" / "game_features.manifest.json"
    if feature_manifest.is_file():
        feature_metadata = read_json_safe(feature_manifest)
    return snapshot_metadata, feature_metadata
