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
from nfl_ats.best_pick_nomination import NOMINATION_V2_ENABLED, NominationV2Result, nominate_v2
from nfl_ats.card_view import resolve_nomination as _card_view_resolve_nomination
from nfl_ats.card_view import resolve_overlay as _card_view_resolve_overlay
from nfl_ats.card_view import (
    resolve_player_arrests_overlay as _card_view_resolve_player_arrests_overlay,
)
from nfl_ats.card_view import v2_nomination_inputs
from nfl_ats.coach_fade_overlay import OverlayResult
from nfl_ats.data import DataContractError
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


# ---------------------------------------------------------------------------
# Coach-fade overlay + Best Pick nomination -- what actually gets submitted
# ---------------------------------------------------------------------------
#
# The weekly forecast artifact (``recommendations.csv``, everything above
# reads it as ``forecast.recommendations``) holds the model's OWN, un-overlaid
# picks. Two levers change what actually gets submitted to the pool without
# ever rewriting that file:
#
# 1. The year-1-coach-fade overlay (``nfl_ats.coach_fade_overlay``) flips a
#    handful of picks post-prediction (weeks 1-8, "clean case" only).
# 2. Best Pick nomination v2 (``nfl_ats.best_pick_nomination``) usually
#    replaces the incumbent ``sweep_robustness`` signal (v1) for choosing
#    WHICH game gets the week's bonus pick.
#
# ``nfl_ats.publishing.publish_active_predictions`` applies both before
# writing ``CURRENT_PREDICTIONS.md`` -- the actual submitted card. A page
# that reads ``recommendations`` raw is therefore showing a DIFFERENT pick
# than the one already published, for any flipped or v2-renominated game
# (confirmed live, 2026 Week 1: BAL at IND is BAL un-overlaid, IND on the
# published card). :func:`display_overlay` and :func:`resolve_best_pick`
# are thin wrappers over :mod:`nfl_ats.card_view` -- the ONE shared
# implementation of "what actually gets submitted," also used by
# ``publishing.py`` and ``public_board.py`` -- so every surface agrees with
# what was actually submitted, the same "one anchor for current" contract
# :mod:`nfl_ats.dashboard.state` enforces for artifact staleness. This used
# to be a second, hand-duplicated copy of ``publishing``'s private overlay/
# nomination functions; that duplication is exactly the mirror-drift
# ``nfl_ats.card_view``'s module docstring warns about, and is gone now --
# only the Streamlit-specific caching below stays dashboard-local.


def display_overlay(recommendations: pd.DataFrame, data_root: Path) -> OverlayResult:
    """Both production side policies when the local arrest source is fresh.

    The return type remains ``OverlayResult`` for the dashboard's established
    adapter surface; its prediction frame is the final coach-then-arrest card.
    The interactive dashboard remains diagnostic and fail-open; tracked
    Markdown/HTML publication is the release path and is fail-closed.
    """

    coach = _card_view_resolve_overlay(recommendations, data_root)
    arrest = _card_view_resolve_player_arrests_overlay(
        coach.overlaid_predictions,
        data_root,
        require_fresh=False,
    )
    return OverlayResult(
        arrest.overlaid_predictions,
        coach.flips,
        coach.both_year_one_games,
        coach.week_max,
        coach.enabled,
    )


@dataclass(frozen=True)
class BestPickNomination:
    """Which game the dashboard shows as this week's Best Pick.

    A thin adapter over :class:`nfl_ats.card_view.BestPickNomination`: same
    field NAMES this page has always used (``game_id``, ``tie_note``), so
    ``workbench.py`` keeps reading it unchanged, while the actual v1-vs-v2
    decision and note text live once in :mod:`nfl_ats.card_view`.
    """

    active_rule: str  # "v1" | "v2"
    game_id: str | None
    tie_note: str
    method_note: str


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner="Ranking this week's Best Pick candidates...", ttl="15m")
def _cached_nominate_v2(
    game_ids: tuple[str, ...],
    feature_table: str,
    feature_table_mtime: float,
    market_root: str,
    season: int,
    week: int,
    regressor: str,
    feature_profile: Any,
    min_train_games: int,
) -> NominationV2Result | None:
    """``nominate_v2``, cached: its own walk-forward fit is cheap, but the
    cross-book dispersion pool it joins against
    (``nfl_ats.best_pick_nomination.week_dispersion_pool``) scans the FULL
    local market-quote archive with no recency filter -- roughly 40 seconds
    against the current 2020-2025 archive, measured live. Uncached, that cost
    is paid on every Streamlit rerun (every widget interaction, not just page
    loads). ``feature_table_mtime`` is a cache-key-only staleness signal, the
    same mtime-keyed idiom :mod:`nfl_ats.dashboard.state` uses for parquet
    reads -- a rebuilt feature table invalidates the cache even within the
    TTL. ``game_ids`` stands in for the full predictions frame (cheaper to
    hash, and this rule reads nothing else from it once REG-only eligibility
    is already checked by the caller). This is the one piece of v2 machinery
    that stays dashboard-local -- it is caching, not the nomination decision
    itself, which lives in :mod:`nfl_ats.card_view`.
    """

    del feature_table_mtime  # cache key only
    try:
        features = pd.read_parquet(feature_table)
        return nominate_v2(
            pd.DataFrame({"game_id": list(game_ids)}),
            features,
            market_root=Path(market_root),
            season=season,
            week=week,
            regressor=regressor,
            feature_profile=feature_profile,
            min_train_games=min_train_games,
        )
    except (OSError, ValueError, DataContractError):
        return None


def resolve_best_pick(
    recommendations: pd.DataFrame,
    sweep: pd.DataFrame,
    metadata: dict[str, Any],
    data_root: Path,
) -> BestPickNomination:
    """This week's Best Pick nomination, exactly as ``publish-predictions`` would choose it.

    The v1-vs-v2 decision and disclosure text live once in
    :func:`nfl_ats.card_view.resolve_nomination`; this wrapper only supplies
    v2's result from the cached dispersion-pool scan above (or ``None`` when
    v2 cannot run this week), matching
    :func:`nfl_ats.card_view.compute_v2_nomination`'s own inputs exactly.
    """

    v2_result = None
    if NOMINATION_V2_ENABLED:
        inputs = v2_nomination_inputs(metadata, data_root)
        if inputs is not None:
            v2_result = _cached_nominate_v2(
                tuple(recommendations["game_id"].astype(str)),
                inputs.feature_table,
                _file_mtime(Path(inputs.feature_table)),
                inputs.market_root,
                inputs.season,
                inputs.week,
                inputs.regressor,
                inputs.feature_profile,
                inputs.min_train_games,
            )
    resolved = _card_view_resolve_nomination(
        recommendations, sweep, metadata, data_root, v2_result=v2_result
    )
    return BestPickNomination(
        active_rule=resolved.active_rule,
        game_id=resolved.active_game_id,
        tie_note=resolved.active_tie_note,
        method_note=resolved.method_note,
    )


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
# Market-decomposition explanations (optional, feature-detected)
# ---------------------------------------------------------------------------


def find_latest_market_decomposition(root: Path) -> Path | None:
    """The most recent market-decomposition run's directory, if any.

    ``nfl-ats market-decomposition`` is optional and manual; feature-detected
    so a fresh clone or a season nobody has run it for yet renders an empty
    state instead of raising. Keyed on ``metadata.json`` -- the file present
    even when a run skipped attribution (``--no-attribution``) -- rather than
    ``attribution.parquet``, so the family-weight tables stay available on
    their own.
    """

    directories = artifact_directories(root / "market_decomposition", "metadata.json")
    return directories[0] if directories else None


# ---------------------------------------------------------------------------
# Predicted close (optional, feature-detected)
# ---------------------------------------------------------------------------


def find_latest_close_predictions(root: Path) -> Path | None:
    """The most recent predicted-close artifact, if ``predict-close`` has run.

    ``nfl-ats predict-close`` (the MKT-06 wiring) only writes this after the
    target week's live Tuesday opener capture exists, so it is legitimately
    absent for most of any week -- feature-detected so the page never assumes
    it exists.
    """

    directories = artifact_directories(root / "close_predictions", "predictions.parquet")
    return directories[0] if directories else None


@st.cache_data(show_spinner=False, ttl="10m")
def load_close_predictions(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path / "predictions.parquet")
    except (ValueError, OSError):
        return pd.DataFrame()


def find_latest_clv_ledger(root: Path) -> Path | None:
    """The most recent scored paper-decision CLV artifact, if the ledger has run.

    ``nfl-ats clv-ledger`` (the MKT-04 wiring) scores every published card's
    pre-kickoff picks against the close; before the first published week it is
    legitimately absent -- feature-detected so the page never assumes it exists.
    """

    directories = artifact_directories(root / "clv_ledger", "scored_decisions.parquet")
    return directories[0] if directories else None


@st.cache_data(show_spinner=False, ttl="10m")
def load_clv_ledger(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path / "scored_decisions.parquet")
    except (ValueError, OSError):
        return pd.DataFrame()


def find_latest_opener_evaluation(root: Path) -> Path | None:
    """The most recent opener-graded evaluation, if `opener-evaluation` has run.

    The primary pool goal is graded against Tuesday openers
    (docs/pool_edge_plan.md); this artifact carries that measurement.
    Feature-detected so the page never assumes it exists.
    """

    directories = artifact_directories(root / "opener_evaluation", "per_game.parquet")
    return directories[0] if directories else None


@st.cache_data(show_spinner=False, ttl="10m")
def load_opener_evaluation(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """The opener evaluation's metadata (metrics/uncertainty) and season table."""

    try:
        metadata = read_json(path / "metadata.json")
    except (OSError, ValueError):
        metadata = {}
    try:
        seasons = pd.read_csv(path / "season_summary.csv")
    except (OSError, ValueError):
        seasons = pd.DataFrame()
    return metadata, seasons


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
# Pool-format simulation (POL-05, optional, feature-detected)
# ---------------------------------------------------------------------------


def load_pool_levers(root: Path) -> dict[str, Any] | None:
    """POL-05's precomputed field-size/prize-structure simulation, if it has run.

    ``scripts/pool_levers.py`` is a batch job (its own docstring: pure
    arithmetic conditional on an accuracy measured elsewhere, never re-run
    inside the Streamlit process) that writes
    ``artifacts/pool_levers/levers.json``. Feature-detected like every other
    optional artifact here: ``None`` when nobody has run it yet, never an
    exception.
    """

    return read_json_safe(root / "pool_levers" / "levers.json")


# ---------------------------------------------------------------------------
# Data / feature-table health
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactSelection:
    """Which saved run a research page should feature, vs. list as older.

    The dashboard owner's complaint this exists to fix: research pages used to
    pick "the latest artifact of their own type" (newest directory name) with no
    regard for which one is actually linked to the active model, so a page could
    silently show a different run than the one backing "This week's picks." See
    :func:`select_research_artifact`.
    """

    featured: Path | None
    featured_is_active: bool
    active_declared_but_missing: bool
    older: tuple[Path, ...]


def select_research_artifact(
    directories: list[Path], active_path: Path | None = None
) -> ArtifactSelection:
    """Partition a research artifact type's saved runs into "featured" + "older".

    When ``active_path`` is given (the active model's manifest declares one for
    this artifact type -- e.g. ``historical_evaluation.artifact`` for the
    ``margins`` directory), the run it points at is always featured first, never
    displaced by a newer-but-unpromoted run, and every other saved run is
    "older". If the manifest declares one but it is not among ``directories``
    (missing on disk, or missing its required file), ``featured`` is ``None`` and
    ``active_declared_but_missing`` is ``True`` -- callers must say so explicitly
    rather than silently falling back to some other run (see the docstring on
    ``active_ats_model.json``'s consumers).

    When ``active_path`` is ``None`` (this artifact type has no manifest linkage
    at all -- most research/experiment archives), the newest directory (by name,
    which is how :func:`nfl_ats.reporting.artifact_directories` already sorts) is
    featured as "the latest run" and the rest are "older": there is no
    "active model" concept for a standalone experiment archive, so recency is the
    correct, non-stale ordering.
    """

    if active_path is not None:
        resolved_active = active_path.resolve()
        match = next((path for path in directories if path.resolve() == resolved_active), None)
        if match is None:
            # Also covers `directories` being empty entirely -- an active model
            # that declares an artifact with zero saved runs on disk is exactly
            # the "missing" case, not "no active-model concept."
            return ArtifactSelection(None, False, True, tuple(directories))
        older = tuple(path for path in directories if path.resolve() != resolved_active)
        return ArtifactSelection(match, True, False, older)
    if not directories:
        return ArtifactSelection(None, False, False, ())
    return ArtifactSelection(directories[0], False, False, tuple(directories[1:]))


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
