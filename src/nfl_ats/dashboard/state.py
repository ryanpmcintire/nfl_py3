"""One definition of "current" for every dashboard page.

The dashboard's historical failure mode (diagnosed 2026-08-17): each page
chose its own anchor -- the active-model manifest, "newest directory of my
artifact type," or a direct read of ``data/processed`` -- so pages showed
numbers computed from different pipeline states with no visible warning.

:func:`project_state` computes the anchor ONCE per rerun: the active model,
the forecast it declares, and every optional artifact the pages surface,
each tagged with whether it is consistent with that active model. Pages
never call ``artifact_directories`` directly; they render this object, and
they render its ``issues`` (plain-English staleness statements) instead of
silently disagreeing with each other.

Caching is keyed on file modification times, never TTLs: a rebuild is
visible on the next rerun, and an untouched artifact never re-reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from nfl_ats.dashboard.data import (
    WeeklyForecast,
    artifacts_root,
    find_latest_close_predictions,
    find_latest_clv_ledger,
    find_latest_market_decomposition,
    find_latest_opener_evaluation,
    latest_weekly_forecast,
    load_active_model,
    read_json_safe,
)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _read_parquet_mtime(path: str, mtime: float) -> pd.DataFrame:
    del mtime  # cache key only
    try:
        return pd.read_parquet(path)
    except (ValueError, OSError):
        return pd.DataFrame()


def load_parquet(path: Path) -> pd.DataFrame:
    """Modification-time-keyed parquet read: fresh after any rebuild."""

    return _read_parquet_mtime(str(path), _mtime(path))


@st.cache_data(show_spinner=False)
def _read_csv_mtime(path: str, mtime: float) -> pd.DataFrame:
    del mtime
    try:
        return pd.read_csv(path)
    except (ValueError, OSError):
        return pd.DataFrame()


def load_csv(path: Path) -> pd.DataFrame:
    return _read_csv_mtime(str(path), _mtime(path))


@dataclass(frozen=True)
class ArtifactState:
    """One optional artifact the dashboard surfaces, with its provenance."""

    directory: Path | None
    metadata: dict[str, Any] = field(default_factory=dict)
    consistent_with_active: bool | None = None  # None = no linkage to check


@dataclass(frozen=True)
class ProjectState:
    active: dict[str, Any] | None
    forecast: WeeklyForecast | None
    opener_evaluation: ArtifactState
    clv_ledger: ArtifactState
    close_predictions: ArtifactState
    market_decomposition: ArtifactState
    issues: tuple[str, ...]

    @property
    def synchronized(self) -> bool:
        return self.active is not None and self.forecast is not None and self.forecast.is_active


def _active_feature_sha(active: dict[str, Any] | None) -> str | None:
    if not active:
        return None
    value = active.get("feature_table_sha256")
    return str(value) if value else None


def _opener_state(root: Path, active: dict[str, Any] | None) -> tuple[ArtifactState, list[str]]:
    directory = find_latest_opener_evaluation(root)
    if directory is None:
        return ArtifactState(None), []
    metadata = read_json_safe(directory / "metadata.json") or {}
    recorded_sha = (
        metadata.get("provenance", {}).get("feature_table", {}).get("sha256")
        if isinstance(metadata.get("provenance"), dict)
        else None
    )
    active_sha = _active_feature_sha(active)
    issues: list[str] = []
    consistent: bool | None = None
    if recorded_sha and active_sha:
        consistent = recorded_sha == active_sha
        if not consistent:
            issues.append(
                "The opening-line grade was measured on an earlier build of the "
                "model's inputs than the one behind this week's picks. Re-run "
                "`nfl-ats opener-evaluation` to refresh it."
            )
    return ArtifactState(directory, metadata, consistent), issues


def _market_decomposition_state(
    root: Path, active: dict[str, Any] | None
) -> tuple[ArtifactState, list[str]]:
    """Mirrors :func:`_opener_state` verbatim: same sha256 comparison, same shape.

    The model-explanation page's family weights and per-game attribution are
    only trustworthy against the *current* build of the model's inputs -- a
    stray research run must never quietly render as this week's explanation
    (the same class of bug ``opener_evaluation`` had before this check
    existed). ``market-decomposition``'s own metadata does not record which
    active model it targets, but its ``provenance.feature_table.sha256`` is a
    strictly finer-grained check than a profile-name comparison would be.
    """

    directory = find_latest_market_decomposition(root)
    if directory is None:
        return ArtifactState(None), []
    metadata = read_json_safe(directory / "metadata.json") or {}
    recorded_sha = (
        metadata.get("provenance", {}).get("feature_table", {}).get("sha256")
        if isinstance(metadata.get("provenance"), dict)
        else None
    )
    active_sha = _active_feature_sha(active)
    issues: list[str] = []
    consistent: bool | None = None
    if recorded_sha and active_sha:
        consistent = recorded_sha == active_sha
        if not consistent:
            issues.append(
                "The model-explanation breakdown was measured on an earlier build of "
                "the model's inputs than the one behind this week's picks. Re-run "
                "`nfl-ats market-decomposition` to refresh it."
            )
    return ArtifactState(directory, metadata, consistent), issues


def project_state() -> ProjectState:
    root = artifacts_root()
    active = load_active_model(root)
    forecast = latest_weekly_forecast(root)
    issues: list[str] = []

    if active is None:
        issues.append(
            "No synchronized active model: run `nfl-ats margin-backtest` and "
            "`nfl-ats margin-predict` to activate one."
        )
    elif forecast is not None and not forecast.is_active:
        issues.append(
            "This pick card is not the active model's card (the model's inputs "
            "changed after it was made). Re-run `nfl-ats margin-predict` to "
            "regenerate and re-activate."
        )

    opener, opener_issues = _opener_state(root, active)
    issues.extend(opener_issues)

    decomposition, decomposition_issues = _market_decomposition_state(root, active)
    issues.extend(decomposition_issues)

    ledger_directory = find_latest_clv_ledger(root)
    close_directory = find_latest_close_predictions(root)

    return ProjectState(
        active=active,
        forecast=forecast,
        opener_evaluation=opener,
        clv_ledger=ArtifactState(ledger_directory),
        close_predictions=ArtifactState(close_directory),
        market_decomposition=decomposition,
        issues=tuple(issues),
    )
