"""Prospective recorder for the four-overlay policy's immediate incumbent."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.clv import load_paper_decisions, refuse_if_outside_recording_lock_window
from nfl_ats.data import DataContractError
from nfl_ats.four_overlay_composition import INCUMBENT_CHALLENGER_ID, POLICY_ID
from nfl_ats.io import atomic_parquet
from nfl_ats.prospective_scoring import (
    ACTIVE_CHALLENGER_STATUS,
    CHALLENGER_DECISION_COLUMNS,
    artifact_model_config,
    challenger_ledger_path,
    config_fingerprint,
    find_challenger,
    load_challenger_decisions,
)
from nfl_ats.provenance import sha256_file


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_former_production_incumbent_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record the former coach->arrests side frozen in the primary ledger.

    The primary recorder runs first and freezes both the newly played side and
    ``former_policy_pick_side`` from the exact same resolved card. Reading that
    field here prevents a second source load from producing a subtly different
    paired control.
    """

    del data_root  # the primary paper ledger already froze every required input
    entry = find_challenger(artifacts_root, INCUMBENT_CHALLENGER_ID)
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {INCUMBENT_CHALLENGER_ID!r} is registered as {status!r}; only "
            f"{ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )
    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError("No synchronized active ATS model is available")
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        raise ValueError("Active ATS model has no linked weekly forecast")
    metadata_path = forecast / "metadata.json"
    card_path = forecast / "recommendations.csv"
    if not metadata_path.is_file() or not card_path.is_file():
        raise ValueError(f"Linked weekly forecast is incomplete: {forecast}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observed_config = artifact_model_config(metadata)
    declared = config_fingerprint(entry.get("model", {}))
    observed = config_fingerprint(observed_config)
    if declared != observed:
        raise DataContractError(
            f"Challenger {INCUMBENT_CHALLENGER_ID!r} is pinned to {declared}, but the "
            f"active forecast uses {observed}"
        )

    card = pd.read_csv(card_path)
    recorded_at = _record_instant(now)
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Weekly forecast has games without a kickoff timestamp")
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")

    primary = load_paper_decisions(artifacts_root)
    if primary.empty:
        raise DataContractError("Primary paper ledger has no four-overlay rows to pair")
    season = int(card["season"].iloc[0])
    week = int(card["week"].iloc[0])
    primary = primary.loc[
        primary["season"].astype(int).eq(season)
        & primary["week"].astype(int).eq(week)
        & primary["decision_policy_id"].astype(str).eq(POLICY_ID)
    ].copy()
    if set(primary["game_id"].astype(str)) != set(card["game_id"].astype(str)):
        raise DataContractError(
            "Primary paper ledger does not contain the complete current four-overlay card"
        )
    primary = card[["game_id"]].merge(primary, on="game_id", how="left", validate="one_to_one")

    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[existing["challenger_id"].astype(str).eq(INCUMBENT_CHALLENGER_ID)]
    already = primary["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    pre_kickoff = pd.to_datetime(primary["kickoff"], utc=True).gt(recorded_at)
    fresh = primary.loc[~already & pre_kickoff].copy()
    provenance = metadata.get("provenance") if isinstance(metadata.get("provenance"), dict) else {}
    feature = provenance.get("feature_table") if isinstance(provenance, dict) else {}
    feature = feature if isinstance(feature, dict) else {}
    decisions = pd.DataFrame(
        {
            "recorded_at_utc": recorded_at,
            "challenger_id": INCUMBENT_CHALLENGER_ID,
            "config_fingerprint": observed,
            "source_artifact": forecast.name,
            "source_sha256": sha256_file(card_path),
            "forecast_created_at_utc": pd.to_datetime(
                metadata.get("created_at_utc"), utc=True, errors="coerce"
            ),
            "feature_profile": str(metadata.get("feature_profile")),
            "feature_table_sha256": str(feature.get("sha256", "")),
            "game_id": fresh["game_id"].astype(str),
            "season": fresh["season"].astype(int),
            "week": fresh["week"].astype(int),
            "kickoff": pd.to_datetime(fresh["kickoff"], utc=True),
            "away_team": fresh["away_team"].astype(str),
            "home_team": fresh["home_team"].astype(str),
            "pick_side": fresh["former_policy_pick_side"].astype(str),
            "bet_side": "PASS",
            "decision_home_spread": fresh["decision_home_spread"].astype(float),
            "edge": np.nan,
        }
    )
    if not decisions.empty:
        combined = (
            decisions if existing.empty else pd.concat([existing, decisions], ignore_index=True)
        )
        atomic_parquet(
            combined[list(CHALLENGER_DECISION_COLUMNS)], challenger_ledger_path(artifacts_root)
        )
        ledger_rows = len(combined)
    else:
        ledger_rows = len(existing)
    return {
        "challenger_id": INCUMBENT_CHALLENGER_ID,
        "season": season,
        "week": week,
        "config_fingerprint": observed,
        "recorded": len(decisions),
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "ledger_rows": int(ledger_rows),
    }


__all__ = ["record_former_production_incumbent_decisions"]
