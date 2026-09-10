from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.clv import (
    COMPOSITION_POLICY_IDS,
    load_paper_decisions,
    refuse_if_outside_recording_lock_window,
)
from nfl_ats.data import DataContractError
from nfl_ats.four_overlay_composition import RETIRED_THREE_MEMBER_CHALLENGER_ID
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
from nfl_ats.recorder_override import replace_week_rows, resolve_recording_forecast


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def retired_three_member_sides(primary: pd.DataFrame) -> pd.Series:

    flip = (
        primary["coach_fade_flip"].astype(bool)
        | primary["division_revenge_flip"].astype(bool)
        | primary["player_arrests_flip"].astype(bool)
    )
    model = primary["model_pick_side"].astype(str)
    return model.where(~flip, model.map({"HOME": "AWAY", "AWAY": "HOME"}))


def record_retired_three_member_union_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
    forecast_artifact: str | None = None,
    replace_week: bool = False,
) -> dict[str, Any]:

    del data_root
    entry = find_challenger(artifacts_root, RETIRED_THREE_MEMBER_CHALLENGER_ID)
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {RETIRED_THREE_MEMBER_CHALLENGER_ID!r} is registered as {status!r}; "
            f"only {ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )
    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError("No synchronized active ATS model is available")
    forecast, metadata = resolve_recording_forecast(
        artifacts_root, active, forecast_artifact=forecast_artifact
    )
    card_path = forecast / "recommendations.csv"
    observed_config = artifact_model_config(metadata)
    declared = config_fingerprint(entry.get("model", {}))
    observed = config_fingerprint(observed_config)
    if declared != observed:
        raise DataContractError(
            f"Challenger {RETIRED_THREE_MEMBER_CHALLENGER_ID!r} is pinned to {declared}, but "
            f"the active forecast uses {observed}"
        )

    card = pd.read_csv(card_path)
    recorded_at = _record_instant(now)
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Weekly forecast has games without a kickoff timestamp")
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")

    primary = load_paper_decisions(artifacts_root)
    if primary.empty:
        raise DataContractError("Primary paper ledger has no composed rows to pair")
    season = int(card["season"].iloc[0])
    week = int(card["week"].iloc[0])
    primary = primary.loc[
        primary["season"].astype(int).eq(season)
        & primary["week"].astype(int).eq(week)
        & primary["decision_policy_id"].astype(str).isin(COMPOSITION_POLICY_IDS)
    ].copy()
    if set(primary["game_id"].astype(str)) != set(card["game_id"].astype(str)):
        raise DataContractError(
            "Primary paper ledger does not contain the complete current composed card"
        )
    primary = card[["game_id"]].merge(primary, on="game_id", how="left", validate="one_to_one")

    existing = load_challenger_decisions(artifacts_root)
    pre_kickoff = pd.to_datetime(primary["kickoff"], utc=True).gt(recorded_at)
    replaced_rows = 0
    left_post_kickoff = 0
    if replace_week and bool(pre_kickoff.any()):
        existing, replaced_rows, left_post_kickoff = replace_week_rows(
            existing,
            challenger_ledger_path(artifacts_root),
            season=season,
            week=week,
            recorded_at=recorded_at,
            columns=CHALLENGER_DECISION_COLUMNS,
            challenger_id=RETIRED_THREE_MEMBER_CHALLENGER_ID,
        )
    mine = existing.loc[
        existing["challenger_id"].astype(str).eq(RETIRED_THREE_MEMBER_CHALLENGER_ID)
    ]
    already = primary["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    fresh = primary.loc[~already & pre_kickoff].copy()
    provenance = metadata.get("provenance") if isinstance(metadata.get("provenance"), dict) else {}
    feature = provenance.get("feature_table") if isinstance(provenance, dict) else {}
    feature = feature if isinstance(feature, dict) else {}
    former_sides = retired_three_member_sides(fresh)
    decisions = pd.DataFrame(
        {
            "recorded_at_utc": recorded_at,
            "challenger_id": RETIRED_THREE_MEMBER_CHALLENGER_ID,
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
            "pick_side": former_sides,
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
        "challenger_id": RETIRED_THREE_MEMBER_CHALLENGER_ID,
        "season": season,
        "week": week,
        "config_fingerprint": observed,
        "recorded": len(decisions),
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "replaced_rows": replaced_rows,
        "left_post_kickoff": left_post_kickoff,
        "ledger_rows": int(ledger_rows),
    }


__all__ = ["record_retired_three_member_union_decisions", "retired_three_member_sides"]
