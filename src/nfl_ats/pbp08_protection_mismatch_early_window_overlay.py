from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.pbp08_protection_mismatch_tilt_overlay import (
    TiltFlip,
    TiltResult,
    apply_pbp08_protection_mismatch_tilt,
    flags_for_week_fail_open,
)
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

CHALLENGER_ID = "pbp08_protection_mismatch_early_window_v1"
PREDECLARED_SPLIT_BOUNDARY_WEEK = 4


def apply_pbp08_protection_mismatch_early_window_tilt(
    predictions: pd.DataFrame,
    flags: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:
    tilt = apply_pbp08_protection_mismatch_tilt(predictions, flags, enabled=enabled)
    if not enabled or tilt.flip_count == 0:
        return tilt
    base = predictions.reset_index(drop=True).copy()
    if "week" not in base.columns:
        return TiltResult(base, (), enabled, tilt.flag_summary)
    weeks = pd.to_numeric(base["week"], errors="coerce")
    allowed = set(base.loc[weeks.le(PREDECLARED_SPLIT_BOUNDARY_WEEK), "game_id"].astype(str))
    kept: tuple[TiltFlip, ...] = tuple(flip for flip in tilt.flips if str(flip.game_id) in allowed)
    if len(kept) == tilt.flip_count:
        return tilt
    overlaid = tilt.overlaid_predictions.copy()
    keep_ids = {flip.game_id for flip in kept}
    revert_ids = {flip.game_id for flip in tilt.flips} - keep_ids
    revert = base["game_id"].astype(str).isin(revert_ids)
    overlaid.loc[revert, "home_cover_probability"] = base.loc[revert, "home_cover_probability"]
    return TiltResult(overlaid, kept, enabled, tilt.flag_summary)


def overlay_disclosure_note(result: TiltResult) -> str:
    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.original_pick_team} -> {flip.flipped_to_team}"
        for flip in result.flips
    )
    return (
        f"**Early-window tilt applied: {result.flip_count} pick{plural} flipped** off an offense "
        "whose four-game pressure-allowed rate is top-quartile against a defense generating "
        "pressure at a top-quartile rate. Early window only (weeks 1-4; week 5+ records "
        "the raw-model side). Mechanism: a line-versus-rush matchup that only last season's "
        "tail shows is not yet priced in September; by mid-season pressure rates are public "
        f"and priced. {detail}. See docs/pbp08_matchup_screen.md. "
        "Prospective evidence only -- not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_pbp08_protection_mismatch_early_window_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
    forecast_artifact: str | None = None,
    replace_week: bool = False,
) -> dict[str, Any]:
    entry = find_challenger(artifacts_root, CHALLENGER_ID)
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {CHALLENGER_ID!r} is registered as {status!r}; only "
            f"{ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )
    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError("No synchronized active ATS model is available to record tilt decisions")
    forecast, metadata = resolve_recording_forecast(
        artifacts_root, active, forecast_artifact=forecast_artifact
    )
    card_path = forecast / "recommendations.csv"
    observed_config = artifact_model_config(metadata)
    declared_fingerprint = config_fingerprint(entry.get("model", {}))
    observed_fingerprint = config_fingerprint(observed_config)
    if declared_fingerprint != observed_fingerprint:
        raise DataContractError(
            f"Challenger {CHALLENGER_ID!r} is registered pinned to configuration fingerprint "
            f"{declared_fingerprint}, but the current active forecast {forecast} was produced "
            f"with {observed_fingerprint}; the active model changed underneath this tilt -- "
            "re-register before recording"
        )
    card = pd.read_csv(card_path)
    required = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
    }
    missing = sorted(required.difference(card.columns))
    if missing:
        raise DataContractError(f"Active forecast card is missing columns: {', '.join(missing)}")
    if card["game_id"].duplicated().any():
        raise DataContractError("Active forecast card contains duplicate games")
    spreads = pd.to_numeric(card["spread_line"], errors="coerce")
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Active forecast card has games without a kickoff timestamp")
    season = int(card["season"].iloc[0])
    week = int(card["week"].iloc[0])
    flags = flags_for_week_fail_open(data_root, season=season, week=week)
    tilt = apply_pbp08_protection_mismatch_early_window_tilt(card, flags)
    tilted_card = tilt.overlaid_predictions
    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_challenger_decisions(artifacts_root)
    replaced_rows = 0
    left_post_kickoff = 0
    if replace_week and bool(pre_kickoff.any()):
        existing, replaced_rows, left_post_kickoff = replace_week_rows(
            existing,
            challenger_ledger_path(artifacts_root),
            season=int(card["season"].iloc[0]),
            week=int(card["week"].iloc[0]),
            recorded_at=recorded_at,
            columns=CHALLENGER_DECISION_COLUMNS,
            challenger_id=CHALLENGER_ID,
        )
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID)]
    already = card["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    keep = pre_kickoff & ~already
    fresh = tilted_card.loc[keep]
    decisions = pd.DataFrame(
        {
            "recorded_at_utc": recorded_at,
            "challenger_id": CHALLENGER_ID,
            "config_fingerprint": observed_fingerprint,
            "source_artifact": forecast.name,
            "source_sha256": sha256_file(card_path),
            "forecast_created_at_utc": pd.to_datetime(
                metadata.get("created_at_utc"), utc=True, errors="coerce"
            ),
            "feature_profile": str(metadata.get("feature_profile")),
            "feature_table_sha256": str(observed_config.get("feature_table_sha256")),
            "game_id": fresh["game_id"].astype(str),
            "season": fresh["season"].astype(int),
            "week": fresh["week"].astype(int),
            "kickoff": kickoffs.loc[fresh.index],
            "away_team": fresh["away_team"].astype(str),
            "home_team": fresh["home_team"].astype(str),
            "pick_side": np.where(
                pd.to_numeric(fresh["home_cover_probability"], errors="coerce").ge(0.5),
                "HOME",
                "AWAY",
            ).astype(str),
            "bet_side": "PASS",
            "decision_home_spread": spreads.loc[fresh.index].astype(float),
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
        "challenger_id": CHALLENGER_ID,
        "season": season,
        "week": week,
        "source_artifact": forecast.name,
        "config_fingerprint": observed_fingerprint,
        "recorded": len(decisions),
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "replaced_rows": replaced_rows,
        "left_post_kickoff": left_post_kickoff,
        "ledger_rows": int(ledger_rows),
        "flip_count": tilt.flip_count,
        "flipped_game_ids": [flip.game_id for flip in tilt.flips],
        "flag_summary": tilt.flag_summary,
    }


__all__ = [
    "CHALLENGER_ID",
    "PREDECLARED_SPLIT_BOUNDARY_WEEK",
    "TiltFlip",
    "TiltResult",
    "apply_pbp08_protection_mismatch_early_window_tilt",
    "flags_for_week_fail_open",
    "overlay_disclosure_note",
    "record_pbp08_protection_mismatch_early_window_challenger_decisions",
]
