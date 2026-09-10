from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.data import DataContractError
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
from nfl_ats.surgical_gating import VALUE_LOST_DIFF_COLUMNS

CHALLENGER_ID = "injury_value_lost_tilt_overlay"

PLAYER_FEATURE_TABLE_NAME = "game_features_player.parquet"


def raw_value_lost_diff(features: pd.DataFrame) -> pd.DataFrame:

    required = {"game_id", *VALUE_LOST_DIFF_COLUMNS}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"features is missing value-lost columns: {', '.join(missing)}")

    total = pd.Series(0.0, index=features.index, dtype=float)
    for column in VALUE_LOST_DIFF_COLUMNS:
        total = total + pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    return pd.DataFrame({"game_id": features["game_id"].astype(str), "value_lost_diff": total})


@dataclass(frozen=True)
class TiltFlip:
    game_id: str
    matchup: str
    hurt_team: str
    healthier_team: str
    value_lost_diff: float


@dataclass(frozen=True)
class TiltResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_injury_value_tilt_overlay(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:

    required = {"game_id", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return TiltResult(base, (), enabled)

    diff = raw_value_lost_diff(features)
    merged = base.merge(diff, on="game_id", how="left", validate="one_to_one")
    merged["value_lost_diff"] = merged["value_lost_diff"].fillna(0.0)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = merged["home_cover_probability"].ge(0.5)
    home_lost_more = merged["value_lost_diff"].gt(0.0)
    away_lost_more = merged["value_lost_diff"].lt(0.0)

    flip_mask = eligible & ((home_pick & home_lost_more) | (~home_pick & away_lost_more))

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        row_home_pick = bool(row["home_cover_probability"] >= 0.5)
        hurt_team = str(row["home_team"] if row_home_pick else row["away_team"])
        healthier_team = str(row["away_team"] if row_home_pick else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                hurt_team=hurt_team,
                healthier_team=healthier_team,
                value_lost_diff=float(row["value_lost_diff"]),
            )
        )

    return TiltResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: TiltResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.hurt_team} -> {flip.healthier_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the injury "
        "value-lost tilt (the model's pick sat on the side that lost strictly more "
        f"injury value than its opponent). {detail}. See "
        "docs/injury_value_lost_tilt_overlay.md. Prospective evidence only -- "
        "not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_injury_value_tilt_challenger_decisions(
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
        raise ValueError(
            "No synchronized active ATS model is available to record tilt decisions from"
        )
    forecast, metadata = resolve_recording_forecast(
        artifacts_root, active, forecast_artifact=forecast_artifact
    )
    card_path = forecast / "recommendations.csv"

    observed_config = artifact_model_config(metadata)
    declared_fingerprint = config_fingerprint(entry.get("model", {}))
    observed_fingerprint = config_fingerprint(observed_config)
    if declared_fingerprint != observed_fingerprint:
        raise DataContractError(
            f"Challenger {CHALLENGER_ID!r} is registered pinned to configuration "
            f"fingerprint {declared_fingerprint}, but the current active forecast "
            f"{forecast} was produced with {observed_fingerprint}; the active model "
            "changed underneath this tilt -- re-register before recording"
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
    if not np.isfinite(spreads.to_numpy(dtype=float)).all():
        raise DataContractError("Active forecast card has games without a decision spread")
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Active forecast card has games without a kickoff timestamp")

    feature_path = data_root / "processed" / PLAYER_FEATURE_TABLE_NAME
    if not feature_path.is_file():
        raise ValueError(f"Player feature table is not built yet: {feature_path}")
    features = pd.read_parquet(feature_path, columns=["game_id", *VALUE_LOST_DIFF_COLUMNS])

    tilt = apply_injury_value_tilt_overlay(card, features)
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
        "season": int(card["season"].iloc[0]),
        "week": int(card["week"].iloc[0]),
        "source_artifact": forecast.name,
        "config_fingerprint": observed_fingerprint,
        "recorded": len(decisions),
        "replaced_rows": replaced_rows,
        "left_post_kickoff": left_post_kickoff,
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "ledger_rows": int(ledger_rows),
        "flip_count": tilt.flip_count,
        "flipped_game_ids": [flip.game_id for flip in tilt.flips],
    }
