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

CHALLENGER_ID = "low_total_div_home_dog_challenger"

LOW_TOTAL_MAX = 42.0


@dataclass(frozen=True)
class TiltFlip:
    game_id: str
    matchup: str
    total_line: float
    spread_line: float


@dataclass(frozen=True)
class TiltResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_low_total_div_home_dog_overlay(
    predictions: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:

    required = {
        "game_id",
        "home_team",
        "away_team",
        "home_cover_probability",
        "div_game",
        "total_line",
        "spread_line",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return TiltResult(base, (), enabled)

    div_game = pd.to_numeric(base["div_game"], errors="coerce").eq(1.0)
    total_line = pd.to_numeric(base["total_line"], errors="coerce")
    spread_line = pd.to_numeric(base["spread_line"], errors="coerce")
    low_total = total_line.notna() & total_line.le(LOW_TOTAL_MAX)
    home_dog = spread_line.notna() & spread_line.lt(0.0)

    eligible = div_game & low_total & home_dog
    if "game_type" in base.columns:
        eligible &= base["game_type"].astype(str).eq("REG")

    away_pick = base["home_cover_probability"].lt(0.5)
    flip_mask = eligible & away_pick

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for idx in base.loc[flip_mask].index:
        row = base.loc[idx]
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                total_line=float(total_line.loc[idx]),
                spread_line=float(spread_line.loc[idx]),
            )
        )

    return TiltResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: TiltResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup} (total {flip.total_line:.1f}, spread {flip.spread_line:+.1f}): "
        "AWAY -> HOME"
        for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** onto the home underdog "
        f"in a divisional game with a decision total at or below {LOW_TOTAL_MAX:.0f}, where the "
        f"model's own pick was on the away side. {detail}. See docs/schedule_flag_battery.md "
        "(Wave 2, LEAD-42). Prospective evidence only -- not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_paired_overlay_arms(
    artifacts_root: Path,
    challenger_id: str,
    decisions: pd.DataFrame,
    card: pd.DataFrame,
    *,
    replace_week: bool = False,
) -> tuple[int, int]:
    if decisions.empty:
        return 0, 0
    paired = decisions.copy()
    baseline = card.set_index("game_id")["home_cover_probability"]
    paired["baseline_pick_side"] = np.where(paired["game_id"].map(baseline).ge(0.5), "HOME", "AWAY")
    path = artifacts_root / "prospective" / f"{challenger_id}_paired_decisions.parquet"
    replaced_rows = 0
    left_post_kickoff = 0
    if path.is_file():
        existing = pd.read_parquet(path)
        if replace_week and not existing.empty:
            existing, replaced_rows, left_post_kickoff = replace_week_rows(
                existing,
                path,
                season=int(paired["season"].iloc[0]),
                week=int(paired["week"].iloc[0]),
                recorded_at=pd.Timestamp(paired["recorded_at_utc"].iloc[0]),
                columns=tuple(existing.columns),
                challenger_id=challenger_id,
            )
        paired = pd.concat([existing, paired], ignore_index=True).drop_duplicates(
            subset=["challenger_id", "game_id"], keep="first"
        )
    atomic_parquet(paired, path)
    return replaced_rows, left_post_kickoff


def record_low_total_div_home_dog_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
    forecast_artifact: str | None = None,
    replace_week: bool = False,
) -> dict[str, Any]:

    del data_root

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
            "No synchronized active ATS model is available to record overlay decisions from"
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
            "changed underneath this overlay -- re-register before recording"
        )

    card = pd.read_csv(card_path)
    source_columns = {"div_game", "total_line"}
    if card.empty or not source_columns.issubset(card.columns):
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "skipped": True,
            "reason": "decision total/divisional source is absent",
        }
    required = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "away_team",
        "home_team",
        "spread_line",
        "total_line",
        "div_game",
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

    tilt = apply_low_total_div_home_dog_overlay(card)
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
    paired_replaced = 0
    paired_left_post_kickoff = 0
    if not decisions.empty:
        paired_replaced, paired_left_post_kickoff = record_paired_overlay_arms(
            artifacts_root, CHALLENGER_ID, decisions, card, replace_week=replace_week
        )
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
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "replaced_rows": replaced_rows,
        "left_post_kickoff": left_post_kickoff,
        "paired_replaced_rows": paired_replaced,
        "paired_left_post_kickoff": paired_left_post_kickoff,
        "ledger_rows": int(ledger_rows),
        "flip_count": tilt.flip_count,
        "flipped_game_ids": [flip.game_id for flip in tilt.flips],
    }


__all__ = [
    "CHALLENGER_ID",
    "LOW_TOTAL_MAX",
    "TiltFlip",
    "TiltResult",
    "apply_low_total_div_home_dog_overlay",
    "overlay_disclosure_note",
    "record_low_total_div_home_dog_challenger_decisions",
]
