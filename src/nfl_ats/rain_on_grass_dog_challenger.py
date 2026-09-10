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
from nfl_ats.forecast_cold_visitor_tilt_overlay import STATION_MAP_RELATIVE_PATH
from nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay import (
    LIVE_FORECAST_CUTOFF_MODE,
    FetchBulletin,
    fetch_kickoff_nearest_forecasts_fail_open,
    fetch_mos_bulletin,
    games_for_forecast_fetch,
    validate_live_forecast_provenance,
)
from nfl_ats.io import atomic_parquet
from nfl_ats.low_total_div_home_dog_challenger import record_paired_overlay_arms
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
from nfl_ats.snapshots import latest_snapshot, load_snapshot
from nfl_ats.surface_switch_tilt_overlay import GRASS_SURFACES

CHALLENGER_ID = "rain_on_grass_dog_challenger"

PRECIP_PROB_THRESHOLD_PCT = 60.0


def rain_on_grass_flag_by_game(schedules: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:

    required_forecast = {"game_id", "forecast_precip_prob_pct"}
    missing = sorted(required_forecast.difference(forecasts.columns))
    if missing:
        raise DataContractError(f"forecasts is missing columns: {', '.join(missing)}")
    required_schedule = {"game_id", "game_type", "surface"}
    missing_schedule = sorted(required_schedule.difference(schedules.columns))
    if missing_schedule:
        raise DataContractError(
            f"schedules is missing columns for rain-on-grass tracking: "
            f"{', '.join(missing_schedule)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["game_id"] = reg["game_id"].astype(str)
    reg["grass"] = reg["surface"].astype(str).str.strip().str.lower().isin(GRASS_SURFACES)

    forecasts = forecasts.copy()
    forecasts["game_id"] = forecasts["game_id"].astype(str)
    forecasts["forecast_precip_prob_pct"] = pd.to_numeric(
        forecasts["forecast_precip_prob_pct"], errors="coerce"
    )

    frame = reg[["game_id", "grass"]].merge(
        forecasts[["game_id", "forecast_precip_prob_pct"]].drop_duplicates(subset="game_id"),
        on="game_id",
        how="left",
    )
    flag = frame["grass"] & frame["forecast_precip_prob_pct"].ge(PRECIP_PROB_THRESHOLD_PCT)
    frame["rain_on_grass_flag"] = flag.fillna(False).astype(bool)
    return frame[["game_id", "rain_on_grass_flag", "forecast_precip_prob_pct"]]


@dataclass(frozen=True)
class RainOnGrassFlip:
    game_id: str
    matchup: str
    underdog_team: str
    forecast_precip_prob_pct: float
    spread_line: float


@dataclass(frozen=True)
class RainOnGrassResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[RainOnGrassFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_rain_on_grass_dog_tilt_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    enabled: bool = True,
) -> RainOnGrassResult:

    required = {
        "game_id",
        "season",
        "home_team",
        "away_team",
        "home_cover_probability",
        "spread_line",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return RainOnGrassResult(base, (), enabled)

    base["game_id"] = base["game_id"].astype(str)
    flags = rain_on_grass_flag_by_game(schedules, forecasts)
    merged = base.merge(
        flags[["game_id", "rain_on_grass_flag", "forecast_precip_prob_pct"]],
        on="game_id",
        how="left",
    )
    merged["rain_on_grass_flag"] = merged["rain_on_grass_flag"].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    spread_line = pd.to_numeric(merged["spread_line"], errors="coerce")
    has_dog = spread_line.notna() & spread_line.ne(0.0)
    underdog_is_home = spread_line.lt(0.0)
    current_pick_home = merged["home_cover_probability"].ge(0.5)

    qualifies = eligible & merged["rain_on_grass_flag"] & has_dog
    flip_mask = qualifies & current_pick_home.ne(underdog_is_home)

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[RainOnGrassFlip] = []
    for idx in merged.loc[flip_mask].index:
        row = merged.loc[idx]
        underdog_home = bool(underdog_is_home.loc[idx])
        underdog_team = str(row["home_team"] if underdog_home else row["away_team"])
        precip = row["forecast_precip_prob_pct"]
        flips.append(
            RainOnGrassFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                underdog_team=underdog_team,
                forecast_precip_prob_pct=float(precip) if pd.notna(precip) else float("nan"),
                spread_line=float(spread_line.loc[idx]),
            )
        )

    return RainOnGrassResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: RainOnGrassResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup} (precip {flip.forecast_precip_prob_pct:.0f}%, "
        f"spread {flip.spread_line:+.1f}): -> {flip.underdog_team}"
        for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** onto the underdog by the "
        "live (pool-decision) rain-on-grass tilt (grass-surface game with a high live forecast "
        "precipitation probability, and the model's own pick was not already on the dog). "
        f"{detail}. See docs/weather_venue_leads.md (LEAD-37). Prospective evidence only -- "
        "not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_rain_on_grass_dog_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    registry_root: Path,
    *,
    now: datetime | None = None,
    fetch_bulletin: FetchBulletin = fetch_mos_bulletin,
    forecasts: pd.DataFrame | None = None,
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

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    try:
        schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    except FileNotFoundError as error:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "skipped": True,
            "reason": f"schedule source is absent: {error}",
        }

    if forecasts is None:
        games_for_fetch = games_for_forecast_fetch(card, schedules)
        station_map_path = registry_root / STATION_MAP_RELATIVE_PATH
        forecasts = fetch_kickoff_nearest_forecasts_fail_open(
            games_for_fetch, station_map_path, fetch_bulletin=fetch_bulletin
        )

    if (
        forecasts.empty
        or "forecast_precip_prob_pct" not in forecasts
        or pd.to_numeric(forecasts["forecast_precip_prob_pct"], errors="coerce").notna().sum() == 0
    ):
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "skipped": True,
            "reason": "precipitation forecast source is absent",
            "flip_count": 0,
            "forecast_cutoff_mode": LIVE_FORECAST_CUTOFF_MODE,
        }

    validate_live_forecast_provenance(forecasts, card)
    issuance = pd.to_datetime(forecasts["issuance_runtime_utc"], utc=True, errors="coerce")
    if issuance.gt(recorded_at).any():
        raise DataContractError("Supplied live forecasts include a post-recording issuance")

    tilt = apply_rain_on_grass_dog_tilt_overlay(card, schedules, forecasts)
    tilted_card = tilt.overlaid_predictions

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
        "forecast_fetch_status_counts": (
            forecasts["fetch_status"].value_counts().to_dict() if not forecasts.empty else {}
        ),
        "forecast_cutoff_mode": LIVE_FORECAST_CUTOFF_MODE,
    }


__all__ = [
    "CHALLENGER_ID",
    "PRECIP_PROB_THRESHOLD_PCT",
    "RainOnGrassFlip",
    "RainOnGrassResult",
    "apply_rain_on_grass_dog_tilt_overlay",
    "overlay_disclosure_note",
    "rain_on_grass_flag_by_game",
    "record_rain_on_grass_dog_challenger_decisions",
]
