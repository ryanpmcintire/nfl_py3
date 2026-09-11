from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.forecast_cold_visitor_tilt_overlay import OUTDOOR_ROOFS, STATION_MAP_RELATIVE_PATH
from nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay import (
    LIVE_FORECAST_CUTOFF_MODE,
    FetchBulletin,
    fetch_kickoff_nearest_forecasts_fail_open,
    fetch_mos_bulletin,
    games_for_forecast_fetch,
    validate_live_forecast_provenance,
)
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
from nfl_ats.snapshots import latest_snapshot, load_snapshot

CHALLENGER_ID = "forecast_weather_kn_wind_pass_heavy_fade"

WIND_THRESHOLD_MPH = 15.0
PASS_RATE_FADE_QUARTILE = 4


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def team_pass_rate_quartile_by_season(data_root: Path) -> pd.DataFrame:

    from nfl_ats.experiment_runner import _lag_and_quartile, _team_season_pass_rate
    from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot

    snapshot = latest_pbp_snapshot(data_root / "pbp" / "raw")
    pbp = load_pbp_snapshot(snapshot, include_postseason=False)
    rate = _team_season_pass_rate(pbp)
    lagged = _lag_and_quartile(rate)
    return lagged[["team", "season", "quartile"]].rename(columns={"quartile": "pass_rate_quartile"})


def team_pass_rate_quartile_fail_open(data_root: Path) -> pd.DataFrame:

    try:
        return team_pass_rate_quartile_by_season(data_root)
    except (FileNotFoundError, KeyError, ValueError, DataContractError) as error:
        warnings.warn(
            f"{CHALLENGER_ID}: pass-rate trait build failed, proceeding with zero flags "
            f"({type(error).__name__}: {error})",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=["team", "season", "pass_rate_quartile"])


def wind_pass_heavy_fade_flag_by_game(
    schedules: pd.DataFrame, forecasts: pd.DataFrame, trait: pd.DataFrame
) -> pd.DataFrame:

    required_forecast = {"game_id", "forecast_wind_mph"}
    missing = sorted(required_forecast.difference(forecasts.columns))
    if missing:
        raise DataContractError(f"forecasts is missing columns: {', '.join(missing)}")
    required_schedule = {"game_id", "game_type", "season", "home_team", "away_team", "roof"}
    missing_schedule = sorted(required_schedule.difference(schedules.columns))
    if missing_schedule:
        raise DataContractError(
            "schedules is missing columns for wind pass-heavy tracking: "
            f"{', '.join(missing_schedule)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["game_id"] = reg["game_id"].astype(str)
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = reg["season"].astype(int)
    reg["outdoor"] = reg["roof"].isin(OUTDOOR_ROOFS)

    forecasts = forecasts.copy()
    forecasts["game_id"] = forecasts["game_id"].astype(str)
    forecasts["forecast_wind_mph"] = pd.to_numeric(forecasts["forecast_wind_mph"], errors="coerce")

    frame = reg[["game_id", "season", "home_team", "away_team", "outdoor"]].merge(
        forecasts[["game_id", "forecast_wind_mph"]].drop_duplicates(subset="game_id"),
        on="game_id",
        how="left",
    )
    weather_ok = frame["outdoor"] & frame["forecast_wind_mph"].ge(WIND_THRESHOLD_MPH)

    trait = trait.copy()
    if not trait.empty:
        trait["team"] = _canonical_team(trait["team"])
        trait["season"] = trait["season"].astype(int)

    home_trait = trait.rename(
        columns={"team": "home_team", "pass_rate_quartile": "home_pass_rate_quartile"}
    )
    frame = frame.merge(home_trait, on=["home_team", "season"], how="left")
    away_trait = trait.rename(
        columns={"team": "away_team", "pass_rate_quartile": "away_pass_rate_quartile"}
    )
    frame = frame.merge(away_trait, on=["away_team", "season"], how="left")

    frame["fade_home"] = (
        weather_ok & frame["home_pass_rate_quartile"].eq(PASS_RATE_FADE_QUARTILE)
    ).fillna(False)
    frame["fade_away"] = (
        weather_ok & frame["away_pass_rate_quartile"].eq(PASS_RATE_FADE_QUARTILE)
    ).fillna(False)
    return frame[["game_id", "fade_home", "fade_away", "forecast_wind_mph"]]


@dataclass(frozen=True)
class WindPassHeavyFadeFlip:
    game_id: str
    matchup: str
    faded_team: str
    faded_side: str
    forecast_wind_mph: float


@dataclass(frozen=True)
class WindPassHeavyFadeResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[WindPassHeavyFadeFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_wind_pass_heavy_fade_tilt_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    forecasts: pd.DataFrame,
    trait: pd.DataFrame,
    *,
    enabled: bool = True,
) -> WindPassHeavyFadeResult:

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return WindPassHeavyFadeResult(base, (), enabled)

    base["game_id"] = base["game_id"].astype(str)
    flags = wind_pass_heavy_fade_flag_by_game(schedules, forecasts, trait)
    merged = base.merge(
        flags[["game_id", "fade_home", "fade_away", "forecast_wind_mph"]], on="game_id", how="left"
    )
    merged["fade_home"] = merged["fade_home"].fillna(False).astype(bool)
    merged["fade_away"] = merged["fade_away"].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    pick_home = merged["home_cover_probability"].ge(0.5)
    flip_mask = eligible & ((pick_home & merged["fade_home"]) | (~pick_home & merged["fade_away"]))

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[WindPassHeavyFadeFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        was_home_pick = bool(row["home_cover_probability"] >= 0.5)
        faded_team = str(row["home_team"]) if was_home_pick else str(row["away_team"])
        faded_side = "HOME" if was_home_pick else "AWAY"
        wind = row["forecast_wind_mph"]
        flips.append(
            WindPassHeavyFadeFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                faded_team=faded_team,
                faded_side=faded_side,
                forecast_wind_mph=float(wind) if pd.notna(wind) else float("nan"),
            )
        )

    return WindPassHeavyFadeResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: WindPassHeavyFadeResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: fade {flip.faded_team} ({flip.faded_side}, forecast wind "
        f"{flip.forecast_wind_mph:.0f}mph)"
        for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the forecast "
        "(pool-decision) wind pass-heavy tilt (the model's picked team was playing outdoors "
        "with a decision-time forecast wind of 15mph or more, and that team's own "
        "prior-season pass-attempt rate is in the most pass-heavy quartile). "
        f"{detail}. See docs/weather_interactions.md. Prospective evidence only -- "
        "not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_forecast_weather_kn_wind_pass_heavy_fade_challenger_decisions(
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

    schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))

    if forecasts is None:
        games_for_fetch = games_for_forecast_fetch(card, schedules)
        station_map_path = registry_root / STATION_MAP_RELATIVE_PATH
        forecasts = fetch_kickoff_nearest_forecasts_fail_open(
            games_for_fetch, station_map_path, fetch_bulletin=fetch_bulletin
        )

    validate_live_forecast_provenance(forecasts, card)

    trait = team_pass_rate_quartile_fail_open(data_root)
    tilt = apply_wind_pass_heavy_fade_tilt_overlay(card, schedules, forecasts, trait)
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
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "replaced_rows": replaced_rows,
        "left_post_kickoff": left_post_kickoff,
        "ledger_rows": int(ledger_rows),
        "flip_count": tilt.flip_count,
        "flipped_game_ids": [flip.game_id for flip in tilt.flips],
        "forecast_fetch_status_counts": (
            forecasts["fetch_status"].value_counts().to_dict() if not forecasts.empty else {}
        ),
        "forecast_cutoff_mode": LIVE_FORECAST_CUTOFF_MODE,
    }
