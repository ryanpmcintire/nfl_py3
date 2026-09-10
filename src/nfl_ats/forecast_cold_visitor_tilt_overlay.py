from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
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
from nfl_ats.snapshots import latest_snapshot, load_snapshot

CHALLENGER_ID = "forecast_cold_visitor_tilt"

OUTDOOR_ROOFS = frozenset({"outdoors", "open"})
TEMP_GAP_THRESHOLD_F = 25.0


MOS_API = "https://mesonet.agron.iastate.edu/api/1/mos.json"
MOS_MODEL = "MEX"
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
DELAY_SECONDS_DEFAULT = 0.3
MAX_LOOKBACK_STEPS_DEFAULT = 10
ET = ZoneInfo("America/New_York")

STATION_MAP_RELATIVE_PATH = Path("reference") / "stadium_station_map.csv"


def tuesday_noon_et_cutoff_utc(kickoff_utc: pd.Timestamp) -> pd.Timestamp:

    kickoff_et = kickoff_utc.tz_convert(ET)
    et_date: date = kickoff_et.date()
    days_since_tuesday = (et_date.weekday() - 1) % 7
    tuesday_date = et_date - timedelta(days=days_since_tuesday)
    tuesday_noon_et = datetime(
        tuesday_date.year, tuesday_date.month, tuesday_date.day, 12, 0, tzinfo=ET
    )
    return pd.Timestamp(tuesday_noon_et).tz_convert(UTC)


def floor_to_12h_utc(dt: pd.Timestamp) -> datetime:

    hour = 12 if dt.hour >= 12 else 0
    return datetime(dt.year, dt.month, dt.day, hour, 0, tzinfo=UTC)


def candidate_runtimes(cutoff_utc: pd.Timestamp, max_steps: int) -> list[datetime]:

    start = floor_to_12h_utc(cutoff_utc)
    return [start - timedelta(hours=12 * i) for i in range(max_steps)]


class MosFetchError(RuntimeError):
    pass


def fetch_mos_bulletin(
    station: str,
    runtime_utc: datetime,
    *,
    model: str,
    timeout: float = 20.0,
    retries: int = 2,
) -> list[dict[str, Any]]:

    runtime_str = runtime_utc.strftime("%Y-%m-%dT%H:%MZ")
    url = f"{MOS_API}?station={station}&model={model}&runtime={runtime_str}"
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                payload = json.load(resp)
            if "data" in payload:
                return list(payload["data"])
            return []
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            time.sleep(1.0 * (attempt + 1))
    raise MosFetchError(f"{station} {runtime_str}: {last_exc}")


def nearest_row(rows: list[dict[str, Any]], kickoff_utc: pd.Timestamp) -> dict[str, Any] | None:

    if not rows:
        return None
    target = kickoff_utc.to_pydatetime()
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)

    def gap(row: dict[str, Any]) -> float:
        ftime = datetime.fromisoformat(row["ftime_utc"]).replace(tzinfo=UTC)
        return abs((ftime - target).total_seconds())

    return min(rows, key=gap)


FetchBulletin = Callable[..., list[dict[str, Any]]]


def fetch_one_game_temp(
    station: str,
    kickoff_utc: pd.Timestamp,
    cutoff_utc: pd.Timestamp,
    *,
    model: str = MOS_MODEL,
    max_lookback_steps: int = MAX_LOOKBACK_STEPS_DEFAULT,
    delay_seconds: float = DELAY_SECONDS_DEFAULT,
    fetch_bulletin: FetchBulletin = fetch_mos_bulletin,
) -> dict[str, Any]:

    for runtime_utc in candidate_runtimes(cutoff_utc, max_lookback_steps):
        try:
            rows = fetch_bulletin(station, runtime_utc, model=model)
        except MosFetchError:
            time.sleep(delay_seconds)
            return {"forecast_temp_f": None, "fetch_status": "transport_error"}
        time.sleep(delay_seconds)
        if rows:
            row = nearest_row(rows, kickoff_utc)
            assert row is not None
            tmp = row.get("tmp")
            return {
                "forecast_temp_f": float(tmp) if tmp is not None else None,
                "fetch_status": "ok",
            }
    return {"forecast_temp_f": None, "fetch_status": "no_bulletin_within_lookback"}


def _fetch_tuesday_noon_forecast_temps(
    games: pd.DataFrame,
    station_map_path: Path,
    *,
    model: str = MOS_MODEL,
    max_lookback_steps: int = MAX_LOOKBACK_STEPS_DEFAULT,
    delay_seconds: float = DELAY_SECONDS_DEFAULT,
    fetch_bulletin: FetchBulletin = fetch_mos_bulletin,
) -> pd.DataFrame:

    required = {"game_id", "stadium", "kickoff"}
    missing = sorted(required.difference(games.columns))
    if missing:
        raise DataContractError(
            f"games is missing columns for forecast fetch: {', '.join(missing)}"
        )
    if not station_map_path.is_file():
        raise FileNotFoundError(f"No stadium/station map at {station_map_path}")

    station_map = pd.read_csv(station_map_path)
    merged = games[["game_id", "stadium", "kickoff"]].merge(
        station_map[["stadium", "icao_station", "mappable"]], on="stadium", how="left"
    )
    unmapped = sorted(
        merged.loc[merged["mappable"].isna(), "stadium"].astype(str).unique().tolist()
    )
    if unmapped:
        raise ValueError(f"{len(unmapped)} stadium(s) not in {station_map_path}: {unmapped}")
    merged["mappable"] = merged["mappable"].astype(bool)

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        game_id = str(row.game_id)
        if not row.mappable:
            rows.append(
                {
                    "game_id": game_id,
                    "forecast_temp_f": None,
                    "fetch_status": "unmappable_international_stadium",
                }
            )
            continue
        kickoff_utc = pd.Timestamp(row.kickoff)  # type: ignore[arg-type]
        if kickoff_utc.tzinfo is None:
            kickoff_utc = kickoff_utc.tz_localize(UTC)
        cutoff_utc = tuesday_noon_et_cutoff_utc(kickoff_utc)
        fetched = fetch_one_game_temp(
            str(row.icao_station),
            kickoff_utc,
            cutoff_utc,
            model=model,
            max_lookback_steps=max_lookback_steps,
            delay_seconds=delay_seconds,
            fetch_bulletin=fetch_bulletin,
        )
        rows.append({"game_id": game_id, **fetched})
    return pd.DataFrame(rows)


def fetch_tuesday_noon_forecast_temps_fail_open(
    games: pd.DataFrame,
    station_map_path: Path,
    **kwargs: Any,
) -> pd.DataFrame:

    try:
        return _fetch_tuesday_noon_forecast_temps(games, station_map_path, **kwargs)
    except Exception as exc:
        warnings.warn(
            "forecast_cold_visitor_tilt: forecast fetch failed, proceeding with zero "
            f"flags ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(
            {
                "game_id": games["game_id"].astype(str),
                "forecast_temp_f": np.nan,
                "fetch_status": "fetch_failed",
            }
        )


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def team_climate_temp_by_away_game(schedules: pd.DataFrame) -> pd.DataFrame:

    required = {
        "game_id",
        "season",
        "game_type",
        "gameday",
        "home_team",
        "away_team",
        "roof",
        "temp",
    }
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for forecast cold-visitor tracking: {', '.join(missing)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["gameday"] = pd.to_datetime(reg["gameday"], errors="raise")
    reg["outdoor"] = reg["roof"].isin(OUTDOOR_ROOFS)
    reg["temp"] = pd.to_numeric(reg["temp"], errors="coerce")

    home = (
        reg[["home_team", "gameday", "game_id", "outdoor", "temp"]]
        .sort_values(["gameday", "game_id"])
        .reset_index(drop=True)
    )
    outdoor_temp = home["temp"].where(home["outdoor"])
    valid = outdoor_temp.notna()
    cum_sum = outdoor_temp.fillna(0.0).groupby(home["home_team"]).cumsum()
    cum_count = valid.groupby(home["home_team"]).cumsum()
    home = home.assign(climate_temp_after=(cum_sum / cum_count.replace(0, np.nan)).to_numpy())

    climate_timeline = (
        home[["home_team", "gameday", "climate_temp_after"]]
        .rename(columns={"home_team": "team"})
        .sort_values("gameday")
        .reset_index(drop=True)
    )
    away = (
        reg[["game_id", "away_team", "gameday"]]
        .rename(columns={"away_team": "team"})
        .sort_values("gameday")
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        away,
        climate_timeline,
        on="gameday",
        by="team",
        direction="backward",
        allow_exact_matches=False,
    )
    result = merged[["game_id", "climate_temp_after"]].rename(
        columns={"climate_temp_after": "climate_temp"}
    )
    result["game_id"] = result["game_id"].astype(str)
    return result


def forecast_cold_visitor_flag_by_game(
    schedules: pd.DataFrame, forecasts: pd.DataFrame
) -> pd.DataFrame:

    required_forecast = {"game_id", "forecast_temp_f"}
    missing = sorted(required_forecast.difference(forecasts.columns))
    if missing:
        raise DataContractError(f"forecasts is missing columns: {', '.join(missing)}")
    required_schedule = {"game_id", "game_type", "roof"}
    missing_schedule = sorted(required_schedule.difference(schedules.columns))
    if missing_schedule:
        raise DataContractError(
            f"schedules is missing columns for forecast cold-visitor tracking: "
            f"{', '.join(missing_schedule)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["game_id"] = reg["game_id"].astype(str)
    reg["outdoor"] = reg["roof"].isin(OUTDOOR_ROOFS)

    climate = team_climate_temp_by_away_game(schedules)
    frame = reg[["game_id", "outdoor"]].merge(climate, on="game_id", how="left")

    forecasts = forecasts.copy()
    forecasts["game_id"] = forecasts["game_id"].astype(str)
    forecasts["forecast_temp_f"] = pd.to_numeric(forecasts["forecast_temp_f"], errors="coerce")
    frame = frame.merge(
        forecasts[["game_id", "forecast_temp_f"]].drop_duplicates(subset="game_id"),
        on="game_id",
        how="left",
    )

    gap = frame["climate_temp"] - frame["forecast_temp_f"]
    flag = frame["outdoor"] & gap.ge(TEMP_GAP_THRESHOLD_F)
    frame["forecast_cold_visitor_flag"] = flag.fillna(False).astype(bool)
    return frame[["game_id", "forecast_cold_visitor_flag", "climate_temp", "forecast_temp_f"]]


@dataclass(frozen=True)
class ForecastColdVisitorFlip:
    game_id: str
    matchup: str
    away_team: str
    home_team: str
    climate_temp: float
    forecast_temp_f: float
    temp_gap_f: float


@dataclass(frozen=True)
class ForecastColdVisitorResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[ForecastColdVisitorFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_forecast_cold_visitor_tilt_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    enabled: bool = True,
) -> ForecastColdVisitorResult:

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return ForecastColdVisitorResult(base, (), enabled)

    base["game_id"] = base["game_id"].astype(str)
    flags = forecast_cold_visitor_flag_by_game(schedules, forecasts)
    merged = base.merge(flags, on="game_id", how="left")
    merged["forecast_cold_visitor_flag"] = (
        merged["forecast_cold_visitor_flag"].fillna(False).astype(bool)
    )

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    away_pick = merged["home_cover_probability"].lt(0.5)
    flip_mask = eligible & merged["forecast_cold_visitor_flag"] & away_pick

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[ForecastColdVisitorFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        climate_temp = row["climate_temp"]
        forecast_temp_f = row["forecast_temp_f"]
        flips.append(
            ForecastColdVisitorFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                away_team=str(row["away_team"]),
                home_team=str(row["home_team"]),
                climate_temp=float(climate_temp) if pd.notna(climate_temp) else float("nan"),
                forecast_temp_f=float(forecast_temp_f)
                if pd.notna(forecast_temp_f)
                else float("nan"),
                temp_gap_f=(
                    float(climate_temp) - float(forecast_temp_f)
                    if pd.notna(climate_temp) and pd.notna(forecast_temp_f)
                    else float("nan")
                ),
            )
        )

    return ForecastColdVisitorResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: ForecastColdVisitorResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: AWAY -> HOME (gap {flip.temp_gap_f:.0f}F)" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the forecast "
        "cold-visitor tilt (the model's pick was on an away team whose own climate is "
        "much warmer than this game's Tuesday-noon forecast temperature). "
        f"{detail}. See docs/forecast_weather_screen.md. Prospective evidence only -- "
        "not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_forecast_cold_visitor_tilt_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    registry_root: Path,
    *,
    now: datetime | None = None,
    fetch_bulletin: FetchBulletin = fetch_mos_bulletin,
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

    schedules_lookup = schedules[["game_id", "stadium"]].copy()
    schedules_lookup["game_id"] = schedules_lookup["game_id"].astype(str)
    games_for_fetch = card[["game_id", "kickoff"]].copy()
    games_for_fetch["game_id"] = games_for_fetch["game_id"].astype(str)
    games_for_fetch = games_for_fetch.merge(schedules_lookup, on="game_id", how="left")

    station_map_path = registry_root / STATION_MAP_RELATIVE_PATH
    forecasts = fetch_tuesday_noon_forecast_temps_fail_open(
        games_for_fetch, station_map_path, fetch_bulletin=fetch_bulletin
    )

    tilt = apply_forecast_cold_visitor_tilt_overlay(card, schedules, forecasts)
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
    }
