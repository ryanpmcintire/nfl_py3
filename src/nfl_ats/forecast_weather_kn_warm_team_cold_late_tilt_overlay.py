from __future__ import annotations

import time
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
from nfl_ats.forecast_cold_visitor_tilt_overlay import (
    DELAY_SECONDS_DEFAULT,
    MAX_LOOKBACK_STEPS_DEFAULT,
    OUTDOOR_ROOFS,
    STATION_MAP_RELATIVE_PATH,
    FetchBulletin,
    MosFetchError,
    candidate_runtimes,
    fetch_mos_bulletin,
    nearest_row,
)
from nfl_ats.io import atomic_parquet
from nfl_ats.nfl_week import pool_decision_cutoff
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

CHALLENGER_ID = "forecast_weather_kn_warm_team_cold_late_tilt"

WARM_METRO_TEAM_CODES = frozenset(
    {"MIA", "TB", "JAX", "ARI", "SF", "OAK", "LA", "LAC", "SD", "HOU", "DAL", "NO", "LV"}
)
WARM_TEAM_COLD_LATE_TEMP_THRESHOLD_F = 35.0
WARM_TEAM_COLD_LATE_MIN_WEEK = 13


MOS_MODEL = "GFS"
LIVE_FORECAST_CUTOFF_MODE = "pool_decision"


def _live_cutoff_metadata(kickoff_utc: pd.Timestamp) -> dict[str, str]:
    cutoff = pool_decision_cutoff(kickoff_utc.to_pydatetime())
    return {
        "cutoff_mode": LIVE_FORECAST_CUTOFF_MODE,
        "decision_cutoff_utc": cutoff.isoformat(),
    }


def _nearest_row_with_field(
    rows: list[dict[str, Any]], kickoff_utc: pd.Timestamp, field: str
) -> dict[str, Any] | None:

    candidates = [row for row in rows if row.get(field) is not None]
    return nearest_row(candidates, kickoff_utc)


def fetch_one_game_kickoff_nearest(
    station: str,
    kickoff_utc: pd.Timestamp,
    *,
    model: str = MOS_MODEL,
    max_lookback_steps: int = MAX_LOOKBACK_STEPS_DEFAULT,
    delay_seconds: float = DELAY_SECONDS_DEFAULT,
    fetch_bulletin: FetchBulletin = fetch_mos_bulletin,
) -> dict[str, Any]:

    cutoff_metadata = _live_cutoff_metadata(kickoff_utc)
    cutoff_utc = pd.Timestamp(cutoff_metadata["decision_cutoff_utc"])
    for runtime_utc in candidate_runtimes(cutoff_utc, max_lookback_steps):
        try:
            rows = fetch_bulletin(station, runtime_utc, model=model)
        except MosFetchError:
            time.sleep(delay_seconds)
            return {
                "forecast_temp_f": None,
                "forecast_precip_prob_pct": None,
                "fetch_status": "transport_error",
                "issuance_runtime_utc": None,
                **cutoff_metadata,
            }
        time.sleep(delay_seconds)
        if rows:
            row = nearest_row(rows, kickoff_utc)
            assert row is not None
            issuance = pd.to_datetime(str(row.get("runtime_utc")), utc=True, errors="coerce")
            if pd.isna(issuance) or issuance > cutoff_utc:
                return {
                    "forecast_temp_f": None,
                    "forecast_precip_prob_pct": None,
                    "fetch_status": "invalid_issuance_timestamp",
                    "issuance_runtime_utc": row.get("runtime_utc"),
                    **cutoff_metadata,
                }
            tmp = row.get("tmp")
            precip_row = _nearest_row_with_field(
                rows, kickoff_utc, "p06"
            ) or _nearest_row_with_field(rows, kickoff_utc, "p12")
            precip_prob_pct: float | None = None
            if precip_row is not None:
                if precip_row.get("p06") is not None:
                    precip_prob_pct = float(precip_row["p06"])
                elif precip_row.get("p12") is not None:
                    precip_prob_pct = float(precip_row["p12"])
            return {
                "forecast_temp_f": float(tmp) if tmp is not None else None,
                "forecast_precip_prob_pct": precip_prob_pct,
                "fetch_status": "ok",
                "issuance_runtime_utc": issuance.isoformat(),
                **cutoff_metadata,
            }
    return {
        "forecast_temp_f": None,
        "forecast_precip_prob_pct": None,
        "fetch_status": "no_bulletin_within_lookback",
        "issuance_runtime_utc": None,
        **cutoff_metadata,
    }


def games_for_forecast_fetch(card: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:

    schedules_lookup = schedules[["game_id", "stadium"]].copy()
    schedules_lookup["game_id"] = schedules_lookup["game_id"].astype(str)
    games_for_fetch = card[["game_id", "kickoff"]].copy()
    games_for_fetch["game_id"] = games_for_fetch["game_id"].astype(str)
    return games_for_fetch.merge(schedules_lookup, on="game_id", how="left")


def _fetch_kickoff_nearest_forecasts(
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
        kickoff_utc = pd.Timestamp(str(row.kickoff))
        if kickoff_utc.tzinfo is None:
            kickoff_utc = kickoff_utc.tz_localize(UTC)
        cutoff_metadata = _live_cutoff_metadata(kickoff_utc)
        if not row.mappable:
            rows.append(
                {
                    "game_id": game_id,
                    "forecast_temp_f": None,
                    "forecast_precip_prob_pct": None,
                    "fetch_status": "unmappable_international_stadium",
                    "issuance_runtime_utc": None,
                    **cutoff_metadata,
                }
            )
            continue
        fetched = fetch_one_game_kickoff_nearest(
            str(row.icao_station),
            kickoff_utc,
            model=model,
            max_lookback_steps=max_lookback_steps,
            delay_seconds=delay_seconds,
            fetch_bulletin=fetch_bulletin,
        )
        rows.append({"game_id": game_id, **fetched})
    return pd.DataFrame(rows)


def fetch_kickoff_nearest_forecasts_fail_open(
    games: pd.DataFrame,
    station_map_path: Path,
    **kwargs: Any,
) -> pd.DataFrame:

    try:
        return _fetch_kickoff_nearest_forecasts(games, station_map_path, **kwargs)
    except Exception as exc:
        warnings.warn(
            "forecast_weather_kn_warm_team_cold_late_tilt: forecast fetch failed, proceeding "
            f"with zero flags ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        rows = []
        for game in games.itertuples(index=False):
            kickoff = pd.Timestamp(str(game.kickoff))  # type: ignore[attr-defined]
            if kickoff.tzinfo is None:
                kickoff = kickoff.tz_localize(UTC)
            rows.append(
                {
                    "game_id": str(game.game_id),
                    "forecast_temp_f": np.nan,
                    "forecast_precip_prob_pct": np.nan,
                    "fetch_status": "fetch_failed",
                    "issuance_runtime_utc": None,
                    **_live_cutoff_metadata(kickoff),
                }
            )
        return pd.DataFrame(rows)


def fetch_shared_kickoff_nearest_forecasts_fail_open(
    artifacts_root: Path,
    data_root: Path,
    registry_root: Path,
    *,
    fetch_bulletin: FetchBulletin = fetch_mos_bulletin,
    forecast_artifact: str | None = None,
) -> pd.DataFrame | None:

    try:
        active = load_active_ats_model(artifacts_root)
        if active is None:
            return None
        forecast, _metadata = resolve_recording_forecast(
            artifacts_root, active, forecast_artifact=forecast_artifact
        )
        card_path = forecast / "recommendations.csv"
        if not card_path.is_file():
            return None
        card = pd.read_csv(card_path)
        if not {"game_id", "kickoff"}.issubset(card.columns):
            return None
        schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    except (OSError, ValueError, FileNotFoundError, DataContractError):
        return None

    games_for_fetch = games_for_forecast_fetch(card, schedules)
    station_map_path = registry_root / STATION_MAP_RELATIVE_PATH
    return fetch_kickoff_nearest_forecasts_fail_open(
        games_for_fetch, station_map_path, fetch_bulletin=fetch_bulletin
    )


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def warm_team_cold_late_flag_by_game(
    schedules: pd.DataFrame, forecasts: pd.DataFrame
) -> pd.DataFrame:

    required_forecast = {"game_id", "forecast_temp_f"}
    missing = sorted(required_forecast.difference(forecasts.columns))
    if missing:
        raise DataContractError(f"forecasts is missing columns: {', '.join(missing)}")
    required_schedule = {"game_id", "game_type", "week", "away_team", "roof"}
    missing_schedule = sorted(required_schedule.difference(schedules.columns))
    if missing_schedule:
        raise DataContractError(
            "schedules is missing columns for warm-team cold-late tracking: "
            f"{', '.join(missing_schedule)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["game_id"] = reg["game_id"].astype(str)
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["outdoor"] = reg["roof"].isin(OUTDOOR_ROOFS)
    reg["week"] = pd.to_numeric(reg["week"], errors="coerce")

    forecasts = forecasts.copy()
    forecasts["game_id"] = forecasts["game_id"].astype(str)
    forecasts["forecast_temp_f"] = pd.to_numeric(forecasts["forecast_temp_f"], errors="coerce")

    frame = reg[["game_id", "away_team", "outdoor", "week"]].merge(
        forecasts[["game_id", "forecast_temp_f"]].drop_duplicates(subset="game_id"),
        on="game_id",
        how="left",
    )

    flag = (
        frame["away_team"].isin(WARM_METRO_TEAM_CODES)
        & frame["outdoor"]
        & frame["forecast_temp_f"].le(WARM_TEAM_COLD_LATE_TEMP_THRESHOLD_F)
        & frame["week"].ge(WARM_TEAM_COLD_LATE_MIN_WEEK)
    )
    frame["warm_team_cold_late_flag"] = flag.fillna(False).astype(bool)
    return frame[["game_id", "warm_team_cold_late_flag", "forecast_temp_f"]]


@dataclass(frozen=True)
class WarmTeamColdLateFlip:
    game_id: str
    matchup: str
    away_team: str
    home_team: str
    forecast_temp_f: float


@dataclass(frozen=True)
class WarmTeamColdLateResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[WarmTeamColdLateFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_warm_team_cold_late_tilt_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    enabled: bool = True,
) -> WarmTeamColdLateResult:

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return WarmTeamColdLateResult(base, (), enabled)

    base["game_id"] = base["game_id"].astype(str)
    flags = warm_team_cold_late_flag_by_game(schedules, forecasts)
    merged = base.merge(flags, on="game_id", how="left")
    merged["warm_team_cold_late_flag"] = (
        merged["warm_team_cold_late_flag"].fillna(False).astype(bool)
    )

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    away_pick = merged["home_cover_probability"].lt(0.5)
    flip_mask = eligible & merged["warm_team_cold_late_flag"] & away_pick

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[WarmTeamColdLateFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        forecast_temp_f = row["forecast_temp_f"]
        flips.append(
            WarmTeamColdLateFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                away_team=str(row["away_team"]),
                home_team=str(row["home_team"]),
                forecast_temp_f=float(forecast_temp_f)
                if pd.notna(forecast_temp_f)
                else float("nan"),
            )
        )

    return WarmTeamColdLateResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: WarmTeamColdLateResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: AWAY -> HOME (forecast {flip.forecast_temp_f:.0f}F)"
        for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the forecast "
        "(pool-decision) warm-team-cold-late tilt (the model's pick was on an away "
        "team from a warm-winter metro, playing outdoors in week 13 or later with a "
        "decision-time forecast at or below 35F). "
        f"{detail}. See docs/forecast_weather_screen.md. Prospective evidence only -- "
        "not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def validate_live_forecast_provenance(forecasts: pd.DataFrame, card: pd.DataFrame) -> None:
    required = {
        "game_id",
        "cutoff_mode",
        "decision_cutoff_utc",
        "issuance_runtime_utc",
        "fetch_status",
    }
    missing = sorted(required.difference(forecasts.columns))
    if missing:
        raise DataContractError(
            "Supplied live forecasts lack pool-decision provenance: " + ", ".join(missing)
        )
    if forecasts["game_id"].astype(str).duplicated().any():
        raise DataContractError("Supplied live forecasts contain duplicate games")
    if not forecasts["cutoff_mode"].astype(str).eq(LIVE_FORECAST_CUTOFF_MODE).all():
        raise DataContractError("Supplied live forecasts are not labeled pool_decision")

    expected = card[["game_id", "kickoff"]].copy()
    expected["game_id"] = expected["game_id"].astype(str)
    expected["expected_cutoff"] = [
        pool_decision_cutoff(value.to_pydatetime())
        for value in pd.to_datetime(expected["kickoff"], utc=True)
    ]
    observed = forecasts[list(required)].copy()
    observed["game_id"] = observed["game_id"].astype(str)
    joined = expected.merge(observed, on="game_id", how="left", validate="one_to_one")
    if joined["cutoff_mode"].isna().any():
        raise DataContractError("Supplied live forecasts do not cover every card game")
    cutoff = pd.to_datetime(joined["decision_cutoff_utc"], utc=True, errors="coerce")
    expected_cutoff = pd.to_datetime(joined["expected_cutoff"], utc=True)
    if cutoff.isna().any() or not cutoff.eq(expected_cutoff).all():
        raise DataContractError("Supplied live forecasts carry an incorrect pool decision cutoff")
    issuance = pd.to_datetime(joined["issuance_runtime_utc"], utc=True, errors="coerce")
    ok = joined["fetch_status"].astype(str).eq("ok")
    if issuance.loc[ok].isna().any() or issuance.loc[ok].gt(cutoff.loc[ok]).any():
        raise DataContractError("Supplied live forecasts include a post-lock issuance")


def record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions(
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

    tilt = apply_warm_team_cold_late_tilt_overlay(card, schedules, forecasts)
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
