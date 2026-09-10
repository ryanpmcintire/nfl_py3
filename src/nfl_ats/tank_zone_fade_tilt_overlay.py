from __future__ import annotations

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
from nfl_ats.features import add_ats_outcomes
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
from nfl_ats.provenance import sha256_file, stamp_sidecar
from nfl_ats.recorder_override import replace_week_rows, resolve_recording_forecast
from nfl_ats.snapshots import latest_snapshot, load_snapshot

CHALLENGER_ID = "tank_zone_fade_tilt_overlay"

OVERLAY_WEEK_MIN = 14
OVERLAY_WEEK_MAX = 18

TANK_ZONE_SIZE = 2

HOME_FLAG_COLUMN = "_tank_zone_tilt_home"
AWAY_FLAG_COLUMN = "_tank_zone_tilt_away"

_REQUIRED_SCHEDULE_COLUMNS = frozenset(
    {"game_id", "season", "week", "game_type", "home_team", "away_team", "result", "spread_line"}
)


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _season_tank_zone_by_week(season_games: pd.DataFrame) -> dict[int, frozenset[str]]:

    teams = sorted(set(season_games["home_team"]) | set(season_games["away_team"]))
    wins: dict[str, int] = dict.fromkeys(teams, 0)
    losses: dict[str, int] = dict.fromkeys(teams, 0)
    settled = season_games.loc[season_games["_standings_eligible"]]
    by_week: dict[int, frozenset[str]] = {}
    for week in sorted(int(value) for value in season_games["week"].unique()):
        ordered = sorted(teams, key=lambda team: (wins[team], -losses[team], team))
        by_week[week] = frozenset(ordered[:TANK_ZONE_SIZE])
        this_week = settled.loc[settled["week"].eq(week), ["home_team", "away_team", "result"]]
        for home, away, result in this_week.to_numpy():
            margin = float(result)
            if margin > 0:
                wins[str(home)] += 1
                losses[str(away)] += 1
            elif margin < 0:
                wins[str(away)] += 1
                losses[str(home)] += 1
    return by_week


def tank_zone_flag_by_game(schedules: pd.DataFrame) -> pd.DataFrame:

    missing = sorted(_REQUIRED_SCHEDULE_COLUMNS.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for tank-zone tracking: {', '.join(missing)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = pd.to_numeric(reg["season"], errors="coerce").astype(int)
    reg["week"] = pd.to_numeric(reg["week"], errors="coerce").astype(int)
    reg["_standings_eligible"] = add_ats_outcomes(reg)["home_cover"].notna().to_numpy()

    if reg.empty:
        return pd.DataFrame(
            {
                "game_id": pd.Series([], dtype=object),
                "season": pd.Series([], dtype=int),
                "tank_zone_home": pd.Series([], dtype=bool),
                "tank_zone_away": pd.Series([], dtype=bool),
            }
        )

    frames: list[pd.DataFrame] = []
    for _season, season_games in reg.groupby("season", sort=True):
        by_week = _season_tank_zone_by_week(season_games)
        home_flags = [
            str(team) in by_week[int(week)]
            for team, week in season_games[["home_team", "week"]].to_numpy()
        ]
        away_flags = [
            str(team) in by_week[int(week)]
            for team, week in season_games[["away_team", "week"]].to_numpy()
        ]
        frames.append(
            pd.DataFrame(
                {
                    "game_id": season_games["game_id"].astype(str).to_numpy(),
                    "season": season_games["season"].to_numpy(),
                    "tank_zone_home": home_flags,
                    "tank_zone_away": away_flags,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class TiltFlip:
    game_id: str
    matchup: str
    tank_zone_team: str
    opponent_team: str


@dataclass(frozen=True)
class TiltResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    both_tank_zone_games: tuple[str, ...]
    week_min: int
    week_max: int
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_tank_zone_fade_tilt_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    week_min: int = OVERLAY_WEEK_MIN,
    week_max: int = OVERLAY_WEEK_MAX,
    enabled: bool = True,
) -> TiltResult:

    required = {"game_id", "season", "week", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    base["game_id"] = base["game_id"].astype(str)
    if not enabled:
        return TiltResult(base, (), (), week_min, week_max, enabled)

    flags = tank_zone_flag_by_game(schedules).rename(
        columns={"tank_zone_home": HOME_FLAG_COLUMN, "tank_zone_away": AWAY_FLAG_COLUMN}
    )
    merged = base.merge(
        flags,
        on=["game_id", "season"],
        how="left",
        validate="one_to_one",
    )
    for column in (HOME_FLAG_COLUMN, AWAY_FLAG_COLUMN):
        if column not in merged.columns:
            merged[column] = False
        merged[column] = merged[column].fillna(False).astype(bool)

    weeks = pd.to_numeric(merged["week"], errors="coerce")
    eligible = weeks.ge(week_min) & weeks.le(week_max)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = pd.to_numeric(merged["home_cover_probability"], errors="coerce").ge(0.5)
    both_tank = merged[HOME_FLAG_COLUMN] & merged[AWAY_FLAG_COLUMN]
    picked_is_tank = merged[HOME_FLAG_COLUMN].where(home_pick, merged[AWAY_FLAG_COLUMN])

    flip_mask = eligible & picked_is_tank & ~both_tank

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        row_home_pick = bool(float(row["home_cover_probability"]) >= 0.5)
        tank_team = str(row["home_team"] if row_home_pick else row["away_team"])
        opponent = str(row["away_team"] if row_home_pick else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                tank_zone_team=tank_team,
                opponent_team=opponent,
            )
        )

    both_ids = tuple(merged.loc[eligible & both_tank, "game_id"].astype(str))
    return TiltResult(overlaid, tuple(flips), both_ids, week_min, week_max, enabled)


def overlay_disclosure_note(result: TiltResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.tank_zone_team} -> {flip.opponent_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the tank-zone fade "
        f"(weeks {result.week_min}-{result.week_max}, clean case only: the model sided with a "
        "team holding one of the league's two worst records entering the week, against an "
        f"opponent that does not). {detail}. See docs/tank_zone_fade_tilt_overlay.md. "
        "Prospective evidence only -- not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_tank_zone_fade_tilt_challenger_decisions(
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

    schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    tilt = apply_tank_zone_fade_tilt_overlay(card, schedules)
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
        ledger_path = challenger_ledger_path(artifacts_root)
        atomic_parquet(combined[list(CHALLENGER_DECISION_COLUMNS)], ledger_path)
        stamp_sidecar(
            ledger_path, extra={"challenger_id": CHALLENGER_ID, "rows_appended": len(decisions)}
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
        "both_tank_zone_games": list(tilt.both_tank_zone_games),
    }
