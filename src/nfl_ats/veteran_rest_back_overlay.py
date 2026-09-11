from __future__ import annotations

import re
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

CHALLENGER_ID = "veteran_rest_back_overlay"

AGE_THRESHOLD = 30.0
REST_PATTERN = re.compile(r"\brest(?:ed|ing)?\b|load management", re.I)
REASON_COLUMNS = (
    "report_primary_injury",
    "report_secondary_injury",
    "practice_primary_injury",
    "practice_secondary_injury",
)
DNP_STATUS = "Did Not Participate In Practice"


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _latest_under(root: Path, pattern: str) -> Path:
    candidates = sorted(root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no match for {pattern} under {root}")
    return candidates[-1]


def load_players_master(data_root: Path) -> pd.DataFrame:
    path = _latest_under(data_root / "players" / "raw", "*/players.parquet")
    return pd.read_parquet(path)


def load_raw_injuries(data_root: Path) -> pd.DataFrame:
    path = _latest_under(data_root / "raw" / "nflverse_injuries", "*/injuries.parquet")
    return pd.read_parquet(path)


def team_week_kickoff_index(schedules: pd.DataFrame) -> pd.DataFrame:
    reg = schedules.copy()
    reg["season"] = reg["season"].astype(int)
    reg["week"] = reg["week"].astype(int)
    local = pd.to_datetime(reg["gameday"].astype(str).str[:10] + " " + reg["gametime"].astype(str))
    reg["kickoff"] = local.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    home = reg[["season", "week", "home_team", "kickoff"]].rename(columns={"home_team": "team"})
    away = reg[["season", "week", "away_team", "kickoff"]].rename(columns={"away_team": "team"})
    long_df = pd.concat([home, away], ignore_index=True)
    long_df["team"] = _canonical_team(long_df["team"])
    return long_df.drop_duplicates(["season", "week", "team"]).reset_index(drop=True)


def rest_tagged_veteran_flag_by_team_week(
    injuries: pd.DataFrame, players: pd.DataFrame, team_week_kickoff: pd.DataFrame
) -> pd.DataFrame:

    required = {"season", "week", "team", "gsis_id", "practice_status", *REASON_COLUMNS}
    missing = sorted(required.difference(injuries.columns))
    if missing:
        raise DataContractError(f"injuries is missing columns: {', '.join(missing)}")

    frame = injuries.copy()
    frame["team"] = _canonical_team(frame["team"])
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)

    rest_mask = pd.Series(False, index=frame.index)
    for column in REASON_COLUMNS:
        rest_mask |= frame[column].fillna("").astype(str).str.contains(REST_PATTERN)
    frame["rest_reason"] = rest_mask
    frame["not_dnp"] = frame["practice_status"].notna() & frame["practice_status"].ne(DNP_STATUS)

    birth = players.loc[players["gsis_id"].notna(), ["gsis_id", "birth_date"]].drop_duplicates(
        "gsis_id"
    )
    frame = frame.merge(birth, on="gsis_id", how="left", validate="many_to_one")
    frame = frame.merge(
        team_week_kickoff, on=["season", "week", "team"], how="left", validate="many_to_one"
    )
    birth_date = pd.to_datetime(frame["birth_date"], errors="coerce", utc=True)
    kickoff = pd.to_datetime(frame["kickoff"], errors="coerce", utc=True)
    age_years = (kickoff - birth_date).dt.days / 365.25
    frame["age_eligible"] = birth_date.notna() & kickoff.notna() & age_years.ge(AGE_THRESHOLD)

    frame["flagged"] = frame["rest_reason"] & frame["age_eligible"] & frame["not_dnp"]
    team_week = (
        frame.groupby(["season", "week", "team"])["flagged"]
        .any()
        .rename("team_flagged")
        .reset_index()
    )
    return team_week


def veteran_rest_flag_by_game(
    schedules: pd.DataFrame, injuries: pd.DataFrame, players: pd.DataFrame
) -> pd.DataFrame:

    required = {"game_id", "season", "week", "game_type", "gameday", "home_team", "away_team"}
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for veteran-rest tracking: {', '.join(missing)}"
        )

    team_week_kickoff = team_week_kickoff_index(schedules)
    team_week_flag = rest_tagged_veteran_flag_by_team_week(injuries, players, team_week_kickoff)

    reg = schedules.copy()
    reg["season"] = reg["season"].astype(int)
    reg["week"] = reg["week"].astype(int)
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    game_lookup = reg[["game_id", "season", "week", "home_team", "away_team"]].drop_duplicates(
        "game_id"
    )

    home_flag = team_week_flag.rename(columns={"team": "home_team", "team_flagged": "home_flagged"})
    away_flag = team_week_flag.rename(columns={"team": "away_team", "team_flagged": "away_flagged"})
    flags = game_lookup.merge(
        home_flag[["season", "week", "home_team", "home_flagged"]],
        on=["season", "week", "home_team"],
        how="left",
    ).merge(
        away_flag[["season", "week", "away_team", "away_flagged"]],
        on=["season", "week", "away_team"],
        how="left",
    )
    flags["home_flagged"] = flags["home_flagged"].fillna(False).astype(bool)
    flags["away_flagged"] = flags["away_flagged"].fillna(False).astype(bool)
    flags["game_id"] = flags["game_id"].astype(str)
    return flags[["game_id", "season", "home_flagged", "away_flagged"]]


@dataclass(frozen=True)
class BackFlip:
    game_id: str
    matchup: str
    backed_team: str
    opponent_team: str


@dataclass(frozen=True)
class BackResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[BackFlip, ...]
    both_flagged_games: tuple[str, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_veteran_rest_back_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    injuries: pd.DataFrame,
    players: pd.DataFrame,
    *,
    enabled: bool = True,
) -> BackResult:

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return BackResult(base, (), (), enabled)

    flags = veteran_rest_flag_by_game(schedules, injuries, players)
    merged = base.merge(
        flags[["game_id", "home_flagged", "away_flagged"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    merged["home_flagged"] = merged["home_flagged"].fillna(False).astype(bool)
    merged["away_flagged"] = merged["away_flagged"].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = merged["home_cover_probability"].ge(0.5)
    unique_flag = merged["home_flagged"].ne(merged["away_flagged"])
    already_on_flagged_side = merged["home_flagged"].where(home_pick, merged["away_flagged"])

    flip_mask = eligible & unique_flag & ~already_on_flagged_side

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[BackFlip] = []
    for idx in merged.index[flip_mask]:
        row = merged.loc[idx]
        new_home_pick = bool(overlaid.loc[idx, "home_cover_probability"] >= 0.5)
        backed_team = str(row["home_team"] if new_home_pick else row["away_team"])
        opponent_team = str(row["away_team"] if new_home_pick else row["home_team"])
        flips.append(
            BackFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                backed_team=backed_team,
                opponent_team=opponent_team,
            )
        )

    both_ids = tuple(
        merged.loc[eligible & merged["home_flagged"] & merged["away_flagged"], "game_id"].astype(
            str
        )
    )
    return BackResult(overlaid, tuple(flips), both_ids, enabled)


def overlay_disclosure_note(result: BackResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.opponent_team} -> {flip.backed_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the veteran "
        "rest-day BACK overlay (a 30+ starter was rest-tagged this week and the "
        f"model's pick was not already on that side). {detail}. See "
        "docs/veteran_rest_back_overlay.md. Prospective evidence only -- not applied "
        "to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_veteran_rest_back_overlay_decisions(
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
    players = load_players_master(data_root)
    injuries = load_raw_injuries(data_root)
    tilt = apply_veteran_rest_back_overlay(card, schedules, injuries, players)
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
        "both_flagged_games": list(tilt.both_flagged_games),
    }
