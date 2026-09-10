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

OVERLAY_ENABLED = True

OVERLAY_WEEK_MAX = 8

CHALLENGER_ID = "hc_year_one_fade_overlay"


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def team_season_primary_coach(schedules: pd.DataFrame) -> pd.DataFrame:

    required = {"home_team", "away_team", "home_coach", "away_coach", "season", "game_type"}
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for coach tenure: {', '.join(missing)}"
        )
    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")]
    home = pd.DataFrame(
        {
            "team": _canonical_team(reg["home_team"]),
            "season": reg["season"].astype(int),
            "coach": reg["home_coach"],
        }
    )
    away = pd.DataFrame(
        {
            "team": _canonical_team(reg["away_team"]),
            "season": reg["season"].astype(int),
            "coach": reg["away_coach"],
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long = long.loc[long["coach"].notna()]
    primary = (
        long.groupby(["team", "season"])["coach"]
        .agg(lambda values: values.value_counts().idxmax())
        .reset_index()
        .rename(columns={"coach": "primary_coach"})
    )
    return primary


def year_one_by_game(schedules: pd.DataFrame) -> pd.DataFrame:

    required = {
        "game_id",
        "season",
        "game_type",
        "home_team",
        "away_team",
        "home_coach",
        "away_coach",
    }
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for coach tenure: {', '.join(missing)}"
        )

    primary = team_season_primary_coach(schedules)

    frame = pd.DataFrame(
        {
            "game_id": schedules["game_id"].astype(str),
            "season": schedules["season"].astype(int),
            "home_team": _canonical_team(schedules["home_team"]),
            "away_team": _canonical_team(schedules["away_team"]),
            "home_coach": schedules["home_coach"],
            "away_coach": schedules["away_coach"],
        }
    )
    frame["prior_season"] = frame["season"] - 1

    home_prior = primary.rename(
        columns={"team": "home_team", "season": "prior_season", "primary_coach": "home_prior_coach"}
    )
    frame = frame.merge(home_prior, on=["home_team", "prior_season"], how="left")
    away_prior = primary.rename(
        columns={"team": "away_team", "season": "prior_season", "primary_coach": "away_prior_coach"}
    )
    frame = frame.merge(away_prior, on=["away_team", "prior_season"], how="left")

    frame["year_one_home"] = (
        frame["home_prior_coach"].notna()
        & frame["home_coach"].notna()
        & frame["home_coach"].ne(frame["home_prior_coach"])
    )
    frame["year_one_away"] = (
        frame["away_prior_coach"].notna()
        & frame["away_coach"].notna()
        & frame["away_coach"].ne(frame["away_prior_coach"])
    )
    return frame[["game_id", "season", "year_one_home", "year_one_away"]]


@dataclass(frozen=True)
class OverlayFlip:
    game_id: str
    matchup: str
    year_one_team: str
    opponent_team: str


@dataclass(frozen=True)
class OverlayResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[OverlayFlip, ...]
    both_year_one_games: tuple[str, ...]
    week_max: int
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_coach_fade_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    week_max: int = OVERLAY_WEEK_MAX,
    enabled: bool = OVERLAY_ENABLED,
) -> OverlayResult:

    required = {"game_id", "season", "week", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return OverlayResult(base, (), (), week_max, enabled)

    flags = year_one_by_game(schedules)
    merged = base.merge(
        flags,
        on=["game_id", "season"],
        how="left",
        validate="one_to_one",
    )
    merged["year_one_home"] = merged["year_one_home"].fillna(False).astype(bool)
    merged["year_one_away"] = merged["year_one_away"].fillna(False).astype(bool)

    eligible = merged["week"].astype(int).le(week_max)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = merged["home_cover_probability"].ge(0.5)
    picked_is_year_one = merged["year_one_home"].where(home_pick, merged["year_one_away"])
    opponent_is_year_one = merged["year_one_away"].where(home_pick, merged["year_one_home"])
    both_year_one = merged["year_one_home"] & merged["year_one_away"]

    flip_mask = eligible & picked_is_year_one & ~opponent_is_year_one

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[OverlayFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        row_home_pick = bool(row["home_cover_probability"] >= 0.5)
        year_one_team = str(row["home_team"] if row_home_pick else row["away_team"])
        opponent_team = str(row["away_team"] if row_home_pick else row["home_team"])
        flips.append(
            OverlayFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                year_one_team=year_one_team,
                opponent_team=opponent_team,
            )
        )

    both_ids = tuple(merged.loc[eligible & both_year_one, "game_id"].astype(str))
    return OverlayResult(overlaid, tuple(flips), both_ids, week_max, enabled)


def overlay_disclosure_note(result: OverlayResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.year_one_team} -> {flip.opponent_team}" for flip in result.flips
    )
    return (
        f"**Overlay applied: {result.flip_count} pick{plural} flipped** by the year-1 "
        f"head-coach fade (weeks 1-{result.week_max}, clean case only: the model sided "
        "with a first-year coach's team against a coach the opponent KEPT). "
        f"{detail}. See docs/coach_fade_overlay.md."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_overlay_challenger_decisions(
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
    overlay = apply_coach_fade_overlay(card, schedules)
    overlaid_card = overlay.overlaid_predictions

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
    fresh = overlaid_card.loc[keep]

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
        "flip_count": overlay.flip_count,
        "flipped_game_ids": [flip.game_id for flip in overlay.flips],
    }
