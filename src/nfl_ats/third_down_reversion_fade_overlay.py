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
from nfl_ats.pbp import analysis_plays, latest_pbp_snapshot, load_pbp_snapshot
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

CHALLENGER_ID = "third_down_reversion_fade_overlay"

THIRD_DOWN_TOP_QUARTILE_CENTERED = 0.03392624406886406

HOME_FLAG_COLUMN = "_third_down_reversion_fade_home"
AWAY_FLAG_COLUMN = "_third_down_reversion_fade_away"

_REQUIRED_SCHEDULE_COLUMNS = frozenset({"game_id", "season", "game_type", "home_team", "away_team"})


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _third_down_conv_rate_centered_by_team_season(pbp: pd.DataFrame) -> pd.DataFrame:

    plays = analysis_plays(pbp)
    plays = plays.loc[plays["season_type"].astype(str).eq("REG")].copy()
    plays["season"] = pd.to_numeric(plays["season"], errors="raise").astype(int)
    plays["posteam"] = _canonical_team(plays["posteam"])
    plays["down"] = pd.to_numeric(plays["down"], errors="coerce")
    plays["first_down"] = pd.to_numeric(plays["first_down"], errors="coerce")

    third = plays.loc[plays["down"].eq(3.0)]
    off_third = (
        third.groupby(["season", "posteam"], sort=False)
        .agg(n_third_downs=("play_id", "size"), third_conversions=("first_down", "sum"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    off_third["third_down_conv_rate"] = off_third["third_conversions"] / off_third[
        "n_third_downs"
    ].replace(0, np.nan)

    league_mean = off_third.groupby("season")["third_down_conv_rate"].transform("mean")
    off_third["third_down_conv_rate_centered"] = off_third["third_down_conv_rate"] - league_mean
    return off_third[["season", "team", "third_down_conv_rate_centered"]]


def third_down_over_flag_by_game(schedules: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:

    missing = sorted(_REQUIRED_SCHEDULE_COLUMNS.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for third-down reversion tracking: {', '.join(missing)}"
        )

    panel = _third_down_conv_rate_centered_by_team_season(pbp)
    prior = panel.copy()
    prior["season"] = prior["season"] + 1
    prior = prior.rename(columns={"third_down_conv_rate_centered": "prior_centered_rate"})

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = pd.to_numeric(reg["season"], errors="coerce").astype(int)

    home_prior = prior.rename(
        columns={"team": "home_team", "prior_centered_rate": "home_prior_rate"}
    )
    away_prior = prior.rename(
        columns={"team": "away_team", "prior_centered_rate": "away_prior_rate"}
    )

    frame = reg[["game_id", "season", "home_team", "away_team"]].merge(
        home_prior, on=["home_team", "season"], how="left"
    )
    frame = frame.merge(away_prior, on=["away_team", "season"], how="left")

    frame["third_down_over_home"] = (
        frame["home_prior_rate"].ge(THIRD_DOWN_TOP_QUARTILE_CENTERED).fillna(False).astype(bool)
    )
    frame["third_down_over_away"] = (
        frame["away_prior_rate"].ge(THIRD_DOWN_TOP_QUARTILE_CENTERED).fillna(False).astype(bool)
    )
    return frame[["game_id", "season", "third_down_over_home", "third_down_over_away"]]


@dataclass(frozen=True)
class TiltFlip:
    game_id: str
    matchup: str
    flagged_team: str
    opponent_team: str


@dataclass(frozen=True)
class TiltResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    both_flagged_games: tuple[str, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_third_down_reversion_fade_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    pbp: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    base["game_id"] = base["game_id"].astype(str)
    if not enabled:
        return TiltResult(base, (), (), enabled)

    flags = third_down_over_flag_by_game(schedules, pbp).rename(
        columns={
            "third_down_over_home": HOME_FLAG_COLUMN,
            "third_down_over_away": AWAY_FLAG_COLUMN,
        }
    )
    merged = base.merge(flags, on=["game_id", "season"], how="left", validate="one_to_one")
    for column in (HOME_FLAG_COLUMN, AWAY_FLAG_COLUMN):
        if column not in merged.columns:
            merged[column] = False
        merged[column] = merged[column].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = pd.to_numeric(merged["home_cover_probability"], errors="coerce").ge(0.5)
    both_flagged = merged[HOME_FLAG_COLUMN] & merged[AWAY_FLAG_COLUMN]
    picked_is_flagged = merged[HOME_FLAG_COLUMN].where(home_pick, merged[AWAY_FLAG_COLUMN])

    flip_mask = eligible & picked_is_flagged & ~both_flagged

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        row_home_pick = bool(float(row["home_cover_probability"]) >= 0.5)
        flagged_team = str(row["home_team"] if row_home_pick else row["away_team"])
        opponent = str(row["away_team"] if row_home_pick else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                flagged_team=flagged_team,
                opponent_team=opponent,
            )
        )

    both_ids = tuple(merged.loc[eligible & both_flagged, "game_id"].astype(str))
    return TiltResult(overlaid, tuple(flips), both_ids, enabled)


def overlay_disclosure_note(result: TiltResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.flagged_team} -> {flip.opponent_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the third-down "
        "mean-reversion fade (the model sided with a team whose PRIOR season's centered "
        "3rd-down conversion rate was in the league's top quartile, against an opponent "
        f"that was not). {detail}. See docs/third_down_reversion_fade_overlay.md. "
        "Prospective evidence only -- not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_third_down_reversion_fade_challenger_decisions(
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
            "No synchronized active ATS model is available to record fade decisions from"
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
            "changed underneath this fade -- re-register before recording"
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
    pbp_snapshot = latest_pbp_snapshot(data_root / "pbp" / "raw")
    pbp = load_pbp_snapshot(pbp_snapshot)
    tilt = apply_third_down_reversion_fade_overlay(card, schedules, pbp)
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
        "both_flagged_game_ids": list(tilt.both_flagged_games),
    }


__all__ = [
    "AWAY_FLAG_COLUMN",
    "CHALLENGER_ID",
    "HOME_FLAG_COLUMN",
    "THIRD_DOWN_TOP_QUARTILE_CENTERED",
    "TiltFlip",
    "TiltResult",
    "apply_third_down_reversion_fade_overlay",
    "overlay_disclosure_note",
    "record_third_down_reversion_fade_challenger_decisions",
    "third_down_over_flag_by_game",
]
