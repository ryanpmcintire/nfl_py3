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
from nfl_ats.provenance import sha256_file, stamp_sidecar
from nfl_ats.recorder_override import replace_week_rows, resolve_recording_forecast
from nfl_ats.snapshots import latest_snapshot, load_snapshot

CHALLENGER_ID = "bye_edge_fade_overlay"

POST_BYE_GAP_DAYS = 12

HOME_OFF_BYE_COLUMN = "_bye_edge_fade_home_off_bye"
AWAY_OFF_BYE_COLUMN = "_bye_edge_fade_away_off_bye"


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def bye_edge_flag_by_game(schedules: pd.DataFrame) -> pd.DataFrame:

    required = {"game_id", "season", "game_type", "gameday", "home_team", "away_team"}
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for bye-edge tracking: {', '.join(missing)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = reg["season"].astype(int)
    reg["gameday_dt"] = pd.to_datetime(reg["gameday"], errors="coerce")

    long_rows = []
    for _, game in reg.iterrows():
        for side, team in (("home", game["home_team"]), ("away", game["away_team"])):
            long_rows.append(
                {
                    "game_id": game["game_id"],
                    "season": game["season"],
                    "team": team,
                    "side": side,
                    "gameday_dt": game["gameday_dt"],
                }
            )
    long_df = pd.DataFrame(
        long_rows, columns=["game_id", "season", "team", "side", "gameday_dt"]
    ).sort_values(["team", "season", "gameday_dt"])
    long_df["gap_days"] = long_df.groupby(["team", "season"])["gameday_dt"].diff().dt.days
    long_df["post_bye"] = (long_df["gap_days"] >= POST_BYE_GAP_DAYS).fillna(False).astype(bool)

    def side_map(side: str) -> pd.Series:
        joined = reg[["game_id"]].merge(
            long_df.loc[long_df["side"] == side, ["game_id", "post_bye"]],
            on="game_id",
            how="left",
        )
        return joined["post_bye"].fillna(False).astype(bool)

    frame = reg[["game_id", "season"]].reset_index(drop=True).copy()
    frame["home_off_bye"] = side_map("home").to_numpy()
    frame["away_off_bye"] = side_map("away").to_numpy()
    return frame


@dataclass(frozen=True)
class TiltFlip:
    game_id: str
    matchup: str
    bye_team: str
    opponent_team: str


@dataclass(frozen=True)
class TiltResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_bye_edge_fade_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return TiltResult(base, (), enabled)

    flags = bye_edge_flag_by_game(schedules)
    merged = base.merge(
        flags.rename(
            columns={
                "home_off_bye": HOME_OFF_BYE_COLUMN,
                "away_off_bye": AWAY_OFF_BYE_COLUMN,
            }
        ),
        on=["game_id", "season"],
        how="left",
        validate="one_to_one",
    )
    for column in (HOME_OFF_BYE_COLUMN, AWAY_OFF_BYE_COLUMN):
        if column not in merged.columns:
            merged[column] = False
        merged[column] = merged[column].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_is_bye_team = merged[HOME_OFF_BYE_COLUMN] & ~merged[AWAY_OFF_BYE_COLUMN]
    away_is_bye_team = merged[AWAY_OFF_BYE_COLUMN] & ~merged[HOME_OFF_BYE_COLUMN]
    home_pick = merged["home_cover_probability"].ge(0.5)

    flip_mask = eligible & ((home_is_bye_team & home_pick) | (away_is_bye_team & ~home_pick))

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        bye_is_home = bool(row[HOME_OFF_BYE_COLUMN])
        bye_team = str(row["home_team"] if bye_is_home else row["away_team"])
        opponent_team = str(row["away_team"] if bye_is_home else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                bye_team=bye_team,
                opponent_team=opponent_team,
            )
        )

    return TiltResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: TiltResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.bye_team} -> {flip.opponent_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the bye-edge fade "
        "(the model's pick was on the team coming off a strict, >=12-day-gap bye, and its "
        "opponent was not also off a bye this week). "
        f"{detail}. See docs/bye_edge_fade_overlay.md. Prospective evidence only -- not "
        "applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_bye_edge_fade_challenger_decisions(
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
    fade = apply_bye_edge_fade_overlay(card, schedules)
    faded_card = fade.overlaid_predictions

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
    fresh = faded_card.loc[keep]

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
        "flip_count": fade.flip_count,
        "flipped_game_ids": [flip.game_id for flip in fade.flips],
    }
