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

CHALLENGER_ID = "pace_mismatch_dog_tilt_overlay"

PACE_DIFF_ABS_THRESHOLD = 2.1685022294778378

OVERLAY_FLAG_COLUMN = "_pace_mismatch_dog_tilt_flag"

TEAM_SEASON_STYLE_REQUIRED_COLUMNS = frozenset({"season", "team", "seconds_per_play_pace_centered"})


def team_season_style_path(data_root: Path) -> Path:

    return data_root / "pbp" / "team_style" / "team_season_style.parquet"


def _canonical_team(team: pd.Series) -> pd.Series:

    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def pace_mismatch_flag_by_game(
    schedules: pd.DataFrame, team_season_style: pd.DataFrame
) -> pd.DataFrame:

    required_schedule = {"game_id", "season", "game_type", "home_team", "away_team"}
    missing_schedule = sorted(required_schedule.difference(schedules.columns))
    if missing_schedule:
        raise DataContractError(
            f"schedules is missing columns for pace-mismatch tracking: "
            f"{', '.join(missing_schedule)}"
        )
    missing_style = sorted(TEAM_SEASON_STYLE_REQUIRED_COLUMNS.difference(team_season_style.columns))
    if missing_style:
        raise DataContractError(
            f"team_season_style is missing columns for pace-mismatch tracking: "
            f"{', '.join(missing_style)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = reg["season"].astype(int)

    style = team_season_style[["team", "season", "seconds_per_play_pace_centered"]].copy()
    style["team"] = _canonical_team(style["team"])
    style["season"] = style["season"].astype(int)
    style["season"] = style["season"] + 1

    prior_home = style.rename(
        columns={"team": "home_team", "seconds_per_play_pace_centered": "home_prior_pace_centered"}
    )
    reg = reg.merge(prior_home, on=["home_team", "season"], how="left")

    prior_away = style.rename(
        columns={"team": "away_team", "seconds_per_play_pace_centered": "away_prior_pace_centered"}
    )
    reg = reg.merge(prior_away, on=["away_team", "season"], how="left")

    reg["pace_diff_abs"] = (reg["home_prior_pace_centered"] - reg["away_prior_pace_centered"]).abs()
    reg["pace_mismatch_flag"] = reg["pace_diff_abs"].ge(PACE_DIFF_ABS_THRESHOLD).fillna(False)

    return reg[["game_id", "season", "pace_diff_abs", "pace_mismatch_flag"]].reset_index(drop=True)


def pace_mismatch_flags_fail_open(data_root: Path) -> pd.DataFrame:

    try:
        style_path = team_season_style_path(data_root)
        if not style_path.is_file():
            raise FileNotFoundError(f"missing team-season pace cache: {style_path}")
        snapshot = latest_snapshot(data_root / "raw")
        schedules, _team_stats = load_snapshot(snapshot)
        team_season_style = pd.read_parquet(style_path)
        return pace_mismatch_flag_by_game(schedules, team_season_style)
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        DataContractError,
    ) as error:  # pragma: no cover - exercised via the warning path
        warnings.warn(
            f"{CHALLENGER_ID}: flag build failed, proceeding with zero flags ({error})",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=["game_id", "season", "pace_diff_abs", "pace_mismatch_flag"])


@dataclass(frozen=True)
class TiltFlip:
    game_id: str
    matchup: str
    original_pick_team: str
    flipped_to_team: str
    spread_line: float
    pace_diff_abs: float


@dataclass(frozen=True)
class TiltResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_pace_mismatch_dog_tilt_overlay(
    predictions: pd.DataFrame,
    flags: pd.DataFrame,
    *,
    enabled: bool = True,
) -> TiltResult:

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
        return TiltResult(base, (), enabled)

    flag_columns = ["game_id", "season", "pace_diff_abs", "pace_mismatch_flag"]
    if flags.empty or "pace_mismatch_flag" not in flags.columns:
        flags_to_merge = pd.DataFrame(columns=flag_columns)
    else:
        flags_to_merge = flags[flag_columns]

    merged = base.merge(
        flags_to_merge.rename(
            columns={"pace_mismatch_flag": OVERLAY_FLAG_COLUMN, "pace_diff_abs": "_pace_diff_abs"}
        ),
        on=["game_id", "season"],
        how="left",
        validate="one_to_one",
    )
    if OVERLAY_FLAG_COLUMN not in merged.columns:
        merged[OVERLAY_FLAG_COLUMN] = False
    merged[OVERLAY_FLAG_COLUMN] = merged[OVERLAY_FLAG_COLUMN].fillna(False).astype(bool)
    if "_pace_diff_abs" not in merged.columns:
        merged["_pace_diff_abs"] = np.nan

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    spread_line = pd.to_numeric(merged["spread_line"], errors="coerce")
    home_favored = spread_line.gt(0.0)
    away_favored = spread_line.lt(0.0)
    eligible &= home_favored | away_favored

    pick_home = merged["home_cover_probability"].ge(0.5)
    model_has_favorite = (home_favored & pick_home) | (away_favored & ~pick_home)

    flip_mask = eligible & merged[OVERLAY_FLAG_COLUMN] & model_has_favorite

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for idx in merged.loc[flip_mask].index:
        row = merged.loc[idx]
        row_home_pick = bool(pick_home.loc[idx])
        original_team = str(row["home_team"] if row_home_pick else row["away_team"])
        flipped_team = str(row["away_team"] if row_home_pick else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                original_pick_team=original_team,
                flipped_to_team=flipped_team,
                spread_line=float(spread_line.loc[idx]),
                pace_diff_abs=float(row["_pace_diff_abs"]),
            )
        )

    return TiltResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: TiltResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup} ({flip.spread_line:+.1f}, pace gap {flip.pace_diff_abs:.2f}): "
        f"{flip.original_pick_team} -> {flip.flipped_to_team}"
        for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** off the market favourite "
        "onto the underdog in a top-quartile prior-season pace mismatch (fewer, longer "
        f"possessions favour the dog). {detail}. See docs/pace_mismatch_dog_tilt_overlay.md. "
        "Prospective evidence only -- not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_pace_mismatch_dog_tilt_challenger_decisions(
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

    flags = pace_mismatch_flags_fail_open(data_root)
    tilt = apply_pace_mismatch_dog_tilt_overlay(card, flags)
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
    }


__all__ = [
    "CHALLENGER_ID",
    "OVERLAY_FLAG_COLUMN",
    "PACE_DIFF_ABS_THRESHOLD",
    "TEAM_SEASON_STYLE_REQUIRED_COLUMNS",
    "TiltFlip",
    "TiltResult",
    "apply_pace_mismatch_dog_tilt_overlay",
    "overlay_disclosure_note",
    "pace_mismatch_flag_by_game",
    "pace_mismatch_flags_fail_open",
    "record_pace_mismatch_dog_tilt_challenger_decisions",
    "team_season_style_path",
]
