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

CHALLENGER_ID = "special_teams_return_tilt_overlay"

QUARTILE_TOP = 0.75

RETURN_COMPOSITE_LEGS: tuple[str, ...] = ("punt_return_yards", "kickoff_return_yards")

REQUIRED_TEAM_SEASON_COLUMNS = {
    "season",
    "team",
    "punt_return_yards_centered",
    "kickoff_return_yards_centered",
}


def return_composite_z_with_threshold(team_season: pd.DataFrame) -> tuple[pd.DataFrame, float]:

    missing = REQUIRED_TEAM_SEASON_COLUMNS.difference(team_season.columns)
    if missing:
        raise DataContractError(
            f"team_season is missing columns for the return composite: {', '.join(sorted(missing))}"
        )

    result = team_season.copy()
    for leg in RETURN_COMPOSITE_LEGS:
        centered = f"{leg}_centered"
        sd = float(result[centered].std(ddof=1))
        result[f"{leg}_z"] = result[centered] / sd if sd > 0 else np.nan

    result["return_composite_z"] = result[[f"{leg}_z" for leg in RETURN_COMPOSITE_LEGS]].mean(
        axis=1
    )
    threshold = float(result["return_composite_z"].quantile(QUARTILE_TOP))
    return result, threshold


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def special_teams_return_flag_by_game(
    schedules: pd.DataFrame, team_season: pd.DataFrame
) -> pd.DataFrame:

    required = {"game_id", "season", "game_type", "home_team", "away_team"}
    missing = required.difference(schedules.columns)
    if missing:
        raise DataContractError(
            "schedules is missing columns for special-teams return tracking: "
            f"{', '.join(sorted(missing))}"
        )

    composite, threshold = return_composite_z_with_threshold(team_season)
    composite = composite[["season", "team", "return_composite_z"]].copy()
    composite["team"] = _canonical_team(composite["team"])
    composite["season"] = composite["season"].astype(int)
    composite["season"] = composite["season"] + 1

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = reg["season"].astype(int)

    home_prior = composite.rename(
        columns={"team": "home_team", "return_composite_z": "home_prior_return_composite_z"}
    )
    frame = reg[["game_id", "season", "home_team", "away_team"]].merge(
        home_prior, on=["home_team", "season"], how="left"
    )
    away_prior = composite.rename(
        columns={"team": "away_team", "return_composite_z": "away_prior_return_composite_z"}
    )
    frame = frame.merge(away_prior, on=["away_team", "season"], how="left")

    frame["home_return_top_quartile"] = (
        frame["home_prior_return_composite_z"].ge(threshold).fillna(False).astype(bool)
    )
    frame["away_return_top_quartile"] = (
        frame["away_prior_return_composite_z"].ge(threshold).fillna(False).astype(bool)
    )
    frame["return_composite_top_quartile_threshold"] = threshold
    return frame[
        [
            "game_id",
            "season",
            "home_return_top_quartile",
            "away_return_top_quartile",
            "home_prior_return_composite_z",
            "away_prior_return_composite_z",
            "return_composite_top_quartile_threshold",
        ]
    ]


_EMPTY_FLAGS_COLUMNS = ("game_id", "season", "home_return_top_quartile", "away_return_top_quartile")


def latest_special_teams_team_season(data_root: Path) -> Path | None:

    root = data_root / "raw" / "special_teams"
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*/team_season.parquet"))
    return candidates[-1] if candidates else None


def special_teams_return_flag_by_game_fail_open(
    data_root: Path, schedules: pd.DataFrame
) -> pd.DataFrame:

    try:
        team_season_path = latest_special_teams_team_season(data_root)
        if team_season_path is None:
            raise FileNotFoundError(
                f"no {data_root / 'raw' / 'special_teams'}/*/team_season.parquet snapshot found"
            )
        team_season = pd.read_parquet(team_season_path)
        return special_teams_return_flag_by_game(schedules, team_season)
    except (FileNotFoundError, KeyError, ValueError, DataContractError) as error:
        warnings.warn(
            f"{CHALLENGER_ID}: flag build failed, proceeding with zero flags "
            f"({type(error).__name__}: {error})",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(
            {column: pd.Series([], dtype=object) for column in _EMPTY_FLAGS_COLUMNS}
        )


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


def apply_special_teams_return_tilt_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    data_root: Path,
    *,
    enabled: bool = True,
) -> TiltResult:

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = required.difference(predictions.columns)
    if missing:
        raise DataContractError(
            f"predictions is missing overlay columns: {', '.join(sorted(missing))}"
        )

    base = predictions.reset_index(drop=True).copy()
    base["game_id"] = base["game_id"].astype(str)
    if not enabled:
        return TiltResult(base, (), (), enabled)

    flags = special_teams_return_flag_by_game_fail_open(data_root, schedules)
    merged = base.merge(
        flags[["game_id", "home_return_top_quartile", "away_return_top_quartile"]],
        on="game_id",
        how="left",
    )
    for column in ("home_return_top_quartile", "away_return_top_quartile"):
        if column not in merged.columns:
            merged[column] = False
        merged[column] = merged[column].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = merged["home_cover_probability"].ge(0.5)
    both_flagged = merged["home_return_top_quartile"] & merged["away_return_top_quartile"]

    flip_to_home = eligible & merged["home_return_top_quartile"] & ~both_flagged & ~home_pick
    flip_to_away = eligible & merged["away_return_top_quartile"] & ~both_flagged & home_pick
    flip_mask = flip_to_home | flip_to_away

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        flagged_is_home = bool(row["home_return_top_quartile"])
        flagged_team = str(row["home_team"] if flagged_is_home else row["away_team"])
        opponent_team = str(row["away_team"] if flagged_is_home else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                flagged_team=flagged_team,
                opponent_team=opponent_team,
            )
        )

    both_ids = tuple(merged.loc[eligible & both_flagged, "game_id"].astype(str))
    return TiltResult(overlaid, tuple(flips), both_ids, enabled)


def overlay_disclosure_note(result: TiltResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.opponent_team} -> {flip.flagged_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** onto a team whose "
        "prior-season special-teams return composite (mean z of punt-return and "
        "kickoff-return yards) ranks in the top quartile league-wide, where the model's own "
        f"pick was not already on that side. {detail}. See "
        "docs/special_teams_return_tilt_overlay.md. Prospective evidence only -- not applied "
        "to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_special_teams_return_tilt_challenger_decisions(
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
    missing = required.difference(card.columns)
    if missing:
        raise DataContractError(
            f"Active forecast card is missing columns: {', '.join(sorted(missing))}"
        )
    if card["game_id"].duplicated().any():
        raise DataContractError("Active forecast card contains duplicate games")
    spreads = pd.to_numeric(card["spread_line"], errors="coerce")
    if not np.isfinite(spreads.to_numpy(dtype=float)).all():
        raise DataContractError("Active forecast card has games without a decision spread")
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Active forecast card has games without a kickoff timestamp")

    schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    tilt = apply_special_teams_return_tilt_overlay(card, schedules, data_root)
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
    "CHALLENGER_ID",
    "QUARTILE_TOP",
    "RETURN_COMPOSITE_LEGS",
    "TiltFlip",
    "TiltResult",
    "apply_special_teams_return_tilt_overlay",
    "latest_special_teams_team_season",
    "overlay_disclosure_note",
    "record_special_teams_return_tilt_challenger_decisions",
    "return_composite_z_with_threshold",
    "special_teams_return_flag_by_game",
    "special_teams_return_flag_by_game_fail_open",
]
