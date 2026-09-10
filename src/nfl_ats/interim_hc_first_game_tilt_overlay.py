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
from nfl_ats.provenance import sha256_file
from nfl_ats.recorder_override import replace_week_rows, resolve_recording_forecast

CHALLENGER_ID = "interim_hc_first_game_tilt_overlay"

_EMPTY_FLAGS_COLUMNS = ("game_id", "team", "entry_id")


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def interim_first_game_flag_by_game_fail_open(repo_root: Path) -> pd.DataFrame:

    from nfl_ats.experiment_runner import _build_interim_coach_trait_data

    try:
        trait_data = _build_interim_coach_trait_data(repo_root)
    except Exception as exc:
        warnings.warn(
            "interim_hc_first_game_tilt: interim-coach join failed, proceeding with zero "
            f"flags ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(
            {column: pd.Series([], dtype=object) for column in _EMPTY_FLAGS_COLUMNS}
        )

    game_trait = trait_data.game_trait
    first_only = game_trait.loc[game_trait["first_game_under_interim"]].copy()
    first_only["game_id"] = first_only["game_id"].astype(str)
    first_only["team"] = _canonical_team(first_only["team"])
    return first_only[list(_EMPTY_FLAGS_COLUMNS)].reset_index(drop=True)


@dataclass(frozen=True)
class TiltFlip:
    game_id: str
    matchup: str
    interim_team: str
    opponent_team: str


@dataclass(frozen=True)
class TiltResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[TiltFlip, ...]
    both_first_game_games: tuple[str, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_interim_hc_first_game_tilt_overlay(
    predictions: pd.DataFrame,
    repo_root: Path,
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

    flags = interim_first_game_flag_by_game_fail_open(repo_root)

    merged = base.copy()
    merged["home_team_canon"] = _canonical_team(merged["home_team"])
    merged["away_team_canon"] = _canonical_team(merged["away_team"])

    if flags.empty:
        merged["home_first_game"] = False
        merged["away_first_game"] = False
    else:
        home_flags = flags.rename(columns={"team": "home_team_canon"})[
            ["game_id", "home_team_canon"]
        ].copy()
        home_flags["home_first_game"] = True
        merged = merged.merge(
            home_flags.drop_duplicates(subset=["game_id", "home_team_canon"]),
            on=["game_id", "home_team_canon"],
            how="left",
        )

        away_flags = flags.rename(columns={"team": "away_team_canon"})[
            ["game_id", "away_team_canon"]
        ].copy()
        away_flags["away_first_game"] = True
        merged = merged.merge(
            away_flags.drop_duplicates(subset=["game_id", "away_team_canon"]),
            on=["game_id", "away_team_canon"],
            how="left",
        )

        merged["home_first_game"] = merged["home_first_game"].fillna(False).astype(bool)
        merged["away_first_game"] = merged["away_first_game"].fillna(False).astype(bool)

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    home_pick = merged["home_cover_probability"].ge(0.5)
    both_first_game = merged["home_first_game"] & merged["away_first_game"]

    flip_to_home = eligible & merged["home_first_game"] & ~both_first_game & ~home_pick
    flip_to_away = eligible & merged["away_first_game"] & ~both_first_game & home_pick
    flip_mask = flip_to_home | flip_to_away

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[TiltFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        interim_is_home = bool(row["home_first_game"])
        interim_team = str(row["home_team"] if interim_is_home else row["away_team"])
        opponent_team = str(row["away_team"] if interim_is_home else row["home_team"])
        flips.append(
            TiltFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                interim_team=interim_team,
                opponent_team=opponent_team,
            )
        )

    both_ids = tuple(merged.loc[eligible & both_first_game, "game_id"].astype(str))
    return TiltResult(overlaid, tuple(flips), both_ids, enabled)


def overlay_disclosure_note(result: TiltResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.opponent_team} -> {flip.interim_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the interim head-coach "
        "first-game tilt (a team playing its first game under a newly appointed interim head "
        "coach, and the model's own pick was not already on that side). "
        f"{detail}. See docs/interim_coach_screen.md. Prospective evidence only -- not applied "
        "to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_interim_hc_first_game_tilt_challenger_decisions(
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

    tilt = apply_interim_hc_first_game_tilt_overlay(card, data_root.parent)
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
        "both_first_game_games": list(tilt.both_first_game_games),
    }
