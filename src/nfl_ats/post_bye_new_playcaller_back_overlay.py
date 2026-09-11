from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.bye_edge_fade_overlay import bye_edge_flag_by_game
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

CHALLENGER_ID = "post_bye_new_playcaller_back_overlay"

NEW_PLAYCALLER_TENURE_YEARS = ("1", "2")


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _normalize_person(name: Any) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(name)).strip().casefold()


def _latest_under(root: Path, pattern: str) -> Path:
    candidates = sorted(root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no match for {pattern} under {root}")
    return candidates[-1]


def load_coordinator_history(data_root: Path) -> pd.DataFrame:
    path = _latest_under(data_root / "raw" / "coordinators", "*/coordinator_history.parquet")
    return pd.read_parquet(path)


def oc_tenure_by_team_season(coordinator_history: pd.DataFrame) -> pd.DataFrame:

    required = {"season", "team", "role", "person", "sample_mode"}
    missing = sorted(required.difference(coordinator_history.columns))
    if missing:
        raise DataContractError(f"coordinator_history is missing columns: {', '.join(missing)}")

    preseason_oc = coordinator_history.loc[
        coordinator_history["sample_mode"].eq("preseason") & coordinator_history["role"].eq("OC")
    ].copy()
    preseason_oc["team"] = _canonical_team(preseason_oc["team"])
    preseason_oc["person_norm"] = preseason_oc["person"].map(_normalize_person)
    preseason_oc["season"] = preseason_oc["season"].astype(int)
    lookup: dict[tuple[int, str], str] = {
        (int(row["season"]), str(row["team"])): str(row["person_norm"])
        for row in preseason_oc.to_dict(orient="records")
    }
    seasons = sorted(preseason_oc["season"].unique())
    teams = sorted(preseason_oc["team"].unique())
    rows = []
    for season in seasons:
        for team in teams:
            oc_s = lookup.get((season, team))
            if oc_s is None:
                continue
            oc_s1 = lookup.get((season - 1, team))
            if oc_s1 is None:
                tenure = "unknown"
            elif oc_s != oc_s1:
                tenure = "1"
            else:
                oc_s2 = lookup.get((season - 2, team))
                if oc_s2 is None:
                    tenure = "unknown"
                elif oc_s1 == oc_s2:
                    tenure = "3+"
                else:
                    tenure = "2"
            rows.append({"season": season, "team": team, "oc_tenure": tenure})
    return pd.DataFrame(rows, columns=["season", "team", "oc_tenure"])


def post_bye_new_oc_flag_by_game(
    schedules: pd.DataFrame, coordinator_history: pd.DataFrame
) -> pd.DataFrame:

    required = {"game_id", "season", "game_type", "gameday", "home_team", "away_team"}
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for post-bye playcaller tracking: {', '.join(missing)}"
        )

    bye_flags = bye_edge_flag_by_game(schedules)
    tenure = oc_tenure_by_team_season(coordinator_history)
    tenure["new_oc"] = tenure["oc_tenure"].isin(NEW_PLAYCALLER_TENURE_YEARS)

    reg = schedules.copy()
    reg["season"] = reg["season"].astype(int)
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    game_lookup = reg[["game_id", "season", "home_team", "away_team"]].drop_duplicates("game_id")

    home_tenure = tenure.rename(columns={"team": "home_team", "new_oc": "home_new_oc"})[
        ["season", "home_team", "home_new_oc"]
    ]
    away_tenure = tenure.rename(columns={"team": "away_team", "new_oc": "away_new_oc"})[
        ["season", "away_team", "away_new_oc"]
    ]

    merged = (
        game_lookup.merge(
            bye_flags[["game_id", "home_off_bye", "away_off_bye"]], on="game_id", how="left"
        )
        .merge(home_tenure, on=["season", "home_team"], how="left")
        .merge(away_tenure, on=["season", "away_team"], how="left")
    )
    for column in ("home_off_bye", "away_off_bye", "home_new_oc", "away_new_oc"):
        merged[column] = merged[column].fillna(False).astype(bool)

    merged["home_flagged"] = merged["home_off_bye"] & merged["home_new_oc"]
    merged["away_flagged"] = merged["away_off_bye"] & merged["away_new_oc"]
    merged["game_id"] = merged["game_id"].astype(str)
    return merged[["game_id", "season", "home_flagged", "away_flagged"]]


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


def apply_post_bye_new_playcaller_back_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    coordinator_history: pd.DataFrame,
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

    flags = post_bye_new_oc_flag_by_game(schedules, coordinator_history)
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
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the post-bye "
        "new-playcaller BACK overlay (the team is coming off its own bye with an "
        "offensive coordinator in his first or second season, and the model's pick "
        f"was not already on that side). {detail}. See "
        "docs/playcaller_change_leads.md. Prospective evidence only -- not applied "
        "to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_post_bye_new_playcaller_back_overlay_decisions(
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
    coordinator_history = load_coordinator_history(data_root)
    tilt = apply_post_bye_new_playcaller_back_overlay(card, schedules, coordinator_history)
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
