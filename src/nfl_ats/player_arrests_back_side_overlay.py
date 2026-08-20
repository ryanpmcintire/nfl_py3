"""Production broad player-arrest back-side overlay and paired incumbent.

The played card applies this policy after the year-1-coach fade.  Its former
coach-only production arm is recorded under a separate prospective challenger
identity so every live week remains a paired decision comparison.

Recording is fail-closed on source freshness. The newest snapshot directory
must contain a complete manifest and its hash-verified safe index, and the
manifest's fetch timestamp must be between zero and 36 hours old at the
recording instant. A failed current ingest therefore cannot masquerade as a
week with no affected teams.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
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

PROMOTED_OVERLAY_ID = "player_arrests_recent_14d_back_side_overlay"
CHALLENGER_ID = "player_arrests_recent_14d_no_overlay_incumbent"
PRODUCTION_OVERLAY_ENABLED = True
WINDOW_DAYS = 14
MAX_SNAPSHOT_AGE = pd.Timedelta(hours=36)

TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "JAC": "JAX",
    "IN": "IND",
}


@dataclass(frozen=True)
class ArrestSnapshot:
    snapshot_id: str
    directory: Path
    manifest_path: Path
    safe_index_path: Path
    fetched_at_utc: pd.Timestamp
    age_hours: float
    safe_index_sha256: str
    rows_cached: int


@dataclass(frozen=True)
class ArrestFlip:
    game_id: str
    matchup: str
    original_pick_team: str
    flipped_to_team: str


@dataclass(frozen=True)
class ArrestOverlayResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[ArrestFlip, ...]
    home_flags: pd.Series
    away_flags: pd.Series
    enabled: bool = PRODUCTION_OVERLAY_ENABLED
    snapshot_id: str | None = None
    snapshot_fetched_at_utc: pd.Timestamp | None = None
    safe_index_sha256: str | None = None

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def load_latest_complete_arrest_snapshot(
    data_root: Path,
    *,
    now: datetime | None = None,
) -> ArrestSnapshot:
    """Load the newest snapshot only when its complete safe view is fresh."""

    recorded_at = _record_instant(now)
    root = data_root / "raw" / "player_arrests"
    if not root.is_dir():
        raise FileNotFoundError(f"No player-arrests snapshot root at {root}")
    snapshots = sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
    if not snapshots:
        raise FileNotFoundError(f"No player-arrests snapshots under {root}")

    directory = snapshots[0]
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise DataContractError(
            f"Newest player-arrests snapshot {directory.name} has no manifest; "
            "refusing challenger recording"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DataContractError(
            f"Newest player-arrests snapshot has invalid manifest JSON: {manifest_path}"
        ) from error
    if not isinstance(manifest, dict):
        raise DataContractError(f"Player-arrests manifest is not an object: {manifest_path}")
    if manifest.get("complete") is not True:
        raise DataContractError(
            f"Newest player-arrests snapshot {directory.name} is incomplete; "
            "refusing challenger recording"
        )
    if str(manifest.get("snapshot_id")) != directory.name:
        raise DataContractError(
            f"Player-arrests manifest snapshot_id does not match directory {directory.name}"
        )

    fetched_value = manifest.get("fetched_at_utc")
    if not isinstance(fetched_value, str):
        raise DataContractError(
            f"Player-arrests manifest has invalid fetched_at_utc: {manifest_path}"
        )
    fetched_at = pd.to_datetime(fetched_value, utc=True, errors="coerce")
    if pd.isna(fetched_at):
        raise DataContractError(
            f"Player-arrests manifest has invalid fetched_at_utc: {manifest_path}"
        )
    fetched_at = pd.Timestamp(fetched_at)
    age = recorded_at - fetched_at
    if age < pd.Timedelta(0):
        raise DataContractError(
            f"Player-arrests snapshot {directory.name} is future-dated by "
            f"{-age.total_seconds() / 3600.0:.2f} hours; refusing challenger recording"
        )
    if age > MAX_SNAPSHOT_AGE:
        raise DataContractError(
            f"Player-arrests snapshot {directory.name} is stale at "
            f"{age.total_seconds() / 3600.0:.2f} hours old (maximum 36); "
            "refusing challenger recording"
        )

    policy = manifest.get("point_in_time_policy")
    policy = policy if isinstance(policy, dict) else {}
    safe_name = str(policy.get("safe_index", ""))
    if not safe_name or Path(safe_name).name != safe_name:
        raise DataContractError(
            f"Player-arrests manifest does not name a safe index file: {manifest_path}"
        )
    safe_index_path = directory / safe_name
    if not safe_index_path.is_file():
        raise DataContractError(
            f"Complete player-arrests snapshot is missing safe index: {safe_index_path}"
        )
    files = manifest.get("files")
    files = files if isinstance(files, dict) else {}
    expected_hash = files.get(safe_name)
    if not isinstance(expected_hash, str) or not expected_hash:
        raise DataContractError(
            f"Player-arrests manifest does not hash safe index {safe_name}: {manifest_path}"
        )
    observed_hash = sha256_file(safe_index_path)
    if observed_hash != expected_hash:
        raise DataContractError(
            f"Player-arrests safe-index hash mismatch for {safe_index_path}; "
            "refusing challenger recording"
        )
    return ArrestSnapshot(
        snapshot_id=directory.name,
        directory=directory,
        manifest_path=manifest_path,
        safe_index_path=safe_index_path,
        fetched_at_utc=fetched_at,
        age_hours=age.total_seconds() / 3600.0,
        safe_index_sha256=observed_hash,
        rows_cached=int(manifest.get("rows_cached", 0)),
    )


def _broad_side_flags(
    predictions: pd.DataFrame, incidents: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    required_predictions = {"game_id", "gameday", "home_team", "away_team"}
    missing = sorted(required_predictions.difference(predictions.columns))
    if missing:
        raise DataContractError(
            f"Predictions are missing arrest-flag columns: {', '.join(missing)}"
        )
    required_incidents = {"record_id", "incident_date", "team"}
    missing = sorted(required_incidents.difference(incidents.columns))
    if missing:
        raise DataContractError(f"Safe incident index is missing columns: {', '.join(missing)}")
    if predictions["game_id"].duplicated().any():
        raise DataContractError("Predictions contain duplicate games")
    if incidents["record_id"].duplicated().any():
        raise DataContractError("Safe incident index contains duplicate record_id rows")

    identity = predictions[["game_id", "gameday", "home_team", "away_team"]].copy()
    for column in ("home_team", "away_team"):
        identity[column] = (
            identity[column].astype("string").str.strip().replace(TEAM_ALIASES).astype(object)
        )
    identity["gameday"] = pd.to_datetime(identity["gameday"], errors="coerce")
    if identity["gameday"].isna().any():
        raise DataContractError("Predictions contain invalid gameday values")
    days_since_tuesday = (identity["gameday"].dt.weekday - 1) % 7
    identity["decision_date"] = (
        identity["gameday"] - pd.to_timedelta(days_since_tuesday, unit="D")
    ).dt.normalize()

    safe = incidents[["record_id", "incident_date", "team"]].copy()
    safe["incident_date"] = pd.to_datetime(safe["incident_date"], errors="coerce")
    if safe["incident_date"].isna().any():
        raise DataContractError("Safe incident index contains invalid incident dates")
    safe["team"] = safe["team"].astype("string").str.strip().replace(TEAM_ALIASES).astype(object)
    schedule_teams = set(identity["home_team"]) | set(identity["away_team"])
    safe = safe.loc[safe["team"].isin(schedule_teams)].copy()
    safe = safe.sort_values(["incident_date", "team", "record_id"])
    if safe.empty:
        no_flags = pd.Series(False, index=predictions.index, dtype=bool)
        return no_flags.copy(), no_flags.copy()

    flags: dict[str, pd.Series] = {}
    for side, team_column in (("home", "home_team"), ("away", "away_team")):
        team_games = identity[["game_id", "decision_date", team_column]].rename(
            columns={team_column: "team"}
        )
        team_games = team_games.sort_values(["decision_date", "team", "game_id"])
        joined = pd.merge_asof(
            team_games,
            safe,
            by="team",
            left_on="decision_date",
            right_on="incident_date",
            direction="backward",
            allow_exact_matches=False,
        )
        age = (joined["decision_date"] - joined["incident_date"]).dt.days
        keyed = pd.Series(
            age.between(1, WINDOW_DAYS, inclusive="both").to_numpy(),
            index=joined["game_id"].astype(str),
        )
        flags[side] = predictions["game_id"].astype(str).map(keyed).fillna(False).astype(bool)
    return flags["home"], flags["away"]


def apply_player_arrests_back_side_overlay(
    predictions: pd.DataFrame,
    incidents: pd.DataFrame,
) -> ArrestOverlayResult:
    """Apply the frozen sole-affected-side back-side policy to a weekly card."""

    required = {"game_id", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"Predictions are missing overlay columns: {', '.join(missing)}")
    base = predictions.reset_index(drop=True).copy()
    home_flags, away_flags = _broad_side_flags(base, incidents)
    return apply_frozen_player_arrests_back_side_overlay(
        base, home_flags=home_flags, away_flags=away_flags
    )


def apply_frozen_player_arrests_back_side_overlay(
    predictions: pd.DataFrame,
    *,
    home_flags: pd.Series,
    away_flags: pd.Series,
    snapshot_id: str | None = None,
    snapshot_fetched_at_utc: pd.Timestamp | None = None,
    safe_index_sha256: str | None = None,
) -> ArrestOverlayResult:
    """Apply already-frozen Tuesday flags without reading a newer snapshot."""

    required = {"game_id", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"Predictions are missing overlay columns: {', '.join(missing)}")
    base = predictions.reset_index(drop=True).copy()
    home_flags = pd.Series(home_flags).reset_index(drop=True)
    away_flags = pd.Series(away_flags).reset_index(drop=True)
    if len(home_flags) != len(base) or len(away_flags) != len(base):
        raise DataContractError("Frozen arrest flags do not align with the prediction card")
    if home_flags.isna().any() or away_flags.isna().any():
        raise DataContractError("Frozen arrest flags contain missing values")
    home_flags = home_flags.astype(bool)
    away_flags = away_flags.astype(bool)
    exactly_one = home_flags ^ away_flags
    original_home_pick = pd.to_numeric(base["home_cover_probability"], errors="coerce").ge(0.5)
    if pd.to_numeric(base["home_cover_probability"], errors="coerce").isna().any():
        raise DataContractError("Predictions contain invalid home_cover_probability values")
    flip_mask = exactly_one & original_home_pick.ne(home_flags)

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )
    flips: list[ArrestFlip] = []
    for index in base.loc[flip_mask].index:
        row = base.loc[index]
        home_pick = bool(original_home_pick.loc[index])
        flips.append(
            ArrestFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                original_pick_team=str(row["home_team"] if home_pick else row["away_team"]),
                flipped_to_team=str(
                    row["home_team"] if home_flags.loc[index] else row["away_team"]
                ),
            )
        )
    return ArrestOverlayResult(
        overlaid,
        tuple(flips),
        home_flags,
        away_flags,
        snapshot_id=snapshot_id,
        snapshot_fetched_at_utc=snapshot_fetched_at_utc,
        safe_index_sha256=safe_index_sha256,
    )


def arrest_overlay_disclosure_note(result: ArrestOverlayResult) -> str:
    """Describe a played arrest-overlay change without overstating the evidence."""

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.original_pick_team} -> {flip.flipped_to_team}"
        for flip in result.flips
    )
    return (
        f"**Overlay applied: {result.flip_count} pick{plural} flipped** to the sole team "
        "with a broad player-arrest incident dated 1-14 days before the Tuesday decision "
        "date. The opener-graded historical policy scored 53.76% versus the production "
        "rule's 53.36% (+0.399 points, probability_positive=0.8562); its direction was "
        "discovered on overlapping history, so the gain is tracked prospectively rather "
        f"than claimed as confirmed. {detail}. See docs/player_arrests_back_side_overlay.md."
    )


def record_player_arrests_no_overlay_incumbent_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the former coach-only production arm as the paired incumbent."""

    entry = find_challenger(artifacts_root, CHALLENGER_ID)
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {CHALLENGER_ID!r} is registered as {status!r}; only "
            f"{ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )

    recorded_at = _record_instant(now)
    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError("No synchronized active ATS model is available for arrest decisions")
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        raise ValueError("Active ATS model has no linked weekly forecast")
    metadata_path = forecast / "metadata.json"
    card_path = forecast / "recommendations.csv"
    if not metadata_path.is_file() or not card_path.is_file():
        raise ValueError(f"Linked weekly forecast is incomplete: {forecast}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("active_model_id") != active.get("model_id"):
        raise ValueError("Weekly forecast model ID does not match the active model")
    if metadata.get("synchronization_status") != "SYNCHRONIZED":
        raise ValueError("Weekly forecast is not synchronized with an evaluation")

    observed_config = artifact_model_config(metadata)
    declared_fingerprint = config_fingerprint(entry.get("model", {}))
    observed_fingerprint = config_fingerprint(observed_config)
    if declared_fingerprint != observed_fingerprint:
        raise DataContractError(
            f"Challenger {CHALLENGER_ID!r} is registered pinned to configuration "
            f"fingerprint {declared_fingerprint}, but the current active forecast "
            f"{forecast} was produced with {observed_fingerprint}; re-register before recording"
        )

    card = pd.read_csv(card_path)
    required = {
        "game_id",
        "season",
        "week",
        "gameday",
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

    # Imported lazily to avoid the intentional card_view -> this module edge.
    from nfl_ats.card_view import resolve_overlay

    incumbent = resolve_overlay(card, data_root)
    incumbent_card = incumbent.overlaid_predictions

    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID)]
    already = card["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    keep = pre_kickoff & ~already
    fresh = incumbent_card.loc[keep]

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
        "ledger_rows": ledger_rows,
        "coach_flip_count": incumbent.flip_count,
        "coach_flipped_game_ids": [flip.game_id for flip in incumbent.flips],
    }


__all__ = [
    "CHALLENGER_ID",
    "MAX_SNAPSHOT_AGE",
    "PRODUCTION_OVERLAY_ENABLED",
    "PROMOTED_OVERLAY_ID",
    "ArrestFlip",
    "ArrestOverlayResult",
    "ArrestSnapshot",
    "apply_frozen_player_arrests_back_side_overlay",
    "apply_player_arrests_back_side_overlay",
    "arrest_overlay_disclosure_note",
    "load_latest_complete_arrest_snapshot",
    "record_player_arrests_no_overlay_incumbent_decisions",
]
