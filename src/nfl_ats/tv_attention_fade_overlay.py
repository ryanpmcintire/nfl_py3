from __future__ import annotations

import json
from bisect import bisect_left
from collections.abc import Hashable
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

CHALLENGER_ID = "tv_attention_fade_overlay"

N_TRAILING = 3
LIVE_ERA_START_SEASON = 2022
MIN_EDGE_POOL_ROWS = 30
TOP_TERCILE_QUANTILE = 2.0 / 3.0

PRIMETIME_WINDOWS = {
    "SNF",
    "SNF**",
    "MNF",
    "MNF**",
    "MNF early",
    "MNF late",
    "TNF",
    "TNF*",
    "TNF**",
    "Kickoff",
}
SUNDAY_LATE_WINDOWS = {"Late DH", "National", "Late Sat"}
SUNDAY_EARLY_WINDOWS = {
    "Early DH",
    "Regional",
    "Single",
    "Early Sat",
    "Sat 1",
    "Sat 3",
    "London",
    "London*",
    "Special",
    "Special**",
}


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def window_bucket(window: str) -> str:
    if window in PRIMETIME_WINDOWS:
        return "primetime"
    if window in SUNDAY_LATE_WINDOWS:
        return "sunday_late"
    if window in SUNDAY_EARLY_WINDOWS:
        return "sunday_early"
    return "unknown"


def latest_smw_publications_snapshot(data_root: Path) -> Path:
    candidates = sorted(
        (data_root / "raw" / "sports_media_watch_publications").glob("*/manifest.json")
    )
    complete = []
    for manifest_path in candidates:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") == "COMPLETE":
            complete.append(manifest_path.parent)
    pool = complete if complete else [path.parent for path in candidates]
    if not pool:
        raise FileNotFoundError(
            f"no sports_media_watch_publications snapshot found under {data_root}"
        )
    return sorted(pool)[-1] / "ratings_rows.parquet"


def load_usable_rows(snapshot: Path) -> pd.DataFrame:
    frame = pd.read_parquet(snapshot)
    usable = frame[
        (frame["point_in_time_usable"] == True)  # noqa: E712
        & frame["home_team"].notna()
        & frame["away_team"].notna()
    ].copy()
    usable["season"] = usable["season"].astype(int)
    usable["week"] = usable["week"].astype(int)
    usable["bucket"] = usable["window"].apply(window_bucket)
    usable["home_team"] = _canonical_team(usable["home_team"])
    usable["away_team"] = _canonical_team(usable["away_team"])
    return usable


def build_team_z_long(usable: pd.DataFrame) -> pd.DataFrame:
    bucket_stats = usable.groupby("bucket")["viewers"].agg(["mean", "std"])
    merged = usable.merge(
        bucket_stats.rename(columns={"mean": "bucket_mean", "std": "bucket_std"}),
        on="bucket",
        how="left",
    )
    merged["z"] = (merged["viewers"] - merged["bucket_mean"]) / merged["bucket_std"]

    home_side = merged[["season", "week", "home_team", "z"]].rename(columns={"home_team": "team"})
    away_side = merged[["season", "week", "away_team", "z"]].rename(columns={"away_team": "team"})
    team_long = pd.concat([home_side, away_side], ignore_index=True)
    team_long["key"] = team_long["season"] * 100 + team_long["week"]
    return team_long.sort_values(["team", "key"]).reset_index(drop=True)


def rolling_trailing(team_long: pd.DataFrame) -> pd.DataFrame:
    result = team_long.copy()
    result["trailing"] = result.groupby("team")["z"].transform(
        lambda s: s.rolling(N_TRAILING, min_periods=N_TRAILING).mean()
    )
    return result


def build_trailing_lookup(team_long: pd.DataFrame) -> dict[str, tuple[list[int], list[float]]]:
    lookup: dict[str, tuple[list[int], list[float]]] = {}
    for team, group in team_long.dropna(subset=["trailing"]).groupby("team"):
        lookup[str(team)] = (group["key"].tolist(), group["trailing"].tolist())
    return lookup


def trailing_as_of(lookup: dict[str, tuple[list[int], list[float]]], team: str, key: int) -> float:
    if team not in lookup:
        return float("nan")
    keys, values = lookup[team]
    position = bisect_left(keys, key)
    while position > 0 and keys[position - 1] >= key:
        position -= 1
    if position == 0:
        return float("nan")
    return values[position - 1]


def top_tercile_threshold(edge_pool: pd.DataFrame, season: int) -> float | None:
    prior = edge_pool.loc[edge_pool["season"] < season, "trailing"].dropna()
    if len(prior) < MIN_EDGE_POOL_ROWS:
        return None
    return float(np.quantile(prior.to_numpy(dtype=float), TOP_TERCILE_QUANTILE))


@dataclass(frozen=True)
class FadeFlip:
    game_id: str
    matchup: str
    faded_team: str
    backed_team: str


@dataclass(frozen=True)
class FadeResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[FadeFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_tv_attention_fade_overlay(
    predictions: pd.DataFrame,
    smw_snapshot: Path,
    *,
    enabled: bool = True,
) -> FadeResult:

    required = {
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_cover_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return FadeResult(base, (), enabled)

    usable = load_usable_rows(smw_snapshot)
    z_long = build_team_z_long(usable)
    historical_trailing = rolling_trailing(z_long)
    live_trailing = rolling_trailing(
        z_long.loc[z_long["season"] >= LIVE_ERA_START_SEASON].reset_index(drop=True)
    )
    live_lookup = build_trailing_lookup(live_trailing)
    edge_pool = historical_trailing.dropna(subset=["trailing"])

    merged = base.copy()
    merged["home_team"] = _canonical_team(merged["home_team"])
    merged["away_team"] = _canonical_team(merged["away_team"])
    merged["picked_home"] = merged["home_cover_probability"].ge(0.5)
    merged["picked_team"] = np.where(
        merged["picked_home"], merged["home_team"], merged["away_team"]
    )
    merged["opponent_team"] = np.where(
        merged["picked_home"], merged["away_team"], merged["home_team"]
    )

    eligible_mask = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible_mask &= merged["game_type"].astype(str).eq("REG")
    eligible = eligible_mask.to_numpy()

    records: list[dict[Hashable, Any]] = merged[
        ["season", "week", "picked_team", "game_id", "away_team", "home_team", "opponent_team"]
    ].to_dict(orient="records")

    flip_list: list[bool] = [False] * len(merged)
    threshold_cache: dict[int, float | None] = {}
    for position, item in enumerate(records):
        if not bool(eligible[position]):
            continue
        season = int(item["season"])
        week = int(item["week"])
        key = season * 100 + week
        trailing = trailing_as_of(live_lookup, str(item["picked_team"]), key)
        if not np.isfinite(trailing):
            continue
        if season not in threshold_cache:
            threshold_cache[season] = top_tercile_threshold(edge_pool, season)
        threshold = threshold_cache[season]
        if threshold is None:
            continue
        if trailing >= threshold:
            flip_list[position] = True

    flip_mask = pd.Series(flip_list, index=merged.index)
    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[FadeFlip] = []
    for position, flipped in enumerate(flip_list):
        if not flipped:
            continue
        item = records[position]
        flips.append(
            FadeFlip(
                game_id=str(item["game_id"]),
                matchup=f"{item['away_team']} at {item['home_team']}",
                faded_team=str(item["picked_team"]),
                backed_team=str(item["opponent_team"]),
            )
        )

    return FadeResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: FadeResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.faded_team} -> {flip.backed_team}" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the TV-attention "
        "FADE overlay (the model's picked team's trailing national-broadcast audience sits "
        f"in the top tercile of the live-tracking distribution). {detail}. See "
        "docs/tv_attention_screen.md. Prospective evidence only -- not applied to the "
        "published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_tv_attention_fade_overlay_decisions(
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

    smw_snapshot = latest_smw_publications_snapshot(data_root)
    tilt = apply_tv_attention_fade_overlay(card, smw_snapshot)
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
        "smw_snapshot": str(smw_snapshot),
    }
