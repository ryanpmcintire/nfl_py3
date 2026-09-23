from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.mass_preserving_lattice import (
    BAND_HALF_WIDTH,
    BAND_STEP,
    MAX_BAND,
    MIN_BAND_GAMES,
    MassPreservingRead,
    prior_pool,
    prior_pool_for_week,
    tilted_atoms,
)
from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock
from nfl_ats.prospective_scoring import ACTIVE_CHALLENGER_STATUS, find_challenger
from nfl_ats.provenance import sha256_file
from nfl_ats.recorder_override import replace_week_rows, resolve_recording_forecast

CHALLENGER_ID = "total_conditioned_key_number_lattice_v1"

TOTAL_BAND_EDGES: tuple[float, float] = (42.5, 47.5)
TOTAL_BAND_LABELS: tuple[str, str, str] = ("low", "mid", "high")
ATOM_TOLERANCE = 1e-9
CHALLENGER_POLICY = "discrete_conditional_non_push_total_band_v1"

LEDGER_COLUMNS: tuple[str, ...] = (
    "recorded_at_utc",
    "challenger_id",
    "challenger_policy",
    "source_artifact",
    "source_sha256",
    "forecast_created_at_utc",
    "game_id",
    "season",
    "week",
    "kickoff",
    "deadline",
    "home_team",
    "away_team",
    "spread_line",
    "total_line",
    "total_band",
    "served_cover_probability",
    "served_pick_side",
    "challenger_cover_probability",
    "challenger_pick_side",
    "challenger_cover",
    "challenger_push",
    "challenger_loss",
    "challenger_band",
    "challenger_band_games",
    "challenger_fallback_to_unconditioned",
    "prior_pool_rows",
)


def total_band(value: float) -> str:
    lo, hi = TOTAL_BAND_EDGES
    if value < lo:
        return TOTAL_BAND_LABELS[0]
    if value <= hi:
        return TOTAL_BAND_LABELS[1]
    return TOTAL_BAND_LABELS[2]


def total_conditioned_read(
    pool_line: np.ndarray,
    pool_margin: np.ndarray,
    pool_total_band: np.ndarray,
    target_band: str,
    line: float,
    point: float,
    half_width: float = BAND_HALF_WIDTH,
    min_band_games: int = MIN_BAND_GAMES,
) -> tuple[MassPreservingRead, bool]:
    mask_total = pool_total_band == target_band
    band = half_width
    while True:
        selected = (np.abs(pool_line - line) <= band) & mask_total
        if int(selected.sum()) >= min_band_games or band >= MAX_BAND:
            break
        band = min(band + BAND_STEP, MAX_BAND)
    fallback = int(selected.sum()) < min_band_games
    if fallback:
        band = half_width
        while True:
            selected = np.abs(pool_line - line) <= band
            if int(selected.sum()) >= min_band_games or band >= MAX_BAND:
                break
            band = min(band + BAND_STEP, MAX_BAND)
    margins = pool_margin[selected]
    values, counts = np.unique(margins, return_counts=True)
    mass, theta = tilted_atoms(values, counts.astype(float), line, point)
    is_push = np.abs(values - line) < ATOM_TOLERANCE
    read = MassPreservingRead(
        cover=float(mass[values > line + ATOM_TOLERANCE].sum()),
        push=float(mass[is_push].sum()),
        loss=float(mass[values < line - ATOM_TOLERANCE].sum()),
        theta=theta,
        band=band,
        band_games=int(selected.sum()),
        atoms=int(values.size),
        key_mass_3=float(mass[np.abs(np.abs(values) - 3.0) < ATOM_TOLERANCE].sum()),
    )
    return read, fallback


def ledger_path(artifacts_root: Path) -> Path:
    return artifacts_root / "prospective" / "total_conditioned_lattice_decisions.parquet"


def load_decisions(artifacts_root: Path) -> pd.DataFrame:
    path = ledger_path(artifacts_root)
    if not path.is_file():
        return pd.DataFrame(columns=list(LEDGER_COLUMNS))
    ledger = pd.read_parquet(path)
    missing = sorted(set(LEDGER_COLUMNS).difference(ledger.columns))
    if missing:
        raise DataContractError(f"{CHALLENGER_ID} ledger is missing columns: {', '.join(missing)}")
    if ledger["game_id"].duplicated().any():
        raise DataContractError(f"{CHALLENGER_ID} ledger contains duplicate rows: {path}")
    return ledger[list(LEDGER_COLUMNS)]


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_total_conditioned_lattice_decisions(
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
            "No synchronized active ATS model is available to record challenger decisions from"
        )
    forecast, metadata = resolve_recording_forecast(
        artifacts_root, active, forecast_artifact=forecast_artifact
    )
    card_path = forecast / "recommendations.csv"
    card = pd.read_csv(card_path)
    required = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "home_team",
        "away_team",
        "spread_line",
        "total_line",
        "home_cover_probability",
    }
    missing = sorted(required.difference(card.columns))
    if missing:
        raise DataContractError(f"Active forecast card is missing columns: {', '.join(missing)}")
    if card.empty:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "skipped": True,
            "reason": "empty forecast card",
        }
    if card["game_id"].duplicated().any():
        raise DataContractError("Active forecast card contains duplicate games")

    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Active forecast card has games without a kickoff timestamp")

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger=CHALLENGER_ID)

    season = int(card["season"].iloc[0])
    week = int(card["week"].iloc[0])
    week_lock = sunday_pick_lock(kickoffs)
    deadlines = kickoffs.map(lambda kickoff: pick_deadline(kickoff, week_lock))

    feature_table = data_root / "processed" / "game_features_weak_stack.parquet"
    features = pd.read_parquet(feature_table)
    features["gameday"] = pd.to_datetime(features["gameday"], errors="raise", utc=True)
    pool = prior_pool(features)
    totals = features.loc[:, ["game_id", "total_line"]].copy()
    totals["game_id"] = totals["game_id"].astype(str)
    totals["total_line"] = pd.to_numeric(totals["total_line"], errors="coerce")
    pool = pool.merge(totals, on="game_id", how="left")
    pool["total_band"] = pool["total_line"].map(
        lambda value: total_band(float(value)) if pd.notna(value) else None
    )

    eligible = prior_pool_for_week(
        pool,
        season=season,
        week=week,
        cutoff=recorded_at,
        exclude_game_ids=set(card["game_id"].astype(str)),
    )
    eligible = eligible.loc[eligible["total_band"].notna()]
    if eligible.empty:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "skipped": True,
            "reason": "no prior total-conditioned pool games precede this week",
        }
    pool_line = eligible["line"].to_numpy(dtype=float)
    pool_margin = eligible["result"].to_numpy(dtype=float)
    pool_total_band = eligible["total_band"].to_numpy()
    prior_pool_rows = len(eligible)

    existing = load_decisions(artifacts_root)
    replaced_rows = 0
    left_post_kickoff = 0
    pre_kickoff = kickoffs.gt(recorded_at)
    if replace_week and bool(pre_kickoff.any()):
        existing, replaced_rows, left_post_kickoff = replace_week_rows(
            existing,
            ledger_path(artifacts_root),
            season=season,
            week=week,
            recorded_at=recorded_at,
            columns=LEDGER_COLUMNS,
            challenger_id=CHALLENGER_ID,
        )

    already = set(existing["game_id"].astype(str)) if not existing.empty else set()
    already_ids = card["game_id"].astype(str).isin(already)
    keep = pre_kickoff & ~already_ids

    rows: list[dict[str, Any]] = []
    missing_total_line = 0
    for idx in card.loc[keep].index:
        row = card.loc[idx]
        total_line_value = pd.to_numeric(row["total_line"], errors="coerce")
        if pd.isna(total_line_value):
            missing_total_line += 1
            continue
        spread_line = float(row["spread_line"])
        band_label = total_band(float(total_line_value))
        point = spread_line
        read, fallback = total_conditioned_read(
            pool_line, pool_margin, pool_total_band, band_label, spread_line, point
        )
        challenger_cover_probability = read.conditional_cover_probability
        served_cover_probability = float(row["home_cover_probability"])
        rows.append(
            {
                "recorded_at_utc": recorded_at,
                "challenger_id": CHALLENGER_ID,
                "challenger_policy": CHALLENGER_POLICY,
                "source_artifact": forecast.name,
                "source_sha256": sha256_file(card_path),
                "forecast_created_at_utc": pd.to_datetime(
                    metadata.get("created_at_utc"), utc=True, errors="coerce"
                ),
                "game_id": str(row["game_id"]),
                "season": season,
                "week": week,
                "kickoff": kickoffs.loc[idx],
                "deadline": deadlines.loc[idx],
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "spread_line": spread_line,
                "total_line": float(total_line_value),
                "total_band": band_label,
                "served_cover_probability": served_cover_probability,
                "served_pick_side": "HOME" if served_cover_probability >= 0.5 else "AWAY",
                "challenger_cover_probability": challenger_cover_probability,
                "challenger_pick_side": "HOME" if challenger_cover_probability >= 0.5 else "AWAY",
                "challenger_cover": read.cover,
                "challenger_push": read.push,
                "challenger_loss": read.loss,
                "challenger_band": read.band,
                "challenger_band_games": read.band_games,
                "challenger_fallback_to_unconditioned": bool(fallback),
                "prior_pool_rows": prior_pool_rows,
            }
        )

    already_recorded = int((pre_kickoff & already_ids).sum())
    post_kickoff_skipped = int((~pre_kickoff).sum())

    if not rows:
        return {
            "challenger_id": CHALLENGER_ID,
            "season": season,
            "week": week,
            "source_artifact": forecast.name,
            "recorded": 0,
            "already_recorded": already_recorded,
            "post_kickoff_skipped": post_kickoff_skipped,
            "missing_total_line_skipped": missing_total_line,
            "replaced_rows": replaced_rows,
            "left_post_kickoff": left_post_kickoff,
            "ledger_rows": len(existing),
        }

    fresh = pd.DataFrame(rows)
    combined = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
    atomic_parquet(combined[list(LEDGER_COLUMNS)], ledger_path(artifacts_root))
    return {
        "challenger_id": CHALLENGER_ID,
        "season": season,
        "week": week,
        "source_artifact": forecast.name,
        "recorded": len(fresh),
        "already_recorded": already_recorded,
        "post_kickoff_skipped": post_kickoff_skipped,
        "missing_total_line_skipped": missing_total_line,
        "replaced_rows": replaced_rows,
        "left_post_kickoff": left_post_kickoff,
        "ledger_rows": len(combined),
    }


__all__ = [
    "ATOM_TOLERANCE",
    "CHALLENGER_ID",
    "CHALLENGER_POLICY",
    "LEDGER_COLUMNS",
    "TOTAL_BAND_EDGES",
    "TOTAL_BAND_LABELS",
    "ledger_path",
    "load_decisions",
    "record_total_conditioned_lattice_decisions",
    "total_band",
    "total_conditioned_read",
]
