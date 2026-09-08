"""Uncorrected point read, paired against the promoted home-side offset.

MOD-18 lane S promotion (2026-09-07, docs/home_side_offset_promotion.md): the
served ``market_residual`` point now carries a walk-forward home-side offset
by spread bucket. This challenger tracks the read WITHOUT that offset -- the
model exactly as it was served before the promotion -- so the 2026 season
scores both arms on the same games at no rotation-window cost.

Unlike the mapping incumbents it never refits: ``margin-predict`` writes both
reads per game into ``home_side_offset.json`` beside the card, and this
recorder reads the uncorrected probability from that sidecar verbatim. It
refuses a forecast without the sidecar (a pre-promotion card carries no
paired read to record) and a fingerprint drift, exactly like its siblings.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.data import DataContractError
from nfl_ats.home_side_location import HOME_SIDE_OFFSET_FILENAME, load_forecast_home_side_offsets
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

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "home_side_offset_off_incumbent"


def uncorrected_card(card: pd.DataFrame, sidecar: dict[str, Any]) -> pd.DataFrame:
    """``card`` with ``home_cover_probability`` replaced by the uncorrected read.

    Every game on the card must appear in the sidecar; a missing game is a
    contract failure, never a silent zero, because the whole point of the
    pair is that both arms scored the same games.
    """

    games = sidecar.get("games")
    if not isinstance(games, list):
        raise DataContractError(f"{HOME_SIDE_OFFSET_FILENAME} carries no per-game reads")
    by_game: dict[str, float] = {}
    for row in games:
        if not isinstance(row, dict):
            continue
        value = row.get("home_cover_probability_uncorrected")
        if value is None:
            continue
        by_game[str(row.get("game_id"))] = float(value)
    ids = card["game_id"].astype(str)
    missing = sorted(set(ids) - set(by_game))
    if missing:
        raise DataContractError(
            f"{HOME_SIDE_OFFSET_FILENAME} lacks the uncorrected read for: {', '.join(missing)}"
        )
    result = card.copy()
    result["home_cover_probability"] = ids.map(by_game).astype(float).to_numpy()
    return result


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_home_side_offset_incumbent_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the uncorrected read's forced picks to the prospective challenger ledger.

    ``data_root`` is accepted for signature parity with the other recorders
    (the lock-day rehearsal calls every recorder the same way); this one
    needs only the active forecast directory.
    """

    del data_root
    entry = find_challenger(artifacts_root, CHALLENGER_ID)
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {CHALLENGER_ID!r} is registered as {status!r}; only "
            f"{ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError("No synchronized active ATS model is available to record decisions from")
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
            f"{forecast} was produced with {observed_fingerprint}; the active model "
            "changed underneath this pair -- re-register before recording"
        )

    sidecar = load_forecast_home_side_offsets(forecast)
    if sidecar is None or not sidecar.get("served"):
        raise DataContractError(
            f"Active forecast {forecast.name} carries no served {HOME_SIDE_OFFSET_FILENAME}; "
            "a card produced without the home-side offset has no paired read to record"
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

    paired = uncorrected_card(card, sidecar)
    served_home = pd.to_numeric(card["home_cover_probability"], errors="coerce").ge(0.5)
    paired_home = pd.to_numeric(paired["home_cover_probability"], errors="coerce").ge(0.5)
    flipped = card.loc[served_home.ne(paired_home), "game_id"].astype(str).tolist()

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID)]
    already = card["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    keep = pre_kickoff & ~already
    fresh = paired.loc[keep]

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
        "ledger_rows": int(ledger_rows),
        "flip_count": len(flipped),
        "flipped_game_ids": flipped,
    }
