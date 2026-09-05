"""LEAD-23: production plus trade-deadline integration drag, paired prospectively."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.low_total_div_home_dog_challenger import record_paired_overlay_arms
from nfl_ats.margin import MarginFeatureProfile
from nfl_ats.nfl_week import pool_decision_cutoff
from nfl_ats.outcomes import fit_margin_models_for_week
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
from nfl_ats.transaction_flag_features import (
    attach_deadline_integration_drag_features,
    default_schedule,
    default_snap_counts,
    default_transactions_index,
)

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "weak_stack_deadline_drag"

#: The candidate refits production plus the deadline flag alone.
CANDIDATE_FEATURE_PROFILE: MarginFeatureProfile = "weak_stack_deadline_drag"


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def visible_transactions(
    transactions: pd.DataFrame,
    cutoff: pd.Timestamp,
    now: pd.Timestamp,
    snapshot_time: pd.Timestamp,
    prospective_season: int,
) -> pd.DataFrame:
    """Use observed times live; disclose month-end availability for legacy history.

    Sitemap last-modified values are contaminated and never used as observation
    times. Untimestamped current-season rows are known only at snapshot capture.
    """
    years = pd.to_numeric(transactions["url_year"], errors="raise").astype(int)
    months = pd.to_numeric(transactions["url_month"], errors="raise").astype(int)
    proxy = pd.to_datetime({"year": years, "month": months, "day": 1}, utc=True)
    proxy = proxy + pd.offsets.MonthBegin(1)
    observed = proxy.where(years.lt(prospective_season), snapshot_time)
    if "observed_at_utc" in transactions:
        observed = pd.to_datetime(transactions["observed_at_utc"], utc=True, errors="coerce")
    return transactions.loc[observed.lt(cutoff) & observed.le(now)].copy()


def build_stacked_features(
    base_features: pd.DataFrame, data_root: Path, now: pd.Timestamp, card: pd.DataFrame
) -> pd.DataFrame:
    """Attach only transactions visible before each game's pool decision."""
    paths = sorted((data_root / "raw/pfr_transactions").glob("*/index.parquet"))
    if not paths:
        raise FileNotFoundError("Transaction information is unavailable")
    snapshot_time = pd.Timestamp(
        datetime.strptime(paths[-1].parent.name, "%Y%m%dT%H%M%SZ"), tz="UTC"
    )
    if snapshot_time > now:
        raise DataContractError("Transaction snapshot is future-dated")
    transactions = default_transactions_index(snapshot=paths[-1])
    schedule = default_schedule(data_root.parent)
    snaps = default_snap_counts(data_root.parent)
    features = base_features.drop(
        columns=["deadline_integration_drag_flag"], errors="ignore"
    ).copy()
    kickoffs = pd.to_datetime(features["kickoff"], utc=True, errors="coerce")
    if kickoffs.isna().any():
        raise DataContractError("Feature table has games without a kickoff timestamp")
    cutoffs = kickoffs.map(pool_decision_cutoff)
    pieces = []
    for cutoff, group in features.groupby(cutoffs, sort=False):
        season_transactions = transactions.loc[
            pd.to_numeric(transactions["url_year"]).isin(group["season"])
        ]
        visible = visible_transactions(
            season_transactions, cutoff, now, snapshot_time, int(card["season"].min())
        )
        pieces.append(
            attach_deadline_integration_drag_features(
                group,
                schedule=schedule,
                transactions_index=visible,
                snap_counts=snaps,
            )
        )
    return pd.concat(pieces, ignore_index=True)


def record_deadline_drag_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Freeze candidate and baseline forced picks at the same decision spread.

    The active recipe is fingerprint-pinned. Missing local inputs skip; future
    forecasts refuse. Historical fits remain chronological, and no active
    model, forecast or production feature profile is changed.
    """

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
            "No synchronized active ATS model is available to record stacked-candidate "
            "decisions from"
        )
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
            "changed underneath this challenger -- re-register before recording"
        )

    card = pd.read_csv(card_path)
    if card.empty:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "skipped": True,
            "reason": "No games are available",
        }
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

    feature_table_path = observed_config.get("feature_table")
    if not feature_table_path:
        raise DataContractError(
            "Active forecast metadata carries no feature_table path to refit the "
            "stacked candidate from"
        )
    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    created = pd.to_datetime(metadata.get("created_at_utc"), utc=True, errors="coerce")
    if pd.isna(created) or created > recorded_at:
        raise DataContractError("Weekly forecast is future-dated or has no creation time")
    try:
        base_features = pd.read_parquet(feature_table_path)
        if "kickoff" in base_features:
            training_kickoffs = pd.to_datetime(base_features["kickoff"], utc=True)
            base_features = base_features.loc[
                training_kickoffs.lt(recorded_at) | base_features["game_id"].isin(card["game_id"])
            ].copy()
        for column in ("spread_line", "total_line"):
            if column in card:
                mapped = base_features["game_id"].map(card.set_index("game_id")[column])
                base_features.loc[mapped.notna(), column] = mapped.loc[mapped.notna()]
        features = build_stacked_features(base_features, data_root, recorded_at, card)
    except FileNotFoundError as error:
        return {
            "challenger_id": CHALLENGER_ID,
            "recorded": 0,
            "skipped": True,
            "reason": "Transaction information is unavailable before the decision cutoff",
            "source_error": str(error),
        }

    season = int(card["season"].iloc[0])
    week = int(card["week"].iloc[0])
    min_train_games = metadata.get("min_train_games")
    target, margin_models = fit_margin_models_for_week(
        features,
        season=season,
        week=week,
        regressor=str(metadata.get("regressor", "ridge")),
        min_train_games=int(min_train_games) if min_train_games else DEFAULT_MIN_TRAIN_GAMES,
        feature_profile=CANDIDATE_FEATURE_PROFILE,
        ridge_alpha=float(metadata.get("ridge_alpha", 10.0)),
        methods=("market_residual",),
    )
    model = margin_models["market_residual"]
    # probability_method="gaussian" matches nfl_ats.outcomes.score_outcome_week
    # -- production's own weekly-forecast entry point -- so the candidate's
    # probability is computed the same way the active card's own would be.
    predicted = model.predict(target, probability_method="gaussian")
    candidate = pd.DataFrame(
        {
            "game_id": target["game_id"].astype(str).to_numpy(),
            "candidate_home_cover_probability": predicted["home_cover_probability"].to_numpy(),
        }
    )

    card_with_id = card.copy()
    card_with_id["game_id"] = card_with_id["game_id"].astype(str)
    scored = card_with_id.merge(candidate, on="game_id", how="left", validate="one_to_one")
    if scored["candidate_home_cover_probability"].isna().any():
        raise DataContractError(
            "Stacked candidate model produced no prediction for one or more card games"
        )

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID)]
    already = scored["game_id"].isin(set(mine["game_id"].astype(str)))
    keep = pre_kickoff & ~already
    fresh = scored.loc[keep]

    active_pick_home = pd.to_numeric(scored["home_cover_probability"], errors="coerce").ge(0.5)
    candidate_pick_home = scored["candidate_home_cover_probability"].ge(0.5)
    picks_differing_from_active = int((active_pick_home != candidate_pick_home).sum())

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
            "feature_profile": CANDIDATE_FEATURE_PROFILE,
            "feature_table_sha256": str(observed_config.get("feature_table_sha256")),
            "game_id": fresh["game_id"].astype(str),
            "season": fresh["season"].astype(int),
            "week": fresh["week"].astype(int),
            "kickoff": kickoffs.loc[fresh.index],
            "away_team": fresh["away_team"].astype(str),
            "home_team": fresh["home_team"].astype(str),
            "pick_side": np.where(
                fresh["candidate_home_cover_probability"].ge(0.5), "HOME", "AWAY"
            ).astype(str),
            "bet_side": "PASS",
            "decision_home_spread": spreads.loc[fresh.index].astype(float),
            "edge": np.nan,
        }
    )
    if not decisions.empty:
        record_paired_overlay_arms(artifacts_root, CHALLENGER_ID, decisions, card)
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
        "season": season,
        "week": week,
        "source_artifact": forecast.name,
        "config_fingerprint": observed_fingerprint,
        "recorded": len(decisions),
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "ledger_rows": int(ledger_rows),
        "picks_differing_from_active": picks_differing_from_active,
    }


__all__ = [
    "CANDIDATE_FEATURE_PROFILE",
    "CHALLENGER_ID",
    "build_stacked_features",
    "record_deadline_drag_challenger_decisions",
]
