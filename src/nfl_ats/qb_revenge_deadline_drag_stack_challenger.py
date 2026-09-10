from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.card_refit import load_card_refit
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.margin import MarginFeatureProfile
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
from nfl_ats.qb_identity_features import attach_qb_revenge_features
from nfl_ats.recorder_override import replace_week_rows, resolve_recording_forecast
from nfl_ats.transaction_flag_features import attach_deadline_integration_drag_features

CHALLENGER_ID = "weak_stack_qb_revenge_deadline_drag"

CANDIDATE_FEATURE_PROFILE: MarginFeatureProfile = "weak_stack_qb_revenge_deadline_drag"


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def build_stacked_features(base_features: pd.DataFrame) -> pd.DataFrame:

    with_revenge = attach_qb_revenge_features(base_features)
    return attach_deadline_integration_drag_features(with_revenge)


def record_qb_revenge_deadline_drag_stack_challenger_decisions(
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
            "No synchronized active ATS model is available to record stacked-candidate "
            "decisions from"
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
            "changed underneath this challenger -- re-register before recording"
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

    feature_table_path = observed_config.get("feature_table")
    if not feature_table_path:
        raise DataContractError(
            "Active forecast metadata carries no feature_table path to refit the "
            "stacked candidate from"
        )
    base_features = pd.read_parquet(feature_table_path)
    features = build_stacked_features(base_features)

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
    card_refit = load_card_refit(metadata, card, forecast)
    predicted = card_refit.predict(model, target, replay_served_pick=False)
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
        "replaced_rows": replaced_rows,
        "left_post_kickoff": left_post_kickoff,
        "ledger_rows": int(ledger_rows),
        "picks_differing_from_active": picks_differing_from_active,
        "warnings": list(card_refit.warnings),
    }


__all__ = [
    "CANDIDATE_FEATURE_PROFILE",
    "CHALLENGER_ID",
    "build_stacked_features",
    "record_qb_revenge_deadline_drag_stack_challenger_decisions",
]
