"""LEAD-62: production plus frozen expected lineup loss, paired prospectively."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from nfl_ats import margin, outcomes
from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.data import DataContractError
from nfl_ats.expected_lineup_loss_features import (
    EXPECTED_LINEUP_LOSS_COLUMNS,
    attach_expected_lineup_loss_features,
)
from nfl_ats.io import atomic_parquet
from nfl_ats.low_total_div_home_dog_challenger import record_paired_overlay_arms
from nfl_ats.margin import MarginFeatureProfile
from nfl_ats.outcomes import fit_margin_models_for_week
from nfl_ats.play_probability import (
    attach_history_features,
    canonicalize_depth_chart_history,
    depth_rank_bucket,
)
from nfl_ats.players import latest_player_snapshot, load_player_snapshot
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
from nfl_ats.quarterbacks import latest_depth_snapshot, load_depth_snapshot

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "weak_stack_expected_lineup_loss"

#: The profile this challenger's OWN weekly refit actually uses -- distinct
#: from the active model's registered "weak_stack" (see module docstring's
#: "Declared deviation" note).
CANDIDATE_FEATURE_PROFILE = cast(MarginFeatureProfile, "weak_stack_expected_lineup_loss")


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


@contextmanager
def candidate_profile() -> Iterator[None]:
    """Extend feature sets only for this refit, restoring them on every exit."""
    original = margin.margin_feature_set

    def feature_set(target: Any, feature_profile: Any = "base") -> str:
        if feature_profile == CANDIDATE_FEATURE_PROFILE:
            return "full_lead62" if target == "market_residual" else "football_lead62"
        return original(target, feature_profile)

    additions = {
        f"{prefix}_lead62": FEATURE_SETS[f"{prefix}_weak_stack"] + EXPECTED_LINEUP_LOSS_COLUMNS
        for prefix in ("football", "full")
    }
    with (
        patch.dict(FEATURE_SETS, additions),
        patch.object(margin, "margin_feature_set", feature_set),
        patch.object(outcomes, "margin_feature_set", feature_set),
        patch.object(
            outcomes,
            "MARGIN_FEATURE_PROFILES",
            (*outcomes.MARGIN_FEATURE_PROFILES, CANDIDATE_FEATURE_PROFILE),
        ),
    ):
        yield


def visible_sources(
    panel: pd.DataFrame, injuries: pd.DataFrame, now: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Publication time additionally bounds each game's pool decision cutoff."""
    observed = pd.to_datetime(
        panel.get("depth_observed_at", pd.Series(pd.NaT, index=panel.index)), utc=True
    )
    decision = pd.to_datetime(panel["decision_at"], utc=True)
    legacy = panel["source_schema"].eq("legacy_week") & decision.lt(now)
    visible = legacy | (observed.lt(decision) & observed.le(now))
    visible &= observed.isna() | (observed.lt(decision) & observed.le(now))
    panel = panel.loc[visible].copy()
    injury_time = pd.to_datetime(injuries["effective_observed_at"], utc=True, errors="coerce")
    return panel, injuries.loc[injury_time.le(now)].copy()


def build_stacked_features(
    base_features: pd.DataFrame, data_root: Path, now: pd.Timestamp, card: pd.DataFrame
) -> pd.DataFrame:
    """Build historical losses and current starters from local predecision inputs."""
    panel = pd.read_parquet(data_root / "processed/play_probability_panel.parquet")
    snapshot = latest_player_snapshot(data_root / "players/raw")
    injuries, rosters, snaps = load_player_snapshot(snapshot, include_postseason=False)
    depth = load_depth_snapshot(latest_depth_snapshot(data_root / "quarterbacks/depth/raw"))
    depth = depth.loc[pd.to_datetime(depth["dt"], utc=True).le(now)].copy()
    if depth.empty:
        raise FileNotFoundError("No lineup observations are available before publication")
    schedule = card.copy()
    if "game_type" not in schedule:
        schedule["game_type"] = "REG"
    local_kickoff = pd.to_datetime(schedule["kickoff"], utc=True).dt.tz_convert("America/New_York")
    schedule["gameday"] = local_kickoff.dt.strftime("%Y-%m-%d")
    schedule["gametime"] = local_kickoff.dt.strftime("%H:%M")
    current = canonicalize_depth_chart_history(depth, schedule)
    current["depth_rank_bucket"] = current["depth_rank"].map(depth_rank_bucket)
    current["season_week"] = current["week"].astype(float)
    current = attach_history_features(current, rosters, snaps)
    panel = pd.concat([panel, current], ignore_index=True)
    panel, injuries = visible_sources(panel, injuries, now)
    features = attach_expected_lineup_loss_features(base_features, panel=panel, injuries=injuries)
    current_features = features.loc[features["game_id"].isin(card["game_id"])]
    if (
        len(current_features) != len(card)
        or current_features[list(EXPECTED_LINEUP_LOSS_COLUMNS)].isna().any().any()
    ):
        raise FileNotFoundError("A current lineup is unavailable before the decision cutoff")
    return features


def record_expected_lineup_loss_challenger_decisions(
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
            "reason": "Lineup information is unavailable before the decision cutoff",
            "source_error": str(error),
        }

    season = int(card["season"].iloc[0])
    week = int(card["week"].iloc[0])
    min_train_games = metadata.get("min_train_games")
    with candidate_profile():
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
    "record_expected_lineup_loss_challenger_decisions",
]
