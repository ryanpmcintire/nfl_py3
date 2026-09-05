"""``weak_stack_qb_revenge_deadline_drag`` prospective challenger (lane T).

``docs/promotion_eval_20260905.md`` measured this stacked profile (production
``weak_stack`` plus BOTH ``qb_revenge_flag`` and ``deadline_integration_drag_flag``)
on the full reused 2020-2025 opener archive: +0.0665 accuracy points,
week-blocked P+ 0.5336, season-blocked P+ 0.588 -- barely favourable, while
its own two components individually read AGAINST the candidate on that same
population (qb_revenge P+ 0.342, deadline_drag P+ 0.251), and the 2026
Week-1 card impact was exactly zero. The coordinator's decision is
**do-not-promote on SELECTION grounds**: the stacked arm is the best of
three correlated arms drawn from one reused window, both of its own
components read against it on the identical population, and no live pick
changes this week either way -- NOT because ``probability_positive`` sits
below some threshold (a promotion bar is not a decision bar, AGENTS.md).
Since the archive read is confounded by multiplicity, the clean way to keep
testing this candidate is prospective 2026 evidence, which costs no
rotation-registry window at all (the same path
``docs/player_arrests_policy_eval.md`` and ``mod07_weak_signal_stack``
already recommend/use for exactly this situation).

This module is the no-window-cost path, registered in
``artifacts/prospective/challengers.json`` under
:data:`CHALLENGER_ID`, pinned to the active model's own configuration
fingerprint (``config_fingerprint``, computed from a SNAPSHOT of the active
model's own recipe -- identical in role to every sibling challenger's
``model`` block, e.g. ``mod07_weak_signal_stack``, ``division_revenge_tilt_overlay``,
``best_pick_nomination_v3``: it exists only to detect "did the active model
I ride on top of change under me", never to describe THIS challenger's own
distinct recipe).

**Not a pick-level post-prediction tilt** (unlike ``division_revenge_tilt_overlay``
/ the other ``*_tilt_overlay`` modules): this challenger genuinely REFITS its
own ``weak_stack_qb_revenge_deadline_drag`` margin model each week, walk-forward,
via ``nfl_ats.outcomes.fit_margin_models_for_week`` -- the same public entry
point ``nfl_ats.best_pick_nomination.fit_candidate_probabilities`` uses for
its own alpha=2000 candidate, and the same leak-safe training cutoff
``nfl_ats.outcomes.score_outcome_week`` (production's own weekly-forecast
entry point) uses. It differs from ``mod07_weak_signal_stack`` (the OTHER
genuinely-retrained challenger) in one respect: it never needs a separate
``margin-predict`` artifact of its own, because the two extra columns
(``qb_revenge_flag``, ``deadline_integration_drag_flag``) are cheap to attach
onto the active model's OWN base feature table at record time
(:func:`nfl_ats.qb_identity_features.attach_qb_revenge_features` /
:func:`nfl_ats.transaction_flag_features.attach_deadline_integration_drag_features`,
both already leakage-tested, both reading only newest local snapshots --
never a network fetch), exactly as ``scripts/promotion_eval_20260905.py``
built the same two-column table for the archive look. **Declared deviation
from the tilt-overlay/nomination ledger convention**: every sibling
challenger's ``feature_profile`` ledger column is a literal copy of the
ACTIVE model's own metadata field (because their own internal fit, if any,
reuses the SAME feature_profile as the active model). This challenger
genuinely fits a DIFFERENT profile, so its ledger rows record
:data:`CANDIDATE_FEATURE_PROFILE` instead -- copying the active's "weak_stack"
value here would misdescribe what actually produced the row.

**Nothing here is wired into ``publishing.py`` or the production pick
path.** No owner decision to play this on the real card has been made; it
is dual-tracked only, exactly like every ``*_tilt_overlay``/``*_nomination``
sibling.
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
from nfl_ats.transaction_flag_features import attach_deadline_integration_drag_features

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "weak_stack_qb_revenge_deadline_drag"

#: The profile this challenger's OWN weekly refit actually uses -- distinct
#: from the active model's registered "weak_stack" (see module docstring's
#: "Declared deviation" note).
CANDIDATE_FEATURE_PROFILE: MarginFeatureProfile = "weak_stack_qb_revenge_deadline_drag"


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def build_stacked_features(base_features: pd.DataFrame) -> pd.DataFrame:
    """Attach BOTH candidate columns onto the active model's own base table.

    Mirrors ``scripts/promotion_eval_20260905.py``'s ``build_combined_features``
    exactly: both attach functions default to the newest local schedule/
    roster/combine/transaction-wire/snap-count snapshots, so no precomputed
    candidate-specific parquet is read or written.
    """

    with_revenge = attach_qb_revenge_features(base_features)
    return attach_deadline_integration_drag_features(with_revenge)


def record_qb_revenge_deadline_drag_stack_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append this week's stacked-candidate picks to the prospective ledger.

    Refuses (raising ``ValueError``/``DataContractError``, caught by the
    ``publish-predictions --record-decisions`` try/except chain exactly like
    every sibling challenger) unless: the challenger is registered
    ``ACTIVE_PROSPECTIVE``; a synchronized active model with a linked weekly
    forecast exists; that forecast's own configuration fingerprint still
    matches what this challenger was registered against (an active-model
    promotion under this challenger's feet must not silently convert into
    "prospective evidence" for the stack); and the forecast card carries
    every column this recorder needs.

    ``bet_side`` is always ``"PASS"`` and ``edge`` is always NaN, mirroring
    every sibling challenger's identical rationale: this tracks the
    candidate's forced-pick accuracy only, never a fabricated paper-bet
    edge. ``decision_home_spread`` is the SAME market line the active
    model's own card was graded at (``spread_line``), so both arms are
    compared at one shared decision line, never re-picked at a different
    price.
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
    "record_qb_revenge_deadline_drag_stack_challenger_decisions",
]
