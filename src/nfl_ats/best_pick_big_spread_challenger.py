"""Prospective-only Best-Pick challenger that excludes 10+ point spreads.

``docs/opener_error_analysis.md`` measured the active production-rule read
against the opener and found its largest spread bucket (absolute opener spread
at least 10 points) below the unfiltered baseline.  The result remains
``unresolved_below_power``; this module does not claim otherwise.  It implements
the document's direct policy lead at no rotation-window cost by recording an
alternative weekly Best Pick prospectively.

This challenger composes one additional eligibility rule with v2, after v2's
below-median-dispersion pool is built: exclude candidates whose absolute
decision spread is at least ``BIG_SPREAD_THRESHOLD``.  Candidate probabilities,
primary ranking, and tie breaks are otherwise v2 byte-for-byte.  If every
v2-eligible game is a big spread, fall back to the unmodified v2 pool so the
forced weekly nomination is never dropped.

**Superseded on the played card, 2026-09-09.** The same mechanism at the
boundary ``docs/spread_hole_diagnosis.md`` actually locates -- exclude 7 points
and up -- is now the SERVED nomination rule
(``nfl_ats.best_pick_nomination.nominate_v2_small_spread``,
``docs/best_pick_bucket_confidence.md``), and both rules share this module's
former screen, now ``nfl_ats.best_pick_nomination.apply_spread_eligibility``.
This 10-point arm keeps recording unchanged so its prospective history is not
lost; it is no longer the only spread-screened nominator.

Only the separate prospective challenger ledger is written here.  No published
prediction, side, probability, or ``is_best_pick`` flag is changed by this
module.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.best_pick_nomination import (
    NominationV2Result,
    apply_spread_eligibility,
    nominate_v2,
)
from nfl_ats.best_pick_nomination import (
    SpreadEligibilityResult as BigSpreadNominationResult,
)
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.margin import MarginFeatureProfile
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

CHALLENGER_ID = "best_pick_big_spread_eligibility"
BIG_SPREAD_THRESHOLD = 10.0


def apply_big_spread_eligibility(
    predictions: pd.DataFrame,
    base: NominationV2Result,
    *,
    threshold: float = BIG_SPREAD_THRESHOLD,
) -> BigSpreadNominationResult:
    """This challenger's 10-point screen, on the shared eligibility primitive."""

    return apply_spread_eligibility(predictions, base, threshold=threshold)


def nominate_big_spread_challenger(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    market_root: Path,
    season: int,
    week: int,
    regressor: str,
    feature_profile: MarginFeatureProfile,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> BigSpreadNominationResult | None:
    """Nominate from v2's pool after excluding absolute spreads of 10+."""

    base = nominate_v2(
        predictions,
        features,
        market_root=market_root,
        season=season,
        week=week,
        regressor=regressor,
        feature_profile=feature_profile,
        min_train_games=min_train_games,
    )
    if base is None:
        return None
    return apply_big_spread_eligibility(predictions, base)


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_big_spread_nomination_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
    forecast_artifact: str | None = None,
    replace_week: bool = False,
) -> dict[str, Any]:
    """Append exactly one prospective-only challenger nominee for the week.

    This mirrors the established v2/v3 Best-Pick recorder contracts: active
    registration and configuration fingerprint required, whole week must be
    pre-kickoff, recording must be within ``RECORDING_LOCK_WINDOW``, and an
    existing challenger/game row is never rewritten.
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
        raise ValueError("No synchronized active ATS model is available to record from")
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
            "changed underneath this nomination rule -- re-register before recording"
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
        raise DataContractError("Active forecast metadata carries no feature_table path")
    features = pd.read_parquet(feature_table_path)
    season = int(card["season"].iloc[0])
    week = int(card["week"].iloc[0])
    min_train_games = metadata.get("min_train_games")
    nomination = nominate_big_spread_challenger(
        card,
        features,
        market_root=data_root / "market" / "raw",
        season=season,
        week=week,
        regressor=str(metadata.get("regressor", "ridge")),
        feature_profile=metadata.get("feature_profile"),
        min_train_games=int(min_train_games) if min_train_games else DEFAULT_MIN_TRAIN_GAMES,
    )
    nominee_id = nomination.game_id if nomination is not None else None

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    whole_week_pre_kickoff = bool(kickoffs.gt(recorded_at).all())
    existing = load_challenger_decisions(artifacts_root)
    replaced_rows = 0
    left_post_kickoff = 0
    if replace_week and nominee_id is not None and whole_week_pre_kickoff:
        existing, replaced_rows, left_post_kickoff = replace_week_rows(
            existing,
            challenger_ledger_path(artifacts_root),
            season=season,
            week=week,
            recorded_at=recorded_at,
            columns=CHALLENGER_DECISION_COLUMNS,
            challenger_id=CHALLENGER_ID,
        )
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID)]
    already_ids = set(mine["game_id"].astype(str))
    already = nominee_id is not None and nominee_id in already_ids
    post_kickoff_skipped = nominee_id is not None and not already and not whole_week_pre_kickoff
    if nominee_id is None or already or not whole_week_pre_kickoff:
        fresh = card.iloc[0:0]
    else:
        fresh = card.loc[card["game_id"].astype(str).eq(nominee_id)]

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
        "season": season,
        "week": week,
        "source_artifact": forecast.name,
        "config_fingerprint": observed_fingerprint,
        "nominated_game_id": nominee_id,
        "base_v2_game_id": nomination.base_v2_game_id if nomination is not None else None,
        "excluded_game_ids": list(nomination.excluded_game_ids) if nomination is not None else [],
        "fallback_to_v2": nomination.fallback_to_v2 if nomination is not None else False,
        "recorded": len(decisions),
        "replaced_rows": replaced_rows,
        "left_post_kickoff": left_post_kickoff,
        "already_recorded": int(already),
        "post_kickoff_skipped": int(post_kickoff_skipped),
        "ledger_rows": int(ledger_rows),
    }
