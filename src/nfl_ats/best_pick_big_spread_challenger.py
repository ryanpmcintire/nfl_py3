"""Prospective-only Best-Pick challenger that excludes 10+ point spreads.

``docs/opener_error_analysis.md`` measured the active production-rule read
against the opener and found its largest spread bucket (absolute opener spread
at least 10 points) below the unfiltered baseline.  The result remains
``unresolved_below_power``; this module does not claim otherwise.  It implements
the document's direct policy lead at no rotation-window cost by recording an
alternative weekly Best Pick prospectively.

The incumbent published nomination remains v2.  This challenger composes one
additional eligibility rule with v2, after v2's below-median-dispersion pool is
built: exclude candidates whose absolute decision spread is at least
``BIG_SPREAD_THRESHOLD``.  Candidate probabilities, primary ranking, and tie
breaks are otherwise v2 byte-for-byte.  If every v2-eligible game is a big
spread, fall back to the unmodified v2 pool so the forced weekly nomination is
never dropped.

Only the separate prospective challenger ledger is written.  Nothing in this
module is imported by ``nfl_ats.publishing`` or ``nfl_ats.card_view``, and no
published prediction, side, probability, or ``is_best_pick`` flag is changed.
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
from nfl_ats.best_pick_nomination import NominationV2Result, nominate_v2, select_nominee
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

CHALLENGER_ID = "best_pick_big_spread_eligibility"
BIG_SPREAD_THRESHOLD = 10.0


@dataclass(frozen=True)
class BigSpreadNominationResult:
    """The challenger nominee and an auditable copy of its eligibility table."""

    game_id: str
    n_tied_at_max: int
    tie_break: str
    probability_table: pd.DataFrame
    excluded_game_ids: tuple[str, ...]
    fallback_to_v2: bool
    base_v2_game_id: str


def apply_big_spread_eligibility(
    predictions: pd.DataFrame,
    base: NominationV2Result,
    *,
    threshold: float = BIG_SPREAD_THRESHOLD,
) -> BigSpreadNominationResult:
    """Apply the predeclared spread screen to an already-computed v2 result.

    ``spread_line`` is the card's frozen decision-line input.  The transform
    uses its absolute magnitude only; it does not read outcomes, closing lines,
    post-kickoff data, or even the card's pick side/probability.
    """

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Big-spread eligibility threshold must be finite and positive")
    required_predictions = {"game_id", "spread_line"}
    missing_predictions = sorted(required_predictions.difference(predictions.columns))
    if missing_predictions:
        raise DataContractError(
            "Best-Pick big-spread challenger is missing card columns: "
            f"{', '.join(missing_predictions)}"
        )
    if predictions["game_id"].astype(str).duplicated().any():
        raise DataContractError("Best-Pick big-spread challenger card contains duplicate games")

    required_table = {"game_id", "candidate_dist", "spread_std", "pool_pass"}
    missing_table = sorted(required_table.difference(base.probability_table.columns))
    if missing_table:
        raise DataContractError(
            f"Best-Pick v2 probability table is missing columns: {', '.join(missing_table)}"
        )

    spreads = predictions[["game_id", "spread_line"]].copy()
    spreads["game_id"] = spreads["game_id"].astype(str)
    spreads["spread_line"] = pd.to_numeric(spreads["spread_line"], errors="coerce")
    if not np.isfinite(spreads["spread_line"].to_numpy(dtype=float)).all():
        raise DataContractError(
            "Best-Pick big-spread challenger found a non-finite decision spread"
        )

    table = base.probability_table.copy()
    table["game_id"] = table["game_id"].astype(str)
    table = table.merge(spreads, on="game_id", how="left", validate="one_to_one")
    if len(table) != len(spreads) or table["spread_line"].isna().any():
        raise DataContractError("Best-Pick big-spread spread join dropped or duplicated games")

    v2_pool = table["pool_pass"].astype(bool)
    if not bool(v2_pool.any()):
        raise DataContractError("Best-Pick v2 eligibility pool contains no candidates")
    table["big_spread_pass"] = table["spread_line"].abs().lt(threshold)
    challenger_pool = v2_pool & table["big_spread_pass"]
    fallback_to_v2 = not bool(challenger_pool.any())
    final_pool = v2_pool if fallback_to_v2 else challenger_pool

    nominee, n_tied, tie_break = select_nominee(table.loc[final_pool])
    excluded = tuple(
        sorted(table.loc[v2_pool & ~table["big_spread_pass"], "game_id"].astype(str).tolist())
    )
    return BigSpreadNominationResult(
        game_id=nominee,
        n_tied_at_max=n_tied,
        tie_break=tie_break,
        probability_table=table.sort_values("game_id").reset_index(drop=True),
        excluded_game_ids=excluded,
        fallback_to_v2=fallback_to_v2,
        base_v2_game_id=base.game_id,
    )


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
        "already_recorded": int(already),
        "post_kickoff_skipped": int(post_kickoff_skipped),
        "ledger_rows": int(ledger_rows),
    }
