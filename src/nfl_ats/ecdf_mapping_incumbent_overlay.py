"""ECDF-mapping-incumbent overlay: tracking the FORMER production probability
read as a challenger, after MOD-08's promotion (2026-08-19,
``docs/smooth_cdf_mapping.md``).

Before this promotion, ``smooth_cdf_mapping_overlay.py`` tracked the Gaussian
CDF read as a challenger against the (then-ECDF-native) published card. The
opener-grade decision measurement in ``docs/smooth_cdf_mapping.md``
(``probability_positive`` 0.5536, production pick rule, week-blocked) fired
the frozen promotion rule: the Gaussian read is now
``nfl_ats.outcomes.score_outcome_week``'s own default, so the published card
IS the Gaussian arm. ``smooth_cdf_mapping_overlay.py`` is retired (its
``smooth_cdf_mapping`` challenger registration is marked superseded in
``artifacts/prospective/challengers.json``, not deleted) rather than reused
here, because its verification step assumes the supplied card reproduces the
ECDF control -- exactly backwards now that the card IS the Gaussian arm; a
retuned in-place version could silently start comparing against the wrong
control the day it stops raising.

This module is the mirror image: it refits the active card's EXACT recipe
(same feature profile, regressor, ridge alpha, ``min_train_games``, target
``market_residual``) via ``nfl_ats.outcomes.fit_margin_models_for_week`` --
the same leak-safe training cutoff ``score_outcome_week`` used -- and
requires the refit's **Gaussian** read to reproduce the card's own
``home_cover_probability`` to floating-point precision before trusting
anything (proof this is reading the SAME residual draws the now-Gaussian
card was built from, not a drifted reimplementation). It then replaces
``home_cover_probability`` everywhere with the **ECDF** read of that same
sample -- the production mapping every published card used before this
promotion -- so the former incumbent keeps accruing dual-tracked,
no-rotation-registry-window-cost 2026 prospective evidence against its
successor, exactly as the other overlay challengers do
(``docs/prospective_evidence.md``).

``bet_side`` is always ``"PASS"`` and ``edge`` is always NaN: this challenger
tracks the ECDF read's forced-pick (``decision_line``) accuracy only, never a
fabricated paper-bet edge for the pre-promotion side.
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
from nfl_ats.calibration import smoothed_home_cover_probability
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
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

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "ecdf_mapping_incumbent"

_REQUIRED_PREDICTION_COLUMNS = frozenset(
    {"game_id", "season", "week", "home_team", "away_team", "spread_line", "home_cover_probability"}
)


@dataclass(frozen=True)
class EcdfMappingIncumbentFlip:
    """One game whose forced pick moved sides under the ECDF read."""

    game_id: str
    matchup: str
    from_side: str
    to_side: str
    gaussian_probability: float
    ecdf_probability: float


@dataclass(frozen=True)
class EcdfMappingIncumbentResult:
    """The overlay's effect on one or more weeks' cards.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on every row (the ECDF read replaces the
    Gaussian read for every game, not only flipped ones) -- every other
    column stays byte-identical, mirroring
    ``smooth_cdf_mapping_overlay.SmoothCdfMappingResult``.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[EcdfMappingIncumbentFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_ecdf_mapping_incumbent_overlay(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    regressor: str = "ridge",
    ridge_alpha: float = 10.0,
    feature_profile: str = "weak_stack",
    min_train_games: int = 500,
    enabled: bool = True,
) -> EcdfMappingIncumbentResult:
    """Replace ``home_cover_probability`` with the ECDF read of the same
    out-of-time residual sample the (post-promotion) production Gaussian read
    reads.

    ``predictions`` may span more than one (season, week) group; each group is
    refit independently with a training cutoff strictly before that week's
    earliest kickoff, exactly as ``score_outcome_week`` does for the real
    card. Every group's refit GAUSSIAN probability is required to reproduce
    the supplied ``home_cover_probability`` to floating-point precision --
    this is the module's proof that it is reading the SAME residual draws the
    card was built from, not a drifted reimplementation, and it raises
    ``DataContractError`` rather than silently comparing against a moved
    target if the feature table, configuration, or (pre-2026-08-19) mapping
    default has changed underneath it.
    """

    missing = sorted(_REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled or base.empty:
        return EcdfMappingIncumbentResult(base, (), enabled)

    base["game_id"] = base["game_id"].astype(str)
    ecdf_probability_by_game: dict[str, float] = {}

    for _, group in base.groupby(["season", "week"], sort=True):
        season = int(group["season"].iloc[0])
        week = int(group["week"].iloc[0])
        target, margin_models = fit_margin_models_for_week(
            features,
            season=season,
            week=week,
            regressor=regressor,
            min_train_games=min_train_games,
            feature_profile=feature_profile,  # type: ignore[arg-type]
            ridge_alpha=ridge_alpha,
            methods=("market_residual",),
        )
        model = margin_models["market_residual"]

        target = target.copy()
        target["game_id"] = target["game_id"].astype(str)
        if target["game_id"].duplicated().any():
            raise DataContractError(
                f"Refitting season {season} week {week} produced duplicate game IDs "
                "in the target universe"
            )
        target_indexed = target.set_index("game_id", drop=False)

        group_ids = group["game_id"].tolist()
        missing_games = sorted(set(group_ids).difference(target_indexed.index))
        if missing_games:
            raise DataContractError(
                f"Refitting season {season} week {week} is missing games from the "
                f"target universe: {', '.join(missing_games)} -- the feature table "
                "has likely drifted from the one that produced this card"
            )

        aligned = target_indexed.loc[group_ids]
        predicted = model.predict(aligned)
        centers = predicted["predicted_margin"].to_numpy(dtype=float)
        spread = aligned["spread_line"].to_numpy(dtype=float)

        gaussian_check = smoothed_home_cover_probability(
            model.residuals, centers, spread, method="gaussian"
        )
        supplied = group["home_cover_probability"].to_numpy(dtype=float)
        if not np.allclose(gaussian_check, supplied, rtol=0.0, atol=1e-9):
            raise DataContractError(
                f"Refit Gaussian probabilities for season {season} week {week} do not "
                "reproduce the supplied card's home_cover_probability -- either the "
                "feature table/configuration has drifted from the one that produced "
                "this card, or the active card is not the post-promotion Gaussian "
                "mapping this challenger expects (see docs/smooth_cdf_mapping.md)"
            )

        ecdf_probability = smoothed_home_cover_probability(
            model.residuals, centers, spread, method="ecdf"
        )
        for game_id, probability in zip(group_ids, ecdf_probability, strict=True):
            ecdf_probability_by_game[game_id] = float(probability)

    overlaid = base.copy()
    overlaid["home_cover_probability"] = base["game_id"].map(ecdf_probability_by_game)
    if overlaid["home_cover_probability"].isna().any():  # pragma: no cover - defensive
        raise DataContractError("Every game must receive a mapped probability")
    overlaid["home_cover_probability"] = overlaid["home_cover_probability"].astype(float)

    flips: list[EcdfMappingIncumbentFlip] = []
    for _, row in base.iterrows():
        game_id = str(row["game_id"])
        original_probability = float(row["home_cover_probability"])
        mapped_probability = ecdf_probability_by_game[game_id]
        original_side = "HOME" if original_probability >= 0.5 else "AWAY"
        mapped_side = "HOME" if mapped_probability >= 0.5 else "AWAY"
        if mapped_side != original_side:
            flips.append(
                EcdfMappingIncumbentFlip(
                    game_id=game_id,
                    matchup=f"{row['away_team']} at {row['home_team']}",
                    from_side=original_side,
                    to_side=mapped_side,
                    gaussian_probability=original_probability,
                    ecdf_probability=mapped_probability,
                )
            )

    return EcdfMappingIncumbentResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: EcdfMappingIncumbentResult) -> str:
    """Plain-language provenance sentence, mirroring
    ``smooth_cdf_mapping_overlay.overlay_disclosure_note``.

    Empty when the overlay is off or changed no picks this week. Not
    currently surfaced on the published card -- this overlay is dual-tracked
    only.
    """

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.from_side} -> {flip.to_side}" for flip in result.flips
    )
    return (
        f"**ECDF-mapping-incumbent overlay applied: {result.flip_count} pick{plural} "
        "flipped** (the raw empirical-CDF read every published card used before the "
        f"2026-08-19 MOD-08 promotion, off the SAME out-of-time residual sample). "
        f"{detail}. See docs/smooth_cdf_mapping.md. Prospective evidence only -- not "
        "applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_ecdf_mapping_incumbent_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the ECDF-incumbent overlay's picks to the prospective challenger
    ledger.

    Mirrors
    ``smooth_cdf_mapping_overlay.record_smooth_cdf_mapping_challenger_decisions``
    exactly for the write-path guarantees: this is not a retrained model with
    its own ``margin-predict`` artifact -- its "model" IS the active model's
    own recipe, refit and read differently -- so it reads the active model's
    own synchronized weekly forecast rather than searching
    ``artifacts/margin_predictions/`` by fingerprint, and it refuses to
    record if the active model's live fingerprint no longer matches the
    snapshot this challenger was registered against (a promotion under this
    challenger's feet must not silently convert into "prospective evidence"
    for a different base model).
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
            "No synchronized active ATS model is available to record mapping decisions from"
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
            "changed underneath this mapping -- re-register before recording"
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

    feature_table = observed_config.get("feature_table")
    if not feature_table:
        raise ValueError("Active forecast metadata has no feature table path recorded")
    feature_path = Path(str(feature_table))
    if not feature_path.is_file():
        feature_path = data_root / "processed" / feature_path.name
    if not feature_path.is_file():
        raise ValueError(f"Feature table for the active model is not built yet: {feature_path}")
    features = pd.read_parquet(feature_path)

    mapping = apply_ecdf_mapping_incumbent_overlay(
        card,
        features,
        regressor=str(observed_config.get("regressor")),
        ridge_alpha=float(observed_config.get("ridge_alpha", 10.0)),
        feature_profile=str(observed_config.get("feature_profile")),
        min_train_games=int(observed_config.get("min_train_games", 500)),
    )
    mapped_card = mapping.overlaid_predictions

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID)]
    already = card["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    keep = pre_kickoff & ~already
    fresh = mapped_card.loc[keep]

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
        "flip_count": mapping.flip_count,
        "flipped_game_ids": [flip.game_id for flip in mapping.flips],
    }
