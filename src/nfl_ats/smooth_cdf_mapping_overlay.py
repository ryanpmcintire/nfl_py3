from __future__ import annotations

import json
from collections.abc import Mapping
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
from nfl_ats.home_side_location import center_offsets_from_metadata
from nfl_ats.io import atomic_parquet
from nfl_ats.key_line_pick_read import apply_pick_overrides, pick_overrides_from_metadata
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

CHALLENGER_ID = "smooth_cdf_mapping"

_REQUIRED_PREDICTION_COLUMNS = frozenset(
    {"game_id", "season", "week", "home_team", "away_team", "spread_line", "home_cover_probability"}
)


@dataclass(frozen=True)
class SmoothCdfMappingFlip:
    game_id: str
    matchup: str
    from_side: str
    to_side: str
    ecdf_probability: float
    gaussian_probability: float


@dataclass(frozen=True)
class SmoothCdfMappingResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[SmoothCdfMappingFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_smooth_cdf_mapping_overlay(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    regressor: str = "ridge",
    ridge_alpha: float = 10.0,
    feature_profile: str = "weak_stack",
    min_train_games: int = 500,
    enabled: bool = True,
    center_offsets: Mapping[str, float] | None = None,
    pick_overrides: Mapping[str, float] | None = None,
) -> SmoothCdfMappingResult:

    missing = sorted(_REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled or base.empty:
        return SmoothCdfMappingResult(base, (), enabled)

    base["game_id"] = base["game_id"].astype(str)
    gaussian_probability_by_game: dict[str, float] = {}

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
        if center_offsets is not None:
            centers = centers + np.asarray(
                [float(center_offsets.get(str(game_id), 0.0)) for game_id in group_ids],
                dtype=float,
            )
        spread = aligned["spread_line"].to_numpy(dtype=float)

        ecdf_check = smoothed_home_cover_probability(
            model.residuals, centers, spread, method="ecdf"
        )
        ecdf_check = apply_pick_overrides(ecdf_check, group_ids, pick_overrides)
        supplied = group["home_cover_probability"].to_numpy(dtype=float)
        if not np.allclose(ecdf_check, supplied, rtol=0.0, atol=1e-9):
            raise DataContractError(
                f"Refit ECDF probabilities for season {season} week {week} do not "
                "reproduce the supplied card's home_cover_probability -- the feature "
                "table or configuration has drifted from the one that produced this "
                "card, so a Gaussian comparison against it would not be reading the "
                "same residual draws"
            )

        gaussian_probability = smoothed_home_cover_probability(
            model.residuals, centers, spread, method="gaussian"
        )
        for game_id, probability in zip(group_ids, gaussian_probability, strict=True):
            gaussian_probability_by_game[game_id] = float(probability)

    overlaid = base.copy()
    overlaid["home_cover_probability"] = base["game_id"].map(gaussian_probability_by_game)
    if overlaid["home_cover_probability"].isna().any():  # pragma: no cover - defensive
        raise DataContractError("Every game must receive a mapped probability")
    overlaid["home_cover_probability"] = overlaid["home_cover_probability"].astype(float)

    flips: list[SmoothCdfMappingFlip] = []
    for _, row in base.iterrows():
        game_id = str(row["game_id"])
        original_probability = float(row["home_cover_probability"])
        mapped_probability = gaussian_probability_by_game[game_id]
        original_side = "HOME" if original_probability >= 0.5 else "AWAY"
        mapped_side = "HOME" if mapped_probability >= 0.5 else "AWAY"
        if mapped_side != original_side:
            flips.append(
                SmoothCdfMappingFlip(
                    game_id=game_id,
                    matchup=f"{row['away_team']} at {row['home_team']}",
                    from_side=original_side,
                    to_side=mapped_side,
                    ecdf_probability=original_probability,
                    gaussian_probability=mapped_probability,
                )
            )

    return SmoothCdfMappingResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: SmoothCdfMappingResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.from_side} -> {flip.to_side}" for flip in result.flips
    )
    return (
        f"**Smooth CDF mapping applied: {result.flip_count} pick{plural} flipped** "
        "(a Gaussian read of the same out-of-time residual sample the production "
        f"ECDF reads). {detail}. See docs/smooth_cdf_mapping.md. Prospective "
        "evidence only -- not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_smooth_cdf_mapping_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
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

    mapping = apply_smooth_cdf_mapping_overlay(
        card,
        features,
        center_offsets=center_offsets_from_metadata(metadata, card),
        pick_overrides=pick_overrides_from_metadata(metadata),
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
