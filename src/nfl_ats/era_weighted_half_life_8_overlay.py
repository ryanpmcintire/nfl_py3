from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.calibration import ResidualSmoothingMethod, smoothed_home_cover_probability
from nfl_ats.card_refit import CardRefit, load_card_refit
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.margin import (
    MarginFeatureProfile,
    MarginModel,
    make_margin_estimator,
    margin_feature_columns,
)
from nfl_ats.modeling import regular_season_rows
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

CHALLENGER_ID = "era_weighted_half_life_8"

HALF_LIFE_SEASONS = 8.0

_REQUIRED_PREDICTION_COLUMNS = frozenset(
    {"game_id", "season", "week", "home_team", "away_team", "spread_line", "home_cover_probability"}
)


def half_life_weights(
    seasons: npt.NDArray[np.float64], predict_season: int, half_life: float
) -> npt.NDArray[np.float64]:

    if half_life <= 0.0:
        raise ValueError("half_life must be positive")
    elapsed = np.clip(predict_season - seasons.astype(np.float64), 0.0, None)
    return np.power(0.5, elapsed / half_life)


def fit_weighted_ridge_margin(
    sorted_frame: pd.DataFrame,
    *,
    target: npt.NDArray[np.float64],
    feature_columns: tuple[str, ...],
    weights: npt.NDArray[np.float64],
    ridge_alpha: float = 10.0,
    distribution_fraction: float = 0.20,
    min_distribution_rows: int = 10,
    min_rows: int = 50,
    random_state: int = 42,
    model_name: str = "ridge",
) -> MarginModel:

    if len(sorted_frame) < min_rows:
        raise ValueError(f"At least {min_rows} completed games are required to fit")
    if len(sorted_frame) != len(target) or len(sorted_frame) != len(weights):
        raise ValueError("sorted_frame, target, and weights must be the same length")
    if not 0.10 <= distribution_fraction < 0.5:
        raise ValueError("distribution_fraction must be in [0.10, 0.5)")

    distribution_rows = int(len(sorted_frame) * distribution_fraction)
    if distribution_rows < min_distribution_rows or len(sorted_frame) - distribution_rows < 40:
        raise ValueError("Not enough rows for an out-of-time residual distribution")
    split = len(sorted_frame) - distribution_rows

    columns = list(feature_columns)
    temporary = make_margin_estimator(model_name, random_state, ridge_alpha=ridge_alpha)
    temporary.fit(
        sorted_frame.iloc[:split].loc[:, columns],
        target[:split],
        regressor__sample_weight=weights[:split],
    )
    calibration_prediction = np.asarray(
        temporary.predict(sorted_frame.iloc[split:].loc[:, columns]), dtype=float
    )
    residuals = np.asarray(target[split:] - calibration_prediction, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < min_distribution_rows:
        raise ValueError("Out-of-time residual distribution has too few finite values")

    estimator = make_margin_estimator(model_name, random_state, ridge_alpha=ridge_alpha)
    estimator.fit(
        sorted_frame.loc[:, columns],
        target,
        regressor__sample_weight=weights,
    )
    return MarginModel(
        estimator=estimator,
        residuals=residuals,
        model_name=model_name,
        ridge_alpha=ridge_alpha if model_name == "ridge" else None,
        target="market_residual",
        feature_columns=feature_columns,
        training_rows=len(sorted_frame),
        distribution_rows=len(residuals),
        training_max_gameday=pd.to_datetime(sorted_frame["gameday"]).max().date().isoformat(),
    )


def _target_values(frame: pd.DataFrame) -> pd.Series:

    return pd.to_numeric(frame["ats_margin"], errors="coerce")


def _leak_safe_training_frame(
    features: pd.DataFrame, *, season: int, week: int
) -> tuple[pd.DataFrame, pd.DataFrame]:

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    target = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)].copy()
    if target.empty:
        raise ValueError(f"No games found for {season} week {week}")
    cutoff = target["gameday"].min()
    training = regular_season_rows(frame)
    training = training.loc[training["gameday"].lt(cutoff) & training["result"].notna()].copy()
    return target, training


def _prepare_sorted_training(training: pd.DataFrame) -> pd.DataFrame:

    prepared = training.loc[_target_values(training).notna()].copy()
    prepared["gameday"] = pd.to_datetime(prepared["gameday"], errors="raise")
    prepared = prepared.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return prepared


@dataclass(frozen=True)
class EraWeightedFlip:
    game_id: str
    matchup: str
    from_side: str
    to_side: str
    baseline_probability: float
    era_weighted_probability: float


@dataclass(frozen=True)
class EraWeightedResult:
    overlaid_predictions: pd.DataFrame
    flips: tuple[EraWeightedFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_era_weighted_half_life_8_overlay(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    regressor: str = "ridge",
    ridge_alpha: float = 10.0,
    feature_profile: MarginFeatureProfile = "weak_stack",
    min_train_games: int = 500,
    half_life: float = HALF_LIFE_SEASONS,
    card_refit: CardRefit | None = None,
    enabled: bool = True,
) -> EraWeightedResult:

    missing = sorted(_REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled or base.empty:
        return EraWeightedResult(base, (), enabled)

    base["game_id"] = base["game_id"].astype(str)
    feature_columns = margin_feature_columns("market_residual", feature_profile)
    era_weighted_probability_by_game: dict[str, float] = {}

    for _, group in base.groupby(["season", "week"], sort=True):
        season = int(group["season"].iloc[0])
        week = int(group["week"].iloc[0])
        target, training = _leak_safe_training_frame(features, season=season, week=week)
        if len(training) < min_train_games:
            raise DataContractError(
                f"Only {len(training)} eligible games precede season {season} week {week}; "
                f"need {min_train_games} to refit the era-weighted arm"
            )
        sorted_frame = _prepare_sorted_training(training)
        target_values = _target_values(sorted_frame).to_numpy(dtype=float)

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
        spread = aligned["spread_line"].to_numpy(dtype=float)

        uniform_weights = np.ones(len(sorted_frame), dtype=float)
        uniform_model = fit_weighted_ridge_margin(
            sorted_frame,
            target=target_values,
            feature_columns=feature_columns,
            weights=uniform_weights,
            ridge_alpha=ridge_alpha,
            model_name=regressor,
        )
        uniform_predicted = (
            card_refit.predict(uniform_model, aligned)
            if card_refit is not None
            else uniform_model.predict(aligned)
        )
        probability_method: ResidualSmoothingMethod = (
            card_refit.probability_method if card_refit is not None else "gaussian"
        )
        uniform_centers = uniform_predicted["predicted_margin"].to_numpy(dtype=float)
        uniform_check = (
            uniform_predicted["home_cover_probability"].to_numpy(dtype=float)
            if card_refit is not None
            else smoothed_home_cover_probability(
                uniform_model.residuals, uniform_centers, spread, method=probability_method
            )
        )
        supplied = group["home_cover_probability"].to_numpy(dtype=float)
        if not np.allclose(uniform_check, supplied, rtol=0.0, atol=1e-9):
            raise DataContractError(
                f"Uniform-weight refit for season {season} week {week} does not "
                "reproduce the supplied card's home_cover_probability -- the feature "
                "table or configuration has drifted from the one that produced this "
                "card, so the half-life-8 refit would not be a like-for-like comparison"
            )

        seasons_arr = sorted_frame["season"].to_numpy(dtype=float)
        weights = half_life_weights(seasons_arr, predict_season=season, half_life=half_life)
        weighted_model = fit_weighted_ridge_margin(
            sorted_frame,
            target=target_values,
            feature_columns=feature_columns,
            weights=weights,
            ridge_alpha=ridge_alpha,
            model_name=regressor,
        )
        weighted_predicted = (
            card_refit.predict(weighted_model, aligned)
            if card_refit is not None
            else weighted_model.predict(aligned)
        )
        weighted_centers = weighted_predicted["predicted_margin"].to_numpy(dtype=float)
        weighted_probability = smoothed_home_cover_probability(
            weighted_model.residuals, weighted_centers, spread, method=probability_method
        )
        for game_id, probability in zip(group_ids, weighted_probability, strict=True):
            era_weighted_probability_by_game[game_id] = float(probability)

    overlaid = base.copy()
    overlaid["home_cover_probability"] = base["game_id"].map(era_weighted_probability_by_game)
    if overlaid["home_cover_probability"].isna().any():  # pragma: no cover - defensive
        raise DataContractError("Every game must receive a re-weighted probability")
    overlaid["home_cover_probability"] = overlaid["home_cover_probability"].astype(float)

    flips: list[EraWeightedFlip] = []
    for _, row in base.iterrows():
        game_id = str(row["game_id"])
        original_probability = float(row["home_cover_probability"])
        mapped_probability = era_weighted_probability_by_game[game_id]
        original_side = "HOME" if original_probability >= 0.5 else "AWAY"
        mapped_side = "HOME" if mapped_probability >= 0.5 else "AWAY"
        if mapped_side != original_side:
            flips.append(
                EraWeightedFlip(
                    game_id=game_id,
                    matchup=f"{row['away_team']} at {row['home_team']}",
                    from_side=original_side,
                    to_side=mapped_side,
                    baseline_probability=original_probability,
                    era_weighted_probability=mapped_probability,
                )
            )

    return EraWeightedResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: EraWeightedResult) -> str:

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.from_side} -> {flip.to_side}" for flip in result.flips
    )
    return (
        f"**Era-weighted (half-life 8) refit applied: {result.flip_count} pick{plural} "
        "flipped** (the active recipe refit with exponential season-decay sample "
        "weights, half-life 8 seasons, MOD-14's selected arm). "
        f"{detail}. See docs/era_weighting_screen.md. Prospective evidence only -- not "
        "applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_era_weighted_half_life_8_challenger_decisions(
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
            "No synchronized active ATS model is available to record era-weighted decisions from"
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

    feature_table = observed_config.get("feature_table")
    if not feature_table:
        raise ValueError("Active forecast metadata has no feature table path recorded")
    feature_path = Path(str(feature_table))
    if not feature_path.is_file():
        feature_path = data_root / "processed" / feature_path.name
    if not feature_path.is_file():
        raise ValueError(f"Feature table for the active model is not built yet: {feature_path}")
    features = pd.read_parquet(feature_path)

    card_refit = load_card_refit(metadata, card, forecast)
    result = apply_era_weighted_half_life_8_overlay(
        card,
        features,
        regressor=str(observed_config.get("regressor")),
        ridge_alpha=float(observed_config.get("ridge_alpha", 10.0)),
        feature_profile=str(observed_config.get("feature_profile")),  # type: ignore[arg-type]
        min_train_games=int(observed_config.get("min_train_games", 500)),
        card_refit=card_refit,
    )
    reweighted_card = result.overlaid_predictions

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
    already = card["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    keep = pre_kickoff & ~already
    fresh = reweighted_card.loc[keep]

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
        "replaced_rows": replaced_rows,
        "left_post_kickoff": left_post_kickoff,
        "ledger_rows": int(ledger_rows),
        "flip_count": result.flip_count,
        "warnings": list(card_refit.warnings),
        "flipped_game_ids": [flip.game_id for flip in result.flips],
    }
