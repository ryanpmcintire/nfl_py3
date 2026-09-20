from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.data import DataContractError
from nfl_ats.pbp08_matchup_flags import build_flag_table
from nfl_ats.pbp08_protection_mismatch_tilt_overlay import latest_pbp_snapshot
from nfl_ats.pick_probability import (
    ACTIVE_PICK_PROBABILITY_FILENAME,
    BASE_PROBABILITY_POLICY,
    COEFFICIENTS_FILENAME,
    CONFIDENCE_BAND_EDGES,
    COUNTED_FLAG_COLUMNS,
    FLAG_SUM_COLUMN,
    MARKET_MOVE_FEATURE_LEGACY,
    MARKET_MOVE_FEATURE_SUNDAY,
    METADATA_FILENAME,
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
    PICK_PROBABILITY_ARTIFACT_ROOT,
    PICK_PROBABILITY_POLICY,
    PROBABILITY_EPSILON,
    SCHEMA_VERSION,
    STRENGTH_BAND_QUANTILES,
    STRENGTH_ROUNDING_PLACES,
    STRENGTH_WORDS,
    ConfidenceBand,
    PickProbabilityModel,
    PickProbabilitySourceError,
    StrengthBand,
    active_market_move_feature_version,
    signed_composition_flags,
)
from nfl_ats.public_board import find_matching_opener_evaluation
from nfl_ats.snapshots import latest_snapshot, load_snapshot

FIT_FEATURES = ("model_logit", FLAG_SUM_COLUMN, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN)
FIT_RIDGE = 1e-3
FIT_ITERATIONS = 50
MODEL_PROBABILITY_SOURCE = "home_cover_probability_at_open"
FORECAST_TEMP_ARCHIVE = "raw/forecast_archive/full_2020_2025/forecasts.parquet"
MARKET_MOVE_ARTIFACT_ROOT = "sharp_weighted_follow"
MARKET_MOVE_COLUMN = "leader_median_net"
SUNDAY_MARKET_MOVE_ARTIFACT = Path("sunday_market_probability/20260920_fixed/market_move.parquet")
SCHEDULE_COLUMNS = (
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
)


def _fit_logit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    for _ in range(FIT_ITERATIONS):
        z = np.clip(x @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        gradient = x.T @ (y - p) - ridge * beta
        hessian = (x.T * w) @ x + ridge * np.eye(x.shape[1])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta = beta + step
    return beta


def _design(frame: pd.DataFrame, means: dict[str, float], stds: dict[str, float]) -> np.ndarray:
    columns: list[Any] = [np.ones(len(frame))]
    for name in FIT_FEATURES:
        columns.append(((frame[name].astype(float) - means[name]) / stds[name]).to_numpy())
    return np.column_stack(columns)


def _standardisers(train: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    means = {name: float(train[name].mean()) for name in FIT_FEATURES}
    stds = {name: float(train[name].std(ddof=0)) or 1.0 for name in FIT_FEATURES}
    return means, stds


def _natural_coefficients(
    beta: np.ndarray, means: dict[str, float], stds: dict[str, float]
) -> dict[str, float]:
    natural = {name: float(beta[index + 1] / stds[name]) for index, name in enumerate(FIT_FEATURES)}
    intercept = float(beta[0])
    for index, name in enumerate(FIT_FEATURES):
        intercept -= float(beta[index + 1]) * means[name] / stds[name]
    natural["intercept"] = intercept
    return natural


def _predict(
    frame: pd.DataFrame, beta: np.ndarray, means: dict[str, float], stds: dict[str, float]
) -> np.ndarray:
    z = np.clip(_design(frame, means, stds) @ beta, -35.0, 35.0)
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=float)


def _newest_artifact_directory(root: Path, filename: str) -> Path | None:
    if not root.is_dir():
        return None
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir() and (path / filename).is_file()),
        key=lambda path: path.name,
    )
    return candidates[-1] if candidates else None


def _market_move_table(
    artifacts_root: Path, feature_version: str
) -> tuple[pd.DataFrame, str | None]:
    if feature_version == MARKET_MOVE_FEATURE_SUNDAY:
        path = artifacts_root / SUNDAY_MARKET_MOVE_ARTIFACT
        summary_path = path.parent / "summary.json"
        if not path.is_file() or not summary_path.is_file():
            raise PickProbabilitySourceError("Sunday market-move training artifact is missing")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("market_move_feature_version") != MARKET_MOVE_FEATURE_SUNDAY:
            raise PickProbabilitySourceError(
                "Sunday market-move training artifact has wrong version"
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary.get("market_move_sha256"):
            raise PickProbabilitySourceError("Sunday market-move training artifact hash differs")
        frame = pd.read_parquet(path)
        if frame.game_id.duplicated().any() or frame[MARKET_MOVE_COLUMN].isna().any():
            raise PickProbabilitySourceError("Sunday market-move training artifact is invalid")
        return frame[["game_id", MARKET_MOVE_COLUMN]], str(SUNDAY_MARKET_MOVE_ARTIFACT)
    if feature_version != MARKET_MOVE_FEATURE_LEGACY:
        raise PickProbabilitySourceError("Unsupported market-move training feature version")
    directory = _newest_artifact_directory(
        artifacts_root / MARKET_MOVE_ARTIFACT_ROOT, "per_game.parquet"
    )
    if directory is None:
        return pd.DataFrame(columns=["game_id", MARKET_MOVE_COLUMN]), None
    frame = pd.read_parquet(directory / "per_game.parquet")
    if MARKET_MOVE_COLUMN not in frame.columns:
        return pd.DataFrame(columns=["game_id", MARKET_MOVE_COLUMN]), str(directory.name)
    table = frame[["game_id", MARKET_MOVE_COLUMN]].copy()
    table["game_id"] = table["game_id"].astype(str)
    return table.drop_duplicates("game_id"), str(directory.name)


def _arrest_incidents(data_root: Path) -> pd.DataFrame:
    root = data_root / "raw" / "player_arrests"
    if not root.is_dir():
        return pd.DataFrame(columns=["record_id", "incident_date", "team"])
    directories = sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
    for directory in directories:
        path = directory / "incidents_point_in_time.parquet"
        if path.is_file():
            return pd.read_parquet(path, columns=["record_id", "incident_date", "team"])
    return pd.DataFrame(columns=["record_id", "incident_date", "team"])


def _forecast_temperatures(data_root: Path) -> pd.DataFrame:
    path = data_root / FORECAST_TEMP_ARCHIVE
    if not path.is_file():
        return pd.DataFrame(columns=["game_id", "forecast_temp_f"])
    return pd.read_parquet(path)[["game_id", "forecast_temp_f"]].copy()


def _protection_back_side(schedules: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    snapshot = latest_pbp_snapshot(data_root)
    if snapshot is None:
        return pd.DataFrame(columns=["game_id", "back_side"])
    scoped = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    try:
        table = build_flag_table(scoped, snapshot)
    except (OSError, ValueError, DataContractError):
        return pd.DataFrame(columns=["game_id", "back_side"])
    return table[["game_id", "back_side"]].copy()


def build_fit_population(
    artifacts_root: Path,
    data_root: Path,
    *,
    market_move_feature_version: str = MARKET_MOVE_FEATURE_LEGACY,
) -> tuple[pd.DataFrame, dict[str, Any]]:

    active = load_active_ats_model(artifacts_root)
    matched = find_matching_opener_evaluation(artifacts_root, active)
    if matched is None:
        raise PickProbabilitySourceError(
            "no opener evaluation matches the active model; re-run the opener evaluation "
            "before fitting the pick probability"
        )
    opener_metadata, opener_directory = matched
    per_game = pd.read_parquet(opener_directory / "per_game.parquet")
    required = {
        "game_id",
        "season",
        "week",
        "margin_vs_open",
        MODEL_PROBABILITY_SOURCE,
        "tue_open_home_spread",
    }
    missing = sorted(required.difference(per_game.columns))
    if missing:
        raise PickProbabilitySourceError(
            f"opener evaluation is missing columns for the fit: {', '.join(missing)}"
        )
    if (
        "base_probability_policy" not in per_game
        or not per_game["base_probability_policy"].eq(BASE_PROBABILITY_POLICY).all()
    ):
        raise PickProbabilitySourceError(
            "opener evaluation does not use the served discrete probability; "
            "re-run opener-evaluation before fitting the pick probability"
        )

    schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    schedule_columns = [column for column in SCHEDULE_COLUMNS if column in schedules.columns]
    joined = per_game.merge(
        schedules[schedule_columns].drop_duplicates("game_id"),
        on=["game_id", "season"],
        how="left",
        suffixes=("", "_schedule"),
    )
    if "week_schedule" in joined.columns:
        joined = joined.drop(columns=["week_schedule"])
    if joined["home_team"].isna().any():
        raise PickProbabilitySourceError(
            "opener evaluation rows have no matching schedule row; refresh the schedules snapshot"
        )

    margin = pd.to_numeric(joined["margin_vs_open"], errors="coerce")
    graded = joined.loc[margin.notna() & margin.ne(0.0)].reset_index(drop=True)
    pushes = int(margin.eq(0.0).sum())
    ungraded = int(margin.isna().sum())
    graded["home_covered"] = (
        pd.to_numeric(graded["margin_vs_open"], errors="coerce").gt(0.0).astype(float)
    )
    stated = pd.to_numeric(graded[MODEL_PROBABILITY_SOURCE], errors="coerce").clip(
        PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
    )
    graded["model_probability"] = stated
    graded["model_logit"] = np.log(stated / (1.0 - stated))
    graded["model_pick_home"] = stated.ge(0.5)
    graded["model_correct"] = (
        graded["model_pick_home"].astype(float).eq(graded["home_covered"]).astype(float)
    )

    incidents = _arrest_incidents(data_root)
    forecasts = _forecast_temperatures(data_root)
    protection = _protection_back_side(schedules, data_root)
    flags = signed_composition_flags(
        graded,
        schedules,
        incidents=incidents,
        forecasts_tuesday_noon=forecasts,
        protection_back_side=protection,
    )
    graded = graded.merge(flags, on="game_id", how="left", validate="one_to_one")

    market, market_artifact = _market_move_table(artifacts_root, market_move_feature_version)
    graded = graded.merge(market, on="game_id", how="left")
    raw_move = (
        pd.to_numeric(graded[MARKET_MOVE_COLUMN], errors="coerce")
        if MARKET_MOVE_COLUMN in graded.columns
        else pd.Series(np.nan, index=graded.index, dtype=float)
    )
    graded[MOVE_AVAILABLE_COLUMN] = raw_move.notna().astype(float)
    graded[MOVE_COLUMN] = raw_move.fillna(0.0)

    provenance = {
        "opener_evaluation": str(opener_directory.relative_to(artifacts_root)).replace("\\", "/"),
        "opener_evaluation_model_id": str(opener_metadata.get("active_model_id") or ""),
        "active_model_id": str((active or {}).get("model_id") or ""),
        "model_probability_source": MODEL_PROBABILITY_SOURCE,
        "base_probability_policy": BASE_PROBABILITY_POLICY,
        "market_move_artifact": market_artifact,
        "market_move_feature_version": market_move_feature_version,
        "graded_games": len(graded),
        "pushes_dropped": pushes,
        "ungraded_dropped": ungraded,
        "games_with_market_move": int(graded[MOVE_AVAILABLE_COLUMN].sum()),
        "seasons": sorted(int(value) for value in graded["season"].unique()),
        "counted_flag_columns": list(COUNTED_FLAG_COLUMNS),
        "arrest_incidents": len(incidents),
        "forecast_rows": len(forecasts),
        "protection_rows": len(protection),
    }
    return graded, provenance


def _confidence_bands(frame: pd.DataFrame, column: str) -> list[ConfidenceBand]:
    confidence = pd.to_numeric(frame[column], errors="coerce")
    picked_home = pd.to_numeric(frame[column], errors="coerce").ge(0.5)
    correct = picked_home.astype(float).eq(frame["home_covered"]).astype(float)
    shown = confidence.where(picked_home, 1.0 - confidence)
    bands: list[ConfidenceBand] = []
    for lower, upper in pairwise(CONFIDENCE_BAND_EDGES):
        mask = shown.ge(lower) & (shown.lt(upper) if upper < 1.0 else shown.le(1.0))
        games = int(mask.sum())
        accuracy = float(correct.loc[mask].mean()) if games else None
        bands.append(ConfidenceBand(lower=lower, upper=upper, games=games, accuracy=accuracy))
    return bands


def _strength_bands(frame: pd.DataFrame, column: str) -> list[StrengthBand]:
    probability = pd.to_numeric(frame[column], errors="coerce")
    picked_home = probability.ge(0.5)
    correct = picked_home.astype(float).eq(frame["home_covered"]).astype(float)
    shown = probability.where(picked_home, 1.0 - probability)
    lower, upper = np.quantile(shown.dropna().to_numpy(dtype=float), STRENGTH_BAND_QUANTILES)
    minimums = (
        0.5,
        round(float(lower), STRENGTH_ROUNDING_PLACES),
        round(float(upper), STRENGTH_ROUNDING_PLACES),
    )
    rounded = shown.round(STRENGTH_ROUNDING_PLACES)
    words = pd.Series(
        np.select(
            [rounded.ge(minimums[2]), rounded.ge(minimums[1])],
            [STRENGTH_WORDS[2], STRENGTH_WORDS[1]],
            default=STRENGTH_WORDS[0],
        ),
        index=rounded.index,
        dtype=object,
    )
    bands: list[StrengthBand] = []
    for index, word in enumerate(STRENGTH_WORDS):
        mask = words.eq(word)
        games = int(mask.sum())
        accuracy = float(correct.loc[mask].mean()) if games else None
        bands.append(
            StrengthBand(word=word, minimum=minimums[index], games=games, accuracy=accuracy)
        )
    return bands


def _probability_metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    selected = frame.loc[frame[column].notna()]
    if selected.empty:
        return {"games": 0}
    probability = selected[column].to_numpy(dtype=float).clip(1e-9, 1.0 - 1e-9)
    target = selected["home_covered"].to_numpy(dtype=float)
    return {
        "games": len(selected),
        "accuracy": float(np.mean((probability >= 0.5) == target)),
        "brier": float(np.mean((probability - target) ** 2)),
        "log_loss": float(
            -np.mean(target * np.log(probability) + (1.0 - target) * np.log1p(-probability))
        ),
    }


def fit_pick_probability(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
    activate: bool = True,
    market_move_feature_version: str | None = None,
) -> tuple[PickProbabilityModel, Path, dict[str, Any]]:

    if market_move_feature_version is None:
        market_move_feature_version = active_market_move_feature_version(artifacts_root)
    population, provenance = build_fit_population(
        artifacts_root,
        data_root,
        market_move_feature_version=market_move_feature_version,
    )
    if population.empty:
        raise PickProbabilitySourceError("the pick-probability fit population is empty")

    target = population["home_covered"].astype(float).to_numpy()
    means, stds = _standardisers(population)
    beta = _fit_logit(_design(population, means, stds), target, FIT_RIDGE)
    natural = _natural_coefficients(beta, means, stds)
    population["in_sample_home_probability"] = _predict(population, beta, means, stds)

    seasons = sorted(int(value) for value in population["season"].unique())
    out_of_season = pd.Series(np.nan, index=population.index, dtype=float)
    fold_coefficients: dict[str, dict[str, float]] = {}
    for held in seasons:
        train = population.loc[population["season"].ne(held)]
        test = population.loc[population["season"].eq(held)]
        if train.empty or test.empty:
            continue
        fold_means, fold_stds = _standardisers(train)
        fold_beta = _fit_logit(
            _design(train, fold_means, fold_stds),
            train["home_covered"].astype(float).to_numpy(),
            FIT_RIDGE,
        )
        out_of_season.loc[test.index] = _predict(test, fold_beta, fold_means, fold_stds)
        fold_coefficients[str(held)] = _natural_coefficients(fold_beta, fold_means, fold_stds)
    population["out_of_season_home_probability"] = out_of_season
    chronological = pd.Series(np.nan, index=population.index, dtype=float)
    chronological_coefficients: dict[str, dict[str, float]] = {}
    for held in seasons:
        train = population.loc[population["season"].lt(held)]
        test = population.loc[population["season"].eq(held)]
        if train["season"].nunique() < 2 or test.empty:
            continue
        if int(train["season"].max()) >= held:
            raise PickProbabilitySourceError("calibration training reaches its test season")
        fold_means, fold_stds = _standardisers(train)
        fold_beta = _fit_logit(
            _design(train, fold_means, fold_stds),
            train["home_covered"].astype(float).to_numpy(),
            FIT_RIDGE,
        )
        chronological.loc[test.index] = _predict(test, fold_beta, fold_means, fold_stds)
        chronological_coefficients[str(held)] = _natural_coefficients(
            fold_beta, fold_means, fold_stds
        )
    population["chronological_home_probability"] = chronological
    population["neutral_market_probability"] = 0.5
    scored = population.loc[population["out_of_season_home_probability"].notna()].copy()

    strength = tuple(_strength_bands(scored, "out_of_season_home_probability"))
    confidence = tuple(_confidence_bands(scored, "out_of_season_home_probability"))
    model_confidence = tuple(_confidence_bands(scored, "model_probability"))

    calibrated_pick_home = scored["out_of_season_home_probability"].ge(0.5)
    calibrated_correct = calibrated_pick_home.astype(float).eq(scored["home_covered"]).astype(float)
    records = {
        "calibrated_out_of_season": _record(calibrated_correct),
        "model_only": _record(scored["model_correct"]),
    }

    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    relative = f"{PICK_PROBABILITY_ARTIFACT_ROOT}/{stamp}"
    model = PickProbabilityModel(
        intercept=natural["intercept"],
        model_logit=natural["model_logit"],
        flag_sum=natural[FLAG_SUM_COLUMN],
        move_toward_home=natural[MOVE_COLUMN],
        move_available=natural[MOVE_AVAILABLE_COLUMN],
        strength_bands=strength,
        confidence_bands=confidence,
        fitted_games=len(population),
        fitted_seasons=tuple(seasons),
        artifact=relative,
        policy=PICK_PROBABILITY_POLICY,
        base_probability_policy=BASE_PROBABILITY_POLICY,
        market_move_feature_version=market_move_feature_version,
    )

    directory = artifacts_root / PICK_PROBABILITY_ARTIFACT_ROOT / stamp
    directory.mkdir(parents=True, exist_ok=True)
    population.to_parquet(directory / "per_game.parquet", index=False)
    (directory / COEFFICIENTS_FILENAME).write_text(
        json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "policy": PICK_PROBABILITY_POLICY,
        "created_at_utc": (now or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "command": "nfl-ats fit-pick-probability",
        "features": list(FIT_FEATURES),
        "ridge": FIT_RIDGE,
        "records": records,
        "fold_coefficients": fold_coefficients,
        "chronological_coefficients": chronological_coefficients,
        "prediction_artifact": "per_game.parquet",
        "activated": activate,
        "validation_metrics": {
            column: _probability_metrics(population, column)
            for column in (
                "model_probability",
                "neutral_market_probability",
                "in_sample_home_probability",
                "out_of_season_home_probability",
                "chronological_home_probability",
            )
        },
        "validation_limitations": (
            "Features were selected using these seasons; this is not an untouched outer test. "
            "Weekly ranking uncertainty is assessed separately from average calibration."
        ),
        "model_only_confidence_bands": [band.to_dict() for band in model_confidence],
        **provenance,
    }
    (directory / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    pointer = {
        "activated_at_utc": (now or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "artifact": relative,
        "policy": PICK_PROBABILITY_POLICY,
        "schema_version": SCHEMA_VERSION,
        "fitted_games": len(population),
        "fitted_seasons": seasons,
        "active_model_id": provenance["active_model_id"],
        "base_probability_policy": BASE_PROBABILITY_POLICY,
        "market_move_feature_version": market_move_feature_version,
    }
    (directory / "activation.json").write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if activate:
        from nfl_ats.io import atomic_text

        atomic_text(
            json.dumps(pointer, indent=2, sort_keys=True) + "\n",
            artifacts_root / ACTIVE_PICK_PROBABILITY_FILENAME,
        )
    return model, directory, metadata


def _record(correct: pd.Series) -> str:
    wins = int(pd.to_numeric(correct, errors="coerce").fillna(0.0).sum())
    return f"{wins}-{len(correct) - wins}"


__all__ = [
    "FIT_FEATURES",
    "FIT_RIDGE",
    "MODEL_PROBABILITY_SOURCE",
    "build_fit_population",
    "fit_pick_probability",
]
