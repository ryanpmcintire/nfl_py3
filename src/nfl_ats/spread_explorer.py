from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.calibration import smoothed_home_cover_probability
from nfl_ats.data import DataContractError
from nfl_ats.key_line_pick_read import apply_pick_overrides
from nfl_ats.margin import _three_way_probabilities
from nfl_ats.mass_preserving_lattice import DiscretePushReader, residual_location
from nfl_ats.outcomes import fit_margin_models_for_week

SPREAD_EXPLORER_MIN_LINE = -20.0
SPREAD_EXPLORER_MAX_LINE = 20.0
SPREAD_EXPLORER_STEP = 0.5

_REQUIRED_PREDICTION_COLUMNS = frozenset(
    {
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "spread_line",
        "home_cover_probability",
    }
)


@dataclass(frozen=True)
class SpreadExplorerGameParams:
    game_id: str
    home_team: str
    away_team: str
    center: float
    residual_mean: float
    residual_std: float
    card_line: float
    card_home_cover_probability: float
    key_line_pinned: bool = False


def load_feature_table_for_forecast(metadata: Mapping[str, Any], data_root: Path) -> pd.DataFrame:

    provenance = metadata.get("provenance")
    feature_table = provenance.get("feature_table") if isinstance(provenance, dict) else None
    path_value = feature_table.get("path") if isinstance(feature_table, dict) else None
    if not path_value:
        raise DataContractError("Forecast metadata has no feature table path recorded")
    feature_path = Path(str(path_value))
    if not feature_path.is_file():
        feature_path = data_root / "processed" / feature_path.name
    if not feature_path.is_file():
        raise DataContractError(
            f"Feature table for the active forecast is not available locally: {feature_path}"
        )
    return pd.read_parquet(feature_path)


def compute_spread_explorer_params(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    regressor: str,
    ridge_alpha: float,
    feature_profile: str,
    min_train_games: int,
    probability_method: str = "gaussian",
    center_offsets: Mapping[str, float] | None = None,
    pick_overrides: Mapping[str, float] | None = None,
) -> dict[str, SpreadExplorerGameParams]:

    if probability_method not in ("gaussian", "gaussian_median"):
        raise DataContractError("Spread-explorer widget requires a Gaussian location estimator")
    missing = sorted(_REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise DataContractError(
            f"predictions is missing spread-explorer columns: {', '.join(missing)}"
        )
    if predictions.empty:
        return {}

    base = predictions.reset_index(drop=True).copy()
    base["game_id"] = base["game_id"].astype(str)

    params: dict[str, SpreadExplorerGameParams] = {}
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
                f"target universe: {', '.join(missing_games)} -- the feature table has "
                "likely drifted from the one that produced this card"
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

        gaussian_check = smoothed_home_cover_probability(
            model.residuals, centers, spread, method=probability_method
        )
        expected = apply_pick_overrides(gaussian_check, group_ids, pick_overrides)
        supplied = group["home_cover_probability"].to_numpy(dtype=float)
        if not np.allclose(expected, supplied, rtol=0.0, atol=1e-9):
            raise DataContractError(
                f"Refit Gaussian probabilities for season {season} week {week} do not "
                "reproduce the supplied card's home_cover_probability -- the feature "
                "table or configuration has drifted from the one that produced this "
                "card; refusing to build a spread-explorer widget that could disagree "
                "with the published pick"
            )

        mean = float(
            np.median(model.residuals)
            if probability_method == "gaussian_median"
            else np.mean(model.residuals)
        )
        std = float(np.std(model.residuals, ddof=1))
        rows_by_id = {str(row["game_id"]): row for _, row in group.iterrows()}
        for game_id, center, line, probability in zip(
            group_ids, centers, spread, supplied, strict=True
        ):
            row = rows_by_id[game_id]
            params[game_id] = SpreadExplorerGameParams(
                game_id=game_id,
                home_team=str(row["home_team"]),
                away_team=str(row["away_team"]),
                center=float(center),
                residual_mean=mean,
                residual_std=std,
                card_line=float(line),
                card_home_cover_probability=float(probability),
                key_line_pinned=bool(pick_overrides and game_id in pick_overrides),
            )
    return params


def _erf_abramowitz_stegun(x: float) -> float:

    sign = -1.0 if x < 0 else 1.0
    x = abs(x)
    a1, a2, a3, a4, a5, p = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
        0.3275911,
    )
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y


def widget_home_cover_probability(line: float, center: float, mean: float, std: float) -> float:

    threshold = line - center
    z = (threshold - mean) / (std * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + _erf_abramowitz_stegun(z))
    return 1.0 - cdf


def spread_explorer_payload(
    params: Mapping[str, SpreadExplorerGameParams],
) -> dict[str, dict[str, Any]]:

    payload: dict[str, dict[str, Any]] = {}
    for game_id, p in params.items():
        entry: dict[str, Any] = {
            "home": p.home_team,
            "away": p.away_team,
            "center": round(p.center, 6),
            "mean": round(p.residual_mean, 6),
            "std": round(p.residual_std, 6),
            "line": round(p.card_line, 3),
        }
        if p.key_line_pinned:
            entry["pinned"] = round(p.card_home_cover_probability, 6)
        payload[game_id] = entry
    return payload


@dataclass(frozen=True)
class SpreadExplorerGameDistribution:
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    center: float
    residuals: npt.NDArray[np.float64]
    card_line: float
    card_home_cover_probability: float
    card_probability_method: str
    key_line_pinned: bool = False


def compute_spread_explorer_distribution(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    game_id: str,
    regressor: str,
    ridge_alpha: float,
    feature_profile: str,
    min_train_games: int,
    probability_method: str = "gaussian",
    center_offsets: Mapping[str, float] | None = None,
    pick_overrides: Mapping[str, float] | None = None,
) -> SpreadExplorerGameDistribution:

    missing = sorted(_REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise DataContractError(
            f"predictions is missing spread-explorer columns: {', '.join(missing)}"
        )
    rows = predictions.loc[predictions["game_id"].astype(str) == str(game_id)]
    if rows.empty:
        raise DataContractError(f"Game {game_id!r} is not on the supplied predictions")
    if len(rows) > 1:
        raise DataContractError(
            f"Game {game_id!r} appears more than once in the supplied predictions"
        )
    row = rows.iloc[0]
    season = int(row["season"])
    week = int(row["week"])

    _target, margin_models = fit_margin_models_for_week(
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

    target = _target.copy()
    target["game_id"] = target["game_id"].astype(str)
    target_rows = target.loc[target["game_id"] == str(game_id)]
    if target_rows.empty:
        raise DataContractError(
            f"Refitting season {season} week {week} is missing game {game_id!r} -- the "
            "feature table has likely drifted from the one that produced this card"
        )

    predicted = model.predict(target_rows, probability_method=probability_method)  # type: ignore[arg-type]
    center = float(predicted["predicted_margin"].iloc[0])
    if center_offsets is not None:
        center += float(center_offsets.get(str(game_id), 0.0))
    line = float(target_rows["spread_line"].iloc[0])
    supplied = float(row["home_cover_probability"])

    check = float(
        smoothed_home_cover_probability(
            model.residuals,
            np.array([center]),
            np.array([line]),
            method=probability_method,  # type: ignore[arg-type]
        )[0]
    )
    pinned = bool(pick_overrides and str(game_id) in pick_overrides)
    if pinned:
        check = float(apply_pick_overrides([check], [str(game_id)], pick_overrides)[0])
    if not math.isclose(check, supplied, rel_tol=0.0, abs_tol=1e-9):
        raise DataContractError(
            f"Refit {probability_method!r} probability for season {season} week {week} game "
            f"{game_id!r} does not reproduce the supplied card's home_cover_probability -- the "
            "feature table or configuration has drifted from the one that produced this card; "
            "refusing to answer a spread query that could disagree with the published pick"
        )

    return SpreadExplorerGameDistribution(
        game_id=str(game_id),
        season=season,
        week=week,
        home_team=str(row["home_team"]),
        away_team=str(row["away_team"]),
        center=center,
        residuals=model.residuals,
        card_line=line,
        card_home_cover_probability=supplied,
        card_probability_method=probability_method,
        key_line_pinned=pinned,
    )


def spread_explorer_three_way(
    distribution: SpreadExplorerGameDistribution,
    line: float,
    discrete_read: DiscretePushReader | None = None,
) -> tuple[float, float, float]:

    if discrete_read is not None:
        point = distribution.center + residual_location(
            distribution.residuals, distribution.card_probability_method
        )
        return discrete_read.three_way(
            float(line), point, conditioning_line=float(distribution.card_line)
        )
    sample = np.asarray(distribution.center + distribution.residuals, dtype=np.float64)
    return _three_way_probabilities(sample, float(line))


__all__ = [
    "SPREAD_EXPLORER_MAX_LINE",
    "SPREAD_EXPLORER_MIN_LINE",
    "SPREAD_EXPLORER_STEP",
    "SpreadExplorerGameDistribution",
    "SpreadExplorerGameParams",
    "compute_spread_explorer_distribution",
    "compute_spread_explorer_params",
    "load_feature_table_for_forecast",
    "spread_explorer_payload",
    "spread_explorer_three_way",
    "widget_home_cover_probability",
]
