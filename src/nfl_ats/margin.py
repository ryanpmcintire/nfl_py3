"""Fair-margin and market-residual models with empirical predictive distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_ats.constants import FEATURE_SETS
from nfl_ats.data import DataContractError
from nfl_ats.odds import no_vig_probabilities

MarginTarget = Literal["margin", "market_residual"]
MarginFeatureProfile = Literal[
    "base",
    "pbp",
    "pbp_adjusted",
    "drive",
    "graph",
    "player_qb",
    "player_injuries",
    "player_continuity",
    "player_qb_injuries",
    "player_qb_continuity",
    "player_injuries_continuity",
    "player",
    "player_injury_value",
    "player_value",
]
MARGIN_MODEL_NAMES = ("ridge", "hgb")
MARGIN_TARGETS: tuple[MarginTarget, ...] = ("margin", "market_residual")
MARGIN_FEATURE_PROFILES: tuple[MarginFeatureProfile, ...] = (
    "base",
    "pbp",
    "pbp_adjusted",
    "drive",
    "graph",
    "player_qb",
    "player_injuries",
    "player_continuity",
    "player_qb_injuries",
    "player_qb_continuity",
    "player_injuries_continuity",
    "player",
    "player_injury_value",
    "player_value",
)

_MARGIN_PROFILE_FEATURE_SETS: dict[MarginFeatureProfile, tuple[str, str]] = {
    "base": ("football", "full"),
    "pbp": ("football_pbp", "full_pbp"),
    "pbp_adjusted": ("football_pbp_adjusted", "full_pbp_adjusted"),
    "drive": ("football_drive", "full_drive"),
    "graph": ("football_graph_schedule", "full_graph_schedule"),
    "player_qb": ("football_player_qb", "full_player_qb"),
    "player_injuries": ("football_player_injuries", "full_player_injuries"),
    "player_continuity": ("football_player_continuity", "full_player_continuity"),
    "player_qb_injuries": ("football_player_qb_injuries", "full_player_qb_injuries"),
    "player_qb_continuity": ("football_player_qb_continuity", "full_player_qb_continuity"),
    "player_injuries_continuity": (
        "football_player_injuries_continuity",
        "full_player_injuries_continuity",
    ),
    "player": ("football_player", "full_player"),
    "player_injury_value": (
        "football_player_injury_value",
        "full_player_injury_value",
    ),
    "player_value": ("football_player_value", "full_player_value"),
}


def margin_feature_set(target: MarginTarget, feature_profile: MarginFeatureProfile = "base") -> str:
    """Return the named feature set backing a margin-profile target."""

    if feature_profile not in MARGIN_FEATURE_PROFILES:
        raise ValueError(f"Unknown margin feature profile: {feature_profile}")
    if target == "margin":
        return _MARGIN_PROFILE_FEATURE_SETS[feature_profile][0]
    if target == "market_residual":
        return _MARGIN_PROFILE_FEATURE_SETS[feature_profile][1]
    raise ValueError(f"Unknown margin target: {target}")


def margin_feature_columns(
    target: MarginTarget, feature_profile: MarginFeatureProfile = "base"
) -> tuple[str, ...]:
    """Return the explicit feature contract for each margin question."""

    return FEATURE_SETS[margin_feature_set(target, feature_profile)]


def _target_values(frame: pd.DataFrame, target: MarginTarget) -> pd.Series:
    if target == "margin":
        return pd.to_numeric(frame["result"], errors="coerce")
    return pd.to_numeric(frame["ats_margin"], errors="coerce")


def make_margin_estimator(
    model_name: str,
    random_state: int = 42,
    *,
    ridge_alpha: float = 10.0,
) -> BaseEstimator:
    if not np.isfinite(ridge_alpha) or ridge_alpha <= 0.0:
        raise ValueError("ridge_alpha must be finite and positive")
    if model_name == "ridge":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=ridge_alpha)),
            ]
        )
    if model_name == "hgb":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "regressor",
                    HistGradientBoostingRegressor(
                        learning_rate=0.04,
                        l2_regularization=2.0,
                        max_iter=100,
                        max_leaf_nodes=15,
                        early_stopping=False,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown margin model {model_name!r}; choose one of {MARGIN_MODEL_NAMES}")


def _smoothed_probability(samples: npt.NDArray[np.float64], threshold: float) -> float:
    successes = float(np.count_nonzero(samples > threshold))
    return (successes + 0.5) / (len(samples) + 1.0)


@dataclass
class MarginModel:
    estimator: BaseEstimator | None
    residuals: npt.NDArray[np.float64]
    model_name: str
    ridge_alpha: float | None
    target: MarginTarget | Literal["market"]
    feature_columns: tuple[str, ...]
    training_rows: int
    distribution_rows: int
    training_max_gameday: str

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"spread_line"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise DataContractError(f"Margin scoring is missing columns: {', '.join(missing)}")
        spread = pd.to_numeric(frame["spread_line"], errors="coerce").to_numpy(dtype=float)
        if np.isnan(spread).any():
            raise ValueError("Margin scoring requires a spread for every game")

        if self.target == "market":
            predicted_margin = spread.copy()
            predicted_residual = np.zeros(len(frame), dtype=float)
            market_cover_probability = [
                no_vig_probabilities(row.get("home_spread_odds"), row.get("away_spread_odds"))[0]
                for _, row in frame.iterrows()
            ]
        else:
            market_cover_probability = []
            missing_features = sorted(set(self.feature_columns).difference(frame.columns))
            if missing_features:
                raise DataContractError(
                    f"Margin scoring is missing features: {', '.join(missing_features)}"
                )
            if self.estimator is None:
                raise RuntimeError("Fitted margin model has no estimator")
            raw = np.asarray(
                self.estimator.predict(frame.loc[:, list(self.feature_columns)]), dtype=float
            )
            if self.target == "margin":
                predicted_margin = raw
                predicted_residual = raw - spread
            else:
                predicted_residual = raw
                predicted_margin = spread + raw

        probabilities_win: list[float] = []
        probabilities_cover: list[float] = []
        lower_50: list[float] = []
        upper_50: list[float] = []
        lower_80: list[float] = []
        upper_80: list[float] = []
        for row_index, (center, line) in enumerate(zip(predicted_margin, spread, strict=True)):
            distribution = np.asarray(center + self.residuals, dtype=np.float64)
            probabilities_win.append(_smoothed_probability(distribution, 0.0))
            probabilities_cover.append(
                market_cover_probability[row_index]
                if self.target == "market"
                else _smoothed_probability(distribution, float(line))
            )
            quantiles = np.quantile(distribution, [0.10, 0.25, 0.75, 0.90])
            lower_80.append(float(quantiles[0]))
            lower_50.append(float(quantiles[1]))
            upper_50.append(float(quantiles[2]))
            upper_80.append(float(quantiles[3]))

        return pd.DataFrame(
            {
                "predicted_margin": predicted_margin,
                "fair_spread": predicted_margin,
                "market_spread": spread,
                "predicted_market_residual": predicted_residual,
                "home_win_probability": probabilities_win,
                "home_cover_probability": probabilities_cover,
                "margin_lower_50": lower_50,
                "margin_upper_50": upper_50,
                "margin_lower_80": lower_80,
                "margin_upper_80": upper_80,
            },
            index=frame.index,
        )


def fit_margin_model(
    frame: pd.DataFrame,
    *,
    target: MarginTarget = "margin",
    model_name: str = "ridge",
    distribution_fraction: float = 0.20,
    min_distribution_rows: int = 10,
    random_state: int = 42,
    feature_profile: MarginFeatureProfile = "base",
    ridge_alpha: float = 10.0,
) -> MarginModel:
    feature_columns = margin_feature_columns(target, feature_profile)
    required = {"game_id", "gameday", "result", "ats_margin", *feature_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"Margin training is missing columns: {', '.join(missing)}")
    if not 0.10 <= distribution_fraction < 0.5:
        raise ValueError("distribution_fraction must be in [0.10, 0.5)")

    training = frame.loc[_target_values(frame, target).notna()].copy()
    training["gameday"] = pd.to_datetime(training["gameday"], errors="raise")
    training = training.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    if len(training) < 50:
        raise ValueError("At least 50 completed games are required for a margin model")
    distribution_rows = int(len(training) * distribution_fraction)
    if distribution_rows < min_distribution_rows or len(training) - distribution_rows < 40:
        raise ValueError("Not enough rows for an out-of-time residual distribution")

    split = len(training) - distribution_rows
    fit_part = training.iloc[:split]
    distribution_part = training.iloc[split:]
    temporary = make_margin_estimator(model_name, random_state, ridge_alpha=ridge_alpha)
    temporary.fit(
        fit_part.loc[:, list(feature_columns)],
        _target_values(fit_part, target),
    )
    calibration_prediction = np.asarray(
        temporary.predict(distribution_part.loc[:, list(feature_columns)]), dtype=float
    )
    residuals = np.asarray(
        _target_values(distribution_part, target).to_numpy(dtype=float) - calibration_prediction,
        dtype=np.float64,
    )
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < min_distribution_rows:
        raise ValueError("Out-of-time residual distribution has too few finite values")

    estimator = make_margin_estimator(model_name, random_state, ridge_alpha=ridge_alpha)
    estimator.fit(training.loc[:, list(feature_columns)], _target_values(training, target))
    return MarginModel(
        estimator=estimator,
        residuals=residuals,
        model_name=model_name,
        ridge_alpha=ridge_alpha if model_name == "ridge" else None,
        target=target,
        feature_columns=feature_columns,
        training_rows=len(training),
        distribution_rows=len(residuals),
        training_max_gameday=training["gameday"].max().date().isoformat(),
    )


def fit_market_baseline(frame: pd.DataFrame) -> MarginModel:
    training = frame.loc[frame["ats_margin"].notna()].copy()
    training["gameday"] = pd.to_datetime(training["gameday"], errors="raise")
    training = training.sort_values(["gameday", "game_id"])
    if len(training) < 50:
        raise ValueError("At least 50 completed games are required for the market baseline")
    residuals = (
        pd.to_numeric(training["ats_margin"], errors="coerce").dropna().to_numpy(dtype=float)
    )
    return MarginModel(
        estimator=None,
        residuals=np.asarray(residuals, dtype=np.float64),
        model_name="market",
        ridge_alpha=None,
        target="market",
        feature_columns=(),
        training_rows=len(training),
        distribution_rows=len(residuals),
        training_max_gameday=training["gameday"].max().date().isoformat(),
    )


def margin_model_metadata(model: MarginModel) -> dict[str, Any]:
    return {
        "model_name": model.model_name,
        "ridge_alpha": model.ridge_alpha,
        "target": model.target,
        "feature_columns": list(model.feature_columns),
        "training_rows": model.training_rows,
        "distribution_rows": model.distribution_rows,
        "training_max_gameday": model.training_max_gameday,
        "residual_mean": float(np.mean(model.residuals)),
        "residual_std": float(np.std(model.residuals, ddof=1)),
    }
