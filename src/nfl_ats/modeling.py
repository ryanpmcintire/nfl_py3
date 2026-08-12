"""Model construction and chronological probability calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_ats.constants import FEATURE_SETS, MODEL_FEATURE_COLUMNS, OUTCOME_COLUMNS
from nfl_ats.data import DataContractError

MODEL_NAMES = ("logistic", "hgb")


def resolve_feature_columns(feature_set: str | None = None) -> tuple[str, ...]:
    name = feature_set or "full"
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {name!r}; choose one of {tuple(FEATURE_SETS)}")
    return FEATURE_SETS[name]


def validate_model_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
) -> None:
    if not feature_columns:
        raise ValueError("At least one model feature is required")
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise DataContractError(f"Model table is missing features: {', '.join(missing)}")
    leaked = set(feature_columns).intersection(OUTCOME_COLUMNS)
    if leaked:
        raise RuntimeError(f"Outcome columns leaked into the feature allowlist: {sorted(leaked)}")


def make_estimator(model_name: str, random_state: int = 42) -> BaseEstimator:
    if model_name == "logistic":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.25,
                        max_iter=2_000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "hgb":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        l2_regularization=1.0,
                        max_iter=100,
                        max_leaf_nodes=15,
                        early_stopping=False,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model {model_name!r}; choose one of {MODEL_NAMES}")


def _positive_probability(
    estimator: BaseEstimator, features: pd.DataFrame
) -> npt.NDArray[np.float64]:
    probabilities = estimator.predict_proba(features)  # type: ignore[attr-defined]
    classes = np.asarray(estimator.classes_)  # type: ignore[attr-defined]
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise RuntimeError("Fitted estimator does not contain a unique positive class")
    return np.asarray(probabilities[:, matches[0]], dtype=float)


def _logit(probabilities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.asarray(np.log(clipped / (1.0 - clipped)).reshape(-1, 1), dtype=np.float64)


@dataclass
class CoverModel:
    """A fitted estimator plus an optional out-of-time Platt calibrator."""

    estimator: BaseEstimator
    calibrator: LogisticRegression | None
    model_name: str
    training_rows: int
    calibration_rows: int
    training_max_gameday: str
    feature_set: str
    feature_columns: tuple[str, ...]

    def predict_home_cover(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        validate_model_frame(frame, self.feature_columns)
        features = frame.loc[:, list(self.feature_columns)]
        raw = _positive_probability(self.estimator, features)
        if self.calibrator is None:
            return raw
        return np.asarray(self.calibrator.predict_proba(_logit(raw))[:, 1], dtype=np.float64)


def fit_cover_model(
    frame: pd.DataFrame,
    model_name: str = "logistic",
    calibration_fraction: float = 0.2,
    min_calibration_rows: int = 100,
    random_state: int = 42,
    feature_set: str = "full",
) -> CoverModel:
    """Fit on past games and calibrate on the chronological tail.

    The calibrator sees predictions from a model that was trained only on rows
    before its calibration window. The final base estimator is then refit on
    every supplied row; no random cross-validation is used.
    """

    feature_columns = resolve_feature_columns(feature_set)
    validate_model_frame(frame, feature_columns)
    if not 0.0 <= calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be in [0, 0.5)")

    training = frame.loc[frame["home_cover"].notna()].copy()
    training["gameday"] = pd.to_datetime(training["gameday"], errors="raise")
    training = training.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    if len(training) < 50:
        raise ValueError("At least 50 non-push games are required to train a model")
    target = training["home_cover"].astype(int)
    if target.nunique() != 2:
        raise ValueError("Training data must contain both home and away covers")

    calibration_rows = int(len(training) * calibration_fraction)
    calibrator: LogisticRegression | None = None
    if calibration_rows >= min_calibration_rows:
        split = len(training) - calibration_rows
        fit_part = training.iloc[:split]
        calibration_part = training.iloc[split:]
        if fit_part["home_cover"].nunique() == 2 and calibration_part["home_cover"].nunique() == 2:
            calibration_estimator = make_estimator(model_name, random_state)
            calibration_estimator.fit(
                fit_part.loc[:, list(feature_columns)], fit_part["home_cover"].astype(int)
            )
            raw = _positive_probability(
                calibration_estimator, calibration_part.loc[:, list(feature_columns)]
            )
            calibrator = LogisticRegression(C=1.0, max_iter=1_000, random_state=random_state)
            calibrator.fit(_logit(raw), calibration_part["home_cover"].astype(int))
        else:
            calibration_rows = 0
    else:
        calibration_rows = 0

    estimator = make_estimator(model_name, random_state)
    estimator.fit(training.loc[:, list(feature_columns)], target)
    return CoverModel(
        estimator=estimator,
        calibrator=calibrator,
        model_name=model_name,
        training_rows=len(training),
        calibration_rows=calibration_rows,
        training_max_gameday=training["gameday"].max().date().isoformat(),
        feature_set=feature_set,
        feature_columns=feature_columns,
    )


def model_metadata(model: CoverModel) -> dict[str, Any]:
    return {
        "model_name": model.model_name,
        "training_rows": model.training_rows,
        "calibration_rows": model.calibration_rows,
        "training_max_gameday": model.training_max_gameday,
        "feature_set": model.feature_set,
        "feature_columns": list(model.feature_columns),
    }


def logistic_coefficients(model: CoverModel) -> pd.DataFrame:
    """Return named standardized coefficients, including missingness flags."""

    if model.model_name != "logistic":
        raise ValueError("Named coefficients are available only for the logistic model")
    pipeline = model.estimator
    named_steps = pipeline.named_steps  # type: ignore[attr-defined]
    imputer = named_steps["imputer"]
    classifier = named_steps["classifier"]
    names = list(model.feature_columns)
    indicator_indices = getattr(imputer.indicator_, "features_", [])
    names.extend(f"missing:{model.feature_columns[int(index)]}" for index in indicator_indices)
    coefficients = np.asarray(classifier.coef_[0], dtype=float)
    if len(names) != len(coefficients):
        raise RuntimeError("Unable to align logistic coefficients with transformed features")
    return pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False, ignore_index=True)
