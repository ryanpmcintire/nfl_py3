from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.constants import MODEL_FEATURE_COLUMNS
from nfl_ats.data import DataContractError
from nfl_ats.modeling import (
    fit_cover_model,
    logistic_coefficients,
    make_estimator,
    model_metadata,
    resolve_feature_columns,
)


def test_fit_model_and_predict_calibrated_probabilities(model_frame: pd.DataFrame) -> None:
    model = fit_cover_model(
        model_frame,
        model_name="logistic",
        calibration_fraction=0.2,
        min_calibration_rows=20,
    )
    probabilities = model.predict_home_cover(model_frame.tail(10))
    assert probabilities.shape == (10,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert model.calibration_rows == 32
    assert model_metadata(model)["feature_columns"] == list(MODEL_FEATURE_COLUMNS)
    coefficients = logistic_coefficients(model)
    assert len(coefficients) >= len(MODEL_FEATURE_COLUMNS)
    assert coefficients["feature"].is_unique


def test_hgb_model_path(model_frame: pd.DataFrame) -> None:
    model = fit_cover_model(model_frame, model_name="hgb", calibration_fraction=0.0)
    assert len(model.predict_home_cover(model_frame.head(3))) == 3
    assert model.calibrator is None


def test_model_validation_and_training_guards(model_frame: pd.DataFrame) -> None:
    with pytest.raises(DataContractError, match="missing features"):
        fit_cover_model(model_frame.drop(columns=MODEL_FEATURE_COLUMNS[0]))
    with pytest.raises(ValueError, match="At least 50"):
        fit_cover_model(model_frame.head(20))
    one_class = model_frame.copy()
    one_class["home_cover"] = 1.0
    with pytest.raises(ValueError, match="both home and away"):
        fit_cover_model(one_class)
    with pytest.raises(ValueError, match="Unknown model"):
        make_estimator("forest")
    with pytest.raises(ValueError, match="Unknown feature set"):
        resolve_feature_columns("mystery")
    with pytest.raises(ValueError, match="only for the logistic"):
        logistic_coefficients(fit_cover_model(model_frame, model_name="hgb"))


def test_smaller_feature_set_is_preserved(model_frame: pd.DataFrame) -> None:
    model = fit_cover_model(model_frame, feature_set="market", calibration_fraction=0.0)
    assert model.feature_columns == ("spread_line", "total_line")
    assert model.estimator.n_features_in_ == 2
