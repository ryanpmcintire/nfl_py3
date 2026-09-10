from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.joint_residual_model import (
    JOINT_RIDGE_ALPHA,
    UNION_FEATURES,
    make_joint_estimator,
    realised_residual_frame,
)
from nfl_ats.totals import TotalsDataError, TotalsView, design_matrix

ServedTotalMethod = Literal["blend_k01", "joint_residual"]

BLEND_K01_WEIGHT = 0.1

JOINT_TOTAL_BLEND_WEIGHT = 0.1

DEFAULT_JOINT_FEATURES_FILENAME = "game_features_weak_stack.parquet"

SERVED_TOTAL_METHOD: ServedTotalMethod = "joint_residual"

_JOINT_TARGET_COLUMNS: tuple[str, str] = ("margin_residual", "total_residual")


def apply_blend(market_total: float, totals_view: TotalsView | None, *, weight: float) -> float:

    if totals_view is None:
        return float(market_total)
    return float(market_total) + float(weight) * float(totals_view.residual)


def served_total_blend_k01(
    market_total: float, totals_view: TotalsView | None, *, weight: float = BLEND_K01_WEIGHT
) -> float:

    return apply_blend(market_total, totals_view, weight=weight)


def joint_residual_total_view(
    game_id: str,
    data_root: Path,
    *,
    features_path: Path | None = None,
    feature_columns: tuple[str, ...] = UNION_FEATURES,
    ridge_alpha: float = JOINT_RIDGE_ALPHA,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> TotalsView | None:

    path = (
        features_path
        if features_path is not None
        else (data_root / "processed" / DEFAULT_JOINT_FEATURES_FILENAME)
    )
    if not path.is_file():
        return None
    features = pd.read_parquet(path)
    rows = features.loc[features["game_id"].astype(str).eq(game_id)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    if pd.isna(row.get("total_line")):
        return None
    season, week = int(row["season"]), int(row["week"])

    population = realised_residual_frame(features, feature_columns=feature_columns)
    prior = population.loc[
        (population["season"] < season)
        | ((population["season"] == season) & (population["week"] < week))
    ]
    if len(prior) < min_train_games:
        return None

    estimator = make_joint_estimator(ridge_alpha=ridge_alpha)
    estimator.fit(
        design_matrix(prior, feature_columns),
        prior.loc[:, list(_JOINT_TARGET_COLUMNS)].astype(float).to_numpy(),
    )
    target_design = design_matrix(rows.iloc[[0]], feature_columns)
    predicted = np.asarray(estimator.predict(target_design), dtype=float).reshape(1, -1)
    total_index = _JOINT_TARGET_COLUMNS.index("total_residual")
    residual = float(predicted[0, total_index])
    market_total = float(row["total_line"])
    return TotalsView(
        predicted_total=market_total + residual,
        market_total=market_total,
        residual=residual,
        train_games=len(prior),
        source=(
            f"joint residual ridge (union {len(feature_columns)} cols, alpha={ridge_alpha:g}) "
            f"trained on {len(prior)} games before {season} week {week}"
        ),
    )


def served_total_joint_residual(
    market_total: float, joint_view: TotalsView | None, *, weight: float = JOINT_TOTAL_BLEND_WEIGHT
) -> float | None:

    if joint_view is None:
        return None
    return apply_blend(market_total, joint_view, weight=weight)


def served_total(
    method: ServedTotalMethod,
    *,
    market_total: float,
    blend_view: TotalsView | None,
    joint_view: TotalsView | None,
    blend_weight: float = BLEND_K01_WEIGHT,
    joint_weight: float = JOINT_TOTAL_BLEND_WEIGHT,
) -> tuple[float, ServedTotalMethod]:

    if method == "joint_residual":
        value = served_total_joint_residual(market_total, joint_view, weight=joint_weight)
        if value is not None:
            return value, "joint_residual"
        return served_total_blend_k01(market_total, blend_view, weight=blend_weight), "blend_k01"
    if method == "blend_k01":
        return served_total_blend_k01(market_total, blend_view, weight=blend_weight), "blend_k01"
    raise ValueError(f"Unknown served-total method: {method!r}")


__all__ = [
    "BLEND_K01_WEIGHT",
    "DEFAULT_JOINT_FEATURES_FILENAME",
    "JOINT_TOTAL_BLEND_WEIGHT",
    "SERVED_TOTAL_METHOD",
    "ServedTotalMethod",
    "TotalsDataError",
    "apply_blend",
    "joint_residual_total_view",
    "served_total",
    "served_total_blend_k01",
    "served_total_joint_residual",
]
