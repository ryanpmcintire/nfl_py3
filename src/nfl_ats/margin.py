"""Fair-margin and market-residual models with empirical predictive distributions."""

from __future__ import annotations

from collections.abc import Sequence
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

from nfl_ats.constants import FEATURE_SETS, MIN_FITTABLE_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.modeling import regular_season_rows
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
    "player_participation",
    "weak_stack",
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
    "player_participation",
    "weak_stack",
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
    "player_participation": (
        "football_player_participation",
        "full_player_participation",
    ),
    # MOD-07 candidate profile: only ever fitted on the weak-stack table, which
    # carries learned-availability injury columns under the same names the fixed
    # -prior table uses. Never point this profile at game_features_player.parquet.
    "weak_stack": ("football_weak_stack", "full_weak_stack"),
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


# The quoted line plus a symmetric grid of alternative home spreads, spanning
# the "key number" region around most NFL closing lines in half-point steps.
LINE_SWEEP_MIN_OFFSET = -4.0
LINE_SWEEP_MAX_OFFSET = 4.0
LINE_SWEEP_STEP = 0.5
DEFAULT_LINE_SWEEP_OFFSETS: tuple[float, ...] = tuple(
    round(float(offset), 1)
    for offset in np.arange(
        LINE_SWEEP_MIN_OFFSET, LINE_SWEEP_MAX_OFFSET + LINE_SWEEP_STEP / 2, LINE_SWEEP_STEP
    )
)


def _is_integer_line(line: float) -> bool:
    """True when a spread cannot mathematically be pushed against.

    A real NFL final margin (home score minus away score) is always an
    integer, so a push -- the predictive distribution landing exactly on the
    line -- is only possible when the line itself is an integer. Half-point
    lines can never push; this is a football fact, not a modeling choice, so
    it is enforced here rather than left to accidental floating-point ties.
    """

    return bool(np.isclose(float(line) % 1.0, 0.0, atol=1e-9))


def _three_way_probabilities(
    distribution: npt.NDArray[np.float64], line: float
) -> tuple[float, float, float]:
    """Split the empirical predictive distribution at a line into win/push/loss.

    A real football margin is a whole number of points, so the question "does
    this game land exactly on the line" can only be asked of integer outcomes.
    The predictive sample is continuous (a float centre plus float residuals),
    and this function previously tested it for exact equality with the line --
    a condition that essentially never fires in floating point. Every card the
    active model published therefore carried ``push_probability = 0.0000``
    while roughly 4.8% of games on an integer line actually push, rising to
    **9.0%** at a line of 3, and ``home_loss_probability`` silently absorbed
    all of it (measured 2009-2025, n=4,431).

    Rounding the sample to integers fixes that. Note the deliberate asymmetry
    that remains: ``home_cover_probability`` still comes from
    ``_smoothed_probability``'s continuous ``>`` test, so it is unchanged and
    no pick moves. Only the three-way split is corrected here. Making the
    cover probability itself push-aware would change the frozen model's
    published probabilities and is a scored change, not a bug fix.
    """

    n = len(distribution)
    if _is_integer_line(line):
        outcomes = np.rint(distribution)
        win_count = int(np.count_nonzero(outcomes > line))
        push_count = int(np.count_nonzero(outcomes == line))
    else:
        # No integer margin can land on a half-point line, so a push is
        # impossible by construction and the contract requires exactly zero.
        win_count = int(np.count_nonzero(distribution > line))
        push_count = 0
    loss_count = n - win_count - push_count
    return win_count / n, push_count / n, loss_count / n


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

    def _spread(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        required = {"spread_line"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise DataContractError(f"Margin scoring is missing columns: {', '.join(missing)}")
        spread = pd.to_numeric(frame["spread_line"], errors="coerce").to_numpy(dtype=float)
        if np.isnan(spread).any():
            raise ValueError("Margin scoring requires a spread for every game")
        return spread

    def _predicted_margin(
        self, frame: pd.DataFrame, spread: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (predicted_margin, predicted_market_residual) for a frame.

        Shared by ``predict`` and ``line_sweep`` so the distribution center is
        computed identically -- the center never depends on which line it is
        later compared against.
        """

        if self.target == "market":
            predicted_margin = spread.copy()
            predicted_residual = np.zeros(len(frame), dtype=float)
            return predicted_margin, predicted_residual

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
            return raw, raw - spread
        return spread + raw, raw

    def distribution(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return the full empirical predictive sample for every row.

        Shape ``(len(frame), len(self.residuals))``: row ``i`` is
        ``predicted_margin[i] + self.residuals``. Exposed for analyses that
        need the raw samples rather than a single summarized probability
        (e.g. key-number mass, reliability diagrams) without duplicating the
        center computation.
        """

        spread = self._spread(frame)
        predicted_margin, _ = self._predicted_margin(frame, spread)
        return predicted_margin[:, np.newaxis] + self.residuals[np.newaxis, :]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        spread = self._spread(frame)
        predicted_margin, predicted_residual = self._predicted_margin(frame, spread)
        if self.target == "market":
            market_cover_probability = [
                no_vig_probabilities(row.get("home_spread_odds"), row.get("away_spread_odds"))[0]
                for _, row in frame.iterrows()
            ]
        else:
            market_cover_probability = []

        probabilities_win: list[float] = []
        probabilities_cover: list[float] = []
        probabilities_cover_excluding_push: list[float] = []
        probabilities_push: list[float] = []
        probabilities_loss: list[float] = []
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
            win, push, loss = _three_way_probabilities(distribution, float(line))
            probabilities_cover_excluding_push.append(win)
            probabilities_push.append(push)
            probabilities_loss.append(loss)
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
                "home_cover_probability_excluding_push": probabilities_cover_excluding_push,
                "push_probability": probabilities_push,
                "home_loss_probability": probabilities_loss,
                "margin_lower_50": lower_50,
                "margin_upper_50": upper_50,
                "margin_lower_80": lower_80,
                "margin_upper_80": upper_80,
            },
            index=frame.index,
        )

    def line_sweep(
        self,
        frame: pd.DataFrame,
        *,
        offsets: Sequence[float] = DEFAULT_LINE_SWEEP_OFFSETS,
    ) -> pd.DataFrame:
        """Evaluate the predictive distribution across alternative home spreads.

        For every game, sweeps a grid of alternative home spreads built from
        the quoted ``spread_line`` plus each offset, and reports the
        win/push/loss split at every alternative line. The predictive margin
        distribution is fixed once, conditioned on the actually quoted market
        line (for ``market_residual`` the market's information enters through
        that real quote); only the settlement threshold moves across the
        sweep. This answers the decision question "what is our cover
        probability if this game settles at line ``s``" — for example when a
        pool posts a different number than the market or when shopping
        half-points across books — not the counterfactual "what would the
        model predict if the market itself had quoted ``s``".

        Returns a tidy per-game-per-line table: one row per (game, offset).
        """

        if not offsets:
            raise ValueError("line_sweep requires at least one offset")
        required = {"game_id", "spread_line"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise DataContractError(f"Line sweep is missing columns: {', '.join(missing)}")
        quoted = self._spread(frame)
        game_ids = frame["game_id"].to_numpy()
        predicted_margin, _ = self._predicted_margin(frame, quoted)

        rows: list[dict[str, Any]] = []
        for offset in offsets:
            alternative = quoted + float(offset)
            for game_id, quoted_line, alt_line, center in zip(
                game_ids, quoted, alternative, predicted_margin, strict=True
            ):
                distribution = np.asarray(center + self.residuals, dtype=np.float64)
                cover_probability = _smoothed_probability(distribution, float(alt_line))
                win, push, loss = _three_way_probabilities(distribution, float(alt_line))
                pick_probability = (
                    cover_probability if cover_probability >= 0.5 else 1.0 - cover_probability
                )
                rows.append(
                    {
                        "game_id": game_id,
                        "quoted_line": float(quoted_line),
                        "line_offset": round(float(offset), 4),
                        "alternative_line": float(alt_line),
                        "home_cover_probability": cover_probability,
                        "home_cover_probability_excluding_push": win,
                        "push_probability": push,
                        "home_loss_probability": loss,
                        "pick_probability": pick_probability,
                        "confidence": pick_probability - 0.5,
                    }
                )
        return pd.DataFrame(rows).sort_values(["game_id", "line_offset"]).reset_index(drop=True)


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

    training = regular_season_rows(frame)
    training = training.loc[_target_values(training, target).notna()].copy()
    training["gameday"] = pd.to_datetime(training["gameday"], errors="raise")
    training = training.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    if len(training) < MIN_FITTABLE_TRAIN_GAMES:
        raise ValueError(
            f"At least {MIN_FITTABLE_TRAIN_GAMES} completed games are required for a margin model"
        )
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
    training = regular_season_rows(frame)
    training = training.loc[training["ats_margin"].notna()].copy()
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
