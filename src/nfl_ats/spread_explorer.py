"""Spread explorer: "what would the model say if the line were different?"

Owner request, 2026-08-20: pick a spread for a game and see the model's read
of the odds of covering, as of the last site build. This module is the
library half of that feature -- the picks-page widget
(:mod:`nfl_ats.public_board`) and the CLI query tool (``scripts/cover_odds.py``)
both call it rather than duplicating the math.

Scope, matching ``docs/smooth_cdf_mapping.md``'s own declared scope exactly:
only the two-way ``home_cover_probability`` is a Gaussian read (the MOD-08
mapping, promoted 2026-08-19 to the sole production probability method,
``nfl_ats.outcomes.score_outcome_week``'s default). Push probability is a
DIFFERENT computation -- the production three-way split
(``margin._three_way_probabilities``) reads a discrete rounding of the raw
out-of-time residual SAMPLE, a fundamentally different (and, per that
function's own docstring, deliberately un-smoothed) computation from the
continuous Gaussian fit. Under a continuous distribution the probability of
landing exactly on any single line is mathematically zero, so synthesizing a
"push probability" from the Gaussian mean/sd ALONE would not be a real
number -- it would be an invented one. Two call shapes follow from this:
:func:`compute_spread_explorer_params` (mean/sd only, for the picks-page
widget's compact JSON embedding) never reports push -- callers surface a
plain-English note instead. :func:`compute_spread_explorer_distribution`
(the full residual sample, for the CLI query tool, which does not have a
page-weight budget) DOES report an honest push probability, via the SAME
discrete-rounding ``margin._three_way_probabilities`` production itself
uses -- never a number invented from the Gaussian fit. This mirrors the
exact scope line ``docs/smooth_cdf_mapping.md`` already drew: "Only
home_cover_probability... is mapped" -- the two-way number is Gaussian, the
three-way split stays exactly as production already computes it.

The mean/sd/centre this module returns are never independently re-derived
from summary statistics or approximated -- they come from refitting the
EXACT production recipe (feature profile, regressor, ridge alpha,
``min_train_games``) via ``nfl_ats.outcomes.fit_margin_models_for_week``, the
same public entry point ``nfl_ats.smooth_cdf_mapping_overlay`` already uses
for an identical purpose (obtaining the fitted ``MarginModel`` for one week
rather than a pre-summarized card). Ridge and the Gaussian fit are both
deterministic (``random_state=42`` throughout ``margin.py``), so the refit
reproduces the exact predicted centre and out-of-time residual sample the
active card's own Gaussian read used -- proven, not assumed, by re-deriving
that Gaussian probability from the refit and requiring it to match the
card's own ``home_cover_probability`` to floating-point precision before
anything is trusted for the widget. A mismatch (e.g. the feature table was
rebuilt after the card was produced) raises ``DataContractError`` rather than
silently shipping a widget that could disagree with the published pick.
"""

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
from nfl_ats.margin import _three_way_probabilities
from nfl_ats.outcomes import fit_margin_models_for_week

#: The slider's range and granularity. 0.5-point steps cover both the
#: half-point AND whole-point lines real NFL spreads actually use (roughly
#: half of any given week's card sits on a whole number -- see the module
#: docstring on why push is deliberately not modeled at those lines rather
#: than the slider excluding them).
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
    """One game's Gaussian read: the predicted centre plus the out-of-time
    residual sample's mean/sd, all read from the SAME refit that reproduces
    the published card's own ``home_cover_probability``.

    ``home_cover_probability(line) = 1 - Phi(((line - center) - residual_mean) / residual_std)``
    -- exactly ``nfl_ats.calibration.smoothed_home_cover_probability(...,
    method="gaussian")``'s formula, generalized from the card's one quoted
    line to an arbitrary hypothetical one.
    """

    game_id: str
    home_team: str
    away_team: str
    center: float
    residual_mean: float
    residual_std: float
    card_line: float
    card_home_cover_probability: float


def load_feature_table_for_forecast(metadata: Mapping[str, Any], data_root: Path) -> pd.DataFrame:
    """Load the exact feature table a forecast's own provenance points to.

    Mirrors ``nfl_ats.smooth_cdf_mapping_overlay.record_smooth_cdf_mapping_challenger_decisions``'s
    resolution exactly: try the absolute path recorded in
    ``metadata["provenance"]["feature_table"]["path"]`` first (works on the
    machine that built the forecast), then fall back to
    ``data_root/processed/<same file name>`` (works on any machine with the
    same local pipeline outputs, since that absolute path is rarely portable
    across checkouts). Raises ``DataContractError`` rather than returning an
    empty frame -- a caller with no feature table cannot refit anything.
    """

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
) -> dict[str, SpreadExplorerGameParams]:
    """Refit each (season, week) group and return every game's widget params.

    ``predictions`` is the active model's own ``recommendations.csv``
    (already filtered to the ``market_residual`` method -- the only method
    the Gaussian mapping applies to). Every group is refit independently with
    a training cutoff strictly before that week's earliest kickoff, exactly
    as the real card was produced, via ``fit_margin_models_for_week``. See
    the module docstring for the proof-before-trust discipline this follows.
    """

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
        spread = aligned["spread_line"].to_numpy(dtype=float)

        gaussian_check = smoothed_home_cover_probability(
            model.residuals, centers, spread, method="gaussian"
        )
        supplied = group["home_cover_probability"].to_numpy(dtype=float)
        if not np.allclose(gaussian_check, supplied, rtol=0.0, atol=1e-9):
            raise DataContractError(
                f"Refit Gaussian probabilities for season {season} week {week} do not "
                "reproduce the supplied card's home_cover_probability -- the feature "
                "table or configuration has drifted from the one that produced this "
                "card; refusing to build a spread-explorer widget that could disagree "
                "with the published pick"
            )

        mean = float(np.mean(model.residuals))
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
            )
    return params


# ---------------------------------------------------------------------------
# The widget's own formula -- a pure Abramowitz-Stegun erf approximation,
# NOT scipy. This is deliberately a re-implementation of what the embedded
# browser JS computes (see ``public_board._SPREAD_EXPLORER_SCRIPT``), kept in
# lock-step by ``tests/test_spread_explorer.py``. The point of this function
# is the build-time consistency assertion in ``public_board.py``: proving the
# EXACT formula shipped to the browser reproduces the published number, which
# scipy's more precise implementation cannot demonstrate on its own (that
# correctness -- that mean/std/center are the production ones -- is already
# proven above, to floating-point precision, via scipy, inside
# ``compute_spread_explorer_params``).
# ---------------------------------------------------------------------------


def _erf_abramowitz_stegun(x: float) -> float:
    """Abramowitz & Stegun 7.1.26, max absolute error ~1.5e-7 -- the same
    approximation embedded in the browser widget's JS."""

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
    """The home-cover probability at a hypothetical ``line``, computed with
    the SAME erf approximation the browser widget uses -- not scipy. Callers
    that want the production-precision number should use
    ``nfl_ats.calibration.smoothed_home_cover_probability`` instead; this
    function exists to let Python and JS be checked against each other."""

    threshold = line - center
    z = (threshold - mean) / (std * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + _erf_abramowitz_stegun(z))
    return 1.0 - cdf


def spread_explorer_payload(
    params: Mapping[str, SpreadExplorerGameParams],
) -> dict[str, dict[str, Any]]:
    """The JSON-serializable per-game blob the picks page embeds inline.

    Rounded to 6 decimal places -- comfortably more precision than the
    page's own displayed 0.1%, and small enough that the round-trip through
    JSON never meaningfully perturbs the widget's own consistency check
    (``public_board._assert_spread_explorer_matches_card`` checks these
    EXACT rounded values, not the unrounded Python floats, since the rounded
    values are what actually ships to the browser).
    """

    return {
        game_id: {
            "home": p.home_team,
            "away": p.away_team,
            "center": round(p.center, 6),
            "mean": round(p.residual_mean, 6),
            "std": round(p.residual_std, 6),
            "line": round(p.card_line, 3),
        }
        for game_id, p in params.items()
    }


# ---------------------------------------------------------------------------
# Single-game distribution (for scripts/cover_odds.py) -- the FULL residual
# sample, not just its mean/sd, so callers with no page-weight budget can
# report an honest push probability via the SAME production discrete-
# rounding function (``margin._three_way_probabilities``), not a number
# invented from the Gaussian fit. See the module docstring's "Two call
# shapes" paragraph.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpreadExplorerGameDistribution:
    """One game's fitted predictive distribution: the predicted centre plus
    the FULL out-of-time residual sample, proven (not assumed) to reproduce
    the published card's own ``home_cover_probability`` at ``probability_method``
    before being returned."""

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
) -> SpreadExplorerGameDistribution:
    """Refit ONE game's week (the exact production recipe) and return its
    centre plus full residual sample, verified against the published card's
    own ``home_cover_probability`` first -- the same refit-and-verify
    discipline as :func:`compute_spread_explorer_params`, generalized to
    whichever ``probability_method`` the active card was actually built
    with (``compute_spread_explorer_params`` is hardcoded to "gaussian"
    because that is the only method the picks-page widget's formula reads;
    this function is reused by the CLI tool, which checks the active
    model's own recorded method rather than assuming).
    """

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
    )


def spread_explorer_three_way(
    distribution: SpreadExplorerGameDistribution, line: float
) -> tuple[float, float, float]:
    """``(home_covers, push, home_does_not_cover)`` at a hypothetical
    ``line`` -- the SAME discrete-rounding three-way split
    ``margin._three_way_probabilities`` computes for the real published
    card (unconditionally, regardless of probability method), applied to
    this game's own refit residual sample. The three values always sum to
    1.0. ``home_does_not_cover`` is exactly "the away side covers" once push
    is accounted for separately (a two-outcome ATS market has no third
    option once a push is excluded).
    """

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
