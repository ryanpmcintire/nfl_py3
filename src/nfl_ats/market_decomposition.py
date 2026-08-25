"""Market decomposition: what the betting market prices vs. what reality prices.

Three ridge regressions are fit on **identical** games, features, and
preprocessing (the active model's ``player`` feature profile, median
imputation with missingness indicators, standardization, ridge alpha 10 --
see :mod:`nfl_ats.margin`), refit on the same weekly walk-forward cadence as
:func:`nfl_ats.outcomes.walk_forward_outcomes`:

    (a) margin ~ X                    -- reality's weights
    (b) spread_line ~ X               -- the market's weights (the close)
    (c) (margin - spread_line) ~ X    -- the residual model

``X`` deliberately excludes the market feature family (``spread_line``,
``total_line``) -- see :func:`decomposition_feature_columns` -- both because
a market feature predicting the market target would be degenerate, and
because it is what makes the reconciliation identity ``(a) - (b) == (c)``
meaningful: for a fixed ``X`` and ridge alpha, the ridge closed-form solution
is linear in the target, and ``(margin) - (spread_line) == (margin -
spread_line)`` exactly, so the fitted coefficients and predictions of (c)
must equal (a) minus (b) up to floating-point error. :func:`
walk_forward_decomposition` asserts this every refit window.

Coefficients are standardized (the fitted ``StandardScaler`` sits between
imputation and the ridge regressor, exactly as in
:func:`nfl_ats.margin.make_margin_estimator`) and are aggregated to feature
**families** -- the named groups in ``nfl_ats.constants.FEATURE_FAMILIES``,
the project's honest attribution unit; ridge smears weight across correlated
features, so only family-level aggregates are meaningful, never individual
feature coefficients. See :func:`family_weights_table` and
:func:`classify_families` for the four-bucket classification (``priced``,
``unpriced_predictive``, ``overpriced``, ``noise``) and its declared,
non-magic thresholds.

Honesty notes (also written into every artifact this module produces):

- ``unpriced_predictive`` is a hypothesis-generating diagnostic, not evidence
  of edge. The outer-season record is the only adjudicator of edge in this
  project; see the 2014-2017 replication closure of the QB+continuity
  profile for what an actual adjudicated result looks like.
- Ridge regression smears correlated features together, so only family-level
  aggregates are read here -- never an individual feature's coefficient.
- This fits on 2018-2025 outcomes the project has already viewed and scored
  repeatedly elsewhere. It is explanatory, not confirmatory, and scores no
  new pick stream -- it does not count as a new candidate stream under this
  project's multiplicity accounting.

Per-game attribution schema
----------------------------
:func:`attribute_predictions` returns (and the CLI writes as
``attribution.parquet``) a tidy, long frame with one row per
``(game_id, family)``:

============================  =======  ================================================
column                        dtype    meaning
============================  =======  ================================================
``game_id``                   str      nflverse game id
``season``                    int      season of the target week
``week``                      int      week of the target week
``home_team``                 str      home team abbreviation
``away_team``                 str      away team abbreviation
``family``                    str      a feature family name, or the sentinel
                                        ``"intercept"`` for the fitted model's constant
``contribution``               float   points of the predicted residual attributable to
                                        that family (coefficient x standardized value,
                                        summed over the family's design columns) for that
                                        game; for ``family == "intercept"`` this is the
                                        ridge intercept, unchanged across games
``predicted_residual``         float   the model's total predicted
                                        ``margin - spread_line`` for that game; identical
                                        on every row for the same ``game_id``; equals the
                                        sum of ``contribution`` across that game's rows
                                        within :data:`ATTRIBUTION_ATOL` (asserted at
                                        build time)
``actual_residual``            float   ``result - spread_line`` if the game has been
                                        played, else ``NaN``
``explanation``                str     a plain-English, pick-side-oriented sentence from
                                        :func:`explain_game`; identical on every row for
                                        the same ``game_id``
============================  =======  ================================================

This residual model is refit for :func:`attribute_predictions` using the
*same* ``X`` as the matched-regression suite above (excluding the market
family), so it is directly comparable to the family weights this module
reports elsewhere. It is therefore **not** numerically identical to the
deployed ``market_residual`` production model (see
``nfl_ats.margin.MARGIN_FEATURE_PROFILES``), which also conditions on
``spread_line`` itself -- that asymmetric feature contract is what makes the
production model well-calibrated, but it would make this decomposition's
"what does the market not see" framing circular.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_FAMILIES
from nfl_ats.data import DataContractError
from nfl_ats.margin import MarginFeatureProfile, make_margin_estimator, margin_feature_columns

# ---------------------------------------------------------------------------
# Frozen configuration -- matches the active model (see
# artifacts/active_ats_model.json: feature_profile="player", regressor=
# "ridge", ridge_alpha=10.0). No tuning is performed by this module.
# ---------------------------------------------------------------------------

DECOMPOSITION_TARGETS: tuple[str, ...] = ("margin", "spread", "residual")

DEFAULT_FEATURE_PROFILE: MarginFeatureProfile = "player"
DEFAULT_RIDGE_ALPHA = 10.0
DEFAULT_START_SEASON = 2018
DEFAULT_END_SEASON = 2025

# A family's "weight" below is its mean per-refit sum of |standardized
# coefficient| across its design columns, expressed as a *share* of the
# target's total weight across every family (so it is scale-free and
# comparable between the margin and spread targets, which share a points
# scale). These two thresholds define the four-bucket classification in
# `classify_family` and are written into every artifact's metadata so the
# cutoffs are auditable rather than implicit.
DEFAULT_NOISE_SHARE_THRESHOLD = 0.03
DEFAULT_OVERPRICED_RATIO_THRESHOLD = 1.5

# Reconciliation identity (a) - (b) == (c): walk-forward tolerance, in
# points, on each held-out game's |((margin-model) - (spread-model)) -
# (residual-model)| prediction. Ridge's closed-form solution is linear in
# the target for fixed X and alpha, so this should be within floating-point
# error; a violation indicates a real bug (e.g. divergent preprocessing
# state between the three fits), not numerical noise, so it raises.
RECONCILIATION_ATOL = 1e-4

# Per-game attribution identity: |sum(family contributions) - predicted| in
# points. Looser than machine epsilon to tolerate summation-order floating
# error across ~100 design columns, tight enough to catch a real bug.
ATTRIBUTION_ATOL = 1e-6

INTERCEPT_FAMILY = "intercept"

OPENER_MIN_GAMES_DEFAULT = 50

# Per-game plain-English explanation thresholds (see `explain_game`).
DEFAULT_MATERIALITY_POINTS = 0.25
DEFAULT_NEGLIGIBLE_GAP_POINTS = 0.5
DEFAULT_MAX_DRIVERS = 3
DEFAULT_MAX_OFFSETS = 1


# ---------------------------------------------------------------------------
# Family registry plumbing
# ---------------------------------------------------------------------------


def build_family_map(
    feature_columns: Sequence[str],
    families: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str]:
    """Invert a feature-family registry into a feature -> family lookup.

    Every element of ``feature_columns`` must be covered by exactly one
    family in ``families`` (default: ``nfl_ats.constants.FEATURE_FAMILIES``,
    the project's honest attribution unit). Raising on an uncovered or
    doubly-covered feature keeps the registry authoritative: a raw feature
    added to a feature set without a family assignment fails loudly here
    instead of silently vanishing from every downstream weight table.
    """

    registry = FEATURE_FAMILIES if families is None else families
    lookup: dict[str, str] = {}
    for family, members in registry.items():
        for member in members:
            if member in lookup:
                raise ValueError(
                    f"Feature {member!r} is claimed by both {lookup[member]!r} and {family!r}"
                )
            lookup[member] = family
    missing = [column for column in feature_columns if column not in lookup]
    if missing:
        raise ValueError(f"Features without a family assignment: {', '.join(missing)}")
    return {column: lookup[column] for column in feature_columns}


def _family_for_design_column(name: str, family_map: Mapping[str, str]) -> str:
    """Resolve a fitted pipeline's design-column name to its feature family.

    ``SimpleImputer(add_indicator=True)`` appends ``missing:<feature>``
    indicator columns after the original features (see
    :func:`nfl_ats.modeling.logistic_coefficients` for the same convention).
    An indicator is attributed to its source feature's family: "we don't
    observe this family's inputs for this game" is itself part of that
    family's signal.
    """

    base = name[len("missing:") :] if name.startswith("missing:") else name
    return family_map[base]


def decomposition_feature_columns(
    feature_profile: MarginFeatureProfile = DEFAULT_FEATURE_PROFILE,
) -> tuple[str, ...]:
    """The single, shared ``X`` for the three matched regressions.

    Uses the *margin*-target feature contract for ``feature_profile`` (e.g.
    ``football_player`` for the active ``"player"`` profile) rather than the
    market-residual contract, because it deliberately excludes the market
    family (``spread_line``, ``total_line``). All three matched regressions
    must share identical inputs for the reconciliation identity in
    :func:`walk_forward_decomposition` to hold, and a market feature
    predicting the market target directly would be a degenerate fit.
    """

    return margin_feature_columns("margin", feature_profile)


# ---------------------------------------------------------------------------
# Shared pipeline plumbing
# ---------------------------------------------------------------------------


def _target_series(frame: pd.DataFrame, target: str) -> pd.Series:
    if target == "margin":
        return pd.to_numeric(frame["result"], errors="coerce")
    if target == "spread":
        return pd.to_numeric(frame["spread_line"], errors="coerce")
    if target == "residual":
        return pd.to_numeric(frame["result"], errors="coerce") - pd.to_numeric(
            frame["spread_line"], errors="coerce"
        )
    raise ValueError(
        f"Unknown decomposition target {target!r}; choose one of {DECOMPOSITION_TARGETS}"
    )


def _design_column_names(estimator: Pipeline, feature_columns: Sequence[str]) -> list[str]:
    imputer = estimator.named_steps["imputer"]
    indicator_indices = getattr(imputer.indicator_, "features_", np.array([], dtype=int))
    names = list(feature_columns)
    names.extend(f"missing:{feature_columns[int(index)]}" for index in indicator_indices)
    return names


def _standardized_design(
    estimator: Pipeline, frame: pd.DataFrame, feature_columns: Sequence[str]
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Standardized design matrix and column names for a fitted margin-style pipeline.

    Mirrors the fitted pipeline's ``imputer -> scaler`` steps exactly, so the
    returned columns line up 1:1 with
    ``estimator.named_steps["regressor"].coef_``.
    """

    imputer = estimator.named_steps["imputer"]
    scaler = estimator.named_steps["scaler"]
    imputed = imputer.transform(frame.loc[:, list(feature_columns)])
    standardized = np.asarray(scaler.transform(imputed), dtype=np.float64)
    return standardized, _design_column_names(estimator, feature_columns)


def _fit_ridge(
    training: pd.DataFrame, feature_columns: Sequence[str], y: pd.Series, ridge_alpha: float
) -> Pipeline:
    estimator = make_margin_estimator("ridge", ridge_alpha=ridge_alpha)
    estimator.fit(training.loc[:, list(feature_columns)], y)
    return estimator


# ---------------------------------------------------------------------------
# 1. Three matched walk-forward regressions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardDecomposition:
    coefficients: pd.DataFrame  # season, week, target, feature, family, coefficient
    predictions: pd.DataFrame  # season, week, game_id, target, predicted, actual
    reconciliation: pd.DataFrame  # season, week, game_id, error
    refit_weeks: int
    feature_columns: tuple[str, ...]
    ridge_alpha: float
    min_train_games: int
    start_season: int
    end_season: int


def walk_forward_decomposition(
    features: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    start_season: int = DEFAULT_START_SEASON,
    end_season: int = DEFAULT_END_SEASON,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    reconciliation_atol: float = RECONCILIATION_ATOL,
    families: Mapping[str, Sequence[str]] | None = None,
) -> WalkForwardDecomposition:
    """Fit matched margin/spread/residual ridge regressions on weekly refits.

    Mirrors :func:`nfl_ats.outcomes.walk_forward_outcomes`'s cadence exactly:
    for every ``(season, week)`` in ``[start_season, end_season]``, training
    is every completed game strictly before that week's earliest kickoff,
    refit before every week. Coefficients from every refit window are
    retained (long format) so callers can report both the mean standardized
    coefficient and its variability across refits, not just a single
    full-history fit.

    ``families`` overrides the default ``nfl_ats.constants.FEATURE_FAMILIES``
    registry (see :func:`build_family_map`) -- production callers should
    leave it unset; it exists so tests can exercise the classification logic
    against a small, fully controlled synthetic family map.
    """

    feature_columns = tuple(feature_columns)
    if not feature_columns:
        raise ValueError("At least one feature column is required")
    if end_season < start_season:
        raise ValueError("end_season cannot be earlier than start_season")
    required_columns = set(feature_columns) | {
        "result",
        "spread_line",
        "gameday",
        "season",
        "week",
        "game_id",
    }
    missing_columns = sorted(required_columns.difference(features.columns))
    if missing_columns:
        raise DataContractError(
            f"Decomposition table is missing columns: {', '.join(missing_columns)}"
        )

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna() & frame["spread_line"].notna()].copy()
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError(
            f"No completed games found from season {start_season} through {end_season}"
        )

    family_map = build_family_map(feature_columns, families)
    coefficient_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []
    refit_weeks = 0

    for (season, week), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        refit_weeks += 1
        fitted: dict[str, Pipeline] = {}
        for target in DECOMPOSITION_TARGETS:
            estimator = _fit_ridge(
                training, feature_columns, _target_series(training, target), ridge_alpha
            )
            fitted[target] = estimator
            ridge = estimator.named_steps["regressor"]
            names = _design_column_names(estimator, feature_columns)
            coefficients = np.asarray(ridge.coef_, dtype=np.float64)
            if len(names) != len(coefficients):
                raise RuntimeError(
                    "Unable to align decomposition coefficients with transformed features"
                )
            for name, coefficient in zip(names, coefficients, strict=True):
                coefficient_rows.append(
                    {
                        "season": int(str(season)),
                        "week": int(str(week)),
                        "target": target,
                        "feature": name,
                        "family": _family_for_design_column(name, family_map),
                        "coefficient": float(coefficient),
                    }
                )
            predicted = np.asarray(
                estimator.predict(weekly_games.loc[:, list(feature_columns)]), dtype=np.float64
            )
            actual = _target_series(weekly_games, target).to_numpy(dtype=np.float64)
            for game_id, predicted_value, actual_value in zip(
                weekly_games["game_id"], predicted, actual, strict=True
            ):
                prediction_rows.append(
                    {
                        "season": int(str(season)),
                        "week": int(str(week)),
                        "game_id": game_id,
                        "target": target,
                        "predicted": float(predicted_value),
                        "actual": float(actual_value) if np.isfinite(actual_value) else np.nan,
                    }
                )

        pred_margin = np.asarray(
            fitted["margin"].predict(weekly_games.loc[:, list(feature_columns)]), dtype=np.float64
        )
        pred_spread = np.asarray(
            fitted["spread"].predict(weekly_games.loc[:, list(feature_columns)]), dtype=np.float64
        )
        pred_residual = np.asarray(
            fitted["residual"].predict(weekly_games.loc[:, list(feature_columns)]), dtype=np.float64
        )
        error = np.abs((pred_margin - pred_spread) - pred_residual)
        if error.max() > reconciliation_atol:
            raise RuntimeError(
                f"Reconciliation identity (a)-(b)=(c) failed for {season} week {week}: "
                f"max |error| = {error.max():.6g} exceeds tolerance {reconciliation_atol:.2g}"
            )
        for game_id, err in zip(weekly_games["game_id"], error, strict=True):
            reconciliation_rows.append(
                {
                    "season": int(str(season)),
                    "week": int(str(week)),
                    "game_id": game_id,
                    "error": float(err),
                }
            )

    if refit_weeks == 0:
        raise ValueError("No walk-forward window had enough prior training games")

    return WalkForwardDecomposition(
        coefficients=pd.DataFrame(coefficient_rows),
        predictions=pd.DataFrame(prediction_rows),
        reconciliation=pd.DataFrame(reconciliation_rows),
        refit_weeks=refit_weeks,
        feature_columns=feature_columns,
        ridge_alpha=ridge_alpha,
        min_train_games=min_train_games,
        start_season=start_season,
        end_season=end_season,
    )


# ---------------------------------------------------------------------------
# 2. Family aggregation and four-bucket classification
# ---------------------------------------------------------------------------


def family_weights_table(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a walk-forward coefficient table to family-level weights.

    Returns one row per ``(target, family)`` with:

    - ``mean_abs_weight`` / ``share``: the family's mean per-refit L1
      (sum of ``|coefficient|``) weight, and that weight's share of the
      target's total mean weight across every family -- the quantity the
      four-bucket classification thresholds are defined on.
    - ``mean_signed_weight``: mean per-refit signed sum, informative only
      when a family's features do not cancel each other out (ridge smears
      correlated features, so a near-zero signed weight can still hide real,
      offsetting predictive content -- read family-level only, never as a
      "this family doesn't matter" conclusion on its own).
    - ``refit_std_abs_weight``: standard deviation of the per-refit weight
      across every weekly refit window (fine-grained variability).
    - ``season_std_abs_weight``: standard deviation of each season's *mean*
      per-refit weight across seasons (season-level stability).
    """

    if coefficients.empty:
        raise ValueError("Coefficient table is empty")
    per_refit = coefficients.groupby(["season", "week", "target", "family"], as_index=False).agg(
        abs_weight=("coefficient", lambda values: float(np.abs(values).sum())),
        signed_weight=("coefficient", "sum"),
    )
    season_level = per_refit.groupby(["season", "target", "family"], as_index=False).agg(
        season_mean_abs_weight=("abs_weight", "mean"),
    )
    overall = per_refit.groupby(["target", "family"], as_index=False).agg(
        mean_abs_weight=("abs_weight", "mean"),
        refit_std_abs_weight=("abs_weight", "std"),
        mean_signed_weight=("signed_weight", "mean"),
        refits=("abs_weight", "size"),
    )
    season_stability = season_level.groupby(["target", "family"], as_index=False).agg(
        season_std_abs_weight=("season_mean_abs_weight", "std"),
        seasons=("season_mean_abs_weight", "size"),
    )
    result = overall.merge(season_stability, on=["target", "family"], how="left")
    result["refit_std_abs_weight"] = result["refit_std_abs_weight"].fillna(0.0)
    result["season_std_abs_weight"] = result["season_std_abs_weight"].fillna(0.0)
    result["share"] = result.groupby("target")["mean_abs_weight"].transform(
        lambda values: values / values.sum() if values.sum() else 0.0
    )
    return result.sort_values(["target", "share"], ascending=[True, False], ignore_index=True)


def classify_family(
    spread_share: float,
    margin_share: float,
    *,
    noise_share_threshold: float = DEFAULT_NOISE_SHARE_THRESHOLD,
    overpriced_ratio_threshold: float = DEFAULT_OVERPRICED_RATIO_THRESHOLD,
) -> str:
    """Four-bucket classification of one family's spread vs. margin share.

    - ``noise``: both shares are at or below ``noise_share_threshold``.
    - ``unpriced_predictive``: spread share is at or below the noise
      threshold, but margin share is not -- the market gives this family
      approximately zero weight, reality does not.
    - ``overpriced``: spread share is not noise, and either margin share is
      noise, or ``spread_share / margin_share >= overpriced_ratio_threshold``
      -- the market weights this family meaningfully more than reality
      warrants.
    - ``priced``: everything else -- both shares are non-trivial and roughly
      proportionate.
    """

    if not 0.0 <= spread_share <= 1.0 or not 0.0 <= margin_share <= 1.0:
        raise ValueError("shares must be between 0 and 1")
    spread_noise = spread_share <= noise_share_threshold
    margin_noise = margin_share <= noise_share_threshold
    if spread_noise and margin_noise:
        return "noise"
    if spread_noise:
        return "unpriced_predictive"
    if margin_noise:
        return "overpriced"
    if spread_share / margin_share >= overpriced_ratio_threshold:
        return "overpriced"
    return "priced"


def classify_families(
    family_weights: pd.DataFrame,
    *,
    noise_share_threshold: float = DEFAULT_NOISE_SHARE_THRESHOLD,
    overpriced_ratio_threshold: float = DEFAULT_OVERPRICED_RATIO_THRESHOLD,
) -> pd.DataFrame:
    """Pivot :func:`family_weights_table`'s output to one row per family.

    Columns: ``family``, and for each of ``margin``/``spread``/``residual``:
    ``weight_in_<target>``, ``<target>_share``, ``net_signed_in_<target>``,
    ``refit_std_in_<target>``, ``season_std_in_<target>`` (the "stability
    across seasons (per-season refit spread)" the BUILD spec asks for), plus
    a final ``classification`` column from :func:`classify_family`.
    """

    required_targets = set(DECOMPOSITION_TARGETS)
    missing_targets = required_targets.difference(family_weights["target"].unique())
    if missing_targets:
        raise ValueError(f"family_weights is missing targets: {sorted(missing_targets)}")

    by_target = {
        target: family_weights.loc[family_weights["target"].eq(target)].set_index("family")
        for target in DECOMPOSITION_TARGETS
    }
    families = sorted(set().union(*(frame.index for frame in by_target.values())))

    rows: list[dict[str, Any]] = []
    for family in families:
        row: dict[str, Any] = {"family": family}
        for target in DECOMPOSITION_TARGETS:
            frame = by_target[target]
            if family in frame.index:
                values = frame.loc[family]
                row[f"weight_in_{target}"] = float(values["mean_abs_weight"])
                row[f"{target}_share"] = float(values["share"])
                row[f"net_signed_in_{target}"] = float(values["mean_signed_weight"])
                row[f"refit_std_in_{target}"] = float(values["refit_std_abs_weight"])
                row[f"season_std_in_{target}"] = float(values["season_std_abs_weight"])
            else:
                row[f"weight_in_{target}"] = 0.0
                row[f"{target}_share"] = 0.0
                row[f"net_signed_in_{target}"] = 0.0
                row[f"refit_std_in_{target}"] = 0.0
                row[f"season_std_in_{target}"] = 0.0
        row["classification"] = classify_family(
            spread_share=row["spread_share"],
            margin_share=row["margin_share"],
            noise_share_threshold=noise_share_threshold,
            overpriced_ratio_threshold=overpriced_ratio_threshold,
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("margin_share", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. R^2 accounting and reconciliation reporting
# ---------------------------------------------------------------------------


def r_squared_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample, walk-forward R^2 per target, pooling every held-out week.

    Every prediction is genuinely out of sample (scored by the refit whose
    training cutoff strictly precedes that week), so this is explanatory
    bookkeeping on a real leak-safe evaluation -- not a new prospective
    claim; see the module docstring's multiplicity note. The spread target's
    R^2 measures how much of the market's own line is reconstructable from
    non-market features (expect high); the margin target's R^2 measures how
    much of the actual outcome those same features explain (expect low --
    football is noisy). The gap between them is, by construction,
    information the market has that these features do not.
    """

    rows: list[dict[str, Any]] = []
    for target, group in predictions.groupby("target"):
        valid = group.dropna(subset=["actual"])
        if len(valid) < 2:
            continue
        error = valid["actual"] - valid["predicted"]
        rows.append(
            {
                "target": target,
                "games": len(valid),
                "r_squared": float(r2_score(valid["actual"], valid["predicted"])),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
            }
        )
    return pd.DataFrame(rows).sort_values("target", ignore_index=True)


def reconciliation_summary(reconciliation: pd.DataFrame) -> dict[str, float]:
    """Summarize the per-game reconciliation errors from a walk-forward run."""

    if reconciliation.empty:
        raise ValueError("Reconciliation table is empty")
    return {
        "games": len(reconciliation),
        "mean_abs_error": float(reconciliation["error"].mean()),
        "max_abs_error": float(reconciliation["error"].max()),
        "p99_abs_error": float(reconciliation["error"].quantile(0.99)),
    }


# ---------------------------------------------------------------------------
# 4. Opener variant (2025 free open/close sample; directional only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenerVariantResult:
    available: bool
    reason: str | None
    games: int
    coefficients: pd.DataFrame | None  # target, feature, family, coefficient
    family_weights: pd.DataFrame | None  # target, family, abs_weight, share
    r_squared: pd.DataFrame | None  # in-sample; directional only, see module docs


def _opener_unavailable(reason: str, *, games: int = 0) -> OpenerVariantResult:
    return OpenerVariantResult(
        available=False,
        reason=reason,
        games=games,
        coefficients=None,
        family_weights=None,
        r_squared=None,
    )


def latest_open_close_games_path(root: Path) -> Path | None:
    """Feature-detect the most recent open/close snapshot's ``games.parquet``.

    Mirrors ``nfl_ats.cli._cmd_market_open_close_backfill``'s destination
    (``data/market/historical/open_close/raw``): each snapshot is a
    timestamp-named directory, so the lexicographically last one containing
    a ``games.parquet`` is the most recent. Returns ``None`` if no snapshot
    has been fetched yet -- this module performs no network I/O itself.
    """

    if not root.is_dir():
        return None
    candidates = sorted(
        entry for entry in root.iterdir() if entry.is_dir() and (entry / "games.parquet").is_file()
    )
    if not candidates:
        return None
    return candidates[-1] / "games.parquet"


def opener_variant_decomposition(
    opener_games: pd.DataFrame,
    features: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    min_games: int = OPENER_MIN_GAMES_DEFAULT,
    families: Mapping[str, Sequence[str]] | None = None,
) -> OpenerVariantResult:
    """Fit ``spread_open ~ X`` and ``(close - open) ~ X`` on the opener sample.

    A single in-sample ridge fit (no walk-forward: the sample is small,
    roughly 270 games from one season) -- report as directional only, per
    the BUILD spec. ``opener_games`` is
    ``nfl_ats.open_close_market.summarize_open_close_games``'s output
    (``nflverse_game_id``, ``opening_home_spread``,
    ``consensus_closing_home_spread``); rows with an inconsistent or
    unmatched opener/close are already ``NaN`` there and are dropped here.
    ``families`` overrides the default family registry; see
    :func:`walk_forward_decomposition`.
    """

    feature_columns = tuple(feature_columns)
    required = {"nflverse_game_id", "opening_home_spread", "consensus_closing_home_spread"}
    missing = sorted(required.difference(opener_games.columns))
    if missing:
        return _opener_unavailable(f"opener table is missing columns: {', '.join(missing)}")

    matched = opener_games.dropna(
        subset=["nflverse_game_id", "opening_home_spread", "consensus_closing_home_spread"]
    )
    if matched.empty:
        return _opener_unavailable(
            "no opener rows have both a matched nflverse game id and a consistent open and close"
        )

    joined = matched.merge(
        features.loc[:, ["game_id", *feature_columns]],
        left_on="nflverse_game_id",
        right_on="game_id",
        how="inner",
    )
    if len(joined) < min_games:
        return _opener_unavailable(
            f"only {len(joined)} matched games; need at least {min_games}", games=len(joined)
        )

    joined = joined.copy()
    joined["absorption"] = joined["consensus_closing_home_spread"] - joined["opening_home_spread"]
    family_map = build_family_map(feature_columns, families)

    coefficient_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for target, y in (
        ("open", joined["opening_home_spread"]),
        ("absorption", joined["absorption"]),
    ):
        estimator = _fit_ridge(joined, feature_columns, y, ridge_alpha)
        ridge = estimator.named_steps["regressor"]
        names = _design_column_names(estimator, feature_columns)
        coefficients = np.asarray(ridge.coef_, dtype=np.float64)
        for name, coefficient in zip(names, coefficients, strict=True):
            coefficient_rows.append(
                {
                    "target": target,
                    "feature": name,
                    "family": _family_for_design_column(name, family_map),
                    "coefficient": float(coefficient),
                }
            )
        predicted = np.asarray(
            estimator.predict(joined.loc[:, list(feature_columns)]), dtype=np.float64
        )
        actual = y.to_numpy(dtype=np.float64)
        for game_id, predicted_value, actual_value in zip(
            joined["game_id"], predicted, actual, strict=True
        ):
            prediction_rows.append(
                {
                    "target": target,
                    "game_id": game_id,
                    "predicted": float(predicted_value),
                    "actual": float(actual_value),
                }
            )

    coefficients_frame = pd.DataFrame(coefficient_rows)
    weights = coefficients_frame.groupby(["target", "family"], as_index=False).agg(
        abs_weight=("coefficient", lambda values: float(np.abs(values).sum())),
    )
    weights["share"] = weights.groupby("target")["abs_weight"].transform(
        lambda values: values / values.sum() if values.sum() else 0.0
    )
    weights = weights.sort_values(["target", "share"], ascending=[True, False], ignore_index=True)

    return OpenerVariantResult(
        available=True,
        reason=None,
        games=len(joined),
        coefficients=coefficients_frame,
        family_weights=weights,
        r_squared=r_squared_table(pd.DataFrame(prediction_rows)),
    )


# ---------------------------------------------------------------------------
# 6. Plain-English, pick-side-oriented per-game explanations
# ---------------------------------------------------------------------------

# Every key of `nfl_ats.constants.FEATURE_FAMILIES`, plus the `intercept`
# sentinel used only inside attribution, mapped to a phrase a non-modeler
# can read without knowing any column names. `test_market_decomposition.py`
# asserts every registry family is covered here.
# Contributions from design columns whose value is identical for every game
# in the attributed slate (week-of-season encodings, a uniformly-zero rest
# differential in week 1, ...) cannot explain why one game differs from the
# market more than another. They are routed to this pseudo-family and kept
# out of the per-game "mainly because" narrative; the additive identity is
# preserved because the bucket still appears in the attribution rows.
WEEKLY_CONTEXT_FAMILY = "weekly_context"

FAMILY_PHRASES: dict[str, str] = {
    "market": "the market line itself",
    "context": "rest and schedule spots",
    WEEKLY_CONTEXT_FAMILY: "a league-wide adjustment shared by every game this week",
    "elo": "overall team strength ratings",
    "experience": "team experience (games played)",
    "offense": "recent offensive performance (opponent-adjusted efficiency)",
    "results": "recent scoring margin and against-the-spread trend",
    "defense": "recent defensive performance (opponent-adjusted efficiency)",
    "pbp": "advanced play-by-play efficiency",
    "pbp_opponent_adjusted": "opponent-adjusted play-by-play matchups",
    "drive": "drive-level efficiency",
    "quarterback": "quarterback efficiency (EPA and CPOE)",
    "player_qb": "quarterback status and recent play",
    "player_injuries": "expected player availability",
    "player_continuity": "lineup continuity",
    "player_values": "estimated value lost to injuries",
    "player_values_js_prior": "estimated value lost to injuries (position-prior shrinkage)",
    "player_participation_values": "participation-weighted value lost to injuries",
    "graph": "opponent-network strength ratings",
    "schedule_rating": "strength of schedule",
    "bias": "documented early-season line biases (playoff holdovers, last week's result)",
    "surface_switch": "a grass-accustomed visitor playing on artificial turf",
    "gap_v3_bias": "division revenge, sandwich spots, and post-blowout letdown/bounce",
    "gap_v3_penalty": "each team's recent penalty rate",
    "gap_v3_travel": "short-week Thursday games and return-trip travel fatigue",
    "forecast_weather": "the forecast temperature, wind and rain at kickoff",
    INTERCEPT_FAMILY: "the model's baseline adjustment",
}


def _phrase_for_family(family: str, phrases: Mapping[str, str]) -> str:
    return phrases.get(family, family.replace("_", " "))


def _join_with_and(items: Sequence[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


@dataclass(frozen=True)
class DriverContribution:
    family: str
    phrase: str
    points: float  # pick-side-oriented: positive supports the pick, negative opposes it


@dataclass(frozen=True)
class GameExplanation:
    game_id: str
    pick_side: str  # "HOME" or "AWAY"
    pick_team: str
    other_team: str
    gap_points: float  # abs(predicted_residual); always >= 0
    drivers: tuple[DriverContribution, ...]
    offsets: tuple[DriverContribution, ...]
    sentence: str


def explain_game_structured(
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    predicted_residual: float,
    family_contributions: Mapping[str, float],
    phrases: Mapping[str, str] = FAMILY_PHRASES,
    materiality_threshold: float = DEFAULT_MATERIALITY_POINTS,
    negligible_gap_threshold: float = DEFAULT_NEGLIGIBLE_GAP_POINTS,
    max_drivers: int = DEFAULT_MAX_DRIVERS,
    max_offsets: int = DEFAULT_MAX_OFFSETS,
) -> GameExplanation:
    """Plain-English, pick-side-oriented explanation of one game's model-vs-market gap.

    ``family_contributions`` must be home-oriented points (the same sign
    convention as ``predicted_residual = predicted_margin - spread_line``;
    positive favors the home team), exactly as :func:`attribute_predictions`
    produces. This function re-orients every contribution to the *pick
    side* -- the side the model favors more than the market does -- so a
    positive ``points`` value always means "supports the pick" regardless of
    whether the model leans home or away. That re-orientation is the one
    piece of sign logic every caller must get right, so it lives here, once,
    rather than at each call site.

    Only contributors with ``|points| >= materiality_threshold`` are named,
    capped at ``max_drivers`` supporting drivers (highest first) and
    ``max_offsets`` opposing drivers (lowest first). When the total gap is
    below ``negligible_gap_threshold``, the sentence degrades to "the model
    essentially agrees with the market" regardless of any individual
    family's size.
    """

    if materiality_threshold < 0:
        raise ValueError("materiality_threshold must be non-negative")
    if negligible_gap_threshold < 0:
        raise ValueError("negligible_gap_threshold must be non-negative")

    pick_side = "HOME" if predicted_residual >= 0.0 else "AWAY"
    pick_team = home_team if pick_side == "HOME" else away_team
    other_team = away_team if pick_side == "HOME" else home_team
    pick_sign = 1.0 if pick_side == "HOME" else -1.0
    gap_points = abs(float(predicted_residual))
    reoriented = {
        family: pick_sign * float(value) for family, value in family_contributions.items()
    }

    if gap_points < negligible_gap_threshold:
        sentence = (
            "The model essentially agrees with the market on this game "
            f"(a {gap_points:.1f}-point gap)."
        )
        return GameExplanation(
            game_id, pick_side, pick_team, other_team, gap_points, (), (), sentence
        )

    # Slate-shared terms (weekly context, the fitted intercept) apply to
    # every game equally, so they never appear as game-specific drivers or
    # offsets; a material shared component is disclosed in a trailing note.
    shared_families = {WEEKLY_CONTEXT_FAMILY, INTERCEPT_FAMILY}
    shared_points = sum(value for family, value in reoriented.items() if family in shared_families)
    game_specific = {
        family: value for family, value in reoriented.items() if family not in shared_families
    }
    driver_pool = sorted(
        (
            (family, value)
            for family, value in game_specific.items()
            if value >= materiality_threshold
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:max_drivers]
    offset_pool = sorted(
        (
            (family, value)
            for family, value in game_specific.items()
            if value <= -materiality_threshold
        ),
        key=lambda item: item[1],
    )[:max_offsets]
    drivers = tuple(
        DriverContribution(family, _phrase_for_family(family, phrases), value)
        for family, value in driver_pool
    )
    offsets = tuple(
        DriverContribution(family, _phrase_for_family(family, phrases), value)
        for family, value in offset_pool
    )

    if not drivers:
        sentence = (
            f"The model leans {pick_team} {gap_points:.1f} points more than the market, but no "
            f"single feature family clears the {materiality_threshold:.2f}-point materiality "
            "bar -- the gap is spread across many small factors."
        )
    else:
        driver_text = _join_with_and(
            [f"{driver.phrase} (+{driver.points:.1f})" for driver in drivers]
        )
        offset_text = (
            ", partly offset by "
            + _join_with_and([f"{offset.phrase} ({offset.points:.1f})" for offset in offsets])
            if offsets
            else ""
        )
        sentence = (
            f"The model leans {pick_team} {gap_points:.1f} points more than the market mainly "
            f"because of {driver_text}{offset_text}."
        )
    if abs(shared_points) >= materiality_threshold:
        sentence += (
            f" (A general adjustment of {shared_points:+.1f} applied equally to every game "
            "this week is included in the gap.)"
        )
    return GameExplanation(
        game_id, pick_side, pick_team, other_team, gap_points, drivers, offsets, sentence
    )


def explain_game(
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    predicted_residual: float,
    family_contributions: Mapping[str, float],
    phrases: Mapping[str, str] = FAMILY_PHRASES,
    materiality_threshold: float = DEFAULT_MATERIALITY_POINTS,
    negligible_gap_threshold: float = DEFAULT_NEGLIGIBLE_GAP_POINTS,
    max_drivers: int = DEFAULT_MAX_DRIVERS,
    max_offsets: int = DEFAULT_MAX_OFFSETS,
) -> str:
    """The rendered sentence from :func:`explain_game_structured`."""

    return explain_game_structured(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        predicted_residual=predicted_residual,
        family_contributions=family_contributions,
        phrases=phrases,
        materiality_threshold=materiality_threshold,
        negligible_gap_threshold=negligible_gap_threshold,
        max_drivers=max_drivers,
        max_offsets=max_offsets,
    ).sentence


# ---------------------------------------------------------------------------
# 5. Per-game attribution
# ---------------------------------------------------------------------------


def attribute_predictions(
    features: pd.DataFrame,
    *,
    season: int,
    week: int,
    feature_columns: Sequence[str],
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    materiality_threshold: float = DEFAULT_MATERIALITY_POINTS,
    negligible_gap_threshold: float = DEFAULT_NEGLIGIBLE_GAP_POINTS,
    max_drivers: int = DEFAULT_MAX_DRIVERS,
    max_offsets: int = DEFAULT_MAX_OFFSETS,
    families: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Exact per-game, per-family decomposition of the predicted residual.

    Refits ``(result - spread_line) ~ X`` -- the same matched-regression
    design :func:`walk_forward_decomposition` uses for its ``"residual"``
    target -- on every completed game strictly before ``season``/``week``'s
    earliest kickoff (leak-safe walk-forward, consistent with the rest of
    this module), then decomposes each target game's prediction into
    coefficient x standardized-value contributions summed per feature
    family, plus one ``"intercept"`` row for the fitted model's constant
    term. See the module docstring for the returned schema. Raises if the
    additive identity does not hold within :data:`ATTRIBUTION_ATOL`.
    ``families`` overrides the default family registry; see
    :func:`walk_forward_decomposition`.
    """

    feature_columns = tuple(feature_columns)
    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    target_games = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)].copy()
    if target_games.empty:
        raise ValueError(f"No games found for {season} week {week}")
    if target_games["spread_line"].isna().any():
        raise DataContractError(
            "Cannot attribute a residual prediction while a target spread is missing"
        )

    cutoff = target_games["gameday"].min()
    completed = frame.loc[
        frame["gameday"].lt(cutoff) & frame["result"].notna() & frame["spread_line"].notna()
    ]
    if len(completed) < min_train_games:
        raise ValueError(
            f"Only {len(completed)} eligible games precede {season} week {week}; "
            f"need {min_train_games}"
        )

    estimator = _fit_ridge(
        completed, feature_columns, _target_series(completed, "residual"), ridge_alpha
    )
    ridge = estimator.named_steps["regressor"]
    standardized, names = _standardized_design(estimator, target_games, feature_columns)
    coefficients = np.asarray(ridge.coef_, dtype=np.float64)
    intercept = float(ridge.intercept_)
    if len(names) != len(coefficients):
        raise RuntimeError("Unable to align attribution coefficients with transformed features")

    family_map = build_family_map(feature_columns, families)
    families_by_column = [_family_for_design_column(name, family_map) for name in names]
    # Design columns identical across the whole slate (week-of-season terms,
    # a uniformly-zero rest differential, ...) contribute the same points to
    # every game and cannot explain game-to-game differences vs the market;
    # route them to the shared weekly-context bucket. A single-game slate
    # cannot make the distinction, so it keeps the ordinary families.
    if len(target_games) > 1:
        slate_constant = np.all(np.isclose(standardized, standardized[0:1, :], atol=1e-12), axis=0)
        families_by_column = [
            WEEKLY_CONTEXT_FAMILY if slate_constant[index] else family
            for index, family in enumerate(families_by_column)
        ]
    unique_families = sorted(set(families_by_column))

    contributions = standardized * coefficients[np.newaxis, :]
    predicted = np.asarray(
        estimator.predict(target_games.loc[:, list(feature_columns)]), dtype=np.float64
    )
    actual = (
        pd.to_numeric(target_games["result"], errors="coerce")
        - pd.to_numeric(target_games["spread_line"], errors="coerce")
    ).to_numpy(dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for row_index, (_, game) in enumerate(target_games.iterrows()):
        family_totals: dict[str, float] = dict.fromkeys(unique_families, 0.0)
        for column_index, family in enumerate(families_by_column):
            family_totals[family] += float(contributions[row_index, column_index])
        family_totals[INTERCEPT_FAMILY] = intercept

        total = sum(family_totals.values())
        predicted_value = float(predicted[row_index])
        if not np.isclose(total, predicted_value, atol=ATTRIBUTION_ATOL, rtol=1e-6):
            raise RuntimeError(
                f"Attribution does not reconcile for {game['game_id']}: "
                f"sum(contributions)={total:.6f} != predicted={predicted_value:.6f}"
            )

        actual_value = float(actual[row_index])
        explanation = explain_game(
            game_id=str(game["game_id"]),
            home_team=str(game["home_team"]),
            away_team=str(game["away_team"]),
            predicted_residual=predicted_value,
            family_contributions=family_totals,
            materiality_threshold=materiality_threshold,
            negligible_gap_threshold=negligible_gap_threshold,
            max_drivers=max_drivers,
            max_offsets=max_offsets,
        )
        for family, contribution in family_totals.items():
            rows.append(
                {
                    "game_id": game["game_id"],
                    "season": int(season),
                    "week": int(week),
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "family": family,
                    "contribution": contribution,
                    "predicted_residual": predicted_value,
                    "actual_residual": actual_value if np.isfinite(actual_value) else np.nan,
                    "explanation": explanation,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level orchestration and markdown rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketDecompositionResult:
    feature_profile: str
    feature_columns: tuple[str, ...]
    start_season: int
    end_season: int
    ridge_alpha: float
    min_train_games: int
    refit_weeks: int
    coefficients: pd.DataFrame
    family_weights: pd.DataFrame
    classification: pd.DataFrame
    r_squared: pd.DataFrame
    reconciliation: dict[str, float]
    thresholds: dict[str, float]
    opener_variant: OpenerVariantResult


def run_market_decomposition(
    features: pd.DataFrame,
    *,
    feature_profile: MarginFeatureProfile = DEFAULT_FEATURE_PROFILE,
    start_season: int = DEFAULT_START_SEASON,
    end_season: int = DEFAULT_END_SEASON,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    noise_share_threshold: float = DEFAULT_NOISE_SHARE_THRESHOLD,
    overpriced_ratio_threshold: float = DEFAULT_OVERPRICED_RATIO_THRESHOLD,
    opener_games: pd.DataFrame | None = None,
    opener_min_games: int = OPENER_MIN_GAMES_DEFAULT,
) -> MarketDecompositionResult:
    """Run the full market decomposition: matched regressions through classification."""

    feature_columns = decomposition_feature_columns(feature_profile)
    walk_forward = walk_forward_decomposition(
        features,
        feature_columns=feature_columns,
        start_season=start_season,
        end_season=end_season,
        ridge_alpha=ridge_alpha,
        min_train_games=min_train_games,
    )
    family_weights = family_weights_table(walk_forward.coefficients)
    classification = classify_families(
        family_weights,
        noise_share_threshold=noise_share_threshold,
        overpriced_ratio_threshold=overpriced_ratio_threshold,
    )
    r_squared = r_squared_table(walk_forward.predictions)
    reconciliation = reconciliation_summary(walk_forward.reconciliation)

    opener_variant = (
        opener_variant_decomposition(
            opener_games,
            features,
            feature_columns=feature_columns,
            ridge_alpha=ridge_alpha,
            min_games=opener_min_games,
        )
        if opener_games is not None
        else _opener_unavailable("no opener sample supplied")
    )

    return MarketDecompositionResult(
        feature_profile=str(feature_profile),
        feature_columns=feature_columns,
        start_season=start_season,
        end_season=end_season,
        ridge_alpha=ridge_alpha,
        min_train_games=min_train_games,
        refit_weeks=walk_forward.refit_weeks,
        coefficients=walk_forward.coefficients,
        family_weights=family_weights,
        classification=classification,
        r_squared=r_squared,
        reconciliation=reconciliation,
        thresholds={
            "noise_share_threshold": noise_share_threshold,
            "overpriced_ratio_threshold": overpriced_ratio_threshold,
            "materiality_points": DEFAULT_MATERIALITY_POINTS,
            "negligible_gap_points": DEFAULT_NEGLIGIBLE_GAP_POINTS,
            "reconciliation_atol": RECONCILIATION_ATOL,
            "attribution_atol": ATTRIBUTION_ATOL,
        },
        opener_variant=opener_variant,
    )


def market_decomposition_markdown(
    result: MarketDecompositionResult,
    *,
    attribution: pd.DataFrame | None = None,
) -> str:
    """Human-readable markdown summary of a :func:`run_market_decomposition` result."""

    lines: list[str] = ["# Market decomposition", ""]
    lines.append(
        "Three matched ridge regressions -- `margin ~ X` (reality), `spread_line ~ X` (the "
        "market), and `(margin - spread_line) ~ X` (the residual) -- fit on identical games, "
        f"features, and preprocessing (`{result.feature_profile}` profile, "
        f"{len(result.feature_columns)} features, ridge alpha {result.ridge_alpha}), refit "
        f"weekly across {result.start_season}-{result.end_season} ({result.refit_weeks} refit "
        "windows), matching the active model's frozen configuration -- no tuning."
    )
    lines.append("")

    lines.append("## R^2 accounting (out-of-sample, walk-forward)")
    lines.append("")
    lines.append("| target | games | R^2 | MAE | RMSE |")
    lines.append("| --- | --- | --- | --- | --- |")
    for target in DECOMPOSITION_TARGETS:
        target_rows = result.r_squared.loc[result.r_squared["target"].eq(target)]
        if target_rows.empty:
            continue
        row = target_rows.iloc[0]
        games = int(row["games"])
        lines.append(
            f"| {target} | {games} | {row['r_squared']:.3f} | {row['mae']:.2f} | "
            f"{row['rmse']:.2f} |"
        )
    lines.append("")
    lines.append(
        "The spread model reconstructs how much of the market's own line is recoverable from "
        "these non-market features (expected to be high). The margin model's R^2 is expected "
        "to be low -- football outcomes are noisy -- and the gap between the spread model's "
        "R^2 and the margin model's R^2 is, by construction, information the market has that "
        "these features do not. Residual R^2 is reported for completeness only; see the "
        "honesty notes below before reading it as evidence of edge."
    )
    lines.append("")

    lines.append("## Reconciliation: (margin ~ X) - (spread ~ X) ~= (residual ~ X)")
    lines.append("")
    lines.append(
        f"- games checked: {result.reconciliation['games']}\n"
        f"- mean |error|: {result.reconciliation['mean_abs_error']:.3g} points\n"
        f"- max |error|: {result.reconciliation['max_abs_error']:.3g} points\n"
        f"- p99 |error|: {result.reconciliation['p99_abs_error']:.3g} points"
    )
    lines.append("")

    lines.append("## Family classification")
    lines.append("")
    thresholds_json = json.dumps(result.thresholds, sort_keys=True)
    lines.append(f"Thresholds (declared, not hardcoded magic): `{thresholds_json}`")
    lines.append("")
    lines.append(
        "| family | weight_in_spread | weight_in_margin | weight_in_residual | spread_share | "
        "margin_share | season_std_in_margin | classification |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, row in result.classification.iterrows():
        lines.append(
            f"| {row['family']} | {row['weight_in_spread']:.3f} | {row['weight_in_margin']:.3f} | "
            f"{row['weight_in_residual']:.3f} | {row['spread_share']:.1%} | "
            f"{row['margin_share']:.1%} | {row['season_std_in_margin']:.3f} | "
            f"{row['classification']} |"
        )
    lines.append("")

    lines.append("## Opener variant (2025 open/close sample; directional only)")
    lines.append("")
    opener = result.opener_variant
    if not opener.available:
        lines.append(f"Not available: {opener.reason}")
    else:
        lines.append(
            f"Matched games: {opener.games} (small-n, single in-sample fit -- not walk-forward; "
            "read as directional only)."
        )
        lines.append("")
        if opener.r_squared is not None:
            lines.append("| target | games | in-sample R^2 | MAE |")
            lines.append("| --- | --- | --- | --- |")
            for _, row in opener.r_squared.iterrows():
                games = int(row["games"])
                lines.append(
                    f"| {row['target']} | {games} | {row['r_squared']:.3f} | {row['mae']:.2f} |"
                )
            lines.append("")
        if opener.family_weights is not None:
            lines.append(
                "Top families predicting the opener and its intra-week absorption (close - open):"
            )
            lines.append("")
            lines.append("| target | family | share |")
            lines.append("| --- | --- | --- |")
            top = (
                opener.family_weights.sort_values(["target", "share"], ascending=[True, False])
                .groupby("target", sort=False)
                .head(5)
            )
            for _, row in top.iterrows():
                lines.append(f"| {row['target']} | {row['family']} | {row['share']:.1%} |")
    lines.append("")

    if attribution is not None and not attribution.empty:
        lines.append(
            "## Per-game attribution and plain-English explanations (latest prediction artifact)"
        )
        lines.append("")
        for game_id, game_rows in attribution.groupby("game_id", sort=False):
            sentence = str(game_rows["explanation"].iloc[0])
            lines.append(f"- **{game_id}**: {sentence}")
        lines.append("")

    lines.append("## Honesty notes")
    lines.append("")
    lines.append(
        "- `unpriced_predictive` is a hypothesis-generating diagnostic, not evidence of edge; "
        "the outer-season record remains the only adjudicator. See the 2014-2017 replication "
        "closure of the QB+continuity profile for what an actual adjudicated result looks like."
    )
    lines.append(
        "- Ridge regression smears weight across correlated features; only family-level "
        "aggregates (never an individual feature's coefficient) are meaningful here."
    )
    lines.append(
        "- This fits on 2018-2025 outcomes the project has already viewed and scored "
        "repeatedly. It is explanatory, not confirmatory, and scores no new pick stream -- it "
        "does not count as a new candidate stream under this project's multiplicity accounting."
    )
    lines.append("")
    return "\n".join(lines)
