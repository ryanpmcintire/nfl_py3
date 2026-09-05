"""Fair-margin and market-residual models with empirical predictive distributions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_ats.calibration import ResidualSmoothingMethod, smoothed_home_cover_probability
from nfl_ats.constants import (
    FEATURE_FAMILIES,
    FEATURE_SETS,
    MIN_FITTABLE_TRAIN_GAMES,
    SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS,
)
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
    "weak_stack_surface",
    "weak_stack_js_prior",
    "weak_stack_v3",
    "weak_stack_v4",
    "weak_stack_oracle_weather",
    "weak_stack_graph_sack",
    "weak_stack_graph_def_ypp",
    "weak_stack_graph_off_rush_epa",
    "weak_stack_fluview_home",
    "weak_stack_fluview_away",
    "weak_stack_illness_away",
    "weak_stack_illness_home",
    "weak_stack_reddit_ratio_home",
    "weak_stack_reddit_spike_away",
    "weak_stack_team_style_pace",
    "weak_stack_redzone_third_down",
    "weak_stack_durability",
    "weak_stack_source_availability",
    "weak_stack_post_ot",
    "weak_stack_mnf_road",
    "weak_stack_home_thursday",
    "weak_stack_opener_softness",
    "weak_stack_ml_divergence",
    "weak_stack_new_stadium",
    "weak_stack_dome_shootout",
    "weak_stack_low_total_div_dog",
    "weak_stack_sept_heat",
    "weak_stack_road_fav_fade",
    "weak_stack_division_dog",
    "weak_stack_week1_dog",
    "weak_stack_ats_streak_regress",
    "weak_stack_opening_drive_epa",
    "weak_stack_q3_point_diff",
    "weak_stack_fourth_down_interaction",
    "weak_stack_rookie_qb_debut_fade",
    "weak_stack_qb_revenge",
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
    "weak_stack_surface",
    "weak_stack_js_prior",
    "weak_stack_v3",
    "weak_stack_v4",
    "weak_stack_oracle_weather",
    "weak_stack_graph_sack",
    "weak_stack_graph_def_ypp",
    "weak_stack_graph_off_rush_epa",
    "weak_stack_fluview_home",
    "weak_stack_fluview_away",
    "weak_stack_illness_away",
    "weak_stack_illness_home",
    "weak_stack_reddit_ratio_home",
    "weak_stack_reddit_spike_away",
    "weak_stack_team_style_pace",
    "weak_stack_redzone_third_down",
    "weak_stack_durability",
    "weak_stack_source_availability",
    "weak_stack_post_ot",
    "weak_stack_mnf_road",
    "weak_stack_home_thursday",
    "weak_stack_opener_softness",
    "weak_stack_ml_divergence",
    "weak_stack_new_stadium",
    "weak_stack_dome_shootout",
    "weak_stack_low_total_div_dog",
    "weak_stack_sept_heat",
    "weak_stack_road_fav_fade",
    "weak_stack_division_dog",
    "weak_stack_week1_dog",
    "weak_stack_ats_streak_regress",
    "weak_stack_opening_drive_epa",
    "weak_stack_q3_point_diff",
    "weak_stack_fourth_down_interaction",
    "weak_stack_rookie_qb_debut_fade",
    "weak_stack_qb_revenge",
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
    # MOD-08 candidate profile (docs/surface_switch_feature_arm.md): weak_stack
    # plus the surface-switch tilt feature. Same table-pinning caveat as
    # weak_stack above -- fit only on a table carrying surface_switch_flag
    # (game_features_weak_stack_surface.parquet for this experiment).
    "weak_stack_surface": ("football_weak_stack_surface", "full_weak_stack_surface"),
    # MOD-06 candidate profile (docs/mod06_position_prior_shrinkage.md): weak_stack
    # with the player_values family replaced by player_values_js_prior. Same
    # table-pinning caveat as weak_stack above -- fit only on a table carrying
    # the *_js_prior columns (game_features_weak_stack_js_prior.parquet for
    # this experiment).
    "weak_stack_js_prior": ("football_weak_stack_js_prior", "full_weak_stack_js_prior"),
    # weak_stack_v3 candidate profile (docs/weak_stack_v3.md), MOD-07's
    # sequel: weak_stack_surface plus the registry gap columns computed in
    # nfl_ats.weak_stack_v3_features. Same table-pinning caveat as every
    # profile above -- fit only on a table carrying those columns
    # (game_features_weak_stack_v3.parquet for this experiment). Never used
    # by the active model.
    "weak_stack_v3": ("football_weak_stack_v3", "full_weak_stack_v3"),
    # weak_stack_v4 candidate profile (docs/weak_stack_v4.md): PRODUCTION
    # weak_stack plus the six continuous forecast-weather columns built in
    # nfl_ats.forecast_weather_features. Same table-pinning caveat as every
    # profile above -- fit only on a table carrying those columns
    # (game_features_weak_stack_v4.parquet for this experiment). Never used by
    # the active model.
    "weak_stack_v4": ("football_weak_stack_v4", "full_weak_stack_v4"),
    # POSITIVE CONTROL ONLY (docs/weak_stack_v4.md): weak_stack plus OBSERVED
    # weather. Deliberately leaky, never promotable -- it bounds the weather
    # channel rather than competing for production.
    "weak_stack_oracle_weather": (
        "football_weak_stack_oracle_weather",
        "full_weak_stack_oracle_weather",
    ),
    # weak_stack_graph_sack candidate profile (docs/graph_team_stat_on_production.md):
    # weak_stack plus the one graph-propagated off_sack_rate column. Same
    # table-pinning caveat as every profile above -- fit only on a table
    # carrying that column (game_features_weak_stack_graph_sack.parquet for
    # this experiment). Never used by the active model.
    "weak_stack_graph_sack": (
        "football_weak_stack_graph_sack",
        "full_weak_stack_graph_sack",
    ),
    # weak_stack_graph_def_ypp candidate profile
    # (docs/graph_team_stat_def_ypp_on_production.md): weak_stack plus the one
    # graph-propagated def_yards_per_play column. Same table-pinning caveat as
    # every profile above -- fit only on a table carrying that column
    # (game_features_weak_stack_graph_def_ypp.parquet for this experiment).
    # Never used by the active model.
    "weak_stack_graph_def_ypp": (
        "football_weak_stack_graph_def_ypp",
        "full_weak_stack_graph_def_ypp",
    ),
    # weak_stack_graph_off_rush_epa candidate profile
    # (docs/graph_team_stat_off_rush_epa_on_production.md): weak_stack plus the
    # one graph-propagated off_rush_epa_per_play column. Same table-pinning
    # caveat as every profile above -- fit only on a table carrying that column
    # (game_features_weak_stack_graph_off_rush_epa.parquet for this
    # experiment). Never used by the active model.
    "weak_stack_graph_off_rush_epa": (
        "football_weak_stack_graph_off_rush_epa",
        "full_weak_stack_graph_off_rush_epa",
    ),
    # weak_stack_fluview_home / weak_stack_fluview_away candidate profiles
    # (docs/fluview_on_production.md): weak_stack plus exactly one of the two
    # FluView elevated-illness columns. Same table-pinning caveat as every
    # profile above -- fit only on a table carrying those columns
    # (game_features_weak_stack_fluview.parquet for this experiment). Never
    # used by the active model.
    "weak_stack_fluview_home": (
        "football_weak_stack_fluview_home",
        "full_weak_stack_fluview_home",
    ),
    "weak_stack_fluview_away": (
        "football_weak_stack_fluview_away",
        "full_weak_stack_fluview_away",
    ),
    # 2026-09-01 on-production sweep (docs/on_production_sweep_20260901.md):
    # six candidate arms, each PRODUCTION weak_stack plus exactly one new
    # column. Same table-pinning caveat as every profile above -- fit only on
    # the widened table that carries the column
    # (game_features_weak_stack_illness / _reddit / _team_style_pace /
    # _redzone_third_down.parquet). Never used by the active model.
    "weak_stack_illness_away": (
        "football_weak_stack_illness_away",
        "full_weak_stack_illness_away",
    ),
    "weak_stack_illness_home": (
        "football_weak_stack_illness_home",
        "full_weak_stack_illness_home",
    ),
    "weak_stack_reddit_ratio_home": (
        "football_weak_stack_reddit_ratio_home",
        "full_weak_stack_reddit_ratio_home",
    ),
    "weak_stack_reddit_spike_away": (
        "football_weak_stack_reddit_spike_away",
        "full_weak_stack_reddit_spike_away",
    ),
    "weak_stack_team_style_pace": (
        "football_weak_stack_team_style_pace",
        "full_weak_stack_team_style_pace",
    ),
    "weak_stack_redzone_third_down": (
        "football_weak_stack_redzone_third_down",
        "full_weak_stack_redzone_third_down",
    ),
    # PER-13 Stage 2 candidate profile
    # (docs/per13_durability_stage2_on_production.md): weak_stack with its nine
    # availability-derived injury columns REPLACED by versions rebuilt on a
    # durability-augmented P(plays) -- a replacement, not an addition, so the
    # column count matches production exactly. Same table-pinning caveat as
    # every profile above: fit only on a table carrying the *_durability columns
    # (game_features_weak_stack_durability.parquet). Never used by the active
    # model.
    "weak_stack_durability": (
        "football_weak_stack_durability",
        "full_weak_stack_durability",
    ),
    "weak_stack_source_availability": (
        "football_weak_stack_source_availability",
        "full_weak_stack_source_availability",
    ),
    # LEAD-21/22/40 (docs/schedule_flag_battery.md): three pure-schedule
    # flags, each PRODUCTION weak_stack plus exactly one new column computed
    # in nfl_ats.schedule_flag_features from the newest schedules.parquet
    # snapshot only. Same table-pinning caveat as every profile above -- fit
    # only on a table carrying that column. Never used by the active model.
    "weak_stack_post_ot": ("football_weak_stack_post_ot", "full_weak_stack_post_ot"),
    "weak_stack_mnf_road": ("football_weak_stack_mnf_road", "full_weak_stack_mnf_road"),
    "weak_stack_home_thursday": (
        "football_weak_stack_home_thursday",
        "full_weak_stack_home_thursday",
    ),
    # Phase 12 market microstructure leads (docs/market_lead_battery.md,
    # LEAD-05/LEAD-03): each PRODUCTION weak_stack plus exactly one new
    # column built entirely from the local point-in-time odds archive
    # (nfl_ats.market_lead_features). Same table-pinning caveat as every
    # profile above -- fit only on a table carrying that column
    # (game_features_weak_stack_opener_softness.parquet /
    # game_features_weak_stack_ml_divergence.parquet). Never used by the
    # active model, never mixed with each other.
    "weak_stack_opener_softness": (
        "football_weak_stack_opener_softness",
        "full_weak_stack_opener_softness",
    ),
    "weak_stack_ml_divergence": (
        "football_weak_stack_ml_divergence",
        "full_weak_stack_ml_divergence",
    ),
    # Wave 2 venue/market-context leads (docs/schedule_flag_battery.md "Wave
    # 2", LEAD-39/LEAD-41/LEAD-42/LEAD-35): each PRODUCTION weak_stack plus
    # exactly one new column computed in nfl_ats.schedule_flag_features, fit
    # directly on the base weak_stack table (no separate candidate-specific
    # parquet -- every input is either a schedule fact or the Tuesday-opener
    # market consensus, both computed at runtime). Never used by the active
    # model, never mixed with each other or with the Wave 1 trio above.
    "weak_stack_new_stadium": ("football_weak_stack_new_stadium", "full_weak_stack_new_stadium"),
    "weak_stack_dome_shootout": (
        "football_weak_stack_dome_shootout",
        "full_weak_stack_dome_shootout",
    ),
    "weak_stack_low_total_div_dog": (
        "football_weak_stack_low_total_div_dog",
        "full_weak_stack_low_total_div_dog",
    ),
    "weak_stack_sept_heat": ("football_weak_stack_sept_heat", "full_weak_stack_sept_heat"),
    # Wave 3 public-claim leads on production (docs/schedule_flag_battery.md
    # "Wave 3", LEAD-57 leads): each PRODUCTION weak_stack plus exactly one
    # new column computed in nfl_ats.schedule_flag_features, fit directly on
    # the base weak_stack table. Never used by the active model, never mixed
    # with each other or with Wave 1/2.
    "weak_stack_road_fav_fade": (
        "football_weak_stack_road_fav_fade",
        "full_weak_stack_road_fav_fade",
    ),
    "weak_stack_division_dog": (
        "football_weak_stack_division_dog",
        "full_weak_stack_division_dog",
    ),
    "weak_stack_week1_dog": ("football_weak_stack_week1_dog", "full_weak_stack_week1_dog"),
    "weak_stack_ats_streak_regress": (
        "football_weak_stack_ats_streak_regress",
        "full_weak_stack_ats_streak_regress",
    ),
    # Wave 4 (docs/schedule_flag_battery.md "Wave 4"), LEAD-26/27/30.
    "weak_stack_opening_drive_epa": (
        "football_weak_stack_opening_drive_epa",
        "full_weak_stack_opening_drive_epa",
    ),
    "weak_stack_q3_point_diff": (
        "football_weak_stack_q3_point_diff",
        "full_weak_stack_q3_point_diff",
    ),
    "weak_stack_fourth_down_interaction": (
        "football_weak_stack_fourth_down_interaction",
        "full_weak_stack_fourth_down_interaction",
    ),
    # Wave 5 (docs/schedule_flag_battery.md "Wave 5"), LEAD-20/LEAD-25.
    "weak_stack_rookie_qb_debut_fade": (
        "football_weak_stack_rookie_qb_debut_fade",
        "full_weak_stack_rookie_qb_debut_fade",
    ),
    "weak_stack_qb_revenge": (
        "football_weak_stack_qb_revenge",
        "full_weak_stack_qb_revenge",
    ),
}

# Production's frozen imputer remains SimpleImputer(add_indicator=True).  This
# sole candidate replaces only the seven source-era indicators; all unrelated
# columns retain the production treatment.
_PROFILE_SUPPRESSED_MISSING_INDICATORS: dict[MarginFeatureProfile, tuple[str, ...]] = {
    "weak_stack_source_availability": SOURCE_ERA_ROSTER_CONTINUITY_COLUMNS,
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


# ---------------------------------------------------------------------------
# Group-wise (block-wise) ridge penalties
# ---------------------------------------------------------------------------
#
# A single global ``alpha`` assumes every feature block deserves the same
# penalty. That is wrong whenever blocks differ in signal-to-noise: the market
# columns are a near-sufficient statistic and want a light penalty, while thin
# player/availability columns are mostly noise and want a heavy one.
#
# Generalized ridge minimises ``||y - X b||^2 + sum_j lambda_j b_j^2``. With
# ``lambda_j = alpha * m_j`` it is implemented EXACTLY by scaling column ``j``
# by ``1 / sqrt(m_j)`` and running an ordinary ``Ridge(alpha=alpha)``: writing
# ``S = diag(1/sqrt(m))`` and ``g`` for the coefficients on the scaled design,
# ``b = S g`` and ``alpha * ||g||^2 = alpha * sum_j m_j b_j^2``. Algebraically,
# ``S (S X'X S + alpha I)^-1 S = (X'X + alpha diag(m))^-1``, so no approximation
# is involved. ``tests/test_margin_groupwise.py`` pins that identity.
#
# Why this is not the MOD-06 no-op. The forced pick is ``sign(predicted
# residual)`` and MOD-06 closed every method whose whole effect is to multiply
# the prediction by a positive scalar. Group-wise penalties are not such a
# method: they change the DIRECTION of the coefficient vector, not just its
# length. In an orthogonal standardised design ``b_j = d_j b_j^OLS / (d_j +
# lambda_j)``, so proportionality between two penalty vectors would require
# ``d_j + lambda_j`` to be a common multiple of ``d_j + lambda_j'`` for every
# ``j`` at once -- impossible once the ``lambda_j`` differ across blocks and the
# ``d_j`` are not all equal. The prediction therefore becomes a different linear
# functional of the features, and its sign can flip.

_MISSING_INDICATOR_PREFIX = "missingindicator_"

_FEATURE_GROUPS: dict[str, str] = {}
for _family, _columns in FEATURE_FAMILIES.items():
    for _column in _columns:
        _existing = _FEATURE_GROUPS.get(_column)
        if _existing is not None and _existing != _family:
            raise RuntimeError(
                f"Feature {_column!r} is claimed by both {_existing!r} and {_family!r}"
            )
        _FEATURE_GROUPS[_column] = _family


def resolve_feature_groups(feature_columns: Sequence[str]) -> tuple[str, ...]:
    """Label every feature column with the ``FEATURE_FAMILIES`` block it belongs to.

    The families are the project's own declared blocks, so this introduces no
    new taxonomy. Raises rather than guessing when a column is unclaimed: a
    silent fallback group would hide a typo behind a plausible penalty.
    """

    unknown = sorted({column for column in feature_columns if column not in _FEATURE_GROUPS})
    if unknown:
        raise ValueError(f"No declared feature family covers: {', '.join(unknown)}")
    return tuple(_FEATURE_GROUPS[column] for column in feature_columns)


def margin_feature_groups(
    target: MarginTarget, feature_profile: MarginFeatureProfile = "base"
) -> tuple[str, ...]:
    """Block labels aligned with ``margin_feature_columns(target, profile)``."""

    return resolve_feature_groups(margin_feature_columns(target, feature_profile))


def column_penalty_multipliers(
    feature_columns: Sequence[str],
    groups: Sequence[str],
    block_multipliers: Mapping[str, float],
    *,
    normalize: bool = True,
) -> dict[str, float]:
    """Expand per-block penalty multipliers to a per-column mapping.

    ``normalize`` divides through by the count-weighted geometric mean, which
    holds the AVERAGE penalty fixed at ``ridge_alpha`` and leaves only the
    relative allocation across blocks free. That separation matters: the global
    penalty level is a different axis, already swept and closed by MOD-06, so a
    group-wise screen that quietly moved it too could not attribute its result.
    """

    if len(feature_columns) != len(groups):
        raise ValueError("feature_columns and groups must have the same length")
    if not feature_columns:
        raise ValueError("At least one feature column is required")
    known = set(groups)
    unknown = sorted(set(block_multipliers).difference(known))
    if unknown:
        raise ValueError(f"Unknown penalty blocks: {', '.join(unknown)}")
    values = np.array([float(block_multipliers.get(group, 1.0)) for group in groups], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Penalty multipliers must be finite and positive")
    if normalize:
        values = values / float(np.exp(np.mean(np.log(values))))
    return {
        str(column): float(value) for column, value in zip(feature_columns, values, strict=True)
    }


class SelectiveMissingnessImputer(TransformerMixin, BaseEstimator):
    """Median-impute every value while suppressing indicators for named sources.

    ``SimpleImputer`` can only add indicators for all columns or none.  MOD-13
    needs the production behavior everywhere except its seven source-era
    continuity columns, whose shared availability is represented by one input
    feature instead.  This class is intentionally candidate-only; its default
    is never reached by the active profile.
    """

    def __init__(self, suppressed_indicator_columns: tuple[str, ...] = ()) -> None:
        self.suppressed_indicator_columns = suppressed_indicator_columns

    def fit(self, X: Any, y: Any = None) -> SelectiveMissingnessImputer:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SelectiveMissingnessImputer requires named pandas columns")
        self.feature_names_in_ = np.asarray([str(column) for column in X.columns], dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        values = X.to_numpy(dtype=float, copy=True)
        missing = ~np.isfinite(values)
        medians = np.nanmedian(values, axis=0)
        # A source that is wholly absent in a training split has no value to
        # impute.  Zero is the same neutral fill SimpleImputer effectively
        # supplies after dropping an empty feature, while retaining the fixed
        # feature contract required for chronology.
        medians = np.where(np.isfinite(medians), medians, 0.0)
        self.statistics_ = medians
        suppressed = set(self.suppressed_indicator_columns)
        self.indicator_features_ = np.asarray(
            [
                index
                for index, (name, has_missing) in enumerate(
                    zip(self.feature_names_in_, missing.any(axis=0), strict=True)
                )
                if has_missing and name not in suppressed
            ],
            dtype=int,
        )
        return self

    def transform(self, X: Any) -> npt.NDArray[np.float64]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SelectiveMissingnessImputer requires named pandas columns")
        names = np.asarray([str(column) for column in X.columns], dtype=object)
        if not np.array_equal(names, self.feature_names_in_):
            raise ValueError("SelectiveMissingnessImputer received unexpected column names")
        values = X.to_numpy(dtype=float, copy=True)
        missing = ~np.isfinite(values)
        values[missing] = np.broadcast_to(self.statistics_, values.shape)[missing]
        if not len(self.indicator_features_):
            return values
        indicators = missing[:, self.indicator_features_].astype(float)
        return np.hstack((values, indicators))

    def get_feature_names_out(self, input_features: Any = None) -> npt.NDArray[np.object_]:
        indicator_names = [
            f"{_MISSING_INDICATOR_PREFIX}{self.feature_names_in_[index]}"
            for index in self.indicator_features_
        ]
        return np.asarray([*self.feature_names_in_, *indicator_names], dtype=object)


class GroupPenaltyScaler(TransformerMixin, BaseEstimator):
    """Scale column ``j`` by ``1 / sqrt(m_j)`` so a plain ridge penalises it by ``alpha * m_j``.

    Sits between the ``StandardScaler`` and the ``Ridge`` -- it must come after
    standardisation, because standardising afterwards would divide the scaling
    straight back out.

    Missing-value indicator columns added by ``SimpleImputer(add_indicator=True)``
    arrive named ``missingindicator_<source>`` and inherit their source column's
    multiplier, so a block's missingness flags are penalised with the block.
    """

    def __init__(self, column_multipliers: Mapping[str, float] | None = None) -> None:
        self.column_multipliers = column_multipliers

    def _multiplier(self, name: str) -> float:
        multipliers = self.column_multipliers or {}
        if name in multipliers:
            return float(multipliers[name])
        if name.startswith(_MISSING_INDICATOR_PREFIX):
            source = name[len(_MISSING_INDICATOR_PREFIX) :]
            if source in multipliers:
                return float(multipliers[source])
        raise ValueError(f"No penalty multiplier declared for transformed column {name!r}")

    def fit(self, X: Any, y: Any = None) -> GroupPenaltyScaler:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "GroupPenaltyScaler needs named columns; set the pipeline output to pandas"
            )
        names = [str(column) for column in X.columns]
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        multipliers = np.array([self._multiplier(name) for name in names], dtype=float)
        if not np.all(np.isfinite(multipliers)) or np.any(multipliers <= 0.0):
            raise ValueError("Penalty multipliers must be finite and positive")
        self.penalty_multipliers_ = multipliers
        self.scale_ = 1.0 / np.sqrt(multipliers)
        return self

    def transform(self, X: Any) -> Any:
        scale = getattr(self, "scale_", None)
        if scale is None:
            raise RuntimeError("GroupPenaltyScaler is not fitted")
        if isinstance(X, pd.DataFrame):
            if [str(column) for column in X.columns] != list(self.feature_names_in_):
                raise ValueError("GroupPenaltyScaler received unexpected column names")
            return X.mul(pd.Series(scale, index=X.columns), axis=1)
        array = np.asarray(X, dtype=float)
        if array.shape[1] != len(scale):
            raise ValueError("GroupPenaltyScaler received an unexpected column count")
        return array * scale

    def get_feature_names_out(self, input_features: Any = None) -> npt.NDArray[np.object_]:
        return np.asarray(self.feature_names_in_, dtype=object)


def make_margin_estimator(
    model_name: str,
    random_state: int = 42,
    *,
    ridge_alpha: float = 10.0,
    column_penalties: Mapping[str, float] | None = None,
    suppressed_indicator_columns: tuple[str, ...] = (),
) -> BaseEstimator:
    if not np.isfinite(ridge_alpha) or ridge_alpha <= 0.0:
        raise ValueError("ridge_alpha must be finite and positive")
    if column_penalties is not None and model_name != "ridge":
        raise ValueError("Group-wise penalties apply only to the ridge margin model")
    if model_name == "ridge":
        imputer: BaseEstimator
        if suppressed_indicator_columns:
            imputer = SelectiveMissingnessImputer(suppressed_indicator_columns)
        else:
            imputer = SimpleImputer(strategy="median", add_indicator=True)
        if column_penalties is None:
            # Frozen path, deliberately untouched: same steps, same objects, no
            # output container change. Group penalties are strictly opt-in.
            return Pipeline(
                steps=[
                    ("imputer", imputer),
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge(alpha=ridge_alpha)),
                ]
            )
        pipeline = Pipeline(
            steps=[
                ("imputer", imputer),
                ("scaler", StandardScaler()),
                ("group_penalty", GroupPenaltyScaler(dict(column_penalties))),
                ("regressor", Ridge(alpha=ridge_alpha)),
            ]
        )
        # Names must survive to the group step so indicator columns can be
        # matched back to the block they flag.
        pipeline.set_output(transform="pandas")
        return pipeline
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
    #: Per-column ridge penalty multipliers, or ``None`` for a single global
    #: penalty. Defaulted so every existing construction is unchanged.
    column_penalties: Mapping[str, float] | None = field(default=None)

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

    def predict(
        self, frame: pd.DataFrame, *, probability_method: ResidualSmoothingMethod = "ecdf"
    ) -> pd.DataFrame:
        """Predict every game in ``frame``.

        ``probability_method`` controls ONLY ``home_cover_probability`` (the
        two-way forced-pick threshold) -- ``home_win_probability`` and the
        push/three-way split stay on the raw ECDF unconditionally, per
        ``docs/smooth_cdf_mapping.md``'s declared scope. The default,
        ``"ecdf"``, reproduces every historical caller's output bit-for-bit
        (this method is called from dozens of research/backtest call sites
        that must never silently change). ``nfl_ats.outcomes.score_outcome_week``
        -- the sole production weekly-forecast entry point -- passes
        ``"gaussian"`` explicitly (MOD-08 promotion, 2026-08-19,
        `probability_positive` 0.5536 at the opener grade, see
        ``docs/smooth_cdf_mapping.md``); every other caller is unaffected.
        """

        spread = self._spread(frame)
        predicted_margin, predicted_residual = self._predicted_margin(frame, spread)
        if self.target == "market":
            market_cover_probability = [
                no_vig_probabilities(row.get("home_spread_odds"), row.get("away_spread_odds"))[0]
                for _, row in frame.iterrows()
            ]
        else:
            market_cover_probability = []

        # Computed once, vectorized, rather than refitting a
        # ``ResidualSmoother`` on the SAME residual sample inside the per-row
        # loop below. ``None`` when the caller wants the default ECDF path,
        # so that path's per-row ``_smoothed_probability`` call (identical to
        # this method's behavior before ``probability_method`` existed) stays
        # bit-for-bit unchanged.
        mapped_cover_probability: npt.NDArray[np.float64] | None = None
        if self.target != "market" and probability_method != "ecdf":
            mapped_cover_probability = smoothed_home_cover_probability(
                self.residuals, predicted_margin, spread, method=probability_method
            )

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
            if self.target == "market":
                probabilities_cover.append(market_cover_probability[row_index])
            elif mapped_cover_probability is not None:
                probabilities_cover.append(float(mapped_cover_probability[row_index]))
            else:
                probabilities_cover.append(_smoothed_probability(distribution, float(line)))
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
    column_penalties: Mapping[str, float] | None = None,
) -> MarginModel:
    feature_columns = margin_feature_columns(target, feature_profile)
    suppressed_indicator_columns = _PROFILE_SUPPRESSED_MISSING_INDICATORS.get(feature_profile, ())
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
    temporary = make_margin_estimator(
        model_name,
        random_state,
        ridge_alpha=ridge_alpha,
        column_penalties=column_penalties,
        suppressed_indicator_columns=suppressed_indicator_columns,
    )
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

    estimator = make_margin_estimator(
        model_name,
        random_state,
        ridge_alpha=ridge_alpha,
        column_penalties=column_penalties,
        suppressed_indicator_columns=suppressed_indicator_columns,
    )
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
        column_penalties=None if column_penalties is None else dict(column_penalties),
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
    metadata: dict[str, Any] = {
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
    # Emitted only when group penalties are actually in use, so a frozen
    # single-penalty run keeps a byte-identical metadata payload.
    if model.column_penalties is not None:
        metadata["column_penalties"] = {
            str(column): float(value) for column, value in model.column_penalties.items()
        }
    return metadata
