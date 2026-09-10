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

DECOMPOSITION_TARGETS: tuple[str, ...] = ("margin", "spread", "residual")

DEFAULT_FEATURE_PROFILE: MarginFeatureProfile = "player"
DEFAULT_RIDGE_ALPHA = 10.0
DEFAULT_START_SEASON = 2018
DEFAULT_END_SEASON = 2025

DEFAULT_NOISE_SHARE_THRESHOLD = 0.03
DEFAULT_OVERPRICED_RATIO_THRESHOLD = 1.5

RECONCILIATION_ATOL = 1e-4

ATTRIBUTION_ATOL = 1e-6

INTERCEPT_FAMILY = "intercept"

OPENER_MIN_GAMES_DEFAULT = 50

DEFAULT_MATERIALITY_POINTS = 0.25
DEFAULT_NEGLIGIBLE_GAP_POINTS = 0.5
DEFAULT_MAX_DRIVERS = 3
DEFAULT_MAX_OFFSETS = 1


def build_family_map(
    feature_columns: Sequence[str],
    families: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str]:

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

    base = name[len("missing:") :] if name.startswith("missing:") else name
    return family_map[base]


def decomposition_feature_columns(
    feature_profile: MarginFeatureProfile = DEFAULT_FEATURE_PROFILE,
) -> tuple[str, ...]:

    return margin_feature_columns("margin", feature_profile)


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


@dataclass(frozen=True)
class WalkForwardDecomposition:
    coefficients: pd.DataFrame
    predictions: pd.DataFrame
    reconciliation: pd.DataFrame
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


def family_weights_table(coefficients: pd.DataFrame) -> pd.DataFrame:

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


def r_squared_table(predictions: pd.DataFrame) -> pd.DataFrame:

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

    if reconciliation.empty:
        raise ValueError("Reconciliation table is empty")
    return {
        "games": len(reconciliation),
        "mean_abs_error": float(reconciliation["error"].mean()),
        "max_abs_error": float(reconciliation["error"].max()),
        "p99_abs_error": float(reconciliation["error"].quantile(0.99)),
    }


@dataclass(frozen=True)
class OpenerVariantResult:
    available: bool
    reason: str | None
    games: int
    coefficients: pd.DataFrame | None
    family_weights: pd.DataFrame | None
    r_squared: pd.DataFrame | None


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


WEEKLY_CONTEXT_FAMILY = "weekly_context"

FAMILY_PHRASES: dict[str, str] = {
    "per07_coord_change_on_production": "changes in the coaching staff since last season",
    "home_dog_location_points": "how many points the home team is getting as the underdog",
    "home_dog_location_hinge": "how far past a touchdown the home team is getting as the underdog",
    "home_side_location_hinge": "how far past a touchdown the point spread is, on either side",
    "spread_regime": "Spread size and distance to common winning margins",
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
    "quarterback_depth": "named starter and backup quarterback state",
    "player_qb": "quarterback status and recent play",
    "player_injuries": "expected player availability",
    "player_continuity": "lineup continuity",
    "roster_returning_snaps": "prior-season snap share retained on the current roster",
    "roster_continuity_source_availability": (
        "whether source-era roster continuity data is available"
    ),
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
    "travel_geometry": "decision-time travel distance and body-clock direction",
    "rest_context": "pregame rest, bye, short-week, and road-streak context",
    "forecast_weather": "the forecast temperature, wind and rain at kickoff",
    "observed_weather": "the weather that actually happened (a control, never played)",
    "graph_team_stat_off_sack_rate": "opponent-adjusted sack rate (schedule-graph transformed)",
    "graph_team_stat_def_yards_per_play": (
        "opponent-adjusted yards allowed per play (schedule-graph transformed)"
    ),
    "graph_team_stat_off_rush_epa_per_play": (
        "opponent-adjusted rushing efficiency (schedule-graph transformed)"
    ),
    "fluview_home_elevated_on_production": (
        "elevated flu-like illness in the home team's own market"
    ),
    "fluview_away_elevated_on_production": (
        "elevated flu-like illness in the away team's own market"
    ),
    "fluview_away_ili_asof_on_production": (
        "how much flu was circulating in the visiting team's market this week"
    ),
    "apm_unit_on_production": (
        "each side's offensive and defensive unit strength from play-level ratings"
    ),
    "illness_away_active_ge1_on_production": (
        "at least one away player listed with an illness on the injury report"
    ),
    "illness_home_ge2_on_production": (
        "two or more home players listed with an illness on the injury report"
    ),
    "reddit_home_ratio_elevated_on_production": (
        "an unusually argumentative week on the home team's fan forum"
    ),
    "reddit_away_spike_on_production": ("a spike in chatter on the away team's fan forum"),
    "team_style_pace_mismatch_on_production": ("a big gap between the two teams' offensive tempo"),
    "post_ot_fatigue_on_production": "a team coming off an overtime game last week",
    "mnf_road_short_week_on_production": (
        "a team that played on the road on Monday night and plays again Sunday"
    ),
    "home_thursday_on_production": "the home team in a Thursday-night game",
    "new_stadium_home_on_production": (
        "the home team in only its venue's first two seasons of NFL use"
    ),
    "dome_shootout_favorite_on_production": (
        "the favourite in a dome/closed-roof, high-total, near-even-spread game"
    ),
    "low_total_div_home_dog_on_production": ("the home underdog in a low-scoring divisional game"),
    "sept_heat_home_on_production": (
        "a heat-acclimated home team hosting a cold-climate visitor in September"
    ),
    "road_fav_big_fade_on_production": (
        "a big road favourite (or, mirrored, a big home favourite) at the opener"
    ),
    "division_dog_on_production": "the underdog in a divisional game",
    "week1_dog_on_production": "the underdog in a Week 1 game",
    "ats_streak_regress_on_production": (
        "a team on a three-or-more-game losing streak against the spread"
    ),
    "opening_drive_script_on_production": (
        "a team with a strong track record on its own opening drive"
    ),
    "q3_adjustment_on_production": ("a team with a strong history of third-quarter adjustments"),
    "fourth_down_aggression_interaction_on_production": (
        "an aggressive fourth-down team that is also the underdog at the opener"
    ),
    "opener_softness_fade_on_production": (
        "a side implied only by the least accurate book's opening line"
    ),
    "ml_spread_divergence_on_production": (
        "the moneyline and the point spread disagreeing about who is favoured"
    ),
    "redzone_third_down_over_fade_on_production": (
        "a team coming off an unsustainably good third-down season"
    ),
    "player_injuries_durability": (
        "expected player availability, using each player's own injury history"
    ),
    "player_values_durability": (
        "estimated value lost to injuries, using each player's own injury history"
    ),
    "rookie_qb_debut_fade_on_production": (
        "a quarterback making his first-ever career start as a rookie"
    ),
    "qb_revenge_on_production": ("a quarterback facing the franchise that drafted him"),
    "holdout_slow_start_on_production": (
        "a team starting a regular who just ended a training-camp holdout"
    ),
    "deadline_integration_drag_on_production": (
        "a team integrating a high-snap player it just acquired at the trade deadline"
    ),
    "suspension_return_rust_on_production": (
        "a team playing a player just back from a long suspension"
    ),
    "crew_second_meeting_favorite_on_production": (
        "the favorite facing a referee crew that already worked one of these teams "
        "earlier this season"
    ),
    "rookie_crew_underdog_on_production": (
        "the underdog officiated by a first- or second-year referee crew"
    ),
    "open_corner_wind_dog_on_production": (
        "the underdog at an open-corner stadium in a high-wind game"
    ),
    "rain_on_grass_dog_on_production": (
        "the underdog on a grass field with a high forecast chance of rain"
    ),
    "ir_return_bump_on_production": ("a team getting a starter back from injured reserve"),
    "specialist_absence_fade_on_production": ("a team missing its long snapper or punter"),
    "rookie_wall_dependence_on_production": (
        "a team that leans heavily on high-draft-pick rookies late in the season"
    ),
    "kicker_change_underdog_on_production": (
        "the underdog in a game where a team just changed its placekicker"
    ),
    "backup_tenure_gap_on_production": (
        "a team starting a backup quarterback who has been with it two or more "
        "seasons, versus a team starting a brand-new backup"
    ),
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
    points: float


@dataclass(frozen=True)
class GameExplanation:
    game_id: str
    pick_side: str
    pick_team: str
    other_team: str
    gap_points: float
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
