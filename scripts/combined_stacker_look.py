"""Combined weak-signal stacker: the ONE predeclared confirmation look.

Executes docs/combined_stacker_predeclaration.md (frozen 2026-08-21) exactly:

* baseline arm -- the production recipe (``feature_profile=weak_stack``,
  ``target=market_residual``, ridge alpha 10, no calibration, Gaussian
  probabilities, expanding walk-forward, ``min_train_games=500``, training
  strictly prior) on ``data/processed/game_features_weak_stack.parquet``;
* candidate arm -- identical in every respect except four appended design-matrix
  columns (90 -> 94): the narrowed fixed-prior injury-value-lost top-tertile
  pair (Saturday decision cutoff, tercile boundaries from strictly-prior
  completed games only), the kickoff-nearest temp-gap-cold-visitor cell, the
  kickoff-nearest warm-team-cold-late cell, and the spread-gap-zone binary.

Both arms are graded by ``clv.opener_pick_evaluation`` restricted to the
rotation registry's assigned window via ``rotation.confirmation_split``; forced
picks use the production probability rule (``home_cover_probability >= 0.5``).
The reported evidence is the paired accuracy delta with week-blocked bootstrap,
season-blocked secondary, and Brier/log-loss direction-only secondaries.

Leakage conventions, disclosed up front: the climatological baseline for the
temp-gap column is computed from every STRICTLY EARLIER outdoor home game the
away team has played, across any season -- the documented pregame-safe
adaptation used by nfl_ats.forecast_cold_visitor_tilt_overlay (the registered
research construction used a within-season all-games aggregate, which is not
pregame-safe); the injury tercile boundary for each game uses only completed
games strictly before that game's kickoff; any game with a missing value-lost,
forecast-temp, or roof input fails closed to "not flagged".

This script produces numbers and artifacts only. It writes no registry JSON
itself: ``rotation record`` and ``nfl-ats weak-signals record`` are separate,
deliberate steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats import constants
from nfl_ats import margin as margin_module
from nfl_ats.clv import opener_pick_evaluation, week_blocked_bootstrap
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact
from nfl_ats.rotation import confirmation_split, load_registry

REPO = Path(__file__).resolve().parents[1]

BASELINE_PROFILE = "weak_stack"
CANDIDATE_PROFILE = "combined_stacker"
REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0
MIN_TRAIN_GAMES = 500

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260817
CONFIRM_AT = 0.90
CLOSE_NEGATIVE_AT = 0.10
MIN_PAIRED_GAMES = 400

IVL_HOME_COMPONENTS = (
    "home_injury_skill_epa_value_lost",
    "home_injury_defense_disruption_value_lost",
)
IVL_AWAY_COMPONENTS = (
    "away_injury_skill_epa_value_lost",
    "away_injury_defense_disruption_value_lost",
)
IVL_TERCILE_QUANTILE = 2.0 / 3.0

FORECAST_OUTDOOR_ROOFS = frozenset({"outdoors", "open"})
TEMP_GAP_THRESHOLD_F = 25.0
WARM_METRO_TEAM_CODES = frozenset(
    {"MIA", "TB", "JAX", "ARI", "SF", "LA", "LAC", "HOU", "DAL", "NO", "LV"}
)
WARM_TEAM_TEMP_THRESHOLD_F = 35.0
WARM_TEAM_MIN_WEEK = 13
SPREAD_GAP_LOW = 7.0
SPREAD_GAP_HIGH = 10.0

CANDIDATE_COLUMNS: tuple[str, ...] = (
    "ivl_home_top_tertile",
    "ivl_away_top_tertile",
    "kn_temp_gap_cold_visitor_pre2020",
    "warm_team_cold_late_pre2020",
    "spread_gap_zone",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def register_candidate_profile() -> None:
    """Register the 94-column candidate profile in-process.

    The production machinery selects design-matrix columns by profile name, so
    the candidate registers a feature set equal to the production weak_stack
    set plus the four frozen columns, without touching any tracked src file:
    the mutation lives and dies with this process.
    """

    football_set = "football_combined_stacker"
    full_set = "full_combined_stacker"
    if football_set in constants.FEATURE_SETS or full_set in constants.FEATURE_SETS:
        raise RuntimeError("combined_stacker feature sets are already registered")
    if CANDIDATE_PROFILE in margin_module.MARGIN_FEATURE_PROFILES:
        raise RuntimeError("combined_stacker margin profile is already registered")
    constants.FEATURE_SETS[football_set] = (
        tuple(constants.FEATURE_SETS["football_weak_stack"]) + CANDIDATE_COLUMNS
    )
    constants.FEATURE_SETS[full_set] = (
        tuple(constants.FEATURE_SETS["full_weak_stack"]) + CANDIDATE_COLUMNS
    )
    margin_module.MARGIN_FEATURE_PROFILES = (
        *margin_module.MARGIN_FEATURE_PROFILES,
        CANDIDATE_PROFILE,  # type: ignore[arg-type]
    )
    margin_module._MARGIN_PROFILE_FEATURE_SETS[CANDIDATE_PROFILE] = (  # type: ignore[index]
        football_set,
        full_set,
    )


def _ivl_side_value(frame: pd.DataFrame, components: tuple[str, ...]) -> pd.Series:
    total = pd.to_numeric(frame[f"{components[0]}_fx"], errors="coerce").copy()
    for extra in components[1:]:
        total = total + pd.to_numeric(frame[f"{extra}_fx"], errors="coerce")
    return total


def _strictly_prior_boundary(values: pd.DataFrame, quantile: float) -> Callable[[Any], float]:
    """Return a lookup giving the quantile of ``values`` strictly before a date.

    ``values`` is a frame with columns ``gameday`` and ``value`` (the pooled
    per-side magnitudes of every completed regular-season game). The returned
    callable maps a gameday to the quantile computed ONLY over entries whose
    gameday is strictly earlier, so a game's own value can never move its own
    boundary.
    """

    ordered = values.dropna(subset=["value"]).sort_values("gameday", kind="mergesort")
    days = ordered["gameday"].to_numpy()
    mags = ordered["value"].to_numpy(dtype=float)
    cache: dict[Any, float] = {}

    def boundary(day: Any) -> float:
        if day not in cache:
            position = int(np.searchsorted(days, day, side="left"))
            cache[day] = (
                float(np.quantile(mags[:position], quantile)) if position > 0 else float("nan")
            )
        return cache[day]

    return boundary


def attach_injury_tercile_columns(base: pd.DataFrame, player_value: pd.DataFrame) -> pd.DataFrame:
    """Append ``ivl_home_top_tertile`` / ``ivl_away_top_tertile``.

    Side magnitude = skill-EPA value lost + defense-disruption value lost from
    the fixed-prior-severity player-value table (Saturday decision cutoff),
    joined by game_id; each side is flagged iff its magnitude sits at or above
    the 2/3 quantile of the pooled home+away magnitude distribution over every
    completed regular-season game STRICTLY BEFORE this game's gameday. Missing
    inputs fail closed to not-flagged.
    """

    join_columns = ["game_id", *[f"{c}_fx" for c in (*IVL_HOME_COMPONENTS, *IVL_AWAY_COMPONENTS)]]
    renamed = player_value[["game_id", *IVL_HOME_COMPONENTS, *IVL_AWAY_COMPONENTS]].rename(
        columns={c: f"{c}_fx" for c in (*IVL_HOME_COMPONENTS, *IVL_AWAY_COMPONENTS)}
    )
    frame = base.merge(renamed[join_columns], on="game_id", how="left", validate="one_to_one")
    home_value = _ivl_side_value(frame, IVL_HOME_COMPONENTS)
    away_value = _ivl_side_value(frame, IVL_AWAY_COMPONENTS)

    gameday = pd.to_datetime(frame["gameday"], errors="raise")
    is_reg_completed = frame["result"].notna() & frame["game_type"].eq("REG")
    completed = frame.loc[is_reg_completed]
    pooled = pd.concat(
        [
            pd.DataFrame(
                {
                    "gameday": pd.to_datetime(completed["gameday"], errors="raise"),
                    "value": home_value.loc[completed.index],
                }
            ),
            pd.DataFrame(
                {
                    "gameday": pd.to_datetime(completed["gameday"], errors="raise"),
                    "value": away_value.loc[completed.index],
                }
            ),
        ],
        ignore_index=True,
    )
    boundary = _strictly_prior_boundary(pooled, IVL_TERCILE_QUANTILE)

    home_flag = home_value.notna() & gameday.map(boundary).le(home_value)
    away_flag = away_value.notna() & gameday.map(boundary).le(away_value)
    frame["ivl_home_top_tertile"] = home_flag.fillna(False).astype(float)
    frame["ivl_away_top_tertile"] = away_flag.fillna(False).astype(float)
    frame["csl_home_ivl_value"] = home_value
    frame["csl_away_ivl_value"] = away_value
    return frame


def attach_weather_columns(
    base: pd.DataFrame, forecasts: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    """Append the two kickoff-nearest cold-visitor weather columns.

    ``kn_temp_gap_cold_visitor_pre2020``: outdoor AND (away team's mean ACTUAL
    temp over its strictly-earlier outdoor HOME games, any season, minus this
    game's kickoff-nearest forecast temp) >= 25F. ``warm_team_cold_late_
    pre2020``: away team in the static warm-winter-metro list AND outdoor AND
    forecast temp <= 35F AND week >= 13. Missing roof/forecast/climate inputs
    fail closed to not-flagged.
    """

    frame = base.merge(
        schedules[["game_id", "roof"]], on="game_id", how="left", validate="one_to_one"
    )
    frame = frame.merge(
        forecasts[["game_id", "forecast_temp_f"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    frame["forecast_temp_f"] = pd.to_numeric(frame["forecast_temp_f"], errors="coerce")
    outdoor = frame["roof"].isin(FORECAST_OUTDOOR_ROOFS)

    home_games = frame.loc[
        outdoor & frame["temp"].notna(), ["game_id", "gameday", "home_team", "temp"]
    ].copy()
    home_games["gameday"] = pd.to_datetime(home_games["gameday"], errors="raise")
    home_games = home_games.sort_values(["home_team", "gameday", "game_id"], kind="mergesort")
    climates: list[pd.Series] = []
    for _, group in home_games.groupby("home_team", sort=False):
        climates.append(group["temp"].expanding().mean())
    home_games["running_climate"] = (
        pd.concat(climates).sort_index() if climates else pd.Series(dtype=float)
    )

    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    climate_history = home_games.rename(columns={"home_team": "away_team"})[
        ["gameday", "away_team", "running_climate"]
    ].sort_values("gameday", kind="mergesort")
    frame = frame.sort_values("gameday", kind="mergesort").reset_index(drop=True)
    frame = pd.merge_asof(
        frame,
        climate_history,
        on="gameday",
        by="away_team",
        direction="backward",
        allow_exact_matches=False,
    )
    climate = frame["running_climate"].rename("climate_temp")
    outdoor = frame["roof"].isin(FORECAST_OUTDOOR_ROOFS)

    temp_gap = climate - frame["forecast_temp_f"]
    frame["kn_temp_gap_cold_visitor_pre2020"] = (
        outdoor & temp_gap.notna() & temp_gap.ge(TEMP_GAP_THRESHOLD_F)
    ).astype(float)
    frame["warm_team_cold_late_pre2020"] = (
        frame["away_team"].isin(WARM_METRO_TEAM_CODES)
        & outdoor
        & frame["forecast_temp_f"].le(WARM_TEAM_TEMP_THRESHOLD_F)
        & (frame["week"].astype(int) >= WARM_TEAM_MIN_WEEK)
    ).astype(float)
    frame["csl_outdoor"] = outdoor.astype(float)
    frame["csl_climate_temp"] = climate
    frame["csl_forecast_temp_f"] = frame["forecast_temp_f"]
    return frame


def attach_spread_gap_zone(base: pd.DataFrame) -> pd.DataFrame:
    """Append the spread-gap-zone binary: ``7 < |spread_line| <= 10``."""

    spread = pd.to_numeric(base["spread_line"], errors="coerce").abs()
    base["spread_gap_zone"] = (spread.gt(SPREAD_GAP_LOW) & spread.le(SPREAD_GAP_HIGH)).astype(float)
    return base


def build_candidate_table(
    base: pd.DataFrame,
    *,
    player_value: pd.DataFrame,
    forecasts: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    frame = attach_injury_tercile_columns(base, player_value)
    frame = attach_weather_columns(frame, forecasts, schedules)
    frame = attach_spread_gap_zone(frame)
    return frame


def _config(profile: str) -> dict[str, Any]:
    return {
        "feature_profile": profile,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "target": "market_residual",
    }


def arm(
    features: pd.DataFrame,
    *,
    registry: Any,
    family: str,
    profile: str,
    market_root: Path,
    min_train_games: int,
) -> pd.DataFrame:
    training, window = confirmation_split(features, registry, family)
    scoped = pd.concat([training, window], ignore_index=True)
    scored = opener_pick_evaluation(
        market_root,
        scoped,
        active_model_config=_config(profile),
        min_train_games=min_train_games,
    )
    seasons = sorted(window["season"].astype(int).unique())
    scored = scored.loc[scored["season"].isin(seasons)]
    return scored.loc[scored["correct_at_open_probability_rule"].notna()].copy()


def paired_frame(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "game_id",
        "season",
        "week",
        "tue_open_home_spread",
        "correct_at_open_probability_rule",
        "pick_home_at_open_probability_rule",
        "home_cover_probability_at_open",
        "margin_vs_open",
    ]
    left = baseline[keep].rename(
        columns={
            "correct_at_open_probability_rule": "baseline_correct_open",
            "pick_home_at_open_probability_rule": "baseline_pick_home",
            "home_cover_probability_at_open": "baseline_prob_open",
        }
    )
    right = candidate[keep].rename(
        columns={
            "correct_at_open_probability_rule": "candidate_correct_open",
            "pick_home_at_open_probability_rule": "candidate_pick_home",
            "home_cover_probability_at_open": "candidate_prob_open",
        }
    )
    merged = left.merge(right, on="game_id", how="inner", suffixes=("", "_drop"))
    merged = merged.drop(columns=[column for column in merged.columns if column.endswith("_drop")])
    for column in ("baseline_correct_open", "candidate_correct_open"):
        merged[column] = merged[column].astype(float)
    merged["actual_home_cover_open"] = np.where(
        merged["margin_vs_open"] > 0.0,
        1.0,
        np.where(merged["margin_vs_open"] < 0.0, 0.0, np.nan),
    )
    return merged.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _accuracy_metric_fn() -> Callable[[pd.DataFrame], dict[str, float]]:
    def metric_fn(frame: pd.DataFrame) -> dict[str, float]:
        return {
            "candidate_minus_baseline": float(
                frame["candidate_correct_open"].mean() - frame["baseline_correct_open"].mean()
            )
        }

    return metric_fn


def _brier_metric_fn() -> Callable[[pd.DataFrame], dict[str, float]]:
    def metric_fn(frame: pd.DataFrame) -> dict[str, float]:
        valid = frame.dropna(subset=["actual_home_cover_open"])
        actual = valid["actual_home_cover_open"].to_numpy(dtype=float)
        candidate = valid["candidate_prob_open"].to_numpy(dtype=float)
        baseline = valid["baseline_prob_open"].to_numpy(dtype=float)
        clip = lambda p: np.clip(p, 1e-15, 1.0 - 1e-15)  # noqa: E731
        return {
            "brier_improvement": float(
                np.mean(np.square(baseline - actual)) - np.mean(np.square(candidate - actual))
            ),
            "log_loss_improvement": float(
                np.mean(
                    -(actual * np.log(clip(baseline)) + (1 - actual) * np.log(1 - clip(baseline)))
                )
                - np.mean(
                    -(actual * np.log(clip(candidate)) + (1 - actual) * np.log(1 - clip(candidate)))
                ),
            ),
        }

    return metric_fn


def _bootstrap(paired: pd.DataFrame, metric_fn: Callable[..., dict[str, float]], block: str):
    return week_blocked_bootstrap(
        paired, metric_fn, block=block, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )


def _verdict(probability_positive: float, lower: float) -> tuple[str, str]:
    if lower < 0.0 and probability_positive <= CLOSE_NEGATIVE_AT:
        return "closed_negative", "refuted_mechanism"
    if probability_positive >= CONFIRM_AT:
        return "confirmed", "unresolved_below_power"
    return "unresolved", "unresolved_below_power"


def run(args: argparse.Namespace) -> dict[str, Any]:
    base = pd.read_parquet(args.features)
    player_value = pd.read_parquet(args.player_value_features)
    forecasts = pd.read_parquet(args.forecast_archive)
    schedules = pd.read_parquet(args.schedules, columns=["game_id", "roof"])

    register_candidate_profile()
    augmented = build_candidate_table(
        base, player_value=player_value, forecasts=forecasts, schedules=schedules
    )

    registry = load_registry(args.registry)
    declared = registry.families[args.family]
    window = declared.assigned_window
    if window is None:
        raise SystemExit(f"Family {args.family!r} holds no assigned window; run rotation assign")

    baseline_reference = arm(
        base,
        registry=registry,
        family=args.family,
        profile=BASELINE_PROFILE,
        market_root=args.market_root,
        min_train_games=args.min_train_games,
    )
    baseline_check = arm(
        augmented,
        registry=registry,
        family=args.family,
        profile=BASELINE_PROFILE,
        market_root=args.market_root,
        min_train_games=args.min_train_games,
    )
    self_check = baseline_reference.merge(
        baseline_check[
            ["game_id", "pick_home_at_open_probability_rule", "correct_at_open_probability_rule"]
        ],
        on="game_id",
        how="inner",
        suffixes=("", "_augmented"),
    )
    if len(self_check) != len(baseline_reference) or len(self_check) != len(baseline_check):
        raise AssertionError("Baseline self-check lost paired games across tables")
    if not (
        self_check["pick_home_at_open_probability_rule"]
        .eq(self_check["pick_home_at_open_probability_rule_augmented"])
        .all()
    ):
        raise AssertionError(
            "Appending the candidate columns changed the baseline arm's picks; "
            "the augmentation is not append-only"
        )

    candidate = arm(
        augmented,
        registry=registry,
        family=args.family,
        profile=CANDIDATE_PROFILE,
        market_root=args.market_root,
        min_train_games=args.min_train_games,
    )
    paired = paired_frame(baseline_reference, candidate)
    if len(paired) < MIN_PAIRED_GAMES:
        raise SystemExit(
            f"Only {len(paired)} paired opener games; predeclaration section 7 "
            f"aborts below {MIN_PAIRED_GAMES}"
        )

    primary = _bootstrap(paired, _accuracy_metric_fn(), "week").iloc[0]
    season_secondary = _bootstrap(paired, _accuracy_metric_fn(), "season").iloc[0]
    brier_week = _bootstrap(paired, _brier_metric_fn(), "week")
    brier_season = _bootstrap(paired, _brier_metric_fn(), "season")

    disagreements = paired.loc[paired["baseline_pick_home"].ne(paired["candidate_pick_home"])]
    probability_positive = float(primary["probability_positive"])
    rotation_verdict, signal_classification = _verdict(
        probability_positive, float(primary["lower"])
    )

    flag_counts = {
        column: int(augmented.loc[augmented["game_id"].isin(set(paired["game_id"])), column].sum())
        for column in CANDIDATE_COLUMNS
    }

    summary: dict[str, Any] = {
        "family": args.family,
        "window_seasons": [int(window.seasons[0]), int(window.seasons[1])],
        "grade": "opener",
        "pick_rule": "production probability rule (home_cover_probability >= 0.5)",
        "baseline_profile": BASELINE_PROFILE,
        "candidate_profile": CANDIDATE_PROFILE,
        "candidate_columns": list(CANDIDATE_COLUMNS),
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "min_train_games": args.min_train_games,
        "paired_games": len(paired),
        "weeks": int(paired.groupby(["season", "week"]).ngroups),
        "seasons": sorted(int(s) for s in paired["season"].unique()),
        "baseline_accuracy_opener_pr": float(paired["baseline_correct_open"].mean()),
        "candidate_accuracy_opener_pr": float(paired["candidate_correct_open"].mean()),
        "delta": float(primary["estimate"]),
        "week_blocked": {
            "estimate": float(primary["estimate"]),
            "lower": float(primary["lower"]),
            "upper": float(primary["upper"]),
            "probability_positive": probability_positive,
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "season_blocked_secondary": {
            "estimate": float(season_secondary["estimate"]),
            "lower": float(season_secondary["lower"]),
            "upper": float(season_secondary["upper"]),
            "probability_positive": float(season_secondary["probability_positive"]),
        },
        "brier_log_loss_direction_secondaries": {
            "brier_improvement_week_blocked": {
                "estimate": float(brier_week.iloc[0]["estimate"]),
                "lower": float(brier_week.iloc[0]["lower"]),
                "upper": float(brier_week.iloc[0]["upper"]),
                "probability_positive": float(brier_week.iloc[0]["probability_positive"]),
            },
            "log_loss_improvement_week_blocked": {
                "estimate": float(brier_week.iloc[1]["estimate"]),
                "lower": float(brier_week.iloc[1]["lower"]),
                "upper": float(brier_week.iloc[1]["upper"]),
                "probability_positive": float(brier_week.iloc[1]["probability_positive"]),
            },
            "brier_improvement_season_blocked": {
                "estimate": float(brier_season.iloc[0]["estimate"]),
                "lower": float(brier_season.iloc[0]["lower"]),
                "upper": float(brier_season.iloc[0]["upper"]),
                "probability_positive": float(brier_season.iloc[0]["probability_positive"]),
            },
            "log_loss_improvement_season_blocked": {
                "estimate": float(brier_season.iloc[1]["estimate"]),
                "lower": float(brier_season.iloc[1]["lower"]),
                "upper": float(brier_season.iloc[1]["upper"]),
                "probability_positive": float(brier_season.iloc[1]["probability_positive"]),
            },
        },
        "picks_disagreeing": len(disagreements),
        "disagreement_baseline_correct": (
            float(disagreements["baseline_correct_open"].mean())
            if len(disagreements)
            else float("nan")
        ),
        "disagreement_candidate_correct": (
            float(disagreements["candidate_correct_open"].mean())
            if len(disagreements)
            else float("nan")
        ),
        "candidate_column_flagged_games_in_window": flag_counts,
        "claim_gate": CONFIRM_AT,
        "rotation_verdict": rotation_verdict,
        "weak_signal_classification": signal_classification,
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (
                args.features,
                args.player_value_features,
                args.forecast_archive,
                args.schedules,
            )
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

    metadata = {
        "created_at_utc": summary["generated_at_utc"],
        "command": "scripts/combined_stacker_look.py",
        "predeclaration": "docs/combined_stacker_predeclaration.md",
        "summary": summary,
        "provenance": artifact_provenance(
            configuration={
                "script": "scripts/combined_stacker_look.py",
                "family": args.family,
                "baseline_profile": BASELINE_PROFILE,
                "candidate_profile": CANDIDATE_PROFILE,
                "candidate_columns": list(CANDIDATE_COLUMNS),
                "regressor": REGRESSOR,
                "ridge_alpha": RIDGE_ALPHA,
                "target": "market_residual",
                "min_train_games": args.min_train_games,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "claim_gate": CONFIRM_AT,
            },
            feature_path=args.features,
            project_root=REPO,
        ),
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    write_experiment_artifact(
        out_dir,
        "result.json",
        metadata,
        command="combined-stacker-look",
        metrics={
            "paired_games": len(paired),
            "weeks": summary["weeks"],
            "baseline_accuracy": summary["baseline_accuracy_opener_pr"],
            "candidate_accuracy": summary["candidate_accuracy_opener_pr"],
            "delta_accuracy_points": summary["delta"],
            "interval_low": summary["week_blocked"]["lower"],
            "interval_high": summary["week_blocked"]["upper"],
            "probability_positive": probability_positive,
            "sample_blocks": summary["weeks"],
            "effect_units": "accuracy_points",
            "rotation_verdict": rotation_verdict,
            "classification": signal_classification,
        },
        notes=(
            "Predeclared combined-stacker confirmation look on the assigned "
            f"{summary['window_seasons']} opener window; recorded separately via "
            "`nfl-ats rotation record` / `nfl-ats weak-signals record`."
        ),
        source="docs/combined_stacker_predeclaration.md",
        rotation_family=args.family,
        project_root=REPO,
    )
    paired.to_parquet(out_dir / "opener_paired.parquet")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {result_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="combined_stacker")
    parser.add_argument(
        "--features", type=Path, default=REPO / "data/processed/game_features_weak_stack.parquet"
    )
    parser.add_argument(
        "--player-value-features",
        type=Path,
        default=REPO / "data/processed/game_features_player_value.parquet",
    )
    parser.add_argument(
        "--forecast-archive",
        type=Path,
        default=REPO / "data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet",
    )
    parser.add_argument(
        "--schedules", type=Path, default=REPO / "data/raw/20260817T235649Z/schedules.parquet"
    )
    parser.add_argument("--market-root", type=Path, default=REPO / "data/market/raw")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--min-train-games", type=int, default=MIN_TRAIN_GAMES)
    parser.add_argument("--out-dir", type=Path, default=REPO / "artifacts/combined_stacker_look")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
