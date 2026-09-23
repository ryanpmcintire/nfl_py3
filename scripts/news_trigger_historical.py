from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, build_fit_population
from nfl_ats.pick_probability_fit import _fit_logit as fit_logit
from nfl_ats.signal_atlas import _cell as signal_cell

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = ARTIFACTS_ROOT / "news_trigger_historical"

POINT_IN_TIME_COLUMN = "home_injury_observed_at"
FIRST_SEASON = 2020
LAST_SEASON = 2024
QUESTIONABLE_SEVERITY = 0.35
FITTED_SLOPE_PER_UNIT_VALUE_LOST_DIFF = -0.320
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923
INTERVAL_LEVEL = 0.95
CANDIDATE_FEATURES = (*FIT_FEATURES, "news_trigger_value_shift")


def load_kickoff(data_root: Path) -> pd.DataFrame:
    return pd.read_parquet(
        data_root / "processed" / "game_features.parquet",
        columns=["game_id", "kickoff"],
    )


def load_injury_totals(data_root: Path) -> pd.DataFrame:
    table = pd.read_parquet(
        data_root / "processed" / "game_features_player_value.parquet",
        columns=[
            "game_id",
            "home_injury_skill_epa_value_lost",
            "home_injury_defense_disruption_value_lost",
            "away_injury_skill_epa_value_lost",
            "away_injury_defense_disruption_value_lost",
            POINT_IN_TIME_COLUMN,
            "away_injury_observed_at",
        ]
    )
    table["home_total_value_lost"] = (
        table["home_injury_skill_epa_value_lost"]
        + table["home_injury_defense_disruption_value_lost"]
    )
    table["away_total_value_lost"] = (
        table["away_injury_skill_epa_value_lost"]
        + table["away_injury_defense_disruption_value_lost"]
    )
    table["point_in_time"] = (
        table[POINT_IN_TIME_COLUMN].notna() & table["away_injury_observed_at"].notna()
    )
    return table[
        ["game_id", "home_total_value_lost", "away_total_value_lost", "point_in_time"]
    ]


def team_resolution_table(data_root: Path) -> pd.DataFrame:
    outcomes = pd.read_parquet(
        data_root / "processed" / "injury_play_outcomes.parquet",
        columns=["game_id", "team", "report_category", "unavailable", "fixed_unavailability"],
    )
    severity_sum = (
        outcomes.groupby(["game_id", "team"], observed=True)["fixed_unavailability"]
        .sum()
        .rename("severity_sum")
    )
    questionable = outcomes.loc[outcomes["report_category"] == "questionable"].copy()
    questionable["resolution_delta"] = (
        questionable["unavailable"] - QUESTIONABLE_SEVERITY
    )
    resolution_sum = (
        questionable.groupby(["game_id", "team"], observed=True)["resolution_delta"]
        .sum()
        .rename("resolution_sum")
    )
    questionable_count = (
        questionable.groupby(["game_id", "team"], observed=True)["resolution_delta"]
        .size()
        .rename("questionable_count")
    )
    merged = pd.concat([severity_sum, resolution_sum, questionable_count], axis=1).reset_index()
    merged["resolution_sum"] = merged["resolution_sum"].fillna(0.0)
    merged["questionable_count"] = merged["questionable_count"].fillna(0).astype(int)
    return merged


def gate_visible(kickoff_et: pd.Series) -> pd.Series:
    weekday = kickoff_et.dt.day_name()
    hour = kickoff_et.dt.hour
    monday_block = weekday.eq("Monday")
    late_sunday_block = weekday.eq("Sunday") & hour.ge(16)
    return ~(monday_block | late_sunday_block)


def build_population(artifacts_root: Path, data_root: Path) -> tuple[pd.DataFrame, dict]:
    population, provenance = build_fit_population(artifacts_root, data_root)
    population = population.merge(
        load_kickoff(data_root), on="game_id", how="left", validate="one_to_one"
    )
    population = population.merge(
        load_injury_totals(data_root), on="game_id", how="left", validate="one_to_one"
    )

    resolutions = team_resolution_table(data_root)
    home = resolutions.rename(
        columns={
            "team": "home_team",
            "severity_sum": "home_severity_sum",
            "resolution_sum": "home_resolution_sum",
            "questionable_count": "home_questionable_count",
        }
    )
    away = resolutions.rename(
        columns={
            "team": "away_team",
            "severity_sum": "away_severity_sum",
            "resolution_sum": "away_resolution_sum",
            "questionable_count": "away_questionable_count",
        }
    )
    population = population.merge(home, on=["game_id", "home_team"], how="left")
    population = population.merge(away, on=["game_id", "away_team"], how="left")
    for column in (
        "home_severity_sum",
        "home_resolution_sum",
        "home_questionable_count",
        "away_severity_sum",
        "away_resolution_sum",
        "away_questionable_count",
    ):
        population[column] = population[column].fillna(0.0)

    home_v = np.where(
        population["home_severity_sum"] > 0.0,
        population["home_total_value_lost"] / population["home_severity_sum"],
        0.0,
    )
    away_v = np.where(
        population["away_severity_sum"] > 0.0,
        population["away_total_value_lost"] / population["away_severity_sum"],
        0.0,
    )
    home_shift = home_v * population["home_resolution_sum"]
    away_shift = away_v * population["away_resolution_sum"]

    kickoff_et = population["kickoff"].dt.tz_convert("America/New_York")
    population["visible_before_deadline"] = gate_visible(kickoff_et)
    raw_shift = home_shift - away_shift
    population["news_trigger_value_shift"] = np.where(
        population["visible_before_deadline"], raw_shift, 0.0
    )
    population["news_trigger_margin_points"] = (
        FITTED_SLOPE_PER_UNIT_VALUE_LOST_DIFF * population["news_trigger_value_shift"]
    )
    population["questionable_players_pregame"] = (
        population["home_questionable_count"] + population["away_questionable_count"]
    )
    return population, provenance


def design_matrix(
    frame: pd.DataFrame, features: tuple[str, ...], means: dict[str, float], stds: dict[str, float]
) -> np.ndarray:
    columns = [np.ones(len(frame))]
    for name in features:
        columns.append(((frame[name].astype(float) - means[name]) / stds[name]).to_numpy())
    return np.column_stack(columns)


def standardisers(
    train: pd.DataFrame, features: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, float]]:
    means = {name: float(train[name].mean()) for name in features}
    stds = {name: float(train[name].std(ddof=0)) or 1.0 for name in features}
    return means, stds


def predict(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    beta: np.ndarray,
    means: dict[str, float],
    stds: dict[str, float],
) -> np.ndarray:
    z = np.clip(design_matrix(frame, features, means, stds) @ beta, -35.0, 35.0)
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=float)


def natural_coefficients(
    beta: np.ndarray, features: tuple[str, ...], means: dict[str, float], stds: dict[str, float]
) -> dict[str, float]:
    natural = {name: float(beta[index + 1] / stds[name]) for index, name in enumerate(features)}
    intercept = float(beta[0])
    for index, name in enumerate(features):
        intercept -= float(beta[index + 1]) * means[name] / stds[name]
    natural["intercept"] = intercept
    return natural


def loso_logistic(
    population: pd.DataFrame, features: tuple[str, ...]
) -> tuple[pd.Series, dict[str, dict[str, float]]]:
    seasons = sorted(int(value) for value in population["season"].unique())
    out = pd.Series(np.nan, index=population.index, dtype=float)
    folds: dict[str, dict[str, float]] = {}
    for held in seasons:
        train = population.loc[population["season"].ne(held)]
        test = population.loc[population["season"].eq(held)]
        if train.empty or test.empty:
            continue
        means, stds = standardisers(train, features)
        beta = fit_logit(
            design_matrix(train, features, means, stds),
            train["home_covered"].astype(float).to_numpy(),
            FIT_RIDGE,
        )
        out.loc[test.index] = predict(test, features, beta, means, stds)
        folds[str(held)] = natural_coefficients(beta, features, means, stds)
    return out, folds


def classify(probability_positive: float) -> str:
    if probability_positive >= 0.975 or probability_positive <= 0.025:
        return "resolved_directional"
    return "unresolved_below_power"


def main() -> None:
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    population, provenance = build_population(ARTIFACTS_ROOT, DATA_ROOT)

    scoped = population.loc[
        population["season"].between(FIRST_SEASON, LAST_SEASON)
        & population["point_in_time"].fillna(False)
        & population["news_trigger_value_shift"].notna()
    ].reset_index(drop=True)

    identification = {
        "scoped_games": len(scoped),
        "games_with_any_questionable_pregame": int(
            scoped["questionable_players_pregame"].gt(0).sum()
        ),
        "games_passing_visibility_gate": int(scoped["visible_before_deadline"].sum()),
        "games_with_nonzero_resolved_shift": int(
            scoped["news_trigger_value_shift"].ne(0.0).sum()
        ),
        "games_gated_out_with_questionable_pregame": int(
            (
                scoped["questionable_players_pregame"].gt(0)
                & ~scoped["visible_before_deadline"]
            ).sum()
        ),
        "mean_abs_margin_points_when_nonzero": float(
            scoped.loc[
                scoped["news_trigger_value_shift"].ne(0.0), "news_trigger_margin_points"
            ]
            .abs()
            .mean()
        )
        if scoped["news_trigger_value_shift"].ne(0.0).any()
        else 0.0,
    }

    baseline_oos, baseline_folds = loso_logistic(scoped, FIT_FEATURES)
    candidate_oos, candidate_folds = loso_logistic(scoped, CANDIDATE_FEATURES)

    scored = scoped.copy()
    scored["rating_full"] = candidate_oos
    scored["rating_reduced"] = baseline_oos
    scored = scored.loc[scored["rating_full"].notna() & scored["rating_reduced"].notna()].copy()

    declaration = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "interval_level": INTERVAL_LEVEL,
        "reliability_edges": [0.0, 0.5, 1.0],
    }
    probability_cell = signal_cell(scored, "rating", "overall", declaration)

    summary = {
        "predeclaration": {
            "predictor": "news_trigger_value_shift = v_home * sum(unavailable_actual - 0.35 "
            "over home questionable players) - v_away * sum(unavailable_actual - 0.35 over "
            "away questionable players), v = team pregame total_value_lost / team severity_sum",
            "visibility_gate": "not Monday and not (Sunday and kickoff_hour_et >= 16)",
            "fitted_slope_source": "docs/lanes/injury-scenario-producer.md Unit 2, "
            f"{FITTED_SLOPE_PER_UNIT_VALUE_LOST_DIFF} points per unit value_lost_diff",
            "seasons": [FIRST_SEASON, LAST_SEASON],
            "point_in_time_definition": f"{POINT_IN_TIME_COLUMN} and "
            "away_injury_observed_at not null",
            "opener_evaluation": provenance["opener_evaluation"],
        },
        "identification": identification,
        "cover_probability_paired_look": {
            "baseline_features": list(FIT_FEATURES),
            "candidate_features": list(CANDIDATE_FEATURES),
            "baseline_folds": baseline_folds,
            "candidate_folds": candidate_folds,
            "cell": probability_cell,
            "classification": classify(probability_cell["probability_positive"]),
        },
    }

    scoped.to_parquet(output_dir / "scoped_population.parquet", index=False)
    scored.to_parquet(output_dir / "paired_probability_population.parquet", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(identification, indent=2))
    print(
        "probability cell accuracy_delta_points:",
        probability_cell["accuracy_delta_points"],
        "probability_positive:",
        probability_cell["probability_positive"],
    )
    print(
        "decisive games:",
        probability_cell["decisive_games"],
        "full(candidate) wins:",
        probability_cell["full_decisive_wins"],
        "reduced(baseline) wins:",
        probability_cell["reduced_decisive_wins"],
    )
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
