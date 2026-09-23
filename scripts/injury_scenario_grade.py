from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from nfl_ats.availability import availability_rate_lookup, build_season_lagged_availability_rates
from nfl_ats.discrete_margin_mapping import discrete_side_read
from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, build_fit_population
from nfl_ats.pick_probability_fit import _fit_logit as fit_logit
from nfl_ats.signal_atlas import _cell as signal_cell

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = ARTIFACTS_ROOT / "injury_scenario_grade"
MARGIN_MAP_SUMMARY = (
    ARTIFACTS_ROOT / "injury_value_margin_map" / "20260923T210029Z" / "summary.json"
)

VALUE_LOST_COLUMNS = (
    "home_injury_skill_epa_value_lost",
    "home_injury_defense_disruption_value_lost",
    "away_injury_skill_epa_value_lost",
    "away_injury_defense_disruption_value_lost",
)
POINT_IN_TIME_COLUMN = "home_injury_observed_at"
FIRST_SEASON = 2020
LAST_SEASON = 2024
SPARSE_CELL_FLOOR = 250
SIGNAL_REPORT_CATEGORIES = frozenset(("out", "doubtful", "questionable"))
SIGNAL_PRACTICE_CATEGORIES = frozenset(("dnp", "limited"))
MAX_BORDERLINE_PER_SIDE = 6
CENTER_BRACKET = (-40.0, 40.0)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923
INTERVAL_LEVEL = 0.95
UNIT_COUPLING_MULTIPLIER = {
    "offensive_line": 1.192,
    "skill": 1.235,
    "front": 1.175,
    "secondary": 1.245,
}


def load_margin_slope_folds() -> dict[int, float]:
    summary = json.loads(MARGIN_MAP_SUMMARY.read_text(encoding="utf-8"))
    folds = summary["margin_regression"]["loso_folds"]
    return {int(season): float(fold["slope_points_per_unit"]) for season, fold in folds.items()}


def load_player_value(data_root: Path) -> pd.DataFrame:
    table = pd.read_parquet(
        data_root / "processed" / "game_features_player_value.parquet",
        columns=["game_id", *VALUE_LOST_COLUMNS, POINT_IN_TIME_COLUMN],
    )
    for column in VALUE_LOST_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="raise")
    table["base_home_value_lost"] = (
        table["home_injury_skill_epa_value_lost"]
        + table["home_injury_defense_disruption_value_lost"]
    )
    table["base_away_value_lost"] = (
        table["away_injury_skill_epa_value_lost"]
        + table["away_injury_defense_disruption_value_lost"]
    )
    table["base_value_lost_diff"] = table["base_home_value_lost"] - table["base_away_value_lost"]
    table["point_in_time"] = table[POINT_IN_TIME_COLUMN].notna()
    return table[
        [
            "game_id",
            "base_home_value_lost",
            "base_away_value_lost",
            "base_value_lost_diff",
            "point_in_time",
        ]
    ]


def build_rate_lookup(
    outcomes: pd.DataFrame, target_seasons: list[int]
) -> tuple[dict[tuple[int, str, str, str], float], dict[tuple[int, str, str, str], int]]:
    rates = build_season_lagged_availability_rates(outcomes, target_seasons=target_seasons)
    lookup = availability_rate_lookup(rates)
    observations = {
        (
            int(row.target_season),
            str(row.report_category),
            str(row.practice_category),
            str(row.position_group),
        ): int(row.observations)
        for row in rates.itertuples(index=False)
    }
    return lookup, observations


def resolve_severity(
    lookup: dict[tuple[int, str, str, str], float],
    observations: dict[tuple[int, str, str, str], int],
    *,
    season: int,
    report_category: str,
    practice_category: str,
    position_group: str,
    fixed_value: float,
    floor: int,
) -> tuple[float, str]:
    for key in (
        (season, report_category, practice_category, position_group),
        (season, report_category, practice_category, "__all__"),
    ):
        if key in lookup and observations.get(key, 0) >= floor:
            return lookup[key], "season_lagged_rate"
    return fixed_value, "fixed_status_prior"


def build_borderline_table(
    outcomes: pd.DataFrame,
    lookup: dict[tuple[int, str, str, str], float],
    observations: dict[tuple[int, str, str, str], int],
) -> pd.DataFrame:
    rows = outcomes.copy()
    rows = rows.loc[
        rows["report_category"].isin(SIGNAL_REPORT_CATEGORIES)
        | rows["practice_category"].isin(SIGNAL_PRACTICE_CATEGORIES)
    ].reset_index(drop=True)
    resolved = [
        resolve_severity(
            lookup,
            observations,
            season=int(season),
            report_category=str(report),
            practice_category=str(practice),
            position_group=str(group),
            fixed_value=float(fixed),
            floor=SPARSE_CELL_FLOOR,
        )
        for season, report, practice, group, fixed in zip(
            rows["season"],
            rows["report_category"],
            rows["practice_category"],
            rows["position_group"],
            rows["fixed_unavailability"],
            strict=True,
        )
    ]
    rows["sit_probability"] = [value for value, _ in resolved]
    rows["severity_source"] = [source for _, source in resolved]
    return rows.loc[rows["sit_probability"] > 0.0].reset_index(drop=True)


def side_subsets(players: list[dict]) -> tuple[list[dict], bool]:
    truncated = len(players) > MAX_BORDERLINE_PER_SIDE
    kept = sorted(players, key=lambda player: player["sit_probability"], reverse=True)[
        :MAX_BORDERLINE_PER_SIDE
    ]
    if not kept:
        return [{"probability": 1.0, "sit_fraction": 0.0}], truncated
    total_severity = sum(player["sit_probability"] for player in kept)
    subsets = []
    for mask in itertools.product((False, True), repeat=len(kept)):
        sitting = [player for player, sit in zip(kept, mask, strict=True) if sit]
        independent_probability = 1.0
        for player, sit in zip(kept, mask, strict=True):
            independent_probability *= (
                player["sit_probability"] if sit else 1.0 - player["sit_probability"]
            )
        coupled_probability = independent_probability
        for left, right in itertools.combinations(sitting, 2):
            if left["unit"] == right["unit"] and left["unit"] in UNIT_COUPLING_MULTIPLIER:
                coupled_probability *= UNIT_COUPLING_MULTIPLIER[left["unit"]]
        sit_severity = sum(player["sit_probability"] for player in sitting)
        sit_fraction = sit_severity / total_severity if total_severity > 0 else 0.0
        subsets.append({"probability": coupled_probability, "sit_fraction": sit_fraction})
    normalizer = sum(subset["probability"] for subset in subsets)
    for subset in subsets:
        subset["probability"] = subset["probability"] / normalizer
    return subsets, truncated


def find_center(
    pool_line: np.ndarray, pool_margin: np.ndarray, line: float, target: float
) -> tuple[float | None, bool]:
    def probability_at(point: float) -> float:
        return discrete_side_read(pool_line, pool_margin, line, point).home_cover_probability

    lo, hi = CENTER_BRACKET
    f_lo, f_hi = probability_at(lo) - target, probability_at(hi) - target
    if f_lo > 0.0 or f_hi < 0.0:
        return None, False
    if f_lo == 0.0:
        return lo, True
    if f_hi == 0.0:
        return hi, True
    point = brentq(lambda p: probability_at(p) - target, lo, hi, xtol=1e-6)
    return float(point), True


def main() -> None:
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    population, provenance = build_fit_population(ARTIFACTS_ROOT, DATA_ROOT)
    injury = load_player_value(DATA_ROOT)
    population = population.merge(injury, on="game_id", how="left", validate="one_to_one")
    scoped = population.loc[
        population["season"].between(FIRST_SEASON, LAST_SEASON)
        & population["point_in_time"].fillna(False)
        & population["base_value_lost_diff"].notna()
    ].reset_index(drop=True)

    outcomes = pd.read_parquet(DATA_ROOT / "processed" / "injury_play_outcomes.parquet")
    lookup, observations = build_rate_lookup(outcomes, list(range(FIRST_SEASON, LAST_SEASON + 1)))
    borderline = build_borderline_table(outcomes, lookup, observations)
    borderline_by_game_team: dict[tuple[str, str], list[dict]] = {}
    for row in borderline.itertuples(index=False):
        key = (str(row.game_id), str(row.team))
        borderline_by_game_team.setdefault(key, []).append(
            {
                "unit": str(row.position_group),
                "sit_probability": float(row.sit_probability),
            }
        )

    game_features = pd.read_parquet(
        DATA_ROOT / "processed" / "game_features.parquet",
        columns=["season", "spread_line", "result"],
    )
    game_features = game_features.loc[
        game_features["spread_line"].notna() & game_features["result"].notna()
    ]
    pools: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        prior = game_features.loc[game_features["season"] < season]
        pools[season] = (
            prior["spread_line"].to_numpy(dtype=float),
            prior["result"].to_numpy(dtype=float),
        )

    slope_folds = load_margin_slope_folds()

    mixed_probabilities = []
    scenario_counts = []
    truncated_flags = []
    center_failures = 0
    zero_borderline_nonzero_diff = 0
    for row in scoped.itertuples(index=False):
        season = int(row.season)
        pool_line, pool_margin = pools[season]
        served = float(row.model_probability)
        home_players = borderline_by_game_team.get((str(row.game_id), str(row.home_team)), [])
        away_players = borderline_by_game_team.get((str(row.game_id), str(row.away_team)), [])
        if not home_players and not away_players:
            if abs(float(row.base_value_lost_diff)) > 1e-9:
                zero_borderline_nonzero_diff += 1
            mixed_probabilities.append(served)
            scenario_counts.append(1)
            truncated_flags.append(False)
            continue
        point, ok = find_center(pool_line, pool_margin, float(row.tue_open_home_spread), served)
        if not ok:
            center_failures += 1
            mixed_probabilities.append(served)
            scenario_counts.append(0)
            truncated_flags.append(False)
            continue
        home_subsets, home_trunc = side_subsets(home_players)
        away_subsets, away_trunc = side_subsets(away_players)
        slope = slope_folds[season]
        mixed = 0.0
        count = 0
        for home_subset, away_subset in itertools.product(home_subsets, away_subsets):
            scenario_probability = home_subset["probability"] * away_subset["probability"]
            scenario_home_value = float(row.base_home_value_lost) * home_subset["sit_fraction"]
            scenario_away_value = float(row.base_away_value_lost) * away_subset["sit_fraction"]
            scenario_diff = scenario_home_value - scenario_away_value
            margin_shift = slope * (scenario_diff - float(row.base_value_lost_diff))
            line = float(row.tue_open_home_spread) - margin_shift
            read_probability = discrete_side_read(
                pool_line, pool_margin, line, point
            ).home_cover_probability
            mixed += scenario_probability * read_probability
            count += 1
        mixed_probabilities.append(mixed)
        scenario_counts.append(count)
        truncated_flags.append(bool(home_trunc or away_trunc))

    scoped = scoped.copy()
    scoped["scenario_mixed_home_cover_probability"] = mixed_probabilities
    scoped["scenario_count"] = scenario_counts
    scoped["scenario_truncated"] = truncated_flags
    scoped["scenario_shift"] = (
        scoped["scenario_mixed_home_cover_probability"] - scoped["model_probability"]
    )

    declaration = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "interval_level": INTERVAL_LEVEL,
        "reliability_edges": [0.0, 0.5, 1.0],
    }

    raw = scoped.copy()
    raw["rating_full"] = raw["scenario_mixed_home_cover_probability"].clip(1e-6, 1.0 - 1e-6)
    raw["rating_reduced"] = raw["model_probability"].clip(1e-6, 1.0 - 1e-6)
    raw_cell = signal_cell(raw, "rating", "overall", declaration)

    def design(frame: pd.DataFrame, features: tuple[str, ...], means, stds) -> np.ndarray:
        columns = [np.ones(len(frame))]
        for name in features:
            columns.append(((frame[name].astype(float) - means[name]) / stds[name]).to_numpy())
        return np.column_stack(columns)

    def standardisers(train: pd.DataFrame, features: tuple[str, ...]):
        means = {name: float(train[name].mean()) for name in features}
        stds = {name: float(train[name].std(ddof=0)) or 1.0 for name in features}
        return means, stds

    def predict(frame: pd.DataFrame, features: tuple[str, ...], beta, means, stds) -> np.ndarray:
        z = np.clip(design(frame, features, means, stds) @ beta, -35.0, 35.0)
        return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=float)

    def loso(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.Series:
        seasons = sorted(int(value) for value in frame["season"].unique())
        out = pd.Series(np.nan, index=frame.index, dtype=float)
        for held in seasons:
            train = frame.loc[frame["season"].ne(held)]
            test = frame.loc[frame["season"].eq(held)]
            if train.empty or test.empty:
                continue
            means, stds = standardisers(train, features)
            beta = fit_logit(
                design(train, features, means, stds),
                train["home_covered"].astype(float).to_numpy(),
                FIT_RIDGE,
            )
            out.loc[test.index] = predict(test, features, beta, means, stds)
        return out

    candidate_features = (*FIT_FEATURES, "scenario_shift")
    baseline_oos = loso(scoped, FIT_FEATURES)
    candidate_oos = loso(scoped, candidate_features)
    fitted = scoped.copy()
    fitted["rating_full"] = candidate_oos
    fitted["rating_reduced"] = baseline_oos
    fitted = fitted.loc[fitted["rating_full"].notna() & fitted["rating_reduced"].notna()].copy()
    fitted["rating_full"] = fitted["rating_full"].clip(1e-6, 1.0 - 1e-6)
    fitted["rating_reduced"] = fitted["rating_reduced"].clip(1e-6, 1.0 - 1e-6)
    fitted_cell = signal_cell(fitted, "rating", "overall", declaration)

    summary = {
        "predeclaration_lane": "docs/lanes/injury-scenario-producer.md Unit 5 predeclaration",
        "scoped_games": len(scoped),
        "games_zero_borderline_both_sides_with_nonzero_base_diff": zero_borderline_nonzero_diff,
        "center_bisection_failures": center_failures,
        "games_with_side_truncated_at_cap": int(sum(truncated_flags)),
        "sparse_cell_floor_observations": SPARSE_CELL_FLOOR,
        "max_borderline_per_side": MAX_BORDERLINE_PER_SIDE,
        "margin_slope_folds_source": str(
            MARGIN_MAP_SUMMARY.relative_to(REPO_ROOT)
        ).replace("\\", "/"),
        "margin_slope_folds": slope_folds,
        "opener_evaluation": provenance["opener_evaluation"],
        "raw_scenario_vs_served": raw_cell,
        "fitted_term_scenario_shift_vs_4term_baseline": fitted_cell,
    }
    scoped.to_parquet(output_dir / "scoped_population.parquet", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(
        "scoped_games:", len(scoped),
        "zero_borderline_nonzero_diff:", zero_borderline_nonzero_diff,
        "center_failures:", center_failures,
        "truncated:", int(sum(truncated_flags)),
    )
    print(
        "raw accuracy_delta_points:", raw_cell["accuracy_delta_points"],
        "probability_positive:", raw_cell["probability_positive"],
        "brier_improvement:", raw_cell["brier_improvement"],
        "brier_probability_positive:", raw_cell["brier_probability_positive"],
        "log_loss_improvement:", raw_cell["log_loss_improvement"],
        "log_loss_probability_positive:", raw_cell["log_loss_probability_positive"],
        "decisive_games:", raw_cell["decisive_games"],
        "full_decisive_wins:", raw_cell["full_decisive_wins"],
        "reduced_decisive_wins:", raw_cell["reduced_decisive_wins"],
        "exact_null_p:", raw_cell["exact_null_p"],
    )
    print(
        "fitted accuracy_delta_points:", fitted_cell["accuracy_delta_points"],
        "probability_positive:", fitted_cell["probability_positive"],
        "brier_improvement:", fitted_cell["brier_improvement"],
        "brier_probability_positive:", fitted_cell["brier_probability_positive"],
        "log_loss_improvement:", fitted_cell["log_loss_improvement"],
        "log_loss_probability_positive:", fitted_cell["log_loss_probability_positive"],
        "decisive_games:", fitted_cell["decisive_games"],
        "full_decisive_wins:", fitted_cell["full_decisive_wins"],
        "reduced_decisive_wins:", fitted_cell["reduced_decisive_wins"],
        "exact_null_p:", fitted_cell["exact_null_p"],
    )
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
