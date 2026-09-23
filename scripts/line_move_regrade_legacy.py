from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from line_move_yardstick_paired_eval import cell_stats  # noqa: E402
from roof_state_screen import build_prediction_table as roof_state_prediction_table  # noqa: E402

from nfl_ats.bye_edge_fade_overlay import bye_edge_flag_by_game  # noqa: E402
from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay import (  # noqa: E402
    precip_high_total_flag_by_game,
)
from nfl_ats.interim_hc_first_game_tilt_overlay import (  # noqa: E402
    interim_first_game_flag_by_game_fail_open,
)
from nfl_ats.pick_probability_fit import build_fit_population  # noqa: E402
from nfl_ats.schedule_flag_features import (  # noqa: E402
    DIVISION_DOG_COLUMN,
    WEEK1_DOG_COLUMN,
    default_schedule,
    derive_division_dog_features,
    derive_week1_dog_features,
)
from nfl_ats.tank_zone_fade_tilt_overlay import tank_zone_flag_by_game  # noqa: E402
from nfl_ats.transaction_flag_features import (  # noqa: E402
    DEADLINE_INTEGRATION_DRAG_COLUMN,
    attach_deadline_integration_drag_features,
)

BASE_FEATURES = ("model_logit", "composition_flag_sum")
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923
FORECAST_ARCHIVE = (
    REPO / "data" / "raw" / "forecast_archive" / "pool_decision_2009_2025" / "forecasts.parquet"
)

SELECTION_RULE = (
    "Predeclared before fitting (2026-09-23): among registry families in category "
    "schedule/environment/offfield/health with classification unresolved_below_power and "
    "effect_units accuracy_points, rank distinct families by |effect|/standard_error "
    "(standard_error from the field, else (interval_high-interval_low)/(2*1.96)); take each "
    "family's own highest-ratio entry; exclude CFB-only/benchmark-transfer families (out of "
    "NFL 2020-2025 scope), positive-control and refuted_mechanism/closed entries, families "
    "already graded on line-move today (players_on_field_rating_v1, opener_error_transfer_v1/2/3, "
    "pbp08_protection_mismatch_gated_flag_sum_fit, reddit/pooled_signal_sixth_fit), referee/crew "
    "families (this repo's own crew_tilt_stacked_on_production entry documents the underlying "
    "flag as a 'late-week officiating-crew tilt', not Tuesday-knowable), and health-category "
    "families whose construction depends on in-week injury/practice reports (Friday designations, "
    "Wednesday DNP, weekly Out reports) rather than information knowable by Tuesday noon. Of the "
    "families with a confirmed standalone rebuildable NFL feature builder already in src/nfl_ats "
    "(no CFB pipeline, no in-week injury report, no referee-crew timing), take the top 8 by ratio."
)

RATIO_TABLE = (
    ("deadline_integration_drag_on_production", 2.218, "offfield"),
    ("tank_zone_fade_tilt__spread_band_short_2020_2025", 2.098, "onfield/standings"),
    ("bye_edge_fade__spread_band_long_2020_2025", 1.940, "onfield/rest"),
    ("roof_state_predicted_open_fade_on_production", 1.572, "environment/weather-forecast"),
    (
        "precip_high_total_tilt__week_in_season_weeks_1_4_2020_2025",
        1.481,
        "onfield/weather-forecast",
    ),
    ("week1_dog_on_production", 1.473, "schedule"),
    ("interim_hc_first_game_tilt__week_in_season_weeks_5_12_2020_2025", 1.449, "onfield/coach"),
    ("division_dog_on_production", 1.338, "schedule"),
)


def add_tank_zone_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    flags = tank_zone_flag_by_game(schedule)
    flags = flags.drop_duplicates(subset="game_id")[["game_id", "tank_zone_home", "tank_zone_away"]]
    out = population.merge(flags, on="game_id", how="left")
    out["tank_zone_home"] = out["tank_zone_home"].fillna(False)
    out["tank_zone_away"] = out["tank_zone_away"].fillna(False)
    out["tank_zone_term"] = np.where(
        out["tank_zone_home"], 1.0, np.where(out["tank_zone_away"], -1.0, 0.0)
    )
    return out


def add_bye_edge_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    flags = bye_edge_flag_by_game(schedule)
    flags = flags.drop_duplicates(subset="game_id")[["game_id", "home_off_bye", "away_off_bye"]]
    out = population.merge(flags, on="game_id", how="left")
    out["home_off_bye"] = out["home_off_bye"].fillna(False)
    out["away_off_bye"] = out["away_off_bye"].fillna(False)
    out["bye_edge_term"] = np.where(
        out["home_off_bye"], 1.0, np.where(out["away_off_bye"], -1.0, 0.0)
    )
    return out


def add_roof_state_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    table = roof_state_prediction_table()[["game_id", "predicted_open"]].drop_duplicates(
        subset="game_id"
    )
    table["game_id"] = table["game_id"].astype(str)
    out = population.merge(table, on="game_id", how="left")
    out["predicted_open"] = out["predicted_open"].fillna(False)
    out["roof_state_term"] = out["predicted_open"].astype(float)
    return out


def add_precip_high_total_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    forecasts = pd.read_parquet(FORECAST_ARCHIVE)
    total_lines = schedule[["game_id", "total_line"]].copy()
    total_lines["game_id"] = total_lines["game_id"].astype(str)
    flags = precip_high_total_flag_by_game(schedule, forecasts, total_lines)
    flags = flags.drop_duplicates(subset="game_id")[["game_id", "precip_high_total_flag"]]
    out = population.merge(flags, on="game_id", how="left")
    out["precip_high_total_flag"] = out["precip_high_total_flag"].fillna(False)
    out["precip_high_total_term"] = out["precip_high_total_flag"].astype(float)
    return out


def add_week1_dog_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    opener_lines = population[["game_id", "tue_open_home_spread"]].copy()
    derived = derive_week1_dog_features(schedule, opener_lines)
    derived = derived.drop_duplicates(subset="game_id")
    out = population.merge(derived, on="game_id", how="left")
    out[WEEK1_DOG_COLUMN] = out[WEEK1_DOG_COLUMN].fillna(0.0)
    return out


def add_division_dog_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    opener_lines = population[["game_id", "tue_open_home_spread"]].copy()
    derived = derive_division_dog_features(schedule, opener_lines)
    derived = derived.drop_duplicates(subset="game_id")
    out = population.merge(derived, on="game_id", how="left")
    out[DIVISION_DOG_COLUMN] = out[DIVISION_DOG_COLUMN].fillna(0.0)
    return out


def add_interim_hc_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    flags = interim_first_game_flag_by_game_fail_open(REPO)
    sched = schedule[["game_id", "home_team", "away_team"]].copy()
    sched["game_id"] = sched["game_id"].astype(str)
    if flags.empty:
        sched["interim_hc_term"] = 0.0
    else:
        flags = flags.copy()
        flags["game_id"] = flags["game_id"].astype(str)
        home_hits = set(zip(flags["game_id"], flags["team"], strict=False))
        sched["home_first_game"] = [
            (gid, team) in home_hits
            for gid, team in zip(sched["game_id"], sched["home_team"], strict=False)
        ]
        sched["away_first_game"] = [
            (gid, team) in home_hits
            for gid, team in zip(sched["game_id"], sched["away_team"], strict=False)
        ]
        sched["interim_hc_term"] = np.where(
            sched["home_first_game"], 1.0, np.where(sched["away_first_game"], -1.0, 0.0)
        )
    out = population.merge(sched[["game_id", "interim_hc_term"]], on="game_id", how="left")
    out["interim_hc_term"] = out["interim_hc_term"].fillna(0.0)
    return out


def add_deadline_drag_term(population: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    features = population[["game_id"]].copy()
    features["game_id"] = features["game_id"].astype(str)
    try:
        attached = attach_deadline_integration_drag_features(features, schedule=schedule)
    except DataContractError:
        attached = features.copy()
        attached[DEADLINE_INTEGRATION_DRAG_COLUMN] = 0.0
    attached = attached.drop_duplicates(subset="game_id")
    out = population.merge(attached, on="game_id", how="left")
    out[DEADLINE_INTEGRATION_DRAG_COLUMN] = out[DEADLINE_INTEGRATION_DRAG_COLUMN].fillna(0.0)
    return out


def loso(frame: pd.DataFrame, features: tuple[str, ...]) -> tuple[np.ndarray, dict]:
    frame = frame.reset_index(drop=True)
    x_cols = list(features)
    oos = np.full(len(frame), np.nan)
    folds: dict[str, list[float]] = {}
    for season in sorted(frame["season"].unique()):
        train = frame.loc[frame["season"] != season]
        test_idx = frame.index[frame["season"] == season]
        means = {name: float(train[name].mean()) for name in x_cols}
        stds = {name: float(train[name].std(ddof=0)) or 1.0 for name in x_cols}
        design_train = np.column_stack(
            [np.ones(len(train))]
            + [((train[name] - means[name]) / stds[name]).to_numpy() for name in x_cols]
        )
        y_train = train["home_covered"].astype(float).to_numpy()
        beta = np.zeros(design_train.shape[1])
        ridge = 1e-3
        for _ in range(50):
            z = np.clip(design_train @ beta, -35.0, 35.0)
            p = 1.0 / (1.0 + np.exp(-z))
            w = np.clip(p * (1.0 - p), 1e-6, None)
            gradient = design_train.T @ (y_train - p) - ridge * beta
            hessian = (design_train.T * w) @ design_train + ridge * np.eye(design_train.shape[1])
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            beta = beta + step
        test = frame.loc[test_idx]
        design_test = np.column_stack(
            [np.ones(len(test))]
            + [((test[name] - means[name]) / stds[name]).to_numpy() for name in x_cols]
        )
        z_test = np.clip(design_test @ beta, -35.0, 35.0)
        oos[test_idx] = 1.0 / (1.0 + np.exp(-z_test))
        folds[str(int(season))] = beta.tolist()
    return oos, folds


TERM_DECLARATIONS = (
    {
        "label": "tank_zone_fade_tilt",
        "term_columns": ("tank_zone_term",),
        "builder": add_tank_zone_term,
    },
    {"label": "bye_edge_fade", "term_columns": ("bye_edge_term",), "builder": add_bye_edge_term},
    {
        "label": "roof_state_predicted_open",
        "term_columns": ("roof_state_term",),
        "builder": add_roof_state_term,
    },
    {
        "label": "precip_high_total_tilt",
        "term_columns": ("precip_high_total_term",),
        "builder": add_precip_high_total_term,
    },
    {"label": "week1_dog", "term_columns": (WEEK1_DOG_COLUMN,), "builder": add_week1_dog_term},
    {
        "label": "interim_hc_first_game_tilt",
        "term_columns": ("interim_hc_term",),
        "builder": add_interim_hc_term,
    },
    {
        "label": "division_dog",
        "term_columns": (DIVISION_DOG_COLUMN,),
        "builder": add_division_dog_term,
    },
    {
        "label": "deadline_integration_drag",
        "term_columns": (DEADLINE_INTEGRATION_DRAG_COLUMN,),
        "builder": add_deadline_drag_term,
    },
)


def variant_report(
    declaration: dict, population: pd.DataFrame, schedule: pd.DataFrame, seed: int, draws: int
) -> dict[str, object]:
    enriched = declaration["builder"](population, schedule)
    term_columns = list(declaration["term_columns"])
    subset = enriched.dropna(subset=[*term_columns, "open_move"]).reset_index(drop=True)

    features_variant = (*BASE_FEATURES, *term_columns)
    base_oos, base_folds = loso(subset, BASE_FEATURES)
    variant_oos, variant_folds = loso(subset, features_variant)

    frame = subset.copy()
    frame["base_oos_p"] = base_oos
    frame["variant_oos_p"] = variant_oos
    frame = frame.loc[frame["base_oos_p"].notna() & frame["variant_oos_p"].notna()].copy()

    frame["base_pick_home"] = frame["base_oos_p"].ge(0.5)
    frame["variant_pick_home"] = frame["variant_oos_p"].ge(0.5)
    frame["lm_base"] = np.where(frame["base_pick_home"], 1.0, -1.0) * frame["open_move"]
    frame["lm_variant"] = np.where(frame["variant_pick_home"], 1.0, -1.0) * frame["open_move"]
    frame["diff_line_move"] = frame["lm_variant"] - frame["lm_base"]

    frame["base_correct"] = (
        frame["base_pick_home"].astype(float).eq(frame["home_covered"]).astype(float)
    )
    frame["variant_correct"] = (
        frame["variant_pick_home"].astype(float).eq(frame["home_covered"]).astype(float)
    )
    frame["diff_accuracy"] = frame["variant_correct"] - frame["base_correct"]

    line_move_cell = cell_stats(
        f"{declaration['label']}_line_move_toward_pick", frame, "diff_line_move", seed, draws
    )
    accuracy_cell = cell_stats(
        f"{declaration['label']}_accuracy_companion", frame, "diff_accuracy", seed + 500, draws
    )
    positive_rate = (
        float(frame[term_columns[0]].astype(bool).mean()) if len(term_columns) == 1 else None
    )
    return {
        "label": declaration["label"],
        "term_columns": term_columns,
        "base_features": list(BASE_FEATURES),
        "variant_features": list(features_variant),
        "coverage_games_before_pairing": len(subset),
        "paired_games": len(frame),
        "seasons": sorted(int(v) for v in frame["season"].unique()),
        "term_nonzero_rate": positive_rate,
        "base_fold_coefficients": base_folds,
        "variant_fold_coefficients": variant_folds,
        "line_move_toward_pick_cell": line_move_cell,
        "accuracy_companion_cell": accuracy_cell,
    }


def main() -> int:
    now = datetime.now(UTC)
    population, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    population = population.dropna(subset=["open_move"]).reset_index(drop=True)
    population["game_id"] = population["game_id"].astype(str)
    schedule = default_schedule()
    schedule["game_id"] = schedule["game_id"].astype(str)

    reports = []
    for declaration in TERM_DECLARATIONS:
        try:
            reports.append(
                variant_report(declaration, population, schedule, BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
            )
        except Exception as exc:
            reports.append({"label": declaration["label"], "error": f"{type(exc).__name__}: {exc}"})

    results = {
        "command": "python scripts/line_move_regrade_legacy.py",
        "created_at_utc": now.isoformat(),
        "unit": "Tuesday-terms line-move regrade, legacy registry families",
        "selection_rule": SELECTION_RULE,
        "predeclared_ratio_table": [
            {"registry_name": name, "prior_abs_effect_over_se": ratio, "category": cat}
            for name, ratio, cat in RATIO_TABLE
        ],
        "base_population_games": len(population),
        "seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "opener_evaluation_provenance": provenance,
        "terms": reports,
    }

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts" / "line_move_regrade_legacy" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        __import__("json").dumps(results, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    summary = {
        "artifact": str(out_dir.relative_to(REPO)).replace("\\", "/"),
        "base_population_games": len(population),
        "terms": [
            {
                "label": report.get("label"),
                "error": report.get("error"),
                "paired_games": report.get("paired_games"),
                "line_move_mean_points": (report.get("line_move_toward_pick_cell") or {}).get(
                    "mean_points"
                ),
                "line_move_season_block_interval": [
                    (report.get("line_move_toward_pick_cell") or {}).get(
                        "season_block_interval_low"
                    ),
                    (report.get("line_move_toward_pick_cell") or {}).get(
                        "season_block_interval_high"
                    ),
                ],
                "line_move_season_block_probability_positive": (
                    report.get("line_move_toward_pick_cell") or {}
                ).get("season_block_probability_positive"),
                "accuracy_mean_points": (report.get("accuracy_companion_cell") or {}).get(
                    "mean_points"
                ),
                "accuracy_season_block_probability_positive": (
                    report.get("accuracy_companion_cell") or {}
                ).get("season_block_probability_positive"),
            }
            for report in reports
        ],
    }
    print(__import__("json").dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
