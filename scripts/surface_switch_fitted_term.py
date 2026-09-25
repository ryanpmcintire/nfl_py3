from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from line_move_yardstick_paired_eval import _design, _standardise, cell_stats  # noqa: E402
from line_move_yardstick_paired_eval import _predict as _predict_features  # noqa: E402

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES  # noqa: E402
from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.pick_probability_fit import (  # noqa: E402
    FIT_FEATURES,
    FIT_RIDGE,
    _fit_logit,
    build_fit_population,
)
from nfl_ats.snapshots import latest_snapshot, load_snapshot  # noqa: E402

EXTENDED_POPULATION_PATH = (
    REPO / "artifacts/extended_fit_population/20260923T205910Z/population.parquet"
)
FEATURES_PATH = REPO / "data/processed/game_features_weak_stack.parquet"
SBR_ODDS_PATH = REPO / "data/processed/sbr_odds.parquet"
SBR_SEASON_START = 2013
SBR_SEASON_END = 2019

GRASS_SURFACES = frozenset({"grass", "dessograss"})
TURF_SURFACES = frozenset(
    {"fieldturf", "sportturf", "matrixturf", "astroturf", "a_turf", "astroplay"}
)
NEW_TERM = "surface_switch_flag_pit"
OPENER_ONLY_BASE = ("model_logit", "composition_flag_sum")
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_DRAWS = 2000
LOOKS: list[str] = []


def record_look(label: str) -> None:
    LOOKS.append(label)


def _normalize_surface(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if value in GRASS_SURFACES:
        return "grass"
    if value in TURF_SURFACES:
        return "turf"
    return None


def _mode_from_counts(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def point_in_time_surface_switch_flags(schedules: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "season", "game_type", "gameday", "home_team", "away_team", "surface"}
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for surface-switch tracking: {', '.join(missing)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = reg["home_team"].astype(str).replace(TEAM_ABBREVIATION_ALIASES)
    reg["away_team"] = reg["away_team"].astype(str).replace(TEAM_ABBREVIATION_ALIASES)
    reg["season"] = pd.to_numeric(reg["season"], errors="raise").astype(int)
    reg["gameday"] = pd.to_datetime(reg["gameday"], errors="coerce")
    reg["surface_norm"] = reg["surface"].map(_normalize_surface)
    reg = reg.dropna(subset=["gameday"]).reset_index(drop=True)

    history = reg.sort_values(["home_team", "gameday"]).reset_index(drop=True)
    modal_after: dict[int, str | None] = {}
    for _team, group in history.groupby("home_team", sort=False):
        counts: dict[str, int] = {}
        for row_index, surface in zip(group.index, group["surface_norm"], strict=True):
            modal_after[row_index] = _mode_from_counts(counts)
            if surface is not None:
                counts[surface] = counts.get(surface, 0) + 1
    history["home_modal_surface_after"] = pd.Series(modal_after)

    history_by_team = (
        history[["home_team", "gameday", "home_modal_surface_after"]]
        .rename(columns={"home_team": "team"})
        .sort_values("gameday")
        .reset_index(drop=True)
    )
    target = (
        reg[["game_id", "away_team", "gameday", "surface_norm"]]
        .rename(columns={"away_team": "team"})
        .sort_values("gameday")
        .reset_index(drop=True)
    )
    matched = pd.merge_asof(
        target,
        history_by_team,
        on="gameday",
        by="team",
        direction="backward",
        allow_exact_matches=False,
    )
    matched[NEW_TERM] = (
        matched["surface_norm"].eq("turf") & matched["home_modal_surface_after"].eq("grass")
    ).astype(float)
    return matched[["game_id", NEW_TERM]].drop_duplicates("game_id").reset_index(drop=True)


def _natural_coefficients(
    beta: np.ndarray, features: tuple[str, ...], means: dict[str, float], stds: dict[str, float]
) -> dict[str, float]:
    natural = {name: float(beta[index + 1] / stds[name]) for index, name in enumerate(features)}
    intercept = float(beta[0])
    for index, name in enumerate(features):
        intercept -= float(beta[index + 1]) * means[name] / stds[name]
    natural["intercept"] = intercept
    return natural


def loso_fit(
    population: pd.DataFrame, features: tuple[str, ...]
) -> tuple[pd.Series, dict[str, dict[str, float]]]:
    out = pd.Series(np.nan, index=population.index, dtype=float)
    fold_betas: dict[str, dict[str, float]] = {}
    for held in sorted(int(value) for value in population["season"].unique()):
        train = population.loc[population["season"].ne(held)]
        test = population.loc[population["season"].eq(held)]
        if train.empty or test.empty:
            continue
        means, stds = _standardise(train, features)
        beta = _fit_logit(
            _design(train, features, means, stds),
            train["home_covered"].astype(float).to_numpy(),
            FIT_RIDGE,
        )
        out.loc[test.index] = _predict_features(test, features, beta, means, stds)
        fold_betas[str(held)] = _natural_coefficients(beta, features, means, stds)
    return out, fold_betas


def _record(correct: pd.Series) -> str:
    wins = int(pd.to_numeric(correct, errors="coerce").fillna(0.0).sum())
    return f"{wins}-{len(correct) - wins}"


def accuracy_grading(population: pd.DataFrame, label: str) -> dict[str, Any]:
    plus_features = (*FIT_FEATURES, NEW_TERM)
    base_p, base_betas = loso_fit(population, FIT_FEATURES)
    record_look(f"{label}_accuracy_base4")
    plus_p, plus_betas = loso_fit(population, plus_features)
    record_look(f"{label}_accuracy_plus5")

    frame = population.copy()
    frame["base_oos_p"] = base_p
    frame["plus_oos_p"] = plus_p
    frame = frame.loc[frame["base_oos_p"].notna() & frame["plus_oos_p"].notna()].reset_index(
        drop=True
    )
    frame["base_pick_home"] = frame["base_oos_p"].ge(0.5)
    frame["plus_pick_home"] = frame["plus_oos_p"].ge(0.5)
    frame["base_correct"] = (
        frame["base_pick_home"].astype(float).eq(frame["home_covered"]).astype(float)
    )
    frame["plus_correct"] = (
        frame["plus_pick_home"].astype(float).eq(frame["home_covered"]).astype(float)
    )
    frame["diff_accuracy"] = frame["plus_correct"] - frame["base_correct"]

    accuracy_cell = cell_stats(
        f"{label}_accuracy_plus_vs_base4", frame, "diff_accuracy", BOOTSTRAP_SEED, BOOTSTRAP_DRAWS
    )
    record_look(f"{label}_accuracy_cell")
    return {
        "label": label,
        "games": len(frame),
        "seasons": sorted(int(value) for value in frame["season"].unique()),
        "base4_record": _record(frame["base_correct"]),
        "plus5_record": _record(frame["plus_correct"]),
        "base_fold_betas": base_betas,
        "plus_fold_betas": plus_betas,
        "accuracy_cell": accuracy_cell,
        "flag_coverage_by_season": {
            str(season): int(rows[NEW_TERM].sum()) for season, rows in frame.groupby("season")
        },
    }


def _line_move_report(
    frame: pd.DataFrame, base_pick_col: str, plus_pick_col: str, label: str
) -> dict[str, Any]:
    work = frame.copy()
    work["lm_base"] = np.where(work[base_pick_col], 1.0, -1.0) * work["open_move"]
    work["lm_plus"] = np.where(work[plus_pick_col], 1.0, -1.0) * work["open_move"]
    work["diff_line_move"] = work["lm_plus"] - work["lm_base"]
    cell = cell_stats(label, work, "diff_line_move", BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
    record_look(label)
    return {"games": len(work), "cell": cell}


def line_move_grading_population_a(population: pd.DataFrame) -> dict[str, Any]:
    frame = population.loc[population["open_move"].notna()].reset_index(drop=True)
    plus_features = (*FIT_FEATURES, NEW_TERM)
    base4_p, _ = loso_fit(frame, FIT_FEATURES)
    plus5_p, _ = loso_fit(frame, plus_features)
    opener_base_p, _ = loso_fit(frame, OPENER_ONLY_BASE)
    opener_plus_p, _ = loso_fit(frame, (*OPENER_ONLY_BASE, NEW_TERM))
    frame["base4_pick_home"] = base4_p.ge(0.5)
    frame["plus5_pick_home"] = plus5_p.ge(0.5)
    frame["opener_base_pick_home"] = opener_base_p.ge(0.5)
    frame["opener_plus_pick_home"] = opener_plus_p.ge(0.5)
    ready = frame.loc[
        frame["base4_pick_home"].notna()
        & frame["plus5_pick_home"].notna()
        & frame["opener_base_pick_home"].notna()
        & frame["opener_plus_pick_home"].notna()
    ].reset_index(drop=True)
    return {
        "games": len(ready),
        "seasons": sorted(int(value) for value in ready["season"].unique()),
        "four_term_based": _line_move_report(
            ready,
            "base4_pick_home",
            "plus5_pick_home",
            "population_a_line_move_four_term_based",
        ),
        "opener_only_uncontaminated": _line_move_report(
            ready,
            "opener_base_pick_home",
            "opener_plus_pick_home",
            "population_a_line_move_opener_only_uncontaminated",
        ),
    }


def _attach_line_move_fields(extended: pd.DataFrame) -> pd.DataFrame:
    extended = extended.copy()
    extended["game_id"] = extended["game_id"].astype(str)

    teams = pd.read_parquet(FEATURES_PATH, columns=["game_id", "home_team", "away_team"])
    teams["game_id"] = teams["game_id"].astype(str)
    teams = teams.drop_duplicates("game_id")
    extended = extended.merge(teams, on="game_id", how="left")

    true_pop, _provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    true_pop = true_pop.copy()
    true_pop["game_id"] = true_pop["game_id"].astype(str)
    true_join = true_pop[["game_id", "gameday", "tue_open_home_spread", "open_move"]]

    sbr = pd.read_parquet(
        SBR_ODDS_PATH,
        columns=["game_id", "season", "game_date", "open_home_spread", "close_home_spread"],
    )
    sbr = sbr.loc[sbr["game_id"].notna()].copy()
    sbr["game_id"] = sbr["game_id"].astype(str)
    sbr = sbr.dropna(subset=["open_home_spread", "close_home_spread"])
    sbr = sbr.loc[sbr["season"].between(SBR_SEASON_START, SBR_SEASON_END)]
    sbr = sbr.drop_duplicates(subset="game_id")
    sbr["sbr_open_move"] = sbr["close_home_spread"] - sbr["open_home_spread"]
    sbr_join = sbr.rename(columns={"game_date": "gameday_sbr"})[
        ["game_id", "gameday_sbr", "open_home_spread", "sbr_open_move"]
    ]

    extended = extended.merge(true_join, on="game_id", how="left")
    extended = extended.merge(sbr_join, on="game_id", how="left")
    extended["gameday"] = extended["gameday"].fillna(extended["gameday_sbr"])
    extended["open_move"] = extended["open_move"].fillna(extended["sbr_open_move"])
    extended = extended.drop(
        columns=["gameday_sbr", "tue_open_home_spread", "open_home_spread", "sbr_open_move"]
    )
    return extended


def line_move_grading_population_b(population: pd.DataFrame) -> dict[str, Any]:
    joined = _attach_line_move_fields(population)
    frame = joined.loc[joined["open_move"].notna()].reset_index(drop=True)
    plus_features = (*FIT_FEATURES, NEW_TERM)
    base4_p, _ = loso_fit(frame, FIT_FEATURES)
    plus5_p, _ = loso_fit(frame, plus_features)
    opener_base_p, _ = loso_fit(frame, OPENER_ONLY_BASE)
    opener_plus_p, _ = loso_fit(frame, (*OPENER_ONLY_BASE, NEW_TERM))
    frame["base4_pick_home"] = base4_p.ge(0.5)
    frame["plus5_pick_home"] = plus5_p.ge(0.5)
    frame["opener_base_pick_home"] = opener_base_p.ge(0.5)
    frame["opener_plus_pick_home"] = opener_plus_p.ge(0.5)
    ready = frame.loc[
        frame["base4_pick_home"].notna()
        & frame["plus5_pick_home"].notna()
        & frame["opener_base_pick_home"].notna()
        & frame["opener_plus_pick_home"].notna()
    ].reset_index(drop=True)
    return {
        "games": len(ready),
        "seasons": sorted(int(value) for value in ready["season"].unique()),
        "excluded_no_open_move_source": int(joined["open_move"].isna().sum()),
        "four_term_based": _line_move_report(
            ready,
            "base4_pick_home",
            "plus5_pick_home",
            "population_b_line_move_four_term_based",
        ),
        "opener_only_uncontaminated": _line_move_report(
            ready,
            "opener_base_pick_home",
            "opener_plus_pick_home",
            "population_b_line_move_opener_only_uncontaminated",
        ),
    }


def main() -> int:
    schedules, _team_stats = load_snapshot(latest_snapshot(REPO / "data" / "raw"))
    flag_table = point_in_time_surface_switch_flags(schedules)
    record_look("point_in_time_surface_switch_flag_builder")

    population_a, provenance_a = build_fit_population(REPO / "artifacts", REPO / "data")
    population_a["game_id"] = population_a["game_id"].astype(str)
    population_a = population_a.merge(flag_table, on="game_id", how="left")
    population_a[NEW_TERM] = population_a[NEW_TERM].fillna(0.0)

    population_b = pd.read_parquet(EXTENDED_POPULATION_PATH)
    population_b["game_id"] = population_b["game_id"].astype(str)
    population_b = population_b.merge(flag_table, on="game_id", how="left")
    population_b[NEW_TERM] = population_b[NEW_TERM].fillna(0.0)

    accuracy_a = accuracy_grading(population_a, "population_a_served_2020_2025")
    accuracy_b = accuracy_grading(population_b, "population_b_extended_2011_2025")
    line_move_a = line_move_grading_population_a(population_a)
    line_move_b = line_move_grading_population_b(population_b)

    full_coverage = flag_table.loc[flag_table[NEW_TERM].gt(0.0)]
    coverage_all_history = len(full_coverage)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts" / "surface_switch_fitted_term" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "command": "python scripts/surface_switch_fitted_term.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "new_term": NEW_TERM,
        "base_features": list(FIT_FEATURES),
        "plus_features": [*FIT_FEATURES, NEW_TERM],
        "opener_only_base_features": list(OPENER_ONLY_BASE),
        "ridge": FIT_RIDGE,
        "seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "flag_games_ge1_all_reg_history_2009_2026_schedule_snapshot": coverage_all_history,
        "population_a_provenance": provenance_a,
        "population_a_accuracy": accuracy_a,
        "population_b_accuracy": accuracy_b,
        "population_a_line_move": line_move_a,
        "population_b_line_move": line_move_b,
        "look_count": len(LOOKS),
        "look_log": LOOKS,
        "line_move_contamination_note": (
            "market_move_toward_home / market_move_available are built from post-opener "
            "sharp-book movement; the four_term_based line-move cells compare two variants "
            "that BOTH already carry that information (base4 vs plus5), so the diff isolates "
            "the added surface term rather than re-deriving the served-vs-naive-baseline "
            "artifact flagged CONTAMINATED in artifacts/line_move_yardstick. The "
            "opener_only_uncontaminated cells drop the market-move terms entirely as a "
            "cross-check, mirroring scripts/line_move_yardstick_paired_eval.py."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(f"artifact_dir={out_dir}")
    print("flag_games_all_history", coverage_all_history)
    print(
        "population_a_records",
        accuracy_a["base4_record"],
        "->",
        accuracy_a["plus5_record"],
    )
    print(
        "population_b_records",
        accuracy_b["base4_record"],
        "->",
        accuracy_b["plus5_record"],
    )
    print("population_a_accuracy_cell", json.dumps(accuracy_a["accuracy_cell"]))
    print("population_b_accuracy_cell", json.dumps(accuracy_b["accuracy_cell"]))
    print(
        "population_a_line_move_four_term",
        json.dumps(line_move_a["four_term_based"]["cell"]),
    )
    print(
        "population_b_line_move_four_term",
        json.dumps(line_move_b["four_term_based"]["cell"]),
    )
    print("look_count", len(LOOKS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
