import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _repo_src() -> None:
    path = str(REPO / "src")
    if path not in sys.path:
        sys.path.insert(0, path)


BASE_FEATURES = (
    "model_logit",
    "composition_flag_sum",
    "market_move_toward_home",
    "market_move_available",
)
EXTRA_FEATURES = ("open_corner_wind_dog_tue_flag",)
FULL_FEATURES = BASE_FEATURES + EXTRA_FEATURES
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260821
EPSILON = 1e-6
TUESDAY_WIND_ARCHIVE = Path("data/raw/forecast_archive/full_2020_2025/forecasts.parquet")
TUESDAY_WIND_THRESHOLD_MPH = 15.0
OPEN_CORNER_STADIUM_IDS = frozenset(
    {
        "BUF00",
        "CHI98",
        "BOS00",
        "CLE00",
        "GNB00",
        "PIT00",
        "KAN00",
        "DEN00",
        "NYC01",
        "PHI00",
    }
)
OUTDOOR_ROOFS = frozenset({"outdoors", "open"})


def fit_logit(design_matrix, target, ridge, iterations):
    beta = np.zeros(design_matrix.shape[1])
    eye = np.eye(design_matrix.shape[1])
    for _ in range(iterations):
        linear = np.clip(design_matrix @ beta, -35.0, 35.0)
        prob = 1.0 / (1.0 + np.exp(-linear))
        weight = np.clip(prob * (1.0 - prob), 1e-6, None)
        gradient = design_matrix.T @ (target - prob) - ridge * beta
        hessian = (design_matrix.T * weight) @ design_matrix + ridge * eye
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta = beta + step
    return beta


def design_matrix(frame, features, means, stds):
    columns = [np.ones(len(frame))]
    for name in features:
        columns.append(((frame[name].astype(float) - means[name]) / stds[name]).to_numpy())
    return np.column_stack(columns)


def standardisers(train, features):
    means = {name: float(train[name].mean()) for name in features}
    stds = {name: float(train[name].std(ddof=0)) or 1.0 for name in features}
    return means, stds


def natural_coefficients(beta, features, means, stds):
    natural = {name: float(beta[index + 1] / stds[name]) for index, name in enumerate(features)}
    intercept = float(beta[0])
    for index, name in enumerate(features):
        intercept -= float(beta[index + 1]) * means[name] / stds[name]
    natural["intercept"] = intercept
    return natural


def predict_proba(frame, features, beta, means, stds):
    linear = np.clip(design_matrix(frame, features, means, stds) @ beta, -35.0, 35.0)
    return np.asarray(1.0 / (1.0 + np.exp(-linear)), dtype=float)


def loso_fit(population, features, ridge, iterations):
    out = pd.Series(np.nan, index=population.index, dtype=float)
    fold_coefficients = {}
    seasons = sorted(int(value) for value in population["season"].unique())
    for held in seasons:
        train = population.loc[population["season"].ne(held)]
        test = population.loc[population["season"].eq(held)]
        if train.empty or test.empty:
            continue
        fold_means, fold_stds = standardisers(train, features)
        fold_beta = fit_logit(
            design_matrix(train, features, fold_means, fold_stds),
            train["home_covered"].astype(float).to_numpy(),
            ridge,
            iterations,
        )
        out.loc[test.index] = predict_proba(test, features, fold_beta, fold_means, fold_stds)
        fold_coefficients[str(held)] = natural_coefficients(
            fold_beta, features, fold_means, fold_stds
        )
    full_means, full_stds = standardisers(population, features)
    full_beta = fit_logit(
        design_matrix(population, features, full_means, full_stds),
        population["home_covered"].astype(float).to_numpy(),
        ridge,
        iterations,
    )
    in_sample = predict_proba(population, features, full_beta, full_means, full_stds)
    full_natural = natural_coefficients(full_beta, features, full_means, full_stds)
    return out, fold_coefficients, in_sample, full_natural


def week_block_bootstrap(frame, draws, seed):
    rng = np.random.default_rng(seed)
    blocks = sorted(frame["block"].unique())
    effects = np.empty(draws)
    for draw in range(draws):
        picked = rng.choice(len(blocks), size=len(blocks), replace=True)
        pooled = pd.concat([frame.loc[frame["block"].eq(blocks[index])] for index in picked])
        effects[draw] = pooled["paired_diff"].mean()
    low, high = np.quantile(effects, [0.025, 0.975])
    return float(low), float(high), float((effects > 0).mean())


def accuracy(probs, truth):
    picked_home = pd.Series(np.asarray(probs)).ge(0.5).to_numpy()
    return float((picked_home.astype(float) == truth.to_numpy(dtype=float)).mean())


def brier(probs, truth):
    return float(((np.asarray(probs) - truth.to_numpy(dtype=float)) ** 2).mean())


def logloss(probs, truth):
    clipped = np.clip(np.asarray(probs, dtype=float), EPSILON, 1.0 - EPSILON)
    truth_values = truth.to_numpy(dtype=float)
    return float(
        -(truth_values * np.log(clipped) + (1.0 - truth_values) * np.log(1.0 - clipped)).mean()
    )


def paired_summary(scored, column):
    frame = scored[["season", "week", column]].rename(columns={column: "paired_diff"}).copy()
    effect_points = float(frame["paired_diff"].mean() * 100.0)
    low, high, probability_positive = week_block_bootstrap(
        frame.assign(block=frame["season"].astype(str) + "-w" + frame["week"].astype(str)),
        BOOTSTRAP_DRAWS,
        BOOTSTRAP_SEED,
    )
    decisive = frame.loc[frame["paired_diff"].ne(0.0)]
    wins = int((decisive["paired_diff"] > 0).sum())
    losses = int((decisive["paired_diff"] < 0).sum())
    per_season = (
        frame.groupby("season")["paired_diff"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "diff_fraction", "size": "games"})
    )
    per_season_detail = {
        str(season): {
            "diff_points": float(row["diff_fraction"] * 100.0),
            "games": int(row["games"]),
        }
        for season, row in per_season.iterrows()
    }
    return {
        "effect_accuracy_points": effect_points,
        "interval_low": low * 100.0,
        "interval_high": high * 100.0,
        "probability_positive": probability_positive,
        "decisive_games": len(decisive),
        "decisive_wins": wins,
        "decisive_losses": losses,
        "per_season": per_season_detail,
    }


def build_extra_columns(population, schedules):
    archive = pd.read_parquet(REPO / TUESDAY_WIND_ARCHIVE)
    wind_lookup = archive.drop_duplicates("game_id").set_index(archive["game_id"].astype(str))[
        "forecast_wind_mph"
    ]
    venue = schedules.drop_duplicates("game_id").set_index(schedules["game_id"].astype(str))[
        ["stadium_id", "roof"]
    ]
    enriched = population.copy()
    game_ids = enriched["game_id"].astype(str)
    enriched["tue_stadium_id"] = game_ids.map(venue["stadium_id"].astype(str))
    enriched["tue_roof"] = game_ids.map(venue["roof"].astype(str))
    enriched["tue_wind_mph"] = pd.to_numeric(game_ids.map(wind_lookup), errors="coerce")
    venue_ok = enriched["tue_stadium_id"].astype(str).isin(OPEN_CORNER_STADIUM_IDS)
    roof_ok = enriched["tue_roof"].astype(str).str.lower().isin(OUTDOOR_ROOFS)
    wind_ok = enriched["tue_wind_mph"].ge(TUESDAY_WIND_THRESHOLD_MPH)
    qualifies = (venue_ok & roof_ok & wind_ok).to_numpy()
    spread = pd.to_numeric(enriched["tue_open_home_spread"], errors="coerce").to_numpy()
    flag = np.where(
        qualifies & (spread < 0.0),
        1.0,
        np.where(qualifies & (spread > 0.0), -1.0, 0.0),
    )
    enriched["open_corner_wind_dog_tue_flag"] = pd.Series(flag, index=enriched.index, dtype=float)
    coverage = {}
    for name in EXTRA_FEATURES:
        coverage[name] = {
            "non_null": int(enriched[name].notna().sum()),
            "null_filled_zero": int(enriched[name].isna().sum()),
            "nonzero": int(enriched[name].fillna(0.0).ne(0.0).sum()),
        }
        enriched[name] = enriched[name].fillna(0.0).astype(float)
    coverage["tuesday_wind_join"] = {
        "graded_games": len(enriched),
        "with_tuesday_wind": int(enriched["tue_wind_mph"].notna().sum()),
        "qualifying_games": int(qualifies.sum()),
        "positive_flags": int((enriched["open_corner_wind_dog_tue_flag"] > 0.0).sum()),
        "negative_flags": int((enriched["open_corner_wind_dog_tue_flag"] < 0.0).sum()),
    }
    return enriched, coverage


def main(argv=None):
    global BOOTSTRAP_DRAWS, BOOTSTRAP_SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--out", type=str, default="tests/scratch/pooled_signal_third_fit.json")
    args = parser.parse_args(argv)
    BOOTSTRAP_DRAWS = int(args.draws)
    BOOTSTRAP_SEED = int(args.seed)

    _repo_src()
    from nfl_ats.pick_probability_fit import FIT_ITERATIONS, FIT_RIDGE, build_fit_population
    from nfl_ats.snapshots import latest_snapshot, load_snapshot

    population, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    schedules, _team_stats = load_snapshot(latest_snapshot(REPO / "data" / "raw"))
    enriched, coverage = build_extra_columns(population, schedules)

    base_oos, base_folds, base_insample, _base_full = loso_fit(
        enriched, list(BASE_FEATURES), FIT_RIDGE, FIT_ITERATIONS
    )
    full_oos, full_folds, full_insample, full_natural = loso_fit(
        enriched, list(FULL_FEATURES), FIT_RIDGE, FIT_ITERATIONS
    )
    scored = enriched.copy()
    scored["base_oos_prob"] = base_oos
    scored["full_oos_prob"] = full_oos
    scored = scored.loc[scored["full_oos_prob"].notna()].copy()
    scored["full_correct"] = (
        scored["full_oos_prob"].ge(0.5).astype(float).eq(scored["home_covered"]).astype(float)
    )
    scored["base_correct"] = (
        scored["base_oos_prob"].ge(0.5).astype(float).eq(scored["home_covered"]).astype(float)
    )
    scored["diff_vs_model_only"] = scored["full_correct"] - scored["model_correct"]
    scored["diff_vs_four_term"] = scored["full_correct"] - scored["base_correct"]

    vs_model = paired_summary(scored, "diff_vs_model_only")
    vs_four = paired_summary(scored, "diff_vs_four_term")

    truth = scored["home_covered"]
    metrics = {
        "full_oos_accuracy": accuracy(scored["full_oos_prob"], truth),
        "full_insample_accuracy": accuracy(full_insample, enriched["home_covered"]),
        "base_oos_accuracy": accuracy(scored["base_oos_prob"], truth),
        "base_insample_accuracy": accuracy(base_insample, enriched["home_covered"]),
        "model_only_accuracy": float(scored["model_correct"].mean()),
        "full_oos_brier": brier(scored["full_oos_prob"], truth),
        "base_oos_brier": brier(scored["base_oos_prob"], truth),
        "model_oos_brier": brier(scored["model_probability"], truth),
        "full_oos_logloss": logloss(scored["full_oos_prob"], truth),
        "base_oos_logloss": logloss(scored["base_oos_prob"], truth),
        "model_oos_logloss": logloss(scored["model_probability"], truth),
    }
    metrics["full_insample_minus_oos_gap"] = (
        metrics["full_insample_accuracy"] - metrics["full_oos_accuracy"]
    )
    metrics["base_insample_minus_oos_gap"] = (
        metrics["base_insample_accuracy"] - metrics["base_oos_accuracy"]
    )

    fold_frame = pd.DataFrame(full_folds).T
    stability = {
        name: {
            "mean": float(fold_frame[name].mean()),
            "sd": float(fold_frame[name].std(ddof=0)),
            "min": float(fold_frame[name].min()),
            "max": float(fold_frame[name].max()),
            "full_sample": float(full_natural[name]),
        }
        for name in fold_frame.columns
    }

    results = {
        "command": "python scripts/pooled_signal_third_fit.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_features": list(BASE_FEATURES),
        "extra_features": list(EXTRA_FEATURES),
        "n_extra_columns": len(EXTRA_FEATURES),
        "dropped_candidates": [
            "snow_game_home_prep: predeclared definition needs Tuesday forecast "
            "precip probability at or above 50 percent with forecast temp at or "
            "below 32F, and the Tuesday archive carries wind and temp only with "
            "no precip column, so no faithful Tuesday column exists without a "
            "new archive re-cut",
            "rookie_qb_debut_fade: current builder gates depth rows at the pool "
            "decision cutoff which postdates the Tuesday opener, and strict "
            "before Tuesday gating needs timestamped historical depth rows while "
            "the all-position archive carries no observed-at column and the "
            "quarterback depth archive covers 2026 only, so the 2020-2025 "
            "population is ungateable without new snapshot history",
            "backup_tenure_gap: no derive builder exists in src and the "
            "documented design reads post-hoc schedule starters, while strict "
            "gating needs a latest depth row before the Tuesday opener which the "
            "archives above cannot supply for 2020-2025, so it is ungateable "
            "without new snapshot history",
            "ol_rush_continuity: the library defines it as a weeks 1-4 split "
            "conditioner with high equals all four unit continuities above their "
            "medians and no pick direction, so there is no signed flag to pool "
            "and inventing one would substitute an approximation",
        ],
        "looks_count": 5,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "blocking": "season-week",
        "sample_games": len(scored),
        "sample_blocks": int(
            (scored["season"].astype(str) + "-w" + scored["week"].astype(str)).nunique()
        ),
        "vs_model_only": vs_model,
        "vs_four_term": vs_four,
        "metrics": metrics,
        "fold_coefficients": full_folds,
        "base_fold_coefficients": base_folds,
        "coefficient_stability": stability,
        "extra_column_coverage": coverage,
        "non_commensurable_columns": [
            "open_corner_wind_dog_tue_flag: signed plus or minus 1 dog-side flag "
            "that fires only on qualifying Tuesday-wind games at ten frozen "
            "open-corner venues, about 7 percent of graded games, unlike the "
            "continuous model_logit and market terms",
        ],
        "population_provenance": provenance,
    }
    out_path = (REPO / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(out_path.relative_to(REPO)).replace("\\", "/"),
                "n_extra_columns": len(EXTRA_FEATURES),
                "vs_model_only": vs_model,
                "vs_four_term": vs_four,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
