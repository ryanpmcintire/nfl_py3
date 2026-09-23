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
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import pooled_signal_second_fit as base_fit  # noqa: E402

BASE_TERMS = ("model_logit", "market_move_toward_home", "market_move_available")
SHARED_TERM = "composition_flag_sum"
DEVIATION_TERMS = (
    "flag_coach",
    "flag_division",
    "flag_arrests",
    "flag_bye",
    "flag_cold_visitor",
    "flag_protection",
    "flag_tank_zone",
)
ALL_TERMS = BASE_TERMS + (SHARED_TERM,) + DEVIATION_TERMS
KAPPA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 10000.0)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260821
FIT_ITERATIONS = 50


def ridge_vector(ridge_base, kappa):
    penalties = [0.0]
    for name in ALL_TERMS:
        penalties.append(ridge_base * kappa if name in DEVIATION_TERMS else ridge_base)
    return np.asarray(penalties, dtype=float)


def fit_logit_grouped(design_matrix, target, penalties, iterations):
    beta = np.zeros(design_matrix.shape[1])
    diag = np.diag(penalties)
    for _ in range(iterations):
        linear = np.clip(design_matrix @ beta, -35.0, 35.0)
        prob = 1.0 / (1.0 + np.exp(-linear))
        weight = np.clip(prob * (1.0 - prob), 1e-6, None)
        gradient = design_matrix.T @ (target - prob) - penalties * beta
        hessian = (design_matrix.T * weight) @ design_matrix + diag
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta = beta + step
    return beta


def fit_hierarchical(train, kappa, ridge_base):
    means, stds = base_fit.standardisers(train, list(ALL_TERMS))
    design = base_fit.design_matrix(train, list(ALL_TERMS), means, stds)
    penalties = ridge_vector(ridge_base, kappa)
    beta = fit_logit_grouped(design, train["home_covered"].astype(float).to_numpy(), penalties, FIT_ITERATIONS)
    return beta, means, stds


def inner_loso_logloss(train, kappa, ridge_base):
    inner_seasons = sorted(int(v) for v in train["season"].unique())
    losses = []
    for held in inner_seasons:
        inner_train = train.loc[train["season"].ne(held)]
        inner_test = train.loc[train["season"].eq(held)]
        if inner_train.empty or inner_test.empty:
            continue
        beta, means, stds = fit_hierarchical(inner_train, kappa, ridge_base)
        probs = base_fit.predict_proba(inner_test, list(ALL_TERMS), beta, means, stds)
        losses.append(base_fit.logloss(probs, inner_test["home_covered"]))
    return float(np.mean(losses)) if losses else float("inf")


def select_kappa(train, ridge_base, grid):
    scored = {kappa: inner_loso_logloss(train, kappa, ridge_base) for kappa in grid}
    best = min(scored, key=scored.get)
    return best, scored


def natural_from_beta(beta, means, stds):
    natural = {name: float(beta[i + 1] / stds[name]) for i, name in enumerate(ALL_TERMS)}
    intercept = float(beta[0])
    for i, name in enumerate(ALL_TERMS):
        intercept -= float(beta[i + 1]) * means[name] / stds[name]
    natural["intercept"] = intercept
    return natural


def outer_loso(population, ridge_base, grid):
    out = pd.Series(np.nan, index=population.index, dtype=float)
    fold_report = {}
    seasons = sorted(int(v) for v in population["season"].unique())
    for held in seasons:
        train = population.loc[population["season"].ne(held)]
        test = population.loc[population["season"].eq(held)]
        if train.empty or test.empty:
            continue
        kappa, inner_scores = select_kappa(train, ridge_base, grid)
        beta, means, stds = fit_hierarchical(train, kappa, ridge_base)
        out.loc[test.index] = base_fit.predict_proba(test, list(ALL_TERMS), beta, means, stds)
        natural = natural_from_beta(beta, means, stds)
        fold_report[str(held)] = {
            "kappa": kappa,
            "inner_logloss_by_kappa": {str(k): v for k, v in inner_scores.items()},
            "mu_shared": natural[SHARED_TERM],
            "deviations": {name: natural[name] for name in DEVIATION_TERMS},
            "effective_flag_weight": {
                name: natural[SHARED_TERM] + natural[name] for name in DEVIATION_TERMS
            },
            "base_terms": {name: natural[name] for name in BASE_TERMS},
            "intercept": natural["intercept"],
        }
    return out, fold_report


def in_sample_fit(population, ridge_base, grid):
    kappa, inner_scores = select_kappa(population, ridge_base, grid)
    beta, means, stds = fit_hierarchical(population, kappa, ridge_base)
    probs = base_fit.predict_proba(population, list(ALL_TERMS), beta, means, stds)
    natural = natural_from_beta(beta, means, stds)
    return probs, kappa, inner_scores, natural


def reliability_table(probs, truth, bins):
    frame = pd.DataFrame({"prob": np.asarray(probs, dtype=float), "truth": truth.to_numpy(dtype=float)})
    frame["bin"] = pd.qcut(frame["prob"], bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        predicted_mean=("prob", "mean"), observed_rate=("truth", "mean"), games=("truth", "size")
    )
    return [
        {
            "bin": str(idx),
            "predicted_mean": float(row["predicted_mean"]),
            "observed_rate": float(row["observed_rate"]),
            "games": int(row["games"]),
        }
        for idx, row in grouped.iterrows()
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--out", type=str, default="tests/scratch/pooled_signal_fourth_fit.json")
    args = parser.parse_args(argv)

    from nfl_ats.pick_probability_fit import FIT_RIDGE, build_fit_population

    population, provenance = build_fit_population(REPO / "artifacts", REPO / "data")

    base_oos, base_folds, base_insample, _base_natural = base_fit.loso_fit(
        population, list(base_fit.BASE_FEATURES), FIT_RIDGE, FIT_ITERATIONS
    )

    hier_oos, hier_folds = outer_loso(population, FIT_RIDGE, KAPPA_GRID)
    hier_insample, is_kappa, is_inner_scores, is_natural = in_sample_fit(population, FIT_RIDGE, KAPPA_GRID)

    scored = population.copy()
    scored["base_oos_prob"] = base_oos
    scored["hier_oos_prob"] = hier_oos
    scored = scored.loc[scored["hier_oos_prob"].notna() & scored["base_oos_prob"].notna()].copy()

    scored["hier_correct"] = (
        scored["hier_oos_prob"].ge(0.5).astype(float).eq(scored["home_covered"]).astype(float)
    )
    scored["base_correct"] = (
        scored["base_oos_prob"].ge(0.5).astype(float).eq(scored["home_covered"]).astype(float)
    )
    scored["diff_vs_model_only"] = scored["hier_correct"] - scored["model_correct"]
    scored["diff_vs_four_term"] = scored["hier_correct"] - scored["base_correct"]

    vs_model = base_fit.paired_summary(scored, "diff_vs_model_only")
    vs_four = base_fit.paired_summary(scored, "diff_vs_four_term")

    truth = scored["home_covered"]
    hier_insample_full = pd.Series(hier_insample, index=population.index)
    metrics = {
        "hier_oos_accuracy": base_fit.accuracy(scored["hier_oos_prob"], truth),
        "hier_insample_accuracy": base_fit.accuracy(hier_insample_full, population["home_covered"]),
        "base_oos_accuracy": base_fit.accuracy(scored["base_oos_prob"], truth),
        "base_insample_accuracy": base_fit.accuracy(base_insample, population["home_covered"]),
        "model_only_accuracy": float(scored["model_correct"].mean()),
        "hier_oos_brier": base_fit.brier(scored["hier_oos_prob"], truth),
        "base_oos_brier": base_fit.brier(scored["base_oos_prob"], truth),
        "model_oos_brier": base_fit.brier(scored["model_probability"], truth),
        "hier_oos_logloss": base_fit.logloss(scored["hier_oos_prob"], truth),
        "base_oos_logloss": base_fit.logloss(scored["base_oos_prob"], truth),
        "model_oos_logloss": base_fit.logloss(scored["model_probability"], truth),
    }
    metrics["hier_insample_minus_oos_gap"] = (
        metrics["hier_insample_accuracy"] - metrics["hier_oos_accuracy"]
    )
    metrics["base_insample_minus_oos_gap"] = (
        metrics["base_insample_accuracy"] - metrics["base_oos_accuracy"]
    )

    kappas = [fold["kappa"] for fold in hier_folds.values()]
    deviation_frame = pd.DataFrame({k: v["deviations"] for k, v in hier_folds.items()}).T
    mu_series = pd.Series({k: v["mu_shared"] for k, v in hier_folds.items()})
    stability = {
        "kappa_by_fold": {k: v["kappa"] for k, v in hier_folds.items()},
        "kappa_median": float(np.median(kappas)),
        "kappa_min": float(np.min(kappas)),
        "kappa_max": float(np.max(kappas)),
        "mu_shared": {
            "mean": float(mu_series.mean()),
            "sd": float(mu_series.std(ddof=0)),
            "min": float(mu_series.min()),
            "max": float(mu_series.max()),
        },
        "deviation_stability": {
            name: {
                "mean": float(deviation_frame[name].mean()),
                "sd": float(deviation_frame[name].std(ddof=0)),
                "min": float(deviation_frame[name].min()),
                "max": float(deviation_frame[name].max()),
            }
            for name in DEVIATION_TERMS
        },
    }

    reliability = reliability_table(scored["hier_oos_prob"], truth, 5)

    results = {
        "command": "python scripts/pooled_signal_fourth_fit_hierarchical.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_terms": list(BASE_TERMS),
        "shared_term": SHARED_TERM,
        "deviation_terms": list(DEVIATION_TERMS),
        "kappa_grid": list(KAPPA_GRID),
        "inner_selection_criterion": "mean inner-LOSO log loss over training seasons",
        "looks_count": 2,
        "bootstrap_draws": args.draws,
        "seed": args.seed,
        "blocking": "season-week",
        "sample_games": len(scored),
        "sample_blocks": int(
            (scored["season"].astype(str) + "-w" + scored["week"].astype(str)).nunique()
        ),
        "vs_model_only": vs_model,
        "vs_four_term": vs_four,
        "metrics": metrics,
        "fold_report": hier_folds,
        "base_fold_coefficients": base_folds,
        "coefficient_stability": stability,
        "in_sample_kappa": is_kappa,
        "in_sample_inner_scores": {str(k): v for k, v in is_inner_scores.items()},
        "in_sample_natural_coefficients": is_natural,
        "reliability_table": reliability,
        "population_provenance": provenance,
    }
    out_path = (REPO / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(out_path.relative_to(REPO)).replace("\\", "/"),
                "vs_model_only": vs_model,
                "vs_four_term": vs_four,
                "metrics": metrics,
                "stability": stability,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
