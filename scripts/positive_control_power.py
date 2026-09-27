import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import lead59_type_trait_bins as lead59  # noqa: E402
import pooled_signal_second_fit as base_fit  # noqa: E402

import nfl_ats.experiment_runner as experiment_runner  # noqa: E402
from nfl_ats.pick_probability_fit import (  # noqa: E402
    FIT_FEATURES,
    FIT_ITERATIONS,
    FIT_RIDGE,
    build_fit_population,
)

BASE_TERMS = base_fit.BASE_FEATURES
DEFAULT_FIT_ITERATIONS = 25
DEFAULT_SIMS = 200
DEFAULT_DRAWS = 400
DEFAULT_GRID = (0.10, 0.25, 0.45, 0.70, 1.00, 1.40, 1.80)
TARGET_POWER = 0.80
EVALUATOR_BOOTSTRAP_METHOD = "evaluator_season_week"
LEAD59_FIXED_CONTROL_COLUMNS = (
    "dpi_tilt_pass_heavy_favorite",
    "holding_tilt_run_heavy",
)
PREVALENCE_CASES = (
    ("binary_p03", "binary", 0.03),
    ("binary_p10", "binary", 0.10),
    ("binary_p50", "binary", 0.50),
    ("continuous_std", "continuous", None),
)
EXTENDED_POPULATION_PATH = (
    REPO / "artifacts" / "extended_fit_population" / "20260923T205910Z" / "population.parquet"
)


def evaluator_matched_bootstrap(frame, value_col, draws, seed):
    blocks = []
    for season in sorted(frame.season.unique()):
        weeks = []
        for week in sorted(frame.loc[frame.season == season, "week"].unique()):
            mask = ((frame.season == season) & (frame.week == week)).to_numpy()
            values = frame.loc[mask, value_col].to_numpy()
            weeks.append(np.asarray([values.sum(), mask.sum()], dtype=float))
        blocks.append(np.asarray(weeks, dtype=float))
    rng = np.random.default_rng(seed)
    effects = np.zeros(draws)
    for draw in range(draws):
        total = np.zeros(2)
        for index in rng.integers(0, len(blocks), size=len(blocks)):
            block = blocks[index]
            total += block[rng.integers(0, len(block), size=len(block))].sum(axis=0)
        effects[draw] = total[0] / total[1]
    return effects


def block_bootstrap(frame, value_col, draws, seed, method):
    point = float(frame[value_col].mean())
    if method == EVALUATOR_BOOTSTRAP_METHOD:
        effects = evaluator_matched_bootstrap(frame, value_col, draws, seed)
    else:
        rng = np.random.default_rng(seed)
        group_columns = ["season"] if method == "season" else ["season", "week"]
        grouped = [
            group[value_col].to_numpy()
            for _, group in frame.groupby(group_columns, sort=True, dropna=False)
        ]
        n = len(grouped)
        if n < 2:
            probability_positive = float(point > 0.0) + 0.5 * float(point == 0.0)
            return point, point, point, probability_positive
        effects = np.empty(draws)
        for draw in range(draws):
            picked = rng.integers(0, n, size=n)
            pooled = np.concatenate([grouped[i] for i in picked])
            effects[draw] = pooled.mean()
    low, high = np.quantile(effects, [0.025, 0.975])
    probability_positive = float((effects > 0.0).mean() + 0.5 * (effects == 0.0).mean())
    return point, float(low), float(high), probability_positive


def synth_term(rng, n, kind, prevalence):
    if kind == "binary":
        return rng.binomial(1, prevalence, size=n).astype(float)
    return rng.standard_normal(n)


def standardize(values):
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0)) or 1.0
    return (values - mean) / std, mean, std


def optimizer_snapshot(design_matrix, target, beta, ridge):
    linear = np.clip(design_matrix @ beta, -35.0, 35.0)
    probability = 1.0 / (1.0 + np.exp(-linear))
    objective = -np.sum(
        target * np.log(probability) + (1.0 - target) * np.log1p(-probability)
    ) + 0.5 * ridge * float(beta @ beta)
    gradient = design_matrix.T @ (target - probability) - ridge * beta
    return float(objective), float(np.linalg.norm(gradient, ord=np.inf))


def fit_logit_with_diagnostics(design_matrix, target, ridge, iterations):
    beta = np.zeros(design_matrix.shape[1], dtype=float)
    identity = np.eye(design_matrix.shape[1], dtype=float)
    initial_objective, initial_gradient_norm = optimizer_snapshot(
        design_matrix, target, beta, ridge
    )
    fallback_count = 0
    completed_iterations = 0
    last_step_norm = None
    solver_failure = None
    for _ in range(iterations):
        linear = np.clip(design_matrix @ beta, -35.0, 35.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weight = np.clip(probability * (1.0 - probability), 1e-6, None)
        gradient = design_matrix.T @ (target - probability) - ridge * beta
        hessian = (design_matrix.T * weight) @ design_matrix + ridge * identity
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            fallback_count += 1
            try:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            except np.linalg.LinAlgError as exc:
                solver_failure = type(exc).__name__
                break
        beta = beta + step
        completed_iterations += 1
        last_step_norm = float(np.linalg.norm(step, ord=np.inf))
    final_objective, final_gradient_norm = optimizer_snapshot(design_matrix, target, beta, ridge)
    finite_result = bool(
        np.isfinite(beta).all()
        and np.isfinite(final_objective)
        and np.isfinite(final_gradient_norm)
        and (last_step_norm is None or np.isfinite(last_step_norm))
    )
    if solver_failure is not None:
        status = "linear_solver_failed"
    elif finite_result:
        status = "finite_completed"
    else:
        status = "nonfinite_result"
    diagnostics = {
        "status": status,
        "convergence_status": "not_assessed_fixed_iterations",
        "requested_iterations": iterations,
        "completed_iterations": completed_iterations,
        "linear_solve_fallback_count": fallback_count,
        "solver_failure": solver_failure,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "initial_gradient_infinity_norm": initial_gradient_norm,
        "final_gradient_infinity_norm": final_gradient_norm,
        "last_step_infinity_norm": last_step_norm,
    }
    return beta, diagnostics


def loso_fit_with_diagnostics(frame, feature_names, ridge, iterations):
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    fold_coefficients = {}
    fold_diagnostics = {}
    in_sample = pd.Series(np.nan, index=frame.index, dtype=float)
    for held_season in sorted(frame["season"].unique()):
        train = frame["season"] != held_season
        test = frame["season"] == held_season
        means, stds = base_fit.standardisers(frame.loc[train], feature_names)
        train_matrix = base_fit.design_matrix(frame.loc[train], feature_names, means, stds)
        target = frame.loc[train, "home_covered"].to_numpy(dtype=float)
        beta, optimizer = fit_logit_with_diagnostics(train_matrix, target, ridge, iterations)
        predictions = base_fit.predict_proba(frame.loc[test], feature_names, beta, means, stds)
        out.loc[test] = predictions
        natural = base_fit.natural_coefficients(beta, feature_names, means, stds)
        fold_coefficients[str(int(held_season))] = natural
        fold_diagnostics[str(int(held_season))] = {
            "train_rows": int(train.sum()),
            "evaluation_rows": int(test.sum()),
            "coefficients": natural,
            "prediction_nonfinite_count": int((~np.isfinite(predictions)).sum()),
            "optimizer": optimizer,
        }

    means, stds = base_fit.standardisers(frame, feature_names)
    full_matrix = base_fit.design_matrix(frame, feature_names, means, stds)
    target = frame["home_covered"].to_numpy(dtype=float)
    full_beta, full_optimizer = fit_logit_with_diagnostics(full_matrix, target, ridge, iterations)
    full_predictions = base_fit.predict_proba(frame, feature_names, full_beta, means, stds)
    in_sample.loc[:] = full_predictions
    full_natural = base_fit.natural_coefficients(full_beta, feature_names, means, stds)
    diagnostics = {
        "folds": fold_diagnostics,
        "full_fit": {
            "train_rows": len(frame),
            "evaluation_rows": len(frame),
            "coefficients": full_natural,
            "prediction_nonfinite_count": int((~np.isfinite(full_predictions)).sum()),
            "optimizer": full_optimizer,
        },
    }
    return out, fold_coefficients, in_sample, full_natural, diagnostics


def numeric_summary(values):
    observed = np.asarray([value for value in values if value is not None], dtype=float)
    finite = observed[np.isfinite(observed)]
    summary = {
        "count": len(values),
        "observed_count": len(observed),
        "finite_count": len(finite),
        "nonfinite_count": int(len(observed) - len(finite)),
    }
    if len(finite):
        summary.update(
            {
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite, ddof=0)),
                "min": float(np.min(finite)),
                "median": float(np.median(finite)),
                "max": float(np.max(finite)),
            }
        )
    return summary


def count_values(values):
    return {
        str(value): values.count(value)
        for value in sorted(set(values), key=lambda value: str(value))
    }


def aggregate_fit_records(records):
    optimizers = [record["optimizer"] for record in records]
    coefficient_names = sorted({name for record in records for name in record["coefficients"]})
    return {
        "fit_count": len(records),
        "status_counts": count_values([optimizer["status"] for optimizer in optimizers]),
        "convergence_status_counts": count_values(
            [optimizer["convergence_status"] for optimizer in optimizers]
        ),
        "solver_failure_counts": count_values(
            [optimizer["solver_failure"] for optimizer in optimizers]
        ),
        "prediction_nonfinite_count": int(
            sum(record["prediction_nonfinite_count"] for record in records)
        ),
        "linear_solve_fallback_count": int(
            sum(optimizer["linear_solve_fallback_count"] for optimizer in optimizers)
        ),
        "requested_iterations": numeric_summary(
            [optimizer["requested_iterations"] for optimizer in optimizers]
        ),
        "completed_iterations": numeric_summary(
            [optimizer["completed_iterations"] for optimizer in optimizers]
        ),
        "initial_objective": numeric_summary(
            [optimizer["initial_objective"] for optimizer in optimizers]
        ),
        "final_objective": numeric_summary(
            [optimizer["final_objective"] for optimizer in optimizers]
        ),
        "initial_gradient_infinity_norm": numeric_summary(
            [optimizer["initial_gradient_infinity_norm"] for optimizer in optimizers]
        ),
        "final_gradient_infinity_norm": numeric_summary(
            [optimizer["final_gradient_infinity_norm"] for optimizer in optimizers]
        ),
        "last_step_infinity_norm": numeric_summary(
            [optimizer["last_step_infinity_norm"] for optimizer in optimizers]
        ),
        "coefficients": {
            name: numeric_summary([record["coefficients"].get(name) for record in records])
            for name in coefficient_names
        },
    }


def aggregate_fit_diagnostics(diagnostics):
    fold_names = sorted(
        {fold_name for diagnostic in diagnostics for fold_name in diagnostic["folds"]}
    )
    fold_records = {
        fold_name: [
            diagnostic["folds"][fold_name]
            for diagnostic in diagnostics
            if fold_name in diagnostic["folds"]
        ]
        for fold_name in fold_names
    }
    pooled_folds = [record for records in fold_records.values() for record in records]
    return {
        "simulation_count": len(diagnostics),
        "folds": {
            fold_name: aggregate_fit_records(records) for fold_name, records in fold_records.items()
        },
        "fold_coefficient_stability": aggregate_fit_records(pooled_folds)["coefficients"],
        "full_fit": aggregate_fit_records([diagnostic["full_fit"] for diagnostic in diagnostics]),
    }


def fit_diagnostics_successful(diagnostics):
    records = [*diagnostics["folds"].values(), diagnostics["full_fit"]]
    return all(
        record["optimizer"]["status"] == "finite_completed"
        and record["prediction_nonfinite_count"] == 0
        for record in records
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    return str(value)


def canonical_frame_sha256(frame, columns):
    ordered = frame.sort_values("game_id", kind="stable")
    rows = (
        json.dumps(
            [canonical_scalar(value) for value in row],
            separators=(",", ":"),
        )
        for row in ordered.loc[:, columns].itertuples(index=False, name=None)
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def display_path(path):
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def load_lead59_historical_input(source):
    source = source.resolve()
    summary_path = source / "summary.json"
    per_game_path = source / "per_game.parquet"
    if not summary_path.is_file() or not per_game_path.is_file():
        raise ValueError("historical LEAD59 input requires summary.json and per_game.parquet")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != 1:
        raise ValueError("historical LEAD59 input has unsupported schema_version")
    historical_features = tuple(summary.get("served_features", ()))
    if historical_features != tuple(FIT_FEATURES):
        raise ValueError("historical LEAD59 served_features do not match FIT_FEATURES")
    if historical_features != tuple(BASE_TERMS):
        raise ValueError("historical LEAD59 served_features do not match harness base terms")
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("historical LEAD59 input is missing provenance")
    active_model_id = provenance.get("active_model_id")
    opener_model_id = provenance.get("opener_evaluation_model_id")
    if not active_model_id or active_model_id != opener_model_id:
        raise ValueError("historical LEAD59 model provenance is inconsistent")
    if not provenance.get("opener_evaluation"):
        raise ValueError("historical LEAD59 input is missing opener evaluation provenance")

    frame = pd.read_parquet(per_game_path)
    required = [
        "game_id",
        "season",
        "week",
        "home_covered",
        "model_probability",
        *BASE_TERMS,
        *LEAD59_FIXED_CONTROL_COLUMNS,
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(
            f"historical LEAD59 per-game input is missing columns: {', '.join(missing)}"
        )
    if len(frame) != provenance.get("graded_games"):
        raise ValueError("historical LEAD59 graded-game count does not match provenance")

    game_ids = frame["game_id"].astype("string")
    if game_ids.isna().any() or game_ids.str.strip().eq("").any():
        raise ValueError("historical LEAD59 input contains blank game IDs")
    if game_ids.duplicated().any():
        raise ValueError("historical LEAD59 input contains duplicate game IDs")

    numeric_columns = [
        "season",
        "week",
        "home_covered",
        "model_probability",
        *BASE_TERMS,
        *LEAD59_FIXED_CONTROL_COLUMNS,
    ]
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("historical LEAD59 input contains nonfinite replay values")
    if not numeric["home_covered"].isin((0.0, 1.0)).all():
        raise ValueError("historical LEAD59 outcomes are not binary")
    if not numeric["model_probability"].between(0.0, 1.0, inclusive="neither").all():
        raise ValueError("historical LEAD59 model probabilities are outside (0, 1)")
    for column in LEAD59_FIXED_CONTROL_COLUMNS:
        if not numeric[column].isin((0.0, 1.0)).all():
            raise ValueError(f"historical LEAD59 control {column} is not binary")

    implied_probability = 1.0 / (1.0 + np.exp(-numeric["model_logit"].to_numpy()))
    if not np.allclose(
        implied_probability,
        numeric["model_probability"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("historical LEAD59 model_logit does not match model_probability")

    seasons = sorted(int(value) for value in numeric["season"].unique())
    if seasons != [int(value) for value in provenance.get("seasons", ())]:
        raise ValueError("historical LEAD59 seasons do not match provenance")

    population_columns = ["game_id", "season", "week", "home_covered", *BASE_TERMS]
    population = frame.loc[:, population_columns].copy()
    population["game_id"] = game_ids.to_numpy()
    population.loc[:, numeric_columns[:3] + list(BASE_TERMS)] = numeric.loc[
        :, numeric_columns[:3] + list(BASE_TERMS)
    ]

    controls = {
        column: numeric[column].to_numpy(dtype=float) for column in LEAD59_FIXED_CONTROL_COLUMNS
    }
    control_diagnostics = {
        "source_mode": "historical_per_game",
        "controls": {},
    }
    for label, values in controls.items():
        positive = population.loc[values == 1.0, ["game_id", "season"]]
        positive_ids = sorted(positive["game_id"].astype(str))
        control_diagnostics["controls"][label] = {
            "positive_count": len(positive_ids),
            "prevalence": float(values.mean()),
            "positive_game_ids_sha256": hashlib.sha256(
                "\n".join(positive_ids).encode("utf-8")
            ).hexdigest(),
            "positive_count_by_season": {
                str(int(season)): int(count)
                for season, count in positive.groupby("season").size().items()
            },
        }

    identity_columns = ["game_id", "season", "week", "home_covered"]
    replay_columns = [
        *identity_columns,
        *BASE_TERMS,
        *LEAD59_FIXED_CONTROL_COLUMNS,
    ]
    source_metadata = {
        "mode": "historical_lead59_per_game",
        "artifact_directory": display_path(source),
        "summary_path": display_path(summary_path),
        "per_game_path": display_path(per_game_path),
        "summary_sha256": file_sha256(summary_path),
        "per_game_sha256": file_sha256(per_game_path),
        "game_count": len(frame),
        "seasons": seasons,
        "fit_features": list(FIT_FEATURES),
        "base_terms": list(BASE_TERMS),
        "fixed_control_columns": list(LEAD59_FIXED_CONTROL_COLUMNS),
        "game_key_sha256": hashlib.sha256(
            "\n".join(sorted(game_ids.astype(str))).encode("utf-8")
        ).hexdigest(),
        "population_identity_sha256": canonical_frame_sha256(frame, identity_columns),
        "replay_inputs_sha256": canonical_frame_sha256(frame, replay_columns),
        "model_provenance": {
            "active_model_id": active_model_id,
            "opener_evaluation": provenance["opener_evaluation"],
            "opener_evaluation_model_id": opener_model_id,
            "base_probability_policy": provenance.get("base_probability_policy"),
            "model_probability_source": provenance.get("model_probability_source"),
        },
    }
    return population, provenance, controls, control_diagnostics, source_metadata


def build_lead59_fixed_controls(population):
    merged = experiment_runner._merge_home_pass_rate_quartile(population.copy(), REPO)
    population_ids = population["game_id"].astype(str)
    merged_ids = merged["game_id"].astype(str)
    if population_ids.duplicated().any() or merged_ids.duplicated().any():
        raise ValueError("lead59 fixed-control population contains duplicate game IDs")
    if len(population_ids) != len(merged_ids) or set(population_ids) != set(merged_ids):
        raise ValueError("lead59 pass-rate merge changed the served population")

    dpi_trait, dpi_diagnostics = lead59.build_type_trait_archive_on(
        REPO, experiment_runner._DPI_PENALTY_TYPE
    )
    holding_trait, holding_diagnostics = lead59.build_type_trait_archive_on(
        REPO, experiment_runner._HOLDING_PENALTY_TYPE
    )
    dpi_lookup = dpi_trait.drop_duplicates("game_id").set_index("game_id")["lag_type_quartile"]
    holding_lookup = holding_trait.drop_duplicates("game_id").set_index("game_id")[
        "lag_type_quartile"
    ]
    merged["dpi_lag_quartile"] = merged["game_id"].map(dpi_lookup)
    merged["holding_lag_quartile"] = merged["game_id"].map(holding_lookup)

    home_favorite = merged["tue_open_home_spread"].astype(float) > 0.0
    pass_heavy_top = merged["home_pass_rate_quartile"].eq(lead59.TOP_QUARTILE)
    run_heavy_bottom = merged["home_pass_rate_quartile"].eq(lead59.BOTTOM_QUARTILE)
    flags = {
        "dpi_tilt_pass_heavy_favorite": (
            home_favorite
            & pass_heavy_top
            & merged["dpi_lag_quartile"].eq(lead59.TOP_QUARTILE).fillna(False)
        ),
        "holding_tilt_run_heavy": (
            run_heavy_bottom & merged["holding_lag_quartile"].eq(lead59.TOP_QUARTILE).fillna(False)
        ),
    }
    controls = {}
    diagnostics = {
        "dpi_trait": dpi_diagnostics,
        "holding_trait": holding_diagnostics,
        "controls": {},
    }
    for label, flag in flags.items():
        flag_by_game = pd.Series(flag.to_numpy(dtype=bool), index=merged["game_id"])
        values = population["game_id"].map(flag_by_game).astype(float).to_numpy()
        positive = population.loc[values == 1.0, ["game_id", "season"]].copy()
        positive_ids = sorted(positive["game_id"].astype(str))
        controls[label] = values
        diagnostics["controls"][label] = {
            "positive_count": len(positive_ids),
            "prevalence": float(values.mean()),
            "positive_game_ids_sha256": hashlib.sha256(
                "\n".join(positive_ids).encode("utf-8")
            ).hexdigest(),
            "positive_count_by_season": {
                str(int(season)): int(count)
                for season, count in positive.groupby("season").size().items()
            },
        }
    return controls, diagnostics


def run_cell(
    population,
    base_probability_dgp,
    kind,
    prevalence,
    coef,
    sims,
    draws,
    fit_iterations,
    seed,
    bootstrap_block,
    fixed_term=None,
):
    detections = 0
    points = []
    base_diagnostics = []
    augmented_diagnostics = []
    failed_simulations = []
    epsilon = 1e-6
    base_logit = np.log(
        np.clip(base_probability_dgp, epsilon, 1.0 - epsilon)
        / (1.0 - np.clip(base_probability_dgp, epsilon, 1.0 - epsilon))
    )
    n = len(population)
    fixed_term_std = None
    if fixed_term is not None:
        fixed_term = np.asarray(fixed_term, dtype=float)
        if len(fixed_term) != n:
            raise ValueError("fixed control length does not match population")
        fixed_term_std, _, _ = standardize(fixed_term)
    for sim in range(sims):
        rng = np.random.default_rng(seed + sim)
        if fixed_term is None:
            term_raw = synth_term(rng, n, kind, prevalence)
            term_std, _, _ = standardize(term_raw)
        else:
            term_raw = fixed_term
            term_std = fixed_term_std
        synth_logit = np.clip(base_logit + coef * term_std, -35.0, 35.0)
        synth_prob = 1.0 / (1.0 + np.exp(-synth_logit))
        synth_outcome = rng.binomial(1, synth_prob).astype(float)

        sim_pop = population[["season", "week", *BASE_TERMS]].copy()
        sim_pop["synthetic_term"] = term_raw
        sim_pop["home_covered"] = synth_outcome

        base_outputs = loso_fit_with_diagnostics(
            sim_pop, list(BASE_TERMS), FIT_RIDGE, fit_iterations
        )
        new_outputs = loso_fit_with_diagnostics(
            sim_pop, [*BASE_TERMS, "synthetic_term"], FIT_RIDGE, fit_iterations
        )
        base_oos, base_diagnostic = base_outputs[0], base_outputs[4]
        new_oos, new_diagnostic = new_outputs[0], new_outputs[4]
        base_diagnostics.append(base_diagnostic)
        augmented_diagnostics.append(new_diagnostic)
        if not (
            fit_diagnostics_successful(base_diagnostic)
            and fit_diagnostics_successful(new_diagnostic)
        ):
            failed_simulations.append(sim)
            continue
        mask = base_oos.notna() & new_oos.notna()
        scored = sim_pop.loc[mask].copy()
        scored["base_correct"] = (
            base_oos.loc[mask].ge(0.5).astype(float).eq(scored["home_covered"]).astype(float)
        )
        scored["new_correct"] = (
            new_oos.loc[mask].ge(0.5).astype(float).eq(scored["home_covered"]).astype(float)
        )
        scored["diff_vs_four_term"] = scored["new_correct"] - scored["base_correct"]

        point, low, _high, _pp = block_bootstrap(
            scored,
            "diff_vs_four_term",
            draws,
            seed + 900000 + sim,
            bootstrap_block,
        )
        points.append(point * 100.0)
        if low > 0.0:
            detections += 1

    completed_sims = len(points)
    detection_rate = detections / completed_sims if completed_sims else None
    return {
        "coefficient_logit_sd": coef,
        "sims": sims,
        "completed_sims": completed_sims,
        "failed_simulations": failed_simulations,
        "fit_failure_rate": len(failed_simulations) / sims,
        "detection_rate_denominator": completed_sims,
        "detection_rate": detection_rate,
        "mean_effect_accuracy_points": float(np.mean(points)) if points else None,
        "median_effect_accuracy_points": float(np.median(points)) if points else None,
        "fit_diagnostics": {
            "base": aggregate_fit_diagnostics(base_diagnostics),
            "augmented": aggregate_fit_diagnostics(augmented_diagnostics),
        },
    }


def mde_from_grid(grid_results, target_power):
    if any(
        row["detection_rate"] is None or row["mean_effect_accuracy_points"] is None
        for row in grid_results
    ):
        return None, "not estimable because a grid point had no completed simulations"
    ordered = sorted(grid_results, key=lambda row: row["mean_effect_accuracy_points"])
    if not ordered:
        return None, "no grid points evaluated"
    if ordered[0]["detection_rate"] >= target_power:
        return ordered[0]["mean_effect_accuracy_points"], "below smallest tested effect"
    if ordered[-1]["detection_rate"] < target_power:
        return None, "not reached within tested coefficient range"
    for lower, upper in pairwise(ordered):
        if lower["detection_rate"] < target_power <= upper["detection_rate"]:
            span = upper["detection_rate"] - lower["detection_rate"]
            if span <= 0:
                return upper["mean_effect_accuracy_points"], "interpolated (flat span)"
            frac = (target_power - lower["detection_rate"]) / span
            mde = lower["mean_effect_accuracy_points"] + frac * (
                upper["mean_effect_accuracy_points"] - lower["mean_effect_accuracy_points"]
            )
            return float(mde), "linear interpolation between grid points"
    return None, "power was not monotone in the tested grid"


def population_seasons_and_games(population, label, bootstrap_block):
    block_columns = ["season"] if bootstrap_block == "season" else ["season", "week"]
    return {
        "label": label,
        "games": len(population),
        "seasons": sorted(int(value) for value in population["season"].unique()),
        "season_blocks": int(population["season"].nunique()),
        "bootstrap_block": bootstrap_block,
        "bootstrap_blocks": int(population.groupby(block_columns, dropna=False).ngroups),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--fit-iterations", type=int, default=DEFAULT_FIT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--grid", default=",".join(str(value) for value in DEFAULT_GRID))
    parser.add_argument(
        "--bootstrap-block",
        choices=("season", "week", EVALUATOR_BOOTSTRAP_METHOD),
        default="season",
    )
    parser.add_argument(
        "--population",
        choices=("all", "served_2020_2025", "extended_2011_2025"),
        default="all",
    )
    parser.add_argument(
        "--prevalence-cases",
        default=",".join(label for label, _, _ in PREVALENCE_CASES),
    )
    parser.add_argument("--lead59-fixed-controls", action="store_true")
    parser.add_argument("--lead59-historical-input", type=Path)
    args = parser.parse_args(argv)
    if args.bootstrap_block == EVALUATOR_BOOTSTRAP_METHOD and args.fit_iterations != FIT_ITERATIONS:
        parser.error(
            f"--bootstrap-block {EVALUATOR_BOOTSTRAP_METHOD} requires "
            f"--fit-iterations {FIT_ITERATIONS}"
        )
    grid = tuple(float(value) for value in args.grid.split(","))
    prevalence_by_label = {
        label: (label, kind, prevalence) for label, kind, prevalence in PREVALENCE_CASES
    }
    selected_labels = tuple(
        value.strip() for value in args.prevalence_cases.split(",") if value.strip()
    )
    if not selected_labels:
        parser.error("at least one prevalence case is required")
    if args.lead59_historical_input and not args.lead59_fixed_controls:
        parser.error("--lead59-historical-input requires --lead59-fixed-controls")
    if args.lead59_fixed_controls:
        if args.population != "served_2020_2025":
            parser.error("--lead59-fixed-controls requires --population served_2020_2025")
    else:
        unknown_labels = sorted(set(selected_labels).difference(prevalence_by_label))
        if unknown_labels:
            parser.error(f"unknown prevalence cases: {', '.join(unknown_labels)}")
        selected_prevalence_cases = tuple(prevalence_by_label[label] for label in selected_labels)

    started = time.time()
    fixed_controls = None
    fixed_control_diagnostics = None
    population_input = {"mode": "active_model_pointer"}
    if args.lead59_historical_input:
        try:
            (
                served_population,
                provenance,
                fixed_controls,
                fixed_control_diagnostics,
                population_input,
            ) = load_lead59_historical_input(args.lead59_historical_input)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    else:
        served_population, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    populations = {}
    if args.population in ("all", "served_2020_2025"):
        populations["served_2020_2025"] = served_population
    if args.population in ("all", "extended_2011_2025"):
        populations["extended_2011_2025"] = pd.read_parquet(EXTENDED_POPULATION_PATH)
    if args.lead59_fixed_controls and fixed_controls is None:
        fixed_controls, fixed_control_diagnostics = build_lead59_fixed_controls(served_population)
    if fixed_controls:
        selected_labels = tuple(fixed_controls)

    results = {
        "command": "python scripts/positive_control_power.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "sims_per_cell": args.sims,
        "bootstrap_draws": args.draws,
        "fit_iterations": args.fit_iterations,
        "seed": args.seed,
        "coefficient_grid_logit_sd": list(grid),
        "target_power": TARGET_POWER,
        "optimizer_diagnostic_contract": {
            "objective": "ridge_penalized_negative_log_likelihood",
            "gradient_norm": "infinity_norm_at_final_coefficients",
            "step_norm": "infinity_norm_of_last_newton_step",
            "convergence": "not_assessed_by_fixed_iteration_fitter",
        },
        "base_terms": list(BASE_TERMS),
        "bootstrap_block": args.bootstrap_block,
        "bootstrap_hierarchy": (
            "season_then_week"
            if args.bootstrap_block == EVALUATOR_BOOTSTRAP_METHOD
            else "flat_blocks"
        ),
        "selected_population": args.population,
        "prevalence_cases": list(selected_labels),
        "control_family": "lead59_fixed_vectors" if fixed_controls else "iid",
        "fixed_control_diagnostics": fixed_control_diagnostics,
        "populations": {},
        "population_provenance_served": provenance,
        "population_input": population_input,
        "extended_population_path": str(EXTENDED_POPULATION_PATH.relative_to(REPO)).replace(
            "\\", "/"
        ),
        "cells": {},
        "mde_table": {},
    }

    for pop_label, population in populations.items():
        results["populations"][pop_label] = population_seasons_and_games(
            population, pop_label, args.bootstrap_block
        )
        base_fit_outputs = loso_fit_with_diagnostics(
            population, list(BASE_TERMS), FIT_RIDGE, args.fit_iterations
        )
        base_in_sample = base_fit_outputs[2]
        results["populations"][pop_label]["dgp_fit_diagnostics"] = base_fit_outputs[4]
        if fixed_controls:
            cases = tuple(
                (
                    label,
                    "binary",
                    fixed_control_diagnostics["controls"][label]["prevalence"],
                    values,
                    fixed_control_diagnostics["controls"][label],
                )
                for label, values in fixed_controls.items()
            )
        else:
            cases = tuple(
                (label, kind, prevalence, None, {})
                for label, kind, prevalence in selected_prevalence_cases
            )
        for prev_label, kind, prevalence, fixed_term, control_metadata in cases:
            cell_key = f"{pop_label}__{prev_label}"
            grid_rows = []
            for coef in grid:
                row = run_cell(
                    population,
                    base_in_sample,
                    kind,
                    prevalence,
                    coef,
                    args.sims,
                    args.draws,
                    args.fit_iterations,
                    args.seed,
                    args.bootstrap_block,
                    fixed_term,
                )
                grid_rows.append(row)
                print(
                    json.dumps(
                        {
                            "cell": cell_key,
                            "coef": coef,
                            "detection_rate": row["detection_rate"],
                            "mean_effect_accuracy_points": row["mean_effect_accuracy_points"],
                            "elapsed_s": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
            mde, mde_note = mde_from_grid(grid_rows, TARGET_POWER)
            results["cells"][cell_key] = {
                "population": pop_label,
                "prevalence_case": prev_label,
                "term_kind": kind,
                "prevalence": prevalence,
                "fixed_control": control_metadata or None,
                "grid": grid_rows,
                "minimum_detectable_effect_accuracy_points_at_80pct_power": mde,
                "mde_note": mde_note,
            }
            results["mde_table"][cell_key] = mde

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts" / "positive_control_power" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results["artifact"] = str(out_path.relative_to(REPO)).replace("\\", "/")
    results["elapsed_s"] = round(time.time() - started, 1)
    summary = {
        "artifact": results["artifact"],
        "mde_table": results["mde_table"],
        "elapsed_s": results["elapsed_s"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
