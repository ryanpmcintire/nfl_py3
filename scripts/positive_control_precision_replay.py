from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import positive_control_power as power  # noqa: E402

POPULATION_LABEL = "served_2020_2025"
COEFFICIENTS = (0.0, 1.8)
DRAW_PREFIXES = (400, 2000, 20000)
HISTORICAL_INPUT = REPO / "artifacts" / "lead59_type_trait_bins" / "20260923T205414Z"
OUTPUT_ROOT = REPO / "artifacts" / "positive_control_precision_replay"


def wilson_interval(successes, total, z=1.959963985):
    if total <= 0:
        return None, None
    rate = successes / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def reference_row(reference, control, coefficient):
    key = f"{POPULATION_LABEL}__{control}"
    rows = reference["cells"][key]["grid"]
    matches = [row for row in rows if float(row["coefficient_logit_sd"]) == float(coefficient)]
    if len(matches) != 1:
        raise ValueError(f"reference cell {key} coefficient {coefficient} is not unique")
    return matches[0]


def load_pinned_population(reference):
    (
        population,
        provenance,
        controls,
        control_diagnostics,
        source_metadata,
    ) = power.load_lead59_historical_input(HISTORICAL_INPUT)
    if source_metadata != reference.get("population_input"):
        raise ValueError("pinned historical input hashes do not match the reference")
    if provenance != reference.get("population_provenance_served"):
        raise ValueError("pinned historical provenance does not match the reference")
    for control in power.LEAD59_FIXED_CONTROL_COLUMNS:
        expected = reference["cells"][f"{POPULATION_LABEL}__{control}"]["fixed_control"]
        if control_diagnostics["controls"][control] != expected:
            raise ValueError(f"pinned control metadata does not match reference for {control}")
    return population, controls, control_diagnostics, source_metadata


def validate_reference(reference):
    expected = {
        "bootstrap_block": power.EVALUATOR_BOOTSTRAP_METHOD,
        "bootstrap_draws": DRAW_PREFIXES[0],
        "fit_iterations": power.FIT_ITERATIONS,
        "seed": 20260923,
        "selected_population": POPULATION_LABEL,
        "sims_per_cell": 200,
        "control_family": "lead59_fixed_vectors",
    }
    mismatches = {
        key: {"expected": value, "observed": reference.get(key)}
        for key, value in expected.items()
        if reference.get(key) != value
    }
    if mismatches:
        raise ValueError(f"reference contract mismatch: {json.dumps(mismatches, sort_keys=True)}")
    if tuple(float(value) for value in COEFFICIENTS) != COEFFICIENTS:
        raise ValueError("coefficient declaration is invalid")
    for control in power.LEAD59_FIXED_CONTROL_COLUMNS:
        for coefficient in COEFFICIENTS:
            reference_row(reference, control, coefficient)


def input_freeze(reference_path, population, controls, base_dgp, sources):
    population_columns = ["game_id", "season", "week", "home_covered", *power.BASE_TERMS]
    control_frame = population[["game_id"]].copy()
    for control, values in controls.items():
        control_frame[control] = np.asarray(values, dtype=float)
    dgp_frame = population[["game_id"]].copy()
    dgp_frame["base_probability_dgp"] = np.asarray(base_dgp, dtype=float)
    return {
        "reference_results_path": power.display_path(reference_path),
        "reference_results_sha256": power.file_sha256(reference_path),
        "positive_control_power_path": power.display_path(Path(power.__file__)),
        "positive_control_power_sha256": power.file_sha256(Path(power.__file__)),
        "precision_replay_path": power.display_path(Path(__file__)),
        "precision_replay_sha256": power.file_sha256(Path(__file__)),
        "population_columns": population_columns,
        "population_sha256": power.canonical_frame_sha256(population, population_columns),
        "population_order_sha256": hashlib.sha256(
            "\n".join(population["game_id"].astype(str)).encode("utf-8")
        ).hexdigest(),
        "control_columns": list(power.LEAD59_FIXED_CONTROL_COLUMNS),
        "controls_sha256": power.canonical_frame_sha256(
            control_frame,
            ["game_id", *power.LEAD59_FIXED_CONTROL_COLUMNS],
        ),
        "base_dgp_sha256": power.canonical_frame_sha256(
            dgp_frame,
            ["game_id", "base_probability_dgp"],
        ),
        "sources": sources,
    }


def exact_reference_check(observed, expected):
    fields = (
        "coefficient_logit_sd",
        "sims",
        "completed_sims",
        "failed_simulations",
        "fit_failure_rate",
        "detection_rate_denominator",
        "detection_rate",
        "mean_effect_accuracy_points",
        "median_effect_accuracy_points",
        "fit_diagnostics",
    )
    mismatches = [field for field in fields if observed.get(field) != expected.get(field)]
    if mismatches:
        raise ValueError(
            "400-draw replay does not exactly match the reference fields: " + ", ".join(mismatches)
        )
    return {"matched": True, "fields": list(fields)}


def compact_optimizer_checks(fit_diagnostics):
    output = {}
    for model, diagnostic in fit_diagnostics.items():
        groups = [*diagnostic["folds"].values(), diagnostic["full_fit"]]
        fit_count = sum(group["fit_count"] for group in groups)
        finite_completed = sum(
            group["status_counts"].get("finite_completed", 0) for group in groups
        )
        output[model] = {
            "fit_count": fit_count,
            "finite_completed_count": finite_completed,
            "nonfinite_or_failed_count": fit_count - finite_completed,
            "prediction_nonfinite_count": sum(
                group["prediction_nonfinite_count"] for group in groups
            ),
            "linear_solve_fallback_count": sum(
                group["linear_solve_fallback_count"] for group in groups
            ),
            "solver_failure_count": sum(
                count
                for group in groups
                for label, count in group["solver_failure_counts"].items()
                if label != "None"
            ),
            "max_final_gradient_infinity_norm": max(
                group["final_gradient_infinity_norm"]["max"] for group in groups
            ),
            "max_last_step_infinity_norm": max(
                group["last_step_infinity_norm"]["max"] for group in groups
            ),
        }
    return output


def detection_summaries(replicates):
    summaries = []
    group_columns = ["control", "coefficient_logit_sd", "bootstrap_draws"]
    for keys, group in replicates.groupby(group_columns, sort=True):
        detections = int(group["detected"].sum())
        total = len(group)
        low, high = wilson_interval(detections, total)
        summaries.append(
            {
                "control": keys[0],
                "coefficient_logit_sd": float(keys[1]),
                "bootstrap_draws": int(keys[2]),
                "detections": detections,
                "simulations": total,
                "detection_rate": detections / total,
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )
    return summaries


def disagreement_summaries(replicates):
    summaries = []
    cell_columns = ["control", "coefficient_logit_sd"]
    for keys, group in replicates.groupby(cell_columns, sort=True):
        pivot = group.pivot(index="simulation", columns="bootstrap_draws", values="detected")
        for left, right in combinations(DRAW_PREFIXES, 2):
            left_values = pivot[left].astype(bool)
            right_values = pivot[right].astype(bool)
            left_only = int((left_values & ~right_values).sum())
            right_only = int((~left_values & right_values).sum())
            disagreements = left_only + right_only
            total = len(pivot)
            low, high = wilson_interval(disagreements, total)
            summaries.append(
                {
                    "control": keys[0],
                    "coefficient_logit_sd": float(keys[1]),
                    "left_draws": left,
                    "right_draws": right,
                    "simulations": total,
                    "both_detected": int((left_values & right_values).sum()),
                    "neither_detected": int((~left_values & ~right_values).sum()),
                    "left_only_detected": left_only,
                    "right_only_detected": right_only,
                    "disagreements": disagreements,
                    "disagreement_rate": disagreements / total,
                    "wilson_95_low": low,
                    "wilson_95_high": high,
                }
            )
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--draw-prefixes", default=",".join(str(value) for value in DRAW_PREFIXES))
    args = parser.parse_args(argv)
    prefixes = tuple(int(value) for value in args.draw_prefixes.split(","))
    if prefixes != DRAW_PREFIXES:
        parser.error(f"--draw-prefixes must be {','.join(str(value) for value in DRAW_PREFIXES)}")
    reference_path = args.reference.resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    validate_reference(reference)

    population, controls, control_diagnostics, sources = load_pinned_population(reference)
    dgp_outputs = power.loso_fit_with_diagnostics(
        population,
        list(power.BASE_TERMS),
        power.FIT_RIDGE,
        power.FIT_ITERATIONS,
    )
    base_dgp = dgp_outputs[2]
    expected_dgp = reference["populations"][POPULATION_LABEL]["dgp_fit_diagnostics"]
    if dgp_outputs[4] != expected_dgp:
        raise ValueError("pinned base-DGP fit does not exactly match the reference")
    frozen_inputs = input_freeze(reference_path, population, controls, base_dgp, sources)
    print(json.dumps({"input_freeze": frozen_inputs}, sort_keys=True), flush=True)

    started = time.time()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    staging_root = OUTPUT_ROOT / ".staging" / stamp
    final_root = OUTPUT_ROOT / stamp
    if staging_root.exists() or final_root.exists():
        raise FileExistsError(f"precision replay output already exists: {final_root}")
    staging_root.mkdir(parents=True)
    prediction_directory = staging_root / "predictions"
    prediction_directory.mkdir()

    all_replicates = []
    cell_summaries = {}
    fit_diagnostics = {}
    predictions = {}
    seed = int(reference["seed"])
    sims = int(reference["sims_per_cell"])
    for control in power.LEAD59_FIXED_CONTROL_COLUMNS:
        prevalence = control_diagnostics["controls"][control]["prevalence"]
        fixed_term = controls[control]
        for coefficient in COEFFICIENTS:
            prediction_batches = []
            original_bootstraps = {}
            replicate_rows = []
            cell_id = f"{control}__coef_{coefficient:.2f}"

            def capture(
                simulation,
                scored,
                original_bootstrap,
                *,
                selected_control=control,
                selected_coefficient=coefficient,
                selected_cell=cell_id,
                batches=prediction_batches,
                originals=original_bootstraps,
            ):
                scored.insert(0, "simulation", simulation)
                scored.insert(1, "synthetic_outcome_seed", seed + simulation)
                scored.insert(2, "bootstrap_seed", seed + 900000 + simulation)
                scored.insert(3, "control", selected_control)
                scored.insert(4, "coefficient_logit_sd", selected_coefficient)
                batches.append(scored)
                originals[simulation] = original_bootstrap
                if (simulation + 1) % 20 == 0:
                    print(
                        json.dumps(
                            {
                                "cell": selected_cell,
                                "fit_replicates_completed": simulation + 1,
                                "elapsed_s": round(time.time() - started, 1),
                            }
                        ),
                        flush=True,
                    )

            observed = power.run_cell(
                population,
                base_dgp,
                "binary",
                prevalence,
                coefficient,
                sims,
                DRAW_PREFIXES[0],
                power.FIT_ITERATIONS,
                seed,
                power.EVALUATOR_BOOTSTRAP_METHOD,
                fixed_term,
                capture,
            )
            expected = reference_row(reference, control, coefficient)
            reproduction = exact_reference_check(observed, expected)
            for simulation, scored in enumerate(prediction_batches):
                effects = power.evaluator_matched_bootstrap(
                    scored,
                    "diff_vs_four_term",
                    DRAW_PREFIXES[-1],
                    seed + 900000 + simulation,
                )
                point = float(scored["diff_vs_four_term"].mean())
                for draws in DRAW_PREFIXES:
                    prefix = effects[:draws]
                    low, high = np.quantile(prefix, [0.025, 0.975])
                    probability_positive = float(
                        (prefix > 0.0).mean() + 0.5 * (prefix == 0.0).mean()
                    )
                    if draws == DRAW_PREFIXES[0]:
                        prefix_check = {
                            "draws": draws,
                            "point": point,
                            "low": float(low),
                            "high": float(high),
                            "probability_positive": probability_positive,
                            "detected": bool(low > 0.0),
                        }
                        if prefix_check != original_bootstraps[simulation]:
                            raise ValueError(
                                f"400-draw stream is not an exact prefix for {cell_id} "
                                f"simulation {simulation}"
                            )
                    replicate_rows.append(
                        {
                            "control": control,
                            "coefficient_logit_sd": coefficient,
                            "simulation": simulation,
                            "synthetic_outcome_seed": seed + simulation,
                            "bootstrap_seed": seed + 900000 + simulation,
                            "bootstrap_draws": draws,
                            "point_effect_accuracy_points": point * 100.0,
                            "interval_95_low_accuracy_points": float(low) * 100.0,
                            "interval_95_high_accuracy_points": float(high) * 100.0,
                            "probability_positive": probability_positive,
                            "detected": bool(low > 0.0),
                        }
                    )
                if (simulation + 1) % 20 == 0:
                    print(
                        json.dumps(
                            {
                                "cell": cell_id,
                                "precision_replicates_completed": simulation + 1,
                                "elapsed_s": round(time.time() - started, 1),
                            }
                        ),
                        flush=True,
                    )
            reproduction["per_replicate_400_prefix_matched"] = True
            replicate_frame = pd.DataFrame(replicate_rows)
            first_prefix = replicate_frame.loc[
                replicate_frame["bootstrap_draws"].eq(DRAW_PREFIXES[0])
            ]
            if int(first_prefix["detected"].sum()) != round(observed["detection_rate"] * sims):
                raise ValueError(f"400-draw prefix classifications disagree for {cell_id}")
            if not np.isclose(
                first_prefix["point_effect_accuracy_points"].mean(),
                observed["mean_effect_accuracy_points"],
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"prediction rows do not reproduce mean effect for {cell_id}")
            prediction_frame = pd.concat(prediction_batches, ignore_index=True)
            prediction_name = f"{cell_id.replace('.', 'p')}.parquet"
            prediction_path = prediction_directory / prediction_name
            prediction_frame.to_parquet(prediction_path, index=False)
            predictions[cell_id] = {
                "path": f"predictions/{prediction_name}",
                "sha256": power.file_sha256(prediction_path),
                "rows": len(prediction_frame),
            }
            all_replicates.extend(replicate_rows)
            fit_diagnostics[cell_id] = observed["fit_diagnostics"]
            cell_summaries[cell_id] = {
                "control": control,
                "coefficient_logit_sd": coefficient,
                "reference_400_reproduction": reproduction,
                "optimizer_checks": compact_optimizer_checks(observed["fit_diagnostics"]),
                "prediction_output": predictions[cell_id],
            }
            print(
                json.dumps(
                    {
                        "cell": cell_id,
                        "reference_400_matched": True,
                        "elapsed_s": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )

    replicates = pd.DataFrame(all_replicates)
    replicate_path = staging_root / "replicates.parquet"
    replicates.to_parquet(replicate_path, index=False)
    fit_diagnostics_path = staging_root / "fit_diagnostics.json"
    fit_diagnostics_path.write_text(
        json.dumps(fit_diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    detection = detection_summaries(replicates)
    disagreements = disagreement_summaries(replicates)
    if len(detection) != 12 or len(disagreements) != 12:
        raise ValueError("reported diagnostic look counts do not match the predeclaration")
    results = {
        "schema_version": 1,
        "command": "python scripts/positive_control_precision_replay.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "predeclaration": "docs/lanes/positive-control-power.md",
        "reference": power.display_path(reference_path),
        "population": POPULATION_LABEL,
        "controls": list(power.LEAD59_FIXED_CONTROL_COLUMNS),
        "coefficients_logit_sd": list(COEFFICIENTS),
        "draw_prefixes": list(DRAW_PREFIXES),
        "simulations_per_cell": sims,
        "seed": seed,
        "fit_iterations": power.FIT_ITERATIONS,
        "bootstrap_method": power.EVALUATOR_BOOTSTRAP_METHOD,
        "reported_diagnostic_looks": {
            "detection_configurations": 12,
            "paired_disagreement_summaries": 12,
            "total": 24,
            "reference_400_reproduction_checks_are_selected_findings": False,
        },
        "input_freeze": frozen_inputs,
        "fixed_control_diagnostics": control_diagnostics,
        "cells": cell_summaries,
        "detection_estimates": detection,
        "paired_classification_disagreements": disagreements,
        "replicates": {
            "path": "replicates.parquet",
            "sha256": power.file_sha256(replicate_path),
            "rows": len(replicates),
        },
        "fit_diagnostics": {
            "path": "fit_diagnostics.json",
            "sha256": power.file_sha256(fit_diagnostics_path),
        },
        "predictions": predictions,
        "interpretation_contract": {
            "zero_crossing_closes_signal": False,
            "one_fitted_probability_selects_side": True,
            "model_selection_or_tuning": False,
            "may_promote_close_enroll_or_serve": False,
        },
        "elapsed_s": round(time.time() - started, 1),
    }
    result_path = staging_root / "results.json"
    result_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    staging_root.replace(final_root)
    print(
        json.dumps(
            {
                "artifact": power.display_path(final_root / "results.json"),
                "elapsed_s": results["elapsed_s"],
                "reported_diagnostic_looks": 24,
                "reference_400_cells_matched": 4,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
