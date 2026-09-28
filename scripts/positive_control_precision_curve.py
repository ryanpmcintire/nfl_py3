from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import positive_control_power as power  # noqa: E402
import positive_control_precision_replay as replay  # noqa: E402

COEFFICIENTS = (0.1, 0.25, 0.45, 0.7, 1.0, 1.4)
REUSED_COEFFICIENTS = (0.0, 1.8)
DRAW_PREFIXES = (400, 2000, 20000)
OUTPUT_ROOT = REPO / "artifacts" / "positive_control_precision_curve"


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_once(path, value):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {power.display_path(path)}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {power.display_path(temporary)}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_record(base, record, label):
    path = (base / record["path"]).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"{label} escapes its artifact directory")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {power.display_path(path)}")
    if power.file_sha256(path) != record["sha256"]:
        raise ValueError(f"{label} hash mismatch")
    return path


def validate_reused_artifact(path, expected_freeze):
    artifact = read_json(path)
    if artifact.get("schema_version") != 1:
        raise ValueError("reused precision artifact schema mismatch")
    if artifact.get("coefficients_logit_sd") != list(REUSED_COEFFICIENTS):
        raise ValueError("reused precision artifact coefficient mismatch")
    if artifact.get("draw_prefixes") != list(DRAW_PREFIXES):
        raise ValueError("reused precision artifact draw-prefix mismatch")
    if artifact.get("simulations_per_cell") != 200:
        raise ValueError("reused precision artifact simulation-count mismatch")
    looks = artifact.get("reported_diagnostic_looks", {})
    if (
        looks.get("detection_configurations") != 12
        or looks.get("paired_disagreement_summaries") != 12
    ):
        raise ValueError("reused precision artifact look-count mismatch")
    if len(artifact.get("detection_estimates", [])) != 12:
        raise ValueError("reused precision artifact detection-summary mismatch")
    if len(artifact.get("paired_classification_disagreements", [])) != 12:
        raise ValueError("reused precision artifact disagreement-summary mismatch")
    frozen_keys = (
        "reference_results_path",
        "reference_results_sha256",
        "positive_control_power_path",
        "positive_control_power_sha256",
        "precision_replay_path",
        "precision_replay_sha256",
        "population_columns",
        "population_sha256",
        "population_order_sha256",
        "control_columns",
        "controls_sha256",
        "base_dgp_sha256",
        "sources",
    )
    observed_freeze = artifact.get("input_freeze", {})
    mismatches = [
        key for key in frozen_keys if observed_freeze.get(key) != expected_freeze.get(key)
    ]
    if mismatches:
        raise ValueError("reused precision artifact input mismatch: " + ", ".join(mismatches))
    base = path.parent
    verified = {
        "results": {"path": power.display_path(path), "sha256": power.file_sha256(path)},
        "replicates": artifact["replicates"],
        "fit_diagnostics": artifact["fit_diagnostics"],
        "predictions": artifact["predictions"],
    }
    verify_record(base, artifact["replicates"], "reused replicates")
    verify_record(base, artifact["fit_diagnostics"], "reused fit diagnostics")
    expected_cells = {
        f"{control}__coef_{coefficient:.2f}"
        for control in power.LEAD59_FIXED_CONTROL_COLUMNS
        for coefficient in REUSED_COEFFICIENTS
    }
    if set(artifact.get("cells", {})) != expected_cells:
        raise ValueError("reused precision artifact cell set mismatch")
    if set(artifact.get("predictions", {})) != expected_cells:
        raise ValueError("reused precision artifact prediction set mismatch")
    for cell_id, record in artifact["predictions"].items():
        verify_record(base, record, f"reused predictions {cell_id}")
        reproduction = artifact["cells"][cell_id].get("reference_400_reproduction", {})
        if reproduction.get("matched") is not True:
            raise ValueError(f"reused reference reproduction is not verified for {cell_id}")
    return artifact, verified


def contract_payload(reference_path, checkpoint, frozen_inputs, reused_path, reused_verified):
    return {
        "schema_version": 1,
        "predeclaration": "docs/lanes/positive-control-power.md",
        "predeclaration_sha256": power.file_sha256(
            REPO / "docs" / "lanes" / "positive-control-power.md"
        ),
        "checkpoint": power.display_path(checkpoint),
        "reference": power.display_path(reference_path),
        "population": replay.POPULATION_LABEL,
        "controls": list(power.LEAD59_FIXED_CONTROL_COLUMNS),
        "new_coefficients_logit_sd": list(COEFFICIENTS),
        "reused_coefficients_logit_sd": list(REUSED_COEFFICIENTS),
        "draw_prefixes": list(DRAW_PREFIXES),
        "simulations_per_cell": 200,
        "seed": 20260923,
        "fit_iterations": power.FIT_ITERATIONS,
        "bootstrap_method": power.EVALUATOR_BOOTSTRAP_METHOD,
        "input_freeze": frozen_inputs,
        "extension_runner": {
            "path": power.display_path(Path(__file__)),
            "sha256": power.file_sha256(Path(__file__)),
        },
        "reused_artifact": {
            "path": power.display_path(reused_path),
            "verified_files": reused_verified,
        },
        "reported_diagnostic_looks": {
            "new_detection_configurations": 36,
            "new_paired_disagreement_summaries": 36,
            "new_total": 72,
            "combined_detection_configurations": 48,
            "combined_paired_disagreement_summaries": 48,
            "combined_total": 96,
            "original_coefficient_cell_looks": 16,
            "reference_400_reproduction_checks_are_selected_findings": False,
        },
    }


def initialize_checkpoint(checkpoint, contract):
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if checkpoint.exists():
        if not checkpoint.is_dir() or not manifest_path.is_file():
            raise FileExistsError("checkpoint exists without a compatible manifest")
        if read_json(manifest_path) != contract:
            raise ValueError("checkpoint manifest does not match the frozen run contract")
        return False
    checkpoint.mkdir(parents=True)
    (checkpoint / "cells").mkdir()
    (checkpoint / "predictions").mkdir()
    (checkpoint / "replicates").mkdir()
    (checkpoint / "fit_diagnostics").mkdir()
    write_json_once(manifest_path, contract)
    return True


def cell_paths(checkpoint, cell_id):
    safe = cell_id.replace(".", "p")
    return {
        "checkpoint": checkpoint / "cells" / f"{safe}.json",
        "predictions": checkpoint / "predictions" / f"{safe}.parquet",
        "replicates": checkpoint / "replicates" / f"{safe}.parquet",
        "fit_diagnostics": checkpoint / "fit_diagnostics" / f"{safe}.json",
    }


def load_completed_cell(checkpoint, cell_id, control, coefficient, contract_sha):
    paths = cell_paths(checkpoint, cell_id)
    if not paths["checkpoint"].exists():
        occupied = [name for name, path in paths.items() if name != "checkpoint" and path.exists()]
        if occupied:
            raise FileExistsError(f"incomplete outputs exist for {cell_id}: {', '.join(occupied)}")
        return None
    record = read_json(paths["checkpoint"])
    expected = {
        "cell_id": cell_id,
        "control": control,
        "coefficient_logit_sd": coefficient,
        "contract_sha256": contract_sha,
    }
    mismatches = [key for key, value in expected.items() if record.get(key) != value]
    if mismatches:
        raise ValueError(f"completed checkpoint mismatch for {cell_id}: {', '.join(mismatches)}")
    for label in ("predictions", "replicates", "fit_diagnostics"):
        verify_record(checkpoint, record[label], f"checkpoint {cell_id} {label}")
    if record.get("reference_400_reproduction", {}).get("matched") is not True:
        raise ValueError(f"completed checkpoint lacks reference reproduction for {cell_id}")
    return record


def run_extension_cell(
    checkpoint,
    contract_sha,
    population,
    base_dgp,
    control_diagnostics,
    controls,
    reference,
    control,
    coefficient,
    started,
):
    cell_id = f"{control}__coef_{coefficient:.2f}"
    paths = cell_paths(checkpoint, cell_id)
    completed = load_completed_cell(checkpoint, cell_id, control, coefficient, contract_sha)
    if completed is not None:
        print(json.dumps({"cell": cell_id, "status": "verified_checkpoint_reuse"}), flush=True)
        return completed
    prediction_batches = []
    original_bootstraps = {}
    seed = int(reference["seed"])
    sims = int(reference["sims_per_cell"])

    def capture(simulation, scored, original_bootstrap):
        scored.insert(0, "simulation", simulation)
        scored.insert(1, "synthetic_outcome_seed", seed + simulation)
        scored.insert(2, "bootstrap_seed", seed + 900000 + simulation)
        scored.insert(3, "control", control)
        scored.insert(4, "coefficient_logit_sd", coefficient)
        prediction_batches.append(scored)
        original_bootstraps[simulation] = original_bootstrap
        if (simulation + 1) % 20 == 0:
            print(
                json.dumps(
                    {
                        "cell": cell_id,
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
        control_diagnostics["controls"][control]["prevalence"],
        coefficient,
        sims,
        DRAW_PREFIXES[0],
        power.FIT_ITERATIONS,
        seed,
        power.EVALUATOR_BOOTSTRAP_METHOD,
        controls[control],
        capture,
    )
    reproduction = replay.exact_reference_check(
        observed,
        replay.reference_row(reference, control, coefficient),
    )
    replicate_rows = []
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
            probability_positive = float((prefix > 0.0).mean() + 0.5 * (prefix == 0.0).mean())
            prefix_record = {
                "draws": draws,
                "point": point,
                "low": float(low),
                "high": float(high),
                "probability_positive": probability_positive,
                "detected": bool(low > 0.0),
            }
            if draws == DRAW_PREFIXES[0] and prefix_record != original_bootstraps[simulation]:
                raise ValueError(
                    f"400-draw stream is not an exact prefix for {cell_id} simulation {simulation}"
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
    first_prefix = replicate_frame.loc[replicate_frame["bootstrap_draws"].eq(DRAW_PREFIXES[0])]
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
    occupied = [name for name, path in paths.items() if path.exists()]
    if occupied:
        raise FileExistsError(f"refusing to overwrite outputs for {cell_id}: {', '.join(occupied)}")
    prediction_frame.to_parquet(paths["predictions"], index=False)
    replicate_frame.to_parquet(paths["replicates"], index=False)
    write_json_once(paths["fit_diagnostics"], observed["fit_diagnostics"])
    record = {
        "cell_id": cell_id,
        "control": control,
        "coefficient_logit_sd": coefficient,
        "contract_sha256": contract_sha,
        "reference_400_reproduction": reproduction,
        "optimizer_checks": replay.compact_optimizer_checks(observed["fit_diagnostics"]),
        "reference_mean_effect_accuracy_points": observed["mean_effect_accuracy_points"],
        "predictions": {
            "path": paths["predictions"].relative_to(checkpoint).as_posix(),
            "sha256": power.file_sha256(paths["predictions"]),
            "rows": len(prediction_frame),
        },
        "replicates": {
            "path": paths["replicates"].relative_to(checkpoint).as_posix(),
            "sha256": power.file_sha256(paths["replicates"]),
            "rows": len(replicate_frame),
        },
        "fit_diagnostics": {
            "path": paths["fit_diagnostics"].relative_to(checkpoint).as_posix(),
            "sha256": power.file_sha256(paths["fit_diagnostics"]),
        },
    }
    write_json_once(paths["checkpoint"], record)
    print(
        json.dumps(
            {
                "cell": cell_id,
                "status": "checkpoint_written",
                "elapsed_s": round(time.time() - started, 1),
            }
        ),
        flush=True,
    )
    return record


def validate_completed_results(path, contract_sha, checkpoint):
    result = read_json(path)
    if result.get("checkpoint_contract_sha256") != contract_sha:
        raise ValueError("completed result contract mismatch")
    if result.get("reported_diagnostic_looks", {}).get("combined_total") != 96:
        raise ValueError("completed result look-count mismatch")
    if len(result.get("detection_estimates", [])) != 48:
        raise ValueError("completed result detection-summary mismatch")
    if len(result.get("paired_classification_disagreements", [])) != 48:
        raise ValueError("completed result disagreement-summary mismatch")
    expected_cells = {
        f"{control}__coef_{coefficient:.2f}"
        for control in power.LEAD59_FIXED_CONTROL_COLUMNS
        for coefficient in COEFFICIENTS
    }
    if set(result.get("new_cells", {})) != expected_cells:
        raise ValueError("completed result cell-set mismatch")
    for cell_id, saved in result["new_cells"].items():
        verified = load_completed_cell(
            checkpoint,
            cell_id,
            saved["control"],
            saved["coefficient_logit_sd"],
            contract_sha,
        )
        if saved != verified:
            raise ValueError(f"completed result cell manifest mismatch for {cell_id}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reuse-artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    reference_path = args.reference.resolve()
    reused_path = args.reuse_artifact.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_relative_to(OUTPUT_ROOT.resolve()):
        parser.error(f"--checkpoint must be below {power.display_path(OUTPUT_ROOT)}")
    reference = read_json(reference_path)
    replay.validate_reference(reference)
    for control in power.LEAD59_FIXED_CONTROL_COLUMNS:
        for coefficient in COEFFICIENTS:
            replay.reference_row(reference, control, coefficient)
    population, controls, control_diagnostics, sources = replay.load_pinned_population(reference)
    dgp_outputs = power.loso_fit_with_diagnostics(
        population,
        list(power.BASE_TERMS),
        power.FIT_RIDGE,
        power.FIT_ITERATIONS,
    )
    base_dgp = dgp_outputs[2]
    if dgp_outputs[4] != reference["populations"][replay.POPULATION_LABEL]["dgp_fit_diagnostics"]:
        raise ValueError("pinned base-DGP fit does not exactly match the reference")
    frozen_inputs = replay.input_freeze(reference_path, population, controls, base_dgp, sources)
    reused, reused_verified = validate_reused_artifact(reused_path, frozen_inputs)
    contract = contract_payload(
        reference_path,
        checkpoint,
        frozen_inputs,
        reused_path,
        reused_verified,
    )
    contract_sha = canonical_sha256(contract)
    created = initialize_checkpoint(checkpoint, contract)
    print(
        json.dumps(
            {
                "checkpoint": power.display_path(checkpoint),
                "contract_sha256": contract_sha,
                "status": "created" if created else "verified_resume",
            }
        ),
        flush=True,
    )
    result_path = checkpoint / "results.json"
    if result_path.exists():
        completed = validate_completed_results(result_path, contract_sha, checkpoint)
        print(
            json.dumps(
                {
                    "artifact": power.display_path(result_path),
                    "status": "verified_already_complete",
                    "reported_diagnostic_looks": completed["reported_diagnostic_looks"][
                        "combined_total"
                    ],
                }
            ),
            flush=True,
        )
        return 0
    started = time.time()
    cell_records = {}
    replicate_frames = []
    for control in power.LEAD59_FIXED_CONTROL_COLUMNS:
        for coefficient in COEFFICIENTS:
            record = run_extension_cell(
                checkpoint,
                contract_sha,
                population,
                base_dgp,
                control_diagnostics,
                controls,
                reference,
                control,
                coefficient,
                started,
            )
            cell_records[record["cell_id"]] = record
            replicate_frames.append(pd.read_parquet(checkpoint / record["replicates"]["path"]))
    new_replicates = pd.concat(replicate_frames, ignore_index=True)
    new_detection = replay.detection_summaries(new_replicates)
    new_disagreements = replay.disagreement_summaries(new_replicates)
    if len(new_detection) != 36 or len(new_disagreements) != 36:
        raise ValueError("new diagnostic look counts do not match the predeclaration")
    detection = sorted(
        [*reused["detection_estimates"], *new_detection],
        key=lambda row: (
            row["control"],
            row["coefficient_logit_sd"],
            row["bootstrap_draws"],
        ),
    )
    disagreements = sorted(
        [*reused["paired_classification_disagreements"], *new_disagreements],
        key=lambda row: (
            row["control"],
            row["coefficient_logit_sd"],
            row["left_draws"],
            row["right_draws"],
        ),
    )
    if len(detection) != 48 or len(disagreements) != 48:
        raise ValueError("combined diagnostic look counts do not match the predeclaration")
    results = {
        "schema_version": 1,
        "command": "python scripts/positive_control_precision_curve.py",
        "predeclaration": "docs/lanes/positive-control-power.md",
        "checkpoint_contract_sha256": contract_sha,
        "reference": power.display_path(reference_path),
        "population": replay.POPULATION_LABEL,
        "controls": list(power.LEAD59_FIXED_CONTROL_COLUMNS),
        "coefficients_logit_sd": sorted([*COEFFICIENTS, *REUSED_COEFFICIENTS]),
        "draw_prefixes": list(DRAW_PREFIXES),
        "simulations_per_cell": int(reference["sims_per_cell"]),
        "seed": int(reference["seed"]),
        "fit_iterations": power.FIT_ITERATIONS,
        "bootstrap_method": power.EVALUATOR_BOOTSTRAP_METHOD,
        "reported_diagnostic_looks": contract["reported_diagnostic_looks"],
        "input_freeze": frozen_inputs,
        "reused_artifact": contract["reused_artifact"],
        "new_cells": cell_records,
        "detection_estimates": detection,
        "paired_classification_disagreements": disagreements,
        "interpretation_contract": {
            "zero_crossing_closes_signal": False,
            "one_fitted_probability_selects_side": True,
            "model_selection_or_tuning": False,
            "natural_signal_bounded": False,
            "interpolation_authorized": False,
            "may_promote_close_enroll_or_serve": False,
        },
        "run_elapsed_s": round(time.time() - started, 1),
    }
    write_json_once(result_path, results)
    print(
        json.dumps(
            {
                "artifact": power.display_path(result_path),
                "new_cells": len(cell_records),
                "reported_diagnostic_looks": 96,
                "run_elapsed_s": results["run_elapsed_s"],
                "status": "complete",
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
