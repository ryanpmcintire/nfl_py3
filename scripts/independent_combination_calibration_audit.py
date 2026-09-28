from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from independent_combination_replay import (
    exact_disagreement,
    features,
    logit_frame,
    metrics,
    predict_matrix,
    reliability,
)

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.io import atomic_json
from nfl_ats.provenance import sha256_file


def main() -> None:
    declaration_path = Path(
        "registry/studies/combined_vs_raw_historical_nested_20260928_calibration_diagnostic.json"
    )
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    source_path = Path(declaration["source_report"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    protocol = json.loads(Path(source["protocol_path"]).read_text(encoding="utf-8"))
    if sha256_file(Path(protocol["input"])) != source["input_sha256"]:
        raise ValueError("Replay input changed")
    replay_source = source_path.parent / "replay_source.py"
    if not replay_source.exists():
        replay_source = Path("scripts/independent_combination_replay.py")
    if sha256_file(replay_source) != source["script_sha256"]:
        raise ValueError("Replay implementation changed")
    destination = source_path.parent / "calibration_diagnostic"
    if destination.exists():
        raise ValueError("Refusing to replace the calibration diagnostic")
    frame = pd.read_parquet(protocol["input"])
    predictions = pd.read_parquet(source_path.parent / "predictions.parquet")
    stage_gaps = []
    for fold in source["folds"]:
        season = fold["outer_season"]
        part = frame.loc[frame.season.eq(season)].copy()
        development = frame.loc[frame.season.lt(season)].copy()
        for added, calibrated in (
            ("pre_nested", "nested_selected"),
            ("pre_fixed", "fixed_combination"),
        ):
            fitted = fold["fitted_models"][calibrated]
            flags, market = tuple(fitted["flags"]), fitted["market"]
            p = predict_matrix(fitted["base"], features(part, flags, market))
            lookup = pd.Series(p, index=part.game_id)
            mask = predictions.season.eq(season)
            predictions.loc[mask, added] = predictions.loc[mask, "game_id"].map(lookup)
            recovered = predict_matrix(fitted["calibrator"], logit_frame(p, part.index))
            actual = predictions.set_index("game_id").loc[part.game_id, calibrated].to_numpy()
            if not np.allclose(recovered, actual, rtol=0, atol=1e-12):
                raise ValueError("Stored calibration does not reproduce the primary prediction")
            training_end = fold["windows"]["validation"]["last_season_week"]
            trained = development.loc[
                (development.season * 100 + development.week).le(training_end)
            ]
            train_p = predict_matrix(fitted["base"], features(trained, flags, market))
            train_metrics = metrics(trained.home_covered.to_numpy(dtype=float), train_p)
            outer_metrics = metrics(part.home_covered.to_numpy(dtype=float), p)
            stage_gaps.append(
                {
                    "season": season,
                    "arm": added,
                    "fitted_training": train_metrics,
                    "outer": outer_metrics,
                    "training_minus_outer_accuracy_points": 100
                    * (train_metrics["accuracy"] - outer_metrics["accuracy"]),
                    "outer_minus_training_brier": outer_metrics["brier"] - train_metrics["brier"],
                    "coefficients": fitted["base"]["coefficients"],
                }
            )
    y = predictions.home_covered.to_numpy(dtype=float)
    arms = {}
    for arm in declaration["arms_added"]:
        p = predictions[arm].to_numpy(dtype=float)
        arms[arm] = {
            **metrics(y, p),
            "reliability": reliability(y, p, protocol["reliability_bands"]),
            "seasons": {
                str(season): metrics(
                    group.home_covered.to_numpy(dtype=float), group[arm].to_numpy(dtype=float)
                )
                for season, group in predictions.groupby("season")
            },
        }
    paired = predictions[["season", "week"]].copy()
    disagreements = {}
    for candidate, baseline in declaration["contrasts"]:
        name = candidate + "_vs_" + baseline
        a, b = predictions[candidate].to_numpy(), predictions[baseline].to_numpy()
        paired[name + "__brier_improvement"] = (b - y) ** 2 - (a - y) ** 2
        paired[name + "__log_loss_improvement"] = (
            -y * np.log(b) - (1 - y) * np.log1p(-b) + y * np.log(a) + (1 - y) * np.log1p(-a)
        )
        paired[name + "__accuracy_points"] = 100 * (
            ((a >= 0.5) == y).astype(float) - ((b >= 0.5) == y).astype(float)
        )
        disagreements[name] = exact_disagreement(y, a, b)
    names = list(paired.columns[2:])
    values = paired[names].to_numpy(dtype=float)

    def metric_fn(sample: pd.DataFrame) -> dict[str, float]:
        return {name: float(sample[name].mean()) for name in names}

    def factory(valid: pd.DataFrame) -> Any:
        def draw(positions: np.ndarray) -> dict[str, float]:
            return dict(zip(names, values[positions].mean(axis=0).tolist(), strict=True))

        return draw

    settings = declaration["bootstrap"]
    intervals = week_blocked_bootstrap(
        paired,
        metric_fn,
        samples=settings["samples"],
        confidence=settings["confidence"],
        seed=settings["seed"],
        metric_columns=names,
        metric_draw_factory=factory,
    )
    result = {
        "study_id": declaration["study_id"],
        "declaration_sha256": sha256_file(declaration_path),
        "script_sha256": sha256_file(Path(__file__)),
        "primary_report_sha256": sha256_file(source_path),
        "post_result_diagnostic": True,
        "new_fits": 0,
        "games": len(predictions),
        "weeks": source["weeks"],
        "arms": arms,
        "stage_gaps": stage_gaps,
        "paired_intervals": intervals.to_dict(orient="records"),
        "disagreements": disagreements,
        "look_accounting": declaration["look_accounting"],
    }
    destination.mkdir()
    (destination / "replay_source.py").write_bytes(Path(__file__).read_bytes())
    predictions.to_parquet(destination / "predictions.parquet", index=False)
    atomic_json(result, destination / "report.json")
    cells = []
    for row in intervals.to_dict(orient="records"):
        contrast, units = row["metric"].split("__")
        cells.append(
            {
                "name": declaration["study_id"] + "_" + contrast + "_" + units,
                "description": "Post-result calibration-stage diagnostic: " + contrast,
                "effect": row["estimate"],
                "effect_units": units,
                "interval_low": row["lower"],
                "interval_high": row["upper"],
                "probability_positive": row["probability_positive"],
            }
        )
    batch = {
        "defaults": {
            "league": "nfl",
            "season_start": 2024,
            "season_end": 2025,
            "family": declaration["study_id"],
            "category": "modeling",
            "classification": "unresolved_below_power",
            "classification_evidence": "Post-result diagnostic of a newly added calibration "
            "stage; cannot close an underlying signal or undo prior feature-design selection.",
            "source": (destination / "report.json").as_posix(),
            "sample_games": len(predictions),
            "sample_blocks": source["weeks"],
            "plain_summary": "Reuses stored fitted models to isolate small-block calibration "
            "without refitting or replacing the primary historical replay.",
            "notes": "All 15 diagnostic contrasts are retained. No serving change.",
        },
        "cells": cells,
    }
    shared = batch.pop("defaults")
    batch["family"] = shared["family"]
    batch["cells"] = [{**shared, **cell} for cell in batch["cells"]]
    atomic_json(batch, destination / "weak_signals_batch.json")
    print(
        json.dumps(
            {
                "arms": {
                    name: {k: v for k, v in row.items() if k != "reliability"}
                    for name, row in arms.items()
                },
                "paired_intervals": result["paired_intervals"],
                "disagreements": disagreements,
                "report": (destination / "report.json").as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
