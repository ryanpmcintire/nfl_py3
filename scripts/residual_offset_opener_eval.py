"""Frozen MOD-06 offset study; see docs/residual_offset_study.md before running."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nfl_ats.margin import fit_margin_model
from nfl_ats.modeling import regular_season_rows
from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact

REPO = Path(__file__).resolve().parents[1]
FAMILY = "mod06_residual_offset_opener_v1"
ARMS = (
    "zero",
    "production",
    "shrink_025",
    "shrink_050",
    "shrink_075",
    "ew_64",
    "ew_256",
    "short_128",
    "long_40pct",
    "median",
)


def estimate_offsets(residuals: np.ndarray, longer: np.ndarray) -> dict[str, float]:
    """Inputs are finite, chronological out-of-time home-oriented errors."""
    values = np.asarray(residuals, dtype=float)
    long_values = np.asarray(longer, dtype=float)
    if not len(values) or not len(long_values):
        raise ValueError("Residual samples must be nonempty")
    if not np.isfinite(values).all() or not np.isfinite(long_values).all():
        raise ValueError("Residual samples must be finite")
    mean = float(values.mean())
    age = np.arange(len(values) - 1, -1, -1)
    return {
        "zero": 0.0,
        "production": mean,
        "shrink_025": 0.25 * mean,
        "shrink_050": 0.5 * mean,
        "shrink_075": 0.75 * mean,
        "ew_64": float(np.average(values, weights=2.0 ** (-age / 64))),
        "ew_256": float(np.average(values, weights=2.0 ** (-age / 256))),
        "short_128": float(values[-128:].mean()),
        "long_40pct": float(long_values.mean()),
        "median": float(np.median(values)),
    }


def training_before(features: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    return frame.loc[frame["result"].notna() & frame["gameday"].lt(cutoff)].sort_values(
        ["gameday", "game_id"]
    )


def weekly_models(features: pd.DataFrame, cutoff: pd.Timestamp, config: dict):
    training = training_before(features, cutoff)
    kwargs = {
        "target": "market_residual",
        "model_name": config["regressor"],
        "feature_profile": config["feature_profile"],
        "ridge_alpha": config["ridge_alpha"],
    }
    model = fit_margin_model(training, **kwargs)
    longer = fit_margin_model(training, distribution_fraction=0.40, **kwargs)
    return model, estimate_offsets(model.residuals, longer.residuals), training


def build_predictions(features: pd.DataFrame, archive: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Reproduce the established opener evaluator and fail on archive drift."""
    features = regular_season_rows(features)
    rows = []
    for (season, week), group in archive.groupby(["season", "week"], sort=True):
        scoring = features.loc[features["game_id"].isin(group["game_id"])].copy()
        archive_cutoff = pd.to_datetime(scoring["gameday"]).min()
        model, offsets, training = weekly_models(features, archive_cutoff, config)
        archived_training_max = training["gameday"].max()
        incumbent_offset = offsets["production"]
        full_week = features.loc[features["season"].eq(season) & features["week"].eq(week)]
        cutoff = min(archive_cutoff, pd.to_datetime(full_week["gameday"]).min())
        if cutoff < archive_cutoff:
            _, offsets, training = weekly_models(features, cutoff, config)
        clean_mean = offsets["production"]
        # The requested incumbent is the archived rule; candidate offsets exclude
        # an unarchived opening game even when the inherited archive refit does not.
        offsets["production"] = incumbent_offset
        scoring = scoring.merge(group[["game_id", "tue_open_home_spread"]], on="game_id")
        scoring["spread_line"] = scoring["tue_open_home_spread"]
        predicted = model.predict(scoring, probability_method="gaussian")
        predicted["game_id"] = scoring["game_id"].to_numpy()
        predicted = predicted.set_index("game_id")
        result = group.copy().set_index("game_id")
        predicted = predicted.reindex(result.index)
        np.testing.assert_allclose(
            predicted["predicted_market_residual"], result["residual_at_open"], rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            predicted["home_cover_probability"],
            result["home_cover_probability_at_open"],
            rtol=0,
            atol=1e-10,
        )
        result["decision_cutoff"] = cutoff
        result["archive_refit_cutoff"] = archive_cutoff
        result["archive_training_max_gameday"] = archived_training_max
        result["training_max_gameday"] = training["gameday"].max()
        result["clean_production_mean"] = clean_mean
        result["training_rows"] = model.training_rows
        result["distribution_rows"] = model.distribution_rows
        for arm, offset in offsets.items():
            result[f"offset_{arm}"] = offset
            result[f"pick_{arm}"] = result["residual_at_open"].add(offset).ge(0.0)
            result[f"correct_{arm}"] = (
                result[f"pick_{arm}"]
                .eq(result["margin_vs_open"].gt(0))
                .astype(float)
                .where(result["margin_vs_open"].ne(0))
            )
        if not result["pick_production"].equals(result["pick_home_at_open_probability_rule"]):
            raise ValueError("Rebuilt Gaussian boundary differs from archived production picks")
        rows.append(result.reset_index())
        print(f"rebuilt {season} week {week}: {len(group)} games", flush=True)
    return pd.concat(rows, ignore_index=True).sort_values(["season", "week", "game_id"])


def paired_bootstrap(frame: pd.DataFrame, samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Algebraically identical whole-week sum/count bootstrap; no rho padding."""
    differences = (
        frame[[f"correct_{arm}" for arm in ARMS]].to_numpy()
        - frame["correct_production"].to_numpy()[:, None]
    )
    groups = list(frame.groupby(["season", "week"], sort=False).indices.values())
    sums = np.array([differences[g].sum(axis=0) for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(ARMS)))
    for start in range(0, samples, 500):
        selected = rng.integers(0, len(groups), size=(min(500, samples - start), len(groups)))
        draws[start : start + len(selected)] = (
            100 * sums[selected].sum(axis=1) / counts[selected].sum(axis=1)[:, None]
        )
    return 100 * differences.mean(axis=0), draws


def frozen_pick_null(frame: pd.DataFrame, samples: int, seed: int) -> np.ndarray:
    picks = frame[[f"pick_{arm}" for arm in ARMS]].to_numpy(dtype=bool)
    truth = frame["margin_vs_open"].gt(0).to_numpy()
    groups = list(frame.groupby(["season", "week"], sort=False).indices.values())
    baseline = ARMS.index("production")
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(ARMS)))
    for i in range(samples):
        shuffled = truth.copy()
        for group in groups:
            shuffled[group] = rng.permutation(truth[group])
        correct = (picks == shuffled[:, None]).astype(float)
        draws[i] = 100 * (correct - correct[:, baseline, None]).mean(axis=0)
    return draws


def summarize(
    predictions: pd.DataFrame,
    samples: int = 20_000,
    seed: int = 20260817,
    null_samples: int = 2_000,
) -> dict:
    valid = predictions.loc[predictions["margin_vs_open"].ne(0)].reset_index(drop=True)
    estimate, draws = paired_bootstrap(valid, samples, seed)
    null = frozen_pick_null(valid, null_samples, seed)
    arms = {}
    for index, arm in enumerate(ARMS):
        low, high = np.quantile(draws[:, index], [0.025, 0.975])
        arms[arm] = {
            "delta": float(estimate[index]),
            "interval_low": float(low),
            "interval_high": float(high),
            "probability_positive": float(np.mean(draws[:, index] > 0)),
            "standard_error": float(draws[:, index].std(ddof=1)),
            "n": len(valid),
            "accuracy": float(valid[f"correct_{arm}"].mean()),
            "disagreements": int(valid[f"pick_{arm}"].ne(valid["pick_production"]).sum()),
            "season_deltas": {
                str(season): float(100 * (g[f"correct_{arm}"] - g["correct_production"]).mean())
                for season, g in valid.groupby("season")
            },
            "null_mean": float(null[:, index].mean()),
            "null_interval": np.quantile(null[:, index], [0.025, 0.975]).tolist(),
            "null_observed_percentile": float(np.mean(null[:, index] <= estimate[index])),
            "classification": "refuted_mechanism" if high < 0 else "unresolved_below_power",
        }
    alternatives = [i for i, arm in enumerate(ARMS) if arm != "production"]
    null_max = null[:, alternatives].max(axis=1)
    order = ["production", *(arm for arm in ARMS if arm != "production")]
    winner = max(order, key=lambda arm: arms[arm]["delta"])
    return {
        "family": FAMILY,
        "grade": "opener",
        "effect_units": "accuracy_points",
        "archive_rows": len(predictions),
        "graded_games": len(valid),
        "pushes": len(predictions) - len(valid),
        "weeks": int(valid.groupby(["season", "week"]).ngroups),
        "season_start": int(valid["season"].min()),
        "season_end": int(valid["season"].max()),
        "bootstrap_samples": samples,
        "seed": seed,
        "null_samples": null_samples,
        "within_week_correlation": 0,
        "arms": arms,
        "decision_arm": winner,
        "family_max_null_mean": float(null_max.mean()),
        "family_max_null_interval": np.quantile(null_max, [0.025, 0.975]).tolist(),
        "family_max_null_observed_percentile": float(np.mean(null_max <= arms[winner]["delta"])),
        "discount": (
            "Mined archive; inherited MOD-08/model selection; ten correlated arms; "
            "descriptive union, not independent confirmation."
        ),
    }


def record_results(directory: Path, *, union: bool = False, replace_signals: bool = False) -> None:
    """Explicit recording mode: all registry mutations go through public CLIs."""
    result = json.loads((directory / "results.json").read_text())
    # Verdict corrections never alter a measurement or relax a validator.
    for metrics in result["arms"].values():
        metrics["classification"] = (
            "refuted_mechanism" if metrics["interval_high"] < 0 else "unresolved_below_power"
        )
    write_stamped_artifact(result, directory / "results.json")
    artifact = str(directory / "results.json")
    commands = []

    def cli(arguments: list[str]) -> None:
        command = [str(REPO / ".tools/uv.exe"), "run", "--no-sync", "nfl-ats", *arguments]
        completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        commands.append(
            {
                "argv": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        (directory / "recording_commands.json").write_text(json.dumps(commands, indent=2))
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)

    for arm, metrics in result["arms"].items():
        arguments = [
            "weak-signals",
            "record",
            "--name",
            f"{FAMILY}_{arm}_{result['season_start']}_{result['season_end']}",
            "--description",
            f"Opener forced-pick accuracy: {arm} residual offset versus Gaussian production.",
            "--source",
            artifact,
            "--effect",
            str(metrics["delta"]),
            "--effect-units",
            "accuracy_points",
            "--classification",
            metrics["classification"],
            "--league",
            "nfl",
            "--season-start",
            str(result["season_start"]),
            "--season-end",
            str(result["season_end"]),
            "--interval-low",
            str(metrics["interval_low"]),
            "--interval-high",
            str(metrics["interval_high"]),
            "--probability-positive",
            str(metrics["probability_positive"]),
            "--sample-games",
            str(metrics["n"]),
            "--sample-blocks",
            str(result["weeks"]),
            "--family",
            FAMILY,
            "--category",
            "control" if arm == "production" else "modeling",
            "--classification-evidence",
            (
                "Resolved wrong sign for this offset arm on this block: upper endpoint is negative."
                if metrics["classification"] == "refuted_mechanism"
                else "No resolved wrong sign or reliability/control closing ground. "
                "Incumbent self-comparison is an exact tie."
            ),
            "--plain-summary",
            f"Average earlier prediction errors using {arm.replace('_', ' ')} "
            "to choose a side against the opening spread.",
            "--notes",
            result["discount"]
            + (
                " Union overlaps all three recorded blocks."
                if union
                else " Sequential frozen block; all ten arms retained."
            ),
        ]
        if metrics["standard_error"] > 0:
            arguments.extend(["--standard-error", str(metrics["standard_error"])])
        if metrics["classification"] == "refuted_mechanism":
            arguments.extend(["--closing-ground", "wrong_sign_resolved"])
        if replace_signals:
            arguments.append("--replace")
        cli(arguments)
    if not union and not replace_signals:
        winner = result["arms"][result["decision_arm"]]
        cli(
            [
                "rotation",
                "record",
                "--name",
                FAMILY,
                "--artifact",
                artifact,
                "--verdict",
                "unresolved",
                "--probability-positive",
                str(winner["probability_positive"]),
                "--effect",
                str(winner["delta"]),
                "--effect-units",
                "accuracy_points",
                "--interval-low",
                str(winner["interval_low"]),
                "--interval-high",
                str(winner["interval_high"]),
                "--sample-blocks",
                str(result["weeks"]),
                "--notes",
                "All ten arms recorded separately. Headline is selected block winner "
                f"{result['decision_arm']}; not a preselected arm. " + result["discount"],
            ]
        )
    print(
        f"Recorded {len(result['arms'])} arms; "
        f"rotation look recorded: {not union and not replace_signals}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=REPO / "artifacts/opener_evaluation/20260907T122841Z"
    )
    parser.add_argument(
        "--features", type=Path, default=REPO / "data/processed/game_features_weak_stack.parquet"
    )
    parser.add_argument("--season-start", type=int, default=2020)
    parser.add_argument("--season-end", type=int, default=2025)
    parser.add_argument("--combine", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--null-samples", type=int, default=2_000)
    parser.add_argument("--record-results", type=Path)
    parser.add_argument("--union-record", action="store_true")
    parser.add_argument("--replace-signals", action="store_true")
    parser.add_argument("--replay", action="store_true", help="Verify exact rerun; never overwrite")
    parser.add_argument(
        "--repair-existing",
        action="store_true",
        help="Repair timing provenance only if every pick and metric is unchanged",
    )
    args = parser.parse_args()
    if args.record_results:
        record_results(
            args.record_results, union=args.union_record, replace_signals=args.replace_signals
        )
        return
    out = args.out_dir or REPO / "artifacts/residual_offset_study" / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    metadata = json.loads((args.archive / "metadata.json").read_text())
    config = metadata["active_model_config"]
    if args.combine:
        predictions = pd.concat([pd.read_parquet(p / "per_game.parquet") for p in args.combine])
        if predictions["game_id"].duplicated().any():
            raise ValueError("Combined blocks overlap")
        expected = pd.read_parquet(args.archive / "per_game.parquet")
        if set(predictions["game_id"]) != set(expected["game_id"]):
            raise ValueError("Combined games differ from the frozen archive")
    else:
        digest = hashlib.sha256(args.features.read_bytes()).hexdigest()
        if digest != metadata["feature_table_sha256"]:
            raise ValueError("Feature digest differs from frozen archive")
        registry = json.loads((REPO / "registry/rotation_registry.json").read_text())
        windows = registry["families"][FAMILY]["windows"]
        allowed = [
            w
            for w in windows
            if w["seasons"] == [args.season_start, args.season_end]
            and (
                w["state"] == "assigned"
                or (
                    (args.replay or args.repair_existing)
                    and w["artifact"]
                    and Path(w["artifact"]).resolve() == (out / "results.json").resolve()
                )
            )
        ]
        if len(allowed) != 1:
            raise ValueError("Assign precisely this season block through rotation before scoring")
        archive = pd.read_parquet(args.archive / "per_game.parquet")
        archive = archive.loc[archive["season"].between(args.season_start, args.season_end)]
        with threadpool_limits(limits=1):
            predictions = build_predictions(pd.read_parquet(args.features), archive, config)
    predictions = predictions.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    results = summarize(predictions, args.samples, null_samples=args.null_samples)
    results.update(
        {
            "active_model_config": config,
            "archive": str(args.archive),
            "predeclaration": "docs/residual_offset_study.md",
            "predeclaration_sha256": hashlib.sha256(
                (REPO / "docs/residual_offset_study.md")
                .read_bytes()
                .split(b"\n## Post-score leakage audit correction")[0]
            ).hexdigest(),
        }
    )
    if args.replay or args.repair_existing:
        previous = json.loads((out / "results.json").read_text())
        for key, value in results.items():
            if value != previous[key]:
                raise ValueError(f"Replay changed result field {key}")
        old_predictions = pd.read_parquet(out / "per_game.parquet")
        if args.repair_existing:
            columns = ["game_id", *[f"pick_{a}" for a in ARMS], *[f"correct_{a}" for a in ARMS]]
            pd.testing.assert_frame_equal(predictions[columns], old_predictions[columns])
            print("Timing correction verified: every pick and every statistical result unchanged")
        else:
            pd.testing.assert_frame_equal(predictions, old_predictions)
            print(f"Exact prediction and result replay verified: {out}")
            return
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out / "per_game.parquet", index=False)
    stamp_sidecar(out / "per_game.parquet")
    write_stamped_artifact(results, out / "results.json")
    print(
        json.dumps(
            {"out_dir": str(out), "decision_arm": results["decision_arm"], "arms": results["arms"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
