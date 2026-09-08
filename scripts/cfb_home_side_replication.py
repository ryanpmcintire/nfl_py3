"""Frozen Lane O replication; see docs/cfb_home_side_replication.md."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nfl_ats.cfb_benchmark import CFB_CLEAN_CORE_SEASONS, fit_cfb_residual_model
from nfl_ats.home_side_location import fit_home_side_offsets, prior_rows_before
from nfl_ats.margin import fit_market_baseline
from nfl_ats.provenance import sha256_file, stamp_sidecar, write_stamped_artifact
from nfl_ats.spread_regime import BUCKETS, spread_bucket

OUT = Path("artifacts/research/laneO")
INPUT = Path("data/processed/cfb_game_features.parquet")
FAMILY = "mod18_home_side_location_cfb_v1"
SEED = 20260817
DRAWS = 20_000


def diagnose(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["bucket"] = spread_bucket(result.spread_line)
    result["side"] = np.select(
        [result.spread_line.gt(0), result.spread_line.lt(0)],
        ["home_favourite", "home_underdog"],
        default="pickem",
    )
    result["error"] = result.result - result.point_incumbent
    return result


def eligible_prior(stream: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    prior = prior_rows_before(stream, int(target.season.iloc[0]), int(target.week.iloc[0]))
    return prior.loc[
        (pd.to_datetime(prior.gameday) + pd.Timedelta(days=1)).lt(target.gameday.min())
    ]


def predict(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame.gameday)
    frame = frame.loc[frame.result.notna() & frame.ats_margin.notna()]
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    batches = []
    for (season, week), target in frame.loc[frame.season.between(2006, 2025)].groupby(
        ["season", "week"], sort=True
    ):
        training = frame.loc[frame.gameday.lt(target.gameday.min())]
        if len(training) < 500:
            continue
        model = fit_cfb_residual_model(training)
        raw = model.predict(target)
        batch = target[["game_id", "season", "week", "gameday", "spread_line", "result"]].copy()
        batch["point_incumbent"] = raw.predicted_margin
        batch["p_raw"] = raw.home_cover_probability
        batch["p_market"] = fit_market_baseline(training).predict(target).home_cover_probability
        stream = pd.concat(batches) if batches else batch.iloc[:0]
        prior = eligible_prior(stream, target)
        for arm in ("s3", "s2"):
            fitted = fit_home_side_offsets(prior, all_buckets=arm == "s2")
            offset = fitted.offset_for(target.spread_line)
            adjusted = model.predict(target, center_offset=offset.to_numpy())
            batch[f"offset_{arm}"] = offset
            batch[f"p_{arm}"] = adjusted.home_cover_probability
            batch[f"prior_n_{arm}"] = spread_bucket(target.spread_line).map(fitted.prior_games)
        batch["training_rows"] = len(training)
        batch["training_max_gameday"] = training.gameday.max()
        batches.append(batch)
        if int(week) == 1:
            print(f"Fitted season {season}: {len(training)} prior games", flush=True)
    return diagnose(pd.concat(batches, ignore_index=True))


def interval(frame: pd.DataFrame, values: np.ndarray) -> dict:
    groups = list(frame.groupby(["season", "week"], sort=True).indices.values())
    sums = np.array([values[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(DRAWS // 1000):
        draw = rng.integers(0, len(groups), (1000, len(groups)))
        samples.append(sums[draw].sum(axis=1) / counts[draw].sum(axis=1))
    draws = np.concatenate(samples)
    return {
        "effect": float(values.mean()),
        "interval_low": float(np.quantile(draws, 0.025)),
        "interval_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float((draws > 0).mean()),
        "sample_games": len(frame),
        "sample_blocks": len(groups),
        "season_start": int(frame.season.min()),
        "season_end": int(frame.season.max()),
    }


def subsets(frame: pd.DataFrame):
    yield "all", frame
    yield "clean_core", frame.loc[frame.season.isin(CFB_CLEAN_CORE_SEASONS)]
    for start, end in ((2006, 2011), (2012, 2019), (2020, 2020), (2021, 2025)):
        yield f"era_{start}_{end}", frame.loc[frame.season.between(start, end)]
    for season, group in frame.groupby("season"):
        yield f"season_{season}", group


def bucket_subsets(frame: pd.DataFrame):
    yield "all", frame
    for bucket in BUCKETS:
        yield bucket, frame.loc[frame.bucket.eq(bucket)]
    yield "14.5-21", frame.loc[frame.spread_line.abs().gt(14) & frame.spread_line.abs().le(21)]
    yield "21+", frame.loc[frame.spread_line.abs().gt(21)]


def metrics(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    outcome = frame.result.gt(frame.spread_line).to_numpy(dtype=float)
    result = {}
    for arm in ("raw", "s3", "s2", "market"):
        p = frame[f"p_{arm}"].to_numpy()
        q = np.clip(p, 1e-15, 1 - 1e-15)
        result[f"{arm}_accuracy_points"] = ((p >= 0.5) == outcome) * 100.0
        result[f"{arm}_brier_improvement"] = (p - outcome) ** 2
        result[f"{arm}_log_loss_improvement"] = -(
            outcome * np.log(q) + (1 - outcome) * np.log1p(-q)
        )
    return result


def evaluate(frame: pd.DataFrame) -> list[dict]:
    cells = []
    for window, era in subsets(frame):
        for bucket, group in bucket_subsets(era):
            for side in ("all", "home_favourite", "home_underdog"):
                cell = group if side == "all" else group.loc[group.side.eq(side)]
                if cell.empty:
                    continue
                cells.append(
                    {
                        "kind": "diagnosis",
                        "window": window,
                        "bucket": bucket,
                        "side": side,
                        "units": "ats_points",
                        **interval(cell, cell.error.to_numpy()),
                    }
                )
            nonpush = group.loc[group.result.ne(group.spread_line)]
            if nonpush.empty:
                continue
            scores = metrics(nonpush)
            for arm in ("s3", "s2"):
                for units in ("accuracy_points", "brier_improvement", "log_loss_improvement"):
                    baseline, candidate = scores[f"raw_{units}"], scores[f"{arm}_{units}"]
                    difference = (
                        candidate - baseline if units == "accuracy_points" else baseline - candidate
                    )
                    cells.append(
                        {
                            "kind": arm,
                            "window": window,
                            "bucket": bucket,
                            "side": "all",
                            "units": units,
                            "baseline": float(baseline.mean()),
                            "candidate": float(candidate.mean()),
                            "market": float(scores[f"market_{units}"].mean()),
                            **interval(nonpush, difference),
                        }
                    )
    for cell in cells:
        suffix = "_".join(str(cell[k]) for k in ("kind", "window", "bucket", "side", "units"))
        cell["name"] = FAMILY + "_" + re.sub(r"[^a-z0-9_]+", "_", suffix.lower()).strip("_")
    assert len({c["name"] for c in cells}) == len(cells)
    return cells


def record(cells: list[dict]) -> None:
    from nfl_ats.cli import main
    from nfl_ats.cli_commands.registry import weak_signal_registry_path

    commands = []
    for cell in cells:
        registry_path = weak_signal_registry_path()
        existing = (
            json.loads(registry_path.read_text(encoding="utf-8"))["signals"]
            if registry_path.exists()
            else {}
        )
        if cell["name"] in existing:
            saved = existing[cell["name"]]
            assert abs(saved["effect"] - cell["effect"]) < 1e-10
            continue
        args = [
            "weak-signals",
            "record",
            "--name",
            cell["name"],
            "--family",
            FAMILY,
            "--description",
            f"CFB {cell['kind']} {cell['window']} {cell['bucket']} {cell['side']}",
            "--source",
            str(OUT / "cells.json"),
            "--league",
            "cfb",
            "--classification",
            "unresolved_below_power",
            "--effect-units",
            cell["units"],
            "--category",
            "modeling",
            "--notes",
            "Independent CFB replication of frozen NFL shape; close-proxy grade; "
            "overlapping cells; no NFL card change.",
        ]
        for key in (
            "effect",
            "interval_low",
            "interval_high",
            "probability_positive",
            "sample_games",
            "sample_blocks",
            "season_start",
            "season_end",
        ):
            value = cell[key]
            args.extend(
                [
                    "--" + key.replace("_", "-"),
                    f"{value:.12f}" if isinstance(value, float) else str(value),
                ]
            )
        output = io.StringIO()
        for attempt in range(10):
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    code = main(args)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.5)
        commands.append({"argv": args, "returncode": code, "output": output.getvalue()})
        if code:
            write_stamped_artifact({"commands": commands}, OUT / "record_commands.json")
            raise RuntimeError(output.getvalue())
    write_stamped_artifact({"commands": commands}, OUT / "record_commands.json")
    saved = json.loads(weak_signal_registry_path().read_text(encoding="utf-8"))["signals"]
    missing = [c["name"] for c in cells if c["name"] not in saved]
    if missing:
        raise RuntimeError(
            f"Registry verification found {len(missing)} missing rows; resume recording"
        )
    write_stamped_artifact(
        {"names": [c["name"] for c in cells], "count": len(cells)}, OUT / "registry_verified.json"
    )
    print(f"Verified {len(cells)} CFB cells; recorded {len(commands)} this invocation", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--record-only", action="store_true")
    args = parser.parse_args()
    if args.record_only:
        record(json.loads((OUT / "cells.json").read_text())["cells"])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if args.reuse_predictions:
        frame = pd.read_parquet(OUT / "predictions.parquet")
    else:
        with threadpool_limits(limits=1):
            frame = predict(pd.read_parquet(INPUT))
        frame.to_parquet(OUT / "predictions.parquet", index=False)
        stamp_sidecar(
            OUT / "predictions.parquet",
            {
                "input_sha256": sha256_file(INPUT),
                "predeclaration_sha256": sha256_file(Path("docs/cfb_home_side_replication.md")),
            },
        )
    cells = evaluate(frame)
    write_stamped_artifact(
        {"cells": cells, "draws": DRAWS, "seed": SEED, "within_week_correlation": 0},
        OUT / "cells.json",
    )
    print(json.dumps([c for c in cells if c["window"] == "clean_core" and c["bucket"] == "all"]))
    if args.record:
        record(cells)


if __name__ == "__main__":
    main()
