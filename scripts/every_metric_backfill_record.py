from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import every_metric_backfill as em  # noqa: E402

UNIT1_RESULTS = os.path.join(
    REPO_ROOT, "artifacts", "every_metric", "20260916T152725Z", "results.json"
)
OUTPUT_ROOT = os.path.join(REPO_ROOT, "artifacts", "every_metric_backfill")
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260916
MIN_PAIRED_GAMES = 200

METRICS = ("accuracy", "brier", "log_loss", "margin_mae")
UNIT_OF = {
    "accuracy": "accuracy_points",
    "brier": "brier_improvement",
    "log_loss": "log_loss_improvement",
    "margin_mae": "mae_improvement",
}
SCALE_OF = {"accuracy": 100.0, "brier": 1.0, "log_loss": 1.0, "margin_mae": 1.0}
PLAIN_OF = {
    "accuracy": "Re-reading these past forecasts on winners picked: {words}.",
    "brier": "Re-reading these past forecasts on sharpness of the weekly chances: {words}.",
    "log_loss": "Re-reading these past forecasts on confidence calibration: {words}.",
    "margin_mae": "Re-reading these past forecasts on final-score closeness: {words}.",
}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "single"


def per_game_metric_frame(path):
    df, _available = em.load_needed_columns(path)
    if df is None or df.empty:
        return None
    margin = em.actual_margin_series(df)
    market_signed, _used = em.market_column(df)
    cover = em.actual_cover_series(df, margin, market_signed)
    seasons = em.season_series(df)
    frames = {}
    if "home_cover_probability" in df.columns and cover is not None:
        prob = pd.to_numeric(df["home_cover_probability"], errors="coerce").clip(
            em.EPS, 1.0 - em.EPS
        )
        cov = pd.to_numeric(cover, errors="coerce")
        ok = prob.notna() & cov.notna()
        pick = (prob[ok] > 0.5).astype(float)
        frames["accuracy"] = pd.DataFrame(
            {
                "game_id": df.loc[ok, "game_id"].astype(str)
                if "game_id" in df.columns
                else ok.index.astype(str),
                "season": pd.to_numeric(seasons, errors="coerce").loc[ok],
                "diff": (pick.to_numpy() == cov[ok].to_numpy()).astype(float) - 0.5,
            }
        )
        frames["brier"] = pd.DataFrame(
            {
                "game_id": frames["accuracy"]["game_id"],
                "season": frames["accuracy"]["season"],
                "diff": 0.25 - (prob[ok].to_numpy() - cov[ok].to_numpy()) ** 2,
            }
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = -(cov[ok].to_numpy() * np.log(prob[ok].to_numpy()))
            ll = ll - ((1.0 - cov[ok].to_numpy()) * np.log(1.0 - prob[ok].to_numpy()))
        frames["log_loss"] = pd.DataFrame(
            {
                "game_id": frames["accuracy"]["game_id"],
                "season": frames["accuracy"]["season"],
                "diff": float(em.LN2) - ll,
            }
        )
    if "predicted_margin" in df.columns and margin is not None and market_signed is not None:
        predicted = pd.to_numeric(df["predicted_margin"], errors="coerce")
        mgn = pd.to_numeric(margin, errors="coerce")
        mkt = pd.to_numeric(market_signed, errors="coerce")
        ok = predicted.notna() & mgn.notna() & mkt.notna()
        frames["margin_mae"] = pd.DataFrame(
            {
                "game_id": df.loc[ok, "game_id"].astype(str)
                if "game_id" in df.columns
                else ok.index.astype(str),
                "season": pd.to_numeric(seasons, errors="coerce").loc[ok],
                "diff": np.abs(mkt[ok].to_numpy() - mgn[ok].to_numpy())
                - np.abs(predicted[ok].to_numpy() - mgn[ok].to_numpy()),
            }
        )
    return frames


def season_bootstrap_interval(diffs, seasons, draws, seed):
    frame = pd.DataFrame({"season": list(seasons), "diff": list(diffs)})
    frame = frame.dropna()
    seasons_present = sorted(frame["season"].unique())
    if len(seasons_present) < 2:
        return None
    rng = np.random.default_rng(seed)
    effects = np.empty(draws)
    for draw in range(draws):
        picked = rng.choice(seasons_present, size=len(seasons_present), replace=True)
        pooled = pd.concat([frame.loc[frame["season"].eq(s)] for s in picked])
        effects[draw] = pooled["diff"].mean()
    low, high = np.quantile(effects, [0.025, 0.975])
    return float(low), float(high), float((effects > 0).mean())


def label_of(df, path):
    if "method" in df.columns:
        methods = sorted(str(v) for v in df["method"].dropna().unique())
        models = (
            sorted(str(v) for v in df["model_name"].dropna().unique())
            if "model_name" in df.columns
            else []
        )
        tag = "+".join(methods[:3])
        if models:
            tag += "|" + "+".join(models[:3])
        return tag or "single"
    return "single"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    with open(UNIT1_RESULTS, encoding="utf-8") as fh:
        unit1 = json.load(fh)
    inventory = [
        item
        for item in unit1["inventory"]
        if item.get("skip_reason") is None and item.get("supports")
    ]
    inventory.sort(key=lambda item: item["path"])
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(OUTPUT_ROOT, stamp)
    os.makedirs(out_dir, exist_ok=True)
    pooled = {}
    for item in inventory:
        path = os.path.join(REPO_ROOT, item["path"].replace("/", os.sep))
        if not os.path.isfile(path):
            continue
        df, _available = em.load_needed_columns(path)
        if df is None or df.empty:
            continue
        tag = label_of(df, path)
        frames = per_game_metric_frame(path)
        if not frames:
            continue
        key = (item["family"], tag)
        bucket = pooled.setdefault(key, {"files": [], "frames": {}})
        bucket["files"].append(item["path"])
        for metric, frame in frames.items():
            prev = bucket["frames"].get(metric)
            combined = frame if prev is None else pd.concat([prev, frame])
            combined = combined.sort_values("game_id").drop_duplicates("game_id", keep="last")
            bucket["frames"][metric] = combined
    batches = {}
    skipped = []
    recorded_plan = []
    for (family, tag), bucket in sorted(pooled.items()):
        cells = []
        seasons_all = set()
        for metric in METRICS:
            frame = bucket["frames"].get(metric)
            if frame is None or frame.empty:
                continue
            frame = frame.dropna(subset=["season", "diff"])
            frame["season"] = frame["season"].astype(int)
            if len(frame) < MIN_PAIRED_GAMES:
                skipped.append(
                    {
                        "family": family,
                        "label": tag,
                        "metric": metric,
                        "games": len(frame),
                        "reason": "below minimum paired games",
                    }
                )
                continue
            boot = season_bootstrap_interval(
                frame["diff"].to_numpy(),
                frame["season"].to_numpy(),
                args.draws,
                args.seed,
            )
            if boot is None:
                skipped.append(
                    {
                        "family": family,
                        "label": tag,
                        "metric": metric,
                        "games": len(frame),
                        "reason": "fewer than two seasons",
                    }
                )
                continue
            low, high, pplus = boot
            scale = SCALE_OF[metric]
            effect = float(frame["diff"].mean() * scale)
            seasons_all.update(frame["season"].unique().tolist())
            words = (
                "ahead of the market"
                if effect > 0
                else "behind the market"
                if effect < 0
                else "level with the market"
            )
            cells.append(
                {
                    "name": f"{slug(family)}__{slug(tag)}__{metric}_backfill",
                    "description": (
                        f"ENG-46 unit 2 backfill: {family} {tag} on {metric}, "
                        f"{len(bucket['files'])} preserved predictions files pooled "
                        f"with game_id dedupe (latest file wins), paired per game "
                        f"with the market baseline, season-block bootstrap "
                        f"{args.draws} draws"
                    ),
                    "effect": effect,
                    "effect_units": UNIT_OF[metric],
                    "season_start": int(frame["season"].min()),
                    "season_end": int(frame["season"].max()),
                    "standard_error": (high - low) * scale / 3.92,
                    "interval_low": low * scale,
                    "interval_high": high * scale,
                    "probability_positive": pplus,
                    "sample_games": len(frame),
                    "sample_blocks": int(frame["season"].nunique()),
                    "plain_summary": PLAIN_OF[metric].format(words=words),
                }
            )
        if not cells:
            continue
        batch = {
            "family": family,
            "source": "artifacts/every_metric/20260916T152725Z",
            "classification": "unresolved_below_power",
            "league": "nfl",
            "category": "modeling",
            "classification_evidence": (
                "backfilled sibling metrics on preserved predictions; no "
                "adjudication in the backfill"
            ),
            "notes": (
                "ENG-46 unit 2 backfilled cell (flagged backfilled in the name); "
                "new metrics on old looks, not counted as new looks"
            ),
            "cells": cells,
        }
        batch_path = os.path.join(out_dir, f"batch_{slug(family)}.json")
        with open(batch_path, "w", encoding="utf-8") as fh:
            json.dump(batch, fh, indent=2, sort_keys=True)
            fh.write("\n")
        batches[family] = batch_path
        recorded_plan.extend([cell["name"] for cell in cells])
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "unit1_artifact": "artifacts/every_metric/20260916T152725Z",
        "families": sorted(batches),
        "cells_planned": len(recorded_plan),
        "skipped": skipped,
        "dry": args.dry,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    results = {"artifact": os.path.relpath(out_dir, REPO_ROOT), "summary": summary}
    if not args.dry:
        uv = os.path.join(REPO_ROOT, ".tools", "uv.exe")
        outcomes = []
        for family in sorted(batches):
            proc = subprocess.run(
                [
                    uv,
                    "run",
                    "--no-sync",
                    "nfl-ats",
                    "weak-signals",
                    "record",
                    "--batch",
                    batches[family],
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            outcomes.append(
                {
                    "family": family,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[-500:],
                    "stderr": proc.stderr[-500:],
                }
            )
        results["record_outcomes"] = outcomes
        with open(os.path.join(out_dir, "record_outcomes.json"), "w", encoding="utf-8") as fh:
            json.dump(outcomes, fh, indent=2, sort_keys=True)
            fh.write("\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
