"""Frozen Lane G S4/S4b research; docs/home_side_side_aware.md."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import home_side_location_opener_eval as lane_s
import numpy as np
import pandas as pd
import spread_regime_opener_eval as common
from threadpoolctl import threadpool_limits

from nfl_ats.home_side_location import (
    HOME_SIDE_OFFSET_BUCKETS,
    archive_prior_stream,
    fit_home_side_offsets,
    prior_rows_before,
)
from nfl_ats.margin import fit_margin_model
from nfl_ats.modeling import regular_season_rows
from nfl_ats.provenance import sha256_file, write_stamped_artifact
from nfl_ats.public_board import find_matching_opener_evaluation

OUT = common.REPO / "artifacts/research/laneG"
ARMS = {"S4": 100.0, "S4b": 50.0}


def side(line: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(line.gt(0), "home_favourite", np.where(line.lt(0), "home_underdog", "pickem")),
        index=line.index,
    )


def offsets(prior: pd.DataFrame, lines: pd.Series, weight: float) -> np.ndarray:
    completed = prior.loc[prior.result.notna() & prior.point_incumbent.notna()]
    buckets = common.spread_bucket(completed.spread_line)
    sides = side(completed.spread_line)
    target_buckets = common.spread_bucket(lines)
    target_sides = side(lines)
    result = np.zeros(len(lines))
    for bucket in HOME_SIDE_OFFSET_BUCKETS:
        for home_side in ("home_favourite", "home_underdog"):
            mask = buckets.eq(bucket) & sides.eq(home_side)
            value = (
                completed.loc[mask, "result"] - completed.loc[mask, "point_incumbent"]
            ).sum() / (int(mask.sum()) + weight)
            result[(target_buckets.eq(bucket) & target_sides.eq(home_side)).to_numpy()] = value
    return result


def replay(archive: pd.DataFrame, active: dict, archive_path: Path) -> None:
    if sha256_file(common.FEATURES) != active["feature_table_sha256"]:
        raise ValueError("Feature digest changed")
    features = regular_season_rows(pd.read_parquet(common.FEATURES))
    features["gameday"] = pd.to_datetime(features.gameday)
    completed = features.loc[features.result.notna()]
    stream = archive_prior_stream(archive)
    batches = []
    for (season, week), group in archive.groupby(["season", "week"], sort=True):
        scoring = features.set_index("game_id").loc[group.game_id].reset_index()
        scoring["spread_line"] = group.tue_open_home_spread.to_numpy()
        training = completed.loc[completed.gameday.lt(scoring.gameday.min())]
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name=active["regressor"],
            feature_profile=active["feature_profile"],
            ridge_alpha=active["ridge_alpha"],
        )
        prior = prior_rows_before(stream, int(season), int(week))
        row = group[["game_id"]].copy()
        s3 = fit_home_side_offsets(prior).offset_for(scoring.spread_line).to_numpy()
        for arm, shift in [
            ("S3", s3),
            *[(a, offsets(prior, scoring.spread_line, w)) for a, w in ARMS.items()],
        ]:
            prediction = model.predict(
                scoring, probability_method="gaussian_median", center_offset=shift
            )
            row[f"p_{arm}"] = prediction.home_cover_probability.to_numpy()
            row[f"offset_{arm}"] = shift
            row[f"residual_{arm}"] = prediction.predicted_market_residual.to_numpy()
        gap = np.max(np.abs(row.p_S3.to_numpy() - group.home_cover_probability_at_open.to_numpy()))
        if gap > 1e-9:
            raise ValueError(f"STOP: S3 replay mismatch {season}/{week}: {gap}")
        batches.append(row)
        if int(week) == 1:
            print(f"S3 replay verified {season}", flush=True)
    result = archive.merge(pd.concat(batches), on="game_id", validate="one_to_one")
    common.table(result, OUT / "replay.parquet")
    write_stamped_artifact(
        {
            "archive": str(archive_path),
            "active_model_id": active["model_id"],
            "feature_sha256": sha256_file(common.FEATURES),
            "predeclaration_sha256": sha256_file(common.REPO / "docs/home_side_side_aware.md"),
            "max_probability_gap": float(
                (result.p_S3 - result.home_cover_probability_at_open).abs().max()
            ),
            "max_offset_gap": float(
                (result.offset_S3 - result.home_side_offset_at_open).abs().max()
            ),
        },
        OUT / "reproduction.json",
    )


def score(archive_path: Path) -> None:
    frame = pd.read_parquet(OUT / "replay.parquet")
    _, base_card = common.composed_picks(frame, archive_path / "per_game.parquet")
    frame["card_S3"] = base_card
    for arm in ARMS:
        candidate = frame.copy()
        candidate["home_cover_probability_at_open"] = candidate[f"p_{arm}"]
        candidate["residual_at_open"] = candidate[f"residual_{arm}"]
        candidate["home_side_offset_at_open"] = candidate[f"offset_{arm}"]
        for suffix, pick in [
            ("", candidate.residual_at_open.gt(0)),
            ("_probability_rule", candidate.home_cover_probability_at_open.ge(0.5)),
        ]:
            candidate[f"pick_home_at_open{suffix}"] = pick
            candidate[f"correct_at_open{suffix}"] = (
                pick.eq(candidate.margin_vs_open.gt(0))
                .astype(float)
                .where(candidate.margin_vs_open.ne(0))
            )
        directory = OUT / "opener_evaluation" / arm
        common.table(candidate, directory / "per_game.parquet")
        metadata = json.loads((archive_path / "metadata.json").read_text())
        metadata.update(
            research_arm=arm,
            active_model_id=f"research_laneG_{arm}",
            research_scope="Opener only; close columns retain S3, not candidate evidence.",
        )
        metadata["active_model_config"] = {
            **metadata["active_model_config"],
            "model_id": f"research_laneG_{arm}",
        }
        write_stamped_artifact(metadata, directory / "metadata.json")
        lane_s.original_cli(
            OUT,
            "overlay-composition",
            "--per-game-artifact",
            str(directory / "per_game.parquet"),
            "--bootstrap-samples",
            "20000",
            "--bootstrap-seed",
            "20260817",
        )
        _, card = common.composed_picks(candidate, directory / "per_game.parquet")
        frame[f"card_{arm}"] = card
    frame["bucket"] = common.spread_bucket(frame.tue_open_home_spread)
    frame["home_side"] = side(frame.tue_open_home_spread)
    cells = {}
    groups = [("overall", frame)]
    groups += [(f"season_{s}", g) for s, g in frame.groupby("season")]
    groups += [
        (lane_s.cell_name("cell", "bucket", str(b), str(s)), g)
        for (b, s), g in frame.groupby(["bucket", "home_side"], observed=True)
    ]
    for arm in ARMS:
        for label, group in groups:
            for kind in ("standalone", "card"):
                cp = group[f"p_{arm}"].ge(0.5) if kind == "standalone" else group[f"card_{arm}"]
                bp = group.p_S3.ge(0.5) if kind == "standalone" else group.card_S3
                cells[f"s4_{arm.lower()}_{label}_{kind}"] = common.comparison(
                    group, cp.to_numpy(), bp.to_numpy()
                )
            valid = group.loc[
                group.margin_vs_open.notna() & group.margin_vs_open.ne(0)
            ].reset_index(drop=True)
            y = valid.margin_vs_open.gt(0).to_numpy(float)
            p, b = [np.clip(valid[f"p_{a}"].to_numpy(), 1e-9, 1 - 1e-9) for a in (arm, "S3")]
            for kind, cl, bl in [
                ("brier", (p - y) ** 2, (b - y) ** 2),
                (
                    "log_loss",
                    -(y * np.log(p) + (1 - y) * np.log1p(-p)),
                    -(y * np.log(b) + (1 - y) * np.log1p(-b)),
                ),
            ]:
                cells[f"s4_{arm.lower()}_{label}_{kind}"] = {
                    **lane_s.metric(valid, bl - cl),
                    "candidate": float(cl.mean()),
                    "baseline": float(bl.mean()),
                }
    common.table(frame, OUT / "scored.parquet")
    write_stamped_artifact(cells, OUT / "cells.json")
    print(json.dumps({k: v for k, v in cells.items() if "overall" in k}, indent=2), flush=True)


def week1(active: dict, archive_path: Path) -> None:
    forecast = common.REPO / "artifacts" / active["weekly_forecast"]["artifact"]
    sidecar = json.loads((forecast / "home_side_offset.json").read_text())
    scoring = pd.read_csv(forecast / "predictions.csv")
    scoring = scoring.loc[scoring.method.eq("market_residual")].reset_index(drop=True)
    recorded = pd.DataFrame(sidecar["games"]).set_index("game_id").loc[scoring.game_id]
    features = regular_season_rows(pd.read_parquet(common.FEATURES))
    training = features.loc[
        features.result.notna()
        & pd.to_datetime(features.gameday).lt(pd.to_datetime(scoring.gameday).min())
    ]
    model = fit_margin_model(
        training,
        target="market_residual",
        model_name=active["regressor"],
        feature_profile=active["feature_profile"],
        ridge_alpha=active["ridge_alpha"],
    )
    base = model.predict(
        scoring,
        probability_method="gaussian_median",
        center_offset=recorded.home_side_offset.to_numpy(),
    )
    gap = float(
        np.abs(
            base.home_cover_probability.to_numpy() - recorded.home_cover_probability.to_numpy()
        ).max()
    )
    if gap > 1e-9:
        raise ValueError(f"Week 1 sidecar replay mismatch: {gap}")
    prior = prior_rows_before(
        archive_prior_stream(pd.read_parquet(archive_path / "per_game.parquet")), 2026, 1
    )
    rows = scoring[["game_id", "home_team", "away_team", "spread_line"]].copy()
    rows["p_S3"] = recorded.home_cover_probability.to_numpy()
    changes = {}
    for arm, weight in ARMS.items():
        shift = offsets(prior, scoring.spread_line, weight)
        prediction = model.predict(
            scoring, probability_method="gaussian_median", center_offset=shift
        )
        rows[f"p_{arm}"] = prediction.home_cover_probability.to_numpy()
        rows[f"offset_{arm}"] = shift
        changed = rows[f"p_{arm}"].ge(0.5).ne(rows.p_S3.ge(0.5))
        changes[arm] = rows.loc[changed].to_dict(orient="records")
    common.table(rows, OUT / "week1.parquet")
    write_stamped_artifact(
        {
            "forecast": str(forecast),
            "sidecar_replay_gap": gap,
            "games": len(rows),
            "changes": changes,
        },
        OUT / "week1.json",
    )


def record() -> None:
    lane_s.DISCOUNT = (
        "Further post-hoc side split on the mined archive on which S2/S3 were selected; "
        "correlated sensitivity; not independent confirmation."
    )
    lane_s.SUMMARIES["S1"] = (
        "Adjust the home team's predicted margin using earlier games with the same spread "
        "size and home favourite or underdog status."
    )
    cells = json.loads((OUT / "cells.json").read_text())
    for name, m in cells.items():
        if not isinstance(m, dict) or "delta" not in m:
            continue
        units = (
            "brier_improvement"
            if name.endswith("brier")
            else "log_loss_improvement"
            if name.endswith("log_loss")
            else "accuracy_points"
        )
        season = int(name.split("season_")[1].split("_")[0]) if "season_" in name else None
        lane_s.record(OUT, name, m, units, season or 2020, season or 2025, OUT / "cells.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("replay", "score", "week1", "record"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["NFL_ATS_ARTIFACTS_DIR"] = str(OUT)
    os.environ["NFL_ATS_REGISTRY_DIR"] = str(
        common.REPO / "registry" if args.stage == "record" else OUT / "registry"
    )
    active = json.loads((common.REPO / "artifacts/active_ats_model.json").read_text())
    match = find_matching_opener_evaluation(common.REPO / "artifacts", active)
    if match is None:
        raise ValueError("No matching opener evaluation")
    with threadpool_limits(limits=1):
        if args.stage == "replay":
            replay(pd.read_parquet(match[1] / "per_game.parquet"), active, match[1])
        elif args.stage == "score":
            score(match[1])
        elif args.stage == "week1":
            week1(active, match[1])
        else:
            record()


if __name__ == "__main__":
    main()
