"""Frozen Lane H M1/M1b research; docs/big_spread_lattice.md."""

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

from nfl_ats.conditional_margin import predict_conditional_margin
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

OUT = common.REPO / "artifacts/research/laneH"
ARMS = {"M1": "conditional_margin_lattice", "M1b": "conditional_margin_lattice_keyside"}


def side(line: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(line.gt(0), "home_favourite", np.where(line.lt(0), "home_underdog", "pickem")),
        index=line.index,
    )


def hybrid(
    history: pd.DataFrame,
    targets: pd.DataFrame,
    served: np.ndarray,
    method: str,
    served_push: np.ndarray | None = None,
) -> pd.DataFrame:
    """Preserve small buckets bit-for-bit and use frozen K on big buckets."""
    result = targets.copy()
    result["home_cover_probability"] = np.asarray(served)
    result["push_probability"] = 0.0 if served_push is None else np.asarray(served_push)
    big = common.spread_bucket(result.spread_line).isin(HOME_SIDE_OFFSET_BUCKETS)
    if big.any():
        mapped = predict_conditional_margin(history, result.loc[big], method=method)
        for col in ("home_cover_probability", "push_probability"):
            result.loc[big, col] = mapped[col]
    return result


def build_centers(archive: pd.DataFrame) -> pd.DataFrame:
    """Lane K's frozen sample recipe, recomputed rather than trusting old artifacts."""
    features = regular_season_rows(pd.read_parquet(common.FEATURES)).copy()
    features["gameday"] = pd.to_datetime(features.gameday)
    completed = features.loc[features.result.notna()]
    targets = completed.loc[completed.season.between(2011, 2025)]
    batches = []
    for (season, week), batch in targets.groupby(["season", "week"], sort=True):
        training = completed.loc[(completed.gameday + pd.Timedelta(days=1)).lt(batch.gameday.min())]
        if len(training) < 500:
            continue
        scoring = batch.merge(
            archive[["game_id", "tue_open_home_spread"]], on="game_id", how="left"
        )
        scoring["spread_line"] = scoring.tue_open_home_spread.fillna(scoring.spread_line)
        model = fit_margin_model(training, target="market_residual", feature_profile="weak_stack")
        pred = model.predict(scoring, probability_method="gaussian_median")
        rows = scoring[["game_id", "season", "week", "gameday", "spread_line", "result"]].copy()
        rows["predicted_margin"] = pred.predicted_margin.to_numpy() + float(
            np.median(model.residuals)
        )
        batches.append(rows)
        if int(week) == 1:
            print(f"K history built {season}", flush=True)
    stream = pd.concat(batches, ignore_index=True)
    common.table(stream, OUT / "centers.parquet")
    return stream


def map_replay() -> None:
    frame = pd.read_parquet(OUT / "replay.parquet")
    verify_replay(frame.p_S3.to_numpy(), frame.home_cover_probability_at_open.to_numpy())
    history = build_centers(frame)
    targets = history.set_index("game_id").loc[frame.game_id].reset_index()
    # Target point comes from the exact served replay, including its median center.
    targets["predicted_margin"] = frame.center_S3.to_numpy()
    for arm, method in ARMS.items():
        mapped = hybrid(history, targets, frame.p_S3.to_numpy(), method, frame.push_S3.to_numpy())
        frame[f"p_{arm}"] = mapped.home_cover_probability.to_numpy()
        frame[f"push_{arm}"] = mapped.push_probability.to_numpy()
        frame[f"offset_{arm}"] = frame.offset_S3
        frame[f"residual_{arm}"] = frame.residual_S3
    common.table(frame, OUT / "replay.parquet")


def verify_replay(actual: np.ndarray, served: np.ndarray) -> float:
    """Fail closed before challenger construction, including missing probabilities."""
    if not np.isfinite(actual).all() or not np.isfinite(served).all():
        raise ValueError("STOP: S3 replay contains missing probabilities")
    gap = float(np.max(np.abs(actual - served)))
    if gap > 1e-9:
        raise ValueError(f"STOP: S3 replay mismatch: {gap}")
    return gap


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
        ]:
            prediction = model.predict(
                scoring, probability_method="gaussian_median", center_offset=shift
            )
            row[f"p_{arm}"] = prediction.home_cover_probability.to_numpy()
            row[f"push_{arm}"] = prediction.push_probability.to_numpy()
            row[f"offset_{arm}"] = shift
            row[f"residual_{arm}"] = prediction.predicted_market_residual.to_numpy()
            row[f"center_{arm}"] = prediction.predicted_margin.to_numpy() + float(
                np.median(model.residuals)
            )
        verify_replay(row.p_S3.to_numpy(), group.home_cover_probability_at_open.to_numpy())
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
            "predeclaration_sha256": sha256_file(common.REPO / "docs/big_spread_lattice.md"),
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
        candidate["residual_at_open"] = frame.residual_at_open
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
            active_model_id=f"research_laneH_{arm}",
            research_scope="Opener only; close columns retain S3, not candidate evidence.",
        )
        metadata["active_model_config"] = {
            **metadata["active_model_config"],
            "model_id": f"research_laneH_{arm}",
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
        (lane_s.cell_name("cell", "bucket", str(b), "all"), g)
        for b, g in frame.groupby("bucket", observed=True)
    ]
    for arm in ARMS:
        for label, group in groups:
            for kind in ("standalone", "card"):
                cp = group[f"p_{arm}"].ge(0.5) if kind == "standalone" else group[f"card_{arm}"]
                bp = group.p_S3.ge(0.5) if kind == "standalone" else group.card_S3
                cells[f"m1_{arm.lower()}_{label}_{kind}"] = common.comparison(
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
                cells[f"m1_{arm.lower()}_{label}_{kind}"] = {
                    **lane_s.metric(valid, bl - cl),
                    "candidate": float(cl.mean()),
                    "baseline": float(bl.mean()),
                }
    push_rows = []
    push_group = frame.loc[frame.tue_open_home_spread.abs().eq(3)].reset_index(drop=True)
    truth_push = push_group.margin_vs_open.eq(0).to_numpy(float)
    for arm in ("S3", *ARMS):
        predicted = push_group[f"push_{arm}"].to_numpy()
        push_rows.append(
            {
                "arm": arm,
                "n": len(push_group),
                "predicted": float(predicted.mean()),
                "realized": float(truth_push.mean()),
            }
        )
        if arm != "S3":
            base_loss = (push_group.push_S3.to_numpy() - truth_push) ** 2
            candidate_loss = (predicted - truth_push) ** 2
            cells[f"m1_{arm.lower()}_push_at_3_brier"] = {
                **lane_s.metric(push_group, base_loss - candidate_loss),
                "candidate": float(candidate_loss.mean()),
                "baseline": float(base_loss.mean()),
            }
    write_stamped_artifact({"rows": push_rows}, OUT / "push_calibration.json")
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
    rows = scoring[["game_id", "home_team", "away_team", "spread_line"]].copy()
    rows["p_S3"] = recorded.home_cover_probability.to_numpy()
    changes = {}
    history = pd.read_parquet(OUT / "centers.parquet")
    targets = scoring.copy()
    targets["predicted_margin"] = base.predicted_margin.to_numpy() + float(
        np.median(model.residuals)
    )
    for arm, method in ARMS.items():
        prediction = hybrid(history, targets, rows.p_S3.to_numpy(), method)
        rows[f"p_{arm}"] = prediction.home_cover_probability.to_numpy()
        changed = rows[f"p_{arm}"].ge(0.5).ne(rows.p_S3.ge(0.5))
        changes[arm] = rows.loc[changed].to_dict(orient="records")
    from nfl_ats.four_overlay_composition import apply_four_overlay_composition_for_publication
    from nfl_ats.snapshots import latest_snapshot, load_snapshot

    schedules, _ = load_snapshot(latest_snapshot(common.REPO / "data/raw"))
    card_changes = {}
    for arm in ("S3", *ARMS):
        candidate = scoring.copy()
        candidate["home_cover_probability"] = rows[f"p_{arm}"].to_numpy()
        composed = apply_four_overlay_composition_for_publication(
            candidate, schedules, common.REPO / "data"
        ).overlaid_predictions
        rows[f"card_home_{arm}"] = composed.home_cover_probability.ge(0.5).to_numpy()
        if arm != "S3":
            changed = rows[f"card_home_{arm}"].ne(rows.card_home_S3)
            card_changes[arm] = rows.loc[changed].to_dict(orient="records")
    common.table(rows, OUT / "week1.parquet")
    write_stamped_artifact(
        {
            "forecast": str(forecast),
            "sidecar_replay_gap": gap,
            "games": len(rows),
            "changes": changes,
            "card_changes": card_changes,
        },
        OUT / "week1.json",
    )


def record() -> None:
    lane_s.FAMILY = "mod18_conditional_margin_v1"
    lane_s.DISCOUNT = (
        "Post-hoc big-spread restriction on the mined archive lanes K and S "
        "were selected on; correlated siblings, not independent confirmation."
    )
    lane_s.SUMMARIES["S1"] = (
        "On big spreads, read the home team's chance from prior integer football margins "
        "near its corrected prediction; keep the existing read on small spreads."
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
    parser.add_argument(
        "--stage", choices=("replay", "map", "score", "week1", "record"), required=True
    )
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
        elif args.stage == "map":
            map_replay()
        elif args.stage == "score":
            score(match[1])
        elif args.stage == "week1":
            week1(active, match[1])
        else:
            record()


if __name__ == "__main__":
    main()
