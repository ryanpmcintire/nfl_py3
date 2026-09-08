"""Frozen lane L experiment; all output lives in an isolated scratch root."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import conditional_margin_opener_eval as lattice_eval
import numpy as np
import pandas as pd
import spread_regime_opener_eval as common

from nfl_ats.conditional_margin import predict_conditional_margin
from nfl_ats.hybrid_margin import HYBRID_MARGIN_METHODS, predict_hybrid_margin
from nfl_ats.provenance import sha256_file, write_stamped_artifact
from nfl_ats.public_board import find_matching_opener_evaluation

FAMILY = "mod18_hybrid_margin_v1"
ARMS = dict(zip(("H1", "H2"), HYBRID_MARGIN_METHODS, strict=True))
DISCOUNT = (
    "Mined archive; inherited model, H/K spread diagnosis, lattice and overlay subset "
    "selection; two correlated hybrids; descriptive reused evidence, not independent confirmation."
)
SUMMARY = (
    "Near common winning margins such as three or seven points, let past games decide "
    "how much those scores should change our chances."
)
original_cli = common.cli


def cli(out: Path, *args: str) -> str:
    arguments = list(args)
    if arguments[:2] == ["weak-signals", "record"]:
        arguments.extend(["--plain-summary", SUMMARY])
    if "--notes" in arguments:
        i = arguments.index("--notes") + 1
        arguments[i] = arguments[i].replace("Five frozen arms", "Two frozen arms")
    if arguments[0] not in {"weak-signals", "rotation"}:
        return original_cli(out, *arguments)
    # Execute the actual parser and handler, retaining exact CLI argv; avoid
    # reimporting the entire application for each diagnostic cell.
    from nfl_ats.cli import main as cli_main

    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            code = cli_main(arguments)
        except SystemExit as exc:
            code = int(str(exc.code))
    path = out / "commands.json"
    commands = json.loads(path.read_text()).get("commands", []) if path.exists() else []
    commands.append(
        {
            "argv": ["nfl-ats", *arguments],
            "returncode": code,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "execution": "nfl_ats.cli.main: real parser and handler, in process",
        }
    )
    write_stamped_artifact({"commands": commands}, path)
    if code:
        raise RuntimeError(stdout.getvalue() + stderr.getvalue())
    return stdout.getvalue()


common.cli = cli
common.FAMILY = lattice_eval.FAMILY = FAMILY
common.DISCOUNT = lattice_eval.DISCOUNT = DISCOUNT
common.ARMS = tuple(ARMS)
lattice_eval.ARMS = ARMS


def build(out: Path, archive: pd.DataFrame, centers_path: Path) -> None:
    active = json.loads((common.REPO / "artifacts/active_ats_model.json").read_text())
    assert active["model_id"] == "a4c757efd2525da6"
    assert sha256_file(common.FEATURES) == active["feature_table_sha256"]
    centers = pd.read_parquet(centers_path)
    centers["p_smooth"] = centers.game_id.map(
        archive.set_index("game_id").home_cover_probability_at_open
    ).fillna(centers.p_base)
    overlapping = centers.loc[centers.game_id.isin(archive.game_id)].set_index("game_id")
    archived = archive.set_index("game_id").reindex(overlapping.index)
    np.testing.assert_allclose(overlapping.spread_line, archived.tue_open_home_spread)
    point_difference = float((overlapping.residual_base - archived.residual_at_open).abs().max())
    np.testing.assert_allclose(overlapping.result, archived.result)
    targets = centers.loc[centers.season.gt(centers.season.min())].copy()
    mapped = predict_conditional_margin(centers, targets)
    targets["p_lattice"] = mapped.home_cover_probability
    targets["lattice_push"] = mapped.push_probability
    common.table(targets, out / "component_history.parquet")
    output = targets.loc[targets.season.ge(2020)].copy()
    for arm, method in ARMS.items():
        fitted = predict_hybrid_margin(targets, output, method=method)
        output[f"p_{arm}"] = fitted.home_cover_probability
        for column in fitted.columns:
            if column.startswith("hybrid_"):
                output[f"{column}_{arm}"] = fitted[column]
        output[f"push_probability_{arm}"] = fitted.hybrid_weight * output.lattice_push
    output["residual_base"] = output.game_id.map(
        archive.set_index("game_id").residual_at_open
    ).fillna(output.residual_base)
    common.table(output, out / "stream.parquet")
    write_stamped_artifact(
        {
            "active": active,
            "archive": str(common.ARCHIVE),
            "centers": str(centers_path),
            "centers_sha256": sha256_file(centers_path),
            "maximum_archive_point_difference": point_difference,
            "predeclaration_sha256": sha256_file(common.REPO / "docs/hybrid_margin_mapping.md"),
        },
        out / "inputs.json",
    )


def diagnosis(out: Path, archive: pd.DataFrame, stream: pd.DataFrame) -> None:
    frame = archive.merge(stream, on=["game_id", "season", "week"], suffixes=("", "_stream"))
    frame["bucket"] = common.spread_bucket(frame.tue_open_home_spread)
    frame["point_error"] = frame.result - (frame.tue_open_home_spread + frame.residual_at_open)
    frame["conditioning_error"] = frame.result - frame.predicted_margin
    frame["favourite_error"] = np.sign(frame.tue_open_home_spread) * frame.point_error
    pick = frame.home_cover_probability_at_open.ge(0.5)
    frame["pick_side"] = np.where(
        pick.eq(frame.tue_open_home_spread.gt(0)), "favourite", "underdog"
    )
    frame.loc[frame.tue_open_home_spread.eq(0), "pick_side"] = "pickem"
    frame["home_side"] = np.where(
        frame.tue_open_home_spread.gt(0), "home_favourite", "home_underdog"
    )
    frame.loc[frame.tue_open_home_spread.eq(0), "home_side"] = "pickem"
    rows = []
    for bucket, group in frame.groupby("bucket", observed=True):
        selections = {"all": group}
        selections.update({str(s): g for s, g in group.groupby("pick_side")})
        selections.update({str(s): g for s, g in group.groupby("home_side")})
        for side, g in selections.items():
            for name in ("point_error", "conditioning_error", "favourite_error"):
                m = lattice_eval.metric(g.reset_index(drop=True), g[name].to_numpy())
                rows.append({"bucket": str(bucket), "side": side, "metric": name, **m})
            g = g.loc[g.margin_vs_open.ne(0)]
            for arm, col in {
                "smooth": "home_cover_probability_at_open",
                "lattice": "p_lattice",
                **{a: f"p_{a}" for a in ARMS},
            }.items():
                p = g[col]
                gaps = {
                    "home_calibration_gap": g.margin_vs_open.gt(0).astype(float) - p,
                    "pick_calibration_gap": p.ge(0.5).eq(g.margin_vs_open.gt(0)).astype(float)
                    - p.where(p.ge(0.5), 1 - p),
                }
                for name, values in gaps.items():
                    m = lattice_eval.metric(g.reset_index(drop=True), values.to_numpy())
                    rows.append(
                        {"bucket": str(bucket), "side": side, "metric": f"{arm}_{name}", **m}
                    )
    common.table(pd.DataFrame(rows), out / "diagnosis.parquet")


def record_cells(out: Path) -> None:
    # Diagnosis D is a read-only signed bias, not a candidate improvement.
    # Keep its full cells in diagnosis.parquet, outside the effect pool.
    frame = pd.read_parquet(out / "paired.parquet")
    for season, group in frame.groupby("season"):
        for arm in ARMS:
            metrics = common.comparison(
                group,
                group[f"p_{arm}"].ge(0.5).to_numpy(),
                group.pick_home_at_open_probability_rule.to_numpy(),
            )
            lattice_eval.record(
                out,
                arm + "_season",
                metrics,
                "accuracy_points",
                int(str(season)),
                int(str(season)),
                out / "paired.parquet",
            )
    frame = frame.loc[frame.margin_vs_open.ne(0)].reset_index(drop=True)
    frame["bucket"] = common.spread_bucket(frame.tue_open_home_spread)
    for arm in ARMS:
        pick = frame[f"p_{arm}"].ge(0.5)
        frame["side"] = np.where(pick.eq(frame.tue_open_home_spread.gt(0)), "favourite", "underdog")
        frame.loc[frame.tue_open_home_spread.eq(0), "side"] = "pickem"
        for bucket, group in frame.groupby("bucket", observed=True):
            for side in ("all", "favourite", "underdog", "pickem"):
                g = group if side == "all" else group.loc[group.side.eq(side)]
                if g.empty:
                    continue
                m = common.comparison(
                    g,
                    g[f"p_{arm}"].ge(0.5).to_numpy(),
                    g.pick_home_at_open_probability_rule.to_numpy(),
                )
                name = f"{arm}_bucket_{bucket}_{side}".replace(".", "p").replace("+", "plus")
                lattice_eval.record(
                    out, name, m, "accuracy_points", 2020, 2025, out / "paired.parquet"
                )


def week1(out: Path) -> None:
    directories = sorted((out / "artifacts/margin_predictions").glob("2026-week-01-*"))
    scratch = pd.read_csv(directories[-1] / "predictions.csv")
    scratch = scratch.loc[scratch.method.eq("market_residual")].set_index("game_id")
    active = json.loads((common.REPO / "artifacts/active_ats_model.json").read_text())
    current = pd.read_csv(
        common.REPO / "artifacts" / active["weekly_forecast"]["artifact"] / "predictions.csv"
    )
    current = (
        current.loc[current.method.eq("market_residual")]
        .set_index("game_id")
        .reindex(scratch.index)
    )
    np.testing.assert_allclose(
        scratch.home_cover_probability, current.home_cover_probability, atol=1e-12
    )
    history = pd.read_parquet(out / "component_history.parquet")
    targets = history.loc[history.season.eq(2026)].copy()
    indexed = scratch.reindex(targets.game_id)
    np.testing.assert_allclose(targets.p_smooth, indexed.home_cover_probability, atol=1e-12)
    for arm, method in ARMS.items():
        mapped = predict_hybrid_margin(history, targets, method=method)
        candidate = (
            indexed[
                [
                    "season",
                    "week",
                    "gameday",
                    "home_team",
                    "away_team",
                    "spread_line",
                    "predicted_margin",
                ]
            ]
            .reset_index()
            .copy()
        )
        candidate["home_cover_probability"] = mapped.home_cover_probability.to_numpy()
        candidate["push_probability"] = (mapped.hybrid_weight * targets.lattice_push).to_numpy()
        candidate["home_cover_probability_excluding_push"] = (
            candidate.home_cover_probability - 0.5 * candidate.push_probability
        )
        candidate["home_loss_probability"] = (
            1 - candidate.home_cover_probability - 0.5 * candidate.push_probability
        )
        candidate["bet_side"] = np.where(candidate.home_cover_probability.ge(0.5), "home", "away")
        candidate["hybrid_weight"] = mapped.hybrid_weight.to_numpy()
        common.table(candidate, out / "artifacts/margin_predictions" / arm / "predictions.parquet")
    write_stamped_artifact(
        {
            "scratch_cli_artifact": str(directories[-1]),
            "games": len(scratch),
            "maximum_probability_difference": float(
                (scratch.home_cover_probability - current.home_cover_probability).abs().max()
            ),
            "candidate_method": (
                "CLI incumbent followed by explicit prior-only hybrid mapper; "
                "not production integration"
            ),
        },
        out / "week1_cli_check.json",
    )


def replay(out: Path) -> None:
    commands = json.loads((out / "commands.json").read_text())["commands"]
    family = json.loads((out / "registry/rotation_registry.json").read_text())["families"][FAMILY]
    os.environ["NFL_ATS_REGISTRY_DIR"] = str(common.REPO / "registry")
    log = out / "live_replay"
    cli(
        log,
        "rotation",
        "declare",
        "--name",
        FAMILY,
        "--description",
        family["description"],
        "--grade",
        "opener",
        "--acknowledge-mined",
        "--plain-summary",
        SUMMARY,
    )
    cli(log, "rotation", "assign", "--name", FAMILY)
    for command in commands:
        argv = command["argv"]
        if command["returncode"]:
            continue
        if any(value.startswith(FAMILY + "_D_") for value in argv):
            continue  # Invalid diagnostic records are not comparative effects.
        start = next(
            (i for i, value in enumerate(argv) if value in {"weak-signals", "rotation"}), None
        )
        if start is None:
            continue
        cli(log, *argv[start:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--centers", type=Path)
    parser.add_argument(
        "--stage",
        choices=(
            "build",
            "score",
            "losses",
            "composition",
            "diagnosis",
            "cells",
            "week1",
            "replay",
        ),
        required=True,
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(common.REPO):
        raise ValueError("Use an isolated scratch root outside the repository")
    os.environ["NFL_ATS_ARTIFACTS_DIR"] = str(out / "artifacts")
    os.environ["NFL_ATS_REGISTRY_DIR"] = str(out / "registry")
    active = json.loads((common.REPO / "artifacts/active_ats_model.json").read_text())
    match = find_matching_opener_evaluation(common.REPO / "artifacts", active)
    if match is None:
        raise ValueError("No matched opener evaluation")
    common.ARCHIVE = match[1]
    archive = pd.read_parquet(common.ARCHIVE / "per_game.parquet")
    if args.stage == "build":
        build(out, archive, args.centers)
    else:
        stream = pd.read_parquet(out / "stream.parquet")
        if args.stage == "score":
            print(json.dumps(common.score(out, archive, stream), indent=2))
        elif args.stage == "losses":
            lattice_eval.losses(out, archive, stream)
        elif args.stage == "composition":
            lattice_eval.composition(out, archive, stream)
            for arm in ARMS:
                path = out / "artifacts/opener_evaluation" / arm / "metadata.json"
                metadata = json.loads(path.read_text())
                metadata["active_model_id"] = f"research_laneL_{arm}"
                metadata["active_model_config"]["model_id"] = f"research_laneL_{arm}"
                metadata["incumbent_model_id"] = active["model_id"]
                metadata["research_scope"] = (
                    "Only opener columns recomputed; close columns retain incumbent values "
                    "and are not candidate evidence."
                )
                write_stamped_artifact(metadata, path)
        elif args.stage == "diagnosis":
            diagnosis(out, archive, stream)
        elif args.stage == "week1":
            week1(out)
        elif args.stage == "replay":
            replay(out)
        else:
            record_cells(out)


if __name__ == "__main__":
    main()
