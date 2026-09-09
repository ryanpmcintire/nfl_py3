"""Frozen lane T experiment (docs/home_side_mapping.md); isolated scratch root only."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import conditional_margin_opener_eval as lattice_eval
import numpy as np
import pandas as pd
import spread_regime_opener_eval as common
from scipy import stats

from nfl_ats.conditional_margin import predict_conditional_margin
from nfl_ats.home_side_mapping import (
    HOME_SIDE_MAPPING_METHODS,
    home_side,
    predict_home_side_mapping,
)
from nfl_ats.provenance import sha256_file, stamp_sidecar, write_stamped_artifact
from nfl_ats.public_board import find_matching_opener_evaluation

FAMILY = "mod18_home_side_mapping_v1"
ARMS = dict(zip(("T1", "T2"), HOME_SIDE_MAPPING_METHODS, strict=True))
CENTERS_SHA256 = "560de573b9523834a98f881b73d441653ffd986314151ae2b2c2d0bbe36120c0"
DISCOUNT = (
    "Mined archive; inherited active model, H/K/L/P home-side diagnosis, lattice and overlay "
    "subset selection; two correlated arms; descriptive reused evidence, not independent "
    "confirmation."
)
FAMILY_SUMMARY = (
    "Home underdogs on big spreads have covered more often than the model expected. These "
    "two reads keep the same picks line but learn from past games, separately for home "
    "underdogs and home favourites, how confident to be in each side."
)
SUMMARIES = {
    "T1": (
        "When the home team is an underdog or a favourite of a given size, the model has "
        "missed the final margin by a few points on average. This nudges how confident we are "
        "in each side by that past average, without moving the projected margin itself."
    ),
    "T2": (
        "Final margins pile up on three, seven and ten points. This reads how often the home "
        "team covers from past games with the home side in the same spot, home underdog or "
        "home favourite, so the shape near those numbers is learned for each side."
    ),
}
original_cli = common.cli


def cli(out: Path, *args: str) -> str:
    arguments = list(args)
    if arguments[0] not in {"weak-signals", "rotation"}:
        return original_cli(out, *arguments)
    if arguments[:2] == ["weak-signals", "record"]:
        name = arguments[arguments.index("--name") + 1]
        arm = next((a for a in ARMS if f"{FAMILY}_{a}_" in name), None)
        arguments.extend(["--plain-summary", SUMMARIES.get(arm or "", FAMILY_SUMMARY)])
    for flag in ("--description", "--notes"):
        if flag in arguments:
            i = arguments.index(flag) + 1
            arguments[i] = (
                arguments[i]
                .replace("Frozen spread-regime", "Frozen home-side mapping")
                .replace("Conditional integer margins", "Home-side mapping")
                .replace("Five frozen arms", "Two frozen arms")
            )
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


def setup(out: Path, active: dict) -> None:
    write_stamped_artifact(active, out / "artifacts/active_ats_model.json")
    (out / "registry").mkdir(parents=True, exist_ok=True)
    for name in ("rotation_registry.json", "weak_signals.json"):
        shutil.copyfile(common.REPO / "registry" / name, out / "registry" / name)
        stamp_sidecar(out / "registry" / name)
    declare(out)


def declare(out: Path) -> None:
    cli(
        out,
        "rotation",
        "declare",
        "--name",
        FAMILY,
        "--description",
        (
            "T1 home-side shrunken centre shift of the gaussian_median read and T2 same-side "
            "K1 integer-margin lattice; docs/home_side_mapping.md; inherited H/K/L/P "
            "diagnosis, model and overlay selection; mined descriptive reuse"
        ),
        "--grade",
        "opener",
        "--acknowledge-mined",
        "--plain-summary",
        FAMILY_SUMMARY,
    )
    cli(out, "rotation", "assign", "--name", FAMILY)


def build(out: Path, archive: pd.DataFrame, centers_path: Path, active: dict) -> None:
    if active["model_id"] != "a4c757efd2525da6":
        raise ValueError("Active model changed")
    if sha256_file(common.FEATURES) != active["feature_table_sha256"]:
        raise ValueError("Feature digest changed")
    if sha256_file(centers_path) != CENTERS_SHA256:
        raise ValueError("Cached centre stream digest changed")
    centers = pd.read_parquet(centers_path)
    archived = archive.set_index("game_id")
    centers["p_smooth"] = centers.game_id.map(archived.home_cover_probability_at_open).fillna(
        centers.p_base
    )
    z = stats.norm.ppf(centers.p_base)
    estimate = pd.Series(
        np.where(np.abs(z) > 1e-4, (centers.predicted_margin - centers.spread_line) / z, np.nan),
        index=centers.index,
    )
    weekly = estimate.groupby([centers.season, centers.week]).agg(["median", "min", "max"])
    scale_spread = float((weekly["max"] - weekly["min"]).max())
    if scale_spread > 1e-8:
        raise ValueError(f"Gaussian scale is not constant within a week: {scale_spread}")
    centers["smooth_scale"] = (
        centers.set_index(["season", "week"]).index.map(weekly["median"]).to_numpy()
    )
    overlapping = centers.loc[centers.game_id.isin(archive.game_id)].set_index("game_id")
    paired = archived.reindex(overlapping.index)
    np.testing.assert_allclose(overlapping.spread_line, paired.tue_open_home_spread)
    np.testing.assert_allclose(overlapping.result, paired.result)
    point_difference = float((overlapping.residual_base - paired.residual_at_open).abs().max())
    targets = centers.loc[centers.season.ge(2020)].copy()
    reference = predict_conditional_margin(centers, targets)
    targets["p_K1"] = reference.home_cover_probability
    targets["push_probability_K1"] = reference.push_probability
    for arm, method in ARMS.items():
        mapped = predict_home_side_mapping(centers, targets, method=method)
        targets[f"p_{arm}"] = mapped.home_cover_probability
        for column in mapped.columns.difference(targets.columns):
            if column.startswith(("home_side_", "lattice_")):
                targets[f"{column}_{arm}"] = mapped[column]
        targets[f"push_probability_{arm}"] = (
            mapped.push_probability if "push_probability" in mapped else 0.0
        )
    targets["residual_base"] = targets.game_id.map(archived.residual_at_open).fillna(
        targets.residual_base
    )
    common.table(targets, out / "stream.parquet")
    write_stamped_artifact(
        {
            "active": active,
            "archive": str(common.ARCHIVE),
            "centers": str(centers_path),
            "centers_sha256": CENTERS_SHA256,
            "maximum_archive_point_difference": point_difference,
            "maximum_within_week_scale_spread": scale_spread,
            "scale_summary": weekly["median"].describe().to_dict(),
            "predeclaration_sha256": sha256_file(common.REPO / "docs/home_side_mapping.md"),
            "t2_side_used_share": float(targets.lattice_side_used_T2.mean()),
        },
        out / "inputs.json",
    )


def home_split(out: Path, archive: pd.DataFrame) -> None:
    frame = pd.read_parquet(out / "paired.parquet")
    frame = frame.loc[frame.margin_vs_open.ne(0)].reset_index(drop=True)
    frame["bucket"] = common.spread_bucket(frame.tue_open_home_spread)
    frame["home_side"] = home_side(frame.tue_open_home_spread.to_numpy(dtype=float))
    columns = {
        "incumbent": "home_cover_probability_at_open",
        "K1": "p_K1",
        **{a: f"p_{a}" for a in ARMS},
    }
    rows = []
    for bucket, group in frame.groupby("bucket", observed=True):
        for side in ("all", *sorted(group.home_side.unique())):
            g = group if side == "all" else group.loc[group.home_side.eq(side)]
            g = g.reset_index(drop=True)
            t = g.margin_vs_open.gt(0).astype(float)
            base = g.home_cover_probability_at_open
            for arm, column in columns.items():
                p = g[column]
                pick = p.ge(0.5)
                correct = pick.eq(t.eq(1)).astype(float)
                confidence = p.where(pick, 1 - p)
                gap = lattice_eval.metric(g, (t - p).to_numpy())
                pick_gap = lattice_eval.metric(g, (correct - confidence).to_numpy())
                row = {
                    "arm": arm,
                    "bucket": str(bucket),
                    "home_side": side,
                    "n": len(g),
                    "home_cover_rate": float(t.mean()),
                    "stated_home_probability": float(p.mean()),
                    "home_gap_pp": 100 * gap["delta"],
                    "home_gap_lower": 100 * gap["lower"],
                    "home_gap_upper": 100 * gap["upper"],
                    "home_gap_probability_positive": gap["probability_positive"],
                    "accuracy": float(correct.mean()),
                    "confidence": float(confidence.mean()),
                    "pick_gap_pp": 100 * pick_gap["delta"],
                    "pick_gap_lower": 100 * pick_gap["lower"],
                    "pick_gap_upper": 100 * pick_gap["upper"],
                }
                if arm in ARMS:
                    accuracy = common.comparison(
                        g, pick.to_numpy(), g.pick_home_at_open_probability_rule.to_numpy()
                    )
                    brier = lattice_eval.metric(g, ((base - t) ** 2 - (p - t) ** 2).to_numpy())
                    row.update(
                        accuracy_delta=accuracy["delta"],
                        accuracy_lower=accuracy["lower"],
                        accuracy_upper=accuracy["upper"],
                        accuracy_probability_positive=accuracy["probability_positive"],
                        flips=accuracy["flips"],
                        brier_improvement=brier["delta"],
                        brier_lower=brier["lower"],
                        brier_upper=brier["upper"],
                        brier_probability_positive=brier["probability_positive"],
                    )
                    name = f"{arm}_home_{bucket}_{side}".replace(".", "p").replace("+", "plus")
                    lattice_eval.record(
                        out,
                        name + "_accuracy",
                        accuracy,
                        "accuracy_points",
                        2020,
                        2025,
                        out / "home_split.parquet",
                    )
                    lattice_eval.record(
                        out,
                        name + "_brier",
                        brier,
                        "brier_improvement",
                        2020,
                        2025,
                        out / "home_split.parquet",
                    )
                rows.append(row)
    common.table(pd.DataFrame(rows), out / "home_split.parquet")


def record_seasons(out: Path) -> None:
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


def week1(out: Path, active: dict) -> None:
    command = [
        str(common.REPO / ".tools/uv.exe"),
        "run",
        "--no-sync",
        "nfl-ats",
        "margin-predict",
        "--season",
        "2026",
        "--week",
        "1",
        "--probability-method",
        "gaussian_median",
    ]
    result = subprocess.run(command, cwd=common.REPO, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    directories = sorted((out / "artifacts/margin_predictions").glob("2026-week-01-*"))
    scratch = pd.read_csv(directories[-1] / "predictions.csv")
    scratch = scratch.loc[scratch.method.eq("market_residual")].set_index("game_id")
    current = pd.read_csv(
        common.REPO / "artifacts" / active["weekly_forecast"]["artifact"] / "predictions.csv"
    )
    current = current.loc[current.method.eq("market_residual")].set_index("game_id")
    current = current.reindex(scratch.index)
    cli_difference = float(
        (scratch.home_cover_probability - current.home_cover_probability).abs().max()
    )
    stream = pd.read_parquet(out / "stream.parquet")
    week = stream.loc[stream.season.eq(2026)].set_index("game_id").reindex(scratch.index)
    np.testing.assert_allclose(week.p_smooth, current.home_cover_probability, atol=1e-12)
    for arm in ARMS:
        candidate = scratch[
            [
                "season",
                "week",
                "gameday",
                "home_team",
                "away_team",
                "spread_line",
                "predicted_margin",
            ]
        ].reset_index()
        candidate["home_cover_probability"] = week[f"p_{arm}"].to_numpy()
        candidate["push_probability"] = week[f"push_probability_{arm}"].to_numpy()
        candidate["bet_side"] = np.where(candidate.home_cover_probability.ge(0.5), "home", "away")
        if arm == "T1":
            candidate["home_side_shift"] = week.home_side_shift_T1.to_numpy()
        else:
            candidate["lattice_side_used"] = week.lattice_side_used_T2.to_numpy()
        common.table(candidate, out / "artifacts/margin_predictions" / arm / "predictions.parquet")
    diff = scratch[["home_team", "away_team", "spread_line"]].copy()
    diff["home_cover_probability"] = current.home_cover_probability
    for arm in ARMS:
        diff[f"p_{arm}"] = week[f"p_{arm}"]
    for arm in ARMS:
        diff[f"changed_{arm}"] = diff[f"p_{arm}"].ge(0.5) != diff.home_cover_probability.ge(0.5)
    common.table(diff.reset_index(), out / "week1_diff.parquet")
    write_stamped_artifact(
        {
            "scratch_cli_artifact": str(directories[-1]),
            "games": len(scratch),
            "active_forecast": active["weekly_forecast"]["artifact"],
            "scratch_cli_matches_active_forecast": bool(cli_difference <= 1e-12),
            "maximum_probability_difference": cli_difference,
            "stream_matches_active_forecast": True,
            "changes": {a: int(diff[f"changed_{a}"].sum()) for a in ARMS},
            "candidate_method": (
                "CLI incumbent followed by the explicit prior-only home-side mapper; "
                "not production integration"
            ),
        },
        out / "week1_cli_check.json",
    )


def replay(out: Path) -> None:
    commands = json.loads((out / "commands.json").read_text())["commands"]
    os.environ["NFL_ATS_REGISTRY_DIR"] = str(common.REPO / "registry")
    log = out / "live_replay"
    for command in commands:
        argv = command["argv"]
        if command["returncode"] or argv[0] != "nfl-ats":
            continue
        cli(log, *argv[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--centers", type=Path)
    parser.add_argument(
        "--stage",
        choices=(
            "setup",
            "build",
            "score",
            "losses",
            "composition",
            "homesplit",
            "seasons",
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
    if args.stage == "setup":
        setup(out, active)
    elif args.stage == "build":
        build(out, archive, args.centers, active)
    elif args.stage == "replay":
        replay(out)
    elif args.stage == "week1":
        week1(out, active)
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
                metadata["active_model_id"] = f"research_laneT_{arm}"
                metadata["active_model_config"]["model_id"] = f"research_laneT_{arm}"
                metadata["incumbent_model_id"] = active["model_id"]
                metadata["research_scope"] = (
                    "Only opener columns recomputed; close columns retain incumbent values "
                    "and are not candidate evidence."
                )
                write_stamped_artifact(metadata, path)
        elif args.stage == "homesplit":
            home_split(out, archive)
        else:
            record_seasons(out)


if __name__ == "__main__":
    main()
