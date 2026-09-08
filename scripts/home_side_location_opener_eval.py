"""Frozen lane S home-side location experiment; see docs/home_side_location.md.

Everything is written under an isolated scratch root (artifacts and registry
copies); ``data/`` is read-only and the live registries are only touched by
the explicit ``replay`` stage, which re-issues the recorded CLI commands.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import spread_regime_opener_eval as common
from threadpoolctl import threadpool_limits

from nfl_ats.active_model import active_artifact_path
from nfl_ats.home_side_location import (
    HOME_SIDE_HINGE_COLUMNS,
    TRAILING_SEASONS,
    attach_home_side_location,
    fit_home_side_offsets,
    gaussian_median_cover_probability,
    walk_forward_home_offsets,
)
from nfl_ats.margin import MarginModel, fit_margin_model
from nfl_ats.modeling import regular_season_rows
from nfl_ats.provenance import sha256_file, stamp_sidecar, write_stamped_artifact
from nfl_ats.public_board import find_matching_opener_evaluation

FAMILY = "mod18_home_side_location_v1"
PROFILES = {"S1": "weak_stack_home_side_hinge_7"}
ARMS = ("S1", "S2")
DISCOUNT = (
    "Mined archive; inherited active-model, lane H/K/L/Q spread diagnosis (which motivated "
    "both arms; S2 also trains on this archive walk-forward) and overlay-subset selection; "
    "two correlated arms; descriptive reuse, not independent confirmation."
)
SUMMARIES = {
    "S1": (
        "Once the point spread is bigger than a touchdown, the model learns from past seasons "
        "how much extra to give the home team, whichever side of the spread it is on, so "
        "big-spread games lean the way the record says instead of selling the home side short."
    ),
    "S2": (
        "After the model makes its call, the number is nudged toward the home team by the "
        "amount home teams have beaten it by in earlier games with a similar-size spread, so "
        "big-spread picks stop selling the home side short."
    ),
}
BUCKET_SIDES = ("all", "home_favourite", "home_underdog")
original_cli = common.cli


def cli(out: Path, *args: str) -> str:
    """Run registry commands in-process (exact argv kept); others via the real CLI."""
    arguments = list(args)
    if arguments[:2] == ["weak-signals", "record"] and "--plain-summary" not in arguments:
        name = arguments[arguments.index("--name") + 1]
        arguments.extend(["--plain-summary", SUMMARIES["S2" if "_S2" in name else "S1"]])
    if "--notes" in arguments:
        i = arguments.index("--notes") + 1
        arguments[i] = arguments[i].replace("Five frozen arms", "Two frozen arms")
    if arguments[0] not in {"weak-signals", "rotation"}:
        return original_cli(out, *arguments)
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


def fixed(value: float) -> str:
    """Fixed-point text: argparse reads scientific notation like -1e-05 as a flag."""
    return f"{value:.12f}"


def record(out: Path, name: str, m: dict, units: str, start: int, end: int, source: Path) -> None:
    full_name = f"{FAMILY}_{name}_{start}_{end}"
    registry = Path(os.environ["NFL_ATS_REGISTRY_DIR"]) / "weak_signals.json"
    if registry.exists() and full_name in json.loads(registry.read_text())["signals"]:
        return  # Already recorded by an earlier partial run of this stage; keep it verbatim.
    cli(
        out,
        "weak-signals",
        "record",
        "--name",
        full_name,
        "--family",
        FAMILY,
        "--description",
        f"Home-side location arm {name}, positive favours the candidate",
        "--source",
        str(source),
        "--classification",
        "unresolved_below_power",
        "--classification-evidence",
        "No resolved wrong sign, reliability or positive-control closing ground established.",
        "--league",
        "nfl",
        "--season-start",
        str(start),
        "--season-end",
        str(end),
        "--sample-games",
        str(m["n"]),
        "--sample-blocks",
        str(m["weeks"]),
        "--category",
        "modeling",
        "--effect",
        fixed(m["delta"]),
        "--effect-units",
        units,
        "--interval-low",
        fixed(m["lower"]),
        "--interval-high",
        fixed(m["upper"]),
        "--probability-positive",
        fixed(m["probability_positive"]),
        "--notes",
        DISCOUNT,
    )


def record_arm(out: Path, name: str, metrics: dict, start: int, end: int, source: Path) -> None:
    record(out, name, metrics, "accuracy_points", start, end, source)


common.cli = cli
common.record_arm = record_arm
common.FAMILY = FAMILY
common.DISCOUNT = DISCOUNT
common.ARMS = ARMS


def metric(frame: pd.DataFrame, differences: np.ndarray) -> dict:
    """Week-blocked bootstrap of a per-game quantity; within-week correlation zero."""
    groups = list(frame.groupby(["season", "week"], sort=True).indices.values())
    sums = np.array([differences[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    draw = np.random.default_rng(common.SEED).integers(0, len(groups), (20000, len(groups)))
    values = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return {
        "delta": float(differences.mean()),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "probability_positive": float((values > 0).mean()),
        "standard_error": float(values.std(ddof=1)),
        "n": len(frame),
        "weeks": len(groups),
    }


def per_point_coefficients(model: MarginModel) -> dict[str, float]:
    """Fitted ridge weight per point of spread above seven, unscaled."""
    estimator = model.estimator
    if estimator is None:
        return {}
    scale = estimator.named_steps["scaler"].scale_
    coefficients = estimator.named_steps["regressor"].coef_
    out = {}
    for column in HOME_SIDE_HINGE_COLUMNS:
        if column in model.feature_columns:
            j = model.feature_columns.index(column)
            out[column] = float(coefficients[j] / scale[j]) if scale[j] > 0 else 0.0
    return out


def build(out: Path, archive: pd.DataFrame) -> None:
    active = json.loads((common.REPO / "artifacts/active_ats_model.json").read_text())
    if active["model_id"] != "a4c757efd2525da6":
        raise ValueError("Active model changed")
    if sha256_file(common.FEATURES) != active["feature_table_sha256"]:
        raise ValueError("Feature digest changed")
    write_stamped_artifact(active, out / "artifacts/active_ats_model.json")
    (out / "registry").mkdir(parents=True, exist_ok=True)
    for name in ("rotation_registry.json", "weak_signals.json"):
        shutil.copyfile(common.REPO / "registry" / name, out / "registry" / name)
        stamp_sidecar(out / "registry" / name)
    features = attach_home_side_location(regular_season_rows(pd.read_parquet(common.FEATURES)))
    features["gameday"] = pd.to_datetime(features.gameday)
    common.table(features, out / "features.parquet")
    completed = features.loc[features.result.notna()]
    archived_ids = set(archive.game_id)
    archive_seasons = set(archive.season.unique())
    targets = features.loc[
        features.season.between(2011, 2025) & features.result.notna()
        | (features.season.eq(2026) & features.week.eq(1))
    ]
    batches, weights = [], []
    for (season, week), group in targets.groupby(["season", "week"], sort=True):
        if int(str(season)) in archive_seasons:
            scoring = group.loc[group.game_id.isin(archived_ids)].merge(
                archive[["game_id", "tue_open_home_spread"]], on="game_id", validate="one_to_one"
            )
            if scoring.empty:
                continue
            scoring["spread_line"] = scoring.tue_open_home_spread
            grade = "opener"
        else:
            scoring, grade = group.copy(), "nflverse_spread"
        cutoff = scoring.gameday.min()
        training = completed.loc[completed.gameday.lt(cutoff)]
        if len(training) < 500:
            continue
        scoring = attach_home_side_location(scoring)
        rows = scoring[["game_id", "season", "week", "gameday", "spread_line", "result"]].copy()
        rows["grade"] = grade
        margin = rows.result - rows.spread_line
        rows["home_cover"] = margin.gt(0).astype(float).where(margin.ne(0) & margin.notna())
        rows["training_rows"] = len(training)
        rows["training_max_gameday"] = training.gameday.max()
        for name, profile in (("base", "weak_stack"), *PROFILES.items()):
            model = fit_margin_model(
                training,
                target="market_residual",
                model_name="ridge",
                feature_profile=profile,
                ridge_alpha=10.0,
            )
            predicted = model.predict(scoring, probability_method="gaussian_median")
            rows[f"p_{name}"] = predicted.home_cover_probability.to_numpy()
            rows[f"residual_{name}"] = predicted.predicted_market_residual.to_numpy()
            rows[f"point_{name}"] = rows.spread_line + rows[f"residual_{name}"]
            if name == "base":
                # The gaussian_median read is a Normal with the fitted week's
                # residual median and sample standard deviation; S2 re-evaluates
                # exactly this read at the corrected point.
                rows["gm_median_base"] = float(np.median(model.residuals))
                rows["gm_std_base"] = float(np.std(model.residuals, ddof=1))
            else:
                weights.append(
                    {"season": season, "week": week, "arm": name, **per_point_coefficients(model)}
                )
        batches.append(rows)
        if int(week) == 1:
            print(f"Built {season}; prior training rows {len(training)}", flush=True)
    stream = pd.concat(batches, ignore_index=True)
    reproduced = gaussian_median_cover_probability(
        stream.spread_line, stream.point_base, stream.gm_median_base, stream.gm_std_base
    )
    mapping_gap = float(np.abs(reproduced - stream.p_base).max())
    if mapping_gap > 1e-9:
        raise ValueError(
            f"Stored gaussian_median parameters do not reproduce p_base: {mapping_gap}"
        )
    stream = attach_s2(stream, archive)
    common.table(stream, out / "stream.parquet")
    common.table(pd.DataFrame(weights), out / "coefficients.parquet")
    write_stamped_artifact(
        {
            "active": active,
            "archive": str(common.ARCHIVE),
            "predeclaration_sha256": sha256_file(common.REPO / "docs/home_side_location.md"),
            "profiles": PROFILES,
            "gaussian_median_reproduction_max_gap": mapping_gap,
        },
        out / "inputs.json",
    )


def attach_s2(stream: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    """S2 on the archived out-of-time points (2020-2025) and on Week 1 2026.

    The training stream is the archive itself: ``tue_open_home_spread +
    residual_at_open`` per game, actual home margin ``result``. The week's
    offsets are fitted on prior completed games only (walk-forward).
    """

    stream = stream.copy()
    points = archive[["game_id", "tue_open_home_spread", "residual_at_open"]].rename(
        columns={"residual_at_open": "residual_archive"}
    )
    frame = stream.loc[stream.grade.eq("opener") | stream.season.eq(2026)].merge(
        points, on="game_id", how="left", validate="one_to_one"
    )
    opener = frame.grade.eq("opener")
    if int(opener.sum()) != len(archive):
        raise ValueError("Opener-grade stream rows do not cover the archive")
    replay_gap = float(
        (frame.loc[opener, "residual_archive"] - frame.loc[opener, "residual_base"]).abs().max()
    )
    if replay_gap > 1e-6:
        raise ValueError(f"Base stream does not replay the archived points: {replay_gap}")
    frame["point_incumbent"] = np.where(
        opener, frame.spread_line + frame.residual_archive, frame.point_base
    )
    offsets = walk_forward_home_offsets(frame.set_index("game_id")).reset_index()
    frame = frame.merge(offsets, on="game_id", validate="one_to_one")
    frame["p_S2"] = gaussian_median_cover_probability(
        frame.spread_line, frame.point_corrected, frame.gm_median_base, frame.gm_std_base
    )
    frame["residual_S2"] = frame.point_corrected - frame.spread_line
    frame["point_S2"] = frame.point_corrected
    frame["s2_replay_gap"] = replay_gap
    columns = [
        "game_id",
        "bucket",
        "home_side_offset",
        "prior_games_in_bucket",
        "p_S2",
        "residual_S2",
        "point_S2",
        "s2_replay_gap",
    ]
    return stream.merge(frame[columns], on="game_id", how="left", validate="one_to_one")


def sides(frame: pd.DataFrame, line: pd.Series) -> pd.DataFrame:
    frame = frame.copy()
    frame["bucket"] = common.spread_bucket(line)
    frame["home_side"] = np.where(line.gt(0), "home_favourite", "home_underdog")
    frame.loc[line.eq(0), "home_side"] = "pickem"
    return frame


def cell_name(arm: str, kind: str, bucket: str, side: str) -> str:
    return f"{arm}_{kind}_{bucket}_{side}".replace(".", "p").replace("+", "plus")


def losses(out: Path, archive: pd.DataFrame, stream: pd.DataFrame) -> None:
    frame = archive.merge(
        stream, on=["game_id", "season", "week"], validate="one_to_one", suffixes=("", "_stream")
    )
    valid = frame.loc[frame.margin_vs_open.ne(0)].reset_index(drop=True)
    valid = sides(valid, valid.tue_open_home_spread)
    truth = valid.margin_vs_open.gt(0).to_numpy(dtype=float)
    base = np.clip(valid.home_cover_probability_at_open.to_numpy(), 1e-9, 1 - 1e-9)
    result: dict = {}
    for arm in ARMS:
        p = np.clip(valid[f"p_{arm}"].to_numpy(), 1e-9, 1 - 1e-9)
        logbase = -(truth * np.log(base) + (1 - truth) * np.log1p(-base))
        logcandidate = -(truth * np.log(p) + (1 - truth) * np.log1p(-p))
        result[arm] = {}
        for name, (bl, cl) in {
            "brier": ((base - truth) ** 2, (p - truth) ** 2),
            "log_loss": (logbase, logcandidate),
        }.items():
            m = metric(valid, bl - cl)
            m.update(candidate=float(cl.mean()), baseline=float(bl.mean()))
            result[arm][name] = m
    path = out / "losses.json"
    write_stamped_artifact(result, path)
    for arm in ARMS:
        for name, m in result[arm].items():
            record(out, f"{arm}_{name}", m, f"{name}_improvement", 2020, 2025, path)
    rows, cells = [], {}
    base_pick = valid.home_cover_probability_at_open.ge(0.5)
    base_correct = base_pick.eq(valid.margin_vs_open.gt(0)).astype(float)
    for arm in ("incumbent", *ARMS):
        column = "home_cover_probability_at_open" if arm == "incumbent" else f"p_{arm}"
        f = valid.copy()
        pick = f[column].ge(0.5)
        f["correct"] = pick.eq(f.margin_vs_open.gt(0)).astype(float)
        f["confidence"] = f[column].where(pick, 1 - f[column])
        f["pick_side"] = np.where(pick.eq(f.tue_open_home_spread.gt(0)), "favourite", "underdog")
        f.loc[f.tue_open_home_spread.eq(0), "pick_side"] = "pickem"
        f["base_correct"] = base_correct
        for bucket, group in f.groupby("bucket", observed=True):
            selections = {"all": group}
            selections.update({str(s): g for s, g in group.groupby("home_side")})
            selections.update({f"pick_{s}": g for s, g in group.groupby("pick_side")})
            for side, g in selections.items():
                g = g.reset_index(drop=True)
                m = metric(g, g.correct.to_numpy())
                rows.append(
                    {
                        "arm": arm,
                        "bucket": str(bucket),
                        "side": side,
                        "n": len(g),
                        "accuracy": float(g.correct.mean()),
                        "lower": m["lower"],
                        "upper": m["upper"],
                        "confidence": float(g.confidence.mean()),
                        "calibration_gap": float(g.correct.mean() - g.confidence.mean()),
                        "picks_home": float(g[column].ge(0.5).mean()),
                    }
                )
                if arm in ARMS and side in BUCKET_SIDES:
                    delta = metric(g, 100 * (g.correct - g.base_correct).to_numpy())
                    cells[cell_name(arm, "bucket", str(bucket), side)] = delta
    common.table(pd.DataFrame(rows), out / "reliability.parquet")
    path = out / "bucket_cells.json"
    write_stamped_artifact(cells, path)
    for name, m in cells.items():
        record(out, name, m, "accuracy_points", 2020, 2025, path)
    seasons = {}
    for season, group in frame.loc[frame.margin_vs_open.ne(0)].groupby("season"):
        for arm in ARMS:
            seasons[f"{arm}_{season}"] = common.comparison(
                group.reset_index(drop=True),
                group[f"p_{arm}"].ge(0.5).to_numpy(),
                group.pick_home_at_open_probability_rule.to_numpy(dtype=bool),
            )
    path = out / "seasons.json"
    write_stamped_artifact(seasons, path)
    for name, m in seasons.items():
        arm, season = name.split("_")
        record(out, f"{arm}_season", m, "accuracy_points", int(season), int(season), path)
    # S1 only: S2 has no pre-2020 training stream (docs/home_side_location.md).
    early = stream.loc[stream.grade.eq("nflverse_spread") & stream.season.le(2019)].copy()
    early = early.loc[early.home_cover.notna()].reset_index(drop=True)
    replication = {}
    for label, block in (
        ("2011-2017", early.loc[early.season.le(2017)]),
        ("2018-2019", early.loc[early.season.ge(2018)]),
    ):
        block = block.reset_index(drop=True)
        truth_early = block.home_cover.eq(1).to_numpy()
        base_early = block.p_base.ge(0.5).to_numpy()
        diff = (block.p_S1.ge(0.5).to_numpy() == truth_early).astype(float) - (
            base_early == truth_early
        ).astype(float)
        replication[f"S1_{label}"] = metric(block, 100 * diff)
    write_stamped_artifact(replication, out / "nflverse_replication.json")


def diagnosis(out: Path, archive: pd.DataFrame, stream: pd.DataFrame) -> None:
    """Diagnosis D (home point error) on the incumbent AND each arm's point forecast."""
    frame = archive.merge(
        stream, on=["game_id", "season", "week"], validate="one_to_one", suffixes=("", "_stream")
    )
    frame = sides(frame, frame.tue_open_home_spread)
    frame["point_incumbent"] = frame.tue_open_home_spread + frame.residual_at_open
    pick = frame.home_cover_probability_at_open.ge(0.5)
    frame["pick_side"] = np.where(
        pick.eq(frame.tue_open_home_spread.gt(0)), "favourite", "underdog"
    )
    frame.loc[frame.tue_open_home_spread.eq(0), "pick_side"] = "pickem"
    rows, cells = [], {}
    for bucket, group in frame.groupby("bucket", observed=True):
        selections = {"all": group}
        selections.update({str(s): g for s, g in group.groupby("home_side")})
        selections.update({f"pick_{s}": g for s, g in group.groupby("pick_side")})
        for side, g in selections.items():
            g = g.reset_index(drop=True)
            incumbent_error = (g.result - g.point_incumbent).to_numpy()
            for arm in ("incumbent", *ARMS):
                error = (g.result - g[f"point_{arm}"]).to_numpy()
                m = metric(g, error)
                rows.append(
                    {
                        "bucket": str(bucket),
                        "side": side,
                        "arm": arm,
                        "metric": "home_point_error",
                        "mean_absolute_error": float(np.abs(error).mean()),
                        **m,
                    }
                )
                if arm in ARMS:
                    improvement = metric(g, np.abs(incumbent_error) - np.abs(error))
                    rows.append(
                        {
                            "bucket": str(bucket),
                            "side": side,
                            "arm": arm,
                            "metric": "mae_improvement",
                            "mean_absolute_error": float(np.abs(error).mean()),
                            **improvement,
                        }
                    )
                    if side in BUCKET_SIDES:
                        cells[cell_name(arm, "mae", str(bucket), side)] = improvement
    overall = frame.reset_index(drop=True)
    for arm in ARMS:
        improvement = metric(
            overall,
            np.abs(overall.result - overall.point_incumbent).to_numpy()
            - np.abs(overall.result - overall[f"point_{arm}"]).to_numpy(),
        )
        cells[f"{arm}_mae_all"] = improvement
    common.table(pd.DataFrame(rows), out / "diagnosis.parquet")
    offsets = (
        frame.groupby(["season", "bucket"], observed=True)
        .agg(
            offset_mean=("home_side_offset", "mean"),
            offset_min=("home_side_offset", "min"),
            offset_max=("home_side_offset", "max"),
            prior_games_mean=("prior_games_in_bucket", "mean"),
        )
        .reset_index()
    )
    common.table(offsets, out / "offsets_by_season.parquet")
    path = out / "mae_cells.json"
    write_stamped_artifact(cells, path)
    for name, m in cells.items():
        record(out, name, m, "mae_improvement", 2020, 2025, path)


def composition(out: Path, archive: pd.DataFrame, stream: pd.DataFrame) -> None:
    _, incumbent = common.composed_picks(archive, common.ARCHIVE / "per_game.parquet")
    results = {}
    picks = archive[["game_id", "season", "week"]].assign(incumbent=incumbent)
    indexed = stream.set_index("game_id").reindex(archive.game_id)
    for arm in ARMS:
        profile = PROFILES.get(arm, "weak_stack + walk-forward home-side offset by spread size")
        candidate = archive.copy()
        candidate["home_cover_probability_at_open"] = indexed[f"p_{arm}"].to_numpy()
        candidate["residual_at_open"] = indexed[f"residual_{arm}"].to_numpy()
        candidate["pick_home_at_open"] = candidate.residual_at_open.gt(0)
        candidate["correct_at_open"] = (
            candidate.pick_home_at_open.eq(candidate.margin_vs_open.gt(0))
            .astype(float)
            .where(candidate.margin_vs_open.ne(0))
        )
        candidate["pick_home_at_open_probability_rule"] = (
            candidate.home_cover_probability_at_open.ge(0.5)
        )
        candidate["correct_at_open_probability_rule"] = (
            candidate.pick_home_at_open_probability_rule.eq(candidate.margin_vs_open.gt(0))
            .astype(float)
            .where(candidate.margin_vs_open.ne(0))
        )
        directory = out / "artifacts/opener_evaluation" / arm
        common.table(candidate, directory / "per_game.parquet")
        metadata = json.loads((common.ARCHIVE / "metadata.json").read_text())
        metadata.update(
            research_arm=arm,
            incumbent_model_id=metadata.get("active_model_id"),
            active_model_id=f"research_laneS_{arm}",
            feature_profile=profile,
            research_scope=(
                "Only opener columns recomputed; close columns retain incumbent values "
                "and are not candidate evidence."
            ),
        )
        metadata["active_model_config"] = {
            **metadata["active_model_config"],
            "feature_profile": profile,
            "model_id": f"research_laneS_{arm}",
        }
        write_stamped_artifact(metadata, directory / "metadata.json")
        cli(
            out,
            "overlay-composition",
            "--per-game-artifact",
            str(directory / "per_game.parquet"),
            "--bootstrap-samples",
            "20000",
            "--bootstrap-seed",
            str(common.SEED),
        )
        _, composed = common.composed_picks(candidate, directory / "per_game.parquet")
        results[arm] = common.comparison(archive, composed, incumbent)
        picks[arm] = composed
    path = out / "composition.json"
    write_stamped_artifact(results, path)
    common.table(picks, out / "composed_picks.parquet")
    for arm, m in results.items():
        record(out, f"{arm}_three_member_card", m, "accuracy_points", 2020, 2025, path)


def week1(out: Path, archive: pd.DataFrame, stream: pd.DataFrame) -> None:
    active = json.loads((common.REPO / "artifacts/active_ats_model.json").read_text())
    forecast = active_artifact_path(common.REPO / "artifacts", active, "weekly_forecast")
    if forecast is None:
        raise ValueError("Active manifest names no weekly forecast")
    current = pd.read_csv(forecast / "predictions.csv")
    if "method" in current:
        current = current.loc[current.method.eq("market_residual")]
    week = stream.loc[stream.season.eq(2026) & stream.week.eq(1)].copy()
    week = week.merge(
        current[
            [
                "game_id",
                "home_team",
                "away_team",
                "spread_line",
                "home_cover_probability",
                "predicted_market_residual",
            ]
        ],
        on="game_id",
        validate="one_to_one",
        suffixes=("", "_forecast"),
    )
    week["forecast_line_matches"] = week.spread_line.eq(week.spread_line_forecast)
    # S2 from the live forecast's own point, with offsets fitted on the
    # archived games of the five trailing seasons (all completed before Week 1
    # 2026); must equal the stream's walk-forward value.
    trailing = archive.loc[archive.season.ge(2026 - TRAILING_SEASONS)]
    fitted = fit_home_side_offsets(
        trailing.assign(
            spread_line=trailing.tue_open_home_spread,
            point_incumbent=trailing.tue_open_home_spread + trailing.residual_at_open,
        )
    )
    forecast_point = week.spread_line_forecast + week.predicted_market_residual
    week["p_S2_from_forecast"] = gaussian_median_cover_probability(
        week.spread_line_forecast,
        forecast_point + fitted.offset_for(week.spread_line_forecast).to_numpy(),
        week.gm_median_base,
        week.gm_std_base,
    )
    for arm in ARMS:
        week[f"changed_{arm}"] = week[f"p_{arm}"].ge(0.5) != week.home_cover_probability.ge(0.5)
        week[f"side_{arm}"] = np.where(week[f"p_{arm}"].ge(0.5), week.home_team, week.away_team)
    week["side_forecast"] = np.where(
        week.home_cover_probability.ge(0.5), week.home_team, week.away_team
    )
    common.table(week, out / "week1_diff.parquet")
    cli_gap = {}
    for arm, profile in PROFILES.items():
        original_cli(
            out,
            "margin-predict",
            "--season",
            "2026",
            "--week",
            "1",
            "--features",
            str(out / "features.parquet"),
            "--feature-profile",
            profile,
            "--ridge-alpha",
            "10",
            "--probability-method",
            "gaussian_median",
            "--no-line-sweep",
        )
        directories = sorted((out / "artifacts/margin_predictions").glob("2026-week-01-*"))
        scratch = pd.read_csv(directories[-1] / "predictions.csv")
        scratch = scratch.loc[scratch.method.eq("market_residual")].set_index("game_id")
        harness = week.set_index("game_id").reindex(scratch.index)
        cli_gap[arm] = float((scratch.home_cover_probability - harness[f"p_{arm}"]).abs().max())
    write_stamped_artifact(
        {
            "forecast": str(forecast),
            "games": len(week),
            "baseline_max_probability_difference": float(
                (week.p_base - week.home_cover_probability).abs().max()
            ),
            "forecast_lines_match": bool(week.forecast_line_matches.all()),
            "cli_max_probability_difference": cli_gap,
            "s2_forecast_vs_stream_max_probability_difference": float(
                (week.p_S2_from_forecast - week.p_S2).abs().max()
            ),
            "s2_offsets_2026": fitted.offsets,
            "s2_prior_games_2026": fitted.prior_games,
            "changes": {
                arm: week.loc[
                    week[f"changed_{arm}"],
                    [
                        "game_id",
                        "spread_line",
                        "side_forecast",
                        f"side_{arm}",
                        "p_base",
                        f"p_{arm}",
                    ],
                ].to_dict(orient="records")
                for arm in ARMS
            },
        },
        out / "week1.json",
    )


def replay(out: Path) -> None:
    commands = json.loads((out / "commands.json").read_text())["commands"]
    os.environ["NFL_ATS_REGISTRY_DIR"] = str(common.REPO / "registry")
    log = out / "live_replay"
    for command in commands:
        argv = command["argv"]
        if command["returncode"] or argv[1] not in {"weak-signals", "rotation"}:
            continue
        cli(log, *argv[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("build", "score", "losses", "diagnosis", "composition", "week1", "replay"),
        required=True,
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(common.REPO):
        raise ValueError("Use an isolated scratch root outside the repository")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["NFL_ATS_ARTIFACTS_DIR"] = str(out / "artifacts")
    os.environ["NFL_ATS_REGISTRY_DIR"] = str(out / "registry")
    active = json.loads((common.REPO / "artifacts/active_ats_model.json").read_text())
    match = find_matching_opener_evaluation(common.REPO / "artifacts", active)
    if match is None:
        raise ValueError("No opener evaluation matches the active model")
    common.ARCHIVE = match[1]
    archive = pd.read_parquet(common.ARCHIVE / "per_game.parquet")
    with threadpool_limits(limits=1):
        if args.stage == "build":
            build(out, archive)
            return
        stream = pd.read_parquet(out / "stream.parquet")
        if args.stage == "score":
            print(json.dumps(common.score(out, archive, stream), indent=2))
        elif args.stage == "losses":
            losses(out, archive, stream)
        elif args.stage == "diagnosis":
            diagnosis(out, archive, stream)
        elif args.stage == "composition":
            composition(out, archive, stream)
        elif args.stage == "week1":
            week1(out, archive, stream)
        else:
            replay(out)


if __name__ == "__main__":
    main()
