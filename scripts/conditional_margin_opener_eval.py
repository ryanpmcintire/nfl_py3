"""Frozen lane K conditional-lattice experiment; isolated artifacts and registries."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import spread_regime_opener_eval as common
from threadpoolctl import threadpool_limits

from nfl_ats.conditional_margin import CONDITIONAL_MARGIN_METHODS, predict_conditional_margin
from nfl_ats.margin import fit_margin_model
from nfl_ats.modeling import regular_season_rows
from nfl_ats.provenance import sha256_file, stamp_sidecar, write_stamped_artifact

FAMILY = "mod18_conditional_margin_v1"
ARMS = dict(zip(("K1", "K2", "K3"), CONDITIONAL_MARGIN_METHODS, strict=True))
DISCOUNT = (
    "Mined archive; inherited active-model, lattice, spread diagnosis and overlay subset "
    "selection; three correlated arms; descriptive reuse, not independent confirmation."
)
common.FAMILY = FAMILY
common.DISCOUNT = DISCOUNT
common.ARMS = tuple(ARMS)


def build(out: Path, archive: pd.DataFrame) -> None:
    active = json.loads((common.REPO / "artifacts/active_ats_model.json").read_text())
    if active["model_id"] != "a4c757efd2525da6":
        raise ValueError("Active model changed")
    if sha256_file(common.FEATURES) != active["feature_table_sha256"]:
        raise ValueError("Feature digest changed")
    write_stamped_artifact(active, out / "artifacts/active_ats_model.json")
    for name in ("rotation_registry.json", "weak_signals.json"):
        (out / "registry").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(common.REPO / "registry" / name, out / "registry" / name)
        stamp_sidecar(out / "registry" / name)
    features = regular_season_rows(pd.read_parquet(common.FEATURES)).copy()
    features["gameday"] = pd.to_datetime(features.gameday)
    completed = features.loc[features.result.notna()]
    targets = features.loc[
        features.season.between(2011, 2025) & features.result.notna()
        | (features.season.eq(2026) & features.week.eq(1))
    ]
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
        rows["residual_base"] = pred.predicted_market_residual.to_numpy()
        rows["p_base"] = pred.home_cover_probability.to_numpy()
        batches.append(rows)
        if int(week) == 1:
            print(f"Built prior-only centers {season}", flush=True)
    stream = pd.concat(batches, ignore_index=True)
    common.table(stream, out / "centers.parquet")
    target = stream.loc[stream.season.ge(2020)].copy()
    for arm, method in ARMS.items():
        mapped = predict_conditional_margin(stream, target, method=method)
        target[f"p_{arm}"] = mapped.home_cover_probability
        for col in (
            "push_probability",
            "conditional_keyside_used",
            "conditional_effective_rows",
            "conditional_shift",
        ):
            target[f"{col}_{arm}"] = mapped[col]
    common.table(target, out / "stream.parquet")


def metric(frame: pd.DataFrame, differences: np.ndarray) -> dict:
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


def record(out: Path, name: str, m: dict, units: str, start: int, end: int, source: Path) -> None:
    common.cli(
        out,
        "weak-signals",
        "record",
        "--name",
        f"{FAMILY}_{name}_{start}_{end}",
        "--family",
        FAMILY,
        "--description",
        f"Conditional integer margins {name}, positive favors candidate",
        "--source",
        str(source),
        "--classification",
        "unresolved_below_power",
        "--classification-evidence",
        "No admissible mechanism or matched positive-control closure established.",
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
        str(m["delta"]),
        "--effect-units",
        units,
        "--interval-low",
        str(m["lower"]),
        "--interval-high",
        str(m["upper"]),
        "--probability-positive",
        str(m["probability_positive"]),
        "--notes",
        DISCOUNT,
    )


def losses(out: Path, archive: pd.DataFrame, stream: pd.DataFrame) -> None:
    frame = archive.merge(stream, on=["game_id", "season", "week"], validate="one_to_one")
    valid = frame.loc[frame.margin_vs_open.ne(0)].reset_index(drop=True)
    truth = valid.margin_vs_open.gt(0).to_numpy(dtype=float)
    base = valid.home_cover_probability_at_open.to_numpy()
    result = {}
    for arm in ARMS:
        p = np.clip(valid[f"p_{arm}"].to_numpy(), 1e-9, 1 - 1e-9)
        logbase = -(truth * np.log(base) + (1 - truth) * np.log1p(-base))
        logcandidate = -(truth * np.log(p) + (1 - truth) * np.log1p(-p))
        cells = {
            "brier": ((base - truth) ** 2, (p - truth) ** 2),
            "log_loss": (logbase, logcandidate),
        }
        result[arm] = {}
        for name, (bl, cl) in cells.items():
            m = metric(valid, bl - cl)
            m.update(candidate=float(cl.mean()), baseline=float(bl.mean()))
            result[arm][name] = m
    path = out / "losses.json"
    write_stamped_artifact(result, path)
    for arm in ARMS:
        for name, m in result[arm].items():
            record(out, f"{arm}_{name}", m, f"{name}_improvement", 2020, 2025, path)
    rows = []
    for arm, column in {
        "incumbent": "home_cover_probability_at_open",
        **{a: f"p_{a}" for a in ARMS},
    }.items():
        f = valid.copy()
        f["bucket"] = common.spread_bucket(f.tue_open_home_spread)
        pick = f[column].ge(0.5)
        f["correct"] = pick.eq(f.margin_vs_open.gt(0)).astype(float)
        f["confidence"] = f[column].where(pick, 1 - f[column])
        f["side"] = np.where(pick.eq(f.tue_open_home_spread.gt(0)), "favourite", "underdog")
        f.loc[f.tue_open_home_spread.eq(0), "side"] = "pickem"
        for allocation in ("7.5_to_upper", "7.5_to_lower"):
            if allocation == "7.5_to_lower":
                f.loc[f.tue_open_home_spread.abs().eq(7.5), "bucket"] = "7"
            for bucket, group in f.groupby("bucket", observed=True):
                for side in ("all", "favourite", "underdog", "pickem"):
                    g = group if side == "all" else group.loc[group.side.eq(side)]
                    if g.empty:
                        continue
                    m = metric(g.reset_index(drop=True), g.correct.to_numpy())
                    rows.append(
                        {
                            "arm": arm,
                            "allocation": allocation,
                            "bucket": str(bucket),
                            "side": side,
                            "n": len(g),
                            "accuracy": float(g.correct.mean()),
                            "lower": m["lower"],
                            "upper": m["upper"],
                            "confidence": float(g.confidence.mean()),
                        }
                    )
    common.table(pd.DataFrame(rows), out / "reliability.parquet")
    pushes = []
    for arm in ARMS:
        for key in (3, 7):
            g = frame.loc[frame.tue_open_home_spread.abs().eq(key)]
            pushes.append(
                {
                    "arm": arm,
                    "absolute_line": key,
                    "n": len(g),
                    "predicted": float(g[f"push_probability_{arm}"].mean()),
                    "realized": float(g.margin_vs_open.eq(0).mean()),
                }
            )
    write_stamped_artifact({"rows": pushes}, out / "push_calibration.json")
    week = stream.loc[stream.season.eq(2026)].copy()
    forecast = json.loads((common.REPO / "artifacts/active_ats_model.json").read_text())[
        "weekly_forecast"
    ]["artifact"]
    current = pd.read_csv(common.REPO / "artifacts" / forecast / "predictions.csv")
    if "method" in current:
        current = current.loc[current.method.eq("market_residual")]
    week = week.merge(
        current[["game_id", "home_cover_probability"]], on="game_id", validate="one_to_one"
    )
    for arm in ARMS:
        week[f"changed_{arm}"] = week[f"p_{arm}"].ge(0.5) != week.home_cover_probability.ge(0.5)
    common.table(week, out / "week1_diff.parquet")


def composition(out: Path, archive: pd.DataFrame, stream: pd.DataFrame) -> None:
    _, incumbent = common.composed_picks(archive, common.ARCHIVE / "per_game.parquet")
    results = {}
    picks = archive[["game_id", "season", "week"]].assign(incumbent=incumbent)
    for arm, method in ARMS.items():
        candidate = archive.copy()
        indexed = stream.set_index("game_id").reindex(archive.game_id)
        candidate["home_cover_probability_at_open"] = indexed[f"p_{arm}"].to_numpy()
        candidate["residual_at_open"] = indexed.residual_base.to_numpy()
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
            probability_method=method, active_model_id=f"research_laneK_{arm}", research_arm=arm
        )
        metadata["active_model_config"]["probability_method"] = method
        write_stamped_artifact(metadata, directory / "metadata.json")
        common.cli(
            out,
            "overlay-composition",
            "--per-game-artifact",
            str(directory / "per_game.parquet"),
            "--bootstrap-samples",
            "20000",
            "--bootstrap-seed",
            str(common.SEED),
        )
        _, cp = common.composed_picks(candidate, directory / "per_game.parquet")
        results[arm] = common.comparison(archive, cp, incumbent)
        picks[arm] = cp
    path = out / "composition.json"
    write_stamped_artifact(results, path)
    common.table(picks, out / "composed_picks.parquet")
    for arm, m in results.items():
        record(out, f"{arm}_three_member_card", m, "accuracy_points", 2020, 2025, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("build", "score", "losses", "composition"), required=True
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(common.REPO / "artifacts") or out.is_relative_to(common.REPO / "data"):
        raise ValueError("Use an isolated scratch root")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["NFL_ATS_ARTIFACTS_DIR"] = str(out / "artifacts")
    os.environ["NFL_ATS_REGISTRY_DIR"] = str(out / "registry")
    archive = pd.read_parquet(common.ARCHIVE / "per_game.parquet")
    with threadpool_limits(limits=1):
        if args.stage == "build":
            build(out, archive)
        else:
            stream = pd.read_parquet(out / "stream.parquet")
            if args.stage == "score":
                print(json.dumps(common.score(out, archive, stream), indent=2))
            elif args.stage == "losses":
                losses(out, archive, stream)
            else:
                composition(out, archive, stream)


if __name__ == "__main__":
    main()
