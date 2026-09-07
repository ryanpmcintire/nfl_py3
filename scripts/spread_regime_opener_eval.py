"""Execute the frozen MOD-18 program in isolated roots; see its predeclaration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nfl_ats.key_numbers import implied_key_number_mass
from nfl_ats.margin import fit_margin_model
from nfl_ats.modeling import regular_season_rows
from nfl_ats.overlay_composition import (
    DEFAULT_FEATURES,
    DEFAULT_INCIDENTS,
    build_predictions_frame,
    load_inputs,
    reconstruct_arrest_flip_set,
    run_overlays,
)
from nfl_ats.provenance import sha256_file, stamp_sidecar, write_stamped_artifact
from nfl_ats.spread_regime import attach_spread_regime, calibrate_spread_stream, spread_bucket

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "artifacts/opener_evaluation/20260907T152026Z"
FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"
FAMILY = "mod18_spread_regime_v1"
ARMS = ("C1", "C2", "C3", "C1_C2", "C1_C3")
SEED = 20260817
DISCOUNT = (
    "Mined archive; inherited active-model, spread-bound and 127-subset union selection; "
    "five correlated arms; descriptive reuse, not independent confirmation."
)


def table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    stamp_sidecar(path)


def cli(out: Path, *args: str) -> str:
    command = [str(REPO / ".tools/uv.exe"), "run", "--no-sync", "nfl-ats", *args]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    path = out / "commands.json"
    prior = json.loads(path.read_text()).get("commands", []) if path.exists() else []
    prior.append(
        {
            "argv": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )
    write_stamped_artifact({"commands": prior}, path)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def build_stream(features: pd.DataFrame, archive: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Reuse pre2018 screen's strictly-prior weekly ridge recipe, two profiles."""
    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame.gameday)
    completed = frame.loc[frame.result.notna()].sort_values(["gameday", "game_id"])
    targets = frame.loc[
        frame.season.between(2011, 2025) & frame.result.notna()
        | (frame.season.eq(2026) & frame.week.eq(1))
    ]
    batches = []
    for (season, week), group in targets.groupby(["season", "week"], sort=True):
        cutoff = group.gameday.min()
        training = completed.loc[completed.gameday.lt(cutoff)]
        if len(training) < 500:
            continue
        scoring = group.merge(
            archive[["game_id", "tue_open_home_spread"]],
            on="game_id",
            how="left",
            validate="one_to_one",
        )
        scoring["spread_line"] = scoring.tue_open_home_spread.fillna(scoring.spread_line)
        scoring = attach_spread_regime(scoring)
        result = scoring[["game_id", "season", "week", "gameday", "spread_line", "result"]].copy()
        margin = result.result - result.spread_line
        result["home_cover"] = margin.gt(0).astype(float).where(margin.ne(0) & margin.notna())
        result["training_max_gameday"] = training.gameday.max()
        for profile, prefix in (("weak_stack", "base"), ("weak_stack_spread_regime", "C1")):
            model = fit_margin_model(
                training,
                target="market_residual",
                model_name="ridge",
                feature_profile=profile,
                ridge_alpha=10.0,
            )
            for method, name in (
                ("gaussian_median", prefix),
                ("discrete_residual", "C2" if prefix == "base" else "C1_C2"),
            ):
                predicted = model.predict(scoring, probability_method=method)
                result[f"p_{name}"] = predicted.home_cover_probability.to_numpy()
                result[f"residual_{name}"] = predicted.predicted_market_residual.to_numpy()
            if prefix == "base":
                center = model.predict(scoring).predicted_margin.to_numpy()
                mass = implied_key_number_mass(center[:, None] + model.residuals, (3, 7, 10, 14))
                for column in mass:
                    result[column] = mass[column].to_numpy()
        batches.append(result)
        if int(week) == 1:
            print(f"Predicted {season}; prior training rows {len(training)}", flush=True)
    stream = pd.concat(batches, ignore_index=True)
    for source, name in (("base", "C3"), ("C1", "C1_C3")):
        raw = stream.assign(home_cover_probability=stream[f"p_{source}"])
        calibrated = calibrate_spread_stream(raw)
        stream[f"p_{name}"] = calibrated.home_cover_probability
        stream[f"calibration_rows_{name}"] = calibrated.calibration_rows
    table(stream, out / "stream.parquet")
    return stream


def comparison(
    frame: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray, samples: int = 20_000
) -> dict:
    valid = frame.margin_vs_open.ne(0) & frame.margin_vs_open.notna()
    scored = frame.loc[valid].reset_index(drop=True)
    truth = scored.margin_vs_open.gt(0).to_numpy()
    cp, bp = np.asarray(candidate)[valid], np.asarray(baseline)[valid]
    diff = (cp == truth).astype(float) - (bp == truth).astype(float)
    groups = list(scored.groupby(["season", "week"], sort=True).indices.values())
    sums = np.array([diff[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(SEED)
    selected = rng.integers(0, len(groups), (samples, len(groups)))
    draws = 100 * sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return {
        "delta": float(100 * diff.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "probability_positive": float((draws > 0).mean()),
        "standard_error": float(draws.std(ddof=1)),
        "n": len(scored),
        "weeks": len(groups),
        "candidate_accuracy": float((cp == truth).mean()),
        "baseline_accuracy": float((bp == truth).mean()),
        "flips": int((cp != bp).sum()),
    }


def record_arm(out: Path, name: str, metrics: dict, start: int, end: int, source: Path) -> None:
    fields = [
        "--effect",
        str(metrics["delta"]),
        "--effect-units",
        "accuracy_points",
        "--interval-low",
        str(metrics["lower"]),
        "--interval-high",
        str(metrics["upper"]),
        "--probability-positive",
        str(metrics["probability_positive"]),
        "--sample-blocks",
        str(metrics["weeks"]),
        "--notes",
        DISCOUNT,
    ]
    cli(
        out,
        "weak-signals",
        "record",
        "--name",
        f"{FAMILY}_{name}_{start}_{end}",
        "--family",
        FAMILY,
        "--description",
        f"Frozen spread-regime {name} paired opener accuracy",
        "--source",
        str(source),
        "--classification",
        "unresolved_below_power",
        "--classification-evidence",
        "No reliability or positive-control closing ground established; retained unresolved.",
        "--league",
        "nfl",
        "--season-start",
        str(start),
        "--season-end",
        str(end),
        "--sample-games",
        str(metrics["n"]),
        "--category",
        "modeling",
        *fields,
    )


def score(out: Path, archive: pd.DataFrame, stream: pd.DataFrame) -> dict:
    paired = archive.merge(
        stream, on=["game_id", "season", "week"], validate="one_to_one", suffixes=("", "_stream")
    )
    if len(paired) != len(archive):
        raise ValueError("Candidate archive coverage changed")
    baseline = paired.pick_home_at_open_probability_rule.to_numpy(dtype=bool)
    results = {}
    # The first block was assigned before this script was run. Subsequent
    # frozen blocks are assigned only after recording their predecessor.
    for start, end in ((2020, 2021), (2022, 2023), (2024, 2025), (2020, 2025)):
        block = paired.loc[paired.season.between(start, end)]
        base = block.pick_home_at_open_probability_rule.to_numpy(dtype=bool)
        metrics = {
            arm: comparison(block, block[f"p_{arm}"].ge(0.5).to_numpy(), base) for arm in ARMS
        }
        path = out / f"window_{start}_{end}.json"
        write_stamped_artifact({"arms": metrics, "discount": DISCOUNT}, path)
        for arm in ARMS:
            record_arm(out, arm, metrics[arm], start, end, path)
        if end - start == 1:
            selected = max(ARMS, key=lambda name: metrics[name]["delta"])
            m = metrics[selected]
            cli(
                out,
                "rotation",
                "record",
                "--name",
                FAMILY,
                "--artifact",
                str(path),
                "--verdict",
                "unresolved",
                "--effect",
                str(m["delta"]),
                "--effect-units",
                "accuracy_points",
                "--interval-low",
                str(m["lower"]),
                "--interval-high",
                str(m["upper"]),
                "--probability-positive",
                str(m["probability_positive"]),
                "--sample-blocks",
                str(m["weeks"]),
                "--notes",
                f"Five frozen arms; headline selected {selected}. {DISCOUNT}",
            )
            if end < 2025:
                cli(out, "rotation", "assign", "--name", FAMILY)
        else:
            results = metrics
    valid = paired.loc[paired.margin_vs_open.ne(0)].reset_index(drop=True)
    picks = np.column_stack([valid[f"p_{arm}"].ge(0.5) for arm in ARMS])
    base = valid.pick_home_at_open_probability_rule.to_numpy(dtype=bool)
    truth = valid.margin_vs_open.gt(0).to_numpy()
    groups = list(valid.groupby(["season", "week"], sort=True).indices.values())
    rng = np.random.default_rng(SEED)
    null = np.empty((2000, len(ARMS)))
    for draw in range(len(null)):
        shuffled = truth.copy()
        for group in groups:
            shuffled[group] = rng.permutation(truth[group])
        null[draw] = 100 * (
            (picks == shuffled[:, None]).astype(float) - (base == shuffled)[:, None]
        ).mean(axis=0)
    winner = max(ARMS, key=lambda name: results[name]["delta"])
    for i, arm in enumerate(ARMS):
        results[arm]["null_interval"] = np.quantile(null[:, i], [0.025, 0.975]).tolist()
        results[arm]["null_observed_percentile"] = float(
            (null[:, i] <= results[arm]["delta"]).mean()
        )
        results[arm]["season_deltas"] = {
            str(year): comparison(
                g,
                g[f"p_{arm}"].ge(0.5).to_numpy(),
                g.pick_home_at_open_probability_rule.to_numpy(dtype=bool),
            )["delta"]
            for year, g in paired.groupby("season")
        }
    payload = {
        "arms": results,
        "winner": winner,
        "archive_rows": len(paired),
        "seed": SEED,
        "samples": 20000,
        "within_week_correlation": 0,
        "null_samples": 2000,
        "discount": DISCOUNT,
        "family_max_null_interval": np.quantile(null.max(axis=1), [0.025, 0.975]).tolist(),
        "strict_week_baseline_pick_disagreements": int((paired.p_base.ge(0.5) != baseline).sum()),
        "strict_week_baseline_max_probability_difference": float(
            (paired.p_base - paired.home_cover_probability_at_open).abs().max()
        ),
    }
    table(paired, out / "paired.parquet")
    write_stamped_artifact(payload, out / "results.json")
    return payload


def diagnosis(out: Path, archive: pd.DataFrame, stream: pd.DataFrame) -> None:
    early = stream.loc[stream.season.le(2019)].copy()
    current = archive.merge(stream[["game_id", "gameday"]], on="game_id")
    current = current.assign(
        spread_line=current.tue_open_home_spread,
        p_base=current.home_cover_probability_at_open,
        result=current.margin_vs_open + current.tue_open_home_spread,
    )
    rows = []
    for label, grade, frame in (
        ("2011-2017", "nflverse spread", early.loc[early.season.le(2017)]),
        ("2018-2019", "nflverse spread", early.loc[early.season.ge(2018)]),
        ("2020-2025", "opener", current),
    ):
        f = frame.copy()
        f["margin_vs_open"] = f.result - f.spread_line
        f = f.loc[f.margin_vs_open.ne(0)].copy()
        f["bucket"] = spread_bucket(f.spread_line)
        pick = f.p_base.ge(0.5)
        f["correct"] = pick.eq(f.margin_vs_open.gt(0)).astype(float)
        f["confidence"] = f.p_base.where(pick, 1 - f.p_base)
        f["favourite_pick"] = pick.eq(f.spread_line.gt(0)).where(f.spread_line.ne(0))
        f["favourite_cover"] = (
            f.margin_vs_open.gt(0).eq(f.spread_line.gt(0)).where(f.spread_line.ne(0))
        )
        f["side"] = np.where(f.favourite_pick, "favourite", "underdog")
        f.loc[f.spread_line.eq(0), "side"] = "pickem"
        for bucket, group in f.groupby("bucket", sort=False):
            for side in ("all", "favourite", "underdog", "pickem"):
                g = group if side == "all" else group.loc[group.side.eq(side)]
                if g.empty:
                    continue
                block = g.groupby(["season", "week"]).correct.agg(["sum", "count"])
                draw = np.random.default_rng(SEED).integers(0, len(block), (20000, len(block)))
                rates = block["sum"].to_numpy()[draw].sum(axis=1) / block["count"].to_numpy()[
                    draw
                ].sum(axis=1)
                rows.append(
                    {
                        "era": label,
                        "grade": grade,
                        "bucket": bucket,
                        "side": side,
                        "n": len(g),
                        "accuracy": float(g.correct.mean()),
                        "lower": float(np.quantile(rates, 0.025)),
                        "upper": float(np.quantile(rates, 0.975)),
                        "stated_probability": float(g.confidence.mean()),
                        "calibration_gap": float(g.correct.mean() - g.confidence.mean()),
                        "probability_positive": float((rates > g.confidence.mean()).mean()),
                        "favourite_picks": float(g.favourite_pick.mean()),
                        "favourite_covers": float(g.favourite_cover.mean()),
                    }
                )
    table(pd.DataFrame(rows), out / "diagnosis.parquet")


def composed_picks(per_game: pd.DataFrame, path: Path) -> tuple[np.ndarray, np.ndarray]:
    _, schedules, player, _, _ = load_inputs(path, REPO / "data")
    predictions = build_predictions_frame(per_game, schedules)
    overlays = run_overlays(predictions, schedules, player)
    arrests, _ = reconstruct_arrest_flip_set(per_game, DEFAULT_FEATURES, DEFAULT_INCIDENTS)
    no_zone = set(arrests)
    for member in ("coach_fade_overlay", "division_revenge_tilt_overlay"):
        no_zone.update(flip.game_id for flip in overlays[member].flips)
    zone = {flip.game_id for flip in overlays["spread_gap_zone_fade_overlay"].flips}
    base = per_game.pick_home_at_open_probability_rule.to_numpy(dtype=bool)
    return base ^ per_game.game_id.isin(no_zone | zone), base ^ per_game.game_id.isin(no_zone)


def composition(out: Path, archive: pd.DataFrame, stream: pd.DataFrame, result: dict) -> None:
    winner = result["winner"]
    candidate = archive.copy()
    indexed = stream.set_index("game_id").reindex(archive.game_id)
    candidate["home_cover_probability_at_open"] = indexed[f"p_{winner}"].to_numpy()
    residual_name = (
        winner if winner in {"C1", "C2", "C1_C2"} else ("base" if winner == "C3" else "C1")
    )
    candidate["residual_at_open"] = indexed[f"residual_{residual_name}"].to_numpy()
    candidate["pick_home_at_open"] = candidate.residual_at_open.gt(0)
    candidate["correct_at_open"] = (
        candidate.pick_home_at_open.eq(candidate.margin_vs_open.gt(0))
        .astype(float)
        .where(candidate.margin_vs_open.ne(0))
    )
    candidate["pick_home_at_open_probability_rule"] = candidate.home_cover_probability_at_open.ge(
        0.5
    )
    candidate["correct_at_open_probability_rule"] = (
        candidate.pick_home_at_open_probability_rule.eq(candidate.margin_vs_open.gt(0))
        .astype(float)
        .where(candidate.margin_vs_open.ne(0))
    )
    directory = out / "artifacts/opener_evaluation/candidate"
    table(candidate, directory / "per_game.parquet")
    metadata = json.loads((ARCHIVE / "metadata.json").read_text())
    metadata.update(
        research_arm=winner,
        incumbent_model_id=metadata.get("active_model_id"),
        active_model_id=f"research_mod18_{winner}",
    )
    method = "discrete_residual" if winner in {"C2", "C1_C2"} else "gaussian_median"
    profile = "weak_stack_spread_regime" if winner.startswith("C1") else "weak_stack"
    metadata["probability_method"] = method
    metadata["active_model_config"] = {
        **metadata["active_model_config"],
        "probability_method": method,
        "feature_profile": profile,
    }
    metadata["research_scope"] = "Opener columns recomputed; close columns retained from incumbent."
    write_stamped_artifact(metadata, directory / "metadata.json")
    cli(
        out,
        "overlay-composition",
        "--per-game-artifact",
        str(directory / "per_game.parquet"),
        "--bootstrap-samples",
        "20000",
        "--bootstrap-seed",
        str(SEED),
    )
    incumbent_four, _ = composed_picks(archive, ARCHIVE / "per_game.parquet")
    candidate_four, candidate_three = composed_picks(candidate, directory / "per_game.parquet")
    pairs = {
        "candidate_four_vs_incumbent_four": (candidate_four, incumbent_four),
        "candidate_three_vs_incumbent_four": (candidate_three, incumbent_four),
        "candidate_zone_marginal": (candidate_four, candidate_three),
    }
    metrics = {name: comparison(archive, cp, bp) for name, (cp, bp) in pairs.items()}
    path = out / "composition.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous is not None and previous["comparisons"] != metrics:
        raise ValueError("Existing composition changed; declare a new look before rescoring")
    write_stamped_artifact({"winner": winner, "comparisons": metrics}, path)
    if previous is None:
        for name, m in metrics.items():
            record_arm(out, name, m, 2020, 2025, path)
    table(
        archive[["game_id", "season", "week"]].assign(
            incumbent_four=incumbent_four,
            candidate_four=candidate_four,
            candidate_three=candidate_three,
        ),
        out / "composed_picks.parquet",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("build", "score", "composition", "diagnosis"), required=True
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(REPO / "data") or out.is_relative_to(REPO / "artifacts"):
        raise ValueError("Use an isolated scratch output root")
    os.environ["NFL_ATS_ARTIFACTS_DIR"] = str(out / "artifacts")
    os.environ["NFL_ATS_REGISTRY_DIR"] = str(out / "registry")
    archive = pd.read_parquet(ARCHIVE / "per_game.parquet")
    if args.stage == "build":
        active = json.loads((REPO / "artifacts/active_ats_model.json").read_text())
        if (
            active["model_id"] != "a4c757efd2525da6"
            or sha256_file(FEATURES) != active["feature_table_sha256"]
        ):
            raise ValueError("Frozen active model or feature source changed")
        features = attach_spread_regime(pd.read_parquet(FEATURES))
        table(features, out / "features.parquet")
        with threadpool_limits(limits=1):
            build_stream(features, archive, out)
    else:
        stream = pd.read_parquet(out / "stream.parquet")
        if args.stage == "score":
            print(json.dumps(score(out, archive, stream), indent=2))
        elif args.stage == "diagnosis":
            diagnosis(out, archive, stream)
        else:
            composition(out, archive, stream, json.loads((out / "results.json").read_text()))


if __name__ == "__main__":
    main()
