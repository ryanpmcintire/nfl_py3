"""MOD-14 promotion confirmation look: half_life_8 era weighting vs uniform
baseline at the Tuesday opener, on the rotation-registry window assigned to
family ``era_weighting_half_life_8``.

Predeclaration: ``docs/era_weighting_promotion.md`` (written before declare,
assign, and this run). Primary endpoint: week-blocked paired forced-pick
accuracy improvement (production probability rule) on the assigned window's
paired opener games. Secondary: Brier/log-loss improvement, direction only.
Decision rule: play decision under expected value; the predeclared P+ >= 0.90
promotion-claim threshold governs only what docs may CLAIM, never which card
is played.

Machinery mirrors ``scripts/era_weighting_opener_read.py``'s weekly-refit
archive/pairing loop, restricted to the assigned window via
``nfl_ats.rotation.confirmation_split`` (the ``scripts/mod07_weak_stack.py``
split discipline). Self-check: the baseline arm's window-season predictions
must reproduce the Section 8 information-read artifact's baseline predictions
game-for-game before the candidate arm is interpreted.

Usage::

    uv run python scripts/era_weighting_promotion_look.py --output <dir>
    uv run python scripts/era_weighting_promotion_look.py --record <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from era_weighting_lib import (
    ERA_WEIGHTING_ARMS,
    WeightingArm,
    fit_weighted_ridge_margin,
    half_life_weights,
)

from nfl_ats.clv import CLOSE_LABEL_PRIORITY, build_pairing_table, close_reference_table
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.margin import MarginModel, margin_feature_columns
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact
from nfl_ats.rotation import confirmation_split, load_registry

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"
DEFAULT_MARKET_ROOT = REPO / "data/market/raw"
SECTION8_PREDICTIONS = (
    REPO / "artifacts/era_weighting_opener_read/20260820T002230Z/predictions.parquet"
)
PREDECLARATION = "docs/era_weighting_promotion.md"
FAMILY = "era_weighting_half_life_8"

FEATURE_PROFILE = "weak_stack"
REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0
TARGET = "market_residual"
DISTRIBUTION_FRACTION = 0.20

BASELINE_ARM_NAME = "baseline"
CANDIDATE_ARM_NAME = "half_life_8"
LOOK_ARMS: tuple[WeightingArm, ...] = tuple(
    arm for arm in ERA_WEIGHTING_ARMS if arm.name in (BASELINE_ARM_NAME, CANDIDATE_ARM_NAME)
)
assert {arm.name for arm in LOOK_ARMS} == {BASELINE_ARM_NAME, CANDIDATE_ARM_NAME}

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819
CONFIRM_AT = 0.90
SELF_CHECK_TOLERANCE = 1e-9

CLOSING_RULE_NOTE = (
    "Binding taxonomy: an interval containing zero is NEVER grounds to reject/close "
    "(expected shape for a real small signal at this evaluator's resolution). Only a "
    "RESOLVED wrong sign (whole interval below zero) or a positive-control bound may "
    "close a line of work; no positive control was run, so only wrong_sign_resolved is "
    "ever available and only when the whole week-blocked interval sits below zero. "
    "Otherwise: unresolved_below_power, reported with probability_positive."
)

SELECTION_DISCLOSURE = (
    "(a) SELECTION INFLATION: half_life_8 was selected best-of-six on two screens "
    "(CFB clean-core week-blocked P+ 0.8987; NFL close-grade week-blocked P+ 0.8505 / "
    "season-blocked P+ 0.9533) and read a third time at the opener (P+ 0.2990, "
    "docs/era_weighting_screen.md Section 8) -- this confirmation is the arm's FOURTH "
    "look, not an independent blind draw. "
    "(b) POPULATION OVERLAP: the opener pool is 2020-2025 and the Section 8 information "
    "read scored ALL of it, so this window's games were already seen once; the two "
    "reads' P+ figures are not independent and are never multiplied. "
    "(c) MINED-LEDGER DISCOUNT: the window intersects 2018-2025 (~130-150-look ledger, "
    "ROADMAP RWB-16), acknowledged at declaration. "
    "(d) CROSS-FAMILY REUSE: [2020, 2021] was already spent by mod07_weak_signal_stack "
    "and best_pick_ranker_opener; rule 4 permits per-family retirement, global "
    "multiplicity on these seasons rises accordingly."
)


def run_window_walk_forward(
    market_root: Path,
    scoped: pd.DataFrame,
    *,
    window_seasons: tuple[int, ...],
    arms: tuple[WeightingArm, ...],
    min_train_games: int,
) -> pd.DataFrame:
    feature_columns = margin_feature_columns(TARGET, FEATURE_PROFILE)
    required = {
        "game_id",
        "season",
        "week",
        "gameday",
        "result",
        "ats_margin",
        "spread_line",
        *feature_columns,
    }
    missing = sorted(required.difference(scoped.columns))
    if missing:
        raise DataContractError(f"Promotion look is missing columns: {', '.join(missing)}")
    frame = scoped.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")

    pairing = build_pairing_table(
        market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=frame,
    )
    if pairing.empty:
        raise ValueError(
            f"No {HISTORICAL_CAPTURE_KIND!r} snapshots with decision quotes under {market_root}"
        )
    close = close_reference_table(pairing, frame)
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "season", "week", "home_spread"]
    ].rename(columns={"home_spread": "tue_open_home_spread"})
    paired = tue_open.merge(close, on="game_id", how="inner")
    paired = paired.loc[paired["season"].astype(int).isin(window_seasons)].copy()

    outcomes = frame[["game_id", "result"]].drop_duplicates("game_id")
    paired = paired.merge(outcomes, on="game_id", how="inner")
    paired = paired.loc[pd.to_numeric(paired["result"], errors="coerce").notna()].copy()
    if paired.empty:
        raise ValueError("No completed window games have both a Tuesday opener and a close")

    completed = frame.loc[frame["result"].notna()].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    skip_counts: dict[str, int] = {arm.name: 0 for arm in arms}
    rows: list[pd.DataFrame] = []
    for (season, week), group in paired.groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"]))]
        if week_rows.empty:
            continue
        cutoff = week_rows["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        predict_season = int(str(season))
        target_full = pd.to_numeric(training["ats_margin"], errors="raise").to_numpy(dtype=float)
        seasons_full = training["season"].to_numpy(dtype=float)

        scoring = week_rows.merge(
            group[["game_id", "tue_open_home_spread"]],
            on="game_id",
            how="inner",
        ).copy()

        models: dict[str, MarginModel] = {}
        for arm in arms:
            weights = (
                np.ones(len(training), dtype=float)
                if arm.kind == "uniform"
                else half_life_weights(seasons_full, predict_season, arm.parameter or 1.0)
            )
            try:
                models[arm.name] = fit_weighted_ridge_margin(
                    training,
                    target=target_full,
                    feature_columns=feature_columns,
                    weights=weights,
                    ridge_alpha=RIDGE_ALPHA,
                    distribution_fraction=DISTRIBUTION_FRACTION,
                    min_distribution_rows=10,
                    random_state=42,
                    model_name=REGRESSOR,
                )
            except ValueError:
                skip_counts[arm.name] += 1

        base = scoring[["game_id", "result", "tue_open_home_spread"]].copy()
        base["season"] = predict_season
        base["week"] = int(str(week))
        margin_open = base["result"] - base["tue_open_home_spread"]
        base["home_cover_open"] = np.select(
            [margin_open > 0, margin_open < 0], [1.0, 0.0], default=np.nan
        )

        for arm_name, model in models.items():
            at_open = scoring.copy()
            at_open["spread_line"] = at_open["tue_open_home_spread"]
            predicted_open = model.predict(at_open)
            batch = base.copy()
            batch["feature_set"] = arm_name
            batch["residual_open"] = predicted_open["predicted_market_residual"].to_numpy(
                dtype=float
            )
            batch["home_cover_probability_open"] = predicted_open[
                "home_cover_probability"
            ].to_numpy(dtype=float)
            batch["distribution_rows"] = model.distribution_rows
            batch["training_rows"] = model.training_rows
            rows.append(batch)

    if not rows:
        raise ValueError("No paired window week had enough prior training games")
    predictions = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["game_id", "feature_set"])
        .reset_index(drop=True)
    )
    predictions.attrs["skip_counts"] = skip_counts
    return predictions


def self_check(baseline_predictions: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    scored = baseline_predictions.loc[baseline_predictions["home_cover_open"].notna()]
    own = scored[["game_id", "home_cover_probability_open"]].rename(
        columns={"home_cover_probability_open": "own_probability"}
    )
    ref = reference.loc[
        reference["feature_set"].eq(BASELINE_ARM_NAME) & reference["home_cover_open"].notna(),
        ["game_id", "home_cover_probability_open"],
    ].rename(columns={"home_cover_probability_open": "reference_probability"})
    merged = own.merge(ref, on="game_id", how="inner")
    if len(merged) != len(own):
        raise SystemExit(
            "Self-check failed: baseline game set does not match the Section 8 "
            f"artifact game-for-game (own={len(own)}, reference_window_games="
            f"{len(ref)}, merged={len(merged)}). This is a bug in this script's "
            "adaptation, not a finding -- fix before interpreting half_life_8."
        )
    max_diff = float((merged["own_probability"] - merged["reference_probability"]).abs().max())
    picks_agree = bool(
        merged["own_probability"].ge(0.5).equals(merged["reference_probability"].ge(0.5))
    )
    return {
        "games": len(merged),
        "max_probability_abs_diff": max_diff,
        "probability_rule_picks_all_agree": picks_agree,
        "tolerance": SELF_CHECK_TOLERANCE,
        "reference": str(SECTION8_PREDICTIONS),
    }


def paired_report(predictions: pd.DataFrame, *, samples: int, seed: int) -> pd.DataFrame:
    view = predictions.rename(
        columns={
            "home_cover_open": "home_cover",
            "home_cover_probability_open": "home_cover_probability",
        }
    )[["game_id", "season", "week", "feature_set", "home_cover", "home_cover_probability"]]
    rows: list[pd.DataFrame] = []
    for block in ("week", "season"):
        result = paired_feature_comparisons(
            view,
            baseline_feature_set=BASELINE_ARM_NAME,
            samples=samples,
            block=block,
            seed=seed,
        )
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def pct(fraction: float) -> float:
    return fraction * 100.0


def classify(lower_pts: float, upper_pts: float) -> tuple[str, str | None, str]:
    if upper_pts < 0.0:
        return (
            "refuted_mechanism",
            "wrong_sign_resolved",
            f"Whole week-blocked interval [{lower_pts:+.4f}, {upper_pts:+.4f}] sits below "
            "zero -- the only admissible closing ground available (no positive control "
            "was run). " + CLOSING_RULE_NOTE,
        )
    return (
        "unresolved_below_power",
        None,
        f"Week-blocked interval [{lower_pts:+.4f}, {upper_pts:+.4f}] does not sit entirely "
        "below zero -- neither admissible closing ground applies (no positive control was "
        "run; a wholly-above-zero interval has no resolved-positive terminal state in "
        "this taxonomy either). " + CLOSING_RULE_NOTE,
    )


def rotation_verdict(lower_pts: float, upper_pts: float, probability_positive: float) -> str:
    if upper_pts < 0.0:
        return "closed_negative"
    if probability_positive >= CONFIRM_AT:
        return "confirmed"
    return "unresolved"


def run_look(args: argparse.Namespace) -> None:
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    registry = load_registry(args.registry)
    features = pd.read_parquet(args.features)
    training, window = confirmation_split(features, registry, FAMILY)
    window_seasons = tuple(sorted(int(s) for s in window["season"].unique()))
    print(f"assigned window seasons: {window_seasons}")
    scoped = pd.concat([training, window], ignore_index=True)

    print("=== self-check: baseline arm vs Section 8 artifact, BEFORE half_life_8 ===")
    baseline_only = run_window_walk_forward(
        args.market_root,
        scoped,
        window_seasons=window_seasons,
        arms=(next(a for a in LOOK_ARMS if a.name == BASELINE_ARM_NAME),),
        min_train_games=args.min_train_games,
    )
    reference = pd.read_parquet(SECTION8_PREDICTIONS)
    checks = self_check(baseline_only, reference)
    print(json.dumps(checks, indent=2))
    if (
        not checks["probability_rule_picks_all_agree"]
        or checks["max_probability_abs_diff"] > SELF_CHECK_TOLERANCE
    ):
        raise SystemExit(
            "Self-check failed: baseline arm does not reproduce the Section 8 "
            "information-read artifact on the window games. Per "
            "docs/era_weighting_promotion.md this is a bug in this script's "
            "adaptation, not a finding -- fix before trusting half_life_8."
        )
    print("self-check passes -- proceeding to compute half_life_8.\n")

    predictions = run_window_walk_forward(
        args.market_root,
        scoped,
        window_seasons=window_seasons,
        arms=LOOK_ARMS,
        min_train_games=args.min_train_games,
    )
    predictions.to_parquet(output / "predictions.parquet", index=False)
    print(f"scored rows: {len(predictions)}; skip counts: {predictions.attrs['skip_counts']}")

    paired = paired_report(predictions, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    paired.to_csv(output / "paired_comparisons.csv", index=False)

    primary = paired.loc[
        paired["metric"].eq("accuracy_improvement") & paired["block"].eq("week")
    ].iloc[0]
    season_row = paired.loc[
        paired["metric"].eq("accuracy_improvement") & paired["block"].eq("season")
    ].iloc[0]
    scored = predictions.loc[
        predictions["feature_set"].eq(CANDIDATE_ARM_NAME) & predictions["home_cover_open"].notna()
    ]
    baseline_scored = predictions.loc[
        predictions["feature_set"].eq(BASELINE_ARM_NAME) & predictions["home_cover_open"].notna()
    ]

    diagnostics: dict[str, Any] = {
        "predeclaration": PREDECLARATION,
        "family": FAMILY,
        "grade": "opener",
        "window_seasons": list(window_seasons),
        "games": int(scored["game_id"].nunique()),
        "weeks": int(scored.groupby(["season", "week"]).ngroups),
        "baseline_accuracy": float(baseline_scored["home_cover_open"].mean()),
        "candidate_accuracy": float(scored["home_cover_open"].mean()),
        "primary": {
            "metric": "accuracy_improvement, probability rule, week-blocked",
            "estimate_points": pct(float(primary["estimate"])),
            "lower_points": pct(float(primary["lower"])),
            "upper_points": pct(float(primary["upper"])),
            "probability_positive": float(primary["probability_positive"]),
            "paired_games": int(primary.get("paired_games", primary.get("games", 0))),
            "blocks": int(primary["blocks"]),
        },
        "season_blocked_context_degenerate_two_blocks": {
            "estimate_points": pct(float(season_row["estimate"])),
            "probability_positive": float(season_row["probability_positive"]),
        },
        "secondary_direction_only": {
            row["metric"]: {
                "estimate": float(row["estimate"]),
                "lower": float(row["lower"]),
                "upper": float(row["upper"]),
                "probability_positive": float(row["probability_positive"]),
            }
            for _, row in paired.loc[
                paired["metric"].isin(["brier_improvement", "log_loss_improvement"])
                & paired["block"].eq("week")
            ].iterrows()
        },
        "decision_rule": (
            "Play decision under expected value graded at the opener; the predeclared "
            f"P+ >= {CONFIRM_AT} promotion-claim threshold governs only what docs may "
            "CLAIM, never which card is played (AGENTS.md, 'a promotion bar is not a "
            "decision bar')."
        ),
        "selection_disclosure": SELECTION_DISCLOSURE,
        "recipe": {
            "target": TARGET,
            "regressor": REGRESSOR,
            "ridge_alpha": RIDGE_ALPHA,
            "feature_profile": FEATURE_PROFILE,
            "min_train_games": args.min_train_games,
            "feature_table": str(args.features),
            "market_root": str(args.market_root),
        },
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "skip_counts": predictions.attrs["skip_counts"],
        "self_check": checks,
        "arms": [arm.name for arm in LOOK_ARMS],
    }
    configuration = {
        "command": "era-weighting-promotion-look",
        "features": str(args.features),
        "market_root": str(args.market_root),
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "registry": str(args.registry),
    }
    diagnostics["provenance"] = artifact_provenance(configuration, args.features, project_root=REPO)
    write_experiment_artifact(
        output,
        "diagnostics.json",
        diagnostics,
        command="era-weighting-promotion-look",
        metrics=diagnostics,
        notes=(
            "MOD-14 promotion confirmation: half_life_8 vs baseline at the opener on "
            "rotation window [2020, 2021]; play decision under EV, claim threshold "
            "P+ >= 0.90 governs claims only."
        ),
    )

    print("\n=== PRIMARY: opener grade, probability rule, week-blocked, paired ===")
    print(
        paired.loc[
            paired["metric"].eq("accuracy_improvement") & paired["block"].eq("week"),
            ["metric", "estimate", "lower", "upper", "probability_positive", "paired_games"],
        ].to_string(index=False)
    )
    print("\n=== secondary/context ===")
    print(
        paired.loc[
            :,
            ["block", "metric", "estimate", "lower", "upper", "probability_positive"],
        ].to_string(index=False)
    )
    print(f"\nartifacts: {output}")


def record_look(output: Path) -> None:
    paired = pd.read_csv(output / "paired_comparisons.csv")
    diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
    primary = paired.loc[
        paired["metric"].eq("accuracy_improvement") & paired["block"].eq("week")
    ].iloc[0]
    estimate_pts = pct(float(primary["estimate"]))
    lower_pts = pct(float(primary["lower"]))
    upper_pts = pct(float(primary["upper"]))
    p_plus = float(primary["probability_positive"])
    paired_games = int(primary.get("paired_games", primary.get("games", 0)))
    blocks = int(primary["blocks"])
    classification, closing_ground, evidence = classify(lower_pts, upper_pts)
    verdict = rotation_verdict(lower_pts, upper_pts, p_plus)
    seasons = diagnostics["window_seasons"]
    artifact_ref = f"{PREDECLARATION}; {output.as_posix()}"
    season_row = paired.loc[
        paired["metric"].eq("accuracy_improvement") & paired["block"].eq("season")
    ].iloc[0]
    season_estimate_pts = pct(float(season_row["estimate"]))
    season_p_plus = float(season_row["probability_positive"])

    notes = (
        f"PRIMARY (opener, production probability rule, week-blocked): {estimate_pts:+.4f} pts, "
        f"95% [{lower_pts:+.4f}, {upper_pts:+.4f}] pts, P+={p_plus:.4f}, paired games="
        f"{paired_games}, blocks={blocks}. Baseline arm reproduced the Section 8 "
        "information-read artifact game-for-game (max |prob diff| "
        f"{diagnostics['self_check']['max_probability_abs_diff']:.2e}) before half_life_8 "
        f"was interpreted. Absolute accuracy: baseline {diagnostics['baseline_accuracy']:.4f}, "
        f"candidate {diagnostics['candidate_accuracy']:.4f}. Season-blocked context "
        f"(degenerate, 2 blocks): {season_estimate_pts:+.4f} pts, P+={season_p_plus:.4f}. "
        f"Secondary (direction only): {json.dumps(diagnostics['secondary_direction_only'])}. "
        f"SELECTION DISCLOSURE: {SELECTION_DISCLOSURE} DECISION RULE: play decision under "
        f"expected value graded at the opener; P+ >= {CONFIRM_AT} governs claims only. "
        + CLOSING_RULE_NOTE
    )

    cmd = [
        sys.executable,
        "-m",
        "nfl_ats.cli",
        "rotation",
        "record",
        "--name",
        FAMILY,
        "--artifact",
        artifact_ref,
        "--verdict",
        verdict,
        "--probability-positive",
        f"{p_plus:.10f}",
        "--effect",
        f"{estimate_pts:.10f}",
        "--effect-units",
        "accuracy_points",
        "--interval-low",
        f"{lower_pts:.10f}",
        "--interval-high",
        f"{upper_pts:.10f}",
        "--sample-blocks",
        str(blocks),
        "--notes",
        notes,
    ]
    if closing_ground is not None:
        cmd += ["--closing-ground", closing_ground]
    print("=== rotation record ===")
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(
            "rotation record failed; per AGENTS.md, if a record command errors the "
            "verdict is wrong, not the validator -- fix the invocation, do not weaken "
            "the classification."
        )

    weak = {
        "name": "era_weighting_half_life_8_opener_confirmation",
        "description": (
            "MOD-14 promotion confirmation (docs/era_weighting_promotion.md, predeclared "
            "2026-08-21): exponential season-decay sample weighting at an 8-season half-life "
            "vs a uniform-weight all-history baseline, frozen market-residual Ridge recipe "
            "(weak_stack/ridge/alpha=10.0 unchanged), graded at the Tuesday opener on the "
            "rotation-registry window assigned to family era_weighting_half_life_8."
        ),
        "source": artifact_ref,
        "effect": f"{estimate_pts:.10f}",
        "effect_units": "accuracy_points",
        "classification": classification,
        "league": "nfl",
        "season_start": str(min(seasons)),
        "season_end": str(max(seasons)),
        "interval_low": f"{lower_pts:.10f}",
        "interval_high": f"{upper_pts:.10f}",
        "probability_positive": f"{p_plus:.10f}",
        "sample_games": str(paired_games),
        "sample_blocks": str(blocks),
        "classification_evidence": evidence,
        "notes": notes,
    }
    cmd = [sys.executable, "-m", "nfl_ats.cli", "weak-signals", "record"]
    for key, value in weak.items():
        cmd += ["--" + key.replace("_", "-"), value]
    print("=== weak-signals record ===")
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(
            "weak-signals record failed; per AGENTS.md, if a record command errors the "
            "verdict is wrong, not the validator -- fix the invocation, do not weaken "
            "the classification."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--record", type=Path, default=None, help="artifact dir to record from")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    if args.record is not None:
        record_look(args.record)
        return
    if args.output is None:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output = REPO / "artifacts" / "era_weighting_promotion_look" / run_id
    run_look(args)


if __name__ == "__main__":
    main()
