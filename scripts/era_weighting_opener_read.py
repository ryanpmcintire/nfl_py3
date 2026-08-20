"""MOD-14 opener-grade information read: half_life_8 era-weighting arm at
the 1,537-paired-game opener protocol.

INFORMATION READ, NOT A PROMOTION. `half_life_8` was selected today
(2026-08-19) as the best-of-six MOD-14 grid arm on two below-power screens
(``docs/era_weighting_screen.md`` Sections 3-6): CFB clean-core week-blocked
P+ 0.8987, NFL close-grade week-blocked P+ 0.8505 / season-blocked P+
0.9533. Neither resolves under the binding taxonomy (every interval crosses
zero). This script asks a THIRD, disclosed question: does the selected
arm's lean survive at the OPENER, on the project's own decision-grade
protocol (``docs/opener_evaluation.md``, AGENTS.md "grade the decision at
the opener" rule)? Selection inflation, population overlap with Section 6,
and below-power status are all disclosed here and in the registry entry --
this number is read-only context for a FUTURE promotion decision, not
itself one.

Binding closing-grounds taxonomy (verbatim, per AGENTS.md/CLAUDE.md):
An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that
size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator.

Predeclaration: ``docs/era_weighting_screen.md`` Section 8 (written and
committed to the doc BEFORE this script ran).

Protocol: mirrors ``scripts/smooth_cdf_mapping_opener_measurement.py``'s own
weekly-refit archive/pairing machinery (``docs/opener_evaluation.md``'s
1,537-paired-game 2020-2025 ``tue_open``/close archive) combined with
``scripts/era_weighting_lib.py``'s arm-fitting machinery (identical to
``scripts/era_weighting_nfl_screen.py``). Two arms per week, fit on
IDENTICAL strictly-earlier training rows: ``baseline`` (uniform
``sample_weight=1``) and ``half_life_8`` (exponential season-decay sample
weight, half-life 8 seasons) -- both scored at the opener AND the close,
under BOTH the production probability rule (PRIMARY,
``home_cover_probability >= 0.5``, model's default ECDF read -- the
identical mapping ``nfl_ats.clv.opener_pick_evaluation`` has always used)
and the sign rule (SECONDARY, ``predicted_market_residual > 0``,
diagnostic). Self-check runs the ``baseline`` arm alone FIRST and compares
its opener probability-rule accuracy against ``docs/opener_evaluation.md``'s
53.3599% production number before the ``half_life_8`` arm is computed at
all. Rotation registry: untouched (rule 8 -- this reads the frozen,
already-public opener/close archive, not a reserved season).
"""

from __future__ import annotations

import argparse
import json
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
from nfl_ats.modeling import regular_season_rows
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"
DEFAULT_MARKET_ROOT = REPO / "data/market/raw"

# Production recipe, identical to docs/opener_evaluation.md and
# scripts/smooth_cdf_mapping_opener_measurement.py's "weak_stack profile,
# the ACTIVE model" run.
FEATURE_PROFILE = "weak_stack"
REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0
TARGET = "market_residual"
DISTRIBUTION_FRACTION = 0.20

BASELINE_ARM_NAME = "baseline"
CANDIDATE_ARM_NAME = "half_life_8"
READ_ARMS: tuple[WeightingArm, ...] = tuple(
    arm for arm in ERA_WEIGHTING_ARMS if arm.name in (BASELINE_ARM_NAME, CANDIDATE_ARM_NAME)
)
assert {arm.name for arm in READ_ARMS} == {BASELINE_ARM_NAME, CANDIDATE_ARM_NAME}

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819

# docs/opener_evaluation.md "Addendum, 2026-08-19": production-rule opener
# accuracy on the frozen 1,537-paired-game archive. Read this session from
# artifacts/opener_evaluation/20260819T174244Z/metadata.json
# ("metrics" -> "opener_accuracy_probability_rule").
PRODUCTION_OPENER_ACCURACY_PROBABILITY_RULE = 0.5335994677312043
SELF_CHECK_TOLERANCE = 1e-6


def run_opener_walk_forward(
    market_root: Path,
    features: pd.DataFrame,
    *,
    arms: tuple[WeightingArm, ...],
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> pd.DataFrame:
    """One row per (paired game, arm in `arms`) at the opener protocol.

    Reuses ``docs/opener_evaluation.md``'s exact pairing archive and
    weekly-refit cutoff logic; the only new logic is fitting one
    ``era_weighting_lib``-weighted model per arm per week (instead of the
    single frozen model ``opener_pick_evaluation`` fits) and reading both
    the probability-rule and sign-rule picks off each arm's own fit, at
    both the opener and close lines.
    """

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
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(
            f"Opener era-weighting read is missing columns: {', '.join(missing)}"
        )
    frame = regular_season_rows(features).copy()
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

    outcomes = frame[["game_id", "result"]].drop_duplicates("game_id")
    paired = paired.merge(outcomes, on="game_id", how="inner")
    paired = paired.loc[pd.to_numeric(paired["result"], errors="coerce").notna()].copy()
    if paired.empty:
        raise ValueError("No completed games have both a Tuesday opener and a close")

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
            group[["game_id", "tue_open_home_spread", "close_home_spread"]],
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

        base = scoring[["game_id", "result", "tue_open_home_spread", "close_home_spread"]].copy()
        base["season"] = predict_season
        base["week"] = int(str(week))
        margin_open = base["result"] - base["tue_open_home_spread"]
        margin_close = base["result"] - base["close_home_spread"]
        base["home_cover_open"] = np.select(
            [margin_open > 0, margin_open < 0], [1.0, 0.0], default=np.nan
        )
        base["home_cover_close"] = np.select(
            [margin_close > 0, margin_close < 0], [1.0, 0.0], default=np.nan
        )

        for arm_name, model in models.items():
            at_open = scoring.copy()
            at_open["spread_line"] = at_open["tue_open_home_spread"]
            at_close = scoring.copy()
            at_close["spread_line"] = at_close["close_home_spread"]
            # probability_method left at its default ("ecdf") deliberately --
            # matches nfl_ats.clv.opener_pick_evaluation's own call exactly,
            # so this is the same probability mapping behind the production
            # 53.36% number, not the MOD-08 Gaussian default (which only
            # nfl_ats.outcomes.score_outcome_week passes explicitly).
            predicted_open = model.predict(at_open)
            predicted_close = model.predict(at_close)

            batch = base.copy()
            batch["feature_set"] = arm_name
            batch["residual_open"] = predicted_open["predicted_market_residual"].to_numpy(
                dtype=float
            )
            batch["residual_close"] = predicted_close["predicted_market_residual"].to_numpy(
                dtype=float
            )
            batch["home_cover_probability_open"] = predicted_open[
                "home_cover_probability"
            ].to_numpy(dtype=float)
            batch["home_cover_probability_close"] = predicted_close[
                "home_cover_probability"
            ].to_numpy(dtype=float)
            batch["sign_pick_open"] = (batch["residual_open"] > 0.0).astype(float)
            batch["sign_pick_close"] = (batch["residual_close"] > 0.0).astype(float)
            batch["distribution_rows"] = model.distribution_rows
            batch["training_rows"] = model.training_rows
            rows.append(batch)

    if not rows:
        raise ValueError("No paired week had enough prior training games")
    predictions = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["game_id", "feature_set"])
        .reset_index(drop=True)
    )
    predictions.attrs["skip_counts"] = skip_counts
    return predictions


def self_check(baseline_predictions: pd.DataFrame) -> dict[str, Any]:
    """Baseline arm alone vs. docs/opener_evaluation.md's production number.

    Run and printed BEFORE the half_life_8 arm is computed at all -- this
    function's caller in ``main`` invokes ``run_opener_walk_forward`` with
    ``arms=(baseline,)`` only, matching the predeclared discipline in
    ``docs/era_weighting_screen.md`` Section 8.
    """

    scored = baseline_predictions.loc[baseline_predictions["home_cover_open"].notna()]
    own_accuracy = float(
        (scored["home_cover_probability_open"].ge(0.5) == scored["home_cover_open"]).mean()
    )
    diff = own_accuracy - PRODUCTION_OPENER_ACCURACY_PROBABILITY_RULE
    return {
        "own_games": len(scored),
        "own_accuracy_probability_rule": own_accuracy,
        "production_reference": PRODUCTION_OPENER_ACCURACY_PROBABILITY_RULE,
        "production_reference_source": (
            "artifacts/opener_evaluation/20260819T174244Z/metadata.json "
            "(metrics.opener_accuracy_probability_rule), quoted as 53.36% in "
            "docs/opener_evaluation.md's Addendum, 2026-08-19"
        ),
        "accuracy_diff": diff,
        "within_tolerance": bool(abs(diff) <= SELF_CHECK_TOLERANCE),
        "tolerance": SELF_CHECK_TOLERANCE,
    }


def _rule_view(predictions: pd.DataFrame, *, grade: str, rule: str) -> pd.DataFrame:
    """One (grade, rule) view in paired_feature_comparisons' required shape."""

    if rule == "probability":
        probability_column = f"home_cover_probability_{grade}"
    elif rule == "sign":
        probability_column = f"sign_pick_{grade}"
    else:
        raise ValueError(f"Unknown rule: {rule!r}")
    frame = predictions.rename(
        columns={
            f"home_cover_{grade}": "home_cover",
            probability_column: "home_cover_probability",
        }
    )[["game_id", "season", "week", "feature_set", "home_cover", "home_cover_probability"]]
    return frame


def paired_report(predictions: pd.DataFrame, *, samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for grade in ("open", "close"):
        for rule in ("probability", "sign"):
            view = _rule_view(predictions, grade=grade, rule=rule)
            for block in ("week", "season"):
                result = paired_feature_comparisons(
                    view,
                    baseline_feature_set=BASELINE_ARM_NAME,
                    samples=samples,
                    block=block,
                    seed=seed,
                )
                result.insert(0, "grade", grade)
                result.insert(1, "rule", rule)
                rows.append(result)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(args.features)

    baseline_arm = next(arm for arm in READ_ARMS if arm.name == BASELINE_ARM_NAME)
    print("=== self-check: baseline arm alone, BEFORE half_life_8 is computed ===")
    baseline_only = run_opener_walk_forward(
        args.market_root,
        features,
        arms=(baseline_arm,),
        min_train_games=args.min_train_games,
    )
    checks = self_check(baseline_only)
    print(json.dumps(checks, indent=2))
    if not checks["within_tolerance"]:
        raise SystemExit(
            "Self-check failed: baseline arm does not reproduce "
            "docs/opener_evaluation.md's production-rule opener accuracy within "
            f"tolerance ({checks['accuracy_diff']:+.10f} vs {SELF_CHECK_TOLERANCE}). "
            "Per docs/era_weighting_screen.md Section 8 this is a bug in this "
            "script's reimplementation, not a finding -- fix before trusting "
            "half_life_8's numbers."
        )
    print("self-check passes -- proceeding to compute half_life_8.\n")

    predictions = run_opener_walk_forward(
        args.market_root,
        features,
        arms=READ_ARMS,
        min_train_games=args.min_train_games,
    )
    predictions.to_parquet(output / "predictions.parquet", index=False)
    print(f"scored rows: {len(predictions)}; skip counts: {predictions.attrs['skip_counts']}")

    paired = paired_report(predictions, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    paired.to_csv(output / "paired_comparisons.csv", index=False)

    games = int(predictions["game_id"].nunique())
    diagnostics: dict[str, Any] = {
        "predeclaration": "docs/era_weighting_screen.md#8-opener-grade-information-read",
        "information_read_not_a_promotion": True,
        "rotation_registry_touched": False,
        "grade": "opener",
        "games": games,
        "selection_disclosure": {
            "a_selection_inflation": (
                "half_life_8 was selected best-of-six today (2026-08-19) on two "
                "below-power screens (CFB clean-core week-blocked P+ 0.8987, NFL "
                "close-grade week-blocked P+ 0.8505 / season-blocked P+ 0.9533) "
                "recorded in docs/era_weighting_screen.md; this is the arm's THIRD "
                "look, not an independent blind draw."
            ),
            "b_population_overlap": (
                "This opener archive is 2020-2025; docs/era_weighting_screen.md "
                "Section 6's NFL close-grade population includes 2022-2025 from the "
                "same span -- not disjoint from this read."
            ),
            "c_below_power_not_confirmation": (
                "No rotation-registry window spent (rule 8 -- frozen public "
                "archive, not a reserved season). Read-only information for a "
                "future promotion decision; makes no promotion decision itself."
            ),
        },
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
        "arms": [arm.name for arm in READ_ARMS],
        "primary_metric": (
            "accuracy_improvement, grade=open, rule=probability, block=week "
            "(production pick rule, at the opener)"
        ),
    }
    configuration = {
        "command": "era-weighting-opener-read",
        "features": str(args.features),
        "market_root": str(args.market_root),
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    diagnostics["provenance"] = artifact_provenance(configuration, args.features, project_root=REPO)
    write_experiment_artifact(
        output,
        "diagnostics.json",
        diagnostics,
        command="era-weighting-opener-read",
        metrics=diagnostics,
        notes=(
            "MOD-14 opener-grade information read: half_life_8 vs. baseline at the "
            "production opener protocol -- a below-power read, not a promotion decision."
        ),
    )

    print("\n=== PRIMARY: opener grade, probability rule, week-blocked, paired ===")
    primary = paired.loc[
        paired["grade"].eq("open") & paired["rule"].eq("probability") & paired["block"].eq("week")
    ]
    print(
        primary.loc[
            :,
            ["metric", "estimate", "lower", "upper", "probability_positive", "paired_games"],
        ].to_string(index=False)
    )
    print("\n=== secondary: opener grade, sign rule, week-blocked, paired (accuracy only) ===")
    secondary = paired.loc[
        paired["grade"].eq("open")
        & paired["rule"].eq("sign")
        & paired["block"].eq("week")
        & paired["metric"].eq("accuracy_improvement")
    ]
    print(
        secondary.loc[
            :,
            ["metric", "estimate", "lower", "upper", "probability_positive", "paired_games"],
        ].to_string(index=False)
    )
    print("\n=== context: close grade, probability rule, week-blocked, paired ===")
    close_context = paired.loc[
        paired["grade"].eq("close") & paired["rule"].eq("probability") & paired["block"].eq("week")
    ]
    print(
        close_context.loc[
            :,
            ["metric", "estimate", "lower", "upper", "probability_positive", "paired_games"],
        ].to_string(index=False)
    )
    print(f"\nartifacts: {output}")


if __name__ == "__main__":
    main()
