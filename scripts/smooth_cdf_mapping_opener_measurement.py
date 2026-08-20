"""MOD-08 opener-grade decision measurement: Gaussian CDF mapping vs. the
production ECDF, at the 1,537-paired-game opener protocol.

Predeclaration: ``docs/smooth_cdf_mapping.md``'s "Opener-grade decision
measurement" section (written and committed to the doc BEFORE this script
ran). This is a SECOND, independently-labeled look at the same MOD-08
candidate already measured close-graded in that document's earlier
"Historical measurement" section (accuracy ``probability_positive`` 0.8666,
recorded ``mod08_smooth_cdf_mapping``) -- multiplicity disclosed, same
convention as ``docs/opener_evaluation.md``'s addendum.

Reuses ``docs/opener_evaluation.md``'s exact 1,537-paired-game archive and
weekly-refit protocol (``nfl_ats.clv.build_pairing_table``/
``close_reference_table``, ``tue_open`` + close, 2020-2025 regular season,
production recipe ``weak_stack``/``ridge``/alpha 10.0/``market_residual``,
``min_train_games=500``) but scores BOTH the ``ecdf`` and ``gaussian``
probability reads off the SAME fitted model and out-of-time residual sample
each week, at BOTH the opener and close lines, rather than only the
production ``ecdf``. Per ``docs/opener_evaluation.md``'s own admissibility
argument: nothing is selected or tuned here -- both mappings were already
fully specified by the close-graded measurement -- so this is a
re-measurement of an already-fixed pair of pick-stream generators at a
different settlement line, not a new selection dimension. No
rotation-registry window is spent or implied (rule 8).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.calibration import smoothed_home_cover_probability
from nfl_ats.clv import CLOSE_LABEL_PRIORITY, build_pairing_table, close_reference_table
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.margin import fit_margin_model, margin_feature_columns
from nfl_ats.modeling import regular_season_rows
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"
DEFAULT_MARKET_ROOT = REPO / "data/market/raw"

# Production recipe, artifacts/active_ats_model.json (2026-08-19) -- identical
# to scripts/smooth_cdf_mapping_measurement.py and docs/opener_evaluation.md's
# "weak_stack profile, the ACTIVE model" run.
FEATURE_PROFILE = "weak_stack"
REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0
TARGET = "market_residual"

BASELINE_ARM = "ecdf"
CANDIDATE_ARM = "gaussian"


def run_opener_walk_forward(
    market_root: Path,
    features: pd.DataFrame,
    *,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> pd.DataFrame:
    """One row per (game, feature_set in {ecdf, gaussian}) at the opener protocol.

    Mirrors ``nfl_ats.clv.opener_pick_evaluation``'s weekly-refit loop exactly
    (one market-residual model per (season, week), trained on completed games
    strictly before that week's first kickoff) but keeps the fitted model
    around long enough to read its residual sample through both probability
    methods, at both the opener and the close line, instead of only the
    production ECDF.
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
            f"Opener mapping measurement is missing columns: {', '.join(missing)}"
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

    rows: list[pd.DataFrame] = []
    for (season, week), group in paired.groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"]))]
        if week_rows.empty:
            continue
        cutoff = week_rows["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        model = fit_margin_model(
            training,
            target=TARGET,  # type: ignore[arg-type]
            model_name=REGRESSOR,
            feature_profile=FEATURE_PROFILE,  # type: ignore[arg-type]
            ridge_alpha=RIDGE_ALPHA,
        )
        scoring = week_rows.merge(
            group[["game_id", "tue_open_home_spread", "close_home_spread"]],
            on="game_id",
            how="inner",
        ).copy()

        at_open = scoring.copy()
        at_open["spread_line"] = at_open["tue_open_home_spread"]
        predicted_open = model.predict(at_open)
        centers_open = predicted_open["predicted_margin"].to_numpy(dtype=float)
        residual_open = predicted_open["predicted_market_residual"].to_numpy(dtype=float)
        line_open = at_open["spread_line"].to_numpy(dtype=float)

        at_close = scoring.copy()
        at_close["spread_line"] = at_close["close_home_spread"]
        predicted_close = model.predict(at_close)
        centers_close = predicted_close["predicted_margin"].to_numpy(dtype=float)
        residual_close = predicted_close["predicted_market_residual"].to_numpy(dtype=float)
        line_close = at_close["spread_line"].to_numpy(dtype=float)

        base = scoring[["game_id", "result", "tue_open_home_spread", "close_home_spread"]].copy()
        base["season"] = int(str(season))
        base["week"] = int(str(week))
        margin_open = base["result"] - base["tue_open_home_spread"]
        margin_close = base["result"] - base["close_home_spread"]
        base["home_cover_open"] = np.select(
            [margin_open > 0, margin_open < 0], [1.0, 0.0], default=np.nan
        )
        base["home_cover_close"] = np.select(
            [margin_close > 0, margin_close < 0], [1.0, 0.0], default=np.nan
        )
        # Sign-rule pick is a pure function of the fitted centre vs. zero --
        # it never calls the probability-mapping function, so it is identical
        # for the ecdf and gaussian arms by construction at a fixed line.
        base["sign_pick_home_open"] = residual_open > 0.0
        base["sign_pick_home_close"] = residual_close > 0.0
        base["distribution_rows"] = model.distribution_rows

        for method in (BASELINE_ARM, CANDIDATE_ARM):
            batch = base.copy()
            batch["feature_set"] = method
            batch["home_cover_probability_open"] = smoothed_home_cover_probability(
                model.residuals, centers_open, line_open, method=method
            )
            batch["home_cover_probability_close"] = smoothed_home_cover_probability(
                model.residuals, centers_close, line_close, method=method
            )
            rows.append(batch)

    if not rows:
        raise ValueError("No paired week had enough prior training games")
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["game_id", "feature_set"])
        .reset_index(drop=True)
    )


def _probability_rule_paired(
    predictions: pd.DataFrame, *, grade: str, samples: int, seed: int
) -> pd.DataFrame:
    frame = predictions.rename(
        columns={
            f"home_cover_{grade}": "home_cover",
            f"home_cover_probability_{grade}": "home_cover_probability",
        }
    )[["game_id", "season", "week", "feature_set", "home_cover", "home_cover_probability"]]
    paired = pd.concat(
        [
            paired_feature_comparisons(
                frame, baseline_feature_set=BASELINE_ARM, samples=samples, block="week", seed=seed
            ),
            paired_feature_comparisons(
                frame,
                baseline_feature_set=BASELINE_ARM,
                samples=samples,
                block="season",
                seed=seed,
            ),
        ],
        ignore_index=True,
    )
    paired.insert(0, "grade", grade)
    return paired


def _sign_rule_report(predictions: pd.DataFrame) -> dict[str, Any]:
    """Diagnostic: confirm the sign rule is mapping-invariant, report its accuracy."""

    report: dict[str, Any] = {}
    for grade in ("open", "close"):
        home_cover_col = f"home_cover_{grade}"
        sign_pick_col = f"sign_pick_home_{grade}"
        ecdf_rows = predictions.loc[predictions["feature_set"].eq(BASELINE_ARM)].set_index(
            "game_id"
        )
        gaussian_rows = predictions.loc[predictions["feature_set"].eq(CANDIDATE_ARM)].set_index(
            "game_id"
        )
        common = ecdf_rows.index.intersection(gaussian_rows.index)
        flips = int(
            (
                ecdf_rows.loc[common, sign_pick_col].to_numpy()
                != gaussian_rows.loc[common, sign_pick_col].to_numpy()
            ).sum()
        )
        cover = pd.to_numeric(ecdf_rows.loc[common, home_cover_col], errors="coerce")
        pick = ecdf_rows.loc[common, sign_pick_col].astype(bool)
        scored = cover.notna()
        correct = np.where(pick[scored], cover[scored] > 0.5, cover[scored] < 0.5)
        report[grade] = {
            "games": len(common),
            "flips_between_ecdf_and_gaussian": flips,
            "accuracy": float(np.mean(correct)) if len(correct) else float("nan"),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260819)
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(args.features)
    predictions = run_opener_walk_forward(
        args.market_root, features, min_train_games=args.min_train_games
    )
    predictions.to_parquet(output / "predictions.parquet", index=False)

    open_paired = _probability_rule_paired(
        predictions, grade="open", samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )
    close_paired = _probability_rule_paired(
        predictions, grade="close", samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )
    paired = pd.concat([open_paired, close_paired], ignore_index=True)
    paired.to_csv(output / "paired_comparisons.csv", index=False)

    sign_rule = _sign_rule_report(predictions)

    games = int(predictions["game_id"].nunique())
    diagnostics: dict[str, Any] = {
        "predeclaration": "docs/smooth_cdf_mapping.md#opener-grade-decision-measurement",
        "second_look_of": "mod08_smooth_cdf_mapping (close-graded, docs/smooth_cdf_mapping.md)",
        "rotation_registry_touched": False,
        "games": games,
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
        "primary_metric": "accuracy_improvement, grade=open, block=week (production pick rule)",
        "sign_rule_diagnostic": sign_rule,
        "distribution_rows_range": [
            int(predictions["distribution_rows"].min()),
            int(predictions["distribution_rows"].max()),
        ],
    }
    configuration = {
        "command": "smooth-cdf-mapping-opener-measurement",
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
        command="smooth-cdf-mapping-opener-measurement",
        metrics=diagnostics,
        notes=(
            "MOD-08 opener-grade decision measurement: Gaussian CDF mapping vs. the "
            "production ECDF at the 1,537-paired-game opener protocol -- second, "
            "disclosed look at the same candidate already measured close-graded."
        ),
    )

    print(
        "\n=== OPENER grade, probability rule, week-blocked, paired "
        "(positive = gaussian better) ==="
    )
    open_week = open_paired.loc[open_paired["block"].eq("week")]
    print(
        open_week.loc[
            :, ["metric", "estimate", "lower", "upper", "probability_positive", "paired_games"]
        ].to_string(index=False)
    )
    print("\n=== CLOSE grade, probability rule, week-blocked, paired (context only) ===")
    close_week = close_paired.loc[close_paired["block"].eq("week")]
    print(
        close_week.loc[
            :, ["metric", "estimate", "lower", "upper", "probability_positive", "paired_games"]
        ].to_string(index=False)
    )
    print(f"\nSign-rule diagnostic (mapping-invariance check): {json.dumps(sign_rule, indent=2)}")
    print(f"\nartifacts: {output}")


if __name__ == "__main__":
    main()
