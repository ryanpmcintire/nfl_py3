"""Missingness audit for the production margin-model feature set (WP15 / MOD-13).

Diagnostic-only, read-only script. It never fits anything to disk and never
writes to the experiment registry -- it exists to answer one prediction-safety
question ahead of the 2026 Week 1 lock: does the production
``SimpleImputer(strategy="median", add_indicator=True)`` step (see
``src/nfl_ats/margin.py::make_margin_estimator``) hand the ridge regressor a
missingness PATTERN on the live card that never occurred during training? If
it does, the imputer + ridge combination is extrapolating rather than
interpolating on that row, because ``add_indicator=True`` appends one binary
"this column was missing" feature per training-missing column, and those
indicators behave as SOURCE-ERA markers for any feature whose underlying data
source only exists from some later season.

Three things this script computes, all read-only over
``data/processed/game_features_weak_stack.parquet`` and
``src/nfl_ats/constants.py``:

1. Per production feature column, per season 2009-2026: the fraction of
   regular-season rows with a missing value. Columns are classified as
   ``source_era`` (>=95% missing in some season, <=5% missing in another --
   the signature of a data source that switched on), ``sporadic`` (some
   missingness that is not a season step function), ``complete`` (never
   missing in 2009-2025), or ``always_missing`` (missing in every 2009-2025
   row -- dead weight in training).
2. For the live 2026 Week 1 card: for every column and every one of the 16
   games, is the value missing, and did THAT missing/present state occur in
   at least 1% of 2025 regular-season training rows for that column? A state
   that occurred in under 1% of training is a row the imputer's learned
   indicator coefficient never saw enough of to have been fit safely on.
3. The standardized ridge coefficient magnitude on every
   ``missingindicator_*`` column versus the median magnitude on the real
   (non-indicator) columns, from the exact production fit recipe
   (``fit_margin_model(training, target="market_residual", model_name="ridge",
   feature_profile="weak_stack", ridge_alpha=10.0)`` with the same walk-forward
   cutoff ``nfl_ats.outcomes._target_and_models_for_week`` uses -- see that
   module, read 2026-09-01). The active model artifact
   (``artifacts/active_ats_model.json``, model_id ``d1f07d773475dc58``) has no
   persisted pipeline anywhere on disk -- neither its weekly-forecast artifact
   directory (``artifacts/margin_predictions/2026-week-01-20260824T120725Z/``,
   listed 2026-09-01: predictions/metadata/csv only, no ``model.joblib``) nor
   its `margin-predict` CLI code path (``src/nfl_ats/cli.py::_cmd_margin_predict``
   calls ``score_outcome_week`` -> ``fit_margin_model``, never
   ``joblib.dump``) -- so this script REFITS the identical recipe in memory
   each run rather than loading a saved pipeline. Nothing is written to disk.

Usage:

    ./.tools/uv.exe run --no-sync python scripts/missingness_audit.py
    ./.tools/uv.exe run --no-sync python scripts/missingness_audit.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.margin import (  # noqa: E402
    MarginFeatureProfile,
    MarginTarget,
    fit_margin_model,
    margin_feature_columns,
)
from nfl_ats.modeling import regular_season_rows  # noqa: E402

DEFAULT_PARQUET = REPO_ROOT / "data" / "processed" / "game_features_weak_stack.parquet"

# Step-function thresholds for the "source-era" classification: a column
# counts as source-era if some season is at-or-above STEP_HIGH missing and
# some other season is at-or-below STEP_LOW missing.
STEP_HIGH = 0.95
STEP_LOW = 0.05

# A missing/present state occurring in under this fraction of the reference
# season's training rows is flagged as a 2026 Week 1 extrapolation risk.
RARE_THRESHOLD = 0.01

INDICATOR_PREFIX = "missingindicator_"


def load_frame(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Feature table not found: {parquet_path}. This is a local-only "
            "generated artifact -- absent in a fresh clone. Run "
            "`nfl-ats build-features` (weak_stack profile) first."
        )
    frame = pd.read_parquet(parquet_path)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    return frame


def per_season_missingness(
    frame: pd.DataFrame, columns: Sequence[str], seasons: Sequence[int]
) -> pd.DataFrame:
    """Regular-season missing fraction for every (season, column) pair."""

    reg = regular_season_rows(frame)
    rows: list[dict[str, Any]] = []
    for season in seasons:
        subset = reg.loc[reg["season"].eq(season)]
        n = len(subset)
        for column in columns:
            missing = int(subset[column].isna().sum()) if n else 0
            rows.append(
                {
                    "season": season,
                    "column": column,
                    "rows": n,
                    "missing": missing,
                    "missing_frac": (missing / n) if n else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def classify_columns(
    missingness: pd.DataFrame, classification_seasons: Sequence[int]
) -> pd.DataFrame:
    """One row per column: source_era / sporadic / complete / always_missing.

    Classification is restricted to ``classification_seasons`` (the fully
    realized seasons) so a partially-played season never masquerades as a
    step transition.
    """

    subset = missingness.loc[missingness["season"].isin(classification_seasons)]
    out: list[dict[str, Any]] = []
    for column, group in subset.groupby("column", sort=False):
        group = group.sort_values("season")
        fracs = group["missing_frac"].to_numpy(dtype=float)
        max_frac = float(np.nanmax(fracs))
        min_frac = float(np.nanmin(fracs))
        if max_frac >= STEP_HIGH and min_frac <= STEP_LOW:
            category = "source_era"
        elif min_frac >= STEP_HIGH:
            category = "always_missing"
        elif max_frac <= 0.0:
            category = "complete"
        else:
            category = "sporadic"
        source_begin_season = None
        last_high_missing_season = None
        if category == "source_era":
            low = group.loc[group["missing_frac"] <= STEP_LOW, "season"]
            high = group.loc[group["missing_frac"] >= STEP_HIGH, "season"]
            if not low.empty:
                source_begin_season = int(low.min())
            if not high.empty:
                last_high_missing_season = int(high.max())
        out.append(
            {
                "column": column,
                "category": category,
                "min_missing_frac": min_frac,
                "max_missing_frac": max_frac,
                "mean_missing_frac": float(np.nanmean(fracs)),
                "source_begin_season": source_begin_season,
                "last_high_missing_season": last_high_missing_season,
            }
        )
    result = pd.DataFrame(out)
    category_order = {"source_era": 0, "always_missing": 1, "sporadic": 2, "complete": 3}
    result["_order"] = result["category"].map(category_order)
    return result.sort_values(["_order", "column"]).drop(columns="_order").reset_index(drop=True)


def week1_extrapolation_risk(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    season: int,
    week: int,
    reference_season: int,
) -> pd.DataFrame:
    """Per (column, game) row: is this column's missing/present state on the
    live card rare (<1%) relative to the reference season's training rows?
    """

    reg = regular_season_rows(frame)
    target = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)]
    if target.empty:
        raise ValueError(f"No games found for season={season} week={week}")
    reference = reg.loc[reg["season"].eq(reference_season)]
    if reference.empty:
        raise ValueError(f"No regular-season rows found for reference season={reference_season}")
    n_reference = len(reference)

    rows: list[dict[str, Any]] = []
    for column in columns:
        reference_missing_frac = float(reference[column].isna().mean())
        reference_present_frac = 1.0 - reference_missing_frac
        for _, game in target.iterrows():
            is_missing = bool(pd.isna(game[column]))
            state_frac = reference_missing_frac if is_missing else reference_present_frac
            rows.append(
                {
                    "column": column,
                    "game_id": game["game_id"],
                    "value_missing": is_missing,
                    "reference_season": reference_season,
                    "reference_rows": n_reference,
                    "reference_state_frac": state_frac,
                    "rare_relative_to_reference": bool(state_frac < RARE_THRESHOLD),
                }
            )
    return pd.DataFrame(rows)


def fit_production_recipe(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    profile: MarginFeatureProfile,
    target: MarginTarget,
    model_name: str,
    ridge_alpha: float,
    season: int,
    week: int,
) -> tuple[Any, pd.DataFrame]:
    """Refit the exact production recipe in memory; nothing is written to disk.

    Mirrors ``nfl_ats.outcomes._target_and_models_for_week``'s leak-safe
    cutoff (training strictly precedes the target week's earliest kickoff)
    and ``nfl_ats.outcomes._fit_week_models``'s call into
    ``fit_margin_model`` for the ``market_residual`` method -- read both,
    2026-09-01, to confirm this reproduces the production fit bit-for-bit
    (same training rows, same columns, same ``random_state=42`` default).
    """

    target_rows = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)]
    if target_rows.empty:
        raise ValueError(f"No games found for season={season} week={week}")
    cutoff = target_rows["gameday"].min()
    training = regular_season_rows(frame)
    training = training.loc[training["gameday"].lt(cutoff) & training["result"].notna()].copy()
    model = fit_margin_model(
        training,
        target=target,
        model_name=model_name,
        feature_profile=profile,
        ridge_alpha=ridge_alpha,
    )
    _ = columns  # feature_columns come from margin_feature_columns(target, profile) inside fit
    return model, training


def indicator_coefficient_summary(model: Any, columns: Sequence[str]) -> dict[str, Any]:
    pipeline = model.estimator
    if pipeline is None:
        raise RuntimeError("Fitted margin model has no estimator to inspect")
    imputer = pipeline.named_steps["imputer"]
    regressor = pipeline.named_steps["regressor"]
    names = np.asarray(imputer.get_feature_names_out(list(columns)), dtype=object)
    coefs = np.asarray(regressor.coef_, dtype=float).ravel()
    if len(names) != len(coefs):
        raise RuntimeError(
            f"Feature-name/coefficient length mismatch: {len(names)} names vs {len(coefs)} coefs"
        )
    abs_coefs = np.abs(coefs)
    is_indicator = np.array([str(n).startswith(INDICATOR_PREFIX) for n in names])
    real_abs = abs_coefs[~is_indicator]
    indicator_abs = abs_coefs[is_indicator]
    indicator_names = names[is_indicator]
    ranked = sorted(zip(indicator_names, indicator_abs, strict=True), key=lambda t: -t[1])
    median_real = float(np.median(real_abs)) if len(real_abs) else float("nan")
    median_indicator = float(np.median(indicator_abs)) if len(indicator_abs) else float("nan")
    return {
        "n_real_features": int((~is_indicator).sum()),
        "n_indicator_features": int(is_indicator.sum()),
        "median_abs_coef_real": median_real,
        "median_abs_coef_indicator": median_indicator,
        "max_abs_coef_real": float(np.max(real_abs)) if len(real_abs) else float("nan"),
        "max_abs_coef_indicator": float(np.max(indicator_abs))
        if len(indicator_abs)
        else float("nan"),
        "ratio_median_indicator_to_real": (
            median_indicator / median_real
            if len(real_abs) and len(indicator_abs) and median_real not in (0.0,)
            else float("nan")
        ),
        "top_indicators": [
            {"column": str(name)[len(INDICATOR_PREFIX) :], "abs_standardized_coef": float(value)}
            for name, value in ranked[:15]
        ],
        "training_rows": model.training_rows,
        "training_max_gameday": model.training_max_gameday,
        "model_name": model.model_name,
        "ridge_alpha": model.ridge_alpha,
        "target": model.target,
    }


def _format_trajectory(group: pd.DataFrame, seasons: Sequence[int]) -> str:
    by_season = dict(zip(group["season"], group["missing_frac"], strict=True))
    return " ".join(f"{season}:{by_season.get(season, float('nan')):.2f}" for season in seasons)


def render_markdown(
    *,
    command: str,
    columns: Sequence[str],
    profile: str,
    target: str,
    seasons: Sequence[int],
    classification_seasons: Sequence[int],
    missingness: pd.DataFrame,
    classification: pd.DataFrame,
    week1_risk: pd.DataFrame,
    coefficient_summary: dict[str, Any],
    season: int,
    week: int,
    reference_season: int,
) -> str:
    lines: list[str] = []
    lines.append("# Missingness audit output")
    lines.append("")
    lines.append("Command:")
    lines.append("")
    lines.append(f"    {command}")
    lines.append("")
    lines.append(
        f"Production profile `{profile}` / target `{target}`: {len(columns)} feature columns "
        f"(measured: `nfl_ats.margin.margin_feature_columns('{target}', '{profile}')`)."
    )
    lines.append("")

    counts = classification["category"].value_counts()
    lines.append("## 1. Season-level classification (2009-2025 fully realized seasons)")
    lines.append("")
    for category in ("source_era", "always_missing", "sporadic", "complete"):
        lines.append(f"- {category}: {int(counts.get(category, 0))} columns")
    lines.append("")

    source_era = classification.loc[classification["category"] == "source_era"]
    lines.append(f"### source_era columns ({len(source_era)})")
    lines.append("")
    if source_era.empty:
        lines.append("(none)")
    else:
        lines.append(
            "| column | source_begin_season | last_high_missing_season | min_frac | max_frac |"
        )
        lines.append("|---|---|---|---|---|")
        for _, row in source_era.iterrows():
            lines.append(
                f"| {row['column']} | {row['source_begin_season']} | "
                f"{row['last_high_missing_season']} | {row['min_missing_frac']:.3f} | "
                f"{row['max_missing_frac']:.3f} |"
            )
        lines.append("")
        lines.append("Per-season trajectory (missing fraction, REG rows only):")
        lines.append("")
        for _, row in source_era.iterrows():
            column = row["column"]
            group = missingness.loc[missingness["column"] == column]
            lines.append(f"- `{column}`: {_format_trajectory(group, seasons)}")
    lines.append("")

    always_missing = classification.loc[classification["category"] == "always_missing"]
    if not always_missing.empty:
        lines.append(
            f"### always_missing columns ({len(always_missing)}) -- dead weight in training"
        )
        lines.append("")
        for _, row in always_missing.iterrows():
            lines.append(f"- `{row['column']}`")
        lines.append("")

    sporadic = classification.loc[classification["category"] == "sporadic"]
    lines.append(f"### sporadic columns ({len(sporadic)})")
    lines.append("")
    if sporadic.empty:
        lines.append("(none)")
    else:
        lines.append("| column | min_frac | max_frac | mean_frac |")
        lines.append("|---|---|---|---|")
        for _, row in sporadic.iterrows():
            lines.append(
                f"| {row['column']} | {row['min_missing_frac']:.3f} | "
                f"{row['max_missing_frac']:.3f} | {row['mean_missing_frac']:.3f} |"
            )
    lines.append("")

    lines.append(
        f"## 2. {season} Week {week} extrapolation-risk list (vs {reference_season} regular season)"
    )
    lines.append("")
    risky = week1_risk.loc[week1_risk["rare_relative_to_reference"]]
    if risky.empty:
        lines.append(
            f"None. Every {season} Week {week} column's missing/present state occurred in "
            f">= {RARE_THRESHOLD:.0%} of {reference_season} regular-season training rows."
        )
    else:
        by_column = (
            risky.groupby("column")
            .agg(
                games_affected=("game_id", "count"),
                reference_state_frac=("reference_state_frac", "first"),
                value_missing=("value_missing", "first"),
            )
            .reset_index()
            .sort_values("reference_state_frac")
        )
        lines.append(
            f"{len(by_column)} column(s) with a state occurring in < {RARE_THRESHOLD:.0%} of "
            f"{reference_season} training rows:"
        )
        lines.append("")
        lines.append(
            "| column | state | games affected (of 16) | reference_state_frac | example game_ids |"
        )
        lines.append("|---|---|---|---|---|")
        for _, row in by_column.iterrows():
            column = row["column"]
            state = "missing" if row["value_missing"] else "present"
            examples = risky.loc[risky["column"] == column, "game_id"].head(3).tolist()
            lines.append(
                f"| {column} | {state} | {int(row['games_affected'])} | "
                f"{row['reference_state_frac']:.4f} | {', '.join(examples)} |"
            )
    lines.append("")

    lines.append("## 3. Standardized ridge coefficient magnitude: indicators vs real features")
    lines.append("")
    lines.append(
        f"Refit in memory (no artifact written): `{coefficient_summary['model_name']}` "
        f"target=`{coefficient_summary['target']}` ridge_alpha="
        f"{coefficient_summary['ridge_alpha']}, {coefficient_summary['training_rows']} "
        f"training rows through {coefficient_summary['training_max_gameday']}."
    )
    lines.append("")
    lines.append(f"- real features: {coefficient_summary['n_real_features']}")
    lines.append(f"- indicator features: {coefficient_summary['n_indicator_features']}")
    lines.append(f"- median |coef| real: {coefficient_summary['median_abs_coef_real']:.5f}")
    lines.append(
        f"- median |coef| indicator: {coefficient_summary['median_abs_coef_indicator']:.5f}"
    )
    lines.append(f"- max |coef| real: {coefficient_summary['max_abs_coef_real']:.5f}")
    lines.append(f"- max |coef| indicator: {coefficient_summary['max_abs_coef_indicator']:.5f}")
    lines.append(
        f"- ratio (median indicator / median real): "
        f"{coefficient_summary['ratio_median_indicator_to_real']:.3f}"
    )
    lines.append("")
    lines.append("Top indicator columns by |standardized coefficient|:")
    lines.append("")
    lines.append("| indicator on column | abs standardized coef |")
    lines.append("|---|---|")
    for entry in coefficient_summary["top_indicators"]:
        lines.append(f"| {entry['column']} | {entry['abs_standardized_coef']:.5f} |")
    lines.append("")

    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--profile", default="weak_stack")
    parser.add_argument(
        "--target", default="market_residual", choices=("margin", "market_residual")
    )
    parser.add_argument("--model-name", default="ridge")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, default=1)
    parser.add_argument("--reference-season", type=int, default=None, help="Default: season - 1")
    parser.add_argument("--start-season", type=int, default=2009)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument(
        "--classification-end-season",
        type=int,
        default=2025,
        help="Last season used for the source_era/sporadic step-function classification.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    reference_season = (
        args.reference_season if args.reference_season is not None else args.season - 1
    )

    frame = load_frame(args.parquet)
    columns = margin_feature_columns(args.target, args.profile)  # type: ignore[arg-type]

    seasons = list(range(args.start_season, args.end_season + 1))
    classification_seasons = [s for s in seasons if s <= args.classification_end_season]

    missingness = per_season_missingness(frame, columns, seasons)
    classification = classify_columns(missingness, classification_seasons)
    week1_risk = week1_extrapolation_risk(
        frame,
        columns,
        season=args.season,
        week=args.week,
        reference_season=reference_season,
    )
    model, _training = fit_production_recipe(
        frame,
        columns,
        profile=args.profile,  # type: ignore[arg-type]
        target=args.target,  # type: ignore[arg-type]
        model_name=args.model_name,
        ridge_alpha=args.ridge_alpha,
        season=args.season,
        week=args.week,
    )
    coefficient_summary = indicator_coefficient_summary(model, columns)

    command = "./.tools/uv.exe run --no-sync python " + " ".join(
        ["scripts/missingness_audit.py", *(argv if argv is not None else sys.argv[1:])]
    )

    if args.as_json:
        payload = {
            "command": command,
            "profile": args.profile,
            "target": args.target,
            "columns": list(columns),
            "seasons": seasons,
            "classification_seasons": classification_seasons,
            "season": args.season,
            "week": args.week,
            "reference_season": reference_season,
            "rare_threshold": RARE_THRESHOLD,
            "step_high": STEP_HIGH,
            "step_low": STEP_LOW,
            "missingness": missingness.to_dict(orient="records"),
            "classification": classification.to_dict(orient="records"),
            "week1_risk": week1_risk.to_dict(orient="records"),
            "coefficient_summary": coefficient_summary,
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(
        render_markdown(
            command=command,
            columns=columns,
            profile=args.profile,
            target=args.target,
            seasons=seasons,
            classification_seasons=classification_seasons,
            missingness=missingness,
            classification=classification,
            week1_risk=week1_risk,
            coefficient_summary=coefficient_summary,
            season=args.season,
            week=args.week,
            reference_season=reference_season,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
