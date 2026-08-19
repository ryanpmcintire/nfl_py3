"""Dry run ONLY: report what v1 (incumbent) and v2 (measured, POL-09) would
each nominate as this week's Best Pick, and the probability/dispersion table
behind v2's choice. Does NOT call ``publish-predictions``, does NOT write to
any ledger, does NOT touch ``registry/*`` or ``docs/*``.

Reads the CURRENTLY ACTIVE model's already-synchronized weekly forecast
artifact (whatever ``artifacts/active_ats_model.json`` points at) exactly the
way ``nfl_ats.publishing`` and ``nfl_ats.best_pick_nomination`` would, and
prints a plain report. See ``nfl_ats.best_pick_nomination`` for the rule's
full definition and evidence base.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/best_pick_nomination_dry_run.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.best_pick import best_pick_tie_note, select_best_pick
from nfl_ats.best_pick_nomination import (
    NOMINATION_RIDGE_ALPHA,
    NOMINATION_V2_ENABLED,
    nominate_v2,
    nomination_v2_disclosure_note,
    nomination_v2_tie_note,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.prospective_scoring import artifact_model_config

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=REPO / "artifacts")
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    args = parser.parse_args()

    active = load_active_ats_model(args.artifacts_root)
    if active is None:
        raise SystemExit("No synchronized active ATS model found.")
    forecast = active_artifact_path(args.artifacts_root, active, "weekly_forecast")
    if forecast is None:
        raise SystemExit("Active ATS model has no linked weekly forecast.")
    metadata = json.loads((forecast / "metadata.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(forecast / "recommendations.csv")

    print(
        f"Active model:    {active['model_id']}  ({active['method']}, {active['feature_profile']})"
    )
    print(f"Weekly forecast: {forecast}")
    print(f"Season/Week:     {metadata['season']} / {metadata['week']}")
    print(f"Games on card:   {len(predictions)}")
    print(f"NOMINATION_V2_ENABLED = {NOMINATION_V2_ENABLED}")
    print()

    # --- v1: incumbent sweep_robustness -------------------------------------
    sweep_path = forecast / "line_sweep.parquet"
    if sweep_path.is_file():
        sweep = pd.read_parquet(sweep_path)
        if "method" in sweep.columns and "method" in predictions.columns:
            sweep = sweep.loc[sweep["method"].isin(set(predictions["method"].astype(str)))]
        v1_id = select_best_pick(predictions, sweep)
        v1_tie = best_pick_tie_note(predictions, sweep) if v1_id is not None else ""
    else:
        v1_id, v1_tie = None, ""
    print("=== v1 (incumbent, sweep_robustness) ===")
    print(f"Nominee: {v1_id}")
    if v1_tie:
        print(f"Disclosure: {v1_tie}")
    print()

    # --- v2: measured winner -------------------------------------------------
    observed_config = artifact_model_config(metadata)
    feature_table = observed_config.get("feature_table")
    print("=== v2 (measured, POL-09) ===")
    if not feature_table:
        print("UNAVAILABLE: forecast metadata carries no feature_table path.")
        return
    features = pd.read_parquet(feature_table)
    try:
        result = nominate_v2(
            predictions,
            features,
            market_root=args.data_root / "market" / "raw",
            season=int(metadata["season"]),
            week=int(metadata["week"]),
            regressor=str(metadata.get("regressor", "ridge")),
            feature_profile=str(metadata.get("feature_profile")),
            min_train_games=int(metadata.get("min_train_games") or DEFAULT_MIN_TRAIN_GAMES),
            ridge_alpha=NOMINATION_RIDGE_ALPHA,
        )
    except (ValueError, DataContractError) as error:
        print(f"UNAVAILABLE this run: {error}")
        return

    if result is None:
        print("UNAVAILABLE: not a regular-season card (v2's gate matches v1's).")
        return

    print(f"Nominee: {result.game_id}")
    print(f"Tie-break used: {result.tie_break}  (n_tied_at_max={result.n_tied_at_max})")
    dispersion = result.dispersion
    print(
        f"Dispersion pool: fallback={dispersion.fallback} "
        f"reason={dispersion.fallback_reason} "
        f"n_pool_pass={dispersion.n_pool_pass}/{dispersion.n_games} "
        f"n_missing_spread_std={dispersion.n_missing}"
    )
    tie_note = nomination_v2_tie_note(result)
    if tie_note:
        print(f"Tie disclosure: {tie_note}")
    print(f"Card disclosure: {nomination_v2_disclosure_note(result)}")
    print()
    print("Both rules agree on:", v1_id == result.game_id if v1_id else "n/a (v1 unavailable)")
    print()

    table = result.probability_table.merge(
        predictions[["game_id", "home_team", "away_team", "spread_line"]].assign(
            game_id=lambda frame: frame["game_id"].astype(str)
        ),
        on="game_id",
        how="left",
    )
    table = table.sort_values("candidate_dist", ascending=False)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(
            table[
                [
                    "game_id",
                    "away_team",
                    "home_team",
                    "spread_line",
                    "candidate_home_cover_probability",
                    "candidate_dist",
                    "spread_std",
                    "pool_pass",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
