from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from line_move_regrade_legacy import loso  # noqa: E402
from line_move_yardstick_paired_eval import cell_stats  # noqa: E402
from roof_state_screen import build_prediction_table as roof_state_prediction_table  # noqa: E402

EXTENDED_POPULATION = (
    REPO / "artifacts" / "extended_fit_population" / "20260923T205910Z" / "population.parquet"
)
SBR_ODDS = REPO / "data" / "processed" / "sbr_odds.parquet"
LEGACY_RESULTS = (
    REPO / "artifacts" / "line_move_regrade_legacy" / "20260923T220145Z" / "results.json"
)

BASE_FEATURES = ("model_logit", "composition_flag_sum")
REPLICATION_SEASON_START = 2011
REPLICATION_SEASON_END = 2019
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923
ORIGINAL_LOOKS = 8

FORECAST_ARCHIVE_NOTE = (
    "Tuesday-noon-cutoff forecast archives do NOT exist pre-2020: "
    "docs/forecast_archive_build.md's cutoff-mode table reports the tuesday_noon "
    "cutoff's MOS model (GFS MOS Extended) archive start measured at 2020-07-12, "
    "confirmed absent at 2015-09-01 and 2009-09-01. The pool_decision cutoff "
    "archive (data/raw/forecast_archive/pool_decision_2009_2025/forecasts.parquet, "
    "cutoff = min(kickoff, Sunday 16:00 America/New_York)) does exist for "
    "2011-2019 (fetch_status='ok' for the large majority of games each season) "
    "and is what scripts/roof_state_screen.py's build_prediction_table() already "
    "reads for every season, including the 2020-2025 rows behind Unit 1's "
    "roof_state_predicted_open result. This replication reuses that same "
    "function unchanged for 2011-2019, per this lane's instruction to fall back "
    "to the registry family's own historical construction when a Tuesday-noon "
    "archive is unavailable. Caveat carried forward, not resolved here: the "
    "pool_decision cutoff is near-kickoff, not Tuesday-noon, so the "
    "'predicted_open' feature is not demonstrated Tuesday-actionable in either "
    "the original family or this replication; both share the same construction "
    "and therefore the same caveat."
)


def load_population() -> pd.DataFrame:
    population = pd.read_parquet(EXTENDED_POPULATION)
    population["game_id"] = population["game_id"].astype(str)
    population = population.loc[
        population["season"].between(REPLICATION_SEASON_START, REPLICATION_SEASON_END)
    ].reset_index(drop=True)
    return population


def load_sbr_open_move() -> pd.DataFrame:
    sbr = pd.read_parquet(
        SBR_ODDS,
        columns=["game_id", "season", "open_home_spread", "close_home_spread"],
    )
    sbr = sbr.loc[sbr["game_id"].notna()].copy()
    sbr["game_id"] = sbr["game_id"].astype(str)
    sbr = sbr.dropna(subset=["open_home_spread", "close_home_spread"])
    sbr = sbr.drop_duplicates(subset="game_id")
    sbr["sbr_open_move"] = sbr["close_home_spread"] - sbr["open_home_spread"]
    return sbr[["game_id", "open_home_spread", "close_home_spread", "sbr_open_move"]]


def load_roof_state_term(schedule_game_ids: pd.Series) -> pd.DataFrame:
    table = roof_state_prediction_table()[
        ["game_id", "predicted_open", "prediction_basis"]
    ].drop_duplicates(subset="game_id")
    table["game_id"] = table["game_id"].astype(str)
    table = table.loc[table["game_id"].isin(set(schedule_game_ids))].copy()
    table["roof_state_term"] = table["predicted_open"].astype(float)
    return table


def multiplicity_adjustment(original_results_path: Path, n_looks: int) -> dict[str, object]:
    import json

    payload = json.loads(original_results_path.read_text(encoding="utf-8"))
    roof_term = next(t for t in payload["terms"] if t.get("label") == "roof_state_predicted_open")
    lm_cell = roof_term["line_move_toward_pick_cell"]
    adjustments = {}
    for block_name, probability_positive in (
        ("season_block", lm_cell["season_block_probability_positive"]),
        ("week_block", lm_cell["week_block_probability_positive"]),
    ):
        implied_two_sided_p = 2.0 * min(probability_positive, 1.0 - probability_positive)
        bonferroni_p = min(1.0, implied_two_sided_p * n_looks)
        sidak_p = 1.0 - (1.0 - implied_two_sided_p) ** n_looks
        adjustments[block_name] = {
            "unadjusted_probability_positive": probability_positive,
            "implied_two_sided_p": implied_two_sided_p,
            "n_looks": n_looks,
            "bonferroni_adjusted_p": bonferroni_p,
            "sidak_adjusted_p": sidak_p,
        }
    return {
        "source": str(original_results_path.relative_to(REPO)).replace("\\", "/"),
        "original_mean_points": lm_cell["mean_points"],
        "original_season_block_interval": [
            lm_cell["season_block_interval_low"],
            lm_cell["season_block_interval_high"],
        ],
        "original_week_block_interval": [
            lm_cell["week_block_interval_low"],
            lm_cell["week_block_interval_high"],
        ],
        "original_accuracy_decisive_record": {
            "wins": roof_term["accuracy_companion_cell"]["decisive_side_a_wins"],
            "losses": roof_term["accuracy_companion_cell"]["decisive_side_b_wins"],
        },
        "adjustments": adjustments,
    }


def main() -> int:
    now = datetime.now(UTC)
    population = load_population()
    sbr = load_sbr_open_move()
    merged = population.merge(sbr, on="game_id", how="inner")

    roof_table = load_roof_state_term(merged["game_id"])
    merged = merged.merge(roof_table, on="game_id", how="left")
    merged["predicted_open"] = merged["predicted_open"].fillna(False)
    merged["roof_state_term"] = merged["predicted_open"].astype(float)

    required_cols = [
        "roof_state_term",
        "sbr_open_move",
        "model_logit",
        "composition_flag_sum",
        "home_covered",
    ]
    subset = merged.dropna(subset=required_cols).reset_index(drop=True)

    features_variant = (*BASE_FEATURES, "roof_state_term")
    base_oos, base_folds = loso(subset, BASE_FEATURES)
    variant_oos, variant_folds = loso(subset, features_variant)

    frame = subset.copy()
    frame["base_oos_p"] = base_oos
    frame["variant_oos_p"] = variant_oos
    frame = frame.loc[frame["base_oos_p"].notna() & frame["variant_oos_p"].notna()].copy()

    frame["base_pick_home"] = frame["base_oos_p"].ge(0.5)
    frame["variant_pick_home"] = frame["variant_oos_p"].ge(0.5)
    frame["lm_base"] = np.where(frame["base_pick_home"], 1.0, -1.0) * frame["sbr_open_move"]
    frame["lm_variant"] = np.where(frame["variant_pick_home"], 1.0, -1.0) * frame["sbr_open_move"]
    frame["diff_line_move"] = frame["lm_variant"] - frame["lm_base"]

    frame["base_correct"] = (
        frame["base_pick_home"].astype(float).eq(frame["home_covered"]).astype(float)
    )
    frame["variant_correct"] = (
        frame["variant_pick_home"].astype(float).eq(frame["home_covered"]).astype(float)
    )
    frame["diff_accuracy"] = frame["variant_correct"] - frame["base_correct"]

    line_move_cell = cell_stats(
        "roof_state_predicted_open_line_move_toward_pick_replication_2011_2019",
        frame,
        "diff_line_move",
        BOOTSTRAP_SEED,
        BOOTSTRAP_DRAWS,
    )
    accuracy_cell = cell_stats(
        "roof_state_predicted_open_accuracy_companion_replication_2011_2019",
        frame,
        "diff_accuracy",
        BOOTSTRAP_SEED + 500,
        BOOTSTRAP_DRAWS,
    )

    has_basis = "prediction_basis" in subset.columns
    walk_forward_rows = (
        int(subset["prediction_basis"].eq("logistic_walk_forward_prior_seasons_only").sum())
        if has_basis
        else None
    )
    cold_start_rows = (
        int(len(subset) - (walk_forward_rows or 0)) if walk_forward_rows is not None else None
    )

    results = {
        "command": "python scripts/roof_state_line_move_replication.py",
        "created_at_utc": now.isoformat(),
        "unit": (
            "Out-of-sample replication of roof_state_predicted_open on 2011-2019 "
            "using SBR proxy open/close lines"
        ),
        "population_source": str(EXTENDED_POPULATION.relative_to(REPO)).replace("\\", "/"),
        "sbr_odds_source": str(SBR_ODDS.relative_to(REPO)).replace("\\", "/"),
        "season_start": REPLICATION_SEASON_START,
        "season_end": REPLICATION_SEASON_END,
        "extended_population_rows_2011_2019": len(population),
        "sbr_matched_rows": len(merged),
        "paired_games": len(frame),
        "roof_state_term_nonzero_rate": float(frame["roof_state_term"].astype(bool).mean()),
        "roof_state_walk_forward_rows_in_subset": walk_forward_rows,
        "roof_state_cold_start_rows_in_subset": cold_start_rows,
        "forecast_archive_note": FORECAST_ARCHIVE_NOTE,
        "opener_source_note": (
            "artifacts/extended_fit_population/20260923T205910Z/summary.json confirms "
            "2011-2019 rows in this population already carry opener_source="
            "'sbr_proxy_discrete', i.e. model_logit and home_covered for these rows are "
            "already built and settled against the same SBR proxy open this script uses "
            "for the line-move grade, so the line-move and accuracy companion here are "
            "scored consistently against one instrument."
        ),
        "base_features": list(BASE_FEATURES),
        "variant_features": list(features_variant),
        "seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "base_fold_coefficients": base_folds,
        "variant_fold_coefficients": variant_folds,
        "line_move_toward_pick_cell": line_move_cell,
        "accuracy_companion_cell": accuracy_cell,
        "multiplicity_adjusted_original_result": multiplicity_adjustment(
            LEGACY_RESULTS, ORIGINAL_LOOKS
        ),
    }

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts" / "roof_state_line_move_replication" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    summary = {
        "artifact": str(out_dir.relative_to(REPO)).replace("\\", "/"),
        "paired_games": len(frame),
        "roof_state_term_nonzero_rate": results["roof_state_term_nonzero_rate"],
        "line_move_mean_points": line_move_cell["mean_points"],
        "line_move_season_block_interval": [
            line_move_cell["season_block_interval_low"],
            line_move_cell["season_block_interval_high"],
        ],
        "line_move_season_block_probability_positive": line_move_cell[
            "season_block_probability_positive"
        ],
        "line_move_week_block_interval": [
            line_move_cell["week_block_interval_low"],
            line_move_cell["week_block_interval_high"],
        ],
        "line_move_week_block_probability_positive": line_move_cell[
            "week_block_probability_positive"
        ],
        "accuracy_mean_points": accuracy_cell["mean_points"],
        "accuracy_decisive_record": [
            accuracy_cell["decisive_side_a_wins"],
            accuracy_cell["decisive_side_b_wins"],
        ],
        "multiplicity_adjusted_original_result": results["multiplicity_adjusted_original_result"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
