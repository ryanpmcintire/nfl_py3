from __future__ import annotations

import json
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

from line_move_yardstick_paired_eval import cell_stats  # noqa: E402
from opener_error_transfer_unit2 import FEATURES as CFB_FEATURES  # noqa: E402
from opener_error_transfer_unit2 import (  # noqa: E402
    attach_transfer_features,
    key_number_distance,
    load_cfb_population,
)
from opener_error_transfer_unit2 import design as cfb_design  # noqa: E402
from opener_error_transfer_unit2 import fit_logit_beta as cfb_fit_logit_beta  # noqa: E402
from opener_error_transfer_unit2 import predict_logit as cfb_predict_logit  # noqa: E402
from opener_error_transfer_unit2 import standardize as cfb_standardize  # noqa: E402
from players_on_field_rating_eval import loso  # noqa: E402
from pooled_signal_sixth_fit_new_family import NEW_TERM as REDDIT_TERM  # noqa: E402
from pooled_signal_sixth_fit_new_family import REDDIT_PARQUET  # noqa: E402

from nfl_ats.pick_probability import FLAG_SUM_COLUMN  # noqa: E402
from nfl_ats.pick_probability_fit import build_fit_population  # noqa: E402

BASE_FEATURES = ("model_logit", "composition_flag_sum")
EARLY_WINDOW_MAX_WEEK = 4
RATINGS_TABLE = (
    REPO / "artifacts" / "latent_ratings_on_production" / "expected_lineup_ratings.parquet"
)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923


def add_lineup_terms(population: pd.DataFrame) -> pd.DataFrame:
    ratings = pd.read_parquet(RATINGS_TABLE)[
        ["game_id", "diff_lineup_total", "diff_divergence"]
    ].copy()
    return population.merge(ratings, on="game_id", how="left")


def add_gated_flag_term(population: pd.DataFrame) -> pd.DataFrame:
    week = population["week"].astype(int)
    early = week.le(EARLY_WINDOW_MAX_WEEK)
    protection = population["flag_protection"].astype(float)
    served_flag_sum = population[FLAG_SUM_COLUMN].astype(float)
    out = population.copy()
    out["gated_flag_sum"] = served_flag_sum - protection * (~early).astype(float)
    return out


def add_reddit_term(population: pd.DataFrame) -> pd.DataFrame:
    reddit = pd.read_parquet(REDDIT_PARQUET, columns=["game_id", REDDIT_TERM])
    lookup = reddit.drop_duplicates("game_id").set_index("game_id")[REDDIT_TERM]
    out = population.copy()
    mapped = out["game_id"].astype(str).map(lookup)
    out[REDDIT_TERM] = mapped.fillna(0.0).astype(float)
    return out


def add_transfer_term(population: pd.DataFrame) -> pd.DataFrame:
    cfb = load_cfb_population()
    means_cfb, stds_cfb = cfb_standardize(cfb, CFB_FEATURES)
    beta_logit = cfb_fit_logit_beta(
        cfb_design(cfb, CFB_FEATURES, means_cfb, stds_cfb), cfb["home_covered"].to_numpy()
    )
    out = population.copy()
    out["home_favorite"] = out["tue_open_home_spread"].lt(0.0).astype(float)
    out["spread_size"] = out["tue_open_home_spread"].abs()
    out["key_number_distance"] = key_number_distance(out["tue_open_home_spread"])
    out = attach_transfer_features(out, date_col="gameday", move_col="open_move")
    p = cfb_predict_logit(out, CFB_FEATURES, beta_logit, means_cfb, stds_cfb)
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    out["cfb_transfer_logit"] = np.log(clipped / (1.0 - clipped))
    return out


TERM_DECLARATIONS = (
    {
        "label": "diff_lineup_total",
        "term_name": "diff_lineup_total",
        "term_columns": ("diff_lineup_total",),
        "builder": add_lineup_terms,
        "source": "players_on_field_rating_eval.py / artifacts/latent_ratings_on_production",
    },
    {
        "label": "diff_divergence",
        "term_name": "diff_divergence",
        "term_columns": ("diff_divergence",),
        "builder": add_lineup_terms,
        "source": "players_on_field_rating_eval.py / artifacts/latent_ratings_on_production",
    },
    {
        "label": "cfb_transfer_logit",
        "term_name": "cfb_transfer_logit",
        "term_columns": ("cfb_transfer_logit",),
        "builder": add_transfer_term,
        "source": "opener_error_transfer_unit2.py (CFB-trained opener-error transfer logit)",
    },
    {
        "label": "reddit_home_comment_ratio_elevated",
        "term_name": REDDIT_TERM,
        "term_columns": (REDDIT_TERM,),
        "builder": add_reddit_term,
        "source": (
            "pooled_signal_sixth_fit_new_family.py / "
            "data/processed/game_features_weak_stack_reddit.parquet"
        ),
    },
    {
        "label": "week_gated_protection_flag_sum",
        "term_name": "gated_flag_sum",
        "term_columns": ("gated_flag_sum",),
        "builder": add_gated_flag_term,
        "source": "lead65_gated_flag_in_fit.py (protection flag zeroed after week 4)",
    },
)


def variant_report(
    declaration: dict, population: pd.DataFrame, seed: int, draws: int
) -> dict[str, object]:
    enriched = declaration["builder"](population)
    term_columns = list(declaration["term_columns"])
    subset = enriched.dropna(subset=[*term_columns, "open_move"]).reset_index(drop=True)

    features_variant = (*BASE_FEATURES, *term_columns)
    base_oos, base_folds = loso(subset, BASE_FEATURES)
    variant_oos, variant_folds = loso(subset, features_variant)

    frame = subset.copy()
    frame["base_oos_p"] = base_oos
    frame["variant_oos_p"] = variant_oos
    frame = frame.loc[frame["base_oos_p"].notna() & frame["variant_oos_p"].notna()].copy()

    frame["base_pick_home"] = frame["base_oos_p"].ge(0.5)
    frame["variant_pick_home"] = frame["variant_oos_p"].ge(0.5)
    frame["lm_base"] = np.where(frame["base_pick_home"], 1.0, -1.0) * frame["open_move"]
    frame["lm_variant"] = np.where(frame["variant_pick_home"], 1.0, -1.0) * frame["open_move"]
    frame["diff_line_move"] = frame["lm_variant"] - frame["lm_base"]

    frame["base_correct"] = (
        frame["base_pick_home"].astype(float).eq(frame["home_covered"]).astype(float)
    )
    frame["variant_correct"] = (
        frame["variant_pick_home"].astype(float).eq(frame["home_covered"]).astype(float)
    )
    frame["diff_accuracy"] = frame["variant_correct"] - frame["base_correct"]

    line_move_cell = cell_stats(
        f"{declaration['label']}_line_move_toward_pick", frame, "diff_line_move", seed, draws
    )
    accuracy_cell = cell_stats(
        f"{declaration['label']}_accuracy_companion", frame, "diff_accuracy", seed + 500, draws
    )
    return {
        "label": declaration["label"],
        "term_name": declaration["term_name"],
        "source": declaration["source"],
        "base_features": list(BASE_FEATURES),
        "variant_features": list(features_variant),
        "coverage_games_before_pairing": len(subset),
        "paired_games": len(frame),
        "seasons": sorted(int(v) for v in frame["season"].unique()),
        "base_fold_coefficients": base_folds,
        "variant_fold_coefficients": variant_folds,
        "line_move_toward_pick_cell": line_move_cell,
        "accuracy_companion_cell": accuracy_cell,
    }


def main() -> int:
    now = datetime.now(UTC)
    population, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    population = population.dropna(subset=["open_move"]).reset_index(drop=True)

    reports = [
        variant_report(declaration, population, BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
        for declaration in TERM_DECLARATIONS
    ]

    results = {
        "command": "python scripts/tuesday_terms_line_move.py",
        "created_at_utc": now.isoformat(),
        "unit": "ENG-47 tuesday-terms line-move yardstick",
        "predeclared_design": (
            "base = Tuesday-knowable fit (model_logit + composition_flag_sum only, the "
            "two market-move FIT_FEATURES terms market_move_toward_home / "
            "market_move_available dropped per the CONTAMINATED finding in "
            "artifacts/line_move_yardstick/); variant = base plus one Tuesday-knowable "
            "term at a time (diff_lineup_total, diff_divergence, cfb_transfer_logit, "
            "reddit_home_comment_ratio_elevated, gated_flag_sum -- the served four-term "
            "fit's protection-mismatch flag zeroed for weeks after 4). LOSO 2020-2025 "
            "refit per fold for base and variant on the same paired game subset. Primary "
            "metric: signed close-minus-open points toward the pick (line_move_toward_pick "
            "construction from src/nfl_ats/clv.py), variant minus base, season-block "
            "bootstrap (2000 draws, season-week block secondary). Accuracy (home_covered "
            "pick correctness) is the companion look, same bootstrap, not the primary "
            "decision."
        ),
        "market_move_terms_excluded_from_base": [
            "market_move_toward_home",
            "market_move_available",
        ],
        "base_population_games": len(population),
        "seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "opener_evaluation_provenance": provenance,
        "terms": reports,
    }

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts" / "tuesday_terms_line_move" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    summary = {
        "artifact": str(out_dir.relative_to(REPO)).replace("\\", "/"),
        "base_population_games": len(population),
        "terms": [
            {
                "label": report["label"],
                "paired_games": report["paired_games"],
                "line_move_mean_points": report["line_move_toward_pick_cell"]["mean_points"],
                "line_move_season_block_interval": [
                    report["line_move_toward_pick_cell"]["season_block_interval_low"],
                    report["line_move_toward_pick_cell"]["season_block_interval_high"],
                ],
                "line_move_season_block_probability_positive": report[
                    "line_move_toward_pick_cell"
                ]["season_block_probability_positive"],
                "accuracy_mean_points": report["accuracy_companion_cell"]["mean_points"],
                "accuracy_season_block_probability_positive": report["accuracy_companion_cell"][
                    "season_block_probability_positive"
                ],
            }
            for report in reports
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
