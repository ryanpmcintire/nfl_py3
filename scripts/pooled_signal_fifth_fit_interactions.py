import argparse
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

import pooled_signal_second_fit as base_fit  # noqa: E402

BASE_TERMS = base_fit.BASE_FEATURES
INTERACTION_TERMS = ("flag_sum_x_move_toward_home", "model_logit_x_move_available")
ALL_TERMS = BASE_TERMS + INTERACTION_TERMS
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923


def add_interaction_columns(population):
    enriched = population.copy()
    enriched["flag_sum_x_move_toward_home"] = enriched["composition_flag_sum"].astype(
        float
    ) * enriched["market_move_toward_home"].astype(float)
    enriched["model_logit_x_move_available"] = enriched["model_logit"].astype(float) * enriched[
        "market_move_available"
    ].astype(float)
    return enriched


def season_block_bootstrap(frame, value_col, draws, seed):
    rng = np.random.default_rng(seed)
    seasons = sorted(int(value) for value in frame["season"].unique())
    grouped = {
        season: frame.loc[frame["season"].eq(season), value_col].to_numpy() for season in seasons
    }
    point = float(frame[value_col].mean())
    effects = np.empty(draws)
    n = len(seasons)
    for draw in range(draws):
        picked = rng.integers(0, n, size=n)
        pooled = np.concatenate([grouped[seasons[i]] for i in picked])
        effects[draw] = pooled.mean()
    low, high = np.quantile(effects, [0.025, 0.975])
    probability_positive = float((effects > 0.0).mean())
    return point, float(low), float(high), probability_positive


def reliability_table(probs, truth, bins):
    frame = pd.DataFrame(
        {"prob": np.asarray(probs, dtype=float), "truth": truth.to_numpy(dtype=float)}
    )
    frame["bin"] = pd.qcut(frame["prob"], bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        predicted_mean=("prob", "mean"), observed_rate=("truth", "mean"), games=("truth", "size")
    )
    return [
        {
            "bin": str(idx),
            "predicted_mean": float(row["predicted_mean"]),
            "observed_rate": float(row["observed_rate"]),
            "games": int(row["games"]),
        }
        for idx, row in grouped.iterrows()
    ]


def line_move_cell(population, interaction_probs, base_probs, seed, draws):
    frame = population.copy()
    frame["interaction_pick_home"] = interaction_probs.ge(0.5)
    frame["base_pick_home"] = base_probs.ge(0.5)
    frame = frame.loc[interaction_probs.notna() & base_probs.notna() & frame["open_move"].notna()]
    frame["lm_interaction"] = (
        np.where(frame["interaction_pick_home"], 1.0, -1.0) * frame["open_move"]
    )
    frame["lm_base"] = np.where(frame["base_pick_home"], 1.0, -1.0) * frame["open_move"]
    frame["diff_interaction_vs_base"] = frame["lm_interaction"] - frame["lm_base"]
    point, low, high, probability_positive = season_block_bootstrap(
        frame, "diff_interaction_vs_base", draws, seed + 2
    )
    decisive = frame.loc[frame["diff_interaction_vs_base"].ne(0.0)]
    return {
        "label": "interaction_vs_base_line_move_toward_pick",
        "note": (
            "both variants include market_move_toward_home / "
            "market_move_available, so this cell inherits the "
            "line_move_yardstick_paired_eval contamination caveat; "
            "secondary check, not decision-relevant alone"
        ),
        "games": len(frame),
        "mean_points_of_line": point,
        "season_block_interval_low": low,
        "season_block_interval_high": high,
        "season_block_probability_positive": probability_positive,
        "decisive_games": len(decisive),
        "decisive_interaction_wins": int((decisive["diff_interaction_vs_base"] > 0.0).sum()),
        "decisive_base_wins": int((decisive["diff_interaction_vs_base"] < 0.0).sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)

    from nfl_ats.pick_probability_fit import FIT_ITERATIONS, FIT_RIDGE, build_fit_population

    population, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    enriched = add_interaction_columns(population)

    base_oos, base_folds, base_insample, _base_natural = base_fit.loso_fit(
        enriched, list(BASE_TERMS), FIT_RIDGE, FIT_ITERATIONS
    )
    inter_oos, inter_folds, inter_insample, inter_natural = base_fit.loso_fit(
        enriched, list(ALL_TERMS), FIT_RIDGE, FIT_ITERATIONS
    )

    scored = enriched.copy()
    scored["base_oos_prob"] = base_oos
    scored["inter_oos_prob"] = inter_oos
    scored = scored.loc[scored["base_oos_prob"].notna() & scored["inter_oos_prob"].notna()].copy()

    scored["base_correct"] = (
        scored["base_oos_prob"].ge(0.5).astype(float).eq(scored["home_covered"]).astype(float)
    )
    scored["inter_correct"] = (
        scored["inter_oos_prob"].ge(0.5).astype(float).eq(scored["home_covered"]).astype(float)
    )
    scored["diff_vs_four_term"] = scored["inter_correct"] - scored["base_correct"]
    scored["diff_vs_model_only"] = scored["inter_correct"] - scored["model_correct"]

    vs_four_week_block = base_fit.paired_summary(scored, "diff_vs_four_term")
    vs_model_week_block = base_fit.paired_summary(scored, "diff_vs_model_only")

    season_point, season_low, season_high, season_pp = season_block_bootstrap(
        scored, "diff_vs_four_term", args.draws, args.seed
    )
    decisive_four = scored.loc[scored["diff_vs_four_term"].ne(0.0)]
    vs_four_season_block = {
        "effect_accuracy_points": season_point * 100.0,
        "interval_low": season_low * 100.0,
        "interval_high": season_high * 100.0,
        "probability_positive": season_pp,
        "decisive_games": len(decisive_four),
        "decisive_wins": int((decisive_four["diff_vs_four_term"] > 0.0).sum()),
        "decisive_losses": int((decisive_four["diff_vs_four_term"] < 0.0).sum()),
    }

    truth = scored["home_covered"]
    metrics = {
        "inter_oos_accuracy": base_fit.accuracy(scored["inter_oos_prob"], truth),
        "inter_insample_accuracy": base_fit.accuracy(inter_insample, enriched["home_covered"]),
        "base_oos_accuracy": base_fit.accuracy(scored["base_oos_prob"], truth),
        "base_insample_accuracy": base_fit.accuracy(base_insample, enriched["home_covered"]),
        "model_only_accuracy": float(scored["model_correct"].mean()),
        "inter_oos_brier": base_fit.brier(scored["inter_oos_prob"], truth),
        "base_oos_brier": base_fit.brier(scored["base_oos_prob"], truth),
        "model_oos_brier": base_fit.brier(scored["model_probability"], truth),
        "inter_oos_logloss": base_fit.logloss(scored["inter_oos_prob"], truth),
        "base_oos_logloss": base_fit.logloss(scored["base_oos_prob"], truth),
        "model_oos_logloss": base_fit.logloss(scored["model_probability"], truth),
    }
    metrics["inter_insample_minus_oos_gap"] = (
        metrics["inter_insample_accuracy"] - metrics["inter_oos_accuracy"]
    )
    metrics["base_insample_minus_oos_gap"] = (
        metrics["base_insample_accuracy"] - metrics["base_oos_accuracy"]
    )

    fold_frame = pd.DataFrame(inter_folds).T
    stability = {
        name: {
            "mean": float(fold_frame[name].mean()),
            "sd": float(fold_frame[name].std(ddof=0)),
            "min": float(fold_frame[name].min()),
            "max": float(fold_frame[name].max()),
            "full_sample": float(inter_natural[name]),
        }
        for name in fold_frame.columns
    }

    reliability = reliability_table(scored["inter_oos_prob"], truth, 5)

    lm_cell = None
    if "open_move" in enriched.columns:
        lm_cell = line_move_cell(
            scored, scored["inter_oos_prob"], scored["base_oos_prob"], args.seed, args.draws
        )

    results = {
        "command": "python scripts/pooled_signal_fifth_fit_interactions.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_terms": list(BASE_TERMS),
        "interaction_terms": list(INTERACTION_TERMS),
        "mechanism": {
            "flag_sum_x_move_toward_home": (
                "composition flags detect schedule-spot mismatches the market "
                "may already be pricing; interaction tests whether flag_sum's "
                "marginal weight is conditional on the direction the market "
                "already moved rather than constant"
            ),
            "model_logit_x_move_available": (
                "market_move_available is 0/1 for whether an independent "
                "sharp-book move signal was observed; interaction lets "
                "model_logit's weight differ between games with and without "
                "an independent corroborating/contradicting market read"
            ),
        },
        "looks_count": 2,
        "bootstrap_draws": args.draws,
        "seed": args.seed,
        "blocking_vs_four_term_primary": "season",
        "blocking_vs_four_term_week_block_secondary": "season-week",
        "blocking_vs_model_only": "season-week",
        "sample_games": len(scored),
        "sample_blocks_season": int(scored["season"].nunique()),
        "sample_blocks_season_week": int(
            (scored["season"].astype(str) + "-w" + scored["week"].astype(str)).nunique()
        ),
        "vs_four_term_season_block": vs_four_season_block,
        "vs_four_term_week_block": vs_four_week_block,
        "vs_model_only_week_block": vs_model_week_block,
        "metrics": metrics,
        "fold_coefficients": inter_folds,
        "base_fold_coefficients": base_folds,
        "coefficient_stability": stability,
        "reliability_table": reliability,
        "line_move_toward_pick_cell": lm_cell,
        "population_provenance": provenance,
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts" / "pooled_signal_fifth_fit" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(out_path.relative_to(REPO)).replace("\\", "/"),
                "vs_four_term_season_block": vs_four_season_block,
                "vs_four_term_week_block": vs_four_week_block,
                "vs_model_only_week_block": vs_model_week_block,
                "metrics": metrics,
                "line_move_toward_pick_cell": lm_cell,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
