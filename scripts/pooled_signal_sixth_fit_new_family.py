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
NEW_TERM = "reddit_home_comment_ratio_elevated"
ALL_TERMS = (*BASE_TERMS, NEW_TERM)
REDDIT_PARQUET = REPO / "data" / "processed" / "game_features_weak_stack_reddit.parquet"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923


def add_reddit_column(population):
    reddit = pd.read_parquet(REDDIT_PARQUET, columns=["game_id", NEW_TERM])
    lookup = reddit.drop_duplicates("game_id").set_index("game_id")[NEW_TERM]
    enriched = population.copy()
    mapped = enriched["game_id"].astype(str).map(lookup)
    coverage = {
        "matched_games": int(mapped.notna().sum()),
        "unmatched_games": int(mapped.isna().sum()),
        "positive_flag_games_among_matched": int((mapped.fillna(0.0) > 0).sum()),
    }
    enriched[NEW_TERM] = mapped.fillna(0.0).astype(float)
    return enriched, coverage


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


def line_move_cell(population, new_probs, base_probs, seed, draws):
    frame = population.copy()
    frame["new_pick_home"] = new_probs.ge(0.5)
    frame["base_pick_home"] = base_probs.ge(0.5)
    frame = frame.loc[new_probs.notna() & base_probs.notna() & frame["open_move"].notna()]
    frame["lm_new"] = np.where(frame["new_pick_home"], 1.0, -1.0) * frame["open_move"]
    frame["lm_base"] = np.where(frame["base_pick_home"], 1.0, -1.0) * frame["open_move"]
    frame["diff_new_vs_base"] = frame["lm_new"] - frame["lm_base"]
    point, low, high, probability_positive = season_block_bootstrap(
        frame, "diff_new_vs_base", draws, seed + 2
    )
    decisive = frame.loc[frame["diff_new_vs_base"].ne(0.0)]
    return {
        "label": "new_term_vs_base_line_move_toward_pick",
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
        "decisive_new_wins": int((decisive["diff_new_vs_base"] > 0.0).sum()),
        "decisive_base_wins": int((decisive["diff_new_vs_base"] < 0.0).sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)

    from nfl_ats.pick_probability_fit import FIT_ITERATIONS, FIT_RIDGE, build_fit_population

    population, provenance = build_fit_population(REPO / "artifacts", REPO / "data")
    enriched, coverage = add_reddit_column(population)

    base_oos, base_folds, base_insample, _base_natural = base_fit.loso_fit(
        enriched, list(BASE_TERMS), FIT_RIDGE, FIT_ITERATIONS
    )
    new_oos, new_folds, new_insample, new_natural = base_fit.loso_fit(
        enriched, list(ALL_TERMS), FIT_RIDGE, FIT_ITERATIONS
    )

    scored = enriched.copy()
    scored["base_oos_prob"] = base_oos
    scored["new_oos_prob"] = new_oos
    scored = scored.loc[scored["base_oos_prob"].notna() & scored["new_oos_prob"].notna()].copy()

    scored["base_correct"] = (
        scored["base_oos_prob"].ge(0.5).astype(float).eq(scored["home_covered"]).astype(float)
    )
    scored["new_correct"] = (
        scored["new_oos_prob"].ge(0.5).astype(float).eq(scored["home_covered"]).astype(float)
    )
    scored["diff_vs_four_term"] = scored["new_correct"] - scored["base_correct"]
    scored["diff_vs_model_only"] = scored["new_correct"] - scored["model_correct"]

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
        "new_oos_accuracy": base_fit.accuracy(scored["new_oos_prob"], truth),
        "new_insample_accuracy": base_fit.accuracy(new_insample, enriched["home_covered"]),
        "base_oos_accuracy": base_fit.accuracy(scored["base_oos_prob"], truth),
        "base_insample_accuracy": base_fit.accuracy(base_insample, enriched["home_covered"]),
        "model_only_accuracy": float(scored["model_correct"].mean()),
        "new_oos_brier": base_fit.brier(scored["new_oos_prob"], truth),
        "base_oos_brier": base_fit.brier(scored["base_oos_prob"], truth),
        "model_oos_brier": base_fit.brier(scored["model_probability"], truth),
        "new_oos_logloss": base_fit.logloss(scored["new_oos_prob"], truth),
        "base_oos_logloss": base_fit.logloss(scored["base_oos_prob"], truth),
        "model_oos_logloss": base_fit.logloss(scored["model_probability"], truth),
    }
    metrics["new_insample_minus_oos_gap"] = (
        metrics["new_insample_accuracy"] - metrics["new_oos_accuracy"]
    )
    metrics["base_insample_minus_oos_gap"] = (
        metrics["base_insample_accuracy"] - metrics["base_oos_accuracy"]
    )

    fold_frame = pd.DataFrame(new_folds).T
    stability = {
        name: {
            "mean": float(fold_frame[name].mean()),
            "sd": float(fold_frame[name].std(ddof=0)),
            "min": float(fold_frame[name].min()),
            "max": float(fold_frame[name].max()),
            "full_sample": float(new_natural[name]),
        }
        for name in fold_frame.columns
    }

    reliability = reliability_table(scored["new_oos_prob"], truth, 5)

    lm_cell = None
    if "open_move" in enriched.columns:
        lm_cell = line_move_cell(
            scored, scored["new_oos_prob"], scored["base_oos_prob"], args.seed, args.draws
        )

    results = {
        "command": "python scripts/pooled_signal_sixth_fit_new_family.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "unit": "MOD-20 unit 6",
        "family_chosen": "attention_battery / reddit_home_comment_ratio_elevated",
        "family_selection_reasoning": (
            "Predeclared before fitting. Selected by prior evidence only from "
            "registry/weak_signals.json and `nfl-ats weak-signals pool "
            "--league nfl --effect-units accuracy_points`, restricted to families "
            "not among the served 7 composition flags (coach, division, arrests, "
            "bye, cold_visitor, protection, tank_zone) and not player-on-field "
            "(MOD-22) or college-transfer (XLG-09) constructs. Reddit home "
            "comment-ratio attention has three independent, all-positive-sign "
            "measurements at increasing rigor: bare-baseline battery screen "
            "+0.329 pts P+ 0.885 (n=3365, season-blocked secondary 95% "
            "[+0.078,+0.603] P+ 0.996, docs/reddit_attention_on_production.md "
            "section 1); on-production stack (added to the played weak_stack "
            "chain) +1.072 pts week-blocked 95% [+0.133,+2.125] P+ 0.979 "
            "(n=768, docs/reddit_attention_on_production.md section 6, entire "
            "interval positive); and opener-grade confirmation on the "
            "rotation-assigned 2020-2021 window +0.439 pts P+ 0.665 (n=456, "
            "docs/reddit_attention_opener_confirmation.md). No sign flip across "
            "three independent windows -- more consistent than weather/body-clock/"
            "bias-battery candidates surveyed (pp mostly 0.2-0.4 or a "
            "same-hypothesis replication that flipped sign, e.g. "
            "attention_battery_both_cold pp 0.857 vs its gdelt replication pp "
            "0.232). Feature is rebuildable point-in-time for 2020-2025 from "
            "local data: data/raw/arctic_shift/*_{posts,comments}_timeseries_full.json "
            "covers 2010-09 through 2026-09 per team; the already-built widened "
            "table data/processed/game_features_weak_stack_reddit.parquet "
            "(built 2026-09-01 from that raw data via the now-pruned frozen "
            "scripts/arctic_shift_battery_screen.py construction, which used a "
            "Tuesday-ending 7-day window ending at least 5 days before Sunday "
            "kickoff and a shift(1) trailing baseline reset per team-season, "
            "point-in-time-safe by construction) is used here directly rather "
            "than re-deriving the raw pipeline, since only the frozen construction "
            "not a re-implementation, was ever validated as leakage-safe."
        ),
        "predeclared_look": (
            "One look: add reddit_home_comment_ratio_elevated as a fifth fitted "
            "term to the served four-term base (model_logit, "
            "composition_flag_sum, market_move_toward_home, "
            "market_move_available), same ridge (FIT_RIDGE=1e-3) and "
            "standardisation, LOSO 2020-2025 on the 1,503-game fit population, "
            "vs the four-term base as the decision-relevant comparison "
            "(season-block bootstrap primary) and vs model-only as a companion "
            "(week-block, consistent with units 1-5). Missing coverage (no "
            "computable trailing baseline, mostly each team's season-opening "
            "week) is filled to 0.0 (not-elevated) rather than dropped, so the "
            "full 1,503-game population is preserved for comparability with "
            "units 1-5; this is the conservative direction (pulls the new term "
            "toward the null, never away from it, per the same reasoning in "
            "docs/reddit_attention_on_production.md section 2)."
        ),
        "reddit_column_coverage": coverage,
        "base_terms": list(BASE_TERMS),
        "new_term": NEW_TERM,
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
        "fold_coefficients": new_folds,
        "base_fold_coefficients": base_folds,
        "coefficient_stability": stability,
        "reliability_table": reliability,
        "line_move_toward_pick_cell": lm_cell,
        "population_provenance": provenance,
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts" / "pooled_signal_sixth_fit" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(out_path.relative_to(REPO)).replace("\\", "/"),
                "reddit_column_coverage": coverage,
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
