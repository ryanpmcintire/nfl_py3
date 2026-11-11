from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lead98_price_skew import BASE, fit, loso, metrics, predict

from nfl_ats import clv
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.margin import _MARGIN_PROFILE_FEATURE_SETS, margin_feature_columns

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/pick_probability/20261111T011859Z/per_game.parquet"
SERVED_FEATURES = ROOT / "data/processed/game_features_weak_stack.parquet"
SBR = ROOT / "data/processed/sbr_odds.parquet"
MARKET_ROOT = ROOT / "data/market/raw"
OUT = ROOT / "tests/scratch/mod57/p2c"
PROFILE = "weak_stack"
SET_NAME = _MARGIN_PROFILE_FEATURE_SETS[PROFILE][1]
BASE_ALPHA = 10.0
QUIET = 0.5
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 20261111
SCORE_SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
PROBABILITY_METHOD = "gaussian_median"
FAMILY = "mod57_uncorrected_error_model"
BASE_CONFIG = {
    "feature_profile": PROFILE,
    "regressor": "ridge",
    "target": "market_residual",
    "probability_method": PROBABILITY_METHOD,
    "calibration_method": "none",
}
FULL_COLUMNS = tuple(margin_feature_columns("market_residual", PROFILE))
FIT_LOG: list[dict] = []


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1.0 - p))


def make_fit(original, opener, mode):
    def wrapped(frame, **kwargs):
        sub = frame
        if mode != "all":
            known = frame["game_id"].map(opener)
            keep = known.notna()
            if mode == "quiet":
                keep = keep & ((frame["spread_line"] - known).abs() <= QUIET)
            sub = frame.loc[keep]
        model = original(sub, **kwargs)
        rows = sub.loc[sub["ats_margin"].notna() & sub["ats_margin"].ne(0)]
        p = model.predict(rows, probability_method=PROBABILITY_METHOD)["home_cover_probability"]
        hits = int(((p.to_numpy() >= 0.5) == (rows["ats_margin"].to_numpy() > 0)).sum())
        FIT_LOG.append(
            {
                "rows": len(sub),
                "scored_rows": len(rows),
                "hits": hits,
                "max_gameday": str(pd.to_datetime(sub["gameday"]).max().date()),
            }
        )
        return model

    return wrapped


def run_eval(features, tag, mode, opener):
    original_fit = clv.fit_margin_model
    original_set = FEATURE_SETS[SET_NAME]
    FEATURE_SETS[SET_NAME] = FULL_COLUMNS
    clv.fit_margin_model = make_fit(original_fit, opener, mode)
    del FIT_LOG[:]
    try:
        config = {**BASE_CONFIG, "ridge_alpha": BASE_ALPHA, "mod57_p2c_arm": tag}
        scored = clv.opener_pick_evaluation(
            MARKET_ROOT,
            features,
            active_model_config=config,
            min_train_games=DEFAULT_MIN_TRAIN_GAMES,
        )
    finally:
        clv.fit_margin_model = original_fit
        FEATURE_SETS[SET_NAME] = original_set
    log = list(FIT_LOG)
    scored = scored.loc[scored["season"].isin(SCORE_SEASONS)].copy()
    return scored, log


def in_sample_accuracy(log):
    n = sum(r["scored_rows"] for r in log)
    return sum(r["hits"] for r in log) / n


def blocks_for(frame):
    return frame.groupby(["season", "week"]).ngroup().to_numpy()


def boot(d, block, rng):
    n_blocks = int(block.max()) + 1
    sums = np.bincount(block, weights=d, minlength=n_blocks)
    sizes = np.bincount(block, minlength=n_blocks)
    idx = rng.integers(0, n_blocks, (BOOTSTRAP_SAMPLES, n_blocks))
    draws = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return {
        "mean": float(d.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_positive": float((draws > 0).mean()),
    }


def paired(frame, p_new, p_old, rng):
    y = frame["home_covered"].to_numpy(float)
    pn = np.clip(p_new, 1e-9, 1 - 1e-9)
    po = np.clip(p_old, 1e-9, 1 - 1e-9)
    hit_new = ((pn >= 0.5) == (y == 1)).astype(float)
    hit_old = ((po >= 0.5) == (y == 1)).astype(float)
    ll_new = -(y * np.log(pn) + (1 - y) * np.log1p(-pn))
    ll_old = -(y * np.log(po) + (1 - y) * np.log1p(-po))
    block = blocks_for(frame)
    acc = boot(hit_new - hit_old, block, rng)
    acc = {**acc, "mean": acc["mean"] * 100, "ci95": [v * 100 for v in acc["ci95"]]}
    differ = (pn >= 0.5) != (po >= 0.5)
    return {
        "accuracy_points": acc,
        "log_loss_improvement": boot(ll_old - ll_new, block, rng),
        "brier_improvement": boot((po - y) ** 2 - (pn - y) ** 2, block, rng),
        "changed_pick_games": int(differ.sum()),
        "new_wins_on_changed": int(hit_new[differ].sum()),
        "old_wins_on_changed": int(hit_old[differ].sum()),
    }


def per_season(frame, p):
    y = frame["home_covered"].to_numpy(float)
    hit = (np.asarray(p) >= 0.5) == (y == 1)
    out = {}
    for season, idx in frame.groupby("season").indices.items():
        out[str(season)] = f"{int(hit[idx].sum())}-{int(len(idx) - hit[idx].sum())}"
    return out


def opener_table(features, archive_open):
    sbr = pd.read_parquet(SBR).dropna(subset=["game_id", "open_home_spread"])
    sbr = sbr.drop_duplicates("game_id").set_index("game_id")["open_home_spread"]
    sbr = sbr.loc[sbr.index.isin(features.loc[features["season"].lt(2020), "game_id"])]
    return pd.concat([sbr, archive_open.loc[~archive_open.index.isin(sbr.index)]])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    served_features = pd.read_parquet(SERVED_FEATURES)
    per_game = pd.read_parquet(ARTIFACT)
    per_game = per_game.loc[per_game["season"].between(2020, 2025)].reset_index(drop=True)
    move = per_game["open_move"].to_numpy(float)
    quiet = np.abs(move) <= QUIET
    y = per_game["home_covered"].to_numpy(float)
    result = {"family": FAMILY, "games": len(per_game), "quiet_games": int(quiet.sum())}

    base_scored, base_log = run_eval(served_features, "served", "all", pd.Series(dtype=float))
    base_scored = base_scored.set_index("game_id")
    ids = per_game["game_id"]
    base_p = base_scored["home_cover_probability_at_open"].reindex(ids).to_numpy(float)
    model_alone = per_game["model_probability"].to_numpy(float)
    result["baseline"] = {
        "model_alone_all": metrics(model_alone, y),
        "model_alone_quiet": metrics(model_alone[quiet], y[quiet]),
        "rerun_all": metrics(base_p, y),
        "rerun_quiet": metrics(base_p[quiet], y[quiet]),
        "rerun_matches_artifact_max_abs": float(np.nanmax(np.abs(base_p - model_alone))),
        "rerun_missing": int(np.isnan(base_p).sum()),
    }

    archive_open = base_scored["tue_open_home_spread"].dropna()
    opener = opener_table(served_features, archive_open)
    result["opener_known_games"] = len(opener)

    arms = {"served_rerun": base_p}
    resid = {"served_rerun": base_scored["residual_at_open"].reindex(ids).to_numpy(float)}
    result["in_sample_accuracy_vs_close"] = {"served_rerun": in_sample_accuracy(base_log)}
    fit_logs = {"served_rerun": base_log}
    for tag, mode in (("Qc_known_opener_all", "known"), ("Q_quiet_trained", "quiet")):
        scored, log = run_eval(served_features, tag, mode, opener)
        scored = scored.set_index("game_id")
        arms[tag] = scored["home_cover_probability_at_open"].reindex(ids).to_numpy(float)
        resid[tag] = scored["residual_at_open"].reindex(ids).to_numpy(float)
        result["in_sample_accuracy_vs_close"][tag] = in_sample_accuracy(log)
        fit_logs[tag] = log
        result[f"{tag}_missing_games"] = int(np.isnan(arms[tag]).sum())
        result[f"{tag}_training_rows_first_last"] = [log[0]["rows"], log[-1]["rows"]]

    week_start = per_game.groupby(["season", "week"])["gameday"].min().sort_index()
    leaks = 0
    for tag, log in fit_logs.items():
        if len(log) != len(week_start):
            note = f"{tag}: {len(log)} fits for {len(week_start)} weeks"
            result.setdefault("truncation_note", []).append(note)
            continue
        for fit_row, start in zip(log, week_start.to_numpy(), strict=True):
            leaks += int(fit_row["max_gameday"] >= str(start))
    result["training_row_on_or_after_week_start_count"] = leaks

    four_oos, four_folds = loso(per_game, BASE)
    four_in = predict(per_game, BASE, *fit(per_game, BASE))
    arm_four, arm_four_folds, arm_four_in = {}, {}, {}
    for tag in ("Q_quiet_trained", "Qc_known_opener_all"):
        data = per_game.copy()
        data["model_logit"] = logit(arms[tag])
        arm_four[tag], arm_four_folds[tag] = loso(data, BASE)
        arm_four_in[tag] = predict(data, BASE, *fit(data, BASE))
    intercept_oos, _ = loso(per_game, ())
    model_home = per_game["model_pick_home"].to_numpy(bool)
    pick_home = np.where(move > 0, True, np.where(move < 0, False, model_home))
    move_only_p = np.where(pick_home, 0.75, 0.25)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pops = {"quiet": quiet, "all_games": np.ones(len(per_game), dtype=bool)}
    series = {
        "model_alone": model_alone,
        "served_rerun": arms["served_rerun"],
        "Qc_known_opener_all": arms["Qc_known_opener_all"],
        "Q_quiet_trained": arms["Q_quiet_trained"],
        "four_term_served": four_oos,
        "four_term_Q": arm_four["Q_quiet_trained"],
        "four_term_Qc": arm_four["Qc_known_opener_all"],
        "intercept_only": intercept_oos,
        "line_move_only": move_only_p,
    }
    cells = []
    result["populations"] = {}
    for pop, mask in pops.items():
        sub = per_game.loc[mask].reset_index(drop=True)
        ys = y[mask]
        entry = {"games": int(mask.sum()), "records": {}, "vs_model_alone": {}, "per_season": {}}
        for name, p in series.items():
            entry["records"][name] = metrics(p[mask], ys)
            entry["per_season"][name] = per_season(sub, p[mask])
        for name in (
            "Qc_known_opener_all",
            "Q_quiet_trained",
            "intercept_only",
            "line_move_only",
            "four_term_Q",
            "four_term_Qc",
        ):
            entry["vs_model_alone"][name] = paired(sub, series[name][mask], model_alone[mask], rng)
        entry["four_term_Q_vs_four_term_served"] = paired(
            sub, series["four_term_Q"][mask], four_oos[mask], rng
        )
        entry["Q_vs_Qc"] = paired(
            sub, series["Q_quiet_trained"][mask], series["Qc_known_opener_all"][mask], rng
        )
        entry["in_sample_four_term_served"] = metrics(four_in[mask], ys)
        entry["in_sample_four_term_Q"] = metrics(arm_four_in["Q_quiet_trained"][mask], ys)
        result["populations"][pop] = entry
        for metric in ("accuracy_points", "log_loss_improvement", "brier_improvement"):
            v = entry["vs_model_alone"]["Q_quiet_trained"][metric]
            description = (
                "Quiet-trained ridge (training rows whose close is within a half point of "
                f"the opener) minus model alone, held out at the Tuesday opener, {pop}"
            )
            cells.append((f"mod57_p2c_Q_vs_model_alone_{pop}_{metric}", description, v, metric))
            if pop == "all_games":
                v = entry["four_term_Q_vs_four_term_served"][metric]
                description = (
                    "Four-term leave-one-season-out with the quiet-trained model logit "
                    "minus the served four-term, all games"
                )
                cells.append(
                    (f"mod57_p2c_four_term_Q_vs_served_{pop}_{metric}", description, v, metric)
                )

    result["fold_coefficients"] = {
        "four_term_served": four_folds,
        "four_term_Q": arm_four_folds["Q_quiet_trained"],
    }
    result["move_correlation"] = {
        pop: {tag: float(np.corrcoef(resid[tag][mask], move[mask])[0, 1]) for tag in resid}
        for pop, mask in pops.items()
    }
    r_open = per_game["residual_at_open"].to_numpy(float)
    r_close = per_game["residual_at_close"].to_numpy(float)
    result["armK_diagnostic_leaky_not_scored"] = {
        "mean_abs_residual_open_minus_close": float(np.mean(np.abs(r_open - r_close))),
        "pick_agreement_open_vs_close_residual": float(np.mean((r_open > 0) == (r_close > 0))),
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    batch = [
        {
            "name": name,
            "description": description + ", week-block bootstrap",
            "effect": v["mean"],
            "effect_units": units,
            "interval_low": v["ci95"][0],
            "interval_high": v["ci95"][1],
            "probability_positive": v["probability_positive"],
            "season_start": 2020,
            "season_end": 2025,
        }
        for name, description, v, units in cells
    ]
    (OUT / "cells.json").write_text(json.dumps({"cells": batch}, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
