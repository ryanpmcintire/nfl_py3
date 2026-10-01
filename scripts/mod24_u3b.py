from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod23_unit3 as u3
from lead66_unit1 import OPENER, atomic_counts, mass_at_theta, select_band
from nfl_ats.margin import fit_margin_model
from nfl_ats.mass_preserving_lattice import prior_pool, prior_pool_for_week

ROOT = u3.ROOT
OUT = ROOT / "artifacts" / "mod24_u3b"
NGS = ROOT / "data" / "processed" / "game_features_ngs.parquet"
SEED = 20261001
DRAWS = 10_000
P = "diff_ngs_"
COMPACT = [
    "passing__avg_time_to_throw",
    "passing__aggressiveness",
    "passing__completion_percentage_above_expectation",
    "receiving__avg_separation",
    "rushing__rush_yards_over_expected_per_att",
    "rushing__percent_attempts_gte_eight_defenders",
]
METRICS = ("log_loss", "brier", "rps")


def arm_columns(all_cols):
    own = [P + c for c in COMPACT]
    allowed = [P + "allowed__" + c for c in COMPACT]
    return {
        "a_all24": list(all_cols),
        "b_compact6": own,
        "c_compact6_plus_allowed": own + allowed,
    }


def theta_for(atoms, counts, line, target):
    home, away = atoms > line, atoms < line
    if not 0 < target < 1:
        return None

    def diff(theta):
        mass = mass_at_theta(atoms, counts, theta)
        return float(mass[home].sum() / mass[home | away].sum() - target)

    if diff(-2.0) > 0 or diff(2.0) < 0:
        return None
    return brentq(diff, -2.0, 2.0, xtol=1e-13)


def rps(atoms, mass, actual):
    grid = np.arange(min(atoms.min(), actual), max(atoms.max(), actual) + 1)
    cum = np.concatenate([[0.0], np.cumsum(mass)])
    cdf = cum[np.searchsorted(atoms, grid, side="right")]
    return float(np.square(cdf - (actual <= grid)).sum())


def fit_temp(z, y):
    def nll(par):
        q = par[0] * z
        return float(np.mean(np.logaddexp(0, q) - y * q))

    return float(minimize(nll, [1.0], method="BFGS").x[0])


def recal(f):
    d = f.loc[f.margin_vs_open.ne(0)]
    z_all = logit(f.home_cover_probability_at_open.clip(1e-6, 1 - 1e-6))
    out = pd.Series(np.nan, index=f.index)
    temps = {}
    for s in sorted(f.season.unique()):
        tr = d.loc[d.season.ne(s)]
        t = fit_temp(logit(tr.home_cover_probability_at_open.clip(1e-6, 1 - 1e-6)).to_numpy(), (tr.margin_vs_open > 0).astype(float).to_numpy())
        temps[int(s)] = t
        m = f.season.eq(s)
        out[m] = expit(t * z_all[m])
    return out, temps


def score_tables(frames, pool, features, opener_lines):
    result = features.set_index("game_id").result
    gameday = features.set_index("game_id").gameday
    base = frames["base"]
    games = base.assign(gameday=pd.to_datetime(base.index.map(gameday)))
    games = games.reset_index()
    P_ = {}
    temps = {}
    for arm, f in frames.items():
        P_[arm], temps[arm] = recal(f)
    rps_rows = {arm: {} for arm in frames}
    for (season, week), group in games.groupby(["season", "week"], sort=True):
        target = features.loc[features.season.eq(season) & features.week.eq(week)]
        prior = prior_pool_for_week(pool, season=int(season), week=int(week), cutoff=pd.to_datetime(target.gameday).min(), exclude_game_ids=group.game_id)
        for gid in group.game_id:
            line = float(opener_lines.loc[gid])
            actual = float(result.loc[gid])
            selected, _ = select_band(prior, line)
            atoms, counts = atomic_counts(selected)
            for arm, f in frames.items():
                if gid not in f.index or pd.isna(P_[arm].loc[gid]):
                    continue
                t = theta_for(atoms, counts, line, float(P_[arm].loc[gid]))
                if t is not None:
                    rps_rows[arm][gid] = rps(atoms, mass_at_theta(atoms, counts, t), actual)
    tables = {}
    for arm, f in frames.items():
        d = f.loc[f.margin_vs_open.ne(0)]
        y = (d.margin_vs_open > 0).astype(float)
        p = P_[arm].loc[d.index].clip(1e-6, 1 - 1e-6)
        praw = d.home_cover_probability_at_open.clip(1e-6, 1 - 1e-6)
        tables[arm] = pd.DataFrame(
            {
                "season": d.season,
                "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)),
                "brier": (p - y) ** 2,
                "rps": pd.Series(rps_rows[arm]),
                "raw_log_loss": -(y * np.log(praw) + (1 - y) * np.log(1 - praw)),
                "raw_brier": (praw - y) ** 2,
                "correct": d.correct_at_open_probability_rule.astype(float),
                "pick": d.pick_home_at_open_probability_rule,
            }
        )
    return tables, temps


def season_boot(diff, season, seasons, draws):
    s = diff.groupby(season).agg(["sum", "count"]).reindex(seasons).fillna(0)
    num, den = s["sum"].to_numpy(), s["count"].to_numpy()
    boot = num[draws].sum(1) / den[draws].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    per = (s["sum"] / s["count"]).to_numpy()
    return {
        "mean": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "pplus": float((boot > 0).mean()),
        "seasons_pos": int((per > 0).sum()),
        "per_season": {int(a): float(b) for a, b in zip(seasons, per)},
    }


def grade(tables, arms, label):
    seasons = np.array(sorted(tables["base"].season.unique()))
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(seasons), size=(DRAWS, len(seasons)))
    b = tables["base"]
    rows = []
    for arm in arms:
        a = tables[arm]
        j = a.join(b, rsuffix="_b", how="inner")
        row = {"variant": label, "arm": arm, "n": len(j), "record": f"{int(a.correct.sum())}-{len(a) - int(a.correct.sum())}"}
        acc = season_boot(100.0 * (j.correct - j.correct_b), j.season, seasons, draws)
        row["acc_diff_points"] = acc["mean"]
        row["acc_lo"], row["acc_hi"], row["acc_pplus"], row["acc_seasons_pos"] = acc["lo"], acc["hi"], acc["pplus"], acc["seasons_pos"]
        for m in ("log_loss", "brier", "rps", "raw_log_loss", "raw_brier"):
            diff = (j[m + "_b"] - j[m]).dropna()
            g = season_boot(diff, j.season.loc[diff.index], seasons, draws)
            row[m] = g["mean"]
            row[m + "_lo"], row[m + "_hi"], row[m + "_pplus"], row[m + "_seasons_pos"] = g["lo"], g["hi"], g["pplus"], g["seasons_pos"]
            if m == "log_loss":
                row["log_loss_per_season"] = g["per_season"]
        fl = j.loc[j.pick.fillna(-1) != j.pick_b.fillna(-1)]
        row["flipped"] = len(fl)
        row["flip_w_l"] = f"{int(fl.correct.sum())}-{len(fl) - int(fl.correct.sum())}"
        row["base_flip_w_l"] = f"{int(fl.correct_b.sum())}-{len(fl) - int(fl.correct_b.sum())}"
        rows.append(row)
    return rows


def run(features, columns_by_arm, label, store):
    frames = {"base": u3.decisive(u3.run_arm(features, u3.FULL_COLUMNS))}
    for arm, cols in columns_by_arm.items():
        frames[arm] = u3.decisive(u3.run_arm(features, list(u3.FULL_COLUMNS) + cols))
    for arm in frames:
        frames[arm] = frames[arm].loc[frames[arm].season.between(2020, 2025)]
        frames[arm].to_parquet(OUT / f"{label}_{arm}_per_game.parquet")
    store[label] = frames
    return frames


def coefficients(features, cols):
    original = u3.FEATURE_SETS[u3.SET_NAME]
    u3.FEATURE_SETS[u3.SET_NAME] = tuple(list(u3.FULL_COLUMNS) + cols)
    try:
        completed = features.loc[features.result.notna()].copy()
        completed["gameday"] = pd.to_datetime(completed["gameday"])
        out = {}
        for s in range(2020, 2026):
            cutoff = completed.loc[completed.season.eq(s), "gameday"].min()
            tr = completed.loc[completed.gameday.lt(cutoff)]
            model = fit_margin_model(tr, target="market_residual", model_name="ridge", feature_profile=u3.PROFILE, ridge_alpha=u3.BASE_ALPHA)
            coef = model.estimator.named_steps["regressor"].coef_
            names = list(model.feature_columns)
            out[s] = {c: float(coef[names.index(c)]) for c in cols} if len(coef) >= len(names) else None
        return out
    finally:
        u3.FEATURE_SETS[u3.SET_NAME] = original


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ngs = pd.read_parquet(NGS)
    all_cols = [c for c in ngs.columns if c.startswith(P)]
    assert len(all_cols) == 24
    base_features = pd.read_parquet(u3.FEATURES)
    features = base_features.merge(ngs[["game_id"] + all_cols], on="game_id", how="left", validate="one_to_one")
    features[all_cols] = features[all_cols].fillna(0.0)
    arms = arm_columns(all_cols)
    opener = pd.read_parquet(OPENER / "per_game.parquet").set_index("game_id")
    lines = opener.tue_open_home_spread
    slim = pd.read_parquet(u3.FEATURES, columns=["game_id", "season", "week", "gameday", "spread_line", "result"])
    pool = prior_pool(slim, lines)
    store = {}
    allrows = []
    temps_out = {}
    for label, feats in (("zero_fill_all_seasons", features), ("train_2016_plus", features.loc[features.season.ge(2016)].reset_index(drop=True))):
        frames = run(feats, arms, label, store)
        art = pd.read_parquet(u3.BASELINE_DIR / "per_game.parquet").set_index("game_id")["correct_at_open_probability_rule"].dropna()
        if label == "zero_fill_all_seasons":
            mine = frames["base"]["correct_at_open_probability_rule"]
            print("baseline", int(mine.sum()), len(mine), "artifact", int(art.sum()), len(art), flush=True)
        tables, temps = score_tables(frames, pool, slim, lines)
        temps_out[label] = temps
        rows = grade(tables, list(arms), label)
        allrows += rows
        for r in rows:
            print(label, r["arm"], r["record"], round(r["acc_diff_points"], 3), round(r["acc_pplus"], 3), "LL", round(r["log_loss"], 5), round(r["log_loss_pplus"], 3), "BR", round(r["brier"], 6), round(r["brier_pplus"], 3), "RPS", round(r["rps"], 5), round(r["rps_pplus"], 3), "flip", r["flip_w_l"], flush=True)
    coefs = coefficients(features, [P + c for c in COMPACT])
    json.dump({"rows": allrows, "temps": temps_out, "arm_b_standardized_coefficients": coefs}, open(OUT / "results.json", "w"), indent=1, default=str)
    pd.DataFrame(allrows).to_csv(OUT / "results.csv", index=False)
    print(json.dumps(coefs, indent=0))
    cells = []
    for r in allrows:
        if r["variant"] != "zero_fill_all_seasons":
            continue
        cells.append(
            {
                "name": f"mod24_u3b_ngs_{r['arm']}_recal_log_loss_improvement",
                "description": f"NGS family arm {r['arm']} added to the 90-input base, model alone, opener 2020-2025: paired cover log loss after out-of-season temperature recalibration of both sides (positive = arm better); season-blocked bootstrap.",
                "source": "artifacts/mod24_u3b/results.json",
                "effect": r["log_loss"],
                "interval_low": r["log_loss_lo"],
                "interval_high": r["log_loss_hi"],
                "probability_positive": r["log_loss_pplus"],
                "sample_games": r["n"],
                "sample_blocks": 6,
                "plain_summary": "Checks whether adding tracked passing, rushing and receiving measures of each team improves the quality of the model's own cover probabilities.",
            }
        )
    batch = {
        "effect_units": "log_loss_improvement",
        "classification": "unresolved_below_power",
        "classification_evidence": "Interval not resolved to wrong side; no positive control.",
        "category": "modeling",
        "league": "nfl",
        "season_start": 2020,
        "season_end": 2025,
        "family": "mod24_ngs_model_alone",
        "cells": cells,
    }
    json.dump(batch, open(OUT / "weak_signals_batch.json", "w"), indent=1)


main()
