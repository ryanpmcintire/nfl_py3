from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_ats.clv import load_decision_quotes
from nfl_ats.odds import no_vig_probabilities
from nfl_ats.pick_probability import (
    FLAG_SUM_COLUMN,
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
)
from nfl_ats.pick_probability_fit import FIT_RIDGE, _fit_logit
from per_flag_weights_eval import ARTIFACT, SEASONS, SEED, bootstrap, prepare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests/scratch/lead98_price_skew"
SKEW = "price_skew"
RELIABILITY_EDGES = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0]
BASE = ("model_logit", FLAG_SUM_COLUMN, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN)
VARIANTS = {
    "four_term": BASE,
    "five_term": (*BASE, SKEW),
    "model_only": ("model_logit",),
}
VIEWS = ("tuesday", "full")


def skew_table(games):
    q = load_decision_quotes(ROOT / "data/market/raw", labels=["tue_open"], seasons=SEASONS)
    q = q.loc[q["nflverse_game_id"].notna() & q["market"].eq("spreads")]
    q = q.merge(
        games[["game_id", "tue_open_home_spread"]],
        left_on="nflverse_game_id",
        right_on="game_id",
        how="inner",
    )
    q = q.loc[np.isclose(q["home_spread_line"], q["tue_open_home_spread"])]
    keys = ["nflverse_game_id", "bookmaker_key"]
    home = q.loc[q["outcome_side"].eq("HOME")].drop_duplicates(keys, keep="last")
    away = q.loc[q["outcome_side"].eq("AWAY")].drop_duplicates(keys, keep="last")
    both = home[[*keys, "price", "observed_at_utc"]].merge(
        away[[*keys, "price"]], on=keys, suffixes=("_home", "_away")
    )
    both["p_home"] = [
        no_vig_probabilities(h, a)[0] for h, a in zip(both["price_home"], both["price_away"])
    ]
    per_game = both.groupby("nflverse_game_id").agg(
        p_home=("p_home", "median"),
        skew_books=("p_home", "size"),
        last_observed=("observed_at_utc", "max"),
    )
    per_game[SKEW] = np.log(per_game["p_home"] / (1.0 - per_game["p_home"]))
    return per_game.reset_index().rename(columns={"nflverse_game_id": "game_id"})


def stats(train, features):
    means, stds = {}, {}
    for n in features:
        v = train[n].astype(float)
        means[n] = float(v.mean(skipna=True))
        stds[n] = float(v.std(ddof=0, skipna=True)) or 1.0
    return means, stds


def design(frame, features, means, stds):
    cols = [np.ones(len(frame))]
    for n in features:
        z = (frame[n].astype(float) - means[n]) / stds[n]
        cols.append(z.fillna(0.0).to_numpy())
    return np.column_stack(cols)


def fit(train, features):
    means, stds = stats(train, features)
    beta = _fit_logit(
        design(train, features, means, stds), train["home_covered"].to_numpy(float), FIT_RIDGE
    )
    return beta, means, stds


def predict(frame, features, beta, means, stds):
    z = np.clip(design(frame, features, means, stds) @ beta, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def natural(features, beta, stds):
    return {n: float(beta[i + 1] / stds[n]) for i, n in enumerate(features)}


def loso(frame, features):
    oos = np.full(len(frame), np.nan)
    folds = {}
    for held in SEASONS:
        tr = frame.season.ne(held).to_numpy()
        te = ~tr
        b, m, s = fit(frame.loc[tr], features)
        oos[te] = predict(frame.loc[te], features, b, m, s)
        folds[str(held)] = natural(features, b, s)
    return oos, folds


def metrics(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    correct = (p >= 0.5) == (y == 1)
    return {
        "games": int(len(y)),
        "record": f"{int(correct.sum())}-{int(len(y) - correct.sum())}",
        "accuracy": float(correct.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
    }


def reliability(p, y):
    idx = np.digitize(p, RELIABILITY_EDGES[1:-1])
    rows = []
    for b in range(len(RELIABILITY_EDGES) - 1):
        m = idx == b
        rows.append(
            {
                "bin": f"[{RELIABILITY_EDGES[b]:.2f},{RELIABILITY_EDGES[b + 1]:.2f})",
                "games": int(m.sum()),
                "mean_predicted": float(p[m].mean()) if m.any() else None,
                "observed_home_cover": float(y[m].mean()) if m.any() else None,
            }
        )
    return rows


def summary(folds):
    keys = list(next(iter(folds.values())).keys())
    out = {}
    for k in keys:
        v = np.array([folds[f][k] for f in folds])
        out[k] = {
            "mean": float(v.mean()),
            "min": float(v.min()),
            "max": float(v.max()),
            "positive_folds": int((v > 0).sum()),
        }
    return out


def main():
    data = pd.read_parquet(ARTIFACT)
    data = data.loc[data.season.between(2020, 2025)].reset_index(drop=True)
    skew = skew_table(data)
    data = data.merge(skew[["game_id", SKEW, "skew_books", "last_observed"]], on="game_id", how="left")
    y = data["home_covered"].to_numpy(float)
    covered = data[SKEW].notna()
    cov = {
        "games": int(len(data)),
        "with_price": int(covered.sum()),
        "by_season": {
            str(s): [int(g[SKEW].notna().sum()), int(len(g))] for s, g in data.groupby("season")
        },
        "books_median": float(data["skew_books"].median()),
        "latest_quote_weekday_hour_utc": str(data["last_observed"].max()),
        "skew_std": float(data[SKEW].std()),
        "skew_mean": float(data[SKEW].mean()),
    }
    rng = np.random.default_rng(SEED)
    result = {"coverage": cov, "views": {}}
    for view in VIEWS:
        frame = prepare(data, view)
        probs, entry = {}, {}
        for name, feats in VARIANTS.items():
            oos, folds = loso(frame, feats)
            probs[name] = oos
            entry[name] = {
                "oos": metrics(oos, y),
                "oos_on_quoted": metrics(oos[covered.to_numpy()], y[covered.to_numpy()]),
                "reliability": reliability(oos, y),
                "fold_coefficients": folds,
                "fold_summary": summary(folds),
            }
        entry["market"] = {"oos": metrics(np.full(len(y), 0.5), y)}
        new, old = probs["five_term"], probs["four_term"]
        entry["paired_five_minus_four"] = bootstrap(data, new, old, rng)
        entry["paired_five_minus_model_only"] = bootstrap(data, new, probs["model_only"], rng)
        differ = (new >= 0.5) != (old >= 0.5)
        entry["decisive"] = {
            "changed_pick_games": int(differ.sum()),
            "five_term_wins": int(((new >= 0.5) == (y == 1))[differ].sum()),
            "four_term_wins": int(((old >= 0.5) == (y == 1))[differ].sum()),
        }
        result["views"][view] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=1, default=str)
    (OUT / "results.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
