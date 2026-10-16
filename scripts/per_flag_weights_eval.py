from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.pick_probability import (
    COUNTED_FLAG_COLUMNS,
    FLAG_SUM_COLUMN,
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
)
from nfl_ats.pick_probability_fit import FIT_RIDGE, _arrest_incidents, _fit_logit

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/pick_probability/20261013T222004Z/per_game.parquet"
WEEK6 = ROOT / "artifacts/margin_predictions/2026-week-06-20261013T223016Z/predictions.csv"
OUT = ROOT / "tests/scratch/per_flag_weights"
SEED = 20261014
DRAWS = 10000
BINS = 5
SEASONS = range(2020, 2026)
VARIANTS = {
    "pooled": ("model_logit", FLAG_SUM_COLUMN, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN),
    "per_flag": ("model_logit", *COUNTED_FLAG_COLUMNS, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN),
}
VIEWS = ("tuesday", "full")


def prepare(frame, view):
    out = frame.copy()
    if view == "tuesday":
        out[MOVE_COLUMN] = 0.0
        out[MOVE_AVAILABLE_COLUMN] = 0.0
    return out


def design(frame, features, means, stds):
    cols = [np.ones(len(frame))]
    for n in features:
        cols.append(((frame[n].astype(float) - means[n]) / stds[n]).to_numpy())
    return np.column_stack(cols)


def fit(train, features):
    means = {n: float(train[n].mean()) for n in features}
    stds = {n: float(train[n].std(ddof=0)) or 1.0 for n in features}
    x = design(train, features, means, stds)
    beta = _fit_logit(x, train["home_covered"].to_numpy(float), FIT_RIDGE)
    return beta, means, stds


def predict(frame, features, beta, means, stds):
    z = np.clip(design(frame, features, means, stds) @ beta, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def natural(features, beta, means, stds):
    out = {n: float(beta[i + 1] / stds[n]) for i, n in enumerate(features)}
    out["intercept"] = float(
        beta[0] - sum(beta[i + 1] * means[n] / stds[n] for i, n in enumerate(features))
    )
    return out


def metrics(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    correct = (p >= 0.5) == (y == 1)
    return {
        "games": int(len(y)),
        "accuracy": float(correct.mean()),
        "record": f"{int(correct.sum())}-{int(len(y) - correct.sum())}",
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
    }


def reliability(p, y):
    order = np.argsort(p, kind="stable")
    rows = []
    for chunk in np.array_split(order, BINS):
        rows.append(
            {
                "games": int(len(chunk)),
                "mean_predicted": float(p[chunk].mean()),
                "observed_home_cover": float(y[chunk].mean()),
            }
        )
    return rows


def run_variant(data, view, name):
    features = VARIANTS[name]
    frame = prepare(data, view)
    beta, means, stds = fit(frame, features)
    in_sample = predict(frame, features, beta, means, stds)
    oos = pd.Series(np.nan, index=frame.index)
    folds = {}
    for held in SEASONS:
        train = frame.loc[frame.season.ne(held)]
        test = frame.loc[frame.season.eq(held)]
        b, m, s = fit(train, features)
        oos.loc[test.index] = predict(test, features, b, m, s)
        folds[str(held)] = natural(features, b, m, s)
    return {
        "in_sample": in_sample,
        "oos": oos.to_numpy(float),
        "folds": folds,
        "full_fit": natural(features, beta, means, stds),
    }


def fold_summary(folds):
    keys = list(next(iter(folds.values())).keys())
    out = {}
    for k in keys:
        v = np.array([folds[f][k] for f in folds])
        out[k] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    return out


def bootstrap(data, p_new, p_old, rng):
    y = data["home_covered"].to_numpy(float)
    block = data.groupby(["season", "week"]).ngroup().to_numpy()
    n_blocks = block.max() + 1
    acc_d = ((p_new >= 0.5) == (y == 1)).astype(float) - ((p_old >= 0.5) == (y == 1)).astype(float)
    br_d = (p_new - y) ** 2 - (p_old - y) ** 2
    n_games = np.bincount(block, minlength=n_blocks).astype(float)
    acc_sum = np.bincount(block, weights=acc_d, minlength=n_blocks)
    br_sum = np.bincount(block, weights=br_d, minlength=n_blocks)
    draws = rng.integers(0, n_blocks, size=(DRAWS, n_blocks))
    games = n_games[draws].sum(axis=1)
    acc = 100.0 * acc_sum[draws].sum(axis=1) / games
    br = br_sum[draws].sum(axis=1) / games
    return {
        "accuracy_points_delta_mean": float(acc.mean()),
        "accuracy_points_ci95": [float(np.quantile(acc, 0.025)), float(np.quantile(acc, 0.975))],
        "accuracy_probability_positive": float((acc > 0).mean()),
        "brier_delta_mean": float(br.mean()),
        "brier_ci95": [float(np.quantile(br, 0.025)), float(np.quantile(br, 0.975))],
        "brier_probability_improvement": float((br < 0).mean()),
    }


def week6(data):
    from nfl_ats.pick_probability import signed_composition_flags
    from nfl_ats.snapshots import latest_snapshot, load_snapshot

    data_root = ROOT / "data"
    preds = pd.read_csv(WEEK6).drop_duplicates("game_id").reset_index(drop=True)
    schedules, _ = load_snapshot(latest_snapshot(data_root / "raw"))
    flags = signed_composition_flags(preds, schedules, incidents=_arrest_incidents(data_root))
    preds = preds.merge(flags, on="game_id", how="left")
    p = preds["home_cover_probability"].astype(float).clip(1e-6, 1 - 1e-6)
    preds["model_logit"] = np.log(p / (1 - p))
    preds[MOVE_COLUMN] = 0.0
    preds[MOVE_AVAILABLE_COLUMN] = 0.0
    frame = prepare(data, "tuesday")
    out = {}
    for name, features in VARIANTS.items():
        b, m, s = fit(frame, features)
        out[name] = predict(preds, features, b, m, s)
    rows = []
    for i, r in preds.iterrows():
        rows.append(
            {
                "game": f"{r['away_team']}@{r['home_team']}",
                "model_home_cover_probability": float(p.iloc[i]),
                "flags": {c: int(r[c]) for c in COUNTED_FLAG_COLUMNS if int(r[c]) != 0},
                "pooled_home_probability": float(out["pooled"][i]),
                "per_flag_home_probability": float(out["per_flag"][i]),
                "pooled_pick": r["home_team"] if out["pooled"][i] >= 0.5 else r["away_team"],
                "per_flag_pick": r["home_team"] if out["per_flag"][i] >= 0.5 else r["away_team"],
            }
        )
    return rows


def main():
    data = pd.read_parquet(ARTIFACT)
    data = data.loc[data.season.between(2020, 2025)].reset_index(drop=True)
    y = data["home_covered"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    result = {"games": len(data), "variants": {}, "paired": {}, "decisive": {}}
    probs = {}
    for view in VIEWS:
        for name in VARIANTS:
            r = run_variant(data, view, name)
            probs[(view, name)] = r
            mo = metrics(r["oos"], y)
            mi = metrics(r["in_sample"], y)
            result["variants"][f"{name}|{view}"] = {
                "oos": mo,
                "in_sample": mi,
                "gap_in_minus_oos": {k: mi[k] - mo[k] for k in ("accuracy", "brier", "log_loss")},
                "reliability_oos": reliability(r["oos"], y),
                "fold_coefficients": r["folds"],
                "fold_summary": fold_summary(r["folds"]),
                "full_fit": r["full_fit"],
            }
        new = probs[(view, "per_flag")]["oos"]
        old = probs[(view, "pooled")]["oos"]
        result["paired"][view] = bootstrap(data, new, old, rng)
        differ = (new >= 0.5) != (old >= 0.5)
        result["decisive"][view] = {
            "games_different_side": int(differ.sum()),
            "per_flag_wins": int(((new >= 0.5) == (y == 1))[differ].sum()),
            "pooled_wins": int(((old >= 0.5) == (y == 1))[differ].sum()),
        }
    result["looks"] = {
        "count": len(VARIANTS) * len(VIEWS) * 4 + len(VIEWS) * 2,
        "family": "2 variants x 2 views x 4 scoring metrics (accuracy, Brier, log loss, "
        "reliability) plus 2 views x 2 paired deltas (accuracy, Brier); coefficient and "
        "decisive reports are descriptive",
    }
    try:
        result["week6_tuesday_view"] = week6(data)
    except Exception as exc:
        result["week6_tuesday_view"] = f"skipped: {type(exc).__name__}: {exc}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    lines = []
    for key, v in result["variants"].items():
        o, i = v["oos"], v["in_sample"]
        g = v["gap_in_minus_oos"]
        lines.append(
            f"{key}: OOS acc {o['accuracy']:.4f} {o['record']} brier {o['brier']:.5f} "
            f"ll {o['log_loss']:.5f} | IS acc {i['accuracy']:.4f} brier {i['brier']:.5f} "
            f"ll {i['log_loss']:.5f} | gap acc {g['accuracy']:.4f} "
            f"brier {g['brier']:.5f} ll {g['log_loss']:.5f}"
        )
        for row in v["reliability_oos"]:
            lines.append(
                f"   rel n={row['games']} pred {row['mean_predicted']:.3f} "
                f"obs {row['observed_home_cover']:.3f}"
            )
        for k, s in v["fold_summary"].items():
            lines.append(f"   coef {k}: mean {s['mean']:.4f} min {s['min']:.4f} max {s['max']:.4f}")
    lines.append(json.dumps(result["paired"], indent=1))
    lines.append(json.dumps(result["decisive"], indent=1))
    lines.append(json.dumps(result["looks"], indent=1))
    w6 = result["week6_tuesday_view"]
    if isinstance(w6, str):
        lines.append(w6)
    else:
        for r in w6:
            if r["game"] in ("HOU@JAX", "LAC@KC", "CAR@PHI") or r["pooled_pick"] != r["per_flag_pick"]:
                lines.append(json.dumps(r))
    text = "\n".join(lines)
    (OUT / "results.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
