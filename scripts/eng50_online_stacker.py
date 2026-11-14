import argparse
import json
from pathlib import Path

import forward_monitor
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEDGER = "artifacts/forward_ledger_core/ledger.parquet"
PENALTY_EXPONENTS = range(-2, 3)
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 50
NEWTON_STEPS = 50
CLIP = 1e-6


def _logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _fit(x, offset, y, penalty):
    beta = np.zeros(x.shape[1])
    for _ in range(NEWTON_STEPS):
        p = _sigmoid(offset + x @ beta)
        grad = x.T @ (p - y) + penalty * beta
        hess = (x * (p * (1 - p))[:, None]).T @ x + penalty * np.eye(len(beta))
        step = np.linalg.solve(hess, grad)
        beta = beta - step
        if np.max(np.abs(step)) < CLIP:
            break
    return beta


def _loss(p, y):
    p = np.clip(p, CLIP, 1 - CLIP)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _choose_penalty(x, offset, y, groups):
    units = np.unique(groups) if len(np.unique(groups)) > 1 else np.arange(len(y))
    keys = groups if len(np.unique(groups)) > 1 else np.arange(len(y))
    best = None
    for exponent in PENALTY_EXPONENTS:
        penalty = 10.0**exponent
        total = 0.0
        for unit in units:
            held = keys == unit
            beta = _fit(x[~held], offset[~held], y[~held], penalty)
            total += _loss(_sigmoid(offset[held] + x[held] @ beta), y[held]).sum()
        if best is None or total <= best[0]:
            best = (total, penalty)
    return best[1]


def _terms(matched):
    return [
        c[: -len("__p")]
        for c in matched.columns
        if c.endswith("__p")
        and not c.startswith("served")
        and c[: -len("__p")] not in forward_monitor.MONITOR_ONLY
        and matched[c].notna().all()
        and c[: -len("__p")] + "__written_at" in matched
        and (matched[c[: -len("__p")] + "__written_at"] < matched["deadline_utc"]).all()
    ]


def _interval(diff):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(diff, size=(BOOTSTRAP_DRAWS, len(diff)), replace=True).mean(axis=1)
    return (
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
        float((draws > 0).mean()),
    )


def main():
    parser = argparse.ArgumentParser(
        description="ENG-50 dry replay of an online ridge-logistic stacker over forward candidates"
    )
    parser.add_argument("--out", default="artifacts/eng50_online_stacker")
    args = parser.parse_args()
    ledger = pd.read_parquet(ROOT / LEDGER)
    matched = ledger.loc[ledger["matched"] & ledger["status"].eq("graded")].copy()
    terms = _terms(matched)
    offset = _logit(matched["served_p"].to_numpy())
    x = np.column_stack([_logit(matched[f"{t}__p"].to_numpy()) - offset for t in terms])
    y = matched["home_cover"].to_numpy(dtype=float)
    weeks = matched["week"].to_numpy()
    predicted = []
    path = []
    for week in sorted(set(weeks))[1:]:
        train = weeks < week
        test = weeks == week
        penalty = _choose_penalty(x[train], offset[train], y[train], weeks[train])
        beta = _fit(x[train], offset[train], y[train], penalty)
        p = _sigmoid(offset[test] + x[test] @ beta)
        frame = matched.loc[test, ["game_id", "week", "served_p", "home_cover"]].copy()
        frame["stack_p"] = p
        predicted.append(frame)
        path.append(
            {
                "week": int(week),
                "train_games": int(train.sum()),
                "penalty": penalty,
                "penalty_cv": "leave_one_week_out"
                if len(set(weeks[train])) > 1
                else "leave_one_game_out",
                "coefficients": dict(zip(terms, map(float, beta), strict=True)),
            }
        )
    out = pd.concat(predicted)
    home = out["home_cover"].eq(1.0)
    out["served_side_home"] = out["served_p"].ge(0.5)
    out["stack_side_home"] = out["stack_p"].ge(0.5)
    out["decisive"] = out["served_side_home"] != out["stack_side_home"]
    out["served_loss"] = _loss(out["served_p"].to_numpy(), home.to_numpy(dtype=float))
    out["stack_loss"] = _loss(out["stack_p"].to_numpy(), home.to_numpy(dtype=float))
    out["served_brier"] = (out["served_p"] - home.astype(float)) ** 2
    out["stack_brier"] = (out["stack_p"] - home.astype(float)) ** 2
    out["served_correct"] = out["served_side_home"] == home
    out["stack_correct"] = out["stack_side_home"] == home

    def summarize(frame):
        diff = (frame["served_loss"] - frame["stack_loss"]).to_numpy()
        low, high, p_pos = _interval(diff)
        dec = frame.loc[frame["decisive"]]
        return {
            "games": len(frame),
            "decisive_games": len(dec),
            "decisive_stack_wins": int(dec["stack_correct"].sum()),
            "decisive_served_wins": int(dec["served_correct"].sum()),
            "served_log_loss": float(frame["served_loss"].mean()),
            "stack_log_loss": float(frame["stack_loss"].mean()),
            "log_loss_improvement": float(diff.mean()),
            "interval_low": low,
            "interval_high": high,
            "probability_positive": p_pos,
            "served_brier": float(frame["served_brier"].mean()),
            "stack_brier": float(frame["stack_brier"].mean()),
            "served_accuracy": float(frame["served_correct"].mean()),
            "stack_accuracy": float(frame["stack_correct"].mean()),
        }

    result = {
        "terms": terms,
        "per_week": {int(w): summarize(g) for w, g in out.groupby("week")},
        "cumulative": summarize(out),
        "coefficient_path": path,
        "looks": {
            "prediction_weeks": len(path),
            "penalty_grid": len(list(PENALTY_EXPONENTS)),
            "terms": len(terms),
            "cumulative_cells": 1,
        },
    }
    target = ROOT / args.out
    target.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target / "predictions.parquet", index=False)
    (target / "summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
