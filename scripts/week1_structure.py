from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from lead98_price_skew import design, fit, predict
from per_flag_weights_eval import SEED

from nfl_ats.pick_probability import (
    FLAG_SUM_COLUMN,
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
    active_market_move_feature_version,
)
from nfl_ats.pick_probability_fit import build_fit_population
from nfl_ats.pool_hook import crossed_push_mass, tuesday_market_lines

OUT = ROOT / "artifacts/week1_structure"
HARNESS = ROOT / "tests/scratch/fixpicks/root_m2_games.csv"
COEFFS = ROOT / "artifacts/pick_probability/20261111T011859Z/coefficients.json"
FAMILY = "week1_model_weight"
TERM = "model_w1"
BASE = ("model_logit", FLAG_SUM_COLUMN, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN)
ARM = (*BASE, TERM)
PUSH_LABEL = 0.5
DRAWS = 2000


def add_term(frame):
    frame = frame.copy()
    frame[TERM] = frame["model_logit"] * frame["week"].eq(1).astype(float)
    return frame


def loso(frame, features, fit_mask):
    oos = np.full(len(frame), np.nan)
    folds = {}
    for held in sorted(frame["season"].unique()):
        tr = (frame["season"].ne(held) & fit_mask).to_numpy()
        te = frame["season"].eq(held).to_numpy()
        b, m, s = fit(frame.loc[tr], features)
        oos[te] = predict(frame.loc[te], features, b, m, s)
        folds[str(held)] = {n: float(b[i + 1] / s[n]) for i, n in enumerate(features)}
    return oos, folds


def scores(p, y, push):
    q = np.clip(p, 1e-9, 1 - 1e-9)
    decided = ~push
    right = ((q >= 0.5) == (y == 1)) & decided
    wins = int(right.sum())
    games = int(decided.sum())
    return {
        "record": f"{wins}-{games - wins}",
        "accuracy": wins / games if games else float("nan"),
        "log_loss": float(-np.mean(y * np.log(q) + (1 - y) * np.log1p(-q))),
        "brier": float(np.mean((q - y) ** 2)),
    }


def changed(pa, pb, y, push):
    ch = ((pa >= 0.5) != (pb >= 0.5)) & ~push
    arm_right = int((((pa >= 0.5) == (y == 1)) & ch).sum())
    return {"changed": int(ch.sum()), "arm_wins": arm_right, "base_wins": int(ch.sum()) - arm_right}


def block_boot(frame, pa, pb, y, push, mask, rng):
    blocks = (frame["season"].astype(str) + "_" + frame["week"].astype(str)).to_numpy()
    keys = np.unique(blocks[mask])
    idx = {k: np.flatnonzero((blocks == k) & mask) for k in keys}
    qa, qb = np.clip(pa, 1e-9, 1 - 1e-9), np.clip(pb, 1e-9, 1 - 1e-9)
    ra = ((qa >= 0.5) == (y == 1)).astype(float)
    rb = ((qb >= 0.5) == (y == 1)).astype(float)
    la = -(y * np.log(qa) + (1 - y) * np.log1p(-qa))
    lb = -(y * np.log(qb) + (1 - y) * np.log1p(-qb))
    ba, bb = (qa - y) ** 2, (qb - y) ** 2
    out = {"accuracy_points": [], "log_loss_improvement": [], "brier_improvement": []}
    for _ in range(DRAWS):
        rows = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        dec = rows[~push[rows]]
        out["accuracy_points"].append(100.0 * (ra[dec].mean() - rb[dec].mean()))
        out["log_loss_improvement"].append(lb[rows].mean() - la[rows].mean())
        out["brier_improvement"].append(bb[rows].mean() - ba[rows].mean())
    dec = mask & ~push
    point = {
        "accuracy_points": 100.0 * (ra[dec].mean() - rb[dec].mean()),
        "log_loss_improvement": float(lb[mask].mean() - la[mask].mean()),
        "brier_improvement": float(bb[mask].mean() - ba[mask].mean()),
    }
    res = {}
    for k, v in out.items():
        v = np.array(v)
        res[k] = {
            "effect": float(point[k]),
            "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
            "probability_positive": float((v > 0).mean()),
        }
    return res


def cell(scope, unit, r, games, start, end, grade):
    return {
        "name": f"week1_model_weight_{scope}_{unit}",
        "description": (
            f"Week 1 model weight: model_logit x 1[week 1] added as one fitted term, {scope}, "
            f"{grade}; LOSO by season, week-block bootstrap, improvement positive; declared "
            "2026-11-16T21:19:52Z before fitting."
        ),
        "effect": r["effect"],
        "effect_units": unit,
        "interval_low": r["ci95"][0],
        "interval_high": r["ci95"][1],
        "probability_positive": r["probability_positive"],
        "season_start": start,
        "season_end": end,
        "sample_games": games,
    }


def opener(result, cells, rng):
    version = active_market_move_feature_version(ROOT / "artifacts")
    pop, _ = build_fit_population(
        ROOT / "artifacts", ROOT / "data", market_move_feature_version=version, keep_pushes=True
    )
    pop = add_term(pop[pop["season"].between(2020, 2025)].reset_index(drop=True))
    push = pop["margin_vs_open"].eq(0.0).to_numpy()
    y = np.where(push, PUSH_LABEL, pop["home_covered"].astype(float))
    fit_mask = pd.Series(~push)
    pb, fb = loso(pop, BASE, fit_mask)
    pa, fa = loso(pop, ARM, fit_mask)
    w1 = pop["week"].eq(1).to_numpy()
    allg = np.ones(len(pop), dtype=bool)
    tr = pop.loc[~push]
    bi, mi, si = fit(tr, BASE)
    ai, am, asd = fit(tr, ARM)
    ins_b, ins_a = predict(pop, BASE, bi, mi, si), predict(pop, ARM, ai, am, asd)
    result["opener"] = {
        "games": len(pop),
        "pushes": int(push.sum()),
        "week1_games": int(w1.sum()),
        "fold_coefficients_arm": fa,
        "fold_week1_model_weight": {
            k: v["model_logit"] + v[TERM] for k, v in fa.items()
        },
        "fold_coefficients_base": fb,
    }
    for scope, mask in (("all_games", allg), ("week1", w1), ("weeks2_18", ~w1)):
        e = {
            "base_oos": scores(pb[mask], y[mask], push[mask]),
            "arm_oos": scores(pa[mask], y[mask], push[mask]),
            "base_in": scores(ins_b[mask], y[mask], push[mask]),
            "arm_in": scores(ins_a[mask], y[mask], push[mask]),
            "changed": changed(pa[mask], pb[mask], y[mask], push[mask]),
            "boot": block_boot(pop, pa, pb, y, push, mask, rng),
        }
        e["accuracy_gap_in_minus_oos_points"] = 100.0 * (
            e["arm_in"]["accuracy"] - e["arm_oos"]["accuracy"]
        )
        result["opener"][scope] = e
    b = result["opener"]["all_games"]["boot"]
    for unit in ("accuracy_points", "log_loss_improvement", "brier_improvement"):
        cells.append(cell("2020_25_opener_all", unit, b[unit], len(pop), 2020, 2025, "opener"))
    cells.append(
        cell(
            "2020_25_opener_week1",
            "accuracy_points",
            result["opener"]["week1"]["boot"]["accuracy_points"],
            int(w1.sum()),
            2020,
            2025,
            "opener",
        )
    )
    return pop, push


def close_screen(result, cells, rng):
    runs = []
    for m in sorted((ROOT / "artifacts/margins").glob("*/metadata.json")):
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(meta.get("start_season")) == "2011" and meta.get("feature_profile") == "base":
            runs.append(m.parent)
    if not runs:
        result["close_screen"] = "no 2011-start base backtest on disk"
        return
    run = runs[-1]
    d = pd.read_parquet(
        run / "predictions.parquet",
        columns=["game_id", "season", "week", "method", "ats_margin", "home_cover_probability"],
    )
    d = d[d["method"].eq("market_residual") & d["season"].between(2011, 2025)]
    d = d.dropna(subset=["ats_margin", "home_cover_probability"]).reset_index(drop=True)
    p = d["home_cover_probability"].clip(1e-6, 1 - 1e-6)
    d["model_logit"] = np.log(p / (1 - p))
    d = add_term(d)
    push = d["ats_margin"].eq(0).to_numpy()
    d["home_covered"] = (d["ats_margin"] > 0).astype(float)
    y = np.where(push, PUSH_LABEL, d["home_covered"])
    fm = pd.Series(~push)
    pb, _ = loso(d, ("model_logit",), fm)
    pa, fa = loso(d, ("model_logit", TERM), fm)
    w1 = d["week"].eq(1).to_numpy()
    allg = np.ones(len(d), dtype=bool)
    result["close_screen"] = {
        "artifact": str(run.relative_to(ROOT)),
        "games": len(d),
        "week1_games": int(w1.sum()),
        "fold_week1_model_weight": {k: v["model_logit"] + v[TERM] for k, v in fa.items()},
        "fold_model_weight": {k: v["model_logit"] for k, v in fa.items()},
    }
    for scope, mask in (("all_games", allg), ("week1", w1)):
        result["close_screen"][scope] = {
            "base_oos": scores(pb[mask], y[mask], push[mask]),
            "arm_oos": scores(pa[mask], y[mask], push[mask]),
            "changed": changed(pa[mask], pb[mask], y[mask], push[mask]),
            "boot": block_boot(d, pa, pb, y, push, mask, rng),
        }
    b = result["close_screen"]["all_games"]["boot"]
    for unit in ("accuracy_points", "log_loss_improvement"):
        cells.append(cell("2011_25_close_all", unit, b[unit], len(d), 2011, 2025, "close"))


def forward(result, cells, pop, push, rng):
    co = json.loads(COEFFS.read_text(encoding="utf-8"))
    c = co["coefficients"]
    pm = {float(a): float(m) for a, m in co["hook_push_mass"]}
    d = pd.read_csv(HARNESS)
    d = d[d["week"].eq(1)].reset_index(drop=True)
    anchor = tuesday_market_lines(ROOT / "data", 2026, 1, d.game_id.tolist())
    d["anchor"] = d.game_id.map(anchor)
    hooks = []
    for r in d.itertuples():
        m = 0.0 if pd.isna(r.anchor) else crossed_push_mass(float(r.anchor), float(r.pool_home_spread), pm)
        x = 0.0 if m <= 0 else float(np.sign(r.pool_home_spread - r.anchor))
        hooks.append(0.0 if x == 0 else np.log((0.5 * (1 - x * m)) / (1 - 0.5 * (1 - x * m))))
    d["hook"] = hooks
    s = d["raw_model_home_prob_final"].clip(1e-6, 1 - 1e-6)
    d["model_logit"] = np.log(s / (1 - s))
    av = d["served_move_available"].fillna(0.0).astype(float)
    d[MOVE_AVAILABLE_COLUMN] = av
    d[MOVE_COLUMN] = d["served_move"].where(av > 0)
    d[FLAG_SUM_COLUMN] = np.nan
    d["season"] = 2026
    d = add_term(d)
    y = d["hc"].to_numpy(float)
    z = (
        c["intercept"]
        + c["model_logit"] * d["model_logit"]
        + c["move_toward_home"] * d["served_move"].fillna(0.0)
        + c["move_available"] * av
        + c["pool_hook"] * d["hook"]
    )
    current = (1 / (1 + np.exp(-z))).to_numpy()
    tr = pop.loc[~push]
    P = {"current": current}
    for name, f in (("frozen_base", BASE), ("frozen_arm", ARM)):
        b, m, sd = fit(tr, f)
        P[name] = 1 / (1 + np.exp(-(design(d, f, m, sd) @ b + c["pool_hook"] * d["hook"].to_numpy())))
    np_ = np.zeros(len(d), dtype=bool)
    result["forward_2026_week1"] = {
        "games": len(d),
        **{k: scores(v, y, np_) for k, v in P.items()},
        "changed_vs_current": changed(P["frozen_arm"], current, y, np_),
        "changed_vs_frozen_base": changed(P["frozen_arm"], P["frozen_base"], y, np_),
        "changed_games": d.loc[(P["frozen_arm"] >= 0.5) != (current >= 0.5), ["game_id", "pool_home_spread", "hc"]]
        .assign(current=current[(P["frozen_arm"] >= 0.5) != (current >= 0.5)])
        .to_dict("records"),
    }
    frame = d[["season", "week"]]
    bt = block_boot(frame, P["frozen_arm"], current, y, np_, np.ones(len(d), dtype=bool), rng)
    result["forward_2026_week1"]["boot_vs_current"] = bt
    cells.append(cell("2026_week1_forward", "accuracy_points", bt["accuracy_points"], len(d), 2026, 2026, "frozen 2020-25 fit, pool-line harness"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    result, cells = {}, []
    pop, push = opener(result, cells, rng)
    close_screen(result, cells, rng)
    forward(result, cells, pop, push, rng)
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    batch = {
        "family": FAMILY,
        "league": "nfl",
        "category": "schedule",
        "classification": "unresolved_below_power",
        "source": "artifacts/week1_structure/results.json",
        "cells": cells,
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
