from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lead98_price_skew import BASE, design, fit, loso, metrics, predict, reliability, summary
from mod57_u3_deadline_fair_line import block_bootstrap, decisive
from mod57_u4_quiet_game_slopes import ARTIFACT, to_improvement
from per_flag_weights_eval import SEED

from nfl_ats.pick_probability import MOVE_AVAILABLE_COLUMN, MOVE_COLUMN
from nfl_ats.pool_hook import crossed_push_mass, tuesday_market_lines

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/mod57_p2t"
FAMILY = "mod57_quiet_game_slopes"
PHASES = ("w1", "w2_6", "w7_12", "w13_18")
PHASE_UPPER = (1, 6, 12, 18)
MODEL_COLS = [f"model_{p}" for p in PHASES]
MOVE_COLS = [f"move_{p}" for p in PHASES]
PERMUTATIONS = 200
HARNESS = ROOT / "tests/scratch/fixpicks/root_m2_games.csv"
COEFFS = ROOT / "artifacts/pick_probability/20261111T011859Z/coefficients.json"


def phase_index(week):
    w = np.asarray(week)
    return np.where(w <= 1, 0, np.where(w <= 6, 1, np.where(w <= 12, 2, 3)))


def add_phase_columns(data, index):
    data = data.copy()
    for i, p in enumerate(PHASES):
        inside = index == i
        data[f"model_{p}"] = data["model_logit"].where(inside)
        data[f"move_{p}"] = data[MOVE_COLUMN].where(inside)
    return data


def variants():
    return {
        "four_term": BASE,
        "arm_m4_model_slope_by_phase": (*MODEL_COLS, *BASE[1:]),
        "arm_v4_move_slope_by_phase": ("model_logit", BASE[1], *MOVE_COLS, MOVE_AVAILABLE_COLUMN),
        "arm_b_both_by_phase": (*MODEL_COLS, BASE[1], *MOVE_COLS, MOVE_AVAILABLE_COLUMN),
        "model_only": ("model_logit",),
        "intercept_only": (),
    }


ARMS = ("arm_m4_model_slope_by_phase", "arm_v4_move_slope_by_phase", "arm_b_both_by_phase")


def permuted_index(data, rng):
    index = np.empty(len(data), dtype=int)
    for season in data["season"].unique():
        rows = (data["season"] == season).to_numpy()
        weeks = np.sort(data.loc[rows, "week"].unique())
        labels = phase_index(weeks)
        shuffled = rng.permutation(labels)
        lookup = dict(zip(weeks.tolist(), shuffled.tolist(), strict=True))
        index[rows] = [lookup[w] for w in data.loc[rows, "week"].tolist()]
    return index


def acc_ll(p, y):
    q = np.clip(p, 1e-9, 1 - 1e-9)
    acc = float(((q >= 0.5) == (y == 1)).mean())
    ll = float(-np.mean(y * np.log(q) + (1 - y) * np.log1p(-q)))
    return acc, ll


def cell(arm, scope, metric, units, eff, lo, hi, pp, games, start, end):
    return {
        "name": f"mod57_p2t_{arm}_{scope}_{metric}",
        "description": (
            f"MOD-57 P2-T: {arm} versus the four-term served probability, {scope}, "
            "week-block bootstrap; improvement positive; phase boundaries chosen after "
            "descriptive reads of the same games."
        ),
        "effect": eff,
        "effect_units": units,
        "interval_low": lo,
        "interval_high": hi,
        "probability_positive": pp,
        "season_start": start,
        "season_end": end,
        "sample_games": games,
    }


def paired_cells(arm, scope, boot, games, start, end):
    cells = []
    for metric, units in (
        ("accuracy_points", "accuracy_points"),
        ("log_loss", "log_loss_improvement"),
        ("brier", "brier_improvement"),
    ):
        eff, lo, hi, pp = to_improvement(boot, metric)
        cells.append(cell(arm, scope, metric, units, eff, lo, hi, pp, games, start, end))
    return cells


def backtest(result, cells):
    data = pd.read_parquet(ARTIFACT)
    data = data.loc[data["season"].between(2020, 2025)].reset_index(drop=True)
    index = phase_index(data["week"].to_numpy())
    data = add_phase_columns(data, index)
    y = data["home_covered"].to_numpy(float)
    feats = variants()
    probs, folds, insample = {}, {}, {}
    for name, f in feats.items():
        oos, fo = loso(data, f)
        probs[name] = oos
        folds[name] = fo
        b, m, s = fit(data, f)
        insample[name] = predict(data, f, b, m, s)
    result["games"] = len(data)
    result["phase_games"] = {p: int((index == i).sum()) for i, p in enumerate(PHASES)}
    result["fold_coefficients"] = {n: folds[n] for n in ("four_term", *ARMS)}
    result["fold_summary"] = {n: summary(folds[n]) for n in result["fold_coefficients"]}
    rng = np.random.default_rng(SEED)
    scopes = {"all_games": np.ones(len(data), dtype=bool)}
    for i, p in enumerate(PHASES):
        scopes[p] = index == i
    result["scopes"] = {}
    for scope, mask in scopes.items():
        sub = data.loc[mask].reset_index(drop=True)
        ys = y[mask]
        entry = {"games": int(mask.sum()), "variants": {}, "decisive": {}, "paired": {}}
        for name in probs:
            entry["variants"][name] = {"oos": metrics(probs[name][mask], ys)}
            if name in insample:
                entry["variants"][name]["in_sample"] = metrics(insample[name][mask], ys)
                entry["variants"][name]["accuracy_gap_in_minus_oos_points"] = 100.0 * (
                    entry["variants"][name]["in_sample"]["accuracy"]
                    - entry["variants"][name]["oos"]["accuracy"]
                )
        for arm in ARMS:
            entry["decisive"][arm] = decisive(probs[arm][mask], probs["four_term"][mask], ys)
            boot = block_bootstrap(sub, probs[arm][mask], probs["four_term"][mask], rng)
            entry["paired"][arm] = boot
            if scope == "all_games":
                cells.extend(
                    paired_cells(arm, "2020_25_all_games", boot, int(mask.sum()), 2020, 2025)
                )
        if scope in ("all_games", "w13_18"):
            entry["reliability"] = {
                n: reliability(probs[n][mask], ys) for n in ("four_term", *ARMS, "model_only")
            }
        result["scopes"][scope] = entry
    base_acc, base_ll = acc_ll(probs["four_term"], y)
    control = {arm: {"acc_points": [], "ll_improvement": []} for arm in ARMS}
    for _ in range(PERMUTATIONS):
        shuffled = permuted_index(data, rng)
        pdata = add_phase_columns(data, shuffled)
        for arm in ARMS:
            oos, _ = loso(pdata, feats[arm])
            a, ll = acc_ll(oos, y)
            control[arm]["acc_points"].append(100.0 * (a - base_acc))
            control[arm]["ll_improvement"].append(base_ll - ll)
    result["permuted_phase_control"] = {"permutations": PERMUTATIONS}
    for arm in ARMS:
        a, ll = acc_ll(probs[arm], y)
        actual_acc, actual_ll = 100.0 * (a - base_acc), base_ll - ll
        ca = np.array(control[arm]["acc_points"])
        cl = np.array(control[arm]["ll_improvement"])
        result["permuted_phase_control"][arm] = {
            "actual_acc_points": actual_acc,
            "control_acc_mean": float(ca.mean()),
            "control_acc_p05_p95": [float(np.quantile(ca, 0.05)), float(np.quantile(ca, 0.95))],
            "share_control_acc_at_or_above_actual": float((ca >= actual_acc).mean()),
            "actual_ll_improvement": actual_ll,
            "control_ll_mean": float(cl.mean()),
            "control_ll_p05_p95": [float(np.quantile(cl, 0.05)), float(np.quantile(cl, 0.95))],
            "share_control_ll_at_or_above_actual": float((cl >= actual_ll).mean()),
        }
    return data, feats


def forward(result, cells, data, feats):
    co = json.loads(COEFFS.read_text(encoding="utf-8"))
    c = co["coefficients"]
    push = {float(a): float(m) for a, m in co["hook_push_mass"]}
    d = pd.read_csv(HARNESS)
    anchor = {}
    for week, g in d.groupby("week"):
        anchor.update(tuesday_market_lines(ROOT / "data", 2026, int(week), g.game_id.tolist()))
    d["anchor"] = d.game_id.map(anchor)
    masses, dirs = [], []
    for r in d.itertuples():
        m = (
            0.0
            if pd.isna(r.anchor)
            else crossed_push_mass(float(r.anchor), float(r.pool_home_spread), push)
        )
        masses.append(m)
        dirs.append(0.0 if m <= 0 else float(np.sign(r.pool_home_spread - r.anchor)))
    d["mass"], d["dir"] = masses, dirs
    d["hook"] = [
        0.0 if x == 0 else np.log((0.5 * (1 - x * m)) / (1 - 0.5 * (1 - x * m)))
        for x, m in zip(d["dir"], d["mass"], strict=True)
    ]
    s = d["raw_model_home_prob_final"].clip(1e-6, 1 - 1e-6)
    d["model_logit"] = np.log(s / (1 - s))
    y = d["hc"].to_numpy(float)
    mv = d["served_move"].fillna(0.0)
    av = d["served_move_available"].fillna(0.0).astype(float)
    z = (
        c["intercept"]
        + c["model_logit"] * d["model_logit"]
        + c["move_toward_home"] * mv
        + c["move_available"] * av
        + c["pool_hook"] * d["hook"]
    )
    current = (1 / (1 + np.exp(-z))).to_numpy()
    d[MOVE_AVAILABLE_COLUMN] = av
    d[MOVE_COLUMN] = d["served_move"].where(av > 0)
    d["season"] = 2026
    d["home_covered"] = y
    d["composition_flag_sum"] = np.nan
    index = phase_index(d["week"].to_numpy())
    d = add_phase_columns(d, index)
    P = {"current": current}
    frozen = {}
    for name in ("four_term", *ARMS):
        b, m, sd = fit(data, feats[name])
        frozen[name] = {
            "features": list(feats[name]),
            "beta": [float(x) for x in b],
            "means": m,
            "stds": sd,
        }
        zz = design(d, feats[name], m, sd) @ b + c["pool_hook"] * d["hook"].to_numpy()
        P[name] = 1 / (1 + np.exp(-zz))
    result["forward"] = {"games": len(d), "frozen": frozen, "variants": {}, "phases": {}}
    for k, p in P.items():
        result["forward"]["variants"][k] = metrics(p, y)
    frame = d[["season", "week", "home_covered"]].copy()
    rng = np.random.default_rng(SEED)
    for arm in ARMS:
        entry = {"decisive": decisive(P[arm], current, y)}
        entry["paired_vs_current"] = block_bootstrap(frame, P[arm], current, rng)
        cells.extend(
            paired_cells(arm, "2026_wk1_9_forward", entry["paired_vs_current"], len(d), 2026, 2026)
        )
        entry["decisive_vs_frozen_four_term"] = decisive(P[arm], P["four_term"], y)
        entry["paired_vs_frozen_four_term"] = block_bootstrap(frame, P[arm], P["four_term"], rng)
        ch = (P[arm] >= 0.5) != (current >= 0.5)
        entry["changed_games"] = (
            d.loc[ch, ["game_id", "pool_home_spread", "week", "hc"]]
            .assign(current=current[ch], arm=P[arm][ch])
            .round(3)
            .to_dict("records")
        )
        result["forward"][arm] = entry
    for i, p in enumerate(PHASES):
        mask = index == i
        if not mask.any():
            continue
        result["forward"]["phases"][p] = {
            "games": int(mask.sum()),
            **{k: metrics(v[mask], y[mask]) for k, v in P.items()},
            "decisive": {a: decisive(P[a][mask], current[mask], y[mask]) for a in ARMS},
        }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    cells = []
    data, feats = backtest(result, cells)
    forward(result, cells, data, feats)
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    batch = {
        "family": FAMILY,
        "league": "nfl",
        "category": "market",
        "classification": "unresolved_below_power",
        "source": "artifacts/mod57_p2t/results.json",
        "cells": cells,
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print(json.dumps(result["fold_summary"], indent=1))
    print(json.dumps(result["permuted_phase_control"], indent=1))
    for scope, e in result["scopes"].items():
        print(scope, e["games"], {n: v["oos"]["record"] for n, v in e["variants"].items()})
        print(" decisive", e["decisive"])
        for arm in ARMS:
            b = e["paired"][arm]
            print(
                " ",
                arm,
                {
                    m: (round(b[m]["delta_mean"], 4), b[m]["ci95"], b[m]["probability_positive"])
                    for m in b
                },
            )
    print(json.dumps(result["forward"]["variants"], indent=1))
    for arm in ARMS:
        f = result["forward"][arm]
        print(arm, f["decisive"], f["changed_games"])
        print(
            " vs current",
            {
                m: (
                    round(f["paired_vs_current"][m]["delta_mean"], 4),
                    f["paired_vs_current"][m]["probability_positive"],
                )
                for m in f["paired_vs_current"]
            },
        )
    print(json.dumps(result["forward"]["phases"], indent=1))


if __name__ == "__main__":
    main()
