import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import sim04_loso as base  # noqa: E402
import sim08_evaluate as ev  # noqa: E402

SEED = 20261111
KEYS = (3, 7, 10, 14, 17)
SHAPE_BANDS = (
    ("0_2.5", 0.0, 2.5),
    ("3_3.5", 3.0, 3.5),
    ("4_6.5", 4.0, 6.5),
    ("7_9.5", 7.0, 9.5),
    ("10plus", 10.0, 1e9),
)
OUT = REPO / "artifacts" / "sim08_amendment6"
FIRST_SEASON = 2009
ESS_FLOOR = 0.25 / 0.026**2
LINE_GRID = [x / 2 for x in range(0, 21)]


def real_games():
    g = pd.read_parquet(base.GAME_FEATURES_PATH)
    g = g.loc[g["game_type"] == "REG"].copy()
    g = g.loc[g["season"] >= FIRST_SEASON]
    g = g.dropna(subset=["spread_line", "home_score", "away_score"])
    g["result"] = (g["home_score"] - g["away_score"]).astype(float)
    g["line"] = g["spread_line"].astype(float)
    return g.reset_index(drop=True)


def stage_one():
    g = real_games()
    inv, nb = ev.blocks_of(g)
    rng = np.random.default_rng(SEED)
    sign = np.where(g["line"] >= 0, 1.0, -1.0)
    absl = np.abs(g["line"].to_numpy())
    fav = g["result"].to_numpy() * sign
    diff = fav - absl
    cover = (diff > 0).astype(float)
    push = (np.abs(diff) < 1e-9).astype(float)
    nonpush = push < 0.5
    rows = []
    for level in LINE_GRID:
        at = np.abs(absl - level) < 1e-9
        if at.sum() < 2:
            continue
        whole = abs(level - round(level)) < 1e-9
        row = {
            "line": level,
            "n": int(at.sum()),
            "n_nonpush": int((at & nonpush).sum()),
            "mean_margin_minus_line": ev.boot_mean(diff, inv, nb, rng, at),
            "cover_share_decisive": ev.boot_mean(cover, inv, nb, rng, at & nonpush),
        }
        if whole:
            row["push_share"] = ev.boot_mean(push, inv, nb, rng, at)
        rows.append(row)
    n_all = len(g)
    return {
        "n_games": n_all,
        "n_blocks": nb,
        "seasons": [int(g["season"].min()), int(g["season"].max())],
        "rows": rows,
    }


def balance_weights(margins, line):
    diff = margins - line[:, None]
    cov = diff > base.ATOM_TOLERANCE
    los = diff < -base.ATOM_TOLERANCE
    lo = np.full(len(margins), -2.0)
    hi = np.full(len(margins), 2.0)
    for _ in range(80):
        theta = (lo + hi) / 2.0
        z = theta[:, None] * diff
        w = np.exp(z - z.max(1, keepdims=True))
        decided = (w * (cov | los)).sum(1)
        share = (w * cov).sum(1) / np.clip(decided, 1e-300, None)
        below = share < 0.5
        lo = np.where(below, theta, lo)
        hi = np.where(below, hi, theta)
    theta = (lo + hi) / 2.0
    z = theta[:, None] * diff
    w = np.exp(z - z.max(1, keepdims=True))
    w = w / w.sum(1, keepdims=True)
    ess = 1.0 / (w**2).sum(1)
    share = (w * cov).sum(1) / np.clip((w * (cov | los)).sum(1), 1e-300, None)
    return w * margins.shape[1], ess, share, theta


def wcounts(margins, w, line):
    diff = margins - line[:, None]
    c = (w * (diff > base.ATOM_TOLERANCE)).sum(1)
    lo = (w * (diff < -base.ATOM_TOLERANCE)).sum(1)
    return c, margins.shape[1] - c - lo, lo


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def row_loss(p, y):
    q = np.clip(np.where(y > 0.5, p, 1 - p), base.EPS, 1 - base.EPS)
    return -np.log(q)


def stage_two():
    from nfl_ats import pool_hook as ph
    from nfl_ats.schedule_flag_features import default_opener_lines

    frame = base.load_games()
    lines = default_opener_lines(frame[["game_id", "season", "week"]])
    frame = frame.merge(lines[["game_id", "tue_open_total_line"]], on="game_id", how="left")
    g = pd.read_parquet(base.GAME_FEATURES_PATH)
    g = g.loc[g["game_type"] == "REG", ["game_id", "home_score", "away_score"]]
    frame = frame.merge(g, on="game_id", how="left")
    outcome = frame["outcome"].to_numpy()
    seasons = frame["season"].to_numpy()
    line = frame["line"].to_numpy(float)
    result = frame["result"].to_numpy(float)
    inv, nb = ev.blocks_of(frame)
    rng = np.random.default_rng(SEED)
    cov_s, push_s, loss_s = base.baseline_three_way(frame)
    S = ev.load_arm("main", "primary", frame)
    M = S["margins"]
    n = S["draws"]
    w, ess, share, theta = balance_weights(M, line)
    np.savez_compressed(
        OUT / "balance_weights.npz",
        game_id=frame["game_id"].to_numpy().astype(str),
        weights=w.astype(np.float32),
        ess=ess,
        theta=theta,
    )
    cc, pp, ll = wcounts(M, w, line)
    sim3 = ((cc + 0.5) / (n + 1.5), (pp + 0.5) / (n + 1.5), (ll + 0.5) / (n + 1.5))
    push_sim = sim3[1]
    real_push = (outcome == "push").astype(float)
    whole = np.abs(line - np.rint(line)) < 1e-9
    table_push = np.full(len(frame), np.nan)
    for s_ in base.GRADED_SEASONS:
        tab = ph.push_mass_table(frame.loc[seasons != s_])
        for i in np.flatnonzero((seasons == s_) & whole):
            table_push[i] = tab.get(float(line[i]), np.nan)
    wmask = whole & np.isfinite(table_push)
    tp = np.clip(np.nan_to_num(table_push, nan=0.0), 1e-6, 1 - 1e-6)
    table3 = ((1 - tp) / 2, tp, (1 - tp) / 2)
    ll_served = ev.score((cov_s, push_s, loss_s), outcome)[0]
    ll_sim = ev.score(sim3, outcome)[0]
    ll_table = ev.score(table3, outcome)[0]
    a3 = whole & (np.abs(line) == 3)
    a7 = whole & (np.abs(line) == 7)
    res = {
        "n_games": len(frame),
        "n_blocks": nb,
        "draws": n,
        "ess": {
            "mean": float(ess.mean()),
            "min": float(ess.min()),
            "pcts_5_25_50_75_95": [float(v) for v in np.percentile(ess, [5, 25, 50, 75, 95])],
            "below_floor": int((ess < ESS_FLOOR).sum()),
        },
        "balance_exactness_max_abs_share_minus_half": float(np.abs(share - 0.5).max()),
        "population": {
            "whole_line_games": int(whole.sum()),
            "whole_line_with_table": int(wmask.sum()),
            "real_push_share_whole": float(real_push[whole].mean()),
            "served_push_mean_whole": float(push_s[whole].mean()),
            "sim_push_mean_whole": float(push_sim[whole].mean()),
            "table_push_mean_whole": float(np.nanmean(table_push[whole])),
            "real_push_share_at_3": float(real_push[a3].mean()),
            "real_push_share_at_7": float(real_push[a7].mean()),
            "sim_push_at_3": float(push_sim[a3].mean()),
            "sim_push_at_7": float(push_sim[a7].mean()),
            "served_push_at_3": float(push_s[a3].mean()),
            "served_push_at_7": float(push_s[a7].mean()),
            "n_at_3": int(a3.sum()),
            "n_at_7": int(a7.sum()),
        },
    }
    print(json.dumps(res["population"], indent=1))
    cells = {}

    def cell(name, d, mask=None):
        cells[name] = ev.boot_mean(d, inv, nb, rng, mask)

    cell("A6_W1", ll_served - ll_sim, wmask)
    cell("A6_W2", ll_table - ll_sim, wmask)
    cell("A6_W3", (push_s - real_push) ** 2 - (push_sim - real_push) ** 2, wmask)
    cell("A6_W4", (tp - real_push) ** 2 - (push_sim - real_push) ** 2, wmask)
    cell("A6_T1", ll_served - ll_sim)
    res["whole_line_log_loss"] = {
        "served": float(ll_served[wmask].mean()),
        "sim": float(ll_sim[wmask].mean()),
        "table": float(ll_table[wmask].mean()),
    }
    res["reliability_sim_push_whole"] = ev.reliability(push_sim[whole], real_push[whole], bins=5)

    h = pd.read_parquet(
        REPO / "artifacts" / "pick_probability" / "20261111T011859Z" / "hook_rows.parquet"
    )
    pos = {gid: i for i, gid in enumerate(frame["game_id"])}
    idx = np.array([pos[x] for x in h["game_id"]])
    pool = h["pool_line"].to_numpy(float)
    y = h["home_covered"].to_numpy(float)
    wh = w[idx]
    p_sim = ((wh * (M[idx] > pool[:, None])).sum(1) + 0.5) / (n + 1.0)
    p_served = h["two_stage_out_of_season_home_probability"].to_numpy(float)
    s1 = h["stage_one_logit_out_of_season"].to_numpy(float)
    hook = h["pool_hook"].to_numpy(float)
    p_emp = sigmoid(s1 + hook)
    direction = np.sign(pool - line[idx])
    p_sim_hook = sigmoid(s1 + ph.hook_value(direction, push_sim[idx]))
    inv_h = inv[idx]
    half = np.full(len(h), 0.5)

    def hcell(name, d):
        cells[name] = ev.boot_mean(d, inv_h, nb, rng, half)

    def acc(p):
        return ((p > 0.5) == (y > 0.5)).astype(float)

    hcell("A6_H1", row_loss(p_served, y) - row_loss(p_sim, y))
    hcell("A6_H2", row_loss(p_emp, y) - row_loss(p_sim, y))
    hcell("A6_H3", 100 * (acc(p_served) - acc(p_sim)))
    hcell("A6_H4", 100 * (acc(p_emp) - acc(p_sim)))
    hcell("A6_H5", row_loss(p_emp, y) - row_loss(p_sim_hook, y))
    res["hook"] = {
        "rows": len(h),
        "games": int(h["game_id"].nunique()),
        "log_loss": {
            "served_two_stage": float(row_loss(p_served, y).mean()),
            "empirical_crossed": float(row_loss(p_emp, y).mean()),
            "sim": float(row_loss(p_sim, y).mean()),
            "stage_one_plus_sim_hook": float(row_loss(p_sim_hook, y).mean()),
            "constant_half": float(np.log(2)),
        },
        "accuracy": {
            "served_two_stage": float(acc(p_served).mean()),
            "empirical_crossed": float(acc(p_emp).mean()),
            "sim": float(acc(p_sim).mean()),
            "stage_one_plus_sim_hook": float(acc(p_sim_hook).mean()),
        },
        "sim_mean_p_by_direction": {
            "pool_above_line": float(p_sim[direction > 0].mean()),
            "pool_below_line": float(p_sim[direction < 0].mean()),
        },
        "real_home_cover_by_direction": {
            "pool_above_line": float(y[direction > 0].mean()),
            "pool_below_line": float(y[direction < 0].mean()),
        },
        "decisive": {},
        "per_season_ll_served_minus_sim": {
            int(s_): float((row_loss(p_served, y) - row_loss(p_sim, y))[h["season"] == s_].mean())
            for s_ in base.GRADED_SEASONS
        },
    }
    for nm, other in (("served", p_served), ("empirical", p_emp)):
        d = (p_sim > 0.5) != (other > 0.5)
        res["hook"]["decisive"][f"sim_vs_{nm}"] = {
            "rows_differ": int(d.sum()),
            "sim_right": int(acc(p_sim)[d].sum()),
            "other_right": int(acc(other)[d].sum()),
        }
    print(json.dumps(res["hook"], indent=1))

    sgn = np.where(line >= 0, 1.0, -1.0)
    fav_sim = M * sgn[:, None]
    fav_real = result * sgn
    wt = w / n
    absl = np.abs(line)
    key_sim = {k: (wt * (np.abs(fav_sim - k) < 1e-9)).sum(1) for k in KEYS}
    key_real = {k: (np.abs(fav_real - k) < 1e-9).astype(float) for k in KEYS}
    key_raw = {k: (np.abs(fav_sim - k) < 1e-9).mean(1) for k in KEYS}
    res["key_mass"] = {
        str(k): {
            "balanced": float(key_sim[k].mean()),
            "raw": float(key_raw[k].mean()),
            "real": float(key_real[k].mean()),
        }
        for k in KEYS
    }
    for k in KEYS:
        cell(f"A6_K{k}", 100 * (key_sim[k] - key_real[k]))
    res["key_mass_by_band"] = {}
    for name, lo_, hi_ in SHAPE_BANDS:
        mk = (absl >= lo_ - 1e-9) & (absl <= hi_ + 1e-9)
        res["key_mass_by_band"][name] = {"n_games": int(mk.sum())}
        for k in (3, 7):
            cell(f"A6_K{k}_band_{name}", 100 * (key_sim[k] - key_real[k]), mk)
            res["key_mass_by_band"][name][str(k)] = {
                "balanced": float(key_sim[k][mk].mean()),
                "raw": float(key_raw[k][mk].mean()),
                "real": float(key_real[k][mk].mean()),
            }
    res["cells"] = cells
    res["n_cells_recorded"] = len(cells)
    (OUT / "grade.json").write_text(json.dumps(res, indent=1))
    units = {}
    for name in cells:
        if "_K" in name or name in ("A6_H3", "A6_H4"):
            units[name] = "rate_pp"
        elif name in ("A6_W3", "A6_W4"):
            units[name] = "brier_improvement"
        else:
            units[name] = "log_loss_improvement"
    batch_cells = []
    for name, v in cells.items():
        batch_cells.append(
            {
                "name": f"sim08_a6_{name[3:].lower()}",
                "description": f"Amendment 6 balance-point tilted simulator, cell {name[3:]}",
                "effect": round(v["effect"], 6),
                "effect_units": units[name],
                "standard_error": round(v["se"], 6),
                "interval_low": round(v["ci95"][0], 6),
                "interval_high": round(v["ci95"][1], 6),
                "probability_positive": v["probability_positive"],
                "sample_games": v["n_games"],
                "sample_blocks": nb,
            }
        )
    batch = {
        "family": "sim08_market_anchored",
        "source": "artifacts/sim08_amendment6/grade.json",
        "classification": "unresolved_below_power",
        "classification_evidence": (
            "No admissible closing ground; zero crossing closes nothing; third conditioning "
            "examined on the saved 2026-11-03 draws under Amendment 6"
        ),
        "league": "nfl",
        "season_start": 2020,
        "season_end": 2025,
        "category": "modeling",
        "notes": (
            "Exponential tilt to cover share one half at the opener; 2026-11-03 engine without "
            "Amendment 4 terminal transitions. Improvement cells: positive means simulated read "
            "better; K cells simulated minus real mass, percentage points"
        ),
        "cells": batch_cells,
    }
    (OUT / "record_batch.json").write_text(json.dumps(batch, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == 2:
        stage_two()
    if a.stage == 1:
        res = stage_one()
        (OUT / "step1.json").write_text(json.dumps(res, indent=1))
        for r in res["rows"]:
            m = r["mean_margin_minus_line"]
            c = r["cover_share_decisive"]
            p = r.get("push_share")
            print(
                f"{r['line']:5.1f} n={r['n']:4d} mean-line={m['effect']:+.2f} "
                f"[{m['ci95'][0]:+.2f},{m['ci95'][1]:+.2f}] cover={c['effect']:.3f} "
                f"[{c['ci95'][0]:.3f},{c['ci95'][1]:.3f}]"
                + (f" push={p['effect']:.3f}" if p else "")
            )


if __name__ == "__main__":
    main()
