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

RUN_ROOT = REPO / "artifacts" / "sim08_market_anchored"
KEY_NUMBERS = base.KEY_NUMBERS
N_BOOT = 2000
SEED = 20261103
BANDS = (("le3", 0.0, 3.0), ("3.5to6.5", 3.5, 6.5), ("7to9.5", 7.0, 9.5), ("10plus", 10.0, 1e9))
REAL_TOTAL_SLOPE = 0.867


def blocks_of(frame):
    key = (frame["season"].astype(str) + "_" + frame["week"].astype(str)).to_numpy()
    _, inv = np.unique(key, return_inverse=True)
    return inv, int(inv.max()) + 1


def boot_mean(d, inv, nb, rng, mask=None):
    w = np.ones(len(d)) if mask is None else mask.astype(float)
    s = np.bincount(inv, weights=d * w, minlength=nb)
    n = np.bincount(inv, weights=w, minlength=nb)
    mult = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=N_BOOT).astype(float)
    num, den = mult @ s, mult @ n
    ok = den > 0
    b = num[ok] / den[ok]
    point = float((d * w).sum() / w.sum())
    return {
        "effect": point,
        "se": float(b.std(ddof=1)),
        "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
        "probability_positive": float(np.mean(b > 0.0)),
        "n_games": int(w.sum()),
    }


def boot_slope_diff(x, y_sim, y_real, inv, nb, rng):
    def slope(w, y):
        mx = (w * x).sum() / w.sum()
        my = (w * y).sum() / w.sum()
        return (w * (x - mx) * (y - my)).sum() / (w * (x - mx) ** 2).sum()

    mult = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=N_BOOT)
    bd = np.empty(N_BOOT)
    for i in range(N_BOOT):
        w = mult[i][inv].astype(float)
        bd[i] = slope(w, y_sim) - slope(w, y_real)
    ones = np.ones(len(x))
    s_sim, s_real = slope(ones, y_sim), slope(ones, y_real)
    return {
        "effect": float(s_sim - s_real),
        "sim_slope": float(s_sim),
        "real_slope": float(s_real),
        "se": float(bd.std(ddof=1)),
        "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
        "probability_positive": float(np.mean(bd > 0.0)),
        "n_games": len(x),
    }


def loso_cond(x_cols, y_cover, nonpush, seasons):
    from sklearn.linear_model import LogisticRegression

    out = np.empty(len(y_cover))
    coefs = {}
    for s_ in base.GRADED_SEASONS:
        tr = (seasons != s_) & nonpush
        te = seasons == s_
        m = LogisticRegression(C=1e10, max_iter=2000).fit(x_cols[tr], y_cover[tr])
        out[te] = m.predict_proba(x_cols[te])[:, 1]
        coefs[int(s_)] = [float(m.intercept_[0])] + [float(c) for c in m.coef_[0]]
    return out, coefs


def three_way(cond, push):
    return (1 - push) * cond, push, (1 - push) * (1 - cond)


def score(p3, outcome):
    return base.three_way_score(*p3, outcome)


def load_arm(run, arm, frame):
    parts = []
    for s_ in base.GRADED_SEASONS:
        z = np.load(RUN_ROOT / run / f"{arm}_{s_}.npz")
        parts.append({k: z[k] for k in z.files})
    ids = np.concatenate([p["game_id"] for p in parts])
    assert list(ids) == list(frame["game_id"]), "game order mismatch"
    return {
        "margins": np.concatenate([p["margins"] for p in parts]).astype(float),
        "totals": np.concatenate([p["totals"] for p in parts]).astype(float),
        "ot": np.concatenate([p["ot"] for p in parts]).astype(float),
        "x_diff": np.concatenate([p["x_diff"] for p in parts]),
        "x_env": np.concatenate([p["x_env"] for p in parts]),
        "draws": int(parts[0]["draws"]),
        "seconds": [float(p["seconds"]) for p in parts],
    }


def sim_counts(margins, line):
    diff = margins - line[:, None]
    c = (diff > base.ATOM_TOLERANCE).sum(1).astype(float)
    l_ = (diff < -base.ATOM_TOLERANCE).sum(1).astype(float)
    return c, margins.shape[1] - c - l_, l_


def market_hist_three_way(frame):
    g = pd.read_parquet(base.GAME_FEATURES_PATH)
    g = g.loc[g["game_type"] == "REG"]
    margin = (g["home_score"].astype(float) - g["away_score"].astype(float)).to_numpy()
    seas = g["season"].to_numpy()
    cov = np.empty(len(frame))
    push = np.empty(len(frame))
    loss = np.empty(len(frame))
    cache = {}
    for i, (s_, line) in enumerate(zip(frame["season"], frame["line"], strict=True)):
        if s_ not in cache:
            cache[s_] = np.rint(margin[seas < s_])
        h = base.tilt_hist(cache[s_], float(line))
        cov[i], push[i], loss[i] = base.hist_to_three_way(h, float(line))
    return cov, push, loss


def real_overtime(frame):
    out = {}
    for s_ in base.GRADED_SEASONS:
        p = pd.read_parquet(
            REPO / "data" / "pbp" / "raw" / "20260925T202544Z" / f"season={s_}" / "plays.parquet",
            columns=["game_id", "qtr"],
        )
        out.update(p.groupby("game_id")["qtr"].max().ge(5).to_dict())
    return frame["game_id"].map(out).astype(float).to_numpy()


def reliability(p, y, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return [
        {
            "bin": b,
            "n": int((idx == b).sum()),
            "mean_p": float(p[idx == b].mean()),
            "rate": float(y[idx == b].mean()),
        }
        for b in range(bins)
        if (idx == b).any()
    ]


def arm_looks(label, S, frame, ctx, rng):
    outcome, seasons, inv, nb = ctx["outcome"], ctx["seasons"], ctx["inv"], ctx["nb"]
    nonpush, y = ctx["nonpush"], ctx["y"]
    line = frame["line"].to_numpy(dtype=float)
    c, pu, l_ = sim_counts(S["margins"], line)
    n = S["draws"]
    raw3 = ((c + 0.5) / (n + 1.5), (pu + 0.5) / (n + 1.5), (l_ + 0.5) / (n + 1.5))
    sim_logit = base.safe_logit((c + 0.5) / (c + l_ + 1.0))
    X = np.column_stack([ctx["served_logit"], sim_logit])
    blend_cond, blend_coefs = loso_cond(X, y, nonpush, seasons)
    blend3 = three_way(blend_cond, ctx["served"][1])
    ll_raw, br_raw = score(raw3, outcome)
    ll_bl, br_bl = score(blend3, outcome)
    ll_rec, br_rec = ctx["ll_recal"], ctx["br_recal"]  # noqa: F841
    cells = {}

    def cell(name, d, mask=None):
        cells[name] = boot_mean(d, inv, nb, rng, mask)

    cell("L1", ctx["ll_served"] - ll_raw)
    cell("L2", ctx["ll_model"] - ll_raw)
    cell("L3", ctx["ll_market"] - ll_raw)
    d4 = ll_rec - ll_bl
    cell("L4", d4)
    cell("L5", ctx["ll_served"] - ll_bl)
    d1 = ctx["ll_served"] - ll_raw
    for s_ in base.GRADED_SEASONS:
        m = seasons == s_
        cell(f"S_L1_{s_}", d1, m)
        cell(f"S_L4_{s_}", d4, m)
    push_mask = outcome == "push"
    cell("P", ctx["ll_served"] - ll_raw, push_mask)
    absl = np.abs(line)
    for name, lo, hi in BANDS:
        cell(f"B_{name}", d4, (absl >= lo - 1e-9) & (absl <= hi + 1e-9))
    pick_bl = blend_cond > 0.5
    pick_rec = ctx["recal_cond"] > 0.5
    dec = nonpush & (pick_bl != pick_rec)
    decisive = {
        "n_decisive": int(dec.sum()),
        "blend_correct": int((pick_bl[dec] == (y[dec] > 0.5)).sum()),
        "recal_correct": int((pick_rec[dec] == (y[dec] > 0.5)).sum()),
    }
    full_X = X[nonpush]
    from sklearn.linear_model import LogisticRegression

    mfull = LogisticRegression(C=1e10, max_iter=2000).fit(full_X, y[nonpush])
    in_cond = mfull.predict_proba(X)[:, 1]
    in_ll = score(three_way(in_cond, ctx["served"][1]), outcome)[0]
    mrec = LogisticRegression(C=1e10, max_iter=2000).fit(
        ctx["served_logit"][nonpush, None], y[nonpush]
    )
    inr_ll = score(
        three_way(mrec.predict_proba(ctx["served_logit"][:, None])[:, 1], ctx["served"][1]), outcome
    )[0]
    coef_arr = np.array(list(blend_coefs.values()))
    return {
        "label": label,
        "cells": cells,
        "blend_fold_coefs_intercept_served_sim": blend_coefs,
        "blend_coef_range": {
            "intercept": [float(coef_arr[:, 0].min()), float(coef_arr[:, 0].max())],
            "served": [float(coef_arr[:, 1].min()), float(coef_arr[:, 1].max())],
            "sim": [float(coef_arr[:, 2].min()), float(coef_arr[:, 2].max())],
        },
        "decisive_games": decisive,
        "in_sample_L4_delta": float((inr_ll - in_ll).mean()),
        "out_of_sample_L4_delta": float(d4.mean()),
        "log_loss": {"sim_raw": float(ll_raw.mean()), "blend": float(ll_bl.mean())},
        "brier": {"sim_raw": float(br_raw.mean()), "blend": float(br_bl.mean())},
        "reliability_sim_raw_cover": reliability(raw3[0], (outcome == "cover").astype(float)),
        "reliability_blend_cover": reliability(blend3[0], (outcome == "cover").astype(float)),
        "sim_logit_minus_served_sd_by_train_fold": {
            int(s_): float(np.std((sim_logit - ctx["served_logit"])[seasons != s_], ddof=1))
            for s_ in base.GRADED_SEASONS
        },
        "anchor_check": {
            "mean_margin_err": float((S["margins"].mean(1) - line).mean()),
            "mean_total_err": float(np.nanmean(S["totals"].mean(1) - ctx["total"])),
        },
    }


def shape_looks(S, frame, ctx, rng):
    inv, nb = ctx["inv"], ctx["nb"]
    result = frame["result"].to_numpy(dtype=float)
    out = {}
    for k in KEY_NUMBERS:
        sim_k = (np.abs(S["margins"]) == k).mean(1)
        real_k = (np.abs(result) == k).astype(float)
        r = boot_mean(sim_k - real_k, inv, nb, rng)
        r["sim_mass"] = float(sim_k.mean())
        r["real_mass"] = float(real_k.mean())
        out[f"K_{k}"] = r
    tot = ctx["total"]
    ok = np.isfinite(tot)
    out["T"] = boot_slope_diff(
        tot[ok],
        S["totals"].mean(1)[ok],
        ctx["real_total"][ok],
        inv[ok],
        nb,
        rng,
    )
    rot = ctx["real_ot"]
    out["O"] = boot_mean(S["ot"].mean(1) - rot, inv, nb, rng)
    out["O"]["sim_rate"] = float(S["ot"].mean())
    out["O"]["real_rate"] = float(rot.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--arms", default="primary")
    a = ap.parse_args()
    from nfl_ats.schedule_flag_features import default_opener_lines

    frame = base.load_games()
    lines = default_opener_lines(frame[["game_id", "season", "week"]])
    frame = frame.merge(lines[["game_id", "tue_open_total_line"]], on="game_id", how="left")
    frame["total"] = frame["tue_open_total_line"].astype(float)
    g = pd.read_parquet(base.GAME_FEATURES_PATH)
    g = g.loc[g["game_type"] == "REG", ["game_id", "home_score", "away_score"]]
    frame = frame.merge(g, on="game_id", how="left")
    outcome = frame["outcome"].to_numpy()
    seasons = frame["season"].to_numpy()
    nonpush = outcome != "push"
    y = (outcome == "cover").astype(float)
    cov, push, loss = base.baseline_three_way(frame)
    served_logit = base.safe_logit(cov / np.clip(cov + loss, base.EPS, None))
    recal_cond, recal_coefs = loso_cond(served_logit[:, None], y, nonpush, seasons)
    ll_recal, br_recal = score(three_way(recal_cond, push), outcome)
    ll_served = score((cov, push, loss), outcome)[0]
    mdl = frame["home_cover_probability_at_open_discrete_uncorrected"].to_numpy(dtype=float)
    mdl_push = frame["push_probability_at_open"].to_numpy(dtype=float)
    mdl3 = (np.clip(mdl, 0, 1 - mdl_push), mdl_push, np.clip(1 - mdl - mdl_push, 0, 1))
    ll_model = score(mdl3, outcome)[0]
    m3 = market_hist_three_way(frame)
    ll_market = score(m3, outcome)[0]
    inv, nb = blocks_of(frame)
    ctx = {
        "outcome": outcome,
        "seasons": seasons,
        "inv": inv,
        "nb": nb,
        "nonpush": nonpush,
        "y": y,
        "served": (cov, push, loss),
        "served_logit": served_logit,
        "recal_cond": recal_cond,
        "ll_recal": ll_recal,
        "br_recal": br_recal,
        "ll_served": ll_served,
        "ll_model": ll_model,
        "ll_market": ll_market,
        "total": frame["total"].to_numpy(dtype=float),
        "real_total": (frame["home_score"] + frame["away_score"]).to_numpy(dtype=float),
        "real_ot": real_overtime(frame),
    }
    rng = np.random.default_rng(SEED)
    res = {
        "n_games": len(frame),
        "n_blocks": nb,
        "baselines_log_loss": {
            "served_raw": float(ll_served.mean()),
            "served_recal": float(ll_recal.mean()),
            "model_only": float(ll_model.mean()),
            "market_hist": float(ll_market.mean()),
        },
        "baselines_brier": {"served_recal": float(br_recal.mean())},
        "recal_fold_coefs": recal_coefs,
        "arms": {},
    }
    for arm in a.arms.split(","):
        S = load_arm(a.run, arm, frame)
        res["arms"][arm] = arm_looks(arm, S, frame, ctx, rng)
        res["arms"][arm]["draws"] = S["draws"]
        res["arms"][arm]["fold_seconds"] = S["seconds"]
        if arm == "primary":
            res["shape"] = shape_looks(S, frame, ctx, rng)
        print(arm, json.dumps(res["arms"][arm]["cells"]["L4"]), flush=True)
    out = RUN_ROOT / a.run / "evaluation.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
