import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead106_injury_language as L6
import lead110_absorption as A

OUT = A.OUT.parent / "lead109_position_weights"
SEED = 109
REPS = 2000
BASELINES = ("des", "desprac")
GROUPS = list(A.GROUPS)


def load():
    s = A.sample_rows()
    ver = pd.read_parquet(L6.ART / "population.parquet")
    ver = L6.id_outcomes(ver)
    ver = ver[ver["verified"]]
    popn = ver[ver["week"] > L6.FIRST_WEEK].groupby("designation").size()
    samn = s.groupby("designation").size()
    s["w"] = s["designation"].map(lambda d: popn[d] / samn[d])
    s["grp"] = s["position"].map(A.POS_TO_GROUP)
    rep = json.loads((A.OUT / "report.json").read_text())
    val = {k: abs(v) for k, v in rep["weights_spread_pts_per_out"].items()}
    s["val"] = s["grp"].map(val)
    y = s["played"].to_numpy()
    for b in BASELINES:
        lp = s["lp_" + b].to_numpy()
        rp = s["rp"].to_numpy()
        ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))  # noqa: E731
        s["d_" + b] = ll(lp) - ll(rp)
    s["cl"] = s["team"] + s["week"].astype(str)
    return s, val


def wm(d, w):
    return float(np.sum(d * w) / np.sum(w))


def valued_mean(d, w, v):
    return wm(d, w * v)


def gap(d, w, v):
    return wm(d, w * v) - wm(d, w)


def cluster_index(cl):
    ids = pd.factorize(cl)[0]
    return [np.where(ids == i)[0] for i in range(ids.max() + 1)]


def boot_stat(by, stat, rng, *arrs):
    n = len(by)
    est = []
    for _ in range(REPS):
        ix = np.concatenate([by[i] for i in rng.integers(0, n, n)])
        est.append(stat(*(a[ix] for a in arrs)))
    e = np.array(est)
    return float(np.quantile(e, 0.025)), float(np.quantile(e, 0.975)), float((e > 0).mean())


def group_cells(s, rng):
    out = []
    for b in BASELINES:
        d = s["d_" + b].to_numpy()
        w = s["w"].to_numpy()
        for g in GROUPS:
            m = (s["grp"] == g).to_numpy()
            if m.sum() < 2:
                out.append({"baseline": b, "group": g, "rows": int(m.sum())})
                continue
            sub = s[m]
            by = cluster_index(sub["cl"])
            dd, ww = d[m], w[m]
            lo, hi, pp = boot_stat(by, wm, rng, dd, ww)
            out.append(
                {
                    "baseline": b,
                    "group": g,
                    "rows": int(m.sum()),
                    "estimate": wm(dd, ww),
                    "low": lo,
                    "high": hi,
                    "probability_positive": pp,
                }
            )
    return out


def weighted_cells(s, rng):
    desig = s["designation"].to_numpy()
    out = []
    for b in BASELINES:
        d = s["d_" + b].to_numpy()
        w = s["w"].to_numpy()
        v = s["val"].to_numpy()
        ok = ~np.isnan(v)
        dd, ww, vv = d[ok], w[ok], v[ok]
        by_ok = cluster_index(s["cl"].to_numpy()[ok])
        plain = wm(dd, ww)
        valued = wm(dd, ww * vv)
        diff = valued - plain
        lo, hi, pp = boot_stat(by_ok, gap, rng, dd, ww, vv)
        lo_v, hi_v, pp_v = boot_stat(by_ok, valued_mean, rng, dd, ww, vv)
        dz = desig[ok]
        null = []
        for _ in range(REPS):
            vs = vv.copy()
            for dg in np.unique(dz):
                m = np.where(dz == dg)[0]
                vs[m] = vv[rng.permutation(m)]
            null.append(wm(dd, ww * vs) - plain)
        null = np.array(null)
        out.append(
            {
                "baseline": b,
                "rows": int(ok.sum()),
                "unweighted_gain": plain,
                "value_weighted_gain": valued,
                "value_weighted_low": lo_v,
                "value_weighted_high": hi_v,
                "value_weighted_probability_positive": pp_v,
                "difference": diff,
                "difference_low": lo,
                "difference_high": hi,
                "difference_probability_positive": pp,
                "null_mean": float(null.mean()),
                "null_p_ge": float((null >= diff).mean()),
                "null_p_abs_ge": float((np.abs(null) >= abs(diff)).mean()),
            }
        )
    return out


def lowo(s, rng):
    weeks = sorted(s["week"].unique())
    out = []
    for b in BASELINES:
        d = s["d_" + b].to_numpy()
        w = s["w"].to_numpy()
        pred_g = np.zeros(len(s))
        pred_0 = np.zeros(len(s))
        for wk in weeks:
            te = (s["week"] == wk).to_numpy()
            tr = ~te
            pred_0[te] = wm(d[tr], w[tr])
            for g in GROUPS:
                gm = (s["grp"] == g).to_numpy()
                trg = tr & gm
                tg = te & gm
                if tg.any():
                    pred_g[tg] = wm(d[trg], w[trg]) if trg.sum() >= 2 else pred_0[tg][0]
        se0 = (d - pred_0) ** 2
        seg = (d - pred_g) ** 2
        gain = se0 - seg
        by = cluster_index(s["cl"])
        lo, hi, pp = boot_stat(by, wm, rng, gain, w)
        val = s["val"].to_numpy()
        ok = ~np.isnan(val)
        gm_held = pd.Series(pred_g[ok]).corr(pd.Series(val[ok]), method="spearman")
        out.append(
            {
                "baseline": b,
                "held_out_weeks": [int(x) for x in weeks],
                "mse_grand_mean": wm(se0, w),
                "mse_group_means": wm(seg, w),
                "mse_improvement": wm(gain, w),
                "low": lo,
                "high": hi,
                "probability_positive": pp,
                "spearman_heldout_group_gain_vs_value": float(gm_held),
            }
        )
    return out


def halves(s):
    out = []
    for b in BASELINES:
        d = s["d_" + b].to_numpy()
        w = s["w"].to_numpy()
        v = s["val"].to_numpy()
        ok = ~np.isnan(v)
        ev = (s["week"] % 2 == 0).to_numpy()
        out.append(
            {
                "baseline": b,
                "even_unweighted": wm(d[ev], w[ev]),
                "odd_unweighted": wm(d[~ev], w[~ev]),
                "even_value_weighted": wm(d[ev & ok], (w * v)[ev & ok]),
                "odd_value_weighted": wm(d[~ev & ok], (w * v)[~ev & ok]),
            }
        )
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s, val = load()
    rng = np.random.default_rng(SEED)
    rep = {
        "value_source": "lead110 report.json OLS 2011-2025 spread points per Out, absolute value",
        "values": val,
        "rows": len(s),
        "rows_by_group": s["grp"].value_counts(dropna=False).to_dict(),
        "group_cells": group_cells(s, rng),
        "weighted": weighted_cells(s, rng),
        "lowo": lowo(s, rng),
        "halves": halves(s),
    }
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
