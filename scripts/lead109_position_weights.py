import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead106_injury_language as L6
import lead110_absorption as A

OUT = A.OUT.parent / "lead109_position_weights"
SEED = 109
REPS = 2000
BASELINES = ("des", "desprac")
ALPHA = 0.05
POWER = 0.8
Z_MDE = NormalDist().inv_cdf(1 - ALPHA / 2) + NormalDist().inv_cdf(POWER)
HIGH_VALUE = ("QB", "OL")
VALUE_SOURCE = "lead110 report.json OLS 2011-2025 spread points per Out, absolute value"
FORWARD = L6.ART / "forward_score"
GROUPS = list(A.GROUPS)


def load():
    s = A.sample_rows()
    ver = pd.read_parquet(L6.ART / "population.parquet")
    ver = L6.id_outcomes(ver)
    ver = ver[ver["verified"]]
    popn = ver[ver["week"] > L6.FIRST_WEEK].groupby("designation").size()
    samn = s.groupby("designation").size()
    s["w"] = s["designation"].map(lambda d: popn[d] / samn[d])
    return finish(s)


def load_forward(weeks):
    fr = [pd.read_parquet(FORWARD / f"2026_week{wk:02d}_scored.parquet") for wk in weeks]
    s = pd.concat(fr, ignore_index=True)
    s = s[s["verified"]].copy().reset_index(drop=True)
    s["lp_des"] = s["lp_designation"]
    s["lp_desprac"] = s["lp_designation_x_practice"]
    s["w"] = 1.0
    return finish(s)


def finish(s):
    s["grp"] = s["position"].map(A.POS_TO_GROUP)
    rep = json.loads((A.OUT / "report.json").read_text())
    val = {k: abs(v) for k, v in rep["weights_spread_pts_per_out"].items()}
    s["val"] = s["grp"].map(val)
    y = s["played"].to_numpy(float)
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
    return (
        float(np.quantile(e, ALPHA / 2)),
        float(np.quantile(e, 1 - ALPHA / 2)),
        float((e > 0).mean()),
        float(Z_MDE * e.std(ddof=1)),
    )


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
            lo, hi, pp, md = boot_stat(by, wm, rng, dd, ww)
            out.append(
                {
                    "baseline": b,
                    "group": g,
                    "rows": int(m.sum()),
                    "estimate": wm(dd, ww),
                    "low": lo,
                    "high": hi,
                    "probability_positive": pp,
                    "detectable": md,
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
        lo, hi, pp, md = boot_stat(by_ok, gap, rng, dd, ww, vv)
        lo_v, hi_v, pp_v, md_v = boot_stat(by_ok, valued_mean, rng, dd, ww, vv)
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
                "value_weighted_detectable": md_v,
                "difference": diff,
                "difference_low": lo,
                "difference_high": hi,
                "difference_probability_positive": pp,
                "difference_detectable": md,
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
        lo, hi, pp, md = boot_stat(by, wm, rng, gain, w)
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
                "detectable": md,
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


def binom_two_sided(k, n):
    if n == 0:
        return None
    pk = [math.comb(n, i) / 2**n for i in range(n + 1)]
    return float(min(1.0, sum(q for q in pk if q <= pk[k] * (1 + 1e-12))))


def decisive(s):
    out = []
    y = s["played"].to_numpy(float) > 0
    for b in BASELINES:
        rs = s["rp"].to_numpy() > 0.5
        ls = s["lp_" + b].to_numpy() > 0.5
        m = rs != ls
        right = (rs == y) & m
        by = {}
        for g in GROUPS:
            gm = m & (s["grp"] == g).to_numpy()
            by[g] = [int((right & gm).sum()), int(gm.sum())]
        hv = m & s["grp"].isin(HIGH_VALUE).to_numpy()
        k, n = int(right.sum()), int(m.sum())
        out.append(
            {
                "baseline": b,
                "read_right": k,
                "ladder_right": n - k,
                "rows": n,
                "exact_two_sided_p": binom_two_sided(k, n),
                "qb_ol_read_right": int((right & hv).sum()),
                "qb_ol_rows": int(hv.sum()),
                "by_group_read_right_of_rows": by,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser(description="LEAD-109 position-value weighting of the read")
    ap.add_argument("--forward-weeks", type=int, nargs="*", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    if a.forward_weeks:
        s, val = load_forward(a.forward_weeks)
        filed = s[s["status_filed"].astype(bool)].reset_index(drop=True)
        rep = {
            "population": "lead106 forward scored rows, verified, full population",
            "weeks": a.forward_weeks,
            "value_source": VALUE_SOURCE,
            "values": val,
            "rows": len(s),
            "games": int(s["game_id"].nunique()),
            "team_week_clusters": int(s["cl"].nunique()),
            "rows_by_group": s["grp"].value_counts(dropna=False).to_dict(),
            "decisive": decisive(s),
            "group_cells": group_cells(s, rng),
            "weighted": weighted_cells(s, rng),
            "lowo": lowo(s, rng),
            "halves": halves(s),
            "filed_only_rows": len(filed),
            "filed_only_weighted": weighted_cells(filed, rng),
            "z_detectable": Z_MDE,
        }
        name = "forward_report_weeks_" + "_".join(str(w) for w in a.forward_weeks) + ".json"
    else:
        s, val = load()
        rep = {
            "value_source": VALUE_SOURCE,
            "values": val,
            "rows": len(s),
            "rows_by_group": s["grp"].value_counts(dropna=False).to_dict(),
            "group_cells": group_cells(s, rng),
            "weighted": weighted_cells(s, rng),
            "lowo": lowo(s, rng),
            "halves": halves(s),
        }
        name = "report.json"
    (OUT / name).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
