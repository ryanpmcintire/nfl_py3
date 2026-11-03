import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim09_dynamics import CT
from sim13_defensive_fatigue import FAMILY, OUT, build

KNOT = 3.8
NBOOT = 400
RNG = np.random.default_rng(20261104)
BASE = [c for c in CT if c != "pos_home"]


def prep(p):
    p = p.copy()
    p["pl"] = p.own_off_pl
    for nm, c in (("d", "cum_def"), ("p", "pl"), ("o", "cum_off")):
        p[f"{nm}_lo"] = p[c].clip(upper=KNOT)
        p[f"{nm}_hi"] = (p[c] - KNOT).clip(lower=0)
    tr = p.t / 3600
    p["sd_t"] = p.sd * tr
    p["sd_t2"] = p.sd * tr**2
    p["sd_s"] = p.sd / 10
    p["sd_t"] = p.sd_t / 10
    p["sd_t2"] = p.sd_t2 / 10
    dr = p.fixed_drive.clip(upper=25).astype(int)
    for k in sorted(dr.unique())[1:]:
        p[f"dr{k}"] = (dr == k).astype(float)
    return p


def design(p, terms, strong):
    ctl = [*BASE, "own_off_min"]
    if strong:
        dum = [c for c in p.columns if c.startswith("dr") and c[2:].isdigit()]
        ctl = [*ctl, "sd_s", "sd_t", "sd_t2", *dum]
    cols = terms + ctl
    s = p.dropna(subset=[*cols, "epa"])
    X = s[cols].astype(float)
    X = X - X.groupby(s.dkey.values).transform("mean")
    y = s.epa.astype(float)
    y = y - y.groupby(s.dkey.values).transform("mean")
    keep = [c for c in X.columns if X[c].std() > 1e-9]
    return s, X[keep], y, keep


def gram(X, y, gid, ng):
    Xa, ya = X.values, y.values
    k = Xa.shape[1]
    xx = np.zeros((ng, k, k))
    xy = np.zeros((ng, k))
    order = np.argsort(gid, kind="stable")
    b = np.searchsorted(gid[order], np.arange(ng + 1))
    for g in range(ng):
        ix = order[b[g] : b[g + 1]]
        xx[g] = Xa[ix].T @ Xa[ix]
        xy[g] = Xa[ix].T @ ya[ix]
    return xx, xy


def contrasts(terms, keep, defs):
    out = []
    for a, b in defs:
        v = np.zeros(len(keep))
        v[keep.index(a)] += 1
        if b:
            v[keep.index(b)] -= 1
        out.append(v)
    return np.array(out)


def fit(p, terms, defs, strong, label):
    s, X, y, keep = design(p, terms, strong)
    gid = pd.factorize(s.game_id)[0]
    ng = gid.max() + 1
    xx, xy = gram(X, y, gid, ng)
    C = contrasts(terms, keep, defs)
    b = np.linalg.lstsq(xx.sum(0), xy.sum(0), rcond=1e-10)[0]
    d = np.array(
        [
            np.linalg.lstsq(np.tensordot(w, xx, 1), w @ xy, rcond=1e-10)[0]
            for w in (np.bincount(RNG.integers(0, ng, ng), minlength=ng) for _ in range(NBOOT))
        ]
    )
    rows = []
    for i, (a, bb) in enumerate(defs):
        est = C[i] @ b
        dr = d @ C[i]
        rows.append(
            {
                "spec": label,
                "strong": strong,
                "contrast": f"{a}-{bb}" if bb else a,
                "n": len(s),
                "coef": est,
                "lo": np.percentile(dr, 2.5),
                "hi": np.percentile(dr, 97.5),
                "prob_pos": float((dr > 0).mean()),
            }
        )
    return rows, (s, X, y, keep, C)


def loso(pack, ci, label):
    s, X, y, _, C = pack
    Xa, ya, se = X.values, y.values, s.season.values
    rows = []
    for yr in sorted(set(se)):
        tr = se != yr
        b = np.linalg.lstsq(Xa[tr].T @ Xa[tr], Xa[tr].T @ ya[tr], rcond=1e-10)[0]
        rows.append({"spec": label, "season": int(yr), "diff_beta": C[ci] @ b})
    return rows


def main():
    p = prep(build())
    looks = 0
    res, lo = [], []
    for strong in (False, True):
        r, pk = fit(
            p,
            ["cum_def", "pl", "cum_off"],
            [
                ("cum_def", "pl"),
                ("cum_def", "cum_off"),
                ("cum_def", None),
                ("pl", None),
                ("cum_off", None),
            ],
            strong,
            "overall",
        )
        res += r
        looks += 5
        if strong:
            lo += loso(pk, 0, "overall_main_minus_placebo1")
        r, pk = fit(
            p,
            ["d_lo", "d_hi", "p_lo", "p_hi", "o_lo", "o_hi"],
            [
                ("d_hi", "p_hi"),
                ("d_hi", "o_hi"),
                ("d_lo", "p_lo"),
                ("d_hi", None),
                ("p_hi", None),
                ("o_hi", None),
            ],
            strong,
            "knot38",
        )
        res += r
        looks += 6
        if strong:
            lo += loso(pk, 0, "above38_main_minus_placebo1")
            lo += loso(pk, 1, "above38_main_minus_placebo2")
    r = pd.DataFrame(res)
    ls = pd.DataFrame(lo)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "cells_unit3.csv", index=False)
    ls.to_csv(OUT / "loso_unit3.csv", index=False)
    pd.set_option("display.width", 220)
    print(FAMILY, "unit3 looks", looks, "plus loso 3")
    print(r.round(4).to_string(index=False))
    print(ls.pivot(index="season", columns="spec", values="diff_beta").round(4).to_string())
    g = ls.groupby("spec").diff_beta.agg(["mean", "std", "min", "max", lambda x: (x > 0).sum()])
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()
