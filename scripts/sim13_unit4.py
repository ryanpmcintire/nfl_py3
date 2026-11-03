import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim09_dynamics import CT, load, prplays
from sim13_defensive_fatigue import FAMILY, OUT, drive_table

NBOOT = 400
RNG = np.random.default_rng(20261105)
SHORT = 3
STR = ["off_epa", "opp_off", "own_def", "def_epa"]


def drives(d):
    t = drive_table(d)
    res = (
        d.dropna(subset=["fixed_drive_result"])
        .groupby(["game_id", "fixed_drive"])
        .fixed_drive_result.last()
    )
    t = t.merge(res.rename("res"), on=["game_id", "fixed_drive"], how="left")
    t["turn"] = t.res.fillna("").str.contains("turnover|interception|fumble", case=False)
    g = t.groupby("game_id")
    for k in (1, 2):
        t[f"off{k}"] = g.off.shift(k)
        t[f"dfn{k}"] = g.dfn.shift(k)
        t[f"pl{k}"] = g.plays.shift(k)
        t[f"turn{k}"] = g.turn.shift(k)
    t["alt"] = (t.off1 == t.dfn) & (t.off2 == t.off) & (t.dfn1 == t.off)
    return t


def expectation(p, t):
    f = p.groupby(["game_id", "fixed_drive"]).first()[STR].reset_index()
    t = t.merge(f, on=["game_id", "fixed_drive"], how="left").merge(
        p.drop_duplicates(["game_id"])[["game_id", "season"]], on="game_id", how="left"
    )
    ok = t.dropna(subset=[*STR, "plays", "season"])
    X = np.column_stack([np.ones(len(ok)), ok[STR].values])
    y = ok.plays.values.astype(float)
    pred = np.full(len(t), np.nan)
    se = ok.season.values
    for yr in np.unique(se):
        tr = se != yr
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        pred[ok.index[~tr]] = X[~tr] @ b
    t["pred"] = pred
    t["resid"] = t.plays - t.pred
    return t


def flags(t):
    t = t.copy()
    sd = t.resid.std()
    r1 = t.groupby("game_id").resid.shift(1)
    r2 = t.groupby("game_id").resid.shift(2)
    ok = t.alt
    t["A"] = np.where(ok, ((t.pl1 <= SHORT) | t.turn1.astype(bool)).astype(float), np.nan)
    t["B"] = np.where(ok, r2 / sd, np.nan)
    t["pA"] = np.where(ok, ((t.pl2 <= SHORT) | t.turn2.astype(bool)).astype(float), np.nan)
    t["pB"] = np.where(ok, r1 / sd, np.nan)
    return t[["game_id", "fixed_drive", "A", "B", "pA", "pB"]], sd


def gram(X, y, gid, ng):
    order = np.argsort(gid, kind="stable")
    b = np.searchsorted(gid[order], np.arange(ng + 1))
    k = X.shape[1]
    xx = np.zeros((ng, k, k))
    xy = np.zeros((ng, k))
    for g in range(ng):
        ix = order[b[g] : b[g + 1]]
        xx[g] = X[ix].T @ X[ix]
        xy[g] = X[ix].T @ y[ix]
    return xx, xy


def run(s, yv, shocks, defs, label, rows, lo):
    ctl = [*[c for c in CT if c != "pos_home"], "pos_home", "q2", "q3", "q4", "sd_t"]
    cols = shocks + ctl
    s = s.dropna(subset=[*cols, yv])
    X = np.column_stack([np.ones(len(s)), s[cols].astype(float).values])
    X[:, 1:] -= X[:, 1:].mean(0)
    names = ["c", *cols]
    y = s[yv].astype(float).values
    gid = pd.factorize(s.game_id)[0]
    ng = gid.max() + 1
    xx, xy = gram(X, y, gid, ng)
    b = np.linalg.lstsq(xx.sum(0), xy.sum(0), rcond=1e-10)[0]
    d = np.array(
        [
            np.linalg.lstsq(np.tensordot(w, xx, 1), w @ xy, rcond=1e-10)[0]
            for w in (np.bincount(RNG.integers(0, ng, ng), minlength=ng) for _ in range(NBOOT))
        ]
    )
    se = s.season.values
    for a, c in defs:
        v = np.zeros(len(names))
        v[names.index(a)] += 1
        if c:
            v[names.index(c)] -= 1
        dr = d @ v
        rows.append(
            {
                "y": yv,
                "spec": label,
                "contrast": f"{a}-{c}" if c else a,
                "n": len(s),
                "coef": float(v @ b),
                "lo": np.percentile(dr, 2.5),
                "hi": np.percentile(dr, 97.5),
                "prob_pos": float((dr > 0).mean()),
            }
        )
        if yv == "epa":
            for yr in sorted(set(se)):
                tr = se != yr
                bb = np.linalg.lstsq(X[tr].T @ X[tr], X[tr].T @ y[tr], rcond=1e-10)[0]
                lo.append(
                    {
                        "spec": label,
                        "contrast": f"{a}-{c}" if c else a,
                        "season": int(yr),
                        "beta": float(v @ bb),
                    }
                )


def main():
    d = load()
    p = prplays(d)
    t = expectation(p, drives(d))
    f, sd = flags(t)
    p = p.merge(f, on=["game_id", "fixed_drive"], how="left")
    p["q2"] = (p.qtr == 2).astype(float)
    p["q3"] = (p.qtr == 3).astype(float)
    p["q4"] = (p.qtr == 4).astype(float)
    p["sd_t"] = p.sd * p.t / 3600 / 10
    rows, lo = [], []
    run(p, "epa", ["A", "B"], [("A", None), ("B", None)], "main_only", rows, lo)
    run(p, "succ", ["A", "B"], [("A", None), ("B", None)], "main_only", rows, lo)
    jd = [("A", None), ("pA", None), ("A", "pA"), ("B", None), ("pB", None), ("B", "pB")]
    run(p, "epa", ["A", "B", "pA", "pB"], jd, "joint", rows, lo)
    run(p, "succ", ["A", "B", "pA", "pB"], jd, "joint", rows, lo)
    r = pd.DataFrame(rows)
    ls = pd.DataFrame(lo)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "cells_unit4.csv", index=False)
    ls.to_csv(OUT / "loso_unit4.csv", index=False)
    pd.set_option("display.width", 220)
    print(FAMILY, "unit4 looks", len(r), "resid sd", round(sd, 3), "A rate", round(f.A.mean(), 3))
    print(r.round(4).to_string(index=False))
    g = ls.groupby(["spec", "contrast"]).beta.agg(
        ["mean", "std", "min", "max", lambda x: (x > 0).sum()]
    )
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()
