import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

OUTD = REPO / "artifacts" / "mod25e3" / "sel"
FIT = OUTD / "fit.json"
WFILE = OUTD / "w.npy"
SHRINK = (0.0, 100.0, 1000.0, 10000.0, 100000.0)
IPW_BINS = [-99, -14, -8, -4, -1, 0, 3, 7, 13, 99]
COARSE = [-99, -8.5, -0.5, 0.5, 8.5, 99]
NB = 200
SCHEMES = ("none", "qtr", "coarse", "coarse_qtr", "ipw_qtr")


def enabled():
    return os.environ.get("SEL") == "1"


def scheme_cells(sd, qt, name):
    sdc = np.clip(sd, -24, 24)
    q = np.minimum(qt, 5).astype(int)
    if name == "none":
        return np.zeros(len(sd), dtype=int)
    if name == "qtr":
        return q
    if name == "coarse_qtr":
        return pd.cut(sdc, COARSE, labels=False).astype(int) * 6 + q
    if name == "ipw_qtr":
        return pd.cut(sdc, IPW_BINS, labels=False).astype(int) * 6 + q
    if name == "coarse":
        return pd.cut(sdc, COARSE, labels=False).astype(int)
    raise ValueError(name)


def fit_params(X, cell, shrink):
    mu0 = X.mean(axis=0)
    nc = int(cell.max()) + 1
    n = np.bincount(cell, minlength=nc).astype(float)
    m = np.stack([np.bincount(cell, weights=X[:, j], minlength=nc) for j in range(2)], axis=1) / np.maximum(n, 1)[:, None]
    mu = mu0 + (n / (n + shrink))[:, None] * (m - mu0)
    mu[n == 0] = mu0
    resid = X - mu[cell]
    S = resid.T @ resid / len(X)
    return mu0, mu, S


def quad(X, mu, Si):
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, Si, d)


def cond_ll(X, cell, mu, S):
    Si = np.linalg.inv(S)
    ld = np.linalg.slogdet(S)[1]
    return -0.5 * quad(X, mu[cell], Si) - 0.5 * ld - np.log(2 * np.pi)


def ratio(X, cell, mu0, S0, mu, S):
    return np.exp(-0.5 * quad(X, mu0, np.linalg.inv(S0)) + 0.5 * quad(X, mu[cell], np.linalg.inv(S)) - 0.5 * np.linalg.slogdet(S0)[1] + 0.5 * np.linalg.slogdet(S)[1])


def load_rows():
    import mod25e_srcconc as sc

    tables, trans, pbp = sc.engine()
    a = tables["arrays"]
    gcode, ots, dts, rv, gs, _ = sc.source_tables(tables, trans, pbp)
    home = a["is_home_off"].astype(bool)
    uv = np.nan_to_num(gs.set_index(["g", "h"]).r.reindex(pd.MultiIndex.from_arrays([gcode, ~home])).to_numpy(), nan=0.0)
    meta = pbp.groupby("game_id", sort=False).agg(season=("season", "first"))
    season = meta.season.reindex(trans["game_id"].to_numpy()).to_numpy().astype(int)
    return dict(a=a, gcode=gcode, v=rv, u=uv, season=season, qtr=trans["qtr_actual"].to_numpy(), sd=trans["sc_raw"].to_numpy(dtype=float), code=a["play_type_code"], clk=a["clock_elapsed"].astype(float))


def fit():
    R = load_rows()
    ok = np.isfinite(R["sd"]) & np.isfinite(R["v"]) & np.isfinite(R["u"])
    X = np.c_[R["v"], R["u"]]
    seasons = np.unique(R["season"][ok])
    res = {}
    looks = 0
    for sch in SCHEMES:
        cell_all = scheme_cells(np.where(ok, R["sd"], 0.0), R["qtr"], sch)
        for sh in SHRINK:
            ll = 0.0
            for s in seasons:
                te = ok & (R["season"] == s)
                tr = ok & (R["season"] != s)
                _, mu, S = fit_params(X[tr], cell_all[tr], sh)
                ll += cond_ll(X[te], cell_all[te], mu, S).sum()
            res[f"{sch}|{sh:g}"] = ll / ok.sum()
            looks += 1
    best = max(res, key=res.get)
    sch, sh = best.split("|")
    sh = float(sh)
    cell_all = scheme_cells(np.where(ok, R["sd"], 0.0), R["qtr"], sch)
    w = np.ones(len(X))
    for s in seasons:
        te = ok & (R["season"] == s)
        tr = ok & (R["season"] != s)
        _, mu, S = fit_params(X[tr], cell_all[tr], sh)
        mu0, _, S0 = fit_params(X[tr], np.zeros(int(tr.sum()), dtype=int), 0.0)
        w[te] = ratio(X[te], cell_all[te], mu0, S0, mu, S)
    OUTD.mkdir(parents=True, exist_ok=True)
    np.save(WFILE, w)
    spec = dict(scheme=sch, shrink=sh, heldout_ll_per_row=res, best_minus_unconditional=res[best] - res["none|0"], folds=len(seasons), n_rows=int(ok.sum()), rows=len(X), dist_checksum=float(np.asarray(R["a"]["dist_raw"], dtype=float).sum()), looks=looks, w_mean=float(w[ok].mean()), w_sd=float(w[ok].std()), w_min=float(w[ok].min()), w_max=float(w[ok].max()), cell_edges=dict(coarse=COARSE, ipw=IPW_BINS))
    FIT.write_text(json.dumps(spec), encoding="utf-8")
    print("best", best, "gain over unconditional", spec["best_minus_unconditional"], "w mean/sd/min/max", spec["w_mean"], spec["w_sd"], spec["w_min"], spec["w_max"], "looks", looks)
    print({k: round(v, 5) for k, v in res.items()})
    return R, w, ok, spec


def slopes(g, q, x, y, wt, ngr, nb, rng):
    key = g * 4 + (q - 1)
    k = ngr * 4

    def agg(arr):
        return np.bincount(key, weights=arr, minlength=k).reshape(ngr, 4)

    A = [agg(wt), agg(wt * x), agg(wt * y), agg(wt * x * x), agg(wt * x * y)]
    ws = [np.ones(ngr)] + [np.bincount(rng.integers(0, ngr, ngr), minlength=ngr).astype(float) for _ in range(nb)]
    out = []
    for w_ in ws:
        n_, x_, y_, xx_, xy_ = (w_ @ M for M in A)
        cv = xy_ - x_ * y_ / n_
        vr = xx_ - x_ * x_ / n_
        out.append(np.r_[cv.sum() / vr.sum(), cv / vr])
    out = np.array(out)
    return out[0], out[1:]


def validate(R=None, w=None, ok=None, spec=None):
    if R is None:
        R = load_rows()
        spec = json.loads(FIT.read_text(encoding="utf-8"))
        w = np.load(WFILE)
        ok = np.isfinite(R["sd"]) & np.isfinite(R["v"]) & np.isfinite(R["u"])
    rng = np.random.default_rng(63)
    lines = []

    def say(s):
        print(s)
        lines.append(s)

    q = R["qtr"].astype(int)
    m = ok & (q >= 1) & (q <= 4)
    g = pd.factorize(R["gcode"][m])
    x = R["sd"][m] / 7.0
    say(f"E63 offline validation on real pool rows (n={int(m.sum())}), scheme {spec['scheme']} shrink {spec['shrink']:g}; w cross-fitted by season; game bootstrap {NB}")
    for nm in ("v", "u"):
        y = R[nm][m]
        for tag, wt in (("unweighted", np.ones(int(m.sum()))), ("reweighted", w[m])):
            p, bs = slopes(g[0], q[m], x, y, wt, len(g[1]), NB, rng)
            say(f"  {nm} {tag}: pooled within-quarter slope per 7 pts {p[0]:+.4f} [{np.percentile(bs[:, 0], 5):+.4f},{np.percentile(bs[:, 0], 95):+.4f}] pp {float((bs[:, 0] > 0).mean()):.2f}; by quarter " + " ".join(f"Q{i + 1} {p[i + 1]:+.4f}" for i in range(4)))
    band = np.digitize(R["sd"][m], np.array([-8.5, -0.5, 0.5, 8.5]))
    cell = (q[m] - 1) * 5 + band
    code = R["code"][m]
    clk = R["clk"][m]
    rp = np.isin(code, (0, 1))
    wm = w[m]
    say("  state effects by quarter-band cell, unweighted -> reweighted (pass share of run/pass plays; mean clock elapsed s)")
    dpass, dclk, ch = [], [], []
    for c in range(20):
        s = (cell == c) & rp
        if s.sum() < 200:
            continue
        pu, pw = (code[s] == 1).mean(), np.average(code[s] == 1, weights=wm[s])
        cu, cw = clk[cell == c].mean(), np.average(clk[cell == c], weights=wm[cell == c])
        dpass.append(pw - pu)
        dclk.append(cw - cu)
        ch.append(f"Q{c // 5 + 1}b{c % 5} {pu:.3f}->{pw:.3f} {cu:.2f}->{cw:.2f}")
    say("    " + "; ".join(ch))
    say(f"    pass share change mean {np.mean(dpass):+.4f} mean abs {np.mean(np.abs(dpass)):.4f} max abs {np.max(np.abs(dpass)):.4f}; clock change mean {np.mean(dclk):+.3f} mean abs {np.mean(np.abs(dclk)):.3f} max abs {np.max(np.abs(dclk)):.3f}")
    say(f"  looks: 2 outcomes x 2 weightings x 5 slopes = 20 slope prints; {len(dpass)} cells x 2 state effects; fit looks {spec['looks']}")
    (OUTD / "validate.txt").write_text("\n".join(lines), encoding="utf-8")


def install_sel():
    import mod25d_variance as dv

    spec = json.loads(FIT.read_text(encoding="utf-8"))
    w = np.load(WFILE)
    t = dv._G["tables"]
    ns = dv._G["ns"]
    a = t["arrays"]
    assert len(w) == len(a["off_row"])
    assert abs(float(np.asarray(a["dist_raw"], dtype=float).sum()) - spec["dist_checksum"]) < 1e-6
    assert "IPW" in ns
    ns["IPW"] = np.asarray(ns["IPW"], dtype=np.float64) * w
    t["nn_cache_cond"].clear()
    t["nn_weight_cache_cond"].clear()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    sys.argv = sys.argv[:1] + ["200"]
    if cmd == "fit":
        fit()
    elif cmd == "validate":
        validate()
    else:
        R, w, ok, spec = fit()
        validate(R, w, ok, spec)
