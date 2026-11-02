import json
import sys
import time

import numpy as np
import pandas as pd
import sim07_personnel as m

REPS = 200
B_HALF = 100
B_FULL = 40
LADDER = (0.0, 0.5, 1.0, 2.0, 4.0)
SEED = 20261103


def boot_r(fo, fe, io, ie, nc, ok, rng):
    est = np.zeros((2, nc))
    c1, *_ = fo.fit(np.ones(len(fo.y)))
    c2, *_ = fe.fit(np.ones(len(fe.y)))
    est[0, io], est[1, ie] = c1, c2
    bs = np.zeros((2, B_HALF, nc))
    for b in range(B_HALF):
        c1, *_ = fo.fit(fo.boot(rng))
        c2, *_ = fe.fit(fe.boot(rng))
        bs[0, b, io] = c1
        bs[1, b, ie] = c2
    se = bs.std(axis=1, ddof=1)
    w = np.where(ok, 1.0 / (se[0] ** 2 + se[1] ** 2 + 1e-300), 0.0)
    r0 = m.wcorr(est[0][ok], est[1][ok], w[ok])[0]
    rb = np.array([m.wcorr(bs[0, b][ok], bs[1, b][ok], w[ok])[0] for b in range(B_HALF)])
    rb = rb[np.isfinite(rb)]
    return r0, np.quantile(rb, 0.025), np.quantile(rb, 0.975)


def full_tau2(ff, rng):
    cell, sw, *_ = ff.fit(np.ones(len(ff.y)))
    bs = np.zeros((B_FULL, ff.n_c))
    for b in range(B_FULL):
        bs[b] = ff.fit(ff.boot(rng))[0]
    se = bs.std(axis=0, ddof=1)
    k = sw > 0
    return m.tau2_moment(cell[k], sw[k], se[k] ** 2)


def main():
    nrep = int(sys.argv[1]) if len(sys.argv) > 1 else REPS
    d, _, _ = m.prepare(m.build())
    d = d.reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    real = pd.read_csv(m.OUT / "cells_all.csv")
    unit = float(real.se.median())
    od = d[d.week % 2 == 1].reset_index(drop=True)
    ev = d[d.week % 2 == 0].reset_index(drop=True)
    fo, fe, ff = m.Fitter(od), m.Fitter(ev), m.Fitter(d)
    cells = sorted(set(fo.cells) | set(fe.cells))
    io = np.array([cells.index(c) for c in fo.cells])
    ie = np.array([cells.index(c) for c in fe.cells])
    nc = len(cells)
    ok = np.ones(nc, bool)
    cell_of = {id(f): np.array([cells.index(c) for c in f.cells])[f.c] for f in (fo, fe, ff)}
    y0 = {id(f): f.y.copy() for f in (fo, fe, ff)}
    ncells = nc // 3
    ng = nc
    out = []
    sizes = [dict(mult=k, delta=k * unit) for k in LADDER]
    for s in sizes:
        t0 = time.time()
        rl = np.zeros(nrep)
        rh = np.zeros(nrep)
        r0 = np.zeros(nrep)
        t2 = np.zeros(nrep)
        for i in range(nrep):
            pick = rng.choice(ng, ncells, replace=False)
            sign = rng.choice([-1.0, 1.0], ncells)
            off = np.zeros(ng)
            off[pick] = sign * s["delta"]
            for f in (fo, fe, ff):
                f.y[:] = y0[id(f)] + off[cell_of[id(f)]]
            ok2 = ok & np.isin(np.arange(nc), np.union1d(io, ie))
            r0[i], rl[i], rh[i] = boot_r(fo, fe, io, ie, nc, ok2, rng)
            t2[i] = full_tau2(ff, rng)
        det_r = rl > 0
        det_t = t2 > 0
        row = dict(
            mult=s["mult"],
            delta=s["delta"],
            reps=nrep,
            power_r=float(det_r.mean()),
            power_tau=float(det_t.mean()),
            power_both=float((det_r & det_t).mean()),
            r_median=float(np.median(r0)),
            tau_median=float(np.sqrt(np.median(t2))),
            seconds=time.time() - t0,
        )
        out.append(row)
        print(row, flush=True)
        (m.OUT / "control_summary.json").write_text(
            json.dumps(
                dict(
                    unit_median_se=unit,
                    cells_planted=ncells,
                    b_half=B_HALF,
                    b_full=B_FULL,
                    rows=out,
                ),
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
