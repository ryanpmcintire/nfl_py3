import glob
import json
import time

import numpy as np
import pandas as pd
import sim07_personnel as m

BOOT = 2000
PLANT = 4000
NPLANT = 6
SEED = 20261103
TARGET = 0.8
ALPHA = 0.05
RELEVANT = 0.005
OLD_UNIT = 0.0113431092149095
EXTRA = ["fumble_lost", "xpass", "wp", "success"]


def extra():
    pbp = sorted(glob.glob(str(m.ROOT / "data/pbp/raw/*")))[-1]
    fr = []
    for s in m.SEASONS:
        x = pd.read_parquet(
            f"{pbp}/season={s}/plays.parquet",
            columns=["game_id", "play_id"] + EXTRA,  # noqa: RUF005
        )
        fr.append(x)
    e = pd.concat(fr, ignore_index=True)
    return e.drop_duplicates(["game_id", "play_id"])


class Aug(m.Fitter):
    def __init__(self, d, cols):
        super().__init__(d)
        if cols:
            z = np.column_stack([d[c].values.astype(np.float32) for c in cols])
            self.X = np.hstack([self.X, z]).astype(np.float32)
            self.buf = np.empty_like(self.X)


def cell_game_matrices(f, resid):
    k = f.n_c
    ng = len(f.games)
    idx = f.gidx * k + f.c
    s = np.bincount(idx, resid, ng * k).reshape(ng, k)
    n = np.bincount(idx, minlength=ng * k).reshape(ng, k).astype(float)
    return s, n


def boot_means(s, n, rng, b):
    out = np.zeros((b, s.shape[1]))
    ng = s.shape[0]
    for i in range(b):
        w = rng.poisson(1.0, ng)
        sw = w @ n
        out[i] = (w @ s) / np.where(sw > 0, sw, 1.0)
    return out


def projector(cells, tot):
    og = sorted({c.split("|")[0] for c in cells})
    dp = sorted({c.split("|")[1] for c in cells})
    r = np.zeros((len(cells), len(og) + len(dp)))
    for i, c in enumerate(cells):
        o, q = c.split("|")
        r[i, og.index(o)] = 1.0
        r[i, len(og) + dp.index(q)] = 1.0
    w = np.diag(tot)
    return np.eye(len(cells)) - r @ np.linalg.pinv(r.T @ w @ r) @ r.T @ w, (len(og) - 1) * (
        len(dp) - 1
    )


def power_curve(mb, proj, rank, rng, grid):
    nc = mb.shape[1]
    noise = (mb - mb.mean(axis=0)) @ proj.T
    cov = np.cov(noise.T)
    ev, evec = np.linalg.eigh(cov)
    keep = np.argsort(ev)[::-1][:rank]
    inv = evec[:, keep] / ev[keep] @ evec[:, keep].T

    def q(e):
        return np.einsum("...i,ij,...j->...", e, inv, e)

    half = len(noise) // 2
    crit = np.quantile(q(noise[:half]), 1 - ALPHA)
    pool = noise[half:]
    out = []
    for dlt in grid:
        hit = 0
        for _ in range(PLANT):
            th = np.zeros(nc)
            th[rng.choice(nc, NPLANT, replace=False)] = rng.choice([-1.0, 1.0], NPLANT) * dlt
            e = pool[rng.integers(0, len(pool))] + proj @ th
            hit += q(e) > crit
        out.append(hit / PLANT)
    return crit, out, q


def smallest(grid, pw):
    pw = np.array(pw)
    i = np.argmax(pw >= TARGET)
    if pw[i] < TARGET:
        return None
    if i == 0:
        return float(grid[0])
    x0, x1, y0, y1 = grid[i - 1], grid[i], pw[i - 1], pw[i]
    return float(x0 + (TARGET - y0) * (x1 - x0) / (y1 - y0))


def main():
    t0 = time.time()
    d, _, _ = m.prepare(m.build())
    d = d.merge(extra(), on=["game_id", "play_id"], how="left").reset_index(drop=True)
    d = d[d.xpass.notna() & d.wp.notna() & d.fumble_lost.notna()].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    variants = {
        "raw_epa": ("epa", []),
        "luck_stripped_epa": ("stripped", ["xpass", "wp"]),
        "success_rate": ("success", ["xpass", "wp"]),
    }
    gamma = float(np.polyfit(d.fumble_lost.values.astype(float), d.epa.values, 1)[0])
    d["stripped"] = d.epa - gamma * d.fumble_lost.astype(float)
    grid = np.array([0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.045])
    res = {}
    for name, (ycol, cols) in variants.items():
        f = Aug(d, cols)
        f.y[:] = d[ycol].values.astype(np.float64)
        cell, sw, *_ = f.fit(np.ones(len(f.y)))
        resid = f.t1.copy()
        s, n = cell_game_matrices(f, resid)
        mb = boot_means(s, n, rng, BOOT)
        se = mb.std(axis=0, ddof=1)
        sd = float(resid.std())
        proj, rank = projector(f.cells, sw)
        crit, pw, qf = power_curve(mb, proj, rank, rng, grid)
        det = smallest(grid, pw)
        q_obs = float(qf(proj @ cell))
        p_obs = float((qf((mb - mb.mean(axis=0)) @ proj.T) >= q_obs).mean())
        pc = proj @ cell
        rms = float(
            np.sqrt(max(0.0, ((pc**2) - (proj @ np.diag(se**2) @ proj.T).diagonal()).mean()))
        )
        res[name] = dict(  # noqa: C408
            label_sd=sd,
            cell_se_median=float(np.median(se)),
            cell_se_min=float(se.min()),
            cell_se_max=float(se.max()),
            power_grid=dict(zip(map(str, grid), pw)),  # noqa: B905
            detectable_at_power_0_8=det,
            detectable_in_sd_units=None if det is None else det / sd,
            omnibus_q=q_obs,
            omnibus_rank=int(rank),
            omnibus_crit=float(crit),
            omnibus_p_boot=p_obs,
            omnibus_rms_cell_deviation=rms,
            plays=len(d),
            cells=int(f.n_c),
            seconds=time.time() - t0,
        )
        print(name, json.dumps(res[name]), flush=True)
    res["_meta"] = dict(  # noqa: C408
        gamma_fumble_lost=gamma,
        boot=BOOT,
        plant=PLANT,
        cells_planted=NPLANT,
        looks=3 * 18 + 3,
        relevant=RELEVANT,
        old_instrument_detectable_power_0_47=0.0454,
        old_unit_median_se=OLD_UNIT,
    )
    (m.OUT / "power_summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
