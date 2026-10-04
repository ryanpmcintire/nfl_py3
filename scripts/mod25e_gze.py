import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "artifacts" / "mod25e3" / "gze"
FIT = OUT / "fit.json"
REAL = REPO / "artifacts" / "sim09" / "f2" / "real_fit.parquet"
MGRID = (20, 80, 320, 640, 1280, 2560)
BANDS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25))
SALT = 8129
PFLOOR = 0.005


def enabled():
    return os.environ.get("GZE") == "1"


def late_frame():
    d = pd.read_parquet(REAL)
    d = d[d.code.isin([0, 1]) & d.down.isin([1, 2, 3]) & (((d.qtr == 4) & (d.gsr <= 600)) | ((d.qtr == 2) & (d.gsr <= 1980)))].copy()
    d["hs"] = np.where(d.qtr == 2, d.gsr - 1800.0, d.gsr)
    d["td"] = (d.po >= 6).astype(float)
    d["sdc"] = d.sd.clip(-24, 24)
    return d.reset_index(drop=True)


def pool_ix(tr, q, need):
    a = (tr.down.to_numpy() == q.down) & (tr.code.to_numpy() == q.code)
    sel = a & (tr.qtr.to_numpy() == q.qtr)
    if sel.sum() < need:
        sel = a
    return np.flatnonzero(sel)


def old_p(tr, q, sd4, z4):
    ix = pool_ix(tr, q, 8)
    qv = np.array([q.dist, q.yl, q.sdc, q.hs]) / sd4
    d = ((z4[ix] - qv) ** 2).sum(axis=1)
    k = max(int(np.sqrt(len(ix))), 1)
    near = ix[np.argpartition(d, k - 1)[:k]] if k < len(ix) else ix
    gain = tr.yards.to_numpy()[near]
    flip = tr.flip.to_numpy()[near].astype(bool)
    return float(((gain >= q.yl) & ~flip).mean()), float((np.round(tr.yl.to_numpy()[near]) == round(q.yl)).mean())


def new_p(tr, q, sd3, z3, m):
    ix = pool_ix(tr, q, m)
    gap = np.abs(tr.yl.to_numpy()[ix] - q.yl)
    w = np.sort(gap)[min(m, len(gap)) - 1]
    ix = ix[gap <= w]
    qv = np.array([q.dist, q.sdc, q.hs]) / sd3
    d = ((z3[ix] - qv) ** 2).sum(axis=1)
    k = max(int(np.sqrt(len(ix))), 1)
    near = ix[np.argpartition(d, k - 1)[:k]] if k < len(ix) else ix
    return float(tr.td.to_numpy()[near].mean()), float((np.round(tr.yl.to_numpy()[near]) == round(q.yl)).mean())


def replay(L, m_list, test_mask):
    c4 = ["dist", "yl", "sdc", "hs"]
    c3 = ["dist", "sdc", "hs"]
    rows = []
    for s in sorted(L.season.unique()):
        tr = L[L.season != s].reset_index(drop=True)
        te = L[(L.season == s) & test_mask]
        sd4 = tr[c4].to_numpy(float).std(axis=0)
        sd3 = tr[c3].to_numpy(float).std(axis=0)
        z4 = tr[c4].to_numpy(float) / sd4
        z3 = tr[c3].to_numpy(float) / sd3
        for q in te.itertuples():
            o, ox = old_p(tr, q, sd4, z4)
            r = {"season": s, "yl": q.yl, "hs": q.hs, "td": q.td, "old": o, "old_ex": ox}
            for m in m_list:
                n, nx = new_p(tr, q, sd3, z3, m)
                r[f"new{m}"] = n
                r[f"nx{m}"] = nx
            rows.append(r)
    return pd.DataFrame(rows)


def nll(p, y):
    p = np.clip(np.asarray(p, float), PFLOOR, 1 - PFLOOR)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit():
    OUT.mkdir(parents=True, exist_ok=True)
    L = late_frame()
    mask = (L.yl <= BANDS[-1][1]) & (L.hs <= 300)
    R = replay(L, MGRID, mask)
    cols = ["old"] + [f"new{m}" for m in MGRID]
    out = {"m_grid": list(MGRID), "n_test": int(len(R)), "bands": {}}
    lines = []
    for lo, hi in BANDS:
        b = R[(R.yl > lo) & (R.yl <= hi)]
        res = {c: {"ll": float(nll(b[c], b.td.to_numpy()).mean()), "folds": [float(nll(g[c], g.td.to_numpy()).mean()) for _, g in b.groupby("season")]} for c in cols}
        out["bands"][f"{lo}-{hi}"] = {"n": int(len(b)), "res": res, "td": float(b.td.mean())}
        lines.append(f"yl {lo}-{hi} n {len(b)} real TD {b.td.mean():.3f} " + " ".join(f"{c} {res[c]['ll']:.4f}" for c in cols))
    tot = {c: float(nll(R[c], R.td.to_numpy()).mean()) for c in cols}
    best = min(MGRID, key=lambda m: tot[f"new{m}"])
    wins = {}
    yb = 0
    for lo, hi in BANDS:
        k = f"{lo}-{hi}"
        o = np.array(out["bands"][k]["res"]["old"]["folds"])
        n = np.array(out["bands"][k]["res"][f"new{best}"]["folds"])
        wins[k] = f"{int((n < o).sum())}/{len(o)}"
        if (n < o).sum() * 2 > len(o) and yb == lo:
            yb = hi
    out.update({"m": int(best), "total_ll": tot, "fold_wins_new_vs_old": wins, "yb": int(yb), "looks": len(MGRID) * len(BANDS) + len(BANDS)})
    FIT.write_text(json.dumps(out), encoding="utf-8")
    R.to_parquet(OUT / "replay.parquet")
    print("\n".join(lines))
    print("total", tot, "best m", best, "fold wins", wins, "yb", yb)


def validate():
    spec = json.loads(FIT.read_text(encoding="utf-8"))
    R = pd.read_parquet(OUT / "replay.parquet")
    m = spec["m"]
    out = []
    for nm, lo, hi in (("hs<=15", 0, 15), ("hs16-60", 15, 60)):
        for zn, zl, zh in (("yl<=10", 0, 10), ("yl11-25", 10, 25)):
            b = R[(R.hs > lo) & (R.hs <= hi) & (R.yl > zl) & (R.yl <= zh)]
            out.append(f"{nm} {zn} n {len(b)} real TD/play {b.td.mean():.3f} old {b.old.mean():.3f} new(m={m}) {b[f'new{m}'].mean():.3f} exact-yl share of neighbours old {b.old_ex.mean():.3f} new {b[f'nx{m}'].mean():.3f}")
    (OUT / "validate.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


def install_gze():
    import mod25d_variance as dv
    import mod25e_draw as dr
    import mod25e_endgame as eg

    spec = json.loads(FIT.read_text(encoding="utf-8"))
    m, yb = int(spec["m"]), float(spec["yb"])
    ns = dv._G["ns"]
    seed = int(dv._G["cfg"].get("seed", 3))
    wide = os.environ.get("KNW") == "1"
    T = eg.train_frame()
    L = T[((T.qtr == 4) & (T.gsr <= 600)) | ((T.qtr == 2) & (T.gsr <= 1980))]
    L = L[(L.down <= 3) & L.code.isin([0, 1])].reset_index(drop=True)
    Lyl = L.yl.to_numpy(float)
    Ld = L.down.to_numpy()
    Lc = L.code.to_numpy()
    Lq = L.qtr.to_numpy()
    Lpos = L.pos.to_numpy().astype(int)
    P = np.column_stack([L.dist, np.clip(L.sd, -24, 24), L.hs]).astype(float)
    psd = P.std(axis=0)
    psd[psd == 0] = 1.0
    P = P / psd
    arrays = dv._G["tables"]["arrays"]
    st = {"k": None, "rng": None}
    dec = ns["DECIDE"]

    def inwin(qtr, clock_val):
        return dr.in_window(qtr, clock_val) or (wide and ((qtr == 4 and 0 < clock_val <= 600.0) or (qtr == 2 and 1800.0 < clock_val <= 1980.0)))

    def decide(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, *rest):
        r = dec(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, *rest)
        if in_ot or down not in (1, 2, 3) or not inwin(qtr, clock_val) or yardline > yb:
            return r
        c = int(arrays["play_type_code"][r])
        if c not in (0, 1):
            return r
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, seed)
        a = (Ld == down) & (Lc == c)
        sel = np.flatnonzero(a & (Lq == qtr))
        if len(sel) < m:
            sel = np.flatnonzero(a)
        if len(sel) < 1:
            return r
        gap = np.abs(Lyl[sel] - yardline)
        w = np.sort(gap)[min(m, len(gap)) - 1]
        sel = sel[gap <= w]
        hs = clock_val - 1800.0 if qtr == 2 else clock_val
        qv = np.array([distance, min(max(score_diff, -24.0), 24.0), hs]) / psd
        d = ((P[sel] - qv) ** 2).sum(axis=1)
        k = max(int(np.sqrt(len(sel))), 1)
        near = sel[np.argpartition(d, k - 1)[:k]] if k < len(sel) else sel
        return int(Lpos[int(near[int(st["rng"].integers(0, len(near)))])])

    ns["DECIDE"] = decide


if __name__ == "__main__":
    {"fit": fit, "validate": validate}[sys.argv[1]]()
