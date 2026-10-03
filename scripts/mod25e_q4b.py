import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_endgame as eg  # noqa: E402

D = REPO / "artifacts" / "mod25e3" / "q4" / "log1"
S = pd.concat([pd.read_parquet(p) for p in sorted(D.glob("q4log_*.parquet"))])
T = eg.train_frame()
L = T[((T.qtr == 4) & (T.gsr <= 600)) | ((T.qtr == 2) & (T.gsr <= 1980))]
L = L[(L.down <= 3) & L.code.isin(eg.LATE_CODES)].reset_index(drop=True)
yL = L.code.map(eg.CLS).to_numpy()
F = lambda d, ps: eg.dec_feats(d.down.to_numpy(), d.dist.to_numpy(), d.yl.to_numpy(), d.sd.to_numpy(), d.gsr.to_numpy(), d.hs.to_numpy(), (d.qtr == 4).astype(float).to_numpy(), d.oto.to_numpy(), d.dto.to_numpy(), ps)
clf = eg.fit_hgb(F(L, L.pstop.to_numpy()), yL)
R = T[(T.down <= 3) & (T.qtr == 4) & (T.hs <= 120) & (T.hs > 0)].copy()
R = R[R.code.isin(eg.LATE_CODES)]
S = S[S.qtr == 4].copy()
S["gsr"] = S.hs
def win(d):
    return d[(d.sd >= -3) & (d.sd <= 0) & (d.yl <= 40)]
R["pfg"] = clf.predict_proba(F(R, R.pstop.to_numpy()))[:, 2]
R["pfg1"] = clf.predict_proba(F(R, np.ones(len(R))))[:, 2]
S["pfg1"] = clf.predict_proba(F(S, np.ones(len(S))))[:, 2]
r = win(R)
s = win(S)
print("window real n", len(r), "sim n", len(s))
for dk in (1, 2, 3):
    a = r[r.down == dk]
    b = s[s.down == dk]
    print(f"down {dk} real fg {np.mean(a.code == 3):.3f} pred(real pstop) {a.pfg.mean():.3f} pred(ps=1) {a.pfg1.mean():.3f} | sim c0 {np.mean(b.c0 == 3):.3f} c_eg {np.mean(b.c_eg == 3):.3f} pred(ps=1) {b.pfg1.mean():.3f}")
print("pstop real mean", r.pstop.mean())
for nm, d in (("real", r), ("sim", s)):
    d = d.copy()
    d["tb"] = pd.cut(d.hs, [0, 15, 30, 60, 120])
    d["to"] = np.minimum(d.oto, 3)
    d["yb"] = pd.cut(d.yl, [-1, 20, 30, 40]) if d.yl.max() > 1 else 0
    print(nm, "hs bins", d.groupby("tb", observed=True).size().div(len(d)).round(3).to_dict())
    print(nm, "oto", d.groupby("to").size().div(len(d)).round(3).to_dict())
    print(nm, "yl bins", d.groupby("yb", observed=True).size().div(len(d)).round(3).to_dict())
    print(nm, "yl mean", d.yl.mean().round(1), "dist", d.dist.mean().round(2))
for nm, d, col in (("real", r, "code"), ("sim", s, "c_eg")):
    d = d.copy()
    d["tb"] = pd.cut(d.hs, [0, 15, 30, 60, 120])
    d["to"] = np.minimum(d.oto, 3)
    print(nm, "fg by hs", d.groupby("tb", observed=True)[col].apply(lambda v: round(float(np.mean(v == 3)), 3)).to_dict())
    print(nm, "fg by oto", d.groupby("to")[col].apply(lambda v: round(float(np.mean(v == 3)), 3)).to_dict())
a = r[r.hs <= 60]; b = s[s.hs <= 60]
print("hs<=60 real fg", np.mean(a.code == 3), "pred", a.pfg1.mean(), "sim c_eg", np.mean(b.c_eg == 3), "pred", b.pfg1.mean(), "n", len(a), len(b))
print("sim c_eg vs c0 vs class draw: share where c_eg!=c0", np.mean(s.c_eg != s.c0))
print("---- matched bins hs<=15")
for nm, d, col in (("real", r, "code"), ("sim", s, "c_eg")):
    d = d[d.hs <= 15].copy()
    d["to"] = np.minimum(d.oto, 1)
    d["yb"] = pd.cut(d.yl, [-1, 10, 25, 40])
    d["sdb"] = d.sd.round()
    g = d.groupby("to")
    print(nm, "n", len(d), "by oto n", g.size().to_dict(), "fg", g[col].apply(lambda v: round(float(np.mean(v == 3)), 3)).to_dict(), "pred", g.pfg1.mean().round(3).to_dict())
    g = d.groupby("yb", observed=True)
    print(nm, "by yl n", g.size().to_dict(), "fg", g[col].apply(lambda v: round(float(np.mean(v == 3)), 3)).to_dict(), "pred", g.pfg1.mean().round(3).to_dict())
    g = d.groupby("down")
    print(nm, "by down fg", g[col].apply(lambda v: round(float(np.mean(v == 3)), 3)).to_dict(), "pred", g.pfg1.mean().round(3).to_dict(), "dto0", g.dto.apply(lambda v: round(float(np.mean(v == 0)), 2)).to_dict())
    print(nm, "hs mean", d.hs.mean().round(1), "sd mean", d.sd.mean().round(2), "dist", d.dist.mean().round(1))
    print(nm, "hs dist", np.histogram(d.hs, [0, 3, 6, 9, 12, 15])[0])
print("real pstop in hs<=15", r[r.hs <= 15].pstop.mean())
print("real pred actual pstop hs<=15", r[r.hs <= 15].pfg.mean(), "real fg", np.mean(r[r.hs <= 15].code == 3))
import mod25e_drill as dr
print("dr.in_window src check", dr.MIN_POOL)
