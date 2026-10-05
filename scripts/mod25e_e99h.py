import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import mod25e_endgame as eg
T = eg.train_frame()
L = T[((T.qtr == 4) & (T.gsr <= 600)) | ((T.qtr == 2) & (T.gsr <= 1980))]
L = L[(L.down <= 3) & L.code.isin(eg.LATE_CODES)].reset_index(drop=True)
y = L.code.map(eg.CLS).to_numpy()
F = lambda d: eg.dec_feats(d.down, d.dist, d.yl, d.sd, d.gsr, d.hs, (d.qtr == 4).astype(float), d.oto, d.dto, d.pstop)
clf = eg.fit_hgb(F(L), y)
P = clf.predict_proba(F(L))
L["pfg"] = P[:, 2]
L["fg"] = (y == 2).astype(float)
def sel(d):
    return d[(d.qtr == 4) & (d.gsr <= 300) & (d.sd <= 0) & (d.sd >= -2) & (d.yl <= 40) & (d.down <= 3)]
def band(d):
    return pd.cut(d.gsr, [0, 30, 120, 300])
R = sel(L)
S = []
for sd in (11, 12, 13):
    for f in sorted(glob.glob(f"artifacts/mod25e3/e5_crHpqokgndecsmfwtjo2as2ypw2_s{sd}/play_*_*.parquet")):
        w, s = (int(x) for x in Path(f).stem.split("_")[1:])
        if s < 2:
            continue
        d = pd.read_parquet(f)
        d = d[d.code.isin(eg.LATE_CODES)]
        d = d[(d.qtr == 4) & (d.gsr <= 300) & (d.down <= 3)].copy()
        d["hs"] = d.gsr
        S.append(d)
S = pd.concat(S, ignore_index=True)
S["pfg"] = clf.predict_proba(F(S))[:, 2]
S["fg"] = (S.code == 3).astype(float)
S = sel(S)
print("real n", len(R), "sim n", len(S))
for dn in (1, 2, 3):
    for nm, X in (("real", R), ("sim", S)):
        g = X[X.down == dn].groupby(band(X[X.down == dn]), observed=True).agg(n=("fg", "size"), obs=("fg", "mean"), hgb=("pfg", "mean"))
        print(dn, nm, g.round(3).to_dict("index"))
L.to_pickle("C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/e9fe6e6d-b0ac-425e-968d-4ee8497028b8/scratchpad/L.pkl")
print("---features gsr<=30 down 2-3")
for nm, X in (("real", R), ("sim", S)):
    z = X[(X.gsr <= 30) & (X.down >= 2)]
    print(nm, len(z), z[["gsr", "yl", "dist", "sd", "oto", "dto", "pstop", "pfg", "fg"]].mean().round(3).to_dict())
    print(nm, "gsr q", z.gsr.quantile([.1, .25, .5, .75, .9]).round(1).tolist(), "sd==0", (z.sd == 0).mean().round(3), "oto>0", (z.oto > 0).mean().round(3), "yl<=30", (z.yl <= 30).mean().round(3))
    for lo, hi in ((0, 5), (5, 15), (15, 30)):
        zz = z[(z.gsr > lo) & (z.gsr <= hi)]
        print(nm, lo, hi, len(zz), zz[["yl", "oto", "pstop", "pfg", "fg"]].mean().round(3).to_dict())
print("---el of go plays")
Tq = T[(T.qtr == 4) & (T.gsr <= 300) & (T.sd <= 0) & (T.sd >= -2) & (T.yl <= 40) & T.code.isin([0, 1]) & (T.down <= 3)]
S2 = []
for sd in (11, 12, 13):
    for f in sorted(glob.glob(f"artifacts/mod25e3/e5_crHpqokgndecsmfwtjo2as2ypw2_s{sd}/play_*_*.parquet")):
        w, s = (int(x) for x in Path(f).stem.split("_")[1:])
        if s < 2:
            continue
        d = pd.read_parquet(f)
        d = d[(d.qtr == 4) & (d.gsr <= 300) & (d.sd <= 0) & (d.sd >= -2) & (d.yl <= 40) & d.code.isin([0, 1]) & (d.down <= 3)]
        S2.append(d)
S2 = pd.concat(S2, ignore_index=True)
for lo, hi in ((30, 60), (60, 120), (120, 300)):
    for nm, X in (("real", Tq), ("sim", S2)):
        z = X[(X.gsr > lo) & (X.gsr <= hi)]
        rem = (z.gsr - z.el).clip(lower=0)
        print(lo, hi, nm, len(z), "el mean", round(z.el.mean(), 2), "q25/50/75", z.el.quantile([.25, .5, .75]).round(1).tolist(), "rem<=5", round((rem <= 5).mean(), 3), "rem<=30", round((rem <= 30).mean(), 3), "stopflag", round(z.pstop.mean(), 3) if "pstop" in z else "")
print("---go plays gsr<=30 (all downs<=3), by oto>0")
def q(d):
    return d[(d.qtr == 4) & (d.gsr <= 30) & (d.sd <= 0) & (d.sd >= -2) & (d.yl <= 40) & d.code.isin([0, 1, 4, 5]) & (d.down <= 3)]
S4 = []
for sd in (11, 12, 13):
    for f in sorted(glob.glob(f"artifacts/mod25e3/e5_crHpqokgndecsmfwtjo2as2ypw2_s{sd}/play_*_*.parquet")):
        w, s = (int(x) for x in Path(f).stem.split("_")[1:])
        if s >= 2:
            S4.append(q(pd.read_parquet(f)))
S4 = pd.concat(S4, ignore_index=True)
Tq2 = q(T)
for nm, X in (("real", Tq2), ("sim", S4)):
    for ct in (0, 1, 4, 5):
        z = X[X.code == ct] if ct != 1 else X[X.code.isin([0, 1])]
        if ct == 0:
            continue
        for lo, hi in ((0, 10), (10, 20), (20, 30)):
            zz = z[(z.gsr > lo) & (z.gsr <= hi)]
            print(nm, "code", ct, lo, hi, len(zz), "el", round(zz.el.mean(), 1), "frac down1", round((zz.down == 1).mean(), 2), "oto>0", round((zz.oto > 0).mean(), 2))
    print(nm, "kneel/spike share by gsr(0,30] ", round(X.code.isin([4, 5]).mean(), 3), "n", len(X))
print("---class shares gsr<=10 by down")
for nm, X in (("real", Tq2), ("sim", S4)):
    X = X.copy()
    X["hs"] = X.gsr
    X["pstop"] = X.pstop if "pstop" in X else 1.0
    pr = clf.predict_proba(F(X))
    for dn in (1, 2, 3):
        for lo, hi in ((0, 5), (5, 10), (10, 30)):
            m = ((X.down == dn) & (X.gsr > lo) & (X.gsr <= hi)).to_numpy()
            if m.sum() == 0:
                continue
            obs = [round(float((X.code[m] == c).mean()), 3) for c in (1, 3, 4, 5)]
            ex = [round(float(pr[m][:, k].mean()), 3) for k in (0, 1, 2, 3, 4)]
            print(nm, dn, lo, hi, int(m.sum()), "obs run/pass(0,1 merged) fg kneel spike [code1,3,4,5]", obs, "hgb cls0..4", ex)
print("---transition: next snap gsr<=5 given cur gsr in band, tied/trail<=2, Q4, yl<=40, cur code in go/spike/kneel")
def nxt(d, key):
    d = d.sort_values(key, kind="stable").reset_index(drop=True)
    d["ng"] = d.groupby("g").gsr.shift(-1)
    d["nsd"] = d.groupby("g").sd.shift(-1)
    return d
Tn = nxt(T[T.qtr == 4], ["g", "play_id"])
S5 = []
for sd in (11, 12, 13):
    for f in sorted(glob.glob(f"artifacts/mod25e3/e5_crHpqokgndecsmfwtjo2as2ypw2_s{sd}/play_*_*.parquet")):
        w, s = (int(x) for x in Path(f).stem.split("_")[1:])
        if s >= 2:
            d = pd.read_parquet(f)
            d = d[d.qtr == 4].copy()
            d["g"] = d.g + 1000000 * (w * 10 + s) + sd * 1e9
            S5.append(d)
S5 = pd.concat(S5, ignore_index=True)
print("sim order check", (S5.groupby("g").gsr.apply(lambda s: (s.diff().dropna() <= 0).mean())).mean().round(3))
Sn = nxt(S5, ["g"]) if False else None
S5["o"] = np.arange(len(S5))
Sn = S5.copy()
Sn["ng"] = Sn.groupby("g").gsr.shift(-1)
for nm, X in (("real", Tn), ("sim", Sn)):
    z = X[(X.sd <= 0) & (X.sd >= -2) & (X.yl <= 40) & X.code.isin([0, 1, 4, 5]) & X.ng.notna()]
    for lo, hi in ((5, 15), (15, 30), (30, 60), (60, 120)):
        zz = z[(z.gsr > lo) & (z.gsr <= hi)]
        print(nm, lo, hi, len(zz), "P(next gsr<=5)", round((zz.ng <= 5).mean(), 3), "P(next<=10)", round((zz.ng <= 10).mean(), 3), "median el", round(zz.el.median(), 1))
print("---pool emulation")
R0 = pd.read_parquet("artifacts/sim09/f2/real_fit.parquet")
print(R0.columns.tolist()[:40], len(R0), R0.gsr.max())
R = R0[(R0.qtr == 4) & (R0.gsr > 0) & (R0.gsr <= 120)].reset_index(drop=True)
code = R.code.to_numpy(); cl = (R.otu > 0).astype(int) + 2 * (R.dtu > 0).astype(int)
sc = (R.po + R.pdf) > 0
stp = (R.flip.astype(bool) | sc | ((code == 1) & (R.yards == 0))).astype(int).to_numpy()
cls = cl.to_numpy()
Fz = np.column_stack([R.gsr, np.clip(R.sd, -24, 24), (R.oto > 0).astype(float), (R.dto > 0).astype(float)]).astype(float)
sdv = Fz.std(axis=0); Z = Fz / sdv
el = R.el.to_numpy(float)
print("pool n", len(R), "sdv", sdv.round(2))
Sg = S5[(S5.gsr <= 120) & (S5.gsr > 15) & (S5.sd <= 0) & (S5.sd >= -2) & (S5.yl <= 40) & S5.code.isin([0, 1])].sample(2000, random_state=1)
rg = np.random.default_rng(5)
em, ac, rl = [], [], []
for _, r in Sg.iterrows():
    c = int(r.code); st_ = int(bool(r.flip) or (r.po + r.pdf) > 0 or (c == 1 and r.yards == 0)); c_ = int(r.lo_otu > 0) + 2 * int(r.lo_dtu > 0)
    m0 = code == c
    sel = None
    for m in (m0 & (stp == st_) & (cls == c_), m0 & (stp == st_), m0):
        ix = np.flatnonzero(m)
        if len(ix) >= 30:
            sel = ix; break
    q = np.array([r.gsr, min(max(r.sd, -24), 24), float(r.oto > 0), float(r.dto > 0)]) / sdv
    d = ((Z[sel] - q) ** 2).sum(axis=1); k = max(int(np.sqrt(len(sel))), 1)
    near = sel[np.argpartition(d, k - 1)[:k]]
    em.append(el[int(near[rg.integers(0, len(near))])]); ac.append(r.el)
print("emulated el median", np.median(em), "mean", np.mean(em), "actual sim el median", np.median(ac), "mean", np.mean(ac))
rr = Tn[(Tn.gsr <= 120) & (Tn.gsr > 15) & (Tn.sd <= 0) & (Tn.sd >= -2) & (Tn.yl <= 40) & Tn.code.isin([0, 1])]
print("real in-state el median", rr.el.median(), "mean", rr.el.mean(), "n", len(rr), "stop share real", rr.stop.mean(), "sim", ((Sg.flip > 0) | (Sg.po + Sg.pdf > 0)).mean())
