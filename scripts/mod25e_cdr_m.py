import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import mod25e_endgame as eg

LAB = "crHpqokgndecsmfwtjo2as2ypw2"
T = eg.train_frame()
T = T[T.season <= 2019]
T["otu_f"] = (T.otu > 0).astype(int)
S = []
for sd in (11, 12, 13):
    for f in sorted(glob.glob(f"artifacts/mod25e3/e5_{LAB}_s{sd}/play_*_*.parquet")):
        w, s = (int(x) for x in Path(f).stem.split("_")[1:])
        if s >= 2:
            S.append(pd.read_parquet(f))
S = pd.concat(S, ignore_index=True)
S["otu_f"] = ((S.lo_otu - S.oto) > 0).astype(int) if "lo_otu" in S else 0
print(S[["oto", "lo_otu"]].describe().round(2).to_dict())
print(T[["oto", "otu"]].describe().round(2).to_dict())
S["otu_f"] = (S.lo_otu > 0).astype(int)
S["stp"] = (S.flip.astype(bool) | ((S.po + S.pdf) > 0) | ((S.code == 1) & (S.yards == 0))).astype(int)
BANDS = [0, 10, 20, 30, 45, 60, 90, 120]


def st(d):
    return d[(d.qtr == 4) & (d.gsr > 0) & (d.gsr <= 120) & (d.sd <= 0) & (d.sd >= -2) & (d.yl <= 50) & (d.down <= 3) & d.code.isin([0, 1, 4, 5])].copy()


for nm, X in (("real", st(T)), ("sim", st(S))):
    X["b"] = pd.cut(X.gsr, BANDS)
    X["hasto"] = X.oto > 0
    X["rem"] = (X.gsr - X.el).clip(lower=0)
    print("==", nm, len(X))
    g = X.groupby("b", observed=True).apply(lambda z: pd.Series({
        "n": len(z), "p_otu": z.otu_f.mean(), "p_otu|hasto": z[z.hasto].otu_f.mean(), "run": (z.code == 0).mean(), "pass": (z.code == 1).mean(),
        "kneel": (z.code == 4).mean(), "spike": (z.code == 5).mean(), "stop": z.stp.mean(), "el_med": z.el.median(), "el_mean": z.el.mean(),
        "el_nostop_hasto": z[(z.stp == 0) & z.hasto & (z.code.isin([0, 1]))].el.mean(),
        "rem<=5": (z.rem <= 5).mean()}), include_groups=False)
    print(g.round(3).to_string())
    z = X[X.otu_f == 1]
    print("otu rows by prev gsr band: n", len(z), "gsr rem after play q10/25/50/75/90", z.rem.quantile([.1, .25, .5, .75, .9]).round(1).tolist())
    print("pstop of timeout plays", z.pstop.mean().round(3) if "pstop" in z else "")
print("######## by otu within band, plays code 0/1 with oto>0")
for nm, X in (("real", st(T)), ("sim", st(S))):
    X["b"] = pd.cut(X.gsr, BANDS)
    X["rem"] = (X.gsr - X.el).clip(lower=0)
    X = X[(X.oto > 0) & X.code.isin([0, 1])]
    print("==", nm)
    for o in (0, 1):
        z = X[X.otu_f == o]
        g = z.groupby("b", observed=True).agg(n=("el", "size"), el=("el", "mean"), elmed=("el", "median"), rem_mean=("rem", "mean"), rem5=("rem", lambda r: (r <= 5).mean()), rem15=("rem", lambda r: (r <= 15).mean()))
        print("otu", o); print(g.round(2).to_string())
