import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_endgame as eg  # noqa: E402
import sim09_f2 as f2  # noqa: E402

A = REPO / "artifacts" / "mod25e3"
VARS = {"crH": A / "crH" / "play6", "crHh": A / "hurry" / "play_h"}
LO, HI = 30.0, 120.0


def prep(d, real, cut):
    d = d.copy()
    if real:
        d = d.rename(columns={"game_id": "gk"})
        d["gk"] = pd.factorize(d["gk"])[0]
        d = d.sort_values(["gk", "play_id"], kind="stable")
        gk = "gk"
        oto = d["oto"]
    else:
        gk = "g"
        oto = d["oto_sim"]
    d["oto_"] = oto.to_numpy()
    d["gs"] = d.groupby(gk)["stop"].shift(1)
    hur = d["stop"] | (d["code"].isin([0, 1]) & (d["el"] <= cut))
    d["hs_prev"] = hur.groupby(d[gk]).shift(1)
    d["pq"] = d.groupby(gk)["qtr"].shift(1)
    d["ph"] = d.groupby(gk)["gsr"].shift(1)
    return d


def snaps(d):
    m = (d.qtr == 4) & (d.gsr > LO) & (d.gsr <= HI) & (d.sd <= 0) & d.code.isin([0, 1]) & (d.down <= 4) & (d.pq == 4) & d.gs.notna()
    return d[m]


def rate(x):
    return f"{(x.otu > 0).mean():.3f} (n {len(x)})"


def left_at(d, t, gk):
    q = d[(d.qtr == 4) & (d.sd <= 0) & (d.gsr <= t) & (d.sd < 0)]
    f = q.groupby(gk).head(1)
    return f"{f.oto_.mean():.2f} n {len(f)}"


def main():
    R0 = f2.real_frame(range(2009, 2018))
    cut = eg.hurry_cut(R0["code"], R0["yards"], R0["el"])
    R = prep(R0, True, cut)
    out = {"real": (R, "gk")}
    for k, p in VARS.items():
        out[k] = (prep(f2.sim_frame(str(p)), False, cut), "g")
    print("hurry cut", cut)
    for nm, (d, gk) in out.items():
        s = snaps(d)
        print(f"== {nm} snaps {len(s)} otu rate {rate(s)} mean oto {s.oto_.mean():.2f}")
        for a, lab in ((0, "prev f2-stopped"), (1, "prev f2-running")):
            x = s[s.gs.astype(float) == 1 - a]
            print(f"  {lab}: " + " | ".join(f"oto{o}: {rate(x[(x.oto_.clip(0, 2)) == o])}" for o in (0, 1, 2)) + f" | all {rate(x)}")
        for a, lab in ((0, "prev hurry-stopped"), (1, "prev hurry-running")):
            x = s[s.hs_prev.astype(float) == 1 - a]
            print(f"  {lab}: " + " | ".join(f"oto{o}: {rate(x[(x.oto_.clip(0, 2)) == o])}" for o in (0, 1, 2)) + f" | all {rate(x)}")
        h = s[s.oto_ > 0]
        for lo, hi, lab in ((-1e9, -9, "trail>8"), (-8, -4, "trail 4-8"), (-3, -1, "trail 1-3"), (0, 0, "tied")):
            x = h[(h.sd >= lo) & (h.sd <= hi)]
            print(f"  oto>0 {lab}: {rate(x)}")
        print("  oto>0 by gsr bin: " + " | ".join(f"({a},{b}] {rate(h[(h.gsr > a) & (h.gsr <= b)])}" for a, b in ((30, 45), (45, 60), (60, 90), (90, 120))) + " | by el_prev<=cut running " + rate(h[h.hs_prev.astype(float) == 0]))
        print("  oto left at first Q4 trailing snap with gsr<=60/30/15: " + " | ".join(left_at(d, t, gk) for t in (60, 30, 15)))


main()
