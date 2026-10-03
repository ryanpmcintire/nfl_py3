import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("PASS_MODE", "epa")

import mod25d_variance as dv  # noqa: E402
import mod25e_crH as H  # noqa: E402
import mod25e_crG as crG  # noqa: E402
import sim04_engine as sim  # noqa: E402

OUT = REPO / "artifacts" / "mod25e3"
STAGES = ("team", "ipw", "tilt", "dk", "epa")


def build():
    cfg = H.flags("")
    cfg.update(condition=1, def_sign=1.0, yard_bias=0.0)
    dv._d_init(dv.TRAIN, json.dumps(cfg))
    return dv._G, cfg


def stage_weights(G, nb, row, key, m):
    ns, t = G["ns"], G["tables"]
    a = t["arrays"]
    off_sim, def_sim, home = a["off_row"][row], a["def_row"][row], a["is_home_off"][row]
    h = t["team_kernel_h"]
    d2 = (a["off_row"][nb] - off_sim) ** 2 + (a["def_row"][nb] - def_sim) ** 2
    w = {}
    w["team"] = np.exp(-d2 / (2.0 * h * h)) * np.where(a["is_home_off"][nb] == home, sim.TEAM_KERNEL_LAMBDA, 1.0)
    w["ipw"] = ns["IPW"][nb]
    w["tilt"] = np.exp(ns["TILT_T"][nb] @ (ns["TILT_AO"] * (off_sim - ns["TILT_LO"]) + ns["TILT_AD"] * (def_sim - ns["TILT_LD"])))
    w["dk"] = np.exp(-0.5 * ((ns["DISTR"][nb] - key[2] * ns["DKR"]) / (ns["DKH"] * max(1.0, key[2] * ns["DKR"]) ** 0.5)) ** 2)
    epa = np.asarray(t["attrs"]["epa"], dtype=np.float64)
    epa = epa - float(epa.mean())
    s = (off_sim - ns["TILT_LO"]) + (def_sim - ns["TILT_LD"])
    w["epa"] = np.exp(m * s * epa[nb]) if m else np.ones(len(nb))
    return w


def neighbors_of(G, row, k):
    t = G["tables"]
    a = t["arrays"]
    down = int(a["down_i"][row])
    phase = int(G["phase"][row])
    entry = t["nn_trees_cond"].get((down, phase)) or t["nn_trees_cond"][(down, 0)]
    tree, sub = entry
    feat = sim.feature_matrix(np.array([a["dist_raw"][row]]), np.array([a["fp_raw"][row]]), np.array([G["sc"][row]]), np.array([G["tm"][row]]), np.array([G["oto"][row]]), np.array([G["dto"][row]]), np.array([phase]))[0]
    _, ind = tree.query(feat, k=min(k, len(sub)))
    nb = sub[np.atleast_1d(ind)]
    key = G["ns"]["round_state_key"](down, phase, a["dist_raw"][row], a["fp_raw"][row], G["sc"][row], G["tm"][row], G["oto"][row], G["dto"][row])
    return nb, key


def main():
    G, cfg = build()
    crG_m = float(crG.EPA_M)
    pbp = sim.load_reg_seasons(tuple(dv.TRAIN))
    trans = sim.build_transition_frame(pbp)
    G["phase"] = trans["phase"].to_numpy()
    G["sc"] = trans["sc_raw"].to_numpy()
    G["tm"] = trans["time_raw"].to_numpy()
    G["oto"] = trans["off_to_raw"].to_numpy()
    G["dto"] = trans["def_to_raw"].to_numpy()
    a = G["tables"]["arrays"]
    yds = a["yards_gained"].astype(float)
    code = a["play_type_code"]
    rp = np.isin(code, (0, 1))
    sel = np.flatnonzero((a["down_i"] == 1) & (np.abs(a["dist_raw"] - 10) < 0.5))
    rng = np.random.default_rng(7)
    sel = rng.choice(sel, size=min(int(os.environ.get("NQ", 4000)), len(sel)), replace=False)
    bands = [(0, 40), (40, 63), (63, 77), (77, 100)]
    cum = {b: {} for b in bands}
    loo = {b: {} for b in bands}
    real = {b: [] for b in bands}
    ess = {b: {} for b in bands}
    orders = {"u": ()}
    for i in range(len(STAGES)):
        orders["+" + STAGES[i]] = STAGES[: i + 1]
    for s_ in STAGES:
        orders["-" + s_] = tuple(x for x in STAGES if x != s_)
    for row in sel:
        fp = a["fp_raw"][row]
        b = next(x for x in bands if x[0] <= fp < x[1])
        nb, key = neighbors_of(G, row, sim.K_STATE)
        nb = nb[nb != row]
        w = stage_weights(G, nb, row, key, crG_m)
        real[b].append(yds[row])
        for nm, st in orders.items():
            ww = np.ones(len(nb))
            for s_ in st:
                ww = ww * w[s_]
            ww = ww / ww.sum()
            ym = float((ww * yds[nb]).sum())
            cum[b].setdefault(nm, []).append(ym)
            ess[b].setdefault(nm, []).append(1.0 / float((ww**2).sum()))
    lines = []
    for b in bands:
        r = np.array(real[b])
        lines.append(f"fp {b} n {len(r)} real mean yards {r.mean():.3f}")
        for nm in orders:
            v = np.array(cum[b][nm])
            lines.append(f"  {nm:7s} drawn mean {v.mean():.3f} minus real {v.mean() - r.mean():+.3f} ess {np.mean(ess[b][nm]):.1f}")
    ks = [int(x) for x in os.environ.get("KS", "25,50,100,200,400").split(",")]
    sub = sel[: int(os.environ.get("NK", 1500))]
    for K in ks:
        acc = {b: {"u": [], "f": [], "r": [], "dfp": [], "sdfp": []} for b in bands}
        for row in sub:
            fp = a["fp_raw"][row]
            b = next(x for x in bands if x[0] <= fp < x[1])
            nb, key = neighbors_of(G, row, K)
            nb = nb[nb != row]
            w = stage_weights(G, nb, row, key, crG_m)
            ww = np.ones(len(nb))
            for s_ in STAGES:
                ww = ww * w[s_]
            ww = ww / ww.sum()
            acc[b]["u"].append(float(yds[nb].mean()))
            acc[b]["f"].append(float((ww * yds[nb]).sum()))
            acc[b]["r"].append(yds[row])
            acc[b]["dfp"].append(float((ww * a["fp_raw"][nb]).sum() - fp))
            acc[b]["sdfp"].append(float(a["fp_raw"][nb].std()))
        for b in bands:
            v = acc[b]
            lines.append(f"K {K} fp {b} n {len(v['r'])} real {np.mean(v['r']):.3f} uniform-real {np.mean(v['u']) - np.mean(v['r']):+.3f} full-real {np.mean(v['f']) - np.mean(v['r']):+.3f} mean fp offset {np.mean(v['dfp']):+.2f} nb fp sd {np.mean(v['sdfp']):.2f}")
    txt = "\n".join(lines)
    (OUT / "draw.log").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
