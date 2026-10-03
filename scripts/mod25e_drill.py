import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import sim09_f2 as f2  # noqa: E402

dv = f2.dv
OUT = REPO / "artifacts" / "mod25e3" / "drill"
SRC = {"crzhc": REPO / "artifacts" / "mod25e3" / "revert_sim" / "crzhc", "f2": REPO / "artifacts" / "sim09" / "f2" / "play_f2"}
META = ["game_id", "play_id", "complete_pass", "qb_spike", "qb_kneel", "penalty", "sack", "touchdown", "interception", "fumble_lost"]
GB = [0, 15, 45, 120, 300]
HB = [(0, 8), (8, 15), (15, 45), (45, 120), (120, 300)]


def meta_frame():
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = dv.sim.build_transition_frame(pbp)
    ex = pd.concat([pd.read_parquet(f"{dv.m25.SNAP}/season={y}/plays.parquet", columns=META) for y in dv.TRAIN]).drop_duplicates(["game_id", "play_id"])
    m = tr[["game_id", "play_id"]].merge(ex, on=["game_id", "play_id"], how="left")
    assert len(m) == len(tr)
    return m


def real_df():
    d = pd.read_parquet(REPO / "artifacts" / "sim09" / "f2" / "real_fit.parquet")
    m = meta_frame()
    d = d.drop(columns=[c for c in ("penalty", "qb_kneel", "qb_spike") if c in d.columns]).merge(m, on=["game_id", "play_id"], how="left")
    d["oto_"] = d["oto"]
    d["dto_"] = d["dto"]
    return d


def sim_df(path):
    S = f2.sim_frame(path)
    m = meta_frame()
    ix = S["idx"].to_numpy().astype(int)
    for c in META[2:]:
        if c == "penalty":
            continue
        S[c] = m[c].to_numpy()[ix]
    S["oto_"] = S["oto_sim"]
    S["dto_"] = S["dto_sim"]
    return S


def outcome(d):
    o = np.full(len(d), "other", dtype=object)
    c = d["code"].to_numpy()
    pen = d["penalty"].fillna(0).to_numpy() == 1
    inc = (c == 1) & (d["complete_pass"].fillna(0).to_numpy() != 1) & (d["sack"].fillna(0).to_numpy() != 1) & ~pen
    cp = d["complete_pass"].fillna(0).to_numpy() == 1
    sc = (d["po"].to_numpy() + d["pdf"].to_numpy()) > 0
    o[c == 0] = "run"
    o[(c == 1) & cp] = "pass_comp"
    o[(c == 1) & (d["sack"].fillna(0).to_numpy() == 1)] = "sack"
    o[(c == 1) & inc] = "pass_inc"
    o[c == 2] = "punt"
    o[c == 3] = "fg"
    o[c == 4] = "kneel"
    o[c == 5] = "spike"
    o[c == 6] = "no_play"
    o[(c == 1) & inc] = "pass_inc"
    o[pen & (c < 2)] = "penalty"
    o[sc & (c < 2)] = "score_play"
    o[d["flip"].to_numpy().astype(bool) & (c < 2) & ~sc] = "turnover_play"
    return o


def hurry(d):
    return (d["qtr"] == 4) & (d["gsr"] <= 300) & (d["sd"] <= 0) & (d["sd"] >= -8)


def an_one(d, name, L):
    d = d.copy()
    d["oc"] = outcome(d)
    h = d[hurry(d)]
    n = d["g"].nunique()
    L.append(f"== {name}: games {n}, hurry snaps (Q4, gsr<=300, offence tied or trailing<=8) {len(h)} = {len(h) / n:.2f}/game")
    L.append("  seconds per snap (el) by outcome: share, mean el, mean el|gsr>45")
    for o, g in h.groupby("oc"):
        a = g[g.gsr > 45]
        L.append(f"    {o:14s} share {len(g) / len(h):.3f} el {g.el.mean():5.1f}  gsr>45 el {a.el.mean() if len(a) else float('nan'):5.1f} (n {len(a)})")
    L.append("  by gsr band: snaps/game, spike, kneel, pass share, complete|pass, mean el, timeouts used by off after snap, el>gsr share")
    for lo, hi in HB:
        b = h[(h.gsr >= lo) & (h.gsr < hi)]
        if not len(b):
            continue
        p = b[b.code == 1]
        L.append(f"    [{lo:3d},{hi:3d}) {len(b) / n:5.2f} spike {np.mean(b.code == 5):.3f} kneel {np.mean(b.code == 4):.3f} pass {np.mean(b.code == 1):.3f} comp|pass {np.mean(p.complete_pass.fillna(0) == 1) if len(p) else float('nan'):.3f} el {b.el.mean():5.1f} to_used {np.mean(b.otu > 0):.3f} el>gsr {np.mean(b.el > b.gsr):.3f}")
    L.append("  FG attempt share of snaps at yl<=45 by gsr band x down x offence timeouts>0")
    f = h[h.yl <= 45]
    for lo, hi in ((0, 8), (8, 15), (15, 30), (30, 60), (60, 120)):
        for dn in (1, 2, 3, 4):
            for tt in (0, 1):
                b = f[(f.gsr >= lo) & (f.gsr < hi) & (f.down == dn) & ((f.oto_ > 0) == bool(tt))]
                if len(b) >= 25 * n / 1000:
                    L.append(f"    [{lo:3d},{hi:3d}) down {dn} to>0 {tt} n/game {len(b) / n * 100:6.2f}/100g fg {np.mean(b.code == 3):.3f} spike {np.mean(b.code == 5):.3f} kneel {np.mean(b.code == 4):.3f}")
    hm = h[(h.gsr <= 12) & (h.yl >= 30) & (h.code == 1)]
    L.append(f"  Hail Mary-like snaps (gsr<=12, yl>=30, pass) per 100 games {len(hm) / n * 100:.2f}; mean yards {hm.yards.mean():.1f}, mean el {hm.el.mean():.1f}")
    L.append("  last snap of hurry drives: time left at snap (gsr) and el, by whether the snap ends the game window")
    e = h[h.gsr <= 120]
    L.append(f"    el>=gsr (clock expires inside snap) share {np.mean(e.el >= e.gsr):.3f}; mean gsr remaining after snap {np.mean(np.clip(e.gsr - e.el, 0, None)):.1f}")
    s = h[h.gsr <= 120].copy()
    s["gk"] = s.g.astype(str)
    gl = s.groupby("gk").size()
    L.append(f"    snaps per game-with-hurry-in-last-2min: {gl.mean():.2f}")


def cmd_an(a):
    OUT.mkdir(parents=True, exist_ok=True)
    L = []
    an_one(real_df(), "real 2009-17 (pool transitions)", L)
    for k in a.sims.split(","):
        an_one(sim_df(SRC[k]), f"sim {k}", L)
    txt = "\n".join(L)
    (OUT / "an.txt").write_text(txt)
    print(txt)


dv.DV["drl"] = dict(dv.DV["f2"], dr=1)
dv.DV["drl0"] = dict(dv.DV["f2"], dr=0)
SALT = 4101
MIN_POOL = 8


def in_window(qtr, clock_val):
    return (qtr == 4 and 0 < clock_val <= f2.u4g.WARN_AT[4]) or (qtr == 2 and f2.u4g.FLOOR[2] < clock_val <= f2.u4g.WARN_AT[2])


def install_dr():
    if not int(dv._G["cfg"].get("dr", 0)):
        return
    R = pd.read_parquet(REPO / "artifacts" / "sim09" / "f2" / "real_fit.parquet")
    qt = R["qtr"].to_numpy()
    gs = R["gsr"].to_numpy(float)
    win = ((qt == 4) & (gs > 0) & (gs <= f2.u4g.WARN_AT[4])) | ((qt == 2) & (gs > f2.u4g.FLOOR[2]) & (gs <= f2.u4g.WARN_AT[2]))
    R = R[win].reset_index(drop=True)
    half = R["qtr"].to_numpy()
    code = R["code"].to_numpy()
    yd = R["yards"].to_numpy()
    sc = (R["po"].to_numpy() + R["pdf"].to_numpy()) > 0
    stp = (R["flip"].to_numpy().astype(bool) | sc | ((code == 1) & (yd == 0))).astype(int)
    cls = (R["otu"].to_numpy() > 0).astype(int) + 2 * (R["dtu"].to_numpy() > 0).astype(int)
    F = np.column_stack([R["gsr"].to_numpy(float), np.clip(R["sd"].to_numpy(float), -24, 24), (R["oto"].to_numpy() > 0).astype(float), (R["dto"].to_numpy() > 0).astype(float)])
    sdv = F.std(axis=0)
    sdv[sdv == 0] = 1.0
    Z = F / sdv
    el = R["el"].to_numpy(float)
    ns = dv._G["ns"]
    dec = ns["DECIDE"]
    st = {"to": (3.0, 3.0), "k": None, "rng": None}
    cseed = int(dv._G["cfg"].get("seed", 3))

    def decide(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, *rest):
        st["to"] = (float(off_to), float(def_to))
        return dec(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, *rest)

    ns["DECIDE"] = decide
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        if not in_window(qtr, clock_val):
            return base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, cseed)
        rng = st["rng"]
        c = int(drawn["play_type_code"])
        sc_ = (float(drawn["points_off"]) + float(drawn["points_def"])) > 0
        stp_ = int(bool(drawn["flip"]) or sc_ or (c == 1 and float(drawn["yards_gained"]) == 0))
        cl_ = int(float(drawn["off_to_used"]) > 0) + 2 * int(float(drawn["def_to_used"]) > 0)
        m0 = (half == qtr) & (code == c)
        sel = None
        for m in (m0 & (stp == stp_) & (cls == cl_), m0 & (stp == stp_), m0):
            idx = np.flatnonzero(m)
            if len(idx) >= MIN_POOL:
                sel = idx
                break
        if sel is None:
            return base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        q = np.array([clock_val, min(max(score_diff, -24.0), 24.0), float(st["to"][0] > 0), float(st["to"][1] > 0)]) / sdv
        d = ((Z[sel] - q) ** 2).sum(axis=1)
        k = max(int(np.sqrt(len(sel))), 1)
        near = sel[np.argpartition(d, k - 1)[:k]]
        j = int(near[int(rng.integers(0, len(near)))])
        drawn = dict(drawn)
        drawn["clock_elapsed"] = float(min(el[j], clock_val))
        return base(down, distance, yardline, score_diff, qtr, clock_val, drawn)

    dv._G["pol"] = pol


def drl_init_budget(setting):
    f2.f2_init_budget(setting)
    install_dr()


def drl_init_ss(setting):
    f2.f2_init_ss(setting)
    install_dr()


def cmd_sim(a):
    f2.u4g.hk.hk_init_budget = drl_init_budget
    f2.u4g.hk.cmd_sim(a)


def cmd_e5(a):
    f2.u4g.hk.hk_init_ss = drl_init_ss
    f2.u4g.hk.cmd_e5(a)


def cmd_val(a):
    import mod25e_late as lt

    lt.SIMD = Path(a.sim_dir)
    lt.OUT = OUT / a.tag
    for fn in (lt.cmd_state, lt.cmd_clock, lt.cmd_fgdown, lt.cmd_fourth):
        fn(argparse.Namespace())
    L = []
    an_one(sim_df(Path(a.sim_dir)), f"sim {a.tag}", L)
    (OUT / a.tag / "an.txt").write_text(chr(10).join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("an")
    s.add_argument("--sims", default="crzhc,f2")
    for nm in ("sim", "e5"):
        x = sub.add_parser(nm)
        x.add_argument("--variant", default="drl")
        x.add_argument("--out-dir", dest="out_dir", default=str(OUT / "play_drl"))
        x.add_argument("--worlds", type=int, default=6 if nm == "sim" else 8)
        x.add_argument("--seasons", type=int, default=8)
        x.add_argument("--workers", type=int, default=3)
        x.add_argument("--seed", type=int, default=21 if nm == "sim" else 11)
    v = sub.add_parser("val")
    v.add_argument("--sim-dir", dest="sim_dir", required=True)
    v.add_argument("--tag", required=True)
    a = ap.parse_args()
    {"an": cmd_an, "sim": cmd_sim, "e5": cmd_e5, "val": cmd_val}[a.cmd](a)


if __name__ == "__main__":
    main()
