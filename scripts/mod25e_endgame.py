import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_drill as dr  # noqa: E402

dv = dr.dv
f2 = dr.f2

OUT = REPO / "artifacts" / "mod25e3" / "endgame"


def cmd_el(a):
    OUT.mkdir(parents=True, exist_ok=True)
    S = dr.sim_df(Path(a.sim_dir))
    Rr = dr.real_df()
    L = []
    for nm, d in (("real", Rr), ("sim", S)):
        h = d[(d.qtr == 4) & (d.gsr > 0) & (d.gsr <= 120) & (d.sd <= 0) & (d.sd >= -8)].copy()
        h["over"] = h.el > h.gsr
        L.append(f"== {nm}: n {len(h)}")
        for lo, hi in ((0, 8), (8, 15), (15, 45), (45, 120)):
            b = h[(h.gsr >= lo) & (h.gsr < hi)]
            L.append(f"  gsr[{lo},{hi}) n {len(b)} mean gsr {b.gsr.mean():.1f} el {b.el.mean():.2f} over {b.over.mean():.3f}")
            for c, g in b.groupby("code"):
                L.append(f"     code {c} n {len(g)} el {g.el.mean():.2f} over {g.over.mean():.3f}")
    txt = "\n".join(L)
    (OUT / "el.txt").write_text(txt)
    print(txt)


LATE_CODES = (0, 1, 3, 4, 5)
CLS = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4}


def late_rows(d):
    h = np.where(d.qtr.to_numpy() >= 3, 2, 1)
    d = d.assign(half=h, hs=np.where(d.qtr.to_numpy() == 2, d.gsr - 1800.0, np.where(d.qtr.to_numpy() == 4, d.gsr, np.nan)))
    d = d.sort_values(["g", "play_id"]).reset_index(drop=True)
    prev = d.groupby(["g", "half"])["stop"].shift(1)
    d["pstop"] = prev.fillna(True).astype(float)
    m = ((d.qtr == 4) & (d.gsr <= 600)) | ((d.qtr == 2) & (d.gsr <= 1980))
    m &= (d.down <= 3) & d.code.isin(LATE_CODES)
    return d[m].reset_index(drop=True)


def feats(d, ext):
    base = [d.down, d.dist, d.yl, np.clip(d.sd, -24, 24), d.gsr, (d.oto > 0).astype(float), (d.dto > 0).astype(float)]
    if ext:
        base += [d.hs, (d.qtr == 4).astype(float), d.oto, d.dto, d.pstop]
    return np.column_stack(base)


def snap_axes(d):
    import mod25_mechanisms as m
    q = d.copy()
    for col, ax in (("dist", m.DL_AX), ("yl", m.YL_AX), ("sd", m.SD_AX), ("gsr", m.TL_AX)):
        v = q[col].to_numpy(float)
        if col == "sd":
            v = np.clip(v, -24, 24)
        q[col] = ax[np.abs(ax[None, :] - v[:, None]).argmin(axis=1)]
    return q


def fit_hgb(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=5, max_iter=200, learning_rate=0.08, random_state=1).fit(X, y)


def ll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, None))))


def cmd_fit(a):
    OUT.mkdir(parents=True, exist_ok=True)
    R = late_rows(dr.real_df())
    y = R.code.map(CLS).to_numpy()
    tr = (R.season <= 2015).to_numpy()
    te = ~tr
    win = ((R.qtr == 4) & (R.gsr <= 120)) | ((R.qtr == 2) & (R.gsr > 1800) & (R.gsr <= 1920))
    win = win.to_numpy()
    L = [f"train 2009-15 late rows {tr.sum()}, test 2016-17 {te.sum()}, test two-minute window {int((te & win).sum())}"]
    freq = np.bincount(y[tr], minlength=5) / tr.sum()
    P = {}
    cl = fit_hgb(feats(R[tr], False), y[tr])
    P["poll_exact"] = cl.predict_proba(feats(R[te], False))
    P["poll_grid"] = cl.predict_proba(feats(snap_axes(R[te]), False))
    ce = fit_hgb(feats(R[tr], True), y[tr])
    P["ext"] = ce.predict_proba(feats(R[te], True))
    P["prior"] = np.tile(freq, (te.sum(), 1))
    wt = win[te]
    yt = y[te]
    for k, p in P.items():
        L.append(f"  {k:11s} logloss all late {ll(p, yt):.4f}  window {ll(p[wt], yt[wt]):.4f}  fg-share pred/real window {p[wt][:, 2].mean():.3f}/{np.mean(yt[wt] == 2):.3f} spike {p[wt][:, 4].mean():.3f}/{np.mean(yt[wt] == 4):.3f} kneel {p[wt][:, 3].mean():.3f}/{np.mean(yt[wt] == 3):.3f}")
    for s in (2016, 2017):
        m = (R.season[te] == s).to_numpy() & wt
        L.append(f"  season {s} window n {m.sum()} " + " ".join(f"{k} {ll(p[m], yt[m]):.4f}" for k, p in P.items()))
    txt = chr(10).join(L)
    (OUT / "fit.txt").write_text(txt)
    print(txt)


def cmd_td(a):
    OUT.mkdir(parents=True, exist_ok=True)
    S = dr.sim_df(Path(a.sim_dir))
    Rr = dr.real_df()
    L = []
    for nm, d in (("real", Rr), ("sim", S)):
        d = d.copy()
        d["oc"] = dr.outcome(d)
        h = d[(d.qtr == 4) & (d.gsr > 0) & (d.gsr <= 120) & (d.sd <= 0) & (d.sd >= -8)]
        n = d.g.nunique()
        L.append(f"== {nm}: games {n} hurry<=120 snaps {len(h)}")
        for lo, hi in ((0, 30), (30, 60), (60, 120)):
            b = h[(h.gsr >= lo) & (h.gsr < hi) & (h.code <= 1)]
            p = b[b.code == 1]
            c = p[p.oc.isin(["pass_comp", "score_play"]) & (p.yards > 0)]
            L.append(f"  gsr[{lo},{hi}) scrimmage n {len(b)} pass share {len(p) / len(b):.3f} inc share of passes {np.mean(p.oc == 'pass_inc'):.3f} sack {np.mean(p.oc == 'sack'):.3f} mean yds/play {b.yards.mean():.2f} pass yds {p.yards.mean():.2f} P(yds>=20|pass) {np.mean(p.yards >= 20):.3f} P(yds>=10) {np.mean(p.yards >= 10):.3f} score_play {np.mean(b.oc == 'score_play'):.4f} flip {np.mean(b.oc == 'turnover_play'):.4f} stop-after share {np.mean(b.el < 6):.3f}")
    txt = chr(10).join(L)
    (OUT / "td.txt").write_text(txt)
    print(txt)


dv.DV["egl"] = dict(dv.DV["f2"], dr=0, eg=1)
dv.DV["egd"] = dict(dv.DV["f2"], dr=0, eg=2)
SALT_EL = 4201
SALT_DC = 4202
CODE_OF = {v: k for k, v in CLS.items()}


HC = {"cut": None}


def hurry_cut(code, yards, el):
    code = np.asarray(code)
    yards = np.asarray(yards)
    el = np.asarray(el, float)
    inc = el[(code == 1) & (yards == 0)]
    comp = el[(code == 1) & (yards > 0)]
    lo = int(np.ceil(np.median(inc)))
    hi = int(np.median(comp[comp > np.median(inc)]))
    edges = np.arange(lo, hi + 2)
    cnt = np.histogram(comp, bins=edges)[0]
    return float(edges[int(np.argmin(cnt))])


def stop_after(code, yards, flip, scored, otu, dtu, el=None):
    s = np.asarray(flip).astype(bool) | np.asarray(scored).astype(bool) | ((np.asarray(code) == 1) & (np.asarray(yards) == 0)) | (np.asarray(code) == 5) | (np.asarray(otu) > 0) | (np.asarray(dtu) > 0)
    if HC["cut"] is not None and el is not None:
        s = s | ((np.asarray(code) <= 1) & (np.asarray(el, float) <= HC["cut"]))
    return s.astype(float)


def train_frame():
    d = dr.real_df()
    m = dr.meta_frame()[["game_id", "play_id"]].reset_index().rename(columns={"index": "pos"})
    d = d.merge(m, on=["game_id", "play_id"], how="left")
    d["sc"] = (d.po + d.pdf) > 0
    d["stp"] = stop_after(d.code, d.yards, d.flip, d.sc, d.otu, d.dtu, d.el)
    d["half"] = np.where(d.qtr.to_numpy() >= 3, 2, 1)
    d = d.sort_values(["g", "play_id"]).reset_index(drop=True)
    d["pstop"] = d.groupby(["g", "half"])["stp"].shift(1).fillna(1.0)
    d["hs"] = np.where(d.qtr == 2, d.gsr - 1800.0, np.where(d.qtr == 4, d.gsr, np.nan))
    return d


def dec_feats(down, dist, yl, sd, gsr, hs, q4, oto, dto, pstop):
    return np.column_stack([down, dist, yl, np.clip(sd, -24, 24), gsr, (np.asarray(oto) > 0).astype(float), (np.asarray(dto) > 0).astype(float), hs, q4, oto, dto, pstop])


def install_eg():
    mode = int(dv._G["cfg"].get("eg", 0))
    if not mode:
        return
    R = pd.read_parquet(REPO / "artifacts" / "sim09" / "f2" / "real_fit.parquet")
    qt = R["qtr"].to_numpy()
    gs = R["gsr"].to_numpy(float)
    win = ((qt == 4) & (gs > 0) & (gs <= f2.u4g.WARN_AT[4])) | ((qt == 2) & (gs > f2.u4g.FLOOR[2]) & (gs <= f2.u4g.WARN_AT[2]))
    R = R[win].reset_index(drop=True)
    HC["cut"] = hurry_cut(R["code"], R["yards"], R["el"]) if mode >= 4 else None
    half = R["qtr"].to_numpy()
    code = R["code"].to_numpy()
    sc = (R["po"].to_numpy() + R["pdf"].to_numpy()) > 0
    stp = (R["flip"].to_numpy().astype(bool) | sc | ((code == 1) & (R["yards"].to_numpy() == 0))).astype(int)
    cls = (R["otu"].to_numpy() > 0).astype(int) + 2 * (R["dtu"].to_numpy() > 0).astype(int)
    F = np.column_stack([R["gsr"].to_numpy(float), np.clip(R["sd"].to_numpy(float), -24, 24), (R["oto"].to_numpy() > 0).astype(float), (R["dto"].to_numpy() > 0).astype(float)])
    sdv = F.std(axis=0)
    sdv[sdv == 0] = 1.0
    Z = F / sdv
    el = R["el"].to_numpy(float)
    cseed = int(dv._G["cfg"].get("seed", 3))
    ns = dv._G["ns"]
    st = {"k": None, "rng": None, "rng2": None, "to": (3.0, 3.0), "ps": (None, 1.0)}
    dec = ns["DECIDE"]
    dv._G["egst"] = st
    wide = os.environ.get("KNW") == "1"

    def inwin(qtr, clock_val):
        return dr.in_window(qtr, clock_val) or (wide and ((qtr == 4 and 0 < clock_val <= 600.0) or (qtr == 2 and 1800.0 < clock_val <= 1980.0)))
    if mode >= 2:
        T = train_frame()
        L = T[((T.qtr == 4) & (T.gsr <= 600)) | ((T.qtr == 2) & (T.gsr <= 1980))]
        L = L[(L.down <= 3) & L.code.isin(LATE_CODES)].reset_index(drop=True)
        yL = L.code.map(CLS).to_numpy()
        clf = fit_hgb(dec_feats(L.down, L.dist, L.yl, L.sd, L.gsr, L.hs, (L.qtr == 4).astype(float), L.oto, L.dto, L.pstop), yL)
        Pz = np.column_stack([L.dist, L.yl, np.clip(L.sd, -24, 24), L.hs]).astype(float)
        psd = Pz.std(axis=0)
        psd[psd == 0] = 1.0
        Pz = Pz / psd
        Lc = L.code.to_numpy()
        Ld = L.down.to_numpy()
        Lq = L.qtr.to_numpy()
        Lpos = L.pos.to_numpy().astype(int)
        cache = {}

        def rerow(rng, dk, c, q, dist, yl, sd, hs):
            sel = np.flatnonzero((Ld == dk) & (Lc == c) & (Lq == q))
            if len(sel) < dr.MIN_POOL:
                sel = np.flatnonzero((Ld == dk) & (Lc == c))
            if not len(sel):
                return None
            qv = np.array([dist, yl, min(max(sd, -24.0), 24.0), hs]) / psd
            d = ((Pz[sel] - qv) ** 2).sum(axis=1)
            k = max(int(np.sqrt(len(sel))), 1)
            near = sel[np.argpartition(d, k - 1)[:k]]
            return int(Lpos[int(near[int(rng.integers(0, len(near)))])])

    def rerow_ct(rng, dk, c, q, ct, sp, dist, yl, sd, hs):
        h = dv._G["f2h"]
        lc = h["cls"][Lpos]
        ls = h["stop"][Lpos]
        base_m = (Ld == dk) & (Lc == c) & (lc == ct) & (ls == sp)
        sel = np.flatnonzero(base_m & (Lq == q))
        if len(sel) < dr.MIN_POOL:
            sel = np.flatnonzero(base_m)
        if len(sel) < dr.MIN_POOL:
            sel = np.flatnonzero((Ld == dk) & (Lc == c) & (lc == ct))
        if not len(sel):
            return None
        qv = np.array([dist, yl, min(max(sd, -24.0), 24.0), hs]) / psd
        d = ((Pz[sel] - qv) ** 2).sum(axis=1)
        k = max(int(np.sqrt(len(sel))), 1)
        near = sel[np.argpartition(d, k - 1)[:k]]
        return int(Lpos[int(near[int(rng.integers(0, len(near)))])])

    def tofix(r, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, hs):
        h = dv._G["f2h"]
        c = int(h["code"][r])
        cl = int(h["cls"][r])
        if c in (0, 1):
            ct = h["ctdraw"](r, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to)
        else:
            ct = (cl & 1 if off_to > 0 else 0) + (cl & 2 if def_to > 0 else 0)
        if ct == cl:
            return r
        for t in ((ct, 0) if ct else (0,)):
            j = rerow_ct(st["rng2"], down, c, qtr, t, int(h["stop"][r]), distance, yardline, score_diff, hs)
            if j is not None:
                return j
        return r

    def reset(k):
        if st["k"] != k:
            st["k"] = k
            st["rng"] = dv.task_rng(SALT_EL, cseed)
            st["rng2"] = dv.task_rng(SALT_DC, cseed)
            st["ps"] = (None, 1.0)

    def decide(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, *rest):
        st["to"] = (float(off_to), float(def_to))
        reset(dv._G.get("task_key"))
        if mode >= 2 and (not in_ot) and down in (1, 2, 3) and inwin(qtr, clock_val):
            code0 = int(dv._G["tables"]["arrays"]["play_type_code"][idx])
            if code0 == 6:
                return idx
            hs = clock_val - 1800.0 if qtr == 2 else clock_val
            ps = st["ps"][1] if (st["ps"][0] is not None and abs(st["ps"][0] - clock_val) < 1e-6) else 1.0
            dv._G["egps"] = (ps, None if st["ps"][0] is None else float(st["ps"][0] - clock_val))
            f = dec_feats(np.array([down]), np.array([distance]), np.array([yardline]), np.array([score_diff]), np.array([clock_val]), np.array([hs]), np.array([float(qtr == 4)]), np.array([off_to]), np.array([def_to]), np.array([ps]))
            key = tuple(np.round(f[0], 1))
            pr = cache.get(key)
            if pr is None:
                pr = clf.predict_proba(f)[0]
                cache[key] = pr
            c = min(int(np.searchsorted(np.cumsum(pr), st["rng2"].random() * pr.sum())), len(pr) - 1)
            if c == CLS.get(code0, -1):
                r = idx
            else:
                j = rerow(st["rng2"], down, CODE_OF[c], qtr, distance, yardline, score_diff, hs)
                r = idx if j is None else j
            if mode >= 3:
                r = tofix(r, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, hs)
            return r
        return dec(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, *rest)

    ns["DECIDE"] = decide
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        if not dr.in_window(qtr, clock_val):
            return drawn
        reset(dv._G.get("task_key"))
        c = int(drawn["play_type_code"])
        sc_ = (float(drawn["points_off"]) + float(drawn["points_def"])) > 0
        stp_ = int(bool(drawn["flip"]) or sc_ or (c == 1 and float(drawn["yards_gained"]) == 0))
        cl_ = int(float(drawn["off_to_used"]) > 0) + 2 * int(float(drawn["def_to_used"]) > 0)
        m0 = (half == qtr) & (code == c)
        sel = None
        for m in (m0 & (stp == stp_) & (cls == cl_), m0 & (stp == stp_), m0):
            ix = np.flatnonzero(m)
            if len(ix) >= dr.MIN_POOL:
                sel = ix
                break
        if sel is not None:
            q = np.array([clock_val, min(max(score_diff, -24.0), 24.0), float(st["to"][0] > 0), float(st["to"][1] > 0)]) / sdv
            d = ((Z[sel] - q) ** 2).sum(axis=1)
            k = max(int(np.sqrt(len(sel))), 1)
            near = sel[np.argpartition(d, k - 1)[:k]]
            j = int(near[int(st["rng"].integers(0, len(near)))])
            hs = clock_val - 1800.0 if qtr == 2 else clock_val
            rem = max(float(np.round(hs)) - float(el[j]), 0.0)
            drawn = dict(drawn)
            drawn["clock_elapsed"] = float(max(hs - rem, 0.0))
            lg = dv._G.get("log")
            if lg:
                t = list(lg[-1])
                t[10] = drawn["clock_elapsed"]
                lg[-1] = tuple(t)
        stn = float(stop_after(c, float(drawn["yards_gained"]), bool(drawn["flip"]), sc_, float(drawn["off_to_used"]), float(drawn["def_to_used"]), float(drawn["clock_elapsed"])))
        st["ps"] = (clock_val - float(drawn["clock_elapsed"]), stn)
        return drawn

    dv._G["pol"] = pol


def eg_init_budget(setting):
    f2.f2_init_budget(setting)
    install_eg()


def eg_init_ss(setting):
    f2.f2_init_ss(setting)
    install_eg()


def cmd_sim(a):
    f2.u4g.hk.hk_init_budget = eg_init_budget
    f2.u4g.hk.cmd_sim(a)


def cmd_e5(a):
    f2.u4g.hk.hk_init_ss = eg_init_ss
    f2.u4g.hk.cmd_e5(a)


def cmd_val(a):
    dr.OUT = OUT
    dr.cmd_val(a)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("el")
    s.add_argument("--sim-dir", dest="sim_dir", default=str(dr.OUT / "play_drl"))
    sub.add_parser("fit")
    for nm in ("sim", "e5"):
        x = sub.add_parser(nm)
        x.add_argument("--variant", default="egd")
        x.add_argument("--out-dir", dest="out_dir", default=str(OUT / "play_egd"))
        x.add_argument("--worlds", type=int, default=6 if nm == "sim" else 8)
        x.add_argument("--seasons", type=int, default=8)
        x.add_argument("--workers", type=int, default=3)
        x.add_argument("--seed", type=int, default=21 if nm == "sim" else 11)
    v = sub.add_parser("val")
    v.add_argument("--sim-dir", dest="sim_dir", required=True)
    v.add_argument("--tag", required=True)
    t = sub.add_parser("td")
    t.add_argument("--sim-dir", dest="sim_dir", default=str(dr.OUT / "play_drl"))
    a = ap.parse_args()
    {"el": cmd_el, "fit": cmd_fit, "td": cmd_td, "sim": cmd_sim, "e5": cmd_e5, "val": cmd_val}[a.cmd](a)


if __name__ == "__main__":
    main()
