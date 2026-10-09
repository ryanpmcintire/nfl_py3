import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "cdt" / "fit.json"
SALT = 4313
HORIZON = (60.0, 120.0)
QSPLIT = (0, 1)
MODES = ("rem", "el")


def enabled():
    return os.environ.get("CDT") == "1"


def rows():
    import mod25e_clk as ck

    R = ck.real_rows()
    R = R[R.code.isin([0, 1]) & R.qtr.isin([2, 4])].copy()
    R["rem"] = (R.hs - R.el).clip(lower=0)
    R["u"] = np.where(R.otu > 0, 1, np.where(R.dtu > 0, 2, 0))
    return R.reset_index(drop=True)


def keys(R, mk, edges, qs):
    import mod25e_cdr as cdr

    h = cdr.hb(R.hs.to_numpy()[mk], edges)
    q = (R.qtr.to_numpy()[mk] == 4).astype(int) if qs else np.zeros(mk.sum(), int)
    return h * 2 + q


def fold(R, hz, edges, a, qs, mode, cond):
    import mod25e_cdr as cdr

    scope = (R.hs <= hz).to_numpy()
    tm = scope & (R.u > 0).to_numpy()
    sub = scope if not cond else tm
    ks_all = keys(R, scope, edges, qs)
    full = np.full(len(R), -1)
    full[scope] = ks_all
    kk = full[sub]
    tgt = cdr.ebin(R.rem.to_numpy()[sub] if mode == "rem" else R.el.to_numpy()[sub])
    ss = R.season.to_numpy()[sub]
    uu = R.u.to_numpy()[sub]
    hsv = R.hs.to_numpy()[sub]
    ev = tm[sub]
    out = {}
    nk = int(kk.max()) + 1
    for s in np.unique(ss):
        te, tr = ss == s, ss != s
        cnt = np.zeros((3 if cond else 1, nk, cdr.NBIN))
        np.add.at(cnt, (uu[tr] if cond else np.zeros(tr.sum(), int), kk[tr], tgt[tr]), 1.0)
        tot, n, mp = 0.0, 0, 0.0
        for i in np.flatnonzero(te & ev):
            c = cnt[uu[i] if cond else 0, kk[i]]
            tot += cdr.trunc_ll(c, tgt[i], hsv[i], a)
            n += 1
            lim = min(int(round(hsv[i])), cdr.TAIL + 1)
            w = c[: lim + 1] + a / cdr.NBIN
            e = (np.arange(lim + 1) * w).sum() / w.sum()
            mp += (hsv[i] - e) if mode == "rem" else e
        out[int(s)] = (tot, n, mp)
    return out


def fit():
    R = rows()
    out = {"results": [], "looks": 0}
    import mod25e_cdr as cdr

    for hz, en, a, qs in itertools.product(HORIZON, cdr.EDGES, cdr.SMOOTH, QSPLIT):
        base = fold(R, hz, cdr.EDGES[en], a, qs, "el", False)
        for mode in MODES:
            cd = fold(R, hz, cdr.EDGES[en], a, qs, mode, True)
            n = sum(v[1] for v in cd.values())
            llc = sum(v[0] for v in cd.values()) / n
            llb = sum(v[0] for v in base.values()) / n
            pos = sum(1 for s in cd if cd[s][0] / max(cd[s][1], 1) > base[s][0] / max(base[s][1], 1))
            out["results"].append({"hz": hz, "edges": en, "a": a, "qs": qs, "mode": mode, "n": int(n), "ll_cdt": llc, "ll_base": llb, "gain": llc - llb, "seasons_pos": pos, "seasons": len(cd)})
            out["looks"] += 1
    best = max(out["results"], key=lambda r: r["gain"])
    out["best"] = best
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out), encoding="utf-8")
    for r in sorted(out["results"], key=lambda r: -r["gain"])[:8]:
        print(r)
    print("looks", out["looks"])


def install_cdt():
    import mod25d_variance as dv
    import mod25e_cdr as cdr
    import mod25e_endgame as eg

    b = json.loads(FIT.read_text(encoding="utf-8"))["best"]
    edges = cdr.EDGES[b["edges"]]
    R = rows()
    mk = ((R.hs <= b["hz"]) & (R.u > 0)).to_numpy()
    sub = R[mk]
    kk = keys(R, mk, edges, b["qs"])
    val = sub.rem.to_numpy() if b["mode"] == "rem" else sub.el.to_numpy()
    uu = sub.u.to_numpy()
    pool = {(int(u), int(k)): val[(uu == u) & (kk == k)] for u in (1, 2) for k in np.unique(kk)}
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        hs = clock_val - 1800.0 if qtr == 2 else clock_val
        if code not in (0, 1) or qtr not in (2, 4) or hs <= 0 or hs > b["hz"]:
            return drawn
        u = 1 if float(drawn["off_to_used"]) > 0 else 2 if float(drawn["def_to_used"]) > 0 else 0
        if u == 0:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, cseed)
        k = int(cdr.hb([hs], edges)[0]) * 2 + (1 if b["qs"] and qtr == 4 else 0)
        src = pool.get((u, k))
        if src is None:
            return drawn
        ok = src[src <= hs] if b["mode"] == "rem" else src[src <= hs]
        if not len(ok):
            return drawn
        r = float(ok[int(st["rng"].random() * len(ok))])
        new = float(hs - r) if b["mode"] == "rem" else r
        old = float(drawn["clock_elapsed"])
        drawn = dict(drawn)
        drawn["clock_elapsed"] = new
        lg = dv._G.get("log")
        if lg:
            tp = list(lg[-1])
            tp[10] = new
            lg[-1] = tuple(tp)
        es = dv._G.get("egst")
        if es is not None and es["ps"][0] is not None and abs(es["ps"][0] - (clock_val - old)) < 1e-6:
            stn = float(eg.stop_after(code, float(drawn["yards_gained"]), bool(drawn["flip"]), (drawn["points_off"] + drawn["points_def"]) > 0, float(drawn["off_to_used"]), float(drawn["def_to_used"]), new))
            es["ps"] = (clock_val - new, stn)
        return drawn

    dv._G["pol"] = pol


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        import mod25e_gfl as gf

        if gf.enabled():
            gf.patch()
        fit()
