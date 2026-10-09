import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "cku" / "fit.json"
A1 = (0.5, 8.0, 128.0, 2048.0)
A2 = (0.5, 8.0, 128.0, 2048.0)


def enabled():
    return os.environ.get("CKU") == "1" and os.environ.get("CKS") == "1" and os.environ.get("CKH") == "1"


def rows():
    import mod25e_ckc as kc

    R = kc.rows()
    R["used"] = ((R.otu > 0) | (R.dtu > 0)).astype(int)
    return R


def used_flag(off_used, def_used):
    return int(float(off_used) > 0 or float(def_used) > 0)


def feats_u(hs, sd, oto, dto, qtr, used, spec):
    import mod25e_cks as cks

    L1, L2, L3 = cks.feats(hs, sd, oto, dto, qtr, spec)
    u = np.asarray(used).astype(int)
    h1 = L1 * 2 + u
    sg = L2 - L1 * 8
    tt = L3 - L2 * 8
    h2 = h1 * 8 + sg
    h3 = h2 * 8 + tt
    return h1, h2, h3


def end_fit(sub, spec, a1, a2, split):
    import mod25e_cks as cks

    y = sub.cens.to_numpy().astype(float)
    ss = sub.season.to_numpy()
    args = (sub.hs.to_numpy(), sub.sd.to_numpy(), sub.oto.to_numpy(), sub.dto.to_numpy(), sub.qtr.to_numpy())
    if split:
        L1, L2, L3 = feats_u(*args, sub.used.to_numpy(), spec)
    else:
        L1, L2, L3 = cks.feats(*args, spec)
    p = np.zeros(len(y))
    for se in np.unique(ss):
        tr, te = ss != se, ss == se
        f = cks.estimator(L1[tr], L2[tr], L3[tr], y[tr], a1, a2)
        p[te] = [f(L1[i], L2[i], L3[i]) for i in np.where(te)[0]]
    return p


def fit():
    import mod25e_ckc as kc
    import mod25e_clk as ck
    import mod25e_cks as cks

    R = rows()
    cs = json.loads(cks.FIT.read_text(encoding="utf-8"))["classes"]
    hs_ = json.loads(kc.FIT.read_text(encoding="utf-8"))["classes"]
    out = {"classes": {}, "looks": 0}
    for nm, s in cs.items():
        k = s["k"]
        sub = R[R.cls == k].reset_index(drop=True)
        y = sub.cens.to_numpy().astype(float)
        ss = sub.season.to_numpy()
        spec = {kk: s["best"][kk] for kk in ("edges", "sign", "to", "qs")}
        p0 = end_fit(sub, spec, s["best"]["a1"], s["best"]["a2"], False)
        ll0 = cks.bll(y, p0)
        best = None
        for a1, a2 in itertools.product(A1, A2):
            p1 = end_fit(sub, spec, a1, a2, True)
            ll1 = cks.bll(y, p1)
            if best is None or ll1 > best[0]:
                best = (ll1, a1, a2, p1)
        ll1, a1, a2, p1 = best
        seas = np.unique(ss)
        pos = sum(cks.bll(y[ss == se], p1[ss == se]) > cks.bll(y[ss == se], p0[ss == se]) for se in seas)
        beh = {}
        for q in (2, 4):
            for u in (0, 1):
                m = (sub.qtr.to_numpy() == q) & (sub.used.to_numpy() == u)
                beh[f"q{q}u{u}"] = {"n": int(m.sum()), "obs": float(y[m].mean()), "pred_split": float(p1[m].mean()), "pred_base": float(p0[m].mean())}
        hsp = hs_[nm]
        pm = {}
        for a in ck.SMOOTH:
            tot = 0.0
            n = 0
            tots = 0.0
            for u in (0, 1):
                su = sub[sub.used == u].reset_index(drop=True)
                lk, _, _, _ = kc.fold_rows(su, hsp, a, "cp")
                tots += float(lk.sum())
            lb, _, _, _ = kc.fold_rows(sub, hsp, a, "cp")
            pm[a] = (float(lb.mean()), tots / len(sub))
        ab = max(pm, key=lambda a: pm[a][1])
        abb = max(pm, key=lambda a: pm[a][0])
        out["classes"][nm] = {"k": k, "spec": spec, "a1": a1, "a2": a2, "ll_end_base": ll0, "ll_end_split": ll1, "end_gain": ll1 - ll0, "end_seasons_pos": int(pos), "seasons": int(len(seas)), "beh": beh, "a_pmf": ab, "ll_pmf_split": pm[ab][1], "ll_pmf_base": pm[abb][0], "a_base": abb, "pmf_gain": pm[ab][1] - pm[abb][0]}
        out["looks"] += len(A1) * len(A2) + 2 * len(ck.SMOOTH)
        print(nm, out["classes"][nm], flush=True)
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out), encoding="utf-8")
    print("looks", out["looks"])


def tables():
    import mod25e_ckc as kc
    import mod25e_clk as ck
    import mod25e_cks as cks

    R = rows()
    f = json.loads(FIT.read_text(encoding="utf-8"))["classes"]
    hs_ = json.loads(kc.FIT.read_text(encoding="utf-8"))["classes"]
    T, E = {}, {}
    for nm, s in f.items():
        k = s["k"]
        sub = R[R.cls == k].reset_index(drop=True)
        y = sub.cens.to_numpy().astype(float)
        L1, L2, L3 = feats_u(sub.hs.to_numpy(), sub.sd.to_numpy(), sub.oto.to_numpy(), sub.dto.to_numpy(), sub.qtr.to_numpy(), sub.used.to_numpy(), s["spec"])
        E[k] = {"spec": s["spec"], "f": cks.estimator(L1, L2, L3, y, s["a1"], s["a2"])}
        hsp = hs_[nm]
        edges = ck.EDGES[hsp["edges"]]
        for u in (0, 1):
            su = sub[sub.used == u].reset_index(drop=True)
            c = ck.cell(su.hs.to_numpy(), su.sd.to_numpy(), su.oto.to_numpy(), su.dto.to_numpy(), edges, hsp["sign"], hsp["to"])
            hb = ck.hbin(su.hs.to_numpy(), edges)
            ev, cb, last = kc.obs(su)
            uc, ci = np.unique(c, return_inverse=True)
            nh = len(edges) + 1
            cn = np.bincount(ci, minlength=len(uc)).astype(float)
            pc = kc.cp_pmf(ci, len(uc), ev, cb, last) / np.maximum(cn[:, None], 1.0)
            mh = kc.cp_pmf(hb, nh, ev, cb, last)
            ph = (mh + 1.0 / (kc.NB + 1)) / (mh.sum(axis=1, keepdims=True) + 1.0)
            hbc = np.zeros(len(uc), int)
            hbc[ci] = hb
            pm = kc.smooth_pm(cn, pc, ph[hbc], s["a_pmf"])
            T[(k, u)] = {"edges": edges, "sign": hsp["sign"], "to": hsp["to"], "cdf": {int(cv): kc.cdf_of(pm[i]) for i, cv in enumerate(uc)}, "hcdf": {h: kc.cdf_of(ph[h]) for h in range(nh)}}
    return T, E


def p_end(t, hs, sd, qtr, oto, dto, used):
    L1, L2, L3 = feats_u([hs], [sd], [oto], [dto], [qtr], [used], t["spec"])
    return float(t["f"](L1[0], L2[0], L3[0]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        import mod25e_gfl as gf

        if gf.enabled():
            gf.patch()
        fit()
