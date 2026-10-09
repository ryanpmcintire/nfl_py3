import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "tpo" / "fit.json"
CDW_FIT = REPO / "artifacts" / "mod25e3" / "cdw" / "fit.json"
SALT = 4319
LO = 300.0
HI = 900.0
KINDS = ("band5s", "band7")
NQ = 10
MIN_WINS = 5
CLASS_IDS = (2, 3)


def enabled():
    return os.environ.get("TPO") == "1"


def child_sign(sd, kind):
    sd = np.asarray(sd, float)
    if kind == "band7":
        return np.select([sd <= -17, sd <= -9, sd < 0, sd == 0, sd < 9, sd < 17], [0, 1, 2, 3, 4, 5], 6)
    return np.select([sd <= -17, sd <= -9, sd < 0, sd == 0], [0, 1, 2, 3], 4)


def parent_keys(R, spec):
    import mod25e_clk as ck

    h = ck.hbin(R.hs.to_numpy(), ck.EDGES[spec["edges"]])
    c = ck.cell(R.hs.to_numpy(), R.sd.to_numpy(), R.oto.to_numpy(), R.dto.to_numpy(), ck.EDGES[spec["edges"]], spec["sign"], spec["to"])
    return c * 2 + 1, h * 2 + 1


def child_key(R, spec, kind):
    import mod25e_clk as ck

    h = ck.hbin(R.hs.to_numpy(), ck.EDGES[spec["edges"]])
    return ((h * 8 + child_sign(R.sd.to_numpy(float), kind)) * 8) * 2 + 1


def table(c, b, el, tr):
    import mod25e_clk as ck

    uc, inv = np.unique(c[tr], return_inverse=True)
    cnt = np.zeros((len(uc), ck.NBIN))
    np.add.at(cnt, (inv, b[tr]), 1.0)
    nn = cnt.sum(1)
    return uc, cnt, nn, np.bincount(inv, weights=el[tr]) / np.maximum(nn, 1)


def fold_class(Q, spec, kinds, aps, avals, excl=None):
    import mod25e_clk as ck

    se = Q.season.to_numpy()
    cp, hb = parent_keys(Q, spec)
    cc = {kd: child_key(Q, spec, kd) for kd in kinds}
    b = ck.ebin(Q.el.to_numpy())
    el = Q.el.to_numpy(float)
    sd = Q.sd.to_numpy(float)
    gid = Q.game_id.to_numpy()
    dom = ((Q.hs > LO) & (Q.hs <= HI)).to_numpy()
    out = {}
    for s in np.unique(se):
        if excl is not None and s == excl:
            continue
        te = (se == s) & dom
        tr = (se != s) & (se != excl) if excl is not None else se != s
        nh = int(hb.max()) + 1
        mh = np.zeros((nh, ck.NBIN))
        np.add.at(mh, (hb[tr], b[tr]), 1.0)
        margh = (mh + 1.0) / (mh.sum(1, keepdims=True) + ck.NBIN)
        emh = np.bincount(hb[tr], weights=el[tr], minlength=nh) / np.maximum(np.bincount(hb[tr], minlength=nh), 1)
        qd = np.quantile(el[tr & dom], np.linspace(0.0, 1.0, NQ + 1)[1:-1])
        gi = np.searchsorted(qd, np.arange(ck.NBIN), side="left")
        G = np.zeros((ck.NBIN, NQ))
        G[np.arange(ck.NBIN), gi] = 1.0
        ge, be = gi[b[te]], b[te]
        ar = np.arange(int(te.sum()))
        up, cntp, nnp, ecp = table(cp, b, el, tr)
        ip = np.minimum(np.searchsorted(up, cp[te]), len(up) - 1)
        hp = up[ip] == cp[te]
        npar = np.where(hp, nnp[ip], 0.0)
        cpar = np.where(hp[:, None], cntp[ip], 0.0)
        epar = np.where(hp, ecp[ip], 0.0)
        tabs = {kd: table(cc[kd], b, el, tr) for kd in kinds}
        for ap in aps:
            wp = npar / (npar + ap)
            Pp = (cpar + ap * margh[hb[te]]) / (npar[:, None] + ap)
            mp = wp * epar + (1 - wp) * emh[hb[te]]
            out[("par", ap, None)] = out.get(("par", ap, None), {})
            out[("par", ap, None)][int(s)] = (np.log((Pp @ G)[ar, ge]), mp, el[te], sd[te], gid[te], np.log(Pp[ar, be]))
            for kd in kinds:
                uc, cnt, nn, ecs = tabs[kd]
                ic = np.minimum(np.searchsorted(uc, cc[kd][te]), len(uc) - 1)
                hc = uc[ic] == cc[kd][te]
                n2 = np.where(hc, nn[ic], 0.0)
                c2 = np.where(hc[:, None], cnt[ic], 0.0)
                e2 = np.where(hc, ecs[ic], 0.0)
                for a in avals:
                    P = (c2 + a * Pp) / (n2[:, None] + a)
                    w = n2 / (n2 + a)
                    key = (kd, ap, a)
                    out[key] = out.get(key, {})
                    out[key][int(s)] = (np.log((P @ G)[ar, ge]), w * e2 + (1 - w) * mp, el[te], sd[te], gid[te], np.log(P[ar, be]))
    return out


def pooled(o):
    return sum(v[0].sum() for v in o.values()) / sum(len(v[0]) for v in o.values())


def rows():
    import mod25e_clk as ck

    R = ck.real_rows()
    return R[R.qtr == 4].reset_index(drop=True)


def fit():
    import mod25e_clk as ck

    R = rows()
    spec_all = json.loads(ck.FIT.read_text(encoding="utf-8"))["classes"]
    out = {"lo": LO, "hi": HI, "classes": {}}
    look = 0
    for k in CLASS_IDS:
        nm = ck.CLASSES[k]
        spec = spec_all[nm]["best"]
        Q = R[R.cls == k].reset_index(drop=True)
        full = fold_class(Q, spec, KINDS, ck.SMOOTH, ck.SMOOTH)
        look += len(full)
        base_cfg = ("par", float(spec["a"]), None)
        cur = fold_class(Q, spec, (), (float(spec["a"]),), ())[base_cfg]
        seasons = sorted(cur)
        per = {}
        for so in seasons:
            inner = fold_class(Q, spec, KINDS, ck.SMOOTH, ck.SMOOTH, excl=so)
            sc = {c: pooled(o) for c, o in inner.items()}
            per[so] = max(sc, key=lambda c: sc[c])
        chosen = [full[per[s]][s] for s in seasons]
        N = sum(len(r[0]) for r in chosen)
        d = np.concatenate([r[0] - cur[s][0] for s, r in zip(seasons, chosen)])
        g = np.concatenate([r[4] for r in chosen])
        ug, inv = np.unique(g, return_inverse=True)
        gs = np.bincount(inv, weights=d)
        rng = np.random.default_rng(7)
        bs = np.array([gs[rng.integers(0, len(gs), len(gs))].sum() for _ in range(2000)]) / N
        wins = sum(1 for s, r in zip(seasons, chosen) if r[0].mean() > cur[s][0].mean())
        sc = {c: pooled(o) for c, o in full.items()}
        best = max(sc, key=lambda c: sc[c])
        ll_cur = pooled(cur)
        out["classes"][nm] = {
            "k": k,
            "parent": {"edges": spec["edges"], "sign": spec["sign"], "to": spec["to"]},
            "n_dom": int(N),
            "current_decile_ll": float(ll_cur),
            "per_fold": {str(s): [per[s][0], float(per[s][1]), None if per[s][2] is None else float(per[s][2])] for s in seasons},
            "nested_gain": float(d.sum() / N),
            "nested_wins": int(wins),
            "nested_boot90": [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))],
            "probability_positive": float((bs > 0).mean()),
            "kind": best[0],
            "ap": float(best[1]),
            "a": None if best[2] is None else float(best[2]),
            "full_decile_ll": float(sc[best]),
            "apply": bool(best[0] != "par" and d.sum() > 0 and wins >= MIN_WINS),
        }
        print(nm, json.dumps(out["classes"][nm]), flush=True)
    out["looks"] = look
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out), encoding="utf-8")


def install_tpo():
    import mod25d_variance as dv
    import mod25e_clk as ck
    import mod25e_endgame as eg

    fit_ = json.loads(FIT.read_text(encoding="utf-8"))
    cdw_hz = 0.0
    if os.environ.get("CDW") == "1":
        cdw_hz = float(json.loads(CDW_FIT.read_text(encoding="utf-8"))["chosen"]["spec"]["hz"])
    lo = max(fit_["lo"], cdw_hz)
    R = rows()
    el_all = R.el.to_numpy(float)
    kk = R.cls.to_numpy()
    plan = {}
    for nm, cf in fit_["classes"].items():
        if not cf["apply"]:
            continue
        k = cf["k"]
        mk = kk == k
        Q = R[mk]
        cp, hb = parent_keys(Q, cf["parent"])
        cc = child_key(Q, cf["parent"], cf["kind"])
        el = el_all[mk]
        pc, pp, pm = {}, {}, {}
        for v in np.unique(cc):
            pc[int(v)] = el[cc == v]
        for v in np.unique(cp):
            pp[int(v)] = el[cp == v]
        for v in np.unique(hb):
            pm[int(v)] = el[hb == v]
        plan[k] = {"cf": cf, "child": pc, "parent": pp, "marg": pm, "edges": ck.EDGES[cf["parent"]["edges"]]}
    cseed = int(dv._G["cfg"].get("seed", 3))
    st = {"k": None, "rng": None}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        fr = sys._getframe(1)
        while "offense" not in fr.f_locals:
            fr = fr.f_back
        offense, off_to, def_to = fr.f_locals["offense"], fr.f_locals["off_to"], fr.f_locals["def_to"]  # noqa: F841
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        code = int(drawn["play_type_code"])
        if qtr != 4 or code not in (0, 1):
            return drawn
        hs = clock_val
        if hs <= lo or hs > fit_["hi"]:
            return drawn
        k = int(ck.klass(code, drawn["yards_gained"], drawn["flip"], drawn["points_off"], drawn["points_def"]))
        pl = plan.get(k)
        if pl is None:
            return drawn
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(SALT, cseed)
        rng = st["rng"]
        cf = pl["cf"]
        h = int(ck.hbin([hs], pl["edges"])[0])
        pk = int(ck.cell([hs], [score_diff], [float(off_to)], [float(def_to)], pl["edges"], cf["parent"]["sign"], cf["parent"]["to"])[0]) * 2 + 1
        ck_ = ((h * 8 + int(child_sign([score_diff], cf["kind"])[0])) * 8) * 2 + 1
        u1, u2, v = rng.random(), rng.random(), rng.random()
        ch = pl["child"].get(ck_)
        pa = pl["parent"].get(pk)
        if ch is not None and u1 < len(ch) / (len(ch) + cf["a"]):
            src = ch
        elif pa is not None and u2 < len(pa) / (len(pa) + cf["ap"]):
            src = pa
        else:
            src = pl["marg"][h * 2 + 1]
        new = float(min(src[int(v * len(src))], hs))
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
        fit()
