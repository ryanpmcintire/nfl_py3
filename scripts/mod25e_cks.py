import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FIT = REPO / "artifacts" / "mod25e3" / "cks" / "fit.json"
EDGE_SET = ("w16", "w24", "w36")
SIGNS = ("none", "sign", "band")
TOS = ("none", "any", "both")
QS = (0, 1)
A1 = (0.5, 8.0, 128.0, 2048.0)
A2 = (0.5, 8.0, 128.0, 2048.0)
PRIOR = 2.0


def enabled():
    return os.environ.get("CKS") == "1" and os.environ.get("CKH") == "1"


def rows():
    import mod25e_ckc as kc

    return kc.rows()


def feats(hs, sd, oto, dto, qtr, spec):
    import mod25e_clk as ck

    hs, sd = np.asarray(hs, float), np.asarray(sd, float)
    h = ck.hbin(hs, ck.EDGES[spec["edges"]])
    if spec["sign"] == "sign":
        s = (np.sign(sd) + 1).astype(int)
    elif spec["sign"] == "band":
        s = np.select([sd <= -9, sd < 0, sd == 0, sd < 9], [0, 1, 2, 3], 4)
    else:
        s = np.zeros(len(sd), int)
    oa, da = np.asarray(oto) > 0, np.asarray(dto) > 0
    if spec["to"] == "any":
        t = (oa | da).astype(int)
    elif spec["to"] == "both":
        t = oa.astype(int) * 2 + da.astype(int)
    else:
        t = np.zeros(len(sd), int)
    q = (np.asarray(qtr) == 4).astype(int) if spec["qs"] else np.zeros(len(sd), int)
    L1 = h * 2 + q
    L2 = L1 * 8 + s
    L3 = L2 * 8 + t
    return L1, L2, L3


def stat(L, y):
    u, inv = np.unique(L, return_inverse=True)
    n = np.bincount(inv, minlength=len(u)).astype(float)
    k = np.bincount(inv, weights=y, minlength=len(u))
    return {int(x): (n[i], k[i]) for i, x in enumerate(u)}


def estimator(L1, L2, L3, y, a1, a2):
    d1, d2, d3 = stat(L1, y), stat(L2, y), stat(L3, y)
    g = float(y.mean())

    def p(x1, x2, x3):
        n1, k1 = d1.get(int(x1), (0.0, 0.0))
        p1 = (k1 + PRIOR * g) / (n1 + PRIOR)
        n2, k2 = d2.get(int(x2), (0.0, 0.0))
        p2 = (k2 + a2 * p1) / (n2 + a2)
        n3, k3 = d3.get(int(x3), (0.0, 0.0))
        return (k3 + a1 * p2) / (n3 + a1)

    return p


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit():
    import mod25e_clk as ck

    R = rows()
    out = {"classes": {}, "looks": 0}
    for k in sorted(R.cls.unique()):
        nm = ck.CLASSES[int(k)]
        sub = R[R.cls == k].reset_index(drop=True)
        y = sub.cens.to_numpy().astype(float)
        ss = sub.season.to_numpy()
        res = []
        for en, sg, to, qs in itertools.product(EDGE_SET, SIGNS, TOS, QS):
            spec = {"edges": en, "sign": sg, "to": to, "qs": qs}
            L1, L2, L3 = feats(sub.hs.to_numpy(), sub.sd.to_numpy(), sub.oto.to_numpy(), sub.dto.to_numpy(), sub.qtr.to_numpy(), spec)
            for a1, a2 in itertools.product(A1, A2):
                if sg == "none" and a2 != A2[0]:
                    continue
                if to == "none" and a1 != A1[0]:
                    continue
                p = np.zeros(len(y))
                for se in np.unique(ss):
                    tr, te = ss != se, ss == se
                    f = estimator(L1[tr], L2[tr], L3[tr], y[tr], a1, a2)
                    p[te] = [f(L1[i], L2[i], L3[i]) for i in np.where(te)[0]]
                res.append((bll(y, p), dict(spec, a1=a1, a2=a2)))
        res.sort(key=lambda r: -r[0])
        out["looks"] += len(res)
        out["classes"][nm] = {"k": int(k), "n": int(len(sub)), "ll": res[0][0], "best": res[0][1], "top3": [dict(r[1], ll=r[0]) for r in res[:3]]}
        print(nm, len(sub), res[0], flush=True)
    FIT.parent.mkdir(parents=True, exist_ok=True)
    FIT.write_text(json.dumps(out), encoding="utf-8")
    print("looks", out["looks"])


def tables():
    spec = json.loads(FIT.read_text(encoding="utf-8"))["classes"]
    R = rows()
    T = {}
    for nm, s in spec.items():
        sub = R[R.cls == s["k"]].reset_index(drop=True)
        b = s["best"]
        L1, L2, L3 = feats(sub.hs.to_numpy(), sub.sd.to_numpy(), sub.oto.to_numpy(), sub.dto.to_numpy(), sub.qtr.to_numpy(), b)
        T[s["k"]] = {"spec": b, "f": estimator(L1, L2, L3, sub.cens.to_numpy().astype(float), b["a1"], b["a2"])}
    return T


def p_end(t, hs, sd, qtr, oto, dto):
    L1, L2, L3 = feats([hs], [sd], [oto], [dto], [qtr], t["spec"])
    return float(t["f"](L1[0], L2[0], L3[0]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        import mod25e_gfl as gf

        if gf.enabled():
            gf.patch()
        fit()
