import os
import sys
from pathlib import Path

os.environ.setdefault("A2_LABEL", "crHpqokgndecsmfwtjo2as2")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_adj2 as a2  # noqa: E402
import mod25e_late as late  # noqa: E402

OUTD = REPO / "artifacts" / "mod25e3" / "e95"
NB = 300
OUT = []
LOOKS = [0]
NONSC = ["punt", "to", "downs", "fgmiss"]
rng = np.random.default_rng(95)


def say(s=""):
    print(s)
    OUT.append(s)


def sim_nlive(sd):
    sdir = a2.ART / f"e5_{a2.LABEL}_s{sd}"
    out = []
    for f in sorted(sdir.glob("play_*_*.parquet")):
        w, s = (int(x) for x in f.stem.split("_")[1:])
        if s < late.BURN:
            continue
        d = pd.read_parquet(f, columns=["g", "qtr", "offhome", "po", "pdf", "code"])
        d = d[d.qtr <= 4].sort_values("g", kind="stable").reset_index(drop=True)
        g, oh, q = d.g.to_numpy(), d.offhome.to_numpy(), d.qtr.to_numpy()
        sc = ((d.po > 0) | (d.pdf > 0)).to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2)) | sc[:-1]]
        did = np.cumsum(new)
        lv = d.code.isin([0, 1, 2, 3]).to_numpy().astype(float)
        out.append(np.bincount(did, weights=lv)[1:])
    return np.concatenate(out)


def prep(D, nlive):
    D = D.copy()
    D["nlive"] = nlive
    D["S"] = (D.s < 0).astype(int)
    X = a2.xtable(D)
    Xa = X.rename(columns={"x": "xa"})
    Xb = X.assign(S=1 - X.S).rename(columns={"x": "xb"})
    H = D[D.h == 1].merge(Xa, on=["gk", "S"], how="inner").merge(Xb, on=["gk", "S"], how="inner")
    H["s0"] = 100.0 - H.yl0
    H["E"] = 100.0 - H.ylast
    H["G"] = H.E - H.s0
    H["nb"] = (H.nlive - 1).clip(lower=0).astype(float)
    H["ypp"] = np.where(H.nb > 0, H.G / H.nb.where(H.nb > 0, 1), np.nan)
    H["score"] = H.end.isin(["td", "fg"]).astype(float)
    H["defscore"] = (H.end == "defscore").astype(float)
    for t in NONSC + ["other"]:
        H["m_" + t] = (H.end == t).astype(float)
    H["d4"] = (H.dlast == 4).astype(float)
    H["d3"] = (H.dlast == 3).astype(float)
    H["d1"] = (H.dlast == 1).astype(float)
    return H


def centre(S, cols):
    S = S.copy()
    for c in cols:
        S[c + "_c"] = S[c] - S.groupby([S.q0, S.bi])[c].transform("mean")
    return S


def gsums(gi, G, *arrs):
    return [np.bincount(gi, weights=a, minlength=G) for a in arrs]


def boot_slope(x, y, gi, W):
    G = W.shape[1]
    one = np.bincount(gi, minlength=G).astype(float)
    Sx, Sy, Sxx, Sxy = gsums(gi, G, x, y, x * x, x * y)
    N = W @ one
    mx, my = (W @ Sx) / N, (W @ Sy) / N
    return ((W @ Sxy) / N - mx * my) / ((W @ Sxx) / N - mx**2)


def boot_two(x1, x2, y, gi, W):
    G = W.shape[1]
    one = np.bincount(gi, minlength=G).astype(float)
    S11, S12, S22, S1y, S2y, S1, S2, Sy = gsums(gi, G, x1 * x1, x1 * x2, x2 * x2, x1 * y, x2 * y, x1, x2, y)
    N = W @ one
    m1, m2, my = (W @ S1) / N, (W @ S2) / N, (W @ Sy) / N
    c11 = (W @ S11) / N - m1 * m1
    c12 = (W @ S12) / N - m1 * m2
    c22 = (W @ S22) / N - m2 * m2
    c1y = (W @ S1y) / N - m1 * my
    c2y = (W @ S2y) / N - m2 * my
    det = c11 * c22 - c12 * c12
    return (c22 * c1y - c12 * c2y) / det, (c11 * c2y - c12 * c1y) / det


def fmt(a, d=4):
    return f"{a[0]:+.{d}f} [{np.quantile(a[1:], .025):+.{d}f},{np.quantile(a[1:], .975):+.{d}f}]"


def gapline(R, Sm, label, d=4):
    g = Sm - R
    LOOKS[0] += 1
    return f"  {label:30s} real {fmt(R, d)} | sim {fmt(Sm, d)} | s-r {fmt(g, d)} P(s-r>0) {np.mean(g[1:] > 0):.2f}"


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    ns = late.load_xq()
    real = late.real_play_drives(ns["POOL"])
    sim = pd.concat([a2.sim_drives(sd) for sd in a2.SEEDS], ignore_index=True)
    nl_sim = np.concatenate([sim_nlive(sd) for sd in a2.SEEDS])
    assert len(nl_sim) == len(sim), (len(nl_sim), len(sim))
    nl_real = (real.run + real.pas + real.kneel + real.spike + (real.lastpt == "punt").astype(int) + real.fga).to_numpy()
    rc, _ = ns["components"](real)
    fpmap = pd.Series(rc.p.to_numpy() - rc.p.mean() - rc.c_strength.to_numpy()).groupby(pd.cut(rc.yl0, ns["YL_BINS"], right=False).to_numpy(), observed=True).mean()
    Dr = a2.decorate(ns, real, None, fpmap, None)
    Ds = a2.decorate(ns, sim, None, fpmap, None)
    say(f"E95 A-side H2 drive end spot on A's own H1 residual xa; label {a2.LABEL} seeds {a2.SEEDS}; no sim run; boots {NB}; units yards toward goal (E = 100 - yardline_100 of last live snap, s0 = 100 - start yardline, G = E - s0, nb = live plays before the last live play, ypp = G/nb); slopes per point of xa, quarter x lead-band cells demeaned within each subset; game bootstrap")
    Hd = {"real": prep(Dr, nl_real), "sim": prep(Ds, nl_sim)}
    for nm in Hd:
        H = Hd[nm]
        q = H[H.end.isin(NONSC)]
        say(f"{nm}: A-H2 drives {len(H)} games {H.gk.nunique()} sd(xa) {H.xa.std():.2f}; end mix " + " ".join(f"{t} {np.mean(H.end == t):.3f}" for t in ["td", "fg", "punt", "to", "downs", "fgmiss", "defscore", "end", "other"]))
        say(f"   nonscoring n {len(q)}: mean E {q.E.mean():.2f} s0 {q.s0.mean():.2f} G {q.G.mean():.2f} nb {q.nb.mean():.2f} ypp {q.ypp.mean():.3f}")

    subsets = {"nonscoring": lambda H: H.end.isin(NONSC).to_numpy()}
    for t in NONSC:
        subsets[t] = (lambda t: lambda H: (H.end == t).to_numpy())(t)
    ycols = ["E", "s0", "G", "nb", "ypp"]
    res = {}
    means = {}
    for nm in Hd:
        H = Hd[nm]
        gk = pd.Index(sorted(H.gk.unique()))
        gi_all = gk.get_indexer(H.gk)
        W = np.vstack([np.ones((1, len(gk))), rng.multinomial(len(gk), np.ones(len(gk)) / len(gk), size=NB).astype(float)])
        res[nm] = {}
        for sn, fn in subsets.items():
            m = fn(H) & H.ypp.notna().to_numpy()
            S = centre(H[m].reset_index(drop=True), ycols + ["xa", "xb"])
            gi = gi_all[m]
            x = S.xa_c.to_numpy()
            r = {c: boot_slope(x, S[c + "_c"].to_numpy(), gi, W) for c in ycols}
            xb = S.xb_c.to_numpy()
            for c in ("E", "G"):
                r[c + "|2"] = boot_two(x, xb, S[c + "_c"].to_numpy(), gi, W)
            r["n"] = int(m.sum())
            res[nm][sn] = r
            means[(nm, sn)] = (S.nb.mean(), S.ypp.mean())
        S = centre(H, ["score", "defscore", "xa", "xb"] + ["m_" + t for t in NONSC + ["other"]])
        x = S.xa_c.to_numpy()
        res[nm]["mix"] = {c: boot_slope(x, S[c + "_c"].to_numpy(), gi_all, W) for c in ["score", "defscore"] + ["m_" + t for t in NONSC + ["other"]]}
        res[nm]["mix"]["score|2"] = boot_two(x, S.xb_c.to_numpy(), S.score_c.to_numpy(), gi_all, W)
        nm_m = H.end.isin(NONSC).to_numpy()
        S2 = centre(H[nm_m].reset_index(drop=True), ["d4", "d3", "d1", "xa"])
        gi2 = gi_all[nm_m]
        res[nm]["down"] = {c: boot_slope(S2.xa_c.to_numpy(), S2[c + "_c"].to_numpy(), gi2, W) for c in ("d4", "d3", "d1")}
    for sn in subsets:
        say(f"subset {sn} (n real {res['real'][sn]['n']} sim {res['sim'][sn]['n']})")
        for c, lab in (("E", "end spot E (yd/pt)"), ("s0", "start s0"), ("G", "gain G = E - s0"), ("nb", "plays before last"), ("ypp", "net yd per play")):
            say(gapline(res["real"][sn][c], res["sim"][sn][c], lab))
        gE = res["sim"][sn]["E"][0] - res["real"][sn]["E"][0]
        say(f"  share of E gap: s0 {(res['sim'][sn]['s0'][0] - res['real'][sn]['s0'][0]) / gE:+.2f}, G {(res['sim'][sn]['G'][0] - res['real'][sn]['G'][0]) / gE:+.2f}")
        for nm in Hd:
            mnb, myp = means[(nm, sn)]
            a, b = myp * res[nm][sn]["nb"][0], mnb * res[nm][sn]["ypp"][0]
            say(f"  {nm}: G slope {res[nm][sn]['G'][0]:+.4f}; linearised plays-channel {a:+.4f} + per-play-yards-channel {b:+.4f} = {a + b:+.4f}")
        for c in ("E", "G"):
            for k, lab in ((0, "on xa"), (1, "on xb")):
                say(gapline(res["real"][sn][c + "|2"][k], res["sim"][sn][c + "|2"][k], f"2-reg {c} {lab} (xb=B H1)"))
    say("end-type mix: slope of indicator on xa over all A H2 drives")
    for c in res["real"]["mix"]:
        if c.endswith("|2"):
            for k, lab in ((0, "xa"), (1, "xb")):
                say(gapline(res["real"]["mix"][c][k], res["sim"]["mix"][c][k], f"2-reg {c} on {lab}", 5))
        else:
            say(gapline(res["real"]["mix"][c], res["sim"]["mix"][c], c, 5))
    say("stall down among nonscoring endings: slope of share on xa")
    for c in ("d4", "d3", "d1"):
        say(gapline(res["real"]["down"][c], res["sim"]["down"][c], c, 5))
    say("stall distance is not in the drive table; not tested")
    say(f"looks {LOOKS[0]}")
    (OUTD / "e95.txt").write_text("\n".join(OUT))


if __name__ == "__main__":
    main()
