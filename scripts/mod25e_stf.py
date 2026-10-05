import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUTD = REPO / "artifacts" / "mod25e3" / "stf"
FIT = OUTD / "fit.json"
GROUPS = {"K": ("td", "fg", "fgmiss", "defscore"), "P": ("punt",), "T": ("to",)}


def enabled():
    return os.environ.get("STF") == "1"


def start_resid(D):
    import numpy as np

    D = D.copy()
    D["S"] = (D.s < 0).astype(int)
    D["ylw"] = D.ylc - D.groupby("ptype").ylc.transform("mean")
    D["sk"] = np.where(D.gk.str.match(r"^\d{4}_"), D.gk.str[:4], D.gk.str.rsplit("_", n=1).str[0])
    D["k"] = D.groupby(["gk", "S"]).cumcount() % 2
    return D


def lgo(D, key, col):
    g = D.groupby([key, col])
    sm = g.ylw.transform("sum")
    ct = g.ylw.transform("size")
    gg = D.groupby(["gk", key, col])
    gs = gg.ylw.transform("sum")
    gc = gg.ylw.transform("size")
    n = ct - gc
    return ((sm - gs) / n.where(n > 0)).to_numpy()


def parity_cov(D, col="ylw", boots=None):
    import numpy as np

    P = D.groupby(["gk", "S", "k"])[col].mean().unstack().dropna()
    return float(np.cov(P[0].to_numpy(), P[1].to_numpy())[0, 1]), len(P)


def group_stats(D, lab, say):
    import numpy as np

    for gname, ts in GROUPS.items():
        d = D[D.ptype.isin(ts)].copy()
        d["k"] = d.groupby(["gk", "S"]).cumcount() % 2
        c, n = parity_cov(d)
        say(f"  {lab} group {gname} drives {len(d)} split-half cov {c:+.3f} yd^2 n {n}")
    M = D.groupby(["gk", "S", D.ptype.map({t: g for g, ts in GROUPS.items() for t in ts}).fillna("O")]).ylw.mean().unstack()
    for a, b in (("K", "P"), ("K", "T"), ("P", "T")):
        m = M[[a, b]].dropna()
        say(f"  {lab} cross-group cov {a}-{b} {np.cov(m[a], m[b])[0, 1]:+.3f} n {len(m)}")


def season_split(D):
    import numpy as np

    D = D.copy()
    D["a"] = lgo(D, "sk", "off")
    D["b"] = lgo(D, "sk", "dfn")
    ok = D.a.notna() & D.b.notna()
    d = D[ok]
    X = np.c_[np.ones(len(d)), d.a.to_numpy(), d.b.to_numpy()]
    beta = np.linalg.lstsq(X, d.ylw.to_numpy(), rcond=None)[0]
    d = d.assign(ylr=d.ylw.to_numpy() - X @ beta + beta[0])
    ca, n = parity_cov(d, "ylw")
    cr, _ = parity_cov(d, "ylr")
    return beta, float(d.a.var() * beta[1] ** 2), float(d.b.var() * beta[2] ** 2), ca, cr, n


def cross_side(D):
    import numpy as np

    M = D.groupby(["gk", "S"]).ylw.mean().unstack().dropna()
    return float(np.cov(M[0], M[1])[0, 1]), len(M)


def cmd_an():
    import numpy as np

    import mod25e_adj2 as a2
    import mod25e_late as late

    OUT = []

    def say(s=""):
        print(s)
        OUT.append(s)

    OUTD.mkdir(parents=True, exist_ok=True)
    ns = late.load_xq()
    real = late.real_play_drives(ns["POOL"])
    rc, _ = ns["components"](real)
    fpmap = __import__("pandas").Series(rc.p.to_numpy() - rc.p.mean() - rc.c_strength.to_numpy()).groupby(__import__("pandas").cut(rc.yl0, ns["YL_BINS"], right=False).to_numpy(), observed=True).mean()
    Dr = start_resid(a2.decorate(ns, real, None, fpmap, None))
    sims = []
    for sd in a2.SEEDS:
        Ds = a2.decorate(ns, a2.sim_drives(sd), None, fpmap, None)
        sims.append(start_resid(Ds))
    import pandas as pd

    Ds = pd.concat(sims, ignore_index=True)
    say(f"E90 special-teams form decomposition; label {a2.LABEL} seeds {a2.SEEDS}; real drives {len(Dr)} sim {len(Ds)}")
    res = {}
    for lab, D in (("real", Dr), ("sim", Ds)):
        tb = D[~D.ptype.isin(["to", "other"]) & (D.yl0 != 75.0)].copy()
        tb["k"] = tb.groupby(["gk", "S"]).cumcount() % 2
        say(f"  {lab} non-touchback K+P starts split-half cov {parity_cov(tb)[0]:+.3f}")
        c, n = parity_cov(D)
        xs, nx = cross_side(D)
        beta, va, vb, ca, cr, n2 = season_split(D)
        say(f"{lab}: odd-even cov all {c:+.3f} (n {n}); cross-side cov {xs:+.3f} (n {nx}); season split: beta own-return {beta[1]:+.3f} opp-coverage {beta[2]:+.3f}, var contribution return {va:.3f} coverage {vb:.3f}; cov before {ca:+.3f} after season removal {cr:+.3f}")
        group_stats(D, lab, say)
        res[lab] = dict(c=c, xs=xs, va=va, vb=vb, ca=ca, cr=cr)
    say("")
    (OUTD / "an.txt").write_text(chr(10).join(OUT))
    r, m = res["real"], res["sim"]
    G = {}
    for lab, D in (("real", Dr), ("sim", Ds)):
        out = {}
        for gname, ts in (("K", GROUPS["K"]), ("P", GROUPS["P"])):
            d = D[D.ptype.isin(ts)].copy()
            d["k"] = d.groupby(["gk", "S"]).cumcount() % 2
            out[gname] = parity_cov(d)[0]
        M = D.groupby(["gk", "S", D.ptype.map({t: g for g, ts in GROUPS.items() for t in ts}).fillna("O")]).ylw.mean().unstack()
        out["KP"] = float(np.cov(M[["K", "P"]].dropna().K, M[["K", "P"]].dropna().P)[0, 1])
        G[lab] = out
    Vs = G["real"]["KP"] - G["sim"]["KP"]
    Vk = G["real"]["K"] - G["sim"]["K"] - Vs
    Vp = G["real"]["P"] - G["sim"]["P"] - Vs
    w = r["va"] / (r["va"] + r["vb"])
    fit = dict(V=dict(s=Vs, K=Vk, P=Vp), w=w, xs_excess=r["xs"] - m["xs"], real=r, sim=m, groups=G)
    say(f"fit V shared {Vs:.3f} K-specific {Vk:.3f} P-specific {Vp:.3f}; return share w {w:.3f}; cross-side excess {fit['xs_excess']:+.3f}")
    rng = np.random.default_rng(90)
    cs = {}
    for kap in (0.0, 1.0):
        fit["kappa"] = kap
        cs[kap] = replay_metrics(Ds, fit, np.random.default_rng(90), False)["xs"]
    slope = cs[1.0] - cs[0.0]
    kap = float(np.clip((r["xs"] - cs[0.0]) / slope, -1.0, 1.0))
    fit["kappa"] = kap
    say(f"kappa (same-team return/coverage correlation) from cross-side moment: {kap:+.3f} (xs base {m['xs']:+.3f}, with latents kappa0 {cs[0.0]:+.3f} kappa1 {cs[1.0]:+.3f}, target {r['xs']:+.3f})")
    for ex in (False, True):
        mt = replay_metrics(Ds, fit, np.random.default_rng(90), ex)
        say(f"replay exempt_touchback={ex}: all {mt['c']:+.3f} (real {r['c']:+.3f}) K {mt['K']:+.3f} (real {G['real']['K']:+.3f}) P {mt['P']:+.3f} (real {G['real']['P']:+.3f}) K-P {mt['KP']:+.3f} (real {G['real']['KP']:+.3f}) cross-side {mt['xs']:+.3f} (real {r['xs']:+.3f})")
        fit[f"replay_ex{int(ex)}"] = mt
    (OUTD / "an.txt").write_text(chr(10).join(OUT))
    FIT.write_text(json.dumps(fit, indent=1))


def replay_metrics(D, fit, rng, exempt):
    import numpy as np

    keys = D.gk.to_numpy()
    uk, gi = np.unique(keys, return_inverse=True)
    R, C = latents(rng, fit, len(uk))
    S = D.S.to_numpy().astype(int)
    comp = D.ptype.map({"punt": 2, "td": 1, "fg": 1, "fgmiss": 1, "defscore": 1}).fillna(-1).to_numpy().astype(int)
    z = np.zeros(len(D))
    for j in (1, 2):
        m = comp == j
        gm, sm = gi[m], S[m]
        z[m] = (R[gm, sm, 0] + R[gm, sm, j]) - (C[gm, 1 - sm, 0] + C[gm, 1 - sm, j])
    if exempt:
        z = np.where(D.yl0.to_numpy() >= 75.0, 0.0, z)
    d = D.assign(ylw=D.ylw.to_numpy() + z)
    out = {"c": parity_cov(d)[0], "xs": cross_side(d)[0]}
    for gname in ("K", "P"):
        x = d[d.ptype.isin(GROUPS[gname])].copy()
        x["k"] = x.groupby(["gk", "S"]).cumcount() % 2
        out[gname] = parity_cov(x)[0]
    M = d.groupby(["gk", "S", d.ptype.map({t: g for g, ts in GROUPS.items() for t in ts}).fillna("O")]).ylw.mean().unstack()
    mm = M[["K", "P"]].dropna()
    out["KP"] = float(np.cov(mm.K, mm.P)[0, 1])
    return out


def latents(rng, fit, n):
    import numpy as np

    V = np.array([fit["V"]["s"], fit["V"]["K"], fit["V"]["P"]])
    w, kap = float(fit["w"]), float(fit["kappa"])
    z1 = rng.normal(size=(n, 2, 3))
    z2 = rng.normal(size=(n, 2, 3))
    R = np.sqrt(w * V) * z1
    C = np.sqrt((1.0 - w) * V) * (kap * z1 + np.sqrt(1.0 - kap * kap) * z2)
    return R, C


def load_fit():
    return json.loads(FIT.read_text())


def install_stf():
    import sys as _sys

    import mod25d_variance as dv

    fit = load_fit()
    team = {"home": 0, "away": 1}
    state = {"k": None, "rng": None, "gsr": None, "z": None}
    base = dv._G["pol"]

    def engine_frame():
        fr = _sys._getframe(2)
        while fr is not None and "offense" not in fr.f_locals:
            fr = fr.f_back
        return fr

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        fr = engine_frame()
        L = fr.f_locals
        offense, off_to, def_to = L["offense"], L["off_to"], L["def_to"]  # noqa: F841
        qtr_l, gsr, possessions = L.get("qtr"), L.get("gsr"), L.get("possessions")  # noqa: F841
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        if state["k"] != dv._G.get("task_key"):
            state["k"] = dv._G.get("task_key")
            state["rng"] = dv.task_rng(9001, int(dv._G["cfg"].get("seed", 3)))
            state["gsr"] = None
        g = gsr if gsr is not None else 3600.0
        if state["gsr"] is None or g > state["gsr"] + 1.0 or possessions == 0:
            rng = state["rng"]
            state["z"] = tuple(a[0] for a in latents(rng, fit, 1))
        state["gsr"] = g
        if not bool(drawn["flip"]) or qtr > 4:
            return drawn
        recv = team["away" if offense == "home" else "home"]
        kick = team[offense]
        po, pdf = float(drawn["points_off"]), float(drawn["points_def"])
        if po >= 3 or pdf >= 6:
            j = 1
        elif int(drawn.get("play_type_code", -1)) == 2:
            j = 2
        else:
            return drawn
        R, C = state["z"]
        z = float(R[recv, 0] + R[recv, j] - C[kick, 0] - C[kick, j])
        ny = float(drawn["next_yardline"])
        new = dict(drawn)
        ny2 = float(min(max(ny + z, 1.0), 99.0))
        nd = float(drawn["next_distance"])
        new["next_yardline"] = ny2
        new["next_distance"] = ny2 if nd >= ny else min(nd, ny2)
        return new

    dv._G["pol"] = pol


def H_budget(setting):
    import mod25e_crH as c

    c.H_budget(setting)
    if enabled():
        install_stf()


def H_ss(setting):
    import mod25e_crH as c

    c.H_ss(setting)
    if enabled():
        install_stf()


def cmd_smoke():
    import argparse

    import mod25e_crG as crG
    import mod25e_crH as c

    c.install()
    crG.crG_budget = H_budget
    crG.crG_ss = H_ss
    c.crG.cmd_sim(argparse.Namespace(off="", worlds=1, seasons=1, workers=1, seed=31, out_dir=str(OUTD / "smoke")))


if __name__ == "__main__":
    cmd_smoke() if sys.argv[1:] == ["smoke"] else cmd_an()
