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

OUTD = REPO / "artifacts" / "mod25e3" / "e91"
NB = 300
OUT = []
LOOKS = [0]
GROUPS = ["all", "punt", "to", "fgmiss", "fg", "td"]
rng = np.random.default_rng(91)


def say(s=""):
    print(s)
    OUT.append(s)


def prep(D):
    D = D.copy()
    gk = D.gk.to_numpy()
    same = np.r_[False, gk[1:] == gk[:-1]]
    off = D.off.to_numpy()
    D["poff"] = np.r_[[None], off[:-1]]
    D["pylast"] = np.r_[[np.nan], D.ylast.to_numpy()[:-1]]
    D["chgp"] = same & (D.poff.to_numpy() != off)
    S = D[(D.h == 1) & D.chgp & D.ptype.isin(GROUPS[1:]) & D.pylast.notna()].copy()
    S["S"] = (S.s < 0).astype(int)
    S["Sopp"] = 1 - S.S
    X = a2.xtable(D).rename(columns={"S": "Sopp", "x": "xo"})
    S = S.merge(X, on=["gk", "Sopp"], how="inner")
    S["e_raw"] = 100.0 - S.pylast
    S["s_raw"] = S.yl0
    S["t_raw"] = S.s_raw - S.e_raw
    return S


def centre(S, cols):
    S = S.copy()
    for c in cols:
        S[c + "_c"] = S[c] - S.groupby([S.q0, S.bi])[c].transform("mean")
    return S


def boot_slope(x, y, gi, W):
    G = W.shape[1]
    one = np.bincount(gi, minlength=G).astype(float)
    Sx = np.bincount(gi, weights=x, minlength=G)
    Sy = np.bincount(gi, weights=y, minlength=G)
    Sxx = np.bincount(gi, weights=x * x, minlength=G)
    Sxy = np.bincount(gi, weights=x * y, minlength=G)
    N = W @ one
    mx, my = (W @ Sx) / N, (W @ Sy) / N
    return ((W @ Sxy) / N - mx * my) / ((W @ Sxx) / N - mx**2)


def fmt(a, d=4):
    return f"{a[0]:+.{d}f} [{np.quantile(a[1:], .025):+.{d}f},{np.quantile(a[1:], .975):+.{d}f}]"


def gapline(R, Sm, label):
    d = Sm - R
    LOOKS[0] += 1
    return f"  {label:34s} real {fmt(R)} | sim {fmt(Sm)} | s-r {fmt(d)} P(s-r>0) {np.mean(d[1:] > 0):.2f}"


def part1(Dr, Ds):
    say("1. Decomposition of B's H2 start yardline (B yardline_100) after a possession change on A's H1 residual x (xo); s = start, e = 100 - yardline_100 of A's last live play (B perspective), t = s - e (net punt, return, turnover return, final-play gain); s = e + t exactly; per-drive slopes, state-cell (quarter x lead band) centred within group; game bootstrap")
    res = {}
    for nm, D in (("real", Dr), ("sim", Ds)):
        S = centre(prep(D), ["s_raw", "e_raw", "t_raw"])
        gk = pd.Index(sorted(S.gk.unique()))
        gi = gk.get_indexer(S.gk)
        W = np.vstack([np.ones((1, len(gk))), rng.multinomial(len(gk), np.ones(len(gk)) / len(gk), size=NB).astype(float)])
        res[nm] = {}
        for gname in GROUPS:
            m = (S.ptype == gname).to_numpy() if gname != "all" else np.ones(len(S), bool)
            gi_m = gi[m]
            x = S.xo.to_numpy()[m]
            res[nm][gname] = {c: boot_slope(x, S[c + "_c"].to_numpy()[m], gi_m, W) for c in ("s_raw", "e_raw", "t_raw")}
            res[nm][gname]["n"] = int(m.sum())
        say(f"  {nm}: drives {len(S)} games {len(gk)} sd(x) {S.xo.std():.2f}; n by group " + " ".join(f"{g} {res[nm][g]['n']}" for g in GROUPS))
    for gname in GROUPS:
        say(f"group {gname}")
        for c, lab in (("s_raw", "start s (yd/pt)"), ("e_raw", "end-spot e"), ("t_raw", "transfer t")):
            say(gapline(res["real"][gname][c], res["sim"][gname][c], lab))
        gs = res["sim"][gname]["s_raw"] - res["real"][gname]["s_raw"]
        ge = res["sim"][gname]["e_raw"] - res["real"][gname]["e_raw"]
        gt = res["sim"][gname]["t_raw"] - res["real"][gname]["t_raw"]
        say(f"  share of s gap: e {ge[0] / gs[0]:+.2f}, t {gt[0] / gs[0]:+.2f} (point)")
    return res


def part2_mediation(Dr, Ds):
    say("2a. Attenuation test: slope of B start s on A end-spot e within start type (real vs sim), slope of e on x, and mediated product b_se*b_ex vs total slope of s on x")
    out = {}
    for nm, D in (("real", Dr), ("sim", Ds)):
        S = centre(prep(D), ["s_raw", "e_raw"])
        gk = pd.Index(sorted(S.gk.unique()))
        gi = gk.get_indexer(S.gk)
        W = np.vstack([np.ones((1, len(gk))), rng.multinomial(len(gk), np.ones(len(gk)) / len(gk), size=NB).astype(float)])
        out[nm] = {}
        for gname in ("punt", "to", "fgmiss"):
            m = (S.ptype == gname).to_numpy()
            e, s, x = S.e_raw_c.to_numpy()[m], S.s_raw_c.to_numpy()[m], S.xo.to_numpy()[m]
            g = gi[m]
            bse = boot_slope(e, s, g, W)
            bex = boot_slope(x, e, g, W)
            bsx = boot_slope(x, s, g, W)
            out[nm][gname] = (bse, bex, bsx)
    for gname in ("punt", "to", "fgmiss"):
        say(f"group {gname}")
        for i, lab in enumerate(("b_se (s on e)", "b_ex (e on x)", "b_sx (s on x)")):
            say(gapline(out["real"][gname][i], out["sim"][gname][i], lab))
        pr = out["real"][gname][0] * out["real"][gname][1]
        ps = out["sim"][gname][0] * out["sim"][gname][1]
        say(gapline(pr, ps, "mediated b_se*b_ex"))
    return out


def part2_flip(seeds):
    import sim09_u4g as u4g

    say("2b. Engine coupling on flip plays: drawn pool row yardline (pool.parquet[idx].yl) minus sim yardline on the flip play, rows whose pool flip/code/down agree with the sim play (policy overrides excluded)")
    pool = pd.read_parquet(u4g.OUT / "pool.parquet", columns=["yl", "flip", "code", "down"])
    py, pf, pc, pdn = (pool[c].to_numpy() for c in ("yl", "flip", "code", "down"))
    rows = []
    for sd in seeds:
        sdir = a2.ART / f"e5_{a2.LABEL}_s{sd}"
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < late.BURN:
                continue
            d = pd.read_parquet(f, columns=["g", "qtr", "yl", "down", "code", "po", "pdf", "flip", "idx"])
            d = d[(d.flip == 1) & (d.po + d.pdf == 0) & (d.qtr <= 4)]
            ix = d.idx.to_numpy().astype(int)
            rows.append(pd.DataFrame({"gk": f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str), "code": d.code, "yl": d.yl, "pyl": py[ix], "ok": (pf[ix] == (d.flip == 1)) & (pc[ix] == d.code) & (pdn[ix] == d.down)}))
    F = pd.concat(rows, ignore_index=True)
    say(f"  flip plays without score {len(F)}; pool-consistent {F.ok.mean():.3f}")
    F = F[F.ok].copy()
    F["gap"] = F.pyl - F.yl
    F["grp"] = np.select([F.code == 2, F.code.isin([0, 1])], ["punt", "to"], "other")
    gk = pd.Index(sorted(F.gk.unique()))
    gi = gk.get_indexer(F.gk)
    W = np.vstack([np.ones((1, len(gk))), rng.multinomial(len(gk), np.ones(len(gk)) / len(gk), size=NB).astype(float)])
    for gname in ("punt", "to", "all"):
        m = (F.grp == gname).to_numpy() if gname != "all" else np.ones(len(F), bool)
        g = gi[m]
        gap = F.gap.to_numpy()[m]
        yl = F.yl.to_numpy()[m]
        pyl = F.pyl.to_numpy()[m]
        G = len(gk)
        n = np.bincount(g, minlength=G).astype(float)
        sg = np.bincount(g, weights=gap, minlength=G)
        sa = np.bincount(g, weights=np.abs(gap), minlength=G)
        mean_gap = (W @ sg) / (W @ n)
        mean_abs = (W @ sa) / (W @ n)
        att = boot_slope(yl, pyl, g, W)
        LOOKS[0] += 3
        say(f"  {gname:5s} n {int(m.sum()):6d} mean gap (drawn row yl - sim yl) {fmt(mean_gap, 3)} mean |gap| {fmt(mean_abs, 2)} slope of drawn-row yl on sim yl {fmt(att, 3)} P(slope<1) {np.mean(att[1:] < 1):.2f}")
    sdg = F.gap.std()
    say(f"  all-flip sd(gap) {sdg:.2f}; sd(sim yl) {F.yl.std():.2f}; corr(drawn yl, sim yl) {np.corrcoef(F.yl, F.pyl)[0, 1]:.3f}")


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    ns = late.load_xq()
    real = late.real_play_drives(ns["POOL"])
    sim = pd.concat([a2.sim_drives(sd) for sd in a2.SEEDS], ignore_index=True)
    rc, _ = ns["components"](real)
    fpmap = pd.Series(rc.p.to_numpy() - rc.p.mean() - rc.c_strength.to_numpy()).groupby(pd.cut(rc.yl0, ns["YL_BINS"], right=False).to_numpy(), observed=True).mean()
    Dr = a2.decorate(ns, real, None, fpmap, None)
    Ds = a2.decorate(ns, sim, None, fpmap, None)
    say(f"E91 start-spot decomposition; label {a2.LABEL} seeds {a2.SEEDS}; no sim run; real games {Dr.gk.nunique()} sim games {Ds.gk.nunique()}; boots {NB}")
    Hr, Hs = a2.drive_x(Dr), a2.drive_x(Ds)
    br, bs = a2.lsq(Hr.xo.to_numpy(), Hr.ylc.to_numpy()), a2.lsq(Hs.xo.to_numpy(), Hs.ylc.to_numpy())
    say(f"E88 reproduction (all B H2 drives, ylc on xo): real {br:+.4f} sim {bs:+.4f} gap s-r {bs - br:+.4f} yd/pt")
    LOOKS[0] += 1
    part1(Dr, Ds)
    part2_mediation(Dr, Ds)
    part2_flip(a2.SEEDS)
    say(f"looks {LOOKS[0]}")
    (OUTD / "e91.txt").write_text("\n".join(OUT))


if __name__ == "__main__":
    main()
