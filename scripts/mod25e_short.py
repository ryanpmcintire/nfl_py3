import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_draw as dr
import mod25e_late as ml

ART = REPO / "artifacts" / "mod25e3"
LABEL = os.environ.get("SHORT_LABEL", "crHpqokgnd")
SEEDS = tuple(int(x) for x in os.environ.get("SHORT_SEEDS", "11").split(","))
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = []
LOOKS = [0]
RNG = np.random.default_rng(9)


def say(s=""):
    print(s)
    OUT.append(s)


SC = None
TM = None


def load_sim(a, term):
    use = ["g", "down", "dist", "yl", "qtr", "code", "idx"]
    out = []
    gid = 0
    for sd in SEEDS:
        for f in sorted((ART / f"e5_{LABEL}_s{sd}").glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < ml.BURN:
                continue
            d = pd.read_parquet(f, columns=use)
            d = d[d.qtr <= 4].reset_index(drop=True)
            d["gk"] = gid + pd.factorize(d.g)[0]
            gid += int(d.g.nunique())
            out.append(d)
    S = pd.concat(out, ignore_index=True)
    S = S[S.code.isin([0, 1]) & S.down.isin([2, 4])].copy()
    ix = S.idx.to_numpy().astype(int)
    S["dist"] = np.round(S.dist).astype(int)
    S["sd"] = np.round(a["dist_raw"][ix]).astype(int)
    S["y"] = np.round(a["yards_gained"][ix]).astype(int)
    S["pterm"] = term[ix]
    S["ptd"] = a["points_off"][ix] >= 6
    S["pnd"] = a["next_down"][ix]
    S["sdown"] = a["down_i"][ix]
    S["sfp"] = a["fp_raw"][ix]
    S["soff"] = a["off_row"][ix]
    S["ssc"] = SC[ix]
    S["stm"] = TM[ix]
    S["scode"] = a["play_type_code"][ix]
    S["fd"] = (~S.pterm & (S.pnd == 1)) | S.ptd
    S["yc"] = (~S.pterm & (S.y >= S.dist)) | S.ptd
    return S


def load_real(a, term, ipw, sc, tm):
    R = pd.DataFrame({"down": a["down_i"], "dist": np.round(a["dist_raw"]).astype(int), "yl": a["fp_raw"], "code": a["play_type_code"], "y": np.round(a["yards_gained"]).astype(int),
                      "pterm": term, "ptd": a["points_off"] >= 6, "pnd": a["next_down"], "off": a["off_row"], "ipw": ipw, "sc": sc, "tm": tm})
    R = R[R.code.isin([0, 1]) & R.down.isin([2, 4])].copy()
    R["fd"] = (~R.pterm & (R.pnd == 1)) | R.ptd
    R["yc"] = (~R.pterm & (R.y >= R.dist)) | R.ptd
    return R


def boot(vals, gk, ng, real_vals):
    sk = np.bincount(gk, weights=vals.astype(float), minlength=ng)
    sn = np.bincount(gk, minlength=ng).astype(float)
    est_s = sk.sum() / max(sn.sum(), 1)
    est_r = float(np.mean(real_vals))
    nr = len(real_vals)
    d = np.empty(NB)
    for b in range(NB):
        w = RNG.multinomial(ng, np.ones(ng) / ng)
        d[b] = (w * sk).sum() / max((w * sn).sum(), 1) - real_vals[RNG.integers(0, nr, nr)].mean()
    LOOKS[0] += 1
    lo, hi = np.percentile(d, [2.5, 97.5])
    return est_r, est_s, est_s - est_r, lo, hi, float((d > 0).mean())


def fmt(nm, t):
    return f"{nm} real {t[0]:.3f} sim {t[1]:.3f} diff {t[2]:+.3f} [{t[3]:+.3f},{t[4]:+.3f}] pp {t[5]:.2f}"


def cell(S, R, dn, lo, hi, ng, label):
    ms = (S.down == dn) & (S.dist >= lo) & (S.dist <= hi)
    mr = (R.down == dn) & (R.dist >= lo) & (R.dist <= hi)
    s, r = S[ms], R[mr]
    s = s[~s.pterm | s.ptd] if False else s
    sg = s.gk.to_numpy()
    say(f"-- {label} n sim {len(s)} real {len(r)}")
    say(f"source down == state down {(s.sdown == dn).mean():.3f}; sd==dist {(s.sd == s.dist).mean():.3f} sd>dist {(s.sd > s.dist).mean():.3f} sd<dist {(s.sd < s.dist).mean():.3f}; scode in 0/1 {s.scode.isin([0, 1]).mean():.3f}")
    say(fmt("run share", boot((s.code == 0).to_numpy(), sg, ng, (r.code == 0).to_numpy().astype(float))))
    say(fmt("conv state-sticks", boot(s.yc.to_numpy(), sg, ng, r.yc.to_numpy().astype(float))))
    say(fmt("conv source fdnb", boot(s.fd.to_numpy(), sg, ng, r.fd.to_numpy().astype(float))))
    ex = (s.sd == s.dist).to_numpy()
    say(fmt("conv fdnb | sd==dist", boot(s.fd.to_numpy()[ex], sg[ex], ng, r.fd.to_numpy().astype(float))) + f" (share {ex.mean():.3f})")
    say(fmt("conv fdnb | sd!=dist", boot(s.fd.to_numpy()[~ex], sg[~ex], ng, r.fd.to_numpy().astype(float))) + f" (share {(~ex).mean():.3f}; mean sd-dist {(s.sd - s.dist)[~ex].mean():+.2f})")
    for k, nm in ((0, "run"), (1, "pass")):
        rk, sk_ = r[r.code == k], s[s.code == k]
        say(fmt(f"conv state-sticks {nm}", boot(sk_.yc.to_numpy(), sk_.gk.to_numpy(), ng, rk.yc.to_numpy().astype(float))) + f" n {len(sk_)}/{len(rk)}")
    mix = np.mean([r[r.code == k].yc.mean() * (s.code == k).mean() for k in (0, 1)]) * 0
    wr = sum(r[r.code == k].yc.mean() * (s.code == k).mean() for k in (0, 1))
    say(f"real conv reweighted to sim run/pass mix {wr:.3f} (real {r.yc.mean():.3f}, sim {s.yc.mean():.3f}): mix part {wr - r.yc.mean():+.4f}, within-kind part {s.yc.mean() - wr:+.4f}")
    cuts = np.quantile(r.yl, [1 / 3, 2 / 3])
    zr, zs = np.digitize(r.yl, cuts), np.digitize(s.yl, cuts)
    wz = sum(r[zr == z].yc.mean() * (zs == z).mean() for z in range(3))
    say(f"real conv reweighted to sim zone mix {wz:.3f}: zone part {wz - r.yc.mean():+.4f}; mean state yl sim {s.yl.mean():.1f} real {r.yl.mean():.1f}; mean source fp sim {s.sfp.mean():.1f} (source minus state {(s.sfp - s.yl).mean():+.2f}, mean abs {(s.sfp - s.yl).abs().mean():.1f})")
    say(f"offence rating of source rows sim {s.soff.mean():+.4f} real pool cell {r.off.mean():+.4f} (diff {s.soff.mean() - r.off.mean():+.4f}); corr(source off, fdnb) sim {np.corrcoef(s.soff, s.fd)[0, 1]:+.3f} real {np.corrcoef(r.off, r.fd)[0, 1]:+.3f}")
    ex = (s.sd == s.dist).to_numpy()
    w = r.ipw.to_numpy()
    fd = r.fd.to_numpy().astype(float)
    n = len(r)
    bs = np.empty(NB)
    for b in range(NB):
        i = RNG.integers(0, n, n)
        bs[b] = (w[i] * fd[i]).sum() / w[i].sum() - fd[i].mean()
    lo_, hi_ = np.percentile(bs, [2.5, 97.5])
    say(f"IPW-weighted pool conv {(w * fd).sum() / w.sum():.3f} vs unweighted {fd.mean():.3f} (shift {(w * fd).sum() / w.sum() - fd.mean():+.4f} [{lo_:+.4f},{hi_:+.4f}]); sim fdnb | sd==dist {s.fd.to_numpy()[ex].mean():.3f}; IPW mean {w.mean():.3f} sd {w.std():.3f} corr(ipw, fdnb) {np.corrcoef(w, fd)[0, 1]:+.3f}")
    qs, qt = np.quantile(r.sc, [1 / 3, 2 / 3]), np.quantile(r.tm, [1 / 3, 2 / 3])
    cr = np.digitize(r.sc, qs) * 3 + np.digitize(r.tm, qt)
    cs_ = np.digitize(s.ssc, qs) * 3 + np.digitize(s.stm, qt)
    wsm = sum(r.fd.to_numpy()[cr == c].mean() * (cs_ == c).mean() for c in range(9) if (cr == c).any())
    say(f"pool conv reweighted to drawn-source score x time mix (3x3 terciles) {wsm:.3f} vs pool {r.fd.mean():.3f} (shift {wsm - r.fd.mean():+.4f}); drawn minus pool mean score {s.ssc.mean() - r.sc.mean():+.2f}, mean time {s.stm.mean() - r.tm.mean():+.0f}s; sim fdnb | sd==dist {s.fd.to_numpy()[ex].mean():.3f}")
    fb = np.array([0, 5, 10, 20, 35, 50, 65, 80, 90, 95, 101])
    fr, fs = np.digitize(r.yl, fb), np.digitize(s.sfp.to_numpy()[ex], fb)
    wf = sum(r.fd.to_numpy()[fr == c].mean() * (fs == c).mean() for c in np.unique(fr))
    say(f"pool conv reweighted to drawn (sd==dist) source fp mix over 10 bins {wf:.3f} vs pool {r.fd.mean():.3f} (shift {wf - r.fd.mean():+.4f}); distinct source rows {s.idx.nunique()} of {len(s)} draws, pool cell rows {len(r)}; top 1% rows carry {np.sort(s.idx.value_counts().to_numpy())[::-1][: max(1, len(r) // 100)].sum() / len(s):.3f} of draws")
    LOOKS[0] += 9


def main():
    G, cfg = dr.build()
    a = G["tables"]["arrays"]
    term = a["possession_flip"] | (a["points_off"] > 0) | (a["points_def"] > 0)
    import mod25d_variance as dv
    import sim04_engine as sim

    trans = sim.build_transition_frame(sim.load_reg_seasons(tuple(dv.TRAIN)))
    global SC, TM
    SC = trans["sc_raw"].to_numpy(dtype=float)
    TM = trans["time_raw"].to_numpy(dtype=float)
    assert len(SC) == len(a["down_i"])
    S = load_sim(a, term)
    R = load_real(a, term, np.asarray(G["ns"]["IPW"], dtype=float), SC, TM)
    ng = int(S.gk.max()) + 1
    say(f"{LABEL} s{SEEDS} sim go rows {len(S)}, pool go rows {len(R)}, game bootstrap {NB}; terminal rows kept via ptd credit, as in mod25e_dist P3")
    say("== 2nd-down short, cells by state distance (conv at state sticks = yards>=dist or TD)")
    for lo, hi in ((1, 1), (2, 3), (4, 6)):
        cell(S, R, 2, lo, hi, ng, f"2nd&{lo}" if lo == hi else f"2nd&{lo}-{hi}")
    say("== 4th-down go, cells as mod25e_dist P3 (dist 1 | 2-5 | 6+)")
    for lo, hi in ((1, 1), (2, 5), (6, 99)):
        cell(S, R, 4, lo, hi, ng, f"4th&{lo}" if lo == hi else f"4th&{lo}-{hi}")
    Sg, Rg = S[S.down == 4], R[R.down == 4]
    say(f"4th go dist tail: mean dist sim {Sg.dist.mean():.2f} real {Rg.dist.mean():.2f}; share dist>=11 sim {(Sg.dist >= 11).mean():.3f} real {(Rg.dist >= 11).mean():.3f}; share dist>=6 sim {(Sg.dist >= 6).mean():.3f} real {(Rg.dist >= 6).mean():.3f}")
    say(f"looks {LOOKS[0]}")
    d = ART / os.environ.get("SHORT_OUT", "short")
    d.mkdir(exist_ok=True)
    (d / "short.txt").write_text("\n".join(OUT), encoding="utf-8")


main()
