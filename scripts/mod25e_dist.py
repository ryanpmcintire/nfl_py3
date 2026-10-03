import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_draw as dr
import mod25e_late as ml

ART = REPO / "artifacts" / "mod25e3"
LABEL = "crHpqokg"
SEEDS = (11, 12, 13)
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = []
LOOKS = [0]
RNG = np.random.default_rng(5)


def say(s=""):
    print(s)
    OUT.append(s)


def sim_frame(a, term):
    use = ["g", "down", "dist", "yl", "qtr", "code", "offhome", "idx", "po", "flip"]
    out = []
    gid = 0
    for sd in SEEDS:
        sdir = ART / f"e5_{LABEL}_s{sd}"
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < ml.BURN:
                continue
            d = pd.read_parquet(f, columns=use)
            d = d[d.qtr <= 4].sort_values("g", kind="stable").reset_index(drop=True)
            g, oh, dn, ds = d.g.to_numpy(), d.offhome.to_numpy(), d.down.to_numpy(), d.dist.to_numpy()
            same = np.r_[(g[1:] == g[:-1]) & (oh[1:] == oh[:-1]), False]
            d["nd"] = np.where(same, np.r_[dn[1:], np.nan], np.nan)
            d["ndist"] = np.where(same, np.r_[ds[1:], np.nan], np.nan)
            d["gk"] = gid + pd.factorize(d.g)[0]
            gid += int(d.g.nunique())
            out.append(d)
    S = pd.concat(out, ignore_index=True)
    S = S[S.code.isin([0, 1]) & S.down.isin([1, 2, 3, 4])].copy()
    ix = S.idx.to_numpy().astype(int)
    S["dist"] = np.round(S.dist).astype(int)
    S["sd"] = np.round(a["dist_raw"][ix]).astype(int)
    S["y"] = np.round(a["yards_gained"][ix]).astype(int)
    S["pterm"] = term[ix]
    S["ptd"] = a["points_off"][ix] >= 6
    S["pnd"] = a["next_down"][ix]
    return S


def real_frame(a, term):
    R = pd.DataFrame({"down": a["down_i"], "dist": np.round(a["dist_raw"]).astype(int), "yl": a["fp_raw"], "code": a["play_type_code"], "y": np.round(a["yards_gained"]).astype(int),
                      "pterm": term, "ptd": a["points_off"] >= 6, "pnd": a["next_down"], "nd": a["next_down"], "ndist": a["next_distance"]})
    return R[R.code.isin([0, 1]) & R.down.isin([1, 2, 3, 4])].copy()


def metrics(df, nd_ok):
    y, d = df.y.to_numpy(), df.dist.to_numpy()
    m = {"conv": y >= d, "s1": y == d - 1, "s2": y == d - 2, "s3": y == d - 3, "short13": (y >= d - 3) & (y < d), "zero": y == 0, "neg": y < 0, "mean": y.astype(float)}
    if nd_ok:
        nd, ndist = df.nd.to_numpy(), df.ndist.to_numpy()
        is3 = nd == 3
        m["to3"] = is3
        m["to3_le3"] = is3 & (ndist <= 3)
        m["to3_ge8"] = is3 & (ndist >= 8)
    return m


def cell_stats(S, R, gcodes, ng, nd_ok):
    ms, mr = metrics(S, nd_ok), metrics(R, nd_ok)
    res = {}
    for k in ms:
        sk = np.bincount(gcodes, weights=ms[k].astype(float), minlength=ng)
        sn = np.bincount(gcodes, minlength=ng).astype(float)
        rn = len(R)
        rk = float(mr[k].sum())
        est_s, est_r = sk.sum() / max(sn.sum(), 1), rk / max(rn, 1)
        diffs = np.empty(NB)
        for b in range(NB):
            w = RNG.multinomial(ng, np.ones(ng) / ng)
            ss = (w * sk).sum() / max((w * sn).sum(), 1)
            if k == "mean":
                rs = R.y.to_numpy()[RNG.integers(0, rn, rn)].mean()
            else:
                rs = RNG.binomial(rn, est_r) / rn
            diffs[b] = ss - rs
        LOOKS[0] += 1
        res[k] = (est_r, est_s, est_s - est_r, float((diffs > 0).mean()))
    return res, len(S), len(R)


def part1(S, R):
    say("== P1 gain vs sticks, real pool rows vs sim draws at the same state (down, exact distance bin, zone, play type). nonterminal pool rows, yards>-20. real/sim (diff, pp = prob(diff>0))")
    cuts = np.quantile(R[R.down.isin([1, 2])].yl, [1 / 3, 2 / 3])
    say(f"zone cuts yl (real 1st/2nd terciles) {cuts[0]:.0f} {cuts[1]:.0f}")
    S = S[~S.pterm & (S.y > -20)].copy()
    R = R[~R.pterm & (R.y > -20)].copy()
    S["zone"] = np.digitize(S.yl, cuts)
    R["zone"] = np.digitize(R.yl, cuts)
    gcodes_all = S.gk.to_numpy()
    ng = int(gcodes_all.max()) + 1
    bins = {1: [("10", 10, 10)], 2: [("1", 1, 1), ("2-3", 2, 3), ("4-6", 4, 6), ("7-9", 7, 9), ("10+", 10, 99)]}
    for dn in (1, 2):
        for nm, lo, hi in bins[dn]:
            for zn in ("all", 0, 1, 2):
                for kn in ("all", 0, 1):
                    ms = (S.down == dn) & (S.dist >= lo) & (S.dist <= hi)
                    mr = (R.down == dn) & (R.dist >= lo) & (R.dist <= hi)
                    if zn != "all":
                        ms &= S.zone == zn
                        mr &= R.zone == zn
                    if kn != "all":
                        ms &= S.code == kn
                        mr &= R.code == kn
                    if ms.sum() < 300 or mr.sum() < 300:
                        continue
                    r, ns_, nr_ = cell_stats(S[ms], R[mr], gcodes_all[ms.to_numpy()], ng, dn == 2)
                    s = " ".join(f"{k} {v[0]:.3f}/{v[1]:.3f}({v[2]:+.3f},pp{v[3]:.2f})" for k, v in r.items())
                    say(f"d{dn} dist{nm} zone{zn} kind{kn} nS {ns_} nR {nr_} | {s}")


def part2(S):
    say("== P2 drawn source distance vs state distance (sim draws, all rows; sd = distance of the pool row drawn)")
    for dn in (1, 2, 3, 4):
        for nm, lo, hi in (("1", 1, 1), ("2-3", 2, 3), ("4-6", 4, 6), ("7-9", 7, 9), ("10", 10, 10), ("11+", 11, 99)):
            m = (S.down == dn) & (S.dist >= lo) & (S.dist <= hi)
            if m.sum() < 300:
                continue
            x = S[m]
            off = (x.sd - x.dist).to_numpy()
            ownc = (x.y >= x.sd).mean()
            stc = (x.y >= x.dist).mean()
            say(f"d{dn} dist{nm} n {m.sum()} mean(sd-dist) {off.mean():+.3f} P(sd==dist) {(off == 0).mean():.3f} P(sd<dist) {(off < 0).mean():.3f} P(sd>dist) {(off > 0).mean():.3f} | conv own-sticks {ownc:.3f} state-sticks {stc:.3f}")


def part3(S, R):
    say("== P3 4th-down go success by distance, real vs sim with credit read at the source row (fdnb) and yards-only")
    Rg = R[R.down == 4]
    Sg = S[S.down == 4]
    cuts = np.quantile(Rg.dist, [1 / 3, 2 / 3])
    say(f"4th go dist tercile cuts {cuts[0]:.0f} {cuts[1]:.0f}; n real {len(Rg)} sim {len(Sg)}")
    rs = ((~Rg.pterm & (Rg.pnd == 1)) | Rg.ptd).astype(float)
    rsy = ((~Rg.pterm & (Rg.y >= Rg.dist)) | Rg.ptd).astype(float)
    fd = ((~Sg.pterm & (Sg.pnd == 1)) | Sg.ptd).astype(float)
    ydo = ((~Sg.pterm & (Sg.y >= Sg.dist)) | Sg.ptd).astype(float)
    ng = int(Sg.gk.max()) + 1
    for b in range(3):
        mr, ms = np.digitize(Rg.dist, cuts, right=True) == b, np.digitize(Sg.dist, cuts, right=True) == b
        gk = Sg.gk.to_numpy()[ms]
        out = []
        for nm, sv, rv in (("fdnb", fd, rs), ("yardsonly", ydo, rsy)):
            sk = np.bincount(gk, weights=sv.to_numpy()[ms], minlength=ng)
            sn = np.bincount(gk, minlength=ng).astype(float)
            er, es = rv.to_numpy()[mr].mean(), sv.to_numpy()[ms].mean()
            nr = int(mr.sum())
            df = np.empty(NB)
            for i in range(NB):
                w = RNG.multinomial(ng, np.ones(ng) / ng)
                df[i] = (w * sk).sum() / max((w * sn).sum(), 1) - RNG.binomial(nr, er) / nr
            LOOKS[0] += 1
            out.append(f"{nm} real {er:.3f} sim {es:.3f} ({es - er:+.3f}, pp {(df > 0).mean():.2f})")
        x = Sg[ms]
        say(f"tercile{b} n sim {int(ms.sum())} real {int(mr.sum())} mean dist sim {x.dist.mean():.2f} real {Rg.dist.to_numpy()[mr].mean():.2f} mean(sd-dist) {(x.sd - x.dist).mean():+.2f} | " + " | ".join(out))


def part4(R):
    say("== P4 kernel width on distance, held-out log loss of P(first down | distance) on pool rows; 9 contiguous blocks (pool is chronological); prior m=1 at the down base rate; h = c*sqrt(max(1,dist)), c=0 exact; engine c=0.6")
    cs = (0.0, 0.15, 0.3, 0.6, 1.2, 2.4)
    nf = 9
    for dn in (2, 3, 4):
        X = R[R.down == dn].reset_index(drop=True)
        conv = ((~X.pterm & (X.pnd == 1)) | X.ptd).to_numpy().astype(float)
        dd = np.clip(X.dist.to_numpy(), 1, 30)
        fold = np.minimum((np.arange(len(X)) * nf) // len(X), nf - 1)
        grid = np.arange(1, 31)
        ll = np.zeros((nf, len(cs)))
        for f in range(nf):
            tr = fold != f
            n = np.bincount(dd[tr], minlength=31)[1:].astype(float)
            k = np.bincount(dd[tr], weights=conv[tr], minlength=31)[1:]
            base = conv[tr].mean()
            te = ~tr
            for ci, c in enumerate(cs):
                if c == 0:
                    W = np.eye(30)
                else:
                    h = c * np.sqrt(np.maximum(1, grid))
                    W = np.exp(-0.5 * ((grid[None, :] - grid[:, None]) / h[:, None]) ** 2)
                p = (W @ k + base) / (W @ n + 1.0)
                pe = np.clip(p[dd[te] - 1], 1e-6, 1 - 1e-6)
                ll[f, ci] = -np.mean(conv[te] * np.log(pe) + (1 - conv[te]) * np.log(1 - pe))
        mean = ll.mean(axis=0)
        ref = cs.index(0.6)
        say(f"down{dn} n {len(X)} base {conv.mean():.3f} | " + " ".join(f"c{c}:{m:.5f}" for c, m in zip(cs, mean)) + f" | best c {cs[int(mean.argmin())]}; folds where c beats 0.6: " + " ".join(f"c{c}:{int((ll[:, i] < ll[:, ref]).sum())}/{nf}" for i, c in enumerate(cs) if c != 0.6))
        LOOKS[0] += len(cs)


def main():
    G, cfg = dr.build()
    a = G["tables"]["arrays"]
    term = a["possession_flip"] | (a["points_off"] > 0) | (a["points_def"] > 0)
    S = sim_frame(a, term)
    R = real_frame(a, term)
    say(f"{LABEL} s{SEEDS} sim rp rows {len(S)}, pool rp rows {len(R)}, game bootstrap {NB}")
    part2(S)
    part3(S, R)
    part4(R)
    part1(S, R)
    say(f"looks {LOOKS[0]}")
    d = ART / "dist"
    d.mkdir(exist_ok=True)
    (d / "dist.txt").write_text("\n".join(OUT), encoding="utf-8")


main()
