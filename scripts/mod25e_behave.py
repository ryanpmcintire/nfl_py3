import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
sys.argv = sys.argv[:1]
LABEL = os.environ.get("BH_LABEL", "crHpqokgndecsk2mfwt")
SEEDS = tuple(int(x) for x in os.environ.get("BH_SEEDS", "11,12,13").split(","))
ART = REPO / "artifacts" / "mod25e3"
OUTD = ART / "behave"
OUT = []
LOOKS = [0]
BN = ["trail>8", "trail1-8", "tied", "lead1-8", "lead>8"]
EDGES = np.array([-8.5, -0.5, 0.5, 8.5])
rng = np.random.default_rng(75)


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def iv(d, pt):
    LOOKS[0] += 1
    d = np.asarray(d)
    return f"{pt:+.4f} [{np.percentile(d, 5):+.4f},{np.percentile(d, 95):+.4f}] pp {float((d > 0).mean()):.2f}"


def real_data():
    import mod25e_srcconc as sc

    tables, trans, pbp = sc.engine()
    pb = pbp.drop_duplicates(["game_id", "play_id"])[["game_id", "play_id", "epa", "season", "posteam", "defteam", "down", "ydstogo", "yardline_100"]]
    t = trans[["game_id", "play_id", "play_type_code", "qtr_actual", "gsr_actual", "sc_raw", "yards_gained", "is_home_off", "home_margin_post"]].reset_index(drop=True)
    t = t.merge(pb, on=["game_id", "play_id"], how="left")
    fm = t.groupby("game_id", sort=False).home_margin_post.last()
    gid, gn = pd.factorize(t.game_id.to_numpy())
    t["gid"] = gid
    ok = t.play_type_code.isin([0, 1]) & (t.qtr_actual <= 4) & t.epa.notna() & t.sc_raw.notna() & t.down.notna() & t.posteam.notna()
    t = t[ok].sort_values(["gid", "play_id"], kind="stable").reset_index(drop=True)
    tm = pd.factorize(pd.concat([t.posteam, t.defteam]))[0]
    n = len(t)
    return {"gid": t.gid.to_numpy(), "G": len(gn), "side": t.is_home_off.to_numpy().astype(int), "q": t.qtr_actual.to_numpy().astype(int), "sd": t.sc_raw.to_numpy(float),
            "gsr": t.gsr_actual.to_numpy(float), "code": t.play_type_code.to_numpy().astype(int), "down": t.down.to_numpy().astype(int), "dist": t.ydstogo.to_numpy(float),
            "yl": t.yardline_100.to_numpy(float), "ots": t.season.to_numpy().astype(int) * 64 + tm[:n], "dts": t.season.to_numpy().astype(int) * 64 + tm[n:],
            "epa": t.epa.to_numpy(float), "yards": t.yards_gained.to_numpy(float), "fm": fm.reindex(gn).to_numpy(float)}, (tables, trans, pbp)


def sim_data(ctx):
    import mod25e_late as late

    tables, trans, pbp = ctx
    epa_t = trans[["game_id", "play_id"]].merge(pbp[["game_id", "play_id", "epa"]].drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left").epa.to_numpy(float)
    cols = ["g", "qtr", "offhome", "yl", "down", "dist", "sd", "gsr", "code", "yards", "idx"]
    parts = {k: [] for k in ("gk", "side", "q", "sd", "gsr", "code", "down", "dist", "yl", "ots", "dts", "epa", "yards")}
    gmargin = {}
    tmap = {}
    fidx = 0
    for sdn in SEEDS:
        sdir = ART / f"e5_{LABEL}_s{sdn}"
        sg = pd.read_parquet(sdir / "sim_games.parquet", columns=["game_id", "home_team", "away_team", "home_score", "away_score"])
        sg["w"] = sg.game_id.str[1:5].astype(int)
        sg["s"] = sg.game_id.str[6:8].astype(int)
        sg["g"] = sg.game_id.str[-3:].astype(int)
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < late.BURN:
                continue
            d = pd.read_parquet(f, columns=cols)
            d = d[(d.qtr <= 4) & d.code.isin([0, 1]) & d.idx.notna()].sort_values(["g"], kind="stable").reset_index(drop=True)
            tmg = sg[(sg.w == w) & (sg.s == s)].set_index("g")
            gi = d.g.to_numpy().astype(int)
            ht = tmg.home_team.reindex(gi).to_numpy()
            at = tmg.away_team.reindex(gi).to_numpy()
            oh = d.offhome.to_numpy() == 1
            off, dfn = np.where(oh, ht, at), np.where(oh, at, ht)
            for nm in np.unique(np.r_[off, dfn]):
                tmap.setdefault(nm, len(tmap))
            fidx += 1
            parts["ots"].append(np.fromiter((fidx * 64 + tmap[x] for x in off), np.int64, len(off)))
            parts["dts"].append(np.fromiter((fidx * 64 + tmap[x] for x in dfn), np.int64, len(off)))
            base = ((sdn * 10000 + w) * 100 + s) * 100000
            parts["gk"].append(base + gi.astype(np.int64))
            for g_, hs, as_ in zip(tmg.index.to_numpy(), tmg.home_score.to_numpy(), tmg.away_score.to_numpy()):
                gmargin[base + int(g_)] = float(hs - as_)
            parts["side"].append(oh.astype(int))
            parts["q"].append(d.qtr.to_numpy().astype(int))
            parts["sd"].append(d.sd.to_numpy(float))
            parts["gsr"].append(d.gsr.to_numpy(float))
            parts["code"].append(d.code.to_numpy().astype(int))
            parts["down"].append(d.down.to_numpy().astype(int))
            parts["dist"].append(d.dist.to_numpy(float))
            parts["yl"].append(d.yl.to_numpy(float))
            parts["epa"].append(epa_t[d.idx.to_numpy().astype(int)])
            parts["yards"].append(d.yards.to_numpy(float))
    A = {k: np.concatenate(v) for k, v in parts.items()}
    m = np.isfinite(A["epa"]) & np.isfinite(A["down"]) & np.isfinite(A["sd"])
    A = {k: v[m] for k, v in A.items()}
    gu, gid = np.unique(A.pop("gk"), return_inverse=True)
    A["gid"] = gid
    A["G"] = len(gu)
    A["fm"] = np.array([gmargin[int(k)] for k in gu])
    return A


def cells(A):
    db = np.digitize(A["dist"], [1.5, 3.5, 6.5, 9.5, 10.5])
    yb = np.digitize(A["yl"], [10, 20, 40, 60, 80])
    return pd.factorize(((A["code"] * 5 + np.clip(A["down"], 1, 4)) * 6 + db) * 6 + yb)[0]


def lgo(A, v, G):
    ntg = 2 * G
    og = A["gid"] * 2 + A["side"]
    dg = A["gid"] * 2 + 1 - A["side"]
    ts_of = np.full(ntg, -1)
    ts_of[og] = A["ots"]
    ts_of[dg] = A["dts"]
    ok = ts_of >= 0
    tsu, tsi = np.unique(ts_of[ok], return_inverse=True)
    out = []
    for tg in (og, dg):
        S = np.bincount(tg, weights=v, minlength=ntg)
        N = np.bincount(tg, minlength=ntg).astype(float)
        TS = np.bincount(tsi, weights=S[ok])
        TN = np.bincount(tsi, weights=N[ok])
        Sk = np.zeros(ntg)
        Nk = np.zeros(ntg)
        Sk[ok], Nk[ok] = TS[tsi] - S[ok], TN[tsi] - N[ok]
        val = np.where(Nk > 0, Sk / np.maximum(Nk, 1), 0.0)
        out.append(val[tg])
    return out


def residual(A, y, cell):
    G = A["G"]
    cm = np.bincount(cell, weights=y) / np.bincount(cell)
    r0 = y - cm[cell]
    so, sdf = lgo(A, r0, G)
    X = np.c_[so, sdf]
    b, *_ = np.linalg.lstsq(X, r0, rcond=None)
    return r0 - X @ b, b


def net_delta(A):
    G = A["G"]
    og = A["gid"] * 2 + A["side"]
    dg = A["gid"] * 2 + 1 - A["side"]
    ts_of = np.full(2 * G, -1)
    ts_of[og] = A["ots"]
    ts_of[dg] = A["dts"]
    mg = np.zeros(2 * G)
    mg[0::2] = -A["fm"]
    mg[1::2] = A["fm"]
    ok = ts_of >= 0
    tsu, tsi = np.unique(ts_of[ok], return_inverse=True)
    TS = np.bincount(tsi, weights=mg[ok])
    TN = np.bincount(tsi).astype(float)
    net = np.zeros(2 * G)
    net[ok] = np.where(TN[tsi] > 1, (TS[tsi] - mg[ok]) / np.maximum(TN[tsi] - 1, 1), 0.0)
    return net[og] - net[dg]


def boot_w(G, nb):
    return [np.ones(G)] + [np.bincount(rng.integers(0, G, G), minlength=G).astype(float) for _ in range(nb)]


def slope_boot(gid, a, b, x, y, G, A_, B_, WS):
    key = (gid * A_ + a) * B_ + b
    k = G * A_ * B_
    n = np.bincount(key, minlength=k).astype(float).reshape(G, A_, B_)
    S = [np.bincount(key, weights=w, minlength=k).reshape(G, A_, B_) for w in (x, y, x * x, x * y)]
    pts, pool, per, cell = None, [], [[] for _ in range(A_)], [[[] for _ in range(B_)] for _ in range(A_)]
    for i, w in enumerate(WS):
        N, Sx, Sy, Sxx, Sxy = (np.tensordot(w, z, 1) for z in (n, *S))
        Nq, Xq, Yq, XXq, XYq = (z.sum(axis=1) for z in (N, Sx, Sy, Sxx, Sxy))
        cv = XYq - Xq * Yq / np.maximum(Nq, 1)
        vr = XXq - Xq * Xq / np.maximum(Nq, 1)
        po = cv.sum() / vr.sum()
        qs = cv / np.maximum(vr, 1e-12)
        ce = Sy / np.maximum(N, 1) - (Yq / np.maximum(Nq, 1))[:, None]
        if i == 0:
            pts = (po, qs, ce)
        else:
            pool.append(po)
            for q in range(A_):
                per[q].append(qs[q])
                for bb in range(B_):
                    cell[q][bb].append(ce[q, bb])
    return pts, (np.array(pool), [np.array(p) for p in per], [[np.array(c) for c in r] for r in cell])


def band_idx(sd):
    return np.digitize(sd, EDGES)


def cmp_lines(tag, pr, ps):
    ptr, drr = pr
    pts, drs = ps
    say(f"  {tag} pooled within-quarter slope per 7 pts: real {iv(drr[0], ptr[0])}; sim {iv(drs[0], pts[0])}; sim-real {iv(drs[0] - drr[0], pts[0] - ptr[0])}")
    for q in range(4):
        say(f"    Q{q + 1}: real {iv(drr[1][q], ptr[1][q])}; sim {iv(drs[1][q], pts[1][q])}; sim-real {iv(drs[1][q] - drr[1][q], pts[1][q] - ptr[1][q])}")


def wcov(w, x, y):
    sw = w.sum()
    mx, my = (w * x).sum() / sw, (w * y).sum() / sw
    return (w * (x - mx) * (y - my)).sum() / sw


def half_table(A, r):
    G = A["G"]
    tg = A["gid"] * 2 + A["side"]
    half = (A["q"] >= 3).astype(int)
    key = tg * 2 + half
    order = np.argsort(key, kind="stable")
    ks = key[order]
    start = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    pos = np.arange(len(ks)) - np.repeat(start, np.diff(np.r_[start, len(ks)]))
    par = np.empty(len(ks), int)
    par[order] = pos % 2
    out = {}
    nt = 2 * G
    for h in (0, 1):
        m = half == h
        out[f"n{h}"] = np.bincount(tg[m], minlength=nt).astype(float)
        out[f"s{h}"] = np.bincount(tg[m], weights=r[m], minlength=nt)
        for p in (0, 1):
            mm = m & (par == p)
            out[f"n{h}{p}"] = np.bincount(tg[mm], minlength=nt).astype(float)
            out[f"s{h}{p}"] = np.bincount(tg[mm], weights=r[mm], minlength=nt)
    ok = (out["n00"] > 0) & (out["n01"] > 0) & (out["n10"] > 0) & (out["n11"] > 0)
    ok = ok & ok[np.arange(nt) ^ 1]
    return out, ok


def part3(A, r, WS):
    G = A["G"]
    T, ok = half_table(A, r)
    x = {k: np.where(ok, T["s" + k] / np.maximum(T["n" + k], 1), 0.0) for k in ("0", "1", "00", "01", "10", "11")}
    idx = np.arange(2 * G)
    opp = idx ^ 1
    gtg = idx // 2
    n1, n2 = T["n0"][ok].mean(), T["n1"][ok].mean()
    res = {k: [] for k in ("cov12", "cov_oe", "cov_oe2", "slope", "slope2", "rel", "rel2", "cov_opp", "var1")}
    pt = None
    for i, W in enumerate(WS):
        w = W[gtg] * ok
        c12 = wcov(w, x["0"], x["1"])
        coe = wcov(w, x["00"], x["01"])
        coe2 = wcov(w, x["10"], x["11"])
        v1 = wcov(w, x["0"], x["0"])
        v2 = wcov(w, x["1"], x["1"])
        copp = wcov(w, x["0"], x["1"][opp])
        d = {"cov12": c12, "cov_oe": coe, "cov_oe2": coe2, "slope": c12 / coe if coe > 0 else np.nan, "slope2": c12 / np.sqrt(max(coe, 1e-12) * max(coe2, 1e-12)),
             "rel": coe / v1, "rel2": coe2 / v2, "cov_opp": copp, "var1": v1}
        if i == 0:
            pt = d
        else:
            for k in res:
                res[k].append(d[k])
    return pt, {k: np.array(v) for k, v in res.items()}, (n1, n2)


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    R, ctx = real_data()
    S = sim_data(ctx)
    say(f"## behave: real plays {len(R['gid'])} games {R['G']}; sim plays {len(S['gid'])} games {S['G']} ({LABEL} seeds {SEEDS}); bootstrap {NB}; run/pass plays qtr 1-4 only")
    for A in (R, S):
        A["cell"] = cells(A)
        A["delta"] = net_delta(A)
        A["mstar"] = A["sd"] - A["delta"] * (3600.0 - A["gsr"]) / 3600.0
        A["y"] = {"epa": A["epa"], "succ": (A["epa"] > 0).astype(float), "expl": (A["yards"] >= 15).astype(float)}
        A["ws"] = boot_w(A["G"], NB)
    res = {}
    say("## 1. Offence efficiency by own lead band x quarter, net of play type x down x distance x field-position cells and of leave-game-out offence and defence season strength")
    say("  (the defence facing the offence sees the mirrored band; offence and defence responses share the same plays and are identified only as their sum)")
    for o in ("epa", "succ", "expl"):
        P = {}
        for nm, A in (("real", R), ("sim", S)):
            r, b = residual(A, A["y"][o], A["cell"])
            A["r_" + o] = r
            say(f"  {o} {nm}: strength slopes off {b[0]:+.3f} def {b[1]:+.3f}; residual sd {r.std():.4f}; mean {A['y'][o].mean():+.4f}")
            P[nm] = slope_boot(A["gid"], A["q"] - 1, band_idx(A["sd"]), A["sd"] / 7.0, r, A["G"], 4, 5, A["ws"])
        res[o] = P
        cmp_lines(o, P["real"], P["sim"])
    say("  EPA band mean minus quarter mean (per play), band x quarter: real | sim | sim-real")
    ptr, drr = res["epa"]["real"]
    pts, drs = res["epa"]["sim"]
    for q in range(4):
        row = []
        for b in range(5):
            row.append(f"{BN[b]} {ptr[2][q, b]:+.3f} | {pts[2][q, b]:+.3f} | {iv(drs[2][q][b] - drr[2][q][b], pts[2][q, b] - ptr[2][q, b])}")
        say(f"    Q{q + 1}: " + "; ".join(row))
    say("## 2. Residual EPA/play vs lead, by time remaining (8 bins of 450 s; slope per 7 pts of lead; raw score diff and strength-adjusted margin = diff minus pregame leave-game-out net-margin gap x elapsed fraction)")
    for xn, xk in (("raw diff", "sd"), ("adjusted margin", "mstar")):
        P = {}
        for nm, A in (("real", R), ("sim", S)):
            tbin = np.clip(((3600.0 - A["gsr"]) // 450).astype(int), 0, 7)
            P[nm] = slope_boot(A["gid"], tbin, np.zeros(len(tbin), int), A[xk] / 7.0, A["r_epa"], A["G"], 8, 1, A["ws"])
        ptr, drr = P["real"]
        pts, drs = P["sim"]
        say(f"  {xn}: pooled within-time-bin slope real {iv(drr[0], ptr[0])}; sim {iv(drs[0], pts[0])}; sim-real {iv(drs[0] - drr[0], pts[0] - ptr[0])}")
        for tb in range(8):
            say(f"    t{tb} ({tb * 450}-{tb * 450 + 450}s): real {iv(drr[1][tb], ptr[1][tb])}; sim {iv(drs[1][tb], pts[1][tb])}; sim-real {iv(drs[1][tb] - drr[1][tb], pts[1][tb] - ptr[1][tb])}")
    say("## 3. Halftime: slope of H2 offence residual EPA/play on H1 residual, same team-game, net of noise (denominator = covariance of odd and even plays within H1 = true H1 variance)")
    for variant, key in (("net of cell+strength", "r_epa"), ("also net of lead band x quarter mean", "r2")):
        P = {}
        for nm, A in (("real", R), ("sim", S)):
            if key == "r2":
                bq = (A["q"] - 1) * 5 + band_idx(A["sd"])
                mm = np.bincount(bq, weights=A["r_epa"]) / np.bincount(bq)
                A["r2"] = A["r_epa"] - mm[bq]
            P[nm] = part3(A, A[key], A["ws"])
        say(f"  {variant}")
        pr, ps = P["real"], P["sim"]
        for k, lab in (("cov12", "cov(H1,H2) own"), ("cov_oe", "cov(H1 odd,H1 even)"), ("slope", "deattenuated slope H2~H1"), ("rel", "H1 reliability"), ("cov_opp", "cov(own H1, opponent H2)")):
            say(f"    {lab}: real {iv(pr[1][k], pr[0][k])}; sim {iv(ps[1][k], ps[0][k])}; sim-real {iv(ps[1][k] - pr[1][k], ps[0][k] - pr[0][k])}")
        n1r, n2r = pr[2]
        n1s, n2s = ps[2]
        dpt_own = n1s * n2s * ps[0]["cov12"] - n1r * n2r * pr[0]["cov12"]
        dpt_opp = n1s * n2s * ps[0]["cov_opp"] - n1r * n2r * pr[0]["cov_opp"]
        dd_own = n1s * n2s * ps[1]["cov12"] - n1r * n2r * pr[1]["cov12"]
        dd_opp = n1s * n2s * ps[1]["cov_opp"] - n1r * n2r * pr[1]["cov_opp"]
        say(f"    plays per half real {n1r:.1f}/{n2r:.1f} sim {n1s:.1f}/{n2s:.1f}; cross-half point covariance gap (plays x plays x cov, inferred 1 EPA = 1 point): own {iv(dd_own, dpt_own)}; opp {iv(dd_opp, dpt_opp)}")
        say(f"    margin-style cross-half persistence gap 2(own-opp) {iv(2 * (dd_own - dd_opp), 2 * (dpt_own - dpt_opp))} points^2 vs catch-up E gap +14.8 (E58, inferred mapping)")
    say(f"looks: interval prints {LOOKS[0]}; family = 3 outcomes x (1 pooled + 4 quarters) x 3 + 20 EPA band cells + 2 x (1 + 8) time slopes x 3 + 2 variants x 5 half stats x 3 + 4 gap lines")
    (OUTD / "behave.txt").write_text("\n".join(OUT), encoding="utf-8")


if __name__ == "__main__":
    main()
