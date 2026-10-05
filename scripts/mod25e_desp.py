import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
ART = REPO / "artifacts" / "mod25e3"
OUTD = ART / "desp"
NV = Path("C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad/nv")
LABEL = os.environ.get("DESP_LABEL", "crHpqokgndecsmfwt")
SEEDS = tuple(int(x) for x in os.environ.get("DESP_SEEDS", "11,12,13").split(","))
BURN = int(os.environ.get("KICK_BURN", 2))
EDGES = [-1e9, -8.5, -0.5, 0.5, 8.5, 1e9]
BANDS = ["trail>8", "trail1-8", "tied", "lead1-8", "lead>8"]
TB_EDGES = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
YB_EDGES = [20, 50, 80]
NB = 200
OUT = []
LAT = ["lateral_reception", "lateral_rush", "lateral_return", "lateral_recovery"]


def enabled():
    return os.environ.get("DESP") == "1"


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def frame():
    f = OUTD / "frame.parquet"
    if f.exists():
        return pd.read_parquet(f)
    import sim04_engine as sim

    import mod25d_variance as dv

    pbp = sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = sim.build_transition_frame(pbp)
    cols = ["game_id", "play_id", "down_i", "play_type_code", "dist_raw", "fp_raw", "sc_raw", "qtr_actual", "gsr_actual", "off_to_raw", "def_to_raw", "possession_flip", "points_off", "points_def", "yards_gained", "clock_elapsed"]
    F = tr[cols].copy().reset_index(drop=True)
    nv = pd.concat([pd.read_parquet(NV / f"pbp_{s}.parquet", columns=["game_id", "play_id", "season", "air_yards", "half_seconds_remaining", "pass_attempt", "play_type"] + LAT) for s in range(2009, 2018)], ignore_index=True)
    nv = nv[nv.season_type.eq("REG")] if "season_type" in nv else nv
    nv = nv.drop_duplicates(["game_id", "play_id"])
    n0 = len(F)
    F = F.merge(nv, on=["game_id", "play_id"], how="left", validate="m:1")
    assert len(F) == n0
    F["lat"] = F[LAT].fillna(0).to_numpy().max(1) > 0
    F["flip"] = F.possession_flip.astype(bool)
    OUTD.mkdir(parents=True, exist_ok=True)
    F.to_parquet(f)
    return F


def derive(F):
    code = F.play_type_code.to_numpy()
    rp = (code == 0) | (code == 1)
    po, pdf = F.points_off.to_numpy(), F.points_def.to_numpy()
    flip = F.flip.to_numpy()
    q = F.qtr_actual.to_numpy()
    gsr = F.gsr_actual.to_numpy()
    hsr = np.where(q <= 2, gsr - 1800.0, np.where(q <= 4, gsr, gsr))
    hsr = np.where(np.isfinite(F.half_seconds_remaining.to_numpy(float)), F.half_seconds_remaining.to_numpy(float), hsr)
    return dict(
        rp=rp, pas=code == 1, lost=rp & ((pdf > 0) | (flip & (po == 0))), td=rp & (po >= 6), hsr=hsr, q=q, gsr=gsr,
        band=pd.cut(F.sc_raw.to_numpy(), EDGES, labels=False).astype(int), half=(q >= 3).astype(int),
        to=np.minimum(F.off_to_raw.to_numpy(), 3).astype(int), yl=F.fp_raw.to_numpy(float), air=F.air_yards.to_numpy(float),
        yds=F.yards_gained.to_numpy(float), season=F.season.to_numpy(), g=pd.factorize(F.game_id)[0],
    )


def gboot(g, nb, rng):
    ng = int(g.max()) + 1
    W = rng.multinomial(ng, np.ones(ng) / ng, size=nb).astype(float)
    return np.r_[np.ones((1, ng)), W]


def grate(g, W, num, den):
    n = np.bincount(g, weights=num.astype(float), minlength=W.shape[1])
    d = np.bincount(g, weights=den.astype(float), minlength=W.shape[1])
    r = (W @ n) / np.maximum(W @ d, 1e-12)
    return r[0], np.percentile(r[1:], 2.5), np.percentile(r[1:], 97.5)


def fmt(t):
    return f"{t[0]:.4f} [{t[1]:.4f},{t[2]:.4f}]"


def dclass(d, thr, H):
    return d["rp"] & (F_lat(d) | (d["pas"] & (d["air"] >= thr) & (d["hsr"] <= H)))


def F_lat(d):
    return d["lat"]


def sim_flags(Dflag):
    out = []
    for sdn in SEEDS:
        sdir = ART / f"e5_{LABEL}_s{sdn}"
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < BURN:
                continue
            x = pd.read_parquet(f, columns=["g", "down", "yl", "sd", "gsr", "qtr", "code", "po", "pdf", "flip", "idx"])
            x = x[x.qtr <= 4].copy()
            x["gk"] = ((sdn * 10000 + w) * 100 + s) * 100000 + x.g.astype(np.int64)
            out.append(x)
    S = pd.concat(out, ignore_index=True)
    code = S.code.to_numpy()
    rp = (code == 0) | (code == 1)
    flip = S.flip.to_numpy().astype(bool)
    q, gsr = S.qtr.to_numpy(), S.gsr.to_numpy()
    d = dict(
        rp=rp, pas=code == 1, lost=rp & ((S.pdf.to_numpy() > 0) | (flip & (S.po.to_numpy() == 0))), td=rp & (S.po.to_numpy() >= 6),
        hsr=np.where(q <= 2, gsr - 1800.0, gsr), q=q, gsr=gsr, band=pd.cut(S.sd.to_numpy(), EDGES, labels=False).astype(int),
        half=(q >= 3).astype(int), g=pd.factorize(S.gk)[0], yl=S.yl.to_numpy(float), D=Dflag[S.idx.to_numpy().astype(int)],
    )
    return d


def chars():
    F = frame()
    d = derive(F)
    d["lat"] = F.lat.to_numpy()
    rng = np.random.default_rng(79)
    pa = d["pas"] & np.isfinite(d["air"])
    thr = float(np.percentile(d["air"][pa & (d["hsr"] > 120)], 99))
    say(f"## E79 chars: {len(F)} rows 2009-17 REG; code counts {dict(pd.Series(F.play_type_code).value_counts().sort_index())}; lateral rows {int(d['lat'].sum())} (run/pass {int((d['lat'] & d['rp']).sum())})")
    say(f"deep threshold: air_yards 99th percentile of passes with hsr>120 = {thr:.0f}")
    W = gboot(d["g"], NB, rng)
    base = d["pas"] & (d["hsr"] > 640)
    rb = grate(d["g"], W, base & pa & (d["air"] >= thr), base & pa)
    say(f"ordinary deep rate (hsr>640) {fmt(rb)}")
    edges = [0] + TB_EDGES[:-1] + [1e9]
    H = 0
    ok = True
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = d["pas"] & pa & (d["hsr"] >= lo) & (d["hsr"] < hi)
        r = grate(d["g"], W, m & (d["air"] >= thr), m)
        up = r[1] > rb[0] and ok
        ok = up
        if up:
            H = hi
        say(f"  hsr[{lo:.0f},{hi:.0f}) deep rate {fmt(r)} n {int(m.sum())} lateral rows {int((d['lat'] & (d['hsr'] >= lo) & (d['hsr'] < hi)).sum())}")
    say(f"H = largest window edge with lower bound above ordinary: {H}")
    D = dclass(d, thr, H)
    say(f"class D rows {int(D.sum())}: lateral {int((D & d['lat']).sum())}, deep-late {int((D & ~d['lat']).sum())}; per game {D.sum() / (d['g'].max() + 1):.4f}")
    json.dump(dict(thr=thr, H=H), open(OUTD / "def.json", "w"))
    np.save(OUTD / "D.npy", D)
    say("### outcome by class: D | non-D rp rows in hsr<=H | all rp")
    for nm, m in (("D", D), ("nonD hsr<=H", d["rp"] & ~D & (d["hsr"] <= H)), ("all rp", d["rp"])):
        say(f"  {nm:12s} n {int(m.sum()):6d} lost {d['lost'][m].mean():.3f} td {d['td'][m].mean():.3f} mean yards {d['yds'][m].mean():.1f}")
    say("### D rows per 1000 rp plays by deficit band x half-end window (real)")
    S = sim_flags(np.load(OUTD / "D.npy"))
    Ws = gboot(S["g"], NB, rng)
    for hf in (0, 1):
        for b in range(5):
            r = []
            for dd, WW, DD in ((d, W, D), (S, Ws, S["D"])):
                m = dd["rp"] & (dd["band"] == b) & (dd["half"] == hf) & (dd["hsr"] <= H)
                r.append((grate(dd["g"], WW, DD & m, m), int(m.sum())))
            say(f"  half{hf + 1} {BANDS[b]:9s} hsr<={H}: real {fmt(r[0][0])} n {r[0][1]} | sim {fmt(r[1][0])} n {r[1][1]}")
    say("### lost / td per rp play, trail>8, q4 gsr<=450: real vs sim; D share")
    for nm, dd, WW, DD in (("real", d, W, D), ("sim", S, Ws, S["D"])):
        m = dd["rp"] & (dd["band"] == 0) & (dd["q"] == 4) & (dd["gsr"] <= 450)
        say(f"  {nm}: lost {fmt(grate(dd['g'], WW, dd['lost'] & m, m))} td {fmt(grate(dd['g'], WW, dd['td'] & m, m))} Dshare {fmt(grate(dd['g'], WW, DD & m, m))} n {int(m.sum())}")
    pool_test(F, d, D, H)
    (OUTD / "chars.txt").write_text("\n".join(OUT), encoding="utf-8")


def pool_test(F, d, D, H):
    import mod25e_draw as dr
    import sim04_engine as sim

    import mod25d_variance as dv

    G, _ = dr.build()
    pbp = sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = sim.build_transition_frame(pbp)
    G["phase"] = tr["phase"].to_numpy()
    G["sc"] = tr["sc_raw"].to_numpy()
    G["tm"] = tr["time_raw"].to_numpy()
    G["oto"] = tr["off_to_raw"].to_numpy()
    G["dto"] = tr["def_to_raw"].to_numpy()
    assert len(G["tables"]["arrays"]["down_i"]) == len(D)
    rng = np.random.default_rng(80)
    k = int(sim.K_STATE)
    st = np.flatnonzero(d["rp"] & (d["band"] == 0) & (d["q"] == 4) & (d["gsr"] <= 450))
    pick = rng.choice(st, size=min(len(st), 3000), replace=False)
    inpool = []
    for r in pick:
        nb, _ = dr.neighbors_of(G, int(r), k)
        inpool.append(float(D[nb].mean()))
    inpool = np.array(inpool)
    realD = D[st].mean()
    say(f"### E79 pool: {len(pick)} real trail>8 q4 gsr<={450} states, kNN pool k={k}: mean D share in pool {inpool.mean():.4f}; share of states with >=1 D row {np.mean(inpool > 0):.3f}; real D share at those states {realD:.4f}; D rows in all tables {int(D.sum())}")


def design(half, tb, band, to, yb, form):
    n = len(half)
    cols = [np.ones((n, 1))]
    oh = lambda x, k: (x[:, None] == np.arange(k)[None, :]).astype(float)
    if form >= 1:
        cols.append(oh(tb, len(TB_EDGES) + 1)[:, 1:])
    if form >= 2:
        cols.append(oh(tb, len(TB_EDGES) + 1)[:, 1:] * half[:, None])
        cols.append(half[:, None].astype(float))
    if form >= 3:
        cols.append(oh(band, 5)[:, 1:])
    if form >= 4:
        T = oh(tb, len(TB_EDGES) + 1)[:, 1:4]
        B = oh(band, 5)[:, [0, 1]]
        cols.append((T[:, :, None] * B[:, None, :]).reshape(n, -1))
    if form >= 5:
        cols.append(oh(to, 4)[:, 1:])
        cols.append(oh(yb, len(YB_EDGES) + 1)[:, 1:])
    return np.hstack(cols)


def newton(X, n, k, lam, iters=40):
    b = np.zeros(X.shape[1])
    P = np.eye(X.shape[1]) * lam
    P[0, 0] = 0.0
    for _ in range(iters):
        z = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-z))
        w = n * p * (1 - p) + 1e-12
        gr = X.T @ (k - n * p) - P @ b
        H = X.T @ (X * w[:, None]) + P + 1e-9 * np.eye(len(b))
        step = np.linalg.solve(H, gr)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    return b


def cells(d, D, H):
    m = d["rp"] & np.isfinite(d["hsr"]) & (d["q"] <= 4) & ((d["hsr"] <= 1800) | True)
    tb = np.digitize(d["hsr"], TB_EDGES)
    yb = np.digitize(100 - d["yl"], YB_EDGES)
    X = pd.DataFrame(dict(season=d["season"][m], half=d["half"][m], tb=tb[m], band=d["band"][m], to=d["to"][m], yb=yb[m], k=D[m].astype(float)))
    X["n"] = 1.0
    A = X.groupby(["season", "half", "tb", "band", "to", "yb"], as_index=False).agg(n=("n", "sum"), k=("k", "sum"))
    return A


def fit():
    F = frame()
    d = derive(F)
    d["lat"] = F.lat.to_numpy()
    sp = json.load(open(OUTD / "def.json"))
    thr, H = sp["thr"], sp["H"]
    D = np.load(OUTD / "D.npy")
    rng = np.random.default_rng(81)
    A = cells(d, D, H)
    seasons = sorted(A.season.unique())
    say(f"## E79 fit: D rows {int(D.sum())} of {int(d['rp'].sum())} rp plays; cells {len(A)}; thr {thr:.0f} H {H}")
    lams = (1e-3, 1e-1, 10.0)
    res = {}
    for form in range(6):
        for lam in lams:
            ll = []
            for s in seasons:
                tr, te = A[A.season != s], A[A.season == s]
                Xt = design(tr.half.to_numpy(), tr.tb.to_numpy(), tr.band.to_numpy(), tr.to.to_numpy(), tr.yb.to_numpy(), form)
                b = newton(Xt, tr.n.to_numpy(), tr.k.to_numpy(), lam)
                Xe = design(te.half.to_numpy(), te.tb.to_numpy(), te.band.to_numpy(), te.to.to_numpy(), te.yb.to_numpy(), form)
                p = np.clip(1 / (1 + np.exp(-np.clip(Xe @ b, -30, 30))), 1e-12, 1 - 1e-12)
                ll.append(float((te.k * np.log(p) + (te.n - te.k) * np.log(1 - p)).sum()))
            res[(form, lam)] = np.array(ll)
    nrows = A.groupby("season").n.sum().to_numpy()
    ref = res[(0, lams[0])]
    best = max(res, key=lambda key: res[key].sum())
    for key, v in sorted(res.items()):
        dv_ = (v - ref) / nrows * 1e4
        say(f"  form {key[0]} lam {key[1]:g}: heldout LL {v.sum():.1f} gain vs constant {np.sum(v - ref):.1f} (per 1e4 rows, per season mean {dv_.mean():.3f} sd {dv_.std(ddof=1):.3f}); seasons better {int((v > ref).sum())}/{len(v)}")
    say(f"looks {len(res)} (form x lambda); chosen form {best[0]} lam {best[1]:g}")
    form, lam = best
    Xa = design(A.half.to_numpy(), A.tb.to_numpy(), A.band.to_numpy(), A.to.to_numpy(), A.yb.to_numpy(), form)
    coef = newton(Xa, A.n.to_numpy(), A.k.to_numpy(), lam)
    say("full-fit coefficients per fold stability (max abs diff across leave-one-out folds):")
    cf = []
    for s in seasons:
        tr = A[A.season != s]
        cf.append(newton(design(tr.half.to_numpy(), tr.tb.to_numpy(), tr.band.to_numpy(), tr.to.to_numpy(), tr.yb.to_numpy(), form), tr.n.to_numpy(), tr.k.to_numpy(), lam))
    cf = np.array(cf)
    say(f"  coef sd across folds mean {cf.std(0).mean():.3f} max {cf.std(0).max():.3f}")

    yd = d["yds"]
    lost, td = d["lost"], d["td"]
    q75 = np.percentile(yd[D & ~lost & ~td], [33.3, 66.7])
    cat = np.where(lost, 0, np.where(td, 1, 2 + np.digitize(yd, q75)))
    di = np.flatnonzero(D)
    gfreq = np.bincount(cat[di], minlength=5) / len(di)
    ms = (10, 20, 40, 80, 160, len(di))
    ss = (0.5, 1.0, 2.0, 4.0)
    pl = {}
    for m in ms:
        for s_ in ss:
            tot = 0.0
            for s in seasons:
                trn = di[d["season"][di] != s]
                te = di[d["season"][di] == s]
                gf = np.bincount(cat[trn], minlength=5) / len(trn)
                o = np.argsort(d["yl"][trn], kind="stable")
                ylt, ct = d["yl"][trn][o], cat[trn][o]
                for r in te:
                    mm = min(m, len(trn))
                    c = np.searchsorted(ylt, d["yl"][r])
                    lo = max(0, min(c - mm // 2, len(trn) - mm))
                    cnt = np.bincount(ct[lo : lo + mm], minlength=5)
                    tot += np.log((cnt[cat[r]] + s_ * gf[cat[r]]) / (mm + s_))
            pl[(m, s_)] = tot
    bm = max(pl, key=pl.get)
    say(f"pool size grid {ms} x prior grid {ss}: looks {len(pl)}; chosen m {bm[0]} prior {bm[1]}; LL {pl[bm]:.1f} (pooled-all m {len(di)} prior {bm[1]} LL {pl[(len(di), bm[1])]:.1f})")
    OUTD.mkdir(parents=True, exist_ok=True)
    json.dump(dict(form=form, lam=lam, coef=coef.tolist(), thr=thr, H=H, m=int(bm[0]), tb_edges=TB_EDGES, yb_edges=YB_EDGES), open(OUTD / "desp_fit.json", "w"))
    np.save(OUTD / "Didx.npy", di)

    say("### replay LOSO: real q4 gsr<=450 trailing states; lost and TD per rp play (real | sim base | sim with DESP)")
    S = sim_flags(D)
    pr = np.zeros(len(D))
    for s in seasons:
        tr, te = A[A.season != s], A[A.season == s]
        b = newton(design(tr.half.to_numpy(), tr.tb.to_numpy(), tr.band.to_numpy(), tr.to.to_numpy(), tr.yb.to_numpy(), form), tr.n.to_numpy(), tr.k.to_numpy(), lam)
        ms_ = np.flatnonzero(d["season"] == s)
        X = design(d["half"][ms_], np.digitize(d["hsr"][ms_], TB_EDGES), d["band"][ms_], d["to"][ms_], np.digitize(100 - d["yl"][ms_], YB_EDGES), form)
        pr[ms_] = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
    pool_lost = np.zeros(len(D))
    pool_td = np.zeros(len(D))
    for s in seasons:
        trn = di[d["season"][di] != s]
        o = np.argsort(d["yl"][trn], kind="stable")
        ylt = d["yl"][trn][o]
        lt, tt = lost[trn][o].astype(float), td[trn][o].astype(float)
        ms_ = np.flatnonzero(d["season"] == s)
        mm = min(int(bm[0]), len(trn))
        c = np.searchsorted(ylt, d["yl"][ms_])
        lo = np.clip(c - mm // 2, 0, len(trn) - mm)
        cl, ct_ = np.r_[0, np.cumsum(lt)], np.r_[0, np.cumsum(tt)]
        pool_lost[ms_] = (cl[lo + mm] - cl[lo]) / mm
        pool_td[ms_] = (ct_[lo + mm] - ct_[lo]) / mm
    W = gboot(d["g"], NB, rng)
    Ws = gboot(S["g"], NB, rng)
    cells_ = [("trail>8 q4 gsr<=450", 0, 450, 4), ("trail1-8 q4 gsr<=450", 1, 450, 4), ("trail>8 q4 gsr<=900", 0, 900, 4), ("trail>8 q4 450<gsr<=900", 0, 900, 4)]
    for nm, b, gl, qq in cells_:
        lo_g = 450 if "450<" in nm else -1
        mr = d["rp"] & (d["band"] == b) & (d["q"] == qq) & (d["gsr"] <= gl) & (d["gsr"] > lo_g)
        msim = S["rp"] & (S["band"] == b) & (S["q"] == qq) & (S["gsr"] <= gl) & (S["gsr"] > lo_g)
        pbar = pr[mr].mean()
        for key, nmx, real, simv, pool in (("lost", "lost", lost, S["lost"], pool_lost), ("td", "td", td, S["td"], pool_td)):
            rr = grate(d["g"], W, real & mr, mr)
            nonD = simv & msim & ~S["D"]
            den = msim & ~S["D"]
            sr = grate(S["g"], Ws, simv & msim, msim)
            snd = nonD.sum() / max(den.sum(), 1)
            mix = (1 - pbar) * snd + pbar * pool[mr].mean()
            say(f"  {nm:24s} {nmx:4s} real {fmt(rr)} | sim {fmt(sr)} | sim+DESP {mix:.4f} (pbar {pbar:.4f}, nonD sim {snd:.4f}, pool {pool[mr].mean():.4f}) n {int(mr.sum())}")
    say(f"looks in replay {2 * len(cells_)}")
    (OUTD / "fit.txt").write_text("\n".join(OUT), encoding="utf-8")


def install_desp():
    import mod25d_variance as dv

    spec = json.loads((OUTD / "desp_fit.json").read_text(encoding="utf-8"))
    di = np.load(OUTD / "Didx.npy")
    ns = dv._G["ns"]
    a = dv._G["tables"]["arrays"]
    yl = np.asarray(a["fp_raw"], float)
    o = np.argsort(yl[di], kind="stable")
    dsort = di[o]
    ysort = yl[dsort]
    coef = np.asarray(spec["coef"])
    form, m = int(spec["form"]), int(spec["m"])
    base = ns["pick_index_nn_conditioned"]
    cache = {}

    def pick(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim):
        loc = sys._getframe(1).f_locals
        qtr, gsr, offense = loc.get("qtr"), loc.get("gsr"), loc.get("offense")
        if qtr is None or gsr is None or qtr > 4:
            return base(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)
        hsr = gsr - 1800.0 if qtr <= 2 else gsr
        half = int(qtr >= 3)
        tb = int(np.digitize(hsr, TB_EDGES))
        band = int(pd.cut([score], EDGES, labels=False)[0])
        yb = int(np.digitize(100.0 - fp, YB_EDGES))
        ck = (half, tb, band, int(min(off_to, 3)), yb)
        p = cache.get(ck)
        if p is None:
            x = design(np.array([half]), np.array([tb]), np.array([band]), np.array([int(min(off_to, 3))]), np.array([yb]), form)
            p = float(1 / (1 + np.exp(-np.clip(x @ coef, -30, 30)))[0])
            cache[ck] = p
        if rng.random() < p:
            c = int(np.searchsorted(ysort, fp))
            lo = max(0, min(c - m // 2, len(dsort) - m))
            return int(dsort[lo + int(rng.integers(0, m))])
        return base(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)

    ns["pick_index_nn_conditioned"] = pick


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "chars"
    {"chars": chars, "fit": fit}[cmd]()


if __name__ == "__main__":
    main()
