import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
ART = REPO / "artifacts" / "mod25e3"
OUTD = ART / "risk"
LABEL = os.environ.get("RISK_LABEL", "crHpqokgndecsk2mfwt")
SEEDS = tuple(int(x) for x in os.environ.get("RISK_SEEDS", "11,12,13").split(","))
BURN = int(os.environ.get("KICK_BURN", 2))
EDGES = [-1e9, -8.5, -0.5, 0.5, 8.5, 1e9]
BANDS = ["trail>8", "trail1-8", "tied", "lead1-8", "lead>8"]
TQ = ["Q1", "Q2", "Q3", "Q4a", "Q4b"]
NB = 200
OUT = []


def enabled():
    return os.environ.get("RISK") == "1"


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def real_plays():
    f = OUTD / "real_P.parquet"
    if not f.exists():
        import mod25d_variance as dv

        P, _, meta, _ = dv.real_load(list(range(2009, 2018)))
        P["season"] = meta.season.to_numpy()[P.g.to_numpy().astype(int)]
        OUTD.mkdir(parents=True, exist_ok=True)
        P.to_parquet(f)
    P = pd.read_parquet(f)
    if "season" not in P.columns:
        f.unlink()
        return real_plays()
    P["gk"] = P.g.astype(np.int64)
    return P[P.qtr <= 4].reset_index(drop=True)


def sim_plays():
    out = []
    for sdn in SEEDS:
        sdir = ART / f"e5_{LABEL}_s{sdn}"
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < BURN:
                continue
            d = pd.read_parquet(f, columns=["g", "down", "dist", "yl", "sd", "gsr", "qtr", "code", "po", "pdf", "flip", "offhome", "idx"])
            d = d[d.qtr <= 4].copy()
            d["gk"] = ((sdn * 10000 + w) * 100 + s) * 100000 + d.g.astype(np.int64)
            out.append(d)
    return pd.concat(out, ignore_index=True)


def tcell(qtr, gsr):
    q = qtr.astype(int)
    return np.where(q <= 3, q - 1, np.where(gsr <= 450.0, 4, 3))


def flags(P):
    code = P.code.to_numpy()
    rp = (code == 0) | (code == 1)
    flip = P.flip.to_numpy().astype(bool)
    po, pdf = P.po.to_numpy(), P.pdf.to_numpy()
    lost = rp & ((pdf > 0) | (flip & (po == 0)))
    down = P.down.to_numpy()
    band = pd.cut(P.sd.to_numpy(), EDGES, labels=False).astype(int)
    tq = tcell(P.qtr.to_numpy(), P.gsr.to_numpy())
    return dict(pas=code == 1, run=code == 0, rp=rp, lost=lost, d4=down == 4, go4=rp & (down == 4), kick4=((code == 2) | (code == 3)) & (down == 4), band=band, tq=tq, cell=band * 5 + tq)


def gboot(gk, nb, rng):
    gi, gn = pd.factorize(gk)
    G = len(gn)
    W = np.vstack([np.ones((1, G)), rng.multinomial(G, np.ones(G) / G, size=nb).astype(float)])
    return gi, G, W


def rate(gi, G, W, num, den):
    n = np.bincount(gi, weights=num.astype(float), minlength=G)
    d = np.bincount(gi, weights=den.astype(float), minlength=G)
    return (W @ n) / np.maximum(W @ d, 1e-9)


def iv(d):
    return f"{d[0]:+.4f} [{np.percentile(d[1:], 5):+.4f},{np.percentile(d[1:], 95):+.4f}] pp {float((d[1:] > 0).mean()):.2f}"


def diag():
    rng = np.random.default_rng(78)
    R, S = real_plays(), sim_plays()
    fr, fs = flags(R), flags(S)
    gr = gboot(R.gk, NB, rng)
    gs = gboot(S.gk, NB, rng)
    say(f"## E78 diag: real {R.gk.nunique()} games {len(R)} plays | sim {S.gk.nunique()} games {len(S)} plays; labels {LABEL} seeds {SEEDS}; game bootstrap {NB}")
    looks = [0]

    def line(name, num, den, mask=None):
        out = []
        for g, f in ((gr, fr), (gs, fs)):
            m = np.ones(len(f["pas"]), bool) if mask is None else mask(f)
            out.append(rate(g[0], g[1], g[2], num(f) & m, den(f) & m))
        looks[0] += 1
        n = [int((den(f) & (np.ones(len(f["pas"]), bool) if mask is None else mask(f))).sum()) for f in (fr, fs)]
        say(f"  {name:28s} real {out[0][0]:.4f} sim {out[1][0]:.4f} sim-real {iv(out[1] - out[0])} n {n[0]}|{n[1]}")

    defs = [
        ("pass lost (int/fum)", lambda f: f["lost"] & f["pas"] & ~f["d4"], lambda f: f["pas"] & ~f["d4"]),
        ("run lost (fum)", lambda f: f["lost"] & f["run"] & ~f["d4"], lambda f: f["run"] & ~f["d4"]),
        ("4th go fail", lambda f: f["lost"] & f["d4"], lambda f: f["go4"]),
        ("4th go share", lambda f: f["go4"], lambda f: f["go4"] | f["kick4"]),
    ]
    say("### overall")
    for nm, n_, d_ in defs:
        line(nm, n_, d_)
    say("### by lead band (offence), all quarters")
    for b in range(5):
        for nm, n_, d_ in defs:
            line(f"{BANDS[b]} {nm}", n_, d_, lambda f, b=b: f["band"] == b)
    say("### by quarter cell, all bands")
    for t in range(5):
        for nm, n_, d_ in defs:
            line(f"{TQ[t]} {nm}", n_, d_, lambda f, t=t: f["tq"] == t)
    say("### band x quarter cell, pass lost and 4th go fail/share (cells with n>=200 both sides)")
    for c in range(25):
        for nm, n_, d_ in (defs[0], defs[2], defs[3]):
            if min((d_(f) & (f["cell"] == c)).sum() for f in (fr, fs)) >= 200:
                line(f"{BANDS[c // 5]} {TQ[c % 5]} {nm}", n_, d_, lambda f, c=c: f["cell"] == c)
    say("### per-game lost-possession plays (run/pass, all downs) by lead band: real | sim, sim-real")
    for b in range(5):
        r = []
        for g, f in ((gr, fr), (gs, fs)):
            n = np.bincount(g[0], weights=(f["lost"] & (f["band"] == b)).astype(float), minlength=g[1])
            r.append((g[2] @ n) / g[2].sum(1))
        looks[0] += 1
        say(f"  {BANDS[b]:9s} {r[0][0]:.4f} | {r[1][0]:.4f} {iv(r[1] - r[0])}")
    say(f"looks {looks[0]} interval prints")
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "diag.txt").write_text("\n".join(OUT), encoding="utf-8")


PROW = OUTD / "prow.npy"
PROWJ = OUTD / "prow.json"
KICKS = ART / "kick" / "kicks_2009.parquet"
KFIT = OUTD / "kick_fit.json"
TFIT = OUTD / "tov_fit.json"
KNN_GRID = (20, 40, 80, 160, 320, 640)
HK_GRID = (0.05, 0.1, 0.2, 0.4)
HT_GRID = (0.05, 0.1, 0.2, 0.4, 1.0)
PRIOR = 1.0
KCAT = 100


def kick_cat(K):
    return (K.nfp.to_numpy(float).astype(int) + KCAT * K.retained.to_numpy(int)).astype(int)


def kick_weights(kind, par, kdt, gst, sk, st, kdv, tv):
    zk = (kdt[None, :] - kdv[:, None]) / sk
    zt = (gst[None, :] - tv[:, None]) / st
    if kind == "gauss":
        return np.exp(-0.5 * (zk / par[0]) ** 2 - 0.5 * (zt / par[1]) ** 2)
    d = zk * zk + zt * zt
    m = int(par[0])
    thr = np.partition(d, m - 1, axis=1)[:, m - 1:m]
    return (d <= thr).astype(float)


def kick_predict(K, tr, te, kind, par, chunk=400):
    kdt, gst = K.kd.to_numpy(float)[tr], K.gsr.to_numpy(float)[tr]
    sk, st = kdt.std(), gst.std()
    cat = kick_cat(K)
    ct = cat[tr]
    pi = (np.bincount(ct, minlength=2 * KCAT) + 0.5) / (len(ct) + 0.5 * 2 * KCAT)
    nfpt = K.nfp.to_numpy(float)[tr]
    rt = K.retained.to_numpy(float)[tr]
    kde, gse, ce = K.kd.to_numpy(float)[te], K.gsr.to_numpy(float)[te], cat[te]
    ll = np.zeros(len(kde))
    mn = np.zeros(len(kde))
    mr = np.zeros(len(kde))
    for a in range(0, len(kde), chunk):
        sl = slice(a, a + chunk)
        W = kick_weights(kind, par, kdt, gst, sk, st, kde[sl], gse[sl])
        num = (W * (ct[None, :] == ce[sl][:, None])).sum(1)
        sw = W.sum(1)
        ll[sl] = np.log((num + PRIOR * pi[ce[sl]]) / (sw + PRIOR))
        mn[sl] = (W @ nfpt + PRIOR * (pi * np.arange(2 * KCAT) % KCAT).sum()) / (sw + PRIOR)
        mr[sl] = (W @ rt) / np.maximum(sw, 1e-12)
    return ll, mn, mr


def kfit():
    K = pd.read_parquet(KICKS)
    K = K[K.qtr <= 4].reset_index(drop=True)
    seasons = np.unique(K.season)
    models = [("knn", (int(np.sqrt(len(K) * (len(seasons) - 1) / len(seasons))),), "current sqrt(N) nearest")]
    models += [("knn", (m,), f"knn m={m}") for m in KNN_GRID]
    models += [("gauss", (hk, ht), f"gauss hk={hk} ht={ht}") for hk in HK_GRID for ht in HT_GRID]
    sub = {"all": np.ones(len(K), bool), "trail_late450": (K.kd.to_numpy() < 0) & (K.gsr.to_numpy() <= 450), "trail_late900": (K.kd.to_numpy() < 0) & (K.gsr.to_numpy() <= 900)}
    res = {}
    preds = {}
    for kind, par, nm in models:
        ll = np.zeros(len(K))
        mn = np.zeros(len(K))
        mr = np.zeros(len(K))
        for s in seasons:
            te = np.flatnonzero(K.season.to_numpy() == s)
            tr = np.flatnonzero(K.season.to_numpy() != s)
            a, b, c = kick_predict(K, tr, te, kind, par)
            ll[te], mn[te], mr[te] = a, b, c
        res[nm] = ll
        preds[nm] = (mn, mr)
    base = res[models[0][2]]
    say(f"## E78 kick fit: {len(K)} kickoffs 2009-17, LOSO by season ({len(seasons)} folds), Dirichlet prior strength {PRIOR:g} on pooled pmf; {len(models)} models = {len(models)} looks")
    say("  held-out mean log lik per kick; gain vs current (sqrt(N) nearest in standardised kd x gsr); fold wins of gain>0 out of 9; subset gains trail_late450 (n) | trail_late900 (n)")
    best = max(res, key=lambda k: res[k].sum())
    for kind, par, nm in sorted(models, key=lambda m: -res[m[2]].sum())[:8] + [models[0]]:
        g = res[nm] - base
        fw = sum(float(g[K.season.to_numpy() == s].sum()) > 0 for s in seasons)
        sg = " ".join(f"{g[sub[k]].mean():+.4f}" for k in ("trail_late450", "trail_late900"))
        say(f"  {nm:26s} LL {res[nm].mean():.4f} gain {g.mean():+.5f} folds {fw}/9 sub {sg}")
    say(f"  subset n: {int(sub['trail_late450'].sum())} | {int(sub['trail_late900'].sum())}; best {best}")
    kind, par, _ = next(m for m in models if m[2] == best)
    KFIT.write_text(__import__("json").dumps({"kind": kind, "par": list(par), "name": best, "gain": float((res[best] - base).mean()), "n": len(K)}), encoding="utf-8")
    say("### replay of real kickoff states: kicker post-score lead band x time; mean next-start yardline_100 (holder's) and retained share, real | current | fitted (held-out)")
    bd = pd.cut(K.kd.to_numpy(), [-1e9, -8.5, -0.5, 0.5, 8.5, 1e9], labels=False).astype(int)
    tq = np.where(K.gsr.to_numpy() <= 450, 0, np.where(K.gsr.to_numpy() <= 900, 1, 2))
    mc, mf = preds[models[0][2]], preds[best]
    for b in range(5):
        for t in range(3):
            m = (bd == b) & (tq == t)
            if m.sum() < 30:
                continue
            say(f"  {BANDS[b].replace('trail', 'kicker-trail').replace('lead', 'kicker-lead'):18s} {['<=450s', '<=900s', 'rest'][t]:6s} n {int(m.sum()):5d} nfp {K.nfp[m].mean():.2f} | {mc[0][m].mean():.2f} | {mf[0][m].mean():.2f} ; retained {K.retained[m].mean():.4f} | {mc[1][m].mean():.4f} | {mf[1][m].mean():.4f}")
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "kfit.txt").write_text(chr(10).join(OUT), encoding="utf-8")


TEDGES = (120.0, 300.0, 600.0, 900.0)
TMODELS = ("m0 const", "m1 band x quarter cells", "m2 time-in-half bins", "m3 time bins + lead", "m4 time bins + lead x time bins", "m5 lead band x time bins")
LEAD_CLIP = 24.0
LEAD_UNIT = 7.0
RIDGE = 1e-2


def state_vars(qtr, gsr, sd):
    qtr = np.asarray(qtr, float)
    gsr = np.asarray(gsr, float)
    sd = np.asarray(sd, float)
    half = (qtr >= 3).astype(int)
    tih = np.where(qtr <= 2, gsr - 1800.0, gsr)
    T = half * 5 + np.digitize(tih, TEDGES)
    lead = np.clip(sd, -LEAD_CLIP, LEAD_CLIP) / LEAD_UNIT
    band = pd.cut(sd, EDGES, labels=False).astype(int)
    cell = band * 5 + tcell(qtr, gsr)
    return lead, T, cell


def tov_design(mi, lead, T, cell):
    n = len(lead)
    oh = lambda v, k: (np.asarray(v)[:, None] == np.arange(k)[None, :]).astype(float)
    if mi == 0:
        return np.ones((n, 1))
    if mi == 1:
        return np.c_[np.ones(n), oh(cell, 25)[:, 1:]]
    if mi == 2:
        return oh(T, 10)
    if mi == 3:
        return np.c_[oh(T, 10), lead]
    if mi == 4:
        return np.c_[oh(T, 10), oh(T, 10) * lead[:, None]]
    return oh((np.asarray(cell) // 5) * 10 + T, 50)


def logit_fit(X, y):
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(X @ b)))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + RIDGE * np.eye(X.shape[1])
        st = np.linalg.solve(H, g)
        b = b + st
        if np.abs(st).max() < 1e-8:
            break
    return b


def sigm(z):
    return 1.0 / (1.0 + np.exp(-z))


def tov_rows():
    R = real_plays()
    f = flags(R)
    m = f["rp"] & (R.down.to_numpy() < 4)
    R = R[m].reset_index(drop=True)
    lead, T, cell = state_vars(R.qtr.to_numpy(), R.gsr.to_numpy(), R.sd.to_numpy())
    return R, lead, T, cell, (R.code.to_numpy() == 1).astype(int), flags(R)["lost"].astype(float)


def tfit():
    import json

    R, lead, T, cell, typ, y = tov_rows()
    seasons = np.unique(R.season)
    S = R.season.to_numpy()
    say(f"## E78 turnover fit: {len(R)} real run/pass plays down 1-3 2009-17, LOSO by season ({len(seasons)} folds); 6 models x 2 play types = 10 fits per fold")
    ll = {}
    coefs = {}
    for mi in range(len(TMODELS)):
        tot = np.zeros(len(R))
        for tp in (0, 1):
            X = tov_design(mi, lead, T, cell)
            for s in seasons:
                tr = (typ == tp) & (S != s)
                te = (typ == tp) & (S == s)
                b = logit_fit(X[tr], y[tr])
                coefs[(mi, tp, int(s))] = b
                p = np.clip(sigm(X[te] @ b), 1e-9, 1 - 1e-9)
                tot[te] = y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)
        ll[mi] = tot
    base = ll[0]
    best = max(ll, key=lambda k: ll[k].sum())
    for mi in range(len(TMODELS)):
        g = ll[mi] - base
        fw = sum(float(g[S == s].sum()) > 0 for s in seasons)
        late = (T % 5) <= 2
        say(f"  {TMODELS[mi]:36s} LL/play {ll[mi].mean():.5f} gain vs const {g.mean() * 1e4:+.2f}e-4 total {g.sum():+.1f} folds {fw}/9 ; last-600s-of-half gain {g[late].mean() * 1e4:+.2f}e-4 (n {int(late.sum())})")
    say(f"  best by LOSO LL: {TMODELS[best]}; fits {len(TMODELS) * 2 * len(seasons)}")
    full = {}
    for tp in (0, 1):
        X = tov_design(best, lead, T, cell)
        full[tp] = logit_fit(X[typ == tp], y[typ == tp])
    spec = {"model": best, "name": TMODELS[best], "coef": {"run": full[0].tolist(), "pass": full[1].tolist()}, "folds": {f"{mi}|{tp}|{s}": coefs[(mi, tp, s)].tolist() for (mi, tp, s) in coefs if mi == best}}
    TFIT.write_text(json.dumps(spec), encoding="utf-8")
    for tp, tn in ((0, "run"), (1, "pass")):
        lo = np.min([coefs[(best, tp, int(s))] for s in seasons], axis=0)
        hi = np.max([coefs[(best, tp, int(s))] for s in seasons], axis=0)
        say(f"  {tn} coefficient range across folds (min..max): " + " ".join(f"{a:+.2f}..{b:+.2f}" for a, b in zip(lo, hi)))
    return R, lead, T, cell, typ, y, coefs, best, seasons


def tov_prob(spec, tp, qtr, gsr, sd):
    lead, T, cell = state_vars(np.atleast_1d(qtr), np.atleast_1d(gsr), np.atleast_1d(sd))
    X = tov_design(spec["model"], lead, T, cell)
    return sigm(X @ np.asarray(spec["coef"]["pass" if tp else "run"]))


REPLAY_EVERY = 6
K_NN = 200


def replay(R, coefs, best, seasons):
    sys.argv = sys.argv[:1] + ["200"]
    import mod25e_srcconc as sc

    tables, trans, pbp = sc.engine()
    a = tables["arrays"]
    meta = pbp.groupby("game_id", sort=False).agg(season=("season", "first"))
    season = meta.season.reindex(trans["game_id"].to_numpy()).to_numpy().astype(int)
    code = np.asarray(a["play_type_code"])
    flip = np.asarray(a["possession_flip"]).astype(bool)
    po, pdf = np.asarray(a["points_off"], float), np.asarray(a["points_def"], float)
    down = np.asarray(a["down_i"])
    qtr, gsr, sd = trans["qtr_actual"].to_numpy(float), trans["gsr_actual"].to_numpy(float), trans["sc_raw"].to_numpy(float)
    rp = (code == 0) | (code == 1)
    lost_row = rp & ((pdf > 0) | (flip & (po == 0)))
    wfile = ART / "sel" / "w.npy"
    wsel = np.load(wfile) if wfile.exists() else np.ones(len(code))
    samp = np.flatnonzero(rp & (down < 4) & (qtr <= 4) & (np.arange(len(code)) % REPLAY_EVERY == 0) & np.isfinite(sd))
    pi = [lost_row[rp & (down < 4) & (code == t)].mean() for t in (0, 1)]
    import json

    sp = json.loads(TFIT.read_text(encoding="utf-8"))
    fin = np.isfinite(sd)
    lead_a, T_a, cell_a = state_vars(np.nan_to_num(qtr, nan=1.0), np.nan_to_num(gsr, nan=3600.0), np.where(fin, sd, 0.0))
    X_a = tov_design(best, lead_a, T_a, cell_a)
    prow = np.c_[sigm(X_a @ np.asarray(sp["coef"]["run"])), sigm(X_a @ np.asarray(sp["coef"]["pass"]))]
    np.save(PROW, prow)
    PROWJ.write_text(json.dumps({"dist_checksum": float(np.asarray(a["dist_raw"], float).sum()), "n": int(len(code))}), encoding="utf-8")
    Pfold = {}
    for sn in seasons:
        Pfold[int(sn)] = np.c_[sigm(X_a @ coefs[(best, 0, int(sn))]), sigm(X_a @ coefs[(best, 1, int(sn))])]
    st_season = season
    rows = []
    for (d, ph), (tree, sub) in tables["nn_trees_cond"].items():
        if d == 4:
            continue
        r_in = np.intersect1d(sub, samp)
        if len(r_in) == 0:
            continue
        pos = pd.Series(np.arange(len(sub)), index=sub)[r_in].to_numpy()
        _, ind = tree.query(tree.data[pos], k=min(K_NN, len(sub)))
        nb = sub[ind]
        w, ty, ls = wsel[nb], code[nb], lost_row[nb]
        out = {}
        for t in (0, 1):
            mt = (ty == t) * w
            wt = mt.sum(1)
            ne = wt ** 2 / np.maximum((mt * w).sum(1), 1e-12)
            pn = (mt * ls).sum(1) / np.maximum(wt, 1e-12)
            out[t] = (wt, (ne * pn + PRIOR * pi[t]) / (ne + PRIOR), pn)
        sr = st_season[r_in]
        hy = {}
        for t in (0, 1):
            Pn = np.zeros(nb.shape)
            Ps = np.zeros(len(r_in))
            for sn in np.unique(sr):
                m = sr == sn
                Pn[m] = Pfold[int(sn)][nb[m], t]
                Ps[m] = Pfold[int(sn)][r_in[m], t]
            mult = np.where(ls, Ps[:, None] / Pn, (1.0 - Ps[:, None]) / (1.0 - Pn))
            wm = (ty == t) * w * mult
            wth = wm.sum(1)
            ne = wth ** 2 / np.maximum((wm * wm).sum(1), 1e-12)
            pnh = (wm * ls).sum(1) / np.maximum(wth, 1e-12)
            hy[t] = (wth, (ne * pnh + PRIOR * pi[t]) / (ne + PRIOR), pnh)
        rows.append(pd.DataFrame({"r": r_in, "wt0": out[0][0], "wt1": out[1][0], "pc0": out[0][1], "pc1": out[1][1], "pn0": out[0][2], "pn1": out[1][2], "wh0": hy[0][0], "wh1": hy[1][0], "ph0": hy[0][1], "ph1": hy[1][1], "pnh0": hy[0][2], "pnh1": hy[1][2]}))
    Q = pd.concat(rows, ignore_index=True)
    r = Q.r.to_numpy()
    Q["season"], Q["qtr"], Q["gsr"], Q["sd"], Q["tt"], Q["y"] = season[r], qtr[r], gsr[r], sd[r], code[r], lost_row[r].astype(float)
    lead, T, cell = state_vars(Q.qtr.to_numpy(), Q.gsr.to_numpy(), Q.sd.to_numpy())
    X = tov_design(best, lead, T, cell)
    for t in (0, 1):
        pf = np.zeros(len(Q))
        for s in seasons:
            m = Q.season.to_numpy() == s
            pf[m] = sigm(X[m] @ coefs[(best, t, int(s))])
        Q[f"pf{t}"] = pf
    tt = Q.tt.to_numpy()
    y = Q.y.to_numpy()
    pcur = np.where(tt == 1, Q.pc1, Q.pc0)
    pfit = np.where(tt == 1, Q.pf1, Q.pf0)
    phyb = np.where(tt == 1, Q.ph1, Q.ph0)
    llc = y * np.log(np.clip(pcur, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - pcur, 1e-9, 1))
    llf = y * np.log(np.clip(pfit, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - pfit, 1e-9, 1))
    llh = y * np.log(np.clip(phyb, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - phyb, 1e-9, 1))
    S = Q.season.to_numpy()
    g = llf - llc
    gh = llh - llc
    late = (T % 5) <= 2
    say(f"### replay of real states ({len(Q)} sampled source rows, every {REPLAY_EVERY}th, down 1-3 run/pass; K={K_NN} neighbours weighted by the SEL row weight, team-rating kernel not applied)")
    say(f"  held-out LL per play of the row's own outcome: current neighbour rate {llc.mean():.5f} fitted {llf.mean():.5f} gain {g.mean() * 1e4:+.2f}e-4 total {g.sum():+.1f}; fold wins {sum(float(g[S == s].sum()) > 0 for s in seasons)}/9; last-600s-of-half gain {g[late].mean() * 1e4:+.2f}e-4 (n {int(late.sum())})")
    say(f"  state-ratio tilt (neighbour row reweighted from its own state's fitted rate to the target state's): LL {llh.mean():.5f} gain vs current {gh.mean() * 1e4:+.2f}e-4 total {gh.sum():+.1f}; fold wins {sum(float(gh[S == s].sum()) > 0 for s in seasons)}/9; last-600s-of-half gain {gh[late].mean() * 1e4:+.2f}e-4")
    wh0, wh1 = Q.wh0.to_numpy(), Q.wh1.to_numpy()
    pnh_all = (wh0 * Q.pnh0.to_numpy() + wh1 * Q.pnh1.to_numpy()) / np.maximum(wh0 + wh1, 1e-12)
    wt0, wt1 = Q.wt0.to_numpy(), Q.wt1.to_numpy()
    den = np.maximum(wt0 + wt1, 1e-12)
    pn_all = (wt0 * np.nan_to_num(Q.pn0.to_numpy()) + wt1 * np.nan_to_num(Q.pn1.to_numpy())) / den
    pf_all = (wt0 * Q.pf0.to_numpy() + wt1 * Q.pf1.to_numpy()) / den
    band = pd.cut(Q.sd.to_numpy(), EDGES, labels=False).astype(int)
    tq = tcell(Q.qtr.to_numpy(), Q.gsr.to_numpy())
    say("  mean lost-possession rate of run/pass plays: real (row outcome) | current neighbour draw | after state-only tilt | after state-ratio tilt")
    for tqi in (2, 3, 4):
        for b in range(5):
            m = (band == b) & (tq == tqi)
            if m.sum() < 100:
                continue
            say(f"    {BANDS[b]:9s} {TQ[tqi]}: {y[m].mean():.4f} | {pn_all[m].mean():.4f} | {pf_all[m].mean():.4f} | {pnh_all[m].mean():.4f} n {int(m.sum())}")
    for tqi in (0, 1, 4):
        m = tq == tqi
        say(f"    all bands {TQ[tqi]}: {y[m].mean():.4f} | {pn_all[m].mean():.4f} | {pf_all[m].mean():.4f} | {pnh_all[m].mean():.4f} n {int(m.sum())}")
    m = (tq == 4) & (tt == 0)
    say(f"    Q4b run only: {y[m].mean():.4f} | {Q.pn0.to_numpy()[m].mean():.4f} | {Q.pf0.to_numpy()[m].mean():.4f} | {Q.pnh0.to_numpy()[m].mean():.4f}")
    say(f"    all: {y.mean():.4f} | {pn_all.mean():.4f} | {pf_all.mean():.4f} | {pnh_all.mean():.4f}")
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "tfit.txt").write_text(chr(10).join(OUT), encoding="utf-8")


def install_tov():
    import json

    import mod25d_variance as dv

    spec = json.loads(TFIT.read_text(encoding="utf-8"))
    ns = dv._G["ns"]
    tables = dv._G["tables"]
    a = tables["arrays"]
    code = np.asarray(a["play_type_code"])
    rp = (code == 0) | (code == 1)
    lost_row = rp & ((np.asarray(a["points_def"], float) > 0) | (np.asarray(a["possession_flip"]).astype(bool) & (np.asarray(a["points_off"], float) == 0)))
    prow = np.load(PROW)
    pj = json.loads(PROWJ.read_text(encoding="utf-8"))
    assert len(prow) == len(code) == pj["n"]
    assert abs(float(np.asarray(a["dist_raw"], float).sum()) - pj["dist_checksum"]) < 1e-6
    base = ns["pick_index_nn_conditioned"]
    cache = {}

    def pick(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim):
        loc = sys._getframe(1).f_locals
        qtr, gsr = loc.get("qtr"), loc.get("gsr")
        idx = base(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)
        down_key = down if down in (1, 2, 3, 4) else 4
        if down_key == 4 or qtr is None or gsr is None or qtr > 4:
            return idx
        key = ns["round_state_key"](down_key, phase, dist, fp, score, time_raw, off_to, def_to)
        nb = tbl["nn_cache_cond"].get(key)
        cw = tbl["nn_weight_cache_cond"].get((key, off_sim, def_sim, is_home_sim))
        if nb is None or cw is None or idx not in nb:
            return idx
        lead, T, cell = state_vars([qtr], [gsr], [score])
        ck = (key, off_sim, def_sim, is_home_sim, int(T[0]), int(cell[0]))
        c = cache.get(ck)
        if c is None:
            w = np.diff(cw[0], prepend=0.0)
            mult = np.ones(len(w))
            ty, ls = code[nb], lost_row[nb]
            for t in (0, 1):
                mt = ty == t
                wt = w[mt].sum()
                if wt <= 0:
                    continue
                ps = float(tov_prob(spec, t, qtr, gsr, score)[0])
                pr = prow[nb, t]
                mult[mt] = np.where(ls[mt], ps / pr[mt], (1.0 - ps) / (1.0 - pr[mt]))
            cdf = np.cumsum(w * mult)
            c = (cdf, float(cdf[-1]))
            cache[ck] = c
        j = min(int(np.searchsorted(c[0], rng.random() * c[1], side="right")), len(nb) - 1)
        return int(nb[j])

    ns["pick_index_nn_conditioned"] = pick


def install_kick():
    import json

    import mod25d_variance as dv
    import mod25e_kick as mk

    cfg = dv._G["cfg"]
    mode = int(cfg.get("kick", 1))
    seed = int(cfg.get("seed", 3))
    fit = json.loads(KFIT.read_text(encoding="utf-8"))
    K = pd.read_parquet(mk.OUT / f"kicks_{mk.POOL[0]}.parquet")
    kd = K.kd.to_numpy(float)
    gs = K.gsr.to_numpy(float)
    sk, st_ = float(kd.std()), float(gs.std())
    ret = K.retained.to_numpy(int)
    nfp = K.nfp.to_numpy(float)
    rtd = K.rtd.to_numpy(int)
    pe = K.pat_extra.dropna().to_numpy(int)
    ok_all = np.ones(len(K), dtype=bool)
    ok_nt = rtd == 0
    state = {"k": None, "rng": None}

    def draw(rng, kdv, tv, mask):
        idx = np.flatnonzero(mask)
        W = kick_weights(fit["kind"], fit["par"], kd[idx], gs[idx], sk, st_, np.array([kdv]), np.array([tv]))[0]
        cdf = np.cumsum(W)
        j = min(int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right")), len(idx) - 1)
        return int(idx[j])

    if fit["kind"] == "class":
        draw = make_class_draw(fit, K, ok_nt)
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        po = float(drawn["points_off"])
        pdf = float(drawn["points_def"])
        if mode == 0 or qtr > 4 or not ((po >= 3 and pdf == 0) or (pdf >= 6 and po == 0)):
            return base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        if state["k"] != dv._G.get("task_key"):
            state["k"] = dv._G.get("task_key")
            state["rng"] = dv.task_rng(15485863, seed)
        rng = state["rng"]
        k_off = po > 0
        kdv = score_diff + po if k_off else pdf - score_diff
        t = max(float(clock_val) - float(drawn["clock_elapsed"]), 0.0)
        i = draw(rng, kdv, t, ok_all)
        new = dict(drawn)
        holder_k = bool(ret[i])
        fp = nfp[i]
        if rtd[i]:
            b = int(pe[int(rng.integers(0, len(pe)))]) if len(pe) else 1
            if k_off:
                new["points_def"] = 6.0 + b
            else:
                new["points_off"] = 6.0 + b
            j = draw(rng, 6.0 + b - kdv, t, ok_nt)
            holder_k = not bool(ret[j])
            fp = nfp[j]
        holder_off = holder_k == k_off
        new["flip"] = not holder_off
        new["next_down"] = 1.0
        new["next_distance"] = min(10.0, float(fp))
        new["next_yardline"] = float(fp)
        return base(down, distance, yardline, score_diff, qtr, clock_val, new)

    dv._G["pol"] = pol


CLASSES = ("return TD", "onside kept", "onside lost", "short kick", "regular kick")


def class_labels(K):
    rtd = K.rtd.to_numpy() == 1
    att = K.att.to_numpy() == 1
    ret = K.retained.to_numpy() == 1
    sh = K.short.to_numpy() == 1
    return np.where(rtd, 0, np.where(att & ret, 1, np.where(att, 2, np.where(sh, 3, 4)))).astype(int)


def kick_state(K):
    g = K.gsr.to_numpy(float)
    q = np.clip(5 - np.ceil(g / 900.0), 1, 4)
    return state_vars(q, g, K.kd.to_numpy(float))


def comp_masks(cls):
    return {"onside attempt": (cls != 0, (cls == 1) | (cls == 2)), "onside kept": ((cls == 1) | (cls == 2), cls == 1), "short kick": ((cls == 3) | (cls == 4), cls == 3)}


NFP_OPTS = (None, (0.4, 1.0), (0.2, 0.4))


def class_ll(K, tr, te, forms, nk=None):
    cls = class_labels(K)
    lead, T, cell = kick_state(K)
    nfp = K.nfp.to_numpy(float).astype(int)
    kd, gs = K.kd.to_numpy(float), K.gsr.to_numpy(float)
    ll = np.zeros(len(K))
    pm = np.zeros((len(K), 5))
    p0 = (cls[tr] == 0).mean()
    ll[te] += np.where(cls[te] == 0, np.log(p0), np.log(1 - p0))
    pm[te, 0] = p0
    P = {}
    for nm, (dom, yv) in comp_masks(cls).items():
        X = tov_design(forms[nm], lead, T, cell)
        b = logit_fit(X[tr & dom], yv[tr & dom].astype(float))
        p = np.clip(sigm(X @ b), 1e-9, 1 - 1e-9)
        m = te & dom
        ll[m] += np.where(yv[m], np.log(p[m]), np.log(1 - p[m]))
        P[nm] = p
    pa, pb, ps = P["onside attempt"], P["onside kept"], P["short kick"]
    q = 1 - p0
    pm[te, 1] = (q * pa * pb)[te]
    pm[te, 2] = (q * pa * (1 - pb))[te]
    pm[te, 3] = (q * (1 - pa) * ps)[te]
    pm[te, 4] = (q * (1 - pa) * (1 - ps))[te]
    tei = np.flatnonzero(te)
    mean_rows = np.zeros((len(K), 5))
    sk, st = kd[tr].std(), gs[tr].std()
    for c in range(0, 5):
        trc = np.flatnonzero(tr & (cls == c))
        if len(trc) == 0:
            continue
        pc = (np.bincount(nfp[trc], minlength=100) + 0.5) / (len(trc) + 50.0)
        mc = nfp[trc].astype(float)
        if nk is None or c == 0:
            mean_rows[tei, c] = mc.mean()
            if c > 0:
                m = te & (cls == c)
                ll[m] += np.log(pc[nfp[m]])
            continue
        for a in range(0, len(tei), 500):
            ti = tei[a:a + 500]
            W = kick_weights("gauss", nk, kd[trc], gs[trc], sk, st, kd[ti], gs[ti])
            sw = W.sum(1)
            mean_rows[ti, c] = (W @ mc) / np.maximum(sw, 1e-12)
            num = (W * (nfp[trc][None, :] == nfp[ti][:, None])).sum(1)
            lc = np.log((num + PRIOR * pc[nfp[ti]]) / (sw + PRIOR))
            ll[ti] += np.where(cls[ti] == c, lc, 0.0)
    return ll, (pm * mean_rows).sum(1), pm[:, 1], pm


def kclass():
    import json

    K = pd.read_parquet(KICKS)
    K = K[K.qtr <= 4].reset_index(drop=True)
    S = K.season.to_numpy()
    seasons = np.unique(S)
    cls = class_labels(K)
    say(f"## E78 kick class fit: {len(K)} kickoffs; classes " + ", ".join(f"{CLASSES[c]} {int((cls == c).sum())}" for c in range(5)))
    lead, T, cell = kick_state(K)
    comp = comp_masks(cls)
    best = {}
    ntot = 0
    for nm, (dom, yv) in comp.items():
        tot = {}
        for mi in range(len(TMODELS)):
            X = tov_design(mi, lead, T, cell)
            t = 0.0
            for s in seasons:
                tr, te = dom & (S != s), dom & (S == s)
                b = logit_fit(X[tr], yv[tr].astype(float))
                p = np.clip(sigm(X[te] @ b), 1e-9, 1 - 1e-9)
                t += float((np.where(yv[te], np.log(p), np.log(1 - p))).sum())
                ntot += 1
            tot[mi] = t
        best[nm] = max(tot, key=tot.get)
        say(f"  {nm:15s} (n {int(dom.sum())}, events {int((dom & yv).sum())}) LOSO LL total: " + " ".join(f"{TMODELS[mi][:2]} {tot[mi]:.1f}" for mi in range(len(TMODELS))) + f" -> {TMODELS[best[nm]]}")
    llc = np.zeros(len(K))
    llk = np.zeros(len(K))
    llm = np.zeros(len(K))
    mnc, mnk, mnm = np.zeros(len(K)), np.zeros(len(K)), np.zeros(len(K))
    lls = [np.zeros(len(K)) for _ in NFP_OPTS]
    mns = [np.zeros(len(K)) for _ in NFP_OPTS]
    rts = [np.zeros(len(K)) for _ in NFP_OPTS]
    atts = np.zeros(len(K))
    rtc, rtk, rtm = np.zeros(len(K)), np.zeros(len(K)), np.zeros(len(K))
    kj = json.loads(KFIT.read_text(encoding="utf-8")) if KFIT.exists() else None
    kk = (kj["kind"], tuple(kj["par"])) if kj is not None and kj.get("kind") in ("gauss", "knn") else ("gauss", (0.4, 1.0))
    for s in seasons:
        te = np.flatnonzero(S == s)
        trn = np.flatnonzero(S != s)
        a, b, c = kick_predict(K, trn, te, "knn", (int(np.sqrt(len(trn))),))
        llc[te], mnc[te], rtc[te] = a, b, c
        a, b, c = kick_predict(K, trn, te, kk[0], kk[1])
        llk[te], mnk[te], rtk[te] = a, b, c
        trm, tem = S != s, S == s
        for oi, nk in enumerate(NFP_OPTS):
            a, b, c, pmx = class_ll(K, trm, tem, best, nk)
            lls[oi][te], mns[oi][te], rts[oi][te] = a[te], b[te], c[te]
            if oi == 0:
                atts[te] = (pmx[:, 1] + pmx[:, 2])[te]
    sub = {"all": np.ones(len(K), bool), "trail_late450": (K.kd.to_numpy() < 0) & (K.gsr.to_numpy() <= 450), "trail_late900": (K.kd.to_numpy() < 0) & (K.gsr.to_numpy() <= 900)}
    for oi, nk in enumerate(NFP_OPTS):
        say(f"  start-yardline given class: {'pooled' if nk is None else 'gauss ' + str(nk)} LL/kick {lls[oi].mean():.4f} trail_late450 {lls[oi][sub['trail_late450']].mean():.4f}")
    bo = int(np.argmax([x.sum() for x in lls]))
    llm, mnm, rtm = lls[bo], mns[bo], rts[bo]
    say(f"  chosen start-yardline form: {'pooled' if NFP_OPTS[bo] is None else 'gauss ' + str(NFP_OPTS[bo])}")
    say(f"  LOSO held-out LL per kick: current sqrt(N) nearest {llc.mean():.4f}; best wide kernel {kk[0]} {kk[1]} {llk.mean():.4f}; class mixture {llm.mean():.4f}; component-form LOSO fits {ntot}")
    for nm, l in (("class mixture", llm), ("wide kernel", llk)):
        g = l - llc
        fw = sum(float(g[S == s].sum()) > 0 for s in seasons)
        say(f"  {nm:14s} gain vs current {g.mean():+.4f} folds {fw}/9 ; trail_late450 {g[sub['trail_late450']].mean():+.4f} trail_late900 {g[sub['trail_late900']].mean():+.4f}")
    g = llm - llk
    say(f"  class mixture vs wide kernel: {g.mean():+.4f} folds {sum(float(g[S == s].sum()) > 0 for s in seasons)}/9; trail_late450 {g[sub['trail_late450']].mean():+.4f}")
    say("### replay of real kickoff states (held-out): mean next-start yardline_100 and onside-kept share, real | current | wide kernel | class mixture")
    bd = pd.cut(K.kd.to_numpy(), EDGES, labels=False).astype(int)
    tq = np.where(K.gsr.to_numpy() <= 450, 0, np.where(K.gsr.to_numpy() <= 900, 1, 2))
    for b in range(5):
        for t in range(3):
            m = (bd == b) & (tq == t)
            if m.sum() < 30:
                continue
            say(f"  kicker {BANDS[b]:9s} {['<=450s', '<=900s', 'rest'][t]:6s} n {int(m.sum()):5d} att {K.att[m].mean():.3f}|{atts[m].mean():.3f} nfp {K.nfp[m].mean():.2f} | {mnc[m].mean():.2f} | {mnk[m].mean():.2f} | {mnm[m].mean():.2f} ; kept {K.retained[m].mean():.4f} | {rtc[m].mean():.4f} | {rtk[m].mean():.4f} | {rtm[m].mean():.4f}")
    full = {"kind": "class", "forms": best, "p_rtd": float((cls == 0).mean()), "coef": {}}
    for nm, (dom, yv) in comp.items():
        X = tov_design(best[nm], lead, T, cell)
        full["coef"][nm] = logit_fit(X[dom], yv[dom].astype(float)).tolist()
    full["name"] = "class mixture"
    full["nfp_kernel"] = None if NFP_OPTS[bo] is None else list(NFP_OPTS[bo])
    full["gain"] = float((llm - llc).mean())
    full["n"] = len(K)
    full["fold_wins_vs_current"] = int(sum(float((llm - llc)[S == s].sum()) > 0 for s in seasons))
    KFIT.write_text(json.dumps(full), encoding="utf-8")
    (OUTD / "kclass.txt").write_text(chr(10).join(OUT), encoding="utf-8")


def make_class_draw(fit, K, ok_nt):
    cls = class_labels(K)
    by = [np.flatnonzero(cls == c) for c in range(5)]
    kd, gs = K.kd.to_numpy(float), K.gsr.to_numpy(float)
    sk, st = kd.std(), gs.std()
    nk = fit.get("nfp_kernel")
    forms = fit["forms"]
    coef = {k: np.asarray(v) for k, v in fit["coef"].items()}

    def draw(rng, kdv, tv, mask):
        q = float(np.clip(5 - np.ceil(tv / 900.0), 1, 4))
        lead, T, cell = state_vars([q], [tv], [kdv])
        pa, pb, ps = (float(sigm(tov_design(forms[nm], lead, T, cell) @ coef[nm])[0]) for nm in ("onside attempt", "onside kept", "short kick"))
        p0 = 0.0 if mask is ok_nt else fit["p_rtd"]
        pr = np.array([p0, (1 - p0) * pa * pb, (1 - p0) * pa * (1 - pb), (1 - p0) * (1 - pa) * ps, (1 - p0) * (1 - pa) * (1 - ps)])
        pr = pr * np.array([len(r) > 0 for r in by])
        c = min(int(np.searchsorted(np.cumsum(pr), rng.random() * pr.sum(), side="right")), 4)
        rows = by[c]
        if nk is None or c == 0:
            return int(rows[min(int(rng.random() * len(rows)), len(rows) - 1)])
        cdf = np.cumsum(kick_weights("gauss", nk, kd[rows], gs[rows], sk, st, np.array([kdv]), np.array([tv]))[0])
        return int(rows[min(int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right")), len(rows) - 1)])

    return draw


def tov_enabled():
    return enabled() and "tov" in os.environ.get("RISK_PARTS", "tov,kick").split(",")


def kick_enabled():
    return enabled() and "kick" in os.environ.get("RISK_PARTS", "tov,kick").split(",")



if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if cmd == "diag":
        diag()
    elif cmd == "kfit":
        kfit()
    elif cmd == "kclass":
        kclass()
    elif cmd == "tfit":
        out = tfit()
        replay(out[0], out[6], out[7], out[8])
    elif cmd == "treplay":
        import json

        sp = json.loads(TFIT.read_text(encoding="utf-8"))
        cf = {}
        for k, v in sp["folds"].items():
            mi, tp, sn = (int(x) for x in k.split("|"))
            cf[(mi, tp, sn)] = np.asarray(v)
        replay(None, cf, sp["model"], sorted({k[2] for k in cf}))
