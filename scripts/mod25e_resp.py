import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_late as ml

ART = REPO / "artifacts" / "mod25e3"
OUTD = ART / "resp"
LABEL = "crHpqokg"
SEEDS = (11, 12, 13)
POOL = ml.POOL
BURN = ml.BURN
PBP = ml.PBP
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 300
MODE = sys.argv[2] if len(sys.argv) > 2 else "all"
BANDS = [("Q1", lambda g: g > 2700), ("Q2early", lambda g: (g <= 2700) & (g > 1920)), ("Q2last2", lambda g: (g <= 1920) & (g > 1800)),
         ("Q3", lambda g: (g <= 1800) & (g > 900)), ("Q4early", lambda g: (g <= 900) & (g >= 300)), ("Q4last5", lambda g: g < 300), ("all", lambda g: g >= 0)]
OUT = []


def say(s=""):
    OUT.append(s)
    print(s, flush=True)


def fmt(pt, bt):
    lo, hi = np.quantile(bt, [0.025, 0.975])
    return f"{pt:+.3f} [{lo:+.3f},{hi:+.3f}] pp {float((bt > 0).mean()):.2f}"


def real_drives():
    gf = pd.read_parquet(REPO / "data" / "processed" / "game_features_pbp.parquet", columns=["game_id", "home_score", "away_score"]).set_index("game_id")
    cols = ["game_id", "play_id", "season_type", "posteam", "defteam", "home_team", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential", "down", "ydstogo", "qb_kneel", "qb_spike", "interception", "fumble_lost", "yards_gained"]
    out = []
    for s in POOL:
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=cols)
        p["season"] = s
        p = p[(p.season_type == "REG") & p.posteam.notna() & p.score_differential.notna()]
        ot = p[p.qtr > 4].sort_values(["game_id", "play_id"], kind="stable").groupby("game_id").first()
        regfin = pd.Series((np.where(ot.posteam == ot.home_team, 1.0, -1.0) * ot.score_differential).to_numpy(), index=ot.index)
        p = p[p.qtr <= 4].sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
        p["m"] = np.where(p.posteam == p.home_team, 1.0, -1.0) * p.score_differential
        fin = regfin.reindex(p.game_id).fillna((gf.home_score - gf.away_score).reindex(p.game_id)).to_numpy()
        g = p.game_id.to_numpy()
        last = np.r_[g[1:] != g[:-1], True]
        nxt = np.r_[p.m.to_numpy()[1:], 0.0]
        p["dm"] = np.where(last, fin - p.m.to_numpy(), nxt - p.m.to_numpy())
        p = p[p.dm.notna()]
        ko = (p.play_type == "kickoff").to_numpy()
        p = p.assign(kocum=np.cumsum(ko))
        p = p[~ko & p.play_type.notna()].copy()
        g, pos, kc, q = p.game_id.to_numpy(), p.posteam.to_numpy(), p.kocum.to_numpy(), p.qtr.to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (pos[1:] != pos[:-1]) | (kc[1:] != kc[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
        p["d"] = np.cumsum(new)
        kn = (p.qb_kneel == 1) | (p.qb_spike == 1)
        rp = p.play_type.isin(["run", "pass"]) & ~kn
        p["is_run"] = (rp & (p.play_type == "run")).astype(int)
        p["is_pass"] = (rp & (p.play_type == "pass")).astype(int)
        p["x3a"] = (rp & (p.down == 3)).astype(int)
        p["x3c"] = (rp & (p.down == 3) & (p.yards_gained >= p.ydstogo)).astype(int)
        p["expl"] = (rp & (p.yards_gained >= 20)).astype(int)
        p["to"] = (rp & (p.down <= 3) & ((p.interception == 1) | (p.fumble_lost == 1))).astype(int)
        p["r4"] = ((p.down == 4) & p.play_type.isin(["run", "pass", "punt", "field_goal"]) & ~kn).astype(int)
        p["punt"] = (p.play_type == "punt").astype(int)
        p["fga"] = (p.play_type == "field_goal").astype(int)
        gp = p.groupby("d", sort=True)
        fo = gp.posteam.first()
        D = pd.DataFrame({"gk": gp.season.first().astype(str) + "_" + gp.game_id.first(), "off": fo, "dfn": gp.defteam.first(),
                          "s": np.where(fo == gp.home_team.first(), 1.0, -1.0), "yl0": gp.yardline_100.first(), "gsr0": gp.game_seconds_remaining.first(),
                          "sd0": gp.score_differential.first(), "dmsum": gp.dm.sum(), "run": gp.is_run.sum(), "pas": gp.is_pass.sum(), "x3a": gp.x3a.sum(),
                          "x3c": gp.x3c.sum(), "expl": gp.expl.sum(), "to": gp.to.max(), "r4": gp.r4.max(), "punt": gp.punt.max(), "fga": gp.fga.max(), "rz": (gp.yardline_100.min() <= 20).astype(int)})
        D["p"] = D.s * D.dmsum
        D["season"] = gp.season.first().astype(str)
        D["tm"] = D.season + "_" + D.off
        D["td"] = D.season + "_" + D.dfn
        out.append(D.drop(columns=["dmsum"]))
    return pd.concat(out, ignore_index=True)


def sim_drives():
    out = []
    for sd in SEEDS:
        sdir = ART / f"e5_{LABEL}_s{sd}"
        sg = pd.read_parquet(sdir / "sim_games.parquet", columns=["game_id", "home_team", "away_team"])
        sg["w"] = sg.game_id.str[1:5].astype(int)
        sg["s"] = sg.game_id.str[6:8].astype(int)
        sg["g"] = sg.game_id.str[-3:].astype(int)
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < BURN:
                continue
            d = pd.read_parquet(f)
            d = d[d.qtr <= 4].sort_values("g", kind="stable").reset_index(drop=True)
            k = sg[(sg.w == w) & (sg.s == s)].set_index("g")
            gi = d.g.astype(int).to_numpy()
            d["ht"] = k.home_team.reindex(gi).to_numpy()
            d["at"] = k.away_team.reindex(gi).to_numpy()
            g, oh, q = d.g.to_numpy(), d.offhome.to_numpy(), d.qtr.to_numpy()
            sc = ((d.po > 0) | (d.pdf > 0)).to_numpy()
            new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2)) | sc[:-1]]
            d["d"] = np.cumsum(new)
            c = d.code
            rp = c.isin([0, 1])
            d["pp"] = d.po - d.pdf
            d["is_run"] = (c == 0).astype(int)
            d["is_pass"] = (c == 1).astype(int)
            d["x3a"] = (rp & (d.down == 3)).astype(int)
            d["x3c"] = (rp & (d.down == 3) & (d.yards >= d.dist)).astype(int)
            d["expl"] = (rp & (d.yards >= 20)).astype(int)
            d["punt"] = (c == 2).astype(int)
            d["fga"] = (c == 3).astype(int)
            d["tp"] = (rp & (d.flip == 1) & (d.po == 0) & (d.down <= 3)).astype(int)
            d["r4"] = ((d.down == 4) & c.isin([0, 1, 2, 3])).astype(int)
            d["off"] = np.where(d.offhome == 1, d.ht, d["at"])
            d["dfn"] = np.where(d.offhome == 1, d["at"], d.ht)
            lv = c.isin([0, 1, 2, 3])
            gp = d.groupby("d", sort=True)
            gl = d[lv].groupby("d", sort=True)
            lastc = gl.code.last().reindex(gp.size().index)
            flip = gp.flip.last()
            key = f"{sd}_{w}_{s}"
            D = pd.DataFrame({"gk": key + "_" + gp.g.first().astype(int).astype(str), "off": gp.off.first(), "dfn": gp.dfn.first(),
                              "s": np.where(gp.offhome.first() == 1, 1.0, -1.0), "yl0": gp.yl.first(), "gsr0": gp.gsr.first(), "sd0": gp.sd.first(), "p": gp.pp.sum(),
                              "run": gp.is_run.sum(), "pas": gp.is_pass.sum(), "x3a": gp.x3a.sum(), "x3c": gp.x3c.sum(), "expl": gp.expl.sum(),
                              "to": gp.tp.max(), "r4": gp.r4.max(), "punt": gp.punt.max(), "fga": gp.fga.max(), "rz": (gp.yl.min() <= 20).astype(int)})
            D["season"] = key
            D["tm"] = key + "_" + D.off.astype(str)
            D["td"] = key + "_" + D.dfn.astype(str)
            out.append(D)
    return pd.concat(out, ignore_index=True)


def enrich(D):
    D = D.reset_index(drop=True)
    D["o"] = D.p.clip(lower=0)
    D["dd"] = (-D.p).clip(lower=0)
    D["td6"] = (D.p >= 6).astype(float)
    D["fg3"] = ((D.p > 0) & (D.p < 6)).astype(float)
    D["lead"] = D.sd0.clip(lower=0) / 7.0
    D["trail"] = (-D.sd0).clip(lower=0) / 7.0
    D["yl2"] = D.yl0 ** 2 / 100.0
    for nm, key in (("ro", "tm"), ("rd", "td")):
        gs = D.groupby(key)["o"].transform("sum")
        gn = D.groupby(key)["o"].transform("size")
        ga = D.groupby([key, "gk"])["o"].transform("sum")
        gk_n = D.groupby([key, "gk"])["o"].transform("size")
        D[nm] = (gs - ga) / (gn - gk_n).clip(lower=1)
    D["fgm"] = D.fga * (1 - D.fg3)
    D["oth"] = 1 - D.td6 - D.fg3 - D.punt - D.to - D.fgm
    D["bq"] = np.clip(1 + ((3600 - D.gsr0) // 900).astype(int), 1, 4)
    return D


def prep(D, mask, ycols, extra=()):
    S = D[mask]
    cols = ["lead", "trail", "yl0", "yl2", "ro", "rd", *extra]
    X = np.c_[np.ones(len(S)), S[cols].to_numpy(float)]
    Y = S[list(ycols)].to_numpy(float)
    gi = pd.factorize(S.gk)[0]
    G = gi.max() + 1
    k = X.shape[1]
    XX = np.zeros((G, k, k))
    for a in range(k):
        for b in range(a, k):
            v = np.bincount(gi, weights=X[:, a] * X[:, b], minlength=G)
            XX[:, a, b] = v
            XX[:, b, a] = v
    XY = np.stack([np.stack([np.bincount(gi, weights=X[:, a] * Y[:, j], minlength=G) for j in range(Y.shape[1])], 0) for a in range(k)], 0)
    return np.transpose(XX, (0, 1, 2)), np.transpose(XY, (2, 0, 1)), cols


def coefs(XX, XY, w):
    A = np.tensordot(w, XX, 1)
    B = np.tensordot(w, XY, 1)
    return np.linalg.solve(A + 1e-9 * np.eye(A.shape[0]), B)


def boot_coefs(XX, XY, rng):
    G = XX.shape[0]
    pt = coefs(XX, XY, np.ones(G))
    bt = np.stack([coefs(XX, XY, np.bincount(rng.integers(0, G, G), minlength=G).astype(float)) for _ in range(NB)])
    return pt, bt


def slopes(R, S, mr, ms, ycols, extra, rng):
    XXr, XYr, cols = prep(R, mr, ycols, extra)
    XXs, XYs, _ = prep(S, ms, ycols, extra)
    pr, br = boot_coefs(XXr, XYr, rng)
    ps, bs = boot_coefs(XXs, XYs, rng)
    return pr, br, ps, bs, cols


def report(tag, ycols, pr, br, ps, bs, cols, which=("trail", "lead")):
    for j, y in enumerate(ycols):
        for nm in which:
            i = 1 + cols.index(nm)
            say(f"  {tag}{y:5s} {nm:5s} real {fmt(pr[i, j], br[:, i, j])} | sim {fmt(ps[i, j], bs[:, i, j])} | sim-real {fmt(ps[i, j] - pr[i, j], bs[:, i, j] - br[:, i, j])}")


def games_frame(D, oc="o", dc="dd"):
    h = D.s.to_numpy() > 0
    gi, gk = pd.factorize(D.gk)
    G = len(gk)
    H = np.bincount(gi, weights=np.where(h, D[oc], D[dc]), minlength=G)
    A = np.bincount(gi, weights=np.where(h, D[dc], D[oc]), minlength=G)
    mg = D.s.to_numpy() * (D[oc].to_numpy() - D[dc].to_numpy())
    Q = np.stack([np.bincount(gi, weights=np.where(D.bq.to_numpy() == q, mg, 0.0), minlength=G) for q in (1, 2, 3, 4)], 1)
    return H, A, Q


def wcov(a, b, w):
    w = w / w.sum()
    return float((w * (a - w @ a) * (b - w @ b)).sum())


def stats(H, A, Q, w):
    xq = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            xq += 2 * wcov(Q[:, i], Q[:, j], w)
    return np.array([wcov(H, A, w), xq])


def cf_section(R, S, rng):
    say("== 3. counterfactual: sim drive points with the real-minus-sim response removed (first order, sim states held, band-specific fits, conditional on point slopes)")
    S2 = S.copy()
    mr, ms = {n: f(R.gsr0.to_numpy()) for n, f in BANDS}, {n: f(S.gsr0.to_numpy()) for n, f in BANDS}
    adj_o = np.zeros(len(S))
    adj_d = np.zeros(len(S))
    for nm, _ in BANDS[:-1]:
        XXr, XYr, cols = prep(R, mr[nm], ["o", "dd"], ())
        XXs, XYs, _ = prep(S, ms[nm], ["o", "dd"], ())
        br_ = coefs(XXr, XYr, np.ones(XXr.shape[0]))
        bs_ = coefs(XXs, XYs, np.ones(XXs.shape[0]))
        it, il = 1 + cols.index("trail"), 1 + cols.index("lead")
        idx = np.flatnonzero(ms[nm])
        for j, arr in ((0, adj_o), (1, adj_d)):
            arr[idx] = S.trail.to_numpy()[idx] * (bs_[it, j] - br_[it, j]) + S.lead.to_numpy()[idx] * (bs_[il, j] - br_[il, j])
    S2["o"] = S.o - adj_o
    S2["dd"] = S.dd - adj_d
    say(f"  mean removed per drive: offence {adj_o.mean():+.4f} defence {adj_d.mean():+.4f}; sim o/g {S.o.sum() / S.gk.nunique():.3f} -> {S2.o.sum() / S.gk.nunique():.3f}")
    Hr, Ar, Qr = games_frame(R)
    H0, A0, Q0 = games_frame(S)
    H1, A1, Q1 = games_frame(S2)
    Gr, Gs = len(Hr), len(H0)
    base = [stats(Hr, Ar, Qr, np.ones(Gr)), stats(H0, A0, Q0, np.ones(Gs)), stats(H1, A1, Q1, np.ones(Gs))]
    bt = []
    for _ in range(NB):
        wr = np.bincount(rng.integers(0, Gr, Gr), minlength=Gr).astype(float)
        ws = np.bincount(rng.integers(0, Gs, Gs), minlength=Gs).astype(float)
        bt.append([stats(Hr, Ar, Qr, wr), stats(H0, A0, Q0, ws), stats(H1, A1, Q1, ws)])
    bt = np.array(bt)
    for k, nm in enumerate(["cov(H,A)", "cross-quarter 2xcov (raw margin increments, not strength adjusted)"]):
        r, s0, s1 = (b[k] for b in base)
        say(f"  {nm}: real {r:+.2f} sim {s0:+.2f} sim_cf {s1:+.2f}")
        say(f"    gap sim-real before {fmt(s0 - r, bt[:, 1, k] - bt[:, 0, k])}; after {fmt(s1 - r, bt[:, 2, k] - bt[:, 0, k])}; accounted (before-after) {fmt(s0 - s1, bt[:, 1, k] - bt[:, 2, k])}")


def real_plays():
    cols = ["game_id", "season_type", "posteam", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential", "down", "ydstogo", "qb_kneel", "qb_spike", "interception", "fumble_lost", "yards_gained"]
    out = []
    for s in POOL:
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=cols)
        p = p[(p.season_type == "REG") & p.posteam.notna() & p.score_differential.notna() & (p.qtr >= 3) & (p.qtr <= 4) & p.down.notna()]
        kn = (p.qb_kneel == 1) | (p.qb_spike == 1)
        kind = np.select([(p.play_type == "run") & ~kn, (p.play_type == "pass") & ~kn, p.play_type == "punt", p.play_type == "field_goal"], [0, 1, 2, 3], -1)
        p = p.assign(kind=kind)
        p = p[p.kind >= 0]
        to = (p.kind <= 1) & (p.down <= 3) & ((p.interception == 1) | (p.fumble_lost == 1))
        out.append(pd.DataFrame({"gk": str(s) + "_" + p.game_id, "off": p.posteam.astype(str), "qtr": p.qtr, "gsr": p.game_seconds_remaining, "sd": p.score_differential,
                                 "down": p.down, "dist": p.ydstogo, "yl": p.yardline_100, "kind": p.kind, "yards": np.where((p.kind <= 1) & (p.yards_gained >= p.yardline_100), p.yardline_100, p.yards_gained), "to": to.astype(int)}))
    return pd.concat(out, ignore_index=True)


def sim_plays():
    out = []
    use = ["g", "down", "dist", "yl", "sd", "gsr", "qtr", "code", "po", "flip", "offhome", "yards"]
    for sd in SEEDS:
        sdir = ART / f"e5_{LABEL}_s{sd}"
        sg = pd.read_parquet(sdir / "sim_games.parquet", columns=["game_id", "home_team", "away_team"])
        sg["w"] = sg.game_id.str[1:5].astype(int)
        sg["s"] = sg.game_id.str[6:8].astype(int)
        sg["g"] = sg.game_id.str[-3:].astype(int)
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < BURN:
                continue
            d = pd.read_parquet(f, columns=use)
            d = d[(d.qtr >= 3) & (d.qtr <= 4) & d.code.isin([0, 1, 2, 3])]
            k = sg[(sg.w == w) & (sg.s == s)].set_index("g")
            gi = d.g.astype(int).to_numpy()
            ht = k.home_team.reindex(gi).to_numpy()
            at = k.away_team.reindex(gi).to_numpy()
            key = f"{sd}_{w}_{s}"
            to = (d.code.isin([0, 1]) & (d.flip == 1) & (d.po == 0) & (d.down <= 3)).astype(int)
            out.append(pd.DataFrame({"gk": key + "_" + d.g.astype(int).astype(str).to_numpy(), "off": np.where(d.offhome.to_numpy() == 1, ht, at).astype(str), "qtr": d.qtr.to_numpy(), "gsr": d.gsr.to_numpy(),
                                     "sd": d.sd.to_numpy(), "down": d.down.to_numpy(), "dist": d.dist.to_numpy(), "yl": d.yl.to_numpy(), "kind": d.code.to_numpy().astype(int), "yards": np.where(d.code.isin([0, 1]).to_numpy() & (d.po.to_numpy() >= 6), d.yl.to_numpy(), d.yards.to_numpy()), "to": to.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def penrich(P, D):
    rt = D[["gk", "off", "ro", "rd"]].copy()
    rt["off"] = rt.off.astype(str)
    rt = rt.drop_duplicates(["gk", "off"])
    P = P.merge(rt, on=["gk", "off"], how="left")
    P["ro"] = P.ro.fillna(rt.ro.mean())
    P["rd"] = P.rd.fillna(rt.rd.mean())
    q = np.quantile(rt.ro, [1 / 3, 2 / 3])
    P["st"] = np.digitize(P.ro, q)
    rp = P.kind <= 1
    nt = P.to == 0
    P["ispass"] = (P.kind == 1).astype(float)
    P["trail"] = (-P.sd).clip(lower=0) / 7.0
    P["lead"] = P.sd.clip(lower=0) / 7.0
    P["yl2"] = P.yl ** 2 / 100.0
    P["yl3"] = P.yl ** 3 / 1e4
    P["dist2"] = P.dist ** 2 / 100.0
    P["yld"] = P.yl * P.dist / 100.0
    for n in (2, 3, 4):
        P[f"d{n}"] = (P.down == n).astype(float)
    P["fd"] = (rp & nt & (P.yards >= P.dist)).astype(float)
    P["expl"] = (rp & nt & (P.yards >= 20)).astype(float)
    P["neg"] = (rp & nt & (P.yards < 0)).astype(float)
    P["zero"] = (rp & nt & (P.yards == 0)).astype(float)
    P["punt"] = (P.kind == 2).astype(float)
    P["fg"] = (P.kind == 3).astype(float)
    P["go"] = rp.astype(float)
    P["to"] = P.to.astype(float)
    return P


def gprep(P, mask, ycols, xcols, gi, G):
    S = P[mask]
    X = np.c_[np.ones(len(S)), S[xcols].to_numpy(float)]
    Y = S[ycols].to_numpy(float)
    g = gi[mask]
    k = X.shape[1]
    XX = np.zeros((G, k, k))
    XY = np.zeros((G, k, Y.shape[1]))
    for a in range(k):
        for b in range(a, k):
            v = np.bincount(g, weights=X[:, a] * X[:, b], minlength=G)
            XX[:, a, b] = v
            XX[:, b, a] = v
        for j in range(Y.shape[1]):
            XY[:, a, j] = np.bincount(g, weights=X[:, a] * Y[:, j], minlength=G)
    return XX, XY


def fit_models(P, models, gi, G, ws):
    pt, bt = [], []
    for name, mask, ycols, xcols in models:
        XX, XY = gprep(P, mask, ycols, xcols, gi, G)
        pt.append(coefs(XX, XY, np.ones(G)))
        bt.append(np.stack([coefs(XX, XY, w) for w in ws]))
    return pt, bt


def shared_boot(R, S, mr, ms, models, rng):
    gR, kR = pd.factorize(R.gk)
    gS, kS = pd.factorize(S.gk)
    GR, GS = len(kR), len(kS)
    wR = [np.bincount(rng.integers(0, GR, GR), minlength=GR).astype(float) for _ in range(NB)]
    wS = [np.bincount(rng.integers(0, GS, GS), minlength=GS).astype(float) for _ in range(NB)]
    mR = [(n, mk(R) & mr, y, x) for n, mk, y, x in models]
    mS = [(n, mk(S) & ms, y, x) for n, mk, y, x in models]
    pr, br = fit_models(R, mR, gR, GR, wR)
    ps, bs = fit_models(S, mS, gS, GS, wS)
    return pr, br, ps, bs


def report_models(models, pr, br, ps, bs, which=("trail", "lead")):
    for m, (n, _, ys, xs) in enumerate(models):
        for j, y in enumerate(ys):
            for nm in which:
                i = 1 + xs.index(nm)
                say(f"  {n:6s} {y:6s} {nm:5s} real {fmt(pr[m][i, j], br[m][:, i, j])} | sim {fmt(ps[m][i, j], bs[m][:, i, j])} | sim-real {fmt(ps[m][i, j] - pr[m][i, j], bs[m][:, i, j] - br[m][:, i, j])}")


XC = ["trail", "lead", "ro", "rd", "yl", "yl2", "dist", "dist2", "d2", "d3", "d4"]
XC4 = ["trail", "lead", "ro", "rd", "yl", "yl2", "yl3", "dist", "dist2", "yld"]
XCM = ["trail", "lead", "ro", "rd"]


def play_models():
    isrp = lambda P: (P.kind <= 1).to_numpy()
    isp = lambda P: (P.kind == 1).to_numpy()
    isr = lambda P: (P.kind == 0).to_numpy()
    nt = lambda P: (P.to == 0).to_numpy()
    return [("mix", isrp, ["ispass", "to", "fd", "expl"], XC), ("pass", isp, ["to", "fd", "expl", "neg", "zero"], XC), ("run", isr, ["to", "fd", "expl", "neg"], XC),
            ("ypa", lambda P: isp(P) & nt(P), ["yards"], XC), ("ypc", lambda P: isr(P) & nt(P), ["yards"], XC), ("ypp", lambda P: isrp(P) & nt(P), ["yards"], XC)]


def decomp(PR, PS, pr, br, ps, bs, sel):
    info = []
    for P in (PR, PS):
        a = P[sel(P)]
        rp = a[a.kind <= 1]
        n_ = rp[rp.to == 0]
        info.append((rp.ispass.mean(), n_[n_.kind == 1].yards.mean(), n_[n_.kind == 0].yards.mean()))
    for nm, i in (("trail", 1), ("lead", 2)):
        row = []
        for (s, yp, yr), p_, b_ in zip(info, (pr, ps), (br, bs)):
            def comp(sh, ypa, ypc, ypp):
                return np.array([sh * (yp - yr), s * ypa, (1 - s) * ypc, ypp])
            pt = comp(p_[0][i, 0], p_[3][i, 0], p_[4][i, 0], p_[5][i, 0])
            bt = np.stack([comp(b_[0][t][i, 0], b_[3][t][i, 0], b_[4][t][i, 0], b_[5][t][i, 0]) for t in range(NB)])
            row.append((pt, bt))
        say(f"    {nm}: mean share real {info[0][0]:.3f} sim {info[1][0]:.3f}; Yp-Yr real {info[0][1] - info[0][2]:+.2f} sim {info[1][1] - info[1][2]:+.2f}")
        for k, lab in enumerate(["mix shift", "within pass", "within run", "total ypp slope"]):
            say(f"      {lab:15s} real {fmt(row[0][0][k], row[0][1][:, k])} | sim {fmt(row[1][0][k], row[1][1][:, k])} | sim-real {fmt(row[1][0][k] - row[0][0][k], row[1][1][:, k] - row[0][1][:, k])}")


def cellstr(a):
    pa = a[a.kind == 1]
    ra = a[a.kind == 0]
    pn = pa[pa.to == 0]
    rn = ra[ra.to == 0]
    return f"n {len(a):6d} pass {a.ispass.mean():.3f} | pass: to {pa.to.mean():.3f} zero {pa.zero.mean():.3f} neg {pa.neg.mean():.3f} fd {pa.fd.mean():.3f} expl {pa.expl.mean():.3f} ypa {pn.yards.mean():.2f} | run: ypc {rn.yards.mean():.2f} fd {ra.fd.mean():.3f}"


def table(PR, PS, qlab, qf):
    say(f"-- descriptive {qlab}: offence margin band, run/pass plays; real then sim")
    for lo, hi in [(-99, -17), (-16, -9), (-8, -1), (0, 0), (1, 8), (9, 16), (17, 99)]:
        for lab, P in (("real", PR), ("sim ", PS)):
            say(f"  sd [{lo},{hi}] {lab} " + cellstr(P[qf(P) & (P.sd >= lo) & (P.sd <= hi) & (P.kind <= 1)]))


def strata_table(PR, PS, qlab, qf):
    say(f"-- descriptive {qlab} by offence strength tercile (own-dataset terciles of leave-game-out offence rating)")
    for lab, sf in (("trail", lambda P: P.sd < 0), ("lead", lambda P: P.sd > 0)):
        for st in (0, 1, 2):
            for nm, P in (("real", PR), ("sim ", PS)):
                say(f"  {lab} st{st} {nm} " + cellstr(P[qf(P) & sf(P) & (P.st == st) & (P.kind <= 1)]))


def play_main(R, S, rng):
    PR = penrich(real_plays(), R)
    PS = penrich(sim_plays(), S)
    say(f"plays Q3-Q4: real {len(PR)} games {PR.gk.nunique()}; sim {len(PS)} games {PS.gk.nunique()}; NB {NB}")
    for qlab, qf in (("Q3", lambda P: (P.gsr <= 1800) & (P.gsr > 900)), ("Q4", lambda P: P.gsr <= 900)):
        table(PR, PS, qlab, qf)
        strata_table(PR, PS, qlab, qf)
    pm = play_models()
    pbands = [("Q3", lambda P: ((P.gsr <= 1800) & (P.gsr > 900)).to_numpy()), ("Q4early", lambda P: ((P.gsr <= 900) & (P.gsr >= 300)).to_numpy()),
              ("Q4last5", lambda P: (P.gsr < 300).to_numpy()), ("Q4all", lambda P: (P.gsr <= 900).to_numpy())]
    for st in (0, 1, 2):
        pbands.append((f"Q4all_st{st}", (lambda s_: lambda P: ((P.gsr <= 900) & (P.st == s_)).to_numpy())(st)))
    for nm, bf in pbands:
        say(f"== 2. play mix and outcomes, band {nm}: OLS slopes per 7 points (trail, lead), controls ro rd yl yl2 dist dist2 down dummies; game bootstrap, shared draws")
        pr, br, ps, bs = shared_boot(PR, PS, bf(PR), bf(PS), pm, rng)
        report_models(pm, pr, br, ps, bs)
        say("  ypp slope decomposition (share slope x (Yp-Yr), share x pass slope, (1-share) x run slope)")
        decomp(PR, PS, pr, br, ps, bs, bf)
    d4 = [("4th", lambda P: (P.down == 4).to_numpy(), ["punt", "fg", "go"], XC4), ("4thst", lambda P: (P.down == 4).to_numpy(), ["yl", "dist"], XCM)]
    for nm, bf in pbands[:4]:
        say(f"== 3a. fourth-down decision and state, band {nm}: slope per 7 points (controls yl yl2 yl3 dist dist2 yl*dist ro rd)")
        pr, br, ps, bs = shared_boot(PR, PS, bf(PR), bf(PS), d4, rng)
        report_models(d4, pr, br, ps, bs)
    ends = ["td6", "fg3", "fgm", "punt", "to", "oth", "r4"]
    say("== 3b. drive endings and reach-4th by lead (drive level, controls yl0 yl0^2 ro rd)")
    mr = {n: f(R.gsr0.to_numpy()) for n, f in BANDS}
    ms = {n: f(S.gsr0.to_numpy()) for n, f in BANDS}
    for nm in ("Q3", "Q4early", "Q4last5"):
        say(f"-- band {nm}: leading-offence means real/sim " + " ".join(f"{c} {R[mr[nm] & (R.sd0 > 0).to_numpy()][c].mean():.3f}/{S[ms[nm] & (S.sd0 > 0).to_numpy()][c].mean():.3f}" for c in ends))
        pr, br, ps, bs, cols = slopes(R, S, mr[nm], ms[nm], ends, (), rng)
        report("", ends, pr, br, ps, bs, cols)
    return R, S


def main():
    rng = np.random.default_rng(7)
    R = enrich(real_drives())
    S = enrich(sim_drives())
    if MODE == "play":
        return play_main(R, S, rng)
    say(f"turnover flag mean real {R.to.mean():.3f} sim {S.to.mean():.3f}; r4 real {R.r4.mean():.3f} sim {S.r4.mean():.3f}")
    say(f"real drives {len(R)} games {R.gk.nunique()} o/g {R.o.sum() / R.gk.nunique():.2f}; sim drives {len(S)} games {S.gk.nunique()} o/g {S.o.sum() / S.gk.nunique():.2f}; NB {NB}")
    gt = pd.read_parquet(REPO / "data" / "processed" / "game_features_pbp.parquet", columns=["game_id", "season", "home_score", "away_score"])
    gt["gk"] = gt.season.astype(int).astype(str) + "_" + gt.game_id
    Hr, Ar, _ = games_frame(R)
    kk = pd.factorize(R.gk)[1]
    gg = gt.set_index("gk").reindex(kk)
    say(f"check real reconstruction vs game table: H mean {Hr.mean():.2f}/{gg.home_score.mean():.2f} A mean {Ar.mean():.2f}/{gg.away_score.mean():.2f} cov(H,A) {np.cov(Hr, Ar)[0, 1]:+.2f}/{np.cov(gg.home_score, gg.away_score)[0, 1]:+.2f} corr(H,Hgame) {np.corrcoef(Hr, gg.home_score)[0, 1]:.3f}")
    mr = {n: f(R.gsr0.to_numpy()) for n, f in BANDS}
    ms = {n: f(S.gsr0.to_numpy()) for n, f in BANDS}
    if MODE == "all":
        say("== 1. response of the offence to the margin: trail = deficit/7 pts, lead = lead/7 pts; OLS with yl0, yl0^2/100, leave-game-out team offence and opponent defence rating; same estimator both; game bootstrap")
        outs = ["o", "dd", "p", "td6", "fg3"]
        for nm, _ in BANDS:
            say(f"-- band {nm} real n {mr[nm].sum()} sim n {ms[nm].sum()}")
            pr, br, ps, bs, cols = slopes(R, S, mr[nm], ms[nm], outs, (), rng)
            report("", outs, pr, br, ps, bs, cols)
    say("== 2a. phase variables per drive, response to trail/lead (same controls)")
    ph = ["run", "pas", "expl", "x3a", "x3c", "to", "rz", "punt", "fga"]
    for nm in ("Q3", "Q4early", "Q4last5", "all"):
        say(f"-- band {nm}")
        pr, br, ps, bs, cols = slopes(R, S, mr[nm], ms[nm], ph, (), rng)
        report("", ph, pr, br, ps, bs, cols)
        for sel, tag in ((R.sd0 < 0, "trailing"), (R.sd0 > 0, "leading")):
            pass
        rm = mr[nm]
        sm = ms[nm]
        for lab, fr, fs in (("trailing", R.sd0 < 0, S.sd0 < 0), ("leading", R.sd0 > 0, S.sd0 > 0)):
            a, b = R[rm & fr.to_numpy()], S[sm & fs.to_numpy()]
            say(f"  means {lab} offences real n {len(a)} sim n {len(b)}: " + " ".join(f"{c} {a[c].mean():.3f}/{b[c].mean():.3f}" for c in ["yl0", "run", "pas", "expl", "x3a", "to", "rz", "punt", "fga", "o", "dd"]) + f" 3rd conv {a.x3c.sum() / max(a.x3a.sum(), 1):.3f}/{b.x3c.sum() / max(b.x3a.sum(), 1):.3f}")
    say("== 2b. mediation: trail/lead slope of net drive points p after adding phase controls cumulatively (order dependent, inferred)")
    stages = [("base", ()), ("+pass,run", ("pas", "run")), ("+explosive", ("pas", "run", "expl")), ("+3rd att/conv", ("pas", "run", "expl", "x3a", "x3c")),
              ("+turnover", ("pas", "run", "expl", "x3a", "x3c", "to")), ("+red zone", ("pas", "run", "expl", "x3a", "x3c", "to", "rz")), ("+punt,fga", ("pas", "run", "expl", "x3a", "x3c", "to", "rz", "punt", "fga"))]
    for nm in ("Q3", "Q4early", "Q4last5", "all"):
        say(f"-- band {nm}")
        for sname, ex in stages:
            pr, br, ps, bs, cols = slopes(R, S, mr[nm], ms[nm], ["p"], ex, rng)
            report(f"{sname:14s}", ["p"], pr, br, ps, bs, cols)
    if MODE == "all":
        cf_section(R, S, rng)
    return R, S


if __name__ == "__main__":
    main()
    OUTD.mkdir(exist_ok=True)
    (OUTD / {"all": "resp.txt", "med": "resp_med.txt", "play": "resp_play.txt"}[MODE]).write_text("\n".join(OUT))
