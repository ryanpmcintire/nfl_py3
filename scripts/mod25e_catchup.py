import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_late as late  # noqa: E402

ART = REPO / "artifacts" / "mod25e3"
LABEL = os.environ.get("CU_LABEL", "crHpqokgnd")
SEEDS = tuple(int(x) for x in os.environ.get("CU_SEEDS", "11,12,13").split(","))
OUTDIR = ART / os.environ.get("CU_OUT", "catchup")
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = []
LOOKS = [0]
BANDS = ["trail>8", "trail1-8", "tied", "lead1-8", "lead>8"]
PAIRS = [(q, r) for q in range(4) for r in range(q + 1, 4)]
PNAME = [f"{q + 1}{r + 1}" for q, r in PAIRS]
rng = np.random.default_rng(58)
NBND = [5]
FINE_EDGES = [-np.inf] + [float(x) + 0.5 for x in range(-21, 21, 3)] + [np.inf]


def say(s=""):
    print(s)
    OUT.append(s)


def pp(d):
    return float((d > 0).mean())


def ci(d, pt, tag=""):
    LOOKS[0] += 1
    return f"{pt:+.3f} [{np.percentile(d, 5):+.3f},{np.percentile(d, 95):+.3f}] pp {pp(d):.2f}"


def sim_drives(sd):
    sdir = ART / f"e5_{LABEL}_s{sd}"
    sg = pd.read_parquet(sdir / "sim_games.parquet", columns=["game_id", "home_team", "away_team"])
    sg["w"] = sg.game_id.str[1:5].astype(int)
    sg["s"] = sg.game_id.str[6:8].astype(int)
    sg["g"] = sg.game_id.str[-3:].astype(int)
    out = []
    for f in sorted(sdir.glob("play_*_*.parquet")):
        w, s = (int(x) for x in f.stem.split("_")[1:])
        if s < late.BURN:
            continue
        tm = sg[(sg.w == w) & (sg.s == s)].set_index("g")
        d = pd.read_parquet(f, columns=["g", "qtr", "offhome", "yl", "el", "sd", "po", "pdf", "code"])
        d = d[d.qtr <= 4].sort_values(["g"], kind="stable").reset_index(drop=True)
        g, oh, q = d.g.to_numpy(), d.offhome.to_numpy(), d.qtr.to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
        d["d"] = np.cumsum(new)
        d["pp"] = d.po - d.pdf
        gi = d.g.astype(int)
        d["ht"] = tm.home_team.reindex(gi).to_numpy()
        d["at"] = tm.away_team.reindex(gi).to_numpy()
        d["off"] = np.where(d.offhome == 1, d.ht, d["at"])
        d["dfn"] = np.where(d.offhome == 1, d["at"], d.ht)
        gp = d.groupby("d", sort=True)
        key = f"{sd}_{w}_{s}"
        D = pd.DataFrame({"gk": key + "_" + gp.g.first().astype(int).astype(str), "season": 0, "off": gp.off.first(), "dfn": gp.dfn.first(),
                          "s": np.where(gp.offhome.first() == 1, 1.0, -1.0), "ql": gp.qtr.last(), "yl0": gp.yl.first(), "secs": gp.el.sum(), "n": gp.size(),
                          "sd0": gp.sd.first(), "p": gp.pp.sum(), "pt": np.where(gp.code.last() == 2, "punt", "x")})
        D["tm"] = key + "_" + D.off.astype(str)
        D["td"] = key + "_" + D.dfn.astype(str)
        out.append(D)
    return pd.concat(out, ignore_index=True)


def drives_all(ns):
    real = ns["real_drives"](ns["POOL"])[0]
    return real, pd.concat([sim_drives(sd) for sd in SEEDS], ignore_index=True)


def prep(ns, D, edges=None):
    D, _ = ns["components"](D)
    D = D.assign(a=D.p - D.c_strength)
    gk = pd.Index(sorted(D.gk.unique()))
    D["gi"] = gk.get_indexer(D.gk)
    D["qi"] = np.clip(D.ql.astype(int).to_numpy(), 1, 4) - 1
    D["bi"] = pd.cut(D.sd0, edges or ns["LEAD_EDGES"], labels=False).astype(int)
    return D, len(gk)


def wcov(W, x, y):
    sw = W.sum(1, keepdims=True)
    mx = (W * x).sum(1, keepdims=True) / sw
    my = (W * y).sum(1, keepdims=True) / sw
    return (W * x * y).sum(1, keepdims=True)[:, 0] / sw[:, 0] - mx[:, 0] * my[:, 0]


def cross_terms(D, G, W, abar_override=None):
    s = D.s.to_numpy()
    a = D.a.to_numpy()
    gi, qi, bi = D.gi.to_numpy(), D.qi.to_numpy(), D.bi.to_numpy()
    nb = NBND[0]
    cell = qi * nb + bi
    Wd = W[:, gi]
    num = np.stack([np.bincount(cell, weights=Wd[b] * a, minlength=4 * nb) for b in range(len(W))])
    den = np.stack([np.bincount(cell, weights=Wd[b], minlength=4 * nb) for b in range(len(W))])
    abar = num / np.maximum(den, 1e-9)
    if abar_override is not None:
        abar = abar_override
    I = np.zeros((G, 4))
    np.add.at(I, (gi, qi), s * a)
    m = np.zeros((G, 4 * nb))
    np.add.at(m, (gi, cell), s)
    ab_row = abar[:, None, :]
    IBS = m[None, :, :] * ab_row
    IBS = IBS.reshape(len(W), G, 4, nb)
    IS = IBS.sum(3)
    return I, IBS, IS, abar


def pair_cov_table(Dr, Gr, Ds, Gs):
    nbo = NBND[0] if NBND[0] == 5 else 1
    Wr = rng.multinomial(Gr, np.ones(Gr) / Gr, size=NB).astype(float)
    Ws = rng.multinomial(Gs, np.ones(Gs) / Gs, size=NB).astype(float)
    ones_r, ones_s = np.ones((1, Gr)), np.ones((1, Gs))
    res = {}
    for nm, D, G, W, o in (("real", Dr, Gr, Wr, ones_r), ("sim", Ds, Gs, Ws, ones_s)):
        out = {}
        for tag, WW in (("pt", o), ("bt", W)):
            I, IBS, IS, abar = cross_terms(D, G, WW)
            IE = I[None] - IS
            tot = np.stack([2 * wcov(WW, np.broadcast_to(I[:, q], (len(WW), G)), np.broadcast_to(I[:, r], (len(WW), G))) for q, r in PAIRS], 1)
            S = np.stack([2 * wcov(WW, np.broadcast_to(I[:, q], (len(WW), G)), IS[:, :, r]) for q, r in PAIRS], 1)
            E = tot - S
            band = np.stack([np.stack([2 * wcov(WW, np.broadcast_to(I[:, q], (len(WW), G)), IBS[:, :, r, b]) for b in range(nbo)], 1) for q, r in PAIRS], 1)
            out[tag] = dict(tot=tot, S=S, E=E, band=band, abar=abar, WW=WW)
        res[nm] = out
    cfS = {}
    for tag in ("pt", "bt"):
        ar = res["real"][tag]["abar"]
        if tag == "bt":
            ar_use = ar
        else:
            ar_use = ar
        WWs = res["sim"][tag]["WW"]
        I, IBS, IS, _ = cross_terms(Ds, Gs, WWs, abar_override=ar_use)
        cfS[tag] = np.stack([2 * wcov(WWs, np.broadcast_to(I[:, q], (len(WWs), Gs)), IS[:, :, r]) for q, r in PAIRS], 1)
    return res, cfS


def report_cross(res, cfS):
    say("## A. cross-quarter covariance (2x cov, strength-adjusted drive points, home perspective); S = state response (band mean pts per drive x signed drive count), E = within-band residual")
    r, s = res["real"], res["sim"]
    for key, lab in (("tot", "total"), ("S", "S state response"), ("E", "E residual")):
        say(f"  {lab}: pair real | sim | sim-real [5,95] pp")
        for i, n in enumerate(PNAME):
            d = s["bt"][key][:, i] - r["bt"][key][:, i]
            pt = s["pt"][key][0, i] - r["pt"][key][0, i]
            say(f"    {n} {r['pt'][key][0, i]:+.2f} | {s['pt'][key][0, i]:+.2f} | {ci(d, pt)}")
        d = s["bt"][key].sum(1) - r["bt"][key].sum(1)
        say(f"    sum {r['pt'][key].sum():+.2f} | {s['pt'][key].sum():+.2f} | {ci(d, s['pt'][key].sum() - r['pt'][key].sum())}")
    say("  S channel counterfactual: sim signed drive counts with real band means (sim S minus counterfactual = rate-response gap; counterfactual minus real S = occupancy/count gap)")
    for i, n in enumerate(PNAME):
        rate = s["bt"]["S"][:, i] - cfS["bt"][:, i]
        occ = cfS["bt"][:, i] - r["bt"]["S"][:, i]
        say(f"    {n} rate-response gap {ci(rate, s['pt']['S'][0, i] - cfS['pt'][0, i])} | occupancy gap {ci(occ, cfS['pt'][0, i] - r['pt']['S'][0, i])}")
    d = s["bt"]["S"].sum(1) - cfS["bt"].sum(1)
    say(f"    sum rate-response {ci(d, s['pt']['S'].sum() - cfS['pt'].sum())}")
    d = cfS["bt"].sum(1) - r["bt"]["S"].sum(1)
    say(f"    sum occupancy {ci(d, cfS['pt'].sum() - r['pt']['S'].sum())}")
    say("  band contribution to S (sim-real), pair x offence-lead band at drive start in later quarter")
    for i, n in enumerate(PNAME):
        row = []
        for b in range(5):
            d = s["bt"]["band"][:, i, b] - r["bt"]["band"][:, i, b]
            pt = s["pt"]["band"][0, i, b] - r["pt"]["band"][0, i, b]
            row.append(f"{BANDS[b]} {ci(d, pt)}")
        say(f"    {n}: " + " ; ".join(row))
    say("  band mean adjusted pts per drive by quarter (real | sim | sim-real)")
    for q in range(4):
        row = []
        for b in range(5):
            c = q * 5 + b
            d = s["bt"]["abar"][:, c] - r["bt"]["abar"][:, c]
            pt = s["pt"]["abar"][0, c] - r["pt"]["abar"][0, c]
            row.append(f"{BANDS[b]} {r['pt']['abar'][0, c]:.2f}|{s['pt']['abar'][0, c]:.2f}|{ci(d, pt)}")
        say(f"    Q{q + 1}: " + " ; ".join(row))


def gsum_boot(gi, G, cell, ncell, vals):
    K = len(vals)
    idx = gi * ncell + cell
    S = np.stack([np.bincount(idx, weights=v, minlength=G * ncell).reshape(G, ncell) for v in vals], 2)
    return S


def boot_tot(S, W):
    return np.einsum("bg,gck->bck", W, S)


def drive_table(Dr, Gr, Ds, Gs):
    say("## B. offence drive stats by quarter (drive's last quarter) x offence lead band at drive start; sim minus real, game bootstrap")
    out = {}
    for nm, D, G in (("real", Dr, Gr), ("sim", Ds, Gs)):
        cell = (D.qi * 5 + D.bi).to_numpy()
        S = gsum_boot(D.gi.to_numpy(), G, cell, 20, [np.ones(len(D)), D.a.to_numpy(), D.secs.to_numpy(), D.n.to_numpy().astype(float)])
        W = rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)
        out[nm] = (boot_tot(S, np.ones((1, G))), boot_tot(S, W), G)
    for q in range(4):
        say(f"  Q{q + 1}")
        for b in range(5):
            c = q * 5 + b
            row = []
            for lab, f in (("drives/g", lambda t, G: t[..., c, 0] / G), ("adj pts/drive", lambda t, G: t[..., c, 1] / np.maximum(t[..., c, 0], 1e-9)),
                           ("secs/drive", lambda t, G: t[..., c, 2] / np.maximum(t[..., c, 0], 1e-9)), ("rows/drive", lambda t, G: t[..., c, 3] / np.maximum(t[..., c, 0], 1e-9))):
                pr = f(out["real"][0], out["real"][2])[0]
                ps = f(out["sim"][0], out["sim"][2])[0]
                d = f(out["sim"][1], out["sim"][2]) - f(out["real"][1], out["real"][2])
                row.append(f"{lab} {pr:.3f}|{ps:.3f} {ci(d, ps - pr)}")
            say(f"    {BANDS[b]:8s} " + " ; ".join(row))


def real_plays(ns):
    cols = ["game_id", "season_type", "posteam", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential", "down", "yards_gained", "qb_kneel", "qb_spike"]
    out = []
    for s in ns["POOL"]:
        p = pd.read_parquet(late.PBP / f"season={s}" / "plays.parquet", columns=cols)
        p = p[(p.season_type == "REG") & p.posteam.notna() & p.score_differential.notna() & (p.qtr <= 4) & p.down.notna()]
        p = p[p.play_type.isin(["run", "pass", "punt", "field_goal"]) & (p.qb_kneel != 1) & (p.qb_spike != 1)]
        code = p.play_type.map({"run": 0, "pass": 1, "punt": 2, "field_goal": 3}).to_numpy()
        out.append(pd.DataFrame({"gk": str(s) + "_" + p.game_id.to_numpy(), "qtr": p.qtr.astype(int).to_numpy(), "gsr": p.game_seconds_remaining.to_numpy(), "sd": p.score_differential.to_numpy(),
                                 "down": p.down.to_numpy(), "code": code, "yards": p.yards_gained.fillna(0).to_numpy()}))
    return pd.concat(out, ignore_index=True)


def sim_plays():
    out = []
    use = ["g", "down", "sd", "gsr", "qtr", "code", "yards"]
    for sd in SEEDS:
        for f in sorted((ART / f"e5_{LABEL}_s{sd}").glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < late.BURN:
                continue
            d = pd.read_parquet(f, columns=use)
            d = d[(d.qtr <= 4) & d.code.isin([0, 1, 2, 3]) & d.down.notna()]
            out.append(pd.DataFrame({"gk": f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str).to_numpy(), "qtr": d.qtr.astype(int).to_numpy(), "gsr": d.gsr.to_numpy(), "sd": d.sd.to_numpy(),
                                     "down": d.down.to_numpy(), "code": d.code.astype(int).to_numpy(), "yards": d.yards.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def play_frames(P, ns):
    gk = pd.Index(sorted(P.gk.unique()))
    P = P.assign(gi=gk.get_indexer(P.gk))
    P["bi"] = pd.cut(P.sd, ns["LEAD_EDGES"], labels=False).astype(int)
    P["rp"] = P.code.isin([0, 1])
    P["clean"] = P.rp & (P.yards > -20)
    return P, len(gk)


def play_table(Pr, Gr, Ps, Gs):
    say("## C. offence play stats by quarter x offence lead band (state at the play); defence of the leader = offence rows in the trail bands; sim minus real")
    say("   yards use rows with yards > -20 (pool terminal-row artifact); explosive = yards >= 20; go rate = run/pass share of 4th-down punt/FG/run/pass")
    out = {}
    for nm, P, G in (("real", Pr, Gr), ("sim", Ps, Gs)):
        cell = ((P.qtr - 1) * 5 + P.bi).to_numpy()
        rp, cl = P.rp.to_numpy(), P.clean.to_numpy()
        pas = (P.code == 1).to_numpy()
        y = P.yards.to_numpy()
        f4 = (P.down == 4).to_numpy()
        vals = [rp.astype(float), (rp & pas).astype(float), np.where(cl, y, 0.0), cl.astype(float), (cl & (y >= 20)).astype(float), f4.astype(float), (f4 & rp).astype(float),
                np.where(cl & pas, y, 0.0), (cl & pas).astype(float), np.where(cl & ~pas, y, 0.0), (cl & ~pas).astype(float)]
        S = gsum_boot(P.gi.to_numpy(), G, cell, 20, vals)
        W = rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)
        out[nm] = (boot_tot(S, np.ones((1, G))), boot_tot(S, W))
    stats = [("plays/g", lambda t, c, G: t[..., c, 0] / G), ("pass rate", lambda t, c, G: t[..., c, 1] / np.maximum(t[..., c, 0], 1e-9)),
             ("ypp", lambda t, c, G: t[..., c, 2] / np.maximum(t[..., c, 3], 1e-9)), ("explosive", lambda t, c, G: t[..., c, 4] / np.maximum(t[..., c, 3], 1e-9)),
             ("ypa", lambda t, c, G: t[..., c, 7] / np.maximum(t[..., c, 8], 1e-9)), ("ypc", lambda t, c, G: t[..., c, 9] / np.maximum(t[..., c, 10], 1e-9)),
             ("4th go", lambda t, c, G: t[..., c, 6] / np.maximum(t[..., c, 5], 1e-9))]
    for q in range(4):
        say(f"  Q{q + 1}")
        for b in range(5):
            c = q * 5 + b
            row = []
            for lab, f in stats:
                pr = f(out["real"][0], c, Gr)[0]
                ps = f(out["sim"][0], c, Gs)[0]
                d = f(out["sim"][1], c, Gs) - f(out["real"][1], c, Gr)
                row.append(f"{lab} {pr:.3f}|{ps:.3f} {ci(d, ps - pr)}")
            say(f"    off {BANDS[b]:8s} (def {BANDS[4 - b]:8s}) " + " ; ".join(row))


def slope_table(Pr, Gr, Ps, Gs):
    say("## D. yards and mix per 7 points of offence deficit (OLS slope on clip(sd,+-21)/7, run/pass rows; positive = trailing offence does better), sim minus real, no team-strength control")
    groups = [("Q1", lambda P: P.qtr == 1), ("Q2", lambda P: P.qtr == 2), ("Q3", lambda P: P.qtr == 3), ("Q4", lambda P: P.qtr == 4), ("Q4 last5", lambda P: (P.qtr == 4) & (P.gsr <= 300))]
    mets = [("ypp", lambda P: P.clean, lambda P: P.yards), ("ypc", lambda P: P.clean & (P.code == 0), lambda P: P.yards), ("ypa", lambda P: P.clean & (P.code == 1), lambda P: P.yards),
            ("explosive", lambda P: P.clean, lambda P: (P.yards >= 20).astype(float)), ("pass rate", lambda P: P.rp, lambda P: (P.code == 1).astype(float))]
    res = {}
    for nm, P, G in (("real", Pr, Gr), ("sim", Ps, Gs)):
        W = rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)
        x = -np.clip(P.sd.to_numpy(), -21, 21) / 7.0
        for mn, mk, yf in mets:
            for gn, gf in groups:
                m = (mk(P) & gf(P)).to_numpy()
                xi, yi = x[m], np.asarray(yf(P))[m].astype(float)
                S = gsum_boot(P.gi.to_numpy()[m], G, np.zeros(m.sum(), dtype=int), 1, [np.ones(m.sum()), xi, yi, xi * yi, xi * xi])
                res[(nm, mn, gn)] = (boot_tot(S, np.ones((1, G)))[:, 0], boot_tot(S, W)[:, 0])

    def sl(t):
        n, sx, sy, sxy, sxx = (t[..., k] for k in range(5))
        return (sxy - sx * sy / n) / (sxx - sx * sx / n)

    for mn, _, _ in mets:
        row = []
        for gn, _ in groups:
            pr = sl(res[("real", mn, gn)][0])[0]
            ps = sl(res[("sim", mn, gn)][0])[0]
            d = sl(res[("sim", mn, gn)][1]) - sl(res[("real", mn, gn)][1])
            row.append(f"{gn} {pr:+.4f}|{ps:+.4f} {ci(d, ps - pr)}")
        say(f"  {mn}: " + " ; ".join(row))


def fine_check(ns, real, sim):
    say("## A2. same S/E split with finer state bands (3-point bins of offence lead at drive start, clipped at 21, per quarter); sum over the 6 pairs, 2x cov")
    NBND[0] = len(FINE_EDGES) - 1
    Dr, Gr = prep(ns, real, FINE_EDGES)
    Ds, Gs = prep(ns, sim, FINE_EDGES)
    res, cfS = pair_cov_table(Dr, Gr, Ds, Gs)
    r, s = res["real"], res["sim"]
    for key, lab in (("tot", "total"), ("S", "S"), ("E", "E")):
        d = s["bt"][key].sum(1) - r["bt"][key].sum(1)
        say(f"  {lab}: real {r['pt'][key].sum():+.2f} | sim {s['pt'][key].sum():+.2f} | sim-real {ci(d, s['pt'][key].sum() - r['pt'][key].sum())}")
    d = s["bt"]["S"].sum(1) - cfS["bt"].sum(1)
    say(f"  S rate-response (sim band means vs real band means on sim counts) {ci(d, s['pt']['S'].sum() - cfS['pt'].sum())}")
    NBND[0] = 5


def side_split(Dr, Gr, Ds, Gs):
    say("## A3. E residual by side: own = cov(early home margin, home drives' residual pts later), opp = cov(early home margin, minus away drives' residual pts later); 2x cov summed over pairs, per later quarter r")
    out = {}
    for nm, D, G in (("real", Dr, Gr), ("sim", Ds, Gs)):
        W = rng.multinomial(G, np.ones(G) / G, size=NB).astype(float)
        W = np.vstack([np.ones((1, G)), W])
        I, IBS, IS, abar = cross_terms(D, G, W)
        gi, qi, bi = D.gi.to_numpy(), D.qi.to_numpy(), D.bi.to_numpy()
        e = D.a.to_numpy()[None, :] - abar[:, qi * 5 + bi]
        home = (D.s.to_numpy() > 0)
        EH = np.zeros((len(W), G, 4))
        EA = np.zeros((len(W), G, 4))
        for b in range(len(W)):
            np.add.at(EH[b], (gi[home], qi[home]), e[b, home])
            np.add.at(EA[b], (gi[~home], qi[~home]), e[b, ~home])
        res = {}
        for r in range(1, 4):
            own = sum(2 * wcov(W, np.broadcast_to(I[:, q], W.shape), EH[:, :, r]) for q in range(r))
            opp = sum(2 * wcov(W, np.broadcast_to(I[:, q], W.shape), -EA[:, :, r]) for q in range(r))
            res[r] = (own, opp)
        out[nm] = res
    for r in range(1, 4):
        row = []
        for k, lab in ((0, "own side"), (1, "opposite side")):
            dr, ds = out["real"][r][k], out["sim"][r][k]
            row.append(f"{lab} real {dr[0]:+.2f} sim {ds[0]:+.2f} diff {ci(ds[1:] - dr[1:], ds[0] - dr[0])}")
        say(f"  later Q{r + 1} (early quarters before it): " + " ; ".join(row))
    for k, lab in ((0, "own side"), (1, "opposite side")):
        dr = sum(out["real"][r][k] for r in range(1, 4))
        ds = sum(out["sim"][r][k] for r in range(1, 4))
        say(f"  all later quarters {lab}: real {dr[0]:+.2f} sim {ds[0]:+.2f} diff {ci(ds[1:] - dr[1:], ds[0] - dr[0])}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ns = late.load_xq()
    real, sim = drives_all(ns)
    Dr, Gr = prep(ns, real)
    Ds, Gs = prep(ns, sim)
    say(f"label {LABEL} seeds {SEEDS}; real 2009-17 games {Gr}, sim games {Gs}; bootstrap {NB}; cells: adjusted points = drive net margin change minus leave-game-out team strength component (mod25e_xq.components)")
    res, cfS = pair_cov_table(Dr, Gr, Ds, Gs)
    report_cross(res, cfS)
    fine_check(ns, real, sim)
    side_split(Dr, Gr, Ds, Gs)
    drive_table(Dr, Gr, Ds, Gs)
    Pr, Gpr = play_frames(real_plays(ns), ns)
    Ps, Gps = play_frames(sim_plays(), ns)
    play_table(Pr, Gpr, Ps, Gps)
    slope_table(Pr, Gpr, Ps, Gps)
    say(f"looks counted (interval prints): {LOOKS[0]}")
    (OUTDIR / "catchup.txt").write_text("\n".join(OUT))


if __name__ == "__main__":
    main()
