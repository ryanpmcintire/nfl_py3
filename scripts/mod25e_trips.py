import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("CLK_LABEL", "crHpqokgndecsmfw")
os.environ.setdefault("CLK_SEEDS", "11")
os.environ.setdefault("HALFEND_OUT", "trips")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
he = types.ModuleType("halfend_lib")
he.__file__ = str(REPO / "scripts" / "mod25e_halfend.py")
exec((REPO / "scripts" / "mod25e_halfend.py").read_text().rsplit("\nmain()", 1)[0], he.__dict__)
h2, c2 = he.h2, he.c2
say = h2.say
EB = (("ent<=5", -1, 5), ("ent6-10", 5, 10), ("ent11-30", 10, 30), ("ent31-60", 30, 60), ("ent61-120", 60, 120), ("ent121-300", 120, 300), ("ent>300", 300, 5000))
YB = (("eyl<=10", 0, 10), ("eyl11-25", 10, 25))
FB = (("fg hs<=5", -1, 5), ("fg hs6-15", 5, 15), ("fg hs16-30", 15, 30), ("fg hs31-60", 30, 60), ("fg hs>60", 60, 5000))


def build(P, D, last=True):
    L = P[P.d.isin(set(D[((D.lh == 1) | (not last)) & (D.qd == 2)].index))].copy()
    L = L.sort_values("d", kind="stable").reset_index(drop=True)
    L["rz"] = (L.yl <= 25).astype(int)
    L["rzc"] = L.groupby("d").rz.cumsum()
    L["nd"] = L.groupby("d").down.shift(-1)
    L["fd"] = ((L.nd == 1) & (L.k_rp == 1)).astype(int)
    T = L[L.rzc >= 1].copy()
    T["one"] = 1
    T["inc"] = (T.cls == 0).astype(int)
    T["run"] = ((T.cls == 2) & (T.td == 0)).astype(int)
    T["comp"] = ((T.cls == 3) & (T.td == 0)).astype(int)
    T["fdp"] = T.fd * (T.td == 0)
    T["stop"] = (T.k_rp.astype(bool) & T.cls.isin([2, 3]) & (T.secs <= 12) & (T.td == 0)).astype(int)
    T["gl"] = ((T.yl <= 10) & T.k_rp.astype(bool)).astype(int)
    T["fgh"] = np.where(T.k_fg == 1, T.hs, np.nan)
    T["fgdown"] = np.where(T.k_fg == 1, T.down, np.nan)
    g = T.groupby("d")
    fr = g.first()
    G = pd.DataFrame({"gk": fr.gk, "ent": fr.hs, "eyl": fr.yl, "edown": fr.down, "edist": fr.dist, "etou": fr.tou, "esd": fr.sd})
    G["np"] = g.size()
    G["nrp"] = g.k_rp.sum()
    G["spk"] = g.k_spike.sum()
    G["fgd"] = (g.k_fg.sum() > 0).astype(int)
    G["td"] = (g.td.max() > 0).astype(int)
    G["gl"] = (g.gl.sum() > 0).astype(int)
    G["tdgl"] = G.td * G.gl
    G["fgh"] = g.fgh.max()
    G["fgdown"] = g.fgdown.max()
    G["lhs"] = g.hs.last()
    G["one"] = 1
    G["fg_late"] = G.fgd * (G.fgh > 5)
    G["fg_d4"] = G.fgd * (G.fgdown == 4)
    G["fg_d13"] = G.fgd * (G.fgdown <= 3)
    for n, lo, hi in FB:
        G[n] = G.fgd * ((G.fgh > lo) & (G.fgh <= hi))
    G["ebin"] = -1
    for i, (n, lo, hi) in enumerate(EB):
        G.loc[(G.ent > lo) & (G.ent <= hi), "ebin"] = i
    G["ybin"] = (G.eyl > 10).astype(int)
    G["cell"] = G.ebin * 2 + G.ybin
    T = T.join(G[["ebin", "ybin", "cell"]], on="d")
    return G.reset_index(drop=True), T.reset_index(drop=True)


def cells(G, games, num, den, nc):
    X = np.zeros((len(games), nc))
    N = np.zeros((len(games), nc))
    ix = pd.Series(np.arange(len(games)), index=games)
    gi = ix.reindex(G.gk).to_numpy()
    ok = ~np.isnan(gi) & (G.cell.to_numpy() >= 0)
    np.add.at(X, (gi[ok].astype(int), G.cell.to_numpy()[ok]), G[num].to_numpy()[ok])
    np.add.at(N, (gi[ok].astype(int), G.cell.to_numpy()[ok]), G[den].to_numpy()[ok])
    return X, N


def shares(xr, nr, xs, ns):
    wr, ws = nr / max(nr.sum(), 1e-12), ns / max(ns.sum(), 1e-12)
    rr = np.divide(xr, nr, out=np.zeros_like(xr), where=nr > 0)
    rs = np.divide(xs, ns, out=np.zeros_like(xs), where=ns > 0)
    rr = np.where(nr > 0, rr, rs)
    rs = np.where(ns > 0, rs, rr)
    tot = xs.sum() / max(ns.sum(), 1e-12) - xr.sum() / max(nr.sum(), 1e-12)
    mix = ((ws - wr) * (rr + rs) / 2).sum()
    rate = ((wr + ws) / 2 * (rs - rr)).sum()
    return np.array([tot, mix, rate])


def shift(rng, Gr, Gs, rg, sg, num, den, label, nc=len(EB) * 2):
    c2.LOOKS[0] += 1
    xr, nr = cells(Gr, rg, num, den, nc)
    xs, ns = cells(Gs, sg, num, den, nc)
    est = shares(xr.sum(0), nr.sum(0), xs.sum(0), ns.sum(0))
    bs = np.empty((c2.NB, 3))
    for b in range(c2.NB):
        i, j = rng.integers(0, len(rg), len(rg)), rng.integers(0, len(sg), len(sg))
        bs[b] = shares(xr[i].sum(0), nr[i].sum(0), xs[j].sum(0), ns[j].sum(0))
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    pp = (bs > 0).mean(0)

    def f(k):
        return f"{est[k]:+.3f} [{lo[k]:+.3f},{hi[k]:+.3f}] pp {pp[k]:.2f}"

    say(f"{label:30s} total {f(0)} | mix {f(1)} | rate {f(2)}")


def main():
    rng = np.random.default_rng(0)
    R, S = h2.real_plays(), he.sim_x()
    PR, DR = he.decorate(R, False)
    PS, DS = he.decorate(S, True)
    rg, sg = np.sort(R.gk.unique()), np.sort(S.gk.unique())
    c2.NB = he.NB
    C = c2.Cmp(rng, rg, sg)
    GR, TR = build(PR, DR)
    GS, TS = build(PS, DS)
    say(f"label {c2.LABEL} seeds {c2.SEEDS} real games {len(rg)} sim games {len(sg)} NB {c2.NB}; trips = Q2 last drive with a snap at yl<=25; plays from the first such snap on")
    say(f"trips real {len(GR)} sim {len(GS)}")
    say("== 1 entry state (sim vs real)")
    for n, lo, hi in EB:
        for G in (GR, GS):
            G["b_" + n] = ((G.ent > lo) & (G.ent <= hi)).astype(int)
        C.row(f"  share {n}", GR, "b_" + n, "one", GS, "b_" + n, "one")
    for n, lo, hi in YB:
        for G in (GR, GS):
            G["y_" + n] = ((G.eyl > lo) & (G.eyl <= hi)).astype(int)
        C.row(f"  share {n}", GR, "y_" + n, "one", GS, "y_" + n, "one")
    for col in ("eyl", "edown", "edist", "etou", "esd", "ent"):
        C.row(f"  mean {col}", GR, col, "one", GS, col, "one", fmt="{:.2f}")
    for d in (1, 2, 3, 4):
        for G in (GR, GS):
            G[f"ed{d}"] = (G.edown == d).astype(int)
        C.row(f"  share entry down {d}", GR, f"ed{d}", "one", GS, f"ed{d}", "one")
    for G in (GR, GS):
        G["tou0"] = (G.etou == 0).astype(int)
        G["tou4"] = (G.etou >= 4).astype(int)
    C.row("  share entry no timeouts left", GR, "tou0", "one", GS, "tou0", "one")
    C.row("  share entry 4+ timeouts left", GR, "tou4", "one", GS, "tou4", "one")
    say("== 2 trip outcomes and plays run per trip, by entry-seconds band")
    for n, lo, hi in (("all", -1, 5000),) + EB:
        a, b = GR[(GR.ent > lo) & (GR.ent <= hi)], GS[(GS.ent > lo) & (GS.ent <= hi)]
        say(f"-- {n} trips real {len(a)} sim {len(b)}")
        if len(a) < 15 or len(b) < 15:
            continue
        for met in ("td", "fgd", "np", "nrp", "spk", "gl", "tdgl"):
            C.row(f"  {met} per trip", a, met, "one", b, met, "one", fmt="{:.3f}")
        C.row("  td per goal-line trip", a, "tdgl", "gl", b, "tdgl", "gl")
    say("== 3 per-play outcome mix inside trips (plays from entry on), by entry band")
    for n, lo, hi in (("all", -1, 5000),) + EB:
        ta, tb = TR[TR.ebin >= 0], TS[TS.ebin >= 0]
        if n != "all":
            i = [e[0] for e in EB].index(n)
            ta, tb = ta[ta.ebin == i], tb[tb.ebin == i]
        say(f"-- {n} plays real {len(ta)} sim {len(tb)}")
        if len(ta) < 30 or len(tb) < 30:
            continue
        for met in ("td", "inc", "run", "comp", "fdp", "stop", "k_fg", "k_spike", "k_kneel"):
            C.row(f"  {met} per play", ta, met, "one", tb, met, "one")
        rp_a, rp_b = ta[ta.k_rp == 1], tb[tb.k_rp == 1]
        C.row("  td per run/pass play", rp_a, "td", "one", rp_b, "td", "one")
        C.row("  secs per run/pass play", rp_a[rp_a.secs.notna()], "secs", "one", rp_b[rp_b.secs.notna()], "secs", "one", fmt="{:.2f}")
    say("== 3b per-play outcome mix at yl<=10 inside trips, by seconds left at the play")
    for hn, hl, hh in he.HBINS + (("hs61-120", 60, 120), ("hs121-300", 120, 300)):
        ta = TR[(TR.hs > hl) & (TR.hs <= hh) & (TR.yl <= 10) & (TR.down <= 3)]
        tb = TS[(TS.hs > hl) & (TS.hs <= hh) & (TS.yl <= 10) & (TS.down <= 3)]
        say(f"-- {hn} yl<=10 down1-3 plays real {len(ta)} sim {len(tb)}")
        if len(ta) < 30 or len(tb) < 30:
            continue
        for met in ("td", "inc", "run", "comp", "fdp", "stop", "k_fg", "k_spike"):
            C.row(f"  {met} per play", ta, met, "one", tb, met, "one")
    say("== 4 FG attempts: when taken (hs and down), per trip")
    for met in ("fgd", "fg_d13", "fg_d4", "fg_late") + tuple(n for n, _, _ in FB):
        C.row(f"  {met} per trip", GR, met, "one", GS, met, "one")
    C.row("  mean hs at FG attempt", GR[GR.fgd == 1], "fgh", "one", GS[GS.fgd == 1], "fgh", "one", fmt="{:.2f}")
    say("== 5 trips entered with <=10 s (structure)")
    for G in (GR, GS):
        G["short10"] = (G.ent <= 10).astype(int)
        G["short10_fg"] = G.short10 * G.fgd
        G["short10_td"] = G.short10 * G.td
        G["short10_none"] = G.short10 * (1 - G.fgd) * (1 - G.td)
    C.row("  share of trips ent<=10", GR, "short10", "one", GS, "short10", "one")
    C.row("  FG among ent<=10", GR, "short10_fg", "short10", GS, "short10_fg", "short10")
    C.row("  TD among ent<=10", GR, "short10_td", "short10", GS, "short10_td", "short10")
    C.row("  neither among ent<=10", GR, "short10_none", "short10", GS, "short10_none", "short10")
    C.row("  contribution to TD per trip (ent<=10)", GR, "short10_td", "one", GS, "short10_td", "one")
    C.row("  contribution to FG per trip (ent<=10)", GR, "short10_fg", "one", GS, "short10_fg", "one")
    say("== 6 shift-share, cells = entry-seconds band (7) x entry yl band (2); mix = entry state, rate = within-cell; sums to total")
    for num, den, lab in (("td", "one", "TD per trip"), ("fgd", "one", "FG attempt per trip"), ("nrp", "one", "run/pass plays per trip"), ("gl", "one", "reach yl<=10 per trip"), ("tdgl", "gl", "TD per goal-line trip")):
        shift(rng, GR, GS, rg, sg, num, den, lab)
    say("== 7 chain: TD per trip = P(reach yl<=10) x P(TD | reached) + P(TD without a yl<=10 snap)")
    C.row("  reach yl<=10 per trip", GR, "gl", "one", GS, "gl", "one")
    C.row("  TD per goal-line trip", GR, "tdgl", "gl", GS, "tdgl", "gl")
    GR["m"], GS["m"] = GR.td * (1 - GR.gl), GS.td * (1 - GS.gl)
    GR["nr"], GS["nr"] = 1 - GR.gl, 1 - GS.gl
    C.row("  TD among trips never at yl<=10", GR, "m", "nr", GS, "m", "nr")
    say("== 8 selection check: a TD with time left hands the opponent a drive, so the TD drive is not the last drive of the half")
    for nm, D_, R_ in (("real", DR, rg), ("sim", DS, sg)):
        x = D_[(D_.qd == 2) & (D_.td == 1)].copy()
        nx = D_.shift(-1)
        x["fol"] = ((nx.gk == D_.gk) & (nx.qd == 2)).reindex(x.index).astype(int)
        x["hs_end"] = x.hl
        for n, lo, hi in (("hl<=15", -1, 15), ("hl16-60", 15, 60), ("hl61-120", 60, 120), ("hl121-300", 120, 300)):
            y = x[(x.hs_end > lo) & (x.hs_end <= hi)]
            gap = (y.hs_end - nx.hs0.reindex(y.index))[y.fol == 1]
            say(f"  {nm} Q2 TD drives ending {n}: n {len(y)} followed by another Q2 drive {y.fol.mean() if len(y) else float('nan'):.3f}; TD snap to next snap {gap.mean():.1f} s (median {gap.median():.1f}), next snap within 5 s of the TD snap {(gap < 5).mean():.3f}")
    GRa, TRa = build(PR, DR, last=False)
    GSa, TSa = build(PS, DS, last=False)
    say(f"-- all Q2 drives with a snap at yl<=25 entered with hs<=120: real {int((GRa.ent <= 120).sum())} sim {int((GSa.ent <= 120).sum())}")
    for n, lo, hi in (("ent<=30", -1, 30), ("ent31-120", 30, 120)):
        a, b = GRa[(GRa.ent > lo) & (GRa.ent <= hi)], GSa[(GSa.ent > lo) & (GSa.ent <= hi)]
        say(f"-- {n} all-drive trips real {len(a)} sim {len(b)}")
        for met in ("td", "fgd", "nrp"):
            C.row(f"  {met} per trip", a, met, "one", b, met, "one")
    say(f"looks {c2.LOOKS[0]}")
    out = Path(he.OUTDIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trips.txt").write_text("\n".join(c2.OUT) + "\n")


main()
