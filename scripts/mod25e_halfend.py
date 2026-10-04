import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("CLK_LABEL", "crHpqokgndecsmf")
os.environ.setdefault("CLK_SEEDS", "11")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.argv = [sys.argv[0], "100"]
h2 = types.ModuleType("h2lib")
h2.__file__ = str(REPO / "scripts" / "mod25e_half2.py")
exec((REPO / "scripts" / "mod25e_half2.py").read_text().rsplit("\nmain()", 1)[0], h2.__dict__)
c2 = h2.c2
NB = 100
OUTDIR = REPO / "artifacts" / "mod25e3" / os.environ.get("HALFEND_OUT", "halfend")
ZONES = (("yl<=10", 0, 10), ("yl11-25", 10, 25), ("yl26-40", 25, 40), ("yl>40", 40, 200))
HBINS = (("hs0-5", -1, 5), ("hs6-15", 5, 15), ("hs16-30", 15, 30), ("hs31-60", 30, 60))


def sim_x():
    pool = pd.read_parquet(c2.ART.parent / "sim09" / "u4g" / "pool.parquet", columns=["yl", "gsr", "qtr", "el"])
    pyl, pgs, pq, pel = pool.yl.to_numpy(), pool.gsr.to_numpy(), pool.qtr.to_numpy(), pool.el.to_numpy()
    use = ["g", "down", "dist", "yl", "sd", "gsr", "qtr", "code", "po", "pdf", "flip", "offhome", "yards", "oto", "dto", "idx"]
    out, off = [], 0
    for sd in c2.SEEDS:
        for f in sorted((c2.ART / f"e5_{c2.LABEL}_s{sd}").glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < c2.BURN:
                continue
            d = pd.read_parquet(f, columns=use)
            d = d[d.qtr <= 4].sort_values("g", kind="stable").reset_index(drop=True)
            g, oh, q = d.g.to_numpy(), d.offhome.to_numpy(), d.qtr.to_numpy()
            sc = ((d.po > 0) | (d.pdf > 0)).to_numpy()
            new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2)) | sc[:-1]]
            d["dd"] = np.cumsum(new) + off
            off += int(new.sum()) + 1
            d = d[d.code.isin([0, 1, 2, 3, 4, 5])]
            c = d.code.astype(int).to_numpy()
            rp = c <= 1
            fl, po = d.flip.to_numpy(), d.po.to_numpy()
            ix = d.idx.astype(int).to_numpy()
            cls = np.select([c == 5, c == 4, (c == 1) & (d.yards.to_numpy() == 0) & (fl == 0) & (po == 0), c == 0, c == 1], [4, 5, 0, 2, 3], -1)
            D = pd.DataFrame({"gk": f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str).to_numpy(), "d": d.dd.to_numpy(), "qtr": d.qtr.to_numpy(), "gsr": d.gsr.to_numpy(), "sd": d.sd.to_numpy(),
                              "down": d.down.to_numpy(), "dist": d.dist.to_numpy(), "yl": d.yl.to_numpy(), "kind": c, "to": (rp & (fl == 1) & (po == 0) & (d.down <= 3).to_numpy()).astype(int),
                              "td": (rp & (po >= 6)).astype(int), "fgm": ((c == 3) & (po == 3)).astype(int), "cls": cls,
                              "yd": d.yards.to_numpy(), "tou": (d.oto + d.dto).to_numpy(),
                              "pyl": pyl[ix], "phs": np.where(pq[ix] == 2, pgs[ix] - 1800.0, np.nan), "pq": pq[ix], "pel": pel[ix]})
            out.append(D[D.down.notna()].reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


def decorate(P, sim):
    P, D = c2.frames(P, np.array([25.0, 50.0, 75.0]))
    P["hs"] = np.where(P.qtr == 2, P.gsr - 1800.0, np.nan)
    gp = P.groupby("d", sort=False)
    P["pk"] = gp.kind.shift(1)
    P["one"] = 1
    P["k_fg"] = (P.kind == 3).astype(int)
    P["k_spike"] = (P.kind == 5).astype(int)
    P["k_rp"] = (P.kind <= 1).astype(int)
    P["k_kneel"] = (P.kind == 4).astype(int)
    P["fg_after_spike"] = P.k_fg * (P.pk == 5)
    P["fg_after_rp"] = P.k_fg * (P.pk <= 1)
    P["fg_first"] = P.k_fg * P.pk.isna()
    P["td_rp"] = P.td * P.k_rp
    if sim:
        P["srcgap"] = (P.pyl - P.yl).abs()
        P["srcdeep"] = (P.pyl - P.yl > 5).astype(int)
        P["srcmore"] = (P.phs > P.hs + 30).astype(int)
        P["srcq"] = (P.pq != 2).astype(int)
        P["srcel_over"] = (P.pel > P.hs).astype(int)
        P["srcexact"] = (P.pyl.round() == P.yl.round()).astype(int)
    return P, D


def drv(P, D):
    L = P[P.d.isin(set(D[(D.lh == 1) & (D.qd == 2)].index))].copy()
    L["rz"] = (L.yl <= 25).astype(int)
    L["rp15"] = L.k_rp * (L.hs <= 15)
    L["rp5"] = L.k_rp * (L.hs <= 5)
    L["rp30"] = L.k_rp * (L.hs <= 30)
    L["hsz"] = L.hs.where(L.yl <= 25)
    g = L.groupby("d").agg(gk=("gk", "first"), rz=("rz", "max"), ent=("hsz", "max"), lhs=("hs", "last"), rp15=("rp15", "sum"), rp5=("rp5", "sum"), rp30=("rp30", "sum"), spk=("k_spike", "sum"), fg=("k_fg", "sum"), td=("td", "max"), kn=("k_kneel", "sum"))
    g = g[g.rz == 1]
    for c in ("punt", "turn", "downs", "other"):
        g[c] = D.loc[g.index, c].to_numpy()
    g = g.reset_index(drop=True)
    g["tm"] = g.other * (g.fg == 0) * (g.td == 0)
    g["tm_under5"] = g.tm * (g.lhs <= 5)
    g["tm_under15"] = g.tm * (g.lhs <= 15)
    g["one"] = 1
    g["fgd"] = (g.fg > 0).astype(int)
    return g


def main():
    rng = np.random.default_rng(0)
    R, S = h2.real_plays(), sim_x()
    PR, DR = decorate(R, False)
    PS, DS = decorate(S, True)
    rg, sg = np.sort(R.gk.unique()), np.sort(S.gk.unique())
    c2.NB = NB
    C = c2.Cmp(rng, rg, sg)
    say = h2.say
    say(f"label {c2.LABEL} seeds {c2.SEEDS} real games {len(rg)} sim games {len(sg)} NB {NB}; Q2 plays, all drives unless stated")
    q2r = PR[(PR.qtr == 2) & PR.hs.notna()]
    q2s = PS[(PS.qtr == 2) & PS.hs.notna()]
    say("== 1 downs 1-3 play shares by seconds left x field position (Q2)")
    for hn, hl, hh in HBINS:
        for zn, zl, zh in ZONES:
            def m(d):
                return (d.hs > hl) & (d.hs <= hh) & (d.yl > zl) & (d.yl <= zh) & (d.down <= 3)
            a, b = q2r[m(q2r)], q2s[m(q2s)]
            if len(a) < 30 or len(b) < 30:
                say(f"-- {hn} {zn} n real {len(a)} sim {len(b)} (skipped <30)")
                continue
            say(f"-- {hn} {zn} n real {len(a)} sim {len(b)}")
            for met in ("k_fg", "k_spike", "k_rp", "k_kneel", "td_rp"):
                C.row(f"  {met}", a, met, "one", b, met, "one")
    say("== 2 FG on downs 1-3 at yl<=40, hs<=60: what preceded it")
    a = q2r[(q2r.hs <= 60) & (q2r.yl <= 40) & (q2r.down <= 3)]
    b = q2s[(q2s.hs <= 60) & (q2s.yl <= 40) & (q2s.down <= 3)]
    for met in ("k_fg", "fg_after_spike", "fg_after_rp", "fg_first", "k_spike"):
        C.row(f"  {met} per down1-3 play", a, met, "one", b, met, "one")
    say("== 3 drive level: Q2 last drives reaching yl<=25")
    gr, gs_ = drv(PR, DR), drv(PS, DS)
    say(f"-- trips real {len(gr)} sim {len(gs_)}")
    for met in ("rp30", "rp15", "rp5", "spk", "fgd", "td", "kn"):
        C.row(f"  {met} per trip", gr, met, "one", gs_, met, "one")
    say("== 4 elapsed on non-terminal run/pass plays at yl<=25 (secs to next snap)")
    for hn, hl, hh in (("hs<=30", -1, 30), ("hs31-120", 30, 120), ("hs121-300", 120, 300)):
        def m2(d):
            return (d.hs > hl) & (d.hs <= hh) & (d.yl <= 25) & (d.k_rp == 1) & d.secs.notna()
        a, b = q2r[m2(q2r)], q2s[m2(q2s)]
        say(f"-- {hn} n real {len(a)} sim {len(b)}")
        C.row("  secs", a, "secs", "one", b, "secs", "one", fmt="{:.2f}")
    say("== 5 sim only: source row of run/pass plays at yl<=10 down<=3 (state hs vs source hs, state yl vs source yl)")
    for hn, hl, hh in (("Q2 hs<=15", -1, 15), ("Q2 hs16-60", 15, 60), ("Q2 hs61-180", 60, 180), ("Q2 hs>180", 180, 900)):
        s = q2s[(q2s.hs > hl) & (q2s.hs <= hh) & (q2s.yl <= 10) & (q2s.down <= 3) & (q2s.k_rp == 1)]
        t = s[s.td == 1]
        say(f"-- {hn} plays {len(s)} TD plays {len(t)}")
        for nm, x in (("all", s), ("TD", t)):
            if len(x) < 5:
                continue
            say(f"   {nm:3s} source yl mean {x.pyl.mean():.2f} state yl {x.yl.mean():.2f} |gap| {x.srcgap.mean():.2f} exact-yl share {x.srcexact.mean():.3f} source >5yd deeper {x.srcdeep.mean():.3f} source hs>state+30 {x.srcmore.mean():.3f} source not Q2 {x.srcq.mean():.3f} source el>state hs {x.srcel_over.mean():.3f}")
    say("== 6 sim only reference: exact-yl share, yl<=10 down<=3 run/pass outside the half end")
    for nm, x in (("Q1/Q3/Q4 all", PS[(PS.qtr != 2) & (PS.yl <= 10) & (PS.down <= 3) & (PS.k_rp == 1)]), ("Q2 hs>180", q2s[(q2s.hs > 180) & (q2s.yl <= 10) & (q2s.down <= 3) & (q2s.k_rp == 1)])):
        say(f"   {nm:14s} n {len(x)} exact-yl {x.srcexact.mean():.3f} |gap| {x.srcgap.mean():.2f} TD share of plays {x.td.mean():.3f}")
    say("== 7 trips by seconds left when the drive first has a snap at yl<=25")
    for hn, hl, hh in (("ent<=30", -1, 30), ("ent31-60", 30, 60), ("ent61-120", 60, 120), ("ent121-300", 120, 300), ("ent>300", 300, 2000)):
        a, b = gr[(gr.ent > hl) & (gr.ent <= hh)], gs_[(gs_.ent > hl) & (gs_.ent <= hh)]
        say(f"-- {hn} trips real {len(a)} sim {len(b)} share real {len(a) / len(gr):.3f} sim {len(b) / len(gs_):.3f}")
        if len(a) >= 20 and len(b) >= 20:
            for met in ("td", "fgd", "rp30"):
                C.row(f"  {met} per trip", a, met, "one", b, met, "one")
    say("== 7b how trips end (drive level, trips with a snap at yl<=25)")
    for met in ("td", "fgd", "turn", "downs", "punt", "tm", "tm_under5", "tm_under15"):
        C.row(f"  {met} per trip", gr, met, "one", gs_, met, "one")
    for hn, hl, hh in (("ent<=30", -1, 30), ("ent31-120", 30, 120)):
        a, b = gr[(gr.ent > hl) & (gr.ent <= hh)], gs_[(gs_.ent > hl) & (gs_.ent <= hh)]
        say(f"-- {hn} trips real {len(a)} sim {len(b)}")
        for met in ("turn", "downs", "tm", "tm_under5"):
            C.row(f"  {met} per trip", a, met, "one", b, met, "one")
    say("== 8 elapsed on non-terminal run/pass plays, hs31-300, by zone and play type")
    for zn, zl, zh in (("yl<=25", 0, 25), ("yl26-40", 25, 40), ("yl>40", 40, 200)):
        for kn_, kv in (("run", 0), ("pass", 1)):
            def m3(d):
                return (d.hs > 30) & (d.hs <= 300) & (d.yl > zl) & (d.yl <= zh) & (d.kind == kv) & d.secs.notna()
            a, b = q2r[m3(q2r)], q2s[m3(q2s)]
            say(f"-- {zn} {kn_} n real {len(a)} sim {len(b)}")
            C.row("  secs", a, "secs", "one", b, "secs", "one", fmt="{:.2f}")
    say("== 9 elapsed by outcome class, non-terminal run/pass plays, Q2 hs31-300 (cls 0 incomplete, 2 run, 3 completed pass)")
    for zn, zl, zh in (("yl<=25", 0, 25), ("all yl", 0, 200)):
        def m4(d):
            return (d.hs > 30) & (d.hs <= 300) & (d.yl > zl) & (d.yl <= zh) & (d.k_rp == 1) & d.secs.notna()
        a, b = q2r[m4(q2r)], q2s[m4(q2s)]
        say(f"-- {zn} n real {len(a)} sim {len(b)}")
        C.row("  secs all", a, "secs", "one", b, "secs", "one", fmt="{:.2f}")
        for cv in (0, 2, 3):
            a2 = a.assign(mc=(a.cls == cv).astype(int), ms=a.secs * (a.cls == cv))
            b2 = b.assign(mc=(b.cls == cv).astype(int), ms=b.secs * (b.cls == cv))
            C.row(f"  share cls{cv}", a2, "mc", "one", b2, "mc", "one")
            C.row(f"  secs cls{cv}", a2, "ms", "mc", b2, "ms", "mc", fmt="{:.2f}")
    say("== 10 elapsed of in-bounds run and completed-pass plays by seconds left (all yl, non-terminal); elapsed redraw window is hs<=120, decision window hs<=180")
    for hn, hl, hh in (("hs31-60", 30, 60), ("hs61-120", 60, 120), ("hs121-180", 120, 180), ("hs181-240", 180, 240), ("hs241-300", 240, 300), ("hs301-600", 300, 600)):
        for cv in (2, 3):
            def m5(d):
                return (d.hs > hl) & (d.hs <= hh) & (d.cls == cv) & d.secs.notna()
            a, b = q2r[m5(q2r)], q2s[m5(q2s)]
            C.row(f"  {hn} cls{cv} secs", a, "secs", "one", b, "secs", "one", fmt="{:.2f}")
    say(f"looks {c2.LOOKS[0]}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "halfend.txt").write_text("\n".join(c2.OUT) + "\n")


main()
