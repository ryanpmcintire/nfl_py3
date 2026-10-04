import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import types

c2 = types.ModuleType("mod25e_clock2_lib")
c2.__file__ = str(REPO / "scripts" / "mod25e_clock2.py")
exec((REPO / "scripts" / "mod25e_clock2.py").read_text().rsplit("\nmain()", 1)[0], c2.__dict__)

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUTDIR = REPO / "artifacts" / "mod25e3" / "half2"
OUTNAME = sys.argv[2] if len(sys.argv) > 2 else "half2.txt"
OUT = c2.OUT


def say(s=""):
    print(s)
    OUT.append(s)


def real_plays():
    cols = ["game_id", "play_id", "season_type", "posteam", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential", "down", "ydstogo", "qb_kneel", "qb_spike", "interception", "fumble_lost", "touchdown", "posteam_score", "posteam_score_post", "complete_pass", "yards_gained", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "fixed_drive_result"]
    out = []
    for s in c2.POOL:
        p = pd.read_parquet(c2.PBP / f"season={s}" / "plays.parquet", columns=cols)
        p = p[(p.season_type == "REG") & p.posteam.notna() & p.score_differential.notna() & (p.qtr <= 4)]
        p = p.sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
        ko = (p.play_type == "kickoff").to_numpy()
        p["kocum"] = np.cumsum(ko)
        p = p[~ko & p.play_type.isin(["run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike"])].copy()
        g, pos, kc, q = p.game_id.to_numpy(), p.posteam.to_numpy(), p.kocum.to_numpy(), p.qtr.to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (pos[1:] != pos[:-1]) | (kc[1:] != kc[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
        spike, kneel = (p.qb_spike == 1).to_numpy(), (p.qb_kneel == 1).to_numpy()
        pt = p.play_type.to_numpy()
        inc = ((p.complete_pass == 0) & (p.yards_gained.fillna(0) == 0)).to_numpy()
        kind = np.select([spike, kneel, pt == "run", pt == "pass", pt == "punt", pt == "field_goal"], [5, 4, 0, 1, 2, 3], -1)
        rpm = (kind >= 0) & (kind <= 1)
        to = rpm & (p.down <= 3).to_numpy() & ((p.interception == 1) | (p.fumble_lost == 1)).to_numpy()
        td = rpm & (p.touchdown == 1).to_numpy() & (p.posteam_score_post > p.posteam_score).to_numpy()
        cls = np.select([kind == 5, kind == 4, (kind == 1) & inc, kind == 0, kind == 1], [4, 5, 0, 2, 3], -1)
        D = pd.DataFrame({"gk": str(s) + "_" + p.game_id.to_numpy(), "d": np.cumsum(new) + (s - c2.POOL[0]) * 10**7, "qtr": q, "gsr": p.game_seconds_remaining.to_numpy(),
                          "sd": p.score_differential.to_numpy(), "down": p.down.to_numpy(), "dist": p.ydstogo.to_numpy(), "yl": p.yardline_100.to_numpy(), "kind": kind, "to": to.astype(int), "td": td.astype(int),
                          "fgm": ((pt == "field_goal") & (p.fixed_drive_result == "Field goal").to_numpy()).astype(int), "cls": cls,
                          "yd": p.yards_gained.fillna(0).to_numpy(), "tou": (p.posteam_timeouts_remaining + p.defteam_timeouts_remaining).to_numpy()})
        out.append(D[D.down.notna()].reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


def sim_plays():
    out = []
    use = ["g", "down", "dist", "yl", "sd", "gsr", "qtr", "code", "po", "pdf", "flip", "offhome", "yards", "oto", "dto"]
    off = 0
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
            cls = np.select([c == 5, c == 4, (c == 1) & (d.yards.to_numpy() == 0) & (fl == 0) & (po == 0), c == 0, c == 1], [4, 5, 0, 2, 3], -1)
            D = pd.DataFrame({"gk": f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str).to_numpy(), "d": d.dd.to_numpy(), "qtr": d.qtr.to_numpy(), "gsr": d.gsr.to_numpy(), "sd": d.sd.to_numpy(),
                              "down": d.down.to_numpy(), "dist": d.dist.to_numpy(), "yl": d.yl.to_numpy(), "kind": c, "to": (rp & (fl == 1) & (po == 0) & (d.down <= 3).to_numpy()).astype(int),
                              "td": (rp & (po >= 6)).astype(int), "fgm": ((c == 3) & (po == 3)).astype(int), "cls": cls,
                              "yd": d.yards.to_numpy(), "tou": (d.oto + d.dto).to_numpy()})
            out.append(D[D.down.notna()].reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


def prep(P, cuts):
    P, D = c2.frames(P, cuts)
    q2 = D[(D.lh == 1) & (D.qd == 2)].copy()
    q2["hb"] = np.digitize(q2.hs0, [30, 60, 120, 240, 600])
    L = P[P.d.isin(set(q2.index))].copy()
    L["hs"] = L.gsr - 1800.0
    for k, nm in enumerate(("run", "pass", "punt", "fg", "kneel", "spike")):
        L["k_" + nm] = (L.kind == k).astype(int)
    L["rp"] = L.kind.isin([0, 1]).astype(int)
    L["d4"] = (L.down == 4).astype(int)
    L["d123"] = 1 - L.d4
    L["d4_punt"], L["d4_fg"], L["d4_go"] = L.d4 * L.k_punt, L.d4 * L.k_fg, L.d4 * L.rp
    L["yd_pass"] = L.yd * L.k_pass
    L["fg_len"] = (L.yl + 17.0) * L.k_fg
    L["td_pass"], L["td_run"] = L.td * L.k_pass, L.td * L.k_run
    L["pass_deep"] = L.k_pass * (L.dist >= 15).astype(int)
    return L, q2


def post_strat(Dr, Ds, rng, cols, outs, nb):
    rg, sg = np.sort(Dr.gk.unique()), np.sort(Ds.gk.unique())
    res = {}
    for o in outs:
        def cellsums(D):
            key = D[cols].astype(str).agg("|".join, axis=1)
            T = pd.DataFrame({"gk": D.gk.to_numpy(), "cell": key.to_numpy(), "n": 1, "v": D[o].to_numpy()})
            return T.groupby(["gk", "cell"]).agg(n=("n", "sum"), v=("v", "sum")).reset_index()
        A, B = cellsums(Dr), cellsums(Ds)
        cells = sorted(set(A.cell) | set(B.cell))
        ci = {c: i for i, c in enumerate(cells)}

        def mat(T, games):
            gi = {g: i for i, g in enumerate(games)}
            N = np.zeros((len(games), len(cells)))
            V = np.zeros_like(N)
            N[T.gk.map(gi).to_numpy(), T.cell.map(ci).to_numpy()] = T.n.to_numpy()
            V[T.gk.map(gi).to_numpy(), T.cell.map(ci).to_numpy()] = T.v.to_numpy()
            return N, V

        Nr, Vr = mat(A, rg)
        Ns, Vs = mat(B, sg)

        def est(ir, is_):
            nr, vr, ns, vs = Nr[ir].sum(0), Vr[ir].sum(0), Ns[is_].sum(0), Vs[is_].sum(0)
            rr = vr / np.maximum(nr, 1)
            w = ns / ns.sum()
            ok = nr > 0
            real, sim = vr.sum() / nr.sum(), vs.sum() / ns.sum()
            realw = (w[ok] * rr[ok]).sum() / w[ok].sum()
            return np.array([sim - real, realw - real, sim - realw])

        e0 = est(np.arange(len(rg)), np.arange(len(sg)))
        bs = np.array([est(rng.integers(0, len(rg), len(rg)), rng.integers(0, len(sg), len(sg))) for _ in range(nb)])
        res[o] = (e0, np.quantile(bs, 0.025, axis=0), np.quantile(bs, 0.975, axis=0), (bs > 0).mean(0))
    return res


def main():
    rng = np.random.default_rng(0)
    R, S = real_plays(), sim_plays()
    cuts = np.quantile(R[R.down == 1].yl, [0.25, 0.5, 0.75])
    LR, QR = prep(R, cuts)
    LS, QS = prep(S, cuts)
    rg, sg = np.sort(R.gk.unique()), np.sort(S.gk.unique())
    c2.NB = NB
    C = c2.Cmp(rng, rg, sg)
    QR["one"], QS["one"], LR["one"], LS["one"] = 1, 1, 1, 1
    say(f"label {c2.LABEL} seeds {c2.SEEDS} real games {len(rg)} sim games {len(sg)} NB {NB}; last drive of Q2 only; drives real {len(QR)} sim {len(QS)}")
    say("== 1 last-drive-of-Q2 end type (drive level)")
    for met in ("td", "punt", "fg", "fgm", "turn", "downs", "other"):
        C.row(f" {met}", QR, met, "one", QS, met, "one")
    say("== 2 start-state mix vs within-state (real rates re-weighted to sim start mix; total = sim-real, mix = real-on-sim-mix minus real, within = sim minus real-on-sim-mix)")
    for nm, cl in (("hs0 bin", ["hb"]), ("hs0 bin x zone", ["hb", "z"]), ("hs0 bin x zone x score sign", ["hb", "z", "ss"])):
        r = post_strat(QR, QS, rng, cl, ("td", "punt", "fg", "fgm"), NB)
        for o, (e, lo, hi, pp) in r.items():
            say(f" {nm:28s} {o:5s} total {e[0]:+.3f} [{lo[0]:+.3f},{hi[0]:+.3f}] pp {pp[0]:.2f} | mix {e[1]:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}] pp {pp[1]:.2f} | within {e[2]:+.3f} [{lo[2]:+.3f},{hi[2]:+.3f}] pp {pp[2]:.2f}")
    say("== 3 start state means (drive level)")
    for col in ("hs0", "yl0", "tou"):
        C.row(f" mean {col}", QR, col, "one", QS, col, "one", fmt="{:.2f}")
    say("== 4 4th-down choice in last drive of Q2 (per 4th-down play)")
    for nm, m in (("all", lambda d: d.hs > -1), ("hs<=120", lambda d: d.hs <= 120), ("hs<=60", lambda d: d.hs <= 60), ("hs<=10", lambda d: d.hs <= 10), ("hs>120", lambda d: d.hs > 120)):
        for zn, zm in (("yl<=40", lambda d: d.yl <= 40), ("yl 41-60", lambda d: (d.yl > 40) & (d.yl <= 60)), ("yl>60", lambda d: d.yl > 60)):
            a, b = LR[m(LR) & zm(LR) & (LR.d4 == 1)], LS[m(LS) & zm(LS) & (LS.d4 == 1)]
            if len(a) < 30 or len(b) < 30:
                continue
            say(f"-- {nm} {zn} n real {len(a)} sim {len(b)}")
            for met in ("d4_punt", "d4_fg", "d4_go"):
                C.row(f"  4th {met[3:]}", a, met, "one", b, met, "one")
    say("== 5 FG attempts, Q2 last drive")
    for nm, m in (("all", lambda d: d.hs > -1), ("hs<=120", lambda d: d.hs <= 120), ("hs<=10", lambda d: d.hs <= 10)):
        a, b = LR[m(LR) & (LR.k_fg == 1)], LS[m(LS) & (LS.k_fg == 1)]
        say(f"-- {nm} n real {len(a)} sim {len(b)}")
        C.row("  fg kick length mean", a, "fg_len", "k_fg", b, "fg_len", "k_fg", fmt="{:.2f}")
        for lim in (35, 45, 55):
            a2, b2 = a.assign(m=(a.fg_len >= lim).astype(int)), b.assign(m=(b.fg_len >= lim).astype(int))
            C.row(f"  fg share length>={lim}", a2, "m", "one", b2, "m", "one")
        C.row("  fg made per attempt", a, "fgm", "k_fg", b, "fgm", "k_fg")
    say("== 6 play mix on downs 1-3 by seconds left, Q2 last drive")
    for nm, lo, hi in (("hs<=10", -1, 10), ("10<hs<=30", 10, 30), ("30<hs<=60", 30, 60), ("60<hs<=120", 60, 120)):
        a, b = LR[(LR.hs > lo) & (LR.hs <= hi) & (LR.d123 == 1)], LR.iloc[0:0]
        b = LS[(LS.hs > lo) & (LS.hs <= hi) & (LS.d123 == 1)]
        say(f"-- {nm} n real {len(a)} sim {len(b)}")
        for met in ("k_run", "k_pass", "k_spike", "k_kneel"):
            C.row(f"  share {met[2:]}", a, met, "one", b, met, "one")
    say("== 7 run/pass plays by field position, Q2 last drive")
    for nm, m in (("yl<=10", lambda d: d.yl <= 10), ("10<yl<=25", lambda d: (d.yl > 10) & (d.yl <= 25)), ("25<yl<=50", lambda d: (d.yl > 25) & (d.yl <= 50)), ("yl>50", lambda d: d.yl > 50)):
        a, b = LR[m(LR) & (LR.rp == 1)], LS[m(LS) & (LS.rp == 1)]
        say(f"-- {nm} n real {len(a)} sim {len(b)}")
        C.row("  td per play", a, "td", "one", b, "td", "one")
        C.row("  pass share", a, "k_pass", "one", b, "k_pass", "one")
        C.row("  td per pass", a, "td_pass", "k_pass", b, "td_pass", "k_pass")
        C.row("  td per run", a, "td_run", "k_run", b, "td_run", "k_run")
        C.row("  yards per pass", a, "yd_pass", "k_pass", b, "yd_pass", "k_pass", fmt="{:.2f}")
        C.row("  pass dist>=15 share", a, "pass_deep", "one", b, "pass_deep", "one")
    say("== 8 hs<=10 run/pass")
    a, b = LR[(LR.hs <= 10) & (LR.rp == 1)], LS[(LS.hs <= 10) & (LS.rp == 1)]
    C.row("  td per play", a, "td", "one", b, "td", "one")
    C.row("  plays per last drive", QR, "n", "one", QS, "n", "one", fmt="{:.2f}")
    say(f"looks {c2.LOOKS[0]} plus 12 post-strat outcome x spec fits")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / OUTNAME).write_text("\n".join(OUT) + "\n")


main()
