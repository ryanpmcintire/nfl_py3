import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_late as ml

ART = REPO / "artifacts" / "mod25e3"
LABEL = os.environ.get("CLK_LABEL", "crHpqokg")
SEEDS = tuple(int(x) for x in os.environ.get("CLK_SEEDS", "11,12,13").split(","))
POOL, BURN, PBP = ml.POOL, ml.BURN, ml.PBP
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUTNAME = sys.argv[2] if len(sys.argv) > 2 else "clock2.txt"
OUT, LOOKS = [], [0]
CL = ("inc", "oob", "run", "pass", "spike", "kneel")


def say(s=""):
    print(s)
    OUT.append(s)


def real_plays():
    cols = ["game_id", "play_id", "season_type", "posteam", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential", "down", "qb_kneel", "qb_spike", "interception", "fumble_lost", "touchdown", "posteam_score", "posteam_score_post", "complete_pass", "yards_gained", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "fixed_drive_result"]
    out = []
    for s in POOL:
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=cols)
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
        D = pd.DataFrame({"gk": str(s) + "_" + p.game_id.to_numpy(), "d": np.cumsum(new) + (s - POOL[0]) * 10**7, "qtr": q, "gsr": p.game_seconds_remaining.to_numpy(),
                          "sd": p.score_differential.to_numpy(), "down": p.down.to_numpy(), "yl": p.yardline_100.to_numpy(), "kind": kind, "to": to.astype(int), "td": td.astype(int),
                          "fgm": ((pt == "field_goal") & (p.fixed_drive_result == "Field goal").to_numpy()).astype(int), "cls": cls,
                          "tou": (p.posteam_timeouts_remaining + p.defteam_timeouts_remaining).to_numpy()})
        out.append(D[D.down.notna()].reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


def sim_plays():
    out = []
    use = ["g", "down", "yl", "sd", "gsr", "qtr", "code", "po", "pdf", "flip", "offhome", "yards", "oto", "dto"]
    off = 0
    for sd in SEEDS:
        for f in sorted((ART / f"e5_{LABEL}_s{sd}").glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < BURN:
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
                              "down": d.down.to_numpy(), "yl": d.yl.to_numpy(), "kind": c, "to": (rp & (fl == 1) & (po == 0) & (d.down <= 3).to_numpy()).astype(int),
                              "td": (rp & (po >= 6)).astype(int), "fgm": ((c == 3) & (po == 3)).astype(int), "cls": cls,
                              "tou": (d.oto + d.dto).to_numpy()})
            out.append(D[D.down.notna()].reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


def frames(P, cuts):
    P = P.sort_values("d", kind="stable").reset_index(drop=True)
    gp = P.groupby("d", sort=False)
    P["secs"] = P.gsr - gp.gsr.shift(-1)
    first, last = gp.first(), gp.last()
    lk = last.kind
    D = pd.DataFrame({"gk": first.gk, "qd": first.qtr.astype(int), "hs0": np.where(first.qtr == 2, first.gsr - 1800.0, np.where(first.qtr == 4, first.gsr, np.nan)),
                      "yl0": first.yl, "ss": np.sign(first.sd).astype(int) + 1})
    D["hl"] = np.where(last.qtr == 2, last.gsr - 1800.0, np.where(last.qtr == 4, last.gsr, np.nan))
    D["n"] = gp.size()
    D["r4"] = (P.down == 4).groupby(P.d, sort=False).max().astype(int)
    D["punt"] = (lk == 2).astype(int)
    D["fg"] = (lk == 3).astype(int)
    D["td"] = (last.td == 1).astype(int)
    D["fgm"] = (last.fgm == 1).astype(int)
    D["turn"] = (last.to == 1).astype(int)
    D["downs"] = ((lk <= 1) & (lk >= 0) & (last.down == 4) & (last.td == 0) & (last.to == 0)).astype(int)
    D["other"] = (1 - D[["punt", "fg", "td", "turn", "downs"]].sum(axis=1).clip(upper=1)).astype(int)
    D["tou"] = first.tou - last.tou
    D["one"] = 1
    D["pre4"] = 1 - D.r4
    gk, half = D.gk.to_numpy(), (D.qd.to_numpy() >= 3)
    D["lh"] = np.r_[(gk[1:] != gk[:-1]) | (half[1:] != half[:-1]), True].astype(int)
    D["oth_pre4"] = D.other * D.pre4
    D["oth_pre4_lh"] = D.oth_pre4 * D.lh
    D["oth_pre4_nlh"] = D.oth_pre4 * (1 - D.lh)
    D["clk_all"] = D.other * D.lh
    D["clk_pre4"] = D.clk_all * D.pre4
    D["lhx"] = D.lh * (1 - D.other)
    D["sc"] = np.where(D.td == 1, 2, np.where(D.fgm == 1, 1, 0))
    D["z"] = np.digitize(D.yl0, cuts)
    return P, D


def gs(df, col, games):
    return df.groupby("gk")[col].sum().reindex(games).fillna(0.0).to_numpy()


def ratio(a, b):
    return a.sum() / max(b.sum(), 1e-12)


def boot(nr_, dr_, ns_, ds_, rng, nb):
    nr, ns = len(nr_), len(ns_)
    est = ratio(ns_, ds_) - ratio(nr_, dr_)
    o = np.empty(nb)
    for i in range(nb):
        ir, is_ = rng.integers(0, nr, nr), rng.integers(0, ns, ns)
        o[i] = ratio(ns_[is_], ds_[is_]) - ratio(nr_[ir], dr_[ir])
    return est, np.quantile(o, 0.025), np.quantile(o, 0.975), (o > 0).mean()


class Cmp:
    def __init__(self, rng, rg, sg):
        self.rng, self.rg, self.sg = rng, rg, sg

    def row(self, label, ra, rn, rd, sa, sn, sd, fmt="{:.3f}"):
        LOOKS[0] += 1
        a, b, c, d = gs(ra, rn, self.rg), gs(ra, rd, self.rg), gs(sa, sn, self.sg), gs(sa, sd, self.sg)
        e, lo, hi, pp = boot(a, b, c, d, self.rng, NB)
        f = fmt.format
        say(f"{label:34s} real {f(ratio(a, b))} sim {f(ratio(c, d))} diff {e:+.3f} [{lo:+.3f},{hi:+.3f}] pp {pp:.2f} n {b.sum():.0f}/{d.sum():.0f}")


def burn_block(C, R, S, a_, b_):
    PR, PS = R[R.d.isin(set(a_.index)) & R.secs.notna() & (R.cls >= 0)], S[S.d.isin(set(b_.index)) & S.secs.notna() & (S.cls >= 0)]
    PR, PS = PR[PR.secs.between(0, 120)].copy(), PS[PS.secs.between(0, 120)].copy()
    PR["one"], PS["one"] = 1, 1
    C.row(" secs per non-final play", PR, "secs", "one", PS, "secs", "one", fmt="{:.2f}")
    for k, nm in enumerate(CL):
        x = PR.assign(m=(PR.cls == k).astype(int), ms=PR.secs * (PR.cls == k))
        y = PS.assign(m=(PS.cls == k).astype(int), ms=PS.secs * (PS.cls == k))
        C.row(f"  share {nm}", x, "m", "one", y, "m", "one")
        if x.m.sum() > 20 and y.m.sum() > 20:
            C.row(f"  secs {nm}", x, "ms", "m", y, "ms", "m", fmt="{:.2f}")
    mr, ms = PR.groupby("cls").secs.agg(["mean", "size"]), PS.groupby("cls").secs.agg(["mean", "size"])
    idx = mr.index.intersection(ms.index)
    wr, ws = mr["size"][idx] / mr["size"][idx].sum(), ms["size"][idx] / ms["size"][idx].sum()
    mix = (ws * mr["mean"][idx]).sum() - (wr * mr["mean"][idx]).sum()
    rate = (ws * (ms["mean"][idx] - mr["mean"][idx])).sum()
    say(f" secs/play sim-real {(ws * ms['mean'][idx]).sum() - (wr * mr['mean'][idx]).sum():+.2f} = class-mix {mix:+.2f} + per-class rate {rate:+.2f} (point estimates, classes {list(idx)})")


def main():
    rng = np.random.default_rng(0)
    R, S = real_plays(), sim_plays()
    cuts = np.quantile(R[R.down == 1].yl, [0.25, 0.5, 0.75])
    R, DR = frames(R, cuts)
    S, DS = frames(S, cuts)
    rg, sg = np.sort(DR.gk.unique()), np.sort(DS.gk.unique())
    C = Cmp(rng, rg, sg)
    say(f"label {LABEL} seeds {SEEDS} real games {len(rg)} sim games {len(sg)} NB {NB}")
    say("== 1 definitions: pre-4th other (E51 def) split by last drive of half; clock-ended = last drive of half ending other ==")
    for nm, m in (("all", None), ("Q1", 1), ("Q2", 2), ("Q3", 3), ("Q4", 4)):
        a, b = (DR, DS) if m is None else (DR[DR.qd == m], DS[DS.qd == m])
        say(f"-- {nm}")
        for met in ("oth_pre4", "oth_pre4_lh", "oth_pre4_nlh", "clk_all", "clk_pre4", "lhx"):
            C.row(f" drive {met}", a, met, "one", b, met, "one")
    say("== 1b last-drive-of-half end type shares ==")
    for q in (2, 4):
        a, b = DR[(DR.qd == q) & (DR.lh == 1)], DS[(DS.qd == q) & (DS.lh == 1)]
        for met in ("other", "punt", "fg", "td", "turn", "downs"):
            C.row(f" Q{q} lastdrive {met}", a, met, "one", b, met, "one")
    for T in (120, 240):
        say(f"== 2 drives starting with <= {T}s left in Q2/Q4 ==")
        a_, b_ = DR[(DR.hs0 <= T) & DR.qd.isin([2, 4])], DS[(DS.hs0 <= T) & DS.qd.isin([2, 4])]
        for nm, m in (("Q2+Q4", None), ("Q2", 2), ("Q4", 4), ("trail", "ss0"), ("tied", "ss1"), ("lead", "ss2")):
            if m is None:
                a, b = a_, b_
            elif isinstance(m, int):
                a, b = a_[a_.qd == m], b_[b_.qd == m]
            else:
                a, b = a_[a_.ss == int(m[2:])], b_[b_.ss == int(m[2:])]
            say(f"-- {nm}")
            for met in ("clk_all", "td", "fgm", "punt", "turn", "downs"):
                C.row(f" {met}", a, met, "one", b, met, "one")
            for met in ("hs0", "yl0", "n", "tou", "hl"):
                C.row(f" mean {met}", a, met, "one", b, met, "one", fmt="{:.2f}")
        C.row(" no score", a_.assign(x=(a_.sc == 0).astype(int)), "x", "one", b_.assign(x=(b_.sc == 0).astype(int)), "x", "one")
        say("-- play burn inside these drives (non-final plays; seconds to next play)")
        burn_block(C, R, S, a_, b_)
    say("== 3 start-state standardisation of clock-ended share among drives starting <= 240s (hs bin 40s x zone x score sign x quarter) ==")
    a_, b_ = DR[(DR.hs0 <= 240) & DR.qd.isin([2, 4])].copy(), DS[(DS.hs0 <= 240) & DS.qd.isin([2, 4])].copy()
    for x in (a_, b_):
        x["hb"] = np.minimum((x.hs0 // 40).astype(int), 5)
    keys = ["hb", "z", "ss", "qd"]
    for met in ("clk_all", "clk_pre4"):
        cell = a_.groupby(keys)[met].mean().rename("ex").reset_index()
        b2 = b_.merge(cell, on=keys, how="left")
        b2["ex"] = b2.ex.fillna(a_[met].mean())
        LOOKS[0] += 1
        ga, gb = gs(a_, met, rg), gs(a_, "one", rg)
        gc, gd, ge = gs(b2, "ex", sg), gs(b2, "one", sg), gs(b2, met, sg)
        e1, e2 = boot(ga, gb, gc, gd, rng, NB), boot(gc, gd, ge, gd, rng, NB)
        say(f" {met}: real {ratio(ga, gb):.3f} | real rates on sim start mix {ratio(gc, gd):.3f} | sim actual {ratio(ge, gd):.3f}")
        say(f"   start-mix part {e1[0]:+.3f} [{e1[1]:+.3f},{e1[2]:+.3f}] pp {e1[3]:.2f}; within-state part {e2[0]:+.3f} [{e2[1]:+.3f},{e2[2]:+.3f}] pp {e2[3]:.2f}")
    say("== 4 clock-ended drives only: time left at the last play, plays, start state ==")
    a_, b_ = DR[(DR.clk_all == 1) & DR.qd.isin([2, 4])], DS[(DS.clk_all == 1) & DS.qd.isin([2, 4])]
    for q in (2, 4):
        a, b = a_[a_.qd == q], b_[b_.qd == q]
        say(f"-- Q{q}")
        for met in ("hs0", "hl", "n", "yl0", "tou"):
            C.row(f" mean {met}", a, met, "one", b, met, "one", fmt="{:.2f}")
        for lim in (5, 15, 30):
            C.row(f" last play started <= {lim}s", a.assign(x=(a.hl <= lim).astype(int)), "x", "one", b.assign(x=(b.hl <= lim).astype(int)), "x", "one")
    say(f"looks {LOOKS[0]}")
    (ART / "clock2").mkdir(exist_ok=True)
    (ART / "clock2" / OUTNAME).write_text("\n".join(OUT))


main()
