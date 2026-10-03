import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_late as ml

ART = REPO / "artifacts" / "mod25e3"
LABEL = "crHpqokg"
SEEDS = (11, 12, 13)
POOL = ml.POOL
BURN = ml.BURN
PBP = ml.PBP
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = []
LOOKS = [0]


def say(s=""):
    print(s)
    OUT.append(s)


def common(p):
    g, q = p.gk.to_numpy(), p.qtr.to_numpy()
    half = (q >= 3).astype(int)
    p["lasthalf"] = np.r_[(g[1:] != g[:-1]) | (half[1:] != half[:-1]), True]
    d = p.d.to_numpy()
    nd = np.r_[d[1:] == d[:-1], False]
    p["nextdown"] = np.where(nd, np.r_[p.down.to_numpy()[1:], np.nan], np.nan)
    p["nextsame"] = nd
    return p


def real_plays():
    cols = ["game_id", "play_id", "season_type", "posteam", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential", "down", "ydstogo", "qb_kneel", "qb_spike", "interception", "fumble_lost", "touchdown", "posteam_score", "posteam_score_post"]
    out = []
    for s in POOL:
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=cols)
        p = p[(p.season_type == "REG") & p.posteam.notna() & p.score_differential.notna() & (p.qtr <= 4)]
        p = p.sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
        ko = (p.play_type == "kickoff").to_numpy()
        p["kocum"] = np.cumsum(ko)
        p = p[~ko & p.play_type.isin(["run", "pass", "punt", "field_goal"])].copy()
        g, pos, kc, q = p.game_id.to_numpy(), p.posteam.to_numpy(), p.kocum.to_numpy(), p.qtr.to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (pos[1:] != pos[:-1]) | (kc[1:] != kc[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
        kn = (p.qb_kneel == 1) | (p.qb_spike == 1)
        kind = np.select([(p.play_type == "run") & ~kn, (p.play_type == "pass") & ~kn, p.play_type == "punt", p.play_type == "field_goal"], [0, 1, 2, 3], -1)
        rpm = (kind >= 0) & (kind <= 1)
        to = rpm & (p.down <= 3).to_numpy() & ((p.interception == 1) | (p.fumble_lost == 1)).to_numpy()
        td = rpm & (p.touchdown == 1).to_numpy() & (p.posteam_score_post > p.posteam_score).to_numpy()
        D = pd.DataFrame({"gk": str(s) + "_" + p.game_id.to_numpy(), "d": np.cumsum(new) + (s - POOL[0]) * 10**7, "qtr": q, "gsr": p.game_seconds_remaining.to_numpy(),
                          "sd": p.score_differential.to_numpy(), "down": p.down.to_numpy(), "dist": p.ydstogo.to_numpy(), "yl": p.yardline_100.to_numpy(), "kind": kind, "to": to.astype(int), "td": td.astype(int)})
        D = D[D.down.notna()]
        out.append(common(D.reset_index(drop=True)))
    return pd.concat(out, ignore_index=True)


def sim_plays():
    out = []
    use = ["g", "down", "dist", "yl", "sd", "gsr", "qtr", "code", "po", "pdf", "flip", "offhome"]
    off = 0
    for sd in SEEDS:
        sdir = ART / f"e5_{LABEL}_s{sd}"
        for f in sorted(sdir.glob("play_*_*.parquet")):
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
            d = d[d.code.isin([0, 1, 2, 3])]
            c = d.code.astype(int)
            rp = c <= 1
            D = pd.DataFrame({"gk": f"{sd}_{w}_{s}_" + d.g.astype(int).astype(str).to_numpy(), "d": d.dd.to_numpy(), "qtr": d.qtr.to_numpy(), "gsr": d.gsr.to_numpy(), "sd": d.sd.to_numpy(),
                              "down": d.down.to_numpy(), "dist": d.dist.to_numpy(), "yl": d.yl.to_numpy(), "kind": c.to_numpy(),
                              "to": (rp & (d.flip == 1) & (d.po == 0) & (d.down <= 3)).astype(int).to_numpy(), "td": (rp & (d.po >= 6)).astype(int).to_numpy()})
            D = D[D.down.notna()]
            out.append(common(D.reset_index(drop=True)))
    return pd.concat(out, ignore_index=True)


def enrich(P, cuts):
    P = P.copy()
    rp = P.kind.isin([0, 1])
    P["rp"] = rp
    P["conv"] = ((rp & (P.td == 1)) | (rp & P.nextsame & (P.nextdown == 1))).astype(int)
    P["qd"] = P.qtr.astype(int)
    P["ss"] = np.sign(P.sd).astype(int) + 1
    P["zone"] = np.digitize(P.yl, cuts["yl"])
    P["db"] = np.digitize(P.dist, cuts["dist"])
    P["one"] = 1
    return P


def drive_frame(P):
    P = P.sort_values("d", kind="stable")
    gp = P.groupby("d", sort=False)
    first, last = gp.first(), gp.last()
    lk = last.kind
    D = pd.DataFrame({"gk": first.gk, "qd": first.qtr.astype(int), "ss": np.sign(first.sd).astype(int) + 1, "yl0": first.yl})
    D["r4"] = (P.down == 4).groupby(P.d, sort=False).max().astype(int)
    D["ser"] = (P.down == 1).groupby(P.d, sort=False).sum()
    D["punt"] = (lk == 2).astype(int)
    D["fg"] = (lk == 3).astype(int)
    D["td"] = (last.td == 1).astype(int)
    D["turn"] = (last.to == 1).astype(int)
    D["downs"] = ((lk <= 1) & (lk >= 0) & (last.down == 4) & (last.td == 0) & (last.to == 0)).astype(int)
    D["other"] = (1 - D[["punt", "fg", "td", "turn", "downs"]].sum(axis=1).clip(upper=1)).astype(int)
    D["pre4"] = 1 - D.r4
    D["td_b"] = D.td * D.pre4
    D["to_b"] = D.turn * D.pre4
    D["oth_b"] = D.other * D.pre4
    D["one"] = 1
    return D


def fourth_frame(P):
    F = P[(P.down == 4) & P.kind.isin([0, 1, 2, 3])].copy()
    F["punt"] = (F.kind == 2).astype(int)
    F["fg"] = (F.kind == 3).astype(int)
    F["go"] = F.kind.isin([0, 1]).astype(int)
    return F


def gs(df, col, games):
    return df.groupby("gk")[col].sum().reindex(games).fillna(0.0).to_numpy()


def boot(nr_, dr_, ns_, ds_, rng, nb):
    nr, ns = len(nr_), len(ns_)
    est = ns_.sum() / max(ds_.sum(), 1e-12) - nr_.sum() / max(dr_.sum(), 1e-12)
    o = np.empty(nb)
    for i in range(nb):
        ir = rng.integers(0, nr, nr)
        is_ = rng.integers(0, ns, ns)
        o[i] = ns_[is_].sum() / max(ds_[is_].sum(), 1e-12) - nr_[ir].sum() / max(dr_[ir].sum(), 1e-12)
    return est, np.quantile(o, 0.025), np.quantile(o, 0.975), (o > 0).mean()


class Cmp:
    def __init__(self, nb, rng, rg, sg):
        self.nb, self.rng, self.rg, self.sg = nb, rng, rg, sg

    def row(self, label, ra, rn, rd, sa, sn, sd):
        LOOKS[0] += 1
        a, b = gs(ra, rn, self.rg), gs(ra, rd, self.rg)
        c, d = gs(sa, sn, self.sg), gs(sa, sd, self.sg)
        e, lo, hi, pp = boot(a, b, c, d, self.rng, self.nb)
        say(f"{label:30s} real {a.sum() / max(b.sum(), 1e-12):.3f} sim {c.sum() / max(d.sum(), 1e-12):.3f} diff {e:+.3f} [{lo:+.3f},{hi:+.3f}] pp {pp:.2f} n {int(b.sum())}/{int(d.sum())}")


def main():
    rng = np.random.default_rng(0)
    R = real_plays()
    S = sim_plays()
    rp3 = R[(R.down == 3) & R.kind.isin([0, 1])]
    cuts = {"yl": np.quantile(R[R.down == 1].yl, [0.25, 0.5, 0.75]), "dist": np.quantile(rp3.dist, [1 / 3, 2 / 3])}
    say(f"cuts yl {cuts['yl']} dist {cuts['dist']}")
    R, S = enrich(R, cuts), enrich(S, cuts)
    DR, DS = drive_frame(R), drive_frame(S)
    for c in (DR, DS):
        c["yz"] = np.digitize(c.yl0, cuts["yl"])
    rg, sg = np.sort(R.gk.unique()), np.sort(S.gk.unique())
    C = Cmp(NB, rng, rg, sg)
    say("== drive chain: punt share = reach4 x punt given 4th ==")
    slices = [("all", "one", 1)] + [(f"Q{q}", "qd", q) for q in (1, 2, 3, 4)] + [(n, "ss", v) for v, n in ((0, "trail"), (1, "tied"), (2, "lead"))] + [(f"zone{z}", "yz", z) for z in range(4)]
    for nm, col, v in slices:
        a, b = DR[DR[col] == v], DS[DS[col] == v]
        say(f"-- {nm}")
        for met in ("punt", "r4", "td_b", "to_b", "oth_b", "fg", "td", "turn", "downs", "other"):
            C.row(f" drive {met}", a, met, "one", b, met, "one")
        C.row(" punt|reach4", a.assign(pr=a.punt * a.r4), "pr", "r4", b.assign(pr=b.punt * b.r4), "pr", "r4")
        C.row(" series per drive", a, "ser", "one", b, "ser", "one")
    say("== conversion hazard by down and dist tercile (next play down 1 or TD; last play of half excluded) ==")
    RR, SS = R[R.rp & ~R.lasthalf], S[S.rp & ~S.lasthalf]
    for dn in (1, 2, 3, 4):
        for db in (0, 1, 2):
            a, b = RR[(RR.down == dn) & (RR.db == db)], SS[(SS.down == dn) & (SS.db == db)]
            C.row(f" down{dn} dist{db} conv", a, "conv", "one", b, "conv", "one")
            C.row(f" down{dn} dist{db} turnover", a, "to", "one", b, "to", "one")
    say("== state mix ==")
    a3, b3 = R[(R.down == 3) & R.rp], S[(S.down == 3) & S.rp]
    FR, FS = fourth_frame(R), fourth_frame(S)
    for db in (0, 1, 2):
        C.row(f" 3rd dist{db} share", a3.assign(x=(a3.db == db).astype(int)), "x", "one", b3.assign(x=(b3.db == db).astype(int)), "x", "one")
        C.row(f" 4th dist{db} share", FR.assign(x=(FR.db == db).astype(int)), "x", "one", FS.assign(x=(FS.db == db).astype(int)), "x", "one")
    for z in range(4):
        C.row(f" 4th zone{z} share", FR.assign(x=(FR.zone == z).astype(int)), "x", "one", FS.assign(x=(FS.zone == z).astype(int)), "x", "one")
    say("== 4th-down decision given state ==")
    for z in range(4):
        for met in ("punt", "fg", "go"):
            C.row(f" 4th zone{z} {met}", FR[FR.zone == z], met, "one", FS[FS.zone == z], met, "one")
    for q in (1, 2, 3, 4):
        C.row(f" 4th Q{q} punt", FR[FR.qd == q], "punt", "one", FS[FS.qd == q], "punt", "one")
    for v, n in ((0, "trail"), (1, "tied"), (2, "lead")):
        C.row(f" 4th {n} punt", FR[FR.ss == v], "punt", "one", FS[FS.ss == v], "punt", "one")
    say("== standardised: real cell punt rate (zone x dist x qtr x score) on each side's 4th states ==")
    keys = ["zone", "db", "qd", "ss"]
    cell = FR.groupby(keys).punt.mean().rename("pr").reset_index()
    FR2 = FR.merge(cell, on=keys, how="left")
    FS2 = FS.merge(cell, on=keys, how="left")
    FR2["pr"] = FR2.pr.fillna(FR.punt.mean())
    FS2["pr"] = FS2.pr.fillna(FR.punt.mean())
    C.row(" expected punt|4th: real states", FR2, "pr", "one", FS2.iloc[0:0], "pr", "one") if False else None
    LOOKS[0] += 1
    a, b = gs(FR2, "pr", rg), gs(FR2, "one", rg)
    c, d, e_ = gs(FS2, "pr", sg), gs(FS2, "one", sg), gs(FS2, "punt", sg)
    e, lo, hi, pp = boot(a, b, c, d, rng, NB)
    say(f" expected|real-state {a.sum() / b.sum():.3f} expected|sim-state {c.sum() / d.sum():.3f} state-mix diff {e:+.3f} [{lo:+.3f},{hi:+.3f}] pp {pp:.2f}")
    LOOKS[0] += 1
    e, lo, hi, pp = boot(c, d, e_, d, rng, NB)
    say(f" sim actual {e_.sum() / d.sum():.3f} minus sim-state expected {c.sum() / d.sum():.3f}: decision diff {e:+.3f} [{lo:+.3f},{hi:+.3f}] pp {pp:.2f} (same games, paired resample approximated by independent)")
    say(f"looks {LOOKS[0]}")
    (ART / "punt").mkdir(exist_ok=True)
    (ART / "punt" / "punt.txt").write_text("\n".join(OUT))


main()
