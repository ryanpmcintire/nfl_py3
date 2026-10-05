import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
sys.argv = sys.argv[:1]
import mod25e_drivecomp as dc  # noqa: E402

LABEL = dc.LABEL
SEEDS = dc.SEEDS
ART = dc.ART
OUTD = ART / "fpos"
BANDS = dc.BANDS
TYPES = ["start", "koTD", "koFG", "punt", "other"]
rng = np.random.default_rng(77)
OUT = []
LOOKS = [0]
MASK = dc.MASK


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def iv(d, pt):
    LOOKS[0] += 1
    d = np.asarray(d)
    return f"{pt:+.3f} [{np.percentile(d, 5):+.3f},{np.percentile(d, 95):+.3f}] pp {float((d > 0).mean()):.2f}"


def real_build(ns):
    gf = pd.read_parquet(REPO / "data" / "processed" / "game_features_pbp.parquet", columns=["game_id", "home_score", "away_score"]).set_index("game_id")
    cols = ["game_id", "play_id", "season_type", "posteam", "defteam", "home_team", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential", "epa", "fixed_drive_result"]
    out = []
    for s in ns["POOL"]:
        p = pd.read_parquet(ns["PBP"] / f"season={s}" / "plays.parquet", columns=cols)
        p["season"] = s
        p = p[(p.season_type == "REG") & p.posteam.notna() & p.score_differential.notna() & (p.qtr <= 4)]
        p = p.sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)
        p["m"] = np.where(p.posteam == p.home_team, 1.0, -1.0) * p.score_differential
        fin = (gf.home_score - gf.away_score).reindex(p.game_id).to_numpy()
        g = p.game_id.to_numpy()
        last = np.r_[g[1:] != g[:-1], True]
        nxt = np.r_[p.m.to_numpy()[1:], 0.0]
        p["dm"] = np.where(last, fin - p.m.to_numpy(), nxt - p.m.to_numpy())
        p = p[p.dm.notna()]
        ko = (p.play_type == "kickoff").to_numpy()
        p = p.assign(kocum=np.cumsum(ko))
        p = p[~ko & p.play_type.notna()].copy()
        live = p.play_type.isin(dc.LIVE).to_numpy()
        p["ep"] = np.where(live, p.epa.fillna(0.0).to_numpy(), 0.0)
        p["lv"] = live.astype(int)
        g = p.game_id.to_numpy()
        pos = p.posteam.to_numpy()
        kc = p.kocum.to_numpy()
        q = p.qtr.to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (pos[1:] != pos[:-1]) | (kc[1:] != kc[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
        p["d"] = np.cumsum(new)
        gp = p.groupby("d", sort=True)
        first_pos = gp.posteam.first()
        res = gp.fixed_drive_result.last()
        D = pd.DataFrame({"gk": gp.season.first().astype(str) + "_" + gp.game_id.first(), "season": gp.season.first(), "off": first_pos, "dfn": gp.defteam.first(),
                          "s": np.where(first_pos == gp.home_team.first(), 1.0, -1.0), "ql": gp.qtr.last(), "yl0": gp.yardline_100.first(), "yle": gp.yardline_100.last(),
                          "n": gp.size(), "dmsum": gp.dm.sum(), "sd0": gp.score_differential.first(), "e": gp.ep.sum(), "nlive": gp.lv.sum(), "gsr0": gp.game_seconds_remaining.first()})
        D["out"] = dc.outcome_code((res == "Touchdown").to_numpy(), (res == "Field goal").to_numpy(), (res == "Punt").to_numpy())
        D["p"] = D.s * D.dmsum
        D["tm"] = D.season.astype(str) + "_" + D.off
        D["td"] = D.season.astype(str) + "_" + D.dfn
        out.append(D.drop(columns=["dmsum"]))
    return pd.concat(out, ignore_index=True)


def sim_build(epa_t, late):
    out = []
    for sdn in SEEDS:
        sdir = ART / f"e5_{LABEL}_s{sdn}"
        sg = pd.read_parquet(sdir / "sim_games.parquet", columns=["game_id", "home_team", "away_team"])
        sg["w"] = sg.game_id.str[1:5].astype(int)
        sg["s"] = sg.game_id.str[6:8].astype(int)
        sg["g"] = sg.game_id.str[-3:].astype(int)
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < late.BURN:
                continue
            tm = sg[(sg.w == w) & (sg.s == s)].set_index("g")
            d = pd.read_parquet(f, columns=["g", "qtr", "offhome", "yl", "sd", "po", "pdf", "code", "gsr", "idx"])
            d = d[d.qtr <= 4].sort_values(["g"], kind="stable").reset_index(drop=True)
            live = d.code.isin([0, 1, 2, 3]).to_numpy()
            ix = d.idx.to_numpy()
            ok = live & np.isfinite(ix)
            ep = np.zeros(len(d))
            ep[ok] = np.nan_to_num(epa_t[ix[ok].astype(int)])
            d["ep"] = ep
            d["lv"] = live.astype(int)
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
            key = f"{sdn}_{w}_{s}"
            own = gp.po.sum()
            D = pd.DataFrame({"gk": ((sdn * 10000 + w) * 100 + s) * 100000 + gp.g.first().astype(np.int64), "season": 0, "off": gp.off.first(), "dfn": gp.dfn.first(),
                              "s": np.where(gp.offhome.first() == 1, 1.0, -1.0), "ql": gp.qtr.last(), "yl0": gp.yl.first(), "yle": gp.yl.last(), "n": gp.size(), "sd0": gp.sd.first(),
                              "p": gp.pp.sum(), "e": gp.ep.sum(), "nlive": gp.lv.sum(), "gsr0": gp.gsr.first()})
            D["out"] = dc.outcome_code((own >= 6).to_numpy(), (own == 3).to_numpy(), (gp.code.last() == 2).to_numpy())
            D["tm"] = key + "_" + D.off
            D["td"] = key + "_" + D.dfn
            out.append(D)
    return pd.concat(out, ignore_index=True)


def annotate(D, ns):
    D = D.reset_index(drop=True)
    g = D.gk.to_numpy()
    same_prev = np.r_[False, g[1:] == g[:-1]]
    same_next = np.r_[g[1:] == g[:-1], False]
    po = np.r_[[""], D.out.to_numpy()[:-1]]
    pg = np.r_[np.nan, D.gsr0.to_numpy()[:-1]]
    gs0 = D.gsr0.to_numpy()
    half = same_prev & (gs0 <= 1800) & (pg > 1800)
    ty = np.where(~same_prev | half, "start", np.where(po == "TD", "koTD", np.where(po == "FG", "koFG", np.where(po == "P", "punt", "other"))))
    D["ty"] = ty
    D["bi"] = pd.cut(D.sd0, ns["LEAD_EDGES"], labels=False).astype(int)
    D["qi"] = np.clip(D.ql.astype(int).to_numpy(), 1, 4) - 1
    D["nyl0"] = np.where(same_next, np.r_[D.yl0.to_numpy()[1:], np.nan], np.nan)
    nty = np.r_[D.ty.to_numpy()[1:], ["start"]]
    D["nvalid"] = same_next & (nty != "start")
    D["net"] = np.where((D.out == "P") & D.nvalid, D.nyl0 - 100.0 + D.yle, np.nan)
    return D


def components(D, ns):
    comp = {}
    for nm, x in (("p", D.p), ("e", D.e)):
        comp[nm] = ns["components"](D.assign(p=x.to_numpy()))[0]
    return comp["e"].c_rest.to_numpy(), comp["p"].c_startfp.to_numpy()


def boot(G, n, r):
    return r.multinomial(G, np.ones(G) / G, size=n).astype(float)


def emat(D, x, ys, nb_w):
    gi, gn = pd.factorize(D.gk)
    G = len(gn)
    idx = gi * 20 + D.qi.to_numpy() * 5 + D.bi.to_numpy()
    s = D.s.to_numpy()

    def gs(v):
        return np.bincount(idx, weights=np.asarray(v, float), minlength=G * 20).reshape(G, 20)

    cnt = gs(np.ones(len(D)))
    msg = gs(s).reshape(G, 4, 5)
    Ix = gs(s * x).reshape(G, 4, 5).sum(2)
    res = {}
    W = np.vstack([np.ones((1, G)), nb_w])
    nb = len(W)
    sw = W.sum(1)
    tc = W @ cnt

    def cov(X, Y):
        Xb = np.broadcast_to(X, (nb, G, 4))
        Yb = np.broadcast_to(Y, (nb, G, 4))
        mx = np.einsum("bg,bgq->bq", W, Xb) / sw[:, None]
        my = np.einsum("bg,bgq->bq", W, Yb) / sw[:, None]
        c = np.matmul((Xb * W[:, :, None]).transpose(0, 2, 1), Yb) / sw[:, None, None] - mx[:, :, None] * my[:, None, :]
        return (c * MASK).sum((1, 2))

    for nm, y in ys.items():
        ab = ((W @ gs(y)) / np.maximum(tc, 1e-9)).reshape(nb, 4, 5)
        Iy = gs(s * y).reshape(G, 4, 5).sum(2)
        IE = Iy[None] - np.einsum("grd,brd->bgr", msg, ab)
        res[nm] = 2 * cov(Ix, IE)
    return res


def chunked(D, x, ys):
    G = D.gk.nunique()
    parts = []
    done = 0
    first = True
    while done < NB:
        nb = min(10, NB - done)
        r = emat(D, x, ys, boot(G, nb, rng))
        parts.append(r if first else {k: v[1:] for k, v in r.items()})
        first = False
        done += nb
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def cellstat(G, gi, mask, v, W):
    sv = np.bincount(gi, weights=np.where(mask, v, 0.0), minlength=G)
    sm = np.bincount(gi, weights=mask.astype(float), minlength=G)
    return (W @ sv) / np.maximum(W @ sm, 1e-9), float(sv.sum() / max(sm.sum(), 1e-9)), int(sm.sum())


def section_a(Dr, Ds):
    say("## A. later-drive start type (prior drive end) mix and mean start yardline_100 (higher = worse), real | sim")
    Gr, Gs = Dr.gk.nunique(), Ds.gk.nunique()
    gir, gis = pd.factorize(Dr.gk)[0], pd.factorize(Ds.gk)[0]
    Wr, Ws = boot(Gr, NB, rng), boot(Gs, NB, rng)
    ones_r, ones_s = np.ones(len(Dr), bool), np.ones(len(Ds), bool)
    for t in TYPES:
        mr, ms = (Dr.ty == t).to_numpy(), (Ds.ty == t).to_numpy()
        sr, pr, _ = cellstat(Gr, gir, ones_r, mr.astype(float), Wr)
        ss, ps, _ = cellstat(Gs, gis, ones_s, ms.astype(float), Ws)
        yr, ypr, nr = cellstat(Gr, gir, mr, Dr.yl0.to_numpy(float), Wr)
        ys, yps, ns_ = cellstat(Gs, gis, ms, Ds.yl0.to_numpy(float), Ws)
        say(f"  {t:6s} share {pr:.3f}|{ps:.3f} {iv(ss - sr, ps - pr)} ; mean yl0 {ypr:.2f}|{yps:.2f} {iv(ys - yr, yps - ypr)} ; n {nr}|{ns_}")
    say("  start yl0 by type and quarter of drive end (real|sim, sim-real)")
    for t in TYPES:
        for q in range(4):
            mr, ms = ((Dr.ty == t) & (Dr.qi == q)).to_numpy(), ((Ds.ty == t) & (Ds.qi == q)).to_numpy()
            if mr.sum() < 30 or ms.sum() < 30:
                continue
            yr, ypr, nr = cellstat(Gr, gir, mr, Dr.yl0.to_numpy(float), Wr)
            ys, yps, ns_ = cellstat(Gs, gis, ms, Ds.yl0.to_numpy(float), Ws)
            say(f"    {t:6s} Q{q + 1} yl0 {ypr:.2f}|{yps:.2f} {iv(ys - yr, yps - ypr)} n {nr}|{ns_}")


def section_a2(Dr, Ds, ns):
    say("## A2. early EFF -> later F cross-quarter term (2x cov over 6 quarter pairs) split by start type; F = type mix (real type-mean F) + within-type yardline; real | sim, sim-real")
    xr, Fr = components(Dr, ns)
    xs, Fs = components(Ds, ns)
    tm = {t: float(Fr[(Dr.ty == t).to_numpy()].mean()) for t in TYPES}
    say("  type-mean F (points expectation of start position, real-fitted): " + ", ".join(f"{t} {tm[t]:+.3f}" for t in TYPES))
    outs = {}
    for nm, D, x, F in (("real", Dr, xr, Fr), ("sim", Ds, xs, Fs)):
        ty = D.ty.to_numpy()
        fmix = np.array([tm[t] for t in ty])
        ys = {"F": F, "mix": fmix, "within": F - fmix}
        for t in TYPES:
            ys["F_" + t] = np.where(ty == t, F, 0.0)
            ys["mix_" + t] = np.where(ty == t, fmix, 0.0)
            ys["within_" + t] = np.where(ty == t, F - fmix, 0.0)
        outs[nm] = chunked(D, x, ys)
    for k in outs["real"]:
        r, s = outs["real"][k], outs["sim"][k]
        say(f"  {k:14s} real {r[0]:+.2f} | sim {s[0]:+.2f} | sim-real {iv(s[1:] - r[1:], s[0] - r[0])}")


def section_b(Dr, Ds):
    say("## B. drive that precedes a start, by its offence lead band at drive start x quarter of its end; real | sim, sim-real")
    say("   shares over all drives in the cell; plays, punt spot yardline_100 (yle), net punt, next-team start yl0 after punt / after score over drives meeting each condition")
    Gr, Gs = Dr.gk.nunique(), Ds.gk.nunique()
    gir, gis = pd.factorize(Dr.gk)[0], pd.factorize(Ds.gk)[0]
    Wr, Ws = boot(Gr, NB, rng), boot(Gs, NB, rng)

    def pn(D):
        return (D.out == "P").to_numpy() & D.nvalid.to_numpy()

    def sc(D):
        return D.out.isin(["TD", "FG"]).to_numpy() & D.nvalid.to_numpy()

    specs = [("P share", lambda D: (D.out == "P").to_numpy(float), None), ("score share", lambda D: D.out.isin(["TD", "FG"]).to_numpy(float), None),
             ("other share", lambda D: (D.out == "O").to_numpy(float), None),
             ("punt-drive plays", lambda D: D.n.to_numpy(float), pn), ("punt yle", lambda D: D.yle.to_numpy(float), pn),
             ("net punt", lambda D: D.net.to_numpy(float), pn), ("yl0 after punt", lambda D: D.nyl0.to_numpy(float), pn),
             ("yl0 after score", lambda D: D.nyl0.to_numpy(float), sc)]
    for q in range(4):
        say(f"  Q{q + 1}")
        for b in range(5):
            row = []
            for nm, vf, mf in specs:
                cr = ((Dr.bi == b) & (Dr.qi == q)).to_numpy()
                cs = ((Ds.bi == b) & (Ds.qi == q)).to_numpy()
                mr = cr if mf is None else cr & mf(Dr)
                ms = cs if mf is None else cs & mf(Ds)
                if mr.sum() < 30 or ms.sum() < 30:
                    row.append(f"{nm} n/a")
                    continue
                br, pr, nr = cellstat(Gr, gir, mr, vf(Dr), Wr)
                bs, ps, ns_ = cellstat(Gs, gis, ms, vf(Ds), Ws)
                row.append(f"{nm} {pr:.3f}|{ps:.3f} {iv(bs - br, ps - pr)}")
            say(f"    {BANDS[b]:8s} " + " ; ".join(row))


def main():
    import mod25e_late as late

    OUTD.mkdir(parents=True, exist_ok=True)
    ns = late.load_xq()
    Dr = annotate(real_build(ns), ns)
    Ds = annotate(sim_build(dc.epa_table(), late), ns)
    say(f"label {LABEL} seeds {SEEDS}; bootstrap {NB}; real drives {len(Dr)} sim drives {len(Ds)}; games {Dr.gk.nunique()} | {Ds.gk.nunique()}")
    section_a(Dr, Ds)
    section_a2(Dr, Ds, ns)
    section_b(Dr, Ds)
    say(f"looks counted (interval prints): {LOOKS[0]}")
    (OUTD / "fpos.txt").write_text("\n".join(OUT))


if __name__ == "__main__":
    main()
