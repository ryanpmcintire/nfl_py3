import os
import sys
from pathlib import Path

os.environ.setdefault("DC_LABEL", "crHpqokgndecsmfwt")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_fpos as f1  # noqa: E402

dc = f1.dc
OUTD = dc.ART / "fpos2"
END_RES = ("End of half", "End of game")
rng = np.random.default_rng(80)


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
        D["res"] = res.fillna("").to_numpy()
        D["out"] = dc.outcome_code((res == "Touchdown").to_numpy(), (res == "Field goal").to_numpy(), (res == "Punt").to_numpy())
        D["p"] = D.s * D.dmsum
        D["tm"] = D.season.astype(str) + "_" + D.off
        D["td"] = D.season.astype(str) + "_" + D.dfn
        out.append(D.drop(columns=["dmsum"]))
    return pd.concat(out, ignore_index=True)


def annotate2(D, ns, real):
    D = f1.annotate(D, ns)
    g = D.gk.to_numpy()
    same_next = np.r_[g[1:] == g[:-1], False]
    D["nsd0"] = np.where(same_next, np.r_[D.sd0.to_numpy()[1:], np.nan], np.nan)
    D["nbi"] = pd.cut(D.nsd0, ns["LEAD_EDGES"], labels=False)
    D["tb"] = (D.nyl0 == 80).astype(float)
    D["gain"] = (100.0 - D.yle) - D.nyl0
    if real:
        endp = np.r_[False, D.res.isin(END_RES).to_numpy()[:-1]] & np.r_[False, g[1:] == g[:-1]]
        D["endprev"] = endp
        D.loc[endp, "ty"] = "start"
    else:
        D["endprev"] = False
    return D


def pergame(gi, G, mask, x, y):
    w = mask.astype(float)
    xm = np.where(mask, x, 0.0)
    ym = np.where(mask, y, 0.0)
    return [np.bincount(gi, weights=a, minlength=G) for a in (w, w * xm, w * ym, w * xm * xm, w * xm * ym)]


def slope_stats(gi, G, mask, x, y, W, qmask):
    S = []
    for q in qmask:
        a = pergame(gi, G, mask & q, x, y)
        S.append(tuple(W @ v for v in a))
    return S


def slope_from(S, sel):
    num = 0.0
    den = 0.0
    for k in sel:
        n, sx, sy, sxx, sxy = S[k]
        nn = np.maximum(n, 1e-9)
        num = num + (sxy - sx * sy / nn)
        den = den + (sxx - sx * sx / nn)
    return num / np.maximum(den, 1e-9)


def slope_block(label, Dr, Ds, mr, ms, vr, vs, xr, xs, NB):
    Gr, Gs = Dr.gk.nunique(), Ds.gk.nunique()
    gir, gis = pd.factorize(Dr.gk)[0], pd.factorize(Ds.gk)[0]
    Wr = np.vstack([np.ones((1, Gr)), f1.boot(Gr, NB, rng)])
    Ws = np.vstack([np.ones((1, Gs)), f1.boot(Gs, NB, rng)])
    qr = [(Dr.qi == q).to_numpy() for q in range(4)]
    qs = [(Ds.qi == q).to_numpy() for q in range(4)]
    Sr = slope_stats(gir, Gr, mr, xr, vr, Wr, qr)
    Ss = slope_stats(gis, Gs, ms, xs, vs, Ws, qs)
    f1.say(f"  {label} per lead-band step (slope):")
    for nm, sel in (("Q1", [0]), ("Q2", [1]), ("Q3", [2]), ("Q4", [3]), ("allQ", [0, 1, 2, 3])):
        br, bs = slope_from(Sr, sel), slope_from(Ss, sel)
        f1.say(f"     {nm} real {br[0]:+.3f} sim {bs[0]:+.3f} s-r {f1.iv(bs[1:] - br[1:], bs[0] - br[0])}")


def loso(Dr, m, col, x):
    mm = m.to_numpy() if hasattr(m, "to_numpy") else m
    d = pd.DataFrame({"v": Dr[col].to_numpy(float)[mm], "x": np.asarray(x)[mm], "season": Dr.season.to_numpy()[mm], "qi": Dr.qi.to_numpy()[mm]})
    d = d[np.isfinite(d.v) & np.isfinite(d.x)]
    Q = np.eye(4)[d.qi.to_numpy().astype(int)]
    X1 = np.column_stack([Q, d.x.to_numpy()])
    y = d.v.to_numpy()
    gains = []
    for s in sorted(d.season.unique()):
        te = (d.season == s).to_numpy()
        e0 = y[te] - Q[te] @ np.linalg.lstsq(Q[~te], y[~te], rcond=None)[0]
        e1 = y[te] - X1[te] @ np.linalg.lstsq(X1[~te], y[~te], rcond=None)[0]
        gains.append(float((e0 ** 2).sum() - (e1 ** 2).sum()))
    b = np.linalg.lstsq(X1, y, rcond=None)[0][-1]
    sse0 = float(((y - Q @ np.linalg.lstsq(Q, y, rcond=None)[0]) ** 2).sum())
    return b, sum(gains), sum(g > 0 for g in gains), len(gains), sum(gains) / sse0


def section_mix(Dr, Ds, ns):
    f1.say("## 1. start-type mix and yl0 with half-last/game-last drives excluded from start-type classification (real result End of half/game -> start)")
    f1.say(f"  real later drives preceded by End of half/game result (reclassified start): {int(Dr.endprev.sum())}; counts by type after reclass: " + ", ".join(f"{t} {int((Dr.ty == t).sum())}" for t in f1.TYPES))
    prev = np.r_[[""], Dr.res.to_numpy()[:-1]][(Dr.ty == "other").to_numpy()]
    f1.say("  real other starts by preceding drive result: " + ", ".join(f"{k} {v}" for k, v in pd.Series(prev).value_counts().items()))
    f1.section_a(Dr, Ds)
    f1.section_a2(Dr, Ds, ns)


def section_strategy(Dr, Ds, NB):
    f1.say("## 2. strategic hypotheses: slope of outcome on lead band index (centred at tied, 5 bands) within quarter; sim minus real")
    xr, xs = Dr.bi.to_numpy(float) - 2.0, Ds.bi.to_numpy(float) - 2.0
    xnr, xns = Dr.nbi.to_numpy(float) - 2.0, Ds.nbi.to_numpy(float) - 2.0
    pr, ps = (Dr.out == "P").to_numpy() & Dr.nvalid.to_numpy(), (Ds.out == "P").to_numpy() & Ds.nvalid.to_numpy()
    f1.say(" PUNTS (x = punting team lead band at drive start; positive = punter leads)")
    for nm, col in (("net punt yards", "net"), ("receiving start yl0", "nyl0"), ("touchback share (start at 80)", "tb"), ("punt spot yle", "yle")):
        slope_block(nm, Dr, Ds, pr, ps, Dr[col].to_numpy(float), Ds[col].to_numpy(float), xr, xs, NB)
    kmask = {}
    for ty, lab in (("TD", "KICKOFF after TD"), ("FG", "KICKOFF after FG")):
        kr = (Dr.out == ty).to_numpy() & Dr.nvalid.to_numpy() & np.isfinite(xnr)
        ks = (Ds.out == ty).to_numpy() & Ds.nvalid.to_numpy() & np.isfinite(xns)
        kmask[ty] = kr
        f1.say(f" {lab} (x = receiving team lead band after the score; positive = receiver leads)")
        slope_block("receiving start yl0", Dr, Ds, kr, ks, Dr.nyl0.to_numpy(float), Ds.nyl0.to_numpy(float), xnr, xns, NB)
        slope_block("touchback share", Dr, Ds, kr, ks, Dr.tb.to_numpy(float), Ds.tb.to_numpy(float), xnr, xns, NB)
    orr, osd = (Dr.out == "O").to_numpy() & Dr.nvalid.to_numpy(), (Ds.out == "O").to_numpy() & Ds.nvalid.to_numpy()
    f1.say(" OTHER drive ends (x = losing team lead band; gain = ground the other side gains over the spot, return yards net of kick-miss)")
    slope_block("return gain yards", Dr, Ds, orr, osd, Dr.gain.to_numpy(float), Ds.gain.to_numpy(float), xr, xs, NB)
    f1.say(" net punt yards by punter lead band, all quarters, real|sim")
    Gr, Gs = Dr.gk.nunique(), Ds.gk.nunique()
    gir, gis = pd.factorize(Dr.gk)[0], pd.factorize(Ds.gk)[0]
    Wr, Ws = f1.boot(Gr, NB, rng), f1.boot(Gs, NB, rng)
    for b in range(5):
        a, pa, na = f1.cellstat(Gr, gir, pr & (Dr.bi == b).to_numpy(), Dr.net.to_numpy(float), Wr)
        c, pc, nc = f1.cellstat(Gs, gis, ps & (Ds.bi == b).to_numpy(), Ds.net.to_numpy(float), Ws)
        f1.say(f"   {f1.BANDS[b]:8s} {pa:.2f}|{pc:.2f} {f1.iv(c - a, pc - pa)} n {na}|{nc}")
    f1.say("## 2b. real-only LOSO (season held out): quarter-mean model plus lead-band slope; gain = held-out SSE reduction (positive = lead helps)")
    for lab, m, col, x in (("punt net", pr, "net", xr), ("punt start yl0", pr, "nyl0", xr), ("punt touchback", pr, "tb", xr),
                           ("ko TD yl0", kmask["TD"], "nyl0", xnr), ("ko FG yl0", kmask["FG"], "nyl0", xnr), ("other gain", orr, "gain", xr)):
        b, tot, w, n, rel = loso(Dr, m, col, x)
        f1.LOOKS[0] += 1
        f1.say(f"  {lab:16s} beta {b:+.3f} per band; LOSO SSE gain {tot:+.1f} ({rel * 100:+.3f}% of quarter-mean SSE); seasons positive {w}/{n}")


def main():
    import mod25e_late as late

    OUTD.mkdir(parents=True, exist_ok=True)
    ns = late.load_xq()
    NB = f1.NB
    Dr = annotate2(real_build(ns), ns, True)
    Ds = annotate2(f1.sim_build(dc.epa_table(), late), ns, False)
    f1.say(f"label {dc.LABEL} seeds {dc.SEEDS}; bootstrap {NB}; real drives {len(Dr)} sim drives {len(Ds)}; games {Dr.gk.nunique()} | {Ds.gk.nunique()}")
    section_mix(Dr, Ds, ns)
    section_strategy(Dr, Ds, NB)
    f1.say(f"looks counted (interval prints and LOSO rows): {f1.LOOKS[0]}")
    (OUTD / "fpos2.txt").write_text("\n".join(f1.OUT))


if __name__ == "__main__":
    main()
