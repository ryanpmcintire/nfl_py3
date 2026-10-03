import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PBP = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
SIM = REPO / "artifacts" / "mod25e3" / "revert_sim" / "crzhc"
OUT = REPO / "artifacts" / "mod25e3" / "xq"
POOL = tuple(range(2009, 2018))
LATE = tuple(range(2018, 2026))
BURN = 2
NBOOT = 300
YL_BINS = [0, 10, 20, 35, 50, 65, 80, 101]
LEAD_EDGES = [-np.inf, -8.5, -0.5, 0.5, 8.5, np.inf]
LEAD_NAMES = ["trail>8", "trail1-8", "tied", "lead1-8", "lead>8"]
COMP = ["count", "strength", "startfp", "leadstate", "rest"]


def real_drives(seasons):
    gf = pd.read_parquet(REPO / "data" / "processed" / "game_features_pbp.parquet", columns=["game_id", "home_score", "away_score"]).set_index("game_id")
    cols = ["game_id", "play_id", "season_type", "posteam", "defteam", "home_team", "qtr", "game_seconds_remaining", "yardline_100", "play_type", "score_differential"]
    out = []
    st = []
    for s in seasons:
        p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=cols)
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
        k = p[ko]
        st.append(pd.DataFrame({"gk": k.season.astype(str) + "_" + k.game_id, "q": k.qtr.astype(int), "stm": k.dm.to_numpy()}))
        p = p.assign(kocum=np.cumsum(ko))
        p = p[~ko & p.play_type.notna()].copy()
        g = p.game_id.to_numpy()
        pos = p.posteam.to_numpy()
        kc = p.kocum.to_numpy()
        q = p.qtr.to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (pos[1:] != pos[:-1]) | (kc[1:] != kc[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
        p["d"] = np.cumsum(new)
        p["pt"] = p.play_type.where(~p.play_type.isin(["extra_point", "no_play"]))
        gp = p.groupby("d", sort=True)
        first_pos = gp.posteam.first()
        D = pd.DataFrame({"gk": gp.season.first().astype(str) + "_" + gp.game_id.first(), "season": gp.season.first(), "off": first_pos, "dfn": gp.defteam.first(),
                          "s": np.where(first_pos == gp.home_team.first(), 1.0, -1.0), "ql": gp.qtr.last(), "yl0": gp.yardline_100.first(),
                          "secs": gp.game_seconds_remaining.first() - gp.game_seconds_remaining.last(), "n": gp.size(), "dmsum": gp.dm.sum(),
                          "sd0": gp.score_differential.first(), "pt": gp.pt.last()})
        D["p"] = D.s * D.dmsum
        D["tm"] = D.season.astype(str) + "_" + D.off
        D["td"] = D.season.astype(str) + "_" + D.dfn
        out.append(D.drop(columns=["dmsum"]))
    return pd.concat(out, ignore_index=True), pd.concat(st, ignore_index=True)


def sim_drives():
    out = []
    for f in sorted(SIM.glob("play_*_*.parquet")):
        w, s = (int(x) for x in f.stem.split("_")[1:])
        if s < BURN:
            continue
        d = pd.read_parquet(f)
        d = d[d.qtr <= 4].sort_values(["g"], kind="stable").reset_index(drop=True)
        g = d.g.to_numpy()
        oh = d.offhome.to_numpy()
        q = d.qtr.to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1]) | ((q[1:] == 3) & (q[:-1] <= 2))]
        d["d"] = np.cumsum(new)
        d["pp"] = d.po - d.pdf
        d["off"] = np.where(d.offhome == 1, d.ht, d["at"])
        d["dfn"] = np.where(d.offhome == 1, d["at"], d.ht)
        gp = d.groupby("d", sort=True)
        key = w * 1000 + s
        D = pd.DataFrame({"gk": f"{key}_" + gp.g.first().astype(int).astype(str), "season": key, "off": gp.off.first(), "dfn": gp.dfn.first(),
                          "s": np.where(gp.offhome.first() == 1, 1.0, -1.0), "ql": gp.qtr.last(), "yl0": gp.yl.first(), "secs": gp.el.sum(), "n": gp.size(),
                          "sd0": gp.sd.first(), "p": gp.pp.sum(), "pt": np.where(gp.code.last() == 2, "punt", "x")})
        D["tm"] = D.season.astype(str) + "_" + D.off.astype(int).astype(str)
        D["td"] = D.season.astype(str) + "_" + D.dfn.astype(int).astype(str)
        out.append(D)
    return pd.concat(out, ignore_index=True)


def components(D):
    mu = D.p.mean()
    po = D.groupby("tm").p.agg(["sum", "count"])
    pdn = D.groupby("td").p.agg(["sum", "count"])
    gs = D.groupby(["gk", "tm"]).p.agg(["sum", "count"])
    gd = D.groupby(["gk", "td"]).p.agg(["sum", "count"])
    o_s = po.reindex(D.tm).to_numpy()
    d_s = pdn.reindex(D.td).to_numpy()
    own = gs.reindex(pd.MultiIndex.from_arrays([D.gk, D.tm])).to_numpy()
    opp = gd.reindex(pd.MultiIndex.from_arrays([D.gk, D.td])).to_numpy()
    om = (o_s[:, 0] - own[:, 0]) / np.maximum(o_s[:, 1] - own[:, 1], 1)
    dm = (d_s[:, 0] - opp[:, 0]) / np.maximum(d_s[:, 1] - opp[:, 1], 1)
    st = (om - mu) + (dm - mu)
    r1 = D.p.to_numpy() - mu - st
    fb = pd.cut(D.yl0, YL_BINS, right=False)
    fp = pd.Series(r1).groupby(fb.to_numpy(), observed=True).transform("mean").to_numpy()
    r2 = r1 - fp
    lb = pd.cut(D.sd0, LEAD_EDGES, labels=LEAD_NAMES)
    ld = pd.Series(r2).groupby(lb.to_numpy(), observed=True).transform("mean").to_numpy()
    rr = r2 - ld
    D = D.assign(c_count=np.full(len(D), mu), c_strength=st, c_startfp=fp, c_leadstate=ld, c_rest=rr, lb=lb.astype(str))
    return D, mu


def game_arrays(D):
    gk = pd.Index(sorted(D.gk.unique()))
    gi = gk.get_indexer(D.gk)
    qi = np.clip(D.ql.astype(int).to_numpy(), 1, 4) - 1
    X = np.zeros((len(gk), 4, len(COMP)))
    for k, c in enumerate(COMP):
        np.add.at(X[:, :, k], (gi, qi), D.s.to_numpy() * D["c_" + c].to_numpy())
    return X, gk


def xq(X):
    G = X.shape[0]
    Y = X.reshape(G, -1)
    C = np.cov(Y, rowvar=False).reshape(4, len(COMP), 4, len(COMP))
    R = np.zeros(len(COMP))
    for q in range(4):
        for r in range(4):
            if q != r:
                R += C[q, :, r, :].sum(axis=1)
    return R


def true_xq(X):
    I = X.sum(axis=2)
    return float(np.var(I.sum(axis=1), ddof=1) - np.var(I, axis=0, ddof=1).sum())


def table(D):
    rows = {}
    ng = D.gk.nunique()
    for nm in LEAD_NAMES:
        d = D[D.lb == nm]
        rows[nm] = dict(drives_g=len(d) / ng, yl0=d.yl0.mean(), secs=d.secs.mean(), plays=d.n.mean(), pts=d.p.mean(), strength=d.c_strength.mean(), leadstate=d.c_leadstate.mean(), rest=d.c_rest.mean(), punt=(d.pt == "punt").mean())
    return rows


def boot(X, seed):
    rng = np.random.default_rng(seed)
    G = X.shape[0]
    return np.array([xq(X[rng.integers(0, G, G)]) for _ in range(NBOOT)])


def run(name, D):
    D, mu = components(D)
    X, gk = game_arrays(D)
    R = xq(X)
    B = boot(X, 1)
    q_by = D.groupby("ql").size() / D.gk.nunique()
    lead_share = float((D.sd0 > 0).sum() / max((D.sd0 != 0).sum(), 1))
    cv = D.assign(v=D.s).groupby(["gk", "ql"]).v.sum().unstack().fillna(0)
    return dict(name=name, games=int(len(gk)), drives_g=len(D) / len(gk), mu=float(mu), R=R.tolist(), total=float(R.sum()), true_xq=true_xq(X), boot=B.tolist(),
                drives_by_q={str(k): float(v) for k, v in q_by.items()}, leader_possession_share=lead_share, table=table(D),
                cnt_var_by_q=[float(cv[q].var()) for q in cv.columns])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "show"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        res = {}
        Dp, STp = real_drives(POOL)
        res["real_pool"] = run("real_pool", Dp)
        Dl, STl = real_drives(LATE)
        res["real_late"] = run("real_late", Dl)
        res["sim"] = run("sim", sim_drives())
        for k, STm in (("real_pool", STp), ("real_late", STl)):
            I = STm.groupby(["gk", "q"]).stm.sum().unstack().fillna(0)
            res[k]["kick_margin_var_by_q"] = I.var().tolist()
        (OUT / "xq.json").write_text(json.dumps(res))
    res = json.loads((OUT / "xq.json").read_text())
    lines = []
    for k, r in res.items():
        lines.append(f"{k} games {r['games']} drives/g {r['drives_g']:.2f} mu {r['mu']:.3f} xq(sum comp) {r['total']:.2f} xq(true by drive-quarter) {r['true_xq']:.2f} leader-poss share {r['leader_possession_share']:.3f}")
        lines.append("  comp " + " ".join(f"{c}={v:+.2f}" for c, v in zip(COMP, r["R"])))
        lines.append("  drives/g by q " + " ".join(f"{q}:{v:.2f}" for q, v in r["drives_by_q"].items()) + "  count-diff var by q " + " ".join(f"{v:.2f}" for v in r["cnt_var_by_q"]))
        for nm, t in r["table"].items():
            lines.append(f"  {nm:9s} " + " ".join(f"{x}={y:.3f}" for x, y in t.items()))
    B = {k: np.array(r["boot"]) for k, r in res.items()}
    for ref in ("real_pool", "real_late"):
        lines.append(f"sim minus {ref} (comp: diff [90% boot], probability_positive)")
        for i, c in enumerate(COMP + ["total"]):
            sv, rv = B["sim"], B[ref]
            sm = sv.sum(1) if c == "total" else sv[:, i]
            rm = rv.sum(1) if c == "total" else rv[:, i]
            n = min(len(sm), len(rm))
            diff = sm[:n] - rm[:n]
            lines.append(f"  {c:10s} {sm.mean() - rm.mean():+.2f} [{np.percentile(diff, 5):+.2f},{np.percentile(diff, 95):+.2f}] pp {(diff > 0).mean():.2f}")
    (OUT / "xq.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if False:
    main()


def diag():
    sims = sim_drives()
    Dp, _ = real_drives(POOL)
    Dl, _ = real_drives(LATE)
    lines = []
    for nm, D in (("real_pool", Dp), ("real_late", Dl), ("sim", sims)):
        D, _ = components(D)
        gk = pd.Index(sorted(D.gk.unique()))
        gi = gk.get_indexer(D.gk)
        qi = np.clip(D.ql.astype(int).to_numpy(), 1, 4) - 1
        H = np.zeros((len(gk), 4))
        A = np.zeros((len(gk), 4))
        h = (D.s > 0).to_numpy()
        r = D.c_rest.to_numpy()
        np.add.at(H, (gi[h], qi[h]), r[h])
        np.add.at(A, (gi[~h], qi[~h]), -r[~h])
        Y = np.hstack([H, A])
        C = np.cov(Y, rowvar=False)
        def pair(a, b):
            return sum(C[a * 4 + q, b * 4 + r2] for q in range(4) for r2 in range(4) if q != r2)
        hh, aa, ha, ah = pair(0, 0), pair(1, 1), pair(0, 1), pair(1, 0)
        adj = sum(C[a * 4 + q, b * 4 + q + 1] + C[a * 4 + q + 1, b * 4 + q] for a in (0, 1) for b in (0, 1) for q in range(3))
        lines.append(f"{nm} rest xq: same-team {hh + aa:+.2f} (home {hh:+.2f} away {aa:+.2f}) opposite-team {ha + ah:+.2f} total {hh + aa + ha + ah:+.2f}; adjacent-quarter pairs {adj:+.2f}")
        Z = np.cov((H + A).T)
        lines.append("  rest quarter increment cov matrix off-diag: " + " ".join(f"{q + 1}{r2 + 1}:{Z[q, r2]:+.2f}" for q in range(4) for r2 in range(q + 1, 4)))
        for lab, m in (("sd0<=-9", D.sd0 <= -9), ("sd0>=9", D.sd0 >= 9), ("|sd0|<9", D.sd0.abs() < 9)):
            d = D[m]
            sl = np.polyfit(d.sd0.to_numpy(float), d.c_rest.to_numpy(), 1)[0]
            lines.append(f"  slope of drive rest on offense lead {lab}: {sl:+.4f} per point (n {len(d)})")
        for q in (1, 2, 3, 4):
            d = D[D.ql == q]
            sl = np.polyfit(d.sd0.to_numpy(float), (d.c_rest + d.c_leadstate).to_numpy(), 1)[0]
            lines.append(f"  q{q} slope of (rest+leadstate) on lead {sl:+.4f}")
    (OUT / "xq_diag.txt").write_text("\n".join(lines))
    print("\n".join(lines))




def common():
    sims = sim_drives()
    Dp, _ = real_drives(POOL)
    Dl, _ = real_drives(LATE)
    lines = []
    rng = np.random.default_rng(3)
    res = {}
    for nm, D in (("real_pool", Dp), ("real_late", Dl), ("sim", sims)):
        D, _ = components(D)
        D = D.assign(side=np.where(D.s > 0, "h", "a"))
        pv = D.groupby(["gk", "side"]).c_rest.sum().unstack().fillna(0)
        nv = D.groupby(["gk", "side"]).size().unstack().fillna(0)
        hm, am = pv.h.to_numpy() / nv.h.to_numpy(), pv.a.to_numpy() / nv.a.to_numpy()
        hq = D[D.s > 0].groupby(["gk", "ql"]).c_rest.sum().unstack().fillna(0)
        aq = D[D.s < 0].groupby(["gk", "ql"]).c_rest.sum().unstack().fillna(0).reindex(hq.index).fillna(0)
        def opp_cross(idx):
            h = hq.to_numpy()[idx]
            a = aq.to_numpy()[idx]
            c = np.cov(np.hstack([h, a]), rowvar=False)
            return sum(c[q, 4 + r] + c[4 + q, r] for q in range(4) for r in range(4) if q != r)
        G = len(hq)
        base = opp_cross(np.arange(G))
        bs = np.array([opp_cross(rng.integers(0, G, G)) for _ in range(200)])
        res[nm] = (base, bs)
        corr = np.corrcoef(hm, am)[0, 1]
        tot = (pv.h + pv.a).to_numpy()
        lines.append(f"{nm}: games {G} corr(home offense mean rest, away offense mean rest) {corr:+.3f}; var of game total rest {tot.var():.1f}; opposite-team cross-quarter cov of rest (sign: margin units, offense-credit) {base:+.2f}")
    for ref in ("real_pool", "real_late"):
        d = res["sim"][1][:200] - res[ref][1][:200]
        lines.append(f"sim minus {ref} opposite-team rest cov: {res['sim'][0] - res[ref][0]:+.2f} [{np.percentile(d, 5):+.2f},{np.percentile(d, 95):+.2f}] pp {(d > 0).mean():.2f}")
    (OUT / "xq_common.txt").write_text("\n".join(lines))
    print("\n".join(lines))


common()
