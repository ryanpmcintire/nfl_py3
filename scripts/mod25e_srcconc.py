import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
sys.argv = sys.argv[:1] + [str(NB)]
import mod25d_variance as dv  # noqa: E402
import mod25e_catchup as cu  # noqa: E402
import mod25e_late as late  # noqa: E402
import mod25e_persist as P  # noqa: E402
import sim04_engine as sim  # noqa: E402

OUTD = REPO / "artifacts" / "mod25e3" / "srcconc"
OUT = []
rng = np.random.default_rng(60)
NQ = 3000
MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)


def say(s=""):
    print(s)
    OUT.append(s)


def iv(d, pt):
    return f"{pt:+.3f} [{np.percentile(d, 5):+.3f},{np.percentile(d, 95):+.3f}] pp {float((d > 0).mean()):.2f}"


def engine():
    cap = {}
    o_load, o_tr = sim.load_reg_seasons, sim.build_transition_frame

    def lh(s):
        p = o_load(s)
        cap["pbp"] = p
        return p

    def th(pbp, team_ratings=None):
        t = o_tr(pbp, team_ratings=team_ratings)
        cap["trans"] = t
        return t

    sim.load_reg_seasons, sim.build_transition_frame = lh, th
    try:
        tables = sim.build_tables(tuple(dv.TRAIN), condition_on_team=True)
    finally:
        sim.load_reg_seasons, sim.build_transition_frame = o_load, o_tr
    return tables, cap["trans"], cap["pbp"]


def ess(codes):
    c = np.bincount(codes)
    p = c[c > 0] / c.sum()
    return 1.0 / float((p * p).sum())


def source_tables(tables, trans, pbp):
    a = tables["arrays"]
    gcode, gnames = pd.factorize(trans["game_id"].to_numpy())
    meta = pbp.groupby("game_id", sort=False).agg(season=("season", "first"), ht=("home_team", "first"), aw=("away_team", "first")).reindex(gnames)
    home = a["is_home_off"].astype(bool)
    ht, at = meta.ht.to_numpy()[gcode], meta.aw.to_numpy()[gcode]
    season = meta.season.to_numpy()[gcode].astype(int)
    off_team = np.where(home, ht, at)
    def_team = np.where(home, at, ht)
    ots = pd.factorize(pd.Series(season.astype(str)) + "_" + pd.Series(off_team.astype(str)))[0]
    dts = pd.factorize(pd.Series(season.astype(str)) + "_" + pd.Series(def_team.astype(str)))[0]
    right = pbp[["game_id", "play_id", "epa"]].drop_duplicates(["game_id", "play_id"])
    epa = trans[["game_id", "play_id"]].merge(right, on=["game_id", "play_id"], how="left")["epa"].fillna(0.0).to_numpy(dtype=np.float64)
    ok = np.isin(a["play_type_code"], (0, 1))
    df = pd.DataFrame({"g": gcode, "h": home, "s": season, "o": off_team, "d": def_team, "e": epa, "rt": a["off_row"]})[ok]
    gs = df.groupby(["g", "h"]).agg(m=("e", "mean"), n=("e", "size"), s=("s", "first"), o=("o", "first"), d=("d", "first"), rt=("rt", "first")).reset_index()
    league = gs.m.mean()
    for key, nm in ((["s", "o"], "own"), (["s", "d"], "alw")):
        sm = gs.groupby(key).m.transform("sum")
        ct = gs.groupby(key).m.transform("count")
        gs[nm] = ((sm - gs.m) / (ct - 1).where(ct > 1)).fillna(league)
    gs["r"] = gs.m - gs.own - (gs.alw - league)
    gl = gs.set_index(["g", "h"]).reindex(pd.MultiIndex.from_arrays([gcode, home]))
    adj = (gl.own + gl.alw - league).to_numpy()
    n, m = gl.n.to_numpy(), gl.m.to_numpy()
    loo = np.where(ok & (n > 1), (m * n - epa) / np.maximum(n - 1, 1), m)
    rv = np.nan_to_num(loo - adj, nan=0.0)
    return gcode, ots, dts, rv, gs, ots


def sim_blocks(gcode, ots, dts, rv):
    rows = []
    for sd in cu.SEEDS:
        sdir = cu.ART / f"e5_{cu.LABEL}_s{sd}"
        for f in sorted(sdir.glob("play_*_*.parquet")):
            w, s = (int(x) for x in f.stem.split("_")[1:])
            if s < late.BURN:
                continue
            d = pd.read_parquet(f, columns=["g", "qtr", "offhome", "idx", "down"])
            d = d[(d.qtr <= 4) & d.idx.notna()].reset_index(drop=True)
            ix = d.idx.to_numpy().astype(int)
            perm = ix.copy()
            for _, grp in d.groupby("down").groups.items():
                pos = np.asarray(list(grp))
                perm[pos] = ix[rng.permutation(pos)]
            for tag, src in (("real", ix), ("perm", perm)):
                d[f"gc_{tag}"] = gcode[src]
                d[f"os_{tag}"] = ots[src]
                d[f"ds_{tag}"] = dts[src]
            d["v"] = rv[ix]
            for (g, oh), b in d.groupby(["g", "offhome"]):
                r = {"gk": f"{sd}_{w}_{s}_{int(g)}", "oh": int(oh), "n": len(b), "v": b.v.mean(), "v1": b.v[b.qtr == 1].mean()}
                for tag in ("real", "perm"):
                    r[f"gess_{tag}"] = ess(b[f"gc_{tag}"].to_numpy())
                    r[f"oess_{tag}"] = ess(b[f"os_{tag}"].to_numpy())
                    r[f"dess_{tag}"] = ess(b[f"ds_{tag}"].to_numpy())
                    r[f"gn_{tag}"] = b[f"gc_{tag}"].nunique()
                rows.append(r)
    return pd.DataFrame(rows)


def section_ess(B):
    say("## S1. effective number of distinct sources behind one sim team-game's offensive plays (1/sum p^2 over source game or source offence/defence team-season); reference = same plays with source ids permuted among plays of the same down within a world-season (breaks the team-game link, keeps the marginal)")
    say(f"  sim team-game blocks {len(B)}, mean plays per block {B.n.mean():.1f}")
    games = B.gk.unique()
    bg = pd.Series(np.arange(len(games)), index=games).reindex(B.gk).to_numpy()
    for nm, c in (("ESS source games", "gess"), ("ESS source offence team-seasons", "oess"), ("ESS source defence team-seasons", "dess"), ("distinct source games", "gn")):
        x, y = B[f"{c}_real"].to_numpy(), B[f"{c}_perm"].to_numpy()
        d = []
        for _ in range(NB):
            w = np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games))[bg]
            d.append((w * (x - y)).sum() / w.sum())
        say(f"  {nm}: sim {x.mean():.2f} (median {np.median(x):.2f}), reference {y.mean():.2f}, sim-ref {iv(np.array(d), x.mean() - y.mean())}")


def section_kernel(tables, trans, gs, ots):
    say("## S3. team kernel width and draw-weight concentration (read: sim04_engine.py:31-34 constants, :778 h = H_SCALE x sd of per-game-side rating over training window, :515-523 weights over the K_STATE nearest state rows with same-home gate x LAMBDA)")
    a = tables["arrays"]
    h = tables["team_kernel_h"]
    say(f"  h {h:.4f} EPA/play = {sim.TEAM_KERNEL_H_SCALE} x sd(game-side off rating) {h / sim.TEAM_KERNEL_H_SCALE:.4f}; row sd off {float(np.nanstd(a['off_row'])):.4f}, def {float(np.nanstd(a['def_row'])):.4f}")
    g = gs.assign(ts=pd.factorize(gs.s.astype(str) + "_" + gs.o.astype(str))[0])
    tm = g.groupby("ts").rt.agg(["mean", "std", "count"])
    tm = tm[tm["count"] >= 8]
    say(f"  sd of team-season mean off rating {tm['mean'].std():.4f}; mean within-team-season sd across games {tm['std'].mean():.4f}; h / team-season sd {h / tm['mean'].std():.2f}; h / within-team sd {h / tm['std'].mean():.2f}")
    say(f"  corr(game-side rating, that game-side's realised mean EPA) {np.corrcoef(g.rt, g.m)[0, 1]:+.3f}; corr(rating, real residual r) {np.corrcoef(g.rt, g.r)[0, 1]:+.3f}")
    say(f"  fitted? H_SCALE {sim.TEAM_KERNEL_H_SCALE} and LAMBDA {sim.TEAM_KERNEL_LAMBDA:g} carry no held-out likelihood fit in the repo; docs/sim04_unit_log.md:2179-2190 holds only a bandwidth sweep (h 0.10 vs 0.5) judged on home edge and backup-QB shift; LAMBDA is a hard same-home gate (weight 1e6), unfitted")
    gid = trans["game_id"].to_numpy()
    cols = ["dist_raw", "fp_raw", "sc_raw", "time_raw", "off_to_raw", "def_to_raw", "phase"]
    qs = rng.choice(len(trans), NQ, replace=False)
    rs = rng.choice(len(trans), NQ)
    res = {m: [] for m in MULTS}
    top = {m: [] for m in MULTS}
    uni = []
    for q, r in zip(qs, rs):
        down = int(a["down_i"][q])
        ph = int(trans["phase"].iat[q])
        tree, sub = tables["nn_trees_cond"].get((down, ph)) or tables["nn_trees_cond"][(down, 0)]
        f = sim.feature_matrix(*[np.array([trans[c].iat[q]]) for c in cols])[0]
        _, ind = tree.query(f, k=min(sim.K_STATE, len(sub)))
        nb = sub[np.atleast_1d(ind)]
        d2 = (a["off_row"][nb] - a["off_row"][r]) ** 2 + (a["def_row"][nb] - a["def_row"][r]) ** 2
        gate = np.where(a["is_home_off"][nb] == a["is_home_off"][r], sim.TEAM_KERNEL_LAMBDA, 1.0)
        codes = pd.factorize(gid[nb])[0]
        uni.append(ess(codes))
        for m in MULTS:
            w = np.exp(-d2 / (2.0 * (h * m) ** 2)) * gate
            p = np.bincount(codes, weights=w)
            p = p[p > 0] / p.sum()
            res[m].append(1.0 / float((p * p).sum()))
            top[m].append(float(p.max()))
    say(f"  {NQ} real states with random sim ratings, K_STATE {sim.K_STATE} nearest rows: distinct-game ESS with uniform weights {np.mean(uni):.1f}")
    for m in MULTS:
        say(f"  h x {m:g}: ESS source games per draw mean {np.mean(res[m]):.1f} median {np.median(res[m]):.1f} p10 {np.percentile(res[m], 10):.1f}; heaviest single source game share mean {np.mean(top[m]):.3f}")


def section_reg(B):
    ns = late.load_xq()
    _, simd = cu.drives_all(ns)
    Ds, Gs = cu.prep(ns, simd)
    gkk = sorted(Ds.gk.unique())
    I, IBS, IS, abar = cu.cross_terms(Ds, Gs, np.ones((1, Gs)))
    gi, qi, bi = Ds.gi.to_numpy(), Ds.qi.to_numpy(), Ds.bi.to_numpy()
    e = Ds.a.to_numpy() - abar[0][qi * 5 + bi]
    home = Ds.s.to_numpy() > 0
    EH = np.zeros((Gs, 4))
    EA = np.zeros((Gs, 4))
    np.add.at(EH, (gi[home], qi[home]), e[home])
    np.add.at(EA, (gi[~home], qi[~home]), e[~home])
    Hr = EH - EA
    X, M = P.game_table(gkk, cu.SEEDS)
    pv = B.pivot_table(index="gk", columns="oh", values="v").reindex(gkk)
    fh, fa = pv[1].fillna(0.0).to_numpy(), pv[0].fillna(0.0).to_numpy()
    f = fh - fa
    pv1 = B.pivot_table(index="gk", columns="oh", values="v1").reindex(gkk)
    f1h, f1a = pv1[1].fillna(0.0).to_numpy(), pv1[0].fillna(0.0).to_numpy()
    f1 = f1h - f1a
    say("## S2. does the mean real game-level residual of a sim game's source plays explain its persistent component; real residual = game-side offence EPA/play excluding the drawn play itself, less leave-game-out own team-season mean less opponent-defence leave-game-out allowed (run/pass plays); f = mean over home-offence source plays minus mean over away-offence source plays")
    say(f"  sim games {Gs}; sd f {f.std():.4f}; corr(f, final home margin) {np.corrcoef(f, M)[0, 1]:+.3f}; corr(f, first-half home net residual) {np.corrcoef(f, Hr[:, 0] + Hr[:, 1])[0, 1]:+.3f}")

    def contrib(idx, extra):
        Xa = np.c_[X, extra]
        out = {"total": 0.0, "rec": 0.0, "new": 0.0}
        for r in range(1, 4):
            Xc = np.c_[np.ones(len(idx)), Xa[idx]]
            beta, *_ = np.linalg.lstsq(Xc, Hr[idx, r], rcond=None)
            fr = X[idx] @ beta[1:11]
            fn = Xa[idx][:, 10:] @ beta[11:]
            for q in range(r):
                out["total"] += 2 * np.cov(I[idx, q], Hr[idx, r])[0, 1]
                out["rec"] += 2 * np.cov(I[idx, q], fr)[0, 1]
                out["new"] += 2 * np.cov(I[idx, q], fn)[0, 1]
        out["ub"] = out["total"] - out["rec"]
        out["ua"] = out["ub"] - out["new"]
        return out

    say(f"  f1 = same from first-quarter source plays only (state near tied, so not a function of the game's later path): sd {f1.std():.4f}; corr(f1, final home margin) {np.corrcoef(f1, M)[0, 1]:+.3f}; corr(f1, Q1 home net residual) {np.corrcoef(f1, Hr[:, 0])[0, 1]:+.3f}; corr(f1, Q2-Q4 home net residual) {np.corrcoef(f1, Hr[:, 1:].sum(1))[0, 1]:+.3f}")
    for nm, ex in (("f all quarters (mediated by state: source residual rises with the drawn state)", f[:, None]), ("f1 first quarter", f1[:, None]), ("f1_home_off and f1_away_off", np.c_[f1h, f1a]), ("permuted f1 (noise column)", rng.permutation(f1)[:, None])):
        pt = contrib(np.arange(Gs), ex)
        bs = [contrib(rng.integers(0, Gs, Gs), ex) for _ in range(NB)]
        say(f"  {nm}: total E {pt['total']:+.2f}; recorded-10 {pt['rec']:+.2f}; unexplained before {pt['ub']:+.2f} [{np.percentile([b['ub'] for b in bs], 5):+.2f},{np.percentile([b['ub'] for b in bs], 95):+.2f}]")
        say(f"    incremental {iv(np.array([b['new'] for b in bs]), pt['new'])}; share of unexplained {iv(np.array([b['new'] / b['ub'] for b in bs]), pt['new'] / pt['ub'])}; unexplained after {pt['ua']:+.2f}")
    say("  in-sample increments over the 10 recorded regressors; the permuted row shows what a noise column absorbs")


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    tables, trans, pbp = engine()
    a = tables["arrays"]
    gcode, ots, dts, rv, gs, ots = source_tables(tables, trans, pbp)
    chk = pd.read_parquet(cu.ART / f"e5_{cu.LABEL}_s{cu.SEEDS[0]}" / "play_0_5.parquet", columns=["down", "idx"]).dropna()
    ixc = chk.idx.to_numpy().astype(int)
    say(f"## S0. alignment: play-file down equals pool down_i[idx] on {float((chk.down.to_numpy() == a['down_i'][ixc]).mean()):.4f} of {len(chk)} plays; pool rows {len(trans)} (idx max {ixc.max()}), pool game-sides {len(gs)}")
    B = sim_blocks(gcode, ots, dts, rv)
    section_ess(B)
    section_reg(B)
    section_kernel(tables, trans, gs, ots)
    say(f"looks: S1 4 rows, S2 3 variants x 2 interval prints, S3 diagnostic sweep of {len(MULTS)} widths; bootstrap {NB}")
    (OUTD / "srcconc.txt").write_text("\n".join(OUT), encoding="utf-8")


if __name__ == "__main__":
    main()
