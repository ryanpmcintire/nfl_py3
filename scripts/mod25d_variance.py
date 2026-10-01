import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25c_noise as c25  # noqa: E402
import mod25_mechanisms as m25  # noqa: E402
import sim04_engine as sim  # noqa: E402

OUT = REPO / "artifacts" / "mod25d"
TRAIN = c25.TRAIN
EVAL = c25.EVAL
_G = c25._G


def real_load(seasons):
    pbp = sim.load_reg_seasons(tuple(seasons))
    tr = sim.build_transition_frame(pbp)
    at = c25.attrs_from(pbp, tr)
    tr = tr.assign(tov=at["tov"], epa=at["epa"], yards=at["yards"]).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    gid, gids = pd.factorize(tr["game_id"])
    P = pd.DataFrame(
        {
            "g": gid, "down": tr["down_i"].to_numpy(), "dist": tr["dist_raw"].to_numpy(), "yl": tr["fp_raw"].to_numpy(),
            "sd": tr["sc_raw"].to_numpy(), "gsr": tr["gsr_actual"].to_numpy(), "qtr": tr["qtr_actual"].to_numpy(),
            "code": tr["play_type_code"].to_numpy(), "po": tr["points_off"].to_numpy(), "pdf": tr["points_def"].to_numpy(),
            "flip": tr["possession_flip"].to_numpy().astype(bool), "el": tr["clock_elapsed"].to_numpy(),
            "epa": tr["epa"].to_numpy(), "yards": tr["yards"].to_numpy(),
        }
    )
    last = tr.groupby(gid).tail(1)
    M = pd.Series(last["home_margin_post"].to_numpy(), index=gid[last.index.to_numpy()])
    meta = pbp.drop_duplicates("game_id").set_index("game_id").loc[gids, ["season", "home_team", "away_team"]].reset_index(drop=True)
    return P, M, meta, pbp


def assign_sides(P):
    g = P["g"].to_numpy()
    qtr = P["qtr"].to_numpy()
    flip = P["flip"].to_numpy()
    n = len(P)
    first = np.r_[True, g[1:] != g[:-1]]
    prev_flip = np.r_[False, flip[:-1]] & ~first
    prev_q = np.r_[0, qtr[:-1]]
    half = (qtr == 3) & (prev_q <= 2) & ~first
    tog = prev_flip & ~half
    ev = first | prev_flip | half
    gstart = np.flatnonzero(first)
    gidx = np.cumsum(first) - 1
    T = np.cumsum(tog)
    Tg = T - T[gstart][gidx]
    Th = np.where(half, Tg, -1)
    Th_game = pd.Series(Th).where(Th >= 0).groupby(gidx).transform("max").to_numpy()
    second = ~np.isnan(Th_game) & (qtr >= 3)
    side = np.where(second, (1 + Tg - np.nan_to_num(Th_game)).astype(np.int64) % 2, Tg % 2)
    poss = np.cumsum(ev) - 1
    return side.astype(np.int8), poss


def poss_analysis(P, M):
    P = P.sort_values("g", kind="stable").reset_index(drop=True)
    reg = P["qtr"].to_numpy() <= 4
    side, poss = assign_sides(P)
    P = P.assign(side=side, poss=poss)
    R = P[reg]
    gp = R.groupby("poss", sort=True)
    pk = pd.DataFrame(
        {
            "g": gp["g"].first(), "side": gp["side"].first(), "po": gp["po"].sum(), "pdf": gp["pdf"].sum(),
            "n": gp.size(), "sd0": gp["sd"].first(), "q0": gp["qtr"].first(), "gsr0": gp["gsr"].first(),
            "secs": gp["el"].sum(),
        }
    ).reset_index()
    pk["y"] = pk["po"] - pk["pdf"]
    pk["sA"] = np.where(pk["side"] == 0, pk["y"], -pk["y"])
    pk["k"] = pk.groupby("g").cumcount()
    ng = pk["g"].nunique()
    out = {}
    cnt = pk.groupby("g").size()
    out["poss_per_game_mean"] = float(cnt.mean())
    out["poss_per_game_var"] = float(cnt.var())
    cA = pk[pk["side"] == 0].groupby("g").size().reindex(cnt.index).fillna(0)
    cB = pk[pk["side"] == 1].groupby("g").size().reindex(cnt.index).fillna(0)
    out["poss_per_team_mean"] = float((cA.mean() + cB.mean()) / 2)
    out["poss_per_team_var"] = float((cA.var() + cB.var()) / 2)
    out["poss_count_corr_AB"] = float(np.corrcoef(cA, cB)[0, 1])
    po = pk["po"].to_numpy()
    out["pts_per_poss_mean"] = float(po.mean())
    out["pts_per_poss_var"] = float(po.var())
    for v in (0, 2, 3, 6, 7, 8):
        out[f"share_pts_{v}"] = float(np.mean(po == v))
    out["share_pts_other"] = float(np.mean(~np.isin(po, (0, 2, 3, 6, 7, 8))))
    out["def_score_share"] = float(np.mean(pk["pdf"].to_numpy() > 0))
    out["td_share"] = float(np.mean(po >= 6))
    out["fg_share"] = float(np.mean(po == 3))
    pk = pk.sort_values(["g", "k"]).reset_index(drop=True)
    y = pk["y"].to_numpy()
    gg = pk["g"].to_numpy()
    for lag, nm in ((1, "across"), (2, "same")):
        ok = gg[lag:] == gg[:-lag]
        a, b = y[:-lag][ok], y[lag:][ok]
        out[f"corr_{nm}_lag{lag}"] = float(np.corrcoef(a, b)[0, 1])
    sA = pk.groupby("g")["sA"].sum()
    out["marginA_var_reg"] = float(sA.var())
    out["marginA_mean_reg"] = float(sA.mean())
    nA = cnt.to_numpy().astype(float)
    sk = pk["sA"].to_numpy()
    out["signed_poss_mean"] = float(sk.mean())
    out["signed_poss_var"] = float(sk.var())
    out["iid_margin_var"] = float(nA.mean() * sk.var() + nA.var() * sk.mean() ** 2)
    tot_pts = pk.groupby("g")["po"].sum() + pk.groupby("g")["pdf"].sum()
    out["cov_nposs_totpts"] = float(np.cov(cnt.to_numpy(), tot_pts.reindex(cnt.index).to_numpy())[0, 1])
    out["corr_nposs_totpts"] = float(np.corrcoef(cnt.to_numpy(), tot_pts.reindex(cnt.index).to_numpy())[0, 1])
    out["totpts_var"] = float(tot_pts.var())
    out["totpts_mean"] = float(tot_pts.mean())
    pk["q4late"] = (pk["q0"] == 4) & (pk["gsr0"] <= 900)
    out["plays_per_poss"] = float(pk["n"].mean())
    out["secs_per_poss"] = float(pk["secs"].mean())
    for nm, m_ in (("all", np.ones(len(pk), bool)), ("q1q3", pk["q0"].isin([1, 2, 3]).to_numpy()), ("q4", (pk["q0"] == 4).to_numpy())):
        d = pk[m_]
        out[f"fb_corr_sd0_y_{nm}"] = float(np.corrcoef(d["sd0"], d["y"])[0, 1])
        out[f"fb_slope_sd0_y_{nm}"] = float(np.polyfit(d["sd0"], d["y"], 1)[0])
        out[f"fb_slope_sd0_secs_{nm}"] = float(np.polyfit(d["sd0"], d["secs"], 1)[0])
        out[f"fb_slope_sd0_plays_{nm}"] = float(np.polyfit(d["sd0"], d["n"], 1)[0])
    pl = P[reg]
    sc = pl.assign(sdA=np.where(pl["side"] == 0, pl["sd"], -pl["sd"]))
    fin = sA.reindex(np.arange(P["g"].max() + 1)).fillna(0.0)
    for t in (2700, 1800, 900, 300):
        f = sc[sc["gsr"] <= t].groupby("g").head(1)
        fut = fin.reindex(f["g"].to_numpy()).to_numpy() - f["sdA"].to_numpy()
        out[f"fb_corr_diff_future_t{t}"] = float(np.corrcoef(f["sdA"], fut)[0, 1])
        out[f"fb_slope_diff_future_t{t}"] = float(np.polyfit(f["sdA"], fut, 1)[0])
    if M is not None:
        Mv = M.reindex(sA.index).to_numpy()
        out["frac_side_margin_match"] = float(np.mean(np.abs(Mv) == np.abs(sA.to_numpy())))
        out["margin_var_all"] = float(np.var(M.to_numpy()))
    out["n_games"] = int(ng)
    return out, pk


def lag_analysis(pk, P):
    pk = pk.sort_values(["g", "k"]).reset_index(drop=True)
    s = pk["sA"].to_numpy()
    g = pk["g"].to_numpy()
    ng = len(np.unique(g))
    out = {}
    for L in range(1, 22):
        ok = g[L:] == g[:-L]
        out[f"cov_lag{L}"] = 2.0 * float((s[:-L][ok] * s[L:][ok]).sum()) / ng
    c = lambda a, b: sum(out[f"cov_lag{L}"] for L in range(a, b))
    out["cov_lag2_4"] = c(2, 5)
    out["cov_lag5_10"] = c(5, 11)
    out["cov_lag11_21"] = c(11, 22)
    out["cov_total"] = c(1, 22)
    out["cov_same_side_even"] = sum(out[f"cov_lag{L}"] for L in range(2, 22, 2))
    out["cov_opp_side_odd"] = sum(out[f"cov_lag{L}"] for L in range(1, 22, 2))
    side = pk["side"].to_numpy()
    q = pk["q0"].to_numpy()
    sg = np.where(side == 0, pk["po"], pk["pdf"]).astype(float)
    sb = np.where(side == 1, pk["po"], pk["pdf"]).astype(float)
    dfq = pd.DataFrame({"g": g, "q": q, "a": sg, "b": sb}).groupby(["g", "q"])[["a", "b"]].sum().unstack("q").fillna(0.0)
    A = dfq["a"].sum(axis=1)
    B = dfq["b"].sum(axis=1)
    out["var_A_pts"] = float(A.var())
    out["var_B_pts"] = float(B.var())
    out["cov_AB_pts"] = float(np.cov(A, B)[0, 1])
    out["corr_AB_pts"] = float(np.corrcoef(A, B)[0, 1])
    m = dfq["a"] - dfq["b"]
    out["q_margin_var"] = [float(m[c_].var()) for c_ in m.columns]
    out["q_margin_cov_matrix"] = np.cov(m.to_numpy().T).tolist()
    out["q_total_var"] = [float((dfq["a"][c_] + dfq["b"][c_]).var()) for c_ in m.columns]
    return out


def random_effects(meta, M, iters=300):
    d = meta.copy()
    d["m"] = M.reindex(np.arange(len(meta))).to_numpy()
    d["h"] = d["season"].astype(str) + "_" + d["home_team"]
    d["a"] = d["season"].astype(str) + "_" + d["away_team"]
    teams = pd.Index(sorted(set(d["h"]) | set(d["a"])))
    n, p = len(d), len(teams)
    X = np.zeros((n, p))
    X[np.arange(n), teams.get_indexer(d["h"])] = 1.0
    X[np.arange(n), teams.get_indexer(d["a"])] -= 1.0
    X = np.c_[np.ones(n), X]
    y = d["m"].to_numpy()
    se2, sa2 = float(y.var()) / 2, 10.0
    for _ in range(iters):
        lam = se2 / sa2
        A = X.T @ X
        A[1:, 1:] += lam * np.eye(p)
        Ainv = np.linalg.inv(A)
        beta = Ainv @ X.T @ y
        r = y - X @ beta
        edf = np.trace(X @ Ainv @ X.T)
        se2_n = float(r @ r) / max(n - edf, 1.0)
        post = se2 * np.trace(Ainv[1:, 1:])
        sa2_n = float((beta[1:] @ beta[1:]) + post) / p
        if abs(se2_n - se2) < 1e-6 and abs(sa2_n - sa2) < 1e-6:
            se2, sa2 = se2_n, sa2_n
            break
        se2, sa2 = se2_n, sa2_n
    return {"hfa": float(beta[0]), "sigma_strength_var_per_team": float(sa2), "strength_diff_var": float(2 * sa2), "noise_var": float(se2), "total_var": float(y.var()), "implied_total": float(2 * sa2 + se2), "n_games": int(n), "n_teams": int(p)}


def team_effects_plays(pbp, seasons, col):
    d = pbp[pbp["season"].isin(seasons) & pbp["play_type"].isin(["run", "pass"]) & pbp[col].notna()].copy()
    d = d[d["posteam"].notna() & d["defteam"].notna()]
    d["o"] = d["season"].astype(str) + "_" + d["posteam"]
    d["f"] = d["season"].astype(str) + "_" + d["defteam"]
    return d


def fit_play_effects(d, col, iters=40):
    y = d[col].to_numpy(dtype=np.float64)
    mu = y.mean()
    oc, ou = pd.factorize(d["o"])
    fc, fu = pd.factorize(d["f"])
    yc = y - mu
    var_tot = yc.var()
    eo = np.zeros(len(ou))
    ef = np.zeros(len(fu))
    no = np.bincount(oc, minlength=len(ou)).astype(float)
    nf = np.bincount(fc, minlength=len(fu)).astype(float)
    so2, sf2 = 0.01 * var_tot, 0.01 * var_tot
    se2 = var_tot
    for _ in range(iters):
        ro = yc - ef[fc]
        eo = np.bincount(oc, weights=ro, minlength=len(ou)) / (no + se2 / so2)
        rf = yc - eo[oc]
        ef = np.bincount(fc, weights=rf, minlength=len(fu)) / (nf + se2 / sf2)
        res = yc - eo[oc] - ef[fc]
        se2 = float(res.var())
        so2 = max(float((eo ** 2).mean() + (se2 / (no + se2 / so2)).mean()), 1e-9)
        sf2 = max(float((ef ** 2).mean() + (se2 / (nf + se2 / sf2)).mean()), 1e-9)
    return {"mu": mu, "var_raw": float(var_tot), "var_resid": float(res.var()), "share_explained": float(1 - res.var() / var_tot), "off_sd": float(np.sqrt(so2)), "def_sd": float(np.sqrt(sf2)), "n": int(len(y))}, (oc, fc, eo, ef, ou, fu, mu)


def d_batch(task):
    seed, n = task
    rng = np.random.default_rng(seed)
    t = _G["tables"]
    ns = _G["ns"]
    tov = t["attrs"]["tov"]
    to_arr = _G["to_arr"]
    rows, margins = [], []
    ep, yd = t["attrs"]["epa"], t["attrs"]["yards"]
    hr = ar = None
    if _G["cfg"].get("avg"):
        import mod25_generator as gen

        lo, ld = gen.league_means()
        hr = {"off": gen.quant(lo), "def": gen.quant(ld)}
        ar = dict(hr)
    for gi in range(n):
        _G["log"].clear()
        ns["PLAYLOG"].clear()
        state = sim.initial_kickoff_state(rng, t["opening_pool"])
        rec, _ = ns["run_one_game"](state, t, rng, 600.0, _G["pol"], sim.K_NEIGHBORS, sim.MAX_PLAYS_PER_GAME, hr, ar)
        a = np.array(_G["log"], dtype=np.float64).reshape(-1, 12)
        ix = a[:, 11].astype(np.int64)
        rows.append(np.column_stack([np.full(len(a), gi), a[:, :11], tov[ix], to_arr[ix], ep[ix], yd[ix]]))
        margins.append(rec["margin"])
    return seed, np.concatenate(rows), np.array(margins)


DCOLN = c25.COLN + ["epa", "yards"]


def run_neutral(variant, games, workers, seed):
    import multiprocessing as mp

    c25.ensure_policies()
    cfg = dict(DV[variant], seed=seed)
    if cfg.get("resid"):
        cfg.update(DEC_COND)
    per = max(50, games // (workers * 4))
    tasks = [(seed * 100000 + i, per) for i in range((games + per - 1) // per)]
    with mp.get_context("spawn").Pool(workers, initializer=d_init, initargs=(TRAIN, json.dumps(cfg))) as pool:
        res = pool.map(d_batch, tasks, chunksize=1)
    frames, ms, off = [], [], 0
    for _, a, m in res:
        d = pd.DataFrame(a, columns=DCOLN)
        d["g"] = d["g"].astype(np.int64) + off
        d["code"] = d["code"].astype(int)
        d["flip"] = d["flip"].astype(bool)
        frames.append(d)
        ms.append(pd.Series(m, index=np.arange(len(m)) + off))
        off += len(m)
    return pd.concat(frames, ignore_index=True), pd.concat(ms)


def cmd_decomp(args):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    t0 = time.time()
    for nm, ss in (("real_train", TRAIN), ("real_eval", EVAL)):
        P, M, meta, pbp = real_load(ss)
        r, pk = poss_analysis(P, M)
        r["lags"] = lag_analysis(pk, P)
        r["RE"] = random_effects(meta, M)
        pl = P[P["code"].isin([0, 1])]
        r["play_epa_var_real"] = float(pl["epa"].var())
        r["play_yards_var_real"] = float(pl["yards"].var())
        for col in ("epa", "yards_gained"):
            d = team_effects_plays(pbp, list(ss), col)
            r["play_effects_" + col], _ = fit_play_effects(d, col)
        res[nm] = r
        print(nm, f"{time.time() - t0:.0f}s", flush=True)
    for v in args.variants.split(","):
        cf = Path(args.cache) / f"sim_{v}.parquet"
        if args.skip_sim and cf.exists():
            S = pd.read_parquet(cf)
            MS = pd.Series(pd.read_parquet(Path(args.cache) / f"simM_{v}.parquet")["m"].to_numpy())
        else:
            S, MS = run_neutral(v, args.games, args.workers, args.seed)
            S.to_parquet(cf)
            pd.DataFrame({"m": MS.to_numpy()}).to_parquet(Path(args.cache) / f"simM_{v}.parquet")
        r, pk = poss_analysis(S, MS)
        r["lags"] = lag_analysis(pk, S)
        pl = S[S["code"].isin([0, 1])]
        r["play_epa_var_sim"] = float(pl["epa"].var())
        r["play_yards_var_sim"] = float(pl["yards"].var())
        r["margin_var_neutral"] = float(MS.var())
        res["sim_" + v] = r
        print(v, f"{time.time() - t0:.0f}s", flush=True)
    (OUT / f"decomp_{args.tag}.json").write_text(json.dumps(m25.clean(res), indent=1, default=float))
    keys = [k for k in res["real_train"] if not isinstance(res["real_train"][k], dict)]
    names = list(res)
    print(f"{'key':34s} " + " ".join(f"{n:>12s}" for n in names))
    for k in keys:
        print(f"{k:34s} " + " ".join(f"{res[n].get(k, float('nan')):12.4f}" for n in names))
    for n in names:
        print("lags", n, json.dumps({a: (round(b, 2) if not isinstance(b, list) else np.round(b, 1).tolist()) for a, b in res[n]["lags"].items() if not a.startswith("cov_lag") or a == "cov_lag1"}))
    for k in ("RE", "play_effects_epa", "play_effects_yards_gained"):
        for n in names:
            if k in res[n]:
                print(k, n, json.dumps({a: round(b, 4) for a, b in res[n][k].items()}))


def blup_rows(pbp, trans, train):
    d = team_effects_plays(pbp, list(train) if pbp["season"].nunique() > 1 else list(pbp["season"].unique()), "epa")
    _, (oc, fc, eo, ef, ou, fu, mu) = fit_play_effects(d, "epa")
    omap = dict(zip(ou, eo))
    fmap = dict(zip(fu, ef))
    meta = pbp[["game_id", "play_id", "season", "posteam", "defteam"]].drop_duplicates(["game_id", "play_id"])
    t = trans[["game_id", "play_id"]].merge(meta, on=["game_id", "play_id"], how="left")
    ok = t["season"].astype(str) + "_" + t["posteam"].astype(str)
    fk = t["season"].astype(str) + "_" + t["defteam"].astype(str)
    return ok.map(omap).fillna(0.0).to_numpy(), fk.map(fmap).fillna(0.0).to_numpy()


IPW_BINS = [-99, -14, -8, -4, -1, 0, 3, 7, 13, 99]


def ipw_fit(trans, eo, ef, shrink=50.0, lo=0.25, hi=4.0):
    sd = np.clip(trans["sc_raw"].to_numpy(dtype=float), -24, 24)
    qt = np.minimum(trans["qtr_actual"].to_numpy(), 5)
    cell = (pd.cut(sd, IPW_BINS, labels=False).astype(int) * 6 + qt).astype(int)
    ncell = int(cell.max()) + 1
    n = np.bincount(cell, minlength=ncell).astype(float)
    logw = np.zeros(len(cell))
    info = {}
    for nm, x in (("off", eo), ("def", ef)):
        mu0 = float(x.mean())
        m = np.bincount(cell, weights=x, minlength=ncell) / np.maximum(n, 1)
        mc = mu0 + n / (n + shrink) * (m - mu0)
        s2 = float(((x - mc[cell]) ** 2).mean())
        logw += -0.5 * (((x - mu0) ** 2 - (x - mc[cell]) ** 2) / s2)
        info[nm] = {"mu0": mu0, "s2": s2}
    w = np.clip(np.exp(logw), lo, hi)
    return w, cell, info


def d_init(train, cfg_json):
    cfg = json.loads(cfg_json)
    c25.c_init(train, cfg_json)
    if not cfg.get("resid"):
        return
    import mod25_generator as gen

    sim.PBP_SNAPSHOT_DIR = m25.SNAP
    pbp = sim.load_reg_seasons(tuple(train))
    trans = sim.build_transition_frame(pbp)
    eo, ef = blup_rows(pbp, trans, train)
    lo, ld = gen.league_means()
    t = _G["tables"]
    a = t["arrays"]
    assert len(eo) == len(a["off_row"])
    a["off_row"] = (lo + eo).astype(np.float64)
    a["def_row"] = (ld + ef).astype(np.float64)
    t["team_kernel_h"] = sim.TEAM_KERNEL_H_SCALE * float(np.std(eo))
    if cfg.get("repa"):
        ep0 = np.asarray(t["attrs"]["epa"], dtype=np.float64)
        assert len(ep0) == len(eo)
        t["attrs"] = dict(t["attrs"], epa=ep0 - eo - ef, adv=eo + ef)
    if cfg.get("ipw"):
        import inspect

        w, _, _ = ipw_fit(trans, eo, ef)
        src = inspect.getsource(sim.pick_index_nn_conditioned)
        old = "weights = np.exp(-dist_sq / (2.0 * h * h))"
        assert old in src
        ns = _G["ns"]
        exec(src.replace(old, old + " * IPW[neighbors]"), ns)
        ns["IPW"] = w
    t["nn_weight_cache_cond"].clear()


def d_gen_init(setting):
    cfg = json.loads(setting["mech"])
    cfg.update(condition=1, yard_gain=setting["yard_gain"], def_sign=1.0, yard_bias=setting.get("yard_bias", 0.75))
    d_init(TRAIN, json.dumps(cfg))


DV = dict(c25.VARIANTS)
DV["cr"] = dict(c25.VARIANTS["c"], resid=1)
DV["b_r"] = dict(c25.VARIANTS["b"], resid=1)
DV["cre"] = dict(DV["cr"], repa=1)
DV["crw"] = dict(DV["cre"], ipw=1)
DEC_COND = dict(condition=1, yard_gain=2.0, def_sign=1.0, yard_bias=0.75, avg=1)


def d_play_season(task):
    tables = _G["tables"]
    arr = tables["attrs"]
    if "adv" not in arr:
        return c25.c_play_season(task)
    ns = _G["ns"]
    cfg = _G["cfg"]
    gain, bias = float(cfg["yard_gain"]), float(cfg["yard_bias"])
    world, sidx, seed, sched, ratings = task
    import mod25_generator as gen

    rng = np.random.default_rng(seed)
    gm_rows, chunks = [], []
    for gi, (week, h, a) in enumerate(sched):
        state = sim.initial_kickoff_state(rng, tables["opening_pool"])
        ns["PLAYLOG"].clear()
        _G["log"].clear()
        hr = {"off": gen.quant(ratings[week][h][0]), "def": gen.quant(ratings[week][h][1])}
        ar = {"off": gen.quant(ratings[week][a][0]), "def": gen.quant(ratings[week][a][1])}
        rec, cap_hit = ns["run_one_game"](state, tables, rng, 600.0, _G["pol"], sim.K_NEIGHBORS, sim.MAX_PLAYS_PER_GAME, hr, ar)
        tables["nn_weight_cache_cond"].clear()
        log = np.array(ns["PLAYLOG"], dtype=np.float64).reshape(-1, 3)
        idx = log[:, 1].astype(np.int64)
        code = arr["code"][idx]
        keep = (code == 0) | (code == 1)
        idx = idx[keep]
        shift = log[keep, 2]
        offhome = log[keep, 0].astype(np.int8)
        kind = np.where(code[keep] == 0, 0, np.where(arr["sack"][idx] == 1, 2, 1)).astype(np.int8)
        epa = arr["epa"][idx] + (shift - bias) / gain + arr["adv"][idx]
        yards = arr["yards"][idx] + shift
        turn = np.where(arr["int"][idx] == 1, 1, np.where(arr["fl"][idx] == 1, 2, 0)).astype(np.int8)
        chunks.append(np.rec.fromarrays([np.full(len(idx), gi, dtype=np.int16), offhome, kind, epa.astype(np.float32), yards.astype(np.float32), turn], names="g,offhome,kind,epa,yards,turn"))
        total, margin = float(rec["total"]), float(rec["margin"])
        gm_rows.append((gi, (total + margin) / 2.0, (total - margin) / 2.0, bool(rec["went_ot"]), bool(cap_hit)))
    plays = np.concatenate(chunks) if chunks else np.empty(0)
    return world, sidx, gm_rows, plays


def gen_eval_d(variant, scale, yard_gain, worlds, seasons, workers, seed, yard_bias):
    c25.ensure_policies()
    import mod25_generator as gen

    cfgj = json.dumps(dict(DV[variant], seed=seed))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": scale, "yard_gain": yard_gain, "def_sign": 1.0, "yard_bias": yard_bias, "drift": 1.0, "mech": cfgj})
    gen.init_worker = d_gen_init
    gen.play_season = d_play_season
    games, plays, latents, elapsed = gen.run_generation(setting, worlds, seasons, workers, seed, progress=False)
    ts = gen.team_stats_from_plays(plays)
    keep = [w * 1000 + s + 1 for w in range(worlds) for s in range(2, seasons)]
    m, ac, dd = gen.all_metrics(games, ts, keep)
    m["r2_pooled"] = float(np.corrcoef(dd["x"], dd["y"])[0, 1] ** 2)
    m["nonstrength_var"] = float(m["margin_sd"] ** 2 * (1 - m["r2_pooled"]))
    m["pts_game"] = float((games["home_score"] + games["away_score"]).mean())
    return m


def cmd_grid(args):
    real = c25.real_gate_metrics()["2011_2017"]
    tg = json.loads((REPO / "artifacts" / "mod25_generator" / "real_targets.json").read_text())["gates"]
    rows = {}
    for sc in [float(x) for x in args.scales.split(",")]:
        m = gen_eval_d(args.variant, sc, args.yard_gain, args.worlds, args.seasons, args.workers, args.seed, args.yard_bias)
        loss = 0.0
        for k in ("epa_autocorr_lag1", "r2_w1_4", "r2_w5_9", "r2_w10_18"):
            hw = (tg[k]["hi"] - tg[k]["lo"]) / 2.0
            loss += ((m[k] - real[k]) / hw) ** 2
        m["loss"] = float(loss)
        rows[str(sc)] = m
        print(sc, json.dumps({k: round(m[k], 4) for k in c25.GK + ["loss"]}), flush=True)
    (OUT / f"grid_{args.variant}.json").write_text(json.dumps({"real_2011_2017": real, "rows": rows}, indent=1))


def cmd_gate(args):
    real = c25.real_gate_metrics()["2018_2025"]
    tg = json.loads((REPO / "artifacts" / "mod25_generator" / "real_targets.json").read_text())["gates"]
    res = {}
    for v in args.variants.split(","):
        res[v] = gen_eval_d(v, args.scale, args.yard_gain, args.worlds, args.seasons, args.workers, args.seed, args.yard_bias)
        print(v, flush=True)
    (OUT / f"gate_{args.tag}.json").write_text(json.dumps({"real_2018_2025": real, "res": res, "scale": args.scale}, indent=1))
    print(f"{'gate':20s} {'real':>9s} {'lo':>7s} {'hi':>7s} " + " ".join(f"{n:>10s}" for n in res))
    for k in c25.GK:
        lo = tg.get(k, {}).get("lo", float("nan"))
        hi = tg.get(k, {}).get("hi", float("nan"))
        cells = " ".join(f"{res[n][k]:9.4f}{'*' if lo <= res[n][k] <= hi else ' '}" for n in res)
        print(f"{k:20s} {real.get(k, float('nan')):9.4f} {lo:7.4f} {hi:7.4f} {cells}")


def cmd_tilt(args):
    P, M, meta, pbp = real_load(TRAIN)
    d = team_effects_plays(pbp, list(TRAIN), "epa")
    _, (oc, fc, eo, ef, ou, fu, mu) = fit_play_effects(d, "epa")
    d = d.assign(adv=eo[oc] + ef[fc])
    fh = d[(d["qtr"] <= 2) & d["score_differential"].notna()]
    bins = [-99, -14, -8, -4, -1, 0, 3, 7, 13, 99]
    fh = fh.assign(b=pd.cut(fh["score_differential"], bins))
    t = fh.groupby("b", observed=True)["adv"].agg(["mean", "size"])
    print(t.to_string())
    print("sd of play-level advantage", float(d["adv"].std()), "slope per point of lead", float(np.polyfit(fh["score_differential"], fh["adv"], 1)[0]))
    t.to_csv(OUT / "tilt.csv")


def cmd_ipwcheck(args):
    P, M, meta, pbp = real_load(TRAIN)
    trans = sim.build_transition_frame(pbp)
    eo, ef = blup_rows(pbp, trans, TRAIN)
    w, cell, info = ipw_fit(trans, eo, ef)
    sd = np.clip(trans["sc_raw"].to_numpy(dtype=float), -24, 24)
    b = pd.cut(sd, IPW_BINS)
    h = trans["qtr_actual"].to_numpy() <= 2
    rows = []
    for lab, g in pd.DataFrame({"b": b, "h": h, "o": eo, "d": ef, "w": w}).groupby(["b"], observed=True):
        rows.append({"bin": str(lab), "n": len(g), "off_raw": g["o"].mean(), "off_w": np.average(g["o"], weights=g["w"]), "def_raw": g["d"].mean(), "def_w": np.average(g["d"], weights=g["w"]), "net_raw": (g["o"] + g["d"]).mean(), "net_w": np.average(g["o"] + g["d"], weights=g["w"])})
    t = pd.DataFrame(rows)
    print(t.to_string())
    t.to_csv(OUT / "ipwcheck.csv", index=False)
    print("weight mean sd min max", float(w.mean()), float(w.std()), float(w.min()), float(w.max()), "ess frac", float(w.sum() ** 2 / (w ** 2).sum() / len(w)), info)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decomp")
    d.add_argument("--variants", default="c")
    d.add_argument("--games", type=int, default=12000)
    d.add_argument("--workers", type=int, default=6)
    d.add_argument("--seed", type=int, default=5)
    d.add_argument("--tag", default="d1")
    d.add_argument("--skip-sim", dest="skip_sim", action="store_true")
    d.add_argument("--cache", default="C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad")
    sub.add_parser("tilt")
    sub.add_parser("ipwcheck")
    for nm in ("grid", "gate"):
        g = sub.add_parser(nm)
        g.add_argument("--variant", default="cr")
        g.add_argument("--variants", default="cr")
        g.add_argument("--scales", default="3,4,5,6")
        g.add_argument("--scale", type=float, default=3.0)
        g.add_argument("--worlds", type=int, default=8 if nm == "gate" else 6)
        g.add_argument("--seasons", type=int, default=8)
        g.add_argument("--workers", type=int, default=6)
        g.add_argument("--seed", type=int, default=9 if nm == "gate" else 21)
        g.add_argument("--yard-gain", dest="yard_gain", type=float, default=2.0)
        g.add_argument("--yard-bias", dest="yard_bias", type=float, default=0.75)
        g.add_argument("--tag", default="g1")
    args = ap.parse_args()
    {"decomp": cmd_decomp, "tilt": cmd_tilt, "ipwcheck": cmd_ipwcheck, "grid": cmd_grid, "gate": cmd_gate}[args.cmd](args)


if __name__ == "__main__":
    main()
