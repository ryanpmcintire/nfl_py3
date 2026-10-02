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
            "epa": tr["epa"].to_numpy(), "yards": tr["yards"].to_numpy(), "tov": tr["tov"].to_numpy(),
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


SIDE_TRK = False


def track_sides(P):
    g = P["g"].to_numpy()
    sd = P["sd"].to_numpy(float)
    d = P["po"].to_numpy(float) - P["pdf"].to_numpy(float)
    fl = P["flip"].to_numpy().astype(bool)
    n = len(P)
    nxt_ok = np.r_[g[1:] == g[:-1], False]
    s = sd + d
    nx = np.r_[sd[1:], 0.0]
    same = nxt_ok & (nx == s) & (nx != -s)
    opp = nxt_ok & (nx == -s) & (nx != s)
    rel = np.where(same, 1, np.where(opp, -1, np.where(fl, -1, 1)))
    sg = np.ones(n, dtype=np.int64)
    cur = 1
    for i in range(n - 1):
        if nxt_ok[i]:
            cur = cur * rel[i]
        else:
            cur = 1
        sg[i + 1] = cur
    return np.where(sg == 1, 0, 1).astype(np.int8)


def poss_analysis(P, M):
    P = P.sort_values("g", kind="stable").reset_index(drop=True)
    reg = P["qtr"].to_numpy() <= 4
    side, poss = assign_sides(P)
    if SIDE_TRK:
        side = track_sides(P)
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
        pl = np.array(ns["PLAYLOG"], dtype=np.float64).reshape(-1, 3)
        rows.append(np.column_stack([np.full(len(a), gi), a[:, :11], tov[ix], to_arr[ix], ep[ix], yd[ix], pl[:, 0], pl[:, 2]]))
        margins.append(rec["margin"])
    return seed, np.concatenate(rows), np.array(margins)


DCOLN = c25.COLN + ["epa", "yards", "offhome", "shift"]


def run_neutral(variant, games, workers, seed):
    import multiprocessing as mp

    c25.ensure_policies()
    cfg = dict(DV[variant], seed=seed)
    if cfg.get("resid"):
        cfg.update(DEC_COND)
    if "ybias" in cfg:
        cfg["yard_bias"] = cfg["ybias"]
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


RATE_NAMES = ["rz_td", "conv", "expl", "sack", "tov"]
RATE_MIX = (0, 1, 2, 3)


def rate_defs(trans, at):
    code = trans["play_type_code"].to_numpy()
    rp = np.isin(code, (0, 1))
    down = trans["down_i"].to_numpy()
    fp = trans["fp_raw"].to_numpy(dtype=float)
    dist = trans["dist_raw"].to_numpy(dtype=float)
    yg = trans["yards_gained"].to_numpy(dtype=float)
    po = trans["points_off"].to_numpy(dtype=float)
    af = trans["auto_first"].to_numpy().astype(bool)
    tov = at["tov"].astype(bool)
    sack = at["sack"].astype(bool)
    yds = at["yards"].astype(float)
    D, I = [], []
    d = rp & (fp <= 20)
    D.append(d)
    I.append(d & (po >= 6))
    d = rp & np.isin(down, (3, 4))
    D.append(d)
    I.append(d & ~tov & ((yg >= dist) | af | (po >= 6)))
    d = rp & ~sack
    D.append(d)
    I.append(d & (((code == 0) & (yds >= 10)) | ((code == 1) & (yds >= 20))))
    d = code == 1
    D.append(d)
    I.append(d & sack)
    d = rp
    D.append(d)
    I.append(d & tov)
    return np.column_stack(D), np.column_stack(I)


def team_keys(pbp, trans):
    meta = pbp[["game_id", "play_id", "season", "posteam", "defteam"]].drop_duplicates(["game_id", "play_id"])
    t = trans[["game_id", "play_id"]].merge(meta, on=["game_id", "play_id"], how="left")
    ok = (t["season"].astype(str) + "_" + t["posteam"].astype(str)).to_numpy()
    fk = (t["season"].astype(str) + "_" + t["defteam"].astype(str)).to_numpy()
    return ok, fk


def rate_effects(trans, pbp, at, train):
    D, I = rate_defs(trans, at)
    ok, fk = team_keys(pbp, trans)
    d0 = team_effects_plays(pbp, list(train), "epa")
    _, (oc, fc, eo, ef, ou, fu, mu) = fit_play_effects(d0, "epa")
    epa_o = pd.Series(eo, index=ou)
    epa_d = pd.Series(ef, index=fu)
    res = []
    for k in range(len(RATE_NAMES)):
        m = D[:, k] & pd.notna(ok) & pd.notna(fk)
        d = pd.DataFrame({"o": ok[m], "f": fk[m], "y": I[m, k].astype(float)})
        st, (oc, fc, eo, ef, ou, fu, mu) = fit_play_effects(d, "y")
        so = pd.Series(eo, index=ou)
        sd_ = pd.Series(ef, index=fu)
        co = so.index.intersection(epa_o.index)
        cd = sd_.index.intersection(epa_d.index)
        res.append({
            "name": RATE_NAMES[k], "p": float(I[D[:, k], k].mean()), "off_sd": st["off_sd"], "def_sd": st["def_sd"],
            "off_corr_epa": float(np.corrcoef(so[co], epa_o[co])[0, 1]), "def_corr_epa": float(np.corrcoef(sd_[cd], epa_d[cd])[0, 1]),
        })
    return res, D, I


def fit_tilt(cfg, pbp, trans, ns):
    import mod25_generator as gen

    at = c25.attrs_from(pbp, trans)
    res, D, I = rate_effects(trans, pbp, at, TRAIN)
    fit = gen.load_fit()
    qb = fit["qb"]
    var_off = max(1e-6, fit["off"]["var_mu"] - qb["backup_rate"] * (1 - qb["backup_rate"]) * qb["backup_off_epa_effect"] ** 2)
    zo, zd = float(np.sqrt(var_off)), float(np.sqrt(fit["def"]["var_mu"]))
    mult = float(cfg.get("tilt_mult", 1.0))
    ao = np.array([np.sign(r["off_corr_epa"]) * r["off_sd"] / (r["p"] * (1 - r["p"]) * zo) for r in res]) * mult
    ad = np.array([np.sign(r["def_corr_epa"]) * r["def_sd"] / (r["p"] * (1 - r["p"]) * zd) for r in res]) * mult
    if cfg.get("tilt_file"):
        tf = json.loads(Path(cfg["tilt_file"]).read_text())
        ao = np.array([tf[n]["off"] for n in RATE_NAMES]) * mult
        ad = np.array([tf[n]["def"] for n in RATE_NAMES]) * mult
    lo, ld = gen.league_means()
    pbar = np.array([r["p"] for r in res])
    T = (I + (~D) * pbar[None, :]).astype(np.float64)
    ns["TILT_T"] = T[:, list(RATE_MIX)]
    ns["TILT_AO"] = ao[list(RATE_MIX)]
    ns["TILT_AD"] = ad[list(RATE_MIX)]
    ns["TILT_LO"] = lo
    ns["TILT_LD"] = ld
    c25.TOV_AO, c25.TOV_AD, c25.TOV_LO, c25.TOV_LD = float(ao[4]), float(ad[4]), lo, ld
    ns["RATING_CTX"] = c25.RATING_CTX
    _G["rates"] = (D, I)
    _G["rate_info"] = res


def tiltp(p):
    th = c25.TOV_AO * (c25.RATING_CTX[0] - c25.TOV_LO) + c25.TOV_AD * (c25.RATING_CTX[1] - c25.TOV_LD)
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    q = p / (1 - p) * float(np.exp(th))
    return q / (1 + q)


def install_tilt():
    import inspect

    if getattr(c25, "_TILT", False):
        return
    c25.RATING_CTX = [0.0, 0.0]
    c25.TOV_AO = c25.TOV_AD = c25.TOV_LO = c25.TOV_LD = 0.0
    c25.TILTP = tiltp
    src = inspect.getsource(c25.make_c_decide)
    old = "tv_t = 1 if rng2.random() < p else 0"
    assert old in src
    exec(src.replace(old, "tv_t = 1 if rng2.random() < TILTP(p) else 0"), c25.__dict__)
    c25._TILT = True


EL_MAX = 60.0


def clean_clock(trans):
    el = trans["clock_elapsed"].to_numpy(dtype=float).copy()
    art = el > EL_MAX
    code = trans["play_type_code"].to_numpy()
    flip = trans["possession_flip"].to_numpy().astype(bool)
    cl = c25.cls_of(code, flip, trans["points_off"].to_numpy(), trans["points_def"].to_numpy(), trans["yards_gained"].to_numpy())
    zn = c25.zone_v(trans["qtr_actual"].to_numpy(), trans["gsr_actual"].to_numpy())
    sb = c25.sb_v(trans["sc_raw"].to_numpy())
    key = (cl * 8 + zn) * 8 + sb
    ok = ~art
    n = 64 * 8 * 8
    cnt = np.bincount(key[ok], minlength=n)
    sm = np.bincount(key[ok], weights=el[ok], minlength=n)
    zk = cl * 8 + zn
    zc = np.bincount(zk[ok], minlength=512)
    zs = np.bincount(zk[ok], weights=el[ok], minlength=512)
    cc = np.bincount(cl[ok], minlength=64)
    cs = np.bincount(cl[ok], weights=el[ok], minlength=64)
    full = np.full(n, np.nan)
    for k in range(n):
        if cnt[k] >= 40:
            full[k] = sm[k] / cnt[k]
        elif zc[k // 8] >= 40:
            full[k] = zs[k // 8] / zc[k // 8]
        elif cc[k // 64] >= 40:
            full[k] = cs[k // 64] / cc[k // 64]
    rep = full[key]
    rep = np.where(np.isnan(rep), float(el[ok].mean()), rep)
    el[art] = rep[art]
    return trans.assign(clock_elapsed=el), int(art.sum())


def install_clean_clock():
    if getattr(sim, "_CLEAN_CLOCK", False):
        return
    orig = sim.build_transition_frame

    def wrapped(pbp, team_ratings=None):
        t = orig(pbp, team_ratings=team_ratings)
        t, _ = clean_clock(t)
        return t

    sim.build_transition_frame = wrapped
    sim._CLEAN_CLOCK = True


CAL_YB = np.array([0, 20, 35, 45, 55, 65, 80, 101], dtype=float)
CAL_DB = np.array([0, 2, 5, 10, 100], dtype=float)
CAL_TB = np.array([-1, 120, 420, 900, 1680, 1800, 2700, 3601], dtype=float)


def cal_cell(yl, dist, gsr):
    a = np.clip(np.searchsorted(CAL_YB, yl, side="left") - 1, 0, len(CAL_YB) - 2)
    b = np.clip(np.searchsorted(CAL_DB, dist, side="left") - 1, 0, len(CAL_DB) - 2)
    c = np.clip(np.searchsorted(CAL_TB, gsr, side="left") - 1, 0, len(CAL_TB) - 2)
    return (a * (len(CAL_DB) - 1) + b) * (len(CAL_TB) - 1) + c


def calibrate_p4(P4, trans, shrink=10.0):
    M = m25
    sd = np.clip(trans["sc_raw"].to_numpy(), -24, 24)
    gsr = trans["gsr_actual"].to_numpy().astype(float)
    qtr = trans["qtr_actual"].to_numpy()
    yl = trans["fp_raw"].to_numpy().astype(float)
    dist = trans["dist_raw"].to_numpy().astype(float)
    down = trans["down_i"].to_numpy()
    code = trans["play_type_code"].to_numpy()
    otf = (trans["off_to_raw"].to_numpy() > 0).astype(int)
    dtf = (trans["def_to_raw"].to_numpy() > 0).astype(int)
    m4 = (down == 4) & np.isin(code, (0, 1, 2, 3))
    ix = lambda ax, x: np.abs(ax[None, :] - x[:, None]).argmin(1)
    sd, gsr, qtr, yl, dist, otf, dtf, code = (a[m4] for a in (sd, gsr, qtr, yl, dist, otf, dtf, code))
    pr = P4[ix(M.SD_AX, sd), ix(M.T4_AX, gsr), (qtr >= 5).astype(int), ix(M.Y4_AX, yl), ix(M.D4_AX, dist), otf, dtf]
    cls = M.cls4_of(code)
    cell = cal_cell(yl, dist, gsr)
    ncell = (len(CAL_YB) - 1) * (len(CAL_DB) - 1) * (len(CAL_TB) - 1)
    n = np.bincount(cell, minlength=ncell).astype(float)
    fac = np.ones((ncell, 3))
    for c in range(3):
        mp = np.bincount(cell, weights=pr[:, c], minlength=ncell)
        rc = np.bincount(cell, weights=(cls == c).astype(float), minlength=ncell)
        mm = np.where(n > 0, mp / np.maximum(n, 1), 0.0)
        fac[:, c] = np.where(mm > 1e-6, (rc + shrink * mm) / np.maximum(mp + shrink * mm, 1e-9), 1.0)
    gy = cal_cell_grid(M)
    out = P4 * fac[gy][..., :]
    return (out / out.sum(-1, keepdims=True)).astype(np.float32)


def cal_cell_grid(M):
    sh = (len(M.SD_AX), len(M.T4_AX), 2, len(M.Y4_AX), len(M.D4_AX), 2, 2)
    T = M.T4_AX[None, :, None, None, None, None, None]
    Y = M.Y4_AX[None, None, None, :, None, None, None]
    D = M.D4_AX[None, None, None, None, :, None, None]
    return cal_cell(np.broadcast_to(Y, sh), np.broadcast_to(D, sh), np.broadcast_to(T, sh))


def install_cal4():
    if getattr(m25, "_CAL4", False):
        return
    orig = m25.fit_decisions

    def wrapped(trans):
        P4, PL, a, b = orig(trans)
        return calibrate_p4(P4, trans), PL, a, b

    m25.fit_decisions = wrapped
    m25._CAL4 = True


SDB = np.array([-99, -16, -8, -4, 0, 4, 8, 16, 99], dtype=float)
DB2 = np.array([0, 2, 5, 100], dtype=float)


def cal2_cell(sd, dist, gsr):
    a = np.clip(np.searchsorted(SDB, sd, side="left") - 1, 0, len(SDB) - 2)
    b = np.clip(np.searchsorted(DB2, dist, side="left") - 1, 0, len(DB2) - 2)
    c = np.clip(np.searchsorted(CAL_TB, gsr, side="left") - 1, 0, len(CAL_TB) - 2)
    return (a * (len(DB2) - 1) + b) * (len(CAL_TB) - 1) + c


def calibrate_p4s(P4, trans, shrink=10.0):
    M = m25
    sd = np.clip(trans["sc_raw"].to_numpy(), -24, 24)
    gsr = trans["gsr_actual"].to_numpy().astype(float)
    qtr = trans["qtr_actual"].to_numpy()
    yl = trans["fp_raw"].to_numpy().astype(float)
    dist = trans["dist_raw"].to_numpy().astype(float)
    down = trans["down_i"].to_numpy()
    code = trans["play_type_code"].to_numpy()
    otf = (trans["off_to_raw"].to_numpy() > 0).astype(int)
    dtf = (trans["def_to_raw"].to_numpy() > 0).astype(int)
    m4 = (down == 4) & np.isin(code, (0, 1, 2, 3))
    ix = lambda ax, x: np.abs(ax[None, :] - x[:, None]).argmin(1)
    sd, gsr, qtr, yl, dist, otf, dtf, code = (a[m4] for a in (sd, gsr, qtr, yl, dist, otf, dtf, code))
    pr = P4[ix(M.SD_AX, sd), ix(M.T4_AX, gsr), (qtr >= 5).astype(int), ix(M.Y4_AX, yl), ix(M.D4_AX, dist), otf, dtf]
    cls = M.cls4_of(code)
    cell = cal2_cell(sd, dist, gsr)
    ncell = (len(SDB) - 1) * (len(DB2) - 1) * (len(CAL_TB) - 1)
    n = np.bincount(cell, minlength=ncell).astype(float)
    fac = np.ones((ncell, 3))
    for c in range(3):
        mp = np.bincount(cell, weights=pr[:, c], minlength=ncell)
        rc = np.bincount(cell, weights=(cls == c).astype(float), minlength=ncell)
        mm = np.where(n > 0, mp / np.maximum(n, 1), 0.0)
        fac[:, c] = np.where(mm > 1e-6, (rc + shrink * mm) / np.maximum(mp + shrink * mm, 1e-9), 1.0)
    sh = (len(M.SD_AX), len(M.T4_AX), 2, len(M.Y4_AX), len(M.D4_AX), 2, 2)
    S = M.SD_AX[:, None, None, None, None, None, None]
    T = M.T4_AX[None, :, None, None, None, None, None]
    D = M.D4_AX[None, None, None, None, :, None, None]
    gy = cal2_cell(np.broadcast_to(S, sh), np.broadcast_to(D, sh), np.broadcast_to(T, sh))
    out = P4 * fac[gy][..., :]
    return (out / out.sum(-1, keepdims=True)).astype(np.float32)


def install_cal4s():
    if getattr(m25, "_CAL4S", False):
        return
    orig = m25.fit_decisions

    def wrapped(trans):
        P4, PL, a, b = orig(trans)
        return calibrate_p4s(P4, trans), PL, a, b

    m25.fit_decisions = wrapped
    m25._CAL4S = True


EP_MODEL_PATH = OUT / "ep_model.joblib"
EP_COLS = ["down", "ydstogo", "yardline_100", "half_seconds_remaining", "score_differential", "posteam_to", "defteam_to", "half"]
NV_DIR = "C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad/nv"


def ep_features(down, dist, yl, hs, sd, to_o, to_d, half):
    return np.column_stack([down, dist, yl, hs, sd, to_o, to_d, half]).astype(np.float64)


def load_nv_ep(seasons):
    frames = []
    for s in seasons:
        d = pd.read_parquet(f"{NV_DIR}/pbp_{s}.parquet", columns=["season_type", "ep", "down", "ydstogo", "yardline_100", "half_seconds_remaining", "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "game_half", "qtr", "season"])
        d = d[(d["season_type"] == "REG") & d["down"].notna() & d["ep"].notna() & d["yardline_100"].notna() & d["score_differential"].notna() & d["half_seconds_remaining"].notna()]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    half = np.where(d["qtr"].to_numpy() >= 5, 3, np.where(d["qtr"].to_numpy() <= 2, 1, 2))
    X = ep_features(d["down"].to_numpy(), d["ydstogo"].to_numpy(), d["yardline_100"].to_numpy(), d["half_seconds_remaining"].to_numpy(), d["score_differential"].to_numpy(), d["posteam_timeouts_remaining"].fillna(3).to_numpy(), d["defteam_timeouts_remaining"].fillna(3).to_numpy(), half)
    return X, d["ep"].to_numpy(dtype=float), d["season"].to_numpy()


def fit_ep_model(X, y, cols=None):
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    return m.fit(X if cols is None else X[:, cols], y)


def cmd_epfit(args):
    import joblib

    OUT.mkdir(parents=True, exist_ok=True)
    X, y, ssn = load_nv_ep(range(2009, 2018))
    tr, te = ssn <= 2015, ssn >= 2016
    res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "ep_var_test": float(y[te].var())}
    for nm, cols in (("state_only_down_dist_yl", [0, 1, 2]), ("full", None)):
        m = fit_ep_model(X[tr], y[tr], cols)
        pr = m.predict(X[te] if cols is None else X[te][:, cols])
        res[nm] = {"r2_heldout_2016_17": float(1 - ((y[te] - pr) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()), "mae_heldout": float(np.abs(y[te] - pr).mean())}
        pi = m.predict(X[tr] if cols is None else X[tr][:, cols])
        res[nm]["r2_insample"] = float(1 - ((y[tr] - pi) ** 2).sum() / ((y[tr] - y[tr].mean()) ** 2).sum())
    joblib.dump(fit_ep_model(X, y), EP_MODEL_PATH)
    (OUT / "epfit.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def install_chain2():
    import joblib

    _G["ep_model"] = joblib.load(EP_MODEL_PATH)
    _G["tolog"] = []
    ns = _G["ns"]
    dec = ns["DECIDE"]
    st = {"to": (3.0, 3.0)}

    def decide(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, *rest):
        st["to"] = (float(off_to), float(def_to))
        return dec(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, *rest)

    ns["DECIDE"] = decide
    base = _G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        _G["tolog"].append(st["to"])
        return base(down, distance, yardline, score_diff, qtr, clock_val, drawn)

    _G["pol"] = pol


def norm_pts(v):
    return np.where(v >= 6, 7.0, v)


def chain2_epa(log, tol, model, offhome):
    down = np.nan_to_num(log[:, 0], nan=1.0)
    qtr = log[:, 5]
    clock = log[:, 4]
    hs = np.where(qtr <= 2, clock - 1800.0, clock)
    half = np.where(qtr >= 5, 3, np.where(qtr <= 2, 1, 2))
    ep = model.predict(ep_features(down, log[:, 1], log[:, 2], np.maximum(hs, 0.0), log[:, 3], tol[:, 0], tol[:, 1], half))
    pts = norm_pts(log[:, 7]) - norm_pts(log[:, 8])
    scored = (log[:, 7] > 0) | (log[:, 8] > 0)
    n = len(ep)
    nxt_ep = np.r_[ep[1:], 0.0]
    same = np.r_[offhome[1:] == offhome[:-1], False]
    half_end = np.r_[(qtr[1:] >= 3) & (qtr[:-1] <= 2), False]
    last = np.r_[np.zeros(n - 1, dtype=bool), True]
    v = np.where(same, nxt_ep, -nxt_ep)
    v = np.where(scored, pts, v)
    v = np.where((half_end | last) & ~scored, 0.0, v)
    return v - ep


EP_DB = np.array([0, 1, 2, 3, 5, 8, 11, 16, 1000], dtype=float)
EP_YB = np.arange(0, 104, 4, dtype=float)


def ep_cell(down, dist, yl):
    d = np.clip(np.nan_to_num(down, nan=1.0), 1, 4).astype(int) - 1
    b = np.clip(np.searchsorted(EP_DB, dist, side="left") - 1, 0, len(EP_DB) - 2)
    y = np.clip(np.searchsorted(EP_YB, yl, side="left") - 1, 0, len(EP_YB) - 2)
    return (d * (len(EP_DB) - 1) + b) * (len(EP_YB) - 1) + y


def fit_ep_table(P, shrink=20.0):
    D, R = drive_table(P)
    R = R.copy()
    R["epa"] = R["epa"].fillna(0.0)
    net = (R["po"] - R["pdf"]).groupby(R["poss"]).transform("sum").to_numpy()
    rem = R["epa"][::-1].groupby(R["poss"][::-1]).cumsum()[::-1].to_numpy()
    ep = net - rem
    cell = ep_cell(R["down"].to_numpy(dtype=float), R["dist"].to_numpy(dtype=float), R["yl"].to_numpy(dtype=float))
    nc = 4 * (len(EP_DB) - 1) * (len(EP_YB) - 1)
    n = np.bincount(cell, minlength=nc).astype(float)
    s = np.bincount(cell, weights=ep, minlength=nc)
    cd = cell // (len(EP_DB) - 1) // (len(EP_YB) - 1) * (len(EP_YB) - 1) + cell % (len(EP_YB) - 1)
    nco = 4 * (len(EP_YB) - 1)
    n2 = np.bincount(cd, minlength=nco).astype(float)
    s2 = np.bincount(cd, weights=ep, minlength=nco)
    coarse = np.where(n2 > 0, s2 / np.maximum(n2, 1), 0.0)
    ci = np.arange(nc) // (len(EP_DB) - 1) // (len(EP_YB) - 1) * (len(EP_YB) - 1) + np.arange(nc) % (len(EP_YB) - 1)
    return (s + shrink * coarse[ci]) / (n + shrink)


def install_chain(P):
    _G["ep_tab"] = fit_ep_table(P)


def chain_epa(log, offhome, tab):
    down, dist, yl, po, pdf = log[:, 0], log[:, 1], log[:, 2], log[:, 7], log[:, 8]
    ep = tab[ep_cell(down, dist, yl)]
    n = len(ep)
    new = np.r_[True, offhome[1:] != offhome[:-1]]
    last = np.r_[new[1:], True]
    drv = np.cumsum(new) - 1
    net = np.bincount(drv, weights=po - pdf)[drv]
    nxt = np.r_[ep[1:], 0.0]
    return np.where(last, net - ep, nxt - ep)


def install_pace(sigma, seed):
    import os

    base = _G["pol"]
    rng = np.random.default_rng(seed * 7919 + os.getpid())
    st = {"m": 1.0}

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        if qtr == 1 and clock_val >= 3599.5:
            st["m"] = float(np.exp(sigma * rng.standard_normal() - 0.5 * sigma * sigma))
        if qtr <= 4 and st["m"] != 1.0:
            drawn = dict(drawn)
            drawn["clock_elapsed"] = float(drawn["clock_elapsed"]) * st["m"]
        return base(down, distance, yardline, score_diff, qtr, clock_val, drawn)

    _G["pol"] = pol


def install_fined(rd):
    import inspect

    ns = _G["ns"]
    src = inspect.getsource(sim.round_state_key)
    assert "ROUND_DIST" in src
    exec(src.replace("ROUND_DIST", repr(rd)), ns)
    psrc = inspect.getsource(sim.pick_index_nn_conditioned)
    if "TILT_T" not in ns.get("pick_index_nn_conditioned").__code__.co_names:
        exec(psrc, ns)
    t = _G["tables"]
    t["nn_cache_cond"].clear()


def d_init(train, cfg_json):
    cfg = json.loads(cfg_json)
    if cfg.get("ydsc"):
        sim.SCALE_YDSTOGO = float(cfg["ydsc"])
    if cfg.get("clean"):
        install_clean_clock()
    if cfg.get("tilt"):
        install_tilt()
    if cfg.get("cal4"):
        install_cal4()
    if cfg.get("cal4s"):
        install_cal4s()
    if cfg.get("fdnb"):
        import inspect

        orig_gs = inspect.getsource

        def patched_gs(o):
            src_ = orig_gs(o)
            if o is sim.run_one_game:
                old_ = 'if drawn["auto_first"] or (not drawn["repeat_down"] and gained >= distance):'
                assert old_ in src_
                src_ = src_.replace(old_, 'if drawn["auto_first"] or (not drawn["repeat_down"] and (gained >= distance or (down >= 4 and arrays["next_down"][idx] == 1))):')
            return src_

        inspect.getsource = patched_gs
        try:
            c25.c_init(train, cfg_json)
        finally:
            inspect.getsource = orig_gs
    else:
        c25.c_init(train, cfg_json)
    if cfg.get("pace"):
        install_pace(float(cfg["pace"]), int(cfg.get("seed", 3)))
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
    if cfg.get("resid") and not cfg.get("tilt"):
        _G["rates"] = rate_defs(trans, c25.attrs_from(pbp, trans))
    if cfg.get("ipw") or cfg.get("tilt") or cfg.get("dkern"):
        import inspect

        src = inspect.getsource(sim.pick_index_nn_conditioned)
        old = "weights = np.exp(-dist_sq / (2.0 * h * h))"
        hdr = "    down_key = down if down in (1, 2, 3, 4) else 4\n    entry = tables"
        assert old in src and hdr in src
        ns = _G["ns"]
        extra = ""
        if cfg.get("ipw"):
            w, _, _ = ipw_fit(trans, eo, ef)
            ns["IPW"] = w
            extra += " * IPW[neighbors]"
        if cfg.get("tilt"):
            fit_tilt(cfg, pbp, trans, ns)
            extra += " * np.exp(TILT_T[neighbors] @ (TILT_AO * (off_sim - TILT_LO) + TILT_AD * (def_sim - TILT_LD)))"
        if cfg.get("dkern"):
            ns["DISTR"] = trans["dist_raw"].to_numpy(dtype=np.float64)
            ns["DKH"] = float(cfg["dkern"])
            ns["DKR"] = float(cfg.get("fined") or sim.ROUND_DIST)
            extra += " * np.exp(-0.5 * ((DISTR[neighbors] - key[2] * DKR) / (DKH * max(1.0, key[2] * DKR) ** 0.5)) ** 2)"
        ctx = "    RATING_CTX[0] = off_sim\n    RATING_CTX[1] = def_sim\n" if cfg.get("tilt") else ""
        exec(src.replace(old, old + extra).replace(hdr, ctx + hdr), ns)
    if cfg.get("chain"):
        P, M, meta, pbp2 = real_load(train)
        install_chain(P)
    if cfg.get("chain2"):
        install_chain2()
    if cfg.get("fined"):
        install_fined(float(cfg["fined"]))
    t["nn_weight_cache_cond"].clear()


def d_gen_init(setting):
    cfg = json.loads(setting["mech"])
    cfg.update(condition=1, yard_gain=setting["yard_gain"], def_sign=1.0, yard_bias=cfg.get("ybias", setting.get("yard_bias", 0.75)))
    d_init(TRAIN, json.dumps(cfg))


DV = dict(c25.VARIANTS)
DV["cr"] = dict(c25.VARIANTS["c"], resid=1)
DV["b_r"] = dict(c25.VARIANTS["b"], resid=1)
DV["cre"] = dict(DV["cr"], repa=1)
DV["crw"] = dict(DV["cre"], ipw=1)
DV["crk"] = dict(DV["crw"], clean=1)
DV["crt"] = dict(DV["crk"], tilt=1, tilt_file=str(OUT / "tilt_coef.json"))
DV["crwt"] = dict(DV["crw"], tilt=1)
for _nm, _drop in (("late", ("late",)), ("fourth", ("fourth",)), ("pat", ("pat",)), ("fine", ("inner", "outer", "stime")), ("ipw", ("ipw",)), ("clean", ("clean",))):
    DV["crt_no" + _nm] = {k: v for k, v in DV["crt"].items() if k not in _drop}
for _sg in (4, 8, 12, 16, 20):
    DV[f"crp{_sg:02d}"] = dict(DV["crt"], pace=_sg / 100.0)
DV["crf"] = dict(DV["crp04"], cal4=1)
DV["crg"] = dict(DV["crf"], cal4s=1)
DV["crh"] = dict(DV["crf"], chain=1)
DV["cri"] = dict(DV["crg"], chain=1)
DV["crj"] = dict(DV["crg"], chain2=1)
DV["crm"] = dict(DV["crj"], fined=1.0)
DV["crn"] = dict(DV["crj"], ydsc=2.5)
DV["cro"] = dict(DV["crj"], ydsc=1.25)
DV["crq0"] = dict(DV["crj"], ybias=0.0)
for _h in (4, 6, 10, 20):
    DV[f"crk{_h:02d}"] = dict(DV["crq0"], dkern=_h / 10.0, fined=1.0)
DV["crf4"] = dict(DV["crk06"], fdnb=1)
DV["crf4m"] = dict(DV["crf4"])
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
    rates = _G.get("rates")
    if rates is not None:
        K = rates[0].shape[1]
        dg = {n: np.zeros((32, K)) for n in ("on", "od", "dn", "dd")}
    for gi, (week, h, a) in enumerate(sched):
        state = sim.initial_kickoff_state(rng, tables["opening_pool"])
        ns["PLAYLOG"].clear()
        _G["log"].clear()
        _G["tolog"].clear() if "tolog" in _G else None
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
        if rates is not None:
            offt = np.where(offhome == 1, h, a)
            deft = np.where(offhome == 1, a, h)
            np.add.at(dg["on"], offt, rates[1][idx].astype(float))
            np.add.at(dg["od"], offt, rates[0][idx].astype(float))
            np.add.at(dg["dn"], deft, rates[1][idx].astype(float))
            np.add.at(dg["dd"], deft, rates[0][idx].astype(float))
        if "ep_model" in _G:
            lg = np.array(_G["log"], dtype=np.float64).reshape(-1, 12)
            pl = np.array(ns["PLAYLOG"], dtype=np.float64).reshape(-1, 3)
            tl = np.array(_G["tolog"], dtype=np.float64).reshape(-1, 2)
            assert len(lg) == len(pl) == len(tl) and np.array_equal(lg[:, 11], pl[:, 1])
            epa = chain2_epa(lg, tl, _G["ep_model"], pl[:, 0])[keep]
        elif "ep_tab" in _G:
            lg = np.array(_G["log"], dtype=np.float64).reshape(-1, 12)
            pl = np.array(ns["PLAYLOG"], dtype=np.float64).reshape(-1, 3)
            assert len(lg) == len(pl) and np.array_equal(lg[:, 11], pl[:, 1])
            epa = chain_epa(lg, pl[:, 0], _G["ep_tab"])[keep]
        else:
            epa = arr["epa"][idx] + (shift - bias) / gain + arr["adv"][idx]
        yards = arr["yards"][idx] + shift
        turn = np.where(arr["int"][idx] == 1, 1, np.where(arr["fl"][idx] == 1, 2, 0)).astype(np.int8)
        chunks.append(np.rec.fromarrays([np.full(len(idx), gi, dtype=np.int16), offhome, kind, epa.astype(np.float32), yards.astype(np.float32), turn], names="g,offhome,kind,epa,yards,turn"))
        total, margin = float(rec["total"]), float(rec["margin"])
        gm_rows.append((gi, (total + margin) / 2.0, (total - margin) / 2.0, bool(rec["went_ot"]), bool(cap_hit)))
    plays = np.concatenate(chunks) if chunks else np.empty(0)
    if rates is not None:
        dd = OUT / "diag"
        dd.mkdir(parents=True, exist_ok=True)
        np.savez(dd / f"{cfg.get('name', 'x')}_{world}_{sidx}.npz", **dg)
    return world, sidx, gm_rows, plays


def gen_eval_d(variant, scale, yard_gain, worlds, seasons, workers, seed, yard_bias):
    c25.ensure_policies()
    import mod25_generator as gen

    cfgj = json.dumps(dict(DV[variant], seed=seed, name=f"{variant}_s{scale:g}"))
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


def cmd_grid2(args):
    import mod25_generator as gen

    OUT.mkdir(parents=True, exist_ok=True)
    real = c25.real_gate_metrics()["2011_2017"]
    tg = json.loads((REPO / "artifacts" / "mod25_generator" / "real_targets.json").read_text())["gates"]
    rows = {}
    for sc in [float(x) for x in args.scales.split(",")]:
        games, plays, latents = gen_full(args.variant, sc, args.worlds, args.seasons, args.workers, args.seed)
        ts = gen.team_stats_from_plays(plays)
        keep = [w * 1000 + s + 1 for w in range(args.worlds) for s in range(2, args.seasons)]
        m, ac, dd = gen.all_metrics(games, ts, keep)
        m["r2_pooled"] = float(np.corrcoef(dd["x"], dd["y"])[0, 1] ** 2)
        m["nonstrength_var"] = float(m["margin_sd"] ** 2 * (1 - m["r2_pooled"]))
        g, tgm = x_frame(games, ts)
        g = g[g["season"].isin(keep)].dropna(subset=["x", "y"])
        tgm = tgm[tgm["season"].isin(keep)]
        sd = strdiag_one(g, tgm)
        w = sd["within"]
        m["within_sd_off"] = w["off_epa_per_play_within_sd"]
        m["between_sd_off"] = w["off_epa_per_play_between_sd"]
        m["slope_pts_per_epa"] = sd["conv"]["y_on_epa_slope"]
        m["resid_var_margin_on_epa"] = sd["conv"]["y_resid_var"]
        loss = 0.0
        for k in ("epa_autocorr_lag1", "r2_w1_4", "r2_w5_9", "r2_w10_18"):
            hw = (tg[k]["hi"] - tg[k]["lo"]) / 2.0
            loss += ((m[k] - real[k]) / hw) ** 2
        loss += ((m["between_sd_off"] - 0.0928) / 0.005) ** 2
        m["loss"] = float(loss)
        rows[str(sc)] = m
        print(sc, json.dumps({k: round(v_, 4) for k, v_ in m.items() if k in ("epa_autocorr_lag1", "r2_w1_4", "r2_w5_9", "r2_w10_18", "within_sd_off", "between_sd_off", "slope_pts_per_epa", "resid_var_margin_on_epa", "margin_sd", "loss")}), flush=True)
    (OUT / f"grid2_{args.tag}.json").write_text(json.dumps({"real_2011_2017": real, "rows": rows}, indent=1))


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
    ab = sub.add_parser("ablate")
    ab.add_argument("--variants", default="crt")
    ab.add_argument("--games", type=int, default=12000)
    ab.add_argument("--workers", type=int, default=6)
    ab.add_argument("--seed", type=int, default=5)
    ab.add_argument("--tag", default="h1")
    ab.add_argument("--cache", default="C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad")
    sc_ = sub.add_parser("simcache")
    sc_.add_argument("--variants", default="crp04")
    sc_.add_argument("--games", type=int, default=12000)
    sc_.add_argument("--workers", type=int, default=6)
    sc_.add_argument("--seed", type=int, default=5)
    sc_.add_argument("--cache", default="C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad")
    pd_ = sub.add_parser("possdiag")
    pd_.add_argument("--variants", default="crp04")
    pd_.add_argument("--tag", default="p1")
    pd_.add_argument("--cache", default="C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad")
    sg = sub.add_parser("strdiag")
    sg.add_argument("--variants", default="crf@0.75")
    sg.add_argument("--worlds", type=int, default=6)
    sg.add_argument("--seasons", type=int, default=8)
    sg.add_argument("--workers", type=int, default=6)
    sg.add_argument("--seed", type=int, default=9)
    sg.add_argument("--tag", default="s1")
    ed = sub.add_parser("epadiag")
    ed.add_argument("--variants", default="crf")
    ed.add_argument("--tag", default="e1")
    ed.add_argument("--cache", default="C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad")
    sub.add_parser("epfit")
    g2 = sub.add_parser("grid2")
    g2.add_argument("--variant", default="crj")
    g2.add_argument("--scales", default="0.6,0.75,0.9,1.0,1.15")
    g2.add_argument("--worlds", type=int, default=6)
    g2.add_argument("--seasons", type=int, default=8)
    g2.add_argument("--workers", type=int, default=6)
    g2.add_argument("--seed", type=int, default=21)
    g2.add_argument("--tag", default="j1")
    pf = sub.add_parser("pacefit")
    pf.add_argument("--variants", default="crt,crp04,crp08,crp12,crp16")
    pf.add_argument("--games", type=int, default=6000)
    pf.add_argument("--workers", type=int, default=6)
    pf.add_argument("--seed", type=int, default=5)
    tc = sub.add_parser("tiltcoef")
    tc.add_argument("--base", default="crk_s1")
    rd = sub.add_parser("ratediag")
    rd.add_argument("--variants", default="crt")
    rd.add_argument("--tag", default="r1")
    pdg = sub.add_parser("pacediag")
    pdg.add_argument("--variants", default="crw")
    pdg.add_argument("--tag", default="e1")
    pdg.add_argument("--cache", default="C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad")
    cd = sub.add_parser("corrdiag")
    cd.add_argument("--variants", default="crw")
    cd.add_argument("--tag", default="e1")
    cd.add_argument("--cache", default="C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad")
    cv_ = sub.add_parser("covdecomp")
    cv_.add_argument("--variants", default="crj")
    cv_.add_argument("--tag", default="k1")
    cv_.add_argument("--cache", default="C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad")
    dd_ = sub.add_parser("downdiag")
    dd_.add_argument("--variants", default="crj")
    dd_.add_argument("--tag", default="k1")
    dd_.add_argument("--cache", default="C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad")
    cs_ = sub.add_parser("covsim")
    cs_.add_argument("--variants", default="crj@0.9")
    cs_.add_argument("--worlds", type=int, default=6)
    cs_.add_argument("--seasons", type=int, default=8)
    cs_.add_argument("--workers", type=int, default=6)
    cs_.add_argument("--seed", type=int, default=21)
    cs_.add_argument("--tag", default="k1")
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
    {"decomp": cmd_decomp, "tilt": cmd_tilt, "ipwcheck": cmd_ipwcheck, "grid": cmd_grid, "gate": cmd_gate, "corrdiag": cmd_corrdiag, "pacediag": cmd_pacediag, "ratediag": cmd_ratediag, "tiltcoef": cmd_tiltcoef, "ablate": cmd_ablate, "pacefit": cmd_pacefit, "simcache": cmd_simcache, "possdiag": cmd_possdiag, "strdiag": cmd_strdiag, "epadiag": cmd_epadiag, "epfit": cmd_epfit, "grid2": cmd_grid2, "covdecomp": cmd_covdecomp, "downdiag": cmd_downdiag, "covsim": cmd_covsim}[args.cmd](args)



def corr_blocks(pk):
    pk = pk.copy()
    pk["pts"] = pk["po"] + pk["pdf"]
    pk["td"] = ((pk["po"] >= 6) | (pk["pdf"] >= 6)).astype(float)
    pk["fgp"] = ((pk["po"] == 3) | (pk["pdf"] == 3)).astype(float)
    out = {}
    g = pk.groupby("g")
    n = g.size()
    pts = g["pts"].sum()
    out["corr_n_pts"] = float(np.corrcoef(n, pts)[0, 1])
    out["corr_n_tdpts"] = float(np.corrcoef(n, g["td"].sum())[0, 1])
    out["corr_n_fgcount"] = float(np.corrcoef(n, g["fgp"].sum())[0, 1])
    out["corr_n_secs"] = float(np.corrcoef(n, g["secs"].sum())[0, 1])
    out["secs_total_mean"] = float(g["secs"].sum().mean())
    out["secs_total_sd"] = float(g["secs"].sum().std())
    out["n_sd"] = float(n.std())
    out["corr_n_plays"] = float(np.corrcoef(n, g["n"].sum())[0, 1])
    out["plays_total_sd"] = float(g["n"].sum().std())
    avg_secs = g["secs"].sum() / n
    ppp = pts / n
    out["corr_avgsecs_ppp"] = float(np.corrcoef(avg_secs, ppp)[0, 1])
    out["corr_avgsecs_n"] = float(np.corrcoef(avg_secs, n)[0, 1])
    out["avgsecs_sd"] = float(avg_secs.std())
    for nm, qs in (("h1", (1, 2)), ("h2", (3, 4))):
        d = pk[pk["q0"].isin(qs)]
        gg = d.groupby("g")
        nn = gg.size()
        pp = gg["pts"].sum()
        out[f"corr_n_pts_{nm}"] = float(np.corrcoef(nn, pp.reindex(nn.index))[0, 1])
        out[f"n_sd_{nm}"] = float(nn.std())
    d = pk[pk["q0"].isin((1, 2, 3))]
    out["corr_secs_pts_poss_q13"] = float(np.corrcoef(d["secs"], d["pts"])[0, 1])
    out["corr_plays_pts_poss_q13"] = float(np.corrcoef(d["n"], d["pts"])[0, 1])
    for nm, m_ in (("td", d["td"] == 1), ("fg", d["fgp"] == 1), ("none", (d["td"] == 0) & (d["fgp"] == 0))):
        out[f"secs_{nm}_mean"] = float(d.loc[m_, "secs"].mean())
        out[f"secs_{nm}_sd"] = float(d.loc[m_, "secs"].std())
        out[f"plays_{nm}_mean"] = float(d.loc[m_, "n"].mean())
        out[f"share_{nm}"] = float(m_.mean())
    d2 = pk[pk["q0"].isin((1, 2, 3))].copy()
    d2["prev_secs"] = d2.groupby("g")["secs"].shift(1)
    d2["prev_td"] = d2.groupby("g")["td"].shift(1)
    d2 = d2.dropna(subset=["prev_secs"])
    out["corr_prevsecs_secs"] = float(np.corrcoef(d2["prev_secs"], d2["secs"])[0, 1])
    out["secs_after_td"] = float(d2.loc[d2["prev_td"] == 1, "secs"].mean())
    out["secs_after_nontd"] = float(d2.loc[d2["prev_td"] == 0, "secs"].mean())
    gs = g["secs"].sum()
    out["corr_gsecs_pts"] = float(np.corrcoef(gs, pts)[0, 1])
    return out


def cmd_corrdiag(args):
    res = {}
    for nm, ss in (("real_train", TRAIN), ("real_eval", EVAL)):
        P, M, meta, pbp = real_load(ss)
        r, pk = poss_analysis(P, M)
        res[nm] = corr_blocks(pk)
    for v in args.variants.split(","):
        S = pd.read_parquet(Path(args.cache) / f"sim_{v}.parquet")
        MS = pd.Series(pd.read_parquet(Path(args.cache) / f"simM_{v}.parquet")["m"].to_numpy())
        r, pk = poss_analysis(S, MS)
        res["sim_" + v] = corr_blocks(pk)
    names = list(res)
    print(f"{'key':28s} " + " ".join(f"{n:>12s}" for n in names))
    for k in res[names[0]]:
        print(f"{k:28s} " + " ".join(f"{res[n][k]:12.4f}" for n in names))
    (OUT / f"corrdiag_{args.tag}.json").write_text(json.dumps(res, indent=1))



def pace_blocks(P):
    P = P[P["qtr"] <= 4].sort_values("g", kind="stable").reset_index(drop=True)
    out = {}
    el = P["el"].to_numpy(dtype=float)
    out["el_mean"] = float(el.mean())
    out["el_median"] = float(np.median(el))
    out["el_sd"] = float(el.std())
    out["el_gt60"] = float(np.mean(el > 60))
    out["el_gt45"] = float(np.mean(el > 45))
    out["el_eq0"] = float(np.mean(el == 0))
    n = P.groupby("g").size()
    out["plays_mean"] = float(n.mean())
    out["plays_sd"] = float(n.std())
    out["g_el_sum_mean"] = float(P.groupby("g")["el"].sum().mean())
    out["g_el_sum_sd"] = float(P.groupby("g")["el"].sum().std())
    gg = P["g"].to_numpy()
    for L in (1, 2, 3):
        ok = gg[L:] == gg[:-L]
        out[f"el_ac{L}"] = float(np.corrcoef(el[:-L][ok], el[L:][ok])[0, 1])
    rp = P["code"].isin([0, 1]).to_numpy()
    for c in (0, 1, 2, 3, 6):
        m = (P["code"] == c).to_numpy()
        out[f"el_mean_code{c}"] = float(el[m].mean())
        out[f"el_sd_code{c}"] = float(el[m].std())
        out[f"share_code{c}"] = float(m.mean())
    out["g_mean_el_sd"] = float((P.groupby("g")["el"].mean()).std())
    for q in (1, 2, 3, 4):
        m = (P["qtr"] == q).to_numpy()
        out[f"plays_q{q}"] = float(m.sum() / P["g"].nunique())
        out[f"el_mean_q{q}"] = float(el[m].mean())
    q = P.groupby(["g", "qtr"]).size().unstack().fillna(0)
    for c_ in q.columns:
        out[f"plays_sd_q{c_}"] = float(q[c_].std())
    gs = P["gsr"].to_numpy(dtype=float)
    ng = P["g"].nunique()
    for lo, hi in ((2700, 3601), (1920, 2700), (1800, 1920), (900, 1800), (300, 900), (120, 300), (0, 120)):
        m = (gs >= lo) & (gs < hi)
        w = P[m].groupby("g").size().reindex(n.index).fillna(0)
        out[f"w{lo}_mean"] = float(w.mean())
        out[f"w{lo}_sd"] = float(w.std())
        out[f"w{lo}_el_mean"] = float(el[m].mean())
        out[f"w{lo}_el_sd"] = float(el[m].std())
        out[f"w{lo}_el_p99"] = float(np.quantile(el[m], 0.99))
        out[f"w{lo}_el_max"] = float(el[m].max())
    out["plays_cov_between_qtrs"] = float(np.cov(q.to_numpy().T)[np.triu_indices(4, 1)].mean())
    out["plays_var_sum_qtrs"] = float(q.var().sum())
    return out


def cmd_pacediag(args):
    res = {}
    for nm, ss in (("real_train", TRAIN), ("real_eval", EVAL)):
        P, M, meta, pbp = real_load(ss)
        res[nm] = pace_blocks(P)
    for v in args.variants.split(","):
        S = pd.read_parquet(Path(args.cache) / f"sim_{v}.parquet")
        res["sim_" + v] = pace_blocks(S)
    names = list(res)
    print(f"{'key':28s} " + " ".join(f"{n:>12s}" for n in names))
    for k in res[names[0]]:
        print(f"{k:28s} " + " ".join(f"{res[n][k]:12.4f}" for n in names))
    (OUT / f"pacediag_{args.tag}.json").write_text(json.dumps(res, indent=1))



def mom_sd(num, den, floor=200):
    m = den >= floor
    r = num[m] / den[m]
    p = num[m].sum() / den[m].sum()
    nv = float((p * (1 - p) / den[m]).mean())
    return float(np.sqrt(max(float(r.var(ddof=1)) - nv, 0.0)))


def real_rate_spread(seasons):
    sim.PBP_SNAPSHOT_DIR = m25.SNAP
    pbp = sim.load_reg_seasons(tuple(seasons))
    trans = sim.build_transition_frame(pbp)
    at = c25.attrs_from(pbp, trans)
    D, I = rate_defs(trans, at)
    ok, fk = team_keys(pbp, trans)
    out = {}
    for side, key in (("off", ok), ("def", fk)):
        for k, nm in enumerate(RATE_NAMES):
            df = pd.DataFrame({"t": key, "n": I[:, k].astype(float), "d": D[:, k].astype(float)}).groupby("t").sum()
            out[f"{side}_{nm}"] = mom_sd(df["n"].to_numpy(), df["d"].to_numpy())
    return out


def cmd_ratediag(args):
    res = {"real_train": real_rate_spread(TRAIN), "real_eval": real_rate_spread(EVAL)}
    for v in args.variants.split(","):
        fs = sorted((OUT / "diag").glob(f"{v}_*.npz"))
        acc = {n: np.concatenate([np.load(f)[n] for f in fs]) for n in ("on", "od", "dn", "dd")}
        o = {}
        for k, nm in enumerate(RATE_NAMES):
            o[f"off_{nm}"] = mom_sd(acc["on"][:, k], acc["od"][:, k])
            o[f"def_{nm}"] = mom_sd(acc["dn"][:, k], acc["dd"][:, k])
        res["sim_" + v] = o
    names = list(res)
    print(f"{'rate sd':16s} " + " ".join(f"{n:>12s}" for n in names))
    for k in res[names[0]]:
        print(f"{k:16s} " + " ".join(f"{res[n][k]:12.5f}" for n in names))
    (OUT / f"ratediag_{args.tag}.json").write_text(json.dumps(res, indent=1))



def cmd_tiltcoef(args):
    import mod25_generator as gen

    sim.PBP_SNAPSHOT_DIR = m25.SNAP
    pbp = sim.load_reg_seasons(tuple(TRAIN))
    trans = sim.build_transition_frame(pbp)
    at = c25.attrs_from(pbp, trans)
    res, D, I = rate_effects(trans, pbp, at, TRAIN)
    fit = gen.load_fit()
    qb = fit["qb"]
    var_off = max(1e-6, fit["off"]["var_mu"] - qb["backup_rate"] * (1 - qb["backup_rate"]) * qb["backup_off_epa_effect"] ** 2)
    zo, zd = float(np.sqrt(var_off)), float(np.sqrt(fit["def"]["var_mu"]))
    real = real_rate_spread(TRAIN)
    fs = sorted((OUT / "diag").glob(f"{args.base}_*.npz"))
    acc = {n: np.concatenate([np.load(f)[n] for f in fs]) for n in ("on", "od", "dn", "dd")}
    out = {}
    for k, r in enumerate(res):
        nm = r["name"]
        so = mom_sd(acc["on"][:, k], acc["od"][:, k])
        sd_ = mom_sd(acc["dn"][:, k], acc["dd"][:, k])
        pq = r["p"] * (1 - r["p"])
        out[nm] = {
            "off": float(np.sign(r["off_corr_epa"]) * max(0.0, real[f"off_{nm}"] - so) / (pq * zo)),
            "def": float(np.sign(r["def_corr_epa"]) * max(0.0, real[f"def_{nm}"] - sd_) / (pq * zd)),
            "real_off": real[f"off_{nm}"], "real_def": real[f"def_{nm}"], "base_off": so, "base_def": sd_,
            "p": r["p"], "off_corr_epa": r["off_corr_epa"], "def_corr_epa": r["def_corr_epa"],
        }
    (OUT / "tilt_coef.json").write_text(json.dumps(out, indent=1))
    for nm, v in out.items():
        print(nm, json.dumps({a: round(b, 5) for a, b in v.items()}))


REV_T = (2700, 1800, 900, 300)


def blup_mu(meta, M, iters=300):
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
        sa2_n = float((beta[1:] @ beta[1:]) + se2 * np.trace(Ainv[1:, 1:])) / p
        done = abs(se2_n - se2) < 1e-6 and abs(sa2_n - sa2) < 1e-6
        se2, sa2 = se2_n, sa2_n
        if done:
            break
    lam = se2 / sa2
    A = X.T @ X
    A[1:, 1:] += lam * np.eye(p)
    Ainv = np.linalg.inv(A)
    beta = Ainv @ X.T @ y
    h = np.einsum("ij,jk,ik->i", X, Ainv, X)
    return (X @ beta - h * y) / (1 - h)


def slope_se(x, y):
    b = np.polyfit(x, y, 1)
    r = y - np.polyval(b, x)
    return float(b[0]), float(np.sqrt(r.var(ddof=2) / ((x - x.mean()) ** 2).sum()))


def real_reversion(seasons):
    sim.PBP_SNAPSHOT_DIR = m25.SNAP
    pbp = sim.load_reg_seasons(tuple(seasons))
    tr = sim.build_transition_frame(pbp).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    gid, gids = pd.factorize(tr["game_id"])
    last = tr.groupby(gid).tail(1)
    Mall = pd.Series(last["home_margin_post"].to_numpy(), index=gid[last.index.to_numpy()])
    meta = pbp.drop_duplicates("game_id").set_index("game_id").loc[gids, ["season", "home_team", "away_team"]].reset_index(drop=True)
    mu = blup_mu(meta, Mall)
    reg = tr["qtr_actual"].to_numpy() <= 4
    R = pd.DataFrame({"g": gid[reg], "gsr": tr["gsr_actual"].to_numpy()[reg], "pre": tr["home_margin_pre"].to_numpy()[reg], "post": tr["home_margin_post"].to_numpy()[reg]})
    fin = R.groupby("g")["post"].last()
    out = {}
    for t in REV_T:
        f = R[R["gsr"] <= t].groupby("g").head(1)
        gi = f["g"].to_numpy()
        fut = fin.reindex(gi).to_numpy() - f["pre"].to_numpy()
        r = f["gsr"].to_numpy() / 3600.0
        m = mu[gi]
        b, se = slope_se(f["pre"].to_numpy(), fut)
        ba, sea = slope_se(f["pre"].to_numpy() - (1 - r) * m, fut - r * m)
        out[f"t{t}"] = {"raw": b, "raw_se": se, "adj": ba, "adj_se": sea, "n": int(len(f))}
    return out


def sim_reversion(S, MS):
    r, pk = poss_analysis(S, MS)
    out = {f"t{t}": r[f"fb_slope_diff_future_t{t}"] for t in REV_T}
    out["cov_total"] = lag_analysis(pk, S)["cov_total"]
    return out, r


def cmd_ablate(args):
    global SIDE_TRK
    SIDE_TRK = True
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "ablate_real.json"
    if f.exists():
        real = json.loads(f.read_text())
    else:
        real = {"train": real_reversion(TRAIN), "eval": real_reversion(EVAL)}
        f.write_text(json.dumps(real, indent=1))
    res = {}
    t0 = time.time()
    for v in args.variants.split(","):
        cf = Path(args.cache) / f"sim_{v}.parquet"
        if cf.exists():
            S = pd.read_parquet(cf)
            MS = pd.Series(pd.read_parquet(Path(args.cache) / f"simM_{v}.parquet")["m"].to_numpy())
        else:
            S, MS = run_neutral(v, args.games, args.workers, args.seed)
            S.to_parquet(cf)
            pd.DataFrame({"m": MS.to_numpy()}).to_parquet(Path(args.cache) / f"simM_{v}.parquet")
        rv, r = sim_reversion(S, MS)
        am = np.abs(MS.to_numpy())
        rv["sd"] = float(MS.std())
        for k in (3, 7, 10, 14, 17):
            rv[f"m{k}"] = float(np.mean(am == k))
        rv["n_games"] = int(len(MS))
        rv["plays_per_game"] = float(len(S) / len(MS))
        res[v] = rv
        print(v, f"{time.time() - t0:.0f}s", json.dumps({k: round(x, 3) for k, x in rv.items()}), flush=True)
        (OUT / f"ablate_{args.tag}.json").write_text(json.dumps({"real": real, "sim": res}, indent=1))
    print("real", json.dumps({p: {t: (round(d["raw"], 3), round(d["adj"], 3), round(d["adj_se"], 3)) for t, d in real[p].items()} for p in real}))


def cmd_pacefit(args):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for nm, ss in (("real_train", TRAIN), ("real_eval", EVAL)):
        P, M, meta, pbp = real_load(ss)
        r, pk = poss_analysis(P, M)
        res[nm] = {**pace_blocks(P), **corr_blocks(pk)}
    for v in args.variants.split(","):
        S, MS = run_neutral(v, args.games, args.workers, args.seed)
        r, pk = poss_analysis(S, MS)
        res["sim_" + v] = {**pace_blocks(S), **corr_blocks(pk)}
        print(v, flush=True)
    keys = ["plays_cov_between_qtrs", "plays_sd", "plays_mean", "plays_var_sum_qtrs", "corr_n_pts", "corr_n_secs", "g_el_sum_sd"]
    print(f"{'key':26s} " + " ".join(f"{n[:11]:>11s}" for n in res))
    for k in keys:
        print(f"{k:26s} " + " ".join(f"{res[n][k]:11.3f}" for n in res))
    (OUT / "pacefit.json").write_text(json.dumps(res, indent=1))


def cmd_simcache(args):
    for v in args.variants.split(","):
        S, MS = run_neutral(v, args.games, args.workers, args.seed)
        S.to_parquet(Path(args.cache) / f"sim_{v}.parquet")
        pd.DataFrame({"m": MS.to_numpy()}).to_parquet(Path(args.cache) / f"simM_{v}.parquet")
        print(v, len(S), flush=True)


YL_BINS = [0, 10, 20, 35, 50, 65, 80, 101]
LEN_BINS = [0, 3, 5, 7, 9, 12, 100]
OUTCOMES = ["td", "fg", "fg_miss", "punt", "turnover", "downs", "def_score", "end_half", "other"]


def drive_table(P):
    P = P.sort_values("g", kind="stable").reset_index(drop=True)
    side = track_sides(P)
    _, poss = assign_sides(P)
    P = P.assign(trk=side, poss=poss)
    R = P[P["qtr"] <= 4]
    gp = R.groupby("poss", sort=True)
    D = pd.DataFrame(
        {
            "g": gp["g"].first(), "trk": gp["trk"].first(), "yl0": gp["yl"].first(), "ylmin": gp["yl"].min(), "q0": gp["qtr"].first(),
            "gsr0": gp["gsr"].first(), "n": gp.size(), "secs": gp["el"].sum(), "po": gp["po"].sum(), "pdf": gp["pdf"].sum(),
            "code": gp["code"].last(), "tov": gp["tov"].last(), "flip": gp["flip"].last(), "sd0": gp["sd"].first(),
            "ql": gp["qtr"].last(), "elmax": gp["el"].max(),
        }
    ).reset_index()
    ng_ = D["g"].to_numpy()
    nq_ = D["q0"].to_numpy()
    last_g = np.r_[ng_[1:] != ng_[:-1], True]
    last_h = np.r_[(ng_[1:] == ng_[:-1]) & (nq_[1:] >= 3), False] & (D["ql"].to_numpy() <= 2)
    term = last_g | last_h
    po, pdf, code, tov, flip = (D[c].to_numpy() for c in ("po", "pdf", "code", "tov", "flip"))
    oc = np.select(
        [po >= 6, (po == 3) & (code == 3), pdf > 0, (code == 3) & (po == 0), code == 2, term, (tov == 1) & flip, flip & (po == 0), ~flip],
        ["td", "fg", "def_score", "fg_miss", "punt", "end_half", "turnover", "downs", "end_half"],
        default="other",
    )
    D["oc"] = oc
    D["pts"] = po
    D["rz"] = D["ylmin"] <= 20
    return D, R


def drive_tables(P):
    D, R = drive_table(P)
    out = {}
    ng = D["g"].nunique()
    out["n_drives"] = int(len(D))
    out["drives_per_game"] = float(len(D) / ng)
    out["pts_per_drive"] = float(D["pts"].mean())
    out["pts_per_drive_var"] = float(D["pts"].var())
    cnt = D.groupby("g").size()
    out["drives_per_game_var"] = float(cnt.var())
    cA = D[D["trk"] == 0].groupby("g").size().reindex(cnt.index).fillna(0)
    cB = D[D["trk"] == 1].groupby("g").size().reindex(cnt.index).fillna(0)
    out["drives_per_team_var"] = float((cA.var() + cB.var()) / 2)
    out["plays_per_drive"] = float(D["n"].mean())
    out["secs_per_drive"] = float(D["secs"].mean())
    out["mix"] = {k: float((D["oc"] == k).mean()) for k in OUTCOMES}
    for lab, m in (("q1_3", D["q0"] <= 3), ("q4", D["q0"] == 4)):
        d = D[m]
        out["mix_" + lab] = {k: float((d["oc"] == k).mean()) for k in OUTCOMES}
        out["pts_" + lab] = float(d["pts"].mean())
    D["ylb"] = pd.cut(D["yl0"], YL_BINS, right=False)
    D["lnb"] = pd.cut(D["n"], LEN_BINS, right=False)
    out["by_start"] = {str(k): dict(n=int(len(g)), pts=float(g["pts"].mean()), td=float((g["oc"] == "td").mean()), fg=float((g["oc"] == "fg").mean()), len=float(g["n"].mean())) for k, g in D.groupby("ylb", observed=True)}
    out["by_len"] = {str(k): dict(n=int(len(g)), share=float(len(g) / len(D)), pts=float(g["pts"].mean()), td=float((g["oc"] == "td").mean()), fg=float((g["oc"] == "fg").mean())) for k, g in D.groupby("lnb", observed=True)}
    rz = D[D["rz"]]
    out["rz"] = dict(share=float(D["rz"].mean()), td=float((rz["oc"] == "td").mean()), fg=float((rz["oc"] == "fg").mean()), fg_miss=float((rz["oc"] == "fg_miss").mean()), tov=float((rz["oc"] == "turnover").mean()), downs=float((rz["oc"] == "downs").mean()), pts=float(rz["pts"].mean()))
    f = R[(R["code"] == 3)].copy()
    f["b"] = pd.cut(f["yl"], [0, 10, 20, 25, 30, 35, 40, 50])
    f["mk"] = (f["po"] == 3)
    out["fg_make"] = {str(k): dict(n=int(len(g)), make=float(g["mk"].mean())) for k, g in f.groupby("b", observed=True)}
    fd = R[(R["down"] == 4) & R["code"].isin([0, 1, 2, 3])].copy()
    fd["b"] = pd.cut(fd["yl"], [0, 10, 20, 25, 30, 35, 40, 50, 60, 75, 101])
    fd["fgp"] = fd["code"] == 3
    fd["pn"] = fd["code"] == 2
    fd["go"] = fd["code"].isin([0, 1])
    out["fourth"] = {str(k): dict(n=int(len(g)), fg=float(g["fgp"].mean()), punt=float(g["pn"].mean()), go=float(g["go"].mean())) for k, g in fd.groupby("b", observed=True)}
    okg = (D.groupby("g")["elmax"].max() <= EL_MAX)
    out["clean_game_share"] = float(okg.mean())
    D = D[D["g"].isin(okg.index[okg.to_numpy()])].copy()
    G = D.groupby("g").agg(n=("n", "sum"), nd=("n", "size"), pts=("pts", "sum"), secs=("secs", "sum"))
    G["pdf"] = D.groupby("g")["pdf"].sum()
    G["tp"] = G["pts"] + G["pdf"]
    G["ppd"] = G["n"] / G["nd"]
    G["spp"] = G["secs"] / G["n"]
    cc = lambda a, b: float(np.corrcoef(G[a], G[b])[0, 1])
    out["corr_plays_pts"] = cc("n", "tp")
    out["corr_ndrives_pts"] = cc("nd", "tp")
    out["corr_plays_ndrives"] = cc("n", "nd")
    out["corr_ppd_pts"] = cc("ppd", "tp")
    out["corr_spp_pts"] = cc("spp", "tp")
    out["corr_spp_plays"] = cc("spp", "n")
    out["corr_ndrives_ppd"] = cc("nd", "ppd")
    out["plays_var"] = float(G["n"].var())
    out["tp_var"] = float(G["tp"].var())
    out["corr_drive_len_pts"] = float(np.corrcoef(D["n"], D["pts"])[0, 1])
    return out


def flat(d, pre=""):
    r = {}
    for k, v in d.items():
        if isinstance(v, dict):
            r.update(flat(v, pre + k + "."))
        else:
            r[pre + k] = v
    return r


def cmd_possdiag(args):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for nm, ss in (("real_train", TRAIN), ("real_eval", EVAL)):
        P, M, meta, pbp = real_load(ss)
        res[nm] = drive_tables(P)
        print(nm, flush=True)
    for v in args.variants.split(","):
        S = pd.read_parquet(Path(args.cache) / f"sim_{v}.parquet")
        res[v] = drive_tables(S)
        print(v, flush=True)
    (OUT / f"possdiag_{args.tag}.json").write_text(json.dumps(res, indent=1))
    F = {k: flat(v) for k, v in res.items()}
    keys = list(F["real_train"])
    lines = [f"{'metric':44s} " + " ".join(f"{n:>11s}" for n in F)]
    for k in keys:
        lines.append(f"{k:44s} " + " ".join(f"{F[n].get(k, float('nan')):11.4f}" for n in F))
    (OUT / f"possdiag_{args.tag}.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def gen_full(variant, scale, worlds, seasons, workers, seed, yard_gain=2.0, yard_bias=0.75):
    c25.ensure_policies()
    import mod25_generator as gen

    cfgj = json.dumps(dict(DV[variant], seed=seed, name=f"{variant}_s{scale:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": scale, "yard_gain": yard_gain, "def_sign": 1.0, "yard_bias": yard_bias, "drift": 1.0, "mech": cfgj})
    gen.init_worker = d_gen_init
    gen.play_season = d_play_season
    games, plays, latents, elapsed = gen.run_generation(setting, worlds, seasons, workers, seed, progress=False)
    return games, plays, latents


def x_frame(games, ts):
    import mod25_generator as gen
    from nfl_ats.constants import DEFAULT_OFFSEASON_RETENTION
    from nfl_ats.features import build_team_game_metrics, build_team_states

    sched = gen.schedules_from_games(games)
    tg = build_team_game_metrics(sched, ts)
    wk = games.set_index("game_id")["week"]
    tg = tg.assign(week=tg["game_id"].map(wk))
    states = build_team_states(tg.drop(columns=["week"]))
    states = states.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    grp = states.groupby("team", sort=False)
    gap = (states["season"] - grp["season"].shift(1)).to_numpy(dtype=float)
    pre = {"game_id": states["game_id"], "team": states["team"]}
    for metric in ("off_epa_per_play", "def_epa_per_play"):
        value = grp[f"state_{metric}"].shift(1).to_numpy(dtype=float)
        lm = grp[f"league_mean_{metric}"].shift(1).to_numpy(dtype=float)
        ret = DEFAULT_OFFSEASON_RETENTION ** np.maximum(1.0, np.nan_to_num(gap, nan=1.0))
        pre[metric] = np.where(gap > 0, lm + ret * (value - lm), value)
    st = pd.DataFrame(pre).set_index(["game_id", "team"])
    g = games.copy()
    for side in ("home", "away"):
        idx = pd.MultiIndex.from_arrays([g["game_id"], g[f"{side}_team"]])
        for c in ("off_epa_per_play", "def_epa_per_play"):
            g[f"{side}_state_{c}"] = st[c].reindex(idx).to_numpy()
    g["x"] = (g["home_state_off_epa_per_play"] - g["home_state_def_epa_per_play"]) - (g["away_state_off_epa_per_play"] - g["away_state_def_epa_per_play"])
    g["y"] = g["home_score"] - g["away_score"]
    return g, tg


def loo_season_net(g, tg):
    t = tg[["game_id", "team", "season", "off_epa_per_play", "def_epa_per_play"]].copy()
    t["net"] = t["off_epa_per_play"] - t["def_epa_per_play"]
    s = t.groupby(["team", "season"])["net"].agg(["sum", "count"])
    t = t.merge(s, left_on=["team", "season"], right_index=True)
    t["loo"] = (t["sum"] - t["net"]) / (t["count"] - 1)
    m = t.set_index(["game_id", "team"])["loo"]
    h = m.reindex(pd.MultiIndex.from_arrays([g["game_id"], g["home_team"]])).to_numpy()
    a = m.reindex(pd.MultiIndex.from_arrays([g["game_id"], g["away_team"]])).to_numpy()
    return h - a


def bucket_corr(g, col_a, col_b):
    out = {}
    for lo, hi in ((1, 4), (5, 9), (10, 18)):
        b = g[(g["week"] >= lo) & (g["week"] <= hi)].dropna(subset=[col_a, col_b])
        out[f"w{lo}_{hi}"] = float(np.corrcoef(b[col_a], b[col_b])[0, 1]) if len(b) > 30 else float("nan")
    return out


def within_stats(tg):
    t = tg.copy()
    t["net"] = t["off_epa_per_play"] - t["def_epa_per_play"]
    out = {}
    for c in ("off_epa_per_play", "def_epa_per_play", "net"):
        dev = t[c] - t.groupby("season")[c].transform("mean")
        tm = dev.groupby([t["team"], t["season"]]).transform("mean")
        out[c + "_total_sd"] = float(dev.std())
        out[c + "_within_sd"] = float((dev - tm).std())
        out[c + "_between_sd"] = float(tm.std())
    return out


def latent_rows(g, latents):
    shock = -0.05668721441704449
    n = len(g)
    zl = np.zeros(n)
    gid = g["game_id"].to_numpy()
    ht = g["home_team"].to_numpy()
    at = g["away_team"].to_numpy()
    wk = g["week"].to_numpy()
    for i in range(n):
        w = int(gid[i][1:5])
        s = int(gid[i][6:8])
        weekly, qb = latents[(w, s)]
        h = int(ht[i].split("T")[1])
        a = int(at[i].split("T")[1])
        k = wk[i] - 1
        oh = weekly[k, h, 0] + (shock if qb[k, h] else 0.0)
        oa = weekly[k, a, 0] + (shock if qb[k, a] else 0.0)
        zl[i] = (oh - weekly[k, h, 1]) - (oa - weekly[k, a, 1])
    return zl


def team_latent(tg, latents):
    shock = -0.05668721441704449
    lo, ld, qq = [], [], []
    for gi, tm, wk_ in zip(tg["game_id"], tg["team"], tg["week"]):
        w = int(gi[1:5])
        s = int(gi[6:8])
        weekly, qb = latents[(w, s)]
        ti = int(tm.split("T")[1])
        k = int(wk_) - 1
        lo.append(float(weekly[k, ti, 0]) + (shock if qb[k, ti] else 0.0))
        ld.append(float(weekly[k, ti, 1]))
        qq.append(bool(qb[k, ti]))
    return np.array(lo), np.array(ld), np.array(qq)


def conv_stats(g, tg):
    m = tg.set_index(["game_id", "team"])["off_epa_per_play"]
    d = m.reindex(pd.MultiIndex.from_arrays([g["game_id"], g["home_team"]])).to_numpy() - m.reindex(pd.MultiIndex.from_arrays([g["game_id"], g["away_team"]])).to_numpy()
    n = tg.set_index(["game_id", "team"])
    ok = ~np.isnan(d)
    y = g["y"].to_numpy(dtype=float)
    c = float(np.corrcoef(d[ok], y[ok])[0, 1])
    sl = float(np.cov(d[ok], y[ok])[0, 1] / np.var(d[ok], ddof=1))
    pts = (g["home_score"] + g["away_score"]).to_numpy(dtype=float)
    return {"y_on_epa_corr": c, "y_on_epa_slope": sl, "y_resid_var": float(np.var(y[ok], ddof=1) * (1 - c * c)), "epa_diff_sd": float(np.std(d[ok], ddof=1)), "y_sd": float(np.std(y[ok], ddof=1)), "total_sd": float(np.std(pts[ok], ddof=1)), "total_mean": float(np.mean(pts[ok]))}


def strdiag_one(g, tg, latents=None):
    g = g.copy()
    g["z2"] = loo_season_net(g, tg)
    res = {"x_vs_loo_season": bucket_corr(g, "x", "z2"), "x_vs_y": bucket_corr(g, "x", "y"), "loo_vs_y": bucket_corr(g, "z2", "y"), "within": within_stats(tg), "sd_x": {}, "sd_z2": float(g["z2"].std()), "conv": conv_stats(g, tg)}
    for lo, hi in ((1, 4), (5, 9), (10, 18)):
        b = g[(g["week"] >= lo) & (g["week"] <= hi)]
        res["sd_x"][f"w{lo}_{hi}"] = float(b["x"].std())
    if latents is not None:
        g["zl"] = latent_rows(g, latents)
        res["x_vs_latent"] = bucket_corr(g, "x", "zl")
        res["latent_vs_y"] = bucket_corr(g, "zl", "y")
        res["loo_vs_latent"] = bucket_corr(g, "z2", "zl")
        res["sd_latent"] = float(g["zl"].std())
        res["y_on_latent_slope"] = float(np.cov(g["zl"], g["y"])[0, 1] / g["zl"].var())
        res["epa_on_latent_slope_net"] = float(np.cov(g["zl"], g["z2"])[0, 1] / g["zl"].var())
        t = tg.copy()
        lo, ld, qq = team_latent(t, latents)
        t["lo"], t["ld"], t["q"] = lo, ld, qq
        for c, lc in (("off_epa_per_play", "lo"), ("def_epa_per_play", "ld")):
            dev = t[c] - t.groupby("season")[c].transform("mean")
            lat = t[lc] - t.groupby("season")[lc].transform("mean")
            res[f"slope_{c}"] = float(np.cov(dev, lat)[0, 1] / lat.var())
            res[f"corr_{c}"] = float(np.corrcoef(dev, lat)[0, 1])
            res[f"latent_sd_{c}"] = float(lat.std())
        dev = t["off_epa_per_play"] - t.groupby(["team", "season"])["off_epa_per_play"].transform("mean")
        res["qb_backup_share"] = float(t["q"].mean())
        res["qb_dev_gap"] = float(dev[t["q"]].mean() - dev[~t["q"]].mean())
    return res


def cmd_strdiag(args):
    import mod25_generator as gen

    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for nm, ss in (("real_2011_2017", range(2011, 2018)), ("real_2018_2025", range(2018, 2026))):
        games = gen.real_games(tuple(ss))
        plays = gen.load_real_plays(ss)
        ts = gen.team_stats_from_plays(plays)
        games = games[games["game_id"].isin(ts["game_id"])].reset_index(drop=True)
        g, tg = x_frame(games, ts)
        g = g.dropna(subset=["x", "y"])
        out[nm] = strdiag_one(g, tg)
        print(nm, flush=True)
    for spec in args.variants.split(","):
        v, sc = spec.split("@")
        games, plays, latents = gen_full(v, float(sc), args.worlds, args.seasons, args.workers, args.seed)
        ts = gen.team_stats_from_plays(plays)
        g, tg = x_frame(games, ts)
        keep = [w * 1000 + s + 1 for w in range(args.worlds) for s in range(2, args.seasons)]
        g = g[g["season"].isin(keep)].dropna(subset=["x", "y"])
        tg = tg[tg["season"].isin(keep)]
        out[spec] = strdiag_one(g, tg, latents)
        print(spec, flush=True)
    (OUT / f"strdiag_{args.tag}.json").write_text(json.dumps(out, indent=1))
    F = {k: flat(v) for k, v in out.items()}
    keys = list(F[next(iter(F))])
    for n in F:
        keys += [k for k in F[n] if k not in keys]
    lines = [f"{'metric':36s} " + " ".join(f"{n[:16]:>16s}" for n in F)]
    for k in keys:
        lines.append(f"{k:36s} " + " ".join(f"{F[n].get(k, float('nan')):16.4f}" for n in F))
    (OUT / f"strdiag_{args.tag}.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def epa_blocks(P):
    D, R = drive_table(P)
    R = R.copy()
    R["pts_off"] = R["po"]
    R["epa"] = R["epa"].fillna(0.0)
    de = R.groupby("poss")["epa"].sum()
    D["epa_sum"] = de.reindex(D["poss"]).to_numpy() if "poss" in D else de.to_numpy()
    D["res"] = D["pts"] + R.groupby("poss")["pdf"].sum().reindex(D["poss"]).to_numpy() * 0 - D["epa_sum"] if False else D["pts"] - D["epa_sum"]
    out = {"drive_res_mean": float(D["res"].mean()), "drive_res_var": float(D["res"].var()), "drive_epa_var": float(D["epa_sum"].var()), "drive_pts_var": float(D["pts"].var()), "drive_corr_pts_epa": float(np.corrcoef(D["pts"], D["epa_sum"])[0, 1])}
    for k in OUTCOMES:
        d = D[D["oc"] == k]
        if len(d) > 50:
            out["res_var_" + k] = float(d["res"].var())
            out["res_mean_" + k] = float(d["res"].mean())
            out["share_" + k] = float(len(d) / len(D))
    g = D.groupby("g")
    cnt = D.groupby(["g", "trk"]).agg(p=("pts", "sum"), e=("epa_sum", "sum")).reset_index()
    a = cnt[cnt["trk"] == 0].set_index("g")
    b = cnt[cnt["trk"] == 1].set_index("g")
    idx = a.index.intersection(b.index)
    dp = (a.loc[idx, "p"] - b.loc[idx, "p"]).to_numpy()
    de2 = (a.loc[idx, "e"] - b.loc[idx, "e"]).to_numpy()
    c = float(np.corrcoef(dp, de2)[0, 1])
    out["game_margin_var"] = float(np.var(dp, ddof=1))
    out["game_epa_diff_var"] = float(np.var(de2, ddof=1))
    out["game_corr_margin_epa"] = c
    out["game_res_var"] = float(np.var(dp, ddof=1) * (1 - c * c))
    out["game_slope"] = float(np.cov(dp, de2)[0, 1] / np.var(de2, ddof=1))
    tp = np.concatenate([a.loc[idx, "p"].to_numpy(), b.loc[idx, "p"].to_numpy()])
    te = np.concatenate([a.loc[idx, "e"].to_numpy(), b.loc[idx, "e"].to_numpy()])
    ct = float(np.corrcoef(tp, te)[0, 1])
    out["team_pts_var"] = float(np.var(tp, ddof=1))
    out["team_epa_var"] = float(np.var(te, ddof=1))
    out["team_corr_pts_epa"] = ct
    out["team_res_var"] = float(np.var(tp, ddof=1) * (1 - ct * ct))
    return out


def cmd_epadiag(args):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for nm, ss in (("real_train", TRAIN), ("real_eval", EVAL)):
        P, M, meta, pbp = real_load(ss)
        res[nm] = epa_blocks(P)
        print(nm, flush=True)
    for v in args.variants.split(","):
        S = pd.read_parquet(Path(args.cache) / f"sim_{v}.parquet")
        res[v] = epa_blocks(S)
        print(v, flush=True)
    (OUT / f"epadiag_{args.tag}.json").write_text(json.dumps(res, indent=1))
    keys = list(res["real_train"])
    lines = [f"{'metric':28s} " + " ".join(f"{n:>11s}" for n in res)]
    for k in keys:
        lines.append(f"{k:28s} " + " ".join(f"{res[n].get(k, float('nan')):11.4f}" for n in res))
    (OUT / f"epadiag_{args.tag}.txt").write_text("\n".join(lines))
    print("\n".join(lines))


SCHED_PATH = REPO / "data" / "raw" / "20260915T170343Z" / "schedules.parquet"
PRECIP_RE = "rain|snow|shower|drizzle|storm|flurr|sleet|wintry|thunder"


def side_game_table(P):
    D, R = drive_table(P)
    D = D[D["q0"] <= 4]
    D = D.assign(opp=1 - D["trk"], pts_for=D["pts"], pts_against=D["pdf"])
    res = {}
    for t in (0, 1):
        m = D["trk"] == t
        a = D[m].groupby("g")
        b = D[~m].groupby("g")
        res[t] = pd.DataFrame({"n": a.size(), "secs": a["secs"].sum(), "plays": a["n"].sum(), "pts": a["pts"].sum() + b["pdf"].sum()})
    G = res[0].join(res[1], lsuffix="A", rsuffix="B", how="inner").fillna(0)
    return G


def trade_stats(G):
    out = {}
    out["games"] = int(len(G))
    for c in ("n", "secs", "plays", "pts"):
        a, b = G[c + "A"], G[c + "B"]
        out[f"{c}_mean"] = float((a.mean() + b.mean()) / 2)
        out[f"{c}_var_team"] = float((a.var() + b.var()) / 2)
        out[f"{c}_cov_AB"] = float(np.cov(a, b)[0, 1])
        out[f"{c}_corr_AB"] = float(np.corrcoef(a, b)[0, 1])
    out["secs_tot_sd"] = float((G["secsA"] + G["secsB"]).std())
    out["secs_diff_sd"] = float((G["secsA"] - G["secsB"]).std())
    out["pts_total_sd"] = float((G["ptsA"] + G["ptsB"]).std())
    out["pts_margin_sd"] = float((G["ptsA"] - G["ptsB"]).std())
    ppdA = G["ptsA"] / G["nA"].clip(lower=1)
    ppdB = G["ptsB"] / G["nB"].clip(lower=1)
    out["ppd_corr_AB"] = float(np.corrcoef(ppdA, ppdB)[0, 1])
    out["ppd_cov_AB"] = float(np.cov(ppdA, ppdB)[0, 1])
    out["spp_corr_AB"] = float(np.corrcoef(G["secsA"] / G["playsA"].clip(lower=1), G["secsB"] / G["playsB"].clip(lower=1))[0, 1])
    return out


def real_cond_frame(seasons):
    sch = pd.read_parquet(SCHED_PATH)
    sch = sch[(sch["season"].isin(list(seasons))) & (sch["game_type"] == "REG")].copy()
    fr = []
    for s in seasons:
        d = pd.read_parquet(f"{NV_DIR}/pbp_{s}.parquet", columns=["game_id", "season_type", "qtr", "play_type", "penalty", "weather", "posteam", "fixed_drive", "game_seconds_remaining", "drive_time_of_possession"])
        d = d[d["season_type"] == "REG"]
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    pl = d[d["play_type"].isin(["pass", "run", "no_play"])]
    g = pl.groupby("game_id")
    rp = d[d["play_type"].isin(["pass", "run"])].groupby("game_id").size()
    C = pd.DataFrame({"plays": rp, "pen_rate": g["penalty"].mean(), "weather": g["weather"].first()})
    C = C.join(sch.set_index("game_id")[["season", "week", "home_score", "away_score", "roof", "surface", "temp", "wind", "referee", "home_team", "away_team", "total_line", "spread_line"]], how="inner")
    C["dome"] = C["roof"].isin(["dome", "closed"]).astype(float)
    C["turf"] = (~C["surface"].isin(["grass", "dessograss"])).astype(float)
    C["precip"] = C["weather"].fillna("").str.lower().str.contains(PRECIP_RE).astype(float)
    outdoor = (C["dome"] == 0)
    C["wind_o"] = np.where(outdoor, C["wind"].fillna(0), 0.0)
    C["windy"] = (C["wind_o"] >= 15).astype(float)
    C["temp_o"] = np.where(outdoor, C["temp"].fillna(C["temp"].median()), 70.0)
    C["cold"] = (C["temp_o"] <= 32).astype(float)
    C["hot"] = (C["temp_o"] >= 85).astype(float)
    C["precip_o"] = C["precip"] * outdoor
    ref = C.groupby("referee")["pen_rate"].agg(["sum", "count"])
    r = C["referee"].map(ref["sum"]) - C["pen_rate"]
    n = C["referee"].map(ref["count"]) - 1
    mu = C["pen_rate"].mean()
    C["crew_pen"] = ((r + 8 * mu) / (n + 8)) - mu
    C["pace_plays"] = C["plays"] - C["plays"].mean()
    return C.reset_index()


COND_GROUPS = {
    "roof_dome": ["dome"],
    "surface_turf": ["turf"],
    "wind": ["wind_o", "windy"],
    "precip": ["precip_o"],
    "temp": ["temp_o", "cold", "hot"],
    "crew_penalty_rate": ["crew_pen"],
    "game_pace_plays": ["pace_plays"],
    "env_all": ["dome", "turf", "wind_o", "windy", "precip_o", "temp_o", "cold", "hot", "crew_pen"],
    "env_all_plus_pace": ["dome", "turf", "wind_o", "windy", "precip_o", "temp_o", "cold", "hot", "crew_pen", "pace_plays"],
}


def crossfit_pred(C, cols, ycol):
    X = np.column_stack([np.ones(len(C))] + [C[c].to_numpy(float) for c in cols])
    y = C[ycol].to_numpy(float)
    pred = np.zeros(len(C))
    for s in sorted(C["season"].unique()):
        te = (C["season"] == s).to_numpy()
        beta = np.linalg.lstsq(X[~te], y[~te], rcond=None)[0]
        pred[te] = X[te] @ beta
    return pred


def cond_decomp(C):
    h = C["home_score"].to_numpy(float)
    a = C["away_score"].to_numpy(float)
    out = {"games": int(len(C)), "cov_home_away": float(np.cov(h, a)[0, 1]), "var_home": float(h.var(ddof=1)), "var_away": float(a.var(ddof=1)), "sd_total": float((h + a).std()), "sd_margin": float((h - a).std())}
    rows = {}
    for nm, cols in COND_GROUPS.items():
        ph = crossfit_pred(C.assign(y=h), cols, "y")
        pa = crossfit_pred(C.assign(y=a), cols, "y")
        pt = crossfit_pred(C.assign(y=h + a), cols, "y")
        cv = float(np.cov(ph, pa)[0, 1])
        r2t = float(1 - ((h + a - pt) ** 2).sum() / ((h + a - (h + a).mean()) ** 2).sum())
        rows[nm] = {"explained_cov": cv, "total_r2_cv": r2t}
    out["groups"] = rows
    return out


def cond_effects(C):
    res = {}
    tot = (C["home_score"] + C["away_score"]).to_numpy(float)
    for nm, cols in COND_GROUPS.items():
        if nm in ("env_all", "env_all_plus_pace"):
            continue
        X = np.column_stack([np.ones(len(C))] + [C[c].to_numpy(float) for c in cols])
        b_tot = np.linalg.lstsq(X, tot, rcond=None)[0][1:]
        b_pl = np.linalg.lstsq(X, C["plays"].to_numpy(float), rcond=None)[0][1:]
        res[nm] = {"cols": cols, "total_pts": [float(x) for x in b_tot], "plays": [float(x) for x in b_pl]}
    return res


def cmd_covdecomp(args):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    C = real_cond_frame(range(2009, 2018))
    res["real_cond"] = cond_decomp(C)
    res["real_effects"] = cond_effects(C)
    res["real_cond_means"] = {c: float(C[c].mean()) for c in ("dome", "turf", "windy", "precip_o", "cold", "hot", "plays", "pen_rate")}
    P, M, meta, pbp = real_load(range(2009, 2018))
    res["real_trade_2009_17"] = trade_stats(side_game_table(P))
    for nm, ss in (("real_train", TRAIN), ("real_eval", EVAL)):
        P, M, meta, pbp = real_load(ss)
        res[f"trade_{nm}"] = trade_stats(side_game_table(P))
    for v in [x for x in args.variants.split(",") if x]:
        S = pd.read_parquet(Path(args.cache) / f"sim_{v}.parquet")
        res[f"trade_{v}"] = trade_stats(side_game_table(S))
    (OUT / f"covdecomp_{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


DB = [0, 1, 2, 4, 7, 10, 100]


def down_tables(P):
    P = P.sort_values("g", kind="stable").reset_index(drop=True)
    g = P["g"].to_numpy()
    down = P["down"].to_numpy(float)
    dist = P["dist"].to_numpy(float)
    code = P["code"].to_numpy()
    flip = P["flip"].to_numpy().astype(bool)
    po = P["po"].to_numpy(float)
    qtr = P["qtr"].to_numpy()
    nd = np.r_[down[1:], np.nan]
    ng = np.r_[g[1:] == g[:-1], False]
    nd = np.where(ng, nd, np.nan)
    rush_pass = (code == 0) | (code == 1)
    out = {}
    reg = qtr <= 4
    for dn in (1, 2, 3, 4):
        m = reg & (down == dn)
        out[f"share_down{dn}"] = float(m.sum() / (reg & ~np.isnan(down)).sum())
    for dn in (2, 3, 4):
        m = reg & (down == dn)
        b = pd.cut(dist[m], DB, right=True)
        vc = pd.Series(b).value_counts(normalize=True, sort=False)
        for k, v in vc.items():
            out[f"dist_share_d{dn}.{k}"] = float(v)
        out[f"dist_mean_d{dn}"] = float(dist[m].mean())
    for dn in (3, 4):
        m = reg & (down == dn) & rush_pass
        conv = (po >= 6) | ((nd == 1) & ~flip)
        b = pd.cut(dist[m], DB, right=True)
        d = pd.DataFrame({"b": b, "c": conv[m].astype(float)})
        for k, v in d.groupby("b", observed=True)["c"].agg(["mean", "size"]).iterrows():
            out[f"conv_d{dn}.{k}"] = float(v["mean"])
            out[f"n_d{dn}.{k}"] = float(v["size"])
        out[f"conv_d{dn}_all"] = float(conv[m].mean())
    m4 = reg & (down == 4) & np.isin(code, [0, 1, 2, 3])
    for lo, hi, nm in ((0, 2, "le2"), (2, 100, "gt2")):
        mm = m4 & (dist > lo) & (dist <= hi)
        out[f"go_rate_d4_{nm}"] = float(np.isin(code[mm], [0, 1]).mean())
    out["d4_le2_share"] = float((m4 & (dist <= 2)).sum() / m4.sum())
    m3 = reg & (down == 3) & rush_pass
    out["d3_next_down4_share"] = float(((nd == 4) & m3).sum() / m3.sum())
    return out


def cmd_downdiag(args):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    P, M, meta, pbp = real_load(TRAIN)
    res["real_train"] = down_tables(P)
    P, M, meta, pbp = real_load(EVAL)
    res["real_eval"] = down_tables(P)
    for v in [x for x in args.variants.split(",") if x]:
        S = pd.read_parquet(Path(args.cache) / f"sim_{v}.parquet")
        res[v] = down_tables(S)
    (OUT / f"downdiag_{args.tag}.json").write_text(json.dumps(res, indent=1))
    keys = list(res["real_train"])
    lines = [f"{'metric':36s} " + " ".join(f"{n:>11s}" for n in res)]
    for k in keys:
        lines.append(f"{k:36s} " + " ".join(f"{res[n].get(k, float('nan')):11.4f}" for n in res))
    (OUT / f"downdiag_{args.tag}.txt").write_text(chr(10).join(lines))
    print(chr(10).join(lines))


def sim_cov_row(games, plays, latents, worlds, seasons):
    import mod25_generator as gen

    keep = [w * 1000 + s + 1 for w in range(worlds) for s in range(2, seasons)]
    g = games[games["season"].isin(keep)].copy()
    h = g["home_score"].to_numpy(float)
    a = g["away_score"].to_numpy(float)
    out = {"games": int(len(g)), "cov_home_away": float(np.cov(h, a)[0, 1]), "var_total": float((h + a).var(ddof=1)), "var_margin": float((h - a).var(ddof=1)), "sd_total": float((h + a).std()), "sd_margin": float((h - a).std())}
    pl = plays[plays["play_type"].isin(["run", "pass"])]
    n = pl.groupby("game_id").size()
    g = g.assign(plays=g["game_id"].map(n).to_numpy(float))
    g["pace_plays"] = g["plays"] - g["plays"].mean()
    g["hs"], g["as_"] = h, a
    out["pace_explained_cov"] = float(np.cov(crossfit_pred(g.assign(y=g["hs"]), ["pace_plays"], "y"), crossfit_pred(g.assign(y=g["as_"]), ["pace_plays"], "y"))[0, 1])
    po = pl.groupby(["game_id", "posteam"]).size().reset_index(name="n")
    po["r"] = po.groupby("game_id").cumcount()
    ph = po[po["r"] == 0].set_index("game_id")["n"].reindex(gm.index).to_numpy(float)
    pa = po[po["r"] == 1].set_index("game_id")["n"].reindex(gm.index).to_numpy(float)
    out["plays_var_team"] = float((np.nanvar(ph, ddof=1) + np.nanvar(pa, ddof=1)) / 2)
    ok = ~np.isnan(ph) & ~np.isnan(pa)
    out["plays_cov_AB"] = float(np.cov(ph[ok], pa[ok])[0, 1])
    out["plays_diff_sd"] = float((ph[ok] - pa[ok]).std())
    return out


def cmd_covsim(args):
    res = {}
    for spec in args.variants.split(","):
        v, sc = spec.split("@")
        games, plays, latents = gen_full(v, float(sc), args.worlds, args.seasons, args.workers, args.seed)
        games.to_parquet(OUT / f"covsim_games_{v}.parquet")
        res[spec] = sim_cov_row(games, plays, latents, args.worlds, args.seasons)
        print(spec, json.dumps(res[spec]), flush=True)
    (OUT / f"covsim_{args.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
