import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25d_variance as dv  # noqa: E402
import mod25e_scorestate as ss  # noqa: E402
import sim09_stakes as S  # noqa: E402
import sim09_u3a as u3  # noqa: E402
import sim09_u3d as u3d  # noqa: E402
import sim09_urgency as U  # noqa: E402

OUTD = REPO / "artifacts" / "sim09" / "u4c"
MODEL = OUTD / "stakes_model.npz"
START_WEEK = 10
BEH = ["nh", "sg", "intc", "go4"]
dv.DV["crzs"] = dict(dv.DV["crzk"], stakes=1)
dv.DV["crzt"] = dict(dv.DV["crzk"], stakes=1, stakes_model="stakes_model_forced.npz")
ORIG_SEASON = ss.s_play_season
STATE = {}
CTX = {"active": False}
SIM_DIV = [list(range(4 * i, 4 * i + 4)) for i in range(8)]
SIM_CONF = {"A": list(range(16)), "B": list(range(16, 32))}


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def feats(sd, tsec, qtr, down, dist, fp, home):
    sd, tsec, qtr, down, dist, fp, home = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (sd, tsec, qtr, down, dist, fp, home))
    z = sd / np.sqrt(tsec / 60.0 + 1.0)
    zt, zl = np.minimum(z, 0.0), np.maximum(z, 0.0)
    F = np.column_stack([np.ones_like(sd), sd / 10.0, zt, zl, tsec / 3600.0, qtr == 2, qtr == 3, qtr == 4, down == 2, down == 3, down == 4, np.log1p(np.maximum(dist, 0.0)), fp / 100.0, home]).astype(float)
    return F, zt, zl


def stk12(b6, zt, zl):
    b6 = np.atleast_2d(b6).astype(float)
    return np.column_stack([b6[:, 0], b6[:, 1], b6[:, 2], b6[:, 3], b6[:, 4], b6[:, 5], b6[:, 0] * zt, b6[:, 0] * zl, b6[:, 2] * zt, b6[:, 2] * zl, b6[:, 4] * zt, b6[:, 4] * zl])


def fit_logit(F, Sx, y, lam):
    nf = F.shape[1]
    ns = Sx.shape[1]

    def f(th):
        g, b = th[:nf], th[nf:]
        eta = F @ g + (Sx @ b if lam is not None else 0.0)
        p = sig(eta)
        nll = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        r = p - y
        gg = F.T @ r
        gb = Sx.T @ r if lam is not None else np.zeros(ns)
        if lam is not None:
            nll += 0.5 * lam * float(b @ b)
            gb = gb + lam * b
        return nll, np.concatenate([gg, gb])

    r = minimize(f, np.zeros(nf + ns), jac=True, method="L-BFGS-B", options={"maxiter": 400})
    return r.x[:nf], (r.x[nf:] if lam is not None else np.zeros(ns))


def ll(F, Sx, y, g, b):
    eta = F @ g + Sx @ b
    return float(np.mean(np.logaddexp(0.0, eta) - y * eta))


def pool_table():
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = dv.sim.build_transition_frame(pbp)
    keys = tr[["game_id", "play_id"]].reset_index(drop=True)
    keys["play_id"] = keys["play_id"].astype(np.int64)
    cols = ["game_id", "play_id", "season", "week", "posteam", "defteam", "home_team", "shotgun", "no_huddle", "interception", "qb_dropback", "sack"]
    nv = pd.concat([pd.read_parquet(U.NV / f"pbp_{s}.parquet", columns=cols) for s in dv.TRAIN], ignore_index=True)
    nv["play_id"] = nv["play_id"].astype(np.int64)
    nv = nv.drop_duplicates(["game_id", "play_id"])
    x = keys.merge(nv, on=["game_id", "play_id"], how="left")
    assert len(x) == len(tr)
    for c in ("posteam", "defteam", "home_team"):
        x[c] = S.norm_team(x[c])
    x["code"] = tr["play_type_code"].to_numpy()
    x["down"] = tr["down_i"].to_numpy()
    x["dist"] = tr["dist_raw"].to_numpy()
    x["fp"] = tr["fp_raw"].to_numpy()
    x["sd"] = tr["sc_raw"].to_numpy()
    x["tsec"] = tr["gsr_actual"].to_numpy()
    x["qtr"] = tr["qtr_actual"].to_numpy()
    tw = pd.read_csv(S.OUT / "stakes_teamweek.csv")[["season", "week", "team", "lev", "elim", "clinch"]]
    own = tw.rename(columns={"team": "posteam", "lev": "b0", "elim": "b2", "clinch": "b4"})
    opp = tw.rename(columns={"team": "defteam", "lev": "b1", "elim": "b3", "clinch": "b5"})
    x = x.merge(own, on=["season", "week", "posteam"], how="left").merge(opp, on=["season", "week", "defteam"], how="left")
    assert len(x) == len(tr)
    return x


def cmd_fit(args):
    OUTD.mkdir(parents=True, exist_ok=True)
    x = pool_table()
    b6 = x[["b0", "b1", "b2", "b3", "b4", "b5"]].to_numpy(float)
    have = np.isfinite(b6).all(axis=1) & x["posteam"].notna().to_numpy() & x["home_team"].notna().to_numpy()
    code = x["code"].to_numpy()
    home = (x["posteam"] == x["home_team"]).to_numpy(float)
    scr = np.isin(code, (0, 1))
    spec = {
        "nh": (have & scr & x["no_huddle"].notna().to_numpy(), x["no_huddle"].fillna(0).to_numpy(float)),
        "sg": (have & scr & x["shotgun"].notna().to_numpy(), x["shotgun"].fillna(0).to_numpy(float)),
        "intc": (have & (code == 1) & (x["qb_dropback"] == 1).to_numpy() & (x["sack"] != 1).to_numpy(), x["interception"].fillna(0).to_numpy(float)),
        "go4": (have & (x["down"].to_numpy() == 4) & np.isin(code, (0, 1, 2, 3)), np.isin(code, (0, 1)).astype(float)),
    }
    F, zt, zl = feats(x["sd"], x["tsec"], x["qtr"], x["down"], x["dist"], x["fp"], home)
    Sx = stk12(np.nan_to_num(b6), zt, zl)
    fitm = (x["week"].to_numpy() >= START_WEEK) & x["week"].notna().to_numpy()
    season = x["season"].to_numpy()
    lams = [None, 1e4, 1e3, 1e2, 1e1, 1.0]
    out = {"b6": np.nan_to_num(b6).astype(np.float32)}
    rows = []
    for nm, (v, y) in spec.items():
        m = v & fitm
        sdv = Sx[m].std(axis=0)
        sdv[sdv == 0] = 1.0
        Ss = Sx / sdv
        res = {}
        for lam in (lams if args.force_lam is None else []):
            tot = 0.0
            n = 0
            for s in dv.TRAIN:
                tr_, te_ = m & (season != s), m & (season == s)
                g, b = fit_logit(F[tr_], Ss[tr_], y[tr_], lam)
                tot += ll(F[te_], Ss[te_], y[te_], g, b) * te_.sum()
                n += te_.sum()
            res[lam] = tot / n
            rows.append({"beh": nm, "lam": "none" if lam is None else lam, "loso_logloss": tot / n, "n": int(n)})
        best = min(res, key=res.get) if args.force_lam is None else args.force_lam
        if args.force_lam is not None:
            res = {args.force_lam: float("nan")}
        g, b = fit_logit(F[m], Ss[m], y[m], best)
        out[f"{nm}_valid"] = v
        out[f"{nm}_y"] = y.astype(np.int8)
        out[f"{nm}_g"] = g
        out[f"{nm}_b"] = b
        out[f"{nm}_sd"] = sdv
        out[f"{nm}_mean"] = Ss[m].mean(axis=0)
        out[f"{nm}_lam"] = np.array(-1.0 if best is None else best)
        print(nm, "n", int(m.sum()), "rate", float(y[m].mean()), "chosen lam", best, "loso", {("none" if k is None else k): round(v_, 6) for k, v_ in res.items()}, "beta_std", np.round(b, 4).tolist(), flush=True)
    np.savez(OUTD / ("stakes_model_forced.npz" if args.force_lam is not None else "stakes_model.npz"), **out)
    if args.force_lam is None:
        pd.DataFrame(rows).to_csv(OUTD / "stakes_model_loso.csv", index=False)
    print("pool rows", len(x), "with stakes tags", int(have.sum()))


class Tracker:
    def __init__(self, sched, seed):
        self.sched = sched
        self.gi = 0
        self.results = []
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 4]))
        self.cache = {}
        self.rows = []
        self.P = json.loads((S.OUT / "stakes_summary.json").read_text())["rating_params"]

    def stakes(self, w):
        if w in self.cache:
            return self.cache[w]
        P = self.P
        S.RNG = self.rng
        S.DIVIDX = SIM_DIV
        S.CONF = SIM_CONF
        n = S.NSIM
        Ssum = np.zeros(32)
        N = np.zeros(32)
        W0 = np.zeros(32)
        for (wk, h, a, m) in self.results:
            if wk >= w:
                continue
            Ssum[h] += m
            Ssum[a] -= m
            N[h] += 1
            N[a] += 1
            W0[h] += 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
            W0[a] += 1.0 if m < 0 else (0.5 if m == 0 else 0.0)
        r = Ssum / (N + P["k"])
        rem = [(i, g) for i, g in enumerate(self.sched) if g[0] >= w]
        hi = np.array([g[1] for _, g in rem])
        ai = np.array([g[2] for _, g in rem])
        e = P["hfa"] + P["slope"] * (r[hi] - r[ai])
        X = self.rng.random((n, len(rem))) < norm.cdf(e / P["sigma"])
        Hm = np.zeros((len(rem), 32))
        Am = np.zeros((len(rem), 32))
        Hm[np.arange(len(rem)), hi] = 1
        Am[np.arange(len(rem)), ai] = 1
        wins = W0[None, :] + X @ Hm + (~X) @ Am
        seed = S.seeds_from_wins(wins, 6)
        mk = seed > 0
        pmk = mk.mean(axis=0)
        out = {}
        for j, (i, g) in enumerate(rem):
            if g[0] != w:
                continue
            for t, o, win in ((g[1], g[2], X[:, j]), (g[2], g[1], ~X[:, j])):
                a_ = mk[win, t].mean() if win.any() else np.nan
                b_ = mk[~win, t].mean() if (~win).any() else np.nan
                lev = a_ - b_ if a_ == a_ and b_ == b_ else 0.0
                out[t] = (float(lev), float(pmk[t] < 0.01), float(pmk[t] > 0.99), float(pmk[t]), float(r[t]), float(r[o]), o)
        self.cache[w] = out
        for t, v in out.items():
            self.rows.append((w, t) + v)
        return out

    def frame(self):
        return pd.DataFrame(self.rows, columns=["week", "team", "lev", "elim", "clinch", "p_make", "r_own", "r_opp", "opp"])


def offline_tracker(games_frame, seed):
    gm = games_frame.sort_values("game_id")
    sched = [(int(r.week), int(r.home_team.split("T")[1]), int(r.away_team.split("T")[1])) for r in gm.itertuples()]
    T = Tracker(sched, seed)
    for (wk, h, a), r in zip(sched, gm.itertuples()):
        if wk >= START_WEEK:
            T.stakes(wk)
        T.results.append((wk, h, a, float(r.home_score - r.away_score)))
    return T.frame()


class StakesCache(dict):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def __setitem__(self, wkey, v):
        if self.q >= 5 or not CTX["active"]:
            return dict.__setitem__(self, wkey, v)
        key, off_sim, def_sim, home = wkey
        nb = CTX["tbl"]["nn_cache_cond"][key]
        cdf, _ = v
        w = np.diff(cdf, prepend=0.0)
        w2 = w * factor(self.q, key, home, nb, w)
        s = w2.sum()
        if not np.isfinite(s) or s <= 0.0:
            return dict.__setitem__(self, wkey, v)
        cdf2 = np.cumsum(w2)
        return dict.__setitem__(self, wkey, (cdf2, float(cdf2[-1])))


def factor(q, key, home, nb, w):
    M = CTX["M"]
    down, phase, r_dist, r_fp, r_score, r_time = key[:6]
    R = dv.sim
    sd = r_score * R.ROUND_SCORE
    time_raw = r_time * R.ROUND_TIME
    tsec = time_raw + (1800.0 if q <= 2 else 0.0)
    F, zt, zl = feats(sd, tsec, q, down, r_dist * R.ROUND_DIST, r_fp * R.ROUND_FP, home)
    base = CTX["H"] if home else CTX["A"]
    rows = M["b6"][nb]
    Srow = stk12(rows, zt[0], zl[0])
    Ssim = stk12(base[None, :], zt[0], zl[0])
    f = np.ones(len(nb))
    for nm in BEH:
        b = M[f"{nm}_b"]
        if not b.any():
            continue
        V = M[f"{nm}_valid"][nb]
        if not V.any():
            continue
        y = M[f"{nm}_y"][nb][V]
        sdv = M[f"{nm}_sd"]
        g = float(F[0] @ M[f"{nm}_g"])
        ls = g + float((Ssim[0] / sdv) @ b)
        lr = g + (Srow[V] / sdv) @ b
        ratio = np.where(y == 1, sig(ls) / sig(lr), (1.0 - sig(ls)) / (1.0 - sig(lr)))
        wv = w[V]
        den = (wv * ratio).sum()
        if not np.isfinite(den) or den <= 0.0 or wv.sum() <= 0.0:
            continue
        ratio = ratio / (den / wv.sum())
        f[V] *= ratio
    return f


class TiltPol4:
    def __init__(self, base):
        self.base = base

    def __getitem__(self, key):
        p = np.asarray(self.base[key], dtype=float)
        if not CTX["active"]:
            return p
        M = CTX["M"]
        b = M["go4_b"]
        if not b.any():
            return p
        fr = sys._getframe(1)
        loc = fr.f_locals
        home = None
        g = fr
        for _ in range(6):
            g = g.f_back
            if g is None:
                break
            if "is_home_sim" in g.f_locals:
                home = g.f_locals["is_home_sim"]
                break
        if home is None or "score_diff" not in loc or "clock_val" not in loc:
            return p
        sd = float(loc["score_diff"])
        tsec = float(loc["clock_val"])
        z = sd / np.sqrt(tsec / 60.0 + 1.0)
        base = CTX["H"] if home else CTX["A"]
        Ssim = stk12(base[None, :], min(z, 0.0), max(z, 0.0))[0] / M["go4_sd"]
        delta = float((Ssim - M["go4_mean"]) @ b)
        p0 = float(p[0])
        if not 0.0 < p0 < 1.0:
            return p
        p1 = sig(np.log(p0 / (1.0 - p0)) + delta)
        out = p.copy()
        out[0] = p1
        out[1:] = p[1:] * (1.0 - p1) / (1.0 - p0)
        return out


def install_stakes(cfg):
    tables = dv._G["tables"]
    ns = dv._G["ns"]
    M = dict(np.load(OUTD / cfg.get("stakes_model", "stakes_model.npz")))
    assert len(M["b6"]) == len(tables["arrays"]["possession_flip"])
    CTX["M"] = M
    caches = {q: StakesCache(q) for q in range(1, 6)}
    plain = {}
    base = ns["pick_index_nn_conditioned"]

    def pick(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim):
        if CTX["active"]:
            q = int(sys._getframe(1).f_locals["qtr"])
            CTX["tbl"] = tbl
            tbl["nn_weight_cache_cond"] = caches[min(max(q, 1), 5)]
        else:
            tbl["nn_weight_cache_cond"] = plain
        return base(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)

    ns["pick_index_nn_conditioned"] = pick
    orig = ns["run_one_game"]

    def run(*a, **k):
        T = STATE["trk"]
        week, h, aw = T.sched[T.gi]
        for c in caches.values():
            dict.clear(c)
        plain.clear()
        CTX["active"] = week >= START_WEEK
        if CTX["active"]:
            st = T.stakes(week)
            hv, av = st[h], st[aw]
            CTX["H"] = np.array([hv[0], av[0], hv[1], av[1], hv[2], av[2]])
            CTX["A"] = np.array([av[0], hv[0], av[1], hv[1], av[2], hv[2]])
        rec, cap = orig(*a, **k)
        T.results.append((week, h, aw, float(rec["margin"])))
        T.gi += 1
        return rec, cap

    ns["run_one_game"] = run


def install_tracker_only():
    ns = dv._G["ns"]
    orig = ns["run_one_game"]

    def run(*a, **k):
        T = STATE["trk"]
        week, h, aw = T.sched[T.gi]
        if week >= START_WEEK:
            T.stakes(week)
        rec, cap = orig(*a, **k)
        T.results.append((week, h, aw, float(rec["margin"])))
        T.gi += 1
        return rec, cap

    ns["run_one_game"] = run


def _init(base_init, setting):
    cfg_pre = json.loads(setting["mech"])
    m25 = dv.m25
    if cfg_pre.get("stakes"):
        orig = m25.make_decide

        def md(tables, trans, pol4, poll, mech, seed):
            return orig(tables, trans, TiltPol4(pol4), poll, mech, seed)

        m25.make_decide = md
    base_init(setting)
    if dv._G["cfg"].get("stakes"):
        install_stakes(dv._G["cfg"])
    else:
        install_tracker_only()


def init_e5(setting):
    _init(u3d.tfix_init, setting)


def init_play(setting):
    _init(u3d.tfix_init_budget, setting)


def _begin(task):
    STATE["trk"] = Tracker(task[3], task[2])


def _end(task):
    outd = Path(os.environ["STK_OUT"])
    outd.mkdir(parents=True, exist_ok=True)
    f = STATE["trk"].frame()
    f.insert(0, "sidx", task[1])
    f.insert(0, "world", task[0])
    f.to_parquet(outd / f"stk_{task[0]}_{task[1]}.parquet")


def season_e5(task):
    _begin(task)
    res = ORIG_SEASON(task)
    _end(task)
    return res


def season_play(task):
    _begin(task)
    res = u3.s_play_season(task)
    _end(task)
    return res


def cmd_sim(args):
    import mod25_generator as gen
    import mod25e_budget as bud

    out = Path(args.out_dir).resolve()
    bud.OUT = out
    os.environ["BUD_OUT"] = str(out)
    os.environ["STK_OUT"] = str(out)
    out.mkdir(parents=True, exist_ok=True)
    dv.c25.ensure_policies()
    v = dv.DV[args.variant]
    cfgj = json.dumps(dict(v, seed=args.seed, name=f"{args.variant}_s{v['scale']:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": v["scale"], "yard_gain": v["yard_gain"], "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = init_play
    gen.play_season = season_play
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(out / "sim_games.parquet")
    print("done", len(games), el, flush=True)


def cmd_e5(args):
    os.environ["STK_OUT"] = str(ss.OUT / f"e5_{args.variant}_s{args.seed}")
    ss.s_init = init_e5
    ss.s_play_season = season_e5
    args.scale = 1.0
    ss.cmd_e5(args)


def cmd_post(args):
    d = Path(args.dir)
    g = pd.read_parquet(d / "sim_games.parquet")
    rows = []
    for sid, gg in g.groupby("season"):
        w, s = (sid - 1) // 1000, (sid - 1) % 1000
        f = offline_tracker(gg, 1000 * w + s)
        f.insert(0, "sidx", s)
        f.insert(0, "world", w)
        rows.append(f)
    out = pd.concat(rows)
    out.to_parquet(OUTD / f"stk_offline_{d.name}.parquet")
    print("offline stakes rows", len(out))


def stk_table(dirp, offline=None):
    import mod25e_budget as bud

    bud.OUT = Path(dirp)
    P = u3.load_sim_frames(2)
    d = u3.to_pbp(P)
    gh = d[d.qtr == 3].groupby("game_id").first()
    hd = (gh["score_differential"] * np.where(gh["posteam"] == gh["home_team"], 1, -1)).rename("hd_home")
    if offline is not None:
        tw = pd.read_parquet(offline)
    else:
        tw = pd.concat([pd.read_parquet(f) for f in sorted(Path(dirp).glob("stk_*_*.parquet"))])
    tw = tw.assign(season=tw.world * 10 + tw.sidx, team=tw.world.astype(str) + "_" + tw.team.astype(int).astype(str))
    tw["locked"] = 0.0
    tw = tw[["season", "week", "team", "p_make", "lev", "elim", "locked", "clinch", "r_own", "r_opp"]]
    return d, hd, tw


def sim_rows(d, hd, tw, outcomes):
    u = d[(d.qtr <= 4) & d.posteam.notna() & d.score_differential.notna() & d.play_type.isin(["pass", "run"]) & (d.qb_kneel != 1) & (d.qb_spike != 1)].copy()
    u["t"] = u.game_seconds_remaining.clip(lower=0)
    u = U.situation(U.controls(u), hd)
    u = S.stake_cols(u, tw)
    u = u.dropna(subset=U.CTRL + ["z", "ht_trail", "lev_own", "lev_opp"])
    rows = S.fit_block(u, outcomes, "play")
    f = d[(d.qtr <= 4) & d.posteam.notna() & d.score_differential.notna() & (d.down == 4) & d.play_type.isin(["pass", "run", "punt", "field_goal"])].copy()
    f = f[(f.qb_kneel != 1) & (f.qb_spike != 1)]
    f["t"] = f.game_seconds_remaining.clip(lower=0)
    off = U.team_prior(u, "posteam", "off_epa")
    dfn = U.team_prior(u, "defteam", "def_epa")
    f["pos_home"] = (f.posteam == f.home_team).astype(float)
    f = f.merge(off, on=["season", "week", "posteam"], how="left").merge(dfn, on=["season", "week", "defteam"], how="left")
    f = f.merge(off.rename(columns={"posteam": "defteam", "off_epa": "opp_off"}), on=["season", "week", "defteam"], how="left")
    f = f.merge(dfn.rename(columns={"defteam": "posteam", "def_epa": "own_def"}), on=["season", "week", "posteam"], how="left")
    f = U.situation(f, hd)
    f = S.stake_cols(f, tw)
    f["go"] = f.play_type.isin(["pass", "run"]).astype(float)
    f["yd"] = f.ydstogo.clip(upper=15).astype(float)
    f["yl"] = f.yardline_100.astype(float)
    f["yl2"] = f.yl**2 / 100.0
    f["fgr"] = (f.yl <= 40).astype(float)
    f = f.dropna(subset=U.CTRL + ["z", "ht_trail", "lev_own", "lev_opp", "yd", "yl"])
    rows += S.fit_block(f, ["go"], "fourth", ["yd", "yl", "yl2", "fgr"])
    return pd.DataFrame(rows)


def cmd_val(args):
    outcomes = ["shotgun", "no_huddle", "intc"]
    orig = S.fit_block
    S.fit_block = lambda df, o, k, e=(): orig(df[df.week >= START_WEEK], o, k, e)
    tw = pd.read_csv(S.OUT / "stakes_teamweek.csv")
    d = U.load()
    d = U.controls(d)
    d = U.situation(d, U.halftime())
    d["posteam"] = S.norm_team(d.posteam)
    d["defteam"] = S.norm_team(d.defteam)
    d = S.stake_cols(d, tw)
    d = d.dropna(subset=U.CTRL + ["z", "ht_trail", "lev_own", "lev_opp"])
    rr = pd.DataFrame(S.fit_block(d, outcomes, "play"))
    r4, _, _ = S.fourth_down(tw, d)
    rr = pd.concat([rr, pd.DataFrame(r4)])
    rr["src"] = "real"
    parts = [rr]
    for name, dirp, off in (("crzk", args.base_dir, args.base_stk), ("crzs", args.new_dir, None)):
        dd, hd, tws = stk_table(dirp, off)
        r = sim_rows(dd, hd, tws, outcomes)
        r["src"] = name
        parts.append(r)
    A = pd.concat(parts)
    A.to_csv(OUTD / "val_slopes.csv", index=False)
    T = A.pivot_table(index=["outcome", "term"], columns="src", values=["coef", "se"])
    out = pd.DataFrame({"outcome": [i[0] for i in T.index], "term": [i[1] for i in T.index]})
    for s_ in ("real", "crzk", "crzs"):
        out[f"coef_{s_}"] = T[("coef", s_)].to_numpy()
        out[f"se_{s_}"] = T[("se", s_)].to_numpy()
    out["crzs_minus_real_z"] = (out.coef_crzs - out.coef_real) / np.sqrt(out.se_crzs**2 + out.se_real**2)
    out["crzk_minus_real_z"] = (out.coef_crzk - out.coef_real) / np.sqrt(out.se_crzk**2 + out.se_real**2)
    out.to_csv(OUTD / "val_slopes_wide.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    fi = sub.add_parser("fit")
    fi.add_argument("--force-lam", dest="force_lam", type=float, default=None)
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="crzs")
    s.add_argument("--out-dir", dest="out_dir", default=str(OUTD / "play_crzs"))
    s.add_argument("--worlds", type=int, default=6)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=21)
    e = sub.add_parser("e5")
    e.add_argument("--variant", default="crzs")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    p = sub.add_parser("post")
    p.add_argument("--dir", default=str(REPO / "artifacts" / "sim09" / "u3d" / "play_crzk"))
    v = sub.add_parser("val")
    v.add_argument("--base-dir", dest="base_dir", default=str(REPO / "artifacts" / "sim09" / "u3d" / "play_crzk"))
    v.add_argument("--base-stk", dest="base_stk", default=str(OUTD / "stk_offline_play_crzk.parquet"))
    v.add_argument("--new-dir", dest="new_dir", default=str(OUTD / "play_crzs"))
    args = ap.parse_args()
    {"fit": cmd_fit, "sim": cmd_sim, "e5": cmd_e5, "post": cmd_post, "val": cmd_val}[args.cmd](args)


if __name__ == "__main__":
    main()
