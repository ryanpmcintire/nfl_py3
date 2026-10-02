import argparse
import atexit
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25d_variance as dv  # noqa: E402
import sim09_fatigue as fat  # noqa: E402
import sim09_half as hf  # noqa: E402
import sim09_hk  # noqa: E402,F401
import sim09_u3a as u3  # noqa: E402
import sim09_u3d as kd  # noqa: E402
import sim09_u4d as u4d  # noqa: E402
import sim09_u4e as u4e  # noqa: E402
import sim09_u4f as u4f  # noqa: E402

OUT = REPO / "artifacts" / "sim09" / "int"
BW_FILE = OUT / "fat_bw.json"
PICK_HOME = "        weights = weights * np.where(is_home_row == is_home_sim, TEAM_KERNEL_LAMBDA, 1.0)\n"
PICK_K = "wkey = (key, off_sim, def_sim, is_home_sim)"
BWJ = json.loads(BW_FILE.read_text()) if BW_FILE.exists() else {}
BW = BWJ.get("chosen_h")
BETA = BWJ.get("chosen_beta")
dv.DV["crI"] = dict(dv.DV["crzhk"], fdnb=0, e1=1, e2=1, fat=1, fh=BW, fb=BETA, intc=1, kl=u4f.REG)
dv.DV["crIs"] = dict(dv.DV["crI"], fh=BWJ.get("silverman_h"), fb=None)
dv.DV["crIk"] = dict(dv.DV["crI"], kl=None)
dv.DV["crIn"] = dict(dv.DV["crI"], fat=0, fh=None, fb=None)
DIAG = {"n": 0, "ess0": 0.0, "ess1": 0.0, "ess2": 0.0, "gap1": 0.0, "gap2": 0.0, "iters": 0}


def flush_diag():
    if DIAG["n"]:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"diag_{os.getpid()}.json").write_text(json.dumps(DIAG))


atexit.register(flush_diag)


def kish(w):
    s = w.sum()
    return float(s * s / (w * w).sum())


def moment_tilt(w1, Z, tgt):
    sd = Z.std(0)
    ok = sd > 1e-12
    if not ok.any():
        return w1, 0
    Zs = (Z[:, ok] - tgt[ok]) / sd[ok]
    th = np.zeros(Zs.shape[1])

    def phi(t):
        a = Zs @ t
        m = a.max()
        return m + np.log((w1 * np.exp(a - m)).sum())

    cur = phi(th)
    it = 0
    for it in range(1, 41):
        a = Zs @ th
        e = w1 * np.exp(a - a.max())
        p = e / e.sum()
        g = p @ Zs
        if np.abs(g).max() < 1e-9:
            break
        H = (Zs * p[:, None]).T @ Zs - np.outer(g, g) + 1e-12 * np.eye(len(th))
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-6:
            nxt = phi(th - t * step)
            if nxt < cur:
                break
            t *= 0.5
        else:
            break
        th = th - t * step
        cur = nxt
    a = Zs @ th
    return w1 * np.exp(a - a.max()), it


def make_compose(L, state, fh, fb):
    def compose(w0, nb, off_row, def_row):
        if not (fh or fb):
            return w0
        s0 = w0.sum()
        if not np.isfinite(s0) or s0 <= 0.0:
            return w0
        if fb:
            a = fb * (L[nb] - state[0])
            w1 = w0 * np.exp(a - a.max())
        else:
            z2 = ((L[nb] - state[0]) / fh) ** 2
            w1 = w0 * np.exp(-0.5 * (z2 - z2.min()))
        Z = np.column_stack([off_row, def_row])
        tgt = (w0 @ Z) / s0
        w2, it = moment_tilt(w1, Z, tgt)
        DIAG["n"] += 1
        DIAG["ess0"] += kish(w0)
        DIAG["ess1"] += kish(w1)
        DIAG["ess2"] += kish(w2)
        DIAG["gap1"] += float(np.abs((w1 @ Z) / w1.sum() - tgt).max())
        DIAG["gap2"] += float(np.abs((w2 @ Z) / w2.sum() - tgt).max())
        DIAG["iters"] += it
        if DIAG["n"] % 500 == 0:
            flush_diag()
        return w2

    return compose


def patch_pick(src):
    assert PICK_HOME in src and PICK_K in src
    src = src.replace(PICK_HOME, PICK_HOME + "        weights = COMPOSE(weights, neighbors, off_row, def_row)\n")
    return src.replace(PICK_K, "wkey = (key, off_sim, def_sim, is_home_sim, FAT_ST[0])")


class IStrata(u4e.Strata):
    def tree(self, dk, ph, c, tv, to, nh, db, hlf=0):
        key = (dk, ph, c, tv, to, nh, db, hlf if ph == 0 else 0)
        t = self.trees.get(key, False)
        if t is not False:
            return t
        m = (self.down == dk) & self.pool_mask[ph] & (self.code == c)
        if ph == 0:
            m &= (self.qtr == 3) if hlf else (self.qtr <= 2)
        if tv is not None:
            m &= self.tov == tv
        if to is not None:
            m &= self.to == to
        if nh is not None:
            m &= self.nh == nh
        if db is not None:
            m &= self.db == db
        sub = np.flatnonzero(m)
        t = (self.sim.KDTree(self.feats[sub]), sub) if len(sub) >= 8 else None
        self.trees[key] = t
        return t

    def weights(self, nb, ctx, wk):
        ent = self.wc.get(wk)
        if ent is None:
            off_sim, def_sim, home_sim, _ = ctx
            tb = dv._G["tables"]
            a = tb["arrays"]
            off_row, def_row, hrow = a["off_row"][nb], a["def_row"][nb], a["is_home_off"][nb]
            h = tb["team_kernel_h"]
            w = np.exp(-((off_row - off_sim) ** 2 + (def_row - def_sim) ** 2) / (2.0 * h * h))
            w = w * np.where(hrow == home_sim, self.sim.TEAM_KERNEL_LAMBDA, 1.0)
            w = self.compose(w, nb, off_row, def_row)
            s = w.sum()
            if not np.isfinite(s) or s <= 0.0:
                w = np.ones_like(w)
                s = w.sum()
            ent = (np.cumsum(w), s)
            if len(self.wc) > 150000:
                self.wc.clear()
            self.wc[wk] = ent
        return ent

    def pick(self, rng, dk, ph, c, tv, to, nh, db, dist, yl, sd, tf, ot_, dt_, hlf, ctx):
        sim = self.sim
        relax = [(tv, to, nh, db), (None, None, nh, db), (None, None, None, db), (None, None, nh, None), (None, None, None, None)]
        seen = []
        for r in relax:
            if r in seen:
                continue
            seen.append(r)
            for p in (ph, 3, 2, 1, 0, 4):
                ent = self.tree(dk, p, c, *r, hlf)
                if ent is None:
                    continue
                key = (sim.round_state_key(dk, p, dist, yl, sd, tf, ot_, dt_), dk, c, hlf if p == 0 else 0) + r
                nb = self.cache.get(key)
                if nb is None:
                    tree, sub = ent
                    f = sim.feature_matrix(np.array([dist]), np.array([yl]), np.array([sd]), np.array([tf]), np.array([ot_]), np.array([dt_]), np.array([p]))
                    _, ind = tree.query(f, k=min(sim.K_NEIGHBORS, len(sub)))
                    nb = sub[np.atleast_1d(ind[0])]
                    self.cache[key] = nb
                cdf, s = self.weights(nb, ctx, (key,) + ctx)
                k = min(int(np.searchsorted(cdf, rng.random() * s, side="right")), len(nb) - 1)
                return int(nb[k])
        return None


def install_int_state(cfg, L):
    ns = dv._G["ns"]
    st = [0.0, 0.0]
    ns["FAT_ST"] = st
    ns["FAT_LAST"] = {"home": 0, "away": 0}
    ns["FAT_CUM"] = {"home": 0, "away": 0}
    ns["FAT_CUR"] = [0]
    comp = make_compose(L, st, cfg.get("fh"), cfg.get("fb"))
    ns["COMPOSE"] = comp
    dv._G["tables"]["nn_weight_cache_cond"].clear()
    return st, comp


def install_decide(cfg, fatst, comp):
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
    ns = dv._G["ns"]
    tr, tov, nh = u4e.pool_frame()
    S = IStrata(tr, tov, nh)
    S.qtr = tr["qtr_actual"].to_numpy()
    S.wc = {}
    S.compose = comp
    assert len(S.qtr) == len(dv._G["tables"]["arrays"]["off_row"])
    del tr
    models = u4e.joblib.load(u4e.MODELS)
    code_arr = S.code
    st = {"prev": 2, "poss": None, "scr": False, "key": None, "rng": None}
    nh_grid = models["nh"]
    dec = ns["DECIDE"]
    base_run = ns["run_one_game"]
    gsr_n = len(u4e.GSR_G)

    def run_wrapped(*a, **k):
        st["prev"], st["poss"], st["scr"] = 2, None, False
        return base_run(*a, **k)

    ns["run_one_game"] = run_wrapped

    def p_nh(state, down, sd, gsr):
        return float(nh_grid[state, down - 1, int(round(sd)) + 24, min(int(round(gsr / 15.0)), gsr_n - 1)])

    def decide(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat):
        idx = dec(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat)
        if st["key"] != dv._G.get("task_key"):
            st["key"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(4097, cfg.get("seed", 3))
        rng2 = st["rng"]
        fl = sys._getframe(1).f_locals
        poss = fl["possessions"]
        state = st["prev"] if (poss == st["poss"] and st["scr"]) else 2
        dk = down if down in (1, 2, 3, 4) else 4
        c0 = int(code_arr[idx])
        tc = c0
        sdc = min(max(score_diff, -24.0), 24.0)
        nh_t = None
        if cfg.get("e1") and tc in (0, 1):
            nh_t = 1 if rng2.random() < p_nh(state, dk, sdc, clock_val) else 0
        db = S.bucket(distance) if (cfg.get("e2") and dk == 4 and tc in (0, 1)) else None
        if db is not None:
            tc = 1 if rng2.random() < S.pass_rate[min(db, len(S.pass_rate) - 1)] else 0
        need = tc != c0 or (nh_t is not None and S.nh[idx] != nh_t) or (db is not None and S.db[idx] != db)
        if need:
            same = tc == c0
            ctx = (fl["off_sim"], fl["def_sim"], fl["is_home_sim"], fatst[0])
            j = S.pick(rng, dk, phase, tc, int(S.tov[idx]) if same else None, int(S.to[idx]) if same else None, nh_t, db, distance, yardline, score_diff, time_feat, off_to, def_to, 1 if int(qtr) == 3 else 0, ctx)
            if j is not None:
                idx = j
        cf = int(code_arr[idx])
        st["poss"] = poss
        st["scr"] = cf in (0, 1)
        st["prev"] = int(S.nh[idx])
        return idx

    ns["DECIDE"] = decide


def int_init(setting, base_init):
    cfg = json.loads(setting["mech"])
    if cfg.get("kl"):
        u4f.apply_consts(cfg["kl"])
    if cfg.get("half"):
        hf.install_half_index()
    orig = inspect.getsource

    def gs(o):
        s = orig(o)
        if o is dv.sim.run_one_game:
            s = fat.patch_run(s)
        elif o is dv.sim.pick_index_nn_conditioned:
            s = patch_pick(s)
        return s

    inspect.getsource = gs
    try:
        base_init(setting)
    finally:
        inspect.getsource = orig
    cfg = dv._G["cfg"]
    L, _ = fat.pool_fatigue()
    assert len(L) == len(dv._G["tables"]["arrays"]["play_type_code"])
    assert "COMPOSE" in dv._G["ns"]["pick_index_nn_conditioned"].__code__.co_names
    fatst, comp = install_int_state(cfg, L)
    if cfg.get("half"):
        hf.install_half_pick()
    if cfg.get("tfix"):
        kd.install_tfix()
    install_decide(cfg, fatst, comp)


def int_init_budget(setting):
    import mod25e_budget as bud

    int_init(setting, bud.e_init)


def int_init_ss(setting):
    import mod25e_scorestate as ss

    int_init(setting, ss.s_init)


def cmd_sim(args):
    import mod25_generator as gen
    import mod25e_budget as bud

    bud.OUT = Path(args.out_dir).resolve()
    os.environ["BUD_OUT"] = str(bud.OUT)
    bud.OUT.mkdir(parents=True, exist_ok=True)
    dv.c25.ensure_policies()
    v = dv.DV[args.variant]
    cfgj = json.dumps(dict(v, seed=args.seed, name=f"{args.variant}_s{v['scale']:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": v["scale"], "yard_gain": v["yard_gain"], "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = int_init_budget
    gen.play_season = u3.s_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(bud.OUT / "sim_games.parquet")
    print("done", len(games), el, flush=True)


def cmd_e5(args):
    import mod25e_scorestate as ss

    ss.s_init = int_init_ss
    args.scale = 1.0
    ss.cmd_e5(args)


def cmd_analyze(args):
    u3.OUT = OUT / f"analyze_{Path(args.post_dir).name}"
    u3.OUT.mkdir(parents=True, exist_ok=True)
    u3.bud.OUT = Path(args.post_dir)
    u3.cmd_analyze(argparse.Namespace(minsid=2, boot=args.boot))


def real_L(seasons):
    d = pd.concat([pd.read_parquet(u4d.NV / f"pbp_{s}.parquet", columns=["game_id", "play_id", "fixed_drive", "defteam", "posteam", "play_type", "qb_kneel", "qb_spike"]) for s in seasons], ignore_index=True)
    d = d.drop_duplicates(["game_id", "play_id"]).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    d["pr"] = (d["play_type"].isin(["pass", "run"]) & (d["qb_kneel"].fillna(0) != 1) & (d["qb_spike"].fillna(0) != 1)).astype(int)
    x = d[d["fixed_drive"].notna() & d["posteam"].notna()]
    t = x.groupby(["game_id", "fixed_drive"]).agg(defn=("defteam", "first"), first=("play_id", "min"), plays=("pr", "sum")).reset_index().sort_values(["game_id", "first"])
    t["prior"] = t.groupby(["game_id", "defn"])["plays"].shift(1)
    return d[["game_id", "play_id", "fixed_drive"]].merge(t[["game_id", "fixed_drive", "prior"]], on=["game_id", "fixed_drive"], how="left")[["game_id", "play_id", "prior"]]


def wcrps(x, y, w):
    o = np.argsort(x, axis=1)
    xs = np.take_along_axis(x, o, 1)
    ws = np.take_along_axis(w, o, 1)
    c = np.cumsum(ws, axis=1)
    t1 = (ws * np.abs(xs - y[:, None])).sum(1)
    t2 = (ws * xs * (2 * c - ws - 1)).sum(1)
    return t1 - t2


def rawW(Ln, L0, h, beta):
    if beta:
        a = beta * (Ln - L0)
        return np.exp(a - a.max(1, keepdims=True))
    if h is None:
        return np.ones(Ln.shape)
    return np.exp(-0.5 * ((Ln - L0) / h) ** 2)


def kern_rows(sc, Lf, Lv, p, h, beta=None):
    K = int(p["K"])
    nb = sc.nb_for(p)[:, :K]
    w = rawW(Lf[nb], Lv[:, None], h, beta)
    w = w / w.sum(1, keepdims=True)
    ess = 1.0 / (w * w).sum(1)
    pp = ((w * sc.pf[nb]).sum(1) * ess + 0.5) / (ess + 1)
    lp = -(sc.pv * np.log(pp) + (1 - sc.pv) * np.log(1 - pp))
    cy = wcrps(sc.yf[nb].astype(float), sc.yv.astype(float), w)
    nba = sc.nb_air(p)[:, :K]
    wa = rawW(Lfa(sc)[nba], Lva(sc)[:, None], h, beta)
    wa = wa / wa.sum(1, keepdims=True)
    ca = wcrps(sc.fp_["air_yards"].to_numpy().astype(float)[nba], sc.vp_["air_yards"].to_numpy().astype(float), wa)
    return {"lp": lp, "cy": cy, "ca": ca}


def Lfa(sc):
    return sc.fp_["prior"].to_numpy()


def Lva(sc):
    return sc.vp_["prior"].to_numpy()


def with_L(d):
    seasons = sorted(d["season"].unique().astype(int))
    m = real_L(seasons)
    d = d.merge(m, on=["game_id", "play_id"], how="left")
    d["prior"] = d["prior"].fillna(0.0)
    return d


def cmd_fatbw(args):
    OUT.mkdir(parents=True, exist_ok=True)
    p = dict(u4f.REG)
    p = {k: p[k] for k in ("a", "b", "T", "D", "F", "O", "K", "H")}
    tf, tv = u4f.prep(with_L(u4d.real_plays(range(2009, 2014)))), u4f.prep(with_L(u4d.real_plays((2014, 2015))))
    fit, test = u4f.prep(with_L(u4d.real_plays(range(2009, 2016)))), u4f.prep(with_L(u4d.real_plays((2016, 2017))))
    L_all, _ = fat.pool_fatigue()
    sil = float(1.06 * np.std(L_all) * dv.sim.K_STATE ** -0.2)
    grid = sorted(set([sil * m for m in (0.25, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0, 8.0)]))
    sv = u4f.Scorer(tf, tv)
    base = None
    res = {}
    for h in [None] + grid:
        r = kern_rows(sv, tf["prior"].to_numpy(), tv["prior"].to_numpy(), p, h)
        m = (float(r["lp"].mean()), float(r["cy"].mean()), float(r["ca"].mean()))
        if base is None:
            base = m
        J = sum(a / b for a, b in zip(m, base))
        res[str(h)] = {"J": J, "pass_ll": m[0], "yards_crps": m[1], "air_crps": m[2]}
        print("val h", h, "J", round(J, 6), m, flush=True)
    best = min(grid, key=lambda h: res[str(h)]["J"])
    use = res[str(best)]["J"] < res["None"]["J"]
    sdL = float(np.std(L_all))
    bgrid = sorted(set([s_ * m / sdL for m in (0.05, 0.1, 0.2, 0.4, 0.8) for s_ in (-1.0, 1.0)]))
    bres = {}
    for b in bgrid:
        r = kern_rows(sv, tf["prior"].to_numpy(), tv["prior"].to_numpy(), p, None, b)
        m = (float(r["lp"].mean()), float(r["cy"].mean()), float(r["ca"].mean()))
        J = sum(a / c for a, c in zip(m, base))
        bres[str(b)] = {"J": J, "pass_ll": m[0], "yards_crps": m[1], "air_crps": m[2]}
        print("val beta", b, "J", round(J, 6), m, flush=True)
    bbest = min(bgrid, key=lambda b: bres[str(b)]["J"])
    buse = bres[str(bbest)]["J"] < res["None"]["J"]
    st = u4f.Scorer(fit, test)
    gt = test["game_id"].to_numpy()
    rows = {}
    for nm, h in (("none", None), ("silverman", sil), ("chosen", best), ("tilt", None)):
        rows[nm] = kern_rows(st, fit["prior"].to_numpy(), test["prior"].to_numpy(), p, h, bbest if nm == "tilt" else None)
        print("test", nm, h, {k: round(float(v.mean()), 5) for k, v in rows[nm].items()}, flush=True)
    ga = test["game_id"].to_numpy()[st.air_rows] if False else st.vp_["game_id"].to_numpy()
    deltas = {}
    for nm in ("silverman", "chosen", "tilt"):
        deltas[nm] = {"pass_ll": u4f.boot_delta(rows["none"]["lp"], rows[nm]["lp"], gt), "yards_crps": u4f.boot_delta(rows["none"]["cy"], rows[nm]["cy"], gt), "air_crps": u4f.boot_delta(rows["none"]["ca"], rows[nm]["ca"], ga)}
    out = {"declared_objective": "J = passLL + yardsCRPS + airCRPS each over the no-fatigue-kernel value, tune 2009-13 val 2014-15, state kernel at kernel_consts.json, Gaussian kernel on prior-drive length, weights self-normalised", "silverman_h": sil, "val": res,
           "chosen_h": best if use else None, "tilt_val": bres, "chosen_beta": bbest if buse else None, "best_beta": bbest, "best_grid_h": best, "kernel_used": bool(use), "test_delta_none_minus_x": deltas}
    BW_FILE.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


def cmd_diag(args):
    tot = {}
    for f in OUT.glob("diag_*.json"):
        for k, v in json.loads(f.read_text()).items():
            tot[k] = tot.get(k, 0) + v
    n = tot.get("n", 0)
    print({k: (v / n if k != "n" else v) for k, v in tot.items()})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fatbw")
    sub.add_parser("diag")
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="crI")
    s.add_argument("--out-dir", dest="out_dir", default=str(OUT / "play_crI"))
    s.add_argument("--worlds", type=int, default=6)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=21)
    e = sub.add_parser("e5")
    e.add_argument("--variant", default="crI")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    n = sub.add_parser("analyze")
    n.add_argument("--post-dir", dest="post_dir", default=str(OUT / "play_crI"))
    n.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    {"fatbw": cmd_fatbw, "diag": cmd_diag, "sim": cmd_sim, "e5": cmd_e5, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
