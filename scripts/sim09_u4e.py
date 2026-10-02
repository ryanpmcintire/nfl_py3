import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25d_variance as dv  # noqa: E402
import sim09_u3a as u3  # noqa: E402
import sim09_u3d as kd  # noqa: E402
import sim09_urgency as urg  # noqa: E402

OUT = REPO / "artifacts" / "sim09" / "u4e"
MODELS = OUT / "models.joblib"
base = dv.DV["crzk"]
dv.DV["u4e1"] = dict(base, e1=1)
dv.DV["u4e2"] = dict(base, fdnb=0, e2=1)
dv.DV["u4e3"] = dict(base, e3=1)
dv.DV["u4e"] = dict(base, fdnb=0, e1=1, e2=1, e3=1)
SD_G = np.arange(-24, 25)
GSR_G = np.arange(0, 3601, 15.0)
Q2_COLS = ["down", "dist", "yl", "sd", "gsr", "off_to", "def_to"]


def pool_frame():
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = dv.sim.build_transition_frame(pbp)
    at = dv.c25.attrs_from(pbp, tr)
    nv = pd.concat([pd.read_parquet(urg.NV / f"pbp_{s}.parquet", columns=["game_id", "play_id", "no_huddle"]) for s in range(min(dv.TRAIN), max(dv.TRAIN) + 1)], ignore_index=True).drop_duplicates(["game_id", "play_id"])
    x = tr[["game_id", "play_id"]].merge(nv, on=["game_id", "play_id"], how="left")
    nh = (x["no_huddle"].fillna(0).to_numpy() == 1).astype(np.int8)
    return tr, np.asarray(at["tov"], dtype=float).astype(np.int8), nh


def nh_state(tr, nh):
    gid, _ = pd.factorize(tr["game_id"])
    code = tr["play_type_code"].to_numpy()
    flip = tr["possession_flip"].to_numpy().astype(bool)
    po = tr["points_off"].to_numpy()
    pdf = tr["points_def"].to_numpy()
    qtr = tr["qtr_actual"].to_numpy()
    same = np.r_[False, gid[1:] == gid[:-1]]
    pc, pf, pp, pq, pn = (np.r_[0, a[:-1]] for a in (code, flip, po + pdf, qtr, nh))
    ok = same & np.isin(pc, (0, 1)) & ~pf & (pp == 0) & ~((pq <= 2) & (qtr >= 3))
    return np.where(ok, pn, 2).astype(np.int8)


def nh_features(tr, state):
    return np.column_stack([state, tr["down_i"].to_numpy(), np.clip(tr["sc_raw"].to_numpy(), -24, 24), tr["gsr_actual"].to_numpy()]).astype(np.float64)


def q2_features(tr):
    return np.column_stack([tr["down_i"].to_numpy(), tr["dist_raw"].to_numpy(), tr["fp_raw"].to_numpy(), np.clip(tr["sc_raw"].to_numpy(), -24, 24), tr["gsr_actual"].to_numpy() - 1800.0, tr["off_to_raw"].to_numpy(), tr["def_to_raw"].to_numpy()]).astype(np.float64)


def hgb():
    return HistGradientBoostingClassifier(max_depth=5, max_iter=200, learning_rate=0.08, random_state=1)


def cmd_fit(args):
    OUT.mkdir(parents=True, exist_ok=True)
    tr, tov, nh = pool_frame()
    state = nh_state(tr, nh)
    code = tr["play_type_code"].to_numpy()
    scr = np.isin(code, (0, 1))
    X = nh_features(tr, state)
    season = pd.to_numeric(tr["game_id"].astype(str).str[:4], errors="coerce").to_numpy()
    y = nh
    tr_m = scr & (season <= 2014)
    te_m = scr & (season >= 2015)
    m = hgb().fit(X[tr_m], y[tr_m])
    p = m.predict_proba(X[te_m])[:, 1]
    p0 = np.full(te_m.sum(), y[tr_m].mean())
    ps = np.array([y[tr_m & (state == s)].mean() for s in (0, 1, 2)])[state[te_m]]
    print("NH held-out 2015-17 log loss: model", round(log_loss(y[te_m], p), 5), "state-only", round(log_loss(y[te_m], ps), 5), "base rate", round(log_loss(y[te_m], p0), 5), "n", int(te_m.sum()))
    nh_model = hgb().fit(X[scr], y[scr])
    G = np.array([(a, b, c, d) for a in (0, 1, 2) for b in (1, 2, 3, 4) for c in SD_G for d in GSR_G], dtype=np.float64)
    grid = nh_model.predict_proba(G)[:, 1].reshape(3, 4, len(SD_G), len(GSR_G)).astype(np.float32)
    win = tr["phase"].to_numpy() == 1
    qm = win & np.isin(code, (0, 1, 2, 3, 4, 5)) & (tr["qtr_actual"].to_numpy() == 2)
    Xq = q2_features(tr)
    classes = np.unique(code[qm])
    trm = qm & (season <= 2014)
    tem = qm & (season >= 2015)
    mq = hgb().fit(Xq[trm], code[trm])
    pq = mq.predict_proba(Xq[tem])
    freq = np.array([(code[trm] == c).mean() for c in mq.classes_])
    sub = np.isin(code[tem], mq.classes_)
    print("Q2 window held-out log loss: model", round(log_loss(code[tem][sub], pq[sub], labels=mq.classes_), 5), "marginal", round(log_loss(code[tem][sub], np.tile(freq, (sub.sum(), 1)), labels=mq.classes_), 5), "n", int(tem.sum()), "classes", classes.tolist())
    q2_model = hgb().fit(Xq[qm], code[qm])
    joblib.dump({"nh": grid, "q2": q2_model}, MODELS)
    print("saved", MODELS)


class Strata:
    def __init__(self, tr, tov, nh):
        sim = dv.sim
        self.sim = sim
        self.down = tr["down_i"].to_numpy()
        self.phase = tr["phase"].to_numpy()
        self.code = tr["play_type_code"].to_numpy()
        self.tov = tov
        self.to = ((tr["off_to_used"].to_numpy() + tr["def_to_used"].to_numpy()) > 0).astype(np.int8)
        self.nh = nh
        self.feats = sim.feature_matrix(tr["dist_raw"].to_numpy(), tr["fp_raw"].to_numpy(), tr["sc_raw"].to_numpy(), tr["time_raw"].to_numpy(), tr["off_to_raw"].to_numpy(), tr["def_to_raw"].to_numpy(), self.phase)
        dist = np.rint(tr["dist_raw"].to_numpy()).astype(int)
        go4 = (self.down == 4) & np.isin(self.code, (0, 1))
        edges = []
        cnt = 0
        top = int(dist[go4].max())
        for d in range(0, top + 1):
            cnt += int(((dist == d) & go4).sum())
            if cnt >= sim.K_STATE:
                edges.append(d)
                cnt = 0
        self.edges = np.array(edges[:-1] if len(edges) > 1 and cnt > 0 else edges, dtype=int)
        self.db = np.searchsorted(self.edges, dist, side="left").astype(np.int16)
        nb_ = int(self.db[go4].max()) + 1
        self.pass_rate = np.array([(self.code[go4 & (self.db == k)] == 1).mean() if (go4 & (self.db == k)).any() else 0.5 for k in range(nb_)])
        self.trees = {}
        self.cache = {}
        self.pool_mask = {p: sim.phase_pool_mask(self.phase, p) for p in range(5)}

    def bucket(self, distance):
        return int(np.searchsorted(self.edges, int(round(distance)), side="left"))

    def tree(self, dk, ph, c, tv, to, nh, db):
        key = (dk, ph, c, tv, to, nh, db)
        t = self.trees.get(key, False)
        if t is not False:
            return t
        m = (self.down == dk) & self.pool_mask[ph] & (self.code == c)
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

    def pick(self, rng, dk, ph, c, tv, to, nh, db, dist, yl, sd, tf, ot_, dt_):
        sim = self.sim
        relax = [(tv, to, nh, db), (None, None, nh, db), (None, None, None, db), (None, None, nh, None), (None, None, None, None)]
        seen = []
        for r in relax:
            if r in seen:
                continue
            seen.append(r)
            for p in (ph, 3, 2, 1, 0, 4):
                ent = self.tree(dk, p, c, *r)
                if ent is None:
                    continue
                key = (sim.round_state_key(dk, p, dist, yl, sd, tf, ot_, dt_), dk, c) + r
                nb = self.cache.get(key)
                if nb is None:
                    tree, sub = ent
                    f = sim.feature_matrix(np.array([dist]), np.array([yl]), np.array([sd]), np.array([tf]), np.array([ot_]), np.array([dt_]), np.array([p]))
                    _, ind = tree.query(f, k=min(sim.K_NEIGHBORS, len(sub)))
                    nb = sub[ind[0]]
                    self.cache[key] = nb
                return int(nb[rng.integers(len(nb))])
        return None


def install_u4e():
    cfg = dv._G["cfg"]
    e1, e2, e3 = cfg.get("e1"), cfg.get("e2"), cfg.get("e3")
    if not (e1 or e2 or e3):
        return
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
    ns = dv._G["ns"]
    tr, tov, nh = pool_frame()
    S = Strata(tr, tov, nh)
    del tr
    models = joblib.load(MODELS)
    code_arr = S.code
    st = {"prev": 2, "poss": None, "scr": False, "key": None, "rng": None}
    q2c = {}
    nh_grid, q2_model = models["nh"], models["q2"]
    q2_classes = q2_model.classes_
    dec = ns["DECIDE"]
    base_run = ns["run_one_game"]

    def run_wrapped(*a, **k):
        st["prev"], st["poss"], st["scr"] = 2, None, False
        return base_run(*a, **k)

    ns["run_one_game"] = run_wrapped

    def p_nh(state, down, sd, gsr):
        return float(nh_grid[state, down - 1, int(round(sd)) + 24, min(int(round(gsr / 15.0)), len(GSR_G) - 1)])

    def p_q2(down, dist, yl, sd, gsr, off_to, def_to):
        key = (down, int(round(dist)), int(round(yl)), int(round(sd)), int((gsr - 1800.0) // 5), int(off_to), int(def_to))
        v = q2c.get(key)
        if v is None:
            x = np.array([[down, round(dist), round(yl), round(sd), (key[4]) * 5 + 2.5, off_to, def_to]], dtype=np.float64)
            v = np.cumsum(q2_model.predict_proba(x)[0])
            q2c[key] = v
        return v

    def decide(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat):
        idx = dec(idx, rng, tbl, qtr, clock_val, in_ot, down, distance, yardline, score_diff, off_to, def_to, phase, time_feat)
        if st["key"] != dv._G.get("task_key"):
            st["key"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(4097, cfg.get("seed", 3))
        rng2 = st["rng"]
        poss = sys._getframe(1).f_locals["possessions"]
        state = st["prev"] if (poss == st["poss"] and st["scr"]) else 2
        dk = down if down in (1, 2, 3, 4) else 4
        c0 = int(code_arr[idx])
        tc = c0
        sdc = min(max(score_diff, -24.0), 24.0)
        if e3 and phase == 1 and not in_ot and c0 != 6:
            cp = p_q2(dk, distance, yardline, sdc, clock_val, off_to, def_to)
            tc = int(q2_classes[min(int(np.searchsorted(cp, rng2.random() * cp[-1])), len(q2_classes) - 1)])
        nh_t = None
        if e1 and tc in (0, 1):
            nh_t = 1 if rng2.random() < p_nh(state, dk, sdc, clock_val) else 0
        db = S.bucket(distance) if (e2 and dk == 4 and tc in (0, 1)) else None
        if db is not None:
            tc = 1 if rng2.random() < S.pass_rate[min(db, len(S.pass_rate) - 1)] else 0
        need = tc != c0 or (nh_t is not None and S.nh[idx] != nh_t) or (db is not None and S.db[idx] != db)
        if need:
            same = tc == c0
            j = S.pick(rng, dk, phase, tc, int(S.tov[idx]) if same else None, int(S.to[idx]) if same else None, nh_t, db, distance, yardline, score_diff, time_feat, off_to, def_to)
            if j is not None:
                idx = j
        cf = int(code_arr[idx])
        st["poss"] = poss
        st["scr"] = cf in (0, 1)
        st["prev"] = int(S.nh[idx])
        return idx

    ns["DECIDE"] = decide


def u4e_init_budget(setting):
    kd.tfix_init_budget(setting)
    install_u4e()


def u4e_init_ss(setting):
    kd.tfix_init(setting)
    install_u4e()


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
    gen.init_worker = u4e_init_budget
    gen.play_season = u3.s_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(bud.OUT / "sim_games.parquet")
    print("done", len(games), el, flush=True)


def cmd_e5(args):
    import mod25e_scorestate as ss

    ss.s_init = u4e_init_ss
    args.scale = 1.0
    ss.cmd_e5(args)


def cmd_analyze(args):
    u3.OUT = OUT / f"analyze_{Path(args.post_dir).name}"
    u3.OUT.mkdir(parents=True, exist_ok=True)
    u3.bud.OUT = Path(args.post_dir)
    u3.cmd_analyze(argparse.Namespace(minsid=2, boot=args.boot))


def cmd_check(args):
    src = (REPO / "scripts" / "sim09_u4b.py").read_text()
    old = 'CRZK = REPO / "artifacts" / "sim09" / "u3d" / "play_crzk"'
    assert old in src
    src = src.replace(old, f"CRZK = Path(r'{Path(args.post_dir)}')")
    for mode in args.modes.split(","):
        sys.argv = ["u4b"] + ([mode] if mode != "1" else [])
        print("\n######## u4b mode", mode, flush=True)
        exec(compile(src, "u4b", "exec"), {"__name__": "u4b_run", "__file__": str(REPO / "scripts" / "sim09_u4b.py")})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="u4e")
    s.add_argument("--out-dir", dest="out_dir", default=None)
    s.add_argument("--worlds", type=int, default=6)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=21)
    e = sub.add_parser("e5")
    e.add_argument("--variant", default="u4e")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    n = sub.add_parser("analyze")
    n.add_argument("--post-dir", dest="post_dir", required=True)
    n.add_argument("--boot", type=int, default=500)
    c = sub.add_parser("check")
    c.add_argument("--post-dir", dest="post_dir", required=True)
    c.add_argument("--modes", default="1,3")
    args = ap.parse_args()
    if args.cmd == "sim" and args.out_dir is None:
        args.out_dir = str(OUT / f"play_{args.variant}")
    {"fit": cmd_fit, "sim": cmd_sim, "e5": cmd_e5, "analyze": cmd_analyze, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
