import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25d_variance as dv  # noqa: E402
import sim09_shell as sh  # noqa: E402

OUTD = REPO / "artifacts" / "sim09" / "u3b"
ART = OUTD / "posture_policy.joblib"
TAGS = OUTD / "pool_tags.npz"
POLICY_SEASONS = (2016, 2017)
STATE_P = ["sd", "zs", "tsec", "half_sec", "qtr", "down", "home", "to_off", "to_def"]
dv.DV["crzp"] = dict(dv.DV["crz"], posture=1)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def mean_ll(p, y):
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def fit_tilt(lp, X, y):
    c = lp.shape[1]

    def mat(a):
        return np.vstack([np.zeros((1, 2)), a.reshape(c - 1, 2)])

    def nll(a):
        return mean_ll(softmax(lp + X @ mat(a).T), y)

    z = np.zeros(2 * (c - 1))
    r = minimize(nll, z, method="L-BFGS-B")
    return mat(r.x), nll(z), float(r.fun)


def pool_frame():
    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    import mod25_generator as gen

    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    trans = dv.sim.build_transition_frame(pbp)
    eo, ef = dv.blup_rows(pbp, trans, dv.TRAIN)
    lo, ld = gen.league_means()
    keys = trans[["game_id", "play_id"]].reset_index(drop=True)
    keys["play_id"] = keys["play_id"].astype(np.int64)
    return keys, (lo + eo).astype(np.float64), (ld + ef).astype(np.float64), pbp


def probs_full(m, X, nc):
    out = np.zeros((len(X), nc))
    out[:, list(m.classes_)] = m.predict_proba(X)
    return out


def cmd_fit(args):
    OUTD.mkdir(parents=True, exist_ok=True)
    d = sh.build()
    d = d[d.season.isin(POLICY_SEASONS) & d[STATE_P].notna().all(axis=1) & d.box.notna() & d.ndb.notna()].copy()
    d["play_id"] = d["play_id"].astype(np.int64)
    lo, hi = d.box.quantile([0.025, 0.975]).tolist()
    d["box_c"] = d.box.clip(lo, hi).round()
    dl, dh = d.ndb.quantile([0.025, 0.975]).tolist()
    d["db_c"] = d.ndb.clip(dl, dh).round()
    cb = sorted(d.box_c.unique().tolist())
    cd = sorted(d.db_c.unique().tolist())
    d["bi"] = np.searchsorted(cb, d.box_c)
    d["di"] = np.searchsorted(cd, d.db_c)
    keys, off_row, def_row, pbp = pool_frame()
    keys["pi"] = np.arange(len(keys))
    d = d.merge(keys, on=["game_id", "play_id"], how="inner").reset_index(drop=True)
    print("policy rows", len(d), "by season", d.groupby("season").size().to_dict(), "box classes", cb, "db classes", cd, flush=True)
    fb = STATE_P
    fd_ = STATE_P + ["box_c"]
    s16 = d[d.season == 2016]
    s17 = d[d.season == 2017]
    cfg_b = sh.fit_clf(s16[fb].to_numpy(), s16.bi.to_numpy(), s17[fb].to_numpy(), s17.bi.to_numpy())
    cfg_d = sh.fit_clf(s16[fd_].to_numpy(), s16.di.to_numpy(), s17[fd_].to_numpy(), s17.di.to_numpy())
    print("cfg box", cfg_b, "cfg db", cfg_d, flush=True)
    oof_b = np.zeros((len(d), len(cb)))
    oof_d = np.zeros((len(d), len(cd)))
    for a_, b_ in ((2016, 2017), (2017, 2016)):
        tr = (d.season == a_).to_numpy()
        te = (d.season == b_).to_numpy()
        mb = sh.refit(d.loc[tr, fb].to_numpy(), d.loc[tr, "bi"].to_numpy(), cfg_b)
        md = sh.refit(d.loc[tr, fd_].to_numpy(), d.loc[tr, "di"].to_numpy(), cfg_d)
        oof_b[te] = probs_full(mb, d.loc[te, fb].to_numpy(), len(cb))
        oof_d[te] = probs_full(md, d.loc[te, fd_].to_numpy(), len(cd))
    oof_b = np.maximum(oof_b, oof_b[oof_b > 0].min())
    oof_d = np.maximum(oof_d, oof_d[oof_d > 0].min())
    oof_b /= oof_b.sum(1, keepdims=True)
    oof_d /= oof_d.sum(1, keepdims=True)
    pi = d.pi.to_numpy()
    X = np.column_stack([off_row[pi], def_row[pi]])
    bi, di = d.bi.to_numpy(), d.di.to_numpy()
    Ab, nb0, nb1 = fit_tilt(np.log(oof_b), X, bi)
    Ad, nd0, nd1 = fit_tilt(np.log(oof_d), X, di)
    prior_b = np.bincount(bi, minlength=len(cb)) / len(d)
    prior_d = np.bincount(di, minlength=len(cd)) / len(d)
    print("OOF log loss box: blind %.4f state %.4f state+team %.4f" % (mean_ll(np.tile(prior_b, (len(d), 1)), bi), nb0, nb1))
    print("OOF log loss db given box: blind %.4f state %.4f state+team %.4f" % (mean_ll(np.tile(prior_d, (len(d), 1)), di), nd0, nd1))
    print("tilt box (off,def) per class", cb, np.round(Ab, 3).tolist())
    print("tilt db (off,def) per class", cd, np.round(Ad, 3).tolist(), flush=True)
    pb = softmax(np.log(oof_b) + X @ Ab.T)
    pdd = softmax(np.log(oof_d) + X @ Ad.T)
    ar = np.arange(len(d))
    n = len(keys)
    tag_b = np.full(n, -1, dtype=np.int64)
    tag_d = np.full(n, -1, dtype=np.int64)
    logden = np.zeros(n)
    tag_b[pi] = bi
    tag_d[pi] = di
    logden[pi] = np.log(pb[ar, bi]) + np.log(pdd[ar, di])
    np.savez(TAGS, tag_b=tag_b, tag_d=tag_d, logden=logden, box_vals=np.array(cb), db_vals=np.array(cd))
    mb = sh.refit(d[fb].to_numpy(), bi, cfg_b)
    md = sh.refit(d[fd_].to_numpy(), di, cfg_d)
    assert list(mb.classes_) == list(range(len(cb))) and list(md.classes_) == list(range(len(cd)))
    joblib.dump({"box_model": mb, "db_model": md, "A_b": Ab, "A_d": Ad, "box_vals": cb, "db_vals": cd, "to_off": float(d.to_off.mean()), "to_def": float(d.to_def.mean()), "clip_box": [lo, hi], "clip_db": [dl, dh], "seasons": POLICY_SEASONS}, ART)
    tagged = int((tag_b >= 0).sum())
    sc = pbp[pbp.play_type.isin(["pass", "run"])].groupby("season").size()
    print("pool rows", n, "tagged", tagged, "share of pool %.4f" % (tagged / n), "2016-17 share of scrimmage pool %.4f" % (sc.loc[[2016, 2017]].sum() / sc.sum()), "tagged share of 2016-17 scrimmage %.4f" % (tagged / sc.loc[[2016, 2017]].sum()), flush=True)


class PostureCache(dict):
    def __init__(self, q, ctx):
        super().__init__()
        self.q = q
        self.ctx = ctx

    def __setitem__(self, wkey, v):
        if self.q >= 5:
            return dict.__setitem__(self, wkey, v)
        c = self.ctx
        key, off_sim, def_sim, home = wkey
        nb = c["tables"]["nn_cache_cond"][key]
        cdf, _ = v
        w = np.diff(cdf, prepend=0.0) * c["factor"](self.q, key, off_sim, def_sim, home, nb)
        s = w.sum()
        if not np.isfinite(s) or s <= 0.0:
            return dict.__setitem__(self, wkey, v)
        cdf2 = np.cumsum(w)
        return dict.__setitem__(self, wkey, (cdf2, float(cdf2[-1])))


def install_posture(cfg):
    sim = dv.sim
    art = joblib.load(ART)
    tg = np.load(TAGS)
    tag_b, tag_d, logden = tg["tag_b"], tg["tag_d"], tg["logden"]
    tables = dv._G["tables"]
    ns = dv._G["ns"]
    mb, md, Ab, Ad = art["box_model"], art["db_model"], art["A_b"], art["A_d"]
    box_vals = np.array(art["box_vals"], dtype=float)
    ncb = len(box_vals)
    cache = {}
    ctx = {"tables": tables}

    def probs_at(q, key, home):
        k2 = (q, key, home)
        r = cache.get(k2)
        if r is not None:
            return r
        down, phase, r_dist, r_fp, r_score, r_time, r_off, r_def = key
        sd = r_score * sim.ROUND_SCORE
        time_raw = r_time * sim.ROUND_TIME
        tsec = time_raw + (1800.0 if q <= 2 else 0.0)
        zs = sd / np.sqrt(tsec / 60.0 + 1.0)
        to_o = float(r_off) if phase in sim.LATE_PHASES else art["to_off"]
        to_d = float(r_def) if phase in sim.LATE_PHASES else art["to_def"]
        vals = {"sd": sd, "zs": zs, "tsec": tsec, "half_sec": time_raw, "qtr": float(q), "down": float(down), "home": float(home), "to_off": to_o, "to_def": to_d}
        x = np.array([[vals[n] for n in STATE_P]])
        lb = np.log(np.maximum(mb.predict_proba(x)[0], 1e-12))
        xd = np.column_stack([np.repeat(x, ncb, axis=0), box_vals])
        ld = np.log(np.maximum(md.predict_proba(xd), 1e-12))
        if len(cache) > 400000:
            cache.clear()
        cache[k2] = (lb, ld)
        return lb, ld

    def factor(q, key, off_sim, def_sim, home, nb):
        lb, ld = probs_at(q, key, home)
        t = np.array([off_sim, def_sim])
        pb = softmax(lb + Ab @ t)
        pdd = softmax(ld + Ad @ t)
        b = tag_b[nb]
        m = b >= 0
        f = np.ones(len(nb))
        f[m] = np.exp(np.log(pb[b[m]]) + np.log(pdd[b[m], tag_d[nb[m]]]) - logden[nb[m]])
        return f

    ctx["factor"] = factor
    caches = {q: PostureCache(q, ctx) for q in range(1, 6)}
    base = ns["pick_index_nn_conditioned"]

    def pick(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim):
        q = int(sys._getframe(1).f_locals["qtr"])
        tbl["nn_weight_cache_cond"] = caches[min(max(q, 1), 5)]
        return base(rng, tbl, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)

    ns["pick_index_nn_conditioned"] = pick


def posture_init(setting):
    import mod25e_scorestate as ss

    ss.s_init(setting)
    cfg = dv._G["cfg"]
    if cfg.get("posture"):
        install_posture(cfg)


def posture_init_budget(setting):
    import mod25e_budget as bud

    bud.e_init(setting)
    cfg = dv._G["cfg"]
    if cfg.get("posture"):
        install_posture(cfg)


def cmd_sim(args):
    import mod25_generator as gen
    import mod25e_budget as bud
    import sim09_u3a as u3

    bud.OUT = Path(args.out_dir).resolve()
    os.environ["BUD_OUT"] = str(bud.OUT)
    bud.OUT.mkdir(parents=True, exist_ok=True)
    dv.c25.ensure_policies()
    v = dv.DV[args.variant]
    cfgj = json.dumps(dict(v, seed=args.seed, name=f"{args.variant}_s{v['scale']:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": v["scale"], "yard_gain": v["yard_gain"], "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = posture_init_budget
    gen.play_season = u3.s_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(bud.OUT / "sim_games.parquet")
    print("done", len(games), el, flush=True)


def cmd_e5(args):
    import mod25e_scorestate as ss

    ss.s_init = posture_init
    args.scale = 1.0
    ss.cmd_e5(args)


CELLS = {"trailer_final2": lambda d: (d.sd < 0) & (d.tsec <= 120) & (d.qtr == 4), "leader_final2": lambda d: (d.sd > 0) & (d.tsec <= 120) & (d.qtr == 4),
         "trailer_q4": lambda d: (d.sd < 0) & (d.qtr == 4), "leader_q4": lambda d: (d.sd > 0) & (d.qtr == 4), "leader_q3": lambda d: (d.sd > 0) & (d.qtr == 3),
         "trailer_q3": lambda d: (d.sd < 0) & (d.qtr == 3), "close_all": lambda d: d.sd.abs() <= 3}


def cmd_mix(args):
    import mod25e_budget as bud
    import sim09_u3a as u3

    art = joblib.load(ART)
    tg = np.load(TAGS)
    clo, chi = art["clip_box"]
    dlo, dhi = art["clip_db"]
    real = sh.build()
    real = real[real.season.between(2018, 2025) & real.box.notna() & real.ndb.notna()].copy()
    real["bx"] = real.box.clip(clo, chi).round()
    real["dbx"] = real.ndb.clip(dlo, dhi).round()
    light, heavy_db = float(np.median(tg["box_vals"])), float(np.median(tg["db_vals"]))
    rows = []
    for nm, f in CELLS.items():
        r = real[f(real)]
        rows.append(("real_2018_25", nm, len(r), r.bx.mean(), (r.bx < light).mean(), r.dbx.mean(), (r.dbx > heavy_db).mean()))
    for tag, od in (("crz", args.base_dir), ("crzp", args.post_dir)):
        bud.OUT = Path(od)
        P = u3.load_sim_frames(2)
        P = P[P.code.isin([0, 1])].copy()
        idx = P["idx"].to_numpy(np.int64)
        P["tb"] = tg["tag_b"][idx]
        P["td"] = tg["tag_d"][idx]
        P["bx"] = np.where(P.tb >= 0, tg["box_vals"][np.maximum(P.tb, 0)], np.nan)
        P["dbx"] = np.where(P.td >= 0, tg["db_vals"][np.maximum(P.td, 0)], np.nan)
        P["tsec"] = P.gsr
        P["qtr"] = P.qtr.astype(int)
        print(tag, "scrimmage draws", len(P), "tagged share %.4f" % (P.tb >= 0).mean())
        tagged = P[P.tb >= 0]
        for nm, f in CELLS.items():
            r = tagged[f(tagged)]
            rows.append((tag, nm, len(r), r.bx.mean(), (r.bx < light).mean(), r.dbx.mean(), (r.dbx > heavy_db).mean()))
    R = pd.DataFrame(rows, columns=["source", "cell", "n", "box_mean", "box_light", "db_mean", "db_heavy"])
    R.to_csv(OUTD / "posture_mix.csv", index=False)
    print(R.round(4).to_string(index=False))


def cmd_analyze(args):
    import argparse as ap_
    import mod25e_budget as bud
    import sim09_u3a as u3

    u3.OUT = OUTD / "analyze"
    bud.OUT = Path(args.post_dir)
    u3.cmd_analyze(ap_.Namespace(minsid=2, boot=args.boot))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="crzp")
    s.add_argument("--out-dir", dest="out_dir", default=str(OUTD / "play_crzp"))
    s.add_argument("--worlds", type=int, default=6)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=21)
    e = sub.add_parser("e5")
    e.add_argument("--variant", default="crzp")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    a = sub.add_parser("analyze")
    a.add_argument("--post-dir", dest="post_dir", default=str(OUTD / "play_crzp"))
    a.add_argument("--boot", type=int, default=500)
    m = sub.add_parser("mix")
    m.add_argument("--base-dir", dest="base_dir", default=str(REPO / "artifacts" / "sim09" / "u3a_play"))
    m.add_argument("--post-dir", dest="post_dir", default=str(OUTD / "play_crzp"))
    args = ap.parse_args()
    {"fit": cmd_fit, "sim": cmd_sim, "e5": cmd_e5, "mix": cmd_mix, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
