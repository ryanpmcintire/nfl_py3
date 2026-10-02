import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sim04_engine as eng  # noqa: E402
import sim09_u4d as u4d  # noqa: E402

OUT = REPO / "artifacts" / "sim09" / "u4f"
KL_FILE = OUT / "kernel_consts.json"
CUR = {"a": eng.SCORE_INNER_SCALE, "b": eng.SCORE_OUTER_SCALE, "T": eng.SCALE_TIME, "D": eng.SCALE_YDSTOGO, "F": eng.SCALE_FP,
       "O": eng.SCALE_TIMEOUTS, "K": eng.K_STATE, "H": 0.0}
KMAX = 400
ALPHA = 1.0
YLO, YHI = -20, 80
ALO, AHI = -10, 60
GRID = [0.25, 0.5, 0.7, 1.0, 1.4, 2.0, 4.0]
FINE = [0.8, 0.9, 1.1, 1.25]
HGRID = [0.0]
WORKERS = 3

REG = {}
if KL_FILE.exists():
    REG = json.loads(KL_FILE.read_text())["chosen"]
    import mod25d_variance as dv  # noqa: E402
    import sim09_u3d  # noqa: E402,F401

    dv.DV["crzkl"] = dict(dv.DV["crzk"], kl=REG)


def apply_consts(c):
    import sim04_engine as e

    e.SCORE_INNER_SCALE = float(c["a"])
    e.SCORE_OUTER_SCALE = float(c["b"])
    e.SCALE_TIME = float(c["T"])
    e.SCALE_YDSTOGO = float(c["D"])
    e.SCALE_FP = float(c["F"])
    e.SCALE_TIMEOUTS = float(c["O"])
    e.K_STATE = int(c["K"])


def kl_init_budget(setting):
    import sim09_u3d as u3d

    cfg = json.loads(setting["mech"])
    if cfg.get("kl"):
        apply_consts(cfg["kl"])
    u3d.tfix_init_budget(setting)


def kl_init(setting):
    import sim09_u3d as u3d

    cfg = json.loads(setting["mech"])
    if cfg.get("kl"):
        apply_consts(cfg["kl"])
    u3d.tfix_init(setting)


def prep(d):
    d = d.reset_index(drop=True)
    d["down"] = d["down"].astype(int)
    d["yc"] = np.clip(d["yards_gained"].to_numpy(), YLO, YHI).astype(int)
    d["key"] = d["pass"].to_numpy().astype(int) * (YHI - YLO + 1) + (d["yc"].to_numpy() - YLO)
    d["airc"] = np.clip(d["air_yards"].to_numpy(), ALO, AHI)
    return d


def feats(d, p):
    sc = np.clip(d["score_differential"].to_numpy(), -eng.SCORE_CLIP, eng.SCORE_CLIP)
    ab = np.abs(sc)
    mag = np.minimum(ab, eng.SCORE_INNER) / p["a"] + np.maximum(ab - eng.SCORE_INNER, 0) / p["b"]
    ph = d["phase"].to_numpy()
    tw = np.isin(ph, eng.LATE_PHASES).astype(float)
    cols = [d["ydstogo"].to_numpy() / p["D"], d["yardline_100"].to_numpy() / p["F"], np.sign(sc) * mag, d["time_raw"].to_numpy() / p["T"],
            d["posteam_timeouts_remaining"].fillna(3).to_numpy() * tw / p["O"], d["defteam_timeouts_remaining"].fillna(3).to_numpy() * tw / p["O"]]
    if p["H"]:
        cols.append(d["half"].to_numpy() * p["H"] * (ph == 0))
    return np.column_stack(cols)


def neighbours(fit, val, p, kmax):
    Xf, Xv = feats(fit, p), feats(val, p)
    ph_f, ph_v = fit["phase"].to_numpy(), val["phase"].to_numpy()
    dn_f, dn_v = fit["down"].to_numpy(), val["down"].to_numpy()
    nb = np.zeros((len(val), kmax), dtype=np.int32)
    for dn in (1, 2, 3):
        for ph in range(5):
            qm = np.flatnonzero((dn_v == dn) & (ph_v == ph))
            if len(qm) == 0:
                continue
            pm = np.flatnonzero((dn_f == dn) & eng.phase_pool_mask(ph_f, ph))
            _, ind = cKDTree(Xf[pm]).query(Xv[qm], k=kmax, workers=WORKERS)
            nb[qm] = pm[ind]
    return nb


class Scorer:
    def __init__(self, fit, val):
        self.fit, self.val = fit, val
        nk = 2 * (YHI - YLO + 1)
        self.g = np.zeros((4, nk))
        self.ga = np.zeros((4, AHI - ALO + 1))
        for dn in (1, 2, 3):
            f = fit[fit["down"] == dn]
            c = np.bincount(f["key"].to_numpy(), minlength=nk) + 0.5
            self.g[dn] = c / c.sum()
            fa = f[(f["pass"] == 1) & f["air_yards"].notna()]
            ca = np.bincount((fa["airc"].to_numpy() - ALO).astype(int), minlength=AHI - ALO + 1) + 0.5
            self.ga[dn] = ca / ca.sum()
        self.kf = fit["key"].to_numpy()
        self.pf = fit["pass"].to_numpy()
        self.af = fit["airc"].to_numpy()
        self.pass_ok = (self.pf == 1) & ~np.isnan(fit["air_yards"].to_numpy())
        self.yf = fit["yards_gained"].to_numpy()
        self.kv = val["key"].to_numpy()
        self.dv = val["down"].to_numpy()
        self.pv = val["pass"].to_numpy()
        self.av = val["airc"].to_numpy()
        self.air_rows = np.flatnonzero((self.pv == 1) & val["air_yards"].notna().to_numpy())
        self.yv = val["yards_gained"].to_numpy()
        gj = self.g[self.dv, self.kv]
        self.base1 = -np.log(gj)
        ga_ = self.ga[self.dv[self.air_rows], (self.av[self.air_rows] - ALO).astype(int)]
        self.base2 = -np.log(ga_)
        self.cache_key, self.cache_nb = None, None
        self.fp_ = fit[(fit["pass"] == 1) & fit["air_yards"].notna()].reset_index(drop=True)
        self.vp_ = val[(val["pass"] == 1) & val["air_yards"].notna()].reset_index(drop=True)
        self.cache_key2, self.cache_nb2 = None, None
        self.ref = None

    def nb_for(self, p):
        k = tuple((n, p[n]) for n in ("a", "b", "T", "D", "F", "O", "H"))
        if k != self.cache_key:
            self.cache_nb = neighbours(self.fit, self.val, p, KMAX)
            self.cache_key = k
        return self.cache_nb

    def nb_air(self, p):
        k = tuple((n, p[n]) for n in ("a", "b", "T", "D", "F", "O", "H"))
        if k != self.cache_key2:
            self.cache_nb2 = neighbours(self.fp_, self.vp_, p, KMAX)
            self.cache_key2 = k
        return self.cache_nb2

    def air_crps(self, p):
        nb = self.nb_air(p)[:, : int(p["K"])]
        return u4d.crps_rows(self.fp_["air_yards"].to_numpy()[nb], self.vp_["air_yards"].to_numpy())

    def rows(self, p, kmax=None):
        K = int(p["K"])
        nb = self.nb_for(p)[:, :K]
        c1 = (self.kf[nb] == self.kv[:, None]).sum(1)
        l1 = -np.log((c1 + ALPHA * self.g[self.dv, self.kv]) / (K + ALPHA))
        nba = nb[self.air_rows]
        ok = self.pass_ok[nba]
        c2 = ((self.af[nba] == self.av[self.air_rows][:, None]) & ok).sum(1)
        n2 = ok.sum(1)
        ga = self.ga[self.dv[self.air_rows], (self.av[self.air_rows] - ALO).astype(int)]
        l2 = -np.log((c2 + ALPHA * ga) / (n2 + ALPHA))
        pp = (self.pf[nb].sum(1) + 0.5) / (K + 1)
        lp = -(self.pv * np.log(pp) + (1 - self.pv) * np.log(1 - pp))
        yd = u4d.crps_rows(self.yf[nb], self.yv)
        return {"l1": l1, "l2": l2, "lp": lp, "cy": yd, "ca": self.air_crps(p)}

    def score(self, p):
        r = self.rows(p)
        s1 = r["l1"].mean() / self.base1.mean()
        s2 = r["l2"].mean() / self.base2.mean()
        m = (float(r["lp"].mean()), float(r["cy"].mean()), float(r["ca"].mean()))
        if self.ref is None:
            self.ref = m
        J = sum(x / y for x, y in zip(m, self.ref))
        return {"J": float(J), "air_crps": m[2], "J_joint_rejected": float(s1 + s2), "s_joint": float(s1), "s_air": float(s2), "ll_joint": float(r["l1"].mean()), "ll_air": float(r["l2"].mean()),
                "pass_ll": float(r["lp"].mean()), "yards_crps": float(r["cy"].mean()), "n_air": int(len(self.air_rows))}


def descend(sc, log):
    p = dict(CUR)
    cache = {}

    def ev(q):
        t = tuple(sorted(q.items()))
        if t not in cache:
            t0 = time.time()
            cache[t] = sc.score(q)["J"]
            log(f"  eval {q} J {cache[t]:.5f} {time.time()-t0:.0f}s")
        return cache[t]

    best = ev(p)
    sweeps = [GRID] * 3 + [FINE]
    for si, grid in enumerate(sweeps):
        before = best
        for name in ("a", "b", "T", "D", "F", "O", "K", "H"):
            base = p[name]
            cands = HGRID if name == "H" else [base * m for m in grid]
            for v in cands:
                q = dict(p)
                if name == "K":
                    v = int(round(v))
                    if v < 5 or v > KMAX:
                        continue
                q[name] = v
                if q == p:
                    continue
                s = ev(q)
                if s < best - 1e-9:
                    best, p = s, q
        log(f"sweep {si} best {best:.5f} p {p}")
        if best > before - 1e-6 and si >= 1:
            break
    return p, best, len(cache)


def boot_delta(a, b, games, reps=1000, seed=7):
    d = a - b
    u, inv = np.unique(games, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    cnt = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(reps, len(u)))
    m = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m > 0).mean())


def gt_air(st, test):
    return st.vp_["game_id"].to_numpy()


def cmd_search(args):
    OUT.mkdir(parents=True, exist_ok=True)
    lf = open(OUT / "search.log", "a")

    def log(s):
        print(s, flush=True)
        lf.write(s + "\n")
        lf.flush()

    tf, tv = prep(u4d.real_plays(range(2009, 2014))), prep(u4d.real_plays((2014, 2015)))
    fit, test = prep(u4d.real_plays(range(2009, 2016))), prep(u4d.real_plays((2016, 2017)))
    sc = Scorer(tf, tv)
    sc.score(CUR)
    p, best, n_eval = descend(sc, log)
    cur_val = sc.score(CUR)
    new_val = sc.score(p)
    st = Scorer(fit, test)
    cur_t = st.score(CUR)
    cur_rows = st.rows(CUR)
    new_t = st.score(p)
    new_rows = st.rows(p)
    gt = test["game_id"].to_numpy()
    ga = gt[st.air_rows]
    res = {"declared_objective": "J = passLL/passLL_cur + yardsCRPS/yardsCRPS_cur + airCRPS/airCRPS_cur on tune-val 2014-15 at current constants, K hard cutoff, lower is better",
           "chosen": p, "current": CUR, "evals": n_eval, "val_current": cur_val, "val_chosen": new_val, "test_current": cur_t, "test_chosen": new_t,
           "test_delta_joint_ll": boot_delta(cur_rows["l1"], new_rows["l1"], gt), "test_delta_air_ll": boot_delta(cur_rows["l2"], new_rows["l2"], ga),
           "test_delta_pass_ll": boot_delta(cur_rows["lp"], new_rows["lp"], gt), "test_delta_yards_crps": boot_delta(cur_rows["cy"], new_rows["cy"], gt), "test_delta_air_crps": boot_delta(cur_rows["ca"], new_rows["ca"], gt_air(st, test))}
    (OUT / "kernel_consts.json").write_text(json.dumps(res, indent=1, default=float))
    log(json.dumps(res, indent=1, default=float))


def cmd_cover(args):
    res = json.loads(KL_FILE.read_text())
    test = prep(u4d.real_plays((2016, 2017)))
    fit = prep(u4d.real_plays(range(2009, 2016)))
    out = {}
    pf = fit
    for name, p in (("current", CUR), ("chosen", res["chosen"])):
        K = int(p["K"])
        nb = neighbours(fit, test, p, K)
        sd_f, tr_f = fit["score_differential"].to_numpy(), fit["time_raw"].to_numpy()
        ds = np.abs(sd_f[nb] - test["score_differential"].to_numpy()[:, None])
        dt = np.abs(tr_f[nb] - test["time_raw"].to_numpy()[:, None])
        yd = np.abs(fit["yardline_100"].to_numpy()[nb] - test["yardline_100"].to_numpy()[:, None])
        dd = np.abs(fit["ydstogo"].to_numpy()[nb] - test["ydstogo"].to_numpy()[:, None])
        hf = fit["half"].to_numpy()[nb]
        same = (hf == test["half"].to_numpy()[:, None]).mean()
        pools = {f"{dn}_{ph}": int(((fit["down"] == dn) & eng.phase_pool_mask(fit["phase"].to_numpy(), ph)).sum()) for dn in (1, 2, 3) for ph in range(5)}
        out[name] = {"K": K, "mean_abs_dscore": float(ds.mean()), "frac_dscore_gt7": float((ds > 7).mean()), "mean_abs_dtime_s": float(dt.mean()),
                     "p90_dtime_s": float(np.percentile(dt, 90)), "mean_abs_dyardline": float(yd.mean()), "mean_abs_dydstogo": float(dd.mean()),
                     "same_half_share": float(same), "min_pool": min(pools.values()), "kth_over_pool_max": K / min(pools.values())}
        print(name, out[name], flush=True)
    (OUT / "coverage.json").write_text(json.dumps(out, indent=1))


def cmd_sim(args):
    import sim09_u3d as u3d

    u3d.tfix_init_budget = kl_init_budget
    u3d.cmd_sim(args)


def cmd_analyze(args):
    import argparse as ap_

    import sim09_u3d as u3d

    out = REPO / "artifacts" / "sim09" / "u4f" / "analyze"
    out.mkdir(parents=True, exist_ok=True)
    u3d.u3.OUT = out
    u3d.u3.bud.OUT = Path(args.post_dir)
    u3d.u3.cmd_analyze(ap_.Namespace(minsid=2, boot=args.boot))


def cmd_e5(args):
    import sim09_u3d as u3d

    u3d.tfix_init = kl_init
    u3d.cmd_e5(args)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("search")
    sub.add_parser("cover")
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="crzkl")
    s.add_argument("--out-dir", dest="out_dir", default=str(REPO / "artifacts" / "sim09" / "u4f" / "play_crzkl"))
    s.add_argument("--worlds", type=int, default=6)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=21)
    e = sub.add_parser("e5")
    e.add_argument("--variant", default="crzkl")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    n = sub.add_parser("analyze")
    n.add_argument("--post-dir", dest="post_dir", default=str(REPO / "artifacts" / "sim09" / "u4f" / "play_crzkl"))
    n.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    {"search": cmd_search, "cover": cmd_cover, "sim": cmd_sim, "e5": cmd_e5, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
