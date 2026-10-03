import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "artifacts" / "mod25e3" / "kstate"
SNAP = REPO / "data" / "pbp" / "raw" / "20260929T191306Z"
POOL = tuple(range(2009, 2018))
TUNE = tuple(range(2009, 2014))
VAL = (2014, 2015)
FIT = tuple(range(2009, 2016))
TEST = (2016, 2017)
KGRID = [50, 100, 200, 400, 800]
HGRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
KBASE, HBASE = 200, 1.0
KMAX = max(KGRID)
YLO, YHI = -20, 80
ALPHA = 1.0
NBINS = 10
NBOOT = 400
LAM = 1_000_000.0
NOSTR = 1000.0


def load_trans():
    import sim04_engine as eng

    eng.PBP_SNAPSHOT_DIR = SNAP
    pbp = eng.load_reg_seasons(POOL)
    tr = eng.build_transition_frame(pbp, team_ratings=eng.load_team_ratings())
    tr["season"] = tr["game_id"].str[:4].astype(int)
    gf = pd.read_parquet(REPO / "data" / "processed" / "game_features_pbp.parquet", columns=["game_id", "home_score", "away_score"]).set_index("game_id")
    hw = np.sign(gf["home_score"] - gf["away_score"])
    tr["hw"] = tr["game_id"].map(hw)
    offh = tr["is_home_off"].to_numpy()
    tr["won"] = np.where(tr["hw"] == 0, np.nan, np.where(offh == 1, tr["hw"] > 0, tr["hw"] < 0).astype(float))
    return tr


def wp_features(sc, tm, dn, ds, fp):
    return np.column_stack([sc, tm, dn, ds, fp])


def fit_wp(tr):
    from sklearn.ensemble import HistGradientBoostingClassifier

    t = tr[tr["won"].notna() & tr["season"].isin(FIT)]
    X = wp_features(t["sc_raw"], t["time_raw"], t["down_i"], t["dist_raw"], t["fp_raw"])
    m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    m.fit(X, t["won"].astype(int).values)
    return m


def late_edges(tr, wp):
    q = tr["qtr_actual"].isin([3, 4]) & tr["season"].isin(FIT)
    e = np.unique(np.quantile(wp[q.to_numpy()], np.linspace(0, 1, NBINS + 1)))
    e[0], e[-1] = 0.0, 1.0001
    return e


class Split:
    def __init__(self, tr, wp, fit_seasons, val_seasons):
        import sim04_engine as eng

        ok = tr["down_i"].isin([1, 2, 3]) & tr["play_type_code"].isin([0, 1])
        self.fit = tr[ok & tr["season"].isin(fit_seasons)].reset_index(drop=True)
        vm = (ok & tr["season"].isin(val_seasons)).to_numpy()
        self.val = tr[vm].reset_index(drop=True)
        self.wp_val = wp[vm]
        self.h = eng.TEAM_KERNEL_H_SCALE * float(np.nanstd(np.concatenate([tr["off_row"].to_numpy(), tr["off_row"].to_numpy()])))
        self.nb, self.dk = self.neighbours(eng)
        nk = 2 * (YHI - YLO + 1)
        f, v = self.fit, self.val
        self.kf = self.key(f)
        self.kv = self.key(v)
        self.fdf = self.fd(f)
        self.fdv = self.fd(v)
        self.g = np.zeros((4, nk))
        self.gfd = np.zeros(4)
        for dn in (1, 2, 3):
            m = (f["down_i"] == dn).to_numpy()
            c = np.bincount(self.kf[m], minlength=nk) + 0.5
            self.g[dn] = c / c.sum()
            self.gfd[dn] = (self.fdf[m].sum() + 0.5) / (m.sum() + 1.0)
        self.dnv = v["down_i"].to_numpy()
        self.pf_ = (f["play_type_code"].to_numpy() == 1).astype(float)
        self.pv = (v["play_type_code"].to_numpy() == 1).astype(float)
        self.yf = np.clip(f["yards_gained"].to_numpy(), YLO, YHI)
        self.yv = np.clip(v["yards_gained"].to_numpy(), YLO, YHI)
        self.gp = np.zeros(4)
        for dn in (1, 2, 3):
            m = (f["down_i"] == dn).to_numpy()
            self.gp[dn] = (self.pf_[m].sum() + 0.5) / (m.sum() + 1.0)
        self.ho = f["off_row"].to_numpy()
        self.hd = f["def_row"].to_numpy()
        self.hh = f["is_home_off"].to_numpy()
        self.vo = v["off_row"].to_numpy()
        self.vd = v["def_row"].to_numpy()
        self.vh = v["is_home_off"].to_numpy()

    @staticmethod
    def key(d):
        yc = np.clip(d["yards_gained"].to_numpy(), YLO, YHI).astype(int)
        return (d["play_type_code"].to_numpy() == 1).astype(int) * (YHI - YLO + 1) + (yc - YLO)

    @staticmethod
    def fd(d):
        return ((d["yards_gained"].to_numpy() >= d["dist_raw"].to_numpy()) & ~d["possession_flip"].to_numpy()).astype(float)

    def neighbours(self, eng):
        def fm(d):
            return eng.feature_matrix(d["dist_raw"].to_numpy(), d["fp_raw"].to_numpy(), d["sc_raw"].to_numpy(), d["time_raw"].to_numpy(), d["off_to_raw"].to_numpy(), d["def_to_raw"].to_numpy(), d["phase"].to_numpy())

        Xf, Xv = fm(self.fit), fm(self.val)
        phf, phv = self.fit["phase"].to_numpy(), self.val["phase"].to_numpy()
        dnf, dnv = self.fit["down_i"].to_numpy(), self.val["down_i"].to_numpy()
        nb = np.zeros((len(self.val), KMAX), dtype=np.int32)
        dk = np.zeros((len(self.val), 2), dtype=np.float32)
        for dn in (1, 2, 3):
            for ph in range(5):
                qm = np.flatnonzero((dnv == dn) & (phv == ph))
                if len(qm) == 0:
                    continue
                pm = np.flatnonzero((dnf == dn) & eng.phase_pool_mask(phf, ph))
                k = min(KMAX, len(pm))
                dd, ind = cKDTree(Xf[pm]).query(Xv[qm], k=k, workers=3)
                nb[qm, :k] = pm[ind]
                if k < KMAX:
                    nb[qm, k:] = pm[ind[:, -1:]]
                dk[qm, 0] = dd[:, min(KBASE, k) - 1]
                dk[qm, 1] = dd[:, 0]
        return nb, dk

    def weights(self, rows, K, hm):
        nb = self.nb[rows, :K]
        d2 = (self.ho[nb] - self.vo[rows, None]) ** 2 + (self.hd[nb] - self.vd[rows, None]) ** 2
        hh = hm * self.h
        w = np.exp(-d2 / (2.0 * hh * hh)) * np.where(self.hh[nb] == self.vh[rows, None], LAM, 1.0)
        s = w.sum(1, keepdims=True)
        bad = ~np.isfinite(s[:, 0]) | (s[:, 0] <= 0)
        w[bad] = 1.0
        s[bad] = K
        w = w / s
        return nb, w, 1.0 / (w**2).sum(1)

    def losses(self, rows, K, hm):
        nb, w, ess = self.weights(rows, K, hm)
        kv = self.kv[rows]
        dn = self.dnv[rows]
        c = (w * (self.kf[nb] == kv[:, None])).sum(1)
        p = (ess * c + ALPHA * self.g[dn, kv]) / (ess + ALPHA)
        pf = (w * self.fdf[nb]).sum(1)
        pfs = (ess * pf + ALPHA * self.gfd[dn]) / (ess + ALPHA)
        y = self.fdv[rows]
        pp = (w * self.pf_[nb]).sum(1)
        pps = (ess * pp + ALPHA * self.gp[dn]) / (ess + ALPHA)
        z = self.pv[rows]
        x = self.yf[nb]
        o = np.argsort(x, axis=1)
        xs = np.take_along_axis(x, o, 1)
        ws = np.take_along_axis(w, o, 1)
        cum = np.cumsum(ws, axis=1)
        crps = (w * np.abs(x - self.yv[rows, None])).sum(1) - (ws * xs * (2 * cum - ws - 1)).sum(1)
        return {"joint": -np.log(p), "fd": -(y * np.log(pfs) + (1 - y) * np.log(1 - pfs)), "pass": -(z * np.log(pps) + (1 - z) * np.log(1 - pps)), "crps": crps, "ess": ess}


def bins_of(wp, edges):
    return np.clip(np.searchsorted(edges, wp, side="right") - 1, 0, len(edges) - 2)


def boot_delta(a, b, games, reps=NBOOT, seed=7):
    d = a - b
    u, inv = np.unique(games, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    cnt = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(reps, len(u)))
    m = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m > 0).mean())


def cmd_search(args):
    OUT.mkdir(parents=True, exist_ok=True)
    import joblib

    tr = load_trans()
    model = fit_wp(tr)
    wp_all = model.predict_proba(wp_features(tr["sc_raw"], tr["time_raw"], tr["down_i"], tr["dist_raw"], tr["fp_raw"]))[:, 1]
    edges = late_edges(tr, wp_all)
    joblib.dump(model, OUT / "wp_model.pkl")
    lines = [f"wp bin edges (state-only wp, Q3/Q4 deciles 2009-15): {np.round(edges, 4).tolist()}"]
    sv = Split(tr, wp_all, TUNE, VAL)
    bv = bins_of(sv.wp_val, edges)
    chosen = {}
    COMP = ("pass", "crps", "fd")
    lines.append(f"looks: {NBINS} bins x {len(KGRID)} K x {len(HGRID)} h = {NBINS * len(KGRID) * len(HGRID)}; objective J = sum over (pass log loss, yards CRPS, first-down log loss) of bin mean / same at K 200 h 1 (lower better), val 2014-15, fit 2009-13")
    for b in range(len(edges) - 1):
        rows = np.flatnonzero(bv == b)
        grid = {}
        for K in KGRID:
            for hm in HGRID:
                r = sv.losses(rows, K, hm)
                grid[(K, hm)] = {c: float(r[c].mean()) for c in COMP + ("joint", "ess")}
        ref = grid[(KBASE, HBASE)]
        J = {k: sum(v[c] / ref[c] for c in COMP) for k, v in grid.items()}
        best = min(J, key=J.get)
        chosen[b] = {"K": int(best[0]), "hm": float(best[1])}
        lines.append(f"bin {b} wp {edges[b]:.3f}-{edges[b + 1]:.3f} n {len(rows)} base ess {ref['ess']:.1f} | chosen K {best[0]} hm {best[1]} J {J[best]:.4f} (base 3) ess {grid[best]['ess']:.1f}")
        lines.append("   J K-only(hm 1): " + " ".join(f"{K}:{J[(K, 1.0)]:.4f}" for K in KGRID) + " | h-only(K200): " + " ".join(f"{h}:{J[(KBASE, h)]:.4f}" for h in HGRID) + f" | nostrength(K200): {sum(sv.losses(rows, KBASE, NOSTR)[c].mean() / ref[c] for c in COMP):.4f}")
    st = Split(tr, wp_all, FIT, TEST)
    bt = bins_of(st.wp_val, edges)
    gid = st.val["game_id"].to_numpy()
    n = len(st.val)
    names = ("base", "chosen", "nostr")
    arrs = {(nm, c): np.zeros(n) for nm in names for c in COMP + ("joint", "ess")}
    for b in range(len(edges) - 1):
        rows = np.flatnonzero(bt == b)
        for nm, (K, hm) in zip(names, ((KBASE, HBASE), (chosen[b]["K"], chosen[b]["hm"]), (KBASE, NOSTR))):
            r = st.losses(rows, K, hm)
            for c in COMP + ("joint", "ess"):
                arrs[(nm, c)][rows] = r[c]
    lines.append("-- test 2016-17 (fit 2009-15); delta = base minus chosen loss (positive = chosen better), game bootstrap; strength-term value = nostrength minus base (positive = strength term helps)")
    res = {"edges": edges.tolist(), "chosen": chosen, "test": {}}
    for b in range(len(edges) - 1):
        m = bt == b
        cells = []
        res["test"][b] = {"n": int(m.sum())}
        for c in COMP + ("joint",):
            d = boot_delta(arrs[("base", c)][m], arrs[("chosen", c)][m], gid[m])
            s_ = boot_delta(arrs[("nostr", c)][m], arrs[("base", c)][m], gid[m])
            cells.append(f"{c} d {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] pp {d[3]:.2f} str {s_[0]:+.4f} [{s_[1]:+.4f},{s_[2]:+.4f}] pp {s_[3]:.2f}")
            res["test"][b][c] = {"delta": d, "strength": s_}
        dk = st.dk[m]
        lines.append(f"bin {b} n {int(m.sum())} ESS base {arrs[('base', 'ess')][m].mean():.1f} (p10 {np.percentile(arrs[('base', 'ess')][m], 10):.1f}) chosen {arrs[('chosen', 'ess')][m].mean():.1f} | kth(200)-dist {dk[:, 0].mean():.3f} nn1-dist {dk[:, 1].mean():.3f}")
        lines.extend("    " + x for x in cells)
    (OUT / "kstate.json").write_text(json.dumps(res, indent=1))
    (OUT / "search.txt").write_text(chr(10).join(lines))
    print(chr(10).join(lines))


def load_tables(null):
    import joblib

    res = json.loads((OUT / "kstate.json").read_text())
    edges = np.array(res["edges"])
    ch = res["chosen"]
    kt = np.array([KBASE if null else ch[str(b)]["K"] for b in range(len(edges) - 1)])
    ht = np.array([HBASE if null else ch[str(b)]["hm"] for b in range(len(edges) - 1)])
    return joblib.load(OUT / "wp_model.pkl"), edges, kt, ht


def install_ks(dv, null):
    model, edges, kt, ht = load_tables(null)
    ns = dv._G["ns"]
    cache = {}

    def bucket(key):
        b = cache.get(key)
        if b is None:
            dn, _, rd, rf, rs, rt = key[:6]
            x = wp_features([rs * ns["ROUND_SCORE"]], [rt * ns["ROUND_TIME"]], [dn], [rd * ns["ROUND_DIST"]], [rf * ns["ROUND_FP"]])
            wp = float(model.predict_proba(x)[0, 1])
            b = int(bins_of(np.array([wp]), edges)[0])
            cache[key] = b
        return b

    ns["KS_K"] = lambda key: int(kt[bucket(key)])
    ns["KS_H"] = lambda key: float(ht[bucket(key)])
    tb = dv._G["tables"]
    tb["nn_weight_cache_cond"].clear()
    nc = tb["nn_cache_cond"]
    keys = list(nc.keys())
    if keys:
        arr = np.array([k[:6] for k in keys], dtype=float)
        x = wp_features(arr[:, 4] * ns["ROUND_SCORE"], arr[:, 5] * ns["ROUND_TIME"], arr[:, 0], arr[:, 2] * ns["ROUND_DIST"], arr[:, 3] * ns["ROUND_FP"])
        bs = bins_of(model.predict_proba(x)[:, 1], edges)
        for k, b in zip(keys, bs):
            cache[k] = int(b)
            want = int(kt[b])
            if want < len(nc[k]):
                nc[k] = nc[k][:want]
            elif want > len(nc[k]):
                del nc[k]


def patched_source(dv):
    orig = inspect.getsource
    ka = "k_eff = min(k_state, len(sub_idx))"
    kb = 'h = tables["team_kernel_h"]'

    def gs(o):
        s = orig(o)
        if o is dv.sim.pick_index_nn_conditioned:
            assert ka in s and kb in s
            s = s.replace(ka, "k_eff = min(KS_K(key), len(sub_idx))").replace(kb, 'h = tables["team_kernel_h"] * KS_H(key)')
        return s

    return orig, gs


def run_init(base_init, setting):
    import mod25d_variance as dv

    orig, gs = patched_source(dv)
    inspect.getsource = gs
    try:
        base_init(setting)
    finally:
        inspect.getsource = orig
    install_ks(dv, dv._G["cfg"].get("ks") == "null")


def ks_init_budget(setting):
    import sim09_hk as hk

    run_init(hk.hk_init_budget, setting)


def ks_init_ss(setting):
    import sim09_hk as hk

    run_init(hk.hk_init_ss, setting)


def setup_variant(name, null):
    import mod25d_variance as dv
    import mod25e_cov as cv

    cv.patch_generator()
    dv.DV["crzhc"] = dict(dv.DV["crzhk"])
    dv.DV[name] = dict(dv.DV["crzhk"], ks="null" if null else "on")
    return dv


def cmd_sim(args):
    import mod25_generator as gen
    import mod25e_budget as bud
    import sim09_hk as hk
    import sim09_u3a as u3

    dv = setup_variant(args.variant, args.null)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bud.OUT = out
    os.environ["BUD_OUT"] = str(out)
    dv.c25.ensure_policies()
    v = dv.DV[args.variant]
    cfgj = json.dumps(dict(v, seed=args.seed, name=f"{args.variant}_s{v['scale']:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": v["scale"], "yard_gain": v["yard_gain"], "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = ks_init_budget
    gen.play_season = u3.s_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(out / "sim_games.parquet")
    arrs = {}
    for (w, s), (weekly, qb_out) in latents.items():
        arrs[f"w_{w}_{s}"] = weekly
        arrs[f"q_{w}_{s}"] = qb_out
    np.savez(out / "latents.npz", **arrs)
    print("done", len(games), el, flush=True)


def cmd_e5(args):
    import mod25e_scorestate as ss
    import sim09_hk as hk

    setup_variant(args.variant, False)
    ss.s_init = ks_init_ss
    args.scale = 1.0
    ss.cmd_e5(args)


def cmd_diff(args):
    a, b = Path(args.a), Path(args.b)
    x, y = pd.read_parquet(a), pd.read_parquet(b)
    print("equal", x.shape == y.shape and x.equals(y), x.shape, y.shape)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("search")
    s = sub.add_parser("sim")
    s.add_argument("--out-dir", dest="out_dir", required=True)
    s.add_argument("--variant", default="crzhs")
    s.add_argument("--null", action="store_true")
    s.add_argument("--worlds", type=int, default=1)
    s.add_argument("--seasons", type=int, default=1)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--seed", type=int, default=31)
    e = sub.add_parser("e5")
    e.add_argument("--variant", default="crzhs")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    args = ap.parse_args()
    {"search": cmd_search, "sim": cmd_sim, "e5": cmd_e5, "diff": cmd_diff}[args.cmd](args)


if __name__ == "__main__":
    main()
