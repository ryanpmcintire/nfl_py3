import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
for _k, _v in (("EGT", "1"), ("EGH", "1"), ("F2PR", "1"), ("SIM_FAST", "1"), ("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_k, _v)

import mod25d_variance as dv  # noqa: E402
import mod25e_crG as crG  # noqa: E402
import mod25e_crH as H  # noqa: E402

ART = REPO / "artifacts" / "mod25e3"
COLS = ["down", "phase", "dist", "fp", "score", "time", "oto", "dto", "off_sim", "def_sim", "home", "rdist", "rfp", "row", "n", "qfp", "qdist",
        "a_ess", "a_rpnum", "a_rpsh", "a_runsh", "a_wfp",
        "n_ess", "n_rpnum", "n_rpsh", "n_runsh", "n_wfp",
        "u_rpnum", "u_rpsh", "u_runsh", "u_fp", "pen_sh", "pen_y", "tag"]
NC = len(COLS)


def install_log():
    import sim09_f3 as f3

    ns = dv._G["ns"]
    t = dv._G["tables"]
    a = t["arrays"]
    top = ns["pick_index_nn_conditioned"]
    holder = None
    cur = top
    while cur.__closure__ and "base" in cur.__code__.co_freevars:
        holder = cur.__closure__[cur.__code__.co_freevars.index("base")]
        cur = holder.cell_contents
    orig = cur
    yards = np.asarray(a["yards_gained"], dtype=np.float64)
    code = np.asarray(a["play_type_code"])
    fpr = np.asarray(a["fp_raw"], dtype=np.float64)
    rp = np.isin(code, (0, 1))
    run = code == 0
    pool = f3.pool_arrays()
    pen = pool.pen.to_numpy().astype(bool) & np.isin(code, (0, 1, 6))
    lo, ld = float(ns["TILT_LO"]), float(ns["TILT_LD"])
    first = {}
    buf = []
    path = Path(os.environ["PICK_OUT"]) / f"pick_{os.getpid()}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "ab")
    dummy = np.random.default_rng(0)

    def stats(w, nb):
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            return [np.nan] * 5
        p = w / s
        r = rp[nb]
        sr = float((p * r).sum())
        return [s * s / float((w * w).sum()), float((p * yards[nb] * r).sum()), sr, float((p * run[nb]).sum()), float((p * r * fpr[nb]).sum() / sr) if sr > 0 else np.nan]

    def weights(key, off, dfn, home):
        cdf, _ = t["nn_weight_cache_cond"][(key, off, dfn, home)]
        return np.diff(cdf, prepend=0.0)

    def logged(rng, tables, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim):
        dk = down if down in (1, 2, 3, 4) else 4
        key = ns["round_state_key"](dk, phase, dist, fp, score, time_raw, off_to, def_to)
        fresh = key not in t["nn_cache_cond"]
        row = orig(rng, tables, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, off_sim, def_sim, is_home_sim)
        nb = t["nn_cache_cond"][key]
        if fresh:
            first[key] = (fp, dist)
        orig(dummy, tables, down, phase, dist, fp, score, time_raw, off_to, def_to, k_state, lo, ld, is_home_sim)
        wa = weights(key, off_sim, def_sim, is_home_sim)
        wn = weights(key, lo, ld, is_home_sim)
        r = rp[nb]
        qf = first[key]
        ur = float(r.mean())
        buf.append([dk, phase, dist, fp, score, time_raw, off_to, def_to, off_sim, def_sim, is_home_sim, key[2], key[3], row, len(nb), qf[0], qf[1],
                    *stats(wa, nb), *stats(wn, nb),
                    float((yards[nb] * r).sum() / r.sum()) if r.sum() else np.nan, ur, float(run[nb].mean()), float(fpr[nb][r].mean()) if r.sum() else np.nan,
                    float(pen[nb].mean()), float(yards[nb][pen[nb] & r].mean()) if (pen[nb] & r).any() else np.nan, 0.0])
        if len(buf) >= 4000:
            np.asarray(buf, dtype=np.float64).tofile(fh)
            fh.flush()
            buf.clear()
        return row

    if holder is None:
        ns["pick_index_nn_conditioned"] = logged
    else:
        holder.cell_contents = logged
    import atexit

    def flush():
        if buf:
            np.asarray(buf, dtype=np.float64).tofile(fh)
            fh.flush()

    atexit.register(flush)


def B(setting):
    H.H_budget(setting)
    install_log()


def S(setting):
    H.H_ss(setting)
    install_log()


def cmd_run(a):
    import mod25e_scorestate as ss

    os.environ["PICK_OUT"] = str(Path(a.out_dir).resolve())
    H.install()
    crG.register(a.off)
    dv.DV["crHp"] = dict(dv.DV["crG"])
    outd = ss.OUT / f"e5_crHp_s{a.seed}"
    outd.mkdir(parents=True, exist_ok=True)
    os.environ["BUD_OUT"] = str(outd)
    ss.DV["crHp"] = dv.DV["crHp"]
    ss.s_init = S
    a.variant = "crHp"
    a.scale = 1.0
    ss.cmd_e5(a)


def load(d):
    parts = [np.fromfile(f, dtype=np.float64).reshape(-1, NC) for f in sorted(glob.glob(str(Path(d) / "pick_*.bin")))]
    X = np.vstack(parts)
    return {c: X[:, i] for i, c in enumerate(COLS)}


def blockboot(num, den, rng, nb=400, blk=100):
    n = len(num)
    k = max(n // blk, 1)
    ids = np.arange(n) // blk
    ids = np.minimum(ids, k - 1)
    sn = np.bincount(ids, weights=num, minlength=k)
    sd = np.bincount(ids, weights=den, minlength=k)
    bt = []
    for _ in range(nb):
        j = rng.integers(0, k, k)
        bt.append(sn[j].sum() / sd[j].sum())
    return num.sum() / den.sum(), np.quantile(bt, [0.025, 0.975])


def cmd_analyze(a):
    import mod25e_draw as dr
    import sim04_engine as sim
    import sim09_f3 as f3

    L = load(a.out_dir)
    G, cfg = dr.build()
    t = G["tables"]
    ar = t["arrays"]
    yards = ar["yards_gained"].astype(np.float64)
    code = ar["play_type_code"]
    rp = np.isin(code, (0, 1))
    fpr = ar["fp_raw"].astype(np.float64)
    dsr = ar["dist_raw"].astype(np.float64)
    dn = ar["down_i"]
    pool = f3.pool_arrays()
    pen = pool.pen.to_numpy().astype(bool) & np.isin(code, (0, 1, 6))
    rng = np.random.default_rng(3)
    out = []
    P = out.append
    bands = [(0, 20), (20, 40), (40, 63), (63, 77), (77, 100)]
    P(f"snaps {len(L['row'])} cols {NC}")
    ry = yards[L["row"].astype(int)]
    rrp = rp[L["row"].astype(int)]
    rrun = code[L["row"].astype(int)] == 0
    for nm, sel, rsel in (("1st&10", (L["down"] == 1) & (np.abs(L["dist"] - 10) < 0.5), (dn == 1) & (np.abs(dsr - 10) < 0.5)), ("2nd", L["down"] == 2, dn == 2)):
        P(f"== {nm}")
        for lo, hi in bands + [(20, 100)]:
            s = sel & (L["fp"] >= lo) & (L["fp"] < hi) & rrp
            r = rsel & rp & (fpr >= lo) & (fpr < hi)
            if s.sum() < 50:
                continue
            ss = sel & (L["fp"] >= lo) & (L["fp"] < hi)
            runs = rrun[s].mean()
            e_run = (L["a_runsh"][ss] / np.where(L["a_rpsh"][ss] > 0, L["a_rpsh"][ss], np.nan)).mean()
            n_run = (L["n_runsh"][ss] / np.where(L["n_rpsh"][ss] > 0, L["n_rpsh"][ss], np.nan)).mean()
            real_run = (code[r] == 0).mean()
            real_rpshare = rp[rsel & (fpr >= lo) & (fpr < hi)].mean()
            simrp = rrp[ss].mean()
            real_y = yards[r].mean()
            sim_y, ci = blockboot(ry[s] * 1.0, np.ones(s.sum()), rng)
            ea, eci = blockboot(L["a_rpnum"][ss], L["a_rpsh"][ss], rng)
            en, nci = blockboot(L["n_rpnum"][ss], L["n_rpsh"][ss], rng)
            eu, uci = blockboot(L["u_rpnum"][ss] * L["u_rpsh"][ss], L["u_rpsh"][ss], rng)
            P(f"  fp[{lo},{hi}) n={s.sum()} runshare sim {runs:.3f} expAct {e_run:.3f} expNeut {n_run:.3f} real {real_run:.3f} | rpshare sim {simrp:.3f} real {real_rpshare:.3f} | real y {real_y:.3f} sim drawn {sim_y:.3f} [{ci[0]:.2f},{ci[1]:.2f}] expAct {ea:.3f} [{eci[0]:.2f},{eci[1]:.2f}] expNeut {en:.3f} [{nci[0]:.2f},{nci[1]:.2f}] expUnw {eu:.3f} | ess a {np.nanmean(L['a_ess'][ss]):.1f} n {np.nanmean(L['n_ess'][ss]):.1f} | wfp-fp act {np.nanmean(L['a_wfp'][ss] - L['fp'][ss]):+.2f} neut {np.nanmean(L['n_wfp'][ss] - L['fp'][ss]):+.2f} unw {np.nanmean(L['u_fp'][ss] - L['fp'][ss]):+.2f} | cell-first fp-fp {np.nanmean(L['qfp'][ss] - L['fp'][ss]):+.2f} | pen cand share {np.mean(L['pen_sh'][ss]):.3f}")
    P("== strengths")
    for nm, sel in (("1st&10", (L["down"] == 1) & (np.abs(L["dist"] - 10) < 0.5) & (L["fp"] >= 20)), ("2nd", (L["down"] == 2) & (L["fp"] >= 20))):
        so = L["off_sim"][sel] + L["def_sim"][sel]
        P(f"  {nm}: off sd {np.std(L['off_sim'][sel]):.3f} def sd {np.std(L['def_sim'][sel]):.3f} mean(off+def) {so.mean():+.4f}; act-neut expected delta {blockboot(L['a_rpnum'][sel] - L['n_rpnum'][sel] * L['a_rpsh'][sel] / L['n_rpsh'][sel], L['a_rpsh'][sel], rng)[0] * 1:+.4f} (rp-weighted num)")
        d = (L["a_rpnum"][sel] / L["a_rpsh"][sel] - L["n_rpnum"][sel] / L["n_rpsh"][sel])
        P(f"  {nm}: per-snap expected yards actual-minus-neutral mean {np.nanmean(d):+.4f}; frac ess<10 {np.mean(L['a_ess'][sel] < 10):.3f}")
    P("== penalty rows in real pool (rp rows)")
    for nm, rsel in (("1st&10", (dn == 1) & (np.abs(dsr - 10) < 0.5) & (fpr >= 20)), ("2nd", (dn == 2) & (fpr >= 20))):
        r = rsel & rp
        P(f"  {nm}: all {yards[r].mean():.3f} n {r.sum()}; non-pen {yards[r & ~pen].mean():.3f}; pen share {pen[r].mean():.3f}; pen mean y {yards[r & pen].mean():.3f}")
    P("== exact-state neighbours vs cache neighbours (unweighted rp yards), sample")
    ix = np.flatnonzero((L["down"] == 1) & (np.abs(L["dist"] - 10) < 0.5) & (L["fp"] >= 20))
    ix = rng.choice(ix, size=min(4000, len(ix)), replace=False)
    ex, ca, band = [], [], []
    for i in ix:
        entry = t["nn_trees_cond"].get((1, int(L["phase"][i]))) or t["nn_trees_cond"][(1, 0)]
        tree, sub = entry
        f = sim.feature_matrix(np.array([L["dist"][i]]), np.array([L["fp"][i]]), np.array([L["score"][i]]), np.array([L["time"][i]]), np.array([L["oto"][i]]), np.array([L["dto"][i]]), np.array([int(L["phase"][i])]))[0]
        _, ind = tree.query(f, k=min(sim.K_STATE, len(sub)))
        nb = sub[np.atleast_1d(ind)]
        r = rp[nb]
        ex.append(yards[nb][r].mean())
        ca.append(L["u_rpnum"][i])
        band.append(L["fp"][i])
    ex, ca, band = np.array(ex), np.array(ca), np.array(band)
    for lo, hi in bands[1:] + [(20, 100)]:
        s = (band >= lo) & (band < hi)
        P(f"  fp[{lo},{hi}) n={s.sum()} exact {ex[s].mean():.3f} cache {np.nanmean(ca[s]):.3f} diff {np.nanmean(ca[s] - ex[s]):+.3f}")
    Path(a.log).write_text("\n".join(out))
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--off", default="")
    r.add_argument("--worlds", type=int, default=2)
    r.add_argument("--seasons", type=int, default=2)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--seed", type=int, default=41)
    r.add_argument("--out-dir", dest="out_dir", default=str(ART / "pick"))
    z = sub.add_parser("analyze")
    z.add_argument("--out-dir", dest="out_dir", default=str(ART / "pick"))
    z.add_argument("--log", default=str(ART / "pick.log"))
    a = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
