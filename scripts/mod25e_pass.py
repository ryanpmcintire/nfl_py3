import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25e_kstate as ks  # noqa: E402
import mod25e_revert_sim as rs  # noqa: E402

OUT = REPO / "artifacts" / "mod25e3" / "pass"
ZCAP = 6.0
NBOOT = 400
DOWN_EDGES = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
OLD_W = "weights = np.exp(-dist_sq / (2.0 * h * h))"


def zfun(wp):
    w = np.clip(wp, 1e-6, 1 - 1e-6)
    return np.minimum(np.abs(np.log(w / (1 - w))), ZCAP) / ZCAP


def wp_model(pool_seasons=rs.POOL):
    import joblib

    path = OUT / "wp_real.pkl"
    if path.exists():
        return joblib.load(path)
    pool = rs.real_frame(pool_seasons)
    pool = rs.add_outcome(pool[pool.season_type == "REG"])
    model = rs.fit_wp(pool)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def prep_real_d(seasons, model):
    pb = rs.real_frame(seasons)
    pb = pb[(pb.season_type == "REG") & (pb.qb_kneel != 1) & (pb.qb_spike != 1)]
    pb = pb[pb.play_type.isin(["pass", "run"]) & pb.epa.notna() & pb.wp.notna() & pb.posteam.notna() & pb.down.notna() & pb.yardline_100.notna()]
    X = np.column_stack([pb.score_differential, pb.game_seconds_remaining, pb.down, pb.ydstogo, pb.yardline_100])
    pb["wpf"] = model.predict_proba(X)[:, 1]
    pb["offhome"] = (pb.posteam == pb.home_team).astype(int)
    return pb.rename(columns={"posteam": "off", "defteam": "dfn"})[["season", "game_id", "off", "dfn", "epa", "wp", "wpf", "qtr", "offhome", "down"]]


def gstats(h):
    gid, gi = np.unique(h.game_id.values, return_inverse=True)
    z = zfun(h.wpf.values)
    s = h.s.values
    y = h.y.values
    X = np.column_stack([np.ones(len(h)), z, s, s * z])
    g = len(gid)
    sxx = np.zeros((g, 4, 4))
    sxy = np.zeros((g, 4))
    for a in range(4):
        sxy[:, a] = np.bincount(gi, X[:, a] * y, g)
        for b in range(a, 4):
            v = np.bincount(gi, X[:, a] * X[:, b], g)
            sxx[:, a, b] = v
            sxx[:, b, a] = v
    return sxx, sxy


def solve(sxx, sxy, w):
    return np.linalg.solve(np.tensordot(w, sxx, 1), w @ sxy)[2:]


def fit_slope(df):
    h = df[df.qtr.isin([3, 4])]
    sxx, sxy = gstats(h)
    rng = np.random.default_rng(0)
    g = len(sxx)
    base = solve(sxx, sxy, np.ones(g))
    boots = np.array([solve(sxx, sxy, rng.multinomial(g, np.ones(g) / g).astype(float)) for _ in range(NBOOT)])
    return base, boots


def tab(df, col, edges):
    base, bs, nb = rs.table(df, col, edges)
    return {"pass": [float(x) for x in base["pass"]], "lo": [float(np.percentile([b["pass"][i] for b in bs], 2.5)) for i in range(nb)], "hi": [float(np.percentile([b["pass"][i] for b in bs], 97.5)) for i in range(nb)], "n": [int(x) for x in base["n"]]}


def summarize(name, df, edges):
    base, boots = fit_slope(df)
    res = {"b": [float(x) for x in base], "b_lo": [float(np.percentile(boots[:, k], 2.5)) for k in range(2)], "b_hi": [float(np.percentile(boots[:, k], 97.5)) for k in range(2)], "pp_b1_pos": float(np.mean(boots[:, 1] > 0)), "boot": boots.tolist()}
    res["bins"] = tab(df, "wpf", edges)
    res["down"] = tab(df, "down", DOWN_EDGES)
    lines = [f"-- {name}: slope(z) = b0 + b1 z, z = min(|logit wp|, {ZCAP:g})/{ZCAP:g}, Q3/Q4, game bootstrap {NBOOT}"]
    lines.append(f"b0 {res['b'][0]:+.3f} [{res['b_lo'][0]:+.3f},{res['b_hi'][0]:+.3f}]  b1 {res['b'][1]:+.3f} [{res['b_lo'][1]:+.3f},{res['b_hi'][1]:+.3f}] probability_positive(b1) {res['pp_b1_pos']:.2f}")
    for i in range(len(res["bins"]["pass"])):
        t = res["bins"]
        lines.append(f"bin {i} wp {edges[i]:.3f}-{edges[i + 1]:.3f} n {t['n'][i]} pass {t['pass'][i]:+.3f} [{t['lo'][i]:+.3f},{t['hi'][i]:+.3f}]")
    for i in range(4):
        t = res["down"]
        lines.append(f"down {i + 1} n {t['n'][i]} pass {t['pass'][i]:+.3f} [{t['lo'][i]:+.3f},{t['hi'][i]:+.3f}]")
    return res, lines


def edges_pool(rp):
    h = rp[rp.qtr.isin([3, 4])]
    e = np.unique(np.quantile(h.wpf, np.linspace(0, 1, rs.NBINS + 1)))
    e[0], e[-1] = 0.0, 1.0001
    return e


def cmd_real(args):
    OUT.mkdir(parents=True, exist_ok=True)
    model = wp_model()
    rp = rs.strengths(prep_real_d(rs.POOL, model))
    rl = rs.strengths(prep_real_d(rs.LATE, model))
    edges = edges_pool(rp)
    out = {"edges": edges.tolist()}
    lines = [f"edges {np.round(edges, 3).tolist()}"]
    for name, df in (("pool2009-17", rp), ("real2016-25", rl)):
        res, ln = summarize(name, df, edges)
        out[name] = res
        lines += ln
    (OUT / "real.json").write_text(json.dumps(out))
    (OUT / "real.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def cmd_an(args):
    model = wp_model()
    real = json.loads((OUT / "real.json").read_text())
    edges = np.array(real["edges"])
    sm = rs.strengths(rs.prep_sim(Path(args.sim_dir), model))
    res, ln = summarize(Path(args.sim_dir).name, sm, edges)
    lines = list(ln)
    for ref in ("pool2009-17", "real2016-25"):
        r = real[ref]
        lines.append(f"-- sim minus {ref}")
        lines.append(f"d_b0 {res['b'][0] - r['b'][0]:+.3f} d_b1 {res['b'][1] - r['b'][1]:+.3f}")
        for i in range(len(edges) - 1):
            lines.append(f"bin {i} d_pass {res['bins']['pass'][i] - r['bins']['pass'][i]:+.3f}")
        for i in range(4):
            lines.append(f"down {i + 1} d_pass {res['down']['pass'][i] - r['down']['pass'][i]:+.3f}")
    (Path(args.sim_dir) / "pass_an.json").write_text(json.dumps(res))
    (Path(args.sim_dir) / "pass_an.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def cmd_fit(args):
    real = json.loads((OUT / "real.json").read_text())
    ref = real[args.ref]
    load = lambda p: np.array(json.loads((Path(p) / "pass_an.json").read_text())["b"])  # noqa: E731
    base, pa, pb = load(args.base), load(args.probe_a), load(args.probe_b)
    r = np.column_stack([(pa - base) / args.da, (pb - base) / args.db])
    target = np.array(ref["b"]) - base
    theta = np.linalg.solve(r, target)
    boots = np.array(ref["boot"])
    tb = np.array([np.linalg.solve(r, b - base) for b in boots])
    m0, m1 = 1.0 + theta[0], theta[1]
    out = {"m0": float(m0), "m1": float(m1), "ref": args.ref, "response": r.tolist(), "base": base.tolist(), "target": ref["b"], "m0_lo": float(1 + np.percentile(tb[:, 0], 2.5)), "m0_hi": float(1 + np.percentile(tb[:, 0], 97.5)), "m1_lo": float(np.percentile(tb[:, 1], 2.5)), "m1_hi": float(np.percentile(tb[:, 1], 97.5))}
    (OUT / "pass.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("m at z=0, 1:", m0, m0 + m1)


def install_pass(dv, pm):
    import joblib

    model = joblib.load(ks.OUT / "wp_model.pkl")
    ns = dv._G["ns"]
    cache = {}

    def mval(key):
        v = cache.get(key)
        if v is None:
            dn, _, rd, rf, rsc, rt = key[:6]
            x = ks.wp_features([rsc * ns["ROUND_SCORE"]], [rt * ns["ROUND_TIME"]], [dn], [rd * ns["ROUND_DIST"]], [rf * ns["ROUND_FP"]])
            wp = float(model.predict_proba(x)[0, 1])
            v = max(pm[0] + pm[1] * float(zfun(wp)), 0.0)
            cache[key] = v
        return v

    def pass_w(key, neighbors, off_sim, def_sim):
        m = mval(key)
        if m == 1.0:
            return 1.0
        th = ns["TILT_T"][neighbors] @ (ns["TILT_AO"] * (off_sim - ns["TILT_LO"]) + ns["TILT_AD"] * (def_sim - ns["TILT_LD"]))
        return np.exp((m - 1.0) * th)

    ns["PASS_W"] = pass_w
    dv._G["tables"]["nn_weight_cache_cond"].clear()


def install_epa_tilt(dv, pm):
    ns = dv._G["ns"]
    epa = np.asarray(dv._G["tables"]["attrs"]["epa"], dtype=np.float64)
    epa = epa - float(epa.mean())
    m = float(pm[0])

    def pass_w(key, neighbors, off_sim, def_sim):
        if m == 0.0:
            return 1.0
        s = (off_sim - ns["TILT_LO"]) + (def_sim - ns["TILT_LD"])
        return np.exp(m * s * epa[neighbors])

    ns["PASS_W"] = pass_w
    dv._G["tables"]["nn_weight_cache_cond"].clear()


def run_init(base_init, setting):
    import mod25d_variance as dv

    orig = inspect.getsource

    def gs(o):
        s = orig(o)
        if o is dv.sim.pick_index_nn_conditioned:
            assert OLD_W in s
            s = s.replace(OLD_W, OLD_W + " * PASS_W(key, neighbors, off_sim, def_sim)")
        return s

    inspect.getsource = gs
    try:
        base_init(setting)
    finally:
        inspect.getsource = orig
    if os.environ.get("PASS_MODE") == "epa":
        install_epa_tilt(dv, dv._G["cfg"]["pm"])
    else:
        install_pass(dv, dv._G["cfg"]["pm"])


def pass_init_budget(setting):
    import sim09_hk as hk

    run_init(hk.hk_init_budget, setting)


def pass_init_ss(setting):
    import sim09_hk as hk

    run_init(hk.hk_init_ss, setting)


def setup_variant(name, pm):
    import mod25d_variance as dv
    import mod25e_cov as cv
    import sim09_hk  # noqa: F401

    cv.patch_generator()
    dv.DV["crzhc"] = dict(dv.DV["crzhk"])
    dv.DV[name] = dict(dv.DV["crzhk"], pm=list(pm))
    return dv


def get_pm(args):
    if args.null:
        return [1.0, 0.0]
    if args.m0 is not None:
        return [args.m0, args.m1]
    f = json.loads((OUT / "pass.json").read_text())
    return [f["m0"], f["m1"]]


def cmd_sim(args):
    import mod25_generator as gen
    import mod25e_budget as bud
    import sim09_u3a as u3

    dv = setup_variant(args.variant, get_pm(args))
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bud.OUT = out
    os.environ["BUD_OUT"] = str(out)
    dv.c25.ensure_policies()
    v = dv.DV[args.variant]
    cfgj = json.dumps(dict(v, seed=args.seed, name=f"{args.variant}_s{v['scale']:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": v["scale"], "yard_gain": v["yard_gain"], "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = pass_init_budget
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

    setup_variant(args.variant, get_pm(args))
    ss.s_init = pass_init_ss
    args.scale = 1.0
    ss.cmd_e5(args)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("real")
    a = sub.add_parser("an")
    a.add_argument("--sim-dir", dest="sim_dir", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--base", required=True)
    f.add_argument("--probe-a", dest="probe_a", required=True)
    f.add_argument("--probe-b", dest="probe_b", required=True)
    f.add_argument("--da", type=float, required=True)
    f.add_argument("--db", type=float, required=True)
    f.add_argument("--ref", default="pool2009-17")
    for nm in ("sim", "e5"):
        s = sub.add_parser(nm)
        s.add_argument("--variant", default="crzhp")
        s.add_argument("--null", action="store_true")
        s.add_argument("--m0", type=float, default=None)
        s.add_argument("--m1", type=float, default=0.0)
        s.add_argument("--worlds", type=int, default=1 if nm == "sim" else 8)
        s.add_argument("--seasons", type=int, default=1 if nm == "sim" else 8)
        s.add_argument("--workers", type=int, default=1 if nm == "sim" else 3)
        s.add_argument("--seed", type=int, default=31 if nm == "sim" else 11)
        if nm == "sim":
            s.add_argument("--out-dir", dest="out_dir", required=True)
    args = ap.parse_args()
    {"real": cmd_real, "an": cmd_an, "fit": cmd_fit, "sim": cmd_sim, "e5": cmd_e5}[args.cmd](args)


if __name__ == "__main__":
    main()
