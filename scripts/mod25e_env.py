import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "mod25e3" / "env"
SCHED = REPO / "data" / "raw" / "20260915T170343Z" / "schedules.parquet"
NBOOT = 400
FS = {
    "dome": ["dome"], "turf": ["turf"], "wind": ["wind_o"], "temp": ["temp_o"],
    "crew": ["crew"], "pace": ["pace"], "snap_load": ["plays_h", "plays_a"],
    "field_pos": ["yl_mean"], "market_total": ["total_line"],
}
FS["weather_surface"] = FS["dome"] + FS["turf"] + FS["wind"] + FS["temp"]
FS["all"] = sum((FS[k] for k in ("dome", "turf", "wind", "temp", "crew", "pace", "snap_load", "field_pos", "market_total")), [])


def load_xq():
    src = (REPO / "scripts" / "mod25e_xq.py").read_text().rsplit("\ncommon()", 1)[0]
    ns = {"__name__": "mod25e_xq_lib", "__file__": str(REPO / "scripts" / "mod25e_xq.py")}
    exec(compile(src, "mod25e_xq_lib", "exec"), ns)
    return ns


def game_table(D, xq):
    D, _ = xq["components"](D)
    D = D.assign(h=D.s > 0)
    g = D.groupby(["gk", "h"]).agg(r=("c_rest", "mean"), n=("c_rest", "size"), pl=("n", "sum")).unstack()
    g = g.dropna()
    G = pd.DataFrame({"H": g[("r", True)], "A": g[("r", False)], "nh": g[("n", True)], "na": g[("n", False)], "plays_h": g[("pl", True)], "plays_a": g[("pl", False)]})
    G["yl_mean"] = D.groupby("gk").yl0.mean().reindex(G.index)
    qm = D.groupby(["gk", "h", "ql"]).c_rest.mean().unstack()
    Hq = qm.xs(True, level="h").reindex(G.index)
    Aq = qm.xs(False, level="h").reindex(G.index)
    return G, Hq, Aq


def add_env(G):
    sch = pd.read_parquet(SCHED)
    sch = sch[sch.game_type == "REG"].set_index("game_id")
    gid = pd.Index([k.split("_", 1)[1] for k in G.index])
    S = sch.reindex(gid)
    out = G.copy()
    out["season"] = [int(k.split("_", 1)[0]) for k in G.index]
    dome = S.roof.isin(["dome", "closed"]).to_numpy(float)
    out["dome"] = dome
    out["turf"] = (~S.surface.isin(["grass", "dessograss"])).to_numpy(float)
    out["wind_o"] = np.where(dome == 0, S.wind.fillna(0).to_numpy(float), 0.0)
    out["temp_o"] = np.where(dome == 0, S.temp.fillna(S.temp.median()).to_numpy(float), 70.0)
    out["total_line"] = S.total_line.fillna(S.total_line.median()).to_numpy(float)
    out["ref"] = S.referee.fillna("unknown").to_numpy()
    out["absm"] = (S.home_score - S.away_score).abs().to_numpy(float)
    out["plays"] = out.plays_h + out.plays_a
    X = np.column_stack([np.ones(len(out)), out.absm.to_numpy()])
    b = np.linalg.lstsq(X, out.plays.to_numpy(float), rcond=None)[0]
    out["pace"] = out.plays.to_numpy(float) - X @ b
    out["m"] = (out.H + out.A) / 2
    return out


def crew_effect(G):
    y = G.m.to_numpy()
    ref = G.ref.to_numpy()
    ser = pd.Series(y)
    s = ser.groupby(ref).agg(["sum", "count"])
    within = float(((ser - ser.groupby(ref).transform("mean")) ** 2).sum() / max(len(y) - len(s), 1))
    total = float(np.var(y, ddof=1))
    between = max(total - within, 1e-9)
    lam = within / between
    mu = y.mean()
    tot = s["sum"].reindex(ref).to_numpy()
    cnt = s["count"].reindex(ref).to_numpy()
    eff = ((tot - y) + lam * mu) / ((cnt - 1) + lam) - mu
    return eff, dict(within=within, between_total=between, lam=float(lam), refs=int(len(s)), games_per_ref=float(s["count"].mean()))


def crossfit(G, cols, y):
    X = np.column_stack([np.ones(len(G))] + [G[c].to_numpy(float) for c in cols])
    yv = G[y].to_numpy(float)
    pred = np.zeros(len(G))
    for s in sorted(G.season.unique()):
        te = (G.season == s).to_numpy()
        beta = np.linalg.lstsq(X[~te], yv[~te], rcond=None)[0]
        pred[te] = X[te] @ beta
    return pred


def share_rows(G, rng):
    H, A = G.H.to_numpy(), G.A.to_numpy()
    tot = float(np.cov(H, A)[0, 1])
    idx = [rng.integers(0, len(G), len(G)) for _ in range(NBOOT)]
    bt = np.array([np.cov(H[i], A[i])[0, 1] for i in idx])
    rows = {"total_cov": dict(v=tot, lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5)))}
    allp = None
    for nm, cols in FS.items():
        ph, pa = crossfit(G, cols, "H"), crossfit(G, cols, "A")
        v = float(np.cov(ph, pa)[0, 1])
        bs = np.array([np.cov(ph[i], pa[i])[0, 1] / max(np.cov(H[i], A[i])[0, 1], 1e-12) for i in idx])
        rows[nm] = dict(v=v, share=v / tot, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), pp=float((bs > 0).mean()))
        if nm == "all":
            allp = (ph, pa)
    rH, rA = H - allp[0], A - allp[1]
    rem = float(np.cov(rH, rA)[0, 1])
    br = np.array([np.cov(rH[i], rA[i])[0, 1] for i in idx])
    rows["remainder_after_all"] = dict(v=rem, share=rem / tot, lo=float(np.percentile(br, 2.5)), hi=float(np.percentile(br, 97.5)), pp=float((br > 0).mean()))
    return rows


def quarter_matrix(Hq, Aq):
    M = np.full((4, 4), np.nan)
    for q in range(1, 5):
        for r in range(1, 5):
            a, b = Hq[q], Aq[r]
            ok = a.notna() & b.notna()
            M[q - 1, r - 1] = float(np.cov(a[ok], b[ok])[0, 1])
    return M


def fit_ar(M):
    best = None
    for phi in np.linspace(0, 1, 201):
        W = np.array([[phi ** abs(q - r) for r in range(4)] for q in range(4)])
        s2 = float((W * M).sum() / (W * W).sum())
        sse = float(((M - s2 * W) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, s2, float(phi))
    return dict(sigma2=best[1], phi=best[2], sse=best[0])


def run_one(D, xq, rng, env=True):
    G, Hq, Aq = game_table(D, xq)
    res = {"games": int(len(G)), "corr_HA": float(np.corrcoef(G.H, G.A)[0, 1]), "cov_HA": float(np.cov(G.H, G.A)[0, 1]), "var_H": float(G.H.var()), "var_A": float(G.A.var())}
    M = quarter_matrix(Hq, Aq)
    res["quarter_cov"] = M.tolist()
    res["same_quarter_mean"] = float(np.mean(np.diag(M)))
    res["cross_quarter_mean"] = float(np.mean(M[~np.eye(4, dtype=bool)]))
    res["ar_fit"] = fit_ar(M)
    qs = np.array([quarter_matrix(Hq.iloc[i], Aq.iloc[i]) for i in [rng.integers(0, len(G), len(G)) for _ in range(NBOOT)]])
    res["boot_phi"] = [fit_ar(m)["phi"] for m in qs[:100]]
    res["boot_sigma2"] = [fit_ar(m)["sigma2"] for m in qs[:100]]
    Hv, Av = G.H.to_numpy(), G.A.to_numpy()
    res["boot_cov"] = [float(np.cov(Hv[i], Av[i])[0, 1]) for i in [rng.integers(0, len(G), len(G)) for _ in range(NBOOT)]]
    if env:
        E = add_env(G)
        eff, info = crew_effect(E)
        E["crew"] = eff
        res["crew_info"] = info
        res["shares"] = share_rows(E, rng)
    return res


def cmd_an(a):
    xq = load_xq()
    rng = np.random.default_rng(7)
    res = {}
    for nm, seas in (("pool", xq["POOL"]), ("late", xq["LATE"])):
        D, _ = xq["real_drives"](seas)
        res[nm] = run_one(D, xq, rng)
    res["sim"] = run_one(xq["sim_drives"](), xq, rng, env=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "env.json").write_text(json.dumps(res))
    cmd_show(a)


def cmd_show(a):
    res = json.loads((OUT / "env.json").read_text())
    L = []
    for k, r in res.items():
        L.append(f"{k} games {r['games']} corr(H,A) {r['corr_HA']:+.3f} cov {r['cov_HA']:+.4f} boot90 [{np.percentile(r['boot_cov'], 5):+.4f},{np.percentile(r['boot_cov'], 95):+.4f}] var H {r['var_H']:.3f} A {r['var_A']:.3f}")
        L.append(f"  quarter cov same-q mean {r['same_quarter_mean']:+.4f} cross-q mean {r['cross_quarter_mean']:+.4f}; AR fit sigma2 {r['ar_fit']['sigma2']:+.4f} phi {r['ar_fit']['phi']:.2f} (boot phi 5-95 {np.percentile(r['boot_phi'], 5):.2f}-{np.percentile(r['boot_phi'], 95):.2f}, sigma2 {np.percentile(r['boot_sigma2'], 5):+.4f}..{np.percentile(r['boot_sigma2'], 95):+.4f})")
        L.append("  quarter cov matrix H_q x A_r: " + " | ".join(" ".join(f"{x:+.3f}" for x in row) for row in r["quarter_cov"]))
        if "shares" in r:
            L.append(f"  crew {r['crew_info']}")
            for nm, s in r["shares"].items():
                if nm == "total_cov":
                    L.append(f"  total cov {s['v']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}]")
                else:
                    L.append(f"  {nm:20s} explained cov {s['v']:+.5f} share {s['share']:+.3f} [{s['lo']:+.3f},{s['hi']:+.3f}] pp {s['pp']:.2f}")
    d = np.array(res["pool"]["boot_cov"][:400]) - np.array(res["sim"]["boot_cov"][:400])
    L.append(f"pool minus sim cov(H,A) {res['pool']['cov_HA'] - res['sim']['cov_HA']:+.4f} [{np.percentile(d, 2.5):+.4f},{np.percentile(d, 97.5):+.4f}] pp {(d > 0).mean():.2f}")
    (OUT / "env.txt").write_text("\n".join(L))
    print("\n".join(L))


def cmd_fit(a):
    d = REPO / "artifacts" / "mod25e3" / "revert_sim" / "crzhc"
    res = json.loads((OUT / "env.json").read_text())
    rows = []
    nd = []
    for f in sorted(d.glob("play_*_*.parquet")):
        w, s = (int(x) for x in f.stem.split("_")[1:])
        if s < xq_burn():
            continue
        p = pd.read_parquet(f, columns=["g", "week", "ht", "at", "offhome", "qtr", "epa"])
        p = p[(p.qtr <= 4) & p.epa.notna()]
        lat = np.load(d / "latents.npz")[f"w_{w}_{s}"]
        off = np.where(p.offhome == 1, p.ht, p["at"]).astype(int)
        wk = p.week.astype(int).to_numpy() - 1
        rows.append(pd.DataFrame({"epa": p.epa.to_numpy(), "lat": lat[wk, off, 0]}))
        g = p.g.to_numpy()
        oh = p.offhome.to_numpy()
        new = np.r_[True, (g[1:] != g[:-1]) | (oh[1:] != oh[:-1])]
        nd.append(len(p) / new.sum())
    R = pd.concat(rows)
    x = R.lat.to_numpy() - R.lat.mean()
    y = R.epa.to_numpy() - R.epa.mean()
    b = float((x * y).sum() / (x * x).sum())
    rng = np.random.default_rng(1)
    chunks = np.array_split(np.arange(len(R)), len(rows))
    bb = []
    for _ in range(200):
        pick = np.concatenate([chunks[i] for i in rng.integers(0, len(chunks), len(chunks))])
        xs, ys = x[pick], y[pick]
        bb.append(float((xs * ys).sum() / (xs * xs).sum()))
    nbar = float(np.mean(nd))
    cov_gap = res["pool"]["cov_HA"] - res["sim"]["cov_HA"]
    sig = float(np.sqrt(max(cov_gap, 0.0)))
    out = dict(b_epa_per_latent=b, b_lo=float(np.percentile(bb, 2.5)), b_hi=float(np.percentile(bb, 97.5)), plays_per_drive=nbar,
               cov_real_pool=res["pool"]["cov_HA"], cov_sim=res["sim"]["cov_HA"], cov_gap=cov_gap, drive_points_sd=sig,
               ar_phi=res["pool"]["ar_fit"]["phi"], envsd=sig / (b * nbar))
    (OUT / "env_fit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def xq_burn():
    return 2


def install_env():
    import mod25d_variance as dv

    cfg = dv._G["cfg"]
    sd = float(cfg.get("envsd", 0.0))
    seed = int(cfg.get("seed", 3))
    if sd == 0.0:
        return
    ns = dv._G["ns"]
    base_run = ns["run_one_game"]
    st = {"k": None, "rng": None}

    def run_one_game(*args, **kw):
        if st["k"] != dv._G.get("task_key"):
            st["k"] = dv._G.get("task_key")
            st["rng"] = dv.task_rng(104729, seed)
        shift = sd * float(st["rng"].standard_normal())
        hr, ar = args[-2], args[-1]
        hr2 = dict(hr, off=hr["off"] + shift)
        ar2 = dict(ar, off=ar["off"] + shift)
        return base_run(*args[:-2], hr2, ar2, **kw)

    ns["run_one_game"] = run_one_game


def env_init_ss(setting):
    import sim09_hk as hk

    hk.hk_init_ss(setting)
    install_env()


def env_init_budget(setting):
    import sim09_hk as hk

    hk.hk_init_budget(setting)
    install_env()


def env_variant(sd):
    import sim09_hk as hk

    hk.dv.DV["crzhe"] = dict(hk.dv.DV["crzhk"], envsd=sd)
    return hk


def cmd_sim(a):
    import mod25_generator as gen
    import mod25e_budget as bud
    import mod25e_cov as cv
    import sim09_u3a as u3

    sd = a.envsd if a.envsd is not None else json.loads((OUT / "env_fit.json").read_text())["envsd"]
    hk = env_variant(sd)
    cv.patch_generator()
    bud.OUT = Path(a.out_dir).resolve()
    import os

    os.environ["BUD_OUT"] = str(bud.OUT)
    bud.OUT.mkdir(parents=True, exist_ok=True)
    hk.dv.c25.ensure_policies()
    v = hk.dv.DV["crzhe"]
    cfgj = json.dumps(dict(v, seed=a.seed, name=f"crzhe_s{v['scale']:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": v["scale"], "yard_gain": v["yard_gain"], "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = env_init_budget
    gen.play_season = u3.s_play_season
    games, plays, latents, el = gen.run_generation(setting, a.worlds, a.seasons, a.workers, a.seed, progress=True)
    games.to_parquet(bud.OUT / "sim_games.parquet")
    np.savez(bud.OUT / "latents.npz", **{f"{k}_{w}_{s}": v2 for (w, s), (wk, qb) in latents.items() for k, v2 in (("w", wk), ("q", qb))})
    print("done", len(games), el, flush=True)


def cmd_e5(a):
    import mod25e_cov as cv
    import mod25e_scorestate as ss

    sd = a.envsd if a.envsd is not None else json.loads((OUT / "env_fit.json").read_text())["envsd"]
    env_variant(sd)
    cv.patch_generator()
    ss.s_init = env_init_ss
    a.variant = "crzhe"
    a.scale = 1.0
    ss.cmd_e5(a)


def cmd_xq(a):
    xq = load_xq()
    xq["SIM"] = Path(a.sim_dir)
    rng = np.random.default_rng(7)
    r = run_one(xq["sim_drives"](), xq, rng, env=False)
    res = json.loads((OUT / "env.json").read_text())
    res["sim_env"] = r
    (OUT / "env_sim.json").write_text(json.dumps(res))
    for k in ("pool", "late", "sim_env"):
        q = res[k]
        print(k, "corr", round(q["corr_HA"], 3), "cov", round(q["cov_HA"], 4), "same-q", round(q["same_quarter_mean"], 4), "cross-q", round(q["cross_quarter_mean"], 4))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("an")
    sub.add_parser("show")
    sub.add_parser("fit")
    s = sub.add_parser("sim")
    s.add_argument("--out-dir", dest="out_dir", required=True)
    s.add_argument("--worlds", type=int, default=1)
    s.add_argument("--seasons", type=int, default=1)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--seed", type=int, default=21)
    s.add_argument("--envsd", type=float, default=None)
    e = sub.add_parser("e5")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    e.add_argument("--envsd", type=float, default=None)
    x = sub.add_parser("xq")
    x.add_argument("--sim-dir", dest="sim_dir", required=True)
    a = ap.parse_args()
    {"an": cmd_an, "show": cmd_show, "fit": cmd_fit, "sim": cmd_sim, "e5": cmd_e5, "xq": cmd_xq}[a.cmd](a)


if __name__ == "__main__":
    main()
