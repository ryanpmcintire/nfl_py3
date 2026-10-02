import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "artifacts" / "mod25e3" / "r2gap"
BK = ((1, 4), (5, 9), (10, 18))
SEEDS = (11, 12, 13)


def season_design(d, teams):
    n = len(d)
    X = np.zeros((n, len(teams) + 1))
    X[:, 0] = 1.0
    X[np.arange(n), 1 + teams.get_indexer(d["home_team"])] = 1.0
    X[np.arange(n), 1 + teams.get_indexer(d["away_team"])] -= 1.0
    return X


def ridge_fit(X, y, lam):
    P = lam * np.eye(X.shape[1])
    P[0, 0] = 0.0
    Ai = np.linalg.inv(X.T @ X + P)
    return Ai, Ai @ X.T @ y


def loo_pred(X, y, lam):
    Ai, b = ridge_fit(X, y, lam)
    H = np.einsum("ij,jk,ik->i", X, Ai, X)
    return (X @ b - H * y) / (1 - H)


def re_pool(G):
    import mod25d_variance as dv

    se, sa = [], []
    for sid, d in G.groupby("season"):
        meta = pd.DataFrame({"season": d["season"].to_numpy(), "home_team": d["home_team"].to_numpy(), "away_team": d["away_team"].to_numpy()})
        M = pd.Series((d["home_score"] - d["away_score"]).to_numpy(float))
        r = dv.random_effects(meta, M, iters=400)
        se.append(r["noise_var"])
        sa.append(r["sigma_strength_var_per_team"])
    return float(np.mean(se)), float(np.mean(sa))


def per_season_stats(G, lam):
    rows, sh = [], []
    for sid, d in G.groupby("season"):
        d = d.reset_index(drop=True)
        teams = pd.Index(sorted(set(d["home_team"]) | set(d["away_team"])))
        X = season_design(d, teams)
        y = (d["home_score"] - d["away_score"]).to_numpy(float)
        rows.append(pd.DataFrame({"season": sid, "week": d["week"].to_numpy(), "y": y, "loo": loo_pred(X, y, lam)}))
        e = d["week"].to_numpy() <= 9
        bs = []
        for m in (e, ~e):
            dd = d[m].reset_index(drop=True)
            _, b = ridge_fit(season_design(dd, teams), y[m], lam)
            bs.append(b[1:])
        sh.append(np.column_stack(bs))
    return pd.concat(rows, ignore_index=True), sh


def bucket_stats(L):
    out = {}
    for lo, hi in BK:
        b = L[(L["week"] >= lo) & (L["week"] <= hi)]
        p = np.polyfit(b["loo"], b["y"], 1)
        out[f"{lo}_{hi}"] = {"n": len(b), "margin_var": float(b["y"].var()), "r2": float(np.corrcoef(b["loo"], b["y"])[0, 1] ** 2), "slope": float(p[0]), "resid_var": float((b["y"] - np.polyval(p, b["loo"])).var())}
    return out


def split_half_corr(sh):
    a = np.concatenate([s[:, 0] - s[:, 0].mean() for s in sh])
    b = np.concatenate([s[:, 1] - s[:, 1].mean() for s in sh])
    return {"corr": float(np.corrcoef(a, b)[0, 1]), "var_first": float(a.var()), "var_second": float(b.var())}


def late_r2(L, ks):
    d = pd.concat([L[(L["season"] == k) & (L["week"] >= 10)] for k in ks])
    return float(np.corrcoef(d["loo"], d["y"])[0, 1] ** 2)


def boot_ci(fn, items, rng, n=400):
    v = np.array([fn([items[i] for i in rng.integers(0, len(items), len(items))]) for _ in range(n)])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def sim_latents(seed, worlds, seasons):
    import mod25_generator as gen

    fit = gen.load_fit()
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": 1.0, "drift": 1.0})
    ws = np.random.SeedSequence(seed).spawn(worlds)
    lat = {}
    for w in range(worlds):
        rng = np.random.default_rng(ws[w])
        for sidx, (weekly, qb) in enumerate(gen.gen_world_latents(rng, seasons, fit, setting)):
            lat[(w, sidx)] = (weekly, qb)
    return lat, fit


def load_sim(seed):
    g = pd.read_parquet(REPO / "artifacts" / "mod25e3" / f"e5_crzhk_s{seed}" / "sim_games.parquet")
    g = g[(g["season"] % 1000) >= 3].copy()
    g["w"] = g["season"] // 1000
    g["sidx"] = g["season"] % 1000 - 1
    return g


def oracle(g, lat, shock):
    ht = g["home_team"].str.extract(r"W(\d+)T(\d+)").astype(int)[1].to_numpy()
    at = g["away_team"].str.extract(r"W(\d+)T(\d+)").astype(int)[1].to_numpy()
    xt, xm = [], []
    for i, (w, s, wk) in enumerate(zip(g["w"], g["sidx"], g["week"])):
        wkl, qb = lat[(int(w), int(s))]
        k = wk - 1

        def net(t):
            return wkl[k, t, 0] + (shock if qb[k, t] else 0.0) - wkl[k, t, 1]

        def netm(t):
            return wkl[:, t, 0].mean() - wkl[:, t, 1].mean()

        xt.append(net(ht[i]) - net(at[i]))
        xm.append(netm(ht[i]) - netm(at[i]))
    return g.assign(xt=xt, xm=xm)


def cmd_a(args):
    import mod25_generator as gen

    rng = np.random.default_rng(3)
    real = gen.real_games(tuple(range(2018, 2026)))
    res = {}
    se, sa = re_pool(real)
    L, sh = per_season_stats(real, se / sa)
    res["real"] = {"noiseRE": se, "strengthRE_team": sa, "buckets": bucket_stats(L), "split_half": split_half_corr(sh)}
    res["real"]["r2_late_ci"] = boot_ci(lambda ks: late_r2(L, ks), sorted(L["season"].unique()), rng)
    fit = gen.load_fit()
    shock = fit["qb"]["backup_off_epa_effect"]
    sims, fh, shh = [], [], []
    for seed in SEEDS:
        lat, _ = sim_latents(seed, args.worlds, args.seasons)
        g = load_sim(seed)
        g = g.assign(season=g["season"] + seed * 100000)
        sims.append(oracle(g, lat, shock))
        for (w, s), (wkl, qb) in lat.items():
            if s < 2:
                continue
            net = wkl[:, :, 0] - wkl[:, :, 1]
            fh.append(net[:9].mean(0) - net[:9].mean())
            shh.append(net[9:].mean(0) - net[9:].mean())
    S = pd.concat(sims, ignore_index=True)
    se_s, sa_s = re_pool(S)
    Ls, shs = per_season_stats(S, se_s / sa_s)
    res["sim"] = {"noiseRE": se_s, "strengthRE_team": sa_s, "buckets": bucket_stats(Ls), "split_half": split_half_corr(shs)}
    res["sim"]["r2_late_ci"] = boot_ci(lambda ks: late_r2(Ls, ks), sorted(Ls["season"].unique()), rng, 200)
    y = (S["home_score"] - S["away_score"]).to_numpy(float)
    orc = {}
    for lo, hi in BK:
        m = ((S["week"] >= lo) & (S["week"] <= hi)).to_numpy()
        row = {}
        for nm in ("xt", "xm"):
            x = S[nm].to_numpy()[m]
            p = np.polyfit(x, y[m], 1)
            row[nm] = {"r2": float(np.corrcoef(x, y[m])[0, 1] ** 2), "slope": float(p[0]), "resid_var": float((y[m] - np.polyval(p, x)).var()), "x_var": float(x.var())}
        orc[f"{lo}_{hi}"] = row
    res["sim_oracle"] = orc
    a, b = np.concatenate(fh), np.concatenate(shh)
    res["sim_latent_halves"] = {"corr_true": float(np.corrcoef(a, b)[0, 1]), "var_first": float(a.var()), "var_second": float(b.var()), "phi_off": fit["off"]["phi"], "phi_def": fit["def"]["phi"], "var_mu_off": fit["off"]["var_mu"], "var_mu_def": fit["def"]["var_mu"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


def cmd_simfeat(args):
    import mod25_generator as gen
    import mod25d_variance as dv
    import mod25e_scorestate as ss
    import sim09_hk as hk

    OUT.mkdir(parents=True, exist_ok=True)
    ss.ensure_css()
    dv.c25.ensure_policies()
    outd = OUT / "feat_sim"
    outd.mkdir(parents=True, exist_ok=True)
    cfgj = json.dumps(dict(dv.DV["crzhk"], seed=args.seed, name="crzhk_s1", outdir=str(outd)))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": 1.0, "yard_gain": 0.0, "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = hk.hk_init_ss
    gen.play_season = ss.s_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=False)
    games.to_parquet(OUT / "feat_games.parquet")
    plays.to_parquet(OUT / "feat_plays.parquet")
    print("done", len(games), el, flush=True)


def feat_track(games, plays, seasons):
    import mod25_generator as gen

    ts = gen.team_stats_from_plays(plays)
    from nfl_ats.features import build_team_game_metrics

    tg = build_team_game_metrics(gen.schedules_from_games(games), ts)
    tg = tg.assign(week=tg["game_id"].map(games.set_index("game_id")["week"]))
    tg = tg[tg["season"].isin(seasons)].sort_values(["team", "season", "week"])
    rows = []
    for (t, s), d in tg.groupby(["team", "season"]):
        v = d["off_epa_per_play"].to_numpy(float)
        wk = d["week"].to_numpy()
        n, tot = len(v), v.sum()
        for i in range(1, n):
            rows.append((s, wk[i], v[:i].mean(), (tot - v[i]) / (n - 1), v[i], i))
    R = pd.DataFrame(rows, columns=["season", "week", "roll", "loo", "v", "i"])
    res = {}
    for lo, hi in BK:
        b = R[(R["week"] >= lo) & (R["week"] <= hi)]
        dm = lambda c: b[c] - b.groupby("season")[c].transform("mean")
        res[f"{lo}_{hi}"] = {"n": len(b), "corr_roll_loo": float(np.corrcoef(dm("roll"), dm("loo"))[0, 1]), "corr_roll_next": float(np.corrcoef(dm("roll"), dm("v"))[0, 1]), "corr_loo_next": float(np.corrcoef(dm("loo"), dm("v"))[0, 1]), "roll_sd": float(dm("roll").std()), "loo_sd": float(dm("loo").std()), "next_sd": float(dm("v").std()), "games_in_roll": float(b["i"].mean())}
    return res


def cmd_feat(args):
    import mod25_generator as gen

    rg = gen.real_games(tuple(range(2018, 2026)))
    rp = gen.load_real_plays(range(2018, 2026))
    rg = rg[rg["game_id"].isin(set(rp["game_id"]))].reset_index(drop=True)
    res = {"real": feat_track(rg, rp, set(range(2018, 2026)))}
    g = pd.read_parquet(OUT / "feat_games.parquet")
    p = pd.read_parquet(OUT / "feat_plays.parquet")
    res["sim"] = feat_track(g, p, set(int(x) for x in g["season"].unique() if x % 1000 >= 3))
    (OUT / "feat.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


def link_track(games, plays, seasons):
    import mod25_generator as gen
    from nfl_ats.features import build_team_game_metrics

    ts = gen.team_stats_from_plays(plays)
    tg = build_team_game_metrics(gen.schedules_from_games(games), ts)
    tg = tg[tg["season"].isin(seasons)].copy()
    tg["net"] = tg["off_epa_per_play"] - tg["def_epa_per_play"]
    grp = tg.groupby(["team", "season"])["net"]
    tg["loo"] = (grp.transform("sum") - tg["net"]) / (grp.transform("size") - 1)
    key = tg.set_index(["game_id", "team"])
    g = games[games["season"].isin(seasons)].copy()
    ix_h = pd.MultiIndex.from_arrays([g["game_id"], g["home_team"]])
    ix_a = pd.MultiIndex.from_arrays([g["game_id"], g["away_team"]])
    g["xl"] = key["loo"].reindex(ix_h).to_numpy() - key["loo"].reindex(ix_a).to_numpy()
    g["xa"] = key["net"].reindex(ix_h).to_numpy() - key["net"].reindex(ix_a).to_numpy()
    g["y"] = g["home_score"] - g["away_score"]
    g = g.dropna(subset=["xl", "xa", "y"])
    res = {}
    for lo, hi in BK:
        b = g[(g["week"] >= lo) & (g["week"] <= hi)]
        pl = np.polyfit(b["xl"], b["y"], 1)
        pa = np.polyfit(b["xa"], b["y"], 1)
        res[f"{lo}_{hi}"] = {"n": len(b), "r2_loo_season_epa_vs_margin": float(np.corrcoef(b["xl"], b["y"])[0, 1] ** 2), "slope_loo": float(pl[0]), "r2_game_epa_vs_margin": float(np.corrcoef(b["xa"], b["y"])[0, 1] ** 2), "resid_var_game_epa": float((b["y"] - np.polyval(pa, b["xa"])).var()), "slope_game": float(pa[0]), "x_var_loo": float(b["xl"].var())}
    return res


def cmd_link(args):
    import mod25_generator as gen

    rg = gen.real_games(tuple(range(2018, 2026)))
    rp = gen.load_real_plays(range(2018, 2026))
    rg = rg[rg["game_id"].isin(set(rp["game_id"]))].reset_index(drop=True)
    res = {"real": link_track(rg, rp, set(range(2018, 2026)))}
    g = pd.read_parquet(OUT / "feat_games.parquet")
    p = pd.read_parquet(OUT / "feat_plays.parquet")
    res["sim"] = link_track(g, p, set(int(x) for x in g["season"].unique() if x % 1000 >= 3))
    (OUT / "link.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("a", cmd_a), ("simfeat", cmd_simfeat), ("feat", cmd_feat), ("link", cmd_link)):
        s = sub.add_parser(nm)
        s.add_argument("--worlds", type=int, default=8)
        s.add_argument("--seasons", type=int, default=8)
        s.add_argument("--workers", type=int, default=6)
        s.add_argument("--seed", type=int, default=11)
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
