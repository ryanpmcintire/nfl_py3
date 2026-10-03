import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "artifacts" / "mod25e3" / "revert_sim"
PBP = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
POOL = tuple(range(2009, 2018))
LATE = tuple(range(2016, 2026))
BURN = 2
NBOOT = 400
NBINS = 10
WPF = ["sd", "gsr", "down", "dist", "yl"]


def cmd_sim(args):
    import mod25_generator as gen
    import mod25d_variance as dv
    import mod25e_budget as bud
    import mod25e_cov as cv
    import sim09_hk as hk
    import sim09_u3a as u3

    cv.patch_generator()
    dv.DV["crzhc"] = dict(dv.DV["crzhk"])
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bud.OUT = out
    os.environ["BUD_OUT"] = str(out)
    dv.c25.ensure_policies()
    v = dv.DV["crzhc"]
    cfgj = json.dumps(dict(v, seed=args.seed, name=f"crzhc_s{v['scale']:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": v["scale"], "yard_gain": v["yard_gain"], "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = hk.hk_init_budget
    gen.play_season = u3.s_play_season
    games, plays, latents, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(out / "sim_games.parquet")
    arrs = {}
    for (w, s), (weekly, qb_out) in latents.items():
        arrs[f"w_{w}_{s}"] = weekly
        arrs[f"q_{w}_{s}"] = qb_out
    np.savez(out / "latents.npz", **arrs)
    print("done", len(games), el, flush=True)


def real_frame(seasons):
    fr = []
    cols = ["game_id", "week", "posteam", "defteam", "epa", "wp", "qtr", "game_seconds_remaining", "score_differential", "down", "ydstogo", "yardline_100", "play_type", "qb_kneel", "qb_spike", "season_type", "home_team", "away_team"]
    for s in seasons:
        pb = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=cols)
        pb["season"] = s
        fr.append(pb)
    return pd.concat(fr, ignore_index=True)


def add_outcome(pb):
    gf = pd.read_parquet(REPO / "data" / "processed" / "game_features_pbp.parquet", columns=["game_id", "home_score", "away_score"])
    fin = gf.set_index("game_id")
    fin["hw"] = np.sign(fin.home_score - fin.away_score)
    pb = pb.merge(fin[["hw"]], left_on="game_id", right_index=True)
    ph = (pb.posteam == pb.home_team).astype(float)
    pb["won"] = np.where(pb.hw == 0, np.nan, np.where(ph == 1, pb.hw > 0, pb.hw < 0).astype(float))
    return pb


def fit_wp(pb):
    from sklearn.ensemble import HistGradientBoostingClassifier

    t = pb[pb.won.notna() & pb.posteam.notna() & pb.down.notna() & pb.score_differential.notna() & pb.yardline_100.notna()]
    X = np.column_stack([t.score_differential, t.game_seconds_remaining, t.down, t.ydstogo, t.yardline_100])
    m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    m.fit(X, t.won.astype(int).values)
    return m


def prep_real(seasons, model):
    pb = real_frame(seasons)
    pb = pb[(pb.season_type == "REG") & (pb.qb_kneel != 1) & (pb.qb_spike != 1)]
    pb = pb[pb.play_type.isin(["pass", "run"]) & pb.epa.notna() & pb.wp.notna() & pb.posteam.notna() & pb.down.notna() & pb.yardline_100.notna()]
    X = np.column_stack([pb.score_differential, pb.game_seconds_remaining, pb.down, pb.ydstogo, pb.yardline_100])
    pb["wpf"] = model.predict_proba(X)[:, 1]
    pb["offhome"] = (pb.posteam == pb.home_team).astype(int)
    return pb.rename(columns={"posteam": "off", "defteam": "dfn"})[["season", "game_id", "off", "dfn", "epa", "wp", "wpf", "qtr", "offhome"]]


def prep_sim(path, model):
    fs = sorted(path.glob("play_*_*.parquet"))
    fr = []
    for f in fs:
        w, s = (int(x) for x in f.stem.split("_")[1:])
        if s < BURN:
            continue
        d = pd.read_parquet(f)
        d = d[d.epa.notna()].copy()
        d["season"] = w * 1000 + s
        d["w"], d["s"] = w, s
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d["game_id"] = d.season.astype(str) + "_" + d.g.astype(int).astype(str)
    d["off"] = np.where(d.offhome == 1, d.ht, d["at"]).astype(int)
    d["dfn"] = np.where(d.offhome == 1, d["at"], d.ht).astype(int)
    X = np.column_stack([d.sd, d.gsr, d.down, d.dist, d.yl])
    d["wpf"] = model.predict_proba(X)[:, 1]
    d["wp"] = d["wpf"]
    lat = np.load(path / "latents.npz")
    lo = np.zeros(len(d))
    ld = np.zeros(len(d))
    for (w, s), idx in d.groupby(["w", "s"]).indices.items():
        wk, qb = lat[f"w_{w}_{s}"], lat[f"q_{w}_{s}"]
        wi = d.week.values[idx].astype(int) - 1
        lo[idx] = wk[wi, d.off.values[idx], 0]
        ld[idx] = wk[wi, d.dfn.values[idx], 1]
    d["lat_off"], d["lat_def"] = lo, ld
    d["qtr"] = d.qtr.astype(int)
    return d


def strengths(df):
    lg = df.epa.mean()
    for side, key in [("oe", "off"), ("de", "dfn")]:
        grp = df.groupby(["season", key]).epa
        tot, n = grp.transform("sum"), grp.transform("count")
        ge = df.groupby(["season", key, "game_id"]).epa
        gs, gn = ge.transform("sum"), ge.transform("count")
        df[side] = (tot - gs) / (n - gn) - lg
    df["s"] = df.oe + df.de
    df["r"] = df.epa - lg - df.s
    df["y"] = df.epa - lg
    return df


def table(df, col, edges, extra=()):
    h = df[df.qtr.isin([3, 4])].copy()
    h["bin"] = np.clip(np.searchsorted(edges, h[col].values, side="right") - 1, 0, len(edges) - 2)
    gid, gi = np.unique(h.game_id.values, return_inverse=True)
    nb = len(edges) - 1
    cols = ["r", "y", "s"] + list(extra)
    S = {c: np.zeros((len(gid), nb)) for c in cols + ["ys", "ss"]}
    N = np.zeros((len(gid), nb))
    b = h.bin.values
    np.add.at(N, (gi, b), 1.0)
    for c in cols:
        np.add.at(S[c], (gi, b), h[c].values)
    np.add.at(S["ys"], (gi, b), h.y.values * h.s.values)
    np.add.at(S["ss"], (gi, b), h.s.values**2)
    rng = np.random.default_rng(0)

    def stat(w):
        n = w @ N
        m = {c: (w @ S[c]) / n for c in cols}
        cov = (w @ S["ys"]) / n - m["y"] * m["s"]
        var = (w @ S["ss"]) / n - m["s"] ** 2
        m["pass"] = cov / var
        m["n"] = n
        return m

    base = stat(np.ones(len(gid)))
    bs = [stat(rng.multinomial(len(gid), np.ones(len(gid)) / len(gid)).astype(float)) for _ in range(NBOOT)]
    return base, bs, nb


def fmt(base, bs, k, i):
    v = np.array([x[k][i] for x in bs])
    return f"{base[k][i]:+.4f} [{np.percentile(v, 2.5):+.4f},{np.percentile(v, 97.5):+.4f}]"


def cmd_an(args):
    OUT.mkdir(parents=True, exist_ok=True)
    pool = real_frame(POOL)
    pool = add_outcome(pool[pool.season_type == "REG"])
    model = fit_wp(pool)
    rp = strengths(prep_real(POOL, model))
    rl = strengths(prep_real(LATE, model))
    sm = strengths(prep_sim(Path(args.sim_dir), model))
    h = rp[rp.qtr.isin([3, 4])]
    edges = np.unique(np.quantile(h.wpf, np.linspace(0, 1, NBINS + 1)))
    edges[0], edges[-1] = 0.0, 1.0001
    lines = [f"fitted-wp bin edges (real pool Q3/Q4 deciles): {np.round(edges, 3).tolist()}", f"plays Q3/Q4: pool {int(rp.qtr.isin([3, 4]).sum())} late {int(rl.qtr.isin([3, 4]).sum())} sim {int(sm.qtr.isin([3, 4]).sum())} games sim {sm.game_id.nunique()}"]
    sm["rl"] = sm.epa - sm.epa.mean() - sm.lat_off - sm.lat_def
    sm["sl"] = sm.lat_off + sm.lat_def
    res = {}
    for name, df, ex in [("pool2009-17", rp, ()), ("real2016-25", rl, ()), ("sim_crzhc", sm, ("rl", "sl"))]:
        res[name] = table(df, "wpf", edges, ex)
    for name, (base, bs, nb) in res.items():
        lines.append(f"-- {name} (residual EPA vs leave-game-out strength; pass = slope of EPA on strength)")
        for i in range(nb):
            extra = ""
            if name == "sim_crzhc":
                extra = f" resid_vs_latent {fmt(base, bs, 'rl', i)} latent_mean {base['sl'][i]:+.4f}"
            lines.append(f"bin {i} wp {edges[i]:.3f}-{edges[i + 1]:.3f} n {int(base['n'][i])} r {fmt(base, bs, 'r', i)} pass {fmt(base, bs, 'pass', i)} s_mean {base['s'][i]:+.4f}{extra}")
    lines.append("-- sim minus real pool and sim minus real 2016-25 (independent bootstraps), residual vs LGO strength; probability_positive of difference")
    sb, sbb, nb = res["sim_crzhc"]
    for other in ("pool2009-17", "real2016-25"):
        ob, obb, _ = res[other]
        for i in range(nb):
            d = np.array([x["r"][i] for x in sbb]) - np.array([x["r"][i] for x in obb])
            dp = np.array([x["pass"][i] for x in sbb]) - np.array([x["pass"][i] for x in obb])
            lines.append(f"{other} bin {i} d_r {sb['r'][i] - ob['r'][i]:+.4f} [{np.percentile(d, 2.5):+.4f},{np.percentile(d, 97.5):+.4f}] pp {np.mean(d > 0):.2f}  d_pass {sb['pass'][i] - ob['pass'][i]:+.3f} pp {np.mean(dp > 0):.2f}")
    for name, df in [("pool2009-17", rp), ("real2016-25", rl)]:
        e2 = np.unique(np.quantile(df[df.qtr.isin([3, 4])].wp, np.linspace(0, 1, NBINS + 1)))
        e2[0], e2[-1] = 0.0, 1.0001
        b, bsb, n2 = table(df, "wp", e2)
        lines.append(f"-- {name} with nflfastR wp deciles {np.round(e2, 3).tolist()}")
        for i in range(n2):
            lines.append(f"bin {i} n {int(b['n'][i])} r {fmt(b, bsb, 'r', i)} pass {fmt(b, bsb, 'pass', i)}")
    (OUT / "revert_sim.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def cmd_drawn(args):
    OUT.mkdir(parents=True, exist_ok=True)
    pool = real_frame(POOL)
    pool = add_outcome(pool[pool.season_type == "REG"])
    model = fit_wp(pool)
    rp = strengths(prep_real(POOL, model))
    gf = pd.read_parquet(REPO / "data" / "processed" / "game_features_pbp.parquet")
    gf = gf[gf.game_type == "REG"][["game_id", "home_off_epa_per_play", "away_off_epa_per_play", "home_def_epa_per_play", "away_def_epa_per_play"]]
    d = rp.merge(gf, on="game_id", how="inner")
    d["po"] = np.where(d.offhome == 1, d.home_off_epa_per_play, d.away_off_epa_per_play)
    d["pdf"] = np.where(d.offhome == 1, d.away_def_epa_per_play, d.home_def_epa_per_play)
    d = d[d.po.notna() & d.pdf.notna()].copy()
    d["po"] -= d.po.mean()
    d["pdf"] -= d.pdf.mean()
    d["sp"] = d.po + d.pdf
    h = d[d.qtr.isin([3, 4])]
    edges = np.unique(np.quantile(h.wpf, np.linspace(0, 1, NBINS + 1)))
    edges[0], edges[-1] = 0.0, 1.0001
    lg = d.epa.mean()
    d["rp"] = d.epa - lg - d.sp
    d["s_lgo"] = d.s
    lines = [f"pool rows with pregame ratings: {len(d)}, Q3/Q4 {len(h)}; edges {np.round(edges, 3).tolist()}"]
    lines.append("bin  n  r_vs_LGO  r_vs_pregame_rating  slope(EPA on pregame rating)  slope(EPA on LGO)  sd_pregame sd_LGO corr(pregame,LGO)")
    h = d[d.qtr.isin([3, 4])].copy()
    h["bin"] = np.clip(np.searchsorted(edges, h.wpf.values, side="right") - 1, 0, len(edges) - 2)
    gid, gi = np.unique(h.game_id.values, return_inverse=True)
    rng = np.random.default_rng(0)
    nb = len(edges) - 1
    cols = ["r", "rp"]
    S = {c: np.zeros((len(gid), nb)) for c in cols}
    N = np.zeros((len(gid), nb))
    np.add.at(N, (gi, h.bin.values), 1.0)
    for c in cols:
        np.add.at(S[c], (gi, h.bin.values), h[c].values)
    ws = [np.ones(len(gid))] + [rng.multinomial(len(gid), np.ones(len(gid)) / len(gid)).astype(float) for _ in range(NBOOT)]
    for i in range(nb):
        g = h[h.bin == i]
        yy = g.epa - lg
        sl_p = np.cov(yy, g.sp)[0, 1] / g.sp.var()
        sl_l = np.cov(yy, g.s)[0, 1] / g.s.var()
        cells = []
        for c in cols:
            v = np.array([(w @ S[c][:, i]) / (w @ N[:, i]) for w in ws])
            cells.append(f"{v[0]:+.4f} [{np.percentile(v[1:], 2.5):+.4f},{np.percentile(v[1:], 97.5):+.4f}]")
        lines.append(f"{i} wp {edges[i]:.3f}-{edges[i + 1]:.3f} n {len(g)} r_LGO {cells[0]} r_pregame {cells[1]} slope_pre {sl_p:.3f} slope_LGO {sl_l:.3f} sd_pre {g.sp.std():.4f} sd_LGO {g.s.std():.4f} corr {np.corrcoef(g.sp, g.s)[0, 1]:.3f}")
    (OUT / "drawn.txt").write_text(chr(10).join(lines))
    print(chr(10).join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--out-dir", dest="out_dir", default=str(OUT / "crzhc"))
    s.add_argument("--worlds", type=int, default=2)
    s.add_argument("--seasons", type=int, default=8)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=31)
    a = sub.add_parser("an")
    a.add_argument("--sim-dir", dest="sim_dir", default=str(OUT / "crzhc"))
    sub.add_parser("drawn")
    args = ap.parse_args()
    {"sim": cmd_sim, "an": cmd_an, "drawn": cmd_drawn}[args.cmd](args)


if __name__ == "__main__":
    main()
