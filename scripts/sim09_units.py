import json
import sys
import numpy as np
import pandas as pd

ROOT = "F:/Repos/nfl_py3/"
PBP = ROOT + "data/pbp/raw/20260929T191306Z/season={}/plays.parquet"
PART = ROOT + "data/players/participation/raw/20260813T131635Z/season={}/participation.parquet"
NGS = ROOT + "data/raw/ngs/20261001T173403Z/ngs_receiving.parquet"
OUT = ROOT + "artifacts/sim09/"
TABLE = ROOT + "data/processed/sim09_unit_ratings.parquet"
TMAP = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "WSH": "WAS", "JAC": "JAX"}
S0 = 2009
TRAIN_END = 2021
SIGN = {
    "cpoe": (1, -1), "int": (-1, 1), "sack_r": (-1, 1), "sack_net": (-1, 1), "press_net": (-1, 1),
    "qbrun": (1, -1), "man_epa": (1, -1), "zone_epa": (1, -1), "sep": (1, -1), "yac": (1, -1),
    "deep": (1, -1), "run_net": (1, -1), "run_light": (1, -1), "run_stack": (1, -1),
    "fg": (1, -1), "fg_long": (1, -1), "epa_all": (1, -1), "epa_pass": (1, -1), "epa_run": (1, -1),
}
FIRST = {"press_net": 2016, "sack_net": 2016, "run_light": 2016, "run_stack": 2016, "deep": 2016,
         "sep": 2016, "yac": 2016, "man_epa": 2018, "zone_epa": 2018}
GRID_HL = [9, 17, 34]
GRID_TAU = [0.5, 1.0, 2.0]
LOOKS = {"tuning": 0, "reliability": 0, "incremental": 0, "interaction": 0}


def tnorm(s):
    return s.map(lambda x: TMAP.get(x, x))


def load():
    fr = []
    for y in range(2009, 2026):
        p = pd.read_parquet(PBP.format(y))
        p = p[p.season_type == "REG"].copy()
        if y >= 2016:
            q = pd.read_parquet(PART.format(y), columns=["game_id", "play_id", "defenders_in_box", "number_of_pass_rushers", "time_to_throw", "was_pressure", "defense_man_zone_type", "ngs_air_yards"])
            q = q.drop_duplicates(["game_id", "play_id"])
            p = p.merge(q, on=["game_id", "play_id"], how="left")
        fr.append(p)
    d = pd.concat(fr, ignore_index=True)
    for c in ["defenders_in_box", "number_of_pass_rushers", "time_to_throw", "was_pressure", "ngs_air_yards"]:
        if c not in d:
            d[c] = np.nan
    if "defense_man_zone_type" not in d:
        d["defense_man_zone_type"] = None
    d["posteam"] = tnorm(d.posteam)
    d["defteam"] = tnorm(d.defteam)
    return d


def linfit(X, y):
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    return np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]


def logit_fit(X, y, it=30):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ b))
        w = p * (1 - p) + 1e-9
        b += np.linalg.solve(X.T @ (X * w[:, None]) + 1e-6 * np.eye(len(b)), X.T @ (y - p))
    return b


def metric_frames(d):
    d = d[d.posteam.notna() & d.defteam.notna()].copy()
    d["off"] = d.posteam
    d["def"] = d.defteam
    db = d[(d.qb_dropback == 1) & (d.qb_kneel == 0) & (d.qb_spike == 0) & d.epa.notna()].copy()
    ru = d[(d.rush_attempt == 1) & (d.qb_dropback == 0) & (d.qb_kneel == 0) & d.epa.notna()].copy()
    fr = {}
    fr["epa_all"] = pd.concat([db, ru]).assign(y=lambda x: x.epa)
    fr["epa_pass"] = db.assign(y=db.epa)
    fr["epa_run"] = ru.assign(y=ru.epa)
    pa = db[(db.pass_attempt == 1)]
    fr["cpoe"] = pa[pa.cpoe.notna()].assign(y=lambda x: x.cpoe)
    fr["int"] = pa.assign(y=pa.interception.astype(float))
    fr["sack_r"] = db.assign(y=db.sack.astype(float))
    fr["qbrun"] = db.assign(y=(db.rush_attempt == 1).astype(float))
    tr = db[db.was_pressure.notna()].copy()
    ttt = tr.time_to_throw
    npr = tr.number_of_pass_rushers
    X = np.column_stack([np.ones(len(tr)), ttt.fillna(0), ttt.fillna(0) ** 2, ttt.isna(), npr.fillna(0), npr.isna()]).astype(float)
    fit = (tr.season <= TRAIN_END).values
    for nm, col in [("press_net", "was_pressure"), ("sack_net", "sack")]:
        yv = tr[col].astype(float).values
        b = linfit(X[fit], yv[fit])
        fr[nm] = tr.assign(y=yv - X @ b)
    man = db[db.defense_man_zone_type.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])]
    fr["man_epa"] = man[man.defense_man_zone_type == "MAN_COVERAGE"].assign(y=lambda x: x.epa)
    fr["zone_epa"] = man[man.defense_man_zone_type == "ZONE_COVERAGE"].assign(y=lambda x: x.epa)
    dp = db[(db.pass_attempt == 1) & (db.ngs_air_yards >= 20) & (db.season <= 2022)]
    fr["deep"] = dp.assign(y=lambda x: x.epa)
    box = ru.defenders_in_box
    cat = np.where(box.isna(), "unk", np.where(box <= 6, "light", np.where(box >= 8, "stack", "mid")))
    ru = ru.assign(cat=cat)
    mu = ru.groupby("cat").epa.mean()
    fr["run_net"] = ru.assign(y=ru.epa - ru.cat.map(mu))
    fr["run_light"] = ru[ru.cat == "light"].assign(y=lambda x: x.epa)
    fr["run_stack"] = ru[ru.cat == "stack"].assign(y=lambda x: x.epa)
    fg = d[(d.play_type == "field_goal") & d.posteam_score_post.notna()].copy()
    fg["dist"] = fg.yardline_100 + 17
    fg["made"] = ((fg.posteam_score_post - fg.posteam_score) == 3).astype(float)
    tr0 = fg[fg.season <= 2015]

    def xf(z):
        return np.column_stack([np.ones(len(z)), z.dist / 10, (z.dist / 10) ** 2])

    bf = logit_fit(xf(tr0), tr0.made.values)
    fg["pe"] = 1 / (1 + np.exp(-xf(fg) @ bf))
    fg["y"] = fg.made - fg.pe
    fr["fg"] = fg
    fr["fg_long"] = fg[fg.dist >= 45]
    out = {}
    for k, f in fr.items():
        g = f.groupby(["game_id", "season", "week", "off", "def"]).y.agg(["mean", "count"]).reset_index()
        g.columns = ["game_id", "season", "week", "off", "def", "y", "n"]
        out[k] = (g, float(f.y.var()))
    return out, d


def ngs_frames(d):
    nv = pd.read_parquet(NGS)
    nv = nv[(nv.season_type == "REG") & (nv.week >= 1)].copy()
    nv["team"] = tnorm(nv.team_abbr)
    sched = d[["game_id", "season", "week", "off", "def"]].drop_duplicates()
    sched = sched[sched.off.notna()]
    out = {}
    for nm, vcol, wcol in [("sep", "avg_separation", "targets"), ("yac", "avg_yac_above_expectation", "receptions")]:
        z = nv[nv[vcol].notna() & (nv[wcol] > 0)].copy()
        z["wv"] = z[vcol] * z[wcol]
        g = z.groupby(["season", "week", "team"]).agg(wv=("wv", "sum"), n=(wcol, "sum")).reset_index()
        g["y"] = g.wv / g.n
        m = g.merge(sched, left_on=["season", "week", "team"], right_on=["season", "week", "off"], how="inner")
        m = m[["game_id", "season", "week", "off", "def", "y", "n"]]
        m = m[m.season <= 2025]
        sig2 = float((z[wcol] * (z[vcol] - np.average(z[vcol], weights=z[wcol])) ** 2).mean())
        out[nm] = (m.reset_index(drop=True), sig2)
    return out


def prep(g):
    g = g.copy()
    g["t"] = (g.season - S0) * 18 + (g.week - 1)
    return g.sort_values("t").reset_index(drop=True)


def tau_hat(g, sig2, side):
    z = g[g.season <= TRAIN_END]
    key = "off" if side == 0 else "def"
    a = z.assign(sy=z.y * z.n).groupby(["season", key]).agg(sy=("sy", "sum"), n=("n", "sum")).reset_index()
    a = a[a.n > 0]
    a["m"] = a.sy / a.n
    v = a.m.var() - (sig2 / a.n).mean()
    return max(v, 0.05 * (sig2 / a.n).mean())


def solve(rows, t0, tau, sig2, hl, teams, decay=True):
    t = rows.t.values
    age = (t0 - t).astype(float)
    wt = rows.n.values / sig2 * ((0.5 ** (age / hl)) if decay else 1.0)
    ss = np.unique(rows.season.values)
    ks = len(ss)
    m = len(rows)
    si = np.searchsorted(ss, rows.season.values)
    nt = len(teams)
    K = ks + 2 * nt
    X = np.zeros((m, K))
    ar = np.arange(m)
    X[ar, si] = 1
    X[ar, ks + rows.oi.values] = 1
    X[ar, ks + nt + rows.di.values] = 1
    P = np.zeros(K)
    P[ks:ks + nt] = 1 / tau[0]
    P[ks + nt:] = 1 / tau[1]
    A = X.T @ (X * wt[:, None]) + np.diag(P)
    b = np.linalg.solve(A, X.T @ (wt * rows.y.values))
    return b[ks:ks + nt], b[ks + nt:]


def index_teams(g, teams):
    tix = {k: i for i, k in enumerate(teams)}
    return g.assign(oi=g.off.map(tix), di=g["def"].map(tix))


def rolling(g, sig2, tau, hl, teams, targets, assert_log=None):
    g = index_teams(g, teams)
    t = g.t.values
    res = {}
    for (s, w) in targets:
        t0 = (s - S0) * 18 + (w - 1)
        lo = np.searchsorted(t, t0 - 54, side="left")
        hi = np.searchsorted(t, t0, side="left")
        rows = g.iloc[lo:hi]
        if len(rows) < 40:
            continue
        assert rows.t.max() < t0
        if assert_log is not None:
            assert_log.append((s, w, int(rows.t.max()), int(t0)))
        res[(s, w)] = solve(rows, t0, tau, sig2, hl, teams)
    return res


def season_means(g):
    return g.assign(sy=g.y * g.n).groupby("season").apply(lambda z: z.sy.sum() / z.n.sum(), include_groups=False)


def tune(name, g, sig2, tau0, teams, targets):
    best = None
    gm = season_means(g)
    gg = index_teams(g, teams)
    key = {k: v for k, v in gg.groupby(["season", "week"])}
    rows_out = []
    for hl in GRID_HL:
        for tm in GRID_TAU:
            tau = (tau0[0] * tm, tau0[1] * tm)
            r = rolling(g, sig2, tau, hl, teams, targets)
            num = den = 0.0
            for (s, w), (o, dd) in r.items():
                z = key.get((s, w))
                if z is None:
                    continue
                pred = o[z.oi.values] + dd[z.di.values]
                num += float((z.n.values * (z.y.values - gm[s] - pred) ** 2).sum())
                den += float(z.n.sum())
            mse = num / den if den else np.inf
            rows_out.append((name, hl, tm, mse))
            LOOKS["tuning"] += 1
            if best is None or mse < best[0]:
                best = (mse, hl, tm)
    return best, rows_out


def build():
    d = load()
    mf, d = metric_frames(d)
    mf.update(ngs_frames(d))
    teams = sorted(set(d.off.dropna()) | set(d["def"].dropna()))
    assert len(teams) == 32, len(teams)
    games = d[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    allt = [(int(s), int(w)) for s, w in games.values]
    table = None
    params = {}
    gfr = {}
    tune_rows = []
    logs = {}
    for name, (g, sig2) in mf.items():
        g = prep(g[g.season <= 2025])
        first = FIRST.get(name, 2009)
        g = g[g.season >= first].reset_index(drop=True)
        gfr[name] = (g, sig2)
        tau0 = (tau_hat(g, sig2, 0), tau_hat(g, sig2, 1))
        vt = [x for x in allt if 2017 <= x[0] <= TRAIN_END and x[0] >= first + (1 if first > 2009 else 0)]
        best, tr = tune(name, g, sig2, tau0, teams, vt)
        tune_rows += tr
        mse, hl, tm = best
        tau = (tau0[0] * tm, tau0[1] * tm)
        params[name] = dict(hl=hl, tau_mult=tm, tau_off=tau0[0] ** .5, tau_def=tau0[1] ** .5, sig2=sig2, mse=mse)
        lg = []
        tg = [x for x in allt if x[0] >= first]
        r = rolling(g, sig2, tau, hl, teams, tg, lg)
        logs[name] = lg
        rows = []
        for (s, w), (o, dd) in r.items():
            for i, tm_ in enumerate(teams):
                rows.append((s, w, tm_, o[i], dd[i]))
        z = pd.DataFrame(rows, columns=["season", "week", "team", name + "_off", name + "_def"])
        table = z if table is None else table.merge(z, on=["season", "week", "team"], how="outer")
        print(name, params[name], flush=True)
    assert all(a[2] < a[3] for lg in logs.values() for a in lg)
    table = table.sort_values(["season", "week", "team"]).reset_index(drop=True)
    table.to_parquet(TABLE)
    pd.DataFrame(tune_rows, columns=["metric", "hl_weeks", "tau_mult", "mse"]).to_csv(OUT + "units_tuning.csv", index=False)
    json.dump(params, open(OUT + "units_params.json", "w"), indent=1)
    return d, gfr, params, teams, table


def static_effects(g, sig2, tau, teams, season, sub=None):
    z = g[g.season == season]
    if sub is not None:
        z = z[sub(z)]
    if len(z) < 40:
        return None
    z = index_teams(z, teams)
    return solve(z, 10 ** 6, tau, sig2, 1.0, teams, decay=False)


def reliability(gfr, params, teams, table):
    out = []
    for name, (g, sig2) in gfr.items():
        p = params[name]
        first = FIRST.get(name, 2009)
        tau = (p["tau_off"] ** 2 * p["tau_mult"], p["tau_def"] ** 2 * p["tau_mult"])
        gz = g.copy()
        gz["gn"] = gz.groupby(["season", "off"]).t.rank(method="first")
        gz["gnd"] = gz.groupby(["season", "def"]).t.rank(method="first")
        sh = {0: [], 1: []}
        co = {0: [], 1: []}
        prev = {}
        for s in range(first, 2026):
            for side, gcol in [(0, "gn"), (1, "gnd")]:
                a = static_effects(gz, sig2, tau, teams, s, lambda z: (z[gcol] % 2 == 1))
                b = static_effects(gz, sig2, tau, teams, s, lambda z: (z[gcol] % 2 == 0))
                if a is not None and b is not None:
                    r = np.corrcoef(a[side], b[side])[0, 1]
                    sh[side].append(2 * r / (1 + r))
            full = static_effects(gz, sig2, tau, teams, s)
            if full is None:
                continue
            for side in (0, 1):
                if (s - 1, side) in prev:
                    co[side].append(np.corrcoef(prev[(s - 1, side)], full[side])[0, 1])
                prev[(s, side)] = full[side]
        tb = table[["season", "week", "team", name + "_off", name + "_def"]]
        gmean = season_means(gz)
        for side in (0, 1):
            keyc = "off" if side == 0 else "def"
            oth = "def" if side == 0 else "off"
            m = gz.merge(tb.rename(columns={"team": oth, name + "_off": "ro_o", name + "_def": "ro_d"}), on=["season", "week", oth], how="left")
            m["oppr"] = m["ro_d"] if side == 0 else m["ro_o"]
            m["dev"] = m.y - m.season.map(gmean) - m.oppr.fillna(0)
            m = m.sort_values([keyc, "t"])
            m["prev"] = m.groupby([keyc, "season"]).dev.shift(1)
            q = m[m.n >= m.n.median() * 0.5].dropna(subset=["dev", "prev"])
            ac = float(np.corrcoef(q.dev, q.prev)[0, 1]) if len(q) > 50 else np.nan
            c = name + ("_off" if side == 0 else "_def")
            w10 = table[(table.week == 10) & (table.season >= first + (1 if first > 2009 else 0))]
            sdv = float(w10.groupby("season")[c].std().mean())
            out.append(dict(metric=name, side=keyc, tau_sd=params[name]["tau_off" if side == 0 else "tau_def"], rating_sd_wk10=sdv, splithalf_sb=float(np.nanmean(sh[side])), lag1_autocorr=ac, carryover=float(np.nanmean(co[side])), n_seasons=len(co[side]) + 1))
            LOOKS["reliability"] += 4
    df = pd.DataFrame(out)
    df.to_csv(OUT + "units_reliability.csv", index=False)
    return df


def covariance(table):
    z = table[(table.week == 10) & (table.season >= 2018)].copy()
    cols = [c for c in z.columns if c.endswith("_off") or c.endswith("_def")]
    for c in cols:
        base, side = c.rsplit("_", 1)
        sgn = SIGN[base][0 if side == "off" else 1]
        z[c] = sgn * (z[c] - z.groupby("season")[c].transform("mean"))
    cm = z[cols].corr()
    cm.to_csv(OUT + "units_corr.csv")
    zz = cm.where(np.triu(np.ones(cm.shape), 1) == 1).stack()
    top = zz.reindex(zz.abs().sort_values(ascending=False).index).head(40)
    top.to_csv(OUT + "units_corr_top.csv")
    return cm, top


def teamgame_table(gfr, table, target):
    g, sig2 = gfr[target]
    m = g.merge(table, left_on=["season", "week", "off"], right_on=["season", "week", "team"], how="inner").drop(columns="team")
    cols = [c for c in table.columns if c.endswith("_def")]
    r = table[["season", "week", "team"] + cols].rename(columns={c: "o_" + c for c in cols}).rename(columns={"team": "def"})
    return m.merge(r, on=["season", "week", "def"], how="inner")


def ols_w(X, y, w, alpha=0.0):
    A = X.T @ (X * w[:, None])
    A = A + (alpha + 1e-9 * np.trace(A) / len(A)) * np.eye(len(A))
    return np.linalg.solve(A, X.T @ (w * y))


def incremental(gfr, table):
    sets = {
        "epa_pass": ["cpoe", "int", "sack_r", "sack_net", "press_net", "qbrun", "man_epa", "zone_epa", "sep", "yac", "deep", "epa_pass"],
        "epa_run": ["run_net", "run_light", "run_stack", "epa_run", "qbrun"],
        "epa_all": ["cpoe", "int", "sack_r", "sack_net", "press_net", "qbrun", "man_epa", "zone_epa", "sep", "yac", "deep", "run_net", "run_light", "run_stack", "epa_pass", "epa_run"],
    }
    rows = []
    rng = np.random.default_rng(0)
    base = ["epa_all"]

    def feat(z, us):
        cols = []
        for u in us:
            cols.append(z[u + "_off"].values)
            cols.append(z["o_" + u + "_def"].values)
        return np.column_stack([np.ones(len(z))] + cols)

    for target, units in sets.items():
        m = teamgame_table(gfr, table, target)
        m = m[m.season >= 2016].copy()
        for u in set(units) | set(base):
            for s in ("_off", "_def"):
                c = u + s if s == "_off" else "o_" + u + s
                m[c] = m[c].fillna(0.0)
        m = m.dropna(subset=["y"])
        tr = m[m.season <= TRAIN_END]
        te = m[m.season > TRAIN_END]
        wtr, wte = tr.n.values.astype(float), te.n.values.astype(float)
        bb = ols_w(feat(tr, base), tr.y.values, wtr)
        eb = (te.y.values - feat(te, base) @ bb) ** 2
        ug, gi = np.unique(te.game_id.values, return_inverse=True)
        B = 1000
        cnt = np.zeros((B, len(ug)))
        for b in range(B):
            cnt[b] = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        vy = np.average((te.y.values - np.average(te.y.values, weights=wte)) ** 2, weights=wte)

        def evaldelta(pu, label, nk):
            eu = (te.y.values - pu) ** 2
            dl = wte * (eb - eu)
            gd = np.bincount(gi, weights=dl, minlength=len(ug))
            gw = np.bincount(gi, weights=wte, minlength=len(ug))
            bs = (cnt @ gd) / (cnt @ gw)
            tot = dl.sum() / wte.sum()
            LOOKS["incremental"] += 1
            rows.append(dict(target=target, added=label, k=nk, mse_base=float((wte * eb).sum() / wte.sum()), delta_mse=float(tot), r2_gain=float(tot / vy), lo90=float(np.quantile(bs, .05)), hi90=float(np.quantile(bs, .95)), prob_pos=float((bs > 0).mean()), n_train=len(tr), n_test=len(te)))

        for u in units:
            if u in base:
                continue
            us = base + [u]
            bu = ols_w(feat(tr, us), tr.y.values, wtr)
            evaldelta(feat(te, us) @ bu, u, 2)
        allu = base + [u for u in units if u not in base]
        Xt, Xe = feat(tr, allu), feat(te, allu)
        sc = Xt.std(0)
        sc[0] = 1
        sc[sc == 0] = 1
        best = None
        for al in [1, 10, 100, 1000, 10000]:
            sse = 0.0
            for s in sorted(tr.season.unique()):
                a = (tr.season != s).values
                bcoef = ols_w(Xt[a] / sc, tr.y.values[a], wtr[a], al)
                sse += float((wtr[~a] * (tr.y.values[~a] - (Xt[~a] / sc) @ bcoef) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, al)
        bcoef = ols_w(Xt / sc, tr.y.values, wtr, best[1])
        evaldelta((Xe / sc) @ bcoef, "ALL(ridge a=%g)" % best[1], len(allu) * 2)
    df = pd.DataFrame(rows)
    df.to_csv(OUT + "units_incremental.csv", index=False)
    return df


def interactions(gfr, table):
    specs = [
        ("press_net", "press_net", "press_net", "pro x rush -> pressure"),
        ("press_net", "press_net", "epa_pass", "pro x rush -> pass EPA"),
        ("sack_net", "sack_net", "sack_net", "pro x rush -> sack"),
        ("sep", "sep", "epa_pass", "receivers x coverage(sep) -> pass EPA"),
        ("yac", "yac", "epa_pass", "yac x yac allowed -> pass EPA"),
        ("cpoe", "cpoe", "epa_pass", "QB acc x cov cpoe -> pass EPA"),
        ("run_net", "run_net", "epa_run", "run block x run D -> run EPA"),
        ("run_net", "run_stack", "run_stack", "run block x stacked-box D -> stacked-box EPA"),
    ]
    rows = []
    rng = np.random.default_rng(1)
    for ou, du, target, label in specs:
        m = teamgame_table(gfr, table, target)
        m = m[m.season >= 2016].dropna(subset=[ou + "_off", "o_" + du + "_def", "epa_all_off", "o_epa_all_def"])
        for tag, sel in [("2016-25", m.season >= 2016), ("2016-20", m.season <= 2020), ("2021-25", m.season >= 2021)]:
            z = m[sel]
            a = z[ou + "_off"].values
            b = z["o_" + du + "_def"].values
            a = (a - a.mean()) / a.std()
            b = (b - b.mean()) / b.std()
            X = np.column_stack([np.ones(len(z)), z.epa_all_off.values, z.o_epa_all_def.values, a, b, a * b])
            w = z.n.values.astype(float)
            y = z.y.values
            ug, gi = np.unique(z.game_id.values, return_inverse=True)
            beta = ols_w(X, y, w)
            idx_by = [np.where(gi == k)[0] for k in range(len(ug))]
            bs = []
            for _ in range(400):
                pick = rng.integers(0, len(ug), len(ug))
                ii = np.concatenate([idx_by[k] for k in pick])
                bs.append(ols_w(X[ii], y[ii], w[ii])[5])
            bs = np.array(bs)
            LOOKS["interaction"] += 1
            rows.append(dict(spec=label, window=tag, target=target, off_unit=ou, def_unit=du, n=len(z), main_off=float(beta[3]), main_def=float(beta[4]), inter=float(beta[5]), lo90=float(np.quantile(bs, .05)), hi90=float(np.quantile(bs, .95)), prob_pos=float((bs > 0).mean()), y_sd=float(np.sqrt(np.average((y - np.average(y, weights=w)) ** 2, weights=w)))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT + "units_interactions.csv", index=False)
    return df


if __name__ == "__main__":
    d, gfr, params, teams, table = build()
    reliability(gfr, params, teams, table)
    covariance(table)
    incremental(gfr, table)
    interactions(gfr, table)
    json.dump(LOOKS, open(OUT + "units_looks.json", "w"))
    print(LOOKS)
