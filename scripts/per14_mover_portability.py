import glob
import itertools
import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

WP_LO, WP_HI = 0.05, 0.95
MIN_SPELL_GAMES = 16
MIN_STINT_GAMES = MIN_SPELL_GAMES // 2
N_BOOT = 4000
N_PERM = 5000
N_QUANT = 4
TIME_EDGES = [0, 300, 600, 900, 1200, 1500, 1800]
LEAD_EDGES = [1, 9, 17, 10**6]
rng = np.random.default_rng(14)
OUT = "artifacts/per14"
os.makedirs(OUT, exist_ok=True)

cols = [
    "game_id",
    "season",
    "season_type",
    "week",
    "posteam",
    "defteam",
    "play_type",
    "epa",
    "wp",
    "qtr",
    "qb_kneel",
    "qb_spike",
    "score_differential",
    "down",
    "ydstogo",
    "game_seconds_remaining",
    "qb_dropback",
    "fixed_drive",
    "pass_attempt",
    "play_id",
]
raw = pd.read_parquet("data/pbp/team_style/raw_pbp_narrow.parquet", columns=cols)
raw = raw[raw.season_type.eq("REG")]


def tbin(s):
    return np.clip(np.searchsorted(TIME_EDGES, s, side="right") - 1, 0, len(TIME_EDGES) - 2)


def lbin(a):
    return np.where(a >= 1, np.searchsorted(LEAD_EDGES, a, side="right") - 1, -1)


def halftime():
    d = raw[raw.play_type.isin(["pass", "run"]) & raw.qtr.between(1, 4)]
    d = d[~d.qb_kneel.fillna(0).astype(bool) & ~d.qb_spike.fillna(0).astype(bool)]
    d = d.dropna(subset=["epa", "wp", "posteam", "defteam"])
    d = d[d.wp.between(WP_LO, WP_HI)].copy()
    d["half"] = np.where(d.qtr <= 2, 1, 2)
    h = (
        d[d.qtr == 2]
        .groupby(["game_id", "posteam"])
        .score_differential.last()
        .rename("margin")
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    def side(gc, oc_, name):
        g = d.groupby(["game_id", "season", "week", gc, oc_, "half"]).epa.agg(["mean", "size"])
        g = g.unstack("half")
        g.columns = [f"{a}{b}" for a, b in g.columns]
        g = g.dropna().reset_index().rename(columns={gc: "team", oc_: "opp"})
        g[name] = g.mean2 - g.mean1
        g["w_" + name] = 1.0 / (1.0 / g.size1 + 1.0 / g.size2)
        return g[["game_id", "season", "week", "team", "opp", name, "w_" + name]]

    off = side("posteam", "defteam", "off_delta")
    de = side("defteam", "posteam", "def_delta")
    g = off.merge(de, on=["game_id", "season", "week", "team", "opp"])
    g = g.merge(h, on=["game_id", "team"], how="left")
    g["margin"] = g.margin.fillna(0.0)
    g = g.sort_values(["season", "week"]).reset_index(drop=True)
    tot = g.groupby(["season", "team"]).def_delta.transform("sum")
    n = g.groupby(["season", "team"]).def_delta.transform("size")
    g["def_loo"] = ((tot - g.def_delta) / (n - 1).replace(0, np.nan)).fillna(0.0)
    loo = g.set_index(["game_id", "team"]).def_loo
    g["opp_def_loo"] = loo.reindex(list(zip(g.game_id, g.opp, strict=True))).fillna(0.0).values
    g["margin_abs"] = g.margin.abs()
    X = np.column_stack(
        [np.ones(len(g)), g.margin.values, g.margin_abs.values, g.opp_def_loo.values]
    )
    yv = g.off_delta.values
    w = g.w_off_delta.values
    res = np.zeros(len(g))
    for sn in sorted(g.season.unique()):
        m = (g.season == sn).values
        sw = np.sqrt(w[~m])
        b = np.linalg.lstsq(X[~m] * sw[:, None], yv[~m] * sw, rcond=None)[0]
        res[m] = yv[m] - X[m] @ b
    return pd.DataFrame({"game_id": g.game_id, "team": g.team, "v": res, "w": w})


def lead():
    d = raw[raw.qtr.isin([3, 4]) & raw.posteam.notna()]
    d = d.dropna(subset=["game_seconds_remaining", "score_differential", "fixed_drive"])
    p = d[d.play_type.isin(["run", "pass"]) | d.qb_kneel.fillna(0).astype(bool)].copy()
    p = p.sort_values(["game_id", "play_id"])
    p["kneel"] = p.qb_kneel.fillna(0).astype(float)
    p["run"] = (p.play_type.eq("run") & p.kneel.eq(0)).astype(float)
    p = p[p.qb_spike.fillna(0).eq(0)].copy()
    p["nxt_gsr"] = p.groupby(["game_id", "fixed_drive"]).game_seconds_remaining.shift(-1)
    p["nxt_qtr"] = p.groupby(["game_id", "fixed_drive"]).qtr.shift(-1)
    p["sec"] = np.where(p.nxt_qtr.eq(p.qtr), p.game_seconds_remaining - p.nxt_gsr, np.nan)
    p["tb"] = tbin(p.game_seconds_remaining.values)
    p["lb"] = lbin(p.score_differential.abs().values)
    cell = p[(p.score_differential > 0) & (p.lb >= 0)].copy()
    cell["cell"] = cell.lb * 10 + cell.tb
    out = {}
    for col in ["sec", "run"]:
        base = np.full(len(cell), np.nan)
        for s in sorted(cell.season.unique()):
            m = (cell.season == s).values
            mu = cell[~m & cell[col].notna().values].groupby("cell")[col].mean()
            base[m] = cell.loc[m, "cell"].map(mu).values
        cell["r"] = cell[col] - base
        sub = cell[cell.r.notna()]
        gg = sub.groupby(["game_id", "posteam"]).r.agg(["mean", "size"]).reset_index()
        out["lead_" + col] = pd.DataFrame(
            {"game_id": gg.game_id, "team": gg.posteam, "v": gg["mean"], "w": gg["size"]}
        )
    return out


def trailing():
    d = raw[raw.qtr.between(1, 4) & raw.posteam.notna()]
    d = d[~d.qb_kneel.fillna(0).astype(bool) & ~d.qb_spike.fillna(0).astype(bool)]
    d = d.dropna(subset=["score_differential", "down", "ydstogo", "game_seconds_remaining"])
    d = d[d.play_type.isin(["pass", "run"])].copy()
    d["trail"] = d.score_differential < 0
    d["db"] = d.qb_dropback.fillna(0).astype(float)
    yb = pd.qcut(d.ydstogo[~d.trail], N_QUANT, retbins=True, duplicates="drop")[1]
    d["cell"] = d.down.astype(int).astype(str) + "_" + np.digitize(d.ydstogo, yb[1:-1]).astype(str)
    ref = d[~d.trail]
    d["ex"] = d.db - d.cell.map(ref.groupby("cell").db.mean())
    d = d[d.ex.notna()]
    t = d[d.trail]
    gx = t.groupby(["game_id", "posteam"]).ex.agg(["sum", "size"]).reset_index()
    gx = gx.rename(columns={"posteam": "team", "sum": "ex", "size": "n"})
    nt = d[~d.trail].groupby(["game_id", "season", "posteam"]).ex.agg(["sum", "size"]).reset_index()
    nt = nt.rename(columns={"posteam": "team"})
    nt["S"] = nt.groupby(["season", "team"])["sum"].transform("sum")
    nt["N"] = nt.groupby(["season", "team"])["size"].transform("sum")
    nt["loo"] = (nt.S - nt["sum"]) / (nt.N - nt["size"]).replace(0, np.nan)
    sh = gx.merge(nt[["game_id", "team", "loo"]], on=["game_id", "team"], how="left")
    sh = sh.dropna(subset=["loo"])
    return {
        "trail_pass_incl": pd.DataFrame(
            {"game_id": gx.game_id, "team": gx.team, "v": gx.ex / gx.n, "w": gx.n}
        ),
        "trail_pass_shift": pd.DataFrame(
            {"game_id": sh.game_id, "team": sh.team, "v": (sh.ex - sh.n * sh.loo) / sh.n, "w": sh.n}
        ),
    }


traits = {"halftime_off": halftime()}
traits.update(lead())
traits.update(trailing())

sch = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])[
    ["game_id", "season", "home_team", "away_team", "home_coach", "away_coach", "gameday"]
]
home = sch.rename(columns={"home_team": "team", "home_coach": "coach"})[
    ["game_id", "season", "team", "coach", "gameday"]
]
away = sch.rename(columns={"away_team": "team", "away_coach": "coach"})[
    ["game_id", "season", "team", "coach", "gameday"]
]
tg = pd.concat([home, away], ignore_index=True)
tg["t"] = (pd.to_datetime(tg.gameday, utc=True) + pd.Timedelta(days=1)).astype(
    "datetime64[ns, UTC]"
)
tg["season"] = tg.season.astype("int64")
ch = pd.read_parquet(sorted(glob.glob("data/raw/coordinators/*/coordinator_history.parquet"))[-1])
ch["effective_observed_at"] = pd.to_datetime(ch.effective_observed_at, utc=True).astype(
    "datetime64[ns, UTC]"
)
oc = ch[ch.role == "OC"].sort_values("effective_observed_at").astype({"season": "int64"})
tg = tg.sort_values("t").reset_index(drop=True)
mm = pd.merge_asof(
    tg[["game_id", "team", "season", "t"]],
    oc[["team", "season", "effective_observed_at", "person"]],
    left_on="t",
    right_on="effective_observed_at",
    by=["team", "season"],
    direction="backward",
)
first = oc.drop_duplicates(["team", "season"])[["team", "season", "person"]].rename(
    columns={"person": "fp"}
)
mm = mm.merge(first, on=["team", "season"], how="left")
tg["oc"] = mm.person.fillna(mm.fp).values
tg = tg[["game_id", "team", "t", "oc", "coach"]]


def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def partial(a, b, c):
    rab, rac, rbc = corr(a, b), corr(a, c), corr(b, c)
    den = np.sqrt((1 - rac**2) * (1 - rbc**2))
    return float((rab - rac * rbc) / den) if den > 0 else np.nan


def cluster_boot(persons, fn):
    u, inv = np.unique(persons, return_inverse=True)
    idx_by = [np.flatnonzero(inv == i) for i in range(len(u))]
    out = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(u), len(u))
        idx = np.concatenate([idx_by[i] for i in pick])
        v = fn(idx)
        if np.isfinite(v):
            out.append(v)
    out = np.array(out)
    return np.percentile(out, [2.5, 97.5]).tolist(), float((out > 0).mean())


def perm_p(x1, x2, stat):
    obs = stat(x1, x2)
    nulls = np.array([stat(x1, rng.permutation(x2)) for _ in range(N_PERM)])
    return float((np.sum(nulls >= obs) + 1) / (N_PERM + 1))


def run_trait(name, df):
    g = df.merge(tg, on=["game_id", "team"]).sort_values("t").reset_index(drop=True)
    g = g.dropna(subset=["oc"])
    g = g.sort_values(["team", "t"]).reset_index(drop=True)
    g["run"] = (g.oc != g.oc.shift()) | (g.team != g.team.shift())
    g["run"] = g.run.cumsum()
    g["vw"] = g.v * g.w
    st = (
        g.groupby("run")
        .agg(
            person=("oc", "first"),
            team=("team", "first"),
            t0=("t", "min"),
            n=("v", "size"),
            W=("w", "sum"),
            vw=("vw", "sum"),
        )
        .reset_index()
    )
    st["m"] = st.vw / st.W
    gm = np.average(g.v, weights=g.w)
    dev = (g.v - g.run.map(st.set_index("run").m)) ** 2 * g.w
    sigma2 = float(dev.sum() / (len(g) - len(st)))
    long = st[st.n >= MIN_SPELL_GAMES]
    tau2 = float(np.var(long.m, ddof=1) - np.mean(sigma2 / long.W))
    rel_unit = tau2 / (tau2 + sigma2 / long.W.median()) if tau2 > 0 else 0.0
    st["shr"] = np.where(tau2 > 0, (st.m - gm) * tau2 / (tau2 + sigma2 / st.W), 0.0)
    st["raw"] = st.m - gm
    s = st.set_index("run")
    ok = s.n >= MIN_STINT_GAMES
    ev = []
    for person, p in st[st.n >= MIN_STINT_GAMES].sort_values("t0").groupby("person"):
        rows = p.to_dict("records")
        for a, b in itertools.pairwise(rows):
            if a["team"] == b["team"]:
                continue
            e = {"person": person, "teamA": a["team"], "teamB": b["team"], "runA": a["run"]}
            e["x1"], e["x2"] = a["shr"], b["shr"]
            e["x1_raw"], e["x2_raw"] = a["raw"], b["raw"]
            e["n1"], e["n2"] = a["n"], b["n"]
            succ = a["run"] + 1
            e["succ"] = (
                s.loc[succ].shr
                if succ in s.index and s.loc[succ].team == a["team"] and ok.loc[succ]
                else np.nan
            )
            prev = b["run"] - 1
            e["prevB"] = (
                s.loc[prev].shr
                if prev in s.index and s.loc[prev].team == b["team"] and ok.loc[prev]
                else np.nan
            )
            ev.append(e)
    ev = pd.DataFrame(ev)
    pers = ev.person.values
    x1, x2 = ev.x1.values, ev.x2.values
    r_person = corr(x1, x2)
    ci_p, pp_p = cluster_boot(pers, lambda i: corr(x1[i], x2[i]))
    out = {
        "n_stints_ge_min": int(ok.sum()),
        "n_movers": len(ev),
        "n_mover_persons": int(ev.person.nunique()),
        "tau2": tau2,
        "sigma2": sigma2,
        "reliability_median_long_spell": float(rel_unit),
        "person": {
            "r": r_person,
            "ci95": ci_p,
            "prob_positive": pp_p,
            "perm_p": perm_p(x1, x2, corr),
            "raw_r": corr(ev.x1_raw.values, ev.x2_raw.values),
        },
    }
    e2 = ev.dropna(subset=["succ"]).reset_index(drop=True)
    a1, a2, ap = e2.x1.values, e2.succ.values, e2.person.values
    ci_t, pp_t = cluster_boot(ap, lambda i: corr(a1[i], a2[i]))
    out["team"] = {
        "n": len(e2),
        "r": corr(a1, a2),
        "ci95": ci_t,
        "prob_positive": pp_t,
        "perm_p": perm_p(a1, a2, corr),
    }
    e3 = ev.dropna(subset=["succ"]).reset_index(drop=True)
    b1, b2, bs, bp = e3.x1.values, e3.x2.values, e3.succ.values, e3.person.values
    r_pp = corr(b1, b2)
    r_tt = corr(b1, bs)

    def dif(i):
        return corr(b1[i], b2[i]) - corr(b1[i], bs[i])

    ci_d, pp_d = cluster_boot(bp, dif)
    out["paired_person_minus_team"] = {
        "n": len(e3),
        "r_person": r_pp,
        "r_team": r_tt,
        "diff": float(r_pp - r_tt),
        "ci95": ci_d,
        "prob_positive": pp_d,
    }
    e4 = ev.dropna(subset=["prevB"]).reset_index(drop=True)
    c1, c2, cp, cb = e4.x1.values, e4.x2.values, e4.prevB.values, e4.person.values
    ci_q, pp_q = cluster_boot(cb, lambda i: partial(c1[i], c2[i], cp[i]))
    out["partial_person_given_new_team_prior"] = {
        "n": len(e4),
        "r_new_team_prior_vs_x2": corr(cp, c2),
        "partial_r": partial(c1, c2, cp),
        "ci95": ci_q,
        "prob_positive": pp_q,
    }
    hc = g.merge(st[["run", "n"]], on="run")
    hc_ev = []
    for person, p in st[st.n >= MIN_STINT_GAMES].sort_values("t0").groupby("person"):
        hg = hc[(hc.coach == person)]
        for tm, gg in hg.groupby("team"):
            if len(gg) < MIN_STINT_GAMES:
                continue
            ocs = p[p.team != tm]
            ocs = ocs[ocs.t0 < gg.t.min()]
            if len(ocs) == 0:
                continue
            a = ocs.iloc[-1]
            hv = float(np.average(gg.v, weights=gg.w)) - gm
            hshr = hv * tau2 / (tau2 + sigma2 / gg.w.sum()) if tau2 > 0 else 0.0
            hc_ev.append((person, a["shr"], hshr))
    out["oc_to_hc"] = {"n": len(hc_ev)}
    if len(hc_ev) >= 3:
        h1 = np.array([e[1] for e in hc_ev])
        h2 = np.array([e[2] for e in hc_ev])
        out["oc_to_hc"].update({"r": corr(h1, h2), "perm_p": perm_p(h1, h2, corr)})
    return out, ev


res = {"min_stint_games": MIN_STINT_GAMES, "traits": {}}
for name, df in traits.items():
    o, ev = run_trait(name, df)
    res["traits"][name] = o
    ev.to_csv(f"{OUT}/events_{name}.csv", index=False)
res["looks"] = {
    "traits": len(traits),
    "correlations_per_trait": 5,
    "total_correlation_looks": 5 * len(traits),
    "family": "OC mover portability declared before any sign seen",
}
with open(f"{OUT}/mover_portability.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
