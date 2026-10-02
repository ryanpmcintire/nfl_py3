import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim09_urgency import NV, cluster_ols, controls, dl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "sim09"
PBP = ROOT / "data" / "pbp" / "raw" / "20260929T191306Z"
PART = ROOT / "data" / "players" / "participation" / "raw" / "20260813T131635Z"
SCHED = ROOT / "data" / "raw" / "20260908T162105Z" / "schedules.parquet"
RNG = np.random.default_rng(20261003)
LOOKS = {"b1": 0, "b2": 0, "c1": 0, "d3": 0}
FIT = (2009, 2017)
HELD = (2018, 2025)
CT = ["pos_home", "off_epa", "opp_off", "own_def", "def_epa", "z_trail", "z_lead", "el", "el2", "d2", "d3", "d4", "logyd", "yl", "yl2"]
COLS = ["play_id", "game_id", "season", "season_type", "week", "home_team", "away_team", "posteam", "defteam", "fixed_drive", "down", "play_type", "yards_gained",
        "epa", "success", "qtr", "ydstogo", "yardline_100", "game_seconds_remaining", "qb_dropback", "qb_kneel", "qb_spike", "score_differential",
        "fixed_drive_result", "passer_player_id"]


def load():
    d = pd.concat([pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=COLS) for s in range(2009, 2026)], ignore_index=True)
    d = d[d.season_type == "REG"].sort_values(["game_id", "play_id"]).reset_index(drop=True)
    return d


def prplays(d):
    p = d[d.play_type.isin(["pass", "run"]) & (d.qb_kneel != 1) & (d.qb_spike != 1) & (d.qtr <= 4) & d.posteam.notna() & d.epa.notna() & d.score_differential.notna() & d.down.notna() & d.yardline_100.notna()].copy()
    p["t"] = p.game_seconds_remaining.clip(lower=0)
    p["sd"] = p.score_differential.astype(float)
    p["z"] = p.sd / np.sqrt(p.t / 60 + 1)
    p["z_trail"] = p.z.clip(upper=0)
    p["z_lead"] = p.z.clip(lower=0)
    p["el"] = (3600 - p.t) / 3600
    p["el2"] = p.el**2
    for k in (2, 3, 4):
        p[f"d{k}"] = (p.down == k).astype(float)
    p["logyd"] = np.log(p.ydstogo.clip(lower=1))
    p["yl"] = p.yardline_100 / 100
    p["yl2"] = p.yl**2
    p["ypp"] = p.yards_gained.astype(float)
    p["succ"] = p.success.astype(float)
    p["second"] = (p.qtr >= 3).astype(float)
    p = p.reset_index(drop=True)
    p = controls(p)
    return p.dropna(subset=["off_epa", "opp_off", "own_def", "def_epa"]).reset_index(drop=True)


def fit(df, terms, y, extra=None, demean_by=None):
    cols = terms + CT + (extra or [])
    s = df.dropna(subset=[y] + terms).copy()
    X = s[cols].astype(float)
    yy = s[y].astype(float)
    if demean_by is not None:
        key = s[demean_by].values
        X = X - X.groupby(key).transform("mean")
        yy = yy - yy.groupby(key).transform("mean")
        X = X.loc[:, X.std() > 0]
    else:
        X.insert(0, "const", 1.0)
    b, se, u = cluster_ols(X, yy, s.game_id)
    rows = []
    for t in terms:
        rows.append({"term": t, "outcome": y, "n": len(s), "mean": float(s[y].mean()), "coef": b[t], "se": se[t], "prob_pos": norm.cdf(b[t] / se[t]), "lo": b[t] - 1.96 * se[t], "hi": b[t] + 1.96 * se[t]})
    return rows, u, s


def by_era(p, fn):
    out = []
    for name, (a, b) in (("fit", FIT), ("held", HELD)):
        q = p[(p.season >= a) & (p.season <= b)]
        for r in fn(q):
            r["era"] = name
            out.append(r)
    return out


def nohuddle(p):
    nv = pd.concat([pd.read_parquet(NV / f"pbp_{s}.parquet", columns=["game_id", "play_id", "no_huddle"]) for s in range(2009, 2018)])
    return p.merge(nv.drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left")


def drive_table(d):
    pr = d.play_type.isin(["pass", "run"]) & (d.qb_kneel != 1) & (d.qb_spike != 1)
    x = d[d.fixed_drive.notna() & d.posteam.notna()].copy()
    x["pr"] = pr.loc[x.index].astype(int)
    x["big"] = (pr.loc[x.index] & (x.yards_gained >= 20)).astype(int)
    g = x.groupby(["game_id", "fixed_drive"])
    t = g.agg(team=("posteam", "first"), defn=("defteam", "first"), first=("play_id", "min"), plays=("pr", "sum"), big=("big", "max"), qtr=("qtr", "first"), res=("fixed_drive_result", lambda s: s.dropna().iloc[-1] if s.notna().any() else None)).reset_index()
    t = t.sort_values(["game_id", "first"]).reset_index(drop=True)
    return t


def b1(p, t):
    t = t.copy()
    t["prior_def_len"] = t.groupby(["game_id", "defn"]).plays.shift(1)
    p = p.merge(t[["game_id", "fixed_drive", "prior_def_len"]], on=["game_id", "fixed_drive"], how="left")
    p["cum_def"] = p.groupby(["game_id", "defteam"]).cumcount() / 100
    p["drv_n"] = p.groupby(["game_id", "fixed_drive"]).cumcount() / 10
    p["prior_len"] = p.prior_def_len.fillna(0) / 10
    p = nohuddle(p)
    p = p.sort_values(["game_id", "play_id"])
    nh = p.groupby(["game_id", "defteam"]).no_huddle.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    p["nh_faced"] = nh
    base = ["cum_def", "drv_n", "prior_len"]
    rows = []

    def fn(q):
        r = []
        for y in ("epa", "succ", "ypp"):
            LOOKS["b1"] += len(base)
            rr, _, _ = fit(q, base, y)
            r += rr
        q2 = q[q.season <= 2017].copy()
        if len(q2) and q2.nh_faced.notna().any():
            q2["nh_faced"] = q2.nh_faced.fillna(q2.nh_faced.mean())
            for y in ("epa", "succ", "ypp"):
                LOOKS["b1"] += 1
                rr, _, _ = fit(q2, base + ["nh_faced"], y)
                r += [x for x in rr if x["term"] == "nh_faced"]
        return r

    rows = by_era(p, fn)
    for r in rows:
        r["unit"] = {"cum_def": "per 100 defensive snaps beyond clock", "drv_n": "per 10 plays into the drive", "prior_len": "per 10 plays of defence's prior drive", "nh_faced": "per 1.0 no-huddle share of prior 10 faced (2009-17 only)"}[r["term"]]
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "dyn_b1.csv", index=False)
    return r


def b2(p, d):
    p = p[p.season >= 2016].copy()
    p["idx"] = p.groupby(["game_id", "posteam"]).cumcount()
    p["idx_d"] = p.groupby(["game_id", "defteam"]).cumcount()
    sched = pd.read_parquet(SCHED, columns=["game_id", "week"]).drop_duplicates("game_id")
    qbs = d.groupby("passer_player_id").size()
    qbs = set(qbs[qbs >= 100].index)
    part = pd.concat([pd.read_parquet(PART / f"season={s}" / "participation.parquet", columns=["game_id", "play_id", "offense_players", "defense_players"]) for s in range(2016, 2026)])
    p = p.merge(part, on=["game_id", "play_id"], how="left")
    res = []
    for unit, tcol, icol, pl in (("off", "posteam", "idx", "offense_players"), ("def", "defteam", "idx_d", "defense_players")):
        u = p[["game_id", "play_id", "season", "week", tcol, icol, pl, "qtr", "sd", "epa", "succ", "pos_home", "off_epa", "opp_off", "own_def", "def_epa", "z_trail", "z_lead", "el", "el2", "d2", "d3", "d4", "logyd", "yl", "yl2"]].rename(columns={tcol: "team", icol: "i", pl: "pl"})
        u["has"] = u.pl.fillna("").str.len() > 0
        n = u.groupby(["game_id", "team"]).agg(N=("i", "size"), cov=("has", "mean"), season=("season", "first"), week=("week", "first")).reset_index()
        n["k"] = n.groupby(["season", "team"]).week.rank(method="first").astype(int)
        e = u[u.has].copy()
        e["pl"] = e.pl.str.split(";")
        e = e.explode("pl")
        pg = e.groupby(["game_id", "team", "pl"]).agg(snaps=("i", "size"), last=("i", "max")).reset_index().merge(n[["game_id", "team", "season", "k", "N", "cov"]], on=["game_id", "team"])
        win_p = []
        win_n = []
        for j in range(1, 7):
            a = pg[["season", "team", "pl", "k", "snaps"]].copy()
            a["k"] = a.k + j
            win_p.append(a)
            b = n[["season", "team", "k", "N"]].copy()
            b["k"] = b.k + j
            win_n.append(b)
        wp = pd.concat(win_p).groupby(["season", "team", "pl", "k"]).snaps.sum().rename("ws").reset_index()
        wn = pd.concat(win_n).groupby(["season", "team", "k"]).N.sum().rename("wN").reset_index()
        pg = pg.merge(wp, on=["season", "team", "pl", "k"], how="left").merge(wn, on=["season", "team", "k"], how="left")
        pg["share"] = pg.ws.fillna(0) / pg.wN
        pg = pg[(pg.k >= 4) & (pg.share >= 0.6) & (pg["cov"] >= 0.97)]
        pg = pg.merge(u[u.has][["game_id", "team", "i", "qtr", "sd"]].rename(columns={"i": "last", "qtr": "dq", "sd": "dsd"}), on=["game_id", "team", "last"], how="left")
        pg["rem"] = pg.N - 1 - pg["last"]
        dep = pg[(pg.rem >= 12) & (pg["last"] >= 5)].copy()
        dep["blow"] = dep.dsd.abs() >= 17
        dep["isqb"] = (unit == "off") & dep.pl.isin(qbs)
        excl = dep[dep.blow][["game_id", "team"]].drop_duplicates().assign(excl=1)
        dep = dep[~dep.blow]
        ok = n[n["cov"] >= 0.97][["game_id", "team"]]
        u = u.merge(ok, on=["game_id", "team"]).merge(excl, on=["game_id", "team"], how="left")
        u = u[u.excl.isna()].copy()
        u["game_id_c"] = u.game_id
        for kind, dd in (("qb", dep[dep.isqb]), ("other", dep[~dep.isqb])):
            m = u[["game_id", "team", "i"]].reset_index().merge(dd[["game_id", "team", "last"]], on=["game_id", "team"], how="left")
            m["c"] = (m["last"] < m.i).astype(int)
            u[f"post_{kind}"] = m.groupby("index").c.sum().reindex(u.index).fillna(0).clip(upper=1).values
        u["tgk"] = u.game_id + u.team
        res.append((unit, u, dep))
    rows = []
    for unit, u, dep in res:
        terms = ["post_qb", "post_other"] if unit == "off" else ["post_other"]
        cols = [c for c in CT if c not in ("off_epa", "opp_off", "own_def", "def_epa", "pos_home")]
        for name, (a, b) in (("fit_2016_20", (2016, 2020)), ("held_2021_25", (2021, 2025))):
            q = u[(u.season >= a) & (u.season <= b)]
            for y in ("epa", "succ"):
                LOOKS["b2"] += len(terms)
                s = q.dropna(subset=[y]).copy()
                X = s[terms + cols].astype(float)
                yy = s[y].astype(float)
                key = s.tgk.values
                X = X - X.groupby(key).transform("mean")
                yy = yy - yy.groupby(key).transform("mean")
                X = X.loc[:, X.std() > 0]
                bb, se, _ = cluster_ols(X, yy, s.game_id)
                nd = dep[(dep.season >= a) & (dep.season <= b)]
                for t in terms:
                    if t not in bb.index:
                        continue
                    kk = nd[nd.isqb] if t == "post_qb" else nd[~nd.isqb]
                    rows.append({"unit": unit, "term": t, "outcome": y, "era": name, "n": len(s), "n_departures": int(len(kk)), "n_unitgames_post": int(s.groupby("tgk")[t].max().sum()), "mean": float(s[y].mean()), "coef": bb[t], "se": se[t], "prob_pos": norm.cdf(bb[t] / se[t]), "lo": bb[t] - 1.96 * se[t], "hi": bb[t] + 1.96 * se[t]})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "dyn_b2.csv", index=False)
    return r


def coach_est(p, side, y, seasons=None, mn=300):
    s = p if seasons is None else p[p.season.isin(seasons)]
    s = s.dropna(subset=[y, side])
    cols = CT + ["second"]
    X = s[cols].astype(float).copy()
    X.insert(0, "const", 1.0)
    b, se, u = cluster_ols(X, s[y], s.game_id)
    s = s.assign(res=u)
    out = []
    for c, g in s.groupby(side):
        if len(g) < mn or g.second.nunique() < 2:
            continue
        x = g.second.values - g.second.mean()
        sxx = (x**2).sum()
        beta = (x * g.res.values).sum() / sxx
        r = g.res.values - g.res.mean() - beta * x
        sc = pd.DataFrame({"a": x * r, "g": g.game_id.values}).groupby("g").a.sum()
        out.append({"coach": c, "slope": beta, "se": np.sqrt((sc**2).sum()) / sxx, "n": len(g)})
    return b["second"], se["second"], pd.DataFrame(out)


def c1(p):
    sch = pd.read_parquet(SCHED, columns=["game_id", "home_coach", "away_coach"]).drop_duplicates("game_id")
    p = p.merge(sch, on="game_id", how="left")
    p["off_coach"] = np.where(p.posteam == p.home_team, p.home_coach, p.away_coach)
    p["def_coach"] = np.where(p.posteam == p.home_team, p.away_coach, p.home_coach)
    cov = float(p.off_coach.notna().mean())
    rows = []
    allc = []
    fitm = p[p.season <= 2017]
    heldm = p[p.season >= 2018]
    for side in ("off_coach", "def_coach"):
        for y in ("epa", "succ"):
            LOOKS["c1"] += 1
            pm, pse, a = coach_est(fitm, side, y)
            mu, tau, q, k = dl(a.slope.values, a.se.values)
            rel = tau**2 / (tau**2 + np.mean(a.se**2)) if tau > 0 else 0.0
            e = coach_est(fitm, side, y, [s for s in range(2009, 2018) if s % 2 == 0], 150)[2].set_index("coach")
            o = coach_est(fitm, side, y, [s for s in range(2009, 2018) if s % 2 == 1], 150)[2].set_index("coach")
            j = e.join(o, lsuffix="_e", rsuffix="_o", how="inner")
            r = np.corrcoef(j.slope_e, j.slope_o)[0, 1] if len(j) > 4 else np.nan
            bs = []
            for _ in range(2000):
                ix = RNG.integers(0, len(j), len(j))
                v = np.corrcoef(j.slope_e.values[ix], j.slope_o.values[ix])[0, 1]
                if v == v:
                    bs.append(v)
            bs = np.array(bs)
            hm, hse, h = coach_est(heldm, side, y, None, 150)
            a["shrunk"] = mu + (tau**2 / (tau**2 + a.se**2)) * (a.slope - mu)
            jj = a.set_index("coach").join(h.set_index("coach"), lsuffix="_f", rsuffix="_h", how="inner")
            if len(jj) > 4 and jj.shrunk.std() > 0:
                X = np.column_stack([np.ones(len(jj)), jj.shrunk.values - jj.shrunk.mean()])
                w = 1 / jj.se_h.values**2
                bh = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * jj.slope_h.values))
                cov_b = np.linalg.inv(X.T @ (X * w[:, None]))
                hs, hse2 = bh[1], np.sqrt(cov_b[1, 1])
                rh = np.corrcoef(jj.shrunk, jj.slope_h)[0, 1]
            else:
                hs = hse2 = rh = np.nan
            rows.append({"side": side, "outcome": y, "pooled_2H_minus_1H_fit": pm, "pooled_se": pse, "pooled_prob_pos": norm.cdf(pm / pse), "held_pooled": hm, "held_se": hse, "held_prob_pos": norm.cdf(hm / hse),
                         "n_coaches": k, "tau_sd": tau, "mean_se": float(np.mean(a.se)), "reliability": rel, "Q": q, "Q_df": k - 1, "n_pairs": len(j), "split_half_r": r, "r_lo": np.percentile(bs, 2.5), "r_hi": np.percentile(bs, 97.5), "prob_r_pos": float((bs > 0).mean()),
                         "spearman_brown": 2 * r / (1 + r) if r == r else np.nan, "heldout_n_coaches": len(jj), "heldout_slope_on_shrunk": hs, "heldout_slope_se": hse2, "heldout_slope_prob_pos": norm.cdf(hs / hse2) if hse2 == hse2 else np.nan, "heldout_r": rh})
            a["side"] = side
            a["outcome"] = y
            allc.append(a)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "dyn_c1.csv", index=False)
    pd.concat(allc).to_csv(OUT / "dyn_c1_coaches.csv", index=False)
    return r, cov


EVENTS = ["e_turnover", "e_missfg", "e_stop4", "e_oppscore", "e_ownscore", "e_ownbig"]


def d3(p, t):
    t = t.copy()
    t["prev_res"] = t.groupby("game_id").res.shift(1)
    t["prev_team"] = t.groupby("game_id").team.shift(1)
    t["own_prev_res"] = t.groupby(["game_id", "team"]).res.shift(1)
    t["own_prev_big"] = t.groupby(["game_id", "team"]).big.shift(1)
    t["prev_qtr"] = t.groupby("game_id").qtr.shift(1)
    half = (t.qtr >= 3).astype(int)
    prev_half = (t.prev_qtr >= 3).astype(int)
    same = t.prev_res.notna() & (half == prev_half) & (t.prev_team != t.team)
    t["e_turnover"] = (same & (t.prev_res == "Turnover")).astype(float)
    t["e_missfg"] = (same & (t.prev_res == "Missed field goal")).astype(float)
    t["e_stop4"] = (same & (t.prev_res == "Turnover on downs")).astype(float)
    t["e_oppscore"] = (same & t.prev_res.isin(["Touchdown", "Field goal"])).astype(float)
    ohalf = t.groupby(["game_id", "team"]).qtr.shift(1)
    sameown = t.own_prev_res.notna() & (half == (ohalf >= 3).astype(int))
    t["e_ownscore"] = (sameown & t.own_prev_res.isin(["Touchdown", "Field goal"])).astype(float)
    t["e_ownbig"] = (sameown & (t.own_prev_big == 1)).astype(float)
    p = p.merge(t[["game_id", "fixed_drive"] + EVENTS], on=["game_id", "fixed_drive"], how="left").dropna(subset=EVENTS)
    p["gid"] = p.game_id

    def fn(q):
        r = []
        for y in ("epa", "succ"):
            LOOKS["d3"] += len(EVENTS)
            rr, _, _ = fit(q, EVENTS, y)
            r += rr
        return r

    rows = by_era(p, fn)
    cnt = p.groupby(p.season.between(*HELD).map({True: "held", False: "fit"}))[EVENTS].sum()
    for r in rows:
        r["n_event_plays"] = int(cnt.loc[r["era"], r["term"]])
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "dyn_d3.csv", index=False)
    return r


def main():
    which = sys.argv[1:] or ["b1", "b2", "c1", "d3"]
    d = load()
    p = prplays(d)
    t = drive_table(d)
    summ = {"n_plays": int(len(p)), "which": which}
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    if "b1" in which:
        r = b1(p, t)
        print(r.round(4).to_string())
    if "b2" in which:
        r = b2(p, d)
        print(r.round(4).to_string())
    if "c1" in which:
        r, cov = c1(p)
        summ["coach_coverage"] = cov
        print(r.round(4).T.to_string())
    if "d3" in which:
        r = d3(p, t)
        print(r.round(4).to_string())
    summ["looks"] = LOOKS
    (OUT / f"dyn_summary_{'_'.join(which)}.json").write_text(json.dumps(summ, indent=1, default=float))
    print(summ)


if __name__ == "__main__":
    sys.exit(main())
