import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim09_urgency as U

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "sim09"
SCHED = ROOT / "data" / "raw" / "20260908T162105Z" / "schedules.parquet"
RNG = np.random.default_rng(20261003)
ALIAS = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
DIVS = {
    "AFC": [["BUF", "MIA", "NE", "NYJ"], ["BAL", "CIN", "CLE", "PIT"], ["HOU", "IND", "JAX", "TEN"], ["DEN", "KC", "LAC", "LV"]],
    "NFC": [["DAL", "NYG", "PHI", "WAS"], ["CHI", "DET", "GB", "MIN"], ["ATL", "CAR", "NO", "TB"], ["ARI", "LA", "SF", "SEA"]],
}
TEAMS = sorted(t for c in DIVS.values() for d in c for t in d)
TI = {t: i for i, t in enumerate(TEAMS)}
CONF = {c: [TI[t] for d in ds for t in d] for c, ds in DIVS.items()}
DIVIDX = [[TI[t] for t in d] for c in DIVS.values() for d in c]
NSIM = 6000
TRAIN = range(2009, 2018)
LOOKS = {"play": 0, "drive": 0, "fourth": 0, "game": 0}


def norm_team(s):
    return s.replace(ALIAS)


def load_sched():
    s = pd.read_parquet(SCHED)
    s = s[(s.season >= 2009) & (s.season <= 2025)].copy()
    s["home_team"] = norm_team(s.home_team)
    s["away_team"] = norm_team(s.away_team)
    reg = s[s.game_type == "REG"].copy().sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    reg["hi"] = reg.home_team.map(TI)
    reg["ai"] = reg.away_team.map(TI)
    reg["margin"] = reg.home_score - reg.away_score
    reg["neutral"] = (reg.location == "Neutral").astype(float)
    po = s[s.game_type.isin(["WC", "DIV"])]
    field = {}
    for se, g in po.groupby("season"):
        field[se] = set(g.home_team) | set(g.away_team)
    return reg, field


def strength_tables(reg):
    out = {}
    last = {}
    for se, g in reg.groupby("season"):
        tot = np.zeros(32)
        n = np.zeros(32)
        for h, a, m in zip(g.hi, g.ai, g.margin):
            tot[h] += m
            tot[a] -= m
            n[h] += 1
            n[a] += 1
        last[se] = np.where(n > 0, tot / np.maximum(n, 1), 0.0)
        for w in range(1, int(g.week.max()) + 2):
            p = g[g.week < w]
            S = np.zeros(32)
            N = np.zeros(32)
            np.add.at(S, p.hi.values, p.margin.values)
            np.add.at(S, p.ai.values, -p.margin.values)
            np.add.at(N, p.hi.values, 1)
            np.add.at(N, p.ai.values, 1)
            out[(se, w)] = (S, N)
    return out, last


def rating(tabs, last, se, w, k, c):
    S, N = tabs[(se, w)]
    prior = c * last.get(se - 1, np.zeros(32))
    return (S + k * prior) / (N + k)


def fit_strength(reg, tabs, last):
    tr = reg[reg.season.isin(TRAIN) & (reg.week >= 2)]
    best = None
    rows = []
    for k in [2, 4, 8, 12, 16, 24]:
        for c in [0.0, 0.25, 0.5, 0.75, 1.0]:
            xs = np.concatenate([np.column_stack([1 - g.neutral.values, rating(tabs, last, se, w, k, c)[g.hi.values] - rating(tabs, last, se, w, k, c)[g.ai.values]]) for (se, w), g in tr.groupby(["season", "week"], sort=False)])
            yy = np.concatenate([g.margin.values for _, g in tr.groupby(["season", "week"], sort=False)])
            b, *_ = np.linalg.lstsq(xs, yy, rcond=None)
            mse = float(np.mean((yy - xs @ b) ** 2))
            rows.append({"k": k, "c": c, "hfa": b[0], "slope": b[1], "sigma": mse**0.5})
            if best is None or mse < best[0]:
                best = (mse, k, c, b[0], b[1])
    pd.DataFrame(rows).to_csv(OUT / "stakes_rating_grid.csv", index=False)
    return {"k": best[1], "c": best[2], "hfa": float(best[3]), "slope": float(best[4]), "sigma": float(best[0] ** 0.5)}


def eval_strength(reg, tabs, last, P):
    rows = []
    for name, ys in [("train_2009_17", range(2009, 2018)), ("heldout_2018_25", range(2018, 2026))]:
        g0 = reg[reg.season.isin(ys) & (reg.week >= 2) & reg.margin.notna() & (reg.margin != 0)]
        pr = []
        for (se, w), g in g0.groupby(["season", "week"], sort=False):
            r = rating(tabs, last, se, w, P["k"], P["c"])
            e = P["hfa"] * (1 - g.neutral.values) + P["slope"] * (r[g.hi.values] - r[g.ai.values])
            pr.append(pd.DataFrame({"p": norm.cdf(e / P["sigma"]), "y": (g.margin.values > 0).astype(float), "m": g.margin.values, "e": e, "sp": g.spread_line.values}))
        d = pd.concat(pr)
        ll = -np.mean(d.y * np.log(d.p) + (1 - d.y) * np.log(1 - d.p))
        mk = d.dropna(subset=["sp"])
        rows.append({"sample": name, "n": len(d), "logloss": ll, "brier": float(np.mean((d.p - d.y) ** 2)), "mse_margin": float(np.mean((d.m - d.e) ** 2)), "mse_market": float(np.mean((mk.m - mk.sp) ** 2)), "mse_model_on_market_rows": float(np.mean((mk.m - mk.e) ** 2)), "base_logloss_p_half": float(np.log(2))})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "stakes_rating_eval.csv", index=False)
    return r


def seeds_from_wins(wins, K):
    N = wins.shape[0]
    score = wins + 0.4 * RNG.random((N, 32))
    isw = np.zeros((N, 32), bool)
    for ids in DIVIDX:
        a = np.argmax(score[:, ids], axis=1)
        isw[np.arange(N), np.array(ids)[a]] = True
    seed = np.zeros((N, 32), np.int8)
    for cols in CONF.values():
        c = np.array(cols)
        sw = np.where(isw[:, c], score[:, c], -np.inf)
        rw = np.argsort(np.argsort(-sw, axis=1), axis=1)
        sl = np.where(~isw[:, c], score[:, c], -np.inf)
        rl = np.argsort(np.argsort(-sl, axis=1), axis=1)
        s = np.where(isw[:, c], rw + 1, np.where(rl < K - 4, rl + 5, 0))
        seed[:, c] = s
    return seed


def simulate(g_all, se, w, tabs, last, P, n):
    K = 6 if se <= 2019 else 7
    played = g_all[g_all.week < w]
    rem = g_all[g_all.week >= w]
    W0 = np.zeros(32)
    for h, a, m in zip(played.hi, played.ai, played.margin):
        W0[h] += 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
        W0[a] += 1.0 if m < 0 else (0.5 if m == 0 else 0.0)
    wins = np.tile(W0, (n, 1))
    X = np.zeros((n, 0), bool)
    if len(rem):
        r = rating(tabs, last, se, w, P["k"], P["c"])
        e = P["hfa"] * (1 - rem.neutral.values) + P["slope"] * (r[rem.hi.values] - r[rem.ai.values])
        ph = norm.cdf(e / P["sigma"])
        X = RNG.random((n, len(rem))) < ph
        Hm = np.zeros((len(rem), 32))
        Am = np.zeros((len(rem), 32))
        Hm[np.arange(len(rem)), rem.hi.values] = 1
        Am[np.arange(len(rem)), rem.ai.values] = 1
        wins = wins + X @ Hm + (~X) @ Am
    seed = seeds_from_wins(wins, K)
    return seed, X, rem, K


def team_weeks(reg, tabs, last, P):
    rows = []
    fieldp = {}
    for se, g in reg.groupby("season"):
        g = g.reset_index(drop=True)
        mw = int(g.week.max())
        seed, _, _, K = simulate(g, se, mw + 1, tabs, last, P, 400)
        fieldp[se] = (seed > 0).mean(axis=0)
        prevqb = {}
        for w in range(1, mw + 1):
            seed, X, rem, K = simulate(g, se, w, tabs, last, P, NSIM)
            mk = seed > 0
            pmk = mk.mean(axis=0)
            pk = np.stack([(seed == s).mean(axis=0) for s in range(K + 1)], axis=1)
            r = rating(tabs, last, se, w, P["k"], P["c"])
            cur = rem[rem.week == w]
            for j, (idx, gm) in enumerate(cur.iterrows()):
                col = list(rem.index).index(idx)
                x = X[:, col]
                for side, t, o, win in [("home", gm.hi, gm.ai, x), ("away", gm.ai, gm.hi, ~x)]:
                    a = mk[win, t].mean() if win.any() else np.nan
                    b = mk[~win, t].mean() if (~win).any() else np.nan
                    lev = a - b if a == a and b == b else 0.0
                    qb = gm.home_qb_id if side == "home" else gm.away_qb_id
                    rows.append({"season": se, "week": w, "team": TEAMS[t], "opp": TEAMS[o], "home": float(side == "home"), "lastweek": float(w == mw), "p_make": pmk[t], "lev": lev, "p_win_cond": a, "p_loss_cond": b, "elim": float(pmk[t] < 0.01), "elim_zero": float(pmk[t] == 0.0), "clinch": float(pmk[t] > 0.99), "clinch_full": float(pmk[t] == 1.0), "locked": float(pk[t].argmax() > 0 and pk[t].max() >= 0.99), "seed_modal": int(pk[t].argmax()), "p_seed_modal": float(pk[t].max()), "p_seed1": float(pk[t, 1]), "r_own": r[t], "r_opp": r[o], "qb": qb})
    tw = pd.DataFrame(rows).sort_values(["season", "team", "week"])
    tw["qb_prev"] = tw.groupby(["season", "team"]).qb.shift(1)
    tw["qb_absent"] = ((tw.qb != tw.qb_prev) & tw.qb_prev.notna() & tw.qb.notna()).astype(float)
    tw["rest_last"] = tw.qb_absent * tw.lastweek
    return tw.drop(columns=["qb_prev"]), fieldp


def validate(tw, field, fieldp):
    rows = []
    for se, p in sorted(fieldp.items()):
        pred = {TEAMS[i] for i in np.argsort(-p)[: (6 if se <= 2019 else 7) * 2]}
        act = field.get(se, set())
        rows.append({"season": se, "actual": len(act), "predicted": len(pred), "overlap": len(pred & act), "missed": sorted(act - pred), "extra": sorted(pred - act), "brier_final": float(np.mean([(p[TI[t]] - (t in act)) ** 2 for t in TEAMS]))})
    v = pd.DataFrame(rows)
    v.to_csv(OUT / "stakes_validate_final.csv", index=False)
    tw = tw.copy()
    tw["made"] = [float(t in field.get(se, set())) for se, t in zip(tw.season, tw.team)]
    cal = []
    for lo, hi in [(0, .01), (.01, .05), (.05, .15), (.15, .35), (.35, .65), (.65, .85), (.85, .95), (.95, .99), (.99, 1.01)]:
        for name, sub in [("week_1_8", tw[tw.week <= 8]), ("week_9_plus", tw[tw.week >= 9]), ("all", tw)]:
            s = sub[(sub.p_make >= lo) & (sub.p_make < hi)]
            cal.append({"bin": f"{lo}-{hi}", "sample": name, "n": len(s), "mean_p": s.p_make.mean(), "made": s.made.mean()})
    pd.DataFrame(cal).to_csv(OUT / "stakes_reliability.csv", index=False)
    st = {}
    late = tw[tw.week >= 9]
    for name in ["elim", "elim_zero", "clinch", "clinch_full"]:
        s = tw[tw[name] == 1]
        st[name] = {"n": int(len(s)), "made_rate": float(s.made.mean()) if len(s) else None, "share_of_week9plus_teamweeks": float((late[name] == 1).mean())}
    st["brier_p_make"] = float(np.mean((tw.p_make - tw.made) ** 2))
    st["brier_p_make_week9plus"] = float(np.mean((late.p_make - late.made) ** 2))
    st["brier_base_rate_week9plus"] = float(np.mean((late.made.mean() - late.made) ** 2))
    st["locked_n"] = int((tw.locked == 1).sum())
    st["lev_mean_week14plus"] = float(tw[tw.week >= 14].lev.mean())
    st["lev_sd_week14plus"] = float(tw[tw.week >= 14].lev.std())
    return v, st


def stake_cols(d, tw):
    k = ["season", "week", "team"]
    o = tw[k + ["p_make", "lev", "elim", "locked", "clinch", "r_own", "r_opp"]]
    own = o.rename(columns={"team": "posteam", "p_make": "p_own", "lev": "lev_own", "elim": "elim_own", "locked": "seedlock_own", "clinch": "lock_own", "r_own": "r_own", "r_opp": "r_opp"})
    d = d.merge(own, on=["season", "week", "posteam"], how="left")
    opp = o[k + ["lev", "elim", "locked", "clinch"]].rename(columns={"team": "defteam", "lev": "lev_opp", "elim": "elim_opp", "locked": "seedlock_opp", "clinch": "lock_opp"})
    d = d.merge(opp, on=["season", "week", "defteam"], how="left")
    for s in ["lev", "elim", "lock"]:
        d[f"{s}_own_zt"] = d[f"{s}_own"] * d.z_trail
        d[f"{s}_own_zl"] = d[f"{s}_own"] * d.z_lead
    d["late"] = (d.week >= 13).astype(float)
    return d


STK = ["lev_own", "lev_opp", "elim_own", "elim_opp", "lock_own", "lock_opp", "lev_own_zt", "lev_own_zl", "elim_own_zt", "elim_own_zl", "lock_own_zt", "lock_own_zl"]
EXTRA = ["late", "r_own", "r_opp"]
BASE = U.MAIN + EXTRA


def fit_block(df, outcomes, kind, extra_terms=()):
    rows = []
    s0 = df.dropna(subset=U.CTRL + ["z", "ht_trail"] + STK + EXTRA)
    for o in outcomes:
        s = s0.dropna(subset=[o])
        terms = BASE + STK + list(extra_terms)
        b, se, _ = U.cluster_ols(U.design(s, terms), s[o], s.game_id)
        sd = s[o].std()
        for t in STK:
            LOOKS[kind] += 1
            rows.append({"level": kind, "outcome": o, "term": t, "n": len(s), "mean": s[o].mean(), "sd": sd, "coef": b[t], "se": se[t], "prob_pos": norm.cdf(b[t] / se[t]), "lo": b[t] - 1.96 * se[t], "hi": b[t] + 1.96 * se[t], "std_coef": b[t] / sd})
    return rows


def play_level(tw):
    d = U.load()
    d = U.controls(d)
    d = U.situation(d, U.halftime())
    d["posteam"] = norm_team(d.posteam)
    d["defteam"] = norm_team(d.defteam)
    d = stake_cols(d, tw)
    d = d.dropna(subset=U.CTRL + ["z", "ht_trail", "lev_own", "lev_opp"])
    X0 = U.design(d, BASE + STK)
    _, _, u0 = U.cluster_ols(X0, d.ypp, d.game_id)
    d["risk"] = np.abs(u0)
    rows = fit_block(d, ["shotgun", "no_huddle", "air", "intc", "ypp", "risk", "pass_oe"], "play")
    g = d.sort_values(["game_id", "drive"]).groupby(["game_id", "drive"])
    agg = {c: (c, "first") for c in ["posteam", "season", "week", "t", "sd", "qtr", "pos_home", "off_epa", "opp_off", "own_def", "def_epa", "ht_trail", "second", "z", "z_trail", "z_lead", "late", "r_own", "r_opp"] + STK}
    dr = g.agg(plays=("ypp", "size"), res=("fixed_drive_result", "first"), **agg).reset_index()
    dr["td"] = (dr.res == "Touchdown").astype(float)
    rows += fit_block(dr, ["td", "plays"], "drive")
    return d, rows


def fourth_down(tw, d_pr):
    cols = U.COLS + ["down", "ydstogo", "yardline_100"]
    f = pd.concat([pd.read_parquet(U.NV / f"pbp_{s}.parquet", columns=cols) for s in range(2009, 2018)], ignore_index=True)
    f = f[(f.season_type == "REG") & (f.qtr <= 4) & f.posteam.notna() & f.score_differential.notna() & (f.down == 4) & f.play_type.isin(["pass", "run", "punt", "field_goal"])].copy()
    f = f[(f.qb_kneel != 1) & (f.qb_spike != 1)]
    for c in ["posteam", "defteam", "home_team", "away_team"]:
        f[c] = norm_team(f[c])
    f["t"] = f.game_seconds_remaining.clip(lower=0)
    off = U.team_prior(d_pr.assign(), "posteam", "off_epa")
    dfn = U.team_prior(d_pr.assign(), "defteam", "def_epa")
    f["pos_home"] = (f.posteam == f.home_team).astype(float)
    f = f.merge(off, on=["season", "week", "posteam"], how="left").merge(dfn, on=["season", "week", "defteam"], how="left")
    f = f.merge(off.rename(columns={"posteam": "defteam", "off_epa": "opp_off"}), on=["season", "week", "defteam"], how="left")
    f = f.merge(dfn.rename(columns={"defteam": "posteam", "def_epa": "own_def"}), on=["season", "week", "posteam"], how="left")
    f = U.situation(f, U.halftime())
    f = stake_cols(f, tw)
    f["go"] = f.play_type.isin(["pass", "run"]).astype(float)
    f["yd"] = f.ydstogo.clip(upper=15).astype(float)
    f["yl"] = f.yardline_100.astype(float)
    f["yl2"] = f.yl**2 / 100.0
    f["fgr"] = (f.yl <= 40).astype(float)
    f = f.dropna(subset=U.CTRL + ["z", "ht_trail", "lev_own", "lev_opp", "yd", "yl"])
    rows = fit_block(f, ["go"], "fourth", ["yd", "yl", "yl2", "fgr"])
    return rows, len(f), float(f.go.mean())


def game_level(reg, tw, P, tabs, last, cut):
    k = ["season", "week", "team"]
    cols = k + ["lev", "elim", "clinch", "p_make", "rest_last", "r_own", "r_opp"]
    h = tw[cols].rename(columns={"team": "home_team", **{c: c + "_h" for c in cols[3:]}})
    a = tw[cols].rename(columns={"team": "away_team", **{c: c + "_a" for c in cols[3:]}})
    g = reg.merge(h, on=["season", "week", "home_team"], how="inner").merge(a, on=["season", "week", "away_team"], how="inner")
    g = g[g.margin.notna() & (g.week >= 10)].copy()
    g["exp_model"] = P["hfa"] * (1 - g.neutral) + P["slope"] * (g.r_own_h - g.r_own_a)
    g["res_model"] = g.margin - g.exp_model
    g["res_market"] = g.margin - g.spread_line
    for s in ["h", "a"]:
        g[f"A_{s}"] = (g[f"lev_{s}"] >= cut).astype(float)
        g[f"Belim_{s}"] = g[f"elim_{s}"]
        g[f"Block_{s}"] = g[f"clinch_{s}"]
        g[f"Brest_{s}"] = g[f"rest_last_{s}"]
        g[f"B_{s}"] = ((g[f"elim_{s}"] + g[f"clinch_{s}"] + g[f"rest_last_{s}"]) > 0).astype(float)
    mm = {"any": "B", "elim": "Belim", "locked": "Block", "rest": "Brest"}
    for name, p in mm.items():
        g[f"m_{name}"] = g.A_h * g[f"{p}_a"] * (1 - g.A_a) - g.A_a * g[f"{p}_h"] * (1 - g.A_h)
    g["levdiff"] = g.lev_h - g.lev_a
    g["sample"] = np.where(g.season <= 2017, "fit_2009_17", "heldout_2018_25")
    g["grp"] = g.season.astype(str) + "_" + g.week.astype(str)
    return g


def game_fits(g):
    rows = []
    fitb = {}
    for samp, s in g.groupby("sample"):
        for rname in ["res_model", "res_market"]:
            ss = s.dropna(subset=[rname])
            for m in ["m_any", "m_elim", "m_locked", "m_rest", "levdiff"]:
                LOOKS["game"] += 1
                X = pd.DataFrame({"const": 1.0, m: ss[m].values})
                b, se, _ = U.cluster_ols(X, ss[rname].reset_index(drop=True), ss.grp.reset_index(drop=True))
                n_m = int((ss[m] != 0).sum()) if m != "levdiff" else len(ss)
                rows.append({"sample": samp, "resid": rname, "term": m, "n": len(ss), "n_nonzero": n_m, "coef": b[m], "se": se[m], "prob_pos": norm.cdf(b[m] / se[m]), "lo": b[m] - 1.96 * se[m], "hi": b[m] + 1.96 * se[m]})
                fitb[(samp, rname, m)] = b[m]
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "stakes_game_mismatch.csv", index=False)
    ho = g[g["sample"] == "heldout_2018_25"]
    chk = []
    for rname in ["res_model", "res_market"]:
        for m in ["m_any", "m_elim", "m_locked", "m_rest", "levdiff"]:
            bf = fitb[("fit_2009_17", rname, m)]
            y = ho[rname].dropna()
            x = ho.loc[y.index, m]
            chk.append({"resid": rname, "term": m, "fit_coef": bf, "mse_no_term": float(np.mean(y**2)), "mse_with_fit_term": float(np.mean((y - bf * x) ** 2))})
    c = pd.DataFrame(chk)
    c.to_csv(OUT / "stakes_game_heldout_check.csv", index=False)
    return r, c


def main():
    reg, field = load_sched()
    tabs, last = strength_tables(reg)
    P = fit_strength(reg, tabs, last)
    print("rating params", P)
    ev = eval_strength(reg, tabs, last, P)
    print(ev.round(4).to_string())
    miss = float(reg.home_qb_id.isna().mean())
    tw, fieldp = team_weeks(reg, tabs, last, P)
    tw.to_csv(OUT / "stakes_teamweek.csv", index=False)
    v, st = validate(tw, field, fieldp)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    print(v.to_string())
    print(json.dumps(st, indent=1))
    d, rows = play_level(tw)
    r4, n4, go = fourth_down(tw, d)
    rows += r4
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "stakes_behaviour.csv", index=False)
    tr = tw[(tw.week >= 10) & (tw.season <= 2017) & (tw.elim == 0) & (tw.clinch == 0)]
    cut = float(tr.lev.quantile(2 / 3))
    g = game_level(reg, tw, P, tabs, last, cut)
    gr, gc = game_fits(g)
    summ = {"rating_params": P, "validation": st, "qb_missing_share": miss, "leverage_cut": cut, "looks": LOOKS, "n_play_rows": int(len(d)), "n_fourth_rows": n4, "fourth_go_rate": go, "game_counts": g.groupby("sample").agg(n=("margin", "size"), m_any=("m_any", lambda s: int((s != 0).sum())), m_elim=("m_elim", lambda s: int((s != 0).sum())), m_locked=("m_locked", lambda s: int((s != 0).sum())), m_rest=("m_rest", lambda s: int((s != 0).sum()))).reset_index().to_dict("records")}
    (OUT / "stakes_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))
    print(res[res.term.isin(STK)].round(4).to_string())
    print(gr.round(4).to_string())
    print(gc.round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
