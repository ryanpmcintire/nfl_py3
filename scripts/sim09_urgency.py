import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

NV = Path("C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad/nv")
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "sim09"
OUT.mkdir(parents=True, exist_ok=True)
COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team", "qtr",
        "game_seconds_remaining", "score_differential", "play_type", "air_yards",
        "shotgun", "no_huddle", "xpass", "epa", "yards_gained", "sack", "interception", "fumble_lost",
        "qb_dropback", "qb_kneel", "qb_spike", "drive", "fixed_drive_result", "home_coach", "away_coach"]
RNG = np.random.default_rng(20261002)
LOOKS = {"coef": 0, "ref": 0, "coach": 0, "drive": 0, "late": 0}


def load():
    d = pd.concat([pd.read_parquet(NV / f"pbp_{s}.parquet", columns=COLS) for s in range(2009, 2018)], ignore_index=True)
    d = d[(d.season_type == "REG") & (d.qtr <= 4) & d.posteam.notna() & d.score_differential.notna()].copy()
    d = d[d.play_type.isin(["pass", "run"]) & (d.qb_kneel != 1) & (d.qb_spike != 1)].copy()
    d["t"] = d.game_seconds_remaining.clip(lower=0)
    return d


def halftime():
    raw = pd.concat([pd.read_parquet(NV / f"pbp_{s}.parquet", columns=["game_id", "qtr", "total_home_score", "total_away_score", "season_type"]) for s in range(2009, 2018)])
    raw = raw[(raw.qtr == 3) & (raw.season_type == "REG")].groupby("game_id").first()
    return (raw.total_home_score - raw.total_away_score).rename("hd_home")


def team_prior(d, team_col, name):
    g = d.groupby(["season", "week", team_col]).agg(oe=("epa", "sum"), on=("epa", "size")).reset_index()
    g = g.sort_values(["season", team_col, "week"])
    g["co"] = g.groupby(["season", team_col]).oe.cumsum() - g.oe
    g["cn"] = g.groupby(["season", team_col]).on.cumsum() - g.on
    g[name] = g.co / (g.cn + 300.0)
    return g[["season", "week", team_col, name]]


def controls(d):
    d["pos_home"] = (d.posteam == d.home_team).astype(float)
    off = team_prior(d, "posteam", "off_epa")
    dfn = team_prior(d, "defteam", "def_epa")
    d = d.merge(off, on=["season", "week", "posteam"], how="left")
    d = d.merge(dfn, on=["season", "week", "defteam"], how="left")
    opp_off = off.rename(columns={"posteam": "defteam", "off_epa": "opp_off"})
    own_def = dfn.rename(columns={"defteam": "posteam", "def_epa": "own_def"})
    d = d.merge(opp_off, on=["season", "week", "defteam"], how="left")
    d = d.merge(own_def, on=["season", "week", "posteam"], how="left")
    return d


def situation(d, hd):
    d = d.merge(hd, on="game_id", how="left")
    d["sd"] = d.score_differential.astype(float)
    d["z"] = d.sd / np.sqrt(d.t / 60.0 + 1.0)
    d["z_trail"] = d.z.clip(upper=0)
    d["z_lead"] = d.z.clip(lower=0)
    d["second"] = (d.qtr >= 3).astype(float)
    hd_pos = np.where(d.posteam == d.home_team, d.hd_home, -d.hd_home)
    d["ht_trail"] = d.second * (-np.minimum(hd_pos, 0) / 10.0)
    d["late"] = (d.week >= 13).astype(float)
    d["coach"] = np.where(d.posteam == d.home_team, d.home_coach, d.away_coach)
    d["dropback"] = (d.qb_dropback == 1).astype(float)
    d["pass_oe"] = d.dropback - d.xpass
    d["air"] = np.where((d.dropback == 1) & (d.sack != 1) & d.air_yards.notna(), d.air_yards, np.nan)
    d["deep"] = np.where(d.air.notna(), (d.air >= 20).astype(float), np.nan)
    d["intc"] = np.where((d.dropback == 1) & (d.sack != 1), d.interception.astype(float), np.nan)
    d["sackr"] = np.where(d.dropback == 1, d.sack.astype(float), np.nan)
    d["expl"] = np.where(d.dropback == 1, (d.yards_gained >= 20).astype(float), (d.yards_gained >= 10).astype(float))
    d["fum"] = d.fumble_lost.astype(float)
    d["ypp"] = d.yards_gained.astype(float)
    d["shotgun"] = d.shotgun.astype(float)
    d["no_huddle"] = d.no_huddle.astype(float)
    return d


CTRL = ["pos_home", "off_epa", "opp_off", "own_def", "def_epa"]


def design(d, terms):
    X = d[terms + CTRL].astype(float).copy()
    X.insert(0, "const", 1.0)
    return X


def cluster_ols(X, y, g):
    Xa = X.values
    ya = y.values.astype(float)
    xtx = np.linalg.inv(Xa.T @ Xa)
    b = xtx @ Xa.T @ ya
    u = ya - Xa @ b
    s = pd.DataFrame(Xa * u[:, None]).groupby(g.values).sum().values
    V = xtx @ (s.T @ s) @ xtx
    n, k = Xa.shape
    ng = s.shape[0]
    V = V * (ng / (ng - 1)) * ((n - 1) / (n - k))
    return pd.Series(b, index=X.columns), pd.Series(np.sqrt(np.diag(V)), index=X.columns), u


OUTCOMES = ["pass_oe", "shotgun", "no_huddle", "air", "deep", "intc", "sackr", "fum", "expl", "ypp", "risk", "epa"]
MAIN = ["z_trail", "z_lead", "ht_trail", "second"]
REFS = {"q3_lead7": (7.0, 1800.0), "q4start_trail10": (-10.0, 900.0), "final2_trail7": (-7.0, 120.0), "q4start_trail3": (-3.0, 900.0), "q4_lead7_5min": (7.0, 300.0)}


def fit_all(d):
    rows = []
    refrows = []
    base = d.dropna(subset=CTRL + ["z", "ht_trail"]).copy()
    X0 = design(base, MAIN)
    _, _, u0 = cluster_ols(X0, base.ypp, base.game_id)
    base["risk"] = np.abs(u0)
    for o in OUTCOMES:
        s = base.dropna(subset=[o])
        X = design(s, MAIN)
        b, se, _ = cluster_ols(X, s[o], s.game_id)
        for t in MAIN:
            LOOKS["coef"] += 1
            rows.append({"outcome": o, "term": t, "n": len(s), "mean": s[o].mean(), "sd": s[o].std(), "coef": b[t], "se": se[t], "prob_pos": norm.cdf(b[t] / se[t]), "lo": b[t] - 1.96 * se[t], "hi": b[t] + 1.96 * se[t], "std_coef": b[t] / s[o].std()})
        for name, (sdv, tv) in REFS.items():
            z = sdv / np.sqrt(tv / 60 + 1)
            t = "z_trail" if z < 0 else "z_lead"
            LOOKS["ref"] += 1
            e = b[t] * z
            w = 1.96 * se[t] * abs(z)
            refrows.append({"outcome": o, "ref": name, "z": z, "effect": e, "se": se[t] * abs(z), "lo": e - w, "hi": e + w, "prob_pos": norm.cdf(e / (se[t] * abs(z))), "mean": s[o].mean(), "std_effect": e / s[o].std()})
    pd.DataFrame(rows).to_csv(OUT / "response_slopes.csv", index=False)
    pd.DataFrame(refrows).to_csv(OUT / "ref_effects.csv", index=False)
    return base, pd.DataFrame(rows), pd.DataFrame(refrows)


def late_early(base):
    rows = []
    s0 = base.copy()
    for k in ["z_trail", "z_lead", "ht_trail"]:
        s0["L_" + k] = s0[k] * s0.late
    terms = MAIN + ["late", "L_z_trail", "L_z_lead", "L_ht_trail"]
    for o in OUTCOMES:
        s = s0.dropna(subset=[o])
        b, se, _ = cluster_ols(design(s, terms), s[o], s.game_id)
        for t in ["L_z_trail", "L_z_lead", "L_ht_trail"]:
            LOOKS["late"] += 1
            rows.append({"outcome": o, "term": t, "coef": b[t], "se": se[t], "prob_pos": norm.cdf(b[t] / se[t]), "lo": b[t] - 1.96 * se[t], "hi": b[t] + 1.96 * se[t], "std_coef": b[t] / s[o].std()})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "late_vs_early.csv", index=False)
    return r


def surface(base):
    sdb = pd.cut(base.sd, [-100, -15, -9, -4, -1, 0, 3, 8, 14, 100], labels=["<=-15", "-14..-9", "-8..-4", "-3..-1", "0", "1..3", "4..8", "9..14", ">=15"])
    tb = pd.cut(base.t, [-1, 120, 480, 900, 1800, 1920, 2700, 3601], labels=["Q4<2m", "Q4 2-8m", "Q4>8m", "Q3", "Q2<2m", "Q2>2m", "Q1"])
    base = base.assign(sdb=sdb, tb=tb)
    agg = base.groupby(["sdb", "tb"], observed=True)[["pass_oe", "air", "deep", "shotgun", "no_huddle", "sackr", "intc", "fum", "expl", "ypp", "risk", "epa"]].mean()
    agg["n"] = base.groupby(["sdb", "tb"], observed=True).size()
    agg.to_csv(OUT / "binned_surface.csv")
    return len(agg) * 12


def dl(slopes, ses):
    w = 1 / ses**2
    mu = (w * slopes).sum() / w.sum()
    q = (w * (slopes - mu) ** 2).sum()
    c = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0.0, (q - (len(slopes) - 1)) / c)
    return mu, np.sqrt(tau2), q, len(slopes)


def coach_slopes(base, o, half=None):
    s = base.dropna(subset=[o, "coach"])
    if half is not None:
        s = s[(s.season % 2) == half]
    X = design(s, ["z_lead", "ht_trail", "second"])
    _, _, u = cluster_ols(X, s[o], s.game_id)
    s = s.assign(res=u)
    out = []
    mn = 150 if half is None else 75
    for c, g in s.groupby("coach"):
        nt = int((g.z_trail < 0).sum())
        if nt < mn:
            continue
        x = g.z_trail.values
        xc = x - x.mean()
        sxx = (xc**2).sum()
        beta = (xc * g.res.values).sum() / sxx
        r = g.res.values - g.res.mean() - beta * xc
        sc = pd.DataFrame({"a": xc * r, "g": g.game_id.values}).groupby("g").a.sum()
        se = np.sqrt((sc**2).sum()) / sxx
        out.append({"coach": c, "slope": beta, "se": se, "n_trail": nt})
    return pd.DataFrame(out)


def coach_het(base):
    rows = []
    allc = []
    for o in ["pass_oe", "air", "deep", "shotgun", "no_huddle"]:
        LOOKS["coach"] += 1
        a = coach_slopes(base, o)
        mu, tau, q, k = dl(a.slope.values, a.se.values)
        a["outcome"] = o
        allc.append(a)
        e = coach_slopes(base, o, 0).set_index("coach")
        f = coach_slopes(base, o, 1).set_index("coach")
        j = e.join(f, lsuffix="_e", rsuffix="_o", how="inner")
        r = np.corrcoef(j.slope_e, j.slope_o)[0, 1] if len(j) > 4 else np.nan
        bs = []
        for _ in range(2000):
            ix = RNG.integers(0, len(j), len(j))
            bs.append(np.corrcoef(j.slope_e.values[ix], j.slope_o.values[ix])[0, 1])
        bs = np.array(bs)
        bs = bs[~np.isnan(bs)]
        rel = tau**2 / (tau**2 + np.mean(a.se**2)) if tau > 0 else 0.0
        rows.append({"outcome": o, "n_coaches": k, "pooled_slope": mu, "tau_between_sd": tau, "Q": q, "Q_df": k - 1, "mean_se": float(np.mean(a.se)), "reliability": rel, "n_pairs": len(j), "split_half_r": r, "r_lo": np.percentile(bs, 2.5), "r_hi": np.percentile(bs, 97.5), "prob_r_pos": float((bs > 0).mean()), "spearman_brown": 2 * r / (1 + r) if r == r else np.nan})
    pd.concat(allc).to_csv(OUT / "coach_slopes.csv", index=False)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "coach_het.csv", index=False)
    return r


def drives(base):
    g = base.sort_values(["game_id", "drive"]).groupby(["game_id", "drive"])
    dr = g.agg(posteam=("posteam", "first"), season=("season", "first"), t=("t", "first"), sd=("sd", "first"), qtr=("qtr", "first"),
               plays=("ypp", "size"), yds=("ypp", "sum"), res=("fixed_drive_result", "first"), pos_home=("pos_home", "first"), off_epa=("off_epa", "first"),
               opp_off=("opp_off", "first"), own_def=("own_def", "first"), def_epa=("def_epa", "first"), ht_trail=("ht_trail", "first"), second=("second", "first")).reset_index()
    dr["game_id"] = dr.game_id
    dr["td"] = (dr.res == "Touchdown").astype(float)
    dr["fg"] = (dr.res == "Field goal").astype(float)
    dr["pts"] = 7 * dr.td + 3 * dr.fg
    dr["ypd"] = dr.yds / dr.plays
    dr["z"] = dr.sd / np.sqrt(dr.t / 60 + 1)
    dr["z_trail"] = dr.z.clip(upper=0)
    dr["z_lead"] = dr.z.clip(lower=0)
    dr = dr.dropna(subset=CTRL + ["z"])
    real = {
        "q3_leader_td": float(dr[(dr.qtr == 3) & (dr.sd > 0)].td.mean()),
        "q3_leader_n": int(((dr.qtr == 3) & (dr.sd > 0)).sum()),
        "final2_trailer_plays": float(dr[(dr.t <= 120) & (dr.sd < 0)].plays.mean()),
        "final2_trailer_pts": float(dr[(dr.t <= 120) & (dr.sd < 0)].pts.mean()),
        "final2_trailer_n": int(((dr.t <= 120) & (dr.sd < 0)).sum()),
    }
    rows = []
    for o in ["td", "pts", "plays", "ypd"]:
        X = design(dr, MAIN)
        b, se, _ = cluster_ols(X, dr[o], dr.game_id)
        for name, (sdv, tv) in REFS.items():
            z = sdv / np.sqrt(tv / 60 + 1)
            t = "z_trail" if z < 0 else "z_lead"
            LOOKS["drive"] += 1
            rows.append({"outcome": o, "ref": name, "effect": b[t] * z, "se": se[t] * abs(z), "prob_pos": norm.cdf(b[t] * z / (se[t] * abs(z))), "mean": dr[o].mean()})
        for t in ["z_trail", "z_lead", "ht_trail"]:
            LOOKS["drive"] += 1
            rows.append({"outcome": o, "ref": "coef_" + t, "effect": b[t], "se": se[t], "prob_pos": norm.cdf(b[t] / se[t]), "mean": dr[o].mean()})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "drive_effects.csv", index=False)
    return real, r


def main():
    d = load()
    d = controls(d)
    d = situation(d, halftime())
    base, sl, ref = fit_all(d)
    la = late_early(base)
    cells = surface(base)
    ch = coach_het(base)
    real, dre = drives(base)
    summ = {"n_plays": int(len(base)), "looks": LOOKS, "descriptive_cells": cells, "real_drive_checks": real}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summ)
    print(sl.round(4).to_string())
    print(ref[ref.ref.isin(["q3_lead7", "final2_trail7", "q4start_trail10"])].round(4).to_string())
    print(la.round(4).to_string())
    print(ch.round(4).to_string())
    print(dre.round(4).to_string())


if __name__ == "__main__":
    sys.exit(main())
