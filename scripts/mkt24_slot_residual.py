import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

OUT = Path("artifacts/mkt24_slot_residual")
OPENER = Path("artifacts/opener_evaluation/20261103T195331Z/per_game.parquet")
PROXY = Path("artifacts/mod24_u1/base_per_game.parquet")
FEATURES = Path("data/processed/game_features.parquet")
GAP_MINUTES = 60
PEN_SCALE = 100.0
C_GRID = [0.01, 0.1, 1.0, 10.0]
DRAWS = 5000
SEED = 1
POPS = {"opener_2020_2025": (2020, 2025), "sbr_proxy_2011_2019": (2011, 2019)}


def kickoff_cuts(sched):
    sun = sched[(sched.weekday == "Sunday") & (sched.location == "Home")]
    mins = np.sort(sun.minute.unique())
    return [
        (mins[i] + mins[i + 1]) / 2
        for i in range(len(mins) - 1)
        if mins[i + 1] - mins[i] > GAP_MINUTES
    ]


def build_slots():
    s = pd.read_parquet(FEATURES)
    s = s[s.game_type == "REG"].copy()
    s["minute"] = s.gametime.str.slice(0, 2).astype(int) * 60 + s.gametime.str.slice(3, 5).astype(
        int
    )
    cuts = kickoff_cuts(s)
    day = pd.to_datetime(s.gameday)
    thanks = (day.dt.month == 11) & (day.dt.weekday == 3) & day.dt.day.between(22, 28)
    xmas = (day.dt.month == 12) & (day.dt.day == 25)
    slot = pd.Series("other_weekday", index=s.index)
    wd = s.weekday
    slot[wd == "Thursday"] = "thursday_night"
    slot[wd == "Monday"] = "monday_night"
    slot[wd == "Saturday"] = "saturday"
    sun = wd == "Sunday"
    slot[sun & (s.minute < cuts[0])] = "sunday_early"
    slot[sun & (s.minute >= cuts[0]) & (s.minute < cuts[1])] = "sunday_late"
    slot[sun & (s.minute >= cuts[1])] = "sunday_night"
    slot[thanks | xmas] = "holiday"
    slot[s.location == "Neutral"] = "international"
    s["slot"] = slot
    return s[["game_id", "slot"]], [float(c) for c in cuts]


def population(name, slots):
    lo, hi = POPS[name]
    if name.startswith("opener"):
        o = pd.read_parquet(OPENER)
        d = o[o.season.between(lo, hi) & o.margin_vs_open.ne(0)].copy()
        d["y"] = (d.margin_vs_open > 0).astype(int)
        d["p"] = d.home_cover_probability_at_open
    else:
        b = pd.read_parquet(PROXY).reset_index()
        d = b[b.season.between(lo, hi) & b.correct.isin([0.0, 1.0])].copy()
        d["y"] = np.where(d.pick_home, d.correct, 1 - d.correct).astype(int)
    d = d[["game_id", "season", "week", "y", "p"]]
    d = d.merge(slots, on="game_id", how="left").reset_index(drop=True)
    d = d[d.slot.notna() & d.slot.ne("other_weekday")].reset_index(drop=True)
    d["z"] = logit(d.p.clip(1e-6, 1 - 1e-6))
    d["r"] = d.y - d.p
    return d


def shrink(d):
    g = d.groupby("slot").r
    n = g.size()
    m = g.mean()
    se2 = (g.var(ddof=1) / n).fillna(0.0)
    w = 1 / se2
    mu = (w * m).sum() / w.sum()
    q = (w * (m - mu) ** 2).sum()
    k = len(n)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    b = tau2 / (tau2 + se2)
    post = b * m
    psd = np.sqrt(b * se2)
    raw_p = pd.Series(norm.cdf(m / np.sqrt(se2)), index=n.index)
    pp = pd.Series(norm.cdf(post / psd.replace(0, np.nan)), index=n.index).fillna(0.5)
    return pd.DataFrame(
        {
            "n": n,
            "raw_resid": m,
            "raw_se": np.sqrt(se2),
            "raw_Ppos": raw_p,
            "tau2": tau2,
            "B": b,
            "shrunk_resid": post,
            "shrunk_sd": psd,
            "shrunk_lo": post - 1.96 * psd,
            "shrunk_hi": post + 1.96 * psd,
            "shrunk_Ppos": pp,
        }
    )


def design(d, slots_list):
    x = pd.DataFrame({"zs": d.z * PEN_SCALE}, index=d.index)
    for s in slots_list:
        x[s] = (d.slot == s).astype(float)
    return x


def fit_pred(tr, te, slots_list, c):
    m = LogisticRegression(C=c, max_iter=2000).fit(design(tr, slots_list), tr.y)
    return m.predict_proba(design(te, slots_list))[:, 1], m


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def choose_c(tr, slots_list):
    best, bll = None, np.inf
    for c in C_GRID:
        tot = 0.0
        for s in sorted(tr.season.unique()):
            a, b = tr[tr.season != s], tr[tr.season == s]
            p, _ = fit_pred(a, b, slots_list, c)
            tot += ll(b.y.values, p).sum()
        if tot < bll:
            best, bll = c, tot
    return best


def loso(d):
    slots_list = sorted(d.slot.unique())
    P = {k: pd.Series(np.nan, index=d.index) for k in ["intercept", "served", "slot"]}
    coefs, cs = {}, {}
    for s in sorted(d.season.unique()):
        tr, te = d[d.season != s], d[d.season == s]
        P["intercept"][te.index] = tr.y.mean()
        mz = LogisticRegression(C=1e6, max_iter=2000).fit(tr[["z"]], tr.y)
        P["served"][te.index] = mz.predict_proba(te[["z"]])[:, 1]
        c = choose_c(tr, slots_list)
        p, m = fit_pred(tr, te, slots_list, c)
        P["slot"][te.index] = p
        cs[int(s)] = c
        coefs[int(s)] = {
            "z": float(m.coef_[0][0] * PEN_SCALE),
            **{k: float(v) for k, v in zip(slots_list, m.coef_[0][1:], strict=True)},
        }
    c_all = choose_c(d, slots_list)
    p_in, _ = fit_pred(d, d, slots_list, c_all)
    mz = LogisticRegression(C=1e6, max_iter=2000).fit(d[["z"]], d.y)
    ins = {"slot": p_in, "served": mz.predict_proba(d[["z"]])[:, 1]}
    return {k: v.clip(1e-6, 1 - 1e-6) for k, v in P.items()}, coefs, cs, ins


def compare(d, P, a, b, rng):
    y = d.y.values
    seas = d.season.values
    useas = np.unique(seas)
    dl = ll(y, P[b].values) - ll(y, P[a].values)
    dbr = (P[b].values - y) ** 2 - (P[a].values - y) ** 2
    acc = ((P[a].values > 0.5) == (y == 1)).astype(float) - (
        (P[b].values > 0.5) == (y == 1)
    ).astype(float)
    bg = np.array([dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(DRAWS)])
    bl, ba = [], []
    for _ in range(DRAWS):
        sel = np.concatenate([np.where(seas == s)[0] for s in rng.choice(useas, len(useas))])
        bl.append(dl[sel].mean())
        ba.append(acc[sel].mean() * 100)
    bl, ba = np.array(bl), np.array(ba)
    sb = pd.Series(dl).groupby(seas).mean()
    return {
        "dLL": float(dl.mean()),
        "dLL_game_CI": [float(np.percentile(bg, 2.5)), float(np.percentile(bg, 97.5))],
        "dLL_game_Ppos": float((bg > 0).mean()),
        "dLL_season_CI": [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))],
        "dLL_season_Ppos": float((bl > 0).mean()),
        "dBrier": float(dbr.mean()),
        "dAccPts": float(acc.mean() * 100),
        "dAcc_season_CI": [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))],
        "dAcc_season_Ppos": float((ba > 0).mean()),
        "seasons_positive": f"{int((sb > 0).sum())}/{len(sb)}",
    }


def analyse(name, slots):
    d = population(name, slots)
    rng = np.random.default_rng(SEED)
    y = d.y.values
    out = {"n": len(d), "slot_counts": d.slot.value_counts().to_dict()}
    out["shrink"] = shrink(d).round(5).reset_index().to_dict("records")
    P, coefs, cs, ins = loso(d)
    arms = {}
    for a in P:
        w = int(((P[a] > 0.5).values == (y == 1)).sum())
        arms[a] = {
            "W": w,
            "L": len(d) - w,
            "LL": float(ll(y, P[a].values).mean()),
            "Brier": float(((P[a].values - y) ** 2).mean()),
        }
    out["arms"] = arms
    fl = (P["slot"] > 0.5).values != (P["served"] > 0.5).values
    fw = int(((P["slot"] > 0.5).values[fl] == (y[fl] == 1)).sum())
    out["decisive_flips"] = {
        "flips": int(fl.sum()),
        "slot_right": fw,
        "served_right": int(fl.sum()) - fw,
    }
    out["comparisons"] = {
        "slot_vs_served": compare(d, P, "slot", "served", rng),
        "served_vs_intercept": compare(d, P, "served", "intercept", rng),
    }
    out["insample"] = {
        "in_sample_dLL": float((ll(y, ins["served"]) - ll(y, ins["slot"])).mean()),
        "out_of_sample_dLL": out["comparisons"]["slot_vs_served"]["dLL"],
    }
    out["chosen_C"] = cs
    out["fold_coefs"] = coefs
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    slots, cuts = build_slots()
    res = {"minute_cuts": cuts, "populations": {n: analyse(n, slots) for n in POPS}}
    (OUT / "report.json").write_text(json.dumps(res, indent=1, default=str))
    cols = ["slot", "n", "raw_resid", "raw_Ppos", "tau2", "shrunk_resid", "shrunk_lo", "shrunk_hi"]
    for n, r in res["populations"].items():
        print(n, r["n"], r["slot_counts"])
        sh = pd.DataFrame(r["shrink"])
        print(sh[[*cols, "shrunk_Ppos"]].round(4).to_string())
        print("arms", json.dumps(r["arms"]))
        print("flips", r["decisive_flips"])
        print(json.dumps(r["comparisons"]))
        print(r["insample"], r["chosen_C"])


main()
