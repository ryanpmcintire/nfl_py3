import argparse
import glob
import json
import sys

import numpy as np
import pandas as pd
import sky03_matchups as m
from sky04_lead_lag import fit_b, logit

ROOT = m.ROOT
OUT = m.OUT
PFR_GLOB = str(ROOT / "data" / "raw" / "pfr_advstats" / "*" / "advstats_week_pass_{s}.csv")
NGS_GLOB = str(ROOT / "data" / "raw" / "ngs" / "*" / "ngs_{k}.parquet")
FIRST = 2018
LAST = 2025
BOOT = 2000
NEW = {
    "oP2": ("press2", "n_db"),
    "dP2": ("press2", "n_db"),
    "oS": ("sep", "tg"),
    "dS": ("sep", "tg"),
    "oT": ("ttt", "att"),
    "dT": ("ttt", "att"),
}


def load_new():
    pf = pd.concat(
        [pd.read_csv(sorted(glob.glob(PFR_GLOB.format(s=s)))[-1]) for s in range(FIRST, LAST + 1)]
    )
    pf = pf[pf.game_type == "REG"]
    p = pf.groupby(["game_id", "team"], as_index=False).times_pressured.sum()
    r = pd.read_parquet(sorted(glob.glob(NGS_GLOB.format(k="receiving")))[-1])
    r = r[(r.season_type == "REG") & (r.week > 0) & r.season.between(FIRST, LAST) & (r.targets > 0)]
    r = (
        r.assign(w=r.targets * r.avg_separation)
        .groupby(["season", "week", "team_abbr"], as_index=False)
        .agg(w=("w", "sum"), tg=("targets", "sum"))
    )
    r["sep"] = r.w / r.tg
    q = pd.read_parquet(sorted(glob.glob(NGS_GLOB.format(k="passing")))[-1])
    q = q[
        (q.season_type == "REG") & (q.week > 0) & q.season.between(FIRST, LAST) & (q.attempts > 0)
    ]
    q = (
        q.assign(w=q.attempts * q.avg_time_to_throw)
        .groupby(["season", "week", "team_abbr"], as_index=False)
        .agg(w=("w", "sum"), att=("attempts", "sum"))
    )
    q["ttt"] = q.w / q.att
    t = m.load_games()
    t = t[t.season.between(FIRST, LAST)].copy()
    t = t.merge(p.rename(columns={"team": "off"}), on=["game_id", "off"], how="left")
    t = t.merge(
        r[["season", "week", "team_abbr", "sep", "tg"]].rename(columns={"team_abbr": "off"}),
        on=["season", "week", "off"],
        how="left",
    )
    t = t.merge(
        q[["season", "week", "team_abbr", "ttt", "att"]].rename(columns={"team_abbr": "off"}),
        on=["season", "week", "off"],
        how="left",
    )
    t["press2"] = t.times_pressured / t.n_db
    cov = {c: float(t[c].notna().mean()) for c in ("press2", "sep", "ttt")}
    t = t.dropna(subset=["press2", "sep", "ttt"]).copy()
    return t, cov


def split_half(t, val, wt, grp, rng):
    rows = []
    for (s, tm), g in t.groupby(["season", grp]):  # noqa: B007
        a, b = g[g.week % 2 == 1], g[g.week % 2 == 0]
        if len(a) and len(b):
            rows.append((s, np.average(a[val], weights=a[wt]), np.average(b[val], weights=b[wt])))
    d = pd.DataFrame(rows, columns=["s", "a", "b"])
    d["a"] -= d.groupby("s").a.transform("mean")
    d["b"] -= d.groupby("s").b.transform("mean")
    x = d[["a", "b"]].values
    r0 = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])
    bs = []
    for _ in range(BOOT):
        i = rng.integers(0, len(x), len(x))
        bs.append(np.corrcoef(x[i, 0], x[i, 1])[0, 1])
    bs = np.array(bs)
    return {
        "r": r0,
        "lo": float(np.nanpercentile(bs, 2.5)),
        "hi": float(np.nanpercentile(bs, 97.5)),
        "p_positive": float(np.nanmean(bs > 0)),
        "team_seasons": len(x),
    }


def slope_rel(d, zc, rc, wc, rng):
    out = {}
    odd, even = d[d.week % 2 == 1], d[d.week % 2 == 0]
    for side, grp in (("offense", "off"), ("defense", "deff")):
        teams = sorted(d[grp].unique())
        so = np.array([m.fit_c(odd[odd[grp] == tm], zc, rc, wc) for tm in teams])
        se = np.array([m.fit_c(even[even[grp] == tm], zc, rc, wc) for tm in teams])
        r0 = float(np.corrcoef(so, se)[0, 1])
        bs = np.array(
            [
                np.corrcoef(so[i], se[i])[0, 1]
                for i in (rng.integers(0, len(teams), len(teams)) for _ in range(BOOT))
            ]
        )
        out[side] = {
            "r": r0,
            "lo": float(np.nanpercentile(bs, 2.5)),
            "hi": float(np.nanpercentile(bs, 97.5)),
            "p_positive": float(np.nanmean(bs > 0)),
        }
    return out


def build(t):
    base = dict(m.TRAITS)
    m.TRAITS.update(NEW)
    rr = m.build_rows(t, tuple(range(FIRST + 1, LAST + 1)))
    m.TRAITS.clear()
    m.TRAITS.update(base)
    rr["zA2"] = rr.oP2 * rr.dP2
    rr["zB2"] = rr.oS * rr.dS
    return rr


def boot(a, rng):
    bs = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(BOOT)])
    return [
        float(a.mean()),
        float(np.percentile(bs, 2.5)),
        float(np.percentile(bs, 97.5)),
        float((bs > 0).mean()),
    ]


def game_x2(rr, seasons):
    rows = []
    for s in seasons:
        for w in sorted(rr[rr.season == s].week.unique()):
            hist = rr[rr.key < s * 100 + w]
            cur = rr[(rr.season == s) & (rr.week == w)]
            cA = m.fit_c(hist, "zA2", "rA", "n_db")
            cB = m.fit_c(hist, "zB2", "rB", "n_ns")
            cur = cur.assign(
                pts=cur.db_exp * (cA * cur.zA2 + cB * cur.zB2 * cur.ns_share), cA=cA, cB=cB
            )
            rows.append(cur[["game_id", "off", "deff", "pts", "cA", "cB"]])
    return pd.concat(rows, ignore_index=True)


def term(rr, seasons, rng):
    g = pd.read_parquet(m.PERGAME)
    g = g[(g.game_type == "REG") & g.season.isin(seasons)]
    sub = g[g.margin_vs_open != 0].copy().reset_index(drop=True)
    gx = game_x2(rr, seasons)
    hp = gx.merge(sub[["game_id", "home_team"]], on="game_id")
    hp["sp"] = np.where(hp.off == hp.home_team, 1.0, -1.0) * hp.pts
    sub["x"] = sub.game_id.map(hp.groupby("game_id").sp.sum())
    sub = sub[sub.x.notna()].reset_index(drop=True)
    y = sub.home_covered.astype(float).values
    pr = sub.out_of_season_home_probability.clip(1e-6, 1 - 1e-6).values
    off, xs, se = logit(pr), sub.x.values, sub.season.values
    pooled, fc = np.zeros(len(sub)), {}
    for s in seasons:
        te = se == s
        b = fit_b(off[~te], xs[~te], y[~te])
        fc[int(s)] = float(b)
        pooled[te] = 1 / (1 + np.exp(-(off[te] + b * xs[te])))
    b_all = fit_b(off, xs, y)
    ins = 1 / (1 + np.exp(-(off + b_all * xs)))
    res = {
        "games": len(sub),
        "x_sd_points": float(xs.std()),
        "fold_coefficients": fc,
        "pooled_coefficient": float(b_all),
        "c_A_last": float(gx.cA.iloc[-1]),
        "c_B_last": float(gx.cB.iloc[-1]),
    }
    for label, arr in (("oos", pooled), ("ins", ins)):
        d_b = (pr - y) ** 2 - (arr - y) ** 2
        d_ll = -(y * np.log(pr) + (1 - y) * np.log(1 - pr)) + (
            y * np.log(arr) + (1 - y) * np.log(1 - arr)
        )
        d_acc = (
            ((arr > 0.5) == (y == 1)).astype(float) - ((pr > 0.5) == (y == 1)).astype(float)
        ) * 100
        dec = (pr > 0.5) != (arr > 0.5)
        res[label] = {
            "brier_gain": boot(d_b, rng),
            "logloss_gain": boot(d_ll, rng),
            "accuracy_gain_points": boot(d_acc, rng),
            "decisive_games": int(dec.sum()),
            "decisive_new_correct": int(((arr > 0.5) == (y == 1))[dec].sum()),
            "decisive_base_correct": int(((pr > 0.5) == (y == 1))[dec].sum()),
        }
    res["gap_brier_gain_ins_minus_oos"] = res["ins"]["brier_gain"][0] - res["oos"]["brier_gain"][0]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rel", "term", "all"], default="rel")
    a = ap.parse_args()
    rng = np.random.default_rng(m.SEED)
    t, cov = load_new()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"coverage_before_dropna": cov, "team_games": len(t)}
    rr = None
    if a.stage in ("rel", "all"):
        tr = {}
        for lab, val, wt in (
            ("pbp_pressure_rate", "press", "n_db"),
            ("pfr_pressure_rate", "press2", "n_db"),
            ("pbp_epa_nonsack", "epa_ns", "n_ns"),
            ("ngs_separation", "sep", "tg"),
            ("ngs_time_to_throw", "ttt", "att"),
        ):
            tr[lab] = {
                side: split_half(t, val, wt, grp, rng)
                for side, grp in (("offense", "off"), ("defense", "deff"))
            }
        res["trait_split_half"] = tr
        rr = build(t)
        d = rr[rr.season.between(m.EVAL_SEASONS[0], LAST) & (rr.week > 1)]
        res["interaction_slope_split_half"] = {
            "A_old_pbp_pressure": slope_rel(d, "zA", "rA", "n_db", rng),
            "A_new_pfr_pressure": slope_rel(d, "zA2", "rA", "n_db", rng),
            "B_old_pbp_epa": slope_rel(d, "zB", "rB", "n_ns", rng),
            "B_new_ngs_separation": slope_rel(d, "zB2", "rB", "n_ns", rng),
        }
    if a.stage in ("term", "all"):
        rr = rr if rr is not None else build(t)
        res["term"] = term(rr, m.EVAL_SEASONS, rng)
    f = OUT / ("pfr_traits_" + a.stage + ".json")
    json.dump(res, open(f, "w"), indent=1)  # noqa: SIM115
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
