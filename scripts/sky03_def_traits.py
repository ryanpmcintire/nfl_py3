import argparse
import glob
import json
import sys

import numpy as np
import pandas as pd
import sky03_matchups as m
import sky03_pfr_traits as p
from sky04_lead_lag import fit_b, logit

DEF_GLOB = str(m.ROOT / "data" / "raw" / "pfr_advstats" / "*" / "advstats_week_def_{s}.csv")
NEW = {
    "oH": ("hur", "n_db"),
    "dH": ("hur", "n_db"),
    "dPD": ("prd", "n_db"),
    "dY": ("ypt", "tg2"),
    "dC": ("cpt", "tg2"),
}
CAND = {
    "A_pbp_hit_plus_pfr_hurry": ("zAh", "rA", "n_db"),
    "A_oP_x_pfr_def_pressures": ("zAd", "rA", "n_db"),
    "B_oN_x_yards_per_target_allowed": ("zBy", "rB", "n_ns"),
    "B_oN_x_completions_per_target_allowed": ("zBc", "rB", "n_ns"),
}
OUTF = m.OUT / "def_traits"


def load():
    d = pd.concat(
        [
            pd.read_csv(sorted(glob.glob(DEF_GLOB.format(s=s)))[-1])
            for s in range(p.FIRST, p.LAST + 1)
        ]
    )
    d = d[d.game_type == "REG"]
    g = d.groupby(["game_id", "team"], as_index=False).agg(
        tg2=("def_targets", "sum"),
        cmp=("def_completions_allowed", "sum"),
        yds=("def_yards_allowed", "sum"),
        hurried=("def_times_hurried", "sum"),
        pres=("def_pressures", "sum"),
    )
    g = g.rename(columns={"team": "deff"})
    t = m.load_games()
    t = t[t.season.between(p.FIRST, p.LAST)].copy()
    t = t.merge(g, on=["game_id", "deff"], how="left")
    cov = float(t.tg2.notna().mean())
    t = t.dropna(subset=["tg2"]).copy()
    t["hur"] = (t.press_s + t.hurried) / t.n_db
    t["prd"] = t.pres / t.n_db
    t["ypt"] = (t.yds / t.tg2.clip(lower=1)).where(t.tg2 > 0, 0.0)
    t["cpt"] = (t["cmp"] / t.tg2.clip(lower=1)).where(t.tg2 > 0, 0.0)
    return t, cov


def build(t):
    base = dict(m.TRAITS)
    m.TRAITS.update(NEW)
    rr = m.build_rows(t, tuple(range(p.FIRST + 1, p.LAST + 1)))
    m.TRAITS.clear()
    m.TRAITS.update(base)
    rr["zAh"] = rr.oH * rr.dH
    rr["zAd"] = rr.oP * rr.dPD
    rr["zBy"] = rr.oN * rr.dY
    rr["zBc"] = rr.oN * rr.dC
    return rr


def term(rr, z, rc, wc, seasons, rng):
    g = pd.read_parquet(m.PERGAME)
    g = g[(g.game_type == "REG") & g.season.isin(seasons)]
    sub = g[g.margin_vs_open != 0].copy().reset_index(drop=True)
    rows = []
    for s in seasons:
        for w in sorted(rr[rr.season == s].week.unique()):
            hist = rr[rr.key < s * 100 + w]
            cur = rr[(rr.season == s) & (rr.week == w)]
            c = m.fit_c(hist, z, rc, wc)
            sc = cur.ns_share if wc == "n_ns" else 1.0
            rows.append(
                cur.assign(pts=cur.db_exp * c * cur[z] * sc, c=c)[["game_id", "off", "pts", "c"]]
            )
    gx = pd.concat(rows)
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
        "c_last": float(gx.c.iloc[-1]),
        "oos_brier_gain_by_season": {
            int(s): float(np.mean(((pr - y) ** 2 - (pooled - y) ** 2)[se == s])) for s in seasons
        },
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
            "brier_gain": p.boot(d_b, rng),
            "logloss_gain": p.boot(d_ll, rng),
            "accuracy_gain_points": p.boot(d_acc, rng),
            "decisive_games": int(dec.sum()),
            "decisive_new_correct": int(((arr > 0.5) == (y == 1))[dec].sum()),
            "decisive_base_correct": int(((pr > 0.5) == (y == 1))[dec].sum()),
        }
    res["gap_brier_gain_ins_minus_oos"] = res["ins"]["brier_gain"][0] - res["oos"]["brier_gain"][0]
    res["model_only_brier"] = float(((pr - y) ** 2).mean())
    res["market_brier_note"] = "market baseline: opener implied 0.5 cover prob, brier 0.25"
    res["market_brier"] = 0.25
    return res, sub, xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rel", "term"], required=True)
    ap.add_argument("--cand", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(m.SEED)
    t, cov = load()
    OUTF.mkdir(parents=True, exist_ok=True)
    rr = build(t)
    if a.stage == "rel":
        res = {"coverage": cov, "team_games": len(t), "trait_split_half": {}}
        for lab, val, wt in (
            ("pbp_sack_hit_plus_pfr_hurry_rate", "hur", "n_db"),
            ("pfr_def_pressure_rate", "prd", "n_db"),
            ("pfr_yards_per_target_allowed", "ypt", "tg2"),
            ("pfr_completions_per_target_allowed", "cpt", "tg2"),
        ):
            res["trait_split_half"][lab] = {
                side: p.split_half(t, val, wt, grp, rng)
                for side, grp in (("offense", "off"), ("defense", "deff"))
            }
        d = rr[rr.season.between(m.EVAL_SEASONS[0], p.LAST) & (rr.week > 1)]
        res["interaction_slope_split_half"] = {
            k: p.slope_rel(d, z, r, w, rng) for k, (z, r, w) in CAND.items()
        }
        json.dump(res, open(OUTF / "rel.json", "w"), indent=1)  # noqa: SIM115
        print(json.dumps(res, indent=1))
    else:
        z, r, w = CAND[a.cand]
        res, sub, xs = term(rr, z, r, w, m.EVAL_SEASONS, rng)  # noqa: RUF059
        res["candidate"] = a.cand
        json.dump(res, open(OUTF / "term.json", "w"), indent=1)  # noqa: SIM115
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
