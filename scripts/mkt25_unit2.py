import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mkt25_opener_revision import (
    BOOT,
    OUT_STATUS,
    SEASONS,
    fit_ratings,
    ols,
    slot,
    week_boot,
    week_groups,
)
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

WINDOWS = (8, 12, 17)
BASE_WINDOW = 17
WEIGHT_FIRST_SEASON = 2011
GROUPS = {
    "QB": ["QB"],
    "OL": ["T", "G", "C"],
    "RECV": ["WR", "TE"],
    "RB": ["RB", "FB"],
    "DL": ["DE", "DT"],
    "LB": ["LB"],
    "DB": ["CB", "S"],
    "ST": ["K", "P", "LS"],
}
POS_TO_GROUP = {p: k for k, v in GROUPS.items() for p in v}
OUT = Path("tests/scratch/mkt25_unit2_out.txt")
OUT.write_text("")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    with OUT.open("a") as f:
        f.write(s + "\n")


def group_counts():
    path = sorted(glob.glob("data/raw/nflverse_injuries/*/injuries.parquet"))[-1]
    inj = pd.read_parquet(path)
    inj = inj[(inj.game_type == "REG") & inj.report_status.isin(OUT_STATUS)].copy()
    inj["grp"] = inj.position.map(POS_TO_GROUP)
    inj = inj.dropna(subset=["grp"])
    pc = inj.groupby(["season", "week", "team", "grp"]).size().unstack(fill_value=0)
    return pc.reindex(columns=list(GROUPS), fill_value=0)


def implied_table(g, window, first_season, last_season):
    rows = []
    sub = g[g.season.between(first_season, last_season) & (g.week >= 2)]
    for (s, w), gw in sub.groupby(["season", "week"]):
        win = g[(g.slot < slot(s, w)) & (g.slot >= slot(s, w) - window)]
        hfa, r, _ = fit_ratings(win)
        for x in gw.itertuples():
            if x.home_team in r and x.away_team in r:
                rows.append((x.game_id, hfa + r[x.home_team] - r[x.away_team]))
    return pd.DataFrame(rows, columns=["game_id", "implied"]).set_index("game_id").implied


def position_weights(g, pc, window):
    imp = implied_table(g, window, WEIGHT_FIRST_SEASON, max(SEASONS))
    gg = g[g.game_id.isin(imp.index)].copy()
    gg["implied"] = gg.game_id.map(imp)
    cols = []
    for k in GROUPS:
        keys_h = zip(gg.season, gg.week, gg.home_team, strict=True)
        keys_a = zip(gg.season, gg.week, gg.away_team, strict=True)
        h = np.array([pc[k].get(key, 0) for key in keys_h])
        a = np.array([pc[k].get(key, 0) for key in keys_a])
        gg["d_" + k] = h - a
        cols.append("d_" + k)
    gg["y"] = gg.spread_line - gg.implied
    weights = {}
    for s in SEASONS:
        tr = gg[gg.season < s]
        b = ols(tr[cols].to_numpy(float), tr.y.to_numpy(float))
        weights[s] = dict(zip(GROUPS, b[1:], strict=True))
    return weights


def build(window, g, o, pc, q):
    sides = []
    for tc, sign in (("home_team", 1), ("away_team", -1)):
        sides.append(
            pd.DataFrame(
                {
                    "game_id": g.game_id,
                    "season": g.season,
                    "week": g.week,
                    "slot": g.slot,
                    "team": g[tc],
                    "cm": sign * (g.result - g.spread_line),
                    "side": sign,
                }
            )
        )
    long = pd.concat(sides).sort_values(["team", "slot"])
    keys = list(zip(long.season, long.week, long.team, strict=True))
    for k in GROUPS:
        long["n_" + k] = [pc[k].get(key, 0) for key in keys]
    long["prev_cm"] = long.groupby("team").cm.shift(1)
    for k in GROUPS:
        long["p1_" + k] = long.groupby("team")["n_" + k].shift(1)
        long["p2_" + k] = long.groupby("team")["n_" + k].shift(2)
    lg = long.set_index(["game_id", "side"])
    weights = position_weights(g, pc, window)
    rows = []
    for (s, w), wk in o[o.week >= 2].groupby(["season", "week"]):
        gw = g[(g.season == s) & (g.week == w)].set_index("game_id")
        win = g[(g.slot < slot(s, w)) & (g.slot >= slot(s, w) - window)]
        hfa, r, _ = fit_ratings(win)
        for x in wk.itertuples():
            gid = x.game_id
            if gid not in gw.index:
                continue
            h, a = gw.loc[gid, "home_team"], gw.loc[gid, "away_team"]
            if h not in r or a not in r:
                continue
            lh, la = lg.loc[(gid, 1)], lg.loc[(gid, -1)]
            row = {
                "game_id": gid,
                "season": s,
                "week": w,
                "R": x.tue_open_home_spread - (hfa + r[h] - r[a]),
                "cm_h": lh.prev_cm,
                "cm_a": la.prev_cm,
                "z": logit(np.clip(x.home_cover_probability_at_open, 1e-6, 1 - 1e-6)),
                "mvo": x.margin_vs_open,
            }
            wn = wl = 0.0
            for k in GROUPS:
                news = (lh["p1_" + k] - lh["p2_" + k]) - (la["p1_" + k] - la["p2_" + k])
                lev = lh["p1_" + k] - la["p1_" + k]
                wn += weights[s][k] * news
                wl += weights[s][k] * lev
            row["wnews"] = wn
            row["wlevel"] = wl
            rows.append(row)
    d = pd.DataFrame(rows).dropna()
    d = d[d.mvo != 0].copy()
    d["y"] = (d.mvo > 0).astype(int)
    d["cm"] = d.cm_h - d.cm_a
    d["cm_pos"] = d.cm_h.clip(lower=0) - d.cm_a.clip(lower=0)
    d["cm_neg"] = d.cm_h.clip(upper=0) - d.cm_a.clip(upper=0)
    d["cm_in"] = d.cm_h.clip(-q, q) - d.cm_a.clip(-q, q)
    d["cm_out"] = d.cm - d.cm_in
    return d, weights


def loso_fit(d, cols):
    out = pd.Series(0.0, index=d.index)
    for s in SEASONS:
        tr, te = d[d.season != s], d.season == s
        b = ols(tr[cols].to_numpy(float), tr.R.to_numpy(float))
        out[te] = b[0] + d.loc[te, cols].to_numpy(float) @ b[1:]
    return out


def logloss(d, p):
    return -(d.y * np.log(p) + (1 - d.y) * np.log(1 - p))


def loso_probs(d, cols):
    p = pd.Series(0.5, index=d.index)
    coef = {}
    for s in SEASONS:
        tr, te = d[d.season != s], d.season == s
        if not cols:
            p[te] = tr.y.mean()
            continue
        m = LogisticRegression(C=1e6, max_iter=1000).fit(tr[cols], tr.y)
        p[te] = m.predict_proba(d.loc[te, cols])[:, 1]
        coef[s] = np.round(m.coef_[0], 3).tolist()
    return p.clip(1e-6, 1 - 1e-6), coef


def term_test(d, label, term, rng, cells, tag):
    d = d.copy()
    d["t"] = term / term.std()
    ps, _ = loso_probs(d, ["z"])
    pt, coef = loso_probs(d, ["z", "t"])
    flip = (pt > 0.5) != (ps > 0.5)
    fw = int(((pt > 0.5) == (d.y == 1))[flip].sum())
    dv = (logloss(d, ps) - logloss(d, pt)).to_numpy()
    groups = week_groups(d)
    bs = np.array(
        [
            np.concatenate([dv[groups[i]] for i in rng.choice(len(groups), len(groups))]).mean()
            for _ in range(BOOT)
        ]
    )
    sb = pd.Series(dv).groupby(d.season.to_numpy()).mean()
    half = [dv[(d.season % 2 == p).to_numpy()].mean() for p in (0, 1)]
    wl_t = int(((pt > 0.5) == (d.y == 1)).sum())
    wl_s = int(((ps > 0.5) == (d.y == 1)).sum())
    bt = ((pt - d.y) ** 2).mean()
    bsv = ((ps - d.y) ** 2).mean()
    m = LogisticRegression(C=1e6, max_iter=1000).fit(d[["z", "t"]], d.y)
    ins = float(logloss(d, m.predict_proba(d[["z", "t"]])[:, 1]).mean())
    gap = float(logloss(d, pt).mean()) - ins
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    say(
        f"TERM {label}: DECISIVE flips {int(flip.sum())} new side {fw}-{int(flip.sum()) - fw}; "
        f"acc term {wl_t}-{len(d) - wl_t} served {wl_s}-{len(d) - wl_s}; "
        f"dLL {dv.mean():.5f} CI[{lo:.5f},{hi:.5f}] P+ {(bs > 0).mean():.3f} "
        f"dBrier {bsv - bt:.5f} seasons+ {(sb > 0).sum()}/6 halves {half[0]:.5f}/{half[1]:.5f} "
        f"LOSO LL {logloss(d, pt).mean():.5f} in-sample {ins:.5f} gap {gap:.5f}"
    )
    say("   per-fold coef [z, term]:", coef)
    cells.append(
        {
            "name": f"mkt25u2_{tag}_term_dLL_vs_served",
            "effect": round(float(dv.mean()), 5),
            "effect_units": "log_loss_improvement",
            "interval_low": round(float(lo), 5),
            "interval_high": round(float(hi), 5),
            "probability_positive": round(float((bs > 0).mean()), 3),
            "classification_evidence": (
                f"LOSO 2020-25 n {len(d)}; {label}; flips {int(flip.sum())} new side "
                f"{fw}-{int(flip.sum()) - fw}; seasons positive {(sb > 0).sum()}/6; "
                f"halves {half[0]:.5f}/{half[1]:.5f}; week-block bootstrap"
            ),
        }
    )


def slope_test(d, label, fn, rng, cells, tag, units):
    b = fn(d)
    bs = week_boot(d, fn, rng)
    sh = [fn(d[d.season % 2 == p]) for p in (0, 1)]
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    say(
        f"SLOPE {label}: {b:.4f} CI[{lo:.4f},{hi:.4f}] P+ {(bs > 0).mean():.3f} "
        f"halves {sh[0]:.4f}/{sh[1]:.4f}"
    )
    cells.append(
        {
            "name": f"mkt25u2_{tag}",
            "effect": round(float(b), 4),
            "effect_units": units,
            "interval_low": round(float(lo), 4),
            "interval_high": round(float(hi), 4),
            "probability_positive": round(float((bs > 0).mean()), 3),
            "classification_evidence": (
                f"{label}; n {len(d)}; split-half {sh[0]:.4f}/{sh[1]:.4f}; week-block bootstrap"
            ),
        }
    )


def s1(c):
    def fn(x):
        return ols(x[[c]].to_numpy(float), x.R.to_numpy(float))[1]

    return fn


def sdiff(a, b):
    def fn(x):
        return s1(a)(x) - s1(b)(x)

    return fn


def main():
    g = pd.read_parquet("data/processed/game_features.parquet")
    g = g[(g.game_type == "REG") & g.spread_line.notna() & g.result.notna()].copy()
    g["slot"] = [slot(s, w) for s, w in zip(g.season, g.week, strict=True)]
    o = pd.read_parquet(sorted(glob.glob("artifacts/opener_evaluation/*/per_game.parquet"))[-1])
    o = o[o.season.isin(SEASONS)]
    pc = group_counts()
    pre = g[g.season < min(SEASONS)]
    q = float((pre.result - pre.spread_line).abs().median())
    say(f"blowout cut q = median |cover margin| pre-{min(SEASONS)} ({len(pre)} games): {q:.3f}")
    rng = np.random.default_rng(2502)
    cells = []
    res = {}
    for wdw in WINDOWS:
        res[wdw] = build(wdw, g, o, pc, q)
        say(f"window {wdw}: n {len(res[wdw][0])}")
    d, weights = res[BASE_WINDOW]
    say("position weights (spread points per extra Out/Doubtful, home minus away), window 17:")
    for s in SEASONS:
        say(" ", s, {k: round(float(v), 3) for k, v in weights[s].items()})

    say("== 1 asymmetry ==")
    for c in ["cm_pos", "cm_neg", "cm_in", "cm_out"]:
        slope_test(d, f"R~{c} (w17)", s1(c), rng, cells, f"slope_{c}", "logit_coefficient")
    slope_test(
        d,
        "slope(cm_pos)-slope(cm_neg)",
        sdiff("cm_pos", "cm_neg"),
        rng,
        cells,
        "diff_sign",
        "slope_difference",
    )
    slope_test(
        d,
        "slope(cm_out)-slope(cm_in)",
        sdiff("cm_out", "cm_in"),
        rng,
        cells,
        "diff_blowout",
        "slope_difference",
    )
    base = loso_fit(d, ["cm"])
    for tag, cols in (("asym_sign", ["cm_pos", "cm_neg"]), ("asym_blowout", ["cm_in", "cm_out"])):
        term_test(d, f"{tag} component", loso_fit(d, cols) - base, rng, cells, tag)

    say("== 2 position-weighted injuries ==")
    slope_test(d, "R~wnews", s1("wnews"), rng, cells, "slope_wnews", "logit_coefficient")
    slope_test(d, "R~wlevel", s1("wlevel"), rng, cells, "slope_wlevel", "logit_coefficient")
    e_pos = d.R - loso_fit(d, ["cm", "wnews"])
    term_test(d, "e after cm + weighted news", e_pos, rng, cells, "posnews")

    say("== 3 window family ==")
    for wdw in WINDOWS:
        dw = res[wdw][0]
        slope_test(
            dw, f"R~cm window {wdw}", s1("cm"), rng, cells, f"slope_cm_w{wdw}", "logit_coefficient"
        )
        term_test(dw, f"e_cm window {wdw}", dw.R - loso_fit(dw, ["cm"]), rng, cells, f"ecm_w{wdw}")
    ps, _ = loso_probs(d, ["z"])
    pi, _ = loso_probs(d, [])
    say(f"served vs intercept control dLL {(logloss(d, pi) - logloss(d, ps)).mean():.5f}")
    say("total cells/looks", len(cells))
    batch = {
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": 2020,
        "season_end": 2025,
        "family": "mkt25_opener_revision_unit2",
        "category": "market",
        "source": "scripts/mkt25_unit2.py; tests/scratch/mkt25_unit2_out.txt",
        "sample_games": len(d),
        "plain_summary": (
            "Whether asymmetric, position-weighted or window-varied opener revision beyond "
            "the prior cover adds to the opener-graded pick probability."
        ),
        "description": (
            "Unit 2: asymmetry by prior-cover sign and blowout size, position-weighted injury "
            "news with weights from prior seasons, rating windows 8/12/17; LOSO terms on the "
            "served logit graded at opener"
        ),
        "cells": cells,
    }
    Path("artifacts/mkt25/record_batch_unit2.json").write_text(json.dumps(batch, indent=1))


if __name__ == "__main__":
    sys.exit(main())
