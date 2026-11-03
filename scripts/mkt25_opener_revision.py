import glob

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

SEASONS = range(2020, 2026)
WINDOW_WEEKS = 17
WEEK_SLOTS = 18
BOOT = 2000
OUT_STATUS = ("Out", "Doubtful")


def slot(season, week):
    return (season - 2000) * WEEK_SLOTS + week


def fit_ratings(win):
    teams = sorted(set(win.home_team) | set(win.away_team))
    ix = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(win), len(teams) + 1))
    X[:, 0] = 1.0
    for k, (h, a) in enumerate(zip(win.home_team, win.away_team, strict=True)):
        X[k, 1 + ix[h]] += 1.0
        X[k, 1 + ix[a]] -= 1.0
    beta = np.linalg.lstsq(X, win.spread_line.to_numpy(float), rcond=None)[0]
    resid = win.spread_line.to_numpy(float) - X @ beta
    return beta[0], {t: beta[1 + i] for t, i in ix.items()}, resid


def injury_counts():
    path = sorted(glob.glob("data/raw/nflverse_injuries/*/injuries.parquet"))[-1]
    inj = pd.read_parquet(path)
    inj = inj[(inj.game_type == "REG") & inj.report_status.isin(OUT_STATUS)]
    return inj.groupby(["season", "week", "team"]).size()


def team_side(g, team_col, sign):
    return pd.DataFrame(
        {
            "game_id": g.game_id,
            "season": g.season,
            "week": g.week,
            "slot": g.slot,
            "team": g[team_col],
            "cm": sign * (g.result - g.spread_line),
            "side": sign,
        }
    )


def build():
    g = pd.read_parquet("data/processed/game_features.parquet")
    g = g[(g.game_type == "REG") & g.spread_line.notna() & g.result.notna()].copy()
    g["slot"] = [slot(s, w) for s, w in zip(g.season, g.week, strict=True)]
    o = pd.read_parquet(sorted(glob.glob("artifacts/opener_evaluation/*/per_game.parquet"))[-1])
    o = o[o.season.isin(SEASONS)]
    inj = injury_counts()
    long = pd.concat([team_side(g, "home_team", 1), team_side(g, "away_team", -1)])
    mv = o.set_index("game_id")
    long["move"] = long.game_id.map(mv.close_home_spread - mv.tue_open_home_spread) * long.side
    long = long.sort_values(["team", "slot"])
    for c in ["cm", "move"]:
        long["prev_" + c] = long.groupby("team")[c].shift(1)
    long["n_out"] = [
        inj.get((s, w, t), 0) for s, w, t in zip(long.season, long.week, long.team, strict=True)
    ]
    long["prev_out"] = long.groupby("team").n_out.shift(1)
    long["prev2_out"] = long.groupby("team").n_out.shift(2)
    long["inj_news"] = long.prev_out - long.prev2_out
    L = long.set_index(["game_id", "side"])
    rows = []
    for (s, w), wk in o[o.week >= 2].groupby(["season", "week"]):
        gw = g[(g.season == s) & (g.week == w)].set_index("game_id")
        win = g[(g.slot < slot(s, w)) & (g.slot >= slot(s, w) - WINDOW_WEEKS)]
        hfa, r, resid = fit_ratings(win)
        for _, x in wk.iterrows():
            gid = x.game_id
            if gid not in gw.index:
                continue
            h, a = gw.loc[gid, "home_team"], gw.loc[gid, "away_team"]
            if h not in r or a not in r:
                continue
            implied = hfa + r[h] - r[a]
            lh, la = L.loc[(gid, 1)], L.loc[(gid, -1)]
            rows.append(
                {
                    "game_id": gid,
                    "season": s,
                    "week": w,
                    "hfa": hfa,
                    "fit_sd": resid.std(),
                    "implied": implied,
                    "R": x.tue_open_home_spread - implied,
                    "R_close": x.close_home_spread - implied,
                    "cm": lh.prev_cm - la.prev_cm,
                    "cm_pos": max(lh.prev_cm, 0) - max(la.prev_cm, 0),
                    "cm_neg": min(lh.prev_cm, 0) - min(la.prev_cm, 0),
                    "move": lh.prev_move - la.prev_move,
                    "inj_news": lh.inj_news - la.inj_news,
                    "inj_level": lh.prev_out - la.prev_out,
                    "z": logit(np.clip(x.home_cover_probability_at_open, 1e-6, 1 - 1e-6)),
                    "mvo": x.margin_vs_open,
                }
            )
    d = pd.DataFrame(rows).dropna()
    d = d[d.mvo != 0].copy()
    d["y"] = (d.mvo > 0).astype(int)
    return d


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def week_groups(d):
    keys = (d.season * 100 + d.week).to_numpy()
    return [np.flatnonzero(keys == k) for k in np.unique(keys)]


def week_boot(d, fn, rng):
    groups = week_groups(d)
    out = []
    for _ in range(BOOT):
        pick = rng.choice(len(groups), len(groups))
        out.append(fn(d.iloc[np.concatenate([groups[i] for i in pick])]))
    return np.array(out)


def main():
    d = build()
    rng = np.random.default_rng(25)
    print("n games", len(d), "seasons", d.season.value_counts().sort_index().to_dict())
    print(
        f"rating fit: hfa mean {d.hfa.mean():.3f} sd {d.hfa.std():.3f}; "
        f"in-window resid sd mean {d.fit_sd.mean():.3f}"
    )
    print(
        f"R opener-implied sd {d.R.std():.3f} mean {d.R.mean():.3f}; "
        f"close-implied sd {d.R_close.std():.3f}; "
        f"corr(R,R_close) {np.corrcoef(d.R, d.R_close)[0, 1]:.3f}"
    )
    looks = 0
    print("== (2) revision vs prior game and injuries (game-level R, home minus away) ==")
    for c in ["cm", "cm_pos", "cm_neg", "move", "inj_news", "inj_level"]:
        looks += 1

        def fn(x, c=c):
            return ols(x[[c]].to_numpy(float), x.R.to_numpy(float))[1]

        b = fn(d)
        bs = week_boot(d, fn, rng)
        sh = [fn(d[d.season % 2 == p]) for p in (0, 1)]
        print(
            f"slope R~{c}: {b:.4f} CI[{np.percentile(bs, 2.5):.4f},{np.percentile(bs, 97.5):.4f}] "
            f"P+ {(bs > 0).mean():.3f} halves {sh[0]:.4f}/{sh[1]:.4f}"
        )
    multi = ["cm", "move", "inj_news"]
    d["Rhat_cm"] = 0.0
    d["Rhat_full"] = 0.0
    for s in SEASONS:
        tr, te = d[d.season != s], d.season == s
        b = ols(tr[["cm"]].to_numpy(float), tr.R.to_numpy(float))
        d.loc[te, "Rhat_cm"] = b[0] + b[1] * d.loc[te, "cm"]
        b = ols(tr[multi].to_numpy(float), tr.R.to_numpy(float))
        d.loc[te, "Rhat_full"] = b[0] + d.loc[te, multi].to_numpy(float) @ b[1:]
    d["e_cm"] = d.R - d.Rhat_cm
    d["e_full"] = d.R - d.Rhat_full
    terms = {"R_raw": "R", "e_cm": "e_cm", "e_full": "e_full"}
    for k, c in terms.items():
        d["t_" + k] = d[c] / d[c].std()
    arms = {"intercept": [], "served": ["z"]}
    for k in terms:
        arms["served+" + k] = ["z", "t_" + k]
    P, coef = {}, {}
    for a, cols in arms.items():
        p = pd.Series(0.5, index=d.index)
        coef[a] = {}
        for s in SEASONS:
            tr, te = d[d.season != s], d.season == s
            if not cols:
                p[te] = tr.y.mean()
                continue
            m = LogisticRegression(C=1e6, max_iter=1000).fit(tr[cols], tr.y)
            p[te] = m.predict_proba(d.loc[te, cols])[:, 1]
            coef[a][s] = np.round(m.coef_[0], 3).tolist()
        P[a] = p.clip(1e-6, 1 - 1e-6)
    looks += len(terms)

    def ll(p):
        return -(d.y * np.log(p) + (1 - d.y) * np.log(1 - p))

    print("== (3) LOSO served-logit terms, looks so far", looks, "==")
    for a in arms:
        w = int(((P[a] > 0.5) == (d.y == 1)).sum())
        brier = ((P[a] - d.y) ** 2).mean()
        line = f"{a}: W-L {w}-{len(d) - w} LL {ll(P[a]).mean():.5f} Brier {brier:.5f}"
        if arms[a]:
            m = LogisticRegression(C=1e6, max_iter=1000).fit(d[arms[a]], d.y)
            pin = m.predict_proba(d[arms[a]])[:, 1]
            ins = float(-(d.y * np.log(pin) + (1 - d.y) * np.log(1 - pin)).mean())
            line += f" in-sample LL {ins:.5f} gap {ll(P[a]).mean() - ins:.5f}"
        print(line)
    groups = week_groups(d)
    for k in terms:
        a = "served+" + k
        flip = (P[a] > 0.5) != (P["served"] > 0.5)
        fw = int(((P[a] > 0.5) == (d.y == 1))[flip].sum())
        print(f"{a} DECISIVE flips {int(flip.sum())}: new side wins {fw}-{int(flip.sum()) - fw}")
        dv = (ll(P["served"]) - ll(P[a])).to_numpy()
        bs = np.array(
            [
                np.concatenate([dv[groups[i]] for i in rng.choice(len(groups), len(groups))]).mean()
                for _ in range(BOOT)
            ]
        )
        sb = pd.Series(dv).groupby(d.season.to_numpy()).mean()
        half = [dv[(d.season % 2 == p).to_numpy()].mean() for p in (0, 1)]
        print(
            f"  dLL vs served {dv.mean():.5f} "
            f"CI[{np.percentile(bs, 2.5):.5f},{np.percentile(bs, 97.5):.5f}] "
            f"P+ {(bs > 0).mean():.3f} seasons+ {(sb > 0).sum()}/6 "
            f"halves {half[0]:.5f}/{half[1]:.5f}"
        )
        print("  per-fold coef [z, term]:", coef[a])
    dv0 = (ll(P["intercept"]) - ll(P["served"])).mean()
    print(f"served vs intercept control dLL {dv0:.5f}")
    print("total looks", looks)


main()
