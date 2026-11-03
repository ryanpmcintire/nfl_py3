import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = "F:/Repos/nfl_py3/"
SEED = 28
LEAD = 8
BOOT = 2000
PERM = 300
REL_START = 2012
EXT_START = 2009
LOSO_FIRST = 2020
LOSO_LAST = 2025
MIN_PRIOR_SEASONS = 3
MIN_GAMES_PER_SEASON = 2
CI_LO = 5
CI_HI = 95
rng = np.random.default_rng(SEED)
pb = pd.read_parquet(
    ROOT + "data/pbp/team_style/raw_pbp_narrow.parquet",
    columns=[
        "game_id",
        "season",
        "posteam",
        "fixed_drive",
        "qtr",
        "yardline_100",
        "game_seconds_remaining",
        "qb_kneel",
        "qb_spike",
        "pass_oe",
        "pass_attempt",
        "rush_attempt",
        "score_differential",
        "posteam_score",
        "posteam_score_post",
        "play_id",
        "aborted_play",
    ],
)
pb = pb[(pb.qtr == 4) & pb.posteam.notna() & pb.fixed_drive.notna()].sort_values(
    ["game_id", "play_id"]
)
pb = pb[(pb.qb_kneel != 1) & (pb.qb_spike != 1) & (pb.aborted_play != 1)]
pb = pb[(pb.pass_attempt == 1) | (pb.rush_attempt == 1)]
rows = []
for (gid, team, dr), d in pb.groupby(["game_id", "posteam", "fixed_drive"], sort=False):
    f = d.iloc[0]
    if not f.score_differential >= LEAD:
        continue
    pts = d.posteam_score_post.iloc[-1] - f.posteam_score
    gs = d.game_seconds_remaining.values
    el = float(gs[0] - gs[-1]) if len(d) > 1 else np.nan
    rows.append(
        (
            gid,
            int(f.season),
            team,
            dr,
            f.yardline_100,
            pts,
            d.pass_oe.mean(),
            el,
            len(d) - 1,
            int(d.pass_oe.notna().sum()),
        )
    )
dv = pd.DataFrame(
    rows, columns=["game_id", "season", "team", "drive", "yl", "pts", "oe", "el", "nint", "noe"]
)
dv = dv.dropna(subset=["yl", "pts"])
b = np.polyfit(dv.yl, dv.pts, 1)
dv["ppd"] = dv.pts - np.polyval(b, dv.yl)
dv["oe_w"] = dv.oe.fillna(0) * dv.noe
dv["el"] = dv.el.fillna(0)
g = (
    dv.groupby(["game_id", "season", "team"])
    .agg(
        ppd=("ppd", "mean"),
        oe_s=("oe_w", "sum"),
        noe=("noe", "sum"),
        el=("el", "sum"),
        nint=("nint", "sum"),
        ndr=("drive", "size"),
    )
    .reset_index()
)
g["oe"] = g.oe_s / g.noe.replace(0, np.nan)
g["sec"] = g.el / g.nint.replace(0, np.nan)
g = g.dropna(subset=["oe", "sec", "ppd"])
TR = {"oe": 1.0, "sec": -1.0, "ppd": 1.0}

j = pd.read_parquet(ROOT + "artifacts/mod24_postseason_rows/base_2011_2019.parquet").reset_index()
ft = pd.read_parquet(
    ROOT + "data/processed/game_features.parquet",
    columns=["game_id", "gameday", "home_team", "away_team", "result"],
)
sb = pd.read_parquet(
    ROOT + "data/processed/sbr_odds.parquet", columns=["game_id", "open_home_spread"]
)
sb = sb.dropna().drop_duplicates("game_id")
pr = j.merge(ft, on="game_id").merge(sb, on="game_id")
pr["kickoff"] = pd.to_datetime(pr.gameday)
pr["y"] = (pr.result - pr.open_home_spread > 0).astype(float)
pr["spread_line"] = pr.open_home_spread
g = g.merge(ft[["game_id", "gameday"]], on="game_id")
g["kickoff"] = pd.to_datetime(g.gameday)
print("rows", len(pr), "pick/y agree", ((pr.pick_home == (pr.y == 1)) == (pr.correct == 1)).mean())


def vc(d, t):
    x = d.set_index(["season", "team"])[t]
    k = x.groupby(level=[0, 1]).size()
    x = x[x.index.map(k) >= MIN_GAMES_PER_SEASON]
    k = x.groupby(level=[0, 1]).size()
    gm = x.groupby(level=[0, 1]).transform("mean")
    sw2 = ((x - gm) ** 2).sum() / (len(x) - len(k))
    N = len(x)
    n0 = (N - (k**2).sum() / N) / (len(k) - 1)
    msb = (k * (x.groupby(level=[0, 1]).mean() - x.mean()) ** 2).sum() / (len(k) - 1)
    return sw2, (msb - sw2) / n0, x.mean()


pr = pr.sort_values("kickoff").reset_index(drop=True)
feat = {t: np.zeros((len(pr), 2)) for t in TR}
gt = {tm: d for tm, d in g.groupby("team")}  # noqa: C416
for s in sorted(pr.season.unique()):
    prior = g[g.season < s]
    if prior.season.nunique() < MIN_PRIOR_SEASONS:
        continue
    par = {t: vc(prior, t) for t in TR}
    for pos in np.where(pr.season.values == s)[0]:
        row = pr.iloc[pos]
        for c, tm in enumerate([row.home_team, row.away_team]):
            w = gt.get(tm)
            if w is None:
                continue
            h = w[w.season.isin([s - 1, s]) & (w.kickoff < row.kickoff)]
            if len(h) == 0:
                continue
            for t in TR:
                sw2, sb2, mu = par[t]
                if sb2 <= 0:
                    continue
                k = sw2 / sb2
                feat[t][pos, c] = TR[t] * ((h[t] - mu).sum() / (len(h) + k)) / np.sqrt(sb2)
    print("as-of", s, {t: round(par[t][0] / max(par[t][1], 1e-9), 1) for t in TR})
for t in TR:
    pr["h_" + t] = feat[t][:, 0]
    pr["a_" + t] = feat[t][:, 1]
for pre in ("h_", "a_"):
    pr[pre + "comp"] = (pr[pre + "oe"] + pr[pre + "sec"] + pr[pre + "ppd"]) / 3
fav_home = pr.spread_line > 0
for t in [*list(TR), "comp"]:
    pr["x_" + t] = np.where(fav_home, pr["h_" + t], -pr["a_" + t])
pr = pr[pr.season >= 2012].copy()
p0 = pr.p.clip(1e-6, 1 - 1e-6)
pr["off"] = np.log(p0 / (1 - p0))
pr["blk"] = pr.season.astype(str) + "-" + pr.week.astype(str)
FIRST = 2012
rng = np.random.default_rng(SEED)


def fit(x, off, y):
    def f(bb):
        return np.sum(np.logaddexp(0, off + bb * x) - y * (off + bb * x))

    return minimize_scalar(f, bounds=(-3, 3), method="bounded").x


LAST = 2019


def scored(p, y):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return (
        -(y * np.log(pc) + (1 - y) * np.log(1 - pc)),
        (pc - y) ** 2,
        ((p >= 0.5) == (y == 1)).astype(float),
    )


def run(sub, col):
    oos = []
    coefs = {}
    for s in range(FIRST, LAST + 1):
        tr = sub[sub.season != s]
        te = sub[sub.season == s]
        if len(te) == 0:
            continue
        sd = tr[col].std()
        bb = fit(tr[col].values / sd, tr.off.values, tr.y.values)
        coefs[s] = round(float(bb), 4)
        t = te.copy()
        t["pm"] = 1 / (1 + np.exp(-(te.off.values + bb * te[col].values / sd)))
        oos.append(t)
    o = pd.concat(oos)
    sd = sub[col].std()
    ball = fit(sub[col].values / sd, sub.off.values, sub.y.values)
    pin = 1 / (1 + np.exp(-(sub.off.values + ball * sub[col].values / sd)))
    lm, bm, am = scored(o.pm.values, o.y.values)
    lb, bbr, ab = scored(o.p.values, o.y.values)
    li, _, _ = scored(pin, sub.y.values)
    lbi, _, _ = scored(sub.p.values, sub.y.values)
    d = lb - lm
    blocks = o.blk.values
    ub = np.unique(blocks)
    idx = {b: np.where(blocks == b)[0] for b in ub}
    bt = []
    for _ in range(BOOT):
        pick = rng.choice(ub, len(ub))
        ii = np.concatenate([idx[b] for b in pick])
        bt.append(d[ii].mean())
    bt = np.array(bt)
    flip = (o.pm.values >= 0.5) != (o.p.values >= 0.5)
    r = {
        "n": len(o),
        "acc_model": float(am.mean()),
        "acc_base": float(ab.mean()),
        "record_model": f"{int(am.sum())}-{int(len(am) - am.sum())}",
        "record_base": f"{int(ab.sum())}-{int(len(ab) - ab.sum())}",
        "flips": int(flip.sum()),
        "flip_model_wins": int(am[flip].sum()),
        "flip_base_wins": int(ab[flip].sum()),
        "ll_model": float(lm.mean()),
        "ll_base": float(lb.mean()),
        "dLL": float(d.mean()),
        "dBrier": float((bbr - bm).mean()),
        "dLL_ci90": [float(np.percentile(bt, 5)), float(np.percentile(bt, 95))],
        "p_pos": float((bt > 0).mean()),
        "blocks": len(ub),
        "coefs": coefs,
        "coef_all": round(float(ball), 4),
        "in_sample_dLL": float((lbi - li).mean()),
    }
    r["gap_in_minus_oos"] = r["in_sample_dLL"] - r["dLL"]
    return r


valid = pr[pr.off.notna()]
res = {}
for name, sub in (("abs_spread_ge7", valid[valid.spread_line.abs() >= 7]), ("all", valid)):
    res["oe_" + name] = run(sub, "x_oe")
    print(name, json.dumps(res["oe_" + name]), flush=True)
with open(sys.argv[1], "w") as fh:
    json.dump(res, fh, indent=1)
