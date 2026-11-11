import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="F:/Repos/nfl_py3/artifacts/sim12/qb_exit.json")
ap.add_argument("--boots", type=int, default=300)
ap.add_argument("--seed", type=int, default=20261111)
args = ap.parse_args()
PBP = "F:/Repos/nfl_py3/data/pbp/raw/20260925T202544Z/season={s}/plays.parquet"
COLS = [
    "game_id",
    "play_id",
    "season_type",
    "posteam",
    "qtr",
    "fixed_drive",
    "play_type",
    "epa",
    "qb_kneel",
    "qb_spike",
    "qb_dropback",
    "passer_player_id",
    "score_differential",
    "game_seconds_remaining",
]
THREE_SCORES = 17
FRACS = (0.25, 0.5, 0.75)
SEASONS = range(2009, 2026)


def split_stats(epa, dr, k):
    pre = epa[dr <= k]
    post = epa[dr > k]
    pre = pre[~np.isnan(pre)]
    post = post[~np.isnan(post)]
    if len(pre) == 0 or len(post) == 0:
        return None
    return post.mean() - pre.mean(), len(post)


events, controls, hazard_rows, pool = [], [], [], []
gcode = {}
for s in SEASONS:
    d = pd.read_parquet(PBP.format(s=s), columns=COLS)
    d = d[(d.season_type == "REG") & d.posteam.notna() & (d.qtr <= 4) & d.fixed_drive.notna()]
    d = d.sort_values(["game_id", "play_id"])
    for (gid, _team), g in d.groupby(["game_id", "posteam"], sort=False):
        if len(g) < 20:
            continue
        gi = gcode.setdefault(gid, len(gcode))
        scr = (
            g.play_type.isin(["pass", "run"])
            & g.epa.notna()
            & (g.qb_kneel != 1)
            & (g.qb_spike != 1)
        )
        epa = np.where(scr, g.epa, np.nan)
        dr = g.fixed_drive.values
        drives = np.unique(dr)
        db = (g.qb_dropback == 1).values & g.passer_player_id.notna().values
        if not db.any():
            continue
        pas = g.passer_player_id.values
        starter = pas[np.flatnonzero(db)[0]]
        other = np.flatnonzero(db & (pas != starter))
        own = np.flatnonzero(db & (pas == starter))
        sd = g.score_differential.values
        qv = g.qtr.values
        exit_flag = 0
        if len(other) and own[-1] < other[0] and (dr > dr[own[-1]]).any():
            r = other[0]
            exit_flag = 1
            res = split_stats(epa, dr, dr[own[-1]])
            blow = int(abs(sd[r]) >= THREE_SCORES)
            hazard_rows.append((s, gi, int(qv[r]), blow, 1, res is not None))
            if res is not None:
                events.append(
                    (
                        s,
                        gi,
                        int(qv[r]),
                        blow,
                        res[0],
                        res[1],
                        int(np.searchsorted(drives, dr[own[-1]])),
                    )
                )
        elif len(other) == 0:
            hazard_rows.append((s, gi, 0, 0, 0, False))
        if exit_flag or len(other):
            continue
        gsr = g.game_seconds_remaining.values
        for q in (1, 2, 3, 4):
            for fr in FRACS:
                mid = 3600 - 900 * (q - 1) - 900 * fr
                idx = np.flatnonzero(gsr >= mid)
                if len(idx) == 0 or idx[-1] + 1 >= len(g):
                    continue
                res = split_stats(epa, dr, dr[idx[-1]])
                if res is not None and abs(sd[idx[-1] + 1]) < THREE_SCORES:
                    controls.append((s, gi, q, res[0]))
        ords = []
        for dv in drives[:-1]:
            res = split_stats(epa, dr, dv)
            if res is None:
                ords.append(np.nan)
                continue
            nxt = np.flatnonzero(dr > dv)[0]
            ords.append(res[0] if abs(sd[nxt]) < THREE_SCORES else np.nan)
        pool.append(np.array(ords))
    print(s, len(events), len(controls), len(pool), flush=True)

ev = pd.DataFrame(events, columns=["season", "g", "q", "blow", "d", "n_post", "ordn"])
ct = pd.DataFrame(controls, columns=["season", "g", "q", "d"])
hz = pd.DataFrame(hazard_rows, columns=["season", "g", "q", "blow", "exit", "scored"])
nb = ev[ev.blow == 0].reset_index(drop=True)
base_all = ct.groupby("q").d.mean()


def wilson(k, n):
    p = k / n
    z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


ntg = len(hz)
haz = {"team_games": ntg, "exits": int(hz.exit.sum())}
haz["overall"] = [round(hz.exit.sum() / ntg, 4), *wilson(int(hz.exit.sum()), ntg)]
haz["blowout_ge17"] = [
    round(((hz.exit == 1) & (hz.blow == 1)).sum() / ntg, 4),
    int(((hz.exit == 1) & (hz.blow == 1)).sum()),
]
haz["non_blowout"] = [
    round(((hz.exit == 1) & (hz.blow == 0)).sum() / ntg, 4),
    int(((hz.exit == 1) & (hz.blow == 0)).sum()),
]
haz["by_quarter_all"] = {
    int(q): [int(c), round(c / ntg, 4)] for q, c in hz[hz.exit == 1].groupby("q").size().items()
}
haz["by_quarter_non_blowout"] = {
    int(q): [int(c), round(c / ntg, 4)]
    for q, c in hz[(hz.exit == 1) & (hz.blow == 0)].groupby("q").size().items()
}
haz["effect_events_scored"] = len(ev), len(nb)
haz["exit_non_scored"] = int(((hz.exit == 1) & (~hz.scored)).sum())

ng = len(gcode)
rng = np.random.default_rng(args.seed)


def effect(nbe, cte, w=None):
    base = cte.groupby("q").apply(
        lambda x: np.average(x.d, weights=None if w is None else w[x.g.values]),
        include_groups=False,
    )
    b = nbe.q.map(base)
    ok = b.notna().values
    ww = np.ones(len(nbe)) if w is None else w[nbe.g.values].astype(float)
    ww = ww * ok
    return float(np.sum(ww * (nbe.d.values - b.fillna(0).values)) / ww.sum())


point = effect(nb, ct)
boots = []
for _ in range(args.boots):
    w = np.bincount(rng.integers(0, ng, ng), minlength=ng)
    boots.append(effect(nb, ct, w))
boots = np.array(boots)
eff = {
    "effect_epa_per_play": round(point, 4),
    "interval95": [
        round(float(np.percentile(boots, 2.5)), 4),
        round(float(np.percentile(boots, 97.5)), 4),
    ],
    "probability_positive": round(float((boots > 0).mean()), 3),
    "n_events": len(nb),
    "mean_post_plays": round(float(nb.n_post.mean()), 2),
    "raw_exit_diff_mean": round(float(nb.d.mean()), 4),
    "control_base_by_q": {int(k): round(float(v), 4) for k, v in base_all.items()},
}

pr = []
cache = {
    int(k): np.array([i for i, a in enumerate(pool) if len(a) > k and not np.isnan(a[k])])
    for k in nb.ordn.unique()
}
for _ in range(args.boots):
    vals = []
    for q, k in zip(nb.q.values, nb.ordn.values, strict=True):
        cand = cache[k]
        vals.append(pool[cand[rng.integers(0, len(cand))]][k] - base_all[q])
    pr.append(np.mean(vals))
pr = np.array(pr)
plc = {
    "mean": round(float(pr.mean()), 4),
    "interval95": [
        round(float(np.percentile(pr, 2.5)), 4),
        round(float(np.percentile(pr, 97.5)), 4),
    ],
    "share_at_or_below_observed": round(float((pr <= point).mean()), 3),
}

loso = {}
for s in SEASONS:
    e = nb[nb.season == s]
    if len(e) == 0:
        continue
    loso[int(s)] = [round(effect(e, ct[ct.season != s]), 4), len(e)]
neg = sum(1 for v in loso.values() if v[0] < 0)
out = {
    "hazard": haz,
    "effect": eff,
    "placebo": plc,
    "loso_by_season": loso,
    "loso_negative_seasons": [neg, len(loso)],
}
post_plays = eff["mean_post_plays"]
nbh = haz["non_blowout"][0]
out["points_per_game"] = {
    "team_hazard_nb": nbh,
    "effect_epa_per_play": eff["effect_epa_per_play"],
    "mean_post_plays": post_plays,
    "both_teams_pts_per_game": round(2 * nbh * eff["effect_epa_per_play"] * post_plays, 4),
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
