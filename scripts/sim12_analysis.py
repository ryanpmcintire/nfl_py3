import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="tests/scratch/sim12")
ap.add_argument("--groups", default="QB,OL,WR1,DB1")
ap.add_argument("--boot", type=int, default=300)
ap.add_argument("--placebo", type=int, default=300)
args = ap.parse_args()
ev = pd.read_parquet(args.dir + "/ev.parquet")
ev["q"] = ev.q.clip(upper=3)
ct = pd.read_parquet(args.dir + "/ct.parquet")
hz = pd.read_parquet(args.dir + "/hz.parquet")
if "frac" not in ct:
    ct["frac"] = 0.5
SIGN = {"QB": 1, "OL": 1, "WR1": 1, "CB1": -1, "DB1": -1}
SLOTS = {"OL": 5}
MET = ["d_epa", "d_ppd"]


def ref_means(c):
    c = c[c.frac == 0.5]
    return c.groupby(["grp", "q", "st"])[MET].mean(), c.groupby(["grp", "q"])[MET].mean()


def did(e, c):
    a, b = ref_means(c)
    out = []
    for r in e.itertuples():
        k = (r.grp, r.q, r.st)
        m = a.loc[k] if k in a.index else b.loc[(r.grp, r.q)]
        out.append((r.d_epa - m.d_epa, r.d_ppd - m.d_ppd))
    return np.array(out)


def stat(e, c, h, grp):
    x = did(e, c)
    hz_ = h.exit.mean()
    imp = SIGN[grp] * hz_ * SLOTS.get(grp, 1) * x[:, 1].mean() * e.n_post.mean()
    return x[:, 0].mean(), x[:, 1].mean(), hz_, e.n_post.mean(), imp


def by_game(d):
    return {g: x for g, x in d.groupby("game")}  # noqa: C416


rng = np.random.default_rng(12)
rows = []
for grp in args.groups.split(","):
    e0 = ev[ev.grp == grp].reset_index(drop=True)
    c0 = ct[ct.grp == grp].reset_index(drop=True)
    h0 = hz[hz.grp == grp].reset_index(drop=True)
    for cell in ["all", 0, 1, 2]:
        e = e0 if cell == "all" else e0[e0.st == cell]
        if len(e) < 5:
            rows.append((grp, cell, len(e)))
            continue
        pt = stat(e, c0, h0, grp)
        gs = h0.game.unique()
        eg, cg, hg = by_game(e), by_game(c0), by_game(h0)
        bs = []
        for _ in range(args.boot):
            pick = rng.choice(gs, len(gs))
            ee = pd.concat([eg[g] for g in pick if g in eg])
            cc = pd.concat([cg[g] for g in pick if g in cg])
            hh = pd.concat([hg[g] for g in pick])
            bs.append(stat(ee, cc, hh, grp))
        bs = np.array(bs)
        lo = np.percentile(bs, 2.5, axis=0)
        hi = np.percentile(bs, 97.5, axis=0)
        pp = (bs > 0).mean(0)
        rows.append((grp, cell, len(e), *pt, *lo, *hi, *pp))
cols = ["grp", "cell", "n_ev", "dEPA", "dPPD", "haz", "npost", "impact"]
full = cols + [f"{c}_{s}" for s in ("lo", "hi", "pp") for c in cols[3:]]
df = pd.DataFrame([r + (np.nan,) * (len(full) - len(r)) for r in rows], columns=full)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(df.round(3).to_string())
df.to_csv(args.dir + "/cells.csv", index=False)
print("HAZARD")
for grp, d in hz.groupby("grp"):
    n = len(d)
    k = d.exit.sum()
    p = k / n
    z = 1.96
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    wl = (p + z * z / (2 * n) - r) / (1 + z * z / n)
    wh = (p + z * z / (2 * n) + r) / (1 + z * z / n)
    print(grp, n, k, round(p, 4), round(wl, 4), round(wh, 4))
print("events/controls", len(ev), len(ct))
print("LOSO per fold (reference means from other seasons only)")
for grp in args.groups.split(","):
    e0 = ev[ev.grp == grp]
    c0 = ct[ct.grp == grp]
    if len(e0) < 5:
        continue
    folds, parts = [], []
    for s in sorted(e0.season.unique()):
        es = e0[e0.season == s]
        if len(es) == 0:
            continue
        x = did(es, c0[c0.season != s])
        folds.append((s, len(es), x[:, 0].mean(), x[:, 1].mean()))
        parts.append(x)
    f = pd.DataFrame(folds, columns=["season", "n", "dEPA", "dPPD"])
    allx = np.vstack(parts)
    print(grp, "pooled LOSO dEPA", round(allx[:, 0].mean(), 4), "dPPD", round(allx[:, 1].mean(), 4))
    print(f.round(3).to_string(index=False))
    print("folds negative dEPA:", int((f.dEPA < 0).sum()), "of", len(f))
print("PLACEBO pseudo-exit at quarter fractions 0.25 and 0.75, disjoint game halves")
for grp in args.groups.split(","):
    e0 = ev[ev.grp == grp]
    c0 = ct[ct.grp == grp]
    if len(e0) < 5 or c0.frac.nunique() < 3:
        continue
    obs = did(e0, c0)[:, 0].mean()
    games = c0.game.unique()
    pl = []
    for _ in range(args.placebo):
        half = set(rng.choice(games, len(games) // 2, replace=False))
        a = c0[c0.game.isin(half) & (c0.frac != 0.5)]
        b = c0[~c0.game.isin(half)]
        pl.append(did(a, b)[:, 0].mean())
    pl = np.array(pl)
    print(
        grp,
        "observed",
        round(obs, 4),
        "placebo mean",
        round(pl.mean(), 4),
        "2.5-97.5",
        np.percentile(pl, [2.5, 97.5]).round(4).tolist(),
        "P(placebo<=obs)",
        round((pl <= obs).mean(), 4),
    )
