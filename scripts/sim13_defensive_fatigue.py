import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim09_dynamics import CT, FIT, HELD, load, prplays

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "sim13"
RNG = np.random.default_rng(20261103)
NBOOT = 400
FAMILY = "sim13_defensive_fatigue"
LOOKS = 0


def drive_table(d):
    pr = d.play_type.isin(["pass", "run"]) & (d.qb_kneel != 1) & (d.qb_spike != 1)
    x = d[d.fixed_drive.notna() & d.posteam.notna() & d.defteam.notna()].copy()
    x["pr"] = pr.loc[x.index].astype(int)
    x["t"] = x.game_seconds_remaining.clip(lower=0)
    g = x.groupby(["game_id", "fixed_drive"])
    t = g.agg(
        off=("posteam", "first"),
        dfn=("defteam", "first"),
        first=("play_id", "min"),
        plays=("pr", "sum"),
        t0=("t", "max"),
        t1=("t", "min"),
    ).reset_index()
    t["sec"] = (t.t0 - t.t1).clip(lower=0)
    return t.sort_values(["game_id", "first"]).reset_index(drop=True)


def cum_before(t, team_col, val):
    return t.groupby(["game_id", team_col])[val].cumsum() - t[val]


def other_side(t, lookup_col, other_col, val):
    out = np.zeros(len(t))
    idx = {}
    for (g, tm), s in t.groupby(["game_id", other_col]):
        idx[(g, tm)] = (s["first"].values, (s[val].cumsum() - s[val]).values, s[val].sum())
    gf, fr, tm = t.game_id.values, t["first"].values, t[lookup_col].values
    for i in range(len(t)):
        a = idx.get((gf[i], tm[i]))
        if a is None:
            continue
        k = np.searchsorted(a[0], fr[i])
        out[i] = a[1][k] if k < len(a[0]) else a[2]
    return out


def build():
    d = load()
    t = drive_table(d)
    t["def_plays_before"] = cum_before(t, "dfn", "plays")
    t["off_plays_before"] = cum_before(t, "off", "plays")
    t["def_own_off_pl"] = other_side(t, "dfn", "off", "plays")
    t["def_own_off_sec"] = other_side(t, "dfn", "off", "sec")
    t["off_own_def_pl"] = other_side(t, "off", "dfn", "plays")
    t["dkey"] = t.game_id + "|" + t.dfn
    t["okey"] = t.game_id + "|" + t.off
    keep = [
        "game_id",
        "fixed_drive",
        "def_plays_before",
        "off_plays_before",
        "def_own_off_pl",
        "def_own_off_sec",
        "off_own_def_pl",
        "dkey",
        "okey",
    ]
    p = (
        prplays(d)
        .merge(t[keep], on=["game_id", "fixed_drive"], how="left")
        .dropna(subset=["def_plays_before"])
    )
    p["cum_def"] = p.def_plays_before / 10
    p["cum_off"] = p.off_plays_before / 10
    p["own_off_pl"] = p.def_own_off_pl / 10
    p["own_off_min"] = p.def_own_off_sec / 600
    p["own_def_pl"] = p.off_own_def_pl / 10
    p["succ"] = p.success.astype(float)
    return p


def cell(s, terms, controls, y, key):
    s = s.dropna(subset=[*terms, *controls, y])
    X = s[terms + controls].astype(float)
    yy = s[y].astype(float)
    X = X - X.groupby(s[key].values).transform("mean")
    yy = yy - yy.groupby(s[key].values).transform("mean")
    X = X[[c for c in X.columns if X[c].std() > 1e-9]]
    cols = list(X.columns)
    gid = pd.factorize(s.game_id)[0]
    ng = gid.max() + 1
    Xa, ya = X.values, yy.values
    k = Xa.shape[1]
    xx = np.zeros((ng, k, k))
    xy = np.zeros((ng, k))
    order = np.argsort(gid, kind="stable")
    bounds = np.searchsorted(gid[order], np.arange(ng + 1))
    for g in range(ng):
        ix = order[bounds[g] : bounds[g + 1]]
        xx[g] = Xa[ix].T @ Xa[ix]
        xy[g] = Xa[ix].T @ ya[ix]
    b = np.linalg.lstsq(xx.sum(0), xy.sum(0), rcond=1e-10)[0]
    draws = np.array(
        [
            np.linalg.lstsq(np.tensordot(w, xx, 1), w @ xy, rcond=1e-10)[0]
            for w in (np.bincount(RNG.integers(0, ng, ng), minlength=ng) for _ in range(NBOOT))
        ]
    )
    out = []
    for term in terms:
        j = cols.index(term)
        out.append(
            {
                "term": term,
                "y": y,
                "n": len(s),
                "games": int(ng),
                "coef": b[j],
                "lo": np.percentile(draws[:, j], 2.5),
                "hi": np.percentile(draws[:, j], 97.5),
                "prob_pos": float((draws[:, j] > 0).mean()),
            }
        )
    return out


def main1():
    global LOOKS
    OUT.mkdir(parents=True, exist_ok=True)
    p = build()
    base = [c for c in CT if c != "pos_home"]
    mc = [*base, "own_off_pl", "own_off_min"]
    pc = [*base, "own_def_pl"]
    rows = []
    for en, (a, b) in {"all": (2009, 2025), "fit": FIT, "held": HELD}.items():
        q = p[(p.season >= a) & (p.season <= b)]
        for qn, qq in (
            ("all", q),
            ("Q1", q[q.qtr == 1]),
            ("Q2", q[q.qtr == 2]),
            ("Q3", q[q.qtr == 3]),
            ("Q4", q[q.qtr == 4]),
        ):
            if en != "all" and qn != "all":
                continue
            for y in ("epa", "succ"):
                for spec, terms, ctrl, key in (
                    ("main", ["cum_def"], mc, "dkey"),
                    ("placebo", ["cum_off"], pc, "okey"),
                ):
                    LOOKS += 1
                    for r in cell(qq, terms, ctrl, y, key):
                        r.update(era=en, qtr=qn, spec=spec)
                        rows.append(r)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "cells.csv", index=False)
    pd.set_option("display.width", 200)
    print(FAMILY, "looks", LOOKS, "per 10 extra snaps")
    print(
        r[["spec", "era", "qtr", "y", "n", "coef", "lo", "hi", "prob_pos"]]
        .round(4)
        .to_string(index=False)
    )


def add_clock(p):
    p = p.copy()
    p["qsec"] = (p.t - (4 - p.qtr) * 900).clip(0, 900) / 900
    for q in (1, 2, 3, 4):
        m = (p.qtr == q).astype(float)
        p[f"qf{q}"] = m
        p[f"qc{q}"] = m * p.qsec
        p[f"qc2{q}"] = m * p.qsec**2
    return p


def unit2_clock(p):
    global LOOKS
    base = [c for c in CT if c != "pos_home"]
    p = add_clock(p)
    p["qsec2"] = p.qsec**2
    rows = []
    for qn in ("all", 1, 2, 3, 4):
        qq = p if qn == "all" else p[p.qtr == qn]
        if qn == "all":
            ctl = (
                [c for c in base if c not in ("el", "el2")]
                + [f"qf{q}" for q in (2, 3, 4)]
                + [f"qc{q}" for q in (1, 2, 3, 4)]
                + [f"qc2{q}" for q in (1, 2, 3, 4)]
            )
        else:
            ctl = [c for c in base if c not in ("el", "el2")] + ["qsec", "qsec2"]
        for spec, terms, extra, key in (
            ("main", ["cum_def"], ["own_off_pl", "own_off_min"], "dkey"),
            ("placebo", ["cum_off"], ["own_def_pl"], "okey"),
        ):
            LOOKS += 1
            for r in cell(qq, terms, ctl + extra, "epa", key):
                r.update(block="clock", spec=spec, qtr=str(qn))
                rows.append(r)
    return rows


def drive_level(p):
    f = p.groupby(["game_id", "fixed_drive"], sort=False).first().reset_index()
    res = f.fixed_drive_result.fillna("")
    f["scored"] = res.isin(["Touchdown", "Field goal"]).astype(float)
    f["points"] = (res == "Touchdown") * 7.0 + (res == "Field goal") * 3.0
    return f


def unit2_drive(p):
    global LOOKS
    f = drive_level(p)
    base = [
        "off_epa",
        "opp_off",
        "own_def",
        "def_epa",
        "z_trail",
        "z_lead",
        "el",
        "el2",
        "yl",
        "yl2",
    ]
    rows = []
    for en, (a, b) in {"all": (2009, 2025), "fit": FIT, "held": HELD}.items():
        q = f[(f.season >= a) & (f.season <= b)]
        for y in ("points", "scored"):
            for spec, terms, extra, key in (
                ("main", ["cum_def"], ["own_off_pl", "own_off_min"], "dkey"),
                ("placebo", ["cum_off"], ["own_def_pl"], "okey"),
            ):
                if en != "all" and spec == "placebo":
                    continue
                LOOKS += 1
                for r in cell(q, terms, base + extra, y, key):
                    r.update(block="drive", era=en, spec=spec)
                    rows.append(r)
    return rows


def unit2_dose(p):
    global LOOKS
    base = [c for c in CT if c != "pos_home"]
    k1, k2 = p.cum_def.quantile([1 / 3, 2 / 3]).values
    kk1, kk2 = p.cum_off.quantile([1 / 3, 2 / 3]).values
    p = p.copy()
    for nm, c, a, b in (("d", "cum_def", k1, k2), ("o", "cum_off", kk1, kk2)):
        p[f"{nm}s1"] = p[c].clip(upper=a)
        p[f"{nm}s2"] = (p[c] - a).clip(0, b - a)
        p[f"{nm}s3"] = (p[c] - b).clip(lower=0)
    rows = []
    for spec, terms, extra, key in (
        ("main", ["ds1", "ds2", "ds3"], ["own_off_pl", "own_off_min"], "dkey"),
        ("placebo", ["os1", "os2", "os3"], ["own_def_pl"], "okey"),
    ):
        LOOKS += 3
        for r in cell(p, terms, base + extra, "epa", key):
            r.update(block="dose", spec=spec, knot1=k1 * 10, knot2=k2 * 10)
            rows.append(r)
    return rows


def unit2_loso(p):
    global LOOKS
    LOOKS += 1
    base = [c for c in CT if c != "pos_home"]
    cols = ["cum_def", *base, "own_off_pl", "own_off_min"]
    s = p.dropna(subset=[*cols, "epa"])
    X = s[cols].astype(float)
    X = X - X.groupby(s.dkey.values).transform("mean")
    y = s.epa.astype(float)
    y = y - y.groupby(s.dkey.values).transform("mean")
    Xa, ya, se = X.values, y.values, s.season.values
    rows = []
    for yr in sorted(set(se)):
        tr, te = se != yr, se == yr
        b = np.linalg.lstsq(Xa[tr].T @ Xa[tr], Xa[tr].T @ ya[tr], rcond=1e-10)[0]
        b0 = b.copy()
        b0[0] = 0.0
        e1 = ((ya[te] - Xa[te] @ b) ** 2).sum()
        e0 = ((ya[te] - Xa[te] @ b0) ** 2).sum()
        rows.append(
            {
                "block": "loso",
                "season": int(yr),
                "beta": b[0],
                "n_test": int(te.sum()),
                "sse_gain": e0 - e1,
            }
        )
    return rows


def main2():
    OUT.mkdir(parents=True, exist_ok=True)
    p = build()
    out = []
    for fn in (unit2_clock, unit2_drive, unit2_dose):
        out += fn(p)
    r = pd.DataFrame(out)
    r.to_csv(OUT / "cells_unit2.csv", index=False)
    lo = pd.DataFrame(unit2_loso(p))
    lo.to_csv(OUT / "loso.csv", index=False)
    pd.set_option("display.width", 220)
    print(FAMILY, "unit2 looks", LOOKS)
    print(r.drop(columns=["games"], errors="ignore").round(4).to_string(index=False))
    print(lo.round(5).to_string(index=False))
    print(
        "beta mean",
        lo.beta.mean(),
        "sd",
        lo.beta.std(),
        "min",
        lo.beta.min(),
        "max",
        lo.beta.max(),
        "pos folds",
        int((lo.beta > 0).sum()),
        "of",
        len(lo),
        "oos sse gain total",
        lo.sse_gain.sum(),
        "folds gain>0",
        int((lo.sse_gain > 0).sum()),
    )


if __name__ == "__main__":
    main2() if "unit2" in sys.argv else main1()
