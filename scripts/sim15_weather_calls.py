import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PBP = ROOT / "data" / "pbp" / "raw" / "20260929T191306Z"
FULL = ROOT / "data" / "raw" / "nflverse_pbp_full" / "20261103T131922Z"
FC = (
    ROOT
    / "data"
    / "raw"
    / "forecast_archive"
    / "pool_decision_2009_2025_aware"
    / "forecasts.parquet"
)
OUT = ROOT / "artifacts" / "sim15"
NBOOT = 300
FAMILY = "sim15_weather_calls"
DEEP_AIR_YARDS = 20
COLS = [
    "play_id",
    "game_id",
    "season",
    "season_type",
    "home_team",
    "posteam",
    "down",
    "play_type",
    "ydstogo",
    "yardline_100",
    "game_seconds_remaining",
    "score_differential",
    "epa",
    "qb_kneel",
    "qb_spike",
    "qtr",
]
CT = ["pos_home", "z_trail", "z_lead", "el", "el2", "d2", "d3", "d4", "logyd", "yl", "yl2"]
CTF = ["pos_home", "z_trail", "z_lead", "el", "logyd", "yl"]
OUTCOMES = [
    "pass_rate",
    "deep_share",
    "fg_le20",
    "fg_21_30",
    "fg_31_40",
    "epa_total",
    "epa_fixed_call",
]
SPECS = ["main_observed", "outdoor_forecast", "placebo_roof_forecast"]
TERMS = ["wind10", "temp10"]
DECLARED_LOOKS = len(OUTCOMES) * len(SPECS) * len(TERMS)


def load():
    fc = pd.read_parquet(FC)
    fc = fc[fc.fetch_status == "ok"][
        [
            "game_id",
            "roof",
            "actual_temp_f",
            "actual_wind_mph",
            "forecast_temp_f",
            "forecast_wind_mph",
        ]
    ].drop_duplicates("game_id")
    fc = fc.rename(
        columns={"roof": "roof_state", "actual_temp_f": "temp", "actual_wind_mph": "wind"}
    )
    d = pd.concat(
        [
            pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=COLS)
            for s in range(2009, 2026)
        ],
        ignore_index=True,
    )
    ay = pd.concat(
        [
            pd.read_parquet(FULL / f"pbp_{s}.parquet", columns=["game_id", "play_id", "air_yards"])
            for s in range(2009, 2018)
        ],
        ignore_index=True,
    )
    d = d.merge(ay, on=["game_id", "play_id"], how="left")
    d = d[
        (d.season_type == "REG")
        & d.posteam.notna()
        & d.score_differential.notna()
        & d.down.notna()
        & d.yardline_100.notna()
    ].copy()
    d = d.merge(fc, on="game_id", how="left")
    t = d.game_seconds_remaining.clip(lower=0)
    z = d.score_differential.astype(float) / np.sqrt(t / 60 + 1)
    d["z_trail"], d["z_lead"] = z.clip(upper=0), z.clip(lower=0)
    d["el"] = (3600 - t) / 3600
    d["el2"] = d.el**2
    for k in (2, 3, 4):
        d[f"d{k}"] = (d.down == k).astype(float)
    d["logyd"] = np.log(d.ydstogo.clip(lower=1))
    d["yl"] = d.yardline_100 / 100
    d["yl2"] = d.yl**2
    d["pos_home"] = (d.posteam == d.home_team).astype(float)
    d["tsk"] = d.posteam + "_" + d.season.astype(str)
    indoor = d.roof_state.isin(["dome", "closed"])
    out = d.roof_state.isin(["outdoors", "open"])
    fcok = d.forecast_temp_f.notna() & d.forecast_wind_mph.notna()
    d["spec_main"] = out & d.temp.notna() & d.wind.notna()
    d["spec_outf"] = out & fcok
    d["spec_plac"] = indoor & fcok
    return d


def frame(d, outcome):
    pr = d.play_type.isin(["pass", "run"]) & (d.qb_kneel != 1) & (d.qb_spike != 1) & (d.qtr <= 4)
    if outcome == "pass_rate":
        s = d[pr].copy()
        s["y"] = (s.play_type == "pass").astype(float)
        return s, CT
    if outcome == "deep_share":
        s = d[pr & (d.play_type == "pass") & d.air_yards.notna()].copy()
        s["y"] = (s.air_yards >= DEEP_AIR_YARDS).astype(float)
        return s, CT
    if outcome.startswith("fg_"):
        dec = (
            d.play_type.isin(["pass", "run", "punt", "field_goal"])
            & (d.down == 4)
            & (d.qb_kneel != 1)
            & (d.qtr <= 4)
        )
        band = {"fg_le20": (0, 20), "fg_21_30": (21, 30), "fg_31_40": (31, 40)}[outcome]
        s = d[dec & d.yardline_100.between(*band)].copy()
        s["y"] = (s.play_type == "field_goal").astype(float)
        return s, CTF
    s = d[pr & d.epa.notna()].copy()
    s["y"] = s.epa.astype(float)
    s["pass"] = (s.play_type == "pass").astype(float)
    return s, CT


def weather(s, spec):
    if spec == "main_observed":
        s = s[s.spec_main].copy()
        s["wind10"], s["temp10"] = s.wind / 10, s.temp / 10
    else:
        s = s[s["spec_outf" if spec == "outdoor_forecast" else "spec_plac"]].copy()
        s["wind10"], s["temp10"] = s.forecast_wind_mph / 10, s.forecast_temp_f / 10
    return s


def design(s, controls, fixed):
    X = s[TERMS + controls].astype(float).copy()
    if fixed:
        pc = s["pass"] - s["pass"].mean()
        X["pass"] = pc
        X["pass_x_wind"] = pc * (s.wind10 - s.wind10.mean())
        X["pass_x_temp"] = pc * (s.temp10 - s.temp10.mean())
    return X


def within(X, y, key):
    X = X - X.groupby(key.values).transform("mean")
    y = y - y.groupby(key.values).transform("mean")
    return X, y


def stats(s, X, y):
    gid = pd.factorize(s.game_id)[0]
    ng = gid.max() + 1
    Xa, ya = X.values, y.values
    k = Xa.shape[1]
    xx = np.zeros((ng, k, k))
    xy = np.zeros((ng, k))
    order = np.argsort(gid, kind="stable")
    bounds = np.searchsorted(gid[order], np.arange(ng + 1))
    for g in range(ng):
        ix = order[bounds[g] : bounds[g + 1]]
        xx[g] = Xa[ix].T @ Xa[ix]
        xy[g] = Xa[ix].T @ ya[ix]
    return xx, xy, ng


def solve(xx, xy, w=None):
    A = xx.sum(0) if w is None else np.tensordot(w, xx, 1)
    b = xy.sum(0) if w is None else w @ xy
    return np.linalg.lstsq(A, b, rcond=1e-10)[0]


def fit(s, controls, fixed, rng):
    X, y = within(design(s, controls, fixed), s.y.astype(float), s.tsk)
    keep = [c for c in X.columns if X[c].std() > 1e-9]
    xx, xy, ng = stats(s, X[keep], y)
    b = solve(xx, xy)
    draws = np.array(
        [solve(xx, xy, np.bincount(rng.integers(0, ng, ng), minlength=ng)) for _ in range(NBOOT)]
    )
    return keep, b, draws


def summarize(keep, b, draws, term):
    j = keep.index(term)
    lo, hi = np.percentile(draws[:, j], [2.5, 97.5])
    return float(b[j]), float(lo), float(hi), float((draws[:, j] > 0).mean())


def loso(s, controls, fixed):
    rows = []
    for season in sorted(s.season.unique()):
        tr = s[s.season != season]
        X, y = within(design(tr, controls, fixed), tr.y.astype(float), tr.tsk)
        keep = [c for c in X.columns if X[c].std() > 1e-9]
        b = np.linalg.lstsq(X[keep].values, y.values, rcond=1e-10)[0]
        te = s[s.season == season]
        Xt, yt = within(design(te, controls, fixed), te.y.astype(float), te.tsk)
        full = Xt[keep].values
        base_idx = [i for i, c in enumerate(keep) if c not in TERMS]
        sse_full = float(((yt.values - full @ b) ** 2).sum())
        sse_base = float(((yt.values - full[:, base_idx] @ b[base_idx]) ** 2).sum())
        rows.append(
            {
                "season": int(season),
                **{t: float(b[keep.index(t)]) for t in TERMS if t in keep},
                "sse_gain": sse_base - sse_full,
            }
        )
    return rows


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    cells, folds = [], []
    for oc in OUTCOMES:
        base, controls = frame(d, oc)
        for sp in SPECS:
            s = weather(base, sp).dropna(subset=[*TERMS, *controls, "y"])
            fixed = oc == "epa_fixed_call"
            keep, b, draws = fit(s, controls, fixed, np.random.default_rng(20261103))
            for t in TERMS:
                est, lo, hi, pp = summarize(keep, b, draws, t)
                cells.append(
                    {
                        "family": FAMILY,
                        "outcome": oc,
                        "spec": sp,
                        "term": t,
                        "est": est,
                        "lo": lo,
                        "hi": hi,
                        "p_pos": pp,
                        "n": len(s),
                        "games": int(s.game_id.nunique()),
                    }
                )
            if sp == "main_observed" or oc in ("pass_rate", "deep_share"):
                for r in loso(s, controls, fixed):
                    folds.append({"outcome": oc, "spec": sp, **r})
            print(oc, sp, len(s), flush=True)
    pd.DataFrame(cells).to_csv(OUT / "cells.csv", index=False)
    pd.DataFrame(folds).to_csv(OUT / "loso.csv", index=False)
    print("looks", len(cells), "declared", DECLARED_LOOKS)


def behaviour():
    d = load()
    base, controls = frame(d, "epa_total")
    rows = []
    for sp in SPECS:
        s = weather(base, sp).dropna(subset=[*TERMS, *controls, "y"])
        k1, b1, d1 = fit(s, controls, False, np.random.default_rng(20261104))
        k2, b2, d2 = fit(s, controls, True, np.random.default_rng(20261104))
        for t in TERMS:
            beh = d1[:, k1.index(t)] - d2[:, k2.index(t)]
            lo, hi = np.percentile(beh, [2.5, 97.5])
            rows.append(
                {
                    "spec": sp,
                    "term": t,
                    "total": float(b1[k1.index(t)]),
                    "direct": float(b2[k2.index(t)]),
                    "behavioural": float(b1[k1.index(t)] - b2[k2.index(t)]),
                    "lo": float(lo),
                    "hi": float(hi),
                    "p_pos": float((beh > 0).mean()),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "behaviour.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    {"run": run, "behaviour": behaviour}[sys.argv[1]]()
