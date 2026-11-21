import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "xlg15"
SEED = 2115
BOOT = 2000
PERMS = 2000
SIMS = 400
RHO_GRID = np.round(np.arange(0.02, 0.62, 0.02), 2)
POWER_TARGET = 0.8
DESIGNS = {
    "unit2": {
        "eras": {"2014-2019": (2014, 2019), "2023-2025": (2023, 2025)},
        "wind": ["cfb_game_wind.parquet"],
        "out": "wind_reliability.json",
    },
    "enlarged": {
        "eras": {"2010-2019": (2010, 2019), "2020-2025": (2020, 2025)},
        "wind": ["cfb_game_wind.parquet", "cfb_game_wind_added.parquet"],
        "out": "wind_reliability_enlarged.json",
    },
}
ERAS = DESIGNS["unit2"]["eras"]
WIND = DESIGNS["unit2"]["wind"]
COLS = [
    "game_id",
    "season",
    "week",
    "seasonType",
    "pos_team_id",
    "rush",
    "pass",
    "kneel_down",
    "pass_oe",
    "EPA",
]
rng = np.random.default_rng(SEED)


def load():
    parts = []
    for lo, hi in ERAS.values():
        for s in range(lo, hi + 1):
            fs = sorted(glob.glob(str(ROOT / f"data/cfb/pbp/raw/*/season={s}/plays.parquet")))
            d = pd.read_parquet(fs[-1], columns=COLS)
            d = d[
                (d.seasonType == 2)
                & (d.rush | d["pass"])
                & ~d.kneel_down.fillna(False).astype(bool)
            ]
            parts.append(d)
    p = pd.concat(parts, ignore_index=True)
    tg = (
        p.groupby(["season", "week", "game_id", "pos_team_id"])
        .agg(pass_oe=("pass_oe", "mean"), epa=("EPA", "mean"))
        .reset_index()
    )
    w = pd.concat([pd.read_parquet(OUT / f) for f in WIND], ignore_index=True)
    w = w[~w.dome][["game_id", "wind_mph"]]
    return tg.merge(w, on="game_id")


def wcorr(a, b, w):
    w = w / w.sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    c = (w * (a - ma) * (b - mb)).sum()
    v = np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())
    return float(c / v) if v > 0 else float("nan")


def halves(tg, outcome, era, split):
    lo, hi = era
    d = tg[(tg.season >= lo) & (tg.season <= hi)].dropna(subset=[outcome, "wind_mph"]).copy()
    d = d.sort_values(["season", "pos_team_id", "week", "game_id"])
    if split == "alternating_games":
        d["h"] = d.groupby(["season", "pos_team_id"]).cumcount() % 2
        key = ["season", "pos_team_id", "h"]
    else:
        d["h"] = d.season % 2
        key = ["season", "pos_team_id"]
    d["x"] = d.wind_mph - d.groupby(key).wind_mph.transform("mean")
    d["y"] = d[outcome] - d.groupby(key)[outcome].transform("mean")
    d["xy"] = d.x * d.y
    d["xx"] = d.x**2
    s = (
        d.groupby(["pos_team_id", "h"])
        .agg(sxy=("xy", "sum"), sxx=("xx", "sum"), g=("x", "size"))
        .unstack("h")
    )
    s = s[(s["sxx"].fillna(0) > 0).all(axis=1)]
    a = (s["sxy"][0] / s["sxx"][0]).values
    b = (s["sxy"][1] / s["sxx"][1]).values
    w = (2 * s["sxx"][0] * s["sxx"][1] / (s["sxx"][0] + s["sxx"][1])).values
    pooled = float(d.xy.sum() / d.xx.sum())
    tids = d.pos_team_id.values
    ut = np.unique(tids)
    ix = {t: np.flatnonzero(tids == t) for t in ut}
    pb = []
    for _ in range(BOOT):
        k = np.concatenate([ix[t] for t in rng.choice(ut, len(ut))])
        pb.append(d.xy.values[k].sum() / d.xx.values[k].sum())
    return a, b, w, int(s["g"].sum().sum()), pooled, np.percentile(pb, [2.5, 97.5]).tolist()


def stats(a, b, w):
    r = wcorr(a, b, w)
    n = len(a)
    bs = np.array([wcorr(a[k], b[k], w[k]) for k in (rng.integers(0, n, n) for _ in range(BOOT))])
    nl = np.array([wcorr(a, rng.permutation(b), w) for _ in range(PERMS)])
    p95 = float(np.percentile(nl, 95))
    ab = np.concatenate([a, b])
    ww = np.concatenate([w, w])
    m = (ww * ab).sum() / ww.sum()
    noise = float((ww * (ab - m) ** 2).sum() / ww.sum())
    power = {}
    med = {}
    for rho in RHO_GRID:
        tau = np.sqrt(noise * rho / (1 - rho))
        sim = np.array(
            [wcorr(a + beta, b + beta, w) for beta in (rng.normal(0, tau, n) for _ in range(SIMS))]
        )
        power[float(rho)] = float((sim > p95).mean())
        med[float(rho)] = float(np.median(sim))
    rho80 = next((k for k, v in power.items() if v >= POWER_TARGET), None)
    r80 = None if rho80 is None else med[rho80]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {
        "n_teams": int(n),
        "r_half": r,
        "r_ci": [float(lo), float(hi)],
        "sb": 2 * r / (1 + r),
        "p_pos": float(np.nanmean(bs > 0)),
        "perm_p": float((nl >= r).mean()),
        "perm_95": p95,
        "r80_detectable": r80,
        "rho80_injected": rho80,
        "power": power,
        "median_r_at_rho": med,
        "powered_zero": bool(r80 is not None and hi < r80),
    }


def main():
    global ERAS, WIND
    ap = argparse.ArgumentParser(
        description="CFB team wind-elasticity split-half reliability, planted positive control."
    )
    ap.add_argument(
        "--design", choices=sorted(DESIGNS), default="unit2", help="era design and wind tables"
    )
    args = ap.parse_args()
    ERAS = DESIGNS[args.design]["eras"]
    WIND = DESIGNS[args.design]["wind"]
    tg = load()
    res = {
        "family": "xlg15_cfb_trait_reliability",
        "weather": "realized station wind, Meteostat",
        "design": args.design,
        "eras": ERAS,
        "cells": {},
        "league_slope": {},
    }
    for outcome in ("pass_oe", "epa"):
        for en, era in ERAS.items():
            for split in ("alternating_games", "season_parity"):
                a, b, w, ng, pooled, pci = halves(tg, outcome, era, split)
                st = stats(a, b, w)
                st["team_games"] = ng
                key = f"{outcome}|{en}|{split}"
                res["cells"][key] = st
                res["league_slope"][f"{outcome}|{en}|{split}"] = {"per_mph": pooled, "ci": pci}
                print(
                    key,
                    {k: v for k, v in st.items() if k != "power"},
                    "league",
                    round(pooled, 5),
                    pci,
                    flush=True,
                )
    (OUT / DESIGNS[args.design]["out"]).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
