import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sky04_lead_lag as L

SEASONS = (2023, 2024, 2025)
SUNDAY = 6
CUTOFF_HOUR = 16
OUT = L.OUT.parent / "mkt21"
FEATURES = ("reversals", "last_move_fraction", "efficiency")


def deadline_ts(gameday, kickoff):
    d = pd.Timestamp(gameday)
    sun = (d - pd.Timedelta(days=(d.weekday() - SUNDAY) % 7)).replace(hour=CUTOFF_HOUR)
    sun = sun.tz_localize(L.ET).tz_convert("UTC")
    ko = pd.Timestamp(kickoff)
    ko = ko.tz_localize("UTC") if ko.tzinfo is None else ko.tz_convert("UTC")
    if d.weekday() == SUNDAY:
        return min(sun, ko).timestamp()
    return ko.timestamp()


def path_features(o, M, t0, t1):
    k0 = np.searchsorted(o, t0, side="right") - 1
    k1 = np.searchsorted(o, t1, side="right") - 1
    if k0 < 0 or k1 <= k0:
        return None
    F = pd.DataFrame(M[k0 : k1 + 1]).ffill().to_numpy()
    with np.errstate(all="ignore"):
        med = np.nanmedian(F, axis=1)
    ok = ~np.isnan(med)
    if ok.sum() < 3:
        return None
    med, ts = med[ok], o[k0 : k1 + 1][ok]
    d = np.diff(med)
    nz = d != 0
    if not nz.any():
        return {
            "reversals": 0.0,
            "last_move_fraction": 0.0,
            "efficiency": 0.0,
            "net": 0.0,
            "path": 0.0,
            "snaps": int(ok.sum()),
        }
    sg = np.sign(d[nz])
    rev = int((sg[1:] != sg[:-1]).sum())
    last = ts[1:][nz][-1]
    path = float(np.abs(d).sum())
    net = float(med[0] - med[-1])
    return {
        "reversals": float(rev),
        "last_move_fraction": float((last - t0) / (t1 - t0)),
        "efficiency": abs(net) / path,
        "net": net,
        "path": path,
        "snaps": int(ok.sum()),
    }


def main():
    OUT.mkdir(exist_ok=True)
    s = L.book_series()
    s = s[s.nflverse_game_id.str[:4].isin([str(x) for x in SEASONS])]
    part = s.sort_values(["nflverse_game_id", "bookmaker_key", "observed_at_utc"]).copy()
    part["chg"] = (
        part.groupby(["nflverse_game_id", "bookmaker_key"]).home_spread_line.diff().fillna(0) != 0
    )
    pc = part.groupby("bookmaker_key").chg.sum()
    books = [b for b in pc.index if pc[b] >= L.MIN_PARTICIPATION]
    inv = {}
    for yr in SEASONS:
        sy = s[s.nflverse_game_id.str[:4] == str(yr)]
        gaps = (
            sy.sort_values("observed_at_utc")
            .groupby("nflverse_game_id")
            .observed_at_utc.apply(lambda x: x.diff().dt.total_seconds().median())
        )
        inv[yr] = {
            "games_with_quotes": int(sy.nflverse_game_id.nunique()),
            "books": sorted(sy.bookmaker_key.unique().tolist()),
            "median_gap_s": float(gaps.median()),
        }
    mats = L.game_mats(s, books)
    g = pd.read_parquet(L.PERGAME)
    g = g[(g.game_type == "REG") & g.season.isin(SEASONS)]
    gf = pd.read_parquet(
        L.ROOT / "data/processed/game_features.parquet", columns=["game_id", "kickoff"]
    )
    g = g.merge(gf, on="game_id", how="left")
    sub = g[g.margin_vs_open != 0].copy().reset_index(drop=True)
    rows = []
    for gid, gd, ko in zip(sub.game_id, sub.gameday, sub.kickoff):  # noqa: B905
        f = None
        if gid in mats and pd.notna(ko):
            o, M, U = mats[gid]  # noqa: RUF059
            f = path_features(o, M, L.decision_ts(gd), deadline_ts(gd, ko))
        rows.append(f)
    ok = np.array([r is not None for r in rows])
    feat = pd.DataFrame([r for r in rows if r is not None])
    inv["multi_snapshot_paths"] = {
        str(yr): int(((sub.season.values == yr) & ok).sum()) for yr in SEASONS
    }
    inv["games_total"] = {str(yr): int((sub.season.values == yr).sum()) for yr in SEASONS}
    sub = sub[ok].reset_index(drop=True)
    for c in feat.columns:
        sub[c] = feat[c].values
    sub.to_parquet(OUT / "path_features.parquet")
    y = sub.home_covered.astype(float).values
    pr = sub.out_of_season_home_probability.clip(1e-6, 1 - 1e-6).values
    off = L.logit(pr)
    seasons = sub.season.values
    dirn = np.sign(sub.net.values)
    rng = np.random.default_rng(L.SEED)
    corr = {c: float(np.corrcoef(sub[c], np.abs(sub.net))[0, 1]) for c in FEATURES}
    out = {
        "inventory": inv,
        "games_scored": len(sub),
        "corr_with_abs_net_move": corr,
        "moved_games": int((dirn != 0).sum()),
        "look_count": len(FEATURES),
        "family": "mkt21_line_path_shape",
        "terms": {},
    }
    base = pr
    for c in FEATURES:
        x = sub[c].values * dirn
        pooled = np.zeros(len(sub))
        ins = np.zeros(len(sub))
        fc = {}
        for sn in SEASONS:
            te = seasons == sn
            b = L.fit_b(off[~te], x[~te], y[~te])
            fc[int(sn)] = float(b)
            pooled[te] = 1 / (1 + np.exp(-(off[te] + b * x[te])))
        b_all = L.fit_b(off, x, y)
        ins = 1 / (1 + np.exp(-(off + b_all * x)))
        t = {"fold_coefficients": fc, "pooled_coefficient": float(b_all)}
        for label, arr in (("oos", pooled), ("ins", ins)):
            d_b = (base - y) ** 2 - (arr - y) ** 2
            d_ll = -(y * np.log(base) + (1 - y) * np.log(1 - base)) + (
                y * np.log(arr) + (1 - y) * np.log(1 - arr)
            )
            d_a = (
                ((arr > 0.5) == (y == 1)).astype(float) - ((base > 0.5) == (y == 1)).astype(float)
            ) * 100
            dec = (base > 0.5) != (arr > 0.5)
            t[label] = {
                "brier_gain": L.boot(d_b, rng),
                "logloss_gain": L.boot(d_ll, rng),
                "accuracy_gain_points": L.boot(d_a, rng),
                "decisive_games": int(dec.sum()),
                "decisive_new_correct": int(((arr > 0.5) == (y == 1))[dec].sum()),
                "decisive_base_correct": int(((base > 0.5) == (y == 1))[dec].sum()),
            }
        t["gap_brier_gain_ins_minus_oos"] = t["ins"]["brier_gain"][0] - t["oos"]["brier_gain"][0]
        out["terms"][c] = t
    Path(OUT / "line_path.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
