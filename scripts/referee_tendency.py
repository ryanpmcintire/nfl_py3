import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = sorted((ROOT / "data" / "raw").glob("2026*/schedules.parquet"))[-1]
PBP_ROOT = ROOT / "data" / "pbp" / "raw"
OUT = ROOT / "artifacts" / "referee_tendency"
HISTORY_START = 2009
SCREEN_START = 2011
LAST_SEASON = 2025
HALVES = ((2011, 2017), (2018, 2025))
DECLARED_SHUFFLES = 200
SHUFFLES = int(sys.argv[1]) if len(sys.argv) > 1 else DECLARED_SHUFFLES
BOOT = 2000
SEED = 20261116
EXPECTED_SIGN = 1.0
BAR = 95.0
REG = "REG"
PLAY_COLUMNS = ["game_id", "season_type", "home_team", "away_team", "penalty", "penalty_team", "penalty_yards"]


def latest_plays():
    latest = {}
    for snap in sorted(p for p in PBP_ROOT.iterdir() if p.is_dir()):
        for f in snap.glob("season=*/plays.parquet"):
            latest[int(f.parent.name.split("=")[1])] = f
    return latest


def penalty_differential():
    paths = latest_plays()
    rows = []
    for season in range(HISTORY_START, LAST_SEASON + 1):
        f = pd.read_parquet(paths[season], columns=PLAY_COLUMNS)
        f = f[f["season_type"].eq(REG)]
        games = f.groupby("game_id")[["home_team", "away_team"]].first()
        p = f[f["penalty"].eq(1) & f["penalty_team"].notna()].copy()
        p["yards"] = p["penalty_yards"].fillna(0.0).astype(float)
        p["home_team"] = p["game_id"].map(games["home_team"])
        p["on_home"] = np.where(p["penalty_team"].eq(p["home_team"]), p["yards"], 0.0)
        p["on_away"] = np.where(p["penalty_team"].eq(p["home_team"]), 0.0, p["yards"])
        g = p.groupby("game_id")[["on_home", "on_away"]].sum().reindex(games.index, fill_value=0.0)
        g["d"] = g["on_away"] - g["on_home"]
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True)


def load():
    s = pd.read_parquet(SCHEDULES)
    s = s[s["game_type"].eq(REG) & s["season"].between(HISTORY_START, LAST_SEASON)]
    s = s[["game_id", "season", "week", "referee", "result", "spread_line"]].dropna(subset=["referee"])
    d = penalty_differential()
    m = s.merge(d, on="game_id", how="inner", validate="one_to_one")
    return m.sort_values(["season", "week", "game_id"]).reset_index(drop=True), len(s), len(d)


def shrinkage(d, ref, season):
    k = {}
    for held in range(SCREEN_START, LAST_SEASON + 1):
        keep = season != held
        g = pd.DataFrame({"r": ref[keep], "d": d[keep]}).groupby("r")["d"]
        n = g.size().to_numpy(float)
        mean = g.mean().to_numpy(float)
        var = g.var(ddof=1).to_numpy(float)
        ok = n > 1
        sigma2 = float(np.sum((n[ok] - 1) * var[ok]) / np.sum(n[ok] - 1))
        tau2 = float(np.var(mean, ddof=1) - np.mean(sigma2 / n))
        k[held] = sigma2 / tau2 if tau2 > 0 else np.inf
    return k


def traits(frame, ref):
    d = frame["d"].to_numpy(float)
    season = frame["season"].to_numpy(int)
    week = frame["week"].to_numpy(int)
    key = season * 100 + week
    wk = pd.DataFrame({"key": key, "d": d}).groupby("key")["d"].agg(["sum", "size"])
    prior_sum = wk["sum"].cumsum().shift(fill_value=0.0)
    prior_n = wk["size"].cumsum().shift(fill_value=0)
    mu = (prior_sum / prior_n.replace(0, np.nan)).fillna(0.0)
    mu_g = pd.Series(key).map(mu).to_numpy(float)
    t = pd.DataFrame({"r": ref, "key": key, "d": d, "i": np.arange(len(d))}).sort_values(["r", "key"])
    grp = t.groupby("r")["d"]
    t["s"] = grp.cumsum() - t["d"]
    t["n"] = grp.cumcount()
    t = t.sort_values("i")
    s_r = t["s"].to_numpy(float)
    n_r = t["n"].to_numpy(float)
    k = shrinkage(d, ref, season)
    kg = np.array([k.get(int(x), np.inf) for x in season])
    out = np.where(np.isfinite(kg), (s_r - n_r * mu_g) / (n_r + np.where(np.isfinite(kg), kg, 0.0)), 0.0)
    return out, k


def slope(x, y):
    xc = x - x.mean()
    den = float(xc @ xc)
    return float(xc @ (y - y.mean()) / den) if den > 0 else 0.0


def per_sd(x, y):
    sd = float(x.std(ddof=1))
    return slope(x / sd, y) if sd > 0 else 0.0


def shuffled_refs(frame, rng):
    ref = frame["referee"].to_numpy(object).copy()
    season = frame["season"].to_numpy(int)
    for s in np.unique(season):
        idx = np.flatnonzero(season == s)
        ref[idx] = ref[rng.permutation(idx)]
    return ref


def boot(x, y, season, rng):
    seasons = np.unique(season)
    groups = [np.flatnonzero(season == s) for s in seasons]
    out = np.empty(BOOT)
    for b in range(BOOT):
        pick = rng.choice(len(groups), len(groups), replace=True)
        idx = np.concatenate([groups[i] for i in pick])
        out[b] = per_sd(x[idx], y[idx])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, n_sched, n_pbp = load()
    trait, k = traits(frame, frame["referee"].to_numpy(object))
    frame["trait"] = trait
    frame["ats"] = frame["result"] - frame["spread_line"]
    scr = frame[frame["season"].between(SCREEN_START, LAST_SEASON)].reset_index(drop=True)
    x = scr["trait"].to_numpy(float)
    y = scr["ats"].to_numpy(float)
    dg = scr["d"].to_numpy(float)
    season = scr["season"].to_numpy(int)
    real = per_sd(x, y)
    first_stage = per_sd(x, dg)
    rng = np.random.default_rng(SEED)
    draws = boot(x, y, season, rng)
    halves = {}
    for a, b in HALVES:
        m = (season >= a) & (season <= b)
        hd = boot(x[m], y[m], season[m], rng)
        halves[f"{a}-{b}"] = {
            "slope_per_sd": per_sd(x[m], y[m]),
            "interval": [float(v) for v in np.percentile(hd, [2.5, 97.5])],
            "probability_positive": float((hd > 0).mean()),
            "games": int(m.sum()),
        }
    folds = {int(s): per_sd(x[season == s], y[season == s]) for s in np.unique(season)}
    null_rng = np.random.default_rng([SEED, 1])
    mask = frame["season"].between(SCREEN_START, LAST_SEASON).to_numpy()
    null, null_first = [], []
    for _ in range(SHUFFLES):
        t, _k = traits(frame, shuffled_refs(frame, null_rng))
        null.append(per_sd(t[mask], y))
        null_first.append(per_sd(t[mask], dg))
    null = np.array(null)
    null_first = np.array(null_first)
    pct = float((null < real).mean() * 100)
    pct_first = float((null_first < first_stage).mean() * 100)
    same_sign = all(np.sign(h["slope_per_sd"]) == EXPECTED_SIGN for h in halves.values())
    survivor = bool(pct >= BAR and same_sign)
    res = {
        "declared": "2026-11-16T21:41:32Z",
        "schedules": str(SCHEDULES.relative_to(ROOT)),
        "schedule_games_with_referee": n_sched,
        "pbp_games": n_pbp,
        "joined_games": len(frame),
        "screen_games": len(scr),
        "screen_pushes": int((y == 0).sum()),
        "referees": int(frame["referee"].nunique()),
        "shrinkage_k_by_held_out_season": {str(s): (None if not np.isfinite(v) else float(v)) for s, v in k.items()},
        "trait_sd_yards": float(x.std(ddof=1)),
        "trait_nonzero_games": int((x != 0).sum()),
        "first_stage_d_per_sd": first_stage,
        "first_stage_null_percentile": pct_first,
        "first_stage_null_q95": float(np.percentile(null_first, 95)),
        "slope_ats_points_per_sd": real,
        "interval": [float(v) for v in np.percentile(draws, [2.5, 97.5])],
        "probability_positive": float((draws > 0).mean()),
        "null_mean": float(null.mean()),
        "null_q05_q95": [float(np.percentile(null, 5)), float(np.percentile(null, 95))],
        "null_percentile": pct,
        "shuffles": SHUFFLES,
        "halves": halves,
        "season_slopes": folds,
        "seasons_expected_sign": int(sum(v * EXPECTED_SIGN > 0 for v in folds.values())),
        "survivor": survivor,
        "wrong_sign_resolved": bool(np.percentile(draws, 97.5) < 0),
    }
    (OUT / "screen.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    scr[["game_id", "season", "week", "referee", "d", "trait", "ats"]].to_parquet(OUT / "screen_games.parquet", index=False)
    print(json.dumps(res, indent=2))


main()
