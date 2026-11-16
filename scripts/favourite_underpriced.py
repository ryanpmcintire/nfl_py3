from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/favourite_underpriced"
SCHEDULES = ROOT / "data/raw"
WIN_TOTALS = ROOT / "data/external/win_totals/nfl_win_totals.parquet"
FUTURES = ROOT / "data/external/preseason_futures/preseason_close.parquet"
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
SCREEN_SEASONS = (2011, 2025)
FIRST_WEEK = 3
LAST_WEEK = 18
HALVES = ((2011, 2018), (2019, 2025))
SHUFFLES = 200
BOOTSTRAPS = 2000
SIGN = 1.0
SEED = 20261116
TRUNCATION_GAMES = 12


def schedules():
    snapshot = sorted(p for p in SCHEDULES.glob("2026*") if (p / "schedules.parquet").is_file())[-1]
    raw = pd.read_parquet(snapshot / "schedules.parquet")
    table = raw.loc[raw["game_type"].eq("REG")].copy()
    for side in ("home_team", "away_team"):
        table[side] = table[side].replace(FRANCHISE)
    table["game_id"] = table["game_id"].astype(str)
    table["home_field"] = np.where(table["location"].astype(str).str.lower().eq("neutral"), 0.0, 1.0)
    return table, snapshot.name, raw


def priors():
    wt = pd.read_parquet(WIN_TOTALS)[["season", "team", "win_total"]]
    fu = pd.read_parquet(FUTURES)
    fu = fu.loc[fu["league"].eq("nfl"), ["season", "team", "implied_probability"]]
    both = wt.merge(fu, on=["season", "team"], how="outer")
    both["team"] = both["team"].replace(FRANCHISE)
    return both


def openers(raw):
    from hindsight_audit_team import SOURCES, tuesday_quotes

    _, spread = tuesday_quotes(raw)
    spread.index = spread.index.astype(str)
    return spread, dict(SOURCES)


def ratings(games, teams):
    rows = games.dropna(subset=["opener"])
    if rows.empty:
        return None
    idx = {t: i for i, t in enumerate(teams)}
    x = np.zeros((len(rows), len(teams) + 1))
    r = np.arange(len(rows))
    x[r, rows["home_team"].map(idx).to_numpy()] = 1.0
    x[r, rows["away_team"].map(idx).to_numpy()] = -1.0
    x[:, -1] = rows["home_field"].to_numpy(float)
    sol, *_ = np.linalg.lstsq(x, rows["opener"].to_numpy(float), rcond=None)
    played = set(rows["home_team"]) | set(rows["away_team"])
    return pd.Series(sol[:-1], index=teams)[[t for t in teams if t in played]]


def pct(series):
    return series.rank(pct=True)


def build(sched, pri):
    out = []
    for season, g in sched.groupby("season"):
        teams = sorted(set(g["home_team"]) | set(g["away_team"]))
        p = pri.loc[pri["season"].eq(season)].drop_duplicates("team").set_index("team").reindex(teams)
        for week, gw in g.groupby("week"):
            rec = gw[["game_id", "season", "week"]].copy()
            rec["pre_gap"] = 0.0
            rec["rated"] = False
            rating = ratings(g.loc[g["week"] < week], teams)
            if rating is not None and len(rating) > 1:
                sub = p.loc[rating.index]
                pre = pd.concat([pct(sub["win_total"]), pct(sub["implied_probability"])], axis=1)
                pre_pct = pct(pre.mean(axis=1, skipna=True))
                team_term = pre_pct - pct(rating)
                h = gw["home_team"].map(team_term)
                a = gw["away_team"].map(team_term)
                rec["rated"] = (h.notna() & a.notna()).to_numpy()
                rec["pre_gap"] = (h - a).fillna(0.0).to_numpy()
            out.append(rec)
    return pd.concat(out, ignore_index=True)


def truncation_check(sched, pri, full):
    rng = np.random.default_rng(SEED)
    pool = full.loc[full["week"].ge(FIRST_WEEK)]
    picks = pool.sample(TRUNCATION_GAMES, random_state=int(rng.integers(1 << 31)))
    worst = 0.0
    for _, row in picks.iterrows():
        cut = sched.loc[sched["season"].eq(row["season"]) & sched["week"].le(row["week"])].copy()
        cut.loc[cut["week"].eq(row["week"]), "opener"] = np.nan
        again = build(cut, pri).set_index("game_id")
        worst = max(worst, abs(float(again.loc[row["game_id"], "pre_gap"]) - float(row["pre_gap"])))
    assert worst < 1e-12, worst
    return {"games_checked": len(picks), "max_abs_difference": worst}


def slope(y, x):
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def shuffle_within(x, groups, rng):
    out = x.copy()
    for idx in groups:
        out[idx] = rng.permutation(out[idx])
    return out


def screen_cell(data, rng):
    x = data["pre_gap"].to_numpy(float)
    sd = float(x.std(ddof=0))
    x = (x - x.mean()) / sd
    y = data["home_cover"].to_numpy(float)
    seasons_arr = data["season"].to_numpy(int)
    uniq = np.unique(seasons_arr)
    groups = [np.flatnonzero(seasons_arr == s) for s in uniq]
    obs = SIGN * slope(y, x) * 100.0
    null = np.array([SIGN * slope(y, shuffle_within(x, groups, rng)) * 100.0 for _ in range(SHUFFLES)])
    boots = []
    for _ in range(BOOTSTRAPS):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[int(np.searchsorted(uniq, s))] for s in pick])
        boots.append(SIGN * slope(y[idx], x[idx]) * 100.0)
    boots = np.array(boots)
    halves = {}
    for lo, hi in HALVES:
        m = (seasons_arr >= lo) & (seasons_arr <= hi)
        halves[f"{lo}-{hi}"] = SIGN * slope(y[m], x[m]) * 100.0
    per_season = {int(s): SIGN * slope(y[seasons_arr == s], x[seasons_arr == s]) * 100.0 for s in uniq}
    raw = data["pre_gap"].to_numpy(float)
    decisive = raw != 0
    back_home = raw > 0
    wins = int(((back_home & (y == 1)) | (~back_home & (y == 0)))[decisive].sum())
    move = (data["spread_line"] - data["opener"]).to_numpy(float)
    have = ~np.isnan(move)
    null_p95 = float(np.quantile(null, 0.95))
    return {
        "n": int(len(data)),
        "term_sd_rank_units": sd,
        "effect_acc_points_per_sd_declared": obs,
        "interval_95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "probability_positive": float((boots > 0).mean()),
        "null_p95": null_p95,
        "null_p05": float(np.quantile(null, 0.05)),
        "null_percentile": float((null < obs).mean() * 100.0),
        "halves": halves,
        "halves_declared_positive": bool(all(v > 0 for v in halves.values())),
        "per_season": per_season,
        "seasons_declared_positive": int(sum(v > 0 for v in per_season.values())),
        "back_preseason_side_record": [wins, int(decisive.sum()) - wins],
        "corr_term_with_week_move_close_minus_opener": float(np.corrcoef(raw[have], move[have])[0, 1]),
        "survivor": bool(obs > null_p95 and all(v > 0 for v in halves.values())),
    }


def screen():
    rng = np.random.default_rng(SEED)
    table, snap, raw = schedules()
    spread, sources = openers(raw)
    table["opener"] = spread.reindex(table["game_id"]).to_numpy(float)
    pri = priors()
    sched = table[["game_id", "season", "week", "home_team", "away_team", "home_field", "opener"]]
    sched = sched.loc[sched["season"].between(SCREEN_SEASONS[0], SCREEN_SEASONS[1] + 1)]
    feats = build(sched, pri)
    trunc = truncation_check(sched, pri, feats)
    OUT.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT / "features.parquet", index=False)
    lo, hi = SCREEN_SEASONS
    games = table.loc[
        table["season"].between(lo, hi) & table["week"].between(FIRST_WEEK, LAST_WEEK)
    ].dropna(subset=["result", "spread_line"])
    margin = games["result"].astype(float) - games["spread_line"].astype(float)
    pushes = int(margin.eq(0).sum())
    games = games.loc[margin.ne(0)].copy()
    games["home_cover"] = (margin.loc[games.index] > 0).astype(float)
    games = games.merge(feats[["game_id", "pre_gap", "rated"]], on="game_id", how="left")
    res = {
        "schedule_snapshot": snap,
        "opener_sources": sources,
        "opener_coverage_screen": int(games["opener"].notna().sum()),
        "games": int(len(games)),
        "pushes_dropped": pushes,
        "rated_games": int(games["rated"].fillna(False).sum()),
        "nonzero_term": int(games["pre_gap"].fillna(0).ne(0).sum()),
        "missing_term": int(games["pre_gap"].isna().sum()),
        "truncation_check": trunc,
        "cell": screen_cell(games.dropna(subset=["pre_gap"]), rng),
    }
    games[
        ["game_id", "season", "week", "home_team", "away_team", "spread_line", "opener", "home_cover", "pre_gap"]
    ].to_parquet(OUT / "screen_games.parquet", index=False)
    (OUT / "screen.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "cell"}))
    print(json.dumps({k: v for k, v in res["cell"].items() if k != "per_season"}))
    print(json.dumps(res["cell"]["per_season"]))


FAMILY = "favourite_underpriced_2026_11_16"
LANE = "docs/lanes/favourite-underpriced-2026-11-16.md"


def batch(classification, evidence, closing_ground):
    res = json.loads((OUT / "screen.json").read_text(encoding="utf-8"))
    cell = res["cell"]
    halves = cell["halves"]
    wins, losses = cell["back_preseason_side_record"]
    entry = {
        "name": "favourite_underpriced_pre_gap_close_screen",
        "description": (
            "Preseason favourite underpriced: percentile rank of preseason strength (mean of win "
            "total and Super Bowl futures ranks) minus percentile rank of opener-implied rating "
            "from earlier weeks, home minus away; declared sign POSITIVE on home cover (back the "
            "preseason-favoured side). Screen against the close, REG 2011-2025 weeks 3-18, pushes "
            "dropped; OLS slope of home cover per SD in accuracy points, declared direction. Halves "
            f"2011-18 {halves['2011-2018']:+.2f}, 2019-25 {halves['2019-2025']:+.2f}; null "
            f"percentile {cell['null_percentile']:.1f} of {SHUFFLES} within-season shuffles (p95 "
            f"{cell['null_p95']:+.2f}); back-preseason-side record {wins}-{losses}; "
            f"{cell['seasons_declared_positive']} of 15 seasons in the declared direction."
        ),
        "source": LANE,
        "effect": round(cell["effect_acc_points_per_sd_declared"], 6),
        "effect_units": "accuracy_points",
        "classification": classification,
        "league": "nfl",
        "season_start": SCREEN_SEASONS[0],
        "season_end": SCREEN_SEASONS[1],
        "family": FAMILY,
        "category": "market",
        "classification_evidence": evidence,
        "notes": (
            f"Family {FAMILY}: confirmation look for a hypothesis suggested by futures_anchor "
            "(college, positive 13/13 folds), team_mispricing (popular teams lean positive) and "
            "win_totals (weeks 7-12 anchor, fitted negative at the opener). Declared in the lane "
            "before outcomes were joined; at most 2 gating reads (screen, served look for a survivor). "
            f"Survivor: {'yes' if cell['survivor'] else 'no'}."
        ),
        "interval_low": round(cell["interval_95"][0], 6),
        "interval_high": round(cell["interval_95"][1], 6),
        "probability_positive": round(cell["probability_positive"], 4),
        "sample_games": int(cell["n"]),
        "sample_blocks": 15,
        "plain_summary": (
            "Checks whether teams the betting market rated highly before the season are priced too "
            "low by the spread later in the year, once the weekly lines have moved away from them."
        ),
    }
    if closing_ground:
        entry["closing_ground"] = closing_ground
    path = OUT / "weak_signals_batch.json"
    path.write_text(json.dumps({"cells": [entry]}, indent=1), encoding="utf-8")
    print(path, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Preseason-favourite-underpriced screen against the close, and its batch cell."
    )
    parser.add_argument("mode", choices=["screen", "batch"])
    parser.add_argument("--classification", default="unresolved_below_power")
    parser.add_argument("--evidence", default="")
    parser.add_argument("--closing-ground", default="")
    args = parser.parse_args()
    if args.mode == "screen":
        screen()
    if args.mode == "batch":
        batch(args.classification, args.evidence, args.closing_ground)


if __name__ == "__main__":
    main()
