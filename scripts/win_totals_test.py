from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = ROOT / "data" / "raw" / "20261116T204645Z" / "schedules.parquet"
WIN_TOTALS = ROOT / "data" / "external" / "win_totals" / "nfl_win_totals.parquet"
FUTURES = ROOT / "data" / "external" / "preseason_futures" / "preseason_close.parquet"
OUT = ROOT / "artifacts" / "win_totals_test"
FIRST_SEASON = 2009
CURRENT_CODE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def schedule():
    s = pd.read_parquet(SCHEDULES)
    s = s[(s.game_type == "REG") & (s.season >= FIRST_SEASON)].copy()
    for side in ("home_team", "away_team"):
        s[side] = s[side].replace(CURRENT_CODE)
    s["home_field"] = np.where(s.location.astype(str).str.lower() == "neutral", 0.0, 1.0)
    return s[["game_id", "season", "week", "home_team", "away_team", "spread_line", "home_field"]]


def priors():
    wt = pd.read_parquet(WIN_TOTALS)[["season", "team", "win_total"]]
    wt = wt.rename(columns={"win_total": "wt"})
    fu = pd.read_parquet(FUTURES)
    fu = fu[fu.league == "nfl"][["season", "team", "implied_probability"]]
    fu = fu.rename(columns={"implied_probability": "sb"})
    return wt.merge(fu, on=["season", "team"], how="outer")


def market_ratings(games, teams):
    idx = {t: i for i, t in enumerate(teams)}
    rows = games.dropna(subset=["spread_line"])
    if rows.empty:
        return None
    x = np.zeros((len(rows), len(teams) + 1))
    r = np.arange(len(rows))
    x[r, rows.home_team.map(idx).to_numpy()] = 1.0
    x[r, rows.away_team.map(idx).to_numpy()] = -1.0
    x[:, -1] = rows.home_field.to_numpy()
    sol, *_ = np.linalg.lstsq(x, rows.spread_line.to_numpy(dtype=float), rcond=None)
    played = set(rows.home_team) | set(rows.away_team)
    return pd.Series(sol[:-1], index=teams)[[t for t in teams if t in played]]


def pct(series):
    return series.rank(pct=True)


def build(sched, pri):
    out = []
    for season, g in sched.groupby("season"):
        teams = sorted(set(g.home_team) | set(g.away_team))
        p = pri[pri.season == season].set_index("team")
        missing = set(teams) - set(p.index.dropna())
        assert len(missing) <= 1, (season, missing)
        p = p.reindex(teams)
        for week, gw in g.groupby("week"):
            rec = gw[["game_id", "season", "week"]].copy()
            rec["wt_drift"] = 0.0
            rec["sb_drift"] = 0.0
            rating = market_ratings(g[g.week < week], teams)
            if rating is not None:
                mk = pct(rating)
                for col in ("wt", "sb"):
                    pr = pct(p.loc[rating.index, col])
                    team_term = pr - mk
                    h = gw.home_team.map(team_term)
                    a = gw.away_team.map(team_term)
                    rec[f"{col}_drift"] = (h - a).fillna(0.0).to_numpy()
            out.append(rec)
    return pd.concat(out, ignore_index=True)


def truncation_check(sched, pri, full):
    rng = np.random.default_rng(20261116)
    picks = full[full.week >= 2].sample(12, random_state=rng.integers(1 << 31))
    worst = 0.0
    for _, row in picks.iterrows():
        cut = sched[(sched.season == row.season) & (sched.week <= row.week)].copy()
        future = sched[(sched.season == row.season) & (sched.week > row.week)]
        assert future.game_id.isin(cut.game_id).sum() == 0
        cut.loc[cut.week == row.week, "spread_line"] = np.nan
        again = build(cut, pri).set_index("game_id")
        for col in ("wt_drift", "sb_drift"):
            worst = max(worst, abs(again.loc[row.game_id, col] - row[col]))
    assert worst < 1e-12, worst
    return {"games_checked": len(picks), "max_abs_difference": worst}


def main():
    ap = argparse.ArgumentParser(description="Preseason win-total and Super Bowl drift terms")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sched = schedule()
    pri = priors()
    full = build(sched, pri)
    check = truncation_check(sched, pri, full)
    full.to_parquet(out / "features.parquet", index=False)
    summary = {
        "games": len(full),
        "seasons": [int(full.season.min()), int(full.season.max())],
        "truncation_check": check,
        "nonzero": {c: int((full[c] != 0).sum()) for c in ("wt_drift", "sb_drift")},
        "sd_by_week": full[full.week <= 12]
        .groupby("week")[["wt_drift", "sb_drift"]]
        .std()
        .round(3)
        .to_dict(),
        "corr_wt_sb": float(full[full.week >= 2][["wt_drift", "sb_drift"]].corr().iloc[0, 1]),
    }
    (out / "features_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
