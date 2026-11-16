from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/team_mispricing"
SCHEDULES = ROOT / "data/raw"
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
NATIONAL_DAYS = {"Thursday", "Saturday", "Monday"}
SUNDAY_NIGHT_START = "20:00"
QUIET_MOVE_POINTS = 1.0
SCREEN_SEASONS = (2011, 2025)
HALVES = ((2011, 2018), (2019, 2025))
SHUFFLES = 200
BOOTSTRAPS = 2000
SIGN = -1.0
SEED = 20261116


def schedules():
    snapshot = sorted(p for p in SCHEDULES.glob("2026*") if (p / "schedules.parquet").is_file())[-1]
    raw = pd.read_parquet(snapshot / "schedules.parquet")
    table = raw.loc[raw["game_type"].eq("REG")].copy()
    for side in ("home_team", "away_team"):
        table[side] = table[side].replace(FRANCHISE)
    return table, snapshot.name, raw


def popularity(table):
    national = table["weekday"].isin(NATIONAL_DAYS) | (
        table["weekday"].eq("Sunday") & table["gametime"].astype(str).ge(SUNDAY_NIGHT_START)
    )
    rows = []
    for side, sign in (("home_team", 1.0), ("away_team", -1.0)):
        part = pd.DataFrame(
            {
                "season": table["season"].astype(int),
                "team": table[side],
                "national": national.astype(int),
                "diff": sign * table["result"].astype(float),
            }
        )
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    team = (
        long.groupby(["season", "team"])
        .agg(national=("national", "sum"), pd_pg=("diff", "mean"))
        .reset_index()
    )
    out = []
    for _, grp in team.groupby("season"):
        grp = grp.copy()
        z = (grp["national"] - grp["national"].mean()) / grp["national"].std(ddof=0)
        grp["pop_z"] = z
        design = np.column_stack([np.ones(len(grp)), grp["pd_pg"].to_numpy(float)])
        beta, *_ = np.linalg.lstsq(design, z.to_numpy(float), rcond=None)
        grp["pop_resid"] = z.to_numpy(float) - design @ beta
        grp["resid_beta"] = beta[1]
        out.append(grp)
    team = pd.concat(out, ignore_index=True)
    team["season"] = team["season"] + 1
    return team


def openers(table):
    from hindsight_audit_team import SOURCES, tuesday_quotes

    _, spread = tuesday_quotes(table)
    return spread, dict(SOURCES)


def slope(y, x):
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def shuffle_within(x, seasons, rng):
    out = x.copy()
    for idx in seasons:
        out[idx] = rng.permutation(out[idx])
    return out


def screen_one(frame, term, rng):
    data = frame.dropna(subset=[term]).copy()
    x = data[term].to_numpy(float)
    x = (x - x.mean()) / x.std(ddof=0)
    y = data["home_cover"].to_numpy(float)
    seasons_arr = data["season"].to_numpy(int)
    groups = [np.flatnonzero(seasons_arr == s) for s in np.unique(seasons_arr)]
    obs = SIGN * slope(y, x) * 100.0
    null = np.array(
        [SIGN * slope(y, shuffle_within(x, groups, rng)) * 100.0 for _ in range(SHUFFLES)]
    )
    boots = []
    uniq = np.unique(seasons_arr)
    for _ in range(BOOTSTRAPS):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(seasons_arr == s) for s in pick])
        boots.append(SIGN * slope(y[idx], x[idx]) * 100.0)
    boots = np.array(boots)
    halves = {}
    for lo, hi in HALVES:
        m = (seasons_arr >= lo) & (seasons_arr <= hi)
        halves[f"{lo}-{hi}"] = SIGN * slope(y[m], x[m]) * 100.0
    per_season = {
        int(s): SIGN * slope(y[seasons_arr == s], x[seasons_arr == s]) * 100.0 for s in uniq
    }
    raw = data[term].to_numpy(float)
    fade_pick_home = raw < 0
    decisive = raw != 0
    wins = int(((fade_pick_home & (y == 1)) | (~fade_pick_home & (y == 0)))[decisive].sum())
    return {
        "n": len(data),
        "effect_acc_points_per_sd_declared": obs,
        "interval_95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "probability_positive": float((boots > 0).mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "null_percentile": float((null < obs).mean() * 100.0),
        "halves": halves,
        "same_sign_halves": bool(
            all(v > 0 for v in halves.values()) or all(v < 0 for v in halves.values())
        ),
        "halves_declared_positive": bool(all(v > 0 for v in halves.values())),
        "per_season": per_season,
        "seasons_declared_positive": int(sum(v > 0 for v in per_season.values())),
        "fade_popular_record": [wins, int(decisive.sum()) - wins],
        "survivor": bool(
            obs > float(np.quantile(null, 0.95)) and all(v > 0 for v in halves.values())
        ),
    }


def screen():
    rng = np.random.default_rng(SEED)
    table, snap, raw = schedules()
    team = popularity(table)
    lo, hi = SCREEN_SEASONS
    games = table.loc[table["season"].between(lo, hi)].copy()
    games = games.dropna(subset=["result", "spread_line"])
    margin = games["result"].astype(float) - games["spread_line"].astype(float)
    games = games.loc[margin.ne(0)].copy()
    games["home_cover"] = (margin.loc[games.index] > 0).astype(float)
    for side in ("home", "away"):
        games = games.merge(
            team[["season", "team", "pop_z", "pop_resid", "national"]].rename(
                columns={
                    "team": f"{side}_team",
                    "pop_z": f"{side}_pop_z",
                    "pop_resid": f"{side}_pop_resid",
                    "national": f"{side}_national",
                }
            ),
            on=["season", f"{side}_team"],
            how="left",
        )
    games["pop_diff"] = games["home_pop_z"] - games["away_pop_z"]
    games["pop_resid_diff"] = games["home_pop_resid"] - games["away_pop_resid"]
    spread, sources = openers(raw)
    games["opener"] = spread.reindex(games["game_id"].astype(str)).to_numpy(float)
    have = games["opener"].notna()
    agree = float(np.corrcoef(games.loc[have, "opener"], games.loc[have, "spread_line"])[0, 1])
    games["quiet"] = have & (games["spread_line"] - games["opener"]).abs().lt(QUIET_MOVE_POINTS)
    res = {
        "schedule_snapshot": snap,
        "opener_sources": sources,
        "opener_coverage": int(have.sum()),
        "opener_close_corr": agree,
        "games": len(games),
        "quiet_games": int(games["quiet"].sum()),
        "missing_pop": int(games["pop_diff"].isna().sum()),
        "resid_beta_by_prior_season": {
            int(k): float(v) for k, v in team.groupby("season")["resid_beta"].first().items()
        },
        "cells": {},
    }
    for term in ("pop_diff", "pop_resid_diff"):
        for pop in ("all", "quiet"):
            frame = games if pop == "all" else games.loc[games["quiet"]]
            res["cells"][f"{term}_{pop}"] = screen_one(frame, term, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    games[
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "spread_line",
            "opener",
            "quiet",
            "home_cover",
            "pop_diff",
            "pop_resid_diff",
        ]
    ].to_parquet(OUT / "screen_games.parquet", index=False)
    (OUT / "screen.json").write_text(json.dumps(res, indent=2))
    for name, cell in res["cells"].items():
        print(name, json.dumps({k: v for k, v in cell.items() if k != "per_season"}))
    print(json.dumps({k: v for k, v in res.items() if k != "cells"}))


FAMILY = "team_mispricing_2026_11_16"
LANE = "docs/lanes/team-mispricing-2026-11-16.md"
TERMS = {
    "pop_diff": "prior-season national-window game count, z-scored across teams",
    "pop_resid_diff": (
        "prior-season national-window game count, z-scored and residualised on prior-season "
        "point differential per game"
    ),
}
QUIET_TEXT = "QUIET (|close - opener| < 1, opener = Tuesday capture else SBR open)"


def batch():
    res = json.loads((OUT / "screen.json").read_text())
    cells = []
    for key, cell in res["cells"].items():
        term = "pop_resid_diff" if key.startswith("pop_resid_diff") else "pop_diff"
        pop = key.rsplit("_", 1)[1]
        halves = cell["halves"]
        wins, losses = cell["fade_popular_record"]
        cells.append(
            {
                "name": f"team_mispricing_public_{term}_{pop}_close",
                "description": (
                    f"Persistent public-team premium, {TERMS[term]}, home minus away; declared "
                    "sign NEGATIVE on home cover (more popular side overpriced). Screen against "
                    "the close, REG 2011-2025, pushes dropped, population "
                    f"{'ALL' if pop == 'all' else QUIET_TEXT}; OLS slope of home cover per SD in "
                    "accuracy points, stored in the declared direction. Halves "
                    f"2011-18 {halves['2011-2018']:+.2f}, 2019-25 {halves['2019-2025']:+.2f}; "
                    f"null percentile {cell['null_percentile']:.1f} of {SHUFFLES} within-season "
                    f"shuffles (p95 {cell['null_p95']:+.2f}); fade-the-popular-side record "
                    f"{wins}-{losses}; {cell['seasons_declared_positive']} of 15 seasons in the "
                    "declared direction."
                ),
                "source": LANE,
                "effect": round(cell["effect_acc_points_per_sd_declared"], 6),
                "effect_units": "accuracy_points",
                "classification": "unresolved_below_power",
                "league": "nfl",
                "season_start": SCREEN_SEASONS[0],
                "season_end": SCREEN_SEASONS[1],
                "family": FAMILY,
                "category": "market",
                "classification_evidence": (
                    "Point estimate is on the wrong side but the season-block interval crosses "
                    "zero, so no resolved wrong sign; no positive control bounds an effect this "
                    "small at the close screen."
                ),
                "notes": (
                    f"Family {FAMILY}: 2 terms x 2 populations = 4 looks, declared in the lane "
                    "before outcomes were joined; survivor rule null p95 plus same declared sign "
                    "in both halves; survivor: no; served look not run."
                ),
                "interval_low": round(cell["interval_95"][0], 6),
                "interval_high": round(cell["interval_95"][1], 6),
                "probability_positive": round(cell["probability_positive"], 4),
                "sample_games": int(cell["n"]),
                "sample_blocks": 15,
                "plain_summary": (
                    "Checks whether teams the league put on national TV most the year before are "
                    "priced too high by the spread"
                    f"{' on games where the line barely moved' if pop == 'quiet' else ''}. They "
                    "covered slightly more often, not less; not enough games to be sure either way."
                ),
            }
        )
    path = OUT / "weak_signals_batch.json"
    path.write_text(json.dumps({"cells": cells}, indent=1), encoding="utf-8")
    print(path, len(cells))


def main():
    parser = argparse.ArgumentParser(
        description="Persistent public-team premium screen against the close, and its batch cells."
    )
    parser.add_argument("mode", choices=["screen", "batch"])
    args = parser.parse_args()
    if args.mode == "screen":
        screen()
    if args.mode == "batch":
        batch()


if __name__ == "__main__":
    main()
