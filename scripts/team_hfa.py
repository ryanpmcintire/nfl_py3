from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/team_hfa"
SCHEDULES = ROOT / "data/raw"
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
SCREEN_SEASONS = (2011, 2025)
HALVES = ((2011, 2018), (2019, 2025))
SHUFFLES = 200
BOOTSTRAPS = 2000
SIGN = 1.0
SEED = 20261116
TRUNCATION_TEAM_SEASONS = 12
DOME_ROOFS = ("dome", "closed")
FAMILY = "team_hfa_2026_11_16"
LANE = "docs/lanes/team-hfa-2026-11-16.md"


def schedules():
    snapshot = sorted(p for p in SCHEDULES.glob("2026*") if (p / "schedules.parquet").is_file())[-1]
    raw = pd.read_parquet(snapshot / "schedules.parquet")
    table = raw.loc[raw["game_type"].eq("REG")].copy()
    for side in ("home_team", "away_team"):
        table[side] = table[side].replace(FRANCHISE)
    table["game_id"] = table["game_id"].astype(str)
    return table, snapshot.name, raw


def primary_venues(table):
    home = table.loc[~table["location"].astype(str).str.lower().eq("neutral")]
    counts = home.groupby(["season", "home_team", "stadium_id"]).size().rename("n").reset_index()
    counts = counts.sort_values(
        ["season", "home_team", "n", "stadium_id"], ascending=[True, True, False, True]
    )
    return counts.drop_duplicates(["season", "home_team"]).rename(
        columns={"home_team": "team", "stadium_id": "venue"}
    )[["season", "team", "venue"]]


def annotate(table):
    prim = primary_venues(table)
    key = dict(zip(zip(prim["season"], prim["team"], strict=False), prim["venue"], strict=False))
    pairs = zip(table["season"], table["home_team"], strict=True)
    venue = np.array([key.get(k) for k in pairs], dtype=object)
    out = table.copy()
    out["primary"] = (out["stadium_id"].to_numpy() == venue) & ~out["location"].astype(
        str
    ).str.lower().eq("neutral")
    out["venue"] = np.where(out["primary"], venue, None)
    res = out["result"].astype(float) - out["spread_line"].astype(float)
    played = out["primary"] & res.notna()
    means = res.where(played).groupby(out["season"]).transform("mean")
    out["r"] = (res - means).where(played)
    return out, prim


def team_seasons(table):
    games, prim = annotate(table)
    hist = games.dropna(subset=["r"])
    rows = []
    for season in sorted(prim["season"].unique()):
        current = prim.loc[prim["season"].eq(season)]
        for team, venue in zip(current["team"], current["venue"], strict=False):
            span = prim.loc[
                prim["team"].eq(team) & prim["venue"].eq(venue) & prim["season"].lt(season),
                "season",
            ]
            h = hist.loc[
                hist["home_team"].eq(team) & hist["venue"].eq(venue) & hist["season"].isin(span),
                "r",
            ]
            road = -hist.loc[hist["away_team"].eq(team) & hist["season"].isin(span), "r"]
            nh, nr = len(h), len(road)
            ok = nh > 0 and nr > 0
            rows.append(
                {
                    "season": int(season),
                    "team": team,
                    "venue": venue,
                    "n_home": nh,
                    "n_road": nr,
                    "d": float(h.mean() - road.mean()) if ok else np.nan,
                    "n_eff": 1.0 / (1.0 / nh + 1.0 / nr) if ok else np.nan,
                }
            )
    return pd.DataFrame(rows), games


def choose_k(train):
    n = train["n_eff"].to_numpy(float)
    d = train["d"].to_numpy(float)
    r = train["r"].to_numpy(float)
    unit = float(n.mean())

    def loss(k):
        lam = unit * k / (1.0 - k)
        return float(((r - n / (n + lam) * d) ** 2).sum())

    fit = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded")
    k = float(fit.x)
    return k, unit * k / (1.0 - k), loss(k), float((r**2).sum())


def term_values(ts, games, seasons):
    lo, hi = seasons
    home = games.loc[games["primary"] & games["r"].notna() & games["season"].between(lo, hi)]
    home = home.merge(
        ts,
        left_on=["season", "home_team", "venue"],
        right_on=["season", "team", "venue"],
        how="left",
    )
    usable = home.dropna(subset=["d"])
    ts = ts.copy()
    ts["w"] = np.nan
    ts["x"] = 0.0
    picks = {}
    for season in range(lo, hi + 1):
        k, lam, sse, sse0 = choose_k(usable.loc[usable["season"].ne(season)])
        picks[season] = {"k": k, "lambda_games": lam, "inner_sse": sse, "inner_sse_no_term": sse0}
        m = ts["season"].eq(season) & ts["d"].notna()
        ts.loc[m, "w"] = ts.loc[m, "n_eff"] / (ts.loc[m, "n_eff"] + lam)
        ts.loc[m, "x"] = ts.loc[m, "w"] * ts.loc[m, "d"]
    return ts, picks


def truncation_check(table, ts):
    rng = np.random.default_rng(SEED)
    pool = ts.loc[ts["season"].between(*SCREEN_SEASONS) & ts["d"].notna()]
    picks = pool.sample(TRUNCATION_TEAM_SEASONS, random_state=int(rng.integers(1 << 31)))
    worst = 0.0
    for _, row in picks.iterrows():
        cut = table.loc[table["season"].le(row["season"])].copy()
        cut.loc[cut["season"].eq(row["season"]), ["result", "home_score", "away_score"]] = np.nan
        again, _ = team_seasons(cut)
        hit = again.loc[again["season"].eq(row["season"]) & again["team"].eq(row["team"])].iloc[0]
        worst = max(
            worst,
            abs(float(hit["d"]) - float(row["d"])),
            abs(float(hit["n_eff"]) - float(row["n_eff"])),
        )
    assert worst < 1e-12, worst
    return {"team_seasons_checked": len(picks), "max_abs_difference": worst}


def slope(y, x):
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def standardise(x):
    return (x - x.mean()) / x.std(ddof=0)


def label_shuffle(team_idx, primary, values_by_season, season_groups, rng):
    out = np.zeros(len(team_idx))
    for rows, vals in zip(season_groups, values_by_season, strict=False):
        perm = rng.permutation(vals)
        out[rows] = perm[team_idx[rows]]
    return np.where(primary, out, 0.0)


def screen_cell(data, ts, rng):
    y = data["home_cover"].to_numpy(float)
    raw = data["x"].to_numpy(float)
    x = standardise(raw)
    seasons_arr = data["season"].to_numpy(int)
    uniq = np.unique(seasons_arr)
    groups = [np.flatnonzero(seasons_arr == s) for s in uniq]
    primary = data["primary"].to_numpy(bool)
    team_idx = np.zeros(len(data), dtype=int)
    values = []
    for s, rows in zip(uniq, groups, strict=False):
        tab = ts.loc[ts["season"].eq(s)].set_index("team")["x"]
        teams = list(tab.index)
        pos = {t: i for i, t in enumerate(teams)}
        team_idx[rows] = data["home_team"].iloc[rows].map(pos).fillna(0).to_numpy(int)
        values.append(tab.to_numpy(float))
    obs = SIGN * slope(y, x) * 100.0
    null = np.array(
        [
            SIGN
            * slope(y, standardise(label_shuffle(team_idx, primary, values, groups, rng)))
            * 100.0
            for _ in range(SHUFFLES)
        ]
    )
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
    per_season = {
        int(s): SIGN * slope(y[seasons_arr == s], x[seasons_arr == s]) * 100.0 for s in uniq
    }
    decisive = raw != 0
    back_home = raw > 0
    wins = int(((back_home & (y == 1)) | (~back_home & (y == 0)))[decisive].sum())
    move = (data["spread_line"] - data["opener"]).to_numpy(float)
    have = ~np.isnan(move)
    null_p95 = float(np.quantile(null, 0.95))
    return {
        "n": len(data),
        "term_sd_points": float(raw.std(ddof=0)),
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
        "follow_edge_record_nonzero": [wins, int(decisive.sum()) - wins],
        "corr_term_with_week_move_close_minus_opener": float(
            np.corrcoef(raw[have], move[have])[0, 1]
        ),
        "survivor": bool(obs > null_p95 and all(v > 0 for v in halves.values())),
    }


def descriptives(ts, games):
    lo, hi = SCREEN_SEASONS
    cur = ts.loc[ts["season"].between(lo, hi) & ts["d"].notna()].copy()
    roof = (
        games.loc[games["primary"]]
        .groupby(["season", "home_team"])["roof"]
        .agg(lambda s: bool(s.astype(str).str.lower().isin(DOME_ROOFS).mode().iloc[0]))
    )
    keys = zip(cur["season"], cur["team"], strict=True)
    cur["dome"] = [bool(roof.get(k, False)) for k in keys]
    cur["den"] = cur["team"].eq("DEN")
    last = ts.loc[ts["season"].eq(hi)].sort_values("x")
    ranked = zip(last["team"], last["x"], strict=True)
    return {
        "team_seasons_with_history": len(cur),
        "corr_x_dome": float(np.corrcoef(cur["x"], cur["dome"].astype(float))[0, 1]),
        "mean_x_dome": float(cur.loc[cur["dome"], "x"].mean()),
        "mean_x_open": float(cur.loc[~cur["dome"], "x"].mean()),
        "den_x_by_season": {
            int(s): round(float(v), 3)
            for s, v in zip(cur.loc[cur["den"], "season"], cur.loc[cur["den"], "x"], strict=False)
        },
        "weight_mean": float(cur["w"].mean()),
        "raw_d_sd": float(cur["d"].std(ddof=0)),
        f"x_{hi}_by_team": {t: round(float(v), 3) for t, v in ranked},
    }


def openers(raw):
    from hindsight_audit_team import SOURCES, tuesday_quotes

    _, spread = tuesday_quotes(raw)
    spread.index = spread.index.astype(str)
    return spread, dict(SOURCES)


def screen():
    rng = np.random.default_rng(SEED)
    table, snap, raw = schedules()
    ts, games = team_seasons(table)
    trunc = truncation_check(table, ts)
    ts, picks = term_values(ts, games, SCREEN_SEASONS)
    OUT.mkdir(parents=True, exist_ok=True)
    ts.to_parquet(OUT / "team_seasons.parquet", index=False)
    spread, sources = openers(raw)
    lo, hi = SCREEN_SEASONS
    sel = games.loc[games["season"].between(lo, hi)].dropna(subset=["result", "spread_line"]).copy()
    margin = sel["result"].astype(float) - sel["spread_line"].astype(float)
    pushes = int(margin.eq(0).sum())
    sel = sel.loc[margin.ne(0)].copy()
    sel["home_cover"] = (margin.loc[sel.index] > 0).astype(float)
    sel["opener"] = spread.reindex(sel["game_id"]).to_numpy(float)
    xmap = ts.set_index(["season", "team"])["x"]
    sel["x"] = [
        float(xmap.get((s, t), 0.0)) if p else 0.0
        for s, t, p in zip(sel["season"], sel["home_team"], sel["primary"], strict=False)
    ]
    res = {
        "schedule_snapshot": snap,
        "opener_sources": sources,
        "opener_coverage_screen": int(sel["opener"].notna().sum()),
        "games": len(sel),
        "pushes_dropped": pushes,
        "neutral_or_secondary_venue_games": int((~games.loc[sel.index, "primary"]).sum()),
        "nonzero_term": int(sel["x"].ne(0).sum()),
        "truncation_check": trunc,
        "inner_loso_shrinkage": picks,
        "descriptives": descriptives(ts, games),
        "cell": screen_cell(sel.reset_index(drop=True), ts, rng),
    }
    sel[
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "spread_line",
            "opener",
            "home_cover",
            "x",
        ]
    ].to_parquet(OUT / "screen_games.parquet", index=False)
    (OUT / "screen.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("cell", "inner_loso_shrinkage")}))
    print(json.dumps({s: round(v["k"], 4) for s, v in picks.items()}))
    print(
        json.dumps({s: round(v["inner_sse"] - v["inner_sse_no_term"], 2) for s, v in picks.items()})
    )
    print(json.dumps({k: v for k, v in res["cell"].items() if k != "per_season"}))
    print(json.dumps({s: round(v, 2) for s, v in res["cell"]["per_season"].items()}))


def batch(classification, evidence, closing_ground):
    res = json.loads((OUT / "screen.json").read_text(encoding="utf-8"))
    cell = res["cell"]
    halves = cell["halves"]
    wins, losses = cell["follow_edge_record_nonzero"]
    entry = {
        "name": "team_hfa_shrunk_venue_edge_close_screen",
        "description": (
            "Team-and-stadium home edge: each home team's prior-season home-minus-road ATS "
            "residual against the close at its primary stadium (league season mean removed), "
            "shrunk toward zero with the shrinkage chosen by inner leave-one-season-out; neutral "
            "and secondary venues zero. Declared sign POSITIVE on home cover. Screen against the "
            "close, REG 2011-2025, pushes dropped; OLS slope of home cover per SD in accuracy "
            f"points. Halves 2011-18 {halves['2011-2018']:+.2f}, 2019-25 "
            f"{halves['2019-2025']:+.2f}; null percentile {cell['null_percentile']:.1f} of "
            f"{SHUFFLES} within-season team-label shuffles (p95 "
            f"{cell['null_p95']:+.2f}); follow-the-edge record {wins}-{losses}; "
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
        "category": "environment",
        "classification_evidence": evidence,
        "notes": (
            f"Family {FAMILY}: one term, declared in the lane before outcomes were joined; at "
            "most 2 gating reads (screen, served look at the opener 2020-25 on QW for a "
            f"survivor). Survivor: {'yes' if cell['survivor'] else 'no'}. Prior related: "
            "mod27_home_nudge (uniform shift), cx15_denver_home, new_stadium_home, roof_state."
        ),
        "interval_low": round(cell["interval_95"][0], 6),
        "interval_high": round(cell["interval_95"][1], 6),
        "probability_positive": round(cell["probability_positive"], 4),
        "sample_games": int(cell["n"]),
        "sample_blocks": 15,
        "plain_summary": (
            "Checks whether stadiums where the home team has beaten the spread more at home "
            "than on the road in past seasons keep doing so, once that history is pulled "
            "toward the league average."
        ),
    }
    if closing_ground:
        entry["closing_ground"] = closing_ground
    path = OUT / "weak_signals_batch.json"
    path.write_text(json.dumps({"cells": [entry]}, indent=1), encoding="utf-8")
    print(path, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Team-and-stadium home edge screen against the close, and its batch cell."
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
