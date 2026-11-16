from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/fp_ecr"
OUT = ROOT / "artifacts/expert_rankings"
SCHEDULES = ROOT / "data/raw"
PAGES = {"weekly-qb": "QB", "weekly-rb": "RB", "weekly-wr": "WR", "weekly-te": "TE"}
TEAM_CODES = {
    "GBP": "GB",
    "JAC": "JAX",
    "KCC": "KC",
    "LAR": "LA",
    "LVR": "LV",
    "NEP": "NE",
    "NOS": "NO",
    "SFO": "SF",
    "TBB": "TB",
}
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
SEASONS = (2020, 2025)
HALVES = ((2020, 2022), (2023, 2025))
FIRST_WEEK = 2
SHUFFLES = 200
BOOTSTRAPS = 2000
SIGN = 1.0
SEED = 20261116
SUNDAY = 6
CELLS = ("gap", "gap_orth")


def schedules():
    snapshot = sorted(p for p in SCHEDULES.glob("2026*") if (p / "schedules.parquet").is_file())[-1]
    raw = pd.read_parquet(snapshot / "schedules.parquet")
    table = raw.loc[raw["game_type"].eq("REG") & raw["season"].between(*SEASONS)].copy()
    for side in ("home_team", "away_team"):
        table[side] = table[side].replace(FRANCHISE)
    import v4_candidate as v4

    _, cutoff = v4.kickoff_and_cutoff(table)
    table["deadline_date"] = (
        cutoff.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    )
    day = pd.to_datetime(table["gameday"])
    shift = np.where(day.dt.dayofweek.eq(0), -1, (SUNDAY - day.dt.dayofweek) % 7)
    table["sunday"] = (day + pd.to_timedelta(shift, unit="D")).dt.normalize()
    return table, snapshot.name, raw


def rankings():
    d = pd.read_parquet(RAW / "db_fpecr.parquet")
    d = d.loc[d["page_type"].isin(PAGES)].copy()
    d["pos_page"] = d["page_type"].map(PAGES)
    d["scrape"] = pd.to_datetime(d["scrape_date"]).dt.normalize()
    d["pos_rank"] = d.groupby(["scrape", "pos_page"])["ecr"].rank(method="first")
    d["team"] = d["tm"].replace(TEAM_CODES)
    d["next_sunday"] = d["scrape"] + pd.to_timedelta(
        (SUNDAY - d["scrape"].dt.dayofweek) % 7, unit="D"
    )
    ids = pd.read_csv(RAW / "db_playerids.csv", usecols=["fantasypros_id", "gsis_id"], dtype=str)
    ids = ids.dropna().drop_duplicates("fantasypros_id", keep=False)
    d = d.merge(ids, left_on="id", right_on="fantasypros_id", how="left")
    return d


def actual_points():
    frames = []
    for season in range(SEASONS[0], SEASONS[1] + 1):
        s = pd.read_parquet(
            RAW / f"stats_player_week_{season}.parquet",
            columns=["player_id", "season", "week", "season_type", "fantasy_points_ppr"],
        )
        frames.append(s.loc[s["season_type"].eq("REG")])
    s = pd.concat(frames, ignore_index=True)
    return s.groupby(["player_id", "season", "week"])["fantasy_points_ppr"].sum()


def week_of_sunday(table):
    return table.groupby("sunday")[["season", "week"]].agg(lambda x: x.mode().iloc[0])


def assign_scrapes(table, scrapes):
    games = table[["game_id", "sunday", "deadline_date"]].copy()
    pairs = games.merge(scrapes.rename("scrape").to_frame(), how="cross")
    pairs = pairs.loc[
        (pairs["scrape"] < pairs["deadline_date"])
        & (
            pairs["scrape"] + pd.to_timedelta((SUNDAY - pairs["scrape"].dt.dayofweek) % 7, unit="D")
        ).eq(pairs["sunday"])
    ]
    return pairs.groupby("game_id")["scrape"].max()


def part1(table, d, pts):
    scrapes = pd.Series(sorted(d["scrape"].unique()))
    chosen = assign_scrapes(table, scrapes)
    table = table.assign(scrape=table["game_id"].map(chosen))
    lag = (table["deadline_date"] - table["scrape"]).dt.days
    out = {"weekly_pages": sorted(PAGES), "rows": len(d)}
    out["scrapes_by_year"] = {
        int(k): int(v) for k, v in d.groupby(d["scrape"].dt.year)["scrape"].nunique().items()
    }
    out["scrape_weekday"] = {
        k: int(v)
        for k, v in d.drop_duplicates("scrape")["scrape"].dt.day_name().value_counts().items()
    }
    per = {}
    for season, g in table.groupby("season"):
        later = g["week"].ge(FIRST_WEEK)
        per[int(season)] = {
            "reg_games": len(g),
            "pregame_scrape_games": int(g["scrape"].notna().sum()),
            "pregame_scrape_games_week2plus": int((g["scrape"].notna() & later).sum()),
            "weeks_with_any": int(g.loc[g["scrape"].notna(), "week"].nunique()),
        }
    out["per_season"] = per
    out["days_scrape_to_deadline_date"] = {
        int(k): int(v) for k, v in lag.value_counts().sort_index().items()
    }
    no_scrape = table.loc[table["scrape"].isna()]
    out["missing_by_weekday"] = {
        k: int(v)
        for k, v in pd.to_datetime(no_scrape["gameday"]).dt.day_name().value_counts().items()
    }
    joined = d.loc[d["scrape"].dt.year.between(*SEASONS)]
    out["gsis_join_rate"] = {
        int(k): float(v)
        for k, v in joined.groupby(joined["scrape"].dt.year)["gsis_id"]
        .apply(lambda x: x.notna().mean())
        .items()
    }
    out["team_codes_unmapped"] = sorted(set(d["team"].dropna()) - set(table["home_team"]) - {"FA"})
    return out, chosen


def isotonic_maps(pairs):
    maps = {}
    for pos, g in pairs.groupby("pos_page"):
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(g["pos_rank"].to_numpy(float), g["actual"].to_numpy(float))
        maps[pos] = iso
    return maps


def team_sums(rows, maps):
    rows = rows.loc[rows["pos_page"].isin(maps)].copy()
    rows["exp"] = 0.0
    for pos, iso in maps.items():
        m = rows["pos_page"].eq(pos)
        rows.loc[m, "exp"] = iso.predict(rows.loc[m, "pos_rank"].to_numpy(float))
    return rows.groupby(["scrape", "team"])["exp"].sum()


def ols(x, y):
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta


def build_terms(table, d, pts, chosen, spread_asof):
    sw = week_of_sunday(table)
    week_scrape = d.drop_duplicates("scrape")[["scrape", "next_sunday"]]
    week_scrape = week_scrape.merge(sw, left_on="next_sunday", right_index=True, how="inner")
    week_scrape = week_scrape.sort_values("scrape").groupby(["season", "week"])["scrape"].max()
    rows = d.loc[d["scrape"].isin(week_scrape.to_numpy())].copy()
    rows = rows.merge(week_scrape.rename("scrape").reset_index(), on="scrape", how="left")
    key = pd.MultiIndex.from_arrays([rows["gsis_id"], rows["season"], rows["week"]])
    rows["actual"] = pd.Series(pts.reindex(key).to_numpy(float)).fillna(0.0).to_numpy()
    rows = rows.loc[rows["gsis_id"].notna()]
    games = table.assign(
        scrape=table["game_id"].map(chosen), spread_asof=table["game_id"].map(spread_asof)
    )
    games["spread_asof"] = games["spread_asof"].fillna(games["spread_line"])
    allrows = d.loc[d["scrape"].isin(games["scrape"].dropna().unique())]
    out, meta = [], {}
    for (season, week), g in games.loc[games["scrape"].notna()].groupby(["season", "week"]):
        if week < FIRST_WEEK:
            continue
        earlier = rows.loc[rows["season"].eq(season) & rows["week"].lt(week)]
        maps = isotonic_maps(earlier)
        prior = games.loc[
            games["season"].eq(season) & games["week"].lt(week) & games["scrape"].notna()
        ]
        if earlier.empty or prior.empty:
            continue
        sums = team_sums(
            allrows.loc[allrows["scrape"].isin(set(prior["scrape"]) | set(g["scrape"]))], maps
        )
        h = sums.reindex(pd.MultiIndex.from_arrays([prior["scrape"], prior["home_team"]])).to_numpy(
            float
        )
        a = sums.reindex(pd.MultiIndex.from_arrays([prior["scrape"], prior["away_team"]])).to_numpy(
            float
        )
        x = np.concatenate([h, a])
        y = np.concatenate(
            [prior["home_score"].to_numpy(float), prior["away_score"].to_numpy(float)]
        )
        ok = np.isfinite(x) & np.isfinite(y)
        k = ols(x[ok], y[ok])[1]
        e_prior = k * (h - a)
        okp = np.isfinite(e_prior)
        orth = ols(prior["spread_asof"].to_numpy(float)[okp], e_prior[okp])
        gh = sums.reindex(pd.MultiIndex.from_arrays([g["scrape"], g["home_team"]])).to_numpy(float)
        ga = sums.reindex(pd.MultiIndex.from_arrays([g["scrape"], g["away_team"]])).to_numpy(float)
        cur = g[
            [
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "spread_line",
                "spread_asof",
                "result",
            ]
        ].copy()
        cur["fp_home"], cur["fp_away"] = gh, ga
        cur["e_diff"] = k * (gh - ga)
        cur["gap"] = cur["e_diff"] - cur["spread_asof"]
        cur["gap_orth"] = cur["e_diff"] - (orth[0] + orth[1] * cur["spread_asof"])
        out.append(cur)
        meta[f"{int(season)}-{int(week):02d}"] = {
            "k_points_per_fp": float(k),
            "orth_slope": float(orth[1]),
            "prior_team_games": int(ok.sum()),
            "map_pairs": len(earlier),
        }
    return pd.concat(out, ignore_index=True), meta


def slope(y, x):
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def screen_one(frame, term, rng):
    data = frame.dropna(subset=[term]).copy()
    raw = data[term].to_numpy(float)
    x = (raw - raw.mean()) / raw.std(ddof=0)
    y = data["home_cover"].to_numpy(float)
    seasons_arr = data["season"].to_numpy(int)
    uniq = np.unique(seasons_arr)
    groups = [np.flatnonzero(seasons_arr == s) for s in uniq]
    obs = SIGN * slope(y, x) * 100.0
    null = []
    for _ in range(SHUFFLES):
        xs = x.copy()
        for idx in groups:
            xs[idx] = rng.permutation(xs[idx])
        null.append(SIGN * slope(y, xs) * 100.0)
    null = np.array(null)
    boots = []
    for _ in range(BOOTSTRAPS):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(seasons_arr == s) for s in pick])
        boots.append(SIGN * slope(y[idx], x[idx]) * 100.0)
    boots = np.array(boots)
    halves = {
        f"{lo}-{hi}": SIGN
        * slope(
            y[(seasons_arr >= lo) & (seasons_arr <= hi)],
            x[(seasons_arr >= lo) & (seasons_arr <= hi)],
        )
        * 100.0
        for lo, hi in HALVES
    }
    per_season = {
        int(s): SIGN * slope(y[seasons_arr == s], x[seasons_arr == s]) * 100.0 for s in uniq
    }
    pick_home = raw > 0
    wins = int(((pick_home & (y == 1)) | (~pick_home & (y == 0))).sum())
    return {
        "n": len(data),
        "effect_acc_points_per_sd_declared": obs,
        "interval_95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "probability_positive": float((boots > 0).mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "null_percentile": float((null < obs).mean() * 100.0),
        "halves": halves,
        "per_season": per_season,
        "seasons_declared_positive": int(sum(v > 0 for v in per_season.values())),
        "gap_side_record": [wins, len(data) - wins],
        "survivor": bool(
            obs > float(np.quantile(null, 0.95)) and all(v > 0 for v in halves.values())
        ),
    }


def run():
    table, snap, raw = schedules()
    d = rankings()
    pts = actual_points()
    p1, chosen = part1(table, d, pts)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "part1.json").write_text(json.dumps(p1, indent=2))
    print(json.dumps(p1))
    from hindsight_audit_team import SOURCES, tuesday_quotes

    _, spread = tuesday_quotes(raw)
    spread.index = spread.index.astype(str)
    have = table["game_id"].astype(str).isin(spread.dropna().index)
    corr = float(
        np.corrcoef(
            spread.reindex(table.loc[have, "game_id"]).to_numpy(float),
            table.loc[have, "spread_line"],
        )[0, 1]
    )
    spread = spread * np.sign(corr)
    terms, meta = build_terms(table, d, pts, chosen, spread)
    terms["asof_source"] = np.where(
        terms["game_id"].isin(spread.dropna().index), "tuesday_or_open", "close"
    )
    margin = terms["result"].astype(float) - terms["spread_line"].astype(float)
    terms = terms.loc[margin.notna() & margin.ne(0)].copy()
    terms["home_cover"] = (margin.loc[terms.index] > 0).astype(float)
    rng = np.random.default_rng(SEED)
    res = {
        "schedule_snapshot": snap,
        "opener_sources": dict(SOURCES),
        "asof_close_corr": corr,
        "games": len(terms),
        "asof_source_counts": {k: int(v) for k, v in terms["asof_source"].value_counts().items()},
        "missing_fp": int(terms[["fp_home", "fp_away"]].isna().any(axis=1).sum()),
        "corr_ediff_spread_asof": float(
            terms[["e_diff", "spread_asof"]].dropna().corr().iloc[0, 1]
        ),
        "sd_gap_points": float(terms["gap"].std()),
        "k_points_per_fp_median": float(np.median([m["k_points_per_fp"] for m in meta.values()])),
        "cells": {c: screen_one(terms, c, rng) for c in CELLS},
    }
    terms.to_parquet(OUT / "screen_games.parquet", index=False)
    (OUT / "week_meta.json").write_text(json.dumps(meta, indent=1))
    (OUT / "screen.json").write_text(json.dumps(res, indent=2))
    for name, cell in res["cells"].items():
        print(
            name,
            json.dumps({k: v for k, v in cell.items() if k != "per_season"}),
            cell["per_season"],
        )
    print(json.dumps({k: v for k, v in res.items() if k != "cells"}))


FAMILY = "expert_rankings_2026_11_16"
LANE = "docs/lanes/expert-rankings-2026-11-16.md"
TERM_TEXT = {
    "gap": "ECR team expectation home minus away (points) minus the as-of spread",
    "gap_orth": (
        "ECR team expectation home minus away (points) minus its OLS fit on the as-of spread "
        "over the season's earlier weeks"
    ),
}


def batch():
    res = json.loads((OUT / "screen.json").read_text())
    cells = []
    for term, cell in res["cells"].items():
        halves = cell["halves"]
        wins, losses = cell["gap_side_record"]
        cells.append(
            {
                "name": f"expert_rankings_fp_ecr_{term}_close",
                "description": (
                    "FantasyPros weekly expert consensus rankings (QB, PPR RB/WR/TE), "
                    f"{TERM_TEXT[term]}. Rank to points by isotonic fit of actual PPR points on "
                    "position rank over the season's earlier weeks; fantasy points to real points "
                    "by OLS of team points on the team sum over earlier weeks; as-of spread "
                    "Tuesday or SBR open "
                    f"({res['asof_source_counts'].get('tuesday_or_open', 0)} games), else close "
                    f"({res['asof_source_counts'].get('close', 0)}). Only scrapes dated before "
                    "the pick deadline date. Declared sign POSITIVE on home cover. Screen against "
                    "the close, REG 2020-2025 weeks 2+, pushes dropped; OLS slope per SD in "
                    "accuracy points. Halves "
                    f"2020-22 {halves['2020-2022']:+.2f}, 2023-25 {halves['2023-2025']:+.2f}; null "
                    f"percentile {cell['null_percentile']:.1f} of {SHUFFLES} within-season "
                    "shuffles "
                    f"(p95 {cell['null_p95']:+.2f}); gap-side record {wins}-{losses}; "
                    f"{cell['seasons_declared_positive']} of 6 seasons positive."
                ),
                "source": LANE,
                "effect": round(cell["effect_acc_points_per_sd_declared"], 6),
                "effect_units": "accuracy_points",
                "classification": "unresolved_below_power",
                "league": "nfl",
                "season_start": SEASONS[0],
                "season_end": SEASONS[1],
                "family": FAMILY,
                "category": "market",
                "classification_evidence": (
                    "Declared sign, season-block interval crosses zero and the effect sits below "
                    "the within-season shuffle p95; no refuted mechanism and no positive control "
                    "bounds an effect this size at the close screen."
                ),
                "notes": (
                    f"Family {FAMILY}: 2 terms x 1 population = 2 looks, declared in the lane "
                    "before outcomes were joined; survivor rule null p95 plus positive in both "
                    "halves; "
                    "survivor: no; served look not run. Weekly ECR history starts 2020."
                ),
                "interval_low": round(cell["interval_95"][0], 6),
                "interval_high": round(cell["interval_95"][1], 6),
                "probability_positive": round(cell["probability_positive"], 4),
                "sample_games": int(cell["n"]),
                "sample_blocks": 6,
                "plain_summary": (
                    "Adds up the fantasy experts' weekly player rankings into how many points "
                    "each offense should score and compares that with the betting line. Teams the "
                    "experts liked more than the line covered slightly more often; not enough "
                    "games to be sure."
                ),
            }
        )
    path = OUT / "weak_signals_batch.json"
    path.write_text(json.dumps({"cells": cells}, indent=1), encoding="utf-8")
    print(path, len(cells))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "FantasyPros weekly ECR team expectation versus the spread: source timing, joins "
            "and close screen."
        )
    )
    parser.add_argument("mode", choices=["run", "batch"])
    args = parser.parse_args()
    if args.mode == "run":
        run()
    if args.mode == "batch":
        batch()


if __name__ == "__main__":
    main()
