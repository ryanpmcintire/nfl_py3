import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

U6_PATH = Path(__file__).resolve().parent / "mod23_unit6.py"
U6 = {"__file__": str(U6_PATH), "__name__": "mod23_unit6_lib"}
exec(compile(U6_PATH.read_text().rsplit("\nmain()", 1)[0], str(U6_PATH), "exec"), U6)

u3 = U6["u3"]
HARNESS = U6["HARNESS"]
FULL = U6["FULL"]
BASE_ALPHA = U6["BASE_ALPHA"]
ROOT = U6["ROOT"]
OUT = ROOT / "artifacts" / "mod23_unit8"
PROC = ROOT / "data" / "processed" / "wiki_pageviews"
RAW = ROOT / "data" / "raw" / "wiki_pageviews"
BASE5 = U6["BASE5"]
WINDOW_DAYS = 7
BASELINE_DAYS = 28
FIRST_SCORED_SEASON = 2016
ARMS = {"A_qb": ("qb",), "B_all": ("qb", "skill", "defense")}


def load_series(run_id, cache):
    manifest = json.loads((RAW / run_id / "manifest.json").read_text())
    by_title = {m["title"]: m for m in manifest["articles"] if m["status"] == 200}
    series = {}
    for title, m in by_title.items():
        body = json.loads((RAW / run_id / m["file"]).read_text())["items"]
        idx = pd.to_datetime([it["timestamp"][:8] for it in body], format="%Y%m%d")
        series[title] = pd.Series([it["views"] for it in body], index=idx, dtype=float)
    return series


def build():
    run_id = (PROC / "latest_run.txt").read_text().strip()
    cache = json.loads((PROC / "title_resolution.json").read_text())
    series = load_series(run_id, cache)
    st = pd.read_parquet(PROC / "starters.parquet")
    st["title"] = [cache.get(f"{p}|{pos}") for p, pos in zip(st["player"], st["position"], strict=True)]
    sched = pd.read_parquet(u3.SNAPSHOT / "schedules.parquet")
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    gday = sched.set_index("game_id")["gameday"]
    st["gameday"] = st["game_id"].map(gday)
    reg = sched.loc[sched["game_type"].eq("REG")].copy()
    reg["monday"] = reg["gameday"] - pd.to_timedelta(reg["gameday"].dt.weekday, unit="D")
    wk_monday = reg.groupby(["season", "week"])["gameday"].min().rename("first_day").reset_index()
    wk_monday["cutoff"] = wk_monday["first_day"] - pd.to_timedelta(wk_monday["first_day"].dt.weekday, unit="D")
    reg = reg.merge(wk_monday[["season", "week", "cutoff"]], on=["season", "week"])
    st = st.loc[st["title"].notna()]
    player_week = {}
    team_games = st.groupby(["team", "game_id"]).agg(gameday=("gameday", "first")).reset_index().sort_values(["team", "gameday"])
    cutoffs = reg[["season", "week", "cutoff"]].drop_duplicates()
    rows = []
    for (season, week), cut in zip(zip(cutoffs["season"], cutoffs["week"], strict=True), cutoffs["cutoff"], strict=True):
        if season < FIRST_SCORED_SEASON:
            continue
        for team, g in team_games.groupby("team"):
            prev = g.loc[g["gameday"] < cut]
            if prev.empty:
                continue
            gid = prev.iloc[-1]["game_id"]
            sub = st.loc[(st["game_id"] == gid) & (st["team"] == team)]
            w0 = cut - pd.Timedelta(days=WINDOW_DAYS)
            b0 = w0 - pd.Timedelta(days=BASELINE_DAYS)
            for role, title in zip(sub["role"], sub["title"], strict=True):
                s = series.get(title)
                if s is None:
                    continue
                win_idx = pd.date_range(w0, cut - pd.Timedelta(days=1))
                base_idx = pd.date_range(b0, w0 - pd.Timedelta(days=1))
                win = s.reindex(win_idx).fillna(0.0)
                base = s.reindex(base_idx).fillna(0.0)
                rows.append(
                    {
                        "season": int(season),
                        "week": int(week),
                        "team": team,
                        "role": role,
                        "title": title,
                        "cutoff": cut,
                        "max_day_used": win_idx.max(),
                        "src_game_day": g.loc[g["game_id"] == gid, "gameday"].iloc[0],
                        "raw": float(np.log1p(win.mean()) - np.log1p(base.mean())),
                    }
                )
    pw = pd.DataFrame(rows)
    if not (pw["max_day_used"] < pw["cutoff"]).all():
        raise SystemExit("LEAK CHECK FAILED: day used >= cutoff")
    if not (pw["src_game_day"] < pw["cutoff"]).all():
        raise SystemExit("LEAK CHECK FAILED: starter game not before cutoff")
    print("leak check rows", len(pw), "max(max_day_used - cutoff) days", float((pw["max_day_used"] - pw["cutoff"]).max().days), flush=True)
    sd = pw.groupby("season")["raw"].agg(["sum", "count"])
    sq = (pw["raw"] ** 2).groupby(pw["season"]).sum()
    scale = {}
    for s in sorted(pw["season"].unique()):
        prior = [x for x in sd.index if x < s]
        if not prior:
            scale[int(s)] = None
            continue
        n = sd.loc[prior, "count"].sum()
        mu = sd.loc[prior, "sum"].sum() / n
        var = sq.loc[prior].sum() / n - mu**2
        scale[int(s)] = (float(mu), float(np.sqrt(var)))
    prior_max = {s: max([x for x in sd.index if x < s], default=None) for s in scale}
    if any(m is not None and m >= s for s, m in prior_max.items()):
        raise SystemExit("SCALE LEAK")
    pw["z"] = [
        0.0 if scale[s] is None else (r - scale[s][0]) / scale[s][1] for s, r in zip(pw["season"], pw["raw"], strict=True)
    ]
    pw.to_parquet(OUT / "player_week_scores.parquet")
    (OUT / "scale_prior_seasons.json").write_text(json.dumps({str(k): v for k, v in scale.items()}, indent=1))
    games = sched.loc[sched["game_type"].eq("REG"), ["game_id", "season", "week", "home_team", "away_team"]]
    out = games[["game_id"]].copy()
    for arm, roles in ARMS.items():
        tw = pw.loc[pw["role"].isin(roles)].groupby(["season", "week", "team"])["z"].mean().rename("att").reset_index()
        h = games.merge(tw, left_on=["season", "week", "home_team"], right_on=["season", "week", "team"], how="left")["att"]
        a = games.merge(tw, left_on=["season", "week", "away_team"], right_on=["season", "week", "team"], how="left")["att"]
        out[f"wiki_{arm}_diff"] = (h.to_numpy() - a.to_numpy())
        cov = float(out[f"wiki_{arm}_diff"].notna().mean())
        print(arm, "coverage", round(cov, 4), flush=True)
        out[f"wiki_{arm}_diff"] = out[f"wiki_{arm}_diff"].fillna(0.0)
    out.to_parquet(OUT / "game_features.parquet")
    print(out.describe().T, flush=True)


def grade():
    features = pd.read_parquet(u3.FEATURES)
    gf = pd.read_parquet(OUT / "game_features.parquet")
    features = features.merge(gf, on="game_id", how="left")
    for c in [c for c in gf.columns if c != "game_id"]:
        features[c] = features[c].fillna(0.0)
    pop = HARNESS["proxy_population"](features)
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    if (int(tr["correct"].sum()), len(tr)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    results = {}
    cells = []
    for arm in ARMS:
        col = f"wiki_{arm}_diff"
        path = OUT / f"{arm}_run.parquet"
        cols = tuple(FULL) + (col,)
        if path.exists():
            run = pd.read_parquet(path)
        else:
            run = HARNESS["run_arm"](features, pop, lambda s, cols=cols: (cols, BASE_ALPHA))
            run.to_parquet(path)
        run = run.loc[run.index.isin(base.index)]
        results[arm] = U6["compare"](run, base)
        for era in ("proxy_2011_2019", "true_2020_2025"):
            r = results[arm][era]
            print(arm, era, r["record"], r["record_base"], round(r["d_acc"], 3), r["d_acc_season_boot"], "flips", r["flips"], r["flips_W_L"], "dLL", round(r["d_ll_arm_minus_base"], 5), "dBrier", round(r["d_brier_arm_minus_base"], 5), flush=True)
            lo, hi = (2011, 2019) if era.startswith("proxy") else (2020, 2025)
            cells.append(
                {
                    "source": "artifacts/mod23_unit8/report.json",
                    "effect_units": "accuracy_points",
                    "classification": "unresolved_below_power",
                    "league": "nfl",
                    "season_start": lo,
                    "season_end": hi,
                    "family": "mod23_unit8_player_pageviews",
                    "category": "modeling",
                    "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
                    "notes": "Two arms, two eras (four accuracy looks); player-level Wikipedia abnormal-attention difference added as one ridge feature; pageviews exist from 2016 so earlier eras carry zeros.",
                    "name": f"mod23_unit8_pageviews_{arm}_{era}_accuracy_diff",
                    "description": f"Base model plus home-minus-away abnormal Wikipedia attention to starters ({arm}), paired vs base, {lo}-{hi}; {r['record']} vs {r['record_base']}.",
                    "effect": r["d_acc"],
                    "interval_low": r["d_acc_season_boot"]["low"],
                    "interval_high": r["d_acc_season_boot"]["high"],
                    "probability_positive": r["d_acc_season_boot"]["pp"],
                    "sample_games": r["n"],
                    "sample_blocks": r["d_acc_season_boot"]["blocks"],
                    "plain_summary": "Adding how unusually busy fans were on the starters' Wikipedia pages did not reliably change picks against the opener.",
                }
            )
    (OUT / "report.json").write_text(json.dumps(results, indent=1, default=str))
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "grade"
    if stage == "build":
        build()
    else:
        grade()


main()
