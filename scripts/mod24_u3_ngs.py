import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ngs"
OUT = ROOT / "data" / "processed" / "game_features_ngs.parquet"
MANIFEST = ROOT / "data" / "processed" / "game_features_ngs.manifest.json"

METRICS = {
    "passing": ("attempts", ["avg_time_to_throw", "aggressiveness", "avg_completed_air_yards", "completion_percentage_above_expectation", "avg_intended_air_yards"]),
    "receiving": ("targets", ["avg_separation", "avg_cushion", "avg_yac_above_expectation"]),
    "rushing": ("rush_attempts", ["efficiency", "percent_attempts_gte_eight_defenders", "rush_yards_over_expected_per_att", "avg_time_to_los"]),
}
SHRINK_GAMES = 4.0


def latest_dir():
    return sorted(p for p in RAW.iterdir() if p.is_dir())[-1]


def team_week_table(src):
    frames = []
    for kind, (wcol, cols) in METRICS.items():
        df = pd.read_parquet(src / f"ngs_{kind}.parquet")
        df = df[df["week"] > 0].dropna(subset=["team_abbr"])
        df = df[df[wcol] > 0]
        for c in cols:
            sub = df[["season", "week", "team_abbr", wcol, c]].dropna()
            sub = sub.assign(wv=sub[c] * sub[wcol])
            g = sub.groupby(["season", "week", "team_abbr"], as_index=False).agg(wv=("wv", "sum"), n=(wcol, "sum"))
            g["metric"] = f"{kind}__{c}"
            g["v"] = g["wv"] / g["n"]
            frames.append(g[["season", "week", "team_abbr", "metric", "v", "n"]])
    return pd.concat(frames, ignore_index=True)


def shrunk_estimates(tw, cutoffs):
    out = []
    tw = tw.copy()
    tw["wv"] = tw["v"] * tw["n"]
    teams = sorted(tw["team_abbr"].unique())
    metrics = sorted(tw["metric"].unique())
    grid = pd.MultiIndex.from_product([teams, metrics], names=["team_abbr", "metric"]).to_frame(index=False)
    for season, week in cutoffs:
        past = tw[(tw["season"] == season) & (tw["week"] < week)]
        prev = tw[tw["season"] == season - 1]
        assert (past["week"] < week).all()
        before = tw[(tw["season"] < season) | ((tw["season"] == season) & (tw["week"] < week))]
        kmap = {m: SHRINK_GAMES * g["n"].mean() for m, g in before.groupby("metric")}
        cur = past.groupby(["team_abbr", "metric"]).agg(wv=("wv", "sum"), n=("n", "sum"), maxw=("week", "max")).reset_index()
        pr = prev.groupby(["team_abbr", "metric"]).agg(pwv=("wv", "sum"), pn=("n", "sum")).reset_index()
        pr["prior"] = pr["pwv"] / pr["pn"]
        lg_prev = (prev.groupby("metric")["wv"].sum() / prev.groupby("metric")["n"].sum()).rename("lg_prev")
        lg_cur = (past.groupby("metric")["wv"].sum() / past.groupby("metric")["n"].sum()).rename("lg_cur")
        base = grid.merge(cur, how="left", on=["team_abbr", "metric"]).merge(pr[["team_abbr", "metric", "prior"]], how="left", on=["team_abbr", "metric"])
        base = base.merge(lg_prev, left_on="metric", right_index=True, how="left").merge(lg_cur, left_on="metric", right_index=True, how="left")
        base["prior"] = base["prior"].fillna(base["lg_prev"]).fillna(base["lg_cur"])
        base["k"] = base["metric"].map(kmap)
        base["wv"] = base["wv"].fillna(0.0)
        base["n"] = base["n"].fillna(0.0)
        base["est"] = (base["wv"] + base["k"] * base["prior"]) / (base["n"] + base["k"])
        base["maxw"] = base["maxw"].fillna(0)
        base["season"] = season
        base["week"] = week
        out.append(base[["season", "week", "team_abbr", "metric", "est", "maxw"]])
    return pd.concat(out, ignore_index=True)


def allowed_table(tw, games):
    sched = pd.concat(
        [
            games[["season", "week", "home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opp"}),
            games[["season", "week", "away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "opp"}),
        ]
    )
    m = sched.merge(tw.rename(columns={"team_abbr": "opp"}), on=["season", "week", "opp"], how="inner")
    m = m.drop(columns="opp").rename(columns={"team": "team_abbr"})
    m["metric"] = "allowed__" + m["metric"]
    return m[["season", "week", "team_abbr", "metric", "v", "n"]]


def main():
    src = latest_dir()
    games = pd.read_parquet(ROOT / "data" / "processed" / "game_features.parquet", columns=["game_id", "season", "week", "home_team", "away_team"])
    games = games[(games["season"] >= 2016) & (games["season"] <= 2026)].copy()
    own = team_week_table(src)
    tw = pd.concat([own, allowed_table(own, games)], ignore_index=True)
    cutoffs = sorted(set(map(tuple, games[["season", "week"]].drop_duplicates().to_numpy())))
    est = shrunk_estimates(tw, cutoffs)
    assert (est["maxw"] < est["week"]).all()
    wide = est.pivot_table(index=["season", "week", "team_abbr"], columns="metric", values="est").reset_index()
    wmax = est.groupby(["season", "week", "team_abbr"])["maxw"].max().rename("max_week_used").reset_index()
    wide = wide.merge(wmax, on=["season", "week", "team_abbr"])
    feats = [c for c in wide.columns if "__" in c]
    hw = wide.rename(columns={"team_abbr": "home_team"}).add_prefix("h_").rename(columns={"h_season": "season", "h_week": "week", "h_home_team": "home_team"})
    aw = wide.rename(columns={"team_abbr": "away_team"}).add_prefix("a_").rename(columns={"a_season": "season", "a_week": "week", "a_away_team": "away_team"})
    h = games.merge(hw, on=["season", "week", "home_team"], how="left").merge(aw, on=["season", "week", "away_team"], how="left")
    res = h[["game_id", "season", "week"]].copy()
    res["max_week_used_home"] = h["h_max_week_used"]
    res["max_week_used_away"] = h["a_max_week_used"]
    for c in feats:
        res["diff_ngs_" + c] = h["h_" + c].to_numpy() - h["a_" + c].to_numpy()
    assert (res["max_week_used_home"].fillna(0) < res["week"]).all() and (res["max_week_used_away"].fillna(0) < res["week"]).all()
    print("missing max_week_used", int(res["max_week_used_home"].isna().sum()), res.loc[res["max_week_used_home"].isna(), "season"].value_counts().to_dict())
    rng = np.random.default_rng(0)
    picks = rng.choice(len(cutoffs), size=12, replace=False)
    worst = 0.0
    for i in picks:
        s, w = cutoffs[i]
        trunc = tw[(tw["season"] < s) | ((tw["season"] == s) & (tw["week"] < w))]
        e2 = shrunk_estimates(trunc, [(s, w)])
        e1 = est[(est["season"] == s) & (est["week"] == w)]
        j = e1.merge(e2, on=["team_abbr", "metric"], suffixes=("_a", "_b"))
        d = float((j["est_a"] - j["est_b"]).abs().max())
        worst = max(worst, d)
        print("truncation check", s, w, "max abs diff", d)
    assert worst < 1e-9
    res.to_parquet(OUT, index=False)
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    MANIFEST.write_text(
        json.dumps(
            {
                "source_dir": str(src.relative_to(ROOT)),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(src.glob("*.parquet"))},
                "output_sha256": sha,
                "rows": len(res),
                "features": ["diff_ngs_" + c for c in feats],
                "shrink_games": SHRINK_GAMES,
                "rule": "game in season s week w uses same-season weeks < w, volume-weighted, shrunk toward last season team value (league mean fallback) with k = 4 mean team-week volumes",
            },
            indent=2,
        )
    )
    print("rows", len(res), "features", len(feats))


def describe():
    sys.path.insert(0, str(ROOT / "src"))
    from nfl_ats.schedule_flag_features import default_opener_lines

    f = pd.read_parquet(OUT)
    g = pd.read_parquet(ROOT / "data" / "processed" / "game_features.parquet", columns=["game_id", "season", "week", "result", "ats_margin", "spread_line"])
    d = f.merge(g.drop(columns=["season", "week"]), on="game_id")
    op = default_opener_lines(g[["game_id", "season", "week"]])
    d = d.merge(op[["game_id", "tue_open_home_spread"]], on="game_id", how="left")
    d = d[(d["season"] >= 2020) & (d["season"] <= 2025)].copy()
    print("games 2020-2025", len(d), "with opener", int(d["tue_open_home_spread"].notna().sum()))
    print("corr(spread_line, tue_open_home_spread)", d["spread_line"].corr(d["tue_open_home_spread"]))
    d["margin_vs_open"] = d["result"] - d["tue_open_home_spread"]
    rows = []
    for c in [c for c in d.columns if c.startswith("diff_ngs_")]:
        x = d[[c, "margin_vs_open", "ats_margin", "result"]].dropna()
        rows.append((c, len(x), x[c].corr(x["result"]), x[c].corr(x["margin_vs_open"]), x[c].corr(x["ats_margin"])))
    out = pd.DataFrame(rows, columns=["feature", "n", "r_result", "r_vs_opener", "r_vs_close"]).round(3)
    print(out.to_string(index=False))
    c0 = [c for c in f.columns if c.startswith("diff_ngs_")][0]
    print(f.groupby("season")[c0].apply(lambda s: s.notna().mean()).round(3).to_dict())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "describe":
        describe()
    else:
        main()
