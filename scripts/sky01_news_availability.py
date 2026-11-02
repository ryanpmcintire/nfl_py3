import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "sky01"
POSTS = ROOT / "data" / "processed" / "beat_reporter_posts.parquet"
PLAY = ROOT / "data" / "processed" / "injury_play_outcomes.parquet"
PERGAME = ROOT / "artifacts" / "pick_probability" / "20260929T192747Z" / "per_game.parquet"
SCHED = ROOT / "data" / "raw" / "20261027T220410Z" / "schedules.parquet"
OUT_CLASSES = ("ruled_out", "did_not_practice", "absent")
LIMITED_CLASSES = ("limited", "questionable", "boot_brace")
ACTIVE_CLASSES = ("full", "expected_to_play", "first_team_reps")
WEEK = pd.Timedelta(days=7)
ET = "America/New_York"
BOOT = 2000
SEED = 20261102
ALIAS = {"LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV", "LVR": "LV"}


def alias(t):
    return ALIAS.get(t, t)


def schedule():
    s = pd.read_parquet(SCHED)
    s = s[s.game_type == "REG"].copy()
    ko = pd.to_datetime(s.gameday + " " + s.gametime.fillna("13:00"))
    s["kickoff"] = ko.dt.tz_localize(ET, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    s["home_team"] = s.home_team.map(alias)
    s["away_team"] = s.away_team.map(alias)
    gd = pd.to_datetime(s.gameday)
    back = (gd.dt.weekday - 1) % 7
    op = gd - pd.to_timedelta(back, unit="D") + pd.Timedelta(hours=12)
    s["opener_ts"] = op.dt.tz_localize(ET, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    return s.dropna(subset=["kickoff"]).reset_index(drop=True)


def team_games(s):
    cols = ["season", "week", "game_id", "kickoff", "opener_ts"]
    a = s[cols + ["home_team"]].rename(columns={"home_team": "team"})
    b = s[cols + ["away_team"]].rename(columns={"away_team": "team"})
    t = pd.concat([a, b]).sort_values(["team", "kickoff"]).reset_index(drop=True)
    t["prev_kickoff"] = t.groupby("team").kickoff.shift(1)
    t["window_start"] = t.prev_kickoff.fillna(t.opener_ts - WEEK)
    return t


def beat_rows(tg):
    p = pd.read_parquet(POSTS)
    p = p[(~p.is_repost) & (p.avail_classes.map(len) > 0) & (p.player_gsis.map(len) > 0)].copy()
    p["team"] = p.team.map(alias)
    p = p.explode("player_gsis").explode("avail_classes").rename(columns={"player_gsis": "gsis_id", "avail_classes": "cls"})
    p = p.dropna(subset=["gsis_id", "cls"])
    p["created_at"] = p.created_at.dt.tz_convert("UTC")
    out = []
    for team, g in p.groupby("team"):
        t = tg[tg.team == team].sort_values("kickoff")
        kv = t.kickoff.dt.tz_convert("UTC").dt.tz_localize(None).values.astype("datetime64[ns]")
        cv = g.created_at.dt.tz_localize(None).values.astype("datetime64[ns]")
        idx = np.searchsorted(kv, cv, side="right")
        ok = idx < len(t)
        gg = g[ok].copy()
        tt = t.iloc[idx[ok]]
        gg["season"] = tt.season.values
        gg["week"] = tt.week.values
        gg["game_id"] = tt.game_id.values
        gg["opener_ts"] = tt.opener_ts.values
        gg["window_start"] = tt.window_start.values
        gg["kickoff"] = tt.kickoff.values
        out.append(gg)
    p = pd.concat(out)
    p["window_start"] = pd.to_datetime(p.window_start, utc=True)
    p["opener_ts"] = pd.to_datetime(p.opener_ts, utc=True)
    p["kickoff"] = pd.to_datetime(p.kickoff, utc=True)
    return p[p.created_at >= p.window_start]


def first_ts(p, classes, name):
    q = p[p.cls.isin(classes)]
    return q.groupby(["season", "week", "gsis_id", "team"]).created_at.min().rename(name)


def official():
    cols = ["season", "week", "gsis_id", "team", "report_status", "practice_status", "decision_observed_at", "observed_at_is_proxy", "played", "position"]
    x = pd.read_parquet(PLAY, columns=cols)
    x["decision_observed_at"] = pd.to_datetime(x.decision_observed_at, utc=True)
    x["team"] = x.team.map(alias)
    return x.rename(columns={"decision_observed_at": "official_ts"}).drop_duplicates(["season", "week", "gsis_id"])


def snapshots():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/nflverse_injuries/*/injuries.parquet"))):
        m = json.load(open(f.replace("injuries.parquet", "manifest.json")))
        d = pd.read_parquet(f, columns=["season", "week", "gsis_id", "team", "report_status", "practice_status", "game_type"])
        d = d[(d.season == 2026) & (d.game_type == "REG")].copy()
        d["fetched"] = pd.to_datetime(m["fetched_utc"], utc=True)
        rows.append(d)
    d = pd.concat(rows)
    d["team"] = d.team.map(alias)
    return d


def snapshot_firsts(d):
    sev = d.report_status.isin(["Out", "Doubtful"]) | d.practice_status.fillna("").str.contains("Did Not|Limited")
    first = d[sev].groupby(["season", "week", "gsis_id", "team"]).fetched.min().rename("snapshot_first_flag")
    fin = d[d.fetched == d.fetched.max()].drop_duplicates(["season", "week", "gsis_id"], keep="last")
    fin = fin.set_index(["season", "week", "gsis_id", "team"])[["report_status", "practice_status"]].add_prefix("final_")
    return first, fin


def inactive_firsts():
    fs = sorted(glob.glob(str(ROOT / "data/players/inactives/*/inactives.parquet")))
    i = pd.concat([pd.read_parquet(f) for f in fs])
    i["team"] = i.team.map(alias)
    i["name"] = i.player_name.str.lower().str.replace(r"[^a-z ]", "", regex=True).str.strip()
    i["captured_at_utc"] = pd.to_datetime(i.captured_at_utc, utc=True)
    return i.groupby(["season", "week", "team", "name"]).captured_at_utc.min().rename("inactive_first").reset_index()


def spread_series():
    cache = OUT / "spread_series.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    for d in sorted(glob.glob(str(ROOT / "data/market/raw/*"))):
        name = Path(d).name
        if name[:4] not in ("2024", "2025", "2026") or name.endswith("futures"):
            continue
        f = Path(d) / "quotes.parquet"
        if not f.exists():
            continue
        q = pd.read_parquet(f, columns=["observed_at_utc", "nflverse_game_id", "market", "home_spread_line"])
        q = q[(q.market == "spreads") & q.nflverse_game_id.notna()]
        if len(q):
            rows.append(q.groupby(["nflverse_game_id", "observed_at_utc"]).home_spread_line.median().reset_index())
    s = pd.concat(rows)
    s["observed_at_utc"] = pd.to_datetime(s.observed_at_utc, utc=True)
    s = s.groupby(["nflverse_game_id", "observed_at_utc"]).home_spread_line.median().reset_index()
    s.to_parquet(cache)
    return s


def first_adverse(series):
    s = series.sort_values(["nflverse_game_id", "observed_at_utc"])
    res = {}
    for gid, g in s.groupby("nflverse_game_id"):
        base = g.home_spread_line.iloc[0]
        up = g[g.home_spread_line > base]
        dn = g[g.home_spread_line < base]
        res[gid] = (g.observed_at_utc.iloc[0], up.observed_at_utc.min() if len(up) else pd.NaT, dn.observed_at_utc.min() if len(dn) else pd.NaT)
    r = pd.DataFrame.from_dict(res, orient="index", columns=["quote_first", "home_worse_ts", "home_better_ts"])
    return r.rename_axis("game_id").reset_index()


def build_table():
    s = schedule()
    tg = team_games(s)
    p = beat_rows(tg)
    keys = ["season", "week", "gsis_id", "team"]
    t = pd.concat([first_ts(p, OUT_CLASSES, "beat_first_out"), first_ts(p, LIMITED_CLASSES, "beat_first_limited"), first_ts(p, ACTIVE_CLASSES, "beat_first_active")], axis=1).reset_index()
    t = t.merge(tg[["season", "week", "team", "game_id", "opener_ts", "kickoff"]], on=["season", "week", "team"], how="left")
    t = t.merge(official().drop(columns=["team"]), on=["season", "week", "gsis_id"], how="left")
    first, fin = snapshot_firsts(snapshots())
    t = t.merge(first.reset_index(), on=keys, how="left").merge(fin.reset_index(), on=keys, how="left")
    return t, p, s


def after_move(series, game_ids, team_home, t0):
    by = {k: g for k, g in series.sort_values("observed_at_utc").groupby("nflverse_game_id")}
    res = []
    for gid, home, ts in zip(game_ids, team_home, t0):
        g = by.get(gid)
        if g is None or pd.isna(ts):
            res.append((pd.NaT, np.nan))
            continue
        before = g[g.observed_at_utc <= ts]
        if not len(before):
            res.append((pd.NaT, np.nan))
            continue
        ref = before.home_spread_line.iloc[-1]
        after = g[g.observed_at_utc > ts]
        hit = after[after.home_spread_line > ref] if home else after[after.home_spread_line < ref]
        res.append((hit.observed_at_utc.min() if len(hit) else pd.NaT, 1.0))
    return res


def lead_times(t, series):
    t = t.copy()
    t["official_ts_eff"] = t.official_ts.fillna(t.snapshot_first_flag)
    home = (t.team == t.game_id.str.split("_").str[3]).values
    out = t[t.beat_first_out.notna()].copy()
    home = (out.team == out.game_id.str.split("_").str[3]).values
    r = after_move(series, out.game_id.values, home, out.beat_first_out.tolist())
    out["market_adverse_after_beat"] = pd.to_datetime([a for a, _ in r], utc=True)
    out["has_line_at_beat"] = [b for _, b in r]
    r2 = after_move(series, out.game_id.values, home, out.official_ts_eff.tolist())
    out["market_adverse_after_official"] = pd.to_datetime([a for a, _ in r2], utc=True)
    out["has_line_at_official"] = [b for _, b in r2]
    out["lead_vs_official_h"] = (out.official_ts_eff - out.beat_first_out).dt.total_seconds() / 3600
    out["beat_to_market_h"] = (out.market_adverse_after_beat - out.beat_first_out).dt.total_seconds() / 3600
    out["official_to_market_h"] = (out.market_adverse_after_official - out.official_ts_eff).dt.total_seconds() / 3600
    out["beat_to_market_before_kick"] = out.market_adverse_after_beat < out.kickoff
    return out


def summ(x):
    x = x.dropna()
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median_h": round(float(x.median()), 2), "q25_h": round(float(x.quantile(.25)), 2), "q75_h": round(float(x.quantile(.75)), 2), "share_beat_earlier": round(float((x > 0).mean()), 3)}


def logit(p):
    return np.log(p / (1 - p))


def fit_b(off, x, y):
    b = 0.0
    for _ in range(50):
        pr = 1 / (1 + np.exp(-(off + b * x)))
        g = np.sum((y - pr) * x)
        h = -np.sum(pr * (1 - pr) * x * x)
        if h == 0:
            return 0.0
        step = g / h
        b -= step
        if abs(step) < 1e-10:
            break
    return b


def feature(p, g):
    q = p[p.cls.isin(OUT_CLASSES) & (p.created_at < p.opener_ts)]
    q = q.drop_duplicates(["game_id", "team", "gsis_id"])
    n = q.groupby(["game_id", "team"]).size().rename("n").reset_index()
    h = g[["game_id", "home_team"]].merge(n, left_on=["game_id", "home_team"], right_on=["game_id", "team"], how="left")["n"].fillna(0).values
    a = g[["game_id", "away_team"]].merge(n, left_on=["game_id", "away_team"], right_on=["game_id", "team"], how="left")["n"].fillna(0).values
    return h - a


def boot(vals, rng):
    idx = rng.integers(0, len(vals), size=(BOOT, len(vals)))
    m = vals[idx].mean(axis=1)
    return [float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m > 0).mean())]


def fitted_term(p):
    g = pd.read_parquet(PERGAME)
    sub = g[g.margin_vs_open != 0].copy().reset_index(drop=True)
    sub["x"] = feature(p, sub)
    y = sub.home_covered.astype(float).values
    pr = sub.out_of_season_home_probability.clip(1e-6, 1 - 1e-6).values
    off = logit(pr)
    x = sub.x.values
    elig = [s for s in sorted(sub.season.unique()) if s in set(p.season.unique())]
    res = {"seasons_with_beat": [int(s) for s in elig], "nonzero_x_by_season": {int(s): int(((sub.season == s) & (sub.x != 0)).sum()) for s in sorted(sub.season.unique())}}
    pooled = np.zeros(len(sub))
    ins = np.zeros(len(sub))
    folds = {}
    for s in elig:
        te = (sub.season == s).values
        tr = np.isin(sub.season.values, [e for e in elig if e != s])
        b = fit_b(off[tr], x[tr], y[tr])
        folds[int(s)] = float(b)
        pooled[te] = 1 / (1 + np.exp(-(off[te] + b * x[te])))
    mask = sub.season.isin(elig).values
    b_all = fit_b(off[mask], x[mask], y[mask])
    ins[mask] = 1 / (1 + np.exp(-(off[mask] + b_all * x[mask])))
    base = pr[mask]
    yy = y[mask]
    rng = np.random.default_rng(SEED)
    for label, arr in (("oos", pooled[mask]), ("ins", ins[mask])):
        d_brier = (base - yy) ** 2 - (arr - yy) ** 2
        d_ll = -(yy * np.log(base) + (1 - yy) * np.log(1 - base)) + (yy * np.log(arr) + (1 - yy) * np.log(1 - arr))
        d_acc = (((arr > 0.5) == (yy == 1)).astype(float) - ((base > 0.5) == (yy == 1)).astype(float)) * 100
        dec = (base > 0.5) != (arr > 0.5)
        res[label] = {"brier_gain": boot(d_brier, rng), "logloss_gain": boot(d_ll, rng), "accuracy_gain_points": boot(d_acc, rng),
                      "decisive_games": int(dec.sum()), "decisive_new_correct": int(((arr > 0.5) == (yy == 1))[dec].sum()),
                      "decisive_base_correct": int(((base > 0.5) == (yy == 1))[dec].sum()), "n": int(mask.sum())}
    res["gap_brier_gain_ins_minus_oos"] = res["ins"]["brier_gain"][0] - res["oos"]["brier_gain"][0]
    res["fold_coefficients"] = folds
    res["pooled_coefficient"] = float(b_all)
    res["x_nonzero_games_in_scored"] = int((x[mask] != 0).sum())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["table", "leads", "all"], default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t, p, s = build_table()
    ins = inactive_firsts()
    t.to_parquet(OUT / "report_to_status.parquet")
    ins.to_parquet(OUT / "inactive_first_seen.parquet")
    print("rows", len(t), "seasons", t.groupby("season").size().to_dict())
    print("beat player-posts", len(p), p.groupby("season").size().to_dict())
    print("with official_ts", int(t.official_ts.notna().sum()), "snapshot_first_flag", int(t.snapshot_first_flag.notna().sum()))
    print("inactive first rows", len(ins), ins.groupby("season").size().to_dict())
    if a.stage == "table":
        return
    series = spread_series()
    out = lead_times(t, series)
    out.to_parquet(OUT / "lead_times.parquet")
    res = {"beat_out_rows": int(len(out)), "by_season": {int(k): int(v) for k, v in out.groupby("season").size().items()}}
    res["beat_vs_official"] = summ(out.lead_vs_official_h)
    res["beat_to_adverse_move"] = summ(out.beat_to_market_h)
    res["official_to_adverse_move"] = summ(out.official_to_market_h)
    res["rows_with_line_at_beat"] = int(out.has_line_at_beat.notna().sum())
    res["rows_adverse_move_after_beat"] = int(out.market_adverse_after_beat.notna().sum())
    res["rows_with_line_at_official"] = int(out.has_line_at_official.notna().sum())
    res["rows_adverse_move_after_official"] = int(out.market_adverse_after_official.notna().sum())
    res["beat_to_adverse_by_season"] = {int(k): summ(g.beat_to_market_h) for k, g in out.groupby("season")}
    res["beat_vs_official_by_season"] = {int(k): summ(g.lead_vs_official_h) for k, g in out.groupby("season")}
    res["market_games_with_series"] = int(series.nflverse_game_id.nunique())
    json.dump(res, open(OUT / "lead_times.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    if a.stage == "leads":
        return
    r = fitted_term(p)
    json.dump(r, open(OUT / "fitted_term.json", "w"), indent=1, default=str)
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
