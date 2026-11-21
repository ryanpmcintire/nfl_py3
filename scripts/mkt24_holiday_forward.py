import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mkt24_slot_residual as H

from nfl_ats.clv import live_tuesday_openers
from nfl_ats.published_picks import game_deadlines

ROOT = H.ROOT
SEASON = 2026
OUT = ROOT / "artifacts" / "mkt24" / "forward_2026"
DECLARATION = OUT / "declaration.json"
SCORE = OUT / "score.json"
RELIABILITY = H.OUT / "reliability.json"
MARKET = ROOT / "data" / "market" / "raw"
SPLASH = ROOT / "data" / "splash"
PUBLISHED = ROOT / "artifacts" / "clv_ledger" / "published_picks.parquet"
LANE = "docs/lanes/mkt-24-slot-opener-residual.md"
STAMP = "%Y%m%dT%H%M%SZ"
HOLIDAY = "holiday"
DECLARED = "declared_holiday"
BLACK_FRIDAY = "descriptive_black_friday"
OPENER_HALF = "half1"
POWER = 0.8


def now_utc():
    return pd.Timestamp(datetime.now(UTC))


def iso(x):
    return None if x is None or pd.isna(x) else pd.Timestamp(x).isoformat()


def num(x):
    return None if x is None or pd.isna(x) else float(x)


def schedule():
    g = pd.read_parquet(
        H.FEATURES,
        columns=[
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            "gameday",
            "weekday",
            "kickoff",
            "location",
            "game_type",
            "result",
        ],
    )
    g = g[(g["season"] == SEASON) & (g["game_type"] == "REG")].copy()
    g["kickoff_utc"] = pd.to_datetime(g["kickoff"], utc=True, format="ISO8601")
    dl = {}
    for _, x in g.groupby("week"):
        dl.update(game_deadlines(x[["game_id"]].assign(kickoff=x["kickoff_utc"])))
    g["pick_deadline_utc"] = g["game_id"].map(dl)
    g["cutoff_utc"] = g[["kickoff_utc", "pick_deadline_utc"]].min(axis=1)
    day = pd.to_datetime(g["gameday"])
    hol = H.holiday_mask(day) & g["location"].ne("Neutral")
    after = (day[hol & day.dt.month.eq(11)] + pd.Timedelta(days=1)).unique()
    g["arm"] = np.where(hol, DECLARED, np.where(day.isin(after), BLACK_FRIDAY, ""))
    return g.reset_index(drop=True)


def holiday_prior():
    rel = json.loads(RELIABILITY.read_text())
    pooled = next(s for s in rel["pooled_slots"] if s["slot"] == HOLIDAY)
    era = next(s for s in rel["splits"]["era"]["slots"] if s["slot"] == HOLIDAY)
    opener = era[OPENER_HALF]
    covers, fails = (int(v) for v in opener["cover_record"].split("-"))
    return {
        "source": str(RELIABILITY.relative_to(ROOT)),
        "residual_sd_points": rel["residual_sd_points"],
        "pooled_2011_2025": {
            "games": pooled["n"],
            "mean_residual_points": pooled["mean"],
            "home_record": f"{pooled['home_covers']}-{pooled['home_fails']}",
            "home_cover_rate": pooled["home_covers"]
            / (pooled["home_covers"] + pooled["home_fails"]),
            "exact_binomial_p": pooled["binom_p"],
        },
        "tuesday_opener_2020_2025": {
            "games": opener["n"],
            "mean_residual_points": opener["mean"]["value"],
            "block_interval": [opener["mean"]["lo"], opener["mean"]["hi"]],
            "probability_positive": opener["mean"]["ppos"],
            "home_record": opener["cover_record"],
            "home_cover_rate": covers / (covers + fails),
            "exact_binomial_p": opener["binom_p"],
        },
    }


def detectable(n, sd, prior_mean, prior_rate):
    a = H.ALPHA_LEVEL
    se = sd / np.sqrt(n)
    za, zp = norm.ppf(1 - a), norm.ppf(POWER)
    k = int(binom.ppf(a, n, H.EVEN))
    k = k if binom.cdf(k, n, H.EVEN) <= a else k - 1
    out = {
        "games": n,
        "alpha_one_sided": a,
        "power_target": POWER,
        "mean_residual_se_points": se,
        "mean_residual_needed_for_alpha_points": -za * se,
        "mean_residual_detectable_at_power_points": -(za + zp) * se,
        "power_at_prior_mean": float(norm.cdf(abs(prior_mean) / se - za)),
        "smallest_exact_binomial_p": H.EVEN**n,
        "home_covers_needed_for_alpha": k if k >= 0 else None,
    }
    if k >= 0:
        out["binomial_power_at_prior_rate"] = float(binom.cdf(k, n, prior_rate))
        out["home_cover_rate_detectable_at_power"] = float(
            brentq(lambda p: binom.cdf(k, n, p) - POWER, 1e-9, H.EVEN)
        )
    else:
        out["binomial_power_at_prior_rate"] = 0.0
        out["home_cover_rate_detectable_at_power"] = None
    return out


def cmd_declare(a):
    if DECLARATION.exists() and not a.replace:
        raise SystemExit(f"{DECLARATION} exists; the declaration is written once")
    now = now_utc()
    g = schedule()
    dec = g[g["arm"].eq(DECLARED)].sort_values("kickoff_utc")
    if (dec["cutoff_utc"] <= now).any():
        raise SystemExit("a declared game is already past its pick deadline")
    prior = holiday_prior()
    sd = prior["residual_sd_points"]
    pm = prior["pooled_2011_2025"]["mean_residual_points"]
    pr = prior["pooled_2011_2025"]["home_cover_rate"]
    first = dec[dec["week"].eq(dec["week"].min())]
    games = [
        {
            "game_id": r.game_id,
            "week": int(r.week),
            "arm": r.arm,
            "matchup": f"{r.away_team} at {r.home_team}",
            "kickoff_utc": iso(r.kickoff_utc),
            "pick_deadline_utc": iso(r.cutoff_utc),
        }
        for r in g[g["arm"].ne("")].sort_values("kickoff_utc").itertuples()
    ]
    d = {
        "declared_utc": now.isoformat(),
        "lane": LANE,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "family": H.FAMILY,
        "history_looks": H.UNIT2_LOOKS,
        "looks_after_this": H.UNIT2_LOOKS + 1,
        "slot_rule": "mkt24_slot_residual.holiday_mask: Thanksgiving Thursday and 25 December, "
        "not neutral site",
        "games": games,
        "declared_game_ids": dec["game_id"].tolist(),
        "quantity": "residual = final home margin minus the Tuesday multi-book opener home line "
        "(nflverse sign), logged before each game's pick deadline",
        "test": "home cover record at the Tuesday opener over the declared games, pushes reported "
        "and dropped, exact binomial one-sided low against one half",
        "graded_when": "once, after the last declared game settles; Week 12 rows are logged and "
        "not graded alone",
        "prior": prior,
        "detectable": {
            "week12_only": detectable(len(first), sd, pm, pr),
            "all_declared": detectable(len(dec), sd, pm, pr),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    DECLARATION.write_text(json.dumps(d, indent=2))
    print(json.dumps({k: d[k] for k in ("declared_game_ids", "detectable")}, indent=2))


def openers():
    o = live_tuesday_openers(MARKET)
    o = o.sort_values(["opener_books", "opener_observed_at_utc"], ascending=[False, True])
    return o.drop_duplicates("game_id").set_index("game_id")


def pool_lines(week, now):
    lines, src = {}, None
    for f in sorted(SPLASH.glob(f"{SEASON}_week{week:02d}_*.json")):
        j = json.loads(f.read_text())
        at = pd.Timestamp(j["captured_at_et"]).tz_convert("UTC")
        if at <= now:
            lines = {x["game_id"]: float(x["home_spread"]) for x in j["games"]}
            src = {"file": f.name, "captured_utc": at.isoformat()}
    return lines, src


def served(week, now):
    p = pd.read_parquet(PUBLISHED)
    p = p[(p["season"] == SEASON) & (p["week"] == week)]
    p = p[pd.to_datetime(p["published_at_utc"], utc=True) <= now]
    return p.sort_values("published_at_utc").groupby("game_id").tail(1).set_index("game_id")


def current_week(g, now):
    ahead = g.loc[g["kickoff_utc"] > now, "week"]
    if ahead.empty:
        raise SystemExit("no unplayed regular-season week")
    return int(ahead.min())


def cmd_log(a):
    now = now_utc()
    if not DECLARATION.exists():
        raise SystemExit("run declare first")
    g = schedule()
    week = current_week(g, now) if a.week == "auto" else int(a.week)
    g = g[g["week"].eq(week) & g["arm"].ne("")]
    if g.empty:
        print(f"week {week}: no declared holiday or Black Friday game; nothing logged")
        return
    op = openers()
    pool, src = pool_lines(week, now)
    sv = served(week, now)
    rows, skipped = [], []
    for r in g.sort_values("kickoff_utc").itertuples():
        if now >= r.cutoff_utc:
            skipped.append(r.game_id)
            continue
        o = op.loc[r.game_id] if r.game_id in op.index else None
        s = sv.loc[r.game_id] if r.game_id in sv.index else None
        p_pick = None if s is None else float(s["displayed_score"])
        p_home = None if s is None else (p_pick if s["pick_team"] == r.home_team else 1 - p_pick)
        rows.append(
            {
                "game_id": r.game_id,
                "arm": r.arm,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "kickoff_utc": iso(r.kickoff_utc),
                "cutoff_utc": iso(r.cutoff_utc),
                "pool_home_line": pool.get(r.game_id),
                "opener_home_line": None if o is None else num(o["tue_open_home_spread"]),
                "opener_books": None if o is None else int(o["opener_books"]),
                "opener_observed_utc": None if o is None else iso(o["opener_observed_at_utc"]),
                "opener_basis": None if o is None else str(o["opener_basis"]),
                "served_side": None if s is None else str(s["pick_team"]),
                "served_side_probability": p_pick,
                "served_home_probability": p_home,
                "served_market_spread": None if s is None else num(s["market_spread"]),
                "served_published_utc": None if s is None else iso(s["published_at_utc"]),
            }
        )
    out = OUT / f"{SEASON}_week{week:02d}_{now.strftime(STAMP)}.json"
    body = {
        "logged_utc": now.isoformat(),
        "week": week,
        "line_sign": "home line positive means home favored (nflverse); "
        "residual = result minus line",
        "pool_source": src,
        "rows": rows,
        "skipped_past_deadline": skipped,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2))
    print(out.relative_to(ROOT))
    for x in rows:
        print(
            f"{x['arm']:<26}{x['away_team']:>4} at {x['home_team']:<4} "
            f"pool {x['pool_home_line']} opener {x['opener_home_line']} "
            f"served {x['served_side']} {x['served_side_probability']}"
        )
    if skipped:
        print("past deadline, not logged:", ", ".join(skipped))


def latest_logs():
    rows = []
    for f in sorted(OUT.glob(f"{SEASON}_week*_*.json")):
        j = json.loads(f.read_text())
        at = pd.Timestamp(j["logged_utc"])
        mtime = pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC")
        for r in j["rows"]:
            cut = pd.Timestamp(r["cutoff_utc"])
            if at < cut and mtime < cut:
                rows.append({**r, "logged_utc": at, "file": f.name})
    if not rows:
        raise SystemExit("no pre-deadline log rows")
    d = pd.DataFrame(rows).sort_values("logged_utc")
    return d.groupby("game_id").tail(1).set_index("game_id")


def record(won, lost):
    return {"won": int(won.sum()), "lost": int(lost.sum())}


def grade(d, sd):
    t = d.dropna(subset=["opener_home_line"]).copy()
    t["resid"] = t["result"] - t["opener_home_line"]
    covers, fails = int((t["resid"] > 0).sum()), int((t["resid"] < 0).sum())
    n = len(t)
    se = sd / np.sqrt(n) if n else np.nan
    mean = float(t["resid"].mean()) if n else np.nan
    z = norm.ppf(1 - H.ALPHA_LEVEL / 2)
    home = t["served_side"].eq(t["home_team"])
    s_open = np.where(home, t["resid"], -t["resid"])
    pool = t.dropna(subset=["pool_home_line"])
    pr = pool["result"] - pool["pool_home_line"]
    s_pool = np.where(pool["served_side"].eq(pool["home_team"]), pr, -pr)
    return {
        "games_with_pre_deadline_opener": n,
        "games_without": sorted(set(d.index) - set(t.index)),
        "home_record_at_opener": f"{covers}-{fails}",
        "pushes_at_opener": int((t["resid"] == 0).sum()),
        "exact_binomial_one_sided_low_p": (
            float(binom.cdf(covers, covers + fails, H.EVEN)) if covers + fails else None
        ),
        "mean_residual_points": mean,
        "interval_points_history_sd": [mean - z * se, mean + z * se],
        "probability_positive": float(norm.cdf(mean / se)) if n else None,
        "served_record_at_opener": record(s_open > 0, s_open < 0),
        "served_record_at_pool_line": record(s_pool > 0, s_pool < 0),
        "per_game": t.reset_index()[
            [
                "game_id",
                "result",
                "opener_home_line",
                "pool_home_line",
                "resid",
                "served_side",
                "served_side_probability",
                "logged_utc",
            ]
        ]
        .astype({"logged_utc": str})
        .to_dict("records"),
    }


def cmd_score(a):
    dec = json.loads(DECLARATION.read_text())
    g = schedule().set_index("game_id")
    logs = latest_logs()
    ids = dec["declared_game_ids"]
    open_games = [i for i in ids if pd.isna(g.loc[i, "result"])]
    if open_games:
        raise SystemExit(f"declared games not settled: {', '.join(open_games)}")
    logs = logs.join(g[["result"]])
    sd = dec["prior"]["residual_sd_points"]
    res = grade(logs.loc[logs.index.intersection(ids)], sd)
    res["declared_not_logged_before_deadline"] = sorted(set(ids) - set(logs.index))
    prior = dec["prior"]["tuesday_opener_2020_2025"]
    res["sign_agrees_with_history"] = bool(res["mean_residual_points"] < 0) == bool(
        prior["mean_residual_points"] < 0
    )
    bf = logs[logs["arm"].eq(BLACK_FRIDAY) & logs["result"].notna()]
    body = {
        "scored_utc": now_utc().isoformat(),
        "declaration": str(DECLARATION.relative_to(ROOT)),
        "declared": res,
        "prior": dec["prior"],
        "descriptive_black_friday": grade(bf, sd) if len(bf) else None,
    }
    SCORE.write_text(json.dumps(body, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "per_game"}, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser(
        description="MKT-24 forward look at 2026 holiday games: declare once, log lines before "
        "each pick deadline, score once after the last declared game. Research only."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("declare", help="write the declaration once, before any declared game")
    d.add_argument("--replace", action="store_true", help="rewrite before any declared deadline")
    d.set_defaults(func=cmd_declare)
    lg = sub.add_parser("log", help="snapshot pool line, opener and served side for this week")
    lg.add_argument("--week", default="auto", help="week number or auto")
    lg.set_defaults(func=cmd_log)
    sc = sub.add_parser("score", help="grade the one declared look after the games settle")
    sc.set_defaults(func=cmd_score)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
