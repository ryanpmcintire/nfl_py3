import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forward_ledger as FL
import mod24_ftn_turnover_luck as m
import mod25_forward_q as mfq
import mod25_wf_forward as mwf
import source_pull_2026_11_16 as SP

from nfl_ats.published_picks import game_deadlines

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "prediction_market_gap"
FROZEN = OUT / "frozen.json"
EXT = ROOT / "data" / "external"
ACTIVE = ROOT / "artifacts" / "active_pick_probability.json"
LANE = "docs/lanes/prediction-market-gap-2026-11-16.md"
FIT_SEASONS = (2024, 2025)
MAP_BEFORE = 2024
SEASON = 2026
SCORED_WEEKS = (1, 10)
FIRST_WEEK = 11
LAST_WEEK = 18
WINDOW = pd.Timedelta(hours=1)
TERMS = ("tue", "deadline")
VENUES = ("kalshi", "polymarket")
SEED = 20261116
REPS = 2000
STAMP = "%Y%m%dT%H%M%SZ"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNFLGAME"
POLY_TAG = "nfl"


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit(x, y, off):
    b = 0.0
    for _ in range(100):
        p = sig(off + b * x)
        h = float((x * x * p * (1 - p)).sum())
        step = float((x * (y - p)).sum()) / h
        b += step
        if abs(step) < 1e-12:
            break
    p = sig(off + b * x)
    return b, float(1.0 / np.sqrt((x * x * p * (1 - p)).sum()))


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(d, blk, rng):
    idx = [np.flatnonzero(blk == w) for w in np.unique(blk)]
    return np.array(
        [d[np.concatenate([rng.choice(v, len(v)) for v in idx])].mean() for _ in range(REPS)]
    )


def record(pa, pb, y):
    da, db = pa > 0.5, pb > 0.5
    dec = da != db
    ya = (da == (y == 1))[dec]
    return {"decisive": int(dec.sum()), "term_wins": int(ya.sum()), "served_wins": int((~ya).sum())}


def side_record(p, y):
    ok = (p > 0.5) == (y == 1)
    return f"{int(ok.sum())}-{int((~ok).sum())}"


def reliability(p, y, bins=5):
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    return [
        {
            "mean_p": float(p[q == k].mean()),
            "hit": float(y[q == k].mean()),
            "n": int((q == k).sum()),
        }
        for k in np.unique(q)
    ]


def residuals():
    g = pd.read_parquet(
        ROOT / "data" / "processed" / "game_features.parquet",
        columns=["season", "game_type", "spread_line", "result"],
    )
    g = g[(g["season"] < MAP_BEFORE) & (g["game_type"] == "REG")].dropna()
    return np.sort((g["result"] - g["spread_line"]).to_numpy(float))


def spread_equiv(p, r):
    return -np.quantile(r, 1 - np.asarray(p, float))


def tuesday_noon(kick):
    et = kick.dt.tz_convert("America/New_York").dt.normalize()
    tue = et - pd.to_timedelta((et.dt.weekday - 1) % 7, unit="D")
    return (tue + pd.Timedelta(hours=12)).dt.tz_convert("UTC")


def stamps(g):
    g = g.copy()
    g["kickoff_utc"] = pd.to_datetime(g["kickoff"], utc=True, format="ISO8601")
    g["T_tue"] = tuesday_noon(g["kickoff_utc"])
    dl = {}
    for _, wk in g.groupby(["season", "week"]):
        dl |= game_deadlines(wk[["game_id"]].assign(kickoff=wk["kickoff_utc"]))
    g["T_deadline"] = g[["kickoff_utc"]].assign(d=g["game_id"].map(dl)).min(axis=1)
    return g


def venue_prices(home_of):
    k = pd.read_parquet(EXT / "kalshi" / "home_yes_hourly_pregame.parquet")
    k = k.dropna(subset=["price_close"])
    k = pd.DataFrame(
        {
            "game_id": k["game_id"],
            "t": pd.to_datetime(k["end_period_ts"], unit="s", utc=True),
            "p": k["price_close"].astype(float),
            "venue": "kalshi",
        }
    )
    pm = pd.read_parquet(EXT / "polymarket" / "outcome0_hourly_pregame.parquet")
    pm["code"] = pm["outcome0"].map(SP.team_from_name)
    pm["home"] = pm["game_id"].map(home_of)
    pm["away"] = pm["game_id"].str.split("_").str[2]
    pm = pm[(pm["code"] == pm["home"]) | (pm["code"] == pm["away"])]
    pm = pd.DataFrame(
        {
            "game_id": pm["game_id"],
            "t": pd.to_datetime(pm["t"], unit="s", utc=True),
            "p": np.where(pm["code"] == pm["home"], pm["p"], 1 - pm["p"]).astype(float),
            "venue": "polymarket",
        }
    )
    return pd.concat([k, pm], ignore_index=True).sort_values("t")


def price_at(prices, g, col):
    out = {}
    for v in VENUES:
        pv = prices[prices["venue"] == v]
        left = g[["game_id", col]].rename(columns={col: "t"}).dropna()
        right = pv[["game_id", "t", "p"]].rename(columns={"t": "tp"})
        j = left.merge(right, on="game_id")
        j = j[(j["tp"] <= j["t"]) & (j["tp"] > j["t"] - WINDOW)]
        out[v] = j.sort_values("tp").groupby("game_id")["p"].last()
    return out


def gaps(g, book_tue, book_dl):
    r = residuals()
    prices = venue_prices(dict(zip(g["game_id"], g["home_team"], strict=True)))
    g = g.set_index("game_id")
    res = pd.DataFrame(index=g.index)
    for term, col, book in (("tue", "T_tue", book_tue), ("deadline", "T_deadline", book_dl)):
        pv = price_at(prices, g.reset_index(), col)
        for v in VENUES:
            p = pv[v].reindex(res.index)
            eq = pd.Series(np.where(p.notna(), spread_equiv(p.fillna(0.5), r), np.nan), res.index)
            res[f"{term}_{v}"] = eq - book.reindex(res.index)
            res[f"{term}_{v}_p"] = p
        res[term] = res[[f"{term}_{v}" for v in VENUES]].mean(axis=1)
    return res


def history():
    act = json.loads(ACTIVE.read_text())
    pg = pd.read_parquet(ROOT / "artifacts" / act["artifact"] / "per_game.parquet")
    pg = pg[pg["season"].isin(FIT_SEASONS)]
    gf = pd.read_parquet(
        ROOT / "data" / "processed" / "game_features.parquet", columns=["game_id", "kickoff"]
    )
    g = stamps(pg.merge(gf, on="game_id"))
    gp = gaps(
        g,
        g.set_index("game_id")["tue_open_home_spread"],
        g.set_index("game_id")["close_home_spread"],
    )
    return g.set_index("game_id").join(gp), act


def coverage(h, label):
    out = {}
    for s, grp in h.groupby("season"):
        out[int(s)] = {"games": len(grp)} | {
            c: int(grp[c].notna().sum())
            for c in (
                "tue_kalshi",
                "tue_polymarket",
                "tue",
                "deadline_kalshi",
                "deadline_polymarket",
                "deadline",
            )
        }
    print(label, json.dumps(out))
    return out


def cmd_coverage(a):
    h, _ = history()
    cov = coverage(h, "fit seasons")
    chk = {t: {"mean_gap": float(h[t].mean()), "sd_gap": float(h[t].std())} for t in TERMS}
    print("gap distribution (no outcomes)", json.dumps(chk))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage_fit.json").write_text(json.dumps({"coverage": cov, "gap": chk}, indent=1))


def grade(x, y, off, blk, b):
    ps, pt = sig(off), sig(off + b * x)
    d = ll(ps, y) - ll(pt, y)
    bs = boot(d, blk, np.random.default_rng(SEED))
    return {
        "games": len(y),
        "served_side_record": side_record(ps, y),
        "term_side_record": side_record(pt, y),
        "changed_pick_record": record(pt, ps, y),
        "mean_log_loss": {"served": float(ll(ps, y).mean()), "term": float(ll(pt, y).mean())},
        "log_loss_gain": float(d.mean()),
        "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
        "probability_positive": float((bs > 0).mean()),
        "reliability_term": reliability(pt, y),
    }


def cmd_fit(a):
    if FROZEN.exists():
        raise SystemExit(f"already frozen: {FROZEN}")
    h, act = history()
    h = h[h["result"] != h["tue_open_home_spread"]]
    rep = {"active_artifact": act["artifact"], "terms": {}}
    frozen = {}
    for t in [*TERMS, *[f"{u}_{v}" for u in TERMS for v in VENUES]]:
        d = h.dropna(subset=[t, "out_of_season_home_probability"])
        y = (d["result"] > d["tue_open_home_spread"]).astype(float).to_numpy()
        off = logit(d["out_of_season_home_probability"].to_numpy(float))
        x = d[t].to_numpy(float)
        s = d["season"].to_numpy()
        blk = s * 100 + d["week"].to_numpy()
        b, se = fit(x, y, off)
        ins = grade(x, y, off, blk, b)
        p_oos = sig(off)
        folds = {}
        for k in FIT_SEASONS:
            tr = s != k
            if not tr.any() or tr.all():
                continue
            folds[int(k)] = fit(x[tr], y[tr], off[tr])[0]
            p_oos[~tr] = sig(off[~tr] + folds[int(k)] * x[~tr])
        dd = ll(sig(off), y) - ll(p_oos, y)
        bs = boot(dd, blk, np.random.default_rng(SEED))
        b0 = fit(np.ones(len(y)), y, off)[0]
        rep["terms"][t] = {
            "coef_logit_per_point": b,
            "se": se,
            "coef_per_held_out_season": folds,
            "in_sample": ins,
            "loso": {
                "log_loss_gain": float(dd.mean()),
                "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                "probability_positive": float((bs > 0).mean()),
                "changed_pick_record": record(p_oos, sig(off), y),
            },
            "in_minus_held_out_gain": ins["log_loss_gain"] - float(dd.mean()),
            "intercept_only_in_sample_gain": float((ll(sig(off), y) - ll(sig(off + b0), y)).mean()),
        }
        if t in TERMS:
            frozen[t] = b
        print(
            t,
            json.dumps({k: v for k, v in rep["terms"][t].items() if k != "in_sample"}),
            "in_sample",
            ins["log_loss_gain"],
            ins["changed_pick_record"],
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fit_2024_2025.json").write_text(json.dumps(rep, indent=1, default=float))
    body = {
        "family": "prediction_market_gap",
        "coefficient_logit_per_point": frozen,
        "rule": "one offset-logistic slope per term, offset logit(served out-of-season home "
        "probability), label home cover vs Tuesday opener, fitted on 2024-2025",
        "map_residual_seasons_before": MAP_BEFORE,
        "window_hours": WINDOW / pd.Timedelta(hours=1),
        "lane": LANE,
        "created_at_utc": datetime.now(UTC).strftime(STAMP),
    }
    FROZEN.write_text(json.dumps(body, indent=1))
    print(json.dumps(body, indent=1))


def season_games():
    s = pd.read_parquet(m.SERVED_FEATURES)
    s = s[(s["season"] == SEASON) & (s["game_type"] == "REG")]
    return s


def cmd_score2026(a):
    path = OUT / "score_2026_w1_10.json"
    if path.exists():
        raise SystemExit(f"already scored once: {path}")
    fz = json.loads(FROZEN.read_text())
    g = season_games()
    g = g[g["week"].between(*SCORED_WEEKS) & g["result"].notna()].copy()
    op = mwf.tuesday_open(g)
    g["open_line"] = g["game_id"].map(lambda i: op.get(i, (np.nan,))[0])
    g = stamps(g.merge(FL._played_card(), on="game_id", how="inner"))
    gp = gaps(g, g.set_index("game_id")["open_line"], g.set_index("game_id")["spread_line"])
    h = g.set_index("game_id").join(gp)
    cov = coverage(h, "2026 W1-10")
    h = h.dropna(subset=["open_line", "served_p"])
    h = h[h["result"] != h["open_line"]]
    rep = {"coverage": cov, "frozen_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest()}
    for t in TERMS:
        d = h.dropna(subset=[t])
        y = (d["result"] > d["open_line"]).astype(float).to_numpy()
        off = logit(d["served_p"].to_numpy(float))
        rep[t] = grade(
            d[t].to_numpy(float), y, off, d["week"].to_numpy(), fz["coefficient_logit_per_point"][t]
        )
        print(t, json.dumps({k: v for k, v in rep[t].items() if k != "reliability_term"}))
    path.write_text(json.dumps(rep, indent=1))


def live_prices(sched, now):
    rows = []
    start, end = int((now - WINDOW).timestamp()), int(now.timestamp())
    mk = []
    cursor = ""
    while True:
        q = urllib.parse.urlencode(
            {"series_ticker": SERIES, "status": "open", "limit": 1000, "cursor": cursor}
        )
        r = SP.fetch(f"{KALSHI}/markets?{q}", 0.2, as_json=True)
        mk += r.get("markets", [])
        cursor = r.get("cursor") or ""
        if not cursor or not r.get("markets"):
            break
    for r in sched.itertuples():
        day = r.kickoff_utc.tz_convert("America/New_York").strftime("%y%b%d").upper()
        tick = [
            x["ticker"]
            for x in mk
            if day in x["event_ticker"] and SP.code(x["ticker"].split("-")[-1]) == r.home_team
        ]
        if tick:
            q = urllib.parse.urlencode({"start_ts": start, "end_ts": end, "period_interval": 60})
            try:
                c = SP.fetch(
                    f"{KALSHI}/series/{SERIES}/markets/{tick[0]}/candlesticks?{q}",
                    0.2,
                    as_json=True,
                )
                px = [
                    (
                        k["end_period_ts"],
                        (k.get("price") or {}).get(
                            "close", (k.get("price") or {}).get("close_dollars")
                        ),
                    )
                    for k in c.get("candlesticks", [])
                    if float(k.get("volume", k.get("volume_fp", 0)) or 0) > 0
                ]
                px = [p for p in px if p[1] is not None and start < p[0] <= end]
                if px:
                    rows.append({"game_id": r.game_id, "venue": "kalshi", "p": float(max(px)[1])})
            except urllib.error.HTTPError:
                pass
    ev = []
    off = 0
    while True:
        q = urllib.parse.urlencode(
            {"tag_slug": POLY_TAG, "closed": "false", "limit": 50, "offset": off}
        )
        try:
            r = SP.fetch(f"https://gamma-api.polymarket.com/events?{q}", 0.2, as_json=True)
        except urllib.error.HTTPError:
            break
        if not r:
            break
        ev += r
        off += len(r)
    for e in ev:
        mt = SP.re.fullmatch(r"nfl-([a-z]+)-([a-z]+)-(\d{4}-\d{2}-\d{2})", e.get("slug", ""))
        if not mt:
            continue
        day = datetime.strptime(mt.group(3), "%Y-%m-%d").date()
        row = SP.match_game(
            sched.assign(gameday=sched["kickoff_utc"].dt.tz_convert("America/New_York").dt.date),
            day,
            [mt.group(1), mt.group(2)],
        )
        mkts = [x for x in e.get("markets", []) if x.get("slug") == e["slug"]]
        if row is None or not mkts:
            continue
        toks = json.loads(mkts[0].get("clobTokenIds") or "[]")
        outs = json.loads(mkts[0].get("outcomes") or "[]")
        if not toks or not outs:
            continue
        q = urllib.parse.urlencode(
            {"market": toks[0], "startTs": start, "endTs": end, "fidelity": 60}
        )
        try:
            hh = SP.fetch(f"https://clob.polymarket.com/prices-history?{q}", 0.2, as_json=True)
        except urllib.error.HTTPError:
            continue
        pts = [(p["t"], p["p"]) for p in hh.get("history", []) if start < p["t"] <= end]
        code = SP.team_from_name(outs[0])
        if pts and code in (row["home_team"], row["away_team"]):
            p0 = float(max(pts)[1])
            rows.append(
                {
                    "game_id": row["game_id"],
                    "venue": "polymarket",
                    "p": p0 if code == row["home_team"] else 1 - p0,
                }
            )
    return pd.DataFrame(rows, columns=["game_id", "venue", "p"])


def schedule(week):
    g = season_games()
    g = stamps(
        g[g["week"] == week][["season", "week", "game_id", "home_team", "away_team", "kickoff"]]
    )
    g["cutoff_utc"] = g["T_deadline"]
    return g.reset_index(drop=True)


def resolve(arg, now):
    if arg != "auto":
        return int(arg)
    g = season_games()
    k = pd.to_datetime(g["kickoff"], utc=True, format="ISO8601")
    ahead = g.loc[k > pd.Timestamp(now), "week"]
    if ahead.empty:
        raise SystemExit("no unplayed regular-season week")
    return int(ahead.min())


def book_line(sched, slot, now):
    if slot == "tue":
        op = mwf.tuesday_open(sched)
        return {k: v[0] for k, v in op.items()}
    mk = mfq.market_frame()
    mk = mk[mk["game_id"].isin(sched["game_id"]) & (mk["observed_at_utc"] <= pd.Timestamp(now))]
    last = mk["observed_at_utc"] == mk.groupby("game_id")["observed_at_utc"].transform("max")
    return mk[last].groupby("game_id")["market_line"].median().to_dict()


def cmd_pregame(a):
    now = pd.Timestamp(datetime.now(UTC))
    fz = json.loads(FROZEN.read_text())
    week = resolve(a.week, now)
    if not a.dry and not FIRST_WEEK <= week <= LAST_WEEK:
        raise SystemExit(f"week {week} outside the declared forward window")
    sched = schedule(week)
    sched = sched if a.dry else sched[sched["cutoff_utc"] > now]
    px = live_prices(sched, now)
    book = book_line(sched, a.slot, now)
    base = {}
    pub = pd.read_parquet(ROOT / "artifacts" / "clv_ledger" / "published_picks.parquet")
    pub = pub[(pub["season"] == SEASON) & (pub["week"] == week)]
    pub = pub[pd.to_datetime(pub["published_at_utc"], utc=True) <= now]
    for r in pub.sort_values("published_at_utc").groupby("game_id").tail(1).itertuples():
        base[r.game_id] = r.displayed_score if r.pick_team == r.home_team else 1 - r.displayed_score
    res = residuals()
    b = fz["coefficient_logit_per_point"][a.slot]
    games = []
    for r in sched.itertuples():
        pv = px[px["game_id"] == r.game_id]
        if pv.empty or r.game_id not in book or r.game_id not in base:
            continue
        gv = {
            v: float(spread_equiv(pv[pv["venue"] == v]["p"].iloc[0], res) - book[r.game_id])
            for v in pv["venue"].unique()
        }
        x = float(np.mean(list(gv.values())))
        p0 = float(base[r.game_id])
        p1 = float(sig(logit(p0) + b * x))
        games.append(
            {
                "game_id": r.game_id,
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "pick_deadline_utc": r.cutoff_utc.isoformat(),
                "book_line": book[r.game_id],
                "venue_gap": gv,
                "term_value": x,
                "base_home_probability": p0,
                "combined_home_probability": p1,
                "side_changed": (p0 >= 0.5) != (p1 >= 0.5),
            }
        )
    doc = {
        "season": SEASON,
        "week": week,
        "slot": a.slot,
        "created_at_utc": now.strftime(STAMP),
        "frozen_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "coefficient_logit_per_point": b,
        "prices": len(px),
        "games": games,
    }
    if a.dry:
        print(json.dumps({k: v for k, v in doc.items() if k != "games"}, default=str))
        for gm in games:
            print(
                gm["game_id"],
                gm["venue_gap"],
                round(gm["term_value"], 3),
                round(gm["base_home_probability"], 4),
                round(gm["combined_home_probability"], 4),
            )
        print("DRY: nothing written")
        return
    if not games:
        print("no game with a price, book line and published card before its cutoff")
        return
    out = OUT / f"{SEASON}_week{week:02d}_{a.slot}_{now.strftime(STAMP)}.json"
    out.write_text(json.dumps(doc, indent=1, default=str))
    print(out, len(games), "games")


def cmd_score(a):
    fz = json.loads(FROZEN.read_text())
    sha = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    rows = []
    for f in sorted(OUT.glob(f"{SEASON}_week*_*.json")):
        d = json.loads(f.read_text())
        if d["frozen_sha256"] != sha or not FIRST_WEEK <= d["week"] <= LAST_WEEK:
            continue
        mtime = pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC")
        made = pd.Timestamp(datetime.strptime(d["created_at_utc"], STAMP).replace(tzinfo=UTC))
        for gm in d["games"]:
            cut = pd.Timestamp(gm["pick_deadline_utc"])
            if made < cut and mtime < cut:
                rows.append(
                    {
                        "game_id": gm["game_id"],
                        "slot": d["slot"],
                        "week": d["week"],
                        "term_value": gm["term_value"],
                        "created": d["created_at_utc"],
                    }
                )
    if not rows:
        print("no admissible forward rows yet; nothing graded")
        return
    t = pd.DataFrame(rows).sort_values("created").drop_duplicates(["game_id", "slot"], keep="last")
    g = season_games()
    g = g[g["game_id"].isin(t["game_id"]) & g["result"].notna()].copy()
    op = mwf.tuesday_open(g)
    g["open_line"] = g["game_id"].map(lambda i: op.get(i, (np.nan,))[0])
    g = g.merge(FL._played_card(), on="game_id").dropna(subset=["open_line", "served_p"])
    g = g[g["result"] != g["open_line"]]
    rep = {}
    for slot in TERMS:
        d = g.merge(t[t["slot"] == slot][["game_id", "term_value"]], on="game_id")
        if d.empty:
            continue
        y = (d["result"] > d["open_line"]).astype(float).to_numpy()
        rep[slot] = grade(
            d["term_value"].to_numpy(float),
            y,
            logit(d["served_p"].to_numpy(float)),
            d["week"].to_numpy(),
            fz["coefficient_logit_per_point"][slot],
        )
    (OUT / "score_forward.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def cmd_argv(a):
    base = [
        ".tools/uv.exe",
        "run",
        "--no-sync",
        "python",
        "scripts/prediction_market_gap.py",
        "pregame",
    ]
    print(
        json.dumps(
            {
                "tue": {"argv": [*base, "--slot", "tue"], "slot": "Tuesday 12:30 ET"},
                "deadline": {
                    "argv": [*base, "--slot", "deadline"],
                    "slots": [
                        "Thursday 19:45 ET",
                        "Saturday 15:45 ET (when games)",
                        "Sunday 12:30 ET",
                        "Sunday 15:45 ET",
                    ],
                },
                "score": {"argv": [*base[:-1], "score"], "slot": "Tuesday 09:00 ET"},
            },
            indent=1,
        )
    )


def main():
    p = argparse.ArgumentParser(
        description=(
            "Prediction-market gap: Kalshi and Polymarket win prices mapped to points with the "
            "market residual distribution, minus the sportsbook line, as one fitted term on the "
            "served card probability. Research only; serves nothing."
        )
    )
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("coverage").set_defaults(fn=cmd_coverage)
    sp.add_parser("fit").set_defaults(fn=cmd_fit)
    sp.add_parser("score2026").set_defaults(fn=cmd_score2026)
    pg = sp.add_parser("pregame")
    pg.add_argument("--week", default="auto")
    pg.add_argument("--slot", choices=TERMS, required=True)
    pg.add_argument("--dry", action="store_true")
    pg.set_defaults(fn=cmd_pregame)
    sp.add_parser("score").set_defaults(fn=cmd_score)
    sp.add_parser("argv").set_defaults(fn=cmd_argv)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
