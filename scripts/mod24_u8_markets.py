import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "prediction_markets"
PROC = ROOT / "data" / "processed"
KAL = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
CODE_FIX = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "LAS": "LV", "OAK": "LV", "SD": "LAC", "STL": "LA"}
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "nfl-ats-research-readonly/1.0"
GAP = {"kalshi": 0.12, "poly": 0.15}


def team(code):
    code = code.upper()
    return CODE_FIX.get(code, code)


def get(url, params=None, kind="kalshi", tries=6):
    for i in range(tries):
        time.sleep(GAP[kind])
        try:
            r = SESSION.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 * (i + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(3 * (i + 1))
            continue
        if r.status_code == 404:
            return None
        if r.ok:
            return r.json()
        return {"_error": r.status_code, "_text": r.text[:200]}
    return {"_error": "retries"}


def latest_schedule():
    runs = sorted(p for p in (ROOT / "data" / "raw").iterdir() if (p / "schedules.parquet").is_file())
    s = pd.read_parquet(runs[-1] / "schedules.parquet")
    s = s[s.game_type == "REG"].copy()
    gf = pd.read_parquet(PROC / "game_features.parquet", columns=["game_id", "kickoff"])
    s = s.merge(gf, on="game_id", how="left")
    s["gameday"] = pd.to_datetime(s["gameday"])
    miss = s["kickoff"].isna()
    if miss.any():
        loc = s.loc[miss, "gameday"].dt.strftime("%Y-%m-%d") + " " + s.loc[miss, "gametime"].fillna("13:00")
        s.loc[miss, "kickoff"] = [
            pd.Timestamp(dt.datetime.strptime(x, "%Y-%m-%d %H:%M").replace(tzinfo=ET)).tz_convert("UTC") for x in loc
        ]
    s["kickoff"] = pd.to_datetime(s["kickoff"], utc=True)
    s["gameday"] = s["kickoff"].dt.tz_convert(ET).dt.tz_localize(None).dt.normalize()
    return s[["game_id", "season", "week", "home_team", "away_team", "gameday", "kickoff", "spread_line"]].reset_index(drop=True), runs[-1].name


def build_lookup(sched):
    lk = {}
    for r in sched.itertuples():
        lk.setdefault((team(r.home_team), team(r.away_team)), []).append(r)
    return lk


def match_game(lk, home, away, day):
    best = None
    for r in lk.get((team(home), team(away)), []):
        d = abs((r.gameday.date() - day).days)
        if d <= 1 and (best is None or d < best[0]):
            best = (d, r)
    return best[1] if best else None


KNOWN = set("ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX JAC KC LA LAR LAC LV LAS MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS WSH".split())


def parse_kal_event(ev):
    m = re.match(r"^KX[A-Z]+-(\d\d)([A-Z]{3})(\d\d)([A-Z]+)$", ev)
    if not m:
        return None
    yy, mon, dd, codes = m.groups()
    splits = [(codes[:i], codes[i:]) for i in range(2, len(codes) - 1) if codes[:i] in KNOWN and codes[i:] in KNOWN]
    if len(splits) != 1:
        return None
    return dt.date(2000 + int(yy), MONTHS[mon], int(dd)), splits[0][0], splits[0][1]


def freeze_points(kick):
    k = kick.tz_convert(ET)
    d = k.normalize()
    days_back = (d.weekday() - 1) % 7
    if days_back == 0:
        days_back = 7
    tue = (d - pd.Timedelta(days=days_back)).tz_localize(None)
    tue_noon = pd.Timestamp(dt.datetime(tue.year, tue.month, tue.day, 12), tz=ET)
    sun = tue + pd.Timedelta(days=5)
    sun4 = pd.Timestamp(dt.datetime(sun.year, sun.month, sun.day, 16), tz=ET)
    deadline = min(kick, sun4.tz_convert("UTC"))
    return tue_noon.tz_convert("UTC"), deadline, kick


def iso_ts(x):
    return int(dt.datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp())


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_manifest(run_dir, extra):
    files = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files[str(p.relative_to(run_dir)).replace("\\", "/")] = {"sha256": sha(p), "bytes": p.stat().st_size}
    man = {"run_id": run_dir.name, "created_utc": dt.datetime.now(UTC).isoformat(), "files": files, **extra}
    (run_dir / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")


def kalshi_list(series):
    out = {}
    for path in ("/historical/markets", "/markets"):
        cur = None
        while True:
            p = {"series_ticker": series, "limit": 1000}
            if cur:
                p["cursor"] = cur
            r = get(KAL + path, p)
            if not r or "markets" not in r:
                break
            for m in r["markets"]:
                out[m["ticker"]] = m
            cur = r.get("cursor")
            if not cur or not r["markets"]:
                break
    return out


def kal_candles(ticker, series, start, end, period, hist):
    path = f"/historical/markets/{ticker}/candlesticks" if hist else f"/series/{series}/markets/{ticker}/candlesticks"
    r = get(KAL + path, {"start_ts": int(start), "end_ts": int(end), "period_interval": period})
    if r is None and hist:
        r = get(KAL + f"/series/{series}/markets/{ticker}/candlesticks", {"start_ts": int(start), "end_ts": int(end), "period_interval": period})
    if r is None and not hist:
        r = get(KAL + f"/historical/markets/{ticker}/candlesticks", {"start_ts": int(start), "end_ts": int(end), "period_interval": period})
    return (r or {}).get("candlesticks", [])


def cd(v, key):
    if v is None:
        return np.nan
    x = v.get(key + "_dollars", v.get(key))
    return fnum(x)


def flatten_kal(c):
    p = c.get("price") or {}
    yb = c.get("yes_bid") or {}
    ya = c.get("yes_ask") or {}
    return {
        "ts": c["end_period_ts"],
        "last": cd(p, "close"),
        "bid": cd(yb, "close"),
        "ask": cd(ya, "close"),
        "vol": fnum(c.get("volume_fp", c.get("volume"))),
        "oi": fnum(c.get("open_interest_fp", c.get("open_interest"))),
    }


def backfill_kalshi(run_dir, sched, lk, max_events=None):
    rows = []
    unmatched = []
    kdir = run_dir / "kalshi"
    for series in ("KXNFLGAME", "KXNFLSPREAD"):
        f = kdir / f"markets_{series}.json"
        if f.is_file():
            ms = json.loads(f.read_text(encoding="utf-8"))
        else:
            ms = kalshi_list(series)
            save_json(f, ms)
        print(series, "markets listed", len(ms), flush=True)
        seen_events = {}
        for t, m in ms.items():
            pe = parse_kal_event(m["event_ticker"])
            if pe is None:
                continue
            day, away, home = pe
            g = match_game(lk, home, away, day)
            if g is None:
                unmatched.append(m["event_ticker"])
                continue
            seen_events[m["event_ticker"]] = g
        todo = []
        for t, m in ms.items():
            g = seen_events.get(m["event_ticker"])
            if g is None:
                continue
            vol = fnum(m.get("volume_fp", m.get("volume")))
            if series == "KXNFLSPREAD" and not (vol > 0):
                continue
            todo.append((t, m, g))
        print(series, "markets to fetch", len(todo), flush=True)
        for i, (t, m, g) in enumerate(todo):
            kick = g.kickoff.timestamp()
            op = iso_ts(m["open_time"])
            start = max(op, kick - 21 * 86400)
            end = kick + 3 * 3600
            hist = m.get("status") in ("finalized", "settled", "determined") and iso_ts(m["close_time"]) < time.time() - 7 * 86400
            cache = kdir / "candles60" / f"{t}.json"
            if cache.is_file():
                cs = json.loads(cache.read_text(encoding="utf-8"))
            else:
                cs = kal_candles(t, series, start, end, 60, hist)
                save_json(cache, cs)
            fine = []
            if series == "KXNFLGAME":
                c1 = kdir / "candles1" / f"{t}.json"
                if c1.is_file():
                    fine = json.loads(c1.read_text(encoding="utf-8"))
                else:
                    fine = kal_candles(t, series, max(op, kick - 4 * 3600), kick, 1, hist)
                    save_json(c1, fine)
            suffix = t.rsplit("-", 1)[1]
            if series == "KXNFLGAME":
                tm = team(suffix)
                strike = np.nan
                mtype = "moneyline"
            else:
                mm = re.match(r"^([A-Z]+)\d+$", suffix)
                tm = team(mm.group(1)) if mm else suffix
                strike = fnum(m.get("floor_strike"))
                mtype = "spread"
            for res, cl in (("h60", cs), ("m1", fine)):
                for c in cl:
                    d = flatten_kal(c)
                    rows.append({"source": "kalshi", "game_id": g.game_id, "market_type": mtype, "market_id": t, "team": tm, "strike": strike, "res": res, **d})
            if i % 200 == 0:
                print(series, i, len(todo), flush=True)
    return rows, unmatched


def poly_events(run_dir):
    f = run_dir / "poly" / "events_nfl.json"
    if f.is_file():
        return json.loads(f.read_text(encoding="utf-8"))
    ev = {}
    off = 0
    while True:
        j = get(f"{GAMMA}/events", {"tag_slug": "nfl", "limit": 500, "offset": off, "order": "startDate", "ascending": "true"}, "poly")
        if not isinstance(j, list) or not j:
            break
        for e in j:
            ev[e["id"]] = e
        off += len(j)
    save_json(f, ev)
    return ev


def poly_hist(token, start, end, fid):
    j = get(f"{CLOB}/prices-history", {"market": token, "startTs": int(start), "endTs": int(end), "fidelity": fid}, "poly")
    return (j or {}).get("history", []) if isinstance(j, dict) else []


def backfill_poly(run_dir, sched, lk):
    ev = poly_events(run_dir)
    rows = []
    unmatched = []
    games = [e for e in ev.values() if re.match(r"^nfl-[a-z]{2,4}-[a-z]{2,4}-\d{4}-\d\d-\d\d$", e["slug"])]
    print("poly game events", len(games), flush=True)
    for i, e in enumerate(games):
        mm = re.match(r"^nfl-([a-z]{2,4})-([a-z]{2,4})-(\d{4})-(\d\d)-(\d\d)$", e["slug"])
        away, home = mm.group(1), mm.group(2)
        day = dt.date(int(mm.group(3)), int(mm.group(4)), int(mm.group(5)))
        g = match_game(lk, home, away, day)
        if g is None:
            unmatched.append(e["slug"])
            continue
        kick = g.kickoff.timestamp()
        ml = [m for m in e["markets"] if m.get("sportsMarketType") == "moneyline"]
        if not ml:
            continue
        names = json.loads(ml[0]["outcomes"])
        name_to_code = {names[0]: team(away), names[1]: team(home)}
        for m in e["markets"]:
            st = m.get("sportsMarketType")
            if st not in ("moneyline", "spreads"):
                continue
            if st == "spreads" and not (fnum(m.get("volumeNum")) > 0):
                continue
            toks = json.loads(m["clobTokenIds"])
            outs = json.loads(m["outcomes"])
            start = kick - 14 * 86400
            end = kick + 2 * 3600
            cache = run_dir / "poly" / "hist" / f"{m['id']}.json"
            if cache.is_file():
                h = json.loads(cache.read_text(encoding="utf-8"))
            else:
                h = poly_hist(toks[0], start, end, 5)
                save_json(cache, h)
            tm = name_to_code.get(outs[0], outs[0])
            if st == "moneyline":
                strike = np.nan
                mtype = "moneyline"
            else:
                q = re.search(r"\(([+-]?\d+(?:\.\d+)?)\)", m["question"])
                line = float(q.group(1)) if q else np.nan
                strike = -line
                mtype = "spread"
            for c in h:
                rows.append({"source": "poly", "game_id": g.game_id, "market_type": mtype, "market_id": str(m["id"]), "team": tm, "strike": strike, "res": "m5", "ts": c["t"], "last": c["p"], "bid": np.nan, "ask": np.nan, "vol": np.nan, "oi": np.nan, "mkt_volume": fnum(m.get("volumeNum"))})
        if i % 50 == 0:
            print("poly", i, len(games), flush=True)
    return rows, unmatched


def mid_of(r):
    if np.isfinite(r["bid"]) and np.isfinite(r["ask"]) and r["ask"] >= r["bid"] and (r["ask"] - r["bid"]) <= 0.10 and r["ask"] > 0:
        return (r["bid"] + r["ask"]) / 2, r["ask"] - r["bid"]
    return r["last"], np.nan


def series_at(df, t):
    d = df[df["ts"] <= t]
    if d.empty:
        return None
    return d


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def implied_spread(points):
    pts = sorted(points)
    if len(pts) < 2:
        return np.nan
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    ys = np.minimum.accumulate(ys)
    for i in range(len(xs) - 1):
        if ys[i] >= 0.5 >= ys[i + 1] and ys[i] > ys[i + 1]:
            a, b = logit(ys[i]), logit(ys[i + 1])
            x = xs[i] + (0 - a) / (b - a) * (xs[i + 1] - xs[i])
            return -x
    return np.nan


def derive(series_df, sched):
    out = []
    smap = sched.set_index("game_id")
    for (source, gid), gdf in series_df.groupby(["source", "game_id"]):
        g = smap.loc[gid]
        home, away = team(g.home_team), team(g.away_team)
        tue, dead, kick = freeze_points(g.kickoff)
        pts = {"freeze": tue.timestamp(), "deadline": dead.timestamp(), "kickoff": kick.timestamp()}
        for label, T in pts.items():
            eff = T
            ml = gdf[gdf.market_type == "moneyline"]
            probs = []
            liq = np.nan
            lastts = np.nan
            sp = np.nan
            for mid_id, mdf in ml.groupby("market_id"):
                if label == "kickoff" and source == "kalshi" and (mdf.res == "m1").any():
                    cand = mdf[(mdf.res == "m1") & mdf["last"].notna()]
                    cand = cand[cand.ts <= T]
                    cand = cand if not cand.empty else mdf[(mdf.res != "m1") & mdf["last"].notna() & (mdf.ts <= T)]
                    if cand.empty:
                        continue
                    r = cand.sort_values("ts").iloc[-1]
                    p = r["last"]
                    spr = np.nan
                else:
                    d = mdf[(mdf.res != "m1") & (mdf.ts <= T)]
                    if d.empty:
                        continue
                    r = d.sort_values("ts").iloc[-1]
                    p, spr = mid_of(r)
                if not np.isfinite(p):
                    continue
                tm = r["team"]
                probs.append(p if tm == home else 1 - p if tm == away else np.nan)
                v24 = mdf[(mdf.res != "m1") & (mdf.ts <= T) & (mdf.ts > T - 86400)]["vol"].sum()
                liq = np.nansum([0 if not np.isfinite(liq) else liq, v24])
                lastts = max(lastts, r["ts"]) if np.isfinite(lastts) else r["ts"]
                sp = spr
            probs = [x for x in probs if np.isfinite(x)]
            ph = float(np.mean(probs)) if probs else np.nan
            ladder = []
            sl = gdf[gdf.market_type == "spread"]
            nvol = 0.0
            for mid_id, mdf in sl.groupby("market_id"):
                d = mdf[(mdf.res != "m1") & (mdf.ts <= T)]
                if d.empty:
                    continue
                r = d.sort_values("ts").iloc[-1]
                if T - r["ts"] > 3 * 86400:
                    continue
                p, _ = mid_of(r)
                if not np.isfinite(p):
                    continue
                k = r["strike"]
                if r["team"] == home:
                    ladder.append((k, p))
                elif r["team"] == away:
                    ladder.append((-k, 1 - p))
                nvol += mdf[(mdf.ts <= T) & (mdf.ts > T - 86400)]["vol"].sum()
            out.append({"source": source, "game_id": gid, "season": int(g.season), "week": int(g.week), "point": label, "point_utc": pd.Timestamp(T, unit="s", tz="UTC"), "home_win_prob": ph, "ml_markets": len(probs), "ml_last_ts": lastts, "ml_bidask": sp, "ml_vol_24h": liq, "implied_home_spread": implied_spread(ladder), "ladder_n": len(ladder), "spread_vol_24h": nvol})
    return pd.DataFrame(out)


def cmd_backfill(a):
    sched, sched_run = latest_schedule()
    lk = build_lookup(sched)
    run_id = a.run_id or dt.datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RAW / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows, un = [], {}
    if "kalshi" in a.sources:
        r, u = backfill_kalshi(run_dir, sched, lk)
        rows += r
        un["kalshi"] = sorted(set(u))
    if "poly" in a.sources:
        r, u = backfill_poly(run_dir, sched, lk)
        rows += r
        un["poly"] = sorted(set(u))
    df = pd.DataFrame(rows)
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROC / "prediction_markets_series.parquet", index=False)
    pts = derive(df, sched)
    pts.to_parquet(PROC / "prediction_markets.parquet", index=False)
    write_manifest(run_dir, {"schedule_run": sched_run, "unmatched": {k: len(v) for k, v in un.items()}, "unmatched_ids": un, "series_rows": len(df)})
    print("done", run_id, len(df), len(pts), {k: len(v) for k, v in un.items()})


def cmd_derive(a):
    sched, _ = latest_schedule()
    df = pd.read_parquet(PROC / "prediction_markets_series.parquet")
    derive(df, sched).to_parquet(PROC / "prediction_markets.parquet", index=False)


def cmd_capture(a):
    run_id = dt.datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-capture"
    run_dir = RAW / run_id
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for series in ("KXNFLGAME", "KXNFLSPREAD"):
        ms = []
        cur = None
        while True:
            p = {"series_ticker": series, "status": "open", "limit": 1000}
            if cur:
                p["cursor"] = cur
            r = get(KAL + "/markets", p)
            if not r or "markets" not in r:
                break
            ms += r["markets"]
            cur = r.get("cursor")
            if not cur or not r["markets"]:
                break
        save_json(run_dir / f"kalshi_{series}.json", ms)
        for m in ms:
            rows.append({"captured_utc": now, "source": "kalshi", "series": series, "market_id": m["ticker"], "event": m["event_ticker"], "title": m.get("title"), "strike": fnum(m.get("floor_strike")), "bid": fnum(m.get("yes_bid_dollars")), "ask": fnum(m.get("yes_ask_dollars")), "last": fnum(m.get("last_price_dollars")), "volume": fnum(m.get("volume_fp")), "open_interest": fnum(m.get("open_interest_fp")), "close_time": m.get("close_time")})
    ev = []
    off = 0
    while True:
        j = get(f"{GAMMA}/events", {"tag_slug": "nfl", "active": "true", "closed": "false", "limit": 500, "offset": off}, "poly")
        if not isinstance(j, list) or not j:
            break
        ev += j
        off += len(j)
    ev = [e for e in ev if re.match(r"^nfl-[a-z]{2,4}-[a-z]{2,4}-\d{4}-\d\d-\d\d$", e["slug"])]
    save_json(run_dir / "poly_events.json", ev)
    for e in ev:
        for m in e["markets"]:
            if m.get("sportsMarketType") not in ("moneyline", "spreads"):
                continue
            pr = json.loads(m["outcomePrices"]) if m.get("outcomePrices") else [None, None]
            rows.append({"captured_utc": now, "source": "poly", "series": m.get("sportsMarketType"), "market_id": str(m["id"]), "event": e["slug"], "title": m["question"], "strike": fnum(m.get("line")), "bid": fnum(m.get("bestBid")), "ask": fnum(m.get("bestAsk")), "last": fnum(pr[0]), "volume": fnum(m.get("volumeNum")), "open_interest": fnum(m.get("liquidityNum")), "close_time": m.get("endDate")})
    snap = pd.DataFrame(rows)
    f = PROC / "prediction_markets_snapshots.parquet"
    if f.is_file():
        snap = pd.concat([pd.read_parquet(f), snap], ignore_index=True)
    snap.to_parquet(f, index=False)
    write_manifest(run_dir, {"mode": "capture", "rows": len(rows)})
    print("captured", run_id, len(rows))


def cmd_report(a):
    pts = pd.read_parquet(PROC / "prediction_markets.parquet")
    sched, _ = latest_schedule()
    oc = pd.read_parquet(sorted((ROOT / "data" / "market" / "historical" / "open_close" / "raw").glob("*/games.parquet"))[-1])
    oc = oc.rename(columns={"nflverse_game_id": "game_id"})
    for src, d in pts.groupby("source"):
        print("==", src)
        piv = d.assign(ok=d.home_win_prob.notna(), sp=d.implied_home_spread.notna())
        t = piv.groupby(["season", "point"]).agg(games=("game_id", "nunique"), ml=("ok", "sum"), spread=("sp", "sum")).unstack("point")
        print(t.to_string())
        fz = d[d.point == "freeze"].merge(sched[["game_id", "home_team", "away_team", "gameday"]], on="game_id")
        fz = fz.merge(oc[["home_team", "away_team", "game_date", "opening_home_spread"]].assign(gameday=lambda x: pd.to_datetime(x.game_date)), on=["home_team", "away_team", "gameday"], how="inner")
        fz = fz[fz.implied_home_spread.notna()]
        if len(fz) > 2:
            print("freeze vs book open n", len(fz), "mean abs diff", round((fz.implied_home_spread + fz.opening_home_spread).abs().mean(), 3), "corr", round(fz.implied_home_spread.corr(-fz.opening_home_spread), 3), "mean diff", round((fz.implied_home_spread + fz.opening_home_spread).mean(), 3))


def main():
    ap = argparse.ArgumentParser(description="Read-only NFL prediction-market acquisition (Kalshi, Polymarket)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill")
    b.add_argument("--run-id")
    b.add_argument("--sources", nargs="+", default=["kalshi", "poly"])
    b.set_defaults(fn=cmd_backfill)
    sub.add_parser("derive").set_defaults(fn=cmd_derive)
    sub.add_parser("capture").set_defaults(fn=cmd_capture)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
