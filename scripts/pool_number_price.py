import argparse
import datetime as dt
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from capture_odds_gap_private import NFL_TEAM_NAMES
from lead98_price_skew import metrics, reliability
from mod24_u8_markets import (
    ET,
    KAL,
    build_lookup,
    flatten_kal,
    fnum,
    latest_schedule,
    match_game,
    parse_kal_event,
    team,
)
from mod57_u3_deadline_fair_line import block_bootstrap, decisive

from nfl_ats.conditional_margin import fit_conditional_margin
from nfl_ats.odds import no_vig_probabilities

EXT = ROOT / "data" / "external" / "alt_spreads"
ART = ROOT / "artifacts" / "pool_number_price"
POLICIES = ROOT / "config" / "source_policies.json"
SERIES = "KXNFLSPREAD"
BOVADA_EVENT = "https://www.bovada.lv/services/sports/event/v2/events/A/description"
BOVADA_LIST = BOVADA_EVENT + "/football/nfl"
BOVADA_GROUPS = ("Game Lines", "Alternate Lines")
REPLAY = ROOT / "tests" / "scratch" / "flag_set_freeze" / "replay_2026.csv"
DEADLINES = ROOT / "tests" / "scratch" / "fixpicks" / "root_m2_games.csv"
SLOTS = (("thu", "19:30"), ("sat", "12:00"), ("sun", "09:15"), ("sun", "12:50"), ("sun", "15:45"))
NAME_TO_CODE = {k: v for k, v in NFL_TEAM_NAMES.items()}


def find_policy(obj, name):
    if isinstance(obj, dict):
        if isinstance(obj.get(name), dict):
            return obj[name]
        for v in obj.values():
            found = find_policy(v, name)
            if found:
                return found
    return None


def policy_delay(name):
    pol = find_policy(json.loads(POLICIES.read_text(encoding="utf-8")), name)
    if not pol or not pol.get("acquisition_allowed"):
        raise SystemExit(f"{name}: acquisition not allowed by source policy")
    for c in pol.get("conditions", []):
        m = re.fullmatch(r"minimum_delay_seconds_(\d+)_(\d+)", c)
        if m:
            return float(f"{m[1]}.{m[2]}")
    raise SystemExit(f"{name}: source policy names no minimum delay")


class Polite:
    def __init__(self, policy):
        self.delay = policy_delay(policy)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "research-readonly/1.0"
        self.requests = 0

    def get(self, url, params=None):
        time.sleep(self.delay)
        self.requests += 1
        r = self.session.get(url, params=params, timeout=60)
        if r.status_code == 404:
            return None
        if r.status_code in (403, 429) or r.status_code >= 500:
            raise SystemExit(f"stop on {r.status_code} from {url}")
        if not r.ok:
            return {"_error": r.status_code}
        return r.json()


def american(x):
    if x is None:
        return np.nan
    s = str(x).strip().upper()
    if s == "EVEN":
        return 100.0
    return fnum(s.replace("+", ""))


def kalshi_markets(k, min_close, status=None):
    out = {}
    paths = ("/markets",) if status else ("/historical/markets", "/markets")
    for path in paths:
        cur = None
        while True:
            p = {"series_ticker": SERIES, "limit": 1000}
            if status:
                p["status"] = status
            else:
                p["min_close_ts"] = int(min_close.timestamp())
            if cur:
                p["cursor"] = cur
            r = k.get(KAL + path, p)
            if not r or "markets" not in r:
                break
            for m in r["markets"]:
                out[m["ticker"]] = m
            cur = r.get("cursor")
            if not cur or not r["markets"]:
                break
    return out


def kalshi_rows(markets, lk):
    rows = []
    for t, m in markets.items():
        ev = parse_kal_event(m.get("event_ticker", ""))
        mt = re.search(r"-([A-Z]+)(\d+)$", t)
        if not ev or not mt or m.get("floor_strike") is None:
            continue
        day, a, b = ev
        g = match_game(lk, b, a, day) or match_game(lk, a, b, day)
        if g is None:
            continue
        side, s = team(mt[1]), float(m["floor_strike"])
        if side == team(g.home_team):
            thr, home_yes = s, True
        elif side == team(g.away_team):
            thr, home_yes = -s, False
        else:
            continue
        rows.append(
            {
                "game_id": g.game_id,
                "kickoff": g.kickoff,
                "ticker": t,
                "team": side,
                "strike": s,
                "threshold": thr,
                "home_yes": home_yes,
                "status": m.get("status"),
                "bid": fnum(m.get("yes_bid_dollars")),
                "ask": fnum(m.get("yes_ask_dollars")),
            }
        )
    return pd.DataFrame(rows)


def two_sided_home(bid, ask, home_yes):
    ok = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask < 1) & (bid <= ask)
    mid = np.where(ok, (bid + ask) / 2, np.nan)
    return np.where(home_yes, mid, 1 - mid)


def tuesday_noon(deadline):
    local = deadline.tz_convert(ET)
    day = local.normalize() - pd.Timedelta(days=(local.weekday() - 1) % 7)
    return day.replace(hour=12).tz_convert("UTC")


def pick_deadline(kickoff):
    local = kickoff.tz_convert(ET)
    back = 1 if local.weekday() == 0 else -(6 - local.weekday())
    sunday = local.normalize() - pd.Timedelta(days=back)
    return min(kickoff, sunday.replace(hour=16).tz_convert("UTC"))


def pool_games():
    d = pd.read_csv(REPLAY)
    sched, run = latest_schedule()
    raw = pd.read_parquet(ROOT / "data" / "raw" / run / "schedules.parquet")
    d = d.merge(raw[["game_id", "season", "result"]], on="game_id", how="left")
    d = d.merge(sched[["game_id", "kickoff"]], on="game_id", how="left")
    d["deadline"] = [pick_deadline(k) for k in d.kickoff]
    ref = pd.read_csv(DEADLINES)[["game_id", "pick_deadline_utc"]]
    ref = d.merge(ref, on="game_id")
    same = pd.to_datetime(ref.pick_deadline_utc, utc=True) == ref.deadline
    print("deadline check against the harness:", int(same.sum()), "of", len(ref))
    played = d.result.notna() & d.pool_home_spread.notna()
    d["hc_result"] = np.where(played, (d.result > d.pool_home_spread).astype(float), np.nan)
    d.loc[d.result == d.pool_home_spread, "hc_result"] = np.nan
    both = d.home_covered.notna() & d.hc_result.notna()
    print(
        "sign check: replay home_covered equals result > pool number on",
        int((d.home_covered[both] == d.hc_result[both]).sum()),
        "of",
        int(both.sum()),
    )
    d["hc"] = d.hc_result
    return d, raw, sched


def cmd_kalshi_backfill(a):
    d, raw, sched = pool_games()
    d = d[(d.week >= a.first_week) & (d.week <= a.last_week)].copy()
    lk = build_lookup(sched)
    k = Polite("kalshi_public_market_data")
    ms = kalshi_markets(k, d.deadline.min() - pd.Timedelta(days=7))
    R = kalshi_rows(ms, lk)
    print(
        "kalshi spread markets listed",
        len(ms),
        "matched to schedule",
        len(R),
        "requests",
        k.requests,
        flush=True,
    )
    R = R.merge(d[["game_id", "week", "pool_home_spread", "deadline"]], on="game_id")
    pick = []
    for (gid, side), g in R.groupby(["game_id", "team"]):
        t = g.pool_home_spread.iloc[0]
        lo = g[g.threshold <= t]
        hi = g[g.threshold >= t]
        if len(lo):
            pick.append(lo.loc[lo.threshold.idxmax()])
        if len(hi):
            pick.append(hi.loc[hi.threshold.idxmin()])
    P = pd.DataFrame(pick).drop_duplicates("ticker")
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = EXT / "kalshi_backfill" / stamp
    (out / "candles").mkdir(parents=True, exist_ok=True)
    json.dump(ms, open(out / "markets.json", "w"))
    rows = []
    for i, r in enumerate(P.itertuples()):
        start = int(tuesday_noon(r.deadline).timestamp())
        end = int(r.deadline.timestamp())
        q = {"start_ts": start, "end_ts": end, "period_interval": 60}
        c = k.get(KAL + f"/series/{SERIES}/markets/{r.ticker}/candlesticks", q)
        if not c or not c.get("candlesticks"):
            c = k.get(KAL + f"/historical/markets/{r.ticker}/candlesticks", q)
        cs = (c or {}).get("candlesticks", [])
        json.dump(cs, open(out / "candles" / f"{r.ticker}.json", "w"))
        for x in cs:
            f = flatten_kal(x)
            if f["ts"] <= end:
                rows.append(
                    {
                        "game_id": r.game_id,
                        "ticker": r.ticker,
                        "threshold": r.threshold,
                        "home_yes": r.home_yes,
                        **f,
                    }
                )
        if i % 50 == 0:
            print("candles", i, len(P), flush=True)
    C = pd.DataFrame(rows)
    C.to_parquet(out / "ladder.parquet")
    print(
        json.dumps(
            {
                "out": str(out),
                "markets_fetched": len(P),
                "games": int(P.game_id.nunique()),
                "candle_rows": len(C),
                "requests": k.requests,
            }
        )
    )


def odds_gap_rows(lk):
    rows = []
    for f in sorted(
        glob.glob(str(ROOT / "data" / "market" / "raw" / "*-odds-gap-private" / "response.json"))
    ):
        j = json.load(open(f))
        scan = pd.to_datetime(j.get("last_updated"), utc=True, errors="coerce")
        for e in j.get("games", []):
            if e.get("sport") != "americanfootball_nfl":
                continue
            h, aw = NFL_TEAM_NAMES.get(e.get("home")), NFL_TEAM_NAMES.get(e.get("away"))
            ko = pd.to_datetime(e.get("commence_time"), utc=True, errors="coerce")
            if h is None or aw is None or pd.isna(ko):
                continue
            g = match_game(lk, h, aw, ko.tz_convert(ET).date())
            if g is None:
                continue
            for book, q in ((e.get("spread_data") or {}).get("all_books") or {}).items():
                rows.append(
                    (
                        g.game_id,
                        scan,
                        book,
                        fnum(q.get("home_line")),
                        fnum(q.get("home_juice")),
                        fnum(q.get("away_juice")),
                    )
                )
    T = pd.DataFrame(
        rows, columns=["game_id", "scan", "book", "home_line", "home_juice", "away_juice"]
    ).dropna()
    T["threshold"] = -T.home_line
    T["p"] = [no_vig_probabilities(hj, aj)[0] for hj, aj in zip(T.home_juice, T.away_juice)]
    return T


class Translator:
    def __init__(self, raw, season, cutoff):
        h = raw[(raw.season >= season - 4) & raw.result.notna() & raw.spread_line.notna()].copy()
        h["gameday"] = pd.to_datetime(h.gameday)
        h = h[(h.gameday + pd.Timedelta(days=1)) < cutoff.tz_convert(ET).tz_localize(None)]
        self.h = pd.DataFrame(
            {"predicted_margin": h.spread_line, "result": h.result, "spread_line": h.spread_line}
        )
        self.lo, self.hi = float(h.result.min()), float(h.result.max())
        self.rows = len(h)

    def cover(self, center, t):
        c, _, l = fit_conditional_margin(self.h, center, t).probabilities(t)
        return c / (c + l)

    def translate(self, t_from, p_from, t_to):
        if t_from == t_to:
            return p_from
        f = lambda c: self.cover(c, t_from) - p_from
        if f(self.lo) > 0 or f(self.hi) < 0:
            return np.nan
        return self.cover(brentq(f, self.lo, self.hi), t_to)


def build_table(a):
    d, raw, sched = pool_games()
    d = d[(d.week >= a.first_week) & (d.week <= a.last_week)].copy()
    lk = build_lookup(sched)
    T = odds_gap_rows(lk)
    T = T.merge(d[["game_id", "deadline"]], on="game_id")
    T = T[T.scan < T.deadline]
    last = T.groupby("game_id").scan.transform("max")
    T = T[T.scan == last]
    K = (
        pd.read_parquet(sorted((EXT / "kalshi_backfill").glob("*/ladder.parquet"))[-1])
        if a.kalshi
        else pd.DataFrame()
    )
    out = []
    for (season, week), wk in d.groupby(["season", "week"]):
        tr = Translator(raw, int(season), wk.deadline.min())
        for g in wk.itertuples():
            t = g.pool_home_spread
            b = T[T.game_id == g.game_id]
            exact = b[np.isclose(b.threshold, t)]
            vals = [tr.translate(x.threshold, x.p, t) for x in b.itertuples()]
            vals = [v for v in vals if np.isfinite(v)]
            kv = []
            if len(K):
                kg = K[K.game_id == g.game_id]
                if len(kg):
                    kl = kg.loc[kg.groupby("ticker").ts.idxmax()]
                    hp = two_sided_home(
                        kl.bid.to_numpy(float), kl.ask.to_numpy(float), kl.home_yes.to_numpy(bool)
                    )
                    for thr, p in zip(kl.threshold, hp):
                        if np.isfinite(p):
                            v = tr.translate(float(thr), float(p), t)
                            if np.isfinite(v):
                                kv.append(v)
            mk = float(np.median(kv)) if kv else np.nan
            venues = vals + ([mk] if kv else [])
            out.append(
                {
                    "game_id": g.game_id,
                    "season": int(season),
                    "week": int(week),
                    "pool_home_spread": t,
                    "card": g.served_home_probability,
                    "hc": g.hc,
                    "scan_lag_h": (g.deadline - b.scan.max()).total_seconds() / 3600
                    if len(b)
                    else np.nan,
                    "books": len(b),
                    "books_exact": len(exact),
                    "kalshi_quotes": len(kv),
                    "M_exact": float(np.median(exact.p)) if len(exact) else np.nan,
                    "M_books": float(np.median(vals)) if vals else np.nan,
                    "M_kalshi": mk,
                    "M_all": float(np.median(venues)) if venues else np.nan,
                    "history_rows": tr.rows,
                }
            )
    return pd.DataFrame(out)


def lg(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def fitted(card, m, beta):
    return sig(lg(card) + beta * np.where(np.isfinite(m), lg(m), 0.0))


def fit_beta(card, m, y):
    def loss(b):
        p = np.clip(fitted(card, m, b), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))

    return float(minimize_scalar(loss).x)


def scored(name, D, p, rng, cells, record):
    y = D.hc.to_numpy(float)
    cu = D.card.to_numpy(float)
    dec = decisive(p, cu, y)
    fr = D[["season", "week"]].copy()
    fr["home_covered"] = y
    b = block_bootstrap(fr, p, cu, rng)
    res = {
        "arm": metrics(p, y),
        "card": metrics(cu, y),
        "decisive": dec,
        "minus_card": b,
        "reliability": reliability(p, y),
    }
    print(name, json.dumps({k: v for k, v in res.items() if k != "reliability"}, default=float))
    print(name, "reliability", json.dumps(res["reliability"]))
    if record:
        for unit, key, sign in (
            ("accuracy_points", "accuracy_points", 1),
            ("log_loss_improvement", "log_loss", -1),
            ("brier_improvement", "brier", -1),
        ):
            v = b[key]
            lo, hi = sorted((sign * v["ci95"][0], sign * v["ci95"][1]))
            cells.append(
                {
                    "name": f"pool_number_price_{name}_{unit.split('_')[0]}",
                    "description": f"Market probability at the pool's exact number (sportsbook lines and Kalshi ladder translated with the conditional margin lattice), {name}, versus the served card, {len(D)} games, week-block bootstrap.",
                    "effect": round(sign * v["delta_mean"], 6),
                    "effect_units": unit,
                    "interval_low": round(lo, 6),
                    "interval_high": round(hi, 6),
                    "probability_positive": round(v["probability_positive"], 4),
                    "sample_games": len(D),
                }
            )
    return res


def cmd_grade(a):
    X = build_table(a)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = ART / stamp
    out.mkdir(parents=True, exist_ok=True)
    X.to_csv(out / "per_game.csv", index=False)
    cov = X.groupby("week").agg(
        games=("game_id", "size"),
        graded=("hc", lambda s: int(s.notna().sum())),
        exact=("M_exact", lambda s: int(s.notna().sum())),
        books=("M_books", lambda s: int(s.notna().sum())),
        kalshi=("M_kalshi", lambda s: int(s.notna().sum())),
        any_venue=("M_all", lambda s: int(s.notna().sum())),
    )
    print("coverage by week")
    print(cov.to_string())
    print("scan lag hours", X.scan_lag_h.describe().round(2).to_dict())
    both = X.M_exact.notna() & X.M_books.notna()
    print(
        "exact vs translated books on exact games, mean abs diff",
        round(float((X.M_exact - X.M_books)[both].abs().mean()), 4),
        "games",
        int(both.sum()),
    )
    bk = X.M_books.notna() & X.M_kalshi.notna()
    print(
        "books vs kalshi translated, mean abs diff",
        round(float((X.M_books - X.M_kalshi)[bk].abs().mean()), 4),
        "games",
        int(bk.sum()),
    )
    G = X[X.hc.notna()].copy()
    rng = np.random.default_rng(65)
    cells = []
    p0 = np.where(G.M_all.notna(), G.M_all, G.card)
    res = {"probability_alone_w3_10": scored("probability_alone_w3_10", G, p0, rng, cells, True)}
    tr, te = G[G.week <= a.fit_last_week], G[G.week > a.fit_last_week]
    beta = fit_beta(tr.card.to_numpy(float), tr.M_all.to_numpy(float), tr.hc.to_numpy(float))
    beta_te = fit_beta(te.card.to_numpy(float), te.M_all.to_numpy(float), te.hc.to_numpy(float))
    print(
        "beta fit on weeks",
        sorted(tr.week.unique()),
        round(beta, 4),
        "beta if fit on the scored weeks (in-sample check)",
        round(beta_te, 4),
    )
    res["fitted_in_sample"] = scored(
        "fitted_in_sample_w3_6",
        tr,
        fitted(tr.card.to_numpy(float), tr.M_all.to_numpy(float), beta),
        rng,
        cells,
        False,
    )
    res["fitted_out_of_sample"] = scored(
        "fitted_w7_10",
        te,
        fitted(te.card.to_numpy(float), te.M_all.to_numpy(float), beta),
        rng,
        cells,
        True,
    )
    p0te = np.where(te.M_all.notna(), te.M_all, te.card)
    res["probability_alone_w7_10"] = scored("probability_alone_w7_10", te, p0te, rng, cells, False)
    ch = (p0 >= 0.5) != (G.card >= 0.5)
    print("changed picks, probability alone")
    print(
        G.loc[
            ch,
            [
                "game_id",
                "pool_home_spread",
                "card",
                "M_exact",
                "M_books",
                "M_kalshi",
                "M_all",
                "hc",
            ],
        ]
        .round(3)
        .to_string()
    )
    json.dump(
        {"beta": beta, "beta_scored_weeks": beta_te, "results": res, "cells": cells},
        open(out / "results.json", "w"),
        default=float,
        indent=1,
    )
    json.dump(cells, open(out / "cells.json", "w"), indent=1)
    print("out", out)


def bovada_ladder(ev, captured):
    comps = {c.get("home"): c.get("name") for c in ev.get("competitors", [])}
    home, away = NAME_TO_CODE.get(comps.get(True)), NAME_TO_CODE.get(comps.get(False))
    prices = {}
    for dg in ev.get("displayGroups", []):
        if dg.get("description") not in BOVADA_GROUPS:
            continue
        for m in dg.get("markets", []):
            if "Spread" not in m.get("description", "") or (m.get("period") or {}).get(
                "description"
            ) not in ("Game", "Match"):
                continue
            for o in m.get("outcomes", []):
                h = fnum((o.get("price") or {}).get("handicap"))
                side = NAME_TO_CODE.get(o.get("description"))
                if side in (home, away) and np.isfinite(h):
                    prices[(side, h)] = american((o.get("price") or {}).get("american"))
    rows = []
    for (side, h), am in prices.items():
        if side != home or (away, -h) not in prices:
            continue
        aw = prices[(away, -h)]
        rows.append(
            {
                "captured_at": captured,
                "source": "bovada",
                "home": home,
                "away": away,
                "kickoff": pd.to_datetime(ev.get("startTime"), unit="ms", utc=True),
                "threshold": -h,
                "home_american": am,
                "away_american": aw,
                "home_cover": no_vig_probabilities(am, aw)[0],
            }
        )
    return rows


def cmd_capture(a):
    now = dt.datetime.now(dt.UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out = EXT / "live" / stamp
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    b = Polite("bovada_alternate_spreads")
    lst = b.get(BOVADA_LIST, {"marketFilterId": "def", "preMatchOnly": "true", "lang": "en"}) or []
    events = [e for g in (lst if isinstance(lst, list) else [lst]) for e in g.get("events", [])]
    kept = []
    for e in events:
        if e.get("live") or pd.to_datetime(e.get("startTime"), unit="ms", utc=True) <= now:
            continue
        j = b.get(BOVADA_EVENT + e["link"], {"lang": "en"})
        full = [x for g in (j if isinstance(j, list) else [j or {}]) for x in g.get("events", [])]
        for x in full:
            x["displayGroups"] = [
                dg for dg in x.get("displayGroups", []) if dg.get("description") in BOVADA_GROUPS
            ]
            kept.append(x)
            rows += bovada_ladder(x, stamp)
    json.dump(kept, open(out / "bovada_events.json", "w"))
    sched, _ = latest_schedule()
    k = Polite("kalshi_public_market_data")
    ms = kalshi_markets(k, None, status="open")
    json.dump(ms, open(out / "kalshi_markets.json", "w"))
    R = kalshi_rows(ms, build_lookup(sched))
    if len(R):
        R = R[R.kickoff > now]
        hp = two_sided_home(R.bid.to_numpy(float), R.ask.to_numpy(float), R.home_yes.to_numpy(bool))
        for r, p in zip(R.itertuples(), hp):
            rows.append(
                {
                    "captured_at": stamp,
                    "source": "kalshi",
                    "home": None,
                    "away": None,
                    "game_id": r.game_id,
                    "kickoff": r.kickoff,
                    "threshold": r.threshold,
                    "home_cover": p,
                    "bid": r.bid,
                    "ask": r.ask,
                    "ticker": r.ticker,
                }
            )
    L = pd.DataFrame(rows)
    if len(L):
        lk = build_lookup(sched)
        miss = L.get("game_id", pd.Series(index=L.index, dtype=object)).isna()
        L.loc[miss, "game_id"] = [
            getattr(
                match_game(lk, r.home, r.away, r.kickoff.tz_convert(ET).date()), "game_id", None
            )
            for r in L[miss].itertuples()
        ]
        L.to_parquet(out / "ladder.parquet")
    summary = {
        "out": str(out),
        "bovada_games": int(L[L.source == "bovada"].game_id.nunique()) if len(L) else 0,
        "bovada_numbers": int((L.source == "bovada").sum()) if len(L) else 0,
        "kalshi_games": int(L[(L.source == "kalshi") & L.home_cover.notna()].game_id.nunique())
        if len(L)
        else 0,
        "kalshi_two_sided_strikes": int(((L.source == "kalshi") & L.home_cover.notna()).sum())
        if len(L)
        else 0,
        "requests": {"bovada": b.requests, "kalshi": k.requests},
    }
    print(json.dumps(summary))


def cmd_slots(a):
    argv = [
        str(ROOT / ".tools" / "uv.exe"),
        "run",
        "--no-sync",
        "python",
        str(ROOT / "scripts" / "pool_number_price.py"),
        "capture",
    ]
    print(json.dumps({"argv": argv, "slots": [{"day": d, "at": t} for d, t in SLOTS]}, indent=1))


def main():
    ap = argparse.ArgumentParser(
        description="Market probability at the pool's exact number: per-number price capture, Kalshi ladder backfill and grading."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("kalshi-backfill", cmd_kalshi_backfill), ("grade", cmd_grade)):
        p = sub.add_parser(
            name,
            help="Kalshi spread ladder candles at each pick deadline"
            if fn is cmd_kalshi_backfill
            else "Build the per-game table and grade on the served card",
        )
        p.add_argument("--first-week", type=int, required=True, help="first 2026 week")
        p.add_argument("--last-week", type=int, required=True, help="last 2026 week")
        p.add_argument(
            "--fit-last-week", type=int, help="last week used to fit the one coefficient"
        )
        p.add_argument(
            "--kalshi", action="store_true", help="include the latest Kalshi ladder backfill"
        )
        p.set_defaults(fn=fn)
    c = sub.add_parser(
        "capture", help="Save Bovada alternate spreads and Kalshi spread ladders for upcoming games"
    )
    c.set_defaults(fn=cmd_capture)
    s = sub.add_parser("slots", help="Print the scheduler argv and pre-deadline slots")
    s.set_defaults(fn=cmd_slots)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
