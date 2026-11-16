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

from capture_odds_gap_private import NFL_TEAM_NAMES  # noqa: E402
from lead98_price_skew import metrics, reliability  # noqa: E402
from mod24_u8_markets import (  # noqa: E402
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
from mod57_u3_deadline_fair_line import block_bootstrap, decisive  # noqa: E402

from nfl_ats.conditional_margin import fit_conditional_margin  # noqa: E402
from nfl_ats.odds import no_vig_probabilities  # noqa: E402

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
NAME_TO_CODE = dict(NFL_TEAM_NAMES)


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
    d, _raw, sched = pool_games()
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
    for _key, g in R.groupby(["game_id", "team"]):
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
    (out / "markets.json").write_text(json.dumps(ms))
    rows = []
    for i, r in enumerate(P.itertuples()):
        start = int(tuesday_noon(r.deadline).timestamp())
        end = int(r.deadline.timestamp())
        q = {"start_ts": start, "end_ts": end, "period_interval": 60}
        c = k.get(KAL + f"/series/{SERIES}/markets/{r.ticker}/candlesticks", q)
        if not c or not c.get("candlesticks"):
            c = k.get(KAL + f"/historical/markets/{r.ticker}/candlesticks", q)
        cs = (c or {}).get("candlesticks", [])
        (out / "candles" / f"{r.ticker}.json").write_text(json.dumps(cs))
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
        j = json.loads(Path(f).read_text(encoding="utf-8"))
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
    T["p"] = [
        no_vig_probabilities(hj, aj)[0] for hj, aj in zip(T.home_juice, T.away_juice, strict=True)
    ]
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
        c, _, lose = fit_conditional_margin(self.h, center, t).probabilities(t)
        return c / (c + lose)

    def translate(self, t_from, p_from, t_to):
        if t_from == t_to:
            return p_from

        def f(c):
            return self.cover(c, t_from) - p_from

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
                    for thr, p in zip(kl.threshold, hp, strict=True):
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
                    "description": (
                        "Market probability at the pool's exact number (sportsbook lines and "
                        "Kalshi ladder translated with the conditional margin lattice), "
                        f"{name}, versus the served card, {len(D)} games, week-block bootstrap."
                    ),
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
    txt = json.dumps(
        {"beta": beta, "beta_scored_weeks": beta_te, "results": res, "cells": cells},
        default=float,
        indent=1,
    )
    (out / "results.json").write_text(txt)
    (out / "cells.json").write_text(json.dumps(cells, indent=1))
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
    (out / "bovada_events.json").write_text(json.dumps(kept))
    sched, _ = latest_schedule()
    k = Polite("kalshi_public_market_data")
    ms = kalshi_markets(k, None, status="open")
    (out / "kalshi_markets.json").write_text(json.dumps(ms))
    R = kalshi_rows(ms, build_lookup(sched))
    if len(R):
        R = R[R.kickoff > now]
        hp = two_sided_home(R.bid.to_numpy(float), R.ask.to_numpy(float), R.home_yes.to_numpy(bool))
        for r, p in zip(R.itertuples(), hp, strict=True):
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


HISTORY_POP = ROOT / "tests" / "scratch" / "mod57" / "u8_run.py"
ATTRIB = ROOT / "artifacts" / "week_attribution" / "2026_season.csv"
U5_REPLAY = ROOT / "tests" / "scratch" / "mod57" / "p2e" / "scored.csv"
DRAWS = 4000


def history_rows():
    code = HISTORY_POP.read_text(encoding="utf-8-sig").split("\ndef boot(")[0]
    ns = {"__name__": "u8_history"}
    exec(compile(code, str(HISTORY_POP), "exec"), ns)
    s = ns["build_rows"]("S")
    s["S"] = ns["run"](s)[0]["two"]
    s["week"] = s.game_id.str.split("_").str[1].astype(int)
    return s[["game_id", "season", "week", "pool", "w", "y", "S", "FD"]].copy()


def cmd_kalshi_history(a):
    s = history_rows()
    sched, _ = latest_schedule()
    s = s.merge(sched[["game_id", "kickoff"]], on="game_id", how="left")
    print("history rows", len(s), "missing kickoff", int(s.kickoff.isna().sum()), flush=True)
    s = s[s.kickoff.notna()].copy()
    s["deadline"] = [pick_deadline(k) for k in s.kickoff]
    k = Polite("kalshi_public_market_data")
    ms = kalshi_markets(k, pd.Timestamp(a.since, tz="UTC"))
    opened = sorted(str(m.get("open_time")) for m in ms.values() if m.get("open_time"))
    R = kalshi_rows(ms, build_lookup(sched))
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = EXT / "kalshi_history" / stamp
    (out / "candles").mkdir(parents=True, exist_ok=True)
    (out / "markets.json").write_text(json.dumps(ms))
    s.to_parquet(out / "rows.parquet")
    gs = s[["game_id", "season"]].drop_duplicates()
    summary = {
        "markets_listed": len(ms),
        "earliest_open": opened[0] if opened else None,
        "latest_open": opened[-1] if opened else None,
        "matched_rows": len(R),
        "kalshi_games_by_season": R.merge(gs, on="game_id")
        .groupby("season")
        .game_id.nunique()
        .to_dict()
        if len(R)
        else {},
        "history_games_by_season": gs.groupby("season").game_id.nunique().to_dict(),
    }
    print(json.dumps(summary, default=str), flush=True)
    R = R[R.game_id.isin(s.game_id)] if len(R) else R
    pools = s.groupby("game_id").pool.apply(lambda x: sorted(set(x))).to_dict()
    dl = s.drop_duplicates("game_id").set_index("game_id").deadline
    pick = []
    for (gid, _side), g in R.groupby(["game_id", "team"]):
        for t in pools[gid]:
            lo = g[g.threshold <= t]
            hi = g[g.threshold >= t]
            if len(lo):
                pick.append(lo.loc[lo.threshold.idxmax()])
            if len(hi):
                pick.append(hi.loc[hi.threshold.idxmin()])
    P = pd.DataFrame(pick).drop_duplicates("ticker")
    rows = []
    for i, r in enumerate(P.itertuples()):
        end = dl[r.game_id]
        q = {
            "start_ts": int(tuesday_noon(end).timestamp()),
            "end_ts": int(end.timestamp()),
            "period_interval": 60,
        }
        c = k.get(KAL + f"/historical/markets/{r.ticker}/candlesticks", q)
        if not c or not c.get("candlesticks"):
            c = k.get(KAL + f"/series/{SERIES}/markets/{r.ticker}/candlesticks", q)
        cs = (c or {}).get("candlesticks", [])
        (out / "candles" / f"{r.ticker}.json").write_text(json.dumps(cs))
        for x in cs:
            f = flatten_kal(x)
            if f["ts"] <= q["end_ts"]:
                rows.append(
                    {
                        "game_id": r.game_id,
                        "ticker": r.ticker,
                        "threshold": r.threshold,
                        "home_yes": r.home_yes,
                        **f,
                    }
                )
        if i % 100 == 0:
            print("candles", i, len(P), flush=True)
    pd.DataFrame(rows).to_parquet(out / "ladder.parquet")
    summary.update({"markets_fetched": len(P), "candle_rows": len(rows), "requests": k.requests})
    (out / "summary.json").write_text(json.dumps(summary, default=str, indent=1))
    print(json.dumps({"out": str(out), **summary}, default=str))


def wstats(p, y, w):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ok = (p >= 0.5) == (y == 1)
    tot = w.sum()
    return {
        "wins": float((w * ok).sum()),
        "losses": float((w * ~ok).sum()),
        "accuracy": float((w * ok).sum() / tot),
        "log_loss": float((w * -(y * np.log(p) + (1 - y) * np.log1p(-p))).sum() / tot),
        "brier": float((w * (p - y) ** 2).sum() / tot),
    }


def wboot(block, y, pn, po, w, rng):
    _, b = np.unique(np.asarray(block), return_inverse=True)
    nb = b.max() + 1
    pn = np.clip(pn, 1e-9, 1 - 1e-9)
    po = np.clip(po, 1e-9, 1 - 1e-9)
    acc = ((pn >= 0.5) == (y == 1)).astype(float) - ((po >= 0.5) == (y == 1)).astype(float)
    brier = (pn - y) ** 2 - (po - y) ** 2
    ll = -(y * np.log(pn) + (1 - y) * np.log1p(-pn)) + (y * np.log(po) + (1 - y) * np.log1p(-po))
    wt = np.bincount(b, weights=w, minlength=nb)
    draws = rng.integers(0, nb, size=(DRAWS, nb))
    den = wt[draws].sum(axis=1)
    out = {"blocks": int(nb)}
    for name, v, scale in (
        ("accuracy_points", acc, 100.0),
        ("brier", brier, 1.0),
        ("log_loss", ll, 1.0),
    ):
        d = scale * np.bincount(b, weights=w * v, minlength=nb)[draws].sum(axis=1) / den
        better = (d > 0) if name == "accuracy_points" else (d < 0)
        out[name] = {
            "point": float(scale * (w * v).sum() / w.sum()),
            "delta_mean": float(d.mean()),
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "probability_positive": float(better.mean()),
        }
    return out


def compare(name, F, pn, po, blocks, rng):
    y = F.y.to_numpy(float)
    w = F.w.to_numpy(float)
    ch = (pn >= 0.5) != (po >= 0.5)
    ok = (pn >= 0.5) == (y == 1)
    res = {
        "rows": len(F),
        "games": int(F.game_id.nunique()),
        "arm": wstats(pn, y, w),
        "base": wstats(po, y, w),
        "changed": {
            "weight": float(w[ch].sum()),
            "arm_wins": float((w * (ch & ok)).sum()),
            "arm_losses": float((w * (ch & ~ok)).sum()),
        },
        "boot": {k: wboot(F[c].to_numpy(), y, pn, po, w, rng) for k, c in blocks.items()},
        "reliability_unweighted": reliability(pn, y),
    }
    print(name, json.dumps({k: v for k, v in res.items() if k != "reliability_unweighted"}))
    return res


def make_cells(prefix, desc, boot, games):
    out = []
    for unit, key, sign in (
        ("accuracy_points", "accuracy_points", 1),
        ("log_loss_improvement", "log_loss", -1),
        ("brier_improvement", "brier", -1),
    ):
        v = boot[key]
        lo, hi = sorted((sign * v["ci95"][0], sign * v["ci95"][1]))
        out.append(
            {
                "name": f"{prefix}_{unit.split('_')[0]}",
                "description": f"{desc}, {games} games, week-block bootstrap.",
                "effect": round(sign * v["delta_mean"], 6),
                "effect_units": unit,
                "interval_low": round(lo, 6),
                "interval_high": round(hi, 6),
                "probability_positive": round(v["probability_positive"], 4),
                "sample_games": int(games),
            }
        )
    return out


def blend(base, m, wt):
    has = np.isfinite(m)
    z = np.where(has, (1 - wt) * lg(base) + wt * lg(np.where(has, m, 0.5)), lg(base))
    return sig(z)


def fit_w(base, m, y, w):
    def loss(t):
        p = np.clip(blend(base, m, t), 1e-9, 1 - 1e-9)
        return -np.sum(w * (y * np.log(p) + (1 - y) * np.log1p(-p))) / w.sum()

    return float(minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded").x)


def history_table(hdir):
    s = pd.read_parquet(hdir / "rows.parquet")
    L = pd.read_parquet(hdir / "ladder.parquet")
    _, run = latest_schedule()
    raw = pd.read_parquet(ROOT / "data" / "raw" / run / "schedules.parquet")
    L["hp"] = two_sided_home(
        L.bid.to_numpy(float), L.ask.to_numpy(float), L.home_yes.to_numpy(bool)
    )
    L = L[np.isfinite(L.hp)]
    last = L.loc[L.groupby("ticker").ts.idxmax()]
    s["M"] = np.nan
    s["quotes"] = 0
    covered = s[s.game_id.isin(last.game_id)]
    for (season, _week), wk in covered.groupby(["season", "week"]):
        tr = Translator(raw, int(season), wk.deadline.min())
        for r in wk.itertuples():
            kl = last[last.game_id == r.game_id]
            vals = [
                tr.translate(float(t), float(p), float(r.pool))
                for t, p in zip(kl.threshold, kl.hp, strict=True)
            ]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                s.loc[r.Index, "M"] = float(np.median(vals))
                s.loc[r.Index, "quotes"] = len(vals)
    s["wk_block"] = s.season * 100 + s.week
    return s


def cmd_decisive(a):
    rng = np.random.default_rng(65)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = ART / f"decisive_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    X = pd.read_csv(sorted(ART.glob("2026*/per_game.csv"))[-1])
    A = pd.read_csv(ATTRIB)[["game_id", "p_current"]]
    U = pd.read_csv(U5_REPLAY)[["game_id", "cur"]]
    G = X[X.hc.notna()].merge(A, on="game_id", how="left").merge(U, on="game_id", how="left")
    G["y"] = G.hc
    G["w"] = 1.0
    G["wk_block"] = G.season * 100 + G.week
    m = G.M_all.to_numpy(float)
    cov = {
        "graded_games": len(G),
        "with_M": int(np.isfinite(m).sum()),
        "with_p_current": int(G.p_current.notna().sum()),
        "with_u5_replay": int(G.cur.notna().sum()),
        "weeks": sorted(int(x) for x in G.week.unique()),
    }
    print("coverage 2026", json.dumps(cov))
    G = G[G.p_current.notna()].reset_index(drop=True)
    m = G.M_all.to_numpy(float)
    cur = G.p_current.to_numpy(float)
    p0 = np.where(np.isfinite(m), m, cur)
    res = {"coverage_2026": cov}
    res["played_card"] = wstats(G.card.to_numpy(float), G.y.to_numpy(float), G.w.to_numpy(float))
    res["p0_vs_current"] = compare("p0_vs_current", G, p0, cur, {"week": "wk_block"}, rng)
    H5 = G[G.cur.notna()].reset_index(drop=True)
    m5 = H5.M_all.to_numpy(float)
    c5 = H5.cur.to_numpy(float)
    res["p0_vs_u5_w3_9"] = compare(
        "p0_vs_u5_w3_9", H5, np.where(np.isfinite(m5), m5, c5), c5, {"week": "wk_block"}, rng
    )
    ch = (p0 >= 0.5) != (cur >= 0.5)
    print(G.loc[ch, ["game_id", "pool_home_spread", "p_current", "M_all", "card", "y"]].round(3))
    c26 = make_cells(
        "pool_number_price_vs_current_formula",
        "Market probability at the pool number in place of the current-formula replay, 2026 W3-10",
        res["p0_vs_current"]["boot"]["week"],
        len(G),
    ) + make_cells(
        "pool_number_price_vs_u5_replay_w3_9",
        "Market probability at the pool number in place of the u5 harness replay, 2026 W3-9",
        res["p0_vs_u5_w3_9"]["boot"]["week"],
        len(H5),
    )
    ch_hist = []
    if a.history:
        hdir = sorted(p.parent for p in (EXT / "kalshi_history").glob("*/ladder.parquet"))[-1]
        H = history_table(hdir)
        H.to_parquet(out / "history_rows.parquet")
        hs = H.groupby("season").agg(
            rows=("game_id", "size"),
            games=("game_id", "nunique"),
            rows_with_M=("M", lambda x: int(x.notna().sum())),
            games_with_M=("game_id", lambda x: int(x[H.loc[x.index, "M"].notna()].nunique())),
        )
        print("coverage history by season")
        print(hs.to_string())
        res["coverage_history"] = hs.reset_index().to_dict("records")
        res["kalshi_summary"] = json.loads((hdir / "summary.json").read_text())
        seasons = [int(x) for x in hs.index[hs.rows_with_M > 0]]
        H = H[H.season.isin(seasons)].reset_index(drop=True)
        hm = H.M.to_numpy(float)
        hS = H.S.to_numpy(float)
        ph = np.where(np.isfinite(hm), hm, hS)
        blocks = {"week": "wk_block"}
        if len(seasons) > 1:
            blocks["season"] = "season"
        res["history_p0_vs_S"] = compare("history_p0_vs_S", H, ph, hS, blocks, rng)
        hy = H.y.to_numpy(float)
        hw = H.w.to_numpy(float)
        pb = np.full(len(H), np.nan)
        folds = {}
        split = H.season.to_numpy() if len(seasons) > 1 else np.where(H.week.to_numpy() <= 9, 0, 1)
        for f in np.unique(split):
            tr = split != f
            wt = fit_w(hS[tr], hm[tr], hy[tr], hw[tr])
            folds[int(f)] = wt
            pb[split == f] = blend(hS, hm, wt)[split == f]
        res["blend_folds"] = folds
        res["blend_split"] = "season" if len(seasons) > 1 else "half_season_weeks_le_9"
        res["history_blend_vs_S"] = compare("history_blend_vs_S", H, pb, hS, blocks, rng)
        w_all = fit_w(hS, hm, hy, hw)
        res["blend_weight_all_history"] = w_all
        pb26 = blend(cur, m, w_all)
        res["blend_2026_vs_current"] = compare(
            "blend_2026_vs_current", G, pb26, cur, {"week": "wk_block"}, rng
        )
        print("blend folds", folds, "weight on all history", round(w_all, 4))
        c26 += make_cells(
            "pool_number_price_blend_history_weight_2026",
            "Blend of the current-formula replay and the market probability at the pool "
            "number, one weight fit on the Kalshi history, scored 2026 W3-10",
            res["blend_2026_vs_current"]["boot"]["week"],
            len(G),
        )
        hg = int(H.game_id.nunique())
        sl = f"seasons {seasons[0]}-{seasons[-1]}"
        ch_hist = make_cells(
            "pool_number_price_history_kalshi_vs_served",
            f"Kalshi ladder probability at the FanDuel-rebuilt pool number in place of the served "
            f"out-of-season probability, {sl}",
            res["history_p0_vs_S"]["boot"]["week"],
            hg,
        ) + make_cells(
            "pool_number_price_history_blend_vs_served",
            f"Blend of served out-of-season probability and Kalshi ladder probability, "
            f"one weight cross-fit ({res['blend_split']}), {sl}",
            res["history_blend_vs_S"]["boot"]["week"],
            hg,
        )
    G.assign(p0=p0).to_csv(out / "per_game_2026.csv", index=False)
    (out / "results.json").write_text(json.dumps(res, default=float, indent=1))
    base = {
        "league": "nfl",
        "category": "market",
        "classification": "unresolved_below_power",
        "source": str((out / "results.json").relative_to(ROOT)).replace("\\", "/"),
    }
    (out / "batch_2026.json").write_text(
        json.dumps(
            {
                **base,
                "family": "pool_number_price_2026",
                "season_start": 2026,
                "season_end": 2026,
                "cells": c26,
            },
            indent=1,
        )
    )
    if ch_hist:
        (out / "batch_history.json").write_text(
            json.dumps(
                {
                    **base,
                    "family": "pool_number_price_history",
                    "season_start": seasons[0],
                    "season_end": seasons[-1],
                    "cells": ch_hist,
                },
                indent=1,
            )
        )
    print("out", out)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Market probability at the pool's exact number: per-number price capture, "
            "Kalshi ladder backfill and grading."
        )
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
    h = sub.add_parser(
        "kalshi-history", help="Kalshi spread ladder candles for the FanDuel-rebuilt history rows"
    )
    h.add_argument("--since", default="2021-01-01", help="earliest market close date to list")
    h.set_defaults(fn=cmd_kalshi_history)
    g = sub.add_parser(
        "decisive", help="Market probability against the current formula and history"
    )
    g.add_argument("--history", action="store_true", help="include the Kalshi history grade")
    g.set_defaults(fn=cmd_decisive)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
