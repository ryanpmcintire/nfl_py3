import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
UA = "Mozilla/5.0 (compatible; research-script/1.0)"
OUTCOME_COLUMN = re.compile(
    r"(score|result|_post$|post_|actual|cover|outcome|winner|game_value|^won$|^final|_win$|^win$|^margin$|point_diff)",
    re.IGNORECASE,
)

NICKNAME_CODE = {
    "Cardinals": "ARI",
    "Falcons": "ATL",
    "Ravens": "BAL",
    "Bills": "BUF",
    "Panthers": "CAR",
    "Bears": "CHI",
    "Bengals": "CIN",
    "Browns": "CLE",
    "Cowboys": "DAL",
    "Broncos": "DEN",
    "Lions": "DET",
    "Packers": "GB",
    "Texans": "HOU",
    "Colts": "IND",
    "Jaguars": "JAX",
    "Chiefs": "KC",
    "Rams": "LA",
    "Chargers": "LAC",
    "Raiders": "LV",
    "Dolphins": "MIA",
    "Vikings": "MIN",
    "Patriots": "NE",
    "Saints": "NO",
    "Giants": "NYG",
    "Jets": "NYJ",
    "Eagles": "PHI",
    "Steelers": "PIT",
    "Seahawks": "SEA",
    "49ers": "SF",
    "Buccaneers": "TB",
    "Titans": "TEN",
    "Redskins": "WAS",
    "Commanders": "WAS",
    "Football Team": "WAS",
    "Washington": "WAS",
}
ALT_CODE = {
    "WSH": "WAS",
    "LAR": "LA",
    "JAC": "JAX",
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "GNB": "GB",
    "KAN": "KC",
    "NWE": "NE",
    "NOR": "NO",
    "SFO": "SF",
    "TAM": "TB",
    "LVR": "LV",
    "LA": "LA",
}
REQUESTS = {"count": 0}


def code(raw):
    raw = str(raw).strip().upper()
    return ALT_CODE.get(raw, raw)


def fetch(url, delay, tries=3, as_json=False):
    for attempt in range(tries):
        time.sleep(delay)
        REQUESTS["count"] += 1
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
            return json.loads(body) if as_json else body
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < tries:
                time.sleep(60 * (attempt + 1))
                continue
            if exc.code == 403:
                raise SystemExit(f"403 stop on {url}")
            raise
        except urllib.error.URLError:
            if attempt + 1 < tries:
                time.sleep(30)
                continue
            raise
    return None


def out_dir(name):
    d = EXT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / ".gitignore").write_text("*\n", encoding="utf-8")
    return d


def repo_games():
    g = pd.read_parquet(
        ROOT / "data" / "processed" / "game_features.parquet",
        columns=["game_id", "season", "week", "gameday", "kickoff", "home_team", "away_team"],
    )
    sched = sorted((ROOT / "data" / "raw").glob("2*/schedules.parquet"))[-1]
    s = pd.read_parquet(sched, columns=["game_id", "roof", "stadium", "stadium_id"])
    g = g.merge(s, on="game_id", how="left")
    g["gameday"] = pd.to_datetime(g["gameday"]).dt.date
    return g


def write_inventory(d, payload):
    payload["requests_made"] = REQUESTS["count"]
    payload["written_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    (d / "inventory.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str)[:4000])


def cells(tr):
    raw = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)
    return [
        re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c)).replace("&nbsp;", " ").strip() for c in raw
    ]


def team_from_name(name):
    for nick, c in sorted(NICKNAME_CODE.items(), key=lambda kv: -len(kv[0])):
        if name.endswith(nick) or nick in name:
            return c
    return None


def run_wins(args):
    d = out_dir("win_totals")
    rows = []
    pages = {}
    for season in range(args.first, args.last + 1):
        url = f"https://www.sportsoddshistory.com/nfl-win/?y={season}&sa=nfl&t=win&o=t"
        try:
            html = fetch(url, args.delay).decode("utf-8", "replace")
        except Exception as exc:
            pages[season] = f"error {exc}"
            continue
        m = re.search(r"As of ([A-Z][a-z]+ \d{1,2}, \d{4})", re.sub(r"<[^>]+>", " ", html))
        as_of = m.group(1) if m else None
        pages[season] = as_of
        for t_idx, table in enumerate(re.findall(r"<table.*?</table>", html, re.S)):
            for tr in re.findall(r"<tr.*?</tr>", table, re.S):
                c = cells(tr)
                if len(c) < 4:
                    continue
                team = team_from_name(c[0])
                if team is None or not re.fullmatch(r"\d+(\.5)?", c[1]):
                    continue
                rows.append(
                    {
                        "season": season,
                        "team": team,
                        "team_name": c[0],
                        "win_total": float(c[1]),
                        "over_odds": c[2],
                        "under_odds": c[3],
                        "page_as_of": as_of,
                        "table_index": t_idx,
                        "source_url": url,
                    }
                )
    df = pd.DataFrame(rows).drop_duplicates(["season", "team"], keep="first")
    df.to_parquet(d / "nfl_win_totals.parquet", index=False)
    g = repo_games()
    team_seasons = pd.concat(
        [
            g[["season", "home_team"]].rename(columns={"home_team": "team"}),
            g[["season", "away_team"]].rename(columns={"away_team": "team"}),
        ]
    ).drop_duplicates()
    j = team_seasons.merge(df[["season", "team"]].assign(hit=1), how="left", on=["season", "team"])
    by = j.groupby("season")["hit"].apply(lambda s: f"{int(s.notna().sum())}/{len(s)}").to_dict()
    write_inventory(
        d,
        {
            "source": "sportsoddshistory_win_totals",
            "rows": len(df),
            "rows_by_season": df.groupby("season").size().to_dict(),
            "page_as_of": pages,
            "repo_team_season_join": by,
            "join_rate": float(j["hit"].notna().mean()),
            "columns_kept": list(df.columns),
            "outcome_columns_dropped": ["Week bet settled", "Actual Wins", "Result"],
        },
    )


def run_weather(args):
    d = out_dir("open_meteo")
    reg = json.loads((ROOT / "registry" / "stadium_coordinates.json").read_text(encoding="utf-8"))
    g = repo_games()
    g = g[(g["season"] >= args.first) & g["roof"].isin(["outdoors", "open"])].copy()
    g["kick_hour"] = g["kickoff"].dt.floor("h")
    missing = sorted(set(g["stadium"].dropna()) - set(reg))
    raw_dir = d / "raw"
    raw_dir.mkdir(exist_ok=True)
    out = []
    endpoints = {
        "hist": (
            "https://historical-forecast-api.open-meteo.com/v1/forecast",
            "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation",
        ),
        "prev": (
            "https://previous-runs-api.open-meteo.com/v1/forecast",
            "temperature_2m_previous_day1,wind_speed_10m_previous_day1,"
            "temperature_2m_previous_day2,wind_speed_10m_previous_day2",
        ),
    }
    stops = {}
    for (stadium, season), grp in g.groupby(["stadium", "season"]):
        if stadium not in reg:
            continue
        lat, lon = reg[stadium]["lat"], reg[stadium]["lon"]
        start = grp["kick_hour"].min().date()
        end = min(grp["kick_hour"].max().date(), datetime.now(UTC).date() - timedelta(days=1))
        if end < start:
            continue
        frames = []
        for key, (base, hourly) in endpoints.items():
            if key in stops:
                continue
            cache = raw_dir / f"{key}_{re.sub(r'[^A-Za-z0-9]+', '_', stadium)}_{season}.json"
            if cache.exists():
                data = json.loads(cache.read_text(encoding="utf-8"))
            else:
                q = urllib.parse.urlencode(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": start.isoformat(),
                        "end_date": end.isoformat(),
                        "hourly": hourly,
                        "temperature_unit": "fahrenheit",
                        "wind_speed_unit": "mph",
                        "timezone": "GMT",
                    }
                )
                try:
                    data = fetch(f"{base}?{q}", args.delay, as_json=True)
                except urllib.error.HTTPError as exc:
                    data = {
                        "error": f"http {exc.code}",
                        "body": exc.read()[:300].decode("utf-8", "replace"),
                    }
                except SystemExit as exc:
                    stops[key] = str(exc)
                    continue
                cache.write_text(json.dumps(data), encoding="utf-8")
            h = data.get("hourly") if isinstance(data, dict) else None
            if not h:
                continue
            f = pd.DataFrame(h)
            f["kick_hour"] = pd.to_datetime(f.pop("time")).dt.tz_localize("UTC")
            f = f.rename(columns={c: f"{key}_{c}" for c in f.columns if c != "kick_hour"})
            frames.append(f)
        sub = grp[["game_id", "season", "stadium", "roof", "kick_hour"]]
        for f in frames:
            sub = sub.merge(f, on="kick_hour", how="left")
        out.append(sub)
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(d / "kickoff_forecast_weather.parquet", index=False)
    cov = df.groupby("season").agg(
        outdoor_games=("game_id", "size"),
        hist_temp=("hist_temperature_2m", lambda s: int(s.notna().sum()))
        if "hist_temperature_2m" in df
        else ("game_id", lambda s: 0),
        prev1_temp=("prev_temperature_2m_previous_day1", lambda s: int(s.notna().sum()))
        if "prev_temperature_2m_previous_day1" in df
        else ("game_id", lambda s: 0),
    )
    total_outdoor = int(len(g))
    write_inventory(
        d,
        {
            "source": "open_meteo_historical_forecast_and_previous_runs",
            "rows": len(df),
            "repo_outdoor_games_since_first": total_outdoor,
            "stadiums_missing_coordinates": missing,
            "coverage_by_season": cov.to_dict(orient="index"),
            "stops": stops,
            "kickoff_hour_rule": "kickoff (UTC, repo game_features) floored to the hour",
        },
    )


def run_elo538(args):
    d = out_dir("fivethirtyeight_elo")
    cdx = fetch(
        "https://web.archive.org/cdx/search/cdx?url=projects.fivethirtyeight.com/nfl-api/nfl_elo.csv"
        "&output=json&filter=statuscode:200",
        args.delay,
        as_json=True,
    )
    latest = cdx[-1][1]
    body = fetch(
        f"https://web.archive.org/web/{latest}id_/https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv",
        args.delay,
    )
    raw = pd.read_csv(io.BytesIO(body))
    keep = [
        "date",
        "season",
        "neutral",
        "playoff",
        "team1",
        "team2",
        "elo1_pre",
        "elo2_pre",
        "elo_prob1",
        "elo_prob2",
        "qbelo1_pre",
        "qbelo2_pre",
        "qb1",
        "qb2",
        "qb1_value_pre",
        "qb2_value_pre",
        "qb1_adj",
        "qb2_adj",
        "qbelo_prob1",
        "qbelo_prob2",
    ]
    df = raw[[c for c in keep if c in raw.columns]].copy()
    df = df[df["season"] >= args.first]
    df["home_team"] = df["team1"].map(code)
    df["away_team"] = df["team2"].map(code)
    df["gameday"] = pd.to_datetime(df["date"]).dt.date
    g = repo_games()
    m = g[["game_id", "season", "gameday", "home_team", "away_team"]].merge(
        df.drop(columns=["season"]), on=["gameday", "home_team", "away_team"], how="left"
    )
    swap = m["elo1_pre"].isna()
    alt = g[["game_id", "gameday", "home_team", "away_team"]].merge(
        df.rename(columns={"home_team": "away_team", "away_team": "home_team"}).drop(
            columns=["season"]
        ),
        on=["gameday", "home_team", "away_team"],
        how="inner",
    )
    m = m[~swap]
    m = pd.concat(
        [
            m,
            alt[alt["game_id"].isin(set(g["game_id"]) - set(m["game_id"]))].assign(
                orientation_swapped=True
            ),
        ]
    )
    m.to_parquet(d / "nfl_elo_pregame_joined.parquet", index=False)
    df.to_parquet(d / "nfl_elo_pregame.parquet", index=False)
    rg = g[g["season"] <= df["season"].max()]
    by = {
        int(s): f"{int(m[m['season'] == s]['game_id'].nunique())}/{int((rg['season'] == s).sum())}"
        for s in sorted(rg["season"].unique())
    }
    write_inventory(
        d,
        {
            "source": "fivethirtyeight_nfl_elo_via_internet_archive",
            "capture_timestamp": latest,
            "captures_listed": len(cdx) - 1,
            "rows_since_first": len(df),
            "seasons": [int(df["season"].min()), int(df["season"].max())],
            "repo_join_by_season": by,
            "join_rate_through_last_season": float(m["game_id"].nunique() / max(len(rg), 1)),
            "columns_kept": list(df.columns),
            "columns_dropped_unread": sorted(set(raw.columns) - set(keep)),
        },
    )


def run_nfelo(args):
    d = out_dir("nfelo")
    base = "https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/"
    g = repo_games()
    info = {}
    for name in [
        "nfelo_games.csv",
        "historic_projected_spreads.csv",
        "prediction_tracker.csv",
        "elo_snapshot.csv",
    ]:
        raw = pd.read_csv(io.BytesIO(fetch(base + name, args.delay)), low_memory=False)
        dropped = [c for c in raw.columns if OUTCOME_COLUMN.search(str(c))]
        df = raw.drop(columns=dropped)
        df.to_parquet(d / name.replace(".csv", ".parquet"), index=False)
        entry = {
            "rows": len(df),
            "columns_kept": list(df.columns),
            "columns_dropped_unread": dropped,
        }
        if "game_id" in df.columns:
            ids = set(df["game_id"].astype(str))
            gs = g.assign(hit=g["game_id"].isin(ids))
            entry["repo_join_by_season"] = (
                gs.groupby("season")["hit"].apply(lambda s: f"{int(s.sum())}/{len(s)}").to_dict()
            )
            entry["join_rate"] = float(gs["hit"].mean())
        if "season" in df.columns:
            entry["rows_by_season"] = df.groupby("season").size().to_dict()
        commits = []
        for page in range(1, 11):
            c = fetch(
                f"https://api.github.com/repos/greerreNFL/nfelo/commits?path=output_data/{name}&per_page=100&page={page}",
                args.delay,
                as_json=True,
            )
            if not c:
                break
            commits += [x["commit"]["committer"]["date"] for x in c]
            if len(c) < 100:
                break
        entry["commit_count"] = len(commits)
        entry["commit_range"] = [min(commits), max(commits)] if commits else None
        pd.Series(commits, name="commit_utc").to_frame().to_parquet(
            d / name.replace(".csv", "_commits.parquet"), index=False
        )
        info[name] = entry
    write_inventory(d, {"source": "github_nfelo_public_output_data", "files": info})


def match_game(g, day, teams):
    teams = {code(t) for t in teams}
    cand = g[g["gameday"].between(day - timedelta(days=1), day + timedelta(days=1))]
    hit = cand[cand.apply(lambda r: {r["home_team"], r["away_team"]} == teams, axis=1)]
    return hit.iloc[0] if len(hit) else None


def run_kalshi(args):
    d = out_dir("kalshi")
    base = "https://api.elections.kalshi.com/trade-api/v2"
    g = repo_games()
    markets = []
    for path, status in [
        ("historical/markets", None),
        ("markets", "settled"),
        ("markets", "closed"),
        ("markets", "open"),
    ]:
        cursor = ""
        while True:
            params = {"series_ticker": args.series, "limit": 1000, "cursor": cursor}
            if status:
                params["status"] = status
            q = urllib.parse.urlencode(params)
            r = fetch(f"{base}/{path}?{q}", args.delay, as_json=True)
            markets += r.get("markets", [])
            cursor = r.get("cursor") or ""
            if not cursor or not r.get("markets"):
                break
    mk = pd.DataFrame(markets).drop_duplicates("ticker")
    keep = [
        c
        for c in [
            "ticker",
            "event_ticker",
            "title",
            "yes_sub_title",
            "open_time",
            "close_time",
            "expected_expiration_time",
            "status",
        ]
        if c in mk.columns
    ]
    mk = mk[keep]
    mk["team"] = mk["ticker"].str.split("-").str[-1]
    events = []
    for ev, grp in mk.groupby("event_ticker"):
        m = re.match(r".*-(\d{2})([A-Z]{3})(\d{2})", ev)
        if not m or len(grp) != 2:
            continue
        day = datetime.strptime(f"20{m.group(1)}{m.group(2)}{m.group(3)}", "%Y%b%d").date()
        row = match_game(g, day, grp["team"].tolist())
        if row is None:
            events.append({"event_ticker": ev, "game_id": None})
            continue
        home = grp[grp["team"].map(code) == row["home_team"]]
        events.append(
            {
                "event_ticker": ev,
                "game_id": row["game_id"],
                "season": row["season"],
                "kickoff": row["kickoff"],
                "home_market": home["ticker"].iloc[0] if len(home) else None,
                "open_time": grp["open_time"].min(),
            }
        )
    ev = pd.DataFrame(events)
    candles = []
    for r in ev.dropna(subset=["game_id", "home_market"]).itertuples():
        start = int(pd.Timestamp(r.open_time).timestamp())
        end = int(pd.Timestamp(r.kickoff).timestamp())
        q = urllib.parse.urlencode({"start_ts": start, "end_ts": end, "period_interval": 60})
        try:
            c = fetch(
                f"{base}/series/{args.series}/markets/{r.home_market}/candlesticks?{q}",
                args.delay,
                as_json=True,
            )
        except urllib.error.HTTPError:
            try:
                c = fetch(
                    f"{base}/historical/markets/{r.home_market}/candlesticks?{q}",
                    args.delay,
                    as_json=True,
                )
            except urllib.error.HTTPError:
                continue
        for k in c.get("candlesticks", []):
            if k["end_period_ts"] > end:
                continue
            candles.append(
                {
                    "game_id": r.game_id,
                    "market": r.home_market,
                    "end_period_ts": k["end_period_ts"],
                    "yes_bid_close": (k.get("yes_bid") or {}).get("close"),
                    "yes_ask_close": (k.get("yes_ask") or {}).get("close"),
                    "price_close": (k.get("price") or {}).get("close"),
                    "volume": k.get("volume"),
                    "open_interest": k.get("open_interest"),
                }
            )
    cd = pd.DataFrame(candles)
    mk.to_parquet(d / "markets.parquet", index=False)
    ev.to_parquet(d / "events_joined.parquet", index=False)
    cd.to_parquet(d / "home_yes_hourly_pregame.parquet", index=False)
    joined = ev.dropna(subset=["game_id"])
    write_inventory(
        d,
        {
            "source": "kalshi_public_market_data",
            "series": args.series,
            "markets": len(mk),
            "events": len(ev),
            "events_joined": len(joined),
            "join_rate": float(len(joined) / max(len(ev), 1)),
            "games_by_season": joined.groupby("season").size().to_dict() if len(joined) else {},
            "games_with_candles": int(cd["game_id"].nunique()) if len(cd) else 0,
            "candle_rows": len(cd),
            "pregame_filter": "candles with end_period_ts <= repo kickoff only; market result field not stored",
        },
    )


def run_polymarket(args):
    d = out_dir("polymarket")
    g = repo_games()
    events = []
    for selector in [{"series_id": 10187}, {"series_id": 1}, {"tag_slug": "nfl"}]:
        offset = 0
        while True:
            q = urllib.parse.urlencode({**selector, "limit": 50, "offset": offset})
            try:
                r = fetch(f"https://gamma-api.polymarket.com/events?{q}", args.delay, as_json=True)
            except urllib.error.HTTPError:
                break
            if not r:
                break
            events += r
            offset += len(r)
    seen = set()
    events = [e for e in events if not (e.get("slug") in seen or seen.add(e.get("slug")))]
    rows = []
    for e in events:
        m = re.fullmatch(r"nfl-([a-z]+)-([a-z]+)-(\d{4}-\d{2}-\d{2})", e.get("slug", ""))
        if not m:
            continue
        day = datetime.strptime(m.group(3), "%Y-%m-%d").date()
        row = match_game(g, day, [m.group(1), m.group(2)])
        mkts = [x for x in e.get("markets", []) if x.get("slug") == e["slug"]] or e.get(
            "markets", []
        )[:1]
        if not mkts:
            continue
        mkt = mkts[0]
        toks = json.loads(mkt.get("clobTokenIds") or "[]")
        outs = json.loads(mkt.get("outcomes") or "[]")
        rows.append(
            {
                "slug": e["slug"],
                "created": e.get("createdAt"),
                "game_id": None if row is None else row["game_id"],
                "season": None if row is None else row["season"],
                "kickoff": None if row is None else row["kickoff"],
                "token0": toks[0] if toks else None,
                "outcome0": outs[0] if outs else None,
            }
        )
    ev = pd.DataFrame(rows)
    hist = []
    for r in ev.dropna(subset=["game_id", "token0"]).itertuples():
        end = int(pd.Timestamp(r.kickoff).timestamp())
        start = int(pd.Timestamp(r.created).timestamp()) if r.created else end - 14 * 86400
        q = urllib.parse.urlencode(
            {"market": r.token0, "startTs": start, "endTs": end, "fidelity": 60}
        )
        try:
            h = fetch(f"https://clob.polymarket.com/prices-history?{q}", args.delay, as_json=True)
        except urllib.error.HTTPError:
            q = urllib.parse.urlencode(
                {"market": r.token0, "startTs": end - 7 * 86400, "endTs": end, "fidelity": 60}
            )
            try:
                h = fetch(
                    f"https://clob.polymarket.com/prices-history?{q}", args.delay, as_json=True
                )
            except urllib.error.HTTPError:
                continue
        for p in h.get("history", []):
            if p["t"] <= end:
                hist.append(
                    {"game_id": r.game_id, "outcome0": r.outcome0, "t": p["t"], "p": p["p"]}
                )
    hd = pd.DataFrame(hist)
    ev.to_parquet(d / "events_joined.parquet", index=False)
    hd.to_parquet(d / "outcome0_hourly_pregame.parquet", index=False)
    joined = ev.dropna(subset=["game_id"])
    write_inventory(
        d,
        {
            "source": "polymarket_public_gamma_clob",
            "nfl_tag_events": len(events),
            "game_events": len(ev),
            "events_joined": len(joined),
            "join_rate": float(len(joined) / max(len(ev), 1)),
            "games_by_season": joined.groupby("season").size().to_dict() if len(joined) else {},
            "games_with_history": int(hd["game_id"].nunique()) if len(hd) else 0,
            "history_rows": len(hd),
            "pregame_filter": "points with t <= repo kickoff only; resolution fields not stored",
        },
    )


def main():
    p = argparse.ArgumentParser(
        description="Pull free public NFL research sources into data/external."
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, first, delay in [
        ("wins", 2009, 3.0),
        ("weather", 2016, 1.0),
        ("elo538", 2009, 2.0),
        ("nfelo", 2009, 2.0),
        ("kalshi", 2009, 0.3),
        ("polymarket", 2009, 0.3),
    ]:
        s = sub.add_parser(name)
        s.add_argument("--first", type=int, default=first)
        s.add_argument("--last", type=int, default=2026)
        s.add_argument("--delay", type=float, default=delay)
        s.add_argument("--series", default="KXNFLGAME")
    a = p.parse_args()
    {
        "wins": run_wins,
        "weather": run_weather,
        "elo538": run_elo538,
        "nfelo": run_nfelo,
        "kalshi": run_kalshi,
        "polymarket": run_polymarket,
    }[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
