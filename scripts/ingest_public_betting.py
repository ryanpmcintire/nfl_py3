from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

CDX_ENDPOINT = "http://web.archive.org/cdx/search/cdx"
WAYBACK_RAW_TEMPLATE = "http://web.archive.org/web/{ts}id_/{url}"
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
WAYBACK_DELAY_SECONDS = 1.0
SNAPSHOT_DIR_RE = re.compile(r"^\d{8}T\d{6}Z$")

ACTIONNETWORK_URL = "https://www.actionnetwork.com/nfl/public-betting"
COVERS_URL = "https://www.covers.com/picks/nfl"

TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "JAC": "JAX",
    "WSH": "WAS",
}


def normalize_team(abbr: str | None) -> str | None:
    if abbr is None:
        return None
    abbr = abbr.strip().upper()
    return TEAM_ALIASES.get(abbr, abbr)


@dataclass
class RateLimiter:
    delay_seconds: float
    _last_request: float | None = field(default=None, init=False)

    def wait(self) -> None:
        if self._last_request is not None:
            elapsed = time.monotonic() - self._last_request
            remaining = self.delay_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request = time.monotonic()


def resolve_snapshot_dir(out_dir: Path, snapshot: str | None) -> Path:

    if snapshot is not None:
        snapshot_dir = out_dir / snapshot
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        return snapshot_dir

    existing = sorted(
        path for path in out_dir.glob("*") if path.is_dir() and SNAPSHOT_DIR_RE.match(path.name)
    )
    if existing:
        return existing[-1]

    new_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = out_dir / new_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir


def _fetch_bytes(url: str, limiter: RateLimiter, *, timeout: int = 30, retries: int = 4) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        limiter.wait()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            print(f"  fetch failed ({attempt + 1}/{retries}) {url}: {error}", file=sys.stderr)
            time.sleep(3.0 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _fetch_text(url: str, limiter: RateLimiter, *, timeout: int = 30) -> str:

    raw = _fetch_bytes(url, limiter, timeout=timeout)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="ignore")


def fetch_cdx(
    url_pattern: str, start_year: int, end_year: int, limiter: RateLimiter
) -> pd.DataFrame:

    query = (
        f"{CDX_ENDPOINT}?url={url_pattern}&output=json&from={start_year}&to={end_year}"
        "&filter=statuscode:200&collapse=timestamp:8"
    )
    raw = _fetch_bytes(query, limiter, timeout=60)
    text = raw.decode("utf-8", errors="ignore")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        time.sleep(5.0)
        raw = _fetch_bytes(query, limiter, timeout=60)
        rows = json.loads(raw.decode("utf-8", errors="ignore"))
    if not rows:
        return pd.DataFrame(columns=["timestamp", "original"])
    header, *data = rows
    frame = pd.DataFrame(data, columns=header)
    frame["capture_ts"] = pd.to_datetime(frame["timestamp"], format="%Y%m%d%H%M%S", utc=True)
    return frame.sort_values("capture_ts").reset_index(drop=True)


_NEXT_DATA_SCRIPT_START_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>\s*')
_NEXT_DATA_INLINE_START_RE = re.compile(r"__NEXT_DATA__\s*=\s*")


def _extract_balanced_json(text: str, open_brace_idx: int) -> str | None:

    if open_brace_idx >= len(text) or text[open_brace_idx] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(open_brace_idx, len(text)):
        char = text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_idx : i + 1]
    return None


def extract_next_data(html: str) -> dict[str, Any] | None:
    for start_re in (_NEXT_DATA_SCRIPT_START_RE, _NEXT_DATA_INLINE_START_RE):
        match = start_re.search(html)
        if match is None:
            continue
        json_text = _extract_balanced_json(html, match.end())
        if json_text is None:
            continue
        try:
            result: dict[str, Any] = json.loads(json_text)
            return result
        except json.JSONDecodeError:
            continue
    return None


def _find_consensus_book_id_era1(init_state: dict[str, Any]) -> str:
    books = init_state.get("booksReducer", {})
    consensus = books.get("consensus")
    return str(consensus) if consensus is not None else "15"


def _find_consensus_book_id_era2(page_props: dict[str, Any]) -> str:
    all_books = page_props.get("allBooks", {})
    for book_id, book in all_books.items():
        if isinstance(book, dict) and book.get("source_name") == "consensus":
            return str(book_id)
    return "15"


def parse_actionnetwork_era1(next_data: dict[str, Any], capture_ts: pd.Timestamp) -> list[dict]:
    page_props = next_data.get("props", {}).get("pageProps", {})
    init_state = page_props.get("initialState")
    if not isinstance(init_state, dict) or "gamesReducer" not in init_state:
        raise KeyError("initialState")
    games = init_state["gamesReducer"].get("games", {})
    book_id = _find_consensus_book_id_era1(init_state)

    rows: list[dict] = []
    for _game_id, game in games.items():
        teams = game.get("teams", [])
        if len(teams) != 2:
            continue
        away_raw, home_raw = teams[0].get("abbr"), teams[1].get("abbr")
        odds = game.get("odds", {}).get(book_id, {}).get("game", {})
        row = {
            "capture_ts": capture_ts,
            "source": "actionnetwork",
            "era": "era1_initial_state",
            "site_game_id": game.get("id"),
            "season": game.get("season"),
            "week": None,
            "status": game.get("status"),
            "away_team_raw": away_raw,
            "home_team_raw": home_raw,
            "away_team": normalize_team(away_raw),
            "home_team": normalize_team(home_raw),
            "start_time_utc": game.get("start_time"),
            "book_id": book_id,
            "spread_home_bet_pct": odds.get("spread_home_public"),
            "spread_away_bet_pct": odds.get("spread_away_public"),
            "spread_home_money_pct": None,
            "spread_away_money_pct": None,
            "ml_home_bet_pct": odds.get("ml_home_public"),
            "ml_away_bet_pct": odds.get("ml_away_public"),
            "ml_home_money_pct": None,
            "ml_away_money_pct": None,
            "total_over_bet_pct": odds.get("total_over_public"),
            "total_under_bet_pct": odds.get("total_under_public"),
            "total_over_money_pct": None,
            "total_under_money_pct": None,
        }
        row["has_any_public_data"] = any(
            row[k] is not None
            for k in (
                "spread_home_bet_pct",
                "spread_away_bet_pct",
                "ml_home_bet_pct",
                "ml_away_bet_pct",
                "total_over_bet_pct",
                "total_under_bet_pct",
            )
        )
        rows.append(row)
    return rows


_SPREAD_SIDE_FIELDS = {
    "spread": (
        "spread_home_bet_pct",
        "spread_away_bet_pct",
        "spread_home_money_pct",
        "spread_away_money_pct",
    ),
    "moneyline": ("ml_home_bet_pct", "ml_away_bet_pct", "ml_home_money_pct", "ml_away_money_pct"),
}


def parse_actionnetwork_era2(next_data: dict[str, Any], capture_ts: pd.Timestamp) -> list[dict]:
    page_props = next_data.get("props", {}).get("pageProps", {})
    if "scoreboardResponse" not in page_props:
        raise KeyError("scoreboardResponse")
    games = page_props["scoreboardResponse"].get("games", [])
    book_id = _find_consensus_book_id_era2(page_props)

    rows: list[dict] = []
    for game in games:
        teams = game.get("teams", [])
        if len(teams) != 2:
            continue
        away_raw, home_raw = teams[0].get("abbr"), teams[1].get("abbr")
        row = {
            "capture_ts": capture_ts,
            "source": "actionnetwork",
            "era": "era2_scoreboard_response",
            "site_game_id": game.get("id"),
            "season": game.get("season"),
            "week": game.get("week"),
            "status": game.get("status"),
            "away_team_raw": away_raw,
            "home_team_raw": home_raw,
            "away_team": normalize_team(away_raw),
            "home_team": normalize_team(home_raw),
            "start_time_utc": game.get("start_time"),
            "book_id": book_id,
            "spread_home_bet_pct": None,
            "spread_away_bet_pct": None,
            "spread_home_money_pct": None,
            "spread_away_money_pct": None,
            "ml_home_bet_pct": None,
            "ml_away_bet_pct": None,
            "ml_home_money_pct": None,
            "ml_away_money_pct": None,
            "total_over_bet_pct": None,
            "total_under_bet_pct": None,
            "total_over_money_pct": None,
            "total_under_money_pct": None,
        }
        market = game.get("markets", {}).get(book_id, {}).get("event", {})
        for market_type, (
            home_bet_k,
            away_bet_k,
            home_money_k,
            away_money_k,
        ) in _SPREAD_SIDE_FIELDS.items():
            for outcome in market.get(market_type, []):
                if outcome.get("period") not in (None, "event"):
                    continue
                side = outcome.get("side")
                bet_info = outcome.get("bet_info", {})
                tickets_pct = bet_info.get("tickets", {}).get("percent")
                money_pct = bet_info.get("money", {}).get("percent")
                if side == "home":
                    row[home_bet_k] = tickets_pct
                    row[home_money_k] = money_pct
                elif side == "away":
                    row[away_bet_k] = tickets_pct
                    row[away_money_k] = money_pct
        for outcome in market.get("total", []):
            if outcome.get("period") not in (None, "event"):
                continue
            side = outcome.get("side")
            bet_info = outcome.get("bet_info", {})
            tickets_pct = bet_info.get("tickets", {}).get("percent")
            money_pct = bet_info.get("money", {}).get("percent")
            if side == "over":
                row["total_over_bet_pct"] = tickets_pct
                row["total_over_money_pct"] = money_pct
            elif side == "under":
                row["total_under_bet_pct"] = tickets_pct
                row["total_under_money_pct"] = money_pct
        row["has_any_public_data"] = any(
            row[k] is not None
            for k in (
                "spread_home_bet_pct",
                "spread_away_bet_pct",
                "ml_home_bet_pct",
                "ml_away_bet_pct",
                "total_over_bet_pct",
                "total_under_bet_pct",
                "spread_home_money_pct",
                "spread_away_money_pct",
                "ml_home_money_pct",
                "ml_away_money_pct",
                "total_over_money_pct",
                "total_under_money_pct",
            )
        )
        rows.append(row)
    return rows


def parse_actionnetwork_snapshot(
    html: str, capture_ts: pd.Timestamp
) -> tuple[str, list[dict], str | None]:

    next_data = extract_next_data(html)
    if next_data is None:
        return "no_next_data", [], "no __NEXT_DATA__ script found or JSON parse failed"
    try:
        rows = parse_actionnetwork_era2(next_data, capture_ts)
        return "era2_scoreboard_response", rows, None
    except KeyError:
        pass
    try:
        rows = parse_actionnetwork_era1(next_data, capture_ts)
        return "era1_initial_state", rows, None
    except KeyError:
        pass
    return "unrecognized_shape", [], "neither known pageProps shape matched"


def ingest_actionnetwork(
    out_dir: Path,
    start_year: int,
    end_year: int,
    limiter: RateLimiter,
    *,
    force: bool = False,
    save_raw_html: bool = True,
) -> dict[str, Any]:
    an_dir = out_dir / "actionnetwork"
    yearly_dir = an_dir / "yearly"
    raw_html_dir = an_dir / "raw_html"
    yearly_dir.mkdir(parents=True, exist_ok=True)
    if save_raw_html:
        raw_html_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching actionnetwork.com CDX index [{start_year}, {end_year}] ...")
    cdx = fetch_cdx("actionnetwork.com/nfl/public-betting", start_year, end_year, limiter)
    cdx.to_parquet(an_dir / "cdx_index.parquet", index=False)
    print(f"CDX index: {len(cdx)} day-collapsed captures")

    era_counts: dict[str, int] = {}
    fetch_failures: list[str] = []
    parse_failures: list[dict[str, str]] = []
    rows_by_year: dict[str, list[dict]] = {}
    n_with_public_data = 0
    n_games_total = 0

    for _, cdx_row in cdx.iterrows():
        ts = str(cdx_row["timestamp"])
        capture_ts: pd.Timestamp = cdx_row["capture_ts"]
        year = str(capture_ts.year)
        cache_path = raw_html_dir / f"{ts}.html"
        if cache_path.exists() and not force:
            html = cache_path.read_text(encoding="utf-8", errors="ignore")
        else:
            url = WAYBACK_RAW_TEMPLATE.format(ts=ts, url=ACTIONNETWORK_URL)
            try:
                html = _fetch_text(url, limiter)
            except Exception as error:
                print(f"  FETCH FAILED {ts}: {error}", file=sys.stderr)
                fetch_failures.append(ts)
                continue
            if save_raw_html:
                cache_path.write_text(html, encoding="utf-8")

        era, rows, error = parse_actionnetwork_snapshot(html, capture_ts)
        era_counts[era] = era_counts.get(era, 0) + 1
        if error is not None:
            parse_failures.append({"timestamp": ts, "era": era, "error": error})
            print(f"  {ts}: {era} -- {error}")
            continue
        n_games_total += len(rows)
        n_with_public_data += sum(1 for r in rows if r["has_any_public_data"])
        rows_by_year.setdefault(year, []).extend(rows)
        print(
            f"  {ts}: {era}, {len(rows)} games, "
            f"{sum(1 for r in rows if r['has_any_public_data'])} with public data"
        )

    for year, rows in rows_by_year.items():
        frame = pd.DataFrame(rows)
        frame.to_parquet(yearly_dir / f"{year}.parquet", index=False)

    all_yearly = sorted(yearly_dir.glob("*.parquet"))
    if all_yearly:
        combined = pd.concat([pd.read_parquet(p) for p in all_yearly], ignore_index=True)
        combined = combined.sort_values(["capture_ts", "away_team", "home_team"]).reset_index(
            drop=True
        )
        combined.to_parquet(an_dir / "index.parquet", index=False)
        combined.to_parquet(out_dir / "index.parquet", index=False)
    else:
        combined = pd.DataFrame()

    manifest = {
        "source": f"Wayback CDX + raw captures of {ACTIONNETWORK_URL}",
        "fetched_at": datetime.now(UTC).isoformat(),
        "requested_year_range": [start_year, end_year],
        "cdx_captures_found": len(cdx),
        "fetch_failures": fetch_failures,
        "era_counts": era_counts,
        "parse_failures": parse_failures,
        "n_captures_parsed_ok": len(cdx) - len(fetch_failures) - len(parse_failures),
        "n_game_rows_total": len(combined),
        "n_game_rows_with_public_data": n_with_public_data,
        "n_games_total_pre_dedup": n_games_total,
        "cumulative_years_on_disk": len(all_yearly),
        "usage_note": (
            "Private research caching only, matching this project's CFBD/PFT/PFR "
            "precedent (docs/data_feasibility.md License item 6). Never republish "
            "raw rows. actionnetwork.com terms of use were not independently "
            "reviewed this session -- this is a policy stance, not a verified legal fact."
        ),
    }
    with (an_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return manifest


_PERCENT_RE = re.compile(r"[0-9]{1,3}(?:\.[0-9])?%")


def _body_percent_scan(html: str) -> list[str]:
    idx = html.find("<body")
    body = html[idx:] if idx >= 0 else html
    return _PERCENT_RE.findall(body)


def sample_covers(out_dir: Path, sample_n: int, limiter: RateLimiter) -> dict[str, Any]:
    covers_dir = out_dir / "covers"
    sample_dir = covers_dir / "sample_html"
    covers_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching covers.com/picks/nfl CDX index [2018, current] ...")
    cdx = fetch_cdx("covers.com/picks/nfl", 2018, datetime.now(UTC).year, limiter)
    cdx.to_parquet(covers_dir / "cdx_index.parquet", index=False)
    print(f"CDX index: {len(cdx)} day-collapsed captures (no fetch yet)")

    if sample_n <= 0 or cdx.empty:
        summary = {
            "cdx_captures_found": len(cdx),
            "samples_fetched": 0,
            "note": "sample_n<=0 or no captures found; CDX inventory saved, nothing fetched",
        }
        with (covers_dir / "verification_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        return summary

    sample_dir.mkdir(parents=True, exist_ok=True)
    years = sorted(cdx["capture_ts"].dt.year.unique())
    per_year_n = max(1, sample_n // max(1, len(years)))
    picked_rows = []
    for year in years:
        year_rows = cdx.loc[cdx["capture_ts"].dt.year == year]
        step = max(1, len(year_rows) // per_year_n)
        picked_rows.append(year_rows.iloc[::step].head(per_year_n))
    picked = pd.concat(picked_rows, ignore_index=True) if picked_rows else cdx.head(sample_n)

    results = []
    for _, cdx_row in picked.iterrows():
        ts = str(cdx_row["timestamp"])
        url = WAYBACK_RAW_TEMPLATE.format(ts=ts, url=COVERS_URL)
        try:
            html = _fetch_text(url, limiter)
        except Exception as error:
            print(f"  FETCH FAILED {ts}: {error}", file=sys.stderr)
            results.append({"timestamp": ts, "fetch_failed": str(error)})
            continue
        (sample_dir / f"{ts}.html").write_text(html, encoding="utf-8")
        pcts = _body_percent_scan(html)
        results.append(
            {
                "timestamp": ts,
                "html_bytes": len(html),
                "body_percent_matches": len(pcts),
                "sample_matches": pcts[:5],
            }
        )
        print(f"  {ts}: {len(html)} bytes, {len(pcts)} body-region '%' matches")

    summary = {
        "cdx_captures_found": len(cdx),
        "samples_fetched": len(results),
        "results": results,
        "finding": (
            "MEASURED this session: covers.com/picks/nfl is a community/handicapper "
            "picks page (win-loss records like '100%'), not a sportsbook bet%/money% "
            "consensus page. Its real per-game content, and that of the linked "
            "contests.covers.com/consensus/topconsensus/nfl/overall page (also sampled, "
            "a 2016 archive), is populated by client-side AJAX after page load. A "
            "body-only regex scan of the sampled snapshots below found no genuine "
            "per-game percentages -- matches traced to Bootstrap grid CSS column widths "
            "or CSS keyframe background-position percentages inside a <style> tag placed "
            "in <body> by the page's component framework. This corrects "
            "docs/archive/data_source_scout_v3.md's claim of 'real moneyline consensus % "
            "confirmed in a 2023-08-19 snapshot' -- that exact snapshot was re-fetched "
            "this session and contains no such data in its server-rendered HTML."
        ),
    }
    with (covers_dir / "verification_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


def build_coverage_report(out_dir: Path, schedule_path: Path) -> dict[str, Any]:
    index_path = out_dir / "index.parquet"
    if not index_path.exists():
        raise SystemExit(f"{index_path} not found -- run actionnetwork ingestion first")
    parsed = pd.read_parquet(index_path)
    schedule_full = pd.read_parquet(schedule_path)
    schedule_full = schedule_full[
        ["game_id", "season", "week", "game_type", "away_team", "home_team", "kickoff"]
    ].copy()
    schedule_full["kickoff"] = pd.to_datetime(schedule_full["kickoff"], utc=True)
    schedule = schedule_full.loc[schedule_full["game_type"] == "REG"].copy()

    merged = parsed.merge(
        schedule, on=["away_team", "home_team"], how="left", suffixes=("", "_sched")
    )
    merged["start_time_utc"] = pd.to_datetime(merged["start_time_utc"], utc=True, errors="coerce")
    merged["kickoff_delta_hours"] = (
        merged["kickoff"] - merged["start_time_utc"]
    ).dt.total_seconds().abs() / 3600.0
    matched = merged.loc[merged["kickoff_delta_hours"] <= 72].copy()
    matched = matched.sort_values("kickoff_delta_hours").drop_duplicates(
        subset=["capture_ts", "site_game_id", "game_id"]
    )

    kickoff_et = matched["kickoff"].dt.tz_convert("US/Eastern")
    days_since_tuesday = (kickoff_et.dt.weekday - 1) % 7
    tuesday_date_et = kickoff_et.dt.normalize() - pd.to_timedelta(days_since_tuesday, unit="D")
    tuesday_noon_et = tuesday_date_et + pd.Timedelta(hours=12)
    matched["tuesday_noon_utc"] = tuesday_noon_et.dt.tz_convert("UTC")
    matched["reading_before_tuesday_noon"] = matched["capture_ts"] <= matched["tuesday_noon_utc"]
    matched["reading_before_kickoff"] = matched["capture_ts"] < matched["kickoff"]

    per_season = []
    for season, group in matched.groupby("season"):
        season_schedule = schedule.loc[schedule["season"] == season]
        pregame = group.loc[group["reading_before_kickoff"]]
        games_with_pregame_reading = pregame["game_id"].nunique()
        games_before_tuesday = pregame.loc[
            pregame["reading_before_tuesday_noon"], "game_id"
        ].nunique()
        games_after_tuesday_only = (
            pregame.groupby("game_id")["reading_before_tuesday_noon"].any().eq(False).sum()
        )
        per_season.append(
            {
                "season": int(season),
                "distinct_captures": int(group["capture_ts"].nunique()),
                "reg_season_games_in_schedule": len(season_schedule),
                "games_with_ge1_pregame_reading": int(games_with_pregame_reading),
                "games_with_reading_before_tuesday_noon": int(games_before_tuesday),
                "games_with_reading_after_tuesday_noon_only": int(games_after_tuesday_only),
            }
        )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "schedule_path": str(schedule_path),
        "n_parsed_rows_all": len(parsed),
        "n_parsed_rows_reg_season_matched_within_72h": len(matched),
        "n_parsed_rows_not_reg_season_match": int(len(parsed) - len(matched)),
        "unmatched_note": (
            "Unmatched rows are mostly genuine non-REG content actionnetwork's page "
            "itself served, not join failures: preseason games (August 'scheduled' "
            "rows, before Week 1) and stale playoff reruns (the same just-finished "
            "Wild Card matchup re-served as 'complete' for weeks into the following "
            "offseason -- measured this session in 2019 Jan-Apr captures) are both "
            "excluded from the REG-only schedule this report joins against, by design."
        ),
        "per_season": sorted(per_season, key=lambda r: r["season"]),
        "note": (
            "'games_with_reading_after_tuesday_noon_only' counts games whose only "
            "captured public-betting reading(s) landed after that game's own-week "
            "Tuesday noon ET -- per project convention (docs, MEMORY 'Picks lock at "
            "kickoff'), picks stay editable to kickoff even though the LINE freezes "
            "Tuesday, so these late readings are playable, not discarded."
        ),
    }
    with (out_dir / "coverage_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw/public_betting"))
    parser.add_argument(
        "--snapshot",
        default=None,
        metavar="YYYYMMDDTHHMMSSZ",
        help="Timestamped snapshot subdirectory under --out to write/resume.",
    )
    parser.add_argument("--an-start-year", type=int, default=2018)
    parser.add_argument("--an-end-year", type=int, default=datetime.now(UTC).year)
    parser.add_argument("--covers-sample-n", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Re-fetch captures already on disk")
    parser.add_argument(
        "--skip-actionnetwork", action="store_true", help="Skip the actionnetwork.com pull"
    )
    parser.add_argument("--skip-covers", action="store_true", help="Skip the covers.com sample")
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Skip all fetching; just (re)build coverage_report.json from an existing snapshot",
    )
    parser.add_argument(
        "--schedule-path", type=Path, default=Path("data/processed/game_features.parquet")
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    snapshot_dir = resolve_snapshot_dir(args.out, args.snapshot)
    print(f"Snapshot directory: {snapshot_dir}")
    limiter = RateLimiter(WAYBACK_DELAY_SECONDS)

    if args.coverage_only:
        report = build_coverage_report(snapshot_dir, args.schedule_path)
        print(json.dumps(report, indent=2, default=str))
        return

    if not args.skip_actionnetwork:
        manifest = ingest_actionnetwork(
            snapshot_dir,
            args.an_start_year,
            args.an_end_year,
            limiter,
            force=args.force,
        )
        print(json.dumps(manifest, indent=2, default=str))

    if not args.skip_covers:
        summary = sample_covers(snapshot_dir, args.covers_sample_n, limiter)
        print(json.dumps(summary, indent=2, default=str))

    if (snapshot_dir / "index.parquet").exists() and args.schedule_path.exists():
        report = build_coverage_report(snapshot_dir, args.schedule_path)
        print(json.dumps(report, indent=2, default=str))

    top_manifest = {
        "snapshot_dir": str(snapshot_dir),
        "built_at": datetime.now(UTC).isoformat(),
        "actionnetwork_skipped": args.skip_actionnetwork,
        "covers_skipped": args.skip_covers,
    }
    with (snapshot_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(top_manifest, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
