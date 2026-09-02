"""Budgeted pilot pull of The Odds API historical player-prop lines.

Follow-up: backlog rank 3 in `docs/archive/data_source_scout_v3.md` ("The Odds API
historical player-prop lines") -- read that section first for the verified
endpoints, per-call cost, and the 2023-05-03T05:30:00Z props-availability
floor this script's `--start-season` default respects.

Mechanism: a player's own prop line (e.g. passing yards O/U) is a market
read on that player's expected role/health, distinct from and possibly
timed differently than the point-spread market. Sudden appearance,
disappearance, or a sharply reduced line for a specific player is a
real-world technique bettors use to infer starter status before an
official announcement.

Point-in-time note: `snapshot_requested_at_utc` is the timestamp this script
asked for (Tuesday noon UTC each NFL week); `snapshot_actual_at_utc` is the
timestamp The Odds API's historical envelope actually returned data for (the
nearest snapshot at-or-before the request -- may differ by minutes). Every
row also carries a per-bookmaker, per-market `bookmaker_last_update_utc`
straight from the provider, which is the strongest point-in-time evidence
available (per-second granularity).

Endpoints used (both under
https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/):
  - `events?date=<ISO>`                          cost: 1 request/call
  - `events/{id}/odds?markets=...&regions=us&date=<ISO>`
                                                   cost: 10 x markets/call

Credential lookup mirrors `scripts/odds_capture.ps1`: process environment
first, then the Windows per-user registry environment
(HKEY_CURRENT_USER\\Environment). The key value is NEVER printed or logged;
only its presence/length is reported, and every logged URL has `apiKey`
redacted.

Budget policy (set from one live quota-check call, applied by the caller
via --budget; see docs/player_props_sourcing.md for the exact tier table
this session used): this script enforces whatever --budget and
--quota-floor are passed, tracking spend from the `x-requests-remaining`
response header after every call, and stops immediately once the next
call would exceed --budget or take remaining quota below --quota-floor.

Pull order: `--markets` (default `player_pass_yds`) across `--seasons` in
the order given (default `2024,2025,2023`), one Tuesday-noon-UTC snapshot
per NFL regular-season week. If `--markets` lists more than one market they
are pulled together in a single per-event HTTP call (cost is additive
either way; combining saves request *count*, not credits). A second
`--phase-b-markets` list (default `player_rush_yds`) is only pulled, per
already-completed week, if the primary pass finishes every week in
`--seasons` with budget still available (see `run_pilot`).

Writes under --out/<UTC snapshot>/ (default
data/raw/odds_api_props/<UTC timestamp>/, gitignored under this repo's
existing data/raw/** rule):

    <snapshot>/weekly/<season>_wk<NN>_<markets>.parquet   one row per
                                                            (event, bookmaker,
                                                            market, player,
                                                            side).
    <snapshot>/index.parquet         concatenation of every weekly file.
    <snapshot>/manifest.json         quota ledger + per-week coverage.

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_player_props.py \\
        --out data/raw/odds_api_props --budget 4000 --quota-floor 1200

    # Quota-check only (spends exactly 1 request, no props pulled):
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_player_props.py \\
        --quota-check-only
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from datetime import time as dt_time
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.nfl_week import week_cycle_sunday

try:
    import winreg
except ImportError:  # pragma: no cover - non-Windows fallback, unused in this env
    winreg = None  # type: ignore[assignment]

SPORT_KEY = "americanfootball_nfl"
BASE_URL = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}"
EVENTS_URL = f"{BASE_URL}/events"
PROPS_FLOOR_UTC = datetime(2023, 5, 3, 5, 30, tzinfo=UTC)
CREDITS_PER_MARKET_REGION = 10
DEFAULT_MARKETS = ("player_pass_yds",)
DEFAULT_PHASE_B_MARKETS = ("player_rush_yds",)
DEFAULT_SEASON_ORDER = (2024, 2025, 2023)
DEFAULT_QUOTA_FLOOR = 1200
DEFAULT_SLEEP_SECONDS = 0.4
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"

# Same team-name map nfl_ats.market_data.NFL_TEAM_NAMES uses (kept standalone
# here per this script's brief: no src/nfl_ats changes, no new dependency on
# that module -- copied, not imported).
NFL_TEAM_NAMES = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

WEEKLY_COLUMNS = (
    "season",
    "week",
    "event_id",
    "home_team_name",
    "away_team_name",
    "home_team",
    "away_team",
    "nflverse_game_id",
    "commence_time_utc",
    "snapshot_requested_at_utc",
    "snapshot_actual_at_utc",
    "bookmaker_key",
    "bookmaker_title",
    "bookmaker_last_update_utc",
    "market",
    "market_last_update_utc",
    "player_name",
    "outcome_side",
    "line",
    "price",
    "raw_response_sha256",
)


def get_api_key() -> str | None:
    """Mirror scripts/odds_capture.ps1's lookup: process env, then HKCU\\Environment.

    Never returns a printed/logged value -- callers must redact it themselves.
    """

    import os

    key = os.environ.get("THE_ODDS_API_KEY")
    if key:
        return key
    if winreg is None:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
            value, _ = winreg.QueryValueEx(handle, "THE_ODDS_API_KEY")
            return str(value) if value else None
    except OSError:
        return None


def _redact(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted_query = [(k, "***") if k == "apiKey" else (k, v) for k, v in query]
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(redacted_query)))


@dataclass
class RateLimiter:
    delay_seconds: float
    _last: float | None = field(default=None, init=False)

    def wait(self) -> None:
        if self._last is not None:
            elapsed = _time.monotonic() - self._last
            remaining = self.delay_seconds - elapsed
            if remaining > 0:
                _time.sleep(remaining)
        self._last = _time.monotonic()


class ApiAbort(RuntimeError):
    """Raised when a call fails after retries -- caller must stop, not burn quota."""


def _fetch_json(
    url: str,
    params: dict[str, str],
    api_key: str,
    limiter: RateLimiter,
    *,
    timeout: int = 30,
    retries: int = 3,
) -> tuple[dict[str, Any], dict[str, str], bytes]:
    query = dict(params)
    query["apiKey"] = api_key
    full_url = url + "?" + urllib.parse.urlencode(query)
    last_error: Exception | None = None
    for attempt in range(retries):
        limiter.wait()
        request = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                quota = {
                    "requests_remaining": response.headers.get("x-requests-remaining", ""),
                    "requests_used": response.headers.get("x-requests-used", ""),
                    "requests_last": response.headers.get("x-requests-last", ""),
                }
                return json.loads(payload), quota, payload
        except urllib.error.HTTPError as error:
            body = ""
            with contextlib.suppress(Exception):
                body = error.read().decode("utf-8", errors="ignore")[:300]
            last_error = error
            print(
                f"  fetch failed ({attempt + 1}/{retries}) HTTP {error.code} "
                f"{_redact(full_url)} :: {body}",
                file=sys.stderr,
            )
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            print(
                f"  fetch failed ({attempt + 1}/{retries}) {_redact(full_url)} :: {error}",
                file=sys.stderr,
            )
        if attempt < retries - 1:
            _time.sleep(2.0 * (attempt + 1))
    raise ApiAbort(f"{_redact(full_url)} failed after {retries} attempts: {last_error}")


def _parse_remaining(quota: dict[str, str]) -> int | None:
    raw = quota.get("requests_remaining", "")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


# Offset in days to subtract from the week's anchor Sunday to land on a given
# weekday, at noon UTC, within the same Tue..Mon NFL week cycle. Tranche 1
# used Tuesday (offset 5) exclusively; tranche 2 added Saturday (offset 1) to
# test whether a later-in-week snapshot captures a denser props board (see
# docs/player_props_sourcing.md tranche-2 section for the measured result).
SNAPSHOT_WEEKDAY_OFFSET_FROM_SUNDAY = {
    "sunday": 0,
    "monday": 6,
    "tuesday": 5,
    "wednesday": 4,
    "thursday": 3,
    "friday": 2,
    "saturday": 1,
}


@dataclass(frozen=True)
class WeekPlan:
    season: int
    week: int
    snapshot_utc: datetime
    games: pd.DataFrame  # columns: home_team, away_team, gameday, game_id


def build_week_plans(
    schedule: pd.DataFrame, seasons: list[int], snapshot_weekday: str = "tuesday"
) -> list[WeekPlan]:
    """One WeekPlan per (season, week) with a noon-UTC snapshot target.

    `snapshot_weekday` selects which day (within the Tue..Mon NFL week cycle
    anchored on that week's modal Sunday) the noon-UTC snapshot targets.
    Tranche 1 used the default "tuesday"; tranche 2 added "saturday".
    """

    offset_days = SNAPSHOT_WEEKDAY_OFFSET_FROM_SUNDAY[snapshot_weekday]
    plans: list[WeekPlan] = []
    frame = schedule.loc[schedule["season"].isin(seasons)].copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"]).dt.date
    for (season, week), group in frame.groupby(["season", "week"], sort=True):
        cycle_sundays = Counter(week_cycle_sunday(day) for day in group["gameday"])
        anchor = min(cycle_sundays.items(), key=lambda item: (-item[1], item[0]))[0]
        snapshot_local = datetime.combine(
            anchor - timedelta(days=offset_days), dt_time(12, 0), tzinfo=UTC
        )
        if snapshot_local < PROPS_FLOOR_UTC:
            continue
        plans.append(
            WeekPlan(
                season=int(season),
                week=int(week),
                snapshot_utc=snapshot_local,
                games=group[["home_team", "away_team", "gameday", "game_id"]].reset_index(
                    drop=True
                ),
            )
        )
    return plans


def order_plans(plans: list[WeekPlan], season_order: tuple[int, ...]) -> list[WeekPlan]:
    order_index = {season: position for position, season in enumerate(season_order)}
    return sorted(
        (p for p in plans if p.season in order_index),
        key=lambda p: (order_index[p.season], p.week),
    )


def fetch_events_list(
    api_key: str, snapshot_at: datetime, limiter: RateLimiter
) -> tuple[list[dict[str, Any]], dict[str, str], datetime | None, bytes]:
    payload, quota, raw = _fetch_json(
        EVENTS_URL,
        {"date": snapshot_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")},
        api_key,
        limiter,
    )
    timestamp = payload.get("timestamp")
    actual = None
    if timestamp:
        actual = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).astimezone(UTC)
    events = payload.get("data") or []
    if not isinstance(events, list):
        raise ApiAbort(f"events response 'data' was not a list: {type(events)}")
    return events, quota, actual, raw


def fetch_event_odds(
    api_key: str,
    event_id: str,
    snapshot_at: datetime,
    markets: tuple[str, ...],
    limiter: RateLimiter,
) -> tuple[dict[str, Any] | None, dict[str, str], datetime | None, bytes]:
    payload, quota, raw = _fetch_json(
        f"{BASE_URL}/events/{event_id}/odds",
        {
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "dateFormat": "iso",
            "date": snapshot_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        api_key,
        limiter,
    )
    timestamp = payload.get("timestamp")
    actual = None
    if timestamp:
        actual = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).astimezone(UTC)
    data = payload.get("data")
    return (data if isinstance(data, dict) else None), quota, actual, raw


def match_events_to_week(
    events: list[dict[str, Any]], plan: WeekPlan
) -> list[tuple[dict[str, Any], str]]:
    """Filter the API's returned events down to this week's actual schedule,
    matching on (home abbr, away abbr). Returns [(event, nflverse_game_id), ...]."""

    lookup = {
        (row.home_team, row.away_team): row.game_id for row in plan.games.itertuples(index=False)
    }
    matched: list[tuple[dict[str, Any], str]] = []
    for event in events:
        home_abbr = NFL_TEAM_NAMES.get(str(event.get("home_team", "")))
        away_abbr = NFL_TEAM_NAMES.get(str(event.get("away_team", "")))
        if home_abbr is None or away_abbr is None:
            continue
        game_id = lookup.get((home_abbr, away_abbr))
        if game_id is not None:
            matched.append((event, game_id))
    return matched


def select_earliest_kickoff_events(
    matched: list[tuple[dict[str, Any], str]],
) -> list[tuple[dict[str, Any], str]]:
    """Keep every event tied for this week's earliest kickoff.

    The Odds API events response is not assumed to be in kickoff order. A
    missing or invalid kickoff fails closed before an odds request is spent.
    """

    if not matched:
        return []
    parsed: list[tuple[datetime, dict[str, Any], str]] = []
    for event, game_id in matched:
        raw = event.get("commence_time")
        try:
            kickoff = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(UTC)
        except (TypeError, ValueError) as error:
            raise ApiAbort(
                f"Matched event {event.get('id')} has invalid commence_time={raw!r}"
            ) from error
        parsed.append((kickoff, event, game_id))
    earliest = min(item[0] for item in parsed)
    return [(event, game_id) for kickoff, event, game_id in parsed if kickoff == earliest]


def normalize_event_odds(
    event_data: dict[str, Any],
    *,
    season: int,
    week: int,
    nflverse_game_id: str,
    snapshot_requested: datetime,
    snapshot_actual: datetime | None,
    raw_sha: str,
) -> pd.DataFrame:
    home_name = str(event_data.get("home_team", ""))
    away_name = str(event_data.get("away_team", ""))
    rows: list[dict[str, Any]] = []
    for bookmaker in event_data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description") or outcome.get("player")
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "event_id": event_data.get("id"),
                        "home_team_name": home_name,
                        "away_team_name": away_name,
                        "home_team": NFL_TEAM_NAMES.get(home_name),
                        "away_team": NFL_TEAM_NAMES.get(away_name),
                        "nflverse_game_id": nflverse_game_id,
                        "commence_time_utc": event_data.get("commence_time"),
                        "snapshot_requested_at_utc": snapshot_requested,
                        "snapshot_actual_at_utc": snapshot_actual,
                        "bookmaker_key": bookmaker.get("key"),
                        "bookmaker_title": bookmaker.get("title"),
                        "bookmaker_last_update_utc": bookmaker.get("last_update"),
                        "market": market.get("key"),
                        "market_last_update_utc": market.get("last_update"),
                        "player_name": player_name,
                        "outcome_side": outcome.get("name"),
                        "line": outcome.get("point"),
                        "price": outcome.get("price"),
                        "raw_response_sha256": raw_sha,
                    }
                )
    frame = pd.DataFrame(rows, columns=WEEKLY_COLUMNS)
    for column in ("commence_time_utc", "snapshot_requested_at_utc", "snapshot_actual_at_utc"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    for column in ("line", "price"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def budget_from_quota(remaining: int) -> tuple[int, str]:
    """AGENTS.md-independent budgeting rule from this task's brief.

    remaining >= 8000 -> spend up to 4000
    3000 <= remaining < 8000 -> spend up to (remaining - 1500)
    remaining < 3000 -> minimal pilot, spend up to 350
    """

    if remaining >= 8000:
        return 4000, "high (remaining>=8000): budget=4000"
    if remaining >= 3000:
        budget = remaining - 1500
        return budget, f"medium (3000<=remaining<8000): budget=remaining-1500={budget}"
    return 350, "low (remaining<3000): minimal pilot, budget=350"


@dataclass
class Ledger:
    requests_spent_total: int = 0
    remaining: int | None = None
    per_week: list[dict[str, Any]] = field(default_factory=list)
    calls_log: list[str] = field(default_factory=list)

    def record_call(self, kind: str, quota: dict[str, str], note: str = "") -> int:
        used = 0
        try:
            used = int(float(quota.get("requests_last", "0") or "0"))
        except (TypeError, ValueError):
            used = 0
        self.requests_spent_total += used
        remaining = _parse_remaining(quota)
        if remaining is not None:
            self.remaining = remaining
        self.calls_log.append(f"{kind} cost={used} remaining={remaining} {note}".strip())
        return used


def run_pilot(
    *,
    api_key: str,
    out_dir: Path,
    seasons: tuple[int, ...],
    primary_markets: tuple[str, ...],
    phase_b_markets: tuple[str, ...],
    quota_floor: int,
    explicit_budget: int | None,
    sleep_seconds: float,
    quota_check_only: bool,
    snapshot_weekday: str = "tuesday",
    skip_existing: bool = True,
    earliest_kickoff_only: bool = False,
) -> dict[str, Any]:
    limiter = RateLimiter(sleep_seconds)
    weekly_dir = out_dir / "weekly"
    weekly_dir.mkdir(parents=True, exist_ok=True)

    print("Loading nflverse schedules (free, no quota cost)...")
    import nflreadpy as nfl

    schedule_pl = nfl.load_schedules(seasons=list(seasons))
    schedule = schedule_pl.filter(schedule_pl["game_type"] == "REG").to_pandas()
    plans = order_plans(build_week_plans(schedule, list(seasons), snapshot_weekday), seasons)
    if not plans:
        raise SystemExit("No regular-season weeks found for the requested seasons")

    ledger = Ledger()

    # Resume/skip guard: if this snapshot dir already has a weekly file for a
    # given (season, week, markets) combo -- e.g. a prior invocation of this
    # same --snapshot was interrupted -- skip it at zero request cost rather
    # than re-fetching and overwriting. Does not apply across different
    # --snapshot directories (each snapshot is independent by construction).
    def existing_summary(plan: WeekPlan, markets: tuple[str, ...]) -> dict[str, Any] | None:
        if not skip_existing:
            return None
        market_tag = "+".join(markets)
        out_path = weekly_dir / f"{plan.season}_wk{plan.week:02d}_{market_tag}.parquet"
        if not out_path.exists():
            return None
        existing = pd.read_parquet(out_path)
        return {
            "season": plan.season,
            "week": plan.week,
            "markets": list(markets),
            "snapshot_utc": plan.snapshot_utc.isoformat(),
            "events_returned_by_api": None,
            "events_matched_to_week": None,
            "events_pulled": None,
            "rows": len(existing),
            "players_distinct": int(existing["player_name"].nunique()) if len(existing) else 0,
            "bookmakers_distinct": int(existing["bookmaker_key"].nunique()) if len(existing) else 0,
            "status": "skipped_existing",
            "file": str(out_path),
        }

    # The very first plan's events-list call is always spent live (never
    # skipped), because it doubles as this run's real quota-check reading --
    # the caller needs a fresh x-requests-remaining regardless of whether
    # week 1 itself is already on disk from a prior interrupted run.
    first_plan = plans[0]
    print(
        f"Quota-check call (doubles as week 1's events list): season {first_plan.season} "
        f"week {first_plan.week} at {first_plan.snapshot_utc.isoformat()}"
    )
    events, quota, _actual_ts, _raw = fetch_events_list(api_key, first_plan.snapshot_utc, limiter)
    spent = ledger.record_call(
        "events_list", quota, note=f"season={first_plan.season} week={first_plan.week}"
    )
    remaining_at_start = ledger.remaining
    print(f"Quota measured (remaining): {remaining_at_start}  (this call cost {spent})")

    if remaining_at_start is None:
        raise SystemExit(
            "Could not parse x-requests-remaining from the quota-check response; aborting "
            "without further calls."
        )

    if explicit_budget is not None:
        budget = explicit_budget
        tier_note = f"explicit --budget={explicit_budget} (quota rule bypassed)"
    else:
        budget, tier_note = budget_from_quota(remaining_at_start)
    print(f"Budget tier: {tier_note}")

    if quota_check_only:
        return {
            "quota_check_only": True,
            "remaining_at_start": remaining_at_start,
            "requests_spent_total": ledger.requests_spent_total,
            "budget": budget,
            "budget_tier_note": tier_note,
        }

    # Cache of (season, week) -> matched [(event, game_id), ...] so a phase-B
    # pass over the same weeks never re-spends on the events-list call.
    matched_cache: dict[tuple[int, int], list[tuple[dict[str, Any], str]]] = {}
    matched_total_cache: dict[tuple[int, int], int] = {}

    def spend_would_breach(estimated_cost: int) -> bool:
        over_budget = ledger.requests_spent_total + estimated_cost > budget
        under_floor = (
            ledger.remaining is not None and ledger.remaining - estimated_cost < quota_floor
        )
        return over_budget or under_floor

    def pull_week(
        plan: WeekPlan, events_list: list[dict[str, Any]], markets: tuple[str, ...]
    ) -> dict[str, Any]:
        cache_key = (plan.season, plan.week)
        matched = matched_cache.get(cache_key)
        if matched is None:
            matched_all = match_events_to_week(events_list, plan)
            matched_total_cache[cache_key] = len(matched_all)
            matched = (
                select_earliest_kickoff_events(matched_all)
                if earliest_kickoff_only
                else matched_all
            )
            matched_cache[cache_key] = matched
        rows_frames: list[pd.DataFrame] = []
        events_pulled = 0
        stopped_mid_week = False
        for event, game_id in matched:
            per_event_cost = CREDITS_PER_MARKET_REGION * len(markets)
            if spend_would_breach(per_event_cost):
                stopped_mid_week = True
                break
            event_id = str(event.get("id"))
            event_data, quota, actual, raw = fetch_event_odds(
                api_key, event_id, plan.snapshot_utc, markets, limiter
            )
            market_note = ",".join(markets)
            ledger.record_call(
                "event_odds",
                quota,
                note=(
                    f"season={plan.season} week={plan.week} event={event_id} markets={market_note}"
                ),
            )
            events_pulled += 1
            if event_data is not None:
                raw_sha = sha256(raw).hexdigest()
                frame = normalize_event_odds(
                    event_data,
                    season=plan.season,
                    week=plan.week,
                    nflverse_game_id=game_id,
                    snapshot_requested=plan.snapshot_utc,
                    snapshot_actual=actual,
                    raw_sha=raw_sha,
                )
                if not frame.empty:
                    rows_frames.append(frame)
        combined = (
            pd.concat(rows_frames, ignore_index=True)
            if rows_frames
            else pd.DataFrame(columns=WEEKLY_COLUMNS)
        )
        market_tag = "+".join(markets)
        out_path = weekly_dir / f"{plan.season}_wk{plan.week:02d}_{market_tag}.parquet"
        combined.to_parquet(out_path, index=False)
        summary = {
            "season": plan.season,
            "week": plan.week,
            "markets": list(markets),
            "snapshot_utc": plan.snapshot_utc.isoformat(),
            "events_returned_by_api": len(events_list),
            "events_matched_to_week": matched_total_cache.get(cache_key, len(matched)),
            "events_selected_for_pull": len(matched),
            "events_pulled": events_pulled,
            "rows": len(combined),
            "players_distinct": int(combined["player_name"].nunique()) if len(combined) else 0,
            "bookmakers_distinct": int(combined["bookmaker_key"].nunique()) if len(combined) else 0,
            "status": "partial_budget_stop" if stopped_mid_week else "complete",
            "file": str(out_path),
        }
        ledger.per_week.append(summary)
        print(
            f"  {plan.season} wk{plan.week:02d} [{market_tag}]: "
            f"{events_pulled}/{len(matched)} events, "
            f"{len(combined)} rows, {summary['players_distinct']} players "
            f"(remaining={ledger.remaining}, spent_total={ledger.requests_spent_total}) "
            f"[{summary['status']}]"
        )
        return summary

    # Phase A: primary markets across every week in season order. The first
    # plan's events list was already fetched above (dual-purpose quota check).
    stop_reason = None
    phase_a_complete_weeks: list[WeekPlan] = []
    for index, plan in enumerate(plans):
        skipped = existing_summary(plan, primary_markets)
        if skipped is not None:
            print(
                f"  {plan.season} wk{plan.week:02d} [{'+'.join(primary_markets)}]: "
                f"already on disk, skipping ({skipped['rows']} rows, "
                f"{skipped['players_distinct']} players) [skipped_existing]"
            )
            ledger.per_week.append(skipped)
            phase_a_complete_weeks.append(plan)
            continue
        if index == 0:
            week_events = events
        else:
            if spend_would_breach(1):
                stop_reason = "budget_or_floor_before_events_list"
                break
            week_events, quota, _actual_ts, _raw = fetch_events_list(
                api_key, plan.snapshot_utc, limiter
            )
            ledger.record_call("events_list", quota, note=f"season={plan.season} week={plan.week}")
        summary = pull_week(plan, week_events, primary_markets)
        if summary["status"] == "complete":
            phase_a_complete_weeks.append(plan)
        else:
            stop_reason = "budget_or_floor_mid_week"
            break

    phase_b_ran = False
    if stop_reason is None and phase_a_complete_weeks and phase_b_markets:
        print(
            f"Phase A covered all {len(phase_a_complete_weeks)} planned weeks with budget left; "
            f"attempting phase B markets {phase_b_markets}."
        )
        for plan in phase_a_complete_weeks:
            cache_key = (plan.season, plan.week)
            week_events_cached = matched_cache.get(cache_key)
            if week_events_cached is None:
                continue
            if spend_would_breach(CREDITS_PER_MARKET_REGION * len(phase_b_markets)):
                stop_reason = "budget_or_floor_phase_b"
                break
            # pull_week re-derives `matched` from matched_cache using the same
            # (season, week) key, so passing an empty events_list is safe here.
            phase_b_ran = True
            pull_week(plan, [], phase_b_markets)

    all_weekly = sorted(weekly_dir.glob("*.parquet"))
    if all_weekly:
        index_frame = pd.concat((pd.read_parquet(p) for p in all_weekly), ignore_index=True)
        index_frame = index_frame.sort_values(
            ["season", "week", "market", "nflverse_game_id", "bookmaker_key", "player_name"]
        ).reset_index(drop=True)
        index_frame.to_parquet(out_dir / "index.parquet", index=False)
    else:
        index_frame = pd.DataFrame(columns=WEEKLY_COLUMNS)

    join_rate = float(index_frame["nflverse_game_id"].notna().mean()) if len(index_frame) else None

    manifest = {
        "schema_version": 1,
        "source": "the-odds-api.com historical player-prop endpoints",
        "endpoints": {
            "events_list": f"{EVENTS_URL}?date=<ISO>",
            "event_odds": f"{BASE_URL}/events/<id>/odds?regions=us&markets=<...>&date=<ISO>",
        },
        "props_floor_utc": PROPS_FLOOR_UTC.isoformat(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "credential_source": "process env or HKCU\\Environment THE_ODDS_API_KEY (never logged)",
        "quota": {
            "remaining_measured_at_start": remaining_at_start,
            "remaining_at_end": ledger.remaining,
            "requests_spent_total": ledger.requests_spent_total,
            "budget": budget,
            "budget_tier_note": tier_note,
            "quota_floor": quota_floor,
        },
        "snapshot_weekday": snapshot_weekday,
        "earliest_kickoff_only": earliest_kickoff_only,
        "skip_existing_enabled": skip_existing,
        "seasons_order": list(seasons),
        "primary_markets": list(primary_markets),
        "phase_b_markets": list(phase_b_markets) if phase_b_ran else [],
        "phase_b_ran": phase_b_ran,
        "stop_reason": stop_reason,
        "weeks_planned": len(plans),
        "weeks_written": len(ledger.per_week),
        "join_rate_nflverse_game_id": join_rate,
        "per_week": ledger.per_week,
        "calls_log": ledger.calls_log,
        "files": {
            "index.parquet": {
                "rows": len(index_frame),
                "sha256": sha256((out_dir / "index.parquet").read_bytes()).hexdigest()
                if (out_dir / "index.parquet").exists()
                else None,
            }
        },
        "usage_note": (
            "Private research caching only, matching this project's existing "
            "The Odds API / CFBD precedent (docs/data_feasibility.md). Never "
            "republish raw rows. Prices are American odds; lines are the O/U "
            "yardage/reception threshold."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw/odds_api_props"))
    parser.add_argument(
        "--snapshot",
        default=None,
        metavar="YYYYMMDDTHHMMSSZ",
        help="Timestamped snapshot subdirectory under --out. Default: fresh UTC timestamp.",
    )
    parser.add_argument(
        "--seasons",
        default=",".join(str(s) for s in DEFAULT_SEASON_ORDER),
        help="Comma-separated season order, e.g. 2024,2025,2023",
    )
    parser.add_argument("--markets", default=",".join(DEFAULT_MARKETS))
    parser.add_argument(
        "--phase-b-markets",
        default=",".join(DEFAULT_PHASE_B_MARKETS),
        help="Second market set, only pulled if the primary pass covers every "
        "planned week with budget still available. Pass '' to disable.",
    )
    parser.add_argument(
        "--budget", type=int, default=None, help="Override the quota-derived budget"
    )
    parser.add_argument("--quota-floor", type=int, default=DEFAULT_QUOTA_FLOOR)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument(
        "--quota-check-only",
        action="store_true",
        help="Spend exactly 1 request to read x-requests-remaining, then exit.",
    )
    parser.add_argument(
        "--snapshot-weekday",
        default="tuesday",
        choices=sorted(SNAPSHOT_WEEKDAY_OFFSET_FROM_SUNDAY),
        help="Weekday (within each week's Tue..Mon NFL cycle) to target at noon UTC. "
        "Tranche 1 used the default 'tuesday'; tranche 2 added 'saturday'.",
    )
    parser.add_argument(
        "--earliest-kickoff-only",
        action="store_true",
        help="After matching the full week, request props only for event(s) tied for "
        "the earliest kickoff. This is the quota-safe early-game availability design.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Disable the resume/skip guard and always re-fetch+overwrite a week's file "
        "even if it already exists in this --snapshot dir. Default: skip guard is ON.",
    )
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        print(
            "THE_ODDS_API_KEY not found in process env or HKCU\\Environment. "
            "Stopping without spending any quota.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(f"API key found (length={len(api_key)}, value redacted).")

    seasons = tuple(int(s) for s in args.seasons.split(",") if s.strip())
    primary_markets = tuple(m.strip() for m in args.markets.split(",") if m.strip())
    phase_b_markets = tuple(m.strip() for m in args.phase_b_markets.split(",") if m.strip())

    args.out.mkdir(parents=True, exist_ok=True)
    if args.snapshot:
        snapshot_dir = args.out / args.snapshot
    else:
        snapshot_dir = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Snapshot directory: {snapshot_dir}")

    try:
        manifest = run_pilot(
            api_key=api_key,
            out_dir=snapshot_dir,
            seasons=seasons,
            primary_markets=primary_markets,
            phase_b_markets=phase_b_markets,
            quota_floor=args.quota_floor,
            explicit_budget=args.budget,
            sleep_seconds=args.sleep_seconds,
            quota_check_only=args.quota_check_only,
            snapshot_weekday=args.snapshot_weekday,
            skip_existing=args.skip_existing,
            earliest_kickoff_only=args.earliest_kickoff_only,
        )
    except ApiAbort as error:
        print(f"ABORTING (no further quota will be spent): {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
