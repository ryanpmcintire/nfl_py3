"""Backfill Jeff Sagarin's NFL power ratings via the Wayback Machine.

Follow-up to rank #2 build-next in docs/archive/data_source_scout_v4.md ("Sagarin
ratings via the Wayback Machine" -- section "1." of that doc's VERIFIED TODAY
list). That doc measured (2026-08-20 follow-up session): a CDX query for
`sagarin.com/sports/nflsend.htm` returns dense captures 2010-2026 (the
"current" NFL ratings page, self-labeled e.g. "NFL 2014 through games of
October 13 Monday - Week #6"); this session additionally measured a second,
earlier URL era hosted on USA Today: `www.usatoday.com/sports/sagarin/nfl<YY>.htm`
(one URL per season, YY = two-digit season year), CDX-confirmed live with
statuscode 200 for YY in 98-11 (1998-2011 seasons), after which USA Today's
site restructure orphaned the path (301/404 from ~2012 onward) and Sagarin's
own domain (sagarin.com/sports/nflsend.htm) became the sole "current season"
URL, continuing through 2026-08 per the scout doc.

Two ratings-table formats were measured this session (both fixed-width text
inside <pre> tags, HTML-tag-and-entity noise stripped before parsing):

  ERA A -- sagarin.com/sports/nflsend.htm (measured live 2014, 2010-2026
  CDX-dense): header line reads e.g.
    "NFL 2014 through games of October 13 Monday - Week #6"
  followed by a 4-bracket HOME ADVANTAGE line (one edge per method):
    "HOME ADVANTAGE=[  2.73]  ...  [  2.75]  [  2.75]  [  2.71]"
  and per-team rows carrying THREE method columns after the overall RATING:
  GOLDEN_MEAN, PURE POINTS, ELO_SCORE (each with its own rank).

  ERA B -- www.usatoday.com/sports/sagarin/nfl<YY>.htm (measured live for
  1998 and 2009 season pages): header line reads e.g.
    "Final NFL 2009 through games of 2010 February 7 Sunday - Super Bowl"
  followed by a single-value HOME ADVANTAGE line:
    "HOME ADVANTAGE=  2.96          RATING    W   L   T  SCHEDL(RANK) ..."
  and per-team rows carrying TWO method columns: ELO_CHESS, PURE POINTS.

Both eras share the same core row shape (rank, team name, overall RATING,
W-L-T, SCHEDL rating+rank, vs-top-10 W-L-T, vs-top-16 W-L-T, then a variable
number of "| VALUE RANK" method columns), so one era-tolerant regex handles
team rows in both; the HOME ADVANTAGE line and the trailing method-column
count are what actually differ, and are parsed with a primary/fallback regex
pair. The full-page document repeats the same header/table for pagination
(a "by RATING top-to-bottom" list, then "by division" lists covering the
same 32 teams again) -- this parser dedupes by team name, keeping the FIRST
occurrence (the top-to-bottom ranked list), since values are identical across
every repeat within one capture.

Point-in-time provenance: this parser records BOTH the Wayback capture
timestamp (`capture_ts`, the crawl's own clock -- the true "when was this
publicly knowable" timestamp) and the page's own internal "through games of
..." date label (`header_raw`, `header_season`) as a secondary, self-reported
cross-check. Per AGENTS.md, only `capture_ts` should ever be used as the
as-of time for a downstream pregame feature; the header label can lag or
(rarely) lead the crawl and must not be trusted alone.

Team-name normalization: Sagarin's pages spell out full team names, not
nflverse's 2-3 letter codes, and history includes three relocations plus one
rename family (Oakland Raiders -> Las Vegas Raiders; San Diego Chargers ->
Los Angeles Chargers; St. Louis Rams -> Los Angeles Rams; Washington
Redskins -> Washington Football Team -> Washington Commanders). This script
maps every historical spelling straight to the CURRENT nflverse code (LV,
LAC, LA, WAS respectively) per the task brief; the raw parsed name is kept
in `team_name_raw` for audit.

Rate limiting: web.archive.org has no published Crawl-delay for CDX/capture
fetches encountered this session (robots.txt was not independently re-checked
this run; the injury-news ingestion script's NBC Sports precedent used a
site-specific published Crawl-delay -- no such value was found for archive.org
this session, so a conservative, self-imposed ~1 req/sec limiter is used
here as instructed by the task, not because a site policy demands it).
web.archive.org was observed this session intermittently returning an HTML
"Internet Archive: Temporarily Offline" error page (not a proper HTTP error
in all cases) -- `_fetch` retries with backoff and treats that specific body
as a transient failure alongside real HTTP errors.

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_sagarin_ratings.py \\
        --out data/raw/sagarin --start-season 2010 --end-season 2025

    # Include the deeper (bonus, "cheap to add") USA Today pre-2010 depth:
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_sagarin_ratings.py \\
        --out data/raw/sagarin --start-season 1998 --end-season 2025

    # Resume an interrupted run (same snapshot dir, skips captures already
    # fetched to disk):
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_sagarin_ratings.py \\
        --out data/raw/sagarin --snapshot <UTC timestamp> \\
        --start-season 2010 --end-season 2025

Writes under --out/<snapshot>/ (default data/raw/sagarin/<UTC timestamp>/ --
gitignored by the repository's existing `data/raw/**` rule):

    <snapshot>/pages/<era>/<url_key>/<timestamp>.html   raw fetched HTML
    <snapshot>/captures_log.parquet    one row per (era, url, timestamp)
                                        attempted: fetch/parse status.
    <snapshot>/index.parquet           one row per (capture_ts, team) parsed
                                        rating observation across all
                                        captures.
    <snapshot>/asof_tuesday_view.parquet   one row per (season, week, team):
                                        the latest capture at/before that
                                        week's Tuesday cutoff, joined from
                                        index.parquet, aligned to the
                                        project's own nflverse schedule
                                        snapshot.
    <snapshot>/manifest.json           run metadata + coverage summary.

A manifest.json never sits directly at data/raw/sagarin/manifest.json (it is
nested one level down under the timestamped snapshot dir), matching this
project's existing convention (see scripts/ingest_injury_news.py) so that
`nfl_ats.snapshots.latest_snapshot()` never mistakes it for a schedules
snapshot.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
RATE_LIMIT_SECONDS = 1.0
SNAPSHOT_DIR_RE = re.compile(r"^\d{8}T\d{6}Z$")
CDX_BASE = "https://web.archive.org/cdx/search/cdx"
CAPTURE_BASE = "https://web.archive.org/web"
TRANSIENT_MARKERS = (b"Temporarily Offline", b"Internet Archive: Temporarily")

ERA_SAGARIN_COM = "sagarin_com"
ERA_USATODAY = "usatoday"

# ---------------------------------------------------------------------------
# Team name -> current nflverse code (relocations/renames folded to current
# identity per the task brief).
# ---------------------------------------------------------------------------
NAME_TO_CODE: dict[str, str] = {
    "ARIZONA CARDINALS": "ARI",
    "ATLANTA FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR",
    "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "CLEVELAND BROWNS": "CLE",
    "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN",
    "DETROIT LIONS": "DET",
    "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU",
    "INDIANAPOLIS COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KC",
    "OAKLAND RAIDERS": "LV",
    "LOS ANGELES RAIDERS": "LV",
    "LAS VEGAS RAIDERS": "LV",
    "ST. LOUIS RAMS": "LA",
    "ST LOUIS RAMS": "LA",
    "LOS ANGELES RAMS": "LA",
    "SAN DIEGO CHARGERS": "LAC",
    "LOS ANGELES CHARGERS": "LAC",
    "MIAMI DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE",
    "NEW ORLEANS SAINTS": "NO",
    "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ",
    "PHILADELPHIA EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT",
    "SEATTLE SEAHAWKS": "SEA",
    "SAN FRANCISCO 49ERS": "SF",
    "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN",
    "TENNESSEE OILERS": "TEN",
    "WASHINGTON REDSKINS": "WAS",
    "WASHINGTON FOOTBALL TEAM": "WAS",
    "WASHINGTON COMMANDERS": "WAS",
}


def team_code_for(name_raw: str) -> str | None:
    key = re.sub(r"\s+", " ", name_raw).strip().upper()
    return NAME_TO_CODE.get(key)


# ---------------------------------------------------------------------------
# Rate limiting + fetch
# ---------------------------------------------------------------------------


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


def _fetch(url: str, limiter: RateLimiter, *, timeout: int = 30, retries: int = 4) -> bytes:
    """Fetch via `curl` subprocess, not `urllib`.

    Measured this session: identical CDX GET requests that `urllib.request`
    repeatedly timed out on (multiple 30s "read operation timed out" retries
    against `web.archive.org/cdx/search/cdx`) succeeded via `curl` in ~3.7s
    every time -- consistent with this project's own prior finding
    (docs/archive/data_source_scout_v4.md, Sagarin section: "direct WebFetch to
    web.archive.org URLs fails at the tool level... curl succeeded") that
    something about this Windows environment's Python HTTP stack (not a
    site-side block; the CDX server itself answered curl instantly) hangs on
    this host. `curl` is used for every fetch in this script accordingly.
    """

    last_error: Exception | None = None
    for attempt in range(retries):
        limiter.wait()
        try:
            completed = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-S",
                    "-L",
                    "--compressed",
                    "--max-time",
                    str(timeout),
                    "-A",
                    USER_AGENT,
                    "-w",
                    "\n__CURL_HTTP_CODE__%{http_code}",
                    url,
                ],
                capture_output=True,
                timeout=timeout + 5,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            last_error = error
            print(f"    curl subprocess timeout ({attempt + 1}/{retries}) {url}", file=sys.stderr)
            time.sleep(3.0 * (attempt + 1))
            continue
        if completed.returncode != 0:
            stderr_text = completed.stderr.decode(errors="replace")[:300]
            last_error = RuntimeError(f"curl exit {completed.returncode}: {stderr_text}")
            print(
                f"    fetch failed ({attempt + 1}/{retries}) {url}: {last_error}", file=sys.stderr
            )
            time.sleep(3.0 * (attempt + 1))
            continue
        marker = b"\n__CURL_HTTP_CODE__"
        idx = completed.stdout.rfind(marker)
        if idx == -1:
            last_error = RuntimeError("curl output missing http-code marker")
            time.sleep(3.0 * (attempt + 1))
            continue
        body = completed.stdout[:idx]
        http_code = completed.stdout[idx + len(marker) :].decode(errors="replace").strip()
        if http_code not in ("200", "301", "302"):
            last_error = RuntimeError(f"http status {http_code}")
            print(
                f"    fetch failed ({attempt + 1}/{retries}) {url}: status {http_code}",
                file=sys.stderr,
            )
            time.sleep(3.0 * (attempt + 1))
            continue
        if any(marker in body[:2000] for marker in TRANSIENT_MARKERS):
            last_error = RuntimeError("archive.org transient 'Temporarily Offline' page")
            print(
                f"    transient offline page ({attempt + 1}/{retries}) {url}",
                file=sys.stderr,
            )
            time.sleep(5.0 * (attempt + 1))
            continue
        if not body:
            # Measured this session: two captures ended up permanently cached
            # as 0-byte files (fetch "succeeded" -- HTTP 200, no transient
            # marker -- but curl returned an empty body, most likely from the
            # same connection instability seen throughout this run). An empty
            # cached file is indistinguishable from a real fetch to the
            # `html_path.exists()` resume check, so it would never self-heal
            # on a later run. Treat empty as a retryable failure instead of a
            # zero-length success.
            last_error = RuntimeError("empty response body")
            time.sleep(3.0 * (attempt + 1))
            continue
        return body
    assert last_error is not None
    raise last_error


def cdx_query(url_pattern: str, limiter: RateLimiter, **params: str) -> list[dict[str, str]]:
    """Query the CDX API and collapse consecutive same-digest rows CLIENT-side.

    Measured this session: passing `collapse=digest` to the CDX API itself
    made the *identical* query go from a reliable ~2-9s round trip to a
    consistent 30s+ timeout / connection failure (code 000) for several of
    the lower-traffic `nfl<YY>.htm` URLs -- a server-side cost of the
    collapse operation, not a client (curl vs urllib) issue as first
    suspected. Fetching the full, uncollapsed capture list and collapsing
    identical-content runs locally gets the same deduplication (skip re-
    fetching a frozen page's unchanged re-crawls) without that server load.
    """

    query = {"url": url_pattern, "output": "json", "filter": "statuscode:200"}
    query.update(params)
    qs = "&".join(f"{k}={urllib.parse.quote(str(v), safe='')}" for k, v in query.items())
    raw = _fetch(f"{CDX_BASE}?{qs}", limiter)
    rows = json.loads(raw.decode("utf-8", errors="replace"))
    if not rows:
        return []
    header = rows[0]
    digest_idx = header.index("digest") if "digest" in header else None
    records = [dict(zip(header, row, strict=True)) for row in rows[1:]]
    records.sort(key=lambda r: r["timestamp"])
    if digest_idx is None:
        return records
    collapsed: list[dict[str, str]] = []
    last_digest: str | None = None
    for record in records:
        digest = record.get("digest")
        if digest is not None and digest == last_digest:
            continue
        collapsed.append(record)
        last_digest = digest
    return collapsed


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"[ \t]+")

HOME_ADVANTAGE_4 = re.compile(
    r"HOME ADVANTAGE=\s*\[\s*([\d.]+)\]\s*\[\s*([\d.]+)\]\s*\[\s*([\d.]+)\]\s*\[\s*([\d.]+)\]"
)
# WP19 fix (2026-09-01, docs/sagarin_backfill.md section 9): a transitional
# 3-bracket HOME ADVANTAGE line, measured on both sagarin.com- and
# usatoday.com-domain captures spanning roughly Nov 2011 - Sep 2013 (before
# ELO_SCORE/GOLDEN_MEAN were introduced), one bracket per method -- RATING,
# ELO_CHESS, PURE POINTS in that order (measured: bracket colors #9900ff /
# #ff0000 / #0000ff match the team rows' own RATING / ELO_CHESS / PURE
# POINTS column colors 1:1). Every one of 2012's captures used this format,
# which the pre-fix 4-then-1-bracket pair never matched, so home_edge_rating
# came back null for the entire season. Checked only when HOME_ADVANTAGE_4
# fails to match (that pattern requires all 4 groups, so it never partially
# matches a 3-bracket line).
HOME_ADVANTAGE_3 = re.compile(
    r"HOME ADVANTAGE=\s*\[\s*([\d.]+)\]\s*\[\s*([\d.]+)\]\s*\[\s*([\d.]+)\]"
)
# WP19 fix: one measured capture in this same transitional window (usatoday
# nfl11@20120109071948, "2012 JANUARY 8 SUNDAY - Wild Card Weekend") used a
# third, comma-separated, unbracketed layout instead: "HOME EDGE=  3.04,
# 2.38,  2.74" -- same RATING/ELO_CHESS/PURE POINTS value order (the page's
# own explanatory text reads "THREE home edges listed for: RATING,
# ELO_CHESS, PREDICTOR(PURE POINTS)"), just a different label ("HOME EDGE="
# not "HOME ADVANTAGE=") and separator. Distinct label means no ordering
# dependency versus HOME_ADVANTAGE_3/_4/_1.
HOME_EDGE_COMMA = re.compile(r"HOME EDGE=\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)")
HOME_ADVANTAGE_1 = re.compile(r"HOME ADVANTAGE=\s*([\d.]+)")

# Primary header: covers both "NFL 2014 Ratings through results of ... -
# Week #6" (sagarin.com) and "NFL 2003 Ratings thru results of SUNDAY,
# FEBRUARY 1, 2004 - Super Bowl - FINAL" (usatoday; note trailing "FINAL"
# rather than a leading "Final " prefix, and "thru" instead of "through").
HEADER_RE = re.compile(
    r"(Final\s+)?NFL\s+(\d{4})\s*(?:Ratings\s*)?(?:through|thru)\s+"
    r"(?:results of|games of)\s+([^\n]*)"
)
# Fallback header for the terse earliest era (measured 1998 usatoday sample:
# "Final 1998 NFL ratings", no as-of date or week at all in the header line
# -- capture_ts is the only as-of signal available for this era).
HEADER_TERSE_RE = re.compile(r"(Final\s+)?(\d{4})\s+NFL\s+ratings", re.IGNORECASE)
# Fallback header for a pre-Week-1 "starting ratings" snapshot (measured this
# session, sagarin.com era, 2013-09-10 capture: "NFL 2013 Starting Ratings",
# no games played yet -- every team row is 0-0-0). Distinct from both header
# patterns above since it has neither a "through/thru results-of/games-of"
# date clause nor the terse "<year> NFL ratings" word order.
HEADER_PRESEASON_RE = re.compile(r"NFL\s+(\d{4})\s+Starting\s+Ratings", re.IGNORECASE)
WEEK_RE = re.compile(r"Week\s*#?\s*(\d+)", re.IGNORECASE)
TRAILING_FINAL_RE = re.compile(r"\bFINAL\b", re.IGNORECASE)

# Team row, era-tolerant: captures rank, name, RATING, W, L, T, SCHEDL(rank),
# vs-top10 W-L-T, vs-top16 W-L-T, then an OPTIONAL trailing "| VALUE RANK ..."
# tail whose method-column count varies by era (the earliest 1998 usatoday
# era, measured this session, has no method-breakdown columns at all -- the
# row ends right after vs-top16).
TEAM_ROW_RE = re.compile(
    r"^\s*(\d{1,2})\s+([A-Za-z][A-Za-z0-9.' ]*?)\s*=\s*([\d.]+)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+"
    r"([\d.]+)\(\s*(\d+)\)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s*\|\s*"
    r"(\d+)\s+(\d+)\s+(\d+)"
    r"(?:\s*\|\s*(.+))?$"
)
METHOD_PAIR_RE = re.compile(r"([\d.]+)\s+(\d+)")


def strip_tags(html: str) -> str:
    import html as html_module

    text = TAG_RE.sub("", html)
    return html_module.unescape(text)


@dataclass
class ParsedCapture:
    season: int | None
    is_final: bool
    header_raw: str
    header_week_number: int | None
    home_edge_rating: float | None
    home_edge_methods: list[float]
    team_rows: list[dict[str, object]]
    era_format: str
    parse_error: str | None


def parse_capture_html(raw_html: bytes) -> ParsedCapture:
    text = strip_tags(raw_html.decode("utf-8", errors="replace"))

    header_match = HEADER_RE.search(text)
    if header_match is not None:
        season = int(header_match.group(2))
        header_raw = header_match.group(0).strip()
        is_final = bool(header_match.group(1)) or bool(
            TRAILING_FINAL_RE.search(header_match.group(3) or "")
        )
    else:
        terse_match = HEADER_TERSE_RE.search(text)
        if terse_match is not None:
            season = int(terse_match.group(2))
            header_raw = terse_match.group(0).strip()
            is_final = bool(terse_match.group(1))
        else:
            preseason_match = HEADER_PRESEASON_RE.search(text)
            if preseason_match is not None:
                season = int(preseason_match.group(1))
                header_raw = preseason_match.group(0).strip()
                is_final = False
            else:
                season = None
                header_raw = ""
                is_final = False
    week_match = WEEK_RE.search(header_raw)
    header_week_number = int(week_match.group(1)) if week_match else None

    home4 = HOME_ADVANTAGE_4.search(text)
    home3 = HOME_ADVANTAGE_3.search(text) if home4 is None else None
    home_comma = HOME_EDGE_COMMA.search(text) if home4 is None and home3 is None else None
    home1 = (
        HOME_ADVANTAGE_1.search(text)
        if home4 is None and home3 is None and home_comma is None
        else None
    )
    if home4 is not None:
        home_edge_rating = float(home4.group(1))
        home_edge_methods = [float(home4.group(i)) for i in (2, 3, 4)]
        era_format = ERA_SAGARIN_COM
    elif home3 is not None or home_comma is not None:
        # WP19 fix: both the 3-bracket and comma-separated transitional
        # layouts carry RATING + ELO_CHESS + PURE POINTS (measured above),
        # the same 2-method column shape as the single-value USATODAY era --
        # tagged era_format=ERA_USATODAY so the team-row method-name lookup
        # (below) resolves to "elo_chess"/"pure_points" instead of generic
        # "method_0"/"method_1", matching the section 5.1 precedent that
        # sagarin.com-domain captures already get era_format="usatoday" when
        # their layout is the simpler one. Only home_edge_rating (group 1,
        # RATING's own edge) is kept -- the two per-method values are NOT
        # written into home_edge_golden_mean/home_edge_elo_score, which are
        # a fixed-position mapping elsewhere in this script that assumes
        # GOLDEN_MEAN/PURE_POINTS/ELO_SCORE order; writing ELO_CHESS/PURE
        # POINTS values into those slots would mislabel them, and no
        # downstream consumer (the frozen sagarin_battery_* predeclaration,
        # docs/sagarin_backfill.md section 8) uses anything but
        # home_edge_rating.
        match = home3 if home3 is not None else home_comma
        assert match is not None
        home_edge_rating = float(match.group(1))
        home_edge_methods = []
        era_format = ERA_USATODAY
    elif home1 is not None:
        home_edge_rating = float(home1.group(1))
        home_edge_methods = []
        era_format = ERA_USATODAY
    else:
        home_edge_rating = None
        home_edge_methods = []
        era_format = "unknown"

    seen_teams: set[str] = set()
    team_rows: list[dict[str, object]] = []
    for line in text.splitlines():
        match = TEAM_ROW_RE.match(line)
        if not match:
            continue
        name_raw = re.sub(r"\s+", " ", match.group(2)).strip()
        if name_raw in seen_teams:
            continue
        seen_teams.add(name_raw)
        tail = match.group(15) or ""
        method_pairs = METHOD_PAIR_RE.findall(tail)
        row: dict[str, object] = {
            "rank": int(match.group(1)),
            "team_name_raw": name_raw,
            "team_code": team_code_for(name_raw),
            "rating": float(match.group(3)),
            "wins": int(match.group(4)),
            "losses": int(match.group(5)),
            "ties": int(match.group(6)),
            "schedule_rating": float(match.group(7)),
            "schedule_rank": int(match.group(8)),
            "vs_top10_w": int(match.group(9)),
            "vs_top10_l": int(match.group(10)),
            "vs_top10_t": int(match.group(11)),
            "vs_top16_w": int(match.group(12)),
            "vs_top16_l": int(match.group(13)),
            "vs_top16_t": int(match.group(14)),
        }
        # Method columns: sagarin.com era = GOLDEN_MEAN, PURE_POINTS(PREDICTOR),
        # ELO_SCORE; usatoday era = ELO_CHESS, PURE_POINTS(PREDICTOR).
        if era_format == ERA_SAGARIN_COM:
            method_names = ["golden_mean", "pure_points", "elo_score"]
        elif era_format == ERA_USATODAY:
            method_names = ["elo_chess", "pure_points"]
        else:
            method_names = [f"method_{i}" for i in range(len(method_pairs))]
        for name, pair in zip(method_names, method_pairs, strict=False):
            value, rank = pair
            row[f"{name}_value"] = float(value)
            row[f"{name}_rank"] = int(rank)
        team_rows.append(row)

    parse_error = None
    if season is None:
        parse_error = "no_header_match"
    elif not team_rows:
        parse_error = "no_team_rows"
    elif len(team_rows) < 28:
        parse_error = f"only_{len(team_rows)}_teams"

    return ParsedCapture(
        season=season,
        is_final=is_final,
        header_raw=header_raw,
        header_week_number=header_week_number,
        home_edge_rating=home_edge_rating,
        home_edge_methods=home_edge_methods,
        team_rows=team_rows,
        era_format=era_format,
        parse_error=parse_error,
    )


# ---------------------------------------------------------------------------
# CDX enumeration
# ---------------------------------------------------------------------------


def enumerate_sagarin_com_captures(limiter: RateLimiter) -> list[dict[str, str]]:
    rows = cdx_query("sagarin.com/sports/nflsend.htm", limiter)
    for row in rows:
        row["era"] = ERA_SAGARIN_COM
        row["url_key"] = "nflsend"
    return rows


def enumerate_usatoday_captures(
    limiter: RateLimiter, start_season: int, end_season: int
) -> list[dict[str, str]]:
    all_rows: list[dict[str, str]] = []
    for season in range(start_season, end_season + 1):
        yy = season % 100
        url = f"www.usatoday.com/sports/sagarin/nfl{yy:02d}.htm"
        try:
            rows = cdx_query(url, limiter)
        except Exception as error:
            print(f"  CDX FAILED for {url}: {error}", file=sys.stderr)
            continue
        for row in rows:
            row["era"] = ERA_USATODAY
            row["url_key"] = f"nfl{yy:02d}"
        all_rows.extend(rows)
        print(f"  CDX {url}: {len(rows)} distinct-content captures")
    return all_rows


def enumerate_cached_captures(pages_dir: Path) -> list[dict[str, str]]:
    """Rebuild a captures list purely from already-cached HTML on disk.

    WP19 addition (2026-09-01): a parser fix (HOME_ADVANTAGE_3/HOME_EDGE_COMMA,
    docs/sagarin_backfill.md section 9) needs the alignment view rebuilt with
    the SAME set of captures already fetched -- not a fresh CDX enumeration,
    which would silently pull in new captures Wayback has crawled since the
    original run and conflate "parser got better" with "archive got denser"
    in the before/after comparison. No network access; `digest` is unknown
    from disk alone so it is left None (only used for CDX-side de-dup, not by
    anything downstream of captures_log.parquet).
    """

    rows: list[dict[str, str]] = []
    for era_dir in sorted(p for p in pages_dir.glob("*") if p.is_dir()):
        era = era_dir.name
        for url_key_dir in sorted(p for p in era_dir.glob("*") if p.is_dir()):
            url_key = url_key_dir.name
            if era == ERA_SAGARIN_COM and url_key == "nflsend":
                original = "sagarin.com/sports/nflsend.htm"
            elif era == ERA_USATODAY:
                original = f"www.usatoday.com/sports/sagarin/{url_key}.htm"
            else:
                original = f"{era}/{url_key}"
            for html_path in sorted(url_key_dir.glob("*.html")):
                rows.append(
                    {
                        "era": era,
                        "url_key": url_key,
                        "timestamp": html_path.stem,
                        "original": original,
                        "digest": None,
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Ingestion driver
# ---------------------------------------------------------------------------


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


def ingest(
    snapshot_dir: Path,
    start_season: int,
    end_season: int,
    *,
    include_usatoday: bool,
    limiter: RateLimiter,
    max_captures: int | None,
    reparse_cache_only: bool = False,
) -> dict[str, object]:
    pages_dir = snapshot_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    captures: list[dict[str, str]] = []
    if reparse_cache_only:
        # WP19: reparse the existing on-disk cache with a fixed parser --
        # zero network calls, same capture set as the run being corrected.
        print("Reparse-cache-only mode: skipping CDX, scanning pages/ on disk...")
        captures.extend(enumerate_cached_captures(pages_dir))
        print(f"  found {len(captures)} cached captures on disk")
    else:
        print("Enumerating CDX captures...")
        if end_season >= 2010:
            captures.extend(enumerate_sagarin_com_captures(limiter))
            print(f"  sagarin.com/sports/nflsend.htm: {len(captures)} distinct-content captures")
        if include_usatoday:
            captures.extend(
                enumerate_usatoday_captures(limiter, start_season, min(end_season, 2011))
            )

    captures.sort(key=lambda r: r["timestamp"])
    if max_captures is not None:
        captures = captures[:max_captures]
    print(f"Total captures to process: {len(captures)}")

    log_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []

    for i, cap in enumerate(captures):
        era = cap["era"]
        url_key = cap["url_key"]
        timestamp = cap["timestamp"]
        original = cap["original"]
        page_dir = pages_dir / era / url_key
        page_dir.mkdir(parents=True, exist_ok=True)
        html_path = page_dir / f"{timestamp}.html"

        capture_ts = datetime.strptime(timestamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC)

        status: dict[str, object] = {
            "era": era,
            "url_key": url_key,
            "timestamp": timestamp,
            "capture_ts": capture_ts.isoformat(),
            "original_url": original,
            "digest": cap.get("digest"),
        }

        try:
            if html_path.exists():
                raw = html_path.read_bytes()
                status["fetch_status"] = "cached"
            else:
                fetch_url = f"{CAPTURE_BASE}/{timestamp}id_/{original}"
                raw = _fetch(fetch_url, limiter)
                html_path.write_bytes(raw)
                status["fetch_status"] = "fetched"
        except Exception as error:
            status["fetch_status"] = "failed"
            status["error"] = str(error)
            log_rows.append(status)
            print(f"  [{i + 1}/{len(captures)}] FETCH FAILED {era}/{url_key}@{timestamp}: {error}")
            continue

        try:
            parsed = parse_capture_html(raw)
        except Exception as error:
            status["fetch_status"] = status.get("fetch_status", "fetched")
            status["parse_status"] = "exception"
            status["error"] = str(error)
            log_rows.append(status)
            print(
                f"  [{i + 1}/{len(captures)}] PARSE EXCEPTION {era}/{url_key}@{timestamp}: {error}"
            )
            continue

        status["season"] = parsed.season
        status["is_final"] = parsed.is_final
        status["header_raw"] = parsed.header_raw
        status["header_week_number"] = parsed.header_week_number
        status["era_format"] = parsed.era_format
        status["home_edge_rating"] = parsed.home_edge_rating
        status["n_teams_parsed"] = len(parsed.team_rows)
        status["n_teams_mapped"] = sum(
            1 for r in parsed.team_rows if r.get("team_code") is not None
        )
        status["parse_status"] = "ok" if parsed.parse_error is None else parsed.parse_error
        log_rows.append(status)

        if parsed.season is not None:
            for row in parsed.team_rows:
                index_rows.append(
                    {
                        "capture_ts": capture_ts,
                        "era": era,
                        "era_format": parsed.era_format,
                        "url_key": url_key,
                        "season": parsed.season,
                        "is_final": parsed.is_final,
                        "header_raw": parsed.header_raw,
                        "header_week_number": parsed.header_week_number,
                        "home_edge_rating": parsed.home_edge_rating,
                        "home_edge_golden_mean": (
                            parsed.home_edge_methods[0]
                            if len(parsed.home_edge_methods) > 0
                            else None
                        ),
                        "home_edge_pure_points": (
                            parsed.home_edge_methods[1]
                            if len(parsed.home_edge_methods) > 1
                            else None
                        ),
                        "home_edge_elo_score": (
                            parsed.home_edge_methods[2]
                            if len(parsed.home_edge_methods) > 2
                            else None
                        ),
                        **row,
                    }
                )

        if (i + 1) % 25 == 0 or i == len(captures) - 1:
            print(
                f"  [{i + 1}/{len(captures)}] {era}/{url_key}@{timestamp} "
                f"status={status['parse_status']} season={parsed.season} "
                f"teams={len(parsed.team_rows)}"
            )

    log_frame = pd.DataFrame(log_rows)
    log_frame.to_parquet(snapshot_dir / "captures_log.parquet", index=False)

    index_frame = pd.DataFrame(index_rows)
    if not index_frame.empty:
        index_frame = index_frame.sort_values(["capture_ts", "team_code"]).reset_index(drop=True)
    index_frame.to_parquet(snapshot_dir / "index.parquet", index=False)

    manifest = {
        "source": (
            "Jeff Sagarin NFL ratings via web.archive.org Wayback Machine "
            "(sagarin.com/sports/nflsend.htm 2010-present; "
            "www.usatoday.com/sports/sagarin/nfl<YY>.htm 1998-2011)"
        ),
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "start_season_requested": start_season,
        "end_season_requested": end_season,
        "include_usatoday_era": include_usatoday,
        "reparse_cache_only": reparse_cache_only,
        "rate_limit_seconds": RATE_LIMIT_SECONDS,
        "user_agent": USER_AGENT,
        "captures_attempted": len(captures),
        "captures_fetch_ok": int((log_frame["fetch_status"].isin(["fetched", "cached"])).sum())
        if not log_frame.empty
        else 0,
        "captures_fetch_failed": int((log_frame["fetch_status"] == "failed").sum())
        if not log_frame.empty
        else 0,
        "captures_parse_ok": int((log_frame["parse_status"] == "ok").sum())
        if not log_frame.empty
        else 0,
        "index_rows": len(index_frame),
        "usage_note": (
            "Private research caching only, matching this project's existing "
            "Wayback/nflverse precedent. Never republish raw rows. Sagarin's "
            "content ownership and Wayback's own terms were not independently "
            "reviewed this session -- policy stance, not a verified legal fact."
        ),
    }
    with (snapshot_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest


# ---------------------------------------------------------------------------
# As-of-Tuesday alignment view
# ---------------------------------------------------------------------------


ET = ZoneInfo("America/New_York")


def _previous_or_same_tuesday(d: date) -> date:
    # Monday=0 ... Sunday=6; Tuesday=1
    days_back = (d.weekday() - 1) % 7
    return d - timedelta(days=days_back)


def build_week_windows(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week): first-kickoff UTC timestamp + the
    at-or-before Tuesday cutoff used for the as-of-Tuesday alignment view."""

    frame = schedules.dropna(subset=["gameday", "week"]).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    frame["gametime"] = frame["gametime"].fillna("13:00")

    def _kickoff_utc(row: pd.Series) -> pd.Timestamp:
        try:
            hour, minute = (int(x) for x in str(row["gametime"]).split(":")[:2])
        except ValueError:
            hour, minute = 13, 0
        local = datetime.combine(row["gameday"].date(), datetime.min.time()).replace(
            hour=hour, minute=minute, tzinfo=ET
        )
        return pd.Timestamp(local.astimezone(UTC))

    frame["kickoff_utc"] = frame.apply(_kickoff_utc, axis=1)
    windows = (
        frame.groupby(["season", "week"], as_index=False)
        .agg(first_kickoff_utc=("kickoff_utc", "min"), last_kickoff_utc=("kickoff_utc", "max"))
        .sort_values(["season", "week"])
    )
    windows["tuesday_cutoff_utc"] = windows["first_kickoff_utc"].apply(
        lambda ts: pd.Timestamp(
            datetime.combine(_previous_or_same_tuesday(ts.date()), datetime.min.time()).replace(
                tzinfo=UTC
            )
        )
    )
    return windows


def build_asof_view(index_frame: pd.DataFrame, week_windows: pd.DataFrame) -> pd.DataFrame:
    if index_frame.empty:
        return pd.DataFrame()

    # One row per (season, capture_ts): a capture is one NFL-wide snapshot,
    # so the "best capture for this week" decision only needs to be made
    # once per (season, capture_ts), then broadcast to all teams in it.
    captures = (
        index_frame[["season", "capture_ts"]]
        .drop_duplicates()
        .sort_values(["season", "capture_ts"])
        .reset_index(drop=True)
    )

    records: list[dict[str, object]] = []
    for season, season_windows in week_windows.groupby("season"):
        season_captures = captures[captures["season"] == season]
        if season_captures.empty:
            continue
        # `.to_numpy()` on a tz-aware Series returns an object array of
        # tz-aware `Timestamp`s, which cannot be compared against the naive
        # `datetime64[ns]` produced by `Timestamp.to_datetime64()` below
        # (measured this session: "can't compare offset-naive and
        # offset-aware datetimes"). Force both sides to the same naive-but-
        # UTC-normalized `datetime64[ns]` representation instead.
        cap_ts = pd.to_datetime(season_captures["capture_ts"], utc=True).to_numpy(
            dtype="datetime64[ns]"
        )
        for _, wrow in season_windows.iterrows():
            tuesday_cutoff = wrow["tuesday_cutoff_utc"]
            first_kickoff = wrow["first_kickoff_utc"]
            eligible_tuesday = cap_ts[cap_ts <= tuesday_cutoff.to_datetime64()]
            eligible_prekickoff = cap_ts[cap_ts < first_kickoff.to_datetime64()]
            # Restore tz-aware UTC (naive datetime64 was only needed for the
            # comparison itself) so this matches `index_frame["capture_ts"]`'s
            # dtype for the merge below -- measured this session: a bare
            # `numpy.datetime64` here produces a naive `datetime64[ns]`
            # column that pandas refuses to merge against `index_frame`'s
            # tz-aware `datetime64[us, UTC]` capture_ts ("You are trying to
            # merge on datetime64[ns] and datetime64[us, UTC] columns").
            best_tuesday = (
                pd.Timestamp(eligible_tuesday.max(), tz="UTC") if len(eligible_tuesday) else None
            )
            best_prekickoff = (
                pd.Timestamp(eligible_prekickoff.max(), tz="UTC")
                if len(eligible_prekickoff)
                else None
            )
            records.append(
                {
                    "season": season,
                    "week": wrow["week"],
                    "tuesday_cutoff_utc": tuesday_cutoff,
                    "first_kickoff_utc": first_kickoff,
                    "asof_tuesday_capture_ts": best_tuesday,
                    "asof_prekickoff_capture_ts": best_prekickoff,
                    "has_tuesday_snapshot": best_tuesday is not None,
                    "has_prekickoff_snapshot": best_prekickoff is not None,
                }
            )
    week_capture_map = pd.DataFrame(records)
    if week_capture_map.empty:
        return week_capture_map
    # Align the merge-key column's exact dtype (unit + tz) to index_frame's
    # own capture_ts dtype -- pandas' merge refuses mismatched datetime
    # units/tz-awareness even when both sides are otherwise valid instants.
    week_capture_map["asof_tuesday_capture_ts"] = week_capture_map[
        "asof_tuesday_capture_ts"
    ].astype(index_frame["capture_ts"].dtype)

    merged = week_capture_map.merge(
        index_frame,
        left_on=["season", "asof_tuesday_capture_ts"],
        right_on=["season", "capture_ts"],
        how="left",
        suffixes=("", "_row"),
    )
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw/sagarin"))
    parser.add_argument("--snapshot", default=None, metavar="YYYYMMDDTHHMMSSZ")
    parser.add_argument("--start-season", type=int, default=2010)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument(
        "--no-usatoday",
        action="store_true",
        help="Skip the pre-2012 USA Today-hosted URL era (sagarin.com only).",
    )
    parser.add_argument(
        "--max-captures",
        type=int,
        default=None,
        help="Cap total captures processed this run (debug/budget control).",
    )
    parser.add_argument(
        "--schedules-snapshot",
        type=Path,
        default=None,
        help=(
            "Path to an existing nflverse schedules.parquet (project convention: "
            "data/raw/<UTC>/schedules.parquet) used to build the as-of-Tuesday "
            "alignment view. Defaults to the most recent one under data/raw."
        ),
    )
    parser.add_argument(
        "--skip-asof-view",
        action="store_true",
        help="Skip building asof_tuesday_view.parquet (index.parquet only).",
    )
    parser.add_argument(
        "--reparse-cache-only",
        action="store_true",
        help=(
            "WP19: rebuild captures_log/index/asof_tuesday_view from the "
            "existing on-disk HTML cache under --snapshot only -- zero "
            "network calls, no CDX enumeration. Use after a parser fix to "
            "measure the coverage delta on the exact same capture set, "
            "without conflating it with newly archived captures."
        ),
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    snapshot_dir = resolve_snapshot_dir(args.out, args.snapshot)
    print(f"Snapshot directory: {snapshot_dir}")
    limiter = RateLimiter(RATE_LIMIT_SECONDS)

    manifest = ingest(
        snapshot_dir,
        args.start_season,
        args.end_season,
        include_usatoday=not args.no_usatoday,
        limiter=limiter,
        max_captures=args.max_captures,
        reparse_cache_only=args.reparse_cache_only,
    )
    print(json.dumps(manifest, indent=2))

    if args.skip_asof_view:
        return

    schedules_path = args.schedules_snapshot
    if schedules_path is None:
        candidates = sorted(
            p
            for p in Path("data/raw").glob("*")
            if (p / "schedules.parquet").is_file() and (p / "manifest.json").is_file()
        )
        if not candidates:
            print("No local schedules.parquet found; skipping as-of-Tuesday view.", file=sys.stderr)
            return
        schedules_path = candidates[-1] / "schedules.parquet"
    print(f"Using schedules snapshot: {schedules_path}")

    schedules = pd.read_parquet(schedules_path)
    index_frame = pd.read_parquet(snapshot_dir / "index.parquet")
    week_windows = build_week_windows(schedules)
    asof_view = build_asof_view(index_frame, week_windows)
    asof_view.to_parquet(snapshot_dir / "asof_tuesday_view.parquet", index=False)
    print(f"Wrote asof_tuesday_view.parquet: {len(asof_view)} rows")


if __name__ == "__main__":
    main()
