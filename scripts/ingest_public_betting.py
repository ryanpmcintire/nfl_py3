"""Wayback Machine backfill of NFL public-betting-percentage pages.

Follow-up to `docs/data_source_scout_v3.md`'s rank-2 candidate
("actionnetwork.com free public betting-percentage archive" -- read that
document's section 2 for the mechanism case: bet%/money% splits by side are
the direct "fade/follow the public" signal). Clones the proven
sitemap/CDX-then-per-page-fetch structure of `scripts/ingest_injury_news.py`
and `scripts/ingest_transaction_news.py` (rate limiter, timestamped-snapshot-
directory convention, manifest.json), adapted for the Wayback Machine's CDX
API instead of a live sitemap, since these are historical percentages that
only exist as archived captures.

Two sources, per the scout doc's plan:

1. `actionnetwork.com/nfl/public-betting` -- the primary source, 2018-2025+.
   **Measured** this session: the page is a Next.js app whose server-rendered
   `__NEXT_DATA__` script tag carries real per-game bet%/money% data from
   Action Network's own "Consensus" book (book_id resolved dynamically per
   snapshot, not hardcoded, though it has been id 15 in every snapshot
   sampled this session). Two template eras were found, both parsed by this
   script:
     - **era1** (`initial_state`, 2018 through Oct/Nov 2022, exact cutover
       not pinned down -- boundary measured to sit between 2022-10-07
       [era1] and 2022-11-04 [era2]): `pageProps.initialState.gamesReducer
       .games`, a dict of per-game records; the Consensus book's `game`-period
       odds carry `{market}_{side}_public` integer fields. **Only bet
       (ticket) percentages exist in this era's schema -- there is no
       money-percentage field at all**, not a parsing gap: sampled directly
       across 2019/2020/2022 snapshots and confirmed absent structurally.
     - **era2** (`scoreboard_response`, ~Nov 2022 onward): `pageProps
       .scoreboardResponse.games`, a list of per-game records with an
       explicit `week` field (era1 has none) and a `markets.{book_id}.event
       .{market_type}` structure where each outcome carries
       `bet_info.tickets.percent` (bet%) AND `bet_info.money.percent`
       (money%) -- both present starting this era, confirmed in 2024 and
       2025 in-season snapshots (era2's earliest snapshots, e.g. one sampled
       from 2022-11-04 and 2022-12-29, had an empty `markets` dict for every
       game on the page -- a genuine per-capture completeness gap, not a
       template problem; see the coverage doc for the measured rate).
   The 2018 snapshot uses an older inline `__NEXT_DATA__ = {...};` assignment
   (no `<script id="__NEXT_DATA__">` wrapper) rather than the JSON script-tag
   form used from 2019 on; both forms are matched. Some Wayback `id_` raw
   fetches return gzip bytes with no `Content-Encoding` header (a Wayback
   storage quirk, not a real HTTP response) -- this script sniffs the gzip
   magic number and decompresses regardless of headers.

2. `covers.com/picks/nfl` -- named in the scout doc as a "cleaner, denser"
   2023+ complement. **Measured this session, and this is a correction to
   the scout doc**: this URL is Covers' community/handicapper PICKS page
   (individual authors' win-loss records, e.g. "100%"), not a sportsbook
   bet%/money% consensus page, and its real content (including the linked
   `contests.covers.com/consensus/topconsensus/nfl/overall` page, checked
   back to a 2016 archive as a secondary candidate) is populated by
   client-side AJAX after page load -- the archived static HTML's `<body>`
   contains essentially no real percentage data (a body-only regex scan of 5
   sampled snapshots spanning 2016/2023/2024/2025/2026 found zero genuine
   per-game percentages; every apparent hit traced back to Bootstrap grid CSS
   column widths or CSS keyframe `background-position` percentages embedded
   in a `<style>` tag placed inside `<body>` by the page's component
   framework). This script still fetches and stores a small verification
   sample (raw HTML on disk + a CDX inventory) so a future session does not
   have to re-derive this finding from scratch, but does not claim to parse
   percentages from it. `--covers-sample-n 0` skips even that.

Robots.txt for both origin sites was fetched and read this session before
building anything:
- `actionnetwork.com/robots.txt`: no `Crawl-delay`, no `Disallow` covering
  `/nfl/public-betting`.
- `covers.com/robots.txt`: no `Crawl-delay`, no `Disallow` covering
  `/picks/nfl`.
Neither is actually hit directly by this script, though -- every fetch here
goes through `web.archive.org` (the Wayback Machine), never the origin
sites, so the operative rate limit is the ~1 req/sec this script enforces
against `web.archive.org` itself (task instruction), not either site's own
policy. The robots.txt check is recorded for transparency/provenance, not
because it gates anything this script actually does.

Private-research use only, matching this project's existing CFBD/PFT/PFR
precedent (`docs/data_feasibility.md` License item 6: "private
caching/retention" permitted, raw tables "must never be republished").
Neither Action Network's nor Covers' terms of use were independently
reviewed this session (label: inferred policy stance, not a verified legal
fact).

Usage::

    # Full actionnetwork.com backfill (CDX index + every capture's HTML +
    # parse), 2018 through the current year, plus a small covers.com sample:
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_public_betting.py \\
        --out data/raw/public_betting

    # actionnetwork.com only, explicit year range, skip covers.com entirely:
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_public_betting.py \\
        --out data/raw/public_betting --an-start-year 2018 --an-end-year 2026 \\
        --covers-sample-n 0

    # Coverage report only, against an already-ingested snapshot (no fetch):
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_public_betting.py \\
        --out data/raw/public_betting --coverage-only \\
        --schedule-path data/processed/game_features.parquet

Writes under --out/<snapshot>/ (default data/raw/public_betting/<UTC
timestamp>) -- gitignored by the repository's existing `data/raw/**` rule.
Matches this repo's snapshot convention (`nfl_ats.snapshots.latest_snapshot()`
treats any directory directly under `data/raw/` with a `manifest.json` as a
candidate schedules snapshot, so `manifest.json` is nested one level down,
never directly at `data/raw/public_betting/manifest.json`).

    <snapshot>/actionnetwork/cdx_index.parquet    every CDX row found
    <snapshot>/actionnetwork/yearly/<YYYY>.parquet  parsed rows, capture year
    <snapshot>/actionnetwork/index.parquet        concatenation of all years
    <snapshot>/actionnetwork/raw_html/<ts>.html   raw fetched HTML (debug/audit)
    <snapshot>/covers/cdx_index.parquet           CDX inventory only
    <snapshot>/covers/sample_html/<ts>.html       small verification sample
    <snapshot>/covers/verification_summary.json   the negative-finding writeup
    <snapshot>/index.parquet                      == actionnetwork/index.parquet
                                                   (covers contributes no rows)
    <snapshot>/coverage_report.json               per-season join vs. schedule
    <snapshot>/manifest.json                      run metadata
"""

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

# nflverse abbreviation is the join target (data/processed/game_features.parquet
# uses these). Site abbreviations vary by era/relocation; map every alternate
# spelling seen this session onto the nflverse code. Extends (does not
# duplicate) src/nfl_ats/constants.py's TEAM_ABBREVIATION_ALIASES, which maps
# OAK->LV, SD->LAC, STL->LA but does not cover JAC/JAX or WAS/WSH -- this
# script is intentionally standalone (no src/nfl_ats import, matching
# ingest_injury_news.py / ingest_transaction_news.py), so the mapping is
# reproduced and extended locally rather than imported.
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
    """Enforces a fixed delay between requests to web.archive.org."""

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
    """Return the timestamped snapshot subdirectory to write/resume into.

    Same convention as ingest_injury_news.py / ingest_transaction_news.py:
    a manifest.json must never sit directly at `out_dir`.
    """

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
    """Fetch a URL and return decoded text, transparently handling Wayback
    `id_` raw captures that are gzip-compressed with no Content-Encoding
    header (measured this session: this is a real, recurring Wayback storage
    quirk, not an edge case)."""

    raw = _fetch_bytes(url, limiter, timeout=timeout)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="ignore")


def fetch_cdx(
    url_pattern: str, start_year: int, end_year: int, limiter: RateLimiter
) -> pd.DataFrame:
    """Query the Wayback CDX API for day-collapsed captures of `url_pattern`
    in [start_year, end_year]. Day-collapse (collapse=timestamp:8) keeps the
    first capture of each day -- this project's own scout doc used the same
    collapse when it measured "~15-45 captures/season", so this stays
    directly comparable."""

    query = (
        f"{CDX_ENDPOINT}?url={url_pattern}&output=json&from={start_year}&to={end_year}"
        "&filter=statuscode:200&collapse=timestamp:8"
    )
    raw = _fetch_bytes(query, limiter, timeout=60)
    text = raw.decode("utf-8", errors="ignore")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        # archive.org occasionally serves a transient HTML error page (a
        # measured 504 "Temporarily Offline" was hit this session) even with
        # a 200-looking curl exit; retry once after a longer pause.
        time.sleep(5.0)
        raw = _fetch_bytes(query, limiter, timeout=60)
        rows = json.loads(raw.decode("utf-8", errors="ignore"))
    if not rows:
        return pd.DataFrame(columns=["timestamp", "original"])
    header, *data = rows
    frame = pd.DataFrame(data, columns=header)
    frame["capture_ts"] = pd.to_datetime(frame["timestamp"], format="%Y%m%d%H%M%S", utc=True)
    return frame.sort_values("capture_ts").reset_index(drop=True)


# ---------------------------------------------------------------------------
# actionnetwork.com parsing
# ---------------------------------------------------------------------------

_NEXT_DATA_SCRIPT_START_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>\s*')
_NEXT_DATA_INLINE_START_RE = re.compile(r"__NEXT_DATA__\s*=\s*")


def _extract_balanced_json(text: str, open_brace_idx: int) -> str | None:
    """Return the substring from `open_brace_idx` (must point at '{') through
    its matching closing brace, respecting quoted strings/escapes.

    Needed because a naive non-greedy regex (`\\{.*?\\}`) under-matches: the
    2018-era inline `__NEXT_DATA__ = {...}</script>` form is immediately
    followed by ANOTHER `<script>...</script>` block with no separating
    whitespace pattern a regex can anchor on, so `.*?` stops at the first
    `}` that happens to precede some later `</script>` -- measured this
    session on a 2019-01-03 snapshot, silently truncating the real JSON and
    producing a `json.JSONDecodeError` ("Extra data"). A brace counter that
    is string/escape-aware finds the true matching close reliably.
    """

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
        # Action Network lists away team first, home team second in `teams`.
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
    """Return (era_label, rows, error). era_label is 'no_next_data',
    'unrecognized_shape', 'era1_initial_state', or 'era2_scoreboard_response'."""

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


# ---------------------------------------------------------------------------
# actionnetwork.com ingestion driver
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# covers.com verification-only sampling (measured dead end, see module docstring)
# ---------------------------------------------------------------------------

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
            "docs/data_source_scout_v3.md's claim of 'real moneyline consensus % "
            "confirmed in a 2023-08-19 snapshot' -- that exact snapshot was re-fetched "
            "this session and contains no such data in its server-rendered HTML."
        ),
    }
    with (covers_dir / "verification_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


# ---------------------------------------------------------------------------
# Coverage report: join parsed actionnetwork rows against the local schedule
# ---------------------------------------------------------------------------


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
    # Regular season only: this project's forced-pick pool is a REG-season
    # product, and actionnetwork's page also carries preseason (August
    # "scheduled" rows, kickoff before Week 1) and stale playoff reruns
    # (the same Wild Card matchup re-served as "complete" for weeks after it
    # ended, measured this session in 2019 Jan-Apr captures) that would
    # otherwise inflate/deflate REG-season coverage stats if merged in.
    schedule = schedule_full.loc[schedule_full["game_type"] == "REG"].copy()

    merged = parsed.merge(
        schedule, on=["away_team", "home_team"], how="left", suffixes=("", "_sched")
    )
    # A team pair can recur across seasons; keep only the schedule row whose
    # kickoff is within 3 days of the game's own start_time_utc as parsed
    # from the site (handles pregame moves/postponements loosely without
    # needing an exact-timestamp match).
    merged["start_time_utc"] = pd.to_datetime(merged["start_time_utc"], utc=True, errors="coerce")
    merged["kickoff_delta_hours"] = (
        merged["kickoff"] - merged["start_time_utc"]
    ).dt.total_seconds().abs() / 3600.0
    matched = merged.loc[merged["kickoff_delta_hours"] <= 72].copy()
    matched = matched.sort_values("kickoff_delta_hours").drop_duplicates(
        subset=["capture_ts", "site_game_id", "game_id"]
    )

    # Own-week Tuesday noon ET cutoff, mirroring
    # scripts/injury_tuesday_cutoff_experiment.py's team_week_tuesday_noon
    # convention exactly: (weekday - 1) % 7 days back from kickoff (in
    # US/Eastern), then +12h, converted back to UTC.
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
