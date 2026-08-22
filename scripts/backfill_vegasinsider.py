"""Multi-year backfill of VegasInsider NFL Las Vegas odds boards via Wayback.

Adapted from scripts/pilot_vegasinsider_wayback.py (measured 14/14 parse
success, 7 books, spreads 99.3% / totals 96.8% on 2011). Processes NFL REG
seasons 2005-2016 in bounded invocations of at most 2 seasons, resumable per
season: a season is skipped when its season_<year>.parquet already exists in
the target artifact snapshot dir. Raw snapshots and line-movement pages are
cached on disk under data/raw/vegasinsider/<run-id>/ so an interrupted season
refetches nothing already downloaded. Polite delay >=3s between archive
requests; hard wall-clock cap (~35 minutes) stops cleanly, records progress,
and exits 0 with a status JSON.

Per completed season this writes:
  artifacts/vegasinsider_backfill/<run-id>/season_<year>.parquet
      columns: capture_ts, game_date, away, home, kickoff_time, book,
               spread_line, total_line
  artifacts/vegasinsider_backfill/<run-id>/coverage_<year>.json
      schedule-match coverage (+/-1 day), share of REG games with >=1
      Tuesday/Wednesday-dated capture, books-per-game distribution,
      book-identity fallback rate (>20% flags reduced confidence).

No ATS evaluation, no registry writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CDX_URL = (
    "https://web.archive.org/cdx/search/cdx"
    "?url=vegasinsider.com/nfl/odds/las-vegas/"
    "&from={from_date}&to={to_date}&filter=statuscode:200&output=json"
)
RAW_ENDPOINT = "https://web.archive.org/web/{ts}id_/{original}"
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
MIN_DELAY_SECONDS = 3.0
SOURCE_DESCRIPTION = "vegasinsider.com/nfl/odds/las-vegas/ via web.archive.org raw endpoint"
DEFAULT_CAPTURES_PER_SEASON = 20
DEFAULT_WALL_CLOCK_MINUTES = 35.0
MAX_SEASONS_PER_INVOCATION = 2
FALLBACK_RATE_FLAG_THRESHOLD = 0.20

FRANCHISE_CODES = frozenset(
    {
        "ARI",
        "ATL",
        "BAL",
        "BUF",
        "CAR",
        "CHI",
        "CIN",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GB",
        "HOU",
        "IND",
        "JAX",
        "KC",
        "LAC",
        "LAR",
        "LV",
        "MIA",
        "MIN",
        "NE",
        "NO",
        "NYG",
        "NYJ",
        "PHI",
        "PIT",
        "SEA",
        "SF",
        "TB",
        "TEN",
        "WAS",
    }
)

TEAM_NAME_ALIASES = {
    "ARIZONA": "ARI",
    "ATLANTA": "ATL",
    "BALTIMORE": "BAL",
    "BUFFALO": "BUF",
    "CAROLINA": "CAR",
    "CHICAGO": "CHI",
    "CINCINNATI": "CIN",
    "CLEVELAND": "CLE",
    "DALLAS": "DAL",
    "DENVER": "DEN",
    "DETROIT": "DET",
    "GREEN BAY": "GB",
    "HOUSTON": "HOU",
    "HOUSTON TEXANS": "HOU",
    "INDIANAPOLIS": "IND",
    "JACKSONVILLE": "JAX",
    "KANSAS CITY": "KC",
    "MIAMI": "MIA",
    "MINNESOTA": "MIN",
    "NEW ENGLAND": "NE",
    "NEW ORLEANS": "NO",
    "N.Y. GIANTS": "NYG",
    "NY GIANTS": "NYG",
    "N.Y. JETS": "NYJ",
    "NY JETS": "NYJ",
    "OAKLAND": "LV",
    "PHILADELPHIA": "PHI",
    "PITTSBURGH": "PIT",
    "ST. LOUIS": "LAR",
    "SAN DIEGO": "LAC",
    "SAN FRAN.": "SF",
    "SAN FRANCISCO": "SF",
    "SEATTLE": "SEA",
    "TAMPA BAY": "TB",
    "TENNESSEE": "TEN",
    "WASHINGTON": "WAS",
}

VI_CODE_TO_SCHEDULE_DEFAULTS = {"LAR": "STL", "LAC": "SD", "LV": "OAK"}
VI_CODE_TO_SCHEDULE_OVERRIDES = {"LAR": {2016: "LA"}}


def normalize_team(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = re.sub(r"\s+", " ", raw.strip().upper())
    if key in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[key]
    return key if key in FRANCHISE_CODES else None


def vi_code_to_schedule(code: str | None, season: int) -> str | None:
    if code is None:
        return None
    default = VI_CODE_TO_SCHEDULE_DEFAULTS.get(code)
    if default is None:
        return code
    overrides = VI_CODE_TO_SCHEDULE_OVERRIDES.get(code, {})
    return overrides.get(season, default)


@dataclass
class RateLimiter:
    delay_seconds: float
    _last_request: float | None = field(default=None, init=False)

    def wait(self) -> None:
        if self._last_request is not None:
            elapsed = time.monotonic() - self._last_request
            remaining = max(self.delay_seconds - elapsed, 0.0)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request = time.monotonic()


class BudgetExceeded(Exception):
    pass


@dataclass
class WallClockBudget:
    deadline_monotonic: float

    @classmethod
    def from_minutes(cls, minutes: float) -> WallClockBudget:
        return cls(deadline_monotonic=time.monotonic() + minutes * 60.0)

    @property
    def remaining_seconds(self) -> float:
        return self.deadline_monotonic - time.monotonic()

    def check(self) -> None:
        if self.remaining_seconds <= 0:
            raise BudgetExceeded("wall-clock budget exhausted")


def fetch_via_curl(
    url: str,
    limiter: RateLimiter,
    budget: WallClockBudget | None = None,
    *,
    retries: int = 2,
) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for _attempt in range(retries):
        if budget is not None:
            budget.check()
        limiter.wait()
        try:
            completed = subprocess.run(
                [
                    "curl.exe",
                    "-s",
                    "-S",
                    "-L",
                    "--compressed",
                    "--max-time",
                    "90",
                    "-A",
                    USER_AGENT,
                    "-w",
                    "\n__CURL_HTTP_CODE__%{http_code}",
                    url,
                ],
                capture_output=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            last_error = error
            continue
        if completed.returncode != 0:
            stderr_text = completed.stderr.decode(errors="replace")[:300]
            last_error = RuntimeError(f"curl exit {completed.returncode}: {stderr_text}")
            continue
        marker = b"\n__CURL_HTTP_CODE__"
        idx = completed.stdout.rfind(marker)
        if idx == -1:
            last_error = RuntimeError("curl output missing http-code marker")
            continue
        body = completed.stdout[:idx]
        http_code = completed.stdout[idx + len(marker) :].decode(errors="replace").strip()
        if http_code != "200":
            last_error = RuntimeError(f"http status {http_code}")
            if http_code in ("404", "403"):
                break
            continue
        return body, http_code
    assert last_error is not None
    raise last_error


@dataclass(frozen=True)
class CdxRow:
    timestamp: str
    original: str
    digest: str
    length: int


def query_cdx(
    from_date: str, to_date: str, limiter: RateLimiter, budget: WallClockBudget | None = None
) -> list[CdxRow]:
    body, _ = fetch_via_curl(CDX_URL.format(from_date=from_date, to_date=to_date), limiter, budget)
    rows = json.loads(body.decode("utf-8"))
    parsed: list[CdxRow] = []
    for row in rows[1:]:
        parsed.append(
            CdxRow(
                timestamp=row[1],
                original=row[2],
                digest=row[5],
                length=int(row[6]) if row[6].isdigit() else 0,
            )
        )
    deduped: dict[str, CdxRow] = {}
    for row in sorted(parsed, key=lambda r: r.timestamp):
        deduped.setdefault(row.digest, row)
    return sorted(deduped.values(), key=lambda r: r.timestamp)


def select_captures(rows: list[CdxRow], n: int) -> list[CdxRow]:
    if len(rows) <= n:
        return rows
    stamps = [datetime.strptime(r.timestamp[:8], "%Y%m%d") for r in rows]
    lo, hi = stamps[0].toordinal(), stamps[-1].toordinal()
    step = (hi - lo) / n
    selected: list[CdxRow] = []
    taken: set[str] = set()
    for i in range(n):
        center = lo + step * (i + 0.5)
        candidates = [
            (row, stamp)
            for row, stamp in zip(rows, stamps, strict=True)
            if row.digest not in taken and abs(stamp.toordinal() - center) <= max(step, 4)
        ]
        if not candidates:
            continue
        best_row, _ = min(
            candidates,
            key=lambda cs: (
                0 if cs[1].weekday() in (1, 2) else 1,
                abs(cs[1].toordinal() - center),
            ),
        )
        taken.add(best_row.digest)
        selected.append(best_row)
    return selected


@dataclass
class SnapshotRecord:
    capture_ts: str
    original_url: str
    wayback_url: str
    file: str | None
    sha256: str | None
    size_bytes: int
    cdx_digest: str
    cdx_length: int
    http_status: str
    error: str | None


def run_id_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def fetch_snapshots(
    snapshot_dir: Path,
    captures: list[CdxRow],
    limiter: RateLimiter,
    budget: WallClockBudget,
) -> list[SnapshotRecord]:
    raw_dir = snapshot_dir / "snapshots"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[SnapshotRecord] = []
    for i, cap in enumerate(captures):
        path = raw_dir / f"{cap.timestamp}.html"
        wayback_url = RAW_ENDPOINT.format(ts=cap.timestamp, original=cap.original)
        if path.exists():
            raw = path.read_bytes()
            status, error = "cached", None
        else:
            try:
                raw, status_code = fetch_via_curl(wayback_url, limiter, budget)
                status, error = status_code, None
            except BudgetExceeded:
                raise
            except Exception as err:
                records.append(
                    SnapshotRecord(
                        capture_ts=cap.timestamp,
                        original_url=cap.original,
                        wayback_url=wayback_url,
                        file=None,
                        sha256=None,
                        size_bytes=0,
                        cdx_digest=cap.digest,
                        cdx_length=cap.length,
                        http_status="error",
                        error=str(err),
                    )
                )
                print(f"  [{i + 1}/{len(captures)}] FAILED {cap.timestamp}: {err}")
                continue
            path.write_bytes(raw)
        records.append(
            SnapshotRecord(
                capture_ts=cap.timestamp,
                original_url=cap.original,
                wayback_url=wayback_url,
                file=f"snapshots/{path.name}",
                sha256=hashlib.sha256(raw).hexdigest(),
                size_bytes=len(raw),
                cdx_digest=cap.digest,
                cdx_length=cap.length,
                http_status=status,
                error=error,
            )
        )
        print(f"  [{i + 1}/{len(captures)}] {cap.timestamp} bytes={len(raw)}")
    return records


def write_manifest(snapshot_dir: Path, records: list[SnapshotRecord], season: int) -> Path:
    manifest = {
        "source": "vegasinsider.com/nfl/odds/las-vegas/ via web.archive.org raw endpoint",
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "user_agent": USER_AGENT,
        "rate_limit_seconds": MIN_DELAY_SECONDS,
        "season_window": {"from": f"{season}0901", "to": f"{season + 1}0114"},
        "snapshots": [
            {
                "capture_timestamp": r.capture_ts,
                "original_url": r.original_url,
                "wayback_url": r.wayback_url,
                "file": r.file,
                "sha256": r.sha256,
                "size_bytes": r.size_bytes,
                "cdx_digest": r.cdx_digest,
                "cdx_length": r.cdx_length,
                "http_status": r.http_status,
                "error": r.error,
            }
            for r in records
        ],
    }
    path = snapshot_dir / f"manifest_{season}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


@dataclass
class OddsCell:
    column_index: int
    anchor: str | None
    raw_text: str
    spread_line: float | None
    total_line: float | None
    book_name: str | None = None


@dataclass
class GameRow:
    game_date_iso: str | None
    kickoff_time: str | None
    away_rotation: int | None
    away_name: str | None
    home_rotation: int | None
    home_name: str | None
    cells: list[OddsCell]


@dataclass
class BoardParse:
    capture_ts: str
    updated_line: str | None
    games: list[GameRow]
    error: str | None = None
    layout: str = "modern"


ROW_TAG_RE = re.compile(r"<tr[^>]*oddsText(?:_odd|_even)[^>]*>", re.IGNORECASE)
KICK_RE = re.compile(r"<b>\s*(\d{1,2}/\d{1,2})\s+(\d{1,2}:\d{2}\s*[AP]M)\s*</b>", re.IGNORECASE)
TEAM_RE = re.compile(r"<b>\s*(\d{1,3})\s*&nbsp;\s*<a[^>]*>([^<]+)</a>", re.IGNORECASE)
ANCHOR_RE = re.compile(r"#([A-Za-z]{1,2})(?![A-Za-z])")
LM_NAME_ANCHOR_RE = re.compile(r'<a\s+name="([A-Za-z]{1,3})"', re.IGNORECASE)


def strip_tags(fragment: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&#183;", "")
        .replace("&frac12;", ".5")
        .replace("&#189;", ".5")
        .replace("&frac14;", ".25")
        .replace("&frac34;", ".75")
    )
    return text


SPREAD_RE_TEMPLATE = r"^([+\-]?(?:\d+(?:\.5)?|PK))(?:\s*([+\-]\d{1,3}))?$"


def parse_spread_token(token: str) -> float | None:
    m = re.match(SPREAD_RE_TEMPLATE, token.strip(), re.IGNORECASE)
    if not m:
        return None
    head = m.group(1)
    if head.upper() == "PK":
        return 0.0
    try:
        value = float(head)
    except ValueError:
        return None
    if abs(value) > 80:
        return None
    return value


def parse_total_token(token: str) -> float | None:
    m = re.match(r"^(\d+(?:\.5)?)\s*[ouOU]?(?:\s*[+\-]\d{1,3})?$", token.strip())
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    if not 10 <= value <= 90:
        return None
    return value


SIGNED_TOKEN_RE = re.compile(r"^([+\-]\d+(?:\.\d+)?)(?:\s*([+\-]\d{1,3}))?$")
OU_TOTAL_RE = re.compile(r"^(\d+(?:\.5)?)\s*[oOuU]")
BARE_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")


def classify_line_tokens(tokens: list[str]) -> tuple[float | None, float | None]:
    spread: float | None = None
    total: float | None = None
    bare_small: list[float] = []
    for token in tokens:
        if re.match(r"PK(?:\s|$)", token, re.IGNORECASE):
            if spread is None:
                spread = 0.0
            continue
        m = SIGNED_TOKEN_RE.match(token)
        if m:
            value = float(m.group(1))
            if abs(value) <= 80 and spread is None:
                spread = value
            continue
        m = OU_TOTAL_RE.match(token)
        if m:
            value = float(m.group(1))
            if 10 <= value <= 90 and total is None:
                total = value
            continue
        if BARE_NUMBER_RE.match(token):
            value = float(token)
            if value >= 15:
                if 10 <= value <= 90 and total is None:
                    total = value
            elif abs(value) <= 80:
                bare_small.append(value)
    if spread is None and bare_small:
        spread = bare_small[0]
    return spread, total


def parse_cells(chunk: str) -> list[OddsCell]:
    pieces = re.findall(
        r'<td\s+width="(?:\d+)"\s+class="oddsText[^"]*"[^>]*>(.*?)</td>',
        chunk,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not pieces:
        pieces = re.findall(
            r"<td\s+width=(?:'|\x22)(?:\d+)(?:'|\x22)[^>]*>(.*?)</td>",
            chunk,
            flags=re.IGNORECASE | re.DOTALL,
        )
    cells: list[OddsCell] = []
    for idx, cell_html in enumerate(pieces):
        anchor_m = ANCHOR_RE.search(cell_html)
        text = strip_tags(cell_html)
        tokens = [t.strip() for t in text.split("\n")]
        tokens = [re.sub(r"\s+", " ", t).strip() for t in tokens]
        tokens = [t for t in tokens if t]
        spread, total = classify_line_tokens(tokens)
        compact_raw = " | ".join(tokens)[:120]
        if not compact_raw:
            continue
        cells.append(
            OddsCell(
                column_index=idx,
                anchor=anchor_m.group(1).upper() if anchor_m else None,
                raw_text=compact_raw,
                spread_line=spread,
                total_line=total,
            )
        )
    return cells


def resolve_game_year(capture_dt: datetime, month: int, day: int) -> str | None:
    best: tuple[int, date] | None = None
    for year in (capture_dt.year - 1, capture_dt.year, capture_dt.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        distance = abs((candidate - capture_dt.date()).days)
        if best is None or distance < best[0]:
            best = (distance, candidate)
    return best[1].isoformat() if best is not None else None


LEGACY_HEADER_RE = re.compile(
    r"<td\s+class=['\"]odds['\"][^>]*>\s*<b>(.*?)</b>\s*</td>",
    re.IGNORECASE | re.DOTALL,
)
LEGACY_GAME_INFO_RE = re.compile(r"<td nowrap><b>([^<]+)</b>", re.IGNORECASE)
LEGACY_KICK_RE = re.compile(r"(\d{1,2}/\d{1,2})\s+(\d{1,2}:\d{2}\s*[AP]M)", re.IGNORECASE)
LEGACY_BOOK_CELL_RE = re.compile(
    r"<td\s+nowrap(?:=[\"']nowrap[\"'])?\s+valign=[\"']?bottom[\"']?\s+"
    r"align=[\"']?center[\"']?>(.*?)</td>",
    re.IGNORECASE | re.DOTALL,
)


def parse_legacy_header_names(html: str) -> list[str]:
    names: list[str] = []
    for m in LEGACY_HEADER_RE.finditer(html):
        raw = re.sub(r"<br\s*/?>", "-", m.group(1), flags=re.IGNORECASE)
        name = re.sub(r"\s+", " ", strip_tags(raw)).strip().upper()
        if name:
            names.append(name)
    return names


def parse_legacy_book_cell(cell_html: str) -> tuple[float | None, float | None]:
    text = strip_tags(cell_html)
    tokens = [re.sub(r"\s+", " ", t).strip() for t in text.split("\n")]
    tokens = [t for t in tokens if t]
    total: float | None = None
    spread: float | None = None
    for token in tokens:
        m = re.match(r"^(\d+(?:\.\d+)?)\s*o\s*/\s*u$", token, re.IGNORECASE)
        if m and total is None:
            value = float(m.group(1))
            if 10 <= value <= 90:
                total = value
            continue
        m = re.match(r"^([+\-]?\d+(?:\.\d+)?)$", token)
        if m and spread is None:
            value = float(m.group(1))
            if abs(value) <= 80:
                spread = value
    return spread, total


def parse_board_legacy(capture_ts: str, html: str) -> BoardParse:
    capture_dt = datetime.strptime(capture_ts[:8], "%Y%m%d")
    header = parse_legacy_header_names(html)
    games: list[GameRow] = []
    row_starts = [
        m.start()
        for m in re.finditer(r"<tr[^>]*class=['\"]bg[012]['\"][^>]*>", html, re.IGNORECASE)
    ]
    for i, start in enumerate(row_starts):
        end = row_starts[i + 1] if i + 1 < len(row_starts) else len(html)
        chunk = html[start:end]
        teams = LEGACY_GAME_INFO_RE.findall(chunk)
        kick = LEGACY_KICK_RE.search(chunk)
        if len(teams) != 2 or not kick:
            continue
        parts = kick.group(1).split("/")
        month, day = int(parts[0]), int(parts[1])
        book_cells: list[OddsCell] = []
        for idx, cell_m in enumerate(LEGACY_BOOK_CELL_RE.finditer(chunk)):
            spread, total = parse_legacy_book_cell(cell_m.group(1))
            compact_raw = re.sub(r"\s+", " ", strip_tags(cell_m.group(1))).strip()[:120]
            if not compact_raw:
                continue
            book_cells.append(
                OddsCell(
                    column_index=idx,
                    anchor=None,
                    raw_text=compact_raw,
                    spread_line=spread,
                    total_line=total,
                    book_name=header[idx] if idx < len(header) else None,
                )
            )
        games.append(
            GameRow(
                game_date_iso=resolve_game_year(capture_dt, month, day),
                kickoff_time=re.sub(r"\s+", " ", kick.group(2)).upper(),
                away_rotation=None,
                away_name=teams[0].strip(),
                home_rotation=None,
                home_name=teams[1].strip(),
                cells=book_cells,
            )
        )
    if not games:
        return BoardParse(
            capture_ts=capture_ts,
            updated_line=None,
            games=[],
            error="legacy layout: no game rows found",
            layout="legacy",
        )
    return BoardParse(
        capture_ts=capture_ts,
        updated_line=None,
        games=games,
        layout="legacy",
        error=None,
    )


VICELL_GAME_TD_RE = re.compile(
    r'<td\s+class="viCellBg[12]\s+cellTextNorm[^"]*"\s+nowrap\s+width="158">',
    re.IGNORECASE,
)
VICELL_TEAM_RE = re.compile(r"(\d{1,3})\s*&nbsp;\s*<b><a[^>]*>([^<]+)</a></b>", re.IGNORECASE)
VICELL_BOOK_CELL_RE = re.compile(
    r'<td\s+class="viCellBg[12][^"]*"\s+width="56"[^>]*>(.*?)</td>',
    re.IGNORECASE | re.DOTALL,
)


def parse_board_vicell(capture_ts: str, html: str) -> BoardParse:
    capture_dt = datetime.strptime(capture_ts[:8], "%Y%m%d")
    starts = [m.start() for m in VICELL_GAME_TD_RE.finditer(html)]
    if not starts:
        return BoardParse(
            capture_ts=capture_ts,
            updated_line=None,
            games=[],
            error="vicell layout: no game rows found",
            layout="vicell",
        )
    games: list[GameRow] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(html)
        chunk = html[start:end]
        kick = LEGACY_KICK_RE.search(chunk)
        teams = VICELL_TEAM_RE.findall(chunk)
        if not kick or len(teams) < 2:
            continue
        parts = kick.group(1).split("/")
        month, day = int(parts[0]), int(parts[1])
        book_cells: list[OddsCell] = []
        for idx, cell_m in enumerate(VICELL_BOOK_CELL_RE.finditer(chunk)):
            anchor_m = ANCHOR_RE.search(cell_m.group(1))
            text = strip_tags(cell_m.group(1))
            tokens = [re.sub(r"\s+", " ", t).strip() for t in text.split("\n")]
            tokens = [t for t in tokens if t]
            spread, total = classify_line_tokens(tokens)
            compact_raw = " | ".join(tokens)[:120]
            if not compact_raw:
                continue
            book_cells.append(
                OddsCell(
                    column_index=idx,
                    anchor=anchor_m.group(1).upper() if anchor_m else None,
                    raw_text=compact_raw,
                    spread_line=spread,
                    total_line=total,
                )
            )
        games.append(
            GameRow(
                game_date_iso=resolve_game_year(capture_dt, month, day),
                kickoff_time=re.sub(r"\s+", " ", kick.group(2)).upper(),
                away_rotation=int(teams[0][0]),
                away_name=teams[0][1].strip(),
                home_rotation=int(teams[1][0]),
                home_name=teams[1][1].strip(),
                cells=book_cells,
            )
        )
    return BoardParse(
        capture_ts=capture_ts,
        updated_line=None,
        games=games,
        layout="vicell",
        error=None if games else "vicell layout: no complete game rows",
    )


def parse_board(capture_ts: str, html: str) -> BoardParse:
    capture_dt = datetime.strptime(capture_ts[:8], "%Y%m%d")
    updated_m = re.search(r"Updated:\s*[^<\n]+", html)
    updated_line = re.sub(r"\s+", " ", updated_m.group(0)).strip() if updated_m else None
    tags = list(ROW_TAG_RE.finditer(html))
    if not tags:
        vicell = parse_board_vicell(capture_ts, html)
        if vicell.games:
            return vicell
        legacy = parse_board_legacy(capture_ts, html)
        if legacy.games:
            return legacy
        return BoardParse(
            capture_ts=capture_ts,
            updated_line=updated_line,
            games=[],
            error="no oddsText rows found; "
            + (vicell.error or legacy.error or "fallbacks found nothing"),
            layout="legacy",
        )
    games: list[GameRow] = []
    for i, m in enumerate(tags):
        end = tags[i + 1].start() if i + 1 < len(tags) else len(html)
        chunk = html[m.end() : end]
        kick = KICK_RE.search(chunk)
        teams = TEAM_RE.findall(chunk)
        if not kick or len(teams) < 2:
            continue
        parts = kick.group(1).split("/")
        month, day = int(parts[0]), int(parts[1])
        games.append(
            GameRow(
                game_date_iso=resolve_game_year(capture_dt, month, day),
                kickoff_time=re.sub(r"\s+", " ", kick.group(2)).upper(),
                away_rotation=int(teams[0][0]),
                away_name=teams[0][1].strip(),
                home_rotation=int(teams[1][0]),
                home_name=teams[1][1].strip(),
                cells=parse_cells(chunk),
            )
        )
    if not games:
        return BoardParse(
            capture_ts=capture_ts,
            updated_line=updated_line,
            games=[],
            error="rows found but none yielded date+two teams",
        )
    return BoardParse(capture_ts=capture_ts, updated_line=updated_line, games=games)


def extract_anchor_names(html: str) -> dict[str, str]:
    anchors: dict[str, str] = {}
    for m in LM_NAME_ANCHOR_RE.finditer(html):
        seg = html[m.end() : m.end() + 800]
        title = re.search(r"([A-Z][A-Z .'&-]{1,30}?)\s+LINE\s+MOVEMENTS", seg, re.IGNORECASE)
        if title:
            anchors[m.group(1).upper()] = re.sub(r"\s+", " ", title.group(1)).strip().upper()
    return anchors


def fetch_book_map(
    capture_ts: str,
    board_html: str,
    limiter: RateLimiter,
    snapshot_dir: Path,
    budget: WallClockBudget,
) -> tuple[dict[str, str], str | None]:
    lm_dir = snapshot_dir / "line_movement"
    lm_dir.mkdir(parents=True, exist_ok=True)
    seen_paths: list[str] = []
    for link_m in re.finditer(r"/nfl/odds/las-vegas/line-movement/[^'\">\s#]+", board_html):
        rel = link_m.group(0).split("#", 1)[0]
        if rel not in seen_paths:
            seen_paths.append(rel)
    if not seen_paths:
        return {}, "no line-movement link found on board"
    errors: list[str] = []
    for rel in seen_paths[:8]:
        lm_path = lm_dir / f"{capture_ts}_{hashlib.md5(rel.encode()).hexdigest()[:8]}.html"
        if lm_path.exists():
            html = lm_path.read_bytes().decode("utf-8", errors="replace")
        else:
            url = f"https://web.archive.org/web/{capture_ts}id_/http://www.vegasinsider.com{rel}"
            try:
                raw, _ = fetch_via_curl(url, limiter, budget)
            except BudgetExceeded:
                raise
            except Exception as err:
                errors.append(f"{rel}: {err}")
                continue
            lm_path.write_bytes(raw)
            html = raw.decode("utf-8", errors="replace")
        anchors = extract_anchor_names(html)
        if anchors:
            return anchors, None
        errors.append(f"{rel}: no anchor-name pairs")
    return {}, "; ".join(errors)[:300]


def latest_schedules_frame() -> pd.DataFrame | None:
    candidates = sorted(REPO.glob("data/raw/*/schedules.parquet"))
    if not candidates:
        return None
    return pd.read_parquet(candidates[-1])


def season_schedule_index(schedules: pd.DataFrame, season: int) -> dict[tuple[str, str], set[str]]:
    reg = schedules.loc[
        (schedules["season"] == season) & (schedules["game_type"].astype(str).eq("REG"))
    ]
    index: dict[tuple[str, str], set[str]] = {}
    for row in reg.itertuples(index=False):
        key = (str(row.home_team), str(row.away_team))
        index.setdefault(key, set()).add(str(row.gameday)[:10])
    return index


def build_tidy(parses: list[BoardParse], book_maps: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    excluded_unanchored = 0
    for parse in parses:
        bmap = book_maps.get(parse.capture_ts, {})
        for game in parse.games:
            for cell in game.cells:
                if cell.anchor is None and cell.book_name is None:
                    excluded_unanchored += 1
                    continue
                rows.append(
                    {
                        "capture_ts": parse.capture_ts,
                        "game_date": game.game_date_iso,
                        "away": normalize_team(game.away_name),
                        "home": normalize_team(game.home_name),
                        "kickoff_time": game.kickoff_time,
                        "book": cell.book_name or bmap.get(cell.anchor),
                        "spread_line": cell.spread_line,
                        "total_line": cell.total_line,
                    }
                )
    frame = pd.DataFrame(
        rows,
        columns=[
            "capture_ts",
            "game_date",
            "away",
            "home",
            "kickoff_time",
            "book",
            "spread_line",
            "total_line",
        ],
    )
    frame.attrs["excluded_unanchored_cells"] = excluded_unanchored
    if not frame.empty:
        frame = frame.sort_values(["capture_ts", "game_date", "kickoff_time"]).reset_index(
            drop=True
        )
    return frame


def compute_coverage(
    season: int,
    tidy: pd.DataFrame,
    parses: list[BoardParse],
    book_map_sources: dict[str, str],
    schedules: pd.DataFrame | None,
) -> dict[str, Any]:
    stats: dict[str, Any] = {"season": season}
    sources = [book_map_sources.get(p.capture_ts) for p in parses]
    needed = [s for s in sources if s is not None]
    direct_sources = {"own_line_movement_page", "header_text_columns"}
    own = sum(1 for s in needed if s == "own_line_movement_page")
    header_named = sum(1 for s in needed if s == "header_text_columns")
    fallback = sum(1 for s in needed if s not in direct_sources)
    fallback_rate = fallback / len(needed) if needed else 0.0
    stats["book_identity"] = {
        "captures_with_own_line_movement_page": own,
        "captures_named_from_board_header_text": header_named,
        "captures_using_cross_capture_fallback": fallback,
        "captures_without_any_map": sum(1 for p in parses if p.capture_ts not in book_map_sources),
        "fallback_rate": round(fallback_rate, 4),
        "reduced_confidence_flag": bool(fallback_rate > FALLBACK_RATE_FLAG_THRESHOLD),
    }

    stats["cells"] = {
        "tidy_rows": len(tidy),
        "unanchored_cells_excluded": int(tidy.attrs.get("excluded_unanchored_cells", 0)),
        "rows_with_spread": int(tidy["spread_line"].notna().sum()),
        "rows_with_total": int(tidy["total_line"].notna().sum()),
        "spread_coverage_rate": round(float(tidy["spread_line"].notna().mean()), 4)
        if len(tidy)
        else 0.0,
        "total_coverage_rate": round(float(tidy["total_line"].notna().mean()), 4)
        if len(tidy)
        else 0.0,
        "named_book_rows": int(tidy["book"].notna().sum()),
    }
    stats["captures"] = {
        "boards_parsed_ok": sum(1 for p in parses if p.error is None and p.games),
        "boards_parse_failed": sum(1 for p in parses if not (p.error is None and p.games)),
    }

    books_per_game: dict[int, int] = {}
    if len(tidy):
        grouped = tidy.groupby(["capture_ts", "game_date", "away", "home"], dropna=False)
        for _, grp in grouped:
            named = set(grp.loc[grp["book"].notna(), "book"])
            books_per_game[len(named)] = books_per_game.get(len(named), 0) + 1
    stats["books_per_game_distribution"] = {str(k): v for k, v in sorted(books_per_game.items())}

    season_in_schedules = schedules is not None and bool((schedules["season"] == season).any())
    if schedules is None or not season_in_schedules:
        stats["schedule_match"] = {
            "available": False,
            "note": (
                "no local nflverse schedules snapshot covers this season "
                "(local coverage starts 2009); match stats unavailable"
            ),
        }
        return stats

    sched_index = season_schedule_index(schedules, season)
    sched_games: list[tuple[str, str, str]] = []
    for (home_team, away_team), dates in sched_index.items():
        for game_day in dates:
            sched_games.append((home_team, away_team, game_day))

    def board_matches_sched(home_code: Any, away_code: Any, game_date: Any) -> bool:
        if pd.isna(home_code) or pd.isna(away_code) or pd.isna(game_date):
            return False
        sched_home = vi_code_to_schedule(str(home_code), season)
        sched_away = vi_code_to_schedule(str(away_code), season)
        dates = sched_index.get((sched_home, sched_away))
        if not dates:
            return False
        try:
            board_day = date.fromisoformat(str(game_date))
        except ValueError:
            return False
        return any(abs((board_day - date.fromisoformat(d)).days) <= 1 for d in dates)

    instances = tidy.drop_duplicates(subset=["capture_ts", "game_date", "away", "home"])
    matched_flags = [
        board_matches_sched(row.home, row.away, row.game_date)
        for row in instances.itertuples(index=False)
    ]
    matched_instances = sum(matched_flags)

    tw_captures = {
        row.capture_ts
        for row in tidy.itertuples(index=False)
        if datetime.strptime(str(row.capture_ts)[:8], "%Y%m%d").weekday() in (1, 2)
    }
    covered_reg = 0
    for home_team, away_team, reg_day in sched_games:
        try:
            reg_date = date.fromisoformat(reg_day)
        except ValueError:
            continue
        hit = any(
            row.capture_ts in tw_captures
            and row.away == away_team
            and row.home == home_team
            and row.game_date is not None
            and abs((date.fromisoformat(str(row.game_date)) - reg_date).days) <= 1
            for row in tidy.itertuples(index=False)
        )
        covered_reg += int(hit)

    unmatched_examples = (
        instances.loc[[not f for f in matched_flags], ["game_date", "away", "home"]]
        .head(10)
        .values.tolist()
    )
    stats["schedule_match"] = {
        "available": True,
        "reg_games_in_local_schedule": len(sched_games),
        "board_game_instances_total": len(instances),
        "board_game_instances_matched_pct": round(matched_instances / len(instances), 4)
        if len(instances)
        else 0.0,
        "reg_games_with_tue_or_wed_capture_pct": round(covered_reg / len(sched_games), 4)
        if sched_games
        else 0.0,
        "unmatched_examples_first_10": unmatched_examples,
    }
    return stats


def process_season(
    season: int,
    snapshot_dir: Path,
    artifacts_dir: Path,
    limiter: RateLimiter,
    budget: WallClockBudget,
    captures_per_season: int,
    schedules: pd.DataFrame | None,
) -> dict[str, Any]:
    season_summary: dict[str, Any] = {"season": season, "status": "in_progress"}
    cdx_cache = snapshot_dir / f"cdx_{season}.json"
    if cdx_cache.exists():
        all_rows = [CdxRow(**r) for r in json.loads(cdx_cache.read_text(encoding="utf-8"))]
        print(f"[{season}] CDX cached: {len(all_rows)} unique-digest captures")
    else:
        budget.check()
        all_rows = query_cdx(f"{season}0901", f"{season + 1}0114", limiter, budget)
        cdx_cache.write_text(json.dumps([r.__dict__ for r in all_rows], indent=2), encoding="utf-8")
        print(f"[{season}] CDX measured: {len(all_rows)} unique-digest captures")

    captures = select_captures(all_rows, captures_per_season)
    print(f"[{season}] selected {len(captures)} captures")
    records = fetch_snapshots(snapshot_dir, captures, limiter, budget)
    manifest_path = write_manifest(snapshot_dir, records, season)
    print(f"[{season}] manifest: {manifest_path}")

    parses: list[BoardParse] = []
    book_maps: dict[str, dict[str, str]] = {}
    book_map_errors: dict[str, str] = {}
    book_map_sources: dict[str, str] = {}
    ok_records = [r for r in records if r.file is not None]
    for i, rec in enumerate(ok_records):
        html = (snapshot_dir / rec.file).read_bytes().decode("utf-8", errors="replace")
        parse = parse_board(rec.capture_ts, html)
        if parse.layout == "legacy" and parse.games:
            book_map_sources[rec.capture_ts] = "header_text_columns"
        if (
            parse.error is None
            and parse.layout in ("modern", "vicell")
            and any(c.anchor for g in parse.games for c in g.cells)
        ):
            bmap, err = fetch_book_map(rec.capture_ts, html, limiter, snapshot_dir, budget)
            if bmap:
                book_maps[rec.capture_ts] = bmap
                book_map_sources[rec.capture_ts] = "own_line_movement_page"
            elif err is not None:
                book_map_errors[rec.capture_ts] = err
        parses.append(parse)
        state = parse.error or f"{len(parse.games)} games"
        print(f"  [{i + 1}/{len(ok_records)}] parse {rec.capture_ts}: {state}")

    union_map: dict[str, str] = {}
    for bmap in book_maps.values():
        for code, name in bmap.items():
            union_map.setdefault(code, name)
    for parse in parses:
        bmap = book_maps.get(parse.capture_ts)
        if not bmap and parse.error is None:
            board_anchors = {c.anchor for g in parse.games for c in g.cells if c.anchor}
            if board_anchors and board_anchors <= set(union_map):
                book_maps[parse.capture_ts] = union_map
                book_map_sources[parse.capture_ts] = "cross_capture_fallback"
        elif bmap:
            missing = {c.anchor for g in parse.games for c in g.cells if c.anchor} - set(bmap)
            fillable = missing & set(union_map)
            if fillable:
                for code in fillable:
                    bmap[code] = union_map[code]

    tidy = build_tidy(parses, book_maps)
    coverage = compute_coverage(season, tidy, parses, book_map_sources, schedules)
    if book_map_errors:
        coverage["book_identity"]["line_movement_fetch_errors"] = dict(
            list(book_map_errors.items())[:10]
        )

    parquet_path = artifacts_dir / f"season_{season}.parquet"
    tidy.to_parquet(parquet_path, index=False)
    coverage_path = artifacts_dir / f"coverage_{season}.json"
    coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(f"[{season}] wrote {parquet_path.name}, {coverage_path.name}")

    season_summary.update(
        {
            "status": "completed",
            "tidy_parquet": str(parquet_path),
            "coverage_json": str(coverage_path),
            "tidy_rows": len(tidy),
            "book_identity_fallback_rate": coverage["book_identity"]["fallback_rate"],
            "reduced_confidence": coverage["book_identity"]["reduced_confidence_flag"],
            "schedule_match": coverage.get("schedule_match"),
        }
    )
    return season_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-out", type=Path, default=Path("data/raw/vegasinsider"))
    parser.add_argument(
        "--artifacts-out", type=Path, default=Path("artifacts/vegasinsider_backfill")
    )
    parser.add_argument("--run-id", default=None, metavar="YYYYMMDDTHHMMSSZ")
    parser.add_argument("--seasons", type=int, nargs="+", required=True, metavar="YEAR")
    parser.add_argument("--captures", type=int, default=DEFAULT_CAPTURES_PER_SEASON)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--wall-clock-minutes", type=float, default=DEFAULT_WALL_CLOCK_MINUTES)
    args = parser.parse_args()

    if len(args.seasons) > MAX_SEASONS_PER_INVOCATION:
        parser.error(f"at most {MAX_SEASONS_PER_INVOCATION} seasons per invocation")

    requested = sorted(set(args.seasons))
    run_id = args.run_id or run_id_now()
    snapshot_dir = args.raw_out / run_id
    artifacts_dir = args.artifacts_out / run_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    delay = max(args.delay, MIN_DELAY_SECONDS)
    limiter = RateLimiter(delay)
    budget = WallClockBudget.from_minutes(args.wall_clock_minutes)
    schedules = latest_schedules_frame()
    print(f"Run directory: {snapshot_dir}")
    print(f"Artifacts directory: {artifacts_dir}")
    print(f"Schedules source: {'latest local snapshot' if schedules is not None else 'NONE'}")

    pending: list[int] = []
    already_done: list[int] = []
    for season in requested:
        if (artifacts_dir / f"season_{season}.parquet").exists():
            already_done.append(season)
        else:
            pending.append(season)

    season_results: list[dict[str, Any]] = [
        {"season": s, "status": "already_completed_previous_invocation"} for s in already_done
    ]
    stopped_reason = "all_complete"
    try:
        for season in pending:
            budget.check()
            result = process_season(
                season,
                snapshot_dir,
                artifacts_dir,
                limiter,
                budget,
                args.captures,
                schedules,
            )
            season_results.append(result)
    except BudgetExceeded:
        stopped_reason = "wall_clock_budget_exhausted"

    done_all_time = sorted(
        already_done + [r["season"] for r in season_results if r["status"] == "completed"]
    )
    remaining = [s for s in requested if s not in done_all_time]
    reduced = [r["season"] for r in season_results if r.get("reduced_confidence")]

    status = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "requested_seasons_this_invocation": requested,
        "completed_before_this_invocation": already_done,
        "completed_this_invocation": [
            r["season"] for r in season_results if r["status"] == "completed"
        ],
        "done_all_time": done_all_time,
        "remaining_after_this_invocation": remaining,
        "stopped_reason": stopped_reason,
        "wall_clock_budget_minutes": args.wall_clock_minutes,
        "delay_seconds": delay,
        "captures_per_season": args.captures,
        "reduced_confidence_seasons_fallback_gt_20pct": reduced,
        "season_summaries": season_results,
        "resumption_command": (
            f".\\.tools\\uv.exe run --no-sync python scripts/backfill_vegasinsider.py "
            f"--run-id {run_id} --seasons {' '.join(str(s) for s in remaining)}"
            if remaining
            else None
        ),
    }
    status_path = artifacts_dir / "status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Status: {status_path}")
    print(f"Done: {done_all_time}; remaining: {remaining}; stopped_reason: {stopped_reason}")
    if remaining:
        print(f"Resume with: {status['resumption_command']}")

    completed_rows = [
        r
        for r in season_results
        if r["status"] in ("completed", "already_completed_previous_invocation")
    ]
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": artifact_provenance(
            {
                "source": SOURCE_DESCRIPTION,
                "run_id": run_id,
                "requested_seasons_this_invocation": requested,
                "delay_seconds": delay,
                "captures_per_season": args.captures,
            },
            status_path,
            project_root=REPO,
        ),
    }
    write_experiment_artifact(
        artifacts_dir,
        "metadata.json",
        metadata,
        command="backfill_vegasinsider",
        metrics={
            "seasons_done_all_time": len(done_all_time),
            "seasons_remaining": len(remaining),
            "tidy_rows_completed_seasons": sum(r.get("tidy_rows", 0) for r in completed_rows),
            "reduced_confidence_seasons": len(reduced),
        },
        notes=(
            "Ingest backfill only: no ATS evaluation and no tracked-registry "
            "write (registry_root redirected inside the gitignored artifact "
            "snapshot). Lines are market quotes, never picks; nothing here "
            "speaks to ATS edge."
        ),
        source="scripts/backfill_vegasinsider.py",
        registry_root=artifacts_dir / "experiment_registry",
        project_root=REPO,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
