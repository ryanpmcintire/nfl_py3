"""Pilot ingest of VegasInsider NFL Las Vegas odds boards via Wayback Machine.

Source lead: docs/data_source_scout_v5.md Section F item 1. Bounded feasibility
pilot for the 2011 NFL season window (Sep-Dec 2011): query the Wayback CDX API,
select ~14 evenly-spaced status-200 captures preferring Tuesdays/Wednesdays,
fetch each through the raw `id_` endpoint with a polite delay and retry-once,
and save immutable snapshot files plus a sha256 manifest.

Board structure (measured this session on capture 20111102063341): game rows are
`<tr class='oddsText_odd|even'>` elements holding a nested info table
(MM/DD kickoff date, kickoff time, rotation number + team-name links) followed
by one `<td>` per sportsbook column containing a spread token (`-7-110`) and/or
a total token (`45u-110`). Book identities are rendered as a header IMAGE, but
every book cell links to a per-game line-movement page fragment (`#BT`, `#J`,
...) and those fragments define `<a name="CODE">` anchors immediately ahead of
`<BOOK NAME> LINE MOVEMENTS` headings, so an anchor-to-book-name map is
recoverable from one line-movement fetch per capture. Columns without anchors
(open-line / affiliate cells) keep a null book name rather than a guessed one.

No ATS evaluation, no registry writes: feasibility only.

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/pilot_vegasinsider_wayback.py
    .\\.tools\\uv.exe run --no-sync python scripts/pilot_vegasinsider_wayback.py `
        --run-id <YYYYMMDDTHHMMSSZ>   # reuse already-fetched raw snapshots
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

CDX_URL = (
    "https://web.archive.org/cdx/search/cdx"
    "?url=vegasinsider.com/nfl/odds/las-vegas/"
    "&from={from_date}&to={to_date}&filter=statuscode:200&output=json"
)
RAW_ENDPOINT = "https://web.archive.org/web/{ts}id_/{original}"
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
MIN_DELAY_SECONDS = 2.5
DEFAULT_CAPTURES = 14
DEFAULT_FROM = "20110901"
DEFAULT_TO = "20111231"

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
    "SAN FRANCISCO": "SF",
    "SEATTLE": "SEA",
    "TAMPA BAY": "TB",
    "TENNESSEE": "TEN",
    "WASHINGTON": "WAS",
}


def normalize_team(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = re.sub(r"\s+", " ", raw.strip().upper())
    if key in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[key]
    return key if key in FRANCHISE_CODES else None


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


def fetch_via_curl(url: str, limiter: RateLimiter, *, retries: int = 2) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for _attempt in range(retries):
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


def query_cdx(from_date: str, to_date: str, limiter: RateLimiter) -> list[CdxRow]:
    body, _ = fetch_via_curl(CDX_URL.format(from_date=from_date, to_date=to_date), limiter)
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
    reuse: bool,
) -> list[SnapshotRecord]:
    raw_dir = snapshot_dir / "snapshots"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[SnapshotRecord] = []
    for i, cap in enumerate(captures):
        path = raw_dir / f"{cap.timestamp}.html"
        wayback_url = RAW_ENDPOINT.format(ts=cap.timestamp, original=cap.original)
        if reuse and path.exists():
            raw = path.read_bytes()
            status, error = "cached", None
        else:
            try:
                raw, status_code = fetch_via_curl(wayback_url, limiter)
                status, error = status_code, None
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


def write_manifest(snapshot_dir: Path, records: list[SnapshotRecord]) -> Path:
    manifest = {
        "source": "vegasinsider.com/nfl/odds/las-vegas/ via web.archive.org raw endpoint",
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "user_agent": USER_AGENT,
        "rate_limit_seconds": MIN_DELAY_SECONDS,
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
    path = snapshot_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


@dataclass
class OddsCell:
    column_index: int
    anchor: str | None
    raw_text: str
    spread_line: float | None
    spread_vig: str | None
    total_line: float | None
    total_juice_side: str | None
    total_vig: str | None


@dataclass
class GameRow:
    game_date_raw: str | None
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


def parse_spread_token(token: str) -> tuple[float | None, str | None]:
    m = re.match(SPREAD_RE_TEMPLATE, token.strip(), re.IGNORECASE)
    if not m:
        return None, None
    head, vig = m.group(1), m.group(2)
    if head.upper() == "PK":
        return 0.0, vig
    try:
        value = float(head)
    except ValueError:
        return None, None
    if abs(value) > 80:
        return None, None
    return value, vig


def parse_total_token(token: str) -> tuple[float | None, str | None, str | None]:
    m = re.match(r"^(\d+(?:\.5)?)\s*([ouOU])?(?:\s*([+\-]\d{1,3}))?$", token.strip())
    if not m:
        return None, None, None
    try:
        value = float(m.group(1))
    except ValueError:
        return None, None, None
    if not 10 <= value <= 90:
        return None, None, None
    return value, (m.group(2).lower() if m.group(2) else None), m.group(3)


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
        spread: float | None = None
        spread_vig: str | None = None
        total: float | None = None
        total_side: str | None = None
        total_vig: str | None = None
        for token in tokens:
            if spread is None:
                candidate, cand_vig = parse_spread_token(token)
                if candidate is not None:
                    spread, spread_vig = candidate, cand_vig
                    continue
            if total is None:
                candidate, side, cand_vig = parse_total_token(token)
                if candidate is not None:
                    total, total_side, total_vig = candidate, side, cand_vig
        compact_raw = " | ".join(tokens)[:120]
        if not compact_raw:
            continue
        cells.append(
            OddsCell(
                column_index=idx,
                anchor=anchor_m.group(1).upper() if anchor_m else None,
                raw_text=compact_raw,
                spread_line=spread,
                spread_vig=spread_vig,
                total_line=total,
                total_juice_side=total_side,
                total_vig=total_vig,
            )
        )
    return cells


def parse_board(capture_ts: str, html: str, capture_year: int) -> BoardParse:
    updated_m = re.search(r"Updated:\s*[^<\n]+", html)
    updated_line = re.sub(r"\s+", " ", updated_m.group(0)).strip() if updated_m else None
    tags = list(ROW_TAG_RE.finditer(html))
    if not tags:
        return BoardParse(
            capture_ts=capture_ts,
            updated_line=updated_line,
            games=[],
            error="no oddsText rows found",
        )
    games: list[GameRow] = []
    for i, m in enumerate(tags):
        end = tags[i + 1].start() if i + 1 < len(tags) else len(html)
        chunk = html[m.end() : end]
        kick = KICK_RE.search(chunk)
        teams = TEAM_RE.findall(chunk)
        if not kick or len(teams) < 2:
            continue
        date_raw = kick.group(1)
        parts = date_raw.split("/")
        month, day = int(parts[0]), int(parts[1])
        year_val = capture_year
        if month <= 2 and capture_year % 100 == 12:
            year_val += 1
        try:
            date_iso = f"{year_val:04d}-{month:02d}-{day:02d}"
        except ValueError:
            date_iso = None
        games.append(
            GameRow(
                game_date_raw=date_raw,
                game_date_iso=date_iso,
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
    reuse: bool,
) -> tuple[dict[str, str], str | None]:
    lm_dir = snapshot_dir / "line_movement"
    lm_dir.mkdir(parents=True, exist_ok=True)
    seen_paths: list[str] = []
    for link_m in re.finditer(r'href="(/nfl/odds/las-vegas/line-movement/[^"]+)"', board_html):
        rel = link_m.group(1).split("#", 1)[0]
        if rel not in seen_paths:
            seen_paths.append(rel)
    if not seen_paths:
        return {}, "no line-movement link found on board"
    errors: list[str] = []
    for rel in seen_paths[:5]:
        lm_path = lm_dir / f"{capture_ts}_{hashlib.md5(rel.encode()).hexdigest()[:8]}.html"
        if reuse and lm_path.exists():
            html = lm_path.read_bytes().decode("utf-8", errors="replace")
        else:
            url = f"https://web.archive.org/web/{capture_ts}id_/http://www.vegasinsider.com{rel}"
            try:
                raw, _ = fetch_via_curl(url, limiter)
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


def build_tidy(
    parses: list[BoardParse],
    book_maps: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for parse in parses:
        bmap = book_maps.get(parse.capture_ts, {})
        for game in parse.games:
            for cell in game.cells:
                rows.append(
                    {
                        "capture_ts": parse.capture_ts,
                        "game_date_raw": game.game_date_raw,
                        "game_date_iso": game.game_date_iso,
                        "kickoff_time": game.kickoff_time,
                        "away_team": game.away_name,
                        "home_team": game.home_name,
                        "away_rotation": game.away_rotation,
                        "home_rotation": game.home_rotation,
                        "away_code": normalize_team(game.away_name),
                        "home_code": normalize_team(game.home_name),
                        "column_index": cell.column_index,
                        "book_anchor": cell.anchor,
                        "book_name": bmap.get(cell.anchor) if cell.anchor else None,
                        "spread_line": cell.spread_line,
                        "spread_vig": cell.spread_vig,
                        "total_line": cell.total_line,
                        "total_juice_side": cell.total_juice_side,
                        "total_vig": cell.total_vig,
                        "raw_cell_text": cell.raw_text,
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["capture_ts", "game_date_iso", "away_rotation", "column_index"]
        ).reset_index(drop=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-out", type=Path, default=Path("data/raw/vegasinsider"))
    parser.add_argument("--artifacts-out", type=Path, default=Path("artifacts/vegasinsider_pilot"))
    parser.add_argument("--run-id", default=None, metavar="YYYYMMDDTHHMMSSZ")
    parser.add_argument("--captures", type=int, default=DEFAULT_CAPTURES)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--from-date", default=DEFAULT_FROM)
    parser.add_argument("--to-date", default=DEFAULT_TO)
    parser.add_argument(
        "--skip-fetch", action="store_true", help="reuse raw snapshots already in the run dir"
    )
    args = parser.parse_args()

    delay = max(args.delay, MIN_DELAY_SECONDS)
    limiter = RateLimiter(delay)
    run_id = args.run_id or run_id_now()
    snapshot_dir = args.raw_out / run_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {snapshot_dir}")

    cdx_cache = snapshot_dir / "cdx.json"
    if cdx_cache.exists():
        payload = json.loads(cdx_cache.read_text(encoding="utf-8"))
        all_rows = [CdxRow(**r) for r in payload]
        print(f"CDX cached: {len(all_rows)} unique-digest captures")
    else:
        all_rows = query_cdx(args.from_date, args.to_date, limiter)
        cdx_cache.write_text(json.dumps([r.__dict__ for r in all_rows], indent=2), encoding="utf-8")
        print(f"CDX measured: {len(all_rows)} unique-digest captures")

    captures = select_captures(all_rows, args.captures)
    print(f"Selected {len(captures)} captures")

    records = fetch_snapshots(snapshot_dir, captures, limiter, reuse=args.skip_fetch)
    manifest_path = write_manifest(snapshot_dir, records)
    print(f"Manifest: {manifest_path}")

    parses: list[BoardParse] = []
    book_maps: dict[str, dict[str, str]] = {}
    book_map_errors: dict[str, str] = {}
    book_map_sources: dict[str, str] = {}
    capture_years = {r.timestamp: int(r.timestamp[:4]) for r in captures}
    ok_records = [r for r in records if r.file is not None]
    for i, rec in enumerate(ok_records):
        html = (snapshot_dir / rec.file).read_bytes().decode("utf-8", errors="replace")
        parse = parse_board(rec.capture_ts, html, capture_years[rec.capture_ts])
        if parse.error is None and any(c.anchor for g in parse.games for c in g.cells):
            bmap, err = fetch_book_map(
                rec.capture_ts, html, limiter, snapshot_dir, reuse=args.skip_fetch
            )
            book_maps[rec.capture_ts] = bmap
            if err:
                book_map_errors[rec.capture_ts] = err
        parses.append(parse)
        state = parse.error or f"{len(parse.games)} games"
        print(f"  [{i + 1}/{len(ok_records)}] parse {rec.capture_ts}: {state}")

    union_map: dict[str, str] = {}
    for bmap in book_maps.values():
        for code, name in bmap.items():
            if code in union_map and union_map[code] != name:
                print(
                    f"  WARNING anchor {code} maps to both "
                    f"{union_map[code]} and {name}; keeping first"
                )
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
                book_map_sources[parse.capture_ts] = (
                    "own_line_movement_page_plus_cross_capture_fallback"
                )

    artifacts_dir = args.artifacts_out / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tidy = build_tidy(parses, book_maps)
    tidy_path = artifacts_dir / "tidy_rows.csv"
    tidy.to_csv(tidy_path, index=False)

    unmapped = pd.concat(
        [
            tidy.loc[tidy["away_code"].isna(), "away_team"],
            tidy.loc[tidy["home_code"].isna(), "home_team"],
        ]
    ).dropna()
    normalization_report = {
        "rows_total": len(tidy),
        "team_slots_total": int(len(tidy) * 2) if len(tidy) else 0,
        "team_slots_mapped": int(tidy["away_code"].notna().sum() + tidy["home_code"].notna().sum())
        if len(tidy)
        else 0,
        "unmapped_values": {str(k): int(v) for k, v in unmapped.value_counts().items()},
        "alias_table_applied": TEAM_NAME_ALIASES,
        "zero_silent_mappings_note": (
            "any raw name absent from the explicit alias/franchise tables lands "
            "here and stays null downstream"
        ),
    }
    (artifacts_dir / "normalization_report.json").write_text(
        json.dumps(normalization_report, indent=2), encoding="utf-8"
    )

    anchored = tidy[tidy["book_anchor"].notna()] if len(tidy) else tidy
    spread_cov = float(anchored["spread_line"].notna().mean()) if len(anchored) else 0.0
    total_cov = float(anchored["total_line"].notna().mean()) if len(anchored) else 0.0

    books_per_page: list[dict[str, Any]] = []
    for parse in parses:
        counts: dict[str, int] = {}
        unmapped_cols = 0
        for game in parse.games:
            for cell in game.cells:
                if cell.anchor is None:
                    continue
                name = book_maps.get(parse.capture_ts, {}).get(cell.anchor)
                if name is None:
                    unmapped_cols += 1
                else:
                    counts[name] = counts.get(name, 0) + 1
        books_per_page.append(
            {
                "capture_ts": parse.capture_ts,
                "updated_line": parse.updated_line,
                "games_parsed": len(parse.games),
                "book_cell_counts": dict(sorted(counts.items())),
                "columns_without_resolved_name": unmapped_cols,
                "book_map_error": book_map_errors.get(parse.capture_ts),
                "book_map_source": book_map_sources.get(
                    parse.capture_ts,
                    "own_line_movement_page" if parse.capture_ts in book_maps else None,
                ),
                "parse_error": parse.error,
            }
        )

    named_sets = [set(p["book_cell_counts"]) for p in books_per_page if p["book_cell_counts"]]
    consistent = (
        "y"
        if len(named_sets) >= 2 and all(s == named_sets[0] for s in named_sets)
        else ("n" if len(named_sets) >= 2 else "indeterminate_single_page_sample")
    )

    pages_ok = sum(1 for p in parses if p.error is None and p.games)
    pages_failed = sum(1 for p in parses if not (p.error is None and p.games))
    fetch_failed = sum(1 for r in records if r.file is None)
    parse_success_rate = round(pages_ok / len(parses), 4) if parses else 0.0

    sample_rows = tidy.head(5).to_dict(orient="records") if len(tidy) else []

    feasibility = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "window": {"from": args.from_date, "to": args.to_date},
        "cdx_unique_digest_captures_in_window": len(all_rows),
        "captures_selected": len(captures),
        "pages_fetched": len(records) - fetch_failed,
        "pages_fetch_failed": fetch_failed,
        "pages_parsed": pages_ok,
        "pages_parse_failed": pages_failed,
        "parse_success_rate": parse_success_rate,
        "tidy_rows_total": len(tidy),
        "anchored_book_cells": len(anchored),
        "spread_coverage_rate_over_anchored_cells": round(spread_cov, 4),
        "total_coverage_rate_over_anchored_cells": round(total_cov, 4),
        "books_detected_per_page": books_per_page,
        "books_consistent_across_pages": consistent,
        "sample_rows_verbatim": sample_rows,
        "effort_estimate_full_backfill_2005_2016": "M",
        "blockers": [],
    }
    feasibility_path = artifacts_dir / "feasibility.json"
    feasibility_path.write_text(json.dumps(feasibility, indent=2), encoding="utf-8")

    configuration = {
        "source": "vegasinsider.com/nfl/odds/las-vegas/ via Wayback raw endpoint",
        "window": [args.from_date, args.to_date],
        "captures_requested": args.captures,
        "delay_seconds": delay,
        "raw_snapshot_dir": str(snapshot_dir.resolve()),
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": artifact_provenance(configuration, tidy_path, project_root=REPO),
    }
    metrics = {
        "pages_fetched": feasibility["pages_fetched"],
        "pages_parsed": feasibility["pages_parsed"],
        "parse_success_rate": parse_success_rate,
        "tidy_rows": len(tidy),
        "spread_coverage_rate": round(spread_cov, 4),
        "total_coverage_rate": round(total_cov, 4),
        "books_consistent_across_pages": consistent,
    }
    write_experiment_artifact(
        artifacts_dir,
        "metadata.json",
        metadata,
        command="pilot_vegasinsider_wayback",
        metrics=metrics,
        notes=(
            "Feasibility pilot only: no ATS evaluation, no tracked-registry "
            "write (registry_root redirected inside the gitignored artifact "
            "snapshot). Historical line accuracy must not be inferred from "
            "this pilot."
        ),
        source="scripts/pilot_vegasinsider_wayback.py",
        registry_root=artifacts_dir / "experiment_registry",
        project_root=REPO,
    )

    print(
        f"Tidy rows: {len(tidy)}; parse success {parse_success_rate}; "
        f"spread cov {spread_cov:.3f}; total cov {total_cov:.3f}"
    )
    print(f"Feasibility: {feasibility_path}")


if __name__ == "__main__":
    main()
