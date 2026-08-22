"""Ingest pilot for SEC Student-Athlete Availability Reports (scout v5 Sec D #2).

Fetches secsports.com/fbreports and /fbreports-archive (HTML pages, no PDFs),
extracts the embedded report surfaces (a public Google Sheet plus a gated
third-party viewer), downloads every readable tab of the master Google Sheet as
CSV, enumerates Wayback Machine point-in-time captures of that sheet, and parses
everything into tidy rows (season/week/date/team/player/status).

Week and date resolution uses the sheet's own 2024 schedule tab, not inference.
Team names are normalized to stable SEC codes with an unmapped-name report.

No registry writes: the experiment-registry stamp is redirected inside the
gitignored artifact directory. No ATS evaluation, no XLG-03 join.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from html import unescape
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

FBREPORTS_URL = "https://www.secsports.com/fbreports"
FBREPORTS_ARCHIVE_URL = "https://www.secsports.com/fbreports-archive"
DEFAULT_SHEET_ID = "1m9NvaYU1N4ViI4MLrXLoTp5SdYYAp2tlWgxS5t9triM"
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
MIN_DELAY_SECONDS = 2.0

DATE_HEADER_RE = re.compile(
    r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*,?\s+"
    r"([a-z]+)\s+(\d{1,2})(st|nd|rd|th)?\s*$",
    re.IGNORECASE,
)
WINDOW_STAMP_RE = re.compile(
    r"^(?P<window>[^:]+?):\s*Updated as of\s+(?P<month>\d{1,2})/(?P<day>\d{1,2})"
    r"\s+at\s+(?P<time>\d{1,2}:\d{2}\s*[AP]M\s*[A-Z]{2})\s*$",
    re.IGNORECASE,
)
KICKOFF_RE = re.compile(r"^\d{1,2}:\d{2}\s*(AM|PM)", re.IGNORECASE)
TAB_ITEM_RE = re.compile(r'items\.push\(\{name:\s*"([^"]+)".*?gid:\s*"?(\d+)"?', re.DOTALL)
MONTHS: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
TEAM_CODES: dict[str, str] = {
    "alabama": "ALA",
    "arkansas": "ARK",
    "auburn": "AUB",
    "florida": "FLA",
    "georgia": "UGA",
    "kentucky": "UK",
    "lsu": "LSU",
    "ole miss": "MISS",
    "mississippi": "MISS",
    "mississippi state": "MSST",
    "missouri": "MIZ",
    "oklahoma": "OKLA",
    "south carolina": "SCAR",
    "tennessee": "TENN",
    "texas": "TEX",
    "texas a&m": "TAMU",
    "vanderbilt": "VAN",
}

REPO = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO / "data" / "raw" / "sec_availability"
ARTIFACT_ROOT = REPO / "artifacts" / "sec_pilot"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def polite_sleep(seconds: float) -> None:
    time.sleep(max(seconds, MIN_DELAY_SECONDS))


def fetch(session: requests.Session, url: str, timeout: int = 90) -> bytes:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


class PilotFetcher:
    def __init__(self, session: requests.Session, delay: float) -> None:
        self.session = session
        self.delay = delay
        self.manifest: list[dict[str, Any]] = []

    def get(self, url: str, label: str, out_path: Path | None = None) -> bytes:
        polite_sleep(self.delay)
        payload = fetch(self.session, url)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(payload)
        self.manifest.append(
            {
                "label": label,
                "url": url,
                "filename": out_path.name if out_path else None,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
                "fetched_at_utc": utc_now_iso(),
            }
        )
        return payload


def extract_embeds(page_html: str) -> dict[str, Any]:
    match = re.search(r'data-page="([^"]+)"', page_html)
    blob = page_html
    if match is not None:
        try:
            page = json.loads(unescape(match.group(1)))
            blocks = page.get("props", {}).get("page", {}).get("content_blocks", [])
            blob = json.dumps(blocks)
        except json.JSONDecodeError:
            blob = page_html
    sheets = sorted(set(re.findall(r"spreadsheets/d/([A-Za-z0-9_-]{20,})", blob)))
    hdi = sorted(set(re.findall(r"https://[a-z.]*hdintelligence-app\.com[^\"\\\s<]*", blob)))
    iframes = sorted(set(re.findall(r"<iframe[^>]*src=\\?\"([^\\\"\s]+)", blob)))
    return {"google_sheet_ids": sheets, "hdintelligence_urls": hdi, "iframe_srcs": iframes}


def sheet_tab_gids(pubhtml_text: str) -> dict[str, str]:
    text = unescape(pubhtml_text).replace("\\/", "/").replace("\\x3d", "=")
    tabs: dict[str, str] = {}
    for name, gid in TAB_ITEM_RE.findall(text):
        if name not in tabs:
            tabs[name] = gid
    return tabs


def read_csv_rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


def canonical_window(raw: str) -> str:
    return re.sub(r"[^a-z]", "", raw.strip().casefold())


def normalize_team(raw: str) -> str | None:
    key = re.sub(r"\s+", " ", raw.strip().casefold())
    return TEAM_CODES.get(key)


def parse_schedule(rows: list[list[str]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    header = [cell.strip() for cell in rows[0]]
    idx = {name: i for i, name in enumerate(header)}
    week_i = idx["Week"]
    visitor_i = idx["Visitor (Team 1)"]
    home_i = idx["Home (Team 2)"]
    date_i = idx["Game Date"]
    games: list[dict[str, Any]] = []
    for row in rows[1:]:
        if len(row) <= max(week_i, visitor_i, home_i, date_i):
            continue
        date_text = row[date_i].strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_text):
            continue
        games.append(
            {
                "week": int(row[week_i]),
                "visitor": row[visitor_i].strip(),
                "home": row[home_i].strip(),
                "game_date": date_text,
            }
        )
    return games


def schedule_index(
    games: list[dict[str, Any]],
) -> dict[tuple[frozenset[str], tuple[int, int]], dict[str, Any]]:
    index: dict[tuple[frozenset[str], tuple[int, int]], dict[str, Any]] = {}
    for game in games:
        pair = frozenset({game["visitor"].casefold(), game["home"].casefold()})
        month_day = (int(game["game_date"][5:7]), int(game["game_date"][8:10]))
        index[(pair, month_day)] = game
    return index


def parse_formatted_grid(rows: list[list[str]]) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    pending_matchup = False
    for row in rows:
        cells = [(cell or "").strip() for cell in row]
        populated = [cell for cell in cells if cell]
        if not populated:
            pending_matchup = False
            state.pop("team", None)
            continue
        joined = populated[0]
        date_match = DATE_HEADER_RE.match(joined)
        if date_match is not None and MONTHS.get(date_match.group(2).casefold()):
            state = {
                "date_month": MONTHS[date_match.group(2).casefold()],
                "date_day": int(date_match.group(3)),
                "date_raw": joined,
                "window": None,
                "updated_raw": None,
                "kickoff": None,
                "reports": {},
            }
            games.append(state)
            pending_matchup = False
            continue
        stamp_match = WINDOW_STAMP_RE.match(joined)
        if stamp_match is not None and state:
            window_name = canonical_window(stamp_match.group("window"))
            state["window"] = window_name
            state["updated_raw"] = (
                f"{stamp_match.group('month')}/{stamp_match.group('day')} "
                f"{stamp_match.group('time')}"
            )
            pending_matchup = False
            continue
        if KICKOFF_RE.match(joined):
            if state:
                state["kickoff"] = joined
            pending_matchup = False
            continue
        if "AT" in cells and len(populated) == 1:
            pending_matchup = True
            continue
        if pending_matchup and len(populated) == 2 and "team_away" not in state:
            state["team_away"] = populated[0]
            state["team_home"] = populated[1]
            pending_matchup = False
            continue
        if "date_raw" not in state:
            continue
        if len(populated) == 1 and populated[0].casefold() != "end":
            state["team"] = populated[0]
            state.setdefault("reports", {}).setdefault(populated[0], [])
            continue
        if len(populated) >= 3 and "team" in state:
            team = state["team"]
            if populated[0].casefold() == "player":
                continue
            if populated[0].casefold() == "end" and populated[-1].casefold() == "end":
                state.pop("team", None)
                continue
            state["reports"][team].append(
                {
                    "player": populated[0],
                    "position": populated[1],
                    "status_raw": populated[2],
                }
            )
    return [game for game in games if "team_away" in game]


def resolve_game(
    game: dict[str, Any], index: dict[tuple[frozenset[str], tuple[int, int]], dict[str, Any]]
) -> dict[str, Any] | None:
    pair = frozenset({game["team_away"].casefold(), game["team_home"].casefold()})
    return index.get((pair, (game["date_month"], game["date_day"])))


def tidy_from_grid(
    games: list[dict[str, Any]],
    index: dict[tuple[frozenset[str], tuple[int, int]], dict[str, Any]],
    source: str,
    captured_at_utc: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    unmapped: list[str] = []
    for game in games:
        scheduled = resolve_game(game, index)
        season = int(scheduled["game_date"][:4]) if scheduled else None
        week = scheduled["week"] if scheduled else None
        game_date = scheduled["game_date"] if scheduled else None
        if scheduled is None:
            unmapped.append(f"{game['team_away']} @ {game['team_home']} ({game['date_raw']})")
        for side, team_name in (("away", game["team_away"]), ("home", game["team_home"])):
            code = normalize_team(team_name)
            if code is None:
                unmapped.append(team_name)
            for entry in game["reports"].get(team_name, []):
                rows.append(
                    {
                        "source": source,
                        "captured_at_utc": captured_at_utc,
                        "season": season,
                        "week": week,
                        "game_date": game_date,
                        "side": side,
                        "window": game["window"],
                        "updated_stamp_raw": game["updated_raw"],
                        "team_raw": team_name,
                        "team_code": code,
                        "player": entry["player"],
                        "position": entry["position"],
                        "status_raw": entry["status_raw"],
                        "status_norm": entry["status_raw"].strip().casefold(),
                    }
                )
    return rows, sorted(set(unmapped))


def tidy_from_tab(
    tab_name: str,
    rows: list[list[str]],
    games: list[dict[str, Any]],
    index: dict[tuple[frozenset[str], tuple[int, int]], dict[str, Any]],
    captured_at_utc: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    tidy: list[dict[str, Any]] = []
    unmapped: list[str] = []
    if not rows:
        return tidy, unmapped
    header = [cell.strip() for cell in rows[0]]
    if header[:4] != ["School", "Full Name", "Position", "Status"]:
        return tidy, [f"{tab_name}: unexpected header {header[:4]}"]
    schools = sorted({row[0].strip() for row in rows[1:] if len(row) >= 4 and row[0].strip()})
    school_keys = frozenset(name.casefold() for name in schools)
    matched: dict[str, Any] | None = None
    for game in games:
        game_pair = frozenset({game["team_away"].casefold(), game["team_home"].casefold()})
        reported_teams = frozenset(team.casefold() for team in game["reports"]) | school_keys
        if game_pair == reported_teams and resolve_game(game, index) is not None:
            matched = game
            break
    if matched is None:
        return tidy, [f"{tab_name}: no formatted game matches schools {schools}"]
    scheduled = resolve_game(matched, index)
    if scheduled is None:
        return tidy, [f"{tab_name}: matched game unresolved against schedule"]
    for row in rows[1:]:
        if len(row) < 4 or not row[0].strip():
            continue
        team_name = row[0].strip()
        code = normalize_team(team_name)
        if code is None:
            unmapped.append(team_name)
        tidy.append(
            {
                "source": f"tab_{tab_name}",
                "captured_at_utc": captured_at_utc,
                "season": int(scheduled["game_date"][:4]),
                "week": scheduled["week"],
                "game_date": scheduled["game_date"],
                "side": None,
                "window": canonical_window(tab_name),
                "updated_stamp_raw": None,
                "team_raw": team_name,
                "team_code": code,
                "player": row[1].strip(),
                "position": row[2].strip(),
                "status_raw": row[3].strip(),
                "status_norm": row[3].strip().casefold(),
            }
        )
    return tidy, sorted(set(unmapped))


def coverage_stats(frame: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {"overall": {"rows": len(frame)}}
    numeric = frame[frame["week"].notna()].copy()
    if not numeric.empty:
        numeric["week_int"] = numeric["week"].astype(int)
    for week, group in numeric.groupby("week_int"):
        statuses = Counter(group["status_norm"].tolist())
        windows = sorted({str(w) for w in group["window"]})
        stats[f"week_{week}"] = {
            "rows": len(group),
            "team_games": int(group.groupby(["game_date", "team_away_key"]).ngroups)
            if "team_away_key" in group.columns
            else len(group),
            "teams": sorted({str(t) for t in group["team_code"]}),
            "players": int(group["player"].nunique()),
            "statuses": dict(sorted(statuses.items())),
            "windows": windows,
            "game_dates": sorted({str(d) for d in group["game_date"]}),
        }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet-id", default=DEFAULT_SHEET_ID)
    parser.add_argument("--out-dir", type=Path, default=RAW_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--delay-seconds", type=float, default=2.5)
    parser.add_argument("--max-wayback-captures", type=int, default=20)
    args = parser.parse_args()

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = args.out_dir / run_id
    artifact_dir = args.artifact_dir / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    status: dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": utc_now_iso(),
        "structure_found": None,
        "pages_fetched": [],
        "embeds": {},
        "sheet_tabs": {},
        "tabs_downloaded": 0,
        "wayback_captures_found": 0,
        "wayback_captures_distinct": 0,
        "tidy_rows": 0,
        "weeks_captured": [],
        "unmapped_report": [],
        "errors": [],
    }

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    fetcher = PilotFetcher(session, args.delay_seconds)

    pages: dict[str, str] = {}
    for label, url in (("fbreports", FBREPORTS_URL), ("fbreports_archive", FBREPORTS_ARCHIVE_URL)):
        try:
            payload = fetcher.get(url, f"page_{label}", raw_dir / f"{label}.html")
            pages[label] = payload.decode("utf-8", errors="replace")
            status["pages_fetched"].append({"label": label, "url": url, "chars": len(pages[label])})
        except requests.RequestException as error:
            status["errors"].append(f"{label}: {error}")

    for label, page in pages.items():
        status["embeds"][label] = extract_embeds(page)

    sheet_id = args.sheet_id
    discovered = status["embeds"].get("fbreports", {}).get("google_sheet_ids") or []
    if discovered:
        sheet_id = discovered[0]

    tabs: dict[str, str] = {}
    try:
        pubhtml = fetcher.get(
            f"https://docs.google.com/spreadsheets/d/{sheet_id}/pubhtml",
            f"sheet_{sheet_id}_pubhtml",
            raw_dir / "sheet_master_pubhtml.html",
        ).decode("utf-8", errors="replace")
        tabs = sheet_tab_gids(pubhtml)
    except requests.RequestException as error:
        status["errors"].append(f"pubhtml: {error}")
    status["sheet_tabs"] = tabs

    tab_texts: dict[str, str] = {}
    for tab_name, gid in tabs.items():
        slug = re.sub(r"[^A-Za-z0-9]+", "_", tab_name).strip("_")
        url = (
            f"https://docs.google.com/spreadsheets/d/{sheet_id}"
            f"/pub?gid={gid}&single=true&output=csv"
        )
        try:
            payload = fetcher.get(url, f"tab_{tab_name}", raw_dir / f"tab_{slug}.csv")
            tab_texts[tab_name] = payload.decode("utf-8", errors="replace")
            status["tabs_downloaded"] += 1
        except requests.RequestException as error:
            status["errors"].append(f"tab {tab_name}: {error}")
    status["manifest_path"] = None

    cdx_url = (
        "https://web.archive.org/cdx/search/cdx"
        f"?url=docs.google.com/spreadsheets/d/{sheet_id}&matchType=prefix"
        "&output=json&limit=200&collapse=digest"
    )
    captures: list[list[str]] = []
    for attempt in range(4):
        try:
            payload = fetcher.get(cdx_url, "wayback_cdx")
            parsed = json.loads(payload.decode("utf-8"))
            captures = [row for row in parsed[1:] if len(row) > 4 and row[4] == "200"]
            break
        except (requests.RequestException, json.JSONDecodeError, IndexError) as error:
            status["errors"].append(f"cdx attempt {attempt}: {error}")
            time.sleep(3)
    status["wayback_captures_found"] = len(captures)

    seen_hashes: set[str] = set()
    wayback_games: list[dict[str, Any]] = []
    wayback_capture_utc_by_state: dict[int, str] = {}
    for row in captures[: max(args.max_wayback_captures, 0)]:
        timestamp, original = row[1], row[2]
        url = f"https://web.archive.org/web/{timestamp}/{original}"
        try:
            payload = fetcher.get(
                url, f"wayback_{timestamp}", raw_dir / f"wayback_{timestamp}.html"
            )
        except requests.RequestException as error:
            status["errors"].append(f"wayback {timestamp}: {error}")
            continue
        text = payload.decode("utf-8", errors="replace")
        tables = re.findall(r"<table[^>]*>.*?</table>", text, re.DOTALL | re.IGNORECASE)
        target = next(
            (t for t in tables if ">Player<" in t and ">Position<" in t and ">Status<" in t),
            None,
        )
        if target is None:
            status["errors"].append(f"wayback {timestamp}: no report table found")
            continue
        grid_rows: list[list[str]] = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", target, re.DOTALL | re.IGNORECASE):
            cells = [
                unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", cell))).strip()
                for cell in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL | re.IGNORECASE)
                if re.sub(r"<[^>]+>", "", cell).strip()
            ]
            if cells:
                grid_rows.append(cells)
        content_digest = sha256_bytes(json.dumps(grid_rows).encode("utf-8"))
        if content_digest in seen_hashes:
            continue
        seen_hashes.add(content_digest)
        parsed = parse_formatted_grid(grid_rows)
        wayback_capture_utc_by_state[len(wayback_games)] = (
            datetime.strptime(timestamp[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC).isoformat()
        )
        wayback_games.extend(parsed)

    status["wayback_captures_distinct"] = len(seen_hashes)

    schedule_rows = read_csv_rows(tab_texts.get("2024 Football Schedule", ""))
    schedule = parse_schedule(schedule_rows)
    index = schedule_index(schedule)

    tidy: list[dict[str, Any]] = []
    unmapped_all: list[str] = []
    formatted_rows = read_csv_rows(tab_texts.get("FORMATTED-PUBLIC", "")) or read_csv_rows(
        tab_texts.get("Formatted-Data", "")
    )
    live_games = parse_formatted_grid(formatted_rows)
    captured_live = next(
        (
            entry["fetched_at_utc"]
            for entry in reversed(fetcher.manifest)
            if entry["label"].startswith("tab_FORMATTED")
        ),
        utc_now_iso(),
    )
    live_tidy, live_unmapped = tidy_from_grid(live_games, index, "live_csv", captured_live)
    tidy.extend(live_tidy)
    unmapped_all.extend(live_unmapped)

    wb_capture_utc = next(iter(wayback_capture_utc_by_state.values()), utc_now_iso())
    wb_tidy, wb_unmapped = tidy_from_grid(wayback_games, index, "wayback_html", wb_capture_utc)
    tidy.extend(wb_tidy)
    unmapped_all.extend(wb_unmapped)

    for tab_name in ("InitialReport", "ThursdayUpdate", "FridayUpdate", "GamedayUpdate"):
        if tab_name not in tab_texts:
            continue
        tab_tidy, tab_unmapped = tidy_from_tab(
            tab_name, read_csv_rows(tab_texts[tab_name]), live_games, index, captured_live
        )
        tidy.extend(tab_tidy)
        unmapped_all.extend(tab_unmapped)

    seen_unmapped: set[str] = set()
    deduped_unmapped: list[str] = []
    for item in unmapped_all:
        if item not in seen_unmapped:
            seen_unmapped.add(item)
            deduped_unmapped.append(item)
    status["unmapped_report"] = deduped_unmapped

    frame = pd.DataFrame(tidy)
    status["tidy_rows"] = len(frame)
    status["weeks_captured"] = sorted({int(w) for w in frame["week"].dropna().unique()})

    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at_utc": utc_now_iso(),
                "sheet_id": sheet_id,
                "entries": fetcher.manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    status["manifest_path"] = str(manifest_path.relative_to(REPO))

    tidy_path = artifact_dir / "tidy_rows.parquet"
    csv_path = artifact_dir / "tidy_rows.csv"
    frame.to_parquet(tidy_path, index=False)
    frame.to_csv(csv_path, index=False)

    away_keys = frame.copy()
    away_keys["team_away_key"] = away_keys.apply(
        lambda r: f"{r['game_date']}|{r['team_code']}" if pd.notna(r["team_code"]) else "",
        axis=1,
    )
    coverage = coverage_stats(away_keys)
    coverage_path = artifact_dir / "coverage_stats.json"
    coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    status["structure_found"] = {
        "sec_pages": "html (Inertia/Vue CMS); no PDF links on either page",
        "current_view": "gated third-party React app (hdintelligence-app.com) embedded by iframe",
        "archive_view": (
            "same gated app (?source=SECarchive); its JSON API returns empty "
            "to unauthenticated callers"
        ),
        "ingestible_surface": (
            "public Google Sheet linked from /fbreports; every tab readable as CSV "
            "via /pub?gid=<gid>&single=true&output=csv without authentication"
        ),
        "pdf_parsers_needed": False,
    }
    status["point_in_time_notes"] = {
        "live_tabs_hold_last_written_week_only": True,
        "history_recovery": "Wayback captures of the pubhtml URL only",
        "school_workbooks": "linked from SchoolWorkbooks tab but HTTP 401 without login",
    }
    status["finished_at_utc"] = utc_now_iso()
    status["provenance"] = artifact_provenance(
        {
            "source": "secsports.com fbreports + embedded public Google Sheet + web.archive.org",
            "run_id": run_id,
            "sheet_id": sheet_id,
            "delay_seconds": max(args.delay_seconds, MIN_DELAY_SECONDS),
        },
        manifest_path,
        project_root=REPO,
    )
    write_experiment_artifact(
        artifact_dir,
        "pilot_status.json",
        status,
        command="pilot_sec_availability",
        metrics={
            "tabs_downloaded": status["tabs_downloaded"],
            "wayback_captures_distinct": status["wayback_captures_distinct"],
            "tidy_rows": status["tidy_rows"],
            "weeks_captured_count": len(status["weeks_captured"]),
        },
        notes=(
            "Ingest snapshot+parsing pilot only: no registry write (registry_root "
            "redirected inside the gitignored artifact snapshot), no XLG-03 join, "
            "no ATS evaluation."
        ),
        source="scripts/pilot_sec_availability.py",
        registry_root=artifact_dir / "experiment_registry",
        project_root=REPO,
    )

    print(
        json.dumps(
            {
                "structure": status["structure_found"]["sec_pages"],
                "tabs_downloaded": status["tabs_downloaded"],
                "wayback_distinct_states": status["wayback_captures_distinct"],
                "tidy_rows": status["tidy_rows"],
                "weeks_captured": status["weeks_captured"],
                "unmapped": deduped_unmapped,
                "errors": status["errors"],
            },
            indent=2,
        )
    )
    print(f"artifacts: {artifact_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
