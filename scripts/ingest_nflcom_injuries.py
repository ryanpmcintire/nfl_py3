"""Immutable-snapshot ingester for NFL.com weekly league injury reports.

Source: https://www.nfl.com/injuries/league/{season}/reg{week} (plain HTML;
per-player practice status and game status per team table). Scope: REG weeks
1-18 for seasons 2022-2024, the era the local nflverse injuries feed covers
before its post-2024 death, so Stage 1b agreement can be measured.

robots.txt is fetched and evaluated BEFORE any page fetch (fail-closed); as
measured 2026-08-21, www.nfl.com/robots.txt disallows nothing under /injuries/
and sets no Crawl-delay; this script still enforces a polite >= 2s delay.

Snapshot convention: data/raw/nflcom_injuries/<UTC ts>/pages/*.html plus a
manifest.json carrying one sha256 per page and every fetch failure. The
snapshot directory must stay nested under --out exactly like every other raw
source (a manifest.json directly at data/raw/ root would be mistaken for a
schedules snapshot by nfl_ats.snapshots.latest_snapshot()).

--agreement runs Stage 1b: joins the parsed snapshot against the local nflverse
injuries feed (data/players/raw/*/injuries.parquet, gsis_id resolved to names
via weekly_rosters.parquet) on season+week+team+normalized name and writes
artifacts/nflcom_injuries/<snapshot_id>/agreement.json.

--current runs the IN-SEASON incremental mode required by
docs/nflcom_friday_refresh.md's frozen integration contract (section
"Refresh-path integration contract", item 2): resolve the live (season, REG
week) from the schedules snapshot and fetch ONLY that week's page into a FRESH
timestamped snapshot directory, instead of the 54-page historical backfill.

Every --current run writes its own snapshot directory on purpose. The NFL.com
league page is a LIVING document that is revised Wednesday through Friday, and
a revision stream cannot be recovered retroactively from a final-state page;
one immutable snapshot per capture is the only way to keep those intermediate
states. It also matters for the consumer: nfl_ats.prospective's
latest_nflcom_injuries_snapshot() reads the lexicographically LAST snapshot
directory only, so a fresh UTC-stamped directory is what makes the newest
capture the one production sees.
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import re
import sys
import time
import unicodedata
import urllib.robotparser
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.provenance import atomic_json  # noqa: E402
from nfl_ats.source_policy import require_acquisition  # noqa: E402

BASE_URL = "https://www.nfl.com/injuries/league/{season}/reg{week}"
ROBOTS_URL = "https://www.nfl.com/robots.txt"
USER_AGENT = "nfl-ats-research-snapshot/1.0 (private research; contact: local repo owner)"
DEFAULT_DELAY_SECONDS = 2.5
REG_WEEKS = range(1, 19)
DEFAULT_SEASONS = (2022, 2023, 2024)

SECTION_SPLIT = re.compile(r'<section class="nfl-o-injury-report__unit">')
TEAM_ABBR = re.compile(r'nfl-c-matchup-strip__team-abbreviation">\s*([A-Z]{2,4})\s*</span>')
SUBTITLE = re.compile(r'd3-o-section-sub-title"><span>([^<]+)</span>')
TABLE = re.compile(
    r'<table class="d3-o-table d3-o-table--detailed d3-o-reports--detailed">(.*?)</table>',
    re.DOTALL,
)
ROW = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL)
TAG = re.compile(r"<[^>]+>")

NICKNAME_TO_CODE = {
    "cardinals": "ARI",
    "falcons": "ATL",
    "ravens": "BAL",
    "bills": "BUF",
    "panthers": "CAR",
    "bears": "CHI",
    "bengals": "CIN",
    "browns": "CLE",
    "cowboys": "DAL",
    "broncos": "DEN",
    "lions": "DET",
    "packers": "GB",
    "texans": "HOU",
    "colts": "IND",
    "jaguars": "JAX",
    "chiefs": "KC",
    "raiders": "LV",
    "chargers": "LAC",
    "rams": "LA",
    "dolphins": "MIA",
    "vikings": "MIN",
    "patriots": "NE",
    "saints": "NO",
    "giants": "NYG",
    "jets": "NYJ",
    "eagles": "PHI",
    "steelers": "PIT",
    "49ers": "SF",
    "seahawks": "SEA",
    "buccaneers": "TB",
    "titans": "TEN",
    "commanders": "WAS",
    "football team": "WAS",
}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def strip_html(raw: str) -> str:
    return html_module.unescape(TAG.sub("", raw)).replace("\xa0", " ").strip()


def normalize_name(name: str) -> str:
    lowered = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    lowered = lowered.casefold().replace("'", "").replace(".", " ")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    tokens = [tok for tok in lowered.split() if tok not in SUFFIXES]
    return " ".join(tokens)


def initial_last_key(name: str) -> tuple[str, str]:
    tokens = normalize_name(name).split()
    if not tokens:
        return ("", "")
    first_initial = tokens[0][0]
    last = tokens[-1] if len(tokens) > 1 else ""
    return (first_initial, last)


def robots_allows(session: requests.Session) -> bool:
    response = session.get(ROBOTS_URL, timeout=60)
    response.raise_for_status()
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(response.text.splitlines())
    return parser.can_fetch("*", BASE_URL.format(season=2022, week=1))


def fetch_page(session: requests.Session, url: str) -> tuple[str | None, int | None, str | None]:
    error: str | None = None
    status: int | None = None
    for attempt in range(2):
        try:
            response = session.get(url, timeout=90)
            status = response.status_code
            if status == 200 and response.text:
                return response.text, 200, None
            error = f"http_{status}"
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt == 0:
            time.sleep(DEFAULT_DELAY_SECONDS)
    return None, status, error


def parse_page(payload: dict[str, Any]) -> list[dict[str, Any]]:
    text = payload["html"]
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    sections = SECTION_SPLIT.split(text)[1:]
    abbrs = TEAM_ABBR.findall(text)
    for section in sections:
        section_abbrs = TEAM_ABBR.findall(section)
        subtitles = SUBTITLE.findall(section)
        tables = TABLE.findall(section)
        codes_by_order = [
            NICKNAME_TO_CODE.get(strip_html(name).casefold(), "") for name in subtitles
        ]
        if len(tables) != len(subtitles):
            warnings.append(f"table/subtitle count mismatch in a {payload['url']} section")
        fallback = section_abbrs or abbrs
        for idx, table_html in enumerate(tables):
            code = ""
            if idx < len(codes_by_order):
                code = codes_by_order[idx]
            elif len(fallback) == 2:
                code = fallback[min(idx, 1)]
            if not code:
                warnings.append(f"unresolved team in a {payload['url']} section")
                continue
            header: list[str] = []
            body_rows: list[list[str]] = []
            for row_html in ROW.findall(table_html):
                cells = [strip_html(cell) for cell in CELL.findall(row_html)]
                if not cells:
                    continue
                if not header and "player" in cells[0].casefold():
                    header = [cell.casefold() for cell in cells]
                    continue
                body_rows.append(cells)
            if not header:
                warnings.append(f"missing header row in {code} table on {payload['url']}")
                continue

            def column(label: str, heads: list[str]) -> int:
                for pos, head in enumerate(heads):
                    if label in head:
                        return pos
                return -1

            col_player = column("player", header)
            col_position = column("position", header)
            col_injury = column("injur", header)
            col_practice = column("practice", header)
            col_game = column("game status", header)

            def value(pos: int, row_cells: list[str]) -> str | None:
                if pos < 0 or pos >= len(row_cells) or not row_cells[pos]:
                    return None
                return row_cells[pos]

            for cells in body_rows:
                player = value(col_player, cells)
                if player is None:
                    continue
                rows.append(
                    {
                        "season": payload["season"],
                        "week": payload["week"],
                        "team": code,
                        "player": player,
                        "position": value(col_position, cells),
                        "injury": value(col_injury, cells),
                        "practice_status": value(col_practice, cells),
                        "game_status": value(col_game, cells),
                        "source_url": payload["url"],
                        "fetched_at_utc": payload["fetched_at_utc"],
                    }
                )
    payload["warnings"].extend(warnings[:20])
    return rows


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_current_reg_week(repo: Path, now: pd.Timestamp | None = None) -> tuple[int, int]:
    """The live (season, REG week) to capture, from the schedules snapshot.

    The capture belongs to the week whose games are still AHEAD: we take the
    earliest REG week that still has an unplayed kickoff. Running on Wednesday
    of week 5 therefore captures week 5, not the just-completed week 4.
    """

    hits = sorted((repo / "data" / "raw").glob("*/schedules.parquet"))
    if not hits:
        raise SystemExit("no data/raw/*/schedules.parquet snapshot to resolve the current week")
    sched = pd.read_parquet(hits[-1])
    sched = sched.loc[sched["game_type"].astype(str) == "REG"].copy()
    local = pd.to_datetime(
        sched["gameday"].astype(str).str.slice(0, 10) + " " + sched["gametime"].astype(str),
        errors="coerce",
    )
    sched["kickoff_utc"] = local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")
    sched = sched.loc[sched["kickoff_utc"].notna()]
    moment = now if now is not None else pd.Timestamp.now(tz="UTC")
    ahead = sched.loc[sched["kickoff_utc"] > moment]
    if ahead.empty:
        raise SystemExit(f"no REG kickoff after {moment.isoformat()} in {hits[-1]}")
    first = ahead.sort_values("kickoff_utc").iloc[0]
    return int(first["season"]), int(first["week"])


def run_ingest(args: argparse.Namespace) -> Path:
    # MKT-09: robots permission does not override NFL.com's systematic-
    # retrieval terms. This tracked policy must be changed only after consent
    # or a fresh terms review; fail before creating a directory or a request.
    require_acquisition("nfl_com_injuries")
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    if args.snapshot:
        snapshot = out_root / args.snapshot
        snapshot.mkdir(parents=True, exist_ok=True)
    elif args.fresh_snapshot:
        # In-season capture: never resume into a prior directory, so each
        # Wed/Thu/Fri revision is preserved as its own immutable snapshot.
        snapshot = out_root / run_timestamp()
        snapshot.mkdir(parents=True, exist_ok=True)
    else:
        existing = sorted(p for p in out_root.iterdir() if p.is_dir())
        snapshot = existing[-1] if existing else out_root / run_timestamp()
        snapshot.mkdir(parents=True, exist_ok=True)
    pages_dir = snapshot / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    robots_ok = robots_allows(session)
    if not robots_ok:
        raise SystemExit("robots.txt disallows /injuries/ fetching; aborting")

    requested = [(season, week) for season in args.seasons for week in args.weeks]
    manifest_pages: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    previous_manifest_path = snapshot / "manifest.json"
    prior: dict[str, Any] = {}
    if previous_manifest_path.exists():
        prior = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    prior_pages = {page["url"]: page for page in prior.get("pages", [])}

    for number, (season, week) in enumerate(requested):
        url = BASE_URL.format(season=season, week=week)
        file_path = pages_dir / f"{season}_reg{week}.html"
        cached = prior_pages.get(url, {})
        if file_path.exists() and file_path.stat().st_size > 0 and cached.get("http_status") == 200:
            html_text = file_path.read_text(encoding="utf-8")
            entry = {
                "season": season,
                "week": week,
                "url": url,
                **cached,
                "sha256": sha256_bytes(file_path.read_bytes()),
                "resumed": True,
            }
            manifest_pages.append(entry)
        else:
            if number > 0:
                time.sleep(args.delay)
            html_text, http_status, error = fetch_page(session, url)
            if html_text is None:
                manifest_pages.append(
                    {
                        "season": season,
                        "week": week,
                        "url": url,
                        "http_status": http_status,
                        "error": error,
                        "fetched_at_utc": utc_now(),
                        "rows": 0,
                    }
                )
                print(f"FAIL {season} reg{week}: {error}")
                continue
            file_path.write_text(html_text, encoding="utf-8")
            entry = {
                "season": season,
                "week": week,
                "url": url,
                "http_status": 200,
                "sha256": sha256_bytes(file_path.read_bytes()),
                "bytes": file_path.stat().st_size,
                "fetched_at_utc": utc_now(),
                "rows": 0,
            }
            manifest_pages.append(entry)
        payload = {
            "season": season,
            "week": week,
            "url": url,
            "html": html_text,
            "fetched_at_utc": entry.get("fetched_at_utc") or utc_now(),
            "warnings": [],
        }
        rows = parse_page(payload)
        entry["rows"] = len(rows)
        entry["warnings"] = payload["warnings"]
        all_rows.extend(rows)
        print(f"ok {season} reg{week}: {len(rows)} rows")

    frame = pd.DataFrame(all_rows)
    if frame.empty:
        columns = [
            "season",
            "week",
            "team",
            "player",
            "position",
            "injury",
            "practice_status",
            "game_status",
            "source_url",
            "fetched_at_utc",
        ]
        frame = pd.DataFrame(columns=columns)
    table_path = snapshot / "injuries.parquet"
    frame.to_parquet(table_path, index=False)

    ok_pages = [p for p in manifest_pages if p.get("http_status") == 200]
    failed_pages = [p for p in manifest_pages if p.get("http_status") != 200]
    manifest = {
        "schema": "nflcom_injuries_snapshot/1",
        "snapshot_id": snapshot.name,
        "source": "nfl.com weekly league injury reports",
        "base_url_template": BASE_URL,
        "robots_check": {"url": ROBOTS_URL, "allowed": robots_ok},
        "user_agent": USER_AGENT,
        "delay_seconds": args.delay,
        "seasons_requested": list(args.seasons),
        "weeks_requested_per_season": list(args.weeks),
        "capture_mode": "in_season_current_week" if args.current else "historical_backfill",
        "pages": manifest_pages,
        "coverage": {
            "pages_ok": len(ok_pages),
            "pages_failed": len(failed_pages),
            "failed_urls": [p["url"] for p in failed_pages],
            "rows_total": len(frame),
            "rows_per_season": {
                str(season): int((frame["season"] == season).sum())
                for season in sorted(frame["season"].unique())
            },
            "teams_seen": sorted(str(t) for t in frame["team"].unique()),
        },
        "generated_at_utc": utc_now(),
    }
    atomic_json(manifest, snapshot / "manifest.json")
    print(f"snapshot: {snapshot} ({len(ok_pages)} ok, {len(failed_pages)} failed)")
    return snapshot


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def latest_under(root: Path, pattern: str) -> Path:
    candidates = sorted(root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no match for {pattern} under {root}")
    return candidates[-1]


def load_nflverse(players_root: Path) -> pd.DataFrame:
    players_snapshot = latest_under(players_root, "*/injuries.parquet")
    rosters_snapshot = latest_under(players_root, "*/weekly_rosters.parquet")
    injuries = pd.read_parquet(players_snapshot)
    rosters = pd.read_parquet(rosters_snapshot)
    name_map = (
        rosters.loc[rosters["gsis_id"].notna() & rosters["full_name"].notna()]
        .groupby("gsis_id")["full_name"]
        .agg(lambda s: s.mode().iat[0])
    )
    injuries = injuries.loc[injuries["game_type"] == "REG"].copy()
    injuries["full_name"] = injuries["gsis_id"].map(name_map)
    return injuries


def agreement(snapshot: Path, players_root: Path, artifacts_root: Path) -> dict[str, Any]:
    parsed = pd.read_parquet(snapshot / "injuries.parquet").copy()
    parsed["norm_name"] = parsed["player"].map(normalize_name)
    parsed["init_last"] = parsed["player"].map(initial_last_key)
    nflverse = load_nflverse(players_root)
    nflverse = nflverse.loc[nflverse["season"].isin(parsed["season"].unique())].copy()
    nflverse["norm_name"] = nflverse["full_name"].map(normalize_name)
    nflverse["init_last"] = nflverse["full_name"].map(initial_last_key)
    nflverse["has_designation"] = nflverse["report_status"].notna()

    exact = nflverse.merge(
        parsed[["season", "week", "team", "norm_name"]],
        on=["season", "week", "team", "norm_name"],
        how="inner",
    )
    matched_keys = set(
        zip(exact["season"], exact["week"], exact["team"], exact["norm_name"], strict=True)
    )
    unmatched_nflcom = parsed.loc[
        ~parsed.apply(
            lambda r: (r["season"], r["week"], r["team"], r["norm_name"]) in matched_keys, axis=1
        )
    ].copy()
    init_counts_nflverse = Counter(
        zip(
            nflverse["season"],
            nflverse["week"],
            nflverse["team"],
            nflverse["init_last"],
            strict=False,
        )
    )
    init_counts_nflcom = Counter(
        zip(
            unmatched_nflcom["season"],
            unmatched_nflcom["week"],
            unmatched_nflcom["team"],
            unmatched_nflcom["init_last"],
            strict=False,
        )
    )
    fuzzy_pairs = {
        key
        for key in set(init_counts_nflcom) & set(init_counts_nflverse)
        if key[3] != ("", "") and init_counts_nflcom[key] == 1 and init_counts_nflverse[key] == 1
    }
    n_fuzzy_matched = len(fuzzy_pairs)

    joined = nflverse.merge(
        parsed.rename(columns={"game_status": "nflcom_game_status"})[
            ["season", "week", "team", "norm_name", "nflcom_game_status"]
        ],
        on=["season", "week", "team", "norm_name"],
        how="outer",
        indicator=True,
    )
    comparable = joined.loc[
        (joined["_merge"] == "both")
        & (joined["report_status"].notna() | joined["nflcom_game_status"].notna())
    ]
    confusion: Counter[str] = Counter()
    agree = 0
    for _, row in comparable.iterrows():
        left = str(row["report_status"]) if pd.notna(row["report_status"]) else "-"
        right = str(row["nflcom_game_status"]) if pd.notna(row["nflcom_game_status"]) else "-"
        confusion[f"{left}|{right}"] += 1
        agree += int(left == right)

    coverage = {
        "nflcom_rows_total": len(parsed),
        "nflcom_rows_per_season": {
            str(s): int((parsed["season"] == s).sum()) for s in sorted(parsed["season"].unique())
        },
        "nflverse_reg_rows_in_scope": len(nflverse),
        "nflverse_rows_with_report_status": int(nflverse["has_designation"].sum()),
        "matched_exact_name": len(exact),
        "matched_fuzzy_initial_last": int(n_fuzzy_matched),
        "match_rate_vs_nflverse": float(
            round((len(exact) + n_fuzzy_matched) / max(len(nflverse), 1), 4)
        ),
        "nflcom_coverage_of_own_rows": float(
            round(len(joined.loc[joined["_merge"] == "both"]) / max(len(parsed), 1), 4)
        ),
    }
    result = {
        "schema": "nflcom_injuries_agreement/1",
        "snapshot_id": snapshot.name,
        "normalization": (
            "casefold, ASCII-fold accents, drop punctuation and suffix tokens "
            "(jr/sr/ii/iii/iv/v); join season+week+team+normalized full name, then "
            "first-initial+last-name when that key is unique on both sides within "
            "the same season+week+team"
        ),
        "coverage": coverage,
        "status_comparison": {
            "comparable_matched_rows": len(comparable),
            "exact_agreement": agree,
            "agreement_rate": float(round(agree / max(len(comparable), 1), 4)),
            "confusion_top": dict(confusion.most_common(25)),
        },
        "generated_at_utc": utc_now(),
    }
    out_dir = artifacts_root / snapshot.name
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(result, out_dir / "agreement.json")
    print(json.dumps(result["coverage"], indent=2))
    print(json.dumps(result["status_comparison"]["confusion_top"], indent=2))
    return result


def resolve_snapshot(args: argparse.Namespace) -> Path:
    existing = sorted(p for p in args.out.iterdir() if p.is_dir())
    if not existing:
        raise SystemExit("no existing snapshot under --out")
    return existing[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "raw" / "nflcom_injuries")
    parser.add_argument("--snapshot", type=str, default=None)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--weeks", type=int, nargs="+", default=list(REG_WEEKS))
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--agreement", action="store_true")
    parser.add_argument(
        "--fresh-snapshot",
        action="store_true",
        help="always start a new timestamped snapshot dir instead of resuming the newest",
    )
    parser.add_argument(
        "--current",
        action="store_true",
        help=(
            "in-season mode: resolve the live (season, REG week) from schedules and fetch "
            "only that page into a fresh timestamped snapshot (implies --fresh-snapshot)"
        ),
    )
    parser.add_argument("--players-root", type=Path, default=REPO / "data" / "players" / "raw")
    parser.add_argument(
        "--artifacts-root", type=Path, default=REPO / "artifacts" / "nflcom_injuries"
    )
    args = parser.parse_args()

    if args.agreement:
        snapshot = Path(args.snapshot) if args.snapshot else resolve_snapshot(args)
        agreement(Path(args.out) / snapshot.name, args.players_root, args.artifacts_root)
        return
    if args.delay < 2.0:
        raise SystemExit("delay must be >= 2 seconds (polite rate limiting)")
    if args.current:
        season, week = resolve_current_reg_week(REPO)
        args.seasons = [season]
        args.weeks = [week]
        args.fresh_snapshot = True
        print(f"current REG week resolved to {season} week {week}")
    run_ingest(args)


if __name__ == "__main__":
    main()
