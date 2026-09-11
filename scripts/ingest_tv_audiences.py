from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.source_policy import (  # noqa: E402
    require_acquisition,
    require_private_raw_destination,
)

if __package__:
    from scripts.ingest_sports_media_watch import TEAM_NAMES
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ingest_sports_media_watch import TEAM_NAMES

SOURCE_ID = "awful_announcing_tv_ratings"
BASE_URL = "https://awfulannouncing.com/"
TAG_URL = "https://awfulannouncing.com/tag/nfl-ratings"
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
REQUEST_DELAY_SECONDS = 2.0
EXTRACTION_RULE_VERSION = "prose_window_network_viewers_v1"

NETWORK_TOKENS = [
    "ManningCast",
    "ESPN2",
    "ESPN+",
    "ESPN",
    "ABC",
    "Amazon Prime Video",
    "Prime Video",
    "Amazon",
    "NFL Network",
    "Netflix",
    "NBC",
    "CBS",
    "Fox",
]
WINDOW_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Sunday Night Football", re.I), "SNF"),
    (re.compile(r"Monday Night Football", re.I), "MNF"),
    (re.compile(r"Thursday Night Football", re.I), "TNF"),
    (re.compile(r"ManningCast"), "MANNINGCAST_ALT"),
]
SUNDAY_WINDOW_RE = re.compile(r"(\d{1,2}:\d{2}\s*[ap]\.m\.\s*ET)\s*window", re.I)
VIEWERS_RE = re.compile(r"([\d]+(?:\.\d+)?)\s*million\s+(?:viewers|people watched)", re.I)
PUBLISHED_RE = re.compile(r'"article:published_time"\s+content="([^"]+)"')
MODIFIED_RE = re.compile(r'"article:modified_time"\s+content="([^"]+)"')
TITLE_RE = re.compile(r"<title>([^<]+)</title>")
WEEK_RE = re.compile(r"\bWeek\s+(\d{1,2})\b", re.I)
LINK_RE = re.compile(r'href="(https://awfulannouncing\.com/(?:nfl|ratings)/[^"]+)"')
EXCLUDED_LINK_TOKENS = ("schedule", "/schedules/", "announcing-", "preseason")
COMPARISON_CLAUSE_RE = re.compile(r"\b(?:up|down)\b[^.]{0,25}\bfrom\b|\bcompared\b", re.I)


class TvAudienceIngestError(ValueError):
    pass


def _fetch(url: str, *, timeout: int = 45, retries: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.9"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            print(f"fetch failed ({attempt + 1}/{retries}) {url}: {error}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _strip_html(payload_text: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", payload_text, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<blockquote[^>]*>.*?</blockquote>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"—\s*[^()\n]{2,40}\(@\w+\)\s*\w+ \d{1,2}, \d{4}", " ", text)
    text = re.sub(r"Credit:[^.]{0,80}Images", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_season(published_at: datetime) -> int:
    return published_at.year - 1 if published_at.month <= 3 else published_at.year


def _find_network(text: str) -> str | None:
    for token in NETWORK_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", text):
            return token
    return None


def _find_window(text: str) -> str:
    for pattern, label in WINDOW_RULES:
        if pattern.search(text):
            return label
    match = SUNDAY_WINDOW_RE.search(text)
    if match:
        return "SUNDAY_" + re.sub(r"\s+", "", match.group(1)).upper()
    return "UNSPECIFIED"


def _primary_clause(text: str) -> str:
    match = COMPARISON_CLAUSE_RE.search(text)
    return text[: match.start()] if match else text


def _find_teams(text: str) -> list[str]:
    matches: list[tuple[int, str]] = []
    seen_codes: set[str] = set()
    for nickname, code in TEAM_NAMES.items():
        if code in seen_codes:
            continue
        found = re.search(rf"\b{re.escape(nickname)}\b", text)
        if found:
            matches.append((found.start(), code))
            seen_codes.add(code)
    matches.sort(key=lambda item: item[0])
    return [code for _, code in matches]


SENTENCE_BOUNDARY_RE = re.compile(r"(?<![pa]\.m)\.\s+(?=[A-Z0-9])", re.I)


def _sentence_span(
    text: str, start: int, end: int, *, max_back: int = 260, max_fwd: int = 220
) -> tuple[int, int]:
    lo = max(0, start - max_back)
    hi = min(len(text), end + max_fwd)
    span_start = lo
    for boundary in SENTENCE_BOUNDARY_RE.finditer(text, lo, start):
        span_start = boundary.end()
    right = SENTENCE_BOUNDARY_RE.search(text, end, hi)
    span_end = right.end() if right else hi
    return span_start, span_end


def extract_rows(
    text: str,
    *,
    season: int,
    week: int | None,
    source_page_url: str,
    source_published_at: str,
    source_modified_at: str | None,
    source_observed_at: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for match in VIEWERS_RE.finditer(text):
        span_start, span_end = _sentence_span(text, match.start(), match.end())
        window_text = text[span_start:span_end]
        teams = _find_teams(_primary_clause(window_text))
        records.append(
            {
                "season": season,
                "week": week,
                "network": _find_network(window_text),
                "window": _find_window(window_text),
                "team_a": teams[0] if len(teams) > 0 else None,
                "team_b": teams[1] if len(teams) > 1 else None,
                "viewers": round(float(match.group(1)) * 1_000_000),
                "source_sentence": window_text.strip(),
                "source_page_url": source_page_url,
                "source_observed_at": source_observed_at,
                "source_published_at": source_published_at,
                "source_modified_at": source_modified_at,
                "timestamp_source_url": source_page_url,
                "timestamp_source_id": source_page_url.rstrip("/").rsplit("/", 1)[-1],
                "timestamp_match_method": "direct_article_published_time_meta",
                "point_in_time_usable": True,
            }
        )
    return pd.DataFrame.from_records(records)


def discover_latest_article(fetcher: Callable[[str], bytes], already_seen: set[str]) -> str | None:
    payload = fetcher(TAG_URL)
    text = payload.decode("utf-8", errors="replace")
    ordered: list[str] = []
    seen_local: set[str] = set()
    for link in LINK_RE.findall(text):
        if link in seen_local:
            continue
        seen_local.add(link)
        ordered.append(link)
    for link in ordered:
        if any(token in link for token in EXCLUDED_LINK_TOKENS):
            continue
        if link in already_seen:
            continue
        return link
    return None


def _is_preseason(title: str, url: str) -> bool:
    return "preseason" in title.lower() or "preseason" in url.lower()


def _latest_schedules_path(repo_root: Path = REPO) -> Path | None:
    candidates = sorted((repo_root / "data" / "raw").glob("*/schedules.parquet"))
    return candidates[-1] if candidates else None


def validate_against_schedule(
    rows: pd.DataFrame, schedules_path: Path | None
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not len(rows):
        return rows, rows, "no_rows_to_validate"
    if schedules_path is None or not schedules_path.exists():
        return rows.assign(schedule_validated=None), rows.iloc[0:0], "schedules_file_not_found"

    schedules = pd.read_parquet(
        schedules_path, columns=["season", "week", "game_type", "home_team", "away_team"]
    )
    season_pairs: dict[int, set[frozenset[str]]] = {}
    week_pairs: dict[tuple[int, int], set[frozenset[str]]] = {}
    for record in schedules.itertuples(index=False):
        pair = frozenset({str(record.home_team), str(record.away_team)})
        season_pairs.setdefault(int(record.season), set()).add(pair)
        if record.game_type == "REG":
            week_pairs.setdefault((int(record.season), int(record.week)), set()).add(pair)

    keep_mask: list[bool] = []
    validated: list[Any] = []
    for row in rows.itertuples(index=False):
        if pd.isna(row.team_a) or pd.isna(row.team_b):
            keep_mask.append(True)
            validated.append(None)
            continue
        pair = frozenset({row.team_a, row.team_b})
        if row.week is not None:
            valid_pairs = week_pairs.get((row.season, row.week), set())
        else:
            valid_pairs = season_pairs.get(row.season, set())
        matched = pair in valid_pairs
        keep_mask.append(matched)
        validated.append(matched)

    annotated = rows.assign(schedule_validated=validated)
    kept = annotated.loc[keep_mask].reset_index(drop=True)
    dropped = annotated.loc[[not flag for flag in keep_mask]].reset_index(drop=True)
    return kept, dropped, str(schedules_path)


def ingest_article(
    url: str,
    output: Path,
    *,
    season_override: int | None,
    week_override: int | None,
    fetcher: Callable[[str], bytes] = _fetch,
    sleeper: Callable[[float], None] = time.sleep,
    observed_at: str | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    pages_dir = output / "pages"
    pages_dir.mkdir(exist_ok=True)
    observed_at = observed_at or datetime.now(UTC).isoformat()

    sleeper(REQUEST_DELAY_SECONDS)
    payload = fetcher(url)
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    page_filename = slug if slug.endswith(".html") else f"{slug}.html"
    (pages_dir / page_filename).write_bytes(payload)
    page_sha256 = hashlib.sha256(payload).hexdigest()

    raw_text = payload.decode("utf-8", errors="replace")
    published_match = PUBLISHED_RE.search(raw_text)
    if published_match is None:
        raise TvAudienceIngestError(f"no article:published_time found at {url}")
    published_at_str = published_match.group(1)
    modified_match = MODIFIED_RE.search(raw_text)
    modified_at_str = modified_match.group(1) if modified_match else None
    title_match = TITLE_RE.search(raw_text)
    title = title_match.group(1) if title_match else ""

    if _is_preseason(title, url):
        manifest = {
            "source": "Awful Announcing weekly NFL TV-ratings recap posts",
            "observed_at": observed_at,
            "article_url": url,
            "article_title": title,
            "article_published_time": published_at_str,
            "status": "skipped_preseason_post",
            "structured_rows": 0,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    published_dt = datetime.fromisoformat(published_at_str)
    season = season_override or infer_season(published_dt)
    week_match = WEEK_RE.search(title) or WEEK_RE.search(raw_text)
    week = week_override or (int(week_match.group(1)) if week_match else None)

    plain_text = _strip_html(raw_text)
    extracted = extract_rows(
        plain_text,
        season=season,
        week=week,
        source_page_url=url,
        source_published_at=published_at_str,
        source_modified_at=modified_at_str,
        source_observed_at=observed_at,
    )
    schedules_path = _latest_schedules_path()
    rows, dropped_rows, schedule_source = validate_against_schedule(extracted, schedules_path)

    if len(rows):
        rows.to_parquet(output / "rows.parquet", index=False)
    if len(dropped_rows):
        dropped_rows.to_parquet(output / "dropped_rows.parquet", index=False)

    provenance = {
        "source": SOURCE_ID,
        "article_url": url,
        "article_title": title,
        "article_published_time": published_at_str,
        "article_modified_time": modified_at_str,
        "page_sha256": page_sha256,
        "extraction_rule_version": EXTRACTION_RULE_VERSION,
        "timestamp_match_method": "direct_article_published_time_meta",
        "schedule_source": schedule_source,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    manifest = {
        "source": "Awful Announcing weekly NFL TV-ratings recap posts",
        "source_base_url": BASE_URL,
        "observed_at": observed_at,
        "article_url": url,
        "article_title": title,
        "article_published_time": published_at_str,
        "article_modified_time": modified_at_str,
        "season": season,
        "week": week,
        "page_sha256": page_sha256,
        "structured_rows": len(rows),
        "rows_dropped_schedule_mismatch": len(dropped_rows),
        "schedule_source": schedule_source,
        "networks_found": sorted(set(rows["network"].dropna())) if len(rows) else [],
        "windows_found": sorted(set(rows["window"].dropna())) if len(rows) else [],
        "extraction_rule_version": EXTRACTION_RULE_VERSION,
        "point_in_time_contract": {
            "same_game_viewership_allowed_as_feature_for_that_game": False,
            "feature_scope": "prior-game or season-to-date lag only, same as sports_media_watch",
            "archive_rows_usable": True,
            "reason": (
                "each row is parsed directly out of a dated recap article; source_published_at "
                "is the article's own article:published_time meta, not a living-page revision, "
                "so no separate publication backfill is required"
            ),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Ingest an Awful Announcing weekly NFL TV audience recap post"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output = Path("data") / "raw" / "tv_audiences" / stamp

    require_acquisition(SOURCE_ID)
    require_private_raw_destination(SOURCE_ID, args.output)

    if not args.url and not args.discover:
        raise SystemExit("pass --url <article-url> or --discover")

    index_path = args.output.parent / "ingested_urls.json"
    already_seen: set[str] = set()
    if index_path.exists():
        already_seen = set(json.loads(index_path.read_text()))

    if args.discover:
        url = discover_latest_article(_fetch, already_seen)
        if url is None:
            manifest = {
                "source": "Awful Announcing weekly NFL TV-ratings recap posts",
                "observed_at": datetime.now(UTC).isoformat(),
                "status": "no_new_article_since_last_discover",
                "structured_rows": 0,
            }
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps(manifest, indent=2))
            return
    else:
        url = args.url
        assert url is not None

    manifest = ingest_article(
        url,
        args.output,
        season_override=args.season,
        week_override=args.week,
    )

    already_seen.add(url)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(sorted(already_seen), indent=2) + "\n")

    print(json.dumps(manifest, indent=2))
    rows_path = args.output / "rows.parquet"
    if rows_path.exists():
        print(pd.read_parquet(rows_path).to_string(index=False))


if __name__ == "__main__":
    main()
