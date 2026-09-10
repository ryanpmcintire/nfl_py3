"""Bulk ingestion of historical, timestamped NFL injury/personnel news headlines.

Follow-up: "injury-news aggregator ingestion -- bulk path unverified"
(docs/injury_news_sourcing.md has the full verdict and evidence).

What this is NOT: it does not touch the official injury/practice report already
ingested via nflverse (data/players/raw/*/injuries.parquet, consumed by
players.py's date_modified as-of filtering). That source is measured
(2026-08-19, 79,818 rows) to be structurally a Wed-Fri artifact -- 81.4% of
rows are dated Friday, only 0.33% ever fall on Monday or Tuesday -- so it
cannot answer "what injury information was public before the pool's Tuesday
noon lock" (docs/pool_edge_plan.md line 80: "Picks lock Tuesday at 12").

Source: NBC Sports' public sitemap index (https://www.nbcsports.com/sitemap.xml),
a chronological archive of every published article, chunked into ~one file per
calendar month (confirmed 2026-08-19: 275 monthly chunks, September 2003
through June 2026, plus a rolling "latest" chunk). Each <url> entry carries an
ISO-8601 <lastmod> timestamp. Spot-checked against individual article pages'
JSON-LD `datePublished` field (see --verify-sample): for in-season months the
per-URL <lastmod> values are minute-granular and spread naturally across the
month (e.g. Sept 2022: 14 ProFootballTalk injury-tagged articles between
2022-09-01T02:56 and 2022-09-01T14:56 alone) rather than clustered on one
migration date -- i.e. <lastmod> is a real per-article publish/update
timestamp for this content, not a platform-migration artifact. (Pre-2009
chunks ARE a migration artifact -- see docs/injury_news_sourcing.md sec 2.)

Filtered to ProFootballTalk NFL articles (path contains "/profootballtalk/")
whose URL slug matches an injury/personnel-move keyword. All ProFootballTalk
NFL urls (not just keyword matches) are retained per month with an
`injury_relevant` flag, so a future session can redraw the keyword line
without re-fetching.

Respects robots.txt `Crawl-delay: 10` for www.nbcsports.com (fetched and read
this session, 2026-08-19: "User-agent: *" / "Crawl-delay: 10"). This bounds
the practical scope of this script to sitemap-level bulk fetches (275 monthly
sitemap requests ~= 46 minutes at 10s/request) -- NOT per-article body-text
fetching at bulk scale (thousands of articles x 10s would take days). Per-
article body fetches are supported only for a small --verify-sample.

Private-research use only, matching this project's existing CFBD/cfbfastR
precedent (docs/data_feasibility.md, License item 6: "private caching/
retention" permitted, "raw tables must never be republished"). NBC Sports'
own terms of use were not independently reviewed this session (label:
inferred policy stance, not a verified legal fact) -- treat this archive the
same way: cache privately, never republish, never redistribute raw rows.

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_injury_news.py \\
        --out data/raw/injury_news --start 200909 --end 202606

    # Fetch a handful of real article pages to confirm datePublished + body
    # text are extractable (task's "verify by actually fetching a sample"):
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_injury_news.py \\
        --out data/raw/injury_news --verify-sample 202010 --sample-n 5

Writes under --out/<snapshot>/ (default data/raw/injury_news/<UTC timestamp>,
e.g. data/raw/injury_news/20260819T191639Z/ -- gitignored by the repository's
existing `data/raw/**` rule, no .gitignore change needed). The timestamped
snapshot subdirectory is deliberate and matches this repo's existing
convention for external raw pulls (data/raw/<timestamp>/,
data/players/raw/<timestamp>/): `nfl_ats.snapshots.latest_snapshot()` treats
ANY directory directly under data/raw/ that contains a manifest.json as a
candidate schedules snapshot, so a manifest.json must never sit directly at
data/raw/injury_news/manifest.json -- it must be nested one level down, under
a snapshot subdirectory, exactly like every other named raw source in this
project. Without --snapshot, a run resumes the most recent existing snapshot
subdirectory under --out if one exists (so re-running the same command
continues, rather than restarting, a long crawl-delay-bound pull), or creates
a fresh UTC-timestamped one if none exists.

    <snapshot>/monthly/<YYYYMM>.parquet   one row per ProFootballTalk NFL url
                                            in that month's sitemap chunk.
    <snapshot>/index.parquet               concatenation of every monthly
                                            file present in this snapshot.
    <snapshot>/manifest.json               run metadata + coverage summary.
    <snapshot>/sample_articles/<slug>.json full-text spot-check fetches
                                            (--verify-sample).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

SITEMAP_INDEX_URL = "https://www.nbcsports.com/sitemap.xml"
SITEMAP_CHUNK_RE = re.compile(r"sitemap-(\d{6})\.xml$")
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
CRAWL_DELAY_SECONDS = 10.0
SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
SNAPSHOT_DIR_RE = re.compile(r"^\d{8}T\d{6}Z$")
HUB_URL = "https://www.nbcsports.com/nfl/profootballtalk"
HUB_ARTICLE_RE = re.compile(
    r'href="(https://www\.nbcsports\.com/nfl/profootballtalk/[^"]+/news/[^"?#]+)"'
)
DATE_PUBLISHED_RE = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')
HEADLINE_RE = re.compile(r'"headline"\s*:\s*"([^"]+)"')


def resolve_snapshot_dir(out_dir: Path, snapshot: str | None) -> Path:
    """Return the timestamped snapshot subdirectory to write/resume into.

    A manifest.json must never sit directly at `out_dir` (see module
    docstring: `nfl_ats.snapshots.latest_snapshot()` treats any directory
    directly under data/raw/ with a manifest.json as a candidate schedules
    snapshot). If `snapshot` is given, use/create `out_dir/snapshot`.
    Otherwise resume the most recent existing `<UTC timestamp>` subdirectory
    under `out_dir`, or create a fresh one.
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


def _latest_existing_snapshot(out_dir: Path, *, exclude: Path) -> Path | None:
    """Most recent existing snapshot dir under out_dir, other than exclude."""

    candidates = sorted(
        path
        for path in out_dir.glob("*")
        if path.is_dir() and SNAPSHOT_DIR_RE.match(path.name) and path != exclude
    )
    return candidates[-1] if candidates else None


def create_fresh_snapshot_dir(
    out_dir: Path, *, refetch_months: set[str], now: datetime | None = None
) -> tuple[Path, list[str]]:
    """New timestamped snapshot dir; prior months copied forward except refetch_months."""

    now = now or datetime.now(UTC)
    new_id = now.strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = out_dir / new_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    previous = _latest_existing_snapshot(out_dir, exclude=snapshot_dir)
    copied_forward: list[str] = []
    if previous is not None:
        src_monthly = previous / "monthly"
        dst_monthly = snapshot_dir / "monthly"
        dst_monthly.mkdir(parents=True, exist_ok=True)
        for path in sorted(src_monthly.glob("*.parquet")):
            month = path.stem
            if month in refetch_months:
                continue
            shutil.copy2(path, dst_monthly / path.name)
            copied_forward.append(month)
        src_current = previous / "current.parquet"
        if src_current.is_file():
            shutil.copy2(src_current, snapshot_dir / "current.parquet")
    return snapshot_dir, copied_forward


INJURY_KEYWORDS = [
    "injured-reserve",
    "-on-ir",
    "-ir-",
    "questionable",
    "doubtful",
    "ruled-out",
    "will-miss",
    "out-for-the-season",
    "out-for-season",
    "exits-",
    "exited-",
    "concussion",
    "hamstring",
    "quadricep",
    "groin",
    "ankle",
    "knee-injury",
    "torn-acl",
    "torn-mcl",
    "shoulder-injury",
    "achilles",
    "-surgery",
    "surgery-",
    "torn-",
    "-tear",
    "fracture",
    "fractured",
    "sprain",
    "sprained",
    "activated-from",
    "return-to-practice",
    "returns-to-practice",
    "injury-report",
    "placed-on-ir",
    "pup-list",
    "non-football-injury",
    "nfi-list",
    "day-to-day",
    "limited-in-practice",
    "did-not-practice",
    "practice-window",
    "week-to-week",
    "designated-to-return",
    "injury-designation",
]


@dataclass
class RateLimiter:
    """Enforces robots.txt Crawl-delay between requests to the same host."""

    delay_seconds: float
    _last_request: float | None = field(default=None, init=False)

    def wait(self) -> None:
        if self._last_request is not None:
            elapsed = time.monotonic() - self._last_request
            remaining = self.delay_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request = time.monotonic()


def _fetch(url: str, limiter: RateLimiter, *, timeout: int = 30, retries: int = 3) -> bytes:
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
            time.sleep(2.0 * (attempt + 1))
    assert last_error is not None
    raise last_error


def fetch_sitemap_index(limiter: RateLimiter) -> list[tuple[str, str]]:
    """Return [(yyyymm, sitemap_url), ...] sorted chronologically."""

    raw = _fetch(SITEMAP_INDEX_URL, limiter)
    root = ElementTree.fromstring(raw)
    chunks: list[tuple[str, str]] = []
    for sitemap_el in root.findall(f"{SITEMAP_NS}sitemap"):
        loc_el = sitemap_el.find(f"{SITEMAP_NS}loc")
        if loc_el is None or not loc_el.text:
            continue
        loc = loc_el.text.strip()
        match = SITEMAP_CHUNK_RE.search(loc)
        if not match:
            continue
        chunks.append((match.group(1), loc))
    chunks.sort(key=lambda pair: pair[0])
    return chunks


def _slug_matches_injury_keyword(slug: str) -> list[str]:
    lowered = slug.lower()
    return [kw for kw in INJURY_KEYWORDS if kw in lowered]


def parse_monthly_sitemap(raw: bytes, month: str) -> pd.DataFrame:
    root = ElementTree.fromstring(raw)
    rows: list[dict[str, object]] = []
    for url_el in root.findall(f"{SITEMAP_NS}url"):
        loc_el = url_el.find(f"{SITEMAP_NS}loc")
        lastmod_el = url_el.find(f"{SITEMAP_NS}lastmod")
        if loc_el is None or not loc_el.text:
            continue
        loc = loc_el.text.strip()
        if "/profootballtalk/" not in loc:
            continue
        lastmod = lastmod_el.text.strip() if lastmod_el is not None and lastmod_el.text else None
        slug = loc.rstrip("/").rsplit("/", 1)[-1]
        matched = _slug_matches_injury_keyword(slug)
        rows.append(
            {
                "month": month,
                "url": loc,
                "lastmod": lastmod,
                "slug": slug,
                "headline_guess": slug.replace("-", " "),
                "matched_keywords": "|".join(matched),
                "injury_relevant": bool(matched),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["lastmod"] = pd.to_datetime(frame["lastmod"], errors="coerce", utc=True)
    return frame


def fetch_hub_article_urls(limiter: RateLimiter, *, hub_url: str = HUB_URL) -> list[str]:
    """Article urls linked from the live PFT hub page (the post-2026-06 URL scheme)."""

    raw = _fetch(hub_url, limiter).decode("utf-8", errors="ignore")
    return sorted(set(HUB_ARTICLE_RE.findall(raw)))


def fetch_current_articles(
    urls: list[str], limiter: RateLimiter, *, existing_urls: frozenset[str] = frozenset()
) -> pd.DataFrame:
    """Per-article JSON-LD datePublished for injury-keyword-matched hub urls not already indexed."""

    rows: list[dict[str, object]] = []
    for url in urls:
        if url in existing_urls:
            continue
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        matched = _slug_matches_injury_keyword(slug)
        if not matched:
            continue
        try:
            raw = _fetch(url, limiter).decode("utf-8", errors="ignore")
        except Exception as error:
            print(f"  FAILED hub article {url}: {error}", file=sys.stderr)
            continue
        date_matches = DATE_PUBLISHED_RE.findall(raw)
        if not date_matches:
            continue
        published = date_matches[0]
        rows.append(
            {
                "month": published[:7].replace("-", ""),
                "url": url,
                "lastmod": published,
                "slug": slug,
                "headline_guess": slug.replace("-", " "),
                "matched_keywords": "|".join(matched),
                "injury_relevant": True,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["lastmod"] = pd.to_datetime(frame["lastmod"], errors="coerce", utc=True)
    return frame


def ingest(
    out_dir: Path,
    start: str,
    end: str,
    *,
    force: bool = False,
    limiter: RateLimiter | None = None,
) -> dict[str, object]:
    limiter = limiter or RateLimiter(CRAWL_DELAY_SECONDS)
    monthly_dir = out_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching sitemap index: {SITEMAP_INDEX_URL}")
    chunks = fetch_sitemap_index(limiter)
    chunks = [(month, url) for month, url in chunks if start <= month <= end]
    print(f"Sitemap index: {len(chunks)} monthly chunks in [{start}, {end}]")

    processed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    totals = {"pft_nfl_urls": 0, "injury_relevant_urls": 0}

    for month, sitemap_url in chunks:
        out_path = monthly_dir / f"{month}.parquet"
        if out_path.exists() and not force:
            skipped.append(month)
            continue
        try:
            raw = _fetch(sitemap_url, limiter)
            frame = parse_monthly_sitemap(raw, month)
        except Exception as error:
            print(f"  FAILED month {month}: {error}", file=sys.stderr)
            failed.append(month)
            continue
        frame.to_parquet(out_path, index=False)
        totals["pft_nfl_urls"] += len(frame)
        totals["injury_relevant_urls"] += int(frame["injury_relevant"].sum()) if len(frame) else 0
        processed.append(month)
        print(
            f"  {month}: {len(frame)} PFT/NFL urls, "
            f"{int(frame['injury_relevant'].sum()) if len(frame) else 0} injury-relevant"
        )

    current_path = out_dir / "current.parquet"
    existing_current = pd.read_parquet(current_path) if current_path.exists() else pd.DataFrame()
    hub_new = 0
    hub_failed: str | None = None
    current_combined = existing_current
    try:
        hub_urls = fetch_hub_article_urls(limiter)
        existing_urls = (
            frozenset(existing_current["url"]) if not existing_current.empty else frozenset()
        )
        fresh = fetch_current_articles(hub_urls, limiter, existing_urls=existing_urls)
        hub_new = len(fresh)
        current_combined = (
            pd.concat([existing_current, fresh], ignore_index=True)
            if not existing_current.empty
            else fresh
        )
        if not current_combined.empty:
            current_combined = current_combined.drop_duplicates("url", keep="last")
            current_combined.to_parquet(current_path, index=False)
        print(f"  hub page: {len(hub_urls)} article links, {hub_new} new injury-relevant")
    except Exception as error:
        hub_failed = f"{type(error).__name__}: {error}"
        print(f"  FAILED hub page pass: {hub_failed}", file=sys.stderr)

    all_monthly = sorted(monthly_dir.glob("*.parquet"))
    frames = [pd.read_parquet(p) for p in all_monthly]
    if not current_combined.empty:
        frames.append(current_combined)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["lastmod", "url"]).reset_index(drop=True)
        combined.to_parquet(out_dir / "index.parquet", index=False)
    else:
        combined = pd.DataFrame()

    manifest = {
        "source": "https://www.nbcsports.com/sitemap.xml (ProFootballTalk NFL articles)",
        "hub_source": HUB_URL,
        "fetched_at": datetime.now(UTC).isoformat(),
        "requested_range": [start, end],
        "months_processed_this_run": processed,
        "months_skipped_already_present": skipped,
        "months_failed": failed,
        "hub_new_injury_relevant_this_run": hub_new,
        "hub_fetch_failed": hub_failed,
        "hub_cumulative_rows": len(current_combined),
        "crawl_delay_seconds_honored": CRAWL_DELAY_SECONDS,
        "user_agent": USER_AGENT,
        "cumulative_index_rows": len(combined),
        "cumulative_injury_relevant_rows": (
            int(combined["injury_relevant"].sum()) if len(combined) else 0
        ),
        "cumulative_months_on_disk": len(all_monthly),
        "injury_keywords": INJURY_KEYWORDS,
        "usage_note": (
            "Private research caching only, matching this project's CFBD/cfbfastR "
            "precedent (docs/data_feasibility.md License item 6). Never republish "
            "raw rows. NBC Sports terms of use were not independently reviewed "
            "this session -- this is a policy stance, not a verified legal fact."
        ),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest


def verify_sample(
    out_dir: Path, month: str, sample_n: int, limiter: RateLimiter | None = None
) -> None:
    """Fetch a handful of real article pages to confirm per-article timestamps
    and body text are extractable -- the task's "verify by actually fetching a
    sample (one historical week)" requirement."""

    limiter = limiter or RateLimiter(CRAWL_DELAY_SECONDS)
    monthly_path = out_dir / "monthly" / f"{month}.parquet"
    if not monthly_path.exists():
        raise SystemExit(
            f"{monthly_path} not found -- run ingestion for {month} first "
            f"(--start {month} --end {month})"
        )
    frame = pd.read_parquet(monthly_path)
    candidates = frame.loc[frame["injury_relevant"]].sort_values("lastmod").head(sample_n)
    if candidates.empty:
        print(f"No injury-relevant urls found in {month}.")
        return

    sample_dir = out_dir / "sample_articles"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for _, row in candidates.iterrows():
        url = str(row["url"])
        slug = str(row["slug"])
        print(f"Fetching sample article: {url}")
        try:
            raw = _fetch(url, limiter).decode("utf-8", errors="ignore")
        except Exception as error:
            print(f"  FAILED: {error}", file=sys.stderr)
            continue
        date_matches = DATE_PUBLISHED_RE.findall(raw)
        headline_matches = HEADLINE_RE.findall(raw)
        record = {
            "url": url,
            "slug": slug,
            "sitemap_lastmod": str(row["lastmod"]),
            "json_ld_date_published_matches": date_matches[:3],
            "json_ld_headline_matches": headline_matches[:1],
            "html_bytes": len(raw),
        }
        with (sample_dir / f"{slug}.json").open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        print(f"  sitemap lastmod:   {row['lastmod']}")
        print(f"  datePublished(s):  {date_matches[:3]}")
        print(f"  headline:          {headline_matches[:1]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw/injury_news"))
    parser.add_argument(
        "--snapshot",
        default=None,
        metavar="YYYYMMDDTHHMMSSZ",
        help=(
            "Timestamped snapshot subdirectory under --out to write/resume. "
            "Default: resume the most recent existing snapshot under --out, "
            "or create a fresh UTC-timestamped one if none exists."
        ),
    )
    parser.add_argument("--start", default="200909", help="YYYYMM, inclusive")
    parser.add_argument("--end", default="202702", help="YYYYMM, inclusive")
    parser.add_argument("--force", action="store_true", help="Refetch months already on disk")
    parser.add_argument(
        "--fresh-snapshot",
        action="store_true",
        help=(
            "Write into a NEW UTC-timestamped snapshot directory instead of resuming "
            "the most recent one. Months already cached in the most recent existing "
            "snapshot are copied forward with zero network requests; only the current "
            "UTC month's chunk is force-refetched (it is still gaining new articles). "
            "Mirrors ingest_transaction_news.py's --fresh-snapshot (ENG-32): "
            "nfl_ats.capture_freshness reads the snapshot DIRECTORY NAME as the "
            "capture instant, so a scheduled job resuming the same directory forever "
            "would never look fresh even after a successful run."
        ),
    )
    parser.add_argument(
        "--verify-sample",
        metavar="YYYYMM",
        default=None,
        help="Instead of bulk ingestion, fetch --sample-n real article pages from this month",
    )
    parser.add_argument("--sample-n", type=int, default=5)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.fresh_snapshot:
        current_month = datetime.now(UTC).strftime("%Y%m")
        snapshot_dir, copied_forward = create_fresh_snapshot_dir(
            args.out, refetch_months={current_month}
        )
        print(
            f"Fresh snapshot directory: {snapshot_dir} "
            f"(copied forward, no network: {copied_forward or 'none'})"
        )
    else:
        snapshot_dir = resolve_snapshot_dir(args.out, args.snapshot)
        print(f"Snapshot directory: {snapshot_dir}")
    limiter = RateLimiter(CRAWL_DELAY_SECONDS)

    if args.verify_sample:
        verify_sample(snapshot_dir, args.verify_sample, args.sample_n, limiter=limiter)
        return

    manifest = ingest(snapshot_dir, args.start, args.end, force=args.force, limiter=limiter)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
