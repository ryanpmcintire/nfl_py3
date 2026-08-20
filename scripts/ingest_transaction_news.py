"""Bulk ingestion of the Pro Football Rumors (PFR-rumors, not pro-football-
reference) transaction-wire archive: signings, cuts, trades, IR moves,
practice-squad churn.

Follow-up to `docs/data_source_scout_v3.md` rank-1 candidate ("Pro Football
Rumors transaction-wire sitemap"). Clones the structure of
`scripts/ingest_injury_news.py` (see `docs/injury_news_sourcing.md` sections
1-4 for the proven pattern this mirrors: sitemap-index -> per-chunk fetch ->
URL/slug extraction -> a small per-article JSON-LD verification sample), with
two adaptations specific to this source:

1. PFR chunks its sitemap by YEAR, not by month
   (`sitemap-posttype-post.YYYY.xml`), so this script fetches yearly chunks.
2. PFR's sitemap `<lastmod>` is measured (this session) to be CONTAMINATED by
   a site-wide bulk retouch: a 2015-09-23 article
   (`/2015/09/minor-nfl-transactions-9-23-15`) carries `<lastmod>
   2025-12-26T23:27:00-06:00` while its own JSON-LD `datePublished` correctly
   reads `2015-09-23T16:47:54-05:00` (`dateModified` also reads
   2025-12-26 -- a genuine re-touch, not a parsing bug). `<lastmod>` is
   therefore recorded for transparency but never treated as a publish-date
   proxy. Two better signals are used instead: (a) the URL path itself
   encodes `/YYYY/MM/slug` -- measured on the same article to match
   `datePublished` exactly (`/2015/09/...` for a 2015-09-23 publish) -- a
   free, reliable, no-extra-fetch year/month bound available for every url in
   the full inventory; (b) a per-article JSON-LD `datePublished` fetch (exact
   to the second) for a stratified verification sample, the same
   per-article-verification pattern the PFT ingestion already uses.

Source: `https://www.profootballrumors.com/sitemap.xml`, a WordPress sitemap
index. Measured this session: yearly post chunks
`sitemap-posttype-post.YYYY.xml` exist for 2013-2026, but 2013 and earlier
return HTTP 200 with an empty urlset (one homepage placeholder `<url>`, no
real content) -- true coverage starts at 2014. A `sitemap-posttype-page.xml`
(static pages, not posts) and `sitemap-home.xml` also exist in the index and
are deliberately excluded (not articles).

Respects `robots.txt` (fetched and read this session):
`https://www.profootballrumors.com/robots.txt` sets `Crawl-delay: 1` for
`User-agent: *` and disallows only unrelated paths (`/wp-admin/`, `/search`,
query-string variants) -- nothing on the sitemap or article paths used here.

Private-research use only, matching this project's existing CFBD/cfbfastR and
PFT precedent (`docs/data_feasibility.md` License item 6: "private
caching/retention" permitted, raw tables "must never be republished"). Pro
Football Rumors' own terms of use were not independently reviewed this
session (label: inferred policy stance, not a verified legal fact) -- treat
this archive the same way: cache privately, never republish, never
redistribute raw rows.

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_transaction_news.py \\
        --out data/raw/pfr_transactions --start 2014 --end 2026

    # Stratified per-article JSON-LD verification sample (requires the
    # yearly chunks for the sampled years already on disk):
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_transaction_news.py \\
        --out data/raw/pfr_transactions --verify-sample 200

Writes under --out/<snapshot>/ (default data/raw/pfr_transactions/<UTC
timestamp>) -- gitignored by the repository's existing `data/raw/**` rule.
The timestamped snapshot subdirectory matches this repo's existing
convention: `nfl_ats.snapshots.latest_snapshot()` treats ANY directory
directly under `data/raw/` that contains a `manifest.json` as a candidate
schedules snapshot, so `manifest.json` must never sit directly at
`data/raw/pfr_transactions/manifest.json` -- it must be nested one level
down. Without `--snapshot`, a run resumes the most recent existing snapshot
subdirectory under `--out` if one exists, or creates a fresh one.

    <snapshot>/yearly/<YYYY>.parquet        one row per PFR post url in that
                                              year's sitemap chunk.
    <snapshot>/index.parquet                concatenation of every yearly
                                              file present in this snapshot.
    <snapshot>/manifest.json                run metadata + coverage summary.
    <snapshot>/sample_articles/<slug>.json  per-article JSON-LD spot checks
                                              (--verify-sample).
    <snapshot>/date_verification_summary.json  aggregate match-rate stats
                                              from the most recent
                                              --verify-sample run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

SITEMAP_INDEX_URL = "https://www.profootballrumors.com/sitemap.xml"
YEARLY_CHUNK_RE = re.compile(r"sitemap-posttype-post\.(\d{4})\.xml$")
URL_YEAR_MONTH_SLUG_RE = re.compile(r"profootballrumors\.com/(\d{4})/(\d{2})/([^/]+)/?$")
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
CRAWL_DELAY_SECONDS = 1.0
SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
SNAPSHOT_DIR_RE = re.compile(r"^\d{8}T\d{6}Z$")


def resolve_snapshot_dir(out_dir: Path, snapshot: str | None) -> Path:
    """Return the timestamped snapshot subdirectory to write/resume into.

    A manifest.json must never sit directly at `out_dir` (see module
    docstring). If `snapshot` is given, use/create `out_dir/snapshot`.
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


# URL-slug keyword list for "transaction relevant" news (signings, cuts,
# trades, IR/practice-squad moves). Deliberately over-inclusive (recall over
# precision), matching the injury-news script's INJURY_KEYWORDS approach:
# every PFR post url is kept regardless of this match, with a
# `transaction_relevant` boolean, so the keyword line can be redrawn later
# without re-fetching. Headline-from-slug is extracted for every url
# regardless, so the full inventory is useful even for urls this list misses.
TRANSACTION_KEYWORDS = [
    "signs",
    "sign-",
    "-sign",
    "signed",
    "signing",
    "re-signs",
    "resigns",
    "re-sign",
    "agree-to-terms",
    "agrees-to-terms",
    "extend",
    "extension",
    "extends",
    "franchise-tag",
    "tenders",
    "tendered",
    "tag-",
    "cut-",
    "cuts",
    "waived",
    "waive-",
    "waives",
    "released",
    "release-",
    "releases",
    "trade",
    "trades",
    "traded",
    "acquire",
    "acquires",
    "acquired",
    "claim-",
    "claims",
    "claimed",
    "activated",
    "activate-",
    "activates",
    "placed-on-ir",
    "injured-reserve",
    "-on-ir",
    "-ir-",
    "promoted",
    "promote-",
    "elevate",
    "elevated",
    "elevation",
    "practice-squad",
    "suspend",
    "suspended",
    "suspension",
    "retires",
    "retirement",
    "retiring",
    "minor-moves",
    "minor-transactions",
    "minor-nfl-transactions",
    "roster-moves",
    "transactions",
    "extra-points",
    "free-agent",
    "free-agency",
    "undrafted",
    "restructure",
    "restructures",
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
    """Return [(year, sitemap_url), ...] sorted chronologically, posts only
    (excludes the pages sitemap and the home sitemap)."""

    raw = _fetch(SITEMAP_INDEX_URL, limiter)
    root = ElementTree.fromstring(raw)
    chunks: list[tuple[str, str]] = []
    for sitemap_el in root.findall(f"{SITEMAP_NS}sitemap"):
        loc_el = sitemap_el.find(f"{SITEMAP_NS}loc")
        if loc_el is None or not loc_el.text:
            continue
        loc = loc_el.text.strip()
        match = YEARLY_CHUNK_RE.search(loc)
        if not match:
            continue
        chunks.append((match.group(1), loc))
    chunks.sort(key=lambda pair: pair[0])
    return chunks


def _slug_matches_transaction_keyword(slug: str) -> list[str]:
    lowered = slug.lower()
    return [kw for kw in TRANSACTION_KEYWORDS if kw in lowered]


def parse_yearly_sitemap(raw: bytes, year: str) -> pd.DataFrame:
    root = ElementTree.fromstring(raw)
    rows: list[dict[str, object]] = []
    for url_el in root.findall(f"{SITEMAP_NS}url"):
        loc_el = url_el.find(f"{SITEMAP_NS}loc")
        lastmod_el = url_el.find(f"{SITEMAP_NS}lastmod")
        if loc_el is None or not loc_el.text:
            continue
        loc = loc_el.text.strip()
        if loc.rstrip("/") == "https://www.profootballrumors.com":
            continue  # empty-year placeholder entry (2013 and earlier)
        lastmod = lastmod_el.text.strip() if lastmod_el is not None and lastmod_el.text else None
        url_match = URL_YEAR_MONTH_SLUG_RE.search(loc)
        if url_match:
            url_year, url_month, slug = url_match.groups()
        else:
            url_year, url_month, slug = None, None, loc.rstrip("/").rsplit("/", 1)[-1]
        matched = _slug_matches_transaction_keyword(slug)
        rows.append(
            {
                "sitemap_year": year,
                "url": loc,
                "lastmod_contaminated": lastmod,
                "url_year": url_year,
                "url_month": url_month,
                "slug": slug,
                "headline_from_slug": slug.replace("-", " "),
                "matched_keywords": "|".join(matched),
                "transaction_relevant": bool(matched),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["lastmod_contaminated"] = pd.to_datetime(
            frame["lastmod_contaminated"], errors="coerce", utc=True
        )
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
    yearly_dir = out_dir / "yearly"
    yearly_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching sitemap index: {SITEMAP_INDEX_URL}")
    chunks = fetch_sitemap_index(limiter)
    chunks = [(year, url) for year, url in chunks if start <= year <= end]
    print(f"Sitemap index: {len(chunks)} yearly post chunks in [{start}, {end}]")

    processed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    per_year_counts: dict[str, dict[str, int]] = {}

    for year, sitemap_url in chunks:
        out_path = yearly_dir / f"{year}.parquet"
        if out_path.exists() and not force:
            skipped.append(year)
            continue
        try:
            raw = _fetch(sitemap_url, limiter)
            frame = parse_yearly_sitemap(raw, year)
        except Exception as error:
            print(f"  FAILED year {year}: {error}", file=sys.stderr)
            failed.append(year)
            continue
        frame.to_parquet(out_path, index=False)
        n_relevant = int(frame["transaction_relevant"].sum()) if len(frame) else 0
        per_year_counts[year] = {"urls": len(frame), "transaction_relevant": n_relevant}
        processed.append(year)
        print(f"  {year}: {len(frame)} urls, {n_relevant} transaction-relevant")

    return _rebuild_index_and_manifest(
        out_dir,
        start=start,
        end=end,
        years_processed_this_run=processed,
        years_skipped_already_present=skipped,
        years_failed=failed,
    )


def _rebuild_index_and_manifest(
    out_dir: Path,
    *,
    start: str,
    end: str,
    years_processed_this_run: list[str],
    years_skipped_already_present: list[str],
    years_failed: list[str],
) -> dict[str, object]:
    """Concatenate every yearly parquet on disk into index.parquet and write
    manifest.json. Shared by `ingest()` (after a fetch) and
    `recompute_keywords()` (after a local, no-fetch keyword redraw)."""

    yearly_dir = out_dir / "yearly"
    all_yearly = sorted(yearly_dir.glob("*.parquet"))
    if all_yearly:
        combined = pd.concat([pd.read_parquet(p) for p in all_yearly], ignore_index=True)
        combined = combined.sort_values(["url_year", "url_month", "url"]).reset_index(drop=True)
        combined.to_parquet(out_dir / "index.parquet", index=False)
        full_per_year_counts = {
            str(y): {
                "urls": int((combined["sitemap_year"] == y).sum()),
                "transaction_relevant": int(
                    combined.loc[combined["sitemap_year"] == y, "transaction_relevant"].sum()
                ),
            }
            for y in sorted(combined["sitemap_year"].unique())
        }
    else:
        combined = pd.DataFrame()
        full_per_year_counts = {}

    manifest = {
        "source": (
            "https://www.profootballrumors.com/sitemap.xml "
            "(Pro Football Rumors transaction-wire posts)"
        ),
        "fetched_at": datetime.now(UTC).isoformat(),
        "requested_range": [start, end],
        "years_processed_this_run": years_processed_this_run,
        "years_skipped_already_present": years_skipped_already_present,
        "years_failed": years_failed,
        "crawl_delay_seconds_honored": CRAWL_DELAY_SECONDS,
        "user_agent": USER_AGENT,
        "cumulative_index_rows": len(combined),
        "cumulative_transaction_relevant_rows": (
            int(combined["transaction_relevant"].sum()) if len(combined) else 0
        ),
        "cumulative_years_on_disk": len(all_yearly),
        "per_year_counts": full_per_year_counts,
        "transaction_keywords": TRANSACTION_KEYWORDS,
        "lastmod_contamination_note": (
            "sitemap <lastmod> is measured to be unreliable as a publish-date proxy "
            "(a site-wide bulk retouch bumped many old posts' lastmod forward, e.g. a "
            "2015-09-23 article carries lastmod 2025-12-26); recorded as "
            "lastmod_contaminated for transparency only. url_year/url_month (parsed "
            "from the /YYYY/MM/slug path) match the true publish date on every article "
            "checked this session and are the reliable free proxy; per-article JSON-LD "
            "datePublished (see --verify-sample) is the ground truth to the second."
        ),
        "usage_note": (
            "Private research caching only, matching this project's CFBD/cfbfastR and "
            "ProFootballTalk precedent (docs/data_feasibility.md License item 6). Never "
            "republish raw rows. Pro Football Rumors' terms of use were not "
            "independently reviewed this session -- this is a policy stance, not a "
            "verified legal fact."
        ),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest


def recompute_keywords(out_dir: Path) -> dict[str, object]:
    """Redraw `matched_keywords`/`transaction_relevant` on every yearly
    parquet already on disk from the current TRANSACTION_KEYWORDS list,
    using the cached `slug` column -- no network fetch. Matches the module
    docstring's stated design: the keyword line can be redrawn without
    re-fetching."""

    yearly_dir = out_dir / "yearly"
    yearly_files = sorted(yearly_dir.glob("*.parquet"))
    if not yearly_files:
        raise SystemExit(f"No yearly chunks found under {yearly_dir} -- run bulk ingestion first")

    for path in yearly_files:
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        matched = frame["slug"].map(_slug_matches_transaction_keyword)
        frame["matched_keywords"] = matched.map("|".join)
        frame["transaction_relevant"] = matched.map(bool)
        frame.to_parquet(path, index=False)
        print(f"  {path.name}: {int(frame['transaction_relevant'].sum())} transaction-relevant")

    return _rebuild_index_and_manifest(
        out_dir,
        start="(recompute)",
        end="(recompute)",
        years_processed_this_run=[],
        years_skipped_already_present=[p.stem for p in yearly_files],
        years_failed=[],
    )


DATE_PUBLISHED_RE = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')
DATE_MODIFIED_RE = re.compile(r'"dateModified"\s*:\s*"([^"]+)"')
HEADLINE_RE = re.compile(r'"headline"\s*:\s*"([^"]+)"')


def _select_stratified_sample(
    out_dir: Path, total_n: int, *, relevant_only: bool = False
) -> pd.DataFrame:
    yearly_dir = out_dir / "yearly"
    yearly_files = sorted(yearly_dir.glob("*.parquet"))
    if not yearly_files:
        raise SystemExit(f"No yearly chunks found under {yearly_dir} -- run bulk ingestion first")

    per_year_n = max(1, total_n // len(yearly_files))
    samples: list[pd.DataFrame] = []
    for path in yearly_files:
        frame = pd.read_parquet(path)
        if relevant_only:
            frame = frame.loc[frame["transaction_relevant"]]
        if frame.empty:
            continue
        frame = frame.sort_values("url").reset_index(drop=True)
        step = max(1, len(frame) // per_year_n)
        picked = frame.iloc[::step].head(per_year_n)
        samples.append(picked)
    if not samples:
        raise SystemExit("No urls available to sample")
    return pd.concat(samples, ignore_index=True)


def verify_sample(
    out_dir: Path,
    total_n: int,
    limiter: RateLimiter | None = None,
    *,
    relevant_only: bool = False,
    summary_filename: str = "date_verification_summary.json",
) -> dict[str, object]:
    """Fetch a stratified (evenly spread across years) sample of real article
    pages, extract JSON-LD datePublished/dateModified/headline, and compare
    against (a) the contaminated sitemap lastmod and (b) the free url_year/
    url_month proxy -- the task's "measure fetch cost and date reliability"
    step."""

    limiter = limiter or RateLimiter(CRAWL_DELAY_SECONDS)
    candidates = _select_stratified_sample(out_dir, total_n, relevant_only=relevant_only)

    sample_dir = out_dir / "sample_articles"
    sample_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    started = time.monotonic()
    for _, row in candidates.iterrows():
        url = str(row["url"])
        slug = str(row["slug"])
        print(f"Fetching sample article: {url}")
        try:
            raw = _fetch(url, limiter).decode("utf-8", errors="ignore")
        except Exception as error:
            print(f"  FAILED: {error}", file=sys.stderr)
            records.append({"url": url, "slug": slug, "fetch_failed": str(error)})
            continue
        date_published = DATE_PUBLISHED_RE.findall(raw)
        date_modified = DATE_MODIFIED_RE.findall(raw)
        headline = HEADLINE_RE.findall(raw)
        record = {
            "url": url,
            "slug": slug,
            "sitemap_year": row["sitemap_year"],
            "url_year": row["url_year"],
            "url_month": row["url_month"],
            "lastmod_contaminated": str(row["lastmod_contaminated"]),
            "json_ld_date_published": date_published[:1],
            "json_ld_date_modified": date_modified[:1],
            "json_ld_headline": headline[:1],
        }
        with (sample_dir / f"{slug}.json").open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        records.append(record)
        print(f"  url_year/month:    {row['url_year']}/{row['url_month']}")
        print(f"  lastmod (contam.): {row['lastmod_contaminated']}")
        print(f"  datePublished:     {date_published[:1]}")

    summary = _summarize_verification(records, elapsed_seconds=time.monotonic() - started)
    with (out_dir / summary_filename).open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _summarize_verification(records: list[dict[str, object]], *, elapsed_seconds: float) -> dict:
    ok = [r for r in records if r.get("json_ld_date_published")]
    n = len(records)
    n_ok = len(ok)

    url_ym_matches = 0
    for r in ok:
        published = str(r["json_ld_date_published"][0])
        try:
            published_dt = datetime.fromisoformat(published)
        except ValueError:
            continue
        if (
            r.get("url_year") == f"{published_dt.year:04d}"
            and r.get("url_month") == f"{published_dt.month:02d}"
        ):
            url_ym_matches += 1

    lastmod_close_matches = 0
    lastmod_comparable = 0
    for r in ok:
        published = str(r["json_ld_date_published"][0])
        lastmod = r.get("lastmod_contaminated")
        if not lastmod or lastmod in ("None", "NaT"):
            continue
        try:
            published_dt = datetime.fromisoformat(published)
            lastmod_dt = datetime.fromisoformat(str(lastmod).replace("Z", "+00:00"))
        except ValueError:
            continue
        lastmod_comparable += 1
        if abs((lastmod_dt - published_dt.astimezone(lastmod_dt.tzinfo)).total_seconds()) <= 3600:
            lastmod_close_matches += 1

    return {
        "sampled": n,
        "fetched_ok": n_ok,
        "fetch_failures": n - n_ok,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "mean_seconds_per_article": round(elapsed_seconds / n, 2) if n else None,
        "url_year_month_matches_datePublished": url_ym_matches,
        "url_year_month_match_rate": round(url_ym_matches / n_ok, 4) if n_ok else None,
        "lastmod_within_1h_of_datePublished": lastmod_close_matches,
        "lastmod_comparable_rows": lastmod_comparable,
        "lastmod_reliability_rate": (
            round(lastmod_close_matches / lastmod_comparable, 4) if lastmod_comparable else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw/pfr_transactions"))
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
    parser.add_argument("--start", default="2014", help="YYYY, inclusive")
    parser.add_argument("--end", default="2026", help="YYYY, inclusive")
    parser.add_argument("--force", action="store_true", help="Refetch years already on disk")
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Instead of bulk ingestion, fetch a stratified (even per-year) sample of "
            "N real article pages total to measure fetch cost and datePublished "
            "reliability vs. the contaminated lastmod and the free url_year/url_month "
            "proxy. Requires bulk ingestion to have already been run for the years "
            "being sampled."
        ),
    )
    parser.add_argument(
        "--relevant-only",
        action="store_true",
        help="Restrict --verify-sample to transaction_relevant=True urls only.",
    )
    parser.add_argument(
        "--recompute-keywords",
        action="store_true",
        help=(
            "Redraw matched_keywords/transaction_relevant on already-fetched yearly "
            "parquets from the current TRANSACTION_KEYWORDS list, no network fetch."
        ),
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    snapshot_dir = resolve_snapshot_dir(args.out, args.snapshot)
    print(f"Snapshot directory: {snapshot_dir}")
    limiter = RateLimiter(CRAWL_DELAY_SECONDS)

    if args.recompute_keywords:
        manifest = recompute_keywords(snapshot_dir)
        print(json.dumps(manifest, indent=2))
        return

    if args.verify_sample is not None:
        summary_filename = (
            "date_verification_summary_relevant_only.json"
            if args.relevant_only
            else "date_verification_summary.json"
        )
        summary = verify_sample(
            snapshot_dir,
            args.verify_sample,
            limiter=limiter,
            relevant_only=args.relevant_only,
            summary_filename=summary_filename,
        )
        print(json.dumps(summary, indent=2))
        return

    manifest = ingest(snapshot_dir, args.start, args.end, force=args.force, limiter=limiter)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
