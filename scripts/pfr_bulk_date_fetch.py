"""Budgeted bulk per-article JSON-LD ``datePublished`` fetch for the Pro
Football Rumors (PFR) transaction-wire archive, targeted at the specific
in-season month-windows needed for `docs/pfr_transactions_sourcing.md`
section 6's two predeclared experiments (PFR-vs-PFT additivity per
(season, week, team); PFR foreshadowing of official roster-status changes),
run 2026-08-20.

Why a new script rather than reusing ``ingest_transaction_news.py --verify-
sample``: that mode draws a random STRATIFIED sample across all 13 years for
date-RELIABILITY measurement (already done, 325 articles, 100% url_year/month
match). This script instead needs COMPLETE coverage of a specific targeted
scope (transaction_relevant articles in-season for seasons 2022-2025) because
the downstream matching needs day/hour precision for every candidate article
that could fall inside a game's own lookback window, not a random subsample
of it -- a missed date is a false negative in the coverage measurement, not
a rounding error.

Scope, decided from the already-ingested manifest (measured this session):
REG weeks 1-18 for seasons 2022-2025 run kickoff-to-kickoff from the first
week of September to the first days of January; with a 9-day lookback before
the earliest cutoff, August of the season's start year is the natural
low-water mark. `data/raw/pfr_transactions/20260820T011126Z/index.parquet`
has 4,361 `transaction_relevant` rows with `url_year`/`url_month` in
Aug(Y)-Jan(Y+1) for Y in {2022, 2023, 2024, 2025} -- inside the ~5,400-fetch
budget without needing to trim further.

Caches every fetched (and every attempted) result into the existing
`sample_articles/<slug>.json` snapshot-dir convention
(`scripts/ingest_transaction_news.py`'s own `verify_sample` format), keyed by
slug, so a future session (or a re-run of this one) never re-fetches a URL
already on disk -- this script's OWN retry-on-resume logic depends on it.

Usage::

    .\\.tools\\uv.exe run --no-sync python scripts/pfr_bulk_date_fetch.py \\
        --snapshot data/raw/pfr_transactions/20260820T011126Z
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
from pathlib import Path

import pandas as pd

USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
CRAWL_DELAY_SECONDS = 1.0

DATE_PUBLISHED_RE = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')
DATE_MODIFIED_RE = re.compile(r'"dateModified"\s*:\s*"([^"]+)"')
HEADLINE_RE = re.compile(r'"headline"\s*:\s*"([^"]+)"')

SEASONS = (2022, 2023, 2024, 2025)


def _parse_seasons(raw: str) -> tuple[int, ...]:
    return tuple(sorted(int(part) for part in raw.split(",") if part.strip()))


def target_year_months(seasons: tuple[int, ...]) -> set[tuple[int, int]]:
    """Aug(Y) through Jan(Y+1) for every season Y -- covers REG weeks 1-18
    plus a lookback margin, measured against the actual REG schedule span
    (season 2025: 2025-09-04 through 2026-01-04, read from
    data/processed/game_features_pbp.parquet this session)."""

    months: set[tuple[int, int]] = set()
    for year in seasons:
        for month in (8, 9, 10, 11, 12):
            months.add((year, month))
        months.add((year + 1, 1))
    return months


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("data/raw/pfr_transactions/20260820T011126Z"),
    )
    parser.add_argument("--max-fetches", type=int, default=5400)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--seasons",
        type=_parse_seasons,
        default=SEASONS,
        metavar="Y1,Y2,...",
        help=(
            "Comma-separated season years to target (Aug(Y)-Jan(Y+1) window each, same "
            "scope logic as the original 2022-2025 run). Default: the original SEASONS "
            "tuple (2022-2025) -- pass a wider list (e.g. 2014,2015,...,2021) to extend "
            "date coverage to earlier seasons for a downstream feature battery. Already-"
            "cached slugs are skipped regardless of scope, so re-running with a wider "
            "list never re-fetches the original 4,361-row scope."
        ),
    )
    args = parser.parse_args()

    index_path = args.snapshot / "index.parquet"
    sample_dir = args.snapshot / "sample_articles"
    sample_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(index_path)
    frame["url_year"] = pd.to_numeric(frame["url_year"], errors="coerce")
    frame["url_month"] = pd.to_numeric(frame["url_month"], errors="coerce")

    months = target_year_months(args.seasons)
    print(f"Seasons in scope: {args.seasons}")
    in_scope = frame["transaction_relevant"] & frame.apply(
        lambda r: (
            (r["url_year"], r["url_month"]) in months
            if pd.notna(r["url_year"]) and pd.notna(r["url_month"])
            else False
        ),
        axis=1,
    )
    scoped = (
        frame.loc[in_scope].sort_values(["url_year", "url_month", "slug"]).reset_index(drop=True)
    )
    print(f"Target scope: {len(scoped)} transaction_relevant rows, seasons {args.seasons}")

    already_cached = 0
    to_fetch: list[tuple[str, str]] = []
    for _, row in scoped.iterrows():
        slug = str(row["slug"])
        cache_path = sample_dir / f"{slug}.json"
        if cache_path.exists():
            already_cached += 1
            continue
        to_fetch.append((str(row["url"]), slug))

    print(f"Already cached: {already_cached}; need to fetch: {len(to_fetch)}")
    to_fetch = to_fetch[: args.max_fetches]
    print(f"Fetching (capped at --max-fetches={args.max_fetches}): {len(to_fetch)}")

    limiter = RateLimiter(CRAWL_DELAY_SECONDS)
    started = time.monotonic()
    n_ok = 0
    n_failed = 0
    for i, (url, slug) in enumerate(to_fetch):
        try:
            raw = _fetch(url, limiter).decode("utf-8", errors="ignore")
            date_published = DATE_PUBLISHED_RE.findall(raw)
            date_modified = DATE_MODIFIED_RE.findall(raw)
            headline = HEADLINE_RE.findall(raw)
            record = {
                "url": url,
                "slug": slug,
                "json_ld_date_published": date_published[:1],
                "json_ld_date_modified": date_modified[:1],
                "json_ld_headline": headline[:1],
                "fetch_failed": None,
            }
            if date_published:
                n_ok += 1
            else:
                n_failed += 1
        except Exception as error:
            record = {
                "url": url,
                "slug": slug,
                "json_ld_date_published": [],
                "json_ld_date_modified": [],
                "json_ld_headline": [],
                "fetch_failed": str(error),
            }
            n_failed += 1
        with (sample_dir / f"{slug}.json").open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        if (i + 1) % args.progress_every == 0 or (i + 1) == len(to_fetch):
            elapsed = time.monotonic() - started
            print(
                f"  [{i + 1}/{len(to_fetch)}] ok={n_ok} failed={n_failed} "
                f"elapsed={elapsed:.0f}s rate={((i + 1) / elapsed if elapsed else 0):.2f}/s",
                flush=True,
            )

    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "target_scope": len(scoped),
                "already_cached_before_run": already_cached,
                "attempted_this_run": len(to_fetch),
                "ok_this_run": n_ok,
                "failed_this_run": n_failed,
                "elapsed_seconds": round(elapsed, 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
