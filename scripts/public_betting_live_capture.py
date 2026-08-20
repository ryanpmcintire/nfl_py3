"""Live, prospective capture of actionnetwork.com/nfl/public-betting (era2 template).

Item 1 of `docs/public_betting_sourcing.md` section 7's "What a follow-up
session should fetch": the Wayback backfill (`scripts/ingest_public_
betting.py`, `data/raw/public_betting/20260820T111148Z/`) is frozen history
by construction -- it can never see the CURRENT week. This script hits the
live page directly (never through Wayback) and appends one timestamped
snapshot, meant to be invoked by `scripts/public_betting_capture.ps1` on a
weekly Task Scheduler cadence (two runs/week: Saturday and Sunday noon ET,
bracketing the slate) the same way `scripts/odds_capture.ps1` wraps
`nfl-ats odds-ingest`.

**Robots.txt**: NOT re-fetched this session -- verified instead from
`docs/public_betting_sourcing.md` section 1 (read this session), which
already measured `actionnetwork.com/robots.txt` has no `Crawl-delay` and no
`Disallow` covering `/nfl/public-betting`, the exact path this script
fetches. Re-fetching robots.txt for a fact already measured and documented
would just be a second, redundant request against the origin site.

**Volume**: one HTTP GET per invocation (the live page itself). No CDX
query, no per-capture loop -- this is the live-site analogue of a single
one of the backfill's ~150 archived fetches, not a bulk job.

**Parsing**: reuses `extract_next_data` / `parse_actionnetwork_snapshot`
from `scripts/ingest_public_betting.py` (same directory, imported directly
rather than duplicated -- that module's brace-matching `__NEXT_DATA__`
extractor and era1/era2 dispatch are exactly what this script also needs,
and duplicating ~150 lines of parser logic would be a second chance to get
it wrong). As of this session the live page has always been observed in
era2 (`scoreboardResponse`) shape -- era1 predates Nov 2022 -- but the
dispatcher tries both, unmodified, so a template regression is reported
(`era` = `unrecognized_shape` / `no_next_data`) rather than silently
mis-parsed.

Writes one snapshot directory per invocation under
`data/raw/public_betting_live/<UTC timestamp>/`:

    raw_html/<timestamp>.html   the fetched page (audit/re-parse without refetch)
    index.parquet                parsed per-game rows, this capture only
    manifest.json                run metadata (era, counts, URL, http status)

This is raw-ingestion output, the same `data/raw/<source>/<UTC
timestamp>/manifest.json` convention `scripts/ingest_public_betting.py`/
`scripts/ingest_injury_news.py` already use, gitignored under the
repository's existing `data/raw/**` rule -- not a generated research
artifact, so it stays out of the gitignored top-level output directory
those use for run results.

Prints one final line, a single-line JSON summary
(`{"snapshot_id": ..., "era": ..., "rows": ..., "rows_with_public_data": ...}`),
for `scripts/public_betting_capture.ps1` to parse into its own log line --
the same convention `nfl-ats odds-ingest`'s stdout already uses for
`scripts/odds_capture.ps1`.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/public_betting_live_capture.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

# Same-directory import, not a package import: `python scripts/foo.py` puts
# `scripts/` at sys.path[0], so this resolves without any src/nfl_ats
# dependency, matching this script's own standalone-script convention.
from ingest_public_betting import (
    USER_AGENT,
    parse_actionnetwork_snapshot,
)

REPO = Path(__file__).resolve().parents[1]
LIVE_URL = "https://www.actionnetwork.com/nfl/public-betting"
DEFAULT_OUT = REPO / "data/raw/public_betting_live"
FETCH_TIMEOUT_SECONDS = 30
FETCH_RETRIES = 3


def fetch_live_page(
    url: str = LIVE_URL, *, timeout: int = FETCH_TIMEOUT_SECONDS
) -> tuple[str, int]:
    """Single GET against the live page. Returns (html_text, http_status).

    No rate limiter -- unlike the Wayback backfill (which makes ~150
    sequential requests against web.archive.org and so needs one), this is
    exactly one request per invocation against the origin site, well within
    the "~1 req" budget this task specified and the robots.txt policy
    `docs/public_betting_sourcing.md` section 1 already measured as
    permitting this path.
    """

    last_error: Exception | None = None
    for attempt in range(FETCH_RETRIES):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                status = getattr(response, "status", 200)
                return raw.decode("utf-8", errors="ignore"), int(status)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            print(f"  fetch failed ({attempt + 1}/{FETCH_RETRIES}) {url}: {error}", file=sys.stderr)
    assert last_error is not None
    raise last_error


def capture(url: str = LIVE_URL, out_root: Path = DEFAULT_OUT) -> dict:
    capture_ts = pd.Timestamp.now(tz="UTC")
    stamp = capture_ts.strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = out_root / stamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    html, status = fetch_live_page(url)
    (snapshot_dir / "raw_html").mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "raw_html" / f"{stamp}.html").write_text(html, encoding="utf-8")

    era, rows, error = parse_actionnetwork_snapshot(html, capture_ts)
    frame = pd.DataFrame(rows)
    frame_path = snapshot_dir / "index.parquet"
    if not frame.empty:
        frame.to_parquet(frame_path, index=False)
    else:
        # An empty DataFrame still needs a parquet on disk so downstream
        # tooling (e.g. `nfl_ats.snapshots.latest_snapshot`-style directory
        # scans) never has to special-case a missing file; write an
        # explicitly empty, schema-less frame rather than skipping the write.
        pd.DataFrame().to_parquet(frame_path, index=False)

    n_rows = len(frame)
    n_with_data = int(frame["has_any_public_data"].sum()) if "has_any_public_data" in frame else 0

    manifest = {
        "source": "actionnetwork_live",
        "url": url,
        "fetched_at": capture_ts.isoformat(),
        "http_status": status,
        "era": era,
        "parse_error": error,
        "n_game_rows": n_rows,
        "n_game_rows_with_public_data": n_with_data,
        "usage_note": (
            "Private research caching only, matching this project's CFBD/PFT/PFR "
            "precedent (docs/data_feasibility.md License item 6). Never republish "
            "raw rows. Robots.txt for this path was verified via docs/public_betting_"
            "sourcing.md section 1 (measured that session), not re-fetched here."
        ),
    }
    with (snapshot_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    summary = {
        "snapshot_id": stamp,
        "era": era,
        "rows": n_rows,
        "rows_with_public_data": n_with_data,
        "http_status": status,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=LIVE_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summary = capture(args.url, args.out)
    # Single-line JSON on its own -- scripts/public_betting_capture.ps1
    # regex-matches this exact shape, mirroring how scripts/odds_capture.ps1
    # regex-matches nfl-ats odds-ingest's own JSON stdout.
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
