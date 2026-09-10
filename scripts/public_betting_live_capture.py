from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
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
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
