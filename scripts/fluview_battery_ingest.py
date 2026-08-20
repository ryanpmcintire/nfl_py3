"""Bulk-ingest CDC Delphi FluView state-level ILI history for the 23 US
states that host an NFL franchise (2010-2025), one HTTP request per state,
covering the FULL multi-issue revision history so the screen script
(``scripts/fluview_battery_screen.py``) can build a point-in-time-safe
as-of feature without re-querying.

**Measured, this session** (frozen in ``docs/fluview_battery.md`` section 1,
read that first): the Delphi Epidata ``issues`` parameter is EXACT-MATCH,
not "latest as of", so reconstructing "what was known as of date X" needs
the full issue history fetched once and filtered client-side -- one bulk
call per state (confirmed: 33-34k rows, ~9MB, ~1.5s, no truncation) rather
than one call per (state, week), which would be thousands of calls.

**Rate limit (read, Delphi's own docs)**: anonymous access is 60
requests/hour; a free API key (self-service, email only) removes the
limit entirely, but none is configured here. This script makes exactly
24 requests (23 states + ``nat``), comfortably inside 60/hour on its own
-- the exploratory verification probing done ahead of writing this script
is what consumed the hourly budget in earlier testing, not this script's
own request count.

Endpoint: ``https://api.delphi.cmu.edu/epidata/fluview/`` -- free, no key.

Output: ``data/raw/fluview/<UTC timestamp>/fluview_raw.parquet`` (one row
per state x epiweek x issue -- the full revision history) plus
``manifest.json`` recording every request made (url, http status, byte
count, row count, retry count, elapsed seconds). Gitignored, per repo
convention (``data/raw`` is never committed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

DELPHI_ENDPOINT = "https://api.delphi.cmu.edu/epidata/fluview/"

# docs/fluview_battery.md section 2 -- static team -> state mapping, unique
# state list (23 states covering all 34 historical nflverse team codes).
STATE_BY_TEAM: dict[str, str] = {
    "ARI": "az",
    "ATL": "ga",
    "BAL": "md",
    "BUF": "ny",
    "CAR": "nc",
    "CHI": "il",
    "CIN": "oh",
    "CLE": "oh",
    "DAL": "tx",
    "DEN": "co",
    "DET": "mi",
    "GB": "wi",
    "HOU": "tx",
    "IND": "in",
    "JAX": "fl",
    "KC": "mo",
    "LA": "ca",
    "LAC": "ca",
    "LV": "nv",
    "MIA": "fl",
    "MIN": "mn",
    "NE": "ma",
    "NO": "la",
    "NYG": "nj",
    "NYJ": "nj",
    "OAK": "ca",
    "PHI": "pa",
    "PIT": "pa",
    "SD": "ca",
    "SEA": "wa",
    "SF": "ca",
    "STL": "mo",
    "TB": "fl",
    "TEN": "tn",
    "WAS": "md",
}
STATES = sorted(set(STATE_BY_TEAM.values()))

# Also fetch national ("nat") for the peak-week predeclaration cross-check
# (docs/fluview_battery.md section 4 -- already measured/frozen, but kept
# here so the raw national series is reproducible from this ingest too).
REGIONS = [*STATES, "nat"]

EPIWEEK_LOW = "201040"
EPIWEEK_HIGH = "202552"
ISSUES_LOW = "201040"
ISSUES_HIGH = "202608"  # a little past 202552 as safety margin for late revisions

RATE_LIMIT_SECONDS = 3.0
MAX_RETRIES = 6
INITIAL_BACKOFF = 10.0

_last_request_ts = 0.0


def _polite_get(url: str) -> tuple[int, bytes, int]:
    """GET url, enforcing a minimum inter-request gap and exponential
    backoff on HTTP 429 (measured this session: Delphi rate-limits after
    roughly 20 rapid single-epiweek probes). Returns (status, body_bytes,
    retries_used)."""

    global _last_request_ts
    backoff = INITIAL_BACKOFF
    retries = 0
    while True:
        elapsed = time.time() - _last_request_ts
        if elapsed < RATE_LIMIT_SECONDS:
            time.sleep(RATE_LIMIT_SECONDS - elapsed)
        _last_request_ts = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nfl-ats-research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.status, resp.read(), retries
        except urllib.error.HTTPError as exc:
            status = exc.code
            body = exc.read() if exc.fp else b""
            if status != 429 or retries >= MAX_RETRIES:
                return status, body, retries
        except urllib.error.URLError:
            if retries >= MAX_RETRIES:
                return 0, b"", retries
        retries += 1
        time.sleep(backoff)
        backoff *= 2.0


def fetch_region(region: str) -> dict[str, Any]:
    url = (
        f"{DELPHI_ENDPOINT}?regions={region}"
        f"&epiweeks={EPIWEEK_LOW}-{EPIWEEK_HIGH}"
        f"&issues={ISSUES_LOW}-{ISSUES_HIGH}"
    )
    t0 = time.time()
    status, body, retries = _polite_get(url)
    elapsed = time.time() - t0
    record: dict[str, Any] = {
        "region": region,
        "url": url,
        "http_status": status,
        "byte_count": len(body),
        "retries": retries,
        "elapsed_seconds": elapsed,
    }
    if status != 200:
        record["parsed_ok"] = False
        record["n_rows"] = 0
        return {"manifest_entry": record, "rows": []}
    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        record["parsed_ok"] = False
        record["error"] = str(exc)
        record["n_rows"] = 0
        return {"manifest_entry": record, "rows": []}
    rows = payload.get("epidata", []) if payload.get("result") == 1 else []
    record["parsed_ok"] = True
    record["delphi_result"] = payload.get("result")
    record["delphi_message"] = payload.get("message")
    record["n_rows"] = len(rows)
    return {"manifest_entry": record, "rows": rows}


def run_ingest(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for i, region in enumerate(REGIONS, start=1):
        print(f"[{i}/{len(REGIONS)}] fetching {region} ...")
        result = fetch_region(region)
        entry = result["manifest_entry"]
        manifest_entries.append(entry)
        print(
            f"  status={entry['http_status']} rows={entry['n_rows']} "
            f"bytes={entry['byte_count']} retries={entry['retries']} "
            f"elapsed={entry['elapsed_seconds']:.2f}s"
        )
        for row in result["rows"]:
            row["region"] = region
            all_rows.append(row)

    if not all_rows:
        raise SystemExit(
            "no rows fetched from any region -- aborting, not writing an empty snapshot"
        )

    df = pd.DataFrame(all_rows)
    keep_cols = [
        "region",
        "epiweek",
        "issue",
        "lag",
        "release_date",
        "num_ili",
        "num_patients",
        "num_providers",
        "wili",
        "ili",
    ]
    df = df.loc[:, [c for c in keep_cols if c in df.columns]]
    df["epiweek"] = df["epiweek"].astype("int64")
    df["issue"] = df["issue"].astype("int64")
    df["lag"] = df["lag"].astype("int64")
    df["ili"] = pd.to_numeric(df["ili"], errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="raise")

    out_path = output_dir / "fluview_raw.parquet"
    df.to_parquet(out_path, index=False)

    per_region_counts = df.groupby("region").size().to_dict()
    manifest = {
        "source": "https://api.delphi.cmu.edu/epidata/fluview/",
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epiweek_range": [EPIWEEK_LOW, EPIWEEK_HIGH],
        "issues_range": [ISSUES_LOW, ISSUES_HIGH],
        "regions_requested": REGIONS,
        "state_by_team": STATE_BY_TEAM,
        "requests": manifest_entries,
        "n_rows_total": len(df),
        "n_rows_per_region": {k: int(v) for k, v in per_region_counts.items()},
        "output_parquet": str(out_path.relative_to(REPO)),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nwrote {out_path} ({len(df)} rows, {out_path.stat().st_size} bytes)")
    print(f"wrote {manifest_path}")
    failed = [e["region"] for e in manifest_entries if not e.get("parsed_ok")]
    if failed:
        print(f"WARNING: {len(failed)} region(s) failed to fetch: {failed}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output or (
        REPO / "data" / "raw" / "fluview" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    run_ingest(output_dir)


if __name__ == "__main__":
    main()
