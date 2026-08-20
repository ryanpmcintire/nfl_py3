"""Ingest USA Today's public NFL player-arrests table through its AJAX endpoint.

The landing page is fetched on every run to obtain the current anonymous
WordPress nonce. Table pages are then requested from ``admin-ajax.php`` with
the same ``cspFetchTable`` form used by the public application. Raw landing
HTML and JSON page responses are append-only inside an ignored timestamped
snapshot; rerunning with ``--snapshot`` skips valid cached pages and resumes
the missing page numbers.

Point-in-time contract: ``incident_date`` is the only source field treated as
an availability date. ``Outcome`` is retained only as ``outcome_archive_only``
in the full archival index and is mechanically absent from
``incidents_point_in_time.parquet``. Descriptions and links are excluded from
that point-in-time view as well because the source exposes no revision history.

Usage:
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_player_arrests.py
    .\\.tools\\uv.exe run --no-sync python scripts/ingest_player_arrests.py \\
        --snapshot 20260820T160000Z
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import certifi
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LANDING_URL = "https://databases.usatoday.com/nfl-arrests/"
AJAX_URL = "https://databases.usatoday.com/wp-admin/admin-ajax.php"
PAGE_ID = "10"
USER_AGENT = "nfl-ats-research/0.1 (private research ingestion)"
DEFAULT_DELAY_SECONDS = 1.5
SNAPSHOT_RE = re.compile(r"^\d{8}T\d{6}Z$")
SITEDATA_RE = re.compile(r"var\s+sitedata\s*=\s*(\{.*?\})\s*;", re.DOTALL)
PAGE_FILE_RE = re.compile(r"^page-(\d{4})\.json$")
CACHED_NONCE_RE = re.compile(rb'"ajax_nonce"\s*:\s*"(?!\[REDACTED_EPHEMERAL_NONCE\])[^\"]+"')
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

ARCHIVE_COLUMNS = (
    "record_id",
    "incident_date",
    "first_name",
    "last_name",
    "team",
    "position",
    "case_type",
    "category",
    "description_archive_only",
    "outcome_archive_only",
    "links_archive_only",
)
POINT_IN_TIME_COLUMNS = (
    "record_id",
    "incident_date",
    "first_name",
    "last_name",
    "team",
    "position",
    "case_type",
    "category",
)


class PlayerArrestsIngestError(RuntimeError):
    """Raised when the public page or AJAX response violates its contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_snapshot_dir(root: Path, snapshot: str | None) -> Path:
    snapshot_id = snapshot or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not SNAPSHOT_RE.fullmatch(snapshot_id):
        raise PlayerArrestsIngestError(f"snapshot must match YYYYMMDDTHHMMSSZ, got {snapshot_id!r}")
    path = root / snapshot_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_sitedata(landing_html: bytes) -> dict[str, Any]:
    text = landing_html.decode("utf-8", errors="replace")
    match = SITEDATA_RE.search(text)
    if match is None:
        raise PlayerArrestsIngestError("Landing page has no parseable `var sitedata = {...}`")
    try:
        payload = json.loads(html.unescape(match.group(1)))
    except json.JSONDecodeError as error:
        raise PlayerArrestsIngestError(
            f"Landing-page sitedata is not valid JSON: {error}"
        ) from error
    required = {"ajax_url", "ajax_nonce", "pageID", "sortBy", "sortOrder"}
    missing = sorted(required.difference(payload))
    if missing:
        raise PlayerArrestsIngestError("Landing-page sitedata missing: " + ", ".join(missing))
    if str(payload["pageID"]) != PAGE_ID:
        raise PlayerArrestsIngestError(
            f"Landing pageID changed from {PAGE_ID} to {payload['pageID']!r}"
        )
    if payload["sortBy"] != "Date" or payload["sortOrder"] != "desc":
        raise PlayerArrestsIngestError(
            "Landing default sort changed; refusing a silently reordered archive"
        )
    return payload


def sanitize_landing_html(landing_html: bytes, nonce: str) -> bytes:
    """Redact the anonymous, ephemeral WordPress nonce before caching HTML."""

    text = landing_html.decode("utf-8", errors="replace")
    if nonce not in text:
        raise PlayerArrestsIngestError("Parsed AJAX nonce is not present in landing HTML")
    return text.replace(nonce, "[REDACTED_EPHEMERAL_NONCE]").encode("utf-8")


def table_post_fields(nonce: str, page: int) -> dict[str, str]:
    if page < 1:
        raise ValueError("page must be >= 1")
    return {
        "action": "cspFetchTable",
        "security": nonce,
        "pageID": PAGE_ID,
        "blogID": "",
        "sortBy": "Date",
        "sortOrder": "desc",
        "page": str(page),
        "searches": "{}",
        "heads": "true",
    }


def _request(
    url: str,
    *,
    data: bytes | None = None,
    timeout: int = 45,
    retries: int = 3,
) -> bytes:
    last_error: Exception | None = None
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        "Referer": LANDING_URL,
    }
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
        headers["X-Requested-With"] = "XMLHttpRequest"
    for attempt in range(retries):
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(3.0 * (attempt + 1))
    assert last_error is not None
    raise PlayerArrestsIngestError(f"Request failed after {retries} attempts: {last_error}")


def fetch_landing() -> bytes:
    return _request(LANDING_URL)


def fetch_table_page(nonce: str, page: int) -> bytes:
    body = urllib.parse.urlencode(table_post_fields(nonce, page)).encode("utf-8")
    return _request(AJAX_URL, data=body)


def parse_table_page(payload: bytes, *, expected_page: int) -> tuple[list[dict[str, Any]], dict]:
    try:
        envelope = json.loads(payload)
    except json.JSONDecodeError as error:
        raise PlayerArrestsIngestError(f"Page {expected_page} is not JSON: {error}") from error
    if envelope.get("success") is not True or not isinstance(envelope.get("data"), dict):
        raise PlayerArrestsIngestError(f"Page {expected_page} returned an unsuccessful envelope")
    data = envelope["data"]
    rows = data.get("Result")
    if not isinstance(rows, list):
        raise PlayerArrestsIngestError(f"Page {expected_page} data.Result is not a list")
    params = data.get("defParams", {})
    reported_page = params.get("q.pageNumber")
    if reported_page is not None and int(reported_page) != expected_page:
        raise PlayerArrestsIngestError(
            f"Requested page {expected_page}, response reports page {reported_page}"
        )
    total = data.get("totalResults")
    page_size = params.get("q.pageSize")
    if total is None or page_size is None or int(page_size) <= 0:
        raise PlayerArrestsIngestError(
            f"Page {expected_page} missing totalResults or positive q.pageSize"
        )
    metadata = {
        "total_results": int(total),
        "page_size": int(page_size),
        "total_pages": math.ceil(int(total) / int(page_size)),
    }
    return rows, metadata


def normalize_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    required = {
        "PK_ID",
        "Date",
        "First_name",
        "Last_name",
        "Team",
        "Position",
        "Case_1",
        "Category",
        "Description",
        "Outcome",
        "Links",
    }
    for index, row in enumerate(rows):
        missing = sorted(required.difference(row))
        if missing:
            raise PlayerArrestsIngestError(
                f"Row {index} missing source fields: {', '.join(missing)}"
            )
    frame = pd.DataFrame(rows).rename(
        columns={
            "PK_ID": "record_id",
            "Date": "incident_date",
            "First_name": "first_name",
            "Last_name": "last_name",
            "Team": "team",
            "Position": "position",
            "Case_1": "case_type",
            "Category": "category",
            "Description": "description_archive_only",
            "Outcome": "outcome_archive_only",
            "Links": "links_archive_only",
        }
    )
    frame = frame.loc[:, ARCHIVE_COLUMNS].copy()
    frame["record_id"] = pd.to_numeric(frame["record_id"], errors="raise").astype("int64")
    frame["incident_date"] = pd.to_datetime(frame["incident_date"], errors="raise").dt.normalize()
    for column in ARCHIVE_COLUMNS:
        if column not in {"record_id", "incident_date"}:
            frame[column] = frame[column].fillna("").astype(str).str.strip()
    return frame.sort_values(["incident_date", "record_id"], ascending=[False, False]).reset_index(
        drop=True
    )


def point_in_time_view(archive: pd.DataFrame) -> pd.DataFrame:
    """Return only columns admitted to future point-in-time event matching."""

    return archive.loc[:, POINT_IN_TIME_COLUMNS].copy()


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise PlayerArrestsIngestError(
                f"Refusing to overwrite changed immutable raw file {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _page_number(path: Path) -> int:
    match = PAGE_FILE_RE.fullmatch(path.name)
    if match is None:
        raise PlayerArrestsIngestError(f"Unexpected page filename: {path.name}")
    return int(match.group(1))


def ingest(
    snapshot_dir: Path,
    *,
    max_pages: int | None,
    delay_seconds: float,
    landing_fetcher: Callable[[], bytes] = fetch_landing,
    page_fetcher: Callable[[str, int], bytes] = fetch_table_page,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    landing = landing_fetcher()
    sitedata = parse_sitedata(landing)
    nonce = str(sitedata["ajax_nonce"])
    landing_path = snapshot_dir / "landing_checks" / f"{run_id}.html"
    _write_once(landing_path, sanitize_landing_html(landing, nonce))

    pages_dir = snapshot_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_one_path = pages_dir / "page-0001.json"
    fetched_pages: list[int] = []
    skipped_pages: list[int] = []
    if not page_one_path.exists():
        raw = page_fetcher(nonce, 1)
        parse_table_page(raw, expected_page=1)
        _write_once(page_one_path, raw)
        fetched_pages.append(1)
    else:
        skipped_pages.append(1)
    _, page_metadata = parse_table_page(page_one_path.read_bytes(), expected_page=1)
    total_pages = int(page_metadata["total_pages"])
    target_pages = total_pages if max_pages is None else min(total_pages, max_pages)

    for page in range(2, target_pages + 1):
        path = pages_dir / f"page-{page:04d}.json"
        if path.exists():
            parse_table_page(path.read_bytes(), expected_page=page)
            skipped_pages.append(page)
            continue
        if delay_seconds > 0:
            sleeper(delay_seconds)
        raw = page_fetcher(nonce, page)
        parse_table_page(raw, expected_page=page)
        _write_once(path, raw)
        fetched_pages.append(page)
        print(f"Fetched page {page}/{total_pages}")

    page_paths = sorted(pages_dir.glob("page-*.json"), key=_page_number)
    all_rows: list[dict[str, Any]] = []
    page_hashes: dict[str, str] = {}
    for path in page_paths:
        page = _page_number(path)
        rows, metadata = parse_table_page(path.read_bytes(), expected_page=page)
        if metadata != page_metadata:
            raise PlayerArrestsIngestError(
                f"Page {page} pagination metadata differs from page 1: "
                f"{metadata} != {page_metadata}"
            )
        all_rows.extend(rows)
        page_hashes[path.name] = sha256_file(path)

    archive = normalize_rows(all_rows)
    if archive["record_id"].duplicated().any():
        duplicates = archive.loc[archive["record_id"].duplicated(), "record_id"].tolist()
        raise PlayerArrestsIngestError(
            f"Duplicate record_id values across cached pages: {duplicates}"
        )
    archive_path = snapshot_dir / "index.parquet"
    safe_path = snapshot_dir / "incidents_point_in_time.parquet"
    archive.to_parquet(archive_path, index=False)
    point_in_time_view(archive).to_parquet(safe_path, index=False)

    cached_page_numbers = [_page_number(path) for path in page_paths]
    complete = cached_page_numbers == list(range(1, total_pages + 1))
    landing_paths = sorted((snapshot_dir / "landing_checks").glob("*.html"))
    nonce_stored = any(CACHED_NONCE_RE.search(path.read_bytes()) for path in landing_paths)
    manifest: dict[str, Any] = {
        "source": LANDING_URL,
        "ajax_endpoint": AJAX_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_id": snapshot_dir.name,
        "landing_check": {
            "path": str(landing_path.relative_to(snapshot_dir)),
            "sha256": sha256_file(landing_path),
        },
        "access": {
            "authentication": "none",
            "action": "cspFetchTable",
            "page_id": PAGE_ID,
            "sort_by": "Date",
            "sort_order": "desc",
            "page_size": page_metadata["page_size"],
            "request_delay_seconds": delay_seconds,
            "nonce_stored": nonce_stored,
            "nonce_note": (
                "Landing captures are sanitized before write. If true, this snapshot also "
                "contains a legacy pre-sanitization landing check; the nonce is anonymous, "
                "ephemeral, and never copied into the manifest."
            ),
        },
        "source_total_results": page_metadata["total_results"],
        "source_total_pages": total_pages,
        "cached_pages": cached_page_numbers,
        "fetched_pages_this_run": fetched_pages,
        "skipped_pages_this_run": skipped_pages,
        "rows_cached": len(archive),
        "complete": complete,
        "incident_date_min": (
            archive["incident_date"].min().date().isoformat() if len(archive) else None
        ),
        "incident_date_max": (
            archive["incident_date"].max().date().isoformat() if len(archive) else None
        ),
        "point_in_time_policy": {
            "availability_field": "incident_date",
            "safe_index": safe_path.name,
            "forbidden_feature_inputs": [
                "outcome_archive_only",
                "description_archive_only",
                "links_archive_only",
            ],
            "note": (
                "The source exposes incident Date but no per-field publication/revision history. "
                "Outcome/resolution is retrospective and forbidden as a feature input."
            ),
        },
        "files": {
            "raw_pages_sha256": page_hashes,
            "index.parquet": sha256_file(archive_path),
            "incidents_point_in_time.parquet": sha256_file(safe_path),
        },
        "resume_command": (
            None
            if complete
            else ".\\.tools\\uv.exe run --no-sync python "
            f"scripts\\ingest_player_arrests.py --snapshot {snapshot_dir.name}"
        ),
    }
    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "data/raw/player_arrests")
    parser.add_argument("--snapshot", type=str, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS)
    args = parser.parse_args()
    if args.max_pages is not None and args.max_pages < 1:
        parser.error("--max-pages must be >= 1")
    if args.delay_seconds < 0:
        parser.error("--delay-seconds must be >= 0")

    snapshot_dir = new_snapshot_dir(args.out, args.snapshot)
    print(f"Snapshot dir: {snapshot_dir}")
    try:
        manifest = ingest(snapshot_dir, max_pages=args.max_pages, delay_seconds=args.delay_seconds)
    except PlayerArrestsIngestError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
