"""Snapshot-delivery pilot for Big Ten Football Availability Reports (XLG-07 unblocking).

Fetches the 2023 availability hub page on bigten.org, enumerates the weekly
Football Availability Report PDF links it lists, and downloads up to
--max-pdfs weekly PDFs with a polite delay between requests. Every artifact is
hashed (SHA-256) into a manifest under data/raw/bigten_availability/<run-id>/.

Text parsing is attempted ONLY if a PDF text-extraction library (pdfminer or
pypdf) is already installed in the environment; the pilot deliberately does not
add dependencies. With no parser present this script stops cleanly at snapshot
delivery and records that outcome in artifacts/bigten_pilot/<run-id>/.

No tidy rows, no XLG-03 join, no ATS evaluation, no registry writes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

HUB_URL_2023 = "https://bigten.org/fb/article/blt19caa3aea8cf4525/"
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
MIN_DELAY_SECONDS = 2.0
WEEK_LINK_PATTERN = re.compile(r"FB_Reporting_Week_(\d+)", re.IGNORECASE)
PDF_HREF_PATTERN = re.compile(r'href="([^"]+\.pdf[^"]*)"', re.IGNORECASE)
PARSER_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("pdfminer", "pdfminer.high_level"),
    ("pypdf", "pypdf"),
)

REPO = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO / "data" / "raw" / "bigten_availability"
ARTIFACT_ROOT = REPO / "artifacts" / "bigten_pilot"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_pdf_parser() -> str | None:
    for name, module in PARSER_CANDIDATES:
        try:
            if importlib.util.find_spec(module) is not None:
                return name
        except ModuleNotFoundError:
            continue
    return None


def fetch_hub(hub_url: str, session: requests.Session) -> str:
    response = session.get(hub_url, timeout=60)
    response.raise_for_status()
    return response.text


def enumerate_weekly_pdf_links(hub_html: str, hub_url: str) -> list[dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    for href in PDF_HREF_PATTERN.findall(hub_html):
        match = WEEK_LINK_PATTERN.search(href)
        if match is None:
            continue
        week = int(match.group(1))
        absolute = urljoin(hub_url, href)
        if week not in found:
            found[week] = {"week": week, "url": absolute}
    return [found[week] for week in sorted(found)]


def download_pdfs(
    links: list[dict[str, Any]],
    out_dir: Path,
    session: requests.Session,
    max_pdfs: int,
    delay_seconds: float,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    effective_delay = max(delay_seconds, MIN_DELAY_SECONDS)
    for index, link in enumerate(links[:max_pdfs]):
        week = link["week"]
        filename = f"2023_week_{week:02d}.pdf"
        target = out_dir / filename
        if index > 0 or entries:
            time.sleep(effective_delay)
        response = session.get(link["url"], timeout=120)
        response.raise_for_status()
        target.write_bytes(response.content)
        size_bytes = target.stat().st_size
        header = response.content[:5]
        entries.append(
            {
                "week": week,
                "source_url": link["url"],
                "filename": filename,
                "sha256": sha256_file(target),
                "size_bytes": size_bytes,
                "is_pdf_header": header == b"%PDF-",
                "fetched_at_utc": utc_now_iso(),
            }
        )
        print(f"downloaded week {week}: {size_bytes} bytes")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-url", default=HUB_URL_2023)
    parser.add_argument("--out-dir", type=Path, default=RAW_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--max-pdfs", type=int, default=6)
    parser.add_argument("--delay-seconds", type=float, default=2.5)
    args = parser.parse_args()

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = args.out_dir / run_id
    artifact_dir = args.artifact_dir / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    status: dict[str, Any] = {
        "run_id": run_id,
        "hub_url": args.hub_url,
        "outcome": "snapshot_only",
        "parser_available": None,
        "weekly_pdf_links_found": 0,
        "pdfs_downloaded": 0,
        "started_at_utc": utc_now_iso(),
    }

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    hub_html = fetch_hub(args.hub_url, session)
    hub_path = raw_dir / "hub_2023.html"
    hub_path.write_text(hub_html, encoding="utf-8")

    links = enumerate_weekly_pdf_links(hub_html, args.hub_url)
    status["weekly_pdf_links_found"] = len(links)
    status["weekly_pdf_links"] = [{"week": entry["week"], "url": entry["url"]} for entry in links]
    print(f"hub fetched ({len(hub_html)} chars); weekly PDF links: {len(links)}")

    if not links:
        status["outcome"] = "no_pdf_links_found"
        status["finished_at_utc"] = utc_now_iso()
        (artifact_dir / "pilot_status.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        print("no weekly PDF links enumerated; stopping", file=sys.stderr)
        return 1

    manifest_entries = download_pdfs(
        links,
        raw_dir,
        session,
        max_pdfs=args.max_pdfs,
        delay_seconds=args.delay_seconds,
    )
    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "hub_url": args.hub_url,
                "created_at_utc": utc_now_iso(),
                "entries": manifest_entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    status["pdfs_downloaded"] = len(manifest_entries)
    status["manifest_path"] = str(manifest_path.relative_to(REPO))

    parser_name = detect_pdf_parser()
    status["parser_available"] = parser_name
    if parser_name is None:
        status["outcome"] = "snapshot_only_no_pdf_parser_installed"
        status["note"] = (
            "No PDF text-extraction library (pdfminer/pypdf) is installed. "
            "Per pilot scope, parsing was NOT attempted and no dependency was "
            "added. Snapshot delivery only."
        )
        print(
            "no pdfminer/pypdf installed; stopping at snapshot delivery (no dependency added)",
            file=sys.stderr,
        )
    else:
        status["outcome"] = "parser_present_parse_step_not_implemented"

    status["finished_at_utc"] = utc_now_iso()
    status["provenance"] = artifact_provenance(
        {
            "source": "bigten.org 2023 Football Availability Reports hub",
            "run_id": run_id,
            "hub_url": args.hub_url,
            "max_pdfs": args.max_pdfs,
            "delay_seconds": max(args.delay_seconds, MIN_DELAY_SECONDS),
        },
        manifest_path if manifest_path.exists() else hub_path,
        project_root=REPO,
    )
    write_experiment_artifact(
        artifact_dir,
        "pilot_status.json",
        status,
        command="pilot_bigten_availability",
        metrics={
            "weekly_pdf_links_found": status["weekly_pdf_links_found"],
            "pdfs_downloaded": status["pdfs_downloaded"],
            "parser_available_present": bool(parser_name),
        },
        notes=(
            "Ingest snapshot pilot only: no parsing (no PDF parser installed, "
            "no dependency added), no tidy rows, no XLG-03 join, no ATS "
            "evaluation, and no tracked-registry write (registry_root "
            "redirected inside the gitignored artifact snapshot)."
        ),
        source="scripts/pilot_bigten_availability.py",
        registry_root=artifact_dir / "experiment_registry",
        project_root=REPO,
    )
    print(f"status written to {artifact_dir / 'pilot_status.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
