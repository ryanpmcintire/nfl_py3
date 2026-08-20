"""Ingest a second, independent public-attention source -- GDELT DOC 2.0 --
to cross-validate the Wikipedia-pageview attention construct used by
``scripts/attention_battery_screen.py`` / ``scripts/attention_followup_screen.py``.

**Measured** (this session, `docs/data_source_scout_v2.md` GDELT section plus
live probes below): GDELT DOC 2.0 (`https://api.gdeltproject.org/api/v2/doc/doc`)
is free, keyless, and returns real per-article timestamps (`seendate`) --
provably point-in-time-safe -- but a bare team-name query pulls entertainment
crossover noise (a Kelce/Swift Hollywood-gossip hit, a Hallmark-movie story,
etc., confirmed live this session). This script filters to a fixed allowlist
of sports/news domains (``DOMAIN_ALLOWLIST`` below) and uses
``mode=timelinevol`` (normalized daily share-of-monitored-coverage matching
the query, NOT raw article count) as the attention-volume signal -- one
GDELT-recommended way to get a long time series in very few requests instead
of paging ``artlist`` week by week.

**Rate limiting (measured, this session):** GDELT's stated limit is "one
request every 5 seconds"; in practice this session saw the literal string
"Please limit requests..." (not JSON) returned intermittently even a full
minute apart -- consistent with a budget shared across this session's
concurrent agents on one egress IP, not just this script's own pacing. This
script enforces a minimum ``RATE_LIMIT_SECONDS`` gap and retries with
exponential backoff (measured necessary; a single 5-second gap was NOT always
sufflicient this session).

**Historical coverage**: probe this script's ``--probe`` mode before a full
run -- do not assume the "2017+" figure from the scout doc without checking;
report what is actually returned for each candidate start year.

Output: raw JSON per (team, name-alias, date-range) chunk under
``data/raw/gdelt/<UTC timestamp>/*.json`` (gitignored -- data/raw is never
committed) plus a ``manifest.json`` recording every request made (url,
http status, byte count, retry count, timestamp).

Derivation into attention z-scores and cross-source correlation against the
Wikipedia construct happens in ``scripts/gdelt_attention_screen.py``, kept
separate so a partial/interrupted ingest run still leaves usable raw data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _load_parent_module():
    spec = importlib.util.spec_from_file_location(
        "attention_battery_screen", REPO / "scripts" / "attention_battery_screen.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_parent_module()  # for TEAM_ARTICLES (team -> alias article slugs)

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

# Sports/news domain allowlist -- keeps the query on football coverage and
# away from entertainment-gossip crossover (Kelce/Swift, Hallmark movies,
# etc., confirmed noisy on a bare team-name query this session).
DOMAIN_ALLOWLIST = [
    "espn.com",
    "nfl.com",
    "cbssports.com",
]

RATE_LIMIT_SECONDS = 6.0
MAX_RETRIES = 6
INITIAL_BACKOFF = 8.0

_last_request_ts: float = 0.0


def _polite_get(url: str) -> tuple[int, str, int]:
    """GET url, enforcing a minimum inter-request gap and exponential backoff
    on GDELT's rate-limit text response (which is HTTP 200 but not JSON).
    Returns (http_status, body_text, retries_used)."""

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
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = resp.status
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = f"HTTPError {exc.code}: {exc.reason}"
        except urllib.error.URLError as exc:
            body = f"URLError: {exc}"
            status = 0

        looks_rate_limited = (
            status == 429 or status == 0 or "Please limit requests" in body or not body.strip()
        )
        if not looks_rate_limited or retries >= MAX_RETRIES:
            return status, body, retries
        retries += 1
        time.sleep(backoff)
        backoff = min(backoff * 1.7, 90.0)


def build_query(name_phrase: str) -> str:
    domain_clause = " OR ".join(f"domainis:{d}" for d in DOMAIN_ALLOWLIST)
    return f'"{name_phrase}" ({domain_clause})'


def fetch_timelinevol(
    name_phrase: str, start: str, end: str, *, mode: str = "timelinevol"
) -> dict[str, Any]:
    query = build_query(name_phrase)
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "startdatetime": start,
        "enddatetime": end,
    }
    url = f"{GDELT_ENDPOINT}?{urllib.parse.urlencode(params)}"
    status, body, retries = _polite_get(url)
    parsed: Any = None
    parse_error = None
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        parse_error = str(exc)
    return {
        "url": url,
        "query": query,
        "name_phrase": name_phrase,
        "start": start,
        "end": end,
        "http_status": status,
        "retries": retries,
        "response_bytes": len(body.encode("utf-8")),
        "parsed_ok": parse_error is None,
        "parse_error": parse_error,
        "body": body,
        "parsed": parsed,
    }


# --------------------------------------------------------------------------
# Probe mode -- verify query shape and historical coverage before a full run
# --------------------------------------------------------------------------


def run_probe() -> None:
    print("=== probe 1: domain-filtered artlist, noise check ===")
    r = fetch_timelinevol("Kansas City Chiefs", "20240901000000", "20240910000000", mode="artlist")
    print(f"  status={r['http_status']} retries={r['retries']} parsed_ok={r['parsed_ok']}")
    if r["parsed_ok"]:
        arts = r["parsed"].get("articles", [])
        print(f"  n_articles={len(arts)}")
        for a in arts[:8]:
            print(f"    {a.get('seendate')} {a.get('domain')} {a.get('title', '')[:70]}")
    else:
        print(f"  body[:300]={r['body'][:300]!r}")

    print("\n=== probe 2: timelinevol, one season ===")
    r = fetch_timelinevol("Kansas City Chiefs", "20180901000000", "20190215000000")
    print(f"  status={r['http_status']} retries={r['retries']} parsed_ok={r['parsed_ok']}")
    if r["parsed_ok"]:
        print(f"  top-level keys: {list(r['parsed'].keys())}")
        print(json.dumps(r["parsed"], indent=2)[:2000])
    else:
        print(f"  body[:300]={r['body'][:300]!r}")

    print("\n=== probe 3: how far back does coverage go? ===")
    for start_year in (2013, 2015, 2017, 2019):
        r = fetch_timelinevol(
            "Kansas City Chiefs", f"{start_year}0101000000", f"{start_year}0201000000"
        )
        n_points = None
        if r["parsed_ok"]:
            tl = r["parsed"].get("timeline", [])
            if tl and isinstance(tl, list):
                series0 = tl[0].get("data", [])
                n_points = len(series0)
        print(
            f"  start={start_year}: status={r['http_status']} parsed_ok={r['parsed_ok']} "
            f"n_points={n_points} retries={r['retries']}"
        )
        if r["parsed_ok"] and n_points:
            print(f"    first 3 points: {tl[0]['data'][:3]}")
        elif not r["parsed_ok"]:
            print(f"    body[:300]={r['body'][:300]!r}")

    print("\n=== probe 4: multi-year single-call range ===")
    r = fetch_timelinevol("Kansas City Chiefs", "20150101000000", "20260101000000")
    print(f"  status={r['http_status']} retries={r['retries']} parsed_ok={r['parsed_ok']}")
    if r["parsed_ok"]:
        tl = r["parsed"].get("timeline", [])
        if tl:
            data = tl[0].get("data", [])
            print(f"  n_points={len(data)}")
            if data:
                print(f"  first={data[0]} last={data[-1]}")
    else:
        print(f"  body[:500]={r['body'][:500]!r}")


# --------------------------------------------------------------------------
# Full ingestion
# --------------------------------------------------------------------------


def team_name_phrases() -> dict[str, list[str]]:
    return {
        team: [a.replace("_", " ") for a in articles]
        for team, articles in base.TEAM_ARTICLES.items()
    }


def run_ingest(
    start_year: int, end_year: int, output_dir: Path, chunk_years: int, resume: bool = False
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    already_done: set[str] = set()
    if resume and manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = prior
        already_done = {r["file"] for r in prior.get("requests", []) if r.get("parsed_ok")}
        print(f"resuming: {len(already_done)} chunks already parsed_ok, skipping those")
    else:
        manifest = {
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "start_year": start_year,
            "end_year": end_year,
            "chunk_years": chunk_years,
            "domain_allowlist": DOMAIN_ALLOWLIST,
            "endpoint": GDELT_ENDPOINT,
            "mode": "timelinevol",
            "requests": [],
        }

    phrases = team_name_phrases()
    n_teams = len(phrases)
    req_i = 0
    total_reqs = sum(len(v) for v in phrases.values()) * len(
        range(start_year, end_year + 1, chunk_years)
    )
    for team, aliases in phrases.items():
        for alias in aliases:
            for chunk_start in range(start_year, end_year + 1, chunk_years):
                chunk_end = min(chunk_start + chunk_years, end_year + 1)
                start = f"{chunk_start}0101000000"
                end = f"{chunk_end}0101000000"
                req_i += 1
                safe_alias = alias.replace(" ", "_")
                fname = f"{team}__{safe_alias}__{chunk_start}_{chunk_end - 1}.json"
                if fname in already_done:
                    print(f"[{req_i}/{total_reqs}] {team} / {alias!r}: already done, skipping")
                    continue
                print(
                    f"[{req_i}/{total_reqs}] {team} / {alias!r} {chunk_start}-{chunk_end - 1}",
                    flush=True,
                )
                result = fetch_timelinevol(alias, start, end)
                (output_dir / fname).write_text(
                    json.dumps(result["parsed"] if result["parsed_ok"] else None), encoding="utf-8"
                )
                manifest["requests"].append(
                    {
                        "team": team,
                        "alias": alias,
                        "chunk_start_year": chunk_start,
                        "chunk_end_year": chunk_end - 1,
                        "file": fname,
                        "url": result["url"],
                        "http_status": result["http_status"],
                        "retries": result["retries"],
                        "response_bytes": result["response_bytes"],
                        "parsed_ok": result["parsed_ok"],
                        "parse_error": result["parse_error"],
                        "body_on_failure": None if result["parsed_ok"] else result["body"][:500],
                    }
                )
                if not result["parsed_ok"]:
                    print(f"    WARNING: parse failed, retries={result['retries']}", flush=True)

                # Write the manifest after every request (not just at the end)
                # so a timeout/interruption mid-run still leaves a usable,
                # resumable record of what succeeded.
                manifest["n_requests_so_far"] = len(manifest["requests"])
                (output_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )

    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["n_teams"] = n_teams
    manifest["n_requests"] = len(manifest["requests"])
    manifest["n_parse_failures"] = sum(1 for r in manifest["requests"] if not r["parsed_ok"])
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {output_dir / 'manifest.json'}")
    print(f"n_requests={manifest['n_requests']} n_parse_failures={manifest['n_parse_failures']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    probe = sub.add_parser("probe", help="run a handful of verification queries, no data saved")
    probe.set_defaults(func=lambda args: run_probe())

    ingest = sub.add_parser("ingest", help="full ingestion run")
    ingest.add_argument("--start-year", type=int, default=2016)
    ingest.add_argument("--end-year", type=int, default=2025)
    ingest.add_argument("--chunk-years", type=int, default=10, help="years per API call")
    ingest.add_argument("--output", type=Path, default=None)
    ingest.add_argument(
        "--resume", action="store_true", help="skip chunks already parsed_ok in --output's manifest"
    )
    ingest.set_defaults(
        func=lambda args: run_ingest(
            args.start_year,
            args.end_year,
            args.output
            or (REPO / "data" / "raw" / "gdelt" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())),
            args.chunk_years,
            args.resume,
        )
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
