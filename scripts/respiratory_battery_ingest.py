"""Bulk-ingest CDC Delphi NSSP (National Syndromic Surveillance Program)
state-level emergency-department (ED) visit percentage history for
COVID-19, influenza, and RSV -- the same 23 NFL-team states as
``scripts/fluview_battery_ingest.py`` (``STATE_BY_TEAM`` imported from that
module, not redefined), extending the project's highest-reliability signal
(FluView state ILI, split-half reliability 0.9814 -- see the FluView
battery's own results, ``docs/fluview_battery.md``) from influenza alone to
total respiratory illness burden.

This script only ever writes to ``data/raw/respiratory/`` (a gitignored raw
snapshot + manifest, the same convention ``fluview_battery_ingest.py`` uses)
-- the provenance-stamped, SCORED results directory belongs entirely to
``scripts/respiratory_battery_screen.py`` (written via
``nfl_ats.provenance.write_experiment_artifact``), not this ingest.

**Source-availability finding, measured this session** (frozen in
``docs/respiratory_battery.md`` section 1, read that first): the Delphi
``covidcast`` endpoint (``https://api.delphi.cmu.edu/epidata/covidcast/``)
exposes real per-time_value revision history for NSSP ED-visit signals via
an ``issues=<range>`` parameter, and unlike FluView's ``issues`` (exact-match
only), covidcast's range form returns every distinct issue in the window
directly in one call. Measured: ``ca`` ``pct_ed_visits_influenza`` epiweek
202301 (an old, fully-backfilled week) carries 121 distinct issues on file;
a THEN-recent epiweek (202510, early 2025) already had its first issue at
``lag=1`` -- one week after the fact -- with the value itself still revising
several weeks later (1.75 -> 1.85 over subsequent reissues). This is denser,
more-real-time revision behavior than FluView's own post-2018 floor.

**Critical scope-limiting finding**: NSSP state-level coverage begins at
epiweek **202239** (late Sept 2022) -- confirmed via the ``covidcast_meta``
endpoint (``min_time: 202239`` for every NSSP state-level signal) and
cross-checked with a live per-state query (51 of 51 state+DC locations
already reporting at 202239, so the floor is a hard start, not a slow
ramp-up). This is a MUCH narrower window than FluView's 2010-2025 raw range
(even after FluView's own ~2017-18 point-in-time-recoverable floor): NSSP
can score at most NFL seasons 2022 onward, roughly 4-5 seasons vs FluView's
~8 effective seasons.

**The official ``pct_ed_visits_combined`` signal is DEAD as of this
ingest**: ``covidcast_meta`` shows it stopped updating at epiweek 202439
(state-level ``max_time: 202439``, ``last_update`` timestamp June 2025 with
no further revisions since), while the three per-pathogen raw signals
(``pct_ed_visits_covid`` / ``_influenza`` / ``_rsv``) continue updating
through the current epiweek. This ingest therefore pulls the three raw
per-pathogen signals separately and the screen script
(``scripts/respiratory_battery_screen.py``) sums them itself into a
``respiratory_total`` AS-OF value (all three required non-missing at a
cutoff, never a partial sum) -- see ``docs/respiratory_battery.md`` section
3 for the exact construction.

**Anonymous rate limit is stricter here than FluView's ingest needed**:
covidcast's anonymous cap of "only two parameters may have multiple
selections" is measured (HTTP 401 "Requested too many multiples for
anonymous queries") to reject any THIRD multi-selection parameter. Getting
one state's full point-in-time history in one call already spends both
slots on ``time_values=<range>`` + ``issues=<range>``, leaving no room to
also request multiple ``signal`` values or multiple ``geo_value`` values in
the same call. So this ingest needs one request per (state, signal) pair:
23 states x 3 signals = 69 requests, which on its own EXCEEDS the anonymous
60-requests/hour cap that FluView's 24-request ingest fit under trivially.
Rather than lean on 429 backoff to recover from a blown quota (which could
need up to an hour of waiting per stall), this script paces every request
at a fixed ``RATE_LIMIT_SECONDS`` interval chosen so sustained throughput
stays under 60/hour by construction (roughly one request per 62 seconds,
~58/hour) -- the run should never trip the limiter in the first place.
Exponential backoff on 429 is kept as a safety net only, unchanged from
``scripts/fluview_battery_ingest.py``'s implementation (reused verbatim).

Endpoint: ``https://api.delphi.cmu.edu/epidata/covidcast/`` -- free, no key.

Output: ``data/raw/respiratory/<UTC timestamp>/respiratory_raw.parquet``
(one row per state x signal x time_value x issue -- the full revision
history) plus ``manifest.json`` recording every request made (url, http
status, byte count, row count, retry count, elapsed seconds). Gitignored,
per repo convention (``data/raw`` is never committed).

**Resumable / incrementally checkpointed** (added after this session's first
attempt was killed mid-run by something outside this script -- not a 429,
not an exception -- with NOTHING written, because the original version only
wrote its output after all 69 requests finished): each completed
``(state, signal)`` response is now written immediately to its own file
under ``<output_dir>/parts/`` (atomic write-then-rename, so a part file is
never left half-written), and the aggregate ``respiratory_raw.parquet`` +
``manifest.json`` are REBUILT from whatever parts exist after every single
request, not just at the end. Re-running with ``--output <same_dir>
--resume`` skips every ``(state, signal)`` pair that already has a part
file and only fetches what's missing -- an interruption at request 40 of 69
now costs at most one request's worth of pacing to recover from, not the
full ~70-minute run.
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
sys.path.append(str(REPO / "scripts"))

from fluview_battery_ingest import STATE_BY_TEAM  # noqa: E402

DELPHI_ENDPOINT = "https://api.delphi.cmu.edu/epidata/covidcast/"
DATA_SOURCE = "nssp"

# The three raw (non-smoothed) per-pathogen NSSP ED-visit-percentage signals
# that are STILL updating as of this ingest (docs/respiratory_battery.md
# section 1) -- the official "combined" signal stopped updating 2024w39 and
# is deliberately not fetched; the screen script sums these three itself.
SIGNALS: list[str] = [
    "pct_ed_visits_covid",
    "pct_ed_visits_influenza",
    "pct_ed_visits_rsv",
]

STATES = sorted(set(STATE_BY_TEAM.values()))

TIME_VALUES_LOW = "202239"  # measured NSSP state-level floor
TIME_VALUES_HIGH = "202708"  # safety margin past the 2026 season's last week
ISSUES_LOW = "202239"
ISSUES_HIGH = "202720"  # safety margin for late revisions past TIME_VALUES_HIGH

# Paced to stay under the anonymous 60-requests/hour cap by construction
# (69 requests this ingest makes, on its own, exceeds 60) rather than
# relying on 429 backoff to recover from a blown quota.
RATE_LIMIT_SECONDS = 62.0
MAX_RETRIES = 6
INITIAL_BACKOFF = 10.0

_last_request_ts = 0.0


def _polite_get(url: str) -> tuple[int, bytes, int]:
    """GET url, enforcing a minimum inter-request gap and exponential
    backoff on HTTP 429 as a safety net (unchanged from
    ``scripts/fluview_battery_ingest.py``). Returns
    (status, body_bytes, retries_used)."""

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


def fetch_state_signal(state: str, signal: str) -> dict[str, Any]:
    url = (
        f"{DELPHI_ENDPOINT}?data_source={DATA_SOURCE}&signal={signal}"
        f"&time_type=week&geo_type=state&geo_value={state}"
        f"&time_values={TIME_VALUES_LOW}-{TIME_VALUES_HIGH}"
        f"&issues={ISSUES_LOW}-{ISSUES_HIGH}"
    )
    t0 = time.time()
    status, body, retries = _polite_get(url)
    elapsed = time.time() - t0
    record: dict[str, Any] = {
        "state": state,
        "signal": signal,
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


def _job_key(state: str, signal: str) -> str:
    return f"{state}__{signal}"


def _parts_dir(output_dir: Path) -> Path:
    return output_dir / "parts"


def _load_completed_parts(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Every ``(state, signal)`` job that already has a complete part file
    on disk -- used both to resume (skip already-done jobs) and to rebuild
    the aggregate output after each new request."""

    parts_dir = _parts_dir(output_dir)
    completed: dict[str, dict[str, Any]] = {}
    if not parts_dir.is_dir():
        return completed
    for path in sorted(parts_dir.glob("*.json")):
        try:
            completed[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # a part that never finished writing cleanly -- refetch it
    return completed


def _write_part(output_dir: Path, state: str, signal: str, result: dict[str, Any]) -> None:
    """Atomic write-then-rename so a kill mid-write never leaves a corrupt
    part file that ``_load_completed_parts`` would then have to discard."""

    parts_dir = _parts_dir(output_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    key = _job_key(state, signal)
    final_path = parts_dir / f"{key}.json"
    tmp_path = parts_dir / f"{key}.json.tmp"
    tmp_path.write_text(json.dumps(result), encoding="utf-8")
    tmp_path.replace(final_path)


def _assemble_output(output_dir: Path, completed: dict[str, dict[str, Any]]) -> None:
    """Rebuild ``respiratory_raw.parquet`` + ``manifest.json`` from every
    completed part on file. Called after EVERY request (not just at the
    end) so an interruption at any point leaves a valid, immediately usable
    partial snapshot -- the reason this ingest survives being killed."""

    manifest_entries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for key in sorted(completed):
        result = completed[key]
        entry = result["manifest_entry"]
        manifest_entries.append(entry)
        for row in result["rows"]:
            row = dict(row)
            row["region"] = entry["state"]
            row["pathogen_signal"] = entry["signal"]
            all_rows.append(row)

    out_path = output_dir / "respiratory_raw.parquet"
    manifest_path = output_dir / "manifest.json"

    if not all_rows:
        return  # nothing completed yet -- leave no (half-formed) output file

    df = pd.DataFrame(all_rows)
    keep_cols = [
        "region",
        "pathogen_signal",
        "time_value",
        "issue",
        "lag",
        "value",
        "missing_value",
    ]
    df = df.loc[:, [c for c in keep_cols if c in df.columns]]
    df["time_value"] = df["time_value"].astype("int64")
    df["issue"] = df["issue"].astype("int64")
    df["lag"] = df["lag"].astype("int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    tmp_out_path = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_out_path, index=False)
    tmp_out_path.replace(out_path)

    per_state_signal_counts = df.groupby(["region", "pathogen_signal"]).size()
    jobs = [(state, signal) for state in STATES for signal in SIGNALS]
    manifest = {
        "source": DELPHI_ENDPOINT,
        "data_source": DATA_SOURCE,
        "signals": SIGNALS,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "time_values_range": [TIME_VALUES_LOW, TIME_VALUES_HIGH],
        "issues_range": [ISSUES_LOW, ISSUES_HIGH],
        "states_requested": STATES,
        "state_by_team": STATE_BY_TEAM,
        "requests": manifest_entries,
        "n_jobs_total": len(jobs),
        "n_jobs_completed": len(completed),
        "complete": len(completed) == len(jobs),
        "n_rows_total": len(df),
        "n_rows_per_state_signal": {
            f"{state}/{signal}": int(count)
            for (state, signal), count in per_state_signal_counts.items()
        },
        "output_parquet": str(out_path.relative_to(REPO)),
    }
    tmp_manifest_path = manifest_path.with_suffix(".json.tmp")
    tmp_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_manifest_path.replace(manifest_path)


def run_ingest(output_dir: Path, *, resume: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_parts(output_dir) if resume else {}
    if completed:
        print(f"resuming {output_dir}: {len(completed)} job(s) already complete")

    jobs = [(state, signal) for state in STATES for signal in SIGNALS]
    remaining = [(s, sig) for s, sig in jobs if _job_key(s, sig) not in completed]
    print(f"{len(completed)}/{len(jobs)} done, {len(remaining)} remaining")

    for i, (state, signal) in enumerate(remaining, start=1):
        print(f"[{i}/{len(remaining)}] fetching {state}/{signal} ...")
        result = fetch_state_signal(state, signal)
        entry = result["manifest_entry"]
        print(
            f"  status={entry['http_status']} rows={entry['n_rows']} "
            f"bytes={entry['byte_count']} retries={entry['retries']} "
            f"elapsed={entry['elapsed_seconds']:.2f}s"
        )
        _write_part(output_dir, state, signal, result)
        completed[_job_key(state, signal)] = result
        _assemble_output(output_dir, completed)  # checkpoint after EVERY request

    if not completed:
        raise SystemExit(
            "no rows fetched from any (state, signal) pair -- aborting, not writing an "
            "empty snapshot"
        )

    out_path = output_dir / "respiratory_raw.parquet"
    manifest_path = output_dir / "manifest.json"
    print(f"\nwrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"wrote {manifest_path}")
    failed = [
        f"{r['manifest_entry']['state']}/{r['manifest_entry']['signal']}"
        for r in completed.values()
        if not r["manifest_entry"].get("parsed_ok")
    ]
    if failed:
        print(
            f"WARNING: {len(failed)} (state, signal) pair(s) failed to fetch: {failed}",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (state, signal) pairs that already have a part file under --output/parts/.",
    )
    args = parser.parse_args()
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else REPO / "data" / "raw" / "respiratory" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    run_ingest(output_dir, resume=args.resume)


if __name__ == "__main__":
    main()
