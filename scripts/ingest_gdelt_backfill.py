"""GDELT per-team backfill (backlog item): full 32-franchise, 2017-2025 daily
news-attention archive, built to be comparable/complementary to the
Wikipedia-pageviews attention work (``scripts/attention_battery_screen.py``,
``docs/attention_followup.md``).

Scope for this script: INGESTION ONLY. No experiments, no registry writes, no
``src/nfl_ats`` changes -- aggregation into per-(season,week,team) features
happens in ``scripts/build_gdelt_weekly_features.py``, and the coverage
write-up lives in ``docs/gdelt_backfill.md``.

Relationship to prior GDELT work in this repo (read this session,
2026-08-20): ``scripts/ingest_gdelt_attention.py`` already established (a)
GDELT DOC 2.0's ``timelinevol``/``timelinevolraw`` modes return TRUE DAILY
granularity even for a single call spanning the full 2017-01-01..2026-01-01
range (**measured**, this session: a re-parse of its
``data/raw/gdelt/20260819_pilot/ARI__Arizona_Cardinals__2017_2025.json``
showed 3,267 distinct daily points, ``date_resolution: "day"``), and (b) a
narrow 3-domain allowlist (espn.com, nfl.com, cbssports.com) works but is
noisy at the day level -- most days return 0-3 raw articles per team, which
is a lot of relative noise for a single-day sanity check. That pilot run
covered only 9 of 32 teams before stopping (DAL hit ``MAX_RETRIES`` on
persistent 429s; **measured**, read from
``data/raw/gdelt/20260819_pilot/manifest.json``) and used ``mode=timelinevol``
(a normalized share metric), not raw article counts.

This script supersedes that pilot for the "per-team backfill" backlog item:

1. **One HTTP request per (team-alias, mode)** for the WHOLE 2017-2025 range
   -- not per-team-year -- because the finding above means per-year chunking
   buys nothing: daily granularity is already preserved in a single call.
   This is a deliberate deviation from the task brief's suggested
   "checkpoint per team-year" mechanism; the *goal* that suggestion serves
   (resumability after an interruption) is achieved instead by checkpointing
   after every (team-alias, mode) request, of which there are far fewer
   (37, one per alias including relocation predecessor names, x 2 modes =
   74 total requests for all 32 teams) than a per-team-year design would
   need (32 teams x 9 years x 2 modes = 576).
2. **Broadened domain allowlist** (``DOMAIN_ALLOWLIST``, 8 sports-only
   outlets vs. the pilot's 3) to reduce single-day zero-inflation while
   staying off entertainment-crossover domains (the scout doc's measured
   Kelce/Swift/Hallmark-movie noise problem on an unfiltered query). This is
   an **inferred** editorial judgment call (all 8 are sports-only outlets by
   reputation), not independently re-verified per-domain this session --
   flagged as such in ``docs/gdelt_backfill.md``.
3. **Both volume and tone**, via two GDELT modes (``timelinevolraw`` for raw
   daily article count + total-monitored-corpus size; ``timelinetone`` for
   average daily tone). These are NOT available from one call (**measured**:
   distinct ``mode=`` params, distinct response shapes) so "tone if cheap
   from the same call" resolves to "not free, but affordable" once (1) above
   cuts total request count to 74 -- both modes fit the same one-call-per-
   alias design.
4. **Team query strings / relocation handling**: reuses
   ``attention_battery_screen.TEAM_ARTICLES`` (imported, not copied) as the
   32-franchise alias table, because it already encodes exactly the
   relocations named in the task brief (OAK->LV as
   ``Oakland_Raiders``/``Las_Vegas_Raiders``, SD->LAC as
   ``San_Diego_Chargers``/``Los_Angeles_Chargers``, STL->LA as
   ``St._Louis_Rams``/``Los_Angeles_Rams``) plus the WAS three-name history
   (``Washington_Redskins``/``Washington_Football_Team``/
   ``Washington_Commanders``) not explicitly called out in the brief but the
   same category of problem. Reusing it (rather than a hand-rolled second
   table) guarantees this archive's team identity resolution is bit-for-bit
   consistent with the Wikipedia construct it is meant to complement.

Output: raw JSON per (team, alias, mode) under
``data/raw/gdelt/<UTC timestamp>/*.json`` plus ``manifest.json`` (gitignored,
matching this repo's existing raw-source convention --
``scripts/ingest_injury_news.py``'s module docstring has the precedent this
script follows: a manifest.json lives one level under ``data/raw/gdelt/``,
never directly at ``data/raw/gdelt/manifest.json``).
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


def _load_team_articles() -> dict[str, list[str]]:
    spec = importlib.util.spec_from_file_location(
        "attention_battery_screen", REPO / "scripts" / "attention_battery_screen.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TEAM_ARTICLES


TEAM_ARTICLES: dict[str, list[str]] = _load_team_articles()

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

# Broadened vs. the 20260819 pilot's 3-domain list (espn.com, nfl.com,
# cbssports.com). All 8 are sports-only outlets by reputation (inferred, not
# independently re-verified per-domain this session) -- chosen to cut
# single-day zero-inflation in the raw daily article count while staying off
# the entertainment-crossover domains the scout doc measured as noisy on an
# unfiltered team-name query.
DOMAIN_ALLOWLIST = [
    "espn.com",
    "nfl.com",
    "cbssports.com",
    "nbcsports.com",
    "foxsports.com",
    "si.com",
    "bleacherreport.com",
    "sportingnews.com",
]

# GDELT DOC 2.0's documented reliable floor is 2017-01-01. End date pushed to
# mid-February to safely cover the 2025 season's full REG Week 18 slate
# (early January 2026) plus a trailing buffer for as-of-Saturday cutoffs on
# the final week.
DEFAULT_START = "20170101000000"
DEFAULT_END = "20260215000000"

RATE_LIMIT_SECONDS = 6.0
MAX_RETRIES = 8
INITIAL_BACKOFF = 8.0
BACKOFF_CAP = 90.0

MODES = ["timelinevolraw", "timelinetone"]

_last_request_ts: float = 0.0


def _polite_get(url: str) -> tuple[int, str, int]:
    """GET url, enforcing a minimum inter-request gap and exponential backoff
    on GDELT's rate-limit text response (HTTP 200 but non-JSON, or a real
    429). Returns (http_status, body_text, retries_used)."""

    global _last_request_ts
    backoff = INITIAL_BACKOFF
    retries = 0
    while True:
        elapsed = time.time() - _last_request_ts
        if elapsed < RATE_LIMIT_SECONDS:
            time.sleep(RATE_LIMIT_SECONDS - elapsed)
        _last_request_ts = time.time()
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "nfl-ats-research/1.0 (gdelt-backfill)"}
            )
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
        except TimeoutError as exc:
            body = f"TimeoutError: {exc}"
            status = 0

        looks_rate_limited = (
            status == 429 or status == 0 or "Please limit requests" in body or not body.strip()
        )
        if not looks_rate_limited or retries >= MAX_RETRIES:
            return status, body, retries
        retries += 1
        time.sleep(backoff)
        backoff = min(backoff * 1.7, BACKOFF_CAP)


def build_query(name_phrase: str) -> str:
    domain_clause = " OR ".join(f"domainis:{d}" for d in DOMAIN_ALLOWLIST)
    return f'"{name_phrase}" ({domain_clause})'


def fetch(
    name_phrase: str, mode: str, start: str = DEFAULT_START, end: str = DEFAULT_END
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
        "mode": mode,
        "start": start,
        "end": end,
        "http_status": status,
        "retries": retries,
        "response_bytes": len(body.encode("utf-8")),
        "parsed_ok": parse_error is None and parsed is not None,
        "parse_error": parse_error,
        "body_on_failure": None if parse_error is None else body[:500],
        "parsed": parsed,
    }


def _work_items() -> list[tuple[str, str, str]]:
    """[(team, alias, mode), ...] -- mode outer-looped across ALL teams first
    (volume, then tone) so a time-boxed run that stops early still has full
    32-team coverage for whichever modes it completed, rather than finishing
    a handful of teams in both modes and leaving the rest untouched."""

    items: list[tuple[str, str, str]] = []
    for mode in MODES:
        for team, aliases in TEAM_ARTICLES.items():
            for alias in aliases:
                items.append((team, alias.replace("_", " "), mode))
    return items


def run_ingest(output_dir: Path, *, resume: bool, time_budget_seconds: float | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    already_done: set[str] = set()
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        already_done = {r["file"] for r in manifest.get("requests", []) if r.get("parsed_ok")}
        print(f"resuming {output_dir}: {len(already_done)} requests already parsed_ok, skipping")
    else:
        manifest = {
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "domain_allowlist": DOMAIN_ALLOWLIST,
            "endpoint": GDELT_ENDPOINT,
            "modes": MODES,
            "start": DEFAULT_START,
            "end": DEFAULT_END,
            "requests": [],
        }

    items = _work_items()
    total = len(items)
    started = time.time()
    n_run_this_session = 0
    stopped_early = False
    for i, (team, alias, mode) in enumerate(items, start=1):
        safe_alias = alias.replace(" ", "_")
        fname = f"{team}__{safe_alias}__{mode}.json"
        if fname in already_done:
            continue
        if time_budget_seconds is not None and (time.time() - started) > time_budget_seconds:
            print(
                f"\nTIME BUDGET EXCEEDED ({time_budget_seconds}s) at item {i}/{total} "
                f"({team} / {alias!r} / {mode}). Stopping; resume with --resume."
            )
            stopped_early = True
            break
        print(f"[{i}/{total}] {team} / {alias!r} / {mode}", flush=True)
        result = fetch(alias, mode)
        (output_dir / fname).write_text(
            json.dumps(result["parsed"]) if result["parsed_ok"] else "null", encoding="utf-8"
        )
        manifest["requests"].append(
            {
                "team": team,
                "alias": alias,
                "mode": mode,
                "file": fname,
                "url": result["url"],
                "http_status": result["http_status"],
                "retries": result["retries"],
                "response_bytes": result["response_bytes"],
                "parsed_ok": result["parsed_ok"],
                "parse_error": result["parse_error"],
                "body_on_failure": result["body_on_failure"],
            }
        )
        n_run_this_session += 1
        if not result["parsed_ok"]:
            print(f"    WARNING: parse failed, retries={result['retries']}", flush=True)
        # Checkpoint after every request.
        manifest["n_requests_so_far"] = len(manifest["requests"])
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["n_requests_total_possible"] = total
    manifest["n_requests_recorded"] = len(manifest["requests"])
    manifest["n_requests_this_session"] = n_run_this_session
    manifest["n_parse_failures"] = sum(1 for r in manifest["requests"] if not r["parsed_ok"])
    manifest["stopped_early_on_time_budget"] = stopped_early
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {manifest_path}")
    print(
        f"recorded={manifest['n_requests_recorded']}/{total} "
        f"failures={manifest['n_parse_failures']} this_session={n_run_this_session}"
    )


def run_status(output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_mode_team: dict[str, set[str]] = {m: set() for m in MODES}
    fail_by_mode_team: dict[str, set[str]] = {m: set() for m in MODES}
    for r in manifest.get("requests", []):
        mode = r["mode"]
        if r["parsed_ok"]:
            by_mode_team.setdefault(mode, set()).add(r["team"])
        else:
            fail_by_mode_team.setdefault(mode, set()).add(r["team"])
    total_teams = len(TEAM_ARTICLES)
    for mode in MODES:
        done = by_mode_team.get(mode, set())
        failed = fail_by_mode_team.get(mode, set()) - done
        print(
            f"{mode}: {len(done)}/{total_teams} teams with >=1 alias parsed_ok; "
            f"failed: {sorted(failed)}"
        )
    print(f"total requests recorded: {len(manifest.get('requests', []))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="run (or resume) the full backfill")
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--resume", action="store_true")
    ingest.add_argument(
        "--time-budget-seconds",
        type=float,
        default=None,
        help="stop cleanly (checkpointed, resumable) after this many seconds",
    )

    status = sub.add_parser("status", help="report coverage progress for a snapshot dir")
    status.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "ingest":
        run_ingest(args.output, resume=args.resume, time_budget_seconds=args.time_budget_seconds)
    elif args.cmd == "status":
        run_status(args.output)


if __name__ == "__main__":
    main()
