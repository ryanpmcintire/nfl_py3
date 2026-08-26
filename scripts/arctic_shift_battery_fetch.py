"""Widen Arctic Shift subreddit coverage for the ATS battery (docs/arctic_shift_ats_battery.md).

The frozen shared-variance gate (``docs/arctic_shift_gate.md``,
``scripts/arctic_shift_gate.py``) sampled 6 team subreddits x REG 2019-2021
and FAILED the shared-variance leg against Wikipedia pageviews (r=0.73 >=
0.7) -- that question is closed. The gate's own caveat: "Correlation here
measures construct overlap, not predictive value; an ATS battery remains a
separate decision." This script does the data-widening half of that separate
decision: since the Arctic Shift ``/api/time_series`` endpoint costs the
SAME one request per (subreddit, kind) regardless of the date range
requested, widening SEASONS is free in request count. Widening TEAMS costs
two requests each (posts, comments), so all 32 NFL team subreddits are
fetched here (the original 6 are re-fetched too, for one consistent full-
history file per team rather than a 2019-2021 file plus a separate wide
file).

Saves every raw response under ``data/raw/arctic_shift/`` (gitignored, per
repo convention) with a sha256 manifest, exactly mirroring
``scripts/arctic_shift_gate.py``'s ``fetch_subreddit_daily_counts`` (same
retry/backoff, same UTC-midnight ``/api/time_series`` endpoint -- the
search/aggregate endpoint's ``T22:00Z`` bucket labels are NOT used, per the
gate's own recorded caveat). Fetch-only: does not build features or score
anything.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

API_BASE = "https://arctic-shift.photon-reddit.com"
REQUEST_DELAY_SECONDS = 3.0
MAX_ATTEMPTS = 6
BACKOFF_BASE_SECONDS = 5.0

# 32 canonical team codes (nfl_ats.constants.TEAM_ABBREVIATION_ALIASES-
# canonicalized) -> best-effort current official team subreddit name.
# Reported from general knowledge, NOT verified before this session; this
# script's own fetch results (item counts, error fields) are the
# verification, logged per team below.
SUBREDDITS_ALL: dict[str, str] = {
    "ARI": "AZCardinals",
    "ATL": "falcons",
    "BAL": "ravens",
    "BUF": "buffalobills",
    "CAR": "panthers",
    "CHI": "CHIBears",
    "CIN": "bengals",
    "CLE": "Browns",
    "DAL": "cowboys",
    "DEN": "DenverBroncos",
    "DET": "detroitlions",
    "GB": "GreenBayPackers",
    "HOU": "Texans",
    "IND": "Colts",
    "JAX": "Jaguars",
    "KC": "KansasCityChiefs",
    "LA": "LosAngelesRams",
    "LAC": "Chargers",  # "LosAngelesChargers" returns zero data; verified this session
    "LV": "raiders",
    "MIA": "miamidolphins",
    "MIN": "minnesotavikings",
    "NE": "Patriots",
    "NO": "Saints",
    "NYG": "NYGiants",
    "NYJ": "nyjets",
    "PHI": "eagles",
    "PIT": "steelers",
    "SEA": "Seahawks",
    "SF": "49ers",
    "TB": "buccaneers",
    "TEN": "Tennesseetitans",
    "WAS": "Commanders",
}

WINDOW_AFTER = "2010-08-01"
WINDOW_BEFORE = "2026-08-27"  # exclusive upper bound, one day past today


def fetch_with_retries(url: str) -> tuple[bytes | None, str | None]:
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "nfl-ats-battery-fetch/0.1"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read()
            payload = json.loads(body.decode("utf-8"))
            if isinstance(payload, dict) and payload.get("error"):
                last_error = str(payload["error"])
                # A named error (e.g. "subreddit not found") is not a
                # transient failure -- do not retry, report it directly.
                return None, last_error
            return body, None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None, "HTTP 404"
            last_error = f"HTTP {exc.code}"
            reset = exc.headers.get("X-RateLimit-Reset") if exc.headers else None
            wait = float(reset) + 5.0 if reset else BACKOFF_BASE_SECONDS * attempt
            print(f"  attempt {attempt} failed ({last_error}); waiting {wait:.0f}s")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = str(exc)
            wait = BACKOFF_BASE_SECONDS * attempt
            print(f"  attempt {attempt} failed ({last_error}); retrying in {wait:.0f}s")
            time.sleep(wait)
    return None, f"giving up after {MAX_ATTEMPTS} attempts: {last_error}"


def main() -> None:
    raw_dir = REPO / "data" / "raw" / "arctic_shift"
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    manifest_entries: list[dict[str, object]] = []
    summary: dict[str, dict[str, object]] = {}

    for team, subreddit in SUBREDDITS_ALL.items():
        summary[team] = {"subreddit": subreddit}
        for kind in ("posts", "comments"):
            params = urllib.parse.urlencode(
                {
                    "key": f"r/{subreddit}/{kind}/count",
                    "precision": "day",
                    "after": WINDOW_AFTER,
                    "before": WINDOW_BEFORE,
                }
            )
            url = f"{API_BASE}/api/time_series?{params}"
            time.sleep(REQUEST_DELAY_SECONDS)
            print(f"fetch {kind} r/{subreddit} ({team})")
            request_started = time.time()
            body, error = fetch_with_retries(url)
            elapsed = round(time.time() - request_started, 2)
            filename = f"{subreddit}_{kind}_timeseries_full.json"
            if body is None:
                summary[team][kind] = {"ok": False, "error": error}
                print(f"  FAILED: {error}")
                continue
            (raw_dir / filename).write_bytes(body)
            manifest_entries.append(
                {
                    "path": f"data/raw/arctic_shift/{filename}",
                    "url": url,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "bytes": len(body),
                    "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "elapsed_seconds": elapsed,
                }
            )
            items = json.loads(body.decode("utf-8"))["data"] or []
            total = sum(float(it["value"]) for it in items)
            summary[team][kind] = {
                "ok": True,
                "n_days": len(items),
                "total_count": total,
                "first_date": items[0]["date"] if items else None,
                "last_date": items[-1]["date"] if items else None,
            }
            print(f"  ok: n_days={len(items)} total={total:.0f}")

    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2) + "\n")

    summary_path = raw_dir / "fetch_summary.json"
    summary_payload = {
        "window_after": WINDOW_AFTER,
        "window_before": WINDOW_BEFORE,
        "api_base": API_BASE,
        "elapsed_seconds": time.time() - started,
        "teams": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

    n_ok = 0
    for team_summary in summary.values():
        for kind in ("posts", "comments"):
            kind_summary = team_summary.get(kind)
            if isinstance(kind_summary, dict) and kind_summary.get("ok"):
                n_ok += 1
    n_total = len(SUBREDDITS_ALL) * 2
    print(f"\n{n_ok}/{n_total} requests ok, elapsed={time.time() - started:.1f}s")
    print(f"wrote {manifest_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
