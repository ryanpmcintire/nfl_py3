"""Capture dated Wikipedia staff revisions; source acquisition, never an ATS experiment.

One historical revision per team/season/cutoff is a conservative observation,
not a claim to have found the first announcement or every intervening change.
Successful responses are reused across immutable snapshots on subsequent runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.provenance import write_stamped_artifact
from nfl_ats.source_policy import require_acquisition, require_private_raw_destination

SOURCE = "wikipedia_coordinator_revisions"
ROOT = Path("data/raw/coordinators")
UA = "NFLATSCoordinatorResearch/0.1 (https://github.com/ryanpmcintire/nfl_py3; research)"
API = "https://en.wikipedia.org/w/api.php"
COLUMNS = [
    "season",
    "team",
    "role",
    "person",
    "effective_observed_at",
    "observed_at_basis",
    "source_url",
    "revision_id",
    "retrieved_at",
    "sampled_as_of",
    "source_title",
]
ROLE_LABELS = {"offensive coordinator": "OC", "defensive coordinator": "DC", "head coach": "HC"}


def revision_url(title: str, cutoff: str) -> str:
    return (
        API
        + "?"
        + urllib.parse.urlencode(
            {
                "action": "query",
                "prop": "revisions",
                "titles": title,
                "rvprop": "ids|timestamp|content",
                "rvslots": "main",
                "rvlimit": "1",
                "rvstart": cutoff,
                "rvdir": "older",
                "format": "json",
                "formatversion": "2",
                "maxlag": "5",
            }
        )
    )


def linked_person(value: str) -> str | None:
    """Accept one unambiguous wikilink, never dates, vacancies, or co-coordinators."""
    value = re.sub(r"<ref\b[^>]*>.*?</ref>|<ref\b[^>]*/>", "", value, flags=re.S)
    matches = re.findall(r"\[\[([^\[\]]+)\]\]", value)
    if len(matches) != 1:
        return None
    residue = re.sub(r"\[\[[^\[\]]+\]\]", "", value).strip(" \t'\u2013\u2014-:")
    if residue:
        return None
    target = matches[0].split("|", 1)[0].strip()
    if any(mark in target for mark in (":", "#", "{")):
        return None
    return target or None


def parse_revision(
    payload: dict[str, Any],
    *,
    season: int,
    team: str,
    cutoff: str,
    retrieved_at: str,
) -> list[dict[str, Any]]:
    if "error" in payload:
        raise ValueError(f"MediaWiki error: {payload['error']}")
    if not isinstance(payload.get("query", {}).get("pages"), list):
        raise ValueError("Missing MediaWiki query.pages response")
    rows = []
    for page in payload.get("query", {}).get("pages", []):
        for revision in page.get("revisions", []):
            stamp = revision.get("timestamp")
            observed = pd.to_datetime(stamp, utc=True, errors="coerce")
            if pd.isna(observed) or observed > pd.Timestamp(cutoff):
                raise ValueError("Revision requires a real timestamp no later than cutoff")
            content = revision.get("slots", {}).get("main", {}).get("content", "")
            # Never expand transclusions: the current template can rewrite historical text.
            names: dict[str, set[str]] = {}
            for line in content.splitlines():
                match = re.match(
                    r"^\s*\*\s*(head coach|offensive coordinator|defensive coordinator)"
                    r"\s*(?:[/&][A-Za-z /&-]+)?\s*[\u2013\u2014-]\s*(.+)$",
                    line,
                    re.I,
                )
                if match:
                    role = ROLE_LABELS[match[1].lower()]
                    person = linked_person(match[2])
                else:
                    match = re.match(
                        r"^\s*\|\s*(offensive_coordinator|defensive_coordinator|coach)\s*=\s*(.+)$",
                        line,
                        re.I,
                    )
                    if not match:
                        continue
                    role = {
                        "offensive_coordinator": "OC",
                        "defensive_coordinator": "DC",
                        "coach": "HC",
                    }[match[1].lower()]
                    person = linked_person(match[2])
                # Unparseable duplicate labels make the role unavailable too.
                names.setdefault(role, set()).add(person or "")
            for role, people in names.items():
                if len(people) != 1 or "" in people:
                    continue
                rows.append(
                    {
                        "season": season,
                        "team": team,
                        "role": role,
                        "person": next(iter(people)),
                        "effective_observed_at": observed.isoformat(),
                        "observed_at_basis": "wikipedia_revision",
                        "revision_id": revision["revid"],
                        "source_url": f"https://en.wikipedia.org/w/index.php?oldid={revision['revid']}",
                        "retrieved_at": retrieved_at,
                        "sampled_as_of": cutoff,
                        "source_title": page["title"],
                    }
                )
    return rows


def fetch(url: str) -> tuple[int | None, bytes]:
    require_acquisition(SOURCE)
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": UA}),
            timeout=30,
        ) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except OSError as error:
        return None, str(error).encode("utf-8")


def capture(
    teams: dict[str, str],
    seasons: list[int],
    *,
    cutoff_suffix: str = "09-01T00:00:00Z",
    max_requests: int = 14,
    delay: float = 1.0,
) -> Path:
    require_private_raw_destination(SOURCE, ROOT)
    if delay < 1 or max_requests < 1:
        raise ValueError("Require delay >= 1 second and positive request budget")
    if any(year < 2009 or year > 2025 for year in seasons):
        raise ValueError("Historical source scope is 2009-2025")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / stamp
    destination.mkdir(parents=True, exist_ok=False)
    requests, rows = [], []
    used = 0
    stopped = None
    for season in seasons:
        cutoff = f"{season}-{cutoff_suffix}"
        if pd.Timestamp(cutoff).year != season:
            raise ValueError("Cutoff must belong to season")
        for team, title in teams.items():
            url = revision_url(title, cutoff)
            key = hashlib.sha256(url.encode()).hexdigest()
            cached = sorted(ROOT.glob(f"*/{key}.request.json"))
            prior = None
            for candidate in cached:
                metadata = json.loads(candidate.read_text(encoding="utf-8"))
                if metadata["status"] == 200 and metadata.get("api_ok"):
                    prior = candidate
                    break
            if prior:
                metadata = json.loads(prior.read_text(encoding="utf-8"))
                body = prior.with_name(f"{key}.response").read_bytes()
                metadata = {**metadata, "cache_origin": str(prior)}
            else:
                if used >= max_requests:
                    stopped = "request_budget"
                    break
                requested_at = datetime.now(UTC).isoformat()
                status, body = fetch(url)
                used += 1
                metadata = {
                    "url": url,
                    "status": status,
                    "requested_at": requested_at,
                    "user_agent": UA,
                    "cache_origin": None,
                }
                time.sleep(delay)
            parsed = []
            try:
                if metadata["status"] != 200:
                    raise ValueError(f"HTTP/transport status {metadata['status']}")
                parsed = parse_revision(
                    json.loads(body),
                    season=season,
                    team=team,
                    cutoff=cutoff,
                    retrieved_at=metadata["requested_at"],
                )
                metadata["api_ok"] = True
            except (ValueError, KeyError, TypeError) as error:
                metadata.update(api_ok=False, error=str(error))
                stopped = "source_error"
            metadata.update(
                season=season,
                team=team,
                assignments=len(parsed),
                sha256=hashlib.sha256(body).hexdigest(),
            )
            (destination / f"{key}.response").write_bytes(body)
            write_stamped_artifact(metadata, destination / f"{key}.request.json")
            requests.append(metadata)
            rows.extend(parsed)
            if stopped:
                break
        if stopped:
            break
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["effective_observed_at"] = pd.to_datetime(frame["effective_observed_at"], utc=True)
    frame.to_parquet(destination / "coordinator_history.parquet", index=False)
    write_stamped_artifact(
        {
            "source": SOURCE,
            "license": "CC-BY-SA-4.0",
            "attribution": "Wikipedia contributors; see source_url",
            "teams": teams,
            "seasons": seasons,
            "cutoff_suffix": cutoff_suffix,
            "requests": requests,
            "network_requests": used,
            "stopped": stopped,
            "assignments": len(frame),
            "scope": "one dated observation per team/role/cutoff; not exhaustive change history",
        },
        destination / "manifest.json",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--team", action="append", required=True, metavar="ABBR=Template:TEAM staff"
    )
    parser.add_argument("--start-season", type=int, default=2009)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--cutoff", default="09-01T00:00:00Z")
    parser.add_argument("--max-requests", type=int, default=14)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    teams = dict(item.split("=", 1) for item in args.team)
    destination = capture(
        teams,
        list(range(args.start_season, args.end_season + 1)),
        cutoff_suffix=args.cutoff,
        max_requests=args.max_requests,
        delay=args.delay,
    )
    print(destination)
    print((destination / "manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
