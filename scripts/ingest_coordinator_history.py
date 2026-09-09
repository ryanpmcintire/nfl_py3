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

from nfl_ats.provenance import stamp_sidecar, write_stamped_artifact
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
    value = re.sub(r"\s*\(interim\)\s*", "", value, flags=re.I)
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
            names: dict[str, set[str]] = {}
            for line in content.splitlines():
                line = re.sub(
                    r"^(\s*\*\s*)\[\[(?:[^\]|]+\|)?"
                    r"(head coach|offensive coordinator|defensive coordinator)\]\]",
                    r"\1\2",
                    line,
                    flags=re.I,
                )
                match = re.match(
                    r"^\s*\*\s*(?:(?:assistant|associate) head coach\s*[/&]\s*)?"
                    r"(?:interim\s+)?"
                    r"(head coach|offensive coordinator|defensive coordinator)"
                    r"(?:\s*\(interim\))?"
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


TEAM_NAMES = dict(
    zip(
        [
            "ARI",
            "ATL",
            "BAL",
            "BUF",
            "CAR",
            "CHI",
            "CIN",
            "CLE",
            "DAL",
            "DEN",
            "DET",
            "GB",
            "HOU",
            "IND",
            "JAX",
            "KC",
            "LA",
            "LAC",
            "LV",
            "MIA",
            "MIN",
            "NE",
            "NO",
            "NYG",
            "NYJ",
            "PHI",
            "PIT",
            "SEA",
            "SF",
            "TB",
            "TEN",
            "WAS",
        ],
        [
            "Arizona Cardinals",
            "Atlanta Falcons",
            "Baltimore Ravens",
            "Buffalo Bills",
            "Carolina Panthers",
            "Chicago Bears",
            "Cincinnati Bengals",
            "Cleveland Browns",
            "Dallas Cowboys",
            "Denver Broncos",
            "Detroit Lions",
            "Green Bay Packers",
            "Houston Texans",
            "Indianapolis Colts",
            "Jacksonville Jaguars",
            "Kansas City Chiefs",
            "Los Angeles Rams",
            "Los Angeles Chargers",
            "Las Vegas Raiders",
            "Miami Dolphins",
            "Minnesota Vikings",
            "New England Patriots",
            "New Orleans Saints",
            "New York Giants",
            "New York Jets",
            "Philadelphia Eagles",
            "Pittsburgh Steelers",
            "Seattle Seahawks",
            "San Francisco 49ers",
            "Tampa Bay Buccaneers",
            "Tennessee Titans",
            "Washington Commanders",
        ],
        strict=True,
    )
)


def historical_title(team: str, season: int) -> str:
    name = TEAM_NAMES[team]
    if team == "LA" and season < 2016:
        name = "St. Louis Rams"
    if team == "LAC" and season < 2017:
        name = "San Diego Chargers"
    if team == "LV" and season < 2020:
        name = "Oakland Raiders"
    if team == "WAS":
        name = (
            "Washington Redskins"
            if season < 2020
            else ("Washington Football Team" if season < 2022 else name)
        )
    return f"Template:{name} staff"


def league_capture(
    *, max_requests: int, delay: float, inseason: bool, schedules_path: Path | None = None
) -> Path:
    """One serial pass, following redirects and API continuation without retries."""
    require_private_raw_destination(SOURCE, ROOT)
    if delay < 1 or max_requests < 1:
        raise ValueError("Require delay >= 1 second and positive request budget")
    destination = ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    cache = {}
    for path in sorted(ROOT.glob("*/*.request.json")):
        meta = json.loads(path.read_bytes())
        if meta.get("status") == 200 and meta.get("api_ok"):
            cache.setdefault(meta["url"], path)
    schedules_path = schedules_path or max(Path("data/raw").glob("*/schedules.parquet"))
    schedules = pd.read_parquet(schedules_path)
    from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES

    for side in ("home", "away"):
        schedules[f"{side}_team"] = schedules[f"{side}_team"].replace(TEAM_ABBREVIATION_ALIASES)
    requests, rows, resolutions = [], [], []
    used = 0
    stopped = None
    jobs = [(year, team, "preseason") for year in range(2009, 2026) for team in TEAM_NAMES]
    if inseason:
        jobs += [(year, team, "inseason") for year in range(2022, 2026) for team in TEAM_NAMES]
    for season, team, mode in jobs:
        start = f"{season}-09-01T00:00:00Z"
        title = historical_title(team, season)
        params = dict(
            urllib.parse.parse_qsl(urllib.parse.urlsplit(revision_url(title, start)).query)
        )
        params["redirects"] = "1"
        cutoff = start
        if mode == "inseason":
            games = schedules.loc[
                schedules.season.eq(season)
                & (schedules.home_team.eq(team) | schedules.away_team.eq(team))
            ]
            cutoff = (pd.Timestamp(games.gameday.max()) + pd.Timedelta(days=1)).strftime(
                "%Y-%m-%dT00:00:00Z"
            )
            params.update(rvdir="newer", rvlimit="50", rvend=cutoff)
        complete = False
        while True:
            url = API + "?" + urllib.parse.urlencode(params)
            key = hashlib.sha256(url.encode()).hexdigest()
            prior = cache.get(url)
            if prior:
                metadata = json.loads(prior.read_bytes())
                body = prior.with_name(f"{key}.response").read_bytes()
                metadata = {**metadata, "cache_origin": str(prior)}
            else:
                if used >= max_requests:
                    stopped = "request_budget"
                    break
                status, body = fetch(url)
                used += 1
                metadata = {
                    "url": url,
                    "status": status,
                    "requested_at": datetime.now(UTC).isoformat(),
                    "user_agent": UA,
                    "cache_origin": None,
                }
                time.sleep(delay)
            payload = {}
            parsed = []
            try:
                if metadata["status"] != 200:
                    raise ValueError(f"HTTP/transport status {metadata['status']}")
                payload = json.loads(body)
                parsed = parse_revision(
                    payload,
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
                mode=mode,
                assignments=len(parsed),
                sha256=hashlib.sha256(body).hexdigest(),
            )
            (destination / f"{key}.response").write_bytes(body)
            write_stamped_artifact(metadata, destination / f"{key}.request.json")
            requests.append(metadata)
            rows.extend({**row, "sample_mode": mode} for row in parsed)
            if stopped:
                break
            continuation = payload.get("continue") if mode == "inseason" else None
            if not continuation:
                complete = True
                break
            params.update(continuation)
        resolutions.append(
            {
                "season": season,
                "team": team,
                "mode": mode,
                "requested_title": title,
                "complete": complete,
                "cutoff": cutoff,
                "pages": payload.get("query", {}).get("pages", [])
                and [
                    {k: v for k, v in page.items() if k != "revisions"}
                    for page in payload["query"]["pages"]
                ],
                "redirects": payload.get("query", {}).get("redirects", []),
            }
        )
        print(f"{mode} {season} {team}: {len(rows)} assignments; {used} requests", flush=True)
        if stopped:
            break
    frame = pd.DataFrame(rows, columns=[*COLUMNS, "sample_mode"])
    frame["effective_observed_at"] = pd.to_datetime(frame.effective_observed_at, utc=True)
    frame.to_parquet(destination / "coordinator_history.parquet", index=False)
    stamp_sidecar(destination / "coordinator_history.parquet")
    write_stamped_artifact(
        {
            "source": SOURCE,
            "license": "CC-BY-SA-4.0",
            "attribution": "Wikipedia contributors; revision permalinks in rows",
            "requests": requests,
            "resolutions": resolutions,
            "network_requests": used,
            "stopped": stopped,
            "assignments": len(rows),
            "schedules_path": str(schedules_path),
        },
        destination / "manifest.json",
    )
    return destination


def normalize_capture(source: Path) -> Path:
    """Reparse saved responses into a new immutable snapshot, never refetch."""
    manifest = json.loads((source / "manifest.json").read_bytes())
    rows = []
    cutoffs = {(r["season"], r["team"], r["mode"]): r["cutoff"] for r in manifest["resolutions"]}
    for request in manifest["requests"]:
        if not request.get("api_ok"):
            continue
        key = hashlib.sha256(request["url"].encode()).hexdigest()
        response = source / f"{key}.response"
        request["raw_response_path"] = str(response)
        parsed = parse_revision(
            json.loads(response.read_bytes()),
            season=request["season"],
            team=request["team"],
            cutoff=cutoffs[(request["season"], request["team"], request["mode"])],
            retrieved_at=request["requested_at"],
        )
        rows.extend({**row, "sample_mode": request["mode"]} for row in parsed)
    destination = ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(rows, columns=[*COLUMNS, "sample_mode"])
    frame["effective_observed_at"] = pd.to_datetime(frame.effective_observed_at, utc=True)
    frame.to_parquet(destination / "coordinator_history.parquet", index=False)
    stamp_sidecar(destination / "coordinator_history.parquet")
    manifest.update(
        source_snapshot=str(source), normalization_network_requests=0, assignments=len(rows)
    )
    write_stamped_artifact(manifest, destination / "manifest.json")
    return destination


def audit_capture(destination: Path) -> dict[str, Any]:
    """Coverage census and revision-observed identity transitions, no invented event dates."""
    manifest = json.loads((destination / "manifest.json").read_bytes())
    history = pd.read_parquet(destination / "coordinator_history.parquet")
    schedules = pd.read_parquet(manifest["schedules_path"])
    from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES

    for side in ("home", "away"):
        schedules[f"{side}_team"] = schedules[f"{side}_team"].replace(TEAM_ABBREVIATION_ALIASES)
    schedules = schedules.loc[schedules.game_type.eq("REG")].copy()
    schedules["kickoff_utc"] = (
        pd.to_datetime(
            schedules.gameday.astype(str).str[:10] + " " + schedules.gametime.fillna("00:00")
        )
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    first = (
        pd.concat(
            [
                schedules[["season", f"{side}_team", "kickoff_utc"]].rename(
                    columns={f"{side}_team": "team"}
                )
                for side in ("home", "away")
            ]
        )
        .groupby(["season", "team"])
        .kickoff_utc.min()
    )
    pre = history.loc[history.sample_mode.eq("preseason")].copy()
    pre["first_kickoff"] = [first.get((r.season, r.team), pd.NaT) for r in pre.itertuples()]
    pre["before_first_kickoff"] = pre.effective_observed_at.lt(pre.first_kickoff)
    pre["lead_days"] = (pre.first_kickoff - pre.effective_observed_at).dt.total_seconds() / 86400
    details = []
    for season in range(2009, 2026):
        for team in TEAM_NAMES:
            subset = pre.loc[pre.season.eq(season) & pre.team.eq(team)]
            details.append(
                {
                    "season": season,
                    "team": team,
                    **{role: bool(subset.role.eq(role).any()) for role in ("OC", "DC", "HC")},
                    "before_kickoff": int(subset.before_first_kickoff.sum()),
                    "no_dated_observation": subset.empty,
                }
            )
    changes = []
    for (season, team, role), group in history.loc[
        history.season.between(2022, 2025) & history.role.isin(["OC", "DC"])
    ].groupby(["season", "team", "role"]):
        group = group.sort_values("effective_observed_at").drop_duplicates("revision_id")
        previous = None
        for row in group.to_dict("records"):
            if (
                previous is not None
                and row["sample_mode"] == "inseason"
                and row["person"] != previous["person"]
            ):
                kickoff = first.get((season, team), pd.NaT)
                changes.append(
                    {
                        "season": int(season),
                        "team": team,
                        "role": role,
                        "previous_person": previous["person"],
                        "person": row["person"],
                        "previous_observed_at": str(previous["effective_observed_at"]),
                        "revision_at": str(row["effective_observed_at"]),
                        "source_url": row["source_url"],
                        "after_first_kickoff": bool(row["effective_observed_at"] > kickoff),
                        "event_date": None,
                        "revision_lag_days": None,
                        "lag_status": "unknown: no independent employment-event date",
                    }
                )
            previous = row
    result = {
        "source": str(destination),
        "schedules_path": manifest["schedules_path"],
        "team_seasons_requested": 544,
        "role_coverage": {role: int(pre.role.eq(role).sum()) for role in ("OC", "DC", "HC")},
        "role_before_kickoff": {
            role: int((pre.role.eq(role) & pre.before_first_kickoff).sum())
            for role in ("OC", "DC", "HC")
        },
        "no_dated_observation": sum(row["no_dated_observation"] for row in details),
        "all_three_roles": sum(all(row[role] for role in ("OC", "DC", "HC")) for row in details),
        "lead_days_min": float(pre.lead_days.min()),
        "lead_days_max": float(pre.lead_days.max()),
        "preseason_complete": sum(
            r["complete"] for r in manifest["resolutions"] if r["mode"] == "preseason"
        ),
        "inseason_complete": sum(
            r["complete"] for r in manifest["resolutions"] if r["mode"] == "inseason"
        ),
        "network_requests": manifest["network_requests"],
        "stopped": manifest["stopped"],
        "team_seasons": details,
        "changes": changes,
        "inseason_observed_identity_changes": sum(row["after_first_kickoff"] for row in changes),
        "unknown_event_lags": len(changes),
        "scope": (
            "Parsed identity changes, not adjudicated appointments; "
            "vacancies and unlinked names omitted"
        ),
    }
    write_stamped_artifact(result, destination / "coverage_audit.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", action="append", metavar="ABBR=Template:TEAM staff")
    parser.add_argument("--start-season", type=int, default=2009)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--cutoff", default="09-01T00:00:00Z")
    parser.add_argument("--max-requests", type=int, default=14)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--league", action="store_true")
    parser.add_argument("--inseason", action="store_true")
    parser.add_argument("--normalize", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.normalize:
        print(normalize_capture(args.normalize))
        return
    if args.audit:
        print(json.dumps(audit_capture(args.audit), indent=2))
        return
    if args.league:
        print(
            league_capture(max_requests=args.max_requests, delay=args.delay, inseason=args.inseason)
        )
        return
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
