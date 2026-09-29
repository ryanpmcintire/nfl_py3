from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

sys.path.insert(0, str(Path("src").resolve()))

from nfl_ats.inactives_capture import (
    _parse_rotowire_grid,
    _parse_shared_design_system,
)
from nfl_ats.injury_headlines import parse_headline

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ROOTS = {
    "nflverse": Path("data/raw/nflverse_injuries"),
    "NFL.com weekly": Path("data/raw/nflcom_injuries"),
    "Sportradar": Path("data/raw/sportradar_injuries"),
    "inactives": Path("data/players/inactives"),
    "Headline archive": Path("data/raw/injury_news"),
}
TIME_PATTERN = re.compile(
    r"datePublished|dateModified|article:published_time|article:modified_time|"
    r"<time\b|last[ _-]?updated|report[ _-]?(?:date|time)|publication[ _-]?time",
    re.IGNORECASE,
)
SEASON_END = re.compile(
    r"season.ending|out.for.(?:the.)?season|miss.(?:the.)?(?:entire.)?season",
    re.IGNORECASE,
)


def clean(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def norm(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", clean(value).lower()).split())


def stamp(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def directory_time(path: Path) -> pd.Timestamp:
    return pd.to_datetime(path.name, format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")


def wilson(k: int, n: int) -> str:
    if not n:
        return "not estimable (n=0)"
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    midpoint = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (
        f"{k}/{n} ({100 * p:.1f}%; 95% "
        f"{100 * (midpoint - radius):.1f}-{100 * (midpoint + radius):.1f}%)"
    )


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in rows)
    return "\n".join(lines)


class Audit:
    def __init__(self, as_of: pd.Timestamp) -> None:
        self.as_of = as_of
        self.year = as_of.year
        self.inventory: list[dict[str, Any]] = []
        self.stats: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.previous: dict[str, dict[tuple[Any, ...], tuple[str, str]]] = {}
        self.seen: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
        self.designation_seen: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
        self.headline_sources: set[str] = set()
        self.prior_weeks: dict[str, dict[tuple[Any, ...], tuple[str, str]]] = defaultdict(dict)
        self.captures: list[dict[str, Any]] = []
        self.html: dict[str, Counter[str]] = defaultdict(Counter)
        self.html_candidates: list[dict[str, Any]] = []
        self.news: dict[str, dict[str, Any]] = {}
        self.parse_cache: dict[str, Any] = {}
        self.latest_injuries: Path | None = None
        self.history: Counter[str] = Counter()
        self.errors: list[str] = []

    def record_file(self, path: Path) -> None:
        payload = path.read_bytes()
        self.inventory.append(
            {
                "path": path.as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    def scan_html(self, source: str, path: Path) -> str:
        self.record_file(path)
        content = path.read_text(encoding="utf-8", errors="replace")
        hits = list(TIME_PATTERN.finditer(content))
        self.html[source]["pages"] += 1
        self.html[source]["pages_with_timestamp_markers"] += bool(hits)
        for hit in hits[:3]:
            self.html_candidates.append(
                {
                    "source": source,
                    "path": path.as_posix(),
                    "context": re.sub(
                        r"\s+", " ", content[max(0, hit.start() - 80) : hit.end() + 160]
                    ),
                }
            )
        return content

    def observe(
        self,
        source: str,
        at: pd.Timestamp,
        path: Path,
        rows: list[dict[str, Any]],
        native_week: bool,
    ) -> None:
        day = at.tz_convert("America/New_York").day_name()
        counts = self.stats[(source, day)]
        counts["captures"] += 1
        current: dict[tuple[Any, ...], tuple[str, str]] = {}
        modified = 0
        weeks: set[int] = set()
        for row in rows:
            key = row["key"]
            value = (clean(row["status"]), clean(row.get("practice")))
            if key in current and current[key] != value:
                raise ValueError(f"Conflicting duplicate at {path}: {key}")
            current[key] = value
            modified += bool(clean(row.get("modified")))
            if row.get("week") is not None:
                weeks.add(int(row["week"]))
        old = self.previous.get(source, {})
        seen = self.seen[source]
        prior_weeks = self.prior_weeks[source]
        for key, value in current.items():
            status, practice = value
            counts["rows"] += 1
            counts["designated"] += bool(status)
            if key not in old:
                counts["new_rows"] += key not in seen
                counts["reappeared_rows"] += key in seen
                if status:
                    counts["reappeared_designation" if key in seen else "first_designation"] += 1
            else:
                prior_status, prior_practice = old[key]
                counts["matched"] += 1
                counts["practice_changes"] += practice != prior_practice
                if status and not prior_status:
                    counts["first_designation"] += 1
                elif status and status != prior_status:
                    counts["changed_designation"] += 1
                elif status:
                    counts["unchanged_designation"] += 1
                elif prior_status:
                    counts["cleared_designation"] += 1
                else:
                    counts["both_blank"] += 1
            if native_week and key not in self.designation_seen[source] and status:
                season, week, team, player = key
                previous_value = prior_weeks.get((season, week - 1, team, player))
                if previous_value and previous_value[0]:
                    counts["new_week_comparable"] += 1
                    counts["new_week_same"] += previous_value[0] == status
        counts["removed_rows"] += len(old.keys() - current.keys())
        counts["native_modified_rows"] += modified
        counts["empty_captures"] += not current
        self.previous[source] = current
        self.seen[source].update(current)
        self.designation_seen[source].update(key for key, value in current.items() if value[0])
        prior_weeks.update(current)
        self.captures.append(
            {
                "source": source,
                "capture_utc": at.isoformat(),
                "capture_et": at.tz_convert("America/New_York").isoformat(),
                "path": path.as_posix(),
                "rows": len(current),
                "designations": sum(bool(v[0]) for v in current.values()),
                "weeks": sorted(weeks),
                "native_modified_rows": modified,
                "week_designations": {
                    str(week): sum(
                        bool(value[0]) for key, value in current.items() if key[1] == week
                    )
                    for week in sorted(weeks)
                }
                if native_week
                else {},
            }
        )

    def weekly(self, source: str, path: Path, at: pd.Timestamp) -> None:
        parquet = path.parent / "injuries.parquet"
        if not parquet.exists():
            self.errors.append(f"{path}: no injuries.parquet")
            return
        self.record_file(parquet)
        frame = pd.read_parquet(parquet)
        self.history[source] += int((frame["season"] != self.year).sum())
        frame = frame.loc[frame["season"] == self.year]
        if source == "nflverse":
            self.latest_injuries = parquet
            if "game_type" in frame:
                frame = frame.loc[frame["game_type"].fillna("REG").eq("REG")]
        rows = []
        for row in frame.to_dict("records"):
            player = clean(row.get("gsis_id")) or norm(row.get("full_name", row.get("player")))
            if pd.isna(row["week"]):
                raise ValueError(f"Missing report week at {parquet}")
            rows.append(
                {
                    "key": (self.year, int(row["week"]), clean(row["team"]), player),
                    "week": row["week"],
                    "status": row.get("report_status", row.get("game_status")),
                    "practice": row.get("practice_status"),
                    "modified": row.get("date_modified"),
                }
            )
        self.observe(source, at, parquet, rows, True)
        if source == "NFL.com weekly":
            for page in sorted((path.parent / "pages").glob("*.html")):
                self.scan_html(source, page)

    def inactives(self, path: Path, manifest: dict[str, Any], at: pd.Timestamp) -> None:
        for filename, source, parser in [
            ("primary.html", "NFL.com inactives", _parse_shared_design_system),
            ("fallback.html", "RotoWire inactives", _parse_rotowire_grid),
        ]:
            page = path.parent / filename
            if not page.exists():
                continue
            content = self.scan_html(source, page)
            rows, warnings = parser(
                content,
                season=int(manifest["season"]),
                week=int(manifest["week"]),
                source_url=source,
                fetched_at_utc=at.isoformat(),
            )
            self.errors.extend(f"{page}: {warning}" for warning in warnings)
            normalized = [
                {"key": (norm(row["team"]), norm(row["player_name"])), "status": row["status"]}
                for row in rows
            ]
            self.observe(source, at, page, normalized, False)

    def headlines(self, path: Path, at: pd.Timestamp) -> None:
        parquet = path.parent / "current.parquet"
        if not parquet.exists():
            self.errors.append(f"{path}: no current.parquet")
            return
        self.record_file(parquet)
        frame = pd.read_parquet(parquet)
        modified = pd.to_datetime(frame["lastmod"], utc=True, errors="coerce")
        frame = frame.loc[modified.dt.year.eq(self.year)]
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in frame.to_dict("records"):
            headline = clean(row["headline_guess"])
            url = clean(row["url"])
            host = urlsplit(url).netloc.lower().removeprefix("www.")
            source = {"nbcsports.com": "NBC Sports headlines"}.get(host, f"{host} headlines")
            by_source[source]
            if url not in self.news:
                self.news[url] = {
                    "headline": headline,
                    "first_seen": at.isoformat(),
                    "lastmod": clean(row["lastmod"]),
                    "url": url,
                    "source": source,
                }
            if headline not in self.parse_cache:
                self.parse_cache[headline] = parse_headline(headline)
            for parsed in self.parse_cache[headline]:
                if parsed.parsed and parsed.designation:
                    by_source[source].append(
                        {
                            "key": (url, norm(parsed.player_name)),
                            "status": parsed.designation,
                            "modified": row["lastmod"],
                        }
                    )
        self.headline_sources.update(by_source)
        for source in sorted(self.headline_sources or {"Headline index (empty)"}):
            self.observe(source, at, parquet, by_source[source], False)

    def run(self) -> None:
        pending = []
        for source, root in ROOTS.items():
            for path in sorted(root.glob("*/manifest.json")):
                capture = directory_time(path.parent)
                if pd.isna(capture) or capture > self.as_of:
                    continue
                manifest = json.loads(path.read_text(encoding="utf-8-sig"))
                clock = next(
                    (
                        manifest[k]
                        for k in [
                            "fetched_utc",
                            "captured_at_utc",
                            "fetched_at",
                            "generated_at_utc",
                        ]
                        if manifest.get(k)
                    ),
                    None,
                )
                if clock is None:
                    raise ValueError(f"Missing capture timestamp at {path}")
                at = stamp(clock)
                if pd.isna(at):
                    raise ValueError(f"Invalid capture timestamp at {path}: {clock}")
                if at <= self.as_of:
                    pending.append((at, source, path, manifest))
        for at, source, path, manifest in sorted(pending, key=lambda item: (item[0], str(item[2]))):
            self.record_file(path)
            if source == "inactives":
                self.inactives(path, manifest, at)
            elif source == "Headline archive":
                self.headlines(path, at)
            else:
                self.weekly(source, path, at)

    def roster_check(self) -> dict[str, Any]:
        candidates = [
            path
            for path in Path("data/players/raw").glob("*/weekly_rosters.parquet")
            if directory_time(path.parent) <= self.as_of
        ]
        if not candidates or self.latest_injuries is None:
            raise ValueError("Missing injury or weekly-roster snapshot")
        latest = max(candidates, key=lambda path: directory_time(path.parent))
        self.record_file(latest)
        roster = pd.read_parquet(latest)
        injuries = pd.read_parquet(self.latest_injuries)
        keys = ["season", "week", "team", "gsis_id"]
        for frame in [roster, injuries]:
            if "game_type" in frame:
                frame.drop(frame.index[~frame["game_type"].fillna("REG").eq("REG")], inplace=True)
            frame["team"] = frame["team"].replace({"LA": "LAR", "JAC": "JAX", "WSH": "WAS"})
        eligible = injuries[["season", "week"]].drop_duplicates()
        roster = roster.merge(eligible, on=["season", "week"], how="inner")
        missing_ids = int(roster["gsis_id"].isna().sum())
        roster = roster.loc[roster["gsis_id"].notna()].drop_duplicates(keys)
        report_keys = injuries[keys].dropna(subset=["gsis_id"]).drop_duplicates()
        joined = roster.merge(report_keys.assign(in_report=True), on=keys, how="left")
        joined["in_report"] = joined["in_report"].eq(True)
        current = joined.loc[joined["season"].eq(self.year)]
        coverage = []
        for week, group in current.groupby("week"):
            for status in ["RES", "ACT"]:
                subset = group.loc[group["status"].eq(status)]
                coverage.append([int(week), status, len(subset), int(subset["in_report"].sum())])
        reserve = joined.loc[joined["status"].eq("RES")]
        examples = []
        cutoff = directory_time(latest.parent)
        names = {norm(name): name for name in current["full_name"].dropna().unique()}
        for article in self.news.values():
            if not SEASON_END.search(article["headline"]) or stamp(article["first_seen"]) > cutoff:
                continue
            text = " " + norm(article["headline"]) + " "
            matches = [
                name for name in names if len(name.split()) >= 2 and " " + name + " " in text
            ]
            for name in matches:
                rows = current.loc[current["full_name"].eq(names[name])]
                examples.append(
                    {
                        "player": names[name],
                        "first_seen": article["first_seen"],
                        "headline": article["headline"],
                        "url": article["url"],
                        "weeks": [
                            {
                                "week": int(row.week),
                                "status": row.status,
                                "in_report": row.in_report,
                            }
                            for row in rows.itertuples()
                        ],
                    }
                )
        return {
            "roster_path": latest.as_posix(),
            "injury_path": self.latest_injuries.as_posix(),
            "coverage": coverage,
            "missing_roster_ids": missing_ids,
            "all_seasons_reserve_rows": len(reserve),
            "all_seasons_reserve_matches": int(reserve["in_report"].sum()),
            "examples": examples,
        }

    def scheduler_check(self) -> dict[str, Any]:
        path = Path("data/scheduler_log.txt")
        if not path.exists():
            return {"counts": [], "matched_captures": 0, "snapshot_mentions": 0}
        self.record_file(path)
        counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        mentions: set[str] = set()
        pattern = re.compile(
            r"^(\S+) (?:(MANUAL-RUN) )?(OK|FAIL(?:\(\d+\))?|SKIP) "
            r"((?:nflverse_injuries|sportradar_injuries|injuries|inactives|injury_news)_\w+):"
        )
        first = None
        last = None
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = pattern.search(line)
                if not match:
                    continue
                at = stamp(match[1])
                if pd.isna(at) or at > self.as_of:
                    continue
                source = next(
                    prefix
                    for prefix in [
                        "nflverse_injuries",
                        "sportradar_injuries",
                        "injuries",
                        "inactives",
                        "injury_news",
                    ]
                    if match[4].startswith(prefix + "_")
                )
                counts[(source, at.tz_convert("America/New_York").day_name())][match[3]] += 1
                mentions.update(re.findall(r"20\d{6}T\d{6}Z", line))
                first = at if first is None else min(first, at)
                last = at if last is None else max(last, at)
        matched = sum(Path(row["path"]).parent.name in mentions for row in self.captures)
        return {
            "counts": [
                {"source": s, "weekday": d, **dict(c)} for (s, d), c in sorted(counts.items())
            ],
            "matched_captures": matched,
            "snapshot_mentions": len(mentions),
            "first": str(first),
            "last": str(last),
        }

    def report(self, ir: dict[str, Any], logs: dict[str, Any]) -> str:
        rows = []
        totals: dict[str, Counter[str]] = defaultdict(Counter)
        for (source, _), counts in self.stats.items():
            totals[source].update(counts)
        for source in sorted(totals):
            for day in WEEKDAYS:
                c = self.stats.get((source, day))
                if c is None:
                    continue
                repeated = c["changed_designation"] + c["unchanged_designation"]
                rows.append(
                    [
                        source,
                        day,
                        c["captures"],
                        c["rows"],
                        c["first_designation"],
                        c["changed_designation"],
                        c["unchanged_designation"],
                        c["cleared_designation"],
                        c["practice_changes"],
                        wilson(c["unchanged_designation"], repeated),
                    ]
                )
        captures = pd.DataFrame(self.captures)
        coverage = []
        for source, group in captures.groupby("source", sort=True):
            c = totals[source]
            coverage.append(
                [
                    source,
                    len(group),
                    group.capture_utc.min(),
                    group.capture_utc.max(),
                    c["empty_captures"],
                    c["native_modified_rows"],
                    "unknown; n=0 official pairs",
                ]
            )
        coverage.append(["Sportradar", 0, "--", "--", "--", "--", "no local snapshot"])
        verse = totals["nflverse"]
        verse_captures = [row for row in self.captures if row["source"] == "nflverse"]
        friday_latest = {}
        for row in verse_captures:
            if row["weeks"] and pd.Timestamp(row["capture_et"]).day_name() == "Friday":
                friday_latest[max(row["weeks"])] = row
        latest_week_counts = verse_captures[-1]["week_designations"]
        parts = [
            "# Friday designation freshness audit",
            f"**Measured:** archive frozen at {self.as_of.isoformat()}. "
            f"Command: .tools/uv.exe run --no-sync python scripts/lead64_freshness_audit.py "
            f"--as-of {self.as_of.isoformat()}.",
            "**Inferred decision:** no audited source has demonstrated a fresh final Friday team "
            "report "
            "at a Friday cutoff. Capture time dates the fetch, not the report. Retain the "
            "kickoff guard "
            "and defer an earlier freshness cutoff pending source-native evidence. This is a "
            "provenance "
            "finding, not a signal closure or an estimate of predictive value.",
            "## Fixed protocol and denominators",
            "Protocol saved in docs/lanes/lead64-friday-designations.md before measurement. "
            "One descriptive audit; zero outcome looks, fits, selected parameters, or folds. "
            "In-sample/out-of-sample gaps and probability_positive do not apply to this "
            "timestamp/identity "
            "census. No signal is rejected for an interval crossing zero.",
            f"All available captures through the freeze are inventoried. Tables use report "
            f"season {self.year}; "
            "bulk historical seasons are counted separately. Weekday is Eastern capture weekday. "
            "Weekly key: season/week/team/player. Headline key: URL/player, with no official "
            "report week. "
            "Inactive key: team/player; the parser's caller-assigned week is not source-native "
            "evidence. "
            "Rows include blank game statuses. All weeks in a cumulative release remain separate "
            "keys.",
            "First means a nonblank designation without one in the preceding capture, including "
            "the "
            "initial local baseline and blank-to-status changes; it is not publication time. "
            "Changed "
            "and Same require a nonblank designation in consecutive captures. Cleared is "
            "status-to-blank. "
            "Practice separately counts practice-status changes on matched rows. Reappearance "
            "after an "
            "absent row is separate in scratch output. Same/(Changed+Same) has a descriptive 95% "
            "Wilson "
            "interval; repeated captures are correlated, so these are not independent-game "
            "intervals. "
            "Unchanged does not itself mean stale.",
            "## Source and capture coverage",
            table(
                [
                    "Source",
                    "Captures/pages",
                    "First capture UTC",
                    "Last capture UTC",
                    "Empty",
                    "Rows with native update field",
                    "Official report-to-capture lag",
                ],
                coverage,
            ),
            "**Measured:** other-season weekly row observations: "
            + "; ".join(f"{s}: {n:,}" for s, n in sorted(self.history.items()))
            + ". "
            "NFL.com has historical backfill and empty current-season attempts. Historical "
            "date_modified and fetch times cannot measure current Friday latency.",
            "## Designation transitions by source and weekday",
            "**Measured:** capture-row observations, not unique players or games. Missing weekdays "
            "have no archived capture for that source.",
            table(
                [
                    "Source",
                    "ET weekday",
                    "Captures",
                    "Rows",
                    "First",
                    "Changed",
                    "Same",
                    "Cleared",
                    "Practice",
                    "Same share, 95% Wilson",
                ],
                rows,
            ),
            f"**Measured:** nflverse has {verse['rows']:,} current-season row observations, "
            f"{verse['native_modified_rows']:,} nonnull date_modified values. New-week repeated "
            f"designations among keys with an observed designated immediately preceding week: "
            f"{wilson(verse['new_week_same'], verse['new_week_comparable'])}. "
            "Recurrence across report weeks does not prove the previous report was reused.",
            "## Friday nflverse capture detail",
            "**Measured:** counts below include each week separately; blank-status practice rows "
            "are not final game designations. Current-week Friday counts may be lower "
            "than cumulative totals. These fetch clocks do not measure official publication lag.",
            table(
                ["Capture ET", "Game designations by report week"],
                [
                    [
                        row["capture_et"],
                        "; ".join(
                            f"W{week}: {count}" for week, count in row["week_designations"].items()
                        ),
                    ]
                    for row in self.captures
                    if row["source"] == "nflverse"
                    and pd.Timestamp(row["capture_et"]).day_name() == "Friday"
                ],
            ),
            table(
                ["Report week", "Last Friday capture ET", "Friday / latest designation counts"],
                [
                    [
                        week,
                        row["capture_et"],
                        f"{row['week_designations'][str(week)]} / {latest_week_counts[str(week)]}",
                    ]
                    for week, row in sorted(friday_latest.items())
                ],
            ),
            "**Measured:** the last Friday run is around 16:30 ET in each observed week. "
            "Later Friday coverage is unmeasured. This comparison uses the latest weekly "
            "count as a reference, not independently verified official completeness.",
            "## Source-native clock evidence and Friday trust",
            table(
                ["HTML source", "Archived pages", "Pages with timestamp markers"],
                [
                    [s, c["pages"], c["pages_with_timestamp_markers"]]
                    for s, c in sorted(self.html.items())
                ],
            ),
            "**Measured:** marker scan checks datePublished/dateModified, article publish/modify "
            "metadata, "
            "time elements, last-updated labels, report date/time and publication time. Matches "
            "retain "
            "bounded context in scratch output. Generic navigation dates do not date a team "
            "report. "
            "This scan does not establish the absence of every possible date-like string.",
            table(
                ["Source", "Trusted fresh at Friday cutoff?", "Limitation"],
                [
                    [
                        "nflverse weekly",
                        "Not demonstrated",
                        "Native season/week identity, but no verified official report time or "
                        "final-report revision identity",
                    ],
                    [
                        "NFL.com weekly",
                        "Not demonstrated",
                        "Week/season URL; current attempts empty; backfill cannot measure Friday "
                        "arrival",
                    ],
                    [
                        "NFL.com inactives",
                        "No Friday-designation evidence",
                        "A different release; archived primary pages lack usable current "
                        "inactive rows",
                    ],
                    [
                        "RotoWire inactives",
                        "Not demonstrated",
                        "No native list date/week/game identity; parser assigns season/week",
                    ],
                    [
                        "Headline archive",
                        "Provisional news only",
                        "lastmod is editorial modification, not team-report publication or a "
                        "complete designation list",
                    ],
                    ["Sportradar", "Unmeasured", "No local injury snapshots"],
                ],
            ),
            "**Read:** scripts/nflverse_injuries_ingest.py:117 stores fetch time and row schema; "
            "scripts/ingest_nflcom_injuries.py:188 stores requested week and fetch time; "
            "src/nfl_ats/inactives_capture.py:231 injects caller season/week. None establishes an "
            "official team publication clock. Official-report-to-capture lag has n=0 verified "
            "pairs "
            "for every source: median/range/interval are not estimable. No fetch/lastmod proxy "
            "is substituted.",
            "## Scheduler corroboration",
            f"**Measured:** {logs['snapshot_mentions']} distinct snapshot stamps in relevant "
            f"completion/skip "
            f"lines; {logs['matched_captures']}/{len(self.captures)} audited source captures "
            f"match a stamp. "
            f"Events span {logs.get('first')} to {logs.get('last')}. These include manual "
            f"reruns, not "
            "unique publications. Missing log matches do not invalidate manifests; successful "
            "fetches "
            "do not prove freshness.",
            table(
                ["Job family", "ET weekday", "OK", "Failures", "SKIP"],
                [
                    [
                        r["source"],
                        r["weekday"],
                        r.get("OK", 0),
                        sum(v for k, v in r.items() if k.startswith("FAIL")),
                        r.get("SKIP", 0),
                    ]
                    for r in logs["counts"]
                ],
            ),
            "## Season-ending IR question",
            f"**Measured:** injury archive {ir['injury_path']} joined to roster archive "
            f"{ir['roster_path']} "
            "by season/week/team/GSIS ID, regular season only. Denominator is limited to "
            "season/weeks "
            f"present in the injury archive; {ir['missing_roster_ids']} missing roster IDs "
            f"excluded; "
            "LA/JAC/WSH aliases normalized. RES is a reserve code, not an explicit "
            "season-ending-IR flag.",
            table(
                ["Week", "Roster code", "Player-weeks", "In weekly injury report, 95% Wilson"],
                [[w, s, n, wilson(k, n)] for w, s, n, k in ir["coverage"]],
            ),
            f"**Measured:** all overlapping archived seasons' reserve-coded report inclusion: "
            f"{wilson(ir['all_seasons_reserve_matches'], ir['all_seasons_reserve_rows'])}. "
            "This is same-week co-occurrence, not an exclusion policy or timestamped IR "
            "transition.",
            table(
                [
                    "Season-ending headline match",
                    "First locally seen UTC",
                    "Roster / injury-row presence",
                ],
                [
                    [
                        e["player"],
                        e["first_seen"],
                        "; ".join(
                            f"W{w['week']} {w['status']}: "
                            f"{'present' if w['in_report'] else 'absent'}"
                            for w in e["weeks"]
                        ),
                    ]
                    for e in ir["examples"]
                ],
            ),
            "**Measured:** examples require an explicit season-ending headline phrase, exact "
            "normalized "
            "roster-name match and local first capture before the roster snapshot. Headlines are "
            "**reported** news, not independently verified medical/transaction evidence. Earlier "
            "weeks "
            "are context, not evidence of when IR began. **Inferred:** intentional nflverse "
            "exclusion "
            "of season-ending IR remains unresolved. RES does not identify that population, roster "
            "snapshots do not timestamp its transitions, and omission cannot establish publisher "
            "intent. "
            "Do not treat absence as healthy/available or necessarily a data defect.",
            "## Proposed cutoff rule (not implemented)",
            "For configured Friday cutoff F in Eastern time, accept a Sunday-game final "
            "designation "
            "only when an archived source-native record names the intended team and game/week, "
            "identifies "
            "the final report/revision, and supplies a timezone-qualified official report time R "
            "such "
            "that R <= capture C <= F < kickoff. The report date must be the intended game's "
            "final-report "
            "day; use the actual reporting calendar for other game days. An authoritative "
            "final-report "
            "identity without a publication clock needs separate validation, absent from this "
            "archive.",
            "Missing/inconsistent identity or time means freshness unknown; retain previously "
            "authorized "
            "behavior. Never silently reuse last week's status or turn missing into available. "
            "An unchanged "
            "payload may be valid when independently dated for this report. First appearance "
            "only bounds "
            "local availability. No fixed maximum age in hours is supported by n=0 official-time "
            "pairs. "
            "Keep the kickoff guard; do not impose a narrower cutoff or serve headline-only "
            "statuses.",
            "## Reproduction and limits",
            "The script reads local manifests, captured payloads, weekly rosters and scheduler "
            "logs only. "
            "Inactive replay is diagnostic; no capture-store or served-card writes. Input "
            "hashes, capture "
            "coverage, clock candidates, transition counters, headline evidence and scheduler "
            "counts are "
            "saved to tests/scratch/codex/lead64_freshness_audit.json. No prediction rows or "
            "outcomes read. "
            "Scratch output is not a committed data artifact.",
            f"**Measured:** parser/input warnings: {len(self.errors)}. "
            + "; ".join(self.errors[:5]),
        ]
        return (
            "\n\n".join(
                part if part.startswith(("|", "#")) else textwrap.fill(part, width=96)
                for part in parts
            )
            + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit local Friday injury-report freshness without outcomes."
    )
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--report", type=Path, default=Path("docs/lead64_freshness_audit.md"))
    parser.add_argument(
        "--summary", type=Path, default=Path("tests/scratch/codex/lead64_freshness_audit.json")
    )
    args = parser.parse_args()
    as_of = stamp(args.as_of)
    if pd.isna(as_of):
        parser.error("--as-of must be an ISO timestamp")
    audit = Audit(as_of)
    audit.run()
    ir = audit.roster_check()
    logs = audit.scheduler_check()
    summary = {
        "as_of_utc": as_of.isoformat(),
        "inventory": audit.inventory,
        "captures": audit.captures,
        "transitions": [
            {"source": s, "weekday": d, **dict(c)} for (s, d), c in sorted(audit.stats.items())
        ],
        "historical_row_observations": dict(audit.history),
        "html_markers": {k: dict(v) for k, v in audit.html.items()},
        "html_candidates": audit.html_candidates,
        "season_ending_ir": ir,
        "scheduler": logs,
        "warnings": audit.errors,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    args.report.write_text(audit.report(ir, logs), encoding="utf-8")
    print(
        f"Audited {len(audit.captures)} source captures; {len(audit.inventory)} local inputs; "
        f"{len(audit.errors)} warnings."
    )
    print(f"Report: {args.report}; summary: {args.summary}")
    print(
        "Official team-report timestamp pairs: 0; no fits, outcomes, registry writes, or serving "
        "changes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
