"""Immutable-snapshot ingester for weekly NFL officiating-crew assignments.

Built for WP22, predeclared by `docs/referee_assignments_capture.md` (read
that file first: full source survey, publication-timing measurements, and
the join argument against the historical crew traits). `docs/referee_battery.md`
and `docs/penalty_crew_tendencies.md` found reliable crew-level penalty-rate
traits from `nflreadpy.load_officials()` (`data/raw/officials/*/officials.parquet`,
2015-2025) but nothing in this repo captures the UPCOMING week's officiating
assignment -- so those cells cannot be played or tracked prospectively. This
module is that capture.

Source (measured this session, 2026-09-01): Football Zebras
(`https://www.footballzebras.com/`) publishes one post per REG week titled
"Week N referee assignments" at a URL slug of the shape
`/{season}/{month}/week-{N}-referee-assignments-{season}/` -- MEASURED to be
unreliable to construct directly (Week 5, 2025's real slug was
`week-5-referee-assignments-5`, not `-2025`), so this module's PRIMARY
strategy is to discover the exact post URL from the site's own reverse-
chronological `https://www.footballzebras.com/category/assignments/` index
page (`find_week_url`), falling back to a direct URL guess
(`_guess_urls`, using the week's own schedule kickoff month) only when the
index does not (yet) list the week at all. No independent second SOURCE was
found: `operations.nfl.com` was checked and measured to carry no weekly
assignments page at all (its own `/robots.txt` 404s -- it is a client-
rendered Next.js app, not a page this fetch model can read), and web
searches for a structured third-party mirror (ProFootballTalk, VSiN,
SportsbookWire) returned only prose betting-preview mentions, never a
per-game listing -- see `docs/referee_assignments_capture.md` Section 1 for
the full, honestly-reported negative result. Robots.txt (measured) disallows
only `/wp-admin/` (with `admin-ajax.php` explicitly re-allowed) -- nothing
under `/category/` or `/{season}/...` is blocked.

Structure (measured against real 2025-season fetches, both saved verbatim
under `tests/fixtures/`): the post body contains a `assignment_list` block of
repeated `b_post` divs, each with a `b_post-game` cell ("Team at Team" or,
for a designated-home international game, "Team vs. Team" -- both forms list
the away team first, matching the schedule's own `game_id` convention, but
this module never trusts that ordering: `home_team`/`away_team`/`game_id`
are always resolved from the local schedule snapshot by the unordered team
PAIR, not by text position) and a `b_post-referee` cell (a plain "First
Last" name). MEASURED: the site emits BOTH single- and double-quoted class
attributes across different posts (`tests/fixtures/
footballzebras_week10_2025_referee_assignments.html` is double-quoted,
`footballzebras_week18_2025_referee_assignments_excerpt.html` is single-
quoted) and decorates some team names with a `<sup>seed</sup>` annotation or
a trailing `*` footnote marker in the season's final week -- both stripped
before nickname lookup.

Team-name -> team-code mapping (`NICKNAME_TO_CODE`), the HTML-stripping
helper (`strip_html`), and the current-week resolver
(`resolve_current_reg_week`) are IMPORTED VERBATIM from
`scripts/ingest_nflcom_injuries.py`, the same reuse this project's other
2026-09-01 capture (`src/nfl_ats/inactives_capture.py`) already makes for the
identical reason: `scripts` is not part of the installed package, and that
parser is the closest proven-real precedent for a "team nickname on a public
NFL page" -> repo team code join.

Referee-name join to the historical crew traits: `officials.parquet`'s own
`official_name` field (what `docs/referee_battery.md`'s and
`docs/penalty_crew_tendencies.md`'s flag builders key on,
`src/nfl_ats/experiment_runner.py`'s `_build_referee_trait_data`) and Football
Zebras' `b_post-referee` text use the same "First Last" convention.
MEASURED this session: of the 17 referees on Football Zebras' own
2026-season crew roster (`https://www.footballzebras.com/2026/08/
officiating-crews-for-the-2026-season/`), 16 match one of the 29 distinct
`official_name` values in `data/raw/officials/20260819T190537Z/
officials.parquet` (2015-2025, position="Referee") EXACTLY -- the lone miss
is "Ron Torbert" (Football Zebras) vs. "Ronald Torbert" (nflverse), a real
mismatch also present in-context in the week-10 fixture's own
"Bills at Dolphins" row. `REFEREE_NAME_ALIASES` maps that one known case
explicitly (not fuzzy-matched, so the join stays exact and auditable),
bringing the measured match rate to 17/17 (100%).

Point-in-time contract: every run writes a FRESH UTC-stamped snapshot
directory under `data/players/referee_assignments/<UTC ts>/` (raw HTML for
both the category index and the resolved post page, `assignments.parquet`,
`manifest.json`) -- never resumes or mutates an older one, matching
`scripts/ingest_player_arrests.py` and `src/nfl_ats/inactives_capture.py`.
Pregame-safety: officiating crew assignments are published by the league
before kickoff (the premise `docs/referee_battery.md` already argues from
for the historical PBP-joined construct); MEASURED across 10 sampled 2025
weeks (`docs/referee_assignments_capture.md` Section 2), Football Zebras'
own publish timestamp is NEVER before Tuesday afternoon and sometimes lands
Wednesday around midday -- so a captured assignment is usable for a
LATE-WEEK refresh (up to each game's own `min(kickoff, Sunday 16:00 ET)`
deadline) but essentially never for the Tuesday-lock/opener card. No
experiment is run by this module; it only captures.

`empty_reason` values (zero-row snapshot, `ok` as noted):
- `no_schedule_snapshot` / `no_upcoming_reg_kickoff` -- `--current` could not
  resolve a live (season, week) at all. Both genuine "nothing to capture yet"
  states, `ok=True`.
- `not_yet_published` -- the category index loaded fine and simply does not
  (yet) list this week's post, and no direct URL guess found it either. The
  EXPECTED state for an early-in-the-week or early-in-the-season run --
  MEASURED true right now (2026-09-01) for 2026 Week 1. `ok=True`.
- `unrecognized_page_structure` -- the index DID list a URL for this week,
  but fetching or parsing it yielded zero rows. This means the guessed
  markup is wrong and needs fixing against the real page -- `ok=False` so
  the scheduler's `FAIL(...)` status surfaces it.
- `primary_and_category_fetch_failed` -- the category index itself could not
  be fetched or robots.txt disallowed it. `ok=False`.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.robotparser
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nfl_ats.io import atomic_json, atomic_parquet  # noqa: E402
from nfl_ats.provenance import sha256_bytes, utc_now  # noqa: E402
from scripts.ingest_nflcom_injuries import (  # noqa: E402
    NICKNAME_TO_CODE,
    resolve_current_reg_week,
    strip_html,
)

CATEGORY_URL = "https://www.footballzebras.com/category/assignments/"
ROBOTS_URL = "https://www.footballzebras.com/robots.txt"
USER_AGENT = "nfl-ats-research-snapshot/1.0 (private research; contact: local repo owner)"
DEFAULT_DELAY_SECONDS = 2.5

# MEASURED this session (2026-09-01): 16 of 17 referees on Football Zebras'
# 2026 crew-roster page match officials.parquet's `official_name` verbatim;
# this is the lone documented exception. See module docstring.
REFEREE_NAME_ALIASES: dict[str, str] = {
    "ron torbert": "Ronald Torbert",
}

EMPTY_REASON_NO_SCHEDULE = "no_schedule_snapshot"
EMPTY_REASON_SEASON_COMPLETE = "no_upcoming_reg_kickoff"
EMPTY_REASON_NOT_YET_PUBLISHED = "not_yet_published"
EMPTY_REASON_UNRECOGNIZED_STRUCTURE = "unrecognized_page_structure"
EMPTY_REASON_FETCH_FAILED = "primary_and_category_fetch_failed"

PARQUET_COLUMNS = [
    "captured_at_utc",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "referee",
    "referee_source_name",
    "crew_number",
    "game_day_label",
    "source_url",
]

# Both quote styles are real: measured double-quoted on the week-10 2025
# fixture, single-quoted on week-18's. `b_post-time` divs are captured but
# discarded -- kickoff time/network is not part of this module's schema.
SUP_TAG = re.compile(r"<sup>.*?</sup>", re.DOTALL)
DAY_HEADER = re.compile(r"<h3>(.*?)</h3>", re.DOTALL)
BLOCK = re.compile(
    r"<div class=['\"]b_post['\"]>\s*"
    r"<div class=['\"]b_post-game['\"]>(?P<game>.*?)</div>\s*"
    r"<div class=['\"]b_post-referee['\"]>(?P<referee>.*?)</div>\s*"
    r"(?:<div class=['\"]b_post-time['\"]>.*?</div>\s*)*"
    r"</div>",
    re.DOTALL,
)
MATCHUP_SEPARATORS = (" vs. ", " at ")

FetchFn = Callable[[str, str], tuple[str | None, int | None, str | None, bool]]


def normalize_referee_name(raw: str) -> str:
    cleaned = strip_html(raw).strip()
    return REFEREE_NAME_ALIASES.get(cleaned.casefold(), cleaned)


def robots_allows(session: requests.Session, url: str, robots_url: str) -> bool:
    response = session.get(robots_url, timeout=60)
    response.raise_for_status()
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(response.text.splitlines())
    return parser.can_fetch("*", url)


def fetch_page(
    session: requests.Session, url: str, delay: float
) -> tuple[str | None, int | None, str | None]:
    error: str | None = None
    status: int | None = None
    for attempt in range(2):
        try:
            response = session.get(url, timeout=90)
            status = response.status_code
            if status == 200 and response.text:
                return response.text, 200, None
            error = f"http_{status}"
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt == 0:
            time.sleep(delay)
    return None, status, error


def _default_fetch(
    url: str, robots_url: str, *, session: requests.Session, delay: float
) -> tuple[str | None, int | None, str | None, bool]:
    try:
        allowed = robots_allows(session, url, robots_url)
    except requests.RequestException as exc:
        return None, None, f"robots_fetch_error: {type(exc).__name__}: {exc}", False
    if not allowed:
        return None, None, "robots_disallowed", False
    html_text, status, error = fetch_page(session, url, delay)
    return html_text, status, error, True


def _build_fetch_fn(delay: float) -> FetchFn:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    def fetch_fn(url: str, robots_url: str) -> tuple[str | None, int | None, str | None, bool]:
        return _default_fetch(url, robots_url, session=session, delay=delay)

    return fetch_fn


def _week_link_pattern(week: int) -> re.Pattern[str]:
    return re.compile(
        r'href="(https://www\.footballzebras\.com/(\d{4})/\d{2}/'
        rf"week-{week}-referee-assignments[^\"]*)\""
    )


def find_week_url(index_html: str, *, season: int, week: int) -> str | None:
    """The post URL for (season, week) per the category index, if listed.

    Prefers a URL whose own path year matches ``season`` (every REG-week post
    observed this session embeds the season, not the calendar publish year,
    in its path -- Week 18's Jan-kickoff games were posted in December under
    `/2025/12/...`). Falls back to the first match regardless of year only if
    no season-matching one exists, so a plausible link is still surfaced for
    inspection rather than silently discarded.
    """

    matches: list[tuple[str, str]] = _week_link_pattern(week).findall(index_html)
    for url, year in matches:
        if year == str(season):
            return url
    return matches[0][0] if matches else None


def _day_label_lookup(html_text: str) -> list[tuple[int, str]]:
    return [(m.start(), strip_html(m.group(1))) for m in DAY_HEADER.finditer(html_text)]


def _day_label_for(position: int, day_marks: list[tuple[int, str]]) -> str | None:
    label: str | None = None
    for mark_pos, mark_label in day_marks:
        if mark_pos <= position:
            label = mark_label
        else:
            break
    return label


def parse_assignment_page(
    html_text: str,
    *,
    season: int,
    week: int,
    source_url: str,
    fetched_at_utc: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse one Football Zebras weekly-assignments post.

    Returns ``(rows, warnings)``. A row is skipped (with a warning) if its
    referee cell is empty, its matchup text does not contain either " at " or
    " vs. ", or either side's nickname is not in ``NICKNAME_TO_CODE`` --
    every skip is recoverable information loss (one bad row), never a reason
    to fail the whole page.
    """

    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    day_marks = _day_label_lookup(html_text)

    for match in BLOCK.finditer(html_text):
        referee_raw = strip_html(match.group("referee"))
        if not referee_raw:
            warnings.append(f"empty referee cell in a {source_url} block")
            continue

        game_text = strip_html(SUP_TAG.sub("", match.group("game")))
        separator = next((sep for sep in MATCHUP_SEPARATORS if sep in game_text), None)
        if separator is None:
            warnings.append(f"unparseable matchup text {game_text!r} in {source_url}")
            continue

        left_raw, right_raw = (part.strip(" *") for part in game_text.split(separator, 1))
        left_code = NICKNAME_TO_CODE.get(left_raw.casefold())
        right_code = NICKNAME_TO_CODE.get(right_raw.casefold())
        if not left_code or not right_code:
            warnings.append(
                f"unresolved team nickname(s) {left_raw!r}/{right_raw!r} in {source_url}"
            )
            continue

        rows.append(
            {
                "season": season,
                "week": week,
                "team_code_a": left_code,
                "team_code_b": right_code,
                "referee_source_name": referee_raw,
                "referee": normalize_referee_name(referee_raw),
                "game_day_label": _day_label_for(match.start(), day_marks),
                "source_url": source_url,
                "fetched_at_utc": fetched_at_utc,
            }
        )
    return rows, warnings


def _week_kickoff_month(repo: Path, season: int, week: int) -> int | None:
    hits = sorted((repo / "data" / "raw").glob("*/schedules.parquet"))
    if not hits:
        return None
    sched = pd.read_parquet(hits[-1], columns=["season", "week", "game_type", "gameday"])
    sched = sched.loc[
        (sched["season"] == season) & (sched["week"] == week) & (sched["game_type"] == "REG")
    ]
    days = pd.to_datetime(sched["gameday"], errors="coerce").dropna()
    if days.empty:
        return None
    return int(days.min().month)


def _guess_urls(repo: Path, season: int, week: int) -> list[str]:
    """Defensive direct-URL fallback, used only when the category index does
    not (yet) list the week at all. MEASURED unreliable in general (a real
    slug can carry a numeric disambiguator instead of the season, e.g. 2025
    Week 5's actual slug was `week-5-referee-assignments-5`), so this is a
    best-effort second chance, not the primary discovery mechanism.
    """

    kickoff_month = _week_kickoff_month(repo, season, week)
    if kickoff_month is None:
        return []
    months = [kickoff_month]
    previous_month = kickoff_month - 1 or 12
    if previous_month != kickoff_month:
        months.append(previous_month)
    return [
        f"https://www.footballzebras.com/{season}/{month:02d}/"
        f"week-{week}-referee-assignments-{season}/"
        for month in months
    ]


def _schedule_pair_lookup(
    repo: Path, season: int, week: int
) -> dict[frozenset[str], tuple[str, str, str]]:
    """Unordered team-pair -> (game_id, home_team, away_team) for one REG week.

    Keyed by an unordered pair (not by the source page's "at"/"vs." reading
    order) because this module never trusts that ordering as authoritative --
    see the module docstring's international-game note.
    """

    hits = sorted((repo / "data" / "raw").glob("*/schedules.parquet"))
    if not hits:
        return {}
    sched = pd.read_parquet(
        hits[-1], columns=["season", "week", "game_type", "game_id", "home_team", "away_team"]
    )
    sched = sched.loc[
        (sched["season"] == season) & (sched["week"] == week) & (sched["game_type"] == "REG")
    ]
    lookup: dict[frozenset[str], tuple[str, str, str]] = {}
    for _, row in sched.iterrows():
        key = frozenset({str(row["home_team"]), str(row["away_team"])})
        lookup[key] = (str(row["game_id"]), str(row["home_team"]), str(row["away_team"]))
    return lookup


def run_capture(
    *,
    season: int | None,
    week: int | None,
    out_root: Path,
    repo: Path = REPO,
    delay: float = DEFAULT_DELAY_SECONDS,
    fetch: FetchFn | None = None,
    now: datetime | None = None,
) -> tuple[Path, bool]:
    """Fetch, parse and write one immutable referee-assignments snapshot.

    Returns ``(snapshot_dir, ok)``. ``ok`` is False only for the two branches
    a future session should treat as a bug to fix
    (``unrecognized_page_structure``, ``primary_and_category_fetch_failed``);
    every other outcome, including a genuinely empty "not published yet"
    snapshot, is a documented, expected zero-row success.
    """

    moment = now or datetime.now(UTC)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    fetched_at_utc = moment.strftime("%Y-%m-%dT%H:%M:%SZ")

    snapshot = out_root / stamp
    snapshot.mkdir(parents=True, exist_ok=True)

    def _write_empty(
        reason: str, *, ok: bool, schedule_error: str | None = None
    ) -> tuple[Path, bool]:
        frame = pd.DataFrame(columns=PARQUET_COLUMNS)
        atomic_parquet(frame, snapshot / "assignments.parquet")
        manifest: dict[str, Any] = {
            "schema": "referee_assignments_snapshot/1",
            "snapshot_id": snapshot.name,
            "captured_at_utc": fetched_at_utc,
            "season": season,
            "week": week,
            "source_used": "none",
            "row_count": 0,
            "teams_seen": [],
            "referees_seen": [],
            "empty_reason": reason,
            "schedule_error": schedule_error,
            "warnings": [],
            "user_agent": USER_AGENT,
            "delay_seconds": delay,
            "ok": ok,
            "generated_at_utc": utc_now(),
        }
        atomic_json(manifest, snapshot / "manifest.json")
        return snapshot, ok

    resolved_season = season
    resolved_week = week
    if resolved_season is None or resolved_week is None:
        try:
            resolved_season, resolved_week = resolve_current_reg_week(repo, pd.Timestamp(moment))
        except SystemExit as exc:
            error_text = str(exc)
            reason = (
                EMPTY_REASON_NO_SCHEDULE
                if error_text.startswith("no data/raw")
                else EMPTY_REASON_SEASON_COMPLETE
            )
            return _write_empty(reason, ok=True, schedule_error=error_text)

    fetch_fn: FetchFn = fetch if fetch is not None else _build_fetch_fn(delay)

    index_html, index_status, index_error, index_robots_ok = fetch_fn(CATEGORY_URL, ROBOTS_URL)
    index_sha: str | None = None
    if index_html is not None:
        index_sha = sha256_bytes(index_html.encode("utf-8"))
        (snapshot / "category_index.html").write_text(index_html, encoding="utf-8")

    week_url = (
        find_week_url(index_html, season=resolved_season, week=resolved_week)
        if index_html is not None
        else None
    )

    post_url: str | None = week_url
    post_html: str | None = None
    post_status: int | None = None
    post_error: str | None = None
    post_robots_ok: bool | None = None
    source_used = "none"
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    if week_url is not None:
        post_html, post_status, post_error, post_robots_ok = fetch_fn(week_url, ROBOTS_URL)
        if post_html is not None:
            rows, warnings = parse_assignment_page(
                post_html,
                season=resolved_season,
                week=resolved_week,
                source_url=week_url,
                fetched_at_utc=fetched_at_utc,
            )
            if rows:
                source_used = "category_index"
    else:
        for guess_url in _guess_urls(repo, resolved_season, resolved_week):
            g_html, g_status, g_error, g_robots_ok = fetch_fn(guess_url, ROBOTS_URL)
            post_url, post_status, post_error, post_robots_ok = (
                guess_url,
                g_status,
                g_error,
                g_robots_ok,
            )
            if g_html is not None:
                post_html = g_html
                g_rows, g_warnings = parse_assignment_page(
                    g_html,
                    season=resolved_season,
                    week=resolved_week,
                    source_url=guess_url,
                    fetched_at_utc=fetched_at_utc,
                )
                if g_rows:
                    rows = g_rows
                    warnings = g_warnings
                    source_used = "direct_guess"
                    warnings.append(
                        "category index did not list this week yet; a direct URL guess succeeded"
                    )
                    break

    if post_html is not None:
        (snapshot / "post.html").write_text(post_html, encoding="utf-8")

    empty_reason: str | None = None
    ok = True
    if not rows:
        if index_html is None:
            empty_reason = EMPTY_REASON_FETCH_FAILED
            ok = False
        elif week_url is not None:
            empty_reason = EMPTY_REASON_UNRECOGNIZED_STRUCTURE
            ok = False
        else:
            empty_reason = EMPTY_REASON_NOT_YET_PUBLISHED

    schedule_map = _schedule_pair_lookup(repo, resolved_season, resolved_week) if rows else {}
    for row in rows:
        key = frozenset({row["team_code_a"], row["team_code_b"]})
        game_id, home_team, away_team = schedule_map.get(key, (None, None, None))
        if game_id is None:
            warnings.append(
                f"no schedule match for team pair {sorted(key)} in season "
                f"{resolved_season} week {resolved_week}"
            )
        row["game_id"] = game_id
        row["home_team"] = home_team
        row["away_team"] = away_team
        row["crew_number"] = None
        row["captured_at_utc"] = row.pop("fetched_at_utc")
        del row["team_code_a"]
        del row["team_code_b"]

    frame = pd.DataFrame(rows, columns=PARQUET_COLUMNS)
    atomic_parquet(frame, snapshot / "assignments.parquet")

    manifest = {
        "schema": "referee_assignments_snapshot/1",
        "snapshot_id": snapshot.name,
        "captured_at_utc": fetched_at_utc,
        "season": resolved_season,
        "week": resolved_week,
        "source_used": source_used,
        "category_index": {
            "url": CATEGORY_URL,
            "robots_check": {"url": ROBOTS_URL, "allowed": index_robots_ok},
            "http_status": index_status,
            "error": index_error,
            "sha256": index_sha,
            "bytes": len(index_html.encode("utf-8")) if index_html else 0,
            "resolved_week_url": week_url,
        },
        "post": (
            {
                "url": post_url,
                "robots_check": {"url": ROBOTS_URL, "allowed": post_robots_ok},
                "http_status": post_status,
                "error": post_error,
                "sha256": sha256_bytes(post_html.encode("utf-8")) if post_html else None,
                "bytes": len(post_html.encode("utf-8")) if post_html else 0,
            }
            if post_url is not None
            else None
        ),
        "row_count": len(frame),
        "teams_seen": (
            sorted(
                {str(t) for t in frame["home_team"].dropna()}
                | {str(t) for t in frame["away_team"].dropna()}
            )
            if len(frame)
            else []
        ),
        "referees_seen": sorted(str(r) for r in frame["referee"].dropna().unique())
        if len(frame)
        else [],
        "empty_reason": empty_reason,
        "warnings": warnings[:50],
        "user_agent": USER_AGENT,
        "delay_seconds": delay,
        "ok": ok,
        "generated_at_utc": utc_now(),
    }
    atomic_json(manifest, snapshot / "manifest.json")
    return snapshot, ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPO / "data" / "players" / "referee_assignments"
    )
    parser.add_argument(
        "--current",
        action="store_true",
        help="resolve the live (season, REG week) from the schedules snapshot",
    )
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)
    args = parser.parse_args(argv)

    if args.delay < 2.0:
        raise SystemExit("delay must be >= 2 seconds (polite rate limiting)")
    if not args.current and (args.season is None or args.week is None):
        raise SystemExit("pass --current, or both --season and --week")

    season = None if args.current else args.season
    week = None if args.current else args.week

    snapshot, ok = run_capture(season=season, week=week, out_root=args.out, delay=args.delay)
    print(f"snapshot: {snapshot} (ok={ok})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
