"""Canonical historical officiating-crew table: nflverse feed + Wayback archive.

LEAD-59 (``ROADMAP.md``). Two disjoint sources describe who officiated an
NFL game, and until this module nothing joined them:

- ``data/raw/officials/<snapshot>/officials.parquet`` -- the nflverse
  officials feed, **2015-2025 only** (measured: its ``season`` column holds
  exactly 2015..2025). Every crew-tilt consumer reads this and only this
  (``nfl_ats.officials_flag_features``, ``nfl_ats.crew_tilt_refresh_overlay``,
  ``nfl_ats.experiment_runner._build_referee_trait_data`` /
  ``._build_referee_type_trait_data``).
- ``data/raw/officials_pfr_wayback/<run-id>/`` -- the polite Wayback sweep's
  immutable capture of Pro-Football-Reference boxscores
  (``scripts/officials_wayback_sweep.py``), one run directory per invocation,
  each holding ``manifest.json`` (schema ``officials_pfr_wayback_manifest/1``)
  and an ``html/`` directory. This is where the FULL seven-person crews for
  2009-2014 live, and nothing consumed it before this module
  (``docs/officials_coverage.md``, lane N).

What this module produces
-------------------------

1. :func:`canonical_crew_table` -- one row per game, wide: the seven on-field
   positions as their own columns, plus which sweep run, which Wayback
   capture, and which URL the row came from. Cross-run duplicates resolve by
   the sweep's own stated preference; every row is validated to join to
   EXACTLY ONE game in the schedule snapshot, and a game that joins to zero
   or to more than one fails closed.
2. :func:`archive_officials_long` -- the same crews reshaped into the
   nflverse feed's own 9-column long schema, keyed on the LEGACY numeric
   ``game_id`` the feed uses (crosswalked through ``schedules.parquet``'s
   ``old_game_id``), so it is drop-in comparable with the feed.
3. :func:`load_officials` -- the single loader every consumer calls. With
   ``include_archive=False`` (:data:`INCLUDE_ARCHIVE_DEFAULT`, the value
   shipped today) it returns the nflverse feed BIT-FOR-BIT, so wiring it in
   changes no production number. With ``include_archive=True`` it returns
   the feed's rows unchanged plus archive rows for games the feed does not
   carry -- **nflverse always wins on overlap**, at game granularity.

Turning the archive on is a separate, measured decision
-------------------------------------------------------

``include_archive=True`` does not change any 2015-2025 ROW (pinned
bit-for-bit in ``tests/test_officials_archive.py``), but it does widen the
POPULATION that every derived crew trait is computed over -- a referee's
``prior_seasons_experience``, the lagged penalty-rate quartile cutpoints,
``describe_referee_left_censoring``'s 2015 censoring count. Those are
supposed to change; that is the point of extending back to 2009. They are
also a card-affecting change, so the flip belongs to LEAD-59's own next step
("re-run the officials flags on production"), not to this loader's default.
:data:`INCLUDE_ARCHIVE_DEFAULT` is the one place to flip it.

Timing contract (read this before using an archive row for anything)
--------------------------------------------------------------------

The Wayback capture timestamp is always AFTER the game: the sweep queries
CDX with ``from=<gameday + 1 day>`` because a PFR boxscore captured before
kickoff is a placeholder page with no officials block at all
(``scripts/officials_wayback_sweep.py``, ``capture_not_before``). So the
archive can prove "this crew officiated this game" and CANNOT prove "this
assignment was published at time T before kickoff".

That is exactly the footing the existing family already stands on. The
nflverse officials feed is itself a post-hoc dataset with no capture
timestamp of any kind, and ``nfl_ats.experiment_runner``'s referee battery
is admissible because (a) crew IDENTITY is public before kickoff in
reality -- the league and Football Zebras publish assignments midweek
(``docs/referee_assignments_capture.md`` section 2) -- and (b) every TRAIT
built on top of it uses only the crew's PRIOR games. The archive is on the
same footing, and carries strictly more provenance than the feed it extends.
This family is therefore labelled :data:`ARCHIVE_TIMING_CLASS` --
``crew_identity_public_pregame_not_provably_captured_pregame`` -- and:

- **Admissible** for HISTORICAL/backtest features whose input is crew
  identity plus that crew's own prior-game history, on the same terms as the
  2015-2025 feed.
- **NOT admissible** for the prospective refresh channel. That channel
  (``nfl_ats.crew_tilt_refresh_overlay``, fed by ``referee_assignments_wed``)
  requires a snapshot whose ``captured_at_utc`` is strictly BEFORE each
  game's own ``min(kickoff, Sunday 16:00 ET)`` deadline. Every archive
  capture is after kickoff by construction, so it can never satisfy that
  test. :func:`load_officials_for_prospective_channel` and
  :func:`refuse_archive_rows` enforce this in code, not in prose, and
  :func:`assert_captures_are_post_game` fails closed if any archive row ever
  claims a capture at or before its own kickoff date (which would mean a
  corrupted manifest, since the sweep cannot produce one).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_ats.data import DataContractError

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The sweep's own ``SOURCE_ID`` (``scripts/officials_wayback_sweep.py``).
#: Retyped here so importing this module never requires loading the script;
#: ``tests/test_officials_archive.py`` pins the two strings equal.
ARCHIVE_SOURCE = "internet_archive_pfr_boxscores"
#: Provenance label for rows that came from the nflverse officials feed.
NFLVERSE_SOURCE = "nflverse_officials"

#: Only a directory whose ``manifest.json`` declares this schema is a sweep
#: run. The two ``laneN_probe_*`` directories declare
#: ``officials_pfr_wayback_probe/1`` instead and are skipped, not crashed on.
MANIFEST_SCHEMA = "officials_pfr_wayback_manifest/1"

#: Measured (2026-09-08, read-only pass over both real run directories):
#: exactly these seven position labels appear across all 9,724 parsed crew
#: rows. No "Replay Official" row has ever been parsed.
CORE_CREW_POSITIONS: tuple[str, ...] = (
    "Referee",
    "Umpire",
    "Head Linesman",
    "Line Judge",
    "Field Judge",
    "Side Judge",
    "Back Judge",
)

#: The NFL renamed "Head Linesman" to "Down Judge" for the 2017 season, after
#: this archive's 2009-2014 window; no captured page has ever used it. The
#: alias keeps the wide table correct if a future sweep window moves past
#: 2016, and matches ``scripts/officials_coverage_report.py``'s own alias.
POSITION_ALIASES: dict[str, str] = {"Down Judge": "Head Linesman"}

#: Wide-table column for each on-field position.
CREW_COLUMN_BY_POSITION: dict[str, str] = {
    "Referee": "referee",
    "Umpire": "umpire",
    "Head Linesman": "head_linesman",
    "Line Judge": "line_judge",
    "Field Judge": "field_judge",
    "Side Judge": "side_judge",
    "Back Judge": "back_judge",
}

CREW_COLUMNS: tuple[str, ...] = tuple(CREW_COLUMN_BY_POSITION[p] for p in CORE_CREW_POSITIONS)

#: :func:`canonical_crew_table`'s frozen column order.
CANONICAL_CREW_COLUMNS: tuple[str, ...] = (
    "game_id",
    "old_game_id",
    "season",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "pfr_id",
    *CREW_COLUMNS,
    "complete_crew",
    "n_crew_rows",
    "n_positions",
    "discarded_duplicate_position_rows",
    "source",
    "source_run_id",
    "wayback_capture_timestamp",
    "wayback_captured_at_utc",
    "wayback_url",
    "fetched_at_utc",
)

#: The nflverse officials feed's own 9 columns, in its own order (measured:
#: ``data/raw/officials/20260819T190537Z/officials.parquet``).
NFLVERSE_OFFICIALS_COLUMNS: tuple[str, ...] = (
    "game_id",
    "game_key",
    "official_name",
    "position",
    "jersey_number",
    "official_id",
    "season",
    "season_type",
    "week",
)

#: The archive holds REG games only -- ``officials_wayback_sweep.load_games``
#: filters ``game_type == "REG"`` before any fetch.
ARCHIVE_SEASON_TYPE = "REG"

#: Timing-contract label for the archive family. See the module docstring.
ARCHIVE_TIMING_CLASS = "crew_identity_public_pregame_not_provably_captured_pregame"

#: Shipped default for :func:`load_officials`. ``False`` == nflverse feed
#: only == today's production behaviour, bit-for-bit. Flipping this is
#: LEAD-59's own next step and must be measured on the played card first.
INCLUDE_ARCHIVE_DEFAULT = False

#: A live sweep replaces ``manifest.json`` atomically; a concurrent read can
#: transiently fail on Windows (measured once, 2026-09-07, WinError 5 --
#: ``docs/officials_archive_probe.md``). Same bounded retry
#: ``scripts/officials_coverage_report.py`` already uses.
_MANIFEST_READ_ATTEMPTS = 5
_MANIFEST_READ_RETRY_SECONDS = 0.05


class OfficialsArchiveError(DataContractError):
    """Raised when the officials archive violates its own contract."""


@dataclass(frozen=True)
class SweepRun:
    """One ``data/raw/officials_pfr_wayback/<run-id>/`` directory."""

    run_id: str
    path: Path
    capture_policy: str | None
    season_start: int | None
    season_end: int | None
    n_manifest_rows: int


# ---------------------------------------------------------------------------
# Parser reuse: the sweep's own ``parse_officials_block``, never a second copy
# ---------------------------------------------------------------------------


_SWEEP_MODULE_NAME = "nfl_ats_officials_wayback_sweep"
_SWEEP_MODULES: dict[str, Any] = {}


def _load_sweep_module(repo_root: Path) -> Any:
    """Import ``scripts/officials_wayback_sweep.py`` by path.

    ``scripts/`` is not part of the installed package, so this is a
    file-location import -- the same pattern ``nfl_ats.lockday_package`` and
    ``nfl_ats.ledger_reconcile`` already use, and it keeps the sweep out of
    ``mypy src``'s import graph. The parser is reused rather than
    reimplemented on purpose: it has been corrected twice against real
    captured pages (``id="ref_info"`` as well as ``id="officials"``, bolded
    position labels), and a second copy would silently drift.

    Unlike those two precedents the module is registered in ``sys.modules``
    before it executes: the sweep defines ``@dataclass`` types, and
    ``dataclasses`` resolves a field annotation through
    ``sys.modules[cls.__module__]``, which raises ``AttributeError`` for a
    module that was never registered.
    """

    path = repo_root / "scripts" / "officials_wayback_sweep.py"
    key = str(path)
    cached = _SWEEP_MODULES.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(_SWEEP_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise OfficialsArchiveError(f"cannot load the Wayback sweep parser from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SWEEP_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_SWEEP_MODULE_NAME, None)
        raise
    _SWEEP_MODULES[key] = module
    return module


def normalize_position(position: str) -> str:
    """Position label with the Down Judge / Head Linesman rename folded in."""

    return POSITION_ALIASES.get(position, position)


# ---------------------------------------------------------------------------
# Run discovery and crew-row parsing
# ---------------------------------------------------------------------------


def _read_manifest(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    last_error: Exception | None = None
    for attempt in range(_MANIFEST_READ_ATTEMPTS):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # transient atomic-replace collision
            last_error = exc
            if attempt + 1 < _MANIFEST_READ_ATTEMPTS:
                time.sleep(_MANIFEST_READ_RETRY_SECONDS)
            continue
        if not isinstance(payload, dict):
            return None
        return payload
    raise OfficialsArchiveError(
        f"could not read {manifest_path} after {_MANIFEST_READ_ATTEMPTS} attempts: {last_error}"
    )


def discover_sweep_runs(raw_root: Path) -> list[SweepRun]:
    """Every sweep run directory under ``raw_root``, oldest run id first.

    A directory whose ``manifest.json`` is missing, unreadable as an object,
    or does not declare :data:`MANIFEST_SCHEMA` is skipped -- that is how the
    two ``laneN_probe_*`` diagnostic directories stay out of the table.
    """

    if not raw_root.is_dir():
        return []
    runs: list[SweepRun] = []
    for run_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        payload = _read_manifest(run_dir)
        if payload is None or payload.get("schema") != MANIFEST_SCHEMA:
            continue
        games = payload.get("games")
        if not isinstance(games, list):
            continue
        runs.append(
            SweepRun(
                run_id=str(payload.get("run_id") or run_dir.name),
                path=run_dir,
                capture_policy=(
                    str(payload["capture_policy"])
                    if payload.get("capture_policy") is not None
                    else None
                ),
                season_start=(
                    int(payload["season_start"])
                    if payload.get("season_start") is not None
                    else None
                ),
                season_end=(
                    int(payload["season_end"]) if payload.get("season_end") is not None else None
                ),
                n_manifest_rows=len(games),
            )
        )
    return runs


def default_archive_root(repo_root: Path | None = None) -> Path:
    return (repo_root or REPO_ROOT) / "data" / "raw" / "officials_pfr_wayback"


#: ``(raw_root, fingerprint) -> parsed crew rows``. Parsing ~1,400 captured
#: pages is pure disk work, so the result is memoised on a fingerprint of
#: every manifest's size and mtime -- a live sweep writing a new row
#: invalidates it on the next call. :func:`clear_cache` empties it.
_CREW_ROW_CACHE: dict[tuple[str, tuple[tuple[str, int, int], ...]], pd.DataFrame] = {}


def _manifest_fingerprint(runs: list[SweepRun]) -> tuple[tuple[str, int, int], ...]:
    fingerprint: list[tuple[str, int, int]] = []
    for run in runs:
        stat = (run.path / "manifest.json").stat()
        fingerprint.append((run.run_id, stat.st_size, stat.st_mtime_ns))
    return tuple(fingerprint)


def clear_cache() -> None:
    """Drop the parsed-crew-row memo (tests, and any caller that must re-read)."""

    _CREW_ROW_CACHE.clear()


#: :func:`load_archive_crew_rows`'s frozen column order.
ARCHIVE_CREW_ROW_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "pfr_id",
    "position",
    "raw_position",
    "official_name",
    "row_order",
    "source",
    "source_run_id",
    "wayback_capture_timestamp",
    "wayback_url",
    "fetched_at_utc",
    "html_file",
)


def load_archive_crew_rows(
    *, repo_root: Path | None = None, raw_root: Path | None = None, use_cache: bool = True
) -> pd.DataFrame:
    """Long crew rows -- one per (game, parsed position) -- across every run.

    Disk is the only truth: a manifest row counts only when the ``html_file``
    it names actually exists (a ``*_failed`` outcome label can still carry a
    page kept from an earlier attempt, and an in-flight sweep can name a page
    it has not finished writing). ``row_order`` preserves the page's own
    ordering so duplicate-position rows resolve deterministically downstream.
    """

    root = repo_root or REPO_ROOT
    archive_root = raw_root if raw_root is not None else default_archive_root(root)
    runs = discover_sweep_runs(archive_root)

    cache_key = (str(archive_root), _manifest_fingerprint(runs))
    if use_cache and cache_key in _CREW_ROW_CACHE:
        return _CREW_ROW_CACHE[cache_key].copy()

    sweep = _load_sweep_module(root)
    records: list[dict[str, Any]] = []
    for run in runs:
        payload = _read_manifest(run.path)
        if payload is None:
            continue
        for game_row in payload.get("games", []):
            html_file = game_row.get("html_file")
            if not html_file:
                continue
            html_path = run.path / str(html_file)
            if not html_path.is_file():
                continue
            html_text = html_path.read_text(encoding="utf-8", errors="replace")
            parsed, _warnings = sweep.parse_officials_block(html_text)
            for order, (position, official_name) in enumerate(parsed):
                records.append(
                    {
                        "game_id": str(game_row.get("game_id")),
                        "season": game_row.get("season"),
                        "week": game_row.get("week"),
                        "gameday": str(game_row.get("gameday")),
                        "home_team": game_row.get("home_team"),
                        "away_team": game_row.get("away_team"),
                        "pfr_id": str(game_row.get("pfr_id")),
                        "position": normalize_position(str(position)),
                        "raw_position": str(position),
                        "official_name": str(official_name),
                        "row_order": order,
                        "source": ARCHIVE_SOURCE,
                        "source_run_id": run.run_id,
                        "wayback_capture_timestamp": game_row.get("wayback_capture_timestamp"),
                        "wayback_url": game_row.get("wayback_url"),
                        "fetched_at_utc": game_row.get("fetch_instant_utc"),
                        "html_file": str(html_file),
                    }
                )

    frame = pd.DataFrame(records, columns=list(ARCHIVE_CREW_ROW_COLUMNS))
    if use_cache:
        _CREW_ROW_CACHE[cache_key] = frame.copy()
    return frame


# ---------------------------------------------------------------------------
# Timing contract
# ---------------------------------------------------------------------------


def _as_text(value: Any) -> str | None:
    """``value`` as text, or ``None`` for anything that is not text-like.

    Manifest rows are JSON, so a timestamp arrives as ``str``; a missing one
    arrives as ``None``. Anything else (a float NaN from a round-trip through
    pandas, a bool) becomes ``None`` and therefore fails the post-game check
    rather than silently parsing.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return None


def _text_series(series: pd.Series) -> pd.Series:
    return pd.Series([_as_text(v) for v in series], index=series.index, dtype="str")


def _capture_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(_text_series(series), format="%Y%m%d%H%M%S", errors="coerce")


def assert_captures_are_post_game(frame: pd.DataFrame) -> None:
    """Fail closed unless every archive capture lands AFTER its game's day.

    The bar is the sweep's own: ``officials_wayback_sweep.capture_not_before``
    sets the CDX ``from=`` bound to ``gameday + 1 day``, so a genuine capture
    is never dated on or before the day of the game. A same-day capture is
    refused too -- a boxscore page captured at 10am on game day is a pre-game
    placeholder, and nothing in the timestamp distinguishes it from an
    evening capture. A violation therefore means the manifest was edited or
    corrupted, and the row would be fabricated crew data rather than early
    knowledge. Malformed or missing capture timestamps fail the same way.
    """

    if frame.empty:
        return
    captured = _capture_datetime(frame["wayback_capture_timestamp"])
    gameday = pd.to_datetime(_text_series(frame["gameday"]), errors="coerce")
    bad = captured.isna() | gameday.isna() | (captured < gameday + pd.Timedelta(days=1))
    if bool(bad.any()):
        offenders = sorted(set(frame.loc[bad, "game_id"].astype("str")))[:5]
        raise OfficialsArchiveError(
            "officials archive capture timestamps must land after the game's own day "
            f"(timing class {ARCHIVE_TIMING_CLASS}); offending game_ids: {', '.join(offenders)}"
        )


def refuse_archive_rows(frame: pd.DataFrame, *, channel: str) -> None:
    """Fail closed if a frame carrying archive rows reaches a prospective channel.

    A frame with no ``source`` column is the raw nflverse feed and passes.
    """

    if "source" not in frame.columns:
        return
    n_archive = int((frame["source"].astype("str") == ARCHIVE_SOURCE).sum())
    if n_archive:
        raise OfficialsArchiveError(
            f"{n_archive} Wayback-archive officials rows were offered to the prospective "
            f"channel {channel!r}; the archive's timing class is {ARCHIVE_TIMING_CLASS}, so "
            "its captures are always after kickoff and can never satisfy a "
            "captured-before-the-pick-deadline test"
        )


# ---------------------------------------------------------------------------
# Canonical per-game crew table
# ---------------------------------------------------------------------------


def latest_schedules_path(repo_root: Path | None = None) -> Path:
    """Newest ``data/raw/*/schedules.parquet``.

    Same glob ``nfl_ats.weak_stack_v3_features.latest_schedules_snapshot``
    and ``nfl_ats.experiment_runner._latest_schedules_snapshot`` both use;
    duplicated here for the same anti-cycle reason as
    :func:`latest_nflverse_officials_path` -- ``experiment_runner`` imports
    this module, so this module imports nothing from the feature packages.
    """

    root = repo_root or REPO_ROOT
    candidates = sorted((root / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise OfficialsArchiveError(f"no data/raw/*/schedules.parquet snapshot found under {root}")
    return candidates[-1]


def _default_schedules(repo_root: Path) -> pd.DataFrame:
    return pd.read_parquet(latest_schedules_path(repo_root))


_SCHEDULE_REQUIRED = ("game_id", "old_game_id", "season", "week", "gameday")


def _resolve_cross_run_duplicates(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep one run per ``game_id``: the newest Wayback capture wins.

    This extends the sweep's own intra-game preference (``capture_policy``
    ``newest_post_game_capture_with_fallback``) to the cross-run case, the
    way ``scripts/officials_coverage_report.py`` already resolves it for
    reporting. Ties break on the earlier run id, so the choice is total and
    deterministic. Measured 2026-09-08: no real game is captured by two runs
    (the two real runs cover disjoint season windows), so this path is
    exercised only by tests today.
    """

    if rows.empty:
        return rows, []
    per_game_run = (
        rows.groupby(["game_id", "source_run_id"], dropna=False)["wayback_capture_timestamp"]
        .max()
        .reset_index()
    )
    duplicated = per_game_run.loc[per_game_run["game_id"].duplicated(keep=False), "game_id"]
    duplicate_games = sorted(set(duplicated.astype("str")))
    if not duplicate_games:
        return rows, []
    per_game_run = per_game_run.sort_values(
        ["game_id", "wayback_capture_timestamp", "source_run_id"], ascending=[True, False, True]
    )
    winners = per_game_run.drop_duplicates("game_id")[["game_id", "source_run_id"]]
    kept = rows.merge(winners, on=["game_id", "source_run_id"], how="inner")
    return kept.reset_index(drop=True), duplicate_games


def _validate_schedule_join(rows: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_SCHEDULE_REQUIRED).difference(schedules.columns))
    if missing:
        raise OfficialsArchiveError(f"schedules snapshot is missing columns: {', '.join(missing)}")
    lookup = schedules.loc[:, list(_SCHEDULE_REQUIRED)].copy()
    lookup["game_id"] = lookup["game_id"].astype("str")
    counts = lookup.groupby("game_id").size()

    archive_games = sorted(set(rows["game_id"].astype("str")))
    unmatched = [g for g in archive_games if g not in counts.index]
    if unmatched:
        raise OfficialsArchiveError(
            "officials archive rows do not join to any schedule game: "
            f"{', '.join(unmatched[:5])} ({len(unmatched)} total)"
        )
    ambiguous = [g for g in archive_games if int(counts.loc[g]) > 1]
    if ambiguous:
        raise OfficialsArchiveError(
            "officials archive rows join to more than one schedule game: "
            f"{', '.join(ambiguous[:5])} ({len(ambiguous)} total)"
        )
    return lookup.drop_duplicates("game_id")


def canonical_crew_table(
    *,
    repo_root: Path | None = None,
    raw_root: Path | None = None,
    rows: pd.DataFrame | None = None,
    schedules: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per archived game, seven crew positions as columns.

    Columns are :data:`CANONICAL_CREW_COLUMNS`. ``season``, ``week``,
    ``gameday``, ``old_game_id`` come from the schedule snapshot (the
    authority), never from the manifest. Games are ordered by season, week,
    game_id.

    Duplicate handling, both measured on the real archive 2026-09-08:

    - **Across runs**: newest Wayback capture wins
      (:func:`_resolve_cross_run_duplicates`); zero real cases today.
    - **Within one page**: PFR occasionally lists a position twice with a
      variant name (``2014_06_DET_MIN`` shows "John Parry" and "John Perry"
      as Referee; ``2014_09_PHI_HOU`` shows two Back Judges). The FIRST row
      in page order wins and ``discarded_duplicate_position_rows`` counts
      what was dropped, so a duplicated Referee can never double-count a
      game downstream.

    ``rows`` / ``schedules``, when given (tests), are used instead of
    reading real snapshots.
    """

    root = repo_root or REPO_ROOT
    crew = rows if rows is not None else load_archive_crew_rows(repo_root=root, raw_root=raw_root)
    if crew.empty:
        return pd.DataFrame(columns=list(CANONICAL_CREW_COLUMNS))

    assert_captures_are_post_game(crew)
    crew, _duplicate_games = _resolve_cross_run_duplicates(crew)

    sched = schedules if schedules is not None else _default_schedules(root)
    lookup = _validate_schedule_join(crew, sched)

    crew = crew.sort_values(["game_id", "row_order"]).reset_index(drop=True)
    deduped = crew.drop_duplicates(["game_id", "position"], keep="first")
    core = deduped.loc[deduped["position"].isin(CORE_CREW_POSITIONS)]
    if core.empty:
        return pd.DataFrame(columns=list(CANONICAL_CREW_COLUMNS))
    discarded = (crew.groupby("game_id").size() - deduped.groupby("game_id").size()).rename(
        "discarded_duplicate_position_rows"
    )

    wide = core.pivot(index="game_id", columns="position", values="official_name").rename(
        columns=CREW_COLUMN_BY_POSITION
    )
    for column in CREW_COLUMNS:
        if column not in wide.columns:
            wide[column] = pd.Series(pd.NA, index=wide.index, dtype="str")
    wide = wide.loc[:, list(CREW_COLUMNS)]

    per_game = crew.drop_duplicates("game_id").set_index("game_id")
    table = wide.join(
        per_game.loc[
            :,
            [
                "home_team",
                "away_team",
                "pfr_id",
                "source",
                "source_run_id",
                "wayback_capture_timestamp",
                "wayback_url",
                "fetched_at_utc",
            ],
        ]
    )
    table["n_crew_rows"] = crew.groupby("game_id").size()
    table["n_positions"] = deduped.groupby("game_id")["position"].nunique()
    table["discarded_duplicate_position_rows"] = discarded.astype("int64")
    table["complete_crew"] = table.loc[:, list(CREW_COLUMNS)].notna().all(axis=1)
    table["wayback_captured_at_utc"] = _capture_datetime(table["wayback_capture_timestamp"])
    table = table.reset_index()

    table = table.merge(
        lookup.loc[:, ["game_id", "old_game_id", "season", "week", "gameday"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    table["season"] = table["season"].astype("int32")
    table["week"] = table["week"].astype("int32")
    table["gameday"] = table["gameday"].astype("str")
    table = table.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    return table.loc[:, list(CANONICAL_CREW_COLUMNS)]


# ---------------------------------------------------------------------------
# nflverse-schema long view of the archive
# ---------------------------------------------------------------------------


def archive_officials_long(
    *,
    repo_root: Path | None = None,
    raw_root: Path | None = None,
    rows: pd.DataFrame | None = None,
    schedules: pd.DataFrame | None = None,
    table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The archive reshaped into the nflverse feed's own long schema.

    ``game_id`` is the LEGACY numeric id (``schedules.old_game_id``), because
    that is what ``officials.parquet`` carries and what every consumer
    crosswalks on. ``game_key``, ``jersey_number`` and ``official_id`` are
    NOT available from a PFR boxscore and are left missing rather than
    invented. Columns are :data:`NFLVERSE_OFFICIALS_COLUMNS` plus ``source``.
    """

    wide = (
        table
        if table is not None
        else canonical_crew_table(
            repo_root=repo_root, raw_root=raw_root, rows=rows, schedules=schedules
        )
    )
    if wide.empty:
        return _empty_merged_frame()

    long = wide.melt(
        id_vars=["old_game_id", "season", "week"],
        value_vars=list(CREW_COLUMNS),
        var_name="crew_column",
        value_name="official_name",
    )
    long = long.loc[long["official_name"].notna()].copy()
    column_to_position = {v: k for k, v in CREW_COLUMN_BY_POSITION.items()}
    long["position"] = long["crew_column"].map(column_to_position)
    long["game_id"] = long["old_game_id"].astype("str")
    long["game_key"] = pd.Series(pd.NA, index=long.index, dtype="str")
    long["jersey_number"] = pd.Series(pd.NA, index=long.index, dtype="Int32")
    long["official_id"] = pd.Series(pd.NA, index=long.index, dtype="str")
    long["season_type"] = ARCHIVE_SEASON_TYPE
    long["source"] = ARCHIVE_SOURCE
    long["official_name"] = long["official_name"].astype("str")
    long["position"] = long["position"].astype("str")
    long["season_type"] = long["season_type"].astype("str")
    long["season"] = long["season"].astype("int32")
    long["week"] = long["week"].astype("int32")

    ordered = long.loc[:, [*NFLVERSE_OFFICIALS_COLUMNS, "source"]]
    position_rank = {name: index for index, name in enumerate(CORE_CREW_POSITIONS)}
    ordered = ordered.assign(_rank=ordered["position"].map(position_rank))
    ordered = ordered.sort_values(["season", "week", "game_id", "_rank"]).drop(columns="_rank")
    return ordered.reset_index(drop=True)


def _empty_merged_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": pd.Series([], dtype="str"),
            "game_key": pd.Series([], dtype="str"),
            "official_name": pd.Series([], dtype="str"),
            "position": pd.Series([], dtype="str"),
            "jersey_number": pd.Series([], dtype="Int32"),
            "official_id": pd.Series([], dtype="str"),
            "season": pd.Series([], dtype="int32"),
            "season_type": pd.Series([], dtype="str"),
            "week": pd.Series([], dtype="int32"),
            "source": pd.Series([], dtype="str"),
        }
    )


# ---------------------------------------------------------------------------
# The single loader every consumer calls
# ---------------------------------------------------------------------------


def latest_nflverse_officials_path(repo_root: Path | None = None) -> Path:
    """Newest ``data/raw/officials/*/officials.parquet``.

    Same glob ``nfl_ats.experiment_runner._latest_officials_snapshot`` uses;
    duplicated here (rather than imported) only to keep this module free of
    an import edge back into ``experiment_runner``, which imports this one.
    """

    root = repo_root or REPO_ROOT
    candidates = sorted((root / "data" / "raw" / "officials").glob("*/officials.parquet"))
    if not candidates:
        raise OfficialsArchiveError(
            f"no data/raw/officials/*/officials.parquet snapshot found under {root}"
        )
    return candidates[-1]


def load_officials(
    repo_root: Path | None = None,
    *,
    officials_path: Path | None = None,
    feed: pd.DataFrame | None = None,
    include_archive: bool = INCLUDE_ARCHIVE_DEFAULT,
    raw_root: Path | None = None,
    rows: pd.DataFrame | None = None,
    schedules: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The officials table every crew consumer reads.

    ``include_archive=False`` returns the nflverse feed EXACTLY as it sits on
    disk -- same rows, same order, same dtypes, same 9 columns -- so routing
    a consumer through this function changes nothing.

    ``include_archive=True`` appends the Wayback archive's crews for games
    the feed does not carry, in the feed's own schema, with a ``source``
    column added to both halves. **nflverse wins on overlap**: any archive
    game whose legacy ``game_id`` already appears in the feed is dropped
    whole, never merged row-by-row. The feed's own rows keep their order and
    values; the single schema difference is ``jersey_number`` widening from
    ``int32`` to the nullable ``Int32``, because a PFR boxscore carries no
    jersey numbers -- every 2015-2025 jersey number is unchanged and
    non-null, pinned in ``tests/test_officials_archive.py``.
    """

    root = repo_root or REPO_ROOT
    if feed is None:
        feed = pd.read_parquet(officials_path or latest_nflverse_officials_path(root))
    if not include_archive:
        # Pure pass-through, deliberately unvalidated: each consumer already
        # states the columns IT needs (``_build_referee_trait_data`` raises
        # its own ``ExperimentRunnerError``), and several tests build a
        # minimal officials fixture with only those columns. Validating here
        # would reject a fixture the consumer is happy with.
        return feed

    missing = sorted(set(NFLVERSE_OFFICIALS_COLUMNS).difference(feed.columns))
    if missing:
        raise OfficialsArchiveError(
            f"nflverse officials feed is missing columns: {', '.join(missing)}"
        )
    archive = archive_officials_long(
        repo_root=root, raw_root=raw_root, rows=rows, schedules=schedules
    )
    feed_side = feed.copy()
    feed_side["jersey_number"] = feed_side["jersey_number"].astype("Int32")
    feed_side["source"] = pd.Series(NFLVERSE_SOURCE, index=feed_side.index, dtype="str")
    feed_side = feed_side.loc[:, [*NFLVERSE_OFFICIALS_COLUMNS, "source"]]
    if archive.empty:
        return feed_side.reset_index(drop=True)

    overlap = archive["game_id"].astype("str").isin(set(feed_side["game_id"].astype("str")))
    archive = archive.loc[~overlap]
    if archive.empty:
        return feed_side.reset_index(drop=True)
    return pd.concat([feed_side, archive], ignore_index=True)


def load_officials_for_prospective_channel(
    repo_root: Path | None = None,
    *,
    officials_path: Path | None = None,
    feed: pd.DataFrame | None = None,
    channel: str = "crew_tilt_refresh",
) -> pd.DataFrame:
    """Officials for a channel that must prove pregame capture timing.

    The archive is never included here, and :func:`refuse_archive_rows`
    re-checks the result so a future edit cannot quietly widen it. See the
    module docstring's timing contract: every Wayback capture is after
    kickoff, so no archive row can satisfy the refresh path's
    ``captured_at_utc < min(kickoff, Sunday 16:00 ET)`` test.
    """

    frame = load_officials(
        repo_root, officials_path=officials_path, feed=feed, include_archive=False
    )
    refuse_archive_rows(frame, channel=channel)
    return frame


# ---------------------------------------------------------------------------
# Coverage description (read-only; for docs and the session report)
# ---------------------------------------------------------------------------


def describe_archive_coverage(
    *,
    repo_root: Path | None = None,
    raw_root: Path | None = None,
    table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Per-season archive coverage from the canonical table."""

    wide = (
        table if table is not None else canonical_crew_table(repo_root=repo_root, raw_root=raw_root)
    )
    if wide.empty:
        return {"n_games": 0, "seasons": {}, "timing_class": ARCHIVE_TIMING_CLASS}
    by_season: dict[str, Any] = {}
    for season, group in wide.groupby("season"):
        by_season[str(cast(int, season))] = {
            "n_games": len(group),
            "n_complete_crews": int(group["complete_crew"].sum()),
            "n_distinct_referees": int(group["referee"].dropna().nunique()),
            "n_discarded_duplicate_position_rows": int(
                group["discarded_duplicate_position_rows"].sum()
            ),
        }
    return {
        "n_games": len(wide),
        "n_complete_crews": int(wide["complete_crew"].sum()),
        "seasons": by_season,
        "timing_class": ARCHIVE_TIMING_CLASS,
        "include_archive_default": INCLUDE_ARCHIVE_DEFAULT,
    }


__all__ = [
    "ARCHIVE_CREW_ROW_COLUMNS",
    "ARCHIVE_SEASON_TYPE",
    "ARCHIVE_SOURCE",
    "ARCHIVE_TIMING_CLASS",
    "CANONICAL_CREW_COLUMNS",
    "CORE_CREW_POSITIONS",
    "CREW_COLUMNS",
    "CREW_COLUMN_BY_POSITION",
    "INCLUDE_ARCHIVE_DEFAULT",
    "MANIFEST_SCHEMA",
    "NFLVERSE_OFFICIALS_COLUMNS",
    "NFLVERSE_SOURCE",
    "POSITION_ALIASES",
    "OfficialsArchiveError",
    "SweepRun",
    "archive_officials_long",
    "assert_captures_are_post_game",
    "canonical_crew_table",
    "clear_cache",
    "default_archive_root",
    "describe_archive_coverage",
    "discover_sweep_runs",
    "latest_nflverse_officials_path",
    "latest_schedules_path",
    "load_archive_crew_rows",
    "load_officials",
    "load_officials_for_prospective_channel",
    "normalize_position",
    "refuse_archive_rows",
]
