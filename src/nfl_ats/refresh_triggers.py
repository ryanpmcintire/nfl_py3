"""Timing-policy instrumentation (ENG-08): detect real, non-clock refresh
triggers and score them prospectively against the fixed-clock checkpoints.

**Binding closing-grounds taxonomy (AGENTS.md), restated verbatim per this
project's rule for any module that scores or adjudicates an experiment:** an
interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is
the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is ``unresolved_below_power``: record it with ``nfl-ats weak-signals
record``, report ``probability_positive``, never the binary "contains zero."
Within-week correlation is ZERO by owner mandate -- pairing below blocks by
WEEK, never estimates or pads a within-week correlation, and this module
never computes "games needed".

This module is INSTRUMENTATION ONLY. It does not adjudicate anything today
and never writes to ``registry/`` -- :func:`compare_trigger_vs_checkpoint`
is a scaffold callers run once real ledger rows exist; it accepts rows and
triggers as plain arguments and never reads or writes a registry file
itself.

What this is, and what it is not
---------------------------------
The project already has one refresh-trigger mechanism: MKT-08's
``trigger_type``/``trigger_source``/``trigger_observed_at_utc`` columns on
``nfl_ats.pick_refresh``'s pick-revision ledger
(``PICK_REVISION_COLUMNS``, ``TRIGGER_CLOCK_DISPATCH`` /
``TRIGGER_NEWS_EVENT`` / ``TRIGGER_UNKNOWN``). That mechanism records
provenance for a refresh pass a human or the scheduler actually RAN, tagged
by whoever invoked ``refresh-picks --trigger-type ... --trigger-source
...``. It has no automatic detector: nothing notices, on its own, that an
inactives list posted, an injury report revised, a projected lineup changed,
or the market moved.

This module is that detector. It reads the capture directories already on
disk (WP17's ``nfl_ats.inactives_capture`` snapshots, the nflverse/Sportradar
injury archives, the lineup-forecast artifact, the market-quote store) and
reconstructs :class:`RefreshTrigger` rows for the four REAL non-clock events
this project can observe pregame, plus the fixed clock checkpoints
themselves (read from the scheduler's own state file, never re-derived from
wall-clock math) -- so the two can be compared on equal footing. Every
timestamp on a :class:`RefreshTrigger` other than ``observation_time`` (the
instant THIS SCAN ran) comes from the underlying snapshot's own manifest or
payload, never from the scanning process's clock -- this is what makes a
"trigger" a fact about the world rather than a fact about when someone
happened to look.

``trigger_source`` here is deliberately more granular than MKT-08's
``trigger_type``: every non-clock value below
(``inactives_posted``/``injury_report_posted``/``lineup_change``/
``line_move``) is a species of MKT-08's coarser ``TRIGGER_NEWS_EVENT``.
:func:`mkt08_trigger_type` maps between the two vocabularies using MKT-08's
own constants (imported, never redefined) so a future step that DOES record
to the pick-revision ledger can carry this module's finer detail through
MKT-08's existing ``trigger_source`` free-text field without inventing a
second one.

Deadline validation
--------------------
Owner rule (binding, restated from ``AGENTS.md``/memory): a game's pick
deadline is ``min(own kickoff, Sunday 16:00 ET of that week)`` -- Sunday
night and Monday games lock EARLY, at the same Sunday-afternoon instant as
the rest of the week, not at their own kickoff. This module never redefines
that arithmetic: every deadline here is
``nfl_ats.pick_refresh.pick_deadline(kickoff, nfl_ats.pick_refresh.sunday_pick_lock(...))``,
imported directly. A trigger whose ``source_capture_time`` is at or after
its game's deadline is ``deadline_valid=False``, tagged ``deadline_violation``
in :attr:`RefreshTrigger.deadline_reason`, and excluded from
:func:`compare_trigger_vs_checkpoint`'s paired population -- a refresh this
project could never actually have acted on must never contribute evidence
for or against acting on it.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.estimation_variance import OnDegenerate, naive_block_bootstrap_interval
from nfl_ats.inactives_refresh_overlay import inactives_rows_for_game, load_inactives_snapshots
from nfl_ats.pick_refresh import (
    MOVEMENT_POLICY_THRESHOLD,
    TRIGGER_CLOCK_DISPATCH,
    TRIGGER_NEWS_EVENT,
    TRIGGER_UNKNOWN,
    current_captured_home_spread,
    original_card,
    pick_deadline,
    sunday_pick_lock,
)
from nfl_ats.weak_signals import CLOSING_GROUNDS, POOLABLE_CLASSIFICATION, TERMINAL_CLASSIFICATIONS

# ---------------------------------------------------------------------------
# trigger_source vocabulary
# ---------------------------------------------------------------------------

TRIGGER_CLOCK_CHECKPOINT = "clock_checkpoint"
TRIGGER_INACTIVES_POSTED = "inactives_posted"
TRIGGER_INJURY_REPORT_POSTED = "injury_report_posted"
TRIGGER_LINEUP_CHANGE = "lineup_change"
TRIGGER_LINE_MOVE = "line_move"
TRIGGER_MANUAL = "manual"

TRIGGER_SOURCES: frozenset[str] = frozenset(
    {
        TRIGGER_CLOCK_CHECKPOINT,
        TRIGGER_INACTIVES_POSTED,
        TRIGGER_INJURY_REPORT_POSTED,
        TRIGGER_LINEUP_CHANGE,
        TRIGGER_LINE_MOVE,
        TRIGGER_MANUAL,
    }
)

#: The scheduler jobs that are purely clock-driven refresh passes
#: (``scripts/capture_scheduler.py`` ``SCHEDULE``, read). ``refresh_thu`` /
#: ``refresh_sat`` / ``refresh_sun`` are the three named checkpoints;
#: ``refresh_*_inactives_*`` fire on a fixed clock offset from their own
#: capture window closing, NOT on the inactives capture actually reporting
#: anything -- they are clock checkpoints under an inactives-flavoured name,
#: exactly as ENG-08's own brief states. This module does not import
#: ``scripts/capture_scheduler.py`` (src/nfl_ats never imports scripts/), so
#: this list is a literal mirror of that file's job names; a name changed
#: there without a matching update here would simply stop matching, fail
#: open, and be visible as a gap in a ``--scan`` summary.
CLOCK_CHECKPOINT_NAMES: tuple[str, ...] = (
    "refresh_thu",
    "refresh_sat",
    "refresh_sun",
    "refresh_thu_inactives_early",
    "refresh_thu_inactives_late",
    "refresh_thu_inactives_primetime",
    "refresh_sat_inactives_early",
    "refresh_sat_inactives_late",
    "refresh_sun_inactives_early",
    "refresh_sun_inactives_late",
)

_SUCCESS_STATUSES = frozenset({"OK", "CAUGHT_UP", "ALREADY-CAPTURED"})


def mkt08_trigger_type(trigger_source: str) -> str:
    """Map this module's granular ``trigger_source`` onto MKT-08's coarser
    ``trigger_type`` vocabulary (``nfl_ats.pick_refresh.TRIGGER_*``), reusing
    those constants rather than redefining them. A future step that records
    a detected trigger onto the pick-revision ledger should pass this
    module's ``trigger_source`` value through ``--trigger-source`` unchanged
    and this function's output through ``--trigger-type``.
    """

    if trigger_source == TRIGGER_CLOCK_CHECKPOINT:
        return TRIGGER_CLOCK_DISPATCH
    if trigger_source == TRIGGER_MANUAL:
        return TRIGGER_UNKNOWN
    if trigger_source in TRIGGER_SOURCES:
        return TRIGGER_NEWS_EVENT
    raise ValueError(f"Unknown trigger_source {trigger_source!r}")


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _as_utc(value: Any) -> pd.Timestamp | None:
    """Parse a manifest/payload timestamp field to a tz-aware UTC Timestamp.

    Every source this module reads writes an offset-bearing ISO string
    (``+00:00``/``-04:00``) or a bare ``YYYYMMDDTHHMMSSZ`` capture stamp;
    both parse tz-aware under ``pandas.Timestamp`` directly (measured this
    session), so this only needs to convert, never guess a zone for a naive
    value found. Returns ``None`` on anything unparseable -- fail-open,
    matching every sibling snapshot reader in this codebase.
    """

    if value is None:
        return None
    try:
        stamp = pd.Timestamp(str(value))
    except (TypeError, ValueError):
        return None
    if bool(pd.isna(stamp)):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _now(observation_time: pd.Timestamp | None) -> pd.Timestamp:
    if observation_time is None:
        return pd.Timestamp.now(tz="UTC")
    stamp = pd.Timestamp(observation_time)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _iso(value: pd.Timestamp | None) -> str:
    if value is None:
        return ""
    stamp = pd.Timestamp(value)
    if bool(pd.isna(stamp)):
        return ""
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.isoformat()


def _validate_deadline(
    source_capture_time: pd.Timestamp | None, deadline: pd.Timestamp
) -> tuple[bool, str]:
    """Strictly-before check against a game's own ``pick_refresh.pick_deadline``.

    Strict, matching ``inactives_refresh_overlay.newest_snapshot_before``'s
    anti-backdating convention: a source captured exactly AT the deadline
    could not have informed a pick made before it, so it is a violation, not
    an edge case.
    """

    if source_capture_time is None or pd.isna(source_capture_time):
        return False, "deadline_violation: source_capture_time is unknown"
    deadline_ts = pd.Timestamp(deadline)
    if bool(pd.isna(deadline_ts)):
        return False, "deadline_violation: game deadline is unknown"
    if source_capture_time < deadline_ts:
        return True, "source_capture_time is strictly before the game's own pick deadline"
    return (
        False,
        "deadline_violation: source_capture_time is at or after the game's pick deadline "
        f"({source_capture_time.isoformat()} >= {deadline_ts.isoformat()})",
    )


# ---------------------------------------------------------------------------
# RefreshTrigger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefreshTrigger:
    """One reconstructed refresh-trigger event for one game.

    Every field is populated from a snapshot manifest, a scheduler state
    record, or a market-quote store -- never invented -- except
    ``observation_time``, which is honestly the scan's own clock (when this
    trigger was RECONSTRUCTED, not when the underlying event happened).
    """

    trigger_source: str
    game_id: str
    season: int
    week: int
    #: When this scan observed/reconstructed the trigger (the scan's clock).
    observation_time: pd.Timestamp
    #: When the underlying source was actually captured -- from the
    #: snapshot's own manifest/payload, NEVER from ``observation_time``.
    source_capture_time: pd.Timestamp
    #: The scheduler job name for a ``clock_checkpoint`` trigger; ``None``
    #: for every genuine non-clock trigger.
    checkpoint_name: str | None
    #: This game's own ``pick_refresh.pick_deadline``.
    deadline: pd.Timestamp
    deadline_valid: bool
    deadline_reason: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.trigger_source not in TRIGGER_SOURCES:
            raise ValueError(
                f"Unknown trigger_source {self.trigger_source!r}; expected one of "
                f"{sorted(TRIGGER_SOURCES)}"
            )

    @property
    def dedupe_key(self) -> tuple[str, str, str]:
        """``(trigger_source, source_capture_time, game_id)`` -- the JSONL
        evidence log's append-only de-duplication key."""

        return (self.trigger_source, _iso(self.source_capture_time), self.game_id)

    def to_record(self) -> dict[str, Any]:
        """A flat, JSON-safe dict for the append-only evidence artifact."""

        return {
            "trigger_source": self.trigger_source,
            "game_id": self.game_id,
            "season": self.season,
            "week": self.week,
            "observation_time": _iso(self.observation_time),
            "source_capture_time": _iso(self.source_capture_time),
            "checkpoint_name": self.checkpoint_name,
            "deadline": _iso(self.deadline),
            "deadline_valid": self.deadline_valid,
            "deadline_reason": self.deadline_reason,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Per-week game windows (kickoff + deadline), reused by every detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameWindow:
    game_id: str
    home_team: str
    away_team: str
    kickoff: pd.Timestamp
    deadline: pd.Timestamp


def schedule_game_windows(repo_root: Path, *, season: int, week: int) -> tuple[GameWindow, ...]:
    """One (season, week)'s REG games, each with its own ``pick_deadline``.

    Reads the newest local ``schedules.parquet`` the same way every sibling
    refresh-time overlay does (``crew_tilt_refresh_overlay.preview_week``,
    ``inactives_capture._schedule_lookup``); returns ``()`` when no schedule
    snapshot or no matching games are present, fail-open like every detector
    below.
    """

    hits = sorted((repo_root / "data" / "raw").glob("*/schedules.parquet"))
    if not hits:
        return ()
    schedules = pd.read_parquet(hits[-1])
    required = {"season", "week", "game_type", "game_id", "home_team", "away_team", "gameday"}
    if not required.issubset(schedules.columns):
        return ()
    week_games = schedules.loc[
        (schedules["season"].astype(int) == int(season))
        & (schedules["week"].astype(int) == int(week))
        & (schedules["game_type"] == "REG")
    ].copy()
    if week_games.empty:
        return ()
    kickoffs = pd.to_datetime(week_games["gameday"].astype(str), errors="coerce")
    if "gametime" in week_games.columns:
        combined = week_games["gameday"].astype(str) + " " + week_games["gametime"].astype(str)
        parsed = pd.to_datetime(combined, errors="coerce")
        kickoffs = parsed.fillna(kickoffs)
    kickoffs = kickoffs.dt.tz_localize(
        "America/New_York", ambiguous=True, nonexistent="shift_forward"
    )
    kickoffs_utc = kickoffs.dt.tz_convert("UTC")
    valid = kickoffs_utc.dropna()
    if valid.empty:
        return ()
    lock = sunday_pick_lock(pd.Series(valid))
    windows: list[GameWindow] = []
    for row, kickoff in zip(week_games.itertuples(index=False), kickoffs_utc, strict=True):
        if bool(pd.isna(kickoff)):
            continue
        kickoff_ts = pd.Timestamp(kickoff)
        windows.append(
            GameWindow(
                game_id=str(row.game_id),
                home_team=str(row.home_team),
                away_team=str(row.away_team),
                kickoff=kickoff_ts,
                deadline=pick_deadline(kickoff_ts, lock),
            )
        )
    return tuple(windows)


# ---------------------------------------------------------------------------
# Detector 1: fixed clock checkpoints (from the scheduler's own state file)
# ---------------------------------------------------------------------------


def detect_clock_checkpoint_triggers(
    scheduler_state: dict[str, Any],
    games: Sequence[GameWindow],
    *,
    season: int,
    week: int,
    checkpoint_names: Sequence[str] = CLOCK_CHECKPOINT_NAMES,
    observation_time: pd.Timestamp | None = None,
) -> tuple[RefreshTrigger, ...]:
    """Every successfully-run clock checkpoint, one trigger per eligible game.

    ``scheduler_state`` is ``json.loads(data/scheduler_state.json)`` --
    passed in rather than read here, so this stays pure and testable on a
    synthetic dict. A checkpoint's own ``source_capture_time`` is its
    scheduler record's ``ran_at`` (when it actually ran) falling back to
    ``window_start`` (its target instant) for a state shape that lacks
    ``ran_at`` -- both are scheduler-clock facts, never this scan's own
    clock, which is the honest sense in which a clock checkpoint's "source"
    IS the clock.
    """

    observed = _now(observation_time)
    runs = scheduler_state.get("runs", {}) if isinstance(scheduler_state, dict) else {}
    names = frozenset(checkpoint_names)
    triggers: list[RefreshTrigger] = []
    for key, record in runs.items():
        if not isinstance(key, str) or "@" not in key:
            continue
        name = key.split("@", 1)[0]
        if name not in names or not isinstance(record, dict):
            continue
        if record.get("status") not in _SUCCESS_STATUSES:
            continue
        source_capture_time = _as_utc(record.get("ran_at")) or _as_utc(record.get("window_start"))
        if source_capture_time is None:
            continue
        for game in games:
            valid, reason = _validate_deadline(source_capture_time, game.deadline)
            triggers.append(
                RefreshTrigger(
                    trigger_source=TRIGGER_CLOCK_CHECKPOINT,
                    game_id=game.game_id,
                    season=season,
                    week=week,
                    observation_time=observed,
                    source_capture_time=source_capture_time,
                    checkpoint_name=name,
                    deadline=game.deadline,
                    deadline_valid=valid,
                    deadline_reason=reason,
                    detail=f"scheduler state {key}: {record.get('status')}",
                )
            )
    return tuple(triggers)


# ---------------------------------------------------------------------------
# Detector 2: a new official inactives snapshot (WP17), reusing its reader
# ---------------------------------------------------------------------------


def detect_inactives_triggers(
    data_root: Path,
    games: Sequence[GameWindow],
    *,
    season: int,
    week: int,
    observation_time: pd.Timestamp | None = None,
) -> tuple[RefreshTrigger, ...]:
    """A real, per-game inactives-posted trigger for every reporting snapshot.

    Reuses ``nfl_ats.inactives_refresh_overlay.load_inactives_snapshots`` /
    ``inactives_rows_for_game`` verbatim -- the exact reader WP41's overlay
    already trusts to distinguish "no report yet" from "nobody is inactive"
    -- rather than re-parsing manifests here. A snapshot only produces a
    trigger for a game it actually names rows for.
    """

    observed = _now(observation_time)
    snapshots = load_inactives_snapshots(data_root)
    triggers: list[RefreshTrigger] = []
    for snapshot in snapshots:
        if not snapshot.reported_inactives:
            continue
        if snapshot.season != season or snapshot.week != week:
            continue
        source_capture_time = snapshot.captured_at_utc
        for game in games:
            aligned = inactives_rows_for_game(
                snapshot,
                season=season,
                week=week,
                game_id=game.game_id,
                home_team=game.home_team,
                away_team=game.away_team,
            )
            if aligned.empty:
                continue
            valid, reason = _validate_deadline(source_capture_time, game.deadline)
            triggers.append(
                RefreshTrigger(
                    trigger_source=TRIGGER_INACTIVES_POSTED,
                    game_id=game.game_id,
                    season=season,
                    week=week,
                    observation_time=observed,
                    source_capture_time=source_capture_time,
                    checkpoint_name=None,
                    deadline=game.deadline,
                    deadline_valid=valid,
                    deadline_reason=reason,
                    detail=f"inactives snapshot {snapshot.snapshot_id}: {len(aligned)} rows listed",
                )
            )
    return tuple(triggers)


# ---------------------------------------------------------------------------
# Detector 3: a new injury-report snapshot (nflverse / Sportradar)
# ---------------------------------------------------------------------------


def _nflverse_injury_snapshots(data_root: Path) -> tuple[tuple[str, pd.Timestamp], ...]:
    """Every readable ``data/players/raw/<snapshot>/manifest.json``.

    This archive is season-wide, not week-specific (``PlayerSnapshot``'s own
    ``injury_seasons`` tuple), so a new snapshot here is treated as relevant
    to whichever (season, week) the caller is scanning -- a fresh pull of the
    official injury archive is itself the event, independent of whether that
    particular week already has rows in it.
    """

    root = data_root / "players" / "raw"
    found: list[tuple[str, pd.Timestamp]] = []
    if not root.is_dir():
        return ()
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        stamp = _as_utc(manifest.get("created_at_utc")) or _as_utc(manifest_path.parent.name)
        if stamp is None:
            continue
        found.append((manifest_path.parent.name, stamp))
    return tuple(found)


def _sportradar_injury_snapshots(
    data_root: Path, *, season: int, week: int
) -> tuple[tuple[str, pd.Timestamp], ...]:
    """Every complete ``data/raw/sportradar_injuries/<snapshot>/manifest.json``
    matching ``(season, week)``. Mirrors the acceptance rule
    ``scripts/capture_sportradar_injuries.py``'s own reader applies (schema
    tag, ``status == "complete"``) without importing that script (src/nfl_ats
    never imports scripts/)."""

    root = data_root / "raw" / "sportradar_injuries"
    found: list[tuple[str, pd.Timestamp]] = []
    if not root.is_dir():
        return ()
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        if manifest.get("status") != "complete":
            continue
        if manifest.get("schema") != "sportradar_nfl_injuries_snapshot/1":
            continue
        try:
            if int(manifest.get("season", -1)) != int(season):
                continue
            if int(manifest.get("week", -1)) != int(week):
                continue
        except (TypeError, ValueError):
            continue
        stamp = _as_utc(manifest.get("captured_at_utc"))
        if stamp is None:
            continue
        found.append((manifest_path.parent.name, stamp))
    return tuple(found)


def detect_injury_report_triggers(
    data_root: Path,
    games: Sequence[GameWindow],
    *,
    season: int,
    week: int,
    observation_time: pd.Timestamp | None = None,
) -> tuple[RefreshTrigger, ...]:
    """A real injury-report-posted trigger for every new nflverse/Sportradar
    snapshot found, one row per game in ``games``."""

    observed = _now(observation_time)
    sources: list[tuple[str, str, pd.Timestamp]] = [
        (snapshot_id, "nflverse", stamp)
        for snapshot_id, stamp in _nflverse_injury_snapshots(data_root)
    ]
    sources.extend(
        (snapshot_id, "sportradar", stamp)
        for snapshot_id, stamp in _sportradar_injury_snapshots(data_root, season=season, week=week)
    )
    triggers: list[RefreshTrigger] = []
    for snapshot_id, provider, stamp in sources:
        for game in games:
            valid, reason = _validate_deadline(stamp, game.deadline)
            triggers.append(
                RefreshTrigger(
                    trigger_source=TRIGGER_INJURY_REPORT_POSTED,
                    game_id=game.game_id,
                    season=season,
                    week=week,
                    observation_time=observed,
                    source_capture_time=stamp,
                    checkpoint_name=None,
                    deadline=game.deadline,
                    deadline_valid=valid,
                    deadline_reason=reason,
                    detail=f"{provider} injury snapshot {snapshot_id}",
                )
            )
    return tuple(triggers)


# ---------------------------------------------------------------------------
# Detector 4: a lineup change between consecutive lineups.json captures
# ---------------------------------------------------------------------------


def _lineup_signature(team_payload: Any) -> frozenset[tuple[str, str, str | None]]:
    if not isinstance(team_payload, dict):
        return frozenset()
    players = team_payload.get("players")
    if not isinstance(players, list):
        return frozenset()
    return frozenset(
        (str(player.get("slot", "")), str(player.get("name", "")), player.get("gsis_id"))
        for player in players
        if isinstance(player, dict)
    )


def detect_lineup_change_triggers(
    archive_dir: Path,
    games: Sequence[GameWindow],
    *,
    season: int,
    week: int,
    observation_time: pd.Timestamp | None = None,
) -> tuple[RefreshTrigger, ...]:
    """A per-game trigger for every roster change between two consecutive
    archived lineup-forecast captures.

    ``scripts/build_week_lineups.py`` writes a REPLACEMENT artifact
    (``artifacts/lineups/current/lineups.json``, one stable path every
    refresh overwrites -- measured this session, confirmed by its own
    ``_remove_legacy_stamped_runs`` cleanup), so there is no on-disk history
    of consecutive captures to diff directly. ``scripts/refresh_trigger_log.py``
    is responsible for archiving a dated copy of that stable file into
    ``archive_dir`` on every scan (keyed by the payload's own
    ``generated_at``, so re-archiving an unchanged file never duplicates);
    this function only reads whatever archive already exists there, which
    keeps it independently testable against a synthetic ``tmp_path`` archive
    with no dependency on the live capture pipeline.

    Each archived file is expected to carry the same shape
    ``build_week_lineups.py`` writes: ``season``, ``week``, ``generated_at``
    (a bare UTC capture stamp, NOT this function's own clock), and
    ``games -> {game_id: {"home": {"players": [...]}, "away": {...}}}``.
    """

    observed = _now(observation_time)
    if not archive_dir.is_dir():
        return ()
    snapshots: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for path in sorted(archive_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            if int(payload.get("season", -1)) != int(season):
                continue
            if int(payload.get("week", -1)) != int(week):
                continue
        except (TypeError, ValueError):
            continue
        stamp = _as_utc(payload.get("generated_at"))
        if stamp is None:
            continue
        snapshots.append((stamp, payload))
    snapshots.sort(key=lambda item: item[0])

    triggers: list[RefreshTrigger] = []
    for (_, previous), (current_stamp, current) in itertools.pairwise(snapshots):
        previous_games_raw = previous.get("games")
        current_games_raw = current.get("games")
        previous_games: dict[str, Any] = (
            previous_games_raw if isinstance(previous_games_raw, dict) else {}
        )
        current_games: dict[str, Any] = (
            current_games_raw if isinstance(current_games_raw, dict) else {}
        )
        for game in games:
            previous_game = previous_games.get(game.game_id)
            current_game = current_games.get(game.game_id)
            if previous_game is None or current_game is None:
                continue
            changed_sides = [
                side
                for side in ("home", "away")
                if _lineup_signature(previous_game.get(side))
                != _lineup_signature(current_game.get(side))
            ]
            if not changed_sides:
                continue
            valid, reason = _validate_deadline(current_stamp, game.deadline)
            triggers.append(
                RefreshTrigger(
                    trigger_source=TRIGGER_LINEUP_CHANGE,
                    game_id=game.game_id,
                    season=season,
                    week=week,
                    observation_time=observed,
                    source_capture_time=current_stamp,
                    checkpoint_name=None,
                    deadline=game.deadline,
                    deadline_valid=valid,
                    deadline_reason=reason,
                    detail=f"lineup changed on {', '.join(changed_sides)} side(s)",
                )
            )
    return tuple(triggers)


def archive_lineup_snapshot(source: Path, archive_dir: Path) -> Path | None:
    """Copy ``source`` (a ``lineups.json``-shaped payload) into ``archive_dir``,
    keyed by its own ``generated_at`` so an unchanged file is never re-archived.

    Returns the archived path, or ``None`` when ``source`` is missing,
    unreadable, or already archived under the same ``generated_at``. Pure
    file I/O, no ledger write, no registry touch -- this is the scan script's
    own bookkeeping for :func:`detect_lineup_change_triggers`, not a new
    capture mechanism.
    """

    if not source.is_file():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    generated_at = str(payload.get("generated_at") or "").strip()
    if not generated_at:
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / f"{generated_at}.json"
    if destination.is_file():
        return None
    staging = destination.with_suffix(".tmp")
    staging.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    staging.replace(destination)
    return destination


# ---------------------------------------------------------------------------
# Detector 5: an opener-vs-current line move beyond MOVEMENT_POLICY_THRESHOLD
# ---------------------------------------------------------------------------


def detect_line_move_triggers(
    artifacts_root: Path,
    data_root: Path,
    games: Sequence[GameWindow],
    *,
    season: int,
    week: int,
    now: pd.Timestamp,
    threshold: float = MOVEMENT_POLICY_THRESHOLD,
    observation_time: pd.Timestamp | None = None,
) -> tuple[RefreshTrigger, ...]:
    """A line-move trigger for every game whose opener-to-current move meets
    ``MOVEMENT_POLICY_THRESHOLD`` (imported from ``nfl_ats.pick_refresh``,
    never redefined here).

    Reuses ``pick_refresh.original_card`` for the "opener" (the frozen
    Tuesday-recorded ``decision_home_spread``, the same anchor every other
    refresh-time module in this codebase uses) and
    ``pick_refresh.current_captured_home_spread`` for the "current" read --
    a read-only local-store lookup, never a live fetch. Fails open (no
    triggers) when either side is unavailable, matching both reused
    functions' own documented fail-open contracts.

    ``source_capture_time`` is the newest quote's ``latest_observed_at_utc``
    across the WHOLE local market store (the same field
    ``current_captured_home_spread``'s own metadata already reports for its
    "fresh" gate) -- a store-wide, not strictly per-game, capture instant.
    This is a disclosed simplification: production's own freshness check is
    equally store-wide, so this does not understate what the live pipeline
    itself already treats as "the current line's capture time".
    """

    observed = _now(observation_time)
    original = original_card(artifacts_root, season=season, week=week)
    if original.empty or "decision_home_spread" not in original.columns:
        return ()
    opener_by_game = original.set_index("game_id")["decision_home_spread"].astype(float).to_dict()
    current_lines, metadata = current_captured_home_spread(data_root, now=now)
    if not current_lines or not metadata.get("fresh", False):
        return ()
    source_capture_time = _as_utc(metadata.get("latest_observed_at_utc"))
    if source_capture_time is None:
        return ()

    triggers: list[RefreshTrigger] = []
    for game in games:
        opener = opener_by_game.get(game.game_id)
        current = current_lines.get(game.game_id)
        if opener is None or current is None or pd.isna(opener):
            continue
        delta = float(current) - float(opener)
        if abs(delta) < float(threshold):
            continue
        valid, reason = _validate_deadline(source_capture_time, game.deadline)
        triggers.append(
            RefreshTrigger(
                trigger_source=TRIGGER_LINE_MOVE,
                game_id=game.game_id,
                season=season,
                week=week,
                observation_time=observed,
                source_capture_time=source_capture_time,
                checkpoint_name=None,
                deadline=game.deadline,
                deadline_valid=valid,
                deadline_reason=reason,
                detail=(
                    f"opener {opener:+.1f} -> current {current:+.1f} "
                    f"({delta:+.1f} pts, threshold {float(threshold):.1f})"
                ),
            )
        )
    return tuple(triggers)


# ---------------------------------------------------------------------------
# Orchestration: every detector, one call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriggerScanRoots:
    """Every path a full scan needs, gathered once so callers (and tests)
    only have to construct one object."""

    repo_root: Path
    data_root: Path
    artifacts_root: Path
    lineup_archive_dir: Path
    scheduler_state: dict[str, Any]


def detect_all_triggers(
    roots: TriggerScanRoots,
    *,
    season: int,
    week: int,
    now: pd.Timestamp | None = None,
) -> tuple[RefreshTrigger, ...]:
    """Every trigger every detector above can currently reconstruct for one
    (season, week), including the fixed clock checkpoints. Returns ``()``
    when the target week has no local schedule coverage at all -- every
    individual detector already fails open on a missing source of its own."""

    observed = _now(now)
    games = schedule_game_windows(roots.repo_root, season=season, week=week)
    if not games:
        return ()
    triggers: list[RefreshTrigger] = []
    triggers.extend(
        detect_clock_checkpoint_triggers(
            roots.scheduler_state, games, season=season, week=week, observation_time=observed
        )
    )
    triggers.extend(
        detect_inactives_triggers(
            roots.data_root, games, season=season, week=week, observation_time=observed
        )
    )
    triggers.extend(
        detect_injury_report_triggers(
            roots.data_root, games, season=season, week=week, observation_time=observed
        )
    )
    triggers.extend(
        detect_lineup_change_triggers(
            roots.lineup_archive_dir, games, season=season, week=week, observation_time=observed
        )
    )
    triggers.extend(
        detect_line_move_triggers(
            roots.artifacts_root,
            roots.data_root,
            games,
            season=season,
            week=week,
            now=observed,
            observation_time=observed,
        )
    )
    return tuple(triggers)


# ---------------------------------------------------------------------------
# The append-only evidence artifact (JSONL, gitignored under artifacts/)
# ---------------------------------------------------------------------------


def evidence_log_path(artifacts_root: Path, *, season: int, week: int) -> Path:
    return artifacts_root / "refresh_triggers" / str(season) / f"week_{week}.jsonl"


def append_triggers_to_evidence_log(
    path: Path, triggers: Sequence[RefreshTrigger]
) -> tuple[int, int]:
    """Append new trigger records to ``path``, idempotently.

    De-duplicates by :attr:`RefreshTrigger.dedupe_key`
    (``trigger_source``, ``source_capture_time``, ``game_id``) against BOTH
    every line already on disk and every other trigger in this same call, so
    re-running a scan over the same capture directories -- the normal,
    expected way this is used -- never appends a second copy of the same
    event. Existing lines are never rewritten, reordered, or removed; this
    function only ever opens the file in append mode.

    Returns ``(written, skipped_as_duplicate)``.
    """

    existing_keys: set[tuple[str, str, str]] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            existing_keys.add(
                (
                    str(record.get("trigger_source", "")),
                    str(record.get("source_capture_time", "")),
                    str(record.get("game_id", "")),
                )
            )

    to_write: list[dict[str, Any]] = []
    seen_this_call: set[tuple[str, str, str]] = set()
    for trigger in triggers:
        key = trigger.dedupe_key
        if key in existing_keys or key in seen_this_call:
            continue
        seen_this_call.add(key)
        to_write.append(trigger.to_record())

    if to_write:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in to_write:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    return len(to_write), len(triggers) - len(to_write)


# ---------------------------------------------------------------------------
# The prospective comparison scaffold
# ---------------------------------------------------------------------------

COMPARISON_REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "checkpoint_pick_home",
    "trigger_pick_home",
    "settle_margin",
)


@dataclass(frozen=True)
class TriggerComparisonResult:
    """Paired trigger-vs-checkpoint forced-pick accuracy, week-blocked.

    ``estimate``/``lower``/``upper``/``probability_positive`` are
    ``trigger`` MINUS ``checkpoint`` -- positive favours acting on the
    non-clock trigger over waiting for the next fixed checkpoint.
    """

    n_games: int
    n_weeks: int
    estimate: float
    lower: float
    upper: float
    probability_positive: float
    degenerate: bool
    classification: str
    closing_ground: str | None
    excluded_deadline_violations: tuple[str, ...]
    detail: str


def compare_trigger_vs_checkpoint(
    ledger_rows: pd.DataFrame,
    triggers: Sequence[RefreshTrigger],
    *,
    samples: int = 20_000,
    seed: int = 20260904,
    on_degenerate: OnDegenerate = "warn",
) -> TriggerComparisonResult:
    """Pair the fixed-checkpoint pick against the trigger-time pick, per game.

    **Binding closing-grounds taxonomy (AGENTS.md), restated verbatim:** an
    interval or CI that contains zero is NEVER grounds to reject, fail, or
    close an experiment. At this evaluator's ~2-point resolution, "contains
    zero" is the EXPECTED outcome for a real small signal. Only two grounds
    ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
    (whole interval on the wrong side of zero) or zero split-half
    reliability; (2) bounded by a positive control proven able to detect an
    effect that size. Everything else is ``unresolved_below_power``: record
    it with ``nfl-ats weak-signals record``, report ``probability_positive``,
    never the binary "contains zero." This function never calls
    ``weak-signals record`` itself -- it is a read-only scaffold; recording
    is a separate, deliberate step once real rows exist.

    ``ledger_rows`` contract (one row per paired game; callers assemble this
    from whatever future step joins the fixed-checkpoint pick-revision
    ledger against a trigger-time refresh's own pick -- neither exists for
    2026 yet, and this function creates nothing):

    - ``game_id``, ``season``, ``week``
    - ``checkpoint_pick_home`` (bool): the pick taken at the fixed clock
      checkpoint (HOME when True).
    - ``trigger_pick_home`` (bool): the pick a trigger-time refresh took.
    - ``settle_margin`` (float): ``result - decision_line``, the SAME frozen
      grading line and result for both arms (``nfl_ats.clv.pick_correct``'s
      own convention: strictly positive is a home cover, exactly zero is a
      push and is excluded here, matching FND-04).

    Only games with a corresponding trigger in ``triggers`` whose
    ``deadline_valid`` is True are paired -- a trigger this project could
    never actually have acted on before its deadline contributes no
    evidence, in either direction. Excluded game ids are reported in
    ``excluded_deadline_violations`` rather than silently dropped.

    ``naive_block_bootstrap_interval`` (``nfl_ats.estimation_variance``) is
    the SAME estimator every other paired comparison in this project already
    reports from; this function does not define a second one.
    Within-week correlation is mandated exactly zero (``AGENTS.md``), so
    blocking is by WEEK, matching every sibling estimator's convention --
    never estimated, never padded, and this function never computes "games
    needed".

    Classification defaults to ``unresolved_below_power``
    (``nfl_ats.weak_signals.POOLABLE_CLASSIFICATION``) and is only
    reclassified ``refuted_mechanism`` with ``closing_ground
    ="wrong_sign_resolved"`` when the WHOLE interval sits strictly below
    zero AND the interval is not itself degenerate (too few week-blocks to
    trust its bounds at all, in which case the honest answer is
    ``unresolved_below_power`` regardless of the point estimate's sign).
    ``bounded_by_control`` is never applied automatically here -- it requires
    an external positive-control result this scaffold is not given, and
    fabricating one would be exactly the violation ``AGENTS.md`` forbids.
    """

    missing = sorted(set(COMPARISON_REQUIRED_COLUMNS).difference(ledger_rows.columns))
    if missing:
        raise DataContractError(
            f"compare_trigger_vs_checkpoint needs columns: {', '.join(missing)}"
        )

    trigger_by_game: dict[str, RefreshTrigger] = {}
    for trigger in triggers:
        trigger_by_game[trigger.game_id] = trigger

    valid_game_ids = {
        game_id for game_id, trigger in trigger_by_game.items() if trigger.deadline_valid
    }
    excluded = tuple(
        sorted(
            game_id for game_id, trigger in trigger_by_game.items() if not trigger.deadline_valid
        )
    )

    rows = ledger_rows.loc[ledger_rows["game_id"].astype(str).isin(valid_game_ids)].copy()
    rows = rows.loc[rows["settle_margin"].notna() & rows["settle_margin"].ne(0.0)]

    if rows.empty:
        return TriggerComparisonResult(
            n_games=0,
            n_weeks=0,
            estimate=0.0,
            lower=0.0,
            upper=0.0,
            probability_positive=0.5,
            degenerate=True,
            classification=POOLABLE_CLASSIFICATION,
            closing_ground=None,
            excluded_deadline_violations=excluded,
            detail="no paired games with a deadline-valid trigger and a settled result",
        )

    actual = np.asarray(rows["settle_margin"].astype(float) > 0.0, dtype=np.float64)
    checkpoint_prob = np.asarray(rows["checkpoint_pick_home"].astype(bool), dtype=np.float64)
    trigger_prob = np.asarray(rows["trigger_pick_home"].astype(bool), dtype=np.float64)
    block_ids = rows["week"].to_numpy()

    interval = naive_block_bootstrap_interval(
        actual,
        checkpoint_prob,
        trigger_prob,
        block_ids,
        samples=samples,
        seed=seed,
        on_degenerate=on_degenerate,
    )

    classification = POOLABLE_CLASSIFICATION
    closing_ground: str | None = None
    if not interval.degenerate and interval.upper < 0.0:
        classification = TERMINAL_CLASSIFICATIONS[0]  # "refuted_mechanism"
        closing_ground = CLOSING_GROUNDS[classification][0]  # "wrong_sign_resolved"

    n_weeks = int(pd.Series(block_ids).nunique())
    return TriggerComparisonResult(
        n_games=len(rows),
        n_weeks=n_weeks,
        estimate=interval.estimate,
        lower=interval.lower,
        upper=interval.upper,
        probability_positive=interval.probability_positive,
        degenerate=interval.degenerate,
        classification=classification,
        closing_ground=closing_ground,
        excluded_deadline_violations=excluded,
        detail=(
            f"{len(rows)} paired games across {n_weeks} week-block(s); probability_positive "
            f"{interval.probability_positive:.4f} that a trigger-time refresh beats waiting for "
            "the fixed checkpoint. Per AGENTS.md, an interval crossing zero is never grounds to "
            "reject or close this line of work -- report probability_positive, not 'contains "
            "zero'."
        ),
    )
