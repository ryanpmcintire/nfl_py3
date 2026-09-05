"""ENG-18: a compact diff between the Tuesday lock and each later refresh pass.

**This module is a diff, not a verdict.** It never adjudicates a signal, never
writes to ``registry/``, and never calls ``nfl-ats weak-signals record`` or
``nfl-ats rotation record-look``. (Binding closing-grounds taxonomy, restated
per this project's rule for any module that touches an experiment: an
interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment; only a RESOLVED wrong sign, zero split-half reliability, or a
proven positive control can close a line of work; everything else is
``unresolved_below_power``. None of that applies here -- this module reports
what changed between two recorded snapshots, nothing more.)

What "decision-time snapshot diff" means here
-----------------------------------------------
The pool freezes its grading line on Tuesday
(``nfl_ats.clv.load_paper_decisions`` / ``publish-predictions
--record-decisions``). Two channels can then produce a LATER pass over the
same week:

1. **The pick-revision ledger** (``nfl_ats.pick_refresh``, POL-11/MKT-08):
   ``refresh-picks --record-decisions`` recomputes the model at the FROZEN
   Tuesday line and appends only the games whose pick actually flipped, under
   one shared ``refresh_run_id``. Overlays (coach-fade, division-revenge,
   player-arrests, spread-gap-zone) are never reloaded or recomputed by this
   channel -- they are copied verbatim from the Tuesday row -- so this
   module's overlay diff for a pick-revision-ledger pass is always
   ``unchanged`` by construction, not a measurement.
2. **A later ``margin_predictions`` forecast artifact** for the same
   season/week (a card republish, or -- honestly, on the disk this project
   actually has right now -- a repeated ``margin-predict`` development/test
   invocation; nothing on disk distinguishes the two). This channel carries
   real market-line/probability/pick data (``predictions.csv``) but never an
   overlay decision (that composition step happens downstream of
   ``margin-predict``), so its overlay diff is always ``no_data``.

Because the pick-revision ledger only records CHANGED, eligible games, a game
absent from one pass's ledger rows is not automatically "unchanged" --
:func:`build_snapshot_diff` uses ``nfl_ats.pick_refresh.pick_deadline`` (a
pure, already-established function, not a live recompute) to tell "eligible
and genuinely unchanged" apart from "ineligible at this pass's instant" where
it can, and falls back to an explicit ``no_data`` otherwise. Every cell in
:func:`render_markdown`'s output carries one of exactly three states --
``changed`` / ``unchanged`` / ``no_data`` (pick cells use ``same`` /
``flipped_<from>_to_<to>`` / ``no_data``) -- and a ``*_basis`` string
explaining how that state was established. Nothing is ever left blank.

Source timestamps (ENG-16 lineage + ENG-14 freshness policy)
--------------------------------------------------------------
``refresh-picks`` never persists a ``lineage.json`` (it recomputes in
memory), so a pick-revision-ledger pass's per-source cells are always
``no_data`` with that reason stated. A ``margin_predictions`` forecast
artifact DOES write one (``nfl_ats.lineage.write_card_lineage``, ENG-16) when
built under current code, so a forecast-artifact-origin pass compares the
Tuesday lock's ``lineage.json`` against its own, field by field, using each
field's own ``source_captured_at``. ``nfl_ats.source_freshness_policy``
(ENG-14) cannot answer a HISTORICAL "as of this pass" question -- its reader
(``observe_from_disk``) only ever reports the newest snapshot on disk RIGHT
NOW, regardless of what instant is passed in -- so it is used here only for
one clearly-labelled, present-tense section (:attr:`SnapshotDiff.
current_source_freshness`), never smuggled into a per-pass historical cell.

Trigger provenance (ENG-08)
-----------------------------
A pick-revision-ledger row already carries its own
``trigger_type``/``trigger_source``/``trigger_observed_at_utc`` (MKT-08).
When those are blank or ``unknown`` (or for a forecast-artifact pass, which
carries no trigger fields at all), this module looks up the nearest
DEADLINE-VALID entry at or before the pass's own instant in
``nfl_ats.refresh_triggers``'s append-only evidence log
(``artifacts/refresh_triggers/<season>/week_<n>.jsonl``), preferring one
naming a game in this pass. When neither source has anything, the trigger is
reported as ``unknown`` -- never fabricated, never defaulted to
``clock_dispatch``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import load_paper_decisions
from nfl_ats.lineage import (
    LINEAGE_FILENAME,
    CardLineage,
    LineageError,
    read_card_lineage,
)
from nfl_ats.pick_refresh import (
    TRIGGER_UNKNOWN,
    load_pick_revisions,
    pick_deadline,
    sunday_pick_lock,
)
from nfl_ats.refresh_triggers import evidence_log_path, mkt08_trigger_type
from nfl_ats.source_freshness_policy import report_for_publication

# ---------------------------------------------------------------------------
# small vocab
# ---------------------------------------------------------------------------

STATE_CHANGED = "changed"
STATE_UNCHANGED = "unchanged"
STATE_NO_DATA = "no_data"

PASS_ORIGIN_PICK_REVISION = "pick_revision_ledger"
PASS_ORIGIN_FORECAST_ARTIFACT = "forecast_artifact"

#: Individual overlay flip flags recorded on the paper-decision ledger
#: (``nfl_ats.clv.PAPER_DECISION_COLUMNS``). ``composed_overlay_flip`` is the
#: union of these four and is deliberately not listed as a fifth "overlay
#: name" here -- it would double-count.
OVERLAY_FLIP_COLUMNS: dict[str, str] = {
    "coach_fade": "coach_fade_flip",
    "division_revenge": "division_revenge_flip",
    "player_arrests": "player_arrests_flip",
    "spread_gap_zone": "spread_gap_zone_flip",
}


# ---------------------------------------------------------------------------
# small, pure helpers
# ---------------------------------------------------------------------------


def _utc(now: datetime | None) -> pd.Timestamp:
    value = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _iso(value: pd.Timestamp) -> str:
    ts = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    return ts.isoformat()


def _iso_from_any(value: Any) -> str | None:
    """Best-effort ISO-8601 UTC string for a timestamp of unknown flavour."""

    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if bool(pd.isna(ts)):
        return None
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.isoformat()


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _numeric_state(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return STATE_NO_DATA
    return STATE_UNCHANGED if abs(a - b) <= 1e-9 else STATE_CHANGED


def _pick_state(a: str | None, b: str | None) -> str:
    if not a or not b:
        return STATE_NO_DATA
    return "same" if a == b else f"flipped_{a.lower()}_to_{b.lower()}"


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameSnapshot:
    """One game's state at one instant (Tuesday lock, or one refresh pass)."""

    game_id: str
    home_team: str
    away_team: str
    kickoff: str | None
    market_line: float | None
    market_line_basis: str
    model_probability: float | None
    model_probability_basis: str
    pick_side: str | None
    pick_basis: str
    #: ``None`` means "unknown whether any overlay fired" (no_data), never
    #: "no overlay fired" -- an empty, non-``None`` tuple means that instead.
    overlays_fired: tuple[str, ...] | None
    overlays_basis: str


@dataclass(frozen=True)
class TuesdayLock:
    """The resolved Tuesday-lock anchor for one season/week."""

    season: int
    week: int
    resolved: bool
    #: ``"lockday_package"`` / ``"paper_decision_ledger"`` /
    #: ``"forecast_artifact_earliest"`` / ``"unresolved"``.
    basis: str
    forecast_artifact: str | None
    forecast_directory: str | None
    forecast_created_at_utc: str | None
    ledger_rows: int
    lineage_available: bool
    games: tuple[GameSnapshot, ...]
    detail: str


@dataclass(frozen=True)
class SourceTimestampCell:
    source_id: str
    tuesday_captured_at: str | None
    refresh_captured_at: str | None
    state: str
    detail: str


@dataclass(frozen=True)
class GameDiffRow:
    game_id: str
    home_team: str
    away_team: str
    tuesday_market_line: float | None
    tuesday_market_line_basis: str
    refresh_market_line: float | None
    refresh_market_line_basis: str
    market_line_state: str
    tuesday_model_probability: float | None
    tuesday_probability_basis: str
    refresh_model_probability: float | None
    refresh_probability_basis: str
    probability_delta: float | None
    probability_state: str
    tuesday_pick_side: str | None
    tuesday_pick_basis: str
    refresh_pick_side: str | None
    refresh_pick_basis: str
    pick_state: str
    overlays_added: tuple[str, ...]
    overlays_removed: tuple[str, ...]
    overlays_unchanged: tuple[str, ...]
    overlay_state: str
    overlay_basis: str


@dataclass(frozen=True)
class RefreshPassDiff:
    refresh_run_id: str
    origin: str
    computed_at_utc: str | None
    trigger_type: str
    trigger_source: str
    trigger_observed_at_utc: str | None
    #: ``"ledger_recorded"`` / ``"evidence_log_nearest"`` / ``"unknown"``.
    trigger_basis: str
    games: tuple[GameDiffRow, ...]
    sources: tuple[SourceTimestampCell, ...]


@dataclass(frozen=True)
class SnapshotDiff:
    season: int
    week: int
    generated_at_utc: str
    tuesday: TuesdayLock
    refresh_passes: tuple[RefreshPassDiff, ...]
    #: A PRESENT-TENSE (evaluated at ``generated_at_utc``, never historical)
    #: ``nfl_ats.source_freshness_policy.SourcePolicyReport.to_metadata()``
    #: snapshot, for context only -- never a per-pass historical read.
    current_source_freshness: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ArtifactRead:
    directory: Path
    metadata: dict[str, Any]
    #: Indexed by ``game_id``, restricted to the artifact's own ``ats_method`` rows.
    predictions: pd.DataFrame
    lineage: CardLineage | None


@dataclass(frozen=True)
class _TriggerInfo:
    trigger_type: str
    trigger_source: str
    trigger_observed_at_utc: str | None
    basis: str


# ---------------------------------------------------------------------------
# reading margin_predictions artifacts
# ---------------------------------------------------------------------------


def _list_margin_prediction_dirs(artifacts_root: Path, *, season: int, week: int) -> list[Path]:
    """Every ``margin_predictions`` directory for one week, chronological.

    Directory names are ``{season}-week-{week:02d}-{run_id}`` (``nfl_ats.cli``
    ``_cmd_margin_predict``), so a lexicographic sort is also a chronological
    one -- same convention ``nfl_ats.cli._latest_margin_prediction_dir`` relies on.
    """

    root = artifacts_root / "margin_predictions"
    if not root.is_dir():
        return []
    prefix = f"{int(season)}-week-{int(week):02d}-"
    return sorted(
        entry for entry in root.iterdir() if entry.is_dir() and entry.name.startswith(prefix)
    )


def _read_margin_prediction_artifact(directory: Path) -> _ArtifactRead | None:
    """Read one ``margin_predictions`` artifact's decision-relevant fields.

    Returns ``None`` when the directory does not look like a real artifact
    (missing ``metadata.json``/``predictions.csv``, or unparseable) --
    fail-open, matching every other reader in this codebase.
    """

    metadata_path = directory / "metadata.json"
    predictions_path = directory / "predictions.csv"
    if not metadata_path.is_file() or not predictions_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(metadata, dict):
        return None
    ats_method = str(metadata.get("ats_method") or "market_residual")
    try:
        frame = pd.read_csv(predictions_path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    if "game_id" not in frame.columns or "method" not in frame.columns:
        return None
    scoped = frame.loc[frame["method"].astype(str).eq(ats_method)].copy()
    scoped["game_id"] = scoped["game_id"].astype(str)
    scoped = scoped.set_index("game_id", drop=False)

    lineage: CardLineage | None = None
    if (directory / LINEAGE_FILENAME).is_file():
        try:
            lineage = read_card_lineage(directory)
        except (OSError, ValueError, LineageError, KeyError):
            lineage = None

    return _ArtifactRead(
        directory=directory, metadata=metadata, predictions=scoped, lineage=lineage
    )


def _artifact_row(artifact: _ArtifactRead, game_id: str) -> pd.Series | None:
    if game_id not in artifact.predictions.index:
        return None
    row = artifact.predictions.loc[game_id]
    if isinstance(row, pd.DataFrame):  # duplicate game_id guard
        row = row.iloc[0]
    return row


# ---------------------------------------------------------------------------
# resolving the Tuesday lock
# ---------------------------------------------------------------------------


def _earliest_lockday_package(
    artifacts_root: Path, *, season: int, week: int
) -> dict[str, Any] | None:
    """The earliest ``lockday_packages`` manifest for this week, if any.

    Reads the manifest as plain JSON (never imports ``nfl_ats.lockday_package``
    for its heavier ``load_package``/verification machinery) -- this module
    only ever needs the ``outputs.forecast.directory`` pointer and
    ``created_at_utc``.
    """

    root = artifacts_root / "lockday_packages"
    if not root.is_dir():
        return None
    prefix = f"{int(season)}_wk{int(week):02d}_"
    candidates = sorted(
        entry for entry in root.iterdir() if entry.is_dir() and entry.name.startswith(prefix)
    )
    for candidate in candidates:
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(manifest, dict) and manifest.get("kind") == "lockday_decision_package":
            return manifest
    return None


def _build_tuesday_games(
    ledger: pd.DataFrame, artifact: _ArtifactRead | None
) -> tuple[GameSnapshot, ...]:
    ledger_by_game: dict[str, pd.Series] = {}
    if not ledger.empty:
        for _, row in ledger.iterrows():
            ledger_by_game[str(row["game_id"])] = row

    game_ids: set[str] = set(ledger_by_game)
    if artifact is not None:
        game_ids.update(str(value) for value in artifact.predictions.index)

    games: list[GameSnapshot] = []
    for game_id in sorted(game_ids):
        ledger_row = ledger_by_game.get(game_id)
        artifact_row = _artifact_row(artifact, game_id) if artifact is not None else None

        home_team = ""
        away_team = ""
        kickoff: str | None = None
        market_line: float | None = None
        market_line_basis = "no_data"
        pick_side: str | None = None
        pick_basis = "no_data"
        overlays_fired: tuple[str, ...] | None = None
        overlays_basis = "no_data: no readable forecast artifact and no paper-decision ledger row"

        if ledger_row is not None:
            home_team = str(ledger_row["home_team"])
            away_team = str(ledger_row["away_team"])
            kickoff = _iso_from_any(ledger_row.get("kickoff"))
            market_line = _num(ledger_row.get("decision_home_spread"))
            market_line_basis = "paper_decision_ledger"
            pick_side = _optional_text(ledger_row.get("pick_side"))
            pick_basis = "paper_decision_ledger"
            overlays_fired = tuple(
                name
                for name, column in OVERLAY_FLIP_COLUMNS.items()
                if bool(ledger_row.get(column, False))
            )
            overlays_basis = "paper_decision_ledger"
        elif artifact_row is not None:
            home_team = str(artifact_row.get("home_team", ""))
            away_team = str(artifact_row.get("away_team", ""))
            kickoff = _iso_from_any(artifact_row.get("kickoff"))
            market_line = _num(artifact_row.get("spread_line"))
            market_line_basis = "forecast_artifact_raw"
            pick_side = _optional_text(artifact_row.get("bet_side"))
            pick_basis = "forecast_artifact_raw"
            overlays_basis = (
                "no_data: no paper-decision ledger row for this game/week; margin-predict "
                "artifacts do not record played-policy overlay composition"
            )

        model_probability: float | None = None
        model_probability_basis = "no_data: no readable forecast artifact for this week"
        if artifact_row is not None:
            model_probability = _num(artifact_row.get("home_cover_probability"))
            model_probability_basis = "forecast_artifact_raw"
            if not home_team:
                home_team = str(artifact_row.get("home_team", ""))
            if not away_team:
                away_team = str(artifact_row.get("away_team", ""))
            if kickoff is None:
                kickoff = _iso_from_any(artifact_row.get("kickoff"))

        games.append(
            GameSnapshot(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                kickoff=kickoff,
                market_line=market_line,
                market_line_basis=market_line_basis,
                model_probability=model_probability,
                model_probability_basis=model_probability_basis,
                pick_side=pick_side,
                pick_basis=pick_basis,
                overlays_fired=overlays_fired,
                overlays_basis=overlays_basis,
            )
        )
    return tuple(games)


def resolve_tuesday_lock(artifacts_root: Path, *, season: int, week: int) -> TuesdayLock:
    """Resolve the Tuesday-lock anchor: lockday package > paper ledger > earliest artifact.

    Never fabricates a lock: when none of the three sources exist for this
    season/week, returns ``resolved=False`` with an explicit ``detail``.
    """

    ledger = load_paper_decisions(artifacts_root)
    week_ledger = (
        ledger.loc[ledger["season"].astype(int).eq(season) & ledger["week"].astype(int).eq(week)]
        if not ledger.empty
        else ledger
    )
    ledger_rows = len(week_ledger)

    forecast_dir: Path | None = None
    basis = "unresolved"
    forecast_artifact_field: str | None = None
    forecast_created_at: str | None = None

    manifest = _earliest_lockday_package(artifacts_root, season=season, week=week)
    if manifest is not None:
        outputs = manifest.get("outputs")
        forecast_block = outputs.get("forecast") if isinstance(outputs, dict) else None
        field_value = forecast_block.get("directory") if isinstance(forecast_block, dict) else None
        if isinstance(field_value, str) and field_value:
            candidate = Path(field_value)
            if candidate.is_dir():
                forecast_dir = candidate
                basis = "lockday_package"
                forecast_artifact_field = field_value
                forecast_created_at = _optional_text(manifest.get("created_at_utc"))

    if forecast_dir is None and not week_ledger.empty:
        ordered = week_ledger.sort_values("recorded_at_utc")
        first_row = ordered.iloc[0]
        artifact_rel = str(first_row.get("forecast_artifact") or "").strip()
        if artifact_rel:
            candidate = artifacts_root / artifact_rel
            if candidate.is_dir():
                forecast_dir = candidate
                basis = "paper_decision_ledger"
                forecast_artifact_field = artifact_rel
                forecast_created_at = _iso_from_any(first_row.get("forecast_created_at_utc"))

    if forecast_dir is None:
        dirs = _list_margin_prediction_dirs(artifacts_root, season=season, week=week)
        if dirs:
            forecast_dir = dirs[0]
            basis = "forecast_artifact_earliest"
            forecast_artifact_field = f"margin_predictions/{forecast_dir.name}"

    if forecast_dir is None and week_ledger.empty:
        return TuesdayLock(
            season=season,
            week=week,
            resolved=False,
            basis="unresolved",
            forecast_artifact=None,
            forecast_directory=None,
            forecast_created_at_utc=None,
            ledger_rows=0,
            lineage_available=False,
            games=(),
            detail=(
                "No lockday_packages manifest, no paper-decision ledger rows, and no "
                "margin_predictions artifact directory found for this season/week; there is "
                "nothing to anchor a Tuesday lock to."
            ),
        )

    artifact = _read_margin_prediction_artifact(forecast_dir) if forecast_dir is not None else None
    if artifact is not None and forecast_created_at is None:
        forecast_created_at = _optional_text(artifact.metadata.get("created_at_utc"))

    games = _build_tuesday_games(week_ledger, artifact)
    lineage_available = artifact is not None and artifact.lineage is not None
    detail = f"resolved via {basis}"
    if forecast_dir is not None and artifact is None:
        detail += "; forecast artifact directory exists but could not be read"
    if forecast_dir is None:
        detail += "; no forecast artifact directory resolved, only ledger rows"

    return TuesdayLock(
        season=season,
        week=week,
        resolved=True,
        basis=basis,
        forecast_artifact=forecast_artifact_field,
        forecast_directory=str(forecast_dir) if forecast_dir is not None else None,
        forecast_created_at_utc=forecast_created_at,
        ledger_rows=ledger_rows,
        lineage_available=lineage_available,
        games=games,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# ENG-08 evidence-log lookups
# ---------------------------------------------------------------------------


def _load_evidence_log_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except ValueError:
            continue
        if isinstance(record, dict):
            rows.append(record)
    return rows


def _nearest_preceding_trigger(
    rows: list[dict[str, Any]], *, before: pd.Timestamp, game_ids: frozenset[str]
) -> dict[str, Any] | None:
    """The latest deadline-valid evidence-log row at or before ``before``.

    Prefers a row naming a game in ``game_ids`` (this pass's own games) over
    any other deadline-valid row, but falls back to the latter rather than
    returning ``None`` outright -- a trigger for a sibling game in the same
    week is still better evidence than nothing.
    """

    candidates: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for row in rows:
        if not row.get("deadline_valid", False):
            continue
        capture = row.get("source_capture_time")
        if not capture:
            continue
        try:
            ts = pd.Timestamp(capture)
        except (TypeError, ValueError):
            continue
        if bool(pd.isna(ts)):
            continue
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if ts > before:
            continue
        candidates.append((ts, row))
    if not candidates:
        return None
    matching = [(ts, row) for ts, row in candidates if row.get("game_id") in game_ids]
    pool = matching or candidates
    pool.sort(key=lambda item: item[0])
    return pool[-1][1]


def _resolve_trigger(
    *,
    ledger_trigger_type: str,
    ledger_trigger_source: str,
    ledger_trigger_observed_at: Any,
    evidence_rows: list[dict[str, Any]],
    computed_at: pd.Timestamp,
    game_ids: frozenset[str],
) -> _TriggerInfo:
    if ledger_trigger_source.strip() or (
        ledger_trigger_type and ledger_trigger_type != TRIGGER_UNKNOWN
    ):
        return _TriggerInfo(
            trigger_type=ledger_trigger_type or TRIGGER_UNKNOWN,
            trigger_source=ledger_trigger_source,
            trigger_observed_at_utc=_iso_from_any(ledger_trigger_observed_at),
            basis="ledger_recorded",
        )
    nearest = _nearest_preceding_trigger(evidence_rows, before=computed_at, game_ids=game_ids)
    if nearest is not None:
        source = str(nearest.get("trigger_source", ""))
        try:
            mapped_type = mkt08_trigger_type(source)
        except ValueError:
            mapped_type = TRIGGER_UNKNOWN
        return _TriggerInfo(
            trigger_type=mapped_type,
            trigger_source=source,
            trigger_observed_at_utc=_optional_text(nearest.get("source_capture_time")),
            basis="evidence_log_nearest",
        )
    return _TriggerInfo(
        trigger_type=TRIGGER_UNKNOWN,
        trigger_source="",
        trigger_observed_at_utc=None,
        basis="unknown",
    )


# ---------------------------------------------------------------------------
# per-source lineage diff (pass-level, not per-game -- see module docstring)
# ---------------------------------------------------------------------------


def _source_cells(
    *,
    tuesday_lineage: CardLineage | None,
    refresh_lineage: CardLineage | None,
    no_tuesday_reason: str,
    no_refresh_reason: str,
) -> tuple[SourceTimestampCell, ...]:
    if tuesday_lineage is None and refresh_lineage is None:
        return tuple(
            SourceTimestampCell(
                source_id=candidate_field,
                tuesday_captured_at=None,
                refresh_captured_at=None,
                state=STATE_NO_DATA,
                detail=f"{no_tuesday_reason}; {no_refresh_reason}",
            )
            for candidate_field in ("pick", "model_probability", "market_line")
        )

    fields: list[str] = []
    seen: set[str] = set()
    for card_lineage in (tuesday_lineage, refresh_lineage):
        if card_lineage is None:
            continue
        for entry in card_lineage.entries:
            if not entry.decision_bearing or entry.card_field in seen:
                continue
            seen.add(entry.card_field)
            fields.append(entry.card_field)
    fields.sort()

    cells: list[SourceTimestampCell] = []
    for card_field in fields:
        tue_entry = tuesday_lineage.field(card_field) if tuesday_lineage is not None else None
        ref_entry = refresh_lineage.field(card_field) if refresh_lineage is not None else None
        tue_captured = (
            tue_entry.lineage.source_captured_at if (tue_entry and tue_entry.lineage) else None
        )
        ref_captured = (
            ref_entry.lineage.source_captured_at if (ref_entry and ref_entry.lineage) else None
        )

        if tuesday_lineage is None:
            detail = no_tuesday_reason
        elif refresh_lineage is None:
            detail = no_refresh_reason
        elif tue_captured is None or ref_captured is None:
            detail = "one or both sides record no source_captured_at for this field"
        else:
            detail = ""

        if (
            tuesday_lineage is None
            or refresh_lineage is None
            or tue_captured is None
            or ref_captured is None
        ):
            state = STATE_NO_DATA
        elif tue_captured == ref_captured:
            state = STATE_UNCHANGED
        else:
            state = STATE_CHANGED

        cells.append(
            SourceTimestampCell(
                source_id=card_field,
                tuesday_captured_at=tue_captured,
                refresh_captured_at=ref_captured,
                state=state,
                detail=detail,
            )
        )
    return tuple(cells)


# ---------------------------------------------------------------------------
# combining a Tuesday snapshot and a refresh snapshot into one row
# ---------------------------------------------------------------------------


def _diff_game(tue: GameSnapshot, refresh: GameSnapshot) -> GameDiffRow:
    market_state = _numeric_state(tue.market_line, refresh.market_line)
    probability_state = _numeric_state(tue.model_probability, refresh.model_probability)
    probability_delta = (
        None
        if tue.model_probability is None or refresh.model_probability is None
        else refresh.model_probability - tue.model_probability
    )
    pick_state = _pick_state(tue.pick_side, refresh.pick_side)

    if tue.overlays_fired is None or refresh.overlays_fired is None:
        overlay_state = STATE_NO_DATA
        added: tuple[str, ...] = ()
        removed: tuple[str, ...] = ()
        unchanged: tuple[str, ...] = ()
        overlay_basis = tue.overlays_basis if tue.overlays_fired is None else refresh.overlays_basis
    else:
        tue_set, ref_set = set(tue.overlays_fired), set(refresh.overlays_fired)
        added = tuple(sorted(ref_set - tue_set))
        removed = tuple(sorted(tue_set - ref_set))
        unchanged = tuple(sorted(tue_set & ref_set))
        overlay_state = STATE_UNCHANGED if not added and not removed else STATE_CHANGED
        overlay_basis = refresh.overlays_basis

    return GameDiffRow(
        game_id=tue.game_id,
        home_team=tue.home_team or refresh.home_team,
        away_team=tue.away_team or refresh.away_team,
        tuesday_market_line=tue.market_line,
        tuesday_market_line_basis=tue.market_line_basis,
        refresh_market_line=refresh.market_line,
        refresh_market_line_basis=refresh.market_line_basis,
        market_line_state=market_state,
        tuesday_model_probability=tue.model_probability,
        tuesday_probability_basis=tue.model_probability_basis,
        refresh_model_probability=refresh.model_probability,
        refresh_probability_basis=refresh.model_probability_basis,
        probability_delta=probability_delta,
        probability_state=probability_state,
        tuesday_pick_side=tue.pick_side,
        tuesday_pick_basis=tue.pick_basis,
        refresh_pick_side=refresh.pick_side,
        refresh_pick_basis=refresh.pick_basis,
        pick_state=pick_state,
        overlays_added=added,
        overlays_removed=removed,
        overlays_unchanged=unchanged,
        overlay_state=overlay_state,
        overlay_basis=overlay_basis,
    )


# ---------------------------------------------------------------------------
# refresh-pass sources: pick-revision ledger
# ---------------------------------------------------------------------------


def _pick_revision_passes(
    artifacts_root: Path,
    *,
    season: int,
    week: int,
    tuesday: TuesdayLock,
    sunday_lock: pd.Timestamp | None,
    evidence_rows: list[dict[str, Any]],
) -> list[RefreshPassDiff]:
    revisions = load_pick_revisions(artifacts_root)
    if revisions.empty:
        return []
    week_revisions = revisions.loc[
        revisions["season"].astype(int).eq(season) & revisions["week"].astype(int).eq(week)
    ]
    if week_revisions.empty:
        return []

    tuesday_by_game = {game.game_id: game for game in tuesday.games}
    passes: list[RefreshPassDiff] = []
    for refresh_run_id, group in week_revisions.groupby("refresh_run_id", sort=True):
        group = group.reset_index(drop=True)
        first = group.iloc[0]
        computed_at_raw = pd.Timestamp(first["revision_recorded_at_utc"])
        computed_at = (
            computed_at_raw.tz_localize("UTC")
            if computed_at_raw.tzinfo is None
            else computed_at_raw.tz_convert("UTC")
        )
        changed_by_game: dict[str, pd.Series] = {
            str(row["game_id"]): row for _, row in group.iterrows()
        }

        trigger = _resolve_trigger(
            ledger_trigger_type=str(first.get("trigger_type") or ""),
            ledger_trigger_source=str(first.get("trigger_source") or ""),
            ledger_trigger_observed_at=first.get("trigger_observed_at_utc"),
            evidence_rows=evidence_rows,
            computed_at=computed_at,
            game_ids=frozenset(changed_by_game),
        )

        games: list[GameDiffRow] = []
        for game_id in sorted(tuesday_by_game):
            tue = tuesday_by_game[game_id]
            changed_row = changed_by_game.get(game_id)
            if changed_row is not None:
                refresh = GameSnapshot(
                    game_id=game_id,
                    home_team=tue.home_team,
                    away_team=tue.away_team,
                    kickoff=tue.kickoff,
                    market_line=tue.market_line,
                    market_line_basis=(
                        "frozen_by_design: refresh-picks always grades at the Tuesday "
                        "decision_home_spread"
                    ),
                    model_probability=_num(changed_row.get("new_home_cover_probability")),
                    model_probability_basis="pick_revision_ledger_post_coach_fade_pre_movement_policy",
                    pick_side=_optional_text(changed_row.get("new_pick_side")),
                    pick_basis="pick_revision_ledger",
                    overlays_fired=tue.overlays_fired,
                    overlays_basis=(
                        "frozen_by_design: refresh-picks never reloads or recomputes coach/"
                        "division-revenge/player-arrests/spread-gap inputs"
                    ),
                )
            else:
                pick_side: str | None = None
                pick_basis = (
                    "no_data: no pick-revision row for this game in this refresh pass "
                    "(the ledger only records CHANGED, eligible picks)"
                )
                if sunday_lock is not None and tue.kickoff is not None:
                    kickoff_ts = pd.Timestamp(tue.kickoff)
                    deadline = pick_deadline(kickoff_ts, sunday_lock)
                    if computed_at >= deadline:
                        pick_basis = (
                            "ineligible_at_this_pass: this game's own kickoff or the week's "
                            "Sunday 4pm ET pick lock had already passed at this refresh's "
                            "computed_at_utc"
                        )
                    else:
                        pick_side = tue.pick_side
                        pick_basis = (
                            "inferred_unchanged: the ledger records only CHANGED picks; this "
                            "game was still eligible at computed_at_utc and is absent from "
                            "this pass's rows, so by the ledger's own recording contract its "
                            "pick did not change"
                        )
                refresh = GameSnapshot(
                    game_id=game_id,
                    home_team=tue.home_team,
                    away_team=tue.away_team,
                    kickoff=tue.kickoff,
                    market_line=tue.market_line,
                    market_line_basis=(
                        "frozen_by_design: refresh-picks always grades at the Tuesday "
                        "decision_home_spread"
                    ),
                    model_probability=None,
                    model_probability_basis=(
                        "no_data: the ledger only records the recomputed probability for "
                        "changed picks"
                    ),
                    pick_side=pick_side,
                    pick_basis=pick_basis,
                    overlays_fired=tue.overlays_fired,
                    overlays_basis=(
                        "frozen_by_design: refresh-picks never reloads or recomputes coach/"
                        "division-revenge/player-arrests/spread-gap inputs"
                    ),
                )
            games.append(_diff_game(tue, refresh))

        passes.append(
            RefreshPassDiff(
                refresh_run_id=str(refresh_run_id),
                origin=PASS_ORIGIN_PICK_REVISION,
                computed_at_utc=_iso(computed_at),
                trigger_type=trigger.trigger_type,
                trigger_source=trigger.trigger_source,
                trigger_observed_at_utc=trigger.trigger_observed_at_utc,
                trigger_basis=trigger.basis,
                games=tuple(games),
                sources=_source_cells(
                    tuesday_lineage=None,
                    refresh_lineage=None,
                    no_tuesday_reason="not evaluated for this pass type",
                    no_refresh_reason=(
                        "refresh-picks recomputes in-memory and does not persist a "
                        "lineage.json for this pass"
                    ),
                ),
            )
        )
    return passes


# ---------------------------------------------------------------------------
# refresh-pass sources: later margin_predictions forecast artifacts
# ---------------------------------------------------------------------------


def _forecast_artifact_passes(
    artifacts_root: Path,
    *,
    season: int,
    week: int,
    tuesday: TuesdayLock,
    tuesday_lineage: CardLineage | None,
    evidence_rows: list[dict[str, Any]],
) -> list[RefreshPassDiff]:
    dirs = _list_margin_prediction_dirs(artifacts_root, season=season, week=week)
    tuesday_dir_name = Path(tuesday.forecast_directory).name if tuesday.forecast_directory else None
    later_dirs = [d for d in dirs if tuesday_dir_name is None or d.name > tuesday_dir_name]
    tuesday_by_game = {game.game_id: game for game in tuesday.games}

    passes: list[RefreshPassDiff] = []
    for directory in later_dirs:
        artifact = _read_margin_prediction_artifact(directory)
        if artifact is None:
            continue
        created_at = _optional_text(artifact.metadata.get("created_at_utc"))
        computed_at_ts = _iso_from_any(created_at)
        lookup_instant = pd.Timestamp(created_at) if created_at else pd.Timestamp.now(tz="UTC")
        lookup_instant = (
            lookup_instant.tz_localize("UTC")
            if lookup_instant.tzinfo is None
            else lookup_instant.tz_convert("UTC")
        )

        trigger = _resolve_trigger(
            ledger_trigger_type="",
            ledger_trigger_source="",
            ledger_trigger_observed_at=None,
            evidence_rows=evidence_rows,
            computed_at=lookup_instant,
            game_ids=frozenset(tuesday_by_game),
        )

        game_ids = sorted(
            set(tuesday_by_game) | {str(value) for value in artifact.predictions.index}
        )
        games: list[GameDiffRow] = []
        for game_id in game_ids:
            row = _artifact_row(artifact, game_id)
            tue = tuesday_by_game.get(game_id)
            if tue is None:
                tue = GameSnapshot(
                    game_id=game_id,
                    home_team=str(row.get("home_team", "")) if row is not None else "",
                    away_team=str(row.get("away_team", "")) if row is not None else "",
                    kickoff=_iso_from_any(row.get("kickoff")) if row is not None else None,
                    market_line=None,
                    market_line_basis="no_data: game absent from the Tuesday lock card",
                    model_probability=None,
                    model_probability_basis="no_data: game absent from the Tuesday lock card",
                    pick_side=None,
                    pick_basis="no_data: game absent from the Tuesday lock card",
                    overlays_fired=None,
                    overlays_basis="no_data: game absent from the Tuesday lock card",
                )
            if row is None:
                refresh = GameSnapshot(
                    game_id=game_id,
                    home_team=tue.home_team,
                    away_team=tue.away_team,
                    kickoff=tue.kickoff,
                    market_line=None,
                    market_line_basis="no_data: game absent from this forecast artifact",
                    model_probability=None,
                    model_probability_basis="no_data: game absent from this forecast artifact",
                    pick_side=None,
                    pick_basis="no_data: game absent from this forecast artifact",
                    overlays_fired=None,
                    overlays_basis=(
                        "no_data: margin-predict artifacts do not record played-policy "
                        "overlay composition"
                    ),
                )
            else:
                refresh = GameSnapshot(
                    game_id=game_id,
                    home_team=str(row.get("home_team", "")),
                    away_team=str(row.get("away_team", "")),
                    kickoff=_iso_from_any(row.get("kickoff")),
                    market_line=_num(row.get("spread_line")),
                    market_line_basis="forecast_artifact_raw",
                    model_probability=_num(row.get("home_cover_probability")),
                    model_probability_basis="forecast_artifact_raw",
                    pick_side=_optional_text(row.get("bet_side")),
                    pick_basis="forecast_artifact_raw",
                    overlays_fired=None,
                    overlays_basis=(
                        "no_data: margin-predict artifacts do not record played-policy "
                        "overlay composition"
                    ),
                )
            games.append(_diff_game(tue, refresh))

        passes.append(
            RefreshPassDiff(
                refresh_run_id=directory.name,
                origin=PASS_ORIGIN_FORECAST_ARTIFACT,
                computed_at_utc=computed_at_ts,
                trigger_type=trigger.trigger_type,
                trigger_source=trigger.trigger_source,
                trigger_observed_at_utc=trigger.trigger_observed_at_utc,
                trigger_basis=trigger.basis,
                games=tuple(games),
                sources=_source_cells(
                    tuesday_lineage=tuesday_lineage,
                    refresh_lineage=artifact.lineage,
                    no_tuesday_reason="Tuesday lock forecast artifact has no lineage.json",
                    no_refresh_reason=f"{directory.name} has no lineage.json",
                ),
            )
        )
    return passes


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def build_snapshot_diff(
    season: int,
    week: int,
    *,
    artifacts_root: Path,
    data_root: Path | None = None,
    now: datetime | None = None,
) -> SnapshotDiff:
    """The full Tuesday-lock-vs-every-later-refresh-pass diff for one week.

    Read-only: never fits a model, never runs ``refresh-picks``/``weekly-run``/
    ``publish-predictions``, never writes to ``registry/``. ``data_root`` is
    optional and only feeds :attr:`SnapshotDiff.current_source_freshness`
    (a present-tense context section); when omitted that field is empty.
    """

    generated_at = _utc(now)
    tuesday = resolve_tuesday_lock(artifacts_root, season=season, week=week)

    tuesday_lineage: CardLineage | None = None
    if tuesday.forecast_directory:
        tuesday_artifact = _read_margin_prediction_artifact(Path(tuesday.forecast_directory))
        tuesday_lineage = tuesday_artifact.lineage if tuesday_artifact is not None else None

    sunday_lock: pd.Timestamp | None = None
    kickoffs = [game.kickoff for game in tuesday.games if game.kickoff]
    if kickoffs:
        try:
            sunday_lock = sunday_pick_lock(pd.Series(pd.to_datetime(kickoffs, utc=True)))
        except ValueError:
            sunday_lock = None

    evidence_rows = _load_evidence_log_rows(
        evidence_log_path(artifacts_root, season=season, week=week)
    )

    refresh_passes: list[RefreshPassDiff] = []
    if tuesday.resolved:
        refresh_passes.extend(
            _pick_revision_passes(
                artifacts_root,
                season=season,
                week=week,
                tuesday=tuesday,
                sunday_lock=sunday_lock,
                evidence_rows=evidence_rows,
            )
        )
        refresh_passes.extend(
            _forecast_artifact_passes(
                artifacts_root,
                season=season,
                week=week,
                tuesday=tuesday,
                tuesday_lineage=tuesday_lineage,
                evidence_rows=evidence_rows,
            )
        )
    refresh_passes.sort(key=lambda item: (item.computed_at_utc or "", item.refresh_run_id))

    current_source_freshness: dict[str, Any] = {}
    if data_root is not None:
        try:
            report = report_for_publication(
                data_root=data_root, artifacts_root=artifacts_root, now=generated_at.to_pydatetime()
            )
            current_source_freshness = report.to_metadata()
        except Exception:  # deliberately broad: an optional context block must never abort the diff
            current_source_freshness = {}

    return SnapshotDiff(
        season=season,
        week=week,
        generated_at_utc=_iso(generated_at),
        tuesday=tuesday,
        refresh_passes=tuple(refresh_passes),
        current_source_freshness=current_source_freshness,
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def to_dict(diff: SnapshotDiff) -> dict[str, Any]:
    return dataclasses.asdict(diff)


def to_json(diff: SnapshotDiff, *, indent: int = 2) -> str:
    return json.dumps(to_dict(diff), indent=indent, sort_keys=True, default=str) + "\n"


def _fmt_num(value: float | None) -> str:
    return "no_data" if value is None else f"{value:g}"


def _fmt_side(value: str | None) -> str:
    return value if value else "?"


def _fmt_overlays(row: GameDiffRow) -> str:
    if row.overlay_state == STATE_NO_DATA:
        return "no_data"
    parts: list[str] = []
    if row.overlays_added:
        parts.append("+" + ",".join(row.overlays_added))
    if row.overlays_removed:
        parts.append("-" + ",".join(row.overlays_removed))
    if row.overlays_unchanged:
        parts.append("=" + ",".join(row.overlays_unchanged))
    return "; ".join(parts) if parts else "(none fired)"


def render_markdown(diff: SnapshotDiff) -> str:
    """A compact, per-game-per-refresh Markdown table plus a flip summary.

    Every cell carries an explicit ``[state]`` tag (``changed`` /
    ``unchanged`` / ``no_data``, or ``same``/``flipped_*``/``no_data`` for
    picks) -- never a blank.
    """

    lines: list[str] = [f"# Decision-time snapshot diff: {diff.season} Week {diff.week}", ""]
    lines.append(f"Generated `{diff.generated_at_utc}`.")
    lines.append("")
    lines.append("## Tuesday lock")
    lines.append("")
    tuesday = diff.tuesday
    if not tuesday.resolved:
        lines.append(f"**UNRESOLVED.** {tuesday.detail}")
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.append(f"- basis: `{tuesday.basis}` -- {tuesday.detail}")
    lines.append(f"- forecast artifact: `{tuesday.forecast_artifact or 'n/a'}`")
    lines.append(f"- forecast created_at_utc: {tuesday.forecast_created_at_utc or 'n/a'}")
    lines.append(f"- paper-decision ledger rows for this week: {tuesday.ledger_rows}")
    lines.append(f"- lineage.json available: {tuesday.lineage_available}")
    lines.append(f"- games on card: {len(tuesday.games)}")
    lines.append("")

    if not diff.refresh_passes:
        lines.append("## Refresh passes")
        lines.append("")
        lines.append(
            "None found: no pick-revision ledger rows and no later margin_predictions "
            "forecast artifact for this season/week."
        )
        lines.append("")
        return "\n".join(lines) + "\n"

    total_games = 0
    total_flips = 0
    flip_summary_rows: list[tuple[str, int, int]] = []

    for refresh_pass in diff.refresh_passes:
        lines.append(f"## Refresh `{refresh_pass.refresh_run_id}` ({refresh_pass.origin})")
        lines.append("")
        lines.append(f"- computed_at_utc: {refresh_pass.computed_at_utc or 'n/a'}")
        lines.append(
            f"- trigger: `{refresh_pass.trigger_type}` / "
            f"`{refresh_pass.trigger_source or '(none)'}` (basis: {refresh_pass.trigger_basis}; "
            f"observed {refresh_pass.trigger_observed_at_utc or 'n/a'})"
        )
        lines.append("")
        lines.append(
            "| game | market line (Tue -> refresh) [state] | model probability "
            "(Tue -> refresh, delta) [state] | pick (Tue -> refresh) [state] | overlays [state] |"
        )
        lines.append("|---|---|---|---|---|")
        flips_this_pass = 0
        for row in refresh_pass.games:
            total_games += 1
            if row.pick_state not in ("same", STATE_NO_DATA):
                flips_this_pass += 1
                total_flips += 1
            market_cell = (
                f"{_fmt_num(row.tuesday_market_line)} -> {_fmt_num(row.refresh_market_line)} "
                f"[{row.market_line_state}]"
            )
            probability_cell = (
                f"{_fmt_num(row.tuesday_model_probability)} -> "
                f"{_fmt_num(row.refresh_model_probability)} "
                f"(delta {_fmt_num(row.probability_delta)}) [{row.probability_state}]"
            )
            pick_cell = (
                f"{_fmt_side(row.tuesday_pick_side)} -> {_fmt_side(row.refresh_pick_side)} "
                f"[{row.pick_state}]"
            )
            overlay_cell = f"{_fmt_overlays(row)} [{row.overlay_state}]"
            lines.append(
                f"| {row.away_team} at {row.home_team} | {market_cell} | {probability_cell} | "
                f"{pick_cell} | {overlay_cell} |"
            )
        if flips_this_pass:
            flip_summary_rows.append(
                (refresh_pass.refresh_run_id, flips_this_pass, len(refresh_pass.games))
            )
        lines.append("")

        if refresh_pass.sources:
            lines.append("Source timestamps (Tuesday vs this refresh, from lineage.json):")
            lines.append("")
            lines.append("| source | Tuesday captured_at | refresh captured_at | state |")
            lines.append("|---|---|---|---|")
            for cell in refresh_pass.sources:
                lines.append(
                    f"| {cell.source_id} | {cell.tuesday_captured_at or 'no_data'} | "
                    f"{cell.refresh_captured_at or 'no_data'} | {cell.state} |"
                )
            lines.append("")

    lines.append("## Flip summary")
    lines.append("")
    if not flip_summary_rows:
        lines.append(
            f"No pick flips across {len(diff.refresh_passes)} refresh pass(es) / "
            f"{total_games} game-row(s)."
        )
    else:
        lines.append("| refresh_run_id | games flipped | games in pass |")
        lines.append("|---|---|---|")
        for run_id_value, flipped, games_in_pass in flip_summary_rows:
            lines.append(f"| {run_id_value} | {flipped} | {games_in_pass} |")
        lines.append("")
        lines.append(
            f"Total: {total_flips} flip(s) across {len(diff.refresh_passes)} pass(es), "
            f"{total_games} game-row(s)."
        )
    lines.append("")

    if diff.current_source_freshness:
        lines.append(
            "## Current source freshness (live, evaluated now -- NOT a historical per-pass read)"
        )
        lines.append("")
        lines.append(
            f"State: `{diff.current_source_freshness.get('state')}` as of "
            f"{diff.current_source_freshness.get('evaluated_at_utc')}. See "
            "`docs/source_freshness_policy.md` for the state machine."
        )
        lines.append("")

    return "\n".join(lines) + "\n"


__all__ = [
    "OVERLAY_FLIP_COLUMNS",
    "PASS_ORIGIN_FORECAST_ARTIFACT",
    "PASS_ORIGIN_PICK_REVISION",
    "STATE_CHANGED",
    "STATE_NO_DATA",
    "STATE_UNCHANGED",
    "GameDiffRow",
    "GameSnapshot",
    "RefreshPassDiff",
    "SnapshotDiff",
    "SourceTimestampCell",
    "TuesdayLock",
    "build_snapshot_diff",
    "render_markdown",
    "resolve_tuesday_lock",
    "to_dict",
    "to_json",
]
