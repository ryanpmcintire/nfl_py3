"""ENG-15: ledger reconciliation and recovery.

Read-only. For one season/week, this module joins three things that are
supposed to agree but are written by different, independently-fail-open
code paths:

(a) **recorder result summaries** -- the JSON dict a recorder returned when it
    ran (``publish-predictions``/``weekly-run``'s printed summary, or the
    ``recorders`` section of an ENG-01 lock-day package manifest, if one is
    supplied). Every recorder in ``nfl_ats.cli`` is deliberately fail-open
    (``try/except -> {"recorded": 0, "error": ...}``), so this is the only
    durable evidence that a recorder skipped for a documented reason rather
    than silently writing nothing.
(b) **each append-only ledger's own rows** for that week -- read directly with
    ``pandas.read_parquet``, never through the strict ``load_*`` loaders in
    ``clv``/``prospective_scoring``/the five refresh modules. Those loaders
    raise ``DataContractError`` on ANY duplicate row anywhere in the whole
    file, which is the right behaviour for a recorder about to *write*, but
    would make this reconciler crash instead of reporting exactly the
    corruption it exists to find.
(c) **the published card's picks** -- parsed from the tracked
    ``CURRENT_PREDICTIONS.md`` (see ``nfl_ats.publishing.publish_active_predictions``),
    which is the externally-visible artefact a paper-decision row is supposed
    to match. Only the active model's paper-decision ledger is compared this
    way; challenger picks are never published to that file, so
    ``card_mismatch`` never fires for a challenger recorder.

Nothing here writes to a ledger, and nothing here mutates a historical row.
Running :func:`reconcile` twice against the same artifacts tree produces the
same report and leaves every ledger's bytes untouched -- see
``tests/test_ledger_reconcile.py::test_reconcile_is_idempotent_and_never_writes``.

Six classification buckets (the DoD's fixed vocabulary; nothing else may be
returned as ``status``):

* ``consistent`` -- ledger rows exist for the week, no duplicate idempotency
  keys, and (when comparable) the ledger's pick agrees with the published
  card.
* ``missing_rows`` -- a result summary explicitly declared rows were recorded
  this run, but the ledger has zero rows for the week. The recorder claimed
  success and the ledger disagrees.
* ``orphan_rows`` -- rows exist in the shared challenger ledger under a
  ``challenger_id`` that is not registered in ``challengers.json`` at all (a
  stray write: typo, deleted registration, hand-edited row). Challenger ids
  that ARE registered but are currently a non-``ACTIVE_PROSPECTIVE`` status
  (e.g. ``SUPERSEDED_BY_PROMOTION``) are reported separately as
  ``informational_inactive_challengers_with_rows`` and are deliberately never
  classified ``orphan_rows`` -- a challenger recorded picks while it was
  active and was superseded later is legitimate history, not corruption.
* ``duplicate_rows`` -- the week's own slice of a ledger contains more than
  one row under the same idempotency key (``game_id`` for the paper ledger;
  ``(challenger_id, game_id)`` for the shared challenger ledger; ``game_id``
  for a dedicated ledger -- see the per-recorder note on why the five refresh
  ledgers use a looser ``(game_id, refresh_run_id)`` key instead, since they
  are revision logs by design).
* ``card_mismatch`` -- (``active_model`` only) the ledger's recorded
  ``pick_side`` disagrees with the same game's pick as parsed from the
  published card, for a week where the card IS for the requested season/week.
* ``not_run`` -- everything else: no ledger rows and no summary evidence of a
  legitimate gate/error, or a recorder that is not wired into any CLI path
  yet (``PENDING_WIRING`` in ``scripts/lockday_verify.py``'s vocabulary).

See ``docs/ledger_reconciliation.md`` for the full design notes, the
per-recorder idempotency-key table, and the recovery-command derivation this
module uses in :func:`recovery_plan`.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import paper_decision_ledger_path
from nfl_ats.prospective_scoring import (
    ACTIVE_CHALLENGER_STATUS,
    challenger_ledger_path,
    challenger_registry_path,
)

#: The DoD's fixed classification vocabulary. Nothing else may appear in a
#: recorder row's ``status`` field.
STATUS_CONSISTENT = "consistent"
STATUS_MISSING_ROWS = "missing_rows"
STATUS_ORPHAN_ROWS = "orphan_rows"
STATUS_DUPLICATE_ROWS = "duplicate_rows"
STATUS_CARD_MISMATCH = "card_mismatch"
STATUS_NOT_RUN = "not_run"

ALL_STATUSES: tuple[str, ...] = (
    STATUS_CONSISTENT,
    STATUS_MISSING_ROWS,
    STATUS_ORPHAN_ROWS,
    STATUS_DUPLICATE_ROWS,
    STATUS_CARD_MISMATCH,
    STATUS_NOT_RUN,
)

#: Recorder id for the active model's own paper-decision ledger. Chosen to
#: never collide with a real ``challenger_id`` (registry ids are all
#: lowercase snake_case module/overlay names; this one is deliberately
#: distinct prose).
ACTIVE_MODEL_RECORDER_ID = "active_model"

#: Marks the pick text a card row carries for that week's nominated Best
#: Pick. Duplicated from ``nfl_ats.publishing.BEST_PICK_MARK`` (a one-line
#: literal, not logic) rather than imported, so this read-only module never
#: has to import the publisher.
_BEST_PICK_MARK = "★ "


# ---------------------------------------------------------------------------
# scripts/lockday_verify.py -- reused, not duplicated
# ---------------------------------------------------------------------------


def _load_lockday_verify(repo_root: Path) -> Any:
    """Import ``scripts/lockday_verify.py`` by file path.

    ``scripts/`` is not part of the installed package, so a normal
    ``import scripts.lockday_verify`` is not reliably available from library
    code. This is the same dynamic-import pattern
    ``nfl_ats.lockday_package._load_lockday_verify`` already uses for exactly
    this reason (and the same reason no ``[tool.mypy.overrides]`` entry is
    needed here: mypy never statically follows a ``spec_from_file_location``
    import).

    Reusing ``DEDICATED_LEDGERS``/``PENDING_REFRESH_LEDGERS`` from there
    instead of re-declaring the challenger -> dedicated-ledger mapping here
    is the "extend, do not duplicate" instruction applied to a mapping rather
    than a command surface: that dict is the maintained source of truth for
    which challengers have their own ledger file.
    """

    path = repo_root / "scripts" / "lockday_verify.py"
    spec = importlib.util.spec_from_file_location("nfl_ats_ledger_reconcile_lockday_verify", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load scripts/lockday_verify.py from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_lockday_verify_module(repo_root: Path) -> Any:
    """Public wrapper around :func:`_load_lockday_verify`, for CLI callers.

    ``scripts/ledger_reconcile.py`` uses this to fold one extra summary line
    (the existing wiring-level audit) into its human-readable output without
    re-implementing the dynamic import.
    """

    return _load_lockday_verify(repo_root)


# ---------------------------------------------------------------------------
# raw ledger reads -- deliberately bypass the strict load_* loaders
# ---------------------------------------------------------------------------


def _read_parquet_raw(path: Path) -> tuple[pd.DataFrame, str | None]:
    """Read a ledger parquet with no schema/duplicate validation.

    Returns ``(frame, read_error)``. A missing file is not an error (empty
    frame, no error string) -- ledgers legitimately do not exist before their
    first write. A file that fails to parse is reported via ``read_error``
    rather than raised, so one corrupted ledger cannot abort reconciliation
    of every other recorder.
    """

    if not path.is_file():
        return pd.DataFrame(), None
    try:
        return pd.read_parquet(path), None
    except Exception as error:  # deliberately broad: a read must never abort reconciliation
        return pd.DataFrame(), f"{type(error).__name__}: {error}"


def _week_rows(frame: pd.DataFrame, *, season: int, week: int) -> pd.DataFrame:
    if frame.empty or "season" not in frame.columns or "week" not in frame.columns:
        return frame.iloc[0:0]
    return frame.loc[
        (pd.to_numeric(frame["season"], errors="coerce") == season)
        & (pd.to_numeric(frame["week"], errors="coerce") == week)
    ]


def _run_id_rows(frame: pd.DataFrame, *, run_id: str | None, run_id_column: str) -> pd.DataFrame:
    if run_id is None or frame.empty or run_id_column not in frame.columns:
        return frame
    return frame.loc[frame[run_id_column].astype(str) == run_id]


def _duplicate_keys(frame: pd.DataFrame, key_columns: tuple[str, ...]) -> list[dict[str, Any]]:
    present = [column for column in key_columns if column in frame.columns]
    if not present or frame.empty:
        return []
    counts = frame.groupby(present, dropna=False).size()
    duplicated = counts.loc[counts.gt(1)]
    if duplicated.empty:
        return []
    keys = (
        duplicated.index.tolist() if len(present) > 1 else [(value,) for value in duplicated.index]
    )
    return [
        {**dict(zip(present, key, strict=True)), "row_count": int(count)}
        for key, count in zip(keys, duplicated.to_numpy(), strict=True)
    ]


# ---------------------------------------------------------------------------
# published card
# ---------------------------------------------------------------------------


class CardParseError(ValueError):
    """The card file exists but is not in the shape ``publish_active_predictions`` writes."""


def parse_published_card(card_path: Path) -> dict[str, Any]:
    """Parse ``CURRENT_PREDICTIONS.md`` (or an equivalent) into per-game picks.

    Deliberately reads the tracked Markdown file rather than the
    ``recommendations.csv`` the ledger recorder itself reads: the CSV is the
    ledger's OWN input, so comparing the ledger to it would only ever confirm
    the recorder copied its own source correctly. The Markdown file is the
    externally-visible artefact a user actually enters picks from -- if it
    has drifted from what the ledger recorded (a stale publish, a hand edit),
    that is the divergence worth catching.

    Returns a dict with ``exists``, ``season``, ``week``, ``model_id``, and
    ``games`` (a ``{game_id: {...}}`` map). A missing file returns
    ``exists: False`` rather than raising, since "no card published yet" is a
    normal state this tool must tolerate.
    """

    if not card_path.is_file():
        return {
            "exists": False,
            "path": str(card_path),
            "season": None,
            "week": None,
            "model_id": None,
            "games": {},
        }

    text = card_path.read_text(encoding="utf-8")
    heading = re.search(r"^# NFL ATS predictions:\s*(\d+)\s*Week\s*(\d+)", text, re.MULTILINE)
    if heading is None:
        raise CardParseError(
            f"{card_path}: missing the '# NFL ATS predictions: <season> Week "
            "<week>' heading `publish_active_predictions` always writes"
        )
    season = int(heading.group(1))
    week = int(heading.group(2))

    model_match = re.search(r"synchronized model `([^`]+)`", text)
    model_id = model_match.group(1) if model_match else None

    games: dict[str, dict[str, Any]] = {}
    lines = text.splitlines()
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("| Date") and "Matchup" in stripped:
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped.startswith("|"):
            break
        if set(stripped.replace("|", "").replace(":", "").replace("-", "").strip()) == set():
            continue  # the "|:---|:---|" separator row
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 4:
            continue
        _date, matchup, ats_prediction, _score = cells
        if " at " not in matchup:
            continue
        away_team, home_team = matchup.split(" at ", 1)
        away_team, home_team = away_team.strip(), home_team.strip()
        is_best_pick = ats_prediction.startswith(_BEST_PICK_MARK)
        prediction = ats_prediction.removeprefix(_BEST_PICK_MARK).strip()
        pick_team = prediction.split(" ", 1)[0].strip() if prediction else ""
        if pick_team == home_team:
            pick_side = "HOME"
        elif pick_team == away_team:
            pick_side = "AWAY"
        else:
            pick_side = None
        game_id = f"{season}_{week:02d}_{away_team}_{home_team}"
        games[game_id] = {
            "game_id": game_id,
            "away_team": away_team,
            "home_team": home_team,
            "pick_team": pick_team,
            "pick_side": pick_side,
            "ats_prediction": prediction,
            "is_best_pick": is_best_pick,
        }

    return {
        "exists": True,
        "path": str(card_path),
        "season": season,
        "week": week,
        "model_id": model_id,
        "games": games,
    }


# ---------------------------------------------------------------------------
# recorder result summaries (run-summary JSON and/or an ENG-01 package manifest)
# ---------------------------------------------------------------------------


class Declaration:
    """What a recorder's own result JSON says happened, for one recorder."""

    __slots__ = ("error", "reason", "recorded", "source")

    def __init__(
        self,
        recorded: int | None = None,
        reason: str | None = None,
        error: str | None = None,
        source: str = "none",
    ) -> None:
        self.recorded = recorded
        self.reason = reason
        self.error = error
        self.source = source

    def as_dict(self) -> dict[str, Any]:
        return {
            "declared_recorded": self.recorded,
            "declared_reason": self.reason,
            "declared_error": self.error,
            "declared_source": self.source,
        }


def _walk_challenger_results(node: Any, by_id: dict[str, dict[str, Any]]) -> None:
    """Collect every ``{"challenger_id": ..., ...}`` dict anywhere in a JSON tree.

    Mirrors ``scripts/lockday_verify.py``'s ``gated_skips`` walk (small enough,
    and specific enough to that file's own docstring contract, that importing
    it felt like tighter coupling than re-stating six lines of tree-walk).
    """

    if isinstance(node, list):
        for item in node:
            _walk_challenger_results(item, by_id)
        return
    if not isinstance(node, dict):
        return
    challenger_id = node.get("challenger_id")
    if isinstance(challenger_id, str):
        by_id.setdefault(challenger_id, node)
    for value in node.values():
        _walk_challenger_results(value, by_id)


def _load_run_summary(run_summary_path: Path | None) -> dict[str, Any] | None:
    if run_summary_path is None:
        return None
    payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{run_summary_path} is not a JSON object")
    return payload


def _load_package_manifest(package_path: Path | None) -> dict[str, Any] | None:
    """Best-effort read of an ENG-01 lock-day package manifest, if one exists.

    ``nfl_ats.lockday_package`` is being built concurrently by another agent
    in this same session; this reconciler must work with or without it. The
    import is local and defensive: if the module is not importable, or the
    file at ``package_path`` is not its manifest shape, this degrades to
    "no package supplied" rather than raising.
    """

    if package_path is None:
        return None
    try:
        from nfl_ats.lockday_package import load_package

        payload = load_package(package_path)
    except Exception:
        # Fall back to a bare JSON read so a hand-supplied or future-shaped
        # manifest can still contribute its "recorders"/"ledgers" sections
        # even if nfl_ats.lockday_package changes shape or is unavailable.
        try:
            manifest_file = (
                package_path / "manifest.json" if package_path.is_dir() else package_path
            )
            payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            return None
    return payload if isinstance(payload, dict) else None


#: ``recorder_id`` (paper ledger / dedicated-refresh challengers) -> the
#: ledger name an ENG-01 package manifest's ``ledgers`` list uses. Only the
#: recorders with a ONE ledger file to themselves are listed: the shared
#: challenger ledger holds every other challenger's rows interleaved, so its
#: package-level ``appended_rows`` cannot be attributed to one challenger_id
#: -- those recorders are read from ``recorders.by_challenger_id`` instead
#: (see :func:`_declarations_from_package`), which IS keyed per-challenger.
_PACKAGE_LEDGER_NAMES: dict[str, str] = {
    ACTIVE_MODEL_RECORDER_ID: "paper_decisions",
    "model_only_refresh_incumbent": "pick_revisions",
    "injury_signal_refresh_tilt": "injury_signal_refresh_decisions",
    "nflcom_friday_refresh_out2_starters_v1": "nflcom_friday_refresh_decisions",
    "inactives_refresh_v1": "inactives_refresh_decisions",
    "crew_tilt_refresh_v1": "crew_tilt_refresh_decisions",
}


def _declarations_from_package(manifest: dict[str, Any]) -> dict[str, Declaration]:
    declared: dict[str, Declaration] = {}
    by_challenger = (manifest.get("recorders") or {}).get("by_challenger_id") or {}
    if isinstance(by_challenger, dict):
        for challenger_id, result in by_challenger.items():
            if not isinstance(result, dict):
                continue
            recorded = result.get("recorded")
            declared[str(challenger_id)] = Declaration(
                recorded=int(recorded) if isinstance(recorded, (int, float)) else None,
                reason=result.get("reason") if isinstance(result.get("reason"), str) else None,
                error=result.get("error") if isinstance(result.get("error"), str) else None,
                source="package_recorder_result",
            )
    ledgers = manifest.get("ledgers") or []
    ledger_by_name = {
        str(entry.get("ledger")): entry for entry in ledgers if isinstance(entry, dict)
    }
    for recorder_id, ledger_name in _PACKAGE_LEDGER_NAMES.items():
        if recorder_id in declared:
            continue
        entry = ledger_by_name.get(ledger_name)
        if entry is None:
            continue
        appended = entry.get("appended_rows")
        declared[recorder_id] = Declaration(
            recorded=int(appended) if isinstance(appended, (int, float)) else None,
            error=entry.get("error") if isinstance(entry.get("error"), str) else None,
            source="package_ledger_diff",
        )
    return declared


def _declarations_from_run_summary(run_summary: dict[str, Any]) -> dict[str, Declaration]:
    declared: dict[str, Declaration] = {}
    by_challenger: dict[str, dict[str, Any]] = {}
    _walk_challenger_results(run_summary, by_challenger)
    for challenger_id, result in by_challenger.items():
        recorded = result.get("recorded")
        declared[challenger_id] = Declaration(
            recorded=int(recorded) if isinstance(recorded, (int, float)) else None,
            reason=result.get("reason") if isinstance(result.get("reason"), str) else None,
            error=result.get("error") if isinstance(result.get("error"), str) else None,
            source="run_summary",
        )
    clv_ledger = run_summary.get("clv_ledger")
    if isinstance(clv_ledger, dict):
        recorded = clv_ledger.get("recorded")
        declared[ACTIVE_MODEL_RECORDER_ID] = Declaration(
            recorded=int(recorded) if isinstance(recorded, (int, float)) else None,
            reason=clv_ledger.get("reason") if isinstance(clv_ledger.get("reason"), str) else None,
            error=clv_ledger.get("error") if isinstance(clv_ledger.get("error"), str) else None,
            source="run_summary",
        )
    return declared


def build_declarations(
    *,
    run_summary: dict[str, Any] | None,
    package_manifest: dict[str, Any] | None,
) -> dict[str, Declaration]:
    """Merge run-summary and package declarations, package taking priority.

    The package manifest's ``ledgers`` section is a measured row-count delta
    (read after the run actually wrote), which is strictly more trustworthy
    than a recorder's own self-reported ``"recorded"`` count; where both are
    available for the same recorder, the package wins.
    """

    declared = _declarations_from_run_summary(run_summary) if run_summary else {}
    if package_manifest:
        declared.update(_declarations_from_package(package_manifest))
    return declared


# ---------------------------------------------------------------------------
# recorder inventory
# ---------------------------------------------------------------------------


def _registry_entries(artifacts_root: Path) -> dict[str, dict[str, Any]]:
    path = challenger_registry_path(artifacts_root)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("challengers") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return {}
    return {
        str(entry["challenger_id"]): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("challenger_id") is not None
    }


def _derive_rerun_command(raw: str, *, season: int, week: int) -> tuple[str | None, str]:
    """Best-effort runnable command from a registry's ``weekly_recording_command``.

    Every entry in ``challengers.json`` is either (a) a directly runnable
    command, optionally followed by a parenthetical explanation, or (b) prose
    beginning ``"N/A -- ..."`` for challengers with no standalone command (see
    ``docs/ledger_reconciliation.md``). Case (b) returns ``None`` rather than
    fabricating a command; the raw text is always returned unchanged as the
    second element so the explanation is never lost.
    """

    raw = raw.strip()
    prefix = raw.split(" (", 1)[0].strip()
    if not prefix or prefix.upper().startswith("N/A"):
        return None, raw
    command = re.sub(r"--season\s+\d+", f"--season {season}", prefix)
    command = re.sub(r"--week\s+(<N>|\d+)", f"--week {week}", command)
    return command, raw


# ---------------------------------------------------------------------------
# per-recorder classification
# ---------------------------------------------------------------------------


def _classify(
    *,
    week_rows: pd.DataFrame,
    key_columns: tuple[str, ...],
    declared: Declaration,
    read_error: str | None,
    card_comparable: bool,
    card: dict[str, Any],
    ledger_pick_column: str,
    wired: bool,
    wiring_note: str,
) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns ``(status, note, duplicate_keys, card_mismatches)``."""

    duplicates = _duplicate_keys(week_rows, key_columns)
    if duplicates:
        return (
            STATUS_DUPLICATE_ROWS,
            f"{len(duplicates)} idempotency-key collision(s) in this week's own ledger slice",
            duplicates,
            [],
        )

    if read_error is not None:
        return STATUS_NOT_RUN, f"ledger read failed: {read_error}", [], []

    rows_this_week = len(week_rows)

    if declared.recorded is not None and declared.recorded > 0 and rows_this_week == 0:
        return (
            STATUS_MISSING_ROWS,
            f"result summary declared {declared.recorded} row(s) recorded this run, but the "
            "ledger has zero rows for this week",
            [],
            [],
        )

    mismatches: list[dict[str, Any]] = []
    if card_comparable and rows_this_week > 0 and card.get("exists") and bool(card.get("games")):
        card_games = card["games"]
        for _, row in week_rows.iterrows():
            game_id = str(row.get("game_id"))
            card_game = card_games.get(game_id)
            if card_game is None or card_game.get("pick_side") is None:
                continue
            ledger_pick = row.get(ledger_pick_column)
            if ledger_pick is None or pd.isna(ledger_pick):
                continue
            if str(ledger_pick) != str(card_game["pick_side"]):
                mismatches.append(
                    {
                        "game_id": game_id,
                        "ledger_pick_side": str(ledger_pick),
                        "card_pick_side": str(card_game["pick_side"]),
                        "card_pick_team": card_game.get("pick_team"),
                    }
                )
        if mismatches:
            return (
                STATUS_CARD_MISMATCH,
                f"{len(mismatches)} game(s) where the ledger's recorded pick disagrees with the "
                "published card's current pick",
                [],
                mismatches,
            )

    if rows_this_week > 0:
        return STATUS_CONSISTENT, "", [], []

    if not wired:
        return STATUS_NOT_RUN, wiring_note, [], []
    if declared.reason:
        return STATUS_NOT_RUN, f"recorder reported a gate: {declared.reason}", [], []
    if declared.error:
        return STATUS_NOT_RUN, f"recorder reported an error: {declared.error}", [], []
    if declared.recorded == 0:
        return (
            STATUS_NOT_RUN,
            "recorder ran and declared 0 rows recorded (nothing new, or already recorded)",
            [],
            [],
        )
    note = "no ledger rows for this week"
    if declared.source == "none":
        note += " (no result summary or package manifest supplied for this recorder)"
    return STATUS_NOT_RUN, note, [], []


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def reconcile(
    artifacts_root: Path,
    *,
    season: int,
    week: int,
    run_id: str | None = None,
    repo_root: Path | None = None,
    card_path: Path | None = None,
    run_summary_path: Path | None = None,
    package_path: Path | None = None,
) -> dict[str, Any]:
    """Read-only reconciliation report for one season/week (and optional run id).

    Never writes anything. Two calls with the same inputs against an
    unchanged artifacts tree return an equal report (module docstring has the
    full contract).
    """

    repo_root = repo_root or Path.cwd()
    card_path = card_path or (repo_root / "CURRENT_PREDICTIONS.md")

    run_summary = _load_run_summary(run_summary_path)
    package_manifest = _load_package_manifest(package_path)
    declared_by_id = build_declarations(run_summary=run_summary, package_manifest=package_manifest)

    card = parse_published_card(card_path)
    card_for_week = (
        bool(card.get("exists")) and card.get("season") == season and card.get("week") == week
    )

    registry_by_id = _registry_entries(artifacts_root)
    registry_available = bool(registry_by_id)
    active_challenger_ids = sorted(
        challenger_id
        for challenger_id, entry in registry_by_id.items()
        if entry.get("status") == ACTIVE_CHALLENGER_STATUS
    )

    verify_module: Any = None
    dedicated_ledgers: dict[str, dict[str, Any]] = {}
    pending_refresh_ledgers: dict[str, dict[str, Any]] = {}
    standalone_pending_wiring: frozenset[str] = frozenset()
    try:
        verify_module = _load_lockday_verify(repo_root)
        dedicated_ledgers = dict(getattr(verify_module, "DEDICATED_LEDGERS", {}))
        pending_refresh_ledgers = dict(getattr(verify_module, "PENDING_REFRESH_LEDGERS", {}))
        standalone_pending_wiring = frozenset(
            getattr(verify_module, "STANDALONE_PENDING_WIRING", ())
        )
    except Exception:
        pass  # degrade to shared-ledger-only classification; noted per-recorder below

    recorders: list[dict[str, Any]] = []

    # --- active model / paper-decision ledger --------------------------------
    paper_path = paper_decision_ledger_path(artifacts_root)
    paper_frame, paper_read_error = _read_parquet_raw(paper_path)
    paper_frame = _run_id_rows(paper_frame, run_id=run_id, run_id_column="forecast_artifact")
    paper_week = _week_rows(paper_frame, season=season, week=week)
    status, note, duplicates, mismatches = _classify(
        week_rows=paper_week,
        key_columns=("game_id",),
        declared=declared_by_id.get(ACTIVE_MODEL_RECORDER_ID, Declaration()),
        read_error=paper_read_error,
        card_comparable=True,
        card=card,
        ledger_pick_column="pick_side",
        wired=True,
        wiring_note="",
    )
    recorders.append(
        {
            "recorder_id": ACTIVE_MODEL_RECORDER_ID,
            "kind": "paper_ledger",
            "ledger_path": str(paper_path),
            "idempotency_key": ["game_id"],
            "rows_this_week": len(paper_week),
            "status": status,
            "note": note,
            "duplicate_keys": duplicates,
            "card_mismatches": mismatches,
            **declared_by_id.get(ACTIVE_MODEL_RECORDER_ID, Declaration()).as_dict(),
        }
    )

    # --- shared challenger ledger (read once, sliced per challenger) ---------
    shared_path = challenger_ledger_path(artifacts_root)
    shared_frame, shared_read_error = _read_parquet_raw(shared_path)
    shared_frame = _run_id_rows(shared_frame, run_id=run_id, run_id_column="source_artifact")
    shared_week_all = _week_rows(shared_frame, season=season, week=week)

    informational: list[dict[str, Any]] = []
    if not shared_week_all.empty and "challenger_id" in shared_week_all.columns:
        present_ids = sorted(shared_week_all["challenger_id"].astype(str).unique().tolist())
        for challenger_id in present_ids:
            if challenger_id in dedicated_ledgers or challenger_id in pending_refresh_ledgers:
                # Known limitation: a stray row written into the SHARED ledger under a
                # dedicated-ledger challenger's id (never expected from any production
                # write path) is silently skipped here rather than flagged, because that
                # id's real evidence is read from its own ledger file in the loop below.
                continue
            entry = registry_by_id.get(challenger_id)
            if entry is None:
                rows = shared_week_all.loc[
                    shared_week_all["challenger_id"].astype(str).eq(challenger_id)
                ]
                duplicates = _duplicate_keys(rows, ("challenger_id", "game_id"))
                recorders.append(
                    {
                        "recorder_id": challenger_id,
                        "kind": "orphan_shared_ledger_challenger",
                        "ledger_path": str(shared_path),
                        "idempotency_key": ["challenger_id", "game_id"],
                        "rows_this_week": len(rows),
                        "status": STATUS_DUPLICATE_ROWS if duplicates else STATUS_ORPHAN_ROWS,
                        "note": (
                            f"{len(duplicates)} duplicate row(s) under an unregistered "
                            "challenger_id"
                            if duplicates
                            else "rows recorded under a challenger_id that is not present in "
                            "artifacts/prospective/challengers.json at all"
                        ),
                        "duplicate_keys": duplicates,
                        "card_mismatches": [],
                        **Declaration(source="none").as_dict(),
                    }
                )
            elif entry.get("status") != ACTIVE_CHALLENGER_STATUS:
                informational.append(
                    {
                        "challenger_id": challenger_id,
                        "registry_status": entry.get("status"),
                        "rows_this_week": len(
                            shared_week_all.loc[
                                shared_week_all["challenger_id"].astype(str).eq(challenger_id)
                            ]
                        ),
                        "note": "registered but not ACTIVE_PROSPECTIVE; excluded from recorder "
                        "classification -- see the module docstring on why this is not "
                        "treated as orphan_rows",
                    }
                )

    # --- every currently active challenger ------------------------------------
    for challenger_id in active_challenger_ids:
        entry = registry_by_id.get(challenger_id, {})
        raw_command = str(entry.get("weekly_recording_command", ""))
        rerun_command, rerun_command_raw = _derive_rerun_command(
            raw_command, season=season, week=week
        )
        declared = declared_by_id.get(challenger_id, Declaration())

        dedicated = dedicated_ledgers.get(challenger_id) or pending_refresh_ledgers.get(
            challenger_id
        )
        if dedicated is not None:
            path = artifacts_root / str(dedicated["ledger"])
            frame, read_error = _read_parquet_raw(path)
            frame = _run_id_rows(frame, run_id=run_id, run_id_column="refresh_run_id")
            week_rows = _week_rows(frame, season=season, week=week)
            wired = bool(dedicated.get("wired", True))
            if (
                challenger_id in pending_refresh_ledgers
                and raw_command
                and "N/A YET" not in raw_command
            ):
                wired = "refresh-picks --record-decisions" in raw_command
            wiring_note = dedicated.get(
                "note", "recorder is not wired into refresh-picks yet (PENDING_WIRING)"
            )
            key_columns = ("game_id", "refresh_run_id")
            status, note, duplicates, mismatches = _classify(
                week_rows=week_rows,
                key_columns=key_columns,
                declared=declared,
                read_error=read_error,
                card_comparable=False,
                card=card,
                ledger_pick_column="",
                wired=wired,
                wiring_note=wiring_note,
            )
            recorders.append(
                {
                    "recorder_id": challenger_id,
                    "kind": "dedicated_ledger",
                    "ledger_path": str(path),
                    "idempotency_key": list(key_columns),
                    "rows_this_week": len(week_rows),
                    "status": status,
                    "note": note,
                    "duplicate_keys": duplicates,
                    "card_mismatches": mismatches,
                    "rerun_command": rerun_command,
                    "rerun_command_raw": rerun_command_raw,
                    **declared.as_dict(),
                }
            )
            continue

        # shared ledger
        rows = (
            shared_week_all.loc[shared_week_all["challenger_id"].astype(str).eq(challenger_id)]
            if not shared_week_all.empty and "challenger_id" in shared_week_all.columns
            else shared_week_all.iloc[0:0]
        )
        standalone_pending = (
            "scripts/record_" in raw_command or challenger_id in standalone_pending_wiring
        )
        status, note, duplicates, mismatches = _classify(
            week_rows=rows,
            key_columns=("challenger_id", "game_id"),
            declared=declared,
            read_error=shared_read_error,
            card_comparable=False,
            card=card,
            ledger_pick_column="",
            wired=not standalone_pending,
            wiring_note="standalone recorder exists but is not wired into the publish/refresh "
            "CLI yet (PENDING_WIRING)",
        )
        recorders.append(
            {
                "recorder_id": challenger_id,
                "kind": "shared_challenger_ledger",
                "ledger_path": str(shared_path),
                "idempotency_key": ["challenger_id", "game_id"],
                "rows_this_week": len(rows),
                "status": status,
                "note": note,
                "duplicate_keys": duplicates,
                "card_mismatches": mismatches,
                "rerun_command": rerun_command,
                "rerun_command_raw": rerun_command_raw,
                **declared.as_dict(),
            }
        )

    recorders.sort(key=lambda row: (row["status"] == STATUS_CONSISTENT, row["recorder_id"]))
    summary = {
        status: sum(1 for row in recorders if row["status"] == status) for status in ALL_STATUSES
    }
    all_consistent = all(row["status"] == STATUS_CONSISTENT for row in recorders)

    return {
        "season": season,
        "week": week,
        "run_id": run_id,
        "artifacts_root": str(artifacts_root),
        "repo_root": str(repo_root),
        "card_path": str(card_path),
        "card_available_for_requested_week": card_for_week,
        "card_model_id": card.get("model_id"),
        "card_season": card.get("season"),
        "card_week": card.get("week"),
        "registry_available": registry_available,
        "run_summary_path": str(run_summary_path) if run_summary_path else None,
        "package_path": str(package_path) if package_path else None,
        "lockday_verify_available": verify_module is not None,
        "recorders": recorders,
        "informational_inactive_challengers_with_rows": informational,
        "summary": summary,
        "all_consistent": all_consistent,
        "recovery_plan": recovery_plan(recorders),
    }


# ---------------------------------------------------------------------------
# recovery plan
# ---------------------------------------------------------------------------

#: Safety notes keyed by ``kind``, applied when a recorder is non-consistent.
#: Grounded in what each recorder's own write path actually does (module
#: docstring cites the exact functions/lines this was read from):
#: ``record_paper_decisions``/``record_challenger_decisions`` dedupe against
#: existing rows before appending (``already = ... isin(existing[...])``), so
#: re-running never creates a duplicate; the five refresh ledgers are
#: revision logs with no such check, so a re-run is safe from corruption but
#: intentionally NOT a no-op -- it appends a fresh revision row every time.
_SAFETY_NOTES: dict[str, str] = {
    "paper_ledger": (
        "record_paper_decisions (src/nfl_ats/clv.py) skips any game_id already present in "
        "the ledger before appending; re-running never creates a duplicate row."
    ),
    "shared_challenger_ledger": (
        "record_challenger_decisions (src/nfl_ats/prospective_scoring.py) skips any "
        "(challenger_id, game_id) already present before appending; re-running never "
        "creates a duplicate row."
    ),
    "dedicated_ledger": (
        "This is a revision log, not a single-write ledger: its record_* function has no "
        "already-recorded check against existing rows, so re-running the command APPENDS A "
        "NEW REVISION ROW every time (by design -- that is what lets a late-week refresh "
        "record a changed pick). Re-running is safe from corruption but is not a no-op; do "
        "not re-run it merely to 'fix' a legitimate not_run/consistent state."
    ),
    "orphan_shared_ledger_challenger": (
        "No active registration exists for this challenger_id, so there is no documented "
        "command to re-run. Do not delete or edit the ledger rows (append-only, read-only "
        "for this tool) -- first check challengers.json for a typo or a retired id before "
        "treating this as corruption."
    ),
}


def recovery_plan(recorders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One recovery entry per non-``consistent`` recorder, report-only.

    Never executes anything. ``rerun_command`` is ``None`` when the registry
    itself documents no standalone command for that recorder (see
    :func:`_derive_rerun_command`); the caller is expected to read
    ``rerun_command_raw``/``note`` in that case.
    """

    plan: list[dict[str, Any]] = []
    for row in recorders:
        if row["status"] == STATUS_CONSISTENT:
            continue
        kind = row["kind"]
        safety_note = _SAFETY_NOTES.get(
            kind,
            "Idempotency behaviour for this recorder kind is "
            "undocumented; verify before re-running.",
        )
        entry: dict[str, Any] = {
            "recorder_id": row["recorder_id"],
            "status": row["status"],
            "note": row["note"],
            "rerun_command": row.get("rerun_command"),
            "rerun_command_raw": row.get("rerun_command_raw"),
            "rerun_is_safe": kind != "orphan_shared_ledger_challenger",
            "rerun_safety_note": safety_note,
        }
        if row["recorder_id"] == ACTIVE_MODEL_RECORDER_ID:
            entry["rerun_command"] = "nfl-ats publish-predictions --record-decisions"
            entry["rerun_command_raw"] = entry["rerun_command"]
            entry["rerun_safety_note"] += (
                " Note: publish-predictions does not take --season/--week; it always acts on "
                "whichever forecast is currently linked as the active model, so confirm that "
                "is the requested season/week before re-running."
            )
        if row["status"] == STATUS_DUPLICATE_ROWS:
            entry["rerun_is_safe"] = False
            entry["rerun_safety_note"] = (
                "Duplicate rows under the same idempotency key were found first. "
                + safety_note
                + " The strict load_* loader for this ledger will raise DataContractError on "
                "read until the duplicate is resolved; re-running the recorder will not fix "
                "existing duplicate rows and may add more. Resolve the duplication manually "
                "before re-running anything against this ledger."
            )
        if row["status"] == STATUS_CARD_MISMATCH:
            entry["rerun_safety_note"] += (
                " This may be EXPECTED, not a defect: republishing a card with a moved line "
                "never rewrites an already-recorded CLV anchor (docs/prospective_evidence.md, "
                "'anti-backdating guarantee'). Confirm which side is actually correct before "
                "treating this as something to fix."
            )
        plan.append(entry)
    return plan


# ---------------------------------------------------------------------------
# human rendering
# ---------------------------------------------------------------------------


def render(report: dict[str, Any]) -> str:
    lines = [
        f"ledger reconciliation  {report['season']} week {report['week']}"
        + (f"  run_id={report['run_id']}" if report.get("run_id") else ""),
        f"  artifacts root : {report['artifacts_root']}",
        f"  card           : {report['card_path']}"
        + (
            f"  (season {report['card_season']} week {report['card_week']}, model "
            f"{report['card_model_id']})"
            if report["card_available_for_requested_week"]
            else "  (not published for the requested season/week -- card_mismatch cannot fire)"
        ),
        f"  registry       : {'loaded' if report['registry_available'] else 'NOT FOUND'}",
        f"  inputs         : run_summary={report['run_summary_path'] or '(none)'}  "
        f"package={report['package_path'] or '(none)'}",
        "",
    ]
    recorders = report["recorders"]
    width = max((len(row["recorder_id"]) for row in recorders), default=10)
    marker = {
        STATUS_CONSISTENT: "ok  ",
        STATUS_MISSING_ROWS: "!!  ",
        STATUS_ORPHAN_ROWS: "??  ",
        STATUS_DUPLICATE_ROWS: "DUP ",
        STATUS_CARD_MISMATCH: "MIS ",
        STATUS_NOT_RUN: "--  ",
    }
    for row in recorders:
        line = (
            f"  {marker[row['status']]}{row['recorder_id']:<{width}}  {row['status']:<14}  "
            f"{row['rows_this_week']:>3} rows"
        )
        if row["note"]:
            line += f"   ({row['note']})"
        lines.append(line)
    summary = report["summary"]
    lines += [
        "",
        "  summary: "
        + ", ".join(f"{count} {status}" for status, count in summary.items() if count),
    ]
    if report["informational_inactive_challengers_with_rows"]:
        lines.append("")
        lines.append("  informational (registered, not ACTIVE_PROSPECTIVE, has rows this week):")
        for item in report["informational_inactive_challengers_with_rows"]:
            lines.append(
                f"    {item['challenger_id']} [{item['registry_status']}]  "
                f"{item['rows_this_week']} rows"
            )
    if report["recovery_plan"]:
        lines.append("")
        lines.append("  recovery plan:")
        for entry in report["recovery_plan"]:
            command = (
                entry["rerun_command"] or f"(no standalone command: {entry['rerun_command_raw']})"
            )
            lines.append(f"    {entry['recorder_id']} [{entry['status']}]")
            lines.append(f"      re-run: {command}")
            lines.append(f"      safe: {entry['rerun_is_safe']}  -- {entry['rerun_safety_note']}")
    return "\n".join(lines)


__all__ = [
    "ACTIVE_MODEL_RECORDER_ID",
    "ALL_STATUSES",
    "STATUS_CARD_MISMATCH",
    "STATUS_CONSISTENT",
    "STATUS_DUPLICATE_ROWS",
    "STATUS_MISSING_ROWS",
    "STATUS_NOT_RUN",
    "STATUS_ORPHAN_ROWS",
    "CardParseError",
    "Declaration",
    "build_declarations",
    "load_lockday_verify_module",
    "parse_published_card",
    "reconcile",
    "recovery_plan",
    "render",
]
