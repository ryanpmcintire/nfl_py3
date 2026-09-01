"""The data contract behind the transparency dashboard's Model Ledger tabular
view: one row per arm (the promoted card plus every registered prospective
challenger), each explicitly badged PROMOTED / CHALLENGER / RETIRED /
SUPERSEDED, sortable by track record and confidence, with per-arm evidence
linked into ``registry/weak_signals.json`` under the same discipline
:func:`nfl_ats.findings_registry.validate_curation` enforces for curated
prose.

Everything here is a pure reader/builder over injected inputs -- this module
never writes an artifact and never edits a registry.

Evidence-linkage rules (challenger_id -> registry keys):

1. ``evidence.registry_source`` is normalized (string or list of strings,
   split on commas). Any fragment containing ``registry/weak_signals.json``
   yields a candidate key: the text after its first ``:``, truncated at the
   first whitespace -- which strips attached prose such as ``(the latter NOT
   the basis ...)``. Marker-less comma-continuation fragments that FOLLOW a
   ``registry/weak_signals.json`` fragment are treated as bare candidate
   keys (the ``...json: key_one, key_two`` shorthand), until a fragment
   starting with ``(`` (attached prose) or a non-registry path ends the run.
   Every candidate is kept only if it exists in the live registry, so stray
   prose tokens can never become evidence.
2. Candidate keys are kept ONLY if they exist in the live weak-signals
   registry loaded from ``weak_signals_path``. Unknown fragments are dropped,
   never invented.
3. Fallback (used only when steps 1-2 produced nothing AND
   ``registry_source`` named no weak_signals fragment): a registry key equal
   to the challenger_id or ending with ``_<challenger_id>`` (the
   ``mod08_smooth_cdf_mapping`` <-> ``smooth_cdf_mapping`` convention).
   A challenger with no admissible link gets an explicit empty evidence
   tuple.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Literal

from nfl_ats.dashboard.findings_content import (
    CHALLENGER_DISPLAY_NAMES,
    LEDGER_PROMOTED_CAVEAT,
    PLAYED_CARD_EXPECTATION_PERCENT,
)
from nfl_ats.dashboard.viz import p_plus_text
from nfl_ats.findings_registry import fingerprint

#: The two interval units a :class:`TrackRecord` can carry (2026-08-31
#: browser-QA fix): the PROMOTED row's interval is a season accuracy-
#: proportion CI (a rate, 0..1, correctly percent-formatted -- e.g.
#: ``[0.508, 0.535]`` -> ``[50.8%, 53.5%]``); every CHALLENGER row's interval
#: comes from an ``evidence`` field whose own name says "points" (
#: ``week_blocked_interval_points`` / ``interval_points`` /
#: ``source_interval_points`` -- see :data:`_INTERVAL_KEYS`), i.e. an
#: accuracy-POINTS effect delta, which must never be multiplied by 100 and
#: shown with a ``%`` sign (that bug rendered e.g. accuracy-points interval
#: ``[0.29, 2.038]`` as ``[29.0%, 203.8%]`` on the generated page). A
#: renderer must branch on this field rather than ever guessing a unit from
#: a number's magnitude.
IntervalUnit = Literal["accuracy_rate", "accuracy_points"]

STATUS_BADGE_PROMOTED = "PROMOTED"
STATUS_BADGE_CHALLENGER = "CHALLENGER"
STATUS_BADGE_RETIRED = "RETIRED"
STATUS_BADGE_SUPERSEDED = "SUPERSEDED"

_CHALLENGER_STATUS_BADGES = {
    "ACTIVE_PROSPECTIVE": STATUS_BADGE_CHALLENGER,
    "SUPERSEDED_BY_PROMOTION": STATUS_BADGE_SUPERSEDED,
    "CLOSED_BEFORE_ACTIVATION": STATUS_BADGE_RETIRED,
}

_RETIRED_STATUS_PREFIXES = ("DEACTIVATED_",)

_REGISTRY_SOURCE_MARKER = "registry/weak_signals.json"
_REGISTRY_FALLBACK_PREFIX = "_"

_GAMES_KEYS = ("sample_games", "paired_games", "sample_games_paired", "games")
_ACCURACY_KEYS = ("candidate_accuracy_at_opener", "accuracy", "source_population_accuracy")
_INTERVAL_KEYS = (
    "week_blocked_interval_points",
    "interval_points",
    "source_interval_points",
    "interval",
)
_PROBABILITY_KEYS = ("probability_positive", "source_probability_positive")

_NUMERIC_TOKEN = re.compile(r"-?\d+(?:\.\d+)?")


class LedgerError(ValueError):
    """A ledger row violates the contract: a badge it cannot support, an
    evidence key the registry does not contain, a summary sentence quoting a
    number no cited field produces, or evidence whose recorded content moved
    after the ledger was built."""


@dataclass(frozen=True)
class EvidenceRef:
    registry_key: str
    effect: float | None
    probability_positive: float | None
    classification: str | None
    fingerprint: str


@dataclass(frozen=True)
class TrackRecord:
    games: int | None
    accuracy: float | None
    interval_low: float | None
    interval_high: float | None
    #: See :data:`IntervalUnit` -- ``"accuracy_rate"`` for the promoted row's
    #: season-CI proportion, ``"accuracy_points"`` for every challenger row's
    #: points-effect interval. A renderer must format the two differently.
    interval_unit: IntervalUnit
    grade: str
    artifact_ref: str | None


@dataclass(frozen=True)
class Agreement:
    vs_promoted_games: int
    agree: int
    disagree: int


@dataclass(frozen=True)
class LedgerRow:
    arm_id: str
    display_name: str
    status_badge: str
    track_record: TrackRecord | None
    evidence: tuple[EvidenceRef, ...]
    summary_sentence: str
    agreement: Agreement | None
    #: ``probability_positive`` recorded directly in the challenger's own
    #: evidence block (2026-08-24 dimension-3 fix): rows whose
    #: ``registry_source`` names no weak_signals key still carry a measured
    #: P+, and an interval rendered without its P+ was exactly the gap.
    own_probability_positive: float | None = None


@dataclass(frozen=True)
class ModelLedger:
    rows: tuple[LedgerRow, ...]
    active_model_id: str
    weak_signals_path: Path


def build_model_ledger(
    challengers_path: str | Path,
    weak_signals_path: str | Path,
    active_manifest_path: str | Path,
    per_game_frames: Mapping[str, Mapping[str, str]] | None = None,
) -> ModelLedger:
    """Build the ledger from the three live sources.

    ``per_game_frames``, when supplied, maps ``arm_id -> {game_id: pick}``;
    agreement-vs-promoted is populated only for rows whose arm (and the
    promoted arm) appear in it.
    """

    challengers_payload = _load_json(Path(challengers_path))
    registry_payload = _load_json(Path(weak_signals_path))
    manifest = _load_json(Path(active_manifest_path))

    signals: Mapping[str, Any] = registry_payload["signals"]
    model_id = str(manifest["model_id"])
    promoted_arm_id = f"promoted:{model_id}"

    rows = [_promoted_row(manifest, model_id)]
    for entry in challengers_payload["challengers"]:
        rows.append(_challenger_row(entry, signals, promoted_arm_id, per_game_frames))
    rows.sort(key=_sort_key)

    return ModelLedger(
        rows=tuple(rows),
        active_model_id=model_id,
        weak_signals_path=Path(weak_signals_path),
    )


def validate_ledger(ledger: ModelLedger) -> None:
    """Hard-fail on any contract violation, re-reading the weak-signals
    registry from disk the way :func:`nfl_ats.findings_registry.validate_curation`
    re-reads its registries at render time."""

    registry_payload = _load_json(ledger.weak_signals_path)
    signals: Mapping[str, Any] = registry_payload["signals"]

    promoted_rows = [row for row in ledger.rows if row.status_badge == STATUS_BADGE_PROMOTED]
    if len(promoted_rows) != 1:
        raise LedgerError(f"expected exactly one PROMOTED row, found {len(promoted_rows)}")
    expected_arm_id = f"promoted:{ledger.active_model_id}"
    if promoted_rows[0].arm_id != expected_arm_id:
        raise LedgerError(
            f"PROMOTED row has arm_id {promoted_rows[0].arm_id!r} but the "
            f"active manifest model_id {ledger.active_model_id!r} implies "
            f"{expected_arm_id!r}"
        )

    seen_arm_ids: set[str] = set()
    for row in ledger.rows:
        if row.arm_id in seen_arm_ids:
            raise LedgerError(f"duplicate arm_id {row.arm_id!r}")
        seen_arm_ids.add(row.arm_id)
        for ref in row.evidence:
            if ref.registry_key not in signals:
                raise LedgerError(
                    f"row {row.arm_id!r} cites registry key "
                    f"{ref.registry_key!r}, which does not exist in "
                    f"{ledger.weak_signals_path}"
                )
            live = fingerprint(signals[ref.registry_key])
            if live != ref.fingerprint:
                raise LedgerError(
                    f"row {row.arm_id!r} is stale against "
                    f"{ref.registry_key!r}: curated fingerprint "
                    f"{ref.fingerprint} does not match the live entry's "
                    f"{live}. Rebuild the ledger."
                )
        _audit_summary_numbers(row)


def render_markdown_table(ledger: ModelLedger) -> str:
    """Deterministic markdown rendering of the ledger for docs embedding."""

    header = (
        "| Arm | Badge | Grade | Games | Accuracy | Interval | Best P+ "
        "| Registry evidence | Agreement | Summary |"
    )
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, separator]
    for row in ledger.rows:
        lines.append(_render_row(row))
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise LedgerError(f"{path} does not contain a JSON object")
    return payload


def _badge_for_status(status: str) -> str:
    badge = _CHALLENGER_STATUS_BADGES.get(status)
    if badge is not None:
        return badge
    if status.startswith(_RETIRED_STATUS_PREFIXES):
        return STATUS_BADGE_RETIRED
    raise LedgerError(f"unknown challenger status {status!r} has no badge mapping")


def _promoted_row(manifest: Mapping[str, Any], model_id: str) -> LedgerRow:
    evaluation = manifest.get("historical_evaluation")
    track_record: TrackRecord | None = None
    if isinstance(evaluation, Mapping):
        intervals = evaluation.get("intervals")
        season_interval = intervals.get("season") if isinstance(intervals, Mapping) else None
        interval = _as_interval(season_interval)
        low = interval[0] if interval else None
        high = interval[1] if interval else None
        track_record = TrackRecord(
            games=_as_int(evaluation.get("games")),
            accuracy=_as_float(evaluation.get("accuracy")),
            interval_low=low,
            interval_high=high,
            # Season accuracy-proportion CI -- a rate, correctly percent-formatted.
            interval_unit="accuracy_rate",
            grade="close",
            artifact_ref=(
                str(evaluation["artifact"]) if evaluation.get("artifact") is not None else None
            ),
        )
    display_name = "Played card \u2014 model + fix-up rules"
    row = LedgerRow(
        arm_id=f"promoted:{model_id}",
        display_name=display_name,
        status_badge=STATUS_BADGE_PROMOTED,
        track_record=track_record,
        evidence=(),
        summary_sentence="",
        agreement=None,
    )
    return _with_summary(row)


def _challenger_row(
    entry: Mapping[str, Any],
    signals: Mapping[str, Any],
    promoted_arm_id: str,
    per_game_frames: Mapping[str, Mapping[str, str]] | None,
) -> LedgerRow:
    challenger_id = str(entry["challenger_id"])
    badge = _badge_for_status(str(entry["status"]))
    evidence_raw = entry.get("evidence")
    evidence_block: Mapping[str, Any] = evidence_raw if isinstance(evidence_raw, Mapping) else {}
    refs = _link_evidence(challenger_id, evidence_block, signals)
    row = LedgerRow(
        arm_id=challenger_id,
        display_name=CHALLENGER_DISPLAY_NAMES.get(challenger_id, challenger_id),
        status_badge=badge,
        track_record=_challenger_track_record(evidence_block),
        evidence=refs,
        summary_sentence="",
        agreement=(
            _agreement(challenger_id, promoted_arm_id, per_game_frames)
            if per_game_frames is not None
            else None
        ),
        own_probability_positive=_as_float(_first_value(evidence_block, _PROBABILITY_KEYS)),
    )
    return _with_summary(row)


def _link_evidence(
    challenger_id: str,
    evidence_block: Mapping[str, Any],
    signals: Mapping[str, Any],
) -> tuple[EvidenceRef, ...]:
    candidates = _registry_source_candidates(evidence_block)
    resolved = [key for key in candidates if key in signals]
    if not resolved and not candidates:
        resolved = [
            key
            for key in signals
            if key == challenger_id or key.endswith(_REGISTRY_FALLBACK_PREFIX + challenger_id)
        ]
    refs = []
    for key in resolved:
        payload = signals[key]
        refs.append(
            EvidenceRef(
                registry_key=key,
                effect=_as_float(payload.get("effect")),
                probability_positive=_as_float(payload.get("probability_positive")),
                classification=(
                    str(payload["classification"])
                    if payload.get("classification") is not None
                    else None
                ),
                fingerprint=fingerprint(payload),
            )
        )
    refs.sort(key=lambda ref: ref.registry_key)
    return tuple(refs)


def _registry_source_candidates(evidence_block: Mapping[str, Any]) -> list[str]:
    raw = evidence_block.get("registry_source")
    fragments: list[str]
    if isinstance(raw, str):
        fragments = raw.split(",")
    elif isinstance(raw, Sequence):
        fragments = []
        for item in raw:
            if isinstance(item, str):
                fragments.extend(item.split(","))
    else:
        fragments = []
    candidates: list[str] = []
    in_registry_list = False
    for fragment in fragments:
        marker_index = fragment.find(_REGISTRY_SOURCE_MARKER)
        if marker_index >= 0:
            in_registry_list = True
            after_marker = fragment[marker_index + len(_REGISTRY_SOURCE_MARKER) :]
            colon_index = after_marker.find(":")
            if colon_index < 0:
                continue
            key = after_marker[colon_index + 1 :].strip()
        elif in_registry_list:
            stripped = fragment.strip()
            if not stripped or stripped.startswith("("):
                in_registry_list = False
                continue
            key = stripped.split()[0]
        else:
            continue
        key = key.split()[0] if key.split() else ""
        if key and key not in candidates:
            candidates.append(key)
    return candidates


def _challenger_track_record(evidence_block: Mapping[str, Any]) -> TrackRecord | None:
    games = _first_number(evidence_block, _GAMES_KEYS)
    accuracy = _first_number(evidence_block, _ACCURACY_KEYS)
    interval = _as_pair(_first_value(evidence_block, _INTERVAL_KEYS))
    artifact_ref = _first_string(evidence_block, ("opener_window_artifact", "replication_artifact"))
    if games is None and accuracy is None and interval is None and artifact_ref is None:
        return None
    opener_hit = False
    if artifact_ref is not None and "opener" in artifact_ref.lower():
        opener_hit = True
    for key in evidence_block:
        if "opener" in key.lower():
            opener_hit = True
    return TrackRecord(
        games=int(games) if games is not None else None,
        accuracy=accuracy,
        interval_low=interval[0] if interval else None,
        interval_high=interval[1] if interval else None,
        # Every _INTERVAL_KEYS candidate is a *_points key -- an
        # accuracy-points effect delta, never a rate.
        interval_unit="accuracy_points",
        grade="opener" if opener_hit else "close",
        artifact_ref=artifact_ref,
    )


def _agreement(
    arm_id: str,
    promoted_arm_id: str,
    per_game_frames: Mapping[str, Mapping[str, str]] | None,
) -> Agreement | None:
    if per_game_frames is None or arm_id == promoted_arm_id:
        return None
    own = per_game_frames.get(arm_id)
    promoted = per_game_frames.get(promoted_arm_id)
    if own is None or promoted is None:
        return None
    shared = sorted(set(own) & set(promoted))
    agree = sum(1 for game_id in shared if own[game_id] == promoted[game_id])
    return Agreement(
        vs_promoted_games=len(shared),
        agree=agree,
        disagree=len(shared) - agree,
    )


def _row_probability(row: LedgerRow) -> float | None:
    """Best available confidence for a row: the strongest linked registry
    entry's ``probability_positive``, falling back to the challenger's own
    registered evidence block when no linked entry carries one."""

    best_ref = max(
        (ref.probability_positive for ref in row.evidence if ref.probability_positive is not None),
        default=None,
    )
    if best_ref is not None:
        return best_ref
    return row.own_probability_positive


def _sort_key(row: LedgerRow) -> tuple[int, float, str]:
    badge_rank = 0 if row.status_badge == STATUS_BADGE_PROMOTED else 1
    best_probability = _row_probability(row)
    return (
        badge_rank,
        -best_probability if best_probability is not None else float("inf"),
        row.arm_id,
    )


def _with_summary(row: LedgerRow) -> LedgerRow:
    parts: list[str] = []
    track = row.track_record
    # 2026-08-23 consolidation law (owner directive): the PROMOTED row is the
    # played card and must not re-quote its own track record -- the picks page
    # carries the one expectation number and the collapsed ladder carries the
    # history. Every other row's track record IS the row's data and stays.
    promoted = row.status_badge == STATUS_BADGE_PROMOTED
    if track is not None and not promoted:
        if track.accuracy is not None:
            sentence_grade = f"{track.grade}-grade"
            if track.games is not None:
                parts.append(
                    f"{sentence_grade} track record {track.accuracy:.1%} over {track.games} games"
                )
            else:
                parts.append(f"{sentence_grade} track record {track.accuracy:.1%}")
        if track.interval_low is not None and track.interval_high is not None:
            parts.append(f"interval [{track.interval_low:.3f}, {track.interval_high:.3f}]")
    if row.evidence:
        count_word = "entry" if len(row.evidence) == 1 else "entries"
        parts.append(f"{len(row.evidence)} registry evidence {count_word}")
        best = max(
            (
                ref.probability_positive
                for ref in row.evidence
                if ref.probability_positive is not None
            ),
            default=None,
        )
        if best is not None:
            parts.append(f"best evidence P+ {best:.3f}")
    elif row.own_probability_positive is not None:
        # No linked registry entry carries a P+, but the challenger's own
        # registration does -- quote it rather than leaving an interval bare.
        parts.append(f"registered evidence P+ {row.own_probability_positive:.3f}")
    if row.agreement is not None:
        parts.append(
            f"agreement vs promoted: {row.agreement.agree} agree, "
            f"{row.agreement.disagree} disagree over "
            f"{row.agreement.vs_promoted_games} shared games"
        )
    # Promoted card only (2026-08-23 consolidation revision): the summary no
    # longer re-quotes the selection-inflation arithmetic -- it names it and
    # points at the picks page's collapsed ladder, the one place the full
    # number set lives. See LEDGER_PROMOTED_CAVEAT in
    # nfl_ats.dashboard.findings_content.
    if row.status_badge == STATUS_BADGE_PROMOTED:
        # ``summary`` rejoins with ". " and appends its own full stop, so the
        # sentence constant ships without one.
        parts.append(LEDGER_PROMOTED_CAVEAT.removesuffix("."))
    summary = "" if not parts else ". ".join(parts) + "."
    return LedgerRow(
        arm_id=row.arm_id,
        display_name=row.display_name,
        status_badge=row.status_badge,
        track_record=row.track_record,
        evidence=row.evidence,
        summary_sentence=summary,
        agreement=row.agreement,
        own_probability_positive=row.own_probability_positive,
    )


def _audit_summary_numbers(row: LedgerRow) -> None:
    audited = row.summary_sentence.replace(row.display_name, "")
    allowed = _allowed_number_strings(row)
    for token in _NUMERIC_TOKEN.findall(audited):
        if token not in allowed:
            raise LedgerError(
                f"row {row.arm_id!r} summary quotes {token}, which no cited field produces"
            )


def _allowed_number_strings(row: LedgerRow) -> set[str]:
    values: list[float] = []
    track = row.track_record
    if track is not None:
        if track.games is not None:
            values.append(float(track.games))
        if track.accuracy is not None:
            values.append(track.accuracy)
        if track.interval_low is not None:
            values.append(track.interval_low)
        if track.interval_high is not None:
            values.append(track.interval_high)
    for ref in row.evidence:
        if ref.effect is not None:
            values.append(ref.effect)
        if ref.probability_positive is not None:
            values.append(ref.probability_positive)
    if row.own_probability_positive is not None:
        values.append(row.own_probability_positive)
    if row.agreement is not None:
        values.extend(
            [
                float(row.agreement.vs_promoted_games),
                float(row.agreement.agree),
                float(row.agreement.disagree),
            ]
        )
    if row.status_badge == STATUS_BADGE_PROMOTED:
        # The promoted summary's caveat quotes exactly one numeral: the
        # pinned played-card expectation percentage ("≈55%", rendered via
        # PLAYED_CARD_EXPECTATION_HERO inside LEDGER_PROMOTED_CAVEAT).
        values.append(float(PLAYED_CARD_EXPECTATION_PERCENT))
    allowed: set[str] = {str(len(row.evidence))}
    for value in values:
        allowed.update(
            {
                repr(value),
                f"{value:.1f}",
                f"{value:.2f}",
                f"{value:.3f}",
                f"{value * 100:.1f}",
                f"{value * 100:.3f}",
            }
        )
        if float(value).is_integer():
            allowed.add(str(int(value)))
    return allowed


def _render_row(row: LedgerRow) -> str:
    track = row.track_record
    games_cell = str(track.games) if track is not None and track.games is not None else "-"
    accuracy_cell = (
        f"{track.accuracy:.1%}" if track is not None and track.accuracy is not None else "-"
    )
    interval_cell = _interval_cell_text(row)
    grade_cell = track.grade if track is not None else "-"
    best_probability = max(
        (ref.probability_positive for ref in row.evidence if ref.probability_positive is not None),
        default=None,
    )
    probability_cell = "-" if best_probability is None else f"{best_probability:.3f}"
    evidence_cell = ", ".join(ref.registry_key for ref in row.evidence) if row.evidence else "-"
    if row.agreement is not None:
        agreement_cell = f"{row.agreement.agree}/{row.agreement.vs_promoted_games} agree"
    else:
        agreement_cell = "-"
    cells = [
        row.display_name,
        row.status_badge,
        grade_cell,
        games_cell,
        accuracy_cell,
        interval_cell,
        probability_cell,
        evidence_cell,
        agreement_cell,
        row.summary_sentence,
    ]
    escaped = (cell.replace("|", "\\|") for cell in cells)
    return "| " + " | ".join(escaped) + " |"


def _first_value(evidence_block: Mapping[str, Any], keys: Sequence[str]) -> Any:
    direct = _scan_mapping(evidence_block, keys)
    if direct is not None:
        return direct
    for value in evidence_block.values():
        if isinstance(value, Mapping):
            nested = _scan_mapping(value, keys)
            if nested is not None:
                return nested
    return None


def _first_number(evidence_block: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    return _as_float(_first_value(evidence_block, keys))


def _first_string(evidence_block: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = evidence_block.get(key)
        if isinstance(value, str):
            return value
    return None


def _scan_mapping(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, (list, tuple)):
            return value
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return int(number) if number is not None else None


def _as_interval(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        return _as_pair([value.get("lower"), value.get("upper")])
    return _as_pair(value)


def _as_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low, high = _as_float(value[0]), _as_float(value[1])
    if low is None or high is None:
        return None
    return low, high


_BADGE_GLYPHS = {
    STATUS_BADGE_PROMOTED: "\u2713",
    STATUS_BADGE_CHALLENGER: "\u25b2",
    STATUS_BADGE_SUPERSEDED: "\u2014",
    STATUS_BADGE_RETIRED: "\u2014",
}

_BADGE_CSS_KINDS = {
    STATUS_BADGE_PROMOTED: "promoted",
    STATUS_BADGE_CHALLENGER: "challenger",
    STATUS_BADGE_SUPERSEDED: "muted",
    STATUS_BADGE_RETIRED: "muted",
}

_LEDGER_COLUMNS = (
    "Configuration",
    "Track record",
    "Best evidence",
    "Interval (accuracy pts)",
    "Evidence",
    "Agreement with promoted",
)

_DASH = "\u2014"


def render_ledger_html(ledger: ModelLedger, *, css_mode: str = "classes") -> str:
    """Deterministic static HTML fragment for the Model Ledger tabular view.

    One row per arm in ledger (promoted-first) order: a glyph+text status
    badge that never relies on color alone, the display name and summary,
    the track record, best-evidence P+, interval, footnote-linked evidence
    count, and agreement-vs-promoted ("--" when unpopulated). Plain headers:
    ordering is fixed and stated once in the caption, never decorated. All
    text is escaped and every numeral is rendered from a ledger field. The
    fragment reuses the design-system classes (``table.data``, ``badge-*``,
    ``fine``, ``num``) and ships no scripts, inline handlers, or external
    references.
    """

    if css_mode != "classes":
        raise LedgerError(f"unsupported css_mode {css_mode!r}; only 'classes' is supported")
    marker_counter = 0
    rows_html: list[str] = []
    evidence_html: list[str] = []
    for row in ledger.rows:
        markers: list[tuple[int, str]] = []
        for _ in row.evidence:
            marker_counter += 1
            markers.append((marker_counter, f"ledger-ev-{marker_counter}"))
        row_class = ' class="row-promoted"' if row.status_badge == STATUS_BADGE_PROMOTED else ""
        name_html = escape(row.display_name)
        if row.status_badge == STATUS_BADGE_PROMOTED:
            name_html = f'<span title="{escape(row.arm_id)}">{name_html}</span>'
        summary_html = (
            f'<span class="fine">{escape(row.summary_sentence)}</span>'
            if row.summary_sentence
            else ""
        )
        cells = [
            "<br>".join(
                [
                    _badge_html(row.status_badge),
                    name_html,
                    summary_html,
                ]
            ),
            _track_record_cell(row.track_record),
            _best_evidence_cell(row),
            _interval_cell(row),
            _evidence_cell(row.evidence, markers),
            (
                (
                    f"{row.agreement.agree}/{row.agreement.disagree} of "
                    f"{row.agreement.vs_promoted_games}"
                )
                if row.agreement is not None
                else _DASH
            ),
        ]
        cells_html = "".join(f"<td>{cell}</td>" for cell in cells)
        rows_html.append(f"<tr{row_class}>{cells_html}</tr>")
        for ref, (_, anchor_id) in zip(row.evidence, markers, strict=True):
            evidence_html.append(_evidence_entry_html(anchor_id, ref))
    head_cells = "".join(f"<th>{escape(label)}</th>" for label in _LEDGER_COLUMNS)
    caption = (
        "Ordering is fixed: the promoted card first, then challengers by "
        "best-evidence P+ descending."
    )
    table = (
        '<table class="data ledger">'
        f'<caption class="fine">{caption}</caption>'
        f"<thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )
    evidence_section = ""
    if evidence_html:
        evidence_section = (
            '<section class="ledger-evidence"><p class="kicker">Registry evidence</p>'
            f"<ol>{''.join(evidence_html)}</ol></section>"
        )
    return f'<div class="ledger-view">{table}{evidence_section}</div>'


def build_and_render(
    challengers_path: str | Path,
    weak_signals_path: str | Path,
    active_manifest_path: str | Path,
) -> str:
    """Build the ledger from live artifacts, validate it, render HTML."""

    ledger = build_model_ledger(challengers_path, weak_signals_path, active_manifest_path)
    validate_ledger(ledger)
    return render_ledger_html(ledger)


def _badge_html(badge: str) -> str:
    glyph = _BADGE_GLYPHS[badge]
    kind = _BADGE_CSS_KINDS[badge]
    return (
        f'<span class="badge badge-{kind}"><span class="badge-glyph">{glyph}</span>'
        f"{escape(badge)}</span>"
    )


def _track_record_cell(track: TrackRecord | None) -> str:
    if track is None:
        return _DASH
    parts: list[str] = []
    if track.accuracy is not None and track.games is not None:
        parts.append(f"{track.accuracy:.1%} over {track.games:,} games")
    elif track.games is not None:
        parts.append(f"{track.games:,} games")
    elif track.accuracy is not None:
        parts.append(f"{track.accuracy:.1%}")
    if not parts:
        return _DASH
    parts.append(f'<span class="fine">{escape(track.grade)}-grade</span>')
    return "<br>".join(parts)


def _best_evidence_cell(row: LedgerRow) -> str:
    best_probability = max(
        (ref.probability_positive for ref in row.evidence if ref.probability_positive is not None),
        default=None,
    )
    if best_probability is None:
        return _DASH
    count_word = "entry" if len(row.evidence) == 1 else "entries"
    return f"P+ {p_plus_text(best_probability)} over n={len(row.evidence)} {count_word}"


def _interval_cell_text(row: LedgerRow) -> str:
    """Plain-text interval cell for the markdown rendering."""

    track = row.track_record
    if track is None or track.interval_low is None or track.interval_high is None:
        return "-"
    cell = f"[{track.interval_low:.3f}, {track.interval_high:.3f}]"
    if row.status_badge == STATUS_BADGE_PROMOTED:
        # The promoted row's interval is a season accuracy-proportion CI, not
        # an accuracy-points effect interval -- no P+ applies to it.
        return cell
    probability = _row_probability(row)
    shown = _DASH if probability is None else p_plus_text(probability)
    return f"{cell} \u00b7 P+ {shown}"


def _interval_cell(row: LedgerRow) -> str:
    """Interval plus, beside it, the row's best available P+ (2026-08-24
    dimension-3 fix: three ledger rows rendered intervals with no P+ at all).
    When no measured P+ exists the cell says so explicitly instead of hiding
    the gap. The promoted row's interval is a season accuracy-proportion CI,
    not an accuracy-points effect interval, so it carries no P+."""

    track = row.track_record
    if track is None or track.interval_low is None or track.interval_high is None:
        return _DASH
    cell = f"[{track.interval_low:.3f}, {track.interval_high:.3f}]"
    if row.status_badge == STATUS_BADGE_PROMOTED:
        return cell
    probability = _row_probability(row)
    shown = _DASH if probability is None else p_plus_text(probability)
    return f'{cell} \u00b7 <span class="fine">P+ {shown}</span>'


def _evidence_cell(evidence: tuple[EvidenceRef, ...], markers: list[tuple[int, str]]) -> str:
    if not evidence:
        return _DASH
    count_word = "entry" if len(evidence) == 1 else "entries"
    links = "".join(
        f' <a class="fn-ref" href="#{anchor_id}">[{number}]</a>' for number, anchor_id in markers
    )
    return f"{len(evidence)} {count_word}{links}"


def _evidence_entry_html(anchor_id: str, ref: EvidenceRef) -> str:
    effect = _DASH if ref.effect is None else f"{ref.effect:+.3f}"
    probability = (
        _DASH if ref.probability_positive is None else p_plus_text(ref.probability_positive)
    )
    classification = _DASH if ref.classification is None else escape(ref.classification)
    return (
        f'<li id="{anchor_id}"><code>{escape(ref.registry_key)}</code> '
        f"effect {effect} accuracy pts \u00b7 P+ {probability} \u00b7 {classification}</li>"
    )
