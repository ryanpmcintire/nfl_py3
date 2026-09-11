from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from nfl_ats import rotation, weak_signals

STORE_WEAK_SIGNAL = "weak_signal"
STORE_ROTATION = "rotation"
STORE_CHALLENGER = "challenger"
STORES = (STORE_WEAK_SIGNAL, STORE_ROTATION, STORE_CHALLENGER)


class CurationError(ValueError):
    pass


def fingerprint(payload: Mapping[str, Any]) -> str:

    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RegistryEntry:
    key: str
    store: str
    name: str
    description: str
    classification: str | None
    effect: float | None
    effect_units: str | None
    probability_positive: float | None
    interval: tuple[float, float] | None
    seasons: tuple[int, int] | None
    league: str | None
    recorded_at: str | None
    fingerprint: str


def _weak_signal_entry(signal: weak_signals.WeakSignal) -> RegistryEntry:
    payload: dict[str, Any] = {
        "recorded_at": signal.recorded_at,
        "description": signal.description,
        "source": signal.source,
        "effect": signal.effect,
        "effect_units": signal.effect_units,
        "classification": signal.classification,
        "classification_evidence": signal.classification_evidence,
        "closing_ground": signal.closing_ground,
        "league": signal.league,
        "seasons": list(signal.seasons),
        "standard_error": signal.standard_error,
        "interval": None if signal.interval is None else list(signal.interval),
        "probability_positive": signal.probability_positive,
        "sample_games": signal.sample_games,
        "sample_blocks": signal.sample_blocks,
        "reliability": signal.reliability,
        "notes": signal.notes,
    }
    return RegistryEntry(
        key=f"{STORE_WEAK_SIGNAL}:{signal.name}",
        store=STORE_WEAK_SIGNAL,
        name=signal.name,
        description=signal.description,
        classification=signal.classification,
        effect=signal.effect,
        effect_units=signal.effect_units,
        probability_positive=signal.probability_positive,
        interval=signal.interval,
        seasons=signal.seasons,
        league=signal.league,
        recorded_at=signal.recorded_at,
        fingerprint=fingerprint(payload),
    )


def _rotation_window_payload(window: rotation.Window) -> dict[str, Any]:
    return {
        "seasons": list(window.seasons),
        "state": window.state,
        "assigned_at": window.assigned_at,
        "spent_at": window.spent_at,
        "artifact": window.artifact,
        "verdict": window.verdict,
        "closing_ground": window.closing_ground,
        "probability_positive": window.probability_positive,
        "effect": window.effect,
        "effect_units": window.effect_units,
        "interval": None if window.interval is None else list(window.interval),
        "standard_error": window.standard_error,
        "sample_blocks": window.sample_blocks,
        "notes": window.notes,
    }


def _rotation_entry(family: rotation.Family) -> RegistryEntry:
    window = family.windows[-1] if family.windows else None
    payload: dict[str, Any] = {
        "status": family.status,
        "description": family.description,
        "grade": family.grade,
        "declared_at": family.declared_at,
        "acknowledges_mined_2018_2025": family.acknowledges_mined_2018_2025,
        "window": _rotation_window_payload(window) if window is not None else None,
    }
    return RegistryEntry(
        key=f"{STORE_ROTATION}:{family.name}",
        store=STORE_ROTATION,
        name=family.name,
        description=family.description,
        classification=(window.verdict if window is not None else family.status),
        effect=window.effect if window is not None else None,
        effect_units=window.effect_units if window is not None else None,
        probability_positive=(window.probability_positive if window is not None else None),
        interval=window.interval if window is not None else None,
        seasons=window.seasons if window is not None else None,
        league=None,
        recorded_at=(
            (window.spent_at or window.assigned_at) if window is not None else family.declared_at
        ),
        fingerprint=fingerprint(payload),
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low, high = _as_float(value[0]), _as_float(value[1])
    return None if low is None or high is None else (low, high)


def _as_seasons(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _challenger_entry(entry: Mapping[str, Any]) -> RegistryEntry:
    challenger_id = str(entry.get("challenger_id", "unknown"))
    evidence_raw = entry.get("evidence")
    evidence = evidence_raw if isinstance(evidence_raw, dict) else {}
    return RegistryEntry(
        key=f"{STORE_CHALLENGER}:{challenger_id}",
        store=STORE_CHALLENGER,
        name=challenger_id,
        description=str(entry.get("status_reason", "")),
        classification=str(entry.get("status", "unknown")),
        effect=_as_float(evidence.get("paired_delta_points")),
        effect_units=("accuracy_points" if "paired_delta_points" in evidence else None),
        probability_positive=_as_float(evidence.get("probability_positive")),
        interval=_as_interval(evidence.get("week_blocked_interval_points")),
        seasons=_as_seasons(evidence.get("opener_window")),
        league="nfl",
        recorded_at=str(entry.get("registered_at_utc", "")),
        fingerprint=fingerprint(dict(entry)),
    )


def load_weak_signal_registry(registry_root: Path | None = None) -> weak_signals.Registry:

    path = (
        registry_root / weak_signals.WEAK_SIGNAL_REGISTRY_FILENAME
        if registry_root is not None
        else weak_signals.default_registry_path()
    )
    return weak_signals.load_registry(path, on_unknown_field="warn")


def load_rotation_registry(registry_root: Path | None = None) -> rotation.Registry:

    path = (
        registry_root / rotation.ROTATION_REGISTRY_FILENAME
        if registry_root is not None
        else rotation.default_registry_path()
    )
    if not path.is_file():
        return rotation.Registry(version=rotation.ROTATION_REGISTRY_VERSION, notes=(), families={})
    return rotation.load_registry(path, on_unknown_field="warn")


def load_all_entries(
    *,
    registry_root: Path | None = None,
    weak_signal_registry: weak_signals.Registry | None = None,
    rotation_registry: rotation.Registry | None = None,
    challengers: Sequence[Mapping[str, Any]] = (),
) -> dict[str, RegistryEntry]:

    if weak_signal_registry is None:
        weak_signal_registry = load_weak_signal_registry(registry_root)
    if rotation_registry is None:
        rotation_registry = load_rotation_registry(registry_root)

    entries: dict[str, RegistryEntry] = {}
    for signal in weak_signal_registry.signals.values():
        entry = _weak_signal_entry(signal)
        entries[entry.key] = entry
    for family in rotation_registry.families.values():
        entry = _rotation_entry(family)
        entries[entry.key] = entry
    for challenger in challengers:
        entry = _challenger_entry(challenger)
        entries[entry.key] = entry
    return entries


def validate_curation(findings: Sequence[Any], entries: Mapping[str, RegistryEntry]) -> None:

    for finding in findings:
        question = finding.question
        if finding.evergreen:
            if finding.registry_keys:
                raise CurationError(
                    f"finding {question!r} is marked evergreen but names "
                    f"registry_keys {finding.registry_keys!r}; an evergreen "
                    "finding must cite none (it is not tracking a live number)"
                )
            continue
        if not finding.registry_keys:
            raise CurationError(
                f"finding {question!r} names no registry_keys and is not "
                "marked evergreen -- every curated claim must trace to a "
                "registry entry or be declared evergreen"
            )
        if len(finding.registry_keys) != len(finding.registry_fingerprints):
            raise CurationError(
                f"finding {question!r} has {len(finding.registry_keys)} "
                f"registry_keys but {len(finding.registry_fingerprints)} "
                "registry_fingerprints -- they must be parallel tuples"
            )
        for key, expected in zip(finding.registry_keys, finding.registry_fingerprints, strict=True):
            entry = entries.get(key)
            if entry is None:
                raise CurationError(
                    f"finding {question!r} names registry key {key!r}, which "
                    "does not exist in any evidence store (weak_signals.json, "
                    "rotation_registry.json, or challengers.json). Fix the "
                    "key, or remove it if the entry was retired."
                )
            if entry.fingerprint != expected:
                raise CurationError(
                    f"finding {question!r} is stale against {key!r}: curated "
                    f"fingerprint {expected} does not match the live entry's "
                    f"{entry.fingerprint}. Re-read {key!r} "
                    f"(classification={entry.classification!r}, "
                    f"effect={entry.effect!r}, P+={entry.probability_positive!r}), "
                    "correct the prose if the story changed, and update "
                    "curated_as_of + registry_fingerprints to the new value."
                )


@dataclass(frozen=True)
class WatchingLead:
    key: str
    name: str
    description: str
    plain_summary: str | None
    effect: float
    effect_units: str
    interval: tuple[float, float] | None
    probability_positive: float
    seasons: tuple[int, int]
    league: str


_OPENER_SUFFIX = re.compile(r"_opener$")
_BATTERY_MARKERS = ("battery", "microstructure")


def _construct_key(name: str) -> str:
    return _OPENER_SUFFIX.sub("", name)


def _battery_key(name: str) -> str:
    tokens = name.split("_")
    for index, token in enumerate(tokens[:3]):
        if token in _BATTERY_MARKERS:
            return "_".join(tokens[: index + 1])
    return name


def _extremity(signal: weak_signals.WeakSignal) -> float:
    return abs((signal.probability_positive or 0.5) - 0.5)


def top_open_leads(
    registry: weak_signals.Registry,
    *,
    limit: int = 12,
    leagues: tuple[str, ...] = ("nfl", "cfb"),
) -> list[WatchingLead]:

    candidates = [
        signal
        for signal in registry.signals.values()
        if signal.classification == weak_signals.POOLABLE_CLASSIFICATION
        and signal.status != "invalidated"
        and signal.league in leagues
        and signal.probability_positive is not None
        and "oracle" not in signal.description.lower()
        and signal.effect_units != "correlation"
        and signal.category != "control"
    ]

    by_construct: dict[str, weak_signals.WeakSignal] = {}
    for signal in sorted(candidates, key=lambda s: s.name):
        construct = _construct_key(signal.name)
        current = by_construct.get(construct)
        prefer_signal = current is None or (
            signal.name.endswith("_opener") and not current.name.endswith("_opener")
        )
        if prefer_signal:
            by_construct[construct] = signal

    by_family: dict[str, weak_signals.WeakSignal] = {}
    for signal in sorted(by_construct.values(), key=lambda s: s.name):
        family = _battery_key(signal.name)
        current = by_family.get(family)
        if current is None or _extremity(signal) > _extremity(current):
            by_family[family] = signal

    ranked = sorted(by_family.values(), key=_extremity, reverse=True)
    return [
        WatchingLead(
            key=f"{STORE_WEAK_SIGNAL}:{signal.name}",
            name=signal.name,
            description=signal.description,
            plain_summary=signal.plain_summary,
            effect=signal.effect,
            effect_units=signal.effect_units,
            interval=signal.interval,
            probability_positive=float(signal.probability_positive),
            seasons=signal.seasons,
            league=signal.league,
        )
        for signal in ranked[:limit]
        if signal.probability_positive is not None
    ]


@dataclass(frozen=True)
class RecentActivityEntry:
    key: str
    store: str
    category: str
    plain_summary: str | None
    effect: float | None
    effect_units: str | None
    probability_positive: float | None
    direction_sentence: str | None
    closed: bool
    closed_label: str | None
    recorded_at: str
    is_instrument_control: bool = False


@dataclass(frozen=True)
class RecentRegistryActivity:
    window_days: int
    screened_count: int
    resolved_count: int
    entries_by_category: tuple[tuple[str, tuple[RecentActivityEntry, ...]], ...]

    @property
    def is_empty(self) -> bool:
        return self.screened_count == 0


CLOSED_ACTIVITY_BADGE_TEXT = "Resolved the other way"


def _parse_registry_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _activity_direction_sentence(probability_positive: float) -> str:

    if probability_positive >= 0.5:
        return (
            f"Leans FOR the pattern described -- {probability_positive:.0%} "
            "confidence in that direction (not yet resolved; see the interval)."
        )
    against = 1.0 - probability_positive
    return (
        "Leans AGAINST the pattern described -- read this as a lead for the "
        f"OTHER side, {against:.0%} confidence in that direction (not yet "
        "resolved; see the interval)."
    )


def _is_activity_candidate(signal: weak_signals.WeakSignal) -> bool:

    return (
        signal.status != "invalidated"
        and signal.effect_units != "correlation"
        and signal.category != "control"
        and "oracle" not in signal.description.lower()
    )


_INSTRUMENT_CONTROL_PHRASES: tuple[str, ...] = (
    "positive control",
    "perfect foresight",
    "perfect-foresight",
    "oracle",
    "cheat rule",
)
_INSTRUMENT_CONTROL_NAME_RE = re.compile(r"(^|_)(pc|control)(_|$)")


def _is_instrument_control_text(name: str, description: str | None) -> bool:

    text = (description or "").lower()
    if any(phrase in text for phrase in _INSTRUMENT_CONTROL_PHRASES):
        return True
    return bool(_INSTRUMENT_CONTROL_NAME_RE.search(name.lower()))


def recent_registry_activity(
    registry: weak_signals.Registry,
    rotation_registry: rotation.Registry,
    as_of: datetime | date,
    *,
    days: int = 7,
) -> RecentRegistryActivity:

    as_of_date = as_of.date() if isinstance(as_of, datetime) else as_of
    window_start = as_of_date - timedelta(days=days)

    entries: list[RecentActivityEntry] = []
    resolved_count = 0

    for signal in registry.signals.values():
        if not _is_activity_candidate(signal):
            continue
        recorded = _parse_registry_date(signal.recorded_at)
        if recorded is None or not (window_start <= recorded <= as_of_date):
            continue
        closed = signal.classification in weak_signals.TERMINAL_CLASSIFICATIONS
        if closed:
            resolved_count += 1
        entries.append(
            RecentActivityEntry(
                key=f"{STORE_WEAK_SIGNAL}:{signal.name}",
                store=STORE_WEAK_SIGNAL,
                category=signal.category or STORE_WEAK_SIGNAL,
                plain_summary=signal.plain_summary,
                effect=signal.effect,
                effect_units=signal.effect_units,
                probability_positive=signal.probability_positive,
                direction_sentence=(
                    _activity_direction_sentence(signal.probability_positive)
                    if signal.probability_positive is not None
                    else None
                ),
                closed=closed,
                closed_label=CLOSED_ACTIVITY_BADGE_TEXT if closed else None,
                recorded_at=signal.recorded_at,
                is_instrument_control=_is_instrument_control_text(signal.name, signal.description),
            )
        )

    for family in rotation_registry.families.values():
        for window in family.windows:
            stamp = window.spent_at or window.assigned_at
            recorded = _parse_registry_date(stamp)
            if recorded is None or not (window_start <= recorded <= as_of_date):
                continue
            closed = window.verdict == "closed_negative"
            if closed:
                resolved_count += 1
            entries.append(
                RecentActivityEntry(
                    key=f"{STORE_ROTATION}:{family.name}",
                    store=STORE_ROTATION,
                    category=STORE_ROTATION,
                    plain_summary=family.plain_summary,
                    effect=window.effect,
                    effect_units=window.effect_units,
                    probability_positive=window.probability_positive,
                    direction_sentence=(
                        _activity_direction_sentence(window.probability_positive)
                        if window.probability_positive is not None
                        else None
                    ),
                    closed=closed,
                    closed_label=CLOSED_ACTIVITY_BADGE_TEXT if closed else None,
                    recorded_at=str(stamp),
                    is_instrument_control=_is_instrument_control_text(
                        family.name, family.description
                    ),
                )
            )

    grouped: dict[str, list[RecentActivityEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.category, []).append(entry)

    def _extremity(entry: RecentActivityEntry) -> float:
        return abs((entry.probability_positive or 0.5) - 0.5)

    entries_by_category = tuple(
        (category, tuple(sorted(rows, key=_extremity, reverse=True)))
        for category, rows in sorted(grouped.items())
    )

    return RecentRegistryActivity(
        window_days=days,
        screened_count=len(entries),
        resolved_count=resolved_count,
        entries_by_category=entries_by_category,
    )


__all__ = [
    "CLOSED_ACTIVITY_BADGE_TEXT",
    "STORES",
    "STORE_CHALLENGER",
    "STORE_ROTATION",
    "STORE_WEAK_SIGNAL",
    "CurationError",
    "RecentActivityEntry",
    "RecentRegistryActivity",
    "RegistryEntry",
    "WatchingLead",
    "fingerprint",
    "load_all_entries",
    "load_rotation_registry",
    "load_weak_signal_registry",
    "recent_registry_activity",
    "top_open_leads",
    "validate_curation",
]
