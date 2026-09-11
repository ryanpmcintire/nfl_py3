from __future__ import annotations

import bisect
import dataclasses
import json
import math
import os
import re
import statistics
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from nfl_ats.evidence_conventions import binomial_two_sided_p
from nfl_ats.io import atomic_json

WEAK_SIGNAL_REGISTRY_VERSION = 1
WEAK_SIGNAL_REGISTRY_FILENAME = "weak_signals.json"

CLASSIFICATIONS = (
    "unresolved_below_power",
    "refuted_mechanism",
    "bounded_by_control",
)
POOLABLE_CLASSIFICATION = "unresolved_below_power"
TERMINAL_CLASSIFICATIONS = ("refuted_mechanism", "bounded_by_control")

CLOSING_GROUNDS: dict[str, tuple[str, ...]] = {
    "refuted_mechanism": ("wrong_sign_resolved", "no_split_half_reliability"),
    "bounded_by_control": ("positive_control_bound",),
}

NO_SPLIT_HALF_RELIABILITY_MAX = 0.10

_FAMILY_DECOMPOSITION_SUFFIXES = (
    re.compile(r"_opener$"),
    re.compile(r"_era_\d{4}_\d{4}$"),
    re.compile(r"_era_\d{4}$"),
    re.compile(r"_\d{4}_\d{4}$"),
    re.compile(r"_(?:pre|post)\d{4}$"),
)

FAMILY_BATTERY_MARKERS = ("battery", "microstructure")

_CLOSING_RULE = (
    "AGENTS.md binding rule: an interval containing zero is NOT grounds for "
    "rejection. Only a resolved wrong sign, zero split-half reliability, or a "
    "positive control proven able to detect an effect that size may close a "
    "line of work; everything else is 'unresolved_below_power' and is recorded, "
    "not closed."
)

EFFECT_UNITS = (
    "ats_points",
    "accuracy_points",
    "brier",
    "log_loss",
    "mae",
    "correlation",
    "mae_improvement",
    "brier_improvement",
    "log_loss_improvement",
    "payout_first_pp",
)

CATEGORIES = (
    "market",
    "onfield",
    "health",
    "schedule",
    "environment",
    "attention",
    "offfield",
    "modeling",
    "control",
)

LEAGUES = ("nfl", "cfb")

_TOP_LEVEL_FIELDS = frozenset({"version", "notes", "signals"})
_SIGNAL_FIELDS = frozenset(
    {
        "recorded_at",
        "description",
        "source",
        "effect",
        "effect_units",
        "standard_error",
        "interval",
        "probability_positive",
        "sample_games",
        "sample_blocks",
        "league",
        "seasons",
        "classification",
        "classification_evidence",
        "closing_ground",
        "reliability",
        "family",
        "notes",
        "plain_summary",
        "category",
        "status",
        "invalidated_reason",
        "superseded_by",
        "corrections",
    }
)

_CORRECTION_FIELDS = frozenset({"at", "field", "from", "to", "reason"})

_CORRECTABLE_FIELDS = frozenset({"probability_positive"})


class WeakSignalError(ValueError):
    pass


class UnknownRegistryFieldWarning(UserWarning):
    pass


OnUnknownField = Literal["raise", "warn"]


def _handle_unknown_fields(message: str, *, on_unknown_field: OnUnknownField) -> None:
    if on_unknown_field == "raise":
        raise WeakSignalError(message)
    warnings.warn(
        f"{message} -- ignored so this cannot abort a site build; add the field "
        "to the matching allowlist in weak_signals.py to stop seeing this warning",
        UnknownRegistryFieldWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class WeakSignal:
    name: str
    recorded_at: str
    description: str
    source: str
    effect: float
    effect_units: str
    classification: str
    league: str
    seasons: tuple[int, int]
    standard_error: float | None = None
    interval: tuple[float, float] | None = None
    probability_positive: float | None = None
    sample_games: int | None = None
    sample_blocks: int | None = None
    classification_evidence: str = ""
    closing_ground: str | None = None
    reliability: float | None = None
    family: str | None = None
    notes: str = ""
    plain_summary: str | None = None
    category: str | None = None
    status: str = "active"
    invalidated_reason: str | None = None
    superseded_by: str | None = None

    @property
    def favours_candidate(self) -> bool:

        return self.effect > 0.0

    @property
    def favours_baseline(self) -> bool:

        return self.effect < 0.0

    @property
    def direction(self) -> int:

        if self.effect > 0.0:
            return 1
        if self.effect < 0.0:
            return -1
        return 0

    @property
    def season_range(self) -> range:
        return range(self.seasons[0], self.seasons[1] + 1)

    def resolved_standard_error(self) -> float | None:

        if self.standard_error is not None:
            return self.standard_error
        if self.interval is not None:
            low, high = self.interval
            width = float(high) - float(low)
            if width > 0.0:
                return width / (2.0 * 1.959963984540054)
        return None


@dataclass(frozen=True)
class Registry:
    version: int
    notes: tuple[str, ...]
    signals: dict[str, WeakSignal]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WeakSignalError(message)


def validate_closure(
    name: str,
    *,
    classification: str,
    closing_ground: str | None,
    classification_evidence: str,
    interval: tuple[float, float] | None,
    reliability: float | None,
    probability_positive: float | None = None,
) -> None:

    if classification in TERMINAL_CLASSIFICATIONS:
        admissible = CLOSING_GROUNDS[classification]
        _require(
            closing_ground in admissible,
            f"Signal {name!r} is {classification!r} but names no admissible "
            f"closing_ground (expected one of {', '.join(admissible)}). {_CLOSING_RULE}",
        )
        _require(
            bool(classification_evidence.strip()),
            f"Signal {name!r} is terminal but classification_evidence is empty; "
            f"a closure must cite its evidence. {_CLOSING_RULE}",
        )
        if closing_ground == "wrong_sign_resolved":
            _require(
                interval is not None and interval[1] < 0.0,
                f"Signal {name!r} claims a RESOLVED wrong sign but its interval "
                f"{None if interval is None else list(interval)} is not entirely "
                f"below zero — the wrong sign is a lean, not a resolution. "
                f"{_CLOSING_RULE}",
            )
        if closing_ground == "no_split_half_reliability":
            _require(
                reliability is not None,
                f"Signal {name!r} claims no split-half reliability but records "
                f"no reliability measurement to cite. {_CLOSING_RULE}",
            )
            assert reliability is not None
            _require(
                reliability <= NO_SPLIT_HALF_RELIABILITY_MAX,
                f"Signal {name!r} claims no split-half reliability but records "
                f"reliability {reliability:.3f}, which is above the "
                f"{NO_SPLIT_HALF_RELIABILITY_MAX:.2f} ceiling this ground admits "
                f"-- a trait this reliable is NOT refuted by its sample size. "
                f"{_CLOSING_RULE}",
            )
        if closing_ground == "positive_control_bound":
            _require(
                interval is not None or probability_positive is not None,
                f"Signal {name!r} claims a positive-control bound but records no "
                "quantitative evidence (neither an interval nor a "
                "probability_positive); a control bound IS a measurement and "
                f"must be citable. {_CLOSING_RULE}",
            )
    else:
        _require(
            closing_ground is None,
            f"Signal {name!r} is {classification!r}, which is not closed and "
            "cannot carry a closing_ground",
        )


def validate_coherence(
    name: str,
    *,
    effect: float,
    interval: tuple[float, float] | None,
) -> None:

    if interval is None:
        return
    low, high = interval
    _require(
        low <= effect <= high,
        f"Signal {name!r} records effect {effect} outside its own interval "
        f"[{low}, {high}] -- a recording contradiction; check the sign or the "
        "interval against the source artifact before recording",
    )


def _validate_signal_numeric_fields(
    name: str,
    *,
    effect: float,
    standard_error: float | None,
    interval: tuple[float, float] | None,
    probability_positive: float | None,
) -> None:

    _require(math.isfinite(effect), f"Signal {name!r} has a non-finite effect")
    if standard_error is not None:
        _require(standard_error > 0.0, f"Signal {name!r} has a non-positive standard_error")
    if interval is not None:
        low, high = interval
        _require(low <= high, f"Signal {name!r} has an inverted interval")
    if probability_positive is not None:
        _require(
            0.0 <= probability_positive <= 1.0,
            f"Signal {name!r} has probability_positive outside [0, 1]",
        )


def coherence_problems(signals: Sequence[WeakSignal]) -> list[dict[str, Any]]:

    problems: list[dict[str, Any]] = []
    for signal in sorted(signals, key=lambda s: s.name):
        if signal.status == "invalidated":
            continue
        if signal.interval is None:
            continue
        low, high = signal.interval
        if not (low <= signal.effect <= high):
            problems.append(
                {
                    "signal": signal.name,
                    "problem": "effect_outside_interval",
                    "effect": signal.effect,
                    "interval": [low, high],
                }
            )
    return problems


def _validate_corrections(
    name: str, corrections: Any, *, on_unknown_field: OnUnknownField = "raise"
) -> None:

    if corrections is None:
        return
    _require(
        isinstance(corrections, list),
        f"Signal {name!r}: corrections must be a list, got {type(corrections).__name__}",
    )
    for index, entry in enumerate(corrections):
        where = f"Signal {name!r} correction {index}"
        _require(isinstance(entry, dict), f"{where}: must be an object")
        missing = sorted(_CORRECTION_FIELDS.difference(entry))
        _require(not missing, f"{where}: missing {', '.join(missing)}")
        extra = sorted(set(entry).difference(_CORRECTION_FIELDS))
        if extra:
            _handle_unknown_fields(
                f"{where}: unknown fields: {', '.join(extra)}",
                on_unknown_field=on_unknown_field,
            )
        field = entry["field"]
        _require(
            field in _CORRECTABLE_FIELDS,
            f"{where}: {field!r} is not correctable in place. Only "
            f"{', '.join(sorted(_CORRECTABLE_FIELDS))} may be restated; everything else "
            "is a measurement and needs re-measurement, not an edit.",
        )
        reason = entry["reason"]
        _require(
            isinstance(reason, str) and bool(reason.strip()),
            f"{where}: reason must say why the stored summary was wrong",
        )


def signal_from_payload(
    name: str, payload: dict[str, Any], *, on_unknown_field: OnUnknownField = "raise"
) -> WeakSignal:

    unknown = sorted(set(payload).difference(_SIGNAL_FIELDS))
    if unknown:
        _handle_unknown_fields(
            f"Signal {name!r} has unknown fields: {', '.join(unknown)}",
            on_unknown_field=on_unknown_field,
        )
    for field in ("recorded_at", "description", "source", "effect", "effect_units"):
        _require(field in payload, f"Signal {name!r} is missing {field!r}")
    _validate_corrections(name, payload.get("corrections"), on_unknown_field=on_unknown_field)
    classification = payload.get("classification")
    status = payload.get("status", "active")
    validate_invalidation(
        status=status,
        classification=classification,
        reason=payload.get("invalidated_reason"),
        superseded_by=payload.get("superseded_by"),
    )
    if status == "invalidated":
        _require(
            "superseded_by" in payload, "invalidated entries require superseded_by (may be null)"
        )
    _require(
        classification in CLASSIFICATIONS,
        f"Signal {name!r} has unknown classification {classification!r}; "
        f"expected one of {', '.join(CLASSIFICATIONS)}",
    )
    units = payload["effect_units"]
    _require(
        units in EFFECT_UNITS,
        f"Signal {name!r} has unknown effect_units {units!r}; "
        f"expected one of {', '.join(EFFECT_UNITS)}",
    )
    league = payload.get("league")
    _require(league in LEAGUES, f"Signal {name!r} has unknown league {league!r}")
    category = payload.get("category")
    _require(
        category is None or category in CATEGORIES,
        f"Signal {name!r} has unknown category {category!r}; "
        f"expected one of {', '.join(CATEGORIES)} or omitted",
    )
    seasons = payload.get("seasons")
    _require(
        isinstance(seasons, (list, tuple)) and len(seasons) == 2,
        f"Signal {name!r} needs seasons as a two-element [start, end]",
    )
    assert isinstance(seasons, (list, tuple))
    start, end = int(seasons[0]), int(seasons[1])
    _require(start <= end, f"Signal {name!r} has seasons out of order")
    effect = float(payload["effect"])

    interval_payload = payload.get("interval")
    interval: tuple[float, float] | None = None
    if interval_payload is not None:
        _require(
            isinstance(interval_payload, (list, tuple)) and len(interval_payload) == 2,
            f"Signal {name!r} needs interval as a two-element [low, high]",
        )
        assert isinstance(interval_payload, (list, tuple))
        low, high = float(interval_payload[0]), float(interval_payload[1])
        interval = (low, high)

    standard_error = payload.get("standard_error")
    if standard_error is not None:
        standard_error = float(standard_error)

    probability_positive = payload.get("probability_positive")
    if probability_positive is not None:
        probability_positive = float(probability_positive)

    _validate_signal_numeric_fields(
        name,
        effect=effect,
        standard_error=standard_error,
        interval=interval,
        probability_positive=probability_positive,
    )

    probability_positive_payload = payload.get("probability_positive")
    reliability = payload.get("reliability")
    closing_ground = payload.get("closing_ground")
    evidence = str(payload.get("classification_evidence", ""))
    validate_closure(
        name,
        classification=str(classification),
        closing_ground=None if closing_ground is None else str(closing_ground),
        classification_evidence=evidence,
        interval=interval,
        reliability=None if reliability is None else float(reliability),
        probability_positive=(
            None if probability_positive_payload is None else float(probability_positive_payload)
        ),
    )

    return WeakSignal(
        name=name,
        recorded_at=str(payload["recorded_at"]),
        description=str(payload["description"]),
        source=str(payload["source"]),
        effect=effect,
        effect_units=str(units),
        classification=str(classification),
        league=str(league),
        seasons=(start, end),
        standard_error=standard_error,
        interval=interval,
        probability_positive=probability_positive,
        sample_games=None if payload.get("sample_games") is None else int(payload["sample_games"]),
        sample_blocks=(
            None if payload.get("sample_blocks") is None else int(payload["sample_blocks"])
        ),
        classification_evidence=evidence,
        closing_ground=None if closing_ground is None else str(closing_ground),
        reliability=None if reliability is None else float(reliability),
        family=None if payload.get("family") is None else str(payload["family"]),
        notes=str(payload.get("notes", "")),
        plain_summary=(
            None if payload.get("plain_summary") is None else str(payload["plain_summary"])
        ),
        category=None if category is None else str(category),
        status=status,
        invalidated_reason=payload.get("invalidated_reason"),
        superseded_by=payload.get("superseded_by"),
    )


def registry_from_payload(
    payload: dict[str, Any], *, on_unknown_field: OnUnknownField = "raise"
) -> Registry:

    unknown = sorted(set(payload).difference(_TOP_LEVEL_FIELDS))
    if unknown:
        _handle_unknown_fields(
            f"Ledger has unknown top-level fields: {', '.join(unknown)}",
            on_unknown_field=on_unknown_field,
        )
    version = int(payload.get("version", 0))
    _require(
        version == WEAK_SIGNAL_REGISTRY_VERSION,
        f"Unsupported weak-signal registry version: {version}",
    )
    raw_signals = payload.get("signals", {})
    _require(isinstance(raw_signals, dict), "Ledger 'signals' must be an object")
    signals = {
        name: signal_from_payload(name, body, on_unknown_field=on_unknown_field)
        for name, body in raw_signals.items()
    }
    notes = tuple(str(note) for note in payload.get("notes", ()))
    return Registry(version=version, notes=notes, signals=signals)


@dataclass(frozen=True)
class QuarantinedSignal:
    name: str
    payload: dict[str, Any]
    error: str


def registry_from_payload_permissive(
    payload: dict[str, Any], *, on_unknown_field: OnUnknownField = "raise"
) -> tuple[Registry, dict[str, QuarantinedSignal]]:

    unknown = sorted(set(payload).difference(_TOP_LEVEL_FIELDS))
    if unknown:
        _handle_unknown_fields(
            f"Ledger has unknown top-level fields: {', '.join(unknown)}",
            on_unknown_field=on_unknown_field,
        )
    version = int(payload.get("version", 0))
    _require(
        version == WEAK_SIGNAL_REGISTRY_VERSION,
        f"Unsupported weak-signal registry version: {version}",
    )
    raw_signals = payload.get("signals", {})
    _require(isinstance(raw_signals, dict), "Ledger 'signals' must be an object")
    signals: dict[str, WeakSignal] = {}
    quarantined: dict[str, QuarantinedSignal] = {}
    for name, body in raw_signals.items():
        key = str(name)
        try:
            signals[key] = signal_from_payload(key, body, on_unknown_field=on_unknown_field)
        except WeakSignalError as error:
            quarantined[key] = QuarantinedSignal(name=key, payload=body, error=str(error))
    notes = tuple(str(note) for note in payload.get("notes", ()))
    return Registry(version=version, notes=notes, signals=signals), quarantined


def registry_to_payload(registry: Registry) -> dict[str, Any]:
    signals: dict[str, Any] = {}
    for name, signal in sorted(registry.signals.items()):
        body: dict[str, Any] = {
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
            "family": signal.family,
            "notes": signal.notes,
            "plain_summary": signal.plain_summary,
            "category": signal.category,
        }
        signals[name] = body
        if signal.status != "active":
            body.update(
                status=signal.status,
                invalidated_reason=signal.invalidated_reason,
                superseded_by=signal.superseded_by,
            )
    return {
        "version": registry.version,
        "notes": list(registry.notes),
        "signals": signals,
    }


def default_registry_path(root: Path | None = None) -> Path:

    base = Path(os.environ.get("NFL_ATS_REGISTRY_DIR", "registry")) if root is None else root
    return base / WEAK_SIGNAL_REGISTRY_FILENAME


def load_registry(path: Path, *, on_unknown_field: OnUnknownField = "raise") -> Registry:

    if not path.is_file():
        return Registry(version=WEAK_SIGNAL_REGISTRY_VERSION, notes=(), signals={})
    payload = json.loads(path.read_text(encoding="utf-8"))
    return registry_from_payload(payload, on_unknown_field=on_unknown_field)


def load_registry_permissive(
    path: Path, *, on_unknown_field: OnUnknownField = "raise"
) -> tuple[Registry, dict[str, QuarantinedSignal]]:

    if not path.is_file():
        return Registry(version=WEAK_SIGNAL_REGISTRY_VERSION, notes=(), signals={}), {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return registry_from_payload_permissive(payload, on_unknown_field=on_unknown_field)


def save_registry(registry: Registry, path: Path) -> None:
    atomic_json(registry_to_payload(registry), path)


def save_registry_preserving_quarantine(
    registry: Registry, quarantined: Mapping[str, QuarantinedSignal], path: Path
) -> None:

    payload = registry_to_payload(registry)
    signals = payload["signals"]
    for name, entry in quarantined.items():
        if name not in registry.signals:
            signals[name] = entry.payload
    payload["signals"] = dict(sorted(signals.items()))
    atomic_json(payload, path)


def record_signal(registry: Registry, signal: WeakSignal, *, replace: bool = False) -> Registry:

    validate_invalidation(
        status=signal.status,
        classification=signal.classification,
        reason=signal.invalidated_reason,
        superseded_by=signal.superseded_by,
    )
    if signal.name in registry.signals and registry.signals[signal.name].status == "invalidated":
        raise WeakSignalError("Keep invalidated history; record the replacement under a new name")
    if signal.name in registry.signals and not replace:
        raise WeakSignalError(
            f"Signal {signal.name!r} is already recorded; pass replace=True to correct it"
        )
    _validate_signal_numeric_fields(
        signal.name,
        effect=signal.effect,
        standard_error=signal.standard_error,
        interval=signal.interval,
        probability_positive=signal.probability_positive,
    )
    validate_closure(
        signal.name,
        classification=signal.classification,
        closing_ground=signal.closing_ground,
        classification_evidence=signal.classification_evidence,
        interval=signal.interval,
        reliability=signal.reliability,
        probability_positive=signal.probability_positive,
    )
    validate_coherence(signal.name, effect=signal.effect, interval=signal.interval)
    floor = _floor_standard_error_for_record(registry, signal)
    if floor is not None and signal.standard_error is not None and signal.standard_error < floor:
        warnings.warn(
            f"Signal {signal.name!r}: standard_error {signal.standard_error:.6g} is narrower "
            f"than its own sample size plausibly supports; widened to {floor:.6g} at record "
            "time instead of stored as offered (docs/weak_signal_pooling.md defect 4's floor, "
            "applied here so this cannot be stored narrower than the pool already trusts)",
            ImplausibleStandardErrorWarning,
            stacklevel=2,
        )
        signal = dataclasses.replace(signal, standard_error=floor)
    signals = dict(registry.signals)
    signals[signal.name] = signal
    return Registry(version=registry.version, notes=registry.notes, signals=signals)


def validate_invalidation(
    *, status: str, classification: str | None, reason: str | None, superseded_by: str | None
) -> None:
    _require(status in ("active", "invalidated"), f"Unknown signal status {status!r}")
    if status == "invalidated":
        _require(
            classification not in TERMINAL_CLASSIFICATIONS,
            "invalidated entries cannot carry a terminal classification; "
            "invalidation is not closure",
        )
        _require(isinstance(reason, str) and bool(reason.strip()), "invalidated_reason is required")
        _require(
            superseded_by is None
            or (isinstance(superseded_by, str) and bool(superseded_by.strip())),
            "superseded_by must be a non-empty entry name or null",
        )
    else:
        _require(
            reason is None and superseded_by is None,
            "active entries cannot carry invalidation metadata",
        )


def invalidate_signal(
    registry: Registry,
    *,
    name: str,
    reason: str,
    superseded_by: str | None = None,
    changed_at: str | None = None,
) -> Registry:
    _require(name in registry.signals, f"No recorded signal named {name!r}")
    signal = registry.signals[name]
    validate_invalidation(
        status="invalidated",
        classification=signal.classification,
        reason=reason,
        superseded_by=superseded_by,
    )
    if superseded_by is not None:
        _require(
            superseded_by != name and superseded_by in registry.signals,
            "superseded_by must name a different recorded entry",
        )
        _require(registry.signals[superseded_by].status == "active", "replacement must be active")
    if (signal.status, signal.invalidated_reason, signal.superseded_by) == (
        "invalidated",
        reason,
        superseded_by,
    ):
        return registry
    timestamp = changed_at or datetime.now(UTC).isoformat()
    audit = (
        f"[{timestamp}] invalidated: {reason}. Superseded by: {superseded_by!r}. "
        "Invalidation is not closure."
    )
    updated = replace(
        signal,
        status="invalidated",
        invalidated_reason=reason,
        superseded_by=superseded_by,
        notes=f"{signal.notes}\n{audit}".strip(),
    )
    return replace(registry, signals={**registry.signals, name: updated})


def retag_effect_units(
    registry: Registry,
    name: str,
    *,
    effect_units: str,
    reason: str,
    changed_at: str | None = None,
) -> Registry:

    _require(name in registry.signals, f"No recorded signal named {name!r}")
    _require(
        effect_units in EFFECT_UNITS,
        f"Unknown effect_units {effect_units!r}; expected one of {', '.join(EFFECT_UNITS)}",
    )
    signal = registry.signals[name]
    timestamp = changed_at or datetime.now(UTC).isoformat()
    audit_line = (
        f"[{timestamp}] effect_units retagged: {signal.effect_units!r} -> "
        f"{effect_units!r}. Reason: {reason}"
    )
    notes = f"{signal.notes}\n{audit_line}" if signal.notes else audit_line
    retagged = replace(signal, effect_units=effect_units, notes=notes)
    signals = dict(registry.signals)
    signals[name] = retagged
    return Registry(version=registry.version, notes=registry.notes, signals=signals)


def set_reliability(
    registry: Registry,
    name: str,
    *,
    reliability: float,
    reliability_low: float,
    reliability_high: float,
    method: str,
    source: str,
    reason: str,
    changed_at: str | None = None,
) -> Registry:

    _require(name in registry.signals, f"No recorded signal named {name!r}")
    for label, value in (
        ("reliability", reliability),
        ("reliability_low", reliability_low),
        ("reliability_high", reliability_high),
    ):
        _require(
            isinstance(value, int | float) and math.isfinite(float(value)),
            f"{label} must be a finite number, got {value!r}; an unmeasurable "
            "reliability is reported as unmeasured, never written as a number",
        )
        _require(
            -1.0 <= float(value) <= 1.0,
            f"{label} {float(value):.4f} is outside [-1, 1]; reliability is a correlation",
        )
    _require(
        reliability_low <= reliability <= reliability_high,
        f"Interval [{reliability_low:.4f}, {reliability_high:.4f}] does not contain the "
        f"point estimate {reliability:.4f}",
    )
    _require(bool(method.strip()), "method is required: name the quantity that was measured")
    _require(bool(source.strip()), "source is required: the artifact path holding the measurement")
    _require(bool(reason.strip()), "reason is required")

    signal = registry.signals[name]
    timestamp = changed_at or datetime.now(UTC).isoformat()
    audit_line = (
        f"[{timestamp}] reliability set: {signal.reliability!r} -> {float(reliability):.4f} "
        f"95% [{float(reliability_low):.4f}, {float(reliability_high):.4f}]. "
        f"Method: {method}. Measured from: {source}. Reason: {reason}. "
        "Reliability fields only; effect/interval/classification/closing_ground untouched, "
        "and this measurement does not by itself reclassify the entry."
    )
    notes = f"{signal.notes}\n{audit_line}" if signal.notes else audit_line
    updated = replace(signal, reliability=float(reliability), notes=notes)
    signals = dict(registry.signals)
    signals[name] = updated
    return Registry(version=registry.version, notes=registry.notes, signals=signals)


def sign_test(signals: Sequence[WeakSignal]) -> dict[str, Any]:

    excluded_invalidated = sum(s.status == "invalidated" for s in signals)
    signals = [s for s in signals if s.status != "invalidated"]
    favourable = sum(1 for signal in signals if signal.direction > 0)
    against = sum(1 for signal in signals if signal.direction < 0)
    ties = len(signals) - favourable - against
    informative = favourable + against
    p_value = binomial_two_sided_p(favourable, informative)
    return {
        "signals": len(signals),
        "excluded_invalidated": excluded_invalidated,
        "tie_convention": (
            "ties excluded from the test (classical sign test); "
            "'*_half_credit' fields report the ties-split-evenly reading"
        ),
        "favouring_candidate": favourable,
        "favouring_baseline": against,
        "ties": ties,
        "informative_signals": informative,
        "favouring_candidate_half_credit": favourable + 0.5 * ties,
        "favouring_baseline_half_credit": against + 0.5 * ties,
        "share_favouring_candidate": (None if informative == 0 else favourable / informative),
        "share_favouring_candidate_half_credit": (
            None if not signals else (favourable + 0.5 * ties) / len(signals)
        ),
        "p_value": p_value,
        "interpretation": (
            "no signals recorded"
            if not signals
            else (
                "every signal is an exact tie; the pile has no direction to test"
                if informative == 0
                else (
                    "directions are consistent with a coin flip"
                    if p_value > 0.10
                    else "directions lean further than chance comfortably explains"
                )
            )
        ),
    }


POOLING_WEIGHTINGS = ("sample_floored", "inverse_variance")

_IMPLAUSIBLE_SE_ROBUST_SIGMAS = 3.0


def _median_games_per_block(signals: Sequence[WeakSignal]) -> float | None:

    per_block = [
        s.sample_games / s.sample_blocks
        for s in signals
        if s.sample_games and s.sample_blocks and s.sample_blocks > 0
    ]
    return statistics.median(per_block) if per_block else None


def _pool_sample_sizes(usable: Sequence[WeakSignal]) -> tuple[list[float] | None, int]:

    games_per_block = _median_games_per_block(usable)

    partial: list[float | None] = []
    for signal in usable:
        if signal.sample_games and signal.sample_games > 0:
            partial.append(float(signal.sample_games))
        elif signal.sample_blocks and signal.sample_blocks > 0 and games_per_block:
            partial.append(float(signal.sample_blocks) * games_per_block)
        else:
            partial.append(None)

    known = [n for n in partial if n is not None]
    if not known:
        return None, 0
    fallback = statistics.median(known)
    return [fallback if n is None else n for n in partial], len(partial) - len(known)


@dataclass(frozen=True)
class _PlausibilityCurve:
    scale: float
    sample_sizes: list[float]
    log_ratios: list[float]
    cutoff: float

    def floor_for(self, index: int) -> float:

        return math.exp(self.cutoff) * math.sqrt(self.scale / self.sample_sizes[index])

    def is_implausible(self, index: int) -> bool:
        return self.log_ratios[index] < self.cutoff


def _plausibility_curve(usable: Sequence[WeakSignal]) -> _PlausibilityCurve | None:

    sample_sizes, _ = _pool_sample_sizes(usable)
    if sample_sizes is None or len(usable) < 3:
        return None
    errors = [s.resolved_standard_error() for s in usable]
    positive = [
        (float(se), n)
        for se, n in zip(errors, sample_sizes, strict=True)
        if se is not None and se > 0.0 and n > 0.0
    ]
    if len(positive) < 3:
        return None
    scale = statistics.median([se**2 * n for se, n in positive])
    if scale <= 0.0:
        return None

    log_ratios = [
        math.log(float(se) / math.sqrt(scale / n))
        if se is not None and se > 0.0 and n > 0.0
        else -math.inf
        for se, n in zip(errors, sample_sizes, strict=True)
    ]
    finite = [r for r in log_ratios if math.isfinite(r)]
    centre = statistics.median(finite)
    deviations = [abs(r - centre) for r in finite]
    robust_sigma = 1.4826 * statistics.median(deviations)
    if robust_sigma <= 0.0:
        robust_sigma = 1.2533 * statistics.fmean(deviations)
    if robust_sigma <= 0.0:
        return None
    cutoff = centre - _IMPLAUSIBLE_SE_ROBUST_SIGMAS * robust_sigma
    return _PlausibilityCurve(
        scale=scale, sample_sizes=sample_sizes, log_ratios=log_ratios, cutoff=cutoff
    )


class ImplausibleStandardErrorWarning(UserWarning):
    pass


def _floor_standard_error_for_record(registry: Registry, signal: WeakSignal) -> float | None:

    if signal.standard_error is None or signal.standard_error <= 0.0:
        return None
    pool_source = [
        other
        for name, other in registry.signals.items()
        if name != signal.name
        and other.status != "invalidated"
        and other.effect_units == signal.effect_units
    ]
    curve = _plausibility_curve(pool_source)
    if curve is None:
        return None
    sample_games: float | None = None
    if signal.sample_games and signal.sample_games > 0:
        sample_games = float(signal.sample_games)
    else:
        games_per_block = _median_games_per_block(pool_source)
        if signal.sample_blocks and signal.sample_blocks > 0 and games_per_block:
            sample_games = float(signal.sample_blocks) * games_per_block
    if sample_games is None or sample_games <= 0.0:
        return None
    return math.exp(curve.cutoff) * math.sqrt(curve.scale / sample_games)


def pooled_effect(
    signals: Sequence[WeakSignal],
    *,
    method: str = "random",
    weighting: str = "sample_floored",
) -> dict[str, Any]:

    _require(method in ("fixed", "random"), f"Unknown pooling method {method!r}")
    _require(
        weighting in POOLING_WEIGHTINGS,
        f"Unknown pooling weighting {weighting!r}. Expected one of "
        f"{', '.join(POOLING_WEIGHTINGS)}.",
    )
    excluded_invalidated = sum(s.status == "invalidated" for s in signals)
    usable = [
        s for s in signals if s.status != "invalidated" and s.resolved_standard_error() is not None
    ]
    if not usable:
        return {
            "signals": 0,
            "excluded_invalidated": excluded_invalidated,
            "pooled_effect": None,
            "standard_error": None,
            "interval": None,
            "note": "no signal carried a standard error or interval to pool",
        }

    units = {s.effect_units for s in usable}
    _require(
        len(units) == 1,
        f"Cannot pool across mixed effect units: {', '.join(sorted(units))}. "
        "Pool within a unit, or convert deliberately before recording.",
    )

    effects = [s.effect for s in usable]
    errors = [float(se) for s in usable if (se := s.resolved_standard_error()) is not None]
    recorded_variances = [se**2 for se in errors]

    _, imputed = _pool_sample_sizes(usable)
    curve = _plausibility_curve(usable) if weighting == "sample_floored" else None
    applied = weighting
    floored: list[dict[str, Any]] = []
    if weighting == "sample_floored" and curve is None:
        applied = "inverse_variance_fallback_no_curve"
        variances = recorded_variances
    elif curve is not None:
        variances = []
        for index, recorded in enumerate(recorded_variances):
            floor = curve.floor_for(index) ** 2
            if curve.is_implausible(index):
                floored.append(
                    {
                        "name": usable[index].name,
                        "recorded_standard_error": math.sqrt(recorded),
                        "floored_to": math.sqrt(floor),
                        "sample_games": usable[index].sample_games,
                    }
                )
                variances.append(floor)
            else:
                variances.append(recorded)
    else:
        variances = recorded_variances

    if any(v <= 0.0 for v in variances):
        raise WeakSignalError(
            "Cannot pool a signal whose variance is zero or negative; "
            "re-measure it or record an interval that has width."
        )

    weights = [1.0 / v for v in variances]
    total_weight = sum(weights)
    fixed_mean = sum(w * e for w, e in zip(weights, effects, strict=True)) / total_weight

    q = sum(w * (e - fixed_mean) ** 2 for w, e in zip(weights, effects, strict=True))
    degrees = len(usable) - 1
    tau_squared = 0.0
    if degrees > 0 and total_weight > 0:
        c = total_weight - sum(w**2 for w in weights) / total_weight
        if c > 0:
            tau_squared = max(0.0, (q - degrees) / c)

    if method == "random" and tau_squared > 0.0:
        weights = [1.0 / (v + tau_squared) for v in variances]
        total_weight = sum(weights)

    mean = sum(w * e for w, e in zip(weights, effects, strict=True)) / total_weight
    standard_error = math.sqrt(1.0 / total_weight)
    half = 1.959963984540054 * standard_error

    shares = [w / total_weight for w in weights]
    heaviest = max(range(len(shares)), key=lambda i: shares[i])
    effective_signals = (total_weight**2) / sum(w**2 for w in weights)

    result: dict[str, Any] = {
        "signals": len(usable),
        "excluded_invalidated": excluded_invalidated,
        "method": method,
        "weighting": applied,
        "effect_units": usable[0].effect_units,
        "pooled_effect": mean,
        "standard_error": standard_error,
        "interval": (mean - half, mean + half),
        "probability_positive": (
            None
            if standard_error <= 0.0
            else 0.5 * math.erfc(-mean / (standard_error * math.sqrt(2.0)))
        ),
        "excludes_zero": bool((mean - half) * (mean + half) > 0.0),
        "heterogeneity_tau_squared": tau_squared,
        "sharpening_vs_best_single": (
            None if standard_error == 0 else math.sqrt(min(variances)) / standard_error
        ),
        "max_weight_share": shares[heaviest],
        "most_influential_signal": usable[heaviest].name,
        "effective_signals": effective_signals,
        "per_game_variance": None if curve is None else curve.scale,
        "sample_sizes_imputed": imputed,
        "variance_floor_log_ratio_cutoff": None if curve is None else curve.cutoff,
        "standard_errors_floored": len(floored),
        "floored_signals": floored,
        "note": (
            "Pooling assumes the inputs are independent. Check overlap_warnings "
            "before believing this interval. Entries listed under "
            "'floored_signals' recorded a band too narrow for their own sample "
            "size and need re-measurement; they are still pooled, at the "
            "weakest precision their sample supports."
        ),
    }
    if applied == "sample_floored":
        note = (
            "The superseded 1/SE^2 weighting, reported so the change stays "
            "auditable. It weighted narrow bootstrap artifacts hardest; do "
            "not quote it."
        )
        try:
            legacy = pooled_effect(
                [s for s in signals if s.status != "invalidated"],
                method=method,
                weighting="inverse_variance",
            )
        except WeakSignalError as error:
            result["legacy_inverse_variance"] = {"unavailable": str(error), "note": note}
        else:
            result["legacy_inverse_variance"] = {
                "pooled_effect": legacy["pooled_effect"],
                "interval": legacy["interval"],
                "excludes_zero": legacy["excludes_zero"],
                "max_weight_share": legacy["max_weight_share"],
                "most_influential_signal": legacy["most_influential_signal"],
                "effective_signals": legacy["effective_signals"],
                "note": note,
            }
    return result


def implausible_standard_errors(signals: Sequence[WeakSignal]) -> list[dict[str, Any]]:

    usable = [
        s for s in signals if s.status != "invalidated" and s.resolved_standard_error() is not None
    ]
    groups: dict[tuple[str, str], list[WeakSignal]] = {}
    for signal in usable:
        groups.setdefault((signal.league, signal.effect_units), []).append(signal)

    flagged: list[dict[str, Any]] = []
    for group in groups.values():
        curve = _plausibility_curve(group)
        if curve is None:
            continue
        flagged.extend(
            {
                "name": signal.name,
                "league": signal.league,
                "effect_units": signal.effect_units,
                "standard_error": signal.resolved_standard_error(),
                "sample_games": signal.sample_games,
                "plausible_floor": curve.floor_for(index),
                "log_ratio": curve.log_ratios[index],
            }
            for index, signal in enumerate(group)
            if curve.is_implausible(index)
        )
    return sorted(flagged, key=lambda row: float(row["log_ratio"]))


def rows_needing_remeasurement(signals: Sequence[WeakSignal]) -> dict[str, Any]:

    active = [s for s in signals if s.status != "invalidated"]
    deterministic = [
        s.name
        for s in active
        if s.probability_positive == 0.0 and s.effect == 0.0 and s.interval == (0.0, 0.0)
    ]
    unknown = [s.name for s in active if s.probability_positive == 0.0 and s.effect != 0.0]
    implausible = implausible_standard_errors(active)
    return {
        "zero_atom_probability_positive": {
            "count": len(deterministic),
            "correctable_value": 0.5,
            "signals": sorted(deterministic),
            "note": (
                "Recorded 0.0 by the strict 'draws > 0' convention; the "
                "measurement is a dead heat, whose value under the corrected "
                "convention is 0.5. Not rewritten in place -- propose the "
                "correction explicitly rather than editing recorded numbers."
            ),
        },
        "strict_zero_with_nonzero_effect": {
            "count": len(unknown),
            "signals": sorted(unknown),
            "note": (
                "Cannot be corrected from the registry: whether a zero atom was "
                "folded in depends on draws that were never stored. Re-measure "
                "to find out; do not assume either way."
            ),
        },
        "implausible_standard_error": {
            "count": len(implausible),
            "signals": [row["name"] for row in implausible],
            "note": (
                "The recorded band is far narrower than this row's own sample "
                "size can support. Pooling floors these rather than trusting "
                "or dropping them."
            ),
        },
        "total_flagged": len(deterministic) + len(unknown) + len(implausible),
        "nothing_is_closed_by_this": (
            "Flagging a measurement for re-measurement is not a verdict on the "
            "signal. No row here is refuted, bounded by a control, or "
            "reclassified."
        ),
    }


def signal_family(signal: WeakSignal) -> str:

    if signal.family:
        return signal.family
    name = signal.name
    changed = True
    while changed:
        changed = False
        for pattern in _FAMILY_DECOMPOSITION_SUFFIXES:
            stripped = pattern.sub("", name)
            if stripped and stripped != name:
                name = stripped
                changed = True
    tokens = name.split("_")
    for index, token in enumerate(tokens[:3]):
        if token in FAMILY_BATTERY_MARKERS:
            return "_".join(tokens[: index + 1])
    return name


def _interval_overlap_counts(lows: Sequence[int], highs: Sequence[int]) -> tuple[int, list[bool]]:
    count = len(lows)
    if count < 2:
        return 0, [False] * count
    sorted_low = sorted(lows)
    sorted_high = sorted(highs)
    before = [bisect.bisect_left(sorted_high, value) for value in lows]
    after = [count - bisect.bisect_right(sorted_low, value) for value in highs]
    pairs = count * (count - 1) // 2 - sum(before)
    participates = [
        earlier + later < count - 1 for earlier, later in zip(before, after, strict=True)
    ]
    return pairs, participates


def family_overlap_warnings(signals: Sequence[WeakSignal]) -> dict[str, Any]:

    excluded_invalidated = sum(s.status == "invalidated" for s in signals)
    ordered = sorted((s for s in signals if s.status != "invalidated"), key=lambda s: s.name)
    families: dict[tuple[str, str], list[WeakSignal]] = {}
    for signal in ordered:
        families.setdefault((signal.league, signal_family(signal)), []).append(signal)

    pairwise_pairs = 0
    within_family: list[dict[str, Any]] = []
    for (league, family), members in sorted(families.items()):
        pair_count, participates = _interval_overlap_counts(
            [signal.seasons[0] for signal in members],
            [signal.seasons[1] for signal in members],
        )
        pairwise_pairs += pair_count
        overlapping = [signal for signal, hit in zip(members, participates, strict=True) if hit]
        if not overlapping:
            continue
        low_season = min(signal.seasons[0] for signal in overlapping)
        high_season = max(signal.seasons[1] for signal in overlapping)
        within_family.append(
            {
                "family": family,
                "league": league,
                "members": len(members),
                "overlapping_members": len({signal.name for signal in overlapping}),
                "shared_seasons": [low_season, high_season],
                "member_names": [signal.name for signal in members],
                "warning": (
                    f"family '{family}' ({league}) holds {len(members)} signals whose "
                    f"measurement windows overlap on {league} seasons {low_season}-"
                    f"{high_season}; they are correlated decompositions of the same "
                    "football, so pooling them or counting their signs separately "
                    "overstates precision"
                ),
            }
        )

    family_spans = [
        (league, family, min(s.seasons[0] for s in members), max(s.seasons[1] for s in members))
        for (league, family), members in sorted(families.items())
    ]
    cross_family_pairs = 0
    for league in {span[0] for span in family_spans}:
        spans = [span for span in family_spans if span[0] == league]
        cross_family_pairs += _interval_overlap_counts(
            [span[2] for span in spans], [span[3] for span in spans]
        )[0]
    total_pairwise_pairs = 0
    for league in {signal.league for signal in ordered}:
        members = [signal for signal in ordered if signal.league == league]
        total_pairwise_pairs += _interval_overlap_counts(
            [signal.seasons[0] for signal in members],
            [signal.seasons[1] for signal in members],
        )[0]

    return {
        "families": len(families),
        "excluded_invalidated": excluded_invalidated,
        "families_with_internal_overlap": len(within_family),
        "within_family": sorted(
            within_family, key=lambda entry: (-entry["members"], entry["family"])
        ),
        "cross_family_shared_window_pairs": cross_family_pairs,
        "pairwise_overlap_pairs": total_pairwise_pairs,
        "pairwise_within_family_pairs": pairwise_pairs,
        "note": (
            "Within-family overlaps are correlated decompositions of shared windows "
            "(AGENTS.md; docs/registry_correlation_audit_20260822.md §3): treat each "
            "family as one dependent vote, not N independent ones. Cross-family pairs "
            "sharing seasons still share football; see the audit doc before trusting "
            "the pooled interval's precision."
        ),
    }


def overlap_warnings(signals: Sequence[WeakSignal]) -> list[str]:

    warnings: list[str] = []
    ordered = sorted((s for s in signals if s.status != "invalidated"), key=lambda s: s.name)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if first.league != second.league:
                continue
            low = max(first.seasons[0], second.seasons[0])
            high = min(first.seasons[1], second.seasons[1])
            if low <= high:
                warnings.append(
                    f"{first.name} and {second.name} share {first.league} seasons "
                    f"{low}-{high}; their errors are correlated and pooling them "
                    "overstates precision"
                )
    return warnings


def poolable_signals(
    registry: Registry,
    *,
    league: str | None = None,
    effect_units: str | None = None,
) -> list[WeakSignal]:

    chosen: list[WeakSignal] = []
    for signal in registry.signals.values():
        if signal.status == "invalidated" or signal.classification != POOLABLE_CLASSIFICATION:
            continue
        if league is not None and signal.league != league:
            continue
        if effect_units is not None and signal.effect_units != effect_units:
            continue
        chosen.append(signal)
    return sorted(chosen, key=lambda s: s.name)


def combination_report(
    registry: Registry,
    *,
    league: str | None = None,
    effect_units: str | None = None,
    method: str = "random",
    weighting: str = "sample_floored",
) -> dict[str, Any]:

    eligible = poolable_signals(registry, league=league, effect_units=effect_units)
    invalidated = [
        s
        for s in registry.signals.values()
        if s.status == "invalidated"
        and league in (None, s.league)
        and effect_units in (None, s.effect_units)
    ]
    leagues = {signal.league for signal in eligible}
    if league is None and len(leagues) > 1:
        raise ValueError(
            "Refusing to pool across leagues ("
            + ", ".join(sorted(leagues))
            + "). Pooled inputs must be commensurable -- same units, same "
            "scale, same population -- so pass an explicit league instead of "
            "averaging two different ones."
        )
    excluded = {
        name: signal.invalidated_reason if signal.status == "invalidated" else signal.classification
        for name, signal in sorted(registry.signals.items())
        if signal.status == "invalidated" or signal.classification != POOLABLE_CLASSIFICATION
    }
    unit_groups: dict[str, list[WeakSignal]] = {}
    for signal in eligible:
        unit_groups.setdefault(signal.effect_units, []).append(signal)

    pooled: dict[str, Any] = {}
    for unit, group in sorted(unit_groups.items()):
        pooled[unit] = pooled_effect(
            group + [s for s in invalidated if s.effect_units == unit],
            method=method,
            weighting=weighting,
        )

    used_seasons = sorted({season for signal in eligible for season in signal.season_range})
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "eligible": [signal.name for signal in eligible],
        "excluded_with_reason": excluded,
        "excluded_invalidated": len(invalidated),
        "sign_test": sign_test(eligible + invalidated),
        "needs_remeasurement": rows_needing_remeasurement(eligible + invalidated),
        "pooled_by_unit": pooled,
        "overlap_warnings": family_overlap_warnings(eligible + invalidated),
        "overlap_pairwise_count": len(overlap_warnings(eligible)),
        "measurement_coherence_problems": coherence_problems(eligible),
        "seasons_touched_by_inputs": used_seasons,
        "guidance": (
            "A pooled estimate is evidence that a combined candidate is worth "
            "building, not evidence that it works. Confirm it once, predeclared, "
            "on a rotation window none of these inputs touched."
        ),
    }


def signals_from_iterable(entries: Iterable[dict[str, Any]]) -> list[WeakSignal]:

    built: list[WeakSignal] = []
    for entry in entries:
        payload = dict(entry)
        name = payload.pop("name", None)
        _require(bool(name), "Every entry needs a 'name'")
        built.append(signal_from_payload(str(name), payload))
    return built
