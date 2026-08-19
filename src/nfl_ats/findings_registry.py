"""The findings page's evidence spine: load the machine-readable registries,
fingerprint what they say, and hand back the auto-generated "leads" content.

Why this module exists: the owner's complaint was that ``findings.html`` goes
stale because its ~24 hand-written entries quote numbers that keep moving in
``registry/weak_signals.json`` (recorded by ``nfl-ats weak-signals record``)
and ``registry/rotation_registry.json`` (recorded by ``nfl-ats rotation
record-look``), and nothing ever re-checks the prose against the record it
claims to summarize. This module is the fix, in two pieces:

1. **Curation validation.** Every curated
   :class:`nfl_ats.dashboard.findings_content.Finding` that is not marked
   ``evergreen`` must name the registry entries it summarizes
   (``registry_keys``) and carry a content fingerprint of each one, taken the
   day the prose was last verified (``curated_as_of``). :func:`validate_curation`
   recomputes each fingerprint from the LIVE registries at render time and
   raises :class:`CurationError` the moment either a named key no longer
   exists or its recorded content has moved out from under the prose. A
   session that records new evidence therefore either does nothing (most
   curated claims are untouched) or gets a loud, specific build failure that
   names the stale finding and the key that moved -- never a silent drift.
2. **Auto-rendered leads.** :func:`top_open_leads` needs no curation at all:
   it reads ``registry/weak_signals.json`` directly and ranks the open,
   ``unresolved_below_power`` leads by how far their ``probability_positive``
   sits from a coin flip. New evidence recorded this way (the overwhelming
   majority of what gets recorded) appears on the findings page the next time
   it is generated, with no prose to write and no key to wire.

Render-semantics contract, restated here because this module is the one place
that decides which registry rows even reach the page (see AGENTS.md, "An
interval crossing zero is NOT grounds for rejection"): ``unresolved_below_power``
is not a negative and this module never filters, ranks, or labels it as one --
it is the ONLY classification :func:`top_open_leads` draws from. Only
``refuted_mechanism`` and ``bounded_by_control`` are real negatives, and this
module does not render verdicts for those at all (the curated findings do,
by hand, because a closed line of work is exactly the kind of claim a human
should phrase). Nothing here writes to any registry -- every function is a
reader.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nfl_ats import rotation, weak_signals

STORE_WEAK_SIGNAL = "weak_signal"
STORE_ROTATION = "rotation"
STORE_CHALLENGER = "challenger"
STORES = (STORE_WEAK_SIGNAL, STORE_ROTATION, STORE_CHALLENGER)


class CurationError(ValueError):
    """A curated finding names a registry key that does not exist, or whose
    recorded content has moved since the prose was last verified against it.

    Raised at RENDER time (``nfl-ats publish-board``), never at import time,
    so a normal test run that never builds the findings page is unaffected --
    only the thing that actually ships a stale claim fails.
    """


# ---------------------------------------------------------------------------
# Fingerprinting: a stable digest of "everything about this entry that a
# curated sentence could have been quoting."
# ---------------------------------------------------------------------------


def fingerprint(payload: Mapping[str, Any]) -> str:
    """A short, stable content hash. Deliberately over-inclusive: it hashes
    every field the registry stores for the entry, not just the ones today's
    prose happens to quote, so a correction to ``notes`` or
    ``classification_evidence`` is caught even when the headline number did
    not move."""

    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RegistryEntry:
    """One fact, normalized from whichever of the three evidence stores it
    came from, plus the fingerprint curation is checked against."""

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
    # The family's most recent window is its current, citable measurement --
    # rotation windows are never re-scored, so "most recent" and "current
    # verdict" are the same thing. A family with no window yet (declared but
    # unscreened, e.g. `cfb_role_continuity`) has no effect to cite, only a
    # status.
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
    # Hash the WHOLE record, not just the evidence block: a status change
    # (e.g. ACTIVE_PROSPECTIVE -> CLOSED_BEFORE_ACTIVATION) is exactly the
    # kind of move a curated sentence about "what we're tracking" must not
    # silently survive.
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
    """The weak-signal registry, from ``registry_root`` or the tracked default.

    A thin, shared path-resolution wrapper so callers outside this module
    (``public_board.py``'s "What we're watching" section, which needs the
    raw :class:`~nfl_ats.weak_signals.Registry` for :func:`top_open_leads`,
    not the flattened :class:`RegistryEntry` map) never have to duplicate the
    ``registry_root``-vs-default-path logic :func:`load_all_entries` also
    uses internally.
    """

    path = (
        registry_root / weak_signals.WEAK_SIGNAL_REGISTRY_FILENAME
        if registry_root is not None
        else weak_signals.default_registry_path()
    )
    return weak_signals.load_registry(path)


def load_rotation_registry(registry_root: Path | None = None) -> rotation.Registry:
    """The rotation registry, from ``registry_root`` or the tracked default.

    Feature-detected like every other optional reader in this codebase: a
    missing file (a fresh checkout with no rotation history yet) is an empty
    registry, not an error -- ``rotation.load_registry`` itself raises on a
    missing file, unlike ``weak_signals.load_registry``, so that is handled
    here.
    """

    path = (
        registry_root / rotation.ROTATION_REGISTRY_FILENAME
        if registry_root is not None
        else rotation.default_registry_path()
    )
    if not path.is_file():
        return rotation.Registry(version=rotation.ROTATION_REGISTRY_VERSION, notes=(), families={})
    return rotation.load_registry(path)


def load_all_entries(
    *,
    registry_root: Path | None = None,
    weak_signal_registry: weak_signals.Registry | None = None,
    rotation_registry: rotation.Registry | None = None,
    challengers: Sequence[Mapping[str, Any]] = (),
) -> dict[str, RegistryEntry]:
    """Every entry from all three evidence stores, keyed ``"<store>:<name>"``.

    ``registry_root`` overrides where ``weak_signals.json`` and
    ``rotation_registry.json`` are read from (both live in the same
    directory in every deployment of this repo); omit it to use each
    module's own ``NFL_ATS_REGISTRY_DIR``-aware default, exactly like every
    other reader in this codebase. ``weak_signal_registry``/
    ``rotation_registry`` accept an already-loaded :class:`~nfl_ats.weak_signals.Registry`/
    :class:`~nfl_ats.rotation.Registry` directly -- tests inject one, and a
    caller that has already loaded a registry for another reason (e.g.
    ``public_board.render_findings_page``'s own ``top_open_leads`` call)
    should never risk it disagreeing with a second, independently re-read
    copy used for curation. ``challengers`` is passed in already loaded (see
    :func:`nfl_ats.public_board.load_prospective_challengers`) rather than
    read here, so this module never needs to know the artifacts layout.
    """

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


# ---------------------------------------------------------------------------
# Curation validation
# ---------------------------------------------------------------------------


def validate_curation(findings: Sequence[Any], entries: Mapping[str, RegistryEntry]) -> None:
    """Raise :class:`CurationError` the instant a curated finding drifts.

    ``findings`` is ``nfl_ats.dashboard.findings_content.FINDINGS`` (typed
    loosely here, as a ``Sequence[Any]``, so this module never has to import
    ``findings_content`` -- that module imports registry helpers from here in
    a later step, and a two-way import would be circular). Each element must
    carry ``question``, ``evergreen``, ``registry_keys`` and
    ``registry_fingerprints``.

    Three failure modes, in the order they matter to a human fixing them:

    1. Not evergreen and cites nothing -- a curated claim with no traceable
       source, which is exactly the drift this module exists to prevent.
    2. Cites a key that does not exist in any live registry -- a typo, or a
       key that only ever existed in someone's scratch notes.
    3. Cites a key whose CONTENT has moved since ``curated_as_of`` -- the
       entry was re-recorded, corrected, or reclassified, and the prose was
       written against the old version.
    """

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


# ---------------------------------------------------------------------------
# Auto-rendered leads: "What we're watching"
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchingLead:
    """One open, below-power lead, ready to render with no hand-typed prose."""

    key: str
    name: str
    description: str
    effect: float
    effect_units: str
    interval: tuple[float, float] | None
    probability_positive: float
    seasons: tuple[int, int]
    league: str


# Two duplication patterns dominate the registry, and this function collapses
# both in two passes -- everything else stays distinct, because two
# differently-worded constructs from the same research family (e.g.
# "division revenge" vs "backup QB start", both bias-battery cells) are
# independently informative, not duplicates of each other.
#
# Pass 1 -- same construct, two grades. A construct measured twice -- once
# close-graded (the larger, older archive) and once re-screened at the
# opener (the line the pool actually uses) -- is recorded under the same
# name with an "_opener" suffix. The pair collapses to its opener-graded
# member (the pool-relevant grade) REGARDLESS of which grade happens to be
# statistically louder -- this is a provenance rule, not a ranking one.
#
# Pass 2 -- one slot per screening battery. A predeclared multi-cell
# screening battery (``bias_battery_*``, ``cfb_bias_battery_*``,
# ``attention_battery_*``, ``weather_battery_*``, ``odds_microstructure_*``)
# records EVERY cell -- 6 to 19 of them per battery -- as its own entry.
# Ranking by raw statistical extremity lets one battery's dozen mined cells
# crowd out every other family's single best lead, which is a sampling
# artifact of how many cells that battery happened to predeclare, not a sign
# the battery's leads are more real. Each battery collapses to its single
# most striking surviving cell.
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
    """The most striking open leads, entirely from the live registry.

    "Striking" is |probability_positive - 0.5|, the same statistic the rest
    of this project already uses to rank a lean without pretending an
    interval that crosses zero is a verdict. Only ``unresolved_below_power``
    signals are eligible -- this function never surfaces a
    ``refuted_mechanism`` or ``bounded_by_control`` entry, because those are
    real negatives, not open leads, and dressing them up as "what we're
    watching" would misstate a closed line of work as an active one.

    Entries whose description names an "oracle" construct (a measurement
    that uses information only available after the decision it is meant to
    inform -- this registry's ceiling/benchmark checks, not candidate
    pregame signals) are excluded on the same principle: they answer "how
    good could a perfect oracle be", already reported elsewhere on this site
    as a ceiling, not "is there an edge here."
    """

    candidates = [
        signal
        for signal in registry.signals.values()
        if signal.classification == weak_signals.POOLABLE_CLASSIFICATION
        and signal.league in leagues
        and signal.probability_positive is not None
        and "oracle" not in signal.description.lower()
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


__all__ = [
    "STORES",
    "STORE_CHALLENGER",
    "STORE_ROTATION",
    "STORE_WEAK_SIGNAL",
    "CurationError",
    "RegistryEntry",
    "WatchingLead",
    "fingerprint",
    "load_all_entries",
    "load_rotation_registry",
    "load_weak_signal_registry",
    "top_open_leads",
    "validate_curation",
]
