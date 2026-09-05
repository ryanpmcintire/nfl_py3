"""Registry of signals too small to resolve alone, and the arithmetic to pool them.

The evaluator resolves roughly 2 ATS points (RWB-15). Almost every candidate
feature in this sport is worth a fraction of that, so "no significant effect"
has been the *expected* outcome for real-but-small signals — and recording each
one as a negative quietly threw away the ones that were genuinely there. Two
such mistakes were caught on 2026-08-17 (4th-down aggressiveness and penalty
discipline, both filed as "priced" on intervals far too wide to say so).

This module is the fix. A signal that lands in category 3 of the taxonomy in
``docs/pool_edge_plan.md`` — *unresolved below detection power*, meaning its
interval contains both zero and the hypothesised effect — is no longer deleted.
It is recorded here with its effect, its uncertainty, and above all its
**direction**, and it waits.

Why waiting works, and this is the whole point:

- **Directions accumulate faster than precision.** Under a true null a point
  estimate is equally likely to fall either side of zero. Ten of twelve
  independent candidates leaning the same way is a binomial event with a
  p-value you can actually compute, long before any single one is resolvable.
  That is `sign_test`.
- **Pooling shrinks the standard error as sqrt(K).** Twelve signals each three
  times too noisy to see individually are, pooled, about 3.46 times sharper —
  which is to say, visible. That is `pooled_effect`.

Both come with a trap this repo has already documented: results measured on the
*same seasons* share noise, so pooling them overstates precision and can
manufacture a finding out of one lucky stretch of football. `overlap_warnings`
reports that rather than hiding it, and the honest use of a pooled estimate is
to justify ONE predeclared combined look on a window none of the inputs touched.

Nothing here scores a model or spends a rotation window. It records what was
measured, and does arithmetic on it.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nfl_ats.io import atomic_json

WEAK_SIGNAL_REGISTRY_VERSION = 1
WEAK_SIGNAL_REGISTRY_FILENAME = "weak_signals.json"

# The taxonomy in docs/pool_edge_plan.md. Only the first is poolable: a refuted
# mechanism is real evidence against, and a control-bounded null is a genuine
# negative. Pooling either of those in would be laundering a known failure.
CLASSIFICATIONS = (
    "unresolved_below_power",
    "refuted_mechanism",
    "bounded_by_control",
)
POOLABLE_CLASSIFICATION = "unresolved_below_power"
TERMINAL_CLASSIFICATIONS = ("refuted_mechanism", "bounded_by_control")

# AGENTS.md, "An interval crossing zero is NOT grounds for rejection (binding)":
# only two things justify closing a line of work, and each terminal entry must
# name which one it stands on. An interval containing zero is not on this list
# and never will be — at this evaluator's ~2-point resolution that outcome is
# EXPECTED for a real small signal, so treating it as a negative silently
# deletes exactly the signals worth keeping. Enforced in `signal_from_payload`
# so a session that never read AGENTS.md still cannot record the violation.
CLOSING_GROUNDS: dict[str, tuple[str, ...]] = {
    "refuted_mechanism": ("wrong_sign_resolved", "no_split_half_reliability"),
    "bounded_by_control": ("positive_control_bound",),
}

#: A ``no_split_half_reliability`` closure asserts the trait has no signal for
#: ANY sample size, so the recorded reliability measurement must actually sit
#: near zero. AGENTS.md: "wrong sign, or the trait has no split-half
#: reliability". A trait measured at reliability 0.719 (e.g. the CFB role
#: continuity traits) is reliable and can NEVER close on this ground; letting
#: such a number through was a validator gap, fixed 2026-08-24.
NO_SPLIT_HALF_RELIABILITY_MAX = 0.10

#: Families are inferred from names when no explicit ``family`` field is set.
#: These suffixes mark one construct decomposed into grades/eras/sub-windows --
#: the same measurement campaign, so its members share football and must not
#: count as independent votes. Mirrors ``findings_registry``'s duplication
#: passes (the ``_opener`` grade suffix and the battery-marker prefix rule).
_FAMILY_DECOMPOSITION_SUFFIXES = (
    re.compile(r"_opener$"),
    re.compile(r"_era_\d{4}_\d{4}$"),
    re.compile(r"_era_\d{4}$"),
    re.compile(r"_\d{4}_\d{4}$"),
    re.compile(r"_(?:pre|post)\d{4}$"),
)

#: A name carrying one of these markers in its first three tokens is a cell of
#: a predeclared multi-cell screening battery; the whole battery is one family
#: (same tokens-up-to-and-including the marker). Same convention as
#: ``findings_registry._battery_key``.
FAMILY_BATTERY_MARKERS = ("battery", "microstructure")

_CLOSING_RULE = (
    "AGENTS.md binding rule: an interval containing zero is NOT grounds for "
    "rejection. Only a resolved wrong sign, zero split-half reliability, or a "
    "positive control proven able to detect an effect that size may close a "
    "line of work; everything else is 'unresolved_below_power' and is recorded, "
    "not closed."
)

# Effects are always stored so that POSITIVE FAVOURS THE CANDIDATE, whatever the
# underlying metric's own polarity. Brier and MAE improve downward, so a caller
# recording those must negate before storing; the unit is kept for provenance.
#
# SCALES, pinned because two of the first entries were recorded on different
# ones and pooling them would have been silently meaningless:
#   ats_points      -- points of margin/line error, e.g. 0.174
#   accuracy_points -- PERCENTAGE POINTS of forced-pick accuracy, e.g. 1.10 for
#                      a 1.1-point gap. NOT a fraction: record 1.10, not 0.011.
#   brier, log_loss -- the raw metric difference, e.g. 0.0015. Ambiguous
#                      on its own: the module-wide "positive favours candidate"
#                      convention means the raw (candidate - baseline) natural
#                      difference has already been NEGATED before storage, but
#                      nothing in the unit name says so -- a pooler has to trust
#                      prose in ``notes`` to know a stored +0.0015 here means
#                      the candidate's Brier/log-loss was 0.0015 LOWER (better),
#                      not higher. Kept, unchanged, for the entries already
#                      recorded this way; prefer ``brier_improvement`` /
#                      ``log_loss_improvement`` below for new entries so the
#                      sign convention is legible from the unit name alone.
#   mae             -- points of mean absolute error, e.g. 0.013. Same
#                      ambiguity and the same fix: prefer ``mae_improvement``.
#   correlation     -- a Pearson (or equivalent) correlation coefficient,
#                      native range [-1, +1]. Positive = the predeclared
#                      candidate-favouring direction, exactly like every other
#                      unit here -- NOT "positive means positively
#                      correlated" independent of what direction was
#                      predeclared. Added 2026-09-01: a CFB entry had been
#                      forced into ``accuracy_points`` as a "numeric container
#                      only" for lack of a real unit.
#   mae_improvement, brier_improvement, log_loss_improvement
#                   -- the metric's own natural units, but HIGHER IS BETTER,
#                      unlike the bare ``mae``/``brier``/``log_loss`` units
#                      above: store (baseline_metric - candidate_metric)
#                      directly, with no extra negation, e.g. +0.00082 means
#                      the candidate's MAE was 0.00082 LOWER (better) than the
#                      baseline's. This is the SAME "positive favours
#                      candidate" convention every unit in this module already
#                      follows (see ``favours_candidate``); the point of these
#                      three units is only that the unit name itself now says
#                      "improvement", so a pooler never has to consult prose
#                      in ``notes`` to know which way is better. Added
#                      2026-09-01 after an NFL totals-residual entry had to be
#                      recorded under bare ``mae`` with the sign explained only
#                      in ``notes``.
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
)

#: Reader-facing taxonomy for the public Signal Ledger page: exactly one of
#: these per signal, optional so the pre-existing registry keeps loading
#: while a signal awaits classification. Fixed vocabulary -- adding a tenth
#: bucket means widening this tuple deliberately, not typing a new string at
#: record time.
#:   market      -- the betting market itself: line movement, cross-book
#:                  disagreement, public money, opener/close mechanics.
#:   onfield     -- on-field play: EPA, drives, pressure, personnel,
#:                  quarterbacks, special teams, penalties and officials.
#:   health      -- injuries, illness, availability, participation.
#:   schedule    -- rest, travel, body clock, revenge/divisional spots,
#:                  byes, the daylight-saving clock change.
#:   environment -- weather, surface, altitude, air quality, venue.
#:   attention   -- media volume, Wikipedia, Reddit, fantasy ADP, TV
#:                  audience.
#:   offfield    -- coaches, arrests, transactions, suspensions.
#:   modeling    -- the model's own settings, calibration, stacking, era
#:                  weighting -- not a claim about the games themselves.
#:   control     -- placebos, oracles, instrument checks, mirror nulls:
#:                  deliberately unplayable arms that exist to prove the
#:                  measuring tools work, not to be played.
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
    }
)


class WeakSignalError(ValueError):
    """Raised when the weak-signal ledger is invalid or a rule would be violated.

    A ``ValueError`` subclass so the CLI reports it as a user-facing error
    rather than a traceback, matching ``RegistryError`` and ``DataContractError``.
    """


@dataclass(frozen=True)
class WeakSignal:
    """One measured effect that was too small for its own test to resolve."""

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
    #: One or two plain-English sentences a football fan with no statistics
    #: background can read on its own -- naming the situation AND what the
    #: rule does about it. Optional so the pre-existing registry keeps
    #: loading; the Signal Ledger page falls back to ``description``,
    #: visibly marked as a raw technical description, when this is unset.
    plain_summary: str | None = None
    #: One of :data:`CATEGORIES`, or ``None`` while unclassified. Validated
    #: against the fixed vocabulary in :func:`signal_from_payload`.
    category: str | None = None
    status: str = "active"
    invalidated_reason: str | None = None
    superseded_by: str | None = None

    @property
    def favours_candidate(self) -> bool:
        """Which side of zero the point estimate fell on.

        The single most valuable field in the registry: precision accumulates
        slowly, but signs accumulate at one bit per experiment.
        """

        return self.effect > 0.0

    @property
    def season_range(self) -> range:
        return range(self.seasons[0], self.seasons[1] + 1)

    def resolved_standard_error(self) -> float | None:
        """The standard error, taken directly or recovered from the interval."""

        if self.standard_error is not None:
            return self.standard_error
        if self.interval is not None:
            low, high = self.interval
            width = float(high) - float(low)
            if width > 0.0:
                # A 95% interval spans 2 * 1.96 standard errors.
                return width / (2.0 * 1.959963984540054)
        return None


@dataclass(frozen=True)
class Registry:
    """The whole weak-signal ledger."""

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
    """Reject any terminal verdict that does not stand on an admissible ground.

    This is the code form of AGENTS.md's binding rule. It runs both when a
    signal is recorded and when the ledger is loaded, so a session that never
    read the prose rule still cannot write the violation — and the error it
    gets quotes the rule it was about to break.
    """

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
            assert reliability is not None  # narrowing for mypy; _require raised above
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
    """Reject a point estimate recorded outside its own interval.

    This is a recording contradiction, not a statistical property: whatever
    produced the row cannot have drawn an effect outside the interval it
    quotes alongside it. Enforced at RECORD time (and in ``record_signal``),
    deliberately NOT at load time -- one pre-existing ledger entry predates
    this check, and AGENTS.md forbids silently editing recorded measurements,
    so history loads unchanged while every new write is held to it. Use
    :func:`coherence_problems` to surface any historical rows at report time.
    """

    if interval is None:
        return
    low, high = interval
    _require(
        low <= effect <= high,
        f"Signal {name!r} records effect {effect} outside its own interval "
        f"[{low}, {high}] -- a recording contradiction; check the sign or the "
        "interval against the source artifact before recording",
    )


def coherence_problems(signals: Sequence[WeakSignal]) -> list[dict[str, Any]]:
    """Report signals whose point estimate sits outside their own interval.

    The load-time counterpart of :func:`validate_coherence`: historical rows
    are never rewritten (AGENTS.md preserves recorded measurements), so the
    contradiction is surfaced to the reader of ``status``/``pool`` output
    instead of blocking the ledger.
    """

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


def signal_from_payload(name: str, payload: dict[str, Any]) -> WeakSignal:
    unknown = sorted(set(payload).difference(_SIGNAL_FIELDS))
    _require(not unknown, f"Signal {name!r} has unknown fields: {', '.join(unknown)}")
    for field in ("recorded_at", "description", "source", "effect", "effect_units"):
        _require(field in payload, f"Signal {name!r} is missing {field!r}")
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
    _require(math.isfinite(effect), f"Signal {name!r} has a non-finite effect")

    interval_payload = payload.get("interval")
    interval: tuple[float, float] | None = None
    if interval_payload is not None:
        _require(
            isinstance(interval_payload, (list, tuple)) and len(interval_payload) == 2,
            f"Signal {name!r} needs interval as a two-element [low, high]",
        )
        assert isinstance(interval_payload, (list, tuple))
        low, high = float(interval_payload[0]), float(interval_payload[1])
        _require(low <= high, f"Signal {name!r} has an inverted interval")
        interval = (low, high)

    standard_error = payload.get("standard_error")
    if standard_error is not None:
        standard_error = float(standard_error)
        _require(standard_error > 0.0, f"Signal {name!r} has a non-positive standard_error")

    probability_positive = payload.get("probability_positive")
    if probability_positive is not None:
        probability_positive = float(probability_positive)
        _require(
            0.0 <= probability_positive <= 1.0,
            f"Signal {name!r} has probability_positive outside [0, 1]",
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


def registry_from_payload(payload: dict[str, Any]) -> Registry:
    unknown = sorted(set(payload).difference(_TOP_LEVEL_FIELDS))
    _require(not unknown, f"Ledger has unknown top-level fields: {', '.join(unknown)}")
    version = int(payload.get("version", 0))
    _require(
        version == WEAK_SIGNAL_REGISTRY_VERSION,
        f"Unsupported weak-signal registry version: {version}",
    )
    raw_signals = payload.get("signals", {})
    _require(isinstance(raw_signals, dict), "Ledger 'signals' must be an object")
    signals = {name: signal_from_payload(name, body) for name, body in raw_signals.items()}
    notes = tuple(str(note) for note in payload.get("notes", ()))
    return Registry(version=version, notes=notes, signals=signals)


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
    """Return the tracked registry path, honouring ``NFL_ATS_REGISTRY_DIR`` when
    ``root`` is not given explicitly -- matching ``rotation.default_registry_path``'s
    convention, so a caller that forgets to thread an explicit root still lands in
    whatever isolated directory a test (or ``NFL_ATS_REGISTRY_DIR``-aware caller)
    has already set up, rather than silently falling back to the real tracked
    ``registry/`` tree.
    """

    base = Path(os.environ.get("NFL_ATS_REGISTRY_DIR", "registry")) if root is None else root
    return base / WEAK_SIGNAL_REGISTRY_FILENAME


def load_registry(path: Path) -> Registry:
    if not path.is_file():
        return Registry(version=WEAK_SIGNAL_REGISTRY_VERSION, notes=(), signals={})
    payload = json.loads(path.read_text(encoding="utf-8"))
    return registry_from_payload(payload)


def save_registry(registry: Registry, path: Path) -> None:
    atomic_json(registry_to_payload(registry), path)


def record_signal(registry: Registry, signal: WeakSignal, *, replace: bool = False) -> Registry:
    """Add a measured signal to the ledger.

    Re-recording an existing name requires ``replace``: a silently overwritten
    effect would let a second look at the same signal masquerade as new
    evidence, which is exactly the accounting this registry exists to prevent.
    """

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
    # The CLI constructs WeakSignal directly, so the closure taxonomy must be
    # enforced here too, not only on load — record time is when the session
    # that is about to write an inadmissible verdict needs to hear about it.
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
    """Retain an invalid measurement for audit, without adjudicating its mechanism."""
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
    """Correct a mis-tagged ``effect_units`` on one already-recorded entry.

    Some entries had no unit that matched what was actually measured (a
    correlation coefficient, an MAE/Brier/log-loss *improvement*) and were
    forced into an existing unit with the true sign convention explained only
    in prose inside ``notes`` -- exactly the kind of note a pooler will not
    read before averaging. This changes ONLY ``effect_units`` and appends one
    audit line to ``notes``; every other field (effect, interval,
    classification, closing_ground, probability_positive, ...) is carried
    over byte-for-byte, because AGENTS.md forbids silently rewriting a
    recorded measurement and a unit correction is not a new measurement.
    """

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
    """Attach a measured split-half reliability to one already-recorded entry.

    Reliability is one of only two admissible closing grounds (AGENTS.md:
    "wrong sign, or the trait has no split-half reliability"), yet most
    entries carry ``reliability: null`` -- so the ground can be neither used
    nor ruled out. This fills that field from a measurement and NOTHING else:
    ``effect``, ``interval``, ``classification``, ``closing_ground``,
    ``probability_positive``, ``source`` and every other field are carried
    over byte-for-byte, exactly as :func:`retag_effect_units` does, because a
    reliability measurement is not a re-measurement of the effect and must
    never silently become one.

    The registry schema has no reliability-interval field, so the interval,
    the METHOD (which quantity was measured -- a trait's split-half
    correlation and a flag's exposure reliability are different quantities and
    must not be compared) and the artifact path are appended to ``notes`` as
    one audit line. Recording a number here does NOT reclassify anything: a
    low value is a *candidate* for the ``no_split_half_reliability`` ground,
    and closing on it stays a separate, explicit decision.
    """

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


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def _binomial_two_sided_p(favourable: int, total: int) -> float:
    """Exact two-sided binomial p-value against a fair coin."""

    if total <= 0:
        return 1.0

    def pmf(k: int) -> float:
        return math.comb(total, k) * 0.5**total

    observed = pmf(favourable)
    # Sum every outcome no more likely than the observed one (the standard
    # small-sample two-sided construction; symmetric here since p = 0.5).
    tolerance = 1e-12
    return min(1.0, sum(pmf(k) for k in range(total + 1) if pmf(k) <= observed + tolerance))


def sign_test(signals: Sequence[WeakSignal]) -> dict[str, Any]:
    """Do the point estimates lean one way more than chance allows?

    Precision accumulates slowly; signs accumulate at one bit per experiment.
    Under a true null each candidate is equally likely to land either side of
    zero, so a lopsided tally is testable even when no single result is.
    """

    excluded_invalidated = sum(s.status == "invalidated" for s in signals)
    signals = [s for s in signals if s.status != "invalidated"]
    favourable = sum(1 for signal in signals if signal.favours_candidate)
    total = len(signals)
    return {
        "signals": total,
        "excluded_invalidated": excluded_invalidated,
        "favouring_candidate": favourable,
        "favouring_baseline": total - favourable,
        "p_value": _binomial_two_sided_p(favourable, total),
        "interpretation": (
            "no signals recorded"
            if total == 0
            else (
                "directions are consistent with a coin flip"
                if _binomial_two_sided_p(favourable, total) > 0.10
                else "directions lean further than chance comfortably explains"
            )
        ),
    }


def pooled_effect(signals: Sequence[WeakSignal], *, method: str = "random") -> dict[str, Any]:
    """Inverse-variance pooled effect across signals sharing one unit.

    Fixed-effect pooling assumes every input estimates the SAME quantity, which
    is rarely true across different football signals, so ``random`` (the
    DerSimonian-Laird estimator) is the default: it inflates the variance by the
    observed between-signal heterogeneity rather than pretending it away.

    The payoff is the sqrt(K) shrinkage in the standard error, which is what
    makes a pile of individually invisible effects visible together.
    """

    _require(method in ("fixed", "random"), f"Unknown pooling method {method!r}")
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
    errors = [s.resolved_standard_error() for s in usable]
    variances = [float(se) ** 2 for se in errors if se is not None]

    weights = [1.0 / v for v in variances]
    total_weight = sum(weights)
    fixed_mean = sum(w * e for w, e in zip(weights, effects, strict=True)) / total_weight

    # DerSimonian-Laird between-signal variance.
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

    smallest_individual = min(float(se) for se in errors if se is not None)
    return {
        "signals": len(usable),
        "excluded_invalidated": excluded_invalidated,
        "method": method,
        "effect_units": usable[0].effect_units,
        "pooled_effect": mean,
        "standard_error": standard_error,
        "interval": (mean - half, mean + half),
        "excludes_zero": bool((mean - half) * (mean + half) > 0.0),
        "heterogeneity_tau_squared": tau_squared,
        "sharpening_vs_best_single": (
            None if standard_error == 0 else smallest_individual / standard_error
        ),
        "note": (
            "Pooling assumes the inputs are independent. Check overlap_warnings "
            "before believing this interval."
        ),
    }


def signal_family(signal: WeakSignal) -> str:
    """The measurement family a signal belongs to, for overlap accounting.

    An explicit ``family`` field wins. Otherwise the family is inferred from
    the name with two rules shared with ``findings_registry``'s duplication
    passes: decomposition suffixes (``_opener`` grades, ``_era_YYYY_YYYY`` and
    bare-year window splits, ``_preYYYY``/``_postYYYY`` splits) are stripped,
    and a battery marker in the first three tokens collapses the whole
    screening battery to its prefix. Inference is advisory and conservative:
    when it is unsure it keeps signals in separate families rather than
    merging measurements that might be distinct. Declare ``family`` explicitly
    at record time whenever the name alone does not capture the grouping.
    """

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


def family_overlap_warnings(signals: Sequence[WeakSignal]) -> dict[str, Any]:
    """Per-family overlap report for a pool of signals.

    The pairwise :func:`overlap_warnings` list grew past 55,000 entries once
    batteries started recording every cell on the same seasons
    (``docs/registry_correlation_audit_20260822.md``, risk #3) -- correct but
    unreadable, which is its own hazard: a warning nobody reads protects
    nothing. This groups the same information by measurement family (see
    :func:`signal_family`) as ``docs/registry_correlation_audit_20260822.md``
    §3's correlation map does: members of one family sharing seasons are
    correlated decompositions of the same football, not independent votes, so
    both the pooled interval and the sign test overstate precision by however
    much those members duplicate each other. Like everything in this module it
    reports rather than blocks; the honest use of a pooled estimate remains ONE
    predeclared combined look on untouched windows.
    """

    excluded_invalidated = sum(s.status == "invalidated" for s in signals)
    ordered = sorted((s for s in signals if s.status != "invalidated"), key=lambda s: s.name)
    families: dict[tuple[str, str], list[WeakSignal]] = {}
    for signal in ordered:
        families.setdefault((signal.league, signal_family(signal)), []).append(signal)

    pairwise_pairs = 0
    within_family: list[dict[str, Any]] = []
    for (league, family), members in sorted(families.items()):
        overlapping: list[WeakSignal] = []
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                low = max(first.seasons[0], second.seasons[0])
                high = min(first.seasons[1], second.seasons[1])
                if low <= high:
                    pairwise_pairs += 1
                    overlapping.extend((first, second))
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
    total_pairwise_pairs = 0
    for index, (league_a, _, low_a, high_a) in enumerate(family_spans):
        for league_b, _, low_b, high_b in family_spans[index + 1 :]:
            if league_a == league_b and max(low_a, low_b) <= min(high_a, high_b):
                cross_family_pairs += 1
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if first.league == second.league and max(first.seasons[0], second.seasons[0]) <= min(
                first.seasons[1], second.seasons[1]
            ):
                total_pairwise_pairs += 1

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
    """Flag pairs measured on overlapping seasons within the same league.

    Results from the same seasons share the same football, so their errors are
    correlated and pooling them overstates precision — the "pooling ten weak
    positives on the SAME window proves nothing" trap in
    ``docs/pool_edge_plan.md``. This does not block anything; it reports.
    """

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
    """Only genuinely unresolved signals are eligible to be pooled.

    A refuted mechanism and a control-bounded null are real negatives; folding
    either into a pool would launder a known failure into a fresh-looking
    positive.
    """

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
) -> dict[str, Any]:
    """Everything needed to decide whether the pile is worth one combined look."""

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
        # AGENTS.md: pooled inputs must be commensurable -- same units, same
        # scale, same POPULATION. NFL and CFB differ in market sharpness and in
        # evaluator resolution, so averaging them is not a finding, it is a
        # units error that happens to typecheck. This stayed latent while the
        # registry held only NFL signals; the first CFB entry would otherwise
        # have silently moved the headline pooled estimate.
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
            group + [s for s in invalidated if s.effect_units == unit], method=method
        )

    used_seasons = sorted({season for signal in eligible for season in signal.season_range})
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "eligible": [signal.name for signal in eligible],
        "excluded_with_reason": excluded,
        "excluded_invalidated": len(invalidated),
        "sign_test": sign_test(eligible + invalidated),
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
    """Build signals from raw dicts, each carrying its own ``name``."""

    built: list[WeakSignal] = []
    for entry in entries:
        payload = dict(entry)
        name = payload.pop("name", None)
        _require(bool(name), "Every entry needs a 'name'")
        built.append(signal_from_payload(str(name), payload))
    return built
