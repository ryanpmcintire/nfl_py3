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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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

# Effects are always stored so that POSITIVE FAVOURS THE CANDIDATE, whatever the
# underlying metric's own polarity. Brier and MAE improve downward, so a caller
# recording those must negate before storing; the unit is kept for provenance.
#
# SCALES, pinned because two of the first entries were recorded on different
# ones and pooling them would have been silently meaningless:
#   ats_points      -- points of margin/line error, e.g. 0.174
#   accuracy_points -- PERCENTAGE POINTS of forced-pick accuracy, e.g. 1.10 for
#                      a 1.1-point gap. NOT a fraction: record 1.10, not 0.011.
#   brier, log_loss -- the raw metric difference, e.g. 0.0015
#   mae             -- points of mean absolute error, e.g. 0.013
EFFECT_UNITS = ("ats_points", "accuracy_points", "brier", "log_loss", "mae")

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
        "reliability",
        "notes",
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
    reliability: float | None = None
    notes: str = ""

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


def signal_from_payload(name: str, payload: dict[str, Any]) -> WeakSignal:
    unknown = sorted(set(payload).difference(_SIGNAL_FIELDS))
    _require(not unknown, f"Signal {name!r} has unknown fields: {', '.join(unknown)}")
    for field in ("recorded_at", "description", "source", "effect", "effect_units"):
        _require(field in payload, f"Signal {name!r} is missing {field!r}")
    classification = payload.get("classification")
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
        classification_evidence=str(payload.get("classification_evidence", "")),
        reliability=(None if payload.get("reliability") is None else float(payload["reliability"])),
        notes=str(payload.get("notes", "")),
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
        signals[name] = body
    return {
        "version": registry.version,
        "notes": list(registry.notes),
        "signals": signals,
    }


def default_registry_path(root: Path | None = None) -> Path:
    base = Path("registry") if root is None else root
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

    if signal.name in registry.signals and not replace:
        raise WeakSignalError(
            f"Signal {signal.name!r} is already recorded; pass replace=True to correct it"
        )
    signals = dict(registry.signals)
    signals[signal.name] = signal
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

    favourable = sum(1 for signal in signals if signal.favours_candidate)
    total = len(signals)
    return {
        "signals": total,
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
    usable = [s for s in signals if s.resolved_standard_error() is not None]
    if not usable:
        return {
            "signals": 0,
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


def overlap_warnings(signals: Sequence[WeakSignal]) -> list[str]:
    """Flag pairs measured on overlapping seasons within the same league.

    Results from the same seasons share the same football, so their errors are
    correlated and pooling them overstates precision — the "pooling ten weak
    positives on the SAME window proves nothing" trap in
    ``docs/pool_edge_plan.md``. This does not block anything; it reports.
    """

    warnings: list[str] = []
    ordered = sorted(signals, key=lambda s: s.name)
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
        if signal.classification != POOLABLE_CLASSIFICATION:
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
        name: signal.classification
        for name, signal in sorted(registry.signals.items())
        if signal.classification != POOLABLE_CLASSIFICATION
    }
    unit_groups: dict[str, list[WeakSignal]] = {}
    for signal in eligible:
        unit_groups.setdefault(signal.effect_units, []).append(signal)

    pooled: dict[str, Any] = {}
    for unit, group in sorted(unit_groups.items()):
        pooled[unit] = pooled_effect(group, method=method)

    used_seasons = sorted({season for signal in eligible for season in signal.season_range})
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "eligible": [signal.name for signal in eligible],
        "excluded_with_reason": excluded,
        "sign_test": sign_test(eligible),
        "pooled_by_unit": pooled,
        "overlap_warnings": overlap_warnings(eligible),
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
