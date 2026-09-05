"""Card-level explanation contract (ENG-12).

A published pick is a lot of machinery collapsed into three cells of a
table: a team, a number, a decision score. This module builds the
DESCRIPTIVE record behind one pick -- the market line it was read against,
this game's own model probability, which overlays fired and what tripped
them, how fresh each input source was, and whether anything has changed
since the Tuesday card -- and renders it as one plain paragraph, without
ever implying causal certainty or a proven, profitable edge (AGENTS.md:
"Never describe the current historical forced-pick accuracy as proof of a
profitable or stable edge. Keep historical accuracy distinct from each
game's model probability.").

:func:`explain_pick` is the single entry point. It is deliberately
duck-typed and side-effect free: ``row`` is any ``Mapping`` with at least
``game_id``/``home_team``/``away_team``/``spread_line``/
``home_cover_probability`` (a plain ``dict``, a ``pandas.Series``, or one
record of a forecast/recommendations frame all satisfy it), ``lineage`` is
an optional :class:`nfl_ats.lineage.CardLineage` (ENG-16) supplying the
market line's source snapshot and capture instant, ``source_report`` is an
optional :class:`nfl_ats.source_freshness_policy.SourcePolicyReport`
(ENG-14) supplying per-source freshness, ``overlays`` is a pre-normalized
sequence of :class:`OverlayFiring` (build one with the adapters below from
whichever overlay result objects the caller already has), and
``refresh_changes`` is an optional :class:`RefreshChangeInput` (or an
equivalent mapping) describing the latest Tuesday-to-refresh delta for this
game.

Every field the caller does not supply degrades to an explicit ``no_data``
state -- never a guess, never silence -- matching the fail-open, explicit-
absence discipline :mod:`nfl_ats.lineage` and
:mod:`nfl_ats.source_freshness_policy` already established (a
``CardLineageEntry`` may carry ``lineage=None`` only with a stated
``reason``; an unobserved source is reported, never folded into "healthy").

Join point with ENG-18 (``nfl_ats.snapshot_diff``, Tuesday-vs-refresh diff)
----------------------------------------------------------------------------
That module did not exist yet when this one was built. When it lands, its
per-game refresh-change summary should be adapted into a
:class:`RefreshChangeInput` (or passed as an equivalent mapping) and handed
to ``refresh_changes`` directly. Until then, :func:`refresh_change_from_pick_revision`
reads `nfl_ats.pick_refresh`'s append-only pick-revision ledger directly --
the exact fallback this task's own instructions name -- so a real refresh
already shows up here without waiting on that module.

Language contract
------------------
:data:`LANGUAGE_CONTRACT` is a literal, case-insensitive substring blocklist
(not a semantic classifier); :func:`check_language` raises
:class:`LanguageContractError` when any phrase appears anywhere in the text,
including inside an otherwise-safe negation, so the template itself is
written to avoid every phrase outright rather than negate it.
:func:`explain_pick` runs this check on its own generated text before
returning, so a caller never receives a :class:`PickExplanation` whose
``text`` violates the contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Any

from nfl_ats.coach_fade_overlay import OverlayFlip
from nfl_ats.division_revenge_tilt_overlay import TiltFlip as DivisionRevengeFlip
from nfl_ats.four_overlay_composition import (
    COACH_FADE,
    DIVISION_REVENGE_TILT,
    PLAYER_ARRESTS_BACK_SIDE_POLICY,
    SPREAD_GAP_ZONE_FADE,
    FourOverlayCompositionResult,
)
from nfl_ats.lineage import FIELD_MARKET_LINE, CardLineage
from nfl_ats.player_arrests_back_side_overlay import ArrestFlip
from nfl_ats.source_freshness_policy import SourcePolicyReport
from nfl_ats.spread_gap_zone_fade_overlay import TiltFlip as SpreadGapFlip

CARD_EXPLANATION_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Provenance -- "label how you know it" (AGENTS.md), stated per component.
# ---------------------------------------------------------------------------

MEASURED_FROM_ARTIFACT = "measured_from_artifact"
COMPUTED_NOW = "computed_now"
NO_DATA = "no_data"
PROVENANCE_VALUES: tuple[str, ...] = (MEASURED_FROM_ARTIFACT, COMPUTED_NOW, NO_DATA)

# ---------------------------------------------------------------------------
# Language contract
# ---------------------------------------------------------------------------

#: Case-insensitive forbidden substrings. The template below is written to
#: avoid every one of these outright (never negated in a sentence), because
#: a substring check cannot distinguish "not profitable" from "profitable".
LANGUAGE_CONTRACT: tuple[str, ...] = (
    "will win",
    "lock",
    "guaranteed",
    "guarantee",
    "profitable",
    "edge proven",
    "because of",
    "caused",
    "beats the market",
    "sure thing",
    "can't lose",
    "cannot lose",
    "no risk",
    "risk-free",
)


class LanguageContractError(ValueError):
    """``text`` uses a phrase :data:`LANGUAGE_CONTRACT` forbids."""


def check_language(text: str) -> None:
    """Raise :class:`LanguageContractError` if ``text`` uses a forbidden phrase."""

    lowered = text.lower()
    violations = [phrase for phrase in LANGUAGE_CONTRACT if phrase in lowered]
    if violations:
        raise LanguageContractError(
            "text uses forbidden phrase(s): " + ", ".join(sorted(set(violations)))
        )


# ---------------------------------------------------------------------------
# Small conversions -- mirrors the discipline nfl_ats.lineage already uses.
# ---------------------------------------------------------------------------


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _pick_oriented_line(home_line: float, pick_is_home: bool) -> float:
    """The market line restated as the PICK's own handicap.

    Mirrors ``nfl_ats.publishing._published_card``'s own
    ``pick_line = (-spread_line) if home_pick else spread_line`` exactly,
    duplicated here (not imported) to avoid a publishing->card_explanation->
    publishing cycle -- the same "duplicate a tiny private formula rather
    than couple modules" discipline ``nfl_ats.board_content`` already uses
    for ``_CONFIDENCE_FILL``.
    """

    return -home_line if pick_is_home else home_line


def _format_line(value: float | None) -> str:
    if value is None:
        return "unknown"
    return "PK" if value == 0.0 else f"{value:+g}"


def _pick_side_and_probability(row: Mapping[str, Any]) -> tuple[str | None, float | None]:
    """The picked team and this game's model probability FOR that side.

    ``row`` may supply an explicit ``pick_team`` (e.g. an already-rendered
    card row); otherwise the pick is derived the same way
    ``publishing._published_card`` derives it: home when
    ``home_cover_probability >= 0.5``, away otherwise.
    """

    home_team = _optional_str(row.get("home_team"))
    away_team = _optional_str(row.get("away_team"))
    pick_team = _optional_str(row.get("pick_team"))
    home_probability = _finite_float(row.get("home_cover_probability"))
    if pick_team is None:
        if home_probability is None or home_team is None or away_team is None:
            return None, None
        pick_team = home_team if home_probability >= 0.5 else away_team
    if home_probability is None:
        return pick_team, None
    probability = home_probability if pick_team == home_team else 1.0 - home_probability
    return pick_team, probability


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketLineComponent:
    """The market line used for this pick: which snapshot, and when."""

    home_spread_line: float | None
    pick_spread_line: float | None
    snapshot_id: str | None
    snapshot_captured_at: str | None
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_spread_line": self.home_spread_line,
            "pick_spread_line": self.pick_spread_line,
            "snapshot_id": self.snapshot_id,
            "snapshot_captured_at": self.snapshot_captured_at,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MarketLineComponent:
        return cls(
            home_spread_line=_finite_float(payload.get("home_spread_line")),
            pick_spread_line=_finite_float(payload.get("pick_spread_line")),
            snapshot_id=_optional_str(payload.get("snapshot_id")),
            snapshot_captured_at=_optional_str(payload.get("snapshot_captured_at")),
            provenance=str(payload.get("provenance", NO_DATA)),
        )


@dataclass(frozen=True)
class ModelProbabilityComponent:
    """This game's own model probability for the pick side -- never an
    accuracy figure (AGENTS.md: keep historical accuracy distinct from each
    game's model probability)."""

    pick_side: str
    probability: float | None
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pick_side": self.pick_side,
            "probability": self.probability,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelProbabilityComponent:
        return cls(
            pick_side=str(payload.get("pick_side") or ""),
            probability=_finite_float(payload.get("probability")),
            provenance=str(payload.get("provenance", NO_DATA)),
        )


@dataclass(frozen=True)
class OverlayFiring:
    """One overlay that fired on this pick.

    Only FIRED overlays are represented (mirrors
    ``nfl_ats.lineage``'s own rule: "an overlay that did not fire changed
    nothing, so it has nothing to justify"). ``changed_pick`` is carried
    explicitly rather than implied, because the four-member production
    policy's OR/complement-once semantics mean every listed firing is
    independently sufficient to have caused the flip, even when another
    member also fired on the same game.
    """

    name: str
    direction: str
    input_value: str
    changed_pick: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "input_value": self.input_value,
            "changed_pick": self.changed_pick,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OverlayFiring:
        return cls(
            name=str(payload.get("name") or ""),
            direction=str(payload.get("direction") or ""),
            input_value=str(payload.get("input_value") or ""),
            changed_pick=bool(payload.get("changed_pick", True)),
        )


@dataclass(frozen=True)
class OverlaysComponent:
    firings: tuple[OverlayFiring, ...]
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "firings": [firing.to_dict() for firing in self.firings],
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OverlaysComponent:
        raw_firings = payload.get("firings") or []
        return cls(
            firings=tuple(OverlayFiring.from_dict(item) for item in raw_firings),
            provenance=str(payload.get("provenance", NO_DATA)),
        )


@dataclass(frozen=True)
class SourceFreshnessEntry:
    """One source's state as of this pick's explanation, per
    :mod:`nfl_ats.source_freshness_policy`'s three-state vocabulary plus this
    module's own ``no_data`` (a source the report never observed at all)."""

    source_id: str
    as_of: str | None
    state: str

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "as_of": self.as_of, "state": self.state}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceFreshnessEntry:
        return cls(
            source_id=str(payload.get("source_id") or ""),
            as_of=_optional_str(payload.get("as_of")),
            state=str(payload.get("state") or NO_DATA),
        )


@dataclass(frozen=True)
class FreshnessComponent:
    sources: tuple[SourceFreshnessEntry, ...]
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": [entry.to_dict() for entry in self.sources],
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FreshnessComponent:
        raw_sources = payload.get("sources") or []
        return cls(
            sources=tuple(SourceFreshnessEntry.from_dict(item) for item in raw_sources),
            provenance=str(payload.get("provenance", NO_DATA)),
        )


#: Tuesday-to-refresh change vocabulary (fixed shape, per ENG-12's DoD).
REFRESH_NONE = "none"
REFRESH_FLIPPED = "pick_flipped"
REFRESH_LINE_MOVED = "line_moved"
REFRESH_OVERLAY_CHANGED = "overlay_added_removed"
REFRESH_NOT_YET = "no_refresh_yet"
REFRESH_STATUSES: tuple[str, ...] = (
    REFRESH_NONE,
    REFRESH_FLIPPED,
    REFRESH_LINE_MOVED,
    REFRESH_OVERLAY_CHANGED,
    REFRESH_NOT_YET,
)


@dataclass(frozen=True)
class RefreshComponent:
    status: str
    detail: str
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "detail": self.detail, "provenance": self.provenance}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RefreshComponent:
        return cls(
            status=str(payload.get("status") or REFRESH_NOT_YET),
            detail=str(payload.get("detail") or ""),
            provenance=str(payload.get("provenance", NO_DATA)),
        )


@dataclass(frozen=True)
class RefreshChangeInput:
    """Generic, source-agnostic Tuesday-to-refresh delta for one game.

    Either ENG-18's ``snapshot_diff`` summary or
    :func:`refresh_change_from_pick_revision` (this module's own fallback
    reading `nfl_ats.pick_refresh`'s ledger) can populate one of these.
    """

    previous_pick_side: str | None = None
    new_pick_side: str | None = None
    movement_delta: float | None = None
    overlays_added: tuple[str, ...] = ()
    overlays_removed: tuple[str, ...] = ()
    note: str = ""


def _classify_refresh(change: RefreshChangeInput) -> tuple[str, str]:
    """Precedence: a pick flip is the most decision-relevant change, then a
    line move, then an overlay change, else no change at all."""

    if (
        change.previous_pick_side
        and change.new_pick_side
        and change.previous_pick_side != change.new_pick_side
    ):
        detail = f"pick moved from {change.previous_pick_side} to {change.new_pick_side}"
        if change.movement_delta is not None:
            detail += f"; captured line moved {change.movement_delta:+g} points"
        return REFRESH_FLIPPED, detail
    if change.movement_delta is not None and change.movement_delta != 0.0:
        return (
            REFRESH_LINE_MOVED,
            f"captured line moved {change.movement_delta:+g} points from the frozen line",
        )
    if change.overlays_added or change.overlays_removed:
        added = ", ".join(change.overlays_added) if change.overlays_added else "none"
        removed = ", ".join(change.overlays_removed) if change.overlays_removed else "none"
        return REFRESH_OVERLAY_CHANGED, f"overlays added: {added}; overlays removed: {removed}"
    note = f" ({change.note})" if change.note else ""
    return REFRESH_NONE, f"no change from the previous pick{note}"


def _refresh_component(change: RefreshChangeInput | Mapping[str, Any] | None) -> RefreshComponent:
    if change is None:
        return RefreshComponent(
            status=REFRESH_NOT_YET, detail="no refresh pass recorded yet", provenance=NO_DATA
        )
    normalized = (
        change
        if isinstance(change, RefreshChangeInput)
        else RefreshChangeInput(
            previous_pick_side=_optional_str(change.get("previous_pick_side")),
            new_pick_side=_optional_str(change.get("new_pick_side")),
            movement_delta=_finite_float(change.get("movement_delta")),
            overlays_added=tuple(change.get("overlays_added") or ()),
            overlays_removed=tuple(change.get("overlays_removed") or ()),
            note=str(change.get("note") or ""),
        )
    )
    status, detail = _classify_refresh(normalized)
    return RefreshComponent(status=status, detail=detail, provenance=MEASURED_FROM_ARTIFACT)


def refresh_change_from_pick_revision(
    revision: Mapping[str, Any] | None,
) -> RefreshChangeInput | None:
    """Adapt one row of ``nfl_ats.pick_refresh``'s append-only pick-revision
    ledger (``load_pick_revisions``) into a :class:`RefreshChangeInput`.

    This is the fallback path this task's own instructions name: ENG-18's
    ``nfl_ats.snapshot_diff`` did not exist yet when this module was built.
    When it lands, prefer its per-game summary directly; this function keeps
    working unchanged as the direct-ledger read either way.
    """

    if revision is None:
        return None
    return RefreshChangeInput(
        previous_pick_side=_optional_str(revision.get("previous_pick_side")),
        new_pick_side=_optional_str(revision.get("new_pick_side")),
        movement_delta=_finite_float(revision.get("movement_delta")),
        note=str(revision.get("movement_policy") or ""),
    )


# ---------------------------------------------------------------------------
# Overlay adapters -- from each overlay module's own flip record, or the
# generic four-overlay composition result, to one normalized OverlayFiring.
# ---------------------------------------------------------------------------

#: Plain-English member labels, duplicated (not imported) from
#: ``nfl_ats.board_content``'s own private ``_MEMBER_LABELS`` -- the same
#: "duplicate a tiny private mapping rather than reach into another
#: module's underscore-prefixed name" discipline that module already uses.
_MEMBER_LABELS: dict[str, str] = {
    COACH_FADE: "coach fade",
    DIVISION_REVENGE_TILT: "division revenge",
    PLAYER_ARRESTS_BACK_SIDE_POLICY: "player arrests",
    SPREAD_GAP_ZONE_FADE: "spread-gap zone",
}


def overlay_firing_from_coach_fade_flip(flip: OverlayFlip) -> OverlayFiring:
    return OverlayFiring(
        name=COACH_FADE,
        direction=f"complemented toward {flip.opponent_team}",
        input_value=f"year-1 head coach matchup: {flip.year_one_team} vs {flip.opponent_team}",
    )


def overlay_firing_from_arrest_flip(flip: ArrestFlip) -> OverlayFiring:
    return OverlayFiring(
        name=PLAYER_ARRESTS_BACK_SIDE_POLICY,
        direction=f"flipped to {flip.flipped_to_team}",
        input_value=f"recent-arrest signal on {flip.original_pick_team}'s side of {flip.matchup}",
    )


def overlay_firing_from_division_revenge_flip(flip: DivisionRevengeFlip) -> OverlayFiring:
    return OverlayFiring(
        name=DIVISION_REVENGE_TILT,
        direction=f"tilted toward {flip.revenge_team}",
        input_value=f"division revenge matchup: {flip.revenge_team} vs {flip.opponent_team}",
    )


def overlay_firing_from_spread_gap_flip(flip: SpreadGapFlip) -> OverlayFiring:
    return OverlayFiring(
        name=SPREAD_GAP_ZONE_FADE,
        direction=f"faded to {flip.flipped_to_team}",
        input_value=f"spread_line={flip.spread_line:+g} inside the spread-gap zone",
    )


def overlay_firings_from_composition(
    composition: FourOverlayCompositionResult, game_id: str
) -> tuple[OverlayFiring, ...]:
    """Generic fallback built from the composed policy's own provenance
    (member id plus raw/final probability), for members whose rich,
    team-level flip record (e.g. division revenge, spread-gap zone) is not
    separately available to the caller. Prefer the specific
    ``overlay_firing_from_*`` adapters above when the caller already holds
    the richer flip record (coach fade, player arrests)."""

    target = str(game_id)
    game_row = next((row for row in composition.games if row.game_id == target), None)
    if game_row is None:
        return ()
    firings = []
    for member_id in game_row.member_ids:
        label = _MEMBER_LABELS.get(member_id, member_id)
        toward_home = game_row.final_home_cover_probability > game_row.raw_home_cover_probability
        direction = "complemented toward home" if toward_home else "complemented toward away"
        input_value = (
            f"raw home-cover probability {game_row.raw_home_cover_probability:.3f} "
            f"complemented to {game_row.final_home_cover_probability:.3f}"
        )
        firings.append(OverlayFiring(name=label, direction=direction, input_value=input_value))
    return tuple(firings)


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------


def _freshness_component(source_report: SourcePolicyReport | None) -> FreshnessComponent:
    if source_report is None:
        return FreshnessComponent(sources=(), provenance=NO_DATA)
    evaluated_at = _parse_iso(source_report.evaluated_at_utc)
    entries = []
    for row in source_report.sources:
        as_of: str | None = None
        if evaluated_at is not None and row.age_minutes is not None:
            as_of = (evaluated_at - timedelta(minutes=row.age_minutes)).isoformat()
        entries.append(SourceFreshnessEntry(source_id=row.source_id, as_of=as_of, state=row.state))
    for source_id in source_report.unobserved:
        entries.append(SourceFreshnessEntry(source_id=source_id, as_of=None, state=NO_DATA))
    return FreshnessComponent(sources=tuple(entries), provenance=MEASURED_FROM_ARTIFACT)


# ---------------------------------------------------------------------------
# The fixed-shape record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PickExplanation:
    """One pick's fixed-shape, descriptive explanation."""

    game_id: str
    matchup: str
    market_line: MarketLineComponent
    model_probability: ModelProbabilityComponent
    overlays: OverlaysComponent
    freshness: FreshnessComponent
    refresh: RefreshComponent
    text: str
    schema_version: int = CARD_EXPLANATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "game_id": self.game_id,
            "matchup": self.matchup,
            "market_line": self.market_line.to_dict(),
            "model_probability": self.model_probability.to_dict(),
            "overlays": self.overlays.to_dict(),
            "freshness": self.freshness.to_dict(),
            "refresh": self.refresh.to_dict(),
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PickExplanation:
        return cls(
            game_id=str(payload.get("game_id") or ""),
            matchup=str(payload.get("matchup") or ""),
            market_line=MarketLineComponent.from_dict(payload.get("market_line") or {}),
            model_probability=ModelProbabilityComponent.from_dict(
                payload.get("model_probability") or {}
            ),
            overlays=OverlaysComponent.from_dict(payload.get("overlays") or {}),
            freshness=FreshnessComponent.from_dict(payload.get("freshness") or {}),
            refresh=RefreshComponent.from_dict(payload.get("refresh") or {}),
            text=str(payload.get("text") or ""),
            schema_version=int(payload.get("schema_version", CARD_EXPLANATION_SCHEMA_VERSION)),
        )


# ---------------------------------------------------------------------------
# Text template
# ---------------------------------------------------------------------------


def _render_text(
    matchup: str,
    market_line: MarketLineComponent,
    model_probability: ModelProbabilityComponent,
    overlays: OverlaysComponent,
    freshness: FreshnessComponent,
    refresh: RefreshComponent,
) -> str:
    sentences: list[str] = []

    if market_line.home_spread_line is None:
        sentences.append(f"{matchup}: no market line is recorded for this pick.")
    else:
        pick_text = _format_line(
            market_line.pick_spread_line
            if market_line.pick_spread_line is not None
            else market_line.home_spread_line
        )
        snapshot_clause = ""
        if market_line.snapshot_id:
            when = (
                f", captured {market_line.snapshot_captured_at}"
                if market_line.snapshot_captured_at
                else ""
            )
            snapshot_clause = f" (snapshot {market_line.snapshot_id}{when})"
        sentences.append(
            f"{matchup}: the market line used for this pick is {pick_text}{snapshot_clause}."
        )

    if model_probability.probability is None:
        sentences.append("No model probability is recorded for this pick.")
    else:
        side = model_probability.pick_side or "the picked side"
        sentences.append(
            f"The model's own probability for {side} to cover this game is "
            f"{model_probability.probability:.1%}; this is a single-game estimate, "
            "not the project's historical accuracy."
        )

    if overlays.firings:
        described = "; ".join(
            f"{firing.name} ({firing.direction}, triggered by {firing.input_value})"
            for firing in overlays.firings
        )
        plural = "" if len(overlays.firings) == 1 else "s"
        sentences.append(
            f"{len(overlays.firings)} overlay{plural} fired on this pick: {described}."
        )
    elif overlays.provenance == NO_DATA:
        sentences.append("Overlay evaluation was not supplied for this pick.")
    else:
        sentences.append("No overlay fired on this pick.")

    if not freshness.sources:
        sentences.append("Source freshness was not evaluated for this pick.")
    else:
        counts: dict[str, int] = {}
        for entry in freshness.sources:
            counts[entry.state] = counts.get(entry.state, 0) + 1
        parts = ", ".join(f"{count} {state}" for state, count in sorted(counts.items()))
        sentences.append(f"Source freshness at publish time: {parts}.")

    refresh_sentences = {
        REFRESH_NOT_YET: "No late-week refresh has run yet for this pick.",
        REFRESH_NONE: f"The latest refresh confirmed this pick with no change ({refresh.detail}).",
        REFRESH_FLIPPED: f"The latest refresh changed this pick: {refresh.detail}.",
        REFRESH_LINE_MOVED: f"The latest refresh recorded a market move: {refresh.detail}.",
        REFRESH_OVERLAY_CHANGED: (
            f"The latest refresh changed the fired overlays: {refresh.detail}."
        ),
    }
    sentences.append(
        refresh_sentences.get(refresh.status, "Refresh status is unrecorded for this pick.")
    )

    sentences.append("This is a descriptive research summary, not a wagering recommendation.")
    return " ".join(sentence for sentence in sentences if sentence)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def explain_pick(
    row: Mapping[str, Any],
    *,
    lineage: CardLineage | None = None,
    source_report: SourcePolicyReport | None = None,
    overlays: Sequence[OverlayFiring] | None = None,
    refresh_changes: RefreshChangeInput | Mapping[str, Any] | None = None,
) -> PickExplanation:
    """Build one pick's fixed-shape, descriptive explanation.

    See the module docstring for what ``row``/``lineage``/``source_report``/
    ``overlays``/``refresh_changes`` accept. Runs :func:`check_language` on
    its own generated text before returning -- a caller can never receive a
    :class:`PickExplanation` whose ``text`` violates the language contract.
    """

    game_id = str(row.get("game_id") or "")
    home_team = _optional_str(row.get("home_team")) or ""
    away_team = _optional_str(row.get("away_team")) or ""
    matchup = (
        f"{away_team} at {home_team}" if home_team and away_team else (game_id or "Unknown matchup")
    )

    pick_team, probability = _pick_side_and_probability(row)
    home_line = _finite_float(row.get("spread_line"))
    pick_is_home = pick_team is not None and home_team != "" and pick_team == home_team
    pick_line = None if home_line is None else _pick_oriented_line(home_line, pick_is_home)

    market_lineage_record = None
    if lineage is not None:
        entry = lineage.field(FIELD_MARKET_LINE)
        market_lineage_record = entry.lineage if entry is not None else None
    market_line = MarketLineComponent(
        home_spread_line=home_line,
        pick_spread_line=pick_line,
        snapshot_id=market_lineage_record.source_snapshot if market_lineage_record else None,
        snapshot_captured_at=(
            market_lineage_record.source_captured_at if market_lineage_record else None
        ),
        provenance=MEASURED_FROM_ARTIFACT if home_line is not None else NO_DATA,
    )

    model_probability = ModelProbabilityComponent(
        pick_side=pick_team or "",
        probability=probability,
        provenance=COMPUTED_NOW if probability is not None else NO_DATA,
    )

    overlays_component = OverlaysComponent(
        firings=tuple(overlays) if overlays is not None else (),
        provenance=MEASURED_FROM_ARTIFACT if overlays is not None else NO_DATA,
    )

    freshness_component = _freshness_component(source_report)
    refresh_component = _refresh_component(refresh_changes)

    text = _render_text(
        matchup,
        market_line,
        model_probability,
        overlays_component,
        freshness_component,
        refresh_component,
    )
    check_language(text)

    return PickExplanation(
        game_id=game_id,
        matchup=matchup,
        market_line=market_line,
        model_probability=model_probability,
        overlays=overlays_component,
        freshness=freshness_component,
        refresh=refresh_component,
        text=text,
    )


def explain_card(
    rows: Sequence[Mapping[str, Any]],
    *,
    lineage: CardLineage | None = None,
    source_report: SourcePolicyReport | None = None,
    overlays_by_game: Mapping[str, Sequence[OverlayFiring]] | None = None,
    refresh_changes_by_game: Mapping[str, RefreshChangeInput | Mapping[str, Any]] | None = None,
) -> list[PickExplanation]:
    """:func:`explain_pick` for every row of a card, keyed by ``game_id``."""

    overlays_map = overlays_by_game or {}
    refresh_map = refresh_changes_by_game or {}
    explanations = []
    for row in rows:
        game_id = str(row.get("game_id") or "")
        explanations.append(
            explain_pick(
                row,
                lineage=lineage,
                source_report=source_report,
                overlays=overlays_map.get(game_id),
                refresh_changes=refresh_map.get(game_id),
            )
        )
    return explanations


def explanations_to_dict(explanations: Sequence[PickExplanation]) -> dict[str, Any]:
    return {
        "schema_version": CARD_EXPLANATION_SCHEMA_VERSION,
        "count": len(explanations),
        "explanations": [explanation.to_dict() for explanation in explanations],
    }


def to_json(explanations: Sequence[PickExplanation], *, indent: int = 2) -> str:
    return json.dumps(explanations_to_dict(explanations), indent=indent) + "\n"


def from_dict(payload: Mapping[str, Any]) -> list[PickExplanation]:
    return [PickExplanation.from_dict(item) for item in payload.get("explanations", [])]


def from_json(payload: str | bytes) -> list[PickExplanation]:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    return from_dict(json.loads(text))


def render_markdown(explanations: Sequence[PickExplanation]) -> str:
    """One short line per pick, for an additive Markdown card section."""

    lines = ["### Pick explanations", ""]
    for explanation in explanations:
        lines.append(f"- **{explanation.matchup}**: {explanation.text}")
    return "\n".join(lines) + "\n"


__all__ = [
    "CARD_EXPLANATION_SCHEMA_VERSION",
    "COMPUTED_NOW",
    "LANGUAGE_CONTRACT",
    "MEASURED_FROM_ARTIFACT",
    "NO_DATA",
    "PROVENANCE_VALUES",
    "REFRESH_FLIPPED",
    "REFRESH_LINE_MOVED",
    "REFRESH_NONE",
    "REFRESH_NOT_YET",
    "REFRESH_OVERLAY_CHANGED",
    "REFRESH_STATUSES",
    "FreshnessComponent",
    "LanguageContractError",
    "MarketLineComponent",
    "ModelProbabilityComponent",
    "OverlayFiring",
    "OverlaysComponent",
    "PickExplanation",
    "RefreshChangeInput",
    "RefreshComponent",
    "SourceFreshnessEntry",
    "check_language",
    "explain_card",
    "explain_pick",
    "explanations_to_dict",
    "from_dict",
    "from_json",
    "overlay_firing_from_arrest_flip",
    "overlay_firing_from_coach_fade_flip",
    "overlay_firing_from_division_revenge_flip",
    "overlay_firing_from_spread_gap_flip",
    "overlay_firings_from_composition",
    "refresh_change_from_pick_revision",
    "render_markdown",
    "to_json",
]
