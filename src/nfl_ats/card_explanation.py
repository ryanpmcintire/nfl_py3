from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Any

from nfl_ats.coach_fade_overlay import OverlayFlip
from nfl_ats.displayed_confidence import displayed_pick_probability, displayed_strength_word
from nfl_ats.division_revenge_tilt_overlay import TiltFlip as DivisionRevengeFlip
from nfl_ats.four_overlay_composition import (
    COACH_FADE,
    DIVISION_REVENGE_TILT,
    PLAYER_ARRESTS_BACK_SIDE_POLICY,
    SPREAD_GAP_ZONE_FADE,
    FourOverlayCompositionResult,
)
from nfl_ats.key_line_pick_read import is_half_point_line
from nfl_ats.lineage import FIELD_MARKET_LINE, CardLineage
from nfl_ats.market_decomposition import GameExplanation, explain_game_structured
from nfl_ats.player_arrests_back_side_overlay import ArrestFlip
from nfl_ats.source_freshness_policy import SourcePolicyReport
from nfl_ats.spread_gap_zone_fade_overlay import TiltFlip as SpreadGapFlip

CARD_EXPLANATION_SCHEMA_VERSION = 1


MEASURED_FROM_ARTIFACT = "measured_from_artifact"
COMPUTED_NOW = "computed_now"
NO_DATA = "no_data"
PROVENANCE_VALUES: tuple[str, ...] = (MEASURED_FROM_ARTIFACT, COMPUTED_NOW, NO_DATA)


BANNED_BOILERPLATE: tuple[str, ...] = (
    "wagering recommendation",
    "descriptive research summary",
    "research preview",
    "not proof of a profitable",
    "not proof of a stable",
    "gambling problem",
    "1-800-gambler",
)

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
    *BANNED_BOILERPLATE,
)

_SNAPSHOT_ID_RE = re.compile(r"\d{8}T\d{6}Z")
_ISO_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
_SHA_LIKE_RE = re.compile(r"\b[0-9a-f]{12,}\b", re.IGNORECASE)
_BANNED_PLUMBING_WORDS: tuple[str, ...] = ("snapshot", "artifact", "lineage")


class LanguageContractError(ValueError):
    pass


def check_language(text: str) -> None:

    if _SNAPSHOT_ID_RE.search(text):
        raise LanguageContractError("text contains a snapshot id (YYYYMMDDTHHMMSSZ pattern)")
    if _ISO_TIMESTAMP_RE.search(text):
        raise LanguageContractError("text contains an ISO timestamp")
    if _SHA_LIKE_RE.search(text):
        raise LanguageContractError("text contains a sha-like hex token")
    lowered = text.lower()
    plumbing_violations = [word for word in _BANNED_PLUMBING_WORDS if word in lowered]
    if plumbing_violations:
        raise LanguageContractError(
            "text uses banned plumbing word(s): " + ", ".join(sorted(plumbing_violations))
        )
    violations = [phrase for phrase in LANGUAGE_CONTRACT if phrase in lowered]
    if violations:
        raise LanguageContractError(
            "text uses forbidden phrase(s): " + ", ".join(sorted(set(violations)))
        )


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

    return -home_line if pick_is_home else home_line


def _format_line(value: float | None) -> str:
    if value is None:
        return "unknown"
    return "PK" if value == 0.0 else f"{value:+g}"


def _pick_side_and_probability(row: Mapping[str, Any]) -> tuple[str | None, float | None]:

    home_team = _optional_str(row.get("home_team"))
    away_team = _optional_str(row.get("away_team"))
    pick_team = _optional_str(row.get("pick_team"))
    home_probability = _finite_float(row.get("home_cover_probability"))
    if pick_team is None:
        if home_probability is None or home_team is None or away_team is None:
            return None, None
        pick_team = home_team if home_probability >= 0.5 else away_team
    displayed = displayed_pick_probability(row)
    if displayed is not None:
        return pick_team, displayed
    if home_probability is None:
        return pick_team, None
    probability = home_probability if pick_team == home_team else 1.0 - home_probability
    return pick_team, probability


@dataclass(frozen=True)
class MarketLineComponent:
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
    pick_side: str
    probability: float | None
    provenance: str
    stated_probability: float | None = None
    strength_word: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pick_side": self.pick_side,
            "probability": self.probability,
            "stated_probability": self.stated_probability,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelProbabilityComponent:
        return cls(
            pick_side=str(payload.get("pick_side") or ""),
            probability=_finite_float(payload.get("probability")),
            provenance=str(payload.get("provenance", NO_DATA)),
            stated_probability=_finite_float(payload.get("stated_probability")),
        )


@dataclass(frozen=True)
class OverlayFiring:
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
    previous_pick_side: str | None = None
    new_pick_side: str | None = None
    movement_delta: float | None = None
    overlays_added: tuple[str, ...] = ()
    overlays_removed: tuple[str, ...] = ()
    note: str = ""


def _classify_refresh(change: RefreshChangeInput) -> tuple[str, str]:

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

    if revision is None:
        return None
    return RefreshChangeInput(
        previous_pick_side=_optional_str(revision.get("previous_pick_side")),
        new_pick_side=_optional_str(revision.get("new_pick_side")),
        movement_delta=_finite_float(revision.get("movement_delta")),
        note=str(revision.get("movement_policy") or ""),
    )


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


@dataclass(frozen=True)
class PickExplanation:
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


_CONFIDENCE_PHRASES: dict[str, str] = {
    "slight": "a slight lean",
    "lean": "a lean",
    "strong": "a strong lean",
}

_FRESHNESS_FAN_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("odds", "lines"),
    ("injur", "injuries"),
    ("weather", "weather"),
)


def _fan_category(source_id: str) -> str | None:
    lowered = source_id.lower()
    for needle, label in _FRESHNESS_FAN_CATEGORIES:
        if needle in lowered:
            return label
    return None


def _join_two_or_three(phrases: Sequence[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, and {phrases[-1]}"


def _lead_sentence(
    market_line: MarketLineComponent, model_probability: ModelProbabilityComponent, matchup: str
) -> str:
    if market_line.home_spread_line is None:
        lead = f"{matchup}: no market line is recorded for this pick."
        if model_probability.probability is None:
            return f"{lead} No model probability is recorded for this pick."
        return lead
    pick_text = _format_line(
        market_line.pick_spread_line
        if market_line.pick_spread_line is not None
        else market_line.home_spread_line
    )
    pick_side = model_probability.pick_side or ""
    label = f"{pick_side} {pick_text}".strip() or matchup
    lead = f"{label}."
    if model_probability.probability is None:
        return f"{lead} No model probability is recorded for this pick."
    phrase = _CONFIDENCE_PHRASES.get(model_probability.strength_word or "", "")
    adjusted = (
        model_probability.stated_probability is not None
        and abs(model_probability.stated_probability - model_probability.probability) >= 0.0005
    )
    basis = " once its record on spreads this size is counted in" if adjusted else ""
    tail = f", {phrase}." if phrase else "."
    return (
        f"{lead} The model gives {pick_side or 'the pick'} a "
        f"{model_probability.probability:.1%} chance to cover{basis}{tail}"
    )


def _what_tips_it_sentence(game_explanation: GameExplanation | None) -> str:

    if game_explanation is None:
        return ""
    if not game_explanation.drivers:
        return "The model and the market largely agree here."
    phrases = _join_two_or_three([driver.phrase for driver in game_explanation.drivers[:2]])
    sentence = f"What tips it: {phrases} favour {game_explanation.pick_team}"
    if game_explanation.offsets:
        sentence += f"; {game_explanation.offsets[0].phrase} pulls the other way"
    return sentence + "."


def _situational_adjustment_sentence(overlays: OverlaysComponent) -> str:
    if overlays.firings:
        names = _join_two_or_three(
            [
                _MEMBER_LABELS.get(firing.name, firing.name.replace("_", " "))
                for firing in overlays.firings
            ]
        )
        return f"Situational adjustment fired: {names}."
    if overlays.provenance == NO_DATA:
        return ""
    return "No situational adjustment fired."


def _freshness_clause(freshness: FreshnessComponent) -> str:
    if not freshness.sources:
        return ""
    fan_state: dict[str, str] = {}
    other_stale = 0
    for entry in freshness.sources:
        category = _fan_category(entry.source_id)
        if category is None:
            if entry.state != "complete":
                other_stale += 1
            continue
        rank = {"complete": 0, "degraded": 1, NO_DATA: 1, "blocked": 2}
        if category not in fan_state or rank.get(entry.state, 1) > rank.get(fan_state[category], 0):
            fan_state[category] = entry.state
    fresh = sorted(category for category, state in fan_state.items() if state == "complete")
    stale = sorted(category for category, state in fan_state.items() if state != "complete")
    parts: list[str] = []
    if fresh:
        verb = "was" if len(fresh) == 1 else "were"
        parts.append(f"{_join_two_or_three(fresh)} {verb} current at publish")
    if stale:
        verb = "was" if len(stale) == 1 else "were"
        parts.append(f"{_join_two_or_three(stale)} {verb} a bit stale")
    if other_stale:
        plural = "" if other_stale == 1 else "s"
        verb = "was" if other_stale == 1 else "were"
        parts.append(f"{other_stale} other source{plural} {verb} a bit stale")
    if not parts:
        return ""
    joined = "; ".join(parts)
    return joined[:1].upper() + joined[1:] + "."


_REFRESH_SHORT_CLAUSES: dict[str, str] = {
    REFRESH_NOT_YET: "Not refreshed since Tuesday.",
    REFRESH_NONE: "Confirmed unchanged on refresh.",
    REFRESH_FLIPPED: "Pick changed on refresh.",
    REFRESH_LINE_MOVED: "Line moved on refresh.",
    REFRESH_OVERLAY_CHANGED: "Situational adjustments changed on refresh.",
}


def _render_text(
    matchup: str,
    market_line: MarketLineComponent,
    model_probability: ModelProbabilityComponent,
    overlays: OverlaysComponent,
    freshness: FreshnessComponent,
    refresh: RefreshComponent,
    game_explanation: GameExplanation | None,
    push_probability: float | None = None,
    key_line_read: bool = False,
) -> str:

    sentences = [
        _lead_sentence(market_line, model_probability, matchup),
        (
            _no_push_sentence(market_line.home_spread_line)
            if is_half_point_line(market_line.home_spread_line)
            else (
                _key_line_sentence(market_line.home_spread_line, push_probability)
                if key_line_read
                else _push_sentence(market_line.home_spread_line, push_probability)
            )
        ),
        _what_tips_it_sentence(game_explanation),
        _situational_adjustment_sentence(overlays),
        _freshness_clause(freshness),
        _REFRESH_SHORT_CLAUSES.get(refresh.status, ""),
    ]
    return " ".join(sentence for sentence in sentences if sentence)


_KEY_NUMBER_LINES: frozenset[int] = frozenset({3, 7, 10, 14})


def _no_push_sentence(home_spread_line: float | None) -> str:

    if home_spread_line is None or not is_half_point_line(home_spread_line):
        return ""
    return (
        "The line is a half point, so the game cannot finish exactly on it: there are no ties "
        "here, and one side or the other covers."
    )


def _push_sentence(home_spread_line: float | None, push_probability: float | None) -> str:

    if home_spread_line is None or push_probability is None:
        return ""
    if not float(home_spread_line).is_integer() or push_probability <= 0.0:
        return ""
    per_hundred = round(push_probability * 100.0)
    if per_hundred < 1:
        return ""
    number = abs(int(home_spread_line))
    where = (
        f"right on {number}, a number games land on a lot"
        if number in _KEY_NUMBER_LINES
        else "on a whole number"
    )
    return (
        f"The line sits {where}: about {per_hundred} in 100 games like this finish exactly "
        "there, a push, and that chance is counted here."
    )


def _key_line_sentence(home_spread_line: float | None, push_probability: float | None) -> str:

    if home_spread_line is None or not float(home_spread_line).is_integer():
        return ""
    number = abs(int(home_spread_line))
    lead = (
        f"The line sits right on {number}, a number games land on a lot, so this pick is "
        "read off how games with lines like this actually finished"
    )
    per_hundred = round(push_probability * 100.0) if push_probability else 0
    if per_hundred >= 1:
        return (
            f"{lead}, and the roughly {per_hundred} in 100 that ended exactly there, a push, "
            "are counted in that chance."
        )
    return f"{lead}."


def family_contributions_from_waterfall_entry(
    entry: Mapping[str, Any] | None,
) -> dict[str, float] | None:

    if not isinstance(entry, Mapping):
        return None
    steps = entry.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    totals: dict[str, float] = {}
    for step in steps:
        if not isinstance(step, Mapping) or step.get("kind") != "family":
            continue
        family = step.get("family")
        if not isinstance(family, str) or not family:
            continue
        value = _finite_float(step.get("delta_points"))
        if value is None:
            continue
        totals[family] = totals.get(family, 0.0) + value
    return totals or None


def _game_explanation_from_contributions(
    game_id: str, home_team: str, away_team: str, family_contributions: Mapping[str, float] | None
) -> GameExplanation | None:
    if not family_contributions or not home_team or not away_team:
        return None
    predicted_residual = sum(family_contributions.values())
    return explain_game_structured(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        predicted_residual=predicted_residual,
        family_contributions=family_contributions,
    )


def explain_pick(
    row: Mapping[str, Any],
    *,
    lineage: CardLineage | None = None,
    source_report: SourcePolicyReport | None = None,
    overlays: Sequence[OverlayFiring] | None = None,
    refresh_changes: RefreshChangeInput | Mapping[str, Any] | None = None,
    waterfall_entry: Mapping[str, Any] | None = None,
    key_line_read: bool = False,
) -> PickExplanation:

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

    home_probability = _finite_float(row.get("home_cover_probability"))
    stated = (
        None
        if home_probability is None
        else (home_probability if pick_is_home else 1.0 - home_probability)
    )
    model_probability = ModelProbabilityComponent(
        pick_side=pick_team or "",
        probability=probability,
        provenance=COMPUTED_NOW if probability is not None else NO_DATA,
        stated_probability=stated,
        strength_word=displayed_strength_word(row),
    )

    overlays_component = OverlaysComponent(
        firings=tuple(overlays) if overlays is not None else (),
        provenance=MEASURED_FROM_ARTIFACT if overlays is not None else NO_DATA,
    )

    freshness_component = _freshness_component(source_report)
    refresh_component = _refresh_component(refresh_changes)
    game_explanation = _game_explanation_from_contributions(
        game_id,
        home_team,
        away_team,
        family_contributions_from_waterfall_entry(waterfall_entry),
    )

    text = _render_text(
        matchup,
        market_line,
        model_probability,
        overlays_component,
        freshness_component,
        refresh_component,
        game_explanation,
        _finite_float(row.get("push_probability")),
        key_line_read=bool(key_line_read),
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
    waterfall_by_game: Mapping[str, Mapping[str, Any]] | None = None,
    key_line_games: Iterable[str] | None = None,
) -> list[PickExplanation]:

    overlays_map = overlays_by_game or {}
    refresh_map = refresh_changes_by_game or {}
    waterfall_map = waterfall_by_game or {}
    key_line_set = {str(game_id) for game_id in (key_line_games or ())}
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
                waterfall_entry=waterfall_map.get(game_id),
                key_line_read=game_id in key_line_set,
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

    lines = ["### Pick explanations", ""]
    for explanation in explanations:
        lines.append(f"- **{explanation.matchup}**: {explanation.text}")
    return "\n".join(lines) + "\n"


__all__ = [
    "BANNED_BOILERPLATE",
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
    "family_contributions_from_waterfall_entry",
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
