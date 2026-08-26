"""Frozen four-member overlay composition for a prospective played policy.

Every member is evaluated independently against the same raw incoming card.
The composed policy then takes the union of their flip game IDs and complements
the raw ``home_cover_probability`` exactly once on each union member.  An
overlap therefore coalesces; it never toggles a pick twice.  This is the joint
OR semantics used by the overlay-subset study, expressed here as a reusable,
deterministic production primitive.

The pure :func:`apply_four_overlay_composition` function accepts already-loaded
frames and a validated arrest snapshot descriptor.  The publication boundary
:func:`apply_four_overlay_composition_for_publication` obtains that descriptor
through the existing freshness/hash verifier and deliberately has no fail-open
path for missing, incomplete, corrupt, future-dated, or stale arrest data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.coach_fade_overlay import OVERLAY_WEEK_MAX, apply_coach_fade_overlay
from nfl_ats.data import DataContractError
from nfl_ats.division_revenge_tilt_overlay import apply_division_revenge_tilt_overlay
from nfl_ats.player_arrests_back_side_overlay import (
    MAX_SNAPSHOT_AGE,
    WINDOW_DAYS,
    ArrestSnapshot,
    apply_player_arrests_back_side_overlay,
    load_latest_complete_arrest_snapshot,
)
from nfl_ats.spread_gap_zone_fade_overlay import (
    SPREAD_GAP_LOWER_BOUND,
    SPREAD_GAP_UPPER_BOUND,
    apply_spread_gap_zone_fade_overlay,
)

POLICY_ID = "overlay_union_coach_division_revenge_player_arrests_spread_gap_v1"
INCUMBENT_CHALLENGER_ID = "overlay_production_chain_coach_arrest_incumbent"

COACH_FADE = "coach_fade"
DIVISION_REVENGE_TILT = "division_revenge_tilt"
PLAYER_ARRESTS_BACK_SIDE_POLICY = "player_arrests_back_side_policy"
SPREAD_GAP_ZONE_FADE = "spread_gap_zone_fade"

# This order is part of the policy identity and provenance.  It controls
# deterministic evaluation/reporting order, not pick precedence: all four
# members see the raw card and the final transform is their joint OR.
COMPOSITION_ORDER = (
    COACH_FADE,
    DIVISION_REVENGE_TILT,
    PLAYER_ARRESTS_BACK_SIDE_POLICY,
    SPREAD_GAP_ZONE_FADE,
)

# ---------------------------------------------------------------------------
# Weak-signal registry evidence per member (owner-directed, 2026-08-26)
#
# The obvious place to read this link was ``artifacts/prospective/
# challengers.json``'s ``evidence.registry_source`` field -- the exact
# mechanism ``nfl_ats.model_ledger`` already uses to link a challenger to its
# registry evidence. Measured against the live artifact: it is ``None`` on
# EVERY player-arrests-related entry (the promoted overlay, its retired
# coach-only incumbent, the union policy itself, and its own incumbent), not
# just the one this module originally flagged as unresolved. So that
# mechanism resolves nothing for this policy, and the right fix is to stop
# depending on it here: an artifact field that is null everywhere is exactly
# how the public Signal Ledger page (nfl_ats.signal_ledger, the one consumer
# of this mapping) shipped with a permanently-empty "On the card" filter.
#
# This mapping lives in CODE, next to the policy it describes, so it is
# version-controlled and reviewed the same way the policy itself is --
# never re-derived from an artifact field that has already been observed to
# go silently empty.
#
#   coach_fade -> hc_year_one_fade
#     Declared: challengers.json's hc_year_one_fade_overlay entry cites
#     "registry/weak_signals.json:hc_year_one_fade" directly.
#
#   division_revenge_tilt -> bias_battery_division_revenge_game AND
#   bias_battery_division_revenge_game_opener (both grades, deliberately not
#   collapsed to one)
#     Declared: challengers.json's division_revenge_tilt_overlay entry cites
#     BOTH registry rows as its evidence -- the same two-entry evidence list
#     nfl_ats.model_ledger already treats as one challenger's full evidence,
#     not as competing claims to pick between. Picking only the opener grade
#     would be a new judgment call this codebase does not already make.
#
#   player_arrests_back_side_policy -> player_arrests_recent_14d_back_side_policy_opener
#     NOT declared anywhere (registry_source is null on every arrest-related
#     challenger entry, as above). Verified instead by an exact NUMBER MATCH
#     against HANDOFF.md's promoted player-arrest policy component line
#     ("53.76% vs 53.36% on 1,503 games; +0.399 accuracy points;
#     probability_positive=0.8562", read 2026-08-26): the registry row
#     records effect=0.3992015968, probability_positive=0.8562,
#     seasons=[2020, 2025], sample_games=1503 -- an exact match on every
#     figure, not a fuzzy name match. If this policy's promoted arrest
#     component is ever re-measured under a new registry name, this line
#     must move with it; the completeness test below only proves the NAME
#     still resolves, not that it is still the right name.
#
#   spread_gap_zone_fade -> pick_conditioned_spread_gap_zone_pre2018
#     Declared: spread_gap_zone_fade_overlay.py's own module docstring reads
#     this exact registry row before the overlay was built, and
#     challengers.json's entry cites it as sole evidence.
#
# tests/test_four_overlay_composition.py fails the build if a member here
# has no entry, or if any cited name is missing from the live registry --
# so a future member added without a link breaks CI instead of silently
# emptying a chip.
MEMBER_REGISTRY_EVIDENCE: dict[str, tuple[str, ...]] = {
    COACH_FADE: ("hc_year_one_fade",),
    DIVISION_REVENGE_TILT: (
        "bias_battery_division_revenge_game",
        "bias_battery_division_revenge_game_opener",
    ),
    PLAYER_ARRESTS_BACK_SIDE_POLICY: ("player_arrests_recent_14d_back_side_policy_opener",),
    SPREAD_GAP_ZONE_FADE: ("pick_conditioned_spread_gap_zone_pre2018",),
}


def on_the_card_registry_names() -> frozenset[str]:
    """Every weak-signal registry name backing a live policy member.

    The single source :mod:`nfl_ats.signal_ledger` reads to derive its
    "On the card" status -- see :data:`MEMBER_REGISTRY_EVIDENCE` above for
    how each entry was established.
    """

    return frozenset(name for names in MEMBER_REGISTRY_EVIDENCE.values() for name in names)


def policy_definition() -> dict[str, Any]:
    """Return the canonical, JSON-serializable policy definition."""

    return {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "semantics": "joint_or_against_raw_card_complement_once",
        "composition_order": list(COMPOSITION_ORDER),
        "members": [
            {
                "member_id": COACH_FADE,
                "implementation": "nfl_ats.coach_fade_overlay.apply_coach_fade_overlay",
                "parameters": {"enabled": True, "week_max": OVERLAY_WEEK_MAX},
                "production_error_contract": "data_contract_error_disables_member",
            },
            {
                "member_id": DIVISION_REVENGE_TILT,
                "implementation": (
                    "nfl_ats.division_revenge_tilt_overlay.apply_division_revenge_tilt_overlay"
                ),
                "parameters": {"enabled": True},
                "production_error_contract": "propagate",
            },
            {
                "member_id": PLAYER_ARRESTS_BACK_SIDE_POLICY,
                "implementation": (
                    "nfl_ats.player_arrests_back_side_overlay."
                    "apply_player_arrests_back_side_overlay"
                ),
                "parameters": {
                    "window_days": WINDOW_DAYS,
                    "maximum_snapshot_age_hours": MAX_SNAPSHOT_AGE.total_seconds() / 3600.0,
                    "sole_affected_side_only": True,
                },
                "production_error_contract": "fail_closed",
            },
            {
                "member_id": SPREAD_GAP_ZONE_FADE,
                "implementation": (
                    "nfl_ats.spread_gap_zone_fade_overlay.apply_spread_gap_zone_fade_overlay"
                ),
                "parameters": {
                    "enabled": True,
                    "lower_bound_inclusive": SPREAD_GAP_LOWER_BOUND,
                    "upper_bound_inclusive": SPREAD_GAP_UPPER_BOUND,
                },
                "production_error_contract": "propagate",
            },
        ],
    }


def _policy_fingerprint() -> str:
    encoded = json.dumps(
        policy_definition(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


POLICY_FINGERPRINT = _policy_fingerprint()


@dataclass(frozen=True)
class MemberProvenance:
    """One member's independently evaluated transform against the raw card."""

    member_id: str
    order: int
    implementation: str
    enabled: bool
    status: str
    flipped_game_ids: tuple[str, ...]
    detail: str | None = None

    @property
    def flip_count(self) -> int:
        return len(self.flipped_game_ids)


@dataclass(frozen=True)
class GameProvenance:
    """Why one game was complemented by the joint-OR policy."""

    game_id: str
    member_ids: tuple[str, ...]
    raw_home_cover_probability: float
    final_home_cover_probability: float


@dataclass(frozen=True)
class FourOverlayCompositionResult:
    """Composed card plus stable policy and source provenance."""

    overlaid_predictions: pd.DataFrame
    policy_id: str
    policy_fingerprint: str
    composition_order: tuple[str, ...]
    members: tuple[MemberProvenance, ...]
    games: tuple[GameProvenance, ...]
    union_flipped_game_ids: tuple[str, ...]
    overlapping_game_ids: tuple[str, ...]
    arrest_snapshot_id: str
    arrest_snapshot_fetched_at_utc: pd.Timestamp
    arrest_safe_index_sha256: str

    @property
    def flip_count(self) -> int:
        return len(self.union_flipped_game_ids)


def _member_provenance(
    member_id: str,
    order: int,
    result: Any,
    raw: pd.DataFrame,
    *,
    status: str = "applied",
    detail: str | None = None,
) -> MemberProvenance:
    """Validate a sibling overlay's complement-only contract."""

    transformed = result.overlaid_predictions.reset_index(drop=True)
    if list(transformed.columns) != list(raw.columns):
        raise DataContractError(f"{member_id} changed the prediction-card schema")
    raw_ids = raw["game_id"].astype(str)
    transformed_ids = transformed["game_id"].astype(str)
    if not transformed_ids.equals(raw_ids):
        raise DataContractError(f"{member_id} changed prediction-card row identity or order")
    other_columns = [column for column in raw.columns if column != "home_cover_probability"]
    if not transformed[other_columns].equals(raw[other_columns]):
        raise DataContractError(f"{member_id} changed columns outside home_cover_probability")

    flipped_ids = tuple(str(flip.game_id) for flip in result.flips)
    if len(set(flipped_ids)) != len(flipped_ids):
        raise DataContractError(f"{member_id} reported duplicate flip game IDs")
    unknown = sorted(set(flipped_ids).difference(raw_ids))
    if unknown:
        raise DataContractError(f"{member_id} reported unknown flip game IDs: {', '.join(unknown)}")

    raw_probability = pd.to_numeric(raw["home_cover_probability"], errors="coerce")
    transformed_probability = pd.to_numeric(transformed["home_cover_probability"], errors="coerce")
    flip_mask = raw_ids.isin(flipped_ids)
    expected = raw_probability.where(~flip_mask, 1.0 - raw_probability)
    if not np.allclose(
        transformed_probability.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise DataContractError(
            f"{member_id} violated the raw-card complement-only overlay contract"
        )

    implementation = next(
        member["implementation"]
        for member in policy_definition()["members"]
        if member["member_id"] == member_id
    )
    return MemberProvenance(
        member_id=member_id,
        order=order,
        implementation=str(implementation),
        enabled=bool(result.enabled),
        status=status,
        flipped_game_ids=flipped_ids,
        detail=detail,
    )


def apply_four_overlay_composition(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    incidents: pd.DataFrame,
    *,
    arrest_snapshot: ArrestSnapshot,
) -> FourOverlayCompositionResult:
    """Apply the frozen joint-OR policy using already-loaded, frozen inputs.

    Every sibling overlay receives ``raw`` rather than a preceding member's
    output.  Their flip IDs are unioned and the raw probability is complemented
    once, so overlaps agree instead of cancelling.  The year-1 coach member
    retains its established production fail-open behavior for
    :class:`DataContractError`; all other member errors propagate.
    """

    required = {
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "home_cover_probability",
        "spread_line",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(
            f"Predictions are missing four-overlay columns: {', '.join(missing)}"
        )
    raw = predictions.reset_index(drop=True).copy()
    if raw["game_id"].astype(str).duplicated().any():
        raise DataContractError("Predictions contain duplicate games")
    probabilities = pd.to_numeric(raw["home_cover_probability"], errors="coerce")
    if (
        not np.isfinite(probabilities.to_numpy(dtype=float)).all()
        or probabilities.lt(0.0).any()
        or probabilities.gt(1.0).any()
    ):
        raise DataContractError("Predictions contain invalid home_cover_probability values")

    member_rows: list[MemberProvenance] = []
    try:
        coach = apply_coach_fade_overlay(raw, schedules, enabled=True)
    except DataContractError as error:
        # card_view.resolve_overlay already treats coach DataContractError as a
        # disabled member.  Preserve that contract and make it visible here.
        coach = apply_coach_fade_overlay(raw, schedules, enabled=False)
        member_rows.append(
            _member_provenance(
                COACH_FADE,
                0,
                coach,
                raw,
                status="disabled_contract_error",
                detail=str(error),
            )
        )
    else:
        member_rows.append(_member_provenance(COACH_FADE, 0, coach, raw))

    division = apply_division_revenge_tilt_overlay(raw, schedules, enabled=True)
    member_rows.append(_member_provenance(DIVISION_REVENGE_TILT, 1, division, raw))

    arrests = apply_player_arrests_back_side_overlay(raw, incidents)
    member_rows.append(_member_provenance(PLAYER_ARRESTS_BACK_SIDE_POLICY, 2, arrests, raw))

    spread_gap = apply_spread_gap_zone_fade_overlay(raw, enabled=True)
    member_rows.append(_member_provenance(SPREAD_GAP_ZONE_FADE, 3, spread_gap, raw))

    flips_by_game: dict[str, list[str]] = {}
    for member in member_rows:
        for game_id in member.flipped_game_ids:
            flips_by_game.setdefault(game_id, []).append(member.member_id)

    raw_ids = raw["game_id"].astype(str)
    union_ids = tuple(game_id for game_id in raw_ids if game_id in flips_by_game)
    overlap_ids = tuple(game_id for game_id in union_ids if len(flips_by_game[game_id]) > 1)
    union_mask = raw_ids.isin(union_ids)
    overlaid = raw.copy()
    overlaid.loc[union_mask, "home_cover_probability"] = 1.0 - probabilities.loc[union_mask]
    game_rows = tuple(
        GameProvenance(
            game_id=game_id,
            member_ids=tuple(flips_by_game[game_id]),
            raw_home_cover_probability=float(probabilities.loc[raw_ids.eq(game_id)].iloc[0]),
            final_home_cover_probability=float(
                overlaid.loc[raw_ids.eq(game_id), "home_cover_probability"].iloc[0]
            ),
        )
        for game_id in union_ids
    )
    return FourOverlayCompositionResult(
        overlaid_predictions=overlaid,
        policy_id=POLICY_ID,
        policy_fingerprint=POLICY_FINGERPRINT,
        composition_order=COMPOSITION_ORDER,
        members=tuple(member_rows),
        games=game_rows,
        union_flipped_game_ids=union_ids,
        overlapping_game_ids=overlap_ids,
        arrest_snapshot_id=arrest_snapshot.snapshot_id,
        arrest_snapshot_fetched_at_utc=arrest_snapshot.fetched_at_utc,
        arrest_safe_index_sha256=arrest_snapshot.safe_index_sha256,
    )


def apply_four_overlay_composition_for_publication(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> FourOverlayCompositionResult:
    """Load the mandatory fresh arrest input and apply the frozen policy.

    ``load_latest_complete_arrest_snapshot`` is intentionally outside a
    ``try`` block: every availability, completeness, freshness and integrity
    error fails closed before a publishable composition result can exist.
    """

    snapshot = load_latest_complete_arrest_snapshot(data_root, now=now)
    incidents = pd.read_parquet(
        snapshot.safe_index_path, columns=["record_id", "incident_date", "team"]
    )
    return apply_four_overlay_composition(
        predictions,
        schedules,
        incidents,
        arrest_snapshot=snapshot,
    )


__all__ = [
    "COACH_FADE",
    "COMPOSITION_ORDER",
    "DIVISION_REVENGE_TILT",
    "INCUMBENT_CHALLENGER_ID",
    "PLAYER_ARRESTS_BACK_SIDE_POLICY",
    "POLICY_FINGERPRINT",
    "POLICY_ID",
    "SPREAD_GAP_ZONE_FADE",
    "FourOverlayCompositionResult",
    "GameProvenance",
    "MemberProvenance",
    "apply_four_overlay_composition",
    "apply_four_overlay_composition_for_publication",
    "policy_definition",
]
