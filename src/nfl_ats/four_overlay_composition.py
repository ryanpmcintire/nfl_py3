"""Frozen three-member overlay composition for a prospective played policy.

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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.bye_edge_fade_overlay import POST_BYE_GAP_DAYS, apply_bye_edge_fade_overlay
from nfl_ats.coach_fade_overlay import OVERLAY_WEEK_MAX, apply_coach_fade_overlay
from nfl_ats.data import DataContractError
from nfl_ats.division_revenge_tilt_overlay import apply_division_revenge_tilt_overlay
from nfl_ats.forecast_cold_visitor_tilt_overlay import (
    STATION_MAP_RELATIVE_PATH,
    TEMP_GAP_THRESHOLD_F,
    apply_forecast_cold_visitor_tilt_overlay,
    fetch_tuesday_noon_forecast_temps_fail_open,
)
from nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay import (
    HIGH_TOTAL_THRESHOLD,
    PRECIP_PROB_THRESHOLD_PCT,
    apply_precip_high_total_tilt_overlay,
)
from nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay import (
    fetch_kickoff_nearest_forecasts_fail_open,
    games_for_forecast_fetch,
)
from nfl_ats.interim_hc_first_game_tilt_overlay import apply_interim_hc_first_game_tilt_overlay
from nfl_ats.pbp08_protection_mismatch_tilt_overlay import (
    apply_pbp08_protection_mismatch_tilt,
    flags_for_week_fail_open,
)
from nfl_ats.player_arrests_back_side_overlay import (
    MAX_SNAPSHOT_AGE,
    WINDOW_DAYS,
    ArrestSnapshot,
    apply_player_arrests_back_side_overlay,
    load_latest_complete_arrest_snapshot,
)
from nfl_ats.tank_zone_fade_tilt_overlay import (
    OVERLAY_WEEK_MAX as TANK_ZONE_WEEK_MAX,
)
from nfl_ats.tank_zone_fade_tilt_overlay import (
    OVERLAY_WEEK_MIN as TANK_ZONE_WEEK_MIN,
)
from nfl_ats.tank_zone_fade_tilt_overlay import (
    apply_tank_zone_fade_tilt_overlay,
)

POLICY_ID = "overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3"
INCUMBENT_CHALLENGER_ID = "overlay_production_chain_coach_arrest_incumbent"
RETIRED_THREE_MEMBER_CHALLENGER_ID = "overlay_three_member_union_retired_20260909"

COACH_FADE = "coach_fade"
DIVISION_REVENGE_TILT = "division_revenge_tilt"
PLAYER_ARRESTS_BACK_SIDE_POLICY = "player_arrests_back_side_policy"
BYE_EDGE_FADE = "bye_edge_fade"
FORECAST_COLD_VISITOR_TILT = "forecast_cold_visitor_tilt"
PBP08_PROTECTION_MISMATCH_TILT = "pbp08_protection_mismatch_tilt"
INTERIM_HC_FIRST_GAME_TILT = "interim_hc_first_game_tilt"
TANK_ZONE_FADE_TILT = "tank_zone_fade_tilt"
PRECIP_HIGH_TOTAL_TILT = "precip_high_total_tilt"
SPREAD_GAP_ZONE_FADE = "spread_gap_zone_fade"

COMPOSITION_ORDER = (
    COACH_FADE,
    DIVISION_REVENGE_TILT,
    PLAYER_ARRESTS_BACK_SIDE_POLICY,
    BYE_EDGE_FADE,
    FORECAST_COLD_VISITOR_TILT,
    PBP08_PROTECTION_MISMATCH_TILT,
    INTERIM_HC_FIRST_GAME_TILT,
    TANK_ZONE_FADE_TILT,
    PRECIP_HIGH_TOTAL_TILT,
)

INPUT_UNAVAILABLE_STATUS = "disabled_input_unavailable"

FAIL_CLOSED_MEMBERS: frozenset[str] = frozenset(
    {COACH_FADE, DIVISION_REVENGE_TILT, PLAYER_ARRESTS_BACK_SIDE_POLICY}
)

MEMBER_REGISTRY_EVIDENCE: dict[str, tuple[str, ...]] = {
    COACH_FADE: ("hc_year_one_fade",),
    DIVISION_REVENGE_TILT: (
        "bias_battery_division_revenge_game",
        "bias_battery_division_revenge_game_opener",
    ),
    PLAYER_ARRESTS_BACK_SIDE_POLICY: ("player_arrests_recent_14d_back_side_policy_opener",),
    BYE_EDGE_FADE: (
        "bye_overval_fade_full_slate_post2011",
        "unserved_tilt_on_played_card_bye_edge_fade_overlay",
    ),
    FORECAST_COLD_VISITOR_TILT: (
        "forecast_weather_temp_gap_cold_visitor",
        "weather_followup_temp_gap_cold_visitor",
        "unserved_tilt_on_played_card_forecast_cold_visitor_tilt_overlay",
    ),
    PBP08_PROTECTION_MISMATCH_TILT: (
        "pbp08_protection_mismatch",
        "unserved_tilt_on_played_card_pbp08_protection_mismatch_tilt_overlay",
    ),
    INTERIM_HC_FIRST_GAME_TILT: (
        "interim_hc_first_game",
        "unserved_tilt_on_played_card_interim_hc_first_game_tilt_overlay",
    ),
    TANK_ZONE_FADE_TILT: (
        "motivation_ladder_tank_zone_wk14_18",
        "unserved_tilt_on_played_card_tank_zone_fade_tilt_overlay",
    ),
    PRECIP_HIGH_TOTAL_TILT: (
        "forecast_weather_kn_precip_high_total_full",
        "forecast_weather_kn_precip_high_total_pre2020",
        "unserved_tilt_on_played_card_forecast_weather_kn_precip_high_total_tilt_overlay",
    ),
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
                "member_id": BYE_EDGE_FADE,
                "implementation": "nfl_ats.bye_edge_fade_overlay.apply_bye_edge_fade_overlay",
                "parameters": {"enabled": True, "post_bye_gap_days": POST_BYE_GAP_DAYS},
                "production_error_contract": "propagate",
            },
            {
                "member_id": FORECAST_COLD_VISITOR_TILT,
                "implementation": (
                    "nfl_ats.forecast_cold_visitor_tilt_overlay."
                    "apply_forecast_cold_visitor_tilt_overlay"
                ),
                "parameters": {
                    "enabled": True,
                    "temp_gap_threshold_f": TEMP_GAP_THRESHOLD_F,
                    "cutoff_mode": "tuesday_noon",
                },
                "production_error_contract": "missing_forecast_disables_member",
            },
            {
                "member_id": PBP08_PROTECTION_MISMATCH_TILT,
                "implementation": (
                    "nfl_ats.pbp08_protection_mismatch_tilt_overlay."
                    "apply_pbp08_protection_mismatch_tilt"
                ),
                "parameters": {"enabled": True},
                "production_error_contract": "missing_flags_disables_member",
            },
            {
                "member_id": INTERIM_HC_FIRST_GAME_TILT,
                "implementation": (
                    "nfl_ats.interim_hc_first_game_tilt_overlay."
                    "apply_interim_hc_first_game_tilt_overlay"
                ),
                "parameters": {"enabled": True},
                "production_error_contract": "missing_repo_root_disables_member",
            },
            {
                "member_id": TANK_ZONE_FADE_TILT,
                "implementation": (
                    "nfl_ats.tank_zone_fade_tilt_overlay.apply_tank_zone_fade_tilt_overlay"
                ),
                "parameters": {
                    "enabled": True,
                    "week_min": TANK_ZONE_WEEK_MIN,
                    "week_max": TANK_ZONE_WEEK_MAX,
                },
                "production_error_contract": "propagate",
            },
            {
                "member_id": PRECIP_HIGH_TOTAL_TILT,
                "implementation": (
                    "nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay."
                    "apply_precip_high_total_tilt_overlay"
                ),
                "parameters": {
                    "enabled": True,
                    "precip_prob_threshold_pct": PRECIP_PROB_THRESHOLD_PCT,
                    "high_total_threshold": HIGH_TOTAL_THRESHOLD,
                    "cutoff_mode": "kickoff_nearest",
                },
                "production_error_contract": "missing_forecast_disables_member",
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


def _empty_temp_forecasts(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"game_id": raw["game_id"].astype(str), "forecast_temp_f": np.nan}
    ).reset_index(drop=True)


def _empty_precip_forecasts(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"game_id": raw["game_id"].astype(str), "forecast_precip_prob_pct": np.nan}
    ).reset_index(drop=True)


def _empty_protection_flags() -> pd.DataFrame:
    return pd.DataFrame(columns=["game_id", "back_side"])


@dataclass(frozen=True)
class _DisabledMemberResult:
    """A member that never ran, shaped like the overlay result it stands in for."""

    overlaid_predictions: pd.DataFrame
    flips: tuple[Any, ...]
    enabled: bool


def _guarded_member(
    member_id: str,
    order: int,
    raw: pd.DataFrame,
    apply: Callable[[bool], Any],
    *,
    available: bool,
) -> MemberProvenance:
    """Run one added member, disabling it rather than raising on a missing input."""

    disabled = _DisabledMemberResult(raw.reset_index(drop=True).copy(), (), False)
    if not available:
        return _member_provenance(
            member_id,
            order,
            disabled,
            raw,
            status=INPUT_UNAVAILABLE_STATUS,
            detail=f"{member_id} received no usable input and contributed no flips",
        )
    try:
        result = apply(True)
    except DataContractError as error:
        return _member_provenance(
            member_id,
            order,
            disabled,
            raw,
            status="disabled_contract_error",
            detail=str(error),
        )
    return _member_provenance(member_id, order, result, raw)


def apply_four_overlay_composition(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    incidents: pd.DataFrame,
    *,
    arrest_snapshot: ArrestSnapshot,
    forecasts_tuesday_noon: pd.DataFrame | None = None,
    forecasts_kickoff_nearest: pd.DataFrame | None = None,
    protection_flags: pd.DataFrame | None = None,
    repo_root: Path | None = None,
) -> FourOverlayCompositionResult:
    """Apply the frozen joint-OR policy using already-loaded, frozen inputs.

    Every sibling overlay receives ``raw`` rather than a preceding member's
    output.  Their flip IDs are unioned and the raw probability is complemented
    once, so overlaps agree instead of cancelling.  The year-1 coach member
    retains its established production fail-open behavior for
    :class:`DataContractError`; all other member errors propagate.

    The six members added 2026-09-09 each need an input this function cannot
    load for itself.  A caller that has nothing to supply passes ``None`` and
    that member is reported as ``disabled_input_unavailable`` with zero flips,
    which is how a research caller with a miniature fixture schedule gets a
    well-defined card instead of an exception.
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

    member_rows.append(
        _guarded_member(
            BYE_EDGE_FADE,
            3,
            raw,
            lambda enabled: apply_bye_edge_fade_overlay(raw, schedules, enabled=enabled),
            available=True,
        )
    )
    member_rows.append(
        _guarded_member(
            FORECAST_COLD_VISITOR_TILT,
            4,
            raw,
            lambda enabled: apply_forecast_cold_visitor_tilt_overlay(
                raw,
                schedules,
                forecasts_tuesday_noon
                if forecasts_tuesday_noon is not None
                else _empty_temp_forecasts(raw),
                enabled=enabled,
            ),
            available=forecasts_tuesday_noon is not None,
        )
    )
    member_rows.append(
        _guarded_member(
            PBP08_PROTECTION_MISMATCH_TILT,
            5,
            raw,
            lambda enabled: apply_pbp08_protection_mismatch_tilt(
                raw,
                protection_flags if protection_flags is not None else _empty_protection_flags(),
                enabled=enabled,
            ),
            available=protection_flags is not None,
        )
    )
    member_rows.append(
        _guarded_member(
            INTERIM_HC_FIRST_GAME_TILT,
            6,
            raw,
            lambda enabled: apply_interim_hc_first_game_tilt_overlay(
                raw, repo_root if repo_root is not None else Path("."), enabled=enabled
            ),
            available=repo_root is not None,
        )
    )
    member_rows.append(
        _guarded_member(
            TANK_ZONE_FADE_TILT,
            7,
            raw,
            lambda enabled: apply_tank_zone_fade_tilt_overlay(raw, schedules, enabled=enabled),
            available=True,
        )
    )
    member_rows.append(
        _guarded_member(
            PRECIP_HIGH_TOTAL_TILT,
            8,
            raw,
            lambda enabled: apply_precip_high_total_tilt_overlay(
                raw,
                schedules,
                forecasts_kickoff_nearest
                if forecasts_kickoff_nearest is not None
                else _empty_precip_forecasts(raw),
                enabled=enabled,
            ),
            available=forecasts_kickoff_nearest is not None and "total_line" in raw.columns,
        )
    )

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


def protection_flags_for_card(predictions: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """The PBP-08 lean table for every (season, week) on the card, fail-open."""

    weeks = (
        predictions[["season", "week"]]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values(["season", "week"])
    )
    frames = [
        flags_for_week_fail_open(data_root, season=int(str(row.season)), week=int(str(row.week)))
        for row in weeks.itertuples()
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_protection_flags()
    return pd.concat(frames, ignore_index=True)


def forecasts_for_card(
    predictions: pd.DataFrame, schedules: pd.DataFrame, registry_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tuesday-noon and kickoff-nearest forecasts for the card, both fail-open.

    Every failure mode of the two fetches is already folded into an all-NaN
    frame by their own wrappers, which reads downstream as "no game flagged".
    Neither weather member can block a publish.
    """

    if "kickoff" not in predictions.columns:
        return _empty_temp_forecasts(predictions), _empty_precip_forecasts(predictions)
    try:
        games = games_for_forecast_fetch(predictions, schedules)
    except (KeyError, ValueError, DataContractError):
        return _empty_temp_forecasts(predictions), _empty_precip_forecasts(predictions)
    station_map_path = registry_root / STATION_MAP_RELATIVE_PATH
    tuesday = fetch_tuesday_noon_forecast_temps_fail_open(games, station_map_path)
    kickoff_nearest = fetch_kickoff_nearest_forecasts_fail_open(games, station_map_path)
    return tuesday, kickoff_nearest


def apply_four_overlay_composition_for_publication(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    data_root: Path,
    *,
    now: datetime | None = None,
    repo_root: Path | None = None,
    forecasts_tuesday_noon: pd.DataFrame | None = None,
    forecasts_kickoff_nearest: pd.DataFrame | None = None,
) -> FourOverlayCompositionResult:
    """Load the mandatory fresh arrest input and apply the frozen policy.

    ``load_latest_complete_arrest_snapshot`` is intentionally outside a
    ``try`` block: every availability, completeness, freshness and integrity
    error fails closed before a publishable composition result can exist.
    The six members added 2026-09-09 are the opposite posture by their own
    module contracts -- each loads through a fail-open helper, so a missing
    snapshot or a failed fetch reads as zero flips rather than a broken
    publish. A caller that already fetched the weather may pass it in so the
    network is not hit twice.
    """

    snapshot = load_latest_complete_arrest_snapshot(data_root, now=now)
    incidents = pd.read_parquet(
        snapshot.safe_index_path, columns=["record_id", "incident_date", "team"]
    )
    root = repo_root if repo_root is not None else data_root.resolve().parent
    if forecasts_tuesday_noon is None or forecasts_kickoff_nearest is None:
        fetched_tuesday, fetched_kickoff = forecasts_for_card(
            predictions, schedules, root / "registry"
        )
        if forecasts_tuesday_noon is None:
            forecasts_tuesday_noon = fetched_tuesday
        if forecasts_kickoff_nearest is None:
            forecasts_kickoff_nearest = fetched_kickoff
    return apply_four_overlay_composition(
        predictions,
        schedules,
        incidents,
        arrest_snapshot=snapshot,
        forecasts_tuesday_noon=forecasts_tuesday_noon,
        forecasts_kickoff_nearest=forecasts_kickoff_nearest,
        protection_flags=protection_flags_for_card(predictions, data_root),
        repo_root=root,
    )


__all__ = [
    "BYE_EDGE_FADE",
    "COACH_FADE",
    "COMPOSITION_ORDER",
    "DIVISION_REVENGE_TILT",
    "FORECAST_COLD_VISITOR_TILT",
    "INCUMBENT_CHALLENGER_ID",
    "INTERIM_HC_FIRST_GAME_TILT",
    "PBP08_PROTECTION_MISMATCH_TILT",
    "PLAYER_ARRESTS_BACK_SIDE_POLICY",
    "POLICY_FINGERPRINT",
    "POLICY_ID",
    "PRECIP_HIGH_TOTAL_TILT",
    "RETIRED_THREE_MEMBER_CHALLENGER_ID",
    "SPREAD_GAP_ZONE_FADE",
    "TANK_ZONE_FADE_TILT",
    "FourOverlayCompositionResult",
    "GameProvenance",
    "MemberProvenance",
    "apply_four_overlay_composition",
    "apply_four_overlay_composition_for_publication",
    "forecasts_for_card",
    "policy_definition",
    "protection_flags_for_card",
]
