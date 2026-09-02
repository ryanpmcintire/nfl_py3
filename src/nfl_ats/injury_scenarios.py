"""Point-in-time-safe mixtures of complete, mutually exclusive injury scenarios.

This module is an isolated distribution kernel. It does not infer joint lineup
probabilities, convert player value to points, register a feature profile, or
alter the played card. Callers must supply a complete joint state and a margin
center for every scenario; those are research inputs, not hidden assumptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.data import DataContractError, require_columns
from nfl_ats.margin import _smoothed_probability, _three_way_probabilities

SCENARIO_REQUIRED_COLUMNS = (
    "game_id",
    "revision_id",
    "scenario_id",
    "probability",
    "predicted_margin",
    "active_player_ids",
    "inactive_player_ids",
    "observed_at_utc",
    "effective_at_utc",
    "source_id",
)
_PROBABILITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class InjuryScenarioComponent:
    """One complete lineup state in a mutually exclusive game-level partition."""

    scenario_id: str
    probability: float
    predicted_margin: float
    active_player_ids: tuple[str, ...]
    inactive_player_ids: tuple[str, ...]


@dataclass(frozen=True)
class InjuryScenarioMarginMixture:
    """Immutable summary of a weighted empirical margin distribution."""

    game_id: str
    revision_id: str
    source_id: str
    observed_at_utc: pd.Timestamp
    effective_at_utc: pd.Timestamp
    decision_at_utc: pd.Timestamp
    spread_line: float
    components: tuple[InjuryScenarioComponent, ...]
    residuals: tuple[float, ...]
    mean: float
    variance: float
    home_win_probability: float
    home_cover_probability: float
    home_cover_probability_excluding_push: float
    push_probability: float
    home_loss_probability: float


def _timestamp(value: Any, label: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise DataContractError(f"Injury scenarios contain an invalid {label}")
    return pd.Timestamp(parsed)


def _player_ids(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple, set, frozenset)):
        raise DataContractError(f"{label} must be a collection of player identities")
    if any(pd.isna(player) for player in value):
        raise DataContractError(f"{label} contains a missing player identity")
    identities = tuple(sorted(str(player).strip() for player in value))
    if any(not player for player in identities) or len(set(identities)) != len(identities):
        raise DataContractError(f"{label} contains blank or duplicate player identities")
    return identities


def _latest_visible_revision(
    revisions: pd.DataFrame, *, game_id: str, decision_at: pd.Timestamp
) -> pd.DataFrame:
    game_rows = revisions.loc[revisions["game_id"].astype(str).eq(game_id)].copy()
    if game_rows.empty:
        raise DataContractError(f"Injury scenarios are missing game {game_id!r}")
    game_rows["observed_at_utc"] = pd.to_datetime(
        game_rows["observed_at_utc"], errors="coerce", utc=True
    )
    game_rows["effective_at_utc"] = pd.to_datetime(
        game_rows["effective_at_utc"], errors="coerce", utc=True
    )
    if game_rows[["observed_at_utc", "effective_at_utc"]].isna().any(axis=None):
        raise DataContractError("Injury scenarios contain invalid provenance timestamps")

    candidates: list[tuple[pd.Timestamp, pd.Timestamp, str, pd.DataFrame]] = []
    for raw_revision_id, group in game_rows.groupby("revision_id", sort=False, dropna=False):
        if pd.isna(raw_revision_id):
            raise DataContractError("Injury scenario revision_id cannot be missing")
        revision_id = str(raw_revision_id).strip()
        if not revision_id:
            raise DataContractError("Injury scenario revision_id cannot be blank")
        if group["observed_at_utc"].nunique() != 1 or group["effective_at_utc"].nunique() != 1:
            raise DataContractError(f"Injury scenario revision {revision_id!r} is not atomic")
        observed_at = pd.Timestamp(group["observed_at_utc"].iloc[0])
        effective_at = pd.Timestamp(group["effective_at_utc"].iloc[0])
        if observed_at <= decision_at and effective_at <= decision_at:
            candidates.append((observed_at, effective_at, revision_id, group.copy()))
    if not candidates:
        raise DataContractError(f"No injury scenario revision is visible for game {game_id!r}")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    latest_observed, latest_effective, _, selected = candidates[-1]
    tied = [
        revision
        for observed, effective, revision, _ in candidates
        if observed == latest_observed and effective == latest_effective
    ]
    if len(tied) > 1:
        raise DataContractError(
            "Injury scenario revisions have ambiguous latest provenance: " + ", ".join(tied)
        )
    return selected


def _components(selected: pd.DataFrame) -> tuple[InjuryScenarioComponent, ...]:
    if selected["source_id"].nunique(dropna=False) != 1:
        raise DataContractError("An injury scenario revision must have one immutable source_id")
    raw_source_id = selected["source_id"].iloc[0]
    source_id = "" if pd.isna(raw_source_id) else str(raw_source_id).strip()
    if not source_id:
        raise DataContractError("Injury scenario source_id cannot be blank")
    if selected["scenario_id"].isna().any():
        raise DataContractError("Injury scenario_id cannot be missing")
    scenario_ids = selected["scenario_id"].astype(str).str.strip()
    if scenario_ids.eq("").any() or scenario_ids.duplicated().any():
        raise DataContractError("Injury scenario_id values must be unique within a revision")
    if len(selected) < 2:
        raise DataContractError("An injury scenario mixture requires at least two scenarios")

    components: list[InjuryScenarioComponent] = []
    signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    identity_universe: frozenset[str] | None = None
    for scenario_id, row in zip(scenario_ids, selected.itertuples(index=False), strict=True):
        active = _player_ids(row.active_player_ids, "active_player_ids")
        inactive = _player_ids(row.inactive_player_ids, "inactive_player_ids")
        if set(active).intersection(inactive):
            raise DataContractError("A player cannot be active and inactive in the same scenario")
        universe = frozenset((*active, *inactive))
        if not universe:
            raise DataContractError("Each scenario must partition at least one player identity")
        if identity_universe is None:
            identity_universe = universe
        elif universe != identity_universe:
            raise DataContractError(
                "Every scenario must provide a full partition of the same player identities"
            )
        signature = (active, inactive)
        if signature in signatures:
            raise DataContractError("Injury scenarios contain a duplicate state signature")
        signatures.add(signature)

        try:
            probability = float(str(row.probability))
            predicted_margin = float(str(row.predicted_margin))
        except ValueError as error:
            raise DataContractError("Scenario probabilities and margins must be numeric") from error
        if not math.isfinite(probability) or probability <= 0.0 or probability > 1.0:
            raise DataContractError("Scenario probabilities must be finite and in (0, 1]")
        if not math.isfinite(predicted_margin):
            raise DataContractError("Scenario predicted margins must be finite")
        components.append(
            InjuryScenarioComponent(
                scenario_id=scenario_id,
                probability=probability,
                predicted_margin=predicted_margin,
                active_player_ids=active,
                inactive_player_ids=inactive,
            )
        )
    total_probability = math.fsum(component.probability for component in components)
    if not math.isclose(total_probability, 1.0, abs_tol=_PROBABILITY_TOLERANCE):
        raise DataContractError(
            f"Scenario probabilities must sum to one; received {total_probability:.12g}"
        )
    return tuple(components)


def build_injury_scenario_margin_mixture(
    revisions: pd.DataFrame,
    *,
    game_id: str,
    decision_at_utc: Any,
    spread_line: float,
    residuals: npt.ArrayLike,
) -> InjuryScenarioMarginMixture:
    """Select the latest visible joint scenario revision and mix its margins.

    Every component uses the same already-fitted, out-of-time residual sample
    as ``MarginModel``. The caller owns the scenario probabilities and centers;
    this kernel validates their point-in-time/provenance contract and performs
    only the distribution mixture.
    """

    require_columns(revisions, SCENARIO_REQUIRED_COLUMNS, "injury scenario revisions")
    decision_at = _timestamp(decision_at_utc, "decision_at_utc")
    line = float(spread_line)
    if not math.isfinite(line):
        raise DataContractError("Injury scenario mixture requires a finite spread line")
    residual_array = np.asarray(residuals, dtype=np.float64)
    if (
        residual_array.ndim != 1
        or residual_array.size == 0
        or not np.isfinite(residual_array).all()
    ):
        raise DataContractError("Injury scenario residuals must be a non-empty finite vector")

    selected = _latest_visible_revision(revisions, game_id=str(game_id), decision_at=decision_at)
    components = _components(selected)
    observed_at = pd.Timestamp(selected["observed_at_utc"].iloc[0])
    effective_at = pd.Timestamp(selected["effective_at_utc"].iloc[0])

    residual_mean = float(np.mean(residual_array))
    residual_variance = float(np.var(residual_array))
    component_means = np.asarray(
        [component.predicted_margin + residual_mean for component in components], dtype=float
    )
    probabilities = np.asarray([component.probability for component in components], dtype=float)
    mean = float(np.dot(probabilities, component_means))
    variance = float(np.dot(probabilities, residual_variance + np.square(component_means - mean)))

    home_win = 0.0
    home_cover = 0.0
    cover_excluding_push = 0.0
    push = 0.0
    loss = 0.0
    for component in components:
        distribution = component.predicted_margin + residual_array
        home_win += component.probability * _smoothed_probability(distribution, 0.0)
        home_cover += component.probability * _smoothed_probability(distribution, line)
        component_cover, component_push, component_loss = _three_way_probabilities(
            distribution, line
        )
        cover_excluding_push += component.probability * component_cover
        push += component.probability * component_push
        loss += component.probability * component_loss

    return InjuryScenarioMarginMixture(
        game_id=str(game_id),
        revision_id=str(selected["revision_id"].iloc[0]),
        source_id=str(selected["source_id"].iloc[0]),
        observed_at_utc=observed_at,
        effective_at_utc=effective_at,
        decision_at_utc=decision_at,
        spread_line=line,
        components=components,
        residuals=tuple(float(value) for value in residual_array),
        mean=mean,
        variance=variance,
        home_win_probability=home_win,
        home_cover_probability=home_cover,
        home_cover_probability_excluding_push=cover_excluding_push,
        push_probability=push,
        home_loss_probability=loss,
    )
