from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.calibration_distortion import implied_pick_threshold
from nfl_ats.constants import FEATURE_FAMILIES, OUTCOME_COLUMNS
from nfl_ats.io import atomic_json, run_id
from nfl_ats.key_numbers import DEFAULT_KEY_NUMBERS
from nfl_ats.market_decomposition import (
    ATTRIBUTION_ATOL,
    INTERCEPT_FAMILY,
    WEEKLY_CONTEXT_FAMILY,
    _family_for_design_column,
    _standardized_design,
    build_family_map,
)
from nfl_ats.provenance import sha256_file

WATERFALL_SCHEMA_VERSION = 1

WATERFALL_RECONCILIATION_ATOL = 1e-6

ARTIFACT_DIRNAME = "attribution_waterfall"

KIND_MARKET = "market"
KIND_FAMILY = "family"
KIND_OVERLAY = "overlay"
KIND_PROBABILITY_RULE = "probability_rule"
KIND_FINAL = "final"

STEP_KINDS: tuple[str, ...] = (
    KIND_MARKET,
    KIND_FAMILY,
    KIND_OVERLAY,
    KIND_PROBABILITY_RULE,
    KIND_FINAL,
)

_SENTINEL_FAMILIES = frozenset({INTERCEPT_FAMILY, WEEKLY_CONTEXT_FAMILY})
ALLOWED_FAMILIES = frozenset(FEATURE_FAMILIES) | _SENTINEL_FAMILIES


class WaterfallInputError(ValueError):
    pass


@dataclass(frozen=True)
class WaterfallStep:
    step_id: str
    label: str
    family: str | None
    kind: str
    delta_points: float
    cumulative_points: float
    direction: int


@dataclass(frozen=True)
class OverlayFlipEvent:
    overlay: str
    would_flip_alone: bool


@dataclass(frozen=True)
class GameWaterfall:
    game_id: str
    steps: tuple[WaterfallStep, ...]
    market_line: float
    predicted_residual: float
    final_probability: float
    picked_side: str
    edge_vs_spread: float
    key_number_distance: float
    flip_events: tuple[OverlayFlipEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reject_outcome_columns(columns: Iterable[str], *, context: str = "waterfall input") -> None:

    present = sorted(set(columns).intersection(OUTCOME_COLUMNS))
    if present:
        raise WaterfallInputError(
            f"{context} carries outcome columns ({', '.join(present)}); "
            "a pregame waterfall must never read results"
        )


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise WaterfallInputError(f"{name} must be finite, got {value!r}")
    return number


def probability_rule_offset(residuals: npt.ArrayLike) -> float:

    values = np.asarray(residuals, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise WaterfallInputError("probability_rule_offset requires at least one finite residual")
    return -implied_pick_threshold(values)


def key_number_distance(
    projected_home_margin: float,
    key_numbers: Sequence[int] = DEFAULT_KEY_NUMBERS,
) -> float:

    if not key_numbers:
        raise ValueError("At least one key number is required")
    rounded = float(np.round(float(projected_home_margin)))
    magnitude = abs(rounded)
    return min(abs(magnitude - k) for k in key_numbers)


def family_contributions_from_ridge(
    estimator: Any,
    feature_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    families: Mapping[str, Sequence[str]] | None = None,
    atol: float = ATTRIBUTION_ATOL,
) -> list[dict[str, float]]:

    reject_outcome_columns(feature_frame.columns, context="feature frame")
    columns = tuple(feature_columns)
    standardized, names = _standardized_design(estimator, feature_frame, columns)
    ridge = estimator.named_steps["regressor"]
    coefficients = np.asarray(ridge.coef_, dtype=np.float64)
    intercept = float(ridge.intercept_)
    if len(names) != len(coefficients):
        raise RuntimeError("Unable to align waterfall coefficients with transformed features")
    family_map = build_family_map(columns, families)
    families_by_column = [_family_for_design_column(name, family_map) for name in names]
    if len(feature_frame) > 1:
        slate_constant = np.all(np.isclose(standardized, standardized[0:1, :], atol=1e-12), axis=0)
        families_by_column = [
            WEEKLY_CONTEXT_FAMILY if slate_constant[index] else family
            for index, family in enumerate(families_by_column)
        ]
    contributions = standardized * coefficients[np.newaxis, :]
    predicted = np.asarray(estimator.predict(feature_frame.loc[:, list(columns)]), dtype=np.float64)
    totals_per_game: list[dict[str, float]] = []
    for row in range(len(feature_frame)):
        totals: dict[str, float] = dict.fromkeys(sorted(set(families_by_column)), 0.0)
        for column_index, family in enumerate(families_by_column):
            totals[family] += float(contributions[row, column_index])
        totals[INTERCEPT_FAMILY] = intercept
        total = sum(totals.values())
        if not np.isclose(total, float(predicted[row]), atol=atol, rtol=1e-6):
            raise RuntimeError(
                f"Family attribution does not reconcile with the pipeline prediction "
                f"for row {row}: {total:.9f} != {float(predicted[row]):.9f}"
            )
        totals_per_game.append(totals)
    return totals_per_game


def build_game_waterfall(
    *,
    game_id: str,
    market_line: float,
    predicted_residual: float,
    family_contributions: Mapping[str, float],
    probability_rule_offset_points: float,
    raw_home_cover_probability: float,
    overlays: Sequence[Mapping[str, Any]] = (),
    families: Mapping[str, Sequence[str]] | None = None,
    tolerance: float = WATERFALL_RECONCILIATION_ATOL,
) -> GameWaterfall:

    line = _finite(market_line, "market_line")
    residual = _finite(predicted_residual, "predicted_residual")
    offset = _finite(probability_rule_offset_points, "probability_rule_offset_points")
    raw_probability = _finite(raw_home_cover_probability, "raw_home_cover_probability")
    if not 0.0 <= raw_probability <= 1.0:
        raise WaterfallInputError(
            f"raw_home_cover_probability must be in [0, 1], got {raw_probability}"
        )
    allowed_families = (
        ALLOWED_FAMILIES if families is None else frozenset(families) | _SENTINEL_FAMILIES
    )
    unknown_families = sorted(set(family_contributions) - allowed_families)
    if unknown_families:
        raise WaterfallInputError(
            f"Unknown feature families (not in the registry): {', '.join(unknown_families)}"
        )
    contribution_total = sum(float(value) for value in family_contributions.values())
    if abs(contribution_total - residual) > tolerance:
        raise WaterfallInputError(
            f"Family contributions sum to {contribution_total:.9f} but predicted_residual "
            f"is {residual:.9f} (tolerance {tolerance:g})"
        )

    firing_overlays: list[str] = []
    seen: set[str] = set()
    for item in overlays:
        unknown_keys = sorted(set(item) - {"overlay", "fires"})
        if unknown_keys:
            raise WaterfallInputError(f"Overlay entry has unknown keys: {', '.join(unknown_keys)}")
        name = str(item["overlay"])
        if bool(item.get("fires", True)) and name not in seen:
            seen.add(name)
            firing_overlays.append(name)
    firing_overlays.sort()

    cumulative = -line
    entries: list[tuple[str, str, str | None, str, float, float]] = [
        ("market", "Market-implied expectation", "market", KIND_MARKET, -line, cumulative)
    ]

    def push(step_id: str, label: str, family: str | None, kind: str, delta: float) -> None:
        nonlocal cumulative
        cumulative += delta
        entries.append((step_id, label, family, kind, delta, cumulative))

    for family in sorted(family_contributions):
        push(
            f"family:{family}",
            f"{family} contribution",
            family,
            KIND_FAMILY,
            float(family_contributions[family]),
        )
    push(
        "probability_rule",
        "Residual-sample median shift (probability rule)",
        None,
        KIND_PROBABILITY_RULE,
        offset,
    )

    adjusted_margin = cumulative
    reconstructed_raw_side = "HOME" if adjusted_margin >= 0.0 else "AWAY"
    deployed_raw_side = "HOME" if raw_probability >= 0.5 else "AWAY"
    if reconstructed_raw_side != deployed_raw_side:
        raise WaterfallInputError(
            f"Injected inputs reconstruct raw pick {reconstructed_raw_side} for {game_id} "
            f"but the deployed prediction row says {deployed_raw_side}"
        )

    flipped = bool(firing_overlays)
    flip_events = tuple(OverlayFlipEvent(name, True) for name in firing_overlays)
    for position, name in enumerate(firing_overlays):
        composed_once = position > 0
        push(
            f"overlay:{name}",
            f"Policy overlay: {name}" + (" (composed once)" if composed_once else ""),
            None,
            KIND_OVERLAY,
            0.0 if composed_once else -2.0 * cumulative,
        )

    final_value = cumulative
    final_probability = 1.0 - raw_probability if flipped else raw_probability
    picked_side = "HOME" if final_probability >= 0.5 else "AWAY"
    if final_value != 0.0 and (final_value > 0.0) != (picked_side == "HOME"):
        raise RuntimeError(
            f"Waterfall internal inconsistency for {game_id}: final cumulative "
            f"{final_value} disagrees with picked side {picked_side}"
        )
    push("final", "Deployed pick", None, KIND_FINAL, 0.0)

    total_delta = sum(entry[4] for entry in entries)
    if abs(total_delta - final_value) > tolerance:
        raise RuntimeError(
            f"Waterfall does not reconcile for {game_id}: steps sum to "
            f"{total_delta:.9f} but end at {final_value:.9f} "
            f"(tolerance {tolerance:g})"
        )
    running = 0.0
    for _step_id, _label, _family, _kind, delta, step_cumulative in entries:
        running += delta
        if abs(running - step_cumulative) > tolerance:
            raise RuntimeError(f"Waterfall chain broken for {game_id} at step {_step_id}")

    sign = 1.0 if picked_side == "HOME" else -1.0
    steps = tuple(
        WaterfallStep(
            step_id=step_id,
            label=label,
            family=family,
            kind=kind,
            delta_points=delta,
            cumulative_points=step_cumulative,
            direction=0 if delta == 0.0 else int(math.copysign(1.0, delta * sign)),
        )
        for step_id, label, family, kind, delta, step_cumulative in entries
    )
    return GameWaterfall(
        game_id=game_id,
        steps=steps,
        market_line=line,
        predicted_residual=residual,
        final_probability=final_probability,
        picked_side=picked_side,
        edge_vs_spread=abs(adjusted_margin),
        key_number_distance=key_number_distance(adjusted_margin),
        flip_events=flip_events,
    )


def write_waterfall_artifact(
    games: Sequence[GameWaterfall],
    out_dir: Path,
    *,
    now: datetime | None = None,
) -> Path:

    instant = now or datetime.now(UTC)
    directory = out_dir / ARTIFACT_DIRNAME / run_id(instant)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": WATERFALL_SCHEMA_VERSION,
        "created_at_utc": instant.isoformat(),
        "waterfalls": [game.to_dict() for game in games],
    }
    atomic_json(payload, directory / "waterfalls.json")
    files = [{"name": "waterfalls.json", "sha256": sha256_file(directory / "waterfalls.json")}]
    manifest = {
        "artifact": ARTIFACT_DIRNAME,
        "schema_version": WATERFALL_SCHEMA_VERSION,
        "created_at_utc": instant.isoformat(),
        "waterfalls": len(games),
        "files": files,
    }
    atomic_json(manifest, directory / "manifest.json")
    return directory


def read_waterfall_artifact(directory: Path) -> list[dict[str, Any]]:

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise WaterfallInputError(f"No manifest.json in {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = directory / str(entry["name"])
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise WaterfallInputError(
                f"Artifact file failed its manifest hash check: {entry['name']}"
            )
    payload = json.loads((directory / "waterfalls.json").read_text(encoding="utf-8"))
    return list(payload["waterfalls"])


__all__ = [
    "ALLOWED_FAMILIES",
    "ARTIFACT_DIRNAME",
    "STEP_KINDS",
    "WATERFALL_RECONCILIATION_ATOL",
    "WATERFALL_SCHEMA_VERSION",
    "GameWaterfall",
    "OverlayFlipEvent",
    "WaterfallInputError",
    "WaterfallStep",
    "build_game_waterfall",
    "family_contributions_from_ridge",
    "key_number_distance",
    "probability_rule_offset",
    "read_waterfall_artifact",
    "reject_outcome_columns",
    "write_waterfall_artifact",
]
