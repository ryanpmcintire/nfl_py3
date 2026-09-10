from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.mass_preserving_lattice import (
    DiscretePushReader,
    MassPreservingRead,
    residual_location,
)

FloatArray = npt.NDArray[np.float64]

KEY_LINE_PICK_READ_SERVED = True
KEY_LINE_PICK_READ_POLICY = "key_line_pick_read_v1"
KEY_LINE_PICK_READ_FILENAME = "key_line_pick_read.json"
KEY_LINE_ATOMS: tuple[float, ...] = (3.0, 7.0)
KEY_LINE_TOLERANCE = 1e-9

KEY_LINE_STATUS_SERVED = "served"
KEY_LINE_STATUS_INAPPLICABLE = "inapplicable"
KEY_LINE_STATUS_NOT_RUN = "not_run"


def key_line_atom(line: float, atoms: Iterable[float] = KEY_LINE_ATOMS) -> float | None:

    if line is None or not np.isfinite(line):
        return None
    size = abs(float(line))
    for atom in atoms:
        if abs(size - float(atom)) < KEY_LINE_TOLERANCE:
            return float(atom)
    return None


def key_line_mask(
    lines: pd.Series | np.ndarray | Iterable[float], atoms: Iterable[float] = KEY_LINE_ATOMS
) -> npt.NDArray[np.bool_]:

    size = np.abs(pd.to_numeric(pd.Series(list(lines)), errors="coerce").to_numpy(dtype=float))
    keys = np.asarray(list(atoms), dtype=float)
    if keys.size == 0:
        return np.zeros(size.shape, dtype=bool)
    hits = np.abs(size[:, None] - keys[None, :]) < KEY_LINE_TOLERANCE
    return np.asarray(np.any(hits, axis=1) & np.isfinite(size), dtype=bool)


def is_half_point_line(line: float | None) -> bool:

    if line is None:
        return False
    value = float(line)
    if not np.isfinite(value):
        return False
    return bool(abs((abs(value) % 1.0) - 0.5) < KEY_LINE_TOLERANCE)


def key_line_read_applicable(line: float | None, atoms: Iterable[float] = KEY_LINE_ATOMS) -> bool:

    if line is None:
        return False
    return key_line_atom(float(line), atoms) is not None


@dataclass(frozen=True)
class KeyLineApplicability:
    games: int
    lines_on_an_atom: int
    half_point_lines: int
    whole_number_lines: int
    atoms: tuple[float, ...] = KEY_LINE_ATOMS

    @property
    def applicable(self) -> bool:

        return self.lines_on_an_atom > 0

    @property
    def status(self) -> str:
        return KEY_LINE_STATUS_SERVED if self.applicable else KEY_LINE_STATUS_INAPPLICABLE

    @property
    def reason(self) -> str | None:

        if self.applicable:
            return None
        numbers = " or ".join(f"{atom:g}" for atom in self.atoms) or "any key number"
        if self.games == 0:
            return f"no line was served this week, so no line could sit on {numbers}"
        if self.half_point_lines == self.games:
            return (
                f"no served line sits exactly on {numbers}: all {self.games} are half points, "
                "which the pool always quotes. The key-number mass still decides these games -- "
                "it falls on one side instead of pushing -- so this is the exact-match test not "
                "matching, not a reason to read them smoothly (MOD-18 candidate C2)"
            )
        return (
            f"no served line sits exactly on {numbers}: of {self.games} lines, "
            f"{self.half_point_lines} are half points and {self.whole_number_lines} are whole "
            "numbers off those atoms (MOD-18 candidate C2 generalises the read to every line)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "reason": self.reason,
            "atoms": list(self.atoms),
            "games": self.games,
            "lines_on_an_atom": self.lines_on_an_atom,
            "half_point_lines": self.half_point_lines,
            "whole_number_lines": self.whole_number_lines,
        }


def key_line_applicability(
    lines: Iterable[float | None], atoms: Iterable[float] = KEY_LINE_ATOMS
) -> KeyLineApplicability:

    declared = tuple(float(atom) for atom in atoms)
    values = [None if line is None else float(line) for line in lines]
    on_atom = sum(1 for line in values if key_line_read_applicable(line, declared))
    half_point = sum(1 for line in values if is_half_point_line(line))
    whole_number = sum(
        1 for line in values if line is not None and np.isfinite(line) and float(line).is_integer()
    )
    return KeyLineApplicability(
        games=len(values),
        lines_on_an_atom=on_atom,
        half_point_lines=half_point,
        whole_number_lines=whole_number,
        atoms=declared,
    )


def key_line_decision_probability(read: MassPreservingRead) -> float:

    return float(read.home_cover_probability)


@dataclass(frozen=True)
class ServedKeyLineRead:
    game_id: str
    line: float
    point: float
    atom: float | None
    touched: bool
    smooth: float
    discrete: float | None
    served: float
    cover: float | None
    push: float | None
    loss: float | None
    theta: float | None
    band: float | None
    band_games: int | None

    @property
    def side_changed(self) -> bool:
        return (self.served >= 0.5) != (self.smooth >= 0.5)

    @property
    def inapplicable_reason(self) -> str | None:

        if self.touched:
            return None
        if is_half_point_line(self.line):
            return (
                "half-point line, so the exact-match test on 3 and 7 cannot fire; the key-number "
                "mass lands wholly on one side here rather than pushing (MOD-18 candidate C2)"
            )
        return "whole-number line, but not one of the declared key numbers"

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "spread_line": self.line,
            "point": self.point,
            "atom": self.atom,
            "touched": self.touched,
            "inapplicable_reason": self.inapplicable_reason,
            "home_cover_probability_smooth": self.smooth,
            "home_cover_probability_discrete": self.discrete,
            "home_cover_probability": self.served,
            "side_changed": self.side_changed,
            "cover": self.cover,
            "push": self.push,
            "loss": self.loss,
            "theta": self.theta,
            "band": self.band,
            "band_games": self.band_games,
        }


@dataclass(frozen=True)
class KeyLinePickRead:
    reader: DiscretePushReader
    atoms: tuple[float, ...] = KEY_LINE_ATOMS
    policy: str = KEY_LINE_PICK_READ_POLICY

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "atoms": list(self.atoms),
            "decision_number": "cover + push / 2",
            "band_half_width": self.reader.half_width,
            "min_band_games": self.reader.min_band_games,
            "prior_rows": self.reader.prior_rows,
            "prior_max_gameday": self.reader.max_gameday,
        }


def apply_key_line_pick_read(
    forecasts: pd.DataFrame,
    games: pd.DataFrame,
    policy: KeyLinePickRead,
    *,
    residuals: FloatArray,
    probability_method: str,
    log: MutableMapping[str, ServedKeyLineRead] | None = None,
) -> pd.DataFrame:

    if len(forecasts) != len(games):
        raise ValueError("forecasts and games must be row-aligned")
    result = forecasts.copy()
    if result.empty:
        return result
    location = residual_location(residuals, probability_method)
    lines = pd.to_numeric(games["spread_line"], errors="raise").to_numpy(dtype=float)
    points = result["predicted_margin"].to_numpy(dtype=float) + location
    smooth = result["home_cover_probability"].to_numpy(dtype=float)
    ids = games["game_id"].astype(str).to_numpy()
    served = smooth.copy()
    touched = key_line_mask(lines, policy.atoms)
    for index, (game_id, line, point) in enumerate(zip(ids, lines, points, strict=True)):
        read = policy.reader.read(float(line), float(point))
        discrete = key_line_decision_probability(read)
        if touched[index]:
            served[index] = discrete
        if log is not None:
            log[str(game_id)] = ServedKeyLineRead(
                game_id=str(game_id),
                line=float(line),
                point=float(point),
                atom=key_line_atom(float(line), policy.atoms),
                touched=bool(touched[index]),
                smooth=float(smooth[index]),
                discrete=discrete,
                served=float(served[index]),
                cover=read.cover,
                push=read.push,
                loss=read.loss,
                theta=read.theta,
                band=read.band,
                band_games=read.band_games,
            )
    result["home_cover_probability"] = served
    return result


def apply_key_line_pick_read_to_sweep(
    sweep: pd.DataFrame,
    policy: KeyLinePickRead,
    *,
    points_by_game: Mapping[str, float],
    quoted_lines_by_game: Mapping[str, float] | None = None,
) -> pd.DataFrame:

    result = sweep.copy()
    if result.empty:
        return result
    ids = result["game_id"].astype(str).to_numpy()
    lines = result["alternative_line"].to_numpy(dtype=float)
    touched = key_line_mask(lines, policy.atoms)
    probability = result["home_cover_probability"].to_numpy(dtype=float).copy()
    for index in np.flatnonzero(touched):
        game_id = ids[index]
        if game_id not in points_by_game:
            raise ValueError(f"No served point for game {game_id!r} in the line sweep")
        quoted = None if quoted_lines_by_game is None else quoted_lines_by_game.get(game_id)
        read = policy.reader.read(
            float(lines[index]), float(points_by_game[game_id]), conditioning_line=quoted
        )
        probability[index] = key_line_decision_probability(read)
    result["home_cover_probability"] = probability
    pick_probability = np.where(probability >= 0.5, probability, 1.0 - probability)
    if "pick_probability" in result.columns:
        current = result["pick_probability"].to_numpy(dtype=float).copy()
        current[touched] = pick_probability[touched]
        result["pick_probability"] = current
    if "confidence" in result.columns:
        confidence = result["confidence"].to_numpy(dtype=float).copy()
        confidence[touched] = pick_probability[touched] - 0.5
        result["confidence"] = confidence
    return result


def key_line_sidecar(
    policy: KeyLinePickRead | None,
    log: Mapping[str, ServedKeyLineRead],
    game_ids: Iterable[str],
    *,
    error: str | None = None,
) -> dict[str, Any]:

    games = [log[game_id].to_dict() for game_id in game_ids if game_id in log]
    served = policy is not None and error is None
    applicability = (
        key_line_applicability([row.get("spread_line") for row in games], policy.atoms)
        if served and policy is not None
        else None
    )
    return {
        "schema": "key_line_pick_read/1",
        "served": served,
        "status": (applicability.status if applicability is not None else KEY_LINE_STATUS_NOT_RUN),
        "applicability": applicability.to_dict() if applicability is not None else None,
        "policy": KEY_LINE_PICK_READ_POLICY,
        "atoms": list(policy.atoms if policy is not None else KEY_LINE_ATOMS),
        "challenger": "smooth_gaussian_median_every_line",
        "fit": policy.to_dict() if policy is not None else None,
        "error": error,
        "games": games,
    }


def key_line_metadata_block(sidecar: Mapping[str, Any]) -> dict[str, Any]:

    games = sidecar.get("games")
    rows = [row for row in games if isinstance(row, Mapping)] if isinstance(games, list) else []
    touched = [row for row in rows if row.get("touched")]
    return {
        "path": KEY_LINE_PICK_READ_FILENAME,
        "policy": sidecar.get("policy"),
        "served": bool(sidecar.get("served")),
        "status": sidecar.get("status"),
        "applicability": sidecar.get("applicability"),
        "atoms": list(sidecar.get("atoms") or []),
        "error": sidecar.get("error"),
        "games": len(rows),
        "touched": [
            {
                "game_id": str(row.get("game_id")),
                "spread_line": row.get("spread_line"),
                "atom": row.get("atom"),
                "home_cover_probability_smooth": row.get("home_cover_probability_smooth"),
                "home_cover_probability": row.get("home_cover_probability"),
                "side_changed": bool(row.get("side_changed")),
            }
            for row in touched
        ],
        "sides_changed": [str(row["game_id"]) for row in touched if row.get("side_changed")],
    }


def load_forecast_key_line_pick_read(forecast_dir: Path) -> dict[str, Any] | None:

    path = forecast_dir / KEY_LINE_PICK_READ_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _overrides_from_rows(rows: Iterable[Any]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("touched"):
            continue
        value = row.get("home_cover_probability")
        if value is None:
            continue
        overrides[str(row.get("game_id"))] = float(value)
    return overrides


def served_pick_overrides(forecast_dir: Path | None) -> dict[str, float] | None:

    if forecast_dir is None:
        return None
    sidecar = load_forecast_key_line_pick_read(forecast_dir)
    if sidecar is None or not sidecar.get("served"):
        return None
    games = sidecar.get("games")
    if not isinstance(games, list):
        return None
    return _overrides_from_rows(games)


def pick_overrides_from_metadata(metadata: Mapping[str, object]) -> dict[str, float] | None:

    block = metadata.get("key_line_pick_read")
    if not isinstance(block, Mapping) or not block.get("served"):
        return None
    touched = block.get("touched")
    if not isinstance(touched, list):
        return None
    return _overrides_from_rows(
        {**row, "touched": True} for row in touched if isinstance(row, Mapping)
    )


def load_pick_overrides(
    metadata: Mapping[str, object], forecast_dir: Path | None
) -> dict[str, float] | None:

    overrides = pick_overrides_from_metadata(metadata)
    if overrides is None:
        overrides = served_pick_overrides(forecast_dir)
    return overrides


def apply_pick_overrides(
    probabilities: pd.Series | np.ndarray | Iterable[float],
    game_ids: Iterable[Any],
    overrides: Mapping[str, float] | None,
) -> FloatArray:

    values = np.asarray(list(probabilities), dtype=float).copy()
    if overrides is None or not overrides:
        return values
    for index, game_id in enumerate(game_ids):
        override = overrides.get(str(game_id))
        if override is not None:
            values[index] = float(override)
    return values


def key_line_touched_games(metadata: Mapping[str, object]) -> frozenset[str]:

    overrides = pick_overrides_from_metadata(metadata)
    return frozenset(overrides) if overrides else frozenset()


__all__ = [
    "KEY_LINE_ATOMS",
    "KEY_LINE_PICK_READ_FILENAME",
    "KEY_LINE_PICK_READ_POLICY",
    "KEY_LINE_PICK_READ_SERVED",
    "KEY_LINE_STATUS_INAPPLICABLE",
    "KEY_LINE_STATUS_NOT_RUN",
    "KEY_LINE_STATUS_SERVED",
    "KEY_LINE_TOLERANCE",
    "KeyLineApplicability",
    "KeyLinePickRead",
    "ServedKeyLineRead",
    "apply_key_line_pick_read",
    "apply_key_line_pick_read_to_sweep",
    "apply_pick_overrides",
    "is_half_point_line",
    "key_line_applicability",
    "key_line_atom",
    "key_line_decision_probability",
    "key_line_mask",
    "key_line_metadata_block",
    "key_line_read_applicable",
    "key_line_sidecar",
    "key_line_touched_games",
    "load_forecast_key_line_pick_read",
    "load_pick_overrides",
    "pick_overrides_from_metadata",
    "served_pick_overrides",
]
