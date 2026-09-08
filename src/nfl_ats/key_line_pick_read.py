"""The key-line pick read: the served side on lines quoted exactly on 3 or 7.

MOD-18 lane T (docs/key_line_lattice.md, ``artifacts/research/laneT/``)
measured that serving lane K's mass-preserving discrete read for the PICK
only where the Tuesday opener line sits exactly ON a key atom -- 3 or 7,
either sign -- gains +0.200 accuracy points through the played three-member
card (95% [-0.271, +0.726], ``probability_positive`` 0.7426; cell
``kl1_kl1b_overall_card``) and +0.200 standalone at the opener
(``probability_positive`` 0.688; ``kl1_kl1b_overall_standalone``), touching
272 of 1,537 archive games. The coordinator's expected-value decision
(AGENTS.md, "A promotion bar is not a decision bar") serves that arm, KL1b,
from the 2026-09-08 Week 1 lock (docs/key_line_pick_read.md).

The mechanism, not a threshold (AGENTS.md, "No unexplained threshold flips"):
a discrete read differs from a smooth read at exactly one place -- the
integer the football piles up on. When the quoted line SITS on such an atom
the atom's weight is the push and the cover / loss split either side of it
is decided by mass the smooth read spreads across a continuum; between the
atoms the push atom is empty and the smooth read was already calibrated.
So the discrete read serves the side where the line is on 3 or 7, and the
smooth ``gaussian_median`` read serves it everywhere else.

What changes on the card: ONLY the two-way ``home_cover_probability`` of a
touched game (and therefore its pick, ``pick_probability`` and decision
columns). It is applied AFTER the served home-side offset -- the point the
atoms are tilted to is the offset-corrected centre plus the residual
location, exactly the served point the discrete push read already uses --
and the three-way push split, which the discrete read already serves on
every game (docs/discrete_push_read.md), is untouched. The decision number
is lane T's, bit for bit: ``cover + push / 2``
(:attr:`~nfl_ats.mass_preserving_lattice.MassPreservingRead.home_cover_probability`),
the frozen research convention, never a re-tuned variant.

Every module that refits or replays the served card (the late-week
refresh, ``card_refit``, the spread explorer, the mapping-incumbent
recorders, the board's reproduction check) reproduces the served
probability on touched games through the per-game override helpers below
(:func:`served_pick_overrides` / :func:`pick_overrides_from_metadata` /
:func:`load_pick_overrides`), the same shape ``home_side_location``'s
``served_center_offsets`` / ``center_offsets_from_metadata`` use, so a
pre-promotion card (no sidecar, no metadata block) behaves exactly as today.
"""

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

#: Flip to ``False`` to serve the smooth two-way read on every game again.
#: The sidecar carries both reads per game whenever the policy is served,
#: so the paired challenger keeps recording either way.
KEY_LINE_PICK_READ_SERVED = True
#: Named served policy: lane T's KL1b (atoms 3 and 7, band 2.5, 200-game
#: floor, the frozen MP1 construction, ``cover + push / 2``).
KEY_LINE_PICK_READ_POLICY = "key_line_pick_read_v1"
#: Sidecar written next to ``predictions.csv`` with both reads per game.
KEY_LINE_PICK_READ_FILENAME = "key_line_pick_read.json"
#: The atoms the served side is read off the lattice at, in absolute points:
#: lane T's predeclared sibling KL1b (3 and 7 only). Declared here once;
#: every caller imports it, nobody re-types it.
KEY_LINE_ATOMS: tuple[float, ...] = (3.0, 7.0)
#: Exact-match tolerance: a quarter-point line (6.75, 7.25) or a half-point
#: line is never on an atom.
KEY_LINE_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# The atom selection and the decision number (lane T's definitions)
# ---------------------------------------------------------------------------


def key_line_atom(line: float, atoms: Iterable[float] = KEY_LINE_ATOMS) -> float | None:
    """The atom ``line`` sits exactly on (``abs(line)`` within 1e-9), else ``None``."""

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
    """True where the quoted line sits exactly ON one of ``atoms`` (either sign).

    Lane T's ``key_line_mask`` (``scripts/key_line_lattice_opener_eval.py``),
    reproduced: ``abs(line)`` against each atom within 1e-9. Half-point and
    quarter-point lines never match; a missing line never matches.
    """

    size = np.abs(pd.to_numeric(pd.Series(list(lines)), errors="coerce").to_numpy(dtype=float))
    keys = np.asarray(list(atoms), dtype=float)
    if keys.size == 0:
        return np.zeros(size.shape, dtype=bool)
    hits = np.abs(size[:, None] - keys[None, :]) < KEY_LINE_TOLERANCE
    return np.asarray(np.any(hits, axis=1) & np.isfinite(size), dtype=bool)


def key_line_decision_probability(read: MassPreservingRead) -> float:
    """Lane T's per-game decision number: ``cover + push / 2``.

    That is :attr:`MassPreservingRead.home_cover_probability`, the frozen
    research convention lane K and lane T both scored (measured: lane T's
    ``scored.parquet`` ``p_MP1`` equals ``cover_MP1 + 0.5 * push_MP1``
    exactly on all 1,537 rows, and the push-renormalised
    ``cover / (cover + loss)`` differs from it by up to 0.018). Serving
    anything else would not be the measured arm.
    """

    return float(read.home_cover_probability)


# ---------------------------------------------------------------------------
# Serving: the policy for one week and the per-game record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServedKeyLineRead:
    """Both two-way reads for one served game, so the paired record never refits."""

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "spread_line": self.line,
            "point": self.point,
            "atom": self.atom,
            "touched": self.touched,
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
    """The served policy for one target week: the week's walk-forward
    lattice reader (the same one the discrete push read serves) and the
    declared atom set."""

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
    """``forecasts`` with ``home_cover_probability`` read off the lattice on key lines.

    ``forecasts`` is a ``MarginModel.predict`` frame aligned row-for-row
    with ``games`` (which supplies ``game_id`` and ``spread_line``), already
    carrying the served home-side offset in ``predicted_margin``; the served
    point is ``predicted_margin`` plus the residual location of
    ``probability_method`` -- exactly the point the discrete push read tilts
    to, so the pick and the card's push chance can never disagree on the
    lattice. Only ``home_cover_probability`` changes, and only on games whose
    ``spread_line`` sits exactly on one of ``policy.atoms``; every other
    column and every other game is returned untouched. ``log``, when given,
    receives one :class:`ServedKeyLineRead` per game (touched or not) with
    the smooth read it replaced or kept.
    """

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
    """A ``MarginModel.line_sweep`` frame with the two-way
    ``home_cover_probability`` / ``pick_probability`` read off the lattice at
    every ALTERNATIVE line that sits on an atom -- the served policy applied
    at each hypothetical line, so the sweep's line-0 row equals the card's
    served number and the flip-line scan reads the policy, not a mix.
    Rows off the atoms are untouched. ``points_by_game`` is game_id -> served
    point (the same map the discrete sweep uses); ``quoted_lines_by_game`` is
    game_id -> the game's own quoted line, which conditions the distribution
    at an alternative line (only the settlement threshold moves, exactly as
    the discrete sweep does), so at offset zero the read is the card's own."""

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
    # The sweep's own derived columns, recomputed on the touched rows only
    # with ``MarginModel.line_sweep``'s formulas, so untouched rows stay
    # bit-for-bit what the sweep produced.
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


# ---------------------------------------------------------------------------
# Sidecar and metadata: the per-game served-probability override
# ---------------------------------------------------------------------------


def key_line_sidecar(
    policy: KeyLinePickRead | None,
    log: Mapping[str, ServedKeyLineRead],
    game_ids: Iterable[str],
    *,
    error: str | None = None,
) -> dict[str, Any]:
    """The ``key_line_pick_read.json`` payload: both reads per served game.

    ``served`` is ``False`` (with ``error``) when the policy could not be
    applied this week -- the smooth read was served on every game and the
    paired challenger has nothing paired to record.
    """

    games = [log[game_id].to_dict() for game_id in game_ids if game_id in log]
    return {
        "schema": "key_line_pick_read/1",
        "served": policy is not None and error is None,
        "policy": KEY_LINE_PICK_READ_POLICY,
        "atoms": list(policy.atoms if policy is not None else KEY_LINE_ATOMS),
        "challenger": "smooth_gaussian_median_every_line",
        "fit": policy.to_dict() if policy is not None else None,
        "error": error,
        "games": games,
    }


def key_line_metadata_block(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    """The ``metadata.json`` ``key_line_pick_read`` block (provenance, never
    reader text): the policy, the atoms, the touched games with both reads
    (so the served probability is rebuildable from metadata alone, the way
    ``center_offsets_from_metadata`` rebuilds the offset), and the sides
    that changed."""

    games = sidecar.get("games")
    rows = [row for row in games if isinstance(row, Mapping)] if isinstance(games, list) else []
    touched = [row for row in rows if row.get("touched")]
    return {
        "path": KEY_LINE_PICK_READ_FILENAME,
        "policy": sidecar.get("policy"),
        "served": bool(sidecar.get("served")),
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
    """The sidecar a served forecast wrote, or ``None`` for an older card."""

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
    """game_id -> served two-way probability for the games the key-line read
    touched, from a forecast's sidecar; ``None`` when the card was produced
    without the promotion (no sidecar, or not served).

    Every module that REFITS the active recipe to reproduce or extend the
    served card applies this AFTER ``predict`` so its probability matches the
    number the card actually played. Untouched games are absent from the
    map and keep the refit's own read.
    """

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
    """Per-game served probabilities rebuilt from a forecast's ``metadata.json``.

    ``margin-predict`` records every touched game with both reads under
    ``key_line_pick_read.touched``, so any module that has the card and its
    metadata (but not the forecast directory) can reproduce the served
    number without file access. ``None`` for a card produced without the
    promotion or one on which the policy was not served.
    """

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
    """Prefer the card's metadata, fall back to its sidecar, never refit."""

    overrides = pick_overrides_from_metadata(metadata)
    if overrides is None:
        overrides = served_pick_overrides(forecast_dir)
    return overrides


def apply_pick_overrides(
    probabilities: pd.Series | np.ndarray | Iterable[float],
    game_ids: Iterable[Any],
    overrides: Mapping[str, float] | None,
) -> FloatArray:
    """``probabilities`` with the served override substituted where a game is
    in ``overrides``; an untouched game or a ``None`` map returns the input
    values unchanged (as a float array)."""

    values = np.asarray(list(probabilities), dtype=float).copy()
    if overrides is None or not overrides:
        return values
    for index, game_id in enumerate(game_ids):
        override = overrides.get(str(game_id))
        if override is not None:
            values[index] = float(override)
    return values


def key_line_touched_games(metadata: Mapping[str, object]) -> frozenset[str]:
    """The game ids whose served side came from the key-line read, from the
    forecast's metadata block (empty for a pre-promotion card)."""

    overrides = pick_overrides_from_metadata(metadata)
    return frozenset(overrides) if overrides else frozenset()


__all__ = [
    "KEY_LINE_ATOMS",
    "KEY_LINE_PICK_READ_FILENAME",
    "KEY_LINE_PICK_READ_POLICY",
    "KEY_LINE_PICK_READ_SERVED",
    "KEY_LINE_TOLERANCE",
    "KeyLinePickRead",
    "ServedKeyLineRead",
    "apply_key_line_pick_read",
    "apply_key_line_pick_read_to_sweep",
    "apply_pick_overrides",
    "key_line_atom",
    "key_line_decision_probability",
    "key_line_mask",
    "key_line_metadata_block",
    "key_line_sidecar",
    "key_line_touched_games",
    "load_forecast_key_line_pick_read",
    "load_pick_overrides",
    "pick_overrides_from_metadata",
    "served_pick_overrides",
]
