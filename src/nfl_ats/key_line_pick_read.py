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

The atom-EQUALITY test cannot fire on the pool's lines. The key-number
mass matters MORE there, not less (2026-09-08)
-----------------------------------------------------------------------

Read this whole section before concluding anything about scope. It is the
one place a future session is most likely to draw the wrong conclusion.

Measured 2026-09-08 on the owner's own pool
(``data/splash/2026_week01_20260908_noon.json``, all sixteen Week 1 games of
the Splash Sports contest the card is played into): **every line the pool
quotes is a half point** -- 3.5, -2.5, 6.5, 8.5, 1.5, 9.5 and so on, with
**nine of the sixteen within half a point of 3** (six at |3.5|, three at
|2.5|). ``KEY_LINE_ATOMS`` is tested by exact equality, so it can never
match one of those lines and this module's override never fires on the
served card.

**That is a limitation of this narrow TEST, not of the mechanism, and it is
the opposite of a reason to stop reading the margin distribution
discretely.** Measured this session on 4,431 completed regular-season games
(``data/processed/game_features.parquet``, 2009-2025), the empirical
home-cover rate at a neutral point against a normal fitted to the same
games:

===========  ===========  ==========  =========================
Line         Empirical    Gaussian    Gaussian error
===========  ===========  ==========  =========================
2.5          50.85%       48.71%      2.14 points TOO LOW
3.5          43.04%       45.99%      2.95 points too high
6.5          35.32%       37.98%      2.66 points too high
7.5          30.92%       35.40%      4.49 points too high
10.5         24.42%       28.11%      3.69 points too high
===========  ===========  ==========  =========================

Crossing the 3 atom from 2.5 to 3.5 the empirical cover chance falls
**7.81 points**; the smooth read says 2.72 -- it understates the cliff by
**2.87x** and errs in OPPOSITE directions on the two sides of the atom. At
2.5 the two reads straddle 0.5 outright, i.e. they pick opposite sides of a
neutral game. The reason is mechanical: 14.58% of finals land exactly on
|3|, and on a WHOLE-number line that mass is absorbed by the push, while on
a half-point line the entire block falls on ONE side of the number. The
discrete conditional distribution is therefore *more* decisive at the
pool's lines than at the ones this module was measured on.

So, precisely:

* **The multimodal premise is untouched and still correct** (AGENTS.md,
  "Football margins are multimodal, not Gaussian"). The numbers above are
  additional evidence FOR it at half-point lines. Nothing here may be cited
  as refuting it, and nothing here disables any discrete read: the served
  three-way split (``docs/discrete_push_read.md``,
  :func:`~nfl_ats.mass_preserving_lattice.serve_discrete_three_way`) keeps
  serving on every game, which is exactly what a half-point line needs.
* **Nothing here closes lane T.** Every lane T cell stays
  ``unresolved_below_power`` at ``probability_positive`` 0.7426 through the
  played card. The read remains correct machinery for a whole-number line,
  a registered paired challenger, and the way the 1,537-game archive -- which
  is graded on whole-number-capable lines, where the push is real -- is read.
* **The generalisation is the open direction, not a dropped idea**: price
  the served cover probability off the discrete conditional distribution at
  EVERY half-point line, not only where the line equals an atom. Stated,
  with these numbers, as a predeclared TODO in
  ``docs/key_line_pick_read.md`` (MOD-18 candidate C2). It is not
  implemented or measured here; another lane owns it.

:func:`key_line_read_applicable` is the atom-equality predicate, named once
and imported by every caller, and :func:`key_line_applicability` summarises
it for a week so the sidecar, the metadata block and the log can say **"the
atom test matched nothing, and why"** (:data:`KEY_LINE_STATUS_INAPPLICABLE`,
with the lattice fitted and recorded) rather than reporting the same bare
zero touched games as **"did not run"** (:data:`KEY_LINE_STATUS_NOT_RUN`,
no lattice, an ``error``). A reader and a future session must be able to
tell those apart.
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

#: The three states the served policy can be in for one week, written into
#: the sidecar and the metadata block as ``status``. They exist because two
#: of them otherwise look identical from outside -- both report zero touched
#: games -- and a reader has to be able to tell them apart:
#:
#: * ``served``      -- the lattice was fitted and at least one served line
#:                      sat on an atom, so the read decided that game's side.
#: * ``inapplicable`` -- the lattice was fitted and the atom-equality test
#:                      matched NO served line. On the owner's pool, whose
#:                      every quote is a half point, this is the expected
#:                      steady state; ``applicability.reason`` says so in
#:                      words. It says nothing about whether a discrete read
#:                      belongs at those lines -- see the module docstring,
#:                      where the measured answer is that it belongs more.
#: * ``not_run``     -- the policy could not be built at all (flag off, or no
#:                      lattice this week). ``error`` says why, ``fit`` is
#:                      null, and there is nothing to be applicable ABOUT.
KEY_LINE_STATUS_SERVED = "served"
KEY_LINE_STATUS_INAPPLICABLE = "inapplicable"
KEY_LINE_STATUS_NOT_RUN = "not_run"


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

    The vectorised form of :func:`key_line_read_applicable`, and equal to it
    element for element -- both go through :func:`key_line_atom`. Use this
    over a frame, that one over a single line, and neither over a rewrite.
    """

    size = np.abs(pd.to_numeric(pd.Series(list(lines)), errors="coerce").to_numpy(dtype=float))
    keys = np.asarray(list(atoms), dtype=float)
    if keys.size == 0:
        return np.zeros(size.shape, dtype=bool)
    hits = np.abs(size[:, None] - keys[None, :]) < KEY_LINE_TOLERANCE
    return np.asarray(np.any(hits, axis=1) & np.isfinite(size), dtype=bool)


def is_half_point_line(line: float | None) -> bool:
    """True when ``line`` is quoted at a half point (3.5, -2.5, 6.5 ...).

    A real NFL final margin is a whole number of points, so a half-point
    line can never be settled exactly and can never be a key number. The
    owner's pool quotes only these (see the module docstring).
    """

    if line is None:
        return False
    value = float(line)
    if not np.isfinite(value):
        return False
    return bool(abs((abs(value) % 1.0) - 0.5) < KEY_LINE_TOLERANCE)


def key_line_read_applicable(line: float | None, atoms: Iterable[float] = KEY_LINE_ATOMS) -> bool:
    """Whether THIS module's atom-equality test can fire on ``line``.

    The named predicate the served path skips on, so "the read did not move
    this game" is an answer with a reason rather than a silence. It is true
    only when the quoted line sits exactly on an atom (3 or 7, either sign);
    a half-point line -- every line the owner's pool quotes -- is false, and
    so is a whole number off the atoms and a missing line.

    Scope warning, and the module docstring has the numbers: false here
    means only that THIS narrow test cannot match, never that a discrete
    read of the margin distribution is inappropriate at that line. The
    opposite holds at a half point -- the 14.58% of finals landing exactly
    on |3| all fall on ONE side of a 2.5 or 3.5 line instead of being
    absorbed by a push, and the smooth read misses the resulting cliff by
    2.87x. The served three-way split stays discrete on every game, and
    generalising the SIDE read to every half-point line is the predeclared
    open direction (MOD-18 candidate C2, ``docs/key_line_pick_read.md``).
    Nothing about this predicate closes or refutes lane T.
    """

    if line is None:
        return False
    return key_line_atom(float(line), atoms) is not None


@dataclass(frozen=True)
class KeyLineApplicability:
    """Whether a week's SERVED lines could match the atom-equality test.

    Recorded in the sidecar and the metadata block so that a week in which
    the read touched nothing carries its reason. Counts are over the served
    games only, in the order the card lists them.

    ``applicable is False`` is a fact about this module's exact-match test,
    never a finding about the margin distribution: at the half-point lines
    that produce it the key-number mass is more decisive, not less (module
    docstring, with the measured table).
    """

    games: int
    lines_on_an_atom: int
    half_point_lines: int
    whole_number_lines: int
    atoms: tuple[float, ...] = KEY_LINE_ATOMS

    @property
    def applicable(self) -> bool:
        """True when at least one served line sat on an atom."""

        return self.lines_on_an_atom > 0

    @property
    def status(self) -> str:
        return KEY_LINE_STATUS_SERVED if self.applicable else KEY_LINE_STATUS_INAPPLICABLE

    @property
    def reason(self) -> str | None:
        """Why the atom test matched nothing, or ``None`` when it matched.

        Names the pool's own convention when every served line is a half
        point, because that is the case the owner's contest produces every
        week and the one a future session is most likely to misread as a
        broken policy -- and points at the open generalisation so it is not
        misread as a dropped idea either.
        """

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
    """Summarise :func:`key_line_read_applicable` over one week's served lines."""

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

    @property
    def inapplicable_reason(self) -> str | None:
        """Why this game's side was NOT read off the lattice, or ``None``.

        Per-game counterpart of :attr:`KeyLineApplicability.reason`: an
        untouched game says whether its line was a half point (the pool's
        own convention, which no final margin can land on) or a whole number
        that simply is not one of the declared atoms.
        """

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

    Where it applies: a game is touched only when
    :func:`key_line_read_applicable` is true of its served line. On a
    half-point line -- every line the owner's pool quotes -- it is false and
    the smooth read is kept for the SIDE, with
    :attr:`ServedKeyLineRead.inapplicable_reason` saying so per game and
    :func:`key_line_applicability` saying so for the week. That records the
    exact-match test not matching. It is NOT a finding that the smooth read
    is right there: measured, it misses the cliff across the 3 atom by 2.87x
    and picks the opposite side of a neutral game at 2.5 (module docstring).
    The three-way split on those same games stays discrete throughout, and
    generalising the side read to every half-point line is MOD-18 candidate
    C2. The multimodal premise and every lane T cell are unaffected.
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
    built this week -- the smooth read was served on every game and the
    paired challenger has nothing paired to record.

    ``status`` separates the two ways a week can show zero touched games
    (see :data:`KEY_LINE_STATUS_SERVED` and friends): ``not_run`` means
    there was no lattice to read, ``inapplicable`` means the lattice was
    fitted and no served line sat on an atom -- the expected steady state on
    the owner's half-point pool. ``applicability`` carries the counts and
    the reason in the second case and is ``None`` in the first, because a
    policy that never ran has nothing to be applicable about.
    """

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
    """The ``metadata.json`` ``key_line_pick_read`` block (provenance, never
    reader text): the policy, the atoms, the touched games with both reads
    (so the served probability is rebuildable from metadata alone, the way
    ``center_offsets_from_metadata`` rebuilds the offset), and the sides
    that changed.

    Carries ``status`` and ``applicability`` through from the sidecar so a
    card whose metadata is read without its forecast directory can still
    distinguish "the read did not apply to any line this week, and here is
    why" from "the read did not run at all"."""

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
