"""One measurement harness for the 2026-09-01 registry reliability sweep (ORCH-D).

**Why this file exists.** ``registry/weak_signals.json`` holds 543 NFL
``accuracy_points`` entries and 365 of them carry ``reliability: null``
(measured 2026-09-01 from the registry itself). Reliability is one of only
TWO admissible grounds for ever closing a line of work -- AGENTS.md: "wrong
sign, or the trait has no split-half reliability" -- so a null there leaves
that ground neither usable nor rulable-out for two thirds of the pile. This
module is what a dozen group scripts import so that every one of those 365
numbers is produced by the SAME estimator, the SAME seed, and the SAME
units, rather than a dozen near-miss reimplementations that cannot be
compared to one another afterwards.

**Binding taxonomy, owned verbatim (AGENTS.md / CLAUDE.md).** An interval or
CI that contains zero is NEVER grounds to reject, fail, or close an
experiment. At this evaluator's ~2-point resolution "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line
of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is ``unresolved_below_power``; report ``probability_positive``, never
"contains zero". Nothing in this module closes anything: it measures, and a
low number is a CANDIDATE for the reliability ground, never the closure
itself.

**Three quantities, never interchangeable.** They all land on the same
[-1, 1] correlation scale, which is exactly why each measurement carries its
:data:`METHOD_*` string into the registry audit note:

``METHOD_TRAIT``
    The construct is a continuous per-team-week quantity (an EPA rate, an
    injury value-lost, a rating divergence). Unit = team-season, halves =
    odd/even weeks. This is the quantity ``NO_SPLIT_HALF_RELIABILITY_MAX``
    (0.10) was calibrated against and the only one for which a low value is
    a candidate ``no_split_half_reliability`` ground.

``METHOD_VENUE``
    Same estimator, unit = venue-season, for venue-level constructs (roof
    state, altitude, playing surface, weather at the stadium). A venue trait
    can be perfectly reliable while telling you nothing about either team.

``METHOD_EXPOSURE``
    The cell is a per-game FLAG with no continuous parent trait (kickoff
    slot, short week, post-bye). What is measured is the flag's per-team-week
    EXPOSURE rate -- does the flag mark stable team/venue structure, or pure
    schedule churn? A low exposure reliability is NOT a closing ground: a
    genuinely random schedule quirk can still move covers, and closing on it
    would be the crossing-zero mistake wearing a different hat.

**What is deliberately NOT written to the registry.** A flag cell's
*effect* replication -- does the cover-rate gap hold up on held-out seasons
-- is a different question from any of the three above, and is reported by
:func:`half_season_replication` into the artifact and the sweep doc only. It
is decision-relevant and it is not a correlation, so it does not belong in a
correlation-scaled field that a validator reads as a closing ground.

**Never write an unmeasurable reliability as a number.**
``split_half_reliability`` returns NaN when too few units survive the
>=2-observations-per-half floor. Every function here returns an explicit
``status``; only :data:`STATUS_MEASURED` may be recorded, and
``nfl-ats weak-signals set-reliability`` refuses a non-finite value as a
second line of defence.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.cfb_qb_dependence import split_half_reliability  # noqa: E402

#: One seed for the whole sweep so any two groups' numbers are comparable and
#: every run is reproducible. Do not override it per group.
RELIABILITY_SEED = 20260901

#: Bootstrap draws, matching ``scripts/reliability_map.py``'s precedent.
N_BOOT = 4000

#: Below this many usable units (team-seasons / venue-seasons surviving the
#: >=2-per-half floor) a split-half correlation is not a sound measurement of
#: anything, and a near-zero value from it must never be read as "no
#: reliability". Reported as :data:`STATUS_INSUFFICIENT_UNITS` instead.
MIN_UNITS = 20

STATUS_MEASURED = "measured"
STATUS_INSUFFICIENT_UNITS = "insufficient_split_units"
STATUS_CONSTANT = "constant_or_all_missing"

METHOD_TRAIT = (
    "team-season odd/even-week split-half of the underlying continuous trait, "
    "Spearman-Brown corrected, block bootstrap over team-seasons "
    "(nfl_ats.cfb_qb_dependence.split_half_reliability, seed 20260901, 4000 draws)"
)
METHOD_VENUE = (
    "venue-season odd/even-week split-half of the underlying venue-level trait, "
    "Spearman-Brown corrected, block bootstrap over venue-seasons "
    "(nfl_ats.cfb_qb_dependence.split_half_reliability, seed 20260901, 4000 draws). "
    "A VENUE trait, not a team trait -- not comparable to a team-season reliability, "
    "and a low value is not a closing ground on its own"
)
METHOD_EXPOSURE = (
    "team-season odd/even-week split-half of the FLAG'S EXPOSURE RATE (share of the "
    "team-season's games carrying the flag), Spearman-Brown corrected, block bootstrap "
    "over team-seasons (nfl_ats.cfb_qb_dependence.split_half_reliability, seed 20260901, "
    "4000 draws). This is flag-exposure reliability, NOT the trait reliability that "
    "NO_SPLIT_HALF_RELIABILITY_MAX was calibrated against: a low value here is NOT an "
    "admissible no_split_half_reliability ground, because a schedule quirk with no stable "
    "team structure can still move covers"
)

_SB_FALLBACK_NOTE = (
    "; reported as the RAW half-length Pearson r with its bootstrap CI because the "
    "Spearman-Brown step-up leaves [-1, 1] at this negative correlation and is not "
    "reportable on the correlation scale"
)


def _spearman_brown(r: float) -> float:
    """Step a half-length correlation up to full length: ``2r / (1 + r)``."""

    if not math.isfinite(r) or r <= -1.0:
        return math.nan
    return (2.0 * r) / (1.0 + r)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _package(raw: dict[str, Any], *, method: str, min_units: int) -> dict[str, Any]:
    """Turn ``split_half_reliability`` output into ready-to-record fields.

    Emits ``reliability``/``reliability_low``/``reliability_high`` on a single
    coherent scale (the interval always contains the point estimate, which the
    ``set-reliability`` validator enforces), plus a ``status`` that says
    whether the number may be recorded at all.
    """

    n_units = int(raw.get("n_team_seasons") or 0)
    r = float(raw.get("pearson_r", math.nan))
    ci = list(raw.get("pearson_r_ci95", [math.nan, math.nan]))
    out: dict[str, Any] = {
        **raw,
        "n_units": n_units,
        "method": method,
        "reliability": None,
        "reliability_low": None,
        "reliability_high": None,
    }

    if not _finite(r):
        out["status"] = STATUS_CONSTANT if n_units >= min_units else STATUS_INSUFFICIENT_UNITS
        return out
    if n_units < min_units:
        out["status"] = STATUS_INSUFFICIENT_UNITS
        return out

    sb = _spearman_brown(r)
    sb_lo, sb_hi = _spearman_brown(float(ci[0])), _spearman_brown(float(ci[1]))
    if _finite(sb) and -1.0 <= sb <= 1.0:
        point, low, high = sb, sb_lo, sb_hi
        out["method"] = method
    else:
        # Spearman-Brown is unbounded below for r <~ -0.33; the raw
        # correlation is always on scale, so report that and say so.
        point, low, high = r, float(ci[0]), float(ci[1])
        out["method"] = method + _SB_FALLBACK_NOTE

    low = -1.0 if not _finite(low) else max(-1.0, min(1.0, low))
    high = 1.0 if not _finite(high) else max(-1.0, min(1.0, high))
    point = max(-1.0, min(1.0, point))
    low, high = min(low, point), max(high, point)

    out["status"] = STATUS_MEASURED
    out["reliability"] = point
    out["reliability_low"] = low
    out["reliability_high"] = high
    return out


def measure_reliability(
    long: pd.DataFrame,
    metric: str,
    *,
    method: str,
    unit_col: str = "team_id",
    seasons: tuple[int, int] | None = None,
    seed: int = RELIABILITY_SEED,
    n_boot: int = N_BOOT,
    min_units: int = MIN_UNITS,
) -> dict[str, Any]:
    """Split-half reliability of ``metric`` on ``long``, restricted to ``seasons``.

    ``long`` needs ``unit_col``, ``season``, ``week`` and ``metric``, one row
    per (unit, game). ``seasons`` is the registry cell's own inclusive season
    range -- measuring a cell's reliability on seasons it never touched would
    be a different construct's number wearing the cell's name.
    """

    required = {unit_col, "season", "week", metric}
    missing = required - set(long.columns)
    if missing:
        raise ValueError(f"reliability frame is missing columns: {sorted(missing)}")

    frame = long.loc[:, [unit_col, "season", "week", metric]].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame.dropna(subset=["season", "week"])
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    if seasons is not None:
        low, high = int(seasons[0]), int(seasons[1])
        frame = frame.loc[frame["season"].between(low, high)]
    frame = frame.rename(columns={unit_col: "team_id"})
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    values = frame[metric].dropna()
    if values.empty or values.nunique(dropna=True) <= 1:
        return _package(
            {
                "metric": metric,
                "n_team_seasons": 0,
                "pearson_r": math.nan,
                "pearson_r_ci95": [math.nan, math.nan],
                "spearman_rho": math.nan,
                "spearman_brown_full_length_reliability": math.nan,
                "probability_positive": math.nan,
                "seasons": None if seasons is None else [int(seasons[0]), int(seasons[1])],
                "unit": unit_col,
            },
            method=method,
            min_units=min_units,
        )

    raw = split_half_reliability(frame, metric, seed=seed, n_boot=n_boot)
    raw["seasons"] = None if seasons is None else [int(seasons[0]), int(seasons[1])]
    raw["unit"] = unit_col
    return _package(raw, method=method, min_units=min_units)


def game_flag_to_team_week(
    games: pd.DataFrame,
    flag: pd.Series,
    *,
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.DataFrame:
    """Explode a game-level boolean flag into a team-week exposure frame.

    Each game contributes two rows (one per side) carrying ``exposure`` in
    {0.0, 1.0}. Feed the result to :func:`measure_reliability` with
    ``metric="exposure"`` and :data:`METHOD_EXPOSURE`.
    """

    mask = flag.reindex(games.index).fillna(False).astype(bool).astype(float)
    pieces = []
    for team_col in (home_col, away_col):
        piece = games.loc[:, ["season", "week", team_col]].rename(columns={team_col: "team_id"})
        piece["exposure"] = mask.to_numpy()
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def half_season_replication(
    games: pd.DataFrame,
    flag: pd.Series,
    *,
    outcome_col: str,
    season_col: str = "season",
    min_flag_per_half: int = 30,
) -> dict[str, Any]:
    """Does the cell's cover-rate gap survive on held-out seasons?

    Splits the seasons odd/even, computes the flagged-minus-complement gap in
    PERCENTAGE POINTS inside each half, and reports both halves with their
    counts and whether they agree in sign.

    **Reported, never recorded.** This is effect replication, not a
    correlation, and it is emphatically not grounds to close anything: two
    halves of a small subset disagreeing in sign is the EXPECTED shape for a
    real-but-small effect at this resolution. It is here so a reader can see
    what the flag does out of half-sample alongside whatever reliability the
    construct's parent trait has.
    """

    frame = games.loc[:, [season_col, outcome_col]].copy()
    frame["flag"] = flag.reindex(games.index).fillna(False).astype(bool).to_numpy()
    frame[outcome_col] = pd.to_numeric(frame[outcome_col], errors="coerce")
    frame = frame.dropna(subset=[outcome_col, season_col])
    frame[season_col] = frame[season_col].astype(int)

    halves: dict[str, dict[str, Any]] = {}
    for label, parity in (("odd_seasons", 1), ("even_seasons", 0)):
        part = frame.loc[frame[season_col] % 2 == parity]
        flagged = part.loc[part["flag"], outcome_col]
        rest = part.loc[~part["flag"], outcome_col]
        gap = (
            100.0 * (float(flagged.mean()) - float(rest.mean()))
            if len(flagged) > 0 and len(rest) > 0
            else math.nan
        )
        halves[label] = {
            "n_flag": len(flagged),
            "n_complement": len(rest),
            "gap_pts": gap,
            "seasons": sorted(int(s) for s in part[season_col].unique()),
        }

    lo, hi = halves["odd_seasons"], halves["even_seasons"]
    both_measured = _finite(lo["gap_pts"]) and _finite(hi["gap_pts"])
    powered = min(lo["n_flag"], hi["n_flag"]) >= min_flag_per_half
    return {
        **halves,
        "sign_agreement": (
            bool(np.sign(lo["gap_pts"]) == np.sign(hi["gap_pts"])) if both_measured else None
        ),
        "status": (STATUS_MEASURED if (both_measured and powered) else STATUS_INSUFFICIENT_UNITS),
        "note": (
            "Effect replication across disjoint season halves, in percentage points. "
            "Reported only -- never written to the registry's reliability field and never "
            "a closing ground; a sign disagreement on a small subset is the expected "
            "shape for a real-but-small effect at this evaluator's ~2-point resolution."
        ),
    }


def battery_replication_correlation(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Across a battery's cells, do odd-season gaps predict even-season gaps?

    The battery-level companion to :func:`half_season_replication`: with K
    cells you get K paired points, and their correlation says whether the
    SCREEN is finding structure that holds up, independent of any one cell.
    Reported, not recorded.
    """

    pairs = [
        (float(c["odd_seasons"]["gap_pts"]), float(c["even_seasons"]["gap_pts"]))
        for c in cells.values()
        if _finite(c.get("odd_seasons", {}).get("gap_pts"))
        and _finite(c.get("even_seasons", {}).get("gap_pts"))
    ]
    if len(pairs) < 3:
        return {"n_cells": len(pairs), "pearson_r": None, "note": "fewer than 3 usable cells"}
    odd = np.array([p[0] for p in pairs])
    even = np.array([p[1] for p in pairs])
    if odd.std() == 0 or even.std() == 0:
        return {"n_cells": len(pairs), "pearson_r": None, "note": "a half has zero variance"}
    return {
        "n_cells": len(pairs),
        "pearson_r": float(np.corrcoef(odd, even)[0, 1]),
        "note": (
            "Correlation across the battery's cells between the odd-season and "
            "even-season cover-rate gaps. Reported, not recorded."
        ),
    }


def positive_control(
    long: pd.DataFrame,
    *,
    unit_col: str = "team_id",
    targets: tuple[float, ...] = (0.0, 0.2, 0.5, 0.8),
    seed: int = RELIABILITY_SEED,
    n_boot: int = 1000,
    min_units: int = MIN_UNITS,
) -> list[dict[str, Any]]:
    """Plant traits of KNOWN reliability on this group's own unit structure.

    A near-zero measured reliability is uninterpretable until the instrument
    has been shown able to find reliability that is genuinely there, at these
    unit counts and these per-unit observation counts. Every group script runs
    this on its own long frame and reports the recovery table beside its
    results; without it, a low number is ``unresolved``, not a finding.

    Each target ``t`` plants a per-unit level with variance ``t`` and
    per-observation noise with variance ``1 - t``, so a unit observed once per
    half would have split-half correlation ``t`` in expectation and more
    observations push the measured value above it -- the recovery table shows
    the actual ceiling this group's data supports.
    """

    rng = np.random.default_rng(seed)
    frame = long.loc[:, [unit_col, "season", "week"]].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame = frame.dropna(subset=[unit_col, "season", "week"])
    frame["season"] = frame["season"].astype(int)
    unit_key = frame[unit_col].astype(str) + "|" + frame["season"].astype(str)
    codes, _ = pd.factorize(unit_key)

    rows: list[dict[str, Any]] = []
    for target in targets:
        levels = rng.normal(0.0, math.sqrt(max(target, 0.0)), size=int(codes.max()) + 1)
        noise = rng.normal(0.0, math.sqrt(max(1.0 - target, 0.0)), size=len(frame))
        planted = frame.copy()
        planted["planted"] = levels[codes] + noise
        measured = measure_reliability(
            planted,
            "planted",
            method="positive control (planted trait)",
            unit_col=unit_col,
            seed=seed,
            n_boot=n_boot,
            min_units=min_units,
        )
        rows.append(
            {
                "planted_unit_variance_share": target,
                "status": measured["status"],
                "n_units": measured["n_units"],
                "recovered_reliability": measured["reliability"],
                "recovered_interval": [
                    measured["reliability_low"],
                    measured["reliability_high"],
                ],
            }
        )
    return rows
