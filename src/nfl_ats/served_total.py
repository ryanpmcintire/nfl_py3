"""MOD-17 served-total provider: which total every published number uses.

``docs/tiebreaker.md``'s "one lattice, one margin, one total" contract (owner
mandate, 2026-09-05) requires exactly ONE served total feed the tiebreaker
centre, the panel, and the board assistant. Before this module, that one
total was always ``market_total + TOTALS_RESIDUAL_WEIGHT * totals_view.residual``
(the wave-2/wave-1 model, ``nfl_ats.tiebreaker.TOTALS_RESIDUAL_WEIGHT``,
``docs/totals_model.md``). This module adds a SECOND named method and an
explicit switch between them, so the choice of which one is served is one
constant, never scattered arithmetic.

Lane AC's research half (docs/mod17_joint_residual_model.md; artifact
``artifacts/mod17_joint_residual/20260905T160219Z/results.json``, registry
entry ``mod17_joint_residual_total_blend``) measured the joint margin/total
residual model's total-side output (``nfl_ats.joint_residual_model.UNION_FEATURES``,
114 columns, two-target ridge alpha=10.0, fit on
``data/processed/game_features_weak_stack.parquet``) beating the currently
served k=0.1 blend by **+0.00491 mae_improvement**, week-blocked bootstrap
95% **[-0.00662, +0.01646]**, ``probability_positive`` **0.791**, over 3,919
walk-forward games (260 week blocks). Per the closing-grounds taxonomy
(AGENTS.md): an interval crossing zero is never grounds to reject -- at this
evaluator's resolution that is the EXPECTED shape for a real small signal.
Per the project's EV decision rule (``probability_positive`` above 0.5
favours the candidate; a promotion bar is not a decision bar), this
measurement promotes the joint model's total output to the served number.
The disclosure that measurement itself carries, restated here because it
still applies to the served choice: the win rides on a small, shrinkage
-heavy blend weight (the joint model's own swept-minimum k is also 0.1) --
the RAW joint fit is measurably worse than the market as a point estimate
(out-of-sample R2 -0.0762 vs the wave-1 baseline's -0.0162), so this is not
evidence of a big edge, only the EV-favoured arm of two small ones. The old
blend stays served whenever the joint model cannot price a game (population
too thin, feature table absent) and is tracked prospectively against the
served number every week regardless (the ``totals_served_method`` challenger,
``artifacts/prospective/challengers.json``), so evidence keeps accruing at
no rotation-registry cost.

Two named, independently callable, pure methods
-------------------------------------------------
- :func:`served_total_blend_k01` -- today's rule, UNCHANGED:
  ``market_total + weight * totals_view.residual`` (or the market total
  alone with no view), byte-for-byte the same arithmetic
  ``nfl_ats.tiebreaker.build_report`` has always used. Hash-pinned in
  ``tests/test_served_total.py`` so this arm can never silently drift once
  it stops being the default.
- :func:`served_total_joint_residual` -- the joint model's own total-column
  prediction, blended at its own measured weight
  (:data:`JOINT_TOTAL_BLEND_WEIGHT`, also 0.1). Takes an already-fitted
  :class:`~nfl_ats.totals.TotalsView` (from :func:`joint_residual_total_view`,
  the I/O+fit half, kept separate so the blend arithmetic itself stays a
  pure function like its sibling above) and returns ``None`` when no view
  was fit -- the caller then falls back to the blend, the same "market/older
  -model-only" degrade every other totals arm in this project already uses.

:func:`served_total` is the dispatcher: it reads :data:`SERVED_TOTAL_METHOD`
(or an explicit override) and returns ``(value, method_actually_used)`` --
the second element can read back ``"blend_k01"`` even when
``SERVED_TOTAL_METHOD`` is ``"joint_residual"``, because a game the joint
model cannot price degrades to the blend rather than failing closed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.joint_residual_model import (
    JOINT_RIDGE_ALPHA,
    UNION_FEATURES,
    make_joint_estimator,
    realised_residual_frame,
)
from nfl_ats.totals import TotalsDataError, TotalsView, design_matrix

ServedTotalMethod = Literal["blend_k01", "joint_residual"]

#: Today's rule's blend weight. Duplicated from (rather than imported from)
#: ``nfl_ats.tiebreaker.TOTALS_RESIDUAL_WEIGHT`` -- this module is imported
#: BY ``tiebreaker.py``, so importing back would be circular. Production
#: callers (``nfl_ats.tiebreaker.build_report``) pass ``weight=
#: TOTALS_RESIDUAL_WEIGHT`` explicitly on every call so the two constants
#: cannot silently diverge; ``tests/test_served_total.py`` pins both this
#: value and their equality directly.
BLEND_K01_WEIGHT = 0.1

#: The joint model's own MAE-minimizing blend weight from the same k-sweep
#: ``nfl_ats.totals`` uses (docs/mod17_joint_residual_model.md Part 2,
#: ``total_side.joint_k`` in the 20260905T160219Z artifact) -- measured
#: separately from :data:`BLEND_K01_WEIGHT` above and happens to equal it.
JOINT_TOTAL_BLEND_WEIGHT = 0.1

#: The production feature table the joint model's own contract is frozen
#: against (docs/mod17_joint_residual_model.md "Frozen contract" -- the
#: production margin `weak_stack` table, a strict column superset of both
#: totals allowlists).
DEFAULT_JOINT_FEATURES_FILENAME = "game_features_weak_stack.parquet"

#: Which method is SERVED -- the one total every published number uses
#: (tiebreaker centre, panel, board assistant; docs/tiebreaker.md "one
#: lattice, one margin, one total"). Measured justification, 2026-09-05:
#: lane AC's joint-total screen (artifact stamp 20260905T160219Z,
#: ``artifacts/mod17_joint_residual/20260905T160219Z/results.json``,
#: registry entry ``mod17_joint_residual_total_blend``) found the joint
#: model's total output beats the k=0.1 blend by +0.00491 mae_improvement,
#: week-blocked bootstrap 95% [-0.00662, +0.01646], probability_positive
#: 0.791, 3,919 games -- EV-favoured per the project's decision rule
#: (probability_positive > 0.5), so it is served. Flipping this back to
#: "blend_k01" restores today's rule everywhere without touching any other
#: code; :func:`served_total_joint_residual` already degrades to the blend
#: automatically whenever it cannot price a game, so this switch only
#: matters when a joint view WAS successfully fit.
SERVED_TOTAL_METHOD: ServedTotalMethod = "joint_residual"

#: Ordered exactly like ``nfl_ats.joint_residual_model``'s own
#: ``_TARGET_COLUMNS`` -- the joint estimator's second output column.
_JOINT_TARGET_COLUMNS: tuple[str, str] = ("margin_residual", "total_residual")


def apply_blend(market_total: float, totals_view: TotalsView | None, *, weight: float) -> float:
    """``market_total + weight * totals_view.residual``, or ``market_total``
    alone when ``totals_view`` is ``None`` -- the one blend formula both
    named methods below apply to their own (differently sourced) view."""

    if totals_view is None:
        return float(market_total)
    return float(market_total) + float(weight) * float(totals_view.residual)


def served_total_blend_k01(
    market_total: float, totals_view: TotalsView | None, *, weight: float = BLEND_K01_WEIGHT
) -> float:
    """Today's rule, unchanged: byte-identical to
    ``nfl_ats.tiebreaker.build_report``'s own arithmetic. Hash-pinned in
    ``tests/test_served_total.py``."""

    return apply_blend(market_total, totals_view, weight=weight)


def joint_residual_total_view(
    game_id: str,
    data_root: Path,
    *,
    features_path: Path | None = None,
    feature_columns: tuple[str, ...] = UNION_FEATURES,
    ridge_alpha: float = JOINT_RIDGE_ALPHA,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> TotalsView | None:
    """The joint model's raw (unblended) total residual for ONE upcoming
    game, fit walk-forward on the union feature set.

    Mirrors ``nfl_ats.totals_wave2.model_total_view_wave2``'s contract and
    (season, week)-block cutoff discipline line for line: training is every
    population game strictly before the target's ``(season, week)``, so the
    number served for a live game is produced the same way the research
    screen's own walk-forward graded it. Loads
    ``data/processed/game_features_weak_stack.parquet`` by default -- the
    joint model's own frozen population table (docs/mod17_joint_residual_model.md)
    -- the same way ``nfl_ats.tiebreaker`` already loads its wave-2 model
    view from ``game_features_pbp.parquet``.

    Returns ``None`` -- never a silent substitution -- when the feature
    table is absent, carries no row for ``game_id``, has no market total for
    it, or fewer than ``min_train_games`` prior games exist; the caller then
    falls back to :func:`served_total_blend_k01`. Raises
    (``nfl_ats.totals.TotalsDataError``/``ValueError``/its subclasses) on a
    genuine data-contract violation, exactly like
    ``model_total_view_wave2`` -- the caller (``nfl_ats.tiebreaker.tiebreaker_report``)
    catches those the same way it already catches wave 2's.
    """

    path = (
        features_path
        if features_path is not None
        else (data_root / "processed" / DEFAULT_JOINT_FEATURES_FILENAME)
    )
    if not path.is_file():
        return None
    features = pd.read_parquet(path)
    rows = features.loc[features["game_id"].astype(str).eq(game_id)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    if pd.isna(row.get("total_line")):
        return None
    season, week = int(row["season"]), int(row["week"])

    population = realised_residual_frame(features, feature_columns=feature_columns)
    prior = population.loc[
        (population["season"] < season)
        | ((population["season"] == season) & (population["week"] < week))
    ]
    if len(prior) < min_train_games:
        return None

    estimator = make_joint_estimator(ridge_alpha=ridge_alpha)
    estimator.fit(
        design_matrix(prior, feature_columns),
        prior.loc[:, list(_JOINT_TARGET_COLUMNS)].astype(float).to_numpy(),
    )
    target_design = design_matrix(rows.iloc[[0]], feature_columns)
    predicted = np.asarray(estimator.predict(target_design), dtype=float).reshape(1, -1)
    total_index = _JOINT_TARGET_COLUMNS.index("total_residual")
    residual = float(predicted[0, total_index])
    market_total = float(row["total_line"])
    return TotalsView(
        predicted_total=market_total + residual,
        market_total=market_total,
        residual=residual,
        train_games=len(prior),
        source=(
            f"joint residual ridge (union {len(feature_columns)} cols, alpha={ridge_alpha:g}) "
            f"trained on {len(prior)} games before {season} week {week}"
        ),
    )


def served_total_joint_residual(
    market_total: float, joint_view: TotalsView | None, *, weight: float = JOINT_TOTAL_BLEND_WEIGHT
) -> float | None:
    """``joint_view``'s own residual blended at :data:`JOINT_TOTAL_BLEND_WEIGHT`
    (measured swept minimum, docs/mod17_joint_residual_model.md), or ``None``
    when no joint view was fit for this game -- the caller then falls back to
    :func:`served_total_blend_k01`."""

    if joint_view is None:
        return None
    return apply_blend(market_total, joint_view, weight=weight)


def served_total(
    method: ServedTotalMethod,
    *,
    market_total: float,
    blend_view: TotalsView | None,
    joint_view: TotalsView | None,
    blend_weight: float = BLEND_K01_WEIGHT,
    joint_weight: float = JOINT_TOTAL_BLEND_WEIGHT,
) -> tuple[float, ServedTotalMethod]:
    """Dispatch to the named method; returns ``(served_total, method_used)``.

    ``method_used`` can read back ``"blend_k01"`` even when ``method`` is
    ``"joint_residual"``: a game the joint model cannot price (no union
    feature row, too little walk-forward history) degrades to the blend
    automatically, the same "market/older-model-only" fallback every other
    totals arm in this project already uses -- never a hard failure over a
    missing optional input.
    """

    if method == "joint_residual":
        value = served_total_joint_residual(market_total, joint_view, weight=joint_weight)
        if value is not None:
            return value, "joint_residual"
        return served_total_blend_k01(market_total, blend_view, weight=blend_weight), "blend_k01"
    if method == "blend_k01":
        return served_total_blend_k01(market_total, blend_view, weight=blend_weight), "blend_k01"
    raise ValueError(f"Unknown served-total method: {method!r}")


__all__ = [
    "BLEND_K01_WEIGHT",
    "DEFAULT_JOINT_FEATURES_FILENAME",
    "JOINT_TOTAL_BLEND_WEIGHT",
    "SERVED_TOTAL_METHOD",
    "ServedTotalMethod",
    "TotalsDataError",
    "apply_blend",
    "joint_residual_total_view",
    "served_total",
    "served_total_blend_k01",
    "served_total_joint_residual",
]
