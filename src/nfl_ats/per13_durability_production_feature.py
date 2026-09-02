"""PER-13 Stage 2: production's nine availability-derived injury columns, rebuilt
on a durability-augmented P(plays) (docs/per13_durability_stage2_on_production.md).

Stage 1 (``docs/per13_durability_prior.md``, ``nfl_ats.durability_prior``)
improved the out-of-season availability Brier from 0.09087 to 0.08332 by adding
six per-player durability columns on top of the designation cell. That is an
improvement to a *probability*, and the production ATS chain never consumes a
probability -- it consumes nine game-level injury columns, each of which
multiplies a per-player severity by a role share and sums over a team's visible
injury report. The severity IS the availability model's output
(``players._injury_unavailability``).

So the honest Stage 2 candidate is a REPLACEMENT, not an addition: swap the
severity input and rebuild the same nine columns with the same aggregation
code. This module supplies the swap and nothing else.

Three design properties, all frozen in the predeclaration and all tested:

1. **The offset is an odds-ratio update, never a recalibration.** The augmented
   probability is ``p e^d / (p e^d + 1 - p)`` where ``p`` is production's own
   number and ``d`` is a linear combination of the six durability columns. The
   refitted base coefficient and intercept of the fitting model are DISCARDED
   on purpose: keeping them would recalibrate production's probability, and the
   recurrence-hazard sibling already measured that recalibration accounts for
   essentially all of a naive comparison's apparent gain
   (``docs/recurrence_hazard_features.md``:102-104).
2. **``d = 0`` reproduces production exactly.** All six durability columns are
   0.0 for a player with no prior history, so a debutant -- and every player in
   a season before the columns' data support -- is scored by the designation
   cell alone and the candidate column is bit-identical to production's.
   ``p in {0, 1}`` is likewise fixed: a player listed Out stays out.
3. **Exactly nine columns move.** ``players._injury_unavailability`` is also
   read by the quarterback-availability branch (``players.py``:1348), which
   feeds ``qb_start_probability`` and ``qb_expected_epa_per_dropback``. Patching
   it directly would silently move eleven columns, not nine, so this module
   patches the two AGGREGATORS (``_injury_features``, ``_injury_value_features``)
   instead and leaves the quarterback branch reading production's own severity.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nfl_ats import players
from nfl_ats.constants import (
    PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS,
    PER13_DURABILITY_SUFFIX,
    PER13_DURABILITY_SWAPPED_BASE_COLUMNS,
)
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.durability_prior import (
    DURABILITY_COLUMNS,
    DurabilityHistory,
    clipped_logit,
    durability_prior_columns,
)

#: Stage 1's own training-floor constant (docs/per13_durability_prior.md sec 5).
MIN_TRAIN_ROWS = 2_000
#: Stage 1's estimator, at Stage 1's hyperparameters, unchanged.
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 1000
LOGISTIC_SOLVER = "lbfgs"

#: ``(season, week, gsis_id)``. Team is deliberately NOT part of the key: the
#: enrichment loop and the availability outcome builder apply team-abbreviation
#: aliases at different points, and a player is on exactly one roster in a
#: given week, so the three-part key is both unambiguous and alias-proof.
OffsetKey = tuple[int, int, str]

TRAINING_COLUMNS = (
    "gsis_id",
    "season",
    "week",
    "kickoff",
    "decision_cutoff",
    "position_group",
    "base_probability",
    "unavailable",
)
TARGET_COLUMNS = (
    "gsis_id",
    "season",
    "week",
    "decision_cutoff",
    "position_group",
)


def durability_column_name(column: str) -> str:
    """``diff_injury_offense_unavailability`` -> ``..._durability``."""

    return f"{column}{PER13_DURABILITY_SUFFIX}"


def augmented_unavailability(base: np.ndarray | float, offset: np.ndarray | float) -> np.ndarray:
    """Production's severity, updated by a log-odds ``offset``.

    ``p_dur = p e^d / (p e^d + 1 - p)``, written in the numerically safe form
    ``p / (p + (1 - p) e^-d)``. Exactly ``p`` when ``d == 0`` (no clipping, no
    floating-point drift, because the branch is taken explicitly), exactly 0
    when ``p == 0`` and exactly 1 when ``p == 1``, and NaN-preserving.
    """

    base_values = np.asarray(base, dtype=float)
    offset_values = np.broadcast_to(np.asarray(offset, dtype=float), base_values.shape)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        updated = base_values / (base_values + (1.0 - base_values) * np.exp(-offset_values))
    return np.asarray(np.where(offset_values == 0.0, base_values, updated), dtype=float)


# ---------------------------------------------------------------------------
# The offset model
# ---------------------------------------------------------------------------


def _fit_offset_coefficients(design: np.ndarray, outcome: np.ndarray) -> np.ndarray | None:
    """Unstandardised durability coefficients ``b_j = beta_j / scale_j``.

    ``design`` is ``[clipped_logit(p_base), x_1..x_6]``. Only the six durability
    coefficients are returned; the base-logit coefficient and the intercept are
    dropped, which is what makes the offset a pure durability contribution
    measured relative to a zero-history row.
    """

    if len(np.unique(outcome)) < 2:
        return None
    scaler = StandardScaler()
    standardised = scaler.fit_transform(design)
    model = LogisticRegression(C=LOGISTIC_C, solver=LOGISTIC_SOLVER, max_iter=LOGISTIC_MAX_ITER)
    model.fit(standardised, outcome)
    scale = np.asarray(scaler.scale_, dtype=float)[1:]
    return np.asarray(model.coef_[0][1:], dtype=float) / scale


def build_durability_offsets(
    history: DurabilityHistory,
    training: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    min_train_rows: int = MIN_TRAIN_ROWS,
) -> pd.DataFrame:
    """One log-odds offset per target player-game, walk-forward by week.

    For each ``(season, week)`` the model is fit on every training row whose
    game kickoff is strictly earlier than that week's earliest decision cutoff,
    so a scored row's offset can never be informed by its own game or by any
    game that had not finished by its decision time. The durability columns
    themselves are calibrated on strictly-prior SEASONS, exactly as Stage 1
    calibrated them, so nothing about the shrinkage strengths is refit inside a
    season either.

    Returns ``targets``' key columns plus ``offset``, ``prior_observations``,
    ``has_history`` and ``has_offset``.
    """

    require_columns(training, TRAINING_COLUMNS, "durability offset training rows")
    require_columns(targets, TARGET_COLUMNS, "durability offset target rows")
    if targets.empty:
        raise DataContractError("durability offset target rows are empty")

    training_aggregates = history.aggregates(training)
    target_aggregates = history.aggregates(targets)

    train_kickoff = pd.to_datetime(training["kickoff"], errors="coerce", utc=True)
    if train_kickoff.isna().any():
        raise DataContractError("durability offset training rows have unusable kickoffs")
    train_kickoff_values = train_kickoff.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    target_cutoff = pd.to_datetime(targets["decision_cutoff"], errors="coerce", utc=True)
    target_cutoff_values = target_cutoff.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")

    outcome = training["unavailable"].to_numpy(dtype=float)
    base_logit = clipped_logit(training["base_probability"].to_numpy(dtype=float))

    target_season = pd.to_numeric(targets["season"], errors="coerce").astype(int).to_numpy()
    target_week = pd.to_numeric(targets["week"], errors="coerce").astype(int).to_numpy()

    offsets = np.zeros(len(targets), dtype=float)
    fitted_weeks = 0
    for season in sorted({int(value) for value in target_season}):
        calibration = history.calibration(before_season=season)
        train_columns = durability_prior_columns(training_aggregates, calibration).to_numpy(
            dtype=float
        )
        target_columns = durability_prior_columns(target_aggregates, calibration).to_numpy(
            dtype=float
        )
        season_rows = target_season == season
        for week in sorted({int(value) for value in target_week[season_rows]}):
            week_rows = season_rows & (target_week == week)
            cutoff = target_cutoff_values[week_rows].min()
            train_rows = train_kickoff_values < cutoff
            if int(train_rows.sum()) < min_train_rows:
                continue
            design = np.column_stack([base_logit[train_rows], train_columns[train_rows]])
            coefficients = _fit_offset_coefficients(design, outcome[train_rows])
            if coefficients is None:
                continue
            fitted_weeks += 1
            offsets[week_rows] = target_columns[week_rows] @ coefficients

    prior_observations = target_aggregates["rate_n"].to_numpy(dtype=float)
    roster_observations = target_aggregates["reserve_n"].to_numpy(dtype=float)
    result = targets.loc[:, ["season", "week", "gsis_id"]].copy()
    result["offset"] = offsets
    result["prior_observations"] = prior_observations
    result["has_history"] = (prior_observations > 0) | (roster_observations > 0)
    result["has_offset"] = offsets != 0.0
    result.attrs["fitted_weeks"] = fitted_weeks
    return result


def offset_lookup(offsets: pd.DataFrame) -> dict[OffsetKey, float]:
    """``(season, week, gsis_id) -> offset``, dropping the exact zeros."""

    require_columns(offsets, ("season", "week", "gsis_id", "offset"), "durability offsets")
    non_zero = offsets.loc[offsets["offset"].ne(0.0)]
    return {
        (int(season), int(week), str(player)): float(offset)
        for season, week, player, offset in zip(
            non_zero["season"],
            non_zero["week"],
            non_zero["gsis_id"],
            non_zero["offset"],
            strict=True,
        )
    }


# ---------------------------------------------------------------------------
# The swap
# ---------------------------------------------------------------------------


def augmented_injury_frame(
    visible: pd.DataFrame | None, offsets: Mapping[OffsetKey, float]
) -> pd.DataFrame | None:
    """A copy of one team's visible injury rows with the severity swapped.

    ``players._injury_unavailability`` is called for the base, so the learned
    -cell / fixed-prior fallback is production's own, not a reimplementation of
    it. The result is written back into ``_unavailability``, which is the column
    that function reads first -- so the aggregators downstream need no changes.
    """

    if visible is None or visible.empty:
        return visible
    frame = visible.copy()
    base = np.array(
        [players._injury_unavailability(row) for _, row in frame.iterrows()], dtype=float
    )
    deltas = np.array(
        [
            offsets.get((int(season), int(week), str(player)), 0.0)
            for season, week, player in zip(
                frame["season"], frame["week"], frame["gsis_id"], strict=True
            )
        ],
        dtype=float,
    )
    frame["_unavailability"] = augmented_unavailability(base, deltas)
    return frame


@contextmanager
def durability_severity(offsets: Mapping[OffsetKey, float]) -> Iterator[None]:
    """Run ``enrich_with_player_features`` with the durability-augmented severity.

    Patches the two injury AGGREGATORS rather than
    ``players._injury_unavailability`` itself, on purpose: that function is also
    read by the quarterback-availability branch, and swapping it there would
    move ``diff_qb_start_probability`` and ``diff_qb_expected_epa_per_dropback``
    too -- eleven changed columns instead of the nine the predeclaration froze.
    Restores both originals on exit, including on an exception.
    """

    original_injury_features = players._injury_features
    original_injury_value_features = players._injury_value_features

    def patched_injury_features(
        visible: pd.DataFrame | None, roles: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        return original_injury_features(augmented_injury_frame(visible, offsets), roles)

    def patched_injury_value_features(
        visible: pd.DataFrame | None,
        roles: dict[str, dict[str, Any]],
        player_values: dict[str, dict[str, float]],
        prior_snaps: float,
        **keywords: Any,
    ) -> dict[str, float]:
        return original_injury_value_features(
            augmented_injury_frame(visible, offsets),
            roles,
            player_values,
            prior_snaps,
            **keywords,
        )

    players._injury_features = patched_injury_features  # type: ignore[assignment]
    players._injury_value_features = patched_injury_value_features  # type: ignore[assignment]
    try:
        yield
    finally:
        players._injury_features = original_injury_features  # type: ignore[assignment]
        players._injury_value_features = original_injury_value_features  # type: ignore[assignment]


def _keyed(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if "game_id" not in frame.columns:
        raise DataContractError("frame is missing the game_id join key")
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise DataContractError(f"frame is missing columns: {', '.join(missing)}")
    keyed = frame.loc[:, ["game_id", *columns]].copy()
    keyed["game_id"] = keyed["game_id"].astype(str)
    if keyed["game_id"].duplicated().any():
        raise DataContractError("frame carries duplicate game_id rows")
    return keyed.set_index("game_id")


def derive_durability_injury_columns(
    production: pd.DataFrame,
    rebuilt_offset: pd.DataFrame,
    rebuilt_baseline: pd.DataFrame,
) -> pd.DataFrame:
    """``(game_id, the nine *_durability columns)``.

    The candidate column is ``production + (rebuilt_offset - rebuilt_baseline)``
    rather than ``rebuilt_offset`` itself. That is additive against production by
    construction: a game whose players all carry a zero offset gets back
    production's own value bit-identically, and nothing the rebuild does
    differently (library drift, snapshot drift) can leak into the comparison.
    """

    base = _keyed(production, PER13_DURABILITY_SWAPPED_BASE_COLUMNS)
    with_offset = _keyed(rebuilt_offset, PER13_DURABILITY_SWAPPED_BASE_COLUMNS)
    without_offset = _keyed(rebuilt_baseline, PER13_DURABILITY_SWAPPED_BASE_COLUMNS)
    aligned_with = with_offset.reindex(base.index)
    aligned_without = without_offset.reindex(base.index)

    derived = pd.DataFrame(index=base.index)
    for column in PER13_DURABILITY_SWAPPED_BASE_COLUMNS:
        transport = (aligned_with[column] - aligned_without[column]).fillna(0.0)
        derived[durability_column_name(column)] = base[column] + transport
    return derived.reset_index()


def attach_durability_injury_columns(
    production: pd.DataFrame,
    rebuilt_offset: pd.DataFrame,
    rebuilt_baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Additively join the nine candidate columns onto the production table."""

    collisions = sorted(
        set(PER13_DURABILITY_ON_PRODUCTION_FEATURE_COLUMNS).intersection(production.columns)
    )
    if collisions:
        raise DataContractError(
            f"features already carry candidate columns: {', '.join(collisions)}"
        )
    derived = derive_durability_injury_columns(production, rebuilt_offset, rebuilt_baseline)
    merged = production.merge(
        derived,
        left_on=production["game_id"].astype(str),
        right_on="game_id",
        how="left",
        suffixes=("", "_durability_join"),
        validate="one_to_one",
    )
    merged = merged.drop(
        columns=[name for name in ("key_0", "game_id_durability_join") if name in merged.columns]
    )
    merged.index = production.index
    return merged


def reproduction_report(production: pd.DataFrame, rebuilt_baseline: pd.DataFrame) -> dict[str, Any]:
    """How exactly the zero-offset rebuild reproduces production's nine columns.

    Reported either way rather than asserted: a reproduction failure is a fact
    about snapshot or library drift that belongs in the write-up, and the
    transport construction in :func:`derive_durability_injury_columns` keeps the
    comparison additive against production regardless.
    """

    base = _keyed(production, PER13_DURABILITY_SWAPPED_BASE_COLUMNS)
    rebuilt = _keyed(rebuilt_baseline, PER13_DURABILITY_SWAPPED_BASE_COLUMNS).reindex(base.index)
    per_column: dict[str, float] = {}
    for column in PER13_DURABILITY_SWAPPED_BASE_COLUMNS:
        difference = (base[column] - rebuilt[column]).abs()
        per_column[column] = float(difference.max(skipna=True) or 0.0)
    return {
        "rows_compared": len(base),
        "rows_missing_from_rebuild": int(rebuilt[list(base.columns)].isna().all(axis=1).sum()),
        "max_absolute_difference_by_column": per_column,
        "bit_identical": all(value == 0.0 for value in per_column.values()),
    }


__all__ = [
    "DURABILITY_COLUMNS",
    "MIN_TRAIN_ROWS",
    "OffsetKey",
    "attach_durability_injury_columns",
    "augmented_injury_frame",
    "augmented_unavailability",
    "build_durability_offsets",
    "derive_durability_injury_columns",
    "durability_column_name",
    "durability_severity",
    "offset_lookup",
    "reproduction_report",
]
