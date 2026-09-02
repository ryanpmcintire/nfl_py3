"""Per-player durability priors for the learned-availability model (PER-13).

The learned-availability model in :mod:`nfl_ats.availability` knows only this
week's designation: report status x practice status x position group, with
season-lagged hierarchical shrinkage. It does not know that *this* player has
been listed Questionable twenty-six times and played twenty-four of them, nor
that another has spent three of the last four Septembers on a reserve list.

This module builds that missing per-player history into six columns, under the
specification frozen in ``docs/per13_durability_prior.md`` (sections 4-6).
Three properties matter and are enforced here rather than left to the caller:

1. **Point in time by kickoff, not by season.** A row's outcome history is
   every earlier outcome of the same player whose game *kickoff* is strictly
   before that row's decision cutoff. A season or week filter would be wrong
   in both directions (it admits a Sunday game into a Thursday game's history
   in the same week, and it discards a completed prior week that a coarse
   filter rounds away). Roster history uses strictly earlier ``(season, week)``,
   which is the contract the player snapshot manifest itself declares
   (``"weekly_rosters": "strictly earlier season/week only"``).
2. **Every prior strength is derived, never hand-picked.** AGENTS.md's
   "underived constants are wrong" rule applies directly: the shrinkage
   strengths come from the data's own between-player variance, by the
   beta-binomial method of moments (Kleinman 1973) for rates and the
   DerSimonian-Laird / Efron-Morris (1975) moment estimator for residuals,
   and are refitted per fold on that fold's strictly-prior seasons.
3. **A player with no history is scored by the designation cell alone.** All
   six columns are exactly ``0.0`` when the prior history is empty, so the
   candidate design matrix degenerates to the baseline one for a debutant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError, require_columns

DURABILITY_PRIOR_VERSION = "v1"

#: The six candidate columns, in the frozen order of the predeclaration's table.
DURABILITY_COLUMNS: tuple[str, ...] = (
    "durability_residual",
    "durability_listed_active_residual",
    "durability_rate_logit_offset",
    "durability_log_observations",
    "roster_absence_rate_logit_offset",
    "roster_reserve_rate_logit_offset",
)

#: Weekly-roster status codes that mean "on a reserve list, not available".
#: This is the roster-status-volatility channel PER-13 names, and it is the
#: only place league suspensions (``SUS``) enter the feature set.
RESERVE_STATUSES = frozenset({"RES", "PUP", "SUS", "NFI", "EXE", "RSN", "RSR", "E01", "E14", "NON"})

#: Probabilities are clipped before any logit so that a player whose prior rate
#: shrinks to an endpoint cannot emit an infinite column.
LOGIT_CLIP = 0.002

#: Bounds on every derived prior strength. The floor keeps a degenerate fold
#: (no between-player variance detectable) from producing zero shrinkage; the
#: cap keeps a near-zero excess variance from shrinking every player to the
#: group mean and silently deleting the feature.
MIN_PRIOR_STRENGTH = 1.0
MAX_PRIOR_STRENGTH = 500.0

_POOLED = "__pooled__"

OUTCOME_HISTORY_COLUMNS = (
    "gsis_id",
    "season",
    "week",
    "kickoff",
    "position_group",
    "report_category",
    "unavailable",
    "cell_probability",
)
ROSTER_HISTORY_COLUMNS = (
    "gsis_id",
    "season",
    "week",
    "position_group",
    "status",
    "played",
    "snap_covered",
)


def clipped_logit(values: np.ndarray | float) -> np.ndarray:
    """Logit with the frozen endpoint clip."""

    array = np.asarray(values, dtype=float)
    clipped = np.clip(array, LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=float)


def beta_binomial_prior_strength(successes: np.ndarray, trials: np.ndarray) -> float:
    """Kleinman (1973) method-of-moments prior strength for a beta-binomial.

    ``M = m (1 - m) / s^2 - 1`` where ``m`` is the pooled rate and ``s^2`` the
    *excess* of the trial-weighted between-player variance over its binomial
    part. When the observed dispersion is no larger than binomial there is no
    player-level signal to keep, and the estimator is capped rather than
    allowed to run to infinity.
    """

    counts = np.asarray(trials, dtype=float)
    hits = np.asarray(successes, dtype=float)
    keep = counts > 0
    counts, hits = counts[keep], hits[keep]
    if counts.size < 2 or counts.sum() <= 0:
        return MAX_PRIOR_STRENGTH
    pooled = float(hits.sum() / counts.sum())
    if not 0.0 < pooled < 1.0:
        return MAX_PRIOR_STRENGTH
    rates = hits / counts
    weighted_variance = float(np.average((rates - pooled) ** 2, weights=counts))
    binomial_part = float(np.average(pooled * (1.0 - pooled) / counts, weights=counts))
    excess = weighted_variance - binomial_part
    if excess <= 0.0:
        return MAX_PRIOR_STRENGTH
    strength = pooled * (1.0 - pooled) / excess - 1.0
    return float(np.clip(strength, MIN_PRIOR_STRENGTH, MAX_PRIOR_STRENGTH))


def residual_prior_strength(residuals: np.ndarray, player_codes: np.ndarray) -> float:
    """Random-effects moment prior strength ``M = var_within / var_between``.

    The shrinkage weight ``n / (n + M)`` this implies is the Efron-Morris /
    DerSimonian-Laird moment estimator: ``var_within`` is the pooled
    within-player residual variance and ``var_between`` the trial-weighted
    variance of the per-player means with its sampling part removed.
    """

    values = np.asarray(residuals, dtype=float)
    codes = np.asarray(player_codes)
    if values.size < 2:
        return MAX_PRIOR_STRENGTH
    frame = pd.DataFrame({"player": codes, "residual": values, "square": values**2})
    grouped = frame.groupby("player", sort=False).agg(
        total=("residual", "sum"), squares=("square", "sum"), count=("residual", "size")
    )
    counts = grouped["count"].to_numpy(dtype=float)
    means = grouped["total"].to_numpy(dtype=float) / counts
    within_sums = grouped["squares"].to_numpy(dtype=float) - counts * means**2
    degrees = float((counts - 1.0)[counts >= 2].sum())
    if degrees <= 0.0:
        return MAX_PRIOR_STRENGTH
    var_within = float(within_sums.sum() / degrees)
    if var_within <= 0.0:
        return MIN_PRIOR_STRENGTH
    weighted_between = float(
        np.average((means - np.average(means, weights=counts)) ** 2, weights=counts)
    )
    var_between = weighted_between - var_within * float(np.average(1.0 / counts, weights=counts))
    if var_between <= 0.0:
        return MAX_PRIOR_STRENGTH
    return float(np.clip(var_within / var_between, MIN_PRIOR_STRENGTH, MAX_PRIOR_STRENGTH))


@dataclass(frozen=True)
class DurabilityCalibration:
    """Everything a fold needs to turn prior counts into the six columns.

    Fitted on one fold's strictly-prior seasons and applied unchanged to that
    fold's training and evaluation rows, so an evaluated row's columns never
    depend on its own season or any later one.
    """

    before_season: int
    residual_strength: float
    active_residual_strength: float
    rate_strength: dict[str, float]
    rate_group_rate: dict[str, float]
    absence_strength: dict[str, float]
    absence_group_rate: dict[str, float]
    reserve_strength: dict[str, float]
    reserve_group_rate: dict[str, float]
    training_rows: int = 0
    training_roster_rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": DURABILITY_PRIOR_VERSION,
            "before_season": self.before_season,
            "residual_strength": self.residual_strength,
            "active_residual_strength": self.active_residual_strength,
            "rate_strength": dict(self.rate_strength),
            "rate_group_rate": dict(self.rate_group_rate),
            "absence_strength": dict(self.absence_strength),
            "absence_group_rate": dict(self.absence_group_rate),
            "reserve_strength": dict(self.reserve_strength),
            "reserve_group_rate": dict(self.reserve_group_rate),
            "training_rows": self.training_rows,
            "training_roster_rows": self.training_roster_rows,
        }


def _group_rates_and_strengths(
    frame: pd.DataFrame, value_column: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-position-group pooled rate and beta-binomial prior strength."""

    rates: dict[str, float] = {}
    strengths: dict[str, float] = {}
    if frame.empty:
        return {_POOLED: 0.5}, {_POOLED: MAX_PRIOR_STRENGTH}
    per_player = frame.groupby(["position_group", "gsis_id"], observed=True)[value_column].agg(
        ["sum", "size"]
    )
    for group_name, block in per_player.groupby(level="position_group", observed=True):
        total = float(block["size"].sum())
        if total <= 0:
            continue
        rates[str(group_name)] = float(block["sum"].sum() / total)
        strengths[str(group_name)] = beta_binomial_prior_strength(
            block["sum"].to_numpy(dtype=float), block["size"].to_numpy(dtype=float)
        )
    pooled = per_player.groupby("gsis_id", observed=True).sum()
    rates[_POOLED] = float(pooled["sum"].sum() / max(float(pooled["size"].sum()), 1.0))
    strengths[_POOLED] = beta_binomial_prior_strength(
        pooled["sum"].to_numpy(dtype=float), pooled["size"].to_numpy(dtype=float)
    )
    return rates, strengths


@dataclass
class DurabilityHistory:
    """Point-in-time per-player outcome and roster history.

    ``outcomes`` is one row per visible player-game with the played/unavailable
    label and, where one exists, the incumbent model's own cell probability.
    ``rosters`` is one row per player-week with its status code and whether the
    player logged a snap that week. Neither frame is filtered here: the
    strictly-earlier boundary is applied per lookup, in :meth:`aggregates`.
    """

    outcomes: pd.DataFrame
    rosters: pd.DataFrame
    _outcome_index: dict[str, dict[str, np.ndarray]] = field(default_factory=dict, repr=False)
    _roster_index: dict[str, dict[str, np.ndarray]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        require_columns(self.outcomes, OUTCOME_HISTORY_COLUMNS, "durability outcome history")
        require_columns(self.rosters, ROSTER_HISTORY_COLUMNS, "durability roster history")
        outcomes = self.outcomes.loc[:, list(OUTCOME_HISTORY_COLUMNS)].copy()
        kickoff = pd.to_datetime(outcomes["kickoff"], errors="coerce", utc=True)
        if kickoff.isna().any():
            raise DataContractError("durability outcome history has unusable kickoff timestamps")
        outcomes["kickoff"] = kickoff.dt.tz_localize(None)
        outcomes["gsis_id"] = outcomes["gsis_id"].astype(str)
        outcomes["unavailable"] = pd.to_numeric(outcomes["unavailable"], errors="coerce").astype(
            float
        )
        outcomes["cell_probability"] = pd.to_numeric(
            outcomes["cell_probability"], errors="coerce"
        ).astype(float)
        outcomes["report_category"] = outcomes["report_category"].astype(str)
        outcomes["position_group"] = outcomes["position_group"].astype(str)
        outcomes = outcomes.sort_values(["gsis_id", "kickoff"]).reset_index(drop=True)
        self.outcomes = outcomes

        rosters = self.rosters.loc[:, list(ROSTER_HISTORY_COLUMNS)].copy()
        rosters["gsis_id"] = rosters["gsis_id"].astype(str)
        rosters["season"] = pd.to_numeric(rosters["season"], errors="coerce").astype(int)
        rosters["week"] = pd.to_numeric(rosters["week"], errors="coerce").astype(int)
        rosters["status"] = rosters["status"].astype(str).str.upper()
        rosters["position_group"] = rosters["position_group"].astype(str)
        rosters["played"] = rosters["played"].astype(bool)
        rosters["snap_covered"] = rosters["snap_covered"].astype(bool)
        rosters = rosters.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)
        self.rosters = rosters

        self._outcome_index = self._build_outcome_index()
        self._roster_index = self._build_roster_index()

    # -- indexing -------------------------------------------------------

    def _build_outcome_index(self) -> dict[str, dict[str, np.ndarray]]:
        index: dict[str, dict[str, np.ndarray]] = {}
        scored = self.outcomes["cell_probability"].notna().to_numpy()
        residual = np.where(
            scored,
            self.outcomes["unavailable"].to_numpy(dtype=float)
            - self.outcomes["cell_probability"].fillna(0.0).to_numpy(dtype=float),
            0.0,
        )
        listed_active = scored & (self.outcomes["report_category"].to_numpy() != "out")
        unavailable = self.outcomes["unavailable"].to_numpy(dtype=float)
        kickoffs = self.outcomes["kickoff"].to_numpy(dtype="datetime64[ns]")
        for player, block in self.outcomes.groupby(
            "gsis_id", sort=False, observed=True
        ).indices.items():
            positions = np.asarray(block)
            index[str(player)] = {
                "kickoff": kickoffs[positions],
                "rate_n": _prefix(np.ones(positions.size)),
                "rate_k": _prefix(unavailable[positions]),
                "resid_n": _prefix(scored[positions].astype(float)),
                "resid_sum": _prefix(np.where(scored[positions], residual[positions], 0.0)),
                "active_n": _prefix(listed_active[positions].astype(float)),
                "active_sum": _prefix(np.where(listed_active[positions], residual[positions], 0.0)),
            }
        return index

    def _build_roster_index(self) -> dict[str, dict[str, np.ndarray]]:
        index: dict[str, dict[str, np.ndarray]] = {}
        keys = self.rosters["season"].to_numpy(dtype=np.int64) * 100 + self.rosters[
            "week"
        ].to_numpy(dtype=np.int64)
        reserve = self.rosters["status"].isin(RESERVE_STATUSES).to_numpy().astype(float)
        covered = self.rosters["snap_covered"].to_numpy()
        absent = (covered & ~self.rosters["played"].to_numpy()).astype(float)
        for player, block in self.rosters.groupby(
            "gsis_id", sort=False, observed=True
        ).indices.items():
            positions = np.asarray(block)
            index[str(player)] = {
                "key": keys[positions],
                # Column 5's denominator is roster weeks in snap-covered
                # seasons only, because outside them "logged no snap" is not
                # observable. Column 6's denominator is every roster week from
                # 2009, the 16-season reach PER-13 names.
                "roster_n": _prefix(covered[positions].astype(float)),
                "roster_absent": _prefix(absent[positions]),
                "reserve_n": _prefix(np.ones(positions.size)),
                "reserve_k": _prefix(reserve[positions]),
            }
        return index

    # -- lookups --------------------------------------------------------

    def aggregates(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Strictly-prior counts and sums for each row of ``rows``.

        ``rows`` needs ``gsis_id``, ``season``, ``week``, ``decision_cutoff``
        and ``position_group``. Outcome history is bounded by kickoff against
        ``decision_cutoff``; roster history by strictly earlier
        ``(season, week)``.
        """

        require_columns(
            rows,
            ("gsis_id", "season", "week", "decision_cutoff", "position_group"),
            "durability target rows",
        )
        players = rows["gsis_id"].astype(str).to_numpy()
        cutoffs = pd.to_datetime(rows["decision_cutoff"], errors="coerce", utc=True)
        if cutoffs.isna().any():
            raise DataContractError("durability target rows have unusable decision cutoffs")
        cutoff_values = cutoffs.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
        roster_keys = (
            pd.to_numeric(rows["season"], errors="coerce").astype(np.int64) * 100
            + pd.to_numeric(rows["week"], errors="coerce").astype(np.int64)
        ).to_numpy()

        names = (
            "rate_n",
            "rate_k",
            "resid_n",
            "resid_sum",
            "active_n",
            "active_sum",
            "roster_n",
            "roster_absent",
            "reserve_n",
            "reserve_k",
        )
        output = {name: np.zeros(len(rows), dtype=float) for name in names}
        for position in range(len(rows)):
            player = players[position]
            outcome_block = self._outcome_index.get(player)
            if outcome_block is not None:
                stop = int(
                    np.searchsorted(outcome_block["kickoff"], cutoff_values[position], side="left")
                )
                for name in ("rate_n", "rate_k", "resid_n", "resid_sum", "active_n", "active_sum"):
                    output[name][position] = outcome_block[name][stop]
            roster_block = self._roster_index.get(player)
            if roster_block is not None:
                stop = int(np.searchsorted(roster_block["key"], roster_keys[position], side="left"))
                for name in ("roster_n", "roster_absent", "reserve_n", "reserve_k"):
                    output[name][position] = roster_block[name][stop]

        aggregates = pd.DataFrame(output, index=rows.index)
        aggregates.insert(0, "position_group", rows["position_group"].astype(str).to_numpy())
        return aggregates

    # -- calibration ----------------------------------------------------

    def calibration(self, *, before_season: int) -> DurabilityCalibration:
        """Fit every prior strength and group rate on seasons before ``before_season``."""

        outcomes = self.outcomes.loc[self.outcomes["season"].lt(before_season)]
        rosters = self.rosters.loc[self.rosters["season"].lt(before_season)]
        scored = outcomes.loc[outcomes["cell_probability"].notna()].copy()
        scored["residual"] = scored["unavailable"] - scored["cell_probability"]
        active = scored.loc[scored["report_category"].ne("out")]

        residual_strength = (
            residual_prior_strength(
                scored["residual"].to_numpy(dtype=float), scored["gsis_id"].to_numpy()
            )
            if not scored.empty
            else MAX_PRIOR_STRENGTH
        )
        active_strength = (
            residual_prior_strength(
                active["residual"].to_numpy(dtype=float), active["gsis_id"].to_numpy()
            )
            if not active.empty
            else MAX_PRIOR_STRENGTH
        )
        rate_rates, rate_strengths = _group_rates_and_strengths(
            outcomes.assign(value=outcomes["unavailable"]), "value"
        )
        covered = rosters.loc[rosters["snap_covered"]]
        roster_absence = covered.assign(value=(~covered["played"]).astype(float))
        absence_rates, absence_strengths = _group_rates_and_strengths(roster_absence, "value")
        roster_reserve = rosters.assign(
            value=rosters["status"].isin(RESERVE_STATUSES).astype(float)
        )
        reserve_rates, reserve_strengths = _group_rates_and_strengths(roster_reserve, "value")
        return DurabilityCalibration(
            before_season=int(before_season),
            residual_strength=residual_strength,
            active_residual_strength=active_strength,
            rate_strength=rate_strengths,
            rate_group_rate=rate_rates,
            absence_strength=absence_strengths,
            absence_group_rate=absence_rates,
            reserve_strength=reserve_strengths,
            reserve_group_rate=reserve_rates,
            training_rows=len(outcomes),
            training_roster_rows=len(rosters),
        )


def _prefix(values: np.ndarray) -> np.ndarray:
    """Cumulative sums with a leading zero, so ``out[k]`` covers the first k rows."""

    result = np.zeros(values.size + 1, dtype=float)
    np.cumsum(values, out=result[1:])
    return result


def _lookup(
    mapping: dict[str, float], groups: np.ndarray, default_key: str = _POOLED
) -> np.ndarray:
    fallback = mapping.get(default_key, 0.5)
    return np.array([float(mapping.get(str(name), fallback)) for name in groups], dtype=float)


def _shrunk_rate_offset(
    successes: np.ndarray,
    trials: np.ndarray,
    groups: np.ndarray,
    rates: dict[str, float],
    strengths: dict[str, float],
) -> np.ndarray:
    """Beta-binomial shrunken rate expressed as a logit offset from its group.

    Exactly zero when ``trials == 0``: the shrunken rate is then the group rate
    and the offset cancels, which is the "a debutant is scored by the
    designation cell alone" property the predeclaration froze.
    """

    group_rate = _lookup(rates, groups)
    strength = _lookup(strengths, groups, _POOLED)
    shrunk = (successes + strength * group_rate) / (trials + strength)
    return np.asarray(clipped_logit(shrunk) - clipped_logit(group_rate), dtype=float)


def durability_prior_columns(
    aggregates: pd.DataFrame, calibration: DurabilityCalibration
) -> pd.DataFrame:
    """Turn strictly-prior counts into the six frozen columns."""

    require_columns(
        aggregates,
        (
            "position_group",
            "rate_n",
            "rate_k",
            "resid_n",
            "resid_sum",
            "active_n",
            "active_sum",
            "roster_n",
            "roster_absent",
            "reserve_n",
            "reserve_k",
        ),
        "durability aggregates",
    )
    groups = aggregates["position_group"].astype(str).to_numpy()
    columns = {
        "durability_residual": aggregates["resid_sum"].to_numpy(dtype=float)
        / (aggregates["resid_n"].to_numpy(dtype=float) + calibration.residual_strength),
        "durability_listed_active_residual": aggregates["active_sum"].to_numpy(dtype=float)
        / (aggregates["active_n"].to_numpy(dtype=float) + calibration.active_residual_strength),
        "durability_rate_logit_offset": _shrunk_rate_offset(
            aggregates["rate_k"].to_numpy(dtype=float),
            aggregates["rate_n"].to_numpy(dtype=float),
            groups,
            calibration.rate_group_rate,
            calibration.rate_strength,
        ),
        "durability_log_observations": np.log1p(aggregates["rate_n"].to_numpy(dtype=float)),
        "roster_absence_rate_logit_offset": _shrunk_rate_offset(
            aggregates["roster_absent"].to_numpy(dtype=float),
            aggregates["roster_n"].to_numpy(dtype=float),
            groups,
            calibration.absence_group_rate,
            calibration.absence_strength,
        ),
        "roster_reserve_rate_logit_offset": _shrunk_rate_offset(
            aggregates["reserve_k"].to_numpy(dtype=float),
            aggregates["reserve_n"].to_numpy(dtype=float),
            groups,
            calibration.reserve_group_rate,
            calibration.reserve_strength,
        ),
    }
    frame = pd.DataFrame(columns, index=aggregates.index)
    return frame.loc[:, list(DURABILITY_COLUMNS)]


def attach_durability_prior(
    rows: pd.DataFrame, history: DurabilityHistory, calibration: DurabilityCalibration
) -> pd.DataFrame:
    """Append the six columns to ``rows`` without reordering or dropping any row."""

    aggregates = history.aggregates(rows)
    columns = durability_prior_columns(aggregates, calibration)
    overlap = sorted(set(columns.columns).intersection(rows.columns))
    if overlap:
        raise DataContractError(
            f"durability columns already present on the target rows: {', '.join(overlap)}"
        )
    result = pd.concat([rows, columns], axis=1)
    if len(result) != len(rows):
        raise DataContractError("attaching durability columns changed the row count")
    return result


def split_half_reliability(
    frame: pd.DataFrame, value_column: str, *, minimum_per_half: int = 10
) -> dict[str, float]:
    """Odd/even split-half reliability of a per-player trait, Spearman-Brown corrected.

    AGENTS.md makes trait reliability the decisive registry field, because zero
    reliability is one of only two admissible closing grounds. It is computed
    on the trait itself, not on any comparison outcome.
    """

    require_columns(frame, ("gsis_id", value_column), "split-half frame")
    working = frame.loc[:, ["gsis_id", value_column]].copy()
    working["half"] = working.groupby("gsis_id", sort=False).cumcount() % 2
    halves = working.groupby(["gsis_id", "half"], sort=False)[value_column].agg(["mean", "size"])
    wide = halves.unstack("half")
    if wide.shape[1] < 4:
        return {"players": 0.0, "correlation": float("nan"), "spearman_brown": float("nan")}
    means = pd.DataFrame(wide["mean"])
    sizes = pd.DataFrame(wide["size"])
    keep = (
        sizes.notna().all(axis=1)
        & means.notna().all(axis=1)
        & sizes.min(axis=1).ge(minimum_per_half)
    )
    subset = means.loc[keep]
    if len(subset) < 3:
        return {
            "players": float(len(subset)),
            "correlation": float("nan"),
            "spearman_brown": float("nan"),
        }
    correlation = float(np.corrcoef(subset.iloc[:, 0], subset.iloc[:, 1])[0, 1])
    brown = 2.0 * correlation / (1.0 + correlation) if correlation > -1.0 else float("nan")
    return {
        "players": float(len(subset)),
        "correlation": correlation,
        "spearman_brown": float(brown),
    }
