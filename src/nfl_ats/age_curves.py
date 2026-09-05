"""LEAD-58: snap-weighted career-age x position-group performance curves.

QUALITY infrastructure, no ATS direction. This module builds descriptive
performance curves against a player's CAREER-AGE axis -- ``years_exp`` from
the weekly-roster feed -- and NOT chronological age. There are no birth
dates anywhere in this repository's local data (``weekly_rosters`` columns:
``season, team, position, status, full_name, gsis_id, pfr_id, years_exp,
week, game_type``), so a chronological-age axis is not buildable locally.
``years_exp`` increments by exactly 1 in 99.98% of consecutive player-seasons
(measured on this snapshot), so it is a faithful, if coarser, stand-in for
career stage. ``data/raw/combine/*/combine.parquet`` carries a ``draft_year``
that could cross-check ``years_exp`` against a debut season, but it only
joins 63.8% of snap rows (reported by the LEAD-58 planning pass, not
independently re-measured here) and is not used by this builder -- it is a
cross-check candidate, not an input.

Inputs (all local, all gitignored under ``data/``, resolved to the newest
snapshot at build time and pinned by snapshot id in the output manifest):

  - ``data/players/raw/<snapshot>/snap_counts.parquet`` via
    :func:`nfl_ats.players.canonicalize_snaps` -- realized offense/defense/
    special-teams snaps per player-game, 2013-2025 regular season.
  - ``data/players/raw/<snapshot>/weekly_rosters.parquet`` via
    :func:`nfl_ats.players.canonicalize_rosters` -- supplies ``years_exp``
    (the career-age axis) and the stable PFR/GSIS crosswalk
    :func:`nfl_ats.players.attach_snap_player_ids` needs to resolve snap
    rows (keyed on PFR ids) onto a GSIS identity.
  - ``data/players/values/raw/<snapshot>/player_stats.parquet`` via
    :func:`nfl_ats.players.canonicalize_player_stats` -- weekly
    rushing/receiving EPA and defensive disruption counting stats, keyed on
    a GSIS-format ``player_id`` that matches snap/roster ``gsis_id`` values
    directly (verified on this snapshot: no PFR->GSIS crosswalk needed for
    this join).
  - ``data/pbp/raw/<snapshot>/season=<season>/plays.parquet`` (QB only) --
    supplies per-dropback EPA, since QB value is conventionally measured
    per dropback rather than per offensive snap.

Deliberately NOT used: ``data/processed/player_participation_ratings.parquet``
-- its 3-season smeared windows would blur exactly the year-over-year
resolution a career-age curve needs.

Position groups and the metric used for each (frozen module constants,
:data:`POSITION_GROUPS` / :data:`METRIC_BY_GROUP`; the snap-table
``position`` column is authoritative -- ``weekly_rosters`` position churn for
the same player, e.g. LB-vs-OLB, is not used to relabel a snap row):

  - QB: sum(EPA) over PBP rows with ``qb_dropback == 1``, divided by the
    matching dropback count -- EPA per dropback, keyed on
    ``passer_player_id``.
  - RB (RB, HB, FB), WR, TE: ``(rushing_epa + receiving_epa) /
    offense_snaps``.
  - EDGE (DE, OLB), DL (DT, NT, DL), LB (LB, ILB, MLB), CB (CB, DB), S (FS,
    SS, S): a defense-disruption composite (the same
    :data:`nfl_ats.players._DEFENSE_DISRUPTION_WEIGHTS` used by the
    injury-value feature: TFL x0.5, forced fumble x2, sack x1.5, QB hit
    x0.25, INT x4, pass defended x0.5) divided by ``defense_snaps``.
  - OL (T, G, C, OL, OT), K, P, LS: **no local performance metric.** These
    groups get a snap-VOLUME curve only (``primary_snaps``); ``raw_rate``,
    ``shrunk_rate``, and ``smoothed_rate`` are null and
    ``coverage_status == "no_local_metric"``. Nothing in this repo's local
    tables measures OL/K/P/LS quality directly.

A player-week that has snaps but no matching ``player_stats`` row
contributes a numerator of 0 over the FULL snap denominator for the
skill/defense metric families (the same "missing production is zero
production" convention the offense/defense state already uses elsewhere in
this repo) -- except for QB, where a week with QB snaps but no linked PBP
dropback row is EXCLUDED rather than forced to 0-over-snaps, because the
QB denominator is dropbacks, not offense snaps, and a QB week with zero
dropbacks (e.g. a holder-only appearance) has no natural dropback
denominator to force a zero over.

Unmapped snap-table positions (multi-position codes like ``C/G``, ``DT/D``;
0.19% of REG snap rows on this snapshot) are dropped and counted in the
returned diagnostics, never silently absorbed into a group.

Cross-sectional curve and shrinkage (:func:`cross_sectional_curve`,
:func:`shrink_cells`): per (position group, career age) cell, ``raw_rate =
sum(metric_numerator) / sum(metric_denominator)`` across every player-season
in that cell, snap-weighted. ``shrunk_rate`` is an empirical-Bayes pull
toward the group's snap-weighted grand mean,

    shrunk_a = (w_a * raw_a + k * grand) / (w_a + k),   w_a = snaps_a

with ``k = tau_within / tau_between`` estimated by the SAME method-of-moments
recipe as ``scripts/cfb_james_stein_unit_screen.py``'s James-Stein shrinkage
(``tau_within``: the pooled, snap-weighted variance of individual
player-season rates around their own age's ``raw_rate``; ``tau_between``:
the weighted variance of the age-level ``raw_rate``s around the grand mean,
minus the mean per-age sampling variance ``tau_within / w_a``, floored at a
small positive epsilon scaled to that group's own variance so a thin group
never divides by an absolute constant borrowed from a different metric's
units). No monotonicity is imposed anywhere in this module -- a real
dip or plateau is reported, never smoothed away by assumption (AGENTS.md's
era-magnitude convention, applied to age instead of era).

:func:`smooth_curve` additionally reports a snap-weighted, 3-point
(age-1, age, age+1) tricube-kernel local-linear smooth of ``raw_rate`` --
a second, independent look at curve shape alongside the empirical-Bayes
``shrunk_rate``, not a replacement for it.

Delta-method curve (:func:`delta_curve`): for a player whose age-``a`` and
age-``(a+1)`` cells both clear a 100-snap (or dropback) floor, ``delta_a =
rate(a+1) - rate(a)``, weighted by ``min(snaps_a, snaps_{a+1})``. Averaging
these WITHIN-PLAYER deltas removes cross-sectional survivorship bias (a
cross-sectional curve conflates "players get better with age" and "only the
players who got better stick around long enough to reach that age"); it
does NOT remove attrition-selection bias in which players survive long
enough to contribute a delta at all (documented, not fixed, here).
``cumulative_delta`` integrates ``mean_delta`` forward and backward from
each group's modal entry age (the career age with the most players in the
cross-sectional curve).

Split-half reliability (:func:`split_half_reliability`): two independent
schemes, per the closing-grounds taxonomy in AGENTS.md -- an interval or
CI containing zero is never grounds to close a line of work; only a
RESOLVED wrong sign or a measured ZERO reliability closes anything here.

  (a) odd-vs-even SEASON halves (bootstrap resamples SEASONS within each
      half -- the natural iid unit for a season-grain panel);
  (b) a single fixed-seed random split of all ``gsis_id``s into two player
      halves (bootstrap resamples PLAYERS within each half).

Both correlate the two halves' age-level ``raw_rate`` vectors across shared
career ages, weighted by ``min(snaps)`` at that age (Pearson AND Spearman),
Spearman-Brown corrected to full-length reliability, with a 2000-draw
percentile bootstrap CI and ``probability_positive`` -- never a bare
"contains zero" verdict. OL/K/P/LS carry no reliability rows: there is no
local rate to test the reliability of.

Point-in-time contract (:func:`build_career_age_panel`,
:func:`build_age_curves`): ``as_of_season`` filters every input source
(snaps, rosters, player stats, PBP) to ``season < as_of_season`` BEFORE any
aggregation, so a curve built as of season Y is bit-identical whether or not
season Y (or later) data exists on disk yet -- verified by
``tests/test_age_curves.py``'s leakage tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nfl_ats.players import (
    _DEFENSE_DISRUPTION_WEIGHTS,
    PLAYER_STATS_REQUIRED_COLUMNS,
    attach_snap_player_ids,
    canonicalize_player_stats,
    canonicalize_rosters,
    canonicalize_snaps,
    latest_player_snapshot,
    latest_player_value_snapshot,
)

AGE_CURVES_VERSION = "age-curves-v1"
CAREER_AGE_AXIS = "years_exp"

#: Frozen snap-table position -> position-group mapping. The snap table's own
#: ``position`` column is authoritative; a player's roster position in a
#: different week is never used to relabel a snap row.
POSITION_GROUPS: dict[str, str] = {
    "QB": "QB",
    "RB": "RB",
    "HB": "RB",
    "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "DE": "EDGE",
    "OLB": "EDGE",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",
    "LB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    "CB": "CB",
    "DB": "CB",
    "FS": "S",
    "SS": "S",
    "S": "S",
    "T": "OL",
    "G": "OL",
    "C": "OL",
    "OL": "OL",
    "OT": "OL",
    "K": "K",
    "P": "P",
    "LS": "LS",
}

#: One label per position group naming the metric definition used for its
#: cells; ``"no_local_metric"`` groups get a snap-volume curve only.
METRIC_BY_GROUP: dict[str, str] = {
    "QB": "epa_per_dropback",
    "RB": "epa_per_offense_snap",
    "WR": "epa_per_offense_snap",
    "TE": "epa_per_offense_snap",
    "EDGE": "disruption_per_defense_snap",
    "DL": "disruption_per_defense_snap",
    "LB": "disruption_per_defense_snap",
    "CB": "disruption_per_defense_snap",
    "S": "disruption_per_defense_snap",
    "OL": "no_local_metric",
    "K": "no_local_metric",
    "P": "no_local_metric",
    "LS": "no_local_metric",
}

NO_LOCAL_METRIC_GROUPS: frozenset[str] = frozenset({"OL", "K", "P", "LS"})
OFFENSE_SKILL_METRIC_GROUPS: frozenset[str] = frozenset({"RB", "WR", "TE"})
DEFENSE_METRIC_GROUPS: frozenset[str] = frozenset({"EDGE", "DL", "LB", "CB", "S"})
QB_METRIC_GROUP = "QB"
_SPECIAL_TEAMS_VOLUME_GROUPS: frozenset[str] = frozenset({"K", "P", "LS"})
_METRIC_GROUPS: tuple[str, ...] = (
    QB_METRIC_GROUP,
    *sorted(OFFENSE_SKILL_METRIC_GROUPS),
    *sorted(DEFENSE_METRIC_GROUPS),
)

DELTA_METHOD_SNAP_FLOOR = 100.0
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260905
_MIN_SHARED_AGES_FOR_RELIABILITY = 3
_MIN_VALID_BOOTSTRAP_DRAWS = 100


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------


def build_career_age_panel(
    snaps_raw: pd.DataFrame,
    rosters_raw: pd.DataFrame,
    stats_raw: pd.DataFrame,
    pbp_frames: Mapping[int, pd.DataFrame],
    *,
    as_of_season: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the player-week career-age panel and return it with diagnostics.

    ``pbp_frames`` maps season -> a play-by-play DataFrame carrying at least
    ``game_id, season_type, qb_dropback, passer_player_id, epa``; only
    seasons actually present are used (a season with no PBP coverage simply
    contributes no QB rows, never an error).

    Returns ``(panel, diagnostics)``. ``panel`` has one row per (player,
    game) snap row that resolved to a GSIS identity, a mapped position
    group, and a known career age, with columns ``gsis_id, season, week,
    team, position, pos_group, career_age, offense_snaps, defense_snaps,
    st_snaps, metric_numerator, metric_denominator, primary_snaps,
    coverage_status``. ``diagnostics`` counts every row dropped and why.
    """

    rosters = canonicalize_rosters(rosters_raw)
    snaps = canonicalize_snaps(snaps_raw)
    # Same "no stats supplied" convention as
    # nfl_ats.players.enrich_with_player_features: an empty, columnless
    # stats_raw (a caller with nothing to attach, e.g. an OL/K/P/LS-only
    # panel) is valid input, not a data-contract violation.
    stats = (
        canonicalize_player_stats(stats_raw)
        if len(stats_raw.columns)
        else pd.DataFrame(columns=PLAYER_STATS_REQUIRED_COLUMNS)
    )

    if as_of_season is not None:
        rosters = rosters.loc[rosters["season"] < as_of_season].copy()
        snaps = snaps.loc[snaps["season"] < as_of_season].copy()
        stats = stats.loc[stats["season"] < as_of_season].copy()
        pbp_frames = {
            season: frame for season, frame in pbp_frames.items() if season < as_of_season
        }

    linked = attach_snap_player_ids(snaps, rosters)
    n_snap_rows = len(linked)
    n_unlinked = int(linked["gsis_id"].isna().sum())
    linked = linked.loc[linked["gsis_id"].notna()].copy()

    linked["pos_group"] = linked["position"].map(POSITION_GROUPS)
    n_unmapped_position = int(linked["pos_group"].isna().sum())
    linked = linked.loc[linked["pos_group"].notna()].copy()

    years_exp = (
        rosters.groupby(["gsis_id", "season"], observed=True)["years_exp"]
        .max()
        .rename("career_age")
        .reset_index()
    )
    linked = linked.merge(years_exp, on=["gsis_id", "season"], how="left", validate="many_to_one")
    n_missing_career_age = int(linked["career_age"].isna().sum())
    linked = linked.loc[linked["career_age"].notna()].copy()
    linked["career_age"] = linked["career_age"].astype(int)

    linked["metric_numerator"] = np.nan
    linked["metric_denominator"] = np.nan

    # Every skill/defense row defaults to "0 production over the full snap
    # denominator" -- a matched player_stats row (below) then overrides just
    # the numerator where production was actually observed. This must hold
    # even when ``stats`` has zero rows at all (an OL/K/P/LS-only caller), not
    # only when a specific player-game is individually missing.
    #
    # IMPORTANT: every mask used AFTER a ``linked = linked.merge(...)`` call is
    # recomputed fresh from the POST-merge frame, never carried across the
    # reassignment. ``DataFrame.merge`` always returns a new frame with a
    # fresh ``RangeIndex`` (even for a row-preserving left join), so a mask
    # built against the pre-merge frame's (possibly non-contiguous, after the
    # ``.loc[...]`` filters above) index would silently misalign on the
    # post-merge frame's index and select the wrong rows -- caught by
    # comparing this function's real-data output against an independent
    # from-scratch join before this module shipped.
    skill_mask = linked["pos_group"].isin(OFFENSE_SKILL_METRIC_GROUPS)
    linked.loc[skill_mask, "metric_numerator"] = 0.0
    linked.loc[skill_mask, "metric_denominator"] = linked.loc[skill_mask, "offense_snaps"]
    if skill_mask.any() and not stats.empty:
        skill_stats = stats.loc[:, ["player_id", "game_id", "team", "rushing_epa", "receiving_epa"]]
        skill_stats = skill_stats.rename(columns={"player_id": "gsis_id"})
        skill_stats = skill_stats.assign(
            skill_epa=skill_stats["rushing_epa"] + skill_stats["receiving_epa"]
        )
        linked = linked.merge(
            skill_stats[["gsis_id", "game_id", "team", "skill_epa"]],
            on=["gsis_id", "game_id", "team"],
            how="left",
            validate="many_to_one",
        )
        matched = (
            linked["pos_group"].isin(OFFENSE_SKILL_METRIC_GROUPS) & linked["skill_epa"].notna()
        )
        linked.loc[matched, "metric_numerator"] = linked.loc[matched, "skill_epa"]
        linked = linked.drop(columns="skill_epa")

    defense_mask = linked["pos_group"].isin(DEFENSE_METRIC_GROUPS)
    linked.loc[defense_mask, "metric_numerator"] = 0.0
    linked.loc[defense_mask, "metric_denominator"] = linked.loc[defense_mask, "defense_snaps"]
    if defense_mask.any() and not stats.empty:
        defense_columns = [name for name, _ in _DEFENSE_DISRUPTION_WEIGHTS]
        defense_stats = stats.loc[:, ["player_id", "game_id", "team", *defense_columns]]
        defense_stats = defense_stats.rename(columns={"player_id": "gsis_id"})
        disruption = sum(
            weight * defense_stats[column] for column, weight in _DEFENSE_DISRUPTION_WEIGHTS
        )
        defense_stats = defense_stats.assign(defense_disruption=disruption)
        linked = linked.merge(
            defense_stats[["gsis_id", "game_id", "team", "defense_disruption"]],
            on=["gsis_id", "game_id", "team"],
            how="left",
            validate="many_to_one",
        )
        matched = (
            linked["pos_group"].isin(DEFENSE_METRIC_GROUPS) & linked["defense_disruption"].notna()
        )
        linked.loc[matched, "metric_numerator"] = linked.loc[matched, "defense_disruption"]
        linked = linked.drop(columns="defense_disruption")

    qb_mask = linked["pos_group"].eq(QB_METRIC_GROUP)
    if qb_mask.any() and pbp_frames:
        qb_agg = _qb_dropback_epa(pbp_frames)
        if not qb_agg.empty:
            linked = linked.merge(
                qb_agg,
                on=["gsis_id", "game_id", "season"],
                how="left",
                validate="many_to_one",
            )
            matched = linked["pos_group"].eq(QB_METRIC_GROUP)
            linked.loc[matched, "metric_numerator"] = linked.loc[matched, "qb_epa"]
            linked.loc[matched, "metric_denominator"] = linked.loc[matched, "qb_dropbacks"]
            linked = linked.drop(columns=["qb_epa", "qb_dropbacks"])

    linked["primary_snaps"] = np.where(
        linked["pos_group"].eq("OL"),
        linked["offense_snaps"],
        np.where(
            linked["pos_group"].isin(_SPECIAL_TEAMS_VOLUME_GROUPS),
            linked["st_snaps"],
            linked["metric_denominator"],
        ),
    )
    linked["coverage_status"] = np.where(
        linked["pos_group"].isin(NO_LOCAL_METRIC_GROUPS), "no_local_metric", "metric"
    )

    panel_columns = [
        "gsis_id",
        "season",
        "week",
        "team",
        "position",
        "pos_group",
        "career_age",
        "offense_snaps",
        "defense_snaps",
        "st_snaps",
        "metric_numerator",
        "metric_denominator",
        "primary_snaps",
        "coverage_status",
    ]
    panel = (
        linked.loc[:, panel_columns]
        .sort_values(["season", "week", "gsis_id"])
        .reset_index(drop=True)
    )

    diagnostics = {
        "snap_rows_total": n_snap_rows,
        "snap_rows_unlinked_to_gsis": n_unlinked,
        "gsis_match_rate": (1.0 - n_unlinked / n_snap_rows) if n_snap_rows else float("nan"),
        "snap_rows_unmapped_position": n_unmapped_position,
        "snap_rows_missing_career_age": n_missing_career_age,
        "snap_rows_in_panel": len(panel),
    }
    return panel, diagnostics


def _qb_dropback_epa(pbp_frames: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """Per (passer gsis_id, game, season) sum(EPA) and dropback count, REG only."""

    rows: list[pd.DataFrame] = []
    for season, frame in pbp_frames.items():
        reg = frame.loc[frame["season_type"] == "REG"]
        dropbacks = reg.loc[
            pd.to_numeric(reg["qb_dropback"], errors="coerce").eq(1.0)
            & reg["passer_player_id"].notna()
        ]
        if dropbacks.empty:
            continue
        grouped = (
            dropbacks.groupby(["game_id", "passer_player_id"], observed=True)["epa"]
            .agg(qb_epa="sum", qb_dropbacks="size")
            .reset_index()
        )
        grouped["season"] = season
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(columns=["gsis_id", "game_id", "season", "qb_epa", "qb_dropbacks"])
    combined = pd.concat(rows, ignore_index=True)
    return combined.rename(columns={"passer_player_id": "gsis_id"})


# ---------------------------------------------------------------------------
# Player-season cells and the cross-sectional curve
# ---------------------------------------------------------------------------


def player_age_cells(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse the player-week panel to one row per (player, group, season, age).

    ``metric_numerator``/``metric_denominator`` sum with ``min_count=1`` so a
    position group with no local metric (every panel row's numerator is NaN)
    aggregates to NaN, not to a false 0.0.
    """

    grouped = panel.groupby(["gsis_id", "pos_group", "season", "career_age"], observed=True)
    cells = grouped.agg(
        metric_numerator=("metric_numerator", lambda s: s.sum(min_count=1)),
        metric_denominator=("metric_denominator", lambda s: s.sum(min_count=1)),
        offense_snaps=("offense_snaps", "sum"),
        defense_snaps=("defense_snaps", "sum"),
        st_snaps=("st_snaps", "sum"),
        n_weeks=("week", "nunique"),
    ).reset_index()
    cells["primary_snaps"] = np.where(
        cells["pos_group"].eq("OL"),
        cells["offense_snaps"],
        np.where(
            cells["pos_group"].isin(_SPECIAL_TEAMS_VOLUME_GROUPS),
            cells["st_snaps"],
            cells["metric_denominator"],
        ),
    )
    return cells


def cross_sectional_curve(cells: pd.DataFrame) -> pd.DataFrame:
    """Per (position group, career age): the snap-weighted cross-sectional rate."""

    grouped = cells.groupby(["pos_group", "career_age"], observed=True)
    curve = grouped.agg(
        n_players=("gsis_id", "nunique"),
        n_player_seasons=("gsis_id", "size"),
        n_player_weeks=("n_weeks", "sum"),
        metric_numerator=("metric_numerator", lambda s: s.sum(min_count=1)),
        metric_denominator=("metric_denominator", lambda s: s.sum(min_count=1)),
        snaps=("primary_snaps", lambda s: s.sum(min_count=1)),
    ).reset_index()
    curve["raw_rate"] = curve["metric_numerator"] / curve["metric_denominator"]
    curve["coverage_status"] = np.where(
        curve["pos_group"].isin(NO_LOCAL_METRIC_GROUPS), "no_local_metric", "metric"
    )
    curve.loc[curve["coverage_status"].eq("no_local_metric"), "raw_rate"] = np.nan
    curve["sparse"] = curve["n_players"] < 5
    return curve.sort_values(["pos_group", "career_age"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Empirical-Bayes shrinkage and the local-linear smooth
# ---------------------------------------------------------------------------


def shrink_cells(curve: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Add ``shrunk_rate`` (and its ``shrinkage_k``/``grand_mean``/tau components).

    Method-of-moments empirical Bayes, the same recipe as
    ``scripts/cfb_james_stein_unit_screen.py`` (see module docstring for the
    derivation of ``k = tau_within / tau_between``). ``NO_LOCAL_METRIC_GROUPS``
    are left entirely null -- there is no rate to shrink.
    """

    result = curve.copy()
    for column in ("shrunk_rate", "shrinkage_k", "grand_mean", "tau_between", "tau_within"):
        result[column] = np.nan

    finite_rate = result["raw_rate"].notna() & np.isfinite(result["raw_rate"].to_numpy(dtype=float))
    age_rate_lookup: dict[tuple[str, int], float] = {
        (str(pos_group), int(age)): float(rate)
        for pos_group, age, rate in zip(
            result.loc[finite_rate, "pos_group"],
            result.loc[finite_rate, "career_age"],
            result.loc[finite_rate, "raw_rate"],
            strict=True,
        )
    }

    for group, group_rows in result.groupby("pos_group"):
        if group in NO_LOCAL_METRIC_GROUPS:
            continue
        idx = group_rows.index
        weights = group_rows["metric_denominator"].to_numpy(dtype=float)
        rates = group_rows["raw_rate"].to_numpy(dtype=float)
        valid = np.isfinite(rates) & (weights > 0)
        if valid.sum() < 2:
            result.loc[idx, "shrunk_rate"] = rates
            continue
        w_valid, rate_valid = weights[valid], rates[valid]
        grand = float(np.average(rate_valid, weights=w_valid))
        total_var = float(np.average((rate_valid - grand) ** 2, weights=w_valid))

        group_cells = cells.loc[
            cells["pos_group"].eq(group)
            & cells["metric_denominator"].gt(0)
            & cells["metric_numerator"].notna()
        ].copy()
        group_cells["age_rate"] = [
            age_rate_lookup.get((str(group), int(age)), np.nan) for age in group_cells["career_age"]
        ]
        group_cells = group_cells.loc[group_cells["age_rate"].notna()]
        if len(group_cells) >= 2:
            player_rate = group_cells["metric_numerator"] / group_cells["metric_denominator"]
            residual = (player_rate - group_cells["age_rate"]).to_numpy(dtype=float)
            residual_weight = group_cells["metric_denominator"].to_numpy(dtype=float)
            tau_within = float(np.average(residual**2, weights=residual_weight))
        else:
            tau_within = 0.0

        epsilon = max(total_var * 1e-6, 1e-12)
        mean_sigma2 = float(np.mean(tau_within / w_valid))
        tau_between = max(total_var - mean_sigma2, epsilon)
        k = tau_within / tau_between if tau_between > 0 else 0.0

        shrunk = rates.copy()
        shrunk[valid] = (w_valid * rate_valid + k * grand) / (w_valid + k)
        result.loc[idx, "shrunk_rate"] = shrunk
        result.loc[idx, "shrinkage_k"] = k
        result.loc[idx, "grand_mean"] = grand
        result.loc[idx, "tau_between"] = tau_between
        result.loc[idx, "tau_within"] = tau_within

    return result


def _tricube(u: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.abs(u), 0.0, 1.0)
    return (1.0 - clipped**3) ** 3


def local_linear_smooth(
    ages: np.ndarray, rates: np.ndarray, weights: np.ndarray, *, bandwidth: float = 1.5
) -> np.ndarray:
    """Snap-weighted 3-point (age-1, age, age+1) tricube local-linear smooth.

    A degree-1 weighted least-squares fit restricted to the focal age's
    immediate neighbours (weight = snaps x tricube(distance / bandwidth)),
    evaluated at the focal age. No monotonicity is imposed. Falls back to a
    weighted mean, then to the lone available point, when a full regression
    is not possible (a boundary age, a singular design, or a gap where a
    neighbour age has zero data).
    """

    n = len(ages)
    smoothed = np.full(n, np.nan)
    for i in range(n):
        window = np.abs(ages - ages[i]) <= 1.0
        window_ages = ages[window]
        window_rates = rates[window]
        window_weights = weights[window]
        valid = np.isfinite(window_rates) & (window_weights > 0)
        window_ages, window_rates, window_weights = (
            window_ages[valid],
            window_rates[valid],
            window_weights[valid],
        )
        if len(window_ages) == 0:
            continue
        if len(window_ages) == 1:
            smoothed[i] = float(window_rates[0])
            continue
        kernel = _tricube((window_ages - ages[i]) / bandwidth)
        combined_weight = window_weights * kernel
        if not np.isfinite(combined_weight).all() or combined_weight.sum() <= 0:
            smoothed[i] = float(np.average(window_rates, weights=window_weights))
            continue
        x = window_ages - ages[i]
        sqrt_weight = np.sqrt(combined_weight)
        design = np.column_stack([np.ones_like(x), x]) * sqrt_weight[:, None]
        target = window_rates * sqrt_weight
        try:
            beta, *_ = np.linalg.lstsq(design, target, rcond=None)
            smoothed[i] = float(beta[0])
        except np.linalg.LinAlgError:
            smoothed[i] = float(np.average(window_rates, weights=combined_weight))
    return smoothed


def smooth_curve(curve: pd.DataFrame, *, bandwidth: float = 1.5) -> pd.DataFrame:
    """Add ``smoothed_rate``: a second, independent look at curve shape."""

    result = curve.copy()
    result["smoothed_rate"] = np.nan
    for group, group_rows in result.groupby("pos_group"):
        if group in NO_LOCAL_METRIC_GROUPS:
            continue
        ordered = group_rows.sort_values("career_age")
        ages = ordered["career_age"].to_numpy(dtype=float)
        rates = ordered["raw_rate"].to_numpy(dtype=float)
        weights = ordered["metric_denominator"].to_numpy(dtype=float)
        result.loc[ordered.index, "smoothed_rate"] = local_linear_smooth(
            ages, rates, weights, bandwidth=bandwidth
        )
    return result


# ---------------------------------------------------------------------------
# Delta-method (within-player) curve
# ---------------------------------------------------------------------------


def delta_curve(
    cells: pd.DataFrame, curve: pd.DataFrame, *, snap_floor: float = DELTA_METHOD_SNAP_FLOOR
) -> pd.DataFrame:
    """Within-player age-to-age deltas, removing cross-sectional survivorship.

    Only players clearing ``snap_floor`` at BOTH consecutive career ages
    contribute a delta; the pair's weight is ``min(snaps_a, snaps_{a+1})``.
    ``cumulative_delta`` integrates ``mean_delta`` forward and backward from
    each group's modal entry age (the age with the most players in the
    cross-sectional curve).
    """

    eligible = cells.loc[
        cells["pos_group"].isin(_METRIC_GROUPS)
        & cells["metric_denominator"].ge(snap_floor)
        & cells["metric_numerator"].notna()
    ].copy()
    per_player_age = (
        eligible.groupby(["gsis_id", "pos_group", "career_age"], observed=True)
        .agg(
            metric_numerator=("metric_numerator", "sum"),
            metric_denominator=("metric_denominator", "sum"),
        )
        .reset_index()
    )
    per_player_age = per_player_age.loc[per_player_age["metric_denominator"].ge(snap_floor)].copy()
    per_player_age["rate"] = (
        per_player_age["metric_numerator"] / per_player_age["metric_denominator"]
    )

    empty_columns = [
        "pos_group",
        "career_age_from",
        "n_pairs",
        "mean_delta",
        "cumulative_delta",
    ]
    if per_player_age.empty:
        return pd.DataFrame(columns=empty_columns)

    pair_rows: list[dict[str, Any]] = []
    for (gsis_id, group), player_rows in per_player_age.groupby(
        ["gsis_id", "pos_group"], observed=True
    ):
        ordered = player_rows.sort_values("career_age")
        ages = ordered["career_age"].to_numpy()
        rates = ordered["rate"].to_numpy()
        snaps = ordered["metric_denominator"].to_numpy()
        for i in range(len(ages) - 1):
            if ages[i + 1] == ages[i] + 1:
                pair_rows.append(
                    {
                        "pos_group": group,
                        "career_age_from": int(ages[i]),
                        "gsis_id": gsis_id,
                        "delta": float(rates[i + 1] - rates[i]),
                        "weight": float(min(snaps[i], snaps[i + 1])),
                    }
                )
    if not pair_rows:
        return pd.DataFrame(columns=empty_columns)

    pairs = pd.DataFrame(pair_rows)
    pairs["delta_weighted"] = pairs["delta"] * pairs["weight"]
    summary = (
        pairs.groupby(["pos_group", "career_age_from"], observed=True)
        .agg(
            n_pairs=("delta", "size"),
            delta_weighted_sum=("delta_weighted", "sum"),
            weight_sum=("weight", "sum"),
        )
        .reset_index()
    )
    summary["mean_delta"] = summary["delta_weighted_sum"] / summary["weight_sum"]
    summary = summary.drop(columns=["delta_weighted_sum", "weight_sum"])

    modal_age: dict[str, int] = {}
    metric_curve = curve.loc[curve["coverage_status"].eq("metric") & curve["n_players"].gt(0)]
    for group, group_rows in metric_curve.groupby("pos_group"):
        group_key = str(group)
        best_row = group_rows.loc[group_rows["n_players"].idxmax(), "career_age"]
        modal_age[group_key] = int(best_row)  # type: ignore[arg-type]

    summary["cumulative_delta"] = np.nan
    for group, group_rows in summary.groupby("pos_group"):
        group_key = str(group)
        deltas = dict(zip(group_rows["career_age_from"], group_rows["mean_delta"], strict=True))
        modal = modal_age.get(group_key, int(group_rows["career_age_from"].min()))
        ages_present = set(deltas)
        max_age = max(ages_present | {modal})
        min_age = min(ages_present | {modal})
        cumulative: dict[int, float] = {modal: 0.0}
        for age in range(modal, max_age):
            cumulative[age + 1] = cumulative[age] + deltas.get(age, 0.0)
        for age in range(modal, min_age, -1):
            cumulative[age - 1] = cumulative[age] - deltas.get(age - 1, 0.0)
        mask = summary["pos_group"].eq(group)
        summary.loc[mask, "cumulative_delta"] = summary.loc[mask, "career_age_from"].map(cumulative)

    return summary.sort_values(["pos_group", "career_age_from"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Split-half reliability
# ---------------------------------------------------------------------------


def _weighted_pearson(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    mean_x = np.average(x, weights=weights)
    mean_y = np.average(y, weights=weights)
    covariance = np.average((x - mean_x) * (y - mean_y), weights=weights)
    variance_x = np.average((x - mean_x) ** 2, weights=weights)
    variance_y = np.average((y - mean_y) ** 2, weights=weights)
    if variance_x <= 0 or variance_y <= 0:
        return float("nan")
    return float(covariance / np.sqrt(variance_x * variance_y))


def _rate_vector_and_bootstrap(
    half_cells: pd.DataFrame,
    ages: np.ndarray,
    *,
    block_column: str,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point-estimate (rate, denominator) by age, plus a bootstrap rate matrix.

    Resamples the ``block_column`` unit (season or player) WITH REPLACEMENT,
    ``samples`` times, vectorized via ``np.bincount`` rather than a per-draw
    groupby: every row inherits its block's resample count as a weight, then
    numerator/denominator are re-summed by age. This is a proper block
    bootstrap over the natural iid unit for each scheme, not a naive
    per-row resample.
    """

    n_ages = len(ages)
    age_to_index = {int(age): position for position, age in enumerate(ages)}
    row_age = half_cells["career_age"].map(age_to_index).to_numpy()
    row_numerator = half_cells["metric_numerator"].to_numpy(dtype=float)
    row_denominator = half_cells["metric_denominator"].to_numpy(dtype=float)

    point_numerator = np.bincount(row_age, weights=row_numerator, minlength=n_ages)
    point_denominator = np.bincount(row_age, weights=row_denominator, minlength=n_ages)
    point_rate = np.divide(
        point_numerator,
        point_denominator,
        out=np.full(n_ages, np.nan),
        where=point_denominator > 0,
    )

    blocks, row_block = np.unique(half_cells[block_column].to_numpy(), return_inverse=True)
    n_blocks = len(blocks)
    rng = np.random.default_rng(seed)
    boot_rates = np.full((samples, n_ages), np.nan)
    if n_blocks > 0:
        for draw in range(samples):
            counts = np.bincount(rng.integers(0, n_blocks, size=n_blocks), minlength=n_blocks)
            row_weight = counts[row_block]
            num_by_age = np.bincount(row_age, weights=row_numerator * row_weight, minlength=n_ages)
            den_by_age = np.bincount(
                row_age, weights=row_denominator * row_weight, minlength=n_ages
            )
            boot_rates[draw] = np.divide(
                num_by_age, den_by_age, out=np.full(n_ages, np.nan), where=den_by_age > 0
            )
    return point_rate, point_denominator, boot_rates


def _reliability_for_group(
    group: str,
    half_a: pd.DataFrame,
    half_b: pd.DataFrame,
    *,
    block_column: str,
    scheme: str,
    samples: int,
    seed: int,
) -> dict[str, Any] | None:
    ages = np.array(sorted(set(half_a["career_age"].unique()) | set(half_b["career_age"].unique())))
    if len(ages) == 0:
        return None

    rate_a, denominator_a, boot_a = _rate_vector_and_bootstrap(
        half_a, ages, block_column=block_column, samples=samples, seed=seed
    )
    rate_b, denominator_b, boot_b = _rate_vector_and_bootstrap(
        half_b, ages, block_column=block_column, samples=samples, seed=seed + 1
    )

    shared = np.isfinite(rate_a) & np.isfinite(rate_b) & (denominator_a > 0) & (denominator_b > 0)
    n_shared = int(shared.sum())
    base: dict[str, Any] = {
        "pos_group": group,
        "scheme": scheme,
        "n_ages_compared": n_shared,
        "bootstrap_samples": samples,
    }
    if n_shared < _MIN_SHARED_AGES_FOR_RELIABILITY:
        return base | {
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_brown_reliability": float("nan"),
            "bootstrap_ci95_low": float("nan"),
            "bootstrap_ci95_high": float("nan"),
            "probability_positive": float("nan"),
            "bootstrap_valid_draws": 0,
            "note": "fewer than 3 shared career-age cells with data in both halves",
        }

    weights = np.minimum(denominator_a, denominator_b)[shared]
    pearson_r = _weighted_pearson(rate_a[shared], rate_b[shared], weights)
    spearman_rho = float(spearmanr(rate_a[shared], rate_b[shared]).correlation)
    spearman_brown = (
        (2.0 * pearson_r) / (1.0 + pearson_r)
        if np.isfinite(pearson_r) and pearson_r > -1.0
        else float("nan")
    )

    boot_r = np.full(samples, np.nan)
    fixed_weights = np.minimum(denominator_a, denominator_b)
    for draw in range(samples):
        shared_draw = np.isfinite(boot_a[draw]) & np.isfinite(boot_b[draw]) & shared
        if shared_draw.sum() < _MIN_SHARED_AGES_FOR_RELIABILITY:
            continue
        boot_r[draw] = _weighted_pearson(
            boot_a[draw][shared_draw], boot_b[draw][shared_draw], fixed_weights[shared_draw]
        )
    valid_boot = boot_r[np.isfinite(boot_r)]
    if len(valid_boot) < _MIN_VALID_BOOTSTRAP_DRAWS:
        ci_low, ci_high, probability_positive = float("nan"), float("nan"), float("nan")
    else:
        ci_low = float(np.quantile(valid_boot, 0.025))
        ci_high = float(np.quantile(valid_boot, 0.975))
        probability_positive = float(np.mean(valid_boot > 0.0))

    return base | {
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "spearman_brown_reliability": spearman_brown,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "probability_positive": probability_positive,
        "bootstrap_valid_draws": len(valid_boot),
        "note": "",
    }


def split_half_reliability(
    cells: pd.DataFrame,
    curve: pd.DataFrame,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Two split-half reliability schemes per metric-bearing position group.

    ``curve`` is accepted (not read) so a future caller can restrict which
    groups get scored without re-deriving the metric-group set here; today
    every group in :data:`_METRIC_GROUPS` present in ``cells`` is scored.
    """

    del curve  # reserved for a future group-restriction hook; not needed today.
    metric_cells = cells.loc[
        cells["pos_group"].isin(_METRIC_GROUPS)
        & cells["metric_denominator"].gt(0)
        & cells["metric_numerator"].notna()
    ].copy()
    if metric_cells.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    metric_cells["season_parity"] = np.where(metric_cells["season"] % 2 == 0, "even", "odd")
    for group in sorted(metric_cells["pos_group"].unique()):
        group_cells = metric_cells.loc[metric_cells["pos_group"] == group]
        half_odd = group_cells.loc[group_cells["season_parity"] == "odd"]
        half_even = group_cells.loc[group_cells["season_parity"] == "even"]
        result = _reliability_for_group(
            group,
            half_odd,
            half_even,
            block_column="season",
            scheme="odd_even_seasons",
            samples=samples,
            seed=seed,
        )
        if result is not None:
            rows.append(result)

    all_players = np.array(sorted(metric_cells["gsis_id"].unique()))
    rng = np.random.default_rng(seed + 1000)
    shuffled = rng.permutation(all_players)
    half_size = len(shuffled) // 2
    players_a = set(shuffled[:half_size])
    metric_cells["player_half"] = np.where(metric_cells["gsis_id"].isin(players_a), "a", "b")
    for group in sorted(metric_cells["pos_group"].unique()):
        group_cells = metric_cells.loc[metric_cells["pos_group"] == group]
        half_a = group_cells.loc[group_cells["player_half"] == "a"]
        half_b = group_cells.loc[group_cells["player_half"] == "b"]
        result = _reliability_for_group(
            group,
            half_a,
            half_b,
            block_column="gsis_id",
            scheme="random_player_halves",
            samples=samples,
            seed=seed + 2000,
        )
        if result is not None:
            rows.append(result)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PBP snapshot loading (self-contained, mirroring scripts/build_metagame_series.py
# rather than importing nfl_ats.pbp, per LEAD-58's import allowlist)
# ---------------------------------------------------------------------------

_PBP_LOAD_COLUMNS = ("game_id", "season_type", "qb_dropback", "passer_player_id", "epa")


def latest_pbp_snapshot_dir(pbp_raw_root: Path) -> Path:
    manifests = sorted(pbp_raw_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No PBP snapshot found in {pbp_raw_root}")
    return manifests[-1].parent


def available_pbp_seasons(snapshot_dir: Path) -> list[int]:
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    return sorted(int(partition["season"]) for partition in manifest["partitions"])


def load_pbp_seasons(snapshot_dir: Path, seasons: Iterable[int]) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for season in seasons:
        path = snapshot_dir / f"season={season}" / "plays.parquet"
        if path.is_file():
            frames[season] = pd.read_parquet(path, columns=list(_PBP_LOAD_COLUMNS))
    return frames


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgeCurvesResult:
    curve: pd.DataFrame
    delta: pd.DataFrame
    reliability: pd.DataFrame
    manifest: dict[str, Any]


def build_age_curves(
    players_raw_root: Path,
    players_values_raw_root: Path,
    pbp_raw_root: Path,
    *,
    as_of_season: int | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> AgeCurvesResult:
    """Resolve the latest local snapshots and run the full LEAD-58 pipeline."""

    player_snapshot = latest_player_snapshot(players_raw_root)
    value_snapshot = latest_player_value_snapshot(players_values_raw_root)

    raw_snaps = pd.read_parquet(player_snapshot.snaps_path)
    raw_rosters = pd.read_parquet(player_snapshot.rosters_path)
    raw_stats = pd.read_parquet(value_snapshot.stats_path)

    snap_seasons = sorted(
        pd.to_numeric(raw_snaps["season"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    if as_of_season is not None:
        snap_seasons = [season for season in snap_seasons if season < as_of_season]

    pbp_snapshot_dir = latest_pbp_snapshot_dir(pbp_raw_root)
    pbp_seasons_needed = sorted(set(snap_seasons) & set(available_pbp_seasons(pbp_snapshot_dir)))
    pbp_frames = load_pbp_seasons(pbp_snapshot_dir, pbp_seasons_needed)

    panel, diagnostics = build_career_age_panel(
        raw_snaps, raw_rosters, raw_stats, pbp_frames, as_of_season=as_of_season
    )
    cells = player_age_cells(panel)
    curve = cross_sectional_curve(cells)
    curve = shrink_cells(curve, cells)
    curve = smooth_curve(curve)
    delta = delta_curve(cells, curve)
    reliability = split_half_reliability(
        cells, curve, samples=bootstrap_samples, seed=bootstrap_seed
    )

    manifest: dict[str, Any] = {
        "builder_version": AGE_CURVES_VERSION,
        "axis": CAREER_AGE_AXIS,
        "as_of_season": as_of_season,
        "season_min": int(min(snap_seasons)) if snap_seasons else None,
        "season_max": int(max(snap_seasons)) if snap_seasons else None,
        "snap_floor_delta_method": DELTA_METHOD_SNAP_FLOOR,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "position_groups": POSITION_GROUPS,
        "metric_by_group": METRIC_BY_GROUP,
        "no_local_metric_groups": sorted(NO_LOCAL_METRIC_GROUPS),
        "resolved_snapshots": {
            "players": player_snapshot.snapshot_id,
            "player_values": value_snapshot.snapshot_id,
            "pbp": pbp_snapshot_dir.name,
        },
        "diagnostics": diagnostics,
    }
    return AgeCurvesResult(curve=curve, delta=delta, reliability=reliability, manifest=manifest)
