"""LEAD-24 Stage 1: rookie workload wall, snap-share dependence metric.

QUALITY-stage measurement only. This module answers three predeclared
questions from ``docs/rookie_wall.md`` (read that file for the full
predeclaration, written BEFORE any number below was computed):

1. Does a "rookie wall" exist -- do top-50-pick rookies who carry a heavy
   (>=70%) offense-or-defense snap share in weeks 1-11 decline, per snap,
   in weeks 12-17, more than a same-position VETERAN control carrying the
   same workload? (:func:`wall_candidates`, :func:`rookie_wall_measurement`)
2. A pregame, leakage-safe TEAM-WEEK dependence metric: what share of a
   team's offensive/defensive snaps went to top-50-pick rookies, trailing
   over the team's last 4 completed games. (:func:`team_week_dependence_shares`,
   :func:`trailing_dependence_feature`, :func:`late_season_high_dependence_flag`)
3. Split-half reliability of that dependence metric -- is it a stable team
   trait at all, independent of whether the wall itself is confirmed?
   (:func:`dependence_split_half_reliability`)

No ATS window is built here and nothing is wired into
``registry/rotation_registry.json``: that is explicitly the NEXT lane's job
per the LEAD-24 roadmap row. Per AGENTS.md's commensurability rule, the wall
delta (a per-snap performance difference) is not an admissible
``weak-signals record --effect-units`` entry and is NEVER recorded there;
only the dependence metric's reliability (a correlation) is recorded, via
``nfl-ats weak-signals record --effect-units correlation``.

Closing-grounds taxonomy (binding, restated here because this module reports
numbers a later session may be tempted to adjudicate): an interval or CI
that contains zero is NEVER grounds to reject, fail, or close a line of
work. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close
anything: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on
the wrong side of zero) or a measured ZERO split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is ``unresolved_below_power``. Nothing in this module
classifies itself; :func:`rookie_wall_measurement` and
:func:`dependence_split_half_reliability` report point estimates,
intervals, and ``probability_positive``/``probability_negative`` only.

Inputs (all local; snapshots resolved and pinned the same way
``nfl_ats.age_curves`` already does):

  - ``data/players/raw/<snapshot>/{snap_counts,weekly_rosters}.parquet`` via
    :mod:`nfl_ats.players` -- snap-level offense/defense percentages
    (pregame-unsafe by themselves; only ever used here as TRAILING, lagged
    features) and ``years_exp`` (career age, ``0`` == rookie).
  - ``data/raw/combine/<snapshot>/combine.parquet`` -- ``draft_ovr`` and
    ``pfr_id``, joined to ``gsis_id`` through the SAME stable pfr<->gsis
    crosswalk :func:`nfl_ats.players._stable_crosswalk` already backs
    ``nfl_ats.qb_identity_features.draft_team_by_gsis_id``. ``draft_ovr`` is
    present on only 61.9% of combine rows (measured, this session, on the
    frozen ``20260822T143152Z`` snapshot: ``draft_ovr.notna().mean() ==
    0.6193``) and the crosswalk itself only resolves a further fraction to a
    ``gsis_id`` (measured 77.1% of the ``draft_year >= 2013`` / draft_ovr-
    present rows, since combine covers drafts back to 2000 but the local
    roster crosswalk only starts in 2013). A combine row that fails either
    step is treated as **NOT a top-50 pick** -- never guessed, never
    imputed -- and :func:`top50_pick_lookup` returns the exact counts and
    rates behind that choice so every downstream measurement can disclose
    them rather than assert them.
  - :func:`nfl_ats.age_curves.build_career_age_panel` -- reused, not
    reimplemented, for the per-position-group per-snap performance metric
    (QB EPA/dropback, RB/WR/TE EPA/offense-snap, EDGE/DL/LB/CB/S disruption/
    defense-snap). OL/K/P/LS carry no local metric there and therefore carry
    none here either: per LEAD-24's own scope note, the "wall" for OL is
    snap-VOLUME only and is not scored by :func:`rookie_wall_measurement`
    (there is no rate to decline). :data:`METRIC_GROUPS` is exactly the nine
    groups :mod:`nfl_ats.age_curves` scores.

Point-in-time discipline: every quantity a hypothetical live card could read
pregame is built from STRICTLY PRIOR completed games only.
:func:`trailing_dependence_feature` computes a rolling mean over a team's
past games and then ``shift(1)``s it, so the CURRENT week's own snap shares
never enter its own trailing value -- checked by
``tests/test_rookie_wall.py``'s leakage test, which perturbs the current
week's shares and asserts the trailing column is unchanged. The wall
measurement itself (weeks 1-11 vs 12-17 performance) is explicitly NOT a
pregame feature -- it is a postgame, within-season descriptive comparison,
exactly like ``nfl_ats.age_curves``'s within-player age deltas -- and is not
represented as one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nfl_ats.age_curves import (
    DEFENSE_METRIC_GROUPS,
    OFFENSE_SKILL_METRIC_GROUPS,
    QB_METRIC_GROUP,
    available_pbp_seasons,
    build_career_age_panel,
    latest_pbp_snapshot_dir,
    load_pbp_seasons,
)
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.players import (
    _stable_crosswalk,
    attach_snap_player_ids,
    canonicalize_rosters,
    canonicalize_snaps,
    latest_player_snapshot,
    latest_player_value_snapshot,
)

ROOKIE_WALL_VERSION = "rookie-wall-v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAYERS_RAW_ROOT = REPO_ROOT / "data" / "players" / "raw"
DEFAULT_PLAYERS_VALUES_RAW_ROOT = REPO_ROOT / "data" / "players" / "values" / "raw"
DEFAULT_PBP_RAW_ROOT = REPO_ROOT / "data" / "pbp" / "raw"
DEFAULT_COMBINE_RAW_ROOT = REPO_ROOT / "data" / "raw" / "combine"

#: Every position group :mod:`nfl_ats.age_curves` scores with a real
#: per-snap/per-dropback rate. OL/K/P/LS are deliberately excluded -- no
#: local performance metric exists for them (see module docstring).
METRIC_GROUPS: tuple[str, ...] = (
    QB_METRIC_GROUP,
    *sorted(OFFENSE_SKILL_METRIC_GROUPS),
    *sorted(DEFENSE_METRIC_GROUPS),
)

TOP50_DRAFT_OVR_MAX = 50
HIGH_WORKLOAD_SNAP_SHARE = 0.70
EARLY_WEEKS: tuple[int, ...] = tuple(range(1, 12))
LATE_WEEKS: tuple[int, ...] = tuple(range(12, 18))
VETERAN_MIN_CAREER_AGE = 3
#: Same 100-snap(/dropback) floor as ``age_curves.DELTA_METHOD_SNAP_FLOOR``,
#: applied to EACH half (weeks 1-11 and weeks 12-17) independently, so a
#: rookie who is benched or injured in one half never contributes a delta
#: built on a handful of garbage-time snaps.
WALL_METRIC_SNAP_FLOOR = 100.0
ERA_WINDOWS: dict[str, tuple[int, int]] = {
    "2013_2018": (2013, 2018),
    "2019_2025": (2019, 2025),
}
DEPENDENCE_TRAILING_WINDOW = 4
DEPENDENCE_HIGH_PERCENTILE = 0.80
DEPENDENCE_LATE_WEEK_MIN = 12
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260905
_MIN_BLOCKS_FOR_BOOTSTRAP = 2
_MIN_VALID_BOOTSTRAP_DRAWS = 100
_MIN_UNITS_FOR_RELIABILITY = 5


# ---------------------------------------------------------------------------
# Top-50-pick identity
# ---------------------------------------------------------------------------


def top50_pick_lookup(
    combine_raw: pd.DataFrame, rosters_raw: pd.DataFrame
) -> tuple[dict[str, bool], dict[str, Any]]:
    """``gsis_id`` -> whether that player was drafted in the top 50 overall.

    Only combine rows with BOTH a non-null ``draft_ovr`` and a ``pfr_id``
    that resolves through :func:`nfl_ats.players._stable_crosswalk` (the
    identical crosswalk ``nfl_ats.qb_identity_features.draft_team_by_gsis_id``
    already uses) contribute. Everything else is left OUT of the returned
    dict; every caller in this module treats a missing key as "not a top-50
    pick" via ``.map(lookup).fillna(False)`` -- disclosed, never imputed.
    A player combine-invited or drafted more than once keeps only the
    EARLIEST ``draft_year`` row.
    """

    required = {"pfr_id", "draft_ovr", "draft_year"}
    missing = sorted(required.difference(combine_raw.columns))
    if missing:
        raise DataContractError(f"combine is missing columns: {', '.join(missing)}")

    rosters = canonicalize_rosters(rosters_raw)
    crosswalk = _stable_crosswalk(rosters)

    n_combine_rows = len(combine_raw)
    has_ovr = combine_raw["draft_ovr"].notna()
    n_with_draft_ovr = int(has_ovr.sum())

    with_ovr = combine_raw.loc[has_ovr & combine_raw["pfr_id"].notna()].copy()
    n_with_ovr_and_pfr_id = len(with_ovr)
    with_ovr["pfr_id"] = with_ovr["pfr_id"].astype(str)
    with_ovr["gsis_id"] = with_ovr["pfr_id"].map(crosswalk)
    joined = with_ovr.loc[with_ovr["gsis_id"].notna()].copy()
    n_joined_to_gsis = len(joined)

    joined["draft_year"] = pd.to_numeric(joined["draft_year"], errors="coerce")
    joined = joined.sort_values(["gsis_id", "draft_year"]).drop_duplicates("gsis_id", keep="first")
    lookup = {
        str(gsis_id): bool(draft_ovr <= TOP50_DRAFT_OVR_MAX)
        for gsis_id, draft_ovr in zip(joined["gsis_id"], joined["draft_ovr"], strict=True)
    }

    diagnostics = {
        "n_combine_rows": n_combine_rows,
        "n_with_draft_ovr": n_with_draft_ovr,
        "pct_with_draft_ovr": (n_with_draft_ovr / n_combine_rows)
        if n_combine_rows
        else float("nan"),
        "n_with_draft_ovr_and_pfr_id": n_with_ovr_and_pfr_id,
        "n_joined_to_gsis": n_joined_to_gsis,
        "join_rate_of_draft_ovr_and_pfr_id_rows": (
            (n_joined_to_gsis / n_with_ovr_and_pfr_id) if n_with_ovr_and_pfr_id else float("nan")
        ),
        "n_unique_players_resolved": len(lookup),
        "n_unique_players_top50": int(sum(lookup.values())),
    }
    return lookup, diagnostics


# ---------------------------------------------------------------------------
# Panel construction: age_curves' per-snap metric panel + raw snap shares
# ---------------------------------------------------------------------------


def build_rookie_wall_panel(
    snaps_raw: pd.DataFrame,
    rosters_raw: pd.DataFrame,
    stats_raw: pd.DataFrame,
    pbp_frames: Mapping[int, pd.DataFrame],
    combine_raw: pd.DataFrame,
    *,
    as_of_season: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse :func:`nfl_ats.age_curves.build_career_age_panel` and attach the
    two things it does not carry: each player-game's ``offense_pct``/
    ``defense_pct`` (needed for the 70%-workload gate and the dependence
    metric) and the top-50-pick / rookie flags.

    Returns ``(panel, diagnostics)``. ``panel`` adds ``offense_pct``,
    ``defense_pct``, ``is_top50_pick``, ``is_rookie`` to every column
    :func:`build_career_age_panel` already returns.
    """

    panel, diagnostics = build_career_age_panel(
        snaps_raw, rosters_raw, stats_raw, pbp_frames, as_of_season=as_of_season
    )

    rosters = canonicalize_rosters(rosters_raw)
    snaps = canonicalize_snaps(snaps_raw)
    if as_of_season is not None:
        rosters = rosters.loc[rosters["season"] < as_of_season].copy()
        snaps = snaps.loc[snaps["season"] < as_of_season].copy()

    linked = attach_snap_player_ids(snaps, rosters)
    linked = linked.loc[linked["gsis_id"].notna()].copy()
    pct_columns = linked.loc[
        :,
        [
            "gsis_id",
            "season",
            "week",
            "team",
            "offense_snaps",
            "defense_snaps",
            "st_snaps",
            "offense_pct",
            "defense_pct",
        ],
    ].copy()
    pct_columns["_snap_total"] = (
        pct_columns["offense_snaps"] + pct_columns["defense_snaps"] + pct_columns["st_snaps"]
    )
    n_pct_rows_before = len(pct_columns)
    pct_columns = pct_columns.sort_values("_snap_total", ascending=False).drop_duplicates(
        ["gsis_id", "season", "week", "team"], keep="first"
    )
    n_pct_duplicates_dropped = n_pct_rows_before - len(pct_columns)
    pct_columns = pct_columns.drop(
        columns=["_snap_total", "offense_snaps", "defense_snaps", "st_snaps"]
    )

    merged = panel.merge(
        pct_columns,
        on=["gsis_id", "season", "week", "team"],
        how="left",
        validate="many_to_one",
    )

    top50_lookup, top50_diagnostics = top50_pick_lookup(combine_raw, rosters_raw)
    merged["is_top50_pick"] = merged["gsis_id"].map(top50_lookup).fillna(False).astype(bool)
    merged["is_rookie"] = merged["career_age"].eq(0)

    diagnostics = dict(diagnostics)
    diagnostics["pct_duplicate_player_game_rows_dropped"] = n_pct_duplicates_dropped
    diagnostics["panel_rows_missing_pct"] = int(merged["offense_pct"].isna().sum())
    diagnostics["top50_pick_lookup"] = top50_diagnostics
    return merged, diagnostics


# ---------------------------------------------------------------------------
# Measurement 1: the wall itself (rookie vs. veteran within-player delta)
# ---------------------------------------------------------------------------


def high_workload_player_seasons(
    panel: pd.DataFrame,
    *,
    early_weeks: Iterable[int] = EARLY_WEEKS,
    threshold: float = HIGH_WORKLOAD_SNAP_SHARE,
) -> pd.DataFrame:
    """Per (player, season, position group): mean weeks-1-11 offense/defense
    share and the resulting ``high_workload`` gate (>=70% on either side,
    mean of the weeks actually played -- unweighted across weeks, since each
    week's own ``*_pct`` is already snap-normalized within that game).
    """

    early_weeks_set = set(early_weeks)
    early = panel.loc[panel["week"].isin(early_weeks_set)].copy()
    grouped = (
        early.groupby(["gsis_id", "season", "pos_group"], observed=True)
        .agg(
            mean_offense_pct=("offense_pct", "mean"),
            mean_defense_pct=("defense_pct", "mean"),
            n_weeks_early=("week", "nunique"),
            career_age=("career_age", "max"),
            is_top50_pick=("is_top50_pick", "max"),
            is_rookie=("is_rookie", "max"),
        )
        .reset_index()
    )
    grouped["high_workload"] = (grouped["mean_offense_pct"] >= threshold) | (
        grouped["mean_defense_pct"] >= threshold
    )
    return grouped


def within_player_half_season_delta(
    panel: pd.DataFrame,
    *,
    early_weeks: Iterable[int] = EARLY_WEEKS,
    late_weeks: Iterable[int] = LATE_WEEKS,
    snap_floor: float = WALL_METRIC_SNAP_FLOOR,
) -> pd.DataFrame:
    """Per (player, season, position group): weeks-12-17-minus-1-11 per-snap
    performance delta, both halves clearing ``snap_floor`` independently.

    Only :data:`METRIC_GROUPS` (the nine groups with a real local rate) are
    scored -- OL/K/P/LS have no ``metric_numerator`` to difference.
    """

    early_set, late_set = set(early_weeks), set(late_weeks)
    eligible = panel.loc[
        panel["pos_group"].isin(METRIC_GROUPS) & panel["metric_numerator"].notna()
    ].copy()
    eligible["half"] = np.where(
        eligible["week"].isin(early_set),
        "early",
        np.where(eligible["week"].isin(late_set), "late", ""),
    )
    eligible = eligible.loc[eligible["half"].ne("")].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "gsis_id",
                "season",
                "pos_group",
                "career_age",
                "is_top50_pick",
                "is_rookie",
                "numerator_early",
                "denominator_early",
                "numerator_late",
                "denominator_late",
                "rate_early",
                "rate_late",
                "delta",
                "weight",
            ]
        )

    grouped = (
        eligible.groupby(["gsis_id", "season", "pos_group", "half"], observed=True)
        .agg(
            numerator=("metric_numerator", "sum"),
            denominator=("metric_denominator", "sum"),
            career_age=("career_age", "max"),
            is_top50_pick=("is_top50_pick", "max"),
            is_rookie=("is_rookie", "max"),
        )
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["gsis_id", "season", "pos_group"],
        columns="half",
        values=["numerator", "denominator"],
    )
    raw_columns: Any = pivot.columns
    flat_columns: list[str] = ["_".join(str(part) for part in col) for col in raw_columns]
    pivot.columns = pd.Index(flat_columns)
    pivot = pivot.reset_index()
    for column in ("numerator_early", "denominator_early", "numerator_late", "denominator_late"):
        if column not in pivot.columns:
            pivot[column] = np.nan

    identity = (
        grouped.groupby(["gsis_id", "season", "pos_group"], observed=True)
        .agg(
            career_age=("career_age", "max"),
            is_top50_pick=("is_top50_pick", "max"),
            is_rookie=("is_rookie", "max"),
        )
        .reset_index()
    )
    result = pivot.merge(identity, on=["gsis_id", "season", "pos_group"], how="left")

    result = result.loc[
        result["denominator_early"].ge(snap_floor) & result["denominator_late"].ge(snap_floor)
    ].copy()
    result["rate_early"] = result["numerator_early"] / result["denominator_early"]
    result["rate_late"] = result["numerator_late"] / result["denominator_late"]
    result["delta"] = result["rate_late"] - result["rate_early"]
    result["weight"] = np.minimum(result["denominator_early"], result["denominator_late"])
    return result.reset_index(drop=True)


def wall_candidates(
    panel: pd.DataFrame,
    *,
    veteran_min_career_age: int = VETERAN_MIN_CAREER_AGE,
    early_weeks: Iterable[int] = EARLY_WEEKS,
    late_weeks: Iterable[int] = LATE_WEEKS,
    threshold: float = HIGH_WORKLOAD_SNAP_SHARE,
    snap_floor: float = WALL_METRIC_SNAP_FLOOR,
) -> pd.DataFrame:
    """Rookie (top-50-pick, high-workload) and veteran-control (high-workload,
    ``career_age >= veteran_min_career_age``) populations, each carrying its
    own within-player half-season delta, stacked with a ``population`` label.

    The veteran control applies the SAME >=70% weeks-1-11 workload gate as
    the rookie population (not just any veteran starter) so the comparison
    isolates the rookie-specific component of any late-season decline from a
    league-wide "heavy-workload players fade" effect at the same snap load --
    predeclared in ``docs/rookie_wall.md``.
    """

    gate = high_workload_player_seasons(panel, early_weeks=early_weeks, threshold=threshold)
    delta = within_player_half_season_delta(
        panel, early_weeks=early_weeks, late_weeks=late_weeks, snap_floor=snap_floor
    )
    merged = delta.merge(
        gate[["gsis_id", "season", "pos_group", "high_workload"]],
        on=["gsis_id", "season", "pos_group"],
        how="inner",
    )
    merged = merged.loc[merged["high_workload"]].copy()

    rookie = merged.loc[merged["is_rookie"] & merged["is_top50_pick"]].copy()
    rookie["population"] = "rookie_top50_high_workload"
    veteran = merged.loc[merged["career_age"] >= veteran_min_career_age].copy()
    veteran["population"] = "veteran_high_workload_control"
    return pd.concat([rookie, veteran], ignore_index=True)


def _paired_season_block_bootstrap_diff(
    rookie_values: np.ndarray,
    rookie_weights: np.ndarray,
    rookie_seasons: np.ndarray,
    veteran_values: np.ndarray,
    veteran_weights: np.ndarray,
    veteran_seasons: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Percentile bootstrap of (rookie mean - veteran mean), resampling the
    SAME drawn season sequence for both arms each draw (so any variation
    shared by a season -- e.g. an unusually rookie-heavy league-wide year --
    is preserved rather than washed out by independent resampling)."""

    all_seasons = np.unique(np.concatenate([rookie_seasons, veteran_seasons]))
    n = len(all_seasons)
    rookie_rows = {season: np.where(rookie_seasons == season)[0] for season in all_seasons}
    veteran_rows = {season: np.where(veteran_seasons == season)[0] for season in all_seasons}
    rng = np.random.default_rng(seed)
    diffs = np.full(samples, np.nan)
    if n < _MIN_BLOCKS_FOR_BOOTSTRAP:
        return diffs
    for i in range(samples):
        drawn = rng.choice(all_seasons, size=n, replace=True)
        r_idx = np.concatenate([rookie_rows[season] for season in drawn])
        v_idx = np.concatenate([veteran_rows[season] for season in drawn])
        r_mean = (
            float(np.average(rookie_values[r_idx], weights=rookie_weights[r_idx]))
            if r_idx.size and rookie_weights[r_idx].sum() > 0
            else np.nan
        )
        v_mean = (
            float(np.average(veteran_values[v_idx], weights=veteran_weights[v_idx]))
            if v_idx.size and veteran_weights[v_idx].sum() > 0
            else np.nan
        )
        diffs[i] = r_mean - v_mean
    return diffs


def rookie_wall_measurement(
    candidates: pd.DataFrame,
    *,
    era_windows: Mapping[str, tuple[int, int]] = ERA_WINDOWS,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Per (position group, era): rookie delta, veteran-control delta, and
    the rookie-minus-veteran headline, each with a season-blocked bootstrap
    CI and ``probability_wall_direction`` = P(rookie-minus-veteran delta is
    NEGATIVE) -- the predeclared FADE direction.

    An era key ``"2013_2025"`` (the full window) is added automatically
    alongside whatever eras ``era_windows`` names, since AGENTS.md's
    per-era-magnitude rule asks for eras to be reported WITHOUT the overall
    figure being hidden.
    """

    windows = dict(era_windows)
    windows.setdefault("2013_2025", (2013, 2025))

    rows: list[dict[str, Any]] = []
    for era_name, (season_start, season_end) in windows.items():
        era_candidates = candidates.loc[candidates["season"].between(season_start, season_end)]
        for pos_group in METRIC_GROUPS:
            group_rows = era_candidates.loc[era_candidates["pos_group"] == pos_group]
            rookie = group_rows.loc[group_rows["population"] == "rookie_top50_high_workload"]
            veteran = group_rows.loc[group_rows["population"] == "veteran_high_workload_control"]

            n_rookie = len(rookie)
            n_veteran = len(veteran)
            row: dict[str, Any] = {
                "pos_group": pos_group,
                "era": era_name,
                "season_start": season_start,
                "season_end": season_end,
                "n_rookie_player_seasons": n_rookie,
                "n_veteran_player_seasons": n_veteran,
                "rookie_delta_mean": float("nan"),
                "veteran_delta_mean": float("nan"),
                "rookie_minus_veteran": float("nan"),
                "bootstrap_ci95_low": float("nan"),
                "bootstrap_ci95_high": float("nan"),
                "probability_wall_direction": float("nan"),
                "bootstrap_valid_draws": 0,
            }
            if n_rookie == 0 or n_veteran == 0:
                rows.append(row)
                continue

            rookie_values = rookie["delta"].to_numpy(dtype=float)
            rookie_weights = rookie["weight"].to_numpy(dtype=float)
            rookie_seasons = rookie["season"].to_numpy()
            veteran_values = veteran["delta"].to_numpy(dtype=float)
            veteran_weights = veteran["weight"].to_numpy(dtype=float)
            veteran_seasons = veteran["season"].to_numpy()

            row["rookie_delta_mean"] = float(np.average(rookie_values, weights=rookie_weights))
            row["veteran_delta_mean"] = float(np.average(veteran_values, weights=veteran_weights))
            row["rookie_minus_veteran"] = row["rookie_delta_mean"] - row["veteran_delta_mean"]

            diffs = _paired_season_block_bootstrap_diff(
                rookie_values,
                rookie_weights,
                rookie_seasons,
                veteran_values,
                veteran_weights,
                veteran_seasons,
                samples=samples,
                seed=seed,
            )
            valid = diffs[np.isfinite(diffs)]
            if len(valid) >= _MIN_VALID_BOOTSTRAP_DRAWS:
                row["bootstrap_ci95_low"] = float(np.quantile(valid, 0.025))
                row["bootstrap_ci95_high"] = float(np.quantile(valid, 0.975))
                row["probability_wall_direction"] = float(np.mean(valid < 0.0))
                row["bootstrap_valid_draws"] = len(valid)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Measurement 2: pregame team-week dependence metric
# ---------------------------------------------------------------------------


def team_week_dependence_shares(panel: pd.DataFrame) -> pd.DataFrame:
    """Per (team, season, week): the share of that team's offensive AND
    defensive snaps taken by top-50-pick rookies.

    Each is the SUM of the individual qualifying players' own ``offense_pct``
    (respectively ``defense_pct``) for that team-game -- ``offense_pct`` is
    already that player's fraction of the team's offensive snaps that game,
    so summing across every top-50-pick rookie on the team gives the
    fraction of an average offensive snap that was filled by a top-50-pick
    rookie (can exceed 1.0 in principle with more than one such rookie
    playing heavy snaps simultaneously; bounded above by 11, the number of
    offensive/defensive slots on the field). NOT pregame by itself -- this is
    the realized, same-week value; :func:`trailing_dependence_feature` is the
    pregame-safe lagged transform of it.
    """

    universe = panel.loc[:, ["team", "season", "week"]].drop_duplicates()
    rookies = panel.loc[panel["is_top50_pick"] & panel["is_rookie"]]
    shares = (
        rookies.groupby(["team", "season", "week"], observed=True)
        .agg(
            offense_share=("offense_pct", "sum"),
            defense_share=("defense_pct", "sum"),
            n_top50_rookies=("gsis_id", "nunique"),
        )
        .reset_index()
    )
    result = universe.merge(shares, on=["team", "season", "week"], how="left")
    for column in ("offense_share", "defense_share", "n_top50_rookies"):
        result[column] = result[column].fillna(0.0)
    result["share_sum"] = result["offense_share"] + result["defense_share"]
    return result.sort_values(["team", "season", "week"]).reset_index(drop=True)


def trailing_dependence_feature(
    shares: pd.DataFrame, *, window: int = DEPENDENCE_TRAILING_WINDOW
) -> pd.DataFrame:
    """Add trailing (strictly-prior-games) versions of the three share
    columns: a rolling mean over the team's last ``window`` games, THEN
    shifted by one game, so the current week's own share never contributes
    to its own trailing value (leakage-tested)."""

    result = shares.sort_values(["team", "season", "week"]).copy()
    for column, out_column in (
        ("offense_share", "trailing_offense_share"),
        ("defense_share", "trailing_defense_share"),
        ("share_sum", "trailing_share_sum"),
    ):
        result[out_column] = result.groupby(["team", "season"], observed=True)[column].transform(
            lambda series, window=window: (
                series.rolling(window=window, min_periods=1).mean().shift(1)
            )
        )
    return result


def late_season_high_dependence_flag(
    trailing: pd.DataFrame,
    *,
    percentile: float = DEPENDENCE_HIGH_PERCENTILE,
    late_week_min: int = DEPENDENCE_LATE_WEEK_MIN,
) -> pd.DataFrame:
    """Add the league-relative, pregame-safe ``late_season_high_dependence``
    flag: this team-week's ``trailing_share_sum`` is at or above the
    ``percentile`` of every team's ``trailing_share_sum`` for the SAME
    season/week (a cross-sectional threshold built only from other teams'
    own trailing -- i.e. also strictly-prior -- values) AND
    ``week >= late_week_min``.
    """

    result = trailing.copy()

    def _threshold(values: pd.Series) -> float:
        valid = values.dropna()
        return float(valid.quantile(percentile)) if len(valid) else float("nan")

    thresholds = result.groupby(["season", "week"], observed=True)["trailing_share_sum"].transform(
        _threshold
    )
    result["league_p80_trailing_share_sum"] = thresholds
    result["late_season_high_dependence"] = (
        result["trailing_share_sum"].notna()
        & (result["trailing_share_sum"] >= thresholds)
        & (result["week"] >= late_week_min)
    )
    return result


# ---------------------------------------------------------------------------
# Measurement 3: split-half reliability of the dependence metric
# ---------------------------------------------------------------------------


def team_season_split_half(shares: pd.DataFrame) -> pd.DataFrame:
    """Per (team, season): mean ``share_sum`` on ODD weeks vs. EVEN weeks --
    the raw (non-trailing) same-week value, since reliability asks whether
    the underlying trait is a stable team characteristic, not whether the
    lagged feature leaks."""

    working = shares.copy()
    working["week_parity"] = np.where(working["week"] % 2 == 0, "even", "odd")
    grouped = (
        working.groupby(["team", "season", "week_parity"], observed=True)["share_sum"]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["team", "season"], columns="week_parity", values="share_sum"
    ).reset_index()
    for column in ("odd", "even"):
        if column not in pivot.columns:
            pivot[column] = np.nan
    return pivot.dropna(subset=["odd", "even"]).reset_index(drop=True)


def season_to_season_pairs(shares: pd.DataFrame) -> pd.DataFrame:
    """Per team: season-mean ``share_sum`` paired against the SAME team's
    next season's mean -- one row per (team, season, season+1) pair present
    in both seasons."""

    team_season_mean = (
        shares.groupby(["team", "season"], observed=True)["share_sum"].mean().reset_index()
    )
    next_season = team_season_mean.copy()
    next_season["season"] = next_season["season"] - 1
    next_season = next_season.rename(columns={"share_sum": "share_next_season"})
    pairs = team_season_mean.merge(
        next_season[["team", "season", "share_next_season"]], on=["team", "season"], how="inner"
    )
    return pairs.rename(columns={"share_sum": "share_this_season"}).reset_index(drop=True)


def _season_block_bootstrap_correlation(
    x: np.ndarray, y: np.ndarray, seasons: np.ndarray, *, samples: int, seed: int
) -> np.ndarray:
    unique_seasons = np.unique(seasons)
    n = len(unique_seasons)
    rows_by_season = {season: np.where(seasons == season)[0] for season in unique_seasons}
    rng = np.random.default_rng(seed)
    draws = np.full(samples, np.nan)
    if n < _MIN_BLOCKS_FOR_BOOTSTRAP:
        return draws
    for i in range(samples):
        drawn = rng.choice(unique_seasons, size=n, replace=True)
        idx = np.concatenate([rows_by_season[season] for season in drawn])
        if idx.size < 3:
            continue
        xi, yi = x[idx], y[idx]
        if np.std(xi) <= 0 or np.std(yi) <= 0:
            continue
        draws[i] = float(np.corrcoef(xi, yi)[0, 1])
    return draws


def _team_shuffle_null_correlation(
    x: np.ndarray, y: np.ndarray, block: np.ndarray, *, draws: int, seed: int
) -> np.ndarray:
    """Permutation null: shuffle which team's ``y`` value pairs with which
    team's ``x`` value WITHIN each ``block`` (season), preserving any
    league-wide shared-season trend while destroying team-specific pairing.
    """

    rng = np.random.default_rng(seed)
    idx_by_block = {value: np.where(block == value)[0] for value in np.unique(block)}
    null_r = np.full(draws, np.nan)
    for i in range(draws):
        perm_idx = np.arange(len(x))
        for _, idxs in idx_by_block.items():
            perm_idx[idxs] = rng.permutation(idxs)
        y_perm = y[perm_idx]
        if np.std(x) <= 0 or np.std(y_perm) <= 0:
            continue
        null_r[i] = float(np.corrcoef(x, y_perm)[0, 1])
    return null_r


def _reliability_row(
    scheme: str,
    x: pd.Series,
    y: pd.Series,
    block: pd.Series,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    block_arr = block.to_numpy()
    n_units = len(x_arr)
    base: dict[str, Any] = {"scheme": scheme, "n_units": n_units, "bootstrap_samples": samples}
    if n_units < _MIN_UNITS_FOR_RELIABILITY or np.std(x_arr) <= 0 or np.std(y_arr) <= 0:
        return base | {
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_brown_reliability": float("nan"),
            "bootstrap_ci95_low": float("nan"),
            "bootstrap_ci95_high": float("nan"),
            "probability_positive": float("nan"),
            "bootstrap_valid_draws": 0,
            "shuffle_null_mean": float("nan"),
            "shuffle_null_ci95_low": float("nan"),
            "shuffle_null_ci95_high": float("nan"),
            "shuffle_null_percentile_of_observed": float("nan"),
            "note": f"fewer than {_MIN_UNITS_FOR_RELIABILITY} units or a constant column",
        }

    pearson_r = float(np.corrcoef(x_arr, y_arr)[0, 1])
    spearman_rho = float(spearmanr(x_arr, y_arr).correlation)
    spearman_brown = (2.0 * pearson_r) / (1.0 + pearson_r) if pearson_r > -1.0 else float("nan")

    boot = _season_block_bootstrap_correlation(x_arr, y_arr, block_arr, samples=samples, seed=seed)
    valid = boot[np.isfinite(boot)]
    if len(valid) >= _MIN_VALID_BOOTSTRAP_DRAWS:
        ci_low, ci_high = float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))
        probability_positive = float(np.mean(valid > 0.0))
    else:
        ci_low = ci_high = probability_positive = float("nan")

    null_r = _team_shuffle_null_correlation(
        x_arr, y_arr, block_arr, draws=samples, seed=seed + 5000
    )
    null_valid = null_r[np.isfinite(null_r)]
    if len(null_valid) >= _MIN_VALID_BOOTSTRAP_DRAWS:
        null_mean = float(np.mean(null_valid))
        null_ci_low, null_ci_high = (
            float(np.quantile(null_valid, 0.025)),
            float(np.quantile(null_valid, 0.975)),
        )
        null_percentile = float(np.mean(null_valid <= pearson_r))
    else:
        null_mean = null_ci_low = null_ci_high = null_percentile = float("nan")

    return base | {
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "spearman_brown_reliability": spearman_brown,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "probability_positive": probability_positive,
        "bootstrap_valid_draws": len(valid),
        "shuffle_null_mean": null_mean,
        "shuffle_null_ci95_low": null_ci_low,
        "shuffle_null_ci95_high": null_ci_high,
        "shuffle_null_percentile_of_observed": null_percentile,
        "note": "",
    }


def dependence_split_half_reliability(
    shares: pd.DataFrame,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Two independent reliability schemes for the team-season dependence
    trait, each with a season-blocked bootstrap and a team-label shuffle
    null: (a) odd-vs-even weeks within season (team-season unit, block =
    season); (b) season-to-season (team unit, block = the earlier season of
    the pair)."""

    rows: list[dict[str, Any]] = []

    within_season = team_season_split_half(shares)
    if not within_season.empty:
        rows.append(
            _reliability_row(
                "odd_even_weeks_team_season",
                within_season["odd"],
                within_season["even"],
                within_season["season"],
                samples=samples,
                seed=seed,
            )
        )

    season_pairs = season_to_season_pairs(shares)
    if not season_pairs.empty:
        rows.append(
            _reliability_row(
                "season_to_season",
                season_pairs["share_this_season"],
                season_pairs["share_next_season"],
                season_pairs["season"],
                samples=samples,
                seed=seed + 10000,
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Input loading (mirrors nfl_ats.age_curves.build_age_curves' snapshot resolution)
# ---------------------------------------------------------------------------


def latest_combine_raw(combine_raw_root: Path) -> pd.DataFrame:
    candidates = sorted(combine_raw_root.glob("*/combine.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no {combine_raw_root}/*/combine.parquet snapshot found")
    return pd.read_parquet(candidates[-1])


def load_rookie_wall_inputs(
    players_raw_root: Path,
    players_values_raw_root: Path,
    pbp_raw_root: Path,
    combine_raw_root: Path,
    *,
    as_of_season: int | None = None,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame], pd.DataFrame, dict[str, str]
]:
    """Resolve every newest local snapshot this module needs.

    Returns ``(raw_snaps, raw_rosters, raw_stats, pbp_frames, raw_combine,
    snapshot_ids)``.
    """

    player_snapshot = latest_player_snapshot(players_raw_root)
    value_snapshot = latest_player_value_snapshot(players_values_raw_root)

    raw_snaps = pd.read_parquet(player_snapshot.snaps_path)
    raw_rosters = pd.read_parquet(player_snapshot.rosters_path)
    raw_stats = pd.read_parquet(value_snapshot.stats_path)
    raw_combine = latest_combine_raw(combine_raw_root)

    snap_seasons = sorted(
        pd.to_numeric(raw_snaps["season"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    if as_of_season is not None:
        snap_seasons = [season for season in snap_seasons if season < as_of_season]

    pbp_snapshot_dir = latest_pbp_snapshot_dir(pbp_raw_root)
    pbp_seasons_needed = sorted(set(snap_seasons) & set(available_pbp_seasons(pbp_snapshot_dir)))
    pbp_frames = load_pbp_seasons(pbp_snapshot_dir, pbp_seasons_needed)

    require_columns(raw_combine, ("pfr_id", "draft_ovr", "draft_year"), "combine")

    snapshot_ids = {
        "players": player_snapshot.snapshot_id,
        "player_values": value_snapshot.snapshot_id,
        "pbp": pbp_snapshot_dir.name,
        "combine": sorted(combine_raw_root.glob("*/combine.parquet"))[-1].parent.name,
    }
    return raw_snaps, raw_rosters, raw_stats, pbp_frames, raw_combine, snapshot_ids
