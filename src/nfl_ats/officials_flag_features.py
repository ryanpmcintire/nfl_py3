"""Officiating-crew leads: LEAD-31 (rookie-referee underdog), LEAD-32
(directional home-cooking reliability + on-production flag), LEAD-34
(crew-familiarity second meetings). LEAD-33 (all-star crews) has no local
marker and is SKIPPED -- see ``docs/officials_crew_leads.md``.

Predeclared in ``docs/officials_crew_leads.md`` BEFORE any outcome in this
family was computed: population, thresholds, seed, and sample counts are
all fixed there.

Reuses, never rebuilds:

- ``nfl_ats.experiment_runner._latest_officials_snapshot`` /
  ``._build_referee_trait_data`` / ``._REFEREE_POSITION`` /
  ``._REFEREE_SEASON_TYPE`` -- the referee battery's own
  officials/game_penalties crosswalk join (``docs/referee_battery.md``) and
  per-(official, season) tenure/trait builder. The same reuse pattern
  ``nfl_ats.crew_tilt_refresh_overlay`` already established for this exact
  module.
- ``nfl_ats.pbp_coaching_traits.build_odd_even_halves`` /
  ``.build_season_to_season_pairs`` / ``.paired_split_half_reliability`` --
  Wave 4's split-half reliability harness (season-blocked bootstrap,
  Spearman-Brown correction, within-season label-shuffle null). Generic
  over the grouping column's NAME, not its meaning: this module renames
  ``official_name`` to the literal column ``"team"`` before calling it, a
  column-name compatibility shim, not a claim that a crew is a team.
- ``nfl_ats.schedule_flag_features.default_opener_lines`` / ``.default_schedule``
  / ``._attach`` -- the Tuesday-opener consensus spread store and the
  additive-merge helper every sibling on-production candidate already uses.

**Penalty YARDS, home/away split, are NOT available locally** (see
``docs/officials_crew_leads.md`` section "Data sources"): the local trimmed
PBP snapshot carries ``penalty``/``penalty_yards`` but not ``penalty_team``,
and ``game_penalties.parquet`` itself only ever persisted COUNTS. LEAD-32 is
built on penalty counts only, disclosed rather than silently narrowed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.experiment_runner import (
    _REFEREE_POSITION,
    _REFEREE_SEASON_TYPE,
    _build_referee_trait_data,
    _latest_officials_snapshot,
)
from nfl_ats.pbp_coaching_traits import (
    PBP_TRAIT_N_BOOT,
    PBP_TRAIT_N_NULL,
    PBP_TRAIT_RELIABILITY_SEED,
    build_odd_even_halves,
    build_season_to_season_pairs,
    paired_split_half_reliability,
)
from nfl_ats.schedule_flag_features import _attach, default_opener_lines
from nfl_ats.weak_stack_v3_features import latest_schedules_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Frozen column names, matching every sibling on-production candidate's
#: single-new-column discipline.
CREW_HOME_BIAS_COLUMN = "crew_home_bias_flag"
SECOND_MEETING_FAVORITE_COLUMN = "crew_second_meeting_favorite_flag"
ROOKIE_CREW_UNDERDOG_COLUMN = "rookie_crew_underdog_flag"

#: LEAD-32 Stage-2: minimum PRIOR games this season before a crew's trailing
#: home-bias is considered eligible (docs/officials_crew_leads.md).
TRAILING_HOME_BIAS_MIN_GAMES = 3

#: LEAD-31: excludes the left-censored 2015 all-"rookies" slate, matching
#: docs/referee_battery.md's own referee_rookie_home_cover population.
ROOKIE_ELIGIBLE_SEASON_FLOOR = 2016
#: "first OR second dataset-visible season" -- prior_seasons_experience is
#: 0-indexed (0 = first season, 1 = second season).
ROOKIE_PRIOR_EXPERIENCE_MAX = 1

#: Columns every ``home_away_penalty_game_table``-shaped frame must carry --
#: enforced on every derived-table entry point (whether loaded fresh or
#: injected by a test), so a malformed table raises ``DataContractError``
#: instead of a raw ``KeyError`` deep inside a ``groupby``.
_PENALTY_TABLE_REQUIRED_COLUMNS = {
    "game_id",
    "official_name",
    "season",
    "week",
    "home_team",
    "away_team",
    "penalties_total",
    "home_minus_away",
}


def _require_penalty_table_columns(table: pd.DataFrame) -> None:
    missing = sorted(_PENALTY_TABLE_REQUIRED_COLUMNS.difference(table.columns))
    if missing:
        raise DataContractError(f"officials penalty table is missing columns: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Shared loader: officials -> schedules crosswalk -> game_penalties, reusing
# the referee battery's own snapshot discovery and join.
# ---------------------------------------------------------------------------


def home_away_penalty_game_table(repo_root: Path | None = None) -> pd.DataFrame:
    """One row per (official_name, REG game with a matched head referee).

    Columns: ``game_id`` (standard format), ``official_name``, ``season``,
    ``week``, ``home_team``, ``away_team``, ``penalties_total``,
    ``penalties_on_home``, ``penalties_on_away``, ``home_minus_away`` (=
    ``penalties_on_home - penalties_on_away``, sign-flipped from
    ``docs/referee_battery.md``'s own ``mean_diff`` which is
    ``away - home``; see that doc for why this is a labelling choice, not a
    new measurement).
    """

    root = repo_root or REPO_ROOT
    officials_path, game_penalties_path, _snapshot_id = _latest_officials_snapshot(root)
    officials = pd.read_parquet(officials_path)
    refs = officials.loc[
        (officials["position"] == _REFEREE_POSITION)
        & (officials["season_type"] == _REFEREE_SEASON_TYPE)
    ].copy()

    schedules = pd.read_parquet(latest_schedules_snapshot(root)).loc[:, ["game_id", "old_game_id"]]
    refs = refs.merge(
        schedules, left_on="game_id", right_on="old_game_id", how="inner", suffixes=("_legacy", "")
    )
    refs = refs.loc[:, ["game_id", "official_name"]]

    game_penalties = pd.read_parquet(game_penalties_path)
    required_gp = {
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "penalties_total",
        "penalties_on_home",
        "penalties_on_away",
    }
    missing_gp = sorted(required_gp.difference(game_penalties.columns))
    if missing_gp:
        raise DataContractError(
            f"{game_penalties_path} is missing columns: {', '.join(missing_gp)}"
        )

    merged = refs.merge(game_penalties, on="game_id", how="inner", suffixes=("", "_gp"))
    merged["game_id"] = merged["game_id"].astype(str)
    merged["season"] = pd.to_numeric(merged["season"], errors="raise").astype(int)
    merged["week"] = pd.to_numeric(merged["week"], errors="raise").astype(int)
    merged["home_minus_away"] = merged["penalties_on_home"] - merged["penalties_on_away"]
    return merged


# ---------------------------------------------------------------------------
# LEAD-32 Stage 1: directional home-cooking reliability (odd/even split-half
# within season, season-to-season by referee) via the reused PBP-trait
# reliability harness.
# ---------------------------------------------------------------------------


def officials_home_bias_reliability(
    repo_root: Path | None = None,
    *,
    table: pd.DataFrame | None = None,
    seed: int = PBP_TRAIT_RELIABILITY_SEED,
    n_boot: int = PBP_TRAIT_N_BOOT,
    n_null: int = PBP_TRAIT_N_NULL,
) -> dict[str, Any]:
    """Both reliability reads for the ``home_minus_away`` directional trait.

    ``within_season_odd_even_week`` is the NEW measurement (Spearman-Brown
    corrected); ``season_to_season_same_referee`` reproduces
    ``docs/referee_battery.md``'s own ``mean_diff`` season-to-season Pearson
    r up to a global sign flip, now with a season-blocked bootstrap CI the
    original point-estimate-only read never had. ``table``, when given
    (tests), is used instead of loading real snapshots via ``repo_root``.
    """

    table = table if table is not None else home_away_penalty_game_table(repo_root or REPO_ROOT)

    long = table.rename(columns={"official_name": "team"})[
        ["team", "season", "week", "home_minus_away"]
    ]
    within_pairs = build_odd_even_halves(long, "home_minus_away", min_per_half=1)
    within = paired_split_half_reliability(
        within_pairs,
        metric="referee_home_minus_away_penalty_count",
        method="within_season_odd_even_week",
        seed=seed,
        n_boot=n_boot,
        n_null=n_null,
        spearman_brown=True,
    )

    team_season = (
        table.groupby(["official_name", "season"])["home_minus_away"]
        .mean()
        .reset_index()
        .rename(columns={"official_name": "team"})
    )
    across_pairs = build_season_to_season_pairs(team_season, "home_minus_away")
    across = paired_split_half_reliability(
        across_pairs,
        metric="referee_home_minus_away_penalty_count",
        method="season_to_season_same_referee",
        seed=seed + 1,
        n_boot=n_boot,
        n_null=n_null,
        spearman_brown=False,
    )

    return {
        "n_game_rows": len(table),
        "n_distinct_officials": int(table["official_name"].nunique()),
        "n_seasons": int(table["season"].nunique()),
        "within_season_odd_even_week": within,
        "season_to_season_same_referee": across,
    }


# ---------------------------------------------------------------------------
# LEAD-32 Stage 2: trailing (prior-games-only, within season) home-bias top
# quartile -> BACK the home team.
# ---------------------------------------------------------------------------


def trailing_home_bias_table(
    repo_root: Path | None = None, *, table: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per (official_name, season) crew-game, the trailing mean
    ``home_minus_away`` over that crew's own PRIOR games THIS SEASON ONLY.

    ``trailing_home_bias`` is NaN until the crew has officiated at least
    :data:`TRAILING_HOME_BIAS_MIN_GAMES` prior games this season -- an
    ``expanding().mean().shift(1)`` over the crew's own within-season game
    order, so game *k*'s own penalty count can never reach its own value,
    but legitimately changes every LATER game's value in the same
    crew-season (both directions are asserted in
    ``tests/test_officials_flag_features.py``). ``table``, when given
    (tests), is used instead of loading real snapshots via ``repo_root``.
    """

    table = table if table is not None else home_away_penalty_game_table(repo_root or REPO_ROOT)
    _require_penalty_table_columns(table)
    table = table.sort_values(["official_name", "season", "week", "game_id"]).reset_index(drop=True)

    def _trailing(group: pd.DataFrame) -> pd.Series:
        trailing_mean = group["home_minus_away"].expanding().mean().shift(1)
        prior_count = np.arange(len(group))
        eligible = prior_count >= TRAILING_HOME_BIAS_MIN_GAMES
        return pd.Series(np.where(eligible, trailing_mean, np.nan), index=group.index)

    table["trailing_home_bias"] = table.groupby(
        ["official_name", "season"], group_keys=False
    ).apply(_trailing)
    return table


def derive_crew_home_bias_features(
    repo_root: Path | None = None, *, table: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return ``(game_id, crew_home_bias_flag)`` for every matched-referee game.

    ``1.0`` when the home team's crew's trailing within-season home-bias
    sits in the GLOBAL top quartile (``pd.qcut(4)`` over every eligible
    trailing value, matching every other quartile-cut trait in this repo);
    ``0.0`` otherwise, including "not yet
    :data:`TRAILING_HOME_BIAS_MIN_GAMES` prior games this season."
    Unsigned, single-sided (BACK home): the same crew officiates both
    sides, so this is not a home/away comparison. ``table``, when given
    (tests), is ``home_away_penalty_game_table``-shaped and used instead of
    loading real snapshots.
    """

    table = trailing_home_bias_table(repo_root, table=table)
    valid_mask = table["trailing_home_bias"].notna()
    quartile = pd.Series(np.nan, index=table.index)
    if int(valid_mask.sum()) >= 4:
        quartile.loc[valid_mask] = (
            pd.qcut(table.loc[valid_mask, "trailing_home_bias"], 4, labels=[1, 2, 3, 4])
            .astype(int)
            .astype(float)
        )
    flag = quartile.eq(4.0).fillna(False).astype(float)
    out = pd.DataFrame({"game_id": table["game_id"].astype(str), CREW_HOME_BIAS_COLUMN: flag})
    return out.drop_duplicates("game_id").reset_index(drop=True)


def attach_crew_home_bias_features(
    features: pd.DataFrame, *, repo_root: Path | None = None, schedule: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Additively join ``crew_home_bias_flag`` onto ``features`` by ``game_id``.

    Games with no matched referee (e.g. outside 2015-2025, or an unmatched
    crosswalk row) default to ``0.0`` -- a documented "no signal" default,
    matching every sibling on-production candidate's convention.
    """

    root = repo_root or REPO_ROOT

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        del sched
        return derive_crew_home_bias_features(root)

    merged = _attach(features, schedule, _derive, (CREW_HOME_BIAS_COLUMN,))
    merged[CREW_HOME_BIAS_COLUMN] = merged[CREW_HOME_BIAS_COLUMN].fillna(0.0)
    return merged


# ---------------------------------------------------------------------------
# LEAD-34: crew-familiarity second meetings (deterministic; reliability
# not_applicable).
# ---------------------------------------------------------------------------


def crew_familiarity_table(
    repo_root: Path | None = None, *, table: pd.DataFrame | None = None
) -> pd.DataFrame:
    """``home_away_penalty_game_table`` plus a ``second_meeting`` boolean.

    ``second_meeting`` is ``True`` when the SAME ``official_name`` has
    already officiated, EARLIER in the same season (by week order), a game
    involving the home team OR the away team of the current game. Purely a
    function of team identity and week order -- never reads any game's own
    penalty count or outcome. ``table``, when given (tests), is used
    instead of loading real snapshots via ``repo_root``.
    """

    table = table if table is not None else home_away_penalty_game_table(repo_root or REPO_ROOT)
    _require_penalty_table_columns(table)
    table = table.sort_values(["official_name", "season", "week", "game_id"]).reset_index(drop=True)

    seen: dict[tuple[str, int], set[str]] = {}
    flags: list[bool] = []
    for row in table.itertuples(index=False):
        key = (str(row.official_name), int(cast(Any, row.season)))
        teams_seen = seen.setdefault(key, set())
        flags.append(bool(row.home_team in teams_seen or row.away_team in teams_seen))
        teams_seen.add(str(row.home_team))
        teams_seen.add(str(row.away_team))
    table = table.assign(second_meeting=flags)
    return table


def describe_crew_familiarity(
    repo_root: Path | None = None, *, table: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Descriptive frequency and penalty-count gap (task: descriptive only)."""

    table = crew_familiarity_table(repo_root, table=table)
    flagged = table.loc[table["second_meeting"]]
    unflagged = table.loc[~table["second_meeting"]]
    return {
        "n_games_with_referee": len(table),
        "n_second_meeting": len(flagged),
        "pct_second_meeting": float(len(flagged) / len(table)) if len(table) else float("nan"),
        "mean_penalties_total_second_meeting": float(flagged["penalties_total"].mean())
        if len(flagged)
        else float("nan"),
        "mean_penalties_total_first_meeting": float(unflagged["penalties_total"].mean())
        if len(unflagged)
        else float("nan"),
        "penalties_total_diff_second_minus_first": (
            float(flagged["penalties_total"].mean() - unflagged["penalties_total"].mean())
            if len(flagged) and len(unflagged)
            else float("nan")
        ),
    }


def derive_second_meeting_favorite_features(
    repo_root: Path | None,
    opener_lines: pd.DataFrame,
    *,
    table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return ``(game_id, crew_second_meeting_favorite_flag)`` for every matched game.

    ``+1`` when ``second_meeting`` AND the home team is favored at the
    Tuesday opener; ``-1`` when ``second_meeting`` AND the away team is
    favored; ``0`` otherwise (not a second meeting, an exact opener
    pick'em, or a missing opener line). ``table``, when given (tests), is
    ``home_away_penalty_game_table``-shaped and used instead of loading
    real snapshots.
    """

    if "game_id" not in opener_lines.columns:
        raise DataContractError("opener_lines is missing the game_id join key")

    familiarity = crew_familiarity_table(repo_root, table=table)[
        ["game_id", "second_meeting"]
    ].drop_duplicates("game_id")
    merged = familiarity.merge(
        opener_lines[["game_id", "tue_open_home_spread"]], on="game_id", how="left"
    )
    spread = merged["tue_open_home_spread"]
    home_favorite = merged["second_meeting"] & spread.notna() & spread.gt(0.0)
    away_favorite = merged["second_meeting"] & spread.notna() & spread.lt(0.0)
    flag = np.where(home_favorite, 1.0, np.where(away_favorite, -1.0, 0.0))
    return pd.DataFrame(
        {"game_id": merged["game_id"].astype(str), SECOND_MEETING_FAVORITE_COLUMN: flag}
    )


def attach_second_meeting_favorite_features(
    features: pd.DataFrame,
    *,
    repo_root: Path | None = None,
    schedule: pd.DataFrame | None = None,
    opener_lines: pd.DataFrame | None = None,
    market_root: Path | None = None,
) -> pd.DataFrame:
    """Additively join ``crew_second_meeting_favorite_flag`` onto ``features``."""

    root = repo_root or REPO_ROOT

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        lines = (
            opener_lines
            if opener_lines is not None
            else default_opener_lines(sched, market_root=market_root)
        )
        return derive_second_meeting_favorite_features(root, lines)

    merged = _attach(features, schedule, _derive, (SECOND_MEETING_FAVORITE_COLUMN,))
    merged[SECOND_MEETING_FAVORITE_COLUMN] = merged[SECOND_MEETING_FAVORITE_COLUMN].fillna(0.0)
    return merged


# ---------------------------------------------------------------------------
# LEAD-31: rookie-referee tenure (left-censoring disclosure) -> take the
# underdog.
# ---------------------------------------------------------------------------


def describe_referee_left_censoring(repo_root: Path | None = None) -> dict[str, Any]:
    """Count of officials whose first dataset-visible season is 2015
    (censored -- unknown true tenure) vs. a genuine 2016-2025 debut."""

    root = repo_root or REPO_ROOT
    officials_path, _game_penalties_path, _snapshot_id = _latest_officials_snapshot(root)
    officials = pd.read_parquet(officials_path)
    refs = officials.loc[
        (officials["position"] == _REFEREE_POSITION)
        & (officials["season_type"] == _REFEREE_SEASON_TYPE)
    ]
    first_season = refs.groupby("official_name")["season"].min()
    n_censored = int((first_season == 2015).sum())
    n_genuine = int((first_season >= 2016).sum())
    return {
        "n_officials_total": len(first_season),
        "n_censored_2015_debut": n_censored,
        "n_genuine_debut_2016_2025": n_genuine,
    }


def rookie_crew_table(
    repo_root: Path | None = None, *, trait: pd.DataFrame | None = None
) -> pd.DataFrame:
    """``(game_id, official_name, season, prior_seasons_experience)`` --
    reused verbatim from ``_build_referee_trait_data``'s own game_trait.

    ``trait``, when given (tests), must already carry those four columns
    and is used instead of loading real snapshots via ``repo_root``.
    """

    if trait is not None:
        return trait[["game_id", "official_name", "season", "prior_seasons_experience"]].copy()
    root = repo_root or REPO_ROOT
    built = _build_referee_trait_data(root).game_trait
    return built[["game_id", "official_name", "season", "prior_seasons_experience"]].copy()


def derive_rookie_crew_underdog_features(
    repo_root: Path | None,
    opener_lines: pd.DataFrame,
    *,
    trait: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return ``(game_id, rookie_crew_underdog_flag)`` for every matched game.

    ``+1`` when the crew is a rookie crew (``prior_seasons_experience`` in
    ``{0, 1}``, restricted to ``season >= ROOKIE_ELIGIBLE_SEASON_FLOOR``) AND
    the home team is the underdog at the Tuesday opener; ``-1`` when rookie
    crew AND the away team is the underdog; ``0`` otherwise. ``trait``, when
    given (tests), is ``rookie_crew_table``-shaped and used instead of
    loading real snapshots.
    """

    if "game_id" not in opener_lines.columns:
        raise DataContractError("opener_lines is missing the game_id join key")

    trait = rookie_crew_table(repo_root, trait=trait)
    eligible_season = trait["season"] >= ROOKIE_ELIGIBLE_SEASON_FLOOR
    is_rookie = trait["prior_seasons_experience"].le(ROOKIE_PRIOR_EXPERIENCE_MAX)
    rookie_crew = eligible_season & is_rookie

    merged = trait.assign(rookie_crew=rookie_crew).merge(
        opener_lines[["game_id", "tue_open_home_spread"]], on="game_id", how="left"
    )
    spread = merged["tue_open_home_spread"]
    home_dog = merged["rookie_crew"] & spread.notna() & spread.lt(0.0)
    away_dog = merged["rookie_crew"] & spread.notna() & spread.gt(0.0)
    flag = np.where(home_dog, 1.0, np.where(away_dog, -1.0, 0.0))
    return pd.DataFrame(
        {"game_id": merged["game_id"].astype(str), ROOKIE_CREW_UNDERDOG_COLUMN: flag}
    )


def attach_rookie_crew_underdog_features(
    features: pd.DataFrame,
    *,
    repo_root: Path | None = None,
    schedule: pd.DataFrame | None = None,
    opener_lines: pd.DataFrame | None = None,
    market_root: Path | None = None,
) -> pd.DataFrame:
    """Additively join ``rookie_crew_underdog_flag`` onto ``features``."""

    root = repo_root or REPO_ROOT

    def _derive(sched: pd.DataFrame) -> pd.DataFrame:
        lines = (
            opener_lines
            if opener_lines is not None
            else default_opener_lines(sched, market_root=market_root)
        )
        return derive_rookie_crew_underdog_features(root, lines)

    merged = _attach(features, schedule, _derive, (ROOKIE_CREW_UNDERDOG_COLUMN,))
    merged[ROOKIE_CREW_UNDERDOG_COLUMN] = merged[ROOKIE_CREW_UNDERDOG_COLUMN].fillna(0.0)
    return merged


__all__ = [
    "CREW_HOME_BIAS_COLUMN",
    "ROOKIE_CREW_UNDERDOG_COLUMN",
    "ROOKIE_ELIGIBLE_SEASON_FLOOR",
    "ROOKIE_PRIOR_EXPERIENCE_MAX",
    "SECOND_MEETING_FAVORITE_COLUMN",
    "TRAILING_HOME_BIAS_MIN_GAMES",
    "attach_crew_home_bias_features",
    "attach_rookie_crew_underdog_features",
    "attach_second_meeting_favorite_features",
    "crew_familiarity_table",
    "derive_crew_home_bias_features",
    "derive_rookie_crew_underdog_features",
    "derive_second_meeting_favorite_features",
    "describe_crew_familiarity",
    "describe_referee_left_censoring",
    "home_away_penalty_game_table",
    "officials_home_bias_reliability",
    "rookie_crew_table",
    "trailing_home_bias_table",
]
