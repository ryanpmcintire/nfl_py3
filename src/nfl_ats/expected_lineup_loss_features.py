"""Expected lineup loss from the play-probability model (LEAD-62).

Predeclared in ``docs/expected_lineup_loss.md`` before any candidate arm was
scored. For each team-game, sum over that team's WEEK-OF depth-chart
starters (``depth_rank == 1`` at each position slot) of
``(1 - P(plays)) * trailing-4-week snap share``, split into three groups
(``qb``, ``offense``, ``defense``); the signed production column for each
group is ``home - away``. ``P(plays)`` is
``nfl_ats.play_probability``'s own walk-forward, isotonic-calibrated model
(imported, never edited); the panel it trains on is the one lane AB built
(``data/processed/play_probability_panel.parquet``).

Safety follows the corrected play-probability implementation: shared pool
cutoff (kickoff or Sunday 16:00 Eastern), visible injury revisions, strictly
earlier daily depth observations, and disjoint chronological calibration.
Legacy week-labelled depth rows retain that archive's pregame assumption;
their actual sub-week observation time cannot be verified. Snap-history
features must come from ``play_probability.attach_history_features``.

The old 24-hour timing and training descriptions in the frozen predeclaration
are superseded by the owner's CX5 task; its formula is unchanged. The current
probability model does not consume roster_status; the compatibility column is
still forced to ACT. See the appended results in ``docs/expected_lineup_loss.md``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from nfl_ats.availability import practice_category, report_category
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.nfl_week import pool_decision_cutoff
from nfl_ats.play_probability import (
    QB1_NOT_APPLICABLE,
    PlayProbabilityModel,
    fit_play_probability_model,
    predict_play_probabilities,
)

#: ``nfl_ats.lineup_availability.depth_chart_position_group`` buckets that
#: count as "offense" for this construct (the QB row is pulled out of
#: "skill" and counted separately -- see the module docstring / predeclaration).
OFFENSE_POSITION_GROUPS: tuple[str, ...] = ("offensive_line", "skill")
#: ... and "defense".
DEFENSE_POSITION_GROUPS: tuple[str, ...] = ("front", "secondary")

LINEUP_GROUPS: tuple[str, ...] = ("qb", "offense", "defense")

EXPECTED_LINEUP_LOSS_OFFENSE_COLUMN = "diff_expected_lineup_loss_offense"
EXPECTED_LINEUP_LOSS_DEFENSE_COLUMN = "diff_expected_lineup_loss_defense"
EXPECTED_LINEUP_LOSS_QB_COLUMN = "diff_expected_lineup_loss_qb"
EXPECTED_LINEUP_LOSS_COLUMNS: tuple[str, ...] = (
    EXPECTED_LINEUP_LOSS_OFFENSE_COLUMN,
    EXPECTED_LINEUP_LOSS_DEFENSE_COLUMN,
    EXPECTED_LINEUP_LOSS_QB_COLUMN,
)

_PANEL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "team",
    "gsis_id",
    "position",
    "position_group",
    "depth_rank",
    "depth_rank_bucket",
    "season_week",
    "weeks_since_last_snap",
    "trailing4_snap_share",
    "decision_at",
    "source_schema",
)
_INJURY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "team",
    "gsis_id",
    "report_status",
    "practice_status",
    "effective_observed_at",
)
_GAMES_REQUIRED_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "home_team",
    "away_team",
    "kickoff",
)


def team_week_decision_instants(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game at min(kickoff, Sunday 16:00 Eastern)."""

    require_columns(games, _GAMES_REQUIRED_COLUMNS, "games")
    long = pd.concat(
        [
            games.rename(columns={"home_team": "team"})[["season", "week", "team", "kickoff"]],
            games.rename(columns={"away_team": "team"})[["season", "week", "team", "kickoff"]],
        ],
        ignore_index=True,
    )
    long["kickoff"] = pd.to_datetime(long["kickoff"], errors="coerce", utc=True)
    long = long.loc[long["kickoff"].notna()].copy()
    long["decision_at"] = long["kickoff"].map(pool_decision_cutoff)
    return long.drop_duplicates(["season", "week", "team"]).reset_index(drop=True)


def visible_injury_lookup(
    injuries: pd.DataFrame, decision_at_by_team_week: pd.DataFrame
) -> pd.DataFrame:
    """Injuries visible strictly at-or-before each (season, week, team)'s own
    ``decision_at`` -- the same ``<= decision_at`` rule
    ``nfl_ats.players._injury_rows_asof`` applies to PRODUCTION's own
    ``diff_injury_*`` block (duplicated, not imported; see module docstring).
    One row per (season, week, team, gsis_id): the LATEST visible revision.
    """

    require_columns(injuries, _INJURY_REQUIRED_COLUMNS, "injuries")
    require_columns(
        decision_at_by_team_week, ("season", "week", "team", "decision_at"), "decision_at table"
    )
    merged = injuries.merge(
        decision_at_by_team_week[["season", "week", "team", "decision_at"]],
        on=["season", "week", "team"],
        how="inner",
    )
    merged["effective_observed_at"] = pd.to_datetime(
        merged["effective_observed_at"], errors="coerce", utc=True
    )
    visible = merged.loc[
        merged["effective_observed_at"].notna()
        & merged["effective_observed_at"].le(merged["decision_at"])
    ].copy()
    visible = visible.sort_values("effective_observed_at").drop_duplicates(
        ["season", "week", "team", "gsis_id"], keep="last"
    )
    return visible[["season", "week", "team", "gsis_id", "report_status", "practice_status"]]


def _lineup_group(position: pd.Series, position_group: pd.Series) -> pd.Series:
    upper_position = position.astype("string").str.upper()
    is_qb = upper_position.eq("QB")
    is_offense = position_group.isin(OFFENSE_POSITION_GROUPS) & ~is_qb
    is_defense = position_group.isin(DEFENSE_POSITION_GROUPS)
    return pd.Series(
        np.select(
            [is_qb, is_offense, is_defense], ["qb", "offense", "defense"], default="excluded"
        ),
        index=position.index,
    )


def select_week_starters(panel: pd.DataFrame) -> pd.DataFrame:
    """Depth-rank-1 rows only, tagged with their ``lineup_group``
    (``qb``/``offense``/``defense``); special-teams-only starters
    (``position_group == "other"``) are dropped."""

    require_columns(panel, _PANEL_REQUIRED_COLUMNS, "play_probability panel")
    visible = _visible_panel(panel)
    starters = visible.loc[pd.to_numeric(visible["depth_rank"], errors="coerce").eq(1)].copy()
    starters["lineup_group"] = _lineup_group(starters["position"], starters["position_group"])
    return starters.loc[starters["lineup_group"].ne("excluded")].reset_index(drop=True)


def _visible_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep only pre-decision depth observations; legacy weeks are archive proxies.

    A prepared panel must carry the same decision cutoff as the game. History
    columns are supplied by play_probability.attach_history_features. Never
    accept a post-decision daily depth row, even as model-training evidence.
    """
    require_columns(panel, ("decision_at", "source_schema"), "play_probability panel")
    decision = pd.to_datetime(panel["decision_at"], utc=True, errors="coerce")
    observed = pd.to_datetime(
        panel.get("depth_observed_at", pd.Series(pd.NaT, index=panel.index)), utc=True
    )
    legacy = panel["source_schema"].eq("legacy_week")
    visible = decision.notna() & (legacy | observed.lt(decision))
    # When precise timestamps exist they take precedence over a legacy proxy.
    visible &= observed.isna() | observed.lt(decision)
    result = panel.loc[visible].copy()
    if "depth_observed_at" in result:
        result = result.sort_values("depth_observed_at", na_position="first")
    return result.drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")


def attach_asof_injury_features(
    starters: pd.DataFrame, injury_lookup: pd.DataFrame
) -> pd.DataFrame:
    """Replace the panel's own (final-week-status) injury columns with the
    asof-visible ones, and force ``roster_status`` to ``"ACT"`` -- see the
    module docstring's "Strict pregame safety" section."""

    result = starters.reset_index(drop=True).copy()
    result["_row"] = np.arange(len(result))
    merged = result.merge(
        injury_lookup, on=["season", "week", "team", "gsis_id"], how="left", validate="m:1"
    )
    merged = merged.drop_duplicates("_row").set_index("_row").reindex(result["_row"])
    result["report_category"] = [report_category(value) for value in merged["report_status"]]
    result["practice_category"] = [practice_category(value) for value in merged["practice_status"]]
    result["roster_status"] = "ACT"
    result["has_injury_designation"] = (
        merged["report_status"].notna().to_numpy() | merged["practice_status"].notna().to_numpy()
    )
    is_qb = result["lineup_group"].eq("qb")
    result["qb1_report_category"] = np.where(is_qb, result["report_category"], QB1_NOT_APPLICABLE)
    result["qb1_practice_category"] = np.where(
        is_qb, result["practice_category"], QB1_NOT_APPLICABLE
    )
    return result.drop(columns="_row")


def attach_play_probabilities(
    starters: pd.DataFrame, panel: pd.DataFrame, *, scored_seasons: Iterable[int] | None = None
) -> pd.DataFrame:
    """Walk-forward ``play_probability`` for every starter row: for each
    scored season Y, ``fit_play_probability_model(panel, scored_season=Y)``
    (fit on every season strictly before Y, calibrated on Y-1) predicts every
    starter row whose own ``season`` is Y. A season with no strictly-prior
    training season (2013, the panel's first) can never be scored and is
    dropped -- the same walk-forward floor
    ``docs/play_probability_model.md``'s own evaluation starts at 2014."""

    panel = _visible_panel(panel)
    available_seasons = sorted(int(value) for value in panel["season"].unique())
    candidate_seasons = (
        sorted(int(value) for value in scored_seasons)
        if scored_seasons is not None
        else available_seasons[1:]
    )
    result = starters.reset_index(drop=True).copy()
    result["play_probability"] = np.nan
    fitted_models: dict[int, PlayProbabilityModel] = {}
    for season in candidate_seasons:
        mask = result["season"].eq(season)
        if not mask.any():
            continue
        training = panel.loc[panel["season"].lt(season)]
        if training.empty:
            continue
        model = fit_play_probability_model(training, scored_season=season)
        fitted_models[season] = model
        predictions = predict_play_probabilities(model, result.loc[mask])
        result.loc[mask, "play_probability"] = predictions["play_probability"].to_numpy()
    result.attrs["fitted_models"] = fitted_models
    return result


def team_week_expected_loss(starters_with_probability: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) with the three group totals.

    A starter with no snap history (``trailing4_snap_share`` is ``NaN``)
    contributes zero to the sum -- see the module docstring's predeclared
    "documented choice"."""

    require_columns(starters_with_probability, ("play_probability",), "starters")
    working = starters_with_probability.loc[
        starters_with_probability["play_probability"].notna()
    ].copy()
    trailing_share = pd.to_numeric(working["trailing4_snap_share"], errors="coerce").fillna(0.0)
    working["expected_snap_share_lost"] = (1.0 - working["play_probability"]) * trailing_share
    totals = (
        working.groupby(["season", "week", "team", "lineup_group"], observed=True)[
            "expected_snap_share_lost"
        ]
        .sum()
        .unstack("lineup_group", fill_value=0)
        .reset_index()
    )
    for group in LINEUP_GROUPS:
        if group not in totals.columns:
            totals[group] = 0.0
    return totals.rename(
        columns={
            "qb": "expected_lineup_loss_qb",
            "offense": "expected_lineup_loss_offense",
            "defense": "expected_lineup_loss_defense",
        }
    )[
        [
            "season",
            "week",
            "team",
            "expected_lineup_loss_qb",
            "expected_lineup_loss_offense",
            "expected_lineup_loss_defense",
        ]
    ]


def attach_expected_lineup_loss_features(
    base_features: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    scored_seasons: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Attach the three ``diff_expected_lineup_loss_*`` columns to
    ``base_features`` (home minus away). ``base_features`` must already
    carry ``season``/``week``/``home_team``/``away_team``/``kickoff``
    (PRODUCTION's ``game_features_weak_stack.parquet`` does). ``panel`` is
    ``nfl_ats.play_probability``'s own training panel
    (``data/processed/play_probability_panel.parquet``); ``injuries`` is the
    SAME player snapshot's raw, already-``week_proxy``-canonicalized
    ``injuries.parquet`` (carries ``effective_observed_at``).

    Never writes ``data/processed`` itself -- the caller decides where the
    augmented table is written, if at all.
    """

    decision_at = team_week_decision_instants(base_features)
    injury_lookup = visible_injury_lookup(injuries, decision_at)
    scoring_panel = panel.merge(
        decision_at[["season", "week", "team", "decision_at"]],
        on=["season", "week", "team"],
        how="inner",
        suffixes=("_panel", ""),
        validate="m:1",
    )
    if (
        not pd.to_datetime(scoring_panel["decision_at_panel"], utc=True)
        .eq(scoring_panel["decision_at"])
        .all()
    ):
        raise DataContractError("Rebuild play-probability panel at the game's pool decision cutoff")
    starters = select_week_starters(scoring_panel.drop(columns="decision_at_panel"))
    starters = attach_asof_injury_features(starters, injury_lookup)
    starters = attach_play_probabilities(starters, panel, scored_seasons=scored_seasons)
    team_week_loss = team_week_expected_loss(starters)

    result = base_features.copy()
    for side in ("home", "away"):
        side_columns = {
            "team": f"{side}_team",
            "expected_lineup_loss_qb": f"{side}_expected_lineup_loss_qb",
            "expected_lineup_loss_offense": f"{side}_expected_lineup_loss_offense",
            "expected_lineup_loss_defense": f"{side}_expected_lineup_loss_defense",
        }
        renamed = team_week_loss.rename(columns=side_columns)
        result = result.merge(
            renamed, on=["season", "week", side_columns["team"]], how="left", validate="m:1"
        )
    for group, column in (
        ("qb", EXPECTED_LINEUP_LOSS_QB_COLUMN),
        ("offense", EXPECTED_LINEUP_LOSS_OFFENSE_COLUMN),
        ("defense", EXPECTED_LINEUP_LOSS_DEFENSE_COLUMN),
    ):
        result[column] = (
            result[f"home_expected_lineup_loss_{group}"]
            - result[f"away_expected_lineup_loss_{group}"]
        )
    return result


def team_season_split_half_reliability(team_week_loss: pd.DataFrame) -> dict[str, float]:
    """Odd/even-week split-half Pearson reliability of the team-season mean
    COMBINED expected loss (``qb + offense + defense``, before home/away
    signing) -- the predeclared reliability measure in
    ``docs/expected_lineup_loss.md``."""

    require_columns(
        team_week_loss,
        (
            "season",
            "week",
            "team",
            "expected_lineup_loss_qb",
            "expected_lineup_loss_offense",
            "expected_lineup_loss_defense",
        ),
        "team_week_loss",
    )
    working = team_week_loss.copy()
    working["combined"] = (
        working["expected_lineup_loss_qb"]
        + working["expected_lineup_loss_offense"]
        + working["expected_lineup_loss_defense"]
    )
    working["half"] = np.where(
        pd.to_numeric(working["week"], errors="coerce") % 2 == 1, "odd", "even"
    )
    pivot = (
        working.groupby(["season", "team", "half"], observed=True)["combined"]
        .mean()
        .unstack("half")
        .dropna()
    )
    if len(pivot) < 3 or "odd" not in pivot.columns or "even" not in pivot.columns:
        return {"n_team_seasons": len(pivot), "reliability": float("nan")}
    correlation = float(np.corrcoef(pivot["odd"], pivot["even"])[0, 1])
    return {"n_team_seasons": len(pivot), "reliability": correlation}


__all__ = [
    "DEFENSE_POSITION_GROUPS",
    "EXPECTED_LINEUP_LOSS_COLUMNS",
    "EXPECTED_LINEUP_LOSS_DEFENSE_COLUMN",
    "EXPECTED_LINEUP_LOSS_OFFENSE_COLUMN",
    "EXPECTED_LINEUP_LOSS_QB_COLUMN",
    "LINEUP_GROUPS",
    "OFFENSE_POSITION_GROUPS",
    "attach_asof_injury_features",
    "attach_expected_lineup_loss_features",
    "attach_play_probabilities",
    "select_week_starters",
    "team_season_split_half_reliability",
    "team_week_decision_instants",
    "team_week_expected_loss",
    "visible_injury_lookup",
]
