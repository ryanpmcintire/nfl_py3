"""XLG-04 cross-league role-delivery replication (predeclared).

This module is the predeclared XLG-04 experiment; see ``docs/cfb_role_replication.md``
for the full predeclaration (frozen before any delivery or absence result was
computed). The frozen configuration below (``FROZEN_ROLE_SEASONS``,
``FROZEN_ROLE_SPAN``, ``FROZEN_MIN_PRIOR_APPEARANCES``,
``FROZEN_ROLE_THRESHOLDS``, ``FROZEN_MIN_TEAM_ACTIONS``,
``FROZEN_CREDIT_COVERAGE_MIN``, and ``FROZEN_REPLICATION_GATES``) mirrors that
document exactly. It replicates the NFL PER-12 "expected role delivery"
diagnostic (``artifacts/role_delivery_experiment/*/predeclaration.md``,
2026-08-16) on college football play-by-play, asking whether a team's
"expected deliverer" of a role (dropbacks, carries, receptions) actually
gets credit for it at close to their trailing rate, and whether that pattern
looks the same across leagues.

CFB has no injury report, no lineup/depth chart with observation timestamps,
and no snap counts -- unlike the NFL side, there is no ground truth for *why*
a credited player didn't appear. Every CFB "absence" event built here is
therefore a **proxy label only**: a qualifying player who received zero
credited actions in an otherwise-valid team-game. Per the XLG-02 play-
participant contract (``nfl_ats.cfb.CFB_PARTICIPANT_AVAILABILITY_CONTRACT``),
**absence of credit is never evidence of absence** -- it can mean the player
was benched, injured, played but went uncredited (see the coverage gate
below), or simply had a quiet game. Absence rows are reported for
description only; they are never used to gate replication.

The gated replication target is instead the **delivery sample**: appearances
that already condition on positive participation evidence (at least one
credited action). Delivery asks a strictly easier, more measurable question
than absence -- "given the player showed up in the box score, did their
share match their trailing role" -- and it is the only one of the two the
predeclared gates (:data:`FROZEN_REPLICATION_GATES`) score.

Module layout
-------------
1. League-neutral core: :func:`build_role_states` (appearance-only EWM role
   state, mirroring ``nfl_ats.players``' role-state update exactly),
   :func:`build_delivery_frame`, :func:`build_absence_frame`,
   :func:`summarize_delivery`, :func:`evaluate_replication_gates`.
2. League adapters: :func:`cfb_role_actions` (ESPN CFB play-by-play) and
   :func:`nfl_role_actions` (the canonical ``nfl_ats.role_actions`` weekly
   action-count snapshot), both producing the same
   actions/team-games/coverage shape.
3. :func:`run_role_replication`, the end-to-end runner the CLI wires up.

Known, flagged deviation from a literal reading of the predeclared spec
------------------------------------------------------------------------
The predeclared spec for the CFB regular-season filter reads "keep rows
where ``seasonType`` string starts with 'regular' case-insensitively." Real
``data/cfb/pbp`` snapshots do not carry that string: ``seasonType`` is the
*numeric* code documented in ``docs/cfb_data.md`` and used throughout
``nfl_ats.cfb``/``nfl_ats.cfb_features`` (``CFB_PBP_SEASON_TYPE_CODES``,
``"2"`` = regular season; ``tests/conftest.py``'s own CFB pbp fixture uses
``seasonType=2``). A literal string-prefix match would silently keep zero
rows against real data. :func:`_regular_season_mask` therefore checks
*both*: the numeric code mapped through ``CFB_PBP_SEASON_TYPE_CODES`` (the
path real snapshots take) and a case-insensitive "regular" prefix match (so
a caller that already has a human-readable value, as the literal spec
describes, still works). This is recorded here and in the implementation
report, not applied silently.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.cfb import CFB_PBP_SEASON_TYPE_CODES
from nfl_ats.data import DataContractError, require_columns

# ---------------------------------------------------------------------------
# Frozen configuration (fixed before any delivery/absence result existed)
# ---------------------------------------------------------------------------

FROZEN_ROLE_SEASONS: tuple[int, int] = (2013, 2025)  # inclusive, both leagues, regular season only
FROZEN_ROLE_SPAN: int = 8  # EWM alpha = 2/(span+1), mirroring nfl_ats.players role_span
FROZEN_MIN_PRIOR_APPEARANCES: int = 3
FROZEN_ROLE_THRESHOLDS: dict[str, float] = {"dropback": 0.50, "carry": 0.20, "reception": 0.15}
FROZEN_MIN_TEAM_ACTIONS: dict[str, int] = {"dropback": 10, "carry": 10, "reception": 5}
FROZEN_CREDIT_COVERAGE_MIN: float = 0.95  # CFB per-(season, action_type) measurement gate
FROZEN_REPLICATION_GATES: dict[str, float] = {
    "median_low": 0.90,
    "median_high": 1.10,
    "median_league_gap_max": 0.10,
    "severe_under_delivery_max": 0.15,
}

ACTION_TYPES: tuple[str, ...] = ("dropback", "carry", "reception")

# The shared league-neutral schemas produced by both league adapters.
ROLE_ACTION_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "order_key",
    "team",
    "player_id",
    "action_type",
    "count",
    "team_total",
)
TEAM_GAME_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "order_key",
    "team",
    "action_type",
    "team_total",
)
COVERAGE_COLUMNS: tuple[str, ...] = (
    "league",
    "season",
    "action_type",
    "eligible_plays",
    "credited_plays",
    "coverage",
    "excluded",
    "note",
)
ROLE_STATE_COLUMNS: tuple[str, ...] = (
    *ROLE_ACTION_COLUMNS,
    "share",
    "prior_share",
    "prior_appearances",
)
ABSENCE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "team",
    "player_id",
    "action_type",
    "prior_share",
    "top_replacement_share",
    "top_replacement_prior_share",
    "team_total",
    "team_total_trailing",
)

# The pbp columns the CFB adapter reads; exported so the CLI can request
# exactly this subset from ``nfl_ats.cfb_features.load_cfb_seasons``.
CFB_ROLE_PBP_LOAD_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "seasonType",
    "pos_team",
    "pass",
    "rush",
    "type.text",
    "passer_player_id",
    "rusher_player_id",
    "receiver_player_id",
)
_NFL_ROLE_STATS_COLUMNS: tuple[str, ...] = (
    "player_id",
    "season",
    "week",
    "game_id",
    "team",
    "attempts",
    "carries",
    "receptions",
    "sacks_taken",
)


# ---------------------------------------------------------------------------
# 1. League-neutral core
# ---------------------------------------------------------------------------


def build_role_states(actions: pd.DataFrame) -> pd.DataFrame:
    """Compute each player's appearance-only span-8 EWM role state.

    ``actions`` holds one row per (game, team, player, action_type) with
    ``count >= 1`` -- appearance rows only, exactly as produced by
    :func:`cfb_role_actions`/:func:`nfl_role_actions`. For each
    ``(team, player_id, action_type)`` group, rows are processed in
    chronological order (``order_key`` then ``game_id``):
    ``prior_share`` is the state *before* this appearance (``NaN`` when
    there is none yet), ``prior_appearances`` is how many appearance rows
    preceded it, and the state is then updated to ``share`` (first
    appearance) or ``alpha * share + (1 - alpha) * state`` with
    ``alpha = 2 / (FROZEN_ROLE_SPAN + 1)``.

    This exactly mirrors ``nfl_ats.players.enrich_with_player_features``'s
    role-state update (``role_states[team].setdefault(...)``,
    ``alpha * value + (1 - alpha) * previous``): no zero-imputation for a
    missed game (a missed game simply has no row here), no cross-season
    decay, and state kept within team so a player changing teams starts a
    fresh state (the groupby key includes ``team``).
    """

    require_columns(actions, ROLE_ACTION_COLUMNS, "role actions")
    if (pd.to_numeric(actions["count"], errors="coerce") < 1).any():
        raise DataContractError("role actions must be appearance rows with count >= 1")

    working = actions.copy()
    working["count"] = pd.to_numeric(working["count"], errors="coerce")
    working["team_total"] = pd.to_numeric(working["team_total"], errors="coerce")
    working["share"] = working["count"] / working["team_total"]
    ordered = working.sort_values(["team", "player_id", "action_type", "order_key", "game_id"])

    alpha = 2.0 / (FROZEN_ROLE_SPAN + 1.0)
    prior_share = pd.Series(np.nan, index=ordered.index, dtype="float64")
    prior_appearances = pd.Series(0, index=ordered.index, dtype="int64")
    for _, group in ordered.groupby(["team", "player_id", "action_type"], sort=False):
        state = math.nan
        for appearances, (index, row) in enumerate(group.iterrows()):
            prior_share.at[index] = state
            prior_appearances.at[index] = appearances
            share = float(row["share"])
            state = share if not math.isfinite(state) else alpha * share + (1.0 - alpha) * state

    working["prior_share"] = prior_share
    working["prior_appearances"] = prior_appearances
    return working.loc[:, list(ROLE_STATE_COLUMNS)]


def build_delivery_frame(
    states: pd.DataFrame,
    thresholds: dict[str, float] = FROZEN_ROLE_THRESHOLDS,
    min_prior: int = FROZEN_MIN_PRIOR_APPEARANCES,
) -> pd.DataFrame:
    """Appearances whose *prior* state already qualified as a delivered role.

    ``ratio = share / prior_share`` compares the realized share for this
    appearance against the trailing role it is scored against -- above 1
    means the player delivered more than their trailing role, below 1
    means less. Unlike the NFL PER-12 diagnostic this is left unclipped.
    """

    require_columns(states, ROLE_STATE_COLUMNS, "role states")
    threshold = states["action_type"].map(thresholds)
    qualifies = (
        states["prior_share"].notna()
        & states["prior_share"].ge(threshold)
        & states["prior_appearances"].ge(min_prior)
    )
    result = states.loc[qualifies].copy()
    result["ratio"] = result["share"] / result["prior_share"]
    return result.reset_index(drop=True)


def _team_total_trailing(team_games: pd.DataFrame, span: int) -> pd.Series:
    """Span-``span`` EWM of each ``(team, action_type)``'s prior ``team_total``."""

    alpha = 2.0 / (span + 1.0)
    trailing = pd.Series(np.nan, index=team_games.index, dtype="float64")
    for _, group in team_games.groupby(["team", "action_type"], sort=False):
        state = math.nan
        for index, row in group.iterrows():
            trailing.at[index] = state
            value = float(row["team_total"])
            state = value if not math.isfinite(state) else alpha * value + (1.0 - alpha) * state
    return trailing


def build_absence_frame(
    team_games: pd.DataFrame,
    states: pd.DataFrame,
    thresholds: dict[str, float] = FROZEN_ROLE_THRESHOLDS,
    min_prior: int = FROZEN_MIN_PRIOR_APPEARANCES,
    min_team_actions: dict[str, int] = FROZEN_MIN_TEAM_ACTIONS,
    span: int = FROZEN_ROLE_SPAN,
) -> pd.DataFrame:
    """One row per (team-game, action_type, qualifying player) the player missed.

    ``team_games`` is the *full* set of team-games per ``(team, action_type)``
    -- not just appearance rows -- restricted here (defensively; the league
    adapters already apply this) to ``team_total >= min_team_actions``. A
    player becomes trackable only after their first credited appearance for
    a ``(team, action_type)``; before that there is no prior role to be
    absent from.

    Once trackable, the player's qualifying state does **not** update on a
    missed game -- it mirrors the appearance-only EWM in
    :func:`build_role_states` exactly, carrying the post-appearance state
    forward unchanged until the next appearance. A qualifying player who
    misses several team-games in a row is therefore reported as absent in
    every one of them at the *same* ``prior_share``, and the streak ends
    their qualification only when a later appearance changes the state
    (potentially below threshold) -- this is intended, not a bug: it is the
    literal consequence of "the state does not update on absence."

    Every row here is a proxy label (see the module docstring): it is never
    evidence the player was actually unavailable, only that they received
    zero credited actions in an otherwise-valid team-game.
    """

    require_columns(team_games, TEAM_GAME_COLUMNS, "team games")
    require_columns(states, ROLE_STATE_COLUMNS, "role states")

    games = team_games.copy()
    games["team"] = games["team"].astype(str)
    games["action_type"] = games["action_type"].astype(str)
    games["game_id"] = games["game_id"].astype(str)
    required_actions = games["action_type"].map(min_team_actions)
    games = games.loc[games["team_total"].ge(required_actions)].copy()
    games = games.sort_values(["team", "action_type", "order_key", "game_id"])
    games["team_total_trailing"] = _team_total_trailing(games, span)

    working_states = states.copy()
    working_states["team"] = working_states["team"].astype(str)
    working_states["player_id"] = working_states["player_id"].astype(str)
    working_states["action_type"] = working_states["action_type"].astype(str)
    working_states["game_id"] = working_states["game_id"].astype(str)

    player_appearance: dict[tuple[str, str, str, str], float] = {}
    for _, row in working_states.iterrows():
        appearance_key = (
            str(row["team"]),
            str(row["player_id"]),
            str(row["action_type"]),
            str(row["game_id"]),
        )
        player_appearance[appearance_key] = float(row["share"])

    top_replacement: dict[tuple[str, str, str], tuple[str, float, float]] = {}
    for group_key, group in working_states.groupby(["game_id", "team", "action_type"], sort=False):
        game_id_key, team_key, action_type_key = group_key
        top_position = int(np.argmax(group["share"].to_numpy()))
        top = group.iloc[top_position]
        top_prior = top["prior_share"]
        top_replacement[(str(game_id_key), str(team_key), str(action_type_key))] = (
            str(top["player_id"]),
            float(top["share"]),
            float(top_prior) if pd.notna(top_prior) else math.nan,
        )

    players_by_team_action: dict[tuple[str, str], list[str]] = {}
    for group_key, group in working_states.groupby(["team", "action_type"], sort=False):
        team_key, action_type_key = group_key
        players_by_team_action[(str(team_key), str(action_type_key))] = sorted(
            str(value) for value in group["player_id"].unique()
        )

    alpha = 2.0 / (span + 1.0)
    rows: list[dict[str, Any]] = []
    for (team, action_type), players in players_by_team_action.items():
        threshold = thresholds.get(action_type)
        if threshold is None:
            continue
        team_history = games.loc[
            games["team"].eq(team) & games["action_type"].eq(action_type)
        ].to_dict("records")
        for player_id in players:
            state = math.nan
            appearances = 0
            for game in team_history:
                game_id = str(game["game_id"])
                appearance_key = (team, player_id, action_type, game_id)
                share = player_appearance.get(appearance_key)
                if share is not None:
                    state = (
                        share if not math.isfinite(state) else alpha * share + (1.0 - alpha) * state
                    )
                    appearances += 1
                    continue
                if appearances >= min_prior and math.isfinite(state) and state >= threshold:
                    replacement = top_replacement.get((game_id, team, action_type))
                    rows.append(
                        {
                            "game_id": game_id,
                            "season": int(game["season"]),
                            "week": int(game["week"]),
                            "team": team,
                            "player_id": player_id,
                            "action_type": action_type,
                            "prior_share": state,
                            "top_replacement_share": replacement[1] if replacement else math.nan,
                            "top_replacement_prior_share": replacement[2]
                            if replacement
                            else math.nan,
                            "team_total": float(game["team_total"]),
                            "team_total_trailing": float(game["team_total_trailing"]),
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=ABSENCE_COLUMNS)
    result = pd.DataFrame(rows, columns=ABSENCE_COLUMNS)
    return result.sort_values(["season", "week", "team", "action_type", "player_id"]).reset_index(
        drop=True
    )


def summarize_delivery(delivery: pd.DataFrame) -> pd.DataFrame:
    """One row per ``action_type``: central tendency and tail behavior of ``ratio``."""

    require_columns(delivery, ("action_type", "ratio"), "delivery frame")
    columns = (
        "action_type",
        "n",
        "median_ratio",
        "mean_ratio",
        "p25_ratio",
        "p75_ratio",
        "fraction_at_or_above_one",
        "fraction_severe_under",
    )
    if delivery.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for action_type, group in delivery.groupby("action_type", sort=True):
        ratio = pd.to_numeric(group["ratio"], errors="coerce").dropna()
        n = len(ratio)
        rows.append(
            {
                "action_type": action_type,
                "n": n,
                "median_ratio": float(ratio.median()) if n else math.nan,
                "mean_ratio": float(ratio.mean()) if n else math.nan,
                "p25_ratio": float(ratio.quantile(0.25)) if n else math.nan,
                "p75_ratio": float(ratio.quantile(0.75)) if n else math.nan,
                "fraction_at_or_above_one": float((ratio >= 1.0).mean()) if n else math.nan,
                "fraction_severe_under": float((ratio <= 0.5).mean()) if n else math.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def evaluate_replication_gates(
    cfb_summary: pd.DataFrame,
    nfl_summary: pd.DataFrame,
    gates: dict[str, float] = FROZEN_REPLICATION_GATES,
) -> dict[str, dict[str, Any]]:
    """Per-action_type pass/fail against the three frozen replication gates.

    ``passed_median_band``: the CFB median ratio falls in
    ``[median_low, median_high]``. ``passed_league_gap``: the CFB and NFL
    median ratios differ by no more than ``median_league_gap_max``.
    ``passed_severe_under``: the CFB severe-under-delivery fraction
    (``ratio <= 0.5``) is at most ``severe_under_delivery_max``.
    ``replicated`` requires all three.
    """

    cfb_by_type = {str(row["action_type"]): row for row in cfb_summary.to_dict("records")}
    nfl_by_type = {str(row["action_type"]): row for row in nfl_summary.to_dict("records")}
    result: dict[str, dict[str, Any]] = {}
    for action_type in ACTION_TYPES:
        cfb_row = cfb_by_type.get(action_type)
        nfl_row = nfl_by_type.get(action_type)
        if cfb_row is None or nfl_row is None:
            result[action_type] = {
                "passed_median_band": False,
                "passed_league_gap": False,
                "passed_severe_under": False,
                "replicated": False,
                "note": "missing delivery summary rows for this action_type",
            }
            continue
        cfb_median = float(cfb_row["median_ratio"])
        nfl_median = float(nfl_row["median_ratio"])
        cfb_severe_under = float(cfb_row["fraction_severe_under"])
        passed_median_band = gates["median_low"] <= cfb_median <= gates["median_high"]
        passed_league_gap = abs(cfb_median - nfl_median) <= gates["median_league_gap_max"]
        passed_severe_under = cfb_severe_under <= gates["severe_under_delivery_max"]
        result[action_type] = {
            "cfb_median_ratio": cfb_median,
            "nfl_median_ratio": nfl_median,
            "cfb_fraction_severe_under": cfb_severe_under,
            "passed_median_band": bool(passed_median_band),
            "passed_league_gap": bool(passed_league_gap),
            "passed_severe_under": bool(passed_severe_under),
            "replicated": bool(passed_median_band and passed_league_gap and passed_severe_under),
        }
    return result


# ---------------------------------------------------------------------------
# 2. League adapters
# ---------------------------------------------------------------------------


def _regular_season_mask(season_type: pd.Series) -> pd.Series:
    """True for CFB pbp rows tagged regular season -- see the module docstring.

    Real snapshots carry the numeric ``CFB_PBP_SEASON_TYPE_CODES`` code
    (``"2"`` = regular); this also accepts a human-readable value with a
    case-insensitive "regular" prefix, matching the predeclared spec's
    literal wording for callers that already have one.
    """

    text = season_type.astype("string")
    coded = text.map(CFB_PBP_SEASON_TYPE_CODES)
    matches_code = coded.eq("regular").fillna(False)
    matches_word = text.str.strip().str.lower().str.startswith("regular").fillna(False)
    return (matches_code | matches_word).astype(bool)


def _drop_below_minimum(
    team_totals: pd.DataFrame, min_team_actions: dict[str, int]
) -> pd.DataFrame:
    required = team_totals["action_type"].map(min_team_actions)
    return team_totals.loc[team_totals["team_total"].ge(required)].copy()


def cfb_role_actions(
    pbp: pd.DataFrame, canonical_games: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adapt ESPN CFB play-by-play into the league-neutral actions/team-games/coverage shape.

    Frozen action definitions:

    - **dropback**: ``pass == True`` and ``passer_player_id`` not null,
      credited to ``passer_player_id``.
    - **carry**: ``rush == True`` and ``rusher_player_id`` not null,
      credited to ``rusher_player_id``. Kneels are included -- CFB pbp has
      no reliable, universal way to separate them out here that the NFL
      ``role_actions`` source (official ``carries``) also excludes, so
      excluding them on one side only would bias the cross-league
      comparison; this matches the NFL source's inability to exclude them.
    - **reception**: ``type.text == "Pass Reception"`` and
      ``receiver_player_id`` not null, credited to ``receiver_player_id``.

    ``canonical_games`` (the CFB canonical feature table) restricts ``pbp``
    to FBS-vs-FBS regular-season games with a resolvable spread (via an
    inner join on ``game_id``, both sides coerced to ``str``) and supplies
    ``gameday`` as ``order_key``. Seasons are restricted to
    ``FROZEN_ROLE_SEASONS``; the regular-season filter is
    :func:`_regular_season_mask`.

    **Coverage gate.** Per ``(season, action_type)``, ``coverage =
    credited_plays / eligible_plays`` (eligible: ``pass``/``rush``/
    ``type.text == "Pass Reception"`` respectively). A season below
    ``FROZEN_CREDIT_COVERAGE_MIN`` for an action_type is *excluded* from
    that action_type's ``actions``/``team_games`` (never patched) and
    flagged in the returned coverage frame.

    ``team_total`` is the sum of credited counts across players for a
    ``(game, team, action_type)`` -- so ``share = count / team_total``
    sums to 1 within a team-game-action_type by construction, matching the
    NFL adapter. Team-games below ``FROZEN_MIN_TEAM_ACTIONS`` are dropped
    from both ``actions`` and ``team_games``.
    """

    require_columns(pbp, CFB_ROLE_PBP_LOAD_COLUMNS, "cfb play_by_play")
    require_columns(canonical_games, ("game_id", "gameday"), "cfb canonical games")

    games = canonical_games.loc[:, ["game_id", "gameday"]].copy()
    games["game_id"] = games["game_id"].astype(str)
    games = games.drop_duplicates("game_id")

    working = pbp.copy()
    working["game_id"] = working["game_id"].astype(str)
    working["season"] = pd.to_numeric(working["season"], errors="coerce")
    working["week"] = pd.to_numeric(working["week"], errors="coerce")
    working = working.merge(games, on="game_id", how="inner", validate="many_to_one")
    working = working.loc[
        working["season"].notna()
        & working["week"].notna()
        & working["season"].between(FROZEN_ROLE_SEASONS[0], FROZEN_ROLE_SEASONS[1])
    ].copy()
    working = working.loc[_regular_season_mask(working["seasonType"])].copy()
    working["season"] = working["season"].astype("int64")
    working["week"] = working["week"].astype("int64")

    pass_flag = pd.to_numeric(working["pass"], errors="coerce").fillna(0).astype(bool)
    rush_flag = pd.to_numeric(working["rush"], errors="coerce").fillna(0).astype(bool)
    reception_flag = working["type.text"].astype("string").eq("Pass Reception")
    definitions: dict[str, tuple[pd.Series, pd.Series]] = {
        "dropback": (pass_flag, working["passer_player_id"]),
        "carry": (rush_flag, working["rusher_player_id"]),
        "reception": (reception_flag, working["receiver_player_id"]),
    }

    coverage_records: list[dict[str, Any]] = []
    excluded_pairs: set[tuple[int, str]] = set()
    credited_frames: list[pd.DataFrame] = []
    for action_type, (eligible, credit_column) in definitions.items():
        credited = eligible & credit_column.notna()
        by_season = (
            pd.DataFrame(
                {"season": working["season"], "_eligible": eligible, "_credited": credited}
            )
            .groupby("season", sort=True)
            .agg(eligible_plays=("_eligible", "sum"), credited_plays=("_credited", "sum"))
        )
        for season, row in by_season.iterrows():
            eligible_plays = int(row["eligible_plays"])
            credited_plays = int(row["credited_plays"])
            coverage_rate = (credited_plays / eligible_plays) if eligible_plays else math.nan
            excluded = eligible_plays == 0 or (
                not math.isnan(coverage_rate) and coverage_rate < FROZEN_CREDIT_COVERAGE_MIN
            )
            coverage_records.append(
                {
                    "league": "cfb",
                    "season": int(str(season)),
                    "action_type": action_type,
                    "eligible_plays": eligible_plays,
                    "credited_plays": credited_plays,
                    "coverage": coverage_rate,
                    "excluded": bool(excluded),
                    "note": "coverage_below_gate" if excluded else "",
                }
            )
            if excluded:
                excluded_pairs.add((int(str(season)), action_type))

        credited_rows = working.loc[
            credited, ["game_id", "season", "week", "gameday", "pos_team"]
        ].copy()
        credited_rows["action_type"] = action_type
        credited_rows["player_id"] = credit_column.loc[credited].astype(str)
        credited_frames.append(credited_rows)

    combined = pd.concat(credited_frames, ignore_index=True)
    if excluded_pairs:
        excluded_keys = pd.Series(
            list(zip(combined["season"].astype(int), combined["action_type"], strict=True)),
            index=combined.index,
        )
        combined = combined.loc[~excluded_keys.isin(excluded_pairs)].copy()

    actions = (
        combined.groupby(
            ["game_id", "season", "week", "gameday", "pos_team", "action_type", "player_id"],
            sort=False,
        )
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"pos_team": "team", "gameday": "order_key"})
    )
    team_totals = (
        actions.groupby(
            ["game_id", "season", "week", "order_key", "team", "action_type"], sort=False
        )["count"]
        .sum()
        .rename("team_total")
        .reset_index()
    )
    valid_team_games = _drop_below_minimum(team_totals, FROZEN_MIN_TEAM_ACTIONS)
    actions = actions.merge(
        valid_team_games[["game_id", "team", "action_type", "team_total"]],
        on=["game_id", "team", "action_type"],
        how="inner",
    )

    actions = actions.loc[:, list(ROLE_ACTION_COLUMNS)].reset_index(drop=True)
    team_games = valid_team_games.loc[:, list(TEAM_GAME_COLUMNS)].reset_index(drop=True)
    coverage = (
        pd.DataFrame(coverage_records, columns=COVERAGE_COLUMNS)
        .sort_values(["season", "action_type"])
        .reset_index(drop=True)
    )
    return actions, team_games, coverage


def nfl_role_actions(role_stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adapt the canonical ``nfl_ats.role_actions`` snapshot to the shared shape.

    ``dropback = attempts + sacks_taken``, ``carry = carries``,
    ``reception = receptions``; ``order_key = season * 100 + week``. Seasons
    are restricted to ``FROZEN_ROLE_SEASONS`` and the same
    ``FROZEN_MIN_TEAM_ACTIONS`` minimums are applied. The coverage gate does
    not apply to official nflverse aggregates: one row is emitted per
    ``(season, action_type)`` with ``coverage = 1.0``, ``excluded = False``,
    and ``note = "official_stats"``, keeping the frame shape identical to
    :func:`cfb_role_actions`'s coverage output.
    """

    require_columns(role_stats, _NFL_ROLE_STATS_COLUMNS, "nfl role_actions")
    working = role_stats.copy()
    working["season"] = pd.to_numeric(working["season"], errors="coerce")
    working["week"] = pd.to_numeric(working["week"], errors="coerce")
    working = working.loc[
        working["season"].notna()
        & working["week"].notna()
        & working["season"].between(FROZEN_ROLE_SEASONS[0], FROZEN_ROLE_SEASONS[1])
    ].copy()
    working["season"] = working["season"].astype("int64")
    working["week"] = working["week"].astype("int64")
    working["game_id"] = working["game_id"].astype(str)
    working["team"] = working["team"].astype(str)
    working["player_id"] = working["player_id"].astype(str)
    working["order_key"] = working["season"] * 100 + working["week"]

    working["dropback"] = pd.to_numeric(working["attempts"], errors="coerce").fillna(
        0.0
    ) + pd.to_numeric(working["sacks_taken"], errors="coerce").fillna(0.0)
    working["carry"] = pd.to_numeric(working["carries"], errors="coerce").fillna(0.0)
    working["reception"] = pd.to_numeric(working["receptions"], errors="coerce").fillna(0.0)

    long_frames: list[pd.DataFrame] = []
    for action_type in ACTION_TYPES:
        piece = working.loc[
            :, ["game_id", "season", "week", "order_key", "team", "player_id", action_type]
        ].rename(columns={action_type: "count"})
        piece["action_type"] = action_type
        long_frames.append(piece)
    long_actions = pd.concat(long_frames, ignore_index=True)

    team_totals = (
        long_actions.groupby(
            ["game_id", "season", "week", "order_key", "team", "action_type"], sort=False
        )["count"]
        .sum()
        .rename("team_total")
        .reset_index()
    )
    valid_team_games = _drop_below_minimum(team_totals, FROZEN_MIN_TEAM_ACTIONS)

    appearances = long_actions.loc[long_actions["count"] >= 1].merge(
        valid_team_games[["game_id", "team", "action_type", "team_total"]],
        on=["game_id", "team", "action_type"],
        how="inner",
    )
    actions = appearances.loc[:, list(ROLE_ACTION_COLUMNS)].reset_index(drop=True)
    team_games = valid_team_games.loc[:, list(TEAM_GAME_COLUMNS)].reset_index(drop=True)

    coverage_records: list[dict[str, Any]] = []
    for season, group in working.groupby("season", sort=True):
        for action_type in ACTION_TYPES:
            total = float(group[action_type].sum())
            coverage_records.append(
                {
                    "league": "nfl",
                    "season": int(str(season)),
                    "action_type": action_type,
                    "eligible_plays": total,
                    "credited_plays": total,
                    "coverage": 1.0,
                    "excluded": False,
                    "note": "official_stats",
                }
            )
    coverage = pd.DataFrame(coverage_records, columns=COVERAGE_COLUMNS)
    return actions, team_games, coverage


# ---------------------------------------------------------------------------
# 3. Runner
# ---------------------------------------------------------------------------


def run_role_replication(
    cfb_pbp: pd.DataFrame,
    cfb_canonical_games: pd.DataFrame,
    nfl_role_stats: pd.DataFrame,
) -> dict[str, Any]:
    """Run the end-to-end XLG-04 replication and return every intermediate frame."""

    cfb_actions, cfb_team_games, cfb_coverage = cfb_role_actions(cfb_pbp, cfb_canonical_games)
    nfl_actions, nfl_team_games, nfl_coverage = nfl_role_actions(nfl_role_stats)

    cfb_states = build_role_states(cfb_actions)
    nfl_states = build_role_states(nfl_actions)

    cfb_delivery = build_delivery_frame(cfb_states)
    nfl_delivery = build_delivery_frame(nfl_states)

    cfb_absences = build_absence_frame(cfb_team_games, cfb_states)
    nfl_absences = build_absence_frame(nfl_team_games, nfl_states)

    cfb_summary = summarize_delivery(cfb_delivery)
    nfl_summary = summarize_delivery(nfl_delivery)
    gates = evaluate_replication_gates(cfb_summary, nfl_summary)

    coverage = pd.concat([cfb_coverage, nfl_coverage], ignore_index=True)

    configuration: dict[str, Any] = {
        "role_seasons": list(FROZEN_ROLE_SEASONS),
        "role_span": FROZEN_ROLE_SPAN,
        "min_prior_appearances": FROZEN_MIN_PRIOR_APPEARANCES,
        "role_thresholds": dict(FROZEN_ROLE_THRESHOLDS),
        "min_team_actions": dict(FROZEN_MIN_TEAM_ACTIONS),
        "credit_coverage_min": FROZEN_CREDIT_COVERAGE_MIN,
        "replication_gates": dict(FROZEN_REPLICATION_GATES),
        "hypothesis_frozen_before_scoring": True,
    }

    return {
        "cfb_summary": cfb_summary.to_dict(orient="records"),
        "nfl_summary": nfl_summary.to_dict(orient="records"),
        "gates": gates,
        "coverage": coverage.to_dict(orient="records"),
        "cfb_delivery": cfb_delivery,
        "nfl_delivery": nfl_delivery,
        "cfb_absences": cfb_absences,
        "nfl_absences": nfl_absences,
        "configuration": configuration,
    }


def summarize_absences(cfb_absences: pd.DataFrame, nfl_absences: pd.DataFrame) -> pd.DataFrame:
    """Aggregate proxy-label absence events per league and action_type (CLI report only).

    Never used for gating (see the module docstring); this is descriptive
    context alongside the delivery-sample gate results.
    """

    columns = (
        "league",
        "action_type",
        "n_events",
        "median_prior_share",
        "median_top_replacement_share",
        "median_team_volume_ratio",
    )
    rows: list[dict[str, Any]] = []
    for league, absences in (("cfb", cfb_absences), ("nfl", nfl_absences)):
        if absences.empty:
            continue
        working = absences.copy()
        working["team_volume_ratio"] = working["team_total"] / working["team_total_trailing"]
        for action_type, group in working.groupby("action_type", sort=True):
            rows.append(
                {
                    "league": league,
                    "action_type": action_type,
                    "n_events": len(group),
                    "median_prior_share": float(group["prior_share"].median()),
                    "median_top_replacement_share": float(group["top_replacement_share"].median()),
                    "median_team_volume_ratio": float(group["team_volume_ratio"].median()),
                }
            )
    return pd.DataFrame(rows, columns=columns)
