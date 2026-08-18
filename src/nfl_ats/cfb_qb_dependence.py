"""CFB QB-dependence interaction feature (SPEC-6 screen; predeclared in
``docs/qb_dependence.md``, mirroring this module's shape after
``cfb_role_features.py``).

**Hypothesis** (direct quote, ``docs/pool_edge_plan.md:207-208``): *"team
output conditioned on QB reliance"* -- a team's offensive output should react
more to a swing in QB quality when the team's offense actually leans on the
QB (a pass-heavy scheme) than when it does not. The project's model is a
linear ridge regression over additive per-team-state features
(``margin.fit_margin_model``), so this interaction can only be captured if a
product term is added explicitly.

This module is CFB-only (free per ``docs/rotation_registry.md`` rule 8) and
builds three new, additive research columns from CFB play-by-play already
ingested -- no new data source, no NFL table touched:

- :data:`off_pass_rate` -- EWM share of a team's competitive offensive plays
  that are passes.
- :data:`qb_starter_epa_per_dropback` -- the CFB analogue of
  ``quarterbacks.build_qb_game_metrics`` / ``build_qb_states``: a per-player
  EWM of EPA/dropback, attached to a game via the same "most recent game's
  leading passer" identity rule the NFL production path uses
  (``players.py``'s ``latest_qb_appearance`` / ``_latest_qb_state``, **not**
  the unwired depth-chart pipeline in ``quarterbacks.py``).
- :data:`qb_dependence_interaction` -- the interaction itself, built **per
  side** and then differenced (``home_qb * home_reliance - away_qb *
  away_reliance``), the literal reading of "a team's own output conditioned
  on its own reliance" rather than a matchup-level mismatch term.

**NFL/CFB asymmetry (flagged prominently, per the task's binding
instruction).** CFB has no pregame injury/availability signal of any kind
(``docs/injury_value_lost.md`` sec 5; ``docs/cfb_data.md``), so
``qb_starter_epa_per_dropback`` here is the **raw** trailing EPA/dropback
state only -- there is no ``start_probability`` / replacement-EPA blend, and
none is attempted. The eventual NFL feature (a separate, later, separately
predeclared spec) would have both ``qb_starter_epa_per_dropback`` **and**
``qb_expected_epa_per_dropback`` (``constants.PLAYER_QB_STATE_METRICS``)
available to multiply by reliance; this CFB screen can only test the raw
half. This is a structural gap, not a shortcut -- see
``docs/injury_value_lost.md`` sec 5 for the precedent of flagging exactly
this kind of CFB data gap the same way.

**Underived constants this module DEFINES FOR ITSELF** (per the task's hard
override: do not touch or inherit ``players.py``'s ``_REPLACEMENT_QB_EPA``,
its mismatched ``qb_min_dropbacks``, or ``constants.DEFAULT_OFFSEASON_RETENTION``):

- ``CFB_QB_MIN_DROPBACKS = 20`` -- copied from ``players.py``'s production
  NFL default (``players.py:960``) as the closest-fidelity recommendation
  (``quarterbacks.build_qb_states``'s own mismatched default of 50 is not
  used either). A local constant, not an import. Judgment call, not a
  measurement (**inferred**).
- ``CFB_QB_STATE_SPAN = 12`` -- copied from ``players.py``'s ``qb_span=12``
  as a reasonable default; no independent CFB derivation (**inferred**).
- ``CFB_QB_MIN_GAME_DROPBACKS = 5`` -- the per-game-row inclusion floor
  before a (game, player) row counts as an appearance at all, copied from
  ``quarterbacks.build_qb_game_metrics``'s identical floor. A data-hygiene
  constant, not one of the three flagged-as-wrong values.
- **No offseason regression is applied to either new state** (no
  ``DEFAULT_OFFSEASON_RETENTION``-style cross-season decay toward a league
  mean). This mirrors ``cfb_role_features.py``'s own player-trail
  convention, which also applies none -- a plain, unbroken chronological EWM
  simply continues across the season boundary. Mathematically identical to
  calling ``cfb_features.build_cfb_team_states`` with ``offseason_retention
  = 1.0``, but implemented standalone here because that function iterates a
  hardcoded module-level metric tuple (``CFB_STATE_METRICS``) and cannot be
  parameterized onto a new metric without touching ``src/nfl_ats/cfb_features.py``.
- ``_REPLACEMENT_QB_EPA`` is not used at all: no ``start_probability`` blend
  exists on CFB (see the NFL/CFB asymmetry above), so there is nothing to
  blend it into.
- ``CFB_PASS_RATE_SPAN = 8`` / ``CFB_PASS_RATE_MIN_PERIODS = 3`` -- **not**
  new. This is the span/maturity every other ``CFB_STATE_METRICS`` column
  already uses (``cfb_features.py``, "NFL parameters taken verbatim").
  Reusing it is precedent, not a fresh unexamined choice.

**Design choice (inferred, flagged): both new states are built from the same
competitive-play (5-95% win-probability) subset ``cfb_features.py`` already
uses for ``CFB_STATE_METRICS``**, via ``cfb_features.cfb_competitive_plays``.
NFL's ``quarterbacks.build_qb_game_metrics`` has no win-probability filter,
so this is not a byte-for-byte port of that function -- only "the CFB
analogue," exactly as the task's spec calls for. Restricting to competitive
plays keeps the interaction's own inputs internally consistent with the
surrounding CFB feature contract it rides alongside (avoids a garbage-time
read on either half).

**Missing values are left as NaN**, not hand-imputed to a neutral constant
(unlike ``cfb_role_features.py``'s ``CONTINUITY_NEUTRAL = 1.0``): every new
column here rides directly in the same numeric feature matrix
``fit_cfb_residual_model`` already feeds through
``margin.make_margin_estimator``'s ``SimpleImputer(strategy="median",
add_indicator=True)`` step, exactly like every other ``CFB_MODEL_FEATURE_COLUMNS``
entry. No custom neutral-value convention is needed.

Module layout
-------------
1. :func:`build_cfb_qb_game_metrics` / :func:`build_cfb_qb_states` -- the CFB
   analogue of ``quarterbacks.build_qb_game_metrics`` / ``build_qb_states``.
2. :func:`build_cfb_pass_rate_team_games` / :func:`build_cfb_pass_rate_states`
   -- the new ``off_pass_rate`` team-state.
3. :func:`attach_cfb_qb_dependence` -- joins both halves onto the canonical
   CFB table via the "most recent game's leading passer" identity rule,
   producing the nine new columns (:data:`CFB_QB_DEPENDENCE_COLUMNS`).
4. :func:`build_and_attach_cfb_qb_dependence` -- one-call convenience wrapper
   from raw play-by-play + canonical games to the joined table.
5. :func:`cfb_qb_dependence_reliability` -- Step 0's split-half reliability
   audit, run BEFORE any accuracy number, mirroring
   ``docs/injury_value_lost.md`` sec 3.1's method exactly (odd/even-week
   team-season split, Pearson r, Spearman-Brown correction, block bootstrap
   CI and ``probability_positive``), on the interaction column and its two
   constituents separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nfl_ats.cfb_features import cfb_competitive_plays
from nfl_ats.data import DataContractError, require_columns

# ---------------------------------------------------------------------------
# Frozen configuration (see docs/qb_dependence.md; fixed before any run that
# touches ATS outcomes). All flagged in the module docstring above.
# ---------------------------------------------------------------------------

CFB_QB_STATE_SPAN: int = 12
CFB_QB_MIN_DROPBACKS: int = 20
CFB_QB_MIN_GAME_DROPBACKS: int = 5
CFB_PASS_RATE_SPAN: int = 8
CFB_PASS_RATE_MIN_PERIODS: int = 3

CFB_QB_DEPENDENCE_METRICS: tuple[str, ...] = (
    "off_pass_rate",
    "qb_starter_epa_per_dropback",
    "qb_dependence_interaction",
)
CFB_QB_DEPENDENCE_COLUMNS: tuple[str, ...] = tuple(
    f"{side}_{metric}" for metric in CFB_QB_DEPENDENCE_METRICS for side in ("home", "away", "diff")
)

# Reference points for Step 0's reliability gate (docs/pool_edge_plan.md,
# "Three kinds of negative"; injury_value_lost sec 3.1; cfb_role_features.md
# sec "6. Split-half reliability"). Comparisons only -- deliberately not a
# hardcoded pass/fail bar (the spec forbids picking a single fixed number
# without derivation).
RELIABILITY_NO_SPLIT_HALF_EXAMPLES: dict[str, float] = {
    "coach_ats_reputation": 0.063,
    "play_epa_dispersion": 0.014,
}
RELIABILITY_CLEARED_EXAMPLES: dict[str, float] = {
    "injury_value_lost_temporal": 0.9325,
    "cfb_role_continuity_dropback": 0.719,
    "cfb_role_continuity_carry": 0.680,
}


def _game_id_key(values: pd.Series) -> pd.Series:
    """Canonical string game-id key, regardless of the source column's numeric dtype.

    ``cfb_features.py`` joins ``pbp``/schedule ``game_id`` columns without an
    explicit dtype normalization step (they already agree in production), but
    a bare ``.astype(str)`` here would silently diverge ("20130101" vs
    "20130101.0") if a caller's ``pbp`` slice ever carried a float dtype. Used
    only for internal dict keys -- never assigned back onto a returned frame,
    so it cannot affect REG bit-identity of any existing column.
    """

    return pd.to_numeric(values, errors="raise").astype("int64").astype(str)


_QB_GAME_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "team_id",
    "passer_player_id",
    "qb_dropbacks",
    "qb_epa_per_dropback",
)


# ---------------------------------------------------------------------------
# 1. QB per-player EPA/dropback state (CFB analogue of quarterbacks.py)
# ---------------------------------------------------------------------------


def build_cfb_qb_game_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (game, team, passer) dropback count and EPA/dropback, competitive plays only.

    The CFB analogue of ``quarterbacks.build_qb_game_metrics``: credited to
    ``passer_player_id`` on ``pass == True`` rows. Restricted to the same
    competitive-play subset ``cfb_features.build_cfb_team_game_metrics`` uses
    (see module docstring) -- NOT a byte-for-byte port of the NFL function,
    which has no win-probability filter. A (game, team, passer) row is kept
    only with at least :data:`CFB_QB_MIN_GAME_DROPBACKS` dropbacks that game
    (copied from the NFL floor).
    """

    require_columns(pbp, ("passer_player_id",), "cfb play_by_play (qb dependence)")
    plays = cfb_competitive_plays(pbp)
    plays = plays.loc[
        plays["competitive_play"] & plays["pass"] & plays["passer_player_id"].notna()
    ].copy()
    if plays.empty:
        return pd.DataFrame(columns=_QB_GAME_COLUMNS)
    plays["team_id"] = pd.to_numeric(plays["pos_team_id"], errors="raise").astype("int64")
    plays["passer_player_id"] = plays["passer_player_id"].astype(str)
    plays["EPA"] = pd.to_numeric(plays["EPA"], errors="coerce")

    grouped = (
        plays.groupby(["game_id", "season", "week", "team_id", "passer_player_id"], sort=False)
        .agg(qb_dropbacks=("EPA", "size"), total_epa=("EPA", "sum"))
        .reset_index()
    )
    grouped = grouped.loc[grouped["qb_dropbacks"].ge(CFB_QB_MIN_GAME_DROPBACKS)].copy()
    grouped["qb_epa_per_dropback"] = grouped["total_epa"] / grouped["qb_dropbacks"]
    return grouped.loc[:, list(_QB_GAME_COLUMNS)].sort_values(
        ["season", "week", "game_id", "team_id"]
    )


def build_cfb_qb_states(qb_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per-player EWM of ``qb_epa_per_dropback``, gated on career dropbacks.

    The CFB analogue of ``quarterbacks.build_qb_states``, minus the offseason
    regression step (see module docstring: not one of the three flagged
    constants, but deliberately not inherited either -- a plain, unbroken
    EWM continues across the season boundary). A player's state at a given
    appearance is exposed (non-NaN) only once their cumulative
    ``qb_dropbacks`` reaches :data:`CFB_QB_MIN_DROPBACKS`.
    """

    require_columns(qb_games, _QB_GAME_COLUMNS, "cfb qb game metrics")
    require_columns(games, ("game_id", "gameday"), "cfb canonical games")

    dates = games.loc[:, ["game_id", "gameday"]].drop_duplicates("game_id")
    states = qb_games.merge(dates, on="game_id", how="left", validate="many_to_one")
    states = states.loc[states["gameday"].notna()].copy()
    states["gameday"] = pd.to_datetime(states["gameday"], errors="raise")
    states = states.sort_values(["passer_player_id", "gameday", "game_id"])

    alpha = 2.0 / (CFB_QB_STATE_SPAN + 1.0)
    output = pd.Series(np.nan, index=states.index, dtype="float64")
    for _, group in states.groupby("passer_player_id", sort=False):
        current = math.nan
        career_dropbacks = 0.0
        for index, row in group.iterrows():
            value = float(row["qb_epa_per_dropback"])
            if math.isfinite(value):
                current = (
                    value if not math.isfinite(current) else alpha * value + (1.0 - alpha) * current
                )
            career_dropbacks += float(row["qb_dropbacks"])
            if career_dropbacks >= CFB_QB_MIN_DROPBACKS:
                output.at[index] = current
    states["state_qb_epa_per_dropback"] = output
    states["career_dropbacks"] = states.groupby("passer_player_id", sort=False)[
        "qb_dropbacks"
    ].cumsum()
    return states.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. off_pass_rate team state
# ---------------------------------------------------------------------------


def build_cfb_pass_rate_team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (game, team) share of competitive offensive plays that are passes.

    Reuses the exact ``pass``/``rush`` flags ``cfb_features.py`` already uses
    for the explosive-play indicator (module docstring trap 4) -- no second
    play-type classification is derived.
    """

    plays = cfb_competitive_plays(pbp)
    plays = plays.loc[plays["competitive_play"]].copy()
    if plays.empty:
        return pd.DataFrame(columns=("game_id", "season", "week", "team_id", "off_pass_rate"))
    plays["team_id"] = pd.to_numeric(plays["pos_team_id"], errors="raise").astype("int64")
    grouped = (
        plays.groupby(["game_id", "season", "week", "team_id"], sort=False)
        .agg(off_pass_rate=("pass", "mean"))
        .reset_index()
    )
    return grouped.sort_values(["season", "week", "game_id", "team_id"]).reset_index(drop=True)


def build_cfb_pass_rate_states(team_games: pd.DataFrame) -> pd.DataFrame:
    """Strictly-lagged span-8 EWM of ``off_pass_rate`` per team, no offseason step.

    Mathematically identical to calling ``cfb_features.build_cfb_team_states``
    with ``offseason_retention=1.0`` -- see module docstring for why this is
    implemented standalone rather than by reusing that function.
    """

    require_columns(
        team_games, ("game_id", "team_id", "gameday", "off_pass_rate"), "pass rate team games"
    )
    states = team_games.copy().sort_values(["team_id", "gameday", "game_id"])
    states["off_pass_rate"] = pd.to_numeric(states["off_pass_rate"], errors="coerce")
    alpha = 2.0 / (CFB_PASS_RATE_SPAN + 1.0)
    output = pd.Series(np.nan, index=states.index, dtype="float64")
    for _, group in states.groupby("team_id", sort=False):
        current = math.nan
        observations = 0
        for index, row in group.iterrows():
            value = float(row["off_pass_rate"])
            if math.isfinite(value):
                current = (
                    value if not math.isfinite(current) else alpha * value + (1.0 - alpha) * current
                )
                observations += 1
            if observations >= CFB_PASS_RATE_MIN_PERIODS:
                output.at[index] = current
    states["state_off_pass_rate"] = output
    return states.loc[:, ["game_id", "team_id", "gameday", "state_off_pass_rate"]].reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# 3. Attach: identity rule + strictly-earlier state lookup + interaction
# ---------------------------------------------------------------------------


def attach_cfb_qb_dependence(
    games: pd.DataFrame,
    qb_games: pd.DataFrame,
    qb_states: pd.DataFrame,
    pass_rate_states: pd.DataFrame,
) -> pd.DataFrame:
    """Join the nine new columns onto the canonical CFB table.

    Single leak-safe chronological pass over ``games``, mirroring
    ``players.py``'s production mechanism exactly: after game *g* is
    processed, ``latest_passer[team]`` becomes that game's leading passer (by
    dropbacks, from the already-gated ``qb_games``); a future game's
    ``qb_starter_epa_per_dropback`` reads THAT player's own strictly-earlier
    EWM state (``build_cfb_qb_states``, not necessarily built from this
    team's games), exactly as ``players.py``'s ``latest_qb_appearance`` /
    ``_latest_qb_state`` do for NFL. ``off_pass_rate`` is looked up the same
    strictly-earlier way per team via :func:`build_cfb_pass_rate_states`.

    The interaction is built **per side** and then differenced (module
    docstring's "Decisions needing review #1" recommendation):
    ``{side}_qb_dependence_interaction = {side}_qb_starter_epa_per_dropback *
    {side}_off_pass_rate``, ``diff = home - away``.

    Every canonical game gets all nine :data:`CFB_QB_DEPENDENCE_COLUMNS`
    columns; a side without a computable value is left ``NaN`` (see module
    docstring -- the existing ridge pipeline's median imputer handles it).
    """

    require_columns(games, ("game_id", "gameday", "home_id", "away_id"), "cfb canonical games")

    # NOTE: ``game_id``'s own dtype is deliberately left untouched (a string
    # cast here, even reverted later, would still break REG bit-identity if
    # the sort order or comparison semantics shifted); ``str(...)`` is used
    # per-row below instead wherever a string key is needed.
    result = games.copy()
    result["gameday"] = pd.to_datetime(result["gameday"], errors="raise")
    result = result.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    qb_games_keyed = qb_games.copy()
    qb_games_keyed["game_id"] = _game_id_key(qb_games_keyed["game_id"])
    qb_games_keyed["passer_player_id"] = qb_games_keyed["passer_player_id"].astype(str)
    qb_appearances: dict[tuple[str, int], pd.DataFrame] = {
        (str(group["game_id"].iloc[0]), int(group["team_id"].iloc[0])): group.sort_values(
            "qb_dropbacks", ascending=False, kind="mergesort"
        ).reset_index(drop=True)
        for _, group in qb_games_keyed.groupby(["game_id", "team_id"], sort=False)
    }

    qb_states_keyed = qb_states.copy()
    qb_states_keyed["passer_player_id"] = qb_states_keyed["passer_player_id"].astype(str)
    qb_histories: dict[str, pd.DataFrame] = {
        str(player): group.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        for player, group in qb_states_keyed.groupby("passer_player_id", sort=False)
    }

    pass_rate_groups: dict[int, pd.DataFrame] = {
        int(group["team_id"].iloc[0]): group.sort_values(["gameday", "game_id"]).reset_index(
            drop=True
        )
        for _, group in pass_rate_states.groupby("team_id", sort=False)
    }

    for side in ("home", "away"):
        result[f"{side}_off_pass_rate"] = np.nan
        result[f"{side}_qb_starter_epa_per_dropback"] = np.nan

    latest_passer: dict[int, str] = {}

    for index, game in result.iterrows():
        game_id = str(int(game["game_id"]))
        game_date = np.datetime64(pd.Timestamp(game["gameday"]), "ns")
        for side in ("home", "away"):
            team_id = int(game[f"{side}_id"])

            pass_history = pass_rate_groups.get(team_id)
            if pass_history is not None and not pass_history.empty:
                dates = pass_history["gameday"].to_numpy(dtype="datetime64[ns]")
                position = int(np.searchsorted(dates, game_date, side="left")) - 1
                if position >= 0:
                    result.at[index, f"{side}_off_pass_rate"] = pass_history.iloc[position][
                        "state_off_pass_rate"
                    ]

            passer_id = latest_passer.get(team_id)
            if passer_id is not None:
                qb_history = qb_histories.get(passer_id)
                if qb_history is not None and not qb_history.empty:
                    dates = qb_history["gameday"].to_numpy(dtype="datetime64[ns]")
                    position = int(np.searchsorted(dates, game_date, side="left")) - 1
                    if position >= 0:
                        result.at[index, f"{side}_qb_starter_epa_per_dropback"] = qb_history.iloc[
                            position
                        ]["state_qb_epa_per_dropback"]

        for side in ("home", "away"):
            team_id = int(game[f"{side}_id"])
            appearances = qb_appearances.get((game_id, team_id))
            if appearances is not None and not appearances.empty:
                latest_passer[team_id] = str(appearances.iloc[0]["passer_player_id"])

    for side in ("home", "away"):
        result[f"{side}_off_pass_rate"] = pd.to_numeric(
            result[f"{side}_off_pass_rate"], errors="coerce"
        )
        result[f"{side}_qb_starter_epa_per_dropback"] = pd.to_numeric(
            result[f"{side}_qb_starter_epa_per_dropback"], errors="coerce"
        )
        result[f"{side}_qb_dependence_interaction"] = (
            result[f"{side}_qb_starter_epa_per_dropback"] * result[f"{side}_off_pass_rate"]
        )
    for metric in CFB_QB_DEPENDENCE_METRICS:
        result[f"diff_{metric}"] = result[f"home_{metric}"] - result[f"away_{metric}"]
    return result


def build_and_attach_cfb_qb_dependence(games: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One-call convenience wrapper: raw pbp + canonical games -> the joined table."""

    qb_games = build_cfb_qb_game_metrics(pbp)
    qb_states = build_cfb_qb_states(qb_games, games)
    pass_rate_team_games = build_cfb_pass_rate_team_games(pbp)
    dates = games.loc[:, ["game_id", "gameday"]].drop_duplicates("game_id").copy()
    dates["game_id"] = _game_id_key(dates["game_id"])
    pass_rate_team_games = pass_rate_team_games.copy()
    pass_rate_team_games["game_id"] = _game_id_key(pass_rate_team_games["game_id"])
    pass_rate_team_games = pass_rate_team_games.merge(
        dates, on="game_id", how="left", validate="many_to_one"
    )
    pass_rate_team_games = pass_rate_team_games.loc[pass_rate_team_games["gameday"].notna()].copy()
    pass_rate_states = build_cfb_pass_rate_states(pass_rate_team_games)
    return attach_cfb_qb_dependence(games, qb_games, qb_states, pass_rate_states)


# ---------------------------------------------------------------------------
# 4. Step 0 -- split-half reliability audit (BEFORE any accuracy number)
# ---------------------------------------------------------------------------


def _reshape_team_game_long(features: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (game, team): ``season``, ``week``, ``team_id``, ``metric``."""

    pieces: list[pd.DataFrame] = []
    for side in ("home", "away"):
        piece = features.loc[:, ["game_id", "season", "week", f"{side}_id"]].rename(
            columns={f"{side}_id": "team_id"}
        )
        piece[metric] = pd.to_numeric(features[f"{side}_{metric}"], errors="coerce")
        pieces.append(piece)
    long = pd.concat(pieces, ignore_index=True)
    long["team_id"] = pd.to_numeric(long["team_id"], errors="coerce").astype("Int64")
    long["season"] = pd.to_numeric(long["season"], errors="raise").astype(int)
    long["week"] = pd.to_numeric(long["week"], errors="raise").astype(int)
    return long


def split_half_reliability(
    long: pd.DataFrame, metric: str, *, seed: int, n_boot: int = 4000
) -> dict[str, Any]:
    """Odd/even-week team-season split-half reliability, per ``docs/injury_value_lost.md`` sec 3.1.

    Each team-season's ``metric`` values are split by odd/even week; the two
    halves' team-season MEANS are correlated (Pearson r, Spearman rho),
    Spearman-Brown corrected to a full-length reliability, and a block
    bootstrap over team-seasons gives a 95% CI and ``probability_positive``
    that the correlation is positive. Requires >=2 observations in each half
    for a team-season to be included (same floor the repo's own precedent
    scripts use, ``cfb_value_weighted_continuity_screen.py`` /
    ``cfb_role_continuity_remeasurement.py``).
    """

    subset = long.loc[long[metric].notna()].copy()
    subset["half"] = np.where(subset["week"] % 2 == 0, "even", "odd")
    means = subset.groupby(["team_id", "season", "half"])[metric].mean().unstack("half")
    counts = subset.groupby(["team_id", "season", "half"]).size().unstack("half")
    has_both = (counts.get("odd", pd.Series(dtype=float)).fillna(0) >= 2) & (
        counts.get("even", pd.Series(dtype=float)).fillna(0) >= 2
    )
    means = means.dropna()
    has_both = has_both.reindex(means.index).fillna(False)
    means = means.loc[has_both]

    n = len(means)
    if n < 3:
        return {
            "metric": metric,
            "n_team_seasons": n,
            "pearson_r": math.nan,
            "pearson_r_ci95": [math.nan, math.nan],
            "spearman_rho": math.nan,
            "spearman_brown_full_length_reliability": math.nan,
            "probability_positive": math.nan,
        }

    odd_vals = means["odd"].to_numpy(dtype=float)
    even_vals = means["even"].to_numpy(dtype=float)
    r = float(np.corrcoef(odd_vals, even_vals)[0, 1])
    rho = float(spearmanr(odd_vals, even_vals).correlation)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        selected = rng.integers(0, n, size=n)
        boots[draw] = np.corrcoef(odd_vals[selected], even_vals[selected])[0, 1]
    spearman_brown = (2.0 * r) / (1.0 + r) if r > -1.0 else math.nan
    return {
        "metric": metric,
        "n_team_seasons": n,
        "pearson_r": r,
        "pearson_r_ci95": [
            float(np.nanquantile(boots, 0.025)),
            float(np.nanquantile(boots, 0.975)),
        ],
        "spearman_rho": rho,
        "spearman_brown_full_length_reliability": spearman_brown,
        "probability_positive": float(np.mean(boots > 0.0)),
    }


@dataclass(frozen=True)
class ReliabilityAudit:
    """Step 0's split-half reliability audit -- the decisive gate before Step 2."""

    interaction: dict[str, Any]
    qb_starter_epa_per_dropback: dict[str, Any]
    off_pass_rate: dict[str, Any]
    no_split_half_examples: dict[str, float]
    cleared_examples: dict[str, float]


def cfb_qb_dependence_reliability(
    features: pd.DataFrame, *, seed_interaction: int = 1, seed_qb: int = 2, seed_pass_rate: int = 3
) -> ReliabilityAudit:
    """Run the split-half reliability audit on the interaction and its two constituents.

    ``features`` is the canonical CFB table already carrying
    :data:`CFB_QB_DEPENDENCE_COLUMNS` (i.e. the output of
    :func:`build_and_attach_cfb_qb_dependence`). Separate audits on the two
    constituent columns let a low interaction reliability be diagnosed as
    "one input is noisy" vs. "the product itself is unstable even though
    both inputs are fine" (products of two noisy quantities can be noisier
    than either factor).
    """

    required = {
        f"{side}_{metric}" for metric in CFB_QB_DEPENDENCE_METRICS for side in ("home", "away")
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(
            f"CFB qb-dependence features are missing columns: {', '.join(missing)}"
        )

    interaction_long = _reshape_team_game_long(features, "qb_dependence_interaction")
    qb_long = _reshape_team_game_long(features, "qb_starter_epa_per_dropback")
    pass_rate_long = _reshape_team_game_long(features, "off_pass_rate")

    return ReliabilityAudit(
        interaction=split_half_reliability(
            interaction_long, "qb_dependence_interaction", seed=seed_interaction
        ),
        qb_starter_epa_per_dropback=split_half_reliability(
            qb_long, "qb_starter_epa_per_dropback", seed=seed_qb
        ),
        off_pass_rate=split_half_reliability(pass_rate_long, "off_pass_rate", seed=seed_pass_rate),
        no_split_half_examples=dict(RELIABILITY_NO_SPLIT_HALF_EXAMPLES),
        cleared_examples=dict(RELIABILITY_CLEARED_EXAMPLES),
    )
