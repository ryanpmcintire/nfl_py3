"""CFB role-continuity feature family (the XLG-04 follow-up).

XLG-04 (``docs/cfb_role_replication.md``) established that role delivery is
league-general for dropbacks and carries: a player with a material trailing
role who participates at all delivers approximately that role, in both
leagues. This module turns that replicated mechanism into ONE pregame CFB
feature family and the machinery to score it against the frozen XLG-03
benchmark -- see ``docs/cfb_role_features.md`` for the predeclaration, which
is frozen before any run that touches ATS outcomes.

The roadmap's stated prerequisite is honored first: **separating permanent
departures from temporary absences**. CFB has no injury reports, so the only
pregame-knowable evidence about a role holder is their own participation
trail. Three consequences shape the frozen feature:

1. **Season scoping** (departure separation, cross-season): a player who has
   not yet appeared for the team in the current season is outside the role
   mass. Graduation, transfer, and the draft all present exactly this way,
   and :func:`absence_separation_study` measures how rarely a season's
   qualified role holders return the following season.
2. **Streak capping** (departure separation, within-season): a qualified
   holder who misses ``streak_cap`` consecutive valid team-games is treated
   as departed/out-for-season and leaves the role mass;
   :func:`absence_separation_study` measures how quickly same-season
   reappearance probability decays with streak length to justify the cap.
3. **Recency** (the signal): among the remaining active mass, the feature is
   the share-weighted fraction that participated in the team's most recent
   valid game -- the pregame-knowable trace of a role disruption.

Everything here is computed from credited actions only (dropback/carry --
the two replicated action types; receptions carry XLG-04's recorded
non-replication and are excluded). Absence of credit is never treated as
proof of unavailability; the feature only measures *observed participation
continuity*, a weaker and honest claim.

Module layout
-------------
1. :func:`absence_separation_study` -- descriptive, participation-only:
   absence episodes with hindsight reappearance labels, plus cross-season
   carryover of qualified role holders. Never reads spreads or outcomes.
2. :func:`build_role_continuity` / :func:`attach_role_continuity` -- the
   frozen pregame feature (leak-free single chronological pass).
3. :func:`cfb_role_benchmark` -- the three-arm walk-forward (market, frozen
   ``market_residual``, ``market_residual_roles``) on identical weeks, with
   paired week/season-blocked comparisons via
   ``nfl_ats.experiments.paired_feature_comparisons``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from nfl_ats.cfb_benchmark import (
    _PREDICTION_PASSTHROUGH,
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    _score_week,
    cfb_evaluation_window,
    fit_cfb_residual_model,
    summarize_cfb_benchmark,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.cfb_roles import (
    FROZEN_MIN_PRIOR_APPEARANCES,
    FROZEN_ROLE_SPAN,
    FROZEN_ROLE_THRESHOLDS,
    ROLE_ACTION_COLUMNS,
    TEAM_GAME_COLUMNS,
)
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.margin import fit_market_baseline

# ---------------------------------------------------------------------------
# Frozen configuration (see docs/cfb_role_features.md; fixed before any
# outcome-touching run)
# ---------------------------------------------------------------------------

# Only the two action types whose role delivery replicated cross-league in
# XLG-04. Receptions are deliberately excluded (recorded non-replication).
ROLE_FEATURE_ACTION_TYPES: tuple[str, ...] = ("dropback", "carry")

# A qualified role holder who misses this many consecutive valid team-games
# is treated as departed/out-for-season and leaves the active role mass.
# Justified by the absence-separation study recorded in the predeclaration.
FROZEN_STREAK_CAP: int = 4

# Neutral value when a team-game has no computable continuity (no valid
# prior game, empty active mass, or a team absent from the actions table).
CONTINUITY_NEUTRAL: float = 1.0

CFB_ROLE_FEATURE_COLUMNS: tuple[str, ...] = (
    "home_dropback_continuity",
    "away_dropback_continuity",
    "diff_dropback_continuity",
    "home_carry_continuity",
    "away_carry_continuity",
    "diff_carry_continuity",
)

ROLE_BENCHMARK_BASELINE_ARM = "cfb_benchmark_v1"
ROLE_BENCHMARK_CANDIDATE_ARM = "cfb_benchmark_v1_roles"

_CONTINUITY_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "team",
    "action_type",
    "continuity",
    "active_mass",
    "absent_mass",
    "qualified_players",
)

_EPISODE_COLUMNS: tuple[str, ...] = (
    "team",
    "player_id",
    "action_type",
    "start_season",
    "start_week",
    "prior_share",
    "length_valid_games",
    "reappeared_same_season",
    "reappeared_later_season",
    "never_reappeared",
)

_CARRYOVER_COLUMNS: tuple[str, ...] = (
    "team",
    "player_id",
    "action_type",
    "season",
    "end_state",
    "appeared_next_season",
    "next_season_observed",
)


@dataclass
class _PlayerTrail:
    state: float = math.nan
    appearances: int = 0
    missed_streak: int = 0
    last_season: int = -1


def _iter_team_action_games(
    actions: pd.DataFrame, team_games: pd.DataFrame
) -> list[tuple[str, str, list[dict[str, Any]], dict[str, dict[str, float]]]]:
    """Chronological valid team-games plus per-game appearance shares.

    Returns one entry per ``(team, action_type)``: the ordered list of that
    pair's valid team-game rows and a ``game_id -> {player_id: share}``
    lookup of credited appearances.
    """

    require_columns(actions, ROLE_ACTION_COLUMNS, "role actions")
    require_columns(team_games, TEAM_GAME_COLUMNS, "team games")

    acting = actions.loc[actions["action_type"].isin(ROLE_FEATURE_ACTION_TYPES)].copy()
    games = team_games.loc[team_games["action_type"].isin(ROLE_FEATURE_ACTION_TYPES)].copy()
    for frame in (acting, games):
        frame["game_id"] = frame["game_id"].astype(str)
        frame["team"] = frame["team"].astype(str)
        frame["action_type"] = frame["action_type"].astype(str)
    acting["player_id"] = acting["player_id"].astype(str)
    acting["share"] = pd.to_numeric(acting["count"], errors="coerce") / pd.to_numeric(
        acting["team_total"], errors="coerce"
    )

    shares: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for group_key, group in acting.groupby(["team", "action_type"], sort=False):
        team_key, action_key = str(group_key[0]), str(group_key[1])
        per_game: dict[str, dict[str, float]] = {}
        for game_value, player_value, share_value in zip(
            group["game_id"], group["player_id"], group["share"], strict=True
        ):
            per_game.setdefault(str(game_value), {})[str(player_value)] = float(share_value)
        shares[(team_key, action_key)] = per_game

    ordered = games.sort_values(["team", "action_type", "order_key", "game_id"])
    result: list[tuple[str, str, list[dict[str, Any]], dict[str, dict[str, float]]]] = []
    for group_key, group in ordered.groupby(["team", "action_type"], sort=False):
        team_key, action_key = str(group_key[0]), str(group_key[1])
        records = cast(list[dict[str, Any]], group.to_dict("records"))
        result.append(
            (
                team_key,
                action_key,
                records,
                shares.get((team_key, action_key), {}),
            )
        )
    return result


# ---------------------------------------------------------------------------
# 1. Departure vs temporary absence: the descriptive separation study
# ---------------------------------------------------------------------------


def absence_separation_study(
    actions: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    thresholds: dict[str, float] = FROZEN_ROLE_THRESHOLDS,
    min_prior: int = FROZEN_MIN_PRIOR_APPEARANCES,
    span: int = FROZEN_ROLE_SPAN,
) -> dict[str, pd.DataFrame]:
    """Label every qualified-holder absence episode with hindsight reappearance.

    Participation data only -- no spreads and no game outcomes are read, so
    this may legitimately run before the feature predeclaration is frozen
    (its results are design inputs, recorded in the predeclaration).

    An **episode** is a maximal run of consecutive valid team-games missed
    by a player who was a qualified role holder when the run began
    (finite state ``>= threshold`` with ``>= min_prior`` appearances).
    Hindsight labels: reappeared for the same team later in the same season,
    reappeared only in a later season, or never reappeared in the data.

    The **carryover** frame asks the cross-season question: for each season
    a player ended as a qualified holder, did they appear for the same team
    in the following season at all? ``next_season_observed`` marks whether
    the team even has valid games in that following season (a final-season
    row cannot count as evidence of departure).
    """

    alpha = 2.0 / (span + 1.0)
    episode_rows: list[dict[str, Any]] = []
    carryover_rows: list[dict[str, Any]] = []

    for team, action_type, games, per_game in _iter_team_action_games(actions, team_games):
        threshold = thresholds[action_type]
        seasons_with_games = sorted({int(game["season"]) for game in games})
        trails: dict[str, _PlayerTrail] = {}
        # Open episodes: player -> mutable episode record.
        open_episodes: dict[str, dict[str, Any]] = {}
        # Every appearance (team, player) -> list of seasons, for carryover.
        appearance_seasons: dict[str, set[int]] = {}
        # End-of-season qualified snapshots: season -> {player: state}.
        season_end_state: dict[int, dict[str, float]] = {}
        previous_season: int | None = None

        for game in games:
            game_season = int(game["season"])
            game_week = int(game["week"])
            if previous_season is not None and game_season != previous_season:
                season_end_state[previous_season] = {
                    player: trail.state
                    for player, trail in trails.items()
                    if math.isfinite(trail.state)
                    and trail.state >= threshold
                    and trail.appearances >= min_prior
                }
            previous_season = game_season

            appeared = per_game.get(str(game["game_id"]), {})
            for player, trail in trails.items():
                if player in appeared:
                    continue
                qualified = (
                    math.isfinite(trail.state)
                    and trail.state >= threshold
                    and trail.appearances >= min_prior
                )
                episode = open_episodes.get(player)
                if episode is not None:
                    episode["length_valid_games"] += 1
                elif qualified:
                    open_episodes[player] = {
                        "team": team,
                        "player_id": player,
                        "action_type": action_type,
                        "start_season": game_season,
                        "start_week": game_week,
                        "prior_share": trail.state,
                        "length_valid_games": 1,
                    }
                trail.missed_streak += 1

            for player, share in appeared.items():
                trail = trails.setdefault(player, _PlayerTrail())
                episode = open_episodes.pop(player, None)
                if episode is not None:
                    episode["reappeared_same_season"] = int(episode["start_season"]) == game_season
                    episode["reappeared_later_season"] = int(episode["start_season"]) != game_season
                    episode["never_reappeared"] = False
                    episode_rows.append(episode)
                trail.state = (
                    share
                    if not math.isfinite(trail.state)
                    else alpha * share + (1.0 - alpha) * trail.state
                )
                trail.appearances += 1
                trail.missed_streak = 0
                trail.last_season = game_season
                appearance_seasons.setdefault(player, set()).add(game_season)

        if previous_season is not None:
            season_end_state[previous_season] = {
                player: trail.state
                for player, trail in trails.items()
                if math.isfinite(trail.state)
                and trail.state >= threshold
                and trail.appearances >= min_prior
            }
        for episode in open_episodes.values():
            episode["reappeared_same_season"] = False
            episode["reappeared_later_season"] = False
            episode["never_reappeared"] = True
            episode_rows.append(episode)

        last_data_season = seasons_with_games[-1] if seasons_with_games else None
        for season, holders in season_end_state.items():
            if last_data_season is not None and season == last_data_season:
                next_observed = False
            else:
                next_observed = (season + 1) in seasons_with_games
            for player, state in holders.items():
                carryover_rows.append(
                    {
                        "team": team,
                        "player_id": player,
                        "action_type": action_type,
                        "season": season,
                        "end_state": state,
                        "appeared_next_season": (season + 1)
                        in appearance_seasons.get(player, set()),
                        "next_season_observed": next_observed,
                    }
                )

    episodes = pd.DataFrame(episode_rows, columns=_EPISODE_COLUMNS)
    carryover = pd.DataFrame(carryover_rows, columns=_CARRYOVER_COLUMNS)
    return {
        "episodes": episodes,
        "carryover": carryover,
        "episode_summary": summarize_absence_episodes(episodes),
        "carryover_summary": summarize_carryover(carryover),
    }


def summarize_absence_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    """Per action_type: same-season reappearance by episode length reached.

    For each length ``k`` (1..8), among episodes that reached at least ``k``
    missed valid games, what fraction eventually reappeared in the same
    season? This is the decay curve the frozen streak cap is read from.
    """

    require_columns(episodes, _EPISODE_COLUMNS, "absence episodes")
    rows: list[dict[str, Any]] = []
    for action_type, group in episodes.groupby("action_type", sort=True):
        total = len(group)
        for k in range(1, 9):
            reached = group.loc[group["length_valid_games"].ge(k)]
            n = len(reached)
            rows.append(
                {
                    "action_type": action_type,
                    "reached_length": k,
                    "episodes": n,
                    "fraction_of_all_episodes": (n / total) if total else math.nan,
                    "reappeared_same_season_rate": (
                        float(reached["reappeared_same_season"].mean()) if n else math.nan
                    ),
                    "never_reappeared_rate": (
                        float(reached["never_reappeared"].mean()) if n else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_carryover(carryover: pd.DataFrame) -> pd.DataFrame:
    """Per action_type: how often a qualified holder returns the next season."""

    require_columns(carryover, _CARRYOVER_COLUMNS, "carryover")
    rows: list[dict[str, Any]] = []
    for action_type, group in carryover.groupby("action_type", sort=True):
        observable = group.loc[group["next_season_observed"]]
        rows.append(
            {
                "action_type": action_type,
                "qualified_holder_seasons": len(group),
                "observable_transitions": len(observable),
                "returned_next_season_rate": (
                    float(observable["appeared_next_season"].mean())
                    if not observable.empty
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. The frozen pregame feature: role continuity
# ---------------------------------------------------------------------------


def build_role_continuity(
    actions: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    thresholds: dict[str, float] = FROZEN_ROLE_THRESHOLDS,
    min_prior: int = FROZEN_MIN_PRIOR_APPEARANCES,
    span: int = FROZEN_ROLE_SPAN,
    streak_cap: int = FROZEN_STREAK_CAP,
) -> pd.DataFrame:
    """Pregame role continuity per valid (team-game, action_type).

    For each valid team-game, evaluated strictly BEFORE that game's own
    credits update any state:

    - **Active role mass**: players with a finite EWM state at or above the
      action's qualification threshold, ``>= min_prior`` prior appearances,
      at least one appearance for this team in the game's own season, and a
      current missed-valid-game streak strictly below ``streak_cap``.
    - ``continuity`` = (state-weighted mass of active players whose streak
      is 0, i.e. who appeared in the team's most recent valid game) divided
      by the total active mass. Empty active mass -> ``CONTINUITY_NEUTRAL``.

    Season scoping and the streak cap are the departure/temporary-absence
    separation: a graduated, transferred, or drafted player never appears in
    the new season (excluded by scoping), and a within-season long-term loss
    ages out of the mass after ``streak_cap`` misses instead of depressing
    continuity all season.
    """

    alpha = 2.0 / (span + 1.0)
    rows: list[dict[str, Any]] = []
    for team, action_type, games, per_game in _iter_team_action_games(actions, team_games):
        threshold = thresholds[action_type]
        trails: dict[str, _PlayerTrail] = {}
        for game in games:
            game_season = int(game["season"])
            active_mass = 0.0
            appeared_mass = 0.0
            qualified_players = 0
            for trail in trails.values():
                if not (
                    math.isfinite(trail.state)
                    and trail.state >= threshold
                    and trail.appearances >= min_prior
                    and trail.last_season == game_season
                    and trail.missed_streak < streak_cap
                ):
                    continue
                qualified_players += 1
                active_mass += trail.state
                if trail.missed_streak == 0:
                    appeared_mass += trail.state
            continuity = (appeared_mass / active_mass) if active_mass > 0.0 else CONTINUITY_NEUTRAL
            rows.append(
                {
                    "game_id": str(game["game_id"]),
                    "season": game_season,
                    "week": int(game["week"]),
                    "team": team,
                    "action_type": action_type,
                    "continuity": continuity,
                    "active_mass": active_mass,
                    "absent_mass": active_mass - appeared_mass,
                    "qualified_players": qualified_players,
                }
            )

            appeared = per_game.get(str(game["game_id"]), {})
            for player, trail in trails.items():
                if player not in appeared:
                    trail.missed_streak += 1
            for player, share in appeared.items():
                trail = trails.setdefault(player, _PlayerTrail())
                trail.state = (
                    share
                    if not math.isfinite(trail.state)
                    else alpha * share + (1.0 - alpha) * trail.state
                )
                trail.appearances += 1
                trail.missed_streak = 0
                trail.last_season = game_season

    if not rows:
        return pd.DataFrame(columns=_CONTINUITY_COLUMNS)
    result = pd.DataFrame(rows, columns=_CONTINUITY_COLUMNS)
    if result.duplicated(["game_id", "team", "action_type"]).any():
        raise DataContractError("Role continuity produced duplicate team-game rows")
    return result.sort_values(["season", "week", "game_id", "team", "action_type"]).reset_index(
        drop=True
    )


def _normalized_id(values: pd.Series) -> pd.Series:
    """Team ids as canonical plain strings ('12', not '12.0' or '12 ').

    Missing ids become the literal '<NA>' string, which matches nothing and
    therefore falls through to the neutral imputation.
    """

    return pd.to_numeric(values, errors="coerce").astype("Int64").astype(str)


def attach_role_continuity(
    canonical_games: pd.DataFrame, continuity: pd.DataFrame, team_ids: pd.DataFrame
) -> pd.DataFrame:
    """Join home/away/diff continuity columns onto the canonical CFB table.

    The continuity frame keys teams by the play-by-play ``pos_team`` display
    name (e.g. "Minnesota Golden Gophers") while the canonical table names
    sides from the schedule source (e.g. "Minnesota"), so names must never
    be the join key (the first benchmark attempt failed exactly this way
    and was voided -- see docs/cfb_role_features.md). ``team_ids`` -- one
    row per ``(game_id, team, team_id)`` from the same play-by-play slice --
    maps continuity rows to ESPN team ids, which are then matched against
    the canonical ``home_id``/``away_id`` columns.

    Every canonical game gets all six :data:`CFB_ROLE_FEATURE_COLUMNS`;
    a side without a computable continuity row (invalid team-game, missing
    volume, week one) is imputed to :data:`CONTINUITY_NEUTRAL` so the
    feature is defined everywhere and encodes "no known disruption".
    """

    require_columns(canonical_games, ("game_id", "home_id", "away_id"), "cfb canonical games")
    require_columns(continuity, _CONTINUITY_COLUMNS, "role continuity")
    require_columns(team_ids, ("game_id", "team", "team_id"), "team id map")

    result = canonical_games.copy()
    result["game_id"] = result["game_id"].astype(str)

    mapping = team_ids.copy()
    mapping["game_id"] = mapping["game_id"].astype(str)
    mapping["team"] = mapping["team"].astype(str)
    mapping["team_id"] = pd.to_numeric(mapping["team_id"], errors="coerce").astype("Int64")
    mapping = mapping.dropna(subset=["team_id"]).drop_duplicates(["game_id", "team"])
    mapping["team_id"] = mapping["team_id"].astype(str)

    working = continuity.copy()
    working["game_id"] = working["game_id"].astype(str)
    working["team"] = working["team"].astype(str)
    working = working.merge(
        mapping[["game_id", "team", "team_id"]], on=["game_id", "team"], how="left"
    )
    unmapped = working["team_id"].isna()
    if unmapped.any():
        examples = ", ".join(
            working.loc[unmapped, "team"].astype(str).drop_duplicates().head(5).tolist()
        )
        raise DataContractError(f"Continuity teams missing a team id: {examples}")

    for action_type in ROLE_FEATURE_ACTION_TYPES:
        subset = working.loc[working["action_type"].eq(action_type)]
        lookup = subset.set_index(["game_id", "team_id"])["continuity"]
        for side in ("home", "away"):
            keys = pd.MultiIndex.from_arrays(
                [result["game_id"], _normalized_id(result[f"{side}_id"])]
            )
            values = pd.Series(lookup.reindex(keys).to_numpy(dtype=float), index=result.index)
            result[f"{side}_{action_type}_continuity"] = values.fillna(CONTINUITY_NEUTRAL)
        result[f"diff_{action_type}_continuity"] = (
            result[f"home_{action_type}_continuity"] - result[f"away_{action_type}_continuity"]
        )
    return result


# ---------------------------------------------------------------------------
# 3. The three-arm benchmark run (scores ATS outcomes -- predeclaration only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfbRoleBenchmarkResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    season_summary: pd.DataFrame
    paired: pd.DataFrame


def cfb_role_benchmark(
    features: pd.DataFrame,
    *,
    start_season: int = CFB_BENCHMARK_START_SEASON,
    end_season: int = CFB_BENCHMARK_END_SEASON,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260817,
) -> CfbRoleBenchmarkResult:
    """Walk-forward the frozen benchmark arms plus the role-continuity arm.

    ``features`` is the canonical CFB table already carrying the six
    :data:`CFB_ROLE_FEATURE_COLUMNS`. Weeks, training windows, and the
    frozen Ridge recipe are identical across arms; the ONLY difference in
    the candidate arm is the six extra columns. The paired frame reports
    per-game improvements of the candidate over the frozen
    ``market_residual`` arm with week- and season-blocked intervals on the
    clean core, via ``paired_feature_comparisons``.
    """

    required = {*_PREDICTION_PASSTHROUGH, *CFB_MODEL_FEATURE_COLUMNS, *CFB_ROLE_FEATURE_COLUMNS}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"CFB role features are missing columns: {', '.join(missing)}")
    # Fail closed on the exact defect that voided the first run: if every
    # role column is constant (e.g. an all-neutral failed join), the
    # candidate arm would silently reproduce the baseline bit-for-bit and
    # the "comparison" would be vacuous.
    role_values = features.loc[:, list(CFB_ROLE_FEATURE_COLUMNS)]
    if bool(role_values.nunique(dropna=False).le(1).all()):
        raise DataContractError(
            "Every role-continuity column is constant; the feature join produced no "
            "information, so a benchmark comparison would be vacuous"
        )

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError(f"No completed CFB games found from {start_season} to {end_season}")

    role_columns = (*CFB_MODEL_FEATURE_COLUMNS, *CFB_ROLE_FEATURE_COLUMNS)
    batches: list[pd.DataFrame] = []
    for (_, _), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        models = {
            "market": fit_market_baseline(training),
            "market_residual": fit_cfb_residual_model(training, ridge_alpha=ridge_alpha),
            "market_residual_roles": fit_cfb_residual_model(
                training, ridge_alpha=ridge_alpha, feature_columns=role_columns
            ),
        }
        batches.extend(_score_week(weekly_games, models))
    if not batches:
        raise ValueError("No CFB week had enough prior training games")

    predictions = pd.concat(batches, ignore_index=True)
    predictions["evaluation_window"] = predictions["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    predictions = predictions.sort_values(["gameday", "game_id", "method"]).reset_index(drop=True)
    summary, season_summary = summarize_cfb_benchmark(predictions)

    clean = predictions.loc[
        predictions["evaluation_window"].eq("clean_core")
        & predictions["method"].isin(("market_residual", "market_residual_roles"))
    ].copy()
    if clean.empty:
        raise ValueError("No clean-core predictions available for the paired comparison")
    clean["feature_set"] = clean["method"].map(
        {
            "market_residual": ROLE_BENCHMARK_BASELINE_ARM,
            "market_residual_roles": ROLE_BENCHMARK_CANDIDATE_ARM,
        }
    )
    paired = pd.concat(
        [
            paired_feature_comparisons(
                clean,
                baseline_feature_set=ROLE_BENCHMARK_BASELINE_ARM,
                samples=bootstrap_samples,
                block="week",
                seed=bootstrap_seed,
            ),
            paired_feature_comparisons(
                clean,
                baseline_feature_set=ROLE_BENCHMARK_BASELINE_ARM,
                samples=bootstrap_samples,
                block="season",
                seed=bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    return CfbRoleBenchmarkResult(
        predictions=predictions,
        summary=summary,
        season_summary=season_summary,
        paired=paired,
    )
