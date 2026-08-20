"""weak_stack_v3 gap-family features (docs/weak_stack_v3.md).

Every NFL registry signal with ``probability_positive >= 0.60`` in
``accuracy_points`` units that is (a) NOT already inside
``FEATURE_SETS["football_weak_stack"]`` and (b) buildable this session from
data already local to the repo. Three sub-families, all computed from the
newest ``data/raw/*/schedules.parquet`` snapshot (and, for penalty rate, the
newest PBP snapshot) alone -- never from ``result``/``spread_line`` at
prediction time, and never from a future game within the same season/team
lookup:

- ``gap_v3_bias`` (:data:`nfl_ats.constants.GAP_V3_BIAS_FEATURE_COLUMNS`):
  division revenge, sandwich spot, and the two post-blowout letdown/bounce
  flags, each as ``_home``/``_away``/``_diff``. Ported from already-reviewed
  constructs, not re-derived, so the registry's own measured
  ``probability_positive``/effect numbers describe exactly these columns:
  ``gap_division_revenge`` mirrors
  ``nfl_ats.experiment_runner.FLAG_BUILDERS["division_revenge_game"]``
  (registry ``bias_battery_division_revenge_game``, P+ 0.8825, and its
  opener re-screen ``bias_battery_division_revenge_game_opener``, P+
  0.8642); ``gap_sandwich_spot`` mirrors
  ``FLAG_BUILDERS["sandwich_spot"]`` (registry ``bias_battery_sandwich_spot``,
  P+ 0.603); the two post-blowout flags mirror
  ``scripts/nfl_bias_battery_screen.py``'s identically-named hypotheses
  (registry ``bias_battery_post_blowout_win_letdown``, P+ 0.7844,
  ``bias_battery_post_blowout_loss_bounce``, P+ 0.6344).
- ``gap_v3_penalty`` (:data:`nfl_ats.constants.GAP_V3_PENALTY_FEATURE_COLUMNS`):
  ``diff_penalty_rate_prior``, a season-lagged team penalty rate, ported
  verbatim from ``scripts/weak_stack_v2_eval.py``'s
  ``team_season_penalty_rate``/``add_penalty_discipline_feature`` (itself
  already verified there to reproduce the registered ``penalty_discipline``
  signal's mean/sd/reliability, P+ 0.6828, and scored once already as an
  opener-graded addition to ``weak_stack`` -- registry
  ``weak_stack_v2_penalty_only``, P+ 0.6939).
- ``gap_v3_travel`` (:data:`nfl_ats.constants.GAP_V3_TRAVEL_FEATURE_COLUMNS`):
  ``gap_thursday_pure_flag`` and ``gap_return_trip_hangover_flag``, ported
  from ``scripts/nfl_travel_rest_battery_screen.py``'s cells 8 and 4
  (registry ``travel_rest_thursday_pure``, P+ 0.7592,
  ``travel_rest_return_trip_hangover``, P+ 0.7528), using the same
  ``registry/stadium_coordinates.json`` reference table and haversine
  formula.

``surface_switch_flag`` (registry ``surface_switch_feature_arm``, P+ 0.6181)
is a fourth registry gap candidate, but it is NOT recomputed here: it is
already a real, tested production column
(``nfl_ats.features.add_surface_switch_features``), and
``data/processed/game_features_weak_stack_surface.parquet`` already carries
it. ``attach_weak_stack_v3_gap_features`` is meant to be called on THAT
table (not the plain ``weak_stack`` one), so weak_stack_v3 gets
surface_switch_flag for free by construction -- see
``FEATURE_SETS["football_weak_stack_v3"]`` in ``nfl_ats.constants``.

Every builder here reads only schedule-level, PBP-level, or static reference
facts -- never a column derived from this game's own outcome -- and every
season/team history lookup is strictly backward-looking (``shift(1)``/
``cumcount``/explicit ``prev_season = season + 1`` join keys), matching this
project's leak-safety convention. ``tests/test_weak_stack_v3_features.py``
carries a leakage regression test per family, per AGENTS.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.constants import (
    GAP_V3_BIAS_METRICS,
    GAP_V3_TRAVEL_FEATURE_COLUMNS,
    TEAM_ABBREVIATION_ALIASES,
)
from nfl_ats.features import add_ats_outcomes
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot

# Predeclared, round thresholds reused verbatim from the registry constructs
# these columns are named after -- never re-fit to this session's data.
BLOWOUT_MARGIN_POINTS = 17.0  # scripts/nfl_bias_battery_screen.py
LONG_DISTANCE_MI = 1500.0  # scripts/nfl_travel_rest_battery_screen.py
RETURN_TRIP_MAX_HOME_REST_DAYS = 8  # scripts/nfl_travel_rest_battery_screen.py
EARTH_RADIUS_MI = 3958.8

POSTSEASON_GAME_TYPES = ("WC", "DIV", "CON", "SB")


def _canonical(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _column_or_default(frame: pd.DataFrame, column: str, default: object) -> pd.Series:
    """``frame[column]`` if present, else a same-length constant column.

    Mirrors ``nfl_ats.features._numeric``'s graceful-default convention for
    schedule-shaped enrichments (never a hard data contract), and sidesteps
    ``DataFrame.get``'s ``Series | None`` return type, which mypy cannot
    thread through ``pd.to_numeric``/``pd.to_datetime``.
    """

    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    return frame[column]


def latest_schedules_snapshot(repo_root: Path) -> Path:
    """Newest ``data/raw/<snapshot>/schedules.parquet``, same convention
    ``nfl_ats.experiment_runner``/``scripts/nfl_bias_battery_screen.py``/
    ``scripts/nfl_travel_rest_battery_screen.py`` each already use."""

    candidates = sorted((repo_root / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no data/raw/*/schedules.parquet snapshot found under {repo_root}")
    return candidates[-1]


# ---------------------------------------------------------------------------
# gap_v3_bias: division revenge, sandwich spot, post-blowout letdown/bounce
# ---------------------------------------------------------------------------


def _team_long_table(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per (REG game, side): team, opponent, div_game, weekday, and
    this team's own raw score margin (signed from the team's perspective).

    REG-only, matching every registry construct these flags are named after
    (the NFL bias battery's own population). Never reads ``spread_line``.
    """

    df = schedules.copy()
    df = df.loc[df["game_type"].astype(str) == "REG"].copy()
    df = add_ats_outcomes(df)
    df["home_team"] = _canonical(df["home_team"])
    df["away_team"] = _canonical(df["away_team"])
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["gameday"] = pd.to_datetime(df["gameday"], errors="raise")
    df["div_game"] = (
        pd.to_numeric(_column_or_default(df, "div_game", 0), errors="coerce").fillna(0).astype(int)
    )
    df["weekday"] = _column_or_default(df, "weekday", "").astype(str)

    sides = []
    for is_home in (True, False):
        team_col, opp_col = ("home_team", "away_team") if is_home else ("away_team", "home_team")
        sign = 1.0 if is_home else -1.0
        sides.append(
            pd.DataFrame(
                {
                    "game_id": df["game_id"].astype(str),
                    "season": df["season"],
                    "week": df["week"],
                    "gameday": df["gameday"],
                    "team": df[team_col],
                    "opponent": df[opp_col],
                    "is_home": is_home,
                    "div_game": df["div_game"],
                    "weekday": df["weekday"],
                    "team_score_margin": sign * pd.to_numeric(df["result"], errors="coerce"),
                }
            )
        )
    long_df = pd.concat(sides, ignore_index=True)
    long_df = long_df.loc[long_df["team_score_margin"].notna()].copy()
    return long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)


def _add_gap_bias_flags(long_df: pd.DataFrame) -> pd.DataFrame:
    long_df = long_df.copy()

    # division_revenge: 2nd+ meeting THIS SEASON vs the SAME opponent, and
    # the team LOST the first meeting (team_score_margin < 0). Mirrors
    # nfl_ats.experiment_runner._flag_division_revenge_game exactly (same
    # (team, opponent, season) grouping, cumcount + transform("first")).
    ordered = long_df.sort_values(["team", "opponent", "season", "gameday"]).copy()
    grouped = ordered.groupby(["team", "opponent", "season"], sort=False)
    meeting_rank = grouped.cumcount()
    first_margin = grouped["team_score_margin"].transform("first")
    ordered["gap_division_revenge"] = (meeting_rank >= 1) & (first_margin < 0)
    long_df = long_df.merge(
        ordered[["game_id", "team", "gap_division_revenge"]], on=["game_id", "team"], how="left"
    )

    # sandwich_spot: non-division game flanked by a division game the week
    # before AND the week after, WITHIN the same team-season. This reads only
    # div_game, a structural full-season schedule fact fixed before Week 1
    # (like surface_switch_flag's modal-surface aggregate) -- never an
    # outcome column. Mirrors FLAG_BUILDERS["sandwich_spot"].
    grouped = long_df.groupby(["team", "season"], sort=False)
    prior_div = grouped["div_game"].shift(1)
    next_div = grouped["div_game"].shift(-1)
    long_df["gap_sandwich_spot"] = (long_df["div_game"] == 0) & (prior_div == 1) & (next_div == 1)

    # post_blowout_win_letdown / loss_bounce: team's IMMEDIATELY PRECEDING
    # game this season (strictly prior by gameday, via shift(1) on a
    # gameday-sorted group) was a >=17-raw-point win / loss. Mirrors
    # scripts/nfl_bias_battery_screen.py's identically-named hypotheses.
    grouped = long_df.groupby(["team", "season"], sort=False)
    prior_margin = grouped["team_score_margin"].shift(1)
    long_df["gap_post_blowout_win_letdown"] = prior_margin >= BLOWOUT_MARGIN_POINTS
    long_df["gap_post_blowout_loss_bounce"] = prior_margin <= -BLOWOUT_MARGIN_POINTS

    for metric in GAP_V3_BIAS_METRICS:
        long_df[metric] = long_df[metric].fillna(False).astype(bool)
    return long_df


def build_gap_bias_features(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per REG game_id with ``_home``/``_away``/``_diff`` columns for
    all four :data:`nfl_ats.constants.GAP_V3_BIAS_METRICS`. POST-season
    games are simply absent (callers must ``fillna(0.0)`` after merging onto
    a table that also carries postseason rows -- these mechanisms are
    undefined there, matching the project's missing-family-default
    convention)."""

    long_df = _add_gap_bias_flags(_team_long_table(schedules))
    wide_frames = []
    for is_home, side in ((True, "home"), (False, "away")):
        subset = long_df.loc[long_df["is_home"] == is_home, ["game_id", *GAP_V3_BIAS_METRICS]]
        subset = subset.rename(columns={m: f"{m}_{side}" for m in GAP_V3_BIAS_METRICS}).astype(
            {f"{m}_{side}": "float64" for m in GAP_V3_BIAS_METRICS}
        )
        wide_frames.append(subset.set_index("game_id"))
    wide = wide_frames[0].join(wide_frames[1], how="outer")
    for metric in GAP_V3_BIAS_METRICS:
        wide[f"{metric}_diff"] = wide[f"{metric}_home"] - wide[f"{metric}_away"]
    return wide.reset_index()


# ---------------------------------------------------------------------------
# gap_v3_penalty: diff_penalty_rate_prior
# ---------------------------------------------------------------------------


def team_season_penalty_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Identical construction to ``scripts/penalty_discipline_interval.py``/
    ``scripts/weak_stack_v2_eval.py.team_season_penalty_rate``: mean(penalty)
    over every raw regular-season play where ``posteam == team``. Reused
    verbatim (not re-derived) because it is the one definition already
    verified to reproduce the registry's recorded mean/sd/reliability."""

    plays = pbp.loc[pbp["posteam"].notna()].copy()
    plays["penalty"] = pd.to_numeric(plays["penalty"], errors="coerce").fillna(0.0)
    plays["team"] = _canonical(plays["posteam"])
    grouped = plays.groupby(["season", "team"]).agg(
        plays=("penalty", "size"), penalties=("penalty", "sum")
    )
    grouped["rate"] = grouped["penalties"] / grouped["plays"]
    return grouped.reset_index()


def build_gap_penalty_feature(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """``diff_penalty_rate_prior`` = home_prior_rate - away_prior_rate, a
    team-season's rate lagged to ``season + 1`` ONLY (no team-season can
    match its own season's plays or any later season's -- see
    ``tests/test_weak_stack_v3_features.py`` for the leak-safety assertion),
    ported verbatim from ``scripts/weak_stack_v2_eval.py.
    add_penalty_discipline_feature``. Teams with no locally-observed prior
    season get ``NaN`` (deliberately left for the ridge pipeline's
    ``SimpleImputer``, matching every other frozen feature's convention --
    not filled to 0.0 like the boolean gap_v3_bias/gap_v3_travel flags)."""

    rate = team_season_penalty_rate(pbp)
    lag = rate.copy()
    lag["prev_season"] = lag["season"] + 1
    lag = lag.rename(columns={"rate": "prior_rate"})[["team", "prev_season", "prior_rate"]]

    games = schedules[["game_id", "season", "home_team", "away_team"]].copy()
    games["game_id"] = games["game_id"].astype(str)
    games["season"] = pd.to_numeric(games["season"], errors="raise").astype(int)
    games["home_team"] = _canonical(games["home_team"])
    games["away_team"] = _canonical(games["away_team"])

    result = games[["game_id"]].copy()
    for side in ("home", "away"):
        joined = games[["game_id", "season", f"{side}_team"]].merge(
            lag,
            left_on=["season", f"{side}_team"],
            right_on=["prev_season", "team"],
            how="left",
        )
        result[f"{side}_penalty_rate_prior"] = joined["prior_rate"].to_numpy()
    result["diff_penalty_rate_prior"] = (
        result["home_penalty_rate_prior"] - result["away_penalty_rate_prior"]
    )
    return result[["game_id", "diff_penalty_rate_prior"]]


# ---------------------------------------------------------------------------
# gap_v3_travel: thursday_pure, return_trip_hangover
# ---------------------------------------------------------------------------


def load_stadium_coordinates(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


def haversine_mi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles. Ported verbatim from
    ``scripts/nfl_travel_rest_battery_screen.py.haversine_mi``."""

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_MI * math.asin(math.sqrt(a))


def build_gap_travel_rest_features(
    schedules: pd.DataFrame, coords: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """``gap_thursday_pure_flag`` and ``gap_return_trip_hangover_flag``, one
    row per REG game_id. Ported from
    ``scripts/nfl_travel_rest_battery_screen.py``'s cells 8
    (``travel_rest_thursday_pure``) and 4
    (``travel_rest_return_trip_hangover``): both are pregame-known schedule/
    geometry facts (``home_rest``/``weekday``/``stadium`` are schedule
    columns, not game-time actuals; stadium lat/lon is a static reference
    fact about a known, scheduled venue) -- no leakage caveat applies, per
    that script's own documented argument."""

    df = schedules.copy()
    df = df.loc[df["game_type"].astype(str) == "REG"].copy()
    df["game_id"] = df["game_id"].astype(str)
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["home_team"] = _canonical(df["home_team"])
    df["away_team"] = _canonical(df["away_team"])
    df["home_rest"] = pd.to_numeric(_column_or_default(df, "home_rest", np.nan), errors="coerce")
    df["gameday_dt"] = pd.to_datetime(_column_or_default(df, "gameday", pd.NaT), errors="coerce")
    df["weekday"] = _column_or_default(df, "weekday", "").astype(str)
    df["stadium"] = _column_or_default(df, "stadium", None)
    df["location"] = _column_or_default(df, "location", "Home")

    thursday_flag = (df["weekday"] == "Thursday").astype(float)

    home_rows = df.loc[df["location"] == "Home"]
    modal_stadium = home_rows.groupby(["home_team", "season"])["stadium"].agg(
        lambda s: s.mode().iat[0] if not s.mode(dropna=True).empty else None  # type: ignore[type-var]
    )

    def coord(name: object) -> dict[str, Any] | None:
        return coords.get(name) if isinstance(name, str) else None

    def team_home_coord(team: str, season: int) -> dict[str, Any] | None:
        name = modal_stadium.get((team, season))
        return coord(name)

    long_rows: list[dict[str, Any]] = []
    for _, g in df.iterrows():
        venue = coord(g["stadium"])
        for team in (g["home_team"], g["away_team"]):
            home_coord = team_home_coord(team, g["season"])
            if venue is None or home_coord is None:
                distance = np.nan
            else:
                distance = haversine_mi(
                    home_coord["lat"], home_coord["lon"], venue["lat"], venue["lon"]
                )
            long_rows.append(
                {
                    "game_id": g["game_id"],
                    "team": team,
                    "season": g["season"],
                    "gameday_dt": g["gameday_dt"],
                    "own_travel_mi": distance,
                }
            )
    long_df = pd.DataFrame(long_rows).sort_values(["team", "season", "gameday_dt"])
    long_df["prev_own_travel_mi"] = long_df.groupby(["team", "season"])["own_travel_mi"].shift(1)

    home_prev = df[["game_id", "home_team"]].merge(
        long_df[["game_id", "team", "prev_own_travel_mi"]],
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team"],
        how="left",
    )[["game_id", "prev_own_travel_mi"]]

    result = df[["game_id"]].copy()
    result["gap_thursday_pure_flag"] = thursday_flag.to_numpy()
    result = result.merge(home_prev, on="game_id", how="left")
    hangover = (result["prev_own_travel_mi"] >= LONG_DISTANCE_MI) & (
        df["home_rest"].to_numpy() <= RETURN_TRIP_MAX_HOME_REST_DAYS
    )
    result["gap_return_trip_hangover_flag"] = hangover.fillna(False).astype(float)
    return result[["game_id", "gap_thursday_pure_flag", "gap_return_trip_hangover_flag"]]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def attach_weak_stack_v3_gap_features(base: pd.DataFrame, *, repo_root: Path) -> pd.DataFrame:
    """Merge all three gap_v3 sub-families onto ``base`` by ``game_id``.

    ``base`` should already carry ``surface_switch_flag`` (i.e. be, or be
    derived from, ``data/processed/game_features_weak_stack_surface.parquet``
    -- see the module docstring); this function does not compute that column.
    Boolean flag families (``gap_v3_bias``, ``gap_v3_travel``) fill missing
    matches with ``0.0`` (postseason rows, or any game_id absent from the
    schedules snapshot); ``diff_penalty_rate_prior`` is deliberately left
    ``NaN`` where unresolved, for the pipeline's imputer.
    """

    schedules_path = latest_schedules_snapshot(repo_root)
    schedules = pd.read_parquet(schedules_path)

    bias = build_gap_bias_features(schedules)
    coords = load_stadium_coordinates(repo_root / "registry" / "stadium_coordinates.json")
    travel = build_gap_travel_rest_features(schedules, coords)

    snapshot = latest_pbp_snapshot(repo_root / "data" / "pbp" / "raw")
    pbp = load_pbp_snapshot(snapshot, include_postseason=False)
    penalty = build_gap_penalty_feature(pbp, schedules)

    result = base.copy()
    result["game_id"] = result["game_id"].astype(str)
    for frame in (bias, travel, penalty):
        frame = frame.copy()
        frame["game_id"] = frame["game_id"].astype(str)
        result = result.merge(frame, on="game_id", how="left")

    for metric in GAP_V3_BIAS_METRICS:
        for side in ("home", "away", "diff"):
            result[f"{metric}_{side}"] = result[f"{metric}_{side}"].fillna(0.0)
    for column in GAP_V3_TRAVEL_FEATURE_COLUMNS:
        result[column] = result[column].fillna(0.0)
    return result
