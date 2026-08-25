"""Per-team pregame state trends for the public team-explorer page.

The page consumes ONLY the canonical team-state feature schema produced by
:func:`nfl_ats.features.build_team_states` -- one row per team per completed
game carrying ``state_<metric>`` columns for every metric in
:data:`nfl_ats.constants.STATE_METRICS`. Each row's ``state_<metric>`` is the
team's exponentially-weighted pregame state (the strictly-earlier value the
model itself reads): offense/defense EPA per play, completion % over expected,
turnover and sack rates, point differential, and the ATS residual. No outcome,
market, or model-probability field is ever read here.

Local parquet data is optional. Every public-facing function degrades to a
clean empty state when the table is absent, and the unit tests drive the same
functions with a deterministic :func:`make_schema_fixture` so the contract
holds without any on-disk data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_ats.constants import STATE_METRICS

# The canonical team-state columns the public page is allowed to read. These are
# the pregame team states, the documented output of build_team_states.
TEAM_STATE_METRICS: tuple[str, ...] = STATE_METRICS
STATE_COLUMNS: tuple[str, ...] = tuple(f"state_{m}" for m in STATE_METRICS)

IDENTIFIER_COLUMNS = ("game_id", "season", "gameday", "team")

# Headline metrics shown in the at-a-glance overview and the matchup comparer.
# A short, opinionated subset keeps the page readable; every metric is still
# available in each team's per-season trend table.
DEFAULT_TREND_METRICS: tuple[str, ...] = (
    "off_epa_per_play",
    "def_epa_per_play",
    "point_diff",
    "ats_residual",
)

# Author copy for table headers / axis ticks -- not data, so inline (no i18n).
METRIC_LABELS: dict[str, str] = {
    "off_epa_per_play": "Offense EPA/play",
    "off_pass_epa_per_play": "Pass EPA/play",
    "off_rush_epa_per_play": "Rush EPA/play",
    "off_cpoe": "Completion % over expected",
    "off_yards_per_play": "Yards/play",
    "off_turnover_rate": "Turnover rate",
    "off_sack_rate": "Sack rate",
    "point_diff": "Point differential",
    "ats_residual": "ATS residual",
    "def_epa_per_play": "Defense EPA/play allowed",
    "def_pass_epa_per_play": "Pass EPA allowed",
    "def_rush_epa_per_play": "Rush EPA allowed",
    "def_yards_per_play": "Yards allowed/play",
    "def_takeaway_rate": "Takeaway rate",
    "def_sack_rate": "Sack rate (defense)",
}


#: Plain-language explanation for every metric, written for a reader who has
#: never seen this dashboard. First-grade rule: no undefined pronouns, no
#: unexplained jargon. Each sentence says what the number is, where it comes
#: from, and which direction is good.
METRIC_HELP: dict[str, str] = {
    "off_epa_per_play": (
        "Scoring value the offense creates per play. EPA ('expected points "
        "added') grades every play by how much it moved the team's chances of "
        "scoring. Around 0 is league average; higher is better."
    ),
    "off_pass_epa_per_play": ("Same scoring-value idea, passing plays only. Higher is better."),
    "off_rush_epa_per_play": ("Same scoring-value idea, running plays only. Higher is better."),
    "off_cpoe": (
        "Completion rate versus what an average quarterback would achieve on "
        "the same throws. Positive means more accurate than expected."
    ),
    "off_yards_per_play": "Total yards gained divided by plays run. Higher is better.",
    "off_turnover_rate": (
        "How often the offense gives the ball away (interceptions and lost "
        "fumbles). Lower is better."
    ),
    "off_sack_rate": ("How often the QB is sacked, per passing play. Lower is better."),
    "point_diff": (
        "Points scored minus points allowed, averaged per game. The most "
        "plain-spoken strength number there is."
    ),
    "ats_residual": (
        "Final margin versus the betting line: positive means the team beat "
        "expectations by more than the market predicted. This is history, not "
        "a prediction."
    ),
    "def_epa_per_play": (
        "Scoring value the defense GIVES UP per play to opposing offenses. Lower is better."
    ),
    "def_pass_epa_per_play": ("Value allowed on opponents' pass plays. Lower is better."),
    "def_rush_epa_per_play": ("Value allowed on opponents' run plays. Lower is better."),
    "def_yards_per_play": "Yards allowed per opponent play. Lower is better.",
    "def_takeaway_rate": (
        "How often the defense takes the ball away (interceptions and fumble "
        "recoveries). Higher is better."
    ),
    "def_sack_rate": (
        "How often the defense sacks the opposing QB, per passing play. Higher is better."
    ),
}


def metric_help(metric: str) -> str:
    return METRIC_HELP.get(metric, "")


def metric_label(metric: str) -> str:
    """Human-readable header for a canonical metric (fallback: the raw name)."""

    return METRIC_LABELS.get(metric, metric)


def coerce_state_table(state_table: pd.DataFrame | None) -> pd.DataFrame:
    """Validate a team-state table and return only the canonical columns.

    Raises :class:`ValueError` on a malformed frame so a bad artifact fails
    loud rather than silently rendering a misleading page.
    """

    if state_table is None or len(state_table) == 0:
        return pd.DataFrame(columns=[*IDENTIFIER_COLUMNS, *STATE_COLUMNS])

    missing_id = [c for c in IDENTIFIER_COLUMNS if c not in state_table.columns]
    if missing_id:
        raise ValueError(f"team-state table missing identifier columns: {missing_id}")
    missing_metric = [c for c in STATE_COLUMNS if c not in state_table.columns]
    if missing_metric:
        raise ValueError(
            "team-state table missing state columns: "
            f"{missing_metric[:3]}{'...' if len(missing_metric) > 3 else ''}"
        )

    frame = state_table[[*IDENTIFIER_COLUMNS, *STATE_COLUMNS]].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce")
    for column in STATE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["team"]).reset_index(drop=True)


@dataclass
class TeamTrends:
    """Aggregated per-team pregame-state trends for one data load.

    ``latest`` is the long-form snapshot of the most recent season with data
    (columns: ``team, metric, value, league_mean, z``). ``trend`` is the
    per-(team, metric, season) mean pregame state. ``long`` is the full
    pregame long form, useful for downstream tests and the rendered charts.
    """

    latest_season: int | None
    latest: pd.DataFrame
    trend: pd.DataFrame
    teams: list[str]
    long: pd.DataFrame

    @classmethod
    def empty(cls) -> TeamTrends:
        return cls(
            latest_season=None,
            latest=pd.DataFrame(columns=["team", "metric", "value", "league_mean", "z"]),
            trend=pd.DataFrame(columns=["team", "metric", "season", "value"]),
            teams=[],
            long=pd.DataFrame(columns=["team", "season", "metric", "value", "league_mean", "z"]),
        )


def aggregate_team_trends(
    state_table: pd.DataFrame | None,
    *,
    metrics: Sequence[str] | None = None,
) -> TeamTrends:
    """Aggregate a canonical team-state table into per-team, per-season trends.

    Each team's pregame state for a season is the mean of its per-game
    ``state_<metric>`` values that season (the exponentially-weighted team
    state the model reads at kickoff). ``z`` is the team's value relative to
    the league mean for that season and metric.
    """

    frame = coerce_state_table(state_table)
    if frame.empty:
        return TeamTrends.empty()

    wanted = list(metrics) if metrics is not None else list(STATE_METRICS)
    unknown = [m for m in wanted if m not in STATE_METRICS]
    if unknown:
        raise ValueError(f"unknown metric requested: {unknown}")

    long = frame.melt(
        id_vars=["team", "season", "gameday", "game_id"],
        value_vars=[f"state_{m}" for m in wanted],
        var_name="state_column",
        value_name="value",
    )
    long["metric"] = long["state_column"].str.removeprefix("state_")
    long = long.drop(columns="state_column").dropna(subset=["value"])
    if long.empty:
        return TeamTrends.empty()

    long["league_mean"] = long.groupby(["season", "metric"])["value"].transform("mean")
    long["z"] = long["value"] - long["league_mean"]

    latest_season = int(long["season"].max())
    latest_season_long = long.loc[long["season"] == latest_season]
    # One row per (team, metric): each team's season-average pregame state and
    # its z (team average minus the league mean for that season/metric).
    latest = latest_season_long.groupby(["team", "metric"])["value"].mean().reset_index()
    league_mean_by_metric = latest_season_long.groupby("metric")["value"].mean().to_dict()
    latest["league_mean"] = latest["metric"].map(league_mean_by_metric)
    latest["z"] = latest["value"] - latest["league_mean"]
    trend = (
        long.groupby(["team", "metric", "season"])["value"]
        .mean()
        .reset_index()
        .sort_values(["team", "metric", "season"])
    )
    teams = sorted(long["team"].dropna().unique().tolist())
    return TeamTrends(
        latest_season=latest_season,
        latest=latest.reset_index(drop=True),
        trend=trend,
        teams=teams,
        long=long.reset_index(drop=True),
    )


def feature_table_to_team_states(feature_table: pd.DataFrame | None) -> pd.DataFrame | None:
    """Convert a forecast's canonical per-game feature table into the team-state
    long form this module consumes.

    The feature table (output of :func:`nfl_ats.features.attach_team_states`)
    carries ``home_<metric>`` / ``away_<metric>`` pregame state columns for
    every canonical metric, plus ``home_team`` / ``away_team`` identifiers. We
    melt each game into two team rows. Returns ``None`` when the table does not
    carry the canonical state columns, so callers can fall back to an empty
    state without guessing.
    """

    if feature_table is None or len(feature_table) == 0:
        return None
    required = ("home_team", "away_team", f"home_{STATE_METRICS[0]}", f"away_{STATE_METRICS[0]}")
    if not all(column in feature_table.columns for column in required):
        return None

    side_frames: list[pd.DataFrame] = []
    keep_id = [c for c in ("game_id", "season", "gameday") if c in feature_table.columns]
    for side in ("home", "away"):
        identifiers = feature_table[[*keep_id, f"{side}_team"]].copy()
        identifiers = identifiers.rename(columns={f"{side}_team": "team"})
        states = feature_table[[f"{side}_{m}" for m in STATE_METRICS]].rename(
            columns={f"{side}_{m}": f"state_{m}" for m in STATE_METRICS}
        )
        side_frames.append(pd.concat([identifiers, states], axis=1))
    combined = pd.concat(side_frames, ignore_index=True)
    combined["season"] = pd.to_numeric(combined["season"], errors="coerce").astype("Int64")
    combined["gameday"] = pd.to_datetime(combined["gameday"], errors="coerce")
    return combined


def make_schema_fixture(
    *,
    teams: Sequence[str] = ("ARI", "BUF", "KC", "SF"),
    seasons: Sequence[int] = (2023, 2024, 2025),
    games_per_season: int = 9,
    metrics: Sequence[str] | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """A deterministic, synthetic canonical team-state table.

    Values are fabricated (clearly not real team quality) and exist ONLY to
    exercise the schema and rendering -- they never assert a football fact.
    The output honours the exact column contract
    (:data:`IDENTIFIER_COLUMNS` + :data:`STATE_COLUMNS`) and is stable for a
    given ``seed`` so tests can assert on it.
    """

    wanted = list(metrics) if metrics is not None else list(STATE_METRICS)
    state_columns = [f"state_{m}" for m in wanted]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for team in teams:
        # One stable base per (team, metric) so each team shows a coherent,
        # distinct trend; a small per-game wobble keeps it realistic.
        bases = {column: float(rng.normal(0.0, 0.3)) for column in state_columns}
        for season in seasons:
            for game in range(1, games_per_season + 1):
                row: dict[str, object] = {
                    "game_id": f"{season}_{game:02d}_{team}",
                    "season": int(season),
                    "gameday": pd.Timestamp(f"{season}-01-01") + pd.Timedelta(days=7 * (game - 1)),
                    "team": str(team),
                }
                for column in state_columns:
                    row[column] = bases[column] + 0.04 * (game - 5) + float(rng.normal(0.0, 0.02))
                rows.append(row)
    return pd.DataFrame(rows)


def team_state_payload(trends: TeamTrends) -> Mapping[str, Mapping[str, float]]:
    """Compact ``{team: {metric: latest_z}}`` map for the matchup comparer's
    embedded JSON. ``z`` (team minus league mean) is what the comparer shows,
    so the payload carries only that, never raw outcome or market data."""

    payload: dict[str, dict[str, float]] = {}
    if trends.latest.empty:
        return payload
    latest = trends.latest
    for _, row in latest.iterrows():
        team = str(row["team"])
        metric = str(row["metric"])
        payload.setdefault(team, {})[metric] = float(row["z"])
    return payload
