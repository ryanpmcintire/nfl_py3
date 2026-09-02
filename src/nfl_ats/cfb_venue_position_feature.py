"""Venue-milestone / schedule-position candidate columns for COLLEGE FOOTBALL.

Cross-league replication of four frozen NFL constructs
(``docs/venue_milestone_screen.md`` cells 1-2 and the NFL bias battery's
``three_plus_road_games`` / ``division_revenge_game`` cells). Predeclared in
``docs/cfb_venue_position_replication.md`` -- read that first: it freezes the
population, the comparator, the four cells, the null, the positive control,
the era split and the recording rules before any outcome sign is computed.

**Every column here is a pure schedule fact.** Three of the four are functions
of the published schedule alone (game order, home/away side, neutral-site flag,
venue identity). The fourth, ``cfb_schedule_revenge_prior_meeting_loss``, reads
the final score of a STRICTLY EARLIER game between the same two teams -- which
is pregame-legal by the same argument the NFL cell uses, and is enforced here
by construction. ``tests/test_cfb_venue_position_feature.py`` regression-tests
both properties.

**Why the full schedules snapshot, not the benchmark table.** The XLG-03
benchmark table (``data/processed/cfb_game_features.parquet``) is a FILTERED
subset -- completed regular-season FBS-vs-FBS games with an orientable spread.
A team's first home game of the season is very often against an FCS opponent
and is therefore absent from that subset, so "first home game" or "3rd
consecutive road game" computed inside the subset would be systematically
wrong. Schedule position is computed on ``data/cfb/schedules/raw/<snapshot>/``,
which carries every benchmark team's FULL season sequence (**measured**
2026-09-01: 2,473 benchmark team-seasons, 0 with no schedule rows, median 12-13
games per team-season) and reaches back to 2001, five seasons before the
benchmark's own 2006 left edge.

**team_info is a diagnostic here, not the operative venue map.** The
cfbfastR-data ``team_info`` snapshot joins on ``(season, team_id)`` with
coverage 1.000 on both sides in every season 2006-2025 (**measured**), but its
``venue_id`` is IDENTICAL across all 20 season partitions for all 706 teams
(**measured**: 0 teams carry more than one distinct value), i.e. it is one
current-state snapshot replicated per season, not a per-season venue history.
Its disagreement with the schedules snapshot's own per-game ``venue_id`` on
non-neutral home games falls monotonically from 9.30% (2006) to 0.32% (2025) --
the exact signature of a current-state map applied to historical games. The
operative venue identity is therefore the schedules snapshot's own per-game
``venue_id``, which is what the predeclaration requires ("relative to the
schedules snapshot's own history"); ``team_info`` is joined anyway and reported
as an agreement diagnostic so the discrepancy stays visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Cell 1 -- NFL sibling ``venue_milestone_home_opener``.
CFB_HOME_OPENER_COLUMN = "cfb_venue_home_opener"
#: Cell 2 -- NFL sibling ``venue_milestone_new_stadium_debut``.
CFB_NEW_VENUE_DEBUT_COLUMN = "cfb_venue_new_venue_debut"
#: Cell 3 -- NFL sibling ``bias_battery_three_plus_road_games``.
CFB_THREE_PLUS_ROAD_COLUMN = "cfb_schedule_three_plus_road"
#: Cell 4 -- ADAPTED from NFL ``bias_battery_division_revenge_game`` (that cell
#: is a WITHIN-season division rematch; CFB has almost no within-season rematch
#: population, so this one looks back across seasons -- see the predeclaration).
CFB_REVENGE_PRIOR_MEETING_COLUMN = "cfb_schedule_revenge_prior_meeting_loss"

CFB_VENUE_POSITION_FEATURE_COLUMNS: tuple[str, ...] = (
    CFB_HOME_OPENER_COLUMN,
    CFB_NEW_VENUE_DEBUT_COLUMN,
    CFB_THREE_PLUS_ROAD_COLUMN,
    CFB_REVENGE_PRIOR_MEETING_COLUMN,
)

#: Predeclared lookback for cell 4: the most recent prior meeting must sit in
#: the current season or one of the two immediately preceding seasons.
REVENGE_LOOKBACK_SEASONS = 2

#: Columns the module needs from the schedules snapshot.
SCHEDULE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "season_type",
    "start_date",
    "neutral_site",
    "venue_id",
    "home_id",
    "away_id",
    "home_points",
    "away_points",
)

#: Columns the module needs from the cfbfastR-data ``team_info`` snapshot.
TEAM_INFO_COLUMNS: tuple[str, ...] = ("team_id", "school", "venue_id", "venue_name")

_REQUIRED_FEATURE_COLUMNS = frozenset({"game_id", "season", "home_id", "away_id"})


# ---------------------------------------------------------------------------
# Input resolution (lazy, so importing this module never requires local data)
# ---------------------------------------------------------------------------


def _latest(glob_pattern: str, label: str) -> Path:
    candidates = sorted(REPO_ROOT.glob(glob_pattern))
    if not candidates:
        raise FileNotFoundError(f"no {label} found matching {glob_pattern!r}")
    return candidates[-1]


def default_schedules_dir() -> Path:
    """Latest CFB schedules snapshot directory (the one holding ``season=*``)."""

    return _latest(
        "data/cfb/schedules/raw/*/season=*/schedules.parquet", "cfb schedules"
    ).parent.parent


def default_team_info_dir() -> Path:
    """Latest cfbfastR-data ``team_info`` snapshot directory."""

    return _latest(
        "data/cfb/team_info/raw/*/season=*/team_info.parquet", "cfb team_info"
    ).parent.parent


def load_schedules(schedules_dir: Path | None = None) -> pd.DataFrame:
    """Every game in the schedules snapshot, all seasons, all season types.

    All season types are kept on purpose: a conference-championship or bowl
    meeting is a genuine prior meeting for cell 4's cross-season lookback, and
    it can never disturb a within-season position (cells 1 and 3) because it
    always kicks off after every regular-season game of the same season.
    """

    directory = schedules_dir or default_schedules_dir()
    paths = sorted(Path(directory).glob("season=*/schedules.parquet"))
    if not paths:
        raise FileNotFoundError(f"no season=*/schedules.parquet files under {directory}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    return normalize_schedules(frame)


def normalize_schedules(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw schedules frame to the dtypes the derivation assumes."""

    missing = sorted(set(SCHEDULE_COLUMNS).difference(frame.columns))
    if missing:
        raise DataContractError(f"CFB schedules are missing columns: {', '.join(missing)}")
    table = frame.loc[:, list(SCHEDULE_COLUMNS)].copy()
    table["game_id"] = pd.to_numeric(table["game_id"], errors="coerce").astype("Int64")
    table["season"] = pd.to_numeric(table["season"], errors="coerce").astype("Int64")
    table["kickoff"] = pd.to_datetime(table["start_date"], utc=True, errors="coerce")
    table["neutral_site"] = (
        pd.to_numeric(table["neutral_site"].astype(float), errors="coerce").fillna(0.0).astype(bool)
    )
    table["venue_id"] = pd.to_numeric(table["venue_id"], errors="coerce").astype("Int64")
    for side in ("home_id", "away_id"):
        table[side] = pd.to_numeric(table[side], errors="coerce").astype("Int64")
    for side in ("home_points", "away_points"):
        table[side] = pd.to_numeric(table[side], errors="coerce")
    if table["kickoff"].isna().any():
        raise DataContractError("CFB schedules carry rows with an unparseable start_date")
    if table["game_id"].duplicated().any():
        raise DataContractError("CFB schedules carry duplicate game_id rows")
    return table.drop(columns=["start_date"])


def load_team_own_venues(team_info_dir: Path | None = None) -> pd.DataFrame:
    """``(season, team_id) -> own venue_id`` from the cfbfastR-data snapshot.

    Kept as a DIAGNOSTIC only. **Measured** 2026-09-01: this map's ``venue_id``
    is constant across all 20 season partitions for all 706 teams, so it is a
    current-state snapshot, not a per-season history; the operative venue
    identity in this module is the schedules snapshot's own per-game
    ``venue_id``.
    """

    directory = team_info_dir or default_team_info_dir()
    paths = sorted(Path(directory).glob("season=*/team_info.parquet"))
    if not paths:
        raise FileNotFoundError(f"no season=*/team_info.parquet files under {directory}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        season = int(path.parent.name.split("=", 1)[1])
        part = pd.read_parquet(path, columns=list(TEAM_INFO_COLUMNS))
        part["season"] = season
        frames.append(part)
    table = pd.concat(frames, ignore_index=True)
    table["team_id"] = pd.to_numeric(table["team_id"], errors="coerce").astype("Int64")
    table["own_venue_id"] = pd.to_numeric(table["venue_id"], errors="coerce").astype("Int64")
    table = table.loc[table["team_id"].notna()].drop_duplicates(subset=["season", "team_id"])
    return table.loc[:, ["season", "team_id", "own_venue_id", "venue_name", "school"]]


# ---------------------------------------------------------------------------
# Team-side sequence
# ---------------------------------------------------------------------------


def build_team_side_sequence(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, kickoff-ordered -- the sequence every cell reads.

    ``is_true_home`` / ``is_true_road`` transcribe the NFL bias battery's own
    definition verbatim (**read**, ``scripts/nfl_bias_battery_screen.py:236``:
    ``is_true_road = (~is_home) & (neutral_site == 0)``), so a neutral-site game
    is neither a home game nor a road game -- it occupies a slot in the
    sequence and therefore BREAKS a road streak.
    """

    common = {
        "game_id": schedules["game_id"],
        "season": schedules["season"],
        "week": schedules["week"],
        "season_type": schedules["season_type"],
        "kickoff": schedules["kickoff"],
        "neutral_site": schedules["neutral_site"],
        "venue_id": schedules["venue_id"],
    }
    home = pd.DataFrame(
        {
            **common,
            "team_id": schedules["home_id"],
            "opponent_id": schedules["away_id"],
            "is_home_side": True,
            "team_points": schedules["home_points"],
            "opponent_points": schedules["away_points"],
        }
    )
    away = pd.DataFrame(
        {
            **common,
            "team_id": schedules["away_id"],
            "opponent_id": schedules["home_id"],
            "is_home_side": False,
            "team_points": schedules["away_points"],
            "opponent_points": schedules["home_points"],
        }
    )
    table = pd.concat([home, away], ignore_index=True)
    table = table.loc[table["team_id"].notna()].copy()
    table["is_true_home"] = table["is_home_side"] & ~table["neutral_site"]
    table["is_true_road"] = ~table["is_home_side"] & ~table["neutral_site"]
    return table.sort_values(["team_id", "kickoff", "game_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cell 1 -- home opener
# ---------------------------------------------------------------------------


def flag_home_opener(sequence: pd.DataFrame) -> pd.Series:
    """The team's FIRST true home game of its season, in kickoff order.

    A neutral-site game is not a true home game, so it neither flags nor
    consumes the slot: a team that opens at a neutral site still flags on its
    first genuine home date later in the season.
    """

    home = sequence["is_true_home"]
    order = sequence.loc[home].groupby(["team_id", "season"], sort=False).cumcount()
    flag = pd.Series(False, index=sequence.index)
    flag.loc[order.index] = order.eq(0).to_numpy()
    return flag & home


# ---------------------------------------------------------------------------
# Cell 2 -- new venue debut
# ---------------------------------------------------------------------------


def declared_home_venues(sequence: pd.DataFrame) -> pd.DataFrame:
    """``(team_id, season) -> declared_home_venue_id``.

    The declared home venue is the venue hosting the PLURALITY of that team's
    true home games that season, ties broken by earliest kickoff. This is a
    schedule fact known at schedule release, and it is what separates a
    permanent home-venue change from a one-off relocated or designated-home
    game -- the distinction the NFL cell makes explicitly (**read**,
    ``docs/venue_milestone_screen.md:54-68``: brand-new stadium or relocation
    destination only; one-off displacements and internationals excluded).
    """

    hosted = sequence.loc[sequence["is_true_home"] & sequence["venue_id"].notna()]
    counts = (
        hosted.groupby(["team_id", "season", "venue_id"], sort=False)
        .agg(games=("kickoff", "size"), first_kickoff=("kickoff", "min"))
        .reset_index()
        .sort_values(
            ["team_id", "season", "games", "first_kickoff"], ascending=[True, True, False, True]
        )
    )
    declared = counts.drop_duplicates(subset=["team_id", "season"])
    return declared.loc[:, ["team_id", "season", "venue_id"]].rename(
        columns={"venue_id": "declared_home_venue_id"}
    )


def flag_new_venue_debut(sequence: pd.DataFrame) -> pd.Series:
    """The team's first true home game in a venue that is NEW to it.

    Three conditions, all predeclared:

    1. the venue is the team's DECLARED home venue for that season (plurality
       of true home games), which excludes one-off relocations;
    2. the team never hosted a true home game at that venue in any STRICTLY
       EARLIER season of the snapshot;
    3. **the snapshot left-edge rule** -- the team hosted at least one true
       home game in a strictly earlier snapshot season. A team's first season
       in the snapshot is never a venue debut, because the snapshot cannot
       distinguish "new venue" from "no history".

    Exactly one game per qualifying team-season flags: the earliest true home
    game at that venue.
    """

    hosted = sequence.loc[sequence["is_true_home"] & sequence["venue_id"].notna()].copy()
    if hosted.empty:
        return pd.Series(False, index=sequence.index)

    declared = declared_home_venues(sequence)
    hosted = hosted.merge(declared, on=["team_id", "season"], how="left")

    first_host_season = (
        hosted.groupby(["team_id", "venue_id"], sort=False)["season"]
        .min()
        .rename("first_host_season")
        .reset_index()
    )
    first_home_season = (
        hosted.groupby("team_id", sort=False)["season"]
        .min()
        .rename("team_first_home_season")
        .reset_index()
    )
    hosted = hosted.merge(
        first_host_season.rename(columns={"venue_id": "declared_home_venue_id"}),
        on=["team_id", "declared_home_venue_id"],
        how="left",
    ).merge(first_home_season, on="team_id", how="left")

    at_declared = hosted["venue_id"].eq(hosted["declared_home_venue_id"])
    venue_is_new = hosted["season"].eq(hosted["first_host_season"])
    past_left_edge = hosted["season"].gt(hosted["team_first_home_season"])
    qualifying = at_declared & venue_is_new & past_left_edge

    earliest = (
        hosted.loc[qualifying]
        .sort_values(["team_id", "season", "kickoff", "game_id"])
        .drop_duplicates(subset=["team_id", "season"])["game_id"]
    )
    debut_ids = set(earliest.dropna().tolist())
    flag = sequence["is_true_home"] & sequence["game_id"].isin(debut_ids)
    return flag.astype(bool)


# ---------------------------------------------------------------------------
# Cell 3 -- third-or-later consecutive true road game
# ---------------------------------------------------------------------------


def flag_three_plus_road(sequence: pd.DataFrame) -> pd.Series:
    """This game and the two immediately preceding games are all true road games.

    Transcribed verbatim from the NFL cell (**read**,
    ``scripts/nfl_bias_battery_screen.py:235-240``): grouped by
    ``(team, season)``, ``is_true_road & shift(1) & shift(2)`` with missing
    shifts filled ``False``. A bye does not break the streak (it is not a row);
    a home game breaks it and so does a NEUTRAL-SITE game, because a
    neutral-site game occupies a sequence slot and is not a true road game.
    """

    grouped = sequence.groupby(["team_id", "season"], sort=False)["is_true_road"]
    previous_one = grouped.shift(1).fillna(False).astype(bool)
    previous_two = grouped.shift(2).fillna(False).astype(bool)
    return sequence["is_true_road"] & previous_one & previous_two


# ---------------------------------------------------------------------------
# Cell 4 -- lost the most recent prior meeting (ADAPTED across seasons)
# ---------------------------------------------------------------------------


def attach_prior_meeting(
    sequence: pd.DataFrame, *, lookback_seasons: int = REVENGE_LOOKBACK_SEASONS
) -> pd.DataFrame:
    """Add the most recent STRICTLY EARLIER meeting with the same opponent.

    The lookup is a ``shift(1)`` over the kickoff-ordered per-``(team,
    opponent)`` sequence, and the strict inequality ``prior_kickoff < kickoff``
    is then imposed as an explicit condition rather than left to the ordering,
    so a game at or after the current kickoff is unreachable twice over. The
    belt-and-braces matters: the snapshot carries exactly TWO team-side rows
    (**measured** 2026-09-01: ``game_id`` 400361387, season 2008, team ids 99
    and 2026) whose immediately preceding same-opponent row shares an identical
    kickoff timestamp -- a duplicated matchup record. A simultaneous game is
    not a prior game, and the strict comparison drops it.
    """

    ordered = sequence.sort_values(["team_id", "opponent_id", "kickoff", "game_id"]).copy()
    grouped = ordered.groupby(["team_id", "opponent_id"], sort=False)
    ordered["prior_kickoff"] = grouped["kickoff"].shift(1)
    ordered["prior_season"] = grouped["season"].shift(1)
    ordered["prior_team_points"] = grouped["team_points"].shift(1)
    ordered["prior_opponent_points"] = grouped["opponent_points"].shift(1)
    seasons_back = ordered["season"].astype("Float64") - ordered["prior_season"].astype("Float64")
    ordered["prior_within_lookback"] = (
        ordered["prior_kickoff"].notna()
        & ordered["prior_kickoff"].lt(ordered["kickoff"])
        & seasons_back.notna()
        & seasons_back.le(float(lookback_seasons)).fillna(False)
        & seasons_back.ge(0.0).fillna(False)
    )
    ordered["prior_same_season"] = ordered["prior_within_lookback"] & seasons_back.eq(0.0).fillna(
        False
    )
    ordered["lost_prior_meeting"] = (
        ordered["prior_within_lookback"]
        & ordered["prior_team_points"].notna()
        & ordered["prior_opponent_points"].notna()
        & ordered["prior_team_points"].lt(ordered["prior_opponent_points"])
    )
    return ordered.sort_index()


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def derive_cfb_venue_position_features(
    features: pd.DataFrame,
    *,
    schedules: pd.DataFrame | None = None,
    team_own_venues: pd.DataFrame | None = None,
    lookback_seasons: int = REVENGE_LOOKBACK_SEASONS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return ``(derived, diagnostics)`` for the four candidate columns.

    ``derived`` is a ``game_id`` frame carrying the four columns. None of them
    is ever NaN: each is a complete schedule fact for every game, so "not
    flagged" is a true statement rather than a missing value. Cell 4 is signed
    (+1 the HOME team lost the most recent prior meeting, -1 the AWAY team did,
    0 no qualifying prior meeting), which is the home-oriented encoding of a
    team-side construct for a home-oriented model.
    """

    missing = sorted(_REQUIRED_FEATURE_COLUMNS.difference(features.columns))
    if missing:
        raise DataContractError(f"CFB features are missing columns: {', '.join(missing)}")

    schedule_table = load_schedules() if schedules is None else normalize_schedules(schedules)
    sequence = build_team_side_sequence(schedule_table)
    sequence = attach_prior_meeting(sequence, lookback_seasons=lookback_seasons)
    sequence["home_opener"] = flag_home_opener(sequence)
    sequence["new_venue_debut"] = flag_new_venue_debut(sequence)
    sequence["three_plus_road"] = flag_three_plus_road(sequence)

    home_side = sequence.loc[sequence["is_home_side"]].set_index("game_id")
    away_side = sequence.loc[~sequence["is_home_side"]].set_index("game_id")

    game_ids = pd.to_numeric(features["game_id"], errors="coerce").astype("Int64")
    unknown = int((~game_ids.isin(home_side.index)).sum())
    if unknown:
        raise DataContractError(
            f"{unknown} feature rows have a game_id absent from the CFB schedules snapshot; "
            "schedule position cannot be derived for them"
        )
    index = pd.Index(game_ids)

    home_opener = home_side["home_opener"].reindex(index).fillna(False).to_numpy(dtype=bool)
    venue_debut = home_side["new_venue_debut"].reindex(index).fillna(False).to_numpy(dtype=bool)
    three_road = away_side["three_plus_road"].reindex(index).fillna(False).to_numpy(dtype=bool)
    home_revenge = home_side["lost_prior_meeting"].reindex(index).fillna(False).to_numpy(dtype=bool)
    away_revenge = away_side["lost_prior_meeting"].reindex(index).fillna(False).to_numpy(dtype=bool)
    revenge = np.where(home_revenge & ~away_revenge, 1.0, 0.0) + np.where(
        away_revenge & ~home_revenge, -1.0, 0.0
    )

    derived = pd.DataFrame(
        {
            "game_id": features["game_id"].to_numpy(),
            CFB_HOME_OPENER_COLUMN: home_opener.astype(float),
            CFB_NEW_VENUE_DEBUT_COLUMN: venue_debut.astype(float),
            CFB_THREE_PLUS_ROAD_COLUMN: three_road.astype(float),
            CFB_REVENGE_PRIOR_MEETING_COLUMN: revenge.astype(float),
        }
    )

    diagnostics = _diagnostics(
        features=features,
        derived=derived,
        sequence=sequence,
        home_side=home_side,
        away_side=away_side,
        index=index,
        schedule_table=schedule_table,
        team_own_venues=team_own_venues,
        lookback_seasons=lookback_seasons,
    )
    return derived, diagnostics


def _venue_agreement(
    schedule_table: pd.DataFrame, team_own_venues: pd.DataFrame | None
) -> dict[str, Any]:
    """How far the cfbfastR-data own-venue map agrees with the per-game venue.

    Reported, never used to build a column. See this module's docstring: the
    map's ``venue_id`` is constant across seasons, so its disagreement with the
    historical per-game venue grows the further back you go, and reporting the
    per-season agreement rate is what keeps that visible.
    """

    if team_own_venues is None:
        try:
            team_own_venues = load_team_own_venues()
        except FileNotFoundError:
            return {"available": False}
    if team_own_venues.empty:
        return {"available": False}

    own = team_own_venues.loc[:, ["season", "team_id", "own_venue_id"]]
    joined = schedule_table.merge(
        own.rename(columns={"team_id": "home_id", "own_venue_id": "home_own_venue_id"}),
        on=["season", "home_id"],
        how="left",
    )
    non_neutral = joined.loc[~joined["neutral_site"]]
    resolvable = non_neutral.loc[
        non_neutral["venue_id"].notna() & non_neutral["home_own_venue_id"].notna()
    ]
    agree = resolvable["venue_id"].eq(resolvable["home_own_venue_id"])
    by_season = agree.groupby(resolvable["season"]).mean()
    distinct = team_own_venues.groupby("team_id")["own_venue_id"].nunique()
    return {
        "available": True,
        "n_non_neutral_home_games": len(non_neutral),
        "n_resolvable": len(resolvable),
        "own_venue_join_coverage": float(non_neutral["home_own_venue_id"].notna().mean()),
        "agreement_rate": float(agree.mean()) if len(resolvable) else float("nan"),
        "agreement_by_season": {str(k): float(v) for k, v in by_season.items()},
        "n_teams": len(distinct),
        "n_teams_with_multiple_own_venues": int((distinct > 1).sum()),
    }


def _diagnostics(
    *,
    features: pd.DataFrame,
    derived: pd.DataFrame,
    sequence: pd.DataFrame,
    home_side: pd.DataFrame,
    away_side: pd.DataFrame,
    index: pd.Index,
    schedule_table: pd.DataFrame,
    team_own_venues: pd.DataFrame | None,
    lookback_seasons: int,
) -> dict[str, Any]:
    season = pd.to_numeric(features["season"], errors="coerce").astype("Int64")
    home_pair = pd.to_numeric(features["home_id"], errors="coerce")
    away_pair = pd.to_numeric(features["away_id"], errors="coerce")
    pair = pd.Series(
        [tuple(sorted((int(a), int(b)))) for a, b in zip(home_pair, away_pair, strict=True)],
        index=features.index,
    )
    meetings = pd.DataFrame({"season": season, "pair": pair}).groupby(["season", "pair"]).size()

    same_season = (
        home_side["prior_same_season"].reindex(index).fillna(False).to_numpy(dtype=bool)
        & home_side["lost_prior_meeting"].reindex(index).fillna(False).to_numpy(dtype=bool)
    ) | (
        away_side["prior_same_season"].reindex(index).fillna(False).to_numpy(dtype=bool)
        & away_side["lost_prior_meeting"].reindex(index).fillna(False).to_numpy(dtype=bool)
    )

    flagged: dict[str, Any] = {}
    for column in CFB_VENUE_POSITION_FEATURE_COLUMNS:
        values = derived[column]
        nonzero = values != 0.0
        flagged[column] = {
            "n_nonzero": int(nonzero.sum()),
            "n_missing": int(values.isna().sum()),
            "mean": float(values.mean()),
            "nonzero_by_season": {
                str(k): int(v) for k, v in nonzero.groupby(season).sum().items() if int(v) > 0
            },
        }

    neutral_count: int | None = None
    if "neutral_site" in features.columns:
        neutral_count = int(
            pd.to_numeric(features["neutral_site"].astype(float), errors="coerce")
            .fillna(0.0)
            .astype(bool)
            .sum()
        )

    return {
        "n_games": len(features),
        "n_schedule_rows": len(schedule_table),
        "n_team_side_rows": len(sequence),
        "schedule_seasons": [
            int(schedule_table["season"].min()),
            int(schedule_table["season"].max()),
        ],
        "n_neutral_site": neutral_count,
        "lookback_seasons": int(lookback_seasons),
        "flagged": flagged,
        "within_season_rematch_pairs": int((meetings >= 2).sum()),
        "within_season_rematch_games": int(meetings.loc[meetings >= 2].sum()),
        "revenge_prior_meeting_same_season": int(same_season.sum()),
        "revenge_prior_meeting_earlier_season": int(
            (derived[CFB_REVENGE_PRIOR_MEETING_COLUMN] != 0.0).sum() - same_season.sum()
        ),
        "team_info_venue_agreement": _venue_agreement(schedule_table, team_own_venues),
    }


def attach_cfb_venue_position_features(
    features: pd.DataFrame,
    *,
    schedules: pd.DataFrame | None = None,
    team_own_venues: pd.DataFrame | None = None,
    lookback_seasons: int = REVENGE_LOOKBACK_SEASONS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Additively join the four candidate columns onto ``features``.

    Every pre-existing column comes back untouched; only the four new columns
    are added, mirroring ``nfl_ats.fluview_cfb_feature``'s additive-merge
    discipline.
    """

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    collisions = sorted(set(CFB_VENUE_POSITION_FEATURE_COLUMNS).intersection(features.columns))
    if collisions:
        raise DataContractError(f"features already carries {', '.join(collisions)}")

    derived, diagnostics = derive_cfb_venue_position_features(
        features,
        schedules=schedules,
        team_own_venues=team_own_venues,
        lookback_seasons=lookback_seasons,
    )
    merged = features.copy()
    for column in CFB_VENUE_POSITION_FEATURE_COLUMNS:
        merged[column] = derived[column].to_numpy()
    return merged, diagnostics


__all__ = [
    "CFB_HOME_OPENER_COLUMN",
    "CFB_NEW_VENUE_DEBUT_COLUMN",
    "CFB_REVENGE_PRIOR_MEETING_COLUMN",
    "CFB_THREE_PLUS_ROAD_COLUMN",
    "CFB_VENUE_POSITION_FEATURE_COLUMNS",
    "REVENGE_LOOKBACK_SEASONS",
    "SCHEDULE_COLUMNS",
    "TEAM_INFO_COLUMNS",
    "attach_cfb_venue_position_features",
    "attach_prior_meeting",
    "build_team_side_sequence",
    "declared_home_venues",
    "default_schedules_dir",
    "default_team_info_dir",
    "derive_cfb_venue_position_features",
    "flag_home_opener",
    "flag_new_venue_debut",
    "flag_three_plus_road",
    "load_schedules",
    "load_team_own_venues",
    "normalize_schedules",
]
