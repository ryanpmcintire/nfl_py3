"""Circadian body-clock / timezone candidate columns for COLLEGE FOOTBALL.

Cross-league replication of the four NFL cells frozen in
``docs/body_clock_screen.md`` (``body_clock_west_road_early``,
``body_clock_east_host_west_visitor_early``),
``docs/travel_rest_battery.md`` (``travel_rest_eastbound_multizone``) and
``docs/body_clock_night_screen.md``
(``body_clock_night_west_road_ge2000et``).

Predeclared in ``docs/cfb_body_clock_replication.md``. Read that first: it
freezes the population, the venue-state -> IANA-timezone map (including the
split-state resolution rule and the neutral-site rule), the comparator, the
four cells, the null, the positive control, the era split and the recording
rules before any outcome sign was computed.

**What differs from the NFL construction, and why**

1. **Timezone source.** The NFL cells read an IANA ``tz`` straight out of
   ``registry/stadium_coordinates.json``, which is NFL-only by its own README
   and carries no CFB venue. No local CFB snapshot carries a venue latitude,
   longitude or timezone either. What CFB *does* carry is the cfbfastR-data
   ``team_info`` snapshot's ``state`` and ``city`` columns, so this module
   builds the zone deterministically from ``(state, city)`` with
   :data:`STATE_TIMEZONES` and :data:`SPLIT_STATE_CITY_TIMEZONES`, DST handled
   by the standard-library ``zoneinfo`` against each game's own kickoff
   instant. No network call and no CFBD API credit.
2. **Kickoff clock.** The NFL cells parse an ET ``gametime`` string. The CFB
   benchmark table carries a tz-aware UTC ``kickoff`` (0 nulls of 12,500,
   measured), converted here to ``America/New_York``. Because CFB plays
   late-night Pacific and Hawaii kickoffs that land after midnight ET, an ET
   kickoff earlier than :data:`PAST_MIDNIGHT_ET_CUTOFF_MINUTES` is carried
   forward as ``+1440`` minutes so it reads as the previous evening's LATE
   window rather than as an "early" kickoff. Predeclared; the count of rows
   this touches is in the diagnostics.
3. **Timezone offsets are evaluated at the KICKOFF INSTANT**, not at
   ``gameday`` midnight as the NFL travel battery did. The NFL table had no
   kickoff timestamp; this one does, and the kickoff instant is both strictly
   more correct across a DST boundary and still purely pregame.

**Neutral sites.** All four cells describe a true road game at a known host
venue. ``neutral_site == 1`` means the host's listed venue is NOT where the
game is played, so the game's own timezone is unknown; every candidate column
comes back **NaN (missing, not "not flagged")** for such a row, and the row is
KEPT in the scored population with its NaN, exactly as
``nfl_ats.fluview_cfb_feature`` keeps neutral-site rows -- imputation belongs
to the model's own training-fold median, never to a feature builder that can
see every season at once.

Every column here is a pure function of pregame facts: the kickoff timestamp,
the two teams' identities, and their own listed venue states/cities that
season. ``tests/test_cfb_body_clock_feature.py`` regression-tests exactly that
by shuffling every outcome column and asserting the columns are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError

REPO_ROOT = Path(__file__).resolve().parents[2]

#: cfbfastR-data ``team_info`` columns this module needs. ``state`` and ``city``
#: sit beside ``venue_id``/``venue_name``, so they describe the school's own
#: listed VENUE -- the body clock the mechanism is about -- and they are
#: per-season, so a venue change is carried correctly.
TEAM_INFO_COLUMNS = ("team_id", "school", "venue_id", "venue_name", "city", "state")

EASTERN = "America/New_York"
CENTRAL = "America/Chicago"
MOUNTAIN = "America/Denver"
ARIZONA = "America/Phoenix"
PACIFIC = "America/Los_Angeles"

#: The NFL cells' WEST body-clock set, taken verbatim from
#: ``docs/body_clock_screen.md`` ("WEST body clock := tz in
#: {America/Los_Angeles, America/Phoenix} ... Denver (Mountain) is deliberately
#: EXCLUDED"). Kept EXACTLY as the NFL set so this replicates the recorded
#: construct rather than a broader one: Mountain-time CFB schools (Colorado,
#: Utah, Wyoming, New Mexico, Montana, Boise, UTEP) sit in the complement just
#: as Denver does, and Hawaii (``Pacific/Honolulu``, 5-6 hours behind ET) is
#: also excluded because its dose is not the NFL construct's 2-3 hours. Both
#: exclusions are disclosed with measured counts in the diagnostics.
WEST_BODY_CLOCK_ZONES: frozenset[str] = frozenset({PACIFIC, ARIZONA})

HAWAII = "Pacific/Honolulu"

#: States that lie wholly inside ONE zone (for every school in this
#: population). Two-letter USPS codes, upper-cased.
STATE_TIMEZONES: dict[str, str] = {
    "AL": CENTRAL,
    "AR": CENTRAL,
    "AZ": ARIZONA,
    "BC": "America/Vancouver",
    "CA": PACIFIC,
    "CO": MOUNTAIN,
    "CT": EASTERN,
    "DC": EASTERN,
    "DE": EASTERN,
    "GA": EASTERN,
    "HI": HAWAII,
    "IA": CENTRAL,
    "IL": CENTRAL,
    "LA": CENTRAL,
    "MA": EASTERN,
    "MD": EASTERN,
    "ME": EASTERN,
    "MN": CENTRAL,
    "MO": CENTRAL,
    "MS": CENTRAL,
    "MT": MOUNTAIN,
    "NC": EASTERN,
    "NH": EASTERN,
    "NJ": EASTERN,
    "NM": MOUNTAIN,
    "NV": PACIFIC,
    "NY": EASTERN,
    "OH": EASTERN,
    "OK": CENTRAL,
    "PA": EASTERN,
    "RI": EASTERN,
    "SC": EASTERN,
    "UT": MOUNTAIN,
    "VA": EASTERN,
    "VT": EASTERN,
    "WA": PACIFIC,
    "WI": CENTRAL,
    "WV": EASTERN,
    "WY": MOUNTAIN,
}

#: States that span two IANA zones. Declared as a SET before any outcome was
#: seen, and resolved by CITY (never by a blanket per-state assumption) --
#: ``docs/cfb_body_clock_replication.md`` section 4.
SPLIT_STATES: frozenset[str] = frozenset(
    {"FL", "TX", "TN", "KY", "IN", "MI", "ND", "SD", "NE", "KS", "OR", "ID"}
)

#: The majority-population zone of each split state. Used ONLY when a school's
#: city is absent from :data:`SPLIT_STATE_CITY_TIMEZONES`; every such use is
#: counted in the diagnostics as ``n_split_state_city_fallback`` so a silent
#: fallback can never hide inside a result.
SPLIT_STATE_DEFAULT_TIMEZONES: dict[str, str] = {
    "FL": EASTERN,
    "TX": CENTRAL,
    "TN": CENTRAL,
    "KY": EASTERN,
    "IN": EASTERN,
    "MI": EASTERN,
    "ND": CENTRAL,
    "SD": CENTRAL,
    "NE": CENTRAL,
    "KS": CENTRAL,
    "OR": PACIFIC,
    "ID": MOUNTAIN,
}

#: ``(state, lower-cased city) -> IANA zone`` for EVERY school in a split state
#: that appears in ``data/processed/cfb_game_features.parquet`` (measured: 42
#: distinct (state, city) pairs covering 43 schools -- Rice and Houston share
#: the city Houston). Entries whose zone equals the state default are listed
#: anyway, so the table is an auditable statement about each school rather than
#: a list of exceptions.
SPLIT_STATE_CITY_TIMEZONES: dict[tuple[str, str], str] = {
    # Florida: every FBS venue is Eastern; the Central panhandle hosts none.
    ("FL", "boca raton"): EASTERN,
    ("FL", "gainesville"): EASTERN,
    ("FL", "miami"): EASTERN,
    ("FL", "miami gardens"): EASTERN,
    ("FL", "orlando"): EASTERN,
    ("FL", "tallahassee"): EASTERN,
    ("FL", "tampa"): EASTERN,
    # Idaho: the panhandle (Moscow) is Pacific, the south (Boise) is Mountain.
    ("ID", "boise"): "America/Boise",
    ("ID", "moscow"): PACIFIC,
    # Indiana: every FBS venue is in the Eastern part of the state.
    ("IN", "bloomington"): "America/Indiana/Indianapolis",
    ("IN", "muncie"): "America/Indiana/Indianapolis",
    ("IN", "notre dame"): "America/Indiana/Indianapolis",
    ("IN", "west lafayette"): "America/Indiana/Indianapolis",
    # Kansas: both FBS venues are in the Central east; only far-west counties
    # are Mountain.
    ("KS", "lawrence"): CENTRAL,
    ("KS", "manhattan"): CENTRAL,
    # Kentucky: Lexington and Louisville are Eastern; Bowling Green is Central.
    ("KY", "bowling green"): CENTRAL,
    ("KY", "lexington"): EASTERN,
    ("KY", "louisville"): EASTERN,
    # Michigan: every FBS venue is Eastern; only the far western UP is Central.
    ("MI", "ann arbor"): "America/Detroit",
    ("MI", "east lansing"): "America/Detroit",
    ("MI", "kalamazoo"): "America/Detroit",
    ("MI", "mount pleasant"): "America/Detroit",
    ("MI", "ypsilanti"): "America/Detroit",
    # Nebraska: Lincoln is Central; only the panhandle is Mountain.
    ("NE", "lincoln"): CENTRAL,
    # Oregon: both FBS venues are Pacific; only Malheur County is Mountain.
    ("OR", "corvallis"): PACIFIC,
    ("OR", "eugene"): PACIFIC,
    # Tennessee: Knoxville is Eastern; the middle and west of the state are
    # Central.
    ("TN", "knoxville"): EASTERN,
    ("TN", "memphis"): CENTRAL,
    ("TN", "murfreesboro"): CENTRAL,
    ("TN", "nashville"): CENTRAL,
    # Texas: every FBS venue is Central except El Paso, which is Mountain.
    ("TX", "austin"): CENTRAL,
    ("TX", "college station"): CENTRAL,
    ("TX", "denton"): CENTRAL,
    ("TX", "el paso"): MOUNTAIN,
    ("TX", "fort worth"): CENTRAL,
    ("TX", "houston"): CENTRAL,
    ("TX", "huntsville"): CENTRAL,
    ("TX", "lubbock"): CENTRAL,
    ("TX", "san antonio"): CENTRAL,
    ("TX", "san marcos"): CENTRAL,
    ("TX", "university park"): CENTRAL,
    ("TX", "waco"): CENTRAL,
}

#: ``kick_min`` below this reads as the PREVIOUS evening's late window rather
#: than as an early kickoff: a 22:30 Pacific kickoff is 01:30 ET the next
#: calendar day, and calling that "before 14:00 ET" would invert the cell.
#: 06:00 ET is comfortably below the earliest genuine CFB kickoff (07:00 ET,
#: measured) and comfortably above the latest post-midnight one.
PAST_MIDNIGHT_ET_CUTOFF_MINUTES = 360

#: ``docs/body_clock_screen.md`` cell 1: "kickoff < 14:00 ET".
EARLY_KICKOFF_ET_MINUTES = 14 * 60
#: ``docs/body_clock_night_screen.md`` cell 1: "kickoff >= 20:00 ET".
NIGHT_KICKOFF_ET_MINUTES = 20 * 60
#: ``docs/travel_rest_battery.md`` cell 2: "tz_delta_eastbound >= 2 (hours)".
EASTBOUND_MULTIZONE_HOURS = 2.0

CFB_BODY_CLOCK_WEST_ROAD_EARLY_COLUMN = "cfb_body_clock_west_road_early"
CFB_BODY_CLOCK_EAST_HOST_WEST_VISITOR_EARLY_COLUMN = "cfb_body_clock_east_host_west_visitor_early"
CFB_TRAVEL_REST_EASTBOUND_MULTIZONE_COLUMN = "cfb_travel_rest_eastbound_multizone"
CFB_BODY_CLOCK_NIGHT_WEST_ROAD_COLUMN = "cfb_body_clock_night_west_road_ge2000et"

#: Cell key (the ``--cell`` value) -> candidate column name.
CFB_BODY_CLOCK_CELL_COLUMNS: dict[str, str] = {
    "west_road_early": CFB_BODY_CLOCK_WEST_ROAD_EARLY_COLUMN,
    "east_host_west_visitor_early": CFB_BODY_CLOCK_EAST_HOST_WEST_VISITOR_EARLY_COLUMN,
    "eastbound_multizone": CFB_TRAVEL_REST_EASTBOUND_MULTIZONE_COLUMN,
    "night_west_road": CFB_BODY_CLOCK_NIGHT_WEST_ROAD_COLUMN,
}

CFB_BODY_CLOCK_FEATURE_COLUMNS: tuple[str, ...] = tuple(CFB_BODY_CLOCK_CELL_COLUMNS.values())

_REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "week",
    "kickoff",
    "home_id",
    "away_id",
    "neutral_site",
}


# ---------------------------------------------------------------------------
# venue state/city -> IANA timezone
# ---------------------------------------------------------------------------


def resolve_timezone(state: object, city: object) -> tuple[str | None, bool]:
    """``(state, city) -> (IANA zone or None, used_split_state_fallback)``.

    Deterministic and offline. A single-zone state resolves from
    :data:`STATE_TIMEZONES`; a :data:`SPLIT_STATES` member resolves from
    :data:`SPLIT_STATE_CITY_TIMEZONES` by city, and only if that city is
    unknown does it fall back to :data:`SPLIT_STATE_DEFAULT_TIMEZONES` -- with
    the fallback flagged so the harness can count it.
    """

    if state is None or (isinstance(state, float) and np.isnan(state)):
        return None, False
    code = str(state).strip().upper()
    if not code:
        return None, False
    if code in SPLIT_STATES:
        town = "" if city is None else str(city).strip().lower()
        zone = SPLIT_STATE_CITY_TIMEZONES.get((code, town))
        if zone is not None:
            return zone, False
        return SPLIT_STATE_DEFAULT_TIMEZONES[code], True
    return STATE_TIMEZONES.get(code), False


def default_team_info_dir() -> Path:
    """Latest cfbfastR-data ``team_info`` snapshot directory."""

    candidates = sorted(REPO_ROOT.glob("data/cfb/team_info/raw/*/season=*/team_info.parquet"))
    if not candidates:
        raise FileNotFoundError(
            "no data/cfb/team_info/raw/*/season=*/team_info.parquet snapshot found"
        )
    return candidates[-1].parent.parent


def load_team_zone_map(team_info_dir: Path | None = None) -> pd.DataFrame:
    """``(season, team_id) -> body_zone`` from the cfbfastR-data snapshot.

    Returns one row per ``(season, team_id)`` with ``school``, ``state``,
    ``city``, the resolved IANA ``body_zone`` and the
    ``split_state_fallback`` flag. This frame is written out as an artifact by
    the harness so every per-school assignment can be audited.
    """

    directory = team_info_dir or default_team_info_dir()
    paths = sorted(Path(directory).glob("season=*/team_info.parquet"))
    if not paths:
        raise FileNotFoundError(f"no season=*/team_info.parquet files under {directory}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        season = int(path.parent.name.split("=", 1)[1])
        frame = pd.read_parquet(path, columns=list(TEAM_INFO_COLUMNS))
        frame["season"] = season
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table["team_id"] = pd.to_numeric(table["team_id"], errors="coerce").astype("Int64")
    table["state"] = table["state"].astype("string").str.strip().str.upper()
    table["city"] = table["city"].astype("string").str.strip()
    table = table.loc[table["team_id"].notna()].drop_duplicates(subset=["season", "team_id"])
    resolved = [
        resolve_timezone(state, city)
        for state, city in zip(table["state"], table["city"], strict=True)
    ]
    table["body_zone"] = [zone for zone, _fallback in resolved]
    table["split_state_fallback"] = [fallback for _zone, fallback in resolved]
    return table.loc[
        :,
        [
            "season",
            "team_id",
            "school",
            "venue_id",
            "venue_name",
            "city",
            "state",
            "body_zone",
            "split_state_fallback",
        ],
    ].reset_index(drop=True)


def zone_utc_offset_hours(kickoff_utc: pd.Series, zones: pd.Series) -> pd.Series:
    """UTC offset in hours of each row's ``zone``, at that row's kickoff instant.

    DST-aware by construction (``zoneinfo`` resolves the offset for the exact
    instant), vectorised one zone at a time. Rows with a missing zone come back
    NaN.
    """

    utc = pd.to_datetime(kickoff_utc, utc=True)
    naive_utc = utc.dt.tz_localize(None)
    offsets = pd.Series(np.nan, index=utc.index, dtype=float)
    for zone in sorted({str(z) for z in zones.dropna().unique()}):
        mask = zones.astype(object).eq(zone).to_numpy()
        if not mask.any():
            continue
        local = utc.loc[mask].dt.tz_convert(ZoneInfo(zone)).dt.tz_localize(None)
        offsets.loc[mask] = (local - naive_utc.loc[mask]).dt.total_seconds() / 3600.0
    return offsets


# ---------------------------------------------------------------------------
# feature construction
# ---------------------------------------------------------------------------


def eastern_kickoff_minutes(kickoff_utc: pd.Series) -> pd.Series:
    """Minutes past midnight ET, with post-midnight kickoffs carried forward.

    A kickoff earlier than :data:`PAST_MIDNIGHT_ET_CUTOFF_MINUTES` ET belongs
    to the PREVIOUS evening's late window (a 22:30 Pacific kickoff is 01:30 ET
    the next calendar day), so it is returned as ``minutes + 1440``. Without
    this the late-window cell would leak into the early-window cell.
    """

    eastern = pd.to_datetime(kickoff_utc, utc=True).dt.tz_convert(ZoneInfo(EASTERN))
    minutes = (eastern.dt.hour * 60 + eastern.dt.minute).astype(float)
    return minutes.where(minutes >= PAST_MIDNIGHT_ET_CUTOFF_MINUTES, minutes + 1440.0)


def attach_cfb_body_clock_context(
    features: pd.DataFrame, team_zones: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Working frame carrying every pregame quantity the four cells need."""

    missing = sorted(_REQUIRED_COLUMNS.difference(features.columns))
    if missing:
        raise DataContractError(f"CFB features are missing columns: {', '.join(missing)}")

    frame = features.loc[:, sorted(_REQUIRED_COLUMNS)].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True, errors="raise")
    frame["home_id"] = pd.to_numeric(frame["home_id"], errors="coerce").astype("Int64")
    frame["away_id"] = pd.to_numeric(frame["away_id"], errors="coerce").astype("Int64")
    frame["is_neutral"] = (
        pd.to_numeric(frame["neutral_site"], errors="coerce").fillna(0).astype(int).eq(1)
    )

    zones = team_zones if team_zones is not None else load_team_zone_map()
    lookup = zones.loc[:, ["season", "team_id", "body_zone", "split_state_fallback", "state"]]
    frame = frame.merge(
        lookup.rename(
            columns={
                "team_id": "home_id",
                "body_zone": "home_zone",
                "split_state_fallback": "home_split_fallback",
                "state": "home_state",
            }
        ),
        on=["season", "home_id"],
        how="left",
    )
    frame = frame.merge(
        lookup.rename(
            columns={
                "team_id": "away_id",
                "body_zone": "away_zone",
                "split_state_fallback": "away_split_fallback",
                "state": "away_state",
            }
        ),
        on=["season", "away_id"],
        how="left",
    )
    frame["home_zone"] = frame["home_zone"].astype(object)
    frame["away_zone"] = frame["away_zone"].astype(object)

    frame["kick_min_et"] = eastern_kickoff_minutes(frame["kickoff"])
    frame["past_midnight_et"] = frame["kick_min_et"].ge(1440.0)
    frame["venue_offset_hours"] = zone_utc_offset_hours(frame["kickoff"], frame["home_zone"])
    frame["away_body_offset_hours"] = zone_utc_offset_hours(frame["kickoff"], frame["away_zone"])
    frame["eastern_offset_hours"] = zone_utc_offset_hours(
        frame["kickoff"], pd.Series(EASTERN, index=frame.index)
    )
    frame["tz_delta_eastbound"] = frame["venue_offset_hours"] - frame["away_body_offset_hours"]
    frame["away_body_west"] = frame["away_zone"].isin(WEST_BODY_CLOCK_ZONES)
    frame["venue_is_eastern"] = frame["venue_offset_hours"].eq(frame["eastern_offset_hours"])
    return frame


def derive_cfb_body_clock_features(
    features: pd.DataFrame, *, team_zones: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return ``(derived, diagnostics)``.

    ``derived`` is a ``game_id`` frame carrying the four candidate columns as
    1.0/0.0/NaN floats. Every cell is NaN on a neutral-site game (the game's
    own timezone is unknown there) and NaN whenever a required zone did not
    resolve.
    """

    frame = attach_cfb_body_clock_context(features, team_zones)
    inapplicable = (
        frame["is_neutral"] | frame["home_zone"].isna() | frame["away_zone"].isna()
    ).to_numpy()

    early = frame["kick_min_et"].lt(EARLY_KICKOFF_ET_MINUTES).to_numpy()
    night = frame["kick_min_et"].ge(NIGHT_KICKOFF_ET_MINUTES).to_numpy()
    west = frame["away_body_west"].to_numpy()
    eastern_host = frame["venue_is_eastern"].to_numpy()
    eastbound = frame["tz_delta_eastbound"].ge(EASTBOUND_MULTIZONE_HOURS).to_numpy()

    flags = {
        CFB_BODY_CLOCK_WEST_ROAD_EARLY_COLUMN: west & early,
        CFB_BODY_CLOCK_EAST_HOST_WEST_VISITOR_EARLY_COLUMN: west & early & eastern_host,
        CFB_TRAVEL_REST_EASTBOUND_MULTIZONE_COLUMN: eastbound,
        CFB_BODY_CLOCK_NIGHT_WEST_ROAD_COLUMN: west & night,
    }
    derived = pd.DataFrame({"game_id": features["game_id"].to_numpy()})
    for column, flag in flags.items():
        derived[column] = np.where(inapplicable, np.nan, flag.astype(float))

    covered = ~inapplicable
    flagged_by_season: dict[str, dict[str, int]] = {}
    for column in CFB_BODY_CLOCK_FEATURE_COLUMNS:
        values = pd.to_numeric(derived[column], errors="coerce")
        flagged_by_season[column] = {
            str(int(str(season))): int(group.sum())
            for season, group in values.groupby(frame["season"].to_numpy())
        }

    diagnostics: dict[str, Any] = {
        "n_games": len(frame),
        "n_neutral_site": int(frame["is_neutral"].sum()),
        "n_unresolved_zone": int((frame["home_zone"].isna() | frame["away_zone"].isna()).sum()),
        "n_split_state_city_fallback": int(
            (
                frame["home_split_fallback"].fillna(False).astype(bool)
                | frame["away_split_fallback"].fillna(False).astype(bool)
            ).sum()
        ),
        "n_past_midnight_et_adjusted": int(frame["past_midnight_et"].sum()),
        "n_applicable": int(covered.sum()),
        "n_hawaii_body_clock_excluded": int(frame["away_zone"].astype(object).eq(HAWAII).sum()),
        "n_mountain_body_clock_excluded": int(
            frame["away_zone"].astype(object).isin({MOUNTAIN, "America/Boise"}).sum()
        ),
        "zones_present": sorted(
            {str(z) for z in frame["home_zone"].dropna().unique()}
            | {str(z) for z in frame["away_zone"].dropna().unique()}
        ),
        "states_present": sorted(
            {str(s) for s in frame["home_state"].dropna().unique()}
            | {str(s) for s in frame["away_state"].dropna().unique()}
        ),
        "flagged_total": {
            column: int(pd.to_numeric(derived[column], errors="coerce").sum())
            for column in CFB_BODY_CLOCK_FEATURE_COLUMNS
        },
        "flagged_by_season": flagged_by_season,
        "home_zone_coverage_by_season": {
            str(int(str(season))): float(group.notna().mean())
            for season, group in frame["home_zone"].groupby(frame["season"].to_numpy())
        },
        "away_zone_coverage_by_season": {
            str(int(str(season))): float(group.notna().mean())
            for season, group in frame["away_zone"].groupby(frame["season"].to_numpy())
        },
        "context": frame,
    }
    return derived, diagnostics


def attach_cfb_body_clock_features(
    features: pd.DataFrame, *, team_zones: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Additively join the four candidate columns onto ``features``.

    Every pre-existing column comes back bit-identical; only the four new
    columns are added, mirroring
    ``nfl_ats.fluview_cfb_feature.attach_cfb_fluview_features``'s additive
    merge discipline.
    """

    if "game_id" not in features.columns:
        raise DataContractError("features is missing the game_id join key")
    collisions = sorted(set(CFB_BODY_CLOCK_FEATURE_COLUMNS).intersection(features.columns))
    if collisions:
        raise DataContractError(f"features already carries {', '.join(collisions)}")

    derived, diagnostics = derive_cfb_body_clock_features(features, team_zones=team_zones)
    merged = features.copy()
    for column in CFB_BODY_CLOCK_FEATURE_COLUMNS:
        merged[column] = derived[column].to_numpy()
    return merged, diagnostics


def cell_exposure_panel(context: pd.DataFrame, derived: pd.DataFrame, column: str) -> pd.DataFrame:
    """Team-game long panel of *exposure to a cell*, for split-half reliability.

    Each cell is a pure schedule fact, so the "trait" whose reliability can be
    measured is a team-season's PROPENSITY to be in the cell. The panel carries
    one row per team per game -- ``team_id``, ``season``, ``week``,
    ``in_cell`` -- where ``in_cell`` is the cell's flag for that game and the
    team is the one the mechanism is about (the visiting team for all four
    cells). ``docs/cfb_body_clock_replication.md`` section 7 states why this
    panel and not another.
    """

    values = pd.to_numeric(derived[column], errors="coerce").to_numpy()
    panel = pd.DataFrame(
        {
            "team_id": context["away_id"].to_numpy(),
            "season": context["season"].to_numpy(),
            "week": context["week"].to_numpy(),
            "in_cell": values,
        }
    )
    return panel.loc[panel["team_id"].notna() & panel["in_cell"].notna()].reset_index(drop=True)


def body_clock_offset_panel(context: pd.DataFrame) -> pd.DataFrame:
    """Team-game long panel of each team's OWN body-clock UTC offset.

    The second reliability read reported in section 7: the underlying tz map
    itself, as a team-season trait. Both sides of every game contribute.
    """

    home = pd.DataFrame(
        {
            "team_id": context["home_id"].to_numpy(),
            "season": context["season"].to_numpy(),
            "week": context["week"].to_numpy(),
            "body_offset_hours": zone_utc_offset_hours(
                context["kickoff"], context["home_zone"]
            ).to_numpy(),
        }
    )
    away = pd.DataFrame(
        {
            "team_id": context["away_id"].to_numpy(),
            "season": context["season"].to_numpy(),
            "week": context["week"].to_numpy(),
            "body_offset_hours": context["away_body_offset_hours"].to_numpy(),
        }
    )
    panel = pd.concat([home, away], ignore_index=True)
    return panel.loc[panel["team_id"].notna() & panel["body_offset_hours"].notna()].reset_index(
        drop=True
    )


def zone_assignment_table(team_zones: pd.DataFrame) -> list[dict[str, Any]]:
    """The per-school assignment table, as plain records for an artifact."""

    frame = team_zones.copy()
    frame["team_id"] = pd.to_numeric(frame["team_id"], errors="coerce").astype("Int64")
    frame["venue_id"] = pd.to_numeric(frame["venue_id"], errors="coerce").astype("Int64")
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype(int)
    frame = frame.sort_values(["state", "city", "school", "season"])

    def text(value: Any) -> str | None:
        """Render a possibly-missing cell as JSON ``null`` rather than "nan"."""

        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return None
        return str(value)

    records: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        venue_id = record["venue_id"]
        records.append(
            {
                "season": int(record["season"]),
                "team_id": int(record["team_id"]),
                "school": text(record["school"]),
                "city": text(record["city"]),
                "state": text(record["state"]),
                "venue_id": None if pd.isna(venue_id) else int(venue_id),
                "venue_name": text(record["venue_name"]),
                "body_zone": text(record["body_zone"]),
                "split_state_fallback": bool(record["split_state_fallback"]),
            }
        )
    return records


__all__ = [
    "CFB_BODY_CLOCK_CELL_COLUMNS",
    "CFB_BODY_CLOCK_EAST_HOST_WEST_VISITOR_EARLY_COLUMN",
    "CFB_BODY_CLOCK_FEATURE_COLUMNS",
    "CFB_BODY_CLOCK_NIGHT_WEST_ROAD_COLUMN",
    "CFB_BODY_CLOCK_WEST_ROAD_EARLY_COLUMN",
    "CFB_TRAVEL_REST_EASTBOUND_MULTIZONE_COLUMN",
    "EARLY_KICKOFF_ET_MINUTES",
    "EASTBOUND_MULTIZONE_HOURS",
    "NIGHT_KICKOFF_ET_MINUTES",
    "PAST_MIDNIGHT_ET_CUTOFF_MINUTES",
    "SPLIT_STATES",
    "SPLIT_STATE_CITY_TIMEZONES",
    "SPLIT_STATE_DEFAULT_TIMEZONES",
    "STATE_TIMEZONES",
    "TEAM_INFO_COLUMNS",
    "WEST_BODY_CLOCK_ZONES",
    "attach_cfb_body_clock_context",
    "attach_cfb_body_clock_features",
    "body_clock_offset_panel",
    "cell_exposure_panel",
    "default_team_info_dir",
    "derive_cfb_body_clock_features",
    "eastern_kickoff_minutes",
    "load_team_zone_map",
    "resolve_timezone",
    "zone_assignment_table",
    "zone_utc_offset_hours",
]
