"""Forecast cold-visitor tilt overlay: a parameter-free pick-level nudge built
on the Tuesday-noon GFS-MOS temperature forecast.

Research chain: ``docs/forecast_weather_screen.md`` (ENV-01 payoff screen,
predeclared 2026-08-19 before any cell was scored) cell 4,
``forecast_weather_temp_gap_cold_visitor``. Measured this session (read
directly from ``registry/weak_signals.json``):
``forecast_weather_temp_gap_cold_visitor`` -- +0.4305 accuracy points, 95%
[-0.2239, +1.0207], ``probability_positive`` 0.9029, REG 2020-2025, n=1,582
paired games (forecast-archive coverage). Its actual-weather sibling,
``weather_followup_temp_gap_cold_visitor`` (the mechanism this forecast-based
cell re-screens with the Tuesday-noon forecast substituted for the game-time
actual, per ``docs/forecast_weather_screen.md``'s "Leakage posture" section):
+0.3836 accuracy points, 95% [+0.0017, +0.7541], ``probability_positive``
0.9755, REG 2009-2025 -- the strongest cell in the whole weather follow-up
battery and the only one of the four re-screened cells whose registered
interval already excludes zero. Forecast-vs-actual flag agreement on the
matched 2020-2025 population: **92.72%** (0.9271523178807947,
n_both_defined=604, both_true=62, both_false=498, forecast_only=36,
actual_only=8 -- read from the ``notes`` field of
``registry/weak_signals.json:forecast_weather_temp_gap_cold_visitor``).

Both intervals cross zero. Per AGENTS.md that is the EXPECTED shape for a
real small signal at this evaluator's ~2-point resolution, never grounds to
decline a no-window-cost prospective challenger; neither admissible closing
ground applies to either read (no resolved wrong sign, no positive-control
bound was run), so both stay ``unresolved_below_power`` in the registry.

**Flag, ported from ``scripts/nfl_forecast_weather_screen.py``'s cell 4 /
``docs/forecast_weather_screen.md`` cell 4:** outdoor AND (away team's own
climatological-normal outdoor home temp -- an ACTUAL-weather baseline --
minus this game's Tuesday-noon forecast temp) ``>= 25`` degrees F.

**Deliberate, disclosed adaptation for live pregame safety.** The REGISTERED
evidence's climatological baseline is a WITHIN-SEASON aggregate over the away
team's OTHER home games that season (``scripts/nfl_forecast_weather_screen.py``'s
own comment: "not season-causal", a disclosed, precedented research
convention shared with the actual-weather battery this cell mirrors) -- fine
for a backtest that scores every 2020-2025 game at once, but NOT pregame-safe
for a live weekly build: a Week 4 prediction cannot see the away team's
actual home temps in Weeks 5-17 of the SAME season. This module therefore
computes ``climate_temp`` from every STRICTLY EARLIER outdoor home game the
away team has played, across ANY season (not just the current one) -- the
same "prior games only" convention every other pregame-safe overlay in this
package already uses (``coach_fade_overlay``, ``backup_qb_fade_overlay``,
``division_revenge_tilt_overlay``). This is a conservative, disclosed
deviation from the exact registered research construction, chosen because
AGENTS.md requires pregame features to use only information available before
the prediction timestamp, plus a leakage regression test for every new
feature family -- not a bug, and not silently made: the registered
+0.4305/P+0.9029 (forecast) and +0.3836/P+0.9755 (actual) figures above do
NOT transfer exactly to this live arm's own climatology; a fresh, independent
2026 measurement is what the prospective ledger accrues for this specific
live construction.

**Live data path, ported from ``scripts/ingest_forecast_archive.py``
(``tuesday_noon`` cutoff, model ``MEX`` -- the SAME cutoff/model the
registered forecast archive used, per ``docs/forecast_archive_build.md``).**
Fetches the archived GFS MOS Extended bulletin issued at-or-before Tuesday
12:00 ET of the target week from the Iowa Environmental Mesonet's public MOS
JSON API, walked strictly BACKWARD in 12-hour steps (never forward, never a
fresher issuance) until a non-empty bulletin is found, keyed by the ICAO
station mapped from each game's ``stadium`` display string via
``registry/reference/stadium_station_map.csv``.

**Deliberate deviation from this task's literal instruction to use
``registry/stadium_coordinates.json``**, disclosed here: ``docs/forecast_archive_build.md``
("Station mapping" section, read this session) explicitly records that
``registry/stadium_coordinates.json`` (lat/lon/tz, built for the unrelated
ENV-03/04 travel-rest battery) was **not used** by ``scripts/ingest_forecast_archive.py`` --
"this pipeline is station-based (ICAO codes into the MOS API), not
grid-based, so it has no lat/lon dependency." Reusing the actual fetch logic
therefore means reusing the actual station-mapping file it depends on,
``registry/reference/stadium_station_map.csv``, not the lat/lon table; using
the latter would require inventing a grid-lookup path the real pipeline does
not have and has never validated. This is the conservative choice per this
task's own instruction to pick the conservative option under ambiguity.

**FAIL-OPEN, unconditionally.** Any failure fetching or parsing live forecast
data -- a missing/unreadable station map, a stadium absent from it, a network
timeout, a malformed API response, anything -- is caught by
:func:`fetch_tuesday_noon_forecast_temps_fail_open`, logged as a
``RuntimeWarning``, and folded into "every game gets no forecast, the flag is
False everywhere" rather than raised. This overlay must never be able to
block a publish; per-game fetch problems (a single station with no bulletin,
one slow request) already fold into "not flagged" one level down, inside
:func:`fetch_one_game_temp`, mirroring ``scripts/ingest_forecast_archive.py``'s
own ``fetch_status`` design.

**Tilt direction: HOME**, matching the cell's own predicted direction
("predicted home_cover edge", ``docs/forecast_weather_screen.md`` cell 4) and
mirroring ``surface_switch_tilt_overlay``'s deliberately ASYMMETRIC pattern:
flip AWAY -> HOME only when the flag fires AND the model's own pick is
currently on the away side; never the reverse (no measured direction exists
for a warm-visitor or non-flagged case).

This module is the no-window-cost path, built on the exact pattern of
``surface_switch_tilt_overlay.py``, ``backup_qb_fade_overlay.py``,
``division_revenge_tilt_overlay.py``, and ``injury_value_tilt_overlay.py``: a
**pick-level, post-prediction transform** of the active model's own forced
pick, dual-tracked against that same active model in the prospective
challenger ledger (``nfl_ats.prospective_scoring``), at no rotation-registry
window cost and with zero training-time feature changes. **Nothing in this
module is wired into ``publishing.py`` or the production pick path** -- like
the tilt/fade siblings, no owner decision to play this on the real card has
been made; it is dual-tracked only.

:func:`record_forecast_cold_visitor_tilt_challenger_decisions` writes the
overlay's own arm to the prospective challenger ledger so 2026 scores it
cleanly, independent of whether it is ever played on the real card.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.prospective_scoring import (
    ACTIVE_CHALLENGER_STATUS,
    CHALLENGER_DECISION_COLUMNS,
    artifact_model_config,
    challenger_ledger_path,
    config_fingerprint,
    find_challenger,
    load_challenger_decisions,
)
from nfl_ats.provenance import sha256_file
from nfl_ats.snapshots import latest_snapshot, load_snapshot

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "forecast_cold_visitor_tilt"

#: Ported verbatim from scripts/nfl_forecast_weather_screen.py's own module
#: constants -- the same normalization/threshold the registry measurement used.
OUTDOOR_ROOFS = frozenset({"outdoors", "open"})
TEMP_GAP_THRESHOLD_F = 25.0  # weather_followup_temp_gap_cold_visitor's threshold, reused verbatim

# ---------------------------------------------------------------------------
# Live-fetch constants and helpers, ported from scripts/ingest_forecast_archive.py
# -- same MOS API, same tuesday_noon cutoff, same model (MEX), same
# point-in-time walk-backward discipline. Only wind/knots handling is
# dropped: this overlay only needs forecast temperature.
# ---------------------------------------------------------------------------

MOS_API = "https://mesonet.agron.iastate.edu/api/1/mos.json"
#: MEX (GFS MOS Extended) matches the registered forecast archive's own
#: tuesday_noon cutoff (docs/forecast_archive_build.md), not the
#: kickoff_nearest pilot's GFS model -- this overlay's evidence and its live
#: cutoff must use the same model.
MOS_MODEL = "MEX"
USER_AGENT = "nfl-ats-research/0.1 (private research; contact ryanpmcintire@gmail.com)"
DELAY_SECONDS_DEFAULT = 0.3
MAX_LOOKBACK_STEPS_DEFAULT = 10  # 10 * 12h = 5 days back from the Tuesday-noon-ET cutoff
ET = ZoneInfo("America/New_York")

#: registry/reference/stadium_station_map.csv is keyed on the schedules
#: parquet's own `stadium` display string (registry_root-relative).
STATION_MAP_RELATIVE_PATH = Path("reference") / "stadium_station_map.csv"


def tuesday_noon_et_cutoff_utc(kickoff_utc: pd.Timestamp) -> pd.Timestamp:
    """Ported verbatim from ``scripts/ingest_forecast_archive.py``.

    Most recent Tuesday <= the kickoff's ET calendar date, at 12:00 ET,
    returned as a UTC timestamp -- the pool's own decision-lock cutoff
    (``docs/pool_edge_plan.md`` line 80).
    """

    kickoff_et = kickoff_utc.tz_convert(ET)
    et_date: date = kickoff_et.date()
    days_since_tuesday = (et_date.weekday() - 1) % 7
    tuesday_date = et_date - timedelta(days=days_since_tuesday)
    tuesday_noon_et = datetime(
        tuesday_date.year, tuesday_date.month, tuesday_date.day, 12, 0, tzinfo=ET
    )
    return pd.Timestamp(tuesday_noon_et).tz_convert(UTC)


def floor_to_12h_utc(dt: pd.Timestamp) -> datetime:
    """Ported verbatim from ``scripts/ingest_forecast_archive.py``."""

    hour = 12 if dt.hour >= 12 else 0
    return datetime(dt.year, dt.month, dt.day, hour, 0, tzinfo=UTC)


def candidate_runtimes(cutoff_utc: pd.Timestamp, max_steps: int) -> list[datetime]:
    """Ported verbatim from ``scripts/ingest_forecast_archive.py``.

    Never selects a bulletin issued after ``cutoff_utc`` -- point-in-time
    discipline is enforced by this walk, starting from the cutoff floored to
    the nearest 00Z/12Z MOS cycle and stepping strictly backward.
    """

    start = floor_to_12h_utc(cutoff_utc)
    return [start - timedelta(hours=12 * i) for i in range(max_steps)]


class MosFetchError(RuntimeError):
    """Ported verbatim from ``scripts/ingest_forecast_archive.py``."""


def fetch_mos_bulletin(
    station: str,
    runtime_utc: datetime,
    *,
    model: str,
    timeout: float = 20.0,
    retries: int = 2,
) -> list[dict[str, Any]]:
    """Ported verbatim from ``scripts/ingest_forecast_archive.py``.

    Returns the ``data`` rows for one station/runtime, or ``[]`` if IEM has
    no bulletin for that exact runtime (a normal, expected outcome to walk
    past, not an error). Raises :class:`MosFetchError` on a genuine
    transport failure after retries, so the caller can distinguish "no
    bulletin" from "IEM is down" -- callers here always catch this and fold
    it into a "not flagged" fetch status, never propagate it.
    """

    runtime_str = runtime_utc.strftime("%Y-%m-%dT%H:%MZ")
    url = f"{MOS_API}?station={station}&model={model}&runtime={runtime_str}"
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                payload = json.load(resp)
            if "data" in payload:
                return list(payload["data"])
            return []  # "no results" detail response -> no bulletin, not an error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            time.sleep(1.0 * (attempt + 1))
    raise MosFetchError(f"{station} {runtime_str}: {last_exc}")


def nearest_row(rows: list[dict[str, Any]], kickoff_utc: pd.Timestamp) -> dict[str, Any] | None:
    """Ported verbatim from ``scripts/ingest_forecast_archive.py``."""

    if not rows:
        return None
    target = kickoff_utc.to_pydatetime()
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)

    def gap(row: dict[str, Any]) -> float:
        ftime = datetime.fromisoformat(row["ftime_utc"]).replace(tzinfo=UTC)
        return abs((ftime - target).total_seconds())

    return min(rows, key=gap)


FetchBulletin = Callable[..., list[dict[str, Any]]]


def fetch_one_game_temp(
    station: str,
    kickoff_utc: pd.Timestamp,
    cutoff_utc: pd.Timestamp,
    *,
    model: str = MOS_MODEL,
    max_lookback_steps: int = MAX_LOOKBACK_STEPS_DEFAULT,
    delay_seconds: float = DELAY_SECONDS_DEFAULT,
    fetch_bulletin: FetchBulletin = fetch_mos_bulletin,
) -> dict[str, Any]:
    """Adapted from ``scripts/ingest_forecast_archive.py::fetch_one_game``,
    trimmed to temperature only (this overlay does not use wind).

    Never raises: a transport failure at any lookback step is folded into
    ``fetch_status="transport_error"`` (per-game fail-open); the outer
    :func:`fetch_tuesday_noon_forecast_temps_fail_open` wrapper handles
    whole-batch failures (a missing station map, an entirely unmapped
    stadium).
    """

    for runtime_utc in candidate_runtimes(cutoff_utc, max_lookback_steps):
        try:
            rows = fetch_bulletin(station, runtime_utc, model=model)
        except MosFetchError:
            time.sleep(delay_seconds)
            return {"forecast_temp_f": None, "fetch_status": "transport_error"}
        time.sleep(delay_seconds)
        if rows:
            row = nearest_row(rows, kickoff_utc)
            assert row is not None
            tmp = row.get("tmp")
            return {
                "forecast_temp_f": float(tmp) if tmp is not None else None,
                "fetch_status": "ok",
            }
    return {"forecast_temp_f": None, "fetch_status": "no_bulletin_within_lookback"}


def _fetch_tuesday_noon_forecast_temps(
    games: pd.DataFrame,
    station_map_path: Path,
    *,
    model: str = MOS_MODEL,
    max_lookback_steps: int = MAX_LOOKBACK_STEPS_DEFAULT,
    delay_seconds: float = DELAY_SECONDS_DEFAULT,
    fetch_bulletin: FetchBulletin = fetch_mos_bulletin,
) -> pd.DataFrame:
    """Fetch the Tuesday-noon-ET forecast temperature for every game.

    ``games`` needs ``game_id``, ``stadium``, ``kickoff`` (tz-aware or naive
    UTC). Raises on a whole-batch problem (missing station map file, a
    stadium entirely absent from it) -- callers must use
    :func:`fetch_tuesday_noon_forecast_temps_fail_open` for the fail-open
    contract this overlay requires; this inner function stays strict so its
    own unit tests can assert on the raise directly.
    """

    required = {"game_id", "stadium", "kickoff"}
    missing = sorted(required.difference(games.columns))
    if missing:
        raise DataContractError(
            f"games is missing columns for forecast fetch: {', '.join(missing)}"
        )
    if not station_map_path.is_file():
        raise FileNotFoundError(f"No stadium/station map at {station_map_path}")

    station_map = pd.read_csv(station_map_path)
    merged = games[["game_id", "stadium", "kickoff"]].merge(
        station_map[["stadium", "icao_station", "mappable"]], on="stadium", how="left"
    )
    # A stadium genuinely absent from the reference table leaves BOTH
    # icao_station and mappable null; a deliberately unmappable international
    # stadium is present with icao_station null but mappable=False (not
    # null) -- only the former is an error to fail on (mirrors
    # scripts/ingest_forecast_archive.py::load_population exactly).
    unmapped = sorted(
        merged.loc[merged["mappable"].isna(), "stadium"].astype(str).unique().tolist()
    )
    if unmapped:
        raise ValueError(f"{len(unmapped)} stadium(s) not in {station_map_path}: {unmapped}")
    merged["mappable"] = merged["mappable"].astype(bool)

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        game_id = str(row.game_id)
        if not row.mappable:
            rows.append(
                {
                    "game_id": game_id,
                    "forecast_temp_f": None,
                    "fetch_status": "unmappable_international_stadium",
                }
            )
            continue
        kickoff_utc = pd.Timestamp(row.kickoff)  # type: ignore[arg-type]
        if kickoff_utc.tzinfo is None:
            kickoff_utc = kickoff_utc.tz_localize(UTC)
        cutoff_utc = tuesday_noon_et_cutoff_utc(kickoff_utc)
        fetched = fetch_one_game_temp(
            str(row.icao_station),
            kickoff_utc,
            cutoff_utc,
            model=model,
            max_lookback_steps=max_lookback_steps,
            delay_seconds=delay_seconds,
            fetch_bulletin=fetch_bulletin,
        )
        rows.append({"game_id": game_id, **fetched})
    return pd.DataFrame(rows)


def fetch_tuesday_noon_forecast_temps_fail_open(
    games: pd.DataFrame,
    station_map_path: Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """FAIL-OPEN wrapper around :func:`_fetch_tuesday_noon_forecast_temps`.

    Any exception -- a missing/unreadable station map, a stadium absent from
    it, a network timeout, a malformed API response -- is caught, logged as
    a ``RuntimeWarning``, and folded into "every game gets no forecast" (a
    ``fetch_temp_f`` of NaN, ``fetch_status="fetch_failed"``) so downstream
    flag computation naturally reads zero flags rather than raising. This
    overlay must never be able to block a publish.
    """

    try:
        return _fetch_tuesday_noon_forecast_temps(games, station_map_path, **kwargs)
    except Exception as exc:
        warnings.warn(
            "forecast_cold_visitor_tilt: forecast fetch failed, proceeding with zero "
            f"flags ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(
            {
                "game_id": games["game_id"].astype(str),
                "forecast_temp_f": np.nan,
                "fetch_status": "fetch_failed",
            }
        )


# ---------------------------------------------------------------------------
# Pregame-safe climatology + flag (see module docstring for why this is a
# strictly-prior-games construction, not the registered within-season one).
# ---------------------------------------------------------------------------


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def team_climate_temp_by_away_game(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per REG-season ``game_id``: the AWAY team's own
    climatological-normal outdoor home temp, ``climate_temp``.

    Pregame-safe by construction: for a game where team A plays away,
    ``climate_temp`` is the mean ACTUAL temp over every outdoor home game A
    played STRICTLY BEFORE this game's own ``gameday`` -- across ANY season,
    not just the current one (see the module docstring for why this departs
    from the registered within-season research construction). ``NaN`` when A
    has no qualifying prior outdoor home game yet (its own first such game,
    or every prior home game was indoors) -- folds into "not flagged"
    downstream, never an error.
    """

    required = {
        "game_id",
        "season",
        "game_type",
        "gameday",
        "home_team",
        "away_team",
        "roof",
        "temp",
    }
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise DataContractError(
            f"schedules is missing columns for forecast cold-visitor tracking: {', '.join(missing)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["gameday"] = pd.to_datetime(reg["gameday"], errors="raise")
    reg["outdoor"] = reg["roof"].isin(OUTDOOR_ROOFS)
    reg["temp"] = pd.to_numeric(reg["temp"], errors="coerce")

    home = (
        reg[["home_team", "gameday", "game_id", "outdoor", "temp"]]
        .sort_values(["gameday", "game_id"])
        .reset_index(drop=True)
    )
    outdoor_temp = home["temp"].where(home["outdoor"])
    valid = outdoor_temp.notna()
    # Cumulative sum/count computed with explicit fill (not pandas cumsum's
    # own skipna semantics) so a non-outdoor/missing-temp home game simply
    # carries the running total forward instead of injecting a NaN gap.
    cum_sum = outdoor_temp.fillna(0.0).groupby(home["home_team"]).cumsum()
    cum_count = valid.groupby(home["home_team"]).cumsum()
    # climate_temp_after INCLUDES the current row's own home game -- correct,
    # because the as-of join below only ever looks up a home game strictly
    # BEFORE the target away game, so "including this home game" is exactly
    # "including every home game up to and including the most recent one
    # strictly prior to the away game".
    home = home.assign(climate_temp_after=(cum_sum / cum_count.replace(0, np.nan)).to_numpy())

    climate_timeline = (
        home[["home_team", "gameday", "climate_temp_after"]]
        .rename(columns={"home_team": "team"})
        .sort_values("gameday")
        .reset_index(drop=True)
    )
    away = (
        reg[["game_id", "away_team", "gameday"]]
        .rename(columns={"away_team": "team"})
        .sort_values("gameday")
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        away,
        climate_timeline,
        on="gameday",
        by="team",
        direction="backward",
        allow_exact_matches=False,
    )
    result = merged[["game_id", "climate_temp_after"]].rename(
        columns={"climate_temp_after": "climate_temp"}
    )
    result["game_id"] = result["game_id"].astype(str)
    return result


def forecast_cold_visitor_flag_by_game(
    schedules: pd.DataFrame, forecasts: pd.DataFrame
) -> pd.DataFrame:
    """One row per REG-season ``game_id``: ``forecast_cold_visitor_flag``,
    ``climate_temp``, ``forecast_temp_f``.

    ``forecasts`` needs ``game_id``, ``forecast_temp_f`` (one row per game;
    extra columns, including a missing/NaN temp for any game, are fine --
    they fold into "not flagged"). Mirrors
    ``scripts/nfl_forecast_weather_screen.py`` cell 4's flag definition
    (outdoor AND ``climate_temp - forecast_temp_f >= 25``), with
    :func:`team_climate_temp_by_away_game`'s pregame-safe climatology in
    place of the registered within-season aggregate.
    """

    required_forecast = {"game_id", "forecast_temp_f"}
    missing = sorted(required_forecast.difference(forecasts.columns))
    if missing:
        raise DataContractError(f"forecasts is missing columns: {', '.join(missing)}")
    required_schedule = {"game_id", "game_type", "roof"}
    missing_schedule = sorted(required_schedule.difference(schedules.columns))
    if missing_schedule:
        raise DataContractError(
            f"schedules is missing columns for forecast cold-visitor tracking: "
            f"{', '.join(missing_schedule)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["game_id"] = reg["game_id"].astype(str)
    reg["outdoor"] = reg["roof"].isin(OUTDOOR_ROOFS)

    # team_climate_temp_by_away_game validates its own (larger) required
    # column set -- season/gameday/home_team/away_team/temp -- and raises
    # the same DataContractError shape if any are missing.
    climate = team_climate_temp_by_away_game(schedules)
    frame = reg[["game_id", "outdoor"]].merge(climate, on="game_id", how="left")

    forecasts = forecasts.copy()
    forecasts["game_id"] = forecasts["game_id"].astype(str)
    forecasts["forecast_temp_f"] = pd.to_numeric(forecasts["forecast_temp_f"], errors="coerce")
    frame = frame.merge(
        forecasts[["game_id", "forecast_temp_f"]].drop_duplicates(subset="game_id"),
        on="game_id",
        how="left",
    )

    gap = frame["climate_temp"] - frame["forecast_temp_f"]
    flag = frame["outdoor"] & gap.ge(TEMP_GAP_THRESHOLD_F)
    frame["forecast_cold_visitor_flag"] = flag.fillna(False).astype(bool)
    return frame[["game_id", "forecast_cold_visitor_flag", "climate_temp", "forecast_temp_f"]]


@dataclass(frozen=True)
class ForecastColdVisitorFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    away_team: str
    home_team: str
    climate_temp: float
    forecast_temp_f: float
    temp_gap_f: float


@dataclass(frozen=True)
class ForecastColdVisitorResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows -- every other column stays
    byte-identical, mirroring ``surface_switch_tilt_overlay.TiltResult``.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[ForecastColdVisitorFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_forecast_cold_visitor_tilt_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    enabled: bool = True,
) -> ForecastColdVisitorResult:
    """Flip the forced pick from AWAY to HOME wherever the flag fires.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present;
    * :func:`forecast_cold_visitor_flag_by_game` fires for the game (outdoor
      AND the away team's pregame-safe climate baseline minus the Tuesday-
      noon forecast temp is at least 25F); and
    * the model's own pick (``home_cover_probability >= 0.5`` picks home) is
      currently on the AWAY side.

    **Deliberately ASYMMETRIC**, mirroring ``surface_switch_tilt_overlay``:
    never flips a HOME pick to AWAY, since the cell's own predicted direction
    is a positive home_cover edge with no measured mirror-direction support.
    Missing climate or forecast data (including a total fetch failure
    upstream) folds into "not flagged", never an error.
    """

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return ForecastColdVisitorResult(base, (), enabled)

    base["game_id"] = base["game_id"].astype(str)
    flags = forecast_cold_visitor_flag_by_game(schedules, forecasts)
    merged = base.merge(flags, on="game_id", how="left")
    merged["forecast_cold_visitor_flag"] = (
        merged["forecast_cold_visitor_flag"].fillna(False).astype(bool)
    )

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    away_pick = merged["home_cover_probability"].lt(0.5)
    flip_mask = eligible & merged["forecast_cold_visitor_flag"] & away_pick

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[ForecastColdVisitorFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        climate_temp = row["climate_temp"]
        forecast_temp_f = row["forecast_temp_f"]
        flips.append(
            ForecastColdVisitorFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                away_team=str(row["away_team"]),
                home_team=str(row["home_team"]),
                climate_temp=float(climate_temp) if pd.notna(climate_temp) else float("nan"),
                forecast_temp_f=float(forecast_temp_f)
                if pd.notna(forecast_temp_f)
                else float("nan"),
                temp_gap_f=(
                    float(climate_temp) - float(forecast_temp_f)
                    if pd.notna(climate_temp) and pd.notna(forecast_temp_f)
                    else float("nan")
                ),
            )
        )

    return ForecastColdVisitorResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: ForecastColdVisitorResult) -> str:
    """Plain-language provenance sentence, mirroring the sibling overlays'.

    Empty when the overlay is off or changed nothing this week. Not
    currently surfaced on the published card -- this overlay is dual-tracked
    only.
    """

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: AWAY -> HOME (gap {flip.temp_gap_f:.0f}F)" for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the forecast "
        "cold-visitor tilt (the model's pick was on an away team whose own climate is "
        "much warmer than this game's Tuesday-noon forecast temperature). "
        f"{detail}. See docs/forecast_weather_screen.md. Prospective evidence only -- "
        "not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_forecast_cold_visitor_tilt_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    registry_root: Path,
    *,
    now: datetime | None = None,
    fetch_bulletin: FetchBulletin = fetch_mos_bulletin,
) -> dict[str, Any]:
    """Append the tilt overlay's picks to the prospective challenger ledger.

    Mirrors ``surface_switch_tilt_overlay.record_surface_switch_tilt_challenger_decisions``
    exactly: this is not a retrained model with its own ``margin-predict``
    artifact -- its "model" IS the active model, transformed post-prediction
    -- so it reads the active model's own synchronized weekly forecast rather
    than searching ``artifacts/margin_predictions/`` by fingerprint, and it
    refuses to record if the active model's live fingerprint no longer
    matches the snapshot this challenger was registered against.

    The live forecast fetch is FAIL-OPEN (see
    :func:`fetch_tuesday_noon_forecast_temps_fail_open`): a network or
    station-mapping failure never raises out of this function, it simply
    yields zero flags for the week.

    ``bet_side`` is always ``"PASS"`` and ``edge`` is always NaN: this
    challenger tracks the tilt's forced-pick (``decision_line``) accuracy
    only, never a fabricated paper-bet edge for the post-tilt side.
    """

    entry = find_challenger(artifacts_root, CHALLENGER_ID)
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {CHALLENGER_ID!r} is registered as {status!r}; only "
            f"{ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError(
            "No synchronized active ATS model is available to record tilt decisions from"
        )
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        raise ValueError("Active ATS model has no linked weekly forecast")
    metadata_path = forecast / "metadata.json"
    card_path = forecast / "recommendations.csv"
    if not metadata_path.is_file() or not card_path.is_file():
        raise ValueError(f"Linked weekly forecast is incomplete: {forecast}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("active_model_id") != active.get("model_id"):
        raise ValueError("Weekly forecast model ID does not match the active model")
    if metadata.get("synchronization_status") != "SYNCHRONIZED":
        raise ValueError("Weekly forecast is not synchronized with an evaluation")

    observed_config = artifact_model_config(metadata)
    declared_fingerprint = config_fingerprint(entry.get("model", {}))
    observed_fingerprint = config_fingerprint(observed_config)
    if declared_fingerprint != observed_fingerprint:
        raise DataContractError(
            f"Challenger {CHALLENGER_ID!r} is registered pinned to configuration "
            f"fingerprint {declared_fingerprint}, but the current active forecast "
            f"{forecast} was produced with {observed_fingerprint}; the active model "
            "changed underneath this tilt -- re-register before recording"
        )

    card = pd.read_csv(card_path)
    required = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
    }
    missing = sorted(required.difference(card.columns))
    if missing:
        raise DataContractError(f"Active forecast card is missing columns: {', '.join(missing)}")
    if card["game_id"].duplicated().any():
        raise DataContractError("Active forecast card contains duplicate games")
    spreads = pd.to_numeric(card["spread_line"], errors="coerce")
    if not np.isfinite(spreads.to_numpy(dtype=float)).all():
        raise DataContractError("Active forecast card has games without a decision spread")
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Active forecast card has games without a kickoff timestamp")

    schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))

    schedules_lookup = schedules[["game_id", "stadium"]].copy()
    schedules_lookup["game_id"] = schedules_lookup["game_id"].astype(str)
    games_for_fetch = card[["game_id", "kickoff"]].copy()
    games_for_fetch["game_id"] = games_for_fetch["game_id"].astype(str)
    games_for_fetch = games_for_fetch.merge(schedules_lookup, on="game_id", how="left")

    station_map_path = registry_root / STATION_MAP_RELATIVE_PATH
    forecasts = fetch_tuesday_noon_forecast_temps_fail_open(
        games_for_fetch, station_map_path, fetch_bulletin=fetch_bulletin
    )

    tilt = apply_forecast_cold_visitor_tilt_overlay(card, schedules, forecasts)
    tilted_card = tilt.overlaid_predictions

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID)]
    already = card["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    keep = pre_kickoff & ~already
    fresh = tilted_card.loc[keep]

    decisions = pd.DataFrame(
        {
            "recorded_at_utc": recorded_at,
            "challenger_id": CHALLENGER_ID,
            "config_fingerprint": observed_fingerprint,
            "source_artifact": forecast.name,
            "source_sha256": sha256_file(card_path),
            "forecast_created_at_utc": pd.to_datetime(
                metadata.get("created_at_utc"), utc=True, errors="coerce"
            ),
            "feature_profile": str(metadata.get("feature_profile")),
            "feature_table_sha256": str(observed_config.get("feature_table_sha256")),
            "game_id": fresh["game_id"].astype(str),
            "season": fresh["season"].astype(int),
            "week": fresh["week"].astype(int),
            "kickoff": kickoffs.loc[fresh.index],
            "away_team": fresh["away_team"].astype(str),
            "home_team": fresh["home_team"].astype(str),
            "pick_side": np.where(
                pd.to_numeric(fresh["home_cover_probability"], errors="coerce").ge(0.5),
                "HOME",
                "AWAY",
            ).astype(str),
            "bet_side": "PASS",
            "decision_home_spread": spreads.loc[fresh.index].astype(float),
            "edge": np.nan,
        }
    )
    if not decisions.empty:
        combined = (
            decisions if existing.empty else pd.concat([existing, decisions], ignore_index=True)
        )
        atomic_parquet(
            combined[list(CHALLENGER_DECISION_COLUMNS)], challenger_ledger_path(artifacts_root)
        )
        ledger_rows = len(combined)
    else:
        ledger_rows = len(existing)

    return {
        "challenger_id": CHALLENGER_ID,
        "season": int(card["season"].iloc[0]),
        "week": int(card["week"].iloc[0]),
        "source_artifact": forecast.name,
        "config_fingerprint": observed_fingerprint,
        "recorded": len(decisions),
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "ledger_rows": int(ledger_rows),
        "flip_count": tilt.flip_count,
        "flipped_game_ids": [flip.game_id for flip in tilt.flips],
        "forecast_fetch_status_counts": (
            forecasts["fetch_status"].value_counts().to_dict() if not forecasts.empty else {}
        ),
    }
