"""Forecast (kickoff-nearest) warm-team-cold-late tilt overlay: a
parameter-free pick-level nudge built on a LIVE kickoff-nearest GFS-MOS
temperature forecast -- a different cutoff/model than
``forecast_cold_visitor_tilt_overlay``'s Tuesday-noon fetch, per this cell's
own registered evidence (see below).

Research chain: ``docs/forecast_weather_screen.md``'s 2026-08-20 extension
("2009-2019 archive backward + a 6-cell family"), cell 1,
``forecast_weather_kn_warm_team_cold_late``. Measured this session (read
directly from ``registry/weak_signals.json``, current after the archive's
full 2009-2025 fetch completed and all 12 kickoff_nearest specs were
re-run with ``--replace``): full window (REG 2009-2025) +0.1697 accuracy
points, week-blocked 95% **[+0.0091, +0.3169]**, ``probability_positive``
0.9800, n_flag=65 of n_total=4,317; pre2020 window (REG 2009-2019) +0.2285
accuracy points, 95% **[+0.0208, +0.4167]**, ``probability_positive`` 0.9848,
n_flag=43 of n_total=2,735. **Both windows' intervals exclude zero** --
per ``docs/forecast_weather_screen.md``'s revised "Wiring recommendations"
section this is the highest-EV cell in the whole forecast-weather family,
"wire first." (An earlier, INTERIM read of this cell, taken before the
archive's 2009-2025 fetch finished, showed a much weaker +0.0771/+0.1265
pts on a checkpoint population that still crossed zero -- superseded, see
the doc's own "Status: complete" section; this module is built against the
FINAL numbers only.)

Neither admissible AGENTS.md closing ground applies in the other direction
either (the interval excluding zero on the POSITIVE side is a strong
confirming read, not a closing ground -- the registry's own mechanical
classifier does not auto-close on that side, and this module does not treat
it as "resolved" or promotable to the real card either). It stays
``unresolved_below_power`` in the registry; wiring it here is a
straightforwardly EV-positive dual-tracked play (``probability_positive``
0.98 > 0.5), not a claim of a proven edge (AGENTS.md "a promotion bar is not
a decision bar").

**Corroborating evidence, a DIFFERENT construction on a DIFFERENT (narrower)
population -- disclosed, not pooled as independent**: this cell's
tuesday_noon sibling, ``forecast_weather_warm_team_cold_late`` (REG
2020-2025 only, model=MEX): +0.3322 accuracy points, 95% [-0.0117, +0.6569],
``probability_positive`` 0.9711 (season-blocked secondary: +0.3322 pts, 95%
[+0.1661, +0.5171], ``probability_positive`` 1.0000). Same mechanism
(warm-metro away team, outdoor, cold late-season forecast), same static
team list and threshold, different cutoff/model and a different (wider)
population for the kickoff_nearest cell measured here -- per
``docs/forecast_weather_screen.md``'s own convention this is reported as
corroboration, never pooled as a second independent evidence point for the
SAME cell.

**Cutoff/model: kickoff_nearest, NOT tuesday_noon -- a deliberate,
doc-directed departure from ``forecast_cold_visitor_tilt_overlay``'s live
fetch.** This cell's registered evidence (above) is built on the
kickoff_nearest archive (model=GFS), not the tuesday_noon archive (model=MEX)
``forecast_cold_visitor_tilt_overlay`` fetches live. Per
``docs/forecast_weather_screen.md``'s wiring recommendation #1: "switch
cutoff to kickoff_nearest per this cell's registered evidence... no new data
source, no new plumbing beyond the cutoff-mode swap." This module therefore
owns its OWN live fetch (below), reusing
``forecast_cold_visitor_tilt_overlay``'s cutoff-agnostic MOS-bulletin
primitives (:func:`nfl_ats.forecast_cold_visitor_tilt_overlay.fetch_mos_bulletin`,
:func:`nfl_ats.forecast_cold_visitor_tilt_overlay.nearest_row`,
:func:`nfl_ats.forecast_cold_visitor_tilt_overlay.candidate_runtimes`)
directly rather than reimplementing them, but computes its OWN cutoff
(kickoff itself -- ``kickoff_nearest_cutoff_utc`` in
``scripts/ingest_forecast_archive.py`` is the identity function on
``kickoff_utc``; ``candidate_runtimes`` floors that to the nearest 00Z/12Z
MOS cycle at-or-before kickoff and walks strictly backward, so point-in-time
discipline -- never a bulletin issued after kickoff -- is enforced by the
walk itself, the same guarantee the tuesday_noon cutoff function provides)
and requests model=GFS, not MEX.

**This module's fetch also captures precipitation probability
(``forecast_precip_prob_pct``, ``p06`` falling back to ``p12``), even though
this cell's own flag does not use it**, so a second kickoff_nearest
challenger needing precip
(``nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay``, cell 4 of
the same 6-cell family, ``docs/forecast_weather_screen.md`` wiring
recommendation #3) can share this module's fetch instead of making its own
outbound HTTP request for the identical (game, station, cutoff) set --
"one fetch, several consumers," ported from
``forecast_cold_visitor_tilt_overlay``'s own wiring note to this new
cutoff/model pair.
:func:`fetch_shared_kickoff_nearest_forecasts_fail_open` is the entry point
a caller wiring both challengers in the same publish call should use.

**FAIL-OPEN, unconditionally.** Any failure fetching or parsing live
forecast data -- a missing/unreadable station map, a stadium absent from it,
a network timeout, a malformed API response, anything -- is caught by
:func:`fetch_kickoff_nearest_forecasts_fail_open`, logged as a
``RuntimeWarning``, and folded into "every game gets no forecast, the flag
is False everywhere" rather than raised. This overlay must never be able to
block a publish. Per-game fetch problems (a single station with no
bulletin, one slow request) already fold into "not flagged" one level down,
inside :func:`fetch_one_game_kickoff_nearest`, mirroring
``scripts/ingest_forecast_archive.py``'s own ``fetch_status`` design.

**Tilt direction: HOME**, matching the cell's own predicted direction
("predicted home_cover edge") and mirroring
``forecast_cold_visitor_tilt_overlay``'s deliberately ASYMMETRIC pattern:
flip AWAY -> HOME only when the flag fires AND the model's own pick is
currently on the away side; never the reverse (no measured direction exists
for a warm-forecast or non-flagged case).

This module is the no-window-cost path, built on the exact pattern of
``forecast_cold_visitor_tilt_overlay.py``, ``surface_switch_tilt_overlay.py``,
and ``interim_hc_first_game_tilt_overlay.py``: a **pick-level,
post-prediction transform** of the active model's own forced pick,
dual-tracked against that same active model in the prospective challenger
ledger (``nfl_ats.prospective_scoring``), at no rotation-registry window
cost and with zero training-time feature changes. **Nothing in this module
is wired into ``publishing.py`` or the production pick path** -- like the
tilt/fade siblings, no owner decision to play this on the real card has been
made; it is dual-tracked only.

:func:`record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions`
writes the overlay's own arm to the prospective challenger ledger so 2026
scores it cleanly, independent of whether it is ever played on the real
card.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.data import DataContractError
from nfl_ats.forecast_cold_visitor_tilt_overlay import (
    DELAY_SECONDS_DEFAULT,
    MAX_LOOKBACK_STEPS_DEFAULT,
    OUTDOOR_ROOFS,
    STATION_MAP_RELATIVE_PATH,
    FetchBulletin,
    MosFetchError,
    candidate_runtimes,
    fetch_mos_bulletin,
    nearest_row,
)
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
CHALLENGER_ID = "forecast_weather_kn_warm_team_cold_late_tilt"

#: Reused verbatim from scripts/nfl_forecast_weather_screen.py /
#: scripts/nfl_weather_battery_screen.py / src/nfl_ats/experiment_runner.py
#: (the warm_team_cold_late mechanism's static warm-winter-metro away-team
#: list -- historical franchise codes (OAK, SD) kept alongside their current
#: ones (LV, LAC) so this set matches regardless of which era's schedules row
#: it is checked against).
WARM_METRO_TEAM_CODES = frozenset(
    {"MIA", "TB", "JAX", "ARI", "SF", "OAK", "LA", "LAC", "SD", "HOU", "DAL", "NO", "LV"}
)
WARM_TEAM_COLD_LATE_TEMP_THRESHOLD_F = 35.0
WARM_TEAM_COLD_LATE_MIN_WEEK = 13

# ---------------------------------------------------------------------------
# Live-fetch constants and helpers -- kickoff_nearest cutoff, model=GFS,
# ported from scripts/ingest_forecast_archive.py's kickoff_nearest path.
# Reuses forecast_cold_visitor_tilt_overlay's cutoff-agnostic MOS-bulletin
# primitives (fetch_mos_bulletin, nearest_row, candidate_runtimes) directly
# rather than reimplementing the HTTP/retry/walk-backward logic; only the
# cutoff computation (kickoff itself, not Tuesday noon) and the extracted
# fields (temp AND precip, not just temp) differ from that module's fetch.
# ---------------------------------------------------------------------------

MOS_MODEL = "GFS"  # kickoff_nearest's model, per MOS_MODEL_BY_CUTOFF_MODE in
# scripts/ingest_forecast_archive.py -- NOT MEX (that is tuesday_noon's model).


def _nearest_row_with_field(
    rows: list[dict[str, Any]], kickoff_utc: pd.Timestamp, field: str
) -> dict[str, Any] | None:
    """Ported verbatim from ``scripts/ingest_forecast_archive.py``: like
    :func:`nearest_row`, restricted to rows where ``field`` is non-null.
    GFS MOS precipitation-probability fields (``p06``/``p12``) are only
    populated on a subset of rows within a bulletin (measured in that
    script's own 2009-2019 extension: every OTHER 3h row, i.e. the
    6h-boundary rows), so the plain :func:`nearest_row` pick (nearest by
    valid time to kickoff, ANY field) frequently lands on a row where
    ``p06``/``p12`` are both null even though a nearby row in the SAME
    already-fetched bulletin has them. This does a second, field-restricted
    nearest-by-valid-time pick over the SAME rows -- no extra HTTP call, no
    relaxation of the point-in-time issuance walk.
    """

    candidates = [row for row in rows if row.get(field) is not None]
    return nearest_row(candidates, kickoff_utc)


def fetch_one_game_kickoff_nearest(
    station: str,
    kickoff_utc: pd.Timestamp,
    *,
    model: str = MOS_MODEL,
    max_lookback_steps: int = MAX_LOOKBACK_STEPS_DEFAULT,
    delay_seconds: float = DELAY_SECONDS_DEFAULT,
    fetch_bulletin: FetchBulletin = fetch_mos_bulletin,
) -> dict[str, Any]:
    """Adapted from ``scripts/ingest_forecast_archive.py::fetch_one_game``,
    trimmed to temperature and precipitation probability (this overlay
    family does not use wind).

    The kickoff_nearest cutoff IS the kickoff itself
    (``kickoff_nearest_cutoff_utc`` in ``scripts/ingest_forecast_archive.py``
    is the identity function) -- :func:`candidate_runtimes` floors that to
    the most recent 00Z/12Z MOS cycle at-or-before kickoff and walks
    strictly backward, so this never selects a bulletin issued after
    kickoff.

    Never raises: a transport failure at any lookback step is folded into
    ``fetch_status="transport_error"`` (per-game fail-open); the outer
    :func:`fetch_kickoff_nearest_forecasts_fail_open` wrapper handles
    whole-batch failures (a missing station map, an entirely unmapped
    stadium).
    """

    cutoff_utc = kickoff_utc
    for runtime_utc in candidate_runtimes(cutoff_utc, max_lookback_steps):
        try:
            rows = fetch_bulletin(station, runtime_utc, model=model)
        except MosFetchError:
            time.sleep(delay_seconds)
            return {
                "forecast_temp_f": None,
                "forecast_precip_prob_pct": None,
                "fetch_status": "transport_error",
            }
        time.sleep(delay_seconds)
        if rows:
            row = nearest_row(rows, kickoff_utc)
            assert row is not None
            tmp = row.get("tmp")
            precip_row = _nearest_row_with_field(
                rows, kickoff_utc, "p06"
            ) or _nearest_row_with_field(rows, kickoff_utc, "p12")
            precip_prob_pct: float | None = None
            if precip_row is not None:
                if precip_row.get("p06") is not None:
                    precip_prob_pct = float(precip_row["p06"])
                elif precip_row.get("p12") is not None:
                    precip_prob_pct = float(precip_row["p12"])
            return {
                "forecast_temp_f": float(tmp) if tmp is not None else None,
                "forecast_precip_prob_pct": precip_prob_pct,
                "fetch_status": "ok",
            }
    return {
        "forecast_temp_f": None,
        "forecast_precip_prob_pct": None,
        "fetch_status": "no_bulletin_within_lookback",
    }


def games_for_forecast_fetch(card: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """The ``(game_id, kickoff, stadium)`` frame the fetch below needs, built
    by joining the active weekly card's own ``kickoff`` onto the newest
    local schedule snapshot's ``stadium`` display string. Mirrors
    ``forecast_cold_visitor_tilt_overlay``'s own helper of the same name
    (that module's tuesday_noon fetch and this module's kickoff_nearest
    fetch each need the identical shape, just consumed by a different
    cutoff/model)."""

    schedules_lookup = schedules[["game_id", "stadium"]].copy()
    schedules_lookup["game_id"] = schedules_lookup["game_id"].astype(str)
    games_for_fetch = card[["game_id", "kickoff"]].copy()
    games_for_fetch["game_id"] = games_for_fetch["game_id"].astype(str)
    return games_for_fetch.merge(schedules_lookup, on="game_id", how="left")


def _fetch_kickoff_nearest_forecasts(
    games: pd.DataFrame,
    station_map_path: Path,
    *,
    model: str = MOS_MODEL,
    max_lookback_steps: int = MAX_LOOKBACK_STEPS_DEFAULT,
    delay_seconds: float = DELAY_SECONDS_DEFAULT,
    fetch_bulletin: FetchBulletin = fetch_mos_bulletin,
) -> pd.DataFrame:
    """Fetch the kickoff-nearest forecast temperature and precipitation
    probability for every game. ``games`` needs ``game_id``, ``stadium``,
    ``kickoff``. Raises on a whole-batch problem (missing station map file,
    a stadium entirely absent from it) -- callers must use
    :func:`fetch_kickoff_nearest_forecasts_fail_open` for the fail-open
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
                    "forecast_precip_prob_pct": None,
                    "fetch_status": "unmappable_international_stadium",
                }
            )
            continue
        kickoff_utc = pd.Timestamp(row.kickoff)  # type: ignore[arg-type]
        if kickoff_utc.tzinfo is None:
            kickoff_utc = kickoff_utc.tz_localize(UTC)
        fetched = fetch_one_game_kickoff_nearest(
            str(row.icao_station),
            kickoff_utc,
            model=model,
            max_lookback_steps=max_lookback_steps,
            delay_seconds=delay_seconds,
            fetch_bulletin=fetch_bulletin,
        )
        rows.append({"game_id": game_id, **fetched})
    return pd.DataFrame(rows)


def fetch_kickoff_nearest_forecasts_fail_open(
    games: pd.DataFrame,
    station_map_path: Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """FAIL-OPEN wrapper around :func:`_fetch_kickoff_nearest_forecasts`.

    Any exception -- a missing/unreadable station map, a stadium absent from
    it, a network timeout, a malformed API response -- is caught, logged as
    a ``RuntimeWarning``, and folded into "every game gets no forecast" (NaN
    ``forecast_temp_f``/``forecast_precip_prob_pct``,
    ``fetch_status="fetch_failed"``) so downstream flag computation naturally
    reads zero flags rather than raising. This overlay must never be able to
    block a publish.
    """

    try:
        return _fetch_kickoff_nearest_forecasts(games, station_map_path, **kwargs)
    except Exception as exc:
        warnings.warn(
            "forecast_weather_kn_warm_team_cold_late_tilt: forecast fetch failed, proceeding "
            f"with zero flags ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(
            {
                "game_id": games["game_id"].astype(str),
                "forecast_temp_f": np.nan,
                "forecast_precip_prob_pct": np.nan,
                "fetch_status": "fetch_failed",
            }
        )


def fetch_shared_kickoff_nearest_forecasts_fail_open(
    artifacts_root: Path,
    data_root: Path,
    registry_root: Path,
    *,
    fetch_bulletin: FetchBulletin = fetch_mos_bulletin,
) -> pd.DataFrame | None:
    """Fetch the live kickoff-nearest GFS-MOS forecast (temp AND precip) ONCE
    for the active model's current weekly card, so a caller wiring more than
    one kickoff_nearest forecast-based challenger in the same
    ``nfl-ats publish-predictions --record-decisions`` call (currently this
    module and ``nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay``
    -- ``docs/forecast_weather_screen.md``'s "Wiring recommendations"
    section: "one fetch, several consumers") can pass the SAME ``forecasts``
    DataFrame to each challenger's own
    ``record_*_challenger_decisions(..., forecasts=...)`` parameter, instead
    of every challenger making its own outbound HTTP request for the
    identical (game, station, cutoff) set.

    Returns ``None`` -- never raises -- on ANY failure in the prologue (no
    active model, no linked forecast, no local schedule snapshot yet, an
    unreadable card): this helper is a pure network-deduplication
    optimization, never a new way for a publish to fail. A caller receiving
    ``None`` should simply omit ``forecasts=`` from each challenger's own
    call and let each recorder fetch (and fail-open) independently. The
    underlying per-game fetch is already fail-open by construction
    (:func:`fetch_kickoff_nearest_forecasts_fail_open`), so a partial
    failure never reaches this function as an exception either.
    """

    try:
        active = load_active_ats_model(artifacts_root)
        if active is None:
            return None
        forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
        if forecast is None:
            return None
        card_path = forecast / "recommendations.csv"
        if not card_path.is_file():
            return None
        card = pd.read_csv(card_path)
        if not {"game_id", "kickoff"}.issubset(card.columns):
            return None
        schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    except (OSError, ValueError, FileNotFoundError, DataContractError):
        return None

    games_for_fetch = games_for_forecast_fetch(card, schedules)
    station_map_path = registry_root / STATION_MAP_RELATIVE_PATH
    return fetch_kickoff_nearest_forecasts_fail_open(
        games_for_fetch, station_map_path, fetch_bulletin=fetch_bulletin
    )


# ---------------------------------------------------------------------------
# Flag + pick-level transform
# ---------------------------------------------------------------------------


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.astype(str).map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def warm_team_cold_late_flag_by_game(
    schedules: pd.DataFrame, forecasts: pd.DataFrame
) -> pd.DataFrame:
    """One row per REG-season ``game_id``: ``warm_team_cold_late_flag``,
    ``forecast_temp_f``.

    ``forecasts`` needs ``game_id``, ``forecast_temp_f`` (one row per game;
    extra columns, including a missing/NaN temp for any game, are fine --
    they fold into "not flagged"). Mirrors the 6-cell family's cell 1
    definition (``docs/forecast_weather_screen.md``): away team in the
    static warm-winter-metro list AND outdoor AND kickoff-nearest forecast
    temp<=35F AND week>=13. Every input is either a static team-list
    membership or this game's own roof/week/forecast, never an aggregate
    over other games, so no pregame-safety adaptation is needed.
    """

    required_forecast = {"game_id", "forecast_temp_f"}
    missing = sorted(required_forecast.difference(forecasts.columns))
    if missing:
        raise DataContractError(f"forecasts is missing columns: {', '.join(missing)}")
    required_schedule = {"game_id", "game_type", "week", "away_team", "roof"}
    missing_schedule = sorted(required_schedule.difference(schedules.columns))
    if missing_schedule:
        raise DataContractError(
            "schedules is missing columns for warm-team cold-late tracking: "
            f"{', '.join(missing_schedule)}"
        )

    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["game_id"] = reg["game_id"].astype(str)
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["outdoor"] = reg["roof"].isin(OUTDOOR_ROOFS)
    reg["week"] = pd.to_numeric(reg["week"], errors="coerce")

    forecasts = forecasts.copy()
    forecasts["game_id"] = forecasts["game_id"].astype(str)
    forecasts["forecast_temp_f"] = pd.to_numeric(forecasts["forecast_temp_f"], errors="coerce")

    frame = reg[["game_id", "away_team", "outdoor", "week"]].merge(
        forecasts[["game_id", "forecast_temp_f"]].drop_duplicates(subset="game_id"),
        on="game_id",
        how="left",
    )

    flag = (
        frame["away_team"].isin(WARM_METRO_TEAM_CODES)
        & frame["outdoor"]
        & frame["forecast_temp_f"].le(WARM_TEAM_COLD_LATE_TEMP_THRESHOLD_F)
        & frame["week"].ge(WARM_TEAM_COLD_LATE_MIN_WEEK)
    )
    frame["warm_team_cold_late_flag"] = flag.fillna(False).astype(bool)
    return frame[["game_id", "warm_team_cold_late_flag", "forecast_temp_f"]]


@dataclass(frozen=True)
class WarmTeamColdLateFlip:
    """One game the overlay flipped, for provenance and ledger recording."""

    game_id: str
    matchup: str
    away_team: str
    home_team: str
    forecast_temp_f: float


@dataclass(frozen=True)
class WarmTeamColdLateResult:
    """The overlay's effect on one week's card.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on flipped rows -- every other column stays
    byte-identical, mirroring ``forecast_cold_visitor_tilt_overlay.ForecastColdVisitorResult``.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[WarmTeamColdLateFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_warm_team_cold_late_tilt_overlay(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    enabled: bool = True,
) -> WarmTeamColdLateResult:
    """Flip the forced pick from AWAY to HOME wherever the flag fires.

    A game flips only when ALL hold:

    * ``game_type == "REG"`` when that column is present;
    * :func:`warm_team_cold_late_flag_by_game` fires for the game (away team
      in the static warm-winter-metro list, outdoor, kickoff-nearest
      forecast temp<=35F, week>=13); and
    * the model's own pick (``home_cover_probability >= 0.5`` picks home) is
      currently on the AWAY side.

    **Deliberately ASYMMETRIC**, mirroring
    ``forecast_cold_visitor_tilt_overlay``: never flips a HOME pick to AWAY,
    since the cell's own predicted direction is a positive home_cover edge
    with no measured mirror-direction support. Missing forecast data
    (including a total fetch failure upstream) folds into "not flagged",
    never an error.
    """

    required = {"game_id", "season", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled:
        return WarmTeamColdLateResult(base, (), enabled)

    base["game_id"] = base["game_id"].astype(str)
    flags = warm_team_cold_late_flag_by_game(schedules, forecasts)
    merged = base.merge(flags, on="game_id", how="left")
    merged["warm_team_cold_late_flag"] = (
        merged["warm_team_cold_late_flag"].fillna(False).astype(bool)
    )

    eligible = pd.Series(True, index=merged.index)
    if "game_type" in merged.columns:
        eligible &= merged["game_type"].astype(str).eq("REG")

    away_pick = merged["home_cover_probability"].lt(0.5)
    flip_mask = eligible & merged["warm_team_cold_late_flag"] & away_pick

    overlaid = base.copy()
    overlaid.loc[flip_mask, "home_cover_probability"] = (
        1.0 - overlaid.loc[flip_mask, "home_cover_probability"]
    )

    flips: list[WarmTeamColdLateFlip] = []
    for _, row in merged.loc[flip_mask].iterrows():
        forecast_temp_f = row["forecast_temp_f"]
        flips.append(
            WarmTeamColdLateFlip(
                game_id=str(row["game_id"]),
                matchup=f"{row['away_team']} at {row['home_team']}",
                away_team=str(row["away_team"]),
                home_team=str(row["home_team"]),
                forecast_temp_f=float(forecast_temp_f)
                if pd.notna(forecast_temp_f)
                else float("nan"),
            )
        )

    return WarmTeamColdLateResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: WarmTeamColdLateResult) -> str:
    """Plain-language provenance sentence, mirroring the sibling overlays'.

    Empty when the overlay is off or changed nothing this week. Not
    currently surfaced on the published card -- this overlay is dual-tracked
    only.
    """

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: AWAY -> HOME (forecast {flip.forecast_temp_f:.0f}F)"
        for flip in result.flips
    )
    return (
        f"**Tilt applied: {result.flip_count} pick{plural} flipped** by the forecast "
        "(kickoff-nearest) warm-team-cold-late tilt (the model's pick was on an away "
        "team from a warm-winter metro, playing outdoors in week 13 or later with a "
        "kickoff-nearest forecast at or below 35F). "
        f"{detail}. See docs/forecast_weather_screen.md. Prospective evidence only -- "
        "not applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    registry_root: Path,
    *,
    now: datetime | None = None,
    fetch_bulletin: FetchBulletin = fetch_mos_bulletin,
    forecasts: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Append the tilt overlay's picks to the prospective challenger ledger.

    Mirrors
    ``forecast_cold_visitor_tilt_overlay.record_forecast_cold_visitor_tilt_challenger_decisions``
    exactly: this is not a retrained model with its own ``margin-predict``
    artifact -- its "model" IS the active model, transformed post-prediction
    -- so it reads the active model's own synchronized weekly forecast rather
    than searching ``artifacts/margin_predictions/`` by fingerprint, and it
    refuses to record if the active model's live fingerprint no longer
    matches the snapshot this challenger was registered against.

    ``forecasts``, when supplied, is used AS-IS instead of fetching again --
    see :func:`fetch_shared_kickoff_nearest_forecasts_fail_open` for the
    "one fetch, several consumers" path this parameter exists for (shared
    with ``nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay``).
    ``None`` (the default) fetches for itself, using the exact same
    fail-open live path (:func:`fetch_kickoff_nearest_forecasts_fail_open`):
    a network or station-mapping failure never raises out of this function,
    it simply yields zero flags for the week.

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

    if forecasts is None:
        games_for_fetch = games_for_forecast_fetch(card, schedules)
        station_map_path = registry_root / STATION_MAP_RELATIVE_PATH
        forecasts = fetch_kickoff_nearest_forecasts_fail_open(
            games_for_fetch, station_map_path, fetch_bulletin=fetch_bulletin
        )

    tilt = apply_warm_team_cold_late_tilt_overlay(card, schedules, forecasts)
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
