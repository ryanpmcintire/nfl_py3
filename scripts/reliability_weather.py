"""Split-half reliability for the 33 weather-group registry cells (ORCH-D).

**What these 33 cells are, and why one estimator can't be applied uniformly.**
The group spans four builders with genuinely different construct shapes:

* ``weather_battery_*`` (8) / ``weather_followup_*`` (5) / the 4 tuesday_noon
  ``forecast_weather_*`` cells / ``wxtot_*`` (4) -- game-level subset flags
  from ``scripts/nfl_weather_battery_screen.py``, ``nfl_weather_followup_screen.py``,
  ``nfl_forecast_weather_screen.py`` and ``weather_total_interaction_screen.py``.
  Each flag THRESHOLDS a continuous quantity (actual/forecast temp or wind,
  or forecast precip probability) that is fundamentally a property of the
  home venue that week -- so the construct behind the cell is that
  continuous quantity's own venue-season split-half reliability
  (``METHOD_VENUE``), not the flag itself. Several cells share one parent
  quantity (e.g. every actual-temp threshold cell inherits the SAME temp
  reliability number, exactly the precedent the six ``attention_battery_*``
  cells already set: "a battery's cells inherit the reliability of the trait
  they are built on", read: docs/reliability_sweep_20260901.md and
  scripts/reliability_graph_team_stat.py's docstring). Two battery cells
  (surface-switch, high-altitude-road) have NO continuous parent -- pure
  structural/categorical subsets -- and get ``METHOD_EXPOSURE`` instead.

* ``forecast_weather_kn_*`` (10) -- NOT built by ``nfl_forecast_weather_screen.py``;
  read 2026-09-01, their construction lives in
  ``src/nfl_ats/experiment_runner.py``'s ``FLAG_BUILDERS`` registry (imported
  here, never reimplemented) and each is a COMPOUND flag over 2-4 conditions
  (a categorical roof/team-list membership AND one or two continuous
  thresholds AND sometimes a week/market-total gate). The runner's own code
  (``run_subset_bias_experiment``, read: experiment_runner.py:3565-3592)
  explicitly refuses ``reliability_method="split_half"`` for exactly these 5
  flag_builders with the reason "no persistent per-entity trait to
  split-half" -- the runner's authors already made this same METHOD_EXPOSURE
  call before this sweep existed. This script measures each kn cell's own
  flag EXPOSURE reliability via ``reliability_lib.game_flag_to_team_week`` on
  the builder's own ``SubsetBiasConstruct.table``/``.flag``, at the entry's
  own season window (``full``=2009-2025, ``pre2020``=2009-2019).

* ``weak_stack_v4_*`` (2) -- a ridge-model paired-opener-accuracy delta from
  stacking six CONTINUOUS forecast-weather columns
  (``nfl_ats.forecast_weather_features.FORECAST_WEATHER_COLUMNS``) onto
  PRODUCTION ``weak_stack``. **Correction to the group brief**: these two are
  genuinely pregame forecast features, NOT the deliberately-leaked oracle
  (that is ``weak_stack_oracle_weather_*``, built from
  ``OBSERVED_WEATHER_FEATURE_COLUMNS`` / actual weather via
  ``scripts/weak_stack_oracle_weather_eval.py``; read 2026-09-01,
  ``src/nfl_ats/constants.py:327-333`` vs.
  ``src/nfl_ats/forecast_weather_features.py:40-47`` -- no
  ``weak_stack_oracle_weather_*`` entries are in this group's 33). A joint
  6-feature ridge-model delta does not reduce to one thresholded quantity's
  reliability the way a subset flag does, so nothing is RECORDED for these
  two; their three substantive continuous inputs (forecast temp/wind/precip)
  are measured as ``METHOD_VENUE`` traits and reported only, per the group
  brief's "measure their INPUTS' reliability if you can ... no reliability
  read from them is evidence about a playable rule".

**Binding taxonomy (verbatim, AGENTS.md / CLAUDE.md).** An interval or CI
that contains zero is NEVER grounds to reject, fail, or close an experiment.
Only two grounds ever close a line of work: (1) refuted mechanism -- a
RESOLVED wrong sign (whole interval on the wrong side of zero) or zero
split-half reliability; (2) bounded by a positive control proven able to
detect an effect that size. Everything else is ``unresolved_below_power``;
report ``probability_positive``, never "contains zero". Reliability is one
of the two closing grounds, so a measured reliability near zero is a real
terminal finding ONLY when the measurement is itself sound -- but this
script NEVER closes, reclassifies, or proposes a ``closing_ground``: it
measures and records the reliability field only, exactly like
``scripts/reliability_graph_team_stat.py``. Within-week correlation is ZERO.

A construct with too few usable units, or one whose measured value is an
artifact of a near-constant column (huge |r| driven by 1-2 structurally
always/never-flagged units, flipping sign across season windows -- the
hazard the orchestrator's own smoke run already hit), is reported as
``not_informative_near_constant`` / unmeasured and never recorded.

Writes ``artifacts/reliability_sweep/weather/<UTC stamp>/results.json`` and
prints the ``set-reliability`` commands it would run (recording itself goes
through the separate locked CLI, never from inside this script).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.append(str(REPO / "scripts"))

import nfl_forecast_weather_screen as fw_screen  # noqa: E402
import nfl_weather_battery_screen as battery_screen  # noqa: E402
import nfl_weather_followup_screen as followup_screen  # noqa: E402
import reliability_lib as rlib  # noqa: E402
import weather_total_interaction_screen as wxtot_screen  # noqa: E402

from nfl_ats.experiment_runner import FLAG_BUILDERS  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.weak_signals import default_registry_path, load_registry  # noqa: E402

OUTDOOR_ROOFS = frozenset({"outdoors", "open"})

KN_ARCHIVE_REL = "data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet"
KN_ARCHIVE = REPO / KN_ARCHIVE_REL
TN_ARCHIVE = REPO / "data/raw/forecast_archive/full_2020_2025/forecasts.parquet"

#: The 33 registry cells in this group (read: <scratchpad>/orchD_manifest.json,
#: key groups.weather.entries, 2026-09-01). Hardcoded rather than read from a
#: scratchpad path at runtime so the script has no dependency on a temp file.
ENTRY_NAMES: tuple[str, ...] = (
    "forecast_weather_dome_team_outdoors_cold",
    "forecast_weather_high_wind_outdoor",
    "forecast_weather_kn_dome_cold_windy_full",
    "forecast_weather_kn_dome_cold_windy_pre2020",
    "forecast_weather_kn_precip_high_total_full",
    "forecast_weather_kn_precip_high_total_pre2020",
    "forecast_weather_kn_temp_gap_cold_visitor_full",
    "forecast_weather_kn_temp_gap_cold_visitor_pre2020",
    "forecast_weather_kn_temp_swing_prior_week_full",
    "forecast_weather_kn_temp_swing_prior_week_pre2020",
    "forecast_weather_kn_warm_team_cold_late_full",
    "forecast_weather_kn_warm_team_cold_late_pre2020",
    "forecast_weather_temp_gap_cold_visitor",
    "forecast_weather_warm_team_cold_late",
    "weak_stack_v4_forecast_weather_opener_probability_rule",
    "weak_stack_v4_forecast_weather_opener_sign_rule",
    "weather_battery_dome_team_outdoors_cold",
    "weather_battery_extreme_cold",
    "weather_battery_high_altitude_road",
    "weather_battery_high_wind_outdoor",
    "weather_battery_high_wind_road_favorite",
    "weather_battery_surface_switch_grass_to_turf",
    "weather_battery_thursday_outdoor_cold",
    "weather_battery_warm_team_cold_late",
    "weather_followup_high_wind_pass_heavy_visitor",
    "weather_followup_rest_disadvantage_cold",
    "weather_followup_surface_switch_x_outdoor_cold",
    "weather_followup_temp_gap_cold_visitor",
    "weather_followup_wind_gap_visitor",
    "wxtot_cold35_top_total",
    "wxtot_precip60_top_total",
    "wxtot_wind15_bottom_total",
    "wxtot_wind15_top_total",
)

#: kn flag_builder name for each *_full/_pre2020 pair (read:
#: src/nfl_ats/experiment_runner.py FLAG_BUILDERS keys, verbatim match to the
#: registry experiment_specs' construct.flag_builder).
KN_BUILDER_FOR: dict[str, str] = {
    "forecast_weather_kn_dome_cold_windy_full": "forecast_weather_kn_dome_cold_windy",
    "forecast_weather_kn_dome_cold_windy_pre2020": "forecast_weather_kn_dome_cold_windy",
    "forecast_weather_kn_precip_high_total_full": "forecast_weather_kn_precip_high_total",
    "forecast_weather_kn_precip_high_total_pre2020": "forecast_weather_kn_precip_high_total",
    "forecast_weather_kn_temp_gap_cold_visitor_full": "forecast_weather_kn_temp_gap_cold_visitor",
    "forecast_weather_kn_temp_gap_cold_visitor_pre2020": (
        "forecast_weather_kn_temp_gap_cold_visitor"
    ),
    "forecast_weather_kn_temp_swing_prior_week_full": "forecast_weather_kn_temp_swing_prior_week",
    "forecast_weather_kn_temp_swing_prior_week_pre2020": (
        "forecast_weather_kn_temp_swing_prior_week"
    ),
    "forecast_weather_kn_warm_team_cold_late_full": "forecast_weather_kn_warm_team_cold_late",
    "forecast_weather_kn_warm_team_cold_late_pre2020": "forecast_weather_kn_warm_team_cold_late",
}

#: name -> (family, method_tag, quantity_key | None, reason)
#: quantity_key indexes into the QUANTITIES long-frame cache below; None means
#: "use the cell's own flag" (EXPOSURE) or "skip" (weak_stack_v4).
FAMILY = "family"
METHOD_TAG = "method_tag"
QUANTITY = "quantity"
REASON = "reason"

ENTRY_SPECS: dict[str, dict[str, Any]] = {
    # --- weather_battery_* (scripts/nfl_weather_battery_screen.py) ---
    "weather_battery_high_wind_outdoor": {
        FAMILY: "battery",
        METHOD_TAG: "VENUE",
        QUANTITY: "wind_actual",
        REASON: (
            "cell thresholds actual game-time wind (wind>=15mph, outdoor); parent quantity is "
            "the venue-week actual wind speed at the home team's stadium"
        ),
    },
    "weather_battery_high_wind_road_favorite": {
        FAMILY: "battery",
        METHOD_TAG: "VENUE",
        QUANTITY: "wind_actual",
        REASON: (
            "cell thresholds actual game-time wind (wind>=15mph, outdoor) AND away-favorite "
            "(a market fact, not measured); parent continuous quantity is the same venue-week "
            "actual wind trait as weather_battery_high_wind_outdoor"
        ),
    },
    "weather_battery_extreme_cold": {
        FAMILY: "battery",
        METHOD_TAG: "VENUE",
        QUANTITY: "temp_actual",
        REASON: (
            "cell thresholds actual game-time temp (temp<=25F, outdoor); parent quantity is the "
            "venue-week actual temperature at the home team's stadium"
        ),
    },
    "weather_battery_dome_team_outdoors_cold": {
        FAMILY: "battery",
        METHOD_TAG: "VENUE",
        QUANTITY: "temp_actual",
        REASON: (
            "cell thresholds actual game-time temp (temp<=40F, outdoor) compounded with a "
            "structural away-team roof-mismatch condition (not a continuous quantity, not "
            "separately measured); parent continuous quantity is the same venue-week actual "
            "temperature trait as weather_battery_extreme_cold"
        ),
    },
    "weather_battery_warm_team_cold_late": {
        FAMILY: "battery",
        METHOD_TAG: "VENUE",
        QUANTITY: "temp_actual",
        REASON: (
            "cell thresholds actual game-time temp (temp<=35F, outdoor, week>=13) compounded "
            "with a static warm-metro team-list membership (not continuous); parent continuous "
            "quantity is the same venue-week actual temperature trait as "
            "weather_battery_extreme_cold"
        ),
    },
    "weather_battery_surface_switch_grass_to_turf": {
        FAMILY: "battery",
        METHOD_TAG: "EXPOSURE",
        QUANTITY: None,
        REASON: (
            "purely categorical roof/surface-mismatch flag (away team's modal home surface "
            "grass, this game's surface turf) with no continuous parent quantity; measures the "
            "flag's own team-week exposure-rate reliability"
        ),
    },
    "weather_battery_high_altitude_road": {
        FAMILY: "battery",
        METHOD_TAG: "EXPOSURE",
        QUANTITY: None,
        REASON: (
            "purely categorical venue-identity flag (game at Denver or a Mexico City/Azteca "
            "neutral site) with no continuous parent quantity; measures the flag's own team-week "
            "exposure-rate reliability"
        ),
    },
    "weather_battery_thursday_outdoor_cold": {
        FAMILY: "battery",
        METHOD_TAG: "VENUE",
        QUANTITY: "temp_actual",
        REASON: (
            "cell thresholds actual game-time temp (temp<=35F, outdoor) compounded with a "
            "Thursday-weekday schedule fact (not continuous); parent continuous quantity is the "
            "same venue-week actual temperature trait as weather_battery_extreme_cold"
        ),
    },
    # --- weather_followup_* (scripts/nfl_weather_followup_screen.py) ---
    "weather_followup_temp_gap_cold_visitor": {
        FAMILY: "followup",
        METHOD_TAG: "VENUE",
        QUANTITY: "temp_actual",
        REASON: (
            "cell thresholds (away team's own same-season climatological-normal temp) minus "
            "(this game's actual temp) >= 25F; the focal-game half of that gap is the venue-week "
            "actual temperature trait, same as weather_battery_extreme_cold"
        ),
    },
    "weather_followup_wind_gap_visitor": {
        FAMILY: "followup",
        METHOD_TAG: "VENUE",
        QUANTITY: "wind_actual",
        REASON: (
            "cell thresholds (away team's own same-season climatological-normal wind<=8mph) AND "
            "(this game's actual wind>=15mph); the focal-game half is the venue-week actual wind "
            "trait, same as weather_battery_high_wind_outdoor"
        ),
    },
    "weather_followup_high_wind_pass_heavy_visitor": {
        FAMILY: "followup",
        METHOD_TAG: "VENUE",
        QUANTITY: "wind_actual",
        REASON: (
            "cell thresholds actual game-time wind (wind>=15mph, outdoor) AND away team's prior-"
            "season pass-rate above median; the prior-pass-rate half IS a persistent per-team-"
            "season trait but is season-constant (does not vary week to week within a season), "
            "which would make the odd/even-WEEK split-half harness return a trivial, "
            "uninformative near-1.0 rather than a real measurement -- so the wind half (the "
            "genuinely week-varying continuous quantity this cell also thresholds) is measured "
            "instead, reusing weather_battery_high_wind_outdoor's venue-week actual wind trait"
        ),
    },
    "weather_followup_rest_disadvantage_cold": {
        FAMILY: "followup",
        METHOD_TAG: "VENUE",
        QUANTITY: "temp_actual",
        REASON: (
            "cell thresholds actual game-time temp (temp<=35F, outdoor) compounded with a rest-"
            "differential schedule fact (not continuous); parent continuous quantity is the same "
            "venue-week actual temperature trait as weather_battery_extreme_cold"
        ),
    },
    "weather_followup_surface_switch_x_outdoor_cold": {
        FAMILY: "followup",
        METHOD_TAG: "VENUE",
        QUANTITY: "temp_actual",
        REASON: (
            "cell thresholds actual game-time temp (temp<=45F, outdoor) compounded with a "
            "structural surface-mismatch condition (not continuous, same as "
            "weather_battery_surface_switch_grass_to_turf); parent continuous quantity is the "
            "same venue-week actual temperature trait as weather_battery_extreme_cold"
        ),
    },
    # --- forecast_weather_* tuesday_noon (scripts/nfl_forecast_weather_screen.py) ---
    "forecast_weather_high_wind_outdoor": {
        FAMILY: "forecast_tn",
        METHOD_TAG: "VENUE",
        QUANTITY: "forecast_wind_tn",
        REASON: (
            "cell thresholds Tuesday-noon GFS-MOS forecast wind (forecast_wind_mph>=15mph, "
            "outdoor); parent quantity is the venue-week forecast wind trait on the tuesday_noon "
            "archive"
        ),
    },
    "forecast_weather_dome_team_outdoors_cold": {
        FAMILY: "forecast_tn",
        METHOD_TAG: "VENUE",
        QUANTITY: "forecast_temp_tn",
        REASON: (
            "cell thresholds Tuesday-noon GFS-MOS forecast temp (forecast_temp_f<=40F, outdoor) "
            "compounded with a structural roof-mismatch condition (not continuous); parent "
            "continuous quantity is the venue-week forecast temp trait on the tuesday_noon archive"
        ),
    },
    "forecast_weather_warm_team_cold_late": {
        FAMILY: "forecast_tn",
        METHOD_TAG: "VENUE",
        QUANTITY: "forecast_temp_tn",
        REASON: (
            "cell thresholds Tuesday-noon GFS-MOS forecast temp (forecast_temp_f<=35F, outdoor, "
            "week>=13) compounded with a warm-metro team-list membership (not continuous); "
            "parent continuous quantity is the same venue-week forecast temp trait as "
            "forecast_weather_dome_team_outdoors_cold"
        ),
    },
    "forecast_weather_temp_gap_cold_visitor": {
        FAMILY: "forecast_tn",
        METHOD_TAG: "VENUE",
        QUANTITY: "forecast_temp_tn",
        REASON: (
            "cell thresholds (away team's own actual-weather climatological-normal temp) minus "
            "(this game's Tuesday-noon forecast temp) >= 25F; the focal-game half is the same "
            "venue-week forecast temp trait as forecast_weather_dome_team_outdoors_cold"
        ),
    },
    # --- wxtot_* (scripts/weather_total_interaction_screen.py) ---
    "wxtot_wind15_top_total": {
        FAMILY: "wxtot",
        METHOD_TAG: "VENUE",
        QUANTITY: "wind_actual",
        REASON: (
            "cell thresholds actual game-time wind (wind>=15mph, outdoor) AND top-tercile total "
            "(a market fact, not measured); parent continuous quantity is the same venue-week "
            "actual wind trait as weather_battery_high_wind_outdoor"
        ),
    },
    "wxtot_cold35_top_total": {
        FAMILY: "wxtot",
        METHOD_TAG: "VENUE",
        QUANTITY: "temp_actual",
        REASON: (
            "cell thresholds actual game-time temp (temp<=35F, outdoor) AND top-tercile total "
            "(a market fact, not measured); parent continuous quantity is the same venue-week "
            "actual temperature trait as weather_battery_extreme_cold"
        ),
    },
    "wxtot_wind15_bottom_total": {
        FAMILY: "wxtot",
        METHOD_TAG: "VENUE",
        QUANTITY: "wind_actual",
        REASON: (
            "CONTROL cell: same actual wind threshold as wxtot_wind15_top_total but bottom-"
            "tercile total; parent continuous quantity is the same venue-week actual wind trait"
        ),
    },
    "wxtot_precip60_top_total": {
        FAMILY: "wxtot",
        METHOD_TAG: "VENUE",
        QUANTITY: "forecast_precip_kn",
        REASON: (
            "cell thresholds kickoff-nearest GFS-MOS forecast precip probability (>=60%, "
            "outdoor) AND top-tercile total (a market fact, not measured); parent continuous "
            "quantity is the venue-week forecast precip-probability trait on the kickoff_nearest "
            "archive"
        ),
    },
}
# kn entries added programmatically below (10, share one reason template).
for _kn_name, _builder_name in KN_BUILDER_FOR.items():
    ENTRY_SPECS[_kn_name] = {
        FAMILY: "forecast_kn",
        METHOD_TAG: "EXPOSURE",
        QUANTITY: _builder_name,
        REASON: (
            f"compound per-game flag (flag_builder={_builder_name!r}, "
            "src/nfl_ats/experiment_runner.py FLAG_BUILDERS) over 2-4 conditions spanning "
            "categorical/structural and continuous inputs with no single stable per-unit parent "
            "quantity; the runner's own run_subset_bias_experiment already refuses "
            "reliability_method='split_half' for this exact flag_builder "
            "(experiment_runner.py:3565-3592, 'no persistent per-entity trait to split-half'). "
            "Measures the flag's own team-week exposure-rate reliability instead."
        ),
    }
for _v4_name in (
    "weak_stack_v4_forecast_weather_opener_probability_rule",
    "weak_stack_v4_forecast_weather_opener_sign_rule",
):
    ENTRY_SPECS[_v4_name] = {
        FAMILY: "weak_stack_v4",
        METHOD_TAG: "SKIP",
        QUANTITY: None,
        REASON: (
            "joint 6-feature ridge-model paired-opener-accuracy delta (production weak_stack + "
            "FORECAST_WEATHER_COLUMNS), not a single thresholded quantity; no one reliability "
            "number represents it. Its three substantive continuous inputs (forecast temp/wind/"
            "precip) are measured as venue traits and reported, never recorded, per the group "
            "brief."
        ),
    }

assert set(ENTRY_SPECS) == set(ENTRY_NAMES), set(ENTRY_SPECS) ^ set(ENTRY_NAMES)


def target_entries() -> dict[str, dict[str, Any]]:
    registry = load_registry(default_registry_path(REPO / "registry"))
    out: dict[str, dict[str, Any]] = {}
    for name in ENTRY_NAMES:
        signal = registry.signals.get(name)
        if signal is None:
            raise SystemExit(f"registry entry not found: {name!r}")
        out[name] = {
            "seasons": (int(signal.seasons[0]), int(signal.seasons[1])),
            "reliability": signal.reliability,
            "effect": signal.effect,
            "classification": signal.classification,
            "source": signal.source,
        }
    return out


# ---------------------------------------------------------------------------
# Shared continuous-quantity long frames (built once, sliced per entry season
# window inside measure_reliability -- avoids recomputing the same trait's
# reliability once per sibling cell).
# ---------------------------------------------------------------------------


def _venue_long(df: pd.DataFrame, metric_col: str, *, outdoor_col: str = "outdoor") -> pd.DataFrame:
    """One row per game: home_team as the venue unit, season, week, metric.

    ``home_team`` is used as the venue unit rather than the schedules'
    ``stadium`` column -- measured (this script, probe run 2026-09-01):
    REG 2009-2025 schedules.parquet carries 82 distinct ``stadium`` strings
    for 35 distinct ``home_team`` codes, and 31 of 35 home_teams have MORE
    THAN ONE distinct stadium string on file (name-variant/sponsor-renamed
    strings, not real venue changes for most). Splitting on ``stadium``
    would fragment true venue-seasons on cosmetic string noise; ``home_team``
    is the stable venue-identity unit and is what every other builder in
    this group (``away_modal_roof``/``away_modal_surface``/``climate_temp``)
    already groups by.
    """

    out = df.loc[:, ["home_team", "season", "week"]].rename(columns={"home_team": "venue"})
    metric = pd.to_numeric(df[metric_col], errors="coerce")
    if outdoor_col in df.columns:
        metric = metric.where(df[outdoor_col].astype(bool))
    out[metric_col] = metric
    return out


def build_quantities() -> dict[str, pd.DataFrame]:
    """Build every shared venue-week continuous-quantity long frame once."""

    quantities: dict[str, pd.DataFrame] = {}

    battery_df = battery_screen.load_population(battery_screen.DEFAULT_SCHEDULES)
    quantities["temp_actual"] = _venue_long(battery_df, "temp")
    quantities["wind_actual"] = _venue_long(battery_df, "wind")
    quantities["_battery_df"] = battery_df  # kept for build_cells() reuse below

    followup_df = followup_screen.load_population(
        followup_screen.default_schedules(), followup_screen.DEFAULT_TEAM_STATS
    )
    quantities["_followup_df"] = followup_df

    fw_df = fw_screen.load_population(fw_screen.default_schedules(), fw_screen.DEFAULT_FORECASTS)
    quantities["_forecast_tn_df"] = fw_df

    tn = pd.read_parquet(
        TN_ARCHIVE,
        columns=[
            "game_id",
            "season",
            "week",
            "home_team",
            "roof",
            "forecast_temp_f",
            "forecast_wind_mph",
        ],
    )
    tn["outdoor"] = tn["roof"].isin(OUTDOOR_ROOFS)
    quantities["forecast_temp_tn"] = _venue_long(tn, "forecast_temp_f")
    quantities["forecast_wind_tn"] = _venue_long(tn, "forecast_wind_mph")

    kn = pd.read_parquet(
        KN_ARCHIVE,
        columns=[
            "game_id",
            "season",
            "week",
            "home_team",
            "roof",
            "forecast_temp_f",
            "forecast_wind_mph",
            "forecast_precip_prob_pct",
        ],
    )
    kn["outdoor"] = kn["roof"].isin(OUTDOOR_ROOFS)
    quantities["forecast_temp_kn"] = _venue_long(kn, "forecast_temp_f")
    quantities["forecast_wind_kn"] = _venue_long(kn, "forecast_wind_mph")
    quantities["forecast_precip_kn"] = _venue_long(kn, "forecast_precip_prob_pct")

    wxtot_df = wxtot_screen.load_population(
        wxtot_screen.default_schedules(), wxtot_screen.DEFAULT_FORECASTS
    )
    quantities["_wxtot_df"] = wxtot_df

    return quantities


_METRIC_COL = {
    "temp_actual": "temp",
    "wind_actual": "wind",
    "forecast_temp_tn": "forecast_temp_f",
    "forecast_wind_tn": "forecast_wind_mph",
    "forecast_temp_kn": "forecast_temp_f",
    "forecast_wind_kn": "forecast_wind_mph",
    "forecast_precip_kn": "forecast_precip_prob_pct",
}


def measure_venue(
    quantities: dict[str, pd.DataFrame], quantity_key: str, seasons: tuple[int, int], *, n_boot: int
) -> dict[str, Any]:
    long = quantities[quantity_key].rename(columns={"venue": "team_id"})
    return rlib.measure_reliability(
        long,
        _METRIC_COL[quantity_key],
        method=rlib.METHOD_VENUE,
        unit_col="team_id",
        seasons=seasons,
        n_boot=n_boot,
    )


def near_constant_check(
    long: pd.DataFrame, metric_col: str, unit_col: str = "team_id"
) -> dict[str, Any]:
    """Flag the hazard the orchestrator's own smoke run hit: a huge |r| driven
    by 1-2 structurally always/never-flagged units rather than real trait
    stability. Reports the count of units that ever carry a nonzero/non-null
    value, not a verdict -- callers decide whether to withhold recording.
    """

    values = pd.to_numeric(long[metric_col], errors="coerce")
    per_unit = values.groupby(long[unit_col]).mean(numeric_only=True)
    n_units_total = int(per_unit.shape[0])
    n_units_active = int((per_unit.fillna(0.0) > 0).sum())
    return {
        "n_units_total": n_units_total,
        "n_units_with_any_positive_value": n_units_active,
        "near_constant": n_units_total >= 5 and n_units_active <= 2,
    }


def dominant_unit_check(
    long: pd.DataFrame,
    metric_col: str,
    seasons: tuple[int, int],
    *,
    unit_col: str,
    method: str,
    baseline_reliability: float | None,
    n_boot: int = 500,
) -> dict[str, Any] | None:
    """Leave-one-unit-out robustness check for the near-constant hazard.

    ``weather_battery_high_altitude_road`` (this session's measurement) is
    the concrete case: DEN's per-team-season exposure is ~0.5 EVERY season
    (its home games are deterministically flagged), a single structurally
    fixed unit sitting far from the population mean every year. That one
    unit alone drove the measured +0.7749 -- excluding it flips the sign to
    -0.2414 (measured, this session). This function excludes the single unit
    whose per-unit mean deviates most from the population mean and
    remeasures; a sign flip on removal means the headline number is that
    unit's determinism, not a real trait/exposure reliability, and the
    reading must be reported as ``not_informative_near_constant``, never
    recorded.
    """

    frame = long.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    scoped = frame.loc[frame["season"].between(seasons[0], seasons[1])]
    values = pd.to_numeric(scoped[metric_col], errors="coerce")
    per_unit = values.groupby(scoped[unit_col]).mean()
    per_unit = per_unit.dropna()
    if per_unit.shape[0] < 6 or baseline_reliability is None:
        return None
    overall_mean = float(values.mean())
    dominant_unit = (per_unit - overall_mean).abs().idxmax()
    excluded = frame.loc[frame[unit_col] != dominant_unit]
    without = rlib.measure_reliability(
        excluded, metric_col, method=method, unit_col=unit_col, seasons=seasons, n_boot=n_boot
    )
    without_rel = without["reliability"]
    sign_flip = bool(
        without_rel is not None
        and without_rel * baseline_reliability < 0
        and abs(without_rel) > 0.05
    )
    return {
        "dominant_unit": str(dominant_unit),
        "dominant_unit_per_unit_mean": float(per_unit.loc[dominant_unit]),
        "population_mean": overall_mean,
        "reliability_with_dominant_unit": baseline_reliability,
        "reliability_without_dominant_unit": without_rel,
        "status_without": without["status"],
        "sign_flip": sign_flip,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=rlib.N_BOOT)
    args = parser.parse_args()

    started = time.time()
    entries = target_entries()
    print(f"=== {len(entries)} weather registry cells in scope ===")

    quantities = build_quantities()
    battery_df = quantities.pop("_battery_df")
    followup_df = quantities.pop("_followup_df")
    forecast_tn_df = quantities.pop("_forecast_tn_df")
    wxtot_df = quantities.pop("_wxtot_df")

    battery_cells = battery_screen.build_cells(battery_df)
    followup_cells = followup_screen.build_cells(followup_df)
    forecast_tn_cells = fw_screen.build_cells(forecast_tn_df)
    wxtot_cells = wxtot_screen.build_cells(wxtot_df)

    features = pd.read_parquet(REPO / "data/processed/game_features.parquet")
    kn_construct_cache: dict[str, Any] = {}

    rows: list[dict[str, Any]] = []
    replication_by_family: dict[str, dict[str, dict[str, Any]]] = {
        "battery": {},
        "followup": {},
        "forecast_tn": {},
        "wxtot": {},
        "forecast_kn": {},
    }
    quantity_cache: dict[tuple[str, tuple[int, int]], dict[str, Any]] = {}
    dominant_cache: dict[tuple[str, tuple[int, int]], dict[str, Any] | None] = {}

    for name in sorted(entries):
        entry = entries[name]
        spec = ENTRY_SPECS[name]
        seasons = entry["seasons"]
        family = spec[FAMILY]
        method_tag = spec[METHOD_TAG]
        row: dict[str, Any] = {
            "entry": name,
            "family": family,
            "method_tag": method_tag,
            "seasons": list(seasons),
            "registry_effect": entry["effect"],
            "registry_classification": entry["classification"],
            "reason": spec[REASON],
        }

        if method_tag == "SKIP":
            row.update({"status": "skipped_joint_feature_family", "reliability": None})
            rows.append(row)
            continue

        if method_tag == "VENUE":
            quantity_key = spec[QUANTITY]
            row["parent_quantity"] = quantity_key
            cache_key = (quantity_key, seasons)
            if cache_key not in quantity_cache:
                quantity_cache[cache_key] = measure_venue(
                    quantities, quantity_key, seasons, n_boot=args.n_boot
                )
            measured = quantity_cache[cache_key]
            if cache_key not in dominant_cache:
                venue_long = quantities[quantity_key].rename(columns={"venue": "team_id"})
                dominant_cache[cache_key] = dominant_unit_check(
                    venue_long,
                    _METRIC_COL[quantity_key],
                    seasons,
                    unit_col="team_id",
                    method=rlib.METHOD_VENUE,
                    baseline_reliability=measured["reliability"],
                )
            dom = dominant_cache[cache_key]
            row["dominant_unit_check"] = dom
            if measured["status"] == rlib.STATUS_MEASURED and dom is not None and dom["sign_flip"]:
                measured = {**measured, "status": "not_informative_near_constant"}
            unit_label = "venue (home_team)"
            flag_source_df = {
                "battery": battery_df,
                "followup": followup_df,
                "forecast_tn": forecast_tn_df,
                "wxtot": wxtot_df,
            }[family]
            cell_dict = {
                "battery": battery_cells,
                "followup": followup_cells,
                "forecast_tn": forecast_tn_cells,
                "wxtot": wxtot_cells,
            }[family]
            flag = cell_dict[name]["flag"]
            outcome_col = "favorite_cover" if family == "wxtot" else "home_cover"
            replication = rlib.half_season_replication(
                flag_source_df, flag, outcome_col=outcome_col
            )
        else:  # EXPOSURE
            if family == "forecast_kn":
                builder_name = spec[QUANTITY]
                if builder_name not in kn_construct_cache:
                    builder = FLAG_BUILDERS[builder_name]
                    kn_construct_cache[builder_name] = builder.build(
                        features,
                        (1900, 2100),
                        {"forecast_archive_path": KN_ARCHIVE_REL},
                        REPO,
                    )
                construct = kn_construct_cache[builder_name]
                season_mask = construct.table["season"].between(seasons[0], seasons[1])
                table = construct.table.loc[season_mask]
                flag = construct.flag.loc[season_mask]
                unit_label = "team-week (home_team/away_team exposure)"
                replication = rlib.half_season_replication(table, flag, outcome_col="team_covered")
            else:
                flag_source_df = {"battery": battery_df}[family]
                cell_dict = {"battery": battery_cells}[family]
                flag = cell_dict[name]["flag"]
                table = flag_source_df
                unit_label = "team-week (home_team/away_team exposure)"
                replication = rlib.half_season_replication(
                    flag_source_df, flag, outcome_col="home_cover"
                )

            long = rlib.game_flag_to_team_week(table, flag)
            constancy = near_constant_check(long, "exposure")
            row["near_constant_check"] = constancy
            measured = rlib.measure_reliability(
                long,
                "exposure",
                method=rlib.METHOD_EXPOSURE,
                unit_col="team_id",
                seasons=seasons,
                n_boot=args.n_boot,
            )
            dom = dominant_unit_check(
                long,
                "exposure",
                seasons,
                unit_col="team_id",
                method=rlib.METHOD_EXPOSURE,
                baseline_reliability=measured["reliability"],
            )
            row["dominant_unit_check"] = dom
            if measured["status"] == rlib.STATUS_MEASURED and (
                constancy["near_constant"] or (dom is not None and dom["sign_flip"])
            ):
                measured = {**measured, "status": "not_informative_near_constant"}

        row.update(
            {
                "unit": unit_label,
                "n_units": measured["n_units"],
                "pearson_r": measured["pearson_r"],
                "pearson_r_ci95": measured["pearson_r_ci95"],
                "spearman_rho": measured.get("spearman_rho"),
                "spearman_brown_full_length_reliability": measured.get(
                    "spearman_brown_full_length_reliability"
                ),
                "probability_positive": measured.get("probability_positive"),
                "reliability": measured["reliability"],
                "reliability_low": measured["reliability_low"],
                "reliability_high": measured["reliability_high"],
                "status": measured["status"],
                "method": measured["method"],
                "half_season_replication": replication,
            }
        )
        rows.append(row)
        replication_by_family[family][name] = replication

        rel = row.get("reliability")
        shown = f"{rel:+.4f}" if isinstance(rel, float) else "  n/a "
        print(f"  {name:<58} n={row.get('n_units', 'n/a'):>4} rel={shown} {row['status']}")

    # weak_stack_v4 report-only inputs: 3 continuous kn forecast columns at
    # the entries' own [2020, 2025] window.
    v4_window = (2020, 2025)
    v4_inputs = {
        q: measure_venue(quantities, q, v4_window, n_boot=args.n_boot)
        for q in ("forecast_temp_kn", "forecast_wind_kn", "forecast_precip_kn")
    }

    # Positive controls: venue-unit and team-unit, once per season window
    # actually used above.
    windows = sorted({tuple(r["seasons"]) for r in rows if r.get("n_units")})
    positive_controls: dict[str, dict[str, list[dict[str, Any]]]] = {}
    a_team_unit_long = (
        rlib.game_flag_to_team_week(
            kn_construct_cache[next(iter(kn_construct_cache))].table,
            kn_construct_cache[next(iter(kn_construct_cache))].flag,
        )
        if kn_construct_cache
        else None
    )
    for window in windows:
        venue_frame = quantities["temp_actual"].rename(columns={"venue": "team_id"})
        venue_frame = venue_frame.loc[venue_frame["season"].between(window[0], window[1])]
        controls: dict[str, list[dict[str, Any]]] = {
            "venue_unit": rlib.positive_control(venue_frame, unit_col="team_id", n_boot=1000)
        }
        if a_team_unit_long is not None:
            team_frame = a_team_unit_long.loc[
                a_team_unit_long["season"].between(window[0], window[1])
            ]
            controls["team_unit_exposure"] = rlib.positive_control(
                team_frame, unit_col="team_id", n_boot=1000
            )
        positive_controls[f"{window[0]}-{window[1]}"] = controls

    battery_repl_corr = rlib.battery_replication_correlation(replication_by_family["battery"])
    followup_repl_corr = rlib.battery_replication_correlation(replication_by_family["followup"])
    forecast_tn_repl_corr = rlib.battery_replication_correlation(
        replication_by_family["forecast_tn"]
    )
    wxtot_repl_corr = rlib.battery_replication_correlation(replication_by_family["wxtot"])
    forecast_kn_repl_corr = rlib.battery_replication_correlation(
        replication_by_family["forecast_kn"]
    )

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO / "artifacts" / "reliability_sweep" / "weather" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    configuration = {
        "command": "reliability-weather",
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "entries": list(ENTRY_NAMES),
    }
    measured_count = sum(1 for r in rows if r["status"] == rlib.STATUS_MEASURED)
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "mapping_provenance": (
            "scripts/nfl_weather_battery_screen.py / nfl_weather_followup_screen.py / "
            "nfl_forecast_weather_screen.py / weather_total_interaction_screen.py build_cells() "
            "for 22 flag-based cells (imported, never reimplemented); "
            "src/nfl_ats/experiment_runner.py FLAG_BUILDERS for the 10 forecast_weather_kn_* "
            "compound flags (imported, never reimplemented); weak_stack_v4_* report-only via "
            "nfl_ats.forecast_weather_features.FORECAST_WEATHER_COLUMNS. Read 2026-09-01."
        ),
        "entry_specs": {name: dict(spec) for name, spec in ENTRY_SPECS.items()},
        "results": rows,
        "weak_stack_v4_report_only_inputs": {
            "seasons": list(v4_window),
            "results": v4_inputs,
            "note": (
                "Reported only, never recorded to either weak_stack_v4_* registry entry: a "
                "6-feature joint ridge-model delta has no single thresholded quantity. Both "
                "weak_stack_v4_* effect numbers remain ceilings by construction for the "
                "forecast-weather channel and no reliability read here is evidence about a "
                "playable rule."
            ),
        },
        "positive_control": positive_controls,
        "battery_replication_correlation": {
            "weather_battery": battery_repl_corr,
            "weather_followup": followup_repl_corr,
            "forecast_weather_tuesday_noon": forecast_tn_repl_corr,
            "wxtot": wxtot_repl_corr,
            "forecast_weather_kn": forecast_kn_repl_corr,
        },
        "provenance": artifact_provenance(
            configuration, battery_screen.DEFAULT_SCHEDULES, project_root=REPO
        ),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="reliability-weather",
        metrics={
            "n_entries": len(rows),
            "n_measured": measured_count,
            "n_unmeasured": len(rows) - measured_count,
        },
        notes=(
            "Measure-only split-half/exposure reliability for the 33 weather-group registry "
            "cells (ORCH-D); nothing is closed or reclassified, per AGENTS.md's binding "
            "closing-grounds taxonomy. weak_stack_v4_* recorded nothing (joint feature family, "
            "inputs reported only)."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")

    print(f"\n{measured_count} of {len(rows)} measured")
    for label, predicate in (("<= 0.10", lambda v: v <= 0.10), (">= 0.80", lambda v: v >= 0.80)):
        hits = [
            r for r in rows if r["status"] == rlib.STATUS_MEASURED and predicate(r["reliability"])
        ]
        print(f"  {label}: {len(hits)}")
        for row in sorted(hits, key=lambda r: r["reliability"]):
            print(
                f"    {row['entry']:<58} {row['reliability']:+.4f} "
                f"[{row['reliability_low']:+.4f}, {row['reliability_high']:+.4f}] "
                f"({row['method_tag']})"
            )

    print("\n=== set-reliability commands (measured cells only) ===")
    for row in sorted(rows, key=lambda r: r["entry"]):
        if row["status"] != rlib.STATUS_MEASURED:
            continue
        print(
            f"nfl-ats weak-signals set-reliability --name {row['entry']} "
            f"--reliability {row['reliability']:.6f} "
            f"--reliability-low {row['reliability_low']:.6f} "
            f"--reliability-high {row['reliability_high']:.6f} "
            f'--method "{row["method"]}" '
            f'--source "{output_dir / "results.json"}" '
            f'--reason "{row["reason"]}"'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
