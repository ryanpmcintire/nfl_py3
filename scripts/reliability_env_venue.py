"""Split-half reliability for the 27 ``env_venue`` registry cells (ORCH-D).

**What these cells are.** Six builders, five constructs: altitude/elevation
(``scripts/altitude_screen.py``), retractable-roof state
(``scripts/roof_decision_screen.py``), playing-surface familiarity
(``scripts/surface_familiarity_screen.py`` and, by shared construct,
``scripts/surface_profile_opener_eval.py``'s opener re-screen and
``era_trend_surface_switch``), the home-county environmental index
(``scripts/environmental_exposure_battery.py``), and venue milestones
(``scripts/venue_milestone_screen.py``). Every flag is imported from its
owning screen's own builder -- ``altitude_screen.build_cells``,
``venue_milestone_screen.build_flags``, the column-building functions inside
``roof_decision_screen.build_long_table``/``merge_forecast``,
``surface_familiarity_screen.build_pair``, and
``environmental_exposure_battery.cell_a_flags``/``cell_b_flags`` -- never
re-derived by hand.

**Binding taxonomy (verbatim, AGENTS.md / CLAUDE.md).** An interval or CI
that contains zero is NEVER grounds to reject, fail, or close an experiment.
Only two grounds ever close a line of work: (1) refuted mechanism -- a
RESOLVED wrong sign (whole interval on the wrong side of zero) or zero
split-half reliability; (2) bounded by a positive control proven able to
detect an effect that size. Everything else is ``unresolved_below_power``;
report ``probability_positive``, never "contains zero". This script CLOSES
NOTHING: it measures, and a low number is a candidate for the reliability
ground, never the closure itself. Within-week correlation is ZERO.

**Method.** ``METHOD_VENUE`` (unit = venue-season, i.e. ``stadium`` where
available else ``home_team``) for every continuous venue quantity
(altitude deficit, roof-open state, AQI, drought index, surface-mismatch
exposure) and ``METHOD_EXPOSURE`` (unit = team-season) for categorical flags
with no continuous parent (venue milestones, the Denver/Mexico-City
scheduling flags). Both reused verbatim from ``scripts/reliability_lib.py``.

**Two hazards handled explicitly (see module docstring sections below and
the printed report):**

(i) A trait CONSTANT within a venue-season (raw stadium elevation; raw
turf/grass surface norm) has a trivial 1.0 split-half by construction. Both
are measured here ONLY as a diagnostic (status forced to
``not_informative_constant_within_unit``, never recorded); the recorded
quantities are the WEEK-VARYING derived traits instead (altitude deficit
against that week's visitor; the visitor's own modal-surface mismatch
against the fixed venue surface).

(ii) A quantity whose SEASON TOTAL is compositionally conserved (a flag that
fires ~once per team-season, like a bye return or a home opener) can return
a strongly negative split-half correlation that is an artifact of the
constraint, not a football trait -- flagged by a concurrent worker
(``env_venue`` HAZARD message, measured on ``rest`` team-week: odd/even
r=-0.9766, random-half-reseed mean r=-0.8514). Every ``METHOD_EXPOSURE``
measurement here is re-run with the week column replaced by within-unit
random halves (5 reseeds); if the negative correlation survives that
randomization, it is reported ``not_applicable_compositional_constraint`` and
never recorded.

A construct with too few usable venue-seasons/team-seasons is reported as
UNMEASURED, never as reliability 0.

Writes ``artifacts/reliability_sweep/env_venue/<stamp>/results.json`` and
prints every proposed ``nfl-ats weak-signals set-reliability`` command
(``--record`` runs nothing itself; recording goes through the locked CLI).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.append(str(REPO / "scripts"))

import altitude_screen  # noqa: E402
import environmental_exposure_battery as env_battery  # noqa: E402
import reliability_lib as rlib  # noqa: E402
import roof_decision_screen  # noqa: E402
import surface_familiarity_screen as surface_screen  # noqa: E402
import venue_milestone_screen  # noqa: E402
from _common import default_schedules  # noqa: E402

from nfl_ats.experiment_runner import _opener_graded_features  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.weak_signals import default_registry_path, load_registry  # noqa: E402

# ---------------------------------------------------------------------------
# The 27-entry manifest, grouped by builder family (read from
# <scratchpad>/orchD_manifest.json, key groups.env_venue.entries, 2026-09-01).
# ---------------------------------------------------------------------------

ALTITUDE_ENTRIES = [
    "altitude_deficit_4000ft",
    "altitude_deficit_4000ft_division",
    "altitude_deficit_4000ft_era_2009_2017",
    "altitude_deficit_4000ft_era_2018_2025",
    "den_home_vs_own_conference",
    "mexico_city_neutral",
]
ROOF_ENTRIES = [
    "roof_battery_closed_benign_forecast_vs_open",
    "roof_battery_home_cover_open_vs_closed",
    "roof_battery_home_cover_open_vs_closed_opener",
    "roof_battery_total_covered_open_vs_closed",
    "roof_battery_visiting_dome_open_vs_closed",
    "roof_battery_visiting_dome_open_vs_closed_opener",
]
SURFACE_ENTRIES = [
    "surface_familiarity_r1_turf_venue_visitor_split",
    "surface_familiarity_r2_grass_venue_mirror",
    "surface_familiarity_r3_era_2009_2017",
    "surface_familiarity_r3_era_2018_2025",
    "surface_profile_opener_head_to_head",
    "surface_switch_feature_arm",
    "era_trend_surface_switch",
]
ENVIRONMENTAL_ENTRIES = [
    "environmental_battery_aqi_high_outdoor",
    "environmental_battery_aqi_high_outdoor_opener",
    "environmental_battery_drought_severe_grass",
    "environmental_battery_drought_severe_grass_opener",
]
MILESTONE_ENTRIES = [
    "venue_milestone_home_opener",
    "venue_milestone_new_stadium_debut",
    "venue_milestone_post_bye_home",
    "venue_milestone_post_bye_road",
]
ALL_ENTRIES = (
    ALTITUDE_ENTRIES + ROOF_ENTRIES + SURFACE_ENTRIES + ENVIRONMENTAL_ENTRIES + MILESTONE_ENTRIES
)

RETRACTABLE_TEAMS = roof_decision_screen.RETRACTABLE_TEAMS

STATUS_NOT_INFORMATIVE_CONSTANT = "not_informative_constant_within_unit"
STATUS_NOT_INFORMATIVE_NEAR_CONSTANT = "not_informative_near_constant"
STATUS_NOT_APPLICABLE_COMPOSITIONAL = "not_applicable_compositional_constraint"


def target_entries() -> dict[str, dict[str, Any]]:
    """This group's 27 registry cells, seasons/effect/source read live."""

    registry = load_registry(default_registry_path(REPO / "registry"))
    out: dict[str, dict[str, Any]] = {}
    for name in ALL_ENTRIES:
        signal = registry.signals[name]
        out[name] = {
            "seasons": (int(signal.seasons[0]), int(signal.seasons[1])),
            "reliability": signal.reliability,
            "effect": signal.effect,
            "classification": signal.classification,
            "source": signal.source,
        }
    return out


# ---------------------------------------------------------------------------
# Hazard (ii) probe: does a negative METHOD_EXPOSURE reliability survive a
# random (non-chronological) within-unit half split?
# ---------------------------------------------------------------------------


def random_half_probe(
    long: pd.DataFrame,
    metric: str,
    *,
    unit_col: str,
    seasons: tuple[int, int],
    n_reseeds: int = 5,
    seed: int = rlib.RELIABILITY_SEED,
) -> dict[str, Any]:
    """Re-measure with ``week`` replaced by a random 0/1 draw within each row.

    ``measure_reliability`` splits on ``week % 2``; substituting an
    independent random coin flip for the real week number destroys any
    chronological structure while preserving each unit's true within-season
    values, so a correlation that is a compositional artifact (season total
    conserved -> a value present in one half is structurally absent from the
    other) survives, while a correlation that depends on schedule ORDER
    (an odd/even trend, a hot streak) does not.
    """

    rng = np.random.default_rng(seed)
    draws: list[float | None] = []
    for i in range(n_reseeds):
        shuffled = long.copy()
        shuffled["week"] = rng.integers(0, 2, size=len(shuffled))
        measured = rlib.measure_reliability(
            shuffled,
            metric,
            method=rlib.METHOD_EXPOSURE,
            unit_col=unit_col,
            seasons=seasons,
            seed=seed + i + 1,
            n_boot=500,
        )
        draws.append(measured["reliability"])
    finite = [d for d in draws if d is not None]
    return {
        "n_reseeds": n_reseeds,
        "reliabilities": draws,
        "mean_reliability": float(np.mean(finite)) if finite else None,
        "note": (
            "week column replaced by an independent random 0/1 draw per row, "
            f"{n_reseeds} reseeds; if the negative correlation survives this it is a "
            "compositional artifact (season total conserved), not a football trait -- "
            "reported not_applicable_compositional_constraint, never recorded"
        ),
    }


COMPOSITIONAL_SURVIVAL_THRESHOLD = -0.30


def _compositional_guard(point_estimate: float | None, probe: dict[str, Any]) -> bool:
    """True if a negative reading looks compositional rather than a real trait."""

    if point_estimate is None or point_estimate >= 0:
        return False
    mean_probe = probe.get("mean_reliability")
    return mean_probe is not None and mean_probe <= COMPOSITIONAL_SURVIVAL_THRESHOLD


def force_diagnostic(measured: dict[str, Any], *, status: str, note: str) -> dict[str, Any]:
    """Overwrite a ``measure_reliability`` result into a diagnostic-only row.

    Used for hazard (i): a quantity constant within its unit (raw stadium
    elevation, raw surface_norm) returns a trivial ~1.0 split-half by
    construction. The number is kept in the artifact for transparency but the
    fields a validator could read as a recordable reliability are wiped, and
    ``status`` is overwritten so nothing downstream mistakes this for
    ``STATUS_MEASURED``.
    """

    out = dict(measured)
    out["status"] = status
    out["reliability"] = None
    out["reliability_low"] = None
    out["reliability_high"] = None
    out["note"] = note
    return out


def roof_open_frame(frame: pd.DataFrame, *, retractable_teams: frozenset[str]) -> pd.DataFrame:
    """This venue's own roof_open trait (1.0 open / 0.0 closed / NaN else).

    Restricted to home rows at the RETRACTABLE_TEAMS venues -- the exact
    ``is_home & home_team.isin(RETRACTABLE_TEAMS)`` gate
    ``roof_decision_screen.build_long_table`` uses to build its own
    ``retract_open``/``retract_closed`` flags (verified equal to those two
    flags on their respective values in ``tests/test_reliability_env_venue.py``),
    generalized from "is it open" / "is it closed" to a single continuous
    0/1 trait so its odd/even-week split-half is measurable at all.
    """

    retract = frame.loc[frame["is_home"] & frame["home_team"].isin(retractable_teams)].copy()
    retract["roof_open"] = np.select(
        [retract["roof"] == "open", retract["roof"] == "closed"], [1.0, 0.0], default=np.nan
    )
    return retract


# ---------------------------------------------------------------------------
# Altitude family
# ---------------------------------------------------------------------------


def altitude_family(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    schedules = default_schedules()
    elevations = altitude_screen.load_elevations(altitude_screen.DEFAULT_ELEVATIONS_PATH)
    df = altitude_screen.load_population(schedules, elevations)
    cells = {c["name"]: c for c in altitude_screen.build_cells(df)}

    # --- hazard (i) diagnostic: raw venue elevation is constant within a
    # venue-season by construction (a stadium's elevation never changes
    # week to week). Measured once, never recorded.
    elev_diag = force_diagnostic(
        rlib.measure_reliability(
            df, "venue_elev_ft", method=rlib.METHOD_VENUE, unit_col="stadium", seasons=(2009, 2025)
        ),
        status=STATUS_NOT_INFORMATIVE_CONSTANT,
        note=(
            "diagnostic only: raw stadium elevation is identical every week within a "
            "venue-season, so any odd/even split-half correlation on it is a statement "
            "about the measurement (constant vs constant), not about football -- never "
            "recorded; the RECORDED quantity is altitude_deficit_ft, which varies weekly "
            "because the visiting team's home elevation changes with the schedule"
        ),
    )

    # --- recorded quantity: altitude_deficit_ft (venue elevation minus that
    # week's visitor's own modal home elevation), unit = stadium ("venue" per
    # the group's METHOD_VENUE convention -- SAID here: stadium, available on
    # this frame). Windows shared across cells whose registry season range
    # matches, per the attention_battery/graph_team_stat family precedent.
    windows = sorted(
        {entries[n]["seasons"] for n in entries if n.startswith("altitude_deficit_4000ft")}
    )
    deficit_by_window = {
        w: rlib.measure_reliability(
            df, "altitude_deficit_ft", method=rlib.METHOD_VENUE, unit_col="stadium", seasons=w
        )
        for w in windows
    }

    entry_measurements: dict[str, dict[str, Any]] = {}
    entry_measurements["altitude_deficit_4000ft"] = deficit_by_window[
        entries["altitude_deficit_4000ft"]["seasons"]
    ]
    entry_measurements["altitude_deficit_4000ft_division"] = deficit_by_window[
        entries["altitude_deficit_4000ft_division"]["seasons"]
    ]
    entry_measurements["altitude_deficit_4000ft_era_2009_2017"] = deficit_by_window[
        entries["altitude_deficit_4000ft_era_2009_2017"]["seasons"]
    ]
    entry_measurements["altitude_deficit_4000ft_era_2018_2025"] = deficit_by_window[
        entries["altitude_deficit_4000ft_era_2018_2025"]["seasons"]
    ]

    # --- den_home_vs_own_conference / mexico_city_neutral: categorical
    # scheduling flags with no continuous parent -> EXPOSURE, team-season unit.
    for name in ("den_home_vs_own_conference", "mexico_city_neutral"):
        lo, hi = entries[name]["seasons"]
        flag = cells[name]["flag"]
        team_week = rlib.game_flag_to_team_week(df, flag)
        measured = rlib.measure_reliability(
            team_week, "exposure", method=rlib.METHOD_EXPOSURE, seasons=(lo, hi)
        )
        probe = None
        if measured["status"] == rlib.STATUS_MEASURED and measured["reliability"] is not None:
            probe = random_half_probe(team_week, "exposure", unit_col="team_id", seasons=(lo, hi))
            if _compositional_guard(measured["reliability"], probe):
                measured = {
                    **measured,
                    "status": STATUS_NOT_APPLICABLE_COMPOSITIONAL,
                    "reliability": None,
                    "reliability_low": None,
                    "reliability_high": None,
                }
        measured["random_half_probe"] = probe
        entry_measurements[name] = measured

    # --- half-season effect replication + battery correlation, per cell,
    # reported only (never recorded) per reliability_lib.half_season_replication.
    half_season: dict[str, Any] = {}
    for name, cell in cells.items():
        games = df.loc[cell["population"]]
        flag = cell["flag"].reindex(games.index)
        half_season[name] = rlib.half_season_replication(games, flag, outcome_col="home_cover")
    battery_corr = rlib.battery_replication_correlation(half_season)

    # --- positive controls: venue-unit (all 32 team stadiums) + the
    # den/mexico team-week frame as a small-N team-unit sanity check.
    controls = {
        "venue_unit_all_stadiums_2009_2025": rlib.positive_control(df, unit_col="stadium"),
    }

    return {
        "diagnostics": {"venue_elev_ft_constant_within_unit": elev_diag},
        "entry_measurements": entry_measurements,
        "half_season_replication": half_season,
        "battery_replication_correlation": battery_corr,
        "positive_control": controls,
        "provenance_note": (
            "scripts/altitude_screen.py:126-273 (load_population, build_cells): "
            "altitude_deficit_ft = venue_elev_ft - away_home_elev_ft, computed against "
            "registry/stadium_elevations.json; den_home_vs_own_conference and "
            "mexico_city_neutral flags read verbatim from build_cells' own cell specs. "
            "Read 2026-09-01."
        ),
    }


# ---------------------------------------------------------------------------
# Roof family
# ---------------------------------------------------------------------------


def roof_family(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    schedules = default_schedules()
    merged = roof_decision_screen.load_merged(roof_decision_screen.DEFAULT_FEATURES, schedules)
    dome_seasons = roof_decision_screen.fixed_dome_team_seasons(schedules)
    long_df = roof_decision_screen.build_long_table(merged, dome_seasons)
    game_df = roof_decision_screen.build_game_table(merged)
    long_df_fc = roof_decision_screen.merge_forecast(
        long_df.loc[long_df["season"].between(2020, 2025)].copy(),
        roof_decision_screen.DEFAULT_FORECAST,
    )
    opener_merged, opener_note = _opener_graded_features(merged, repo_root=REPO, market_root=None)
    opener_long = roof_decision_screen.build_long_table(opener_merged, dome_seasons)

    # --- recorded quantity: roof_open (this venue's own retractable-roof
    # state this week), unit = home_team ("venue" -- SAID here: home_team,
    # because build_long_table carries no separate stadium column and every
    # one of the 5 retractable teams held one venue for the whole window).
    retract_close = roof_open_frame(long_df, retractable_teams=RETRACTABLE_TEAMS)
    windows = sorted({entries[n]["seasons"] for n in ROOF_ENTRIES})
    roof_by_window: dict[tuple[int, int], dict[str, Any]] = {}
    for w in windows:
        # 2020-2025 cells actually score OPENER-grade population for two of
        # the three; the roof_open TRAIT itself (a stadium fact) is
        # unaffected by opener vs close grading, so one measurement per
        # SEASON WINDOW on the close-grade population covers both grades --
        # documented explicitly, not silently assumed.
        roof_by_window[w] = rlib.measure_reliability(
            retract_close, "roof_open", method=rlib.METHOD_VENUE, unit_col="home_team", seasons=w
        )

    entry_measurements = {name: roof_by_window[entries[name]["seasons"]] for name in ROOF_ENTRIES}

    cell_specs: dict[str, tuple[pd.DataFrame, str, str]] = {
        "roof_battery_home_cover_open_vs_closed": (long_df, "retract_open", "team_covered"),
        "roof_battery_home_cover_open_vs_closed_opener": (
            opener_long,
            "retract_open",
            "team_covered",
        ),
        "roof_battery_total_covered_open_vs_closed": (game_df, "retract_open", "total_covered"),
        "roof_battery_visiting_dome_open_vs_closed": (
            long_df,
            "visiting_dome_open",
            "team_covered",
        ),
        "roof_battery_visiting_dome_open_vs_closed_opener": (
            opener_long,
            "visiting_dome_open",
            "team_covered",
        ),
        "roof_battery_closed_benign_forecast_vs_open": (
            long_df_fc,
            "benign_forecast_closed",
            "team_covered",
        ),
    }
    half_season = {
        name: rlib.half_season_replication(frame, frame[flag_col], outcome_col=outcome_col)
        for name, (frame, flag_col, outcome_col) in cell_specs.items()
    }
    battery_corr = rlib.battery_replication_correlation(half_season)

    controls = {
        "venue_unit_retractable_only_2009_2025": rlib.positive_control(
            retract_close, unit_col="home_team"
        ),
    }

    return {
        "entry_measurements": entry_measurements,
        "half_season_replication": half_season,
        "battery_replication_correlation": battery_corr,
        "positive_control": controls,
        "opener_population_note": opener_note,
        "provenance_note": (
            "scripts/roof_decision_screen.py:131-214 (build_long_table, merge_forecast): "
            "roof_open derived from the 'roof' column at each of the 5 RETRACTABLE_TEAMS' "
            "home games (is_home & home_team.isin(RETRACTABLE_TEAMS)), 'open'->1.0, "
            "'closed'->0.0, else NaN. retract_open/visiting_dome_open/benign_forecast_closed "
            "flags read verbatim from that module. Read 2026-09-01."
        ),
    }


# ---------------------------------------------------------------------------
# Surface familiarity family
# ---------------------------------------------------------------------------


def surface_family(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    schedules = default_schedules()
    df = surface_screen.load_population(schedules)

    # --- hazard (i) diagnostic: raw venue playing surface (turf vs grass) is
    # constant within a home team's venue-season by construction. Measured
    # once, never recorded.
    diag_frame = df.assign(is_turf=(df["surface_norm"] == "turf").astype(float))
    surface_diag = force_diagnostic(
        rlib.measure_reliability(
            diag_frame,
            "is_turf",
            method=rlib.METHOD_VENUE,
            unit_col="home_team",
            seasons=(2009, 2025),
        ),
        status=STATUS_NOT_INFORMATIVE_CONSTANT,
        note=(
            "diagnostic only: a home venue's playing surface does not change week to week "
            "within a season, so its odd/even split-half is trivially ~1.0 by construction -- "
            "never recorded; the RECORDED quantity is the visitor-surface-mismatch indicator, "
            "which varies weekly because the visiting team changes"
        ),
    )

    # --- recorded quantity 1: within TURF-venue games, is this week's visitor
    # grass-modal (arm_mismatched from build_pair) -- varies weekly via the
    # opponent on the schedule. Unit = home_team (venue).
    r1_pair_full = surface_screen.build_pair(df, venue_surface="turf", mismatch_surface="grass")
    r2_pair_full = surface_screen.build_pair(df, venue_surface="grass", mismatch_surface="turf")

    turf_windows = sorted(
        {
            entries[n]["seasons"]
            for n in (
                "surface_familiarity_r1_turf_venue_visitor_split",
                "surface_familiarity_r3_era_2009_2017",
                "surface_familiarity_r3_era_2018_2025",
                "surface_switch_feature_arm",
                "era_trend_surface_switch",
                "surface_profile_opener_head_to_head",
            )
        }
    )
    turf_by_window = {
        w: rlib.measure_reliability(
            r1_pair_full,
            "arm_mismatched",
            method=rlib.METHOD_VENUE,
            unit_col="home_team",
            seasons=w,
        )
        for w in turf_windows
    }
    grass_window = entries["surface_familiarity_r2_grass_venue_mirror"]["seasons"]
    grass_measurement = rlib.measure_reliability(
        r2_pair_full,
        "arm_mismatched",
        method=rlib.METHOD_VENUE,
        unit_col="home_team",
        seasons=grass_window,
    )

    entry_measurements = {
        "surface_familiarity_r1_turf_venue_visitor_split": turf_by_window[
            entries["surface_familiarity_r1_turf_venue_visitor_split"]["seasons"]
        ],
        "surface_familiarity_r2_grass_venue_mirror": grass_measurement,
        "surface_familiarity_r3_era_2009_2017": turf_by_window[
            entries["surface_familiarity_r3_era_2009_2017"]["seasons"]
        ],
        "surface_familiarity_r3_era_2018_2025": turf_by_window[
            entries["surface_familiarity_r3_era_2018_2025"]["seasons"]
        ],
        # These three registry cells score a MODEL feature / an opener-grade
        # accuracy delta, not a flag-vs-complement gap directly -- but their
        # feature IS, byte-for-byte, surface_switch_flag ==
        # (away_modal_surface=='grass') & (surface_norm=='turf'), which on the
        # turf-venue population is exactly arm_mismatched above (verified:
        # src/nfl_ats/surface_switch_tilt_overlay.py:169-227, ported verbatim
        # from this same screen's build). Reliability is inherited from that
        # shared construct, not independently re-measured on a model-accuracy
        # column (there is no "trait" behind an accuracy delta to split).
        "surface_switch_feature_arm": turf_by_window[
            entries["surface_switch_feature_arm"]["seasons"]
        ],
        "era_trend_surface_switch": turf_by_window[entries["era_trend_surface_switch"]["seasons"]],
        "surface_profile_opener_head_to_head": turf_by_window[
            entries["surface_profile_opener_head_to_head"]["seasons"]
        ],
    }

    era_1_df = df.loc[df["season"].between(2009, 2017)]
    era_2_df = df.loc[df["season"].between(2018, 2025)]
    half_season = {
        "surface_familiarity_r1_turf_venue_visitor_split": rlib.half_season_replication(
            r1_pair_full, r1_pair_full["arm_mismatched"], outcome_col="home_cover"
        ),
        "surface_familiarity_r2_grass_venue_mirror": rlib.half_season_replication(
            r2_pair_full, r2_pair_full["arm_mismatched"], outcome_col="home_cover"
        ),
        "surface_familiarity_r3_era_2009_2017": rlib.half_season_replication(
            surface_screen.build_pair(era_1_df, venue_surface="turf", mismatch_surface="grass"),
            surface_screen.build_pair(era_1_df, venue_surface="turf", mismatch_surface="grass")[
                "arm_mismatched"
            ],
            outcome_col="home_cover",
        ),
        "surface_familiarity_r3_era_2018_2025": rlib.half_season_replication(
            surface_screen.build_pair(era_2_df, venue_surface="turf", mismatch_surface="grass"),
            surface_screen.build_pair(era_2_df, venue_surface="turf", mismatch_surface="grass")[
                "arm_mismatched"
            ],
            outcome_col="home_cover",
        ),
        "surface_switch_feature_arm": rlib.half_season_replication(
            r1_pair_full.loc[r1_pair_full["season"].between(2018, 2025)],
            r1_pair_full.loc[r1_pair_full["season"].between(2018, 2025), "arm_mismatched"],
            outcome_col="home_cover",
        ),
        "era_trend_surface_switch": rlib.half_season_replication(
            r1_pair_full.loc[r1_pair_full["season"].between(2020, 2025)],
            r1_pair_full.loc[r1_pair_full["season"].between(2020, 2025), "arm_mismatched"],
            outcome_col="home_cover",
        ),
        "surface_profile_opener_head_to_head": rlib.half_season_replication(
            r1_pair_full.loc[r1_pair_full["season"].between(2020, 2025)],
            r1_pair_full.loc[r1_pair_full["season"].between(2020, 2025), "arm_mismatched"],
            outcome_col="home_cover",
        ),
    }
    battery_corr = rlib.battery_replication_correlation(half_season)

    controls = {
        "venue_unit_turf_venues_2009_2025": rlib.positive_control(
            r1_pair_full, unit_col="home_team"
        ),
    }

    return {
        "diagnostics": {"surface_norm_constant_within_unit": surface_diag},
        "entry_measurements": entry_measurements,
        "half_season_replication": half_season,
        "battery_replication_correlation": battery_corr,
        "positive_control": controls,
        "provenance_note": (
            "scripts/surface_familiarity_screen.py:101-115 (build_pair), imported "
            "verbatim, which itself reuses scripts/nfl_weather_battery_screen.py:156-213 "
            "(load_population) for surface_norm/away_modal_surface. Cross-checked against "
            "src/nfl_ats/surface_switch_tilt_overlay.py:169-227's independent port of the "
            "identical construct. Read 2026-09-01."
        ),
    }


# ---------------------------------------------------------------------------
# Environmental exposure family
# ---------------------------------------------------------------------------


def environmental_family(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    df, _counters = env_battery.load_population(
        env_battery.DEFAULT_FEATURES, env_battery.DEFAULT_JOIN
    )
    opener_df, opener_note = _opener_graded_features(df, repo_root=REPO, market_root=None)
    opener_df = opener_df.loc[opener_df["home_cover"].notna()].reset_index(drop=True)
    opener_df["is_outdoor_exposed"] = opener_df["is_outdoor_exposed"].fillna(False).astype(bool)
    opener_df["surface_is_grass"] = (
        opener_df["surface"].astype(str).str.contains("grass", case=False, na=False)
    )

    pop_a, _sub_a = env_battery.cell_a_flags(df)
    pop_b, _sub_b = env_battery.cell_b_flags(df)
    pop_a_op, _sub_a_op = env_battery.cell_a_flags(opener_df)
    pop_b_op, _sub_b_op = env_battery.cell_b_flags(opener_df)

    close_lo, close_hi = entries["environmental_battery_aqi_high_outdoor"]["seasons"]
    open_lo, open_hi = entries["environmental_battery_aqi_high_outdoor_opener"]["seasons"]

    entry_measurements = {
        "environmental_battery_aqi_high_outdoor": rlib.measure_reliability(
            df.loc[pop_a],
            "aqi",
            method=rlib.METHOD_VENUE,
            unit_col="home_team",
            seasons=(close_lo, close_hi),
        ),
        "environmental_battery_aqi_high_outdoor_opener": rlib.measure_reliability(
            opener_df.loc[pop_a_op],
            "aqi",
            method=rlib.METHOD_VENUE,
            unit_col="home_team",
            seasons=(open_lo, open_hi),
        ),
        "environmental_battery_drought_severe_grass": rlib.measure_reliability(
            df.loc[pop_b],
            "drought_d2",
            method=rlib.METHOD_VENUE,
            unit_col="home_team",
            seasons=(close_lo, close_hi),
        ),
        "environmental_battery_drought_severe_grass_opener": rlib.measure_reliability(
            opener_df.loc[pop_b_op],
            "drought_d2",
            method=rlib.METHOD_VENUE,
            unit_col="home_team",
            seasons=(open_lo, open_hi),
        ),
    }

    def _sub_frame(base: pd.DataFrame, population: pd.Series, subset: pd.Series) -> pd.DataFrame:
        scored = base.loc[population].copy()
        scored["_flag"] = subset.loc[population].to_numpy()
        return scored

    half_season = {
        "environmental_battery_aqi_high_outdoor": rlib.half_season_replication(
            _sub_frame(df, pop_a, env_battery.cell_a_flags(df)[1]),
            _sub_frame(df, pop_a, env_battery.cell_a_flags(df)[1])["_flag"],
            outcome_col="home_cover",
        ),
        "environmental_battery_aqi_high_outdoor_opener": rlib.half_season_replication(
            _sub_frame(opener_df, pop_a_op, env_battery.cell_a_flags(opener_df)[1]),
            _sub_frame(opener_df, pop_a_op, env_battery.cell_a_flags(opener_df)[1])["_flag"],
            outcome_col="home_cover",
        ),
        "environmental_battery_drought_severe_grass": rlib.half_season_replication(
            _sub_frame(df, pop_b, env_battery.cell_b_flags(df)[1]),
            _sub_frame(df, pop_b, env_battery.cell_b_flags(df)[1])["_flag"],
            outcome_col="home_cover",
        ),
        "environmental_battery_drought_severe_grass_opener": rlib.half_season_replication(
            _sub_frame(opener_df, pop_b_op, env_battery.cell_b_flags(opener_df)[1]),
            _sub_frame(opener_df, pop_b_op, env_battery.cell_b_flags(opener_df)[1])["_flag"],
            outcome_col="home_cover",
        ),
    }
    battery_corr = rlib.battery_replication_correlation(half_season)

    controls = {
        "venue_unit_all_home_counties_2009_2025": rlib.positive_control(df, unit_col="home_team"),
    }

    return {
        "entry_measurements": entry_measurements,
        "half_season_replication": half_season,
        "battery_replication_correlation": battery_corr,
        "positive_control": controls,
        "opener_population_note": opener_note,
        "provenance_note": (
            "scripts/environmental_exposure_battery.py:95-148 (load_population, "
            "cell_a_flags, cell_b_flags), imported verbatim; aqi/drought_d2 are the raw "
            "home-county index columns from data/processed/environmental_exposures/"
            "game_join.parquet. Read 2026-09-01."
        ),
    }


# ---------------------------------------------------------------------------
# Venue milestone family
# ---------------------------------------------------------------------------


def milestone_family(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    schedules = default_schedules()
    df = venue_milestone_screen.load_population(schedules)
    flags, _diagnostics = venue_milestone_screen.build_flags(df)

    entry_measurements: dict[str, dict[str, Any]] = {}
    for name in MILESTONE_ENTRIES:
        lo, hi = entries[name]["seasons"]
        flag = flags[name]
        team_week = rlib.game_flag_to_team_week(df, flag)
        measured = rlib.measure_reliability(
            team_week, "exposure", method=rlib.METHOD_EXPOSURE, seasons=(lo, hi)
        )
        probe = None
        if measured["status"] == rlib.STATUS_MEASURED and measured["reliability"] is not None:
            probe = random_half_probe(team_week, "exposure", unit_col="team_id", seasons=(lo, hi))
            if _compositional_guard(measured["reliability"], probe):
                measured = {
                    **measured,
                    "status": STATUS_NOT_APPLICABLE_COMPOSITIONAL,
                    "reliability": None,
                    "reliability_low": None,
                    "reliability_high": None,
                }
        measured["random_half_probe"] = probe
        entry_measurements[name] = measured

    half_season = {
        name: rlib.half_season_replication(df, flags[name], outcome_col="home_cover")
        for name in MILESTONE_ENTRIES
    }
    battery_corr = rlib.battery_replication_correlation(half_season)

    controls = {
        "team_unit_2009_2025": rlib.positive_control(
            rlib.game_flag_to_team_week(df, flags["venue_milestone_home_opener"]),
            unit_col="team_id",
        ),
    }

    return {
        "entry_measurements": entry_measurements,
        "half_season_replication": half_season,
        "battery_replication_correlation": battery_corr,
        "positive_control": controls,
        "provenance_note": (
            "scripts/venue_milestone_screen.py:158-282 (load_population, build_flags), "
            "imported verbatim; every flag is the module's own boolean Series, unaltered. "
            "Read 2026-09-01."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _clean(obj: Any) -> Any:
    """Drop non-JSON-serializable pandas artifacts (indexes) before writing."""

    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    started = time.time()
    entries = target_entries()
    print(f"=== {len(entries)} env_venue registry cells in scope ===")

    families = {
        "altitude": altitude_family(entries),
        "roof": roof_family(entries),
        "surface": surface_family(entries),
        "environmental": environmental_family(entries),
        "venue_milestone": milestone_family(entries),
    }

    rows: list[dict[str, Any]] = []
    for family_name, family_entries in (
        ("altitude", ALTITUDE_ENTRIES),
        ("roof", ROOF_ENTRIES),
        ("surface", SURFACE_ENTRIES),
        ("environmental", ENVIRONMENTAL_ENTRIES),
        ("venue_milestone", MILESTONE_ENTRIES),
    ):
        block = families[family_name]
        for name in family_entries:
            measured = block["entry_measurements"][name]
            row = {
                "entry": name,
                "family": family_name,
                "seasons": list(entries[name]["seasons"]),
                "registry_effect": entries[name]["effect"],
                "registry_classification": entries[name]["classification"],
                "n_units": measured.get("n_units"),
                "pearson_r": measured.get("pearson_r"),
                "pearson_r_ci95": measured.get("pearson_r_ci95"),
                "reliability": measured.get("reliability"),
                "reliability_low": measured.get("reliability_low"),
                "reliability_high": measured.get("reliability_high"),
                "status": measured.get("status"),
                "method": measured.get("method"),
                "random_half_probe": measured.get("random_half_probe"),
                "half_season_replication": block["half_season_replication"].get(name),
            }
            rows.append(row)
            rel = row["reliability"]
            shown = f"{rel:+.4f}" if rel is not None else "  n/a "
            print(f"  {name:<52} n={row['n_units']!s:>4} rel={shown} {row['status']}")

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO / "artifacts" / "reliability_sweep" / "env_venue" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    configuration = {
        "command": "reliability-env-venue",
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": rlib.N_BOOT,
        "min_units": rlib.MIN_UNITS,
        "entries": ALL_ENTRIES,
    }
    payload = _clean(
        {
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "elapsed_seconds": time.time() - started,
            "seed": rlib.RELIABILITY_SEED,
            "n_boot": rlib.N_BOOT,
            "min_units": rlib.MIN_UNITS,
            "method_venue": rlib.METHOD_VENUE,
            "method_exposure": rlib.METHOD_EXPOSURE,
            "hazard_i_note": (
                "constant-within-venue-season traits (raw elevation, raw surface_norm) are "
                "measured only as diagnostics, status forced not_informative_constant_within_unit, "
                "never recorded; see families.altitude.diagnostics / families.surface.diagnostics"
            ),
            "hazard_ii_note": (
                "every METHOD_EXPOSURE measurement is re-run with week replaced by an "
                "independent random 0/1 draw (5 reseeds, random_half_probe); a negative "
                "reliability whose randomized-week mean stays <= "
                f"{COMPOSITIONAL_SURVIVAL_THRESHOLD} is reported "
                "not_applicable_compositional_constraint and never recorded"
            ),
            "results": rows,
            "families": families,
            "provenance": artifact_provenance(
                configuration, env_battery.DEFAULT_FEATURES, project_root=REPO
            ),
        }
    )
    measured_count = sum(1 for r in rows if r["status"] == rlib.STATUS_MEASURED)
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="reliability-env-venue",
        metrics={
            "n_entries": len(rows),
            "n_measured": measured_count,
            "n_unmeasured": len(rows) - measured_count,
        },
        notes=(
            "Measure-only split-half reliability for the env_venue registry cells "
            "(altitude, roof, surface, environmental, venue milestone); every cell "
            "measured regardless of sign or interval shape, nothing closed or "
            "reclassified, per AGENTS.md's binding closing-grounds taxonomy."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")

    measured_rows = [r for r in rows if r["status"] == rlib.STATUS_MEASURED]
    print(f"\n{len(measured_rows)} of {len(rows)} measured; {len(rows) - len(measured_rows)} not")
    for label, predicate in (
        ("<= 0.10", lambda v: v <= 0.10),
        (">= 0.80", lambda v: v >= 0.80),
    ):
        hits = [r for r in measured_rows if predicate(r["reliability"])]
        print(f"  {label}: {len(hits)}")
        for row in sorted(hits, key=lambda r: r["reliability"]):
            method_short = row["method"][:60]
            print(
                f"    {row['entry']:<52} {row['reliability']:+.4f} "
                f"[{row['reliability_low']:+.4f}, {row['reliability_high']:+.4f}] {method_short}"
            )

    print("\n=== proposed set-reliability commands (measured entries only) ===")
    for row in measured_rows:
        print(
            f'nfl-ats weak-signals set-reliability --name "{row["entry"]}" '
            f"--reliability {row['reliability']:.4f} "
            f"--reliability-low {row['reliability_low']:.4f} "
            f"--reliability-high {row['reliability_high']:.4f} "
            f'--method "{row["method"]}" '
            f'--source "{output_dir / "results.json"}" '
            f'--reason "..."'
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
