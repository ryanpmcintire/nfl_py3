"""schedule_clock reliability measurement (ORCH-D, 2026-09-01).

Measures a split-half reliability for the 30 body-clock / travel-rest / DST
registry cells assigned to this worker, using ONLY ``scripts/reliability_lib.py``
(the shared estimator for the whole sweep -- never reimplemented) and each
screen's OWN flag builder (``body_clock_screen.build_cells``,
``body_clock_night_screen.build_cells``, ``nfl_travel_rest_battery_screen.
build_cells``, ``dst_transition_battery_screen.build_cells``), imported, never
re-derived by hand.

**Binding taxonomy, owned verbatim (AGENTS.md / CLAUDE.md).** An interval or
CI that contains zero is NEVER grounds to reject, fail, or close an
experiment. At this evaluator's ~2-point resolution "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line
of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
``unresolved_below_power``; report ``probability_positive``, never "contains
zero". Reliability is one of the two closing grounds, so a measured
reliability near zero is a real terminal finding ONLY when the measurement is
itself sound; a reliability measurement never by itself changes any cell's
classification. This script RECORDS reliabilities and never closes,
reclassifies, or proposes a ``closing_ground``. Within-week correlation is
ZERO.

**Method choice, per entry (docs/reliability_sweep_20260901.md sec 1, and the
worker brief):**

- A flag that THRESHOLDS a continuous team-week quantity (kickoff time,
  travel distance, timezone delta, days of rest) uses ``METHOD_TRAIT`` on
  that quantity -- the only one of the three admissible as a closing ground.
- A pure calendar/slot flag (Thursday, neutral site, DST calendar window)
  uses ``METHOD_EXPOSURE`` on the flag's own team-season exposure rate --
  NOT admissible as a closing ground (a schedule quirk with no stable team
  structure can still move covers).

Every ``body_clock_*`` / ``body_clock_night_*`` cell and the LPM-increment
redteam cell threshold the SAME continuous quantity -- kickoff time of day
(``kick_min``, read: scripts/body_clock_screen.py:41-44,85-86,117-119;
scripts/body_clock_night_screen.py:42,58) -- so, following the registry's own
``attention_battery_*`` precedent (a battery's cells inherit the reliability
of the trait they threshold, docs/reliability_sweep_20260901.md sec 1), they
share ONE measured ``kick_min`` team-week reliability per season range that
appears among these entries' own ``seasons`` fields (full 2009-2025, and the
two era splits 2009-2016 / 2017-2025). Similarly the two ``away_rest``-based
travel-rest cells (``travel_rest_away_off_bye``, ``travel_rest_short_week_
road``) share one measured ``away_rest`` reliability -- same trait, same
team, same season range, different threshold.

Writes ``artifacts/reliability_sweep/schedule_clock/<UTC stamp>/results.json``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import body_clock_night_screen as bcns  # noqa: E402
import body_clock_screen as bcs  # noqa: E402
import dst_transition_battery_screen as dsts  # noqa: E402
import nfl_travel_rest_battery_screen as trbs  # noqa: E402
from reliability_lib import (  # noqa: E402
    METHOD_EXPOSURE,
    METHOD_TRAIT,
    MIN_UNITS,
    N_BOOT,
    RELIABILITY_SEED,
    STATUS_MEASURED,
    battery_replication_correlation,
    game_flag_to_team_week,
    half_season_replication,
    measure_reliability,
    positive_control,
)

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

MANIFEST_PATH = Path(
    "C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/"
    "592413d4-e0ee-41ee-9508-ee10cac18dd5/scratchpad/orchD_manifest.json"
)

#: A hazard the ORCH-D coordinator surfaced (session message, 2026-09-01) and
#: this script independently reproduced on its OWN data before acting on it:
#: a quantity whose team-season total is effectively FIXED (rest days sum to
#: a season-length budget; a once-or-near-once-per-season flag like a bye or
#: a Thursday game has almost nowhere else to land) makes odd/even-week
#: split-half means MUTUALLY EXCLUSIVE across the two halves for many units,
#: which manufactures a strongly NEGATIVE Pearson r regardless of any real
#: week-to-week persistence. The diagnostic: reassign each team-season's
#: observations to two RANDOM (not true-week-parity) halves and remeasure. A
#: genuine trait's between-team variance survives an arbitrary reshuffling
#: (true r and random-half r stay close IN SIGN, both positive) -- that is
#: the expected, reassuring shape for kick_min / away_travel_mi /
#: tz_delta_eastbound / prev_own_travel_mi below, all confirmed clean this
#: way. A negative correlation that ALSO reproduces under random reshuffling
#: is not a week-order effect at all -- it is the sparsity/mutual-exclusivity
#: artifact, and recording it as a reliability would plant exactly the
#: booby-trap AGENTS.md's "reliability is a closing ground" rule warns about
#: (a fabricated ``no_split_half_reliability`` candidate on an untested
#: construct). Such entries are reported as ``STATUS_NOT_APPLICABLE_COMPOSITIONAL``
#: and are NEVER passed to ``set-reliability``.
STATUS_NOT_APPLICABLE_COMPOSITIONAL = "not_applicable_compositional_constraint"
RANDOM_HALF_RESEEDS = 8
RANDOM_HALF_N_BOOT = 200  # only the point r is used; a full 4000-draw CI is not needed here.


def random_half_diagnostic(
    long: pd.DataFrame,
    metric: str,
    *,
    method: str,
    seasons: tuple[int, int] | None,
    unit_col: str = "team_id",
    n_reseeds: int = RANDOM_HALF_RESEEDS,
) -> dict[str, Any]:
    """Remeasure ``metric`` with each team-season's weeks reassigned to two
    RANDOM (not true-parity) halves, ``n_reseeds`` times. Returns the true
    ``pearson_r`` alongside the random-half mean/range, plus a boolean
    ``artifact_suspected``: true only when the TRUE correlation is negative
    AND every one of the random-half reseeds is ALSO negative (the
    mutual-exclusivity/compositional signature -- see the module-level note
    above)."""

    true_meas = measure_reliability(long, metric, method=method, seasons=seasons)
    true_r = true_meas.get("pearson_r")

    rng = np.random.default_rng(RELIABILITY_SEED)
    random_rs: list[float] = []
    for reseed in range(n_reseeds):

        def _rand_half(group: pd.DataFrame) -> pd.Series:
            n = len(group)
            half = np.array(([1, 2] * (n // 2 + 1))[:n])
            rng.shuffle(half)
            return pd.Series(half, index=group.index)

        shuffled = long.copy()
        shuffled["week"] = shuffled.groupby([unit_col, "season"], group_keys=False).apply(
            _rand_half
        )
        res = measure_reliability(
            shuffled,
            metric,
            method=method,
            seasons=seasons,
            unit_col=unit_col,
            seed=RELIABILITY_SEED + reseed,
            n_boot=RANDOM_HALF_N_BOOT,
        )
        r = res.get("pearson_r")
        if r is not None and math.isfinite(r):
            random_rs.append(float(r))

    random_mean = float(np.mean(random_rs)) if random_rs else None
    random_range = [float(min(random_rs)), float(max(random_rs))] if random_rs else None
    artifact_suspected = bool(
        true_r is not None
        and math.isfinite(true_r)
        and true_r < 0
        and random_rs
        and all(r < 0 for r in random_rs)
    )
    return {
        "true_pearson_r": true_r,
        "random_half_mean_pearson_r": random_mean,
        "random_half_range_pearson_r": random_range,
        "n_reseeds": len(random_rs),
        "artifact_suspected": artifact_suspected,
    }


# ---------------------------------------------------------------------------
# 1. Population loaders -- every one of these delegates to the owning
#    screen's own ``load_population``/``build_cells``. Nothing here
#    re-derives a flag; the redteam masks (section 3) reuse the same
#    constants (``WEST_TZS``, ``NIGHT_KICK_MIN_MIN``) the owning screens
#    define, because neither ``edge_audit_redteam.run_claim3``'s
#    ``night_pop``/``west_road_pop`` populations nor the plain "night"/
#    "west" masks are exported as standalone flags anywhere.
# ---------------------------------------------------------------------------


def load_body_clock_df() -> pd.DataFrame:
    coords = bcs.load_coords(bcs.DEFAULT_COORDS_PATH)
    return bcs.load_population(bcs.default_schedules(), coords)


#: The registry's own recorded names for the 4 dose-bucket cells drop
#: "_west_road_" from body_clock_night_screen's actual cell keys (registry:
#: "body_clock_night_dose_1300" vs. the screen's
#: "body_clock_night_west_road_dose_1300") -- read: registry/weak_signals.json
#: lines 1465,1493,1521,1549 vs. scripts/body_clock_night_screen.py:108-130.
#: The manifest's own description for these entries already flags a
#: "copy-bug suspicion" on their recorded EFFECT numbers
#: (docs/registry_correlation_audit_20260822.md); this alias fixes ONLY the
#: name lookup used to reach the correct, well-defined dose-bucket FLAG for
#: reliability purposes -- it does not touch, and has no bearing on, the
#: effect-number discrepancy already on file.
_DOSE_NAME_ALIAS = {
    "body_clock_night_west_road_dose_1300": "body_clock_night_dose_1300",
    "body_clock_night_west_road_dose_1400_1659": "body_clock_night_dose_1400_1659",
    "body_clock_night_west_road_dose_1700_1959": "body_clock_night_dose_1700_1959",
    "body_clock_night_west_road_dose_ge2000": "body_clock_night_dose_ge2000",
}


def body_clock_cells(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Union of the two screens' own predeclared cells (6 + 9 = 15), imported
    verbatim; the 4 dose-bucket keys are renamed to the registry's own
    (mismatched) names -- see ``_DOSE_NAME_ALIAS``."""
    cells = dict(bcs.build_cells(df))
    night_cells = dict(bcns.build_cells(df))
    for src, dst in _DOSE_NAME_ALIAS.items():
        night_cells[dst] = night_cells.pop(src)
    cells.update(night_cells)
    return cells


def load_travel_rest_df() -> pd.DataFrame:
    coords = trbs.load_coords(trbs.DEFAULT_COORDS_PATH)
    return trbs.load_population(trbs.DEFAULT_SCHEDULES, coords)


def load_dst_df() -> pd.DataFrame:
    return dsts.build_population(dsts.DEFAULT_SCHEDULES, dsts.DEFAULT_COORDS_PATH)


# ---------------------------------------------------------------------------
# 2. Team-week long-frame builders for METHOD_TRAIT quantities. These are
#    plain reshapes of columns the owning screen's own ``load_population``
#    already computed (kick_min, away_travel_mi, tz_delta_eastbound,
#    home_rest, away_rest, prev_own_travel_mi) -- no new quantity is derived
#    here, only which side of the game the observation belongs to.
# ---------------------------------------------------------------------------


def two_sided_long(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Both teams in each game get one row -- for a quantity that is a
    property of the GAME shared by both sides (kickoff time of day)."""
    home = df.loc[:, ["home_team", "season", "week", metric]].rename(
        columns={"home_team": "team_id"}
    )
    away = df.loc[:, ["away_team", "season", "week", metric]].rename(
        columns={"away_team": "team_id"}
    )
    return pd.concat([home, away], ignore_index=True)


def one_sided_long(df: pd.DataFrame, team_col: str, metric: str) -> pd.DataFrame:
    """Only the named side gets a row -- for a quantity that is inherently
    side-specific (away team's own travel distance / rest days; home team's
    own rest days / own prior-trip hangover distance). Using the two-sided
    helper on a side-specific quantity would attribute one team's schedule
    fact to its opponent's team-week record, which is not what the
    construct means."""
    return df.loc[:, [team_col, "season", "week", metric]].rename(columns={team_col: "team_id"})


def one_sided_exposure_long(
    pop_df: pd.DataFrame, team_col: str, flag_in_pop: pd.Series
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": pop_df[team_col].to_numpy(),
            "season": pop_df["season"].to_numpy(),
            "week": pop_df["week"].to_numpy(),
            "exposure": flag_in_pop.reset_index(drop=True).astype(float).to_numpy(),
        }
    )


# ---------------------------------------------------------------------------
# 3. Redteam population masks (scripts/edge_audit_redteam.py:596-652).
#    ``west_night`` is EXACTLY body_clock_night_screen's own
#    ``body_clock_night_west_road_ge2000et`` flag (verified in the test
#    file by Series equality) -- reused, not recomputed independently.
#    ``night_arr`` / ``away_west`` are the same two-line boolean comparisons
#    edge_audit_redteam.py:599-601 and body_clock_night_screen.py:55,58 use
#    inline (neither module exports them as standalone flags), built here
#    from the SAME imported constants (``bcs.WEST_TZS``,
#    ``bcns.NIGHT_KICK_MIN_MIN``).
# ---------------------------------------------------------------------------


def redteam_masks(bc_df: pd.DataFrame) -> dict[str, pd.Series]:
    away_west = bc_df["away_body_tz"].isin(bcs.WEST_TZS).fillna(False)
    true_home = (bc_df["location"] == "Home").fillna(False)
    night = (bc_df["kick_min"] >= bcns.NIGHT_KICK_MIN_MIN).fillna(False)
    return {
        "away_west": away_west,
        "true_home": true_home,
        "night": night,
        "west_night": away_west & true_home & night,
    }


def redteam_distance_join(bc_df: pd.DataFrame, tr_df: pd.DataFrame) -> pd.Series:
    """Distance-validity mask reused from ``nfl_travel_rest_battery_screen``'s
    already-computed ``away_travel_mi`` (haversine, computed once there) via
    a ``game_id`` join -- not a re-derivation of the haversine formula
    ``edge_audit_redteam.py`` uses locally for the same purpose."""
    joined = bc_df[["game_id"]].merge(
        tr_df[["game_id", "away_travel_mi"]], on="game_id", how="left"
    )
    return joined["away_travel_mi"].notna()


# ---------------------------------------------------------------------------
# 4. Manifest
# ---------------------------------------------------------------------------


def load_manifest_entries() -> list[dict[str, Any]]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return list(payload["groups"]["schedule_clock"]["entries"])


# ---------------------------------------------------------------------------
# 5. Measurement dispatch
# ---------------------------------------------------------------------------

KICK_MIN_PROVENANCE = (
    "kick_min: continuous kickoff time of day in minutes past midnight ET, computed by "
    "body_clock_screen.load_population (read: scripts/body_clock_screen.py:85-86); every "
    "body_clock_*/body_clock_night_* cell and redteam_body_clock_night_increment_within_"
    "west_road_lpm threshold this SAME quantity, so they share one measured team-week "
    "reliability per season range (attention_battery_* precedent: a battery's cells "
    "inherit the reliability of the trait they threshold, docs/reliability_sweep_"
    "20260901.md sec 1)."
)


def _measure_with_compositional_check(
    long: pd.DataFrame,
    metric: str,
    *,
    method: str,
    seasons: tuple[int, int] | None,
    unit_col: str = "team_id",
) -> dict[str, Any]:
    """``measure_reliability`` plus the random-half compositional-artifact
    check (module docstring above ``STATUS_NOT_APPLICABLE_COMPOSITIONAL``).
    Only a NEGATIVE measured reliability is checked -- a positive one that
    survives random reshuffling is the expected signature of a genuine
    between-team trait, not a hazard."""

    meas = measure_reliability(long, metric, method=method, seasons=seasons, unit_col=unit_col)
    if meas["status"] != STATUS_MEASURED or meas["reliability"] is None or meas["reliability"] >= 0:
        meas["compositional_check"] = None
        return meas
    diag = random_half_diagnostic(long, metric, method=method, seasons=seasons, unit_col=unit_col)
    meas["compositional_check"] = diag
    if diag["artifact_suspected"]:
        meas["status"] = STATUS_NOT_APPLICABLE_COMPOSITIONAL
    return meas


def measure_kick_min_family(bc_df: pd.DataFrame, season_ranges: set[tuple[int, int]]) -> dict:
    long = two_sided_long(bc_df, "kick_min")
    out = {}
    for lo, hi in sorted(season_ranges):
        out[(lo, hi)] = _measure_with_compositional_check(
            long, "kick_min", method=METHOD_TRAIT, seasons=(lo, hi)
        )
    return out


def measure_travel_rest_traits(tr_df: pd.DataFrame) -> dict[str, dict]:
    return {
        "away_travel_mi": _measure_with_compositional_check(
            one_sided_long(tr_df, "away_team", "away_travel_mi"),
            "away_travel_mi",
            method=METHOD_TRAIT,
            seasons=(2009, 2025),
        ),
        "tz_delta_eastbound": _measure_with_compositional_check(
            one_sided_long(tr_df, "away_team", "tz_delta_eastbound"),
            "tz_delta_eastbound",
            method=METHOD_TRAIT,
            seasons=(2009, 2025),
        ),
        "home_rest": _measure_with_compositional_check(
            one_sided_long(tr_df, "home_team", "home_rest"),
            "home_rest",
            method=METHOD_TRAIT,
            seasons=(2009, 2025),
        ),
        "away_rest": _measure_with_compositional_check(
            one_sided_long(tr_df, "away_team", "away_rest"),
            "away_rest",
            method=METHOD_TRAIT,
            seasons=(2009, 2025),
        ),
        "prev_own_travel_mi": _measure_with_compositional_check(
            one_sided_long(tr_df, "home_team", "prev_own_travel_mi"),
            "prev_own_travel_mi",
            method=METHOD_TRAIT,
            seasons=(2009, 2025),
        ),
    }


def measure_travel_rest_exposures(tr_df: pd.DataFrame, tr_cells: dict) -> dict[str, dict]:
    out = {}
    for name in ("travel_rest_international_game", "travel_rest_thursday_pure"):
        long = game_flag_to_team_week(tr_df, tr_cells[name]["flag"])
        out[name] = _measure_with_compositional_check(
            long, "exposure", method=METHOD_EXPOSURE, seasons=(2009, 2025)
        )
    return out


def measure_dst_exposures(dst_df: pd.DataFrame, dst_cells: dict) -> dict[str, dict]:
    out = {}
    # Unrestricted population (population == everyone): standard symmetric exposure.
    for name in ("dst_fall_transition_shock", "dst_placebo_shifted_window"):
        long = game_flag_to_team_week(dst_df, dst_cells[name]["flag"])
        out[name] = _measure_with_compositional_check(
            long, "exposure", method=METHOD_EXPOSURE, seasons=(2009, 2025)
        )

    # Restricted population, one-sided (team-identity flag: home/away is ARI).
    for name, team_col in (
        ("dst_arizona_home_shield", "home_team"),
        ("dst_arizona_away_shield", "away_team"),
    ):
        pop = dst_cells[name]["population"]
        pop_df = dst_df.loc[pop].reset_index(drop=True)
        flag_in_pop = dst_cells[name]["flag"].loc[pop].reset_index(drop=True)
        long = one_sided_exposure_long(pop_df, team_col, flag_in_pop)
        out[name] = _measure_with_compositional_check(
            long, "exposure", method=METHOD_EXPOSURE, seasons=(2009, 2025)
        )

    # Restricted population (eastbound-only games), symmetric (D1 calendar window
    # applies to both teams in the game equally).
    name = "dst_transition_eastbound_interaction"
    pop = dst_cells[name]["population"]
    pop_df = dst_df.loc[pop].reset_index(drop=True)
    flag_in_pop = dst_cells[name]["flag"].loc[pop].reset_index(drop=True)
    long = game_flag_to_team_week(pop_df, flag_in_pop)
    out[name] = _measure_with_compositional_check(
        long, "exposure", method=METHOD_EXPOSURE, seasons=(2009, 2025)
    )
    return out


def measure_redteam_exposure(bc_df: pd.DataFrame, tr_df: pd.DataFrame, masks: dict) -> dict:
    ok_distance = redteam_distance_join(bc_df, tr_df)
    night_pop = masks["true_home"] & masks["night"] & ok_distance
    pop_df = bc_df.loc[night_pop].reset_index(drop=True)
    flag_in_pop = masks["away_west"].loc[night_pop].reset_index(drop=True)
    long = one_sided_exposure_long(pop_df, "away_team", flag_in_pop)
    return _measure_with_compositional_check(
        long, "exposure", method=METHOD_EXPOSURE, seasons=(2009, 2025)
    )


# ---------------------------------------------------------------------------
# 6. Entry -> flag / population, for half_season_replication (reported only)
# ---------------------------------------------------------------------------


def entry_flag_and_population(
    name: str,
    *,
    bc_df: pd.DataFrame,
    bc_cells: dict,
    tr_df: pd.DataFrame,
    tr_cells: dict,
    dst_df: pd.DataFrame,
    dst_cells: dict,
    masks: dict,
    ok_distance: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    if name in bc_cells:
        return bc_df, bc_cells[name]["flag"]
    if name in tr_cells:
        return tr_df, tr_cells[name]["flag"]
    if name in dst_cells:
        pop = dst_cells[name]["population"]
        pop_df = dst_df.loc[pop].reset_index(drop=True)
        flag_in_pop = dst_cells[name]["flag"].loc[pop].reset_index(drop=True)
        return pop_df, flag_in_pop
    if name == "redteam_body_clock_night_increment_within_west_road_lpm":
        west_road_pop = masks["true_home"] & masks["away_west"] & ok_distance
        pop_df = bc_df.loc[west_road_pop].reset_index(drop=True)
        flag_in_pop = masks["night"].loc[west_road_pop].reset_index(drop=True)
        return pop_df, flag_in_pop
    if name == "redteam_body_clock_night_west_road_distance_controlled_lpm":
        night_pop = masks["true_home"] & masks["night"] & ok_distance
        pop_df = bc_df.loc[night_pop].reset_index(drop=True)
        flag_in_pop = masks["away_west"].loc[night_pop].reset_index(drop=True)
        return pop_df, flag_in_pop
    raise KeyError(name)


BATTERY_OF = {}
for _n in (
    "body_clock_east_host_west_visitor_early",
    "body_clock_night_dose_1300",
    "body_clock_night_dose_1400_1659",
    "body_clock_night_dose_1700_1959",
    "body_clock_night_dose_ge2000",
    "body_clock_night_east_road_ge2000et",
    "body_clock_night_west_road_ge2000et",
    "body_clock_night_west_road_ge2000et_2009_2016",
    "body_clock_night_west_road_ge2000et_2017_2025",
    "body_clock_night_west_road_true_slots",
    "body_clock_west_host_east_visitor_late",
    "body_clock_west_road_early",
    "body_clock_west_road_early_2009_2016",
    "body_clock_west_road_early_2017_2025",
    "body_clock_west_road_midday_control",
):
    BATTERY_OF[_n] = "body_clock"
for _n in (
    "redteam_body_clock_night_increment_within_west_road_lpm",
    "redteam_body_clock_night_west_road_distance_controlled_lpm",
):
    BATTERY_OF[_n] = "redteam"
for _n in (
    "travel_rest_away_off_bye",
    "travel_rest_eastbound_multizone",
    "travel_rest_home_off_bye",
    "travel_rest_international_game",
    "travel_rest_long_distance_road",
    "travel_rest_return_trip_hangover",
    "travel_rest_short_week_road",
    "travel_rest_thursday_pure",
):
    BATTERY_OF[_n] = "travel_rest"
for _n in (
    "dst_arizona_away_shield",
    "dst_arizona_home_shield",
    "dst_fall_transition_shock",
    "dst_placebo_shifted_window",
    "dst_transition_eastbound_interaction",
):
    BATTERY_OF[_n] = "dst"


# ---------------------------------------------------------------------------
# 7. Main sweep
# ---------------------------------------------------------------------------


def run_sweep() -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    entries = load_manifest_entries()
    assert len(entries) == 30, f"expected 30 manifest entries, got {len(entries)}"

    print("=== loading populations ===")
    bc_df = load_body_clock_df()
    tr_df = load_travel_rest_df()
    dst_df = load_dst_df()
    print(
        f"body_clock population: {len(bc_df)} games; travel_rest: {len(tr_df)}; dst: {len(dst_df)}"
    )

    bc_cells = body_clock_cells(bc_df)
    tr_cells = trbs.build_cells(tr_df)
    dst_cells = dsts.build_cells(dst_df)
    masks = redteam_masks(bc_df)
    ok_distance = redteam_distance_join(bc_df, tr_df)

    season_ranges = {tuple(e["seasons"]) for e in entries if e["name"] in bc_cells}
    season_ranges.add((2009, 2025))  # redteam increment entry, shares the family
    print(f"kick_min season ranges to measure: {sorted(season_ranges)}")
    kick_min_by_range = measure_kick_min_family(bc_df, season_ranges)

    travel_rest_traits = measure_travel_rest_traits(tr_df)
    travel_rest_exposures = measure_travel_rest_exposures(tr_df, tr_cells)
    dst_exposures = measure_dst_exposures(dst_df, dst_cells)
    redteam_exposure = measure_redteam_exposure(bc_df, tr_df, masks)

    # entry name -> (measurement dict, parent-quantity label, method-tag)
    measurement_of: dict[str, tuple[dict, str, str]] = {}
    for e in entries:
        name = e["name"]
        if name in bc_cells:
            key = tuple(e["seasons"])
            measurement_of[name] = (kick_min_by_range[key], "kick_min", "trait")
        elif name == "redteam_body_clock_night_increment_within_west_road_lpm":
            measurement_of[name] = (kick_min_by_range[(2009, 2025)], "kick_min", "trait")
        elif name == "redteam_body_clock_night_west_road_distance_controlled_lpm":
            measurement_of[name] = (redteam_exposure, "west_body_clock_visitor (flag)", "exposure")
        elif name == "travel_rest_long_distance_road":
            measurement_of[name] = (travel_rest_traits["away_travel_mi"], "away_travel_mi", "trait")
        elif name == "travel_rest_eastbound_multizone":
            measurement_of[name] = (
                travel_rest_traits["tz_delta_eastbound"],
                "tz_delta_eastbound",
                "trait",
            )
        elif name == "travel_rest_home_off_bye":
            measurement_of[name] = (travel_rest_traits["home_rest"], "home_rest", "trait")
        elif name in ("travel_rest_away_off_bye", "travel_rest_short_week_road"):
            measurement_of[name] = (travel_rest_traits["away_rest"], "away_rest", "trait")
        elif name == "travel_rest_return_trip_hangover":
            measurement_of[name] = (
                travel_rest_traits["prev_own_travel_mi"],
                "prev_own_travel_mi",
                "trait",
            )
        elif name in ("travel_rest_international_game", "travel_rest_thursday_pure"):
            measurement_of[name] = (
                travel_rest_exposures[name],
                f"{name} (flag)",
                "exposure",
            )
        elif name in dst_exposures:
            measurement_of[name] = (dst_exposures[name], f"{name} (flag)", "exposure")
        else:
            raise KeyError(f"no measurement dispatch for {name}")

    assert len(measurement_of) == 30

    rows = []
    replications = {}
    for e in entries:
        name = e["name"]
        meas, parent_quantity, method_tag = measurement_of[name]
        pop_df, flag = entry_flag_and_population(
            name,
            bc_df=bc_df,
            bc_cells=bc_cells,
            tr_df=tr_df,
            tr_cells=tr_cells,
            dst_df=dst_df,
            dst_cells=dst_cells,
            masks=masks,
            ok_distance=ok_distance,
        )
        repl = half_season_replication(pop_df, flag, outcome_col="home_cover")
        replications[name] = repl
        rows.append(
            {
                "name": name,
                "parent_quantity": parent_quantity,
                "method_tag": method_tag,
                "seasons": e["seasons"],
                "n_units": meas["n_units"],
                "status": meas["status"],
                "reliability": meas["reliability"],
                "reliability_low": meas["reliability_low"],
                "reliability_high": meas["reliability_high"],
                "probability_positive": meas.get("probability_positive"),
                "pearson_r": meas.get("pearson_r"),
                "method": meas["method"],
                "compositional_check": meas.get("compositional_check"),
                "half_season_replication": repl,
            }
        )
        rel_str = f"{meas['reliability']:+.4f}" if meas["reliability"] is not None else "  None"
        print(
            f"{name:<62} status={meas['status']:<32} "
            f"n_units={meas['n_units']:<5} reliability={rel_str}"
        )

    batteries: dict[str, dict] = {}
    for battery in ("body_clock", "redteam", "travel_rest", "dst"):
        cells = {n: replications[n] for n in BATTERY_OF if BATTERY_OF[n] == battery}
        batteries[battery] = battery_replication_correlation(cells)

    print("\n=== positive control ===")
    kick_min_long = two_sided_long(bc_df, "kick_min")
    away_rest_long = one_sided_long(tr_df, "away_team", "away_rest")
    ari_home_pop = dst_cells["dst_arizona_home_shield"]["population"]
    ari_home_pop_df = dst_df.loc[ari_home_pop].reset_index(drop=True)
    ari_home_long = one_sided_long(ari_home_pop_df, "home_team", "home_rest")
    pc = {
        "kick_min_two_sided_full_population": positive_control(kick_min_long),
        "away_rest_one_sided_travel_rest_population": positive_control(away_rest_long),
        "dst_arizona_home_shield_restricted_population": positive_control(
            ari_home_long, min_units=1
        ),
    }
    for label, table in pc.items():
        print(f"  {label}:")
        for row in table:
            print(
                f"    planted={row['planted_unit_variance_share']:.1f} "
                f"status={row['status']} n_units={row['n_units']} "
                f"recovered={row['recovered_reliability']}"
            )

    n_measured = sum(1 for r in rows if r["status"] == STATUS_MEASURED)
    n_insufficient = sum(1 for r in rows if r["status"] == "insufficient_split_units")
    n_compositional = sum(1 for r in rows if r["status"] == STATUS_NOT_APPLICABLE_COMPOSITIONAL)
    n_other_skip = sum(
        1
        for r in rows
        if r["status"]
        not in (STATUS_MEASURED, "insufficient_split_units", STATUS_NOT_APPLICABLE_COMPOSITIONAL)
    )
    print(
        f"\n{n_measured} measured, {n_insufficient} insufficient_split_units, "
        f"{n_compositional} not_applicable_compositional_constraint, "
        f"{n_other_skip} other-skip, of {len(rows)}"
    )

    configuration = {
        "command": "reliability-schedule-clock",
        "seed": RELIABILITY_SEED,
        "n_boot": N_BOOT,
        "min_units": MIN_UNITS,
        "methods": [METHOD_TRAIT, METHOD_EXPOSURE],
        "entries": sorted(measurement_of),
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "seed": RELIABILITY_SEED,
        "group": "schedule_clock",
        "n_entries": len(rows),
        "n_measured": n_measured,
        "n_insufficient_split_units": n_insufficient,
        "n_not_applicable_compositional_constraint": n_compositional,
        "n_other_skip": n_other_skip,
        "rows": rows,
        "battery_replication_correlation": batteries,
        "positive_control": pc,
        "kick_min_provenance": KICK_MIN_PROVENANCE,
        "builder_provenance": {
            "body_clock_*": "scripts/body_clock_screen.py:build_cells (imported)",
            "body_clock_night_*": "scripts/body_clock_night_screen.py:build_cells (imported)",
            "redteam_body_clock_night_*": (
                "scripts/edge_audit_redteam.py:run_claim3 (population reproduced from "
                "the same masks; read scripts/edge_audit_redteam.py:596-652); "
                "west_night mask verified equal to body_clock_night_screen.build_cells"
                "['body_clock_night_west_road_ge2000et']['flag'] in "
                "tests/test_reliability_schedule_clock.py"
            ),
            "travel_rest_*": "scripts/nfl_travel_rest_battery_screen.py:build_cells (imported)",
            "dst_*": "scripts/dst_transition_battery_screen.py:build_cells (imported)",
        },
        "provenance": artifact_provenance(
            configuration, bcs.default_schedules(), project_root=REPO
        ),
    }
    return payload, configuration


def write_artifact(payload: dict[str, Any], configuration: dict[str, Any]) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = REPO / "artifacts" / "reliability_sweep" / "schedule_clock" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_artifact(
        out_dir,
        "results.json",
        payload,
        command=configuration["command"],
        metrics={
            "n_entries": payload["n_entries"],
            "n_measured": payload["n_measured"],
            "n_unmeasured": payload["n_entries"] - payload["n_measured"],
        },
        notes=(
            "Measure-only split-half reliability for the schedule_clock registry cells "
            "(body clock, travel/rest, DST transition); every cell measured regardless of "
            "sign or interval shape, and nothing is closed or reclassified, per AGENTS.md's "
            "binding closing-grounds taxonomy."
        ),
    )
    return out_dir / "results.json"


def main() -> None:
    payload, configuration = run_sweep()
    out_path = write_artifact(payload, configuration)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
