"""Split-half reliability for the 59 ``modeling_overlay`` registry cells (ORCH-D).

**What makes this group different from every other group in the sweep.** The
other ten groups are FEATURE families: a cell names a continuous team-week
column and the column's split-half reliability is the cell's reliability. Most
of the cells here are MODEL-vs-MODEL paired deltas -- "does half-life 8 beat
uniform weighting", "does alpha=2,000 beat alpha=10", "does dropping this
overlay member from the played union change the card". A paired delta between
two models is not a trait, so it has no split-half reliability *in the sense
the registry field means*, and inventing one would be worse than leaving the
field null.

So every cell here is first given a DISPOSITION, and the disposition is the
finding:

``a_trait``
    The cell thresholds or ranks by ONE named continuous quantity (the
    division-revenge gap, market spread magnitude, predicted line-movement
    magnitude, the Best-Pick ranker's own score). That quantity is measured
    with :data:`reliability_lib.METHOD_TRAIT` on the cell's OWN seasons and
    recorded.

``b_exposure``
    The cell's parent is CATEGORICAL with no continuous team-week parent (a
    referee's career stage, a year-one head coach, a recent player arrest).
    What is measurable is the flag's per-team-season EXPOSURE RATE, recorded
    with :data:`reliability_lib.METHOD_EXPOSURE` and flagged as exposure --
    which, per that constant's own docstring, is NOT an admissible
    ``no_split_half_reliability`` ground.

``c_no_trait``
    A pure model / composition / weighting-scheme comparison, or a block of
    two or more distinct parent columns. ``not_applicable: no underlying
    trait to be reliable``. **This is not a closure and not a negative about
    the signal.** It says the reliability instrument is the wrong instrument;
    what these cells need is an out-of-window replication (does the same
    paired delta reappear on seasons the comparison never saw), which is a
    different measurement this sweep does not perform.

**Three ways a number can be produced and still be uninformative.** Each is
detected, reported with its numbers, and NOT recorded:

``not_applicable_compositional_constraint``
    The parent's season TOTAL is conserved (days of rest is the canonical
    case: a season is a fixed number of days, so more rest in one half
    mechanically forces less in the other). Such a quantity returns a strongly
    negative correlation under ANY split. Detected by re-running the
    measurement with the week column replaced by random integers: a real trait
    stays positive under random halves, a conserved one stays negative.

``not_applicable_unit_constant``
    The parent is constant within a team-season by construction (a
    season-lagged rate, a year-one-coach flag), so odd-week and even-week
    means are the same number and the correlation is 1.0 tautologically.

``not_informative_near_constant``
    One value covers essentially the whole column, so |r| is a function of a
    handful of rows and flips with the season window.

**Binding taxonomy (verbatim, AGENTS.md / CLAUDE.md).** An interval or CI that
contains zero is NEVER grounds to reject, fail, or close an experiment. At
this evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome
for a real small signal. Only two grounds ever close a line of work: (1)
refuted mechanism -- a RESOLVED wrong sign (whole interval on the wrong side
of zero) or zero split-half reliability; (2) bounded by a positive control
proven able to detect an effect that size. Everything else is
``unresolved_below_power``; report ``probability_positive``, never the binary
"contains zero". This script CLOSES NOTHING, reclassifies nothing, and
proposes no ``closing_ground``: it measures. Within-week correlation is ZERO.

Writes ``artifacts/reliability_sweep/modeling_overlay/<stamp>/results.json``
and prints the ``set-reliability`` commands the caller runs through the
cross-process lock. This script never writes the registry itself.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _extra in (REPO / "src", REPO / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import reliability_lib as rlib  # noqa: E402
import reliability_map as relmap  # noqa: E402

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.weak_signals import default_registry_path, load_registry  # noqa: E402

DISPOSITION_TRAIT = "a_trait"
DISPOSITION_EXPOSURE = "b_exposure"
DISPOSITION_NO_TRAIT = "c_no_trait"

GUARD_COMPOSITIONAL = "not_applicable_compositional_constraint"
GUARD_UNIT_CONSTANT = "not_applicable_unit_constant"
GUARD_NEAR_CONSTANT = "not_informative_near_constant"

#: A single value covering this share of the column makes |r| a function of a
#: handful of rows; the orchestrator measured such columns flipping sign with
#: the season window, so they are reported, never recorded.
NEAR_CONSTANT_MODAL_SHARE = 0.98

#: Share of units whose within-unit variance is exactly zero above which the
#: odd/even split is tautological (both halves are the same number).
UNIT_CONSTANT_SHARE = 0.995

#: Mean random-halves correlation at or below which a negative measurement is
#: read as a compositional constraint rather than a trait property.
COMPOSITIONAL_RANDOM_HALVES_MAX = -0.30

#: Reseeds for the random-halves check (cheap; each is one bootstrap-light run).
RANDOM_HALVES_RESEEDS = 12
RANDOM_HALVES_N_BOOT = 200

DEFAULT_PER_GAME = REPO / "artifacts/opener_evaluation/20260819T174244Z/per_game.parquet"
RIDGE_ALPHA_ARTIFACT = REPO / "artifacts/ridge_alpha_promotion/20260818T221459Z"
MICROSTRUCTURE_ARTIFACT = REPO / "artifacts/odds_microstructure/20260818T225430Z"
BEST_PICK_PICKS = REPO / "artifacts/best_pick_ranker/screen_2013_2015.picks.parquet"
MOVEMENT_PER_GAME = REPO / "artifacts/movement_tilt_screen/20260819T160330Z/per_game.csv"
GAME_FEATURES = REPO / "data/processed/game_features.parquet"


# ---------------------------------------------------------------------------
# Cell table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One registry cell, its disposition, and how to measure it (or why not)."""

    entry: str
    disposition: str
    #: For (a)/(b): the parent quantity's plain name. For (c): the reason there
    #: is no single parent trait, in one sentence.
    parent: str
    #: Where the mapping was read from, file:line.
    provenance: str
    frame: str | None = None
    metric: str | None = None
    method: str | None = None
    unit_col: str = "team_id"
    #: ``(frame, metric)`` pairs measured and REPORTED (never recorded) so a
    #: block cell's information is not lost.
    members: tuple[tuple[str, str], ...] = ()
    #: Names of other rows in this run that already carry the member parents'
    #: numbers, for composition cells.
    member_rows: tuple[str, ...] = ()
    #: Always run the random-halves conserved-quantity check for this parent.
    conserved_candidate: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


_TRAIT = rlib.METHOD_TRAIT
_EXPOSURE = rlib.METHOD_EXPOSURE
_VENUE = rlib.METHOD_VENUE

_P_LOO = (
    "scripts/overlay_leave_one_out.py:86-92 (VARIANTS) -- each cell drops exactly one "
    "named member from PLAYED_UNION, so the cell's parent is that member's own trigger "
    "quantity. Read 2026-09-01."
)
_P_SUBSET = (
    "scripts/overlay_subset_composition.py / scripts/overlay_subset_holdout_v2.py "
    "(PLAYED_UNION, union_delta) -- an OR-composition over >=2 members. Read 2026-09-01."
)
_P_SPREAD_GAP = (
    "src/nfl_ats/spread_gap_zone_fade_overlay.py:126-127,195-204 -- the overlay's only "
    "trigger is SPREAD_GAP_LOWER_BOUND <= abs(spread_line) <= SPREAD_GAP_UPPER_BOUND, so "
    "the parent quantity is |spread_line|. Read 2026-09-01."
)
_P_REVENGE = (
    "src/nfl_ats/division_revenge_tilt_overlay.py:99-179 (division_revenge_side_by_game, "
    "ported from nfl_ats.experiment_runner._flag_division_revenge_game) with the continuous "
    "parent column gap_division_revenge_home/away carried by "
    "data/processed/game_features_weak_stack_v3.parquet "
    "(scripts/reliability_map.py:86-91 V3_ONLY_PAIR_BASES). Read 2026-09-01."
)
_P_MOVEMENT = (
    "artifacts/movement_tilt_screen/20260819T160330Z/per_game.csv columns "
    "predicted_close_minus_open / abs_predicted / confidence_threshold_median / "
    "confidence_threshold_q75 -- the tilt's gate is on abs_predicted. Read 2026-09-01."
)
_P_BESTPICK_2013 = (
    "scripts/best_pick_ranker.py:40,127-144 (SIGNALS; calibrated_probability and "
    "key_number_distance columns) and artifacts/best_pick_ranker/screen_2013_2015.picks.parquet "
    "which stores both columns per game. Read 2026-09-01."
)
_P_BESTPICK_OPENER = (
    "scripts/best_pick_opener_ranker_eval.py:80-127,170-176 -- candidate_dist = "
    "|candidate_prob_open - 0.5| from artifacts/ridge_alpha_promotion/20260818T221459Z/"
    "opener_paired.parquet; spread_std from artifacts/odds_microstructure/20260818T225430Z/"
    "spread_novig_tue_open.parquet. Read 2026-09-01."
)
_P_ERA_WEIGHTING = (
    "scripts/era_weighting_lib.py / scripts/era_weighting_nfl_screen.py -- the arm changes "
    "the TRAINING SAMPLE WEIGHT (or truncates the training window); no feature column "
    "differs between arms. Read 2026-09-01."
)
_P_FLAG_BUILDER = "src/nfl_ats/experiment_runner.py FLAG_BUILDERS. Read 2026-09-01."

CELLS: tuple[Cell, ...] = (
    # ---- best-pick rankers -------------------------------------------------
    Cell(
        "best_pick_calibrated_probability_top1",
        DISPOSITION_TRAIT,
        "Platt-calibrated pick-side cover probability (the ranker's own score)",
        _P_BESTPICK_2013,
        frame="bestpick2013",
        metric="calibrated_probability",
        method=_TRAIT,
    ),
    Cell(
        "best_pick_key_number_distance_top1",
        DISPOSITION_TRAIT,
        "distance-from-key-number (3, 7) of the market line minus the fair line",
        _P_BESTPICK_2013,
        frame="bestpick2013",
        metric="key_number_distance",
        method=_TRAIT,
    ),
    Cell(
        "best_pick_opener_ranker_candidate_prob_distance_vs_status_quo",
        DISPOSITION_TRAIT,
        "candidate (alpha=2000) opener cover-probability distance from 0.5",
        _P_BESTPICK_OPENER,
        frame="opener",
        metric="candidate_dist",
        method=_TRAIT,
    ),
    Cell(
        "best_pick_opener_ranker_candidate_prob_distance_vs_live_v2",
        DISPOSITION_TRAIT,
        "candidate (alpha=2000) opener cover-probability distance from 0.5",
        _P_BESTPICK_OPENER,
        frame="opener",
        metric="candidate_dist",
        method=_TRAIT,
    ),
    Cell(
        "best_pick_opener_ranker_dispersion_filtered_candidate_vs_unfiltered",
        DISPOSITION_TRAIT,
        "cross-book Tuesday-opener spread dispersion (spread_std), the filter that "
        "distinguishes this arm from its unfiltered parent",
        _P_BESTPICK_OPENER,
        frame="opener",
        metric="spread_std",
        method=_TRAIT,
    ),
    Cell(
        "best_pick_opener_ranker_dispersion_filtered_candidate_vs_live_v2",
        DISPOSITION_TRAIT,
        "cross-book Tuesday-opener spread dispersion (spread_std), the quantity both the "
        "filter and the live rule's tie-break are built on",
        _P_BESTPICK_OPENER,
        frame="opener",
        metric="spread_std",
        method=_TRAIT,
    ),
    # ---- combined / max-EV stacks -----------------------------------------
    Cell(
        "combined_stacker_opener_2022_2023",
        DISPOSITION_NO_TRAIT,
        "A four-column stacker (injury value-lost narrowed pair + two forecast-weather "
        "key-number cells + the spread-gap zone) scored as one model-vs-model delta -- four "
        "distinct parents, so no single trait can occupy a single-valued field.",
        "artifacts/combined_stacker_look/result.json; scripts/combined_stacker_look.py. "
        "Read 2026-09-01.",
        member_rows=("overlay_subset_production_plus_spread_gap_zone",),
        members=(
            ("v34", "injury_skill_epa_value_lost"),
            ("v34", "injury_defense_disruption_value_lost"),
        ),
    ),
    # ---- era trend ---------------------------------------------------------
    Cell(
        "era_trend_extra_rest_edge",
        DISPOSITION_TRAIT,
        "rest differential (own_rest - opp_rest), the quantity extra_rest_edge thresholds at >= 4",
        "src/nfl_ats/experiment_runner.py:1082-1096 (_flag_extra_rest_edge: flag = "
        "own_rest - opp_rest >= 4). Read 2026-09-01.",
        frame="rest_differential",
        metric="rest_differential",
        method=_TRAIT,
        conserved_candidate=True,
    ),
    Cell(
        "era_trend_home_underdog",
        DISPOSITION_EXPOSURE,
        "home_underdog flag (is_home AND spread_line < 0) -- a conjunction of a schedule "
        "assignment and a market threshold, with no single continuous parent",
        "src/nfl_ats/experiment_runner.py:592-610 (_flag_home_underdog; the builder itself "
        "declares reliability=None, 'not a persistent per-team trait'). Read 2026-09-01.",
        frame="exposure:home_underdog",
        metric="exposure",
        method=_EXPOSURE,
    ),
    Cell(
        "era_trend_production_model_opener_proxy_edge",
        DISPOSITION_NO_TRAIT,
        "The frozen production model's own season-by-season accuracy edge over a coin "
        "flip -- a property of the MODEL's output, not of any team-week quantity.",
        "scripts/era_magnitude_profile.py; docs/era_magnitude_profile.md; "
        "scripts/reliability_map.py:543 already files this cell as "
        "'model-level accuracy trend, not a feature'. Read 2026-09-01.",
    ),
    # ---- era weighting -----------------------------------------------------
    *(
        Cell(
            name,
            DISPOSITION_NO_TRAIT,
            "A TRAINING-SAMPLE-WEIGHTING scheme (exponential season decay at a given "
            "half-life, or a truncated rolling window) against a uniform-weight baseline "
            "on an otherwise frozen recipe -- the two arms share every feature column, so "
            "there is no parent trait that differs between them.",
            _P_ERA_WEIGHTING,
        )
        for name in (
            "era_weighting_half_life_8_opener_confirmation",
            "era_weighting_nfl_half_life_16",
            "era_weighting_nfl_half_life_2",
            "era_weighting_nfl_half_life_4",
            "era_weighting_nfl_half_life_8",
            "era_weighting_nfl_half_life_8_opener",
            "era_weighting_nfl_rolling_10",
            "era_weighting_nfl_rolling_6",
        )
    ),
    # ---- availability / shrinkage blocks ----------------------------------
    Cell(
        "learned_availability_ats_2018_2025",
        DISPOSITION_NO_TRAIT,
        "Fixed-vs-learned player-availability SEMANTICS, which move all nine injury "
        "columns at once (seven unavailability shares plus two value-lost columns) -- a "
        "block of nine distinct parents, not one trait.",
        "data/processed/game_features_player_learned_availability.parquet vs "
        "game_features_player.parquet (identical injury_* column sets, different "
        "availability-rate semantics); docs/availability_confirmation.md sec 1.2 (M4). "
        "Read 2026-09-01.",
        members=(
            ("v34", "injury_offense_unavailability"),
            ("v34", "injury_defense_unavailability"),
            ("v34", "injury_special_teams_unavailability"),
            ("v34", "injury_offensive_line_unavailability"),
            ("v34", "injury_skill_unavailability"),
            ("v34", "injury_front_unavailability"),
            ("v34", "injury_secondary_unavailability"),
            ("v34", "injury_skill_epa_value_lost"),
            ("v34", "injury_defense_disruption_value_lost"),
        ),
        extra={"member_seasons": [2018, 2025]},
    ),
    Cell(
        "maxev_full_stack",
        DISPOSITION_NO_TRAIT,
        "A four-edge stack (chain + movement + NFL.com out>=2 + protection tilt) applied "
        "slate-wide against the incumbent chain -- four distinct parents.",
        "artifacts/max_ev_composition/20260823T024809Z/metadata.json; "
        "scripts/max_ev_composition.py. Read 2026-09-01.",
        member_rows=("movement_direction_tilt_opener",),
    ),
    Cell(
        "mod06_position_prior_shrinkage",
        DISPOSITION_NO_TRAIT,
        "Shrinking TWO player-value features (injury_skill_epa_value_lost and "
        "injury_defense_disruption_value_lost) toward a position/channel prior instead of "
        "toward zero -- the arm changes both columns symmetrically, so neither is 'the' "
        "parent; both members' numbers are reported below.",
        "registry/experiment_specs/mod06_position_prior_shrinkage.json; "
        "scripts/mod06_position_prior_shrinkage_build.py. Read 2026-09-01.",
        members=(
            ("v34", "injury_skill_epa_value_lost"),
            ("v34", "injury_defense_disruption_value_lost"),
        ),
        extra={"member_seasons": [2018, 2025]},
    ),
    Cell(
        "mod07_holdover_bias_replication",
        DISPOSITION_TRAIT,
        "bias_playoff_holdover (the Week-1 playoff-holdover favourite marker) as a "
        "team-week column",
        "data/processed/game_features_weak_stack_v4.parquet columns "
        "bias_playoff_holdover_home/away; docs/pool_edge_plan.md:236-238 names no surviving "
        "script. Read 2026-09-01.",
        frame="v34",
        metric="bias_playoff_holdover",
        method=_TRAIT,
    ),
    Cell(
        "mod07_opener_bias_ablation",
        DISPOSITION_NO_TRAIT,
        "A JOINT ablation of three opener-bias feature columns (playoff holdover, "
        "prior-week ATS, week-2 anchoring) -- three distinct parents removed together; the "
        "three member numbers are reported below.",
        "artifacts/availability_experiments/mod07_ablation_2020_2021.json "
        "(contrast C_minus_B_opener_bias); the three columns are "
        "bias_playoff_holdover / bias_prior_week_ats / bias_week2_anchor in "
        "game_features_weak_stack_v4.parquet. Read 2026-09-01.",
        members=(
            ("v34", "bias_playoff_holdover"),
            ("v34", "bias_prior_week_ats"),
            ("v34", "bias_week2_anchor"),
        ),
        extra={"member_seasons": [2020, 2021]},
    ),
    # ---- movement tilt -----------------------------------------------------
    Cell(
        "movement_direction_tilt_opener",
        DISPOSITION_TRAIT,
        "predicted line-movement MAGNITUDE (abs_predicted), the quantity the primary "
        "rule's median confidence gate is applied to",
        _P_MOVEMENT,
        frame="movement",
        metric="abs_predicted",
        method=_TRAIT,
    ),
    Cell(
        "movement_direction_tilt_opener_variant2_top_quartile",
        DISPOSITION_TRAIT,
        "predicted line-movement MAGNITUDE (abs_predicted), the quantity the 75th-"
        "percentile confidence gate is applied to",
        _P_MOVEMENT,
        frame="movement",
        metric="abs_predicted",
        method=_TRAIT,
    ),
    Cell(
        "movement_direction_tilt_opener_variant1_no_filter",
        DISPOSITION_TRAIT,
        "predicted SIGNED line movement (predicted_close_minus_open, team-signed: +home, "
        "-away) -- with the confidence gate set to 0 the direction is the only quantity "
        "left in the rule",
        _P_MOVEMENT,
        frame="movement",
        metric="predicted_move_team_signed",
        method=_TRAIT,
    ),
    # ---- overlay leave-one-out --------------------------------------------
    Cell(
        "overlay_loo_drop_coach_fade",
        DISPOSITION_EXPOSURE,
        "year-one head-coach flag (this season's coach differs from the prior season's "
        "primary coach) -- categorical, no continuous parent",
        "src/nfl_ats/coach_fade_overlay.py:150-194 (year_one_by_game). " + _P_LOO,
        frame="exposure:coach_year_one",
        metric="exposure",
        method=_EXPOSURE,
    ),
    Cell(
        "overlay_loo_drop_division_revenge_tilt",
        DISPOSITION_TRAIT,
        "division-revenge gap (gap_division_revenge), the continuous parent of the "
        "revenge-side flag the overlay tilts to",
        _P_REVENGE + " " + _P_LOO,
        frame="v34",
        metric="gap_division_revenge",
        method=_TRAIT,
    ),
    Cell(
        "overlay_loo_drop_player_arrests_back_side_policy",
        DISPOSITION_EXPOSURE,
        "recent-player-arrest side flag (a broad incident 1-14 days before the Tuesday "
        "decision date) -- categorical, no continuous parent",
        "src/nfl_ats/player_arrests_back_side_overlay.py:43,202-266 (WINDOW_DAYS=14, "
        "_broad_side_flags), same construct as FLAG_BUILDERS['recent_player_arrest'] "
        "(src/nfl_ats/experiment_runner.py:643-683, default window_days=14 and the same "
        "incidents snapshot). " + _P_LOO,
        frame="exposure:recent_player_arrest",
        metric="exposure",
        method=_EXPOSURE,
    ),
    Cell(
        "overlay_loo_drop_spread_gap_zone_fade",
        DISPOSITION_TRAIT,
        "market spread magnitude (|spread_line|), the overlay's only trigger quantity",
        _P_SPREAD_GAP + " " + _P_LOO,
        frame="market",
        metric="abs_spread_line",
        method=_TRAIT,
    ),
    # ---- overlay compositions ---------------------------------------------
    Cell(
        "overlay_single_addition_to_played_union_forward_holdout",
        DISPOSITION_NO_TRAIT,
        "A SELECTION DESIGN: one further overlay chosen from ten candidates on 2020-2022 "
        "and applied unchanged to 2023-2025 -- the construct is the selection procedure "
        "plus a five-member union, not a trait.",
        "scripts/overlay_subset_holdout_v2.py; artifacts/overlay_subset_holdout_v2/"
        "20260825T230706Z/result.json. Read 2026-09-01.",
        member_rows=(
            "overlay_loo_drop_division_revenge_tilt",
            "overlay_loo_drop_spread_gap_zone_fade",
            "overlay_loo_drop_coach_fade",
            "overlay_loo_drop_player_arrests_back_side_policy",
        ),
    ),
    Cell(
        "overlay_stack_combined_opener_2020_2025",
        DISPOSITION_NO_TRAIT,
        "OR-composition of six pick-flipping overlays on the frozen opener baseline -- six "
        "distinct parents fired jointly.",
        "artifacts/overlay_stack_backtest/20260819T191534Z/result.json; "
        "scripts/overlay_stack_backtest.py (OVERLAY_NAMES, run_overlays). Read 2026-09-01.",
        member_rows=(
            "overlay_loo_drop_division_revenge_tilt",
            "overlay_loo_drop_spread_gap_zone_fade",
            "overlay_loo_drop_coach_fade",
        ),
    ),
    Cell(
        "overlay_subset_all_seven_joint",
        DISPOSITION_NO_TRAIT,
        "OR-composition of all seven flips jointly -- seven distinct parents.",
        _P_SUBSET,
        member_rows=(
            "overlay_loo_drop_division_revenge_tilt",
            "overlay_loo_drop_spread_gap_zone_fade",
            "overlay_loo_drop_coach_fade",
            "overlay_loo_drop_player_arrests_back_side_policy",
        ),
    ),
    Cell(
        "overlay_subset_holdout_2020_2022_reverse",
        DISPOSITION_NO_TRAIT,
        "A split-half SELECTION-AND-HOLDOUT design over 127 overlay subsets -- the cell "
        "measures the design, not a trait.",
        "scripts/overlay_selection_holdout.py; artifacts/overlay_selection_holdout/"
        "20260821T195512Z/result.json; scripts/reliability_map.py:547 already files this "
        "cell as 'composition/holdout design, not a single feature'. Read 2026-09-01.",
        member_rows=(
            "overlay_loo_drop_coach_fade",
            "overlay_loo_drop_player_arrests_back_side_policy",
            "overlay_loo_drop_spread_gap_zone_fade",
        ),
    ),
    Cell(
        "overlay_subset_holdout_2023_2025_frozen",
        DISPOSITION_NO_TRAIT,
        "The forward arm of the same 127-subset selection-and-holdout design -- a design, "
        "not a trait.",
        "scripts/overlay_selection_holdout.py; artifacts/overlay_selection_holdout/"
        "20260821T195512Z/result.json. Read 2026-09-01.",
        member_rows=(
            "overlay_loo_drop_coach_fade",
            "overlay_loo_drop_division_revenge_tilt",
            "overlay_loo_drop_spread_gap_zone_fade",
        ),
    ),
    Cell(
        "overlay_subset_production_chain_coach_arrest",
        DISPOSITION_NO_TRAIT,
        "OR-composition of TWO members (coach fade plus the player-arrests back-side "
        "flip) -- two distinct categorical parents, so no single trait.",
        _P_SUBSET,
        member_rows=(
            "overlay_loo_drop_coach_fade",
            "overlay_loo_drop_player_arrests_back_side_policy",
        ),
    ),
    Cell(
        "overlay_subset_production_plus_division_revenge",
        DISPOSITION_TRAIT,
        "division-revenge gap (gap_division_revenge) -- the one member this composition "
        "ADDS to the production chain, and the only quantity that distinguishes it from "
        "overlay_subset_production_chain_coach_arrest",
        _P_REVENGE + " " + _P_SUBSET,
        frame="v34",
        metric="gap_division_revenge",
        method=_TRAIT,
    ),
    Cell(
        "overlay_subset_production_plus_spread_gap_zone",
        DISPOSITION_TRAIT,
        "market spread magnitude (|spread_line|) -- the one member this composition ADDS "
        "to the production chain",
        _P_SPREAD_GAP + " " + _P_SUBSET,
        frame="market",
        metric="abs_spread_line",
        method=_TRAIT,
    ),
    Cell(
        "overlay_subset_reselection_twelve_member_forward_holdout",
        DISPOSITION_NO_TRAIT,
        "Re-selection across all 4,095 subsets of twelve members, chosen on 2020-2022 and "
        "applied to 2023-2025 -- a selection procedure over twelve parents.",
        "scripts/overlay_subset_holdout_v2.py; artifacts/overlay_subset_holdout_v2/"
        "20260825T230706Z/result.json. Read 2026-09-01.",
        member_rows=(
            "overlay_loo_drop_division_revenge_tilt",
            "overlay_loo_drop_spread_gap_zone_fade",
            "overlay_loo_drop_coach_fade",
            "overlay_loo_drop_player_arrests_back_side_policy",
        ),
    ),
    # ---- pbp bundle --------------------------------------------------------
    Cell(
        "pbp_drive_bundle",
        DISPOSITION_NO_TRAIT,
        "A raw-PBP/drive-aggregate feature BUNDLE against base -- twelve distinct drive_* "
        "parents added together; every member's number is reported below.",
        "registry/rotation_registry.json (pbp_drive_bundle window); ROADMAP.md#RWB-16; the "
        "drive_* home/away families in game_features_weak_stack_v4.parquet. Read 2026-09-01.",
        members=(
            ("v34", "drive_plays_per_drive"),
            ("v34", "drive_plays_per_drive_allowed"),
            ("v34", "drive_points_per_drive"),
            ("v34", "drive_points_per_drive_allowed"),
            ("v34", "drive_scoring_rate"),
            ("v34", "drive_scoring_rate_allowed"),
            ("v34", "drive_seconds_per_drive"),
            ("v34", "drive_seconds_per_drive_allowed"),
            ("v34", "drive_takeaway_rate"),
            ("v34", "drive_turnover_rate"),
            ("v34", "drive_yards_per_drive"),
            ("v34", "drive_yards_per_drive_allowed"),
        ),
        extra={"member_seasons": [2013, 2017]},
    ),
    # ---- pick-conditioned screens -----------------------------------------
    Cell(
        "pick_conditioned_off_bye_fade_pre2018",
        DISPOSITION_TRAIT,
        "rest differential in the picked team's favour (rest_diff >= 6), whose parent "
        "quantity is own_rest - opp_rest",
        "scripts/pick_conditioned_pre2018_screen.py bucket "
        "'picked_team_off_bye (rest_diff>=6 in the picked team's favor)'; the rest "
        "differential itself is src/nfl_ats/experiment_runner.py:1082-1096's own_rest - "
        "opp_rest. Read 2026-09-01.",
        frame="rest_differential",
        metric="rest_differential",
        method=_TRAIT,
        conserved_candidate=True,
    ),
    Cell(
        "pick_conditioned_rest_mismatch_pre2018",
        DISPOSITION_TRAIT,
        "rest differential (rest_diff != 0 vs rest_diff == 0), parent quantity own_rest - opp_rest",
        "scripts/pick_conditioned_pre2018_screen.py bucket 'rest_diff != 0 (unequal rest) "
        "vs rest_diff == 0 (equal rest)'. Read 2026-09-01.",
        frame="rest_differential",
        metric="rest_differential",
        method=_TRAIT,
        conserved_candidate=True,
    ),
    Cell(
        "pick_conditioned_road_favorite_pre2018",
        DISPOSITION_EXPOSURE,
        "road-favorite flag (team is away AND favored). NOTE: the registry cell is "
        "PICK-CONDITIONED (our_pick_side=='AWAY'); the model's own pick is not a team "
        "quantity, so what is measured here is the market/schedule half of the conjunction "
        "only, and the exposure number is a lower-information stand-in for the cell's own "
        "population",
        "scripts/pick_conditioned_pre2018_screen.py bucket "
        "\"our_pick_side=='AWAY' and our_pick_is_favorite\"; team_spread sign convention "
        "from src/nfl_ats/experiment_runner.py:530 and :1108 (_flag_large_favorite). "
        "Read 2026-09-01.",
        frame="exposure:road_favorite",
        metric="exposure",
        method=_EXPOSURE,
    ),
    Cell(
        "pick_conditioned_spread_gap_zone_pre2018",
        DISPOSITION_TRAIT,
        "market spread magnitude (|spread_line|); the bucket is 7.0 < |spread_line| <= 10.0",
        "scripts/pick_conditioned_pre2018_screen.py bucket '7.0 < abs(spread_line) <= 10.0'. "
        + _P_SPREAD_GAP,
        frame="market",
        metric="abs_spread_line",
        method=_TRAIT,
    ),
    # ---- QB / lineup continuity blocks ------------------------------------
    Cell(
        "player_qb_continuity_bundled_alpha",
        DISPOSITION_NO_TRAIT,
        "A QB+lineup-continuity feature BLOCK compared at MISMATCHED ridge alpha -- the "
        "arms differ in both a nine-column block and a hyperparameter, so the cell has no "
        "single parent at all; member numbers are reported below.",
        "registry/rotation_registry.json (player_qb_continuity window); "
        "artifacts/qb_continuity_replication/20260816T143913Z. Read 2026-09-01.",
        members=(
            ("v34", "qb_expected_epa_per_dropback"),
            ("v34", "qb_starter_epa_per_dropback"),
            ("v34", "qb_starter_cpoe"),
            ("v34", "qb_start_probability"),
            ("v34", "qb_starter_experience_log"),
            ("v34", "offense_lineup_continuity"),
            ("v34", "defense_lineup_continuity"),
            ("v34", "offensive_line_continuity"),
            ("v34", "active_roster_continuity"),
        ),
        extra={"member_seasons": [2014, 2017]},
    ),
    Cell(
        "player_qb_continuity_matched_alpha",
        DISPOSITION_NO_TRAIT,
        "The same QB+lineup-continuity BLOCK at matched alpha -- nine distinct parents "
        "added together; member numbers reported below.",
        "artifacts/qb_continuity_replication/20260816T143913Z/paired_comparisons.csv; "
        "scripts/audit_terminal_verdicts.py (2026-08-18). Read 2026-09-01.",
        members=(
            ("v34", "qb_expected_epa_per_dropback"),
            ("v34", "qb_starter_epa_per_dropback"),
            ("v34", "qb_starter_cpoe"),
            ("v34", "qb_start_probability"),
            ("v34", "qb_starter_experience_log"),
            ("v34", "offense_lineup_continuity"),
            ("v34", "defense_lineup_continuity"),
            ("v34", "offensive_line_continuity"),
            ("v34", "active_roster_continuity"),
        ),
        extra={"member_seasons": [2014, 2017]},
    ),
    # ---- red team ----------------------------------------------------------
    Cell(
        "redteam_bye_fade_sham_placebo_null",
        DISPOSITION_NO_TRAIT,
        "A PLACEBO NULL DISTRIBUTION: 100 draws that deliberately shift each team's true "
        "bye by +/-2 weeks. The construct is the null distribution of a scrambled flag, "
        "so its stability is a property of the scramble, not of any signal.",
        "artifacts/edge_audit_redteam/20260822T040806Z/results.json; "
        "scripts/edge_audit_redteam.py. Read 2026-09-01.",
    ),
    Cell(
        "redteam_nflcom_out2_nonstarters_only",
        DISPOSITION_TRAIT,
        "count of NFL.com Friday 'Out' designations on NON-starter-caliber players "
        "(out_nonstarter), the continuous parent of the >=2-Out decomposition",
        "scripts/edge_audit_redteam.py:429-451 (run_claim2 builds out_starter / "
        "out_nonstarter per season-week-team from the Friday report rows and a >=50% "
        "prior-week snap-share starter proxy). Read 2026-09-01.",
        frame="nflcom",
        metric="out_nonstarter",
        method=_TRAIT,
    ),
    Cell(
        "redteam_nflcom_out2_starters_only",
        DISPOSITION_TRAIT,
        "count of NFL.com Friday 'Out' designations on starter-caliber players "
        "(out_starter), the continuous parent of the >=2-Out decomposition",
        "scripts/edge_audit_redteam.py:429-451 (run_claim2). Read 2026-09-01.",
        frame="nflcom",
        metric="out_starter",
        method=_TRAIT,
    ),
    Cell(
        "redteam_overlay_subset_loso_cv",
        DISPOSITION_NO_TRAIT,
        "Leave-one-season-out cross-validation of a SUBSET-SELECTION choice -- the cell "
        "measures a selection procedure over 127 subsets, not a trait.",
        "artifacts/edge_audit_redteam/20260822T040806Z/results.json "
        "(attack_leave_one_season_out_cv). Read 2026-09-01.",
        member_rows=(
            "overlay_loo_drop_coach_fade",
            "overlay_loo_drop_division_revenge_tilt",
            "overlay_loo_drop_spread_gap_zone_fade",
        ),
    ),
    # ---- referee battery ---------------------------------------------------
    Cell(
        "referee_battery_rookie_home_cover",
        DISPOSITION_EXPOSURE,
        "flag: the home team's head referee has 0 prior dataset-visible seasons. The "
        "parent is a REFEREE career-stage counter, and the spec itself declares it "
        "not_applicable for split-half; what is measurable on a team-season is the flag's "
        "exposure rate",
        "registry/experiment_specs/referee_battery_rookie_home_cover.json "
        "(reliability_check.method='not_applicable': 'prior_seasons_experience is a "
        "monotonically increasing career-stage counter'); builder "
        "src/nfl_ats/experiment_runner.py:1552-1566. Read 2026-09-01.",
        frame="exposure:referee_rookie_home_cover",
        metric="exposure",
        method=_EXPOSURE,
    ),
    Cell(
        "referee_battery_veteran_home_cover",
        DISPOSITION_EXPOSURE,
        "flag: the home team's head referee has >= 5 prior dataset-visible seasons. Same "
        "referee-level parent, same spec-declared not_applicable trait; exposure measured",
        "registry/experiment_specs/referee_battery_veteran_home_cover.json; builder "
        "src/nfl_ats/experiment_runner.py:1523-1538. Read 2026-09-01.",
        frame="exposure:referee_veteran_home_cover",
        metric="exposure",
        method=_EXPOSURE,
    ),
    # ---- hyperparameter ----------------------------------------------------
    Cell(
        "ridge_alpha_2000_nfl_opener_confirmation",
        DISPOSITION_NO_TRAIT,
        "A HYPERPARAMETER swap (ridge_alpha 10 -> 2,000) with every feature column held "
        "fixed -- nothing about the team-week inputs differs between the arms.",
        "scripts/ridge_alpha_promotion_eval.py; docs/ridge_alpha.md sec4; "
        "artifacts/ridge_alpha_promotion/20260818T221459Z. Read 2026-09-01.",
    ),
    Cell(
        "shrunk_overlay_policy_walkforward",
        DISPOSITION_NO_TRAIT,
        "A LEARNED WEIGHTING (ridge-logistic, LOSO-CV) over seven overlay flip indicators "
        "-- the construct is the learned policy over seven parents, not a trait.",
        "artifacts/shrunk_overlay_weights/20260822T035633Z/result.json; "
        "scripts/shrunk_overlay_weights.py. Read 2026-09-01.",
        member_rows=(
            "overlay_loo_drop_coach_fade",
            "overlay_loo_drop_division_revenge_tilt",
            "overlay_loo_drop_spread_gap_zone_fade",
            "overlay_loo_drop_player_arrests_back_side_policy",
        ),
    ),
    # ---- weak stack v2 / v3 ------------------------------------------------
    Cell(
        "weak_stack_v2_narrowed_only",
        DISPOSITION_NO_TRAIT,
        "Fixed-prior (not learned) injury severity, which moves the injury value-lost PAIR "
        "(skill EPA value lost and defense disruption value lost) together -- two distinct "
        "parents; both members' numbers are reported below.",
        "scripts/weak_stack_v2_eval.py; artifacts/weak_stack_v2/20260818T225248Z/; the "
        "narrowed construct is game_features_player.parquet's injury_*_value_lost pair. "
        "Read 2026-09-01.",
        members=(
            ("v34", "injury_skill_epa_value_lost"),
            ("v34", "injury_defense_disruption_value_lost"),
        ),
        extra={"member_seasons": [2020, 2025]},
    ),
    Cell(
        "weak_stack_v2_penalty_only",
        DISPOSITION_TRAIT,
        "penalty_rate_prior -- the team's PRIOR-season penalty rate, the parent of "
        "diff_penalty_rate_prior",
        "scripts/weak_stack_v2_eval.py:103-156 (team_season_penalty_rate + "
        "add_penalty_discipline_feature: prior_rate is joined on (team, season-1), so it "
        "is one number per team-season). Read 2026-09-01.",
        frame="penalty",
        metric="penalty_rate_prior",
        method=_TRAIT,
    ),
    Cell(
        "weak_stack_v2_stack",
        DISPOSITION_NO_TRAIT,
        "Two feature families stacked at once (fixed-prior injury severity PLUS the new "
        "season-lagged penalty-rate column) -- three distinct parents.",
        "scripts/weak_stack_v2_eval.py; artifacts/weak_stack_v2/20260818T225248Z/. "
        "Read 2026-09-01.",
        members=(
            ("v34", "injury_skill_epa_value_lost"),
            ("v34", "injury_defense_disruption_value_lost"),
            ("penalty", "penalty_rate_prior"),
        ),
        extra={"member_seasons": [2020, 2025]},
    ),
    Cell(
        "weak_stack_v3_nfl_opener_confirmation",
        DISPOSITION_NO_TRAIT,
        "FIFTEEN new gap columns added at once (division revenge, sandwich spot, "
        "post-blowout letdown/bounce, penalty rate, surface switch, thursday-pure, "
        "return-trip hangover) -- fifteen distinct parents; the four continuous gap_* "
        "members carried by the v3 table are reported below.",
        "scripts/weak_stack_v3_opener_eval.py; src/nfl_ats/weak_stack_v3_features.py; "
        "docs/weak_stack_v3.md; the gap_* home/away pairs come from "
        "game_features_weak_stack_v3.parquet (scripts/reliability_map.py:86-91). "
        "Read 2026-09-01.",
        members=(
            ("v34", "gap_division_revenge"),
            ("v34", "gap_sandwich_spot"),
            ("v34", "gap_post_blowout_win_letdown"),
            ("v34", "gap_post_blowout_loss_bounce"),
            ("penalty", "penalty_rate_prior"),
        ),
        extra={"member_seasons": [2020, 2025]},
    ),
    # ---- oracle ceiling ----------------------------------------------------
    Cell(
        "weather_oracle_ceiling_opener_probability_rule",
        DISPOSITION_NO_TRAIT,
        "A deliberately-leaked ORACLE control whose input is the weather that actually "
        "happened -- observed temperature AND wind, two VENUE-level quantities, not one "
        "team trait. Both members are reported below as venue-season measurements "
        "(METHOD_VENUE), which the sweep doc says is never a closing ground on its own.",
        "artifacts/weak_stack_oracle_weather_eval/20260826T005510Z/opener_summary.json; "
        "docs/weak_stack_v4.md; temp/wind are game/venue-level singletons in "
        "game_features_weak_stack_v4.parquet (scripts/reliability_map.py:50-52). "
        "Read 2026-09-01.",
        members=(("venue_weather", "temp"), ("venue_weather", "wind")),
        extra={"member_unit_col": "venue_id", "member_method": "venue"},
    ),
)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def _latest(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no match for {pattern} under {root}")
    return matches[-1]


class Frames:
    """Lazily-built team-week (or venue-week) frames, one per data source."""

    def __init__(self) -> None:
        self._cache: dict[str, pd.DataFrame] = {}
        self._games: dict[str, tuple[pd.DataFrame, pd.Series, str]] = {}
        self._identity: pd.DataFrame | None = None
        self._features: pd.DataFrame | None = None
        self._builders: dict[str, Callable[[], pd.DataFrame]] = {
            "v34": self._build_v34,
            "market": self._build_market,
            "bestpick2013": self._build_bestpick2013,
            "opener": self._build_opener,
            "movement": self._build_movement,
            "rest_differential": self._build_rest_differential,
            "nflcom": self._build_nflcom,
            "penalty": self._build_penalty,
            "venue_weather": self._build_venue_weather,
        }

    # -- helpers ---------------------------------------------------------
    def identity(self) -> pd.DataFrame:
        if self._identity is None:
            features = pd.read_parquet(
                GAME_FEATURES,
                columns=["game_id", "season", "week", "home_team", "away_team", "game_type"],
            )
            features = features.loc[features["game_type"] == "REG"].copy()
            features["game_id"] = features["game_id"].astype(str)
            self._identity = features
        return self._identity

    def features(self) -> pd.DataFrame:
        if self._features is None:
            self._features = pd.read_parquet(GAME_FEATURES)
        return self._features

    def get(self, name: str) -> pd.DataFrame:
        if name in self._cache:
            return self._cache[name]
        if name.startswith("exposure:"):
            frame = self._build_exposure(name.split(":", 1)[1])
        else:
            frame = self._builders[name]()
        self._cache[name] = frame
        return frame

    def games_for(self, name: str) -> tuple[pd.DataFrame, pd.Series, str] | None:
        """``(games, flag, outcome_col)`` for a cell's half-season replication."""

        self.get(name)
        return self._games.get(name)

    # -- builders --------------------------------------------------------
    def _build_v34(self) -> pd.DataFrame:
        features = relmap.load_feature_table()
        dtypes = {column: features[column].dtype for column in features.columns}
        families, _excluded = relmap.discover_family_pairs(list(features.columns), dtypes)
        return relmap.build_long_frame(features, families)

    def _build_market(self) -> pd.DataFrame:
        features = relmap.load_feature_table()
        reg = features.loc[features["game_type"] == "REG"].copy()
        reg["game_id"] = reg["game_id"].astype(str)
        spread = pd.to_numeric(reg["spread_line"], errors="coerce")
        frame = pd.DataFrame(
            {
                "game_id": reg["game_id"].to_numpy(),
                "abs_spread_line": spread.abs().to_numpy(),
                "team_spread_signed": spread.to_numpy(),
            }
        )
        return game_frame_to_team_week(
            frame,
            self.identity(),
            ("abs_spread_line", "team_spread_signed"),
            signed=("team_spread_signed",),
        )

    def _build_bestpick2013(self) -> pd.DataFrame:
        picks = pd.read_parquet(BEST_PICK_PICKS)
        frame = picks[["game_id", "calibrated_probability", "key_number_distance"]].copy()
        frame["game_id"] = frame["game_id"].astype(str)
        return game_frame_to_team_week(
            frame, self.identity(), ("calibrated_probability", "key_number_distance")
        )

    def _build_opener(self) -> pd.DataFrame:
        paired = pd.read_parquet(RIDGE_ALPHA_ARTIFACT / "opener_paired.parquet")
        spread = pd.read_parquet(MICROSTRUCTURE_ARTIFACT / "spread_novig_tue_open.parquet")
        frame = paired[["game_id", "candidate_prob_open"]].copy()
        frame["game_id"] = frame["game_id"].astype(str)
        frame["candidate_dist"] = (
            pd.to_numeric(frame["candidate_prob_open"], errors="coerce") - 0.5
        ).abs()
        disp = spread[["game_id", "spread_std"]].copy()
        disp["game_id"] = disp["game_id"].astype(str)
        frame = frame.merge(disp, on="game_id", how="left")
        return game_frame_to_team_week(frame, self.identity(), ("candidate_dist", "spread_std"))

    def _build_movement(self) -> pd.DataFrame:
        per_game = pd.read_csv(MOVEMENT_PER_GAME)
        frame = per_game[["game_id", "abs_predicted", "predicted_close_minus_open"]].copy()
        frame["game_id"] = frame["game_id"].astype(str)
        frame = frame.rename(columns={"predicted_close_minus_open": "predicted_move_team_signed"})
        return game_frame_to_team_week(
            frame,
            self.identity(),
            ("abs_predicted", "predicted_move_team_signed"),
            signed=("predicted_move_team_signed",),
        )

    def _flag_table(self, flag_builder: str) -> tuple[pd.DataFrame, pd.Series]:
        from nfl_ats.experiment_runner import FLAG_BUILDERS

        construct = FLAG_BUILDERS[flag_builder].build(self.features(), (2009, 2026), {}, REPO)
        table = construct.table.reset_index(drop=True)
        flag = pd.Series(np.asarray(construct.flag), index=table.index).fillna(False).astype(bool)
        if construct.eligible is not None:
            eligible = (
                pd.Series(np.asarray(construct.eligible), index=table.index)
                .fillna(False)
                .astype(bool)
            )
            table = table.loc[eligible].reset_index(drop=True)
            flag = flag.loc[eligible.to_numpy()].reset_index(drop=True)
        return table, flag

    def _build_rest_differential(self) -> pd.DataFrame:
        table, _flag = self._flag_table("extra_rest_edge")
        frame = pd.DataFrame(
            {
                "team_id": table["team"].to_numpy(),
                "season": table["season"].to_numpy(),
                "week": table["week"].to_numpy(),
                "rest_differential": (
                    pd.to_numeric(table["own_rest"], errors="coerce")
                    - pd.to_numeric(table["opp_rest"], errors="coerce")
                ).to_numpy(),
                "own_rest": pd.to_numeric(table["own_rest"], errors="coerce").to_numpy(),
            }
        )
        return frame

    def _build_exposure(self, key: str) -> pd.DataFrame:
        if key == "coach_year_one":
            table, flag = self._coach_year_one()
        elif key == "road_favorite":
            table, base_flag = self._flag_table("large_favorite")
            flag = (~table["is_home"].astype(bool)) & (
                pd.to_numeric(table["team_spread"], errors="coerce") > 0.0
            )
            del base_flag
        else:
            table, flag = self._flag_table(key)
        long = pd.DataFrame(
            {
                "team_id": table["team"].to_numpy(),
                "season": table["season"].to_numpy(),
                "week": table["week"].to_numpy(),
                "exposure": flag.to_numpy().astype(float),
            }
        )
        outcome = "team_covered" if "team_covered" in table.columns else ""
        if outcome:
            self._games[f"exposure:{key}"] = (table, flag, outcome)
        return long

    def _coach_year_one(self) -> tuple[pd.DataFrame, pd.Series]:
        from nfl_ats.coach_fade_overlay import year_one_by_game

        schedules = pd.read_parquet(_latest(REPO / "data" / "raw", "*/schedules.parquet"))
        year_one = year_one_by_game(schedules)
        table, _flag = self._flag_table("home_underdog")
        table = table.copy()
        table["game_id"] = table["game_id"].astype(str)
        year_one["game_id"] = year_one["game_id"].astype(str)
        merged = table.merge(
            year_one[["game_id", "year_one_home", "year_one_away"]], on="game_id", how="left"
        )
        flag = pd.Series(
            np.where(
                merged["is_home"].to_numpy(dtype=bool),
                merged["year_one_home"].fillna(False).to_numpy(dtype=bool),
                merged["year_one_away"].fillna(False).to_numpy(dtype=bool),
            ),
            index=merged.index,
        )
        return merged, flag

    def _build_nflcom(self) -> pd.DataFrame:
        import nflcom_friday_designation_screen as nflcom
        from edge_audit_redteam import initial_last_key, latest, load_report_flags

        schedules_path = latest(REPO / "data" / "raw", "*/schedules.parquet")
        long = nflcom.load_population(schedules_path)
        qa, _counts = load_report_flags(REPO / "data" / "raw" / "nflcom_injuries")
        snaps_path = latest(REPO / "data" / "players" / "raw", "*/snap_counts.parquet")
        starter_exact, starter_fuzzy = nflcom.build_starter_keys(snaps_path)

        starter_status: list[bool] = []
        for season, week, team, name in zip(
            qa["season"], qa["week"], qa["team"], qa["norm_name"], strict=True
        ):
            key3 = (int(season), int(week), str(team))
            init_last = initial_last_key(str(name))
            starter_status.append(
                (*key3, str(name)) in starter_exact
                or (init_last != ("", "") and (*key3, *init_last) in starter_fuzzy)
            )
        qa = qa.copy()
        qa["is_starter"] = starter_status
        out_rows = qa.loc[qa["status_norm"] == "out"].copy()
        out_rows["nonstarter"] = ~out_rows["is_starter"]
        counts = (
            out_rows.groupby(["season", "week", "team"])
            .agg(out_starter=("is_starter", "sum"), out_nonstarter=("nonstarter", "sum"))
            .reset_index()
        )
        work = long.merge(counts, on=["season", "week", "team"], how="left")
        work[["out_starter", "out_nonstarter"]] = work[["out_starter", "out_nonstarter"]].fillna(0)
        return pd.DataFrame(
            {
                "team_id": work["team"].to_numpy(),
                "season": work["season"].to_numpy(),
                "week": work["week"].to_numpy(),
                "out_starter": work["out_starter"].astype(float).to_numpy(),
                "out_nonstarter": work["out_nonstarter"].astype(float).to_numpy(),
            }
        )

    def _build_penalty(self) -> pd.DataFrame:
        import weak_stack_v2_eval as wsv2

        from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot

        snapshot = latest_pbp_snapshot(REPO / "data/pbp/raw")
        pbp = load_pbp_snapshot(snapshot, include_postseason=False)
        rate = wsv2.team_season_penalty_rate(pbp)
        features = self.features()
        attached = wsv2.add_penalty_discipline_feature(features, rate)
        reg = attached.loc[attached["game_type"] == "REG"]
        frame = pd.DataFrame(
            {
                "game_id": reg["game_id"].astype(str).to_numpy(),
                "home_penalty_rate_prior": reg["home_penalty_rate_prior"].to_numpy(),
                "away_penalty_rate_prior": reg["away_penalty_rate_prior"].to_numpy(),
            }
        )
        merged = frame.merge(
            self.identity()[["game_id", "season", "week", "home_team", "away_team"]],
            on="game_id",
            how="inner",
        )
        pieces = []
        for side, team_col in (("home", "home_team"), ("away", "away_team")):
            piece = merged[["season", "week"]].copy()
            piece["team_id"] = merged[team_col].to_numpy()
            piece["penalty_rate_prior"] = merged[f"{side}_penalty_rate_prior"].to_numpy()
            pieces.append(piece)
        return pd.concat(pieces, ignore_index=True)

    def _build_venue_weather(self) -> pd.DataFrame:
        features = relmap.load_feature_table()
        reg = features.loc[features["game_type"] == "REG"].copy()
        return pd.DataFrame(
            {
                "venue_id": reg["home_team"].to_numpy(),
                "season": pd.to_numeric(reg["season"], errors="coerce").to_numpy(),
                "week": pd.to_numeric(reg["week"], errors="coerce").to_numpy(),
                "temp": pd.to_numeric(reg["temp"], errors="coerce").to_numpy(),
                "wind": pd.to_numeric(reg["wind"], errors="coerce").to_numpy(),
            }
        )


def game_frame_to_team_week(
    frame: pd.DataFrame,
    identity: pd.DataFrame,
    metrics: tuple[str, ...],
    *,
    signed: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Explode a game-level metric frame into one row per (game, side).

    A game-level quantity belongs to both sides of the game, so both team-week
    rows carry it; a ``signed`` metric is negated on the away row because its
    sign is stated relative to the home team.
    """

    merged = frame.merge(
        identity[["game_id", "season", "week", "home_team", "away_team"]],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    pieces = []
    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        piece = merged[["season", "week"]].copy()
        piece["team_id"] = merged[team_col].to_numpy()
        for metric in metrics:
            values = pd.to_numeric(merged[metric], errors="coerce")
            piece[metric] = (
                values.to_numpy()
                if side == "home" or metric not in signed
                else (-values.to_numpy())
            )
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def column_diagnostics(
    long: pd.DataFrame, metric: str, unit_col: str, seasons: tuple[int, int]
) -> dict[str, Any]:
    """Shape facts that decide whether a returned correlation means anything."""

    frame = long.loc[:, [unit_col, "season", metric]].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame = frame.dropna(subset=["season"])
    frame = frame.loc[frame["season"].astype(int).between(seasons[0], seasons[1])]
    values = pd.to_numeric(frame[metric], errors="coerce").dropna()
    if values.empty:
        return {
            "n_rows": 0,
            "n_distinct_values": 0,
            "modal_share": math.nan,
            "share_units_zero_within_variance": math.nan,
        }
    counts = values.value_counts()
    frame["_v"] = pd.to_numeric(frame[metric], errors="coerce")
    grouped = frame.dropna(subset=["_v"]).groupby([unit_col, frame["season"].astype(int)])["_v"]
    within_std = grouped.std().fillna(0.0)
    return {
        "n_rows": len(values),
        "n_distinct_values": int(values.nunique()),
        "modal_share": float(counts.iloc[0] / len(values)),
        "share_units_zero_within_variance": float((within_std <= 0.0).mean()),
        "n_units_seen": len(within_std),
    }


def random_halves_check(
    long: pd.DataFrame,
    metric: str,
    *,
    unit_col: str,
    seasons: tuple[int, int],
    reseeds: int = RANDOM_HALVES_RESEEDS,
) -> dict[str, Any]:
    """Re-split each unit at RANDOM instead of odd/even weeks.

    A genuine trait keeps a positive correlation under any split. A quantity
    whose season TOTAL is conserved (days of rest) stays strongly negative,
    because more of it in one half mechanically forces less in the other. This
    is the cheap discriminator the orchestrator measured on team-week ``rest``
    (odd/even r = -0.9766; random halves mean -0.8514 over 20 reseeds).
    """

    values: list[float] = []
    for offset in range(reseeds):
        rng = np.random.default_rng(rlib.RELIABILITY_SEED + offset)
        shuffled = long.copy()
        shuffled["week"] = rng.integers(1, 19, size=len(shuffled))
        measured = rlib.measure_reliability(
            shuffled,
            metric,
            method="random-halves diagnostic",
            unit_col=unit_col,
            seasons=seasons,
            n_boot=RANDOM_HALVES_N_BOOT,
        )
        if measured["status"] == rlib.STATUS_MEASURED and measured["reliability"] is not None:
            values.append(float(measured["reliability"]))
    if not values:
        return {"n_reseeds": 0, "mean": None, "min": None, "max": None}
    return {
        "n_reseeds": len(values),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "note": (
            "Random within-unit halves instead of odd/even weeks. A real trait stays "
            "positive; a conserved-total quantity stays strongly negative."
        ),
    }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


class Measurer:
    def __init__(self, frames: Frames, *, n_boot: int) -> None:
        self.frames = frames
        self.n_boot = n_boot
        self._cache: dict[tuple[str, str, str, tuple[int, int], str], dict[str, Any]] = {}
        self.controls: dict[str, list[dict[str, Any]]] = {}

    def measure(
        self,
        frame_name: str,
        metric: str,
        *,
        method: str,
        seasons: tuple[int, int],
        unit_col: str = "team_id",
        conserved_candidate: bool = False,
    ) -> dict[str, Any]:
        key = (frame_name, metric, method[:24], seasons, unit_col)
        if key in self._cache:
            return self._cache[key]
        long = self.frames.get(frame_name)
        measured = rlib.measure_reliability(
            long,
            metric,
            method=method,
            unit_col=unit_col,
            seasons=seasons,
            n_boot=self.n_boot,
        )
        diagnostics = column_diagnostics(long, metric, unit_col, seasons)
        guard: str | None = None
        random_halves: dict[str, Any] | None = None
        point = measured["reliability"]
        if measured["status"] == rlib.STATUS_MEASURED:
            if diagnostics["modal_share"] >= NEAR_CONSTANT_MODAL_SHARE:
                guard = GUARD_NEAR_CONSTANT
            elif diagnostics["share_units_zero_within_variance"] >= UNIT_CONSTANT_SHARE:
                guard = GUARD_UNIT_CONSTANT
            if conserved_candidate or (point is not None and point < 0.0):
                random_halves = random_halves_check(
                    long, metric, unit_col=unit_col, seasons=seasons
                )
                mean = random_halves.get("mean")
                if (
                    guard is None
                    and mean is not None
                    and mean <= COMPOSITIONAL_RANDOM_HALVES_MAX
                    and point is not None
                    and point < 0.0
                ):
                    guard = GUARD_COMPOSITIONAL
        result = {
            "frame": frame_name,
            "metric": metric,
            "unit_col": unit_col,
            "seasons": [seasons[0], seasons[1]],
            "n_units": measured["n_units"],
            "pearson_r": measured["pearson_r"],
            "pearson_r_ci95": measured["pearson_r_ci95"],
            "spearman_rho": measured["spearman_rho"],
            "spearman_brown_full_length_reliability": measured[
                "spearman_brown_full_length_reliability"
            ],
            "probability_positive": measured["probability_positive"],
            "reliability": measured["reliability"],
            "reliability_low": measured["reliability_low"],
            "reliability_high": measured["reliability_high"],
            "status": measured["status"],
            "method": measured["method"],
            "diagnostics": diagnostics,
            "guard": guard,
            "random_halves": random_halves,
            "recordable": bool(measured["status"] == rlib.STATUS_MEASURED and guard is None),
        }
        self._cache[key] = result
        self.ensure_control(frame_name, seasons, unit_col)
        return result

    def ensure_control(
        self, frame_name: str, seasons: tuple[int, int], unit_col: str = "team_id"
    ) -> None:
        label = f"{frame_name}|{unit_col}|{seasons[0]}-{seasons[1]}"
        if label in self.controls:
            return
        long = self.frames.get(frame_name)
        window = long.copy()
        window["season"] = pd.to_numeric(window["season"], errors="coerce")
        window = window.dropna(subset=["season"])
        window = window.loc[window["season"].astype(int).between(seasons[0], seasons[1])]
        self.controls[label] = rlib.positive_control(window, unit_col=unit_col, n_boot=1000)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def registry_windows() -> dict[str, dict[str, Any]]:
    registry = load_registry(default_registry_path(REPO / "registry"))
    out: dict[str, dict[str, Any]] = {}
    for name, signal in registry.signals.items():
        out[name] = {
            "seasons": (int(signal.seasons[0]), int(signal.seasons[1])),
            "reliability": signal.reliability,
            "effect": signal.effect,
            "classification": signal.classification,
        }
    return out


def set_reliability_command(row: dict[str, Any], source: str) -> list[str]:
    return [
        "nfl-ats",
        "weak-signals",
        "set-reliability",
        "--name",
        row["entry"],
        "--reliability",
        f"{row['reliability']:.6f}",
        "--reliability-low",
        f"{row['reliability_low']:.6f}",
        "--reliability-high",
        f"{row['reliability_high']:.6f}",
        "--method",
        row["method"],
        "--source",
        source,
        "--reason",
        row["record_reason"],
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="ORCH-D modeling_overlay reliability group")
    parser.add_argument("--n-boot", type=int, default=rlib.N_BOOT)
    args = parser.parse_args()

    started = time.time()
    windows = registry_windows()
    missing = [cell.entry for cell in CELLS if cell.entry not in windows]
    if missing:
        raise SystemExit(f"cells not present in the registry: {missing}")

    frames = Frames()
    measurer = Measurer(frames, n_boot=args.n_boot)

    rows: list[dict[str, Any]] = []
    for cell in CELLS:
        seasons = windows[cell.entry]["seasons"]
        row: dict[str, Any] = {
            "entry": cell.entry,
            "disposition": cell.disposition,
            "parent_or_reason": cell.parent,
            "provenance": cell.provenance,
            "seasons": [seasons[0], seasons[1]],
            "registry_effect": windows[cell.entry]["effect"],
            "registry_classification": windows[cell.entry]["classification"],
            "registry_reliability_before": windows[cell.entry]["reliability"],
            "member_rows": list(cell.member_rows),
        }
        if cell.disposition == DISPOSITION_NO_TRAIT:
            row["status"] = "not_applicable"
            row["not_applicable_reason"] = "no underlying trait to be reliable"
            row["reliability"] = None
            row["recordable"] = False
            row["needs_instrument"] = (
                "out-of-window replication of the paired delta, not a split-half reliability"
            )
        else:
            measured = measurer.measure(
                str(cell.frame),
                str(cell.metric),
                method=str(cell.method),
                seasons=seasons,
                unit_col=cell.unit_col,
                conserved_candidate=cell.conserved_candidate,
            )
            row.update(measured)
            row["entry"] = cell.entry
            row["record_reason"] = (
                f"{'Trait' if cell.disposition == DISPOSITION_TRAIT else 'EXPOSURE'} reliability "
                f"of {cell.parent}, the parent quantity this cell is built on "
                f"({cell.provenance}). Measured on the cell's own seasons "
                f"{seasons[0]}-{seasons[1]}. This is a MEASUREMENT only: it closes nothing, "
                f"reclassifies nothing, and names no closing_ground."
                + (
                    " EXPOSURE reliability is NOT the trait reliability "
                    "NO_SPLIT_HALF_RELIABILITY_MAX was calibrated against and is NOT an "
                    "admissible no_split_half_reliability ground."
                    if cell.disposition == DISPOSITION_EXPOSURE
                    else ""
                )
            )
            replication = frames.games_for(str(cell.frame))
            if replication is not None:
                games, flag, outcome = replication
                row["half_season_replication"] = rlib.half_season_replication(
                    games.loc[games["season"].between(seasons[0], seasons[1])].reset_index(
                        drop=True
                    ),
                    flag.loc[
                        games["season"].between(seasons[0], seasons[1]).to_numpy()
                    ].reset_index(drop=True),
                    outcome_col=outcome,
                )

        members: list[dict[str, Any]] = []
        member_seasons = tuple(cell.extra.get("member_seasons", list(seasons)))
        member_unit = cell.extra.get("member_unit_col", "team_id")
        member_method = _VENUE if cell.extra.get("member_method") == "venue" else _TRAIT
        for frame_name, metric in cell.members:
            members.append(
                measurer.measure(
                    frame_name,
                    metric,
                    method=member_method,
                    seasons=(int(member_seasons[0]), int(member_seasons[1])),
                    unit_col=member_unit,
                )
            )
        if members:
            row["member_measurements"] = members
            row["member_measurements_note"] = (
                "REPORTED, NEVER RECORDED: these are the block's individual parent columns. "
                "The cell's construct is the block, so no one of them may occupy the cell's "
                "single-valued reliability field."
            )
        rows.append(row)
        shown = f"{row['reliability']:+.4f}" if row.get("reliability") is not None else "   n/a "
        print(
            f"  {cell.entry:<62} {cell.disposition:<12} {shown} "
            f"{row.get('status', '')}{'/' + row['guard'] if row.get('guard') else ''}"
        )

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO / "artifacts" / "reliability_sweep" / "modeling_overlay" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    source = f"artifacts/reliability_sweep/modeling_overlay/{timestamp}/results.json"

    recordable = [r for r in rows if r.get("recordable")]
    commands = [set_reliability_command(r, source) for r in recordable]

    counts = {
        "a_trait": sum(1 for r in rows if r["disposition"] == DISPOSITION_TRAIT),
        "b_exposure": sum(1 for r in rows if r["disposition"] == DISPOSITION_EXPOSURE),
        "c_no_trait": sum(1 for r in rows if r["disposition"] == DISPOSITION_NO_TRAIT),
        "recordable": len(recordable),
        "measured_but_guarded": sum(1 for r in rows if r.get("guard")),
        "unmeasurable": sum(
            1
            for r in rows
            if r["disposition"] != DISPOSITION_NO_TRAIT
            and r.get("status") not in (rlib.STATUS_MEASURED,)
        ),
    }

    configuration = {
        "command": "reliability-modeling-overlay",
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "entries": [cell.entry for cell in CELLS],
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "random_halves_reseeds": RANDOM_HALVES_RESEEDS,
        "near_constant_modal_share": NEAR_CONSTANT_MODAL_SHARE,
        "unit_constant_share": UNIT_CONSTANT_SHARE,
        "compositional_random_halves_max": COMPOSITIONAL_RANDOM_HALVES_MAX,
        "disposition_counts": counts,
        "disposition_definitions": {
            "a_trait": "the cell thresholds or ranks by ONE named continuous quantity",
            "b_exposure": (
                "the cell's parent is categorical; the flag's per-team-season exposure rate "
                "is measured, and a low value is NOT an admissible closing ground"
            ),
            "c_no_trait": (
                "not_applicable: no underlying trait to be reliable. A model/composition/"
                "weighting-scheme comparison or a block of >=2 distinct parents. NOT a "
                "closure and NOT a negative about the signal -- these cells need an "
                "out-of-window replication instrument, not a reliability"
            ),
        },
        "guard_definitions": {
            GUARD_COMPOSITIONAL: (
                "parent's season total is conserved, so any split returns a strongly "
                "negative correlation; split-half reliability does not apply"
            ),
            GUARD_UNIT_CONSTANT: (
                "parent is constant within a team-season by construction, so odd/even "
                "means are identical and the correlation is 1.0 tautologically"
            ),
            GUARD_NEAR_CONSTANT: (
                "one value covers >= 98% of the column, so |r| is a function of a handful "
                "of rows and flips with the season window"
            ),
        },
        "positive_control": measurer.controls,
        "results": rows,
        "set_reliability_commands": commands,
        "binding_note": (
            "An interval or CI that contains zero is NEVER grounds to reject, fail, or close "
            "an experiment. Only two grounds ever close a line of work: a RESOLVED wrong sign "
            "or zero split-half reliability; or bounded by a positive control proven able to "
            "detect an effect that size. Everything else is unresolved_below_power. This "
            "artifact closes nothing, reclassifies nothing, and proposes no closing_ground."
        ),
        "provenance": artifact_provenance(configuration, relmap.V4_PATH, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="reliability-modeling-overlay",
        metrics={
            "n_entries": len(rows),
            "n_a_trait": counts["a_trait"],
            "n_b_exposure": counts["b_exposure"],
            "n_c_no_trait": counts["c_no_trait"],
            "n_recordable": counts["recordable"],
        },
        notes=(
            "Measure-only reliability dispositions for the modeling_overlay registry cells. "
            "Most cells are model-vs-model paired deltas with no parent trait; those are "
            "reported not_applicable, which is informative and is NOT a closure."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")
    print(f"dispositions: {counts}")
    print(f"\n{len(commands)} set-reliability commands:")
    for command in commands:
        print("  " + " ".join(f'"{part}"' if " " in part else part for part in command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
