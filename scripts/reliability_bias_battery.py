"""Split-half reliability for the 25 ``bias_battery_*`` registry cells (ORCH-D).

**What these cells are.** ``scripts/nfl_bias_battery_screen.py`` predeclares
17 team-side situational/behavioral hypotheses on the team-game long table
(``build_long_table`` + ``add_history_features``; ``build_hypotheses`` returns
each cell's boolean ``flag``). The registry carries 25 ``bias_battery_*``
cells: those 17 base cells plus 8 ``*_opener`` re-grades of the SAME
underlying construct on the 2020-2025 opener-only window (read: each
``_opener`` entry's own ``description``, "Opener-grade re-screen of an
already-recorded close-graded cell"). This script measures each cell's
PARENT quantity's reliability, never the flag itself as a raw 0/1 series --
consistent with the registry's own precedent that a battery's cells inherit
the reliability of the trait they are built on (the six
``attention_battery_*`` cells all share one number,
``scripts/attention_battery_screen.py:437``).

**Two families of parent, decided by reading ``build_hypotheses`` (this
file's own ``HYPOTHESIS_SPEC`` documents the reasoning per cell):**

- Some flags threshold a genuinely continuous team-week quantity that exists
  independently of the flag (a cumulative win pct, a score margin, a rest
  count, a market spread) -- those get ``METHOD_TRAIT`` on that column,
  ``unit_col="team"`` (``build_long_table`` already returns one row per team
  per game, so no game-level explode is needed; ``measure_reliability``'s
  own ``unit_col`` rename handles it).
- Some flags are a fixed-list/schedule-slot/compound categorical condition
  with no continuous parent that exists independently of the flag (team
  membership in a 4-team "marquee" list, a weekday/kickoff-hour slot, a
  division-rematch-and-lost pattern) -- those get ``METHOD_EXPOSURE``: the
  flag's own per-team-week exposure rate, added as a column and measured the
  same way. This is NOT ``reliability_lib.game_flag_to_team_week`` (that
  helper explodes a GAME-level frame with separate home/away columns into
  team-week rows, assuming both sides share one flag value -- wrong here,
  since these flags are already side-specific on an already-team-week frame).
  The exposure column is built directly on ``long_df`` instead; the
  estimator underneath (``measure_reliability`` -> ``split_half_reliability``)
  is identical either way.

**Binding taxonomy (verbatim, AGENTS.md / CLAUDE.md).** An interval or CI that
contains zero is NEVER grounds to reject, fail, or close an experiment. Only
two grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED
wrong sign (whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an effect
that size. Everything else is ``unresolved_below_power``; report
``probability_positive``, never "contains zero". This script CLOSES NOTHING:
it measures, and a low number is a candidate for the reliability ground, never
the closure itself. Within-week correlation is ZERO.

A construct with too few usable team-seasons is reported as UNMEASURED, never
as reliability 0 -- writing a NaN through as a number would manufacture the
appearance of a closing ground out of nothing.

Writes ``artifacts/reliability_sweep/bias_battery/<stamp>/results.json`` via
``nfl_ats.provenance.write_experiment_artifact`` and prints the
``set-reliability`` commands it would run (recording itself goes through the
locked CLI, never this script).
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

import nfl_bias_battery_screen as battery  # noqa: E402
import reliability_lib as rlib  # noqa: E402

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.weak_signals import default_registry_path, load_registry  # noqa: E402

PREFIX = "bias_battery_"
OPENER_SUFFIX = "_opener"


#: Registry entry (minus the ``bias_battery_`` prefix and any ``_opener``
#: suffix) -> the hypothesis key ``build_hypotheses`` returns. Every one of
#: the 17 hypotheses maps to itself; the 8 opener cells strip their suffix to
#: land on the same base hypothesis, per the registry's own stated
#: convention (an opener re-grade shares the base construct, "Opener-grade
#: re-screen of an already-recorded close-graded cell").
def hypothesis_for(entry_name: str) -> str:
    assert entry_name.startswith(PREFIX), entry_name
    stem = entry_name[len(PREFIX) :]
    if stem.endswith(OPENER_SUFFIX):
        stem = stem[: -len(OPENER_SUFFIX)]
    return stem


# Per-hypothesis parent-quantity spec, read off ``nfl_bias_battery_screen.
# build_hypotheses`` (scripts/nfl_bias_battery_screen.py:259-398) cell by
# cell. ``kind`` is "trait" (a continuous team-week column measured with
# METHOD_TRAIT) or "exposure" (the hypothesis's own boolean flag, measured
# with METHOD_EXPOSURE as a per-team-week exposure rate). "column" for a
# trait cell names the continuous parent; two or three hypotheses share one
# parent column deliberately (bad_team_late/great_team_late both threshold
# prior_win_pct; motivation_mismatch also gates on prior_win_pct, read:
# build_hypotheses lines 269-293; post_blowout_win_letdown/loss_bounce both
# threshold prior_score_margin, lines 302-313) -- on the SAME season window
# those measurements are numerically identical, the same convention the
# registry's attention_battery_* cells already use for one shared trait.
HYPOTHESIS_SPEC: dict[str, dict[str, Any]] = {
    "bad_team_late": {
        "kind": "trait",
        "column": "prior_win_pct",
        "reason": (
            "bad_team_late (build_hypotheses:269-276) thresholds prior_win_pct <= 0.300 "
            "inside a week-11-18/prior_games>=9 eligibility window; prior_win_pct "
            "(add_history_features:204-214, cumulative win pct on strictly-prior games "
            "this season) is the continuous team-week trait the flag exists to threshold."
        ),
    },
    "great_team_late": {
        "kind": "trait",
        "column": "prior_win_pct",
        "reason": (
            "great_team_late (build_hypotheses:277-284) thresholds the SAME prior_win_pct "
            "trait at >= 0.800 inside a week-15-18/prior_games>=13 window; shares "
            "bad_team_late's parent trait and, on the same season window, its number."
        ),
    },
    "motivation_mismatch": {
        "kind": "trait",
        "column": "prior_win_pct",
        "reason": (
            "motivation_mismatch (build_hypotheses:285-293) gates the ACTING team's own "
            "prior_win_pct >= 0.400 and separately requires the opponent's opp_prior_win_pct "
            "<= 0.300 (a self-join of the identical prior_win_pct trait onto the opponent's "
            "row, add_history_features:216-224) -- both sides of the flag are the same "
            "underlying team-quality trait, so prior_win_pct is the parent measured here."
        ),
    },
    "post_blowout_win_letdown": {
        "kind": "trait",
        "column": "prior_score_margin",
        "reason": (
            "post_blowout_win_letdown (build_hypotheses:302-307) thresholds "
            "prior_score_margin >= 17; prior_score_margin (add_history_features:226-228, "
            "the immediately preceding game's own score margin) is its continuous parent."
        ),
    },
    "post_blowout_loss_bounce": {
        "kind": "trait",
        "column": "prior_score_margin",
        "reason": (
            "post_blowout_loss_bounce (build_hypotheses:308-313) thresholds the SAME "
            "prior_score_margin trait at <= -17; shares post_blowout_win_letdown's parent "
            "trait and, on the same season window, its number."
        ),
    },
    "short_week": {
        "kind": "trait",
        "column": "own_rest",
        "reason": (
            "short_week (build_hypotheses:314-319) thresholds own_rest <= 5; own_rest is a "
            "continuous team-week rest-day count merged from schedules.parquet in "
            "build_long_table (not from add_history_features, but still the genuine "
            "continuous quantity the flag exists to threshold -- the orchestrator's own "
            "task spec names 'a rest count' as a METHOD_TRAIT example)."
        ),
    },
    "extra_rest_edge": {
        "kind": "trait",
        "column": "rest_diff",
        "reason": (
            "extra_rest_edge (build_hypotheses:320-325) thresholds n = own_rest - opp_rest "
            ">= 4, a local variable in build_hypotheses, not a stored column. This script "
            "adds it to the long frame as rest_diff = own_rest - opp_rest (identical "
            "arithmetic) so it can be measured; it is a DIFFERENT continuous quantity from "
            "short_week's own_rest (a differential, not a raw count), so a separate number."
        ),
    },
    "large_favorite": {
        "kind": "trait",
        "column": "team_spread",
        "reason": (
            "large_favorite (build_hypotheses:392-397) thresholds team_is_favorite & "
            "team_spread > 10; team_spread (build_long_table:149, the market spread signed "
            "to the team's own side) is a genuine continuous per-team-week market quantity, "
            "not a fixed categorical list -- its own threshold IS the flag."
        ),
    },
    "home_underdog": {
        "kind": "trait",
        "column": "team_spread",
        "reason": (
            "home_underdog (build_hypotheses:386-391) is is_home & spread_line < 0, which "
            "for home rows equals is_home & team_spread < 0 (team_spread = spread_line on "
            "home rows, build_long_table:149) -- the same continuous team_spread trait as "
            "large_favorite, restricted to the is_home half of the population, sign- rather "
            "than magnitude-thresholded. is_home is an eligibility restriction (as "
            "prior_games>=9 is for bad_team_late), not a second categorical driver of the "
            "hypothesis's name."
        ),
    },
    "backup_qb_start": {
        "kind": "exposure",
        "eligible": True,
        "reason": (
            "backup_qb_start (build_hypotheses:294-301) is backup_qb_flag == 1.0, itself "
            "already a derived 0/1/NaN indicator (_qb_backup_flag, build_hypotheses:173-190) "
            "with no continuous parent (no 'games since last QB change' count exists in this "
            "frame) -- measured as the flag's own per-team-week EXPOSURE rate. eligible = "
            "backup_qb_flag.notna() excludes the undefined-baseline rows from both arms, "
            "matching build_hypotheses's own semantics."
        ),
    },
    "three_plus_road_games": {
        "kind": "exposure",
        "reason": (
            "three_plus_road_games (build_hypotheses:326-331) is three_plus_road_flag, a "
            "consecutive-True-road-game chain (add_history_features:235-240) with no "
            "continuous parent -- a schedule-position exposure, measured as its own rate."
        ),
    },
    "sandwich_spot": {
        "kind": "exposure",
        "reason": (
            "sandwich_spot (build_hypotheses:332-337) is sandwich_flag, a categorical "
            "div-game-flanking pattern (add_history_features:242-246) with no continuous "
            "parent -- measured as its own per-team-week exposure rate."
        ),
    },
    "west_coast_early_kickoff": {
        "kind": "exposure",
        "reason": (
            "west_coast_early_kickoff (build_hypotheses:338-350) is a compound categorical "
            "condition (team in the fixed PT_TEAMS list, traveling, non-PT opponent, "
            "kickoff hour < 14) with no single continuous parent -- the orchestrator's own "
            "task spec names 'Pacific-time team' as a METHOD_EXPOSURE example verbatim."
        ),
    },
    "december_weather_dogs": {
        "kind": "exposure",
        "reason": (
            "december_weather_dogs (build_hypotheses:351-363) is a compound "
            "schedule/venue/market condition (home underdog, week>=14, outdoor/open roof, "
            "temp<=32F) with no single continuous team-week parent trait -- measured as its "
            "own per-team-week exposure rate, not as a venue trait (it is gated on the "
            "team's own market/home status, not scored at the venue level)."
        ),
    },
    "division_revenge_game": {
        "kind": "exposure",
        "reason": (
            "division_revenge_game (build_hypotheses:364-369) is revenge_flag, a categorical "
            "rematch-and-lost-the-first-meeting pattern (add_history_features:248-253) with "
            "no continuous parent -- measured as its own per-team-week exposure rate."
        ),
    },
    "marquee_favorite": {
        "kind": "exposure",
        "reason": (
            "marquee_favorite (build_hypotheses:370-375) gates on team membership in the "
            "fixed 4-team MARQUEE_TEAMS list -- a pure categorical condition with no "
            "continuous parent (the orchestrator's own task spec names 'marquee team' as a "
            "METHOD_EXPOSURE example verbatim) -- measured as its own exposure rate."
        ),
    },
    "primetime_favorite": {
        "kind": "exposure",
        "reason": (
            "primetime_favorite (build_hypotheses:376-385) gates on a weekday/kickoff-hour "
            "schedule slot (Thu/Mon, or Sunday >=20:00) with no continuous parent -- "
            "measured as its own per-team-week exposure rate, the same class as "
            "west_coast_early_kickoff's schedule-slot condition."
        ),
    },
}


def target_entries() -> dict[str, dict[str, Any]]:
    registry = load_registry(default_registry_path(REPO / "registry"))
    out: dict[str, dict[str, Any]] = {}
    for name, signal in registry.signals.items():
        if signal.league != "nfl" or signal.effect_units != "accuracy_points":
            continue
        if not name.startswith(PREFIX):
            continue
        out[name] = {
            "seasons": (int(signal.seasons[0]), int(signal.seasons[1])),
            "reliability": signal.reliability,
            "effect": signal.effect,
            "classification": signal.classification,
            "source": signal.source,
        }
    return out


#: Hazard, sharpened mid-session by a concurrent ORCH-D worker's independent
#: measurement and confirmed here by re-running its own diagnostic (a random,
#: non-odd/even half split; if the strongly negative correlation survives
#: randomizing the split, it is not an odd/even artifact -- it is a
#: COMPOSITIONAL CONSTRAINT: a quantity whose season TOTAL is conserved (a
#: fixed calendar span split among a team's games) mechanically forces more
#: rest in one half to imply less in the other, no matter how the halves are
#: drawn. Split-half reliability is NOT APPLICABLE to such a quantity -- a
#: low or strongly negative value is a measurement artifact of the
#: estimator's assumption (independent halves), not evidence about a trait,
#: and per the concurrent worker's warning, recording it would plant an
#: illegitimate `no_split_half_reliability` closing ground.
#:
#: own_rest (this script, 2009-2025, measured just now): real odd/even raw
#: Pearson r = -0.9313 (Spearman-Brown reliability -0.9313, already at the
#: [-1,1] floor so SB leaves it unchanged); 20 reseeds of a RANDOM (non-
#: odd/even) half split give mean raw r = -0.8014, range [-0.8321, -0.7751]
#: -- the negative correlation survives randomizing the split, confirming
#: compositional conservation rather than an odd/even-parity artifact.
#: Corroborates a concurrent ORCH-D worker's independent measurement on the
#: 2009-2025 REG schedule's own `rest` column: odd/even r=-0.9766 95%
#: [-0.9816,-0.9713] n=544, random-half mean r=-0.8514 range
#: [-0.8813,-0.8066] (reported to this worker verbatim mid-session).
#:
#: rest_diff (own_rest - opp_rest, this script, 2009-2025): real odd/even
#: raw Pearson r = -0.3132 (Spearman-Brown reliability -0.9122, the
#: correction's amplification of a raw r near the -1/3 singularity of
#: 2r/(1+r) -- the recorded-looking number is far more extreme than the raw
#: correlation that produced it); 20 reseeds of a random half split give
#: mean raw r = -0.3331, range [-0.3714, -0.3006] -- again survives
#: randomizing the split, confirming compositional conservation (rest_diff
#: sums two conserved quantities, so it inherits the same constraint,
#: attenuated by the subtraction).
NEAR_CONSTANT_HAZARD: dict[str, str] = {
    "own_rest": (
        "not_applicable_compositional_constraint: own_rest is a per-team-season conserved "
        "quantity (a fixed calendar span split among a team's games), so more rest in one "
        "half of the season mechanically forces less in the other, independent of any real "
        "team trait. Real odd/even raw r=-0.9313 (reliability -0.9313); random (non-odd/even) "
        "half-split reseeds (n=20) give mean raw r=-0.8014, range [-0.8321, -0.7751] -- the "
        "strongly negative correlation SURVIVES randomizing which weeks fall in which half, "
        "which is the diagnostic for a compositional constraint rather than an odd/even-"
        "parity artifact. Corroborated by a concurrent ORCH-D worker's independent measurement "
        "of the schedule's own `rest` column: odd/even r=-0.9766 95% [-0.9816,-0.9713] n=544, "
        "random-half mean r=-0.8514 range [-0.8813,-0.8066]. Split-half reliability does not "
        "apply to this quantity; reported, not recorded, and NOT an admissible "
        "no_split_half_reliability ground despite the low number."
    ),
    "rest_diff": (
        "not_applicable_compositional_constraint: rest_diff = own_rest - opp_rest inherits "
        "own_rest's per-team-season conservation on both sides. Real odd/even raw r=-0.3132 "
        "(Spearman-Brown-corrected reliability -0.9122 -- the correction amplifies a raw r "
        "near the 2r/(1+r) singularity at r=-1/3, so the recorded-looking number is far more "
        "extreme than the underlying correlation); random half-split reseeds (n=20) give mean "
        "raw r=-0.3331, range [-0.3714, -0.3006] -- again survives randomizing the split, "
        "confirming compositional conservation, not an odd/even artifact. Split-half "
        "reliability does not apply to this quantity; reported, not recorded, and NOT an "
        "admissible no_split_half_reliability ground despite the low/negative number."
    ),
}


def build_frame() -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    merged = battery.load_merged(battery.DEFAULT_FEATURES, battery.default_schedules())
    long_df = battery.build_long_table(merged)
    long_df = battery.add_history_features(long_df)
    long_df["rest_diff"] = long_df["own_rest"] - long_df["opp_rest"]
    hyps = battery.build_hypotheses(long_df)
    assert len(hyps) == 17, f"expected 17 predeclared cells, got {len(hyps)}"
    assert set(HYPOTHESIS_SPEC) == set(hyps), (
        f"HYPOTHESIS_SPEC covers {sorted(set(HYPOTHESIS_SPEC) - set(hyps))} not returned by "
        f"build_hypotheses, and is missing {sorted(set(hyps) - set(HYPOTHESIS_SPEC))}"
    )
    return long_df, hyps


def _population_and_flag(
    long_df: pd.DataFrame,
    hyps: dict[str, dict[str, Any]],
    hyp_name: str,
    seasons: tuple[int, int],
) -> tuple[pd.DataFrame, pd.Series]:
    """Cell population restricted to (a) its eligibility mask, if any, and
    (b) the registry entry's OWN season window -- mirrors
    ``nfl_bias_battery_screen.score_hypothesis``'s own restriction order.
    """

    spec = hyps[hyp_name]
    population = long_df
    flag = spec["flag"]
    eligible = spec.get("eligible")
    if eligible is not None:
        population = population.loc[eligible].reset_index(drop=True)
        flag = flag.loc[eligible].reset_index(drop=True)
    season_mask = population["season"].between(seasons[0], seasons[1])
    population = population.loc[season_mask].reset_index(drop=True)
    flag = flag.loc[season_mask].reset_index(drop=True)
    return population, flag


def _exposure_frame(
    long_df: pd.DataFrame, flag: pd.Series, eligible: pd.Series | None
) -> pd.DataFrame:
    """Team-week exposure frame built directly on the already-team-week
    ``long_df`` (NOT ``reliability_lib.game_flag_to_team_week``, which
    explodes a game-level frame with separate home/away columns and would
    double-count or mis-assign these already side-specific flags).
    """

    frame = long_df.loc[:, ["team", "season", "week"]].rename(columns={"team": "team_id"})
    frame["exposure"] = flag.reindex(long_df.index).fillna(False).astype(bool).astype(float)
    if eligible is not None:
        elig = eligible.reindex(long_df.index).fillna(False)
        frame.loc[~elig, "exposure"] = np.nan
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=rlib.N_BOOT)
    args = parser.parse_args()

    started = time.time()
    entries = target_entries()
    print(f"=== {len(entries)} bias_battery registry cells in scope ===")

    long_df, hyps = build_frame()
    print(f"team-game long frame (pushes dropped): {long_df.shape}")

    rows: list[dict[str, Any]] = []
    replication_cells: dict[str, dict[str, Any]] = {}
    for name in sorted(entries):
        entry = entries[name]
        hyp_name = hypothesis_for(name)
        if hyp_name not in HYPOTHESIS_SPEC or hyp_name not in hyps:
            rows.append(
                {
                    "entry": name,
                    "hypothesis": hyp_name,
                    "seasons": list(entry["seasons"]),
                    "status": "no_matching_hypothesis",
                    "reliability": None,
                }
            )
            continue
        spec = HYPOTHESIS_SPEC[hyp_name]
        eligible = hyps[hyp_name].get("eligible")

        if spec["kind"] == "trait":
            column = spec["column"]
            measured = rlib.measure_reliability(
                long_df,
                column,
                method=rlib.METHOD_TRAIT,
                unit_col="team",
                seasons=entry["seasons"],
                n_boot=args.n_boot,
            )
            parent = column
        else:
            exposure_frame = _exposure_frame(long_df, hyps[hyp_name]["flag"], eligible)
            measured = rlib.measure_reliability(
                exposure_frame,
                "exposure",
                method=rlib.METHOD_EXPOSURE,
                unit_col="team_id",
                seasons=entry["seasons"],
                n_boot=args.n_boot,
            )
            parent = f"{hyp_name}.flag (exposure rate)"

        population, flag = _population_and_flag(long_df, hyps, hyp_name, entry["seasons"])
        replication = rlib.half_season_replication(population, flag, outcome_col="team_covered")
        replication_cells[name] = replication

        hazard = (
            NEAR_CONSTANT_HAZARD.get(spec.get("column", "")) if spec["kind"] == "trait" else None
        )
        row = {
            "entry": name,
            "hypothesis": hyp_name,
            "parent_quantity": parent,
            "kind": spec["kind"],
            "mapping_reason": spec["reason"],
            "not_informative_near_constant": hazard,
            "seasons": list(entry["seasons"]),
            "registry_effect": entry["effect"],
            "registry_classification": entry["classification"],
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
            "half_season_replication": replication,
        }
        rows.append(row)
        rel = measured["reliability"]
        shown = f"{rel:+.4f}" if rel is not None else "  n/a "
        print(
            f"  {name:<48} {parent:<32} n={measured['n_units']:>4} rel={shown} {measured['status']}"
        )

    battery_replication = rlib.battery_replication_correlation(replication_cells)

    windows = sorted({tuple(r["seasons"]) for r in rows if r.get("n_units")})
    controls: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        restricted = long_df.loc[long_df["season"].between(window[0], window[1])]
        controls[f"{window[0]}-{window[1]}"] = rlib.positive_control(
            restricted, unit_col="team", n_boot=1000
        )

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO / "artifacts" / "reliability_sweep" / "bias_battery" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "command": "reliability-bias-battery",
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "entries": sorted(entries),
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "seed": rlib.RELIABILITY_SEED,
        "n_boot": args.n_boot,
        "min_units": rlib.MIN_UNITS,
        "mapping_provenance": (
            "scripts/nfl_bias_battery_screen.py:259-398 (build_hypotheses), read 2026-09-01. "
            "Each registry cell 'bias_battery_<hyp>[_opener]' maps to hypothesis <hyp> "
            "(the '_opener' suffix stripped and the base hypothesis re-graded on 2020-2025); "
            "per-hypothesis parent-quantity provenance is HYPOTHESIS_SPEC[<hyp>]['reason'] "
            "in this script, cited by build_hypotheses line range."
        ),
        "hypothesis_spec": {
            name: {"kind": spec["kind"], "column": spec.get("column"), "reason": spec["reason"]}
            for name, spec in HYPOTHESIS_SPEC.items()
        },
        "positive_control": controls,
        "battery_replication_correlation": battery_replication,
        "results": rows,
        "provenance": artifact_provenance(
            configuration, battery.DEFAULT_FEATURES, project_root=REPO
        ),
    }
    measured_count = sum(1 for r in rows if r["status"] == rlib.STATUS_MEASURED)
    hazard_count = sum(1 for r in rows if r.get("not_informative_near_constant"))
    recordable_count = measured_count - hazard_count
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="reliability-bias-battery",
        metrics={
            "n_entries": len(rows),
            "n_measured": measured_count,
            "n_unmeasured": len(rows) - measured_count,
            "n_not_informative_near_constant": hazard_count,
            "n_recordable": recordable_count,
        },
        notes=(
            "Measure-only split-half reliability for the bias_battery registry cells; "
            "every cell measured regardless of sign or interval shape, and nothing is "
            "closed or reclassified, per AGENTS.md's binding closing-grounds taxonomy. "
            f"{hazard_count} of {measured_count} measured rows (own_rest/rest_diff-parented: "
            "short_week[_opener], extra_rest_edge[_opener]) are flagged "
            "not_applicable_compositional_constraint -- own_rest/rest_diff are per-team-season "
            "conserved quantities (a fixed calendar span split among a team's games), so "
            "split-half reliability's independent-halves assumption does not hold; confirmed "
            "by a random (non-odd/even) half-split reseed that reproduces the same strongly "
            "negative correlation. Reported but NOT recorded, and NOT an admissible "
            "no_split_half_reliability closing ground despite the low/negative number."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")

    measured_rows = [r for r in rows if r["status"] == rlib.STATUS_MEASURED]
    print(f"\n{len(measured_rows)} of {len(rows)} measured; {len(rows) - len(measured_rows)} not")
    print(
        f"{hazard_count} measured rows flagged not_applicable_compositional_constraint "
        f"(not recorded); {recordable_count} recordable"
    )
    for row in rows:
        if row.get("not_informative_near_constant"):
            print(f"  HAZARD {row['entry']}: {row['not_informative_near_constant']}")
    for label, predicate in (
        ("<= 0.10", lambda v: v <= 0.10),
        (">= 0.80", lambda v: v >= 0.80),
    ):
        hits = [r for r in measured_rows if predicate(r["reliability"])]
        print(f"  {label}: {len(hits)}")
        for row in sorted(hits, key=lambda r: r["reliability"]):
            print(
                f"    {row['entry']:<48} {row['reliability']:+.4f} "
                f"[{row['reliability_low']:+.4f}, {row['reliability_high']:+.4f}] "
                f"({row['kind']})"
            )

    print("\n=== battery replication correlation (odd-season gap vs even-season gap) ===")
    print(f"  {battery_replication}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
