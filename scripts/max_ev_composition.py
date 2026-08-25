"""MAX-EV played-card composition study: production chain PLUS all three edges.

Question (owner task, 2026-08-22): the MAX-EV card is the sequential chain
(raw model -> coach fade -> player-arrest policy) with the movement rule
(|close-tue_open| >= 1.0 follow market), the NFL.com Friday out>=2
starter-caliber fade, and the PBP-08 protection-mismatch tilt (prior-4-game
top-quartile pressure-allowed offense vs top-quartile pressure-generating
defense -> back the defense side) applied on top, in that order. All flag
machinery is IMPORTED verbatim from scripts/movement_composition_eval.py,
scripts/nflcom_friday_refresh_feature.py (itself importing
nflcom_friday_designation_screen + player_arrests_policy_eval), and
scripts/pbp08_matchup_screen.py; nothing is re-derived.

Arms and rules are frozen in docs/max_ev_composition.md BEFORE scoring:
  a. incumbent chain reproduction gate (must equal 0.541583499667332 <=1e-9)
  b. chain + movement
  c. chain + movement + nflcom
  d. FULL max-EV stack, slate-wide PARTIAL application (edge fires where its
     data exists, else no-op) -- what would actually ship
  d-restricted. same full-stack picks on games where ALL FOUR sources exist
  e. leave-one-out marginals within the full stack

Bootstrap: nfl_ats.clv.week_blocked_bootstrap, 20,000 samples, seed 20260824,
block="week" primary / block="season" secondary, paired deltas vs arm (a) in
accuracy points, graded at the frozen Tuesday line with pick_correct.

Attribution on already-scored archives only -- UPPER BOUND, disclosed in the
doc and metadata; no rotation window spent. Writes
artifacts/max_ev_composition/<run_id>/ and stamps
registry/experiments/max-ev-composition/. Does NOT write either registry JSON;
the proposed `nfl-ats weak-signals record` line for `maxev_full_stack`
(classification unresolved_below_power) is printed and saved in metadata.json.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from movement_composition_eval import (
    DEFAULT_IDENTITY_FEATURES,
    DEFAULT_INCIDENTS,
    DEFAULT_MARKET_ROOT,
    DEFAULT_OPENER,
    DEFAULT_WEAK_STACK_FEATURES,
    MOVEMENT_THRESHOLD,
    PUBLISHED_SEQUENTIAL_CHAIN,
    build_arrest_flags,
    build_coach_flip_ids,
    compose_chain,
    movement_overlay,
    reload_market_lines,
)
from nflcom_friday_designation_screen import latest
from nflcom_friday_refresh_feature import (
    apply_overlay,
    attach_counts,
    build_out_counts,
)
from pbp08_matchup_screen import (
    _latest_pbp_snapshot,
    build_cells,
    build_game_trait_tables,
    build_long_table,
    load_population,
)
from pbp08_matchup_screen import (
    latest_schedules as _latest_schedules,
)

from nfl_ats.clv import pick_correct, week_blocked_bootstrap
from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.io import atomic_csv, atomic_parquet, run_id
from nfl_ats.provenance import artifact_provenance, sha256_file, write_experiment_artifact

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INJURIES_ROOT = REPO / "data/raw/nflcom_injuries"
DEFAULT_OUTPUT_ROOT = REPO / "artifacts/max_ev_composition"
DEFAULT_REGISTRY_ROOT = REPO / "registry"

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260824
NFLCOM_SEASONS = range(2022, 2025)


def _paired_metric_fn(candidate_col: str, incumbent_col: str):
    def _metric(rows: pd.DataFrame) -> dict[str, float]:
        both = rows.dropna(subset=[candidate_col, incumbent_col])
        if both.empty:
            return {
                "candidate_accuracy": float("nan"),
                "incumbent_accuracy": float("nan"),
                "paired_delta": float("nan"),
            }
        c = both[candidate_col].astype(float)
        p = both[incumbent_col].astype(float)
        return {
            "candidate_accuracy": float(c.mean()),
            "incumbent_accuracy": float(p.mean()),
            "paired_delta": float((c - p).mean()),
        }

    return _metric


def _extract(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    row = frame.loc[frame["metric"].eq(metric)].iloc[0]
    return {
        "estimate": float(row["estimate"]),
        "lower": float(row["lower"]),
        "upper": float(row["upper"]),
        "probability_positive": float(row["probability_positive"]),
    }


def score_arm(
    frame: pd.DataFrame,
    *,
    name: str,
    candidate_pick_col: str,
    incumbent_correct_col: str = "correct_chain",
) -> dict[str, Any]:
    working = frame.copy()
    correct_col = f"_correct_{name}"
    working[correct_col] = pick_correct(working[candidate_pick_col], working["margin_vs_open"])
    scored = working.dropna(subset=[correct_col, incumbent_correct_col])
    ci_week = week_blocked_bootstrap(
        scored,
        _paired_metric_fn(correct_col, incumbent_correct_col),
        block="week",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    ci_season = week_blocked_bootstrap(
        scored,
        _paired_metric_fn(correct_col, incumbent_correct_col),
        block="season",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    cand = _extract(ci_week, "candidate_accuracy")
    inc = _extract(ci_week, "incumbent_accuracy")
    delta_w = _extract(ci_week, "paired_delta")
    delta_s = _extract(ci_season, "paired_delta")
    return {
        "arm": name,
        "n_scored": len(scored),
        "accuracy": float(scored[correct_col].mean()),
        "week_ci_accuracy": [cand["lower"], cand["upper"]],
        "week_probability_positive_accuracy": cand["probability_positive"],
        "paired_delta_vs_chain_points": delta_w["estimate"] * 100.0,
        "paired_delta_week_ci_points": [delta_w["lower"] * 100.0, delta_w["upper"] * 100.0],
        "paired_delta_week_probability_positive": delta_w["probability_positive"],
        "paired_delta_season_ci_points": [delta_s["lower"] * 100.0, delta_s["upper"] * 100.0],
        "paired_delta_season_probability_positive": delta_s["probability_positive"],
        "incumbent_week_ci_reference": [inc["lower"], inc["upper"]],
    }


def build_protection_flags() -> tuple[set[tuple[str, str]], set[tuple[str, str]], dict[str, Any]]:
    """pbp08 protection-mismatch cell flags, rebuilt through the screen's own machinery."""

    schedules_path = _latest_schedules()
    pbp_snapshot = _latest_pbp_snapshot()
    population = load_population(schedules_path)
    offense, defense = build_game_trait_tables(pbp_snapshot)
    long_df = build_long_table(population, offense, defense)
    work, cells = build_cells(long_df)
    flag = cells["pbp08_protection_mismatch"]["flag"]
    flagged = set(
        zip(
            work.loc[flag, "game_id"].astype(str),
            work.loc[flag, "team"].astype(str),
            strict=True,
        )
    )
    data_ok = work["off_press_allow_w_q"].ge(0).to_numpy(dtype=bool) & work["opp_press_gen_w_q"].ge(
        0
    ).to_numpy(dtype=bool)
    available = set(
        zip(
            work.loc[data_ok, "game_id"].astype(str),
            work.loc[data_ok, "team"].astype(str),
            strict=True,
        )
    )
    info = {
        "schedules_snapshot": str(schedules_path),
        "pbp_snapshot": str(pbp_snapshot),
        "population_rows_team_games": len(work),
        "protection_flagged_team_games_2009_2025": len(flagged),
        "protection_data_available_team_games": len(available),
    }
    return flagged, available, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opener", type=Path, default=DEFAULT_OPENER)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--weak-stack-features", type=Path, default=DEFAULT_WEAK_STACK_FEATURES)
    parser.add_argument("--identity-features", type=Path, default=DEFAULT_IDENTITY_FEATURES)
    parser.add_argument("--incidents", type=Path, default=DEFAULT_INCIDENTS)
    parser.add_argument("--injuries-root", type=Path, default=DEFAULT_INJURIES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    args = parser.parse_args(argv)
    started = time.time()

    print(f"Loading opener archive and reloading market lines: {args.opener}")
    opener_raw = pd.read_parquet(args.opener)
    opener, line_coverage = reload_market_lines(
        args.market_root, args.weak_stack_features, opener_raw
    )
    opener["open_move_reloaded"] = opener["_reloaded_close"] - opener["_reloaded_tue_open"]

    print("Reconstructing coach-fade flip set and arrest flags (frozen machinery)")
    coach_flip_ids, schedule_snapshot = build_coach_flip_ids(opener, REPO / "data")
    arrest_indexed = build_arrest_flags(opener, args.identity_features, args.incidents)

    frame = opener.set_index("game_id", drop=False).copy()
    raw_pick = frame["pick_home_at_open_probability_rule"].astype(bool)
    chain_pick, arrest_flip_after_coach = compose_chain(
        raw_pick,
        coach_flip_ids,
        arrest_indexed["home_incident_flag"].astype(bool),
        arrest_indexed["away_incident_flag"].astype(bool),
    )
    frame["chain_pick"] = chain_pick
    frame["correct_chain"] = pick_correct(frame["chain_pick"], frame["margin_vs_open"])
    chain_measured = float(frame["correct_chain"].mean())
    if abs(chain_measured - PUBLISHED_SEQUENTIAL_CHAIN) > 1e-9:
        raise ValueError(
            f"incumbent chain reproduction FAILED: measured {chain_measured!r} "
            f"vs required {PUBLISHED_SEQUENTIAL_CHAIN!r}"
        )
    print(f"  arm (a) reproduction OK: {chain_measured!r}")

    print("Building protection-mismatch flags via pbp08 machinery")
    prot_flagged, prot_available, prot_info = build_protection_flags()

    print("Attaching NFL.com out counts (frozen refresh machinery)")
    snaps_path = latest(REPO / "data" / "players" / "raw", "*/snap_counts.parquet")
    out_counts, out_counts_info = build_out_counts(args.injuries_root, snaps_path)
    frame, counts_coverage = attach_counts(
        frame.reset_index(drop=True), out_counts, args.identity_features
    )

    picked_is_home = frame["chain_pick"].astype(bool)
    nflcom_picked_flag = pd.Series(
        np.where(picked_is_home, frame["_home_starter_out"], frame["_away_starter_out"]),
        index=frame.index,
    ).ge(2)
    nflcom_opp_flag = pd.Series(
        np.where(picked_is_home, frame["_away_starter_out"], frame["_home_starter_out"]),
        index=frame.index,
    ).ge(2)

    identity = pd.read_parquet(
        args.identity_features, columns=["game_id", "home_team", "away_team"]
    )
    identity["home_team"] = identity["home_team"].replace(TEAM_ABBREVIATION_ALIASES)
    identity["away_team"] = identity["away_team"].replace(TEAM_ABBREVIATION_ALIASES)
    team_labels = identity.set_index("game_id")
    home_key = list(
        zip(team_labels.index.astype(str), team_labels["home_team"].astype(str), strict=True)
    )
    away_key = list(
        zip(team_labels.index.astype(str), team_labels["away_team"].astype(str), strict=True)
    )
    frame["_home_prot_flag"] = (
        pd.Series([k in prot_flagged for k in home_key], index=team_labels.index)
        .reindex(frame.index)
        .fillna(False)
        .astype(bool)
    )
    frame["_away_prot_flag"] = (
        pd.Series([k in prot_flagged for k in away_key], index=team_labels.index)
        .reindex(frame.index)
        .fillna(False)
        .astype(bool)
    )
    frame["_home_prot_data"] = (
        pd.Series([k in prot_available for k in home_key], index=team_labels.index)
        .reindex(frame.index)
        .fillna(False)
        .astype(bool)
    )
    frame["_away_prot_data"] = (
        pd.Series([k in prot_available for k in away_key], index=team_labels.index)
        .reindex(frame.index)
        .fillna(False)
        .astype(bool)
    )

    frame["pick_a_chain"] = frame["chain_pick"]
    frame["pick_b_chain_movement"] = movement_overlay(
        frame["chain_pick"], frame["open_move_reloaded"], MOVEMENT_THRESHOLD
    )
    frame["pick_c_chain_movement_nflcom"], nflcom_flip = apply_overlay(
        frame["pick_b_chain_movement"], nflcom_picked_flag, nflcom_opp_flag
    )
    prot_picked_flag = pd.Series(
        np.where(picked_is_home, frame["_home_prot_flag"], frame["_away_prot_flag"]),
        index=frame.index,
    )
    prot_opp_flag = pd.Series(
        np.where(picked_is_home, frame["_away_prot_flag"], frame["_home_prot_flag"]),
        index=frame.index,
    )
    frame["pick_d_full_stack"], protection_flip = apply_overlay(
        frame["pick_c_chain_movement_nflcom"], prot_picked_flag, prot_opp_flag
    )
    frame["pick_e1_no_movement"], _ = apply_overlay(
        frame["chain_pick"], nflcom_picked_flag, nflcom_opp_flag
    )
    frame["pick_e1_no_movement"], _ = apply_overlay(
        frame["pick_e1_no_movement"], prot_picked_flag, prot_opp_flag
    )
    frame["pick_e2_no_nflcom"], _ = apply_overlay(
        frame["pick_b_chain_movement"], prot_picked_flag, prot_opp_flag
    )

    frame["correct_d"] = pick_correct(frame["pick_d_full_stack"], frame["margin_vs_open"])
    frame["movement_flip"] = frame["pick_b_chain_movement"].ne(frame["pick_a_chain"])
    frame["nflcom_flip"] = nflcom_flip
    frame["protection_flip"] = protection_flip
    frame["nflcom_covered"] = frame["season"].isin(list(NFLCOM_SEASONS))
    frame["all_four_sources"] = (
        frame["nflcom_covered"] & frame["_home_prot_data"] & frame["_away_prot_data"]
    )

    pick_columns = {
        "b_chain_movement": "pick_b_chain_movement",
        "c_chain_movement_nflcom": "pick_c_chain_movement_nflcom",
        "d_full_stack_slate_wide": "pick_d_full_stack",
        "e1_loo_without_movement": "pick_e1_no_movement",
        "e2_loo_without_nflcom": "pick_e2_no_nflcom",
        "e3_loo_without_protection": "pick_c_chain_movement_nflcom",
    }

    arms: list[dict[str, Any]] = [
        {
            "arm": "a_incumbent_chain",
            "n_scored": int(frame["correct_chain"].notna().sum()),
            "accuracy": chain_measured,
            "picks_changed_vs_chain": 0,
            "week_probability_positive_accuracy": float("nan"),
        }
    ]
    for name, col in pick_columns.items():
        arm = score_arm(frame, name=name, candidate_pick_col=col)
        arm["picks_changed_vs_chain"] = int(frame[col].ne(frame["pick_a_chain"]).sum())
        arms.append(arm)

    restricted = frame.loc[frame["all_four_sources"]].copy()
    print(
        f"Restricted all-four-sources population: {len(restricted)} games "
        f"({int(restricted['correct_chain'].notna().sum())} scored)"
    )
    restricted_arm = score_arm(
        restricted, name="d_full_stack_all_four_sources", candidate_pick_col="pick_d_full_stack"
    )
    restricted_arm["picks_changed_vs_chain"] = int(
        restricted["pick_d_full_stack"].ne(restricted["pick_a_chain"]).sum()
    )

    season_rows = []
    for season, group in frame.groupby("season"):
        g_scored = group.dropna(subset=["correct_chain"])
        season_rows.append(
            {
                "season": int(season),
                "games": len(group),
                "scored_games": len(g_scored),
                "movement_flips": int(group["movement_flip"].sum()),
                "nflcom_flips": int(group["nflcom_flip"].sum()),
                "protection_flips": int(group["protection_flip"].sum()),
                "a_chain_accuracy": float(g_scored["correct_chain"].mean()),
                "d_full_stack_accuracy": float(g_scored["correct_d"].mean()),
                "d_delta_points_vs_chain": float(
                    (g_scored["correct_d"] - g_scored["correct_chain"]).mean()
                )
                * 100.0,
                "all_four_sources_games": int(group["all_four_sources"].sum()),
            }
        )

    primary = next(a for a in arms if a["arm"] == "d_full_stack_slate_wide")
    evidence = (
        f"Week-blocked paired delta vs incumbent chain = "
        f"{primary['paired_delta_vs_chain_points']:+.4f} accuracy points "
        f"[{primary['paired_delta_week_ci_points'][0]:+.4f}, "
        f"{primary['paired_delta_week_ci_points'][1]:+.4f}], "
        f"P+ {primary['paired_delta_week_probability_positive']:.4f}; season-blocked "
        f"[{primary['paired_delta_season_ci_points'][0]:+.4f}, "
        f"{primary['paired_delta_season_ci_points'][1]:+.4f}] "
        f"P+ {primary['paired_delta_season_probability_positive']:.4f}. Attribution on "
        "already-scored archives (upper bound); three of four stacked components were "
        "selected/red-teamed on overlapping seasons; no terminal ground met."
    )
    notes = (
        "MAX-EV composition (owner task 2026-08-22): chain raw->coach fade->arrests plus "
        "movement rule, NFL.com Friday out>=2 starter fade, and PBP-08 protection tilt, "
        "in that order, each applied where its data exists else no-op; rules frozen in "
        "docs/max_ev_composition.md before scoring; seed 20260824, 20000 samples, week "
        "primary / season secondary; never pool component effects as independent."
    )
    record_line = (
        f"nfl-ats weak-signals record --name maxev_full_stack --league nfl "
        f"--effect-units accuracy_points "
        f"--effect {primary['paired_delta_vs_chain_points']:.6f} "
        f"--interval-low {primary['paired_delta_week_ci_points'][0]:.6f} "
        f"--interval-high {primary['paired_delta_week_ci_points'][1]:.6f} "
        f"--probability-positive {primary['paired_delta_week_probability_positive']:.6f} "
        f"--sample-games {primary['n_scored']} "
        f"--season-start 2020 --season-end 2025 "
        f"--classification unresolved_below_power "
        f"--source artifacts/max_ev_composition/<run_id>/metadata.json; docs/max_ev_composition.md "
        f'--description "FULL MAX-EV card slate-wide partial application: sequential chain '
        f"(raw -> coach fade -> arrests) plus movement rule plus NFL.com Friday out>=2 "
        f"starter-caliber fade plus PBP-08 protection-mismatch tilt, each edge firing only "
        f'where its data exists, graded at the frozen Tuesday line vs the un-composed chain." '
        f'--classification-evidence "{evidence}" '
        f'--notes "{notes}"'
    )

    output_dir = args.output_root / run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        "game_id",
        "season",
        "week",
        "margin_vs_open",
        "open_move_reloaded",
        "pick_a_chain",
        "pick_b_chain_movement",
        "pick_c_chain_movement_nflcom",
        "pick_d_full_stack",
        "pick_e1_no_movement",
        "pick_e2_no_nflcom",
        "correct_chain",
        "correct_d",
        "movement_flip",
        "nflcom_flip",
        "protection_flip",
        "nflcom_covered",
        "_home_prot_flag",
        "_away_prot_flag",
        "_home_prot_data",
        "_away_prot_data",
        "all_four_sources",
    ]
    atomic_parquet(frame.reset_index(drop=True)[keep_cols], output_dir / "per_game.parquet")
    atomic_csv(pd.DataFrame(season_rows), output_dir / "season_summary.csv")
    atomic_csv(pd.DataFrame([*arms, restricted_arm]), output_dir / "arms_summary.csv")

    configuration = {
        "command": "max-ev-composition",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "movement_threshold": MOVEMENT_THRESHOLD,
        "opener_archive": str(args.opener),
        "market_root": str(args.market_root),
        "weak_stack_features": str(args.weak_stack_features),
        "identity_features": str(args.identity_features),
        "incidents": str(args.incidents),
        "injuries_root": str(args.injuries_root),
        "snap_counts": str(snaps_path),
    }
    edge_counts = {
        "movement_eligible_ge_1_0_games": int(
            frame["open_move_reloaded"].abs().ge(MOVEMENT_THRESHOLD).sum()
        ),
        "movement_flips_vs_chain": int(frame["movement_flip"].sum()),
        "nflcom_covered_games": int(frame["nflcom_covered"].sum()),
        "nflcom_flips_vs_prior_arm": int(frame["nflcom_flip"].sum()),
        "protection_flagged_picked_games": int(prot_picked_flag.sum()),
        "protection_flips_vs_prior_arm": int(frame["protection_flip"].sum()),
        "all_four_sources_games": int(frame["all_four_sources"].sum()),
    }
    gaps_2022_2024 = sorted(
        c
        for c in counts_coverage["archive_season_weeks_absent_from_snapshot"]
        if c[:4] in {"2022", "2023", "2024"}
    )
    metadata: dict[str, Any] = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **configuration,
        "input_hashes": {
            "opener_archive": sha256_file(args.opener),
            "identity_features": sha256_file(args.identity_features),
            "incidents": sha256_file(args.incidents),
            "snap_counts": sha256_file(snaps_path),
        },
        "line_reload_verification": dict(line_coverage),
        "chain_reproduction": {
            "measured": chain_measured,
            "required": PUBLISHED_SEQUENTIAL_CHAIN,
            "coach_flips": len(coach_flip_ids),
            "arrest_flips_after_coach": int(arrest_flip_after_coach.sum()),
            "schedule_snapshot": schedule_snapshot,
        },
        "protection_source_info": prot_info,
        "nflcom_out_count_source": out_counts_info,
        "nflcom_week_coverage_gaps_within_2022_2024": gaps_2022_2024,
        "edge_application_counts": edge_counts,
        "arms": [*arms, restricted_arm],
        "season_summary": season_rows,
        "disclosures": {
            "attribution_upper_bound": (
                "Attribution on already-scored archives ONLY. Three of the four stacked "
                "components were selected/red-teamed on seasons this archive overlaps "
                "(movement and arrests registered on 2020-2025; NFL.com fade selected and "
                "red-teamed on exactly 2022-2024; the protection cell was mined on "
                "2009-2025). The composed numbers are an upper bound, continuous evidence, "
                "never a fresh confirmation, and components must never be pooled as "
                "independent. No rotation-registry window was spent."
            ),
            "effective_n_partial_application": (
                "Movement exists on every archived game (the archive is conditioned on a "
                "resolved tue_open/close pair). NFL.com exists only for snapshot-covered "
                "season-weeks of 2022-2024 (~3 of 6 seasons); outside coverage the overlay "
                "is a no-op, never a flip. Protection tilt exists where pbp08 assigns both "
                "teams' relevant quartile rows (early-season incomplete windows and "
                "screen-population pushes are no-ops). Effective n per arm equals its "
                "scored non-push games; per-season edge counts are in season_summary."
            ),
            "two_full_stack_framings": (
                "d_full_stack_slate_wide applies each edge wherever its data exists (what "
                "would actually ship); d_full_stack_all_four_sources restricts to games "
                "where ALL FOUR sources exist (cleanest comparison). Both are reported."
            ),
            "tie_handling": (
                "Both later overlays flip iff the PICKED team is flagged AND the opponent "
                "is not; both flagged keeps the incoming pick. Frozen before scoring."
            ),
            "pushes_preserved": (
                "Graded with nfl_ats.clv.pick_correct against margin_vs_open; pushes are "
                "NaN and excluded from every arm identically."
            ),
        },
        "proposed_weak_signal_record": {
            "name": "maxev_full_stack",
            "classification": "unresolved_below_power",
            "effect_accuracy_points": primary["paired_delta_vs_chain_points"],
            "interval_low_week": primary["paired_delta_week_ci_points"][0],
            "interval_high_week": primary["paired_delta_week_ci_points"][1],
            "probability_positive_week": primary["paired_delta_week_probability_positive"],
            "probability_positive_season": primary["paired_delta_season_probability_positive"],
            "sample_games": primary["n_scored"],
            "command_line": record_line,
            "classification_evidence": evidence,
            "notes": notes,
        },
        "provenance": artifact_provenance(configuration, args.opener, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "metadata.json",
        metadata,
        command="max-ev-composition",
        metrics={
            "chain_reproduction": {"measured": chain_measured},
            "arms": {
                arm["arm"]: {
                    "n_scored": arm["n_scored"],
                    "accuracy": arm["accuracy"],
                    "picks_changed_vs_chain": arm.get("picks_changed_vs_chain", 0),
                    "paired_delta_vs_chain_points": arm.get("paired_delta_vs_chain_points"),
                    "paired_delta_week_probability_positive": arm.get(
                        "paired_delta_week_probability_positive"
                    ),
                    "paired_delta_season_probability_positive": arm.get(
                        "paired_delta_season_probability_positive"
                    ),
                }
                for arm in [*arms, restricted_arm]
            },
            "edge_application_counts": edge_counts,
        },
        notes=notes,
        source="scripts/max_ev_composition.py",
        weak_signal_name=None,
        registry_root=args.registry_root,
    )

    print("\n=== Arm table (slate-wide partial application, graded at the Tuesday line) ===")
    header = (
        f"{'arm':30s} {'n':>5s} {'acc':>8s} {'chg':>4s} {'d_pts':>8s} {'wk_P+':>7s} {'se_P+':>7s}"
    )
    print(header)
    for arm in arms:
        print(
            f"{arm['arm']:30s} {arm['n_scored']:5d} {arm['accuracy'] * 100:7.4f}% "
            f"{arm.get('picks_changed_vs_chain', 0):4d} "
            f"{arm.get('paired_delta_vs_chain_points', 0.0):+7.4f} "
            f"{arm.get('paired_delta_week_probability_positive', float('nan')):7.4f} "
            f"{arm.get('paired_delta_season_probability_positive', float('nan')):7.4f}"
        )
    print("\n=== Restricted framing (games where ALL FOUR sources exist) ===")
    print(
        f"{restricted_arm['arm']:30s} {restricted_arm['n_scored']:5d} "
        f"{restricted_arm['accuracy'] * 100:7.4f}% "
        f"{restricted_arm['picks_changed_vs_chain']:4d} "
        f"{restricted_arm['paired_delta_vs_chain_points']:+7.4f} "
        f"{restricted_arm['paired_delta_week_probability_positive']:7.4f} "
        f"{restricted_arm['paired_delta_season_probability_positive']:7.4f}"
    )
    print("\n=== Proposed weak-signals record line (central recording only) ===")
    print(record_line)
    print(f"\nelapsed: {time.time() - started:.1f}s")
    print(f"artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
