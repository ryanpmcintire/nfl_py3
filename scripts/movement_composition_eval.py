"""Attribution-only composition study: observed-movement rule on the production chain.

Question (owner task, 2026-08-22): production today composes raw model ->
coach fade -> player-arrest policy (53.76% historical opener grade). The
observed-movement rule (`observed_movement_threshold_1_0`: if the current
line moved >= 1.0 pt from the frozen Tuesday line, follow the market side;
else keep the pick; docs/observed_movement_channel.md, +1.863 pts P+ 0.935
solo) is challenger-tracked only. This script measures the COMPOSED chain on
the same paired opener archive used everywhere.

Arms:
  a. incumbent chain reproduction check (raw -> coach fade -> arrest,
     sequential, decision_policy_id coach_fade_then_player_arrests_v1)
  b. chain + movement rule (threshold >= 1.0 on close - tue_open)
  c. movement rule solo reference (threshold >= 1.0 on the raw model pick)

Lines are reloaded from the market archive through the exact path
`opener_pick_evaluation` uses -- `build_pairing_table(root,
capture_kind=HISTORICAL_CAPTURE_KIND, labels=("tue_open", *CLOSE_LABEL_PRIORITY))
+ `close_reference_table` -- which is where docs/observed_movement_channel.md's
script gets its lines; they are verified against the archive's own columns.
No model is refit, no window spent: attribution + already-captured archive only.

Bootstrap: nfl_ats.clv.week_blocked_bootstrap, 20,000 samples, seed 20260822,
block="week" primary and block="season" secondary, paired deltas in accuracy
points, full slate. Disclosures written into the artifact: close-grade line
availability limits which seasons have movement data at all (the archive is
conditioned on a resolvable close, so effective n per arm equals its scored
games per season), and attribution on already-looked-at data is an upper
bound, not a fresh confirmation.

Writes artifacts/movement_composition_eval/<run_id>/ and stamps
registry/experiments/movement-composition-eval/. Does NOT write either
registry JSON (weak_signals.json / rotation_registry.json); the proposed
`nfl-ats weak-signals record` line for `movement_rule_composed_chain` is
printed and saved in metadata.json only.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from player_arrests_policy_eval import apply_frozen_policy, broad_incident_game_flags

from nfl_ats.clv import (
    CLOSE_LABEL_PRIORITY,
    HISTORICAL_CAPTURE_KIND,
    build_pairing_table,
    close_reference_table,
    pick_correct,
    week_blocked_bootstrap,
)
from nfl_ats.io import atomic_csv, atomic_parquet, run_id
from nfl_ats.modeling import regular_season_rows
from nfl_ats.provenance import artifact_provenance, sha256_file, write_experiment_artifact
from nfl_ats.snapshots import latest_snapshot, load_snapshot

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OPENER = REPO / "artifacts/opener_evaluation/20260819T174244Z/per_game.parquet"
DEFAULT_MARKET_ROOT = REPO / "data/market/raw"
DEFAULT_WEAK_STACK_FEATURES = REPO / "data/processed/game_features_weak_stack.parquet"
DEFAULT_IDENTITY_FEATURES = REPO / "data/processed/game_features_pbp.parquet"
DEFAULT_INCIDENTS = (
    REPO / "data/raw/player_arrests/20260820T153000Z/incidents_point_in_time.parquet"
)
DEFAULT_OUTPUT_ROOT = REPO / "artifacts/movement_composition_eval"
DEFAULT_REGISTRY_ROOT = REPO / "registry"

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260822
MOVEMENT_THRESHOLD = 1.0
PUBLISHED_CHAIN_CANDIDATE = 0.5375914836992681
PUBLISHED_SEQUENTIAL_CHAIN = 0.541583499667332
PUBLISHED_BASELINE = 0.5335994677312043
REPRODUCTION_TOLERANCE = 1e-9


def reload_market_lines(
    market_root: Path,
    weak_stack_features: Path,
    opener: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reload tue_open/close from the market archive exactly as opener_pick_evaluation does."""

    features = regular_season_rows(pd.read_parquet(weak_stack_features))
    pairing = build_pairing_table(
        market_root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=features,
    )
    close = close_reference_table(pairing, features)
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "home_spread"]
    ].rename(columns={"home_spread": "_reloaded_tue_open"})
    close_slim = close[["game_id", "close_home_spread"]].rename(
        columns={"close_home_spread": "_reloaded_close"}
    )
    lines = tue_open.merge(close_slim, on="game_id", how="inner")
    merged = opener.merge(lines, on="game_id", how="left", validate="one_to_one")
    if merged[["_reloaded_tue_open", "_reloaded_close"]].isna().any().any():
        raise ValueError("market-archive reload did not cover every archived game")
    open_match = np.allclose(
        merged["_reloaded_tue_open"], merged["tue_open_home_spread"], atol=1e-12
    )
    close_match = np.allclose(merged["_reloaded_close"], merged["close_home_spread"], atol=1e-12)
    if not (open_match and close_match):
        raise ValueError(
            f"reloaded lines disagree with the archive (open={open_match}, close={close_match})"
        )
    coverage = {
        "games": len(merged),
        "open_lines_match_archive": bool(open_match),
        "close_lines_match_archive": bool(close_match),
    }
    return merged, coverage


def build_coach_flip_ids(opener: pd.DataFrame, data_root: Path) -> tuple[set[str], str]:
    """Coach-fade flip set via the frozen overlay, on predictions rebuilt from the archive."""

    snapshot = latest_snapshot(data_root / "raw")
    schedules, _team_stats = load_snapshot(snapshot)
    sched_cols = schedules[["game_id", "home_team", "away_team", "game_type"]].drop_duplicates(
        "game_id"
    )
    predictions = opener[
        ["game_id", "season", "week", "tue_open_home_spread", "home_cover_probability_at_open"]
    ].merge(sched_cols, on="game_id", how="left", validate="one_to_one")
    missing_meta = predictions["home_team"].isna() | predictions["game_type"].isna()
    if missing_meta.any():
        raise ValueError(f"{int(missing_meta.sum())} games lack schedule metadata")
    non_reg = predictions.loc[predictions["game_type"].ne("REG")]
    if not non_reg.empty:
        raise ValueError(f"archive contains non-REG games: {non_reg['game_id'].tolist()}")
    predictions = predictions.rename(
        columns={
            "tue_open_home_spread": "spread_line",
            "home_cover_probability_at_open": "home_cover_probability",
        }
    ).reset_index(drop=True)

    from nfl_ats.coach_fade_overlay import apply_coach_fade_overlay

    result = apply_coach_fade_overlay(predictions, schedules)
    flip_ids = {flip.game_id for flip in result.flips}
    return flip_ids, snapshot.root.name


def build_arrest_flags(
    opener: pd.DataFrame, identity_features: Path, incidents_path: Path
) -> pd.DataFrame:
    """Broad incident flags via the frozen arrest-eval machinery, joined to the archive."""

    identity = pd.read_parquet(
        identity_features, columns=["game_id", "gameday", "home_team", "away_team"]
    )
    identity = identity.loc[identity["game_id"].isin(opener["game_id"])].copy()
    if len(identity) != len(opener):
        raise ValueError(
            f"arrest identity join covers {len(identity)} of {len(opener)} baseline rows"
        )
    incidents = pd.read_parquet(incidents_path, columns=["record_id", "incident_date", "team"])
    flags, _coverage = broad_incident_game_flags(identity, incidents)
    scored = apply_frozen_policy(opener, flags)
    return scored.set_index("game_id")


def compose_chain(
    raw_pick: pd.Series,
    coach_flip_ids: set[str],
    home_flag: pd.Series,
    away_flag: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Sequential incumbent chain: raw pick -> coach fade complement -> arrest back-side flip."""

    chain_pick = raw_pick.copy()
    coach_hits = chain_pick.index.isin(coach_flip_ids)
    chain_pick.loc[coach_hits] = ~chain_pick.loc[coach_hits]
    exactly_one = home_flag ^ away_flag
    opposes = chain_pick.ne(home_flag)
    arrest_flip = exactly_one & opposes
    final_pick = chain_pick.where(~arrest_flip, home_flag)
    return final_pick.astype(bool), arrest_flip


def movement_overlay(pick: pd.Series, open_move: pd.Series, threshold: float) -> pd.Series:
    """Flip to the movement side when |move| clears ``threshold``; else keep the incoming pick."""

    eligible = open_move.abs().ge(threshold)
    movement_home = open_move.gt(0.0)
    out = pick.where(~eligible, movement_home)
    return out.astype(bool)


def _paired_metric_fn(candidate_col: str, incumbent_col: str):
    def _metric(rows: pd.DataFrame) -> dict[str, float]:
        cand = rows[candidate_col].astype(float)
        prod = rows[incumbent_col].astype(float)
        both = rows.loc[cand.notna() & prod.notna()]
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
    incumbent_correct_col: str = "correct_a",
) -> dict[str, Any]:
    working = frame.copy()
    correct_col = f"_correct_{name}"
    working[correct_col] = pick_correct(working[candidate_pick_col], working["margin_vs_open"])
    ci_week = week_blocked_bootstrap(
        working.dropna(subset=[correct_col, incumbent_correct_col]),
        _paired_metric_fn(correct_col, incumbent_correct_col),
        block="week",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    ci_season = week_blocked_bootstrap(
        working.dropna(subset=[correct_col, incumbent_correct_col]),
        _paired_metric_fn(correct_col, incumbent_correct_col),
        block="season",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    both = working.dropna(subset=[correct_col, incumbent_correct_col])
    cand = _extract(ci_week, "candidate_accuracy")
    inc = _extract(ci_week, "incumbent_accuracy")
    delta_w = _extract(ci_week, "paired_delta")
    delta_s = _extract(ci_season, "paired_delta")
    return {
        "arm": name,
        "n_scored": len(both),
        "accuracy": float(both[correct_col].mean()),
        "week_ci": [cand["lower"], cand["upper"]],
        "week_probability_positive_accuracy": cand["probability_positive"],
        "paired_delta_vs_incumbent_points": delta_w["estimate"] * 100.0,
        "paired_delta_week_ci_points": [delta_w["lower"] * 100.0, delta_w["upper"] * 100.0],
        "paired_delta_week_probability_positive": delta_w["probability_positive"],
        "paired_delta_season_ci_points": [delta_s["lower"] * 100.0, delta_s["upper"] * 100.0],
        "paired_delta_season_probability_positive": delta_s["probability_positive"],
        "incumbent_week_ci_reference": [inc["lower"], inc["upper"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opener", type=Path, default=DEFAULT_OPENER)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--weak-stack-features", type=Path, default=DEFAULT_WEAK_STACK_FEATURES)
    parser.add_argument("--identity-features", type=Path, default=DEFAULT_IDENTITY_FEATURES)
    parser.add_argument("--incidents", type=Path, default=DEFAULT_INCIDENTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    args = parser.parse_args(argv)
    started = time.time()

    print(f"Loading baseline archive: {args.opener}")
    opener_raw = pd.read_parquet(args.opener)

    print("Reloading tue_open/close from the market archive (observed_movement_channel path)")
    opener, line_coverage = reload_market_lines(
        args.market_root, args.weak_stack_features, opener_raw
    )
    opener["open_move_reloaded"] = opener["_reloaded_close"] - opener["_reloaded_tue_open"]
    move_matches = np.allclose(opener["open_move_reloaded"], opener["open_move"], atol=1e-12)
    print(f"  reloaded open_move matches archive column: {move_matches}")

    print("Reconstructing coach-fade flip set (frozen overlay)")
    coach_flip_ids, schedule_snapshot = build_coach_flip_ids(opener, REPO / "data")
    print(f"  coach fade flips: {len(coach_flip_ids)}")

    print("Reconstructing arrest flags (frozen policy machinery)")
    arrest_indexed = build_arrest_flags(opener, args.identity_features, args.incidents)
    home_flag = arrest_indexed["home_incident_flag"].astype(bool)
    away_flag = arrest_indexed["away_incident_flag"].astype(bool)
    baseline_flips = arrest_indexed["policy_flip"].astype(bool)

    frame = opener.set_index("game_id", drop=False).copy()
    raw_pick = frame["pick_home_at_open_probability_rule"].astype(bool)
    chain_pick, arrest_flip_after_coach = compose_chain(
        raw_pick, coach_flip_ids, home_flag, away_flag
    )
    movement_b = movement_overlay(chain_pick, frame["open_move_reloaded"], MOVEMENT_THRESHOLD)
    movement_c = movement_overlay(raw_pick, frame["open_move_reloaded"], MOVEMENT_THRESHOLD)

    frame["pick_a_chain"] = chain_pick
    frame["pick_b_chain_movement"] = movement_b
    frame["pick_c_raw_movement"] = movement_c
    frame["arrest_flip_after_coach"] = arrest_flip_after_coach
    frame["baseline_arrest_flip"] = baseline_flips.reindex(frame.index).fillna(False).astype(bool)
    frame["movement_eligible"] = frame["open_move_reloaded"].abs().ge(MOVEMENT_THRESHOLD)

    frame["correct_a"] = pick_correct(frame["pick_a_chain"], frame["margin_vs_open"])
    frame["correct_b"] = pick_correct(frame["pick_b_chain_movement"], frame["margin_vs_open"])
    frame["correct_c"] = pick_correct(frame["pick_c_raw_movement"], frame["margin_vs_open"])

    valid_base = frame["correct_at_open_probability_rule"]
    arrest_on_baseline_correct = np.where(
        frame["baseline_arrest_flip"], 1.0 - valid_base, valid_base
    )
    arrest_on_baseline_accuracy = float(
        pd.Series(arrest_on_baseline_correct, index=frame.index)[valid_base.notna()].mean()
    )

    reproduction = {
        "published_sequential_chain_reference": PUBLISHED_SEQUENTIAL_CHAIN,
        "published_arrest_on_baseline_reference": PUBLISHED_CHAIN_CANDIDATE,
        "published_baseline": PUBLISHED_BASELINE,
        "chain_accuracy_measured": float(frame["correct_a"].mean()),
        "chain_matches_overlay_subset_composition_sequential": bool(
            abs(float(frame["correct_a"].mean()) - PUBLISHED_SEQUENTIAL_CHAIN) <= 1e-9
        ),
        "arrest_on_baseline_accuracy_measured": arrest_on_baseline_accuracy,
        "arrest_on_baseline_matches_published_53_76": bool(
            abs(arrest_on_baseline_accuracy - PUBLISHED_CHAIN_CANDIDATE) <= 1e-9
        ),
        "raw_baseline_accuracy_measured": float(valid_base.mean()),
        "raw_baseline_matches_published": bool(
            abs(float(valid_base.mean()) - PUBLISHED_BASELINE) <= 1e-9
        ),
        "coach_flips": len(coach_flip_ids),
        "baseline_arrest_flips": int(baseline_flips.sum()),
        "arrest_flips_after_coach": int(arrest_flip_after_coach.sum()),
        "n_scored_chain": int(frame["correct_a"].notna().sum()),
        "published_n_scored": 1503,
        "note": (
            "Two published references exist: the sequential coach->arrest chain figure "
            "(0.5415835, artifacts/overlay_subset_composition production_chain_reference."
            "coach_then_arrest_sequential) and the published 53.76% headline, which was "
            "measured by applying the arrest policy directly to the frozen 53.36% baseline "
            "(docs/player_arrests_policy_eval.md). Both are reproduced here; the incumbent "
            "arm composes sequentially, matching card_view's live order."
        ),
    }

    arms = [
        score_arm(frame, name="a_incumbent_chain", candidate_pick_col="pick_a_chain"),
        score_arm(
            frame, name="b_chain_plus_movement_1_0", candidate_pick_col="pick_b_chain_movement"
        ),
        score_arm(frame, name="c_movement_solo_1_0", candidate_pick_col="pick_c_raw_movement"),
    ]
    arms[0]["picks_changed_vs_raw"] = int((frame["pick_a_chain"] != raw_pick).sum())
    b_changed = int((frame["pick_b_chain_movement"] != frame["pick_a_chain"]).sum())
    c_changed = int((frame["pick_c_raw_movement"] != raw_pick).sum())
    arms[1]["picks_changed_vs_arm_a"] = b_changed
    arms[2]["picks_changed_vs_raw"] = c_changed

    season_rows = []
    for season, group in frame.groupby("season"):
        season_rows.append(
            {
                "season": int(season),
                "games": len(group),
                "scored_games": int(group["correct_a"].notna().sum()),
                "movement_eligible_ge_1_0": int(group["movement_eligible"].sum()),
                "a_chain_accuracy": float(group["correct_a"].mean()),
                "b_chain_plus_movement_accuracy": float(group["correct_b"].mean()),
                "c_movement_solo_accuracy": float(group["correct_c"].mean()),
            }
        )

    primary = arms[1]
    record_line = (
        "nfl-ats weak-signals record "
        "--name movement_rule_composed_chain "
        '--description "Observed-movement rule (|close-tue_open|>=1.0, follow market side) '
        "applied ON TOP of the composed production chain raw model -> coach fade -> "
        "player-arrests policy; paired full-slate accuracy delta vs the un-composed incumbent "
        'chain on the same games, graded at the frozen Tuesday line." '
        "--source artifacts/movement_composition_eval/<run_id>/metadata.json; "
        "docs/movement_composition_eval.md "
        "--effect-units accuracy_points "
        "--classification unresolved_below_power "
        "--league nfl --season-start 2020 --season-end 2025 "
    )
    record_values = (
        f"[values] --effect {primary['paired_delta_vs_incumbent_points']:.6f} "
        f"--interval-low {primary['paired_delta_week_ci_points'][0]:.6f} "
        f"--interval-high {primary['paired_delta_week_ci_points'][1]:.6f} "
        f"--probability-positive {primary['paired_delta_week_probability_positive']:.6f} "
        f"--sample-games {primary['n_scored']} "
    )
    evidence = (
        f"Week-blocked paired delta vs incumbent chain = "
        f"{primary['paired_delta_vs_incumbent_points']:+.4f} accuracy points "
        f"[{primary['paired_delta_week_ci_points'][0]:+.4f}, "
        f"{primary['paired_delta_week_ci_points'][1]:+.4f}], "
        f"P+ {primary['paired_delta_week_probability_positive']:.4f}; season-blocked "
        f"interval [{primary['paired_delta_season_ci_points'][0]:+.4f}, "
        f"{primary['paired_delta_season_ci_points'][1]:+.4f}] "
        f"P+ {primary['paired_delta_season_probability_positive']:.4f}. Attribution on "
        "already-looked-at data (upper bound), no window spent, no terminal ground met."
    )
    notes = (
        "Attribution-only composition study (owner task 2026-08-22); seed 20260822, "
        "20000 samples, week primary / season secondary; close-grade line availability limits "
        "which seasons have movement data at all (effective n per arm disclosed in artifact); "
        "movement-rule solo reference arm reproduces observed_movement_threshold_1_0 design at "
        "this seed; NOT written to registry JSON by the script."
    )

    output_dir = args.output_root / run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(frame.reset_index(drop=True), output_dir / "per_game.parquet")
    atomic_csv(pd.DataFrame(season_rows), output_dir / "season_summary.csv")
    cells_frame = pd.DataFrame(arms)
    atomic_csv(cells_frame, output_dir / "arms_summary.csv")

    configuration = {
        "command": "movement-composition-eval",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "movement_threshold": MOVEMENT_THRESHOLD,
        "opener_archive": str(args.opener),
        "market_root": str(args.market_root),
        "weak_stack_features": str(args.weak_stack_features),
        "identity_features": str(args.identity_features),
        "incidents": str(args.incidents),
    }
    metadata: dict[str, Any] = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **configuration,
        "input_hashes": {
            "opener_archive": sha256_file(args.opener),
            "identity_features": sha256_file(args.identity_features),
            "incidents": sha256_file(args.incidents),
        },
        "line_reload_verification": {
            **line_coverage,
            "open_move_matches_archive": bool(move_matches),
        },
        "schedule_snapshot": schedule_snapshot,
        "reproduction_check": reproduction,
        "arms": arms,
        "season_summary": season_rows,
        "disclosures": {
            "attribution_upper_bound": (
                "This is attribution on already-looked-at data: the movement rule, the coach "
                "fade, and the arrest policy were each selected/registered using windows this "
                "2020-2025 archive covers. The composed numbers are an upper bound, continuous "
                "evidence, not a fresh confirmation. No rotation-registry window was spent."
            ),
            "close_grade_availability": (
                "Movement data exists only where the market archive resolved BOTH a Tuesday "
                "opener and a close; the opener archive itself is conditioned on that pair, so "
                "every season present here carries movement data and seasons without close "
                "coverage never entered the population. Effective n per arm equals each arm's "
                "scored (non-push) games; per-season counts are in season_summary."
            ),
            "pushes_preserved": (
                "Graded with nfl_ats.clv.pick_correct against margin_vs_open; pushes are NaN "
                "and excluded from every arm identically."
            ),
        },
        "proposed_weak_signal_record": {
            "name": "movement_rule_composed_chain",
            "classification": "unresolved_below_power",
            "effect_accuracy_points": primary["paired_delta_vs_incumbent_points"],
            "interval_low_week": primary["paired_delta_week_ci_points"][0],
            "interval_high_week": primary["paired_delta_week_ci_points"][1],
            "probability_positive_week": primary["paired_delta_week_probability_positive"],
            "probability_positive_season": primary["paired_delta_season_probability_positive"],
            "sample_games": primary["n_scored"],
            "classification_evidence": evidence,
            "notes": notes,
        },
        "provenance": artifact_provenance(configuration, args.opener, project_root=REPO),
    }
    write_experiment_artifact(
        output_dir,
        "metadata.json",
        metadata,
        command="movement-composition-eval",
        metrics={
            "reproduction_check": reproduction,
            "arms": {
                arm["arm"]: {
                    "n_scored": arm["n_scored"],
                    "accuracy": arm["accuracy"],
                    "paired_delta_vs_incumbent_points": arm["paired_delta_vs_incumbent_points"],
                    "paired_delta_week_probability_positive": arm[
                        "paired_delta_week_probability_positive"
                    ],
                    "paired_delta_season_probability_positive": arm[
                        "paired_delta_season_probability_positive"
                    ],
                }
                for arm in arms
            },
        },
        notes=notes,
        source="scripts/movement_composition_eval.py",
        weak_signal_name=None,
        registry_root=args.registry_root,
    )

    print("\n=== Arm table (full slate, graded at the frozen Tuesday line) ===")
    header = f"{'arm':34s} {'n':>5s} {'acc':>8s} {'d_pts':>8s} {'wk_P+':>7s} {'se_P+':>7s}"
    print(header)
    for arm in arms:
        print(
            f"{arm['arm']:34s} {arm['n_scored']:5d} {arm['accuracy'] * 100:7.4f}% "
            f"{arm['paired_delta_vs_incumbent_points']:+7.4f} "
            f"{arm['paired_delta_week_probability_positive']:7.4f} "
            f"{arm['paired_delta_season_probability_positive']:7.4f}"
        )
    print(f"\nReproduction: {reproduction}")
    print(f"\nRecord line ({record_values.strip()}):\n  {record_line}")
    print(f"\nclassification_evidence: {evidence}")
    print(f"\nelapsed: {time.time() - started:.1f}s")
    print(f"artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
