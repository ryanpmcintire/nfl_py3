"""Refresh-path composition study: NFL.com Friday out>=2 fade ON the production chain.

Question (owner task, 2026-08-22): the red-team-proofed lead
(`redteam_nflcom_out2_starters_only`, -1.2748 pts full-slate scaled,
P(neg)=0.9995, docs/edge_audit_redteam.md Claim 2) is challenger-tracked only.
This script measures the signal AS THE LATE-WEEK REFRESH PATH WOULD ACTUALLY
PLAY IT: composed on top of the production chain (raw model -> coach fade ->
player-arrests policy) at the Saturday cutoff, mirroring the structure of
scripts/movement_composition_eval.py, graded at the frozen Tuesday line.

Population: the paired opener archive RESTRICTED to seasons 2022-2024, the
only seasons where the immutable NFL.com snapshot
(data/raw/nflcom_injuries/) has coverage; effective n is disclosed per arm.
Flags are built with the IDENTICAL normalization, starter proxy, and join
machinery as scripts/nflcom_friday_designation_screen.py (imported, not
reimplemented).

Arms:
  a. incumbent chain reproduction (reproduction gate on the FULL archive
     against 0.541583499667332 before restriction)
  b. chain + Out>=2-on-starter-caliber fade overlay
  c1. chain + Out>=1-on-starter-caliber fade overlay (milder)
  c2. chain + net total-Out differential (picked - opp >= 1) overlay

Overlay rule frozen in docs/nflcom_friday_refresh.md BEFORE scoring: flip to
the opponent iff the picked team is flagged AND the opponent is not; both
flagged keeps the incumbent pick.

Bootstrap: nfl_ats.clv.week_blocked_bootstrap, 20,000 samples, seed 20260823,
block="week" primary and block="season" secondary, paired deltas vs arm a in
accuracy points, full slate, pushes excluded identically. No window spend:
Saturday-cutoff attribution on already-looked-at data only.

Writes artifacts/nflcom_friday_refresh/<run_id>/ and stamps
registry/experiments/nflcom-friday-refresh/. Does NOT write either registry
JSON (weak_signals.json / rotation_registry.json); proposed
``nfl-ats weak-signals record`` lines are printed and saved in metadata.json.
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
    DEFAULT_OPENER,
    PUBLISHED_SEQUENTIAL_CHAIN,
    build_arrest_flags,
    build_coach_flip_ids,
    compose_chain,
)
from nflcom_friday_designation_screen import (
    build_starter_keys,
    initial_last_key,
    latest,
    load_report_flags,
)

from nfl_ats.clv import pick_correct, week_blocked_bootstrap
from nfl_ats.io import atomic_csv, atomic_parquet, run_id
from nfl_ats.provenance import artifact_provenance, sha256_file, write_experiment_artifact

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO / "artifacts/nflcom_friday_refresh"
DEFAULT_REGISTRY_ROOT = REPO / "registry"
DEFAULT_INJURIES_ROOT = REPO / "data/raw/nflcom_injuries"

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260823
SEASON_START = 2022
SEASON_END = 2024


def build_incumbent_chain(
    opener_path: Path,
    identity_features: Path,
    incidents_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct the sequential production chain on the full opener archive."""

    opener = pd.read_parquet(opener_path)
    coach_flip_ids, schedule_snapshot = build_coach_flip_ids(opener, REPO / "data")
    arrest_indexed = build_arrest_flags(opener, identity_features, incidents_path)
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
    measured = float(frame["correct_chain"].mean())
    if abs(measured - PUBLISHED_SEQUENTIAL_CHAIN) > 1e-9:
        raise ValueError(
            f"incumbent chain reproduction failed: measured {measured!r} "
            f"vs published {PUBLISHED_SEQUENTIAL_CHAIN!r}"
        )
    info = {
        "coach_flips": len(coach_flip_ids),
        "arrest_flips_after_coach": int(arrest_flip_after_coach.sum()),
        "chain_accuracy_full_archive_measured": measured,
        "published_sequential_chain_reference": PUBLISHED_SEQUENTIAL_CHAIN,
        "schedule_snapshot": schedule_snapshot,
    }
    return frame, info


def build_out_counts(injuries_root: Path, snaps_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Per team-game Out counts using the screen's exact machinery."""

    qa, counts = load_report_flags(injuries_root)
    out_rows = qa.loc[qa["status_norm"].eq("out")].copy()
    counts["out_rows"] = len(out_rows)
    starter_exact, starter_fuzzy = build_starter_keys(snaps_path)
    is_starter: list[bool] = []
    for season, week, team, name in zip(
        out_rows["season"],
        out_rows["week"],
        out_rows["team"],
        out_rows["norm_name"],
        strict=True,
    ):
        key3 = (int(season), int(week), str(team))
        init_last = initial_last_key(str(name))
        is_starter.append(
            (*key3, str(name)) in starter_exact
            or (init_last != ("", "") and (*key3, *init_last) in starter_fuzzy)
        )
    out_rows["is_starter_caliber"] = is_starter
    grouped = (
        out_rows.groupby(["season", "week", "team"], as_index=False)
        .agg(total_out=("is_starter_caliber", "size"), starter_out=("is_starter_caliber", "sum"))
        .astype({"season": int, "week": int})
    )
    counts["team_games_with_any_out"] = int((grouped["total_out"] > 0).sum())
    return grouped, counts


def attach_counts(
    frame: pd.DataFrame, out_counts: pd.DataFrame, identity_features: Path
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Join home/away team-game Out counts onto the archived games."""

    identity = pd.read_parquet(identity_features, columns=["game_id", "home_team", "away_team"])
    work = frame.reset_index(drop=True).merge(
        identity, on="game_id", how="left", validate="one_to_one"
    )
    if work[["home_team", "away_team"]].isna().any().any():
        raise ValueError("identity join left archived games without team labels")
    home = out_counts.rename(
        columns={
            "team": "home_team",
            "total_out": "_home_total_out",
            "starter_out": "_home_starter_out",
        }
    )
    away = out_counts.rename(
        columns={
            "team": "away_team",
            "total_out": "_away_total_out",
            "starter_out": "_away_starter_out",
        }
    )
    work = work.merge(home, on=["season", "week", "home_team"], how="left")
    work = work.merge(away, on=["season", "week", "away_team"], how="left")
    count_cols = ["_home_total_out", "_home_starter_out", "_away_total_out", "_away_starter_out"]
    archive_weeks = set(zip(work["season"], work["week"], strict=True))
    snapshot_weeks = set(zip(out_counts["season"], out_counts["week"], strict=True))
    zero_out_games = int(work[count_cols].eq(0).all(axis=1).sum())
    coverage = {
        "games": len(work),
        "archive_season_weeks": len(archive_weeks),
        "snapshot_season_weeks_with_any_out_row": len(snapshot_weeks),
        "archive_season_weeks_absent_from_snapshot": sorted(
            f"{s}w{w}" for s, w in archive_weeks - snapshot_weeks
        ),
        "games_where_neither_team_has_any_out_designation": zero_out_games,
    }
    work[count_cols] = work[count_cols].fillna(0)
    work = work.set_index("game_id", drop=False)
    return work, coverage


def apply_overlay(
    chain_pick: pd.Series, picked_flag: pd.Series, opp_flag: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Flip to the opponent iff picked flagged AND opponent not; ties keep."""

    flip = picked_flag & ~opp_flag
    return chain_pick.where(~flip, ~chain_pick).astype(bool), flip


def build_overlay_arms(work: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Frozen Saturday-cutoff overlays on the chain pick."""

    picked_is_home = work["chain_pick"].astype(bool)
    picked_starter_out = pd.Series(
        np.where(picked_is_home, work["_home_starter_out"], work["_away_starter_out"]),
        index=work.index,
    )
    opp_starter_out = pd.Series(
        np.where(picked_is_home, work["_away_starter_out"], work["_home_starter_out"]),
        index=work.index,
    )
    picked_total_out = pd.Series(
        np.where(picked_is_home, work["_home_total_out"], work["_away_total_out"]),
        index=work.index,
    )
    opp_total_out = pd.Series(
        np.where(picked_is_home, work["_away_total_out"], work["_home_total_out"]),
        index=work.index,
    )

    flag_b_picked = picked_starter_out.ge(2)
    flag_b_opp = opp_starter_out.ge(2)
    flag_c1_picked = picked_starter_out.ge(1)
    flag_c1_opp = opp_starter_out.ge(1)
    net_diff_ge1 = picked_total_out.sub(opp_total_out).ge(1)

    pick_b, flip_b = apply_overlay(work["chain_pick"], flag_b_picked, flag_b_opp)
    pick_c1, flip_c1 = apply_overlay(work["chain_pick"], flag_c1_picked, flag_c1_opp)
    pick_c2, flip_c2 = apply_overlay(work["chain_pick"], net_diff_ge1, ~net_diff_ge1.astype(bool))

    work["pick_b"] = pick_b
    work["pick_c1"] = pick_c1
    work["pick_c2"] = pick_c2
    diagnostics = {
        "b_picked_flagged": int(flag_b_picked.sum()),
        "b_both_flagged_kept": int((flag_b_picked & flag_b_opp).sum()),
        "b_flips": int(flip_b.sum()),
        "c1_picked_flagged": int(flag_c1_picked.sum()),
        "c1_both_flagged_kept": int((flag_c1_picked & flag_c1_opp).sum()),
        "c1_flips": int(flip_c1.sum()),
        "c2_net_diff_ge1_games": int(net_diff_ge1.sum()),
        "c2_flips": int(flip_c2.sum()),
        "week1_games_starter_proxy_unavailable": int(
            ((work["week"] == 1) & (picked_starter_out == 0)).sum()
        ),
    }
    return work, diagnostics


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


def record_line(arm: dict[str, Any], name: str, description: str) -> tuple[str, str, str]:
    evidence = (
        f"Week-blocked paired delta vs incumbent chain = "
        f"{arm['paired_delta_vs_chain_points']:+.4f} accuracy points "
        f"[{arm['paired_delta_week_ci_points'][0]:+.4f}, "
        f"{arm['paired_delta_week_ci_points'][1]:+.4f}], "
        f"P+ {arm['paired_delta_week_probability_positive']:.4f}; season-blocked "
        f"[{arm['paired_delta_season_ci_points'][0]:+.4f}, "
        f"{arm['paired_delta_season_ci_points'][1]:+.4f}] "
        f"P+ {arm['paired_delta_season_probability_positive']:.4f}. "
        "Saturday-cutoff attribution on already-looked-at data, 3 seasons only, "
        "no window spent, no terminal ground met."
    )
    notes = (
        "Composition of redteam_nflcom_out2_starters_only with the production chain "
        "(raw -> coach fade -> arrests); never pool as independent. Rule and tie "
        "handling predeclared in docs/nflcom_friday_refresh.md before scoring; seed "
        "20260823, 20000 samples, week primary / season secondary."
    )
    line = (
        f"nfl-ats weak-signals record --name {name} --league nfl "
        "--effect-units accuracy_points "
        f"--effect {arm['paired_delta_vs_chain_points']:.6f} "
        f"--interval-low {arm['paired_delta_week_ci_points'][0]:.6f} "
        f"--interval-high {arm['paired_delta_week_ci_points'][1]:.6f} "
        f"--probability-positive {arm['paired_delta_week_probability_positive']:.6f} "
        f"--sample-games {arm['n_scored']} "
        "--season-start 2022 --season-end 2024 "
        "--classification unresolved_below_power "
        "--source artifacts/nflcom_friday_refresh/<run_id>/metadata.json; "
        "docs/nflcom_friday_refresh "
        f'--description "{description}" '
        f'--classification-evidence "{evidence}" '
        f'--notes "{notes}"'
    )
    return line, evidence, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opener", type=Path, default=DEFAULT_OPENER)
    parser.add_argument("--identity-features", type=Path, default=DEFAULT_IDENTITY_FEATURES)
    parser.add_argument("--incidents", type=Path, default=DEFAULT_INCIDENTS)
    parser.add_argument("--injuries-root", type=Path, default=DEFAULT_INJURIES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    args = parser.parse_args(argv)
    started = time.time()

    print(f"Loading opener archive and reconstructing incumbent chain: {args.opener}")
    full_frame, chain_info = build_incumbent_chain(
        args.opener, args.identity_features, args.incidents
    )
    print(f"  chain reproduction OK: {chain_info}")

    restricted = full_frame.loc[full_frame["season"].between(SEASON_START, SEASON_END)].copy()
    print(
        f"Restricted to NFL.com-covered seasons {SEASON_START}-{SEASON_END}: "
        f"{len(restricted)} games (effective n before push exclusion)"
    )

    snaps_path = latest(REPO / "data" / "players" / "raw", "*/snap_counts.parquet")
    out_counts, out_counts_info = build_out_counts(args.injuries_root, snaps_path)
    work, coverage = attach_counts(restricted, out_counts, args.identity_features)
    work, diagnostics = build_overlay_arms(work)
    print(f"  out-count coverage: {coverage}")
    print(f"  overlay diagnostics: {diagnostics}")

    work["correct_b"] = pick_correct(work["pick_b"], work["margin_vs_open"])
    work["correct_c1"] = pick_correct(work["pick_c1"], work["margin_vs_open"])
    work["correct_c2"] = pick_correct(work["pick_c2"], work["margin_vs_open"])

    arms = [
        {
            "arm": "a_incumbent_chain",
            "n_scored": int(work["correct_chain"].notna().sum()),
            "accuracy": float(work["correct_chain"].mean()),
            "picks_changed_vs_chain": 0,
        },
        score_arm(work, name="b_out2_starters", candidate_pick_col="pick_b"),
        score_arm(work, name="c1_out1_starter", candidate_pick_col="pick_c1"),
        score_arm(work, name="c2_net_out_diff_ge1", candidate_pick_col="pick_c2"),
    ]
    arms[1]["picks_changed_vs_chain"] = diagnostics["b_flips"]
    arms[2]["picks_changed_vs_chain"] = diagnostics["c1_flips"]
    arms[3]["picks_changed_vs_chain"] = diagnostics["c2_flips"]

    season_rows = []
    for season, group in work.groupby("season"):
        season_rows.append(
            {
                "season": int(season),
                "games": len(group),
                "scored_games": int(group["correct_chain"].notna().sum()),
                "a_chain_accuracy": float(group["correct_chain"].mean()),
                "b_out2_starters_accuracy": float(group["correct_b"].mean()),
                "c1_out1_starter_accuracy": float(group["correct_c1"].mean()),
                "c2_net_out_diff_accuracy": float(group["correct_c2"].mean()),
                "b_flips": int((group["pick_b"] != group["chain_pick"]).sum()),
            }
        )

    descriptions = {
        "nflcom_refresh_out2_starters_on_chain": (
            "Late-week refresh overlay AS PLAYED: when the production-chain pick backs a "
            "team carrying >=2 Out designations on starter-caliber players (>=50% "
            "prior-week snap share proxy) per the Friday-final NFL.com league injury page, "
            "and the opponent is unflagged, flip to the opponent; both flagged keeps. On "
            "top of raw model -> coach fade -> arrests, graded at the frozen Tuesday line."
        ),
        "nflcom_refresh_out1_starter_on_chain": (
            "Milder refresh overlay variant: same rule with >=1 starter-caliber Out "
            "designation instead of >=2; both flagged keeps. Composition with the "
            "production chain, graded at the frozen Tuesday line."
        ),
        "nflcom_refresh_net_out_diff_ge1_on_chain": (
            "Net-differential refresh overlay variant: flip when the picked team's total "
            "Out designations exceed the opponent's by >=1 (any players); composition with "
            "the production chain, graded at the frozen Tuesday line."
        ),
    }
    record_names = [
        "nflcom_refresh_out2_starters_on_chain",
        "nflcom_refresh_out1_starter_on_chain",
        "nflcom_refresh_net_out_diff_ge1_on_chain",
    ]
    records = {}
    for arm_record, record_name in zip(arms[1:], record_names, strict=True):
        line, evidence, notes = record_line(arm_record, record_name, descriptions[record_name])
        records[record_name] = {
            "command_line": line,
            "effect_accuracy_points": arm_record["paired_delta_vs_chain_points"],
            "interval_low_week": arm_record["paired_delta_week_ci_points"][0],
            "interval_high_week": arm_record["paired_delta_week_ci_points"][1],
            "probability_positive_week": arm_record["paired_delta_week_probability_positive"],
            "probability_positive_season": arm_record["paired_delta_season_probability_positive"],
            "sample_games": arm_record["n_scored"],
            "classification_evidence": evidence,
            "notes": notes,
        }

    configuration = {
        "command": "nflcom-friday-refresh",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "season_window": [SEASON_START, SEASON_END],
        "opener_archive": str(args.opener),
        "identity_features": str(args.identity_features),
        "incidents": str(args.incidents),
        "injuries_root": str(args.injuries_root),
        "snap_counts": str(snaps_path),
    }
    metadata: dict[str, Any] = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **configuration,
        "input_hashes": {
            "opener_archive": sha256_file(args.opener),
            "identity_features": sha256_file(args.identity_features),
            "incidents": sha256_file(args.incidents),
            "snap_counts": sha256_file(snaps_path),
        },
        "chain_reproduction": chain_info,
        "out_count_source": out_counts_info,
        "coverage": coverage,
        "overlay_diagnostics": diagnostics,
        "arms": arms,
        "season_summary": season_rows,
        "disclosures": {
            "effective_n_three_seasons": (
                "The opener archive spans 2020-2025 but the immutable NFL.com snapshot "
                "covers ONLY seasons 2022-2024; every arm here is scored on that "
                "restricted subset (~799 archived games before push exclusion). Three "
                "seasons is small; intervals must be read accordingly."
            ),
            "attribution_upper_bound": (
                "Attribution on already-looked-at data: the parent signal was selected "
                "and red-teamed on this same 2022-2024 population, and the chain "
                "components were each promoted using overlapping windows. The composed "
                "numbers are an upper bound, continuous evidence, never a fresh "
                "confirmation. No rotation-registry window was spent; Saturday-cutoff "
                "attribution only."
            ),
            "week1_starter_proxy": (
                "Week 1 games have no prior-week snaps, so the starter proxy cannot flag "
                "them; picked_starter_out is forced 0 there (counted in "
                "week1_games_starter_proxy_unavailable), identical to the screen's "
                "missing-required-data handling. The c2 differential arm uses total Out "
                "counts and remains fully defined in Week 1."
            ),
            "pushes_preserved": (
                "Graded with nfl_ats.clv.pick_correct against margin_vs_open; pushes are "
                "NaN and excluded from every arm identically."
            ),
        },
        "proposed_weak_signal_records": records,
        "provenance": artifact_provenance(configuration, args.opener, project_root=REPO),
    }

    output_dir = args.output_root / run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(
        work.reset_index(drop=True)[
            [
                "game_id",
                "season",
                "week",
                "margin_vs_open",
                "chain_pick",
                "pick_b",
                "pick_c1",
                "pick_c2",
                "correct_chain",
                "correct_b",
                "correct_c1",
                "correct_c2",
                "_home_total_out",
                "_home_starter_out",
                "_away_total_out",
                "_away_starter_out",
            ]
        ],
        output_dir / "per_game.parquet",
    )
    atomic_csv(pd.DataFrame(season_rows), output_dir / "season_summary.csv")
    atomic_csv(pd.DataFrame(arms), output_dir / "arms_summary.csv")

    write_experiment_artifact(
        output_dir,
        "metadata.json",
        metadata,
        command="nflcom-friday-refresh",
        metrics={
            "chain_reproduction": chain_info,
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
                for arm in arms
            },
        },
        notes=(
            "Refresh-path composition study (owner task 2026-08-22); rules predeclared in "
            "docs/nflcom_friday_refresh.md before scoring; attribution-only, no window "
            "spent; NOT written to registry JSON by the script."
        ),
        source="scripts/nflcom_friday_refresh_feature.py",
        weak_signal_name=None,
        registry_root=args.registry_root,
    )

    print("\n=== Arm table (full slate, graded at the frozen Tuesday line, 2022-2024) ===")
    header = (
        f"{'arm':26s} {'n':>5s} {'acc':>8s} {'chg':>4s} {'d_pts':>8s} {'wk_P+':>7s} {'se_P+':>7s}"
    )
    print(header)
    for arm in arms:
        print(
            f"{arm['arm']:26s} {arm['n_scored']:5d} {arm['accuracy'] * 100:7.4f}% "
            f"{arm.get('picks_changed_vs_chain', 0):4d} "
            f"{arm.get('paired_delta_vs_chain_points', 0.0):+7.4f} "
            f"{arm.get('paired_delta_week_probability_positive', float('nan')):7.4f} "
            f"{arm.get('paired_delta_season_probability_positive', float('nan')):7.4f}"
        )
    print("\n=== Proposed weak-signals record lines (central recording only) ===")
    for record in records.values():
        print(f"\n{record['command_line']}")
    print(f"\nelapsed: {time.time() - started:.1f}s")
    print(f"artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
