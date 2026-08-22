"""Ceiling attack: decompose ATS-residual MSE against the Tuesday opener into
market error capturable by a better model versus the irreducible execution
noise floor.

Population: the frozen opener-evaluation archive
``artifacts/opener_evaluation/20260819T174244Z/per_game.parquet``
(1,537 paired 2020-2025 REG games, ``weak_stack`` composed card).

Predictors of the opener ATS residual ``margin_vs_open``:

- market line: predicts 0, so its squared error IS the total MSE;
- our composed card: ``residual_at_open``, which embeds market information by
  construction (trained toward the market-residual target);
- oracle blend: in-sample OLS of the outcome on [1, our pred]. Because the
  market predictor is constant this spans every linear blend of market and
  card, so it is an UPPER bound on what shrinkage could achieve;
- perfect movement foresight: OLS on [1, open_move], the late-information
  channel measured separately (upper bound for that channel);
- joint oracle: OLS on [1, our pred, open_move].

Theoretical floor: execution-noise sd from the SIM-02 lite variance
decomposition (latest ``registry/experiments/vardec-noisefloor`` run,
12.70 points). Unmatched matchup variance = Var(opener residual) minus
execution variance is the THEORETICAL ceiling any pregame team-strength
model could remove.

Accuracy translation uses the flat exchange rate of 3 accuracy points per
point of RMS improvement at sigma ~ 12.8. Forced picks grade sign, not
magnitude, so MSE deltas understate sign-channel gains; direct accuracy
deltas are always reported beside flat-exchange numbers and never mixed.

Writes JSON to ``artifacts/ceiling_error_split/<UTC timestamp>/results.json``,
a summary plus the banked-to-wall gap table to stdout, and scoped gate
results. Measure-only: no selection strategy, nothing recorded to
weak_signals.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

DEFAULT_ARCHIVE = REPO / "artifacts" / "opener_evaluation" / "20260819T174244Z"
NOISE_FLOOR_REGISTRY = REPO / "registry" / "experiments" / "vardec-noisefloor"
EXCHANGE_RATE_ACC_PER_PT = 3.0
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260822
EXEC_SD_GATE_LOW = 12.0
EXEC_SD_GATE_HIGH = 13.5

BOOTSTRAP_KEYS = [
    "capturable_share_current_card",
    "capturable_share_late_information",
    "capturable_share_joint",
    "current_card_headroom_acc_flat",
    "late_info_headroom_acc_flat",
    "team_model_headroom_acc_flat",
]


def latest_noise_floor_metrics() -> tuple[Path, dict[str, Any]]:
    runs = sorted(
        (r for r in NOISE_FLOOR_REGISTRY.glob("*.json") if not r.stem.startswith("dev_")),
        key=lambda r: r.stem,
    )
    if not runs:
        raise FileNotFoundError(f"no runs under {NOISE_FLOOR_REGISTRY}")
    path = runs[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return path, payload["metrics"]


def execution_noise_sd(metrics: dict[str, Any]) -> float:
    calibrated = float(metrics["calibrated_sd"])
    floor = float(metrics["floor_sd"])
    if calibrated <= floor:
        raise ValueError("floor_sd exceeds calibrated_sd in noisefloor registry")
    return math.sqrt(calibrated**2 - floor**2)


def ols_fit(y: np.ndarray, design: np.ndarray) -> tuple[float, np.ndarray]:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(np.mean(resid**2)), coef


def core_stats(y: np.ndarray, p: np.ndarray, m: np.ndarray, exec_sd: float) -> dict[str, float]:
    ones = np.ones_like(y)
    mse_market = float(np.mean(y**2))
    rms_market = math.sqrt(mse_market)
    mse_card_raw = float(np.mean((y - p) ** 2))
    mse_blend, blend_coef = ols_fit(y, np.column_stack([ones, p]))
    mse_move, move_coef = ols_fit(y, np.column_stack([ones, m]))
    mse_joint, _ = ols_fit(y, np.column_stack([ones, p, m]))

    corr_pred = float(np.corrcoef(y, p)[0, 1])
    corr_errors = float(np.corrcoef(y, y - p)[0, 1])

    exec_var = exec_sd**2
    unmatched_var = max(mse_market - exec_var, 0.0)
    share_exec_theory = min(exec_var / mse_market, 1.0)
    share_unmatched_theory = 1.0 - share_exec_theory
    team_model_headroom_rms = max(rms_market - exec_sd, 0.0)

    return {
        "n_games": float(len(y)),
        "outcome_mean_points": float(np.mean(y)),
        "outcome_sd_points": float(np.std(y, ddof=1)),
        "mse_market": mse_market,
        "rms_market_points": rms_market,
        "mse_card_unshrunk": mse_card_raw,
        "rms_card_unshrunk_points": math.sqrt(mse_card_raw),
        "card_pred_mean_points": float(np.mean(p)),
        "card_pred_sd_points": float(np.std(p, ddof=1)),
        "corr_outcome_card": corr_pred,
        "corr_market_err_card_err": corr_errors,
        "oracle_blend_intercept": float(blend_coef[0]),
        "oracle_blend_slope_card": float(blend_coef[1]),
        "mse_oracle_blend": mse_blend,
        "capturable_share_current_card": 1.0 - mse_blend / mse_market,
        "movement_foresight_intercept": float(move_coef[0]),
        "movement_foresight_slope": float(move_coef[1]),
        "mse_movement_foresight": mse_move,
        "capturable_share_late_information": 1.0 - mse_move / mse_market,
        "mse_joint_oracle": mse_joint,
        "capturable_share_joint": 1.0 - mse_joint / mse_market,
        "execution_noise_sd_points": exec_sd,
        "unmatched_matchup_variance": unmatched_var,
        "share_execution_noise_theory": share_exec_theory,
        "share_unmatched_matchup_theory": share_unmatched_theory,
        "team_model_headroom_acc_flat": EXCHANGE_RATE_ACC_PER_PT * team_model_headroom_rms,
        "current_card_headroom_acc_flat": EXCHANGE_RATE_ACC_PER_PT
        * (rms_market - math.sqrt(mse_blend)),
        "late_info_headroom_acc_flat": EXCHANGE_RATE_ACC_PER_PT
        * (rms_market - math.sqrt(mse_move)),
    }


def _resample_indices(blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(blocks)
    picked = rng.choice(unique, size=len(unique), replace=True)
    order = np.argsort(blocks, kind="stable")
    sorted_blocks = blocks[order]
    starts = np.searchsorted(sorted_blocks, picked, side="left")
    ends = np.searchsorted(sorted_blocks, picked, side="right")
    return np.concatenate(
        [order[s:e] for s, e in zip(starts, ends, strict=True)]  # type: ignore[call-arg]
    )


def week_blocked_bootstrap(
    y: np.ndarray,
    p: np.ndarray,
    m: np.ndarray,
    blocks: np.ndarray,
    samples: int,
    seed: int,
    exec_sd: float,
) -> dict[str, dict[str, float]]:
    draws: dict[str, list[float]] = {name: [] for name in BOOTSTRAP_KEYS}
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        idx = _resample_indices(blocks, rng)
        s = core_stats(y[idx], p[idx], m[idx], exec_sd)
        for name in BOOTSTRAP_KEYS:
            draws[name].append(s[name])
    out: dict[str, dict[str, float]] = {}
    for name, values in draws.items():
        arr = np.asarray(values)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[name] = {"point": float(arr.mean()), "ci95": [float(lo), float(hi)]}
    return out


def gap_table(
    stats: dict[str, float],
    banked_prob_rule: float,
    banked_sign_rule: float,
    move_oracle: float,
) -> list[dict[str, Any]]:
    omniscient_low, omniscient_high = 57.0, 58.0
    late_slice_direct = 100.0 * (move_oracle - banked_prob_rule)
    total_gap_low = omniscient_low - 100.0 * banked_prob_rule
    total_gap_high = omniscient_high - 100.0 * banked_prob_rule
    bounded_sum = late_slice_direct + stats["team_model_headroom_acc_flat"]
    remainder_low = total_gap_high - bounded_sum
    remainder_high = total_gap_low - bounded_sum
    return [
        {
            "row": "banked_anchor_this_archive",
            "value_acc_points": None,
            "detail": {
                "banked_opener_probability_rule_pct": 100.0 * banked_prob_rule,
                "banked_opener_sign_rule_pct": 100.0 * banked_sign_rule,
                "note": (
                    "anchor is THIS archive's recomputed accuracy; the "
                    "owner-stated 53.8% banked figure does not reproduce from "
                    "this artifact (nearest match is the 2024 season row, "
                    "docs/opener_evaluation.md)"
                ),
            },
        },
        {
            "row": "total_gap_to_omniscient_practical_wall",
            "value_acc_points": [total_gap_low, total_gap_high],
            "detail": "wall = 57-58% vs frozen opener per docs/pool_edge_plan.md ceiling section",
        },
        {
            "row": "slice_capturable_by_better_team_model",
            "value_acc_points": stats["team_model_headroom_acc_flat"],
            "detail": (
                "theoretical ceiling: market MSE - execution variance "
                f"= {stats['unmatched_matchup_variance']:.2f} pts^2, flat exchange "
                f"{EXCHANGE_RATE_ACC_PER_PT:.0f} acc pts/pt at sigma "
                f"{stats['outcome_sd_points']:.2f}; realized capture by the current "
                f"card is {stats['current_card_headroom_acc_flat']:.3f} acc pts"
            ),
        },
        {
            "row": "slice_capturable_by_late_information",
            "value_acc_points": late_slice_direct,
            "detail": (
                "measured separately: movement oracle "
                f"{100.0 * move_oracle:.2f}% minus banked anchor; flat-exchange "
                f"equivalent {stats['late_info_headroom_acc_flat']:.2f} acc pts, "
                "which understates this sign-channel slice"
            ),
        },
        {
            "row": "irreducible_remainder_execution_noise",
            "value_acc_points": None,
            "detail": (
                f"execution-noise sd {stats['execution_noise_sd_points']:.2f} pts "
                f"= {100.0 * stats['share_execution_noise_theory']:.1f}% of market "
                "MSE (share of MSE, NOT acc points); not capturable by any "
                "pregame model"
            ),
        },
        {
            "row": "remainder_beyond_both_slices_vs_wall",
            "value_acc_points": [remainder_low, remainder_high],
            "detail": (
                "omniscience beyond perfect movement foresight (private or "
                "aggregated information the line never sees) plus wall-band width"
            ),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir: Path = args.output or (REPO / "artifacts" / "ceiling_error_split" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.archive} ===")
    frame = pd.read_parquet(args.archive / "per_game.parquet")
    metadata = json.loads((args.archive / "metadata.json").read_text(encoding="utf-8"))

    floor_path, floor_metrics = latest_noise_floor_metrics()
    exec_sd = execution_noise_sd(floor_metrics)
    print(f"noise-floor source {floor_path.name}: execution sd = {exec_sd:.3f} pts")

    y_all = frame["margin_vs_open"].to_numpy(dtype=np.float64)
    p_all = frame["residual_at_open"].to_numpy(dtype=np.float64)
    m_all = frame["open_move"].to_numpy(dtype=np.float64)
    push_mask = y_all == 0.0
    keep = ~push_mask
    y, p, m = y_all[keep], p_all[keep], m_all[keep]
    blocks = (frame["season"].to_numpy() * 100 + frame["week"].to_numpy())[keep]

    banked_prob_rule = float(frame.loc[keep, "correct_at_open_probability_rule"].mean())
    banked_sign_rule = float(frame.loc[keep, "correct_at_open"].mean())
    move_oracle = float(frame.loc[keep, "oracle_correct_at_open"].mean())

    stats = core_stats(y, p, m, exec_sd)

    boot = week_blocked_bootstrap(y, p, m, blocks, args.samples, args.seed, exec_sd)
    boot_repeat = week_blocked_bootstrap(y, p, m, blocks, args.samples, args.seed, exec_sd)
    boot_deterministic = boot == boot_repeat

    gates = [
        {
            "gate": "population_matches_archive_metadata",
            "passed": len(frame) == int(metadata["games"]) == 1537,
            "detail": {"rows": len(frame), "metadata_games": metadata["games"]},
        },
        {
            "gate": "reproduces_archive_accuracies",
            "passed": bool(
                abs(banked_prob_rule - metadata["metrics"]["opener_accuracy_probability_rule"])
                < 1e-12
                and abs(move_oracle - metadata["metrics"]["movement_oracle_accuracy"]) < 1e-12
            ),
            "detail": {
                "recomputed_probability_rule": banked_prob_rule,
                "metadata_probability_rule": metadata["metrics"][
                    "opener_accuracy_probability_rule"
                ],
                "recomputed_movement_oracle": move_oracle,
                "metadata_movement_oracle": metadata["metrics"]["movement_oracle_accuracy"],
            },
        },
        {
            "gate": "execution_noise_source_in_band",
            "passed": EXEC_SD_GATE_LOW <= exec_sd <= EXEC_SD_GATE_HIGH,
            "detail": {"exec_sd_points": exec_sd, "band": [EXEC_SD_GATE_LOW, EXEC_SD_GATE_HIGH]},
        },
        {
            "gate": "nested_oracles_monotone",
            "passed": bool(
                stats["mse_joint_oracle"]
                <= min(stats["mse_oracle_blend"], stats["mse_movement_foresight"]) + 1e-9
                and stats["mse_oracle_blend"]
                <= min(stats["mse_market"], stats["mse_card_unshrunk"]) + 1e-9
            ),
            "detail": {
                "mse_market": stats["mse_market"],
                "mse_card_unshrunk": stats["mse_card_unshrunk"],
                "mse_oracle_blend": stats["mse_oracle_blend"],
                "mse_movement_foresight": stats["mse_movement_foresight"],
                "mse_joint_oracle": stats["mse_joint_oracle"],
            },
        },
        {
            "gate": "theory_shares_sum_to_one",
            "passed": bool(
                abs(
                    stats["share_execution_noise_theory"]
                    + stats["share_unmatched_matchup_theory"]
                    - 1.0
                )
                < 1e-9
            ),
            "detail": {
                "sum": stats["share_execution_noise_theory"]
                + stats["share_unmatched_matchup_theory"]
            },
        },
        {
            "gate": "bootstrap_reproducible_same_seed",
            "passed": boot_deterministic,
            "detail": {"samples": args.samples, "seed": args.seed},
        },
    ]
    all_pass = all(bool(g["passed"]) for g in gates)

    table = gap_table(stats, banked_prob_rule, banked_sign_rule, move_oracle)

    cross_checks = {
        "pool_edge_plan_midweek_channel": {
            "plan_claim": "~2.6 points (55.1 movement oracle vs 52.5 baseline)",
            "this_archive_sign_rule_anchor": 100.0 * (move_oracle - banked_sign_rule),
            "this_archive_banked_anchor": 100.0 * (move_oracle - banked_prob_rule),
        },
        "pool_edge_plan_teams_bounded_near_zero": {
            "plan_claim": "better team-strength measurement is bounded near zero",
            "this_measure_acc_points": 100.0 * stats["team_model_headroom_acc_flat"],
        },
        "pool_edge_plan_sigma_and_rate": {
            "plan_claim": "sigma ~13.1, exchange ~3 acc pts per point",
            "this_subset_sigma_points": stats["outcome_sd_points"],
        },
        "pool_edge_plan_guardrail_60pct": {
            "plan_claim": "any backtest above 60% is a leak",
            "this_measure": (
                f"{100.0 * stats['share_execution_noise_theory']:.1f}% of opener "
                "MSE is execution noise"
            ),
        },
    }

    print("\n=== MSE decomposition vs Tuesday opener (non-push) ===")
    print(f"games={int(stats['n_games'])} pushes_dropped={int(push_mask.sum())}")
    print(f"market RMS error           {stats['rms_market_points']:.3f} pts")
    print(f"card raw RMS error         {stats['rms_card_unshrunk_points']:.3f} pts (unshrunk)")
    print(
        f"oracle blend RMS           {math.sqrt(stats['mse_oracle_blend']):.3f} pts "
        f"(slope {stats['oracle_blend_slope_card']:.3f})"
    )
    print(f"corr(outcome, card)        {stats['corr_outcome_card']:.4f}")
    print(f"corr(market err, card err) {stats['corr_market_err_card_err']:.4f}")
    print(f"capturable share, current card  {100.0 * stats['capturable_share_current_card']:.3f}%")
    print(
        f"capturable share, move oracle   {100.0 * stats['capturable_share_late_information']:.2f}%"
    )
    print(f"capturable share, joint oracle   {100.0 * stats['capturable_share_joint']:.3f}%")
    print(
        f"theory: execution noise {100.0 * stats['share_execution_noise_theory']:.2f}% of MSE, "
        f"unmatched matchup {100.0 * stats['share_unmatched_matchup_theory']:.2f}%"
    )

    print("\n=== gap table: banked -> wall (accuracy points) ===")
    for row in table:
        print(f"{row['row']:45s} {row['value_acc_points']}")

    print("\n=== scoped gates ===")
    for gate in gates:
        print(f"[{'PASS' if gate['passed'] else 'FAIL'}] {gate['gate']}")
    print(f"ALL_GATES: {'PASS' if all_pass else 'FAIL'}")

    payload = {
        "schema": 1,
        "generated_at_utc": timestamp,
        "archive": str(args.archive),
        "noise_floor_source": str(floor_path),
        "population": {
            "rows": len(frame),
            "pushes_dropped": int(push_mask.sum()),
            "scored_games": int(stats["n_games"]),
            "banked_opener_probability_rule": banked_prob_rule,
            "banked_opener_sign_rule": banked_sign_rule,
            "movement_oracle_accuracy": move_oracle,
        },
        "method": {
            "target": "margin_vs_open (ATS residual vs Tuesday opener)",
            "predictors": ["market line (=0)", "residual_at_open", "open_move oracle"],
            "oracles_are_in_sample_upper_bounds": True,
            "exchange_rate_acc_per_point": EXCHANGE_RATE_ACC_PER_PT,
            "bootstrap": "week-blocked percentile 95%",
            "samples": args.samples,
            "seed": args.seed,
            "honest_treatment": (
                "the composed card embeds market information (its training target "
                "is the market residual); its capturable share is read against the "
                "market predictor directly, and sign-channel value is reported "
                "through measured accuracy deltas rather than MSE translations"
            ),
        },
        "decomposition": stats,
        "bootstrap_ci": boot,
        "gap_table": table,
        "cross_checks_pool_edge_plan": cross_checks,
        "gates": gates,
        "all_gates_pass": all_pass,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    configuration = {
        "archive": str(args.archive),
        "samples": args.samples,
        "seed": args.seed,
        "exchange_rate_acc_per_point": EXCHANGE_RATE_ACC_PER_PT,
    }
    payload["provenance"] = artifact_provenance(
        configuration, args.archive / "per_game.parquet", project_root=REPO
    )
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="ceiling-error-split",
        metrics={
            "rms_market_points": stats["rms_market_points"],
            "execution_noise_share": stats["share_execution_noise_theory"],
            "capturable_share_current_card": stats["capturable_share_current_card"],
            "capturable_share_late_information": stats["capturable_share_late_information"],
            "team_model_headroom_acc_points": stats["team_model_headroom_acc_flat"],
            "all_gates_pass": all_pass,
        },
        notes=(
            "Measure-only MSE decomposition of opener ATS residuals into market "
            "error vs execution-noise floor; oracles are in-sample upper bounds; "
            "nothing recorded to weak_signals."
        ),
        source="scripts/ceiling_error_split.py",
    )
    print(f"\nwrote {output_dir / 'results.json'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
