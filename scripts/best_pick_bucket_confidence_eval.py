"""MEASURE-ONLY: does bucket-reliability weighting choose a better weekly Best Pick than served v2?

Predeclared in ``docs/best_pick_bucket_confidence.md`` before any number below
was computed. Reads frozen archives only; fits nothing; writes no registry file.

BINDING (owner mandates, restated because this output feeds a registry write):
an interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (a) refuted mechanism -- a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (b) bounded by a positive
control proven able to detect an effect that size. Everything else is
``unresolved_below_power``: record it with ``nfl-ats weak-signals record``,
report ``probability_positive``, never "contains zero". Within-week correlation
is ZERO by owner mandate: week-blocked bootstrap only. Decide on expected value
-- forced picks, so the nominator with the higher expected Best-Pick accuracy is
served; thresholds like 0.90 govern only what docs may claim. This lane changes
a RANKING, never a side.

Run::

    .\\.tools\\uv.exe run --no-sync python scripts/best_pick_bucket_confidence_eval.py \
        --repo F:/Repos/nfl_py3 --out-dir <artifacts dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats.best_pick_nomination import select_nominee  # noqa: E402
from nfl_ats.clv import week_blocked_bootstrap  # noqa: E402
from nfl_ats.provenance import write_stamped_artifact  # noqa: E402

FAMILY = "best_pick_bucket_confidence"
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
PSEUDO_OBSERVATIONS = 20.0
PSEUDO_MEAN = 0.5
SMALL_SPREAD_MAX = 6.5

OPENER_ARCHIVE = "artifacts/ridge_alpha_promotion/20260818T221459Z"
MICROSTRUCTURE_ARCHIVE = "artifacts/odds_microstructure/20260818T225430Z"
SERVED_ARCHIVE = "artifacts/opener_evaluation/20260909T183120Z"
DEADLINE_ARCHIVE = "artifacts/best_pick_deadline_renomination/20260909T214500Z"

BASE_ARM = "b0_v2_served"
ARMS = (
    BASE_ARM,
    "b1_bucket_reliability",
    "b2_small_spread_only",
    "b3_pick_side_reliability",
    "control_perfect_foresight",
)
GRADES = {
    "served": ("correct_served", "pick_home_served"),
    "archive": ("correct_archive", "pick_home_archive"),
}
SELECT_COLUMNS = ["game_id", "candidate_dist", "spread_std"]


def _load_script(path: Path, name: str) -> ModuleType:
    """Import a sibling eval script by file path, as the deadline harness does."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def spread_bucket(spread: float) -> str:
    """The four declared line-size buckets from docs/spread_hole_diagnosis.md."""
    size = abs(float(spread))
    if size <= 6.5:
        return "0-6.5"
    if size == 7.0:
        return "7"
    if size <= 10.0:
        return "7.5-10"
    return "10.5+"


def pick_side(pick_home: bool, spread: float) -> str:
    """Whether the graded pick sits on the favourite, the underdog, or a pick'em."""
    line = float(spread)
    if line == 0.0:
        return "pickem"
    return "favourite" if bool(pick_home) == (line > 0.0) else "underdog"


def shrunk_reliability(wins: float, count: float) -> float:
    """Expanding cell accuracy with 20 pseudo-observations pulled to 0.5."""
    return (wins + PSEUDO_OBSERVATIONS * PSEUDO_MEAN) / (count + PSEUDO_OBSERVATIONS)


def build_frame(repo: Path, ranker: ModuleType, composed: ModuleType) -> tuple[pd.DataFrame, Any]:
    """The 2020-2025 opener archive with both grades, the production pool, and buckets."""
    work = ranker.load_working_frame(repo / OPENER_ARCHIVE)
    work, pool_summary = ranker.build_dispersion_pool(work, repo / MICROSTRUCTURE_ARCHIVE)
    work = composed.attach_production_pool(work)
    mismatch = work.loc[work["pool_pass"] != work["dispersion_pool_pass"]]
    if not mismatch.empty:
        raise SystemExit(f"production pool disagrees with the eval pool on {len(mismatch)} games")

    served = pd.read_parquet(repo / SERVED_ARCHIVE / "per_game.parquet")[
        ["game_id", "pick_home_at_open", "correct_at_open"]
    ].rename(columns={"pick_home_at_open": "pick_home_served", "correct_at_open": "correct_served"})
    before = len(work)
    work = work.merge(served, on="game_id", how="inner", validate="one_to_one")
    if len(work) != before:
        raise SystemExit(f"served archive join dropped rows: {before} -> {len(work)}")

    work["game_id"] = work["game_id"].astype(str)
    work["correct_archive"] = work["baseline_correct_open"]
    work["pick_home_archive"] = work["baseline_pick_home"]
    work["bucket"] = [spread_bucket(value) for value in work["tue_open_home_spread"]]
    work["abs_spread"] = work["tue_open_home_spread"].abs()
    return work.reset_index(drop=True), pool_summary


def _nominee_by_score(pool: pd.DataFrame, score_column: str) -> str:
    """Production's own selector applied to a re-scored copy of the eligible pool."""
    scored = pool[["game_id", "spread_std"]].copy()
    scored["candidate_dist"] = pool[score_column].to_numpy()
    return select_nominee(scored[SELECT_COLUMNS])[0]


def evaluate_grade(work: pd.DataFrame, correct_col: str, pick_col: str) -> pd.DataFrame:
    """One row per week: every arm's nominee, its graded correctness, and diagnostics."""
    bucket_wins: dict[str, float] = {}
    bucket_count: dict[str, float] = {}
    side_wins: dict[tuple[str, str], float] = {}
    side_count: dict[tuple[str, str], float] = {}
    rows: list[dict[str, Any]] = []

    for (season, week), group in work.groupby(["season", "week"], sort=True):
        table = group.copy()
        table["side_key"] = [
            pick_side(home, line)
            for home, line in zip(table[pick_col], table["tue_open_home_spread"], strict=True)
        ]
        table["bucket_reliability"] = [
            shrunk_reliability(bucket_wins.get(name, 0.0), bucket_count.get(name, 0.0))
            for name in table["bucket"]
        ]
        table["side_reliability"] = [
            shrunk_reliability(side_wins.get(key, 0.0), side_count.get(key, 0.0))
            for key in zip(table["bucket"], table["side_key"], strict=True)
        ]
        table["score_bucket"] = (table["bucket_reliability"] - 0.5) + table["candidate_dist"]
        table["score_side"] = (table["side_reliability"] - 0.5) + table["candidate_dist"]

        pool = table.loc[table["pool_pass"]].copy()
        if pool.empty:
            raise SystemExit(f"({season}, {week}) has no eligible games; fallback rule broken")

        b0 = select_nominee(pool[SELECT_COLUMNS])[0]
        b1 = _nominee_by_score(pool, "score_bucket")
        b3 = _nominee_by_score(pool, "score_side")
        small = pool.loc[pool["abs_spread"].le(SMALL_SPREAD_MAX)]
        b2 = select_nominee(small[SELECT_COLUMNS])[0] if not small.empty else b0
        winners = pool.loc[pool[correct_col].eq(1.0)]
        control = select_nominee(winners[SELECT_COLUMNS])[0] if not winners.empty else b0

        indexed = table.set_index("game_id")
        row: dict[str, Any] = {
            "season": int(str(season)),
            "week": int(str(week)),
            "n_games": len(table),
            "n_pool": len(pool),
            "n_pool_small_spread": len(small),
            "pool_fallback": bool(table["pool_fallback"].iloc[0]),
            "prior_bucket_games": float(sum(bucket_count.values())),
            "b0_bucket_reliability": float(indexed.loc[b0, "bucket_reliability"]),
            "b0_side_reliability": float(indexed.loc[b0, "side_reliability"]),
        }
        picks = {
            BASE_ARM: b0,
            "b1_bucket_reliability": b1,
            "b2_small_spread_only": b2,
            "b3_pick_side_reliability": b3,
            "control_perfect_foresight": control,
        }
        for arm, game_id in picks.items():
            row[f"{arm}_game_id"] = game_id
            row[f"{arm}_correct"] = float(indexed.loc[game_id, correct_col])
            row[f"{arm}_bucket"] = str(indexed.loc[game_id, "bucket"])
            row[f"{arm}_side"] = str(indexed.loc[game_id, "side_key"])
            row[f"{arm}_abs_spread"] = float(indexed.loc[game_id, "abs_spread"])
        rows.append(row)

        settled = table.dropna(subset=[correct_col])
        for name, value, side in zip(
            settled["bucket"], settled[correct_col], settled["side_key"], strict=True
        ):
            bucket_wins[name] = bucket_wins.get(name, 0.0) + float(value)
            bucket_count[name] = bucket_count.get(name, 0.0) + 1.0
            key = (name, side)
            side_wins[key] = side_wins.get(key, 0.0) + float(value)
            side_count[key] = side_count.get(key, 0.0) + 1.0

    return pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)


def paired_cell(weekly: pd.DataFrame, arm: str, base: str) -> dict[str, Any]:
    """Week-blocked paired delta in accuracy points, arm minus base."""
    frame = weekly.dropna(subset=[f"{arm}_correct", f"{base}_correct"]).copy()
    frame["delta"] = (frame[f"{arm}_correct"] - frame[f"{base}_correct"]) * 100.0
    result = week_blocked_bootstrap(
        frame,
        lambda rows: {"mean": float(rows["delta"].mean())},
        block="week",
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    ).iloc[0]
    differs = frame.loc[frame[f"{arm}_game_id"] != frame[f"{base}_game_id"]]
    return {
        "weeks_paired": len(frame),
        "arm_hits": int(frame[f"{arm}_correct"].sum()),
        "base_hits": int(frame[f"{base}_correct"].sum()),
        "arm_accuracy": float(frame[f"{arm}_correct"].mean()),
        "base_accuracy": float(frame[f"{base}_correct"].mean()),
        "effect_accuracy_points": float(result["estimate"]),
        "interval_low": float(result["lower"]),
        "interval_high": float(result["upper"]),
        "probability_positive": float(result["probability_positive"]),
        "weeks_nominee_differs": len(differs),
        "weeks_outcome_differs": int((differs["delta"] != 0).sum()),
    }


def bucket_mix(weekly: pd.DataFrame, arm: str) -> dict[str, int]:
    """How many of an arm's nominations land in each declared line bucket."""
    counts = weekly[f"{arm}_bucket"].value_counts().to_dict()
    return {str(name): int(value) for name, value in sorted(counts.items())}


def grade_summary(work: pd.DataFrame, weekly: pd.DataFrame, correct_col: str) -> dict[str, Any]:
    """Hit rates, paired cells against B0, and the nomination mix for one grade."""
    hit_rates = {
        arm: {
            "weeks_scored": int(weekly[f"{arm}_correct"].notna().sum()),
            "hits": int(weekly[f"{arm}_correct"].sum()),
            "accuracy": float(weekly[f"{arm}_correct"].mean()),
            "bucket_mix": bucket_mix(weekly, arm),
            "mean_abs_spread": float(weekly[f"{arm}_abs_spread"].mean()),
        }
        for arm in ARMS
    }
    settled = work.dropna(subset=[correct_col])
    realised = {
        str(name): {
            "n": len(group),
            "accuracy": float(group[correct_col].mean()),
        }
        for name, group in settled.groupby("bucket", sort=True)
    }
    return {
        "hit_rates": hit_rates,
        "population_bucket_accuracy": realised,
        "population_accuracy": float(settled[correct_col].mean()),
        "cells": {
            f"{arm}_vs_b0": paired_cell(weekly, arm, BASE_ARM) for arm in ARMS if arm != BASE_ARM
        },
    }


def replay_check(repo: Path, weekly: pd.DataFrame) -> dict[str, Any]:
    """B0's nominees against the deadline lane's A0 nominees, week for week."""
    path = repo / DEADLINE_ARCHIVE / "weekly.csv"
    if not path.exists():
        return {"available": False}
    deadline = pd.read_csv(path)
    merged = deadline[
        [
            "season",
            "week",
            "a0_tuesday_game_id",
            "a0_tuesday_correct",
            "a0_tuesday_correct_secondary",
        ]
    ].merge(
        weekly[["season", "week", f"{BASE_ARM}_game_id", f"{BASE_ARM}_correct"]],
        on=["season", "week"],
        how="inner",
        validate="one_to_one",
    )
    same = merged["a0_tuesday_game_id"].astype(str).eq(merged[f"{BASE_ARM}_game_id"].astype(str))
    paired = merged.dropna(subset=["a0_tuesday_correct"])
    return {
        "available": True,
        "weeks_compared": len(merged),
        "weeks_nominee_identical": int(same.sum()),
        "nominee_reproduces_exactly": bool(same.all()),
        "deadline_a0_primary_hits": int(paired["a0_tuesday_correct"].sum()),
        "deadline_a0_primary_weeks": len(paired),
        "deadline_a0_secondary_hits": int(paired["a0_tuesday_correct_secondary"].sum()),
    }


def run(repo: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Score every arm on both grades and assemble the summary."""
    ranker = _load_script(REPO / "scripts/best_pick_opener_ranker_eval.py", "_bp_ranker")
    composed = _load_script(REPO / "scripts/best_pick_composed_rule_eval.py", "_bp_composed")
    work, pool_summary = build_frame(repo, ranker, composed)

    weeklies: dict[str, pd.DataFrame] = {}
    grades: dict[str, Any] = {}
    for grade, (correct_col, pick_col) in GRADES.items():
        weekly = evaluate_grade(work, correct_col, pick_col)
        weeklies[grade] = weekly
        grades[grade] = grade_summary(work, weekly, correct_col)

    summary: dict[str, Any] = {
        "family": FAMILY,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "pseudo_observations": PSEUDO_OBSERVATIONS,
        "pseudo_mean": PSEUDO_MEAN,
        "small_spread_max": SMALL_SPREAD_MAX,
        "primary_grade": "served",
        "population": {
            "games": len(work),
            "weeks": int(work.groupby(["season", "week"]).ngroups),
            "seasons": sorted(int(season) for season in work["season"].unique()),
            "pushes_served": int(work["correct_served"].isna().sum()),
            "pushes_archive": int(work["correct_archive"].isna().sum()),
            "side_disagreements_served_vs_archive": int(
                (
                    work["pick_home_served"].astype(bool) != work["pick_home_archive"].astype(bool)
                ).sum()
            ),
            "opener_archive": OPENER_ARCHIVE,
            "microstructure_archive": MICROSTRUCTURE_ARCHIVE,
            "served_archive": SERVED_ARCHIVE,
        },
        "dispersion_pool": {
            "weeks_total": pool_summary["weeks_total"],
            "weeks_fallback_total": pool_summary["weeks_fallback_total"],
            "weeks_fallback_missing_data": pool_summary["weeks_fallback_missing_data"],
            "weeks_fallback_empty_filter": pool_summary["weeks_fallback_empty_filter"],
            "production_pool_matches_eval_pool_on_every_game": True,
        },
        "b0_replay_check": replay_check(repo, weeklies["archive"]),
        "grades": grades,
        "multiplicity_note": (
            "Fifth reuse of the ~107-week opener population for the Best Pick family, and the "
            "bucket accuracies motivating B1/B3 were measured on this same archive; compounding "
            "look-reuse discount. No rotation window assigned or spent."
        ),
        "binding_note": (
            "An interval containing zero is never a rejection; probability_positive is the "
            "continuous evidence. Closing grounds are limited to refuted_mechanism (resolved "
            "wrong sign or zero split-half reliability) and bounded_by_control."
        ),
    }
    return summary, weeklies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    summary, weeklies = run(args.repo)
    print(json.dumps(summary, indent=2, default=str))
    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = args.repo / "artifacts" / FAMILY / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    write_stamped_artifact(json.loads(json.dumps(summary, default=str)), out_dir / "summary.json")
    for grade, weekly in weeklies.items():
        weekly.to_parquet(out_dir / f"weekly_{grade}.parquet")
        weekly.to_csv(out_dir / f"weekly_{grade}.csv", index=False)
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
