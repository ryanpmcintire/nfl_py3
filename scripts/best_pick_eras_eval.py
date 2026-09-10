"""MEASURE-ONLY: which Best Pick nominator wins across eras, on the frozen 2020-2025 opener archive.

Predeclared in ``docs/best_pick_eras.md`` before any number below was computed.
Reads frozen archives only; fits nothing new; writes no registry file except
through ``nfl-ats weak-signals record`` under ``--record``.

BINDING (owner mandates, restated because this output feeds a registry write):
an interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (a) refuted mechanism -- a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (b) bounded by a positive
control proven able to detect an effect that size. Everything else is
``unresolved_below_power``: record it, report ``probability_positive``, never
"contains zero". Within-week correlation is ZERO; week-blocked bootstrap. Era
readings differ in MAGNITUDE; a weaker era is never absence, but a SIGN
reversal across eras is a diagnosis to publish. Decide on expected value;
predeclared thresholds govern only claims. This lane changes a RANKING, never a
side, and wires nothing.

Run::

    .\\.tools\\uv.exe run --no-sync python scripts/best_pick_eras_eval.py \
        --repo F:/Repos/nfl_py3 --out-dir <artifacts dir>
    ... best_pick_eras_eval.py --record --artifact <summary.json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
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
from nfl_ats.displayed_confidence import (  # noqa: E402
    DISPLAY_BUCKETS,
    archive_display_stream,
    walk_forward_displayed_confidence,
)
from nfl_ats.provenance import write_stamped_artifact  # noqa: E402

FAMILY = "best_pick_eras"
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260821
PSEUDO_OBSERVATIONS = 20.0
PSEUDO_MEAN = 0.5
SMALL_SPREAD_MAX = 6.5

OPENER_ARCHIVE = "artifacts/ridge_alpha_promotion/20260818T221459Z"
MICROSTRUCTURE_ARCHIVE = "artifacts/odds_microstructure/20260818T225430Z"
OPENER_EVALUATION = "artifacts/opener_evaluation/20260909T183120Z"

N0 = "n0_small_spread"
N1 = "n1_v2_unrestricted"
ARMS = (
    N0,
    N1,
    "n2_reliable_bucket",
    "n3_calibrated_pool",
    "n4_calibrated_open",
    "control_perfect_foresight",
)
GRADES = {
    "production": ("correct_production", "pick_home_production"),
    "sign_rule": ("correct_sign", "pick_home_sign"),
}
PRIMARY_GRADE = "production"
ERAS: dict[str, tuple[int, ...]] = {
    "2020_2025": (2020, 2021, 2022, 2023, 2024, 2025),
    "2020_2021": (2020, 2021),
    "2022_2023": (2022, 2023),
    "2024_2025": (2024, 2025),
}
FULL_WINDOW = "2020_2025"
SELECT_COLUMNS = ["game_id", "candidate_dist", "spread_std"]


def _load_script(path: Path, name: str) -> ModuleType:
    """Import a sibling eval script by file path, as the other Best Pick lanes do."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def spread_bucket(spread: float) -> str:
    """The four declared line-size buckets, matching displayed_confidence.DISPLAY_BUCKETS."""
    size = abs(float(spread))
    if size <= 6.5:
        return DISPLAY_BUCKETS[0]
    if size == 7.0:
        return DISPLAY_BUCKETS[1]
    if size <= 10.0:
        return DISPLAY_BUCKETS[2]
    return DISPLAY_BUCKETS[3]


def shrunk_reliability(wins: float, count: float) -> float:
    """Expanding cell accuracy with 20 pseudo-observations pulled to 0.5."""
    return (wins + PSEUDO_OBSERVATIONS * PSEUDO_MEAN) / (count + PSEUDO_OBSERVATIONS)


def build_frame(repo: Path, ranker: ModuleType, composed: ModuleType) -> tuple[pd.DataFrame, Any]:
    """The 2020-2025 opener archive with both grades, the production pool, buckets, display."""
    work = ranker.load_working_frame(repo / OPENER_ARCHIVE)
    work, pool_summary = ranker.build_dispersion_pool(work, repo / MICROSTRUCTURE_ARCHIVE)
    work = composed.attach_production_pool(work)
    mismatch = work.loc[work["pool_pass"] != work["dispersion_pool_pass"]]
    if not mismatch.empty:
        raise SystemExit(f"production pool disagrees with the eval pool on {len(mismatch)} games")

    per_game = pd.read_parquet(repo / OPENER_EVALUATION / "per_game.parquet")
    stream = archive_display_stream(per_game)
    stream["displayed_score"] = walk_forward_displayed_confidence(stream)
    display = stream[["game_id", "displayed_score", "stated", "bucket", "band"]].rename(
        columns={"stated": "stated_probability", "bucket": "display_bucket", "band": "display_band"}
    )

    graded = per_game[
        [
            "game_id",
            "pick_home_at_open_probability_rule",
            "correct_at_open_probability_rule",
            "pick_home_at_open",
            "correct_at_open",
        ]
    ].rename(
        columns={
            "pick_home_at_open_probability_rule": "pick_home_production",
            "correct_at_open_probability_rule": "correct_production",
            "pick_home_at_open": "pick_home_sign",
            "correct_at_open": "correct_sign",
        }
    )
    graded["game_id"] = graded["game_id"].astype(str)
    display["game_id"] = display["game_id"].astype(str)
    work["game_id"] = work["game_id"].astype(str)

    before = len(work)
    work = work.merge(graded, on="game_id", how="inner", validate="one_to_one")
    work = work.merge(display, on="game_id", how="inner", validate="one_to_one")
    if len(work) != before:
        raise SystemExit(f"opener-evaluation join dropped rows: {before} -> {len(work)}")

    work["bucket"] = [spread_bucket(value) for value in work["tue_open_home_spread"]]
    work["abs_spread"] = work["tue_open_home_spread"].abs()
    return work.reset_index(drop=True), pool_summary


def _nominee_by_score(pool: pd.DataFrame, score_column: str) -> str:
    """Production's own selector applied to a re-scored copy of an eligible pool."""
    scored = pool[["game_id", "spread_std"]].copy()
    scored["candidate_dist"] = pool[score_column].to_numpy()
    return select_nominee(scored[SELECT_COLUMNS])[0]


def evaluate_grade(work: pd.DataFrame, correct_col: str) -> pd.DataFrame:
    """One row per week: every arm's nominee, its graded correctness, and diagnostics."""
    bucket_wins: dict[str, float] = {}
    bucket_count: dict[str, float] = {}
    rows: list[dict[str, Any]] = []

    for (season, week), group in work.groupby(["season", "week"], sort=True):
        table = group.copy()
        reliability = {
            name: shrunk_reliability(bucket_wins.get(name, 0.0), bucket_count.get(name, 0.0))
            for name in DISPLAY_BUCKETS
        }
        best_bucket = max(
            DISPLAY_BUCKETS, key=lambda name: (reliability[name], -DISPLAY_BUCKETS.index(name))
        )

        pool = table.loc[table["pool_pass"]].copy()
        if pool.empty:
            raise SystemExit(f"({season}, {week}) has no eligible games; fallback rule broken")

        n1 = select_nominee(pool[SELECT_COLUMNS])[0]
        small = pool.loc[pool["abs_spread"].le(SMALL_SPREAD_MAX)]
        n0 = select_nominee(small[SELECT_COLUMNS])[0] if not small.empty else n1
        reliable = pool.loc[pool["bucket"].eq(best_bucket)]
        n2 = select_nominee(reliable[SELECT_COLUMNS])[0] if not reliable.empty else n1
        n3 = _nominee_by_score(pool, "displayed_score")
        n4 = _nominee_by_score(table, "displayed_score")
        winners = pool.loc[pool[correct_col].eq(1.0)]
        control = select_nominee(winners[SELECT_COLUMNS])[0] if not winners.empty else n1

        indexed = table.set_index("game_id")
        row: dict[str, Any] = {
            "season": int(str(season)),
            "week": int(str(week)),
            "era": era_of(int(str(season))),
            "n_games": len(table),
            "n_pool": len(pool),
            "n_pool_small_spread": len(small),
            "n_pool_reliable_bucket": len(reliable),
            "pool_fallback": bool(table["pool_fallback"].iloc[0]),
            "reliable_bucket": best_bucket,
            "reliable_bucket_value": float(reliability[best_bucket]),
            "prior_games": float(sum(bucket_count.values())),
            **{f"reliability_{name}": float(reliability[name]) for name in DISPLAY_BUCKETS},
        }
        picks = {
            N0: n0,
            N1: n1,
            "n2_reliable_bucket": n2,
            "n3_calibrated_pool": n3,
            "n4_calibrated_open": n4,
            "control_perfect_foresight": control,
        }
        for arm, game_id in picks.items():
            row[f"{arm}_game_id"] = game_id
            row[f"{arm}_correct"] = float(indexed.loc[game_id, correct_col])
            row[f"{arm}_bucket"] = str(indexed.loc[game_id, "bucket"])
            row[f"{arm}_abs_spread"] = float(indexed.loc[game_id, "abs_spread"])
            row[f"{arm}_displayed_score"] = float(indexed.loc[game_id, "displayed_score"])
        rows.append(row)

        settled = table.dropna(subset=[correct_col])
        for name, value in zip(settled["bucket"], settled[correct_col], strict=True):
            bucket_wins[name] = bucket_wins.get(name, 0.0) + float(value)
            bucket_count[name] = bucket_count.get(name, 0.0) + 1.0

    return pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)


def era_of(season: int) -> str:
    """Which declared two-season era a season belongs to."""
    for name, seasons in ERAS.items():
        if name != FULL_WINDOW and season in seasons:
            return name
    raise ValueError(f"season {season} is outside the declared eras")


def paired_cell(weekly: pd.DataFrame, arm: str, base: str) -> dict[str, Any]:
    """Week-blocked paired delta in accuracy points, arm minus base."""
    frame = weekly.dropna(subset=[f"{arm}_correct", f"{base}_correct"]).copy()
    frame["delta"] = (frame[f"{arm}_correct"] - frame[f"{base}_correct"]) * 100.0
    result = week_blocked_bootstrap(
        frame[["season", "week", "delta"]],
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
        "arm_wins_weeks": int((frame["delta"] > 0).sum()),
        "base_wins_weeks": int((frame["delta"] < 0).sum()),
    }


def hit_rates(weekly: pd.DataFrame) -> dict[str, Any]:
    """Each arm's raw weekly Best-Pick record on this slice."""
    return {
        arm: {
            "weeks_scored": int(weekly[f"{arm}_correct"].notna().sum()),
            "hits": int(weekly[f"{arm}_correct"].sum()),
            "accuracy": float(weekly[f"{arm}_correct"].mean()),
            "bucket_mix": {
                str(name): int(value)
                for name, value in sorted(weekly[f"{arm}_bucket"].value_counts().to_dict().items())
            },
            "mean_abs_spread": float(weekly[f"{arm}_abs_spread"].mean()),
        }
        for arm in ARMS
    }


def slice_summary(weekly: pd.DataFrame) -> dict[str, Any]:
    """Hit rates plus every paired cell against both baselines on one slice."""
    cells: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        if arm != N0:
            cells[f"{arm}_vs_n0"] = paired_cell(weekly, arm, N0)
        if arm != N1:
            cells[f"{arm}_vs_n1"] = paired_cell(weekly, arm, N1)
    return {"weeks": len(weekly), "hit_rates": hit_rates(weekly), "cells": cells}


def sign_agreement(grade: dict[str, Any]) -> dict[str, Any]:
    """For each cell, how many of the three eras carry the full window's sign."""
    era_names = [name for name in ERAS if name != FULL_WINDOW]
    report: dict[str, Any] = {}
    for cell_name, full in grade[FULL_WINDOW]["cells"].items():
        full_effect = full["effect_accuracy_points"]
        full_sign = (full_effect > 0) - (full_effect < 0)
        era_effects = {
            era: grade[era]["cells"][cell_name]["effect_accuracy_points"] for era in era_names
        }
        signs = {era: (value > 0) - (value < 0) for era, value in era_effects.items()}
        report[cell_name] = {
            "full_window_effect": full_effect,
            "full_window_sign": full_sign,
            "era_effects": era_effects,
            "era_signs": signs,
            "eras_agreeing_with_full_window": int(
                sum(1 for value in signs.values() if value == full_sign)
            ),
            "eras_positive": int(sum(1 for value in signs.values() if value > 0)),
            "eras_negative": int(sum(1 for value in signs.values() if value < 0)),
            "eras_zero": int(sum(1 for value in signs.values() if value == 0)),
            "positive_in_all_three_eras": bool(all(value > 0 for value in signs.values())),
            "sign_reversal_across_eras": bool(
                any(value > 0 for value in signs.values())
                and any(value < 0 for value in signs.values())
            ),
        }
    return report


def nominee_bucket_table(weekly: pd.DataFrame, arm: str, correct_col_suffix: str) -> dict[str, Any]:
    """How an arm's own nominations split across the four line buckets."""
    frame = weekly.dropna(subset=[f"{arm}{correct_col_suffix}"])
    return {
        str(name): {
            "weeks": len(group),
            "hits": int(group[f"{arm}{correct_col_suffix}"].sum()),
            "accuracy": float(group[f"{arm}{correct_col_suffix}"].mean()),
        }
        for name, group in frame.groupby(f"{arm}_bucket", sort=True)
    }


def run(repo: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Score every arm on both grades, on the full window and each era."""
    ranker = _load_script(REPO / "scripts/best_pick_opener_ranker_eval.py", "_bp_ranker")
    composed = _load_script(REPO / "scripts/best_pick_composed_rule_eval.py", "_bp_composed")
    work, pool_summary = build_frame(repo, ranker, composed)

    weeklies: dict[str, pd.DataFrame] = {}
    grades: dict[str, Any] = {}
    for grade, (correct_col, _pick_col) in GRADES.items():
        weekly = evaluate_grade(work, correct_col)
        weeklies[grade] = weekly
        slices = {
            window: slice_summary(weekly.loc[weekly["season"].isin(seasons)])
            for window, seasons in ERAS.items()
        }
        slices["sign_agreement"] = sign_agreement(slices)
        slices["nominee_buckets"] = {
            arm: nominee_bucket_table(weekly, arm, "_correct") for arm in ARMS
        }
        slices["per_season"] = {
            str(int(season)): {
                arm: {
                    "weeks": int(group[f"{arm}_correct"].notna().sum()),
                    "hits": int(group[f"{arm}_correct"].sum()),
                    "accuracy": float(group[f"{arm}_correct"].mean()),
                }
                for arm in ARMS
            }
            for season, group in weekly.groupby("season", sort=True)
        }
        grades[grade] = slices

    settled = work.dropna(subset=["correct_production"])
    summary: dict[str, Any] = {
        "family": FAMILY,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "pseudo_observations": PSEUDO_OBSERVATIONS,
        "small_spread_max": SMALL_SPREAD_MAX,
        "primary_grade": PRIMARY_GRADE,
        "eras": {name: list(seasons) for name, seasons in ERAS.items()},
        "population": {
            "games": len(work),
            "weeks": int(work.groupby(["season", "week"]).ngroups),
            "seasons": sorted(int(season) for season in work["season"].unique()),
            "pushes_production": int(work["correct_production"].isna().sum()),
            "pushes_sign": int(work["correct_sign"].isna().sum()),
            "production_vs_sign_pick_disagreements": int(
                (
                    work["pick_home_production"].astype(bool) != work["pick_home_sign"].astype(bool)
                ).sum()
            ),
            "population_accuracy_production": float(settled["correct_production"].mean()),
            "population_accuracy_sign": float(
                work.dropna(subset=["correct_sign"])["correct_sign"].mean()
            ),
            "population_bucket_accuracy_production": {
                str(name): {"n": len(group), "accuracy": float(group["correct_production"].mean())}
                for name, group in settled.groupby("bucket", sort=True)
            },
            "opener_archive": OPENER_ARCHIVE,
            "microstructure_archive": MICROSTRUCTURE_ARCHIVE,
            "opener_evaluation": OPENER_EVALUATION,
        },
        "dispersion_pool": {
            "weeks_total": pool_summary["weeks_total"],
            "weeks_fallback_total": pool_summary["weeks_fallback_total"],
            "weeks_fallback_missing_data": pool_summary["weeks_fallback_missing_data"],
            "weeks_fallback_empty_filter": pool_summary["weeks_fallback_empty_filter"],
            "production_pool_matches_eval_pool_on_every_game": True,
        },
        "grades": grades,
        "multiplicity_note": (
            "Sixth reuse of the ~107-week opener population for the Best Pick family, and N0 "
            "was itself SELECTED on this window one day earlier; compounding look-reuse "
            "discount. The three era cells are a correlated decomposition of their own "
            "full-window cell, not independent votes. No rotation window assigned or spent."
        ),
        "binding_note": (
            "An interval containing zero is never a rejection; probability_positive is the "
            "continuous evidence. Closing grounds are limited to refuted_mechanism (resolved "
            "wrong sign or zero split-half reliability) and bounded_by_control. Era readings "
            "are magnitudes; a weaker era is never absence."
        ),
    }
    return summary, weeklies


_ARM_TEXT = {
    N0: "the served small-spread nominator (v2 restricted to spreads of 6.5 or less)",
    N1: "the unrestricted v2 nominator",
    "n2_reliable_bucket": (
        "a nominator restricted each week to the spread bucket with the best walk-forward "
        "realised accuracy (20-pseudo-observation shrinkage, expanding from 2020)"
    ),
    "n3_calibrated_pool": (
        "a nominator taking the highest calibrated displayed score inside v2's "
        "low-disagreement pool"
    ),
    "n4_calibrated_open": (
        "a nominator taking the highest calibrated displayed score across the whole card, "
        "with no dispersion pool"
    ),
    "control_perfect_foresight": (
        "the positive control: perfect-foresight nomination inside v2's pool"
    ),
}
_PLAIN_ARM = {
    N0: "the star rule we play now (never star a game with a spread bigger than 6.5)",
    N1: "the older star rule with no spread limit",
    "n2_reliable_bucket": (
        "starring only games in whichever spread range the model has been best at"
    ),
    "n3_calibrated_pool": "starring the game with the highest honest confidence score",
    "n4_calibrated_open": (
        "starring the game with the highest honest confidence score, ignoring whether the "
        "sportsbooks agree"
    ),
    "control_perfect_foresight": "a cheat rule that already knows the answers",
}


def record_cells(artifact_path: Path, *, replace: bool, recorded_at: str | None) -> list[list[str]]:
    """Record every predeclared cell on the PRIMARY grade, one command at a time."""
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    pop = artifact["population"]
    grade = artifact["grades"][PRIMARY_GRADE]
    timestamp = artifact_path.parent.name
    source = (
        f"scripts/best_pick_eras_eval.py; artifacts/{FAMILY}/{timestamp}/summary.json; "
        "docs/best_pick_eras.md"
    )
    evidence = (
        "Predeclared cell (docs/best_pick_eras.md). Neither admissible closing ground applies: "
        "no interval is resolved on the wrong side of zero, and the positive control resolves "
        "at roughly +40 accuracy points, proving only that this instrument can see a "
        "renomination effect that large -- it was never shown able to see a 2-point one, so it "
        "bounds nothing at the scale these arms live at. unresolved_below_power is the only "
        "admissible classification. An interval containing zero is the EXPECTED shape for a "
        "real-but-small signal at this evaluator's resolution, never a rejection."
    )
    commands: list[list[str]] = []
    for window in ERAS:
        window_slice = grade[window]
        seasons = ERAS[window]
        for cell_name, entry in window_slice["cells"].items():
            arm, base = cell_name.rsplit("_vs_", 1)
            base_arm = N0 if base == "n0" else N1
            name = f"{FAMILY}_{arm}_vs_{base}_{window}"
            agreement = grade["sign_agreement"].get(cell_name, {})
            description = (
                f"Best Pick nomination, {_ARM_TEXT[arm]} vs {_ARM_TEXT[base_arm]}, one nomination "
                f"per week graded at the frozen Tuesday opener on the production probability-rule "
                f"pick (correct_at_open_probability_rule), seasons {seasons[0]}-{seasons[-1]}, "
                f"paired weekly delta in Best-Pick accuracy points"
            )
            plain = (
                f"Which single game gets the week's star. Compares {_PLAIN_ARM[arm]} against "
                f"{_PLAIN_ARM[base_arm]} over {seasons[0]}-{seasons[-1]}. "
                f"{entry['effect_accuracy_points']:+.1f} points a week for the first one, "
                f"{entry['probability_positive']:.0%} likely the better of the two; they star a "
                f"different game in {entry['weeks_nominee_differs']} of {entry['weeks_paired']} "
                f"weeks."
            )
            notes = (
                f"Sixth reuse of the ~107-week opener population for the Best Pick family; the "
                f"incumbent N0 was itself selected on this same window one day earlier "
                f"(docs/best_pick_bucket_confidence.md), so any N0-favouring reading is inflated. "
                f"Bootstrap: seed {artifact['bootstrap_seed']}, {artifact['bootstrap_samples']:,} "
                f"draws, week-blocked on (season, week), within-week correlation zero. Raw record "
                f"{entry['arm_hits']}/{entry['weeks_paired']} vs "
                f"{entry['base_hits']}/{entry['weeks_paired']}; the arm wins "
                f"{entry['arm_wins_weeks']} weeks outright and loses {entry['base_wins_weeks']}. "
                f"Sides never change: this is a ranking only."
            )
            if window == FULL_WINDOW and agreement:
                notes += (
                    f" Era decomposition (2020-21 / 2022-23 / 2024-25): "
                    f"{agreement['era_effects']['2020_2021']:+.2f} / "
                    f"{agreement['era_effects']['2022_2023']:+.2f} / "
                    f"{agreement['era_effects']['2024_2025']:+.2f} accuracy points; "
                    f"{agreement['eras_agreeing_with_full_window']} of 3 eras share the full "
                    f"window's sign; sign reversal across eras: "
                    f"{agreement['sign_reversal_across_eras']}."
                )
            else:
                notes += (
                    " CORRELATED DECOMPOSITION: this era cell is a subset of "
                    f"{FAMILY}_{arm}_vs_{base}_{FULL_WINDOW} on the same weeks and is not an "
                    "independent vote; pool it with the overlap discount, or pool the full "
                    "window instead."
                )
            command = [
                sys.executable,
                "-m",
                "nfl_ats.cli",
                "weak-signals",
                "record",
                "--name",
                name,
                "--description",
                description,
                "--source",
                source,
                "--effect",
                f"{entry['effect_accuracy_points']:.10f}",
                "--effect-units",
                "accuracy_points",
                "--classification",
                "unresolved_below_power",
                "--league",
                "nfl",
                "--season-start",
                str(seasons[0]),
                "--season-end",
                str(seasons[-1]),
                "--interval-low",
                f"{entry['interval_low']:.10f}",
                "--interval-high",
                f"{entry['interval_high']:.10f}",
                "--probability-positive",
                f"{entry['probability_positive']:.10f}",
                "--sample-games",
                str(pop["games"] if window == FULL_WINDOW else entry["weeks_paired"]),
                "--sample-blocks",
                str(entry["weeks_paired"]),
                "--family",
                FAMILY,
                "--category",
                "modeling",
                "--classification-evidence",
                evidence,
                "--plain-summary",
                plain,
                "--notes",
                notes,
            ]
            if recorded_at:
                command += ["--recorded-at", recorded_at]
            if replace:
                command.append("--replace")
            commands.append(command)

    for command in commands:
        name = command[command.index("--name") + 1]
        print(f"=== recording {name} ===")
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
        print(result.stdout.strip()[:400])
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(
                f"weak-signals record failed for {name} (exit {result.returncode}); per "
                "AGENTS.md 'if a record command errors, the verdict is wrong, not the "
                "validator' -- fix the invocation, never weaken the classification."
            )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--recorded-at", default=None)
    args = parser.parse_args()

    if args.record:
        if args.artifact is None:
            raise SystemExit("--record needs --artifact <summary.json>")
        commands = record_cells(args.artifact, replace=args.replace, recorded_at=args.recorded_at)
        (args.artifact.parent / "record_commands.json").write_text(
            json.dumps(commands, indent=2), encoding="utf-8"
        )
        print(f"\nWrote {args.artifact.parent / 'record_commands.json'}")
        return

    summary, weeklies = run(args.repo)
    print(json.dumps(summary, indent=2, default=str))
    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = args.repo / "artifacts" / FAMILY / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    write_stamped_artifact(json.loads(json.dumps(summary, default=str)), out_dir / "summary.json")
    for grade, weekly in weeklies.items():
        weekly.to_csv(out_dir / f"weekly_{grade}.csv", index=False)
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
