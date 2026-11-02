from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lead89_stage_timed import (
    CLOCKS,
    CONTRASTS,
    DRAWS,
    OUTER,
    QUOTES,
    SEASONS,
    SEED,
    STAGES,
    contrast,
    eligibility,
    nominees,
)

TARGET = Path("tests/scratch/codex/lead89_option_policy")
STAGED = Path("tests/scratch/codex/lead89_stage_timed")
SOURCE_DOC = "docs/lead89_option_policy.md"
FAMILY = "lead89_option_policy_contrasts"
FIXED = ("tuesday", "thursday", "sunday")


def stage_times() -> pd.DataFrame:
    quotes = pd.read_parquet(QUOTES, columns=["game_id", "stage", "decision_at", "kickoff"])
    quotes["decision_at"] = pd.to_datetime(quotes.decision_at, utc=True)
    quotes["kickoff"] = pd.to_datetime(quotes.kickoff, utc=True)
    return quotes.groupby(["game_id", "stage"], as_index=False).agg(
        decision_at=("decision_at", "first"), kickoff=("kickoff", "first")
    )


def load_arms() -> dict[str, pd.DataFrame]:
    clocks = pd.read_parquet(CLOCKS)
    outcome = pd.read_parquet("artifacts/loso_upstream/20261102T213642884071Z/predictions.parquet")[
        ["game_id", "home_covered"]
    ]
    times = stage_times()
    arms = {}
    for stage in STAGES:
        frame = pd.read_parquet(STAGED / stage / "predictions.parquet")
        frame = frame.drop(columns=["home_covered"], errors="ignore").merge(
            outcome, on="game_id", validate="one_to_one"
        )
        eligible = eligibility(frame, stage, clocks)
        when = times.loc[times.stage.eq(stage)].set_index("game_id")
        eligible["playable"] = eligible.game_id.map(when.kickoff).gt(
            eligible.game_id.map(when.decision_at)
        )
        arms[stage] = eligible
    return arms


def fit_law(frame: pd.DataFrame) -> dict:
    sunday_slope, sunday_intercept = np.polyfit(frame.thursday, frame.sunday, 1)
    value_thursday = np.maximum(frame.thursday, sunday_intercept + sunday_slope * frame.thursday)
    tuesday_slope, tuesday_intercept = np.polyfit(frame.tuesday, value_thursday, 1)
    return {
        "sunday_on_thursday": [float(sunday_intercept), float(sunday_slope)],
        "thursday_value_on_tuesday": [float(tuesday_intercept), float(tuesday_slope)],
    }


def choose_stage(row: pd.Series, law: dict) -> str:
    intercept, slope = law["thursday_value_on_tuesday"]
    if intercept + slope * row.tuesday > row.tuesday:
        intercept, slope = law["sunday_on_thursday"]
        return "sunday" if intercept + slope * row.thursday > row.thursday else "thursday"
    return "tuesday"


def apply_policy(conf: pd.DataFrame, picks: dict, laws: dict) -> pd.DataFrame:
    rows, stages = [], []
    for key, row in conf.iterrows():
        stage = choose_stage(row, laws[key[0]])
        stages.append(stage)
        rows.append(picks[stage].loc[key])
    frame = pd.DataFrame(rows)
    frame.index = conf.index
    frame["locked_stage"] = stages
    return frame


def cells_for(name: str, fixed: str, results: dict) -> list[dict]:
    weak = []
    pair = f"option policy nominee minus {fixed} nominee"
    for label, cell in results[name].items():
        span = (SEASONS[0], SEASONS[-1])
        if label == "outer_2023_2025":
            span = (OUTER[0], OUTER[-1])
        elif label.isdigit():
            span = (int(label), int(label))
        base = {
            "league": "nfl",
            "season_start": span[0],
            "season_end": span[1],
            "sample_blocks": cell["weeks"],
            "sample_games": cell["weeks"],
        }
        weak.append(
            {
                **base,
                "name": f"lead89_{name}_{label}_reward",
                "description": f"Weekly Best Pick reward, {pair}, {label}, paired by week, "
                "graded at the opener",
                "effect": cell["reward_later_minus_early_accuracy_points"],
                "effect_units": "accuracy_points",
                "interval_low": cell["reward_interval95"][0],
                "interval_high": cell["reward_interval95"][1],
                "probability_positive": cell["reward_probability_positive"],
            }
        )
        if cell["brier_gain_later"] is not None:
            weak.append(
                {
                    **base,
                    "name": f"lead89_{name}_{label}_nominee_brier",
                    "description": f"Nominee Brier gain, {pair}, {label}, decisive weeks, "
                    "graded at the opener",
                    "effect": cell["brier_gain_later"],
                    "effect_units": "brier_improvement",
                    "interval_low": cell["brier_interval95"][0],
                    "interval_high": cell["brier_interval95"][1],
                    "probability_positive": cell["brier_probability_positive"],
                    "sample_blocks": cell["brier_weeks"],
                    "sample_games": cell["brier_weeks"],
                }
            )
    return weak


def main() -> None:
    started = time.time()
    TARGET.mkdir(parents=True, exist_ok=True)
    arms = load_arms()
    raw = {stage: nominees(arms[stage]) for stage in STAGES}
    picks = {stage: nominees(arms[stage].loc[arms[stage].playable]) for stage in STAGES}
    changes = {
        stage: int((raw[stage].game_id != picks[stage].reindex(raw[stage].index).game_id).sum())
        for stage in STAGES
    }
    common = set(picks["tuesday"].index)
    for stage in STAGES:
        common &= set(picks[stage].index)
    weeks = sorted(common)
    index = pd.MultiIndex.from_tuples(weeks, names=["season", "week"])
    conf = pd.DataFrame({s: picks[s].reindex(index).pick_prob for s in STAGES})
    seasons = conf.index.get_level_values("season")
    in_law = fit_law(conf)
    oos_laws = {season: fit_law(conf.loc[seasons != season]) for season in SEASONS}
    oos = apply_policy(conf, picks, oos_laws)
    in_sample = apply_policy(conf, picks, dict.fromkeys(SEASONS, in_law))
    shares = {
        "out_of_season": oos.locked_stage.value_counts().to_dict(),
        "in_sample": in_sample.locked_stage.value_counts().to_dict(),
    }
    rng = np.random.default_rng(SEED)
    populations = {
        "all_2020_2025": weeks,
        "outer_2023_2025": [w for w in weeks if w[0] in OUTER],
        **{str(s): [w for w in weeks if w[0] == s] for s in SEASONS},
    }
    results, looks, gap = {}, 0, {}
    for fixed in FIXED:
        results[f"option_policy_vs_{fixed}"] = {}
        for label, wk in populations.items():
            results[f"option_policy_vs_{fixed}"][label] = contrast(oos, picks[fixed], wk, rng)
            looks += 2
    for fixed in FIXED:
        insample = contrast(in_sample, picks[fixed], weeks, rng)
        a = insample["reward_later_minus_early_accuracy_points"]
        b = results[f"option_policy_vs_{fixed}"]["all_2020_2025"][
            "reward_later_minus_early_accuracy_points"
        ]
        gap[fixed] = {"in_sample_points": a, "out_of_season_points": b, "gap_points": a - b}
    reproduce = {
        f"{later}_vs_{early}": contrast(picks[later], picks[early], weeks, rng)
        for later, early in CONTRASTS
    }
    weak = []
    for name in results:
        weak += cells_for(name, name.rsplit("_", 1)[1], results)
    batch = {
        "family": FAMILY,
        "source": SOURCE_DOC,
        "classification": "unresolved_below_power",
        "category": "market",
        "recorded_at": datetime.now(UTC).date().isoformat(),
        "plain_summary": "Compares a Best Pick that waits only when later games are expected to "
        "offer a stronger pick against locking on Tuesday, Thursday or Sunday.",
        "cells": weak,
    }
    (TARGET / "record_batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    report = {
        "weeks_common": len(weeks),
        "playable_filter_changes_nominee": changes,
        "law_all_seasons": in_law,
        "law_by_held_out_season": oos_laws,
        "lock_stage_counts": shares,
        "in_sample_vs_out_of_season": gap,
        "contrasts": results,
        "fixed_contrasts_reproduced": reproduce,
        "looks_new": looks,
        "bootstrap": {"draws": DRAWS, "seed": SEED, "unit": "season-week"},
        "elapsed_seconds": time.time() - started,
    }
    (TARGET / "results.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8"
    )
    keys = (
        "weeks_common",
        "playable_filter_changes_nominee",
        "law_all_seasons",
        "lock_stage_counts",
        "in_sample_vs_out_of_season",
        "looks_new",
    )
    print(json.dumps({k: report[k] for k in keys}, indent=1, default=str))
    for name, cells in results.items():
        for label in populations:
            c = cells[label]
            print(
                name,
                label,
                c["later_record_wlp"],
                c["early_record_wlp"],
                round(c["reward_later_minus_early_accuracy_points"], 2),
                [round(x, 2) for x in c["reward_interval95"]],
                c["reward_probability_positive"],
                round(c["exact_two_sided_p_discordant"], 3),
            )
    for name, c in reproduce.items():
        print(
            "fixed",
            name,
            c["later_record_wlp"],
            c["early_record_wlp"],
            round(c["reward_later_minus_early_accuracy_points"], 2),
        )


if __name__ == "__main__":
    main()
