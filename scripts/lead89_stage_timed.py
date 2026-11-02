from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_loso_upstream import SEASONS, Replay

from nfl_ats.pick_probability import MOVE_AVAILABLE_COLUMN, MOVE_COLUMN

OUTPUT = Path("tests/scratch/codex/lead89_stage_timed")
FEATURES = Path("data/processed/game_features_weak_stack.parquet")
QUOTES = Path("tests/scratch/codex/lead89_unit1/stage_quotes.parquet")
CLOCKS = Path("tests/scratch/codex/lead89_unit2/input_clock_audit.parquet")
SOURCE_DOC = "docs/lead89_stage_timed.md"
INJURY_ZEROED = (
    "diff_injury_offense_unavailability",
    "diff_injury_defense_unavailability",
    "diff_injury_special_teams_unavailability",
    "diff_injury_offensive_line_unavailability",
    "diff_injury_skill_unavailability",
    "diff_injury_front_unavailability",
    "diff_injury_secondary_unavailability",
    "diff_injury_skill_epa_value_lost",
    "diff_injury_defense_disruption_value_lost",
    "diff_qb_expected_epa_per_dropback",
    "diff_qb_start_probability",
)
STAGES = ("tuesday", "thursday", "sunday")
OUTER = (2023, 2024, 2025)
CONTRASTS = (("sunday", "tuesday"), ("thursday", "tuesday"), ("sunday", "thursday"))
DRAWS = 10000
SEED = 8903
FAMILY = "lead89_stage_timed_nominee_contrasts"


def stage_moves(stage: str) -> pd.DataFrame:
    quotes = pd.read_parquet(
        QUOTES, columns=["game_id", "stage", "move_available", "stage_move_toward_home"]
    )
    rows = quotes.loc[quotes.stage.eq(stage)]
    games = pd.DataFrame({"game_id": rows.game_id.unique()}).set_index("game_id")
    usable = rows.loc[rows.move_available.astype(bool)]
    median = usable.groupby("game_id").stage_move_toward_home.median()
    games["move"] = median.reindex(games.index)
    games["present"] = 1.0
    return games


def prepare(replay: Replay, stage: str, zero: bool) -> None:
    replay.raw_cache.clear()
    replay.input_cache.clear()
    replay.fold_audits.clear()
    if zero:
        staged = pd.read_parquet(OUTPUT / f"stage_features_{stage}.parquet").set_index("game_id")
        cases = pd.read_parquet(OUTPUT / f"stage_cases_{stage}.parquet")
        for frame in (replay.features, replay.games):
            known = frame.game_id.isin(staged.index)
            for column in INJURY_ZEROED:
                mapped = frame.game_id.map(staged[column])
                frame[column] = np.where(known, mapped, frame[column])
        print(stage, cases.case.value_counts().to_dict())
    table = stage_moves(stage)
    move = (
        replay.games.game_id.map(table["move"])
        if stage != "tuesday"
        else pd.Series(np.nan, index=replay.games.index)
    )
    if stage == "tuesday":
        replay.games[MOVE_COLUMN] = 0.0
        replay.games[MOVE_AVAILABLE_COLUMN] = 0.0
    else:
        replay.games[MOVE_COLUMN] = move.fillna(0.0)
        replay.games[MOVE_AVAILABLE_COLUMN] = move.notna().astype(float)


def run_stage(replay: Replay, stage: str, zero: bool) -> tuple[pd.DataFrame, dict]:
    output = OUTPUT / stage
    if not (output / "predictions.parquet").exists():
        prepare(replay, stage, zero)
        replay.run(output)
    frame = pd.read_parquet(output / "predictions.parquet")
    meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    return frame, meta


def eligibility(frame: pd.DataFrame, stage: str, clocks: pd.DataFrame) -> pd.DataFrame:
    table = stage_moves(stage)
    frame = frame.loc[frame.game_id.isin(table.index)].copy()
    if stage == "sunday":
        ok = clocks.loc[clocks.stage.eq("sunday") & clocks.available, "game_id"]
        frame = frame.loc[frame.game_id.isin(ok)]
    return frame


def nominees(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["pick_home"] = frame.four_term_probability.ge(0.5)
    frame["confidence"] = np.maximum(frame.four_term_probability, 1 - frame.four_term_probability)
    frame = frame.sort_values(
        ["season", "week", "confidence", "game_id"], ascending=[True, True, False, True]
    )
    best = frame.groupby(["season", "week"], sort=True).head(1).copy()
    covered = best.home_covered
    best["reward"] = np.where(best.pick_home, covered, 1 - covered)
    best["push"] = covered.isna()
    best["reward"] = best.reward.fillna(0.5)
    best["pick_prob"] = best.confidence
    best["brier"] = np.where(best.push, np.nan, (best.confidence - best.reward) ** 2)
    return best.set_index(["season", "week"])


def interval(draws: np.ndarray) -> tuple[float, float, float]:
    positive = float((draws > 0).mean() + 0.5 * (draws == 0).mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high), positive


def contrast(later: pd.DataFrame, early: pd.DataFrame, weeks: list, rng) -> dict:
    index = pd.MultiIndex.from_tuples(weeks, names=["season", "week"])
    a = later.reindex(index)
    b = early.reindex(index)
    diff = (a.reward - b.reward).to_numpy() * 100.0
    brier_gain = (b.brier - a.brier).to_numpy()
    same = (a.game_id == b.game_id).to_numpy()
    decisive = (~a.push & ~b.push).to_numpy()
    a_win, b_win = (a.reward == 1).to_numpy(), (b.reward == 1).to_numpy()
    a_loss, b_loss = (a.reward == 0).to_numpy(), (b.reward == 0).to_numpy()
    only_a = int((a_win & b_loss).sum())
    only_b = int((b_win & a_loss).sum())
    n = len(diff)
    pick = rng.integers(0, n, size=(DRAWS, n))
    point = float(diff.mean())
    low, high, positive = interval(diff[pick].mean(axis=1))
    valid = ~np.isnan(brier_gain)
    bg = brier_gain[valid]
    bpick = rng.integers(0, len(bg), size=(DRAWS, len(bg)))
    blow, bhigh, bpositive = interval(bg[bpick].mean(axis=1))
    discordant = only_a + only_b
    p_exact = float(binomtest(only_a, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "weeks": n,
        "decisive_both_weeks": int(decisive.sum()),
        "same_game_weeks": int(same.sum()),
        "later_record_wlp": [int(a_win.sum()), int(a_loss.sum()), int((~a_win & ~a_loss).sum())],
        "early_record_wlp": [int(b_win.sum()), int(b_loss.sum()), int((~b_win & ~b_loss).sum())],
        "later_wins_early_loses": only_a,
        "early_wins_later_loses": only_b,
        "exact_two_sided_p_discordant": p_exact,
        "reward_later_minus_early_accuracy_points": point,
        "reward_interval95": [low, high],
        "reward_probability_positive": positive,
        "brier_gain_later": float(bg.mean()) if len(bg) else None,
        "brier_interval95": [blow, bhigh],
        "brier_probability_positive": bpositive,
        "brier_weeks": int(valid.sum()),
    }


def arm_summary(frame: pd.DataFrame) -> dict:
    decisive = frame.loc[frame.home_covered.notna()]
    p = decisive.four_term_probability.clip(1e-6, 1 - 1e-6).to_numpy()
    y = decisive.home_covered.to_numpy()
    model = decisive.model_probability.clip(1e-6, 1 - 1e-6).to_numpy()
    return {
        "games": len(decisive),
        "accuracy": float(((p >= 0.5) == (y == 1)).mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "model_only_accuracy": float(((model >= 0.5) == (y == 1)).mean()),
        "model_only_log_loss": float(-(y * np.log(model) + (1 - y) * np.log1p(-model)).mean()),
        "market_even_log_loss": float(np.log(2)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    clocks = pd.read_parquet(CLOCKS)
    clocks["decision_at"] = pd.to_datetime(clocks.decision_at, utc=True)
    with threadpool_limits(limits=1):
        replay = Replay(
            Path("artifacts/pick_probability/20261102T171412Z"),
            FEATURES,
            Path("data/raw/20260929T191306Z/schedules.parquet"),
            Path("data/pbp/raw/20260929T191306Z"),
        )
        plan = {"tuesday": True, "thursday": True, "sunday": False}
        arms, metas = {}, {}
        for stage in ("sunday", "tuesday", "thursday"):
            arms[stage], metas[stage] = run_stage(replay, stage, plan[stage])
    baseline = pd.read_parquet("artifacts/loso_upstream/20261102T213642884071Z/predictions.parquet")
    outcome = baseline[["game_id", "home_covered"]]
    audit_rows, picks, summaries = [], {}, {}
    for stage in STAGES:
        frame = arms[stage]
        frame = frame.drop(columns=["home_covered"], errors="ignore").merge(
            outcome, on="game_id", validate="one_to_one"
        )
        eligible = eligibility(frame, stage, clocks)
        all_games = clocks.loc[clocks.stage.eq(stage), "game_id"].nunique()
        weeks = eligible.groupby("season").week.nunique()
        stage_clock = clocks.loc[clocks.stage.eq(stage) & clocks.game_id.isin(eligible.game_id)]
        if stage == "sunday":
            leaks = int(
                (
                    stage_clock[["home_injury_observed_at", "away_injury_observed_at"]]
                    .max(axis=1)
                    .gt(stage_clock.decision_at)
                ).sum()
            )
        else:
            leaks = int(
                (
                    arms[stage]
                    .loc[arms[stage].game_id.isin(eligible.game_id), "model_logit"]
                    .isna()
                ).sum()
            )
        audit_rows.append(
            {
                "stage": stage,
                "stage_games": int(all_games),
                "eligible_games": int(eligible.game_id.nunique()),
                "injury_inputs": "reconstructed as known at the stage cutoff (carry rule)"
                if plan[stage]
                else "date_modified at or before decision",
                "injury_clock_violations": leaks,
                "weeks_with_eligible_game": {str(k): int(v) for k, v in weeks.items()},
            }
        )
        picks[stage] = nominees(eligible)
        summaries[stage] = arm_summary(eligible)
    common = set(picks["tuesday"].index)
    for stage in STAGES:
        common &= set(picks[stage].index)
    weeks_all = sorted(common)
    rng = np.random.default_rng(SEED)
    results = {}
    looks = 0
    for later, early in CONTRASTS:
        name = f"{later}_vs_{early}"
        results[name] = {}
        populations = {
            "all_2020_2025": weeks_all,
            "outer_2023_2025": [w for w in weeks_all if w[0] in OUTER],
            **{str(s): [w for w in weeks_all if w[0] == s] for s in SEASONS},
        }
        for label, weeks in populations.items():
            results[name][label] = contrast(picks[later], picks[early], weeks, rng)
            looks += 2
    sunday_available_weeks = {s: int(sum(1 for w in weeks_all if w[0] == s)) for s in SEASONS}
    weak = []
    for name, cells in results.items():
        pair = name.replace("_vs_", " nominee minus ")
        for label, cell in cells.items():
            seasons = (SEASONS[0], SEASONS[-1])
            if label == "outer_2023_2025":
                seasons = (OUTER[0], OUTER[-1])
            elif label.isdigit():
                seasons = (int(label), int(label))
            base = {
                "league": "nfl",
                "season_start": seasons[0],
                "season_end": seasons[1],
                "sample_blocks": cell["weeks"],
                "sample_games": cell["weeks"],
            }
            weak.append(
                {
                    **base,
                    "name": f"lead89_stage_timed_{name}_{label}_reward",
                    "description": f"Weekly Best Pick reward, {pair} nominee, "
                    f"{label}, paired by week, graded at the opener",
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
                        "name": f"lead89_stage_timed_{name}_{label}_nominee_brier",
                        "description": f"Nominee Brier gain, {pair} nominee, "
                        f"{label}, decisive weeks, graded at the opener",
                        "effect": cell["brier_gain_later"],
                        "effect_units": "brier_improvement",
                        "interval_low": cell["brier_interval95"][0],
                        "interval_high": cell["brier_interval95"][1],
                        "probability_positive": cell["brier_probability_positive"],
                        "sample_blocks": cell["brier_weeks"],
                        "sample_games": cell["brier_weeks"],
                    }
                )
    batch = {
        "family": FAMILY,
        "source": SOURCE_DOC,
        "classification": "unresolved_below_power",
        "category": "market",
        "recorded_at": datetime.now(UTC).date().isoformat(),
        "plain_summary": "Compares locking the Best Pick on Tuesday, Thursday or Sunday using only "
        "information that existed at each of those times, to see what waiting is worth.",
        "cells": weak,
    }
    (OUTPUT / "record_batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    report = {
        "audit": audit_rows,
        "arm_summaries": summaries,
        "weeks_common": len(weeks_all),
        "weeks_by_season": sunday_available_weeks,
        "contrasts": results,
        "looks": looks,
        "bootstrap": {"draws": DRAWS, "seed": SEED, "unit": "season-week"},
        "calibration_coefficients": {
            s: {k: v["coefficients"] for k, v in metas[s]["calibration_folds"].items()}
            for s in STAGES
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {"audit": audit_rows, "summaries": summaries, "looks": looks, "weeks": len(weeks_all)},
            indent=1,
            default=str,
        )
    )
    for name, cells in results.items():
        for label in ("all_2020_2025", "outer_2023_2025"):
            print(name, label, json.dumps(cells[label], default=str))


if __name__ == "__main__":
    main()
