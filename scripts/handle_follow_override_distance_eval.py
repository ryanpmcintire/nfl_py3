from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

LANE_S = Path("artifacts/handle_follow_on_card/20260909T232503Z")
POPULATION = LANE_S / "population.parquet"
LANE_S_RESULT = LANE_S / "result.json"
OUT_ROOT = Path("artifacts/handle_follow_override_distance")
SAMPLES = 20_000
SEED = 20260914
MONEY_THRESHOLD = 70.0


def block_bootstrap(deltas: np.ndarray, blocks: pd.DataFrame, key: str) -> dict[str, float]:
    labels = (
        blocks["season"].astype(str)
        if key == "season"
        else blocks["season"].astype(str) + "_" + blocks["week"].astype(str)
    )
    codes, uniques = pd.factorize(labels)
    groups = [np.flatnonzero(codes == index) for index in range(len(uniques))]
    generator = np.random.default_rng(SEED)
    draws = np.empty(SAMPLES, dtype=float)
    for sample in range(SAMPLES):
        chosen = generator.integers(0, len(groups), len(groups))
        positions = np.concatenate([groups[index] for index in chosen])
        draws[sample] = deltas[positions].mean()
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return {
        "estimate_accuracy_points": float(deltas.mean() * 100.0),
        "lower_accuracy_points": float(lower * 100.0),
        "upper_accuracy_points": float(upper * 100.0),
        "probability_positive": float((draws > 0).mean() + 0.5 * (draws == 0).mean()),
        "blocks": len(groups),
    }


def load_population() -> pd.DataFrame:
    pop = pd.read_parquet(POPULATION)
    pop = pop.loc[pop["correct_served"].notna()].reset_index(drop=True)
    return pop


def h1_frame(pop: pd.DataFrame) -> dict[str, object]:
    home_money = pop["spread_home_money_pct"].astype(float)
    away_money = pop["spread_away_money_pct"].astype(float)
    money_ok = home_money.notna() & away_money.notna()
    heavy_home = home_money > away_money
    money_max = pd.concat([home_money, away_money], axis=1).max(axis=1)
    fires = money_ok & money_max.ge(MONEY_THRESHOLD)
    served_pick_home = pop["served_pick_home"].to_numpy(dtype=bool)
    touched = fires.to_numpy() & (heavy_home.to_numpy(dtype=bool) != served_pick_home)
    home_probability = pd.to_numeric(pop["home_cover_probability_at_open"], errors="coerce")
    served_probability = np.where(served_pick_home, home_probability, 1.0 - home_probability)
    return {
        "fires": fires.to_numpy(),
        "touched": touched,
        "override_distance": np.asarray(served_probability, dtype=float) - 0.5,
        "served_correct": pop["correct_served"].to_numpy(dtype=float),
    }


def verify_reproduction(pop: pd.DataFrame, frame: dict[str, object], result: dict) -> dict:
    cell = result["cells"]["h1_handle70"]
    fires = int(np.asarray(frame["fires"]).sum())
    flips = int(np.asarray(frame["touched"]).sum())
    served_correct = np.asarray(frame["served_correct"], dtype=float)
    touched = np.asarray(frame["touched"], dtype=bool)
    candidate = np.where(touched, 1.0 - served_correct, served_correct)
    effect = float((candidate - served_correct).mean() * 100.0)
    stored_effect = float(cell["week_blocked"]["estimate_accuracy_points"])
    checks = {
        "fires": {"ours": fires, "stored": int(cell["rule_fires"])},
        "flips": {"ours": flips, "stored": int(cell["flips"])},
        "effect_accuracy_points": {"ours": effect, "stored": stored_effect},
    }
    if fires != int(cell["rule_fires"]) or flips != int(cell["flips"]):
        raise SystemExit(f"STOP: H1 reproduction mismatch {json.dumps(checks)}")
    if abs(effect - stored_effect) > 1e-9:
        raise SystemExit(f"STOP: H1 effect mismatch {json.dumps(checks)}")
    return checks


def arm_cell(
    pop: pd.DataFrame,
    served_correct: np.ndarray,
    base_flips: np.ndarray,
    arm_flips: np.ndarray,
) -> dict[str, object]:
    base = np.where(base_flips, 1.0 - served_correct, served_correct)
    arm = np.where(arm_flips, 1.0 - served_correct, served_correct)
    deltas = arm - base
    blocks = pop[["season", "week"]]
    return {
        "flips": int(arm_flips.sum()),
        "accuracy": float(arm.mean()),
        "nominee_differs": int((arm_flips != base_flips).sum()),
        "outcome_differs": int((deltas != 0).sum()),
        "week_blocked": block_bootstrap(deltas, blocks, "week"),
        "season_blocked": block_bootstrap(deltas, blocks, "season"),
    }


def flip_record(served_correct: np.ndarray, flips: np.ndarray) -> dict[str, int]:
    served = served_correct[flips]
    return {
        "served_wins": int(served.sum()),
        "served_losses": int(len(served) - served.sum()),
        "followed_wins": int(len(served) - served.sum()),
        "followed_losses": int(served.sum()),
    }


def run() -> dict[str, object]:
    pop = load_population()
    frame = h1_frame(pop)
    stored = json.loads(LANE_S_RESULT.read_text(encoding="utf-8"))
    checks = verify_reproduction(pop, frame, stored)

    served_correct = np.asarray(frame["served_correct"], dtype=float)
    touched = np.asarray(frame["touched"], dtype=bool)
    distance = np.asarray(frame["override_distance"], dtype=float)
    median = float(np.median(distance[touched]))
    shallow = touched & (distance <= median)
    deep = touched & (distance > median)

    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "predeclaration": "docs/handle_follow_override_distance.md",
        "population": str(POPULATION),
        "games": len(pop),
        "week_blocks": int(pop[["season", "week"]].drop_duplicates().shape[0]),
        "h1_reproduction": checks,
        "median_override_distance": median,
        "seed": SEED,
        "samples": SAMPLES,
        "arms": {},
        "flip_records": {},
    }
    no_flip = np.zeros(len(pop), dtype=bool)
    summary["arms"]["g0_incumbent_vs_card"] = arm_cell(pop, served_correct, no_flip, touched)
    summary["arms"]["g1_shallow_only_vs_g0"] = arm_cell(pop, served_correct, touched, shallow)
    summary["arms"]["g2_deep_only_vs_g0"] = arm_cell(pop, served_correct, touched, deep)
    summary["arms"]["g1_shallow_only_vs_card"] = arm_cell(pop, served_correct, no_flip, shallow)
    summary["arms"]["g2_deep_only_vs_card"] = arm_cell(pop, served_correct, no_flip, deep)

    oracle = touched & (served_correct == 0.0)
    summary["arms"]["control_foresight_vs_g0"] = arm_cell(pop, served_correct, touched, oracle)

    summary["flip_records"]["all_flips"] = flip_record(served_correct, touched)
    summary["flip_records"]["shallow_flips"] = flip_record(served_correct, shallow)
    summary["flip_records"]["deep_flips"] = flip_record(served_correct, deep)
    summary["flip_counts"] = {
        "all": int(touched.sum()),
        "shallow": int(shallow.sum()),
        "deep": int(deep.sum()),
    }
    summary["deep_flip_distance_range"] = {
        "min": float(distance[deep].min()) if deep.any() else None,
        "max": float(distance[deep].max()) if deep.any() else None,
    }
    summary["shallow_flip_distance_range"] = {
        "min": float(distance[shallow].min()) if shallow.any() else None,
        "max": float(distance[shallow].max()) if shallow.any() else None,
    }

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)
    detail = pop[["game_id", "season", "week", "tue_open_home_spread"]].copy()
    detail["override_distance"] = distance
    detail["h1_flip"] = touched
    detail["shallow"] = shallow
    detail["deep"] = deep
    detail["served_correct"] = served_correct
    detail.loc[touched].to_csv(out / "flips.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["artifact"] = str(out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="split the promoted heavy-handle follow rule by how far it overrides "
        "the served card's own probability, on the frozen lane-S population"
    )
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
