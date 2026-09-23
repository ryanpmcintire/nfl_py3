from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.mass_preserving_lattice import (
    BAND_HALF_WIDTH,
    BAND_STEP,
    BASE_PROBABILITY_POLICY,
    MAX_BAND,
    MIN_BAND_GAMES,
    MassPreservingRead,
    prior_pool,
    prior_pool_for_week,
    tilted_atoms,
)
from nfl_ats.modeling import regular_season_rows
from nfl_ats.total_conditioned_lattice_challenger import (
    ATOM_TOLERANCE as _ATOM_TOLERANCE,
)
from nfl_ats.total_conditioned_lattice_challenger import (
    CHALLENGER_POLICY,
    TOTAL_BAND_EDGES,
    TOTAL_BAND_LABELS,
    total_band,
    total_conditioned_read,
)

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "data"
ARTIFACTS_ROOT = REPO / "artifacts"
FEATURE_TABLE = DATA_ROOT / "processed" / "game_features_weak_stack.parquet"
OUT_ROOT = ARTIFACTS_ROOT / "total_conditioned_lattice"


def three_way_outcome(result: float, line: float) -> int:
    if abs(result - line) < _ATOM_TOLERANCE:
        return 1
    return 0 if result > line else 2


def three_way_log_loss(read: MassPreservingRead, outcome: int, epsilon: float = 1e-6) -> float:
    probs = (read.cover, read.push, read.loss)
    return -float(np.log(np.clip(probs[outcome], epsilon, 1.0)))


def three_way_brier(read: MassPreservingRead, outcome: int) -> float:
    probs = np.array([read.cover, read.push, read.loss], dtype=float)
    target = np.zeros(3)
    target[outcome] = 1.0
    return float(np.sum((probs - target) ** 2))


def run(seasons: range, *, label: str) -> dict:
    features = pd.read_parquet(FEATURE_TABLE)
    features["gameday"] = pd.to_datetime(features["gameday"], errors="raise")
    pool = prior_pool(features)
    totals = features.loc[:, ["game_id", "total_line"]].copy()
    totals["game_id"] = totals["game_id"].astype(str)
    totals["total_line"] = pd.to_numeric(totals["total_line"], errors="coerce")
    pool = pool.merge(totals, on="game_id", how="left")
    pool["total_band"] = pool["total_line"].map(
        lambda v: total_band(float(v)) if pd.notna(v) else None
    )

    reg = regular_season_rows(features).copy()
    reg["gameday"] = pd.to_datetime(reg["gameday"])
    reg["game_id"] = reg["game_id"].astype(str)
    reg["total_line"] = pd.to_numeric(reg["total_line"], errors="coerce")
    reg["spread_line"] = pd.to_numeric(reg["spread_line"], errors="coerce")
    reg["result"] = pd.to_numeric(reg["result"], errors="coerce")
    targets = reg.loc[
        reg["season"].isin(list(seasons))
        & reg["result"].notna()
        & reg["spread_line"].notna()
        & reg["total_line"].notna()
    ].copy()
    targets["result"] = np.rint(targets["result"].to_numpy(dtype=float))
    targets["target_band"] = targets["total_line"].map(lambda v: total_band(float(v)))

    rows: list[dict] = []
    for (season, week), group in targets.groupby(["season", "week"], sort=True):
        cutoff = pd.Timestamp(group["gameday"].min())
        exclude = set(group["game_id"].astype(str))
        eligible = prior_pool_for_week(
            pool, season=int(season), week=int(week), cutoff=cutoff, exclude_game_ids=exclude
        )
        eligible = eligible.loc[eligible["total_band"].notna()]
        if eligible.empty:
            continue
        pool_line = eligible["line"].to_numpy(dtype=float)
        pool_margin = eligible["result"].to_numpy(dtype=float)
        pool_total_band = eligible["total_band"].to_numpy()
        for _, row in group.iterrows():
            line = float(row["spread_line"])
            point = line
            served_selected = np.abs(pool_line - line) <= BAND_HALF_WIDTH
            band = BAND_HALF_WIDTH
            while True:
                served_selected = np.abs(pool_line - line) <= band
                if int(served_selected.sum()) >= MIN_BAND_GAMES or band >= MAX_BAND:
                    break
                band = min(band + BAND_STEP, MAX_BAND)
            served_margins = pool_margin[served_selected]
            values, counts = np.unique(served_margins, return_counts=True)
            mass, theta = tilted_atoms(values, counts.astype(float), line, point)
            is_push = np.abs(values - line) < _ATOM_TOLERANCE
            served_read = MassPreservingRead(
                cover=float(mass[values > line + _ATOM_TOLERANCE].sum()),
                push=float(mass[is_push].sum()),
                loss=float(mass[values < line - _ATOM_TOLERANCE].sum()),
                theta=theta,
                band=band,
                band_games=int(served_selected.sum()),
                atoms=int(values.size),
                key_mass_3=float(mass[np.abs(np.abs(values) - 3.0) < _ATOM_TOLERANCE].sum()),
            )
            challenger_read, fallback = total_conditioned_read(
                pool_line, pool_margin, pool_total_band, str(row["target_band"]), line, point
            )
            outcome = three_way_outcome(float(row["result"]), line)
            rows.append(
                {
                    "game_id": row["game_id"],
                    "season": int(season),
                    "week": int(week),
                    "line": line,
                    "total_line": float(row["total_line"]),
                    "total_band": row["target_band"],
                    "result": float(row["result"]),
                    "outcome": outcome,
                    "is_integer_line": bool(abs(line - round(line)) < _ATOM_TOLERANCE),
                    "served_cover": served_read.cover,
                    "served_push": served_read.push,
                    "served_loss": served_read.loss,
                    "served_band": served_read.band,
                    "served_band_games": served_read.band_games,
                    "served_key_mass_3": served_read.key_mass_3,
                    "served_log_loss": three_way_log_loss(served_read, outcome),
                    "served_brier": three_way_brier(served_read, outcome),
                    "challenger_cover": challenger_read.cover,
                    "challenger_push": challenger_read.push,
                    "challenger_loss": challenger_read.loss,
                    "challenger_band": challenger_read.band,
                    "challenger_band_games": challenger_read.band_games,
                    "challenger_key_mass_3": challenger_read.key_mass_3,
                    "challenger_fallback_to_unconditioned": fallback,
                    "challenger_log_loss": three_way_log_loss(challenger_read, outcome),
                    "challenger_brier": three_way_brier(challenger_read, outcome),
                }
            )
    return {"label": label, "table": pd.DataFrame(rows)}


def forced_pick_accuracy(table: pd.DataFrame, prefix: str) -> tuple[float, int]:
    decisive = table.loc[table["outcome"] != 1]
    if decisive.empty:
        return float("nan"), 0
    pick_home = decisive[f"{prefix}_cover"] >= decisive[f"{prefix}_loss"]
    actual_home = decisive["outcome"] == 0
    correct = (pick_home == actual_home).astype(float)
    return float(correct.mean()), len(decisive)


def push_calibration(table: pd.DataFrame, prefix: str) -> dict:
    integer_rows = table.loc[table["is_integer_line"]]
    if integer_rows.empty:
        return {"games": 0}
    predicted = integer_rows[f"{prefix}_push"]
    actual = (integer_rows["outcome"] == 1).astype(float)
    return {
        "games": len(integer_rows),
        "mean_predicted_push": float(predicted.mean()),
        "actual_push_rate": float(actual.mean()),
        "predicted_minus_actual": float(predicted.mean() - actual.mean()),
    }


def summarize(result: dict) -> dict:
    table = result["table"]
    if table.empty:
        return {"label": result["label"], "games": 0}
    served_ll = float(table["served_log_loss"].mean())
    challenger_ll = float(table["challenger_log_loss"].mean())
    served_brier = float(table["served_brier"].mean())
    challenger_brier = float(table["challenger_brier"].mean())
    served_acc, decisive_n = forced_pick_accuracy(table, "served")
    challenger_acc, _ = forced_pick_accuracy(table, "challenger")
    both_decisive = table.loc[table["outcome"] != 1].copy()
    served_pick_home = both_decisive["served_cover"] >= both_decisive["served_loss"]
    challenger_pick_home = both_decisive["challenger_cover"] >= both_decisive["challenger_loss"]
    actual_home = both_decisive["outcome"] == 0
    served_correct = served_pick_home == actual_home
    challenger_correct = challenger_pick_home == actual_home
    agree = int((served_pick_home == challenger_pick_home).sum())
    disagree = int((served_pick_home != challenger_pick_home).sum())
    challenger_wins_on_disagreement = int(
        ((served_pick_home != challenger_pick_home) & challenger_correct).sum()
    )
    served_wins_on_disagreement = int(
        ((served_pick_home != challenger_pick_home) & served_correct).sum()
    )
    return {
        "label": result["label"],
        "games": len(table),
        "seasons": sorted(table["season"].unique().tolist()),
        "total_band_counts": table["total_band"].value_counts().to_dict(),
        "log_loss": {"served": served_ll, "challenger": challenger_ll},
        "brier": {"served": served_brier, "challenger": challenger_brier},
        "forced_pick_accuracy": {
            "served": served_acc,
            "challenger": challenger_acc,
            "decisive_games": decisive_n,
        },
        "paired_disagreement": {
            "agree": agree,
            "disagree": disagree,
            "challenger_correct_on_disagreement": challenger_wins_on_disagreement,
            "served_correct_on_disagreement": served_wins_on_disagreement,
        },
        "push_calibration_integer_lines": {
            "served": push_calibration(table, "served"),
            "challenger": push_calibration(table, "challenger"),
        },
        "key_mass_3": {
            "served_mean": float(table["served_key_mass_3"].mean()),
            "challenger_mean": float(table["challenger_key_mass_3"].mean()),
        },
        "challenger_fallback_rate": float(table["challenger_fallback_to_unconditioned"].mean()),
        "challenger_band_games_median": float(table["challenger_band_games"].median()),
        "served_band_games_median": float(table["served_band_games"].median()),
    }


def loso_by_season(table: pd.DataFrame) -> list[dict]:
    out = []
    for season, group in table.groupby("season", sort=True):
        served_acc, n = forced_pick_accuracy(group, "served")
        challenger_acc, _ = forced_pick_accuracy(group, "challenger")
        out.append(
            {
                "season": int(season),
                "games": len(group),
                "decisive_games": n,
                "served_log_loss": float(group["served_log_loss"].mean()),
                "challenger_log_loss": float(group["challenger_log_loss"].mean()),
                "served_accuracy": served_acc,
                "challenger_accuracy": challenger_acc,
            }
        )
    return out


def main() -> None:
    primary = run(range(2020, 2026), label="primary_2020_2025")
    extended = run(range(2009, 2026), label="extended_2009_2025")

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    primary["table"].to_parquet(out_dir / "primary_2020_2025.parquet", index=False)
    extended["table"].to_parquet(out_dir / "extended_2009_2025.parquet", index=False)

    summary = {
        "generated_at_utc": ts,
        "served_base_probability_policy": BASE_PROBABILITY_POLICY,
        "challenger_policy": CHALLENGER_POLICY,
        "total_band_edges": TOTAL_BAND_EDGES,
        "total_band_labels": TOTAL_BAND_LABELS,
        "point_convention": "point = spread_line for both arms; isolates the key-number lattice "
        "mechanism (line-band vs line-band+total-band selection and tilting) from any margin "
        "forecast; not the served predicted_margin point",
        "line_source": "features table spread_line (no opener-evaluation override matched in "
        "this run); identical for both arms so the total-band delta is a controlled comparison",
        "primary_2020_2025": summarize(primary),
        "extended_2009_2025_reported_not_recorded": summarize(extended),
        "loso_by_season_primary": loso_by_season(primary["table"]),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
