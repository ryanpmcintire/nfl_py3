from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

N_TRAILING = 3
FLIP_DECILE_THRESHOLD = 8
N_DECILES = 10
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260910
GRADED_SEASONS = (2020, 2021)
OPENER_EVAL_ARTIFACT = "artifacts/opener_evaluation/20260910T211255Z/per_game.parquet"

PRIMETIME_WINDOWS = {
    "SNF",
    "SNF**",
    "MNF",
    "MNF**",
    "MNF early",
    "MNF late",
    "TNF",
    "TNF*",
    "TNF**",
    "Kickoff",
}
SUNDAY_LATE_WINDOWS = {"Late DH", "National", "Late Sat"}
SUNDAY_EARLY_WINDOWS = {
    "Early DH",
    "Regional",
    "Single",
    "Early Sat",
    "Sat 1",
    "Sat 3",
    "London",
    "London*",
    "Special",
    "Special**",
}


def window_bucket(window: str) -> str:
    if window in PRIMETIME_WINDOWS:
        return "primetime"
    if window in SUNDAY_LATE_WINDOWS:
        return "sunday_late"
    if window in SUNDAY_EARLY_WINDOWS:
        return "sunday_early"
    return "unknown"


def latest_smw_publications_snapshot(repo_root: Path) -> Path:
    candidates = sorted(
        (repo_root / "data" / "raw" / "sports_media_watch_publications").glob("*/manifest.json")
    )
    complete = []
    for manifest_path in candidates:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") == "COMPLETE":
            complete.append(manifest_path.parent)
    pool = complete if complete else [path.parent for path in candidates]
    if not pool:
        raise FileNotFoundError(
            f"no sports_media_watch_publications snapshot found under {repo_root}"
        )
    return sorted(pool)[-1] / "ratings_rows.parquet"


def load_usable_rows(snapshot: Path) -> pd.DataFrame:
    frame = pd.read_parquet(snapshot)
    usable = frame[
        (frame["point_in_time_usable"] == True)  # noqa: E712
        & frame["home_team"].notna()
        & frame["away_team"].notna()
    ].copy()
    usable["season"] = usable["season"].astype(int)
    usable["week"] = usable["week"].astype(int)
    usable["bucket"] = usable["window"].apply(window_bucket)
    return usable


def coverage_by_season(usable: pd.DataFrame) -> dict[str, int]:
    return {
        str(season): int(count)
        for season, count in usable["season"].value_counts().sort_index().items()
    }


def build_team_long(usable: pd.DataFrame) -> pd.DataFrame:
    bucket_stats = usable.groupby("bucket")["viewers"].agg(["mean", "std"])
    merged = usable.merge(
        bucket_stats.rename(columns={"mean": "bucket_mean", "std": "bucket_std"}),
        on="bucket",
        how="left",
    )
    merged["z"] = (merged["viewers"] - merged["bucket_mean"]) / merged["bucket_std"]

    home_side = merged[["season", "week", "home_team", "z"]].rename(columns={"home_team": "team"})
    away_side = merged[["season", "week", "away_team", "z"]].rename(columns={"away_team": "team"})
    team_long = pd.concat([home_side, away_side], ignore_index=True)
    team_long["key"] = team_long["season"] * 100 + team_long["week"]
    team_long = team_long.sort_values(["team", "key"]).reset_index(drop=True)
    team_long["trailing"] = team_long.groupby("team")["z"].transform(
        lambda s: s.rolling(N_TRAILING, min_periods=N_TRAILING).mean()
    )
    return team_long


def build_trailing_lookup(team_long: pd.DataFrame) -> dict[str, tuple[list[int], list[float]]]:
    lookup: dict[str, tuple[list[int], list[float]]] = {}
    for team, group in team_long.groupby("team"):
        lookup[str(team)] = (group["key"].tolist(), group["trailing"].tolist())
    return lookup


def trailing_as_of(lookup: dict[str, tuple[list[int], list[float]]], team: str, key: int) -> float:
    if team not in lookup:
        return float("nan")
    keys, values = lookup[team]
    position = bisect_left(keys, key)
    while position > 0 and keys[position - 1] >= key:
        position -= 1
    if position == 0:
        return float("nan")
    return values[position - 1]


def load_graded_games(repo_root: Path) -> pd.DataFrame:
    per_game = pd.read_parquet(repo_root / OPENER_EVAL_ARTIFACT)
    parts = per_game["game_id"].str.split("_")
    per_game["away_team"] = parts.str[2]
    per_game["home_team"] = parts.str[3]
    per_game["key"] = per_game["season"] * 100 + per_game["week"]
    return per_game[per_game["season"].isin(GRADED_SEASONS)].copy().reset_index(drop=True)


def attach_trailing_and_cover(
    graded: pd.DataFrame, lookup: dict[str, tuple[list[int], list[float]]]
) -> pd.DataFrame:
    graded = graded.copy()
    graded["home_trailing"] = [
        trailing_as_of(lookup, team, key)
        for team, key in zip(graded["home_team"], graded["key"], strict=True)
    ]
    graded["away_trailing"] = [
        trailing_as_of(lookup, team, key)
        for team, key in zip(graded["away_team"], graded["key"], strict=True)
    ]
    graded["home_covered"] = np.where(
        graded["margin_vs_open"] > 0, 1.0, np.where(graded["margin_vs_open"] < 0, 0.0, np.nan)
    )
    graded["away_covered"] = 1.0 - graded["home_covered"]
    return graded


def wilson_interval(favourable: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    phat = favourable / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))) / denom
    return center - half, center + half


def decile_table(graded: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    home_rows = graded[["season", "week", "home_trailing", "home_covered"]].rename(
        columns={"home_trailing": "trailing", "home_covered": "covered"}
    )
    away_rows = graded[["season", "week", "away_trailing", "away_covered"]].rename(
        columns={"away_trailing": "trailing", "away_covered": "covered"}
    )
    team_games = pd.concat([home_rows, away_rows], ignore_index=True)
    valid = team_games.dropna(subset=["trailing", "covered"]).copy()
    valid["decile"], edges = pd.qcut(
        valid["trailing"], N_DECILES, labels=False, duplicates="drop", retbins=True
    )
    valid["decile"] = valid["decile"].astype(int) + 1

    rows = []
    for decile, group in valid.groupby("decile"):
        n = len(group)
        k = round(float(group["covered"].sum()))
        rate = k / n
        lo, hi = wilson_interval(k, n)
        rows.append(
            {
                "decile": int(decile),  # type: ignore[arg-type]
                "n_team_games": n,
                "covers": k,
                "cover_rate": rate,
                "interval_low": lo,
                "interval_high": hi,
                "trailing_min": float(group["trailing"].min()),
                "trailing_max": float(group["trailing"].max()),
            }
        )
    table = pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)
    return table, edges


def block_bootstrap_delta(
    frame: pd.DataFrame, rng: np.random.Generator, samples: int
) -> np.ndarray:
    groups = frame.groupby(["season", "week"]).indices
    keys = list(groups.keys())
    index_arrays = [groups[key] for key in keys]
    n_blocks = len(keys)
    base = frame["correct_at_open_probability_rule"].to_numpy(float)
    flipped = frame["flipped_correct"].to_numpy(float)
    deltas = np.empty(samples)
    for i in range(samples):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([index_arrays[c] for c in chosen])
        deltas[i] = (flipped[idx].mean() - base[idx].mean()) * 100.0
    return deltas


def probability_positive(deltas: np.ndarray) -> float:
    return float((np.sum(deltas > 0) + 0.5 * np.sum(deltas == 0)) / len(deltas))


def production_screen(graded: pd.DataFrame, edges: np.ndarray) -> dict:
    working = graded.copy()
    working["picked_team"] = np.where(
        working["pick_home_at_open_probability_rule"], working["home_team"], working["away_team"]
    )
    working["picked_trailing"] = np.where(
        working["pick_home_at_open_probability_rule"],
        working["home_trailing"],
        working["away_trailing"],
    )
    bin_edges = [float(value) for value in edges]
    bin_edges[0] = float("-inf")
    bin_edges[-1] = float("inf")
    working["picked_decile"] = (
        pd.cut(working["picked_trailing"], bins=bin_edges, labels=False, include_lowest=True) + 1
    )
    working["flip"] = working["picked_decile"] >= FLIP_DECILE_THRESHOLD

    base_correct = working["correct_at_open_probability_rule"]
    working["flipped_correct"] = np.where(working["flip"], 1.0 - base_correct, base_correct)

    graded_only = working.dropna(subset=["correct_at_open_probability_rule"]).copy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    results = {}
    season_slices = {"pooled": GRADED_SEASONS, **{str(s): (s,) for s in GRADED_SEASONS}}
    for label, seasons in season_slices.items():
        subset = graded_only[graded_only["season"].isin(seasons)].reset_index(drop=True)
        base_acc = float(subset["correct_at_open_probability_rule"].mean())
        flip_acc = float(subset["flipped_correct"].mean())
        delta = (flip_acc - base_acc) * 100.0
        deltas = block_bootstrap_delta(subset, rng, BOOTSTRAP_SAMPLES)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        results[label] = {
            "n_games": len(subset),
            "picks_changed": int(subset["flip"].sum()),
            "sample_blocks": int(subset.groupby(["season", "week"]).ngroups),
            "base_accuracy_pct": base_acc * 100.0,
            "flipped_accuracy_pct": flip_acc * 100.0,
            "effect_accuracy_points": delta,
            "interval_low": float(lo),
            "interval_high": float(hi),
            "probability_positive": probability_positive(deltas),
        }
    return results


def split_half_reliability(team_long: pd.DataFrame) -> dict:
    working = team_long.copy()
    working["half"] = np.where(working["season"] % 2 == 1, "odd", "even")
    half_means = working.groupby(["team", "half"])["z"].mean().reset_index()
    pivot = half_means.pivot(index="team", columns="half", values="z").dropna()
    r = float(pivot["odd"].corr(pivot["even"]))
    n = len(pivot)
    spearman_brown = 2 * r / (1 + r)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo = float(np.tanh(z - 1.96 * se))
    hi = float(np.tanh(z + 1.96 * se))
    return {
        "n_teams": int(n),
        "pearson_r_half": r,
        "pearson_r_half_interval_low": lo,
        "pearson_r_half_interval_high": hi,
        "spearman_brown_full_length": float(spearman_brown),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    smw_snapshot = latest_smw_publications_snapshot(REPO_ROOT)
    usable = load_usable_rows(smw_snapshot)
    coverage = coverage_by_season(usable)

    team_long = build_team_long(usable)
    lookup = build_trailing_lookup(team_long)

    graded = load_graded_games(REPO_ROOT)
    graded = attach_trailing_and_cover(graded, lookup)
    both_defined = graded[["home_trailing", "away_trailing"]].notna().all(axis=1).sum()

    table, edges = decile_table(graded)
    screen = production_screen(graded, edges)
    reliability = split_half_reliability(team_long)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "artifacts" / "tv_attention_screen" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_snapshot": str(smw_snapshot.relative_to(REPO_ROOT)),
        "opener_eval_artifact": OPENER_EVAL_ARTIFACT,
        "n_trailing": N_TRAILING,
        "flip_decile_threshold": FLIP_DECILE_THRESHOLD,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "graded_seasons": list(GRADED_SEASONS),
        "coverage_by_season_smw_usable_rows": coverage,
        "graded_games_total": len(graded),
        "graded_games_both_trailing_defined": int(both_defined),
        "decile_table": table.to_dict(orient="records"),
        "decile_bin_edges": edges.tolist(),
        "production_screen": screen,
        "split_half_reliability": reliability,
    }

    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    table.to_csv(output_dir / "decile_table.csv", index=False)
    graded.to_csv(output_dir / "per_game.csv", index=False)

    print(json.dumps(results, indent=2))
    print(f"artifact written to {output_dir}")


if __name__ == "__main__":
    main()
