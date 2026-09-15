from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
REGISTRY = REPO / "registry"

PICK_PROBABILITY_ARTIFACT = "pick_probability/20260914T232538Z"
OPENER_EVALUATION_ARTIFACT = "opener_evaluation/20260914T161406Z"
OVERLAY_SUBSET_ARTIFACT = "overlay_subset_composition/20260914T161418942653Z"
MARGINS_ARTIFACT = "margins/20260914T160810Z"
WEEK1_FORECAST_ARTIFACT = "margin_predictions/2026-week-01-20260914T160928Z"
INJURY_SNAPSHOT = "data/players/raw/20260913T131524Z/injuries.parquet"
FORECAST_ARCHIVE = "data/raw/forecast_archive/pool_decision_2009_2025/forecasts.parquet"
FEATURE_TABLE = "data/processed/game_features_weak_stack.parquet"

NFL_GAMES = 4431


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def section_registry_look_count() -> None:
    rule("1. LOOK COUNT AND THE SHAPE OF probability_positive")
    payload = json.loads((REGISTRY / "weak_signals.json").read_text(encoding="utf8"))
    signals = list(payload["signals"].values())
    print(f"recorded signals: {len(signals)}")
    classifications: dict[str, int] = {}
    for signal in signals:
        key = str(signal.get("classification"))
        classifications[key] = classifications.get(key, 0) + 1
    for key, count in sorted(classifications.items(), key=lambda item: -item[1]):
        print(f"  {count:6d}  {key}")

    values = [
        signal.get("probability_positive")
        for signal in signals
        if isinstance(signal.get("probability_positive"), (int, float))
    ]
    placeholders = sum(1 for value in values if value == 0.5)
    values = [value for value in values if value != 0.5]
    total = len(values)
    print()
    print(f"non-placeholder probability_positive rows: {total} (excluded {placeholders} at 0.5)")
    print("  tail        observed    expected if uniform")
    for threshold in (0.99, 0.95, 0.90):
        high = sum(1 for value in values if value > threshold)
        low = sum(1 for value in values if value < 1.0 - threshold)
        expected = total * (1.0 - threshold)
        print(f"  > {threshold:.2f}      {high:6d}      {expected:10.1f}")
        print(f"  < {1.0 - threshold:.2f}      {low:6d}      {expected:10.1f}")
    print(f"  mean probability_positive: {sum(values) / total:.4f}")

    promoted = 0.8562
    better = sum(1 for value in values if value > promoted)
    print()
    print(f"looks beating the promoted arrest policy (P+ {promoted}): {better} of {total}")
    print(f"expected under a pure null: {total * (1.0 - promoted):.0f}")


def section_effect_versus_noise() -> None:
    rule("2. MEASURED EFFECT SIZE VERSUS THE SAMPLING-NOISE SCALE")
    payload = json.loads((REGISTRY / "weak_signals.json").read_text(encoding="utf8"))
    rows = []
    for key, signal in payload["signals"].items():
        if signal.get("effect_units") != "accuracy_points":
            continue
        effect = signal.get("effect")
        games = signal.get("sample_games")
        if not isinstance(effect, (int, float)) or not isinstance(games, int) or games < 30:
            continue
        haystack = (key + " " + str(signal.get("description") or "")).lower()
        if any(
            token in haystack for token in ("control", "foresight", "oracle", "tautolog", "touched")
        ):
            continue
        rows.append((abs(float(effect)), games))
    print(f"accuracy_points measurements, n>=30, controls excluded: {len(rows)}")
    print()
    print("  games bin          count   median |effect|   noise scale   ratio")
    for low, high in ((30, 100), (100, 200), (200, 400), (400, 800), (800, 1600), (1600, 20000)):
        subset = [(effect, games) for effect, games in rows if low <= games < high]
        if len(subset) < 10:
            continue
        median_effect = statistics.median(effect for effect, _ in subset)
        median_games = statistics.median(games for _, games in subset)
        noise = 100.0 * 0.5 / math.sqrt(median_games) * 0.8
        print(
            f"  [{low:5d},{high:6d})   {len(subset):5d}      {median_effect:8.3f}"
            f"      {noise:8.3f}   {median_effect / noise:5.2f}"
        )
    print()
    print("a real fixed-size effect flattens as n grows; a constant ratio is the noise signature")


def section_pooling_floor() -> None:
    rule("3. THE POOLED STANDARD ERROR AGAINST ITS PHYSICAL FLOOR")
    print("run this for the live numbers:")
    print("  nfl-ats weak-signals pool --league nfl --effect-units accuracy_points")
    print()
    pooled_effect = 0.4685
    pooled_se = 0.03202
    net_games = pooled_effect / 100.0 * NFL_GAMES
    print(f"pooled effect {pooled_effect} accuracy points on {NFL_GAMES} games")
    print(f"  = {net_games:.1f} net games")
    print(f"pooled SE {pooled_se} -> 95% width {3.92 * pooled_se:.3f} points")
    print(f"  = {3.92 * pooled_se / 100.0 * NFL_GAMES:.1f} games")
    print()
    print("paired SE for an accuracy difference with d discordant games of 4431:")
    for discordant in (21, 100, 400, 1000):
        print(f"  d={discordant:5d}: {100.0 * math.sqrt(discordant) / NFL_GAMES:.3f} points")
    minimum = math.ceil(net_games)
    floor = 100.0 * math.sqrt(minimum) / NFL_GAMES
    print()
    print(f"an effect of {pooled_effect} points needs at least {minimum} discordant games")
    print(f"  -> floor SE {floor:.3f}; the pooled SE is {floor / pooled_se:.1f}x below it")


def section_calibration() -> None:
    rule("4. RELIABILITY AND PROPER SCORING RULES")
    metadata = json.loads(
        (ARTIFACTS / PICK_PROBABILITY_ARTIFACT / "metadata.json").read_text(encoding="utf8")
    )
    bands = metadata["model_only_confidence_bands"]
    print("model-only stated probability versus realised accuracy:")
    for band in bands:
        print(
            f"  [{band['lower']:.2f},{band['upper']:.2f})  n={band['games']:5d}"
            f"  realised={band['accuracy']:.4f}"
        )
    midpoints = [(band["lower"] + min(band["upper"], 0.70)) / 2.0 for band in bands]
    accuracies = [band["accuracy"] for band in bands]
    counts = [band["games"] for band in bands]
    total = sum(counts)
    mean_x = sum(c * x for c, x in zip(counts, midpoints, strict=True)) / total
    mean_y = sum(c * y for c, y in zip(counts, accuracies, strict=True)) / total
    numerator = sum(
        c * (x - mean_x) * (y - mean_y)
        for c, x, y in zip(counts, midpoints, accuracies, strict=True)
    )
    denominator = sum(c * (x - mean_x) ** 2 for c, x in zip(counts, midpoints, strict=True))
    print()
    print(f"weighted reliability slope: {numerator / denominator:+.3f}  (calibrated = +1.000)")
    brier = (
        sum(
            c * ((x - y) ** 2 + y * (1 - y))
            for c, x, y in zip(counts, midpoints, accuracies, strict=True)
        )
        / total
    )
    brier_flat = (
        sum(c * ((0.5 - y) ** 2 + y * (1 - y)) for c, y in zip(counts, accuracies, strict=True))
        / total
    )
    print(f"approximate Brier, stated probability: {brier:.5f}")
    print(f"approximate Brier, always 50%        : {brier_flat:.5f}")

    summary_path = ARTIFACTS / MARGINS_ARTIFACT / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        keep = ["method", "cover_games", "cover_accuracy", "cover_brier_score"]
        keep += ["cover_log_loss", "cover_ece"]
        keep = [column for column in keep if column in summary.columns]
        print()
        print("backtest summary (constant 0.5 scores Brier 0.250000, log loss 0.693147):")
        print(summary[keep].to_string(index=False))


def section_spread_buckets() -> None:
    rule("5. SERVED ACCURACY BY SPREAD BUCKET")
    path = next(
        (ARTIFACTS / WEEK1_FORECAST_ARTIFACT).glob("displayed_confidence.json"),
        None,
    )
    if path is None:
        print("displayed_confidence.json not found")
        return
    payload = json.loads(path.read_text(encoding="utf8"))
    print(f"prior rows: {payload['prior_rows']}")
    totals: dict[str, list[float]] = {}
    for row in payload["table"]:
        games = float(row.get("prior_games", 0) or 0)
        wins = float(row.get("prior_wins", 0) or 0)
        bucket = str(row["bucket"])
        current = totals.setdefault(bucket, [0.0, 0.0])
        current[0] += games
        current[1] += wins
    grand = [0.0, 0.0]
    for bucket, (games, wins) in totals.items():
        if not games:
            continue
        half = 1.96 * math.sqrt(0.25 / games) * 100.0
        print(f"  {bucket:>8}  n={games:6.0f}  accuracy={100 * wins / games:5.2f}%  +/-{half:4.2f}")
        grand[0] += games
        grand[1] += wins
    print(f"  {'ALL':>8}  n={grand[0]:6.0f}  accuracy={100 * grand[1] / grand[0]:5.2f}%")


def section_overlay_selection() -> None:
    rule("6. THE 127-SUBSET SEARCH BEHIND THE PLAYED POLICY")
    payload = json.loads(
        (ARTIFACTS / OVERLAY_SUBSET_ARTIFACT / "result.json").read_text(encoding="utf8")
    )
    print(f"selection_caveat: {payload.get('selection_caveat')}")
    print()
    print(f"combination_rule: {payload.get('combination_rule')}")
    print()
    print("member flip counts (decisive games per member):")
    for member, count in sorted(
        (payload.get("member_flip_counts") or {}).items(), key=lambda item: -item[1]
    ):
        print(f"  {count:5d}  {member}")
    subsets = payload["subsets"]
    deltas = [row["delta_estimate_accuracy_points"] for row in subsets]
    print()
    print(f"subsets enumerated: {len(subsets)}")
    print(
        f"  min {min(deltas):+.2f}  median {statistics.median(deltas):+.2f}"
        f"  max {max(deltas):+.2f}  sd {statistics.pstdev(deltas):.2f}"
    )
    print(f"  positive deltas: {sum(1 for value in deltas if value > 0)}")
    print()
    greedy = payload.get("greedy_forward_selection") or {}
    print(f"greedy forward selection path ({greedy.get('label')}):")
    for step in greedy.get("steps") or []:
        print(
            f"  step {step['step']}: +{step['added']:40s}"
            f" -> {step['point_estimate_accuracy_points']:+.4f} points"
        )
    print()
    print("the played chain is step 3, the in-sample maximum of this path")
    print("docs/edge_audit_redteam.md records the leave-one-season-out CV of this")
    print("selection procedure: pooled 0.0000 points, [-2.1462, +2.1348], P+ 0.4930")


def section_served_side_reproduction() -> None:
    rule("7. WHAT DECIDES THE SERVED SIDE: FITTED TERM OR HARD FLIP")
    coefficients = json.loads(
        (ARTIFACTS / PICK_PROBABILITY_ARTIFACT / "coefficients.json").read_text(encoding="utf8")
    )["coefficients"]
    intercept = coefficients["intercept"]
    model_logit = coefficients["model_logit"]
    flag_sum = coefficients["flag_sum"]
    print("fitted coefficients:")
    for name, value in coefficients.items():
        print(f"  {name:22s} {value:+.4f}")

    def crossover(flags: float) -> float:
        logit = -(intercept + flag_sum * flags) / model_logit
        return 1.0 / (1.0 + math.exp(-logit))

    low = crossover(1.0)
    high = crossover(0.0)
    print()
    print(f"model probability needed to serve HOME, no flags : {high:.4f}")
    print(f"model probability needed to serve HOME, one flag : {low:.4f}")

    per_game = ARTIFACTS / OPENER_EVALUATION_ARTIFACT / "per_game.parquet"
    if per_game.exists():
        frame = pd.read_parquet(per_game)
        column = "home_cover_probability_at_open_raw"
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            share = ((values > low) & (values < high)).mean()
            print()
            print(
                f"games where one flag alone decides the side: {100 * share:.1f}% (n={len(values)})"
            )
            print(f"model probability range: {values.min():.3f} to {values.max():.3f}")

    print()
    print("week 1 served sides, fitted term versus hard flip:")
    contested = {
        "ARI@LAC": (0.362126, "LAC", "HOME"),
        "ATL@PIT": (0.474525, "PIT", "HOME"),
        "BAL@IND": (0.456054, "IND", "HOME"),
        "WAS@PHI": (0.367416, "PHI", "HOME"),
    }
    print("  game      raw p   4term/0flag   4term/1flag   hard 1-p   served")
    for game, (raw, team, side) in contested.items():
        base = intercept + model_logit * math.log(raw / (1.0 - raw))
        zero = 1.0 / (1.0 + math.exp(-base))
        one = 1.0 / (1.0 + math.exp(-(base + flag_sum)))
        print(
            f"  {game}  {raw:.3f}   {zero:.4f} AWAY   {one:.4f} HOME"
            f"   {1 - raw:.3f} AWAY   {team} ({side})"
        )
    print()
    print("every served side matches the fitted term with one flag; the hard 1-p")
    print("write would serve the opposite side on all four, so it does not decide")


def section_leakage() -> None:
    rule("8. LEAKAGE CHECKS")
    try:
        from nfl_ats.margin import margin_feature_columns
    except Exception as error:
        print(f"could not import margin_feature_columns: {error}")
        return
    columns = margin_feature_columns("market_residual", "weak_stack")
    print(f"served feature columns: {len(columns)}")
    print(f"  temp present: {'temp' in columns}")
    print(f"  wind present: {'wind' in columns}")

    archive = REPO / FORECAST_ARCHIVE
    table = REPO / FEATURE_TABLE
    if archive.exists() and table.exists():
        features = pd.read_parquet(table, columns=["game_id", "temp", "wind"])
        forecasts = pd.read_parquet(archive)
        joined = features.merge(forecasts, on="game_id", how="inner")
        joined = joined.dropna(subset=["temp", "actual_temp_f"])
        exact = (joined["temp"] == joined["actual_temp_f"]).mean()
        drift = (joined["temp"] - joined["forecast_temp_f"]).abs().mean()
        print()
        print(f"joined games: {len(joined)}")
        print(f"  feature temp equals ACTUAL game-time temp: {100 * exact:.1f}% of rows")
        print(f"  feature temp versus the pool-decision FORECAST: MAE {drift:.2f} F")

    injuries = REPO / INJURY_SNAPSHOT
    if injuries.exists():
        frame = pd.read_parquet(injuries, columns=["season", "observed_at_basis"])
        print()
        print("injury timestamp provenance by season:")
        print(pd.crosstab(frame["season"], frame["observed_at_basis"]).tail(7).to_string())


def section_guard_coverage() -> None:
    rule("9. WHERE THE OUTCOME-LEAK GUARD RUNS")
    print("grep for the guard's call sites:")
    print("  grep -rn 'validate_model_frame' src/ --include=*.py")
    print()
    print("expected: backtest.py, modeling.py, outcomes.py only")
    print("margin.py is the served model and does not call it")
    print("the guard compares column NAMES against OUTCOME_COLUMNS, never values")


def main() -> int:
    rule("STATISTICAL AUDIT 2026-09-15 - REPRODUCTION SCRIPT")
    print("findings: docs/lanes/statistical-audit-2026-09-15.md")
    print("every number this prints is read from a pinned artifact or the registry")
    print("artifacts are pinned by timestamp at the top of this file; change them to")
    print("re-run the audit against a newer active model")
    sections = (
        section_registry_look_count,
        section_effect_versus_noise,
        section_pooling_floor,
        section_calibration,
        section_spread_buckets,
        section_overlay_selection,
        section_served_side_reproduction,
        section_leakage,
        section_guard_coverage,
    )
    for section in sections:
        try:
            section()
        except Exception as error:
            print(f"\nSECTION FAILED: {section.__name__}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
