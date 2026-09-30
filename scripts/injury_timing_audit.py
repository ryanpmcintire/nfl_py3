from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_loso_upstream import SEASONS, Replay

from nfl_ats.nfl_week import week_cycle_sunday

OUTPUT = Path("tests/scratch/codex/injury_timing_audit")
EASTERN = ZoneInfo("America/New_York")
FEATURES = Path("data/processed/game_features_weak_stack.parquet")
CLOCKS = Path("tests/scratch/codex/lead89_unit2/input_clock_audit.parquet")
INJURY_COLUMNS = (
    "diff_injury_offense_unavailability",
    "diff_injury_defense_unavailability",
    "diff_injury_special_teams_unavailability",
    "diff_injury_offensive_line_unavailability",
    "diff_injury_skill_unavailability",
    "diff_injury_front_unavailability",
    "diff_injury_secondary_unavailability",
    "diff_injury_skill_epa_value_lost",
    "diff_injury_defense_disruption_value_lost",
)
QB_INJURY_COLUMNS = ("diff_qb_expected_epa_per_dropback", "diff_qb_start_probability")
ZEROED = INJURY_COLUMNS + QB_INJURY_COLUMNS
DRAWS = 2000
SEED = 8902


def tuesday_noon_utc(kickoff: pd.Timestamp) -> pd.Timestamp:
    eastern = kickoff.tz_convert(EASTERN)
    tuesday = week_cycle_sunday(eastern.date()) - timedelta(days=5)
    return pd.Timestamp(datetime.combine(tuesday, time(12), tzinfo=EASTERN)).tz_convert("UTC")


def describe(hours: pd.Series) -> dict:
    values = hours.dropna()
    if values.empty:
        return {"n": 0}
    quantiles = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(2).tolist()
    return {"n": len(values), "p05_p25_p50_p75_p95": quantiles}


def census() -> dict:
    features = pd.read_parquet(FEATURES)
    features = features.loc[features.game_type.eq("REG") & features.season.between(2020, 2025)]
    features = features.copy()
    features["kickoff"] = pd.to_datetime(features.kickoff, utc=True)
    clocks = pd.read_parquet(CLOCKS)
    tuesday = clocks.loc[clocks.stage.eq("tuesday"), ["game_id", "decision_at"]]
    frame = features.merge(tuesday, on="game_id", how="inner", validate="one_to_one")
    frame["decision_at"] = pd.to_datetime(frame.decision_at, utc=True)
    frame["noon_et"] = frame.kickoff.map(tuesday_noon_utc)
    for side in ("home", "away"):
        frame[f"{side}_obs"] = pd.to_datetime(frame[f"{side}_injury_observed_at"], utc=True)
    frame["obs"] = frame[["home_obs", "away_obs"]].max(axis=1)
    frame["obs_min"] = frame[["home_obs", "away_obs"]].min(axis=1)
    frame["missing"] = frame.obs.isna() | frame.obs_min.isna()
    frame["lag_tuesday_h"] = (frame.obs - frame.decision_at).dt.total_seconds() / 3600
    frame["lag_noon_h"] = (frame.obs - frame.noon_et).dt.total_seconds() / 3600
    frame["lead_kickoff_h"] = (frame.kickoff - frame.obs).dt.total_seconds() / 3600
    frame["builder_cutoff"] = frame.kickoff - pd.Timedelta(hours=24)
    frame["after_builder_cutoff"] = frame.obs.gt(frame.builder_cutoff)
    late = frame.lag_tuesday_h.gt(0)
    basis = pd.concat([frame.home_injury_observed_at_basis, frame.away_injury_observed_at_basis])
    proxy = frame.home_injury_observed_at_is_proxy.fillna(False).astype(bool) | (
        frame.away_injury_observed_at_is_proxy.fillna(False).astype(bool)
    )
    decision_hour = frame.decision_at.dt.tz_convert(EASTERN).dt.hour
    result = {
        "games": len(frame),
        "missing_observation": int(frame.missing.sum()),
        "late_vs_tuesday_decision": int(late.sum()),
        "late_vs_tuesday_noon_et": int(frame.lag_noon_h.gt(0).sum()),
        "after_builder_kickoff_minus_24h_cutoff": int(frame.after_builder_cutoff.sum()),
        "after_kickoff": int(frame.obs.gt(frame.kickoff).sum()),
        "any_proxy_side": int(proxy.sum()),
        "basis_counts_by_side": basis.fillna("missing").value_counts().to_dict(),
        "lag_hours_after_tuesday_decision": describe(frame.lag_tuesday_h),
        "lag_hours_after_tuesday_noon_et": describe(frame.lag_noon_h),
        "hours_before_kickoff_observed": describe(frame.lead_kickoff_h),
        "tuesday_decision_hour_et_min_max": [int(decision_hour.min()), int(decision_hour.max())],
        "by_season_late": {
            str(k): [int(v.sum()), len(v)] for k, v in late.groupby(frame.season, sort=True)
        },
        "nonzero_share_2020_2025": frame[list(ZEROED)].fillna(0).ne(0).mean().round(4).to_dict(),
    }
    live = pd.read_parquet(FEATURES)
    live = live.loc[live.season.eq(2026)]
    nonzero = live[list(INJURY_COLUMNS)].fillna(0).ne(0).mean(axis=1)
    result["live_2026_nonzero_injury_share_by_week"] = {
        str(k): float(v) for k, v in nonzero.groupby(live.week).mean().round(3).items()
    }
    live4 = live.loc[live.week.eq(4)]
    instants = pd.concat([live4.home_injury_observed_at, live4.away_injury_observed_at])
    result["live_2026_week4_observed_at"] = {
        "distinct_instants": sorted({str(x) for x in instants}),
        "bases": live4.home_injury_observed_at_basis.value_counts().to_dict(),
    }
    keep = [
        "game_id",
        "season",
        "week",
        "kickoff",
        "decision_at",
        "noon_et",
        "obs",
        "lag_tuesday_h",
        "lag_noon_h",
        "lead_kickoff_h",
        "missing",
    ]
    frame[keep].to_parquet(OUTPUT / "clock_census.parquet", index=False)
    return result


def per_arm(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_parquet(path / "predictions.parquet")
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return frame.loc[frame.home_covered.notna()].reset_index(drop=True), meta


def rows(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.column_stack(
        (
            ((probability >= 0.5) == (target == 1)).astype(float),
            -(target * np.log(p) + (1 - target) * np.log1p(-p)),
            (probability - target) ** 2,
        )
    )


def block_sums(frame: pd.DataFrame, arms: dict[str, np.ndarray]) -> tuple[dict, np.ndarray]:
    key = (frame.season * 100 + frame.week).to_numpy()
    blocks, inverse = np.unique(key, return_inverse=True)
    out = {}
    for name, prob in arms.items():
        metric = rows(frame.home_covered.to_numpy(), prob)
        summed = np.zeros((len(blocks), 3))
        np.add.at(summed, inverse, metric)
        out[name] = summed
    count = np.bincount(inverse, minlength=len(blocks)).astype(float)
    return out, count


def paired(frame: pd.DataFrame, base: np.ndarray, cand: np.ndarray, rng) -> dict:
    sums, count = block_sums(frame, {"b": base, "c": cand})
    delta = sums["b"] - sums["c"]
    n = len(count)
    idx = rng.integers(0, n, size=(DRAWS, n))
    total = count[idx].sum(axis=1)
    draws = delta[idx].sum(axis=1) / total[:, None]
    point = delta.sum(axis=0) / count.sum()
    out = {}
    for j, name in enumerate(("accuracy_points", "log_loss", "brier")):
        sign = 1.0 if name == "accuracy_points" else -1.0
        gain = sign * draws[:, j]
        positive = (gain > 0).mean() + 0.5 * (gain == 0).mean()
        scale = 100.0 if name == "accuracy_points" else 1.0
        out[name] = {
            "baseline_minus_candidate": float(point[j] * scale),
            "interval95": [float(x) for x in np.quantile(draws[:, j] * scale, [0.025, 0.975])],
            "probability_baseline_better": float(positive),
        }
    out["games"] = int(count.sum())
    out["blocks"] = int(n)
    return out


def summary(target: np.ndarray, probability: np.ndarray) -> dict:
    metric = rows(target, probability)
    wins = int(metric[:, 0].sum())
    return {
        "wins": wins,
        "losses": int(len(target) - wins),
        "accuracy": float(metric[:, 0].mean()),
        "log_loss": float(metric[:, 1].mean()),
        "brier": float(metric[:, 2].mean()),
        "games": len(target),
    }


def reliability(frame: pd.DataFrame, probability: np.ndarray) -> list[dict]:
    pick = np.maximum(probability, 1 - probability)
    win = ((probability >= 0.5) == (frame.home_covered.to_numpy() == 1)).astype(float)
    edges = np.quantile(pick, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    band = np.clip(np.searchsorted(edges, pick, side="right") - 1, 0, 4)
    return [
        {
            "band": int(b),
            "games": int((band == b).sum()),
            "mean_pick_probability": float(pick[band == b].mean()),
            "observed_win_rate": float(win[band == b].mean()),
        }
        for b in range(5)
    ]


def arm_report(frame: pd.DataFrame, meta: dict | None, column: str) -> dict:
    target = frame.home_covered.to_numpy()
    probability = frame[column].to_numpy()
    by_season = {}
    for season in SEASONS:
        mask = frame.season.eq(season).to_numpy()
        by_season[str(season)] = summary(target[mask], probability[mask])
        if meta is not None and column == "four_term_probability":
            fold = meta["calibration_folds"][str(season)]
            by_season[str(season)]["in_sample"] = fold["training_metrics"]
            by_season[str(season)]["coefficients"] = fold["coefficients"]
    report = {
        "pooled": summary(target, probability),
        "by_season": by_season,
        "reliability": reliability(frame, probability),
    }
    if meta is not None and column == "four_term_probability":
        report["coefficient_stability"] = meta["coefficient_stability"]
    return report


def run_arm(replay: Replay, name: str, zero: bool) -> Path:
    replay.raw_cache.clear()
    replay.input_cache.clear()
    replay.fold_audits.clear()
    if zero:
        for column in ZEROED:
            replay.features[column] = 0.0
            replay.games[column] = 0.0
    output = OUTPUT / name
    if not output.exists():
        replay.run(output)
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    clock = census()
    (OUTPUT / "clock_census.json").write_text(json.dumps(clock, indent=2, default=str))
    print("census done", flush=True)
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    with threadpool_limits(limits=2):
        replay = Replay(
            Path("artifacts/pick_probability/20260929T192747Z"),
            FEATURES,
            Path("data/raw/20260929T191306Z/schedules.parquet"),
            Path("data/pbp/raw/20260929T191306Z"),
        )
        baseline_path = run_arm(replay, "baseline", zero=False)
        candidate_path = run_arm(replay, "candidate", zero=True)
    base, base_meta = per_arm(baseline_path)
    cand, cand_meta = per_arm(candidate_path)
    pair = base.merge(
        cand[["game_id", "four_term_probability", "model_probability"]],
        on="game_id",
        suffixes=("", "_cand"),
    )
    reference = pd.read_parquet(
        "artifacts/loso_upstream/20260929T235904133782Z/predictions.parquet"
    )
    reference = reference.set_index("game_id").four_term_probability
    saved = reference.reindex(pair.game_id).to_numpy()
    reproduced = float(np.abs(pair.four_term_probability.to_numpy() - saved).max())
    rng = np.random.default_rng(SEED)
    b4, c4 = pair.four_term_probability.to_numpy(), pair.four_term_probability_cand.to_numpy()
    bm, cm = pair.model_probability.to_numpy(), pair.model_probability_cand.to_numpy()
    by_season = {}
    for season in SEASONS:
        sub = pair.loc[pair.season.eq(season)]
        by_season[str(season)] = paired(
            sub,
            sub.four_term_probability.to_numpy(),
            sub.four_term_probability_cand.to_numpy(),
            rng,
        )
    result = {
        "zeroed_columns": list(ZEROED),
        "baseline_reproduces_saved_upstream_max_abs_diff": reproduced,
        "clock_census": clock,
        "arms": {
            "baseline_four_term": arm_report(base, base_meta, "four_term_probability"),
            "candidate_four_term": arm_report(cand, cand_meta, "four_term_probability"),
            "baseline_model_only": arm_report(base, None, "model_probability"),
            "candidate_model_only": arm_report(cand, None, "model_probability"),
            "frozen_served_evaluation": arm_report(
                base, None, "served_evaluation_home_probability"
            ),
        },
        "market_even_log_loss": float(np.log(2)),
        "paired_pooled_four_term": paired(pair, b4, c4, rng),
        "paired_pooled_model_only": paired(pair, bm, cm, rng),
        "paired_by_season_four_term": by_season,
        "decisive_games": len(pair),
        "pushes": int(base_meta["pushes"]),
        "looks": 151,
        "bootstrap": {"draws": DRAWS, "seed": SEED, "unit": "season-week block"},
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result["paired_pooled_four_term"], indent=2))
    print("reproduced max abs diff", reproduced)
    for name, arm in result["arms"].items():
        print(name, arm["pooled"])


if __name__ == "__main__":
    main()
