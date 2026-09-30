import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import latent_ratings_on_production as lrp

from nfl_ats import clv
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.margin import _MARGIN_PROFILE_FEATURE_SETS, margin_feature_columns

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "game_features_weak_stack.parquet"
MARKET_ROOT = ROOT / "data" / "market" / "raw"
BASELINE_DIR = ROOT / "artifacts" / "opener_evaluation" / "20260929T192743Z"
OUT_DIR = ROOT / "artifacts" / "mod23_unit2"
UNIT4 = ROOT / "tests/scratch/codex/mod22_unit4/20260929T213636Z/per_game.parquet"
UNIT5 = ROOT / "tests/scratch/codex/mod22_unit5/20260929T214747Z/per_game.parquet"
PROFILE = "weak_stack"
SET_NAME = _MARGIN_PROFILE_FEATURE_SETS[PROFILE][1]
BASE_ALPHA = 10.0
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 20260930
BASE_CONFIG = {
    "feature_profile": PROFILE,
    "regressor": "ridge",
    "target": "market_residual",
    "probability_method": "gaussian_median",
    "calibration_method": "none",
    "ridge_alpha": BASE_ALPHA,
}
FULL_COLUMNS = tuple(margin_feature_columns("market_residual", PROFILE))
ARMS = {
    "full_strength": ("diff_full_strength_total",),
    "lineup_total": ("diff_lineup_total_ev",),
    "divergence": ("diff_divergence_ev",),
    "lineup_and_divergence": ("diff_lineup_total_ev", "diff_divergence_ev"),
    "qb_out": ("qb_out_diff",),
    "qb_quality_loss": ("qb_quality_loss_diff",),
}


def evidenced_index():
    injuries = pd.read_parquet(lrp.PLAYER_SNAPSHOT / "injuries.parquet")
    injuries = injuries.loc[
        injuries["game_type"].eq("REG") & injuries["season"].astype(int).ge(lrp.FEATURE_SEASONS[0])
    ].copy()
    observed = pd.to_datetime(injuries["effective_observed_at"], utc=True, errors="coerce")
    modified = pd.to_datetime(injuries["date_modified"], utc=True, errors="coerce")
    evidenced = (
        injuries["observed_at_basis"].eq("date_modified")
        & injuries["observed_at_is_proxy"].eq(False)
        & modified.notna()
        & observed.eq(modified)
    )
    injuries["effective_observed_at"] = modified.where(evidenced)
    print("injury rows", len(injuries), "evidenced", int(evidenced.sum()), flush=True)
    lookup = lrp.availability_rate_lookup(pd.read_parquet(lrp.AVAILABILITY_RATES_PATH))
    severities = []
    for row in injuries.itertuples(index=False):
        unavailable, _ = lrp.resolve_unavailability(
            lookup,
            target_season=int(row.season),
            report_status=row.report_status,
            practice_status=row.practice_status,
            position=row.position,
        )
        severities.append(float(unavailable))
    injuries["unavailability"] = severities
    injuries = injuries.sort_values("effective_observed_at")
    index = {}
    for row in injuries.itertuples(index=False):
        index.setdefault((int(row.season), int(row.week), str(row.team)), []).append(
            (str(row.gsis_id), row.effective_observed_at, float(row.unavailability))
        )
    return index, len(injuries)


def build_rating_terms():
    lrp.FEATURE_TABLE = OUT_DIR / "expected_lineup_ratings_evidenced.parquet"
    lrp._visible_injury_index = evidenced_index
    table = lrp.build_feature_table(OUT_DIR)
    out = table[["game_id"]].copy()
    out["diff_full_strength_total"] = (
        table["home_full_strength_total"] - table["away_full_strength_total"]
    )
    out["diff_lineup_total_ev"] = table["diff_lineup_total"]
    out["diff_divergence_ev"] = table["diff_divergence"]
    return out


def attach_terms(features):
    ratings = build_rating_terms()
    unit4 = pd.read_parquet(UNIT4)[["game_id", "qb_out_diff"]]
    unit5 = pd.read_parquet(UNIT5)[["game_id", "qb_quality_loss_diff"]]
    merged = features.merge(ratings, on="game_id", how="left")
    merged = merged.merge(unit4, on="game_id", how="left")
    merged = merged.merge(unit5, on="game_id", how="left")
    coverage = {}
    for column in sorted({c for cols in ARMS.values() for c in cols}):
        coverage[column] = {
            "rows_with_value": int(merged[column].notna().sum()),
            "nonzero": int((merged[column].fillna(0.0) != 0).sum()),
        }
        merged[column] = merged[column].fillna(0.0)
    return merged, coverage


def run_arm(features, extra):
    original = FEATURE_SETS[SET_NAME]
    FEATURE_SETS[SET_NAME] = tuple(FULL_COLUMNS) + tuple(extra)
    try:
        config = {**BASE_CONFIG, "mod23_variant": json.dumps(list(extra))}
        return clv.opener_pick_evaluation(
            MARKET_ROOT,
            features,
            active_model_config=config,
            min_train_games=DEFAULT_MIN_TRAIN_GAMES,
        )
    finally:
        FEATURE_SETS[SET_NAME] = original


def decisive(scored):
    frame = scored.loc[scored["correct_at_open_probability_rule"].notna()].copy()
    frame["y"] = (frame["margin_vs_open"] > 0).astype(float)
    frame["p"] = frame["home_cover_probability_at_open"].clip(1e-6, 1 - 1e-6)
    frame["ll"] = -(frame["y"] * np.log(frame["p"]) + (1 - frame["y"]) * np.log(1 - frame["p"]))
    frame["brier"] = (frame["p"] - frame["y"]) ** 2
    return frame.set_index("game_id")


def bootstrap_diff(diff_frame):
    blocks = [g["d"].to_numpy() for _, g in diff_frame.groupby(["season", "week"])]
    sums = np.array([b.sum() for b in blocks])
    sizes = np.array([len(b) for b in blocks])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.array(
        [
            100.0 * sums[idx].sum() / sizes[idx].sum()
            for idx in (rng.integers(0, len(blocks), len(blocks)) for _ in range(BOOTSTRAP_SAMPLES))
        ]
    )
    return {
        "low": float(np.percentile(draws, 2.5)),
        "high": float(np.percentile(draws, 97.5)),
        "probability_positive": float((draws > 0).mean()),
        "blocks": len(blocks),
    }


def summarize(name, arm, base):
    joined = arm.join(base, rsuffix="_b", how="inner")
    joined["d"] = (
        joined["correct_at_open_probability_rule"] - joined["correct_at_open_probability_rule_b"]
    )
    wins = int(arm["correct_at_open_probability_rule"].sum())
    n = len(arm)
    result = {
        "arm": name,
        "record": f"{wins}-{n - wins}",
        "games": n,
        "accuracy": wins / n,
        "diff_points": 100.0 * float(joined["d"].mean()),
        "bootstrap": bootstrap_diff(joined),
        "log_loss": float(arm["ll"].mean()),
        "log_loss_base": float(base["ll"].mean()),
        "log_loss_even": float(np.log(2)),
        "brier": float(arm["brier"].mean()),
        "brier_base": float(base["brier"].mean()),
        "brier_even": 0.25,
        "per_season": {},
    }
    for season, g in joined.groupby("season"):
        result["per_season"][int(season)] = {
            "n": len(g),
            "diff_points": 100.0 * float(g["d"].mean()),
            "ll_diff": float((g["ll"] - g["ll_b"]).mean()),
        }
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(FEATURES)
    print("feature sha", hashlib.sha256(FEATURES.read_bytes()).hexdigest())
    features, coverage = attach_terms(features)
    base = decisive(run_arm(features, ()))
    art = (
        pd.read_parquet(BASELINE_DIR / "per_game.parquet")
        .set_index("game_id")["correct_at_open_probability_rule"]
        .dropna()
    )
    mine = base["correct_at_open_probability_rule"]
    print("baseline", int(mine.sum()), len(mine), "artifact", int(art.sum()), len(art), flush=True)
    if int(mine.sum()) != int(art.sum()) or len(mine) != len(art):
        print("REPRODUCTION FAILED")
        return
    summaries = {}
    for name, extra in ARMS.items():
        arm = decisive(run_arm(features, extra))
        arm.to_parquet(OUT_DIR / f"{name}_per_game.parquet")
        summaries[name] = summarize(name, arm, base)
        summaries[name]["columns"] = list(extra)
        print(name, summaries[name]["record"], summaries[name]["diff_points"], flush=True)
    report = {"coverage": coverage, "summaries": summaries, "looks": len(ARMS)}
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


main()
