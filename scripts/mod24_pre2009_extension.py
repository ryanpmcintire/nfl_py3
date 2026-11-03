import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats import clv
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.features import build_game_features
from nfl_ats.margin import (
    _MARGIN_PROFILE_FEATURE_SETS,
    _PROFILE_SUPPRESSED_MISSING_INDICATORS,
    MarginModel,
    _target_values,
    make_margin_estimator,
    margin_feature_columns,
    resolve_feature_groups,
)
from nfl_ats.modeling import regular_season_rows
from nfl_ats.pbp import canonicalize_pbp, enrich_with_pbp_features, load_pbp_snapshot
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root

ROOT = Path(__file__).resolve().parents[1]
NEW_RAW = ROOT / "data" / "raw" / "nflverse_pbp_full" / "20261103T155706Z"
OLD_SNAPSHOT = ROOT / "data" / "raw" / "20261027T220410Z"
OLD_PBP = ROOT / "data" / "pbp" / "raw" / "20261027T220410Z"
SERVED_FEATURES = ROOT / "data" / "processed" / "game_features_weak_stack.parquet"
MARKET_ROOT = ROOT / "data" / "market" / "raw"
OUT_DIR = ROOT / "artifacts" / "mod24_pre2009"
TABLE = OUT_DIR / "features_1999_2025.parquet"
PROFILE = "weak_stack"
SET_NAME = _MARGIN_PROFILE_FEATURE_SETS[PROFILE][1]
BASE_ALPHA = 10.0
SPAN = 8
MIN_PERIODS = 3
RETENTION = 0.67
GRAPH_HALF_LIFE = 8.0
GRAPH_RIDGE = 8.0
GRAPH_MIN_GAMES = 16
OPP_HALF_LIFE = 16.0
OPP_RIDGE = 10.0
OPP_MIN_GAMES = 64
PRE_SEASONS = tuple(range(1999, 2009))
SCORE_SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
COVERAGE_RATIO = 0.95
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 20261103
BASE_CONFIG = {
    "feature_profile": PROFILE,
    "regressor": "ridge",
    "target": "market_residual",
    "probability_method": "gaussian_median",
    "calibration_method": "none",
}
FULL_COLUMNS = tuple(margin_feature_columns("market_residual", PROFILE))
COLUMN_FAMILY = dict(zip(FULL_COLUMNS, resolve_feature_groups(FULL_COLUMNS), strict=True))


def load_inputs(cutoff=None):
    old_schedules = pd.read_parquet(OLD_SNAPSHOT / "schedules.parquet")
    new_schedules = pd.read_parquet(NEW_RAW / "schedules_1999_2008.parquet")
    old_stats = pd.read_parquet(OLD_SNAPSHOT / "team_stats.parquet")
    new_stats = pd.read_parquet(NEW_RAW / "team_stats_1999_2008.parquet")
    new_stats = new_stats.dropna(subset=["game_id", "team"])
    counts = new_stats.groupby("game_id").size()
    incomplete = set(counts.index[counts.ne(2)]) | (
        set(new_schedules["game_id"]) - set(counts.index)
    )
    new_schedules = new_schedules.loc[~new_schedules["game_id"].isin(incomplete)]
    schedules = pd.concat([new_schedules, old_schedules], ignore_index=True)
    schedules = schedules.loc[schedules["season"].le(2025)].reset_index(drop=True)
    new_stats = new_stats.loc[~new_stats["game_id"].isin(incomplete)]
    stats = pd.concat([new_stats, old_stats], ignore_index=True)
    stats = stats.loc[stats["season"].le(2025)].reset_index(drop=True)
    new_pbp = pd.concat(
        [
            canonicalize_pbp(pd.read_parquet(NEW_RAW / f"pbp_{s}.parquet"), season=s)
            for s in PRE_SEASONS
        ],
        ignore_index=True,
    )
    old_pbp = load_pbp_snapshot(pbp_snapshot_from_root(OLD_PBP))
    old_pbp = old_pbp.loc[old_pbp["season"].le(2025)]
    new_pbp = new_pbp.loc[~new_pbp["game_id"].isin(incomplete)]
    pbp = pd.concat([new_pbp, old_pbp], ignore_index=True)
    if cutoff is not None:
        dates = pd.to_datetime(schedules["gameday"])
        future = dates.ge(cutoff)
        future_games = set(schedules.loc[future, "game_id"])
        for column in ("home_score", "away_score", "result", "total"):
            schedules.loc[future, column] = np.nan
        stats = drop_future_stats(stats, schedules, future)
        pbp = pbp.loc[~pbp["game_id"].isin(future_games)]
    return schedules, stats, pbp


def drop_future_stats(stats, schedules, future):
    keys = schedules.loc[future, ["season", "week"]].drop_duplicates()
    first = keys.sort_values(["season", "week"]).iloc[0]
    later = stats["season"].gt(first["season"]) | (
        stats["season"].eq(first["season"]) & stats["week"].ge(first["week"])
    )
    return stats.loc[~later]


def build_table(cutoff=None):
    schedules, stats, pbp = load_inputs(cutoff)
    games = build_game_features(
        schedules,
        stats,
        span=SPAN,
        min_periods=MIN_PERIODS,
        offseason_retention=RETENTION,
        graph_half_life_weeks=GRAPH_HALF_LIFE,
        graph_ridge_alpha=GRAPH_RIDGE,
        graph_min_games=GRAPH_MIN_GAMES,
        include_postseason=True,
    )
    return enrich_with_pbp_features(
        games,
        pbp,
        span=SPAN,
        min_periods=MIN_PERIODS,
        offseason_retention=RETENTION,
        opponent_half_life_weeks=OPP_HALF_LIFE,
        opponent_ridge_alpha=OPP_RIDGE,
        opponent_min_team_games=OPP_MIN_GAMES,
    )


def assert_pregame(table):
    cutoff = pd.Timestamp("2007-11-11")
    truncated = build_table(cutoff)
    in_week = pd.to_datetime(table["gameday"]).between(cutoff, cutoff + pd.Timedelta(days=6))
    ids = table.loc[in_week & table["game_type"].eq("REG"), "game_id"]
    columns = [c for c in FULL_COLUMNS if c in table.columns and c in truncated.columns]
    a = table.set_index("game_id").loc[ids, columns].astype(float)
    b = truncated.set_index("game_id").loc[ids, columns].astype(float)
    same = np.isclose(a.to_numpy(), b.to_numpy(), equal_nan=True, rtol=0, atol=1e-9)
    if not same.all():
        bad = [c for c, ok in zip(columns, same.all(axis=0), strict=True) if not ok]
        raise AssertionError(f"future information reaches pregame features: {bad}")
    print("pregame truncation check passed", len(ids), "games", len(columns), "columns")


def stage_build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    assert_pregame(table)
    table.to_parquet(TABLE)
    stage_coverage()


def stage_coverage():
    table = pd.read_parquet(TABLE)
    reg = regular_season_rows(table)
    rows = []
    for column in FULL_COLUMNS:
        if column not in table.columns:
            rows.append(
                {
                    "column": column,
                    "family": COLUMN_FAMILY[column],
                    "present": False,
                    "buildable": False,
                }
            )
            continue
        by_season = reg.groupby("season")[column].apply(lambda x: float(x.notna().mean()))
        post = float(by_season.loc[by_season.index >= 2009].mean())
        pre_min = float(
            by_season.loc[(by_season.index < 2009) & (by_season.index > PRE_SEASONS[0])].min()
        )
        rows.append(
            {
                "column": column,
                "family": COLUMN_FAMILY[column],
                "present": True,
                "coverage_2009plus": post,
                "coverage_pre2009_min": pre_min,
                "buildable": bool(pre_min >= COVERAGE_RATIO * post and post > 0),
            }
        )
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT_DIR / "coverage.csv", index=False)
    print(cov.groupby("family")["buildable"].agg(["sum", "count"]))
    print("buildable", int(cov["buildable"].sum()), "of", len(cov))


def reduced_columns():
    cov = pd.read_csv(OUT_DIR / "coverage.csv")
    return tuple(cov.loc[cov["buildable"].astype(bool), "column"])


def run_eval(features, columns, tag):
    original = FEATURE_SETS[SET_NAME]
    FEATURE_SETS[SET_NAME] = columns
    try:
        config = {**BASE_CONFIG, "ridge_alpha": BASE_ALPHA, "mod24_arm": tag}
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


def block_bootstrap(joined, column):
    blocks = [g[column].to_numpy() for _, g in joined.groupby(["season", "week"])]
    sums = np.array([b.sum() for b in blocks])
    sizes = np.array([len(b) for b in blocks])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_SAMPLES)
    for i in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(blocks), len(blocks))
        draws[i] = sums[idx].sum() / sizes[idx].sum()
    return draws


def record(frame):
    wins = int(frame["correct_at_open_probability_rule"].sum())
    return f"{wins}-{len(frame) - wins}"


def compare(name, arm, ref):
    common = arm.index.intersection(ref.index)
    a, r = arm.loc[common], ref.loc[common]
    joined = pd.DataFrame(
        {
            "season": a["season"],
            "week": a["week"],
            "acc": 100.0
            * (a["correct_at_open_probability_rule"] - r["correct_at_open_probability_rule"]),
            "ll": r["ll"] - a["ll"],
            "brier": r["brier"] - a["brier"],
        }
    )
    out = {
        "comparison": name,
        "n": len(common),
        "arm_record": record(a),
        "ref_record": record(r),
        "arm_acc": float(a["correct_at_open_probability_rule"].mean()),
        "ref_acc": float(r["correct_at_open_probability_rule"].mean()),
        "arm_ll": float(a["ll"].mean()),
        "ref_ll": float(r["ll"].mean()),
        "arm_brier": float(a["brier"].mean()),
        "ref_brier": float(r["brier"].mean()),
        "per_season": {},
    }
    for column, label in (("acc", "acc_points"), ("ll", "ll_gain"), ("brier", "brier_gain")):
        draws = block_bootstrap(joined, column)
        out[label] = {
            "mean": float(joined[column].mean()),
            "low": float(np.percentile(draws, 2.5)),
            "high": float(np.percentile(draws, 97.5)),
            "probability_positive": float((draws > 0).mean()),
        }
    for season, g in joined.groupby("season"):
        out["per_season"][int(season)] = {
            "n": len(g),
            "acc_points": float(g["acc"].mean()),
            "ll_gain": float(g["ll"].mean()),
        }
    return out


def in_sample(frame, columns):
    suppressed = tuple(
        c for c in _PROFILE_SUPPRESSED_MISSING_INDICATORS.get(PROFILE, ()) if c in columns
    )
    rows = []
    for season in SCORE_SEASONS:
        train = frame.loc[frame["season"].lt(season) & frame["ats_margin"].notna()]
        model = make_margin_estimator(
            "ridge", ridge_alpha=BASE_ALPHA, suppressed_indicator_columns=suppressed
        )
        model.fit(train.loc[:, list(columns)], train["ats_margin"])
        pred = model.predict(train.loc[:, list(columns)])
        target = train["ats_margin"].to_numpy()
        keep = target != 0
        hit = np.sign(pred[keep]) == np.sign(target[keep])
        rows.append((int(keep.sum()), float(hit.mean())))
    n = sum(r[0] for r in rows)
    return sum(r[0] * r[1] for r in rows) / n


def stage_run():
    table = pd.read_parquet(TABLE)
    columns = reduced_columns()
    print("reduced columns", len(columns), "of", len(FULL_COLUMNS), flush=True)
    feat_a = table.loc[table["season"].ge(2009)].reset_index(drop=True)
    served = pd.read_parquet(SERVED_FEATURES)
    results = {}
    scored = {}
    for tag, feats, cols in (
        ("A", feat_a, columns),
        ("B", table, columns),
        ("served", served, FULL_COLUMNS),
    ):
        frame = decisive(run_eval(feats, cols, tag))
        scored[tag] = frame.loc[frame["season"].isin(SCORE_SEASONS)]
        print(tag, len(scored[tag]), record(scored[tag]), flush=True)
    results["B_vs_A"] = compare("B v A", scored["B"], scored["A"])
    results["A_vs_served"] = compare("A v served", scored["A"], scored["served"])
    results["B_vs_served"] = compare("B v served", scored["B"], scored["served"])
    results["in_sample"] = {
        "A": in_sample(regular_season_rows(feat_a), columns),
        "B": in_sample(regular_season_rows(table), columns),
        "served": in_sample(regular_season_rows(served), FULL_COLUMNS),
    }
    results["out_of_sample"] = {
        k: float(v["correct_at_open_probability_rule"].mean()) for k, v in scored.items()
    }
    results["reduced_columns"] = list(columns)
    results["table_sha256"] = hashlib.sha256(TABLE.read_bytes()).hexdigest()
    (OUT_DIR / "report.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({k: v for k, v in results.items() if k != "reduced_columns"}, indent=1))


WEIGHT_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)
CHOSEN_WEIGHTS = {}


def weighted_fit(estimator, frame, columns, weight):
    rows = frame if weight > 0 else frame.loc[frame["season"].ge(2009)]
    w = (
        np.where(rows["season"].to_numpy() < 2009, weight, 1.0)
        if weight > 0
        else np.ones(len(rows))
    )
    estimator.fit(
        rows.loc[:, list(columns)],
        _target_values(rows, "market_residual"),
        regressor__sample_weight=w,
    )
    return estimator


def new_estimator(columns):
    suppressed = tuple(
        c for c in _PROFILE_SUPPRESSED_MISSING_INDICATORS.get(PROFILE, ()) if c in columns
    )
    return make_margin_estimator(
        "ridge", ridge_alpha=BASE_ALPHA, suppressed_indicator_columns=suppressed
    )


def choose_weight(training, columns, last_season):
    complete = training.loc[training["season"].lt(last_season)]
    scores = {}
    for weight in WEIGHT_GRID:
        total = 0.0
        count = 0
        for held in sorted(s for s in complete["season"].unique() if s >= 2009):
            fit_rows = complete.loc[complete["season"].ne(held)]
            test_rows = complete.loc[complete["season"].eq(held)]
            model = weighted_fit(new_estimator(columns), fit_rows, columns, weight)
            err = _target_values(test_rows, "market_residual").to_numpy() - model.predict(
                test_rows.loc[:, list(columns)]
            )
            total += float(np.sum(err**2))
            count += len(err)
        scores[weight] = total / count
    best = min(WEIGHT_GRID, key=lambda w: (scores[w], w))
    return best, scores


def make_weighted_fit_margin_model(columns):
    def fit(
        frame, *, target="margin", model_name="ridge", feature_profile="base", ridge_alpha=10.0, **_
    ):
        training = regular_season_rows(frame)
        training = training.loc[_target_values(training, target).notna()].copy()
        training["gameday"] = pd.to_datetime(training["gameday"], errors="raise")
        training = training.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        last_season = int(training["season"].max())
        if last_season not in CHOSEN_WEIGHTS:
            best, scores = choose_weight(training, columns, last_season)
            CHOSEN_WEIGHTS[last_season] = {"w": best, "loso_mse": scores}
            print("fold", last_season, "w", best, scores, flush=True)
        weight = CHOSEN_WEIGHTS[last_season]["w"]
        distribution_rows = int(len(training) * 0.20)
        split = len(training) - distribution_rows
        fit_part = training.iloc[:split]
        distribution_part = training.iloc[split:]
        temporary = weighted_fit(new_estimator(columns), fit_part, columns, weight)
        prediction = np.asarray(
            temporary.predict(distribution_part.loc[:, list(columns)]), dtype=float
        )
        residuals = _target_values(distribution_part, target).to_numpy(dtype=float) - prediction
        residuals = residuals[np.isfinite(residuals)]
        estimator = weighted_fit(new_estimator(columns), training, columns, weight)
        return MarginModel(
            estimator=estimator,
            residuals=residuals,
            model_name=model_name,
            ridge_alpha=ridge_alpha,
            target=target,
            feature_columns=tuple(columns),
            training_rows=len(training),
            distribution_rows=len(residuals),
            training_max_gameday=training["gameday"].max().date().isoformat(),
            column_penalties=None,
        )

    return fit


def weighted_in_sample(frame, columns):
    rows = []
    for season in SCORE_SEASONS:
        train = frame.loc[frame["season"].lt(season) & frame["ats_margin"].notna()]
        weight = CHOSEN_WEIGHTS[season]["w"]
        model = weighted_fit(new_estimator(columns), train, columns, weight)
        pred = model.predict(train.loc[:, list(columns)])
        target = train["ats_margin"].to_numpy()
        keep = target != 0
        hit = np.sign(pred[keep]) == np.sign(target[keep])
        rows.append((int(keep.sum()), float(hit.mean())))
    n = sum(r[0] for r in rows)
    return sum(r[0] * r[1] for r in rows) / n


def stage_weighted():
    table = pd.read_parquet(TABLE)
    columns = reduced_columns()
    feat_a = table.loc[table["season"].ge(2009)].reset_index(drop=True)
    frame_a = decisive(run_eval(feat_a, columns, "A"))
    frame_a = frame_a.loc[frame_a["season"].isin(SCORE_SEASONS)]
    original = clv.fit_margin_model
    clv.fit_margin_model = make_weighted_fit_margin_model(columns)
    try:
        frame_w = decisive(run_eval(table, columns, "Bw"))
    finally:
        clv.fit_margin_model = original
    frame_w = frame_w.loc[frame_w["season"].isin(SCORE_SEASONS)]
    out = {
        "Bw_vs_A": compare("Bw v A", frame_w, frame_a),
        "chosen": {int(k): v for k, v in CHOSEN_WEIGHTS.items()},
        "in_sample_Bw": weighted_in_sample(regular_season_rows(table), columns),
        "out_of_sample_Bw": float(frame_w["correct_at_open_probability_rule"].mean()),
        "out_of_sample_A": float(frame_a["correct_at_open_probability_rule"].mean()),
    }
    (OUT_DIR / "report_weighted.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))


def main():
    stage = sys.argv[1]
    if stage == "build":
        stage_build()
    elif stage == "coverage":
        stage_coverage()
    elif stage == "run":
        stage_run()
    elif stage == "weighted":
        stage_weighted()
    else:
        raise SystemExit("stage must be build or run")


main()
