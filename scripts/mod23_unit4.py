import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod23_unit3 as u3

from nfl_ats import clv
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.margin import (
    _PROFILE_SUPPRESSED_MISSING_INDICATORS,
    make_margin_estimator,
    resolve_feature_groups,
)
from nfl_ats.modeling import regular_season_rows

ROOT = u3.ROOT
OUT_DIR = ROOT / "artifacts" / "mod23_unit4"
UNIT1_SELECTION = ROOT / "artifacts" / "mod23_unit1" / "selection.json"
FULL = tuple(u3.FULL_COLUMNS)
SET_NAME = u3.SET_NAME
PROFILE = u3.PROFILE
BASE_ALPHA = u3.BASE_ALPHA
SCORE_SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
INNER_SEASONS = 4
BLEND_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SHRINK_GRID = (0.25, 0.5, 0.75, 1.0)
COLUMN_FAMILY = dict(zip(FULL, resolve_feature_groups(FULL), strict=True))
SUPPRESSED = _PROFILE_SUPPRESSED_MISSING_INDICATORS.get(PROFILE, ())
CTX = {}
ORIGINAL_FIT = clv.fit_margin_model
ORIGINAL_FOR_WEEK = clv.DiscretePushReader.for_week


class Mix:
    def __init__(self, served, alt, served_cols, alt_cols, weight):
        self.served = served
        self.alt = alt
        self.served_cols = list(served_cols)
        self.alt_cols = list(alt_cols) if alt_cols is not None else None
        self.weight = weight

    def predict(self, frame):
        primary = np.asarray(self.served.predict(frame.loc[:, self.served_cols]), dtype=float)
        if self.alt is None:
            return self.weight * primary
        secondary = np.asarray(self.alt.predict(frame.loc[:, self.alt_cols]), dtype=float)
        return self.weight * primary + (1.0 - self.weight) * secondary


def estimator(alpha):
    return make_margin_estimator(
        "ridge", 42, ridge_alpha=alpha, suppressed_indicator_columns=SUPPRESSED
    )


def columns_without(dropped):
    return tuple(c for c in FULL if COLUMN_FAMILY[c] not in dropped)


def alt_config(mode, season):
    if mode == "blend_net":
        return tuple(CTX["net_columns"]), BASE_ALPHA
    if mode == "blend_trim":
        choice = CTX["selection"][str(season)]
        return columns_without(set(choice["dropped"])), float(choice["alpha"])
    return None, None


def inner_predictions(mode, season):
    frame = CTX["inner_frame"]
    columns, alpha = alt_config(mode, season)
    served, alt, actual = [], [], []
    for inner in range(season - INNER_SEASONS, season):
        train = frame.loc[frame["season"].lt(inner)]
        test = frame.loc[frame["season"].eq(inner)]
        target = train["ats_margin"]
        model = estimator(BASE_ALPHA).fit(train.loc[:, list(FULL)], target)
        served.append(np.asarray(model.predict(test.loc[:, list(FULL)]), dtype=float))
        if columns is not None:
            other = estimator(alpha).fit(train.loc[:, list(columns)], target)
            alt.append(np.asarray(other.predict(test.loc[:, list(columns)]), dtype=float))
        actual.append(test["ats_margin"].to_numpy(dtype=float))
    return (
        np.concatenate(served),
        np.concatenate(alt) if alt else None,
        np.concatenate(actual),
    )


def choose_weight(mode, season):
    key = (mode, season)
    if key in CTX["chosen"]:
        return CTX["chosen"][key]
    served, alt, actual = inner_predictions(mode, season)
    grid = SHRINK_GRID if mode == "shrink" else BLEND_GRID
    scores = {}
    for weight in grid:
        pred = weight * served if alt is None else weight * served + (1.0 - weight) * alt
        scores[weight] = float(np.mean((actual - pred) ** 2))
    best = min(grid, key=lambda w: (scores[w], -w))
    CTX["chosen"][key] = {
        "weight": best,
        "inner_mse": scores[best],
        "inner_mse_served": scores[1.0],
        "inner_scores": scores,
    }
    return CTX["chosen"][key]


def patched_fit(frame, **kwargs):
    FEATURE_SETS[SET_NAME] = FULL
    try:
        model = ORIGINAL_FIT(frame, **kwargs)
    finally:
        FEATURE_SETS[SET_NAME] = CTX["union"]
    CTX["model"] = model
    CTX["frame"] = frame
    return model


def finalize(season):
    mode = CTX["mode"]
    model = CTX["model"]
    if mode == "fixed":
        return
    training = regular_season_rows(CTX["frame"])
    training = training.loc[training["ats_margin"].notna()].copy()
    training["gameday"] = pd.to_datetime(training["gameday"])
    training = training.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    distribution_rows = int(len(training) * 0.20)
    split = len(training) - distribution_rows
    fit_part = training.iloc[:split]
    distribution_part = training.iloc[split:]
    columns, alpha = alt_config(mode, season)
    choice = choose_weight(mode, season)
    weight = choice["weight"]
    served_temp = estimator(BASE_ALPHA).fit(fit_part.loc[:, list(FULL)], fit_part["ats_margin"])
    served_dist = np.asarray(served_temp.predict(distribution_part.loc[:, list(FULL)]), dtype=float)
    if columns is None:
        alt_full = None
        blended = weight * served_dist
    else:
        alt_temp = estimator(alpha).fit(fit_part.loc[:, list(columns)], fit_part["ats_margin"])
        alt_dist = np.asarray(
            alt_temp.predict(distribution_part.loc[:, list(columns)]), dtype=float
        )
        blended = weight * served_dist + (1.0 - weight) * alt_dist
        alt_full = estimator(alpha).fit(training.loc[:, list(columns)], training["ats_margin"])
    residuals = distribution_part["ats_margin"].to_numpy(dtype=float) - blended
    model.residuals = residuals[np.isfinite(residuals)]
    model.estimator = Mix(model.estimator, alt_full, FULL, columns, weight)
    model.feature_columns = tuple(CTX["union"])
    CTX["weights"][(mode, season)] = weight


def patched_for_week(pool, *, season, **kwargs):
    finalize(int(season))
    return ORIGINAL_FOR_WEEK(pool, season=season, **kwargs)


def run_arm(features, mode, arm_name, extra_columns):
    union = list(FULL) + [c for c in extra_columns if c not in FULL]
    CTX.update({"mode": mode, "union": tuple(union), "weights": {}})
    original = FEATURE_SETS[SET_NAME]
    FEATURE_SETS[SET_NAME] = tuple(union)
    clv.fit_margin_model = patched_fit
    clv.DiscretePushReader.for_week = staticmethod(patched_for_week)
    try:
        config = {
            **u3.BASE_CONFIG,
            "mod23_variant": json.dumps([arm_name, CTX["script_sha"], CTX["selection_sha"]]),
        }
        return clv.opener_pick_evaluation(
            u3.MARKET_ROOT,
            features,
            active_model_config=config,
            min_train_games=DEFAULT_MIN_TRAIN_GAMES,
        )
    finally:
        clv.fit_margin_model = ORIGINAL_FIT
        clv.DiscretePushReader.for_week = ORIGINAL_FOR_WEEK
        FEATURE_SETS[SET_NAME] = original


def reliability(frame):
    p_pick = np.where(frame["p"] >= 0.5, frame["p"], 1.0 - frame["p"])
    band = pd.qcut(pd.Series(p_pick, index=frame.index).rank(method="first"), 5, labels=False)
    rows = []
    for b in range(5):
        mask = (band == b).to_numpy()
        rows.append(
            {
                "band": b + 1,
                "n": int(mask.sum()),
                "mean_pick_probability": float(p_pick[mask].mean()),
                "realized_pick_rate": float(
                    frame["correct_at_open_probability_rule"].to_numpy()[mask].mean()
                ),
            }
        )
    return rows


def outer_mse(frame):
    return float(np.mean((frame["margin_vs_open"] - frame["residual_at_open"]) ** 2))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(u3.FEATURES)
    print("feature sha", hashlib.sha256(u3.FEATURES.read_bytes()).hexdigest(), flush=True)
    schedules = pd.read_parquet(u3.SNAPSHOT / "schedules.parquet")
    team_stats = pd.read_parquet(u3.SNAPSHOT / "team_stats.parquet")
    terms, check = u3.adjusted_columns(features, schedules, team_stats)
    features = features.merge(terms, on="game_id", how="left", validate="one_to_one")
    net_columns = u3.arm_columns()["compact_net"]
    selection = json.loads(UNIT1_SELECTION.read_text())["greedy"]
    inner_frame = regular_season_rows(features)
    inner_frame = inner_frame.loc[inner_frame["ats_margin"].notna() & inner_frame["result"].notna()]
    CTX.update(
        {
            "net_columns": net_columns,
            "selection": selection,
            "inner_frame": inner_frame.copy(),
            "chosen": {},
            "script_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "selection_sha": hashlib.sha256(UNIT1_SELECTION.read_bytes()).hexdigest()[:16],
        }
    )
    base = u3.decisive(run_arm(features, "fixed", "check_w1", []))
    art = (
        pd.read_parquet(u3.BASELINE_DIR / "per_game.parquet")
        .set_index("game_id")["correct_at_open_probability_rule"]
        .dropna()
    )
    mine = base["correct_at_open_probability_rule"]
    print("baseline", int(mine.sum()), len(mine), "artifact", int(art.sum()), len(art), flush=True)
    if int(mine.sum()) != int(art.sum()) or len(mine) != len(art):
        print("REPRODUCTION FAILED")
        return
    specs = {
        "blend_compact_net": ("blend_net", net_columns),
        "blend_unit1_trimmed": ("blend_trim", []),
        "shrink_served": ("shrink", []),
    }
    grids = {"blend_net": BLEND_GRID, "blend_trim": BLEND_GRID, "shrink": SHRINK_GRID}
    summaries = {}
    for name, (mode, extra) in specs.items():
        arm = u3.decisive(run_arm(features, mode, name, extra))
        arm.to_parquet(OUT_DIR / f"{name}_per_game.parquet")
        summary = u3.summarize(name, arm, base)
        summary["reliability"] = reliability(arm.reset_index())
        summary["reliability_base"] = reliability(base.reset_index())
        joined = arm.join(base, rsuffix="_b", how="inner")
        summary["pick_changes"] = int(
            (
                joined["pick_home_at_open_probability_rule"]
                != joined["pick_home_at_open_probability_rule_b"]
            ).sum()
        )
        weights = {}
        gaps = {}
        for season in SCORE_SEASONS:
            choice = CTX["chosen"][(mode, season)]
            sub = arm.loc[arm["season"].eq(season)]
            sub_base = base.loc[base["season"].eq(season)]
            weights[season] = choice["weight"]
            gaps[season] = {
                "inner_mse_chosen": choice["inner_mse"],
                "inner_mse_served": choice["inner_mse_served"],
                "inner_gain": choice["inner_mse_served"] - choice["inner_mse"],
                "outer_mse_chosen": outer_mse(sub),
                "outer_mse_served": outer_mse(sub_base),
                "outer_gain": outer_mse(sub_base) - outer_mse(sub),
            }
        summary["weights"] = weights
        summary["gaps"] = gaps
        summary["inner_looks"] = len(grids[mode]) * len(SCORE_SEASONS)
        summaries[name] = summary
        print(name, summary["record"], summary["diff_points"], weights, flush=True)
    report = {
        "reproduction_max_abs_diff": check,
        "summaries": summaries,
        "looks": {
            "outer_arms": len(specs),
            "inner": sum(s["inner_looks"] for s in summaries.values()),
        },
        "criterion": "margin_mse_on_ats_margin_inner_walk_forward",
        "inner_seasons": INNER_SEASONS,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
