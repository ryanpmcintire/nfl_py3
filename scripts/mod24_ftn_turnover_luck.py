import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats import clv
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.margin import _MARGIN_PROFILE_FEATURE_SETS, margin_feature_columns
from nfl_ats.pbp import load_pbp_snapshot
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root

ROOT = Path(__file__).resolve().parents[1]
FTN_HIST = ROOT / "data" / "raw" / "ftn_charting" / "20261103T160529Z"
FTN_2026 = ROOT / "data" / "raw" / "ftn_charting_2026" / "ftn_charting_2026.parquet"
PBP_ROOT = ROOT / "data" / "pbp" / "raw" / "20261027T220410Z"
SERVED_FEATURES = ROOT / "data" / "processed" / "game_features_weak_stack.parquet"
MARKET_ROOT = ROOT / "data" / "market" / "raw"
OUT_DIR = ROOT / "artifacts" / "mod24_ftn_luck"
PROFILE = "weak_stack"
SET_NAME = _MARGIN_PROFILE_FEATURE_SETS[PROFILE][1]
BASE_ALPHA = 10.0
SPAN = 8
MIN_PERIODS = 3
SCORE_SEASONS = (2022, 2023, 2024, 2025)
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 20261103
BASE_CONFIG = {
    "feature_profile": PROFILE,
    "regressor": "ridge",
    "target": "market_residual",
    "probability_method": "gaussian_median",
    "calibration_method": "none",
}
BASE_COLUMNS = tuple(margin_feature_columns("market_residual", PROFILE))
TEAM_FEATURES = ("off_int_luck", "def_int_luck", "off_drop_luck", "def_drop_luck")
LUCK_COLUMNS = tuple(f"ftn_{name}_diff" for name in TEAM_FEATURES)


def load_plays():
    ftn = pd.concat(
        [pd.read_parquet(FTN_HIST / f"ftn_charting_{s}.parquet") for s in (2022, 2023, 2024, 2025)]
        + [pd.read_parquet(FTN_2026)],
        ignore_index=True,
    )
    ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    pbp = load_pbp_snapshot(pbp_snapshot_from_root(PBP_ROOT))
    pbp = pbp.loc[pbp["season"].ge(2022)]
    keys = pbp[["game_id", "play_id", "posteam", "defteam", "interception"]].copy()
    keys["play_id"] = keys["play_id"].astype(float)
    ftn["play_id"] = ftn["play_id"].astype(float)
    plays = ftn.merge(keys, on=["game_id", "play_id"], how="inner")
    for column in ("is_interception_worthy", "is_catchable_ball", "is_drop", "interception"):
        plays[column] = plays[column].fillna(0).astype(float)
    return plays


def team_game_table(plays, dates):
    off = plays.groupby(["game_id", "posteam", "defteam"], as_index=False).agg(
        ints=("interception", "sum"),
        iw=("is_interception_worthy", "sum"),
        catchable=("is_catchable_ball", "sum"),
        drops=("is_drop", "sum"),
    )
    off["gameday"] = off["game_id"].map(dates)
    return off


def expected_drop_rate(off):
    daily = off.groupby("gameday")[["drops", "catchable"]].sum().sort_index()
    cum = daily.cumsum().shift(1)
    rate = cum["drops"] / cum["catchable"]
    return off["gameday"].map(rate)


def rolling_prior(frame, team_col, value_col):
    frame = frame.sort_values([team_col, "gameday"])
    shifted = frame.groupby(team_col)[value_col].shift(1)
    out = (
        shifted.groupby(frame[team_col])
        .apply(lambda s: s.ewm(span=SPAN, min_periods=MIN_PERIODS, adjust=False).mean())
        .reset_index(level=0, drop=True)
    )
    return out.reindex(frame.index)


def build_luck(plays, games):
    dates = dict(zip(games["game_id"], pd.to_datetime(games["gameday"]), strict=True))
    off = team_game_table(plays, dates)
    rate = expected_drop_rate(off)
    off["off_int_luck"] = off["ints"] - off["iw"]
    off["off_drop_luck"] = off["drops"] - rate * off["catchable"]
    off = off.rename(columns={"posteam": "team"})
    mirror = off.rename(columns={"team": "opp_of", "defteam": "team"})
    mirror["def_int_luck"] = mirror["ints"] - mirror["iw"]
    mirror["def_drop_luck"] = mirror["drops"] - rate * mirror["catchable"]
    per_team = off[["game_id", "team", "gameday", "off_int_luck", "off_drop_luck"]].merge(
        mirror[["game_id", "team", "def_int_luck", "def_drop_luck"]],
        on=["game_id", "team"],
        how="outer",
    )
    per_team["gameday"] = per_team["game_id"].map(dates)
    per_team = per_team.dropna(subset=["gameday"]).sort_values(["team", "gameday"])
    for name in TEAM_FEATURES:
        per_team[name] = rolling_prior(per_team, "team", name)
    return per_team[["game_id", "team", "gameday", *TEAM_FEATURES]]


def attach(features, per_team):
    out = features.copy()
    home = per_team.rename(columns={"team": "home_team"}).drop(columns="gameday")
    away = per_team.rename(columns={"team": "away_team"}).drop(columns="gameday")
    out = out.merge(home, on=["game_id", "home_team"], how="left")
    out = out.merge(away, on=["game_id", "away_team"], how="left", suffixes=("", "_away"))
    for name in TEAM_FEATURES:
        out[f"ftn_{name}_diff"] = (out[name].fillna(0.0) - out[f"{name}_away"].fillna(0.0)).where(
            out["season"].ge(2022), 0.0
        )
    return out.drop(columns=[*TEAM_FEATURES, *(f"{n}_away" for n in TEAM_FEATURES)])


def assert_pregame(plays, features, per_team):
    cutoff = pd.Timestamp("2024-11-10")
    games = features[["game_id", "gameday"]]
    keep = plays["game_id"].map(
        dict(zip(games["game_id"], pd.to_datetime(games["gameday"]), strict=True))
    )
    blanked = plays.copy()
    future = keep.ge(cutoff).to_numpy()
    for column in ("interception", "is_interception_worthy", "is_catchable_ball", "is_drop"):
        blanked.loc[future, column] = 0.0
    truncated = build_luck(blanked, features)
    in_week = pd.to_datetime(features["gameday"]).eq(cutoff)
    ids = set(features.loc[in_week, "game_id"])
    a = per_team.loc[per_team["game_id"].isin(ids)].set_index(["game_id", "team"])
    b = truncated.reindex(columns=per_team.columns)
    b = b.loc[b["game_id"].isin(ids)].set_index(["game_id", "team"])
    both = a.index.intersection(b.index)
    same = np.isclose(
        a.loc[both, list(TEAM_FEATURES)].to_numpy(dtype=float),
        b.loc[both, list(TEAM_FEATURES)].to_numpy(dtype=float),
        equal_nan=True,
        rtol=0,
        atol=1e-9,
    )
    if len(both) == 0 or not same.all():
        raise AssertionError("future information reaches the luck features")
    print("pregame truncation check passed", len(both), "team-games", flush=True)


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
        "flips": int(
            (a["correct_at_open_probability_rule"] != r["correct_at_open_probability_rule"]).sum()
        ),
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


def fold_coefficients(table):
    from nfl_ats.margin import (
        _PROFILE_SUPPRESSED_MISSING_INDICATORS,
        _target_values,
        make_margin_estimator,
    )

    columns = (*BASE_COLUMNS, *LUCK_COLUMNS)
    suppressed = tuple(
        c for c in _PROFILE_SUPPRESSED_MISSING_INDICATORS.get(PROFILE, ()) if c in columns
    )
    rows = {}
    frame = table.loc[table["game_type"].eq("REG") & table["ats_margin"].notna()]
    for season in (2023, 2024, 2025, 2026):
        train = frame.loc[frame["season"].lt(season)]
        model = make_margin_estimator(
            "ridge", ridge_alpha=BASE_ALPHA, suppressed_indicator_columns=suppressed
        )
        model.fit(train.loc[:, list(columns)], _target_values(train, "market_residual"))
        names = list(model[:-1].get_feature_names_out())
        coef = np.ravel(model[-1].coef_)
        rows[season] = {n: float(c) for n, c in zip(names, coef, strict=False) if "ftn_" in n}
    return rows


def stage_run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    served = pd.read_parquet(SERVED_FEATURES)
    plays = load_plays()
    print("matched plays", len(plays), flush=True)
    per_team = build_luck(plays, served)
    assert_pregame(plays, served, per_team)
    table = attach(served, per_team)
    print(table.loc[table["season"].ge(2022), list(LUCK_COLUMNS)].describe().T, flush=True)
    base = decisive(run_eval(table, BASE_COLUMNS, "base"))
    arm = decisive(run_eval(table, (*BASE_COLUMNS, *LUCK_COLUMNS), "luck"))
    results = {}
    for label, seasons in (("2022_2025", SCORE_SEASONS), ("2023_2025", SCORE_SEASONS[1:])):
        a = arm.loc[arm["season"].isin(seasons)]
        b = base.loc[base["season"].isin(seasons)]
        results[label] = compare(f"luck v base {label}", a, b)
        print(label, "base", record(b), "luck", record(a), flush=True)
    a26 = arm.loc[arm["season"].eq(2026)]
    b26 = base.loc[base["season"].eq(2026)]
    print("2026 n", len(a26), len(b26), flush=True)
    if len(a26) and len(b26):
        results["forward_2026"] = compare("luck v base 2026", a26, b26)
    (OUT_DIR / "report.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=1, default=str), flush=True)
    results["fold_coefficients"] = fold_coefficients(table)
    (OUT_DIR / "report.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results["fold_coefficients"], indent=1), flush=True)


def main():
    if sys.argv[1:] == ["run"]:
        stage_run()
    else:
        raise SystemExit("usage: mod24_ftn_turnover_luck.py run")


if __name__ == "__main__":
    main()
