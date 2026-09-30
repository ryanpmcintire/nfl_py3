import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_ats import clv
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_FAMILIES, FEATURE_SETS, STATE_METRICS
from nfl_ats.features import attach_team_states, build_team_game_metrics, build_team_states
from nfl_ats.margin import _MARGIN_PROFILE_FEATURE_SETS, margin_feature_columns

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "game_features_weak_stack.parquet"
MARKET_ROOT = ROOT / "data" / "market" / "raw"
BASELINE_DIR = ROOT / "artifacts" / "opener_evaluation" / "20260929T192743Z"
OUT_DIR = ROOT / "artifacts" / "mod23_unit3"
SNAPSHOT = ROOT / "data" / "raw" / "20260929T191306Z"
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
PAIRS = (
    ("off_epa_per_play", "def_epa_per_play"),
    ("off_pass_epa_per_play", "def_pass_epa_per_play"),
    ("off_rush_epa_per_play", "def_rush_epa_per_play"),
    ("off_yards_per_play", "def_yards_per_play"),
    ("off_turnover_rate", "def_takeaway_rate"),
    ("off_sack_rate", "def_sack_rate"),
)
ADJUSTED = tuple(metric for pair in PAIRS for metric in pair)
NET = (
    ("net_epa", "off_epa_per_play", "def_epa_per_play"),
    ("net_pass_epa", "off_pass_epa_per_play", "def_pass_epa_per_play"),
    ("net_rush_epa", "off_rush_epa_per_play", "def_rush_epa_per_play"),
)
OFFENSE_DEFENSE = set(FEATURE_FAMILIES["offense"]) | set(FEATURE_FAMILIES["defense"])
SIDES = ("home", "away", "diff")


def prior_league_means(team_games, metric):
    values = pd.to_numeric(team_games[metric], errors="coerce")
    per_day = values.groupby(team_games["gameday"]).agg(["sum", "count"]).sort_index()
    running = per_day["sum"].cumsum().shift(1) / per_day["count"].cumsum().shift(1)
    return team_games["gameday"].map(running).to_numpy(dtype=float)


def adjust_team_games(features, team_games):
    frame = team_games.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    keep = ["game_id", "home_team", "away_team"] + [
        f"{side}_{metric}" for side in ("home", "away") for metric in ADJUSTED
    ]
    frame = frame.merge(features[keep], on="game_id", how="left", validate="many_to_one")
    is_home = frame["team"].eq(frame["home_team"]).to_numpy()
    if not (is_home | frame["team"].eq(frame["away_team"]).to_numpy()).all():
        raise SystemExit("team-game rows not matched to the schedule")
    means = {metric: prior_league_means(frame, metric) for metric in ADJUSTED}
    adjusted = frame[["game_id", "season", "gameday", "team", *STATE_METRICS]].copy()
    for offense, defense in PAIRS:
        for own, faced in ((offense, defense), (defense, offense)):
            opponent_state = np.where(
                is_home,
                frame[f"away_{faced}"].to_numpy(dtype=float),
                frame[f"home_{faced}"].to_numpy(dtype=float),
            )
            shift = opponent_state - means[faced]
            shift = np.where(np.isfinite(shift), shift, 0.0)
            adjusted[own] = pd.to_numeric(frame[own], errors="coerce").to_numpy() - shift
    return adjusted


def attached(features, team_games):
    states = build_team_states(team_games)
    return attach_team_states(
        features[["game_id", "season", "gameday", "home_team", "away_team"]], states
    )


def adjusted_columns(features, schedules, team_stats):
    team_games = build_team_game_metrics(schedules, team_stats)
    dated = features.copy()
    dated["gameday"] = pd.to_datetime(dated["gameday"])
    raw = attached(dated, team_games)
    check = {}
    for column in ("home_off_epa_per_play", "away_def_yards_per_play", "diff_off_sack_rate"):
        both = pd.concat([raw[column], dated[column].reset_index(drop=True)], axis=1)[
            dated["game_type"].eq("REG").to_numpy()
        ].dropna()
        check[column] = float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max())
    print("raw state reproduction max abs diff", check, flush=True)
    adjusted = attached(dated, adjust_team_games(dated, team_games))
    out = features[["game_id"]].copy()
    for metric in ADJUSTED:
        for side in SIDES:
            out[f"{side}_adj_{metric}"] = adjusted[f"{side}_{metric}"].to_numpy()
    for name, offense, defense in NET:
        for side in ("home", "away"):
            out[f"{side}_{name}"] = out[f"{side}_adj_{offense}"] - out[f"{side}_adj_{defense}"]
        out[f"diff_{name}"] = out[f"home_{name}"] - out[f"away_{name}"]
    return out, check


def arm_columns():
    base = list(FULL_COLUMNS)
    replaced = []
    for column in base:
        side, _, metric = column.partition("_")
        if column in OFFENSE_DEFENSE and metric in ADJUSTED:
            replaced.append(f"{side}_adj_{metric}")
        else:
            replaced.append(column)
    added = [f"{side}_adj_{metric}" for metric in ADJUSTED for side in SIDES]
    net = [f"{side}_{name}" for name, _, _ in NET for side in SIDES]
    return {
        "replace_adjusted": replaced,
        "add_adjusted": base + added,
        "compact_net": [c for c in base if c not in OFFENSE_DEFENSE] + net,
    }


def run_arm(features, columns):
    original = FEATURE_SETS[SET_NAME]
    FEATURE_SETS[SET_NAME] = tuple(columns)
    try:
        config = {**BASE_CONFIG, "mod23_variant": json.dumps(list(columns))}
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
    schedules = pd.read_parquet(SNAPSHOT / "schedules.parquet")
    team_stats = pd.read_parquet(SNAPSHOT / "team_stats.parquet")
    terms, check = adjusted_columns(features, schedules, team_stats)
    features = features.merge(terms, on="game_id", how="left", validate="one_to_one")
    coverage = {"reproduction_max_abs_diff": check}
    base = decisive(run_arm(features, FULL_COLUMNS))
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
    arms = arm_columns()
    for name, extra in arms.items():
        arm = decisive(run_arm(features, extra))
        arm.to_parquet(OUT_DIR / f"{name}_per_game.parquet")
        summaries[name] = summarize(name, arm, base)
        summaries[name]["n_columns"] = len(extra)
        print(name, summaries[name]["record"], summaries[name]["diff_points"], flush=True)
    report = {"coverage": coverage, "summaries": summaries, "looks": len(arms)}
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
