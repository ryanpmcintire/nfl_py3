import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))

import forward_terms_week
import live_kickoff

UNIT2 = Path(__file__).resolve().parent / "mod54_unit2_penalty.py"
NS = {"__file__": str(UNIT2), "__name__": "mod54_unit2_import"}
exec(compile(UNIT2.read_text().rsplit("\nmain()", 1)[0], str(UNIT2), "exec"), NS)

from nfl_ats import margin as margin_mod  # noqa: E402
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS  # noqa: E402
from nfl_ats.outcomes import score_outcome_week  # noqa: E402
from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock  # noqa: E402

u3 = NS["u3"]
FULL = NS["FULL"]
BASE_ALPHA = NS["BASE_ALPHA"]
MIRROR_ALPHA = NS["ARMS"]["A"][1]
ORIG = NS["ORIG"]
MirrorPipe = NS["MirrorPipe"]
STATE = NS["STATE"]
PLAN = NS["PLAN"]


def covers_plan(X):
    return set(PLAN) <= set(X.columns)


class GuardedMirrorPipe(MirrorPipe):
    def fit(self, X, y=None, **kw):
        STATE["active"] = covers_plan(X)
        if not STATE["active"]:
            return Pipeline.fit(self, X, y, **kw)
        return super().fit(X, y, **kw)

    def predict(self, X, **kw):
        if not covers_plan(X):
            return Pipeline.predict(self, X, **kw)
        return super().predict(X, **kw)


def install(arm):
    STATE["arm"] = arm

    def make(model_name, random_state=42, **kw):
        base = ORIG(model_name, random_state, **kw)
        if model_name != "ridge" or arm == "B":
            return base
        return GuardedMirrorPipe(steps=list(base.steps))

    margin_mod.make_margin_estimator = make


ROOT = u3.ROOT
RUNS_DIR = ROOT / "artifacts" / "mod54_forward" / "runs"
FORECAST_ROOT = ROOT / "artifacts" / "margin_predictions"
SEASON = 2026
PROBABILITY_METHOD = "gaussian_median"
MIN_EDGE = 0.02
SCHEMA = "mod54_forward_log/1"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def side_of(p):
    if p > 0.5:
        return "HOME"
    if p < 0.5:
        return "AWAY"
    return "PASS"


def score(features, week, arm, alpha):
    original = FEATURE_SETS[u3.SET_NAME]
    FEATURE_SETS[u3.SET_NAME] = tuple(FULL)
    install(arm)
    try:
        predictions = score_outcome_week(
            features,
            season=SEASON,
            week=week,
            regressor="ridge",
            min_edge=MIN_EDGE,
            min_train_games=DEFAULT_MIN_TRAIN_GAMES,
            feature_profile=u3.PROFILE,
            ridge_alpha=alpha,
            probability_method=PROBABILITY_METHOD,
        )
    finally:
        margin_mod.make_margin_estimator = ORIG
        FEATURE_SETS[u3.SET_NAME] = original
    return predictions.loc[predictions["method"].eq("market_residual")].set_index("game_id")


def served_forecast(week):
    candidates = sorted(FORECAST_ROOT.glob(f"{SEASON}-week-{week:02d}-*"))
    if not candidates:
        return None, None
    folder = candidates[-1]
    recs = pd.read_csv(folder / "recommendations.csv")
    pool = pd.read_csv(folder / "pool_card.csv")
    merged = recs.merge(
        pool[["game_id", "pool_side", "pick_probability"]], on="game_id", how="left"
    )
    return folder, merged.set_index("game_id")


def existing_run(week):
    return sorted(RUNS_DIR.glob(f"{SEASON}-week-{week:02d}-*.json"))


def record(week):
    existing = existing_run(week)
    if existing:
        print(
            json.dumps({"week": week, "skipped": "week already logged", "path": str(existing[0])})
        )
        return
    now = pd.Timestamp(datetime.now(UTC))
    features = pd.read_parquet(u3.FEATURES)
    features["gameday"] = pd.to_datetime(features["gameday"])
    target = features.loc[features["season"].eq(SEASON) & features["week"].eq(week)].copy()
    if target.empty:
        raise SystemExit(f"no {SEASON} week {week} games in the feature table")
    target["kickoff"] = live_kickoff.earliest_by_game_id(target)
    lock = sunday_pick_lock(pd.to_datetime(target["kickoff"], utc=True))
    target["deadline"] = [
        pick_deadline(pd.Timestamp(k), lock) for k in pd.to_datetime(target["kickoff"], utc=True)
    ]
    target = target.set_index("game_id")
    refused = []
    eligible = []
    for game_id, row in target.iterrows():
        if pd.notna(row["result"]):
            refused.append({"game_id": game_id, "reason": "outcome_present"})
        elif now >= row["deadline"]:
            refused.append({"game_id": game_id, "reason": "past_deadline"})
        else:
            eligible.append(game_id)
    if not eligible:
        print(json.dumps({"week": week, "logged": [], "refused": refused}, indent=1))
        return
    base = score(features, week, "B", BASE_ALPHA)
    mirrored = score(features, week, "M", MIRROR_ALPHA)
    folder, served = served_forecast(week)
    recorded = pd.Timestamp(datetime.now(UTC))
    rows = []
    for game_id in eligible:
        if recorded >= target.loc[game_id, "deadline"]:
            refused.append({"game_id": game_id, "reason": "past_deadline_at_write"})
            continue
        meta = target.loc[game_id]
        pb = float(base.loc[game_id, "home_cover_probability"])
        pm = float(mirrored.loc[game_id, "home_cover_probability"])
        row = {
            "game_id": game_id,
            "season": SEASON,
            "week": week,
            "kickoff": meta["kickoff"].isoformat(),
            "deadline": meta["deadline"].isoformat(),
            "recorded_at_utc": recorded.isoformat(),
            "line": float(mirrored.loc[game_id, "spread_line"]),
            "base_predicted_margin": float(base.loc[game_id, "predicted_margin"]),
            "base_home_cover_probability": pb,
            "base_side": side_of(pb),
            "mirrored_predicted_margin": float(mirrored.loc[game_id, "predicted_margin"]),
            "arm_home_cover_probability": pm,
            "arm_side": side_of(pm),
        }
        if served is not None and game_id in served.index:
            sp = float(served.loc[game_id, "home_cover_probability"])
            row["served_home_cover_probability"] = sp
            row["served_side"] = side_of(sp)
            pool_side = served.loc[game_id, "pool_side"]
            row["served_pool_side"] = None if pd.isna(pool_side) else str(pool_side)
            row["served_line"] = float(served.loc[game_id, "spread_line"])
        rows.append(row)
    if not rows:
        print(json.dumps({"week": week, "logged": [], "refused": refused}, indent=1))
        return
    payload = {
        "schema": SCHEMA,
        "arm": "mirrored_rows_matched_penalty",
        "season": SEASON,
        "week": week,
        "recorded_at_utc": recorded.isoformat(),
        "script_sha256": sha256_file(__file__),
        "feature_table_sha256": sha256_file(u3.FEATURES),
        "served_forecast_dir": None if folder is None else folder.name,
        "frozen": {
            "base_ridge_alpha": BASE_ALPHA,
            "mirrored_ridge_alpha": MIRROR_ALPHA,
            "probability_method": PROBABILITY_METHOD,
            "columns": len(FULL),
        },
        "refused": refused,
        "rows": rows,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = recorded.strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"{SEASON}-week-{week:02d}-{stamp}.json"
    with open(path, "x") as handle:
        handle.write(json.dumps(payload, indent=1, default=str))
    print(
        json.dumps(
            {"path": str(path), "logged": [r["game_id"] for r in rows], "refused": refused},
            indent=1,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Forward log of the MOD-54 mirrored-row base model"
    )
    parser.add_argument("--week", default="auto")
    args = parser.parse_args()
    record(forward_terms_week.resolve_week(args.week))


if __name__ == "__main__":
    main()
