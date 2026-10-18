import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mod23_unit3 as u3
import mod24_forward_log as M

from nfl_ats import clv
from nfl_ats.cli_commands.prediction import (
    MarginPredictRequest,
    _key_line_policy,
    _served_discrete_push_read,
    _served_home_side_offsets,
    _with_discrete_fallback,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.outcomes import score_outcome_week

UNIT_PATH = HERE / "mod23_unit10b.py"
U = {"__file__": str(UNIT_PATH), "__name__": "mod23_unit10b_forward"}
exec(compile(UNIT_PATH.read_text().rsplit("\nmain()", 1)[0], str(UNIT_PATH), "exec"), U)

ROOT = u3.ROOT
OUT_DIR = ROOT / "artifacts" / "draftcap_forward"
RUNS_DIR = OUT_DIR / "runs"
FROZEN_PATH = OUT_DIR / "frozen.json"
SEASON = M.SEASON
FIRST_WEEK = 7
LAST_WEEK = 18
ARM = "arm_b_starter_draft_capital"
PROBABILITY_METHOD = M.PROBABILITY_METHOD
MIN_EDGE = M.MIN_EDGE
ARM_COLUMNS = tuple(U["NEW"][:2])
FULL = tuple(U["FULL"])
ALPHA = U["BASE_ALPHA"]
PROFILE = u3.PROFILE


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prior_provenance(features):
    tg = U["build_inputs"]()
    f = features[["game_id", "season", "week", "home_team", "away_team"]].copy()
    f["order"] = f["season"].astype(int) * 100 + f["week"].astype(int)
    out = {}
    for side in ("home", "away"):
        left = f[["game_id", "order", f"{side}_team"]].rename(columns={f"{side}_team": "team"}).astype({"team": object}).sort_values("order")
        right = tg[["team", "prior_order"]].astype({"team": object})
        m = pd.merge_asof(left, right, left_on="order", right_on="prior_order", by="team", allow_exact_matches=False)
        used = m.loc[m["prior_order"].notna()]
        if not (used["prior_order"] < used["order"]).all():
            raise SystemExit("availability assertion failed")
        out[side] = m.set_index("game_id")["prior_order"]
    return out, tg


def fit_frozen(features_with):
    train = features_with.loc[features_with["season"].le(SEASON - 1) & features_with["result"].notna()]
    original = FEATURE_SETS[u3.SET_NAME]
    FEATURE_SETS[u3.SET_NAME] = FULL + ARM_COLUMNS
    try:
        model = clv.fit_margin_model(
            train, target="market_residual", model_name="ridge", feature_profile=PROFILE, ridge_alpha=ALPHA
        )
    finally:
        FEATURE_SETS[u3.SET_NAME] = original
    cols = list(model.feature_columns)
    stats = np.asarray(model.estimator.named_steps["imputer"].statistics_, dtype=float)
    kept = [c for c, v in zip(cols, stats) if np.isfinite(v)]
    coef = np.asarray(model.estimator.steps[-1][1].coef_, dtype=float)
    if len(coef) < len(kept):
        raise SystemExit("coefficient mapping failed")
    table = {c: float(coef[kept.index(c)]) for c in ARM_COLUMNS}
    text = json.dumps(table, sort_keys=True)
    return table, sha256_text(text), int(train["season"].max())


def freeze(table, digest, dry):
    if FROZEN_PATH.exists():
        frozen = json.loads(FROZEN_PATH.read_text())
        if frozen["arm_coefficients_2026_sha256"] != digest:
            raise SystemExit("draft capital arm coefficients differ from the enrolled frozen set")
        return frozen
    frozen = {
        "enrolled_at_utc": datetime.now(UTC).isoformat(),
        "arm": ARM,
        "fit_seasons": "2013 through 2025",
        "arm_columns": list(ARM_COLUMNS),
        "arm_coefficients_2026": table,
        "arm_coefficients_2026_sha256": digest,
        "ridge_alpha": ALPHA,
        "starters_per_unit": U["STARTERS"],
        "snap_file": str(U["SNAP_FILE"]),
        "combine_file": str(U["COMBINE"]),
        "picks_dir": str(U["PICKS_DIR"]),
        "canonical_row": "latest row per game_id recorded before its pick deadline",
    }
    if dry:
        return frozen
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(FROZEN_PATH, "x") as handle:
        handle.write(json.dumps(frozen, indent=1))
    return frozen


def score_arm(merged, week, cols):
    columns = tuple(cols)
    original = FEATURE_SETS[u3.SET_NAME]
    FEATURE_SETS[u3.SET_NAME] = columns
    try:
        request = MarginPredictRequest(
            features=u3.FEATURES,
            season=SEASON,
            week=week,
            regressor="ridge",
            min_edge=MIN_EDGE,
            min_train_games=DEFAULT_MIN_TRAIN_GAMES,
            feature_profile=u3.PROFILE,
            ridge_alpha=ALPHA,
            probability_method=PROBABILITY_METHOD,
            line_sweep=False,
        )
        home_side = _served_home_side_offsets(merged, request)
        center = home_side["center_offsets"] if home_side is not None else None
        discrete = _served_discrete_push_read(merged, request)
        log = {}

        def score(reader):
            return score_outcome_week(
                merged,
                season=SEASON,
                week=week,
                regressor="ridge",
                min_edge=MIN_EDGE,
                min_train_games=DEFAULT_MIN_TRAIN_GAMES,
                feature_profile=u3.PROFILE,
                ridge_alpha=ALPHA,
                probability_method=PROBABILITY_METHOD,
                center_offsets=center,
                discrete_read=reader,
                discrete_read_log=log,
                key_line_pick_read=_key_line_policy(reader),
            )

        predictions, _ = _with_discrete_fallback(score, discrete, log)
    finally:
        FEATURE_SETS[u3.SET_NAME] = original
    return predictions.loc[predictions["method"].eq("market_residual")].set_index("game_id")


def served_frames(week):
    candidates = sorted(M.FORECAST_ROOT.glob(f"{SEASON}-week-{week:02d}-*"))
    if not candidates:
        return None, None, None
    folder = candidates[-1]
    preds = pd.read_csv(folder / "predictions.csv").drop_duplicates("game_id").set_index("game_id")
    pool = pd.read_csv(folder / "pool_card.csv").drop_duplicates("game_id").set_index("game_id")
    return folder, preds, pool


def record(week, now, dry):
    features = pd.read_parquet(u3.FEATURES)
    features["gameday"] = pd.to_datetime(features["gameday"])
    target = M.week_frame(features, week)
    if target.empty:
        raise SystemExit(f"no {SEASON} week {week} games in the feature table")
    refused = []
    eligible = []
    for row in target.itertuples():
        if pd.notna(row.result):
            refused.append({"game_id": row.game_id, "reason": "outcome_present"})
        elif now >= row.deadline:
            refused.append({"game_id": row.game_id, "reason": "past_deadline"})
        elif pd.isna(row.spread_line):
            refused.append({"game_id": row.game_id, "reason": "no_line"})
        else:
            eligible.append(row.game_id)
    if not eligible:
        print(json.dumps({"week": week, "logged": [], "refused": refused}, indent=1))
        return
    U["SNAP_FILE"] = M.current_season_snaps(week)
    adjusted = U["add_features"](features)
    table, digest, fit_max = fit_frozen(adjusted)
    freeze(table, digest, dry)
    prior, tg = prior_provenance(features)
    arm = score_arm(adjusted, week, FULL + ARM_COLUMNS)
    base = score_arm(features, week, FULL)
    folder, served_preds, pool = served_frames(week)
    recorded = pd.Timestamp(datetime.now(UTC))
    meta = target.set_index("game_id")
    adj = adjusted.set_index("game_id")
    rows = []
    for game_id in eligible:
        if recorded >= meta.loc[game_id, "deadline"]:
            refused.append({"game_id": game_id, "reason": "past_deadline_at_write"})
            continue
        p_b = float(arm.loc[game_id, "home_cover_probability"])
        p_0 = float(base.loc[game_id, "home_cover_probability"])
        row = {
            "game_id": game_id,
            "season": SEASON,
            "week": week,
            "kickoff": meta.loc[game_id, "kickoff"].isoformat(),
            "deadline": meta.loc[game_id, "deadline"].isoformat(),
            "recorded_at_utc": recorded.isoformat(),
            "line": float(arm.loc[game_id, "spread_line"]),
            "arm_home_cover_probability": p_b,
            "arm_side": M.side_of(p_b),
            "base_home_cover_probability": p_0,
            "base_side": M.side_of(p_0),
            "diff_dc_off_ovr": None if pd.isna(adj.loc[game_id, ARM_COLUMNS[0]]) else float(adj.loc[game_id, ARM_COLUMNS[0]]),
            "diff_dc_def_ovr": None if pd.isna(adj.loc[game_id, ARM_COLUMNS[1]]) else float(adj.loc[game_id, ARM_COLUMNS[1]]),
            "home_prior_game_order": None if pd.isna(prior["home"].loc[game_id]) else int(prior["home"].loc[game_id]),
            "away_prior_game_order": None if pd.isna(prior["away"].loc[game_id]) else int(prior["away"].loc[game_id]),
        }
        if served_preds is not None and game_id in served_preds.index:
            sp = float(served_preds.loc[game_id, "home_cover_probability"])
            row["served_model_home_cover_probability"] = sp
            row["served_model_side"] = M.side_of(sp)
            row["served_line"] = float(served_preds.loc[game_id, "spread_line"])
        if pool is not None and game_id in pool.index:
            row["served_pool_side"] = str(pool.loc[game_id, "pool_side"])
            row["served_pool_pick_probability"] = float(pool.loc[game_id, "pick_probability"])
        rows.append(row)
    payload = {
        "schema": "draftcap_forward_log/1",
        "arm": ARM,
        "season": SEASON,
        "week": week,
        "recorded_at_utc": recorded.isoformat(),
        "script_sha256": M.sha256_file(__file__),
        "feature_table_sha256": M.sha256_file(u3.FEATURES),
        "snap_file": str(U["SNAP_FILE"]),
        "latest_snap_order": int(tg["prior_order"].max()),
        "arm_coefficients_2026_sha256": digest,
        "arm_coefficients_fit_max_season": fit_max,
        "served_forecast_dir": None if folder is None else folder.name,
        "refused": refused,
        "rows": rows,
    }
    if dry or not rows:
        print(json.dumps({"dry": dry, "week": week, "refused": refused, "payload": {k: v for k, v in payload.items() if k != "rows"}, "rows": rows}, indent=1, default=str))
        return
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = recorded.strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"{SEASON}-week-{week:02d}-{stamp}.json"
    with open(path, "x") as handle:
        handle.write(json.dumps(payload, indent=1, default=str))
    print(json.dumps({"path": str(path), "logged": [r["game_id"] for r in rows], "refused": refused}, indent=1))


def status():
    files = sorted(RUNS_DIR.glob("*.json")) if RUNS_DIR.is_dir() else []
    out = {}
    for path in files:
        payload = json.loads(path.read_text())
        out.setdefault(payload["week"], []).append(path.name)
    print(json.dumps({"frozen": FROZEN_PATH.exists(), "runs_by_week": out}, indent=1))


def main():
    parser = argparse.ArgumentParser(description="Forward log of MOD-23 unit-10 arm B starter draft capital beside the same-pipeline base")
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
        return
    now = pd.Timestamp(datetime.now(UTC))
    week = args.week
    if week is None:
        features = pd.read_parquet(u3.FEATURES, columns=["season", "week", "kickoff"])
        sub = features.loc[features["season"].eq(SEASON) & features["week"].between(FIRST_WEEK, LAST_WEEK)]
        upcoming = sub.loc[pd.to_datetime(sub["kickoff"], utc=True).gt(now)]
        if upcoming.empty:
            print("no upcoming 2026 week in range")
            return
        week = int(upcoming["week"].min())
    if not args.dry and not FIRST_WEEK <= week <= LAST_WEEK:
        raise SystemExit("week outside the logged range")
    record(week, now, args.dry)


if __name__ == "__main__":
    main()
