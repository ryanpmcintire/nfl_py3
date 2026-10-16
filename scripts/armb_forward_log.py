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

from nfl_ats.cli_commands.prediction import (
    MarginPredictRequest,
    _key_line_policy,
    _served_discrete_push_read,
    _served_home_side_offsets,
    _with_discrete_fallback,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.outcomes import score_outcome_week
from nfl_ats.pick_probability import FLAG_SUM_COLUMN, signed_composition_flags
from nfl_ats.pick_probability_fit import _arrest_incidents
from nfl_ats.snapshots import latest_snapshot, load_snapshot

UNIT6_PATH = HERE / "mod23_unit6.py"
U6 = {"__file__": str(UNIT6_PATH), "__name__": "mod23_unit6_forward"}
exec(compile(UNIT6_PATH.read_text().rsplit("\nmain()", 1)[0], str(UNIT6_PATH), "exec"), U6)

ROOT = u3.ROOT
OUT_DIR = ROOT / "artifacts" / "armb_forward"
RUNS_DIR = OUT_DIR / "runs"
FROZEN_PATH = OUT_DIR / "frozen.json"
ACTIVE_POINTER = ROOT / "artifacts" / "active_pick_probability.json"
SEASON = M.SEASON
FIRST_WEEK = 7
LAST_WEEK = 18
ARM = "arm_b_own_qb_availability"
PROBABILITY_METHOD = M.PROBABILITY_METHOD
MIN_EDGE = M.MIN_EDGE
EPSILON = 1e-6


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build_delta(snapshot):
    schedules, team_stats = load_snapshot(snapshot)
    flags, snap_path = U6["team_game_flags"]()
    ctx = U6["CTX"]
    ctx["flags"] = flags
    ctx["zero_after"] = None
    ctx["fit_seasons"].clear()
    feature_module = U6["F"]
    original = feature_module.build_team_game_metrics
    feature_module.build_team_game_metrics = U6["wrapped"](original)
    built = {}
    try:
        for mode in ("raw", "B"):
            ctx["mode"] = mode
            built[mode] = feature_module._build_features_pass(schedules, team_stats, ("REG",)).set_index("game_id")
            if mode == "B":
                coef = ctx["coef"]
                for season, fit_max in ctx["fit_seasons"].items():
                    if fit_max is not None and fit_max >= season:
                        raise SystemExit(f"coefficient leak: season {season} fit through {fit_max}")
    finally:
        feature_module.build_team_game_metrics = original
    columns = [c for c in U6["STATE_COLS"] if c in built["raw"].columns]
    delta = built["B"][columns].astype(float) - built["raw"][columns].astype(float)
    return delta, coef, schedules, snap_path


def coefficient_digest(coef):
    table = {m: list(v) for m, v in sorted(coef[SEASON].items())}
    text = json.dumps(table, sort_keys=True)
    return table, sha256_text(text)


def card_coefficients():
    pointer = json.loads(ACTIVE_POINTER.read_text())
    directory = ROOT / "artifacts" / pointer["artifact"]
    path = directory / "coefficients.json"
    payload = json.loads(path.read_text())
    c = payload["coefficients"]
    return {
        "artifact": pointer["artifact"],
        "active_model_id": pointer["active_model_id"],
        "coefficients_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "intercept": c["intercept"],
        "model_logit": c["model_logit"],
        "flag_sum": c["flag_sum"],
        "move_toward_home": c["move_toward_home"],
        "move_available": c["move_available"],
    }


def freeze(table, digest, snap_path, dry):
    if FROZEN_PATH.exists():
        frozen = json.loads(FROZEN_PATH.read_text())
        if frozen["armb_coefficients_2026_sha256"] != digest:
            raise SystemExit("arm B coefficients differ from the enrolled frozen set")
        return frozen
    frozen = {
        "enrolled_at_utc": datetime.now(UTC).isoformat(),
        "arm": ARM,
        "armb_fit_seasons": "2013 through 2025",
        "armb_coefficients_2026": table,
        "armb_coefficients_2026_sha256": digest,
        "snap_counts_file_at_enrollment": str(snap_path),
        "card": card_coefficients(),
        "card_view": "tuesday: move_toward_home and move_available set to 0",
        "canonical_row": "latest row per game_id recorded before its pick deadline",
    }
    if dry:
        return frozen
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(FROZEN_PATH, "x") as handle:
        handle.write(json.dumps(frozen, indent=1))
    return frozen


def card_home(model_home, flag_sum, card):
    p = np.clip(np.asarray(model_home, dtype=float), EPSILON, 1 - EPSILON)
    z = card["intercept"] + card["model_logit"] * np.log(p / (1 - p)) + card["flag_sum"] * np.asarray(flag_sum, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def score_arm(merged, week):
    columns = tuple(U6["FULL"])
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
            ridge_alpha=u3.BASE_ALPHA,
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
                ridge_alpha=u3.BASE_ALPHA,
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
    snapshot = latest_snapshot(ROOT / "data" / "raw")
    delta, coef, schedules, snap_path = build_delta(snapshot)
    table, digest = coefficient_digest(coef)
    frozen = freeze(table, digest, snap_path, dry)
    card = frozen["card"]
    adjusted = U6["variant"](features, delta)
    arm = score_arm(adjusted, week)
    base = score_arm(features, week)
    folder, served_preds, pool = served_frames(week)
    sub = target.loc[target["game_id"].isin(eligible)]
    flags = signed_composition_flags(sub, schedules, incidents=_arrest_incidents(ROOT / "data")).set_index("game_id")
    recorded = pd.Timestamp(datetime.now(UTC))
    meta = target.set_index("game_id")
    rows = []
    for game_id in eligible:
        if recorded >= meta.loc[game_id, "deadline"]:
            refused.append({"game_id": game_id, "reason": "past_deadline_at_write"})
            continue
        p_b = float(arm.loc[game_id, "home_cover_probability"])
        flag_sum = float(flags.loc[game_id, FLAG_SUM_COLUMN])
        row = {
            "game_id": game_id,
            "season": SEASON,
            "week": week,
            "kickoff": meta.loc[game_id, "kickoff"].isoformat(),
            "deadline": meta.loc[game_id, "deadline"].isoformat(),
            "recorded_at_utc": recorded.isoformat(),
            "line": float(arm.loc[game_id, "spread_line"]),
            "armb_model_home_cover_probability": p_b,
            "armb_model_side": M.side_of(p_b),
            "composition_flag_sum": flag_sum,
            "card_tuesday_armb_home_probability": float(card_home(p_b, flag_sum, card)),
        }
        row["card_tuesday_armb_side"] = M.side_of(row["card_tuesday_armb_home_probability"])
        p_0 = float(base.loc[game_id, "home_cover_probability"])
        row["base_model_home_cover_probability"] = p_0
        row["base_model_side"] = M.side_of(p_0)
        row["card_tuesday_base_home_probability"] = float(card_home(p_0, flag_sum, card))
        row["card_tuesday_base_side"] = M.side_of(row["card_tuesday_base_home_probability"])
        if served_preds is not None and game_id in served_preds.index:
            sp = float(served_preds.loc[game_id, "home_cover_probability"])
            row["served_model_home_cover_probability"] = sp
            row["served_model_side"] = M.side_of(sp)
            row["served_line"] = float(served_preds.loc[game_id, "spread_line"])
            row["card_tuesday_served_home_probability"] = float(card_home(sp, flag_sum, card))
            row["card_tuesday_served_side"] = M.side_of(row["card_tuesday_served_home_probability"])
        if pool is not None and game_id in pool.index:
            row["served_pool_side"] = str(pool.loc[game_id, "pool_side"])
            row["served_pool_pick_probability"] = float(pool.loc[game_id, "pick_probability"])
        rows.append(row)
    payload = {
        "schema": "armb_forward_log/1",
        "arm": ARM,
        "season": SEASON,
        "week": week,
        "recorded_at_utc": recorded.isoformat(),
        "script_sha256": M.sha256_file(__file__),
        "feature_table_sha256": M.sha256_file(u3.FEATURES),
        "raw_snapshot": snapshot.snapshot_id,
        "snap_counts_file": str(snap_path),
        "armb_coefficients_2026_sha256": digest,
        "card_artifact": card["artifact"],
        "card_coefficients_sha256": card["coefficients_sha256"],
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
    parser = argparse.ArgumentParser(description="Forward log of MOD-23 unit-6 arm B and the four-term card with arm B substituted")
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
