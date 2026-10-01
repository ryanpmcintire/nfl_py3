import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod23_unit3 as u3
import mod23_unit5 as u5

from nfl_ats.cli_commands.prediction import (
    MarginPredictRequest,
    _key_line_policy,
    _served_discrete_push_read,
    _served_home_side_offsets,
    _with_discrete_fallback,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.outcomes import score_outcome_week
from nfl_ats.participation import latest_participation_snapshot, load_participation_snapshot
from nfl_ats.pbp import PbpSnapshot, latest_pbp_snapshot, load_pbp_snapshot
from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock

ROOT = u3.ROOT
OUT_DIR = ROOT / "artifacts" / "mod24_forward"
RUNS_DIR = OUT_DIR / "runs"
FORECAST_ROOT = ROOT / "artifacts" / "margin_predictions"
SEASON = 2026
FIRST_WEEK = 4
LAST_WEEK = 18
ARM = "arm_a_man_zone"
PROBABILITY_METHOD = "gaussian_median"
MIN_EDGE = 0.02
SIDES = u5.SIDES


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def game_deadlines(frame):
    kickoffs = pd.to_datetime(frame["kickoff"], utc=True)
    lock = sunday_pick_lock(kickoffs)
    return pd.Series(
        [pick_deadline(pd.Timestamp(k), lock) for k in kickoffs], index=frame.index
    )


def week_frame(features, week):
    sub = features.loc[features["season"].eq(SEASON) & features["week"].eq(week)].copy()
    sub["kickoff"] = pd.to_datetime(sub["kickoff"], utc=True)
    sub["deadline"] = game_deadlines(sub)
    return sub


def logged_rows():
    rows = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            rows.append({**row, "run_file": path.name})
    return pd.DataFrame(rows)


def current_week(features, now):
    sub = features.loc[features["season"].eq(SEASON) & features["week"].between(FIRST_WEEK, LAST_WEEK)]
    kickoffs = pd.to_datetime(sub["kickoff"], utc=True)
    upcoming = sub.loc[kickoffs.gt(now)]
    if upcoming.empty:
        return None
    return int(upcoming["week"].min())


def status():
    features = pd.read_parquet(u3.FEATURES, columns=["game_id", "season", "week", "kickoff"])
    log = logged_rows()
    now = pd.Timestamp(datetime.now(UTC))
    out = []
    for week in range(FIRST_WEEK, LAST_WEEK + 1):
        sub = features.loc[features["season"].eq(SEASON) & features["week"].eq(week)]
        if sub.empty:
            out.append({"week": week, "games": 0, "logged_games": 0, "runs": 0})
            continue
        done = log.loc[log["week"].eq(week)] if not log.empty else log
        logged = set(done["game_id"]) if not log.empty else set()
        kicked = int(pd.to_datetime(sub["kickoff"], utc=True).le(now).sum())
        out.append(
            {
                "week": week,
                "games": len(sub),
                "logged_games": len(logged & set(sub["game_id"])),
                "already_kicked_off": kicked,
                "runs": int(done["run_file"].nunique()) if not log.empty else 0,
            }
        )
    print(json.dumps(out, indent=1))


def zero_rows(frame, target, value_cols):
    rows = []
    for game in target.itertuples():
        for team in (game.home_team, game.away_team):
            row = {"game_id": game.game_id, "team": team, "gameday": game.gameday}
            row.update({c: 0.0 for c in value_cols})
            rows.append(row)
    extra = pd.DataFrame(rows)
    return pd.concat([frame, extra], ignore_index=True) if not extra.empty else frame


def man_zone_terms(features, joined, target):
    classes = u5.MAN_ZONE
    off, dfn = u5.class_tables(joined, "defense_man_zone_type", classes)
    target = target.assign(gameday=pd.to_datetime(target["gameday"]))
    off = zero_rows(off, target, [c for c in off.columns if c not in ("game_id", "team", "gameday")])
    dfn = zero_rows(dfn, target, [c for c in dfn.columns if c not in ("game_id", "team", "gameday")])
    state = u5.rolling_state(off, dfn)
    frame = u5.matchup_terms(state, classes)
    terms = u5.side_terms(features, frame, classes, "cov_mz")
    value_cols = [c for c in terms.columns if c != "game_id"]
    pre = features["season"].to_numpy() < u5.FIRST_DATA_SEASON
    terms.loc[pre, value_cols] = 0.0
    terms[value_cols] = terms[value_cols].fillna(0.0)
    return terms


def served_forecast(week):
    candidates = sorted(FORECAST_ROOT.glob(f"{SEASON}-week-{week:02d}-*"))
    if not candidates:
        return None, None
    folder = candidates[-1]
    recs = pd.read_csv(folder / "recommendations.csv")
    pool = pd.read_csv(folder / "pool_card.csv")
    meta = json.loads((folder / "metadata.json").read_text())
    merged = recs.merge(
        pool[["game_id", "pool_side", "pick_probability"]], on="game_id", how="left"
    )
    return folder, merged.assign(created_at_utc=meta.get("created_at_utc"))


def side_of(p):
    if p > 0.5:
        return "HOME"
    if p < 0.5:
        return "AWAY"
    return "PASS"


def record(week, now):
    features = pd.read_parquet(u3.FEATURES)
    features["gameday"] = pd.to_datetime(features["gameday"])
    target = week_frame(features, week)
    if target.empty:
        raise SystemExit(f"no {SEASON} week {week} games in the feature table")
    refused = []
    eligible = []
    for row in target.itertuples():
        if pd.notna(row.result):
            refused.append({"game_id": row.game_id, "reason": "outcome_present"})
        elif now >= row.deadline:
            refused.append({"game_id": row.game_id, "reason": "past_deadline"})
        else:
            eligible.append(row.game_id)
    if not eligible:
        print(json.dumps({"week": week, "logged": [], "refused": refused}, indent=1))
        return
    pbp_snap = latest_pbp_snapshot(u5.PBP_ROOT)
    wanted = tuple(s for s in pbp_snap.seasons if s >= u5.FIRST_DATA_SEASON)
    pbp = load_pbp_snapshot(PbpSnapshot(pbp_snap.snapshot_id, pbp_snap.root, wanted))
    psnap = latest_participation_snapshot(u5.PARTICIPATION_ROOT)
    participation = load_participation_snapshot(psnap)
    participation = participation.loc[participation["season"] >= u5.FIRST_DATA_SEASON]
    schedules = pd.read_parquet(u3.SNAPSHOT / "schedules.parquet")
    joined = u5.play_classes(participation, pbp, schedules)
    terms = man_zone_terms(features, joined, target.loc[target["game_id"].isin(eligible)])
    merged = features.merge(terms, on="game_id", how="left", validate="one_to_one")
    columns = u5.arm_definitions()[ARM]
    original = FEATURE_SETS[u3.SET_NAME]
    FEATURE_SETS[u3.SET_NAME] = tuple(columns)
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

        predictions, discrete = _with_discrete_fallback(score, discrete, log)
    finally:
        FEATURE_SETS[u3.SET_NAME] = original
    arm = predictions.loc[predictions["method"].eq("market_residual")].set_index("game_id")
    folder, served = served_forecast(week)
    served = served.set_index("game_id") if served is not None else None
    recorded = pd.Timestamp(datetime.now(UTC))
    late = [g for g in eligible if recorded >= target.set_index("game_id").loc[g, "deadline"]]
    rows = []
    for game_id in eligible:
        if game_id in late:
            refused.append({"game_id": game_id, "reason": "past_deadline_at_write"})
            continue
        meta = target.set_index("game_id").loc[game_id]
        p = float(arm.loc[game_id, "home_cover_probability"])
        row = {
            "game_id": game_id,
            "season": SEASON,
            "week": week,
            "kickoff": meta["kickoff"].isoformat(),
            "deadline": meta["deadline"].isoformat(),
            "recorded_at_utc": recorded.isoformat(),
            "line": float(arm.loc[game_id, "spread_line"]),
            "arm_home_cover_probability": p,
            "arm_side": side_of(p),
            "terms": {c: float(terms.set_index("game_id").loc[game_id, c]) for c in terms.columns if c != "game_id"},
        }
        if served is not None and game_id in served.index:
            sp = float(served.loc[game_id, "home_cover_probability"])
            row["served_home_cover_probability"] = sp
            row["served_side"] = side_of(sp)
            row["served_pool_side"] = None if pd.isna(served.loc[game_id, "pool_side"]) else str(served.loc[game_id, "pool_side"])
            row["served_line"] = float(served.loc[game_id, "spread_line"])
        rows.append(row)
    if not rows:
        print(json.dumps({"week": week, "logged": [], "refused": refused}, indent=1))
        return
    payload = {
        "schema": "mod24_forward_log/1",
        "arm": ARM,
        "season": SEASON,
        "week": week,
        "recorded_at_utc": recorded.isoformat(),
        "script_sha256": sha256_file(__file__),
        "feature_table_sha256": sha256_file(u3.FEATURES),
        "participation_snapshot": psnap.snapshot_id,
        "participation_max_season": int(participation["season"].max()),
        "pbp_snapshot": pbp_snap.snapshot_id,
        "served_forecast_dir": None if folder is None else folder.name,
        "frozen": {
            "window_games": u5.WINDOW,
            "shrink_plays": u5.SHRINK_PLAYS,
            "ridge_alpha": u3.BASE_ALPHA,
            "probability_method": PROBABILITY_METHOD,
            "columns": len(columns),
        },
        "refused": refused,
        "rows": rows,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = recorded.strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"{SEASON}-week-{week:02d}-{stamp}.json"
    with open(path, "x") as handle:
        handle.write(json.dumps(payload, indent=1, default=str))
    print(json.dumps({"path": str(path), "logged": [r["game_id"] for r in rows], "refused": refused}, indent=1))


def main():
    parser = argparse.ArgumentParser(description="Forward log of the MOD-23 unit-5a man/zone arm")
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
        return
    now = pd.Timestamp(datetime.now(UTC))
    week = args.week
    if week is None:
        features = pd.read_parquet(u3.FEATURES, columns=["season", "week", "kickoff"])
        week = current_week(features, now)
        if week is None:
            print("no upcoming 2026 week in range")
            return
    if not FIRST_WEEK <= week <= LAST_WEEK:
        raise SystemExit("week outside the logged range")
    record(week, now)


if __name__ == "__main__":
    main()
