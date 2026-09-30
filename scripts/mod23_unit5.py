import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod23_unit3 as u3

from nfl_ats.participation import latest_participation_snapshot, load_participation_snapshot
from nfl_ats.pbp import (
    PbpSnapshot,
    analysis_plays,
    latest_pbp_snapshot,
    load_pbp_snapshot,
)

ROOT = u3.ROOT
OUT_DIR = ROOT / "artifacts" / "mod23_unit5"
PARTICIPATION_ROOT = ROOT / "data" / "players" / "participation" / "raw"
PBP_ROOT = ROOT / "data" / "pbp" / "raw"
WINDOW = 16
SHRINK_PLAYS = 100.0
MAN_ZONE = ("MAN_COVERAGE", "ZONE_COVERAGE")
COVERS = ("COVER_0", "COVER_1", "COVER_2", "COVER_3", "COVER_4", "COVER_6")
FIRST_DATA_SEASON = 2018
SIDES = ("home", "away", "diff")


def coverage_availability(participation):
    rows = {}
    for season, g in participation.groupby("season"):
        mz = g["defense_man_zone_type"].isin(MAN_ZONE)
        cv = g["defense_coverage_type"].isin(COVERS)
        rows[int(season)] = {
            "plays": len(g),
            "man_zone_nonnull": float(mz.mean()),
            "cover_type_nonnull": float(cv.mean()),
        }
    return rows


def play_classes(participation, pbp, schedules):
    plays = analysis_plays(pbp)
    plays = plays.loc[
        plays["competitive_play"] & plays["epa"].notna(),
        ["game_id", "play_id", "posteam", "defteam", "epa"],
    ].copy()
    plays["play_id"] = pd.to_numeric(plays["play_id"], errors="raise").astype(int)
    cov = participation[
        ["game_id", "play_id", "defense_man_zone_type", "defense_coverage_type"]
    ].copy()
    cov["game_id"] = cov["game_id"].astype(str)
    plays["game_id"] = plays["game_id"].astype(str)
    joined = plays.merge(cov, on=["game_id", "play_id"], how="inner", validate="one_to_one")
    joined["posteam"] = joined["posteam"].astype(str)
    joined["defteam"] = joined["defteam"].astype(str)
    joined["defense_man_zone_type"] = joined["defense_man_zone_type"].astype(str)
    joined["defense_coverage_type"] = joined["defense_coverage_type"].astype(str)
    dates = schedules[["game_id", "gameday"]].drop_duplicates("game_id").copy()
    dates["game_id"] = dates["game_id"].astype(str)
    dates["gameday"] = pd.to_datetime(dates["gameday"])
    return joined.merge(dates, on="game_id", how="left", validate="many_to_one")


def class_tables(joined, column, classes):
    sub = joined.loc[joined[column].isin(classes)]
    off = sub.groupby(["game_id", "posteam", "gameday", column])["epa"].agg(["sum", "count"])
    off = off.unstack(column).reindex(columns=list(classes), level=1)
    off.columns = [f"{stat}_{cls}" for stat, cls in off.columns]
    off = off.fillna(0.0).reset_index().rename(columns={"posteam": "team"})
    for cls in classes:
        for stat in ("sum", "count"):
            if f"{stat}_{cls}" not in off:
                off[f"{stat}_{cls}"] = 0.0
    dfn = sub.groupby(["game_id", "defteam", "gameday", column]).size().unstack(column)
    dfn = dfn.reindex(columns=list(classes)).fillna(0.0)
    dfn.columns = [f"dn_{cls}" for cls in dfn.columns]
    dfn = dfn.reset_index().rename(columns={"defteam": "team"})
    return off, dfn


def rolling_state(off, dfn):
    table = off.merge(dfn, on=["game_id", "team", "gameday"], how="outer").fillna(0.0)
    table = table.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    value_cols = [c for c in table.columns if c not in ("game_id", "team", "gameday")]
    rolled = table.groupby("team")[value_cols].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=1).sum()
    )
    state = pd.concat([table[["game_id", "team", "gameday"]], rolled], axis=1)
    per_day = table.groupby("gameday")[value_cols].sum().sort_index()
    league = per_day.cumsum().shift(1)
    league.columns = [f"L_{c}" for c in value_cols]
    return state.join(league, on="gameday")


def matchup_terms(state, classes):
    dn = np.column_stack([state[f"dn_{c}"].fillna(0.0).to_numpy() for c in classes])
    l_dn = np.column_stack([state[f"L_dn_{c}"].to_numpy() for c in classes])
    league_share = l_dn / l_dn.sum(axis=1, keepdims=True)
    known = dn.sum(axis=1, keepdims=True)
    rates = (dn + SHRINK_PLAYS * league_share) / (known + SHRINK_PLAYS)
    sums = np.column_stack([state[f"sum_{c}"].fillna(0.0).to_numpy() for c in classes])
    counts = np.column_stack([state[f"count_{c}"].fillna(0.0).to_numpy() for c in classes])
    l_sum = np.column_stack([state[f"L_sum_{c}"].to_numpy() for c in classes])
    l_cnt = np.column_stack([state[f"L_count_{c}"].to_numpy() for c in classes])
    league_epa = l_sum / l_cnt
    epa = (sums + SHRINK_PLAYS * league_epa) / (counts + SHRINK_PLAYS)
    frame = pd.DataFrame({"game_id": state["game_id"].to_numpy(), "team": state["team"].to_numpy()})
    deviation = epa - league_epa
    for j, cls in enumerate(classes):
        frame[f"rate_{cls}"] = rates[:, j]
        frame[f"dev_{cls}"] = deviation[:, j]
    return frame


def side_terms(features, frame, classes, label):
    games = features[["game_id", "home_team", "away_team"]]
    by_team = frame.drop_duplicates(["game_id", "team"]).set_index(["game_id", "team"])
    out = games[["game_id"]].copy()
    for offense_side, defense_side in (("home", "away"), ("away", "home")):
        keys_off = pd.MultiIndex.from_arrays([games["game_id"], games[f"{offense_side}_team"]])
        keys_def = pd.MultiIndex.from_arrays([games["game_id"], games[f"{defense_side}_team"]])
        off = by_team.reindex(keys_off)
        dfn = by_team.reindex(keys_def)
        total = np.zeros(len(games))
        for cls in classes:
            total = total + dfn[f"rate_{cls}"].to_numpy() * off[f"dev_{cls}"].to_numpy()
        out[f"{offense_side}_{label}"] = total
    out[f"diff_{label}"] = out[f"home_{label}"] - out[f"away_{label}"]
    return out


def build_terms(features, joined):
    outputs = []
    for label, column, classes in (
        ("cov_mz", "defense_man_zone_type", MAN_ZONE),
        ("cov_ct", "defense_coverage_type", COVERS),
    ):
        off, dfn = class_tables(joined, column, classes)
        state = rolling_state(off, dfn)
        frame = matchup_terms(state, classes)
        outputs.append(side_terms(features, frame, classes, label))
    terms = outputs[0].merge(outputs[1], on="game_id", validate="one_to_one")
    for side in SIDES:
        terms[f"{side}_cov_both"] = terms[f"{side}_cov_mz"] + terms[f"{side}_cov_ct"]
    value_cols = [c for c in terms.columns if c != "game_id"]
    pre = features["season"].to_numpy() < FIRST_DATA_SEASON
    terms.loc[pre, value_cols] = 0.0
    terms[value_cols] = terms[value_cols].fillna(0.0)
    return terms


def arm_definitions():
    base = list(u3.FULL_COLUMNS)
    mz = [f"{s}_cov_mz" for s in SIDES]
    ct = [f"{s}_cov_ct" for s in SIDES]
    return {
        "arm_a_man_zone": base + mz,
        "arm_b_coverage_type": base + ct,
        "arm_c_both": base + mz + ct,
    }


def corr_block(frame, column):
    sub = frame.loc[frame["margin_vs_open"].notna()].reset_index(drop=True)
    x = sub[column].to_numpy(dtype=float)
    y = sub["margin_vs_open"].to_numpy(dtype=float)
    point = float(np.corrcoef(x, y)[0, 1])
    groups = [g.index.to_numpy() for _, g in sub.groupby(["season", "week"])]
    rng = np.random.default_rng(u3.BOOTSTRAP_SEED)
    draws = []
    for _ in range(u3.BOOTSTRAP_SAMPLES):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(np.corrcoef(x[idx], y[idx])[0, 1])
    draws = np.array(draws)
    return {
        "correlation": point,
        "low": float(np.percentile(draws, 2.5)),
        "high": float(np.percentile(draws, 97.5)),
        "probability_positive": float((draws > 0).mean()),
        "games": len(sub),
        "blocks": len(groups),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(u3.FEATURES)
    print("feature sha", hashlib.sha256(u3.FEATURES.read_bytes()).hexdigest())
    schedules = pd.read_parquet(u3.SNAPSHOT / "schedules.parquet")
    psnap = latest_participation_snapshot(PARTICIPATION_ROOT)
    participation = load_participation_snapshot(psnap)
    availability = coverage_availability(participation)
    print("participation", psnap.snapshot_id, json.dumps(availability, indent=1), flush=True)
    pbp_snap = latest_pbp_snapshot(PBP_ROOT)
    wanted = tuple(s for s in pbp_snap.seasons if s >= FIRST_DATA_SEASON)
    pbp = load_pbp_snapshot(PbpSnapshot(pbp_snap.snapshot_id, pbp_snap.root, wanted))
    participation = participation.loc[participation["season"] >= FIRST_DATA_SEASON]
    joined = play_classes(participation, pbp, schedules)
    print("joined plays", len(joined), "missing gameday", int(joined["gameday"].isna().sum()))
    terms = build_terms(features, joined)
    terms.to_parquet(OUT_DIR / "terms.parquet")
    features = features.merge(terms, on="game_id", how="left", validate="one_to_one")
    scored_base = u3.run_arm(features, u3.FULL_COLUMNS)
    base = u3.decisive(scored_base)
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
    summaries = {}
    arms = arm_definitions()
    for name, columns in arms.items():
        arm = u3.decisive(u3.run_arm(features, columns))
        arm.to_parquet(OUT_DIR / f"{name}_per_game.parquet")
        summaries[name] = u3.summarize(name, arm, base)
        summaries[name]["n_columns"] = len(columns)
        print(name, summaries[name]["record"], summaries[name]["diff_points"], flush=True)
    joined_scored = scored_base.merge(terms, on="game_id", how="left")
    predictive = {
        column: corr_block(joined_scored, column)
        for column in ("diff_cov_mz", "diff_cov_ct", "diff_cov_both")
    }
    report = {
        "availability": availability,
        "participation_snapshot": psnap.snapshot_id,
        "window_games": WINDOW,
        "shrink_plays": SHRINK_PLAYS,
        "summaries": summaries,
        "predictive_correlation": predictive,
        "looks": len(arms),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(
        json.dumps({k: v for k, v in report.items() if k != "availability"}, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
