from __future__ import annotations

import bisect
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.features import _kickoff_utc
from nfl_ats.pbp import load_pbp_snapshot
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root
from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock
from nfl_ats.players import (
    _normalized_player_name,
    attach_snap_player_ids,
    canonicalize_rosters,
    canonicalize_snaps,
    load_player_snapshot,
    player_snapshot_from_root,
)
from nfl_ats.quarterbacks import TEAM_ABBREVIATION_ALIASES, build_qb_game_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/g55_u1"
PLAYER_SNAPSHOT = ROOT / "data/players/raw/20261108T141611Z"
PBP_SNAPSHOT = ROOT / "data/pbp/raw/20261111T010519Z"
DEPTH = ROOT / "data/players/raw/depth_charts/20260905T152519Z/depth_charts.parquet"
BASE_FEATURES = ROOT / "data/processed/game_features_pbp.parquet"
SERVED_FEATURES = ROOT / "data/processed/game_features_weak_stack.parquet"
INACTIVES = ROOT / "data/players/inactives"
INACTIVES_POSTED = pd.Timedelta(minutes=90)
EASTERN = "America/New_York"
NIGHT_WINDOW_MINUTES = 19 * 60
SEASONS = tuple(range(2020, 2027))
GRADE_SEASONS = tuple(range(2020, 2026))
LIVE_SEASON = 2026


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def slot(kickoff, lock):
    local = kickoff.tz_convert(EASTERN)
    day = local.day_name()
    if day == "Thursday":
        return "Thu"
    if day == "Saturday":
        return "Sat"
    if day == "Monday":
        return "MNF"
    if day == "Sunday":
        if kickoff < lock:
            return "Sun early"
        if local.hour * 60 + local.minute < NIGHT_WINDOW_MINUTES:
            return "Sun late"
        return "SNF"
    return "other"


def load_games():
    base = pd.read_parquet(
        BASE_FEATURES,
        columns=[
            "game_id",
            "season",
            "week",
            "gameday",
            "gametime",
            "home_team",
            "away_team",
            "game_type",
        ],
    )
    order = base[["game_id", "gameday"]].assign(gameday=lambda f: pd.to_datetime(f["gameday"]))
    games = base.loc[base["season"].isin(SEASONS) & base["game_type"].eq("REG")].copy()
    games["kickoff_utc"] = _kickoff_utc(games)
    games = games.loc[games["kickoff_utc"].notna()].copy()
    locks = (
        games.groupby(["season", "week"])["kickoff_utc"]
        .apply(sunday_pick_lock)
        .rename("sunday_lock_utc")
        .reset_index()
    )
    games = games.merge(locks, on=["season", "week"], how="left")
    games["deadline_utc"] = [
        pick_deadline(k, s)
        for k, s in zip(games["kickoff_utc"], games["sunday_lock_utc"], strict=False)
    ]
    games["deadline_utc"] = pd.to_datetime(games["deadline_utc"], utc=True)
    games["inactives_posted_utc"] = games["kickoff_utc"] - INACTIVES_POSTED
    games["deadline_after_inactives"] = games["deadline_utc"] > games["inactives_posted_utc"]
    games["deadline_ge_kickoff"] = games["deadline_utc"] >= games["kickoff_utc"]
    games["slot"] = [
        slot(k, s) for k, s in zip(games["kickoff_utc"], games["sunday_lock_utc"], strict=False)
    ]
    games["gameday"] = pd.to_datetime(games["gameday"])
    sides = []
    for side, _other in (("home_team", "away_team"), ("away_team", "home_team")):
        part = games.copy()
        part["team"] = part[side].astype(str)
        part["is_home"] = side == "home_team"
        sides.append(part)
    return games, pd.concat(sides, ignore_index=True), order


def status_agreement(rows, snaps, rosters):
    out = {}
    grade = rows.loc[rows["season"].isin(GRADE_SEASONS) & rows["has_snaps"]]
    played_snaps = snaps.loc[snaps["played"] & snaps["season"].isin(GRADE_SEASONS)]
    roster_keys = set(
        zip(
            rosters["season"],
            rosters["week"],
            rosters["team"].astype(str),
            rosters["gsis_id"],
            strict=False,
        )
    )
    for season, part in grade.groupby("season"):
        played = part["played"].fillna(False).astype(bool)
        act = part["status"].eq("ACT")
        ina = part["status"].eq("INA")
        team_games = part.groupby(["game_id", "team"]).ngroups
        ps = played_snaps.loc[played_snaps["season"].eq(season)]
        mapped = ps.loc[ps["gsis_id"].notna()]
        missing = sum(
            (s, w, t, g) not in roster_keys
            for s, w, t, g in zip(
                mapped["season"],
                mapped["week"],
                mapped["team"].astype(str),
                mapped["gsis_id"],
                strict=False,
            )
        )
        out[int(season)] = {
            "team_games": int(team_games),
            "act_per_team_game": round(float(act.sum()) / team_games, 2),
            "ina_per_team_game": round(float(ina.sum()) / team_games, 2),
            "ina_rows": int(ina.sum()),
            "ina_with_snaps": int((ina & played).sum()),
            "act_rows": int(act.sum()),
            "act_without_snaps": int((act & ~played).sum()),
            "played_on_roster": int(played.sum()),
            "played_status_act": int((played & act).sum()),
            "played_status_ina": int((played & ina).sum()),
            "played_status_other": int((played & ~act & ~ina).sum()),
            "played_snap_rows": len(ps),
            "played_unmapped_gsis": int(ps["gsis_id"].isna().sum()),
            "played_missing_from_week_roster": int(missing),
        }
    return out


def o1_agreement(rows):
    cand = rows.loc[rows["season"].isin(GRADE_SEASONS) & rows["has_snaps"] & rows["prior_role"]]
    oracle_out = ~cand["played"].fillna(False).astype(bool)
    servable_out = ~cand["status"].eq("ACT")
    table = {}
    for season, idx in cand.groupby("season").groups.items():
        o = oracle_out.loc[idx]
        s = servable_out.loc[idx]
        table[int(season)] = {
            "candidates": len(idx),
            "both_out": int((o & s).sum()),
            "oracle_out_servable_in": int((o & ~s).sum()),
            "oracle_in_servable_out": int((~o & s).sum()),
            "both_in": int((~o & ~s).sum()),
        }
    o, s = oracle_out, servable_out
    table["all"] = {
        "candidates": len(cand),
        "both_out": int((o & s).sum()),
        "oracle_out_servable_in": int((o & ~s).sum()),
        "oracle_in_servable_out": int((~o & s).sum()),
        "both_in": int((~o & ~s).sum()),
        "agreement": round(float((o == s).mean()), 4),
        "servable_recall_of_oracle_out": round(float((o & s).sum() / max(o.sum(), 1)), 4),
        "servable_precision": round(float((o & s).sum() / max(s.sum(), 1)), 4),
    }
    return table


def incumbent_starters(team_games, act_qbs, qb, order):
    qb = qb.merge(order, on="game_id", how="inner")
    history = defaultdict(list)
    for (team, gameday), grp in qb.groupby(["team", "gameday"], sort=True):
        history[team].append(
            (gameday, dict(zip(grp["player_id"], grp["qb_dropbacks"], strict=False)))
        )
    days = {team: [d for d, _ in items] for team, items in history.items()}
    result = {}
    for game_id, team, gameday in zip(
        team_games["game_id"], team_games["team"], team_games["gameday"], strict=False
    ):
        actives = act_qbs.get((game_id, team), set())
        items = history.get(team, [])
        pos = bisect.bisect_left(days.get(team, []), gameday)
        pick = None
        for _, counts in reversed(items[:pos]):
            live = {p: c for p, c in counts.items() if p in actives and c > 0}
            if live:
                pick = max(sorted(live), key=lambda p: live[p])
                break
        result[(game_id, team)] = pick
    return result


def live_captures(rows, tg):
    frames = []
    for path in sorted(INACTIVES.glob("*/inactives.parquet")):
        frame = pd.read_parquet(path)
        if len(frame):
            frames.append(frame)
    cap = pd.concat(frames, ignore_index=True)
    statuses = cap["status"].value_counts().to_dict()
    cap = cap.loc[cap["season"].eq(LIVE_SEASON)].copy()
    cap["team"] = cap["team"].replace(TEAM_ABBREVIATION_ALIASES).astype(str)
    cap["captured_at_utc"] = pd.to_datetime(cap["captured_at_utc"], utc=True)
    cap["normalized_name"] = cap["player_name"].map(_normalized_player_name)
    cap = cap.merge(
        tg[["game_id", "team", "deadline_utc", "inactives_posted_utc", "week"]].rename(
            columns={"week": "game_week"}
        ),
        on=["game_id", "team"],
        how="left",
    )
    cap = cap.loc[cap["captured_at_utc"] <= cap["deadline_utc"]]
    live = rows.loc[rows["season"].eq(LIVE_SEASON)].copy()
    live["normalized_name"] = live["full_name"].map(_normalized_player_name)
    names = (
        live[["game_id", "team", "normalized_name", "gsis_id", "status"]]
        .rename(columns={"status": "roster_status"})
        .drop_duplicates(["game_id", "team", "normalized_name"])
    )
    cap = cap.merge(names, on=["game_id", "team", "normalized_name"], how="left")
    pre = tg.loc[
        tg["season"].eq(LIVE_SEASON) & tg["week"].between(1, 10) & tg["deadline_after_inactives"]
    ]
    captured = cap.groupby(["game_id", "team"])
    roster_weeks = sorted(live["week"].unique().tolist())
    with_roster = set(zip(live["game_id"], live["team"], strict=False))
    keys = set(captured.groups)
    overlap = keys & with_roster
    capture_names = cap.loc[cap.set_index(["game_id", "team"]).index.isin(list(overlap))]
    roster_ina = live.loc[
        live["status"].eq("INA") & live.set_index(["game_id", "team"]).index.isin(list(overlap))
    ]
    matched = capture_names["gsis_id"].notna()
    hit = set(
        zip(capture_names["game_id"], capture_names["team"], capture_names["gsis_id"], strict=False)
    )
    ina_keys = set(
        zip(roster_ina["game_id"], roster_ina["team"], roster_ina["gsis_id"], strict=False)
    )
    by_week = (
        pd.Series(
            [g.split("_")[1] for g, _ in keys & set(zip(pre["game_id"], pre["team"], strict=False))]
        )
        .astype(int)
        .value_counts()
        .sort_index()
    )
    return {
        "capture_status_values": {str(k): int(v) for k, v in statuses.items()},
        "team_games_wk1_10_deadline_after_inactives": len(pre),
        "team_games_wk1_10_with_pre_deadline_capture": len(
            keys & set(zip(pre["game_id"], pre["team"], strict=False))
        ),
        "pre_deadline_capture_by_week": {str(k): int(v) for k, v in by_week.items()},
        "roster_weeks_2026": roster_weeks,
        "team_games_capture_and_roster": len(overlap),
        "captured_rows_in_overlap": len(capture_names),
        "captured_name_matched": int(matched.sum()),
        "captured_matched_status": capture_names.loc[matched, "roster_status"]
        .value_counts()
        .to_dict(),
        "roster_ina_in_overlap": len(ina_keys),
        "roster_ina_captured": len(ina_keys & hit),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    games, tg, order = load_games()
    _inj, rosters_raw, snaps_raw = load_player_snapshot(player_snapshot_from_root(PLAYER_SNAPSHOT))
    rosters = canonicalize_rosters(rosters_raw)
    rosters["team"] = rosters["team"].astype(str)
    snaps = attach_snap_player_ids(canonicalize_snaps(snaps_raw), rosters)
    snaps["team"] = snaps["team"].astype(str)
    snaps["played"] = (snaps["offense_snaps"] + snaps["defense_snaps"] + snaps["st_snaps"]) > 0
    snaps["role"] = (snaps["offense_pct"] + snaps["defense_pct"]) > 0
    snaps["key"] = snaps["season"] * 100 + snaps["week"]
    played = (
        snaps.loc[snaps["gsis_id"].notna()]
        .groupby(["game_id", "team", "gsis_id"])["played"]
        .max()
        .reset_index()
    )
    snap_team_games = set(zip(snaps["game_id"], snaps["team"], strict=False))
    first_role = (
        snaps.loc[snaps["role"] & snaps["gsis_id"].notna()]
        .groupby(["team", "gsis_id"])["key"]
        .min()
        .rename("first_role_key")
        .reset_index()
    )
    cols = ["game_id", "season", "week", "team", "gameday", "kickoff_utc", "deadline_utc"]
    cols += ["inactives_posted_utc", "deadline_after_inactives", "slot"]
    rows = tg[cols].merge(
        rosters[["season", "week", "team", "gsis_id", "position", "status", "full_name"]],
        on=["season", "week", "team"],
        how="inner",
    )
    rows = rows.merge(played, on=["game_id", "team", "gsis_id"], how="left")
    rows = rows.merge(first_role, on=["team", "gsis_id"], how="left")
    rows["prior_role"] = rows["first_role_key"] < rows["season"] * 100 + rows["week"]
    rows["has_snaps"] = [
        k in snap_team_games for k in zip(rows["game_id"], rows["team"], strict=False)
    ]
    status = status_agreement(rows, snaps, rosters)
    o1 = o1_agreement(rows)

    pbp = load_pbp_snapshot(pbp_snapshot_from_root(PBP_SNAPSHOT))
    pbp = pbp.loc[pd.to_numeric(pbp["season"], errors="coerce") >= min(SEASONS) - 1]
    drop = pbp.loc[
        pbp["qb_dropback"].eq(1) & pbp["passer_player_id"].notna(),
        ["game_id", "posteam", "play_id", "passer_player_id"],
    ]
    first_snap = (
        drop.sort_values(["game_id", "posteam", "play_id"])
        .drop_duplicates(["game_id", "posteam"])
        .rename(columns={"posteam": "team", "passer_player_id": "first_snap_qb_id"})
    )
    first_snap["team"] = first_snap["team"].replace(TEAM_ABBREVIATION_ALIASES).astype(str)
    qb = build_qb_game_metrics(pbp)
    team_col = "team" if "team" in qb.columns else "posteam"
    qb = qb.rename(columns={team_col: "team"})
    qb["team"] = qb["team"].replace(TEAM_ABBREVIATION_ALIASES).astype(str)
    qb = qb.groupby(["game_id", "team", "player_id"], as_index=False)["qb_dropbacks"].sum()

    qbs = rows.loc[rows["position"].eq("QB") & rows["status"].eq("ACT")]
    act_qbs = qbs.groupby(["game_id", "team"])["gsis_id"].agg(set).to_dict()
    incumbent = incumbent_starters(tg, act_qbs, qb, order)
    depth = pd.read_parquet(DEPTH)
    depth = depth.loc[depth["position"].eq("QB")].copy()
    depth["team"] = depth["team"].replace(TEAM_ABBREVIATION_ALIASES).astype(str)
    depth["depth_rank"] = pd.to_numeric(depth["depth_rank"], errors="coerce")
    depth = depth.merge(
        qbs[["game_id", "season", "week", "team", "gsis_id"]],
        on=["season", "week", "team", "gsis_id"],
        how="inner",
    )
    depth_first = (
        depth.sort_values(["game_id", "team", "depth_rank", "gsis_id"])
        .drop_duplicates(["game_id", "team"])
        .set_index(["game_id", "team"])["gsis_id"]
        .to_dict()
    )
    served = pd.read_parquet(SERVED_FEATURES)
    projected = {}
    for side in ("home", "away"):
        for g, t, p in zip(
            served["game_id"],
            served[f"{side}_team"],
            served[f"{side}_projected_qb_id"],
            strict=False,
        ):
            projected[(g, str(t))] = None if pd.isna(p) else str(p)
    totals = qb.groupby(["game_id", "team"])["qb_dropbacks"].sum().to_dict()
    per_qb = qb.set_index(["game_id", "team", "player_id"])["qb_dropbacks"].to_dict()
    fs = first_snap.set_index(["game_id", "team"])["first_snap_qb_id"].to_dict()

    qrows = []
    for g, t, season, known in zip(
        tg["game_id"], tg["team"], tg["season"], tg["deadline_after_inactives"], strict=False
    ):
        key = (g, t)
        dc = depth_first.get(key)
        inc = incumbent.get(key)
        starter = dc if dc is not None else inc
        proj = projected.get(key)
        tot = totals.get(key, np.nan)
        share = per_qb.get((g, t, proj), 0.0) / tot if proj and tot and tot > 0 else np.nan
        actives = act_qbs.get(key, set())
        qrows.append(
            {
                "game_id": g,
                "team": t,
                "season": int(season),
                "known": bool(known),
                "has_roster": key in act_qbs,
                "depth_starter": dc,
                "incumbent_starter": inc,
                "starter_qb_id": starter,
                "starter_source": "depth_chart"
                if dc is not None
                else ("incumbent" if inc else None),
                "projected_qb_id": proj,
                "projected_active": (proj in actives) if proj and key in act_qbs else None,
                "first_snap_qb_id": fs.get(key),
                "o3_share": share,
            }
        )
    q = pd.DataFrame(qrows)
    starters = q.set_index(["game_id", "team"])["starter_qb_id"].to_dict()

    def rate(frame, a, b):
        ok = frame[a].notna() & frame[b].notna()
        return {"n": int(ok.sum()), "agree": int((frame.loc[ok, a] == frame.loc[ok, b]).sum())}

    qb_report = {}
    for label, part in (
        ("2020_2025", q.loc[q["season"].isin(GRADE_SEASONS) & q["has_roster"]]),
        ("2026", q.loc[q["season"].eq(LIVE_SEASON) & q["has_roster"]]),
    ):
        o3_nonzero = part["o3_share"].notna() & (part["o3_share"] < 1)
        flag = part["projected_active"].eq(False) | (
            part["starter_qb_id"].notna() & (part["starter_qb_id"] != part["projected_qb_id"])
        )
        gap = (1 - part["o3_share"]).where(part["o3_share"].notna(), 0.0)
        qb_report[label] = {
            "team_games": len(part),
            "starter_source": part["starter_source"].value_counts(dropna=False).to_dict(),
            "depth_v_first_snap": rate(part, "depth_starter", "first_snap_qb_id"),
            "incumbent_v_first_snap": rate(part, "incumbent_starter", "first_snap_qb_id"),
            "starter_v_first_snap": rate(part, "starter_qb_id", "first_snap_qb_id"),
            "projected_v_first_snap": rate(part, "projected_qb_id", "first_snap_qb_id"),
            "starter_v_projected": rate(part, "starter_qb_id", "projected_qb_id"),
            "projected_inactive": int(part["projected_active"].eq(False).sum()),
            "o3_nonzero": int(o3_nonzero.sum()),
            "servable_flag": int(flag.sum()),
            "o3_nonzero_and_flag": int((o3_nonzero & flag).sum()),
            "o3_mass_share_on_flagged": round(float(gap[flag].sum() / max(gap.sum(), 1e-12)), 4),
            "o3_share_lt_half": int((part["o3_share"] < 0.5).sum()),
            "o3_share_lt_half_and_flag": int(((part["o3_share"] < 0.5) & flag).sum()),
        }

    slot_counts = {}
    for label, part in (
        ("2020_2025", games.loc[games["season"].isin(GRADE_SEASONS)]),
        ("2026", games.loc[games["season"].eq(LIVE_SEASON)]),
    ):
        grp = part.groupby("slot")
        slot_counts[label] = {
            s: {
                "games": len(p),
                "deadline_after_inactives": int(p["deadline_after_inactives"].sum()),
                "deadline_ge_kickoff": int(p["deadline_ge_kickoff"].sum()),
            }
            for s, p in grp
        }

    live = live_captures(rows, tg)

    known = rows["deadline_after_inactives"].astype(bool)
    table = pd.DataFrame(
        {
            "game_id": rows["game_id"],
            "season": rows["season"].astype(int),
            "week": rows["week"].astype(int),
            "team": rows["team"],
            "gsis_id": rows["gsis_id"].astype(str),
            "position": rows["position"],
            "status": rows["status"],
            "prior_role": rows["prior_role"].astype(bool),
            "active": rows["status"].eq("ACT").astype("boolean").where(known, pd.NA),
            "known_at_deadline": known,
            "starter_qb": pd.Series(
                [
                    starters.get((g, t)) == p
                    for g, t, p in zip(rows["game_id"], rows["team"], rows["gsis_id"], strict=False)
                ],
                index=rows.index,
                dtype="boolean",
            ).where(known, pd.NA),
            "kickoff_utc": rows["kickoff_utc"],
            "deadline_utc": rows["deadline_utc"],
            "slot": rows["slot"],
        }
    )
    table_path = OUT / "deadline_actives.parquet"
    table.to_parquet(table_path, index=False)
    qb_path = OUT / "starter_qb_team_games.parquet"
    q.drop(columns=["o3_share", "first_snap_qb_id"]).to_parquet(qb_path, index=False)
    summary = {
        "status_agreement": status,
        "o1_agreement": o1,
        "qb": qb_report,
        "slots": slot_counts,
        "live_2026": live,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    sources = {
        "weekly_rosters": PLAYER_SNAPSHOT / "weekly_rosters.parquet",
        "snap_counts": PLAYER_SNAPSHOT / "snap_counts.parquet",
        "depth_charts": DEPTH,
        "pbp_manifest": PBP_SNAPSHOT / "manifest.json",
        "schedule_features": BASE_FEATURES,
        "served_features": SERVED_FEATURES,
    }
    captures = sorted(INACTIVES.glob("*/inactives.parquet"))
    manifest = {
        "script": "scripts/g55_u1_deadline_actives.py",
        "sources": {
            k: {"path": str(v.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(v)}
            for k, v in sources.items()
        },
        "live_inactives_captures": {
            "count": len(captures),
            "sha256_of_file_hashes": hashlib.sha256(
                "".join(sha256(p) for p in captures).encode()
            ).hexdigest(),
        },
        "as_of_basis": {
            "deadline": "pick_refresh.pick_deadline(kickoff, sunday_pick_lock(week kickoffs))",
            "inactives_posted": "kickoff - 90 minutes (reported: RotoWire FAQ, "
            "docs/inactives_channel.md)",
            "known_at_deadline": "deadline_utc > kickoff_utc - 90 minutes",
            "active": "same-week nflverse weekly_rosters status == ACT; "
            "NA when not known_at_deadline",
            "starter_qb": "depth chart rank-1 QB among same-week ACT QBs, else the ACT QB with "
            "most dropbacks in the team's latest prior game with an ACT QB dropback; "
            "NA when not known",
            "historical_note": "weekly_rosters is a post-hoc record of the gameday list; servable "
            "forward only through the live inactives capture",
        },
        "row_counts": {
            "deadline_actives": len(table),
            "starter_qb_team_games": len(q),
            "known_at_deadline_rows": int(known.sum()),
        },
        "columns": table.columns.tolist(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
