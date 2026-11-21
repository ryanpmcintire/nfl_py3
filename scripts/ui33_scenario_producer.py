import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead106_forward_score as F
import lead106_injury_language as L6
import lead108_forward as LF
import lead108_full_mixture as M
import lead108_mixture_forward as MF
import lead108_wording_mixture as U1
import lead110_absorption as A
import mod25_wf_forward as mwf

ROOT = L6.ROOT
SEASON = 2026
OUT = ROOT / "artifacts" / "ui33_scenarios"
STAMP = LF.STAMP
UNRESOLVED = ("Questionable", "Doubtful")
PENDING_PRACTICE = (
    "Did Not Participate In Practice",
    "Limited Participation in Practice",
)
STARTER_SHARE = 0.5
CODE = (
    "scripts/ui33_scenario_producer.py",
    "scripts/lead108_full_mixture.py",
    "scripts/lead108_mixture_forward.py",
)


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def active_model():
    ats = json.loads((ROOT / "artifacts" / "active_ats_model.json").read_text())
    pp = json.loads((ROOT / "artifacts" / "active_pick_probability.json").read_text())
    if pp.get("active_model_id") != ats["model_id"]:
        raise SystemExit("active pick probability is keyed to a different model than the ATS model")
    return {
        "active_model_id": ats["model_id"],
        "active_forecast_artifact": ats["weekly_forecast"]["artifact"],
        "active_forecast_week": ats["weekly_forecast"]["week"],
        "pick_probability_artifact": pp["artifact"],
        "pick_probability_policy": pp["policy"],
    }


def injury_capture(week, now):
    stamp = pd.Timestamp(now).strftime(STAMP)
    caps = sorted(
        d for d in LF.INJ.iterdir() if d.name <= stamp and (d / "injuries.parquet").exists()
    )
    if not caps:
        raise SystemExit("no injury capture before now")
    cap = caps[-1]
    inj = pd.read_parquet(cap / "injuries.parquet")
    inj = inj[(inj["season"] == SEASON) & (inj["week"] == week)]
    inj = inj.drop_duplicates("gsis_id", keep="last").copy()
    inj["designation"] = inj["report_status"].fillna("none")
    inj["practice_status"] = inj["practice_status"].fillna("none")
    inj["grp"] = inj["position"].map(A.POS_TO_GROUP)
    return inj.reset_index(drop=True), cap.name


def starter_shares(ids, week, now):
    _, pmap = L6.roster_map()
    for w in range(week - 1, 0, -1):
        f = F.SNAPS / f"snap_counts_{SEASON}_before_week_{w + 1:02d}.parquet"
        if f.exists() and MF.mtime(f) <= pd.Timestamp(now):
            break
    else:
        raise SystemExit("no snap capture before now")
    s, path = F.snap_frame(w)
    s["sh"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s["sh"] = s["sh"] / (100.0 if s["sh"].max() > 1 else 1.0)
    last = s.sort_values("week").groupby("pfr_player_id").tail(1).set_index("pfr_player_id")
    out = [float(last.at[pmap[g], "sh"]) if pmap.get(g) in last.index else np.nan for g in ids]
    return np.array(out), path.name


def candidates(inj):
    designated = inj.groupby("team")["designation"].transform(lambda d: (d != "none").any())
    unresolved = inj["designation"].isin(UNRESOLVED)
    pending = ~designated & inj["practice_status"].isin(PENDING_PRACTICE)
    keep = (unresolved | pending) & (inj["starter_share"] > STARTER_SHARE) & inj["grp"].notna()
    out = inj[keep].copy()
    out["unresolved_kind"] = np.where(unresolved[keep], "designation", "designation_pending")
    return out


def scenario_frames(g, inj, cands):
    rows, players, keys = [], [], []
    key = 0
    for r in g.itertuples(index=False):
        both = inj[inj["team"].isin((r.home_team, r.away_team))]
        mine = cands[cands["team"].isin((r.home_team, r.away_team))]
        plan = [(None, None)] + [(i, s) for i in mine.index for s in ("in", "out")]
        for idx, scen in plan:
            key += 1
            rows.append(
                {
                    "home_team": r.home_team,
                    "away_team": r.away_team,
                    "week": key,
                    "p_served": r.p_served,
                    "open_line": r.open_line,
                }
            )
            keys.append((r.game_id, idx, scen))
            p = both[["team", "grp", "lp"]].copy()
            p["week"] = key
            p["rp"] = np.nan
            if idx is not None:
                p.loc[idx, "rp"] = 1.0 if scen == "in" else 0.0
            players.append(p)
    return pd.DataFrame(rows), pd.concat(players, ignore_index=True), keys


def sentence(name, scen, team):
    verb = "sits" if scen == "out" else "plays"
    return f"If {name} {verb}, the pick would change to {team}."


def cmd_mixture(a):
    now = datetime.now(UTC)
    model = active_model()
    week = MF.resolve(a.week, now)
    if week is None:
        print("SKIP: no unplayed week")
        return
    sched = LF.schedule(week)
    base = MF.card(week, now)
    pool = MF.splash(week, now)
    opens = mwf.tuesday_open(sched)
    games, skipped = [], {}
    for r in sched.itertuples():
        if r.cutoff_utc <= pd.Timestamp(now):
            skipped[r.game_id] = "past_cutoff"
            continue
        if r.game_id not in base.index:
            skipped[r.game_id] = "no_published_card"
            continue
        b = base.loc[r.game_id]
        op = opens.get(r.game_id, (np.nan, None))
        if r.game_id in pool:
            line, src = pool[r.game_id][0], "pool_splash"
        elif pd.notna(b["market_spread"]):
            line, src = float(b["market_spread"]), "card_market_spread"
        else:
            line, src = op[0], "tuesday_opener"
        games.append(
            {
                "game_id": r.game_id,
                "away_team": r.away_team,
                "home_team": r.home_team,
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "cutoff_utc": r.cutoff_utc.isoformat(),
                "pick_team": b["pick_team"],
                "card_published_at_utc": str(b["published_at_utc"]),
                "card_forecast_artifact": b["baseline_forecast_artifact"],
                "card_from_active_forecast": bool(
                    b["baseline_forecast_artifact"] == model["active_forecast_artifact"]
                ),
                "line_source": src,
                "open_line": float(line),
                "pool_line": pool.get(r.game_id, (None, None))[0],
                "card_line": float(b["market_spread"]),
                "tuesday_opener": op[0],
                "p_served": float(b["p_home"]),
            }
        )
    g = pd.DataFrame(games)
    if g.empty:
        print("no open game with a published card; nothing written")
        return
    inj, cap = injury_capture(week, now)
    inj = inj[inj["team"].isin(set(g["home_team"]) | set(g["away_team"]))].reset_index(drop=True)
    tab, lmeta = MF.ladder_table(week, now)
    inj["week"] = week
    inj["lp"] = [L6.ladder_pred(r, tab, U1.KEYS)[0] for _, r in inj.iterrows()]
    inj["starter_share"], snap_file = starter_shares(inj["gsis_id"], week, now)
    reads, read_files = MF.admissible_reads(week, sched, now)
    rmap = dict(zip(reads["gsis_id"], reads["rp_raw"], strict=True)) if len(reads) else {}
    inj["read_p_plays"] = inj["gsis_id"].map(rmap)
    cands = candidates(inj)
    hl, hm = MF.history_pool()
    val = json.loads((A.OUT / "report.json").read_text())["weights_spread_pts_per_out"]
    sg, sp, keys = scenario_frames(g, inj, cands)
    mx = M.mixtures(sg, sp, val, hl, hm)
    res = dict(zip(keys, mx.itertuples(index=False), strict=True))
    out_games, wrong_sign = [], 0
    for gm in games:
        gid = gm["game_id"]
        b0 = res[(gid, None, None)]
        served_home = gm["pick_team"] == gm["home_team"]
        mine = cands[cands["team"].isin((gm["home_team"], gm["away_team"]))]
        clist, lines = [], []
        for idx, c in mine.iterrows():
            pin = float(res[(gid, idx, "in")].p_mix_read)
            pout = float(res[(gid, idx, "out")].p_mix_read)
            home = c["team"] == gm["home_team"]
            wrong_sign += int((pin < pout) if home else (pin > pout))
            side_in = gm["home_team"] if pin >= 0.5 else gm["away_team"]
            side_out = gm["home_team"] if pout >= 0.5 else gm["away_team"]
            flagged = side_in != side_out
            text = ""
            if flagged:
                scen, team = ("out", side_out) if side_in == gm["pick_team"] else ("in", side_in)
                text = sentence(c["full_name"], scen, team)
                lines.append(text)
            clist.append(
                {
                    "player": c["full_name"],
                    "team": c["team"],
                    "position": c["position"],
                    "designation": c["designation"],
                    "practice_status": c["practice_status"],
                    "unresolved_kind": c["unresolved_kind"],
                    "starter_share": float(c["starter_share"]),
                    "ladder_play_probability": float(c["lp"]),
                    "read_play_probability": (
                        None if pd.isna(c["read_p_plays"]) else float(c["read_p_plays"])
                    ),
                    "value_pts_per_out": float(val[c["grp"]]),
                    "home_probability_in": pin,
                    "home_probability_out": pout,
                    "side_if_plays": side_in,
                    "side_if_sits": side_out,
                    "flagged": bool(flagged),
                    "sentence": text,
                }
            )
        both = inj[inj["team"].isin((gm["home_team"], gm["away_team"]))]
        out_games.append(
            {
                **{k: v for k, v in gm.items() if k not in ("open_line", "p_served")},
                "lattice_line": gm["open_line"],
                "base_home_probability": gm["p_served"],
                "served_side_home": bool(served_home),
                "ladder_mixture_home_probability": float(b0.p_mix_lad),
                "zero_shift_error": float(b0.zero_err),
                "center_ok": bool(b0.center_ok),
                "listed_players": len(both),
                "valued_players": int(b0.n_players),
                "candidates": clist,
                "flagged_players": [c["player"] for c in clist if c["flagged"]],
                "board_line": " ".join(lines),
            }
        )
    doc = {
        "row": "UI-33",
        "basis": "research_mixture_lead108",
        "purpose": "research only; not the served probability; never rendered",
        **model,
        "season": SEASON,
        "week": week,
        "created_at_utc": now.strftime(STAMP),
        "code_sha256": {c: sha(c) for c in CODE},
        "injury_capture": cap,
        "snap_file": snap_file,
        "value_source": str(A.OUT.relative_to(ROOT) / "report.json"),
        "starter_rule": f"max offense/defense snap share in latest prior game > {STARTER_SHARE}",
        "unresolved_rule": (
            "designation Questionable or Doubtful, or no designation yet for the team and "
            "practice DNP or Limited"
        ),
        "others_held_at": "ladder availability by designation x practice status",
        "read_files": read_files,
        "rows_read": len(reads),
        "ladder": lmeta,
        "skipped": skipped,
        "candidates_total": len(cands),
        "candidates_wrong_sign": wrong_sign,
        "flagged_games": [x["game_id"] for x in out_games if x["flagged_players"]],
        "games": out_games,
    }
    (OUT / "research_mixture").mkdir(parents=True, exist_ok=True)
    path = OUT / "research_mixture" / f"{SEASON}_week{week:02d}_{now.strftime(STAMP)}.json"
    path.write_text(json.dumps(doc, indent=1, default=str))
    print(path)
    print(
        f"model {model['active_model_id']} week {week} games {len(out_games)} "
        f"candidates {len(cands)} wrong_sign {wrong_sign} flagged {doc['flagged_games']}"
    )
    for x in out_games:
        print(
            f"{x['game_id']:<16} pick {x['pick_team']:<3} base {x['base_home_probability']:.4f} "
            f"line {x['lattice_line']:+5.1f} {x['line_source']:<18} active "
            f"{x['card_from_active_forecast']} err {x['zero_shift_error']:.1e} "
            f"listed {x['listed_players']}"
        )
        for c in x["candidates"]:
            print(
                f"    {c['player']:<24} {c['team']:<3} {c['position']:<3} "
                f"{c['designation']:<12} lp {c['ladder_play_probability']:.2f} "
                f"in {c['home_probability_in']:.4f} out {c['home_probability_out']:.4f} "
                f"{'FLAG ' + c['sentence'] if c['flagged'] else ''}"
            )


def _side_team(value, home, away):
    if value in (home, away):
        return value
    return {"home": home, "away": away}.get(str(value).lower(), str(value))


def _qb_view(state, debut):
    if state is None:
        return debut[0], debut[1], 0.0
    epa = state["state_qb_epa_per_dropback"]
    cpoe = state["state_qb_cpoe"]
    return (
        float(epa) if pd.notna(epa) else debut[0],
        float(cpoe) if pd.notna(cpoe) else debut[1],
        float(state["career_dropbacks"]),
    )


def cmd_served(a):
    import math
    import tempfile

    from nfl_ats import players as P
    from nfl_ats.cli_common import _resolve_pbp_snapshot, _resolve_player_snapshot
    from nfl_ats.constants import PLAYER_QB_STATE_METRICS
    from nfl_ats.pbp import load_pbp_snapshot
    from nfl_ats.pick_refresh import plan_refresh
    from nfl_ats.quarterbacks import (
        build_qb_game_metrics,
        build_qb_states,
        depth_snapshot_from_root,
        load_depth_snapshot,
    )
    from nfl_ats.weekly import CARD_PATH_TABLES

    now = datetime.now(UTC)
    model = active_model()
    week = MF.resolve(a.week, now)
    if week is None:
        print("SKIP: no unplayed week")
        return
    ats = json.loads((ROOT / "artifacts" / "active_ats_model.json").read_text())
    data_root = ROOT / "data"
    artifacts_root = ROOT / "artifacts"
    table = data_root / "processed" / CARD_PATH_TABLES[ats["feature_profile"]]
    manifest = json.loads(table.with_name(f"{table.stem}.manifest.json").read_text())
    feats = pd.read_parquet(table)
    feats["game_id"] = feats["game_id"].astype(str)
    sched = LF.schedule(week)
    base = MF.card(week, now)
    open_ids, skipped = [], {}
    for r in sched.itertuples():
        if r.cutoff_utc <= pd.Timestamp(now):
            skipped[r.game_id] = "past_cutoff"
        elif r.game_id not in base.index:
            skipped[r.game_id] = "no_published_card"
        else:
            open_ids.append(r.game_id)
    if not open_ids:
        print("no open game with a published card; nothing written")
        return
    wk = feats[feats["game_id"].isin(open_ids)].set_index("game_id")
    snap = _resolve_player_snapshot(manifest["source_player_snapshot"])
    snap_inj = pd.read_parquet(snap.injuries_path)
    snap_inj = snap_inj[(snap_inj["season"] == SEASON) & (snap_inj["week"] == week)].copy()
    snap_inj["gsis_id"] = snap_inj["gsis_id"].astype(str)
    latest_inj, latest_cap = injury_capture(week, now)
    latest_inj["gsis_id"] = latest_inj["gsis_id"].astype(str)
    depth = load_depth_snapshot(
        depth_snapshot_from_root(
            data_root / "quarterbacks" / "depth" / "raw" / manifest["source_depth_snapshot"]
        )
    )
    depth["observed_at_utc"] = pd.to_datetime(depth["observed_at_utc"], utc=True, errors="coerce")
    depth["pos_rank"] = pd.to_numeric(depth["pos_rank"], errors="coerce")
    depth = depth.dropna(subset=["observed_at_utc", "team", "gsis_id", "pos_rank"]).copy()
    depth["gsis_id"] = depth["gsis_id"].astype(str)
    names = dict(zip(depth["gsis_id"], depth["player_name"], strict=False))
    names.update(dict(zip(latest_inj["gsis_id"], latest_inj["full_name"], strict=False)))
    pbp = load_pbp_snapshot(_resolve_pbp_snapshot(manifest["source_pbp_snapshot"]))
    games_frame = feats.copy()
    games_frame["gameday"] = pd.to_datetime(games_frame["gameday"])
    qb_games = build_qb_game_metrics(pbp)
    qb_states = build_qb_states(
        qb_games,
        games_frame,
        span=a.qb_span,
        min_dropbacks=a.qb_min_dropbacks,
        offseason_retention=a.offseason_retention,
    )
    histories = {
        str(k): grp.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        for k, grp in qb_states.groupby("player_id", sort=False)
    }
    debut = P._debut_qb_fill(qb_states, [SEASON])[SEASON]
    gamedays = dict(zip(games_frame["game_id"], games_frame["gameday"], strict=False))
    qb_games = qb_games.assign(_gd=qb_games["game_id"].astype(str).map(gamedays))

    identity, cands = [], []
    for gid, row in wk.iterrows():
        gday = pd.Timestamp(row["gameday"])
        decision_at = pd.Timestamp(row["kickoff"]) - pd.Timedelta(hours=a.decision_hours)
        for side in ("home", "away"):
            team = str(row[f"{side}_team"])
            pid = row[f"{side}_projected_qb_id"]
            if pd.isna(pid):
                continue
            pid = str(pid)
            epa, cpoe, exp = _qb_view(P._latest_qb_state(histories, pid, gday), debut)
            identity.append(
                max(
                    abs(epa - float(row[f"{side}_qb_starter_epa_per_dropback"])),
                    abs(cpoe - float(row[f"{side}_qb_starter_cpoe"])),
                    abs(math.log1p(exp) - float(row[f"{side}_qb_starter_experience_log"])),
                )
            )
            sp = float(row[f"{side}_qb_start_probability"])
            vis = snap_inj[snap_inj["team"] == team]
            srow = vis[vis["gsis_id"] == pid]
            lrow = latest_inj[(latest_inj["team"] == team) & (latest_inj["gsis_id"] == pid)]
            latest_des = str(lrow["designation"].iloc[-1]) if len(lrow) else "none"
            if not (sp < 1.0 or latest_des in UNRESOLVED):
                continue
            listed_out = set(P._qb_ids_listed_out(vis)) | {pid}
            dvis = depth[(depth["team"] == team) & (depth["observed_at_utc"] <= decision_at)]
            backup, backup_rule = None, ""
            if len(dvis):
                dvis = dvis.sort_values(["observed_at_utc", "pos_rank"], ascending=[False, True])
                newest = dvis[dvis["observed_at_utc"].eq(dvis["observed_at_utc"].iloc[0])]
                elig = newest[~newest["gsis_id"].isin(listed_out)]
                if len(elig):
                    backup, backup_rule = str(elig["gsis_id"].iloc[0]), "depth_chart_next"
            if backup is None:
                prior = qb_games[(qb_games["team"] == team) & (qb_games["_gd"] < gday)]
                if len(prior):
                    last = prior[prior["_gd"].eq(prior["_gd"].max())]
                    order = last.sort_values("qb_dropbacks", ascending=False)["player_id"]
                    rest = [str(x) for x in order if str(x) not in listed_out]
                    if rest:
                        backup, backup_rule = rest[0], "latest_game_dropback_order"
            if backup is None:
                skipped[f"{gid}:{team}"] = "no_next_quarterback"
                continue
            bepa, bcpoe, bexp = _qb_view(P._latest_qb_state(histories, backup, gday), debut)
            brow = vis[vis["gsis_id"] == backup]
            bun = P._injury_unavailability(brow.iloc[-1]) if len(brow) else 0.0
            bsp = 1.0 - bun if np.isfinite(bun) else math.nan
            starter_epa = float(row[f"{side}_qb_starter_epa_per_dropback"])
            cands.append(
                {
                    "game_id": gid,
                    "side": side,
                    "team": team,
                    "player_id": pid,
                    "player": names.get(pid, pid),
                    "served_designation": (
                        str(srow["report_status"].iloc[-1]) if len(srow) else "none"
                    ),
                    "served_practice_status": (
                        str(srow["practice_status"].iloc[-1]) if len(srow) else "none"
                    ),
                    "served_start_probability": sp,
                    "latest_capture_designation": latest_des,
                    "latest_capture_practice_status": (
                        str(lrow["practice_status"].iloc[-1]) if len(lrow) else "none"
                    ),
                    "starter_epa_per_dropback": starter_epa,
                    "served_expected_epa_per_dropback": float(
                        row[f"{side}_qb_expected_epa_per_dropback"]
                    ),
                    "next_qb_id": backup,
                    "next_qb": names.get(backup, backup),
                    "next_qb_rule": backup_rule,
                    "next_qb_epa_per_dropback": bepa,
                    "next_qb_start_probability": bsp,
                    "in": {
                        "projected_qb_id": pid,
                        "qb_start_probability": 1.0,
                        "qb_expected_epa_per_dropback": starter_epa,
                    },
                    "out": {
                        "projected_qb_id": backup,
                        "qb_starter_epa_per_dropback": bepa,
                        "qb_starter_cpoe": bcpoe,
                        "qb_starter_experience_log": math.log1p(bexp),
                        "qb_start_probability": bsp,
                        "qb_expected_epa_per_dropback": (
                            bsp * bepa + (1.0 - bsp) * P._REPLACEMENT_QB_EPA
                        ),
                    },
                }
            )

    def run(frame, tmp, tag):
        path = Path(tmp) / f"{tag}.parquet"
        frame.to_parquet(path)
        res = plan_refresh(
            artifacts_root, data_root, season=SEASON, week=week, features_path=path, now=now
        )
        return {
            str(g.game_id): (
                float(g.new_home_cover_probability),
                g.new_pick_side,
                float(g.decision_home_spread),
            )
            for g in res.games
        }

    def scenario(c, which):
        frame = feats.copy()
        mask = frame["game_id"].eq(c["game_id"])
        for k, v in c[which].items():
            frame.loc[mask, f"{c['side']}_{k}"] = v
        for m in PLAYER_QB_STATE_METRICS:
            frame.loc[mask, f"diff_{m}"] = frame.loc[mask, f"home_{m}"].astype(float) - frame.loc[
                mask, f"away_{m}"
            ].astype(float)
        return frame

    with tempfile.TemporaryDirectory() as tmp:
        base_run = run(feats, tmp, "base")
        for i, c in enumerate(cands):
            c["_in"] = run(scenario(c, "in"), tmp, f"in{i}")
            c["_out"] = run(scenario(c, "out"), tmp, f"out{i}")

    out_games, other_moves = [], 0.0
    for gid in open_ids:
        row = wk.loc[gid]
        home, away = str(row["home_team"]), str(row["away_team"])
        b = base.loc[gid]
        p0, side0, line = base_run.get(gid, (math.nan, "", math.nan))
        clist, lines = [], []
        for c in [c for c in cands if c["game_id"] == gid]:
            pin, sin, _ = c["_in"][gid]
            pout, sout, _ = c["_out"][gid]
            for k, v in base_run.items():
                if k != gid:
                    other_moves = max(
                        other_moves, abs(c["_in"][k][0] - v[0]), abs(c["_out"][k][0] - v[0])
                    )
            side_in = _side_team(sin, home, away)
            side_out = _side_team(sout, home, away)
            flagged = side_in != side_out
            text = ""
            if flagged:
                scen, team = ("out", side_out) if side_in == b["pick_team"] else ("in", side_in)
                text = sentence(c["player"], scen, team)
                lines.append(text)
            clist.append(
                {
                    **{k: v for k, v in c.items() if not k.startswith("_") and k != "game_id"},
                    "home_probability_in": pin,
                    "home_probability_out": pout,
                    "side_if_plays": side_in,
                    "side_if_sits": side_out,
                    "flagged": bool(flagged),
                    "sentence": text,
                }
            )
        out_games.append(
            {
                "game_id": gid,
                "away_team": away,
                "home_team": home,
                "pick_team": b["pick_team"],
                "card_published_at_utc": str(b["published_at_utc"]),
                "card_forecast_artifact": b["baseline_forecast_artifact"],
                "card_from_active_forecast": bool(
                    b["baseline_forecast_artifact"] == model["active_forecast_artifact"]
                ),
                "card_home_probability": float(b["p_home"]),
                "decision_home_spread": float(line),
                "served_home_probability_now": p0,
                "served_side_now": _side_team(side0, home, away),
                "served_minus_card": p0 - float(b["p_home"]),
                "home_projected_qb": names.get(str(row["home_projected_qb_id"]), ""),
                "away_projected_qb": names.get(str(row["away_projected_qb_id"]), ""),
                "home_qb_start_probability": float(row["home_qb_start_probability"]),
                "away_qb_start_probability": float(row["away_qb_start_probability"]),
                "candidates": clist,
                "flagged_players": [c["player"] for c in clist if c["flagged"]],
                "board_line": " ".join(lines),
            }
        )
    stamp = now.strftime(STAMP)
    doc = {
        "row": "UI-33",
        "basis": "served_qb_input",
        "purpose": "informational only; never changes a served side",
        **model,
        "card_published_at_utc": max(g["card_published_at_utc"] for g in out_games),
        "season": SEASON,
        "week": week,
        "created_at_utc": stamp,
        "code_sha256": {
            c: sha(c)
            for c in (
                "scripts/ui33_scenario_producer.py",
                "src/nfl_ats/players.py",
                "src/nfl_ats/pick_refresh.py",
            )
        },
        "served_path": "nfl_ats.pick_refresh.plan_refresh (refresh-picks planning, unrecorded)",
        "feature_table": str(table.relative_to(ROOT)),
        "feature_table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "feature_table_built_at_utc": manifest["built_at_utc"],
        "served_player_snapshot": manifest["source_player_snapshot"],
        "served_depth_snapshot": manifest["source_depth_snapshot"],
        "served_pbp_snapshot": manifest["source_pbp_snapshot"],
        "latest_injury_capture": latest_cap,
        "replacement_qb_epa": P._REPLACEMENT_QB_EPA,
        "candidate_rule": (
            "projected starting QB whose served start probability is below one, or whom the "
            "latest injury capture lists Questionable or Doubtful"
        ),
        "scenario_rule": (
            "in: start probability one, expected EPA = starter EPA; out: starter treated as "
            "listed Out, next QB by the served rule (newest depth chart before the decision "
            "time, else latest game dropback order) with his own state and injury row"
        ),
        "starter_state_identity_max_abs": max(identity) if identity else None,
        "other_games_max_move": other_moves,
        "skipped": skipped,
        "candidates_total": len(cands),
        "flagged_games": [x["game_id"] for x in out_games if x["flagged_players"]],
        "games": out_games,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{SEASON}_week{week:02d}_{stamp}.json"
    path.write_text(json.dumps(doc, indent=1, default=str))
    print(path)
    print(
        f"model {model['active_model_id']} week {week} games {len(out_games)} candidates "
        f"{len(cands)} identity {doc['starter_state_identity_max_abs']} other_moves "
        f"{other_moves:.2e} flagged {doc['flagged_games']}"
    )
    for x in out_games:
        print(
            f"{x['game_id']:<16} pick {x['pick_team']:<3} card {x['card_home_probability']:.4f} "
            f"served {x['served_home_probability_now']:.4f} side {x['served_side_now']:<3} "
            f"line {x['decision_home_spread']:+5.1f} qb {x['away_projected_qb']}/"
            f"{x['home_projected_qb']} sp {x['away_qb_start_probability']:.2f}/"
            f"{x['home_qb_start_probability']:.2f}"
        )
        for c in x["candidates"]:
            print(
                f"    {c['player']:<20} {c['team']:<3} served {c['served_designation']}/"
                f"{c['served_practice_status']} sp {c['served_start_probability']:.3f} latest "
                f"{c['latest_capture_designation']} next {c['next_qb']} ({c['next_qb_rule']}, "
                f"epa {c['next_qb_epa_per_dropback']:+.3f} v {c['starter_epa_per_dropback']:+.3f}) "
                f"in {c['home_probability_in']:.4f} out {c['home_probability_out']:.4f} "
                f"{'FLAG ' + c['sentence'] if c['flagged'] else ''}"
            )


def main():
    p = argparse.ArgumentParser(
        description=(
            "UI-33 producer: for each open game on the active model's card whose projected "
            "starting quarterback carries an unresolved status, the served probability "
            "recomputed through the refresh-picks planning path with the starter in and with "
            "the next quarterback in; flags games whose served side differs. Informational only."
        )
    )
    p.add_argument("--week", default="auto", help="week number or auto (first unplayed week)")
    p.add_argument(
        "--research-mixture",
        action="store_true",
        help="run the LEAD-108 lattice mixture research path instead (never rendered)",
    )
    p.add_argument("--decision-hours", type=int, default=24, help="served build default")
    p.add_argument("--qb-span", type=int, default=12, help="served build default")
    p.add_argument("--qb-min-dropbacks", type=int, default=20, help="served build default")
    p.add_argument("--offseason-retention", type=float, default=0.75, help="served build default")
    a = p.parse_args()
    if a.research_mixture:
        cmd_mixture(a)
    else:
        cmd_served(a)


if __name__ == "__main__":
    main()
