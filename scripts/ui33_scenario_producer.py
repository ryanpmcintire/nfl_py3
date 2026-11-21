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


def cmd_run(a):
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
        "purpose": "informational only; never changes a served side",
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
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{SEASON}_week{week:02d}_{now.strftime(STAMP)}.json"
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


def main():
    p = argparse.ArgumentParser(
        description=(
            "UI-33 producer: for each open game on the active model's card, the served side's "
            "probability at the pool line with each unresolved starter in and out, others at "
            "ladder availability, on the LEAD-108 discrete lattice mixture. Informational only."
        )
    )
    p.add_argument("--week", default="auto", help="week number or auto (first unplayed week)")
    a = p.parse_args()
    cmd_run(a)


if __name__ == "__main__":
    main()
