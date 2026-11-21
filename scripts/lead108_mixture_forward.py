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
import lead108_wording_mixture as U1
import lead110_absorption as A
import mod25_wf_forward as mwf

ROOT = L6.ROOT
SEASON = 2026
FIRST_WEEK = 12
LAST_WEEK = 18
FIRST_FORWARD_LADDER_WEEK = 10
OUT = ROOT / "artifacts" / "lead108_mixture_forward"
SPLASH = ROOT / "data" / "splash"
STAMP = LF.STAMP
CODE = ("scripts/lead108_full_mixture.py", "scripts/lead108_mixture_forward.py")
COLS = ["week", "designation", "practice_status", "played", "snap_share"]


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def mtime(path):
    return pd.Timestamp(Path(path).stat().st_mtime, unit="s", tz="UTC")


def resolve(arg, now):
    if arg != "auto":
        return int(arg)
    g = pd.read_parquet(
        ROOT / "data" / "processed" / "game_features.parquet",
        columns=["season", "week", "kickoff", "game_type"],
    )
    g = g[(g["season"] == SEASON) & (g["game_type"] == "REG")]
    k = pd.to_datetime(g["kickoff"], utc=True, format="ISO8601")
    ahead = g.loc[k > pd.Timestamp(now), "week"]
    return int(ahead.min()) if len(ahead) else None


def admissible_reads(week, sched, now):
    fz = json.loads((L6.ART / "frozen.json").read_text())
    cut = {}
    for r in sched.itertuples():
        cut[r.home_team] = r.cutoff_utc
        cut[r.away_team] = r.cutoff_utc
    rows, files = [], []
    for f in sorted(L6.FORWARD.glob(f"{SEASON}_week{week:02d}_*.json")):
        j = json.loads(f.read_text())
        rec = LF.ts(j["recorded_utc"])
        mt = mtime(f)
        enrolled = j["prompt_sha256"] == fz["prompt_sha256"] and j["model"] == fz["model"]
        before_run = rec <= pd.Timestamp(now) and mt <= pd.Timestamp(now)
        inj = pd.read_parquet(LF.INJ / j["capture_utc"] / "injuries.parquet")
        inj = inj[(inj["season"] == SEASON) & (inj["week"] == week)]
        inj = inj.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")
        kept = 0
        for o in j["rows"] if enrolled and before_run else []:
            c = cut.get(o["team"])
            if c is None or o["gsis_id"] not in inj.index:
                continue
            if not (rec < c and rec < LF.ts(o["kickoff_utc"]) and mt < c):
                continue
            ps = inj.at[o["gsis_id"], "practice_status"]
            rows.append(
                {
                    "gsis_id": o["gsis_id"],
                    "team": o["team"],
                    "week": week,
                    "position": inj.at[o["gsis_id"], "position"],
                    "designation": o["designation"],
                    "practice_status": ps if pd.notna(ps) else "none",
                    "rp_raw": float(o["p_plays"]),
                    "recorded_utc": rec,
                }
            )
            kept += 1
        files.append(
            {
                "file": f.name,
                "recorded_utc": j["recorded_utc"],
                "modified_utc": mt.strftime(STAMP),
                "enrolled": enrolled,
                "before_run": bool(before_run),
                "rows": len(j["rows"]),
                "kept": kept,
            }
        )
    x = pd.DataFrame(rows)
    if x.empty:
        return x, files
    x = x.sort_values("recorded_utc").drop_duplicates("gsis_id", keep="last")
    return x.reset_index(drop=True), files


def ladder_table(week, now):
    ver = M.verified_population()
    w9 = pd.read_parquet(L6.ART / "forward_score" / f"{SEASON}_week09_scored.parquet")
    w9 = w9[w9["verified"].astype(bool)].copy()
    w9["practice_status"] = w9["practice_status"].fillna("none")
    parts, used, missing = [ver[COLS], w9[COLS]], [], []
    for w in range(FIRST_FORWARD_LADDER_WEEK, week):
        f = F.SNAPS / f"snap_counts_{SEASON}_before_week_{w + 1:02d}.parquet"
        if not f.exists() or mtime(f) > pd.Timestamp(now):
            missing.append(w)
            continue
        fw, _, _ = F.load_forward(w)
        if fw.empty:
            missing.append(w)
            continue
        snaps, _ = F.snap_frame(w)
        fo = F.outcomes(fw, snaps, w, w)
        fv = fo[fo["verified"].astype(bool)].copy()
        fv["practice_status"] = fv["practice_status"].fillna("none")
        parts.append(fv[COLS])
        used.append(w)
    prior = pd.concat(parts, ignore_index=True)
    prior = prior[prior["week"] < week]
    tab = {
        week: (
            prior.groupby(U1.KEYS).agg(**M.AGG),
            prior.groupby("designation").agg(**M.AGG),
        )
    }
    meta = {
        "prior_rows": len(prior),
        "prior_weeks": sorted(int(w) for w in prior["week"].unique()),
        "forward_weeks_used": used,
        "forward_weeks_missing_snaps": missing,
    }
    return tab, meta


def card(week, now):
    pub = pd.read_parquet(ROOT / "artifacts" / "clv_ledger" / "published_picks.parquet")
    pub = pub[(pub["season"] == SEASON) & (pub["week"] == week)]
    pub = pub[pd.to_datetime(pub["published_at_utc"], utc=True) <= pd.Timestamp(now)]
    last = pub.sort_values("published_at_utc").groupby("game_id").tail(1).set_index("game_id")
    last["p_home"] = np.where(
        last["pick_team"] == last["home_team"], last["displayed_score"], 1 - last["displayed_score"]
    )
    return last


def splash(week, now):
    out = {}
    for f in sorted(SPLASH.glob(f"{SEASON}_week{week:02d}_*.json")):
        if mtime(f) > pd.Timestamp(now):
            continue
        for g in json.loads(f.read_text()).get("games", []):
            if g.get("home_spread") is not None:
                out[str(g["game_id"])] = (float(g["home_spread"]), f.name)
    return out


def history_pool():
    gf = pd.read_parquet(
        ROOT / "data" / "processed" / "game_features.parquet",
        columns=["season", "spread_line", "result"],
    )
    gf = gf[(gf["season"] < SEASON) & gf["spread_line"].notna() & gf["result"].notna()]
    return gf["spread_line"].to_numpy(float), gf["result"].to_numpy(float)


def cmd_log(a):
    real_now = datetime.now(UTC)
    if a.as_of and not a.smoke:
        raise SystemExit("--as-of is allowed only with --smoke")
    now = pd.Timestamp(a.as_of).tz_convert("UTC").to_pydatetime() if a.as_of else real_now
    week = resolve(a.week, now)
    if week is None or (not a.smoke and not FIRST_WEEK <= week <= LAST_WEEK):
        print(f"SKIP: week {week} outside the declared forward window {FIRST_WEEK}-{LAST_WEEK}")
        return
    sched = LF.schedule(week)
    x, files = admissible_reads(week, sched, now)
    tab, lmeta = ladder_table(week, now)
    clip = 1.0 / (len(x) + 2)
    if not x.empty:
        x["lp"] = [L6.ladder_pred(r, tab, U1.KEYS)[0] for _, r in x.iterrows()]
        x["rp"] = x["rp_raw"].clip(clip, 1 - clip)
        x["grp"] = x["position"].map(A.POS_TO_GROUP)
    else:
        x = pd.DataFrame(columns=["gsis_id", "team", "week", "grp", "lp", "rp"])
    base = card(week, now)
    pool = splash(week, now)
    opens = mwf.tuesday_open(sched)
    rows, skipped = [], {}
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
        rows.append(
            {
                "game_id": r.game_id,
                "away_team": r.away_team,
                "home_team": r.home_team,
                "week": week,
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "pick_deadline_utc": r.pick_deadline_utc.isoformat(),
                "cutoff_utc": r.cutoff_utc.isoformat(),
                "line_source": src,
                "open_line": line,
                "pool_line": pool.get(r.game_id, (None, None))[0],
                "pool_capture": pool.get(r.game_id, (None, None))[1],
                "card_line": float(b["market_spread"]),
                "tuesday_opener": op[0],
                "tuesday_opener_observed_utc": op[1],
                "p_served": float(b["p_home"]),
                "base_published_at_utc": str(b["published_at_utc"]),
                "served_pick_team": b["pick_team"],
                "players_home": int((x["team"] == r.home_team).sum()),
                "players_away": int((x["team"] == r.away_team).sum()),
            }
        )
    g = pd.DataFrame(rows)
    games = []
    if not g.empty:
        hl, hm = history_pool()
        val = json.loads((A.OUT / "report.json").read_text())["weights_spread_pts_per_out"]
        mx = M.mixtures(g, x, val, hl, hm)
        g = pd.concat([g, mx], axis=1)
        off = U1.logit(g["p_served"].to_numpy())
        g["delta_logit_ladder"] = U1.logit(g["p_mix_lad"].to_numpy()) - off
        g["delta_logit_read"] = U1.logit(g["p_mix_read"].to_numpy()) - off
        served_home = g["p_served"] >= 0.5
        g["ladder_side_changed_vs_served"] = (g["p_mix_lad"] >= 0.5) != served_home
        g["read_side_changed_vs_served"] = (g["p_mix_read"] >= 0.5) != served_home
        g["read_vs_ladder_side_differs"] = (g["p_mix_read"] >= 0.5) != (g["p_mix_lad"] >= 0.5)
        g = g.rename(
            columns={
                "open_line": "lattice_line",
                "p_served": "base_home_probability",
                "p_mix_lad": "ladder_arm_home_probability",
                "p_mix_read": "read_arm_home_probability",
                "e_lad": "ladder_expected_shift_pts",
                "e_read": "read_expected_shift_pts",
                "n_players": "valued_players",
                "zero_err": "zero_shift_error",
            }
        )
        games = json.loads(g.to_json(orient="records"))
    doc = {
        "lead": "LEAD-108",
        "arm": "unfitted read-weighted PER-10 lattice scenario mixture vs ladder-weighted",
        "season": SEASON,
        "week": week,
        "smoke": bool(a.smoke),
        "scoreable": bool(not a.smoke and FIRST_WEEK <= week <= LAST_WEEK),
        "created_at_utc": real_now.strftime(STAMP),
        "as_of_utc": pd.Timestamp(now).strftime(STAMP),
        "code_sha256": {c: sha(c) for c in CODE},
        "value_source": str(A.OUT / "report.json"),
        "read_files": files,
        "rows_read": len(x),
        "read_clip": clip,
        "ladder": lmeta,
        "skipped": skipped,
        "games": games,
    }
    summary = {k: v for k, v in doc.items() if k != "games"}
    print(json.dumps(summary, indent=1, default=str))
    for gm in games:
        print(
            f"{gm['game_id']:<16} {gm['line_source']:<18} line {gm['lattice_line']:+5.1f} "
            f"open {gm['tuesday_opener']} card {gm['card_line']} "
            f"n {gm['valued_players']:>2} base {gm['base_home_probability']:.4f} "
            f"lad {gm['ladder_arm_home_probability']:.4f} "
            f"read {gm['read_arm_home_probability']:.4f} "
            f"err {gm['zero_shift_error']:.1e} pick {gm['served_pick_team']} "
            f"chg {gm['read_side_changed_vs_served']}"
        )
    if not games:
        print("no game with a published card before its cutoff; nothing written")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{SEASON}_week{week:02d}_{real_now.strftime(STAMP)}.json"
    out = OUT / (f"smoke_{name}" if a.smoke else name)
    out.write_text(json.dumps(doc, indent=1, default=str))
    print(out, len(games), "games")


def main():
    p = argparse.ArgumentParser(
        description=(
            "LEAD-108 declared forward log: unfitted read-weighted and ladder-weighted injury "
            "scenario mixtures on the discrete margin lattice, per game, before kickoff, weeks "
            "12-18 of 2026. Research only; never changes a served side."
        )
    )
    sp = p.add_subparsers(dest="cmd", required=True)
    lg = sp.add_parser("log", help="write per-game arm probabilities for one week")
    lg.add_argument("--week", default="auto", help="week number or auto (first unplayed week)")
    lg.add_argument(
        "--smoke", action="store_true", help="any week; file marked smoke and never scored"
    )
    lg.add_argument("--as-of", default=None, help="smoke only: UTC time standing in for now")
    lg.set_defaults(fn=cmd_log)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
