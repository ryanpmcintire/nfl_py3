import argparse
import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sim29_lineup_sim as base  # noqa: E402

SEASON = 2026
WEEKS = range(12, 19)
FWD = base.OUT / "forward"
READS = base.LEAD / "forward"
SCHED = REPO / "data" / "raw_schedules"
SNAPDIR = REPO / "data" / "players" / "forward_snaps"
DECISIONS = REPO / "artifacts" / "clv_ledger" / "decisions.parquet"
REVISIONS = REPO / "artifacts" / "prospective" / "pick_revisions.parquet"
STAMP = re.compile(r"^\d{8}T\d{6}Z$")
FMT = "%Y%m%dT%H%M%SZ"
LANE = "docs/lanes/sim-29-simulated-lineup-from-the-wording-read.md"
PINNED = [
    "scripts/sim29_lineup_sim.py",
    "scripts/sim29_forward.py",
    "scripts/sim10_drive.py",
    "scripts/sim08_market_anchored.py",
    "scripts/mod25_generator.py",
    "scripts/sim04_loso.py",
]


def ts(s):
    return pd.to_datetime(s, format=FMT, utc=True)


def git(*args):
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    r = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, creationflags=flags
    )
    return r.stdout.strip()


def blobs():
    out = {
        p: {"work": git("hash-object", p), "head": git("rev-parse", f"HEAD:{p}")} for p in PINNED
    }
    out["head_commit"] = git("rev-parse", "HEAD")
    return out


def schedule(now):
    caps = [
        p
        for p in SCHED.iterdir()
        if STAMP.match(p.name) and (p / "schedules.parquet").exists() and ts(p.name) < now
    ]
    cap = max(caps, key=lambda p: p.name)
    g = pd.read_parquet(cap / "schedules.parquet")
    g = g[(g["season"] == SEASON) & (g["game_type"] == "REG")].copy()
    kick = pd.to_datetime(g["gameday"].astype(str) + " " + g["gametime"].astype(str))
    g["kickoff_utc"] = kick.dt.tz_localize(base.ET).dt.tz_convert("UTC")
    return cap.name, g


def week_reads(week, now):
    rows, files = [], []
    for p in sorted(READS.glob(f"{SEASON}_week{week:02d}_*.json")):
        d = json.loads(p.read_text())
        if ts(d["recorded_utc"]) >= now:
            continue
        files.append(
            {
                "file": p.name,
                "capture_utc": d["capture_utc"],
                "recorded_utc": d["recorded_utc"],
                "prompt_sha256": d.get("prompt_sha256"),
                "rows": len(d["rows"]),
            }
        )
        for r in d["rows"]:
            if r["capture_utc"] <= r["kickoff_utc"]:
                rows.append(
                    (
                        r["team"],
                        r["gsis_id"],
                        r["player"],
                        float(r["p_plays"]),
                        r["capture_utc"],
                        r["kickoff_utc"],
                    )
                )
    df = pd.DataFrame(rows, columns=["team", "gsis_id", "player", "rp", "cap", "kick"])
    df["nm"] = df["player"].map(base.norm)
    return df, files


def snaps(week, now):
    fs = []
    for p in SNAPDIR.glob(f"snap_counts_{SEASON}_before_week_*.parquet"):
        k = int(p.stem.rsplit("_", 1)[1])
        if k <= week and p.stat().st_mtime < now.timestamp():
            fs.append((k, p))
    _, f = max(fs)
    sn = pd.read_parquet(f)
    sn = sn[(sn["season"] == SEASON) & (sn["game_type"] == "REG") & (sn["week"] < week)].copy()
    sn["grp"] = np.where(
        sn["position"] == "QB", "QB", np.where(sn["position"].isin(base.OL_POS), "OL", "")
    )
    sn = sn[sn["grp"] != ""]
    return f.name, int(sn["week"].max()), sn


def card(week, now):
    p = pd.read_parquet(DECISIONS)
    p = p[(p["season"] == SEASON) & (p["week"] == week)].copy()
    p["at"] = pd.to_datetime(p["recorded_at_utc"], utc=True)
    p = p[p["at"] < now].sort_values("at")
    first = p.drop_duplicates("game_id", keep="first").set_index("game_id")
    last = p.drop_duplicates("game_id", keep="last").set_index("game_id")
    cache = {}
    keys = last[["forecast_artifact", "method"]].dropna().drop_duplicates()
    for fa, meth in keys.itertuples(index=False):
        f = REPO / "artifacts" / fa / "predictions.csv"
        if f.exists():
            c = pd.read_csv(f)
            if "method" in c.columns:
                c = c[c["method"] == meth]
            c = c.dropna(subset=["home_cover_probability"]).drop_duplicates("game_id", keep="last")
            cache[(fa, meth)] = dict(zip(c["game_id"], c["home_cover_probability"], strict=True))
    rv = pd.read_parquet(REVISIONS)
    rv = rv[(rv["season"] == SEASON) & (rv["week"] == week)].copy()
    rv["at"] = pd.to_datetime(rv["revision_recorded_at_utc"], utc=True)
    rv = rv[rv["at"] < now].sort_values("at").drop_duplicates("game_id", keep="last")
    return first, last, cache, rv.set_index("game_id")


def served(gid, last, cache, rv):
    if gid in rv.index:
        x = rv.loc[gid]
        side = x["served_pick_side"] if pd.notna(x["served_pick_side"]) else x["new_pick_side"]
        pr = x["served_home_cover_probability"]
        if pd.isna(pr):
            pr = x["new_home_cover_probability"]
        return str(side), float(pr), "pick_revision", x["at"].strftime(FMT)
    if gid in last.index:
        x = last.loc[gid]
        pr = cache.get((x["forecast_artifact"], x["method"]), {}).get(gid)
        pr = None if pr is None or pd.isna(pr) else float(pr)
        return str(x["pick_side"]), pr, "card_forecast", x["at"].strftime(FMT)
    return None, None, None, None


def draw(r, sn, pre, ros, rev, qb_cost, ol_cost, n_draws):
    rng = np.random.default_rng(zlib.crc32(f"draw:{r.game_id}".encode()))
    out, cfg = [], {}
    for t in (r.home_team, r.away_team):
        prior = sn[sn["team"] == t]
        if prior.empty:
            return None, None
        tot = prior.groupby(["grp", "pfr_player_id", "player"])["offense_snaps"].sum().reset_index()
        st = pd.concat(
            [
                tot[tot["grp"] == "QB"].nlargest(1, "offense_snaps"),
                tot[tot["grp"] == "OL"].nlargest(base.OL_SLOTS, "offense_snaps"),
            ]
        )
        rw = pre[pre["team"] == t].sort_values("cap")
        recs = []
        for p in st.itertuples():
            hit = rw[rw["pfr_id"] == p.pfr_player_id]
            if hit.empty:
                hit = rw[rw["nm"] == base.norm(p.player)]
            state = base.roster_state(
                ros, t, p.pfr_player_id, rev.get(p.pfr_player_id, []), p.player
            )
            ka = state in base.ABSENT_STATUS or state == "other_team"
            pr = float(hit["rp"].iloc[-1]) if not hit.empty else float(not ka)
            recs.append(
                {
                    "team": t,
                    "grp": p.grp,
                    "player": p.player,
                    "pfr_id": p.pfr_player_id,
                    "read": pr,
                    "has_read": bool(not hit.empty),
                    "roster_state": state,
                    "known_absent": bool(ka),
                    "prior_offense_snaps": float(p.offense_snaps),
                }
            )
        pr = np.array([x["read"] for x in recs])
        isqb = np.array([x["grp"] == "QB" for x in recs])
        dr = rng.random((n_draws, len(recs))) >= pr
        cfg[t] = np.round(qb_cost * (dr & isqb).sum(1) + ol_cost * (dr & ~isqb).sum(1), 10)
        out += recs
    return out, cfg


def sim_forward(task):
    import sim08_market_anchored as ma

    row, m = task
    eng, fit = base._E["eng"], base._E["fit"]
    xd = ma.solve_diff_only(fit, row["line"])
    res = {}
    for c in [(0.0, 0.0)] + [tuple(c) for c in row["configs"]]:
        if c not in res:
            res[c] = base.sim_mean(eng, xd, c, row["seed"], m, row["week"])
    return {
        "game_id": row["game_id"],
        "sim_base": res[(0.0, 0.0)],
        "config_sim_means": [res[tuple(c)] for c in row["configs"]],
        "x_diff": float(xd),
        "s": float(eng.s),
    }


def log(a):
    t0 = time.time()
    now = pd.Timestamp.now(tz="UTC")
    run = now.strftime(FMT)
    cap, g = schedule(now)
    if a.week == "auto":
        up = g[g["kickoff_utc"] > now]
        week = int(up["week"].min()) if len(up) else None
    else:
        week = int(a.week)
    if week is None or (week not in WEEKS and not a.smoke):
        print(json.dumps({"skip": True, "week": week, "run_utc": run, "window": "weeks 12-18"}))
        return 0
    gw = g[g["week"] == week].sort_values(["kickoff_utc", "game_id"])
    t_open = base.opener_utc(gw["gameday"].min())
    reads, files = week_reads(week, now)
    snap_file, snap_max, sn = snaps(week, now)
    ros_name, ros = base.roster_before(t_open)
    rx = ros[["gsis_id", "pfr_id"]].dropna()
    xw = {**dict(zip(rx["gsis_id"], rx["pfr_id"], strict=True)), **base.crosswalk()}
    rev = {}
    for k, v in xw.items():
        rev.setdefault(v, []).append(k)
    reads["pfr_id"] = reads["gsis_id"].map(xw)
    qb_cost, ol_cost = base.costs()
    first, last, cache, rv = card(week, now)
    tue = t_open.tz_convert(base.ET).date()
    games, tasks = [], []
    for r in gw.itertuples():
        gid = r.game_id
        kick = r.kickoff_utc.strftime(FMT)
        side, sp, ssrc, sat = served(gid, last, cache, rv)
        rec = {
            "game_id": gid,
            "kickoff_utc": kick,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "anchor_spread_line": float(r.spread_line) if pd.notna(r.spread_line) else None,
            "tuesday_line": None,
            "tuesday_line_recorded_utc": None,
            "tuesday_line_is_tuesday": False,
            "served_side": side,
            "served_home_probability": sp,
            "served_source": ssrc,
            "served_recorded_utc": sat,
        }
        if gid in first.index:
            at = first.loc[gid, "at"]
            rec["tuesday_line"] = float(first.loc[gid, "decision_home_spread"])
            rec["tuesday_line_recorded_utc"] = at.strftime(FMT)
            rec["tuesday_line_is_tuesday"] = bool(at.tz_convert(base.ET).date() == tue)
        pre = reads[reads["cap"] < kick]
        covered = bool(
            pre["team"].isin([r.home_team, r.away_team]).any() or (pre["kick"] == kick).any()
        )
        rec["read_covered"] = covered
        if r.kickoff_utc <= now:
            status = "already_kicked_off"
        elif rec["anchor_spread_line"] is None:
            status = "no_anchor_line"
        elif not rec["tuesday_line_is_tuesday"]:
            status = "no_tuesday_line"
        elif not covered:
            status = "no_read_before_kickoff"
        else:
            status = "logged"
        rec["status"] = status
        go = status == "logged" or (
            a.smoke and status in ("no_read_before_kickoff", "no_tuesday_line")
        )
        if go:
            starters, cfg = draw(r, sn, pre, ros, rev, qb_cost, ol_cost, a.draws)
            if starters is None:
                rec["status"] = "incomplete_snaps"
            else:
                pairs = pd.Series(
                    list(zip(cfg[r.home_team], cfg[r.away_team], strict=True))
                ).value_counts()
                rec["starters"] = starters
                rec["configs"] = [[float(x) for x in k] for k in pairs.index]
                rec["freq"] = (pairs.to_numpy() / a.draws).tolist()
                rec["draw_seed"] = zlib.crc32(f"draw:{gid}".encode())
                rec["sim_seed"] = zlib.crc32(f"sim:{gid}".encode())
                tasks.append(
                    {
                        "game_id": gid,
                        "line": rec["anchor_spread_line"],
                        "seed": rec["sim_seed"],
                        "configs": rec["configs"],
                        "week": week,
                    }
                )
        games.append(rec)
    res = {}
    if tasks:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(a.workers, len(tasks)), initializer=base.init_worker) as pool:
            for x in pool.map(sim_forward, [(t, a.sims) for t in tasks], chunksize=1):
                res[x["game_id"]] = x
    for rec in games:
        x = res.get(rec["game_id"])
        if x is None:
            continue
        sim_read = float(
            sum(f * v for f, v in zip(rec["freq"], x["config_sim_means"], strict=True))
        )
        rec.update(
            {
                "config_sim_means": x["config_sim_means"],
                "sim_base": x["sim_base"],
                "sim_read": sim_read,
                "sim_shift": sim_read - x["sim_base"],
                "x_diff": x["x_diff"],
                "engine_s": x["s"],
            }
        )
    out = {
        "season": SEASON,
        "week": week,
        "run_utc": run,
        "smoke": bool(a.smoke),
        "scoring_eligible": not a.smoke,
        "declaration": f"{LANE} forward test 2026-11-21",
        "frozen": {
            "blobs": blobs(),
            "qb_cost": qb_cost,
            "ol_cost": ol_cost,
            "ol_slots": base.OL_SLOTS,
            "draws": a.draws,
            "sims": a.sims,
            "context_season": base.CONTEXT_SEASON,
            "fit": str(base.FIT.relative_to(REPO)),
        },
        "inputs": {
            "schedule_capture": cap,
            "opener_utc": t_open.strftime(FMT),
            "roster_snapshot": ros_name,
            "snap_file": snap_file,
            "snap_max_week": snap_max,
            "read_files": files,
            "decisions": str(DECISIONS.relative_to(REPO)),
            "revisions": str(REVISIONS.relative_to(REPO)),
        },
        "status_counts": pd.Series([x["status"] for x in games]).value_counts().to_dict(),
        "games": games,
        "seconds": time.time() - t0,
    }
    d = FWD / "smoke" if a.smoke else FWD
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{SEASON}_week{week:02d}_{run}{'_smoke' if a.smoke else ''}.json"
    f.write_text(json.dumps(out, indent=1, default=str))
    print(
        json.dumps(
            {
                "file": str(f.relative_to(REPO)),
                "week": week,
                "status_counts": out["status_counts"],
                "simulated": len(res),
                "seconds": out["seconds"],
                "inputs": out["inputs"],
            },
            indent=1,
            default=str,
        )
    )
    return 0


def main():
    p = argparse.ArgumentParser(description="SIM-29 forward log of simulated lineup draws")
    sub = p.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log", help="log the week's pre-kickoff simulated lineup draws")
    lg.add_argument("--week", default="auto")
    lg.add_argument("--smoke", action="store_true")
    lg.add_argument("--draws", type=int, default=200)
    lg.add_argument("--sims", type=int, default=200)
    lg.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    return log(a)


if __name__ == "__main__":
    sys.exit(main())
