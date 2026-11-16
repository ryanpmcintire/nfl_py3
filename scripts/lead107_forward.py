import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forward_ledger as FL
import lead106_injury_language as L6
import lead108_forward as F8
import lead108_wording_mixture as W
import mod24_ftn_turnover_luck as m
import mod25_wf_forward as mwf

ROOT = L6.ROOT
SEASON = F8.SEASON
FIRST_WEEK = F8.FIRST_WEEK
LAST_WEEK = F8.LAST_WEEK
OUT = ROOT / "artifacts" / "lead107_forward"
FROZEN = OUT / "frozen.json"
LANE = "docs/lanes/lead107-forward-2026-11-16.md"
STAMP = F8.STAMP
ts = F8.ts
sha_file = F8.sha_file


def cmd_freeze(a):
    if FROZEN.exists():
        raise SystemExit(f"already frozen: {FROZEN}")
    f8 = json.loads(F8.FROZEN.read_text())
    g = pd.read_parquet(F8.STUDY / "games.parquet")
    if sha_file(F8.STUDY / "games.parquet") != f8["games_sha256"]:
        raise SystemExit("study games.parquet changed since the LEAD-108 freeze")
    g = g[g["week"].between(W.FIRST_WEEK, W.LAST_WEEK)]
    tau2 = float(g["x_read"].var(ddof=1))
    body = {
        "family": "lead108_wording_mixture",
        "term": "lead107_dispersion_weight",
        "coefficient_logit_per_point": f8["coefficient_logit_per_point"],
        "coefficient_source": "artifacts/lead108_forward/frozen.json",
        "lead108_frozen_sha256": sha_file(F8.FROZEN),
        "tau2_between_game_var_pts2": tau2,
        "tau2_source": "var(x_read, ddof 1) over lead108_wording_mixture games.parquet weeks 2-9",
        "tau2_games": len(g),
        "games_sha256": f8["games_sha256"],
        "rule": (
            "term = w * x108, w = tau2 / (tau2 + s2), s2 = ddof-1 variance over admissible read "
            "files of the per-file LEAD-108 game term (0 with one file); delta logit = b * term"
        ),
        "values_spread_pts_per_out": f8["values_spread_pts_per_out"],
        "first_week": FIRST_WEEK,
        "last_week": LAST_WEEK,
        "lane": LANE,
        "no_week10_or_later_outcomes": True,
        "created_at_utc": datetime.now(UTC).strftime(STAMP),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(body, indent=1))
    print(json.dumps(body, indent=1))


def reads_by_file(week, sched, now):
    fz = json.loads((L6.ART / "frozen.json").read_text())
    cut = {}
    for r in sched.itertuples():
        cut[r.home_team] = r.cutoff_utc
        cut[r.away_team] = r.cutoff_utc
    rows, files = [], []
    for f in sorted(L6.FORWARD.glob(f"{SEASON}_week{week:02d}_*.json")):
        j = json.loads(f.read_text())
        rec = ts(j["recorded_utc"])
        mtime = pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC")
        ok = j["prompt_sha256"] == fz["prompt_sha256"] and j["model"] == fz["model"]
        inj = pd.read_parquet(F8.INJ / j["capture_utc"] / "injuries.parquet")
        inj = inj[(inj["season"] == SEASON) & (inj["week"] == week)]
        pos = inj.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")["position"]
        kept = 0
        for o in j["rows"] if ok and rec <= now else []:
            c = cut.get(o["team"])
            k = min(ts(o["kickoff_utc"]), c) if c is not None else None
            if k is None or not (rec < k and mtime < k) or o["gsis_id"] not in pos.index:
                continue
            rows.append(
                {
                    "file": f.name,
                    "gsis_id": o["gsis_id"],
                    "team": o["team"],
                    "position": pos[o["gsis_id"]],
                    "p_plays": float(o["p_plays"]),
                    "recorded_utc": rec,
                }
            )
            kept += 1
        files.append({"file": f.name, "enrolled": ok, "rows": len(j["rows"]), "kept": kept})
    cols = ["file", "gsis_id", "team", "position", "p_plays", "recorded_utc"]
    return pd.DataFrame(rows, columns=cols), files


def cmd_pregame(a):
    now = datetime.now(UTC)
    fz = json.loads(FROZEN.read_text())
    if fz["lead108_frozen_sha256"] != sha_file(F8.FROZEN):
        raise SystemExit("LEAD-108 frozen.json changed since the LEAD-107 freeze")
    week = F8.resolve(a.week, now)
    if not a.dry and not FIRST_WEEK <= week <= LAST_WEEK:
        raise SystemExit(
            f"week {week} outside the declared forward window {FIRST_WEEK}-{LAST_WEEK}"
        )
    sched = F8.schedule(week)
    xf, files = reads_by_file(week, sched, now)
    vals = fz["values_spread_pts_per_out"]
    latest = xf.sort_values("recorded_utc").drop_duplicates("gsis_id", keep="last")
    tx, tn = F8.team_x(latest.reset_index(drop=True), vals)
    per_file = {
        name: F8.team_x(grp.reset_index(drop=True), vals) for name, grp in xf.groupby("file")
    }
    base, base_at = F8.served_now(week, now)
    b = fz["coefficient_logit_per_point"]
    tau2 = fz["tau2_between_game_var_pts2"]
    games, skipped = [], {}
    for r in sched.itertuples():
        if not a.dry and r.cutoff_utc <= pd.Timestamp(now):
            skipped[r.game_id] = "past_cutoff"
            continue
        n = int(tn.get(r.home_team, 0) + tn.get(r.away_team, 0))
        if n == 0:
            skipped[r.game_id] = "no_admissible_read"
            continue
        if r.game_id not in base:
            skipped[r.game_id] = "no_published_card"
            continue
        x108 = float(tx.get(r.home_team, 0.0)) - float(tx.get(r.away_team, 0.0))
        xs = [
            float(t.get(r.home_team, 0.0)) - float(t.get(r.away_team, 0.0))
            for t, c in per_file.values()
            if c.get(r.home_team, 0) + c.get(r.away_team, 0) > 0
        ]
        s2 = float(np.var(xs, ddof=1)) if len(xs) > 1 else 0.0
        w = tau2 / (tau2 + s2)
        p0 = float(base[r.game_id])
        dl = b * w * x108
        pt = float(W.sig(W.logit(p0) + dl))
        games.append(
            {
                "game_id": r.game_id,
                "away_team": r.away_team,
                "home_team": r.home_team,
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "pick_deadline_utc": r.pick_deadline_utc.isoformat(),
                "players_read": n,
                "files_read": len(xs),
                "per_file_x": xs,
                "dispersion_var": s2,
                "weight": w,
                "x108": x108,
                "term_value": w * x108,
                "base_home_probability": p0,
                "base_published_at_utc": base_at[r.game_id],
                "delta_logit": dl,
                "combined_home_probability": pt,
                "side_changed": bool((p0 >= 0.5) != (pt >= 0.5)),
            }
        )
    doc = {
        "season": SEASON,
        "week": week,
        "created_at_utc": now.strftime(STAMP),
        "frozen_created_at_utc": fz["created_at_utc"],
        "frozen_sha256": sha_file(FROZEN),
        "coefficient_logit_per_point": b,
        "tau2_between_game_var_pts2": tau2,
        "read_files": files,
        "players_read": len(latest),
        "skipped": skipped,
        "games": games,
    }
    head = json.dumps({k: v for k, v in doc.items() if k != "games"}, indent=1, default=str)
    if a.dry:
        print(head)
        for gm in games:
            print(
                f"{gm['game_id']:<18} reads {gm['players_read']:>3} files {gm['files_read']} "
                f"x108 {gm['x108']:+.3f} s2 {gm['dispersion_var']:.3f} w {gm['weight']:.3f} "
                f"base {gm['base_home_probability']:.4f} term "
                f"{gm['combined_home_probability']:.4f} changed {gm['side_changed']}"
            )
        print("DRY: nothing written")
        return
    if not games:
        print(head)
        print("no game with an admissible read before its cutoff; nothing written")
        return
    out = OUT / f"{SEASON}_week{week:02d}_{now.strftime(STAMP)}.json"
    out.write_text(json.dumps(doc, indent=1, default=str))
    print(out, len(games), "games")


def cmd_score(a):
    fz = json.loads(FROZEN.read_text())
    b = fz["coefficient_logit_per_point"]
    rows = []
    for f in sorted(OUT.glob(f"{SEASON}_week*_*.json")):
        d = json.loads(f.read_text())
        if d["frozen_sha256"] != sha_file(FROZEN) or not FIRST_WEEK <= d["week"] <= LAST_WEEK:
            continue
        mtime = pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC")
        for gm in d["games"]:
            cut = min(pd.Timestamp(gm["kickoff_utc"]), pd.Timestamp(gm["pick_deadline_utc"]))
            if ts(d["created_at_utc"]) < cut and mtime < cut:
                rows.append({**gm, "week": d["week"], "created": d["created_at_utc"]})
    if not rows:
        print("no admissible forward rows yet; nothing graded")
        return
    t = pd.DataFrame(rows).sort_values("created").drop_duplicates("game_id", keep="last")
    served = pd.read_parquet(m.SERVED_FEATURES)
    g = served[served["game_id"].isin(t["game_id"]) & served["result"].notna()].copy()
    op = mwf.tuesday_open(g)
    g["open_line"] = g["game_id"].map(lambda i: op.get(i, (np.nan,))[0])
    g = g.merge(FL._played_card(), on="game_id", how="inner")
    g = g.merge(t[["game_id", "term_value", "x108"]], on="game_id")
    g = g.dropna(subset=["open_line", "served_p"])
    g = g[g["result"] != g["open_line"]].copy()
    if g.empty:
        print("no settled forward game graded at the opener yet; nothing graded")
        return
    y = (g["result"] > g["open_line"]).astype(float).to_numpy()
    wk = g["week"].to_numpy()
    ps = g["served_p"].to_numpy()
    pt = W.sig(W.logit(ps) + b * g["term_value"].to_numpy())
    p8 = W.sig(W.logit(ps) + b * g["x108"].to_numpy())
    rng = np.random.default_rng(W.SEED)
    rep = {
        "games": len(g),
        "weeks": sorted(int(w) for w in np.unique(wk)),
        "changed_pick_record_term_vs_served": W.record(pt, ps, y),
        "mean_log_loss": {
            "served": float(W.ll(ps, y).mean()),
            "term": float(W.ll(pt, y).mean()),
            "lead108_unweighted": float(W.ll(p8, y).mean()),
        },
    }
    for name, f, ref in (
        ("log_loss_vs_served", W.ll, ps),
        ("brier_vs_served", W.br, ps),
        ("log_loss_vs_lead108", W.ll, p8),
    ):
        d = f(ref, y) - f(pt, y)
        bs = W.boot(d, wk, rng)
        rep[name] = {
            "gain": float(d.mean()),
            "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "p_positive": float((bs > 0).mean()),
        }
    if len(np.unique(wk)) > 1:
        pi, _ = W.lowo(np.ones((len(g), 1)), y, np.zeros(len(g)), wk)
        rep["intercept_only_log_loss"] = float(W.ll(pi, y).mean())
    rep["reliability_term"] = W.reliability(pt, y, 5) if len(g) >= 5 else []
    (OUT / "score.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    p = argparse.ArgumentParser(
        description=(
            "LEAD-107 declared forward test: the frozen LEAD-108 read-lineup term weighted by "
            "the reliability of repeated LEAD-106 reads, added to the served card probability, "
            "weeks 11-18 of 2026. Research only; serves nothing."
        )
    )
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("freeze").set_defaults(fn=cmd_freeze)
    pg = sp.add_parser("pregame")
    pg.add_argument("--week", default="auto")
    pg.add_argument("--dry", action="store_true")
    pg.set_defaults(fn=cmd_pregame)
    sp.add_parser("score").set_defaults(fn=cmd_score)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
