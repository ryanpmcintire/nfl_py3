import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forward_ledger as FL
import lead106_injury_language as L6
import lead108_wording_mixture as W
import lead110_absorption as A
import mod24_ftn_turnover_luck as m
import mod25_wf_forward as mwf

from nfl_ats.published_picks import game_deadlines

ROOT = L6.ROOT
SEASON = 2026
FIRST_WEEK = 11
LAST_WEEK = 18
OUT = ROOT / "artifacts" / "lead108_forward"
FROZEN = OUT / "frozen.json"
STUDY = ROOT / "artifacts" / "lead108_wording_mixture"
INJ = ROOT / "data" / "raw" / "nflverse_injuries"
LANE = "docs/lanes/lead108-forward-2026-11-16.md"
STAMP = "%Y%m%dT%H%M%SZ"


def ts(s):
    return pd.Timestamp(datetime.strptime(s, STAMP).replace(tzinfo=UTC))


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmd_freeze(a):
    if FROZEN.exists():
        raise SystemExit(f"already frozen: {FROZEN}")
    g = pd.read_parquet(STUDY / "games.parquet")
    g = g[g["week"].between(W.FIRST_WEEK, W.LAST_WEEK)]
    b = W.fit(g[["x_read"]].to_numpy(), g["y"].to_numpy(), W.logit(g["p_served"].to_numpy()))
    rep = json.loads((STUDY / "report.json").read_text())
    stored = rep["insample_vs_oos"]["read_coef"]
    if not np.isclose(float(b[0]), stored):
        raise SystemExit(f"refit {b[0]} differs from report {stored}")
    folds = [v[0] for v in rep["coef_per_fold"]["read"].values()]
    body = {
        "family": "lead108_wording_mixture",
        "term": "lead108_read_lineup",
        "coefficient_logit_per_point": float(b[0]),
        "rule": (
            "single slope, logistic with offset logit(served card), fitted on all graded 2026 "
            "games of weeks 2-9; no earlier season carries the read; week 10 excluded"
        ),
        "lowo_fold_mean": float(np.mean(folds)),
        "fit_games": len(g),
        "fit_weeks": sorted(int(w) for w in g["week"].unique()),
        "games_sha256": sha_file(STUDY / "games.parquet"),
        "values_spread_pts_per_out": rep["values"],
        "value_source": rep["value_source"],
        "first_week": FIRST_WEEK,
        "last_week": LAST_WEEK,
        "lane": LANE,
        "no_week10_or_later_outcomes": True,
        "created_at_utc": datetime.now(UTC).strftime(STAMP),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(body, indent=1))
    print(json.dumps(body, indent=1))


def schedule(week):
    g = pd.read_parquet(
        ROOT / "data" / "processed" / "game_features.parquet",
        columns=["season", "week", "game_id", "home_team", "away_team", "kickoff", "game_type"],
    )
    g = g[(g["season"] == SEASON) & (g["week"] == week) & (g["game_type"] == "REG")].copy()
    g["kickoff_utc"] = pd.to_datetime(g["kickoff"], utc=True, format="ISO8601")
    dl = game_deadlines(g[["game_id"]].assign(kickoff=g["kickoff_utc"]))
    g["pick_deadline_utc"] = g["game_id"].map(dl)
    g["cutoff_utc"] = g[["kickoff_utc", "pick_deadline_utc"]].min(axis=1)
    return g.reset_index(drop=True)


def reads(week, sched, now):
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
        inj = pd.read_parquet(INJ / j["capture_utc"] / "injuries.parquet")
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
                    "gsis_id": o["gsis_id"],
                    "team": o["team"],
                    "position": pos[o["gsis_id"]],
                    "p_plays": float(o["p_plays"]),
                    "recorded_utc": rec,
                }
            )
            kept += 1
        files.append({"file": f.name, "enrolled": ok, "rows": len(j["rows"]), "kept": kept})
    x = pd.DataFrame(rows, columns=["gsis_id", "team", "position", "p_plays", "recorded_utc"])
    x = x.sort_values("recorded_utc").drop_duplicates("gsis_id", keep="last")
    return x.reset_index(drop=True), files


def team_x(x, values):
    if x.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    clip = 1.0 / (len(x) + 2)
    p = x["p_plays"].clip(clip, 1 - clip)
    v = x["position"].map(A.POS_TO_GROUP).map(values)
    t = (v * (1 - p)).fillna(0.0).groupby(x["team"]).sum()
    return t, x.groupby("team").size()


def served_now(week, now):
    pub = pd.read_parquet(ROOT / "artifacts" / "clv_ledger" / "published_picks.parquet")
    pub = pub[(pub["season"] == SEASON) & (pub["week"] == week)]
    pub = pub[pd.to_datetime(pub["published_at_utc"], utc=True) <= pd.Timestamp(now)]
    last = pub.sort_values("published_at_utc").groupby("game_id").tail(1)
    home = np.where(
        last["pick_team"] == last["home_team"], last["displayed_score"], 1 - last["displayed_score"]
    )
    return dict(zip(last["game_id"], home, strict=True)), dict(
        zip(last["game_id"], last["published_at_utc"].astype(str), strict=True)
    )


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
    if ahead.empty:
        raise SystemExit("no unplayed regular-season week")
    return int(ahead.min())


def cmd_pregame(a):
    now = datetime.now(UTC)
    fz = json.loads(FROZEN.read_text())
    week = resolve(a.week, now)
    if not a.dry and not FIRST_WEEK <= week <= LAST_WEEK:
        raise SystemExit(
            f"week {week} outside the declared forward window {FIRST_WEEK}-{LAST_WEEK}"
        )
    sched = schedule(week)
    x, files = reads(week, sched, now)
    tx, tn = team_x(x, fz["values_spread_pts_per_out"])
    base, base_at = served_now(week, now)
    b = fz["coefficient_logit_per_point"]
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
        hx = float(tx.get(r.home_team, 0.0))
        ax = float(tx.get(r.away_team, 0.0))
        p0 = float(base[r.game_id])
        dl = b * (hx - ax)
        games.append(
            {
                "game_id": r.game_id,
                "away_team": r.away_team,
                "home_team": r.home_team,
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "pick_deadline_utc": r.pick_deadline_utc.isoformat(),
                "players_read": n,
                "home_x": hx,
                "away_x": ax,
                "term_value": hx - ax,
                "base_home_probability": p0,
                "base_published_at_utc": base_at[r.game_id],
                "delta_logit": dl,
                "combined_home_probability": float(W.sig(W.logit(p0) + dl)),
                "side_changed": bool((p0 >= 0.5) != (W.sig(W.logit(p0) + dl) >= 0.5)),
            }
        )
    doc = {
        "season": SEASON,
        "week": week,
        "created_at_utc": now.strftime(STAMP),
        "frozen_created_at_utc": fz["created_at_utc"],
        "frozen_sha256": sha_file(FROZEN),
        "coefficient_logit_per_point": b,
        "read_files": files,
        "players_read": len(x),
        "skipped": skipped,
        "games": games,
    }
    if a.dry:
        g9 = pd.read_parquet(STUDY / "games.parquet")
        g9 = g9[g9["week"] == week].set_index("game_id")["x_read"]
        if len(g9) and games:
            mine = pd.Series({gm["game_id"]: gm["term_value"] for gm in games})
            both = pd.concat([mine, g9], axis=1, join="inner")
            doc["study_check"] = {
                "games": len(both),
                "corr_with_study_x_read": float(both.corr().iloc[0, 1]),
                "max_abs_diff": float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max()),
            }
        print(json.dumps({k: v for k, v in doc.items() if k != "games"}, indent=1, default=str))
        for gm in games:
            print(
                f"{gm['game_id']:<18} reads {gm['players_read']:>3} x {gm['term_value']:+.3f} "
                f"base {gm['base_home_probability']:.4f} term {gm['combined_home_probability']:.4f}"
                f" changed {gm['side_changed']}"
            )
        print("DRY: nothing written")
        return
    if not games:
        print(json.dumps({k: v for k, v in doc.items() if k != "games"}, indent=1, default=str))
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
    g = g.merge(t[["game_id", "term_value"]], on="game_id")
    g = g.dropna(subset=["open_line", "served_p"])
    g = g[g["result"] != g["open_line"]].copy()
    if g.empty:
        raise SystemExit("no settled forward game graded at the opener yet")
    y = (g["result"] > g["open_line"]).astype(float).to_numpy()
    wk = g["week"].to_numpy()
    ps = g["served_p"].to_numpy()
    pt = W.sig(W.logit(ps) + b * g["term_value"].to_numpy())
    rng = np.random.default_rng(W.SEED)
    rep = {
        "games": len(g),
        "weeks": sorted(int(w) for w in np.unique(wk)),
        "changed_pick_record_term_vs_served": W.record(pt, ps, y),
        "mean_log_loss": {"served": float(W.ll(ps, y).mean()), "term": float(W.ll(pt, y).mean())},
    }
    for name, f in (("log_loss", W.ll), ("brier", W.br)):
        d = f(ps, y) - f(pt, y)
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
            "LEAD-108 declared forward test: frozen read-weighted expected-lineup term added to "
            "the served card probability, weeks 11-18 of 2026. Research only; serves nothing."
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
