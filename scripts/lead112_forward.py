import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead106_injury_language as L6
import lead108_forward as F8
import lead108_wording_mixture as W
import lead110_absorption as A

ROOT = L6.ROOT
SEASON = 2026
FIRST_WEEK = 11
LAST_WEEK = 18
FIT_WEEKS = (2, 9)
OUT = ROOT / "artifacts" / "lead112_forward"
FROZEN = OUT / "frozen.json"
SNAPS = ROOT / "data" / "raw" / "nflverse_injuries"
LEDGER = ROOT / "artifacts" / "forward_ledger_core" / "ledger.parquet"
LANE = "docs/lanes/lead112-forward-2026-11-16.md"
STAMP = F8.STAMP
COLS = ["season", "week", "team", "gsis_id", "position", "practice_status"]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def snapshots(weeks):
    parts = []
    for f in sorted(SNAPS.glob("*/injuries.parquet")):
        try:
            stamp = F8.ts(f.parent.name)
        except ValueError:
            continue
        x = pd.read_parquet(f, columns=COLS)
        x = x[(x["season"] == SEASON) & x["week"].isin(weeks)]
        if x.empty:
            continue
        x = x.assign(observed=stamp, mtime=pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC"))
        parts.append(x)
    if not parts:
        return pd.DataFrame(columns=[*COLS, "observed", "mtime"])
    s = pd.concat(parts, ignore_index=True)
    s["week"] = s["week"].astype(int)
    return s


def ladder(week):
    pop = L6.id_outcomes(pd.read_parquet(L6.ART / "population.parquet"))
    v = pop[pop["verified"] & pop["practice_status"].notna() & (pop["week"] < week)]
    if v.empty:
        return None, 0
    d = v.groupby("practice_status")["played"].agg(["size", "sum"])
    return ((d["sum"] + 1) / (d["size"] + 2)).to_dict(), len(v)


def team_terms(snap, week, team, cutoff, now, lad, values):
    t = snap[(snap["week"] == week) & (snap["team"] == team)]
    t = t[(t["observed"] < cutoff) & (t["mtime"] < cutoff) & (t["observed"] <= now)]
    if t.empty or lad is None:
        return 0.0, 0, 0, None, None
    t0, t1 = t["observed"].min(), t["observed"].max()
    first = t[t["observed"] == t0].drop_duplicates("gsis_id", keep="last").set_index("gsis_id")
    last = t[t["observed"] == t1].drop_duplicates("gsis_id", keep="last").set_index("gsis_id")
    both = first.index.intersection(last.index)
    x, changed = 0.0, 0
    for gid in both:
        a, b = first.at[gid, "practice_status"], last.at[gid, "practice_status"]
        if pd.isna(a) or pd.isna(b) or a not in lad or b not in lad:
            continue
        v = values.get(A.POS_TO_GROUP.get(last.at[gid, "position"]))
        if v is None or a == b:
            continue
        x += v * (lad[a] - lad[b])
        changed += 1
    return x, len(both), changed, t0, t1


def game_term(snap, week, home, away, cutoff, now, lad, values):
    h = team_terms(snap, week, home, cutoff, now, lad, values)
    a = team_terms(snap, week, away, cutoff, now, lad, values)
    return {
        "term_value": h[0] - a[0],
        "home_x": h[0],
        "away_x": a[0],
        "players_both_reports": h[1] + a[1],
        "players_changed": h[2] + a[2],
        "home_first_report_utc": str(h[3]),
        "home_last_report_utc": str(h[4]),
        "away_first_report_utc": str(a[3]),
        "away_last_report_utc": str(a[4]),
    }


def fit_games():
    c = pd.read_parquet(LEDGER)
    c = c[c["status"].eq("graded") & c["week"].between(*FIT_WEEKS)].copy()
    c["away"] = c["game_id"].str.split("_").str[2]
    c["home"] = c["game_id"].str.split("_").str[3]
    c["cut"] = pd.to_datetime(c["deadline_utc"], utc=True)
    weeks = sorted(int(w) for w in c["week"].unique())
    snap = snapshots(weeks)
    values = json.loads(F8.FROZEN.read_text())["values_spread_pts_per_out"]
    lads = {w: ladder(w) for w in weeks}
    rows = []
    for r in c.itertuples():
        w = int(r.week)
        gt = game_term(snap, w, r.home, r.away, r.cut, r.cut, lads[w][0], values)
        rows.append(
            {
                "game_id": r.game_id,
                "week": w,
                "y": float(r.home_cover),
                "served": float(r.served_p),
                **gt,
                "ladder_rows": lads[w][1],
            }
        )
    return pd.DataFrame(rows), values


def cmd_freeze(a):
    if FROZEN.exists():
        raise SystemExit(f"already frozen: {FROZEN}")
    g, values = fit_games()
    if g["week"].max() >= 10:
        raise SystemExit("fit population reaches week 10")
    X = g[["term_value"]].to_numpy()
    y = g["y"].to_numpy()
    off = W.logit(g["served"].to_numpy())
    b = W.fit(X, y, off)
    wk = g["week"].to_numpy()
    p_oos, coefs = W.lowo(X, y, off, wk)
    p_in = W.sig(off + X @ b)
    ps = g["served"].to_numpy()
    folds = [float(np.ravel(v)[0]) for v in coefs.values()]
    OUT.mkdir(parents=True, exist_ok=True)
    g.to_parquet(OUT / "fit_games.parquet")
    body = {
        "family": "lead112_post_opener_availability",
        "term": "lead112_practice_change",
        "coefficient_logit_per_point": float(b[0]),
        "rule": (
            "single slope, logistic with offset logit(served card), no intercept, fitted on all "
            "graded 2026 games of weeks 2-9 in forward_ledger_core; week 10 excluded"
        ),
        "fit_games": len(g),
        "fit_weeks": sorted(int(w) for w in np.unique(wk)),
        "games_nonzero_term": int((g["term_value"] != 0).sum()),
        "players_changed": int(g["players_changed"].sum()),
        "lowo_coef_per_fold": dict(zip([str(int(w)) for w in coefs], folds, strict=True)),
        "lowo_fold_mean": float(np.mean(folds)),
        "log_loss": {
            "served": float(W.ll(ps, y).mean()),
            "in_sample": float(W.ll(p_in, y).mean()),
            "lowo": float(W.ll(p_oos, y).mean()),
        },
        "fit_games_sha256": sha_file(OUT / "fit_games.parquet"),
        "values_spread_pts_per_out": values,
        "value_source": "artifacts/lead108_forward/frozen.json (LEAD-110 OLS 2011-2025)",
        "first_week": FIRST_WEEK,
        "last_week": LAST_WEEK,
        "lane": LANE,
        "no_week10_or_later_outcomes": True,
        "created_at_utc": datetime.now(UTC).strftime(STAMP),
    }
    FROZEN.write_text(json.dumps(body, indent=1))
    print(json.dumps(body, indent=1))


def cmd_pregame(a):
    now = datetime.now(UTC)
    fz = json.loads(FROZEN.read_text())
    week = F8.resolve(a.week, now)
    if not a.dry and not FIRST_WEEK <= week <= LAST_WEEK:
        raise SystemExit(
            f"week {week} outside the declared forward window {FIRST_WEEK}-{LAST_WEEK}"
        )
    sched = F8.schedule(week)
    snap = snapshots([week])
    lad, nlad = ladder(week)
    base, base_at = F8.served_now(week, now)
    b = fz["coefficient_logit_per_point"]
    games, skipped = [], {}
    for r in sched.itertuples():
        if not a.dry and r.cutoff_utc <= pd.Timestamp(now):
            skipped[r.game_id] = "past_cutoff"
            continue
        gt = game_term(
            snap,
            week,
            r.home_team,
            r.away_team,
            r.cutoff_utc,
            pd.Timestamp(now),
            lad,
            fz["values_spread_pts_per_out"],
        )
        if gt["players_both_reports"] == 0:
            skipped[r.game_id] = "no_report_pair"
            continue
        if r.game_id not in base:
            skipped[r.game_id] = "no_published_card"
            continue
        p0 = float(base[r.game_id])
        dl = b * gt["term_value"]
        p1 = float(W.sig(W.logit(p0) + dl))
        games.append(
            {
                "game_id": r.game_id,
                "away_team": r.away_team,
                "home_team": r.home_team,
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "pick_deadline_utc": r.pick_deadline_utc.isoformat(),
                **gt,
                "base_home_probability": p0,
                "base_published_at_utc": base_at[r.game_id],
                "delta_logit": dl,
                "combined_home_probability": p1,
                "side_changed": bool((p0 >= 0.5) != (p1 >= 0.5)),
            }
        )
    doc = {
        "season": SEASON,
        "week": week,
        "created_at_utc": now.strftime(STAMP),
        "frozen_created_at_utc": fz["created_at_utc"],
        "frozen_sha256": sha_file(FROZEN),
        "coefficient_logit_per_point": b,
        "ladder_rows": nlad,
        "snapshots": int(snap["observed"].nunique()),
        "skipped": skipped,
        "games": games,
    }
    if a.dry or not games:
        print(json.dumps({k: v for k, v in doc.items() if k != "games"}, indent=1, default=str))
        for gm in games:
            print(
                f"{gm['game_id']:<18} pairs {gm['players_both_reports']:>3} "
                f"changed {gm['players_changed']:>2} x {gm['term_value']:+.3f} "
                f"base {gm['base_home_probability']:.4f} term {gm['combined_home_probability']:.4f}"
                f" side_changed {gm['side_changed']}"
            )
        print("DRY: nothing written" if a.dry else "no game with a report pair; nothing written")
        return
    out = OUT / f"{SEASON}_week{week:02d}_{now.strftime(STAMP)}.json"
    out.write_text(json.dumps(doc, indent=1, default=str))
    print(out, len(games), "games")


def cmd_score(a):
    F8.OUT = OUT
    F8.FROZEN = FROZEN
    F8.cmd_score(a)


def main():
    p = argparse.ArgumentParser(
        description=(
            "LEAD-112 declared forward test: frozen Wednesday-to-Friday practice-status change "
            "term added to the served card probability, weeks 11-18 of 2026. Research only; "
            "serves nothing."
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
