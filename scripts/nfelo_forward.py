import argparse
import hashlib
import io
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forward_ledger as FL
import lead108_wording_mixture as W
import mod24_ftn_turnover_luck as m
import mod25_wf_forward as mwf
import outside_model_gap as O

from nfl_ats.published_picks import game_deadlines

ROOT = O.ROOT
SEASON = 2026
FIRST_WEEK = 11
LAST_WEEK = 18
OUT = ROOT / "artifacts" / "nfelo_forward"
CAPTURES = OUT / "captures"
FROZEN = OUT / "frozen.json"
STUDY = ROOT / "artifacts" / "outside_model_gap" / "served.json"
LANE = "docs/lanes/nfelo-forward-2026-11-16.md"
STAMP = "%Y%m%dT%H%M%SZ"
REPO = "greerreNFL/nfelo"
PATH = "output_data/nfelo_games.csv"
UA = "research-client/1.0"
ALT = {"OAK": "LV", "LAR": "LA", "WSH": "WAS", "JAC": "JAX", "SD": "LAC", "STL": "LA"}


def ts(s):
    return pd.Timestamp(datetime.strptime(s, STAMP).replace(tzinfo=UTC))


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def code(t):
    t = str(t).strip().upper()
    return ALT.get(t, t)


def cmd_freeze(a):
    if FROZEN.exists():
        raise SystemExit(f"already frozen: {FROZEN}")
    study = json.loads(STUDY.read_text())
    m36 = __import__("xlg14_fumble_luck_nfl_confirmation").load_mod36()
    d, conv = O.frame()
    if conv["pick_artifact"] != study["pick_artifact"]:
        raise SystemExit(
            f"pick artifact {conv['pick_artifact']} differs from {study['pick_artifact']}"
        )
    _, g = m36.games()
    g = g[["game_id", "season", "home_covered", m36.SERVED_COLUMN]].merge(
        d[["game_id", "tue_open_home_spread", "nfelo_margin"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    gap = (g["nfelo_margin"] - g["tue_open_home_spread"]).fillna(0.0)
    if len(g) != study["games"] or int((gap != 0).sum()) != study["nonzero_games"]:
        raise SystemExit(f"population {len(g)}/{int((gap != 0).sum())} differs from study")
    sd = float(gap.std(ddof=1))
    folds = study["fold_coefficients_per_sd"]
    per_sd = float(np.mean(folds))
    body = {
        "family": "outside_model_gap_2026_11_16",
        "term": "nfelo_gap_forward",
        "term_definition": "points_per_elo * nfelo_dif_base minus Tuesday-open home margin line",
        "coefficient_logit_per_point": per_sd / sd,
        "loso_mean_per_sd": per_sd,
        "fold_coefficients_per_sd": folds,
        "term_sd_points": sd,
        "points_per_elo": conv["nfelo_points_per_elo"],
        "points_per_elo_rows": conv["nfelo_conversion_rows"],
        "study_games": len(g),
        "study_sha256": sha_file(STUDY),
        "source": f"github {REPO} {PATH} at a pinned commit",
        "first_week": FIRST_WEEK,
        "last_week": LAST_WEEK,
        "lane": LANE,
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


def fetch_week(week):
    c = json.loads(get(f"https://api.github.com/repos/{REPO}/commits?path={PATH}&per_page=1"))[0]
    sha = c["sha"]
    commit = pd.Timestamp(c["commit"]["committer"]["date"]).tz_convert("UTC")
    body = get(f"https://raw.githubusercontent.com/{REPO}/{sha}/{PATH}")
    n = pd.read_csv(io.BytesIO(body), usecols=["game_id", "nfelo_dif_base"])
    parts = n["game_id"].str.split("_", expand=True)
    n = n[(parts[0] == str(SEASON)) & (parts[1] == f"{week:02d}")].copy()
    if n.empty:
        return sha, commit, hashlib.sha256(body).hexdigest(), n.assign(repo_game_id=[])
    p = n["game_id"].str.split("_", expand=True)
    n["repo_game_id"] = p[0] + "_" + p[1] + "_" + p[2].map(code) + "_" + p[3].map(code)
    return sha, commit, hashlib.sha256(body).hexdigest(), n.dropna(subset=["nfelo_dif_base"])


def cmd_capture(a):
    now = datetime.now(UTC)
    week = resolve(a.week, now)
    sha, commit, digest, n = fetch_week(week)
    sched = schedule(week)
    n = n[n["repo_game_id"].isin(sched["game_id"])]
    doc = {
        "season": SEASON,
        "week": week,
        "captured_at_utc": now.strftime(STAMP),
        "commit_sha": sha,
        "commit_utc": commit.strftime(STAMP),
        "file_sha256": digest,
        "schedule_games": len(sched),
        "rows": [
            {
                "game_id": r.repo_game_id,
                "nfelo_game_id": r.game_id,
                "nfelo_dif_base": float(r.nfelo_dif_base),
            }
            for r in n.itertuples()
        ],
    }
    print(json.dumps({k: v for k, v in doc.items() if k != "rows"}, indent=1), len(n), "rows")
    if n.empty:
        print(f"week {week} not yet published at commit {sha[:8]}; nothing written")
        return
    if a.dry:
        print("DRY: nothing written")
        return
    CAPTURES.mkdir(parents=True, exist_ok=True)
    out = CAPTURES / f"{SEASON}_week{week:02d}_{now.strftime(STAMP)}.json"
    out.write_text(json.dumps(doc, indent=1))
    print(out)


def captured(week, sched, now, dry):
    cut = dict(zip(sched["game_id"], sched["cutoff_utc"], strict=True))
    best = {}
    files = sorted(CAPTURES.glob(f"{SEASON}_week{week:02d}_*.json"))
    for f in files:
        j = json.loads(f.read_text())
        cap, com = ts(j["captured_at_utc"]), ts(j["commit_utc"])
        mtime = pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC")
        for r in j["rows"]:
            c = cut.get(r["game_id"])
            if c is None or cap > pd.Timestamp(now):
                continue
            ok = dry or (cap < c and com < c and mtime < c)
            if ok and (r["game_id"] not in best or best[r["game_id"]][1] < cap):
                best[r["game_id"]] = (
                    r["nfelo_dif_base"],
                    cap,
                    j["commit_sha"],
                    j["commit_utc"],
                )
    if dry and not best:
        sha, commit, _, n = fetch_week(week)
        for r in n.itertuples():
            best[r.repo_game_id] = (
                float(r.nfelo_dif_base),
                pd.Timestamp(now),
                sha,
                commit.strftime(STAMP),
            )
    return best, [f.name for f in files]


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


def cmd_pregame(a):
    now = datetime.now(UTC)
    fz = json.loads(FROZEN.read_text())
    week = resolve(a.week, now)
    if not a.dry and not FIRST_WEEK <= week <= LAST_WEEK:
        raise SystemExit(
            f"week {week} outside the declared forward window {FIRST_WEEK}-{LAST_WEEK}"
        )
    sched = schedule(week)
    nf, files = captured(week, sched, now, a.dry)
    op = mwf.tuesday_open(sched.assign(kickoff=sched["kickoff_utc"]))
    base, base_at = served_now(week, now)
    b = fz["coefficient_logit_per_point"]
    games, skipped = [], {}
    for r in sched.itertuples():
        if not a.dry and r.cutoff_utc <= pd.Timestamp(now):
            skipped[r.game_id] = "past_cutoff"
            continue
        if r.game_id not in nf:
            skipped[r.game_id] = "no_admissible_capture"
            continue
        if r.game_id not in op:
            skipped[r.game_id] = "no_tuesday_open"
            continue
        if r.game_id not in base:
            skipped[r.game_id] = "no_published_card"
            continue
        dif, cap, sha, com = nf[r.game_id]
        margin = fz["points_per_elo"] * dif
        line = op[r.game_id][0]
        x = margin - line
        p0 = float(base[r.game_id])
        pt = float(W.sig(W.logit(p0) + b * x))
        games.append(
            {
                "game_id": r.game_id,
                "away_team": r.away_team,
                "home_team": r.home_team,
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "pick_deadline_utc": r.pick_deadline_utc.isoformat(),
                "commit_sha": sha,
                "commit_utc": com,
                "captured_at_utc": pd.Timestamp(cap).strftime(STAMP),
                "nfelo_margin": margin,
                "open_line": line,
                "open_observed_utc": op[r.game_id][1],
                "term_value": x,
                "base_home_probability": p0,
                "base_published_at_utc": base_at[r.game_id],
                "delta_logit": b * x,
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
        "capture_files": files,
        "skipped": skipped,
        "games": games,
    }
    if a.dry:
        if len(games) > 2:
            gm = pd.DataFrame(games)
            doc["margin_open_corr"] = float(np.corrcoef(gm["nfelo_margin"], gm["open_line"])[0, 1])
        print(json.dumps({k: v for k, v in doc.items() if k != "games"}, indent=1, default=str))
        for gm in games:
            print(
                f"{gm['game_id']:<18} nfelo {gm['nfelo_margin']:+6.2f} "
                f"open {gm['open_line']:+5.1f} "
                f"x {gm['term_value']:+6.2f} base {gm['base_home_probability']:.4f} "
                f"term {gm['combined_home_probability']:.4f} changed {gm['side_changed']}"
            )
        print("DRY: nothing written")
        return
    if not games:
        print(json.dumps({k: v for k, v in doc.items() if k != "games"}, indent=1, default=str))
        print(
            "no game with an admissible capture, opener and card before its cutoff; nothing written"
        )
        return
    out = OUT / f"{SEASON}_week{week:02d}_{now.strftime(STAMP)}.json"
    out.write_text(json.dumps(doc, indent=1, default=str))
    print(out, len(games), "games")


def cmd_score(a):
    if not FROZEN.exists():
        print("not frozen; nothing graded")
        return
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
            if ts(d["created_at_utc"]) < cut and mtime < cut and ts(gm["commit_utc"]) < cut:
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
        print("no settled forward game graded at the opener yet; nothing graded")
        return
    y = (g["result"] > g["open_line"]).astype(float).to_numpy()
    wk = g["week"].to_numpy()
    ps = g["served_p"].to_numpy()
    pt = W.sig(W.logit(ps) + b * g["term_value"].to_numpy())
    rng = np.random.default_rng(W.SEED)
    rep = {
        "games": len(g),
        "nonzero_games": int((g["term_value"] != 0).sum()),
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
            "Declared forward test of nfelo's pregame model margin minus the pool opener, added "
            "with a frozen coefficient to the served card probability, weeks 11-18 of 2026. "
            "Research only."
        )
    )
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("freeze").set_defaults(fn=cmd_freeze)
    cp = sp.add_parser("capture")
    cp.add_argument("--week", default="auto")
    cp.add_argument("--dry", action="store_true")
    cp.set_defaults(fn=cmd_capture)
    pg = sp.add_parser("pregame")
    pg.add_argument("--week", default="auto")
    pg.add_argument("--dry", action="store_true")
    pg.set_defaults(fn=cmd_pregame)
    sp.add_parser("score").set_defaults(fn=cmd_score)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
