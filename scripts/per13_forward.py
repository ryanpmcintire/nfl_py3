import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
for sub in ("scripts", "src"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

import forward_terms_week as ftw  # noqa: E402
import per13_term_design as td  # noqa: E402
import sim09_stakes as ss  # noqa: E402

TERM = "per13_locked_rest"
FWD = td.FWD
FROZEN = FWD / "frozen_coefficients.json"
MANIFEST = FWD / "freeze_manifest.json"
REFRESH = FWD / "refresh"
SNAP_ROOT = ROOT / "data" / "raw_schedules"
STAKES_SUMMARY = ROOT / "artifacts" / "sim09" / "stakes_summary.json"
SEASON = 2026
FORWARD_WEEKS = (17, 18)
LAST_REG_WEEK = 18
PLAYOFF_SEEDS = 7
HISTORY_FIRST = 2009
ET = "America/New_York"


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def frozen():
    body = json.loads(FROZEN.read_text(encoding="utf-8"))
    sha = ftw.sha_file(FROZEN)
    if not MANIFEST.exists():
        raise SystemExit(
            "MISSING INPUT: freeze_manifest.json absent; run `per13_forward.py pin` once"
        )
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if man["frozen_sha256"] != sha or man["coefficient_hash"] != ftw.sha_obj(body["coef"]):
        raise SystemExit(
            f"FROZEN CHANGED: {rel(FROZEN)} sha256 {sha} differs from the pinned manifest"
        )
    return body, man, sha


def rating_params():
    return json.loads(STAKES_SUMMARY.read_text(encoding="utf-8"))["rating_params"]


def kickoff_utc(df):
    local = pd.to_datetime(df.gameday + " " + df.gametime.fillna("13:00"))
    return local.dt.tz_localize(ET).dt.tz_convert("UTC")


def newest_snapshot(now):
    stamps = sorted(p.name for p in SNAP_ROOT.iterdir() if (p / "schedules.parquet").exists())
    usable = [s for s in stamps if s <= ftw.stamp(now)]
    if not usable:
        raise SystemExit(
            f"MISSING INPUT: no schedule snapshot under {rel(SNAP_ROOT)} at or before run time"
        )
    return SNAP_ROOT / usable[-1]


def season_games(now):
    snap = newest_snapshot(now)
    cols = [
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "gametime",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "location",
    ]
    hist = pd.read_parquet(ss.SCHED, columns=cols)
    hist = hist[hist.season.between(HISTORY_FIRST, SEASON - 1)]
    cur = pd.read_parquet(snap / "schedules.parquet", columns=cols)
    cur = cur[cur.season == SEASON]
    s = pd.concat([hist, cur], ignore_index=True)
    s = s[s.game_type == "REG"].copy()
    s["home_team"] = ss.norm_team(s.home_team)
    s["away_team"] = ss.norm_team(s.away_team)
    s = s.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    s["hi"] = s.home_team.map(ss.TI)
    s["ai"] = s.away_team.map(ss.TI)
    s["margin"] = s.home_score - s.away_score
    s["neutral"] = (s.location == "Neutral").astype(float)
    s["kickoff_utc"] = kickoff_utc(s)
    future = (s.season == SEASON) & (s.kickoff_utc >= now)
    s.loc[future, "margin"] = np.nan
    tabs, last = ss.strength_tables(s)
    g = s[s.season == SEASON].reset_index(drop=True)
    played = g[g.margin.notna()]
    meta = {
        "schedule_snapshot": rel(snap),
        "schedule_sha256": ftw.sha_file(snap / "schedules.parquet"),
        "history_schedule": rel(ss.SCHED),
        "history_seasons": [HISTORY_FIRST, SEASON - 1],
        "games_played_used": len(played),
        "last_played_kickoff_utc": played.kickoff_utc.max().isoformat() if len(played) else None,
        "rating_params_file": rel(STAKES_SUMMARY),
        "rating_params_sha256": ftw.sha_file(STAKES_SUMMARY),
    }
    return g, tabs, last, meta


def wins_now(played):
    W = np.zeros(32)
    for h, a, m in zip(played.hi, played.ai, played.margin, strict=True):
        W[h] += 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
        W[a] += 1.0 if m < 0 else (0.5 if m == 0 else 0.0)
    return W


def term_inputs(g, tabs, last, P, week):
    ss.RNG = np.random.default_rng([td.SEED, SEASON, week])
    seed, _, _, K = ss.simulate(g, SEASON, week, tabs, last, P, ss.NSIM)
    pk = np.stack([(seed == q).mean(axis=0) for q in range(K + 1)], axis=1)
    modal = pk.argmax(axis=1)
    p_modal = pk.max(axis=1)
    p_make = (seed > 0).mean(axis=0)
    return pd.DataFrame(
        {
            "team": ss.TEAMS,
            "p_make": p_make,
            "seed_modal": modal,
            "p_seed_modal": p_modal,
            "x_lock": np.where(modal > 0, p_modal, 0.0),
            "x_elim": 1.0 - p_make,
        }
    ).set_index("team")


def forward_probabilities(g, tabs, last, P, w_now):
    rng = np.random.default_rng([td.SEED, SEASON, w_now, 0])
    ss.RNG = rng
    r = ss.rating(tabs, last, SEASON, w_now, P["k"], P["c"])
    played = g[g.margin.notna()]
    rem = g[g.margin.isna()]
    W0 = wins_now(played)

    def play(df, n):
        e = P["hfa"] * (1 - df.neutral.values) + P["slope"] * (r[df.hi.values] - r[df.ai.values])
        x = rng.random((n, len(df))) < norm.cdf(e / P["sigma"])
        H = np.zeros((len(df), 32))
        A = np.zeros((len(df), 32))
        H[np.arange(len(df)), df.hi.values] = 1
        A[np.arange(len(df)), df.ai.values] = 1
        return x @ H + (~x) @ A

    out = {"team": ss.TEAMS, "wins_now": W0}
    for wk in FORWARD_WEEKS:
        if w_now > wk:
            continue
        pre = rem[rem.week < wk]
        post = rem[rem.week >= wk]
        outer = W0 + play(pre, td.OUTER)
        acc = {k: np.zeros(32) for k in ("locked", "elim", "clinch", "x_lock", "x_elim")}
        for i in range(td.OUTER):
            wins = np.tile(outer[i], (td.INNER, 1)) + play(post, td.INNER)
            seed = ss.seeds_from_wins(wins, PLAYOFF_SEEDS)
            pk = np.stack([(seed == q).mean(axis=0) for q in range(PLAYOFF_SEEDS + 1)], axis=1)
            modal = pk.argmax(axis=1)
            pmax = pk.max(axis=1)
            pmk = 1.0 - pk[:, 0]
            acc["locked"] += (modal > 0) & (pmax >= 0.99)
            acc["elim"] += pmk < 0.01
            acc["clinch"] += pmk > 0.99
            acc["x_lock"] += np.where(modal > 0, pmax, 0.0)
            acc["x_elim"] += 1.0 - pmk
        for k, v in acc.items():
            name = f"p_{k}" if k in ("locked", "elim", "clinch") else f"expected_{k}"
            out[f"{name}_at_week{wk}_tuesday"] = v / td.OUTER
    return pd.DataFrame(out)


def cmd_pin(args):
    body = json.loads(FROZEN.read_text(encoding="utf-8"))
    sha = ftw.sha_file(FROZEN)
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        print("PINNED", man["frozen_sha256"], "MATCH" if man["frozen_sha256"] == sha else "CHANGED")
        return 0 if man["frozen_sha256"] == sha else 1
    mtime = datetime.fromtimestamp(FROZEN.stat().st_mtime, UTC).isoformat()
    man = {
        "term": TERM,
        "frozen_file": rel(FROZEN),
        "frozen_sha256": sha,
        "coefficient_hash": ftw.sha_obj(body["coef"]),
        "frozen_created_at_utc": body["created_at"],
        "frozen_file_mtime_utc": mtime,
        "pinned_at_utc": ftw.now_utc().isoformat(),
        "forward_season": SEASON,
        "forward_weeks": list(FORWARD_WEEKS),
        "feature_rule": (
            "sim09_stakes.simulate at the target week with every earlier week played, NSIM "
            "draws; x_lock = modal-seed probability if the modal seed is a playoff seed else 0; "
            "x_elim = 1 - p_make; d = home minus away"
        ),
        "term_rule": (
            "term_home = coef.d_lock * d_lock + coef.d_elim * d_elim, points of expected "
            "opener-to-close move toward the home team"
        ),
        "probability_rule": (
            "served card line_sweep rows of the method that reproduces pool_card.pick_probability "
            "at the pool line; combined home probability read at the half-point sweep line "
            "nearest to (home pool line - term_home), spread_line convention"
        ),
        "as_of_rule": (
            "current-season scores from the newest data/raw_schedules snapshot stamped at or "
            "before run time; a score counts only when its kickoff precedes run time; log fails "
            "closed if any game of an earlier week lacks a final score"
        ),
        "status": "forward-logged challenger only; not served",
    }
    MANIFEST.write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("PINNED", sha)
    return 0


def cmd_refresh(args):
    now = ftw.now_utc()
    frozen()
    g, tabs, last, meta = season_games(now)
    open_weeks = g[g.margin.isna()].week
    if open_weeks.empty or int(open_weeks.min()) > LAST_REG_WEEK:
        print(f"SKIP: no unplayed {SEASON} regular-season week")
        return 0
    w_now = int(open_weeks.min())
    P = rating_params()
    fp = forward_probabilities(g, tabs, last, P, w_now)
    cur = term_inputs(g, tabs, last, P, w_now).add_suffix(f"_asof_week{w_now}_tuesday")
    df = fp.merge(cur, left_on="team", right_index=True)
    doc = {
        "term": TERM,
        "season": SEASON,
        "current_week": w_now,
        "created_at_utc": now.isoformat(),
        "runner": "scripts/per13_forward.py refresh",
        "as_of": meta,
        "definitions": (
            "locked = modal seed is a playoff seed with probability >= 0.99; elim = p_make < "
            "0.01; clinch = p_make > 0.99 (sim09_stakes.team_weeks flags, the training "
            "definitions); expected_x_* are the term inputs averaged over simulated paths to "
            "that Tuesday"
        ),
        "outer_draws": td.OUTER,
        "inner_draws": td.INNER,
        "teams": df.round(6).to_dict(orient="records"),
    }
    out = Path(args.out_dir) if args.out_dir else REFRESH
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{SEASON}_week{w_now:02d}_{ftw.stamp(now)}.json"
    path.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    show = [c for c in df.columns if c.startswith(("p_locked", "p_clinch", "p_elim"))]
    print("WROTE", rel(path) if path.resolve().is_relative_to(ROOT) else path)
    print(
        "as of",
        meta["last_played_kickoff_utc"],
        "games played",
        meta["games_played_used"],
        "current week",
        w_now,
    )
    print(
        df.sort_values(show[0], ascending=False)[["team", "wins_now", *show]]
        .head(12)
        .round(3)
        .to_string(index=False)
    )
    print(
        df.sort_values(show[1], ascending=False)[["team", "wins_now", *show]]
        .head(8)
        .round(3)
        .to_string(index=False)
    )
    return 0


def served_method(sweep, card):
    home_line = np.where(card.pool_side.str.upper() == "HOME", -card.pick_line, card.pick_line)
    ref = card.assign(home_line=home_line)[["game_id", "home_line", "base_home_probability"]]
    for m, rows in sweep.groupby("method"):
        z = rows.merge(ref, on="game_id")
        z = z[np.isclose(z.alternative_line, z.home_line)]
        if z.game_id.nunique() == len(ref) and np.allclose(
            z.home_cover_probability, z.base_home_probability
        ):
            return m
    raise SystemExit(
        "MISSING INPUT: no line_sweep method reproduces the served pool card probability"
    )


def cmd_log(args):
    now = ftw.now_utc()
    week = ftw.resolve_week(args.week)
    if week not in FORWARD_WEEKS and not args.any_week:
        print(f"SKIP: week {week} is outside the declared forward window {list(FORWARD_WEEKS)}")
        return 0
    if args.any_week and not args.out_dir:
        raise SystemExit("--any-week writes only to --out-dir")
    body, man, sha = frozen()
    g, tabs, last, meta = season_games(now)
    stale = g[(g.week < week) & g.margin.isna()]
    if len(stale):
        raise SystemExit(
            f"STALE INPUT: {len(stale)} games before week {week} lack a final score in "
            f"{meta['schedule_snapshot']}"
        )
    P = rating_params()
    x = term_inputs(g, tabs, last, P, week)
    card, cmeta = ftw.base_card(week)
    pc = pd.read_csv(
        ROOT / cmeta["card_directory"] / "pool_card.csv", usecols=["game_id", "pick_line"]
    )
    card = card.merge(pc, on="game_id", how="left")
    sweep = pd.read_parquet(ROOT / cmeta["card_directory"] / "line_sweep.parquet")
    method = served_method(sweep, card)
    sweep = sweep[sweep.method == method]
    c = body["coef"]
    games = []
    for r in card.itertuples():
        h, a = ss.norm_team(pd.Series([r.home_team, r.away_team])).tolist()
        d_lock = float(x.loc[h, "x_lock"] - x.loc[a, "x_lock"])
        d_elim = float(x.loc[h, "x_elim"] - x.loc[a, "x_elim"])
        term = c["d_lock"] * d_lock + c["d_elim"] * d_elim
        home_line = -r.pick_line if str(r.pool_side).upper() == "HOME" else r.pick_line
        rows = sweep[sweep.game_id == r.game_id]
        target = home_line - term
        row = {
            "game_id": r.game_id,
            "away_team": r.away_team,
            "home_team": r.home_team,
            "kickoff_utc": r.kickoff_utc.isoformat(),
            "served_side": str(r.pool_side).upper(),
            "home_pool_line": float(home_line),
            "home_x_lock": float(x.loc[h, "x_lock"]),
            "away_x_lock": float(x.loc[a, "x_lock"]),
            "home_x_elim": float(x.loc[h, "x_elim"]),
            "away_x_elim": float(x.loc[a, "x_elim"]),
            "d_lock": d_lock,
            "d_elim": d_elim,
            "term_value": term,
            "base_home_probability": float(r.base_home_probability),
            "shifted_home_line": None,
            "delta_logit": None,
            "combined_home_probability": None,
            "challenger_side": None,
        }
        lo, hi = rows.alternative_line.min(), rows.alternative_line.max()
        if len(rows) and lo <= target <= hi:
            pick = rows.iloc[int(np.argmin(np.abs(rows.alternative_line.to_numpy() - target)))]
            p = float(pick.home_cover_probability)
            row["shifted_home_line"] = float(pick.alternative_line)
            row["combined_home_probability"] = p
            row["delta_logit"] = float(ftw.logit(p) - ftw.logit(r.base_home_probability))
            row["challenger_side"] = "HOME" if p > 0.5 else "AWAY"
        games.append(row)
    doc = {
        "term": TERM,
        "season": SEASON,
        "week": week,
        "created_at_utc": now.isoformat(),
        "runner": "scripts/per13_forward.py log",
        "coefficient_hash": man["coefficient_hash"],
        "frozen_coefficients_file": rel(FROZEN),
        "frozen_sha256": sha,
        "frozen_coefficients": body,
        "frozen_created_at_utc": body["created_at"],
        "feature_timestamps": meta,
        "base": {**cmeta, "line_sweep_method": method},
        "games_scored_pregame": int(sum(pd.Timestamp(gm["kickoff_utc"]) > now for gm in games)),
        "games_with_term_value": int(sum(abs(gm["term_value"]) > 0 for gm in games)),
        "side_differs": [
            gm["game_id"]
            for gm in games
            if gm["challenger_side"] and gm["challenger_side"] != gm["served_side"]
        ],
        "scoring_rule": (
            "score only after the week settles, against the opener, alongside the served card; "
            "never served"
        ),
        "games": games,
    }
    out = Path(args.out_dir) if args.out_dir else FWD
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{SEASON}_week{week:02d}_{ftw.stamp(now)}.json"
    path.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    print("WROTE", path)
    t = pd.DataFrame(games)[
        [
            "game_id",
            "served_side",
            "home_pool_line",
            "d_lock",
            "d_elim",
            "term_value",
            "shifted_home_line",
            "base_home_probability",
            "combined_home_probability",
            "challenger_side",
        ]
    ]
    print(t.round(3).to_string(index=False))
    print("side differs:", doc["side_differs"])
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=(
            "PER-13 locked-seed rest challenger: weekly lock refresh and Weeks 17-18 forward log"
        )
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "pin", help="write freeze_manifest.json once from the frozen coefficients, or verify it"
    )
    rf = sub.add_parser(
        "refresh",
        help="as-of lock, clinch and elimination probabilities for the Week 17 and 18 Tuesdays",
    )
    rf.add_argument("--out-dir", default=None, help="write here instead of the refresh folder")
    lg = sub.add_parser(
        "log", help="frozen challenger per-game values for Weeks 17-18; SKIP elsewhere"
    )
    lg.add_argument("--week", default="auto", help="week number or auto (latest served card week)")
    lg.add_argument(
        "--any-week", action="store_true", help="compute outside Weeks 17-18; requires --out-dir"
    )
    lg.add_argument("--out-dir", default=None, help="write here instead of the term folder")
    args = ap.parse_args()
    return {"pin": cmd_pin, "refresh": cmd_refresh, "log": cmd_log}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
