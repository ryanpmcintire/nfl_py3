import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mod24_u8_markets import ET, build_lookup, latest_schedule, match_game  # noqa: E402
from pool_number_price import (  # noqa: E402
    BOVADA_EVENT,
    BOVADA_LIST,
    NAME_TO_CODE,
    SLOTS,
    Polite,
    american,
    pick_deadline,
)

from nfl_ats.odds import no_vig_probabilities  # noqa: E402

OUT = ROOT / "data" / "external" / "props"
LOG = OUT / "forward_log.parquet"
GRADED = OUT / "graded.parquet"
MAPPING = OUT / "points_per_yard.json"
PBP = ROOT / "data" / "pbp" / "raw"
SPLASH = ROOT / "data" / "splash"
POLICY = "bovada_player_props"
GROUPS = (
    "Alternate Lines",
    "Passing Yards",
    "Passing Props",
    "Rushing Yards",
    "Rushing Props",
    "TD Scorer Props",
)
PLAYER_MARKETS = {
    "Total Passing Yards": "pass_yds",
    "Total Rushing Yards": "rush_yds",
    "Total Passing Attempts": "pass_att",
    "Total Rush Attempts": "rush_att",
}
PLAYER_RE = re.compile(r"^(" + "|".join(PLAYER_MARKETS) + r") - (.+) \(([A-Z]{2,3})\)$")
NAMED_RE = re.compile(r"^(.+) \(([A-Z]{2,3})\)$")
TEAM_TOTAL = "Total Points - "
ANYTIME = "Anytime Touchdown Scorer"
ALIASES = {"JAC": "JAX", "WSH": "WAS", "LAR": "LA", "LVR": "LV"}
KINDS = ("pass_yds", "rush_yds", "pass_att", "rush_att", "team_total", "anytime_td")
USED = ("pass_yds", "rush_yds", "team_total")
MEDIAN_PROBABILITY = 0.5
MAPPING_SEASONS = (2020, 2025)
FORWARD_SEASON = 2026
FORWARD_FIRST_WEEK = 11
FORWARD_LAST_WEEK = 18
FORWARD_MIN_WEEKS = 4
SCORE_SLOT = {"day": "tue", "at": "09:10"}


def code(abbr):
    return ALIASES.get(abbr, abbr)


def ladder(m):
    by = {}
    for o in m.get("outcomes", []):
        pr = o.get("price") or {}
        h = pd.to_numeric(pr.get("handicap"), errors="coerce")
        if o.get("type") in ("O", "U") and np.isfinite(h):
            by.setdefault(float(h), {})[o["type"]] = american(pr.get("american"))
    out = []
    for h, s in by.items():
        if np.isfinite(s.get("O", np.nan)) and np.isfinite(s.get("U", np.nan)):
            out.append((h, s["O"], s["U"], no_vig_probabilities(s["O"], s["U"])[0]))
    return out


def parse_event(ev, stamp):
    comps = {c.get("home"): c.get("name") for c in ev.get("competitors", [])}
    base = {
        "captured_at": stamp,
        "home": NAME_TO_CODE.get(comps.get(True)),
        "away": NAME_TO_CODE.get(comps.get(False)),
        "kickoff": pd.to_datetime(ev.get("startTime"), unit="ms", utc=True),
    }
    rows = []
    for dg in ev.get("displayGroups", []):
        for m in dg.get("markets", []):
            if (m.get("period") or {}).get("description") not in ("Game", "Match"):
                continue
            d = m.get("description", "")
            mp = PLAYER_RE.match(d)
            if mp or d.startswith(TEAM_TOTAL):
                kind = PLAYER_MARKETS[mp[1]] if mp else "team_total"
                player = mp[2] if mp else None
                team = code(mp[3]) if mp else NAME_TO_CODE.get(d[len(TEAM_TOTAL) :])
                for h, ov, un, p in ladder(m):
                    rows.append(
                        {
                            **base,
                            "kind": kind,
                            "player": player,
                            "team": team,
                            "handicap": h,
                            "over_american": ov,
                            "under_american": un,
                            "p_over": p,
                        }
                    )
            elif d == ANYTIME:
                for o in m.get("outcomes", []):
                    mt = NAMED_RE.match(o.get("description", ""))
                    rows.append(
                        {
                            **base,
                            "kind": "anytime_td",
                            "player": mt[1] if mt else o.get("description"),
                            "team": code(mt[2]) if mt else None,
                            "handicap": np.nan,
                            "over_american": american((o.get("price") or {}).get("american")),
                            "under_american": np.nan,
                            "p_over": np.nan,
                        }
                    )
    return rows


def coverage(P, n_events, stamp, requests):
    c = {"captured_at": stamp, "events": n_events, "requests": requests}
    if not len(P):
        return c
    c["games"] = int(P.game_id.nunique())
    c["events_without_game_id"] = int(
        P[P.game_id.isna()][["home", "away"]].drop_duplicates().shape[0]
    )
    for kind in KINDS:
        k = P[P.kind == kind]
        side = np.where(k.team == k.home, "h", np.where(k.team == k.away, "a", "x"))
        k = k.assign(side=side)
        both = k[k.side != "x"].groupby("game_id").side.nunique()
        c[kind] = {
            "games_any": int(k.game_id.nunique()),
            "games_both_teams": int((both == 2).sum()),
            "players": int(k[["team", "player"]].drop_duplicates().shape[0]),
        }
    c["unmatched_team_codes"] = sorted(
        set(P.team.dropna()) - set(P.home.dropna()) - set(P.away.dropna())
    )
    return c


def capture():
    now = dt.datetime.now(dt.UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out = OUT / "live" / stamp
    out.mkdir(parents=True, exist_ok=True)
    b = Polite(POLICY)
    lst = b.get(BOVADA_LIST, {"marketFilterId": "def", "preMatchOnly": "true", "lang": "en"}) or []
    events = [e for g in (lst if isinstance(lst, list) else [lst]) for e in g.get("events", [])]
    kept, rows = [], []
    for e in events:
        if e.get("live") or pd.to_datetime(e.get("startTime"), unit="ms", utc=True) <= now:
            continue
        j = b.get(BOVADA_EVENT + e["link"], {"lang": "en"})
        for x in [
            x for g in (j if isinstance(j, list) else [j or {}]) for x in g.get("events", [])
        ]:
            x["displayGroups"] = [
                dg for dg in x.get("displayGroups", []) if dg.get("description") in GROUPS
            ]
            kept.append(x)
            rows += parse_event(x, stamp)
    (out / "bovada_props.json").write_text(json.dumps(kept), encoding="utf-8")
    P = pd.DataFrame(rows)
    if len(P):
        P = P.drop_duplicates(["home", "away", "kind", "player", "team", "handicap"])
        lk = build_lookup(latest_schedule()[0])
        keys = P[["home", "away", "kickoff"]].drop_duplicates().copy()
        keys["game_id"] = [
            getattr(
                match_game(lk, r.home, r.away, r.kickoff.tz_convert(ET).date()), "game_id", None
            )
            for r in keys.itertuples()
        ]
        P = P.merge(keys, on=["home", "away", "kickoff"], how="left")
        P.to_parquet(out / "props.parquet")
    else:
        P = pd.DataFrame(columns=["game_id", "home", "away", "kind", "team", "player"])
    print(json.dumps(coverage(P, len(kept), stamp, b.requests)))
    return P


def median_line(d):
    d = d.sort_values("handicap")
    h, p = d.handicap.to_numpy(float), d.p_over.to_numpy(float)
    for i in range(len(h) - 1):
        if p[i] >= MEDIAN_PROBABILITY >= p[i + 1] and p[i] > p[i + 1]:
            return h[i] + (p[i] - MEDIAN_PROBABILITY) / (p[i] - p[i + 1]) * (h[i + 1] - h[i])
    return h[np.argmin(np.abs(p - MEDIAN_PROBABILITY))]


def diagnostic(P, slope):
    rows = []
    for gid, g in P[P.game_id.notna() & P.kind.isin(USED)].groupby("game_id"):
        home, away = g.home.iloc[0], g.away.iloc[0]
        med = {}
        for (kind, team, player), d in g.groupby(["kind", "team", "player"], dropna=False):
            med.setdefault((kind, team), []).append((player, median_line(d)))
        r = {
            "game_id": gid,
            "captured_at": g.captured_at.iloc[0],
            "kickoff": g.kickoff.iloc[0],
            "home": home,
            "away": away,
        }
        missing = []
        for side, t in (("home", home), ("away", away)):
            qbs = med.get(("pass_yds", t), [])
            rush = med.get(("rush_yds", t), [])
            tt = med.get(("team_total", t), [])
            qb = max(qbs, key=lambda x: x[1]) if qbs else (None, np.nan)
            backs = [x for x in rush if x[0] != qb[0]]
            r[f"{side}_qb"] = qb[0]
            r[f"{side}_qb_pass_yds"] = qb[1]
            r[f"{side}_rushers"] = len(rush)
            r[f"{side}_rush_yds"] = sum(x[1] for x in rush) if backs else np.nan
            r[f"{side}_team_total"] = tt[0][1] if tt else np.nan
            r[f"{side}_yards"] = r[f"{side}_qb_pass_yds"] + r[f"{side}_rush_yds"]
            missing += [f"{side}_qb_passing"] if not qbs else []
            missing += [f"{side}_non_qb_rusher"] if not backs else []
            missing += [f"{side}_team_total"] if not tt else []
        r["missing"] = ",".join(missing)
        r["prop_margin"] = slope * (r["home_yards"] - r["away_yards"])
        r["totals_margin"] = r["home_team_total"] - r["away_team_total"]
        r["diagnostic"] = r["prop_margin"] - r["totals_margin"]
        r["lean"] = "HOME" if r["diagnostic"] > 0 else "AWAY" if r["diagnostic"] < 0 else None
        rows.append(r)
    return pd.DataFrame(rows)


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(x) - 2)
    se = np.sqrt(s2 / ((x - x.mean()) @ (x - x.mean())))
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "slope_se": float(se),
        "r2": float(r2),
        "n": len(x),
    }


def cmd_mapping(a):
    run = sorted(p for p in PBP.iterdir() if p.is_dir())[-1]
    cols = [
        "game_id",
        "season_type",
        "posteam",
        "down",
        "yards_gained",
        "complete_pass",
        "rush_attempt",
        "sack",
    ]
    seasons = range(MAPPING_SEASONS[0], MAPPING_SEASONS[1] + 1)
    pbp = pd.concat(
        [pd.read_parquet(run / f"season={s}", columns=cols).assign(season=s) for s in seasons]
    )
    pbp = pbp[pbp.season_type.eq("REG") & pbp.posteam.notna() & pbp.down.notna()]
    pbp = pbp.assign(
        passing_yards=np.where(pbp.complete_pass.eq(1) & pbp.sack.ne(1), pbp.yards_gained, 0.0),
        rushing_yards=np.where(pbp.rush_attempt.eq(1), pbp.yards_gained, 0.0),
    )
    y = (
        pbp.groupby(["game_id", "season", "posteam"])[["passing_yards", "rushing_yards"]]
        .sum()
        .reset_index()
    )
    sched_run = latest_schedule()[1]
    sc = pd.read_parquet(ROOT / "data" / "raw" / sched_run / "schedules.parquet").set_index(
        "game_id"
    )[["home_team", "home_score", "away_score"]]
    y = y.join(sc, on="game_id", how="inner").rename(columns={"home_team": "home"})
    y["points"] = np.where(y.posteam == y.home, y.home_score, y.away_score)
    y["yards"] = y.passing_yards + y.rushing_yards
    fit = ols(y.yards, y.points)
    hy = y[y.posteam == y.home].set_index("game_id")
    ay = y[y.posteam != y.home].set_index("game_id")
    g = hy[["yards", "points"]].join(ay[["yards", "points"]], rsuffix="_away", how="inner")
    check = ols(g.yards - g.yards_away, g.points - g.points_away)
    out = {
        "slope_points_per_yard": fit["slope"],
        "team_game_fit": fit,
        "by_season_slope": {
            int(s): round(ols(d.yards, d.points)["slope"], 5) for s, d in y.groupby("season")
        },
        "check_game_margin_on_yard_difference": check,
        "yards": (
            "regular-season team yards on completed non-sack passes plus rush attempts, "
            "scrimmage downs, nflverse play-by-play"
        ),
        "points": "final team score from the nflverse schedule",
        "schedule_run": sched_run,
        "seasons": list(MAPPING_SEASONS),
        "pbp_run": run.name,
        "fitted_at": dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    MAPPING.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def load_slope():
    if not MAPPING.exists():
        raise SystemExit("no points-per-yard mapping: run the mapping subcommand first")
    return float(json.loads(MAPPING.read_text(encoding="utf-8"))["slope_points_per_yard"])


def cmd_capture(a):
    capture()


def cmd_pregame(a):
    slope = load_slope()
    P = capture()
    D = diagnostic(P, slope) if len(P) else pd.DataFrame()
    if not len(D):
        print("no game with prop rows")
        return
    D["deadline"] = [pick_deadline(k) for k in D.kickoff]
    D["captured_utc"] = pd.to_datetime(D.captured_at, format="%Y%m%dT%H%M%SZ", utc=True)
    D["season"] = D.game_id.str.slice(0, 4).astype(int)
    D["week"] = D.game_id.str.slice(5, 7).astype(int)
    D["before_deadline"] = D.captured_utc <= D.deadline
    D["slope"] = slope
    L = pd.concat([pd.read_parquet(LOG), D]) if LOG.exists() else D
    L = L.drop_duplicates(["game_id", "captured_at"], keep="last")
    L.to_parquet(LOG)
    show = [
        "game_id",
        "home_qb_pass_yds",
        "home_rush_yds",
        "away_qb_pass_yds",
        "away_rush_yds",
        "home_team_total",
        "away_team_total",
        "prop_margin",
        "totals_margin",
        "diagnostic",
        "lean",
        "missing",
    ]
    print(D[show].round(2).to_string(index=False))
    ready = D.diagnostic.notna()
    print(f"games {len(D)}, diagnostic computed {int(ready.sum())}, log rows {len(L)}")


def pool_lines(week, cache):
    if week not in cache:
        files = sorted(SPLASH.glob(f"{FORWARD_SEASON}_week{week:02d}_*.json"))
        games = json.loads(files[-1].read_text(encoding="utf-8"))["games"] if files else []
        cache[week] = {g["game_id"]: float(g["home_spread"]) for g in games}
    return cache[week]


def record(side, settle):
    won = (settle > 0) == side.eq("HOME")
    return int(won.sum()), int((~won).sum())


def cmd_score(a):
    if not LOG.exists():
        print("nothing graded")
        return
    L = pd.read_parquet(LOG)
    L = L[
        L.before_deadline
        & L.season.eq(FORWARD_SEASON)
        & L.week.between(FORWARD_FIRST_WEEK, FORWARD_LAST_WEEK)
    ]
    F = L.sort_values("captured_utc").groupby("game_id").tail(1)
    run = latest_schedule()[1]
    res = pd.read_parquet(ROOT / "data" / "raw" / run / "schedules.parquet")[["game_id", "result"]]
    F = F.merge(res, on="game_id", how="left")
    cache = {}
    F["pool_home_spread"] = [
        pool_lines(w, cache).get(g, np.nan) for g, w in zip(F.game_id, F.week, strict=True)
    ]
    F["settle"] = F.result - F.pool_home_spread
    F["totals_lean"] = np.where(
        F.totals_margin > F.pool_home_spread,
        "HOME",
        np.where(F.totals_margin < F.pool_home_spread, "AWAY", None),
    )
    G = F[F.settle.notna() & F.settle.ne(0) & F.lean.notna()].copy()
    if not len(G):
        print("nothing graded")
        return
    G.to_parquet(GRADED)
    w, lo = record(G.lean, G.settle)
    t = G[G.totals_lean.notna()]
    tw, tl = record(t.totals_lean, t.settle)
    dif = t[t.lean.ne(t.totals_lean)]
    dw, dl = record(dif.lean, dif.settle)
    G["totals_underdog"] = np.where(
        G.totals_margin < 0, "HOME", np.where(G.totals_margin > 0, "AWAY", None)
    )
    u = G[G.totals_underdog.notna()]
    uw, ul = record(u.totals_underdog, u.settle)
    fav = u[u.lean.ne(u.totals_underdog)]
    fw, fl = record(fav.lean, fav.settle)
    weeks = int(G.week.nunique())
    print(f"graded at the pool line {len(G)} games over {weeks} forward weeks")
    print(f"prop diagnostic sign {w}-{lo}; team totals against the pool line {tw}-{tl}")
    print(
        f"sides differ from the team totals at the pool line {len(dif)}: prop diagnostic {dw}-{dl}"
    )
    print(
        f"scale control, team-totals underdog {uw}-{ul}; "
        f"diagnostic on the team-totals favourite {len(fav)}: {fw}-{fl}"
    )
    print(
        f"diagnostic mean {G.diagnostic.mean():.2f}, "
        f"correlation with the team-totals margin {G.diagnostic.corr(G.totals_margin):.2f}"
    )
    by = G.groupby("week").apply(
        lambda d: pd.Series(
            {
                "n": len(d),
                "wins": record(d.lean, d.settle)[0],
                "losses": record(d.lean, d.settle)[1],
            }
        ),
        include_groups=False,
    )
    print(by.to_string())
    if weeks < FORWARD_MIN_WEEKS:
        print(f"no coefficient fitted: {weeks} of {FORWARD_MIN_WEEKS} forward weeks")
    else:
        print(f"{weeks} forward weeks: a fitted coefficient is now admissible")


def cmd_slots(a):
    base = [
        str(ROOT / ".tools" / "uv.exe"),
        "run",
        "--no-sync",
        "python",
        str(ROOT / "scripts" / "prop_market.py"),
    ]
    print(
        json.dumps(
            {
                "pregame": {
                    "argv": [*base, "pregame"],
                    "slots": [{"day": d, "at": t} for d, t in SLOTS],
                },
                "score": {"argv": [*base, "score"], "slots": [SCORE_SLOT]},
            },
            indent=1,
        )
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Player-prop implied production against the team totals: "
            "capture, pregame diagnostic and forward scoring at the pool line."
        )
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, text in (
        ("mapping", cmd_mapping, "Fit points per yard on nflverse team games"),
        ("capture", cmd_capture, "Save Bovada player props and team totals and report coverage"),
        ("pregame", cmd_pregame, "Capture, then log the per-game prop diagnostic"),
        ("score", cmd_score, "Grade the logged diagnostic sign at the pool line"),
        ("slots", cmd_slots, "Print the scheduler argv and slots"),
    ):
        sub.add_parser(name, help=text).set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
