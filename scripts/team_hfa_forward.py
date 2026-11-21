from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))

import expert_rankings_forward as XF
import pooled_stack as PS
import team_hfa as T
from mod27_fenced_repairs_replay import FOUR, Fit

ROOT = T.ROOT
OUT = ROOT / "artifacts/forward_terms/team_hfa"
FROZEN = OUT / "frozen.json"
FIT = OUT / "fit.json"
SCREEN = T.OUT / "screen.json"
SCREEN_TS = T.OUT / "team_seasons.parquet"
SCREEN_GAMES = T.OUT / "screen_games.parquet"
PUBLISHED = ROOT / "artifacts/clv_ledger/published_picks.parquet"
SEASON = 2026
FIRST_WEEK = 12
LAST_WEEK = 18
HALF_SPLIT = 2023
ET = "America/New_York"
STAMP = "%Y%m%dT%H%M%SZ"
FAMILY = "team_hfa_served_2026_11_21"
LANE = "docs/lanes/team-hfa-forward-2026-11-21.md"
OUTCOMES = ("result", "home_score", "away_score", "total", "overtime")
INPUT_COLUMNS = (
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "stadium_id",
    "location",
    "result",
    "spread_line",
)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now_stamp():
    return datetime.now(UTC).strftime(STAMP)


def screen_table():
    snap = json.loads(SCREEN.read_text(encoding="utf-8"))["schedule_snapshot"]
    path = T.SCHEDULES / snap / "schedules.parquet"
    raw = pd.read_parquet(path)
    table = raw.loc[raw["game_type"].eq("REG") & raw["season"].le(SEASON)].copy()
    for side in ("home_team", "away_team"):
        table[side] = table[side].replace(T.FRANCHISE)
    table["game_id"] = table["game_id"].astype(str)
    for col in OUTCOMES:
        if col in table.columns:
            table[col] = table[col].astype(float).where(table["season"].lt(SEASON))
    return table, snap, path


def canonical_sha(table):
    frame = table[list(INPUT_COLUMNS)].sort_values("game_id").reset_index(drop=True)
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def game_terms(hist, games):
    xmap = hist.set_index(["season", "team"])["x"]
    keys = zip(games["season"], games["home_team"], games["primary"], strict=True)
    vals = [float(xmap.get((s, t), 0.0)) if p else 0.0 for s, t, p in keys]
    return pd.Series(vals, index=games["game_id"].astype(str).to_numpy())


def frozen_values(table):
    ts, games = T.team_seasons(table)
    hist, picks = T.term_values(ts, games, T.SCREEN_SEASONS)
    lo, hi = T.SCREEN_SEASONS
    home = games.loc[games["primary"] & games["r"].notna() & games["season"].between(lo, hi)]
    usable = home.merge(
        ts,
        left_on=["season", "home_team", "venue"],
        right_on=["season", "team", "venue"],
        how="left",
    ).dropna(subset=["d"])
    k, lam, sse, sse0 = T.choose_k(usable)
    cur = ts.loc[ts["season"].eq(SEASON)].copy()
    cur["w"] = cur["n_eff"] / (cur["n_eff"] + lam)
    cur["x"] = (cur["w"] * cur["d"]).fillna(0.0)
    shrink = {
        "k": k,
        "lambda_games": lam,
        "unit_mean_n_eff": float(usable["n_eff"].mean()),
        "squared_error": sse,
        "squared_error_no_term": sse0,
        "home_games": len(usable),
        "seasons": [lo, hi],
    }
    return hist, games, cur, shrink, picks


def identity(hist, games):
    stored = pd.read_parquet(SCREEN_TS).set_index(["season", "team"])["x"]
    mine = hist.set_index(["season", "team"])["x"]
    common = mine.index.intersection(stored.index)
    scr = pd.read_parquet(SCREEN_GAMES)
    gx = game_terms(hist, games).reindex(scr["game_id"].astype(str)).to_numpy(float)
    return {
        "team_seasons_compared": len(common),
        "team_season_x_max_abs": float((mine.reindex(common) - stored.reindex(common)).abs().max()),
        "screen_games_compared": len(scr),
        "screen_game_x_max_abs": float(np.nanmax(np.abs(gx - scr["x"].to_numpy(float)))),
        "screen_games_missing": int(np.isnan(gx).sum()),
    }


def served_fit(gx):
    pop = PS.population()
    pop["game_id"] = pop["game_id"].astype(str)
    missing = int((~pop["game_id"].isin(gx.index)).sum())
    if missing:
        raise SystemExit(f"{missing} population games have no schedule row")
    raw = gx.reindex(pop["game_id"]).to_numpy(float)
    season = pop["season"].to_numpy(int)
    decided = ~pop["push"].to_numpy(bool)
    ylab = pop["y"].to_numpy(float)
    seasons = sorted(set(season.tolist()))
    cache = {}

    def off_for(excl, rows):
        key = frozenset(excl)
        if key not in cache:
            cache[key] = Fit(FOUR, pop[decided & ~np.isin(season, list(key))])
        return PS.logit(cache[key].predict(pop[rows]))

    off1 = np.full(len(pop), np.nan)
    for s in seasons:
        off1[season == s] = off_for({s}, season == s)
    qw = expit(off1)
    p = np.full(len(pop), np.nan)
    p_int = np.full(len(pop), np.nan)
    folds = {}
    for s in seasons:
        tr = season != s
        off_tr = np.full(len(pop), np.nan)
        for t in seasons:
            if t != s:
                off_tr[season == t] = off_for({s, t}, season == t)
        xs, mu, sd = XF.standardise(raw, tr)
        b = XF.offset_fit(xs[tr], ylab[tr], off_tr[tr])
        bi = XF.offset_fit(np.ones(int(tr.sum())), ylab[tr], off_tr[tr])
        held = season == s
        p[held] = expit(off1[held] + b * xs[held])
        p_int[held] = expit(off1[held] + bi)
        folds[str(s)] = {"coefficient_per_sd": b, "mean": mu, "sd": sd, "intercept_control": bi}
    xs, mu, sd = XF.standardise(raw, np.ones(len(pop), bool))
    b_full = XF.offset_fit(xs, ylab, off1)
    p_in = expit(off1 + b_full * xs)
    halves = {}
    for name, m in (("2020-22", season < HALF_SPLIT), ("2023-25", season >= HALF_SPLIT)):
        halves[name] = XF.offset_fit(xs[m], ylab[m], off1[m])
    rng = np.random.default_rng(T.SEED)
    wk = season * 100 + pop["week"].to_numpy(int)
    res = {
        "games": len(pop),
        "pushes": int((~decided).sum()),
        "nonzero_term_games": int((raw != 0).sum()),
        "qw_record": PS.record(qw[decided], ylab[decided]),
        "coefficient_per_sd_full": b_full,
        "term_mean": mu,
        "term_sd": sd,
        "fold": folds,
        "folds_positive": int(sum(f["coefficient_per_sd"] > 0 for f in folds.values())),
        "halves_coefficient_per_sd": halves,
        "week_block": XF.evaluate(p, qw, ylab, wk, rng),
        "season_block": XF.evaluate(p, qw, ylab, season, rng),
        "in_sample_log_loss_gain": float((PS.ll(qw, ylab) - PS.ll(p_in, ylab)).mean()),
        "intercept_only_log_loss_gain": float((PS.ll(qw, ylab) - PS.ll(p_int, ylab)).mean()),
        "corr_term_qw_logit": float(np.corrcoef(raw[raw != 0], off1[raw != 0])[0, 1]),
    }
    res["in_sample_minus_held_out_log_loss_gain"] = (
        res["in_sample_log_loss_gain"] - res["week_block"]["log_loss_gain"]["effect"]
    )
    frame = pd.DataFrame(
        {"game_id": pop["game_id"], "season": season, "y": ylab, "qw": qw, "term": p, "x": raw}
    )
    return res, frame


def cmd_freeze(a):
    if FROZEN.exists():
        raise SystemExit(f"already frozen: {FROZEN}")
    OUT.mkdir(parents=True, exist_ok=True)
    table, snap, path = screen_table()
    hist, games, cur, shrink, picks = frozen_values(table)
    ident = identity(hist, games)
    res, frame = served_fit(game_terms(hist, games))
    res["identity_vs_screen"] = ident
    res["shrinkage_2026"] = shrink
    res["inner_loso_shrinkage_history"] = {str(s): v["k"] for s, v in picks.items()}
    frame.to_parquet(OUT / "history_games.parquet", index=False)
    FIT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    teams = {
        r.team: {
            "venue": r.venue,
            "x_points": float(r.x),
            "d_points": None if pd.isna(r.d) else float(r.d),
            "n_home": int(r.n_home),
            "n_road": int(r.n_road),
            "n_eff": None if pd.isna(r.n_eff) else float(r.n_eff),
            "w": None if pd.isna(r.w) else float(r.w),
        }
        for r in cur.sort_values("team").itertuples()
    }
    body = {
        "term": "team_hfa_shrunk_venue_edge",
        "label": "post_hoc",
        "declared_before": "Week 12 2026 pool openers (pool lock Tue 2026-11-24 12:00 ET)",
        "sign": "positive on home cover",
        "season": SEASON,
        "first_week": FIRST_WEEK,
        "last_week": LAST_WEEK,
        "construction": (
            "scripts/team_hfa.py team_seasons: primary venue = stadium hosting most of the "
            "team's 2026 home games (schedule, no outcomes); r = result - spread_line minus the "
            "season's league mean over primary-venue REG games; d = mean r at home at that venue "
            "minus mean -r on the road, seasons < 2026 with the same primary venue; n_eff = "
            "1/(1/n_home + 1/n_road); w = n_eff/(n_eff + lambda), lambda = mean(n_eff) k/(1-k), "
            "k minimising squared error of r on w d over home games of all screen seasons "
            "2011-2025 (term_values inner rule with 2026 as the scored season); x = w d for the "
            "home team at its primary venue, 0 at any other venue, neutral site or no history."
        ),
        "shrinkage": shrink,
        "teams": teams,
        "coefficient_logit_per_sd": res["coefficient_per_sd_full"],
        "term_mean_points": res["term_mean"],
        "term_sd_points": res["term_sd"],
        "fold_coefficients_per_sd": {s: f["coefficient_per_sd"] for s, f in res["fold"].items()},
        "halves_coefficient_per_sd": res["halves_coefficient_per_sd"],
        "fit": (
            "offset logistic, no intercept, QW served stage offset (LOSO, nested), opener "
            "2020-25, pushes 0.5; term per SD of the 2020-25 population"
        ),
        "fit_games": res["games"],
        "fit_sha256": sha_file(FIT),
        "inputs": {
            "schedule_snapshot": snap,
            "schedule_file_sha256": sha_file(path),
            "canonical_inputs_sha256": canonical_sha(table),
            "canonical_inputs_columns": list(INPUT_COLUMNS),
            "screen_json_sha256": sha_file(SCREEN),
            "qw_sha256": sha_file(PS.QW),
        },
        "identity_vs_screen": ident,
        "no_2026_outcomes": True,
        "earliest_scoring": "after Week 18 2026 results settle",
        "lane": LANE,
        "created_at_utc": now_stamp(),
    }
    FROZEN.write_text(json.dumps(body, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("fold",)}, indent=1))
    print(json.dumps({t: round(v["x_points"], 3) for t, v in teams.items()}))
    print(FROZEN, sha_file(FROZEN))


def latest_schedule(season):
    snaps = sorted(p for p in T.SCHEDULES.glob("2026*") if (p / "schedules.parquet").is_file())
    raw = pd.read_parquet(snaps[-1] / "schedules.parquet")
    g = raw.loc[raw["season"].eq(season) & raw["game_type"].eq("REG")].copy()
    for side in ("home_team", "away_team"):
        g[side] = g[side].replace(T.FRANCHISE)
    g["game_id"] = g["game_id"].astype(str)
    local = pd.to_datetime(g["gameday"] + " " + g["gametime"], errors="coerce")
    g["kickoff_utc"] = local.dt.tz_localize(ET).dt.tz_convert("UTC")
    return g, snaps[-1].name


def card(season, week, now):
    pub = pd.read_parquet(PUBLISHED)
    pub = pub.loc[pub["season"].eq(season) & pub["week"].eq(week)]
    pub = pub.loc[pub["published_at_utc"].le(pub["pick_deadline_utc"])]
    pub = pub.loc[pd.to_datetime(pub["published_at_utc"], utc=True).le(pd.Timestamp(now))]
    pub = pub.sort_values("published_at_utc").drop_duplicates("game_id", keep="last").copy()
    home = pub["pick_team"].eq(pub["home_team"])
    pub["served_home_p"] = np.where(home, pub["displayed_score"], 1 - pub["displayed_score"])
    pub["game_id"] = pub["game_id"].astype(str)
    return pub.set_index("game_id")


def cmd_log(a):
    now = datetime.now(UTC)
    fz = json.loads(FROZEN.read_text(encoding="utf-8"))
    sched, snap = latest_schedule(a.season)
    if a.week == "auto":
        ahead = sched.loc[sched["kickoff_utc"] > pd.Timestamp(now), "week"]
        if ahead.empty:
            raise SystemExit("no unplayed regular-season week")
        week = int(ahead.min())
    else:
        week = int(a.week)
    if a.season != fz["season"]:
        raise SystemExit(f"frozen values are for season {fz['season']}")
    if not a.smoke and not fz["first_week"] <= week <= fz["last_week"]:
        raise SystemExit(f"week {week} outside the declared window")
    g = sched.loc[sched["week"].eq(week)].sort_values(["kickoff_utc", "game_id"])
    if g.empty:
        raise SystemExit(f"no schedule rows for {a.season} week {week}")
    served = card(a.season, week, now)
    rows = []
    for r in g.itertuples():
        team = fz["teams"].get(r.home_team)
        neutral = str(r.location).lower() == "neutral"
        at_primary = bool(team is not None and not neutral and r.stadium_id == team["venue"])
        x = float(team["x_points"]) if at_primary else 0.0
        z = (x - fz["term_mean_points"]) / fz["term_sd_points"]
        shift = fz["coefficient_logit_per_sd"] * z
        row = {
            "game_id": r.game_id,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "kickoff_utc": None if pd.isna(r.kickoff_utc) else r.kickoff_utc.isoformat(),
            "stadium_id": r.stadium_id,
            "location": r.location,
            "frozen_venue": None if team is None else team["venue"],
            "at_primary_venue": at_primary,
            "term_points": x,
            "term_sd_units": z,
            "logit_shift": shift,
            "schedule_spread_line": None if pd.isna(r.spread_line) else float(r.spread_line),
            "served": None,
        }
        if r.game_id in served.index:
            c = served.loc[r.game_id]
            base = float(c["served_home_p"])
            p = float(expit(PS.logit(base) + shift))
            row["served"] = {
                "pick_team": c["pick_team"],
                "pick_probability": float(c["displayed_score"]),
                "served_home_probability": base,
                "card_line": float(c["market_spread"]),
                "published_at_utc": str(c["published_at_utc"]),
                "pick_deadline_utc": str(c["pick_deadline_utc"]),
            }
            row["with_term_home_probability"] = p
            row["with_term_side"] = r.home_team if p >= 0.5 else r.away_team
            row["with_term_differs_from_served_side"] = bool(
                row["with_term_side"] != c["pick_team"]
            )
        rows.append(row)
    stamp = now.strftime(STAMP)
    body = {
        "smoke_test": bool(a.smoke),
        "label": "post_hoc",
        "season": a.season,
        "week": week,
        "logged_at_utc": stamp,
        "schedule_snapshot": snap,
        "card_source": str(PUBLISHED.relative_to(ROOT)),
        "card_rule": "last publication at or before the log time and the game's pick deadline",
        "served_card_games": int(sum(r["served"] is not None for r in rows)),
        "frozen_sha256": sha_file(FROZEN),
        "serves_nothing": True,
        "games": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.season}_week{week:02d}_{stamp}.json"
    path.write_text(json.dumps(body, indent=1), encoding="utf-8")
    print(path)
    for r in rows:
        s = r["served"]
        side = "-" if s is None else f"{s['pick_team']} {s['served_home_probability']:.3f}"
        print(
            f"{r['game_id']:18s} x={r['term_points']:+.3f} primary={r['at_primary_venue']} {side}"
        )


def cell(name, look, metric, units, text):
    b = look[metric]
    entry = {
        "name": name,
        "description": text,
        "source": LANE,
        "effect": round(b["effect"], 6),
        "effect_units": units,
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": PS.FIRST,
        "season_end": PS.LAST,
        "family": FAMILY,
        "category": "environment",
        "classification_evidence": (
            "Declared sign; week-block interval spans both signs at this sample; no refuted "
            "mechanism and no positive control bounds an effect this size on the served stage."
        ),
        "notes": (
            f"Family {FAMILY}: post-hoc served read made only to freeze the out-of-season "
            "coefficient for the 2026 Weeks 12-18 forward log; follows the close screen family "
            f"{T.FAMILY}, whose declared served look was not run (screen not a survivor). "
            "Not gating, not served."
        ),
        "interval_low": round(b["interval_low"], 6),
        "interval_high": round(b["interval_high"], 6),
        "probability_positive": round(b["probability_positive"], 4),
        "sample_games": int(look["decided"] if units == "accuracy_points" else look["games"]),
        "sample_blocks": int(b["blocks"]),
        "plain_summary": (
            "Adds each stadium's own past home edge to the model's pick probability for 2020-25 "
            "and checks the picks it would have changed."
        ),
    }
    if entry["interval_high"] < 0:
        entry["classification"] = "refuted_mechanism"
        entry["closing_ground"] = "wrong_sign_resolved"
        entry["classification_evidence"] = (
            "Declared positive; the whole week-block interval lies below zero."
        )
    return entry


def cmd_batch(a):
    fit = json.loads(FIT.read_text(encoding="utf-8"))
    w = fit["week_block"]
    h = fit["halves_coefficient_per_sd"]
    text = (
        "Team-and-stadium home edge (scripts/team_hfa.py construction), one offset-logistic "
        "coefficient per SD on the QW served stage at the opener 2020-25, pushes kept at 0.5, "
        "leave-one-season-out with nested offsets. Full coefficient "
        f"{fit['coefficient_per_sd_full']:+.4f}/SD, folds positive {fit['folds_positive']}/6, "
        f"halves {h['2020-22']:+.4f}/{h['2023-25']:+.4f}; record {w['record_term']} v QW "
        f"{w['record_base']}; changed {w['changed']} ({w['changed_record_term']} v "
        f"{w['changed_record_base']}); in-sample minus held-out log loss "
        f"{fit['in_sample_minus_held_out_log_loss_gain']:+.5f}; intercept-only "
        f"{fit['intercept_only_log_loss_gain']:+.5f}."
    )
    cells = [
        cell(
            "team_hfa_served_opener_posthoc_accuracy", w, "accuracy_points", "accuracy_points", text
        ),
        cell(
            "team_hfa_served_opener_posthoc_log_loss",
            w,
            "log_loss_gain",
            "log_loss_improvement",
            text,
        ),
    ]
    path = OUT / "weak_signals_batch.json"
    path.write_text(json.dumps({"cells": cells}, indent=1), encoding="utf-8")
    print(path, len(cells))


def main():
    p = argparse.ArgumentParser(
        description=(
            "Team home edge forward log: freeze the 2026 per-team values and the out-of-season "
            "served-stage coefficient, then log the frozen term beside the served card each "
            "week. Research only; never changes a served side."
        )
    )
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("freeze").set_defaults(fn=cmd_freeze)
    lg = sp.add_parser("log")
    lg.add_argument("--season", type=int, default=SEASON)
    lg.add_argument("--week", default="auto")
    lg.add_argument("--smoke", action="store_true")
    lg.set_defaults(fn=cmd_log)
    sp.add_parser("batch").set_defaults(fn=cmd_batch)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
