from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, _fit_logit

POPULATION = Path("artifacts/pick_probability/20261020T184404Z/per_game.parquet")
POSTS = Path("data/processed/beat_reporter_posts.parquet")
RATES = Path("data/processed/player_availability_rates.parquet")
RATINGS = Path("data/processed/player_participation_ratings.parquet")
FEATURES = Path("data/processed/game_features.parquet")
OUT = Path("docs/mod24_u9_beat_grade.json")
OUT_FRAME = Path("data/processed/mod24_u9_beat_feature.parquet")
POSITIVE = {"expected_to_play", "first_team_reps", "full"}
SEVERITY = ["ruled_out", "absent", "did_not_practice", "boot_brace", "questionable", "limited"]
SCORED = (2024, 2025)


def unavailability(rates, season):
    r = rates[(rates.target_season == season) & (rates.position_group == "__all__")]

    def cell(rep, pr):
        row = r[(r.report_category == rep) & (r.practice_category == pr)]
        return float(row.unavailability_probability.iloc[0])

    q = r[(r.report_category == "questionable") & (r.practice_category != "__all__")]
    pooled = float((q.unavailability_probability * q.observations).sum() / q.observations.sum())
    return {
        "ruled_out": cell("out", "out"),
        "absent": cell("out", "out"),
        "did_not_practice": cell("none", "dnp"),
        "limited": cell("none", "limited"),
        "questionable": pooled,
        "boot_brace": pooled,
    }


def deadlines(games):
    ko = pd.to_datetime(games.kickoff, utc=True)
    day = pd.to_datetime(games.gameday)
    wd = day.dt.weekday
    shift = np.where(wd == 0, -1, (6 - wd) % 7)
    sunday_local = (day + pd.to_timedelta(shift, unit="D")).dt.tz_localize("America/New_York")
    sunday = (sunday_local + pd.Timedelta(hours=16)).dt.tz_convert("UTC")
    games = games.assign(kickoff_utc=ko)
    games["deadline"] = pd.concat([ko, sunday], axis=1).min(axis=1)
    wed = (sunday_local - pd.Timedelta(days=4)).dt.tz_convert("UTC") + pd.Timedelta(hours=18)
    games["wednesday"] = pd.concat([wed, games.deadline], axis=1).min(axis=1)
    return games


def build_feature(per_game):
    games = pd.read_parquet(
        FEATURES, columns=["game_id", "kickoff", "gameday", "home_team", "away_team", "season"]
    )
    games = deadlines(games.copy())
    sub = games[games.game_id.isin(per_game.game_id)]
    long = pd.concat(
        [sub.assign(team=sub.home_team, side="home"), sub.assign(team=sub.away_team, side="away")]
    )
    allg = pd.concat(
        [
            games.assign(team=games.home_team)[["team", "kickoff_utc", "game_id"]],
            games.assign(team=games.away_team)[["team", "kickoff_utc", "game_id"]],
        ]
    ).sort_values(["team", "kickoff_utc"])
    allg["prev_kick"] = allg.groupby("team").kickoff_utc.shift(1)
    long = long.merge(allg[["team", "game_id", "prev_kick"]], on=["team", "game_id"])
    long["prev_kick"] = long.prev_kick.fillna(long.kickoff_utc - pd.Timedelta(days=7))
    posts = pd.read_parquet(POSTS)
    posts = posts[
        ~posts.is_repost & posts.player_gsis.map(len).gt(0) & posts.avail_classes.map(len).gt(0)
    ]
    posts = posts.assign(created=pd.to_datetime(posts.created_at, utc=True))
    posts = posts.explode("player_gsis").rename(columns={"player_gsis": "player"})
    rates = pd.read_parquet(RATES)
    ratings = pd.read_parquet(RATINGS)
    ratings = ratings.assign(value=(ratings.offense_rating + ratings.defense_rating).clip(lower=0))
    value = ratings.set_index(["target_season", "player_id"]).value
    by_team = {t: g for t, g in posts.groupby("team")}
    rows = []
    for rec in long.itertuples():
        loss = {"all": 0.0, "lead": 0.0}
        n = {"all": 0, "lead": 0}
        g = by_team.get(rec.team)
        if g is not None:
            w = g[(g.created > rec.prev_kick) & (g.created < rec.deadline)]
            if not w.created.lt(rec.deadline).all():
                raise ValueError("post at or after deadline")
            pu = unavailability(rates, int(rec.season))
            for variant, cut in (("all", rec.deadline), ("lead", rec.wednesday)):
                ww = w[w.created < cut].sort_values("created")
                for post in ww.groupby("player").tail(1).itertuples():
                    cls = [c for c in SEVERITY if c in post.avail_classes]
                    if cls:
                        v = float(value.get((int(rec.season), post.player), 0.0))
                        loss[variant] += pu[cls[0]] * v
                        n[variant] += 1
                    elif any(c in POSITIVE for c in post.avail_classes):
                        n[variant] += 1
        rows.append(
            {
                "game_id": rec.game_id,
                "side": rec.side,
                "loss_all": loss["all"],
                "loss_lead": loss["lead"],
                "n_all": n["all"],
                "n_lead": n["lead"],
            }
        )
    t = pd.DataFrame(rows)
    h = t[t.side == "home"].set_index("game_id")
    a = t[t.side == "away"].set_index("game_id")
    out = pd.DataFrame(
        {
            "beat_all": a.loss_all - h.loss_all,
            "beat_lead": a.loss_lead - h.loss_lead,
            "covered_all": (h.n_all + a.n_all).gt(0),
            "covered_lead": (h.n_lead + a.n_lead).gt(0),
        }
    )
    out["nonzero_all"] = out.beat_all.ne(0)
    out["nonzero_lead"] = out.beat_lead.ne(0)
    return out.reset_index()


def fit(train, test, terms):
    x = train[terms].to_numpy(float)
    mu, sd = x.mean(0), x.std(0)
    sd[sd < 1e-12] = 1.0
    d = np.column_stack([np.ones(len(train)), (x - mu) / sd])
    beta = _fit_logit(d, train.home_covered.to_numpy(float), FIT_RIDGE)
    td = np.column_stack([np.ones(len(test)), (test[terms].to_numpy(float) - mu) / sd])
    p = 1 / (1 + np.exp(-np.clip(td @ beta, -35, 35)))
    return p, dict(zip(terms, (beta[1:] / sd).tolist(), strict=True))


def metrics(frame, p):
    y = frame.home_covered.to_numpy(float)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return {
        "acc": ((p >= 0.5) == y).astype(float),
        "brier": (p - y) ** 2,
        "ll": -y * np.log(p) - (1 - y) * np.log1p(-p),
    }


def boot(frame, delta, seed):
    g = pd.DataFrame({"s": frame.season.to_numpy(), "w": frame.week.to_numpy(), "d": delta})
    g = g.groupby(["s", "w"]).d.agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(g), size=(10000, len(g)))
    draws = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    pos = (draws > 1e-12).mean() + 0.5 * (np.abs(draws) <= 1e-12).mean()
    return {
        "effect": float(delta.mean()),
        "low": float(lo),
        "high": float(hi),
        "probability_positive": float(pos),
        "se": float(draws.std(ddof=1)),
        "blocks": len(g),
    }


def main():
    pg = pd.read_parquet(POPULATION).sort_values(["season", "week", "game_id"])
    pg = pg[pg.home_covered.notna()].reset_index(drop=True)
    feat = build_feature(pg)
    feat.to_parquet(OUT_FRAME, index=False)
    f = pg.merge(feat, on="game_id", how="left", validate="one_to_one")
    for c in ("beat_all", "beat_lead"):
        f[c] = f[c].fillna(0.0)
    arms = {
        "base": list(FIT_FEATURES),
        "beat": [*FIT_FEATURES, "beat_all"],
        "beat_lead": [*FIT_FEATURES, "beat_lead"],
    }
    for arm in arms:
        f[f"{arm}_oos"] = np.nan
        f[f"{arm}_is"], _ = fit(f, f, arms[arm])
    coefs = []
    for held in sorted(f.season.unique()):
        tr, te = f[f.season != held], f[f.season == held]
        for arm, terms in arms.items():
            p, nat = fit(tr, te, terms)
            f.loc[te.index, f"{arm}_oos"] = p
            coefs.append({"arm": arm, "fold": int(held), **nat})
    sc = f[f.season.isin(SCORED)].reset_index(drop=True)
    cov = {
        "games": len(sc),
        "covered_all": int(sc.covered_all.sum()),
        "covered_lead": int(sc.covered_lead.sum()),
        "nonzero_all": int(sc.nonzero_all.sum()),
        "nonzero_lead": int(sc.nonzero_lead.sum()),
    }
    for k in ("is", "oos"):
        sc[f"model_{k}"] = sc.model_probability
        sc[f"market_{k}"] = 0.5
    m = {}
    for a in (*arms, "model", "market"):
        for k in ("is", "oos"):
            m[a, k] = metrics(sc, sc[f"{a}_{k}"].to_numpy())
    res = {"coverage": cov, "coefficients": coefs, "levels": {}, "cells": {}}
    for a in (*arms, "model", "market"):
        res["levels"][a] = {
            k: {n: float(v.mean()) for n, v in m[a, k].items()} for k in ("is", "oos")
        }
    for arm in ("beat", "beat_lead"):
        d = m[arm, "oos"]["acc"] - m["base", "oos"]["acc"]
        cell = {"decisive_up": int((d > 0).sum()), "decisive_down": int((d < 0).sum())}
        cell["accuracy"] = boot(sc, d, 2401)
        cell["brier_improvement"] = boot(
            sc, m["base", "oos"]["brier"] - m[arm, "oos"]["brier"], 2402
        )
        cell["ll_improvement"] = boot(sc, m["base", "oos"]["ll"] - m[arm, "oos"]["ll"], 2403)
        ins = float((m[arm, "is"]["acc"] - m["base", "is"]["acc"]).mean())
        cell["is_accuracy_gain"] = ins
        cell["is_oos_gap"] = ins - cell["accuracy"]["effect"]
        res["cells"][arm] = cell
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


main()
