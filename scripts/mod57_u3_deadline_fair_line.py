from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lead98_price_skew import BASE, fit, loso, metrics, predict, reliability, skew_table, summary
from per_flag_weights_eval import DRAWS, SEASONS, SEED

from nfl_ats.clv import LIVE_CAPTURE_KIND, load_decision_quotes
from nfl_ats.odds import no_vig_probabilities

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/pick_probability/20261111T011859Z/per_game.parquet"
RAW = ROOT / "data/market/raw"
OUT = ROOT / "tests/scratch/mod57/u3"
JUICE = "deadline_juice"
DIVERGENCE = "deadline_ml_divergence"
SMALL_MOVE = 0.5
HISTORICAL = "historical_backfill"
KEEP = [
    "nflverse_game_id",
    "bookmaker_key",
    "market",
    "outcome_side",
    "price",
    "home_spread_line",
    "observed_at_utc",
    "bookmaker_last_update_utc",
    "commence_time_utc",
]


def logit(p):
    return np.log(p / (1.0 - p))


def last_pair(q, market):
    m = q.loc[q["market"].eq(market)]
    m = m.loc[
        m["bookmaker_last_update_utc"].isna()
        | m["bookmaker_last_update_utc"].le(m["observed_at_utc"])
    ]
    m = m.loc[m["observed_at_utc"].lt(m["commence_time_utc"])]
    keys = ["nflverse_game_id", "bookmaker_key"]
    home = (
        m.loc[m["outcome_side"].eq("HOME")]
        .sort_values("observed_at_utc")
        .drop_duplicates([*keys, "observed_at_utc"], keep="last")
    )
    away = (
        m.loc[m["outcome_side"].eq("AWAY")]
        .sort_values("observed_at_utc")
        .drop_duplicates([*keys, "observed_at_utc"], keep="last")
    )
    both = home[[*keys, "observed_at_utc", "price", "home_spread_line"]].merge(
        away[[*keys, "observed_at_utc", "price"]],
        on=[*keys, "observed_at_utc"],
        suffixes=("_home", "_away"),
    )
    both = both.sort_values("observed_at_utc").drop_duplicates(keys, keep="last")
    both["p_home"] = [
        no_vig_probabilities(h, a)[0]
        for h, a in zip(both["price_home"], both["price_away"], strict=True)
    ]
    return both


def deadline_books():
    spreads, h2h = [], []
    for season in SEASONS:
        q = load_decision_quotes(RAW, capture_kind=HISTORICAL, seasons=[season])
        q = q.loc[q["nflverse_game_id"].notna() & q["market"].isin(["spreads", "h2h"]), KEEP]
        for column in ("observed_at_utc", "bookmaker_last_update_utc", "commence_time_utc"):
            q[column] = pd.to_datetime(q[column], utc=True, errors="coerce")
        s = last_pair(q, "spreads")
        h = last_pair(q, "h2h")
        s["season"] = season
        h["season"] = season
        spreads.append(s)
        h2h.append(h)
        del q
    return pd.concat(spreads, ignore_index=True), pd.concat(h2h, ignore_index=True)


def spread_win_map(pairs, season):
    prior = pairs.loc[pairs["season"].lt(season)]
    return prior.groupby("home_spread_line")["p_ml"].median()


def build_terms(data, open_table, spreads, h2h):
    sp = spreads.merge(
        open_table[["game_id", "p_home"]].rename(columns={"p_home": "p_open"}),
        left_on="nflverse_game_id",
        right_on="game_id",
        how="inner",
    )
    dj = sp.groupby("nflverse_game_id").agg(
        p_deadline=("p_home", "median"),
        p_open=("p_open", "first"),
        deadline_books=("p_home", "size"),
    )
    dj[JUICE] = logit(dj["p_deadline"]) - logit(dj["p_open"])
    pairs = spreads[["nflverse_game_id", "bookmaker_key", "season", "home_spread_line"]].merge(
        h2h[["nflverse_game_id", "bookmaker_key", "p_home"]].rename(columns={"p_home": "p_ml"}),
        on=["nflverse_game_id", "bookmaker_key"],
        how="inner",
    )
    parts = []
    for season in SEASONS:
        table = spread_win_map(pairs, season)
        cur = pairs.loc[pairs["season"].eq(season)].copy()
        cur["p_fair"] = cur["home_spread_line"].map(table)
        cur = cur.dropna(subset=["p_fair"])
        cur["gap"] = logit(cur["p_ml"]) - logit(cur["p_fair"])
        parts.append(cur)
    div = (
        pd.concat(parts, ignore_index=True)
        .groupby("nflverse_game_id")
        .agg(**{DIVERGENCE: ("gap", "median"), "ml_books": ("gap", "size")})
    )
    out = data.merge(
        dj[[JUICE, "deadline_books"]], left_on="game_id", right_index=True, how="left"
    ).merge(div, left_on="game_id", right_index=True, how="left")
    return out


def block_bootstrap(frame, p_new, p_old, rng):
    y = frame["home_covered"].to_numpy(float)
    block = frame.groupby(["season", "week"]).ngroup().to_numpy()
    n_blocks = block.max() + 1
    pn = np.clip(p_new, 1e-9, 1 - 1e-9)
    po = np.clip(p_old, 1e-9, 1 - 1e-9)
    acc = ((pn >= 0.5) == (y == 1)).astype(float) - ((po >= 0.5) == (y == 1)).astype(float)
    brier = (pn - y) ** 2 - (po - y) ** 2
    ll = -(y * np.log(pn) + (1 - y) * np.log1p(-pn)) + (y * np.log(po) + (1 - y) * np.log1p(-po))
    n_games = np.bincount(block, minlength=n_blocks).astype(float)
    draws = rng.integers(0, n_blocks, size=(DRAWS, n_blocks))
    games = n_games[draws].sum(axis=1)
    out = {}
    for name, v, scale in (
        ("accuracy_points", acc, 100.0),
        ("brier", brier, 1.0),
        ("log_loss", ll, 1.0),
    ):
        d = scale * np.bincount(block, weights=v, minlength=n_blocks)[draws].sum(axis=1) / games
        better = (d > 0) if name == "accuracy_points" else (d < 0)
        out[name] = {
            "delta_mean": float(d.mean()),
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "probability_positive": float(better.mean()),
        }
    return out


def decisive(p_new, p_old, y):
    differ = (p_new >= 0.5) != (p_old >= 0.5)
    return {
        "changed_pick_games": int(differ.sum()),
        "new_wins": int(((p_new >= 0.5) == (y == 1))[differ].sum()),
        "base_wins": int(((p_old >= 0.5) == (y == 1))[differ].sum()),
    }


def live_feasibility():
    q = load_decision_quotes(RAW, capture_kind=LIVE_CAPTURE_KIND, seasons=[2026])
    q = q.loc[q["nflverse_game_id"].notna() & q["market"].isin(["spreads", "h2h"])].copy()
    for column in ("observed_at_utc", "commence_time_utc", "snapshot_timestamp_utc"):
        q[column] = pd.to_datetime(q[column], utc=True, errors="coerce")
    anchor = (
        q.groupby(["season", "week"])["commence_time_utc"]
        .transform("min")
        .dt.tz_convert("America/New_York")
    )
    local = anchor.dt.tz_localize(None).dt.normalize()
    sunday = local + pd.to_timedelta((6 - anchor.dt.weekday) % 7, unit="D")
    deadline = (
        (sunday + pd.Timedelta(hours=16)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    q["deadline"] = np.minimum(q["commence_time_utc"], deadline)
    q = q.loc[q["observed_at_utc"].lt(q["deadline"])]
    last = q.groupby("nflverse_game_id")["snapshot_timestamp_utc"].transform("max")
    q = q.loc[q["snapshot_timestamp_utc"].eq(last)]
    rows = {}
    for game, g in q.groupby("nflverse_game_id"):
        ok = {}
        for market in ("spreads", "h2h"):
            m = g.loc[g["market"].eq(market)]
            m = m.loc[m["outcome_side"].isin(["HOME", "AWAY"])]
            ok[market] = int((m.groupby("bookmaker_key")["outcome_side"].nunique() >= 2).sum())
        rows[game] = ok
    t = pd.DataFrame(rows).T
    both = t.min(axis=1)
    weeks = q.drop_duplicates("nflverse_game_id").set_index("nflverse_game_id")["week"]
    return {
        "games_with_a_pre_deadline_capture": len(t),
        "games_two_books_spread_and_moneyline": int((both >= 2).sum()),
        "games_one_book": int((both == 1).sum()),
        "games_none": int((both == 0).sum()),
        "by_week_two_books": {
            str(int(k)): int(v)
            for k, v in (both >= 2).groupby(weeks.reindex(t.index)).sum().items()
        },
        "by_week_games": {
            str(int(k)): int(v) for k, v in t.groupby(weeks.reindex(t.index)).size().items()
        },
    }


def main():
    data = pd.read_parquet(ARTIFACT)
    data = data.loc[data["season"].between(2020, 2025)].reset_index(drop=True)
    open_table = skew_table(data)
    spreads, h2h = deadline_books()
    data = build_terms(data, open_table, spreads, h2h)
    y = data["home_covered"].to_numpy(float)
    move = (data["close_home_spread"] - data["tue_open_home_spread"]).abs()
    populations = {
        "all_games": np.ones(len(data), dtype=bool),
        "primary_move_half_point_or_less": (move <= SMALL_MOVE).to_numpy(),
    }
    coverage = {
        "games": len(data),
        "primary_games": int(populations["primary_move_half_point_or_less"].sum()),
        "juice_available": int(data[JUICE].notna().sum()),
        "divergence_available": int(data[DIVERGENCE].notna().sum()),
        "by_season": {
            str(s): {
                "games": len(g),
                "juice": int(g[JUICE].notna().sum()),
                "divergence": int(g[DIVERGENCE].notna().sum()),
            }
            for s, g in data.groupby("season")
        },
        "juice_std": float(data[JUICE].std()),
        "divergence_std": float(data[DIVERGENCE].std()),
    }
    variants = {
        "four_term": BASE,
        "plus_deadline_juice": (*BASE, JUICE),
        "plus_deadline_ml_divergence": (*BASE, DIVERGENCE),
        "model_only": ("model_logit",),
        "intercept_only": (),
    }
    probs, folds, insample = {}, {}, {}
    for name, feats in variants.items():
        oos, fo = loso(data, feats)
        probs[name] = oos
        folds[name] = fo
        b, m, s = fit(data, feats)
        insample[name] = predict(data, feats, b, m, s)
    probs["market_half"] = np.full(len(data), 0.5)
    rng = np.random.default_rng(SEED)
    result = {"coverage": coverage, "populations": {}, "fold_coefficients": {}, "fold_summary": {}}
    for name in ("plus_deadline_juice", "plus_deadline_ml_divergence"):
        result["fold_coefficients"][name] = folds[name]
        result["fold_summary"][name] = summary(folds[name])
    served = pd.read_parquet(ARTIFACT)
    served = (
        served.set_index("game_id")["out_of_season_home_probability"]
        .reindex(data["game_id"])
        .to_numpy()
    )
    for pop, mask in populations.items():
        sub = data.loc[mask].reset_index(drop=True)
        ys = y[mask]
        entry = {
            "games": int(mask.sum()),
            "variants": {},
            "paired_vs_four_term": {},
            "decisive": {},
        }
        for name in list(probs):
            entry["variants"][name] = {
                "oos": metrics(probs[name][mask], ys),
                "in_sample": metrics(insample[name][mask], ys) if name in insample else None,
            }
            if name in insample:
                entry["variants"][name]["accuracy_gap_in_minus_oos_points"] = 100.0 * (
                    entry["variants"][name]["in_sample"]["accuracy"]
                    - entry["variants"][name]["oos"]["accuracy"]
                )
        entry["variants"]["served_artifact_probability"] = {"oos": metrics(served[mask], ys)}
        for name in ("plus_deadline_juice", "plus_deadline_ml_divergence"):
            entry["paired_vs_four_term"][name] = block_bootstrap(
                sub, probs[name][mask], probs["four_term"][mask], rng
            )
            entry["decisive"][name] = decisive(probs[name][mask], probs["four_term"][mask], ys)
        entry["reliability"] = {
            name: reliability(probs[name][mask], ys)
            for name in ("four_term", "plus_deadline_juice", "plus_deadline_ml_divergence")
        }
        result["populations"][pop] = entry
    result["live_feasibility_2026"] = live_feasibility()
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=1, default=str)
    (OUT / "results.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
