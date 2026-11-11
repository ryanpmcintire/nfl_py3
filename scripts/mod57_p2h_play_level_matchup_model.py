from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lead98_price_skew import BASE, loso, metrics, summary
from mod57_u3_deadline_fair_line import block_bootstrap, decisive
from per_flag_weights_eval import SEED

from nfl_ats.constants import DEFAULT_OFFSEASON_RETENTION, TEAM_ABBREVIATION_ALIASES
from nfl_ats.pbp import analysis_plays, latest_pbp_snapshot, load_pbp_snapshot

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/pick_probability/20261111T011859Z/per_game.parquet"
OUT = ROOT / "artifacts/mod57_p2h"
SPAN = 8
MIN_PERIODS = 3
ALPHA = 2.0 / (SPAN + 1.0)
RETENTION = DEFAULT_OFFSEASON_RETENTION
SMALL_MOVE = 0.5
SEASON_FIRST = 2009
TEST_SEASONS = range(2020, 2026)
EXPLOSIVE_PASS_YARDS = 20
EXPLOSIVE_RUSH_YARDS = 10
METRICS = [
    "pass_epa",
    "rush_epa",
    "pass_succ",
    "rush_succ",
    "pass_expl",
    "rush_expl",
    "sack_rate",
    "to_rate",
    "early_pass",
]
OFF_COLS = [f"off_{m}" for m in METRICS]
DEF_COLS = [f"def_{m}" for m in METRICS]
SITUATION = ["down", "ydstogo", "yardline_100", "half_seconds", "score_differential"]
FEATURES = [*OFF_COLS, *DEF_COLS, "off_is_home", *SITUATION]
N_TRUNCATION = 12
N_SHUFFLES = 50


def load_plays():
    snapshot = latest_pbp_snapshot(ROOT / "data/pbp/raw")
    plays = analysis_plays(load_pbp_snapshot(snapshot))
    plays = plays.loc[plays["season"].between(SEASON_FIRST, 2025)].copy()
    for column in ("posteam", "defteam", "home_team", "away_team"):
        plays[column] = plays[column].replace(TEAM_ABBREVIATION_ALIASES)
    for column in (
        "down",
        "ydstogo",
        "yardline_100",
        "yards_gained",
        "success",
        "sack",
        "interception",
        "fumble_lost",
        "game_seconds_remaining",
        "score_differential",
        "qtr",
    ):
        plays[column] = pd.to_numeric(plays[column], errors="coerce")
    plays["sack"] = plays["sack"].fillna(0)
    plays["is_pass"] = plays["pass_attempt"].eq(1)
    plays["is_rush"] = plays["rush_attempt"].eq(1) & ~plays["is_pass"]
    plays["explosive"] = (
        (plays["is_pass"] & plays["yards_gained"].ge(EXPLOSIVE_PASS_YARDS))
        | (plays["is_rush"] & plays["yards_gained"].ge(EXPLOSIVE_RUSH_YARDS))
    ).astype(float)
    plays["turnover"] = (
        plays["interception"].fillna(0).eq(1) | plays["fumble_lost"].fillna(0).eq(1)
    ).astype(float)
    plays["early_down"] = plays["down"].isin([1.0, 2.0])
    plays["half_seconds"] = np.where(
        plays["qtr"].le(2),
        plays["game_seconds_remaining"] - 1800.0,
        plays["game_seconds_remaining"],
    )
    plays["off_is_home"] = plays["posteam"].eq(plays["home_team"]).astype(float)
    return plays.reset_index(drop=True)


def unit_metrics(plays, key):
    base = plays.groupby(["game_id", key], sort=False)
    pas = plays.loc[plays["is_pass"]].groupby(["game_id", key], sort=False)
    rus = plays.loc[plays["is_rush"]].groupby(["game_id", key], sort=False)
    early = plays.loc[plays["early_down"]].groupby(["game_id", key], sort=False)
    out = pd.DataFrame(
        {
            "pass_epa": pas["epa"].mean(),
            "rush_epa": rus["epa"].mean(),
            "pass_succ": pas["success"].mean(),
            "rush_succ": rus["success"].mean(),
            "pass_expl": pas["explosive"].mean(),
            "rush_expl": rus["explosive"].mean(),
            "sack_rate": pas["sack"].mean(),
            "to_rate": base["turnover"].mean(),
            "early_pass": early["is_pass"].mean(),
        }
    )
    out.index = out.index.set_names(["game_id", "team"])
    return out


def team_games(plays, dates):
    off = unit_metrics(plays, "posteam").add_prefix("off_")
    dfn = unit_metrics(plays, "defteam").add_prefix("def_")
    tg = off.join(dfn, how="inner").reset_index()
    season = plays.drop_duplicates("game_id").set_index("game_id")["season"]
    tg["season"] = tg["game_id"].map(season)
    tg = tg.merge(dates, on="game_id", how="left", validate="many_to_one")
    if tg["gameday"].isna().any():
        raise ValueError("plays contain games absent from the schedule")
    return tg.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def build_states(tg):
    cols = [*OFF_COLS, *DEF_COLS]
    lm = {int(s): g[cols].mean().to_numpy() for s, g in tg.groupby("season", sort=False)}
    states = {}
    for team, g in tg.groupby("team", sort=False):
        values = g[cols].to_numpy(float)
        seasons = g["season"].to_numpy(int)
        current = np.full(len(cols), np.nan)
        obs = np.zeros(len(cols), dtype=int)
        out = np.full(values.shape, np.nan)
        prev = None
        for i in range(len(g)):
            if prev is not None and seasons[i] != prev:
                finite = np.isfinite(current)
                kept = lm[prev] + RETENTION ** max(1, seasons[i] - prev) * (current - lm[prev])
                current = np.where(finite, kept, current)
            v = values[i]
            ok = np.isfinite(v)
            first = ok & ~np.isfinite(current)
            blend = ok & np.isfinite(current)
            current = np.where(first, v, current)
            current = np.where(blend, ALPHA * v + (1 - ALPHA) * current, current)
            obs += ok.astype(int)
            out[i] = np.where(obs >= MIN_PERIODS, current, np.nan)
            prev = seasons[i]
        states[team] = (g["gameday"].to_numpy("datetime64[ns]"), seasons, out)
    return states, lm


def profile_at(states, lm, team, date, season):
    if team not in states:
        return np.full(len(OFF_COLS) + len(DEF_COLS), np.nan)
    dates, seasons, out = states[team]
    pos = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(date), "ns"), side="left")) - 1
    if pos < 0:
        return np.full(out.shape[1], np.nan)
    value = out[pos].copy()
    gap = int(season) - int(seasons[pos])
    if gap > 0:
        mean = lm[int(seasons[pos])]
        value = np.where(
            np.isfinite(value), mean + RETENTION ** max(1, gap) * (value - mean), value
        )
    return value


def pregame_profiles(tg):
    states, lm = build_states(tg)
    rows = np.vstack(
        [
            profile_at(states, lm, r.team, r.gameday, r.season)
            for r in tg[["team", "gameday", "season"]].itertuples(index=False)
        ]
    )
    pre = tg[["game_id", "team"]].copy()
    pre[[*OFF_COLS, *DEF_COLS]] = rows
    return pre


def truncation_check(tg, pre):
    rng = np.random.default_rng(SEED)
    pool = tg.index[tg["season"].between(2012, 2025)].to_numpy()
    picks = rng.choice(pool, size=N_TRUNCATION, replace=False)
    cols = [*OFF_COLS, *DEF_COLS]
    results = []
    for i in picks:
        row = tg.loc[i]
        trimmed = tg.loc[tg["gameday"].lt(row["gameday"])]
        states, lm = build_states(trimmed)
        redo = profile_at(states, lm, row["team"], row["gameday"], row["season"])
        full = pre.loc[i, cols].to_numpy(float)
        same = bool(np.array_equal(np.isnan(redo), np.isnan(full))) and bool(
            np.allclose(np.nan_to_num(redo), np.nan_to_num(full), rtol=0, atol=1e-12)
        )
        results.append(
            {
                "game_id": row["game_id"],
                "team": row["team"],
                "identical": same,
                "finite_values": int(np.isfinite(full).sum()),
            }
        )
    return results


def play_table(plays, pre):
    off = pre[["game_id", "team", *OFF_COLS]].rename(columns={"team": "posteam"})
    dfn = pre[["game_id", "team", *DEF_COLS]].rename(columns={"team": "defteam"})
    table = plays[["game_id", "season", "posteam", "defteam", "epa", *SITUATION, "off_is_home"]]
    table = table.merge(off, on=["game_id", "posteam"], how="left", validate="many_to_one")
    table = table.merge(dfn, on=["game_id", "defteam"], how="left", validate="many_to_one")
    return table


def roll_up(table, plays, game_ids):
    rows = []
    games = table.drop_duplicates("game_id")[["game_id", "season"]]
    team_games_n = plays.groupby("season")["game_id"].nunique() * 2
    play_counts = plays.groupby("season").size()
    offp = table[["game_id", "posteam", *OFF_COLS]].drop_duplicates(["game_id", "posteam"])
    offp = offp.set_index(["game_id", "posteam"])
    defp = table[["game_id", "defteam", *DEF_COLS]].drop_duplicates(["game_id", "defteam"])
    defp = defp.set_index(["game_id", "defteam"])
    home = plays.drop_duplicates("game_id").set_index("game_id")[["home_team", "away_team"]]
    for season in TEST_SEASONS:
        train = table.loc[table["season"].lt(season)]
        model = HistGradientBoostingRegressor(random_state=SEED)
        model.fit(train[FEATURES].to_numpy(np.float32), train["epa"].to_numpy(np.float32))
        ref = table.loc[table["season"].eq(season - 1), SITUATION].to_numpy(np.float32)
        per_game = float(play_counts[season - 1] / team_games_n[season - 1])
        x = np.empty((len(ref), len(FEATURES)), dtype=np.float32)
        x[:, 19:] = ref
        ids = games.loc[games["season"].eq(season) & games["game_id"].isin(game_ids), "game_id"]
        for gid in ids:
            h, a = home.loc[gid, "home_team"], home.loc[gid, "away_team"]
            pred = {}
            for side, o, d, flag in (("home", h, a, 1.0), ("away", a, h, 0.0)):
                x[:, :9] = offp.loc[(gid, o), OFF_COLS].to_numpy(np.float32)
                x[:, 9:18] = defp.loc[(gid, d), DEF_COLS].to_numpy(np.float32)
                x[:, 18] = flag
                pred[side] = float(model.predict(x).mean())
            rows.append(
                {
                    "game_id": gid,
                    "season": season,
                    "epa_home": pred["home"],
                    "epa_away": pred["away"],
                    "plays_per_team_game": per_game,
                    "margin_hat": (pred["home"] - pred["away"]) * per_game,
                }
            )
        print("season", season, "games", len(ids), "train_rows", len(train), flush=True)
    return pd.DataFrame(rows)


def standalone_cell(frame, pick, model_pick, rng):
    out = block_bootstrap(frame, pick.astype(float), model_pick.astype(float), rng)
    return out["accuracy_points"]


def served_record(frame):
    right = frame["model_correct"].astype(bool)
    return f"{int(right.sum())}-{int((~right).sum())}"


def cell(name, desc, effect, lo, hi, ppos, units):
    return {
        "name": name,
        "description": desc,
        "effect": effect,
        "effect_units": units,
        "interval_low": lo,
        "interval_high": hi,
        "probability_positive": ppos,
        "season_start": 2020,
        "season_end": 2025,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("sklearn", sklearn.__version__, flush=True)
    games = pd.read_parquet(
        ROOT / "data/processed/game_features.parquet", columns=["game_id", "gameday"]
    ).drop_duplicates("game_id")
    games["gameday"] = pd.to_datetime(games["gameday"])
    plays = load_plays()
    counts = plays.groupby("season").size()
    print("scrimmage plays per season", counts.to_dict(), "total", int(counts.sum()), flush=True)
    tg = team_games(plays, games)
    pre = pregame_profiles(tg)
    trunc = truncation_check(tg, pre)
    print("truncation", json.dumps(trunc), flush=True)
    table = play_table(plays, pre)
    table.to_parquet(OUT / "play_table.parquet")
    print(
        "play table rows",
        len(table),
        "rows with any NaN profile",
        int(table[[*OFF_COLS, *DEF_COLS]].isna().any(axis=1).sum()),
        flush=True,
    )
    data = pd.read_parquet(ARTIFACT)
    data = data.loc[data["season"].between(2020, 2025)].reset_index(drop=True)
    mh = roll_up(table, plays, set(data["game_id"]))
    mh.to_parquet(OUT / "margin_hat.parquet")
    data = data.merge(mh[["game_id", "margin_hat"]], on="game_id", how="left", validate="1:1")
    if data["margin_hat"].isna().any():
        raise ValueError("games without margin_hat")
    line = data["tue_open_home_spread"]
    if not np.allclose(data["result"] - line, data["margin_vs_open"], equal_nan=True):
        raise ValueError("result is not the home margin")
    data["play_r"] = data["margin_hat"] - line
    y = data["home_covered"].to_numpy(float)
    move = (data["close_home_spread"] - data["tue_open_home_spread"]).abs()
    pops = {
        "all_games": np.ones(len(data), dtype=bool),
        "quiet_games": (move <= SMALL_MOVE).to_numpy(),
    }
    result = {
        "sklearn": sklearn.__version__,
        "plays_per_season": {str(k): int(v) for k, v in counts.items()},
        "truncation": trunc,
        "descriptive": {
            "corr_margin_hat_vs_margin": float(
                np.corrcoef(data["margin_hat"], data["result"])[0, 1]
            ),
            "corr_margin_hat_vs_line": float(np.corrcoef(data["margin_hat"], line)[0, 1]),
            "corr_line_vs_margin": float(np.corrcoef(line, data["result"])[0, 1]),
            "rmse_margin_hat": float(np.sqrt(np.mean((data["margin_hat"] - data["result"]) ** 2))),
            "rmse_line": float(np.sqrt(np.mean((line - data["result"]) ** 2))),
            "margin_hat_std": float(data["margin_hat"].std()),
            "line_std": float(line.std()),
            "margin_hat_mean": float(data["margin_hat"].mean()),
            "corr_r_vs_margin_vs_open": float(
                np.corrcoef(data["play_r"], data["margin_vs_open"])[0, 1]
            ),
        },
        "populations": {},
    }
    variants = {
        "four_term": BASE,
        "plus_play_r": (*BASE, "play_r"),
        "model_only": ("model_logit",),
        "intercept_only": (),
    }
    probs, folds = {}, {}
    for name, feats in variants.items():
        probs[name], folds[name] = loso(data, feats)
    result["fold_coefficients_plus_play_r"] = folds["plus_play_r"]
    result["fold_summary_plus_play_r"] = summary(folds["plus_play_r"])
    rng_shuffle = np.random.default_rng(SEED)
    shuffled = []
    for _ in range(N_SHUFFLES):
        trial = data.copy()
        trial["play_r"] = trial.groupby("season")["play_r"].transform(
            lambda s: rng_shuffle.permutation(s.to_numpy())
        )
        p, _ = loso(trial, (*BASE, "play_r"))
        shuffled.append(p)
    rng = np.random.default_rng(SEED)
    pick = (data["play_r"] > 0).to_numpy()
    model_pick = data["model_pick_home"].to_numpy(bool)
    cells = []
    for pop, mask in pops.items():
        sub = data.loc[mask].reset_index(drop=True)
        ys = y[mask]
        correct = pick[mask] == (ys == 1)
        entry = {
            "games": int(mask.sum()),
            "standalone": {
                "record": f"{int(correct.sum())}-{int((~correct).sum())}",
                "accuracy": float(correct.mean()),
            },
            "served_model_record": served_record(data.loc[mask]),
            "variants": {n: metrics(probs[n][mask], ys) for n in probs},
        }
        ci = standalone_cell(sub, pick[mask], model_pick[mask], rng)
        entry["standalone"]["vs_served_model"] = ci
        entry["paired_plus_vs_four_term"] = block_bootstrap(
            sub, probs["plus_play_r"][mask], probs["four_term"][mask], rng
        )
        entry["decisive_plus_vs_four_term"] = decisive(
            probs["plus_play_r"][mask], probs["four_term"][mask], ys
        )
        sm = [metrics(p[mask], ys) for p in shuffled]
        entry["shuffled_r_control"] = {
            "draws": N_SHUFFLES,
            "mean_accuracy": float(np.mean([m["accuracy"] for m in sm])),
            "sd_accuracy": float(np.std([m["accuracy"] for m in sm])),
            "mean_log_loss": float(np.mean([m["log_loss"] for m in sm])),
            "mean_brier": float(np.mean([m["brier"] for m in sm])),
        }
        result["populations"][pop] = entry
        desc = f"2020-25 {pop}, decisive opener games, week-block bootstrap, margin_hat v opener"
        cells.append(
            cell(
                f"mod57_p2h_standalone_{pop}_accuracy_points",
                desc + "; standalone sign(margin_hat - opener) minus served model",
                ci["delta_mean"],
                ci["ci95"][0],
                ci["ci95"][1],
                ci["probability_positive"],
                "accuracy_points",
            )
        )
        pb = entry["paired_plus_vs_four_term"]
        for key, units, sign in (
            ("accuracy_points", "accuracy_points", 1.0),
            ("log_loss", "log_loss_improvement", -1.0),
            ("brier", "brier_improvement", -1.0),
        ):
            lo, hi = sorted((sign * pb[key]["ci95"][0], sign * pb[key]["ci95"][1]))
            cells.append(
                cell(
                    f"mod57_p2h_added_term_{pop}_{units}",
                    desc + "; four terms plus standardised r minus four terms",
                    sign * pb[key]["delta_mean"],
                    lo,
                    hi,
                    pb[key]["probability_positive"],
                    units,
                )
            )
    batch = {
        "cells": cells,
        "league": "nfl",
        "classification": "unresolved_below_power",
        "family": "mod57_play_level_matchup_model",
        "source": "artifacts/mod57_p2h/results.json",
        "category": "modeling",
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    text = json.dumps(result, indent=1, default=str)
    (OUT / "results.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
