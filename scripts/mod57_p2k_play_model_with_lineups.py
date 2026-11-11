from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod57_p2h_play_level_matchup_model as h
from lead98_price_skew import BASE, loso, metrics, summary
from mod57_u3_deadline_fair_line import block_bootstrap, decisive
from per_flag_weights_eval import SEED

from nfl_ats.participation import latest_participation_snapshot, load_participation_snapshot

ROOT = h.ROOT
OUT = ROOT / "artifacts/mod57_p2k"
PRIOR_SNAPS = 200.0
FIRST_SEASON = 2016
N_CHECK = 12
GROUP_OF = {
    **dict.fromkeys(["QB"], "off_qb"),
    **dict.fromkeys(["OL", "T", "G", "C", "OT", "OG"], "off_ol"),
    **dict.fromkeys(["WR", "TE"], "off_rec"),
    **dict.fromkeys(["RB", "FB", "HB"], "off_rb"),
    **dict.fromkeys(["DL", "DE", "DT", "NT"], "def_dl"),
    **dict.fromkeys(["LB", "OLB", "ILB", "MLB"], "def_lb"),
    **dict.fromkeys(["DB", "CB", "S", "SS", "FS"], "def_db"),
}
GROUPS = ["off_qb", "off_ol", "off_rec", "off_rb", "def_dl", "def_lb", "def_db"]
G_OFF = GROUPS[:4]
G_DEF = GROUPS[4:]
ARMS = {
    "ref": h.FEATURES,
    "k1": [*h.FEATURES, *GROUPS],
    "k2": ["off_is_home", *h.SITUATION, *GROUPS],
}
RECORDED = ("k1", "k2")


def roster_groups():
    root = ROOT / "data/players/raw"
    path = sorted(root.glob("*/weekly_rosters.parquet"))[-1]
    ros = pd.read_parquet(path, columns=["season", "gsis_id", "position"]).dropna()
    ros["group"] = ros["position"].map(GROUP_OF)
    ros = ros.dropna(subset=["group"])
    by_season = ros.groupby(["gsis_id", "season", "group"]).size().reset_index(name="n")
    by_season = by_season.sort_values(["n", "group"], ascending=[False, True])
    by_season = by_season.drop_duplicates(["gsis_id", "season"])
    overall = ros.groupby(["gsis_id", "group"]).size().reset_index(name="n")
    overall = overall.sort_values(["n", "group"], ascending=[False, True])
    overall = overall.drop_duplicates("gsis_id")
    return by_season[["gsis_id", "season", "group"]], overall[["gsis_id", "group"]]


def explode_side(merged, column, side):
    s = merged[column].fillna("").str.split(";").explode()
    s = s[s.ne("")]
    return pd.DataFrame({"row": s.index.to_numpy(), "player": s.to_numpy(), "side": np.int8(side)})


def build_long(plays):
    snap = latest_participation_snapshot(ROOT / "data/players/participation/raw")
    part = load_participation_snapshot(snap)
    part = part[["game_id", "play_id", "offense_players", "defense_players"]]
    keyed = plays[["game_id", "play_id", "season", "posteam", "defteam", "epa", "date"]].copy()
    keyed["row"] = np.arange(len(keyed))
    keyed = keyed.loc[keyed["season"].ge(FIRST_SEASON)]
    merged = keyed.merge(part, on=["game_id", "play_id"], how="left", validate="one_to_one")
    merged = merged.set_index("row")
    covered = {
        "scrimmage_plays_2016_25": len(merged),
        "with_offense_players": int(merged["offense_players"].fillna("").ne("").sum()),
        "with_defense_players": int(merged["defense_players"].fillna("").ne("").sum()),
    }
    off = explode_side(merged, "offense_players", 0)
    dfn = explode_side(merged, "defense_players", 1)
    long = pd.concat([off, dfn], ignore_index=True)
    pos = np.searchsorted(merged.index.to_numpy(), long["row"].to_numpy())
    long["game_id"] = merged["game_id"].to_numpy()[pos]
    long["season"] = merged["season"].to_numpy()[pos]
    long["epa"] = merged["epa"].to_numpy()[pos]
    long["date"] = merged["date"].to_numpy()[pos]
    long["team"] = np.where(
        long["side"].to_numpy() == 0,
        merged["posteam"].to_numpy()[pos],
        merged["defteam"].to_numpy()[pos],
    )
    by_season, overall = roster_groups()
    long = long.merge(
        by_season.rename(columns={"gsis_id": "player", "group": "g1"}),
        on=["player", "season"],
        how="left",
    )
    long = long.merge(
        overall.rename(columns={"gsis_id": "player", "group": "g2"}), on="player", how="left"
    )
    long["group"] = long["g1"].fillna(long["g2"])
    matched = float(long["group"].notna().mean())
    long = long.drop(columns=["g1", "g2"]).dropna(subset=["group"])
    side_ok = np.where(
        long["side"].to_numpy() == 0,
        long["group"].str.startswith("off"),
        long["group"].str.startswith("def"),
    )
    long = long.loc[side_ok].reset_index(drop=True)
    covered["long_rows_with_position_group"] = len(long)
    covered["share_of_player_slots_with_group"] = matched
    return long, covered


def daily_prior_mean(plays):
    sub = plays.loc[plays["season"].ge(FIRST_SEASON), ["date", "epa"]]
    daily = sub.groupby("date")["epa"].agg(["sum", "count"]).sort_index()
    cs = daily["sum"].cumsum() - daily["sum"]
    cn = daily["count"].cumsum() - daily["count"]
    return cs / cn.where(cn > 0)


def player_states(long, mu_pre):
    pg = (
        long.groupby(["player", "side", "game_id", "team", "date", "group"])["epa"]
        .agg(S="sum", n="size")
        .reset_index()
        .sort_values(["player", "side", "date"])
        .reset_index(drop=True)
    )
    key = pg.groupby(["player", "side"], sort=False)
    pg["cS"] = key["S"].cumsum()
    pg["cn"] = key["n"].cumsum()
    pg["pS"] = pg["cS"] - pg["S"]
    pg["pn"] = pg["cn"] - pg["n"]
    pg["mu"] = pg["date"].map(mu_pre)
    pg["rating_pre"] = (pg["pS"] + PRIOR_SNAPS * pg["mu"]) / (pg["pn"] + PRIOR_SNAPS)
    side_plays = long.drop_duplicates(["row", "side"]).groupby(["game_id", "team", "side"]).size()
    side_plays = side_plays.rename("side_plays").reset_index()
    pg = pg.merge(side_plays, on=["game_id", "team", "side"], how="left", validate="many_to_one")
    pg["share"] = pg["n"] / pg["side_plays"]
    return pg


def actual_group_inputs(long, pg, n_rows):
    rated = long[["row", "player", "side", "game_id", "group"]].merge(
        pg[["player", "side", "game_id", "rating_pre"]],
        on=["player", "side", "game_id"],
        how="left",
        validate="many_to_one",
    )
    means = rated.groupby(["row", "group"])["rating_pre"].mean().unstack()
    return means.reindex(index=np.arange(n_rows), columns=GROUPS)


def expected_lineups(pg, mu_pre):
    tg = pg[["game_id", "team", "date"]].drop_duplicates().sort_values(["team", "date"])
    tg["prev_game"] = tg.groupby("team")["game_id"].shift(1)
    tg = tg.dropna(subset=["prev_game"])
    prev = pg[["game_id", "team", "side", "player", "group", "cS", "cn", "share"]].rename(
        columns={"game_id": "prev_game"}
    )
    rows = tg.merge(prev, on=["prev_game", "team"], how="inner")
    rows["mu_at"] = rows["date"].map(mu_pre)
    rows["rating"] = (rows["cS"] + PRIOR_SNAPS * rows["mu_at"]) / (rows["cn"] + PRIOR_SNAPS)
    rows["wr"] = rows["share"] * rows["rating"]
    agg = rows.groupby(["game_id", "team", "group"]).agg(wr=("wr", "sum"), w=("share", "sum"))
    agg["v"] = agg["wr"] / agg["w"]
    lineup = agg["v"].unstack().reindex(columns=GROUPS).reset_index()
    return lineup, tg


def truncation_checks(long, plays, pg, lineup, tg):
    rng = np.random.default_rng(SEED)
    start = pd.Timestamp(f"{h.TEST_SEASONS[0]}-09-01")
    test = pg.loc[pg["date"].ge(start) & pg["pn"].gt(0)]
    pick = test.iloc[rng.choice(len(test), N_CHECK, replace=False)]
    player_diffs = []
    for r in pick.itertuples():
        past = long.loc[
            long["player"].eq(r.player) & long["side"].eq(r.side) & long["date"].lt(r.date)
        ]
        mu = plays.loc[plays["season"].ge(FIRST_SEASON) & plays["date"].lt(r.date), "epa"].mean()
        direct = (past["epa"].sum() + PRIOR_SNAPS * mu) / (len(past) + PRIOR_SNAPS)
        player_diffs.append(abs(direct - r.rating_pre))
    li = lineup.set_index(["game_id", "team"])
    cand = tg.loc[tg["date"].ge(start)]
    pick2 = cand.iloc[rng.choice(len(cand), N_CHECK, replace=False)]
    lineup_diffs = []
    for r in pick2.itertuples():
        mu = plays.loc[plays["season"].ge(FIRST_SEASON) & plays["date"].lt(r.date), "epa"].mean()
        lg = long.loc[long["team"].eq(r.team) & long["date"].lt(r.date)]
        last = lg.loc[lg["date"].eq(lg["date"].max()), "game_id"].iloc[0]
        got = li.loc[(r.game_id, r.team)]
        for side in (0, 1):
            cur = lg.loc[lg["game_id"].eq(last) & lg["side"].eq(side)]
            plays_n = cur["row"].nunique()
            players = cur.groupby(["player", "group"]).size().rename("n").reset_index()
            hist = long.loc[
                long["player"].isin(players["player"])
                & long["side"].eq(side)
                & long["date"].lt(r.date)
            ]
            sums = hist.groupby("player")["epa"].agg(S="sum", c="size")
            players = players.merge(sums, left_on="player", right_index=True)
            players["rating"] = (players["S"] + PRIOR_SNAPS * mu) / (players["c"] + PRIOR_SNAPS)
            players["share"] = players["n"] / plays_n
            for grp, part in players.groupby("group"):
                direct = (part["share"] * part["rating"]).sum() / part["share"].sum()
                lineup_diffs.append(abs(direct - got[grp]))
    diffs = np.asarray(lineup_diffs, dtype=float)
    return {
        "player_games_checked": N_CHECK,
        "player_max_abs_diff": float(np.max(player_diffs)),
        "lineups_checked": N_CHECK,
        "lineup_group_values_checked": len(diffs),
        "lineup_group_values_nan": int(np.isnan(diffs).sum()),
        "lineup_max_abs_diff": float(np.nanmax(diffs)),
    }


def roll_up(table, plays, game_ids, features, lineup_idx, offp, defp):
    rows = []
    games = table.drop_duplicates("game_id")[["game_id", "season"]]
    team_games_n = plays.groupby("season")["game_id"].nunique() * 2
    play_counts = plays.groupby("season").size()
    home = plays.drop_duplicates("game_id").set_index("game_id")[["home_team", "away_team"]]
    idx = {c: i for i, c in enumerate(features)}
    off_cols = [c for c in features if c in h.OFF_COLS]
    def_cols = [c for c in features if c in h.DEF_COLS]
    g_off = [c for c in features if c in G_OFF]
    g_def = [c for c in features if c in G_DEF]
    sit = [c for c in features if c in h.SITUATION]
    for season in h.TEST_SEASONS:
        train = table.loc[table["season"].between(FIRST_SEASON, season - 1)]
        model = HistGradientBoostingRegressor(random_state=SEED)
        model.fit(train[features].to_numpy(np.float32), train["epa"].to_numpy(np.float32))
        ref = table.loc[table["season"].eq(season - 1), sit].to_numpy(np.float32)
        per_game = float(play_counts[season - 1] / team_games_n[season - 1])
        x = np.empty((len(ref), len(features)), dtype=np.float32)
        for j, c in enumerate(sit):
            x[:, idx[c]] = ref[:, j]
        ids = games.loc[games["season"].eq(season) & games["game_id"].isin(game_ids), "game_id"]
        for gid in ids:
            hm, aw = home.loc[gid, "home_team"], home.loc[gid, "away_team"]
            pred = {}
            for side, o, d, flag in (("home", hm, aw, 1.0), ("away", aw, hm, 0.0)):
                if "off_is_home" in idx:
                    x[:, idx["off_is_home"]] = flag
                if off_cols:
                    vals = offp.loc[(gid, o), off_cols].to_numpy(np.float32)
                    for c, v in zip(off_cols, vals, strict=True):
                        x[:, idx[c]] = v
                if def_cols:
                    vals = defp.loc[(gid, d), def_cols].to_numpy(np.float32)
                    for c, v in zip(def_cols, vals, strict=True):
                        x[:, idx[c]] = v
                for c in g_off:
                    x[:, idx[c]] = lineup_idx.get((gid, o, c), np.nan)
                for c in g_def:
                    x[:, idx[c]] = lineup_idx.get((gid, d, c), np.nan)
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


def grade(arm, mh, data, y, pops, model_pick, rng, cells, record):
    frame = data.merge(mh[["game_id", "margin_hat"]], on="game_id", how="left", validate="1:1")
    if frame["margin_hat"].isna().any():
        raise ValueError("games without margin_hat")
    line = frame["tue_open_home_spread"]
    frame["play_r"] = frame["margin_hat"] - line
    out = {
        "descriptive": {
            "corr_margin_hat_vs_margin": float(
                np.corrcoef(frame["margin_hat"], frame["result"])[0, 1]
            ),
            "corr_margin_hat_vs_line": float(np.corrcoef(frame["margin_hat"], line)[0, 1]),
            "rmse_margin_hat": float(
                np.sqrt(np.mean((frame["margin_hat"] - frame["result"]) ** 2))
            ),
            "rmse_line": float(np.sqrt(np.mean((line - frame["result"]) ** 2))),
            "margin_hat_std": float(frame["margin_hat"].std()),
        }
    }
    variants = {"four_term": BASE, "plus_play_r": (*BASE, "play_r"), "intercept_only": ()}
    probs, folds = {}, {}
    for name, feats in variants.items():
        probs[name], folds[name] = loso(frame, feats)
    out["fold_coefficients_plus_play_r"] = folds["plus_play_r"]
    out["fold_summary_plus_play_r"] = summary(folds["plus_play_r"])
    shuf = np.random.default_rng(SEED)
    shuffled = []
    for _ in range(h.N_SHUFFLES):
        trial = frame.copy()
        trial["play_r"] = trial.groupby("season")["play_r"].transform(
            lambda s: shuf.permutation(s.to_numpy())
        )
        p, _ = loso(trial, (*BASE, "play_r"))
        shuffled.append(p)
    pick = (frame["play_r"] > 0).to_numpy()
    out["populations"] = {}
    for pop, mask in pops.items():
        sub = frame.loc[mask].reset_index(drop=True)
        ys = y[mask]
        correct = pick[mask] == (ys == 1)
        entry = {
            "games": int(mask.sum()),
            "standalone": {
                "record": f"{int(correct.sum())}-{int((~correct).sum())}",
                "accuracy": float(correct.mean()),
            },
            "served_model_record": h.served_record(frame.loc[mask]),
            "variants": {n: metrics(probs[n][mask], ys) for n in probs},
        }
        ci = h.standalone_cell(sub, pick[mask], model_pick[mask], rng)
        entry["standalone"]["vs_served_model"] = ci
        entry["paired_plus_vs_four_term"] = block_bootstrap(
            sub, probs["plus_play_r"][mask], probs["four_term"][mask], rng
        )
        entry["decisive_plus_vs_four_term"] = decisive(
            probs["plus_play_r"][mask], probs["four_term"][mask], ys
        )
        sm = [metrics(p[mask], ys) for p in shuffled]
        entry["shuffled_r_control"] = {
            "draws": h.N_SHUFFLES,
            "mean_accuracy": float(np.mean([m["accuracy"] for m in sm])),
            "sd_accuracy": float(np.std([m["accuracy"] for m in sm])),
        }
        out["populations"][pop] = entry
        if not record:
            continue
        desc = (
            f"2020-25 {pop}, decisive opener games, week-block bootstrap, lineup play model {arm}"
        )
        cells.append(
            h.cell(
                f"mod57_p2k_{arm}_standalone_{pop}_accuracy_points",
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
                h.cell(
                    f"mod57_p2k_{arm}_added_term_{pop}_{units}",
                    desc + "; four terms plus standardised r minus four terms",
                    sign * pb[key]["delta_mean"],
                    lo,
                    hi,
                    pb[key]["probability_positive"],
                    units,
                )
            )
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("sklearn", sklearn.__version__, flush=True)
    games = pd.read_parquet(
        ROOT / "data/processed/game_features.parquet", columns=["game_id", "gameday"]
    ).drop_duplicates("game_id")
    games["date"] = pd.to_datetime(games["gameday"])
    plays = h.load_plays()
    plays = plays.merge(
        games[["game_id", "date"]], on="game_id", how="left", validate="many_to_one"
    )
    table = pd.read_parquet(ROOT / "artifacts/mod57_p2h/play_table.parquet")
    aligned = (
        len(table) == len(plays)
        and (table["game_id"].to_numpy() == plays["game_id"].to_numpy()).all()
    )
    if not aligned or not np.allclose(table["epa"].to_numpy(), plays["epa"].to_numpy()):
        raise ValueError("play table is not aligned with plays")
    long, coverage = build_long(plays)
    print("coverage", json.dumps(coverage), flush=True)
    mu_pre = daily_prior_mean(plays)
    pg = player_states(long, mu_pre)
    actual = actual_group_inputs(long, pg, len(plays))
    lineup, tg = expected_lineups(pg, mu_pre)
    trunc = truncation_checks(long, plays, pg, lineup, tg)
    print("truncation", json.dumps(trunc), flush=True)
    lw = lineup.merge(games[["game_id", "date"]], on="game_id")
    lw = lw.loc[lw["date"].dt.year.between(h.TEST_SEASONS[0], h.TEST_SEASONS[-1])]
    lineup_idx = {
        (gid, team, g): float(v)
        for gid, team, g, v in lineup.melt(id_vars=["game_id", "team"], value_vars=GROUPS)
        .dropna()
        .itertuples(index=False)
    }
    for g in GROUPS:
        table[g] = actual[g].to_numpy()
    table.to_parquet(OUT / "play_table_lineups.parquet")
    lineup.to_parquet(OUT / "expected_lineups.parquet")
    train_rows = table.loc[table["season"].ge(FIRST_SEASON)]
    coverage["training_rows_all_groups_share"] = float(
        train_rows[GROUPS].notna().all(axis=1).mean()
    )
    coverage["training_group_nonnull_share"] = {
        g: float(train_rows[g].notna().mean()) for g in GROUPS
    }
    coverage["expected_lineup_nonnull_share_test_team_games"] = {
        g: float(lw[g].notna().mean()) for g in GROUPS
    }
    print("tables saved", json.dumps(coverage), flush=True)
    offp = table[["game_id", "posteam", *h.OFF_COLS]].drop_duplicates(["game_id", "posteam"])
    offp = offp.set_index(["game_id", "posteam"])
    defp = table[["game_id", "defteam", *h.DEF_COLS]].drop_duplicates(["game_id", "defteam"])
    defp = defp.set_index(["game_id", "defteam"])
    data = pd.read_parquet(h.ARTIFACT)
    data = data.loc[data["season"].between(2020, 2025)].reset_index(drop=True)
    line = data["tue_open_home_spread"]
    if not np.allclose(data["result"] - line, data["margin_vs_open"], equal_nan=True):
        raise ValueError("result is not the home margin")
    y = data["home_covered"].to_numpy(float)
    move = (data["close_home_spread"] - data["tue_open_home_spread"]).abs()
    pops = {
        "all_games": np.ones(len(data), dtype=bool),
        "quiet_games": (move <= h.SMALL_MOVE).to_numpy(),
    }
    model_pick = data["model_pick_home"].to_numpy(bool)
    rng = np.random.default_rng(SEED)
    result = {"sklearn": sklearn.__version__, "coverage": coverage, "truncation": trunc, "arms": {}}
    cells = []
    for arm, features in ARMS.items():
        mh = roll_up(table, plays, set(data["game_id"]), features, lineup_idx, offp, defp)
        mh.to_parquet(OUT / f"margin_hat_{arm}.parquet")
        result["arms"][arm] = grade(arm, mh, data, y, pops, model_pick, rng, cells, arm in RECORDED)
        print("graded", arm, flush=True)
    batch = {
        "cells": cells,
        "league": "nfl",
        "classification": "unresolved_below_power",
        "family": "mod57_play_level_matchup_model",
        "source": "artifacts/mod57_p2k/results.json",
        "category": "modeling",
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    text = json.dumps(result, indent=1, default=str)
    (OUT / "results.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
