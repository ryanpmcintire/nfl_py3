from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v4_candidate as v4
from mod27_fenced_repairs_replay import log_loss, paired, record

from nfl_ats.pick_probability import MOVE_AVAILABLE_COLUMN, MOVE_COLUMN
from nfl_ats.pick_refresh import SUNDAY_PICK_LOCK_LOCAL_TIME
from nfl_ats.sharp_book_movement_features import (
    LEADER_BOOKS,
    LEADERSHIP_WEIGHTS,
    sharp_book_movement_features,
)

ROOT = v4.ROOT
OUT = ROOT / "artifacts/move_timing_parity"
QUOTES = ROOT / "artifacts/sharp_book_weighted_movement/spread_quotes.parquet"
MODEL = ROOT / "artifacts/v4_candidate/opener_QW.parquet"
SEASONS = (2023, 2024, 2025)
LAST_CALL_SUNDAY_ET = pd.Timedelta(hours=15, minutes=35)
TRAINING_SUNDAY_END_ET = pd.Timedelta(hours=12, minutes=45)
LOCK_SUNDAY_ET = pd.Timedelta(
    hours=SUNDAY_PICK_LOCK_LOCAL_TIME.hour, minutes=SUNDAY_PICK_LOCK_LOCAL_TIME.minute
)
PREFERRED_LABEL = "intraday_hourly"
EASTERN = "America/New_York"
FAMILY = "move_timing_parity_2026"


def sunday_of(anchor_utc):
    local = pd.to_datetime(anchor_utc, utc=True).dt.tz_convert(EASTERN)
    return local.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(
        (6 - local.dt.weekday) % 7, unit="D"
    )


def at_et(sunday, offset):
    return (sunday + offset).dt.tz_localize(EASTERN).dt.tz_convert("UTC")


def load_quotes(ids):
    q = pd.read_parquet(QUOTES)
    q = q.loc[
        q.archive_season.isin(SEASONS)
        & q.nflverse_game_id.isin(ids)
        & q.bookmaker_key.isin(LEADERSHIP_WEIGHTS)
        & q.market.eq("spreads")
    ].copy()
    q["_rank"] = (~q.decision_label.eq(PREFERRED_LABEL)).astype(int)
    q = q.sort_values(["nflverse_game_id", "bookmaker_key", "observed_at_utc", "_rank"])
    before = len(q)
    q = q.drop_duplicates(["nflverse_game_id", "bookmaker_key", "observed_at_utc"], keep="first")
    return q.drop(columns="_rank"), before - len(q)


def games_frame(q, pop):
    games = q.groupby("nflverse_game_id", as_index=False).agg(
        commence_time_utc=("commence_time_utc", "min")
    )
    games = games.rename(columns={"nflverse_game_id": "game_id"}).merge(
        pop[["game_id", "season", "week"]], on="game_id", validate="one_to_one"
    )
    games["commence_time_utc"] = pd.to_datetime(games.commence_time_utc, utc=True)
    games["week_first_commence_utc"] = games.groupby(
        ["season", "week"]
    ).commence_time_utc.transform("min")
    sunday = sunday_of(games.week_first_commence_utc)
    games["cutoff_utc"] = at_et(sunday, LAST_CALL_SUNDAY_ET)
    games["lock_utc"] = at_et(sunday, LOCK_SUNDAY_ET)
    games["train_cutoff_utc"] = at_et(sunday, TRAINING_SUNDAY_END_ET)
    return games


def serving_move(q, games):
    out = sharp_book_movement_features(
        q,
        games[["game_id", "commence_time_utc", "week_first_commence_utc", "cutoff_utc"]],
        include_sunday=True,
    )
    available = out.leader_books.gt(0)
    return pd.DataFrame(
        {
            "game_id": out.game_id,
            "move_m": out.leader_median_net_move.where(available, 0.0).astype(float),
            "available_m": available.astype(float),
            "leader_books_m": out.leader_books,
            "cutoff_used_utc": out.cutoff_utc,
        }
    )


def window_end(q, games):
    k = (
        q.drop(columns="commence_time_utc")
        .rename(columns={"nflverse_game_id": "game_id"})
        .merge(
            games[["game_id", "commence_time_utc", "cutoff_utc", "train_cutoff_utc"]], on="game_id"
        )
    )
    k["observed_at_utc"] = pd.to_datetime(k.observed_at_utc, utc=True)
    k["snapshot_timestamp_utc"] = pd.to_datetime(k.snapshot_timestamp_utc, utc=True)
    cut = pd.concat([k.commence_time_utc, k.cutoff_utc], axis=1).min(axis=1)
    lead = k.bookmaker_key.isin(LEADER_BOOKS)
    hourly = k.decision_label.eq(PREFERRED_LABEL)
    m_end = (
        k.loc[lead & k.observed_at_utc.lt(cut) & k.snapshot_timestamp_utc.lt(cut)]
        .groupby("game_id")
        .observed_at_utc.max()
    )
    t_cut = pd.concat([k.commence_time_utc, k.train_cutoff_utc], axis=1).min(axis=1)
    t_end = (
        k.loc[lead & hourly & k.observed_at_utc.lt(t_cut)].groupby("game_id").observed_at_utc.max()
    )
    return t_end, m_end


def summarise(frame, pa, pb, y, mask):
    sub = frame.loc[mask]
    a, b, yy = pa[mask], pb[mask], y[mask]
    changed = (a >= 0.5) != (b >= 0.5)
    return {
        "games": int(mask.sum()),
        "reference_record": record(a, yy),
        "arm_record": record(b, yy),
        "changed_picks": int(changed.sum()),
        "changed_record_arm": record(b[changed], yy[changed]) if changed.any() else "0-0",
        "changed_record_reference": record(a[changed], yy[changed]) if changed.any() else "0-0",
        "reference_log_loss": float(log_loss(a, yy).mean()),
        "arm_log_loss": float(log_loss(b, yy).mean()),
        "reference_brier": float(((a - yy) ** 2).mean()),
        "arm_brier": float(((b - yy) ** 2).mean()),
        "week_block": paired(sub, a, b, "week_block"),
    }


def reliability(p, y, bins):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return [
        {
            "bin": int(i),
            "games": int((idx == i).sum()),
            "mean_p": float(p[idx == i].mean()),
            "observed": float(y[idx == i].mean()),
        }
        for i in range(bins)
        if (idx == i).any()
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Refit the served stage on a training move rebuilt with the serving window."
    )
    parser.add_argument("--reliability-bins", type=int, default=5)
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    pop = pd.read_parquet(v4.PICK / "per_game.parquet")
    pop["game_id"] = pop["game_id"].astype(str)
    pop = pop.reset_index(drop=True)
    ids = set(pop.loc[pop.season.isin(SEASONS), "game_id"])
    q, dropped = load_quotes(ids)
    games = games_frame(q, pop)
    m = serving_move(q, games)
    t_end, m_end = window_end(q, games)
    games = games.merge(m, on="game_id")
    games["train_end_utc"] = games.game_id.map(t_end)
    games["m_end_utc"] = games.game_id.map(m_end)
    kick_et = games.commence_time_utc.dt.tz_convert(EASTERN)
    games["after_cap"] = games.commence_time_utc.gt(games.lock_utc)
    games["sunday_early"] = kick_et.dt.weekday.eq(6) & kick_et.dt.hour.eq(13)
    mp = pd.read_parquet(MODEL)["home_cover_probability_at_open"]
    mp = mp[~mp.index.duplicated()].reindex(pop["game_id"]).to_numpy(float)
    p_ref, folds_ref, full_ref, ins_ref = v4.stage_two(pop, mp)
    arm = pop.copy()
    joined = arm[["game_id"]].merge(
        games[["game_id", "move_m", "available_m"]], on="game_id", how="left"
    )
    have = joined.move_m.notna().to_numpy()
    arm.loc[have, MOVE_COLUMN] = joined.loc[have, "move_m"].to_numpy()
    arm.loc[have, MOVE_AVAILABLE_COLUMN] = joined.loc[have, "available_m"].to_numpy()
    p_arm, folds_arm, full_arm, ins_arm = v4.stage_two(arm, mp)
    y = pop["home_covered"].to_numpy(float)
    frame = pop.assign(week_block=pop["season"] * 100 + pop["week"])
    flag = frame[["game_id"]].merge(
        games[["game_id", "after_cap", "sunday_early"]], on="game_id", how="left"
    )
    after_cap = flag.after_cap.fillna(False).astype(bool).to_numpy()
    early = flag.sunday_early.fillna(False).astype(bool).to_numpy()
    exposed = frame.season.isin(SEASONS).to_numpy()
    diff = (arm[MOVE_COLUMN] - pop[MOVE_COLUMN]).abs().to_numpy()
    result = {
        "family": FAMILY,
        "games": len(pop),
        "quotes_rows": len(q),
        "duplicate_timestamp_rows_dropped": int(dropped),
        "archive_games": len(games),
        "pop_games_replaced": int(have.sum()),
        "move_value": {
            "changed_games_2023_25": int((diff[exposed] > 1e-9).sum()),
            "mean_abs_change_points": float(diff[exposed].mean()),
            "mean_abs_change_after_cap": float(diff[after_cap].mean()),
            "mean_abs_change_sunday_early": float(diff[early].mean()),
            "available_reference": int(pop.loc[exposed, MOVE_AVAILABLE_COLUMN].sum()),
            "available_arm": int(arm.loc[exposed, MOVE_AVAILABLE_COLUMN].sum()),
        },
        "window_end_et": {
            name: games.loc[games[col]]
            .assign(
                t=lambda g: g.train_end_utc.dt.tz_convert(EASTERN).dt.strftime("%a %H:%M"),
                m=lambda g: g.m_end_utc.dt.tz_convert(EASTERN).dt.strftime("%a %H:%M"),
            )[["t", "m"]]
            .value_counts()
            .rename_axis(["train_end", "arm_end"])
            .reset_index()
            .to_dict("records")
            for name, col in (("after_cap", "after_cap"), ("sunday_early", "sunday_early"))
        },
        "reference": {
            "record": record(p_ref, y),
            "log_loss": float(log_loss(p_ref, y).mean()),
            "in_sample_log_loss": float(log_loss(ins_ref, y).mean()),
            "brier": float(((p_ref - y) ** 2).mean()),
            "fold_coefficients": folds_ref,
            "full_coefficients": full_ref.natural(),
            "reliability": reliability(p_ref, y, args.reliability_bins),
        },
        "arm_M": {
            "record": record(p_arm, y),
            "log_loss": float(log_loss(p_arm, y).mean()),
            "in_sample_log_loss": float(log_loss(ins_arm, y).mean()),
            "brier": float(((p_arm - y) ** 2).mean()),
            "fold_coefficients": folds_arm,
            "full_coefficients": full_arm.natural(),
            "reliability": reliability(p_arm, y, args.reliability_bins),
            "season_block": paired(frame, p_ref, p_arm, "season"),
            "week_block": paired(frame, p_ref, p_arm, "week_block"),
            "per_season": v4.season_table(pop, p_arm),
        },
        "reference_per_season": v4.season_table(pop, p_ref),
        "subsets": {
            "all": summarise(frame, p_ref, p_arm, y, np.ones(len(pop), bool)),
            "after_cap": summarise(frame, p_ref, p_arm, y, after_cap),
            "sunday_early": summarise(frame, p_ref, p_arm, y, early),
            "seasons_2023_25": summarise(frame, p_ref, p_arm, y, exposed),
        },
    }
    games.to_parquet(out_dir / "moves.parquet")
    pd.DataFrame(
        {
            "game_id": pop.game_id,
            "season": pop.season,
            "week": pop.week,
            "home_covered": y,
            "p_reference": p_ref,
            "p_arm": p_arm,
            "after_cap": after_cap,
            "sunday_early": early,
        }
    ).to_parquet(out_dir / "per_game.parquet")
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(out_dir)
    print(
        json.dumps({k: result[k] for k in ("move_value", "window_end_et", "subsets")}, default=str)
    )
    print(
        json.dumps(
            {
                "ref": result["reference"]["record"],
                "arm": result["arm_M"]["record"],
                "season_block": result["arm_M"]["season_block"],
            }
        )
    )


if __name__ == "__main__":
    main()
