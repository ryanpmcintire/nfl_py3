import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAKES = ROOT / "artifacts" / "sim09" / "stakes_teamweek.csv"
FEATURES = ROOT / "data" / "processed" / "game_features.parquet"
SBR = ROOT / "data" / "processed" / "sbr_odds.parquet"
PER_GAME = ROOT / "artifacts" / "opener_evaluation" / "20261101T151614Z" / "per_game.parquet"
OUT = ROOT / "artifacts" / "per13_locked_seed_rest"
WEEKS = (17, 18)
SNAP_FIRST_SEASON = 2016
STARTERS = {"offense": 11, "defense": 11}
TUESDAY_OPENER_SEASONS = (2020, 2025)
DRAWS = 2000
SEED = 20261103
FAMILY = "per13_locked_seed_rest"


def team_rows():
    tw = pd.read_csv(STAKES)
    tw = tw[tw.week.isin(WEEKS)][
        ["season", "week", "team", "locked", "elim", "p_make", "seed_modal", "p_seed_modal"]
    ]
    g = pd.read_parquet(FEATURES)[
        ["game_id", "season", "week", "home_team", "away_team", "result", "spread_line"]
    ]
    sbr = (
        pd.read_parquet(SBR)
        .dropna(subset=["game_id", "open_home_spread"])
        .drop_duplicates("game_id")
    )
    sbr = sbr[["game_id", "open_home_spread", "close_home_spread"]]
    pg = pd.read_parquet(PER_GAME)[["game_id", "tue_open_home_spread"]]
    g = g.merge(sbr, on="game_id", how="left").merge(pg, on="game_id", how="left")
    lo, hi = TUESDAY_OPENER_SEASONS
    g["open"] = np.where(g.season.between(lo, hi), g.tue_open_home_spread, g.open_home_spread)
    g["close"] = g.spread_line
    home = g.assign(team=g.home_team, side=1.0)
    away = g.assign(team=g.away_team, side=-1.0)
    t = pd.concat([home, away], ignore_index=True)
    t = t.merge(tw, on=["season", "week", "team"], how="inner")
    t["ats_open"] = t.side * (t.result - t.open)
    t["ats_close"] = t.side * (t.result - t.close)
    t["move_toward"] = t.side * (t.close - t.open)
    return t


def snap_drop(t):
    s = pd.read_parquet(sorted(glob.glob(str(ROOT / "data/players/raw/*/snap_counts.parquet")))[-1])
    s = s[(s.game_type == "REG") & (s.season >= SNAP_FIRST_SEASON)]
    out = []
    for (season, team), grp in s.groupby(["season", "team"]):
        for unit, col in (("offense", "offense_pct"), ("defense", "defense_pct")):
            piv = grp.pivot_table(index="player", columns="week", values=col, aggfunc="sum").fillna(
                0.0
            )
            for w in WEEKS:
                prior = [c for c in piv.columns if c < w]
                if w not in piv.columns or len(prior) < 2:
                    continue
                base = piv[prior].mean(axis=1).sort_values(ascending=False).head(STARTERS[unit])
                out.append(
                    {
                        "season": season,
                        "week": w,
                        "team": team,
                        "unit": unit,
                        "baseline": base.mean(),
                        "share": piv.loc[base.index, w].mean(),
                    }
                )
    d = pd.DataFrame(out)
    d["drop_pp"] = 100.0 * (d.baseline - d.share)
    return d.merge(
        t[["season", "week", "team", "game_id", "locked", "elim"]],
        on=["season", "week", "team"],
        how="inner",
    )


def boot_diff(df, flag, col):
    base = df.dropna(subset=[col])
    is_flag = (base[flag] == 1).to_numpy()
    is_ref = ((base.locked == 0) & (base.elim == 0)).to_numpy()
    v = base[col].to_numpy()
    codes, ids = pd.factorize(base.game_id)
    n = len(ids)
    fs = np.bincount(codes, weights=v * is_flag, minlength=n)
    fc = np.bincount(codes, weights=is_flag.astype(float), minlength=n)
    rs = np.bincount(codes, weights=v * is_ref, minlength=n)
    rc = np.bincount(codes, weights=is_ref.astype(float), minlength=n)
    est = fs.sum() / fc.sum() - rs.sum() / rc.sum()
    rng = np.random.default_rng(SEED)
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=DRAWS)
    draws = (w @ fs) / (w @ fc) - (w @ rs) / (w @ rc)
    return {
        "n_flag": int(fc.sum()),
        "n_ref": int(rc.sum()),
        "flag_mean": float(fs.sum() / fc.sum()),
        "ref_mean": float(rs.sum() / rc.sum()),
        "diff": float(est),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "p_pos": float((draws > 0).mean()),
        "games_flag": int((fc > 0).sum()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = team_rows()
    counts = (
        t.groupby("week")
        .agg(rows=("team", "size"), locked=("locked", "sum"), elim=("elim", "sum"))
        .to_dict("index")
    )
    print("rows", counts, "open missing", int(t.open.isna().sum()))
    res = {}
    for flag in ("locked", "elim"):
        for col in ("ats_open", "ats_close", "move_toward"):
            res[f"{flag}|{col}"] = boot_diff(t, flag, col)
    sd = snap_drop(t)
    for flag in ("locked", "elim"):
        for unit in ("offense", "defense"):
            res[f"{flag}|drop_{unit}"] = boot_diff(sd[sd.unit == unit], flag, "drop_pp")
    dec = t[t.locked == 1].groupby("season").size().to_dict()
    print("locked rows by season", dec)
    both = t[t.locked == 1].merge(
        t[["game_id", "team", "locked", "elim"]].rename(
            columns={"team": "opp", "locked": "opp_locked", "elim": "opp_elim"}
        ),
        on="game_id",
    )
    both = both[both.opp != both.team]
    print("locked rows with locked opp", int((both.opp_locked == 1).sum()), "of", len(both))
    for k, v in res.items():
        print(
            k, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
        )
    (OUT / "cells.json").write_text(json.dumps(res, indent=1))
    t.to_csv(OUT / "team_rows.csv", index=False)


if __name__ == "__main__":
    main()
