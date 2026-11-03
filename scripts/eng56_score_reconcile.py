import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIRST_SEASON = 2011


def pbp_finals(pbp_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(str(pbp_root / "season=*" / "plays.parquet"))):
        cols = [
            "game_id",
            "home_team",
            "posteam",
            "defteam",
            "posteam_score",
            "posteam_score_post",
            "score_differential",
        ]
        frames.append(pd.read_parquet(path, columns=cols))
    plays = pd.concat(frames, ignore_index=True)
    plays = plays[plays["posteam"].notna() & plays["posteam_score"].notna()].copy()
    plays["defteam_score"] = plays["posteam_score"] - plays["score_differential"]
    plays["is_home_pos"] = plays["posteam"] == plays["home_team"]
    plays["home_now"] = np.where(
        plays["is_home_pos"], plays["posteam_score"], plays["defteam_score"]
    )
    plays["away_now"] = np.where(
        plays["is_home_pos"], plays["defteam_score"], plays["posteam_score"]
    )
    post = plays["posteam_score_post"].fillna(plays["posteam_score"])
    plays["home_post"] = np.where(plays["is_home_pos"], post, plays["defteam_score"])
    plays["away_post"] = np.where(plays["is_home_pos"], plays["defteam_score"], post)
    out = plays.groupby("game_id").agg(
        pbp_home=("home_post", "max"),
        pbp_away=("away_post", "max"),
        pbp_home_pre=("home_now", "max"),
        pbp_away_pre=("away_now", "max"),
    )
    return out.reset_index()


def grade(home: pd.Series, away: pd.Series, line: pd.Series) -> pd.Series:
    margin = (home - away) - line
    return pd.Series(np.sign(margin), index=margin.index).where(
        line.notna() & home.notna() & away.notna()
    )


def pool_lines(splash_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(splash_dir.glob("2026_week*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for game in payload.get("games", []):
            if game.get("home_spread") is not None:
                rows.append(
                    {
                        "game_id": game["game_id"],
                        "pool_line": float(game["home_spread"]),
                        "capture": path.name,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(ROOT / "data/processed/game_features.parquet"))
    parser.add_argument("--sbr", default=str(ROOT / "data/processed/sbr_odds.parquet"))
    parser.add_argument("--pbp-root", required=True)
    parser.add_argument("--splash-dir", default=str(ROOT / "data/splash"))
    parser.add_argument("--open-close", required=True)
    parser.add_argument("--out", default=str(ROOT / "artifacts/eng56/reconciliation.csv"))
    args = parser.parse_args()

    feats = pd.read_parquet(args.features)
    feats = feats[(feats["season"] >= FIRST_SEASON) & feats["home_score"].notna()].copy()
    feats = feats[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "spread_line",
        ]
    ]
    sbr = pd.read_parquet(args.sbr)[
        ["game_id", "away_score", "home_score", "open_home_spread", "close_home_spread"]
    ]
    sbr = sbr.dropna(subset=["game_id"]).rename(
        columns={
            "away_score": "sbr_away",
            "home_score": "sbr_home",
            "open_home_spread": "sbr_open",
            "close_home_spread": "sbr_close",
        }
    )
    oc = pd.read_parquet(args.open_close)[["nflverse_game_id", "opening_home_spread"]].rename(
        columns={"nflverse_game_id": "game_id", "opening_home_spread": "oc_open"}
    )
    df = feats.merge(sbr, on="game_id", how="left").merge(
        pbp_finals(Path(args.pbp_root)), on="game_id", how="left"
    )
    df = df.merge(oc.drop_duplicates("game_id"), on="game_id", how="left")
    pool = pool_lines(Path(args.splash_dir))
    pool_min = (
        pool.groupby("game_id")["pool_line"].agg(pool_first="first", pool_last="last").reset_index()
    )
    df = df.merge(pool_min, on="game_id", how="left")

    df["sbr_diff"] = np.where(
        df["sbr_home"].notna(),
        (df["sbr_home"] != df["home_score"]) | (df["sbr_away"] != df["away_score"]),
        np.nan,
    )
    df["pbp_diff"] = np.where(
        df["pbp_home"].notna(),
        (df["pbp_home"] != df["home_score"]) | (df["pbp_away"] != df["away_score"]),
        np.nan,
    )
    df["pbp_pre_diff"] = np.where(
        df["pbp_home_pre"].notna(),
        (df["pbp_home_pre"] != df["home_score"]) | (df["pbp_away_pre"] != df["away_score"]),
        np.nan,
    )
    lines = {
        "close_nflverse": "spread_line",
        "open_sbr": "sbr_open",
        "close_sbr": "sbr_close",
        "open_oc": "oc_open",
        "pool_first": "pool_first",
        "pool_last": "pool_last",
    }
    for name, col in lines.items():
        base = grade(df["home_score"], df["away_score"], df[col])
        for src, h, a in (("sbr", "sbr_home", "sbr_away"), ("pbp", "pbp_home", "pbp_away")):
            alt = grade(df[h], df[a], df[col])
            df[f"chg_{src}_{name}"] = (base.notna() & alt.notna() & (base != alt)).astype(int)
    chg_cols = [c for c in df.columns if c.startswith("chg_")]
    df["any_grade_change"] = df[chg_cols].max(axis=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("games", len(df))
    for src, flag, key in (
        ("sbr", "sbr_diff", "sbr_home"),
        ("pbp_post", "pbp_diff", "pbp_home"),
        ("pbp_pre", "pbp_pre_diff", "pbp_home"),
    ):
        have = df[df[key].notna()]
        print(
            src,
            "compared",
            len(have),
            "exact",
            int((have[flag] == 0).sum()),
            "diff",
            int((have[flag] == 1).sum()),
        )
    print("grade changes per column", df[chg_cols].sum().to_dict())
    cols = [
        "game_id",
        "home_score",
        "away_score",
        "sbr_home",
        "sbr_away",
        "pbp_home",
        "pbp_away",
        "pbp_home_pre",
        "pbp_away_pre",
        *chg_cols,
    ]
    print(df[(df["sbr_diff"] == 1) | (df["pbp_diff"] == 1)][cols].to_string())


main()
