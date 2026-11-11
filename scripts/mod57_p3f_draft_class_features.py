import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "data/raw/draft_picks/20261028T004039Z/draft_picks.parquet"
SNAPS = ROOT / "data/players/raw/20261111T141547Z/snap_counts.parquet"
GAMES = ROOT / "data/processed/game_features.parquet"
OUT = ROOT / "artifacts/mod57_p3f"
TEAM_MAP = {
    "GNB": "GB",
    "KAN": "KC",
    "LVR": "LV",
    "OAK": "LV",
    "SDG": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "NOR": "NO",
    "NWE": "NE",
    "SFO": "SF",
    "TAM": "TB",
}
SIDES = {"off": "O", "def": "D"}
PROBE = [1, 10, 32, 64, 100, 150, 200, 250]


def rookie_shares(draft: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    reg = snaps[snaps.game_type == "REG"]
    team_games = reg.groupby(["season", "team"]).game_id.nunique().rename("team_games")
    per = (
        reg.groupby(["season", "pfr_player_id"])
        .agg(off=("offense_pct", "sum"), dfn=("defense_pct", "sum"))
        .reset_index()
    )
    d = draft.merge(
        per,
        left_on=["season", "pfr_player_id"],
        right_on=["season", "pfr_player_id"],
        how="left",
    )
    d = d.merge(
        team_games.reset_index(),
        left_on=["season", "team_std"],
        right_on=["season", "team"],
        how="left",
        suffixes=("", "_snap"),
    )
    d["matched"] = d["off"].notna() | d["dfn"].notna()
    d["share_off"] = d["off"].fillna(0.0) / d["team_games"]
    d["share_def"] = d["dfn"].fillna(0.0) / d["team_games"]
    return d


def fit_curve(train: pd.DataFrame, side: str, col: str) -> IsotonicRegression:
    sub = train[train.side == side]
    iso = IsotonicRegression(increasing=False, y_min=0.0, out_of_bounds="clip")
    iso.fit(sub["pick_overall"], sub[col])
    return iso


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Draft-class expected rookie snap share per team-season, curve fit walk-forward"
    )
    ap.add_argument("--first-test-season", type=int, default=2015)
    ap.add_argument("--last-test-season", type=int, default=2026)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    draft = pd.read_parquet(
        DRAFT, columns=["season", "round", "pick", "team", "pfr_player_id", "side"]
    )
    draft = draft[draft.season >= 2013].copy()
    draft["pick_overall"] = draft["pick"]
    draft["team_std"] = draft["team"].replace(TEAM_MAP)
    snaps = pd.read_parquet(
        SNAPS,
        columns=[
            "game_id",
            "season",
            "game_type",
            "team",
            "pfr_player_id",
            "offense_pct",
            "defense_pct",
        ],
    )
    snaps["team"] = snaps["team"].replace(TEAM_MAP)
    snaps = snaps[snaps.season <= 2025]
    d = rookie_shares(draft, snaps)
    observed = d[d.season <= 2025]
    failed = {
        "picks_2013_2025": len(observed),
        "null_pfr_id": int(observed.pfr_player_id.isna().sum()),
        "no_snap_row_in_draft_season": int((~observed.matched).sum()),
        "no_snap_row_and_has_id": int(((~observed.matched) & observed.pfr_player_id.notna()).sum()),
        "no_team_games_match": int(observed.team_games.isna().sum()),
    }

    rows = []
    curves = {}
    proof = {}
    for season in range(args.first_test_season, args.last_test_season + 1):
        train = observed[observed.season < season]
        proof[season] = {
            "train_seasons": [int(train.season.min()), int(train.season.max())],
            "max_train_season_lt_test": bool(train.season.max() < season),
        }
        assert train.season.max() < season
        cls = d[d.season == season].copy()
        for name, side in SIDES.items():
            iso = fit_curve(train, side, f"share_{name}")
            curves[(season, name)] = {p: float(iso.predict([p])[0]) for p in PROBE}
            cls[f"exp_{name}"] = np.where(cls.side == side, iso.predict(cls["pick_overall"]), 0.0)
        cls["exp_all"] = cls.exp_off + cls.exp_def
        cls["exp_all_r12"] = np.where(cls["round"] <= 2, cls.exp_all, 0.0)
        agg = (
            cls.groupby("team_std")
            .agg(
                exp_off=("exp_off", "sum"),
                exp_def=("exp_def", "sum"),
                exp_all=("exp_all", "sum"),
                exp_all_r12=("exp_all_r12", "sum"),
                n_picks=("pick", "size"),
            )
            .reset_index()
        )
        agg["season"] = season
        rows.append(agg)
    team = pd.concat(rows, ignore_index=True)
    team.to_parquet(args.out_dir / "team_season_class_value.parquet", index=False)

    g = pd.read_parquet(
        GAMES, columns=["game_id", "season", "week", "home_team", "away_team", "game_type"]
    )
    g = g[(g.game_type == "REG") & g.season.between(args.first_test_season, args.last_test_season)]
    vals = ["exp_off", "exp_def", "exp_all", "exp_all_r12", "n_picks"]
    for side in ("home", "away"):
        t = team.rename(columns={c: f"{side}_{c}" for c in vals})
        g = g.merge(
            t.rename(columns={"team_std": f"{side}_team"}),
            on=["season", f"{side}_team"],
            how="left",
        )
    for c in vals:
        g[f"diff_{c}"] = g[f"home_{c}"] - g[f"away_{c}"]
    g.to_parquet(args.out_dir / "draft_class_features.parquet", index=False)

    cov = g.groupby("season").agg(
        games=("game_id", "size"), with_value=("diff_exp_all", lambda s: int(s.notna().sum()))
    )
    summary = {
        "coverage": cov.reset_index().to_dict("records"),
        "curves": {
            f"{s}_{n}": v
            for (s, n), v in curves.items()
            if s in (args.first_test_season, args.last_test_season)
        },
        "diff_exp_all": {
            k: float(getattr(g.diff_exp_all, k)()) for k in ("mean", "std", "min", "max")
        },
        "join": failed,
        "special_teams_picks": int((observed.side == "S").sum()),
        "walk_forward": proof,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
