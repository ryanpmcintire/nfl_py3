import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mod24_more_games as base  # noqa: E402
import mod24_pooled_levers as lev  # noqa: E402
import pooled_screen as ps  # noqa: E402

NATURAL_WEEK_DAYS = 7
ARMS = {
    "home_rest": ["home_rest_days"],
    "away_rest": ["away_rest_days"],
    "bye_diff": ["bye_diff"],
    "short_week_diff": ["short_week_diff"],
    "rest_bye_bundle": ["home_rest_days", "away_rest_days", "bye_diff", "short_week_diff"],
}


def schedule_features(f):
    f = f.copy()
    f["gameday"] = pd.to_datetime(f["gameday"])
    long = pd.concat(
        [
            f[["game_id", "season", "week", "gameday", "home_team"]].rename(
                columns={"home_team": "team"}
            ),
            f[["game_id", "season", "week", "gameday", "away_team"]].rename(
                columns={"away_team": "team"}
            ),
        ]
    ).sort_values(["team", "season", "gameday", "game_id"])
    g = long.groupby(["team", "season"], sort=False)
    long["rest_days"] = (long["gameday"] - g["gameday"].shift(1)).dt.days.astype(float)
    long["bye"] = ((long["week"] - g["week"].shift(1)) > 1).astype(float)
    long.loc[g.cumcount() == 0, "bye"] = np.nan
    long["short"] = (long["rest_days"] < NATURAL_WEEK_DAYS).astype(float)
    long.loc[long["rest_days"].isna(), "short"] = np.nan
    out = f
    for side in ("home", "away"):
        m = long.rename(
            columns={
                "team": f"{side}_team",
                "rest_days": f"{side}_rest_days",
                "bye": f"{side}_bye",
                "short": f"{side}_short",
            }
        )[["game_id", f"{side}_team", f"{side}_rest_days", f"{side}_bye", f"{side}_short"]]
        out = out.merge(
            m.drop_duplicates(["game_id", f"{side}_team"]),
            on=["game_id", f"{side}_team"],
            how="left",
        )
    out["bye_diff"] = out["home_bye"] - out["away_bye"]
    out["short_week_diff"] = out["home_short"] - out["away_short"]
    return out


def main():
    arm = sys.argv[1]
    extra = ARMS[arm]
    shared = ps.shared_columns()
    real = pd.read_parquet

    def patched_read(path, *a, **k):
        df = real(path, *a, **k)
        if Path(path) in (Path(base.NFL_FEATURES), Path(base.CFB_FEATURES)):
            return schedule_features(df)
        return df

    pd.read_parquet = patched_read
    original = lev.predict_arms

    def install(_):
        def patched(train, test, cols, arms):
            out = original(train, test, shared, ())
            y = train["ats_margin"].to_numpy(float)
            a, b = lev.prep(train[cols].to_numpy(float), test[cols].to_numpy(float))
            from sklearn.linear_model import Ridge

            m = Ridge(alpha=lev.ALPHA).fit(a, y)
            out["variant"] = (m.predict(b), float(np.std(y - m.predict(a), ddof=1)))
            return out

        lev.predict_arms = patched

    ps.install = install
    ps.resolve = lambda s: None
    ps.shared_columns = lambda: shared + extra
    sys.argv = ["pooled_screen", arm, "--out", str(ps.OUT / f"xlg10_{arm}.json")]
    ps.main()


if __name__ == "__main__":
    main()
