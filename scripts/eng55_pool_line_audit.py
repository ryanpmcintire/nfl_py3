import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.pool_hook import POOL_OFFSET, is_whole, tuesday_market_lines
from nfl_ats.splash_lines import load_splash_capture

DATA_ROOT = Path("data")
SEASON = 2026
WEEKS = range(1, 10)
FIELD_FILE = Path("data/splash/field/2026_field_best_picks_w01-08.parquet")


def main():
    rows = []
    for w in WEEKS:
        cap = load_splash_capture(DATA_ROOT, SEASON, w)
        ids = [g.game_id for g in cap.games]
        mk = tuesday_market_lines(DATA_ROOT, SEASON, w, ids)
        for g in cap.games:
            rows.append(
                {
                    "week": w,
                    "game_id": g.game_id,
                    "pool": float(g.home_spread),
                    "market": mk.get(g.game_id, np.nan),
                }
            )
    df = pd.DataFrame(rows)
    df["diff"] = df["pool"] - df["market"]
    df["market_whole"] = df["market"].map(lambda v: bool(np.isfinite(v) and is_whole(v)))
    df["market_half"] = df["market"].notna() & ~df["market_whole"]
    df["hook_inferred"] = df["market_whole"] & np.isclose(df["diff"].abs(), POOL_OFFSET)
    print("games", len(df), "no market", int(df["market"].isna().sum()))
    print(Counter(df["diff"].round(3).fillna(-99).tolist()))
    print(
        "market whole", int(df["market_whole"].sum()), "market half", int(df["market_half"].sum())
    )
    print(df[df["market_half"] | (df["diff"].abs() >= 1)].to_string())
    print("by week hook", df.groupby("week")["hook_inferred"].sum().to_dict())
    w18 = df[df["week"] <= 8]
    print("hook wk1-8", int(w18["hook_inferred"].sum()), "of", len(w18))
    fld = pd.read_parquet(FIELD_FILE)
    h = fld[fld["home"]][["game_id", "line"]].assign(field_pool=lambda x: -x["line"])
    m = df.merge(h, on="game_id", how="left")
    bad = m[(m["week"] <= 8) & ~np.isclose(m["field_pool"], m["pool"])]
    print("field vs capture disagreements", len(bad))
    print(bad.to_string())
    df.to_csv("artifacts/eng55_pool_line_audit.csv", index=False)
    print(json.dumps({"ok": True}))


main()
