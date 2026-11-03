import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.forecast_cold_visitor_tilt_overlay import tuesday_noon_et_cutoff_utc
from nfl_ats.market_data import load_quote_history
from nfl_ats.pool_hook import POOL_OFFSET, is_whole
from nfl_ats.splash_lines import load_splash_capture

DATA_ROOT = Path("data")
SEASON = 2026
WEEKS = range(1, 10)
WINDOW_HOURS = 6


def consensus_series(quotes):
    return (
        quotes.groupby(["nflverse_game_id", "observed_at_utc"])["home_spread_line"]
        .median()
        .reset_index()
        .sort_values("observed_at_utc")
    )


def main():
    rows = []
    for w in WEEKS:
        cap = load_splash_capture(DATA_ROOT, SEASON, w)
        captured = pd.Timestamp(cap.captured_at_et).tz_convert("UTC")
        noon = tuesday_noon_et_cutoff_utc(captured)
        quotes = load_quote_history(DATA_ROOT / "market" / "raw", since=noon)
        ids = [g.game_id for g in cap.games]
        q = quotes.loc[quotes["market"].eq("spreads") & quotes["nflverse_game_id"].isin(ids)].copy()
        q["observed_at_utc"] = pd.to_datetime(q["observed_at_utc"], utc=True)
        commence = pd.to_datetime(q["commence_time_utc"], utc=True)
        q = q.loc[q["observed_at_utc"].ge(noon) & q["observed_at_utc"].lt(commence)]
        snaps = consensus_series(q)
        for g in cap.games:
            s = snaps.loc[snaps["nflverse_game_id"].eq(g.game_id)]
            first = s["home_spread_line"].iloc[0] if len(s) else np.nan
            atcap = s.loc[s["observed_at_utc"].le(captured)]
            at_capture = atcap["home_spread_line"].iloc[-1] if len(atcap) else np.nan
            win = s.loc[s["observed_at_utc"].lt(noon + timedelta(hours=WINDOW_HOURS))]
            window = win["home_spread_line"].median() if len(win) else np.nan
            rows.append(
                {
                    "week": w,
                    "game_id": g.game_id,
                    "pool": float(g.home_spread),
                    "first": first,
                    "at_capture": at_capture,
                    "window": window,
                    "n_snaps": len(s),
                    "captured": captured,
                }
            )
    df = pd.DataFrame(rows)
    out = {}
    for a in ("first", "at_capture", "window"):
        d = (df["pool"] - df[a]).abs()
        whole = df[a].map(lambda v: bool(np.isfinite(v) and is_whole(v)))
        half = df[a].notna() & ~whole
        hook = whole & np.isclose(d, POOL_OFFSET)
        df[f"hook_{a}"] = hook
        df[f"side_{a}"] = np.where(hook, np.sign(df["pool"] - df[a]), 0.0)
        out[a] = {
            "equal": int((d < 1e-9).sum()),
            "hook_exact": int(hook.sum()),
            "whole_market": int(whole.sum()),
            "half_market": int(half.sum()),
            "clean_rows": int((np.isclose(d, 0) | hook).sum()),
            "unclean_rows": int((~(np.isclose(d, 0) | hook)).sum()),
            "hook_wk1_8": int(hook[df["week"] <= 8].sum()),
            "hook_wk9": int(hook[df["week"] == 9].sum()),
        }
    print(json.dumps(out, indent=1))
    audit = pd.read_csv("artifacts/eng55_pool_line_audit.csv")
    odd = audit.loc[(audit["diff"].abs() >= 1.0)]["game_id"]
    cols = ["week", "game_id", "pool", "first", "at_capture", "window", "n_snaps"]
    print(df.loc[df["game_id"].isin(odd), cols].to_string())
    chg = df.loc[
        (df["hook_first"] != df["hook_at_capture"])
        | (df["hook_first"] != df["hook_window"])
        | (df["side_first"] != df["side_at_capture"])
        | (df["side_first"] != df["side_window"]),
        [
            "week",
            "game_id",
            "pool",
            "first",
            "at_capture",
            "window",
            "hook_first",
            "hook_at_capture",
            "hook_window",
            "side_first",
            "side_at_capture",
            "side_window",
        ],
    ]
    print("changed hook/side games", len(chg))
    print(chg.to_string())
    df.drop(columns=["captured"]).to_csv("artifacts/eng55_anchor_compare.csv", index=False)


main()
