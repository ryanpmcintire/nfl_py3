import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "artifacts" / "sim10" / "real_games.parquet"
TARGET = TABLE.with_name("real_games_weather.parquet")
FC = (
    ROOT
    / "data"
    / "raw"
    / "forecast_archive"
    / "pool_decision_2009_2025_aware"
    / "forecasts.parquet"
)
FIELDS = [
    "game_id",
    "roof",
    "fetch_status",
    "actual_temp_f",
    "actual_wind_mph",
    "forecast_temp_f",
    "forecast_wind_mph",
    "forecast_precip_prob_pct",
]
OUTDOOR = ["outdoors", "open"]
INDOOR = ["dome", "closed"]


def build():
    rg = pd.read_parquet(TABLE, columns=["game_id", "season", "week"])
    fc = pd.read_parquet(FC)[FIELDS].drop_duplicates("game_id")
    w = rg.merge(fc, on="game_id", how="left")
    w = w.rename(
        columns={
            "roof": "roof_state",
            "actual_temp_f": "temp_f",
            "actual_wind_mph": "wind_mph",
        }
    )
    w["indoor"] = w.roof_state.isin(INDOOR)
    w["outdoor"] = w.roof_state.isin(OUTDOOR)
    w = w.drop(columns=["season", "week"])
    w.to_parquet(TARGET, index=False)
    return rg, w


def report(rg, w):
    d = rg[["game_id", "season"]].merge(w, on="game_id")
    d["wind_ok"] = d.wind_mph.notna()
    d["temp_ok"] = d.temp_f.notna()
    d["fc_ok"] = d.forecast_wind_mph.notna()
    print("rows", len(w), "table rows", len(rg), "unique", w.game_id.is_unique)
    print(d.groupby("season")[["wind_ok", "temp_ok", "fc_ok"]].mean().round(3).to_string())
    print(
        d.groupby(d.roof_state.fillna("missing"))
        .agg(
            n=("game_id", "size"),
            wind=("wind_ok", "mean"),
            temp=("temp_ok", "mean"),
            fc=("fc_ok", "mean"),
        )
        .round(3)
        .to_string()
    )
    o = d[d.outdoor]
    print(
        "outdoor n",
        len(o),
        "missing wind",
        int((~o.wind_ok).sum()),
        "share",
        round(float((~o.wind_ok).mean()), 4),
    )
    print("missing roof_state", int(d.roof_state.isna().sum()))
    fc = pd.read_parquet(FC)[FIELDS].drop_duplicates("game_id").set_index("game_id")
    m = d.set_index("game_id")
    both = m.index.intersection(fc.index)
    a = m.loc[both, ["wind_mph", "temp_f", "forecast_wind_mph", "forecast_temp_f"]]
    b = fc.loc[both, ["actual_wind_mph", "actual_temp_f", "forecast_wind_mph", "forecast_temp_f"]]
    b.columns = a.columns
    same = (a.eq(b) | (a.isna() & b.isna())).all(axis=1)
    print("ids compared", len(both), "identical", int(same.sum()))
    print(
        "mean wind outdoor",
        round(float(o.wind_mph.mean()), 2),
        "mean temp",
        round(float(o.temp_f.mean()), 2),
    )
    print(
        "precip prob outdoor non-null", round(float(o.forecast_precip_prob_pct.notna().mean()), 3)
    )


if __name__ == "__main__":
    rg, w = build()
    report(rg, w)
    sys.exit(0)
