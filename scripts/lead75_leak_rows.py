import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nfl_ats.pbp import enrich_with_pbp_features, load_pbp_snapshot, snapshot_from_root  # noqa: E402

OUT = ROOT / "artifacts/lead75/leak_rows.json"
COLS = ["home_pbp_off_pass_rate", "away_pbp_off_pass_rate", "diff_pbp_off_pass_rate"]
ET = "America/New_York"


def load_pairs() -> pd.DataFrame:
    paths = glob.glob(str(ROOT / "data/raw/paired_wind_forecast/*/deadline/*/forecasts.parquet"))
    frame = pd.concat([pd.read_parquet(p).assign(run=Path(p).parent.name) for p in paths], ignore_index=True)
    print("deadline rows", len(frame), "fetch_status", frame["fetch_status"].value_counts().to_dict(),
          "pair_status", frame["pair_status"].value_counts().to_dict(), flush=True)
    for col in ["kickoff", "pool_deadline_utc", "captured_at_utc", "tuesday_captured_at_utc"]:
        frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    ok = frame.loc[
        frame["pair_status"].eq("paired")
        & frame["fetch_status"].eq("ok")
        & frame["captured_at_utc"].lt(frame["kickoff"])
        & frame["tuesday_captured_at_utc"].lt(frame["captured_at_utc"])
        & frame["forecast_wind_mph"].notna()
        & frame["tuesday_forecast_wind_mph"].notna()
    ]
    ok = ok.sort_values("captured_at_utc").drop_duplicates(["game_id", "product_id"], keep="last")
    return ok.assign(revision=ok["forecast_wind_mph"] - ok["tuesday_forecast_wind_mph"])


def team_long(games: pd.DataFrame) -> pd.DataFrame:
    base = ["game_id", "season", "week", "gameday"]
    return pd.concat([
        games[base + ["home_team"]].rename(columns={"home_team": "team"}),
        games[base + ["away_team"]].rename(columns={"away_team": "team"}),
    ]).sort_values(["team", "gameday", "game_id"])


def main() -> int:
    manifest = json.loads((ROOT / "data/processed/game_features_pbp.manifest.json").read_text())
    snap = snapshot_from_root(ROOT / "data/pbp/raw" / manifest["source_pbp_snapshot"])
    pbp = load_pbp_snapshot(snap)
    games = pd.read_parquet(ROOT / "data/processed/game_features.parquet")
    games["gameday"] = pd.to_datetime(games["gameday"])
    print("game cols", [c for c in games.columns if any(k in c for k in ("ats", "result", "spread", "gametime", "margin"))], flush=True)
    stored = pd.read_parquet(ROOT / "data/processed/game_features_pbp.parquet").set_index("game_id")
    pairs = load_pairs()
    pairs = pairs.merge(games[["game_id", "gameday", "home_team", "away_team"]], on="game_id", how="left")
    print("pairs", len(pairs), "games", pairs["game_id"].nunique(), "unmatched_to_games", int(pairs["gameday"].isna().sum()), flush=True)

    pbp_games = set(pbp["game_id"].unique())
    long = team_long(games.loc[games["game_id"].isin(pbp_games)])
    rows = []
    for rec in pairs.itertuples(index=False):
        cut_date = rec.tuesday_captured_at_utc.tz_convert(ET).tz_localize(None).normalize()
        for side, team in (("home", rec.home_team), ("away", rec.away_team)):
            hist = long.loc[long["team"].eq(team) & long["gameday"].lt(rec.gameday)]
            last = hist.iloc[-1] if len(hist) else None
            rows.append({
                "game_id": rec.game_id, "product": rec.product_id, "week": int(rec.week), "side": side, "team": team,
                "gameday": str(rec.gameday.date()), "tuesday_captured_at_utc": rec.tuesday_captured_at_utc.isoformat(),
                "last_pbp_game": None if last is None else last["game_id"],
                "last_pbp_gameday": None if last is None else str(last["gameday"].date()),
                "last_game_on_or_after_tuesday_cut": bool(last is not None and last["gameday"] >= cut_date),
                "last_game_is_self_or_later": bool(last is not None and last["gameday"] >= rec.gameday),
            })
    side_rows = pd.DataFrame(rows)

    pbp_day = pbp["game_id"].map(games.set_index("game_id")["gameday"])
    pairs["cut_date"] = pairs["tuesday_captured_at_utc"].dt.tz_convert(ET).dt.tz_localize(None).dt.normalize()
    rebuilt = {}
    t0 = time.time()
    for cut, grp in pairs.groupby("cut_date"):
        trunc = pbp.loc[pbp_day.lt(cut)]
        frame = enrich_with_pbp_features(games, trunc).set_index("game_id")
        for g in grp["game_id"].unique():
            rebuilt[g] = {c: float(frame.loc[g, c]) for c in COLS} | {"removed_plays": int(len(pbp) - len(trunc)), "cut": str(cut.date())}
        print("cut", cut.date(), "games", grp["game_id"].nunique(), round(time.time() - t0), flush=True)

    diffs = []
    for g, vals in rebuilt.items():
        d = []
        for c in COLS:
            a, b = vals[c], float(stored.loc[g, c])
            d.append(0.0 if (np.isnan(a) and np.isnan(b)) else abs(a - b))
        diffs.append({"game_id": g, "max_abs_diff": max(d), "nan_stored": bool(np.isnan(float(stored.loc[g, "diff_pbp_off_pass_rate"]))), "cut": vals["cut"], "removed_plays": vals["removed_plays"]})
    diffs = pd.DataFrame(diffs)

    margin_col = "ats_margin" if "ats_margin" in games.columns else None
    gm = games.set_index("game_id")
    pairs["pass_diff_stored"] = pairs["game_id"].map(stored["diff_pbp_off_pass_rate"])
    pairs["pass_diff_rebuilt"] = pairs["game_id"].map({g: v["diff_pbp_off_pass_rate"] for g, v in rebuilt.items()})
    pairs["ats_margin"] = pairs["game_id"].map(gm[margin_col]) if margin_col else np.nan
    interim = pairs.loc[pairs["week"].eq(7) & pairs["product_id"].eq("MEX") & pairs["roof"].eq("outdoors") & pairs["ats_margin"].notna()]
    look = {}
    for name in ("pass_diff_stored", "pass_diff_rebuilt"):
        x = (interim["revision"] * interim[name]).to_numpy(float)
        y = interim["ats_margin"].to_numpy(float)
        look[name] = {"n": int(len(x)), "slope_through_origin": float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else None,
                      "nonzero_terms": int((np.abs(x) > 0).sum())}
    out = {
        "snapshot": snap.snapshot_id,
        "pairs_ok": int(len(pairs)),
        "pair_games": int(pairs["game_id"].nunique()),
        "pairs_by_week_product": {f"{k[0]}_{k[1]}": int(v) for k, v in pairs.groupby(["week", "product_id"]).size().items()},
        "team_sides_checked": int(len(side_rows)),
        "team_sides_without_prior_pbp": int(side_rows["last_pbp_game"].isna().sum()),
        "team_sides_last_game_on_or_after_tuesday_cut": int(side_rows["last_game_on_or_after_tuesday_cut"].sum()),
        "team_sides_last_game_self_or_later": int(side_rows["last_game_is_self_or_later"].sum()),
        "games_rebuilt_at_tuesday_cut": int(len(diffs)),
        "games_rebuilt_differs": int((diffs["max_abs_diff"] > 1e-12).sum()),
        "max_abs_diff_rebuilt_vs_stored": float(diffs["max_abs_diff"].max()),
        "stored_pass_diff_nan": int(diffs["nan_stored"].sum()),
        "interim_reproduction": look,
        "interim_rows": interim[["game_id", "revision", "pass_diff_stored", "pass_diff_rebuilt", "ats_margin"]].to_dict("records"),
        "side_rows": side_rows.to_dict("records"),
        "game_rows": diffs.to_dict("records"),
    }
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("side_rows", "game_rows")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
