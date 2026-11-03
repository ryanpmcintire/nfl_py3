from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nfl_ats.clv import HISTORICAL_CAPTURE_KIND, load_snapshot_manifest_index
from nfl_ats.features import _kickoff_utc
from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock

SECONDS_PER_HOUR = 3600.0
SEASON_FIRST = 2011
SEASON_LIVE = 2026
SCHEDULE_COLUMNS = ["game_id", "season", "week", "game_type", "gameday", "gametime", "weekday"]
PROCESSED = ("game_features.parquet", "game_features_weak_stack.parquet")
EASTERN = "America/New_York"


def final_schedule(root: Path) -> tuple[pd.DataFrame, Path]:
    latest = sorted(root.glob("2*/schedules.parquet"))[-1]
    s = pd.read_parquet(latest, columns=SCHEDULE_COLUMNS)
    s = s.loc[s["season"].between(SEASON_FIRST, SEASON_LIVE)].copy()
    s["kickoff_final"] = _kickoff_utc(s)
    return s, latest


def opener_commence(root: Path) -> pd.DataFrame:
    index = load_snapshot_manifest_index(root)
    sel = index.loc[
        index["capture_kind"].eq(HISTORICAL_CAPTURE_KIND) & index["decision_label"].eq("tue_open")
    ]
    rows = []
    for r in sel.itertuples(index=False):
        q = pd.read_parquet(
            Path(str(r.dir)) / "quotes.parquet",
            columns=["nflverse_game_id", "commence_time_utc", "market", "outcome_side"],
        )
        q = q.loc[q["market"].eq("spreads") & q["outcome_side"].eq("HOME")]
        for gid, g in q.groupby("nflverse_game_id"):
            rows.append(
                {
                    "game_id": gid,
                    "snapshot_id": r.snapshot_id,
                    "snapshot_utc": pd.Timestamp(r.snapshot_timestamp_utc),
                    "commence_opener": pd.Timestamp(g["commence_time_utc"].iloc[0]),
                    "commence_distinct": int(g["commence_time_utc"].nunique()),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values("snapshot_utc").drop_duplicates("game_id", keep="last")


def slot(ts: pd.Timestamp) -> str:
    local = ts.tz_convert(EASTERN)
    return f"{local.day_name()[:3]} {local:%H:%M}"


def schedule_series(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("2*/schedules.parquet")):
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        s = pd.read_parquet(path, columns=SCHEDULE_COLUMNS)
        s = s.loc[s["season"].eq(SEASON_LIVE)].copy()
        s["kickoff"] = _kickoff_utc(s)
        s["fetched"] = pd.Timestamp(manifest["fetched_at_utc"])
        s["snap"] = path.parent.name
        frames.append(s[["game_id", "kickoff", "fetched", "snap"]])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--market", default="data/market/raw")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument(
        "--eval", default="artifacts/opener_evaluation/20261103T195331Z/per_game.parquet"
    )
    ap.add_argument("--out", default="artifacts/eng58/audit.csv")
    ap.add_argument("--series-out", default="artifacts/eng58/live_schedule_changes.csv")
    ap.add_argument("--processed-out", default="artifacts/eng58/processed_kickoff_stale.csv")
    args = ap.parse_args()
    final, latest = final_schedule(Path(args.raw))
    print("final schedule", latest, len(final))
    df = final.merge(opener_commence(Path(args.market)), on="game_id", how="left")
    graded = set(pd.read_parquet(args.eval, columns=["game_id"])["game_id"])
    df["graded"] = df["game_id"].isin(graded)
    df["has_opener_commence"] = df["commence_opener"].notna()
    df["shift_h"] = (df["kickoff_final"] - df["commence_opener"]).dt.total_seconds() / (
        SECONDS_PER_HOUR
    )
    df["changed"] = df["shift_h"].abs().gt(0)
    df["date_changed"] = df["changed"] & (
        df["kickoff_final"].dt.tz_convert(EASTERN).dt.date
        != df["commence_opener"].dt.tz_convert(EASTERN).dt.date
    )
    print("final games", len(df), "with opener commence", int(df["has_opener_commence"].sum()))
    print(
        "graded",
        int(df["graded"].sum()),
        "graded with commence",
        int((df["graded"] & df["has_opener_commence"]).sum()),
    )
    print(
        "changed",
        int(df["changed"].sum()),
        "changed graded",
        int((df["changed"] & df["graded"]).sum()),
    )
    print("multi-commence games", int(df["commence_distinct"].gt(1).sum()))
    tick_h = df.loc[df["changed"], "shift_h"].abs().round(6).mode().iloc[0]
    print("modal nonzero shift h (feed tick)", tick_h)
    rows = []
    for r in df.loc[df["changed"]].itertuples():
        wk = df.loc[(df["season"] == r.season) & (df["week"] == r.week)]
        lock_final = sunday_pick_lock(wk["kickoff_final"])
        lock_open = sunday_pick_lock(wk["commence_opener"].fillna(wk["kickoff_final"]))
        dl_final = pick_deadline(r.kickoff_final, lock_final)
        dl_open = pick_deadline(r.commence_opener, lock_open)
        rows.append(
            {
                "game_id": r.game_id,
                "season": r.season,
                "week": r.week,
                "game_type": r.game_type,
                "graded": r.graded,
                "snapshot_id": r.snapshot_id,
                "commence_opener_utc": r.commence_opener,
                "kickoff_final_utc": r.kickoff_final,
                "shift_h": round(r.shift_h, 3),
                "class": "feed_tick" if abs(r.shift_h) <= tick_h + 1e-9 else "schedule_move",
                "date_changed": r.date_changed,
                "slot_opener": slot(r.commence_opener),
                "slot_final": slot(r.kickoff_final),
                "slot_changed": slot(r.commence_opener) != slot(r.kickoff_final),
                "deadline_opener_utc": dl_open,
                "deadline_final_utc": dl_final,
                "deadline_shift_h": round(
                    (dl_final - dl_open).total_seconds() / SECONDS_PER_HOUR, 3
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print("audit rows", len(out))
    if len(out):
        print(out.groupby("season")["game_id"].count().to_string())
        cols = ["game_id", "shift_h", "slot_opener", "slot_final", "deadline_shift_h", "graded"]
        print(out[cols].to_string())
    series = schedule_series(Path(args.raw))
    latest_kick = (
        series.sort_values("fetched")
        .drop_duplicates("game_id", keep="last")[["game_id", "kickoff"]]
        .rename(columns={"kickoff": "kickoff_latest"})
    )
    series = series.merge(latest_kick, on="game_id")
    moved = series.loc[series["kickoff"].ne(series["kickoff_latest"])]
    moved.to_csv(args.series_out, index=False)
    print(
        "live 2026 snapshots",
        series["snap"].nunique(),
        series["fetched"].min(),
        series["fetched"].max(),
    )
    print(
        "2026 snapshot rows differing from latest", len(moved), "games", moved["game_id"].nunique()
    )
    stale = []
    live = final.loc[final["season"].eq(SEASON_LIVE), ["game_id", "kickoff_final"]]
    for name in PROCESSED:
        f = pd.read_parquet(Path(args.processed) / name, columns=["game_id", "season", "kickoff"])
        j = f.loc[f["season"].eq(SEASON_LIVE)].merge(live, on="game_id")
        j["kick"] = pd.to_datetime(j["kickoff"], utc=True)
        d = j.loc[j["kick"].ne(j["kickoff_final"])].copy()
        d["file"] = name
        print(name, "rows", len(j), "stale kickoff", len(d))
        stale.append(d[["file", "game_id", "kick", "kickoff_final"]])
    pd.concat(stale).to_csv(args.processed_out, index=False)


if __name__ == "__main__":
    main()
