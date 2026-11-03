from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nflverse_injuries"
OUT = ROOT / "data" / "processed" / "injury_revisions_2026.parquet"
KEY = ["season", "week", "gsis_id"]
FIELDS = [
    "report_status",
    "practice_status",
    "report_primary_injury",
    "report_secondary_injury",
    "practice_primary_injury",
    "practice_secondary_injury",
]
GAME_STATUS = ["Out", "Doubtful", "Questionable"]
ET = "America/New_York"


def load_snapshots(season: int) -> list[tuple[str, pd.Timestamp, str, pd.DataFrame]]:
    snaps = []
    for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        fetched = pd.Timestamp(manifest["fetched_utc"])
        if fetched.tzinfo is None:
            fetched = fetched.tz_localize("UTC")
        path = d / "injuries.parquet"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_parquet(path)
        frame = frame[frame["season"] == season].copy()
        frame["week"] = frame["week"].astype("Int64")
        snaps.append((d.name, fetched, digest, frame))
    snaps.sort(key=lambda s: s[1])
    return snaps


def _clean(v):
    return None if pd.isna(v) else v


def _same(a, b) -> bool:
    return bool((pd.isna(a) and pd.isna(b)) or a == b)


def revisions(snaps) -> pd.DataFrame:
    prev = None
    prev_time = None
    out = []
    for _, fetched, _, frame in snaps:
        cur = frame.drop_duplicates(KEY, keep="last").set_index(KEY)[["team", *FIELDS]]
        if prev is None:
            prev = cur.iloc[0:0]
        events = []
        for idx in cur.index.difference(prev.index):
            events.append((idx, "first_appearance", FIELDS, cur.loc[idx], None))
        for idx in prev.index.difference(cur.index):
            events.append((idx, "removed", FIELDS, None, prev.loc[idx]))
        for idx in cur.index.intersection(prev.index):
            changed = [f for f in FIELDS if not _same(cur.at[idx, f], prev.at[idx, f])]
            if changed:
                events.append((idx, "change", changed, cur.loc[idx], prev.loc[idx]))
        for idx, event, changed, cur_row, prev_row in events:
            rec = dict(zip(KEY, idx, strict=True))
            src = cur_row if cur_row is not None else prev_row
            rec["team"] = src["team"]
            rec["observed_at"] = fetched
            rec["previous_snapshot_at"] = prev_time
            rec["event"] = event
            rec["changed_fields"] = ",".join(changed)
            for f in FIELDS:
                rec[f] = None if cur_row is None else _clean(cur_row[f])
                rec["prev_" + f] = None if prev_row is None else _clean(prev_row[f])
            out.append(rec)
        prev, prev_time = cur, fetched
    df = pd.DataFrame(out)
    df["baseline_snapshot"] = df["observed_at"] == min(s[1] for s in snaps)
    return df.sort_values(["observed_at", *KEY]).reset_index(drop=True)


def final_timing(rev: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    stream = rev[rev["event"] != "removed"].sort_values("observed_at")
    last = stream.groupby(KEY).tail(1)
    last = last[last["report_status"].isin(GAME_STATUS)]
    marks = stream[
        stream["changed_fields"].str.contains("report_status")
        | (stream["event"] == "first_appearance")
    ]
    marks = marks.groupby(KEY).tail(1).set_index(KEY)["observed_at"]
    last = last.drop(columns="observed_at").join(marks.rename("final_status_at"), on=KEY)
    m = last.merge(games, on=["season", "week", "team"], how="left")
    ko_et = m["kickoff"].dt.tz_convert(ET)
    days_back = (ko_et.dt.weekday - 4) % 7
    day = ko_et.dt.tz_localize(None).dt.normalize() - pd.to_timedelta(days_back, unit="D")
    fri = day.dt.tz_localize(ET) + pd.Timedelta(hours=16)
    sat = (day + pd.Timedelta(days=1)).dt.tz_localize(ET)
    m["fri_cut"] = fri.dt.tz_convert("UTC")
    m["sat_cut"] = sat.dt.tz_convert("UTC")
    m["thursday_game"] = ko_et.dt.weekday == 3
    return m


def games_by_team(path: Path, season: int) -> pd.DataFrame:
    g = pd.read_parquet(path, columns=["season", "week", "home_team", "away_team", "kickoff"])
    g = g[g["season"] == season]
    cols = ["season", "week", "team", "kickoff"]
    h = g.rename(columns={"home_team": "team"})[cols]
    a = g.rename(columns={"away_team": "team"})[cols]
    return pd.concat([h, a], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument(
        "--games", type=Path, default=ROOT / "data" / "processed" / "game_features.parquet"
    )
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--timing-week", type=int, default=8)
    args = ap.parse_args()
    snaps = load_snapshots(args.season)
    times = [s[1] for s in snaps]
    rev = revisions(snaps)
    nxt = dict(pairwise(times))
    prev_t = rev["previous_snapshot_at"]
    assert rev["observed_at"].isin(times).all()
    assert (prev_t.isna() | (rev["observed_at"] == prev_t.map(nxt))).all()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rev.to_parquet(args.out, index=False)
    manifest = {
        "output": args.out.name,
        "season": args.season,
        "rows": len(rev),
        "licence": "CC-BY 4.0 nflverse",
        "snapshots": [
            {"dir": s[0], "fetched_utc": s[1].isoformat(), "sha256": s[2], "rows_season": len(s[3])}
            for s in snaps
        ],
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    print("snapshots", len(snaps), "revision rows", len(rev))
    print(rev.groupby(["week", "event"]).size().unstack(fill_value=0).to_string())
    print("baseline rows", int(rev["baseline_snapshot"].sum()))
    t = final_timing(rev, games_by_team(args.games, args.season))
    print("unmatched game", int(t["kickoff"].isna().sum()))
    t = t.dropna(subset=["kickoff"]).copy()
    t["pre_fri"] = t["final_status_at"] < t["fri_cut"]
    t["pre_sat"] = t["final_status_at"] < t["sat_cut"]
    t["pre_ko"] = t["final_status_at"] < t["kickoff"]
    nt = t[~t["thursday_game"]]
    share = (
        nt.groupby("week")
        .agg(
            n=("pre_ko", "size"),
            pre_fri=("pre_fri", "mean"),
            pre_sat=("pre_sat", "mean"),
            pre_ko=("pre_ko", "mean"),
        )
        .round(3)
    )
    print(share.to_string())
    wk = t[t["week"] == args.timing_week].sort_values("final_status_at").copy()
    wk["hours_before_kickoff"] = (wk["kickoff"] - wk["final_status_at"]).dt.total_seconds() / 3600
    cols = [
        "team",
        "gsis_id",
        "report_status",
        "final_status_at",
        "kickoff",
        "hours_before_kickoff",
    ]
    print(wk[cols].head(12).to_string())


if __name__ == "__main__":
    main()
