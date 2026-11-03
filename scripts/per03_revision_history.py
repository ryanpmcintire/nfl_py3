from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "injury_revisions_2026.parquet"
MARKET = ROOT / "data" / "market" / "raw"
OUT = ROOT / "artifacts" / "per03" / "revisions_2026.parquet"
ET = "America/New_York"
FIELD_GROUP = {
    "report_status": "report_status",
    "practice_status": "practice_status",
    "practice_primary_injury": "practice_detail",
    "practice_secondary_injury": "practice_detail",
    "report_primary_injury": "body_part",
    "report_secondary_injury": "body_part",
}
SEVERITY = {"Out": 3, "Doubtful": 2, "Questionable": 1}
DESIGNATIONS = ["Out", "Doubtful", "Questionable"]
TUESDAY_LINE_HOUR_ET = 12
SUNDAY_DEADLINE_HOUR_ET = 16
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def long_changes(src: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field, group in FIELD_GROUP.items():
        old = src[f"prev_{field}"]
        new = src[field]
        differ = ~((old == new) | (old.isna() & new.isna()))
        part = src.loc[differ, ["season", "week", "team", "gsis_id", "event", "observed_at"]].copy()
        part["field"] = field
        part["field_group"] = group
        part["old_value"] = old[differ]
        part["new_value"] = new[differ]
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"observed_at": "first_seen_utc"})
    et = out["first_seen_utc"].dt.tz_convert(ET)
    out["first_seen_dow_et"] = et.dt.dayofweek.map(dict(enumerate(DAYS)))
    return out.sort_values(["week", "team", "gsis_id", "first_seen_utc", "field"]).reset_index(
        drop=True
    )


def game_table() -> pd.DataFrame:
    frames = []
    cols = [
        "observed_at_utc",
        "nflverse_game_id",
        "commence_time_utc",
        "bookmaker_key",
        "home_spread_line",
        "home_team",
        "away_team",
    ]
    for d in sorted(MARKET.iterdir()):
        path = d / "quotes.parquet"
        if not path.is_file() or not d.name.startswith("2026") or d.name.endswith("-ncaaf"):
            continue
        q = pd.read_parquet(path)
        q = q[
            (q["market"] == "spreads")
            & q["home_spread_line"].notna()
            & q["nflverse_game_id"].notna()
        ]
        q = q[q["nflverse_game_id"].str.startswith("2026_")]
        frames.append(q[cols])
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["observed_at_utc", "nflverse_game_id", "bookmaker_key"]
    )


def game_lines(quotes: pd.DataFrame) -> pd.DataFrame:
    med = (
        quotes.groupby(["nflverse_game_id", "observed_at_utc"])
        .agg(
            line=("home_spread_line", "median"),
            kickoff=("commence_time_utc", "first"),
            home=("home_team", "first"),
            away=("away_team", "first"),
        )
        .reset_index()
    )
    out = []
    for gid, g in med.groupby("nflverse_game_id"):
        g = g.sort_values("observed_at_utc")
        ko = g["kickoff"].iloc[0]
        ko_et = ko.tz_convert(ET)
        sunday = (ko_et - pd.Timedelta(days=(ko_et.dayofweek + 1) % 7)).normalize()
        deadline = min(ko, (sunday + pd.Timedelta(hours=SUNDAY_DEADLINE_HOUR_ET)).tz_convert("UTC"))
        tuesday = (
            sunday - pd.Timedelta(days=5) + pd.Timedelta(hours=TUESDAY_LINE_HOUR_ET)
        ).tz_convert("UTC")
        after = g[g["observed_at_utc"] >= tuesday]
        before = g[g["observed_at_utc"] <= deadline]
        if after.empty or before.empty:
            continue
        o, c = after.iloc[0], before.iloc[-1]
        if o["observed_at_utc"] >= c["observed_at_utc"]:
            continue
        out.append(
            {
                "game_id": gid,
                "season": int(gid[:4]),
                "week": int(gid[5:7]),
                "home": g["home"].iloc[0],
                "away": g["away"].iloc[0],
                "kickoff": ko,
                "open_at": o["observed_at_utc"],
                "close_at": c["observed_at_utc"],
                "open_line": o["line"],
                "close_line": c["line"],
            }
        )
    return pd.DataFrame(out)


def designation_table(src: pd.DataFrame, kick: pd.DataFrame):
    s = src[src["report_status"].notna() | src["prev_report_status"].notna()].copy()
    s = s.merge(kick, on=["season", "week", "team"], how="inner")
    s = s.sort_values(["week", "team", "gsis_id", "observed_at"])
    first = s.groupby(["week", "team", "gsis_id"]).head(1)
    first = first[
        first["report_status"].isin(DESIGNATIONS) & (first["observed_at"] < first["kickoff"])
    ]
    later = s[
        (s["event"] == "change")
        & (s["observed_at"] < s["kickoff"])
        & s["prev_report_status"].isin(DESIGNATIONS)
    ]
    later = later[later["report_status"] != later["prev_report_status"]]
    return first, later


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-week", type=int, default=8)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    src = pd.read_parquet(SRC)
    rev = long_changes(src)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rev.to_parquet(OUT, index=False)
    print("rows", len(rev), "player-weeks", rev[["week", "gsis_id"]].drop_duplicates().shape[0])
    print(rev.groupby("event").size().to_string())
    chg = rev[rev["event"] == "change"]
    print(
        chg.pivot_table(
            index="week", columns="field_group", values="gsis_id", aggfunc="count", fill_value=0
        ).to_string()
    )
    print(
        chg.pivot_table(
            index="week",
            columns="first_seen_dow_et",
            values="gsis_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=DAYS)
        .to_string()
    )
    print(
        chg.pivot_table(
            index="field_group",
            columns="first_seen_dow_et",
            values="gsis_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=DAYS)
        .to_string()
    )

    lines = game_lines(game_table())
    lines = lines[lines["week"] <= args.max_week]
    print("games with lines", len(lines))
    home = lines.assign(team=lines["home"], sign=1.0)
    away = lines.assign(team=lines["away"], sign=-1.0)
    kick = pd.concat([home, away], ignore_index=True)
    kick["team_move"] = kick["sign"] * (kick["close_line"] - kick["open_line"])
    kick = kick[["season", "week", "team", "kickoff", "team_move"]]
    first, later = designation_table(src[src["week"] <= args.max_week], kick)
    rows = []
    for d in DESIGNATIONS:
        f = first[first["report_status"] == d]
        ch = later.merge(f[["week", "team", "gsis_id"]], on=["week", "team", "gsis_id"])
        ch = ch[ch["prev_report_status"] == d]
        n = len(f)
        k = ch[["week", "gsis_id"]].drop_duplicates().shape[0]
        to = ch["report_status"].fillna("cleared").value_counts().to_dict()
        rows.append(
            {
                "designation": d,
                "players": n,
                "changed_before_kickoff": k,
                "rate": round(k / n, 4) if n else None,
                "to": to,
            }
        )
    print(json.dumps(rows))
    ca = later.copy()
    ca["delta"] = ca["report_status"].map(SEVERITY).fillna(0) - ca["prev_report_status"].map(
        SEVERITY
    ).fillna(0)
    ca["dow"] = ca["observed_at"].dt.tz_convert(ET).dt.dayofweek.map(dict(enumerate(DAYS)))
    print("late changes by day", ca["dow"].value_counts().to_dict())
    print(
        "late changes direction",
        {"worse": int((ca["delta"] > 0).sum()), "better": int((ca["delta"] < 0).sum())},
    )
    net = ca.groupby(["week", "team"])["delta"].sum().rename("net").reset_index()
    tg = net.merge(kick, on=["week", "team"], how="inner")
    tg = tg[(tg["net"] != 0) & (tg["team_move"] != 0)]
    print("team-games with late net change and nonzero line move", len(tg))
    nets = np.sign(tg["net"].to_numpy())
    obs = float((nets == np.sign(tg["team_move"].to_numpy())).mean()) if len(tg) else float("nan")
    allm = kick[kick["team_move"] != 0]
    weeks = tg["week"].to_numpy()
    pool = {w: allm.loc[allm["week"] == w, "team_move"].to_numpy() for w in np.unique(weeks)}
    rng = np.random.default_rng(args.seed)
    sims = np.empty(args.perms)
    for i in range(args.perms):
        mv = np.array([rng.choice(pool[w]) for w in weeks])
        sims[i] = (nets == np.sign(mv)).mean()
    print(
        json.dumps(
            {
                "agree": round(obs * len(tg)),
                "n": len(tg),
                "rate": round(obs, 4),
                "null_mean": round(float(sims.mean()), 4),
                "p_ge": round(float((sims >= obs).mean()), 4),
            }
        )
    )
    print(tg[["week", "team", "net", "team_move"]].to_string())


if __name__ == "__main__":
    main()
