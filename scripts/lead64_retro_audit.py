import glob
import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from nfl_ats import injury_headlines as ih

R = Path("F:/Repos/nfl_py3")
DES = ["Out", "Doubtful", "Questionable"]
rev = pd.read_parquet(R / "data/processed/injury_revisions_2026.parquet")
with open(R / "data/processed/injury_revisions_2026.manifest.json") as fh:
    man = json.load(fh)
names = {}
for s in man["snapshots"]:
    p = R / "data/raw/nflverse_injuries" / s["dir"] / "injuries.parquet"
    d = pd.read_parquet(p, columns=["season", "gsis_id", "full_name", "team"])
    d = d[d.season == 2026]
    for g, n, t in zip(d.gsis_id, d.full_name, d.team, strict=True):
        names[g] = (n, t)
rev["name"] = rev.gsis_id.map(lambda g: names.get(g, ("", ""))[0])
cards = []
for p in sorted(glob.glob(str(R / "artifacts/margin_predictions/2026-week-0[1-8]-*"))):
    b = os.path.basename(p)
    wk = int(b.split("-")[2])
    ts = pd.to_datetime(b.split("-")[3], format="%Y%m%dT%H%M%SZ", utc=True)
    try:
        with open(p + "/lineage.json") as fh:
            lin = json.load(fh)
    except Exception:
        continue
    inj = [
        f["lineage"]["effective_timestamp"]
        for f in lin["fields"]
        if f["card_field"] == "model_input:player_injuries"
    ]
    cards.append(
        {
            "week": wk,
            "created": ts,
            "inj": pd.to_datetime(inj[0], utc=True) if inj else pd.NaT,
            "dir": b,
        }
    )
cards = pd.DataFrame(cards)
ko = {}
for wk in range(1, 9):
    c = cards[cards.week == wk].iloc[-1]
    k = pd.read_csv(
        R / "artifacts/margin_predictions" / c.dir / "predictions.csv",
        usecols=["game_id", "away_team", "home_team", "kickoff"],
    ).drop_duplicates("game_id")
    k["kickoff"] = pd.to_datetime(k.kickoff, utc=True)
    k["week"] = wk
    ko[wk] = k
games = pd.concat(ko.values())


def sun4(k):
    e = k.tz_convert("America/New_York")
    sun = (e + pd.Timedelta(days=(6 - e.weekday()) % 7)).normalize() + pd.Timedelta(hours=16)
    return (
        sun.tz_localize(None).tz_localize("America/New_York").tz_convert("UTC")
        if sun.tzinfo is None
        else sun.tz_convert("UTC")
    )


def dl(row):
    e = row.kickoff.tz_convert("America/New_York")
    sun = pd.Timestamp(e.date()) + pd.Timedelta(days=(6 - e.weekday()) % 7, hours=16)
    sun = sun.tz_localize("America/New_York").tz_convert("UTC")
    return min(row.kickoff, sun)


games["deadline"] = games.apply(dl, axis=1)
caps = ih.load_all_captures(R / "data", date(2026, 8, 1))
fs = caps.groupby("url")["_capture_ts"].min().rename("first_seen")
caps = caps.drop_duplicates("url").drop(columns=["_capture_ts"]).merge(fs, on="url")
idx = {"name_to_teams": {}, "stripped_to_teams": {}, "snapshot_id": ""}
for _, (n, t) in names.items():
    idx["name_to_teams"].setdefault(ih._normalize_name(n), set()).add(t)
    idx["stripped_to_teams"].setdefault(ih._normalize_name_stripped(n), set()).add(t)
hl = ih.parse_all_headlines(caps, idx)
hl = hl[hl.parsed.astype(bool) & hl.designation.isin(["out", "doubtful", "questionable"])].copy()
hl["key"] = hl.player_name.map(lambda s: ih._normalize_name(str(s))) + "|" + hl.team.astype(str)
rows = []
miss = []
for wk in range(1, 9):
    g = games[
        (games.week == wk) & (games.kickoff.dt.tz_convert("America/New_York").dt.weekday == 6)
    ]
    teams = set(g.away_team) | set(g.home_team)
    dls = g.set_index("home_team").deadline
    wdl = g.deadline.min()
    sun = g.kickoff.min().tz_convert("America/New_York").normalize()
    fri = (sun - pd.Timedelta(days=2) + pd.Timedelta(hours=20)).tz_convert("UTC")
    sat = (sun - pd.Timedelta(days=1) + pd.Timedelta(hours=20)).tz_convert("UTC")
    gd = {}
    for _, r in g.iterrows():
        gd[r.home_team] = r.deadline
        gd[r.away_team] = r.deadline
    w = rev[(rev.week == wk) & rev.team.isin(teams)]

    def state(t, w=w):
        x = w[w.observed_at <= t].sort_values("observed_at").groupby("gsis_id").tail(1)
        return x[x.report_status.isin(DES)].set_index("gsis_id")

    final = state(wdl)
    final = final[[gd[t] > pd.Timestamp.min.tz_localize("UTC") for t in final.team]]
    first = (
        w[w.report_status.isin(DES)]
        .sort_values("observed_at")
        .groupby(["gsis_id", "report_status"])
        .observed_at.min()
    )
    for lab, t in [("Fri20ET", fri), ("Sat20ET", sat)]:
        st = state(t)
        agree = [
            (gid, st.loc[gid, "report_status"] == final.loc[gid, "report_status"])
            for gid in final.index
            if gid in st.index
        ]
        vis = sum(a for _, a in agree)
        c = cards[(cards.week == wk) & (cards.created <= t)]
        cd = c.iloc[-1] if len(c) else None
        cinj = cd.inj if cd is not None else pd.NaT
        reach = sum(
            1
            for gid in final.index
            if pd.notna(cinj) and first.get((gid, final.loc[gid, "report_status"]), pd.NaT) <= cinj
        )
        h = hl[(hl.first_seen <= t) & (hl.first_seen >= sun - pd.Timedelta(days=7))]
        hk = set(zip(h.key, h.designation, strict=True))
        hv = sum(
            1
            for gid in final.index
            if (
                ih._normalize_name(final.loc[gid, "name"]) + "|" + final.loc[gid, "team"],
                final.loc[gid, "report_status"].lower(),
            )
            in hk
        )
        rows.append(
            {
                "week": wk,
                "asof": lab,
                "final_n": len(final),
                "visible": vis,
                "headline_visible": hv,
                "card_reach_asof": reach,
                "card_inj_ts": str(cinj),
            }
        )
    lagr = []
    nfin = 0
    cds = set()
    for d, grp in g.groupby("deadline"):
        tm = set(grp.away_team) | set(grp.home_team)
        cd = cards[(cards.week == wk) & (cards.created <= d)].iloc[-1]
        cds.add(cd.dir)
        x = (
            w[(w.observed_at <= d) & w.team.isin(tm)]
            .sort_values("observed_at")
            .groupby("gsis_id")
            .tail(1)
        )
        x = x[x.report_status.isin(DES)].set_index("gsis_id")
        for gid in x.index:
            f = first.get((gid, x.loc[gid, "report_status"]))
            lagr.append(
                (
                    gid,
                    x.loc[gid, "name"],
                    x.loc[gid, "team"],
                    x.loc[gid, "report_status"],
                    f,
                    f <= cd.inj,
                    (cd.inj - f).total_seconds() / 3600,
                    cd.inj,
                    cd.created,
                    d,
                )
            )
    rows.append(
        {
            "week": wk,
            "asof": "game_deadline_card",
            "final_n": len(lagr),
            "card_reach_asof": sum(x[5] for x in lagr),
            "lag_med_h": pd.Series([x[6] for x in lagr]).median(),
            "lag_max_h": max(x[6] for x in lagr),
            "n_missed": sum(not x[5] for x in lagr),
            "cards": ";".join(sorted(cds)),
        }
    )
    for x in lagr:
        if not x[5]:
            miss.append(
                {
                    "week": wk,
                    "name": x[1],
                    "team": x[2],
                    "status": x[3],
                    "first_obs": str(x[4]),
                    "card_inj": str(x[7]),
                    "card_created": str(x[8]),
                    "game_deadline": str(x[9]),
                }
            )
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
out = pd.DataFrame(rows)
print(out.to_string())
m = pd.DataFrame(miss)
print(len(m))
if len(m):
    print(m.groupby("week").size())
    print(m.groupby(["week", "team"]).size().sort_values(ascending=False).head(15))
out.to_csv(sys.argv[1] + "/l64_table.csv", index=False)
m.to_csv(sys.argv[1] + "/l64_miss.csv", index=False)
print(len(hl), hl.first_seen.min(), hl.first_seen.max())
