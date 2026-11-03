import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="tests/scratch/sim12")
ap.add_argument("--legacy", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
R = "F:/Repos/nfl_py3/data/"
PART = R + "players/participation/raw/20260813T131635Z/season={s}/participation.parquet"
PBP = R + "pbp/raw/20260925T202544Z/season={s}/plays.parquet"
ros = pd.read_parquet(
    sorted(glob.glob(R + "players/raw/*/weekly_rosters.parquet"))[-1],
    columns=["season", "position", "gsis_id", "week"],
)
ros = (
    ros.dropna(subset=["gsis_id"])
    .sort_values("week")
    .groupby(["season", "gsis_id"])
    .position.last()
)
PTS = {"Touchdown": 7.0, "Field goal": 3.0}
OLP = {"T", "G", "C", "OL"}
DBP = {"CB", "DB"}
SCRIM = {"pass", "run"}
FRACS = (0.25, 0.5, 0.75)
STR = [(0, 0.2), (0.2, 0.8), (0.8, 1.01)]


def stratum(w):
    for i, (a, b) in enumerate(STR):
        if a <= w < b:
            return i
    return 1


def window(g, lo, hi):
    m = (g.dr >= lo) & (g.dr <= hi)
    if not m.any():
        return None
    s = g[m]
    d = s.groupby("dr").pts.first()
    return s.epa.mean(), d.mean(), len(d)


events, controls, haz = [], [], []
for s in range(2016, 2026):
    pt = pd.read_parquet(
        PART.format(s=s),
        columns=[
            "game_id",
            "play_id",
            "possession_team",
            "offense_players",
            "defense_players",
            "n_offense",
        ],
    )
    pb = pd.read_parquet(PBP.format(s=s))
    pb = pb[(pb.season_type == "REG") & pb.posteam.notna() & (pb.qtr <= 4)]
    d = pb.merge(pt, on=["game_id", "play_id"])
    d = d[(d.n_offense > 0)].sort_values(["game_id", "play_id"])
    d["pts"] = d.fixed_drive_result.map(PTS).fillna(0.0)
    d["ok"] = d.epa.notna() & (d.qb_kneel != 1) & (d.qb_spike != 1)
    snaps = {}
    for gid, g in d.groupby("game_id", sort=False):
        g = g.reset_index(drop=True)
        wk = int(g.week.iloc[0])
        for side in ("off", "def"):
            for team in (g.home_team.iloc[0], g.away_team.iloc[0]):
                on = (g.posteam == team) if side == "off" else (g.defteam == team)
                tg = g[on]
                if len(tg) < 20:
                    continue
                wp = tg.wp if side == "off" else 1 - tg.wp
                t = pd.DataFrame(
                    {
                        "dr": tg.fixed_drive.values,
                        "q": tg.qtr.values,
                        "gsr": tg.game_seconds_remaining.values,
                        "epa": np.where(tg.ok, tg.epa, np.nan),
                        "pts": tg.pts.values,
                        "wp": wp.values,
                    }
                )
                col = tg.offense_players if side == "off" else tg.defense_players
                scr = np.ones(len(tg), bool) if args.legacy else tg.play_type.isin(SCRIM).values
                sets = [x.split(";") if (x and k) else [] for x, k in zip(col, scr, strict=False)]
                last, first = {}, {}
                for i, st in enumerate(sets):
                    for p in st:
                        last[p] = i
                        first.setdefault(p, i)
                pos = {p: ros.get((s, p), "?") for p in last}
                snaps[(gid, side, team)] = (t, sets, last, first, pos, wk)
    cum = {}
    for (gid, side, team), (t, sets, last, _first, pos, _wk) in sorted(
        snaps.items(), key=lambda kv: kv[1][5]
    ):
        key = (side, team)
        prior = cum.get(key, {})
        ntot = prior.get("_n", 0)
        slots = []
        f0 = 0
        if side == "off":
            f0 = next((i for i, x in enumerate(sets) if x), 0)
            qbs = [p for p in sets[f0] if pos.get(p) == "QB"]
            if qbs:
                slots.append(("QB", qbs[0]))
            ol = [p for p in sets[f0] if pos.get(p) in OLP]
            if len(ol) == 5:
                slots += [("OL", p) for p in ol]
            if ntot > 0:
                w = [(prior.get(p, 0), p) for p in last if pos.get(p) == "WR"]
                if w and max(w)[0] > 0:
                    slots.append(("WR1", max(w)[1]))
        else:
            if ntot > 0:
                c = [
                    (prior.get(p, 0), p)
                    for p in last
                    if pos.get(p) in (({"CB"}) if args.legacy else DBP)
                ]
                if c and max(c)[0] > 0:
                    slots.append(("CB1" if args.legacy else "DB1", max(c)[1]))
        q4 = (t.q == 4).values
        L4 = np.flatnonzero(q4)
        hasq4 = len(L4) > 0
        exited_any = {}
        for grp, p in slots:
            if p not in last:
                continue
            L = last[p]
            ex = hasq4 and t.q.values[L] <= 3 and L4[0] > L and len(t) > L + 1
            haz.append((s, gid, team, grp, p, int(ex), int(t.q.values[L]) if ex else 0))
            if ex:
                exited_any[grp] = True
                nxt = L + 1
                dl = t.dr.values[L]
                pre = window(t, -1, dl)
                post = window(t, dl + 1, 99)
                if pre and post:
                    events.append(
                        (
                            s,
                            gid,
                            team,
                            grp,
                            int(t.q.values[nxt]),
                            stratum(float(t.wp.values[nxt])),
                            post[0] - pre[0],
                            post[1] - pre[1],
                            post[2],
                            pre[2],
                        )
                    )
        for grp in {g for g, _ in slots}:
            if grp in exited_any:
                continue
            for qq in (1, 2, 3):
                for fr in FRACS[1:2] if args.legacy else FRACS:
                    mid = 3600 - 900 * (qq - 1) - 900 * fr
                    idx = np.flatnonzero(t.gsr.values >= mid)
                    if len(idx) == 0 or idx[-1] + 1 >= len(t):
                        continue
                    dl = t.dr.values[idx[-1]]
                    pre = window(t, -1, dl)
                    post = window(t, dl + 1, 99)
                    if pre and post:
                        controls.append(
                            (
                                s,
                                gid,
                                team,
                                grp,
                                qq,
                                stratum(float(t.wp.values[idx[-1] + 1])),
                                post[0] - pre[0],
                                post[1] - pre[1],
                                post[2],
                                pre[2],
                                fr,
                            )
                        )
        c = cum.setdefault(key, {})
        for st in sets:
            for p in st:
                c[p] = c.get(p, 0) + 1
        c["_n"] = c.get("_n", 0) + len(sets)
    print(s, len(events), len(controls), len(haz), file=sys.stderr, flush=True)
cn = ["season", "game", "team", "grp", "q", "st", "d_epa", "d_ppd", "n_post", "n_pre"]
pd.DataFrame(events, columns=cn).to_parquet(args.out + "/ev.parquet")
pd.DataFrame(controls, columns=[*cn, "frac"]).to_parquet(args.out + "/ct.parquet")
pd.DataFrame(haz, columns=["season", "game", "team", "grp", "player", "exit", "q"]).to_parquet(
    args.out + "/hz.parquet"
)
