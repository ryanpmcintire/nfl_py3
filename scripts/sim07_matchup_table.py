import argparse
import json

import numpy as np
import pandas as pd
import sim07_ftn_loso as lo
import sim07_personnel as m

FROZEN = m.OUT / "ftn_frozen_2022_2025.json"
DRIVE_MAP = m.OUT / "personnel_drive_map.json"
TABLE = m.OUT / "matchup_table.parquet"
SUMMARY = m.OUT / "matchup_table_summary.json"
CHECK = m.OUT / "matchup_table_leakage.json"


def load():
    fz = json.loads(FROZEN.read_text())["body"]
    dm = json.loads(DRIVE_MAP.read_text())
    return fz, dm


def counts(d, team_col, mix_col, levels):
    c = d.groupby(["season", "week", team_col, mix_col]).size().unstack(mix_col, fill_value=0)
    c = c.reindex(columns=levels, fill_value=0)
    return c


def prior(c):
    out = {}
    for (season, team), g in c.groupby(level=[0, 2]):
        g = g.droplevel([0, 2]).sort_index()
        cum = g.cumsum().shift(1, fill_value=0)
        for week, row in cum.iterrows():
            out[(int(season), int(week), team)] = row.to_numpy(float)
    return out


def build(d, fz, dm):
    ogs = sorted({c.split("|")[0] for c in fz["cells"]})
    dps = sorted({c.split("|")[1] for c in fz["cells"]}, key=int)
    dev = np.zeros((len(ogs), len(dps)))
    for c, v in zip(fz["cells"], fz["frozen_deviation"], strict=True):
        a, b = c.split("|")
        dev[ogs.index(a), dps.index(b)] = v
    off = prior(counts(d, "posteam", "og", ogs))
    dfn = prior(counts(d, "defteam", "dp", dps))
    games = d[["season", "week", "posteam", "defteam"]].drop_duplicates()
    ppd = dm["plays_per_drive"]
    rows = []
    for season, week, t, o in games.itertuples(index=False):
        a = off.get((int(season), int(week), t))
        b = dfn.get((int(season), int(week), o))
        s = 0.0
        if a is not None and b is not None and a.sum() > 0 and b.sum() > 0:
            s = float((np.outer(a / a.sum(), b / b.sum()) * dev).sum())
        rows.append(
            (int(season), int(week), t, o, s, dm["tau"]["td"] * ppd * s, dm["tau"]["fg"] * ppd * s)
        )
    return pd.DataFrame(
        rows, columns=["season", "week", "team", "opp", "s_play", "td_shift", "fg_shift"]
    ).sort_values(["season", "week", "team", "opp"], ignore_index=True)


def make():
    fz, dm = load()
    d = lo.build()
    t = build(d, fz, dm)
    t.to_parquet(TABLE, index=False)
    live = t[t.s_play != 0.0]
    SUMMARY.write_text(
        json.dumps(
            {
                "rows": len(t),
                "rows_nonzero": len(live),
                "plays_per_drive": dm["plays_per_drive"],
                "tau": dm["tau"],
                "frozen_sha256": json.loads(FROZEN.read_text())["sha256"],
                "sd_td_shift_all": float(t.td_shift.std()),
                "sd_td_shift_nonzero": float(live.td_shift.std()),
                "sd_fg_shift_nonzero": float(live.fg_shift.std()),
                "sd_s_play_nonzero": float(live.s_play.std()),
                "mean_td_shift_nonzero": float(live.td_shift.mean()),
                "max_abs_td_shift": float(live.td_shift.abs().max()),
                "by_season_sd_td": {
                    int(k): float(v) for k, v in live.groupby("season").td_shift.std().items()
                },
            },
            indent=1,
        )
    )
    print(SUMMARY.read_text())


def check(season, week):
    fz, dm = load()
    d = lo.build()
    full = build(d, fz, dm)
    cut = d[(d.season < season) | ((d.season == season) & (d.week <= week))]
    part = build(cut, fz, dm)
    key = ["season", "week", "team", "opp"]
    j = part.merge(full, on=key, suffixes=("_cut", "_full"))
    j = j[(j.season < season) | ((j.season == season) & (j.week <= week + 1))]
    same = np.array_equal(j.td_shift_cut.to_numpy(), j.td_shift_full.to_numpy())
    nxt = j[(j.season == season) & (j.week == week + 1)]
    out = {
        "cut_after": [season, week],
        "team_games_compared": len(j),
        "team_games_nonzero": int((j.s_play_full != 0).sum()),
        "team_games_first_week_after_cut": len(nxt),
        "max_abs_diff_td": float((j.td_shift_cut - j.td_shift_full).abs().max()),
        "identical": bool(same),
        "rows_in_cut_table": len(part),
        "rows_in_full_table": len(full),
    }
    CHECK.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "check"])
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--week", type=int, default=9)
    a = ap.parse_args()
    make() if a.mode == "make" else check(a.season, a.week)


if __name__ == "__main__":
    main()
