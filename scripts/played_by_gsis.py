import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
ART = ROOT / "artifacts" / "per03"
RERUN = ART / "all_rows"
RERUN_COMPLETE = ART / "all_rows_complete"
SKILL = ("QB", "RB", "WR", "TE")
PBP_2026_LAST_WEEK = 7
FIRST_SEASON = 2013
LAST_SEASON = 2026
PARTICIPATION_FIRST = 2016
FULL_PBP_LAST = 2017
FULL_PBP_DIR = "data/raw/nflverse_pbp_full/20261103T131922Z"

GROUPS = {
    "QB": "QB",
    "RB": "RB",
    "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "T": "OL",
    "G": "OL",
    "C": "OL",
    "OL": "OL",
    "DE": "DL",
    "DT": "DL",
    "DL": "DL",
    "LB": "LB",
    "CB": "DB",
    "S": "DB",
    "DB": "DB",
    "K": "ST",
    "P": "ST",
    "LS": "ST",
}


def latest(pattern):
    return sorted(glob.glob(str(ROOT / pattern)))[-1]


def pbp_games():
    root = Path(latest("data/pbp/raw/*"))
    out = []
    for s in range(FIRST_SEASON, LAST_SEASON + 1):
        d = pd.read_parquet(
            root / f"season={s}" / "plays.parquet",
            columns=["game_id", "season", "week", "season_type", "home_team", "away_team"],
        )
        d = d[d["season_type"] == "REG"].drop_duplicates("game_id")
        out.append(d)
    g = pd.concat(out)
    h = g[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "team"})
    a = g[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "team"})
    return pd.concat([h, a])


def id_appearances():
    narrow = Path(latest("data/pbp/raw/*"))
    full = ROOT / FULL_PBP_DIR
    frames = []
    for s in range(FIRST_SEASON, LAST_SEASON + 1):
        paths = [narrow / f"season={s}" / "plays.parquet"]
        if s <= FULL_PBP_LAST:
            paths.append(full / f"pbp_{s}.parquet")
        for f in paths:
            names = pq.ParquetFile(f).schema.names
            cols = [c for c in names if c.endswith("_player_id")]
            d = pd.read_parquet(f, columns=["season", "week", "season_type", *cols])
            d = d[d["season_type"] == "REG"]
            m = d.melt(id_vars=["season", "week"], value_vars=cols, value_name="gsis_id")
            frames.append(m.dropna(subset=["gsis_id"])[["season", "week", "gsis_id"]])
    a = pd.concat(frames).drop_duplicates()
    a["pbp_id"] = 1
    return a


def participation_appearances(games):
    root = Path(latest("data/players/participation/raw/*"))
    wk = games.drop_duplicates("game_id").set_index("game_id")["week"]
    frames = []
    have = []
    for s in range(PARTICIPATION_FIRST, LAST_SEASON + 1):
        f = root / f"season={s}" / "participation.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(
            f,
            columns=["game_id", "season", "players_on_play", "offense_players", "defense_players"],
        )
        d["listed"] = (
            d["offense_players"].fillna("")
            + ";"
            + d["defense_players"].fillna("")
            + ";"
            + d["players_on_play"].fillna("")
        )
        d = d[d["listed"].str.contains("00-", regex=False)]
        d["week"] = d["game_id"].map(wk)
        d = d.dropna(subset=["week"])
        have.append(d[["game_id"]].drop_duplicates())
        e = d.assign(gsis_id=d["listed"].str.split(";")).explode("gsis_id")
        e = e[e["gsis_id"].str.startswith("00-", na=False)]
        frames.append(e[["season", "week", "gsis_id"]].drop_duplicates())
    a = pd.concat(frames).drop_duplicates()
    a["participation"] = 1
    return a, set(pd.concat(have)["game_id"])


def snap_frame():
    cols = [
        "season",
        "game_type",
        "week",
        "pfr_player_id",
        "team",
        "offense_snaps",
        "defense_snaps",
        "st_snaps",
    ]
    s = pd.read_parquet(latest("data/players/raw/*/snap_counts.parquet"), columns=cols)
    s = s[(s["game_type"] == "REG") & (s["season"] <= 2025)]
    f = pd.read_parquet(
        ROOT / "data" / "players" / "forward_snaps" / "snap_counts_2026_before_week_09.parquet",
        columns=cols,
    )
    f = f[(f["game_type"] == "REG") & (f["season"] == 2026)]
    s = pd.concat([s, f])
    s["total"] = s[["offense_snaps", "defense_snaps", "st_snaps"]].fillna(0).sum(axis=1)
    tw = s[["season", "team", "week"]].drop_duplicates().assign(tw_snap=True)
    pl = s.groupby(["season", "week", "pfr_player_id"], as_index=False).agg(snaps=("total", "sum"))
    return pl, tw


def injury_rows():
    inj = pd.read_parquet(latest("data/raw/nflverse_injuries/*/injuries.parquet"))
    inj = inj[(inj["game_type"] == "REG") & inj["season"].between(FIRST_SEASON, LAST_SEASON)]
    inj = inj.sort_values("date_modified").drop_duplicates(
        ["season", "week", "gsis_id"], keep="last"
    )
    inj["week"] = inj["week"].astype(int)
    ros = pd.read_parquet(
        latest("data/players/raw/*/weekly_rosters.parquet"),
        columns=["season", "gsis_id", "pfr_id"],
    ).dropna()
    ros = ros.drop_duplicates(["season", "gsis_id"], keep="last")
    return inj.merge(ros, on=["season", "gsis_id"], how="left")


def cmd_build(_):
    ART.mkdir(parents=True, exist_ok=True)
    games = pbp_games()
    games["week"] = games["week"].astype(int)
    part, part_games = participation_appearances(games)
    part["week"] = part["week"].astype(int)
    ids = id_appearances()
    ids["week"] = ids["week"].astype(int)
    pl, tw = snap_frame()
    games["has_part"] = games["game_id"].isin(part_games)
    gtw = games.groupby(["season", "week", "team"], as_index=False).agg(
        pbp_game=("game_id", "size"), has_part=("has_part", "max")
    )
    x = injury_rows()
    x = x.merge(gtw, on=["season", "week", "team"], how="left")
    x = x.merge(tw, on=["season", "team", "week"], how="left")
    x = x.merge(part, on=["season", "week", "gsis_id"], how="left")
    x = x.merge(ids, on=["season", "week", "gsis_id"], how="left")
    x = x.merge(
        pl,
        left_on=["season", "week", "pfr_id"],
        right_on=["season", "week", "pfr_player_id"],
        how="left",
    )
    in_snap_tw = x["tw_snap"].fillna(False).astype(bool)
    x["team_game"] = x["pbp_game"].notna() | in_snap_tw
    x["played_appear"] = x["participation"].fillna(0).astype(int) | x["pbp_id"].fillna(0).astype(
        int
    )
    x["played_snaps"] = np.where(
        x["pfr_id"].notna() & in_snap_tw,
        (x["snaps"].fillna(0) > 0).astype(float),
        np.nan,
    )
    x["played"] = np.maximum(x["played_appear"], x["played_snaps"].fillna(0)).astype(int)
    full_part = x["has_part"].fillna(False).astype(bool)
    x["source"] = np.where(
        full_part,
        "participation+pbp_ids",
        np.where(x["season"] <= FULL_PBP_LAST, "pbp_ids_full", "pbp_ids_skill"),
    )
    x["source"] = np.where(x["played_snaps"].notna(), x["source"] + "+snaps", x["source"])
    cols = [
        "season",
        "week",
        "team",
        "gsis_id",
        "position",
        "report_status",
        "practice_status",
        "pfr_id",
        "team_game",
        "played",
        "played_appear",
        "played_snaps",
        "source",
    ]
    x[cols].to_parquet(ART / "played_by_gsis.parquet", index=False)
    print(len(x), x["team_game"].mean(), x["played"].mean())


def agree(g):
    a = g["played_appear"].to_numpy()
    s = g["played_snaps"].to_numpy()
    u = g["played"].to_numpy()
    return {
        "rows": len(g),
        "snap_play_rate": float(s.mean()),
        "appear_rate": float(a.mean()),
        "agree_appear": float((a == s).mean()),
        "agree_union": float((u == s).mean()),
        "appear_given_snap": float(a[s == 1].mean()) if (s == 1).any() else None,
        "snap_given_appear": float(s[a == 1].mean()) if (a == 1).any() else None,
    }


def base_source(frame):
    return frame["source"].str.replace("+snaps", "", regex=False)


def cmd_validate(_):
    x = pd.read_parquet(ART / "played_by_gsis.parquet")
    x["pg"] = x["position"].map(GROUPS).fillna("other")
    g = x[x["team_game"]]
    res = {
        "rows": len(x),
        "team_game_rate": float(x["team_game"].mean()),
        "played_rate_team_game": float(g["played"].mean()),
    }
    res["coverage_by_season"] = {
        int(s): {
            "rows": len(d),
            "team_game": float(d["team_game"].mean()),
            "snap_crosswalk": float(d["played_snaps"].notna().mean()),
            "source": base_source(d).value_counts().to_dict(),
        }
        for s, d in x.groupby("season")
    }
    res["coverage_by_position"] = {
        p: {
            "rows": len(d),
            "snap_crosswalk": float(d["played_snaps"].notna().mean()),
            "played_rate": float(d[d["team_game"]]["played"].mean()),
            "appear_rate": float(d[d["team_game"]]["played_appear"].mean()),
        }
        for p, d in x.groupby("pg")
    }
    both = g[g["played_snaps"].notna()]
    res["agreement_overall"] = agree(both)
    res["agreement_by_season"] = {int(s): agree(d) for s, d in both.groupby("season")}
    res["agreement_by_position"] = {p: agree(d) for p, d in both.groupby("pg")}
    res["agreement_by_source"] = {s: agree(d) for s, d in both.groupby(base_source(both))}
    w = g[g["played_snaps"].notna()]
    wo = g[g["played_snaps"].isna()]
    res["with_vs_without_crosswalk"] = {
        "played_with": float(w["played"].mean()),
        "played_without": float(wo["played"].mean()),
        "appear_with": float(w["played_appear"].mean()),
        "appear_without": float(wo["played_appear"].mean()),
    }
    (ART / "validate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def complete_rows(new):
    pg = new["position"].map(GROUPS)
    skill_seen = pg.isin(SKILL) & (new["week"] <= PBP_2026_LAST_WEEK)
    in_part = new["source"].str.startswith("participation")
    return new[
        in_part | ((new["season"] == LAST_SEASON) & (new["played_snaps"].notna() | skill_seen))
    ]


def cmd_lead106(_):
    import shutil

    import lead106_injury_language as L

    out_dir = ART / "lead106_complete"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("population.parquet", "sample.parquet"):
        shutil.copy(L.ART / name, out_dir / name)
    shutil.copytree(L.ART / "raw", out_dir / "raw", dirs_exist_ok=True)
    new = complete_rows(pd.read_parquet(ART / "played_by_gsis.parquet").query("team_game"))
    orig = L.id_outcomes

    def patched(pop):
        x = orig(pop).drop(columns=["verified", "played"])
        n = new[new["season"] == L.SEASON][["week", "gsis_id", "played"]]
        x = x.merge(n, on=["week", "gsis_id"], how="left")
        x["verified"] = x["played"].notna()
        x["played"] = x["played"].fillna(0).astype(int)
        return x

    L.ART = out_dir
    L.id_outcomes = patched
    L.cmd_ids(None)


def cmd_rerun(a):
    import lead106_injury_language as L
    import per16_team_designation_habit as H
    import per16_unit2_confounds as U

    out_dir = RERUN_COMPLETE if a.complete else RERUN
    out_dir.mkdir(parents=True, exist_ok=True)
    H.ART = out_dir
    U.ART = out_dir
    new = pd.read_parquet(ART / "played_by_gsis.parquet")
    new = new[new["team_game"]]
    if a.complete:
        new = complete_rows(new)
        H.FIRST_SEASON = PARTICIPATION_FIRST
        H.DESIGNATIONS = ("Questionable", "Doubtful")
    old = pd.read_parquet(H.ROOT / "artifacts" / "per16" / "history.parquet")
    h = old.drop(columns=["played", "verified"]).merge(
        new[["season", "week", "gsis_id", "played"]],
        on=["season", "week", "gsis_id"],
        how="inner",
    )
    h["verified"] = True
    H.history = lambda: h
    h.to_parquet(out_dir / "history.parquet")
    print(len(old), len(h), flush=True)
    orig = L.id_outcomes

    def patched(pop):
        x = orig(pop).drop(columns=["verified", "played"])
        n = new[new["season"] == L.SEASON][["week", "gsis_id", "played"]]
        x = x.merge(n, on=["week", "gsis_id"], how="left")
        x["verified"] = x["played"].notna()
        x["played"] = x["played"].fillna(0).astype(int)
        return x

    L.id_outcomes = patched
    for fn in (H.cmd_reliability, H.cmd_ladder, U.cmd_confound, H.cmd_check2026, U.cmd_check2026):
        fn(a)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("build", cmd_build),
        ("validate", cmd_validate),
        ("rerun", cmd_rerun),
        ("lead106", cmd_lead106),
    ):
        sp = sub.add_parser(name)
        sp.add_argument("--complete", action="store_true")
        sp.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
