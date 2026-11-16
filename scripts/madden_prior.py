from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/madden"
OUT = ROOT / "artifacts/madden_prior"
SNAPS = ROOT / "data/players/raw/20261111T141547Z/snap_counts.parquet"
SPINE = ROOT / "data/processed/game_features.parquet"
SOURCE = (
    "https://raw.githubusercontent.com/theedgepredictor/nfl-madden-data/main/"
    "data/madden/{kind}/{season}.csv"
)
KIND_PREFIX = {"processed": "", "stage": "stage_"}
USER_AGENT = "Mozilla/5.0 (research-client)"
UNITS = {"off": "offense_pct", "def": "defense_pct"}
MADDEN_COLUMNS = ["season", "fullname", "team", "position", "pfr_id", "overallrating"]


def fetch(seasons, pause):
    RAW.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        for kind, prefix in KIND_PREFIX.items():
            path = RAW / f"{prefix}{season}.csv"
            if path.exists():
                continue
            req = urllib.request.Request(
                SOURCE.format(kind=kind, season=season), headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                path.write_bytes(resp.read())
            time.sleep(pause)


def load_madden(seasons):
    frames, joins = [], {}
    for season in seasons:
        stage = pd.read_csv(RAW / f"stage_{season}.csv", usecols=MADDEN_COLUMNS[:-2] + ["overallrating"])
        assert (stage.season == season).all()
        stage = stage.drop_duplicates()
        reg = pd.read_csv(RAW / f"{season}.csv", usecols=["fullname", "team", "pfr_id"]).dropna()
        by_team = reg.drop_duplicates(["fullname", "team"], keep=False)
        unique_name = reg.drop_duplicates("fullname", keep=False)[["fullname", "pfr_id"]]
        d = stage.merge(by_team, on=["fullname", "team"], how="left")
        fill = d.pfr_id.isna()
        d.loc[fill, "pfr_id"] = d.loc[fill, "fullname"].map(unique_name.set_index("fullname").pfr_id)
        joins[season] = {
            "madden_players": len(d),
            "pfr_by_name_team": int((~fill).sum()),
            "pfr_by_unique_name": int((fill & d.pfr_id.notna()).sum()),
            "unjoined": int(d.pfr_id.isna().sum()),
        }
        frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    m["overallrating"] = pd.to_numeric(m.overallrating, errors="coerce")
    m = m.dropna(subset=["overallrating", "team", "position"])
    return m, joins


def load_snaps():
    s = pd.read_parquet(
        SNAPS,
        columns=["game_id", "season", "game_type", "week", "pfr_player_id", "team", *UNITS.values()],
    )
    s = s[s.game_type == "REG"].drop(columns="game_type")
    s["season"] = s.season.astype(int)
    s["week"] = s.week.astype(int)
    for c in UNITS.values():
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0.0)
    return s


def team_code_map(madden, snaps):
    mapping = {}
    for season, roster in madden.groupby("season"):
        wk = snaps[snaps.season == season]
        known = set(wk.team)
        for team in sorted(set(roster.team) - known):
            ids = set(roster.loc[roster.team == team, "pfr_id"].dropna())
            overlap = wk[wk.pfr_player_id.isin(ids)].groupby("team").pfr_player_id.nunique()
            if len(overlap):
                mapping[(season, team)] = overlap.idxmax()
    return mapping


def unit_index(frame, weight_col):
    w = frame[weight_col].to_numpy(dtype=float)
    r = frame.overallrating.to_numpy(dtype=float)
    rated = np.isfinite(r) & (w > 0)
    total = w[w > 0].sum()
    if not rated.any() or total <= 0:
        return np.nan, np.nan
    return float((w[rated] * r[rated]).sum() / w[rated].sum()), float(w[rated].sum() / total)


def prev_game_indexes(snaps, ratings):
    s = snaps.merge(
        ratings, left_on=["season", "pfr_player_id"], right_on=["season", "pfr_id"], how="left"
    )
    rows = []
    for (season, team, week), g in s.groupby(["season", "team", "week"]):
        rec = {"season": season, "team": team, "snap_week": week}
        for unit, col in UNITS.items():
            rec[f"{unit}_idx"], rec[f"{unit}_cov"] = unit_index(g, col)
        rows.append(rec)
    return pd.DataFrame(rows)


def ranked_roster(madden):
    r = madden.copy()
    r["slot_rank"] = r.groupby(["season", "team", "position"]).overallrating.rank(
        ascending=False, method="first"
    )
    return r


def week1_slot_pct(roster, snaps):
    w1 = snaps[snaps.week == snaps.groupby("season").week.transform("min")]
    w1 = w1.rename(columns={"pfr_player_id": "pfr_id"})[["season", "team", "pfr_id", *UNITS.values()]]
    played_teams = w1[["season", "team"]].drop_duplicates()
    r = roster.merge(played_teams, on=["season", "team"], how="inner")
    r = r.merge(w1, on=["season", "team", "pfr_id"], how="left")
    for c in UNITS.values():
        r[c] = r[c].fillna(0.0)
    return r.groupby(["season", "position", "slot_rank"])[list(UNITS.values())].mean().reset_index()


def week1_indexes(roster, slot_pct, seasons):
    rows = []
    for season in seasons:
        prior = slot_pct[slot_pct.season < season]
        if prior.empty:
            continue
        weights = prior.groupby(["position", "slot_rank"])[list(UNITS.values())].mean().reset_index()
        cur = roster[roster.season == season].merge(weights, on=["position", "slot_rank"], how="left")
        for c in UNITS.values():
            cur[c] = cur[c].fillna(0.0)
        for team, g in cur.groupby("team"):
            rec = {"season": season, "team": team}
            for unit, col in UNITS.items():
                rec[f"{unit}_idx"], rec[f"{unit}_cov"] = unit_index(g, col)
            rows.append(rec)
    return pd.DataFrame(rows)


def loso_vs_line(games, col, line_col):
    out = pd.Series(np.nan, index=games.index)
    ok = games[col].notna() & games[line_col].notna()
    for season in sorted(games.season.unique()):
        train = games[ok & (games.season != season)]
        test = ok & (games.season == season)
        if train.empty or not test.any():
            continue
        b, a = np.polyfit(train[line_col].to_numpy(), train[col].to_numpy(), 1)
        out[test] = games.loc[test, col] - (a + b * games.loc[test, line_col])
    return out


def main():
    ap = argparse.ArgumentParser(description="Madden launch-rating team talent prior per game.")
    ap.add_argument("--first-season", type=int, required=True, help="first season to build")
    ap.add_argument("--last-season", type=int, required=True, help="last season to build")
    ap.add_argument("--pause", type=float, required=True, help="seconds between source requests")
    args = ap.parse_args()
    seasons = list(range(args.first_season, args.last_season + 1))
    fetch(seasons, args.pause)
    madden, joins = load_madden(seasons)
    snaps = load_snaps()
    snaps = snaps[snaps.season.isin(seasons)]
    mapping = team_code_map(madden, snaps)
    madden["team"] = [mapping.get((s, t), t) for s, t in zip(madden.season, madden.team, strict=True)]
    ratings = (
        madden.dropna(subset=["pfr_id"])
        .sort_values("overallrating", ascending=False)
        .drop_duplicates(["season", "pfr_id"])[["season", "pfr_id", "overallrating"]]
    )
    prev = prev_game_indexes(snaps, ratings)
    roster = ranked_roster(madden)
    slot_pct = week1_slot_pct(roster, snaps)
    wk1 = week1_indexes(roster, slot_pct, seasons)

    spine = pd.read_parquet(
        SPINE, columns=["game_id", "season", "week", "game_type", "home_team", "away_team", "spread_line"]
    )
    spine = spine[(spine.game_type == "REG") & spine.season.isin(seasons)].copy()
    spine["season"] = spine.season.astype(int)
    spine["week"] = spine.week.astype(int)
    first_week = spine.groupby("season").week.transform("min")
    team_rows = []
    for side in ("home", "away"):
        t = spine[["game_id", "season", "week"]].copy()
        t["team"] = spine[f"{side}_team"]
        t["side"] = side
        t["is_first"] = spine.week.eq(first_week)
        team_rows.append(t)
    teams = pd.concat(team_rows, ignore_index=True)
    first = teams[teams.is_first].merge(wk1, on=["season", "team"], how="left")
    first["source"] = "week1_template"
    later = teams[~teams.is_first].sort_values("week")
    prev_sorted = prev.sort_values("snap_week")
    later = pd.merge_asof(
        later,
        prev_sorted,
        left_on="week",
        right_on="snap_week",
        by=["season", "team"],
        allow_exact_matches=False,
        direction="backward",
    )
    later["source"] = "previous_game"
    teams = pd.concat([first, later], ignore_index=True)
    teams["idx"] = teams[["off_idx", "def_idx"]].mean(axis=1, skipna=False)
    wide = teams.pivot(index="game_id", columns="side", values=["idx", "off_cov", "def_cov", "source"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    games = spine.merge(wide.reset_index(), on="game_id", how="left")
    games["madden_talent_diff"] = pd.to_numeric(games.idx_home) - pd.to_numeric(games.idx_away)
    games["madden_talent_vs_line"] = loso_vs_line(games, "madden_talent_diff", "spread_line")
    OUT.mkdir(parents=True, exist_ok=True)
    keep = [
        "game_id",
        "season",
        "week",
        "spread_line",
        "madden_talent_diff",
        "madden_talent_vs_line",
        "source_home",
        "off_cov_home",
        "def_cov_home",
        "off_cov_away",
        "def_cov_away",
    ]
    games[keep].to_parquet(OUT / "madden_prior_features.parquet", index=False)
    diag = {
        "pfr_joins": joins,
        "team_code_map": {f"{s}:{t}": v for (s, t), v in mapping.items()},
        "rated_players_by_season": madden.groupby("season").size().to_dict(),
        "rated_with_pfr_share": float(madden.pfr_id.notna().mean()),
        "games": len(games),
        "games_with_term": int(games.madden_talent_vs_line.notna().sum()),
        "missing_by_season": games.groupby("season")
        .madden_talent_vs_line.apply(lambda x: int(x.isna().sum()))
        .to_dict(),
        "mean_snap_coverage": {
            c: float(pd.to_numeric(games[c]).mean())
            for c in ["off_cov_home", "def_cov_home", "off_cov_away", "def_cov_away"]
        },
        "corr_diff_line_by_season": games.groupby("season")
        .apply(lambda g: float(g.madden_talent_diff.corr(g.spread_line)))
        .round(3)
        .to_dict(),
    }
    (OUT / "build_diagnostics.json").write_text(json.dumps(diag, indent=1, default=str), encoding="utf-8")
    print(json.dumps(diag, default=str))


if __name__ == "__main__":
    main()
