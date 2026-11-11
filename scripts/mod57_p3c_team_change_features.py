from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLAYERS_RAW = ROOT / "data/players/raw"
ROSTER_SNAPSHOT = PLAYERS_RAW / "20261111T141547Z"
IDENTITY_GLOB = "*/players.parquet"
HISTORICAL_DEPTH = ROOT / "data/quarterbacks/depth/historical/20260902T225115Z/quarterbacks.parquet"
LIVE_DEPTH_GLOB = "data/raw/depth_charts/*/depth_charts.parquet"
SPINE = ROOT / "data/processed/game_features.parquet"
OUT = ROOT / "artifacts/mod57_p3c"
FIRST_SEASON = 2014
LAST_QB_SEASON = 2025
ON_ROSTER_STATUSES = ("ACT", "INA")
THIRD_EDGES = (0, 6, 12, 99)
THIRD_LABELS = ("weeks_1_6", "weeks_7_12", "weeks_13_18")
TRUNCATION_GAMES = 40
ROSTER_COLUMNS = (
    "ret_snap_share_off",
    "ret_snap_share_def",
    "arr_snap_vol_off",
    "arr_snap_vol_def",
)
QB_COLUMNS = ("qb_changed", "qb_starter_prior_qb_snap_share")
FEATURES = (*ROSTER_COLUMNS, *QB_COLUMNS)


def load_identity():
    path = sorted(glob.glob(str(PLAYERS_RAW / IDENTITY_GLOB)))[-1]
    ident = pd.read_parquet(path, columns=["pfr_id", "gsis_id"]).dropna()
    return dict(zip(ident.pfr_id, ident.gsis_id, strict=True))


def load_sources():
    roster = pd.read_parquet(
        ROSTER_SNAPSHOT / "weekly_rosters.parquet",
        columns=["season", "week", "team", "status", "gsis_id"],
    )
    roster = roster[roster.status.isin(ON_ROSTER_STATUSES) & roster.gsis_id.notna()]
    roster = roster[["season", "week", "team", "gsis_id"]].drop_duplicates()
    roster["season"] = roster.season.astype(int)
    roster["week"] = roster.week.astype(int)
    roster["rk"] = roster.season * 100 + roster.week
    roster["gsis_id"] = roster.gsis_id.astype(str)
    snaps = pd.read_parquet(
        ROSTER_SNAPSHOT / "snap_counts.parquet",
        columns=[
            "game_id",
            "season",
            "game_type",
            "pfr_player_id",
            "position",
            "team",
            "offense_snaps",
            "defense_snaps",
        ],
    )
    snaps = snaps[snaps.game_type == "REG"].copy()
    ident = load_identity()
    snaps["gsis_id"] = snaps.pfr_player_id.map(ident)
    unmapped = float(snaps.gsis_id.isna().mean())
    snaps = snaps.dropna(subset=["gsis_id"])
    snaps["season"] = snaps.season.astype(int)
    return roster, snaps, unmapped


def load_spine():
    g = pd.read_parquet(
        SPINE,
        columns=["game_id", "season", "week", "game_type", "home_team", "away_team", "kickoff"],
    )
    g = g[(g.game_type == "REG") & (g.season >= FIRST_SEASON)].copy()
    g["kickoff"] = pd.to_datetime(g.kickoff, utc=True)
    g["target"] = g.season * 100 + g.week
    return g.sort_values(["kickoff", "game_id"]).reset_index(drop=True)


def load_depth():
    hist = pd.read_parquet(
        HISTORICAL_DEPTH,
        columns=["team", "gsis_id", "pos_rank", "effective_at_utc"],
    )
    hist = hist.rename(columns={"effective_at_utc": "effective_at"})
    live_files = sorted(glob.glob(str(ROOT / LIVE_DEPTH_GLOB)))
    frames = [hist]
    for path in live_files:
        live = pd.read_parquet(path, columns=["dt", "team", "gsis_id", "pos_abb", "pos_rank"])
        live = live[live.pos_abb == "QB"].copy()
        live["effective_at"] = pd.to_datetime(live.dt, utc=True)
        frames.append(live[["team", "gsis_id", "pos_rank", "effective_at"]])
    depth = pd.concat(frames, ignore_index=True)
    depth["effective_at"] = pd.to_datetime(depth.effective_at, utc=True)
    depth["pos_rank"] = pd.to_numeric(depth.pos_rank, errors="coerce")
    depth = depth.dropna(subset=["gsis_id", "pos_rank"])
    best = depth.groupby(["team", "effective_at"]).pos_rank.transform("min")
    depth = depth[depth.pos_rank == best]
    depth = depth.sort_values(["team", "effective_at", "gsis_id"])
    return depth.drop_duplicates(["team", "effective_at"])[["team", "effective_at", "gsis_id"]]


def prior_tables(snaps):
    per = snaps.groupby(["season", "team", "gsis_id"], as_index=False)[
        ["offense_snaps", "defense_snaps"]
    ].sum()
    team_tot = per.groupby(["season", "team"], as_index=False)[
        ["offense_snaps", "defense_snaps"]
    ].sum()
    league = team_tot.groupby("season")[["offense_snaps", "defense_snaps"]].mean()
    qb = (
        snaps[snaps.position == "QB"]
        .groupby(["season", "team", "gsis_id"], as_index=False)["offense_snaps"]
        .sum()
    )
    return per, team_tot, league, qb


def roster_features(roster, per, team_tot, league, pairs):
    rows = pairs.merge(roster[["rk", "team", "gsis_id"]], on="rk", how="inner")
    prev = per.assign(season=per.season + 1).rename(
        columns={"team": "team_prev", "offense_snaps": "off_prev", "defense_snaps": "def_prev"}
    )
    rows = rows.merge(prev, on=["season", "gsis_id"], how="left")
    rows[["off_prev", "def_prev"]] = rows[["off_prev", "def_prev"]].fillna(0.0)
    same = (rows.team_prev == rows.team).to_numpy(dtype=bool, na_value=False)
    rows["ret_off"] = np.where(same, rows.off_prev, 0.0)
    rows["ret_def"] = np.where(same, rows.def_prev, 0.0)
    rows["arr_off"] = np.where(~same, rows.off_prev, 0.0)
    rows["arr_def"] = np.where(~same, rows.def_prev, 0.0)
    agg = rows.groupby(["season", "rk", "team"], as_index=False)[
        ["ret_off", "ret_def", "arr_off", "arr_def"]
    ].sum()
    tot = team_tot.assign(season=team_tot.season + 1).rename(
        columns={"offense_snaps": "tot_off", "defense_snaps": "tot_def"}
    )
    agg = agg.merge(tot, on=["season", "team"], how="inner")
    lg = league.assign(_s=league.index + 1).set_index("_s")
    agg["lg_off"] = agg.season.map(lg.offense_snaps)
    agg["lg_def"] = agg.season.map(lg.defense_snaps)
    agg["ret_snap_share_off"] = agg.ret_off / agg.tot_off
    agg["ret_snap_share_def"] = agg.ret_def / agg.tot_def
    agg["arr_snap_vol_off"] = agg.arr_off / agg.lg_off
    agg["arr_snap_vol_def"] = agg.arr_def / agg.lg_def
    return agg[["season", "rk", "team", *ROSTER_COLUMNS]]


def team_roster_keys(games, roster, side):
    team_col = f"{side}_team"
    table = roster[["team", "rk"]].drop_duplicates().sort_values("rk")
    table["team"] = table.team.astype(object)
    left = games[[team_col, "season", "target"]].copy()
    left["_row"] = np.arange(len(left))
    left[team_col] = left[team_col].astype(object)
    left = left.sort_values("target")
    merged = pd.merge_asof(
        left,
        table.rename(columns={"team": team_col}),
        left_on="target",
        right_on="rk",
        by=team_col,
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row")
    prior = merged.rk.to_numpy(dtype=float)
    prior = np.where(np.floor_divide(prior, 100) >= merged.season.to_numpy() - 1, prior, np.nan)
    has_same = games.target.to_numpy() * 0
    exact = pd.MultiIndex.from_frame(table.rename(columns={"team": team_col})[[team_col, "rk"]])
    probe = pd.MultiIndex.from_arrays([games[team_col].astype(object), games.target])
    same = np.where(probe.isin(exact), games.target.to_numpy(), np.nan) + has_same
    return prior, same


def qb_features(games, depth, qb):
    sides = []
    for side in ("home", "away"):
        part = games[["game_id", "season", "kickoff", f"{side}_team"]].rename(
            columns={f"{side}_team": "team"}
        )
        part["side"] = side
        sides.append(part)
    long = pd.concat(sides, ignore_index=True).sort_values("kickoff")
    long["kickoff"] = long.kickoff.astype("datetime64[ns, UTC]")
    d = depth.copy()
    d["team"] = d.team.astype(object)
    long["team"] = long.team.astype(object)
    d["effective_at"] = d.effective_at.astype("datetime64[ns, UTC]")
    d = d.sort_values("effective_at")
    merged = pd.merge_asof(
        long,
        d,
        left_on="kickoff",
        right_on="effective_at",
        by="team",
        direction="backward",
    )
    merged["qb_depth_age_days"] = (
        merged.kickoff - merged.effective_at
    ).dt.total_seconds() / 86400.0
    prior_qb = qb.assign(season=qb.season + 1)
    tot = prior_qb.groupby(["season", "team"], as_index=False).offense_snaps.sum()
    tot = tot.rename(columns={"offense_snaps": "qb_total"})
    primary = prior_qb.sort_values(
        ["season", "team", "offense_snaps", "gsis_id"], ascending=[True, True, False, True]
    ).drop_duplicates(["season", "team"])
    primary = primary.rename(columns={"gsis_id": "prior_primary_qb"})[
        ["season", "team", "prior_primary_qb"]
    ]
    own = prior_qb.rename(columns={"gsis_id": "gsis_id", "offense_snaps": "starter_prior_snaps"})
    merged = merged.merge(primary, on=["season", "team"], how="left")
    merged = merged.merge(tot, on=["season", "team"], how="left")
    merged = merged.merge(own, on=["season", "team", "gsis_id"], how="left")
    merged["starter_prior_snaps"] = merged.starter_prior_snaps.fillna(0.0)
    have = (
        merged.gsis_id.notna() & merged.prior_primary_qb.notna() & (merged.season <= LAST_QB_SEASON)
    )
    merged["qb_changed"] = np.where(
        have, (merged.gsis_id != merged.prior_primary_qb).astype(float), np.nan
    )
    merged["qb_starter_prior_qb_snap_share"] = np.where(
        have, merged.starter_prior_snaps / merged.qb_total, np.nan
    )
    merged["qb_starter_id"] = merged.gsis_id.where(have)
    return merged[
        [
            "game_id",
            "side",
            "qb_changed",
            "qb_starter_prior_qb_snap_share",
            "qb_starter_id",
            "qb_depth_age_days",
        ]
    ]


def build(games, roster, snaps, depth):
    per, team_tot, league, qb = prior_tables(snaps)
    key_cols = {}
    for side in ("home", "away"):
        prior, same = team_roster_keys(games, roster, side)
        key_cols[f"{side}_rk_prior"] = prior
        key_cols[f"{side}_rk_same"] = same
    games = games.assign(**key_cols)
    out = games[["game_id", "season", "week", "home_team", "away_team", *key_cols]].copy()
    pairs = pd.concat([games[["season", c]].rename(columns={c: "rk"}) for c in key_cols]).dropna()
    pairs["rk"] = pairs.rk.astype(int)
    pairs = pairs.drop_duplicates()
    feats = roster_features(roster, per, team_tot, league, pairs)
    for side in ("home", "away"):
        for variant, key in (("", f"{side}_rk_prior"), ("_sameweek", f"{side}_rk_same")):
            f = feats.rename(columns={"rk": key, "team": f"{side}_team"})
            f = f.rename(columns={c: f"{side}_{c}{variant}" for c in ROSTER_COLUMNS})
            f[key] = f[key].astype(float)
            out = out.merge(f, on=["season", key, f"{side}_team"], how="left")
    qbf = qb_features(games, depth, qb)
    for side in ("home", "away"):
        s = qbf[qbf.side == side].drop(columns="side")
        s = s.rename(columns={c: f"{side}_{c}" for c in s.columns if c != "game_id"})
        out = out.merge(s, on="game_id", how="left")
    for c in ROSTER_COLUMNS:
        for variant in ("", "_sameweek"):
            out[f"diff_{c}{variant}"] = out[f"home_{c}{variant}"] - out[f"away_{c}{variant}"]
    for c in QB_COLUMNS:
        out[f"diff_{c}"] = out[f"home_{c}"] - out[f"away_{c}"]
    out = out.rename(
        columns={
            "home_rk_prior": "home_roster_key_used",
            "away_rk_prior": "away_roster_key_used",
            "home_rk_same": "home_roster_key_sameweek",
            "away_rk_same": "away_roster_key_sameweek",
        }
    )
    return out


def feature_columns(frame):
    return [
        c
        for c in frame.columns
        if c.startswith(("home_", "away_", "diff_"))
        and not c.endswith(("_team", "_sameweek", "_id", "_age_days"))
    ]


def truncation_test(games, roster, snaps, depth, full):
    step = len(games) // TRUNCATION_GAMES
    sample = games.iloc[step // 2 :: step].head(TRUNCATION_GAMES)
    kick = games.set_index("game_id").kickoff
    snap_time = snaps.game_id.map(kick)
    cols = feature_columns(full)
    matched = 0
    detail = []
    for _, g in sample.iterrows():
        r = roster[roster.rk < g.target]
        s = snaps[~(snap_time >= g.kickoff)]
        d = depth[depth.effective_at < g.kickoff]
        one = build(games[games.game_id == g.game_id], r, s, d)
        a = full[full.game_id == g.game_id][cols].reset_index(drop=True)
        b = one[cols].reset_index(drop=True)
        same = bool(
            np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, atol=0, rtol=0)
        )
        matched += same
        detail.append({"game_id": g.game_id, "identical": same})
    return {
        "games_tested": len(sample),
        "identical": matched,
        "columns_compared": len(cols),
        "rows_deleted_rule": (
            "roster rows with (season, week) at or after the game week, snap rows from games "
            "kicking off at or after this kickoff, depth rows effective at or after this kickoff"
        ),
        "detail": detail,
    }


def asof_measurement(roster, snaps):
    snaps = snaps[
        (snaps.season >= FIRST_SEASON)
        & ((snaps.offense_snaps.fillna(0) + snaps.defense_snaps.fillna(0)) > 0)
    ]
    snaps = snaps.assign(week=snaps.game_id.str.slice(5, 7).astype(int))
    played = {
        k: set(v) for k, v in snaps.groupby(["season", "week", "team"]).gsis_id.apply(list).items()
    }
    sets = {k: set(v) for k, v in roster.groupby(["rk", "team"]).gsis_id.apply(list).items()}
    team_rks = {t: np.array(sorted(v)) for t, v in roster.groupby("team").rk.unique().items()}
    rows = []
    for (season, week, team), g in played.items():
        rk = season * 100 + week
        rks = team_rks[team]
        i = np.searchsorted(rks, rk)
        if i == 0 or i >= len(rks) or rks[i] != rk:
            continue
        prev_rk = rks[i - 1]
        if prev_rk // 100 < season - 1:
            continue
        nxt_rk = rks[i + 1] if i + 1 < len(rks) and rks[i + 1] // 100 == season else None
        rn = sets.get((rk, team), set())
        rp = sets.get((prev_rk, team), set())
        rnx = sets.get((nxt_rk, team), set()) if nxt_rk is not None else None
        arrivals = rn - rp
        rows.append(
            {
                "week1": week == 1,
                "played": len(g),
                "played_on_rn": len(g & rn),
                "played_on_rprev": len(g & rp),
                "played_not_on_rn_but_on_rnext": None if rnx is None else len((g - rn) & rnx),
                "has_next": rnx is not None,
                "roster_rows": len(rn),
                "arrival_rows": len(arrivals),
                "arrivals_who_played_that_week": len(arrivals & g),
            }
        )
    d = pd.DataFrame(rows)
    out = {}
    for name, sub in (("week_1", d[d.week1]), ("weeks_2_plus", d[~d.week1])):
        nx = sub[sub.has_next]
        out[name] = {
            "team_weeks": len(sub),
            "share_of_participants_on_week_n_roster": float(
                sub.played_on_rn.sum() / sub.played.sum()
            ),
            "share_of_participants_on_prior_roster": float(
                sub.played_on_rprev.sum() / sub.played.sum()
            ),
            "participants_missing_from_week_n_but_on_next_roster": int(
                nx.played_not_on_rn_but_on_rnext.sum()
            ),
            "team_weeks_with_next_roster": len(nx),
            "arrival_rows_vs_prior_roster": int(sub.arrival_rows.sum()),
            "arrival_rows_share_of_week_n_rows": float(
                sub.arrival_rows.sum() / sub.roster_rows.sum()
            ),
            "arrivals_who_played_that_week": int(sub.arrivals_who_played_that_week.sum()),
            "possible_post_kickoff_share_of_week_n_rows": float(
                sub.arrivals_who_played_that_week.sum() / sub.roster_rows.sum()
            ),
            "possible_post_kickoff_share_of_arrival_rows": float(
                sub.arrivals_who_played_that_week.sum() / max(sub.arrival_rows.sum(), 1)
            ),
        }
    return out


def distributions(full):
    frame = full.copy()
    frame["third"] = pd.cut(frame.week, bins=list(THIRD_EDGES), labels=list(THIRD_LABELS))
    cols = [c for c in frame.columns if c.startswith("diff_")]
    long = frame.melt(id_vars=["season", "third"], value_vars=cols, var_name="column")
    long = long.dropna(subset=["value"])
    by_season = long.groupby(["column", "season", "third"], observed=True).value.agg(
        ["count", "mean", "std", "min", "max"]
    )
    overall = long.groupby(["column", "third"], observed=True).value.agg(
        ["count", "mean", "std", "min", "max"]
    )
    cover = frame.groupby("season").apply(
        lambda d: pd.Series(
            {
                "games": len(d),
                "all_roster_columns_non_null": int(
                    d[[f"diff_{c}" for c in ROSTER_COLUMNS]].notna().all(axis=1).sum()
                ),
                "all_qb_columns_non_null": int(
                    d[[f"diff_{c}" for c in QB_COLUMNS]].notna().all(axis=1).sum()
                ),
                "all_columns_non_null": int(
                    d[[f"diff_{c}" for c in FEATURES]].notna().all(axis=1).sum()
                ),
            }
        ),
        include_groups=False,
    )
    return by_season, overall, cover


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build early-season team-change features (roster continuity, quarterback change) "
            "with as-of proof."
        )
    )
    parser.add_argument("--out-dir", default=str(OUT), help="output directory")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roster, snaps, unmapped = load_sources()
    games = load_spine()
    depth = load_depth()
    full = build(games, roster, snaps, depth)
    full.to_parquet(out_dir / "team_change_features.parquet", index=False)
    by_season, overall, cover = distributions(full)
    by_season.to_csv(out_dir / "distribution_by_season_third.csv")
    overall.to_csv(out_dir / "distribution_by_third.csv")
    cover.to_csv(out_dir / "coverage_by_season.csv")
    asof = asof_measurement(roster, snaps)
    trunc = truncation_test(games, roster, snaps, depth, full)
    report = {
        "rows": len(full),
        "seasons": [int(full.season.min()), int(full.season.max())],
        "snap_rows_unmapped_to_gsis_share": unmapped,
        "asof_roster_measurement": asof,
        "truncation": trunc,
    }
    (out_dir / "asof_proof.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "truncation"}, indent=2))
    print({k: v for k, v in trunc.items() if k != "detail"})
    print(cover.to_string())
    print(overall.round(4).to_string())


if __name__ == "__main__":
    main()
