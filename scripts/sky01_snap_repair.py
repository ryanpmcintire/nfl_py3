import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sky01_drift_weighted as dw
import sky01_news_availability as base

PLAYERS = sorted(glob.glob(str(dw.ROOT / "data" / "players" / "raw" / "*" / "players.parquet")))[-1]
POS_GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "SPEC"]


def id_map():
    pl = pd.read_parquet(PLAYERS, columns=["gsis_id", "pfr_id", "position_group", "draft_round"])
    ro = pd.read_parquet(dw.ROSTERS, columns=["gsis_id", "pfr_id"]).dropna()
    m = pd.concat([pl[["gsis_id", "pfr_id"]].dropna(), ro]).drop_duplicates("pfr_id")
    return pl.drop_duplicates("gsis_id"), m


def season_shares(m):
    s = pd.read_parquet(
        dw.SNAPS, columns=["season", "game_type", "pfr_player_id", "offense_pct", "defense_pct"]
    )
    s = s[s.game_type == "REG"].copy()
    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s = s.merge(m, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    return s.groupby(["gsis_id", "season"]).share.mean().rename("share").reset_index()


def design(df, pl):
    df = df.merge(pl[["gsis_id", "position_group", "draft_round"]], on="gsis_id", how="left")
    x = pd.DataFrame(index=df.index)
    x["const"] = 1.0
    x["has_last"] = df.last_share.notna().astype(float)
    x["last"] = df.last_share.fillna(0.0)
    x["gap"] = df.gap.fillna(0.0) * x.has_last
    for k in (1, 2, 3):
        x[f"round{k}"] = (df.draft_round == k).astype(float)
    x["round47"] = df.draft_round.between(4, 7).astype(float)
    pg = df.position_group.where(df.position_group.isin(POS_GROUPS), "SPEC")
    for g in POS_GROUPS[1:]:
        x["pos_" + g] = (pg == g).astype(float)
    return x.values


def last_features(ps, keys):
    ps = ps.sort_values(["gsis_id", "season"]).copy()
    ps["last_share"] = ps.groupby("gsis_id").share.shift(1)
    ps["last_season"] = ps.groupby("gsis_id").season.shift(1)
    ps["gap"] = ps.season - ps.last_season
    cols = ["gsis_id", "season", "last_share", "gap"]
    return keys.merge(ps[cols], on=["gsis_id", "season"], how="left")


def fitted_expectation(q, ps, pl, seasons):
    out = pd.Series(np.nan, index=q.index)
    tr_all = last_features(ps, ps[["gsis_id", "season"]])
    tr_all = tr_all.merge(ps[["gsis_id", "season", "share"]], on=["gsis_id", "season"])
    for s in seasons:
        tr = tr_all[tr_all.season != s]
        X = design(tr, pl)
        beta = np.linalg.lstsq(X, tr.share.values, rcond=None)[0]
        rows = q[q.season == s]
        last = ps[ps.season < s].sort_values("season").groupby("gsis_id").tail(1)
        f = rows[["gsis_id"]].merge(
            last.rename(columns={"share": "last_share", "season": "ls"}), on="gsis_id", how="left"
        )
        f["gap"] = s - f.ls
        out.loc[rows.index] = np.clip(design(f, pl) @ beta, 0.0, 1.0)
    return out


def repaired_feature(p, g, ps_in, sm, ps, pl):
    q = p[p.cls.isin(base.OUT_CLASSES) & (p.created_at < p.opener_ts)].drop_duplicates(
        ["game_id", "team", "gsis_id"]
    )
    q = dw.weight_for(q[["game_id", "team", "gsis_id", "season", "week"]], ps_in, sm)
    q = q.reset_index(drop=True)
    none = q.w_source == "none"
    fit = fitted_expectation(q[none], ps, pl, sorted(q.season.unique()))
    q.loc[none, "w"] = fit.reindex(q.index[none]).values
    q.loc[none, "w_source"] = "fitted"
    n = q.groupby(["game_id", "team"]).w.sum().rename("n").reset_index()
    cols = {}
    for side in ("home_team", "away_team"):
        cols[side] = (
            g[["game_id", side]]
            .merge(n, left_on=["game_id", side], right_on=["game_id", "team"], how="left")["n"]
            .fillna(0)
            .values
        )
    return cols["home_team"] - cols["away_team"], q


def main():
    pl, m = id_map()
    ps = season_shares(m)
    _t, p, _s = base.build_table()
    g = pd.read_parquet(base.PERGAME)
    sub = g[g.margin_vs_open != 0].copy().reset_index(drop=True)
    ps_in, sm = dw.prior_share()
    x0, src0, _cov0 = dw.weighted_feature(p, sub, ps_in, sm)
    x1, q = repaired_feature(p, sub, ps_in, sm, ps, pl)
    qs = q[q.game_id.isin(sub.game_id)]
    res = {
        "weight_source_counts_repaired": {k: int(v) for k, v in qs.w_source.value_counts().items()},
        "weight_source_counts_declared": {k: int(v) for k, v in src0.items()},
        "fitted_weight_mean": float(qs[qs.w_source == "fitted"].w.mean()),
        "declared": dw.score(sub, x0, base.SEED + 1),
        "repaired": dw.score(sub, x1, base.SEED + 1),
    }
    with open(base.OUT / "snap_repair.json", "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
