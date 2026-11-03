from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
SNAP = sorted(glob.glob(str(REPO / "data/players/raw/*/snap_counts.parquet")))[-1]
ROST = sorted(glob.glob(str(REPO / "data/players/raw/*/weekly_rosters.parquet")))[-1]
INJ = sorted(glob.glob(str(REPO / "data/raw/nflverse_injuries/*/injuries.parquet")))[-1]
SCHED = sorted(glob.glob(str(REPO / "data/raw/*/schedules.parquet")))[-1]
OPEN = REPO / "artifacts/opener_evaluation/20261103T195331Z/per_game.parquet"
OFF = {
    "RB": "RB FB HB RB/W FB/D FB/R FB/T RB/F",
    "WR": "WR WR/R",
    "TE": "TE TE/D",
    "OL": "T G C OL C/G G/T G/C OT T/G G/OT OG",
}
DEF = {
    "DL": "DE DT NT DL DT/D DE/L DE/D",
    "LB": "LB ILB OLB MLB LB/F LB/S",
    "DB": "CB FS SS S DB DB/L CB/R",
}
GROUPS = ["RB", "WR", "TE", "OL", "DL", "LB", "DB"]
POS = {p: g for d in (OFF, DEF) for g, s in d.items() for p in s.split()}
MISSED = (1, 2, 3)
FIRST_EVENT_SEASON = 2014
TEST_SEASONS = range(2020, 2026)
DRAWS = 2000
SEED = 20261103


def build():
    s = pd.read_parquet(SNAP)
    s = s[(s["game_type"] == "REG") & s["pfr_player_id"].notna()].copy()
    s["grp"] = s["position"].map(POS)
    s = s[s["grp"].notna()]
    s["share"] = np.where(s["grp"].isin(OFF), s["offense_pct"], s["defense_pct"]).astype(float)
    snaps = s[["offense_snaps", "defense_snaps", "st_snaps"]].fillna(0).sum(axis=1)
    s["any"] = (snaps > 0).astype(int)
    tg = (
        s[["game_id", "season", "week", "team"]]
        .drop_duplicates()
        .sort_values(["season", "team", "week"])
    )
    r = pd.read_parquet(ROST, columns=["gsis_id", "pfr_id"]).dropna().drop_duplicates()
    pfr2gsis = dict(zip(r["pfr_id"], r["gsis_id"], strict=False))
    inj = pd.read_parquet(INJ, columns=["season", "week", "gsis_id", "report_status"])
    inj = inj[inj["gsis_id"].notna()]
    ik = {
        (g, int(se), int(w)): st
        for g, se, w, st in zip(
            inj["gsis_id"], inj["season"], inj["week"], inj["report_status"], strict=True
        )
    }
    grp_of = s.groupby("pfr_player_id")["grp"].agg(lambda x: x.mode().iloc[0])
    tgs = dict(iter(tg.groupby(["season", "team"])))
    rows = []
    for (se, tm, pid), g in s.groupby(["season", "team", "pfr_player_id"]):
        games = tgs[(se, tm)]
        g = g.drop_duplicates("game_id").set_index("game_id")
        sh = games["game_id"].map(g["share"]).fillna(0.0).to_numpy()
        pl = games["game_id"].map(g["any"]).fillna(0).to_numpy().astype(int)
        wk = games["week"].to_numpy()
        gid = games["game_id"].to_numpy()
        gs = pfr2gsis.get(pid)
        played_sh = []
        run = 0
        run_start = 0
        for k in range(len(gid)):
            base = float(np.mean(played_sh)) if played_sh else np.nan
            listed = False
            status = None
            if gs is not None and run > 0:
                listed = any((gs, se, int(wk[j])) in ik for j in range(run_start, k))
                status = ik.get((gs, se, int(wk[k])))
            rows.append(
                (
                    se,
                    int(wk[k]),
                    gid[k],
                    tm,
                    pid,
                    grp_of[pid],
                    base,
                    sh[k],
                    pl[k],
                    run,
                    listed,
                    status,
                )
            )
            if pl[k] == 1:
                played_sh.append(sh[k])
                run = 0
            else:
                if run == 0:
                    run_start = k
                run += 1
    cols = [
        "season",
        "week",
        "game_id",
        "team",
        "pid",
        "grp",
        "base",
        "share",
        "played",
        "m",
        "listed",
        "status",
    ]
    return pd.DataFrame(rows, columns=cols)


def cutoffs(pg):
    ps = pg[pg["played"] == 1].groupby(["season", "grp", "pid"])["share"].mean().reset_index()
    ps = ps[ps["share"] > 0]
    out = {}
    first = pg["season"].min()
    for se in sorted(pg["season"].unique()):
        prior = ps[ps["season"] < se] if se > first else ps[ps["season"] == se]
        out[se] = prior.groupby("grp")["share"].median().to_dict()
    return out


def boot(x, cl, rng):
    ids, inv = np.unique(cl, return_inverse=True)
    sums = np.bincount(inv, weights=x)
    cnt = np.bincount(inv)
    d = np.empty(DRAWS)
    for i in range(DRAWS):
        p = rng.integers(0, len(ids), len(ids))
        d[i] = sums[p].sum() / cnt[p].sum()
    return {
        "mean": float(np.mean(x)),
        "lo": float(np.percentile(d, 2.5)),
        "hi": float(np.percentile(d, 97.5)),
        "pp": float((d > 0).mean()),
        "n": len(x),
    }


def key(df):
    return (df["team"] + df["season"].astype(str)).to_numpy()


def side_sum(o, agg, col, side):
    m = o[["game_id", f"{side}_team"]].merge(
        agg[["game_id", "team", col]],
        left_on=["game_id", f"{side}_team"],
        right_on=["game_id", "team"],
        how="left",
    )
    return m[col].fillna(0.0).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "tests/scratch/per15"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pg = build()
    cut = cutoffs(pg)
    pg["cut"] = [cut[s].get(g, np.nan) for s, g in zip(pg["season"], pg["grp"], strict=True)]
    pg["starter"] = pg["base"] >= pg["cut"]
    pg["loss"] = pg["base"] - pg["share"]
    late = pg["season"] >= FIRST_EVENT_SEASON
    pg["cand"] = (
        pg["starter"] & pg["m"].isin(MISSED) & pg["listed"] & (pg["status"] != "Out") & late
    )
    ctrl = pg[pg["starter"] & (pg["m"] == 0) & late & pg["base"].notna()]
    cand = pg[pg["cand"]].copy()
    res = {"snap_file": SNAP, "candidates": len(cand), "cells": {}}
    res["control_loss_starters_after_playing"] = boot(ctrl["loss"].to_numpy(), key(ctrl), rng)
    res["played_rate_candidates"] = float(cand["played"].mean())
    for g in ["ALL", *GROUPS]:
        for m in [*MISSED, 0]:
            sel = cand if g == "ALL" else cand[cand["grp"] == g]
            sel = sel if m == 0 else sel[sel["m"] == m]
            if len(sel) < len(MISSED) * len(GROUPS):
                continue
            c = ctrl if g == "ALL" else ctrl[ctrl["grp"] == g]
            pl = sel[sel["played"] == 1]
            res["cells"][f"{g}_m{m}"] = {
                "loss": boot(sel["loss"].to_numpy(), key(sel), rng),
                "loss_minus_control": float(sel["loss"].mean() - c["loss"].mean()),
                "loss_played_only": boot(pl["loss"].to_numpy(), key(pl), rng)
                if len(pl) > len(GROUPS)
                else None,
                "played_rate": float(sel["played"].mean()),
                "mean_baseline": float(sel["base"].mean()),
            }
    sch = pd.read_parquet(
        SCHED, columns=["game_id", "season", "game_type", "home_team", "away_team", "result"]
    )
    sch = sch[sch["game_type"] == "REG"]
    st = pg[pg["starter"] & pg["base"].notna() & late]
    L = (
        st.pivot_table(
            index=["game_id", "team"], columns="grp", values="loss", aggfunc="sum", fill_value=0.0
        )
        .reindex(columns=GROUPS, fill_value=0.0)
        .reset_index()
    )
    h = sch.merge(L, left_on=["game_id", "home_team"], right_on=["game_id", "team"])
    a = sch.merge(L, left_on=["game_id", "away_team"], right_on=["game_id", "team"])
    mg = h[["game_id", "season", "result", *GROUPS]].merge(
        a[["game_id", *GROUPS]], on="game_id", suffixes=("_h", "_a")
    )
    D = np.column_stack([mg[f"{g}_h"] - mg[f"{g}_a"] for g in GROUPS])
    o = pd.read_parquet(OPEN)
    o = o[o["season"].between(2020, 2025) & (o["margin_vs_open"] != 0)].copy()
    o["y"] = (o["margin_vs_open"] > 0).astype(int)
    o["z"] = logit(o["home_cover_probability_at_open"].clip(1e-6, 1 - 1e-6))
    parts = o["game_id"].str.split("_", expand=True)
    o["home_team"] = parts[3]
    o["away_team"] = parts[2]
    aggs, bs = [], {}
    for se in TEST_SEASONS:
        trm = (mg["season"] < se).to_numpy()
        X = np.column_stack([np.ones(trm.sum()), D[trm]])
        beta = np.linalg.lstsq(X, mg.loc[trm, "result"].to_numpy(), rcond=None)[0]
        bs[se] = dict(zip(GROUPS, np.round(beta[1:], 3).tolist(), strict=True))
        E = cand[cand["season"] < se].groupby(["grp", "m"])["loss"].mean()
        te = cand[cand["season"] == se].copy()
        te["unw"] = [E.get((g, m), 0.0) for g, m in zip(te["grp"], te["m"], strict=True)]
        te["val"] = te["unw"] * te["grp"].map(bs[se])
        aggs.append(te.groupby(["game_id", "team"])[["unw", "val"]].sum().reset_index())
    ag = pd.concat(aggs)
    for tag, col in (("v", "val"), ("u", "unw")):
        o[f"x_{tag}"] = side_sum(o, ag, col, "home") - side_sum(o, ag, col, "away")
        o[f"x_{tag}"] = o[f"x_{tag}"] / o[f"x_{tag}"].std()
    o["x_shuf"] = o.groupby("season")["x_v"].transform(lambda v: rng.permutation(v.to_numpy()))
    arms = {
        "intercept": [],
        "served": ["z"],
        "served+x_value": ["z", "x_v"],
        "served+x_unweighted": ["z", "x_u"],
        "served+x_shuffled": ["z", "x_shuf"],
    }
    P, coef = {}, {}
    for an, cols in arms.items():
        p = pd.Series(0.5, index=o.index)
        coef[an] = {}
        for se in TEST_SEASONS:
            trn = o[o["season"] != se]
            tst = o["season"] == se
            if not cols:
                p[tst] = trn["y"].mean()
                continue
            mdl = LogisticRegression(C=1e6, max_iter=1000).fit(trn[cols], trn["y"])
            p[tst] = mdl.predict_proba(o.loc[tst, cols])[:, 1]
            coef[an][se] = np.round(mdl.coef_[0], 3).tolist()
        P[an] = p.clip(1e-6, 1 - 1e-6)

    def ll(p):
        return -(o["y"] * np.log(p) + (1 - o["y"]) * np.log(1 - p))

    act = o["x_v"] != 0
    term = {
        "n_games": len(o),
        "games_with_feature": int(act.sum()),
        "coef_by_fold": coef,
        "position_value_points_per_slot_loss": bs,
        "arms": {},
    }
    sp = P["served"] > 0.5
    win = o["y"] == 1
    for an in arms:
        pk = P[an] > 0.5
        fl = pk != sp
        term["arms"][an] = {
            "wins_all": int((pk == win).sum()),
            "losses_all": int((pk != win).sum()),
            "flips_vs_served": int(fl.sum()),
            "flip_record_W": int(((pk == win) & fl).sum()),
            "flip_record_L": int(((pk != win) & fl).sum()),
            "wins_on_feature_games": int(((pk == win) & act).sum()),
            "n_feature_games": int(act.sum()),
            "logloss": float(ll(P[an]).mean()),
            "brier": float(((P[an] - o["y"]) ** 2).mean()),
        }
        if an not in ("intercept", "served"):
            d = (ll(P["served"]) - ll(P[an])).to_numpy()
            b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(DRAWS)])
            term["arms"][an]["dLL_vs_served"] = {
                "mean": float(d.mean()),
                "lo": float(np.percentile(b, 2.5)),
                "hi": float(np.percentile(b, 97.5)),
                "pp": float((b > 0).mean()),
            }
            sb = pd.Series(d).groupby(o["season"].to_numpy()).mean()
            term["arms"][an]["seasons_positive"] = int((sb > 0).sum())
    xs = o.loc[act]
    term["feature_vs_cover_corr_feature_games"] = float(np.corrcoef(xs["x_v"], xs["y"])[0, 1])
    term["feature_vs_z_corr"] = float(np.corrcoef(xs["x_v"], xs["z"])[0, 1])
    res["term"] = term
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(main())
