from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PBP_ROOT = REPO / "data" / "pbp" / "raw"
INJ_PATH = REPO / "data" / "raw" / "nflverse_injuries"
OPEN_PATH = REPO / "artifacts" / "opener_evaluation" / "20261101T151614Z" / "per_game.parquet"
MIN_MISSED = 3
RETURN_GAMES = (1, 2)
FAMILY = "per12_qb_return"
DRAWS = 4000
SEED = 20261103
OPENER_SEASONS = (2020, 2025)


def newest(root):
    return sorted(p.name for p in root.iterdir() if p.is_dir())[-1]


def load_pbp():
    stamp = newest(PBP_ROOT)
    frames = []
    cols = [
        "game_id",
        "season",
        "season_type",
        "week",
        "home_team",
        "away_team",
        "posteam",
        "qb_dropback",
        "passer_player_id",
        "passer_player_name",
        "epa",
    ]
    for path in sorted(glob.glob(str(PBP_ROOT / stamp / "season=*" / "*.parquet"))):
        s = int(os.path.basename(os.path.dirname(path)).split("=")[1])
        if s < 2009:
            continue
        d = pd.read_parquet(path, columns=cols)
        d = d.loc[
            (d["season_type"] == "REG")
            & (d["qb_dropback"] == 1)
            & d["passer_player_id"].notna()
            & d["epa"].notna()
        ]
        frames.append(d)
    return pd.concat(frames, ignore_index=True), stamp


def build_events(pbp):
    per = pbp.groupby(
        ["game_id", "season", "week", "posteam", "passer_player_id"], as_index=False
    ).agg(
        passer_player_name=("passer_player_name", "first"), db=("epa", "size"), epa=("epa", "sum")
    )
    per = per.sort_values(["game_id", "posteam", "db"], ascending=[True, True, False])
    starter = (
        per.groupby(["game_id", "posteam"], as_index=False)
        .head(1)
        .rename(columns={"passer_player_id": "sid"})
    )
    team_db = per.groupby(["game_id", "posteam"], as_index=False).agg(tdb=("db", "sum"))
    tg = (
        starter.merge(team_db, on=["game_id", "posteam"])
        .sort_values(["posteam", "season", "week"])
        .reset_index(drop=True)
    )
    events = []
    for (team, season), g in tg.groupby(["posteam", "season"]):
        g = g.reset_index(drop=True)
        sids = g["sid"].to_numpy()
        n = len(g)
        for k in range(n - 1):
            q = sids[k]
            if sids[k + 1] == q:
                continue
            m = 1
            while k + 1 + m < n and sids[k + 1 + m] != q:
                m += 1
            if k + 1 + m >= n or m < MIN_MISSED:
                continue
            events.append(
                {
                    "team": team,
                    "season": season,
                    "qb": q,
                    "qb_name": g.loc[k, "passer_player_name"],
                    "missed": m,
                    "abs_games": list(g.loc[k + 1 : k + m, "game_id"]),
                    "abs_weeks": list(g.loc[k + 1 : k + m, "week"]),
                    "ret_games": list(g.loc[k + 1 + m :, "game_id"]),
                    "ret_weeks": list(g.loc[k + 1 + m :, "week"]),
                }
            )
    return pd.DataFrame(events), per, tg


def boot(x, cluster, rng, w=None):
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    ids = pd.unique(cluster)
    idx = {c: np.where(cluster == c)[0] for c in ids}
    est = float(np.sum(x * w) / np.sum(w))
    draws = np.empty(DRAWS)
    for i in range(DRAWS):
        pick = rng.choice(len(ids), len(ids))
        sel = np.concatenate([idx[ids[j]] for j in pick])
        draws[i] = np.sum(x[sel] * w[sel]) / np.sum(w[sel])
    return {
        "mean": est,
        "se": float(draws.std(ddof=1)),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "pp": float((draws > 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "tests" / "scratch" / "per12"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pbp, stamp = load_pbp()
    ev, per, tg = build_events(pbp)
    inj = pd.read_parquet(INJ_PATH / newest(INJ_PATH) / "injuries.parquet")
    inj = inj.loc[(inj["position"] == "QB") & (inj["game_type"] == "REG")]
    inj_key = {}
    for r in inj.itertuples():
        inj_key.setdefault((r.gsis_id, r.season), {})[int(r.week)] = str(r.report_status)
    per_idx = per.set_index(["game_id", "posteam", "passer_player_id"])
    rows = []
    for e in ev.itertuples():
        wk0 = e.abs_weeks[0]
        pre = per[
            (per["passer_player_id"] == e.qb)
            & ((per["season"] < e.season) | ((per["season"] == e.season) & (per["week"] < wk0)))
        ]
        base = pre["epa"].sum() / pre["db"].sum() if pre["db"].sum() > 0 else np.nan
        bk = per[
            per["game_id"].isin(e.abs_games)
            & (per["posteam"] == e.team)
            & (per["passer_player_id"] != e.qb)
        ]
        bepa = bk["epa"].sum() / bk["db"].sum() if bk["db"].sum() > 0 else np.nan
        tdb_abs = tg[tg["game_id"].isin(e.abs_games) & (tg["posteam"] == e.team)]["tdb"].mean()
        stat = [inj_key.get((e.qb, e.season), {}).get(w) for w in e.abs_weeks]
        stat = [s for s in stat if s]
        reason = (
            "injury_report"
            if any(s in ("Out", "Doubtful", "Questionable") for s in stat)
            else ("other_status" if stat else "no_report")
        )
        for j, gid in enumerate(e.ret_games[:4]):
            key = (gid, e.team, e.qb)
            gdb, gepa = (
                (per_idx.loc[key, "db"], per_idx.loc[key, "epa"])
                if key in per_idx.index
                else (0, 0.0)
            )
            srow = tg[(tg["game_id"] == gid) & (tg["posteam"] == e.team)]
            rows.append(
                {
                    "team": e.team,
                    "season": e.season,
                    "qb": e.qb,
                    "qb_name": e.qb_name,
                    "missed": e.missed,
                    "reason": reason,
                    "ret_n": j + 1,
                    "week": e.ret_weeks[j],
                    "game_id": gid,
                    "db": gdb,
                    "epa_db": gepa / gdb if gdb else np.nan,
                    "baseline": base,
                    "backup_epa_db": bepa,
                    "abs_team_db": tdb_abs,
                    "started": int(srow["sid"].iloc[0] == e.qb),
                }
            )
    r = pd.DataFrame(rows)
    ev_tab = r[r["ret_n"] == 1][
        ["team", "season", "week", "qb_name", "missed", "reason", "baseline", "backup_epa_db"]
    ]
    ev_tab.to_csv(out / "events.csv", index=False)
    result = {
        "pbp_stamp": stamp,
        "events_2009_2025": len(ev_tab),
        "by_reason": ev_tab["reason"].value_counts().to_dict(),
        "missed_dist": {
            int(k): int(v) for k, v in ev_tab["missed"].value_counts().sort_index().items()
        },
        "events_2020_2025": int(ev_tab["season"].between(*OPENER_SEASONS).sum()),
    }
    cells = {}
    for n in (1, 2, 3, 4):
        s = r[(r["ret_n"] == n) & (r["started"] == 1) & r["baseline"].notna() & (r["db"] > 0)]
        sb = s[s["backup_epa_db"].notna()]
        cells[f"return{n}"] = {
            "n": len(s),
            "vs_own_baseline": boot(
                (s["epa_db"] - s["baseline"]).to_numpy(),
                s["qb"].to_numpy(),
                rng,
                s["db"].to_numpy(),
            ),
            "n_backup": len(sb),
            "vs_backup": boot(
                (sb["epa_db"] - sb["backup_epa_db"]).to_numpy(),
                sb["qb"].to_numpy(),
                rng,
                sb["db"].to_numpy(),
            ),
        }
    result["epa_cells"] = cells
    o = pd.read_parquet(OPEN_PATH)
    ids = pbp.drop_duplicates("game_id")[["game_id", "home_team", "away_team"]]
    sched = pd.concat(
        [ids.assign(team=ids["home_team"], is_home=1), ids.assign(team=ids["away_team"], is_home=0)]
    )
    sched = sched.merge(
        o[["game_id", "season", "week", "tue_open_home_spread", "close_home_spread", "result"]],
        on="game_id",
    )
    sg = np.where(sched["is_home"] == 1, 1.0, -1.0)
    sched["t_open"] = sg * sched["tue_open_home_spread"]
    sched["t_close"] = sg * sched["close_home_spread"]
    sched["cover"] = sg * sched["result"] - sched["t_open"]
    sched = sched.sort_values(["team", "season", "week"])
    sched["prev_close"] = sched.groupby(["team", "season"])["t_close"].shift(1)
    sched["prev_home"] = sched.groupby(["team", "season"])["is_home"].shift(1)
    sched["move"] = sched["t_open"] - sched["prev_close"]
    sched["pattern"] = sched["is_home"].astype(str) + sched["prev_home"].astype(str)
    rk = r[r["ret_n"].isin(RETURN_GAMES) & (r["started"] == 1)][
        ["game_id", "team", "ret_n", "backup_epa_db", "baseline", "abs_team_db", "qb"]
    ]
    sched = sched.merge(rk, on=["game_id", "team"], how="left")
    sched["is_event"] = sched["ret_n"].notna()
    ctrl = sched[~sched["is_event"] & sched["move"].notna()].groupby("pattern")["move"].mean()
    result["control_mean_cover_non_event"] = float(sched.loc[~sched["is_event"], "cover"].mean())
    evg = sched[sched["is_event"]].copy()
    evg["move_adj"] = evg["move"] - evg["pattern"].map(ctrl)
    evg["ev_swap"] = (evg["baseline"] - evg["backup_epa_db"]) * evg["abs_team_db"]
    evg.to_csv(out / "return_games_opener.csv", index=False)
    dec = {}
    for name, s in (
        ("return1", evg[evg["ret_n"] == 1]),
        ("return2", evg[evg["ret_n"] == 2]),
        ("return1_2", evg),
    ):
        dec[name] = {
            "n": len(s),
            "cover": int((s["cover"] > 0).sum()),
            "loss": int((s["cover"] < 0).sum()),
            "push": int((s["cover"] == 0).sum()),
        }
    result["decisive_record_vs_opener_2020_2025"] = dec
    result["cover_vs_opener_ret12"] = boot(evg["cover"].to_numpy(), evg["qb"].to_numpy(), rng)
    mv = evg[evg["move_adj"].notna()]
    result["move_adj_ret12"] = dict(
        n=len(mv), **boot(mv["move_adj"].to_numpy(), mv["qb"].to_numpy(), rng)
    )
    ok = mv[mv["ev_swap"].notna()]
    X = np.c_[np.ones(len(ok)), ok["ev_swap"].to_numpy()]
    y = ok["move_adj"].to_numpy()
    cl = ok["qb"].to_numpy()
    uq = pd.unique(cl)
    sl = []
    for _ in range(DRAWS):
        pick = rng.choice(len(uq), len(uq))
        sel = np.concatenate([np.where(cl == uq[j])[0] for j in pick])
        sl.append(np.linalg.lstsq(X[sel], y[sel], rcond=None)[0][1])
    sl = np.array(sl)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    result["move_on_ev_swap"] = {
        "n": len(ok),
        "slope": float(beta[1]),
        "intercept": float(beta[0]),
        "se": float(sl.std(ddof=1)),
        "lo": float(np.percentile(sl, 2.5)),
        "hi": float(np.percentile(sl, 97.5)),
        "pp_slope_positive": float((sl > 0).mean()),
        "mean_ev_swap": float(ok["ev_swap"].mean()),
    }
    cb = boot(ok["cover"].to_numpy(), ok["qb"].to_numpy(), rng)
    result["cover_ret12_with_ev"] = cb
    result["cover_vs_ev_corr"] = (
        float(np.corrcoef(ok["cover"], ok["ev_swap"])[0, 1]) if len(ok) > 2 else None
    )
    allg = sched[sched["cover"].notna()].copy()
    allg["ret"] = allg["is_event"].astype(float)
    h = allg[allg["is_home"] == 1][["game_id", "season", "cover", "ret"]]
    a = allg[allg["is_home"] == 0][["game_id", "ret"]].rename(columns={"ret": "ret_away"})
    hg = h.merge(a, on="game_id")
    hg["xd"] = hg["ret"] - hg["ret_away"]
    hg["y"] = hg["cover"]
    loso = []
    for s in sorted(hg["season"].unique()):
        tr = hg[hg["season"] != s]
        te = hg[hg["season"] == s]
        b = np.linalg.lstsq(np.c_[np.ones(len(tr)), tr["xd"]], tr["y"].to_numpy(), rcond=None)[0]
        loso.append(
            {
                "season": int(s),
                "coef": float(b[1]),
                "mse_model": float(((te["y"] - (b[0] + b[1] * te["xd"])) ** 2).mean()),
                "mse_intercept": float(((te["y"] - tr["y"].mean()) ** 2).mean()),
            }
        )
    lo = pd.DataFrame(loso)
    bf = np.linalg.lstsq(np.c_[np.ones(len(hg)), hg["xd"]], hg["y"].to_numpy(), rcond=None)[0]
    result["loso_term_diagnostic"] = {
        "folds": loso,
        "oos_mse_model": float(lo["mse_model"].mean()),
        "oos_mse_intercept": float(lo["mse_intercept"].mean()),
        "in_sample_coef": float(bf[1]),
        "in_sample_mse_model": float(((hg["y"] - bf[0] - bf[1] * hg["xd"]) ** 2).mean()),
        "in_sample_mse_intercept": float(((hg["y"] - hg["y"].mean()) ** 2).mean()),
    }
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(main())
