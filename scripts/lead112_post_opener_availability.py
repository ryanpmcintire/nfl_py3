import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead99_forward_decision_read as D
import lead106_injury_language as L

ROOT = Path(__file__).resolve().parents[1]
SCR = ROOT / "tests" / "scratch" / "lead112"
REV = SCR / "injury_revisions_2026.parquet"
SNAPS = ROOT / "data" / "players" / "forward_snaps" / "snap_counts_2026_before_week_10.parquet"
ET = "America/New_York"
DES = ["Out", "Doubtful", "Questionable"]
FIRST, LAST = 2, 9
RESAMPLES = 4000
PERMS = 2000
SEED = 20261115
ARMS = {
    "move_only": ["move"],
    "designation": ["move", "late_d"],
    "desig_practice": ["move", "late_dp"],
}


def games():
    c = pd.read_parquet(ROOT / "artifacts/forward_ledger_core/ledger.parquet")
    c = c[c["status"].eq("graded") & c["week"].between(1, LAST)].copy()
    f = pd.DataFrame(
        {
            "game_id": c["game_id"].to_numpy(),
            "week": c["week"].astype(int).to_numpy(),
            "y": c["home_cover"].astype(float).to_numpy(),
            "served": c["served_p"].astype(float).to_numpy(),
            "deadline_utc": pd.to_datetime(c["deadline_utc"].to_numpy(), utc=True),
        }
    )
    f["away"] = f["game_id"].str.split("_").str[2]
    f["home"] = f["game_id"].str.split("_").str[3]
    et = f["deadline_utc"].dt.tz_convert(ET)
    tue = et.dt.normalize() - pd.to_timedelta((et.dt.weekday - 1) % 7, unit="D")
    f["tue_start"] = tue.dt.tz_convert("UTC")
    f["tue_noon"] = (tue + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).dt.tz_convert("UTC")
    return f


def market(f):
    q = D._quotes(f["tue_start"].min(), f["deadline_utc"].max(), set(f["game_id"]))
    f["mkt_tue"] = [D._market_at(q, r.game_id, r.tue_start, r.tue_noon) for r in f.itertuples()]
    f["mkt_dl"] = [D._market_at(q, r.game_id, r.tue_start, r.deadline_utc) for r in f.itertuples()]
    f["move"] = f["mkt_dl"] - f["mkt_tue"]
    raw = D._raw_model(f)
    f["raw_model"] = f["game_id"].map({k: v[0] for k, v in raw.items()})
    return f


def ladders():
    pop = L.id_outcomes(pd.read_parquet(L.ART / "population.parquet"))
    ver = pop[pop["verified"]].copy()
    ver["practice_status"] = ver["practice_status"].fillna("none")
    out = {}
    for w in range(FIRST, LAST + 1):
        prior = ver[ver["week"] < w]
        d = prior.groupby("designation")["played"].agg(["size", "sum"])
        dp = prior.groupby(["designation", "practice_status"])["played"].agg(["size", "sum"])
        out[w] = ((d["sum"] + 1) / (d["size"] + 2), (dp["sum"] + 1) / (dp["size"] + 2))
    return out, len(ver)


def weights():
    _, pmap = L.roster_map()
    s = pd.read_parquet(SNAPS)
    s = s[s["season"] == 2026].copy()
    s["total"] = s[["offense_snaps", "defense_snaps", "st_snaps"]].fillna(0).sum(axis=1)
    s["share"] = s[["offense_pct", "defense_pct"]].fillna(0).max(axis=1)
    s = s[s["total"] > 0].groupby(["pfr_player_id", "week"], as_index=False)["share"].max()
    out = {}
    for w in range(FIRST, LAST + 1):
        m = s[s["week"] < w].groupby("pfr_player_id")["share"].mean()
        out[w] = m
    return out, pmap


def prob(tab, w, des, prac):
    d, dp = tab[w]
    if (des, prac) in dp.index:
        return float(dp.loc[(des, prac)]), float(d.loc[des])
    return float(d.loc[des]), float(d.loc[des])


def late_terms(f, tab, wts, pmap):
    rev = pd.read_parquet(REV)
    man = json.loads(REV.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    snaps = pd.to_datetime([s["fetched_utc"] for s in man["snapshots"]], utc=True)
    rows, players = [], []
    for g in f.itertuples():
        if g.week < FIRST:
            rows.append({"game_id": g.game_id, "late_d": np.nan, "late_dp": np.nan, "nsnap": 0})
            continue
        val = {}
        nsnap = 0
        for side, team in (("home", g.home), ("away", g.away)):
            t = rev[(rev["week"] == g.week) & (rev["team"] == team)].sort_values("observed_at")
            t = t[t["observed_at"] <= g.deadline_utc]
            if t.empty:
                val[side] = (0.0, 0.0)
                continue
            t0 = t["observed_at"].min()
            nsnap = max(nsnap, int(((snaps >= t0) & (snaps <= g.deadline_utc)).sum()))
            last = t.groupby("gsis_id").tail(1).set_index("gsis_id")
            first = t[t["observed_at"] <= t0].groupby("gsis_id").tail(1).set_index("gsis_id")
            sd = sdp = 0.0
            for gid, lr in last.iterrows():
                fr = first.loc[gid] if gid in first.index else None
                states = []
                for r in (fr, lr):
                    if r is None or r["event"] == "removed":
                        states.append(("none", "none"))
                    else:
                        des = r["report_status"] if r["report_status"] in DES else "none"
                        pr = r["practice_status"] if pd.notna(r["practice_status"]) else "none"
                        states.append((des, pr))
                p0 = prob(tab, g.week, *states[0])
                p1 = prob(tab, g.week, *states[1])
                wt = float(wts[g.week].get(pmap.get(gid), 0.0))
                sd += (p0[1] - p1[1]) * wt
                sdp += (p0[0] - p1[0]) * wt
                players.append(
                    {
                        "game_id": g.game_id,
                        "team": team,
                        "gsis_id": gid,
                        "first": states[0],
                        "last": states[1],
                        "des_changed": states[0][0] != states[1][0],
                        "cell_changed": states[0] != states[1],
                        "weight": wt,
                    }
                )
            val[side] = (sd, sdp)
        rows.append(
            {
                "game_id": g.game_id,
                "late_d": val["home"][0] - val["away"][0],
                "late_dp": val["home"][1] - val["away"][1],
                "nsnap": nsnap,
            }
        )
    return f.merge(pd.DataFrame(rows), on="game_id"), pd.DataFrame(players)


def lowo(f, cols, perm=None):
    off = D._logit(f["served"].to_numpy())
    y = f["y"].to_numpy()
    x = f[cols].to_numpy(dtype=float)
    if perm is not None:
        x = x.copy()
        x[:, -1] = perm
    p = np.empty(len(f))
    coefs = {}
    for w in sorted(f["week"].unique()):
        tr = (f["week"] != w).to_numpy()
        b, _ = D._fit(x[tr], y[tr], off[tr])
        p[~tr] = D._sigmoid(off[~tr] + x[~tr] @ b)
        coefs[int(w)] = [float(v) for v in b]
    return p, coefs


def block_boot(delta, weeks, rng):
    if len(delta) == 0:
        return {"games": 0}
    wk = np.unique(weeks)
    sums = np.array([delta[weeks == w].sum() for w in wk])
    cnt = np.array([(weeks == w).sum() for w in wk])
    draws = rng.integers(0, len(wk), size=(RESAMPLES, len(wk)))
    m = sums[draws].sum(axis=1) / cnt[draws].sum(axis=1)
    return {
        "mean": float(delta.mean()),
        "low": float(np.quantile(m, 0.025)),
        "high": float(np.quantile(m, 0.975)),
        "probability_positive": float((m > 0).mean()),
        "games": len(delta),
    }


def score(f, p, rng, mask=None):
    base = f["served"].to_numpy()
    y = f["y"].to_numpy()
    m = np.ones(len(f), bool) if mask is None else mask
    out = {}
    for name, fn in (("log_loss", D._ll), ("brier", D._brier)):
        d = fn(base, y) - fn(p, y)
        out[name] = block_boot(d[m], f["week"].to_numpy()[m], rng)
    bs, ps = base >= 0.5, p >= 0.5
    dec = (bs != ps) & m
    won = ((ps == (y == 1)) & dec).sum()
    out["decisive"] = f"{int(won)}-{int(dec.sum() - won)}"
    out["record_arm"] = f"{int(((ps == (y == 1)) & m).sum())}-{int(((ps != (y == 1)) & m).sum())}"
    out["record_base"] = f"{int(((bs == (y == 1)) & m).sum())}-{int(((bs != (y == 1)) & m).sum())}"
    return out


def main():
    rng = np.random.default_rng(SEED)
    f = market(games())
    n_all = len(f)
    tab, nver = ladders()
    wts, pmap = weights()
    f, pl = late_terms(f, tab, wts, pmap)
    report = {"games_ledger_weeks_1_9": n_all, "ladder_rows": nver}
    report["excluded_week1"] = int((f["week"] < FIRST).sum())
    f = f[f["week"] >= FIRST]
    report["excluded_no_revision_history"] = int((f["nsnap"] < 2).sum())
    report["excluded_no_move"] = int(f["move"].isna().sum())
    f = f[(f["nsnap"] >= 2) & f["move"].notna()].reset_index(drop=True)
    report["eligible_games"] = len(f)
    print(report, flush=True)
    pl = pl[pl["game_id"].isin(f["game_id"])]
    report["players"] = {
        "listed_player_weeks": len(pl),
        "designation_changed_after_first_report": int(pl["des_changed"].sum()),
        "ladder_cell_changed_after_first_report": int(pl["cell_changed"].sum()),
        "games_nonzero_late_d": int((f["late_d"].abs() > 0).sum()),
        "games_nonzero_late_dp": int((f["late_dp"].abs() > 0).sum()),
        "late_d_sd": float(f["late_d"].std()),
        "late_dp_sd": float(f["late_dp"].std()),
        "move_sd": float(f["move"].std()),
        "first_to_last_transitions": pl[pl["des_changed"]]
        .assign(t=lambda x: x["first"].str[0] + ">" + x["last"].str[0])["t"]
        .value_counts()
        .to_dict(),
    }
    y = f["y"].to_numpy()
    off = D._logit(f["served"].to_numpy())
    ref = {}
    rp = f["raw_model"].to_numpy()
    keep = np.isfinite(rp)
    ref["raw_model"] = score(f[keep].reset_index(drop=True), rp[keep], rng)
    ref["raw_model"]["games"] = int(keep.sum())
    ic = np.array([y[(f["week"] != w).to_numpy()].mean() for w in f["week"]])
    ref["intercept_only"] = score(f, ic, rng)
    ref["served_base_log_loss"] = float(D._ll(f["served"].to_numpy(), y).mean())
    ref["served_base_brier"] = float(D._brier(f["served"].to_numpy(), y).mean())
    report["reference_vs_served_base"] = ref
    arms = {}
    for arm, cols in ARMS.items():
        p, coefs = lowo(f, cols)
        x = f[cols].to_numpy(dtype=float)
        b_in, cov = D._fit(x, y, off)
        p_in = D._sigmoid(off + x @ b_in)
        ll_in = float((D._ll(f["served"].to_numpy(), y) - D._ll(p_in, y)).mean())
        res = {"all": score(f, p, rng)}
        term = cols[-1]
        nz = (f[term].abs() > 0).to_numpy()
        res["nonzero_term"] = score(f, p, rng, nz)
        res["in_sample_coef"] = [float(v) for v in b_in]
        res["in_sample_se"] = [float(v) for v in np.sqrt(np.diag(cov))]
        res["in_sample_ll_gain"] = ll_in
        res["oos_minus_in_sample_gap"] = ll_in - res["all"]["log_loss"]["mean"]
        res["fold_coefs"] = coefs
        obs = res["all"]["log_loss"]["mean"]
        prng = np.random.default_rng(SEED + len(arm))
        wk = f["week"].to_numpy()
        tv = f[term].to_numpy(dtype=float)
        null = []
        for _ in range(PERMS):
            perm = tv.copy()
            for w in np.unique(wk):
                i = np.where(wk == w)[0]
                perm[i] = tv[prng.permutation(i)]
            pp, _ = lowo(f, cols, perm)
            null.append(float((D._ll(f["served"].to_numpy(), y) - D._ll(pp, y)).mean()))
        null = np.array(null)
        res["perm_null_mean"] = float(null.mean())
        res["perm_p"] = float((null >= obs).mean())
        arms[arm] = res
    report["arms"] = arms
    SCR.mkdir(parents=True, exist_ok=True)
    f.to_parquet(SCR / "games.parquet")
    (SCR / "result.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
