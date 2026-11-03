import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "per16"
FIRST_SEASON = 2013
LAST_SEASON = 2025
MIN_PRIOR_SEASONS = 3
MIN_HALF_ROWS = 3
MIN_YOY_ROWS = 5
DESIGNATIONS = ("Questionable", "Doubtful", "Probable")
BOOT_REPS = 2000
PERM_REPS = 1000
SEED_SOURCE = "per16_team_designation_habit"


def seed_value():
    return int.from_bytes(SEED_SOURCE.encode()[:8], "little") % (2**32)


def latest(pattern):
    return sorted(glob.glob(str(ROOT / pattern)))[-1]


def snap_totals():
    s = pd.read_parquet(
        latest("data/players/raw/*/snap_counts.parquet"),
        columns=[
            "season",
            "game_type",
            "week",
            "pfr_player_id",
            "team",
            "offense_snaps",
            "defense_snaps",
            "st_snaps",
        ],
    )
    s = s[(s["game_type"] == "REG") & s["season"].between(FIRST_SEASON, LAST_SEASON)].copy()
    s["total"] = s[["offense_snaps", "defense_snaps", "st_snaps"]].fillna(0).sum(axis=1)
    tw = s[["season", "team", "week"]].drop_duplicates().assign(tw_ok=True)
    pl = s.groupby(["season", "week", "pfr_player_id"], as_index=False).agg(total=("total", "sum"))
    return pl, tw


def head_coaches():
    d = pd.read_parquet(latest("artifacts/staff_state/*/staff_state.parquet"))
    h = d[(d["role"] == "HC") & d["person"].notna()].sort_values("first_observed_at")
    h = h.drop_duplicates(["season", "team"], keep="first")
    return h[["season", "team", "person"]].rename(columns={"person": "coach"})


def history():
    inj = pd.read_parquet(latest("data/raw/nflverse_injuries/*/injuries.parquet"))
    inj = inj[(inj["game_type"] == "REG") & inj["season"].between(FIRST_SEASON, LAST_SEASON)]
    inj = inj.sort_values("date_modified").drop_duplicates(
        ["season", "week", "gsis_id"], keep="last"
    )
    inj["designation"] = inj["report_status"].fillna("none")
    ros = pd.read_parquet(
        latest("data/players/raw/*/weekly_rosters.parquet"),
        columns=["season", "gsis_id", "pfr_id"],
    ).dropna()
    ros = ros.drop_duplicates(["season", "gsis_id"], keep="last")
    x = inj.merge(ros, on=["season", "gsis_id"], how="left")
    pl, tw = snap_totals()
    x = x.merge(tw, on=["season", "team", "week"], how="left")
    x = x.merge(
        pl,
        left_on=["season", "week", "pfr_id"],
        right_on=["season", "week", "pfr_player_id"],
        how="left",
    )
    x["verified"] = x["pfr_id"].notna() & x["tw_ok"].fillna(False).astype(bool)
    x["played"] = (x["total"].fillna(0) > 0).astype(int)
    x = x.merge(head_coaches(), on=["season", "team"], how="left")
    return x


def coverage(x):
    out = {
        "snap_season_range": [FIRST_SEASON, LAST_SEASON],
        "injury_rows": len(x),
        "verified_rate": float(x["verified"].mean()),
        "verified_rate_by_season": x.groupby("season")["verified"].mean().round(3).to_dict(),
        "coach_missing_rate": float(x["coach"].isna().mean()),
        "by_designation": {},
    }
    for d in (*DESIGNATIONS, "Out", "none"):
        g = x[x["designation"] == d]
        v = g[g["verified"]]
        out["by_designation"][d] = {
            "rows": len(g),
            "verified": len(v),
            "play_rate": float(v["played"].mean()) if len(v) else None,
        }
    return out


def eb_tau2(k, n):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    keep = n > 0
    k, n = k[keep], n[keep]
    pbar = k.sum() / n.sum()
    p = k / n
    var_obs = np.sum(n * (p - pbar) ** 2) / np.sum(n)
    samp = pbar * (1 - pbar) * len(n) / np.sum(n)
    tau2 = var_obs - samp
    m = np.inf if tau2 <= 0 else max(pbar * (1 - pbar) / tau2 - 1, 0.0)
    return float(pbar), float(var_obs), float(samp), float(tau2), float(m)


def corr_ci(a, b, rng):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    r = float(np.corrcoef(a, b)[0, 1])
    idx = rng.integers(0, len(a), (BOOT_REPS, len(a)))
    boots = np.array([np.corrcoef(a[i], b[i])[0, 1] for i in idx])
    boots = boots[np.isfinite(boots)]
    return r, np.percentile(boots, [2.5, 97.5]).tolist(), float((boots > 0).mean())


def split_half(g, key, rng):
    q = g.copy()
    q["odd"] = q["week"] % 2 == 1
    lg = q.groupby(["season", "odd"])["played"].mean().rename("lg")
    q = q.join(lg, on=["season", "odd"])
    q["resid"] = q["played"] - q["lg"]

    def halves(frame):
        t = frame.groupby(["season", key, "odd"])["resid"].agg(["mean", "count"]).unstack("odd")
        t = t.dropna()
        t = t[(t[("count", True)] >= MIN_HALF_ROWS) & (t[("count", False)] >= MIN_HALF_ROWS)]
        return t[("mean", True)].to_numpy(), t[("mean", False)].to_numpy()

    a, b = halves(q)
    r, ci, pp = corr_ci(a, b, rng)
    null = []
    shuf = q.copy()
    for _ in range(PERM_REPS):
        shuf[key] = shuf.groupby("season")[key].transform(lambda s: rng.permutation(s.to_numpy()))
        a2, b2 = halves(shuf)
        null.append(np.corrcoef(a2, b2)[0, 1])
    null = np.array(null)
    return {
        "units": len(a),
        "r": r,
        "ci95": ci,
        "p_positive": pp,
        "spearman_brown": 2 * r / (1 + r),
        "perm_null_mean": float(null.mean()),
        "perm_null_p95": float(np.percentile(null, 95)),
        "perm_p": float((np.sum(null >= r) + 1) / (PERM_REPS + 1)),
    }


def yoy(g, keys, rng):
    q = g.copy()
    lg = q.groupby("season")["played"].mean().rename("lg")
    q = q.join(lg, on="season")
    q["resid"] = q["played"] - q["lg"]
    u = q.groupby(["season", *keys])["resid"].agg(["mean", "count"]).reset_index()
    nxt = u.copy()
    nxt["season"] = nxt["season"] - 1
    m = u.merge(nxt, on=["season", *keys], suffixes=("_t", "_n"))
    m = m[(m["count_t"] >= MIN_YOY_ROWS) & (m["count_n"] >= MIN_YOY_ROWS)].reset_index(drop=True)
    r, ci, pp = corr_ci(m["mean_t"], m["mean_n"], rng)
    null = []
    for _ in range(PERM_REPS):
        sh = m["mean_n"].copy()
        sh = m.groupby("season")["mean_n"].transform(lambda s: rng.permutation(s.to_numpy()))
        null.append(np.corrcoef(m["mean_t"], sh)[0, 1])
    null = np.array(null)
    return {
        "pairs": len(m),
        "r": r,
        "ci95": ci,
        "p_positive": pp,
        "perm_null_p95": float(np.percentile(null, 95)),
        "perm_p": float((np.sum(null >= r) + 1) / (PERM_REPS + 1)),
    }


def posterior_table(train, key, designation):
    g = train[train["designation"] == designation].dropna(subset=[key])
    t = g.groupby(key)["played"].agg(["sum", "count"])
    lg = float(g["played"].mean())
    _, _, _, tau2, m = eb_tau2(t["sum"], t["count"])
    post = pd.Series(lg, index=t.index) if np.isinf(m) else (t["sum"] + m * lg) / (t["count"] + m)
    return lg, m, tau2, post


def ladder_eval(x):
    v = x[x["verified"]].copy()
    rows = []
    meta = []
    for s in range(FIRST_SEASON + MIN_PRIOR_SEASONS, LAST_SEASON + 1):
        train = v[v["season"] < s]
        test = v[v["season"] == s]
        for d in DESIGNATIONS:
            t = test[test["designation"] == d].copy()
            if t.empty:
                continue
            clip = 1.0 / (int((train["designation"] == d).sum()) + 2)
            t["p_league"] = float(train.loc[train["designation"] == d, "played"].mean())
            for arm in ("team", "coach"):
                lgd, m, tau2, post = posterior_table(train, arm, d)
                t[f"p_{arm}"] = t[arm].map(post).fillna(lgd)
                meta.append(
                    {
                        "season": s,
                        "designation": d,
                        "arm": arm,
                        "m": None if np.isinf(m) else m,
                        "tau2": tau2,
                    }
                )
            for c in ("p_league", "p_team", "p_coach"):
                t[c] = t[c].clip(clip, 1 - clip)
            t["cluster"] = t["team"] + t["week"].astype(str)
            rows.append(t)
    return pd.concat(rows), pd.DataFrame(meta)


def ll(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def cluster_boot(delta, cluster, rng, weight=None):
    w = np.ones(len(delta)) if weight is None else weight
    codes, uniq = pd.factorize(cluster)
    num = np.bincount(codes, weights=delta * w, minlength=len(uniq))
    den = np.bincount(codes, weights=w, minlength=len(uniq))
    idx = rng.integers(0, len(uniq), (BOOT_REPS, len(uniq)))
    est = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return (
        float(num.sum() / den.sum()),
        np.percentile(est, [2.5, 97.5]).tolist(),
        float((est > 0).mean()),
    )


def ladder_report(r, rng):
    out = []
    for d in DESIGNATIONS:
        g = r[r["designation"] == d]
        y = g["played"].to_numpy()
        for arm in ("team", "coach"):
            for metric, f in (
                ("log_loss_improvement", lambda p, y=y: ll(p, y)),
                ("brier_improvement", lambda p, y=y: (p - y) ** 2),
            ):
                dlt = f(g["p_league"].to_numpy()) - f(g[f"p_{arm}"].to_numpy())
                est, ci, pp = cluster_boot(dlt, g["cluster"].to_numpy(), rng)
                out.append(
                    {
                        "designation": d,
                        "arm": arm,
                        "metric": metric,
                        "rows": len(g),
                        "estimate": est,
                        "ci95": ci,
                        "p_positive": pp,
                    }
                )
    return out


def cmd_reliability(_):
    rng = np.random.default_rng(seed_value())
    x = history()
    res = {"coverage": coverage(x), "designations": {}}
    v = x[x["verified"]]
    for d in DESIGNATIONS:
        g = v[v["designation"] == d]
        gc = g.dropna(subset=["coach"])
        tu = g.groupby(["season", "team"])["played"].agg(["sum", "count"])
        pbar, vo, sp, tau2, m = eb_tau2(tu["sum"], tu["count"])
        cu = gc.groupby(["season", "team", "coach"])["played"].agg(["sum", "count"])
        entry = {
            "team_season_units": len(tu),
            "mean_n_per_unit": float(tu["count"].mean()),
            "league_rate": pbar,
            "team_season_eb": {
                "var_obs": vo,
                "var_sampling": sp,
                "tau2": tau2,
                "m": None if np.isinf(m) else m,
                "shrink_at_mean_n": None
                if np.isinf(m)
                else float(tu["count"].mean() / (tu["count"].mean() + m)),
            },
            "coach_season_tau2": eb_tau2(cu["sum"], cu["count"])[3],
            "team_splithalf": split_half(g, "team", rng),
            "coach_splithalf": split_half(gc, "coach", rng),
            "team_yoy": yoy(g, ["team"], rng),
            "coach_yoy_any_team": yoy(gc, ["coach"], rng),
            "coach_yoy_same_team": yoy(gc, ["team", "coach"], rng),
        }
        res["designations"][d] = entry
        print(d, json.dumps(entry), flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "reliability.json").write_text(json.dumps(res, indent=1))
    x.to_parquet(ART / "history.parquet")


def cmd_ladder(_):
    rng = np.random.default_rng(seed_value() + 1)
    x = pd.read_parquet(ART / "history.parquet")
    r, meta = ladder_eval(x)
    rep = ladder_report(r, rng)
    r.to_parquet(ART / "ladder_rows.parquet")
    meta.to_csv(ART / "ladder_meta.csv", index=False)
    (ART / "ladder.json").write_text(json.dumps(rep, indent=1))
    for c in rep:
        print(json.dumps(c))


def cmd_check2026(_):
    sys.path.insert(0, str(ROOT / "scripts"))
    import lead106_injury_language as L

    rng = np.random.default_rng(seed_value() + 2)
    v = pd.read_parquet(ART / "history.parquet")
    v = v[v["verified"]]
    pop = L.id_outcomes(pd.read_parquet(L.ART / "population.parquet"))
    ver = pop[pop["verified"]].copy()
    s0 = pd.read_parquet(L.ART / "sample.parquet")[["rid", "gsis_id", "week"]]
    s = s0.merge(ver.drop(columns=["rid"], errors="ignore"), on=["gsis_id", "week"], how="inner")
    reads = L.parse_reads()
    s = s[s["rid"].isin(reads)].copy()
    n = len(s)
    clip = 1.0 / (n + 2)
    s["rp"] = s["rid"].map(lambda q: min(max(reads[q][0], clip), 1 - clip))
    popn = ver[ver["week"] > L.FIRST_WEEK].groupby("designation").size()
    samn = s.groupby("designation").size()
    s["w"] = s["designation"].map(lambda d: popn[d] / samn[d])
    s["p_league"] = np.nan
    s["p_team"] = np.nan
    for d in sorted(s["designation"].unique()):
        mk = s["designation"] == d
        h = v[v["designation"] == d]
        lg = float(h["played"].mean()) if len(h) else 0.5
        s.loc[mk, "p_league"] = lg
        s.loc[mk, "p_team"] = lg
        if d in DESIGNATIONS:
            lgd, _, _, post = posterior_table(v, "team", d)
            s.loc[mk, "p_team"] = s.loc[mk, "team"].map(post).fillna(lgd)
    for c in ("p_league", "p_team"):
        s[c] = s[c].clip(clip, 1 - clip)
    y = s["played"].to_numpy()
    w = s["w"].to_numpy()
    cl = (s["team"] + s["week"].astype(str)).to_numpy()
    out = []
    for arm in ("p_league", "p_team"):
        for sub, mask in (
            ("all", np.ones(n, bool)),
            ("Questionable", (s["designation"] == "Questionable").to_numpy()),
        ):
            dlt = ll(s[arm].to_numpy(), y) - ll(s["rp"].to_numpy(), y)
            est, ci, pp = cluster_boot(dlt[mask], cl[mask], rng, w[mask])
            out.append(
                {
                    "ladder": arm,
                    "subset": sub,
                    "rows": int(mask.sum()),
                    "read_gain_log_loss": est,
                    "ci95": ci,
                    "p_positive": pp,
                }
            )
    q = s[s["designation"] == "Questionable"]
    out.append(
        {
            "q_rows": len(q),
            "q_played": float(q["played"].mean()),
            "team_ladder_min_max": [float(q["p_team"].min()), float(q["p_team"].max())],
        }
    )
    (ART / "check2026.json").write_text(json.dumps(out, indent=1))
    for o in out:
        print(json.dumps(o))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("reliability", cmd_reliability),
        ("ladder", cmd_ladder),
        ("check2026", cmd_check2026),
    ):
        sub.add_parser(name).set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
