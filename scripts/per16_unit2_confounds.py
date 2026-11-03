import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import per16_team_designation_habit as H

ART = H.ART
GROUPS = {
    "QB": "QB",
    "RB": "RB",
    "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "T": "OL",
    "G": "OL",
    "C": "OL",
    "DE": "DL",
    "DT": "DL",
    "LB": "LB",
    "CB": "DB",
    "S": "DB",
    "K": "ST",
    "P": "ST",
    "LS": "ST",
}
PRACTICE = {
    "Full Participation in Practice": "full",
    "Limited Participation in Practice": "limited",
    "Did Not Participate In Practice": "dnp",
}
DESIG = "Questionable"
SKILL = ("QB", "RB", "WR", "TE")


def prep(x):
    x = x.copy()
    x["pg"] = x["position"].map(GROUPS).fillna("other")
    x["prac"] = x["practice_status"].map(PRACTICE).fillna("none")
    return x


def eb_level(g, keys, parent):
    t = (
        g.assign(par=parent)
        .groupby(keys)
        .agg(s=("played", "sum"), n=("played", "count"), par=("par", "mean"))
    )
    p = t["s"] / t["n"]
    var_obs = float((t["n"] * (p - t["par"]) ** 2).sum())
    pv = float((t["n"] * t["par"] * (1 - t["par"])).sum() / t["n"].sum())
    tau2 = (var_obs - float((t["par"] * (1 - t["par"])).sum())) / float(t["n"].sum())
    m = np.inf if tau2 <= 0 else max(pv / tau2 - 1, 0.0)
    post = t["par"] if np.isinf(m) else (t["s"] + m * t["par"]) / (t["n"] + m)
    return post, m, len(t)


def fit_cells(g):
    lg = float(g["played"].mean())
    p1, m1, c1 = eb_level(g, ["prac"], pd.Series(lg, index=g.index))
    par2 = g["prac"].map(p1)
    p2, m2, c2 = eb_level(g, ["prac", "pg"], par2)
    return {"lg": lg, "p1": p1, "p2": p2, "m1": m1, "m2": m2, "c1": c1, "c2": c2}


def predict_cells(f, d):
    p1 = d["prac"].map(f["p1"]).fillna(f["lg"])
    idx = pd.MultiIndex.from_frame(d[["prac", "pg"]])
    p2 = pd.Series(f["p2"].reindex(idx).to_numpy(), index=d.index).fillna(p1)
    return p1, p2


def eb_offset(train, key, base):
    g = train.dropna(subset=[key]).copy()
    g["res"] = g["played"] - g[base]
    g["pv"] = g[base] * (1 - g[base])
    u = g.groupby(key).agg(n=("res", "count"), rb=("res", "mean"), pv=("pv", "mean"))
    s2 = u["pv"] / u["n"]
    var_obs = float((u["n"] * u["rb"] ** 2).sum() / u["n"].sum())
    samp = float((u["n"] * s2).sum() / u["n"].sum())
    tau2 = max(var_obs - samp, 0.0)
    shrink = tau2 / (tau2 + s2)
    return u["rb"] * shrink, tau2


def controlled_reliability(x, rng):
    v = prep(x[x["verified"] & (x["designation"] == DESIG)])
    f = fit_cells(v)
    p1, p2 = predict_cells(f, v)
    v["base_dxp"] = p1
    v["base_ctrl"] = p2
    gc = v.dropna(subset=["coach"])
    out = {
        "rows": len(v),
        "cells_practice": f["c1"],
        "cells_practice_position": f["c2"],
        "eb_m_practice": f["m1"],
        "eb_m_practice_position": f["m2"],
    }
    for base in ("base_dxp", "base_ctrl"):
        a = v.assign(resid=v["played"] - v[base])
        b = gc.assign(resid=gc["played"] - v.loc[gc.index, base])
        out[base] = {
            "team_splithalf": split_half_resid(a, "team", rng),
            "coach_splithalf": split_half_resid(b, "coach", rng),
            "team_yoy": yoy_resid(a, ["team"], rng),
            "coach_yoy_same_team": yoy_resid(b, ["team", "coach"], rng),
        }
    return out


def split_half_resid(q, key, rng):
    q = q.copy()
    q["odd"] = q["week"] % 2 == 1

    def halves(frame):
        t = frame.groupby(["season", key, "odd"])["resid"].agg(["mean", "count"]).unstack("odd")
        t = t.dropna()
        t = t[(t[("count", True)] >= H.MIN_HALF_ROWS) & (t[("count", False)] >= H.MIN_HALF_ROWS)]
        return t[("mean", True)].to_numpy(), t[("mean", False)].to_numpy()

    a, b = halves(q)
    r, ci, pp = H.corr_ci(a, b, rng)
    null = []
    shuf = q.copy()
    for _ in range(H.PERM_REPS):
        shuf[key] = shuf.groupby("season")[key].transform(lambda s: rng.permutation(s.to_numpy()))
        a2, b2 = halves(shuf)
        null.append(np.corrcoef(a2, b2)[0, 1])
    null = np.array(null)
    return {
        "units": len(a),
        "r": r,
        "ci95": ci,
        "p_positive": pp,
        "perm_null_p95": float(np.percentile(null, 95)),
        "perm_p": float((np.sum(null >= r) + 1) / (H.PERM_REPS + 1)),
    }


def yoy_resid(q, keys, rng):
    u = q.groupby(["season", *keys])["resid"].agg(["mean", "count"]).reset_index()
    nxt = u.copy()
    nxt["season"] = nxt["season"] - 1
    m = u.merge(nxt, on=["season", *keys], suffixes=("_t", "_n"))
    m = m[(m["count_t"] >= H.MIN_YOY_ROWS) & (m["count_n"] >= H.MIN_YOY_ROWS)].reset_index(
        drop=True
    )
    r, ci, pp = H.corr_ci(m["mean_t"], m["mean_n"], rng)
    null = []
    for _ in range(H.PERM_REPS):
        sh = m.groupby("season")["mean_n"].transform(lambda s: rng.permutation(s.to_numpy()))
        null.append(np.corrcoef(m["mean_t"], sh)[0, 1])
    null = np.array(null)
    return {
        "pairs": len(m),
        "r": r,
        "ci95": ci,
        "p_positive": pp,
        "perm_null_p95": float(np.percentile(null, 95)),
        "perm_p": float((np.sum(null >= r) + 1) / (H.PERM_REPS + 1)),
    }


def stronger_eval(x):
    v = prep(x[x["verified"] & (x["designation"] == DESIG)])
    rows = []
    meta = []
    for s in range(H.FIRST_SEASON + H.MIN_PRIOR_SEASONS, H.LAST_SEASON + 1):
        train = v[v["season"] < s].copy()
        t = v[v["season"] == s].copy()
        f = fit_cells(train)
        train["p_dxp"], train["p_ctrl"] = predict_cells(f, train)
        t["p_league"] = f["lg"]
        t["p_dxp"], t["p_ctrl"] = predict_cells(f, t)
        meta.append({"season": s, "m1": f["m1"], "m2": f["m2"], "c1": f["c1"], "c2": f["c2"]})
        for base in ("dxp", "ctrl"):
            for arm in ("team", "coach"):
                off, _ = eb_offset(train, arm, f"p_{base}")
                t[f"p_{base}_{arm}"] = t["p_" + base] + t[arm].map(off).fillna(0.0)
        t["cluster"] = t["team"] + t["week"].astype(str)
        rows.append(t)
    r = pd.concat(rows)
    clip = 1.0 / (len(v) + 2)
    for c in [c for c in r.columns if c.startswith("p_")]:
        r[c] = r[c].clip(clip, 1 - clip)
    return r, pd.DataFrame(meta)


COMPARISONS = (
    ("p_league", "p_dxp"),
    ("p_dxp", "p_ctrl"),
    ("p_dxp", "p_dxp_team"),
    ("p_dxp", "p_dxp_coach"),
    ("p_dxp", "p_ctrl_team"),
    ("p_dxp", "p_ctrl_coach"),
    ("p_ctrl", "p_ctrl_team"),
    ("p_ctrl", "p_ctrl_coach"),
)


def report(r, rng):
    y = r["played"].to_numpy()
    out = []
    for base, alt in COMPARISONS:
        for metric, f in (("log_loss", lambda p: H.ll(p, y)), ("brier", lambda p: (p - y) ** 2)):
            dlt = f(r[base].to_numpy()) - f(r[alt].to_numpy())
            est, ci, pp = H.cluster_boot(dlt, r["cluster"].to_numpy(), rng)
            out.append(
                {
                    "baseline": base,
                    "alternative": alt,
                    "metric": metric,
                    "rows": len(r),
                    "gain": est,
                    "ci95": ci,
                    "p_positive": pp,
                }
            )
    return out


def cmd_confound(_):
    rng = np.random.default_rng(H.seed_value() + 10)
    x = pd.read_parquet(ART / "history.parquet")
    rel = controlled_reliability(x, rng)
    (ART / "unit2_reliability.json").write_text(json.dumps(rel, indent=1))
    print(json.dumps(rel), flush=True)
    r, meta = stronger_eval(x)
    r.to_parquet(ART / "unit2_ladder_rows.parquet")
    meta.to_csv(ART / "unit2_ladder_meta.csv", index=False)
    rep = report(r, np.random.default_rng(H.seed_value() + 11))
    (ART / "unit2_ladder.json").write_text(json.dumps(rep, indent=1))
    for c in rep:
        print(json.dumps(c))
    print(meta.to_string())


def appearances():
    root = sorted(glob.glob(str(H.ROOT / "data/pbp/raw/*")))[-1]
    frames = []
    for s in range(H.FIRST_SEASON, H.LAST_SEASON + 1):
        f = Path(root) / f"season={s}" / "plays.parquet"
        names = pq.ParquetFile(f).schema.names
        cols = [c for c in names if c.endswith("_player_id")]
        d = pd.read_parquet(f, columns=["season", "week", "season_type", *cols])
        d = d[d["season_type"] == "REG"]
        m = d.melt(id_vars=["season", "week"], value_vars=cols, value_name="gsis_id")
        frames.append(m.dropna(subset=["gsis_id"])[["season", "week", "gsis_id"]].drop_duplicates())
    a = pd.concat(frames).drop_duplicates()
    a["appear"] = 1
    return a


def cmd_selection(_):
    rng = np.random.default_rng(H.seed_value() + 12)
    x = prep(pd.read_parquet(ART / "history.parquet"))
    q = x[x["designation"] == DESIG].copy()
    q = q.merge(appearances(), on=["season", "week", "gsis_id"], how="left")
    q["appear"] = q["appear"].fillna(0).astype(int)
    q["cluster"] = q["team"] + q["week"].astype(str)
    ver = q[q["verified"]]
    out = {"rows": len(q), "verified_rows": len(ver)}
    sens = (
        ver[ver["played"] == 1]
        .groupby("pg")["appear"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "sens"})
    )
    fa = ver[ver["played"] == 0].groupby("pg")["appear"].agg(["mean", "count"])
    out["sensitivity_by_group"] = sens.round(3).to_dict("index")
    out["false_appear_by_group"] = fa.round(3).to_dict("index")
    q["cellmean"] = q.groupby(["season", "pg", "prac"])["appear"].transform("mean")
    q["aresid"] = q["appear"] - q["cellmean"]
    un = q[~q["verified"]]
    out["appear_rate_verified"] = float(ver["appear"].mean())
    out["appear_rate_unverified"] = float(un["appear"].mean())
    for label, mask in (
        ("all_groups", np.ones(len(q), bool)),
        ("high_sensitivity_groups", q["pg"].isin(SKILL).to_numpy()),
    ):
        g = q[mask]
        cm = (
            g.groupby(["season", "pg", "prac", "verified"])["appear"]
            .agg(["mean", "count"])
            .unstack("verified")
            .dropna()
        )
        cm = cm[(cm[("count", True)] > 0) & (cm[("count", False)] > 0)]
        w = cm[("count", False)].to_numpy()
        d = (cm[("mean", False)] - cm[("mean", True)]).to_numpy()
        stat = float((d * w).sum() / w.sum())
        gk = g.reset_index(drop=True)
        codes, uniq = pd.factorize(gk["cluster"])
        boots = []
        key = gk[["season", "pg", "prac"]].astype(str).agg("|".join, axis=1)
        for _ in range(H.BOOT_REPS):
            pick = rng.integers(0, len(uniq), len(uniq))
            cnt = np.bincount(pick, minlength=len(uniq))[codes]
            h = gk.assign(c=cnt)
            h = h[h["c"] > 0]
            tab = (
                h.assign(a=h["appear"] * h["c"])
                .groupby([key[h.index], h["verified"]])
                .agg(a=("a", "sum"), c=("c", "sum"))
            )
            rate = (tab["a"] / tab["c"]).unstack("verified")
            cnts = tab["c"].unstack("verified")
            ok = rate.dropna().index
            if len(ok) == 0:
                continue
            ww = cnts.loc[ok, False]
            boots.append(float(((rate.loc[ok, False] - rate.loc[ok, True]) * ww).sum() / ww.sum()))
        boots = np.array(boots)
        out[f"stratified_appear_diff_unverified_minus_verified_{label}"] = {
            "estimate": stat,
            "ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
            "p_positive": float((boots > 0).mean()),
            "strata": len(cm),
        }
    hs = q[q["pg"].isin(SKILL)]
    out["skill_groups"] = list(SKILL)
    out["implied_play_rate_by_group"] = {
        g: {
            "verified_played": float(ver.loc[ver["pg"] == g, "played"].mean()),
            "unverified_appear": float(un.loc[un["pg"] == g, "appear"].mean()),
            "unverified_implied_played": float(
                un.loc[un["pg"] == g, "appear"].mean() / sens.loc[g, "sens"]
            ),
            "unverified_rows": int((un["pg"] == g).sum()),
            "verified_rows": int((ver["pg"] == g).sum()),
        }
        for g in SKILL
    }
    out["rows_by_group_verified"] = (
        q.groupby("pg")["verified"].agg(["mean", "count"]).round(3).to_dict("index")
    )
    u = (
        hs.groupby(["season", "team", "verified"])["aresid"]
        .agg(["mean", "count"])
        .unstack("verified")
        .dropna()
    )
    u = u[(u[("count", True)] >= H.MIN_HALF_ROWS) & (u[("count", False)] >= H.MIN_HALF_ROWS)]
    a = u[("mean", True)].to_numpy()
    b = u[("mean", False)].to_numpy()
    r, ci, pp = H.corr_ci(a, b, rng)
    uu = pd.DataFrame({"season": u.index.get_level_values("season"), "a": a, "b": b})
    null = []
    for _ in range(H.PERM_REPS):
        sh = uu.groupby("season")["b"].transform(lambda s: rng.permutation(s.to_numpy()))
        null.append(np.corrcoef(uu["a"], sh)[0, 1])
    null = np.array(null)
    out["team_habit_verified_vs_unverified_appearance"] = {
        "units": len(a),
        "r": r,
        "ci95": ci,
        "p_positive": pp,
        "perm_null_p95": float(np.percentile(null, 95)),
        "perm_p": float((np.sum(null >= r) + 1) / (H.PERM_REPS + 1)),
    }
    rows = []
    for s_, g in ver.groupby("season"):
        rows.append(
            {
                "season": int(s_),
                "played": float(g["played"].mean()),
                "appear": float(g["appear"].mean()),
            }
        )
    out["verified_by_season"] = rows
    (ART / "unit2_selection.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_check2026(_):
    sys.path.insert(0, str(H.ROOT / "scripts"))
    import lead106_injury_language as L

    rng = np.random.default_rng(H.seed_value() + 13)
    hist = prep(pd.read_parquet(ART / "history.parquet"))
    hv = hist[hist["verified"] & (hist["designation"] == DESIG)].copy()
    f = fit_cells(hv)
    hv["p_dxp"], hv["p_ctrl"] = predict_cells(f, hv)
    off = {arm: eb_offset(hv, arm, "p_ctrl")[0] for arm in ("team", "coach")}
    offd = {arm: eb_offset(hv, arm, "p_dxp")[0] for arm in ("team", "coach")}
    pop = L.id_outcomes(pd.read_parquet(L.ART / "population.parquet"))
    ver = pop[pop["verified"]].copy()
    s0 = pd.read_parquet(L.ART / "sample.parquet")[["rid", "gsis_id", "week"]]
    s = s0.merge(ver.drop(columns=["rid"], errors="ignore"), on=["gsis_id", "week"], how="inner")
    reads = L.parse_reads()
    s = s[s["rid"].isin(reads)].copy()
    s = prep(s)
    hc = H.head_coaches()
    hc = hc[hc["season"] == L.SEASON].set_index("team")["coach"]
    s["coach"] = s["team"].map(hc)
    n = len(s)
    clip = 1.0 / (n + 2)
    s["rp"] = s["rid"].map(lambda q: min(max(reads[q][0], clip), 1 - clip))
    popn = ver[ver["week"] > L.FIRST_WEEK].groupby("designation").size()
    samn = s.groupby("designation").size()
    s["w"] = s["designation"].map(lambda d: popn[d] / samn[d])
    rate_other = hist[hist["verified"]].groupby(["designation", "prac"])["played"].mean()
    rate_desig = hist[hist["verified"]].groupby("designation")["played"].mean()
    q = s["designation"] == DESIG
    s["p_dxp"] = [
        rate_other.get((d, p), rate_desig.get(d, 0.5))
        for d, p in zip(s["designation"], s["prac"], strict=True)
    ]
    s["p_ctrl"] = s["p_dxp"]
    s["p_dxp_team"] = s["p_dxp"]
    s["p_ctrl_team"] = s["p_dxp"]
    s["p_ctrl_coach"] = s["p_dxp"]
    sq = s[q]
    a1, a2 = predict_cells(f, sq)
    s.loc[q, "p_dxp"] = a1
    s.loc[q, "p_ctrl"] = a2
    s.loc[q, "p_dxp_team"] = a1 + sq["team"].map(offd["team"]).fillna(0.0)
    s.loc[q, "p_ctrl_team"] = a2 + sq["team"].map(off["team"]).fillna(0.0)
    s.loc[q, "p_ctrl_coach"] = a2 + sq["coach"].map(off["coach"]).fillna(0.0)
    arms = ["p_dxp", "p_ctrl", "p_dxp_team", "p_ctrl_team", "p_ctrl_coach"]
    for c in arms:
        s[c] = s[c].clip(clip, 1 - clip)
    y = s["played"].to_numpy()
    w = s["w"].to_numpy()
    cl = (s["team"] + s["week"].astype(str)).to_numpy()
    out = []
    for arm in arms:
        for sub, mask in (("all", np.ones(n, bool)), (DESIG, q.to_numpy())):
            for metric, fn in (
                ("log_loss", lambda p: H.ll(p, y)),
                ("brier", lambda p: (p - y) ** 2),
            ):
                dlt = fn(s[arm].to_numpy()) - fn(s["rp"].to_numpy())
                est, ci, pp = H.cluster_boot(dlt[mask], cl[mask], rng, w[mask])
                out.append(
                    {
                        "ladder": arm,
                        "subset": sub,
                        "metric": metric,
                        "rows": int(mask.sum()),
                        "read_gain": est,
                        "ci95": ci,
                        "p_positive": pp,
                    }
                )
    out.append(
        {
            "q_rows": int(q.sum()),
            "coach_missing": int(sq["coach"].isna().sum()),
            "ctrl_team_range": [
                float(s.loc[q, "p_ctrl_team"].min()),
                float(s.loc[q, "p_ctrl_team"].max()),
            ],
        }
    )
    (ART / "unit2_check2026.json").write_text(json.dumps(out, indent=1))
    for o in out:
        print(json.dumps(o))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("confound", cmd_confound),
        ("selection", cmd_selection),
        ("check2026", cmd_check2026),
    ):
        sub.add_parser(name).set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
