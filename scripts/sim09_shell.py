import glob
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sim09_personnel import ratings_table  # noqa: E402
from sim09_urgency import cluster_ols  # noqa: E402

OUT = ROOT / "artifacts" / "sim09"
PBP = sorted(glob.glob(str(ROOT / "data/pbp/raw/*")))[-1]
PART = ROOT / "data/players/participation/raw/20260813T131635Z"
CACHE = Path("C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/47c38f23-38ab-4374-afcc-d56f955b121f/scratchpad/shell_plays.parquet")
W_BOX = ((2016, 2021), (2022, 2025))
W_COV = ((2018, 2021), (2022, 2025))
TWO_HIGH = ["COVER_2", "COVER_4", "2_MAN"]
ONE_HIGH = ["COVER_1", "COVER_3", "COVER_0"]
STATE = ["sd", "zs", "tsec", "half_sec", "qtr", "down", "ydstogo", "yl", "home", "to_off", "to_def"]
STRENGTH = ["PO", "PD", "eo", "ed"]
FEATS = STATE + STRENGTH
GRID = [(8, 100), (8, 500), (31, 100), (31, 500)]
LOOKS = {"policy": 0, "outcome": 0, "decomp": 0}


def db_count(s):
    if not isinstance(s, str):
        return np.nan
    return float(sum(int(n) for n, p in re.findall(r"(\d+) (\w+)", s) if p in ("CB", "S", "FS", "SS", "DB")))


def build():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    cols = ["game_id", "play_id", "season", "season_type", "week", "home_team", "posteam", "defteam", "down", "play_type", "yards_gained", "rush_attempt", "sack",
            "epa", "qtr", "ydstogo", "yardline_100", "game_seconds_remaining", "qb_dropback", "qb_kneel", "qb_spike", "complete_pass", "interception", "touchdown",
            "fixed_drive", "fixed_drive_result", "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining"]
    fr = []
    for s in range(2016, 2026):
        x = pd.read_parquet(f"{PBP}/season={s}/plays.parquet", columns=cols)
        fr.append(x[x.season_type == "REG"])
    d = pd.concat(fr, ignore_index=True)
    d = d[d.posteam.notna() & d.play_type.isin(["pass", "run"]) & (d.qb_kneel != 1) & (d.qb_spike != 1) & (d.qtr <= 4) & d.down.notna() & d.score_differential.notna()].copy()
    pf = []
    for s in range(2016, 2026):
        x = pd.read_parquet(PART / f"season={s}" / "participation.parquet", columns=["game_id", "play_id", "defenders_in_box", "defense_personnel", "defense_coverage_type", "defense_man_zone_type"])
        pf.append(x)
    pt = pd.concat(pf, ignore_index=True)
    pt["play_id"] = pt.play_id.astype(d.play_id.dtype)
    pt["ndb"] = pt.defense_personnel.map(db_count)
    pt = pt.drop(columns="defense_personnel").drop_duplicates(["game_id", "play_id"])
    n0 = len(d)
    d = d.merge(pt, on=["game_id", "play_id"], how="left")
    t, _ = ratings_table()
    d = d.merge(t[["season", "week", "team", "PO", "RB", "epa_all_off"]].rename(columns={"team": "posteam", "epa_all_off": "eo"}), on=["season", "week", "posteam"], how="left")
    d = d.merge(t[["season", "week", "team", "PD", "RD", "epa_all_def"]].rename(columns={"team": "defteam", "epa_all_def": "ed"}), on=["season", "week", "defteam"], how="left")
    assert len(d) == n0
    d["tsec"] = d.game_seconds_remaining.clip(lower=0)
    d["half_sec"] = pd.Series(np.where(d.qtr <= 2, d.tsec - 1800, d.tsec), index=d.index).clip(lower=0)
    d["sd"] = d.score_differential.astype(float)
    d["zs"] = d.sd / np.sqrt(d.tsec / 60.0 + 1.0)
    d["yl"] = d.yardline_100 / 100.0
    d["home"] = (d.posteam == d.home_team).astype(float)
    d["to_off"] = d.posteam_timeouts_remaining.astype(float)
    d["to_def"] = d.defteam_timeouts_remaining.astype(float)
    d["down"] = d.down.astype(float)
    d["isp"] = ((d.play_type == "pass") & (d.qb_dropback == 1)).astype(float)
    cov = d.defense_coverage_type.astype("object")
    d["two"] = np.where(cov.isin(TWO_HIGH), 1.0, np.where(cov.isin(ONE_HIGH), 0.0, np.nan))
    mz = d.defense_man_zone_type.astype("object")
    d["man"] = np.where(mz == "MAN_COVERAGE", 1.0, np.where(mz == "ZONE_COVERAGE", 0.0, np.nan))
    d["box"] = d.defenders_in_box.astype(float)
    d["ndb"] = d.ndb.astype(float)
    d.drop(columns=["defense_coverage_type", "defense_man_zone_type", "defenders_in_box"]).to_parquet(CACHE)
    return pd.read_parquet(CACHE)


def split(d, w):
    return d[d.season.between(*w[0])], d[d.season.between(*w[1])]


def fit_clf(Xa, ya, Xv, yv, Xb=None):
    best = None
    for leaf, msl in GRID:
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=leaf, min_samples_leaf=msl, early_stopping=False, random_state=0)
        m.fit(Xa, ya)
        ll = [log_loss(yv, p, labels=m.classes_) for p in m.staged_predict_proba(Xv)]
        k = int(np.argmin(ll))
        if best is None or ll[k] < best[0]:
            best = (ll[k], leaf, msl, k + 1)
    return best


def refit(X, y, cfg):
    _, leaf, msl, it = cfg
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=it, max_leaf_nodes=leaf, min_samples_leaf=msl, early_stopping=False, random_state=0)
    return m.fit(X, y)


def policy(d):
    d = d.copy()
    lo, hi = d.loc[d.season.between(*W_BOX[0]), "box"].quantile([0.025, 0.975]).tolist()
    bx = d.box.clip(lo, hi).round()
    d["box_c"] = bx
    dl_, dh_ = d.loc[d.season.between(*W_BOX[0]), "ndb"].quantile([0.025, 0.975]).tolist()
    d["db_c"] = d.ndb.clip(dl_, dh_).round()
    targets = {"two": ("two", W_COV, d.isp == 1), "man": ("man", W_COV, d.isp == 1), "db": ("db_c", W_BOX, d.isp.notna()), "box": ("box_c", W_BOX, d.isp.notna())}
    abl = [c for c in FEATS if c not in ("sd", "zs", "tsec", "half_sec", "qtr", "to_off", "to_def")]
    rows = []
    art = {"features": FEATS, "box_clip": [lo, hi], "db_clip": [dl_, dh_], "models": {}, "priors": {}, "windows": {"box": W_BOX, "cov": W_COV}}
    for name, (col, w, mask) in targets.items():
        s = d[mask & d[col].notna() & d[FEATS].notna().all(axis=1)]
        fit, held = split(s, w)
        inner = fit[fit.season < w[0][1]]
        val = fit[fit.season == w[0][1]]
        cfg = fit_clf(inner[FEATS], inner[col], val[FEATS], val[col])
        m = refit(fit[FEATS], fit[col], cfg)
        LOOKS["policy"] += 1
        prior = fit[col].value_counts(normalize=True).sort_index()
        pb = np.tile(prior.reindex(m.classes_).values, (len(held), 1))
        ll_m = log_loss(held[col], m.predict_proba(held[FEATS]), labels=m.classes_)
        ll_b = log_loss(held[col], pb, labels=m.classes_)
        cfg2 = fit_clf(inner[abl], inner[col], val[abl], val[col])
        m2 = refit(fit[abl], fit[col], cfg2)
        ll_a = log_loss(held[col], m2.predict_proba(held[abl]), labels=m2.classes_)
        LOOKS["policy"] += 1
        ll_f = log_loss(fit[col], m.predict_proba(fit[FEATS]), labels=m.classes_)
        mfull = refit(s[FEATS], s[col], cfg)
        art["models"][name] = {"fit": m, "full": mfull, "classes": list(m.classes_), "cfg": cfg, "window": w}
        art["priors"][name] = prior.to_dict()
        rows.append({"target": name, "n_fit": len(fit), "n_held": len(held), "fit_window": f"{w[0][0]}-{w[0][1]}", "held_window": f"{w[1][0]}-{w[1][1]}",
                     "ll_blind_held": ll_b, "ll_strength_only_held": ll_a, "ll_model_held": ll_m, "ll_model_fit": ll_f, "gain_vs_blind": ll_b - ll_m, "gain_pct": 100 * (ll_b - ll_m) / ll_b,
                     "gain_state_vs_strength": ll_a - ll_m, "leaf": cfg[1], "min_leaf": cfg[2], "iters": cfg[3]})
    joblib.dump(art, OUT / "shell_policy.joblib")
    pd.DataFrame(rows).to_csv(OUT / "shell_policy_fit.csv", index=False)
    return d, art, pd.DataFrame(rows)


def cells(d):
    c = pd.Series("other", index=d.index)
    c[(d.qtr <= 2)] = "early"
    c[(d.qtr == 3) & (d.sd > 0)] = "leaderQ3"
    c[(d.qtr == 3) & (d.sd < 0)] = "trailerQ3"
    c[(d.qtr == 4) & (d.sd > 0) & (d.tsec > 120)] = "leaderQ4"
    c[(d.qtr == 4) & (d.sd < 0) & (d.tsec > 120)] = "trailerQ4"
    c[(d.qtr == 4) & (d.sd < 0) & (d.tsec <= 120)] = "trailerLast2"
    c[(d.qtr == 4) & (d.sd > 0) & (d.tsec <= 120)] = "leaderLast2"
    return c


def shift_table(d):
    d = d.copy()
    d["cell"] = cells(d)
    fit_mask = d.season.between(2018, 2021)
    q4 = d[fit_mask & (d.qtr >= 3)].sd.abs().quantile([1 / 3, 2 / 3]).tolist()
    d["lead_bin"] = pd.cut(d.sd, [-99, -q4[1], -q4[0], 0, q4[0], q4[1], 99], right=False).astype(str)
    lb = d.box.quantile(1 / 3)
    d["light"] = np.where(d.box.notna(), (d.box <= lb).astype(float), np.nan)
    d["dime"] = np.where(d.ndb.notna(), (d.ndb >= 6).astype(float), np.nan)
    d["base"] = np.where(d.ndb.notna(), (d.ndb <= 4).astype(float), np.nan)
    rows = []
    for era, (a, b) in {"2016-21": (2016, 2021), "2022-25": (2022, 2025)}.items():
        e = d[d.season.between(a, b)]
        for key, grp in list(e.groupby("cell")) + list(e[e.qtr >= 3].groupby("lead_bin")):
            r = {"era": era, "state": key, "n": len(grp)}
            for v in ("two", "man", "light", "dime", "base"):
                r[v] = grp[v].mean()
                r["n_" + v] = int(grp[v].notna().sum())
            r["box_mean"] = grp.box.mean()
            r["ndb_mean"] = grp.ndb.mean()
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "shell_posture_by_state.csv", index=False)
    return d, t, lb


def ctrl_mat(s, extra):
    X = pd.DataFrame({"const": 1.0}, index=s.index)
    for c in extra:
        X[c] = s[c]
    for c in ["zs", "yl", "home", "eo", "ed", "PO", "PD", "RB", "RD"]:
        X[c] = s[c]
    X["lyd"] = np.log(s.ydstogo.clip(lower=1))
    for k in (2, 3, 4):
        X[f"d{k}"] = (s.down == k).astype(float)
    return X


def outcomes(d, expl):
    rows = []
    d = d.copy()
    d["yds"] = d.yards_gained
    d["expl"] = (d.yards_gained >= expl).astype(float)
    d["td"] = d.touchdown.astype(float)
    for name, mask, post, w in [("pass_cov", (d.isp == 1) & d.two.notna(), ["two", "man", "dime", "light"], W_COV),
                                ("pass_box", (d.isp == 1) & d.ndb.notna() & d.box.notna(), ["dime", "light"], W_BOX),
                                ("run_box", (d.play_type == "run") & d.ndb.notna() & d.box.notna(), ["dime", "light"], W_BOX)]:
        for era, (a, b) in (("fit", w[0]), ("held", w[1])):
            s = d[mask & d.season.between(a, b)].dropna(subset=post + ["eo", "ed", "PO", "PD", "RB", "RD", "yds", "epa"])
            for y in ("yds", "expl", "epa", "td"):
                X = ctrl_mat(s, post)
                bb, se, _ = cluster_ols(X, s[y], s.game_id)
                LOOKS["outcome"] += 1
                for p in post:
                    rows.append({"sample": name, "era": era, "y": y, "posture": p, "n": len(s), "est": bb[p], "se": se[p], "p_pos": norm.cdf(bb[p] / se[p]), "mean_y": s[y].mean()})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "shell_outcomes_play.csv", index=False)
    return r


def drive_table(d):
    g = d.sort_values(["game_id", "play_id"])
    g = g[g.fixed_drive.notna()]
    a = g.groupby(["game_id", "fixed_drive"]).agg(season=("season", "first"), posteam=("posteam", "first"), qtr=("qtr", "first"), t0=("tsec", "first"), sd0=("sd", "first"),
                                                 n=("play_id", "size"), res=("fixed_drive_result", "last"), yds=("yards_gained", "sum"), home=("home", "first"),
                                                 eo=("eo", "first"), ed=("ed", "first"), PO=("PO", "first"), PD=("PD", "first"), RB=("RB", "first"), RD=("RD", "first"),
                                                 light=("light", "mean"), dime=("dime", "mean"), base=("base", "mean"), two=("two", "mean"), man=("man", "mean"), nbox=("light", "count"),
                                                 ncov=("two", "count")).reset_index()
    a["td"] = (a.res == "Touchdown").astype(float)
    a["pts"] = np.where(a.res == "Touchdown", 7.0, np.where(a.res == "Field goal", 3.0, np.where(a.res == "Safety", -2.0, 0.0)))
    a["zs"] = a.sd0 / np.sqrt(a.t0 / 60.0 + 1.0)
    a["sd"] = a.sd0
    a["tsec"] = a.t0
    a["cell"] = cells(a)
    return a.dropna(subset=["eo", "ed", "PO", "PD", "RB", "RD"])


def dctrl(s, cell_names, med):
    X = pd.DataFrame({"const": 1.0}, index=s.index)
    for c in cell_names:
        X[c] = (s.cell == c).astype(float)
    for c in med:
        X[c] = s[c]
    for c in ["home", "eo", "ed", "PO", "PD", "RB", "RD"]:
        if c in s:
            X[c] = s[c]
    return X


def decomp(dr):
    cn = ["leaderQ3", "leaderQ4", "trailerQ3", "trailerQ4", "trailerLast2", "leaderLast2"]
    rows = []
    sets = {"M1_box_db": (["light", "dime"], "nbox", W_BOX), "M2_box_db_cov": (["light", "dime", "two", "man"], "ncov", W_COV)}
    rng = np.random.default_rng(20261002)
    for sname, (med, cnt, w) in sets.items():
        for era, (a, b) in (("fit", w[0]), ("held", w[1])):
            s = dr[dr.season.between(a, b) & (dr.cell != "other")].copy()
            s = s[s[cnt] >= 1].dropna(subset=med)
            games = s.game_id.unique()
            gi = {g: np.where(s.game_id.values == g)[0] for g in games}
            for y in ("td", "pts", "yds", "n"):
                Xt = dctrl(s, cn, [])
                Xm = dctrl(s, cn, med)
                bt, set_, _ = cluster_ols(Xt, s[y], s.game_id)
                bd, sed, _ = cluster_ols(Xm, s[y], s.game_id)
                LOOKS["decomp"] += 1
                B = 200
                boots = {c: [] for c in cn}
                Xa, Xb, ya = Xt.values, Xm.values, s[y].values
                for _ in range(B):
                    pick = rng.choice(games, len(games))
                    idx = np.concatenate([gi[g] for g in pick])
                    cb = np.linalg.lstsq(Xa[idx], ya[idx], rcond=None)[0]
                    db = np.linalg.lstsq(Xb[idx], ya[idx], rcond=None)[0]
                    for j, c in enumerate(Xt.columns):
                        if c in cn:
                            boots[c].append((cb[j], db[list(Xm.columns).index(c)]))
                for c in cn:
                    tot, dire = bt[c], bd[c]
                    bb_ = np.array(boots[c])
                    share_b = (bb_[:, 0] - bb_[:, 1]) / np.where(np.abs(bb_[:, 0]) < 1e-12, np.nan, bb_[:, 0])
                    rows.append({"mediators": sname, "era": era, "y": y, "cell": c, "n_drives": int((s.cell == c).sum()), "total": tot, "total_se": set_[c], "direct": dire, "mediated": tot - dire,
                                 "mediated_se_boot": float(np.std(bb_[:, 0] - bb_[:, 1])), "p_med_pos": float(np.mean((bb_[:, 0] - bb_[:, 1]) > 0)),
                                 "share": (tot - dire) / tot if abs(tot) > 1e-12 else np.nan, "share_boot_lo": float(np.nanpercentile(share_b, 5)), "share_boot_hi": float(np.nanpercentile(share_b, 95)),
                                 "p_total_pos": norm.cdf(tot / set_[c])})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "shell_decomp.csv", index=False)
    return r


def main():
    d = build()
    if len(sys.argv) > 1 and sys.argv[1] == "decomp":
        art = joblib.load(OUT / "shell_policy.joblib")
    else:
        d, art, pol = policy(d)
        print(pol.round(4).to_string())
    d, tab, lb = shift_table(d)
    cols = ["era", "state", "n", "two", "man", "light", "dime", "base", "box_mean", "ndb_mean"]
    print(tab[cols].round(3).to_string())
    fit_pass = d[(d.isp == 1) & d.season.between(2016, 2021)].yards_gained
    expl = float(fit_pass.quantile(0.9))
    print("explosive threshold (fit p90 pass yards)", expl, "light box q33", lb)
    o = outcomes(d, expl)
    sel = o[(o.posture.isin(["two", "dime", "light", "man"]))]
    print(sel.pivot_table(index=["sample", "y", "posture"], columns="era", values=["est", "p_pos"]).round(4).to_string())
    dr = drive_table(d)
    dd = decomp(dr)
    print(dd[dd.y.isin(["td", "pts"])].round(4).to_string())
    json.dump({"looks": LOOKS, "explosive_threshold": expl, "light_box_q33": float(lb), "box_clip": art["box_clip"], "db_clip": art["db_clip"]}, open(OUT / "shell_meta.json", "w"), indent=1)
    print(LOOKS)


if __name__ == "__main__":
    main()
