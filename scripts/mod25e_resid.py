import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import mod25e_gain as mg

R2 = mg.R2
OUT = REPO / "artifacts" / "mod25e3" / "resid"
TWO_MIN = 120.0
TYPES = ["td6", "pat", "fg_exp", "fg_luck", "ret_int", "ret_fum", "ret_punt", "ret_kick", "ret_other", "safety", "ot"]
WINS = ["eoh", "rest", "ot"]


def load_real(seasons):
    import mod25d_variance as dv

    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    p = dv.sim.load_reg_seasons(tuple(seasons))
    return p.sort_values(["game_id", "play_id"]).reset_index(drop=True)


def real_events(p):
    p = p[p["qtr"] <= 4].reset_index(drop=True)
    post = p["posteam_score_post"].to_numpy(float)
    pre = p["posteam_score"].to_numpy(float)
    d = np.where(np.isnan(post) | np.isnan(pre), 0.0, post - pre)
    sgn_off = np.where((p["posteam"] == p["home_team"]).to_numpy(), 1.0, -1.0)
    pt = p["play_type"].to_numpy()
    td = p["touchdown"].fillna(0).to_numpy() == 1
    gsr = p["game_seconds_remaining"].to_numpy(float)
    qtr = p["qtr"].to_numpy()
    h = gsr - np.where(qtr <= 2, 1800.0, 0.0)
    win = np.where(np.isin(qtr, (2, 4)) & (h <= TWO_MIN), "eoh", "rest")
    gid = p["game_id"].to_numpy()
    n = len(p)
    rows = []

    def add(mask, comp, pts, sgn):
        rows.append(pd.DataFrame({"game_id": gid[mask], "comp": comp, "pts": pts[mask] * sgn[mask], "win": win[mask]}))

    off_td = (d >= 6) & (pt != "extra_point")
    add(off_td, "td6", np.full(n, 6.0), sgn_off)
    pat = np.where(off_td, d - 6.0, np.where(((pt == "extra_point") | (d == 2)) & (d > 0) & (d < 6), d, 0.0))
    add((pat != 0) | (pt == "extra_point"), "pat", pat, sgn_off)
    ret = td & (d == 0) & (pt != "no_play") & (pt != "extra_point") & ~np.isnan(post)
    ints = p["interception"].fillna(0).to_numpy() == 1
    fum = p["fumble_lost"].fillna(0).to_numpy() == 1
    comp_ret = np.where(pt == "kickoff", "ret_kick", np.where(pt == "punt", "ret_punt", np.where(ints, "ret_int", np.where(fum, "ret_fum", "ret_other"))))
    for c in ("ret_kick", "ret_punt", "ret_int", "ret_fum", "ret_other"):
        add(ret & (comp_ret == c), c, np.full(n, 6.0), -sgn_off)
    last = (~p.duplicated(["game_id", "fixed_drive"], keep="last")).to_numpy()
    saf = last & (p["fixed_drive_result"].to_numpy() == "Safety")
    add(saf, "safety", np.full(n, 2.0), -sgn_off)
    fga = pt == "field_goal"
    fg = pd.DataFrame({"game_id": gid[fga], "dist": p["yardline_100"].to_numpy(float)[fga] + 17.0, "made": (d[fga] == 3).astype(float), "sgn": sgn_off[fga], "win": win[fga]})
    return pd.concat(rows, ignore_index=True), fg


def load_sim():
    f = pd.read_parquet(R2 / "feat_plays.parquet")
    f["k"] = f.groupby("game_id").cumcount()
    fr = []
    for fp in sorted((R2 / "feat_sim").glob("play_*_*.parquet")):
        _, w, s = fp.stem.split("_")
        w, s = int(w), int(s)
        if s < 2:
            continue
        x = pd.read_parquet(fp)
        x["game_id"] = [f"w{w:04d}s{s:02d}g{int(g):03d}" for g in x["g"]]
        fr.append(x)
    A = pd.concat(fr, ignore_index=True)
    P = A[A["qtr"] <= 4].copy()
    P["kept"] = P["code"].isin([0, 1])
    P["k"] = np.where(P["kept"], P.groupby("game_id")["kept"].cumsum() - 1, -1)
    return P.merge(f[["game_id", "k", "epa", "interception", "fumble_lost"]], on=["game_id", "k"], how="left")


def sim_events(P):
    sgn_off = np.where(P["offhome"].to_numpy() == 1, 1.0, -1.0)
    code = P["code"].to_numpy()
    po = P["po"].to_numpy(float)
    pdf = P["pdf"].to_numpy(float)
    qtr = P["qtr"].to_numpy()
    h = P["gsr"].to_numpy(float) - np.where(qtr <= 2, 1800.0, 0.0)
    win = np.where(np.isin(qtr, (2, 4)) & (h <= TWO_MIN), "eoh", "rest")
    gid = P["game_id"].to_numpy()
    n = len(P)
    rows = []

    def add(mask, comp, pts, sgn):
        rows.append(pd.DataFrame({"game_id": gid[mask], "comp": comp, "pts": pts[mask] * sgn[mask], "win": win[mask]}))

    td = (po >= 6) & np.isin(code, (0, 1))
    add(td, "td6", np.full(n, 6.0), sgn_off)
    rt = pdf >= 6
    patp = np.where(td, po - 6.0, np.where(rt, pdf - 6.0, 0.0))
    add(td | rt, "pat", patp, np.where(td, sgn_off, -sgn_off))
    ints = P["interception"].fillna(0).to_numpy() == 1
    fum = P["fumble_lost"].fillna(0).to_numpy() == 1
    kept = np.isin(code, (0, 1))
    comp_ret = np.where(code == 2, "ret_punt", np.where(kept & ints, "ret_int", np.where(kept & fum, "ret_fum", "ret_other")))
    for c in ("ret_punt", "ret_int", "ret_fum", "ret_other"):
        add(rt & (comp_ret == c), c, np.full(n, 6.0), -sgn_off)
    add(pdf == 2, "safety", np.full(n, 2.0), -sgn_off)
    fga = code == 3
    fg = pd.DataFrame({"game_id": gid[fga], "dist": P["yl"].to_numpy(float)[fga] + 17.0, "made": (po[fga] == 3).astype(float), "sgn": sgn_off[fga], "win": win[fga]})
    return pd.concat(rows, ignore_index=True), fg


def fit_fg(fg):
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=1e6, max_iter=1000).fit(fg[["dist"]].to_numpy(), fg["made"].to_numpy().astype(int))
    return lambda dist: m.predict_proba(np.asarray(dist).reshape(-1, 1))[:, 1]


def assemble(ev, fg, pfg):
    ph = pfg(fg["dist"])
    e1 = pd.DataFrame({"game_id": fg["game_id"], "comp": "fg_exp", "pts": 3.0 * ph * fg["sgn"], "win": fg["win"]})
    e2 = pd.DataFrame({"game_id": fg["game_id"], "comp": "fg_luck", "pts": 3.0 * (fg["made"] - ph) * fg["sgn"], "win": fg["win"]})
    return pd.concat([ev, e1, e2], ignore_index=True)


def game_matrix(E, gid, cols, key):
    pv = E.pivot_table(index="game_id", columns=key, values="pts", aggfunc="sum").reindex(columns=cols)
    return pv.reindex(gid).fillna(0.0).reset_index(drop=True)


def resid_matrix(M, e):
    X = np.column_stack([np.ones(len(e)), e])
    beta = np.linalg.lstsq(X, M, rcond=None)[0]
    return M - X @ beta, beta


def contrib(R, rng, nboot):
    tot = R.sum(axis=1)
    n = len(tot)

    def stats(ix):
        r, t = R[ix], tot[ix]
        rc, tc = r - r.mean(axis=0), t - t.mean()
        return np.concatenate([(rc * tc[:, None]).mean(axis=0), rc.var(axis=0), [tc.var()]])

    return stats(np.arange(n)), np.array([stats(rng.integers(0, n, n)) for _ in range(nboot)])


def run_era(seasons, label, nboot, rng, simev, simfg, Tsim):
    p = load_real(seasons)
    Dr, Gr = mg.real_drives(seasons)
    Tr = mg.table(mg.finish(Dr, Gr, mg.wp_model(Dr, Gr)), Gr).reset_index(drop=True)
    ev, fg = real_events(p)
    pfg = fit_fg(fg)
    out = {"era": label}
    mats = {}
    for tag, EV, FG, T in (("real", ev, fg, Tr), ("sim", simev, simfg, Tsim.reset_index(drop=True))):
        E = assemble(EV, FG, pfg)
        fin = T["final"].to_numpy(float)
        Mt = game_matrix(E, T["game_id"], TYPES[:-1], "comp")
        Mt["ot"] = fin - Mt.sum(axis=1)
        Mw = game_matrix(E, T["game_id"], ["eoh", "rest"], "win")
        Mw["ot"] = fin - Mw.sum(axis=1)
        mats[tag] = (Mt, Mw, T[mg.CLASSES], T["e"].to_numpy(float), fin)
        out[tag + "_games"] = len(T)
        out[tag + "_ot_nonzero_share"] = float((np.abs(Mt["ot"].to_numpy()) > 1e-9).mean())
        out[tag + "_ot_mean_abs"] = float(np.abs(Mt["ot"].to_numpy()).mean())
    for dec, ix, cols in (("type", 0, TYPES), ("window", 1, WINS), ("drive", 2, mg.CLASSES)):
        res = {}
        pts = {}
        k = len(cols)
        for tag in ("real", "sim"):
            e, fin = mats[tag][3], mats[tag][4]
            R, beta = resid_matrix(mats[tag][ix].to_numpy(float), e)
            pts[tag] = contrib(R, rng, nboot)
            fr, bf = resid_matrix(fin.reshape(-1, 1), e)
            res[tag + "_resid_var"] = float(fr.var())
            res[tag + "_sum_check"] = float(np.abs(R.sum(axis=1) - fr[:, 0]).max())
            res[tag + "_gain"] = float(bf[1, 0])
        rows = {}
        for j, c in enumerate(cols + ["TOTAL"]):
            for kind, off in (("cov", 0), ("var", k)):
                if c == "TOTAL":
                    if kind == "var":
                        continue
                    jj = 2 * k
                else:
                    jj = off + j
                a_r, a_s = pts["real"][0][jj], pts["sim"][0][jj]
                d = pts["sim"][1][:, jj] - pts["real"][1][:, jj]
                rows[f"{c}_{kind}"] = {"real": float(a_r), "sim": float(a_s), "diff": float(a_s - a_r), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "probability_positive": float((d > 0).mean())}
        res["rows"] = rows
        out[dec] = res
    return out


def epa_credit(Psim, preal):
    out = {}
    q = preal[preal["play_type"].isin(["run", "pass"]) & preal["epa"].notna() & (preal["qtr"] <= 4)]
    d = (q["posteam_score_post"] - q["posteam_score"]).fillna(0.0)
    sets = {"real": (q["epa"], q["yardline_100"], (q["touchdown"] == 1) & (d >= 6), (q["interception"] == 1) | (q["fumble_lost"] == 1), (q["touchdown"] == 1) & (d == 0))}
    s = Psim[Psim["kept"] & Psim["epa"].notna()]
    sets["sim"] = (s["epa"], s["yl"], s["po"] >= 6, (s["interception"] == 1) | (s["fumble_lost"] == 1), s["pdf"] >= 6)
    for tag, (e, yl, td, tov, tdd) in sets.items():
        o = {"n": int(len(e)), "epa_mean": float(e.mean()), "epa_var": float(e.var())}
        o["td_off"] = {"n": int(td.sum()), "mean_epa": float(e[td].mean()), "mean_yl": float(yl[td].mean())}
        o["tov_all"] = {"n": int(tov.sum()), "mean_epa": float(e[tov].mean())}
        o["tov_td"] = {"n": int(tdd.sum()), "mean_epa": float(e[tdd].mean())}
        o["nonscore"] = {"mean_epa": float(e[~td & ~tdd].mean()), "var": float(e[~td & ~tdd].var())}
        b = pd.cut(yl[td], [0, 10, 20, 40, 60, 80, 100])
        o["td_epa_by_yl"] = {str(a): float(v) for a, v in e[td].groupby(b, observed=True).mean().items()}
        out[tag] = o
    return out


def main():
    nboot = 400
    rng = np.random.default_rng(7)
    OUT.mkdir(parents=True, exist_ok=True)
    Psim = load_sim()
    simev, simfg = sim_events(Psim)
    Dsim, Gsim = mg.sim_drives()
    result = {}
    for lab, lo, hi in (("pool_2009_17", 2009, 2017), ("held_2018_25", 2018, 2025)):
        Dr, Gr = mg.real_drives(range(lo, hi + 1))
        wpf = mg.wp_model(Dr, Gr)
        Tsim = mg.table(mg.finish(Dsim.copy(), Gsim, wpf), Gsim)
        r = run_era(range(lo, hi + 1), lab, nboot, rng, simev, simfg, Tsim)
        result[lab] = r
        print(lab, r["real_games"], r["sim_games"], "ot nonzero", r["real_ot_nonzero_share"], r["sim_ot_nonzero_share"], "ot mean abs", r["real_ot_mean_abs"], r["sim_ot_mean_abs"], flush=True)
        for dec in ("type", "window", "drive"):
            x = r[dec]
            print(" ", dec, "resid var real/sim", round(x["real_resid_var"], 2), round(x["sim_resid_var"], 2), "sumchk", x["real_sum_check"], x["sim_sum_check"], "gain", round(x["real_gain"], 3), round(x["sim_gain"], 3), flush=True)
            for key, v in x["rows"].items():
                print(f"    {key:18s} real {v['real']:7.3f} sim {v['sim']:7.3f} d {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] pp {v['probability_positive']:.2f}", flush=True)
    result["epa_credit"] = epa_credit(Psim, load_real(range(2009, 2018)))
    print(json.dumps(result["epa_credit"], indent=1), flush=True)
    (OUT / "resid.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
