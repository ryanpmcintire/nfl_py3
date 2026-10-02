import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25_mechanisms as m25  # noqa: E402
import mod25c_noise as c25  # noqa: E402
import mod25d_variance as d25  # noqa: E402
import sim04_engine as sim  # noqa: E402

OUT = REPO / "artifacts" / "mod25e"
RNG = np.random.default_rng(11)


def load():
    sim.PBP_SNAPSHOT_DIR = m25.SNAP
    pbp = sim.load_reg_seasons(tuple(c25.TRAIN))
    trans = sim.build_transition_frame(pbp)
    at = c25.attrs_from(pbp, trans)
    return pbp, trans, at


def ols(X, y):
    X1 = np.c_[np.ones(len(X)), X]
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b


def cluster_boot(X, y, cl, n=300):
    keys = np.unique(cl)
    idx = {k: np.flatnonzero(cl == k) for k in keys}
    out = []
    for _ in range(n):
        pick = RNG.choice(keys, len(keys))
        ii = np.concatenate([idx[k] for k in pick])
        out.append(ols(X[ii], y[ii])[1:])
    return np.array(out)


def summ(b, boot, names):
    r = {}
    for j, nm in enumerate(names):
        r[nm] = {"est": float(b[j + 1]), "lo": float(np.quantile(boot[:, j], 0.05)), "hi": float(np.quantile(boot[:, j], 0.95)), "p_pos": float((boot[:, j] > 0).mean())}
    return r


def resid_cells(y, keys):
    s = pd.Series(y)
    return y - s.groupby(keys).transform("mean").to_numpy()


def yard_gain_block(pbp, trans, at):
    meta = pbp[["game_id", "play_id", "season", "posteam", "defteam"]].drop_duplicates(["game_id", "play_id"])
    t = trans[["game_id", "play_id"]].merge(meta, on=["game_id", "play_id"], how="left")
    ok = (t["season"].astype(str) + "_" + t["posteam"].astype(str)).to_numpy()
    fk = (t["season"].astype(str) + "_" + t["defteam"].astype(str)).to_numpy()
    gid = trans["game_id"].to_numpy()
    half = (pd.util.hash_pandas_object(pd.Series(gid), index=False).to_numpy() % 2).astype(int)
    pb_game = pbp["game_id"].to_numpy()
    pb_half = (pd.util.hash_pandas_object(pd.Series(pb_game), index=False).to_numpy() % 2).astype(int)
    code = trans["play_type_code"].to_numpy()
    yards = at["yards"].astype(float)
    epa = at["epa"].astype(float)
    down = trans["down_i"].to_numpy()
    dist = np.clip(trans["dist_raw"].to_numpy(), 0, 20).astype(int)
    yl = (trans["fp_raw"].to_numpy() // 10).astype(int)
    cellkey = (code.astype(np.int64) * 10 + down) * 1000 + dist * 20 + yl
    run_pass = (code == 0) | (code == 1)
    res = {}
    eo_x = np.zeros(len(trans))
    ef_x = np.zeros(len(trans))
    for h in (0, 1):
        sub = pbp[pb_half == h]
        d = d25.team_effects_plays(sub, list(c25.TRAIN), "epa")
        _, (oc, fc, eo, ef, ou, fu, mu) = d25.fit_play_effects(d, "epa")
        om, fm = dict(zip(ou, eo)), dict(zip(fu, ef))
        m = half != h
        eo_x[m] = pd.Series(ok[m]).map(om).fillna(0.0).to_numpy()
        ef_x[m] = pd.Series(fk[m]).map(fm).fillna(0.0).to_numpy()
    eo_in, ef_in = d25.blup_rows(pbp, trans, c25.TRAIN)
    season = t["season"].to_numpy()
    cl = (t["season"].astype(str) + "_" + t["posteam"].astype(str)).to_numpy()
    for tag, eo, ef in (("crossfit", eo_x, ef_x), ("insample", eo_in, ef_in)):
        for ctl in ("raw", "state_controlled"):
            m = run_pass
            y = yards[m]
            X = np.c_[eo[m], ef[m]]
            if ctl == "state_controlled":
                keys = cellkey[m]
                y = resid_cells(y, keys)
                X = np.c_[resid_cells(X[:, 0], keys), resid_cells(X[:, 1], keys)]
            b = ols(X, y)
            bt = cluster_boot(X, y, cl[m])
            r = summ(b, bt, ["off_slope", "def_slope"])
            Xs = X[:, 0] + X[:, 1]
            b2 = ols(Xs[:, None], y)
            bt2 = cluster_boot(Xs[:, None], y, cl[m])
            r["sum_slope_yards_per_epa_unit"] = summ(b2, bt2, ["s"])["s"]
            r["sd_eo"] = float(np.std(eo[m]))
            r["sd_ef"] = float(np.std(ef[m]))
            res[f"yards_{tag}_{ctl}"] = r
    m = run_pass
    keys = cellkey[m]
    y = resid_cells(epa[m].astype(float), keys)
    x = resid_cells(yards[m], keys)
    b = ols(x[:, None], y)
    bt = cluster_boot(x[:, None], y, cl[m])
    res["epa_per_yard_state_controlled"] = summ(b, bt, ["slope"])["slope"]
    b = ols(yards[m][:, None], epa[m].astype(float))
    bt = cluster_boot(yards[m][:, None], epa[m].astype(float), cl[m])
    res["epa_per_yard_raw"] = summ(b, bt, ["slope"])["slope"]
    return res


def pace_block(trans):
    el = trans["clock_elapsed"].to_numpy(dtype=float)
    qtr = trans["qtr_actual"].to_numpy()
    code = trans["play_type_code"].to_numpy()
    flip = trans["possession_flip"].to_numpy().astype(bool)
    cl = c25.cls_of(code, flip, trans["points_off"].to_numpy(), trans["points_def"].to_numpy(), trans["yards_gained"].to_numpy())
    zn = c25.zone_v(qtr, trans["gsr_actual"].to_numpy())
    sb = c25.sb_v(trans["sc_raw"].to_numpy())
    key = (cl * 8 + zn) * 8 + sb
    zk = cl * 8 + zn
    s = pd.Series(el)
    mu_f = s.groupby(key).transform("mean").to_numpy()
    n_f = s.groupby(key).transform("size").to_numpy()
    mu_z = s.groupby(zk).transform("mean").to_numpy()
    mu = np.where(n_f >= 40, mu_f, mu_z)
    m = (qtr <= 4) & (mu > 0)
    gid = pd.factorize(trans["game_id"].to_numpy())[0]
    gm = gid[m]
    par = (pd.Series(np.arange(len(gm))).groupby(gm).cumcount().to_numpy() % 2)
    df = pd.DataFrame({"g": gm, "p": par, "el": el[m], "mu": mu[m]})
    ag = df.groupby(["g", "p"])[["el", "mu"]].sum().unstack()
    ag = ag.dropna()
    r0 = (ag[("el", 0)] / ag[("mu", 0)]).to_numpy()
    r1 = (ag[("el", 1)] / ag[("mu", 1)]).to_numpy()

    def sig(a, b):
        c = np.cov(a, b)[0, 1] / (a.mean() * b.mean())
        return float(np.sqrt(np.log1p(max(c, 0.0))))

    ng = len(r0)
    bs = []
    for _ in range(500):
        i = RNG.integers(0, ng, ng)
        bs.append(sig(r0[i], r1[i]))
    bs = np.array(bs)
    return {"games": int(ng), "sigma_pace_est": sig(r0, r1), "lo": float(np.quantile(bs, 0.05)), "hi": float(np.quantile(bs, 0.95)), "p_gt_0.04": float((bs > 0.04).mean()), "corr_halves": float(np.corrcoef(r0, r1)[0, 1])}


def main():
    pbp, trans, at = load()
    res = {"yard": yard_gain_block(pbp, trans, at), "pace": pace_block(trans)}
    (OUT / "knobs.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
