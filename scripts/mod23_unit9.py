import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "F:/Repos/nfl_py3/scripts")
import per_flag_weights_eval as P

ROOT = P.ROOT
SEEDS = 20261016
DRAWS = 10000
BINS = [0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0001]
data = pd.read_parquet(P.ARTIFACT)
data = data.loc[data.season.between(2020, 2025)].reset_index(drop=True)
b = pd.read_parquet(ROOT / "artifacts/mod23_unit6/B_run.parquet")
pb = b["p"].reindex(data["game_id"]).to_numpy(float)
assert not np.isnan(pb).any()
mp = data["model_probability"].to_numpy(float)
print("corr B p vs served model_probability", np.corrcoef(pb, mp)[0, 1], "mean abs diff", np.abs(pb - mp).mean())
print("served model pick agree with B pick_home?", (b["pick_home"].reindex(data["game_id"]).to_numpy() == (mp >= 0.5)).mean())
pb = np.clip(pb, 1e-6, 1 - 1e-6)
data["logit_B"] = np.log(pb / (1 - pb))
print("corr logits", np.corrcoef(data.logit_B, data.model_logit)[0, 1])
y = data["home_covered"].to_numpy(float)
feats = P.VARIANTS["pooled"]
featsB = ("logit_B",) + feats[1:]


def run(view, features):
    frame = P.prepare(data, view)
    oos = np.full(len(frame), np.nan)
    folds = {}
    for held in P.SEASONS:
        tr = frame.loc[frame.season.ne(held)]
        te = frame.loc[frame.season.eq(held)]
        bt, m, s = P.fit(tr, features)
        oos[te.index] = P.predict(te, features, bt, m, s)
        folds[held] = P.natural(features, bt, m, s)
    return oos, folds


def boot(pn, po, rng):
    blk = data["season"].to_numpy() - 2020
    n = np.bincount(blk).astype(float)
    ad = ((pn >= 0.5) == (y == 1)).astype(float) - ((po >= 0.5) == (y == 1))
    bd = (pn - y) ** 2 - (po - y) ** 2
    ld = -(y * np.log(pn) + (1 - y) * np.log1p(-pn)) + (y * np.log(po) + (1 - y) * np.log1p(-po))
    a = np.bincount(blk, weights=ad)
    br = np.bincount(blk, weights=bd)
    ll = np.bincount(blk, weights=ld)
    d = rng.integers(0, 6, size=(DRAWS, 6))
    g = n[d].sum(1)
    A = 100 * a[d].sum(1) / g
    B = br[d].sum(1) / g
    L = ll[d].sum(1) / g
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return {
        "acc_pts": float(100 * ad.mean()), "acc_ci": q(A), "acc_pp": float((A > 0).mean()),
        "brier_delta": float(bd.mean()), "brier_ci": q(B), "brier_p_better": float((B < 0).mean()),
        "ll_delta": float(ld.mean()), "ll_ci": q(L), "ll_p_better": float((L < 0).mean()),
    }


rng = np.random.default_rng(SEEDS)
out = {}
for view in P.VIEWS:
    ps, fs = run(view, feats)
    pn, fn = run(view, featsB)
    ms, mn = P.metrics(ps, y), P.metrics(pn, y)
    ch = (ps >= 0.5) != (pn >= 0.5)
    cw = int(((pn >= 0.5) == (y == 1))[ch].sum())
    r = {"served": ms, "armB": mn, "paired": boot(pn, ps, rng), "changed": {"n": int(ch.sum()), "B_W": cw, "S_W": int(ch.sum()) - cw}}
    rel = []
    for nm, p in (("served", ps), ("armB", pn)):
        bi = np.digitize(p, BINS[1:-1])
        rel.append({nm: [{"bin": f"{BINS[i]}-{BINS[i+1]}", "n": int((bi == i).sum()), "pred": float(p[bi == i].mean()) if (bi == i).any() else None, "obs": float(y[bi == i].mean()) if (bi == i).any() else None} for i in range(6)]})
    r["reliability"] = rel
    seas = {}
    for s in P.SEASONS:
        m = data.season.eq(s).to_numpy()
        a, bq = P.metrics(ps[m], y[m]), P.metrics(pn[m], y[m])
        seas[s] = {"served": [a["record"], round(a["brier"], 5), round(a["log_loss"], 5)], "B": [bq["record"], round(bq["brier"], 5), round(bq["log_loss"], 5)]}
    r["per_season"] = seas
    r["coef_model_per_fold"] = {"served": {s: fs[s]["model_logit"] for s in fs}, "B": {s: fn[s]["logit_B"] for s in fn}}
    r["coef_flag_per_fold"] = {"served": {s: fs[s][P.FLAG_SUM_COLUMN] for s in fs}, "B": {s: fn[s][P.FLAG_SUM_COLUMN] for s in fn}}
    out[view] = r
fl = pd.read_parquet(ROOT / "artifacts/mod23_unit6/flags.parquet")
home = fl.merge(data[["game_id", "home_team"]], on="game_id")
hf = home.loc[home.team.eq(home.home_team)].set_index("game_id")["flag"]
af = home.loc[~home.team.eq(home.home_team)].set_index("game_id")["flag"]
data["qb_signed"] = (hf.reindex(data.game_id).to_numpy() - af.reindex(data.game_id).to_numpy())
data["qb_home"] = hf.reindex(data.game_id).to_numpy()
data["qb_away"] = af.reindex(data.game_id).to_numpy()
data["logit_diff"] = data.logit_B - data.model_logit
cols = ["qb_signed", "qb_home", "qb_away", "logit_diff", P.FLAG_SUM_COLUMN, P.MOVE_COLUMN, P.MOVE_AVAILABLE_COLUMN, "flag_protection"]
out["overlap_corr"] = data[cols].corr().round(3).to_dict()
out["qb_flag_counts"] = {"games_with_own_flag_home": int((data.qb_home == 1).sum()), "away": int((data.qb_away == 1).sum()), "nan": int(data.qb_signed.isna().sum())}
out["rows"] = len(data)
(ROOT / "tests/scratch/unit9_results.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
