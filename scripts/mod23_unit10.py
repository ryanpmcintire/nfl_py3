import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "F:/Repos/nfl_py3/scripts")
import per_flag_weights_eval as P

ROOT = P.ROOT
SEED = 20261016
DRAWS = 10000
BINS = [0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0001]
data = pd.read_parquet(P.ARTIFACT)
data = data.loc[data.season.between(2020, 2025)].reset_index(drop=True)
y = data["home_covered"].to_numpy(float)
SRC = {"u1": "artifacts/mod23_unit1/greedy_per_game.parquet", "u3": "artifacts/mod23_unit3/compact_net_per_game.parquet"}
for k, f in SRC.items():
    d = pd.read_parquet(ROOT / f)
    p = d["p"].reindex(data["game_id"]).to_numpy(float)
    assert not np.isnan(p).any()
    print(k, "corr p vs served", np.corrcoef(p, data["model_probability"])[0, 1], "y match", float(np.mean(d["y"].reindex(data["game_id"]).to_numpy() == y)))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    data["logit_" + k] = np.log(p / (1 - p))
rest = P.VARIANTS["pooled"][1:]
ARMS = {
    "a_u1": ("logit_u1",) + rest,
    "b_u3": ("logit_u3",) + rest,
    "c_served_and_u3": ("model_logit", "logit_u3") + rest,
}
base = P.VARIANTS["pooled"]
wk = data.groupby(["season", "week"]).ngroups


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


def rel(p):
    bi = np.digitize(p, BINS[1:-1])
    return [{"bin": f"{BINS[i]}-{BINS[i+1]}", "n": int((bi == i).sum()), "pred": float(p[bi == i].mean()) if (bi == i).any() else None, "obs": float(y[bi == i].mean()) if (bi == i).any() else None} for i in range(6)]


rng = np.random.default_rng(SEED)
out = {"rows": len(data), "season_weeks": int(wk)}
for view in P.VIEWS:
    ps, fs = run(view, base)
    r = {"served": P.metrics(ps, y), "served_reliability": rel(ps)}
    for name, feats in ARMS.items():
        pn, fn = run(view, feats)
        ch = (ps >= 0.5) != (pn >= 0.5)
        cw = int(((pn >= 0.5) == (y == 1))[ch].sum())
        seas = {}
        for s in P.SEASONS:
            m = data.season.eq(s).to_numpy()
            a, b = P.metrics(ps[m], y[m]), P.metrics(pn[m], y[m])
            seas[s] = {"served": [a["record"], round(a["brier"], 5), round(a["log_loss"], 5)], "arm": [b["record"], round(b["brier"], 5), round(b["log_loss"], 5)]}
        r[name] = {
            "metrics": P.metrics(pn, y), "paired": boot(pn, ps, rng),
            "changed": {"n": int(ch.sum()), "arm_W": cw, "served_W": int(ch.sum()) - cw},
            "reliability": rel(pn), "per_season": seas,
            "coef_per_fold": {s: {k: v for k, v in fn[s].items() if k in ("model_logit", "logit_u1", "logit_u3", P.FLAG_SUM_COLUMN)} for s in fn},
        }
    r["served_coef_per_fold"] = {s: {k: v for k, v in fs[s].items() if k in ("model_logit", P.FLAG_SUM_COLUMN)} for s in fs}
    out[view] = r
(ROOT / "tests/scratch/unit10_results.json").write_text(json.dumps(out, indent=1, default=str))
for view in P.VIEWS:
    print(view, "served", out[view]["served"])
    for name in ARMS:
        o = out[view][name]
        print(view, name, o["metrics"], {k: round(v, 5) if isinstance(v, float) else v for k, v in o["paired"].items()}, o["changed"])
