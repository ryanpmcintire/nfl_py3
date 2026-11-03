import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "mod35"
FC = ROOT / "data/raw/forecast_archive/pool_decision_2009_2025_aware/forecasts.parquet"
PROXY = ROOT / "artifacts/proxy_opener_replication/20260819T194330Z/main_scored.parquet"
SERVED = ROOT / "artifacts/opener_evaluation/20261103T173420Z/per_game.parquet"
FIRST, LAST = 2011, 2025
HALF_WIDTHS = (0, 1, 2, 3, 4, 5, 6)
OUTDOOR = ("outdoors", "open")
INDOOR = ("dome", "closed")
EPS = 1e-6
BOOT = 5000
PERMS = 200
SEED = 35
WC_MAX_T = 50.0
WC_MIN_V = 3.0
WC_A, WC_B, WC_C, WC_D, WC_E = 35.74, 0.6215, 35.75, 0.16, 0.4275


def apparent(t, v):
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    p = np.power(np.where(v > 0, v, np.nan), WC_D)
    wc = WC_A + WC_B * t - WC_C * p + WC_E * t * p
    return np.where((t <= WC_MAX_T) & (v > WC_MIN_V), wc, t)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit(x, y):
    m = LogisticRegression(penalty=None, max_iter=1000)
    m.fit(x, y)
    return m


def build_base():
    a = pd.read_parquet(PROXY)
    a = a.loc[~a.open_ambiguous.astype(bool)]
    a = pd.DataFrame(
        {
            "game_id": a.game_id,
            "season": a.season,
            "week": a.week,
            "p0": a.home_cover_probability_at_open_proxy,
            "open": a.proxy_open_home_spread,
            "result": a.result,
        }
    )
    b = pd.read_parquet(SERVED)
    b = pd.DataFrame(
        {
            "game_id": b.game_id,
            "season": b.season,
            "week": b.week,
            "p0": b.home_cover_probability_excluding_push_at_open,
            "open": b.tue_open_home_spread,
            "result": b.result,
        }
    )
    d = pd.concat([a, b], ignore_index=True)
    d["margin"] = d.result - d.open
    d = d.loc[d.margin != 0].copy()
    d["y"] = (d.margin > 0).astype(int)
    d["z0"] = logit(d.p0.to_numpy())
    parts = d.game_id.str.split("_")
    d["away"] = parts.str[2]
    d["home"] = parts.str[3]
    return d.loc[d.season.between(FIRST, LAST)]


def normals(rec, keys, k):
    out = np.full(len(keys), np.nan)
    for i, (team, s, w) in enumerate(keys):
        sub = rec.loc[(rec.home == team) & (rec.season < s) & ((rec.week - w).abs() <= k)]
        if len(sub):
            out[i] = sub.app.mean()
    return out


def select_half_widths(rec):
    chosen = {}
    err = {}
    for u in sorted(rec.season.unique()):
        tgt = rec.loc[rec.season == u]
        keys = list(zip(tgt.home, tgt.season, tgt.week, strict=True))
        preds = np.vstack([normals(rec, keys, k) for k in HALF_WIDTHS])
        ok = ~np.isnan(preds).any(axis=0)
        if ok.any():
            err[u] = (((preds[:, ok] - tgt.app.to_numpy()[ok]) ** 2).sum(axis=1), int(ok.sum()))
    for s in range(FIRST, LAST + 1):
        prior = [u for u in err if u < s]
        tot = sum(err[u][0] for u in prior)
        n = sum(err[u][1] for u in prior)
        chosen[s] = (HALF_WIDTHS[int(np.argmin(tot))], tot / n)
    return chosen


def metrics(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return {
        "acc": float(((p > 0.5) == (y == 1)).mean()),
        "ll": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
    }


def loso(d, xcol, seasons):
    pb = np.full(len(d), np.nan)
    pf = np.full(len(d), np.nan)
    coefs = []
    for s in seasons:
        tr = (d.season != s).to_numpy()
        te = ~tr
        xtr = d.loc[tr, xcol].to_numpy()
        mu, sd = xtr.mean(), xtr.std()
        zb_tr = d.loc[tr, ["z0"]].to_numpy()
        zb_te = d.loc[te, ["z0"]].to_numpy()
        mb = fit(zb_tr, d.loc[tr, "y"])
        pb[te] = mb.predict_proba(zb_te)[:, 1]
        ftr = np.column_stack([zb_tr[:, 0], (xtr - mu) / sd])
        fte = np.column_stack([zb_te[:, 0], (d.loc[te, xcol].to_numpy() - mu) / sd])
        mf = fit(ftr, d.loc[tr, "y"])
        pf[te] = mf.predict_proba(fte)[:, 1]
        coefs.append(
            {
                "season": int(s),
                "b_x": float(mf.coef_[0][1]),
                "b_z0": float(mf.coef_[0][0]),
                "b_z0_base": float(mb.coef_[0][0]),
                "intercept": float(mf.intercept_[0]),
                "n_train": int(tr.sum()),
            }
        )
    return pb, pf, coefs


def wk_boot(d, pb, pf, rng):
    y = d.y.to_numpy()
    cb = ((pb > 0.5) == (y == 1)).astype(float)
    cf = ((pf > 0.5) == (y == 1)).astype(float)
    pbc = np.clip(pb, EPS, 1 - EPS)
    pfc = np.clip(pf, EPS, 1 - EPS)
    lb = -(y * np.log(pbc) + (1 - y) * np.log(1 - pbc))
    lf = -(y * np.log(pfc) + (1 - y) * np.log(1 - pfc))
    g = d.groupby(["season", "week"]).ngroup().to_numpy()
    ng = g.max() + 1
    sa = np.bincount(g, cf - cb, ng)
    sl = np.bincount(g, lb - lf, ng)
    cnt = np.bincount(g, minlength=ng)
    da, dl = [], []
    for _ in range(BOOT):
        idx = rng.integers(0, ng, ng)
        n = cnt[idx].sum()
        da.append(sa[idx].sum() / n)
        dl.append(sl[idx].sum() / n)

    def s(a):
        a = np.array(a)
        return {
            "mean": float(a.mean()),
            "lo5": float(np.percentile(a, 5)),
            "hi95": float(np.percentile(a, 95)),
            "probability_positive": float((a > 0).mean()),
        }

    return {"acc_gain": s(da), "ll_gain": s(dl)}


def evaluate(d, xcol, seasons, rng, perm=False):
    pb, pf, coefs = loso(d, xcol, seasons)
    y = d.y.to_numpy()
    mb, mf = metrics(y, pb), metrics(y, pf)
    flip = (pb > 0.5) != (pf > 0.5)
    dec = {
        "flipped_games": int(flip.sum()),
        "base_correct": int(((pb > 0.5) == (y == 1))[flip].sum()),
        "term_correct": int(((pf > 0.5) == (y == 1))[flip].sum()),
    }
    out = {"n": len(d), "recal_base": mb, "recal_plus_term": mf, "decisive": dec, "coefs": coefs}
    if not perm:
        out["raw_p0"] = metrics(y, d.p0.to_numpy())
        out["intercept_only"] = metrics(y, np.full(len(d), y.mean()))
        out["boot"] = wk_boot(d, pb, pf, rng)
        z = d[["z0"]].to_numpy()
        xs = d[xcol].to_numpy()
        zs = np.column_stack([z[:, 0], (xs - xs.mean()) / xs.std()])
        mfa = fit(zs, y)
        mba = fit(z, y)
        out["in_sample"] = {
            "base": metrics(y, mba.predict_proba(z)[:, 1]),
            "term": metrics(y, mfa.predict_proba(zs)[:, 1]),
            "b_x": float(mfa.coef_[0][1]),
        }
        bs = np.array([c["b_x"] for c in coefs])
        out["coef_summary"] = {
            "mean": float(bs.mean()),
            "sd": float(bs.std(ddof=1)),
            "positive_folds": int((bs > 0).sum()),
            "folds": len(bs),
        }
        out["gap_ll_gain_in_minus_oos"] = float(
            (out["in_sample"]["base"]["ll"] - out["in_sample"]["term"]["ll"])
            - (mb["ll"] - mf["ll"])
        )
    return out, pb, pf


def main():
    rng = np.random.default_rng(SEED)
    fc = pd.read_parquet(FC)
    fc = fc.loc[fc.fetch_status == "ok"].copy()
    fc["home"] = fc.game_id.str.split("_").str[3]
    fc["fc_at"] = apparent(fc.forecast_temp_f.to_numpy(), fc.forecast_wind_mph.to_numpy())
    rec = fc.loc[
        fc.roof.isin(OUTDOOR) & fc.actual_temp_f.notna() & fc.actual_wind_mph.notna()
    ].copy()
    rec["app"] = apparent(rec.actual_temp_f.to_numpy(), rec.actual_wind_mph.to_numpy())
    rec = rec[["home", "season", "week", "app"]]
    chosen = select_half_widths(rec)
    base = build_base()
    base = base.merge(fc[["game_id", "roof", "fc_at"]], on="game_id", how="left")
    base["nrm_away"] = np.nan
    for s in range(FIRST, LAST + 1):
        k = chosen[s][0]
        m = (base.season == s).to_numpy()
        keys = list(
            zip(base.loc[m, "away"], base.loc[m, "season"], base.loc[m, "week"], strict=True)
        )
        base.loc[m, "nrm_away"] = normals(rec, keys, k)
    base["x"] = (base.nrm_away - base.fc_at).abs()
    seasons = list(range(FIRST, LAST + 1))
    cov = {
        "population": len(base),
        "outdoor": int(base.roof.isin(OUTDOOR).sum()),
        "indoor": int(base.roof.isin(INDOOR).sum()),
        "outdoor_with_x": int((base.roof.isin(OUTDOOR) & base.x.notna()).sum()),
        "indoor_with_x": int((base.roof.isin(INDOOR) & base.x.notna()).sum()),
        "half_widths": {int(s): int(v[0]) for s, v in chosen.items()},
    }
    po = base.loc[base.roof.isin(OUTDOOR) & base.x.notna()].reset_index(drop=True)
    pi = base.loc[base.roof.isin(INDOOR) & base.x.notna()].reset_index(drop=True)
    prim, pb, pf = evaluate(po, "x", seasons, rng)
    plac, _, _ = evaluate(pi, "x", seasons, rng)
    ya = po.y.to_numpy()
    obs_acc = metrics(ya, pf)["acc"] - metrics(ya, pb)["acc"]
    obs_ll = metrics(ya, pb)["ll"] - metrics(ya, pf)["ll"]
    pa, pl = [], []
    groups = po.groupby(["season", "week"]).indices
    for _ in range(PERMS):
        q = po.copy()
        xv = q.x.to_numpy().copy()
        for idx in groups.values():
            xv[idx] = rng.permutation(xv[idx])
        q["x"] = xv
        r, _, _ = evaluate(q, "x", seasons, rng, perm=True)
        pa.append(r["recal_plus_term"]["acc"] - r["recal_base"]["acc"])
        pl.append(r["recal_base"]["ll"] - r["recal_plus_term"]["ll"])
    perm = {
        "obs_acc_gain": obs_acc,
        "obs_ll_gain": obs_ll,
        "perm_acc_gain_mean": float(np.mean(pa)),
        "perm_ll_gain_mean": float(np.mean(pl)),
        "p_acc_ge_obs": float((np.array(pa) >= obs_acc).mean()),
        "p_ll_ge_obs": float((np.array(pl) >= obs_ll).mean()),
        "perms": PERMS,
    }
    res = {
        "coverage": cov,
        "primary": prim,
        "placebo": plac,
        "permutation": perm,
        "x_summary": {"mean": float(po.x.mean()), "sd": float(po.x.std())},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    summ = json.loads(json.dumps(res))
    for k in ("primary", "placebo"):
        summ[k].pop("coefs", None)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
