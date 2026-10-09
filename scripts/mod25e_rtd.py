import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUTD = REPO / "artifacts" / "mod25e3" / "rtd"
FIT = OUTD / "fit.json"
KICKS = REPO / "artifacts" / "mod25e3" / "kick" / "kicks_2009.parquet"
PBP = REPO / "data" / "pbp" / "raw" / "20260925T202544Z"
SALT = 9211
POOL = tuple(range(2009, 2018))
HALF_CLOCK = {1: 3600.0, 3: 1800.0}
FORMS = {"const": (), "half": ("half",), "k35": ("k35",), "tb25": ("tb25",), "tb25_half": ("tb25", "half")}
EVENTS = ("recv", "kick")


def enabled():
    return os.environ.get("RTD") == "1"


def scored(p):
    ishome = (p.posteam == p.home_team).to_numpy()
    ps = p.posteam_score.to_numpy(float)
    ds = ps - p.score_differential.to_numpy(float)
    hs = np.where(ishome, ps, ds)
    as_ = np.where(ishome, ds, ps)
    g = p.game_id.to_numpy()
    last = np.r_[g[1:] != g[:-1], True]
    post = p.posteam_score_post.to_numpy(float)
    nh = np.r_[hs[1:], 0.0]
    na = np.r_[as_[1:], 0.0]
    pp = np.where(np.isnan(post), ps, post)
    nh = np.where(last, np.where(ishome, pp, hs), nh)
    na = np.where(last, np.where(ishome, as_, pp), na)
    dh, da = nh - hs, na - as_
    return np.where(ishome, dh, da), np.where(ishome, da, dh), last


def load_season(s):
    cols = ["game_id", "play_id", "season_type", "posteam", "home_team", "play_type", "qtr", "posteam_score", "score_differential", "posteam_score_post", "game_seconds_remaining"]
    p = pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=cols)
    p = p[(p.season_type == "REG") & p.posteam.notna() & p.score_differential.notna() & p.posteam_score.notna()].copy()
    return p.sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)


def real_events():
    out, clock = [], []
    for s in POOL:
        p = load_season(s)
        pos_d, def_d, last = scored(p)
        gsr = p.game_seconds_remaining.to_numpy(float)
        ngsr = np.where(last, np.nan, np.r_[gsr[1:], np.nan])
        ko = (p.play_type == "kickoff").to_numpy()
        q = p.qtr.to_numpy()
        opening = ko & (((q == 1) & (gsr == 3600.0)) | ((q == 3) & (gsr == 1800.0)))
        d = pd.DataFrame({"season": s, "game_id": p.game_id.to_numpy(), "half": (q == 3).astype(int), "recv": (pos_d >= 6).astype(int), "kick": (def_d >= 6).astype(int)})
        out.append(d[opening])
        nq = np.r_[q[1:], 0]
        tdr = ko & ((pos_d >= 6) | (def_d >= 6)) & ~last & (nq == q)
        clock.append((gsr - ngsr)[tdr])
    E = pd.concat(out, ignore_index=True)
    E["k35"] = (E.season >= 2011).astype(int)
    E["tb25"] = (E.season >= 2016).astype(int)
    x = np.concatenate(clock)
    return E, x[np.isfinite(x) & (x >= 0)]


def design(d, cols):
    return np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols])


def fit_logit(X, y):
    from sklearn.linear_model import LogisticRegression

    if X.shape[1] == 1:
        p = min(max(float(y.mean()), 1e-9), 1 - 1e-9)
        return np.array([np.log(p / (1.0 - p))])
    return LogisticRegression(C=1e9, max_iter=5000, fit_intercept=False).fit(X, y).coef_[0]


def loglik(X, y, b):
    z = X @ b
    return float(np.sum(y * z - np.logaddexp(0.0, z)))


def loso(E, ev, cols):
    per, coefs = {}, {}
    for s in POOL:
        te, tr = E[E.season == s], E[E.season != s]
        b = fit_logit(design(tr, cols), tr[ev].to_numpy(float))
        per[s] = loglik(design(te, cols), te[ev].to_numpy(float), b)
        coefs[s] = b.tolist()
    return sum(per.values()), per, coefs


def pooled_rates(E, cols, b):
    r = {}
    for h in (0, 1):
        z = design(E[E.half == h], cols) @ np.asarray(b)
        r[str(h)] = float(np.mean(1.0 / (1.0 + np.exp(-z))))
    return r


def cmd_fit():
    E, clock = real_events()
    OUTD.mkdir(parents=True, exist_ok=True)
    K = pd.read_parquet(KICKS)
    out = {"seasons": list(POOL), "n_rows": int(len(E)), "events": {}, "elapsed": clock.tolist(), "pat_extra": [int(x) for x in K.pat_extra.dropna()]}
    lines = [f"opening/half kickoffs {len(E)} ({len(E) / E.game_id.nunique():.4f}/game); looks {len(FORMS) * len(EVENTS)} (forms {list(FORMS)} x events {list(EVENTS)})"]
    for ev in EVENTS:
        res = {}
        for nm, cols in FORMS.items():
            ll, per, coefs = loso(E, ev, cols)
            res[nm] = dict(ll=ll, per=per, coefs=coefs, cols=list(cols))
        base = res["const"]
        for nm, r in res.items():
            r["dll"] = r["ll"] - base["ll"]
            r["wins"] = int(sum(r["per"][s] > base["per"][s] for s in POOL))
            lines.append(f"{ev} form {nm} events {int(E[ev].sum())}/{len(E)} LOSO ll {r['ll']:.3f} dll v const {r['dll']:+.3f} fold wins {r['wins']}/9")
        best = max(res, key=lambda k: res[k]["ll"])
        b = fit_logit(design(E, res[best]["cols"]), E[ev].to_numpy(float))
        rates = pooled_rates(E, res[best]["cols"], b)
        out["events"][ev] = dict(form=best, cols=res[best]["cols"], coef=b.tolist(), count=int(E[ev].sum()), n=int(len(E)), pooled_rate_by_half=rates, loso={k: {kk: vv for kk, vv in v.items() if kk != "coefs"} for k, v in res.items()}, fold_coef={k: v["coefs"] for k, v in res.items()})
        lines.append(f"{ev} chosen {best} coef {np.round(b, 4).tolist()} pooled rate by half {rates}")
        for s in POOL:
            sub = E[E.season == s]
            lines.append(f"  fold {s} held-out ll {res[best]['per'][s]:.4f} (const {res['const']['per'][s]:.4f}) events {int(sub[ev].sum())}/{len(sub)} train coef {np.round(res[best]['coefs'][s], 3).tolist()}")
    lines.append(f"kickoff TD play clock n {len(clock)} mean {clock.mean():.1f} p10 {np.percentile(clock, 10):.0f} p90 {np.percentile(clock, 90):.0f}; pat_extra n {len(out['pat_extra'])} mean {np.mean(out['pat_extra']):.3f}")
    FIT.write_text(json.dumps(out, indent=1))
    (OUTD / "an.txt").write_text(chr(10).join(lines))
    print(chr(10).join(lines))


def install_rtd():
    import mod25d_variance as dv

    fit = json.loads(FIT.read_text(encoding="utf-8"))
    K = pd.read_parquet(KICKS)
    ok = np.flatnonzero(K.rtd.to_numpy(int) == 0)
    ret = K.retained.to_numpy(int)[ok]
    nfp = K.nfp.to_numpy(float)[ok]
    kd = K.kd.to_numpy(float)[ok]
    gs = K.gsr.to_numpy(float)[ok]
    sk, st_ = float(K.kd.std()), float(K.gsr.std())
    m = int(np.sqrt(len(ok)))
    pe = np.asarray(fit["pat_extra"], int)
    el = np.asarray(fit["elapsed"], float)
    pr = {h: fit["events"]["recv"]["pooled_rate_by_half"][str(h)] for h in (0, 1)}
    pk = {h: fit["events"]["kick"]["pooled_rate_by_half"][str(h)] for h in (0, 1)}
    state = {"k": None, "rng": None, "prev": None}
    base = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        drawn = base(down, distance, yardline, score_diff, qtr, clock_val, drawn)
        if state["k"] != dv._G.get("task_key"):
            state["k"] = dv._G.get("task_key")
            state["rng"] = dv.task_rng(SALT, int(dv._G["cfg"].get("seed", 3)))
            state["prev"] = None
        prev = state["prev"]
        state["prev"] = float(clock_val)
        first = qtr in HALF_CLOCK and float(clock_val) == HALF_CLOCK[qtr] and prev != float(clock_val)
        if not first:
            return drawn
        rng = state["rng"]
        h = 1 if qtr == 3 else 0
        u, b_i, j_i, e_i = rng.random(), rng.random(), rng.random(), rng.random()
        if u >= pr[h] + pk[h]:
            return drawn
        scorer_off = u < pr[h]
        bonus = float(pe[min(int(b_i * len(pe)), len(pe) - 1)])
        e = float(el[min(int(e_i * len(el)), len(el) - 1)])
        kdv = (float(score_diff) if scorer_off else -float(score_diff)) + 6.0 + bonus
        d2 = ((kd - kdv) / sk) ** 2 + ((gs - max(float(clock_val) - e, 0.0)) / st_) ** 2
        j = int(np.argpartition(d2, m - 1)[:m][min(int(j_i * m), m - 1)])
        holder_off = bool(ret[j]) == scorer_off
        fp = float(nfp[j])
        new = dict(drawn)
        new["points_off"] = 6.0 + bonus if scorer_off else 0.0
        new["points_def"] = 0.0 if scorer_off else 6.0 + bonus
        new["flip"] = not holder_off
        new["next_down"] = 1.0
        new["next_distance"] = min(10.0, fp)
        new["next_yardline"] = fp
        new["clock_elapsed"] = e
        new["play_type_code"] = 6
        new["off_to_used"] = 0
        new["def_to_used"] = 0
        new["rtd_ev"] = True
        return new

    dv._G["pol"] = pol


if __name__ == "__main__":
    cmd_fit()
