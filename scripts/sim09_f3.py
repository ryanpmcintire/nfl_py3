import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
NV = Path("C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/f6b7873f-565c-422a-9dee-b6963fcaad49/scratchpad/nv")
OUT = REPO / "artifacts" / "sim09" / "f3"
PRE = {"False Start", "Delay of Game", "Illegal Formation", "Illegal Shift", "Defensive Offside", "Neutral Zone Infraction", "Encroachment", "Defensive 12 On-field", "Offensive 12 On-field", "Illegal Motion", "Illegal Substitution", "Defensive Delay of Game", "Offensive Offside"}
GROUPS = ["pre_off", "pre_def", "in_off", "in_def"]
FEATS = ["one", "d2", "d3", "d4", "logdist", "yl", "trail", "sd", "q2", "q3", "q4", "ot", "home", "nh", "sg", "ispass", "deep"]
PEN = re.compile(r"(?i)Penalty on (\w+)-[^,]+, ([^,]+), (declined|offsetting)")
COLS = ["game_id", "old_game_id", "play_id", "season", "week", "posteam", "defteam", "home_team", "away_team", "down", "ydstogo", "yardline_100", "score_differential", "qtr", "game_seconds_remaining", "play_type", "no_huddle", "shotgun", "qb_dropback", "qb_kneel", "qb_spike", "air_yards", "pass_length", "yards_gained", "penalty", "penalty_type", "penalty_yards", "penalty_team", "first_down_penalty", "desc", "touchdown", "interception", "fumble_lost", "sack", "posteam_timeouts_remaining", "defteam_timeouts_remaining"]


COLS2 = COLS + ["incomplete_pass"]
HAZ2 = OUT / "f3_haz2.json"
R2 = REPO / "artifacts" / "mod25e3" / "f3w"


def w2():
    return os.environ.get("F3W") == "2"


def fx(down, dist, yl, sd, qtr, home, nh, sg, ispass, deep, sdsd):
    down = np.asarray(down, float)
    n = len(down)
    sdv = np.asarray(sd, float)
    q = np.asarray(qtr)
    cols = [np.ones(n), down == 2, down == 3, down == 4, np.log(np.maximum(np.asarray(dist, float), 1.0)), np.asarray(yl, float) / 100.0, sdv < 0, sdv / sdsd, q == 2, q == 3, q == 4, q >= 5, np.asarray(home, float), np.asarray(nh, float), np.asarray(sg, float), np.asarray(ispass, float), np.asarray(deep, float)]
    return np.column_stack([np.asarray(c, float) for c in cols])


def load_nv(seasons=range(2009, 2018), cols=COLS):
    d = pd.concat([pd.read_parquet(NV / f"pbp_{s}.parquet", columns=cols) for s in seasons], ignore_index=True)
    return d.drop_duplicates(["game_id", "play_id"])


def snap_table(d):
    s = d[d.posteam.notna() & d.down.isin([1, 2, 3, 4]) & d.play_type.isin(["run", "pass", "no_play"]) & (d.qb_kneel != 1) & (d.qb_spike != 1) & d.score_differential.notna()].copy().reset_index(drop=True)
    desc = s["desc"].fillna("")
    nopl = s.play_type == "no_play"
    s["ispass"] = ((s.play_type == "pass") | (nopl & ((s.qb_dropback == 1) | desc.str.contains(r"pass|sacked|scrambl", case=False)))).astype(int)
    s["deep"] = ((s.pass_length == "deep") | desc.str.contains(r"\bdeep\b", case=False)).astype(int) * s.ispass
    s["home"] = (s.posteam == s.home_team).astype(int)
    s["nh"] = s.no_huddle.fillna(0).astype(int)
    s["sg"] = s.shotgun.fillna(0).astype(int)
    acc = (s.penalty == 1) & s.penalty_type.notna() & s.penalty_team.notna()
    s["acc"] = acc
    s["aoff"] = acc & (s.penalty_team == s.posteam)
    dec = desc.str.extractall(PEN).reset_index()
    dec = dec.drop_duplicates("level_0")
    li = dec["level_0"].to_numpy()
    s["dtype"] = None
    s["doff"] = False
    s.loc[li, "dtype"] = dec[1].to_numpy()
    s.loc[li, "doff"] = dec[0].to_numpy() == s.loc[li, "posteam"].to_numpy()
    ftype = s["penalty_type"].where(s["acc"], s["dtype"])
    foff = np.where(s["acc"], s["aoff"], s["doff"])
    s["ftype"] = ftype
    s["group"] = np.where(ftype.isna(), None, np.where(ftype.isin(PRE), "pre", "in") + np.where(foff, "_off", "_def"))
    return s


def Xg(X, g):
    if g.startswith("pre"):
        X = X.copy()
        X[:, FEATS.index("ispass")] = 0.0
        X[:, FEATS.index("deep")] = 0.0
        X[:, len(FEATS):] = 0.0
    return X


def inc_of(s):
    desc = s["desc"].fillna("").str.lower()
    isp = s.play_type == "pass"
    npl = s.play_type == "no_play"
    return ((isp & (s.incomplete_pass == 1)) | (npl & (s.ispass == 1) & desc.str.contains("incomplete"))).astype(float).to_numpy()


def aug(x, inc):
    inc = np.asarray(inc, float)
    return np.column_stack([x, inc, inc * x[:, FEATS.index("deep")]])


def cmd_haz2(args):
    P = json.load(open(OUT / "f3_params.json"))
    s = snap_table(load_nv(cols=COLS2))
    X0 = X_of(s, P["sdsd"])
    inc = inc_of(s)
    Xa = aug(X0, inc)
    seasons = sorted(s.season.unique())
    out = {}
    for g in GROUPS:
        C = P["hazard"][g]["C"]
        y = (s.group == g).to_numpy().astype(float)
        XB, XA = Xg(X0, g), Xg(Xa, g)
        llb = lla = 0.0
        co = []
        for sn in seasons:
            te = (s.season == sn).to_numpy()
            wb = fit_lr(XB[~te], y[~te], C)
            wa = fit_lr(XA[~te], y[~te], C)
            llb += logloss(y[te], predict(wb, XB[te])) * te.sum()
            lla += logloss(y[te], predict(wa, XA[te])) * te.sum()
            co.append(wa[-2:].tolist())
        n = len(y)
        co = np.array(co)
        w = fit_lr(XA, y, C)
        out[g] = dict(C=C, w=w.tolist(), loso_ll_blind=llb / n, loso_ll_aware=lla / n, fold_inc=co[:, 0].tolist(), fold_incdeep=co[:, 1].tolist())
        print(f"{g:7s} LOSO LL blind {llb / n:.5f} aware {lla / n:.5f} gain {(llb - lla) / n:.5f} inc coef mean {co[:, 0].mean():+.3f} sd {co[:, 0].std():.3f} sign+ {int((co[:, 0] > 0).sum())}/{len(co)}; incdeep mean {co[:, 1].mean():+.3f} sd {co[:, 1].std():.3f} sign+ {int((co[:, 1] > 0).sum())}/{len(co)}; full-fit inc {w[-2]:+.3f} incdeep {w[-1]:+.3f}", flush=True)
    json.dump(out, open(HAZ2, "w"))


def X_of(s, sdsd):
    return fx(s.down, s.ydstogo, s.yardline_100, s.score_differential, s.qtr, s.home, s.nh, s.sg, s.ispass, s.deep, sdsd)


def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_lr(X, y, C):
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=C, max_iter=2000)
    m.fit(X[:, 1:], y)
    return np.concatenate([m.intercept_, m.coef_[0]])


def predict(w, X):
    return 1.0 / (1.0 + np.exp(-X @ w))


def hazard_fit(s, sdsd, log):
    X0 = X_of(s, sdsd)
    tr = (s.season <= 2012).to_numpy()
    va = ((s.season >= 2013) & (s.season <= 2014)).to_numpy()
    te = (s.season >= 2015).to_numpy()
    fit_tv = (s.season <= 2014).to_numpy()
    res = {}
    for g in GROUPS:
        X = Xg(X0, g)
        y = (s.group == g).to_numpy().astype(float)
        best = None
        for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0):
            w = fit_lr(X[tr], y[tr], C)
            ll = logloss(y[va], predict(w, X[va]))
            if best is None or ll < best[0]:
                best = (ll, C)
        C = best[1]
        w1 = fit_lr(X[fit_tv], y[fit_tv], C)
        base = float(y[fit_tv].mean())
        llm = logloss(y[te], predict(w1, X[te]))
        llb = logloss(y[te], np.full(te.sum(), base))
        w = fit_lr(X, y, C)
        res[g] = dict(C=C, w=w.tolist(), rate=float(y.mean()), test_ll_model=llm, test_ll_const=llb, test_gain=llb - llm)
        log(f"hazard {g}: rate {y.mean():.4f} C {C} test LL model {llm:.5f} const {llb:.5f} gain {llb - llm:.5f}")
        log("  coefs " + ", ".join(f"{n}={v:+.3f}" for n, v in zip(FEATS, w)))
    return res


def type_tables(s, log):
    a = s[s.acc]
    rows = {}
    dec = s[~s.acc & s.ftype.notna()]
    for t, g in a.groupby("ftype"):
        if len(g) < 40:
            continue
        counted = float(g.play_type.isin(["run", "pass"]).mean())
        mid = g[(g.yardline_100 >= 30) & (g.yardline_100 <= 70)]
        pmf = mid.penalty_yards.round().value_counts(normalize=True).sort_index()
        short = g[g.penalty_yards < g.ydstogo]
        af = float(short.first_down_penalty.fillna(0).mean()) if len(short) else float("nan")
        rows[t] = dict(n=int(len(g)), declined=int((dec.ftype == t).sum()), offense_share=float(g.aoff.mean()), counted=counted, pre=bool(t in PRE), auto_first_share=af, pmf_yards=[float(k) for k in pmf.index], pmf_p=[float(v) for v in pmf.values], mean_yards=float(mid.penalty_yards.mean()), n_short=int(len(short)))
    for t, r in sorted(rows.items(), key=lambda kv: -kv[1]["n"])[:24]:
        log(f"type {t:30s} n {r['n']:5d} declined {r['declined']:4d} off {r['offense_share']:.2f} counted {r['counted']:.2f} autoFD {r['auto_first_share']:.2f} mean_yds {r['mean_yards']:.1f}")
    return rows


def rule_checks(s, log):
    a = s[s.acc & s.aoff & ~s.play_type.isin(["pass", "run"])]
    near = a[(a.yardline_100 >= 90) & a.penalty_yards.notna()]
    half = (100 - near.yardline_100) / 2
    log(f"offensive fouls replayed from own 10 or deeper: n {len(near)} share yards == half distance {float((np.abs(near.penalty_yards - half) <= 0.5).mean()):.3f}, share exceeding half distance {float((near.penalty_yards > half + 0.5).mean()):.3f}")
    nd = s[s.acc & ~s.aoff & s.ftype.isin(["Defensive Offside", "Neutral Zone Infraction", "Encroachment"]) & (s.yardline_100 <= 9)]
    log(f"defensive pre-snap from opp 9 or closer: n {len(nd)} share yards == half distance {float((np.abs(nd.penalty_yards - nd.yardline_100 / 2) <= 0.5).mean()):.3f}")
    dp = s[s.acc & (s.ftype == "Defensive Pass Interference")]
    log(f"DPI: n {len(dp)} yards exceeding yardline_100 share {float((dp.penalty_yards > dp.yardline_100).mean()):.4f}, yards==yl-1 share {float((dp.penalty_yards == dp.yardline_100 - 1).mean()):.3f}, counted-play share {float(dp.play_type.isin(['pass']).mean()):.3f}")
    dpn = dp[dp.yardline_100 <= 20]
    log(f"DPI inside 20: n {len(dpn)} mean yards/yl {float((dpn.penalty_yards / dpn.yardline_100).mean()):.3f}")
    log("DPI yards by deep flag: " + str(dp.groupby("deep").penalty_yards.agg(["mean", "median", "count"]).round(1).to_dict()))
    allacc = s[s.acc & ~s.aoff & ~s.ftype.isin(PRE)]
    log(f"defensive in-play accepted: first_down_penalty share {float(allacc.first_down_penalty.fillna(0).mean()):.3f}")
    n_comm = int(s.ftype.notna().sum())
    n_acc = int(s.acc.sum())
    log(f"snaps {len(s)} committed {n_comm} accepted {n_acc} accept share {n_acc / n_comm:.3f}")
    log("groups " + str(s.groupby("group").size().to_dict()))
    log("accepted share by group " + str(s[s.group.notna()].groupby("group").acc.mean().round(3).to_dict()))


def latents(s, haz, sdsd, log):
    X = X_of(s, sdsd)
    s = s.copy()
    out = {}
    for g in GROUPS:
        s["E"] = predict(np.array(haz[g]["w"]), Xg(X, g))
        s["O"] = (s.group == g).astype(float)
        col = "posteam" if g.endswith("_off") else "defteam"
        u = s.groupby(["season", col]).agg(O=("O", "sum"), E=("E", "sum"))
        r = u.O / u.E
        tau2 = float(r.var() - (1.0 / u.E).mean())
        s["half"] = s.week % 2
        h = s.groupby(["season", col, "half"]).agg(O=("O", "sum"), E=("E", "sum")).reset_index()
        hh = h.pivot(index=["season", col], columns="half", values=["O", "E"]).dropna()
        sh = float(np.corrcoef(hh[("O", 0)] / hh[("E", 0)], hh[("O", 1)] / hh[("E", 1)])[0, 1])
        ru = r.reset_index(name="r")
        ru["next"] = ru["season"] + 1
        m = ru.merge(ru, left_on=[col, "next"], right_on=[col, "season"], suffixes=("", "_n"))
        co = float(np.corrcoef(m["r"], m["r_n"])[0, 1])
        out[g] = dict(team_tau2=tau2, team_tau=float(np.sqrt(max(tau2, 0.0))), split_half_r=sh, carry_over_r=co, n_units=int(len(u)), mean_E=float(u.E.mean()))
        log(f"team latent {g}: units {len(u)} meanE {u.E.mean():.1f} excess var tau2 {tau2:+.4f} (tau {np.sqrt(max(tau2, 0)):.3f}) split-half r {sh:.3f} next-season r {co:.3f}")
    return out


def load_refs(log):
    parts = []
    for p in (REPO / "data" / "processed" / "officials_pfr_wayback").glob("*/officials_2009_2014.parquet"):
        w = pd.read_parquet(p)
        log("wayback cols " + str(w.columns.tolist()))
        parts.append(w)
    raw = pd.read_parquet(sorted((REPO / "data" / "raw" / "officials").glob("*/officials.parquet"))[-1])
    parts.append(raw)
    keep = []
    for p in parts:
        if "official_name" in p and "position" in p and "game_id" in p:
            keep.append(p[["game_id", "official_name", "position"]])
    o = pd.concat(keep, ignore_index=True)
    pos = o.position.astype(str)
    o = o[pos.str.contains("Referee", case=False) & ~pos.str.contains("Umpire|Replay|Head Linesman", case=False)].drop_duplicates("game_id")
    o["game_id"] = o["game_id"].astype(str)
    return o


def crews(s, haz, sdsd, log):
    o = load_refs(log)
    s = s.copy()
    s["gk"] = s["old_game_id"].astype(str)
    m = s.merge(o[["game_id", "official_name"]].rename(columns={"game_id": "gk"}), on="gk", how="inner")
    log(f"games with referee {m.gk.nunique()} of {s.gk.nunique()}")
    X = X_of(m, sdsd)
    out = {}
    for g in GROUPS:
        m["E"] = predict(np.array(haz[g]["w"]), Xg(X, g))
        m["O"] = (m.group == g).astype(float)
        u = m.groupby("official_name").agg(O=("O", "sum"), E=("E", "sum"), ng=("gk", "nunique"))
        u = u[u.ng >= 16]
        r = u.O / u.E
        tau2 = float(r.var() - (1.0 / u.E).mean())
        m["half"] = pd.factorize(m.gk)[0] % 2
        hh = m.groupby(["official_name", "half"]).agg(O=("O", "sum"), E=("E", "sum")).reset_index().pivot(index="official_name", columns="half", values=["O", "E"]).dropna()
        hh = hh.loc[hh.index.isin(u.index)]
        sh = float(np.corrcoef(hh[("O", 0)] / hh[("E", 0)], hh[("O", 1)] / hh[("E", 1)])[0, 1])
        out[g] = dict(crew_tau2=tau2, crew_tau=float(np.sqrt(max(tau2, 0.0))), split_half_r=sh, n_refs=int(len(u)))
        log(f"crew latent {g}: refs {len(u)} (>=16 games) excess var tau2 {tau2:+.5f} (tau {np.sqrt(max(tau2, 0)):.3f}) split-half r {sh:.3f}")
    return out


def cmd_measure(args):
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(x):
        print(x, flush=True)
        lines.append(x)

    s = snap_table(load_nv())
    sdsd = float(s.score_differential.std())
    log(f"snaps {len(s)} sd std {sdsd:.2f}")
    deepmin = float(s[(s.deep == 1) & s.air_yards.notna() & (s.play_type == "pass")].air_yards.quantile(0.01))
    log(f"deep pass air_yards 1st percentile {deepmin}")
    rule_checks(s, log)
    tt = type_tables(s, log)
    haz = hazard_fit(s, sdsd, log)
    lat = latents(s, haz, sdsd, log)
    cr = crews(s, haz, sdsd, log)
    c = s[s.ftype.notna()].copy()
    c["key"] = np.where(c.group.str.startswith("pre"), "00", c.ispass.astype(str) + c.deep.astype(str))
    tp = {}
    for (g, k), z in c.groupby(["group", "key"]):
        z = z[z.ftype.isin(list(tt))]
        v = z.ftype.value_counts(normalize=True)
        tp.setdefault(g, {})[k] = dict(types=list(v.index), p=[float(x) for x in v.values], n=int(len(z)))
    json.dump(dict(typepmf=tp, sdsd=sdsd, deep_air_min=deepmin, hazard=haz, types=tt, team=lat, crew=cr, feats=FEATS), open(OUT / "f3_params.json", "w"))
    (OUT / "measure.log").write_text("\n".join(lines))

EPM = REPO / "artifacts" / "mod25d" / "ep_model.joblib"
AFEATS = ["grp_pre_off", "grp_pre_def", "grp_in_off", "grp_in_def", "g_dec_def", "g_dec_off", "stf", "sd_dec", "late", "g_late"]
HALF_SECONDS = 1800.0


def ep_model():
    import joblib

    return joblib.load(EPM)


class EPQ:
    def __init__(self, c):
        self.c = c
        self.rows = []
        self.res = []

    def add(self, down, dist, yl, swap=False):
        c = self.c
        sd = -c["sd"] if swap else c["sd"]
        a = c["to_d"] if swap else c["to_o"]
        b = c["to_o"] if swap else c["to_d"]
        cols = np.broadcast_arrays(*[np.asarray(v, float) for v in (down, dist, yl, c["hs"], sd, a, b, c["half"])])
        self.rows.append(np.column_stack(cols))
        return len(self.rows) - 1

    def run(self, m):
        X = np.concatenate(self.rows)
        p = m.predict(X)
        self.res = np.split(p, np.cumsum([len(r) for r in self.rows])[:-1])


def af_val(q, c, yd, pfd, same):
    yl, dist, down = c["yl"], c["dist"], c["down"]
    same = np.broadcast_to(np.asarray(same, bool), yl.shape)
    yn = np.clip(yl - yd, 1.0, 99.0)
    hf = q.add(1, np.minimum(10.0, yn), yn)
    nd = np.where(same, down, np.minimum(down + 1, 4))
    hn = q.add(nd, np.maximum(dist - yd, 1.0), yn)
    ht = q.add(1, 10.0, 100.0 - yn, True)

    def f():
        fd = np.where(yd >= dist, 1.0, pfd)
        ef, en, et = q.res[hf], q.res[hn], q.res[ht]
        en = np.where(np.logical_not(same) & (down >= 4), -et, en)
        v = fd * ef + (1.0 - fd) * en
        return np.where(yd >= yl, 7.0, v), -et

    return f


def norm_pts(v):
    return np.where(v >= 6, 7.0, v)


def gain_dec(m, c, y, po, pdf, turn, nom, af, counted, isoff, dpi):
    q = EPQ(c)
    yl, dist = c["yl"], c["dist"]
    vp = af_val(q, c, y, 0.0, False)
    cap_o = np.maximum((100.0 - yl) / 2.0, 0.0)
    cap_d = np.maximum(np.where(dpi, yl - 1.0, yl / 2.0), 0.0)
    amt_r = np.minimum(nom, np.where(isoff, cap_o, cap_d))
    vr = af_val(q, c, np.where(isoff, -amt_r, amt_r), np.where(isoff, 0.0, af), True)
    e = yl - y
    cap_oc = np.maximum((100.0 - e) / 2.0, 0.0)
    cap_dc = np.maximum(np.where(dpi, e - 1.0, e / 2.0), 0.0)
    amt_c = np.minimum(nom, np.where(isoff, cap_oc, cap_dc))
    yc = np.where(isoff, y - amt_c, y + amt_c)
    vc = af_val(q, c, yc, np.where(isoff, 0.0, np.where(yc < dist, af, 1.0)), False)
    q.run(m)
    vpl, tvp = vp()
    play = np.where(po > 0, norm_pts(po), np.where(pdf > 0, -norm_pts(pdf), np.where(turn, tvp, vpl)))
    stf = (po > 0) | (pdf > 0) | turn
    vcn = np.where(stf, play, vc()[0])
    acc = counted * vcn + (1.0 - counted) * vr()[0]
    dv = acc - play
    return np.where(isoff, -dv, dv), stf


def acc_X(g, grp, stf, sd, hs, isoff, sdsd):
    n = len(g)
    oh = np.zeros((n, 4))
    oh[np.arange(n), grp] = 1.0
    sd_dec = np.where(isoff, -sd, sd) / sdsd
    late = 1.0 - np.clip(hs, 0.0, HALF_SECONDS) / HALF_SECONDS
    return np.column_stack([oh, g * isoff, g * np.logical_not(isoff), stf.astype(float), sd_dec, late, g * late])


def nll_grad(w, Xo, yo, Xl, lam, pm):
    eta = Xo @ w
    p = 1.0 / (1.0 + np.exp(-eta))
    ll = -np.sum(np.logaddexp(0, -eta) * yo + np.logaddexp(0, eta) * (1 - yo))
    grad = Xo.T @ (yo - p)
    if len(Xl):
        pl = 1.0 / (1.0 + np.exp(-(Xl @ w)))
        mm = np.clip(pl.mean(1), 1e-12, None)
        ll += np.sum(np.log(mm))
        wt = pl * (1 - pl) / (Xl.shape[1] * mm[:, None])
        grad += np.einsum("ik,ikf->f", wt, Xl)
    n = len(Xo) + len(Xl)
    return -ll / n + 0.5 * lam * np.sum(pm * w * w), -grad / n + lam * pm * w


def acc_ll(w, Xo, yo, Xl):
    return nll_grad(w, Xo, yo, Xl, 0.0, np.zeros(len(w)))[0]


snap_table_clean = [None]


def build_accept_tables(s, P, K, seed):
    m = ep_model()
    clean = snap_table_clean[0]
    s = s[s.ftype.notna() & s.ftype.isin(list(P["types"])) & ~s["desc"].fillna("").str.contains("(?i)offsetting")].copy()
    s = s[s.group.notna()]
    s = s[s.acc | s.play_type.isin(["run", "pass"])].copy()
    isacc = s.acc.to_numpy()
    lat = (isacc & (s.play_type == "no_play")).to_numpy()
    info = s.ftype.map(P["types"])
    s["nom"] = np.where(isacc, s.penalty_yards.abs(), info.map(lambda r: r["mean_yards"]))
    s["af"] = info.map(lambda r: r["auto_first_share"] if np.isfinite(r["auto_first_share"]) else 0.0)
    s["cm"] = np.where(isacc, s.play_type.isin(["run", "pass"]).astype(float), info.map(lambda r: r["counted"])) * (~s.group.str.startswith("pre")).astype(float)
    s["isoff"] = s.group.str.endswith("_off")
    s["dpi"] = s.ftype == "Defensive Pass Interference"
    s["grp"] = s.group.map({g: i for i, g in enumerate(GROUPS)})
    hs0 = np.where(s.qtr <= 2, s.game_seconds_remaining - HALF_SECONDS, s.game_seconds_remaining)
    s["hs"] = np.maximum(hs0, 0.0)
    s["half"] = np.where(s.qtr >= 5, 3, np.where(s.qtr <= 2, 1, 2))
    s["to_o"] = s.posteam_timeouts_remaining.fillna(3.0)
    s["to_d"] = s.defteam_timeouts_remaining.fillna(3.0)
    s["turn"] = ((s.interception == 1) | (s.fumble_lost == 1)).astype(bool)
    s["po"] = np.where((s.touchdown == 1) & ~s.turn, 7.0, 0.0)
    s["pdf"] = np.where((s.touchdown == 1) & s.turn, 7.0, 0.0)
    s["y"] = s.yards_gained.fillna(0.0)
    edges = np.quantile(clean.ydstogo, [0.25, 0.5, 0.75])
    clean = clean.assign(b=np.digitize(clean.ydstogo, edges))
    cells = {k: v for k, v in clean.groupby(["down", "b", "ispass"])}
    rng = np.random.default_rng(seed)
    sdsd = P["sdsd"]

    def ctx(d):
        return dict(down=d.down.to_numpy(float), dist=d.ydstogo.to_numpy(float), yl=d.yardline_100.to_numpy(float), hs=d.hs.to_numpy(float), half=d.half.to_numpy(float), sd=d.score_differential.to_numpy(float), to_o=d.to_o.to_numpy(float), to_d=d.to_d.to_numpy(float))

    o = s[~lat]
    co = ctx(o)
    g, stf = gain_dec(m, co, o.y.to_numpy(float), o.po.to_numpy(float), o.pdf.to_numpy(float), o.turn.to_numpy(bool), o.nom.to_numpy(float), o.af.to_numpy(float), o.cm.to_numpy(float), o.isoff.to_numpy(bool), o.dpi.to_numpy(bool))
    Xo = acc_X(g, o.grp.to_numpy(int), stf, co["sd"], co["hs"], o.isoff.to_numpy(bool), sdsd)
    yo = o.acc.to_numpy(float)
    L = s[lat].reset_index(drop=True)
    cl = ctx(L)
    ys = np.zeros((len(L), K))
    pos = np.zeros((len(L), K))
    pds = np.zeros((len(L), K))
    trn = np.zeros((len(L), K), dtype=bool)
    for i, r in enumerate(L.itertuples()):
        pl = cells.get((r.down, int(np.digitize(r.ydstogo, edges)), r.ispass))
        if pl is None:
            pl = clean[(clean.down == r.down) & (clean.ispass == r.ispass)]
        a = pl.iloc[rng.integers(len(pl), size=K)]
        ys[i] = a.y.to_numpy()
        pos[i] = a.po.to_numpy()
        pds[i] = a.pdf.to_numpy()
        trn[i] = a.turn.to_numpy()
    rep = lambda v: np.repeat(np.asarray(v), K)
    cf = {k: rep(v) for k, v in cl.items()}
    gl, sl = gain_dec(m, cf, ys.ravel(), pos.ravel(), pds.ravel(), trn.ravel(), rep(L.nom.to_numpy(float)), rep(L.af.to_numpy(float)), np.zeros(len(L) * K), rep(L.isoff.to_numpy(bool)), rep(L.dpi.to_numpy(bool)))
    Xl = acc_X(gl, rep(L.grp.to_numpy(int)), sl, cf["sd"], cf["hs"], rep(L.isoff.to_numpy(bool)), sdsd).reshape(len(L), K, -1)
    return dict(Xo=Xo, yo=yo, Xl=Xl, so=o.season.to_numpy(), sl=L.season.to_numpy(), go=o.group.to_numpy(), gl=L.group.to_numpy())


def cmd_accept(args):
    from scipy.optimize import minimize

    P = load_params()
    s = snap_table(load_nv())
    cl = s[s.group.isna() & s.play_type.isin(["run", "pass"])].copy()
    cl["turn"] = ((cl.interception == 1) | (cl.fumble_lost == 1)).astype(bool)
    cl["po"] = np.where((cl.touchdown == 1) & ~cl.turn, 7.0, 0.0)
    cl["pdf"] = np.where((cl.touchdown == 1) & cl.turn, 7.0, 0.0)
    cl["y"] = cl.yards_gained.fillna(0.0)
    snap_table_clean[0] = cl
    T = build_accept_tables(s, P, args.draws, 5)
    nf = len(AFEATS)
    pm = np.array([0.0] * 4 + [1.0] * (nf - 4))
    tr_o, tr_l = T["so"] <= 2014, T["sl"] <= 2014
    va_o, va_l = T["so"] >= 2015, T["sl"] >= 2015

    def fitw(mo, ml, lam):
        return minimize(nll_grad, np.zeros(nf), args=(T["Xo"][mo], T["yo"][mo], T["Xl"][ml], lam, pm), jac=True, method="L-BFGS-B").x

    base = {}
    for g in GROUPS:
        a = float(((T["go"] == g) & tr_o & (T["yo"] == 1)).sum() + (T["gl"] == g)[tr_l].sum())
        n = float(((T["go"] == g) & tr_o).sum() + (T["gl"] == g)[tr_l].sum())
        base[g] = a / n
    vo, vl = T["go"][va_o], T["gl"][va_l]
    yv = T["yo"][va_o]
    bo = np.array([base[g] for g in vo])
    bl = np.array([base[g] for g in vl])
    llb = -(np.sum(np.where(yv == 1, np.log(bo), np.log(1 - bo))) + np.sum(np.log(bl))) / (len(vo) + len(vl))
    best = None
    for lam in (0.0, 0.001, 0.01, 0.1, 1.0):
        w = fitw(tr_o, tr_l, lam)
        ll = acc_ll(w, T["Xo"][va_o], T["yo"][va_o], T["Xl"][va_l])
        print(f"lam {lam} held-out 2015-17 LL {ll:.5f} group-constant {llb:.5f} gain {llb - ll:.5f}", flush=True)
        if best is None or ll < best[0]:
            best = (ll, lam)
    lam = best[1]
    w = fitw(np.ones(len(T["yo"]), bool), np.ones(len(T["Xl"]), bool), lam)
    print("lam", lam, "coefs " + ", ".join(f"{n}={v:+.3f}" for n, v in zip(AFEATS, w)), flush=True)
    po = 1.0 / (1.0 + np.exp(-(T["Xo"] @ w)))
    pl = (1.0 / (1.0 + np.exp(-(T["Xl"] @ w)))).mean(1)
    for g in GROUPS:
        mo, ml = T["go"] == g, T["gl"] == g
        pred = (po[mo].sum() + pl[ml].sum()) / (mo.sum() + ml.sum())
        real = (T["yo"][mo].sum() + ml.sum()) / (mo.sum() + ml.sum())
        print(f"  {g}: committed {mo.sum() + ml.sum()} real accept share {real:.3f} fitted mean {pred:.3f}")
    json.dump(dict(feats=AFEATS, w=w.tolist(), lam=lam, heldout_ll=best[0], const_ll=llb, draws=args.draws), open(OUT / "f3_accept.json", "w"))


SIMS = {}
POOL_TR = [None]
WDRAWS = 1
WCHUNK = 20000


def removal_prob(P, pool, tr, a):
    ep = ep_model()
    acw = np.array(P["accept"]["w"])
    sdsd = P["sdsd"]
    deepmin = P["deep_air_min"]
    hw = [np.array(P["hazard"][g]["w"]) for g in GROUPS]
    incv = pool.inc.to_numpy(float) if w2() else None
    tnames = sorted(P["types"])
    code = np.asarray(a["play_type_code"])
    r = np.zeros(len(code))
    elig = np.flatnonzero(np.isin(code, (0, 1)) & ~pool.pen.to_numpy())
    rng = np.random.default_rng([0, 31])
    num = lambda c: tr[c].to_numpy(float)
    cols = dict(down=num("down_i"), dist=num("dist_raw"), yl=num("fp_raw"), sd=num("sc_raw"), qtr=num("qtr_actual"), gsr=num("gsr_actual"), to_o=np.nan_to_num(num("off_to_raw"), nan=3.0), to_d=np.nan_to_num(num("def_to_raw"), nan=3.0))
    yv = np.asarray(a["yards_gained"], float)
    pov = np.asarray(a["points_off"], float)
    pdv = np.asarray(a["points_def"], float)
    flipv = np.asarray(a["possession_flip"]).astype(bool)
    homev = np.asarray(a["is_home_off"]).astype(float)
    air = pool.air_yards.to_numpy(float)
    nh = pool.no_huddle.fillna(0).to_numpy(float)
    sg = pool.shotgun.fillna(0).to_numpy(float)
    for lo in range(0, len(elig), WCHUNK):
        i = elig[lo:lo + WCHUNK]
        n = len(i)
        c = {k: v[i] for k, v in cols.items()}
        ispass = code[i] == 1
        deep = ispass & np.isfinite(air[i]) & (air[i] >= deepmin)
        hs = np.where(c["qtr"] <= 2, np.maximum(c["gsr"] - HALF_SECONDS, 0.0), np.maximum(c["gsr"], 0.0))
        half = np.where(c["qtr"] >= 5, 3.0, np.where(c["qtr"] <= 2, 1.0, 2.0))
        cx = dict(down=c["down"], dist=c["dist"], yl=c["yl"], hs=hs, half=half, sd=c["sd"], to_o=c["to_o"], to_d=c["to_d"])
        y = yv[i]
        po = pov[i]
        pdf = pdv[i]
        turn = flipv[i] | (pdf > 0)
        x = fx(c["down"], c["dist"], c["yl"], c["sd"], c["qtr"], homev[i], nh[i], sg[i], ispass.astype(int), deep.astype(int), sdsd)
        key = np.where(ispass, np.where(deep, "11", "10"), "00")
        if w2():
            x = aug(x, incv[i])
        for k, g in enumerate(GROUPS):
            pre = g.startswith("pre")
            is_off = g.endswith("_off")
            h = np.minimum(predict(hw[k], Xg(x, g)), 0.5)
            tid = np.zeros(n, int)
            acc_sum = np.zeros(n)
            for d in range(WDRAWS):
                for kk in np.unique(key):
                    m = key == kk
                    kt = "00" if pre else kk
                    t = P["typepmf"][g].get(kt) or P["typepmf"][g].get("00") or next(iter(P["typepmf"][g].values()))
                    pr = np.array(t["p"], float)
                    pr = pr / pr.sum()
                    names = np.array(t["types"])
                    tid[m] = np.array([tnames.index(z) for z in names])[rng.choice(len(names), size=int(m.sum()), p=pr)]
                nom = np.zeros(n)
                af = np.zeros(n)
                cnt = np.zeros(n)
                dpi = np.zeros(n, bool)
                for ti in np.unique(tid):
                    m = tid == ti
                    info = P["types"][tnames[ti]]
                    pm = np.array(info["pmf_p"], float)
                    nom[m] = rng.choice(info["pmf_yards"], size=int(m.sum()), p=pm / pm.sum())
                    af[m] = info["auto_first_share"] if np.isfinite(info["auto_first_share"]) else 0.0
                    if not pre:
                        cnt[m] = (rng.random(int(m.sum())) < info["counted"]).astype(float)
                    dpi[m] = tnames[ti] == "Defensive Pass Interference"
                nom = np.where(dpi & ispass & np.isfinite(air[i]), np.maximum(air[i], 0.0), nom)
                isoff = np.full(n, is_off)
                gv, stf = gain_dec(ep, cx, y, po, pdf, turn, nom, af, cnt, isoff, dpi)
                xa = acc_X(gv, np.full(n, k), stf, cx["sd"], cx["hs"], isoff, sdsd)
                acc_sum += predict(acw, xa)
            r[i] += h * acc_sum / WDRAWS
    return np.clip(r, 0.0, 0.99)


def load_params():
    P = json.load(open(OUT / "f3_params.json"))
    P["accept"] = json.load(open(OUT / "f3_accept.json")) if (OUT / "f3_accept.json").exists() else None
    if w2():
        h2 = json.load(open(HAZ2))
        for g in GROUPS:
            P["hazard"][g]["w"] = h2[g]["w"]
    return P


def removal_cached(P, pool, tr, a):
    h = hashlib.sha1()
    for arr in (a["play_type_code"], a["yards_gained"], a["points_off"], a["points_def"], a["possession_flip"], a["is_home_off"], pool.inc.to_numpy(float), pool.pen.to_numpy(), pool.air_yards.to_numpy(float)):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(json.dumps([P["hazard"], P["accept"], P["typepmf"], P["types"]], sort_keys=True).encode())
    key = h.hexdigest()
    f = R2 / "r2.npy"
    kf = R2 / "r2.key"
    if f.exists() and kf.exists() and kf.read_text() == key:
        return np.load(f)
    r = removal_prob(P, pool, tr, a)
    R2.mkdir(parents=True, exist_ok=True)
    tf = R2 / f"r2.{os.getpid()}.npy"
    np.save(tf, r)
    os.replace(tf, f)
    kf.write_text(key)
    return r


def pool_arrays():
    import mod25d_variance as dv

    dv.sim.PBP_SNAPSHOT_DIR = dv.m25.SNAP
    pbp = dv.sim.load_reg_seasons(tuple(dv.TRAIN))
    tr = dv.sim.build_transition_frame(pbp)
    keys = tr[["game_id", "play_id"]].reset_index(drop=True)
    cols = ["game_id", "play_id", "posteam", "penalty", "penalty_type", "penalty_team", "penalty_yards", "first_down_penalty", "air_yards", "no_huddle", "shotgun", "play_type"]
    if w2():
        cols = cols + ["incomplete_pass"]
    nv = load_nv(range(min(dv.TRAIN), max(dv.TRAIN) + 1), COLS2 if w2() else COLS)[cols].drop_duplicates(["game_id", "play_id"])
    x = keys.merge(nv, on=["game_id", "play_id"], how="left")
    assert len(x) == len(keys)
    if w2():
        x["inc"] = ((x.play_type == "pass") & (x.incomplete_pass == 1)).astype(float)
    x["pen"] = ((x.penalty == 1) & x.penalty_type.notna() & x.penalty_team.notna()).to_numpy()
    x["poff"] = (x.penalty_team == x.posteam).to_numpy()
    x["grp"] = np.where(x.pen, np.where(x.penalty_type.isin(PRE), "pre", "in") + np.where(x.poff, "_off", "_def"), "")
    if os.environ.get("F3W") in ("1", "2"):
        POOL_TR[0] = tr
    return x


class Overlay:
    def __init__(self, P, pool, a, rng_key):
        self.P = P
        self.w = [np.array(P["hazard"][g]["w"]) for g in GROUPS]
        self.types = P["types"]
        self.tp = P["typepmf"]
        self.sdsd = P["sdsd"]
        self.deepmin = P["deep_air_min"]
        self.ep = ep_model()
        self.acw = np.array(P["accept"]["w"])
        self.air = pool.air_yards.to_numpy(float)
        self.nh = pool.no_huddle.fillna(0).to_numpy(float)
        self.sg = pool.shotgun.fillna(0).to_numpy(float)
        self.inc = pool.inc.to_numpy(float) if w2() else None
        self.el = {g: np.asarray(a["clock_elapsed"])[(pool.grp == g).to_numpy() & (np.asarray(a["play_type_code"]) == 6)] for g in GROUPS}
        self.ttau = np.array([P["team"][g]["team_tau"] for g in GROUPS])
        self.ctau = np.array([P["crew"][g]["crew_tau"] for g in GROUPS])
        self.rng = np.random.default_rng([0])
        self.ev = []
        self.task = (0, 0, 0)
        self.sched = None
        self.gi = -1
        self.tm = {}
        self.crew = np.ones(4)
        self.home = 0
        self.away = 0

    def start_task(self, task):
        self.task = (int(task[0]), int(task[1]), int(task[2]))
        self.sched = task[3]
        self.gi = -1
        self.tm = {}
        self.ev = []
        self.rng = np.random.default_rng([self.task[0], self.task[1], 777])

    def team(self, t):
        if t not in self.tm:
            z = np.random.default_rng([self.task[0], self.task[1], int(t), 91]).standard_normal(4)
            self.tm[t] = np.exp(self.ttau * z - 0.5 * self.ttau ** 2)
        return self.tm[t]

    def new_game(self):
        self.gi += 1
        week, h, a = self.sched[self.gi]
        self.home, self.away = h, a
        z = np.random.default_rng([self.task[0], self.task[1], self.gi, 92]).standard_normal(4)
        self.crew = np.exp(self.ctau * z - 0.5 * self.ctau ** 2)

    def haz(self, x, off_team, def_team):
        po = self.team(off_team)
        pd_ = self.team(def_team)
        out = []
        for k, g in enumerate(GROUPS):
            xx = Xg(x, g)
            p = float(predict(self.w[k], xx)[0])
            m = (po[k] if g.endswith("_off") else pd_[k]) * self.crew[k]
            out.append(min(p * m, 0.5))
        return out

    def pick_type(self, g, ispass, deep):
        key = "00" if g.startswith("pre") else f"{int(ispass)}{int(deep)}"
        t = self.tp[g].get(key) or self.tp[g].get("00") or next(iter(self.tp[g].values()))
        return t["types"][int(self.rng.choice(len(t["types"]), p=np.array(t["p"]) / np.sum(t["p"])))]

    def nominal(self, t):
        info = self.types[t]
        p = np.array(info["pmf_p"])
        return float(self.rng.choice(info["pmf_yards"], p=p / p.sum()))

    def apply(self, d, st):
        code = int(d["play_type_code"])
        if code not in (0, 1):
            return d
        down, dist, yl, sd, qtr, off_home, clock_val, off_to, def_to = st
        idx = int(d["idx"])
        ispass = code == 1
        air = self.air[idx]
        deep = int(ispass and np.isfinite(air) and air >= self.deepmin)
        home_off = 1 if off_home else 0
        x = fx([down], [dist], [yl], [sd], [qtr], [home_off], [self.nh[idx]], [self.sg[idx]], [int(ispass)], [deep], self.sdsd)
        if self.inc is not None:
            x = aug(x, [self.inc[idx]])
        ot, dt = (self.home, self.away) if off_home else (self.away, self.home)
        ps = self.haz(x, ot, dt)
        u = self.rng.random()
        gk = -1
        c = 0.0
        for k, p in enumerate(ps):
            c += p
            if u < c:
                gk = k
                break
        y = float(d["yards_gained"])
        turn = bool(d["flip"]) or d["points_def"] > 0
        row = [self.gi, down, dist, yl, qtr, code, gk, -1, 0, 0, 0.0, 0.0, 0, y]
        if gk < 0:
            self.ev.append(row)
            return d
        g = GROUPS[gk]
        t = self.pick_type(g, ispass, deep)
        info = self.types[t]
        tid = sorted(self.types).index(t)
        nom = self.nominal(t)
        if t == "Defensive Pass Interference" and ispass and np.isfinite(air):
            nom = max(air, 0.0)
        af = info["auto_first_share"] if np.isfinite(info["auto_first_share"]) else 0.0
        is_off = g.endswith("_off")
        pre = g.startswith("pre")
        counted = (not pre) and self.rng.random() < info["counted"]
        row[7], row[10] = tid, nom
        if is_off:
            amt = min(nom, max((100.0 - yl) / 2.0, 0.0))
        else:
            cap = yl - 1.0 if t == "Defensive Pass Interference" else yl / 2.0
            amt = min(nom, max(cap, 0.0))
        po = float(d["points_off"])
        pdf = float(d["points_def"])
        hs = max(clock_val - HALF_SECONDS, 0.0) if qtr <= 2 else max(clock_val, 0.0)
        cx = dict(down=np.array([float(down)]), dist=np.array([float(dist)]), yl=np.array([float(yl)]), hs=np.array([hs]), half=np.array([3.0 if qtr >= 5 else (1.0 if qtr <= 2 else 2.0)]), sd=np.array([float(sd)]), to_o=np.array([float(off_to)]), to_d=np.array([float(def_to)]))
        gv, stf = gain_dec(self.ep, cx, np.array([y]), np.array([po]), np.array([pdf]), np.array([turn]), np.array([nom]), np.array([af]), np.array([1.0 if counted else 0.0]), np.array([is_off]), np.array([t == "Defensive Pass Interference"]))
        xa = acc_X(gv, np.array([gk]), stf, cx["sd"], cx["hs"], np.array([is_off]), self.sdsd)
        acc = bool(self.rng.random() < float(predict(self.acw, xa)[0]))
        if not acc:
            row[8] = 0
            self.ev.append(row)
            return d
        out = dict(d)
        if counted:
            if is_off:
                amt = min(nom, max((100.0 - (yl - y)) / 2.0, 0.0))
                out["yards_gained"] = y - amt
                out["auto_first"] = False
            else:
                amt = min(nom, max((yl - y) / 2.0, 0.0))
                out["yards_gained"] = y + amt
                out["auto_first"] = bool(self.rng.random() < af) and (y + amt < dist)
            out["repeat_down"] = False
            row[8:13] = [1, 1, amt if not is_off else -amt, 0.0, int(out["auto_first"])]
            row[11] = 0.0
            self.ev.append(row)
            return out
        el = self.el[g]
        out.update(points_off=0.0, points_def=0.0, flip=False, play_type_code=6, off_to_used=0.0, def_to_used=0.0)
        out["clock_elapsed"] = float(el[int(self.rng.integers(len(el)))]) if len(el) else 0.0
        if is_off:
            out["yards_gained"] = -amt
            out["auto_first"] = False
            out["repeat_down"] = True
            fd = 0
        else:
            fd = int(amt >= dist or self.rng.random() < af)
            out["yards_gained"] = amt
            out["auto_first"] = bool(fd)
            out["repeat_down"] = not fd
        out["dist_gained"] = out["yards_gained"]
        row[8:13] = [1, 0, out["yards_gained"], out["clock_elapsed"], fd]
        self.ev.append(row)
        return out


def f3_install():
    import mod25d_variance as dv

    ns = dv._G["ns"]
    t = dv._G["tables"]
    a = t["arrays"]
    P = load_params()
    pool = pool_arrays()
    assert len(pool) == len(a["possession_flip"])
    code = np.asarray(a["play_type_code"])
    pen = pool.pen.to_numpy() & np.isin(code, (0, 1, 6))
    assert "IPW" in ns
    ipw = np.asarray(ns["IPW"], float) * (~pen)
    if os.environ.get("F3W") == "1":
        ipw = ipw / (1.0 - removal_prob(P, pool, POOL_TR[0], a))
    if w2():
        ipw = ipw / (1.0 - removal_cached(P, pool, POOL_TR[0], a))
    ns["IPW"] = ipw
    t["nn_weight_cache_cond"].clear()
    ov = Overlay(P, pool, a, 0)
    SIMS["ov"] = ov
    inner = dv._G["pol"]

    def pol(down, distance, yardline, score_diff, qtr, clock_val, drawn):
        fr = sys._getframe(1)
        off_home = fr.f_locals["offense"] == "home"
        drawn = ov.apply(drawn, (down, distance, yardline, score_diff, qtr, off_home, clock_val, fr.f_locals["off_to"], fr.f_locals["def_to"]))
        return inner(down, distance, yardline, score_diff, qtr, clock_val, drawn)

    dv._G["pol"] = pol
    orig = ns["run_one_game"]

    def game(*args, **kw):
        ov.new_game()
        return orig(*args, **kw)

    ns["run_one_game"] = game
    origp = dv.d_play_season

    def dps(task):
        ov.start_task(task)
        res = origp(task)
        ev = pd.DataFrame(ov.ev, columns=["gi", "down", "dist", "yl", "qtr", "code", "grp", "tid", "acc", "counted", "yards", "el", "fd", "play_y"])
        ev.to_parquet(Path(__import__("os").environ["BUD_OUT"]) / f"events_{task[0]}_{task[1]}.parquet")
        return res

    dv.d_play_season = dps


def f3_init_budget(setting):
    import sim09_hk as hk

    hk.hk_init_budget(setting)
    f3_install()


def f3_init_ss(setting):
    import sim09_hk as hk

    hk.hk_init_ss(setting)
    f3_install()


def register():
    import mod25d_variance as dv
    import sim09_hk  # noqa: F401

    dv.DV["crzf3"] = dict(dv.DV["crzhk"], f3=1)


def f3_play_season(task):
    import mod25d_variance as dv

    return dv.d_play_season(task)


def cmd_sim(args):
    import os

    import mod25_generator as gen
    import mod25e_budget as bud
    register()
    import mod25d_variance as dv

    bud.OUT = Path(args.out_dir).resolve()
    os.environ["BUD_OUT"] = str(bud.OUT)
    bud.OUT.mkdir(parents=True, exist_ok=True)
    dv.c25.ensure_policies()
    v = dv.DV[args.variant]
    cfgj = json.dumps(dict(v, seed=args.seed, name=f"{args.variant}_s{v['scale']:g}"))
    setting = dict(gen.SETTING_DEFAULTS)
    setting.update({"scale": v["scale"], "yard_gain": v["yard_gain"], "def_sign": 1.0, "yard_bias": 0.0, "drift": 1.0, "mech": cfgj})
    gen.init_worker = f3_init_budget
    gen.play_season = f3_play_season
    games, plays, latents_, el = gen.run_generation(setting, args.worlds, args.seasons, args.workers, args.seed, progress=True)
    games.to_parquet(bud.OUT / "sim_games.parquet")
    print("done", len(games), el, flush=True)


def cmd_e5(args):
    import mod25e_scorestate as ss

    register()
    ss.s_init = f3_init_ss
    args.scale = 1.0
    ss.cmd_e5(args)


def cmd_audit(args):
    P = load_params()
    pool = pool_arrays()
    auto = {t for t, r in P["types"].items() if np.isfinite(r["auto_first_share"]) and r["auto_first_share"] >= 0.5}
    d = Path(args.frames)
    fs = sorted(d.glob("play_*_*.parquet"))
    out = []
    for i, f in enumerate(fs):
        z = pd.read_parquet(f)
        z["f"] = i
        out.append(z)
    S = pd.concat(out, ignore_index=True)
    S["gid"] = S.f * 100000 + S.g
    S["idx"] = S.idx.astype(np.int64)
    ng = S.gid.nunique()
    S = S.sort_values(["gid"], kind="stable").reset_index(drop=True)
    S["pen"] = pool.pen.to_numpy()[S.idx.to_numpy()]
    S["grp"] = pool.grp.to_numpy()[S.idx.to_numpy()]
    S["ptype"] = pool.penalty_type.to_numpy()[S.idx.to_numpy()]
    nxt = S.groupby("gid")[["yl", "down", "offhome"]].shift(-1)
    same = (nxt.offhome == S.offhome) & (S.flip == 0) & (S.po == 0) & (S.pdf == 0)
    S["moved"] = np.where(same, S.yl - nxt.yl, np.nan)
    S["ndown"] = np.where(same, nxt.down, np.nan)
    sc = S[S.code.isin([0, 1, 6])]
    print(f"games {ng} scrimmage snaps {len(sc)} per game {len(sc) / ng:.1f}")
    print(f"no-play share sim {float((sc.code == 6).mean()):.4f}; inherited penalty rows (accepted foul in drawn row) share {float(sc.pen.mean()):.4f}; per game {float(sc.pen.sum()) / ng:.2f}")
    rep = sc[sc.pen & sc.moved.notna() & (sc.code == 6)]
    off = rep[rep.grp.str.endswith("_off")]
    de = rep[rep.grp.str.endswith("_def")]
    half_o = (100 - off.yl) / 2
    v1 = off[(-off.moved) > half_o + 0.5]
    print(f"A offensive replay fouls n {len(off)}: loss beyond half distance to own goal {len(v1)} ({len(v1) / max(len(off), 1):.4f}), per game {len(v1) / ng:.3f}; mean excess yards {float(((-v1.moved) - (100 - v1.yl) / 2).mean()) if len(v1) else 0:.2f}")
    dpi = de[de.ptype == "Defensive Pass Interference"]
    ndp = de[de.ptype != "Defensive Pass Interference"]
    v2 = ndp[ndp.moved > ndp.yl / 2 + 0.5]
    v3 = dpi[dpi.moved > dpi.yl - 1 + 0.5]
    print(f"B defensive replay non-spot fouls n {len(ndp)}: gain beyond half distance to goal {len(v2)} ({len(v2) / max(len(ndp), 1):.4f}) per game {len(v2) / ng:.3f}; DPI n {len(dpi)}: spot beyond goal line {len(v3)} per game {len(v3) / ng:.3f}")
    nd_short = de[(de.moved < de.dist)]
    spur = nd_short[(nd_short.ndown == 1) & ~nd_short.ptype.isin(auto)]
    miss = nd_short[(nd_short.ndown != 1) & nd_short.ptype.isin(auto)]
    print(f"C defensive fouls with gain short of sim distance n {len(nd_short)}: first down credited for non-automatic foul {len(spur)} per game {len(spur) / ng:.3f}; automatic foul without first down {len(miss)} per game {len(miss) / ng:.3f}")
    fd_long = de[(de.moved >= de.dist) & (de.ndown != 1)]
    print(f"D defensive gain >= sim distance without first down {len(fd_long)} per game {len(fd_long) / ng:.3f}")
    byd = sc.groupby("down").apply(lambda q: float((q.code == 6).mean()))
    print("E no-play share by down sim " + str(byd.round(4).to_dict()))
    s = snap_table(load_nv())
    r = s[s.down.isin([1, 2, 3, 4])]
    print("E no-play share by down real " + str(r.groupby("down").apply(lambda q: float((q.play_type == "no_play").mean())).round(4).to_dict()))
    print("E accepted-foul rate by group real " + str(s.groupby("group").apply(lambda q: float(q.acc.sum())).div(len(s)).round(4).to_dict()))
    print("E inherited rows by group sim " + str((sc[sc.pen].groupby("grp").size() / len(sc)).round(4).to_dict()))


def cmd_validate(args):
    P = load_params()
    d = Path(args.events)
    ev = pd.concat([pd.read_parquet(f).assign(task=i) for i, f in enumerate(sorted(d.glob("events_*.parquet")))], ignore_index=True)
    ng = ev.groupby("task").gi.nunique().sum()
    tn = sorted(P["types"])
    print(f"sim snaps {len(ev)} games {ng} per game {len(ev) / ng:.1f}")
    s = snap_table(load_nv())
    s = s[s.play_type.isin(["run", "pass", "no_play"])]
    real_n = len(s)
    print("accepted per snap by group (sim | real)")
    for k, g in enumerate(GROUPS):
        sim_c = (ev.grp == k).mean()
        sim_a = ((ev.grp == k) & (ev.acc == 1)).mean()
        rc = (s.group == g).mean()
        ra = ((s.group == g) & s.acc).mean()
        print(f"  {g}: committed {sim_c:.4f} | {rc:.4f}; accepted {sim_a:.4f} | {ra:.4f}; accept share {sim_a / max(sim_c, 1e-9):.3f} | {ra / rc:.3f}")
    tn_by = {i: t for i, t in enumerate(tn)}
    ev["ft"] = ev.tid.map(tn_by)
    rt = s[s.ftype.notna() & ~s["desc"].fillna("").str.contains("(?i)offsetting")]
    top = rt.ftype.value_counts().index[:10]
    print("accept share by foul type (sim | real)")
    for t in top:
        sc_ = ev[ev.ft == t]
        rc_ = rt[rt.ftype == t]
        print(f"  {t}: n sim {len(sc_)} real {len(rc_)}; accept share {(sc_.acc == 1).mean():.3f} | {rc_.acc.mean():.3f}")
    ev["noplay"] = ((ev.acc == 1) & (ev.counted == 0)).astype(int)
    rn = (s.play_type == "no_play").mean()
    print(f"no-play share sim {ev.noplay.mean():.4f} real {rn:.4f}")
    sdn = ev.groupby("down").apply(lambda q: pd.Series({"acc": (q.acc == 1).mean(), "n": len(q)}))
    rdn = s.groupby("down").apply(lambda q: pd.Series({"acc": q.acc.mean(), "n": len(q)}))
    print("accepted rate by down sim " + str(sdn.acc.round(4).to_dict()) + " real " + str(rdn.acc.round(4).to_dict()))
    qz = lambda df, y: pd.cut(df[y], [-1, 20, 80, 101], labels=["opp<=20", "mid", "own<=20"]).value_counts().sort_index()
    ev["zone"] = pd.cut(ev.yl, [-1, 20, 80, 101], labels=["opp<=20", "mid", "own<=20"])
    s["zone"] = pd.cut(s.yardline_100, [-1, 20, 80, 101], labels=["opp<=20", "mid", "own<=20"])
    print("accepted rate by field zone sim " + str(ev.groupby("zone", observed=True).acc.apply(lambda x: (x == 1).mean()).round(4).to_dict()) + " real " + str(s.groupby("zone", observed=True).acc.mean().round(4).to_dict()))
    a = ev[ev.acc == 1]
    off = a.grp.isin([0, 2])
    ro = a[off & (a.counted == 0)]
    v1 = ro[(-ro.yards) > (100 - ro.yl) / 2 + 1e-9]
    de = a[~off & (a.counted == 0)]
    dpi = tn.index("Defensive Pass Interference")
    v2 = de[(de.tid != dpi) & (de.yards > de.yl / 2 + 1e-9)]
    v3 = de[(de.tid == dpi) & (de.yards > de.yl - 1 + 1e-9)]
    short = de[de.yards < de.dist]
    fdok = short[short.fd == 1]
    af = {i: P["types"][t]["auto_first_share"] for i, t in enumerate(tn)}
    spur = fdok[fdok.tid.map(af).fillna(0) < 0.5]
    print(f"rule consistency per game (sim): half-distance violations offence {len(v1) / ng:.4f}, defence non-spot {len(v2) / ng:.4f}, DPI beyond goal {len(v3) / ng:.4f}; defensive replay with yards>=distance but no first down {int(((de.yards >= de.dist) & (de.fd == 0)).sum()) / ng:.4f}; first downs credited short of distance on non-automatic types (Bernoulli of type share, rule-fitted) {len(spur) / ng:.4f}")
    print(f"events per game sim {len(a) / ng:.2f} accepted; real accepted per game {len(s[s.acc]) / s.game_id.nunique():.2f}")
    print("accepted yards mean (replay) sim off " + str(round(float(ro.yards.mean()), 2)) + " real " + str(round(float(s[s.acc & s.aoff & (s.play_type == 'no_play')].penalty_yards.mean() * -1), 2)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("measure")
    sub.add_parser("haz2")
    ac = sub.add_parser("accept")
    ac.add_argument("--draws", type=int, default=12)
    a = sub.add_parser("audit")
    a.add_argument("--frames", default=str(REPO / "artifacts" / "sim09" / "u3e" / "play_crzhk"))
    s = sub.add_parser("sim")
    s.add_argument("--variant", default="crzf3")
    s.add_argument("--out-dir", dest="out_dir", default=str(OUT / "play_crzf3"))
    s.add_argument("--worlds", type=int, default=2)
    s.add_argument("--seasons", type=int, default=2)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--seed", type=int, default=21)
    e = sub.add_parser("e5")
    e.add_argument("--variant", default="crzf3")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    v = sub.add_parser("validate")
    v.add_argument("--events", default=str(OUT / "play_crzf3"))
    args = ap.parse_args()
    {"measure": cmd_measure, "haz2": cmd_haz2, "accept": cmd_accept, "audit": cmd_audit, "sim": cmd_sim, "e5": cmd_e5, "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    main()
