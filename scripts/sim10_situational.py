import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import sim10_drive as sd  # noqa: E402
from nv_path import NV  # noqa: E402

ALIAS = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
LAST_REG_WEEK = 17
HALF_SECS = 1800.0
ZC_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
ZC_DEFAULT = 1.0
STAKES_FILE = REPO / "artifacts" / "sim09" / "stakes_teamweek.csv"
OOF_FILE = sd.OUT / "oof_targets.parquet"
PARAMS_FILE = sd.OUT / "situational_params.json"
POOL_FILE = sd.OUT / "situational_pool.npz"
LOSO_FILE = sd.OUT / "situational_loso.csv"
RIDGE_GRID = (0.0, 1.0, 10.0, 100.0, 1000.0)
EPS = 1e-6
SECS_PER_MIN = 60.0
URGENCY_TERMS = ("zt", "zl", "zt_h2", "zl_h2", "zt_late")
STAKES_TERMS = (
    "lev_own",
    "lev_opp",
    "elim_own",
    "elim_opp",
    "clinch_own",
    "clinch_opp",
    "lev_own_zt",
    "lev_own_zl",
    "elim_own_zt",
    "elim_own_zl",
    "clinch_own_zt",
    "clinch_own_zl",
)
TARGETS = ("td", "fg", "el")
MECHS = ("urgency", "fatigue", "stakes")


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def zscore(sdv, gsecs, zc):
    return sdv / np.sqrt(gsecs / SECS_PER_MIN + zc)


def urgency_x(sdv, gsecs, half, secs, zc):
    z = zscore(sdv, gsecs, zc)
    zt, zl = np.minimum(z, 0.0), np.maximum(z, 0.0)
    h2 = (np.asarray(half) == 2).astype(float)
    late = (np.asarray(secs) <= sd.TWO_MIN).astype(float)
    return np.column_stack([zt, zl, zt * h2, zl * h2, zt * late])


def fatigue_x(length):
    return np.asarray(length, dtype=float).reshape(-1, 1) / 10.0


def stakes_x(b6, sdv, gsecs, week, zc):
    b6 = np.atleast_2d(np.asarray(b6, dtype=float))
    z = zscore(sdv, gsecs, zc)
    zt, zl = np.minimum(z, 0.0), np.maximum(z, 0.0)
    ramp = np.clip((np.asarray(week, dtype=float) - 1.0) / (LAST_REG_WEEK - 1.0), 0.0, 1.0)
    cols = [
        b6[:, 0],
        b6[:, 1],
        b6[:, 2],
        b6[:, 3],
        b6[:, 4],
        b6[:, 5],
        b6[:, 0] * zt,
        b6[:, 0] * zl,
        b6[:, 2] * zt,
        b6[:, 2] * zl,
        b6[:, 4] * zt,
        b6[:, 4] * zl,
    ]
    base = np.column_stack(cols)
    return np.column_stack([base, base * ramp[:, None]])


def nh_x(has_prev, prev_share, sdv, gsecs, half, secs, zc):
    z = zscore(sdv, gsecs, zc)
    late = (np.asarray(secs) <= sd.TWO_MIN).astype(float)
    hp = np.asarray(has_prev, dtype=float)
    return np.column_stack(
        [
            np.ones(len(hp)),
            hp,
            hp * np.nan_to_num(np.asarray(prev_share, dtype=float)),
            np.minimum(z, 0.0),
            np.maximum(z, 0.0),
            late,
            late * np.minimum(z, 0.0),
            (np.asarray(half) == 2).astype(float),
        ]
    )


def pool():
    seg = pd.read_parquet(sd.OUT / "segments.parquet")
    p = sd.load_pbp()
    p = p[p["game_id"].isin(seg["game_id"].unique())].reset_index(drop=True)
    row = seg["row"].to_numpy()
    assert (p["game_id"].to_numpy()[row] == seg["game_id"].to_numpy()).all()
    nv = pd.concat(
        [
            pd.read_parquet(NV / f"pbp_{s}.parquet", columns=["game_id", "play_id", "no_huddle"])
            for s in sd.SEASONS
        ],
        ignore_index=True,
    ).drop_duplicates(["game_id", "play_id"])
    n0 = len(p)
    p = p.merge(nv, on=["game_id", "play_id"], how="left")
    assert len(p) == n0
    rp = p[p["play_type"].isin(["run", "pass"]) & p["fixed_drive"].notna()]
    drv = rp.groupby(["game_id", "fixed_drive"]).agg(
        plays=("play_id", "size"), nh=("no_huddle", "mean")
    )
    key = pd.MultiIndex.from_arrays([seg["game_id"], p["fixed_drive"].to_numpy()[row]])
    got = drv.reindex(key)
    d = pd.DataFrame(
        {
            "game_id": seg["game_id"].to_numpy(),
            "pos": p["posteam"].to_numpy()[row],
            "dfn": p["defteam"].to_numpy()[row],
            "plays": got["plays"].to_numpy(float),
            "nh": got["nh"].to_numpy(float),
        }
    )
    prior = d.groupby(["game_id", "dfn"])["plays"].shift(1)
    prev_nh = d.groupby(["game_id", "pos"])["nh"].shift(1)
    seg = seg.assign(
        plays=d["plays"].to_numpy(),
        nh=d["nh"].to_numpy(),
        length=prior.fillna(0.0).to_numpy(),
        has_prev=prev_nh.notna().to_numpy().astype(float),
        prev_nh=prev_nh.fillna(0.0).to_numpy(),
        gsecs=np.where(seg["half"] == 1, seg["secs"] + HALF_SECS, seg["secs"]),
    )
    tw = pd.read_csv(STAKES_FILE, usecols=["season", "week", "team", "lev", "elim", "clinch"])
    tw = tw.set_index(["season", "week", "team"])
    ss = seg["season"].to_numpy(int)
    wk = seg["week"].to_numpy(int)
    b6 = np.full((len(seg), 6), np.nan)
    for tm, cols in ((d["pos"], (0, 2, 4)), (d["dfn"], (1, 3, 5))):
        tm = tm.replace(ALIAS)
        g = tw.reindex(pd.MultiIndex.from_arrays([ss, wk, tm.to_numpy()]))
        for name, col in zip(("lev", "elim", "clinch"), cols, strict=True):
            b6[:, col] = g[name].to_numpy(float)
    seg["stakes_ok"] = np.isfinite(b6).all(axis=1).astype(float)
    b6 = np.nan_to_num(b6)
    return seg, b6


def residual_targets(seg):
    oof = pd.read_parquet(OOF_FILE)
    ix = oof.index.to_numpy()
    n = len(seg)
    ok = np.zeros(n, dtype=bool)
    ok[ix] = True
    ok &= (seg["bad"].to_numpy() == 0) & (seg["half"].to_numpy() < 3)
    ok &= np.isfinite(seg["plays"].to_numpy())
    po, _ = sd.segment_points(seg)
    y = {
        "td": (po >= 6).astype(float),
        "fg": (po == 3).astype(float),
        "el": seg["el"].to_numpy(float),
    }
    o = {k: np.full(n, np.nan) for k in TARGETS}
    for k in TARGETS:
        o[k][ix] = oof[k].to_numpy(float)
    return y, o, ok


def fit_logit(X, y, off, lam):
    k = X.shape[1]
    pen = np.r_[0.0, np.full(k - 1, lam)]

    def f(th):
        eta = off + X @ th
        r = sig(eta) - y
        nll = float(np.sum(np.logaddexp(0.0, eta) - y * eta)) + 0.5 * float(pen @ (th * th))
        return nll, X.T @ r + pen * th

    return minimize(f, np.zeros(k), jac=True, method="L-BFGS-B", options={"maxiter": 300}).x


def fit_ols(X, y, lam):
    k = X.shape[1]
    pen = np.diag(np.r_[0.0, np.full(k - 1, lam)])
    return np.linalg.solve(X.T @ X + pen + 1e-9 * np.eye(k), X.T @ y)


def design(Xm, mu, sdv):
    return np.column_stack([np.ones(len(Xm)), (Xm - mu) / sdv])


def fit_target(Xm, y, o, mask, target, lam):
    mu = Xm[mask].mean(axis=0)
    sdv = Xm[mask].std(axis=0)
    sdv[sdv == 0] = 1.0
    D = design(Xm, mu, sdv)
    if target == "el":
        th = fit_ols(D[mask], y[mask] - o[mask], lam)
    else:
        th = fit_logit(D[mask], y[mask], logit(o[mask]), lam)
    return {"mu": mu, "sd": sdv, "th": th}


def heldout_gain(Xm, y, o, mask, target, m):
    D = design(Xm, m["mu"], m["sd"])
    if target == "el":
        r0 = y[mask] - o[mask]
        r1 = r0 - D[mask][:, 1:] @ m["th"][1:]
        return float(np.mean(r1**2) - np.mean(r0**2))
    off = logit(o[mask])
    yy = y[mask]
    e1 = off + D[mask][:, 1:] @ m["th"][1:]
    ll1 = np.mean(np.logaddexp(0.0, e1) - yy * e1)
    ll0 = np.mean(np.logaddexp(0.0, off) - yy * off)
    return float(ll1 - ll0)


def pick_lambda(Xm, y, o, mask, season, target, seasons):
    best, arg = None, 0.0
    for lam in RIDGE_GRID:
        tot, n = 0.0, 0
        for s in seasons:
            tr, te = mask & (season != s), mask & (season == s)
            m = fit_target(Xm, y, o, tr, target, lam)
            tot += heldout_gain(Xm, y, o, te, target, m) * te.sum()
            n += te.sum()
        if best is None or tot / n < best:
            best, arg = tot / n, lam
    return arg


def raw_coef(m):
    b = m["th"][1:] / m["sd"]
    return b


def stakes_mask(seg):
    return seg["stakes_ok"].to_numpy() == 1


def pick_zc(seg, y, o, ok, season, seasons):
    sdv, gs = seg["sd"].to_numpy(float), seg["gsecs"].to_numpy(float)
    half, secs = seg["half"].to_numpy(), seg["secs"].to_numpy(float)
    in_set = np.isin(season, seasons)
    best, arg = None, ZC_DEFAULT
    for zc in ZC_GRID:
        Xm = urgency_x(sdv, gs, half, secs, zc)
        tot, n = 0.0, 0
        for s in seasons:
            tr, te = ok & in_set & (season != s), ok & (season == s)
            m = fit_target(Xm, y["td"], o["td"], tr, "td", 0.0)
            tot += heldout_gain(Xm, y["td"], o["td"], te, "td", m) * te.sum()
            n += te.sum()
        if best is None or tot / n < best:
            best, arg = tot / n, zc
    return arg


def fit_nh(X, y, mask, lam=0.0):
    return fit_ols(X[mask], y[mask], lam)


def nh_matrix(seg, zc):
    sdv, gs = seg["sd"].to_numpy(float), seg["gsecs"].to_numpy(float)
    return nh_x(
        seg["has_prev"],
        seg["prev_nh"],
        sdv,
        gs,
        seg["half"].to_numpy(),
        seg["secs"].to_numpy(float),
        zc,
    )


def mech_matrices(seg, b6, zc):
    sdv, gs = seg["sd"].to_numpy(float), seg["gsecs"].to_numpy(float)
    half, secs, wk = seg["half"].to_numpy(), seg["secs"].to_numpy(float), seg["week"].to_numpy()
    return {
        "urgency": urgency_x(sdv, gs, half, secs, zc),
        "fatigue": fatigue_x(seg["length"].to_numpy()),
        "stakes": stakes_x(b6, sdv, gs, wk, zc),
    }


def cmd_loso(a):
    seg, b6 = pool()
    y, o, ok = residual_targets(seg)
    season = seg["season"].to_numpy(int)
    seasons = sorted(set(season[ok]))
    print(
        "segments",
        len(seg),
        "usable",
        int(ok.sum()),
        "stakes tag match",
        round(float(seg.loc[ok, "stakes_ok"].mean()), 4),
        "nh share defined",
        round(float(np.isfinite(seg.loc[ok, "nh"]).mean()), 4),
        flush=True,
    )
    yn = seg["nh"].to_numpy(float)
    rows = []
    for s in seasons:
        trs = [t for t in seasons if t != s]
        zc = pick_zc(seg, y, o, ok, season, trs)
        print("season", s, "zc", zc, flush=True)
        X = mech_matrices(seg, b6, zc)
        for name in MECHS:
            base = ok & stakes_mask(seg) if name == "stakes" else ok
            tr, te = base & (season != s), base & (season == s)
            for target in TARGETS:
                lam = (
                    pick_lambda(X[name], y[target], o[target], tr, season, target, trs)
                    if name == "stakes"
                    else 0.0
                )
                m = fit_target(X[name], y[target], o[target], tr, target, lam)
                rows.append(
                    {
                        "mech": name,
                        "target": target,
                        "season": s,
                        "lam": lam,
                        "zc": zc,
                        "n_test": int(te.sum()),
                        "gain": heldout_gain(X[name], y[target], o[target], te, target, m),
                        "coef": [round(float(c), 5) for c in raw_coef(m)],
                    }
                )
        Xn = nh_matrix(seg, zc)
        msk = ok & np.isfinite(yn)
        tr, te = msk & (season != s), msk & (season == s)
        b = fit_nh(Xn, yn, tr)
        res = yn[te] - Xn[te] @ b
        rows.append(
            {
                "mech": "nohuddle",
                "target": "nh_share",
                "season": s,
                "lam": 0.0,
                "zc": zc,
                "n_test": int(te.sum()),
                "gain": float(np.mean(res**2) - np.mean((yn[te] - yn[tr].mean()) ** 2)),
                "coef": [round(float(c), 4) for c in b],
                "sigma": round(float(np.sqrt(np.mean((yn[tr] - Xn[tr] @ b) ** 2))), 4),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(LOSO_FILE, index=False)
    for (mech, target), g in df.groupby(["mech", "target"], sort=False):
        w = g["n_test"].to_numpy(float)
        print(
            mech,
            target,
            "pooled held-out gain (negative = better)",
            round(float(np.average(g["gain"], weights=w)), 7),
            "seasons better",
            int((g["gain"] < 0).sum()),
            "of",
            len(g),
            flush=True,
        )
        for r in g.itertuples():
            print("  ", r.season, "lam", r.lam, "zc", r.zc, "gain", round(r.gain, 7), flush=True)


def cmd_fit(a):
    seg, b6 = pool()
    y, o, ok = residual_targets(seg)
    season = seg["season"].to_numpy(int)
    seasons = sorted(set(season[ok]))
    zc = pick_zc(seg, y, o, ok, season, seasons)
    print("zc", zc, flush=True)
    X = mech_matrices(seg, b6, zc)
    out = {"zc": zc, "last_reg_week": LAST_REG_WEEK, "mechs": {}}
    for name in MECHS:
        mask = ok & stakes_mask(seg) if name == "stakes" else ok.copy()
        out["mechs"][name] = {}
        for target in TARGETS:
            lam = (
                pick_lambda(X[name], y[target], o[target], mask, season, target, seasons)
                if name == "stakes"
                else 0.0
            )
            m = fit_target(X[name], y[target], o[target], mask, target, lam)
            out["mechs"][name][target] = {
                "lam": lam,
                "mu": m["mu"].tolist(),
                "coef": raw_coef(m).tolist(),
            }
            print(name, target, "lam", lam, "coef", np.round(raw_coef(m), 5).tolist(), flush=True)
    Xn = nh_matrix(seg, zc)
    yn = seg["nh"].to_numpy(float)
    mn = ok & np.isfinite(yn)
    b = fit_nh(Xn, yn, mn)
    sig_ = float(np.sqrt(np.mean((yn[mn] - Xn[mn] @ b) ** 2)))
    out["nohuddle"] = {"coef": b.tolist(), "sigma": sig_}
    print("nohuddle coef", np.round(b, 4).tolist(), "sigma", round(sig_, 4), flush=True)
    PARAMS_FILE.write_text(json.dumps(out, indent=1))
    m_row = np.where(np.isfinite(yn), Xn @ b, np.nan)
    np.savez(
        POOL_FILE,
        length=seg["length"].to_numpy(float),
        plays=np.nan_to_num(seg["plays"].to_numpy(float)),
        nh=yn,
        m_row=m_row,
    )
    print("wrote", PARAMS_FILE, POOL_FILE, flush=True)


class Situational:
    def __init__(self, flags, params_file=PARAMS_FILE, pool_file=POOL_FILE):
        self.flags = set(flags)
        self.par = json.loads(Path(params_file).read_text())
        self.zc = float(self.par["zc"])
        z = np.load(pool_file)
        self.cnt = z["plays"]
        self.nh = z["nh"]
        self.m_row = z["m_row"]
        self.tags = None
        self.reset()

    def reset(self):
        self.last = [0.0, 0.0]
        self.prev_nh = [None, None]

    def set_stakes(self, tags_by_pos):
        self.tags = tags_by_pos

    def _shift(self, name, Xrow):
        out = {}
        for t in TARGETS:
            c = self.par["mechs"][name][t]
            out[t] = float((Xrow - np.asarray(c["mu"])) @ np.asarray(c["coef"]))
        return out

    def tg_shift(self, tg, hf, secs, sdv, pos, week):
        if hf >= 3:
            return tg
        gs = secs + (HALF_SECS if hf == 1 else 0.0)
        sh = dict.fromkeys(TARGETS, 0.0)
        rows = {}
        if "urgency" in self.flags:
            rows["urgency"] = urgency_x(
                np.array([sdv]), np.array([gs]), np.array([hf]), np.array([secs]), self.zc
            )[0]
        if "fatigue" in self.flags:
            rows["fatigue"] = fatigue_x([self.last[1 - pos]])[0]
        if "stakes" in self.flags and self.tags is not None:
            o, d = self.tags[pos], self.tags[1 - pos]
            b6 = np.array([o[0], d[0], o[1], d[1], o[2], d[2]])
            rows["stakes"] = stakes_x(
                b6, np.array([sdv]), np.array([gs]), np.array([week]), self.zc
            )[0]
        for name, xr in rows.items():
            for t, v in self._shift(name, xr).items():
                sh[t] += v
        tg = np.array(tg, dtype=np.float64)
        for k, t in ((1, "td"), (2, "fg")):
            tg[k] = sig(logit(tg[k]) + sh[t])
        return tg

    def nh_ratio(self, idx, hf, secs, sdv, pos):
        if "nohuddle" not in self.flags or hf >= 3:
            return np.ones(len(idx))
        gs = secs + (HALF_SECS if hf == 1 else 0.0)
        pv = self.prev_nh[pos]
        x = nh_x(
            [0.0 if pv is None else 1.0],
            [0.0 if pv is None else pv],
            np.array([sdv]),
            np.array([gs]),
            np.array([hf]),
            np.array([secs]),
            self.zc,
        )[0]
        b = np.asarray(self.par["nohuddle"]["coef"])
        s2 = self.par["nohuddle"]["sigma"] ** 2
        m_sim = float(x @ b)
        y = self.nh[idx]
        mr = self.m_row[idx]
        r = np.exp(-0.5 * ((y - m_sim) ** 2 - (y - mr) ** 2) / s2)
        return np.where(np.isfinite(r), r, 1.0)

    def after_drive(self, j, pos):
        self.last[1 - pos] = float(self.cnt[j])
        v = self.nh[j]
        self.prev_nh[pos] = None if not np.isfinite(v) else float(v)


class StakesBook:
    def __init__(self, sched, seed):
        import sim09_stakes_gen as sg

        self.t = sg.Tracker(sched, seed)

    def tags(self, week, h, a):
        s = self.t.stakes(week)
        return [s[h][:3], s[a][:3]]

    def record(self, week, h, a, margin):
        self.t.results.append((week, h, a, float(margin)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("loso")
    sub.add_parser("fit")
    a = ap.parse_args()
    {"loso": cmd_loso, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
