import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead106_injury_language as L6
import lead110_absorption as A
import mod24_ftn_turnover_luck as m
import mod25_wf_forward as mwf
import mod25_wfj_forward as wj

OUT = L6.ROOT / "artifacts" / "lead108_wording_mixture"
SEASON = 2026
FIRST_WEEK = 2
LAST_WEEK = 9
FORWARD_WEEK = 9
SEED = 108
REPS = 2000
KEYS = ["designation", "practice_status"]
SUBSETS = {"wk2_8": (2, 8), "wk9": (9, 9), "wk2_9": (2, 9)}


def players():
    pop = L6.id_outcomes(pd.read_parquet(L6.ART / "population.parquet"))
    ver = pop[pop["verified"]].copy()
    ver["practice_status"] = ver["practice_status"].fillna("none")
    L6.LAST_WEEK = FORWARD_WEEK
    tab = L6.ladder_tables(ver, KEYS)
    early = ver[ver["week"].between(FIRST_WEEK, FORWARD_WEEK - 1)].copy()
    early["lp"] = [L6.ladder_pred(r, tab, KEYS)[0] for _, r in early.iterrows()]
    s = A.sample_rows()
    reads = dict(zip(zip(s["gsis_id"], s["week"], strict=True), s["rp"], strict=True))
    early["rp"] = [reads.get(k, np.nan) for k in zip(early["gsis_id"], early["week"], strict=True)]
    fw = pd.read_parquet(
        L6.ART / "forward_score" / f"{SEASON}_week{FORWARD_WEEK:02d}_scored.parquet"
    )
    fw = fw[fw["verified"].astype(bool)].copy()
    fw["practice_status"] = fw["practice_status"].fillna("none")
    fw["lp"] = [L6.ladder_pred(r, tab, KEYS)[0] for _, r in fw.iterrows()]
    lp_check = float(np.abs(fw["lp"] - fw["lp_designation_x_practice"]).max())
    cols = ["team", "week", "gsis_id", "position", "designation", "lp", "rp"]
    p = pd.concat([early[cols], fw[cols]], ignore_index=True)
    p["grp"] = p["position"].map(A.POS_TO_GROUP)
    return p, lp_check


def team_terms(p, val):
    p = p.copy()
    p["v"] = p["grp"].map(val)
    p["has_read"] = p["rp"].notna()
    p["valued"] = p["v"].notna()
    p["pr"] = p["rp"].fillna(p["lp"])
    q = p[p["valued"]]
    t = q.assign(
        x_lad=q["v"] * (1 - q["lp"]),
        x_read=q["v"] * (1 - q["pr"]),
        mass=q["v"].abs() * (1 - q["lp"]),
        mass_read=np.where(q["has_read"], q["v"].abs() * (1 - q["lp"]), 0.0),
    )
    agg = t.groupby(["team", "week"]).agg(
        x_lad=("x_lad", "sum"),
        x_read=("x_read", "sum"),
        mass=("mass", "sum"),
        mass_read=("mass_read", "sum"),
    )
    cnt = p.groupby(["team", "week"]).agg(listed=("gsis_id", "size"), read=("has_read", "sum"))
    return agg.join(cnt, how="outer").fillna(0.0)


def games(tt):
    served = pd.read_parquet(m.SERVED_FEATURES)
    g = served[
        (served["season"] == SEASON)
        & served["week"].between(FIRST_WEEK, LAST_WEEK)
        & (served["game_type"] == "REG")
        & served["result"].notna()
    ].copy()
    op = mwf.tuesday_open(g)
    g["open_line"] = g["game_id"].map(lambda i: op.get(i, (np.nan,))[0])
    pub = pd.read_parquet(L6.ROOT / "artifacts" / "clv_ledger" / "published_picks.parquet")
    pub = pub[pub["season"] == SEASON]
    pub = pub[
        pd.to_datetime(pub["published_at_utc"], utc=True)
        <= pd.to_datetime(pub["pick_deadline_utc"], utc=True)
    ]
    last = pub.sort_values("published_at_utc").groupby("game_id").tail(1)
    home = np.where(
        last["pick_team"] == last["home_team"], last["displayed_score"], 1 - last["displayed_score"]
    )
    g["p_served"] = g["game_id"].map(dict(zip(last["game_id"], home, strict=True)))
    g["pick_line"] = g["game_id"].map(
        dict(zip(last["game_id"], last["market_spread"], strict=True))
    )
    raw = wj.locked_raw_card(list(g["game_id"]), wj.deadlines_for(g))
    g["p_raw"] = g["game_id"].map(lambda i: raw[i]["raw_home_p"] if i in raw else np.nan)
    g["move"] = g["spread_line"] - g["open_line"]
    for side, team in (("h", "home_team"), ("a", "away_team")):
        k = list(zip(g[team], g["week"], strict=True))
        for c in ("x_lad", "x_read", "mass", "mass_read", "listed", "read"):
            g[f"{side}_{c}"] = [tt[c].get(kk, 0.0) for kk in k]
    g["x_lad"] = g["h_x_lad"] - g["a_x_lad"]
    g["x_read"] = g["h_x_read"] - g["a_x_read"]
    g = g.dropna(subset=["open_line", "p_served"])
    g = g[g["result"] != g["open_line"]].copy()
    g["y"] = (g["result"] > g["open_line"]).astype(float)
    return g.reset_index(drop=True)


def sig(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit(X, y, off):
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = sig(off + X @ b)
        gr = X.T @ (y - p)
        h = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(h, gr)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def lowo(X, y, off, wk):
    p = np.empty(len(y))
    coefs = {}
    for w in np.unique(wk):
        tr = wk != w
        b = fit(X[tr], y[tr], off[tr])
        coefs[int(w)] = b.tolist()
        p[~tr] = sig(off[~tr] + X[~tr] @ b)
    return p, coefs


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def br(p, y):
    return (p - y) ** 2


def boot(d, wk, rng):
    idx = {w: np.flatnonzero(wk == w) for w in np.unique(wk)}
    bs = np.empty(REPS)
    for i in range(REPS):
        pick = np.concatenate([rng.choice(v, len(v)) for v in idx.values()])
        bs[i] = d[pick].mean()
    return bs


def perm_x(x, wk, rng):
    out = x.copy()
    for w in np.unique(wk):
        i = np.flatnonzero(wk == w)
        out[i] = x[rng.permutation(i)]
    return out


def record(pa, pb, y):
    da = pa > 0.5
    db = pb > 0.5
    dec = da != db
    ya = (da == (y == 1))[dec]
    return {"decisive": int(dec.sum()), "a_wins": int(ya.sum()), "b_wins": int((~ya).sum())}


def contrast(name, pa, pb, y, wk, rng, null_fn):
    res = {}
    for sname, (lo, hi) in SUBSETS.items():
        msk = (wk >= lo) & (wk <= hi)
        cell = {"games": int(msk.sum()), "record_a_vs_b": record(pa[msk], pb[msk], y[msk])}
        for mname, f in (("log_loss", ll), ("brier", br)):
            d = f(pb[msk], y[msk]) - f(pa[msk], y[msk])
            bs = boot(d, wk[msk], rng)
            cell[mname] = {
                "gain": float(d.mean()),
                "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                "p_positive": float((bs > 0).mean()),
            }
        res[sname] = cell
    if null_fn is not None:
        nl = np.array([null_fn() for _ in range(REPS // 4)])
        for sname, (lo, hi) in SUBSETS.items():
            msk = (wk >= lo) & (wk <= hi)
            obs = float((ll(pb[msk], y[msk]) - ll(pa[msk], y[msk])).mean())
            vals = np.array([v[sname] for v in nl])
            res[sname]["log_loss"]["perm_p_one_sided"] = float((vals >= obs).mean())
    return {name: res}


def reliability(p, y, bins):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    return [
        {
            "bin": int(k),
            "n": int((b == k).sum()),
            "mean_p": float(p[b == k].mean()),
            "hit": float(y[b == k].mean()),
        }
        for k in range(bins)
        if (b == k).any()
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rep110 = json.loads((A.OUT / "report.json").read_text())
    val = rep110["weights_spread_pts_per_out"]
    p, lp_check = players()
    tt = team_terms(p, val)
    g = games(tt)
    y = g["y"].to_numpy()
    wk = g["week"].to_numpy()
    off = logit(g["p_served"].to_numpy())
    one = np.ones((len(g), 1))
    zero = np.zeros(len(g))
    xl = g["x_lad"].to_numpy()[:, None]
    xr = g["x_read"].to_numpy()[:, None]
    p_lad, c_lad = lowo(xl, y, off, wk)
    p_read, c_read = lowo(xr, y, off, wk)
    ps = g["p_served"].to_numpy()
    p_int, _ = lowo(one, y, zero, wk)
    mv = np.column_stack([one, g["move"].to_numpy()])
    p_move, c_move = lowo(mv, y, zero, wk)
    ok_raw = g["p_raw"].notna().to_numpy()
    p_raw = np.full(len(g), np.nan)
    if ok_raw.any():
        rx = np.column_stack([np.ones(ok_raw.sum()), logit(g.loc[ok_raw, "p_raw"].to_numpy())])
        p_raw[ok_raw], _ = lowo(rx, y[ok_raw], zero[ok_raw], wk[ok_raw])
    b_lad_in = fit(xl, y, off)
    b_read_in = fit(xr, y, off)
    insample = {
        "ladder_coef": float(b_lad_in[0]),
        "read_coef": float(b_read_in[0]),
        "ladder_ll_gain_in": float((ll(ps, y) - ll(sig(off + xl @ b_lad_in), y)).mean()),
        "read_ll_gain_in": float((ll(ps, y) - ll(sig(off + xr @ b_read_in), y)).mean()),
        "ladder_ll_gain_oos": float((ll(ps, y) - ll(p_lad, y)).mean()),
        "read_ll_gain_oos": float((ll(ps, y) - ll(p_read, y)).mean()),
    }

    def subset_gain(pa, pb):
        return {
            s: float(
                (
                    ll(pb[(wk >= lo) & (wk <= hi)], y[(wk >= lo) & (wk <= hi)])
                    - ll(pa[(wk >= lo) & (wk <= hi)], y[(wk >= lo) & (wk <= hi)])
                ).mean()
            )
            for s, (lo, hi) in SUBSETS.items()
        }

    def null_read_vs_lad():
        xp = xl[:, 0] + perm_x(xr[:, 0] - xl[:, 0], wk, rng)
        pp, _ = lowo(xp[:, None], y, off, wk)
        return subset_gain(pp, p_lad)

    def null_arm(x):
        def f():
            pp, _ = lowo(perm_x(x[:, 0], wk, rng)[:, None], y, off, wk)
            return subset_gain(pp, ps)

        return f

    rep = {
        "family": "lead108_wording_mixture",
        "looks": 24,
        "value_source": "lead110 report.json weights_spread_pts_per_out, OLS 2011-2025",
        "values": val,
        "served_base": (
            "last published card home probability at or before pick deadline "
            "(clv_ledger/published_picks.parquet); 2026 absent from pick_probability per_game"
        ),
        "fit": (
            "logistic offset logit(served), one slope, leave-one-week-out "
            "(2026 only, LOSO impossible)"
        ),
        "ladder_week9_max_abs_diff_vs_forward_scorer": lp_check,
        "games": len(g),
        "games_by_week": g.groupby("week").size().to_dict(),
        "opener_equals_pick_line": int(
            (g["open_line"] == -g["pick_line"]).sum() + (g["open_line"] == g["pick_line"]).sum()
        ),
        "raw_control_games": int(ok_raw.sum()),
        "coverage": {
            str(w): {
                "team_games": len(x),
                "listed_verified": int(x["listed"].sum()),
                "with_read": int(x["read"].sum()),
                "team_games_any_read": int((x["read"] > 0).sum()),
                "value_mass_share_read": float(x["mass_read"].sum() / x["mass"].sum())
                if x["mass"].sum()
                else 0.0,
            }
            for w, x in tt.reset_index().groupby("week")
        },
        "term_sd": {
            "x_lad": float(xl.std()),
            "x_read": float(xr.std()),
            "x_diff": float((xr - xl).std()),
        },
        "games_read_differs": int((np.abs(xr - xl)[:, 0] > 0).sum()),
        "coef_per_fold": {"ladder": c_lad, "read": c_read, "move": c_move},
        "insample_vs_oos": insample,
        "mean_log_loss": {
            k: float(ll(v, y).mean())
            for k, v in (
                ("served", ps),
                ("ladder", p_lad),
                ("read", p_read),
                ("intercept", p_int),
                ("move", p_move),
            )
        },
        "mean_brier": {
            k: float(br(v, y).mean())
            for k, v in (
                ("served", ps),
                ("ladder", p_lad),
                ("read", p_read),
                ("intercept", p_int),
                ("move", p_move),
            )
        },
        "reliability": {"served": reliability(ps, y, 5), "read": reliability(p_read, y, 5)},
    }
    rep.update(contrast("read_vs_ladder", p_read, p_lad, y, wk, rng, null_read_vs_lad))
    rep.update(contrast("read_vs_served", p_read, ps, y, wk, rng, null_arm(xr)))
    rep.update(contrast("ladder_vs_served", p_lad, ps, y, wk, rng, null_arm(xl)))
    ctrl = {}
    for nm, pc, msk in (
        ("intercept", p_int, np.ones(len(g), bool)),
        ("move", p_move, np.ones(len(g), bool)),
        ("raw", p_raw, ok_raw),
    ):
        out = {}
        for mname, f in (("log_loss", ll), ("brier", br)):
            d = f(pc[msk], y[msk]) - f(ps[msk], y[msk])
            bs = boot(d, wk[msk], rng)
            out[mname] = {
                "served_gain": float(d.mean()),
                "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                "p_positive": float((bs > 0).mean()),
            }
        out["games"] = int(msk.sum())
        ctrl[nm] = out
    rep["controls_served_minus"] = ctrl
    g.to_parquet(OUT / "games.parquet")
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
