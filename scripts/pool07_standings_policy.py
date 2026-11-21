import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pool01_forward_sim import FIELD_FILE, K, load_games, past_state, rival_table, share_given_key
from scipy.optimize import minimize
from scipy.special import expit

OUT = Path("artifacts/pool07")
GAMES = 15
ARMS = {
    "realised": ("realised", "realised", True),
    "home_line_decides": ("home_line", "realised", True),
    "curve_decides": ("curve", "realised", True),
    "home_line_field": ("home_line", "home_line", True),
    "curve_field": ("curve", "curve", True),
    "no_skill": ("realised", "realised", False),
}


def policies(cuts):
    pol = [("a_max_prob", None, "prob", False), ("c_bp_leverage", None, "lev", False)]
    for name, cut in cuts.items():
        pol.append((f"b_tilt_{name}", cut, "prob", False))
    pol.append(("d_trailing_tilt_q50", cuts["q50"], "prob", True))
    return pol


def play(g, rng, state, pol, weeks, dm, dsd, our_points, our_bp0):
    P, E = state[0].shape
    delta = dm + dsd * rng.standard_normal((P, E))
    p, sh, uo, ut = (g[k] for k in ("p", "share", "u_o", "u_t"))
    n = len(p)
    res = {}
    for name in [x[0] for x in pol]:
        res[name] = {
            "pts": np.full(P, float(our_points)),
            "bp": np.full(P, float(our_bp0)),
            "wins": np.zeros(P),
        }
    fp, fb = state[0].copy(), state[1].copy()
    for _ in range(weeks):
        idx = rng.integers(0, n, size=(P, GAMES))
        pw, shw, uow, utw = p[idx], sh[idx], uo[idx], ut[idx]
        cover = rng.random(pw.shape) < pw
        thr = np.where(cover, shw, 1.0 - shw)
        tp = np.clip(thr[:, None, :] + delta[:, :, None], 0.0, 1.0)
        fp += (rng.random(tp.shape, dtype=np.float32) < tp).sum(axis=2)
        eo, et = np.exp(uow), np.exp(utw)
        z = (eo + et).sum(axis=1, keepdims=True)
        rate = (np.where(cover, eo, et) / z).sum(axis=1)
        fb += rng.random((P, E)) < rate[:, None]
        for name, cut, bpmode, trailing in pol:
            r = res[name]
            if cut is None:
                pick_o = np.ones(pw.shape, bool)
            else:
                near = np.abs(pw - 0.5) < cut
                tilt = near & (shw > 0.5)
                if trailing:
                    ours = r["pts"] * K + r["bp"]
                    top = (fp * K + fb).max(axis=1)
                    behind = ours < top
                    tilt = tilt & behind[:, None]
                pick_o = ~tilt
            win = np.where(pick_o, cover, ~cover)
            wp = np.where(pick_o, pw, 1.0 - pw)
            if bpmode == "prob":
                score = wp
            else:
                own = np.where(pick_o, eo, et) / z
                score = wp * (1.0 - own)
            j = np.argmax(score, axis=1)
            rows = np.arange(P)
            r["pts"] += win.sum(axis=1)
            r["bp"] += win[rows, j]
            r["wins"] += win.sum(axis=1)
    out = {}
    fkey = fp * K + fb
    for name, r in res.items():
        ours = r["pts"] * K + r["bp"]
        first = share_given_key(fkey, ours)
        greater = (fkey > ours[:, None]).sum(axis=1)
        equal = (fkey == ours[:, None]).sum(axis=1)
        out[name] = {
            "first": first,
            "top10": (greater < 10).astype(float),
            "rank": 1.0 + greater + 0.5 * equal,
            "wins": r["wins"],
        }
    return out


def summ(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))]


def lev_ratio(p, share):
    gain = 2.0 * share - 1.0
    return np.where(gain > 0, (2.0 * p - 1.0) / np.where(gain > 0, gain, 1.0), np.inf)


def play_lev(g, rng, state, dm, lams, weeks, dsd, our_points, our_bp0):
    P, E = state[0].shape
    delta = dm + dsd * rng.standard_normal((P, E))
    p, sh, uo, ut = (g[k] for k in ("p", "share", "u_o", "u_t"))
    ratio = lev_ratio(p, g["dec"])
    n, L = len(p), len(lams)
    pts = np.full((L, P), float(our_points))
    bp = np.full((L, P), float(our_bp0))
    wins = np.zeros((L, P))
    fp, fb = state[0].copy(), state[1].copy()
    for _ in range(weeks):
        top = (fp * K + fb).max(axis=1)
        behind = (pts * K + bp) < top[None, :]
        idx = rng.integers(0, n, size=(P, GAMES))
        pw, shw, uow, utw, rw = p[idx], sh[idx], uo[idx], ut[idx], ratio[idx]
        cover = rng.random(pw.shape) < pw
        thr = np.where(cover, shw, 1.0 - shw)
        tp = np.clip(thr[:, None, :] + delta[:, :, None], 0.0, 1.0)
        fp += (rng.random(tp.shape, dtype=np.float32) < tp).sum(axis=2)
        eo, et = np.exp(uow), np.exp(utw)
        z = (eo + et).sum(axis=1, keepdims=True)
        rate = (np.where(cover, eo, et) / z).sum(axis=1)
        fb += rng.random((P, E)) < rate[:, None]
        tilt = (rw[None, :, :] <= lams[:, None, None]) & behind[:, :, None]
        win = np.where(tilt, ~cover[None], cover[None])
        wp = np.where(tilt, 1.0 - pw[None], pw[None])
        j = wp.argmax(axis=2)
        pts += win.sum(axis=2)
        wins += win.sum(axis=2)
        bp += np.take_along_axis(win, j[:, :, None], axis=2)[:, :, 0]
    fkey = fp * K + fb
    first = np.stack([share_given_key(fkey, pts[i] * K + bp[i]) for i in range(L)])
    top10 = np.stack([((fkey > (pts[i] * K + bp[i])[:, None]).sum(axis=1) < 10) for i in range(L)])
    return first, wins, top10.astype(float)


def chunked(g, rng, state, dm, lams, a, dsd, our_bp):
    out = []
    for s in range(0, len(dm), a.chunk):
        st = (state[0][s : s + a.chunk], state[1][s : s + a.chunk])
        out.append(play_lev(g, rng, st, dm[s : s + a.chunk], lams, a.weeks, dsd, a.points, our_bp))
    return tuple(np.concatenate([o[i] for o in out], axis=1) for i in range(3))


def side_table(df):
    raw = pd.read_parquet(FIELD_FILE)
    rows = {}
    for gid, g in raw.groupby("game_id"):
        h = g[g["home"]].iloc[0]
        t = g[~g["home"]].iloc[0]
        rows[gid] = (
            -float(h["line"]),
            float(h["picks"]),
            float(t["picks"]),
            float(h["model_p"] > t["model_p"]),
            float(g["covered"].sum()),
        )
    return np.array([rows[gid] for gid in df["game_id"]])


def fit(X, n1, n0):
    def f(b):
        z = X @ b
        nll = (n1 * np.logaddexp(0.0, -z) + n0 * np.logaddexp(0.0, z)).sum()
        return nll, X.T @ ((n1 + n0) * expit(z) - n1)

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def shares(df, side, train, kind):
    if kind == "realised":
        return df["share"].to_numpy(), None
    x = side[:, 0]
    cols = [np.ones(len(x)), x]
    if kind == "curve":
        cols.append(x * np.abs(x))
    X = np.column_stack(cols)
    b = fit(X[train], side[train, 1], side[train, 2])
    q = expit(X @ b)
    return np.where(side[:, 3] > 0, q, 1.0 - q), [float(v) for v in b]


def tails(q, k):
    dist = np.ones(1)
    for v in q:
        dist = np.convolve(dist, [1.0 - v, v])
    return float(dist[k:].sum()), float(dist[: k + 1].sum())


def decisive(tilted):
    games = [t for t in tilted if not t["push"]]
    n = len(games)
    k = int(sum(t["tilt_side_covered"] for t in games))
    q = np.array([1.0 - t["p_served"] for t in games])
    up, lo = tails(q, k)
    up_h, lo_h = tails(np.full(n, 0.5), k)
    return {
        "games": n,
        "pushes_dropped": len(tilted) - n,
        "tilt_side_wins": k,
        "tilt_side_losses": n - k,
        "expected_tilt_wins_served": float(q.sum()),
        "exact_p_at_least_served": up,
        "exact_p_at_most_served": lo,
        "exact_p_at_least_half": up_h,
        "exact_p_at_most_half": lo_h,
    }


def lowo(a, df, rng, state, dm, dsd, our_bp):
    dec_kind, field_kind, skill = ARMS[a.arm]
    if not skill:
        dm, dsd = np.zeros_like(dm), 0.0
    side = side_table(df)
    cols = ("p", "u_o", "u_t")
    folds, tilted = [], []
    for w in sorted(df["week"].unique()):
        trm = (df["week"] != w).to_numpy()
        tem = ~trm
        dec, coef = shares(df, side, trm, dec_kind)
        fld, _ = shares(df, side, trm, field_kind)
        gtr = {k: df[k].to_numpy()[trm] for k in cols} | {"share": fld[trm], "dec": dec[trm]}
        gte = {k: df[k].to_numpy()[tem] for k in cols} | {"share": fld[tem], "dec": dec[tem]}
        rtr = lev_ratio(gtr["p"], gtr["dec"])
        cand = np.unique(rtr[np.isfinite(rtr)])
        lams = np.concatenate([[-np.inf], cand])
        ftr, _, _ = chunked(gtr, rng, state, dm, lams, a, dsd, our_bp)
        dtr = (ftr[1:] - ftr[0]).mean(axis=1)
        best = int(np.argmax(dtr))
        lam = float(cand[best]) if dtr[best] > 0 else -np.inf
        fte, wte, tte = chunked(gte, rng, state, dm, np.array([-np.inf, lam]), a, dsd, our_bp)
        d = fte[1] - fte[0]
        dw = (wte[1] - wte[0]) / a.weeks
        dt = tte[1] - tte[0]
        rte = lev_ratio(gte["p"], gte["dec"])
        tl = rte <= lam
        te = df[tem].reset_index(drop=True)
        ste = side[tem]
        for i in np.where(tl)[0]:
            tilted.append(
                {
                    "week": int(w),
                    "game_id": str(te.loc[i, "game_id"]),
                    "p_served": float(gte["p"][i]),
                    "share_decision": float(gte["dec"][i]),
                    "share_realised": float(te.loc[i, "share"]),
                    "tilt_side_covered": bool(not te.loc[i, "cov_o"]),
                    "push": bool(ste[i, 4] != 1.0),
                }
            )
        folds.append(
            {
                "week": int(w),
                "lambda": lam if np.isfinite(lam) else None,
                "decision_share_coef": coef,
                "train_candidates": len(cand),
                "train_games_tilted": int((rtr <= lam).sum()),
                "heldout_games": len(te),
                "heldout_games_tilted": int(tl.sum()),
                "in_sample_d_first_pp": float(100.0 * max(dtr[best], 0.0)),
                "heldout_d_first_pp": float(100.0 * d.mean()),
                "heldout_d_first_se_pp": float(100.0 * d.std(ddof=1) / np.sqrt(len(d))),
                "heldout_p_first_max_prob": float(fte[0].mean()),
                "heldout_d_wins_per_week": float(dw.mean()),
                "heldout_d_wins_per_week_se": float(dw.std(ddof=1) / np.sqrt(len(dw))),
                "heldout_cost_if_trailing_per_week": float(
                    GAMES * ((2.0 * gte["p"] - 1.0) * tl).mean()
                ),
                "heldout_d_top10_pp": float(100.0 * dt.mean()),
            }
        )
        print({k: v for k, v in folds[-1].items() if k != "decision_share_coef"})
    ho = np.array([f["heldout_d_first_pp"] for f in folds])
    ins = np.array([f["in_sample_d_first_pp"] for f in folds])
    hw = np.array([f["heldout_d_wins_per_week"] for f in folds])
    ht = np.array([f["heldout_d_top10_pp"] for f in folds])
    idx = rng.choice(len(ho), size=(a.boot, len(ho)), replace=True)
    boot, bw, bt = ho[idx].mean(axis=1), hw[idx].mean(axis=1), ht[idx].mean(axis=1)
    lam_fin = [f["lambda"] for f in folds if f["lambda"] is not None]
    return {
        "start": vars(a),
        "arm": {
            "name": a.arm,
            "decision_share": dec_kind,
            "field_share": field_kind,
            "entrant_skill": skill,
        },
        "paths": len(dm),
        "dsd": dsd,
        "paid_places": 1,
        "rule": "tilt a game to the less-owned side when trailing and (2p-1)/(2share-1) <= lambda",
        "folds": folds,
        "heldout_d_first_pp": float(ho.mean()),
        "heldout_interval_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "heldout_probability_positive": float((boot > 0).mean()),
        "in_sample_d_first_pp": float(ins.mean()),
        "gap_pp": float(ins.mean() - ho.mean()),
        "folds_heldout_positive": int((ho > 0).sum()),
        "heldout_d_wins_per_week": float(hw.mean()),
        "heldout_d_wins_per_week_interval_95": [
            float(np.quantile(bw, 0.025)),
            float(np.quantile(bw, 0.975)),
        ],
        "heldout_cost_if_trailing_per_week": float(
            np.mean([f["heldout_cost_if_trailing_per_week"] for f in folds])
        ),
        "heldout_d_top10_pp": float(ht.mean()),
        "heldout_d_top10_interval_95": [
            float(np.quantile(bt, 0.025)),
            float(np.quantile(bt, 0.975)),
        ],
        "lambda_min_max": [min(lam_fin), max(lam_fin)] if lam_fin else None,
        "folds_no_tilt": len(folds) - len(lam_fin),
        "decisive": decisive(tilted),
        "tilted_games": tilted,
    }


def summary(wk):
    rows = {}
    for arm in ARMS:
        f = OUT / f"pool07_lowo_{arm}_w{wk:02d}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        rows[arm] = {
            k: r[k]
            for k in (
                "arm",
                "heldout_d_first_pp",
                "heldout_interval_95",
                "heldout_probability_positive",
                "in_sample_d_first_pp",
                "gap_pp",
                "folds_heldout_positive",
                "heldout_d_wins_per_week",
                "heldout_d_wins_per_week_interval_95",
                "heldout_cost_if_trailing_per_week",
                "heldout_d_top10_pp",
                "heldout_d_top10_interval_95",
                "lambda_min_max",
                "folds_no_tilt",
                "decisive",
            )
        }
    pts = [v["heldout_d_first_pp"] for v in rows.values()]
    res = {
        "arms": rows,
        "looks_heldout_cells": len(rows),
        "point_range_pp": [min(pts), max(pts)],
        "interval_envelope_pp": [
            min(v["heldout_interval_95"][0] for v in rows.values()),
            max(v["heldout_interval_95"][1] for v in rows.values()),
        ],
    }
    (OUT / f"pool07_lowo_field_arms_w{wk:02d}.json").write_text(json.dumps(res, indent=1) + "\n")
    for arm, v in rows.items():
        dv = v["decisive"]
        print(
            f"{arm:18s} dFirst {v['heldout_d_first_pp']:+.3f} {v['heldout_interval_95']} "
            f"P+ {v['heldout_probability_positive']:.3f} folds+ {v['folds_heldout_positive']} "
            f"gap {v['gap_pp']:+.3f} dWins/wk {v['heldout_d_wins_per_week']:+.4f} "
            f"costIfTrail {v['heldout_cost_if_trailing_per_week']:.4f} "
            f"dTop10 {v['heldout_d_top10_pp']:+.3f} "
            f"decisive {dv['tilt_side_wins']}-{dv['tilt_side_losses']} "
            f"exp {dv['expected_tilt_wins_served']:.2f} "
            f"p>=served {dv['exact_p_at_least_served']:.3f} "
            f"p>=half {dv['exact_p_at_least_half']:.3f}"
        )
    print({k: v for k, v in res.items() if k != "arms"})


def main():
    ap = argparse.ArgumentParser(description="POOL-07 standings-conditional policy challenger")
    ap.add_argument("--mode", choices=("quartile", "lowo", "lowo_summary"), default="quartile")
    ap.add_argument("--arm", choices=tuple(ARMS), default="realised")
    ap.add_argument("--paths", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--weeks", type=int, default=9)
    ap.add_argument("--points", type=int, default=65)
    ap.add_argument("--rank", type=int, default=19)
    ap.add_argument("--entries", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261105)
    a = ap.parse_args()
    df = load_games()
    if a.mode == "lowo_summary":
        summary(int(df["week"].max()))
        return
    rng = np.random.default_rng(a.seed)
    E = a.entries - 1
    table = rival_table(a.points)
    kp, kb, kd, have = [], [], [], 0
    while have < a.paths:
        pp, bb, dd, dsd, _ = past_state(df, rng, E, 2000, a.points, a.rank, table)
        kp.append(pp)
        kb.append(bb)
        kd.append(dd)
        have += len(pp)
    state = (np.concatenate(kp)[: a.paths], np.concatenate(kb)[: a.paths])
    dm = np.concatenate(kd)[: a.paths]
    our_bp = float(
        sum(g["cov_o"].to_numpy()[int(np.argmax(g["p"].to_numpy()))] for _, g in df.groupby("week"))
    )
    if a.mode == "lowo":
        res = lowo(a, df, rng, state, dm, dsd, our_bp)
        OUT.mkdir(parents=True, exist_ok=True)
        name = f"pool07_lowo_{a.arm}_w{int(df['week'].max()):02d}.json"
        (OUT / name).write_text(json.dumps(res, indent=1) + "\n")
        print({k: v for k, v in res.items() if k not in ("folds", "start", "tilted_games")})
        return
    arms = {
        "base": (df, dm, dsd),
        "no_skill": (df, np.zeros_like(dm), 0.0),
        "shares_weeks_1_4": (df[df["week"] <= 4], dm, dsd),
        "shares_weeks_5_8": (df[df["week"] >= 5], dm, dsd),
    }
    result = {"start": vars(a), "dsd": dsd, "paid_places": 1, "arms": {}}
    looks = 0
    for arm, (sub, dmm, dss) in arms.items():
        g = {k: sub[k].to_numpy() for k in ("p", "share", "u_o", "u_t")}
        dev = np.abs(g["p"] - 0.5)
        cuts = {f"q{int(q * 100)}": float(np.quantile(dev, q)) for q in (0.25, 0.5, 0.75)}
        pol = policies(cuts)
        out = play(g, rng, state, pol, a.weeks, dmm, dss, a.points, our_bp)
        base = out["a_max_prob"]
        rows = {}
        for name, o in out.items():
            looks += 1
            rows[name] = {
                "p_first": summ(o["first"]),
                "p_top10": summ(o["top10"]),
                "exp_rank": summ(o["rank"]),
                "exp_wins": summ(o["wins"]),
                "d_first_vs_a": summ(o["first"] - base["first"]),
                "d_top10_vs_a": summ(o["top10"] - base["top10"]),
                "d_wins_vs_a": summ(o["wins"] - base["wins"]),
            }
        result["arms"][arm] = {"cuts": cuts, "n_games": len(sub), "policies": rows}
    result["looks"] = looks
    result["paths"] = len(dm)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pool07_standings_policy.json").write_text(json.dumps(result, indent=1) + "\n")
    for arm, v in result["arms"].items():
        print(arm, v["cuts"])
        for n, r in v["policies"].items():
            print(
                f"  {n:22s} first {r['p_first'][0]:.4f}+-{r['p_first'][1]:.4f} "
                f"top10 {r['p_top10'][0]:.4f}+-{r['p_top10'][1]:.4f} "
                f"rank {r['exp_rank'][0]:.1f} wins {r['exp_wins'][0]:.2f} "
                f"dFirst {r['d_first_vs_a'][0]:+.4f}+-{r['d_first_vs_a'][1]:.4f} "
                f"dWins {r['d_wins_vs_a'][0]:+.2f}"
            )


if __name__ == "__main__":
    main()
