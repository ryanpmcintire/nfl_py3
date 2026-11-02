import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import sim04_loso as base  # noqa: E402
import sim10_drive as sd  # noqa: E402

KEY_NUMBERS = (3, 7, 10, 14, 17)
SCRATCH = REPO / "tests" / "scratch" / "sim08_anchor"
ENGINE_DEFAULTS = {
    "tilt": "5",
    "models": 1,
    "pat": 1,
    "mufree": "1,4",
    "dr": 1,
    "qstrata": 1,
    "eoh": "outcome",
    "strength": str(REPO / "artifacts" / "sim11" / "strength_terms.json"),
    "droof": 1,
    "drmu": 0,
    "mulabel": 1,
    "realsched": 0,
    "hecarry": 1,
    "stfull": "",
    "st": "",
    "fgdec": "",
    "q4dec": str(sd.OUT / "q4dec_fg.json"),
    "latent": str(sd.OUT / "latent_fit_2009_25.json"),
    "h": str(sd.OUT / "bandwidths_week.json"),
}


def engine():
    a = Namespace(**ENGINE_DEFAULTS)
    os.environ["SIM10_LATENT"] = a.latent
    os.environ["SIM10_Q4DEC"] = a.q4dec
    os.environ["SIM10_FGDEC"] = a.fgdec
    os.environ["SIM10_FORM"] = "0"
    os.environ["SIM10_HECARRY"] = str(int(a.hecarry))
    os.environ["SIM10_STFULL"] = a.stfull
    os.environ["SIM10_ST"] = a.st
    sd._init(*sd.sim_params(a))
    return sd._W["eng"]


def play(eng, rng, x_diff, x_env, season, week):
    x_home = x_env + 0.5 * x_diff
    x_away = x_env - 0.5 * x_diff
    s = eng.s
    rh = (0.5 * s * x_home, 0.5 * s * x_away)
    ra = (0.5 * s * x_away, 0.5 * s * x_home)
    hs, as_, _, _, went_ot, _ = eng.game(rng, (rh, ra), int(season), float(week))
    return hs - as_, hs + as_, float(went_ot)


def load_frame():
    from nfl_ats.schedule_flag_features import default_opener_lines

    g = base.load_games()
    lines = default_opener_lines(g[["game_id", "season", "week"]])
    g = g.merge(lines[["game_id", "tue_open_total_line"]], on="game_id", how="left")
    g["total"] = g["tue_open_total_line"].astype(float)
    return g


def design_run(eng, pool, n, seed, x_diff_range, x_env_range):
    rng = np.random.default_rng(seed)
    rows = []
    pick = rng.integers(0, len(pool), n)
    xd = rng.uniform(-x_diff_range, x_diff_range, n)
    xe = rng.uniform(-x_env_range, x_env_range, n)
    for i in range(n):
        s_, w_ = pool[pick[i]]
        m, t, ot = play(eng, rng, xd[i], xe[i], s_, w_)
        rows.append((xd[i], xe[i], m, t, ot))
    return pd.DataFrame(rows, columns=["x_diff", "x_env", "margin", "total", "ot"])


def transmission_fit(D):
    X = np.column_stack([np.ones(len(D)), D["x_diff"], D["x_env"]])
    Xq = np.column_stack([X, D["x_diff"] ** 2, D["x_env"] ** 2, D["x_diff"] * D["x_env"]])
    out = {}
    for name in ("margin", "total"):
        y = D[name].to_numpy()
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        r1 = y - X @ c
        cq, *_ = np.linalg.lstsq(Xq, y, rcond=None)
        r2 = y - Xq @ cq
        out[f"{name}_coef"] = c.tolist()
        out[f"{name}_se"] = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * r1.var(ddof=3)).tolist()
        out[f"{name}_r2_linear"] = float(1 - r1.var() / y.var())
        out[f"{name}_r2_quadratic"] = float(1 - r2.var() / y.var())
        out[f"{name}_quad_coef"] = cq[3:].tolist()
    return out


def solve_latents(fit, margin_target, total_target):
    A = np.array([fit["margin_coef"][1:], fit["total_coef"][1:]])
    b = np.array([margin_target - fit["margin_coef"][0], total_target - fit["total_coef"][0]])
    x = np.linalg.solve(A, b)
    return float(x[0]), float(x[1])


def solve_diff_only(fit, margin_target):
    c = fit["margin_coef"]
    return float((margin_target - c[0]) / c[1])


def anchored_draws(eng, row, fit, n_draws, seed, spread_only=False):
    if spread_only or not np.isfinite(row["total"]):
        xd, xe = solve_diff_only(fit, row["line"]), 0.0
    else:
        xd, xe = solve_latents(fit, row["line"], row["total"])
    rng = np.random.default_rng(seed)
    out = np.empty((n_draws, 3))
    for k in range(n_draws):
        out[k] = play(eng, rng, xd, xe, row["season"], row["week"])
    return xd, xe, out


def key_mass(margins):
    a = np.abs(np.asarray(margins))
    return {int(k): float(np.mean(a == k)) for k in KEY_NUMBERS}


def free_gb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30


def cmd_probe(a):
    eng = engine()
    rng = np.random.default_rng(1)
    t = time.time()
    n = a.n
    rows = []
    for xd in (-a.x, 0.0, a.x):
        for xe in (-a.x, 0.0, a.x):
            r = np.array([play(eng, rng, xd, xe, 2025, 9)[:2] for _ in range(n)])
            rows.append((xd, xe, r[:, 0].mean(), r[:, 1].mean()))
    print(pd.DataFrame(rows, columns=["xd", "xe", "margin", "total"]).round(2).to_string())
    print("seconds per game", (time.time() - t) / (9 * n))


def cmd_smoke(a):
    t0 = time.time()
    frame = load_frame()
    held = a.held_out
    train = frame[frame["season"] != held]
    pool = list(zip(train["season"], train["week"], strict=True))
    eng = engine()
    t_engine = time.time() - t0
    t1 = time.time()
    D = design_run(eng, pool, a.design_n, a.seed, a.x_diff_range, a.x_env_range)
    t_design = time.time() - t1
    fit = transmission_fit(D)
    fit.update(
        {
            "design_n": a.design_n,
            "design_seconds": t_design,
            "seconds_per_design_game": t_design / a.design_n,
            "x_diff_range": a.x_diff_range,
            "x_env_range": a.x_env_range,
            "held_out": held,
        }
    )
    test = frame[frame["season"] == held].reset_index(drop=True)
    idx = np.linspace(0, len(test) - 1, a.games).round().astype(int)
    test = test.iloc[idx].reset_index(drop=True)
    recs, allm = [], []
    t2 = time.time()
    for i, r in test.iterrows():
        xd, xe, out = anchored_draws(eng, r, fit, a.draws, a.seed * 1000 + i)
        allm.append(out[:, 0])
        recs.append(
            {
                "game_id": r["game_id"],
                "line": float(r["line"]),
                "total": float(r["total"]) if np.isfinite(r["total"]) else None,
                "x_diff": xd,
                "x_env": xe,
                "sim_mean_margin": float(out[:, 0].mean()),
                "sim_mean_total": float(out[:, 1].mean()),
                "sim_margin_se": float(out[:, 0].std(ddof=1) / np.sqrt(len(out))),
                "sim_total_se": float(out[:, 1].std(ddof=1) / np.sqrt(len(out))),
                "ot_rate": float(out[:, 2].mean()),
            }
        )
    t_sim = time.time() - t2
    R = pd.DataFrame(recs)
    R["margin_z"] = (R["sim_mean_margin"] - R["line"]) / R["sim_margin_se"]
    R["total_z"] = (R["sim_mean_total"] - R["total"]) / R["sim_total_se"]
    has_t = R["total"].notna()
    allm = np.concatenate(allm)
    summary = {
        "transmission": fit,
        "games": len(R),
        "draws": a.draws,
        "mean_abs_margin_err": float((R["sim_mean_margin"] - R["line"]).abs().mean()),
        "mean_signed_margin_err": float((R["sim_mean_margin"] - R["line"]).mean()),
        "mean_abs_total_err": float(
            (R.loc[has_t, "sim_mean_total"] - R.loc[has_t, "total"]).abs().mean()
        ),
        "mean_signed_total_err": float(
            (R.loc[has_t, "sim_mean_total"] - R.loc[has_t, "total"]).mean()
        ),
        "margin_z_rms": float(np.sqrt((R["margin_z"] ** 2).mean())),
        "total_z_rms": float(np.sqrt((R.loc[has_t, "total_z"] ** 2).mean())),
        "x_diff_range_used": [float(R["x_diff"].min()), float(R["x_diff"].max())],
        "x_env_range_used": [float(R["x_env"].min()), float(R["x_env"].max())],
        "sim_abs_margin_mass": key_mass(allm),
        "real_abs_margin_mass_2020_25": key_mass(frame["result"].to_numpy()),
        "sim_ot_rate": float(R["ot_rate"].mean()),
        "seconds_engine_init": t_engine,
        "seconds_per_anchored_game_all_draws": t_sim / len(R),
        "seconds_per_draw": t_sim / (len(R) * a.draws),
        "free_gb_end": free_gb(),
        "per_game": R.to_dict("records"),
    }
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_game"}, indent=1), flush=True)


def planted_weight(x_served, y, delta):
    sign = 2.0 * y - 1.0
    base_ll = np.log1p(np.exp(-sign * x_served))

    def gain(w):
        return float(np.mean(base_ll - np.log1p(np.exp(-sign * (x_served + w * sign)))))

    lo, hi = 0.0, 4.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gain(mid) < delta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def loso_cond(x_cols, y_cover, nonpush, seasons):
    from sklearn.linear_model import LogisticRegression

    out = np.empty(len(y_cover))
    coefs = []
    for s_ in base.GRADED_SEASONS:
        tr = (seasons != s_) & nonpush
        te = seasons == s_
        m = LogisticRegression(C=1e10, max_iter=2000).fit(x_cols[tr], y_cover[tr])
        out[te] = m.predict_proba(x_cols[te])[:, 1]
        coefs.append([float(m.intercept_[0])] + [float(c) for c in m.coef_[0]])
    return out, coefs


def three_way(cond, push):
    return (1 - push) * cond, push, (1 - push) * (1 - cond)


def cmd_control(a):
    frame = base.load_games()
    cov, push, los = base.baseline_three_way(frame)
    outcome = frame["outcome"].to_numpy()
    seasons = frame["season"].to_numpy()
    nonpush = outcome != "push"
    y = (outcome == "cover").astype(float)
    served = base.safe_logit(cov / np.clip(cov + los, base.EPS, None))
    key = (frame["season"].astype(str) + "_" + frame["week"].astype(str)).to_numpy()
    _, inv = np.unique(key, return_inverse=True)
    nb = int(inv.max()) + 1
    rng = np.random.default_rng(a.seed)
    recal, _ = loso_cond(served[:, None], y, nonpush, seasons)
    ll_base = base.three_way_score(*three_way(recal, push), outcome)[0]
    results = []
    for delta in a.deltas:
        recal_logit = base.safe_logit(recal)
        w = planted_weight(recal_logit[nonpush], y[nonpush], delta)
        plant = w * (2.0 * y - 1.0)
        blend = 1.0 / (1.0 + np.exp(-(recal_logit + plant)))
        coefs = []
        ll_c = base.three_way_score(*three_way(blend, push), outcome)[0]
        d = ll_base - ll_c
        bs = np.bincount(inv, weights=d, minlength=nb)
        bn = np.bincount(inv, minlength=nb).astype(float)
        pps = np.empty(a.replicates)
        for r in range(a.replicates):
            outer = rng.integers(0, nb, nb)
            s_o, n_o = bs[outer], bn[outer]
            wts = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=a.inner)
            pps[r] = float(np.mean((wts @ s_o) / (wts @ n_o) > 0.0))
        det = float(np.mean(pps >= 0.90))
        results.append(
            {
                "target_delta": delta,
                "planted_weight": w,
                "realized_pooled_delta": float(d.mean()),
                "detect_rate_pp_ge_0.90": det,
                "median_pp": float(np.median(pps)),
                "detected": bool(det >= 0.80),
            }
        )
        brief = {k: v for k, v in results[-1].items()}
        print(json.dumps(brief), flush=True)
    Path(a.out).write_text(json.dumps(results, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--x", type=float, default=1.0)
    s = sub.add_parser("smoke")
    s.add_argument("--held-out", type=int, default=2025, dest="held_out")
    s.add_argument("--games", type=int, default=20)
    s.add_argument("--draws", type=int, default=200)
    s.add_argument("--design-n", type=int, default=2000, dest="design_n")
    s.add_argument("--x-diff-range", type=float, default=1.0, dest="x_diff_range")
    s.add_argument("--x-env-range", type=float, default=1.0, dest="x_env_range")
    s.add_argument("--seed", type=int, default=8)
    s.add_argument("--out", default=str(SCRATCH / "smoke.json"))
    c = sub.add_parser("control")
    c.add_argument("--deltas", type=float, nargs="+", default=[0.0010, 0.0080])
    c.add_argument("--replicates", type=int, default=500)
    c.add_argument("--inner", type=int, default=1000)
    c.add_argument("--seed", type=int, default=8)
    c.add_argument("--out", default=str(SCRATCH / "control.json"))
    a = ap.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    {"probe": cmd_probe, "smoke": cmd_smoke, "control": cmd_control}[a.cmd](a)


if __name__ == "__main__":
    main()
