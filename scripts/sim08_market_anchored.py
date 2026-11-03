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
    "joint": 1,
    "mucurve": 0,
}


def engine():
    a = Namespace(**ENGINE_DEFAULTS)
    os.environ["SIM10_LATENT"] = a.latent
    os.environ["SIM10_Q4DEC"] = a.q4dec
    os.environ["SIM10_FGDEC"] = a.fgdec
    os.environ["SIM10_FORM"] = "0"
    os.environ["SIM10_JOINT"] = str(int(a.joint))
    os.environ["SIM10_MUCURVE"] = str(int(a.mucurve))
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
        out[f"{name}_quad_coef"] = cq.tolist()
    return out


def response(fit, name, xd, xe):
    c = fit[f"{name}_quad_coef"]
    return c[0] + c[1] * xd + c[2] * xe + c[3] * xd**2 + c[4] * xe**2 + c[5] * xd * xe


def solve_latents(fit, margin_target, total_target):
    from scipy.optimize import least_squares

    bx, be = fit["x_diff_range"], fit["x_env_range"]

    def resid(v):
        return [
            response(fit, "margin", v[0], v[1]) - margin_target,
            response(fit, "total", v[0], v[1]) - total_target,
        ]

    r = least_squares(resid, [0.0, 0.0], bounds=([-bx, -be], [bx, be]))
    return float(r.x[0]), float(r.x[1])


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


def cmd_support(a):
    import mod25_generator as gen

    os.environ.setdefault("QBC", "1")
    os.environ["SIM10_LATENT"] = ENGINE_DEFAULTS["latent"]
    fit = sd.load_latent_fit()
    setting = dict(gen.SETTING_DEFAULTS)
    s_ = None
    eng = engine()
    s_ = eng.s
    xd, xe = [], []
    ws = np.random.SeedSequence(a.seed).spawn(a.worlds)
    for w in range(a.worlds):
        rng = np.random.default_rng(ws[w])
        for weekly, qb_out in gen.gen_world_latents(rng, a.seasons, fit, setting):
            rt = gen.season_ratings(weekly, qb_out, fit, setting, 0.0, 0.0)
            for wk, h, aw in gen.make_schedule(rng):
                rh, ra = rt[wk][h], rt[wk][aw]
                xh = (rh[0] + ra[1]) / s_
                xa = (ra[0] + rh[1]) / s_
                xd.append(xh - xa)
                xe.append(0.5 * (xh + xa))
    qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    xd, xe = np.array(xd), np.array(xe)
    out = {
        "n": len(xd),
        "s": s_,
        "q": qs,
        "xdiff": np.quantile(xd, qs).tolist(),
        "xenv": np.quantile(xe, qs).tolist(),
        "sd": [float(xd.std()), float(xe.std())],
        "corr": float(np.corrcoef(xd, xe)[0, 1]),
        "min_max": [float(xd.min()), float(xd.max()), float(xe.min()), float(xe.max())],
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


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
    pool_frame = frame[frame["season"] == held] if a.heldout_pool else train
    pool = list(zip(pool_frame["season"], pool_frame["week"], strict=True))
    eng = engine()
    t_engine = time.time() - t0
    t1 = time.time()
    if a.design_csv:
        D = pd.read_csv(a.design_csv)
    else:
        D = design_run(eng, pool, a.design_n, a.seed, a.x_diff_range, a.x_env_range)
    t_design = time.time() - t1
    if a.save_design:
        D.to_csv(a.save_design, index=False)
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
                "sim_home_cover": float((out[:, 0] > r["line"]).mean()),
                "sim_home_loss": float((out[:, 0] < r["line"]).mean()),
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
        "margin_err_mean_z": float(
            (R["sim_mean_margin"] - R["line"]).mean()
            / ((R["sim_mean_margin"] - R["line"]).std(ddof=1) / np.sqrt(len(R)))
        ),
        "total_err_mean_z": float(
            (R.loc[has_t, "sim_mean_total"] - R.loc[has_t, "total"]).mean()
            / (
                (R.loc[has_t, "sim_mean_total"] - R.loc[has_t, "total"]).std(ddof=1)
                / np.sqrt(has_t.sum())
            )
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


def noisy_candidate(recal_logit, y, kappa, sd, rng):
    return recal_logit + kappa * (2.0 * y - 1.0) + rng.normal(0.0, sd, len(y))


def blend_delta(recal_logit, cand, y, nonpush, seasons, push, outcome, ll_base):
    X = np.column_stack([recal_logit, cand])
    p, _ = loso_cond(X, y, nonpush, seasons)
    ll = base.three_way_score(*three_way(p, push), outcome)[0]
    return float(np.mean(ll_base - ll))


def solve_kappa(recal_logit, y, nonpush, seasons, push, outcome, ll_base, sd, delta, seed, reps):
    def expected(kappa):
        vals = []
        for r in range(reps):
            rg = np.random.default_rng(seed + r)
            c = noisy_candidate(recal_logit, y, kappa, sd, rg)
            vals.append(blend_delta(recal_logit, c, y, nonpush, seasons, push, outcome, ll_base))
        return float(np.mean(vals))

    lo, hi = 0.0, 2.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if expected(mid) < delta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cmd_control_noisy(a):
    frame = base.load_games()
    cov, push, los = base.baseline_three_way(frame)
    outcome = frame["outcome"].to_numpy()
    seasons = frame["season"].to_numpy()
    nonpush = outcome != "push"
    y = (outcome == "cover").astype(float)
    served = base.safe_logit(cov / np.clip(cov + los, base.EPS, None))
    if a.sd_from_smoke:
        sm = json.loads(Path(a.sd_from_smoke).read_text())
        pg = pd.DataFrame(sm["per_game"])
        pg["sim_logit"] = base.safe_logit(
            (pg["sim_home_cover"] / (pg["sim_home_cover"] + pg["sim_home_loss"])).to_numpy()
        )
        pos = {g: i for i, g in enumerate(frame["game_id"])}
        idx = np.array([pos[g] for g in pg["game_id"]])
        train_sd = float(np.std(pg["sim_logit"].to_numpy() - served[idx], ddof=1))
        draws = int(sm["draws"])
    else:
        train_sd = a.train_sd
        draws = a.draws
    mc_se = 2.0 / np.sqrt(draws)
    sd = float(np.sqrt(train_sd**2 + mc_se**2))
    key = (frame["season"].astype(str) + "_" + frame["week"].astype(str)).to_numpy()
    _, inv = np.unique(key, return_inverse=True)
    nb = int(inv.max()) + 1
    rng = np.random.default_rng(a.seed)
    recal, _ = loso_cond(served[:, None], y, nonpush, seasons)
    recal_logit = base.safe_logit(recal)
    ll_base = base.three_way_score(*three_way(recal, push), outcome)[0]
    results = []
    for delta in a.deltas:
        kappa = solve_kappa(
            recal_logit,
            y,
            nonpush,
            seasons,
            push,
            outcome,
            ll_base,
            sd,
            delta,
            a.seed,
            a.kappa_reps,
        )
        pps, realized = np.empty(a.replicates), np.empty(a.replicates)
        for r in range(a.replicates):
            rg = np.random.default_rng(a.seed * 100003 + r)
            c = noisy_candidate(recal_logit, y, kappa, sd, rg)
            X = np.column_stack([recal_logit, c])
            p, _ = loso_cond(X, y, nonpush, seasons)
            d = ll_base - base.three_way_score(*three_way(p, push), outcome)[0]
            realized[r] = d.mean()
            bs = np.bincount(inv, weights=d, minlength=nb)
            bn = np.bincount(inv, minlength=nb).astype(float)
            outer = rng.integers(0, nb, nb)
            wts = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=a.inner)
            pps[r] = float(np.mean((wts @ bs[outer]) / (wts @ bn[outer]) > 0.0))
        det = float(np.mean(pps >= 0.90))
        results.append(
            {
                "target_delta": delta,
                "kappa": kappa,
                "noise_sd": sd,
                "train_sd": train_sd,
                "mc_logit_se": mc_se,
                "mean_realized_delta": float(realized.mean()),
                "detect_rate_pp_ge_0.90": det,
                "median_pp": float(np.median(pps)),
                "detected": bool(det >= 0.80),
            }
        )
        print(json.dumps(results[-1]), flush=True)
    Path(a.out).write_text(json.dumps(results, indent=1))


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
    s.add_argument("--design-csv", default="", dest="design_csv")
    s.add_argument("--heldout-pool", type=int, default=0, dest="heldout_pool")
    s.add_argument("--save-design", default="", dest="save_design")
    s.add_argument("--seed", type=int, default=8)
    s.add_argument("--out", default=str(SCRATCH / "smoke.json"))
    u = sub.add_parser("support")
    u.add_argument("--worlds", type=int, default=40)
    u.add_argument("--seasons", type=int, default=9)
    u.add_argument("--seed", type=int, default=8)
    u.add_argument("--out", default=str(SCRATCH / "support_joint.json"))
    c = sub.add_parser("control")
    c.add_argument("--deltas", type=float, nargs="+", default=[0.0010, 0.0080])
    c.add_argument("--replicates", type=int, default=500)
    c.add_argument("--inner", type=int, default=1000)
    c.add_argument("--noisy", type=int, default=0)
    c.add_argument("--train-sd", type=float, default=0.0, dest="train_sd")
    c.add_argument("--sd-from-smoke", default="", dest="sd_from_smoke")
    c.add_argument("--draws", type=int, default=372)
    c.add_argument("--kappa-reps", type=int, default=6, dest="kappa_reps")
    c.add_argument("--seed", type=int, default=8)
    c.add_argument("--out", default=str(SCRATCH / "control.json"))
    a = ap.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if a.cmd == "control" and a.noisy:
        cmd_control_noisy(a)
        return
    {"probe": cmd_probe, "smoke": cmd_smoke, "control": cmd_control, "support": cmd_support}[a.cmd](
        a
    )


if __name__ == "__main__":
    main()
