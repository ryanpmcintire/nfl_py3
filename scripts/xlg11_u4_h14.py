import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mod24_more_games as base  # noqa: E402
import mod24_pooled_levers as lev  # noqa: E402
import pooled_screen as ps  # noqa: E402
import xlg11_cfb_centered as u1  # noqa: E402

GRID = np.array(u1.STRENGTHS)
PERMS = 100
OUT = ROOT / "artifacts" / "xlg11"
IDENTITY = {"base_wins": 754, "games": 1503, "arm_wins": 771, "flips": [107, 90]}
STATE = {"gram_mismatch": 0, "lam_mismatch": 0, "fits": 0}
LOG = []


def gram(a, y):
    return {
        "n": float(len(y)),
        "g": a.T @ a,
        "s": a.sum(axis=0),
        "zy": a.T @ y,
        "sy": float(y.sum()),
        "yy": float(y @ y),
    }


def minus(total, part):
    return {k: total[k] - part[k] for k in total}


def decomp(t):
    mu = t["s"] / t["n"]
    w, v = np.linalg.eigh(t["g"] - t["n"] * np.outer(mu, mu))
    return {
        "mu": mu,
        "ybar": t["sy"] / t["n"],
        "w": w,
        "v": v,
        "vcy": v.T @ (t["zy"] - mu * t["sy"]),
    }


def betas(dec, centres, lams):
    lams = np.asarray(lams, dtype=float)
    num = dec["vcy"][:, None] + lams[None, :] * (dec["v"].T @ centres)
    beta = dec["v"] @ (num / (dec["w"][:, None] + lams[None, :]))
    return beta, dec["ybar"] - dec["mu"] @ beta


def sse(t, beta, b0):
    quad = np.einsum("ij,ij->j", beta, t["g"] @ beta)
    return (
        t["yy"]
        - 2.0 * (t["zy"] @ beta)
        - 2.0 * b0 * t["sy"]
        + quad
        + 2.0 * b0 * (t["s"] @ beta)
        + t["n"] * b0**2
    )


def loso(a, y, seasons, centres, grid):
    total = gram(a, y)
    held = np.zeros((len(grid), centres.shape[1]))
    for s in np.unique(seasons):
        m = seasons == s
        part = gram(a[m], y[m])
        dec = decomp(minus(total, part))
        for i, lam in enumerate(grid):
            beta, b0 = betas(dec, centres, np.full(centres.shape[1], lam))
            held[i] += sse(part, beta, b0)
    return total, held


def sign_acc(fit, y):
    nz = y != 0
    return float(np.mean(np.sign(fit[nz]) == np.sign(y[nz])))


def arms_for(train, test, cols):
    cfb = STATE["cfb"]
    cutoff = train["gameday"].max()
    past = cfb.loc[cfb["gameday"].lt(cutoff)]
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    a = sc.fit_transform(imp.fit_transform(train[cols].to_numpy(float)))
    b = sc.transform(imp.transform(test[cols].to_numpy(float)))
    y = train["ats_margin"].to_numpy(float)
    seasons = train["season"].to_numpy()
    n = len(y)
    ca = sc.transform(imp.transform(past[cols].to_numpy(float)))
    cy = past["ats_margin"].to_numpy(float)
    ratio = float(np.std(y, ddof=1) / np.std(cy, ddof=1))
    prior = ratio * Ridge(alpha=lev.ALPHA).fit(ca, cy).coef_
    lam = u1.pick_strength(a, y, seasons, prior)
    m = Ridge(alpha=lam).fit(a, y - a @ prior)
    real_pred = b @ prior + m.predict(b)
    real_fit = a @ prior + m.predict(a)
    out = {"variant": (real_pred, float(np.std(y - real_fit, ddof=1)))}
    zero = np.zeros(len(cols))
    centres = np.column_stack([prior, zero] + [prior[p] for p in STATE["perms"]])
    total, held = loso(a, y, seasons, centres, GRID)
    best = held.argmin(axis=0)
    lam_k = GRID[best]
    dec = decomp(total)
    beta, b0 = betas(dec, centres, lam_k)
    sig = np.sqrt(sse(total, beta, b0) / (n - 1.0))
    STATE["fits"] += 1
    if lam_k[0] != lam:
        STATE["lam_mismatch"] += 1
    elif not np.allclose(b @ beta[:, 0] + b0[0], real_pred, atol=1e-6):
        STATE["gram_mismatch"] += 1
    for k in range(PERMS):
        out[f"perm_{k:03d}"] = (b @ beta[:, 2 + k] + b0[2 + k], float(sig[2 + k]))
    zb, z0 = betas(dec, zero[:, None], [lam])
    out["zero_same"] = (b @ zb[:, 0] + z0[0], float(np.sqrt(sse(total, zb, z0)[0] / (n - 1.0))))
    out["int_only"] = (np.full(len(b), float(y.mean())), float(np.std(y, ddof=1)))
    lt = train["spread_line"].to_numpy(float)
    la = ((lt - lt.mean()) / lt.std())[:, None]
    lb = ((test["spread_line"].to_numpy(float) - lt.mean()) / lt.std())[:, None]
    ml = Ridge(alpha=lev.ALPHA).fit(la, y)
    out["line_only"] = (ml.predict(lb), float(np.std(y - ml.predict(la), ddof=1)))
    m0 = Ridge(alpha=lev.ALPHA).fit(a, y)
    base_fit = m0.predict(a)
    i_base = int(np.argmin(np.abs(GRID - lev.ALPHA)))
    LOG.append(
        {
            "season": int(test["season"].iloc[0]),
            "week": int(test["week"].iloc[0]),
            "lam": float(lam),
            "lam_zero_opt": float(lam_k[1]),
            "placebo_lam_median": float(np.median(lam_k[2:])),
            "placebo_top_share": float(np.mean(lam_k[2:] == GRID[-1])),
            "placebo_bottom_share": float(np.mean(lam_k[2:] == GRID[0])),
            "n": n,
            "cfb_n": len(past),
            "ratio": ratio,
            "train_mse_real": float(np.mean((y - real_fit) ** 2)),
            "loso_mse_real": float(held[best[0], 0] / n),
            "train_mse_base": float(np.mean((y - base_fit) ** 2)),
            "loso_mse_base": float(held[i_base, 1] / n),
            "acc_in_real": sign_acc(real_fit, y),
            "acc_in_base": sign_acc(base_fit, y),
        }
    )
    return out


def install():
    original = lev.predict_arms

    def patched(train, test, cols, arms):
        out = original(train, test, cols, ())
        out.update(arms_for(train, test, cols))
        return out

    lev.predict_arms = patched


def paired(g, arm, ref, rng):
    b = g.loc[g["arm"].eq(ref)].set_index("game_id")
    a = g.loc[g["arm"].eq(arm)].set_index("game_id")
    idx = b.index.intersection(a.index)
    p = pd.DataFrame(
        {
            "block": b.loc[idx, "season"].astype(str) + "-" + b.loc[idx, "week"].astype(str),
            "d_acc": 100 * (a.loc[idx, "correct"] - b.loc[idx, "correct"]),
            "d_ll": b.loc[idx, "ll"] - a.loc[idx, "ll"],
            "d_br": b.loc[idx, "br"] - a.loc[idx, "br"],
        }
    )
    flips = a.loc[idx, "correct"] - b.loc[idx, "correct"]
    return {
        "arm": arm,
        "ref": ref,
        "n": len(idx),
        "arm_wins": int(a.loc[idx, "correct"].sum()),
        "ref_wins": int(b.loc[idx, "correct"].sum()),
        "flips_win": int((flips > 0).sum()),
        "flips_loss": int((flips < 0).sum()),
        "arm_ll": float(a.loc[idx, "ll"].mean()),
        "ref_ll": float(b.loc[idx, "ll"].mean()),
        "arm_br": float(a.loc[idx, "br"].mean()),
        "ref_br": float(b.loc[idx, "br"].mean()),
        "accuracy": base.block_boot(p, "d_acc", rng),
        "log_loss": base.block_boot(p, "d_ll", rng),
        "brier": base.block_boot(p, "d_br", rng),
    }


def placebo_read(real_wins, placebo_wins, n):
    pw = np.asarray(placebo_wins, dtype=float)
    diff = 100.0 * (real_wins - pw) / n
    return {
        "count": len(pw),
        "placebo_wins_mean": float(pw.mean()),
        "placebo_wins_sd": float(pw.std(ddof=1)),
        "placebo_wins_min": float(pw.min()),
        "placebo_wins_q": [float(q) for q in np.percentile(pw, [2.5, 25, 50, 75, 97.5])],
        "placebo_wins_max": float(pw.max()),
        "real_wins": int(real_wins),
        "effect": float(diff.mean()),
        "low": float(np.percentile(diff, 2.5)),
        "high": float(np.percentile(diff, 97.5)),
        "share_beaten": float(np.mean(real_wins > pw) + 0.5 * np.mean(real_wins == pw)),
        "placebos_at_or_above_real": int(np.sum(pw >= real_wins)),
    }


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cfb = pd.read_parquet(base.CFB_FEATURES)
    cfb["gameday"] = pd.to_datetime(cfb["gameday"])
    cols = ps.shared_columns()
    STATE["cfb"] = base.cfb_prior_rows(cfb)
    rng = np.random.default_rng(base.SEED)
    perms = []
    ident = np.arange(len(cols))
    while len(perms) < PERMS:
        p = rng.permutation(len(cols))
        if not np.array_equal(p, ident):
            perms.append(p)
    STATE["perms"] = perms
    install()
    g = ps.nfl_graded(cols)
    print("walk seconds", round(time.time() - t0, 1), "fits", STATE["fits"], flush=True)
    print(
        "gram_mismatch", STATE["gram_mismatch"], "lam_mismatch", STATE["lam_mismatch"], flush=True
    )
    pops = {
        "nfl_opener_2020_25": g.loc[g["season"].between(*ps.NFL_OPENER_SEASONS)],
        "nfl_proxy_2011_19": g.loc[g["season"].between(*ps.NFL_PROXY_SEASONS)],
    }
    lg = pd.DataFrame(LOG)
    report = {
        "columns": cols,
        "grid": [float(x) for x in GRID],
        "fits": STATE["fits"],
        "gram_mismatch": STATE["gram_mismatch"],
        "lam_mismatch": STATE["lam_mismatch"],
        "populations": {},
    }
    for name, pop in pops.items():
        res = {
            "R_vs_base": paired(pop, "variant", "base", rng),
            "Z_vs_base": paired(pop, "zero_same", "base", rng),
            "R_vs_Z": paired(pop, "variant", "zero_same", rng),
            "int_only_vs_base": paired(pop, "int_only", "base", rng),
            "line_only_vs_base": paired(pop, "line_only", "base", rng),
            "R_vs_int_only": paired(pop, "variant", "int_only", rng),
            "R_vs_line_only": paired(pop, "variant", "line_only", rng),
        }
        n = res["R_vs_base"]["n"]
        perm_rows = pop.loc[pop["arm"].str.startswith("perm_")]
        by = perm_rows.groupby("arm")[["correct", "ll", "br"]].agg(
            {"correct": "sum", "ll": "mean", "br": "mean"}
        )
        res["placebo"] = placebo_read(res["R_vs_base"]["arm_wins"], by["correct"].to_numpy(), n)
        res["placebo"]["base_wins"] = res["R_vs_base"]["ref_wins"]
        res["placebo"]["placebos_above_base"] = int(
            np.sum(by["correct"].to_numpy() > res["R_vs_base"]["ref_wins"])
        )
        res["placebo"]["placebo_ll_mean"] = float(by["ll"].mean())
        res["placebo"]["placebo_ll_q"] = [
            float(q) for q in np.percentile(by["ll"], [2.5, 50, 97.5])
        ]
        res["placebo"]["real_ll_share_beaten"] = float(
            np.mean(res["R_vs_base"]["arm_ll"] < by["ll"])
        )
        res["placebo"]["placebo_br_mean"] = float(by["br"].mean())
        lo, hi = ps.NFL_OPENER_SEASONS if name == "nfl_opener_2020_25" else ps.NFL_PROXY_SEASONS
        era = lg.loc[lg["season"].between(lo, hi)]
        res["in_out"] = {
            "fits": len(era),
            "real_in_sample_acc": float(100 * era["acc_in_real"].mean()),
            "real_out_sample_acc": float(100 * res["R_vs_base"]["arm_wins"] / n),
            "base_in_sample_acc": float(100 * era["acc_in_base"].mean()),
            "base_out_sample_acc": float(100 * res["R_vs_base"]["ref_wins"] / n),
            "real_train_mse": float(era["train_mse_real"].mean()),
            "real_loso_mse": float(era["loso_mse_real"].mean()),
            "base_train_mse": float(era["train_mse_base"].mean()),
            "base_loso_mse": float(era["loso_mse_base"].mean()),
        }
        res["in_out"]["real_acc_gap"] = (
            res["in_out"]["real_in_sample_acc"] - res["in_out"]["real_out_sample_acc"]
        )
        res["in_out"]["base_acc_gap"] = (
            res["in_out"]["base_in_sample_acc"] - res["in_out"]["base_out_sample_acc"]
        )
        report["populations"][name] = res
        for key in (
            "R_vs_base",
            "Z_vs_base",
            "R_vs_Z",
            "int_only_vs_base",
            "line_only_vs_base",
            "R_vs_int_only",
            "R_vs_line_only",
        ):
            c = res[key]
            e = c["accuracy"]
            print(
                name,
                key,
                c["arm_wins"],
                c["ref_wins"],
                c["n"],
                "flips",
                c["flips_win"],
                c["flips_loss"],
                "acc",
                round(e["effect"], 2),
                round(e["low"], 2),
                round(e["high"], 2),
                "P+",
                round(e["p_pos"], 3),
                "ll",
                round(c["arm_ll"], 5),
                round(c["ref_ll"], 5),
                "llP+",
                round(c["log_loss"]["p_pos"], 3),
                "br",
                round(c["arm_br"], 5),
                round(c["ref_br"], 5),
                "brP+",
                round(c["brier"]["p_pos"], 3),
                flush=True,
            )
        print(name, "placebo", json.dumps(res["placebo"]), flush=True)
        print(name, "in_out", json.dumps(res["in_out"]), flush=True)
    fold = lg.groupby("season")[["lam", "lam_zero_opt", "placebo_lam_median"]].median()
    fold["lam_min"] = lg.groupby("season")["lam"].min()
    fold["lam_max"] = lg.groupby("season")["lam"].max()
    fold["placebo_top_share"] = lg.groupby("season")["placebo_top_share"].mean()
    fold["placebo_bottom_share"] = lg.groupby("season")["placebo_bottom_share"].mean()
    report["strength_by_season"] = json.loads(fold.to_json(orient="index"))
    report["cfb_ratio_median"] = float(lg["ratio"].median())
    c = report["populations"]["nfl_opener_2020_25"]["R_vs_base"]
    got = {
        "base_wins": c["ref_wins"],
        "games": c["n"],
        "arm_wins": c["arm_wins"],
        "flips": [c["flips_win"], c["flips_loss"]],
    }
    report["identity"] = {"expected": IDENTITY, "got": got, "ok": got == IDENTITY}
    report["seconds"] = time.time() - t0
    (OUT / "report_h14.json").write_text(json.dumps(report, indent=1))
    lg.to_csv(OUT / "fits_h14.csv", index=False)
    print(fold.to_string(), flush=True)
    print("identity", json.dumps(report["identity"]), flush=True)
    print("H14_DONE seconds", round(report["seconds"], 1), flush=True)
    if not report["identity"]["ok"]:
        raise SystemExit("identity check failed")


if __name__ == "__main__":
    main()
