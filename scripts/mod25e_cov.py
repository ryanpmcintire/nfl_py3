import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mod25_generator as gen  # noqa: E402

OUT = REPO / "artifacts" / "mod25e3" / "cov"
POOL = tuple(range(2009, 2018))
MAX_LAG = 6
NB = 2000


def lag_sums(tg):
    d = tg.copy()
    for c in ("off", "dfn"):
        d[c + "_dev"] = d[c] - d.groupby("season")[c].transform("mean")
    rows = {}
    for s, g in d.groupby("season"):
        r = []
        for k in range(1, MAX_LAG + 1):
            a = g.groupby("posteam")["off_dev"].shift(k)
            b = g.groupby("posteam")["dfn_dev"].shift(k)
            ok_a = a.notna().to_numpy()
            x = float(np.sum(g["dfn_dev"].to_numpy()[ok_a] * a.to_numpy()[ok_a]))
            y = float(np.sum(g["off_dev"].to_numpy()[ok_a] * b.to_numpy()[ok_a]))
            r.append((x + y, 2 * int(ok_a.sum())))
        rows[int(s)] = r
    return rows


def cross_cov(rows, seasons, phi_x):
    ks = np.arange(1, MAX_LAG + 1)
    c = np.zeros(MAX_LAG)
    n = np.zeros(MAX_LAG)
    for s in seasons:
        for i, (v, m) in enumerate(rows[s]):
            c[i] += v
            n[i] += m
    ck = c / n
    w = phi_x ** ks
    return float(np.sum(ck * w) / np.sum(w * w)), ck.tolist()


def season_mean_lag(tg):
    d = tg.copy()
    for c in ("off", "dfn"):
        d[c] = d[c] - d.groupby("season")[c].transform("mean")
    sm = d.groupby(["posteam", "season"])[["off", "dfn"]].mean().reset_index()
    nx = sm.copy()
    nx["season"] = nx["season"] - 1
    m = sm.merge(nx, on=["posteam", "season"], suffixes=("", "_n"))
    return float(0.5 * (np.mean(m["off"] * m["dfn_n"]) + np.mean(m["dfn"] * m["off_n"]))), float(np.mean(m["off"] * m["dfn"]))


def joint_latents(rng, n_seasons, fit, setting):
    qb = fit["qb"]
    var_off = max(1e-6, fit["off"]["var_mu"] - qb["backup_rate"] * (1 - qb["backup_rate"]) * qb["backup_off_epa_effect"] ** 2)
    var_def = fit["def"]["var_mu"]
    sd = np.array([math.sqrt(var_off), math.sqrt(var_def)])
    corr = fit["joint_corr_off_dfn"]
    sigma = np.array([[1.0, corr], [corr, 1.0]]) * np.outer(sd, sd)
    phi = np.array([fit["off"]["phi"], fit["def"]["phi"]]) ** setting["drift"]
    rho = np.array([fit["off"]["rho_offseason"], fit["def"]["rho_offseason"]])
    q_drift = sigma - np.outer(phi, phi) * sigma
    q_carry = sigma - np.outer(rho, rho) * sigma
    ch_d = np.linalg.cholesky(q_drift)
    ch_c = np.linalg.cholesky(q_carry)
    ch_l = np.linalg.cholesky(sigma)
    lat = rng.standard_normal((gen.N_TEAMS, 2)) @ ch_l.T
    seasons = []
    for s in range(n_seasons):
        if s > 0:
            lat = rho * lat + rng.standard_normal((gen.N_TEAMS, 2)) @ ch_c.T
        weekly = np.zeros((18, gen.N_TEAMS, 2))
        qb_out = np.zeros((18, gen.N_TEAMS), dtype=bool)
        state = np.zeros(gen.N_TEAMS, dtype=bool)
        cur = lat.copy()
        for w in range(18):
            if w > 0:
                cur = phi * cur + rng.standard_normal((gen.N_TEAMS, 2)) @ ch_d.T
            u = rng.random(gen.N_TEAMS)
            state = np.where(state, u < qb["p_stay_out"], u < qb["p_start_out"])
            weekly[w] = cur
            qb_out[w] = state
        lat = cur
        seasons.append((weekly, qb_out))
    return seasons


def load_joint_fit():
    return json.loads((OUT / "cov_fit.json").read_text())


def model_check(fit, fn, n_seasons=1500, seed=5):
    setting = dict(gen.SETTING_DEFAULTS)
    setting["drift"] = 1.0
    rng = np.random.default_rng(seed)
    lat = fn(rng, n_seasons, fit, setting)
    m = np.stack([w[0].mean(axis=0) for w in lat])
    m = m - m.mean(axis=1, keepdims=True)
    same = float(np.mean(m[:, :, 0] * m[:, :, 1]))
    lag = float(0.5 * (np.mean(m[:-1, :, 0] * m[1:, :, 1]) + np.mean(m[:-1, :, 1] * m[1:, :, 0])))
    return {"cov_same_season_means": same, "cov_lag1_season_means": lag, "corr_season_means": same / math.sqrt(np.mean(m[:, :, 0] ** 2) * np.mean(m[:, :, 1] ** 2))}


def cmd_fit(args):
    plays = gen.load_real_plays(POOL)
    tg = gen.team_game_epa(plays)
    fo = gen.fit_ar1(tg, "off")
    fd = gen.fit_ar1(tg, "dfn")
    base = json.loads((gen.OUT_DIR / "fit_params.json").read_text())
    phi_x = math.sqrt(base["off"]["phi"] * base["def"]["phi"])
    rows = lag_sums(tg)
    seasons = sorted(rows)
    c_hat, ck = cross_cov(rows, seasons, phi_x)
    denom = math.sqrt(fo["var_mu"] * fd["var_mu"])
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(NB):
        pick = [seasons[i] for i in rng.integers(0, len(seasons), len(seasons))]
        boots.append(cross_cov(rows, pick, phi_x)[0] / denom)
    boots = np.array(boots)
    corr = c_hat / denom
    lag_obs, same_obs = season_mean_lag(tg)
    base["joint_corr_off_dfn"] = corr
    base["joint_fit"] = {
        "pool": [POOL[0], POOL[-1]], "phi_cross": phi_x, "cross_cov": c_hat, "cross_lag_cov": ck,
        "pool_off": {"phi": fo["phi"], "var_mu": fo["var_mu"]}, "pool_def": {"phi": fd["phi"], "var_mu": fd["var_mu"]},
        "corr_dfn_units": corr, "corr_boot_lo": float(np.quantile(boots, .025)), "corr_boot_hi": float(np.quantile(boots, .975)),
        "prob_corr_negative": float((boots < 0).mean()),
        "scrim_def_orientation_corr": -corr,
        "obs_season_mean_cov_same": same_obs, "obs_season_mean_cov_lag1": lag_obs,
        "old_generator_corr": base["season_mean_corr_off_def"],
    }
    base["joint_fit"]["model_old"] = model_check(base, gen.gen_world_latents)
    base["joint_fit"]["model_joint"] = model_check(base, joint_latents)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cov_fit.json").write_text(json.dumps(base, indent=1))
    print(json.dumps(base["joint_fit"], indent=1))


def patch_generator():
    fit = load_joint_fit()
    gen.load_fit = lambda: fit
    gen.gen_world_latents = joint_latents


def cmd_sim(args):
    import sim09_hk as hk

    patch_generator()
    hk.dv.DV["crzhc"] = dict(hk.dv.DV["crzhk"])
    args.variant = "crzhc"
    hk.cmd_sim(args)


def cmd_e5(args):
    import sim09_hk as hk

    patch_generator()
    hk.dv.DV["crzhc"] = dict(hk.dv.DV["crzhk"])
    args.variant = "crzhc"
    hk.cmd_e5(args)


FEAT = OUT / "feat"


def cmd_simfeat(args):
    import mod25e_r2gap as rg

    patch_generator()
    rg.OUT = FEAT
    rg.cmd_simfeat(args)


def cmd_channels(args):
    import shutil

    import mod25e_channels as ch
    import mod25e_resid as mr

    root = OUT / "chan"
    (root / "e5_crzhk_s11").mkdir(parents=True, exist_ok=True)
    shutil.copy(FEAT / "feat_games.parquet", root / "e5_crzhk_s11" / "sim_games.parquet")
    mr.R2 = FEAT
    ch.E3 = root
    ch.OUT = root / "out"
    ch.main()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    s = sub.add_parser("sim")
    s.add_argument("--out-dir", dest="out_dir", required=True)
    s.add_argument("--worlds", type=int, default=1)
    s.add_argument("--seasons", type=int, default=1)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--seed", type=int, default=21)
    e = sub.add_parser("e5")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--seasons", type=int, default=8)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    f = sub.add_parser("simfeat")
    f.add_argument("--worlds", type=int, default=4)
    f.add_argument("--seasons", type=int, default=8)
    f.add_argument("--workers", type=int, default=3)
    f.add_argument("--seed", type=int, default=11)
    sub.add_parser("channels")
    args = ap.parse_args()
    {"fit": cmd_fit, "sim": cmd_sim, "e5": cmd_e5, "simfeat": cmd_simfeat, "channels": cmd_channels}[args.cmd](args)


if __name__ == "__main__":
    main()
