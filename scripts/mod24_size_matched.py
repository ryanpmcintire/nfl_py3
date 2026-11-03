import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mod24_more_games as base  # noqa: E402

from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS  # noqa: E402

OUT = base.OUT
SUB_SEEDS = tuple(base.SEED + k for k in range(1, 6))


def nfl_rows_per_season():
    nfl = pd.read_parquet(base.NFL_FEATURES)
    nfl = nfl.loc[nfl["game_type"].eq("REG") & nfl["ats_margin"].notna() & nfl["season"].lt(2025)]
    return float(nfl.groupby("season").size().mean()), int(nfl["season"].nunique())


def subsample(train, quota, rng):
    parts = []
    for _s, g in train.groupby("season"):
        k = min(quota, len(g))
        parts.append(g.iloc[np.sort(rng.choice(len(g), size=k, replace=False))])
    return pd.concat(parts)


def walk_matched(frame, cols, test_mask, min_train, quota, seed):
    rng = np.random.default_rng(seed)
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    rows = []
    test = frame.loc[test_mask(frame)]
    for (_season, _week), wk in test.groupby(["season", "week"], sort=True):
        cutoff = wk["gameday"].min()
        train = frame.loc[frame["gameday"].lt(cutoff) & frame["ats_margin"].notna()]
        if len(train) < min_train:
            continue
        train = subsample(train, quota, rng)
        for alpha in base.ALPHAS:
            pred, sigma = base.fit_predict(
                train[cols].to_numpy(float),
                train["ats_margin"].to_numpy(float),
                wk[cols].to_numpy(float),
                alpha,
            )
            out = wk[["game_id", "season", "week", "spread_line", "open", "result"]].copy()
            out["alpha"] = alpha
            out["pred_margin"] = wk["spread_line"].to_numpy(float) + pred
            out["sigma"] = sigma
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


def main():
    ref = json.loads((OUT / "report.json").read_text())
    quota_f, nfl_seasons = nfl_rows_per_season()
    quota = round(quota_f)
    cfb = pd.read_parquet(base.CFB_FEATURES)
    cfb["gameday"] = pd.to_datetime(cfb["gameday"])
    cfb["open"] = cfb["spread_open"]
    ccols = list(CFB_MODEL_FEATURE_COLUMNS)
    mask = lambda f: (  # noqa: E731
        f["season"].between(2013, 2025)
        & f["season"].ne(2020)
        & f["open"].notna()
        & f["result"].notna()
    )
    seeds = {}
    for sd in SUB_SEEDS:
        g = base.grade(walk_matched(cfb, ccols, mask, base.CFB_MIN_TRAIN, quota, sd))
        seeds[str(sd)] = base.league("cfb", g)
        print(sd, flush=True)
    nfl = ref["nfl"]
    arms = [a for a in base.ALPHAS if a != base.SERVED_ALPHA]
    summ = {}
    for a in arms:
        k = str(a)
        row = {}
        for m in ("accuracy", "log_loss"):
            key = "acc_vs_served" if m == "accuracy" else "ll_vs_served"
            eff = np.array([seeds[s][k][key]["effect"] for s in seeds])
            se = np.array([seeds[s][k][key]["se"] for s in seeds])
            pp = np.array([seeds[s][k][key]["p_pos"] for s in seeds])
            lo = np.array([seeds[s][k][key]["low"] for s in seeds])
            hi = np.array([seeds[s][k][key]["high"] for s in seeds])
            ne = nfl[k][key]
            se_m = float(np.sqrt(np.mean(se**2)))
            z = float((eff.mean() - ne["effect"]) / np.sqrt(se_m**2 + ne["se"] ** 2))
            row[m] = {
                "effect": float(eff.mean()),
                "se": se_m,
                "low": float(lo.mean()),
                "high": float(hi.mean()),
                "p_pos": float(pp.mean()),
                "seed_sd": float(eff.std(ddof=1)),
                "heterogeneity_z": z,
                "nfl_effect": ne["effect"],
                "nfl_se": ne["se"],
                "multiplier": float((1 / ne["se"] ** 2 + 1 / se_m**2) / (1 / ne["se"] ** 2)),
                "games": seeds[next(iter(seeds))][k][key]["games"],
                "blocks": seeds[next(iter(seeds))][k][key]["blocks"],
            }
        summ[k] = row
    acc_n = [nfl[str(a)]["accuracy"] for a in base.ALPHAS]
    ll_n = [nfl[str(a)]["log_loss"] for a in base.ALPHAS]
    rho_acc, rho_ll = [], []
    for s in seeds:
        rho_acc.append(
            float(spearmanr(acc_n, [seeds[s][str(a)]["accuracy"] for a in base.ALPHAS]).statistic)
        )
        rho_ll.append(
            float(spearmanr(ll_n, [seeds[s][str(a)]["log_loss"] for a in base.ALPHAS]).statistic)
        )
    rep = {
        "quota_rows_per_season": quota,
        "quota_unrounded": quota_f,
        "nfl_train_seasons_mean_basis": nfl_seasons,
        "seeds": list(SUB_SEEDS),
        "per_seed_accuracy": {
            s: [seeds[s][str(a)]["accuracy"] for a in base.ALPHAS] for s in seeds
        },
        "spearman_acc_per_seed": rho_acc,
        "spearman_ll_per_seed": rho_ll,
        "summary": summ,
        "full_cfb_spearman_acc": ref["spearman_acc_alpha_nfl_vs_cfb"],
    }
    (OUT / "size_matched_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
