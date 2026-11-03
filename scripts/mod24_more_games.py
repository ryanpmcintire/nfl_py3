import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS  # noqa: E402
from nfl_ats.clv import _ACTIVE_MODEL_FALLBACK_CONFIG  # noqa: E402
from nfl_ats.margin import margin_feature_columns  # noqa: E402

PER_GAME = ROOT / "artifacts" / "opener_evaluation" / "20261101T151614Z" / "per_game.parquet"
NFL_FEATURES = ROOT / "data" / "processed" / "game_features_weak_stack.parquet"
CFB_FEATURES = ROOT / "data" / "processed" / "cfb_game_features.parquet"
OUT = ROOT / "artifacts" / "mod24_more_games"
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
SERVED_ALPHA = 10.0
CFB_MIN_TRAIN = 500
NFL_MIN_TRAIN = 1000
DRAWS = 2000
SEED = 20261103


def fit_predict(train_x, train_y, test_x, alpha):
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    a = sc.fit_transform(imp.fit_transform(train_x))
    b = sc.transform(imp.transform(test_x))
    model = Ridge(alpha=alpha).fit(a, train_y)
    resid = train_y - model.predict(a)
    return model.predict(b), float(np.std(resid, ddof=1))


def walk(frame, cols, test_mask, min_train):
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    rows = []
    test = frame.loc[test_mask(frame)]
    for (_season, _week), wk in test.groupby(["season", "week"], sort=True):
        cutoff = wk["gameday"].min()
        train = frame.loc[frame["gameday"].lt(cutoff) & frame["ats_margin"].notna()]
        if len(train) < min_train:
            continue
        for alpha in ALPHAS:
            pred, sigma = fit_predict(
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


def grade(df):
    edge = df["pred_margin"] - df["open"]
    cover_margin = df["result"] - df["open"]
    df = df.assign(edge=edge, cover_margin=cover_margin)
    df = df.loc[df["cover_margin"].ne(0)].copy()
    df["correct"] = (np.sign(df["edge"]) == np.sign(df["cover_margin"])).astype(float)
    p = norm.cdf(df["edge"] / df["sigma"])
    y = (df["cover_margin"] > 0).astype(float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    df["ll"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return df


def paired(df, alpha):
    base = df.loc[df["alpha"].eq(SERVED_ALPHA)].set_index("game_id")
    arm = df.loc[df["alpha"].eq(alpha)].set_index("game_id")
    idx = base.index.intersection(arm.index)
    out = pd.DataFrame(
        {
            "block": base.loc[idx, "season"].astype(str) + "-" + base.loc[idx, "week"].astype(str),
            "d_acc": 100 * (arm.loc[idx, "correct"] - base.loc[idx, "correct"]),
            "d_ll": base.loc[idx, "ll"] - arm.loc[idx, "ll"],
        }
    )
    return out


def block_boot(p, col, rng):
    blocks = p.groupby("block")[col].agg(["sum", "count"])
    s = blocks["sum"].to_numpy()
    c = blocks["count"].to_numpy()
    n = len(s)
    idx = rng.integers(0, n, size=(DRAWS, n))
    est = s[idx].sum(1) / c[idx].sum(1)
    point = s.sum() / c.sum()
    se = float(np.std(est, ddof=1))
    lo, hi = np.percentile(est, [2.5, 97.5])
    return {
        "effect": float(point),
        "low": float(lo),
        "high": float(hi),
        "se": se,
        "p_pos": float((est > 0).mean()),
        "games": int(c.sum()),
        "blocks": int(n),
    }


def league(name, graded):
    rng = np.random.default_rng(SEED)
    res = {}
    for a in ALPHAS:
        sub = graded.loc[graded["alpha"].eq(a)]
        res[str(a)] = {
            "accuracy": float(100 * sub["correct"].mean()),
            "log_loss": float(sub["ll"].mean()),
            "wins": int(sub["correct"].sum()),
            "games": len(sub),
        }
        if a != SERVED_ALPHA:
            p = paired(graded, a)
            res[str(a)]["acc_vs_served"] = block_boot(p, "d_acc", rng)
            res[str(a)]["ll_vs_served"] = block_boot(p, "d_ll", rng)
    return res


def main():
    cols = list(
        margin_feature_columns("market_residual", _ACTIVE_MODEL_FALLBACK_CONFIG["feature_profile"])
    )
    nfl = pd.read_parquet(NFL_FEATURES)
    missing = [c for c in cols if c not in nfl.columns]
    if missing:
        raise SystemExit(f"nfl missing {missing[:5]} of {len(missing)}")
    nfl = nfl.loc[nfl["game_type"].eq("REG")].copy()
    nfl["gameday"] = pd.to_datetime(nfl["gameday"])
    pg = pd.read_parquet(PER_GAME)[["game_id", "tue_open_home_spread"]]
    nfl = nfl.merge(pg, on="game_id", how="left").rename(columns={"tue_open_home_spread": "open"})
    nfl_pred = walk(
        nfl,
        cols,
        lambda f: f["season"].between(2020, 2025) & f["open"].notna() & f["result"].notna(),
        NFL_MIN_TRAIN,
    )
    nfl_g = grade(nfl_pred)

    cfb = pd.read_parquet(CFB_FEATURES)
    cfb["gameday"] = pd.to_datetime(cfb["gameday"])
    cfb["open"] = cfb["spread_open"]
    ccols = list(CFB_MODEL_FEATURE_COLUMNS)
    cfb_pred = walk(
        cfb,
        ccols,
        lambda f: (
            f["season"].between(2013, 2025)
            & f["season"].ne(2020)
            & f["open"].notna()
            & f["result"].notna()
        ),
        CFB_MIN_TRAIN,
    )
    cfb_g = grade(cfb_pred)

    r_nfl = league("nfl", nfl_g)
    r_cfb = league("cfb", cfb_g)
    ll_n = [r_nfl[str(a)]["log_loss"] for a in ALPHAS]
    ll_c = [r_cfb[str(a)]["log_loss"] for a in ALPHAS]
    acc_n = [r_nfl[str(a)]["accuracy"] for a in ALPHAS]
    acc_c = [r_cfb[str(a)]["accuracy"] for a in ALPHAS]
    per_season = {}
    for s, sub in nfl_g.groupby("season"):
        per_season[int(s)] = [float(sub.loc[sub["alpha"].eq(a), "ll"].mean()) for a in ALPHAS]
    rho_season = {s: float(spearmanr(v, ll_c).statistic) for s, v in per_season.items()}
    mult = {}
    for a in ALPHAS:
        if a == SERVED_ALPHA:
            continue
        sn = r_nfl[str(a)]["acc_vs_served"]["se"]
        sc = r_cfb[str(a)]["acc_vs_served"]["se"]
        mult[str(a)] = float((1 / sn**2 + 1 / sc**2) / (1 / sn**2))
    report = {
        "nfl": r_nfl,
        "cfb": r_cfb,
        "spearman_ll_alpha_nfl_vs_cfb": float(spearmanr(ll_n, ll_c).statistic),
        "spearman_acc_alpha_nfl_vs_cfb": float(spearmanr(acc_n, acc_c).statistic),
        "spearman_ll_nfl_season_vs_cfb": rho_season,
        "info_multiplier_if_poolable": mult,
        "nfl_games": int(nfl_g["game_id"].nunique()),
        "cfb_games": int(cfb_g["game_id"].nunique()),
        "cfb_seasons": sorted(int(s) for s in cfb_g["season"].unique()),
        "nfl_features": len(cols),
        "cfb_features": len(ccols),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
