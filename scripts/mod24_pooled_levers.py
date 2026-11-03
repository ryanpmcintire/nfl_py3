import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mod24_more_games as base  # noqa: E402

from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS  # noqa: E402
from nfl_ats.clv import _ACTIVE_MODEL_FALLBACK_CONFIG  # noqa: E402
from nfl_ats.margin import margin_feature_columns  # noqa: E402

OPENER_REGIME = "sbr_multibook"
ALPHA = base.SERVED_ALPHA
CV_GRID = tuple(10.0 ** (-1 + 0.5 * i) for i in range(9))
CV_FOLDS = 5
STRATA = 3
OUT = ROOT / "artifacts" / "mod24_pooled_levers"
Z_CRIT = 1.959964


def prep(train_x, test_x):
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    return sc.fit_transform(imp.fit_transform(train_x)), sc.transform(imp.transform(test_x))


def tertile_cuts(values):
    return np.nanquantile(values, [k / STRATA for k in range(1, STRATA)])


def stratum(values, cuts):
    return np.searchsorted(cuts, values, side="right")


def ts_weights(train, resid):
    tot_cuts = tertile_cuts(train["total_line"].to_numpy(float))
    spr_cuts = tertile_cuts(np.abs(train["spread_line"].to_numpy(float)))
    t = stratum(train["total_line"].to_numpy(float), tot_cuts)
    s = stratum(np.abs(train["spread_line"].to_numpy(float)), spr_cuts)
    ktr = t * STRATA + s
    s2 = {k: float(np.var(resid[ktr == k], ddof=1)) for k in np.unique(ktr)}
    w = np.array([1.0 / s2[k] for k in ktr])
    return w / w.mean()


def predict_arms(train, test, cols, arms):
    y = train["ats_margin"].to_numpy(float)
    a, b = prep(train[cols].to_numpy(float), test[cols].to_numpy(float))
    res = {}
    m0 = Ridge(alpha=ALPHA).fit(a, y)
    r0 = y - m0.predict(a)
    res["base"] = (m0.predict(b), float(np.std(r0, ddof=1)))
    if "ts" in arms:
        w = ts_weights(train, r0)
        m = Ridge(alpha=ALPHA).fit(a, y, sample_weight=w)
        res["ts"] = (m.predict(b), float(np.std(y - m.predict(a), ddof=1)))
    if "cv" in arms:
        m = RidgeCV(alphas=CV_GRID, cv=KFold(CV_FOLDS)).fit(a, y)
        res["cv"] = (m.predict(b), float(np.std(y - m.predict(a), ddof=1)))
    return res


def walk(frame, cols, test_mask, min_train, arms, post_train=False):
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    test = frame.loc[test_mask(frame)]
    rows = []
    for (_s, _w), wk in test.groupby(["season", "week"], sort=True):
        cutoff = wk["gameday"].min()
        past = frame.loc[frame["gameday"].lt(cutoff) & frame["ats_margin"].notna()]
        reg = past.loc[past["game_type"].eq("REG")]
        if len(reg) < min_train:
            continue
        out = predict_arms(reg, wk, cols, arms)
        if post_train:
            out["post"] = predict_arms(past, wk, cols, ())["base"]
        for name, (pred, sigma) in out.items():
            o = wk[["game_id", "season", "week", "spread_line", "open", "result"]].copy()
            o["arm"] = name
            o["pred_margin"] = wk["spread_line"].to_numpy(float) + pred
            o["sigma"] = sigma
            rows.append(o)
    return pd.concat(rows, ignore_index=True)


def grade(df):
    df = df.assign(edge=df["pred_margin"] - df["open"], cm=df["result"] - df["open"])
    df = df.loc[df["cm"].ne(0)].copy()
    df["correct"] = (np.sign(df["edge"]) == np.sign(df["cm"])).astype(float)
    p = np.clip(norm.cdf(df["edge"] / df["sigma"]), 1e-6, 1 - 1e-6)
    y = (df["cm"] > 0).astype(float)
    df["ll"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    df["br"] = (p - y) ** 2
    return df


def contrast(g, arm, rng):
    b = g.loc[g["arm"].eq("base")].set_index("game_id")
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
        "base": {
            "wins": int(b.loc[idx, "correct"].sum()),
            "n": len(idx),
            "acc": float(100 * b.loc[idx, "correct"].mean()),
            "ll": float(b.loc[idx, "ll"].mean()),
            "br": float(b.loc[idx, "br"].mean()),
        },
        "arm": {
            "wins": int(a.loc[idx, "correct"].sum()),
            "acc": float(100 * a.loc[idx, "correct"].mean()),
            "ll": float(a.loc[idx, "ll"].mean()),
            "br": float(a.loc[idx, "br"].mean()),
        },
        "flips_win": int((flips > 0).sum()),
        "flips_loss": int((flips < 0).sum()),
        "accuracy": base.block_boot(p, "d_acc", rng),
        "log_loss": base.block_boot(p, "d_ll", rng),
        "brier": base.block_boot(p, "d_br", rng),
    }


def pool(n, c):
    out = {}
    for m in ("accuracy", "log_loss", "brier"):
        en, ec = n[m], c[m]
        z = (ec["effect"] - en["effect"]) / np.sqrt(ec["se"] ** 2 + en["se"] ** 2)
        wn, wc = 1 / en["se"] ** 2, 1 / ec["se"] ** 2
        eff = (wn * en["effect"] + wc * ec["effect"]) / (wn + wc)
        se = float(1 / np.sqrt(wn + wc))
        out[m] = {
            "het_z": float(z),
            "admissible": bool(abs(z) < Z_CRIT),
            "effect": float(eff),
            "se": se,
            "low": float(eff - Z_CRIT * se),
            "high": float(eff + Z_CRIT * se),
            "p_pos": float(norm.cdf(eff / se)),
            "multiplier": float((wn + wc) / wn),
        }
    return out


def main():
    nfl_cols = list(
        margin_feature_columns("market_residual", _ACTIVE_MODEL_FALLBACK_CONFIG["feature_profile"])
    )
    common = [c for c in CFB_MODEL_FEATURE_COLUMNS if c in nfl_cols]
    nfl = pd.read_parquet(base.NFL_FEATURES)
    nfl["gameday"] = pd.to_datetime(nfl["gameday"])
    pg = pd.read_parquet(base.PER_GAME)[["game_id", "tue_open_home_spread"]]
    nfl = nfl.merge(pg, on="game_id", how="left").rename(columns={"tue_open_home_spread": "open"})
    ng = grade(
        walk(
            nfl,
            common,
            lambda f: (
                f["game_type"].eq("REG")
                & f["season"].between(2020, 2025)
                & f["open"].notna()
                & f["result"].notna()
            ),
            base.NFL_MIN_TRAIN,
            ("ts", "cv"),
            post_train=True,
        )
    )
    cfb = pd.read_parquet(base.CFB_FEATURES)
    cfb["gameday"] = pd.to_datetime(cfb["gameday"])
    cfb["open"] = cfb["spread_open"]
    cfb["game_type"] = "REG"
    cg = grade(
        walk(
            cfb,
            common,
            lambda f: (
                f["season"].between(2013, 2025)
                & f["season"].ne(2020)
                & f["open"].notna()
                & f["result"].notna()
                & f["source_regime"].eq(OPENER_REGIME)
            ),
            base.CFB_MIN_TRAIN,
            ("ts", "cv"),
        )
    )
    rng = np.random.default_rng(base.SEED)
    report = {
        "common_features": common,
        "postseason_rows_nfl": int(nfl["game_type"].ne("REG").sum()),
    }
    for lever, arm in (("a_postseason", "post"), ("b_ts_weights", "ts"), ("c_ridgecv", "cv")):
        entry = {"nfl": contrast(ng, arm, rng)}
        if arm != "post":
            entry["cfb"] = contrast(cg, arm, rng)
            entry["pooled"] = pool(entry["nfl"], entry["cfb"])
        report[lever] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
