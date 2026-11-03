import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mod24_more_games as base  # noqa: E402
import mod24_pooled_levers as lev  # noqa: E402
import pooled_screen as ps  # noqa: E402
import pooled_sweep_variants as sv  # noqa: E402

OUT = ROOT / "artifacts" / "xlg12_cfb_forward" / "v2"
LAST_TRAIN_SEASON = 2025


def training_frame(cols):
    cfb = pd.read_parquet(base.CFB_FEATURES)
    cfb["gameday"] = pd.to_datetime(cfb["gameday"])
    keep = cfb["season"].le(LAST_TRAIN_SEASON) & cfb["ats_margin"].notna()
    return cfb.loc[keep].sort_values(["gameday", "game_id"]).reset_index(drop=True)


def main():
    cols = ps.shared_columns()
    tr = training_frame(cols)
    x = tr[cols].to_numpy(float)
    y = tr["ats_margin"].to_numpy(float)
    imp = SimpleImputer(strategy="median").fit(x)
    sc = StandardScaler().fit(imp.transform(x))
    a = sc.transform(imp.transform(x))
    created = datetime.now(UTC).isoformat()
    ridge = Ridge(alpha=sv.ALPHA).fit(a, y)
    cv = RidgeCV(alphas=lev.CV_GRID, cv=KFold(lev.CV_FOLDS)).fit(a, y)
    oof = np.zeros(len(y))
    for tri, tei in sv.folds(len(y)):
        oof[tei] = Ridge(alpha=sv.ALPHA).fit(a[tri], y[tri]).predict(a[tei])
    k = float(oof @ y / (oof @ oof))
    fits = {
        "base": (ridge, 1.0, sv.ALPHA),
        "cv": (cv, 1.0, float(cv.alpha_)),
        "slope": (ridge, k, sv.ALPHA),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (m, scale, alpha) in fits.items():
        doc = {
            "variant": name,
            "created_at": created,
            "train_seasons": [int(tr["season"].min()), LAST_TRAIN_SEASON],
            "train_seasons_present": sorted(int(s) for s in tr["season"].unique()),
            "train_rows": len(tr),
            "train_regime": "all_regimes_walk_forward_as_mod24_era_check",
            "version": "v2",
            "features": cols,
            "impute_median": imp.statistics_.tolist(),
            "scaler_mean": sc.mean_.tolist(),
            "scaler_scale": sc.scale_.tolist(),
            "ridge_alpha": alpha,
            "intercept": float(m.intercept_),
            "coef": m.coef_.tolist(),
            "prediction_scale": scale,
            "residual_sd": float(np.std(y - m.predict(a), ddof=1)),
            "target": "ats_margin = result - spread_line",
            "source_sweep": "scripts/mod24_pooled_era_check.py training rule",
        }
        (OUT / f"frozen_{name}.json").write_text(json.dumps(doc, indent=1))
        print(name, created, doc["train_seasons_present"], doc["train_rows"], alpha, scale)


if __name__ == "__main__":
    main()
