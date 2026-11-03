import argparse
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mod24_more_games as base  # noqa: E402
import mod24_pooled_levers as lev  # noqa: E402

from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS  # noqa: E402
from nfl_ats.clv import _ACTIVE_MODEL_FALLBACK_CONFIG  # noqa: E402
from nfl_ats.margin import margin_feature_columns  # noqa: E402

SBR = ROOT / "data" / "processed" / "sbr_odds.parquet"
OUT = ROOT / "artifacts" / "pooled_screen"
NFL_OPENER_SEASONS = (2020, 2025)
NFL_PROXY_SEASONS = (2011, 2019)
CFB_SEASONS = (2013, 2025)
CFB_EXCLUDED_SEASON = 2020
METRICS = (("accuracy", "acc pts"), ("log_loss", "ll gain"), ("brier", "brier gain"))


def arm_ts(a, y, b, train):
    r0 = y - Ridge(alpha=lev.ALPHA).fit(a, y).predict(a)
    w = lev.ts_weights(train, r0)
    m = Ridge(alpha=lev.ALPHA).fit(a, y, sample_weight=w)
    return m.predict(b), float(np.std(y - m.predict(a), ddof=1))


def arm_cv(a, y, b, train):
    m = RidgeCV(alphas=lev.CV_GRID, cv=KFold(lev.CV_FOLDS)).fit(a, y)
    return m.predict(b), float(np.std(y - m.predict(a), ddof=1))


NAMED = {"ts": arm_ts, "cv": arm_cv}


def resolve(spec):
    if spec in NAMED:
        return NAMED[spec]
    target, _, attr = spec.rpartition(":")
    if not target:
        raise SystemExit(
            f"variant {spec!r} is not a named arm {sorted(NAMED)} or module_or_path:callable"
        )
    if target.endswith(".py"):
        mspec = importlib.util.spec_from_file_location(Path(target).stem, target)
        mod = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    return getattr(mod, attr)


def install(variant):
    original = lev.predict_arms

    def patched(train, test, cols, arms):
        out = original(train, test, cols, ())
        y = train["ats_margin"].to_numpy(float)
        a, b = lev.prep(train[cols].to_numpy(float), test[cols].to_numpy(float))
        out["variant"] = variant(a, y, b, train)
        return out

    lev.predict_arms = patched


def shared_columns():
    nfl_cols = list(
        margin_feature_columns("market_residual", _ACTIVE_MODEL_FALLBACK_CONFIG["feature_profile"])
    )
    return [c for c in CFB_MODEL_FEATURE_COLUMNS if c in nfl_cols]


def nfl_graded(cols):
    nfl = pd.read_parquet(base.NFL_FEATURES)
    nfl["gameday"] = pd.to_datetime(nfl["gameday"])
    pg = pd.read_parquet(base.PER_GAME)[["game_id", "tue_open_home_spread"]]
    sbr = pd.read_parquet(SBR).dropna(subset=["game_id", "open_home_spread"])
    sbr = sbr.drop_duplicates("game_id")[["game_id", "open_home_spread"]]
    nfl = nfl.merge(pg, on="game_id", how="left").merge(sbr, on="game_id", how="left")
    lo_o, hi_o = NFL_OPENER_SEASONS
    lo_p, hi_p = NFL_PROXY_SEASONS
    nfl["open"] = np.where(
        nfl["season"].between(lo_o, hi_o), nfl["tue_open_home_spread"], nfl["open_home_spread"]
    )
    return lev.grade(
        lev.walk(
            nfl,
            cols,
            lambda f: (
                f["game_type"].eq("REG")
                & (f["season"].between(lo_o, hi_o) | f["season"].between(lo_p, hi_p))
                & f["open"].notna()
                & f["result"].notna()
            ),
            base.NFL_MIN_TRAIN,
            (),
        )
    )


def cfb_graded(cols):
    cfb = pd.read_parquet(base.CFB_FEATURES)
    cfb["gameday"] = pd.to_datetime(cfb["gameday"])
    cfb["open"] = cfb["spread_open"]
    cfb["game_type"] = "REG"
    lo, hi = CFB_SEASONS
    return lev.grade(
        lev.walk(
            cfb,
            cols,
            lambda f: (
                f["season"].between(lo, hi)
                & f["season"].ne(CFB_EXCLUDED_SEASON)
                & f["open"].notna()
                & f["result"].notna()
                & f["source_regime"].eq(lev.OPENER_REGIME)
            ),
            base.CFB_MIN_TRAIN,
            (),
        )
    )


def contrast_seeded(g):
    return lev.contrast(g, "variant", np.random.default_rng(base.SEED))


def fmt(e):
    return f"{e['effect']:+.5f} [{e['low']:+.5f},{e['high']:+.5f}] P+ {e['p_pos']:.2f}"


def fmt_acc(e):
    return f"{e['effect']:+.2f} [{e['low']:+.2f},{e['high']:+.2f}] P+ {e['p_pos']:.2f}"


def show(label, c):
    b, a = c["base"], c["arm"]
    print(
        f"{label:10s} n={b['n']} base {b['wins']}-{b['n'] - b['wins']} "
        f"variant {a['wins']}-{b['n'] - a['wins']} flips W-L {c['flips_win']}-{c['flips_loss']}"
    )
    print(f"           acc pts  {fmt_acc(c['accuracy'])}")
    print(f"           ll gain  {fmt(c['log_loss'])}")
    print(f"           brier    {fmt(c['brier'])}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Screen a training or estimator variant against ridge alpha 10 on the 14 features "
            "shared by the NFL and CFB harnesses, graded at the opener, with per-metric "
            "heterogeneity and a pooled inverse-variance effect."
        )
    )
    ap.add_argument(
        "variant",
        help=(
            "named arm (ts, cv) or MODULE_OR_PATH.py:CALLABLE; callable(a, y, b, train) returns "
            "(test prediction of ats_margin, residual sd); a and b are imputed, standardized "
            "train and test shared-feature matrices, y is the train ats_margin, train is the "
            "train frame"
        ),
    )
    ap.add_argument(
        "--nfl-eras",
        choices=("opener", "all"),
        default="opener",
        help=(
            "NFL population entering the pooled value: opener = 2020-25 Tuesday opener; "
            "all = adds the 2011-19 SBR-open proxy era"
        ),
    )
    ap.add_argument("--out", default=None, help="JSON report path")
    args = ap.parse_args()

    t0 = time.time()
    install(resolve(args.variant))
    cols = shared_columns()
    ng = nfl_graded(cols)
    cg = cfb_graded(cols)
    lo_o, hi_o = NFL_OPENER_SEASONS
    lo_p, hi_p = NFL_PROXY_SEASONS
    pops = {
        "nfl_opener": ng.loc[ng["season"].between(lo_o, hi_o)],
        "nfl_proxy": ng.loc[ng["season"].between(lo_p, hi_p)],
        "nfl_all": ng,
        "cfb": cg,
    }
    res = {k: contrast_seeded(v) for k, v in pops.items() if len(v)}
    nfl_key = "nfl_opener" if args.nfl_eras == "opener" else "nfl_all"
    pooled = lev.pool(res[nfl_key], res["cfb"])
    report = {
        "variant": args.variant,
        "nfl_eras": args.nfl_eras,
        "shared_features": cols,
        "populations": res,
        "pooled": pooled,
        "seconds": time.time() - t0,
    }
    print(f"variant {args.variant}  shared features {len(cols)}  base ridge alpha {lev.ALPHA}")
    for k, c in res.items():
        show(k, c)
    print(f"pooled ({nfl_key} + cfb opener-matched), inverse variance")
    for m, name in METRICS:
        p = pooled[m]
        w = "" if p["admissible"] else "  NOT ADMISSIBLE"
        d = 2 if m == "accuracy" else 5
        print(
            f"  {name:10s} het z {p['het_z']:+.2f}{w}  pooled {p['effect']:+.{d}f} "
            f"[{p['low']:+.{d}f},{p['high']:+.{d}f}] P+ {p['p_pos']:.2f} mult {p['multiplier']:.1f}"
        )
    out = Path(args.out) if args.out else OUT / f"{Path(args.variant.split(':')[-1]).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"report {out}  seconds {report['seconds']:.0f}")


if __name__ == "__main__":
    main()
