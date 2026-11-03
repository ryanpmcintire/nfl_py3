import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, "F:/Repos/nfl_py3/scripts")
sys.path.insert(0, "F:/Repos/nfl_py3/src")
import mod24_more_games as base
import mod24_pooled_levers as lev
import pooled_screen as ps
import pooled_sweep_variants as sv

ARMS = {f"a{int(a)}": a for a in (1.0, 100.0, 1000.0, 10000.0)}


def mk(al):
    def f(a, y, b, train):
        m = Ridge(alpha=al).fit(a, y)
        return m.predict(b), float(np.std(y - m.predict(a), ddof=1))

    return f


FUN = {k: mk(v) for k, v in ARMS.items()}
FUN["cv"] = ps.arm_cv
FUN["slope"] = sv.slope


def install_all():
    original = lev.predict_arms

    def patched(train, test, cols, arms):
        out = original(train, test, cols, ())
        y = train["ats_margin"].to_numpy(float)
        a, b = lev.prep(train[cols].to_numpy(float), test[cols].to_numpy(float))
        for k, f in FUN.items():
            out[k] = f(a, y, b, train)
        return out

    lev.predict_arms = patched


install_all()
cols = ps.shared_columns()
ng = ps.nfl_graded(cols)
ng = ng.loc[ng["season"].between(2020, 2025)]
cfb = pd.read_parquet(base.CFB_FEATURES)
cfb["gameday"] = pd.to_datetime(cfb["gameday"])
cfb["open"] = cfb["spread_open"]
cfb["game_type"] = "REG"
cg = lev.grade(
    lev.walk(
        cfb,
        cols,
        lambda f: (
            f["season"].between(2023, 2025)
            & f["open"].notna()
            & f["result"].notna()
            & f["source_regime"].eq("espn_book_era")
        ),
        base.CFB_MIN_TRAIN,
        (),
    )
)


def accs(g):
    return {k: float(100 * v["correct"].mean()) for k, v in g.groupby("arm")}, int(
        g.loc[g.arm.eq("base")].shape[0]
    )


an, nn = accs(ng)
ac, nc = accs(cg)
res = {"n_nfl": nn, "n_cfb": nc, "acc_nfl": an, "acc_cfb": ac, "arms": {}}
for k in FUN:
    cn = lev.contrast(ng, k, np.random.default_rng(base.SEED))
    cc = lev.contrast(cg, k, np.random.default_rng(base.SEED))
    p = lev.pool(cn, cc)
    res["arms"][k] = {
        "nfl": {m: cn[m] for m in ("accuracy", "log_loss")},
        "cfb": {m: cc[m] for m in ("accuracy", "log_loss")},
        "het": {m: p[m]["het_z"] for m in p},
    }
names = ["base", *list(FUN)]
res["rho_abs_acc"] = float(spearmanr([an[k] for k in names], [ac[k] for k in names])[0])
res["rho_gain_acc"] = float(
    spearmanr([an[k] - an["base"] for k in FUN], [ac[k] - ac["base"] for k in FUN])[0]
)
res["rho_gain_ll"] = float(
    spearmanr(
        [res["arms"][k]["nfl"]["log_loss"]["effect"] for k in FUN],
        [res["arms"][k]["cfb"]["log_loss"]["effect"] for k in FUN],
    )[0]
)
Path("F:/Repos/nfl_py3/artifacts/pooled_screen/mod24_era_check.json").write_text(
    json.dumps(res, indent=1)
)
print(nn, nc, "rho abs", res["rho_abs_acc"], "gain", res["rho_gain_acc"], "ll", res["rho_gain_ll"])
print("base", round(an["base"], 2), round(ac["base"], 2))
for k in FUN:
    r = res["arms"][k]

    def f(e):
        return f"{e['effect']:+.2f}[{e['low']:+.2f},{e['high']:+.2f}]P{e['p_pos']:.2f}"

    print(
        k,
        "NFL",
        f(r["nfl"]["accuracy"]),
        "CFB",
        f(r["cfb"]["accuracy"]),
        "hetz acc {:+.2f} ll {:+.2f}".format(r["het"]["accuracy"], r["het"]["log_loss"]),
        "LLcfb {:+.5f} P{:.2f}".format(
            r["cfb"]["log_loss"]["effect"], r["cfb"]["log_loss"]["p_pos"]
        ),
    )
