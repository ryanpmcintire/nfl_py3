import json
import sys
import time
from pathlib import Path

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

STRENGTHS = tuple(10.0**k for k in range(0, 10))
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
OUT = ROOT / "artifacts" / "xlg11_cfb_centered"
SERVED = ROOT / "artifacts" / "opener_evaluation" / "20261101T151614Z" / "per_game.parquet"
LOG = []
STATE = {}


def inner_scores(a, y, seasons, p0, grid_lam, grid_s):
    sse = np.zeros((len(grid_lam), len(grid_s)))
    p = a @ p0
    for s in np.unique(seasons):
        tr, te = seasons != s, seasons == s
        for i, lam in enumerate(grid_lam):
            m = Ridge(alpha=lam).fit(a[tr], np.column_stack([y[tr], p[tr]]))
            pr = m.predict(a[te])
            for j, sc in enumerate(grid_s):
                pred = sc * p[te] + pr[:, 0] - sc * pr[:, 1]
                sse[i, j] += float(np.sum((y[te] - pred) ** 2))
    return sse


def arms(train, test, cols, _arms):
    cfb = STATE["cfb"]
    cutoff = train["gameday"].max()
    past = cfb.loc[cfb["gameday"].lt(cutoff)]
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    a = sc.fit_transform(imp.fit_transform(train[cols].to_numpy(float)))
    b = sc.transform(imp.transform(test[cols].to_numpy(float)))
    y = train["ats_margin"].to_numpy(float)
    ca = sc.transform(imp.transform(past[cols].to_numpy(float)))
    cy = past["ats_margin"].to_numpy(float)
    ratio = float(np.std(y, ddof=1) / np.std(cy, ddof=1))
    beta = Ridge(alpha=lev.ALPHA).fit(ca, cy).coef_
    seasons = train["season"].to_numpy()
    m0 = Ridge(alpha=lev.ALPHA).fit(a, y)
    out = {"base": (m0.predict(b), float(np.std(y - m0.predict(a), ddof=1)))}
    sse = inner_scores(a, y, seasons, beta, STRENGTHS, (ratio,))
    lam1 = STRENGTHS[int(np.argmin(sse[:, 0]))]
    prior = ratio * beta
    m = Ridge(alpha=lam1).fit(a, y - a @ prior)
    out["u1"] = (b @ prior + m.predict(b), float(np.std(y - a @ prior - m.predict(a), ddof=1)))
    grid_s = tuple(ratio * s for s in SCALES)
    sse2 = inner_scores(a, y, seasons, beta, STRENGTHS, grid_s)
    i, j = np.unravel_index(int(np.argmin(sse2)), sse2.shape)
    lam2, s2 = STRENGTHS[i], grid_s[j]
    prior2 = s2 * beta
    m2 = Ridge(alpha=lam2).fit(a, y - a @ prior2)
    out["free"] = (
        b @ prior2 + m2.predict(b),
        float(np.std(y - a @ prior2 - m2.predict(a), ddof=1)),
    )
    LOG.append(
        {
            "season": int(test["season"].iloc[0]),
            "lam1": lam1,
            "lam2": lam2,
            "scale_mult": SCALES[j],
            "ratio": ratio,
            "cfb_max_gameday": str(past["gameday"].max().date()),
            "train_max_gameday": str(cutoff.date()),
            "test_min_gameday": str(test["gameday"].min().date()),
        }
    )
    return out


def served_h2h(g):
    sv = pd.read_parquet(SERVED)
    sv = sv.loc[sv["margin_vs_open"].ne(0)].set_index("game_id")
    y = (sv["margin_vs_open"] > 0).astype(float)
    p = np.clip(sv["home_cover_probability_excluding_push_at_open"], 1e-6, 1 - 1e-6)
    sv_ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    sv_br = (p - y) ** 2
    res = {}
    rng = np.random.default_rng(base.SEED)
    for arm in ("base", "u1", "free"):
        a = g.loc[g["arm"].eq(arm)].set_index("game_id")
        idx = a.index.intersection(sv.index)
        sc = sv.loc[idx, "correct_at_open"].astype(float)
        blk = a.loc[idx, "season"].astype(str) + "-" + a.loc[idx, "week"].astype(str)
        d = pd.DataFrame(
            {
                "block": blk,
                "d_acc": 100 * (a.loc[idx, "correct"] - sc),
                "d_ll": sv_ll[idx] - a.loc[idx, "ll"],
                "d_br": sv_br[idx] - a.loc[idx, "br"],
            }
        )
        fl = a.loc[idx, "correct"] - sc
        res[arm] = {
            "n": len(idx),
            "served_wins": int(sc.sum()),
            "arm_wins": int(a.loc[idx, "correct"].sum()),
            "flips_win": int((fl > 0).sum()),
            "flips_loss": int((fl < 0).sum()),
            "served_ll": float(sv_ll[idx].mean()),
            "arm_ll": float(a.loc[idx, "ll"].mean()),
            "served_br": float(sv_br[idx].mean()),
            "arm_br": float(a.loc[idx, "br"].mean()),
            "accuracy": base.block_boot(d, "d_acc", rng),
            "log_loss": base.block_boot(d, "d_ll", rng),
            "brier": base.block_boot(d, "d_br", rng),
        }
    return res


def show(c):
    for m in ("accuracy", "log_loss", "brier"):
        e = c[m]
        print(
            "  ",
            m,
            round(e["effect"], 5),
            round(e["low"], 5),
            round(e["high"], 5),
            "P+",
            round(e["p_pos"], 3),
            "se",
            round(e["se"], 5),
        )


def main():
    t0 = time.time()
    cfb = pd.read_parquet(base.CFB_FEATURES)
    cfb["gameday"] = pd.to_datetime(cfb["gameday"])
    cols = ps.shared_columns()
    STATE["cfb"] = base.cfb_prior_rows(cfb)
    lev.predict_arms = arms
    g = ps.nfl_graded(cols)
    rng = np.random.default_rng(base.SEED)
    pops = {
        "nfl_opener_2020_25": g.loc[g["season"].between(*ps.NFL_OPENER_SEASONS)],
        "nfl_proxy_2011_19": g.loc[g["season"].between(*ps.NFL_PROXY_SEASONS)],
    }
    res = {k: {arm: lev.contrast(v, arm, rng) for arm in ("u1", "free")} for k, v in pops.items()}
    lg = pd.DataFrame(LOG)
    audit = {
        "cfb_not_before_train_end": int((lg["cfb_max_gameday"] >= lg["train_max_gameday"]).sum()),
        "train_not_before_test": int((lg["train_max_gameday"] >= lg["test_min_gameday"]).sum()),
        "weeks": len(lg),
    }
    fold = lg.groupby("season").agg(
        lam1_med=("lam1", "median"),
        lam1_max=("lam1", "max"),
        lam2_med=("lam2", "median"),
        lam2_max=("lam2", "max"),
        scale_med=("scale_mult", "median"),
        scale_min=("scale_mult", "min"),
        scale_max=("scale_mult", "max"),
        ratio=("ratio", "median"),
    )
    top = {
        "lam1_at_top": float((lg["lam1"] == STRENGTHS[-1]).mean()),
        "lam2_at_top": float((lg["lam2"] == STRENGTHS[-1]).mean()),
        "scale_at_top": float((lg["scale_mult"] == SCALES[-1]).mean()),
        "scale_at_zero": float((lg["scale_mult"] == 0).mean()),
    }
    h2h = served_h2h(g.loc[g["season"].between(*ps.NFL_OPENER_SEASONS)])
    report = {
        "populations": res,
        "leakage_audit": audit,
        "grid_edges": top,
        "by_season": json.loads(fold.round(4).to_json(orient="index")),
        "served_head_to_head": h2h,
        "seconds": time.time() - t0,
    }
    (OUT / "report_u2.json").write_text(json.dumps(report, indent=1))
    print(audit, top)
    for k, v in res.items():
        for arm, c in v.items():
            print(
                k,
                arm,
                c["base"]["wins"],
                c["base"]["n"],
                c["arm"]["wins"],
                c["flips_win"],
                c["flips_loss"],
            )
            show(c)
    for arm, c in h2h.items():
        print(
            "H2H",
            arm,
            {
                k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in c.items()
                if not isinstance(v, dict)
            },
        )
        show(c)
    print(fold.to_string())


if __name__ == "__main__":
    main()
