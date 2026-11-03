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

STRENGTHS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
OUT = ROOT / "artifacts" / "xlg11_cfb_centered"
LOG = []
STATE = {}


def pick_strength(a, y, seasons, prior):
    scores = []
    for lam in STRENGTHS:
        sse = 0.0
        for s in np.unique(seasons):
            tr, te = seasons != s, seasons == s
            m = Ridge(alpha=lam).fit(a[tr], y[tr] - a[tr] @ prior)
            sse += float(np.sum((y[te] - a[te] @ prior - m.predict(a[te])) ** 2))
        scores.append(sse)
    return STRENGTHS[int(np.argmin(scores))]


def centered(train, test, cols):
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
    prior = ratio * Ridge(alpha=lev.ALPHA).fit(ca, cy).coef_
    lam = pick_strength(a, y, train["season"].to_numpy(), prior)
    m = Ridge(alpha=lam).fit(a, y - a @ prior)
    pred = b @ prior + m.predict(b)
    fit = a @ prior + m.predict(a)
    LOG.append(
        {
            "season": int(test["season"].iloc[0]),
            "week": int(test["week"].iloc[0]),
            "lam": lam,
            "cfb_n": len(past),
            "ratio": ratio,
        }
    )
    return pred, float(np.std(y - fit, ddof=1))


def install():
    original = lev.predict_arms

    def patched(train, test, cols, arms):
        out = original(train, test, cols, ())
        out["variant"] = centered(train, test, cols)
        return out

    lev.predict_arms = patched


def main():
    t0 = time.time()
    cfb = pd.read_parquet(base.CFB_FEATURES)
    cfb["gameday"] = pd.to_datetime(cfb["gameday"])
    cols = ps.shared_columns()
    STATE["cfb"] = base.cfb_prior_rows(cfb)
    install()
    g = ps.nfl_graded(cols)
    rng = np.random.default_rng(base.SEED)
    pops = {
        "nfl_opener_2020_25": g.loc[g["season"].between(*ps.NFL_OPENER_SEASONS)],
        "nfl_proxy_2011_19": g.loc[g["season"].between(*ps.NFL_PROXY_SEASONS)],
    }
    res = {k: lev.contrast(v, "variant", rng) for k, v in pops.items()}
    piv = g.pivot_table(index="game_id", columns="arm", values=["correct", "ll", "br"])
    seas = g.drop_duplicates("game_id").set_index("game_id")["season"]
    d = pd.DataFrame(
        {
            "season": seas,
            "d_acc": 100 * (piv["correct"]["variant"] - piv["correct"]["base"]),
            "d_ll": piv["ll"]["base"] - piv["ll"]["variant"],
            "d_br": piv["br"]["base"] - piv["br"]["variant"],
        }
    ).dropna()
    per_season = d.groupby("season").mean().round(5)
    per_season["n"] = d.groupby("season").size()
    lg = pd.DataFrame(LOG)
    fold = lg.groupby("season")["lam"].agg(["median", "min", "max"])
    fold["share_100plus"] = lg.groupby("season")["lam"].apply(lambda s: float((s >= 100).mean()))
    report = {
        "populations": res,
        "per_season": json.loads(per_season.to_json(orient="index")),
        "strength_by_season": json.loads(fold.to_json(orient="index")),
        "cfb_ratio_median": float(lg["ratio"].median()),
        "seconds": time.time() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    for k, c in res.items():
        print(
            k,
            "base",
            c["base"]["wins"],
            c["base"]["n"],
            "arm",
            c["arm"]["wins"],
            "flips",
            c["flips_win"],
            c["flips_loss"],
        )
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
                "n",
                e["games"],
            )
        print("   ll", c["base"]["ll"], c["arm"]["ll"], "br", c["base"]["br"], c["arm"]["br"])
    print(per_season.to_string())
    print(fold.to_string())


if __name__ == "__main__":
    main()
