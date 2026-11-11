import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
M10 = Path(__file__).resolve().parent / "mod10_graph_rating.py"
NS = {"__file__": str(M10), "__name__": "mod10_import"}
exec(compile(M10.read_text().rsplit("\nmain()", 1)[0], str(M10), "exec"), NS)

from nfl_ats import clv  # noqa: E402
from nfl_ats.margin import make_margin_estimator  # noqa: E402

u3 = NS["u3"]
HARNESS = NS["HARNESS"]
FULL = tuple(NS["FULL"])
BASE_ALPHA = NS["BASE_ALPHA"]
BASE5 = NS["BASE5"]
SUPP = NS["SUPP"]
compare = NS["compare"]
boot_season = NS["boot_season"]
TAUS_DAYS = NS["TAUS_DAYS"]
LAMBDAS = NS["LAMBDAS"]
fit_block = NS["fit_block"]
choose = NS["choose"]
permuted = NS["permuted"]
load_games = NS["load_games"]
TRUE_SEASONS = NS["TRUE_SEASONS"]
PROXY_SEASONS = NS["PROXY_SEASONS"]
BASE_RECORD = NS["BASE_RECORD"]
OUT = u3.ROOT / "artifacts" / "mod10" / "unit2"
ROLLING_EPA = tuple(
    f"{side}_{m}"
    for m in (
        "off_epa_per_play",
        "off_pass_epa_per_play",
        "off_rush_epa_per_play",
        "def_epa_per_play",
        "def_pass_epa_per_play",
        "def_rush_epa_per_play",
    )
    for side in ("home", "away", "diff")
)
LEVEL_COLS = ("mod10_rh", "mod10_ra", "mod10_diff")
ARMS = {
    "replace3": (ROLLING_EPA, LEVEL_COLS),
    "replace1": (ROLLING_EPA, ("mod10_diff",)),
    "addone": ((), ("mod10_diff",)),
}


def grid(f, teams):
    tid = {t: i for i, t in enumerate(teams)}
    hi = f["home_team"].map(tid).to_numpy()
    ai = f["away_team"].map(tid).to_numpy()
    ne = f["neutral"].to_numpy()
    d = f["day"].to_numpy()
    y = f["y_epa"].to_numpy(float)
    ok = ~np.isnan(y)
    shape = (len(TAUS_DAYS), len(LAMBDAS), len(f))
    rh, ra, hfa = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for idx in f.groupby(["season", "week"]).indices.values():
        idx = np.asarray(idx)
        day0 = d[idx].min()
        prior = np.where((d < day0) & ok)[0]
        if len(prior) < len(teams):
            continue
        for ti, tau in enumerate(TAUS_DAYS):
            for li, lam in enumerate(LAMBDAS):
                beta = fit_block(
                    day0, d[prior], hi[prior], ai[prior], ne[prior], y[prior], len(teams), tau, lam
                )
                r = beta[1:]
                rh[ti, li, idx] = r[hi[idx]]
                ra[ti, li, idx] = r[ai[idx]]
                hfa[ti, li, idx] = beta[0] * (1.0 - ne[idx])
    return rh, ra, hfa, y


def coef_and_gap(f, cols_base, cols_new, season):
    reg = f.loc[f["game_type"].eq("REG") & pd.to_numeric(f["ats_margin"], errors="coerce").notna()]
    train = reg.loc[reg["season"].lt(season)]
    test = reg.loc[reg["season"].eq(season)]
    ytr = pd.to_numeric(train["ats_margin"]).to_numpy(float)
    yte = pd.to_numeric(test["ats_margin"]).to_numpy(float)
    sse = {}
    for tag, c in (("base", list(FULL)), ("new", list(cols_new))):
        est = make_margin_estimator(
            "ridge", 42, ridge_alpha=BASE_ALPHA, suppressed_indicator_columns=SUPP
        )
        est.fit(train.loc[:, c], ytr)
        sse[tag] = (
            float(np.mean((ytr - np.asarray(est.predict(train.loc[:, c]), float)) ** 2)),
            float(np.mean((yte - np.asarray(est.predict(test.loc[:, c]), float)) ** 2)),
        )
        if tag == "new" and "mod10_diff" in c:
            lo = test.loc[:, c].copy()
            hi = lo.copy()
            hi["mod10_diff"] = hi["mod10_diff"] + 1.0
            coef = float(
                np.mean(np.asarray(est.predict(hi), float) - np.asarray(est.predict(lo), float))
            )
    return {
        "coef_margin_per_unit": coef,
        "in_mse_gain": sse["base"][0] - sse["new"][0],
        "out_mse_gain": sse["base"][1] - sse["new"][1],
        "gap": (sse["base"][0] - sse["new"][0]) - (sse["base"][1] - sse["new"][1]),
    }


def main():
    mode, arm = sys.argv[1], sys.argv[2]
    removed, added = ARMS[arm]
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    features = pd.read_parquet(u3.FEATURES)
    base = pd.read_parquet(BASE5)
    n_true = base.loc[base["season"].ge(2020)]
    if (int(n_true["correct"].sum()), len(n_true)) != BASE_RECORD:
        raise SystemExit("base reproduction failed")
    games = load_games()
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    rh, ra, hfa, y = grid(games, teams)
    diff = rh - ra
    control = mode.endswith("control")
    proxy = mode.startswith("proxy")
    seasons = PROXY_SEASONS if proxy else TRUE_SEASONS
    pop = HARNESS["proxy_population"](features) if proxy else None
    key = games["game_id"].to_numpy()
    cols = tuple(c for c in FULL if c not in removed) + tuple(added)
    parts, folds = [], {}
    for s in seasons:
        (ti, li), _mse = choose(games, diff, hfa, y, s)
        f = features.copy()
        for name, a in (("mod10_rh", rh), ("mod10_ra", ra), ("mod10_diff", diff)):
            v = pd.Series(a[ti, li], index=key).reindex(features["game_id"]).to_numpy()
            arr = np.nan_to_num(v, nan=0.0)
            if control:
                arr = permuted(features[["season", "week"]].reset_index(drop=True), arr)
            f[name] = arr
        f = f.loc[f["season"].le(s)].reset_index(drop=True)
        if proxy:
            run = HARNESS["proxy_eval"](
                f, pop.loc[pop["season"].le(s)], lambda _s, c=cols: (c, BASE_ALPHA)
            )
        else:
            run = HARNESS["true_eval"](f, cols, BASE_ALPHA)
        parts.append(run.loc[run["season"].eq(s)])
        folds[int(s)] = {
            "tau_days": TAUS_DAYS[ti],
            "lambda": LAMBDAS[li],
            **coef_and_gap(f, FULL, cols, s),
        }
        print(mode, arm, s, folds[int(s)], flush=True)
    res = pd.concat(parts)
    b = base.loc[base["season"].isin(list(seasons))]
    cmp_ = compare(res.loc[res.index.isin(b.index)], b)
    j = res.join(b, rsuffix="_b", how="inner")
    j["d_ll"] = j["ll"] - j["ll_b"]
    j["d_brier"] = j["brier"] - j["brier_b"]
    j["d_acc"] = 100.0 * (j["correct"] - j["correct_b"])
    j["neg_d_ll"] = -j["d_ll"]
    j["neg_d_brier"] = -j["d_brier"]
    out = {
        "mode": mode,
        "arm": arm,
        "removed": len([c for c in FULL if c in removed]),
        "added": len(added),
        "compare": cmp_,
        "ll_gain_boot": boot_season(j, "neg_d_ll"),
        "brier_gain_boot": boot_season(j, "neg_d_brier"),
        "ll_gain": float(-j["d_ll"].mean()),
        "brier_gain": float(-j["d_brier"].mean()),
        "acc_gain": float(j["d_acc"].mean()),
        "rec": f"{int(j['correct'].sum())}-{len(j) - int(j['correct'].sum())}",
        "base_record": f"{int(j['correct_b'].sum())}-{len(j) - int(j['correct_b'].sum())}",
        "n": len(j),
        "folds": folds,
    }
    (OUT / f"{mode}_{arm}_report.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, default=str)[:5000], flush=True)


main()
