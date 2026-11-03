import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
U8 = Path(__file__).resolve().parent / "mod23_unit8_span.py"
NS = {"__file__": str(U8), "__name__": "mod23_unit8_span_import"}
exec(compile(U8.read_text().rsplit("\nmain()", 1)[0], str(U8), "exec"), NS)

import mod41_cumulative_workload as m41  # noqa: E402
from mod41_unit2_components import components  # noqa: E402

from nfl_ats import clv  # noqa: E402
from nfl_ats.margin import make_margin_estimator  # noqa: E402

u3 = NS["u3"]
HARNESS = NS["HARNESS"]
FULL = tuple(NS["FULL"])
BASE_ALPHA = NS["BASE_ALPHA"]
BASE5 = NS["BASE5"]
compare = NS["compare"]
boot_season = NS["boot_season"]
SUPP = HARNESS["SUPP"]
OUT = u3.ROOT / "artifacts" / "mod41" / "unit5"
COL = "mod41_resid_load"
FIRST_PBP_SEASON = 2011
TRUE_SEASONS = range(2020, 2026)
PROXY_SEASONS = range(2014, 2020)
PERM_SEED = 4105
BASE_RECORD = (802, 1503)


def load_frame():
    m41.SEASONS = range(FIRST_PBP_SEASON, 2026)
    t = m41.team_games()
    components(t)
    g = pd.read_parquet(u3.FEATURES, columns=["game_id", "season", "week"])
    parts = g["game_id"].str.split("_", expand=True)
    g["away_team"], g["home_team"] = parts[2], parts[3]
    for side in ("home", "away"):
        f = t[["game_id", "team", "ok", "load_total", "own_td", "opp3"]].rename(
            columns={
                "team": f"{side}_team",
                "ok": f"{side}_ok",
                "load_total": f"{side}_load",
                "own_td": f"{side}_own",
                "opp3": f"{side}_opp",
            }
        )
        g = g.merge(f, on=["game_id", f"{side}_team"], how="left")
    g["usable"] = g["home_ok"].fillna(False).astype(bool) & g["away_ok"].fillna(False).astype(bool)
    g["s_load"] = g["home_load"] - g["away_load"]
    g["s_own"] = g["home_own"] - g["away_own"]
    g["s_opp"] = g["home_opp"] - g["away_opp"]
    g.loc[g["usable"] & g[["s_load", "s_own", "s_opp"]].isna().any(axis=1), "usable"] = False
    return g


def resid_for_fold(g, season):
    tr = g.loc[g["usable"] & g["season"].lt(season)]
    x = np.column_stack([np.ones(len(tr)), tr["s_own"], tr["s_opp"]])
    beta = np.linalg.lstsq(x, tr["s_load"].to_numpy(), rcond=None)[0]
    u = g["usable"]
    out = pd.Series(0.0, index=g.index)
    xa = np.column_stack([np.ones(u.sum()), g.loc[u, "s_own"], g.loc[u, "s_opp"]])
    out[u] = g.loc[u, "s_load"].to_numpy() - xa @ beta
    return out, beta


def permuted(g, v):
    rng = np.random.default_rng(PERM_SEED)
    out = v.copy()
    for _, idx in g.loc[g["usable"]].groupby(["season", "week"]).groups.items():
        idx = np.asarray(idx)
        out[idx] = rng.permutation(v[idx])
    return out


def fold_frame(features, g, season, control):
    v, beta = resid_for_fold(g, season)
    arr = v.to_numpy()
    if control:
        arr = permuted(g, arr)
    f = features.copy()
    f[COL] = f["game_id"].map(pd.Series(arr, index=g["game_id"].to_numpy())).fillna(0.0)
    return f, [float(b) for b in beta]


def coef_and_gap(f, season):
    reg = f.loc[f["game_type"].eq("REG") & pd.to_numeric(f["ats_margin"], errors="coerce").notna()]
    train = reg.loc[reg["season"].lt(season)]
    test = reg.loc[reg["season"].eq(season)]
    cols = [*list(FULL), COL]
    ytr = pd.to_numeric(train["ats_margin"]).to_numpy(float)
    yte = pd.to_numeric(test["ats_margin"]).to_numpy(float)
    res = {}
    sse = {}
    for tag, c in (("base", list(FULL)), ("add", cols)):
        est = make_margin_estimator(
            "ridge", 42, ridge_alpha=BASE_ALPHA, suppressed_indicator_columns=SUPP
        )
        est.fit(train.loc[:, c], ytr)
        sse[tag] = (
            float(np.mean((ytr - np.asarray(est.predict(train.loc[:, c]), float)) ** 2)),
            float(np.mean((yte - np.asarray(est.predict(test.loc[:, c]), float)) ** 2)),
        )
        if tag == "add":
            lo = test.loc[:, c].copy()
            hi = lo.copy()
            hi[COL] = hi[COL] + 1.0
            res["coef_margin_per_unit"] = float(
                np.mean(np.asarray(est.predict(hi), float) - np.asarray(est.predict(lo), float))
            )
    res["in_mse_gain"] = sse["base"][0] - sse["add"][0]
    res["out_mse_gain"] = sse["base"][1] - sse["add"][1]
    return res


def main():
    mode = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    features = pd.read_parquet(u3.FEATURES)
    base = pd.read_parquet(BASE5)
    n_true = base.loc[base["season"].ge(2020)]
    if (int(n_true["correct"].sum()), len(n_true)) != BASE_RECORD:
        raise SystemExit("base reproduction failed")
    g = load_frame()
    control = mode.endswith("control")
    proxy = mode.startswith("proxy")
    seasons = PROXY_SEASONS if proxy else TRUE_SEASONS
    pop = HARNESS["proxy_population"](features) if proxy else None
    parts, folds = [], {}
    for s in seasons:
        f, beta = fold_frame(features, g, s, control)
        f = f.loc[f["season"].le(s)].reset_index(drop=True)
        cols = (*FULL, COL)
        if proxy:
            run = HARNESS["proxy_eval"](
                f, pop.loc[pop["season"].le(s)], lambda _s, c=cols: (c, BASE_ALPHA)
            )
        else:
            run = HARNESS["true_eval"](f, cols, BASE_ALPHA)
        parts.append(run.loc[run["season"].eq(s)])
        folds[int(s)] = {"beta": beta, **coef_and_gap(f, s)}
        print(mode, s, folds[int(s)], flush=True)
    res = pd.concat(parts)
    res.to_parquet(OUT / f"{mode}_run.parquet")
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
        "compare": cmp_,
        "ll_gain_boot": boot_season(j, "neg_d_ll"),
        "ll_gain": float(-j["d_ll"].mean()),
        "brier_gain": float(-j["d_brier"].mean()),
        "acc_gain": float(j["d_acc"].mean()),
        "base_record": f"{int(j['correct_b'].sum())}-{len(j) - int(j['correct_b'].sum())}",
        "n": len(j),
        "folds": folds,
    }
    (OUT / f"{mode}_report.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, default=str)[:3000], flush=True)


main()
