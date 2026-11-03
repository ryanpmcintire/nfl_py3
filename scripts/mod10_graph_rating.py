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
OUT = u3.ROOT / "artifacts" / "mod10" / "unit1"
PBP = u3.ROOT / "data" / "pbp" / "raw" / "20260929T191306Z"
FIRST_SEASON = 2009
SELECT_FROM = 2011
TRUE_SEASONS = range(2020, 2026)
PROXY_SEASONS = range(2014, 2020)
PERM_SEED = 1001
BASE_RECORD = (802, 1503)
TAUS_DAYS = (60.0, 120.0, 240.0, 480.0, 960.0)
LAMBDAS = (1.0, 4.0, 16.0, 64.0, 256.0)


def load_games():
    f = pd.read_parquet(
        u3.FEATURES,
        columns=[
            "game_id",
            "season",
            "week",
            "kickoff",
            "home_score",
            "away_score",
            "neutral_site",
        ],
    )
    parts = f["game_id"].str.split("_", expand=True)
    f["away_team"], f["home_team"] = parts[2], parts[3]
    f["kick"] = pd.to_datetime(f["kickoff"], utc=True)
    f["day"] = (f["kick"] - pd.Timestamp("2000-01-01", tz="UTC")).dt.total_seconds() / 86400.0
    f["neutral"] = f["neutral_site"].fillna(0).astype(float)
    f["y_margin"] = f["home_score"] - f["away_score"]
    cols = ["game_id", "posteam", "play_type", "epa", "qb_kneel", "qb_spike"]
    p = pd.concat(
        [
            pd.read_parquet(PBP / f"season={s}" / "plays.parquet", columns=cols)
            for s in range(FIRST_SEASON, 2026)
        ],
        ignore_index=True,
    )
    p = p[p.play_type.isin(["pass", "run"]) & (p.qb_kneel != 1) & (p.qb_spike != 1)]
    p = p[p.posteam.notna() & p.epa.notna()]
    e = p.groupby(["game_id", "posteam"])["epa"].mean().rename("e").reset_index()
    h = e.rename(columns={"posteam": "home_team", "e": "e_home"})
    a = e.rename(columns={"posteam": "away_team", "e": "e_away"})
    f = f.merge(h, on=["game_id", "home_team"], how="left").merge(
        a, on=["game_id", "away_team"], how="left"
    )
    f["y_epa"] = f["e_home"] - f["e_away"]
    return f.sort_values("kick").reset_index(drop=True)


def fit_block(day0, d, hi, ai, ne, y, nteam, tau, lam):
    w = np.exp(-(day0 - d) / tau)
    x = np.zeros((len(y), nteam + 1))
    x[:, 0] = 1.0 - ne
    x[np.arange(len(y)), 1 + hi] += 1.0
    x[np.arange(len(y)), 1 + ai] -= 1.0
    xw = x * w[:, None]
    a = xw.T @ x
    pen = np.full(nteam + 1, lam)
    pen[0] = 0.0
    a += np.diag(pen)
    return np.linalg.solve(a, xw.T @ y)


def rating_diffs(f, kind, teams):
    tid = {t: i for i, t in enumerate(teams)}
    hi = f["home_team"].map(tid).to_numpy()
    ai = f["away_team"].map(tid).to_numpy()
    ne = f["neutral"].to_numpy()
    d = f["day"].to_numpy()
    y = f["y_margin" if kind == "margin" else "y_epa"].to_numpy(float)
    ok = ~np.isnan(y)
    diff = np.zeros((len(TAUS_DAYS), len(LAMBDAS), len(f)))
    hfa = np.zeros((len(TAUS_DAYS), len(LAMBDAS), len(f)))
    blocks = f.groupby(["season", "week"]).indices
    for idx in blocks.values():
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
                diff[ti, li, idx] = r[hi[idx]] - r[ai[idx]]
                hfa[ti, li, idx] = beta[0] * (1.0 - ne[idx])
    return diff, hfa, y


def choose(f, diff, hfa, y, season):
    seas = f["season"].to_numpy()
    best, arg = np.inf, None
    for ti in range(len(TAUS_DAYS)):
        for li in range(len(LAMBDAS)):
            per = []
            for s in range(SELECT_FROM, season):
                ms = (seas == s) & ~np.isnan(y)
                if ms.sum():
                    per.append(float(np.mean((y[ms] - diff[ti, li][ms] - hfa[ti, li][ms]) ** 2)))
            v = float(np.mean(per))
            if v < best:
                best, arg = v, (ti, li)
    return arg, best


def permuted(g, v):
    rng = np.random.default_rng(PERM_SEED)
    out = v.copy()
    for idx in g.groupby(["season", "week"]).indices.values():
        idx = np.asarray(idx)
        out[idx] = rng.permutation(v[idx])
    return out


def coef_and_gap(f, col, season):
    reg = f.loc[f["game_type"].eq("REG") & pd.to_numeric(f["ats_margin"], errors="coerce").notna()]
    train = reg.loc[reg["season"].lt(season)]
    test = reg.loc[reg["season"].eq(season)]
    ytr = pd.to_numeric(train["ats_margin"]).to_numpy(float)
    yte = pd.to_numeric(test["ats_margin"]).to_numpy(float)
    res, sse = {}, {}
    for tag, c in (("base", list(FULL)), ("add", [*FULL, col])):
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
            hi[col] = hi[col] + 1.0
            res["coef_margin_per_unit"] = float(
                np.mean(np.asarray(est.predict(hi), float) - np.asarray(est.predict(lo), float))
            )
            res["feature_sd"] = float(train[col].std())
    res["in_mse_gain"] = sse["base"][0] - sse["add"][0]
    res["out_mse_gain"] = sse["base"][1] - sse["add"][1]
    return res


def main():
    mode = sys.argv[1]
    kind = sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    col = f"mod10_graph_{kind}"
    features = pd.read_parquet(u3.FEATURES)
    base = pd.read_parquet(BASE5)
    n_true = base.loc[base["season"].ge(2020)]
    if (int(n_true["correct"].sum()), len(n_true)) != BASE_RECORD:
        raise SystemExit("base reproduction failed")
    print("baseline reproduced", BASE_RECORD, flush=True)
    games = load_games()
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    diff, hfa, y = rating_diffs(games, kind, teams)
    control = mode.endswith("control")
    proxy = mode.startswith("proxy")
    seasons = PROXY_SEASONS if proxy else TRUE_SEASONS
    pop = HARNESS["proxy_population"](features) if proxy else None
    key = games["game_id"].to_numpy()
    parts, folds, feats = [], {}, []
    for s in seasons:
        (ti, li), mse = choose(games, diff, hfa, y, s)
        v = pd.Series(diff[ti, li], index=key)
        arr = np.nan_to_num(v.reindex(features["game_id"]).to_numpy(), nan=0.0)
        if control:
            arr = permuted(features[["season", "week"]].reset_index(drop=True), arr)
        f = features.copy()
        f[col] = arr
        f = f.loc[f["season"].le(s)].reset_index(drop=True)
        cols = (*FULL, col)
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
            "inner_loso_mse": mse,
            **coef_and_gap(f, col, s),
        }
        feats.append(f.loc[f["season"].eq(s) & f["game_type"].eq("REG")])
        print(mode, kind, s, folds[int(s)], flush=True)
    res = pd.concat(parts)
    res.to_parquet(OUT / f"{mode}_{kind}_run.parquet")
    allf = pd.concat(feats)
    xc = allf[[c for c in FULL if c in allf.columns]].apply(pd.to_numeric, errors="coerce")
    corr = {c: float(allf[col].corr(xc[c])) for c in xc.columns if xc[c].std() > 0}
    top = dict(sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:8])
    xm = np.column_stack([np.ones(len(xc)), xc.fillna(xc.median()).to_numpy(float)])
    tgt = allf[col].to_numpy(float)
    beta = np.linalg.lstsq(xm, tgt, rcond=None)[0]
    r2 = float(1.0 - np.var(tgt - xm @ beta) / np.var(tgt))
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
        "kind": kind,
        "compare": cmp_,
        "ll_gain_boot": boot_season(j, "neg_d_ll"),
        "brier_gain_boot": boot_season(j, "neg_d_brier"),
        "ll_gain": float(-j["d_ll"].mean()),
        "brier_gain": float(-j["d_brier"].mean()),
        "acc_gain": float(j["d_acc"].mean()),
        "base_record": f"{int(j['correct_b'].sum())}-{len(j) - int(j['correct_b'].sum())}",
        "n": len(j),
        "corr_with_existing_top": top,
        "r2_on_existing_features": r2,
        "folds": folds,
    }
    (OUT / f"{mode}_{kind}_report.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, default=str)[:6000], flush=True)


main()
