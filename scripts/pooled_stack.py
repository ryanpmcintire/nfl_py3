from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mod27_fenced_repairs_replay import FOUR, Fit

from nfl_ats.pick_probability import active_market_move_feature_version
from nfl_ats.pick_probability_fit import build_fit_population

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ART / "pooled_stack"
FAMILY = "pooled_stack_2026_11_16"
LANE = "docs/lanes/pooled-stack-2026-11-16.md"
QW = ART / "v4_candidate" / "opener_QW.parquet"
PT = ART / "rating_dispersion" / "nfl_dispersion.parquet"
U6 = ART / "xlg14_u6_nfl" / "team_values.parquet"
CY = ART / "contract_year" / "team_values.parquet"
REF = ART / "referee_tendency" / "screen_games.parquet"
WT = ART / "win_totals_test" / "features.parquet"
FIRST, LAST = 2020, 2025
FLAGS = (
    "flag_coach",
    "flag_division",
    "flag_arrests",
    "flag_bye",
    "flag_cold_visitor",
    "flag_protection",
    "flag_tank_zone",
)
L1 = (0.0, 0.25, 0.5, 0.75, 1.0)
PATH_LEN = 20
MIN_RATIO = 1e-4
PUSH_LABEL = 0.5
DRAWS = 200
BOOT = 2000
SEED = 20261116
TOL = 1e-7
MAX_ITER = 5000
EPS = 1e-12
CTX = {}


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def population():
    version = active_market_move_feature_version(ART)
    pop, _ = build_fit_population(
        ART, ROOT / "data", market_move_feature_version=version, keep_pushes=True
    )
    pop = pop[pop["season"].between(FIRST, LAST)].reset_index(drop=True)
    qw = pd.read_parquet(QW)["home_cover_probability_at_open"]
    mp = qw.reindex(pop["game_id"]).to_numpy(float)
    if np.isnan(mp).any():
        raise SystemExit(f"QW model probability missing for {int(np.isnan(mp).sum())} games")
    pop["model_logit"] = logit(mp)
    pop["push"] = pop["margin_vs_open"].eq(0.0)
    pop["y"] = np.where(pop["push"], PUSH_LABEL, pop["home_covered"].astype(float))
    return pop


def two_sided(values, pop, name, ok_col):
    look = values.drop_duplicates(["game_id", "team"]).set_index(["game_id", "team"])
    side = {}
    for t in ("home", "away"):
        key = pd.MultiIndex.from_arrays([pop["game_id"], pop[f"{t}_team"]])
        side[t] = look.reindex(key)
    ok = (side["home"][ok_col].to_numpy(float) > 0) & (side["away"][ok_col].to_numpy(float) > 0)
    diff = side["home"][name].to_numpy(float) - side["away"][name].to_numpy(float)
    return np.where(ok, diff, np.nan)


def signals(pop):
    import contract_year as cy
    import hindsight_audit_team as hat
    import outside_model_gap as omg
    import prediction_market_gap as pmg

    gid = pop["game_id"]
    line = pop["tue_open_home_spread"].to_numpy(float)
    s = pd.DataFrame(index=pop.index)
    for f in FLAGS:
        s[f] = pop[f].astype(float).to_numpy()
    e538, nf, conv = omg.terms()
    e = e538.set_index("game_id").reindex(gid)
    n = nf.set_index("game_id").reindex(gid)
    s["nfelo_gap"] = n["nfelo_margin"].to_numpy(float) - line
    s["qbelo_gap"] = e["qbelo_margin"].to_numpy(float) - line
    s["qb_adjust"] = e["qb_adjust"].to_numpy(float)
    pt = (
        pd.read_parquet(PT, columns=["game_id", "dispersion"])
        .drop_duplicates("game_id")
        .set_index("game_id")
    )
    d = pt["dispersion"].reindex(gid).to_numpy(float)
    s["pt_dispersion_x_model"] = (
        (d - np.nanmean(d)) / np.nanstd(d) * pop["model_logit"].to_numpy(float)
    )
    h, _ = pmg.history()
    s["pm_gap_tue"] = h["tue"].reindex(gid).to_numpy(float)
    u6 = pd.read_parquet(U6)
    for name in ("margin_sos_gap", "last_opponent_strength"):
        s[name] = two_sided(u6, pop, name, "prior_games")
    s["contract_year_gap"] = two_sided(pd.read_parquet(CY), pop, cy.NAME, "prior_snaps")
    ref = (
        pd.read_parquet(REF, columns=["game_id", "trait"])
        .drop_duplicates("game_id")
        .set_index("game_id")
    )
    s["referee_trait"] = ref["trait"].reindex(gid).to_numpy(float)
    wt = pd.read_parquet(WT).drop_duplicates("game_id").set_index("game_id")
    s["wt_drift"] = wt["wt_drift"].reindex(gid).to_numpy(float)
    s["sb_drift"] = wt["sb_drift"].reindex(gid).to_numpy(float)
    s["model_w1"] = pop["model_logit"].to_numpy(float) * pop["week"].eq(1).to_numpy(float)
    table = pd.read_parquet(hat.v4.TABLE)
    table["game_id"] = table["game_id"].astype(str)
    totals, _ = hat.tuesday_quotes(table)
    s["asof_total"] = totals.reindex(gid).to_numpy(float)
    keep = [c for c in s.columns if np.nanstd(s[c].to_numpy(float)) > 0]
    return s[keep], {"dropped_constant": [c for c in s.columns if c not in keep], "nfelo": conv}


def standardise(raw, gaps, mask):
    cols, names = [], []
    for j in range(raw.shape[1]):
        x = raw[:, j]
        obs = ~np.isnan(x)
        tr = obs & mask
        mu = x[tr].mean() if tr.any() else 0.0
        sd = x[tr].std() if tr.sum() > 1 else 0.0
        cols.append(np.where(obs, (x - mu) / sd, 0.0) if sd > 0 else np.zeros(len(x)))
        names.append(j)
        if gaps[j]:
            ind = (~obs).astype(float)
            m, v = ind[mask].mean(), ind[mask].std()
            cols.append((ind - m) / v if v > 0 else np.zeros(len(x)))
            names.append(-1 - j)
    return np.column_stack(cols), names


def enet(X, y, off, lam, a, b0, lip):
    n = len(y)
    step = 1.0 / (lip + lam * (1 - a))
    b = b0.copy()
    z = b.copy()
    t = 1.0
    for _ in range(MAX_ITER):
        g = X.T @ (expit(off + X @ z) - y) / n + lam * (1 - a) * z
        u = z - step * g
        bn = np.sign(u) * np.maximum(np.abs(u) - step * lam * a, 0.0)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = bn + ((t - 1) / tn) * (bn - b)
        if np.max(np.abs(bn - b)) < TOL:
            return bn
        b, t = bn, tn
    return b


def lipschitz(X):
    return 0.25 * np.linalg.norm(X, 2) ** 2 / len(X)


def lam_path(X, y, off):
    top = float(np.max(np.abs(X.T @ (y - expit(off))) / len(y)))
    return np.geomspace(top, top * MIN_RATIO, PATH_LEN)


def path_scores(X, fit, held, y, ylab, off_fit, off_held, lams):
    Xf, yf, of = X[fit], y[fit], off_fit[fit]
    lip = lipschitz(Xf)
    out = np.zeros((len(L1), len(lams) + 1))
    base = ll(expit(off_held[held]), ylab[held]).sum()
    for ai, a in enumerate(L1):
        b = np.zeros(X.shape[1])
        for li, lam in enumerate(lams):
            b = enet(Xf, yf, of, lam, a, b, lip)
            out[ai, li] = ll(expit(off_held[held] + X[held] @ b), ylab[held]).sum()
        out[ai, -1] = base
    return out


def choose(scores, lams):
    best = None
    for li in [len(lams), *list(range(len(lams)))]:
        for ai in range(len(L1)):
            v = scores[ai, li]
            if best is None or v < best[0] - 1e-12:
                best = (v, ai, li)
    return best[1], best[2]


def fit_at(X, y, off, a, lams, li):
    if li >= len(lams):
        return np.zeros(X.shape[1])
    lip = lipschitz(X)
    b = np.zeros(X.shape[1])
    for lam in lams[: li + 1]:
        b = enet(X, y, off, lam, a, b, lip)
    return b


def nested(raw):
    c = CTX
    season, decided, y, ylab, gaps = c["season"], c["decided"], c["y"], c["ylab"], c["gaps"]
    seasons = c["seasons"]
    p = np.full(len(y), np.nan)
    folds = {}
    for s in seasons:
        train = [u for u in seasons if u != s]
        trmask = np.isin(season, train)
        X, names = standardise(raw, gaps, trmask)
        off = c["off"][(s,)]
        fit = trmask & decided
        lams = lam_path(X[fit], y[fit], off[fit])
        scores = np.zeros((len(L1), len(lams) + 1))
        for t in train:
            inner = np.isin(season, [u for u in train if u != t])
            Xi, _ = standardise(raw, gaps, inner)
            oi = c["off"][(s, t)]
            scores += path_scores(Xi, inner & decided, season == t, y, ylab, oi, oi, lams)
        ai, li = choose(scores, lams)
        b = fit_at(X[fit], y[fit], off[fit], L1[ai], lams, li)
        held = season == s
        p[held] = expit(off[held] + X[held] @ b)
        folds[int(s)] = {
            "l1_ratio": L1[ai],
            "lambda_index": int(li),
            "lambda": float(lams[li]) if li < len(lams) else None,
            "coef": {int(k): float(v) for k, v in zip(names, b, strict=True)},
        }
    return p, folds


def insample(raw):
    c = CTX
    season, decided, y, ylab, gaps, seasons = (
        c["season"],
        c["decided"],
        c["y"],
        c["ylab"],
        c["gaps"],
        c["seasons"],
    )
    allm = np.ones(len(y), bool)
    X, _ = standardise(raw, gaps, allm)
    off = logit(c["qw"])
    lams = lam_path(X[decided], y[decided], off[decided])
    scores = np.zeros((len(L1), len(lams) + 1))
    for t in seasons:
        inner = season != t
        Xi, _ = standardise(raw, gaps, inner)
        oi = c["off"][(t,)]
        scores += path_scores(Xi, inner & decided, season == t, y, ylab, oi, oi, lams)
    ai, li = choose(scores, lams)
    b = fit_at(X[decided], y[decided], off[decided], L1[ai], lams, li)
    return expit(off + X @ b), {"l1_ratio": L1[ai], "lambda_index": int(li)}


def init_worker(ctx):
    CTX.update(ctx)


def null_draw(i):
    rng = np.random.default_rng([SEED, i])
    raw = CTX["raw"].copy()
    season = CTX["season"]
    for s in CTX["seasons"]:
        idx = np.flatnonzero(season == s)
        raw[idx] = raw[rng.permutation(idx)]
    p, _ = nested(raw)
    return gains(p, CTX["qw"], CTX["y"], CTX["ylab"], CTX["decided"])


def gains(p, q, y, ylab, decided):
    acc = 100 * (
        ((p[decided] >= 0.5) == (y[decided] == 1)).mean()
        - ((q[decided] >= 0.5) == (y[decided] == 1)).mean()
    )
    return float(acc), float((ll(q, ylab) - ll(p, ylab)).mean())


def block_boot(blocks, a_game, b_game, rng):
    ub, inv = np.unique(blocks, return_inverse=True)
    sa = np.bincount(inv, a_game)
    sb = np.bincount(inv, b_game)
    k = len(ub)
    draws = rng.integers(0, k, size=(BOOT, k))
    est = sa[draws].sum(1) / sb[draws].sum(1)
    point = sa.sum() / sb.sum()
    return {
        "effect": float(point),
        "interval_low": float(np.percentile(est, 2.5)),
        "interval_high": float(np.percentile(est, 97.5)),
        "probability_positive": float((est > 0).mean() + 0.5 * (est == 0).mean()),
        "blocks": int(k),
    }


def record(p, y):
    c = (p >= 0.5) == (y == 1)
    return f"{int(c.sum())}-{int((~c).sum())}"


def reliability(p, y, bins=5):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    k = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    return [
        {
            "games": int((k == i).sum()),
            "mean_p": float(p[k == i].mean()),
            "observed": float(y[k == i].mean()),
        }
        for i in range(bins)
    ]


def offsets(pop, season, decided, excl, seasons, cache):
    out = np.full(len(pop), np.nan)
    for u in seasons:
        key = frozenset(set(excl) | {u})
        if key not in cache:
            cache[key] = Fit(FOUR, pop[decided & ~np.isin(season, list(key))])
        m = season == u
        out[m] = logit(cache[key].predict(pop[m]))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Nested-LOSO elastic-net stack of pregame signals on the QW served offset."
    )
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pop = population()
    sig, sig_meta = signals(pop)
    season = pop["season"].to_numpy(int)
    decided = ~pop["push"].to_numpy(bool)
    y = np.where(decided, pop["home_covered"].astype(float).to_numpy(), 0.0)
    ylab = pop["y"].to_numpy(float)
    seasons = sorted(set(season.tolist()))
    cache = {}
    off = {}
    for s in seasons:
        off[(s,)] = offsets(pop, season, decided, {s}, seasons, cache)
        for t in seasons:
            if t != s:
                off[(s, t)] = offsets(
                    pop, season, decided, {s, t}, [u for u in seasons if u != s], cache
                )
    qw = np.full(len(pop), np.nan)
    for s in seasons:
        qw[season == s] = expit(off[(s,)][season == s])
    raw = sig.to_numpy(float)
    gaps = [bool(np.isnan(raw[:, j]).any()) for j in range(raw.shape[1])]
    ctx = {
        "season": season,
        "decided": decided,
        "y": y,
        "ylab": ylab,
        "gaps": gaps,
        "seasons": seasons,
        "off": off,
        "raw": raw,
        "qw": qw,
    }
    CTX.update(ctx)
    sig.assign(game_id=pop["game_id"].to_numpy(), season=season).to_parquet(
        OUT / "signals.parquet", index=False
    )
    coverage = {
        c: {int(s): int((~np.isnan(raw[season == s, j])).sum()) for s in seasons}
        for j, c in enumerate(sig.columns)
    }
    p, folds = nested(raw)
    names = list(sig.columns)

    def label(k):
        return names[k] if k >= 0 else f"{names[-1 - k]}_missing"

    nonzero = {}
    for f in folds.values():
        f["coef"] = {label(k): v for k, v in f["coef"].items()}
        for k, v in f["coef"].items():
            nonzero[k] = nonzero.get(k, 0) + int(abs(v) > 0)
    yd, pd_, qd = y[decided], p[decided], qw[decided]
    acc, llg = gains(p, qw, y, ylab, decided)
    rng = np.random.default_rng(SEED)
    wk = pop["season"].to_numpy(int) * 100 + pop["week"].to_numpy(int)
    corr_p = ((p >= 0.5) == (y == 1)) & decided
    corr_q = ((qw >= 0.5) == (y == 1)) & decided
    acc_game = 100 * (corr_p.astype(float) - corr_q.astype(float))
    ll_game = ll(qw, ylab) - ll(p, ylab)
    ones = np.ones(len(p))
    dec = decided.astype(float)
    boot = {
        "accuracy_week_block": block_boot(wk, acc_game, dec, rng),
        "accuracy_season_block": block_boot(season, acc_game, dec, rng),
        "log_loss_week_block": block_boot(wk, ll_game, ones, rng),
        "log_loss_season_block": block_boot(season, ll_game, ones, rng),
    }
    changed = (p >= 0.5) != (qw >= 0.5)
    chd = changed & decided
    p_in, in_choice = insample(raw)
    t_fit = round(time.time() - t0, 1)
    print(
        "fit seconds",
        t_fit,
        "qw",
        record(qd, yd),
        "stack",
        record(pd_, yd),
        "acc",
        acc,
        "ll",
        llg,
        flush=True,
    )
    t1 = time.time()
    if args.draws > 0:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=init_worker, initargs=(ctx,)
        ) as ex:
            null = np.array(list(ex.map(null_draw, range(args.draws))))
        pd.DataFrame(null, columns=["accuracy_points", "log_loss_gain"]).to_parquet(
            OUT / "null.parquet", index=False
        )
    else:
        null = np.full((1, 2), np.nan)

    def pct(obs, col):
        return float(100 * ((null[:, col] < obs).mean() + 0.5 * (null[:, col] == obs).mean()))

    pd.DataFrame(
        {
            "game_id": pop["game_id"],
            "season": season,
            "week": pop["week"],
            "push": ~decided,
            "y": ylab,
            "qw": qw,
            "stack": p,
        }
    ).to_parquet(OUT / "per_game.parquet", index=False)
    result = {
        "family": FAMILY,
        "lane": LANE,
        "games": len(p),
        "decided": int(decided.sum()),
        "pushes": int((~decided).sum()),
        "signals": names,
        "signal_meta": sig_meta,
        "coverage_by_season": coverage,
        "qw_record": record(qd, yd),
        "stack_record": record(pd_, yd),
        "accuracy_points": acc,
        "log_loss_gain": llg,
        "log_loss_decided": {"qw": float(ll(qd, yd).mean()), "stack": float(ll(pd_, yd).mean())},
        "brier_decided": {
            "qw": float(((qd - yd) ** 2).mean()),
            "stack": float(((pd_ - yd) ** 2).mean()),
        },
        "bootstrap": boot,
        "null": {
            "draws": len(null),
            "accuracy_percentile": pct(acc, 0),
            "log_loss_percentile": pct(llg, 1),
            "accuracy_null_p95": float(np.percentile(null[:, 0], 95)),
            "log_loss_null_p95": float(np.percentile(null[:, 1], 95)),
            "seconds": round(time.time() - t1, 1),
        },
        "changed_picks": int(changed.sum()),
        "changed_decided_record_stack": record(p[chd], y[chd]) if chd.any() else "0-0",
        "changed_decided_record_qw": record(qw[chd], y[chd]) if chd.any() else "0-0",
        "changed_push_games": int((changed & ~decided).sum()),
        "folds": folds,
        "nonzero_folds": dict(sorted(nonzero.items(), key=lambda kv: -kv[1])),
        "per_season": {
            int(s): {
                "qw": record(qw[(season == s) & decided], y[(season == s) & decided]),
                "stack": record(p[(season == s) & decided], y[(season == s) & decided]),
            }
            for s in seasons
        },
        "in_sample": {
            "choice": in_choice,
            "log_loss_decided_in_sample": float(ll(p_in[decided], yd).mean()),
            "log_loss_decided_out_of_sample": float(ll(pd_, yd).mean()),
            "accuracy_in_sample": float(((p_in[decided] >= 0.5) == (yd == 1)).mean()),
            "accuracy_out_of_sample": float(((pd_ >= 0.5) == (yd == 1)).mean()),
        },
        "fit_seconds": t_fit,
        "reliability_stack": reliability(pd_, yd),
        "reliability_qw": reliability(qd, yd),
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "qw_record",
                    "stack_record",
                    "accuracy_points",
                    "log_loss_gain",
                    "bootstrap",
                    "null",
                    "changed_picks",
                    "changed_decided_record_stack",
                    "changed_decided_record_qw",
                    "nonzero_folds",
                    "seconds",
                )
            },
            indent=1,
        )
    )
    print("pid", os.getpid())


if __name__ == "__main__":
    main()
