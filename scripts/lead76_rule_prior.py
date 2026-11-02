from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from nfl_ats.mass_preserving_lattice import (
    BAND_HALF_WIDTH,
    BAND_STEP,
    MAX_BAND,
    MIN_BAND_GAMES,
    tilted_atoms,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "lead76"
SCHEDULE_NEW = REPO / "data" / "raw" / "20261027T220410Z" / "schedules.parquet"
SCHEDULE_OLD = OUT / "nflverse_schedules_1999_2008.parquet"
OPENER_EVAL = REPO / "artifacts" / "sbr_era_opener_eval" / "20260819T233013Z" / "scored.parquet"
LATTICE = 80
GRID = np.arange(-LATTICE, LATTICE + 1, dtype=float)
FIRST_FOLD = 2013
FIT_START = 2003
BOOT = 4000
SEED = 20261102
KEYS = (3, 7, 10, 14, 17)
ERA_BREAKS = {"pat_15yd": 2015, "ot_modified": 2012, "ot_10min": 2017}


def era_of(season: int) -> int:
    return (
        int(season >= ERA_BREAKS["ot_modified"])
        + int(season >= ERA_BREAKS["pat_15yd"])
        + int(season >= ERA_BREAKS["ot_10min"])
    )


def load_pool() -> pd.DataFrame:
    new = pd.read_parquet(SCHEDULE_NEW)
    old = pd.read_parquet(SCHEDULE_OLD)
    frame = pd.concat([old, new[new.season <= 2021]], ignore_index=True)
    frame = frame[(frame.game_type == "REG") & frame.result.notna() & frame.spread_line.notna()]
    frame = frame.drop_duplicates("game_id")
    pool = frame[["game_id", "season", "week", "spread_line", "result"]].copy()
    pool["line"] = pool.spread_line.astype(float)
    pool["margin"] = np.rint(pool.result.astype(float))
    pool["era"] = pool.season.map(era_of)
    return pool.reset_index(drop=True)


def mass(margins: np.ndarray) -> np.ndarray:
    idx = (margins + LATTICE).astype(int)
    return np.bincount(idx, minlength=GRID.size).astype(float)


def pmfs(pool_lines, pool_margins, pool_eras, line, era):
    band = BAND_HALF_WIDTH
    while True:
        sel = np.abs(pool_lines - line) <= band
        if int(sel.sum()) >= MIN_BAND_GAMES or band >= MAX_BAND:
            break
        band = min(band + BAND_STEP, MAX_BAND)
    pooled = mass(pool_margins[sel])
    esel = sel & (pool_eras == era)
    n_era = int(esel.sum())
    era_counts = mass(pool_margins[esel]) if n_era else None
    return pooled / pooled.sum(), (None if era_counts is None else era_counts / era_counts.sum()), n_era


def tilt(base: np.ndarray, line: float, point: float) -> np.ndarray:
    keep = base > 0
    atoms, _ = tilted_atoms(GRID[keep], base[keep], line, point)
    out = np.zeros_like(base)
    out[keep] = atoms
    return out


def read(p: np.ndarray, line: float) -> tuple[float, float, float]:
    return (
        float(p[GRID > line + 1e-9].sum()),
        float(p[np.abs(GRID - line) < 1e-9].sum()),
        float(p[GRID < line - 1e-9].sum()),
    )


def build_game_pmfs(pool: pd.DataFrame, games: pd.DataFrame):
    rows = {}
    for season, grp in games.groupby("season"):
        prior = pool[pool.season < season]
        lines, margins, eras = (
            prior.line.to_numpy(),
            prior.margin.to_numpy(),
            prior.era.to_numpy(),
        )
        for g in grp.itertuples():
            rows[g.game_id] = pmfs(lines, margins, eras, float(g.line), int(g.era))
    return rows


def mix(pooled, era, w):
    return pooled if era is None else w * era + (1.0 - w) * pooled


def fit_w(fit_games, cache):
    use = [(cache[g.game_id], int(g.margin)) for g in fit_games.itertuples()]
    use = [(c, m) for c, m in use if c[1] is not None and c[2] > 0 and abs(m) <= LATTICE and c[0][m + LATTICE] > 0]
    if len(use) < 2 * MIN_BAND_GAMES:
        return None, len(use)

    def nll(w):
        return -sum(np.log(mix(c[0], c[1], w)[m + LATTICE]) for c, m in use)

    res = optimize.minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded")
    return float(res.x), len(use)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logit(x, y):
    def f(b):
        z = x @ b
        return float(np.sum(np.logaddexp(0, z) - y * z))

    return optimize.minimize(f, np.zeros(x.shape[1]), method="BFGS").x


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def wilson(k, n):
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def block_boot(d, rng):
    bs = np.array([rng.choice(d, size=d.size, replace=True).mean() for _ in range(BOOT)])
    return {
        "mean_by_season": float(d.mean()),
        "lo": float(np.percentile(bs, 2.5)),
        "hi": float(np.percentile(bs, 97.5)),
        "probability_positive": float((bs > 0).mean()),
        "blocks": int(d.size),
        "positive_seasons": int((d > 0).sum()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pool = load_pool()
    scored = pd.read_parquet(OPENER_EVAL)
    scored = scored[~scored.open_ambiguous].copy()
    scored["line"] = scored.proxy_open_home_spread.astype(float)
    scored["margin"] = np.rint(scored.result.astype(float))
    scored["era"] = scored.season.map(era_of)
    scored["point"] = scored.line + scored.residual_at_open_proxy
    cache = build_game_pmfs(pool, scored)
    fit_pool = pool[(pool.season >= FIT_START) & (pool.season <= 2021)]
    fit_cache = build_game_pmfs(pool, fit_pool)

    era_tab = []
    for era, grp in pool.groupby("era"):
        row = {"era": int(era), "seasons": f"{grp.season.min()}-{grp.season.max()}", "games": len(grp)}
        am = np.abs(grp.margin.to_numpy())
        for k in KEYS:
            hit = int((am == k).sum())
            lo, hi = wilson(hit, len(grp))
            row[f"mass_{k}"] = hit / len(grp)
            row[f"lo_{k}"] = lo
            row[f"hi_{k}"] = hi
        era_tab.append(row)
    era_df = pd.DataFrame(era_tab)
    era_df.to_csv(OUT / "key_mass_by_era.csv", index=False)

    pat_pre = np.abs(pool[pool.season < 2015].margin.to_numpy())
    pat_post = np.abs(pool[pool.season >= 2015].margin.to_numpy())
    allm = np.concatenate([pat_pre, pat_post])
    perm = []
    for k in KEYS:
        hits = (allm == k).astype(float)
        n_post = len(pat_post)
        obs = float((pat_post == k).mean() - (pat_pre == k).mean())
        diffs = np.empty(BOOT)
        for i in range(BOOT):
            shuf = rng.permutation(hits)
            diffs[i] = shuf[:n_post].mean() - shuf[n_post:].mean()
        perm.append({"key": k, "post_minus_pre": obs, "perm_p_two_sided": float((np.abs(diffs) >= abs(obs)).mean())})
    pd.DataFrame(perm).to_csv(OUT / "pat_era_permutation.csv", index=False)

    recs = []
    fold_log = []
    for s in range(FIRST_FOLD, 2022):
        tgt = scored[scored.season == s]
        w, n_fit = fit_w(fit_pool[(fit_pool.season < s) & (fit_pool.era > 0)], fit_cache)
        for g in tgt.itertuples():
            pooled, era, n_era = cache[g.game_id]
            wu = w if (w is not None and era is not None) else 0.0
            pm = mix(pooled, era, wu)
            point = float(g.point)
            tmk = tilt(pooled, g.line, g.line)
            tp = tilt(pooled, g.line, point)
            tm = tilt(pm, g.line, point)
            r_mkt, r_pool, r_rule = read(tmk, g.line), read(tp, g.line), read(tm, g.line)
            m = int(g.margin)
            ok = abs(m) <= LATTICE and pooled[m + LATTICE] > 0
            key_sel = np.zeros(GRID.size, dtype=bool)
            for k in KEYS:
                key_sel |= np.abs(np.abs(GRID) - k) < 1e-9
            recs.append(
                {
                    "game_id": g.game_id, "season": s, "era": g.era, "n_era": n_era, "w": wu,
                    "line": g.line, "margin": g.margin,
                    "nonpush": abs(g.margin - g.line) > 1e-9,
                    "home_cover": float(g.margin > g.line),
                    "p_served": float(g.home_cover_probability_at_open_proxy),
                    "p_mkt": r_mkt[0] / (r_mkt[0] + r_mkt[2]),
                    "p_pool": r_pool[0] / (r_pool[0] + r_pool[2]),
                    "p_rule": r_rule[0] / (r_rule[0] + r_rule[2]),
                    "ls_ok": bool(ok),
                    "lm_mkt": float(np.log(tmk[m + LATTICE])) if ok else np.nan,
                    "lm_pool": float(np.log(tp[m + LATTICE])) if ok else np.nan,
                    "lm_rule": float(np.log(tm[m + LATTICE])) if ok else np.nan,
                    "key_rule_minus_pool": float(tm[key_sel].sum() - tp[key_sel].sum()),
                }
            )
        fold_log.append({"season": s, "w": w, "fit_games": n_fit})
    df = pd.DataFrame(recs)
    df.to_parquet(OUT / "game_level.parquet")
    pd.DataFrame(fold_log).to_csv(OUT / "fold_w.csv", index=False)

    base = df[df.nonpush].copy()
    base["y"] = base.home_cover
    base["x_s"] = logit(base.p_served.to_numpy())
    base["x_p"] = logit(base.p_pool.to_numpy())
    base["x_r"] = logit(base.p_rule.to_numpy())
    base["x_d"] = base.x_r - base.x_p
    base["p_term"] = np.nan
    base["p_joint"] = np.nan
    base["p_cal"] = np.nan
    coefs = []
    is_rows = []
    for s in range(FIRST_FOLD, 2022):
        trn = base[base.season < s]
        if trn.season.nunique() < 2:
            continue
        te = base.season == s
        for name, cols, out in (("cal", ["x_s"], "p_cal"), ("term", ["x_s", "x_d"], "p_term"), ("joint", ["x_s", "x_p", "x_r"], "p_joint")):
            b = fit_logit(trn[cols].to_numpy(), trn.y.to_numpy())
            base.loc[te, out] = 1 / (1 + np.exp(-(base.loc[te, cols].to_numpy() @ b)))
            ptr = 1 / (1 + np.exp(-(trn[cols].to_numpy() @ b)))
            ytr = trn.y.to_numpy()
            is_rows.append({"season": s, "model": name, "is_logloss": ll(ptr, ytr), "is_brier": brier(ptr, ytr), "is_acc": float(((ptr > 0.5) == (ytr > 0.5)).mean())})
            coefs.append({"season": s, "model": name, **{c: float(v) for c, v in zip(cols, b)}})
    pd.DataFrame(coefs).to_csv(OUT / "fold_coefficients.csv", index=False)
    ev = base[base.p_term.notna()].copy()
    isd = pd.DataFrame(is_rows)
    models = ["p_mkt", "p_served", "p_cal", "p_pool", "p_rule", "p_term", "p_joint"]
    yv = ev.y.to_numpy()
    summ = {}
    for m_ in models:
        pv = ev[m_].to_numpy()
        summ[m_] = {"n": int(len(ev)), "acc": float(((pv > 0.5) == (yv > 0.5)).mean()), "logloss": ll(pv, yv), "brier": brier(pv, yv)}
    for m_, nm in (("p_cal", "cal"), ("p_term", "term"), ("p_joint", "joint")):
        sub = isd[isd.model == nm]
        summ[m_]["is_logloss_mean"] = float(sub.is_logloss.mean())
        summ[m_]["gap_logloss"] = summ[m_]["logloss"] - summ[m_]["is_logloss_mean"]
        summ[m_]["is_brier_mean"] = float(sub.is_brier.mean())
        summ[m_]["gap_brier"] = summ[m_]["brier"] - summ[m_]["is_brier_mean"]
        summ[m_]["is_acc_mean"] = float(sub.is_acc.mean())

    flipped = {}
    for m_ in ("p_cal", "p_term", "p_joint", "p_pool", "p_rule"):
        d = ev[(ev[m_] > 0.5) != (ev.p_served > 0.5)]
        k_new = int(((d[m_] > 0.5) == (d.y > 0.5)).sum())
        flipped[m_] = {"differs_from_served": int(len(d)), "new_side_correct": k_new, "served_side_correct": int(len(d) - k_new)}

    boots = {}
    for a in ("p_cal", "p_pool", "p_rule", "p_term", "p_joint"):
        refs = ["p_served"] + (["p_pool"] if a == "p_rule" else []) + (["p_cal"] if a in ("p_term", "p_joint") else [])
        for ref in refs:
            for nm, fn in (("log_loss_improvement", ll), ("brier_improvement", brier)):
                d = np.array([fn(g_[ref].to_numpy(), g_.y.to_numpy()) - fn(g_[a].to_numpy(), g_.y.to_numpy()) for _, g_ in ev.groupby("season")])
                boots[f"{a}_vs_{ref}_{nm}"] = block_boot(d, rng)
    mg = df[df.ls_ok]
    for a, ref in (("lm_pool", "lm_mkt"), ("lm_rule", "lm_pool"), ("lm_rule", "lm_mkt")):
        d = np.array([(g_[a] - g_[ref]).mean() for _, g_ in mg.groupby("season")])
        r = block_boot(d, rng)
        r["games"] = int(len(mg))
        r["dropped_out_of_support"] = int((~df.ls_ok).sum())
        boots[f"margin_logscore_{a}_vs_{ref}"] = r
    rel = []
    trn_all = base[base.season < FIRST_FOLD + 1]
    for m_ in models:
        src = trn_all[m_] if m_ in ("p_served", "p_pool", "p_rule", "p_mkt") else base[m_]
        edges = np.quantile(src.dropna(), [0.2, 0.4, 0.6, 0.8])
        band = np.digitize(ev[m_].to_numpy(), edges)
        for b_ in range(5):
            sel = band == b_
            if sel.any():
                rel.append({"model": m_, "band": b_ + 1, "n": int(sel.sum()), "mean_p": float(ev[m_].to_numpy()[sel].mean()), "observed": float(yv[sel].mean())})
    pd.DataFrame(rel).to_csv(OUT / "reliability.csv", index=False)
    rows = []
    for s_, t in ev.groupby("season"):
        yy = t.y.to_numpy() > 0.5
        rows.append({"season": s_, "n": len(t), **{f"acc_{n}": float(((t[c].to_numpy() > 0.5) == yy).mean()) for n, c in (("served", "p_served"), ("pool", "p_pool"), ("rule", "p_rule"), ("term", "p_term"))}})
    pd.DataFrame(rows).to_csv(OUT / "by_season.csv", index=False)
    out = {
        "summary": summ, "decisive_vs_served": flipped, "bootstrap": boots,
        "fold_w": fold_log, "eval_games": int(len(ev)), "eval_seasons": sorted(int(x) for x in ev.season.unique()),
        "pat_permutation": perm,
        "mean_abs_key_mass_shift_rule_vs_pool": float(df.key_rule_minus_pool.abs().mean()),
        "pool_games": int(len(pool)), "pool_seasons": [int(pool.season.min()), int(pool.season.max())],
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("summary", "decisive_vs_served", "eval_games", "eval_seasons", "pool_games", "mean_abs_key_mass_shift_rule_vs_pool")}, indent=1))
    print(json.dumps(boots, indent=1))
    print(json.dumps(fold_log))
    print(era_df.round(4).to_string())
    print(pd.DataFrame(perm).round(4).to_string())


if __name__ == "__main__":
    main()
