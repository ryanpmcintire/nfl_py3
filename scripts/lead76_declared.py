from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from nfl_ats.mass_preserving_lattice import BAND_HALF_WIDTH, BAND_STEP, MAX_BAND, MIN_BAND_GAMES, tilted_atoms

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "lead76_declared"
OPENER = REPO / "artifacts" / "opener_evaluation" / "20261101T151614Z" / "per_game.parquet"
SCHED_NEW = REPO / "data" / "raw" / "20261027T220410Z" / "schedules.parquet"
SCHED_OLD = REPO / "artifacts" / "lead76" / "nflverse_schedules_1999_2008.parquet"
LATTICE = 80
GRID = np.arange(-LATTICE, LATTICE + 1, dtype=float)
FIT_START = 2003
GRADED = list(range(2020, 2026))
BOOT = 4000
SEED = 20261102
ERA_BREAKS = (2012, 2015, 2017, 2025)
ELO_K = 20.0
ELO_HFA = 48.0
ELO_REVERT = 1.0 / 3.0
ELO_MEAN = 1505.0
ELO_PER_POINT = 25.0


def era_of(season):
    return sum(int(season >= b) for b in ERA_BREAKS)


def elo_points(frame):
    games = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    rating = {}
    last = {}
    out = {}
    for g in games.itertuples():
        for t in (g.home_team, g.away_team):
            if t not in rating:
                rating[t] = ELO_MEAN
            elif last.get(t) is not None and last[t] != g.season:
                rating[t] = rating[t] + ELO_REVERT * (ELO_MEAN - rating[t])
            last[t] = g.season
        dh = rating[g.home_team] + ELO_HFA - rating[g.away_team]
        out[g.game_id] = dh / ELO_PER_POINT
        mov = g.home_score - g.away_score
        exp_home = 1.0 / (1.0 + 10 ** (-dh / 400.0))
        actual = 1.0 if mov > 0 else (0.5 if mov == 0 else 0.0)
        winner_diff = dh if mov > 0 else -dh
        mult = np.log(abs(mov) + 1.0) * 2.2 / (winner_diff * 0.001 + 2.2)
        delta = ELO_K * mult * (actual - exp_home)
        rating[g.home_team] += delta
        rating[g.away_team] -= delta
    return out


def load():
    new = pd.read_parquet(SCHED_NEW)
    old = pd.read_parquet(SCHED_OLD)
    allg = pd.concat([old, new[new.season <= 2025]], ignore_index=True).drop_duplicates("game_id")
    allg = allg[allg.home_score.notna()]
    elo = elo_points(allg)
    reg = allg[(allg.game_type == "REG") & allg.spread_line.notna()]
    hist = reg[reg.season <= 2019][["game_id", "season", "spread_line", "result"]].copy()
    hist["line"] = hist.spread_line.astype(float)
    hist["margin"] = np.rint(hist.result.astype(float))
    hist = hist[["game_id", "season", "line", "margin"]]
    op = pd.read_parquet(OPENER)
    op["sched_result"] = op.game_id.map(allg.set_index("game_id").result)
    op["line"] = op.tue_open_home_spread.astype(float)
    op["margin"] = np.rint(op.result.astype(float))
    op["point"] = op.line + op.residual_at_open_served
    op["elo_point"] = op.game_id.map(elo)
    return hist, op


def mass(m):
    return np.bincount((m + LATTICE).astype(int), minlength=GRID.size).astype(float)


def pmfs(lines, margins, eras, line, era):
    band = BAND_HALF_WIDTH
    while True:
        sel = np.abs(lines - line) <= band
        if int(sel.sum()) >= MIN_BAND_GAMES or band >= MAX_BAND:
            break
        band = min(band + BAND_STEP, MAX_BAND)
    pooled = mass(margins[sel])
    esel = sel & (eras == era)
    n_era = int(esel.sum())
    ec = mass(margins[esel]) if n_era else None
    return pooled / pooled.sum(), (None if ec is None else ec / ec.sum()), n_era


def tilt(base, line, point):
    keep = base > 0
    atoms, _ = tilted_atoms(GRID[keep], base[keep], line, point)
    o = np.zeros_like(base)
    o[keep] = atoms
    return o


def read(p, line):
    return (float(p[GRID > line + 1e-9].sum()), float(p[GRID < line - 1e-9].sum()))


def mix(pooled, era, w):
    return pooled if era is None else w * era + (1.0 - w) * pooled


def pool_for(hist, op, season):
    a = hist[["season", "line", "margin"]]
    b = op[op.season < season][["season", "line", "margin"]]
    p = pd.concat([a, b], ignore_index=True)
    p["era"] = p.season.map(era_of)
    return p


def build(pool, rows):
    L, M, E = pool.line.to_numpy(), pool.margin.to_numpy(), pool.era.to_numpy()
    return {g.game_id: pmfs(L, M, E, float(g.line), int(g.era)) for g in rows.itertuples()}


def fit_w(fitrows, cache):
    use = []
    for g in fitrows.itertuples():
        c = cache[g.game_id]
        m = int(g.margin)
        if c[1] is not None and c[2] > 0 and abs(m) <= LATTICE and c[0][m + LATTICE] > 0:
            use.append((c, m))
    if len(use) < 2 * MIN_BAND_GAMES:
        return None, len(use)

    def nll(w):
        return -sum(np.log(mix(c[0], c[1], w)[m + LATTICE]) for c, m in use)

    r = optimize.minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded")
    return float(r.x), len(use)


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


def boot(d, rng):
    bs = np.array([rng.choice(d, size=d.size, replace=True).mean() for _ in range(BOOT)])
    return {
        "mean": float(d.mean()),
        "lo": float(np.percentile(bs, 2.5)),
        "hi": float(np.percentile(bs, 97.5)),
        "probability_positive": float((bs > 0).mean()),
        "blocks": int(d.size),
        "positive_blocks": int((d > 0).sum()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    hist, op = load()
    op["era"] = op.season.map(era_of)
    chk = op.dropna(subset=["sched_result"])
    pop = {
        "games": int(len(op)),
        "by_season": {int(k): int(v) for k, v in op.groupby("season").size().items()},
        "result_matches_schedule": int((np.rint(chk.sched_result) == chk.margin).sum()),
        "result_checked": int(len(chk)),
        "elo_missing": int(op.elo_point.isna().sum()),
        "mean_abs_served_minus_raw_residual": float((op.residual_at_open_served - op.residual_at_open).abs().mean()),
    }
    recs = []
    folds = []
    for s in GRADED:
        pool = pool_for(hist, op, s)
        tgt = op[op.season == s]
        fit_rows = pool[(pool.season >= FIT_START) & (pool.era > 0)].reset_index(drop=True)
        fit_rows["game_id"] = [f"f{i}" for i in range(len(fit_rows))]
        fcache = {}
        for season, grp in fit_rows.groupby("season"):
            pr = pool[pool.season < season]
            L, M, E = pr.line.to_numpy(), pr.margin.to_numpy(), pr.era.to_numpy()
            for g in grp.itertuples():
                fcache[g.game_id] = pmfs(L, M, E, float(g.line), int(g.era))
        w, nfit = fit_w(fit_rows, fcache)
        folds.append({"season": s, "w": w, "fit_games": nfit, "pool_games": int(len(pool))})
        cache = build(pool, tgt)
        for g in tgt.itertuples():
            pooled, era, n_era = cache[g.game_id]
            wu = w if (w is not None and era is not None) else 0.0
            pm = mix(pooled, era, wu)
            tm = tilt(pooled, g.line, g.line)
            tp = tilt(pooled, g.line, float(g.point))
            tr = tilt(pm, g.line, float(g.point))
            te = tilt(pooled, g.line, float(g.elo_point))
            m = int(g.margin)
            ok = abs(m) <= LATTICE and pooled[m + LATTICE] > 0
            nonpush = abs(g.margin - g.line) > 1e-9

            def nz(r):
                return r[0] / (r[0] + r[1])

            recs.append(
                {
                    "game_id": g.game_id, "season": s, "n_era": n_era, "w": wu, "line": g.line, "margin": g.margin,
                    "nonpush": nonpush, "y": float(g.margin > g.line),
                    "p_served": float(g.home_cover_probability_excluding_push_at_open),
                    "p_mkt": nz(read(tm, g.line)), "p_pool": nz(read(tp, g.line)), "p_rule": nz(read(tr, g.line)),
                    "p_elo": nz(read(te, g.line)), "ls_ok": bool(ok),
                    "lm_mkt": float(np.log(tm[m + LATTICE])) if ok else np.nan,
                    "lm_pool": float(np.log(tp[m + LATTICE])) if ok else np.nan,
                    "lm_rule": float(np.log(tr[m + LATTICE])) if ok else np.nan,
                    "lm_elo": float(np.log(te[m + LATTICE])) if ok else np.nan,
                }
            )
    df = pd.DataFrame(recs)
    df.to_parquet(OUT / "game_level.parquet")
    pd.DataFrame(folds).to_csv(OUT / "fold_w.csv", index=False)
    base = df[df.nonpush].copy()
    for c, p in (("x_s", "p_served"), ("x_p", "p_pool"), ("x_r", "p_rule")):
        base[c] = logit(base[p].to_numpy())
    base["x_d"] = base.x_r - base.x_p
    specs = (("cal", ["x_s"]), ("term", ["x_s", "x_d"]), ("joint", ["x_s", "x_p", "x_r"]))
    isr, coefs = [], []

    def run_scheme(name, train_seasons):
        for s in GRADED:
            trn_s = train_seasons(s)
            if len(trn_s) < 2:
                continue
            trn = base[base.season.isin(trn_s)]
            te = base.season == s
            for nm, cols in specs:
                b = fit_logit(trn[cols].to_numpy(), trn.y.to_numpy())
                base.loc[te, f"{nm}_{name}"] = 1 / (1 + np.exp(-(base.loc[te, cols].to_numpy() @ b)))
                ptr = 1 / (1 + np.exp(-(trn[cols].to_numpy() @ b)))
                yt = trn.y.to_numpy()
                isr.append({"scheme": name, "season": s, "model": nm, "is_logloss": ll(ptr, yt), "is_brier": brier(ptr, yt), "is_acc": float(((ptr > 0.5) == (yt > 0.5)).mean())})
                coefs.append({"scheme": name, "season": s, "model": nm, **{c: float(v) for c, v in zip(cols, b)}})

    run_scheme("chron", lambda s: [t for t in GRADED if t < s])
    run_scheme("loso", lambda s: [t for t in GRADED if t != s])
    pd.DataFrame(coefs).to_csv(OUT / "fold_coefficients.csv", index=False)
    isd = pd.DataFrame(isr)
    ev = base[base.cal_chron.notna()].copy()
    yv = ev.y.to_numpy()
    summ = {}
    sets = {
        "all6_unfitted": (base, ["p_mkt", "p_served", "p_pool", "p_rule", "p_elo"]),
        "eval_chron": (ev, ["p_mkt", "p_served", "p_elo", "p_pool", "p_rule", "cal_chron", "term_chron", "joint_chron"]),
        "all6_loso": (base, ["p_mkt", "p_served", "p_elo", "p_pool", "p_rule", "cal_loso", "term_loso", "joint_loso"]),
    }
    cells = 0
    for sname, (fr, cols) in sets.items():
        yy = fr.y.to_numpy()
        for c in cols:
            pv = fr[c].to_numpy()
            r = {"n": int(len(fr)), "acc": float(((pv > 0.5) == (yy > 0.5)).mean()), "logloss": ll(pv, yy), "brier": brier(pv, yy)}
            cells += 3
            if c.endswith(("_chron", "_loso")):
                sub = isd[(isd.scheme == c.split("_")[-1]) & (isd.model == c.split("_")[0])]
                r["is_logloss"] = float(sub.is_logloss.mean())
                r["is_brier"] = float(sub.is_brier.mean())
                r["is_acc"] = float(sub.is_acc.mean())
                r["gap_logloss"] = r["logloss"] - r["is_logloss"]
                r["gap_brier"] = r["brier"] - r["is_brier"]
                cells += 3
            summ[f"{sname}:{c}"] = r
    dec = {}
    for sname, (fr, cols) in sets.items():
        for c in cols:
            if c == "p_served":
                continue
            d = fr[(fr[c] > 0.5) != (fr.p_served > 0.5)]
            k = int(((d[c] > 0.5) == (d.y > 0.5)).sum())
            dec[f"{sname}:{c}"] = {"differs_from_served": int(len(d)), "new_side_correct": k, "served_side_correct": int(len(d) - k)}
            cells += 1
    boots = {}

    def pair(fr, a, ref, tag):
        nonlocal cells
        for nm, fn in (("log_loss_improvement", ll), ("brier_improvement", brier)):
            d = np.array([fn(g_[ref].to_numpy(), g_.y.to_numpy()) - fn(g_[a].to_numpy(), g_.y.to_numpy()) for _, g_ in fr.groupby("season")])
            boots[f"{tag}:{a}_vs_{ref}_{nm}"] = boot(d, rng)
            cells += 1

    for a, ref in (("p_rule", "p_pool"), ("p_pool", "p_mkt"), ("p_rule", "p_mkt"), ("p_served", "p_mkt"), ("p_elo", "p_mkt"), ("p_pool", "p_served")):
        pair(base, a, ref, "all6")
    for a, ref in (("term_chron", "cal_chron"), ("joint_chron", "cal_chron"), ("cal_chron", "p_mkt"), ("term_chron", "p_mkt"), ("term_chron", "p_elo")):
        pair(ev, a, ref, "eval_chron")
    for a, ref in (("term_loso", "cal_loso"), ("joint_loso", "cal_loso"), ("cal_loso", "p_mkt"), ("term_loso", "p_mkt")):
        pair(base, a, ref, "all6_loso")
    mg = df[df.ls_ok]
    for a, ref in (("lm_pool", "lm_mkt"), ("lm_rule", "lm_pool"), ("lm_rule", "lm_mkt"), ("lm_elo", "lm_mkt")):
        d = np.array([(g_[a] - g_[ref]).mean() for _, g_ in mg.groupby("season")])
        r = boot(d, rng)
        r["games"] = int(len(mg))
        r["dropped_out_of_support"] = int((~df.ls_ok).sum())
        boots[f"margin_logscore:{a}_vs_{ref}"] = r
        cells += 1
    rel = []
    for m_ in sets["eval_chron"][1]:
        src = base[m_].dropna() if m_.endswith(("_chron", "_loso")) else base[base.season < 2022][m_]
        edges = np.quantile(src, [0.2, 0.4, 0.6, 0.8])
        band = np.digitize(ev[m_].to_numpy(), edges)
        for b_ in range(5):
            sel = band == b_
            if sel.any():
                rel.append({"model": m_, "band": b_ + 1, "n": int(sel.sum()), "mean_p": float(ev[m_].to_numpy()[sel].mean()), "observed": float(yv[sel].mean())})
                cells += 1
    pd.DataFrame(rel).to_csv(OUT / "reliability.csv", index=False)
    rows = []
    for s_, t in base.groupby("season"):
        yy = t.y.to_numpy() > 0.5
        r = {"season": s_, "n": len(t)}
        for c in ("p_mkt", "p_served", "p_elo", "p_pool", "p_rule", "term_chron", "term_loso"):
            if t[c].notna().all():
                r[f"acc_{c}"] = float(((t[c].to_numpy() > 0.5) == yy).mean())
                r[f"ll_{c}"] = ll(t[c].to_numpy(), t.y.to_numpy())
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "by_season.csv", index=False)
    out = {"population": pop, "folds": folds, "summary": summ, "decisive_vs_served": dec, "bootstrap": boots, "reported_cells": cells}
    (OUT / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"population": pop, "folds": folds, "summary": summ, "decisive": dec, "reported_cells": cells}, indent=1))
    print(json.dumps(boots, indent=1))
    print(pd.DataFrame(rows).round(4).to_string())
    print(pd.DataFrame(coefs).round(3).to_string())
    print(pd.DataFrame(rel).round(3).to_string())


if __name__ == "__main__":
    main()
