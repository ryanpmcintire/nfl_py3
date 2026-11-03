import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import sky04_lead_lag as L  # noqa: E402

from nfl_ats import mass_preserving_lattice as ML  # noqa: E402

OUT = ROOT / "artifacts" / "mkt27"
FEATURES = ROOT / "data" / "processed" / "game_features.parquet"
OPENER_GLOB = "artifacts/opener_evaluation/*/per_game.parquet"
DESCRIBE_SEASONS = range(2011, 2026)
GRADE_SEASONS = range(2020, 2026)
SPREAD_EDGES = (3.0, 6.5, 9.5)
TERCILES = 3
KEYS = (3, 7, 10, 14, 17)
NEAR_PUSH = 1.0
BOOT = 2000
PERM = 2000
SEED = 20261103
LOOKS_DESCRIBE = len(KEYS) + 4
TYPICAL_LINES = (-7.0, -3.0, 3.0, 7.0, 10.0)
TYPICAL_TOTALS = (38.0, 44.0, 51.0)
TOTAL_QUANTILES = (1 / 3, 2 / 3)
LOGIT_CLIP = 1e-4


def stat_matrix(result, line):
    absr = np.abs(result)
    resid = np.abs(result - line)
    cols = [(absr == k).astype(float) for k in KEYS]
    cols.append(np.isin(absr, KEYS).astype(float))
    cols.append(resid)
    cols.append((resid == 0).astype(float))
    cols.append((resid <= NEAR_PUSH).astype(float))
    return np.column_stack(cols)


STAT_NAMES = [f"mass_at_{k}" for k in KEYS] + [
    "mass_at_any_key",
    "abs_ats_residual",
    "push_rate",
    "near_push_rate",
]


def strat_diff(cell, grp, X):
    n_cells = int(cell.max()) + 1
    key = cell * TERCILES + grp
    counts = np.bincount(key, minlength=n_cells * TERCILES).reshape(n_cells, TERCILES)
    out = np.zeros(X.shape[1])
    ok = (counts[:, 0] > 0) & (counts[:, TERCILES - 1] > 0)
    w = (counts[:, 0] + counts[:, TERCILES - 1]) * ok
    for j in range(X.shape[1]):
        s = np.bincount(key, weights=X[:, j], minlength=n_cells * TERCILES).reshape(
            n_cells, TERCILES
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            d = s[:, TERCILES - 1] / counts[:, TERCILES - 1] - s[:, 0] / counts[:, 0]
        out[j] = np.nansum(np.where(ok, d, 0.0) * w) / w.sum()
    return out


def describe():
    g = pd.read_parquet(FEATURES)
    g = g[
        (g.game_type == "REG")
        & g.spread_line.notna()
        & g.result.notna()
        & g.total_line.notna()
        & g.season.isin(DESCRIBE_SEASONS)
    ].reset_index(drop=True)
    g["tercile"] = g.groupby("season").total_line.transform(
        lambda s: pd.qcut(s.rank(method="first"), TERCILES, labels=False)
    )
    g["sbin"] = np.digitize(np.abs(g.spread_line), SPREAD_EDGES, right=True)
    cell = pd.factorize(g.season.astype(str) + "_" + g.sbin.astype(str))[0]
    grp = g.tercile.to_numpy(int)
    X = stat_matrix(g.result.to_numpy(float), g.spread_line.to_numpy(float))
    rng = np.random.default_rng(SEED)
    obs = strat_diff(cell, grp, X)
    boot = np.zeros((BOOT, X.shape[1]))
    for b in range(BOOT):
        i = rng.integers(0, len(g), len(g))
        boot[b] = strat_diff(cell[i], grp[i], X[i])
    perm = np.zeros((PERM, X.shape[1]))
    members = [np.flatnonzero(cell == c) for c in range(cell.max() + 1)]
    for b in range(PERM):
        gp = grp.copy()
        for m in members:
            gp[m] = rng.permutation(grp[m])
        perm[b] = strat_diff(cell, gp, X)
    rows = []
    for j, name in enumerate(STAT_NAMES):
        rows.append(
            {
                "stat": name,
                "high_minus_low": obs[j],
                "lo": float(np.percentile(boot[:, j], 2.5)),
                "hi": float(np.percentile(boot[:, j], 97.5)),
                "probability_positive": float((boot[:, j] > 0).mean()),
                "perm_p_two_sided": float((np.abs(perm[:, j]) >= abs(obs[j])).mean()),
            }
        )
    tab = pd.DataFrame(rows)
    mass = (
        g.assign(**{f"m{k}": (g.result.abs() == k) for k in KEYS})
        .groupby("tercile")[[f"m{k}" for k in KEYS]]
        .mean()
    )
    total_cuts = g.groupby("tercile").total_line.agg(["min", "max", "count"])
    print("games", len(g), "looks", LOOKS_DESCRIBE)
    print(total_cuts)
    print(mass.round(4))
    print(tab.round(4).to_string())
    tab.to_csv(OUT / "describe.csv", index=False)
    mass.to_csv(OUT / "mass_by_tercile.csv")
    return g


def tuesday_totals(games):
    rows = []
    cols = ["observed_at_utc", "nflverse_game_id", "bookmaker_key", "market", "line"]
    for d in sorted(glob.glob(str(ROOT / "data/market/raw/*"))):
        name = Path(d).name
        if name[:4] not in [str(x) for x in GRADE_SEASONS] or name.endswith("futures"):
            continue
        f = Path(d) / "quotes.parquet"
        if not f.exists():
            continue
        q = pd.read_parquet(f, columns=cols)
        q = q[(q.market == "totals") & q.nflverse_game_id.notna()]
        if len(q):
            rows.append(q[["observed_at_utc", "nflverse_game_id", "bookmaker_key", "line"]])
    t = pd.concat(rows, ignore_index=True)
    t["observed_at_utc"] = pd.to_datetime(t.observed_at_utc, utc=True)
    out = {}
    gd = games.set_index("game_id").gameday.to_dict()
    for gid, grp in t.groupby("nflverse_game_id"):
        if gid not in gd:
            continue
        cut = pd.Timestamp(L.decision_ts(gd[gid]), unit="s", tz="UTC")
        sub = grp[grp.observed_at_utc <= cut]
        if sub.empty:
            continue
        out[gid] = float(sub.groupby("bookmaker_key").line.last().median())
    return pd.Series(out)


def total_band_read(pool_line, pool_total, pool_margin, cuts, line, total, point):
    tercile = int(np.digitize(total, cuts))
    pool_tercile = np.digitize(pool_total, cuts)
    band = ML.BAND_HALF_WIDTH
    while True:
        sel = (np.abs(pool_line - line) <= band) & (pool_tercile == tercile)
        if int(sel.sum()) >= ML.MIN_BAND_GAMES or band >= ML.MAX_BAND:
            break
        band = min(band + ML.BAND_STEP, ML.MAX_BAND)
    values, counts = np.unique(pool_margin[sel], return_counts=True)
    mass, _ = ML.tilted_atoms(values, counts.astype(float), line, point)
    cover = float(mass[values > line + ML._ATOM_TOLERANCE].sum())
    loss = float(mass[values < line - ML._ATOM_TOLERANCE].sum())
    push = 1.0 - cover - loss
    return cover, push, loss, int(sel.sum())


def logit(p):
    p = np.clip(p, LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


def fit_logistic(X, y, ridge=0.0):
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=1e6, max_iter=1000)
    m.fit(X, y)
    return m


def logloss(p, y):
    p = np.clip(p, LOGIT_CLIP, 1 - LOGIT_CLIP)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def grade():
    feats = pd.read_parquet(FEATURES)
    opener = pd.read_parquet(sorted(glob.glob(str(ROOT / OPENER_GLOB)))[-1])
    opener = opener[opener.season.isin(GRADE_SEASONS)].copy()
    opener_lines = opener.set_index("game_id").tue_open_home_spread
    pool = ML.prior_pool(feats, opener_lines)
    pool = pool.merge(
        feats[["game_id", "total_line"]].assign(game_id=lambda d: d.game_id.astype(str)),
        on="game_id",
        how="left",
    )
    gm = feats[["game_id", "gameday"]].assign(game_id=lambda d: d.game_id.astype(str))
    opener = opener.merge(gm, on="game_id", how="left")
    tue_total = tuesday_totals(opener)
    opener["tue_total"] = opener.game_id.map(tue_total)
    print("opener rows", len(opener), "with Tuesday total", int(opener.tue_total.notna().sum()))
    opener = opener[opener.tue_total.notna()].reset_index(drop=True)
    close_total = opener.game_id.map(
        feats.assign(game_id=feats.game_id.astype(str)).set_index("game_id").total_line
    )
    print("corr Tuesday vs closing total", float(np.corrcoef(opener.tue_total, close_total)[0, 1]))
    records = []
    for (season, week), grp in opener.groupby(["season", "week"]):
        cutoff = pd.Timestamp(pd.to_datetime(grp.gameday).min())
        elig = ML.prior_pool_for_week(
            pool,
            season=int(season),
            week=int(week),
            cutoff=cutoff,
            exclude_game_ids=set(grp.game_id.astype(str)),
        )
        elig = elig[elig.total_line.notna()]
        pl = elig.line.to_numpy(float)
        pt = elig.total_line.to_numpy(float)
        pm = elig.result.to_numpy(float)
        cuts3 = np.quantile(pt, TOTAL_QUANTILES)
        cuts2 = np.array([np.median(pt)])
        reader = ML.DiscretePushReader.for_week(
            pool,
            season=int(season),
            week=int(week),
            cutoff=cutoff,
            exclude_game_ids=set(grp.game_id.astype(str)),
        )
        for _, r in grp.iterrows():
            line = float(r.tue_open_home_spread)
            point = line + float(r.residual_at_open_served)
            s = reader.read(line, point)
            c3 = total_band_read(pl, pt, pm, cuts3, line, float(r.tue_total), point)
            c2 = total_band_read(pl, pt, pm, cuts2, line, float(r.tue_total), point)
            records.append(
                {
                    "game_id": r.game_id,
                    "season": int(season),
                    "week": int(week),
                    "line": line,
                    "result": float(r.result),
                    "total": float(r.tue_total),
                    "served_file": float(r.home_cover_probability_at_open_discrete_uncorrected),
                    "s_cover": s.cover,
                    "s_loss": s.loss,
                    "s_push": s.push,
                    "t3_cover": c3[0],
                    "t3_loss": c3[2],
                    "t3_push": c3[1],
                    "t3_games": c3[3],
                    "t2_cover": c2[0],
                    "t2_loss": c2[2],
                    "t2_push": c2[1],
                }
            )
    df = pd.DataFrame(records)
    print(
        "replication max abs diff served read vs file",
        float((df.s_cover - df.served_file).abs().max()),
    )
    df.to_csv(OUT / "grade_games.csv", index=False)
    return df


def evaluate(df):
    d = df[df.result != df.line].copy()
    d["y"] = (d.result > d.line).astype(float)
    for tag in ("s", "t3", "t2"):
        d[f"p_{tag}"] = d[f"{tag}_cover"] / (d[f"{tag}_cover"] + d[f"{tag}_loss"])
    print("graded non-push games", len(d), "of", len(df))
    pick_s = d.p_s > 0.5
    rng = np.random.default_rng(SEED)
    weeks = d.season * 100 + d.week
    wk = pd.factorize(weeks)[0]
    out = {}
    for tag in ("t3", "t2"):
        pick_t = d[f"p_{tag}"] > 0.5
        dec = pick_s != pick_t
        served_right = int(((pick_s == (d.y == 1)) & dec).sum())
        total_right = int(((pick_t == (d.y == 1)) & dec).sum())
        print(
            f"{tag} decisive games {int(dec.sum())}: served side right {served_right}, "
            f"total-conditioned side right {total_right}"
        )
        gain = logloss(d.p_s.to_numpy(), d.y.to_numpy()) - logloss(
            d[f"p_{tag}"].to_numpy(), d.y.to_numpy()
        )
        sums = np.bincount(wk, weights=gain)
        cnt = np.bincount(wk)
        bs = np.array(
            [
                sums[idx].sum() / cnt[idx].sum()
                for idx in (rng.integers(0, len(sums), len(sums)) for _ in range(BOOT))
            ]
        )
        out[f"{tag}_raw_gain"] = (
            float(gain.mean()),
            float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)),
            float((bs > 0).mean()),
        )
        print(
            f"{tag} raw log-loss gain vs served {gain.mean():+.5f} "
            f"[{np.percentile(bs, 2.5):+.5f},{np.percentile(bs, 97.5):+.5f}] "
            f"P+ {(bs > 0).mean():.3f}"
        )
    y = d.y.to_numpy()
    ll_const = []
    for s in sorted(d.season.unique()):
        te = d.season == s
        ll_const.append(logloss(np.full(te.sum(), y[~te].mean()), y[te]))
    print("intercept-only LOSO log loss", float(np.concatenate(ll_const).mean()))
    print("served raw log loss", float(logloss(d.p_s.to_numpy(), y).mean()))
    for tag in ("t3", "t2"):
        base_ll, add_ll, coefs = [], [], []
        idxs = []
        for s in sorted(d.season.unique()):
            te = (d.season == s).to_numpy()
            ls = logit(d.p_s.to_numpy())
            delta = logit(d[f"p_{tag}"].to_numpy()) - ls
            m0 = fit_logistic(ls[~te].reshape(-1, 1), y[~te])
            m1 = fit_logistic(np.column_stack([ls, delta])[~te], y[~te])
            p0 = m0.predict_proba(ls[te].reshape(-1, 1))[:, 1]
            p1 = m1.predict_proba(np.column_stack([ls, delta])[te])[:, 1]
            base_ll.append(logloss(p0, y[te]))
            add_ll.append(logloss(p1, y[te]))
            idxs.append(np.flatnonzero(te))
            coefs.append((s, float(m1.coef_[0][0]), float(m1.coef_[0][1])))
        base = np.concatenate(base_ll)
        add = np.concatenate(add_ll)
        order = np.concatenate(idxs)
        gain = np.zeros(len(d))
        gain[order] = base - add
        sums = np.bincount(wk, weights=gain)
        cnt = np.bincount(wk)
        bs = np.array(
            [
                sums[i].sum() / cnt[i].sum()
                for i in (rng.integers(0, len(sums), len(sums)) for _ in range(BOOT))
            ]
        )
        pos = sum(1 for c in coefs if c[2] > 0)
        print(
            f"{tag} LOSO add-one gain over calibrated served {gain.mean():+.5f} "
            f"[{np.percentile(bs, 2.5):+.5f},{np.percentile(bs, 97.5):+.5f}] "
            f"P+ {(bs > 0).mean():.3f}; "
            f"delta coef positive {pos}/{len(coefs)}: {[(c[0], round(c[2], 3)) for c in coefs]}"
        )
        per = pd.DataFrame({"season": d.season.to_numpy(), "g": gain}).groupby("season").g.mean()
        print("  per-season gain", per.round(5).to_dict())
        out[f"{tag}_loso_gain"] = (
            float(gain.mean()),
            float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)),
            float((bs > 0).mean()),
        )
    d.to_csv(OUT / "grade_nonpush.csv", index=False)
    (OUT / "grade_summary.json").write_text(json.dumps(out, indent=2))


def typical():
    feats = pd.read_parquet(FEATURES)
    pool = ML.prior_pool(feats)
    pool = pool.merge(
        feats[["game_id", "total_line"]].assign(game_id=lambda d: d.game_id.astype(str)),
        on="game_id",
        how="left",
    )
    pool = pool[(pool.season >= 2020 - ML.TRAILING_SEASONS) & pool.total_line.notna()]
    pool = pool[pool.season <= 2025]
    pl = pool.line.to_numpy(float)
    pt = pool.total_line.to_numpy(float)
    pm = pool.result.to_numpy(float)
    cuts = np.quantile(pt, TOTAL_QUANTILES)
    print("pool games", len(pool), "total tercile cuts", cuts.round(2))
    rows = []
    for line in TYPICAL_LINES:
        base = ML.band_read(pl, pm, line, line)
        for tot in TYPICAL_TOTALS:
            c, p, lo, n = total_band_read(pl, pt, pm, cuts, line, tot, line)
            rows.append(
                {
                    "home_line": line,
                    "total": tot,
                    "cover_served": base.cover / (base.cover + base.loss),
                    "cover_total_cond": c / (c + lo),
                    "push_served": base.push,
                    "push_total_cond": p,
                    "n": n,
                }
            )
    tab = pd.DataFrame(rows)
    tab["delta"] = tab.cover_total_cond - tab.cover_served
    print(tab.round(4).to_string())
    tab.to_csv(OUT / "typical.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1]
    if stage == "describe":
        describe()
    elif stage == "typical":
        typical()
    elif stage == "grade":
        df = grade()
        evaluate(df)
    elif stage == "evaluate":
        evaluate(pd.read_csv(OUT / "grade_games.csv"))


if __name__ == "__main__":
    main()
