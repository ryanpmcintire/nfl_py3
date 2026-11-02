import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "sky04"
PERGAME = ROOT / "artifacts" / "pick_probability" / "20260929T192747Z" / "per_game.parquet"
SEASONS = ("2023", "2024", "2025", "2026")
ET = "America/New_York"
TUESDAY = 1
NOON = 12
BOOT = 2000
SEED = 20261102
MIN_PARTICIPATION = 2500
LAGS = (-2, -1, 0, 1, 2)


def book_series():
    cache = OUT / "book_series.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    cols = ["observed_at_utc", "nflverse_game_id", "bookmaker_key", "bookmaker_last_update_utc", "market", "home_spread_line"]
    for d in sorted(glob.glob(str(ROOT / "data/market/raw/*"))):
        name = Path(d).name
        if name[:4] not in SEASONS or name.endswith("futures"):
            continue
        f = Path(d) / "quotes.parquet"
        if not f.exists():
            continue
        q = pd.read_parquet(f, columns=cols)
        q = q[(q.market == "spreads") & q.nflverse_game_id.notna()]
        if len(q):
            rows.append(q.groupby(["nflverse_game_id", "bookmaker_key", "observed_at_utc"], as_index=False).agg(
                home_spread_line=("home_spread_line", "median"), upd=("bookmaker_last_update_utc", "max")))
    s = pd.concat(rows, ignore_index=True)
    s["observed_at_utc"] = pd.to_datetime(s.observed_at_utc, utc=True)
    s["upd"] = pd.to_datetime(s.upd, utc=True)
    s = s.drop_duplicates(["nflverse_game_id", "bookmaker_key", "observed_at_utc"])
    s.to_parquet(cache)
    return s


def game_mats(s, books):
    bi = {b: i for i, b in enumerate(books)}
    out = {}
    s = s[s.bookmaker_key.isin(bi)]
    for gid, g in s.groupby("nflverse_game_id"):
        obs = np.sort(g.observed_at_utc.unique())
        if len(obs) < 3:
            continue
        oi = {t: i for i, t in enumerate(obs)}
        L = np.full((len(obs), len(books)), np.nan)
        U = np.full((len(obs), len(books)), np.nan)
        r = g.observed_at_utc.map(oi).values
        c = g.bookmaker_key.map(bi).values
        L[r, c] = g.home_spread_line.values
        U[r, c] = g.upd.map(lambda t: t.timestamp()).values
        out[gid] = (np.array([pd.Timestamp(t).timestamp() for t in obs]), L, U)
    return out


def cadence(mats):
    gaps = np.concatenate([np.diff(o) for o, _, _ in mats.values()])
    q = np.percentile(gaps, [5, 25, 50, 75, 95])
    return {"snapshots_per_game_median": float(np.median([len(o) for o, _, _ in mats.values()])),
            "gap_seconds_q05_q25_q50_q75_q95": [float(x) for x in q], "gap_n": int(len(gaps)),
            "share_gaps_le_600s": float((gaps <= 600).mean()), "share_gaps_le_3600s": float((gaps <= 3600).mean())}


def xcorr(mats, books):
    K = len(books)
    res = {}
    for lag in LAGS:
        sxy = np.zeros((K, K)); sx = np.zeros((K, K)); sy = np.zeros((K, K)); sxx = np.zeros((K, K)); syy = np.zeros((K, K)); n = np.zeros((K, K))
        for o, L, U in mats.values():
            D = np.diff(L, axis=0)
            T = D.shape[0]
            if abs(lag) >= T:
                continue
            X = D[: T - lag] if lag >= 0 else D[-lag:]
            Y = D[lag:] if lag >= 0 else D[: T + lag]
            Mx = (~np.isnan(X)).astype(float); My = (~np.isnan(Y)).astype(float)
            X0 = np.nan_to_num(X); Y0 = np.nan_to_num(Y)
            sxy += X0.T @ Y0
            sx += X0.T @ My
            sy += Mx.T @ Y0
            sxx += (X0 ** 2).T @ My
            syy += Mx.T @ (Y0 ** 2)
            n += Mx.T @ My
        with np.errstate(all="ignore"):
            cov = sxy / n - (sx / n) * (sy / n)
            vx = sxx / n - (sx / n) ** 2
            vy = syy / n - (sy / n) ** 2
            res[lag] = (cov / np.sqrt(vx * vy), n)
    return res


def diffusion(mats, books):
    K = len(books)
    rec = []
    for gid, (o, L, U) in mats.items():
        T = L.shape[0]
        for a in range(K):
            mv = [k for k in range(1, T) if not np.isnan(L[k, a]) and not np.isnan(L[k - 1, a]) and L[k, a] != L[k - 1, a]]
            for ix, k in enumerate(mv):
                end = mv[ix + 1] if ix + 1 < len(mv) else T
                d = np.sign(L[k, a] - L[k - 1, a])
                for b in range(K):
                    if b == a or np.isnan(L[k - 1, b]):
                        continue
                    if L[k - 1, b] == L[k, a] or np.sign(L[k, a] - L[k - 1, b]) != d:
                        rec.append((a, b, 2, np.nan, np.nan))
                        continue
                    hit = None
                    for j in range(k, end):
                        if not np.isnan(L[j, b]) and np.sign(L[j, b] - L[k - 1, b]) == d:
                            hit = j
                            break
                    if hit is None:
                        rec.append((a, b, 0, np.nan, np.nan))
                    else:
                        rec.append((a, b, 1, U[hit, b] - U[k, a], o[hit] - o[k]))
    return pd.DataFrame(rec, columns=["a", "b", "status", "lag_provider_s", "lag_snapshot_s"])


def pair_table(df, books):
    rows = []
    for (a, b), g in df.groupby(["a", "b"]):
        m = g[g.status == 1]
        n_open = int((g.status != 2).sum())
        rows.append({"leader": books[a], "follower": books[b], "a_moves": int(len(g)), "b_already_there": int((g.status == 2).sum()),
                     "diffusion_share": float(len(m) / n_open) if n_open else None, "matched": int(len(m)),
                     "median_lag_provider_s": float(m.lag_provider_s.median()) if len(m) else None,
                     "median_lag_snapshot_s": float(m.lag_snapshot_s.median()) if len(m) else None,
                     "share_same_snapshot": float((m.lag_snapshot_s == 0).mean()) if len(m) else None})
    return pd.DataFrame(rows)


def leadership(mats, books, gids):
    K = len(books)
    part = np.zeros(K); cred = np.zeros(K)
    for gid in gids:
        if gid not in mats:
            continue
        o, L, U = mats[gid]
        for k in range(1, L.shape[0]):
            ch = np.where(~np.isnan(L[k]) & ~np.isnan(L[k - 1]) & (L[k] != L[k - 1]))[0]
            if len(ch) == 0:
                continue
            part[ch] += 1
            t = U[k, ch]
            first = ch[t == np.nanmin(t)]
            cred[first] += 1 / len(first)
    return cred / np.maximum(part, 1)


def decision_ts(gameday):
    d = pd.Timestamp(gameday)
    off = (d.weekday() - TUESDAY) % 7
    t = (d - pd.Timedelta(days=off)).replace(hour=NOON).tz_localize(ET)
    return t.tz_convert("UTC").timestamp()


def feature(mats, gids, gamedays, lead_idx, foll_idx):
    x = np.zeros(len(gids)); has = np.zeros(len(gids), bool)
    for i, (gid, gd) in enumerate(zip(gids, gamedays)):
        if gid not in mats:
            continue
        o, L, U = mats[gid]
        k = np.searchsorted(o, decision_ts(gd), side="right") - 1
        if k < 0:
            continue
        ll = L[k, lead_idx]; ff = L[k, foll_idx]
        ll = ll[~np.isnan(ll)]; ff = ff[~np.isnan(ff)]
        if len(ll) == 0 or len(ff) == 0:
            continue
        x[i] = np.median(ff) - np.median(ll)
        has[i] = True
    return x, has


def logit(p):
    return np.log(p / (1 - p))


def fit_b(off, x, y):
    b = 0.0
    for _ in range(50):
        pr = 1 / (1 + np.exp(-(off + b * x)))
        g = np.sum((y - pr) * x)
        h = -np.sum(pr * (1 - pr) * x * x)
        if h == 0:
            return 0.0
        step = g / h
        b -= step
        if abs(step) < 1e-10:
            break
    return b


def boot(vals, rng):
    idx = rng.integers(0, len(vals), size=(BOOT, len(vals)))
    m = vals[idx].mean(axis=1)
    return [float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m > 0).mean())]


def fitted_term(mats, books):
    g = pd.read_parquet(PERGAME)
    g = g[(g.game_type == "REG") & g.season.isin([2023, 2024, 2025])]
    sub = g[g.margin_vs_open != 0].copy().reset_index(drop=True)
    gids = sub.game_id.values
    gd = sub.gameday.values
    y = sub.home_covered.astype(float).values
    pr = sub.out_of_season_home_probability.clip(1e-6, 1 - 1e-6).values
    off = logit(pr)
    seasons = sub.season.values
    x = np.zeros(len(sub)); has = np.zeros(len(sub), bool)
    leaders = {}
    for s in (2023, 2024, 2025):
        tr_g = list(set(gids[seasons != s]))
        share = leadership(mats, books, tr_g)
        li = [i for i in range(len(books)) if share[i] > 0.5]
        fi = [i for i in range(len(books)) if share[i] <= 0.5]
        leaders[int(s)] = [books[i] for i in li]
        te = seasons == s
        xs, hs = feature(mats, gids[te], gd[te], li, fi)
        x[te] = xs; has[te] = hs
    elig = has
    res = {"leaders_by_heldout_season": leaders, "games_scored": int(elig.sum()), "games_total": int(len(sub)),
           "x_nonzero_games": int((x[elig] != 0).sum()),
           "x_nonzero_by_season": {int(s): int(((seasons == s) & elig & (x != 0)).sum()) for s in (2023, 2024, 2025)},
           "x_sd_points": float(np.std(x[elig]))}
    pooled = np.zeros(len(sub)); ins = np.zeros(len(sub)); fc = {}
    for s in (2023, 2024, 2025):
        te = (seasons == s) & elig
        tr = (seasons != s) & elig
        b = fit_b(off[tr], x[tr], y[tr])
        fc[int(s)] = float(b)
        pooled[te] = 1 / (1 + np.exp(-(off[te] + b * x[te])))
    b_all = fit_b(off[elig], x[elig], y[elig])
    ins[elig] = 1 / (1 + np.exp(-(off[elig] + b_all * x[elig])))
    base = pr[elig]; yy = y[elig]
    rng = np.random.default_rng(SEED)
    for label, arr in (("oos", pooled[elig]), ("ins", ins[elig])):
        d_brier = (base - yy) ** 2 - (arr - yy) ** 2
        d_ll = -(yy * np.log(base) + (1 - yy) * np.log(1 - base)) + (yy * np.log(arr) + (1 - yy) * np.log(1 - arr))
        d_acc = (((arr > 0.5) == (yy == 1)).astype(float) - ((base > 0.5) == (yy == 1)).astype(float)) * 100
        dec = (base > 0.5) != (arr > 0.5)
        res[label] = {"brier_gain": boot(d_brier, rng), "logloss_gain": boot(d_ll, rng), "accuracy_gain_points": boot(d_acc, rng),
                      "decisive_games": int(dec.sum()), "decisive_new_correct": int(((arr > 0.5) == (yy == 1))[dec].sum()),
                      "decisive_base_correct": int(((base > 0.5) == (yy == 1))[dec].sum()), "n": int(elig.sum())}
    res["gap_brier_gain_ins_minus_oos"] = res["ins"]["brier_gain"][0] - res["oos"]["brier_gain"][0]
    res["fold_coefficients"] = fc
    res["pooled_coefficient"] = float(b_all)
    res["corr_x_with_market_move_toward_home"] = float(np.corrcoef(x[elig], sub.market_move_toward_home.fillna(0).values[elig])[0, 1])
    res["corr_x_with_home_cover_residual"] = float(np.corrcoef(x[elig], yy - base)[0, 1])
    res["oos_brier_gain_by_season"] = {int(s): float(np.mean((pr[(seasons == s) & elig] - y[(seasons == s) & elig]) ** 2 - (pooled[(seasons == s) & elig] - y[(seasons == s) & elig]) ** 2)) for s in (2023, 2024, 2025)}
    res["look_count"] = 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["lag", "term", "all"], default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = book_series()
    s = s[s.nflverse_game_id.str[:4].isin(["2023", "2024", "2025"])]
    part = s.sort_values(["nflverse_game_id", "bookmaker_key", "observed_at_utc"]).copy()
    part["chg"] = part.groupby(["nflverse_game_id", "bookmaker_key"]).home_spread_line.diff().fillna(0) != 0
    pc = part.groupby("bookmaker_key").chg.sum().sort_values(ascending=False)
    books = [b for b in pc.index if pc[b] >= MIN_PARTICIPATION]
    mats = game_mats(s, books)
    print("rows", len(s), "games", len(mats), "books", books, flush=True)
    if a.stage in ("lag", "all"):
        res = {"books": books, "participations": {b: int(pc[b]) for b in books}, "cadence": cadence(mats)}
        xc = xcorr(mats, books)
        rows = []
        for i, ai in enumerate(books):
            for j, bj in enumerate(books):
                if i == j:
                    continue
                r = {"a": ai, "b": bj}
                for lag in LAGS:
                    r[f"corr_lag{lag}"] = None if np.isnan(xc[lag][0][i, j]) else float(xc[lag][0][i, j])
                    r[f"n_lag{lag}"] = int(xc[lag][1][i, j])
                r["lead_asym"] = None if r["corr_lag1"] is None or r["corr_lag-1"] is None else r["corr_lag1"] - r["corr_lag-1"]
                rows.append(r)
        pd.DataFrame(rows).to_csv(OUT / "xcorr.csv", index=False)
        df = diffusion(mats, books)
        pt = pair_table(df, books)
        pt.to_csv(OUT / "pair_diffusion.csv", index=False)
        mt = df[df.status == 1]
        res["diffusion_overall"] = {"leader_move_follower_pairs": int(len(df)), "follower_already_there_share": float((df.status == 2).mean()),
                                    "matched_share_of_open": float((df.status == 1).sum() / (df.status != 2).sum()),
                                    "median_lag_provider_s": float(mt.lag_provider_s.median()),
                                    "median_lag_snapshot_s": float(mt.lag_snapshot_s.median()),
                                    "share_same_snapshot": float((mt.lag_snapshot_s == 0).mean())}
        by = {}
        for b, g in pt.groupby("leader"):
            by[b] = {"diffusion_share": float(g.matched.sum() / (g.a_moves.sum() - g.b_already_there.sum())), "median_pair_lag_provider_s": float(g.median_lag_provider_s.median())}
        res["by_leader"] = by
        res["xcorr_mean_by_lag_offdiag"] = {str(l): float(np.nanmean(np.where(np.eye(len(books)) == 1, np.nan, xc[l][0]))) for l in LAGS}
        json.dump(res, open(OUT / "lead_lag.json", "w"), indent=1, default=str)
        print(json.dumps(res, indent=1, default=str), flush=True)
    if a.stage in ("term", "all"):
        r = fitted_term(mats, books)
        json.dump(r, open(OUT / "fitted_term.json", "w"), indent=1, default=str)
        print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
