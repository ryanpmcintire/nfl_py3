import json
import sys

import numpy as np
import pandas as pd
import sky04_lead_lag as L

SEASONS = (2023, 2024, 2025)


def noon_lines(mats, gids, gamedays):
    n = len(gids)
    out = np.full((n, len(L.BOOKS_REF)), np.nan)
    for i, (gid, gd) in enumerate(zip(gids, gamedays)):  # noqa: B905
        if gid not in mats:
            continue
        o, M, U = mats[gid]  # noqa: RUF059
        k = np.searchsorted(o, L.decision_ts(gd), side="right") - 1
        if k >= 0:
            out[i] = M[k]
    return out


def main():
    s = L.book_series()
    s = s[s.nflverse_game_id.str[:4].isin(["2023", "2024", "2025"])]
    part = s.sort_values(["nflverse_game_id", "bookmaker_key", "observed_at_utc"]).copy()
    part["chg"] = (
        part.groupby(["nflverse_game_id", "bookmaker_key"]).home_spread_line.diff().fillna(0) != 0
    )
    pc = part.groupby("bookmaker_key").chg.sum().sort_values(ascending=False)
    books = [b for b in pc.index if pc[b] >= L.MIN_PARTICIPATION]
    L.BOOKS_REF = books
    mats = L.game_mats(s, books)
    g = pd.read_parquet(L.PERGAME)
    g = g[(g.game_type == "REG") & g.season.isin(SEASONS)]
    sub = g[g.margin_vs_open != 0].copy().reset_index(drop=True)
    gids = sub.game_id.values
    X = noon_lines(mats, gids, sub.gameday.values)
    y = sub.home_covered.astype(float).values
    pr = sub.out_of_season_home_probability.clip(1e-6, 1 - 1e-6).values
    off = L.logit(pr)
    res_m = sub.result.values.astype(float)
    seasons = sub.season.values
    med = np.nanmedian(np.where(np.isnan(X).all(axis=1)[:, None], 0, X), axis=1)
    has = (~np.isnan(X)).sum(axis=1) >= 3
    err2 = (res_m[:, None] - X) ** 2
    dev = (res_m[:, None] - X) ** 2 - ((res_m - med)[:, None]) ** 2
    rng = np.random.default_rng(L.SEED)
    book_tab = {}
    for j, b in enumerate(books):
        m = has & ~np.isnan(X[:, j])
        d = dev[m, j]
        bt = L.boot(-d, rng)
        book_tab[b] = {
            "games": int(m.sum()),
            "mse": float(err2[m, j].mean()),
            "mse_minus_median_mse_mean": float(d.mean()),
            "sharper_than_median_mean_boot": bt,
        }
    x = np.zeros(len(sub))
    weights = {}
    for sn in SEASONS:
        tr = (seasons != sn) & has
        w = np.zeros(len(books))
        for j in range(len(books)):
            m = tr & ~np.isnan(X[:, j])
            w[j] = 1 / err2[m, j].mean() if m.sum() > 50 else 0
        w = w / w.sum()
        weights[int(sn)] = {b: float(w[j]) for j, b in enumerate(books)}
        te = (seasons == sn) & has
        Xt = X[te]
        ww = np.where(np.isnan(Xt), 0, w[None, :])
        wm = (np.nan_to_num(Xt) * ww).sum(axis=1) / ww.sum(axis=1)
        x[te] = wm - med[te]
    elig = has
    pooled = np.zeros(len(sub))
    ins = np.zeros(len(sub))
    fc = {}
    for sn in SEASONS:
        te = (seasons == sn) & elig
        tr = (seasons != sn) & elig
        b = L.fit_b(off[tr], x[tr], y[tr])
        fc[int(sn)] = float(b)
        pooled[te] = 1 / (1 + np.exp(-(off[te] + b * x[te])))
    b_all = L.fit_b(off[elig], x[elig], y[elig])
    ins[elig] = 1 / (1 + np.exp(-(off[elig] + b_all * x[elig])))
    base = pr[elig]
    yy = y[elig]
    out = {
        "books": books,
        "per_book": book_tab,
        "weights_by_heldout_season": weights,
        "games_scored": int(elig.sum()),
        "x_nonzero_games": int((x[elig] != 0).sum()),
        "x_sd_points": float(np.std(x[elig])),
        "fold_coefficients": fc,
        "pooled_coefficient": float(b_all),
        "look_count": 1,
    }
    for label, arr in (("oos", pooled[elig]), ("ins", ins[elig])):
        d_b = (base - yy) ** 2 - (arr - yy) ** 2
        d_ll = -(yy * np.log(base) + (1 - yy) * np.log(1 - base)) + (
            yy * np.log(arr) + (1 - yy) * np.log(1 - arr)
        )
        d_a = (
            ((arr > 0.5) == (yy == 1)).astype(float) - ((base > 0.5) == (yy == 1)).astype(float)
        ) * 100
        dec = (base > 0.5) != (arr > 0.5)
        out[label] = {
            "brier_gain": L.boot(d_b, rng),
            "logloss_gain": L.boot(d_ll, rng),
            "accuracy_gain_points": L.boot(d_a, rng),
            "decisive_games": int(dec.sum()),
            "decisive_new_correct": int(((arr > 0.5) == (yy == 1))[dec].sum()),
            "decisive_base_correct": int(((base > 0.5) == (yy == 1))[dec].sum()),
        }
    out["gap_brier_gain_ins_minus_oos"] = out["ins"]["brier_gain"][0] - out["oos"]["brier_gain"][0]
    json.dump(out, open(L.OUT / "sharpness.json", "w"), indent=1)  # noqa: SIM115
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
