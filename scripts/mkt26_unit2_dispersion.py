import json
import sys

import mkt23_unit2_term as U
import mkt26_move_dispersion as K
import numpy as np
import pandas as pd
import sky04_lead_lag as L

from nfl_ats import mass_preserving_lattice as P
from nfl_ats.modeling import regular_season_rows

OUT = K.OUT
FAMILY = K.FAMILY
TARGETS = tuple(range(2013, 2026))
SBR_LAST = 2021
KEYS = (3, 7, 10, 14, 17)
LOOKS = 7
TOL = P._ATOM_TOLERANCE
EPS = np.finfo(float).eps
PLAIN = (
    "Whether a bigger opener-to-close spread move means the final margin lands farther "
    "from the line, and whether reading the margin distribution from games with the same "
    "kind of move improves the cover probability."
)


def sources():
    gf = pd.read_parquet(L.ROOT / "data/processed/game_features.parquet")
    cols = ("game_id", "season", "week", "gameday", "game_type", "result", "spread_line")
    gf = regular_season_rows(gf[[c for c in (*cols, "total_line") if c in gf.columns]]).copy()
    gf = gf[gf.result.notna()].drop_duplicates("game_id").reset_index(drop=True)
    gf["gameday"] = pd.to_datetime(gf.gameday)
    gf["game_id"] = gf.game_id.astype(str)
    sbr = pd.read_parquet(L.ROOT / "data/processed/sbr_odds.parquet")
    sbr = sbr[
        sbr.game_id.notna()
        & ~sbr.open_ambiguous.astype(bool)
        & ~sbr.close_ambiguous.astype(bool)
        & (sbr.season <= SBR_LAST)
    ]
    sbr = sbr.drop_duplicates("game_id", keep=False)
    sbr = sbr[["game_id", "open_home_spread", "close_home_spread"]].assign(source="sbr")
    pg = pd.read_parquet(L.PERGAME)
    pg = pg[pg.season > SBR_LAST][["game_id", "tue_open_home_spread", "close_home_spread"]]
    pg = pg.rename(columns={"tue_open_home_spread": "open_home_spread"}).assign(source="tuesday")
    lines = pd.concat([sbr, pg]).rename(
        columns={"open_home_spread": "open", "close_home_spread": "close"}
    )
    lines["game_id"] = lines.game_id.astype(str)
    lines = lines.dropna(subset=["open", "close"])
    f = gf.merge(lines, on="game_id", how="left").drop_duplicates("game_id", keep=False)
    checks = {}
    for src, d in f.dropna(subset=["close"]).groupby("source"):
        checks[src] = {
            "n": len(d),
            "seasons": [int(d.season.min()), int(d.season.max())],
            "corr_close_vs_nflverse_close": float(np.corrcoef(d.close, d.spread_line)[0, 1]),
            "mean_abs_close_minus_nflverse_close": float((d.close - d.spread_line).abs().mean()),
            "corr_close_vs_result": float(np.corrcoef(d.close, d.result)[0, 1]),
            "share_moved": float((d.open != d.close).mean()),
            "mean_abs_move": float((d.open - d.close).abs().mean()),
        }
    f["line"] = f.close.fillna(f.spread_line).astype(float)
    f["result"] = np.rint(f.result.astype(float))
    f["abs_move"] = (f.close - f.open).abs()
    f["moved"] = np.where(f.abs_move.isna(), np.nan, (f.abs_move > 0).astype(float))
    return f, gf, checks


def lattice(pl, pm, anchor, point):
    band = P.BAND_HALF_WIDTH
    while True:
        sel = np.abs(pl - anchor) <= band
        if int(sel.sum()) >= P.MIN_BAND_GAMES or band >= P.MAX_BAND:
            break
        band = min(band + P.BAND_STEP, P.MAX_BAND)
    values, counts = np.unique(pm[sel], return_counts=True)
    mass, _ = P.tilted_atoms(values, counts.astype(float), anchor, point)
    return values, mass


def cond_cover(values, mass, line):
    c = mass[values > line + TOL].sum()
    lo = mass[values < line - TOL].sum()
    return float(np.clip(c / (c + lo), EPS, 1 - EPS)) if c + lo > 0 else 0.5


def reads(pool, targets, anchor, point, lines, classes=("all", "mv", "un")):
    out = []
    for (s, w), grp in targets.groupby(["season", "week"], sort=True):
        el = P.prior_pool_for_week(
            pool,
            season=int(s),
            week=int(w),
            cutoff=pd.Timestamp(grp.gameday.min()),
            exclude_game_ids=set(grp.game_id.astype(str)),
        )
        parts = {"all": el, "mv": el[el.moved == 1], "un": el[el.moved == 0]}
        arr = {
            k: (parts[k].line.to_numpy(float), parts[k].result.to_numpy(float)) for k in classes
        }
        for r in grp.itertuples():
            row = {"game_id": r.game_id, "pool_all": len(el)}
            for k in classes:
                v, m = lattice(*arr[k], getattr(r, anchor), getattr(r, point))
                for name, col in lines:
                    row[f"p_{k}_{name}"] = cond_cover(v, m, getattr(r, col))
                if k == "all":
                    a = getattr(r, anchor)
                    row["e_abs"] = float((m * np.abs(v - a)).sum())
                    row["e_key"] = float(m[np.isin(np.abs(v), KEYS)].sum())
                    row["band_check"] = P.band_read(*arr[k], getattr(r, lines[0][1]), getattr(r, point), conditioning_line=a).conditional_cover_probability
            out.append(row)
    return pd.DataFrame(out)


def ols_slope(x, y, c):
    return np.linalg.lstsq(np.column_stack([np.ones(len(x)), x, c]), y, rcond=None)[0][1]


def dispersion(g, x, y, ctrl, rng):
    obs = ols_slope(x, y, ctrl)
    n = len(g)
    boots = np.array(
        [ols_slope(x[k], y[k], ctrl[k]) for k in rng.integers(0, n, (L.BOOT, n))]
    )
    idx = [np.where(g.season.values == s)[0] for s in np.unique(g.season.values)]
    null = np.empty(K.PERMUTATIONS)
    for i in range(K.PERMUTATIONS):
        xp = x.copy()
        for ix in idx:
            xp[ix] = x[rng.permutation(ix)]
        null[i] = ols_slope(xp, y, ctrl)
    h = np.array_split(rng.permutation(n), 2)
    se = g.season.values
    by = {str(s): float(ols_slope(x[se == s], y[se == s], ctrl[se == s][:, :2])) for s in np.unique(se)}
    return {
        "slope": float(obs),
        "interval": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "standard_error": float(boots.std(ddof=1)),
        "probability_positive": float((boots > 0).mean()),
        "permutation_p_one_sided_positive": float((null >= obs).mean()),
        "split_half_slopes": [float(ols_slope(x[a], y[a], ctrl[a])) for a in h],
        "by_season": by,
        "seasons_positive": int(sum(v > 0 for v in by.values())),
    }


def compare(y, pa, pb, season, rng, alt=None):
    d = U.loss(pa, y) - U.loss(pb, y)
    res = U.interval(d, rng)
    flip = (pa > 0.5) != (pb > 0.5)
    res.update(
        {
            "n": int(len(y)),
            "log_loss_base": float(U.loss(pa, y).mean()),
            "log_loss_arm": float(U.loss(pb, y).mean()),
            "by_season": {str(s): float(d[season == s].mean()) for s in np.unique(season)},
            "decisive": {
                "flips": int(flip.sum()),
                "arm_right": int(((pb > 0.5) == (y == 1))[flip].sum()),
                "base_right": int(((pa > 0.5) == (y == 1))[flip].sum()),
            },
        }
    )
    if alt is not None:
        idx = [np.where(season == s)[0] for s in np.unique(season)]
        mv, un, cls = alt
        null = np.empty(K.PERMUTATIONS)
        for i in range(K.PERMUTATIONS):
            c = cls.copy()
            for ix in idx:
                c[ix] = cls[rng.permutation(ix)]
            null[i] = (U.loss(pa, y) - U.loss(np.where(c == 1, mv, un), y)).mean()
        res["permutation_p_one_sided_positive"] = float((null >= d.mean()).mean())
    return res


def loso_shrink(z, y, rng):
    lg = np.log(z.pa / (1 - z.pa)).to_numpy()
    x0, x1 = lg[:, None], np.column_stack([lg, lg * z.abs_move.to_numpy()])
    se = z.season.values
    o0, o1, coefs = np.zeros(len(z)), np.zeros(len(z)), {}
    for s in np.unique(se):
        te = se == s
        w0, w1 = U.fit(x0[~te], y[~te]), U.fit(x1[~te], y[~te])
        o0[te], o1[te] = U.predict(w0, x0[te]), U.predict(w1, x1[te])
        coefs[str(s)] = [float(v) for v in w1]
    ins = U.loss(U.predict(U.fit(x1, y), x1), y).mean() - U.loss(U.predict(U.fit(x0, y), x0), y).mean()
    res = compare(y, o0, o1, se, rng)
    res["coefficients_logit_x_move"] = coefs
    res["interaction_negative_folds"] = int(sum(v[2] < 0 for v in coefs.values()))
    res["in_sample_improvement"] = float(-ins)
    res["gap_oos_minus_in"] = float(res["effect"] + ins)
    return res


def cell(name, v, units, evidence):
    return {
        "name": name,
        "effect": v.get("slope", v.get("effect")) * (100 if units == "rate_pp" else 1),
        "effect_units": units,
        "interval_low": v["interval"][0] * (100 if units == "rate_pp" else 1),
        "interval_high": v["interval"][1] * (100 if units == "rate_pp" else 1),
        "standard_error": v["standard_error"] * (100 if units == "rate_pp" else 1),
        "probability_positive": v["probability_positive"],
        "classification_evidence": evidence,
    }


def run():
    rng = np.random.default_rng(L.SEED)
    f, gf, checks = sources()
    res = {"family": FAMILY, "looks": LOOKS, "source_checks": checks}
    tg = f[f.season.isin(TARGETS) & f.open.notna()].reset_index(drop=True)
    rd = reads(f, tg, "close", "close", [("open", "open")])
    g = tg.merge(rd, on="game_id")
    res["band_read_max_abs_diff"] = float((g.p_all_open - g.band_check).abs().max())
    g["abs_line"] = g.close.abs()
    g["r"] = g.result - g.close
    g["ex_abs"] = g.r.abs() - g.e_abs
    g["ex_key"] = np.isin(g.result.abs(), KEYS).astype(float) - g.e_key
    g["total_line"] = g.total_line.fillna(g.total_line.median())
    ctrl = np.column_stack(
        [
            g.abs_line.to_numpy(float),
            g.total_line.to_numpy(float),
            pd.get_dummies(g.season, drop_first=True, dtype=float).to_numpy(),
        ]
    )
    x = g.abs_move.to_numpy(float)
    res["n_dispersion"] = len(g)
    res["mean_abs_move"] = float(x.mean())
    res["share_moved"] = float(g.moved.mean())
    res["mean_excess_abs"] = float(g.ex_abs.mean())
    res["mean_excess_key"] = float(g.ex_key.mean())
    res["D1_excess_abs_resid"] = dispersion(g, x, g.ex_abs.to_numpy(float), ctrl, rng)
    res["D2_excess_key_landing"] = dispersion(g, x, g.ex_key.to_numpy(float), ctrl, rng)
    mv = g.moved.to_numpy() == 1
    r = g.r.to_numpy(float)
    vr = []
    for k in rng.integers(0, len(g), (L.BOOT, len(g))):
        vr.append(r[k][mv[k]].var() / r[k][~mv[k]].var())
    vr = np.array(vr)
    res["D3_variance_ratio_moved_vs_unmoved"] = {
        "ratio": float(r[mv].var() / r[~mv].var()),
        "interval": [float(np.percentile(vr, 2.5)), float(np.percentile(vr, 97.5))],
        "probability_above_one": float((vr > 1).mean()),
        "sd_moved": float(r[mv].std()),
        "sd_unmoved": float(r[~mv].std()),
        "n_moved": int(mv.sum()),
        "by_source": {
            s: float(r[mv & (g.source == s)].var() / r[~mv & (g.source == s)].var())
            for s in g.source.unique()
        },
    }
    o = g[g.result != g.open].reset_index(drop=True)
    y = (o.result > o.open).to_numpy(float)
    pb = np.where(o.moved == 1, o.p_mv_open, o.p_un_open)
    res["L1_market_lattice_move_class"] = compare(
        y,
        o.p_all_open.to_numpy(),
        pb,
        o.season.values,
        rng,
        alt=(o.p_mv_open.to_numpy(), o.p_un_open.to_numpy(), o.moved.to_numpy()),
    )
    res["L4_loso_logit_x_move"] = loso_shrink(o.assign(pa=o.p_all_open), y, rng)
    d = K.deadline_frame(keep_pushes=True)
    d = d.dropna(subset=["tue_open_home_spread", "residual_at_close_served", "hs1"])
    d = d.reset_index(drop=True)
    d["game_id"] = d.game_id.astype(str)
    d["gameday"] = pd.to_datetime(d.gameday)
    d["point_dl"] = d.hs1 + d.residual_at_close_served
    d["point_close"] = d.close_home_spread + d.residual_at_close_served
    d["cls"] = (d.abs_move > 0).astype(float)
    pp = pd.read_parquet(L.PERGAME)
    pg_open = pp.set_index(pp.game_id.astype(str)).tue_open_home_spread.dropna()
    pool = P.prior_pool(gf, pg_open)
    pool["moved"] = pool.game_id.map(f.set_index("game_id").moved)
    rd2 = reads(pool, d, "hs1", "point_dl", [("open", "tue_open_home_spread"), ("dl", "hs1")])
    fid = reads(pool, d.dropna(subset=["close_home_spread"]), "close_home_spread", "point_close", [("close", "close_home_spread")], ("all",))
    fid = fid.merge(d[["game_id", "home_cover_probability_at_close"]], on="game_id")
    res["fidelity_reserve_vs_stored_close"] = {
        "n": len(fid),
        "mean_abs_diff": float((fid.p_all_close - fid.home_cover_probability_at_close).abs().mean()),
        "max_abs_diff": float((fid.p_all_close - fid.home_cover_probability_at_close).abs().max()),
    }
    d = d.merge(rd2, on="game_id")
    res["deadline_class_share_moved"] = float(d.cls.mean())
    for key, line, name in (("L2_deadline_reserve_opener_graded", "tue_open_home_spread", "open"), ("L3_deadline_reserve_deadline_graded", "hs1", "dl")):
        q = d[d.result != d[line]].reset_index(drop=True)
        yy = (q.result > q[line]).to_numpy(float)
        pa = q[f"p_all_{name}"].to_numpy()
        pv, pu = q[f"p_mv_{name}"].to_numpy(), q[f"p_un_{name}"].to_numpy()
        res[key] = compare(yy, pa, np.where(q.cls == 1, pv, pu), q.season.values, rng, alt=(pv, pu, q.cls.to_numpy()))
        if name == "open":
            sv = q.out_of_season_home_probability.astype(float).to_numpy()
            res[key]["served_opener_probability_log_loss_same_games"] = float(
                pd.Series(U.loss(sv, yy)).mean()
            )
    (OUT / "unit2_result.json").write_text(json.dumps(res, indent=1, default=float))
    ev = f"{LOOKS} looks family {FAMILY}; no positive control"
    cells = [
        cell("mkt26u2_D1_excess_abs_resid_per_move", res["D1_excess_abs_resid"], "ats_points", f"OLS on 2013-2025 open-close pairs n {len(g)}, lattice E|r| removed; {ev}"),
        cell("mkt26u2_D2_excess_key_landing_per_move", res["D2_excess_key_landing"], "rate_pp", f"OLS n {len(g)}, lattice key mass removed; {ev}"),
        cell("mkt26u2_L1_market_lattice_move_class", res["L1_market_lattice_move_class"], "log_loss_improvement", f"opener-graded 2013-2025, same-class pool vs full pool, nothing fitted; {ev}"),
        cell("mkt26u2_L4_loso_logit_x_move", res["L4_loso_logit_x_move"], "log_loss_improvement", f"LOSO season folds 2013-2025; {ev}"),
        cell("mkt26u2_L2_deadline_reserve_opener_graded", res["L2_deadline_reserve_opener_graded"], "log_loss_improvement", f"re-served lattice at deadline line 2023-25, opener-graded; {ev}"),
        cell("mkt26u2_L3_deadline_reserve_deadline_graded", res["L3_deadline_reserve_deadline_graded"], "log_loss_improvement", f"re-served lattice at deadline line 2023-25, deadline-graded; {ev}"),
    ]
    batch = {
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": 2013,
        "season_end": 2025,
        "family": FAMILY,
        "category": "market",
        "source": "artifacts/mkt26/unit2_result.json",
        "sample_games": len(g),
        "plain_summary": PLAIN,
        "description": f"Unit 2: 2013-2025 dispersion slopes and move-class lattice reads; {LOOKS} looks",
        "cells": cells,
    }
    (OUT / "unit2_record_batch.json").write_text(json.dumps(batch, indent=1))
    short = {k: v for k, v in res.items()}
    print(json.dumps(short, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(run())
