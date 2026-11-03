import json
import subprocess
import sys

import mkt21_line_path as M
import mkt23_move_decomposition as D
import mkt23_unit2_term as U
import numpy as np
import pandas as pd
import sky04_lead_lag as L

OUT = L.OUT.parent / "mkt26"
FAMILY = "mkt26_move_dispersion"
SEASONS = (2023, 2024, 2025)
SPREAD_BINS = 5
PERMUTATIONS = 2000
LOOKS = 10
BASE_DEADLINE = "home_cover_probability_excluding_push_at_close"
BASE_OPENER = "out_of_season_home_probability"
PLAIN = (
    "Whether the size of the Tuesday-to-deadline spread move or book disagreement "
    "predicts how far the margin lands from the deadline line."
)


def last_at(q, t, value):
    qq = q[q.observed_at_utc <= pd.Timestamp(t, unit="s", tz="UTC")]
    return qq.groupby("bookmaker_key")[value].last()


def deadline_frame():
    s = L.book_series()
    s = s[s.nflverse_game_id.str[:4].isin([str(x) for x in SEASONS])]
    s = s.sort_values("observed_at_utc")
    tot = D.total_series().sort_values("observed_at_utc")
    g = pd.read_parquet(L.PERGAME)
    g = g[(g.game_type == "REG") & g.season.isin(SEASONS)]
    gf = pd.read_parquet(
        L.ROOT / "data/processed/game_features.parquet", columns=["game_id", "kickoff"]
    )
    g = g.merge(gf, on="game_id", how="left")
    g = g[g.kickoff.notna()].reset_index(drop=True)
    ts = {
        gid: (L.decision_ts(gd), M.deadline_ts(gd, ko))
        for gid, gd, ko in zip(g.game_id, g.gameday, g.kickoff)  # noqa: B905
    }
    tq = {gid: q for gid, q in tot.groupby("nflverse_game_id")}  # noqa: C416
    rows = {}
    for gid, q in s.groupby("nflverse_game_id"):
        if gid not in ts or gid not in tq:
            continue
        a, b = (
            last_at(q, ts[gid][0], "home_spread_line"),
            last_at(q, ts[gid][1], "home_spread_line"),
        )
        c = last_at(tq[gid], ts[gid][1], "total_line")
        if a.empty or b.empty or c.empty:
            continue
        rows[gid] = {
            "hs0": float(a.median()),
            "hs1": float(b.median()),
            "disagree": float(b.std(ddof=0)),
            "t1": float(c.median()),
        }
    r = pd.DataFrame.from_dict(rows, orient="index").rename_axis("game_id").reset_index()
    g = g.merge(r, on="game_id").reset_index(drop=True)
    g["abs_move"] = (g.hs1 - g.hs0).abs().round(4)
    g["m_dl"] = g.result - g.hs1
    g = g[g.m_dl != 0].reset_index(drop=True)
    g["abs_resid"] = g.m_dl.abs()
    g["near_push"] = (g.m_dl.abs() <= 0.5).astype(float)
    g["abs_line"] = g.hs1.abs()
    g["sbin"] = pd.qcut(g.abs_line, SPREAD_BINS, labels=False, duplicates="drop")
    return g


def ols_slope(x, y, ctrl):
    X = np.column_stack([np.ones(len(x)), x, ctrl])
    return np.linalg.lstsq(X, y, rcond=None)[0][1]


def dispersion(g, pred, outcome, rng):
    ctrl = g[["abs_line", "t1"]].values.astype(float)
    x, y = g[pred].values.astype(float), g[outcome].values.astype(float)
    obs = ols_slope(x, y, ctrl)
    n = len(g)
    boots = np.empty(L.BOOT)
    for i in range(L.BOOT):
        k = rng.integers(0, n, n)
        boots[i] = ols_slope(x[k], y[k], ctrl[k])
    null = np.empty(PERMUTATIONS)
    idx = [np.where(g.sbin.values == b)[0] for b in np.unique(g.sbin.values)]
    for i in range(PERMUTATIONS):
        xp = x.copy()
        for ix in idx:
            xp[ix] = x[rng.permutation(ix)]
        null[i] = ols_slope(xp, y, ctrl)
    h = np.array_split(rng.permutation(n), 2)
    se = g.season.values
    return {
        "slope": float(obs),
        "interval": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "standard_error": float(boots.std(ddof=1)),
        "probability_positive": float((boots > 0).mean()),
        "permutation_p_one_sided_positive": float((null >= obs).mean()),
        "null_sd": float(null.std(ddof=1)),
        "split_half_slopes": [float(ols_slope(x[a], y[a], ctrl[a])) for a in h],
        "by_season": {
            str(s): float(ols_slope(x[se == s], y[se == s], ctrl[se == s])) for s in SEASONS
        },
    }


def loso(g, y, cols):
    x = g[cols].values.astype(float) if cols else np.empty((len(g), 0))
    oos = np.zeros(len(g))
    coefs = {}
    for s in SEASONS:
        te = g.season.values == s
        w = U.fit(x[~te], y[~te])
        oos[te] = U.predict(w, x[te])
        coefs[str(s)] = [float(v) for v in w]
    w = U.fit(x, y)
    return oos, coefs, float(U.loss(U.predict(w, x), y).mean())


def shrink_arms(g, y, base_col, rng):
    z = g.copy()
    p = z[base_col].astype(float)
    z["logit"] = np.log(p / (1 - p))
    z["l_x_move"] = z.logit * z.abs_move
    z["l_x_dis"] = z.logit * z.disagree
    arms = {
        "b2_base_refit": ["logit"],
        "c_move": ["logit", "l_x_move"],
        "d_disagree": ["logit", "l_x_dis"],
        "e_both": ["logit", "l_x_move", "l_x_dis"],
    }
    base_side = p.values > 0.5
    res = {"n": len(z)}
    preds = {}
    for arm, cols in arms.items():
        oos, coefs, ins = loso(z, y, cols)
        preds[arm] = oos
        ll = U.loss(oos, y)
        flip = (oos > 0.5) != base_side
        res[arm] = {
            "log_loss": float(ll.mean()),
            "in_sample_log_loss": ins,
            "gap_oos_minus_in": float(ll.mean() - ins),
            "coefficients": coefs,
            "served_base_log_loss": float(U.loss(p.values, y).mean()),
            "decisive_vs_served_base": {
                "flips": int(flip.sum()),
                "arm_right": int(((oos > 0.5) == (y == 1))[flip].sum()),
                "base_right": int((base_side == (y == 1))[flip].sum()),
            },
        }
    inter, _, _ = loso(z, y, [])
    res["a_intercept"] = {"log_loss": float(U.loss(inter, y).mean())}
    for arm in ("c_move", "d_disagree", "e_both"):
        d = U.loss(preds["b2_base_refit"], y) - U.loss(preds[arm], y)
        res[arm]["log_loss_improvement_vs_refit"] = U.interval(d, rng)
        res[arm]["by_season"] = {str(s): float(d[z.season.values == s].mean()) for s in SEASONS}
    return res


def cell(name, effect, units, lo, hi, se, pp, evidence):
    return {
        "name": name,
        "effect": effect,
        "effect_units": units,
        "interval_low": lo,
        "interval_high": hi,
        "standard_error": se,
        "probability_positive": pp,
        "classification_evidence": evidence,
    }


def run():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(L.SEED)
    g = deadline_frame()
    moved = g.abs_move > 0
    res = {
        "family": FAMILY,
        "n": len(g),
        "abs_move_summary": g.abs_move.describe().to_dict(),
        "disagree_summary": g.disagree.describe().to_dict(),
        "near_push_rate": float(g.near_push.mean()),
        "mean_abs_resid": float(g.abs_resid.mean()),
        "by_move_group": {
            "moved_n": int(moved.sum()),
            "moved_abs_resid": float(g.abs_resid[moved].mean()),
            "unmoved_abs_resid": float(g.abs_resid[~moved].mean()),
            "moved_near_push": float(g.near_push[moved].mean()),
            "unmoved_near_push": float(g.near_push[~moved].mean()),
        },
        "deadline_vs_close_line_absdiff_mean": float(
            (g.hs1 - g.close_home_spread.fillna(g.hs1)).abs().mean()
        ),
        "dispersion": {},
    }
    for pred in ("abs_move", "disagree"):
        for out in ("abs_resid", "near_push"):
            res["dispersion"][f"{pred}__{out}"] = dispersion(g, pred, out, rng)
    gd = g.dropna(subset=[BASE_DEADLINE])
    res["deadline_view"] = shrink_arms(
        gd.reset_index(drop=True),
        (gd.m_dl > 0).values.astype(float),
        BASE_DEADLINE,
        rng,
    )
    go = g[g.margin_vs_open != 0].dropna(subset=[BASE_OPENER, "home_covered"])
    res["opener_view_information_arriving_later"] = shrink_arms(
        go.reset_index(drop=True), go.home_covered.values.astype(float), BASE_OPENER, rng
    )
    (OUT / "result.json").write_text(json.dumps(res, indent=1, default=float))
    cells = []
    for k, v in res["dispersion"].items():
        cells.append(
            cell(
                f"mkt26_dispersion_{k}",
                v["slope"],
                "margin_points" if "resid" in k else "accuracy_points",
                v["interval"][0],
                v["interval"][1],
                v["standard_error"],
                v["probability_positive"],
                f"OLS slope, controls line and total, n {len(g)}, {LOOKS} looks; no control",
            )
        )
    for view, key in (
        ("deadline", "deadline_view"),
        ("opener_later", "opener_view_information_arriving_later"),
    ):
        for arm in ("c_move", "d_disagree", "e_both"):
            e = res[key][arm]["log_loss_improvement_vs_refit"]
            cells.append(
                cell(
                    f"mkt26_shrink_{view}_{arm}",
                    e["effect"],
                    "log_loss_improvement",
                    e["interval"][0],
                    e["interval"][1],
                    e["standard_error"],
                    e["probability_positive"],
                    f"LOSO 2023-25, {LOOKS} looks; no positive control; {view} view",
                )
            )
    batch = {
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": 2023,
        "season_end": 2025,
        "family": FAMILY,
        "category": "market",
        "source": "artifacts/mkt26/result.json",
        "sample_games": len(g),
        "plain_summary": PLAIN,
        "description": f"Dispersion slopes and LOSO shrink terms; {LOOKS} looks",
        "cells": cells,
    }
    (OUT / "record_batch.json").write_text(json.dumps(batch, indent=1))
    print(json.dumps(res, indent=1, default=float))
    if "--record" in sys.argv:
        subprocess.run(
            [
                str(L.ROOT / ".venv" / "Scripts" / "nfl-ats.exe"),
                "weak-signals",
                "record",
                "--batch",
                str(OUT / "record_batch.json"),
            ],
            check=True,
            cwd=L.ROOT,
        )


if __name__ == "__main__":
    sys.exit(run())
