import json
import os
import sys
from pathlib import Path

import numpy as np

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import sim04_loso as base  # noqa: E402
import sim08_evaluate as ev  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

OUT = REPO / "artifacts" / "sim08_inversion"
SEED = 20261111
N_BOOT_COEF = 400


def fit(X, y):
    return LogisticRegression(C=1e10, max_iter=3000).fit(X, y)


def wald(X, y):
    m = fit(X, y)
    Z = np.column_stack([np.ones(len(X)), X])
    b = np.concatenate([m.intercept_, m.coef_[0]])
    p = 1 / (1 + np.exp(-Z @ b))
    H = Z.T @ (Z * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b[1:], se[1:]


def coef_row(X, y, inv, rng):
    b, se = wald(X, y)
    ids = np.unique(inv)
    members = [np.flatnonzero(inv == i) for i in ids]
    out = []
    for _ in range(N_BOOT_COEF):
        pick = rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([members[i] for i in pick])
        out.append(fit(X[idx], y[idx]).coef_[0])
    out = np.array(out)
    return {
        "coef": [float(v) for v in b],
        "wald_se": [float(v) for v in se],
        "boot_probability_positive": [float(np.mean(out[:, j] > 0)) for j in range(out.shape[1])],
        "n": len(y),
    }


def main():
    from nfl_ats.schedule_flag_features import default_opener_lines

    frame = base.load_games()
    lines = default_opener_lines(frame[["game_id", "season", "week"]])
    frame = frame.merge(lines[["game_id", "tue_open_total_line"]], on="game_id", how="left")
    frame["total"] = frame["tue_open_total_line"].astype(float)
    outcome = frame["outcome"].to_numpy()
    seasons = frame["season"].to_numpy()
    nonpush = outcome != "push"
    y = (outcome == "cover").astype(float)
    line = frame["line"].to_numpy(float)
    result = frame["result"].to_numpy(float)
    cov, push, loss = base.baseline_three_way(frame)
    served_logit = base.safe_logit(cov / np.clip(cov + loss, base.EPS, None))
    ll_served = ev.score((cov, push, loss), outcome)[0]
    m3 = ev.market_hist_three_way(frame)
    ll_market = ev.score(m3, outcome)[0]
    recal_cond, _ = ev.loso_cond(served_logit[:, None], y, nonpush, seasons)
    ll_recal = ev.score(ev.three_way(recal_cond, push), outcome)[0]
    inv, nb = ev.blocks_of(frame)
    rng = np.random.default_rng(SEED)
    S = ev.load_arm("main", "primary", frame)
    M = S["margins"]
    n = M.shape[1]
    mean = M.mean(1)
    sd = M.std(1, ddof=1)
    se = sd / np.sqrt(n)
    left = mean - line
    res = {"n_games": len(frame), "n_blocks": nb}
    tol = base.ATOM_TOLERANCE

    c, pu, l_ = ev.sim_counts(M, line)
    raw_logit = base.safe_logit((c + 0.5) / (c + l_ + 1.0))
    c_away = ((-line)[:, None] + tol < (-M)).sum(1)
    away_logit = base.safe_logit((c_away + 0.5) / (c_away + c + 1.0))
    samp = np.argsort(-np.abs(line))[:40]
    nb_, ib_ = nonpush, inv[nonpush]
    res["H1"] = {
        "mean_on_line_slope": float(np.polyfit(line, mean, 1)[0]),
        "mean_on_line_corr": float(np.corrcoef(line, mean)[0, 1]),
        "sign_agree_lopsided_40": int((np.sign(mean[samp]) == np.sign(line[samp])).sum()),
        "away_count_equals_sim_loss_count_all": bool((c_away == l_).all()),
        "raw_logit_vs_neg_away_logit_corr": float(np.corrcoef(raw_logit, -away_logit)[0, 1]),
        "label_matches_result_minus_line": bool(
            ((result - line > tol) == (outcome == "cover")).all()
        ),
        "real_result_on_line_slope": float(np.polyfit(line, result, 1)[0]),
        "mean_leftover_home_fav": float(left[line > 0].mean()),
        "mean_leftover_road_fav": float(left[line < 0].mean()),
        "solo_raw_logit": coef_row(raw_logit[nb_, None], y[nb_], ib_, rng),
    }
    z = left / se
    res["H2"] = {
        "leftover_pts_mean": float(left.mean()),
        "leftover_pts_sd": float(left.std(ddof=1)),
        "leftover_pts_p5_p95": [float(np.percentile(left, 5)), float(np.percentile(left, 95))],
        "leftover_se_units_mean": float(z.mean()),
        "leftover_se_units_sd": float(z.std(ddof=1)),
        "frac_abs_le_1se": float(np.mean(np.abs(z) <= 1)),
        "frac_abs_gt_2se": float(np.mean(np.abs(z) > 2)),
        "mc_se_pts_mean": float(se.mean()),
        "corr_leftover_xdiff": float(np.corrcoef(left, S["x_diff"])[0, 1]),
        "corr_leftover_line": float(np.corrcoef(left, line)[0, 1]),
        "leftover_on_line_slope": float(np.polyfit(line, left, 1)[0]),
    }
    cc = ((M - left[:, None]) > line[:, None]).sum(1)
    cen_logit = base.safe_logit((cc + 0.5) / (n + 1.0))
    res["H2"]["a_leftover_alone"] = coef_row(left[nb_, None], y[nb_], ib_, rng)
    res["H2"]["b_centred_logit_alone"] = coef_row(cen_logit[nb_, None], y[nb_], ib_, rng)
    res["H2"]["c_leftover_and_centred"] = coef_row(
        np.column_stack([left, cen_logit])[nb_], y[nb_], ib_, rng
    )
    res["H2"]["corr_raw_logit_centred_logit"] = float(np.corrcoef(raw_logit, cen_logit)[0, 1])
    res["H2"]["corr_raw_logit_leftover"] = float(np.corrcoef(raw_logit, left)[0, 1])

    _, cen_coefs = ev.loso_cond(cen_logit[:, None], y, nonpush, seasons)
    blend_cen, bl_coefs = ev.loso_cond(
        np.column_stack([served_logit, cen_logit]), y, nonpush, seasons
    )
    blend_raw, _ = ev.loso_cond(np.column_stack([served_logit, raw_logit]), y, nonpush, seasons)
    pu_s = (pu + 0.5) / (n + 1.5)
    ll_raw = ev.score(((c + 0.5) / (n + 1.5), pu_s, (l_ + 0.5) / (n + 1.5)), outcome)[0]
    pc = (cc + 0.5) / (n + 1.0)
    ll_cen = ev.score(((1 - pu_s) * pc, pu_s, (1 - pu_s) * (1 - pc)), outcome)[0]
    ll_blend_cen = ev.score(ev.three_way(blend_cen, push), outcome)[0]
    ll_blend_raw = ev.score(ev.three_way(blend_raw, push), outcome)[0]
    cells = {}

    def cell(name, d, mask=None):
        cells[name] = ev.boot_mean(d, inv, nb, rng, mask)

    cell("L1_raw_served_minus_raw", ll_served - ll_raw)
    cell("L1_centred_served_minus_centred", ll_served - ll_cen)
    cell("L3_centred_market_minus_centred", ll_market - ll_cen)
    cell("centred_gain_over_raw", ll_raw - ll_cen)
    cell("L4_centred_blend", ll_recal - ll_blend_cen)
    cell("L4_raw_blend", ll_recal - ll_blend_raw)
    absl = np.abs(line)
    for name, lo, hi in ev.BANDS:
        mk = (absl >= lo - 1e-9) & (absl <= hi + 1e-9)
        cell(f"B_{name}_L1_centred", ll_served - ll_cen, mk)
        cell(f"B_{name}_L1_raw", ll_served - ll_raw, mk)
    lls = {
        "served": ll_served,
        "market_hist": ll_market,
        "recal": ll_recal,
        "sim_raw": ll_raw,
        "sim_centred": ll_cen,
        "blend_raw": ll_blend_raw,
        "blend_centred": ll_blend_cen,
    }
    res["H3"] = {
        "log_loss": {k: float(v.mean()) for k, v in lls.items()},
        "centred_solo_fold_coefs": cen_coefs,
        "centred_blend_fold_coefs": bl_coefs,
        "centred_solo_pooled": coef_row(cen_logit[nb_, None], y[nb_], ib_, rng),
        "centred_blend_pooled": coef_row(
            np.column_stack([served_logit, cen_logit])[nb_], y[nb_], ib_, rng
        ),
        "cells": cells,
    }
    sv = served_logit > 0
    yy = y > 0.5
    mpick = m3[0] > m3[2]

    def record(lg, other):
        d = nonpush & ((lg > 0) != other)
        return {
            "n_differ": int(d.sum()),
            "sim_right": int(((lg > 0) == yy)[d].sum()),
            "other_right": int((other == yy)[d].sum()),
        }

    res["H3"]["decisive_vs_served"] = {
        "raw": record(raw_logit, sv),
        "centred": record(cen_logit, sv),
    }
    res["H3"]["decisive_vs_market_hist"] = {
        "raw": record(raw_logit, mpick),
        "centred": record(cen_logit, mpick),
    }
    res["H3"]["band_solo"] = {}
    for name, lo, hi in ev.BANDS:
        mk = (absl >= lo - 1e-9) & (absl <= hi + 1e-9) & nonpush
        res["H3"]["band_solo"][name] = {
            "raw": coef_row(raw_logit[mk, None], y[mk], inv[mk], rng),
            "centred": coef_row(cen_logit[mk, None], y[mk], inv[mk], rng),
        }

    absres = np.abs(result - line)
    tot = frame["total"].to_numpy(float)
    tleft = S["totals"].mean(1) - tot
    ok = np.isfinite(tot)
    res["H4"] = {
        "corr_sim_sd_real_abs_resid": float(np.corrcoef(sd, absres)[0, 1]),
        "slope_real_abs_resid_on_sim_sd": float(np.polyfit(sd, absres, 1)[0]),
        "sim_sd_mean": float(sd.mean()),
        "real_resid_sd": float((result - line).std(ddof=1)),
        "total_leftover_mean": float(np.nanmean(tleft)),
        "total_leftover_sd": float(np.nanstd(tleft)),
        "corr_margin_leftover_total_leftover": float(np.corrcoef(left[ok], tleft[ok])[0, 1]),
        "by_total_tercile": {},
        "by_abs_total_leftover_tercile": {},
    }

    def terc(key, values):
        qs = np.nanpercentile(values[ok], [33.3, 66.7])
        grp = np.digitize(values, qs)
        for g in range(3):
            mk = (grp == g) & ok & nonpush
            res["H4"][key][str(g)] = {
                "n": int(mk.sum()),
                "sim_sd_mean": float(sd[mk].mean()),
                "raw": coef_row(raw_logit[mk, None], y[mk], inv[mk], rng),
                "centred": coef_row(cen_logit[mk, None], y[mk], inv[mk], rng),
            }

    terc("by_total_tercile", tot)
    terc("by_abs_total_leftover_tercile", np.abs(tleft))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnosis.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
