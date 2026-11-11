import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import sim04_loso as base  # noqa: E402
import sim08_evaluate as ev  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

SEED = 20261111
KEYS = (3, 7, 10, 14, 17)
SHAPE_BANDS = (
    ("0_2.5", 0.0, 2.5),
    ("3_3.5", 3.0, 3.5),
    ("4_6.5", 4.0, 6.5),
    ("7_9.5", 7.0, 9.5),
    ("10plus", 10.0, 1e9),
)
COVER_SE_DECLARED = 0.026
ESS_FLOOR = 0.25 / COVER_SE_DECLARED**2
N_BOOT_COEF = 400


OUT = REPO / "artifacts" / "sim08_amendment5"


def tilt_weights(margins, line):
    n = margins.shape[1]
    w = np.empty(margins.shape)
    ess = np.empty(len(margins))
    resid = np.empty(len(margins))
    for i in range(len(margins)):
        m = margins[i]
        values, inverse, counts = np.unique(m, return_inverse=True, return_counts=True)
        per_value = base.tilt_to_mean(values, counts.astype(float), float(line[i]))
        wi = per_value[inverse] / counts[inverse]
        wi = wi / wi.sum()
        w[i] = wi
        ess[i] = 1.0 / float((wi**2).sum())
        resid[i] = float((wi * m).sum() - line[i])
    return w * n, ess, resid


def wcounts(margins, w, line):
    diff = margins - line[:, None]
    c = (w * (diff > base.ATOM_TOLERANCE)).sum(1)
    lo = (w * (diff < -base.ATOM_TOLERANCE)).sum(1)
    return c, margins.shape[1] - c - lo, lo


def fit(X, y):
    return LogisticRegression(C=1e10, max_iter=3000).fit(X, y)


def coef_row(X, y, inv, rng):
    m = fit(X, y)
    ids = np.unique(inv)
    members = [np.flatnonzero(inv == i) for i in ids]
    boots = []
    for _ in range(N_BOOT_COEF):
        pick = rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([members[i] for i in pick])
        boots.append(fit(X[idx], y[idx]).coef_[0])
    boots = np.array(boots)
    return {
        "coef": [float(v) for v in m.coef_[0]],
        "ci95": [
            [float(np.percentile(boots[:, j], 2.5)), float(np.percentile(boots[:, j], 97.5))]
            for j in range(boots.shape[1])
        ],
        "boot_probability_positive": [
            float(np.mean(boots[:, j] > 0)) for j in range(boots.shape[1])
        ],
        "n": len(y),
    }


def two_way_loss(cond, y):
    p = np.clip(np.where(y > 0.5, cond, 1 - cond), base.EPS, 1 - base.EPS)
    return -np.log(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    a = ap.parse_args()
    global OUT
    OUT = (
        REPO
        / "artifacts"
        / ("sim08_amendment5" if a.run == "main" else f"sim08_amendment5_{a.run}")
    )
    from nfl_ats.schedule_flag_features import default_opener_lines

    OUT.mkdir(parents=True, exist_ok=True)
    frame = base.load_games()
    lines = default_opener_lines(frame[["game_id", "season", "week"]])
    frame = frame.merge(lines[["game_id", "tue_open_total_line"]], on="game_id", how="left")
    frame["total"] = frame["tue_open_total_line"].astype(float)
    g = pd.read_parquet(base.GAME_FEATURES_PATH)
    g = g.loc[g["game_type"] == "REG", ["game_id", "home_score", "away_score"]]
    frame = frame.merge(g, on="game_id", how="left")
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
    mdl = frame["home_cover_probability_at_open_discrete_uncorrected"].to_numpy(dtype=float)
    mdl_push = frame["push_probability_at_open"].to_numpy(dtype=float)
    mdl3 = (np.clip(mdl, 0, 1 - mdl_push), mdl_push, np.clip(1 - mdl - mdl_push, 0, 1))
    ll_model = ev.score(mdl3, outcome)[0]
    recal_cond, _ = ev.loso_cond(served_logit[:, None], y, nonpush, seasons)
    ll_recal = ev.score(ev.three_way(recal_cond, push), outcome)[0]
    inv, nb = ev.blocks_of(frame)
    rng = np.random.default_rng(SEED)

    S = ev.load_arm(a.run, "primary", frame)
    M = S["margins"]
    n = S["draws"]
    w, ess, resid = tilt_weights(M, line)
    np.savez_compressed(
        OUT / "tilt_weights.npz",
        game_id=frame["game_id"].to_numpy().astype(str),
        weights=w.astype(np.float32),
        ess=ess,
    )
    wt = w / n
    mean_total_err = (wt * S["totals"]).sum(1) - frame["total"].to_numpy(float)
    raw_mean_total_err = S["totals"].mean(1) - frame["total"].to_numpy(float)
    res = {
        "caveat": (
            "2026-11-03 engine (Amendment 3 one-shot anchor), without Amendment 4 "
            "terminal transitions; the final run will differ in shape"
        ),
        "n_games": len(frame),
        "n_blocks": nb,
        "draws": n,
        "ess_floor": ESS_FLOOR,
        "ess": {
            "mean": float(ess.mean()),
            "min": float(ess.min()),
            "pcts_5_25_50_75_95": [float(v) for v in np.percentile(ess, [5, 25, 50, 75, 95])],
            "below_floor": int((ess < ESS_FLOOR).sum()),
            "below_100": int((ess < 100).sum()),
            "below_50": int((ess < 50).sum()),
        },
        "tilt_exactness": {
            "max_abs_weighted_mean_minus_line": float(np.abs(resid).max()),
            "n_over_1e-6": int((np.abs(resid) > 1e-6).sum()),
        },
        "total_channel": {
            "raw_mean_err": float(np.nanmean(raw_mean_total_err)),
            "tilted_mean_err": float(np.nanmean(mean_total_err)),
            "raw_err_sd": float(np.nanstd(raw_mean_total_err)),
            "tilted_err_sd": float(np.nanstd(mean_total_err)),
        },
    }

    c, pu, lo = wcounts(M, w, line)
    c0, pu0, lo0 = ev.sim_counts(M, line)
    sm = {"tilt": (c, pu, lo), "raw": (c0, pu0, lo0)}
    out_arms = {}
    cells = {}

    def cell(name, d, mask=None):
        cells[name] = ev.boot_mean(d, inv, nb, rng, mask)

    for tag, (cc, pp, ll) in sm.items():
        raw3 = ((cc + 0.5) / (n + 1.5), (pp + 0.5) / (n + 1.5), (ll + 0.5) / (n + 1.5))
        cond = (cc + 0.5) / (cc + ll + 1.0)
        sim_logit = base.safe_logit(cond)
        X = np.column_stack([served_logit, sim_logit])
        blend_cond, blend_coefs = ev.loso_cond(X, y, nonpush, seasons)
        _, solo_coefs = ev.loso_cond(sim_logit[:, None], y, nonpush, seasons)
        blend3 = ev.three_way(blend_cond, push)
        ll_raw, br_raw = ev.score(raw3, outcome)
        ll_bl, _ = ev.score(blend3, outcome)
        out_arms[tag] = {
            "raw3": raw3,
            "cond": cond,
            "sim_logit": sim_logit,
            "blend_cond": blend_cond,
            "ll_raw": ll_raw,
            "ll_bl": ll_bl,
            "blend_coefs": blend_coefs,
            "solo_coefs": solo_coefs,
            "br_raw": br_raw,
        }

    T = out_arms["tilt"]
    R = out_arms["raw"]
    ll_t, ll_r = T["ll_raw"], R["ll_raw"]
    d1 = ll_served - ll_t
    d4 = ll_recal - T["ll_bl"]
    cell("A5_L1", d1)
    cell("A5_L2", ll_model - ll_t)
    cell("A5_L3", ll_market - ll_t)
    cell("A5_L4", d4)
    cell("A5_L5", ll_served - T["ll_bl"])
    for s_ in base.GRADED_SEASONS:
        m = seasons == s_
        cell(f"A5_S_L1_{s_}", d1, m)
        cell(f"A5_S_L4_{s_}", d4, m)
    pm = outcome == "push"
    cell("A5_P", d1, pm)
    absl = np.abs(line)
    for name, lo_, hi_ in ev.BANDS:
        cell(f"A5_B_{name}", d4, (absl >= lo_ - 1e-9) & (absl <= hi_ + 1e-9))
    whole = np.abs(line - np.rint(line)) < 1e-9
    sim_push_t = pu / n
    real_push = (outcome == "push").astype(float)
    cell("A5_whole_push_mass", sim_push_t - real_push, whole)
    sgn = np.where(line >= 0, 1.0, -1.0)
    fav_sim = M * sgn[:, None]
    fav_real = result * sgn
    key_sim = np.column_stack([(wt * (np.abs(fav_sim - k) < 1e-9)).sum(1) for k in KEYS])
    key_raw = np.column_stack([(np.abs(fav_sim - k) < 1e-9).mean(1) for k in KEYS])
    key_real = np.column_stack([(np.abs(fav_real - k) < 1e-9).astype(float) for k in KEYS])
    key_summary = {}
    for j, k in enumerate(KEYS):
        cell(f"A5_K{k}", key_sim[:, j] - key_real[:, j])
        key_summary[str(k)] = {
            "tilt": float(key_sim[:, j].mean()),
            "raw": float(key_raw[:, j].mean()),
            "real": float(key_real[:, j].mean()),
        }
    band_summary = {}
    for name, lo_, hi_ in SHAPE_BANDS:
        mk = (absl >= lo_ - 1e-9) & (absl <= hi_ + 1e-9)
        band_summary[name] = {
            "n_games": int(mk.sum()),
            **{
                str(k): {
                    "tilt": float(key_sim[mk, j].mean()),
                    "raw": float(key_raw[mk, j].mean()),
                    "real": float(key_real[mk, j].mean()),
                }
                for j, k in enumerate(KEYS)
            },
        }
        cell(f"A5_K3_band_{name}", key_sim[:, 0] - key_real[:, 0], mk)

    cond_t = T["cond"]
    two_served = two_way_loss(1 / (1 + np.exp(-served_logit)), y)
    two_tilt = two_way_loss(cond_t, y)
    two_raw = two_way_loss(R["cond"], y)
    two_market = two_way_loss(m3[0] / np.clip(m3[0] + m3[2], base.EPS, None), y)
    res["two_way_cover_miss"] = {
        "served_minus_tilt": ev.boot_mean(two_served - two_tilt, inv, nb, rng, nonpush),
        "market_minus_tilt": ev.boot_mean(two_market - two_tilt, inv, nb, rng, nonpush),
        "served_minus_raw": ev.boot_mean(two_served - two_raw, inv, nb, rng, nonpush),
    }
    wl = whole
    res["whole_line_three_way_served_minus_tilt_descriptive"] = float(d1[wl].mean())
    res["log_loss"] = {
        "served": float(ll_served.mean()),
        "model_only": float(ll_model.mean()),
        "market_hist": float(ll_market.mean()),
        "recal": float(ll_recal.mean()),
        "sim_raw": float(ll_r.mean()),
        "sim_tilt": float(ll_t.mean()),
        "blend_raw": float(R["ll_bl"].mean()),
        "blend_tilt": float(T["ll_bl"].mean()),
        "brier_sim_raw": float(R["br_raw"].mean()),
        "brier_sim_tilt": float(T["br_raw"].mean()),
    }
    res["cells"] = cells
    res["key_mass_by_all"] = key_summary
    res["key_mass_by_band"] = band_summary
    res["fold_coefs_blend_intercept_served_sim"] = {
        "tilt": T["blend_coefs"],
        "raw": R["blend_coefs"],
    }
    res["fold_coefs_solo_intercept_sim"] = {"tilt": T["solo_coefs"], "raw": R["solo_coefs"]}
    nb_ = nonpush
    res["pooled_solo_coef"] = {
        "tilt": coef_row(T["sim_logit"][nb_, None], y[nb_], inv[nb_], rng),
        "raw": coef_row(R["sim_logit"][nb_, None], y[nb_], inv[nb_], rng),
    }
    res["pooled_blend_coef_served_sim"] = coef_row(
        np.column_stack([served_logit, T["sim_logit"]])[nb_], y[nb_], inv[nb_], rng
    )
    sv_pick = served_logit > 0
    yy = y > 0.5
    mk_pick = m3[0] > m3[2]
    rec_pick = recal_cond > 0.5
    bl_pick = T["blend_cond"] > 0.5
    res["decisive"] = {}
    for nm, pick in (("tilt", cond_t > 0.5), ("raw", R["cond"] > 0.5)):
        for ref, other in (("served", sv_pick), ("market_hist", mk_pick)):
            d = nonpush & (pick != other)
            res["decisive"][f"{nm}_vs_{ref}"] = {
                "n_differ": int(d.sum()),
                "sim_right": int((pick == yy)[d].sum()),
                "other_right": int((other == yy)[d].sum()),
            }
    d = nonpush & (bl_pick != rec_pick)
    res["decisive"]["blend_tilt_vs_recal"] = {
        "n_differ": int(d.sum()),
        "blend_right": int((bl_pick == yy)[d].sum()),
        "recal_right": int((rec_pick == yy)[d].sum()),
    }
    res["reliability_tilt_cover"] = ev.reliability(T["raw3"][0], (outcome == "cover").astype(float))
    res["n_cells_recorded"] = len(cells)
    (OUT / "grade.json").write_text(json.dumps(res, indent=1))

    units = {}
    for name in cells:
        units[name] = (
            "rate_pp" if "_K" in name or name == "A5_whole_push_mass" else "log_loss_improvement"
        )
    batch_cells = []
    for name, v in cells.items():
        scale = 100.0 if units[name] == "rate_pp" else 1.0
        batch_cells.append(
            {
                "name": f"sim08_a5_{a.run}_{name[3:].lower()}"
                if a.run != "main"
                else f"sim08_a5_{name[3:].lower()}",
                "description": f"Amendment 5 tilted simulator, cell {name[3:]}, block bootstrap",
                "effect": round(v["effect"] * scale, 6),
                "effect_units": units[name],
                "standard_error": round(v["se"] * scale, 6) if "se" in v else None,
                "interval_low": round(v["ci95"][0] * scale, 6),
                "interval_high": round(v["ci95"][1] * scale, 6),
                "probability_positive": v["probability_positive"],
                "sample_games": v["n_games"],
                "sample_blocks": nb,
            }
        )
    batch = {
        "family": "sim08_market_anchored",
        "source": f"{OUT.relative_to(REPO).as_posix()}/grade.json",
        "classification": "unresolved_below_power",
        "classification_evidence": (
            "No admissible closing ground; zero crossing closes nothing; "
            "second look at saved 2026-11-03 draws under Amendment 5"
        ),
        "league": "nfl",
        "season_start": 2020,
        "season_end": 2025,
        "category": "modeling",
        "notes": (
            "Exponential tilt of saved draws to the opener line; 2026-11-03 engine without "
            "Amendment 4 terminal transitions. Log-loss cells: positive means tilted simulator "
            "better. Key-number cells: simulated minus real mass, percentage points"
        ),
        "cells": batch_cells,
    }
    (OUT / "record_batch.json").write_text(json.dumps(batch, indent=1))


if __name__ == "__main__":
    main()
