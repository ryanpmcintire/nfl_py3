import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from nfl_ats.evidence_conventions import probability_positive_from_draws

FAMILY = "pool05_ownership_v2"
FIELD_DIR = Path("data/splash/field")
BOARD_GLOB = "2026_week*_*.json"
BOARD_DIR = Path("data/splash")
OUT_DIR = Path("artifacts/pool_rank_replay")
LAMBDAS = [1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
BOOT_DRAWS = 4000
BOOT_SEED = 5
PERM_DRAWS = 20000
TEAM_SEED = 55
PP = 100.0
TAIL = 0.025
MODELS = {
    "M0h_home_spread": {"slot": False, "team": False, "curve": False},
    "M1_slot": {"slot": True, "team": False, "curve": False},
    "M2_team": {"slot": False, "team": True, "curve": False},
    "M3_slot_team": {"slot": True, "team": True, "curve": False},
    "M4_slot_team_curve": {"slot": True, "team": True, "curve": True},
}
PRIMARY = "M3_slot_team"
SLOT_EARLY_END_HOUR = 15
SLOT_LATE_END_HOUR = 19
THU, FRI, SAT, MON = 3, 4, 5, 0


def slot_of(kick):
    t = datetime.fromisoformat(kick)
    d, h = t.weekday(), t.hour
    if d == THU:
        return "thu"
    if d == MON:
        return "mon"
    if d in (FRI, SAT):
        return "other"
    if h < SLOT_EARLY_END_HOUR:
        return "sun_early"
    if h < SLOT_LATE_END_HOUR:
        return "sun_late"
    return "sun_night"


def load_games(last_week):
    raw = pd.read_parquet(FIELD_DIR / f"2026_field_best_picks_w01-{last_week:02d}.parquet")
    kicks = {}
    for f in sorted(BOARD_DIR.glob(BOARD_GLOB)):
        for g in json.loads(f.read_text(encoding="utf-8"))["games"]:
            kicks[g["game_id"]] = g["kickoff_et"]
    rows = []
    for gid, g in raw.groupby("game_id"):
        h = g[g["home"]].iloc[0]
        a = g[~g["home"]].iloc[0]
        o = g.loc[g["model_p"].idxmax()]
        t = g.drop(o.name).iloc[0]
        rows.append(
            {
                "game_id": gid,
                "week": int(h["week"]),
                "home": h["team"],
                "away": a["team"],
                "x_home": float(-h["line"]),
                "n_home": float(h["picks"]),
                "n_away": float(a["picks"]),
                "x_o": float(-o["line"]),
                "n_o": float(o["picks"]),
                "n_t": float(t["picks"]),
                "o_home": bool(o["home"]),
                "slot": slot_of(kicks[gid]),
            }
        )
    return pd.DataFrame(rows)


def design(df, spec, teams, slots):
    x = df["x_home"].to_numpy()
    cols = [np.ones(len(df)), x]
    names = ["intercept", "spread"]
    if spec["curve"]:
        cols.append(x * np.abs(x))
        names.append("spread_curve")
    n_base = len(cols)
    if spec["slot"]:
        for s in slots:
            cols.append((df["slot"] == s).to_numpy().astype(float))
            names.append("slot_" + s)
    if spec["team"]:
        for t in teams:
            cols.append(
                (df["home"] == t).to_numpy().astype(float)
                - (df["away"] == t).to_numpy().astype(float)
            )
            names.append("team_" + t)
    return np.column_stack(cols), names, n_base


def fit(X, n1, n0, n_base, lam):
    pen = np.ones(X.shape[1])
    pen[:n_base] = 0.0

    def f(b):
        q = np.clip(expit(X @ b), 1e-12, 1 - 1e-12)
        nll = -(n1 * np.log(q) + n0 * np.log(1 - q)).sum() + 0.5 * lam * (pen * b * b).sum()
        grad = X.T @ ((n1 + n0) * q - n1) + lam * pen * b
        return nll, grad

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def game_ll(q, n1, n0):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return n1 * np.log(q) + n0 * np.log(1 - q)


def inner_select(df, spec, teams, slots):
    weeks = sorted(df["week"].unique())
    best, best_ll = None, -np.inf
    for lam in LAMBDAS:
        tot = 0.0
        for w in weeks:
            tr, te = df[df["week"] != w], df[df["week"] == w]
            Xtr, _, nb = design(tr, spec, teams, slots)
            Xte, _, _ = design(te, spec, teams, slots)
            b = fit(Xtr, tr["n_home"].to_numpy(), tr["n_away"].to_numpy(), nb, lam)
            tot += game_ll(expit(Xte @ b), te["n_home"].to_numpy(), te["n_away"].to_numpy()).sum()
        if tot > best_ll:
            best, best_ll = lam, tot
    return best


def lowo_model(df, spec, teams, slots):
    pred = np.zeros(len(df))
    folds = []
    for w in sorted(df["week"].unique()):
        tr = df[df["week"] != w]
        te_mask = (df["week"] == w).to_numpy()
        lam = inner_select(tr, spec, teams, slots)
        Xtr, names, nb = design(tr, spec, teams, slots)
        b = fit(Xtr, tr["n_home"].to_numpy(), tr["n_away"].to_numpy(), nb, lam)
        Xte, _, _ = design(df[te_mask], spec, teams, slots)
        pred[te_mask] = expit(Xte @ b)
        core = {n: float(v) for n, v in zip(names, b, strict=True) if not n.startswith("team_")}
        tm = np.array([v for n, v in zip(names, b, strict=True) if n.startswith("team_")])
        folds.append(
            {
                "held_out_week": int(w),
                "lambda": lam,
                "coef": core,
                "team_effect_sd": float(tm.std()) if len(tm) else 0.0,
            }
        )
    return pred, folds


def baseline_lowo(df):
    pred = np.zeros(len(df))
    for w in sorted(df["week"].unique()):
        tr = df[df["week"] != w]
        te_mask = (df["week"] == w).to_numpy()
        X = np.column_stack([np.ones(len(tr)), tr["x_o"].to_numpy()])
        b = fit(X, tr["n_o"].to_numpy(), tr["n_t"].to_numpy(), 2, 0.0)
        Xe = np.column_stack([np.ones(te_mask.sum()), df.loc[te_mask, "x_o"].to_numpy()])
        pred[te_mask] = expit(Xe @ b)
    return pred


def boot(diff, weights, weeks, rng, by_week):
    if by_week:
        uw = np.unique(weeks)
        num = np.array([diff[weeks == w].sum() for w in uw])
        den = np.array([weights[weeks == w].sum() for w in uw])
        idx = rng.integers(0, len(uw), (BOOT_DRAWS, len(uw)))
        return num[idx].sum(1) / den[idx].sum(1)
    idx = rng.integers(0, len(diff), (BOOT_DRAWS, len(diff)))
    return diff[idx].sum(1) / weights[idx].sum(1)


def interval(d):
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def split_half(df, spec, teams, slots, lam):
    res = {}
    for name, grp in (("odd", df["week"] % 2 == 1), ("even", df["week"] % 2 == 0)):
        sub = df[grp]
        X, names, nb = design(sub, spec, teams, slots)
        b = fit(X, sub["n_home"].to_numpy(), sub["n_away"].to_numpy(), nb, lam)
        res[name] = {
            n[5:]: float(v) for n, v in zip(names, b, strict=True) if n.startswith("team_")
        }
    common = sorted(set(res["odd"]) & set(res["even"]))
    a = np.array([res["odd"][t] for t in common])
    c = np.array([res["even"][t] for t in common])
    return float(np.corrcoef(a, c)[0, 1])


def team_matrix(df, teams):
    return np.column_stack(
        [
            (df["home"] == t).to_numpy().astype(float) - (df["away"] == t).to_numpy().astype(float)
            for t in teams
        ]
    )


def side_matrix(df, teams):
    side = np.where(df["o_home"].to_numpy(), 1.0, -1.0)
    return team_matrix(df, teams) * side[:, None]


def fit_offset(T, z, n1, n0, lam):
    def f(b):
        q = np.clip(expit(z + T @ b), 1e-12, 1 - 1e-12)
        nll = -(n1 * np.log(q) + n0 * np.log(1 - q)).sum() + 0.5 * lam * (b * b).sum()
        grad = T.T @ ((n1 + n0) * q - n1) + lam * b
        return nll, grad

    return minimize(f, np.zeros(T.shape[1]), jac=True, method="L-BFGS-B").x


def line_offsets(tr, te):
    X = np.column_stack([np.ones(len(tr)), tr["x_o"].to_numpy()])
    b = fit(X, tr["n_o"].to_numpy(), tr["n_t"].to_numpy(), 2, 0.0)
    return b[0] + b[1] * tr["x_o"].to_numpy(), b[0] + b[1] * te["x_o"].to_numpy()


def residual_fit(tr, te, teams, lam):
    ztr, zte = line_offsets(tr, te)
    b = fit_offset(side_matrix(tr, teams), ztr, tr["n_o"].to_numpy(), tr["n_t"].to_numpy(), lam)
    return expit(zte + side_matrix(te, teams) @ b), b


def residual_select(tr, teams):
    best, best_ll = None, -np.inf
    for lam in LAMBDAS:
        tot = 0.0
        for v in sorted(tr["week"].unique()):
            e = tr[tr["week"] == v]
            q, _ = residual_fit(tr[tr["week"] != v], e, teams, lam)
            tot += game_ll(q, e["n_o"].to_numpy(), e["n_t"].to_numpy()).sum()
        if tot > best_ll:
            best, best_ll = lam, tot
    return best


def residual_lowo(df, teams):
    pred = np.zeros(len(df))
    folds = []
    for w in sorted(df["week"].unique()):
        mask = (df["week"] == w).to_numpy()
        lam = residual_select(df[~mask], teams)
        pred[mask], b = residual_fit(df[~mask], df[mask], teams, lam)
        folds.append({"held_out_week": int(w), "lambda": lam, "team_effect_sd": float(b.std())})
    return pred, folds


def team_stats(D, counts, r, odd):
    means = (D.T @ r) / counts
    a = (D[odd].T @ r[odd]) / np.abs(D[odd]).sum(0)
    c = (D[~odd].T @ r[~odd]) / np.abs(D[~odd]).sum(0)
    return (
        float((counts * means * means).sum()),
        float(np.abs(means).max()),
        float(means.std()),
        float(np.corrcoef(a, c)[0, 1]),
    )


def team_residual_unit(df, teams, base_pred, base_ll, wts, weeks):
    rng = np.random.default_rng(TEAM_SEED)
    n_o, n_t = df["n_o"].to_numpy(), df["n_t"].to_numpy()
    share_o = n_o / (n_o + n_t)
    pred, folds = residual_lowo(df, teams)
    diff = game_ll(pred, n_o, n_t) - base_ll
    d_games = boot(diff, wts, weeks, rng, False)
    d_weeks = boot(diff, wts, weeks, rng, True)
    err_base = PP * np.abs(share_o - base_pred)
    err_team = PP * np.abs(share_o - pred)
    d_mae = boot(err_base - err_team, np.ones(len(df)), weeks, rng, False)
    side = np.where(df["o_home"].to_numpy(), 1.0, -1.0)
    resid_home = PP * side * (share_o - base_pred)
    D = team_matrix(df, teams)
    counts = np.abs(D).sum(0)
    odd = weeks % 2 == 1
    obs = team_stats(D, counts, resid_home, odd)
    null = np.zeros((PERM_DRAWS, len(obs)))
    week_idx = [np.where(weeks == w)[0] for w in np.unique(weeks)]
    for i in range(PERM_DRAWS):
        r = resid_home.copy()
        for idx in week_idx:
            r[idx] = rng.permutation(resid_home[idx])
        null[i] = team_stats(D, counts, r, odd)
    effects = []
    for j, t in enumerate(teams):
        vals = (D[:, j] * resid_home)[D[:, j] != 0]
        draws = vals[rng.integers(0, len(vals), (BOOT_DRAWS, len(vals)))].mean(1)
        effects.append(
            {
                "team": t,
                "games": len(vals),
                "effect_pp": float(vals.mean()),
                "interval": interval(draws),
                "probability_positive": float(probability_positive_from_draws(draws)),
            }
        )
    effects.sort(key=lambda e: e["effect_pp"])
    means = np.array([e["effect_pp"] for e in effects])
    r_draws = []
    a = (D[odd].T @ resid_home[odd]) / np.abs(D[odd]).sum(0)
    c = (D[~odd].T @ resid_home[~odd]) / np.abs(D[~odd]).sum(0)
    for _ in range(BOOT_DRAWS):
        k = rng.integers(0, len(teams), len(teams))
        r_draws.append(np.corrcoef(a[k], c[k])[0, 1])
    r_draws = np.array(r_draws)
    return {
        "definition": "team effect on the leave-one-week-out line-proxy residual; "
        "model-side share logit = line-proxy offset + ridge team effect",
        "oow_log_loss": float(-game_ll(pred, n_o, n_t).sum() / wts.sum()),
        "improvement_vs_baseline": float(diff.sum() / wts.sum()),
        "interval_games_bootstrap": interval(d_games),
        "probability_positive_games": float(probability_positive_from_draws(d_games)),
        "interval_weeks_bootstrap": interval(d_weeks),
        "probability_positive_weeks": float(probability_positive_from_draws(d_weeks)),
        "weeks_improved": int(sum(diff[weeks == w].sum() > 0 for w in np.unique(weeks))),
        "mae_pp_team_model": float(err_team.mean()),
        "mae_improvement_pp": float((err_base - err_team).mean()),
        "mae_improvement_interval": interval(d_mae),
        "mae_improvement_probability_positive": float(probability_positive_from_draws(d_mae)),
        "folds": folds,
        "between_team_statistic": obs[0],
        "between_team_permutation_p": float((null[:, 0] >= obs[0]).mean()),
        "largest_abs_team_effect_pp": obs[1],
        "largest_abs_team_effect_permutation_p": float((null[:, 1] >= obs[1]).mean()),
        "team_effect_sd_pp": obs[2],
        "team_effect_sd_pp_null_mean": float(null[:, 2].mean()),
        "team_effect_sd_pp_null_interval": interval(null[:, 2]),
        "split_half_corr_odd_even": obs[3],
        "split_half_corr_interval_team_bootstrap": interval(r_draws),
        "split_half_corr_probability_positive": float(probability_positive_from_draws(r_draws)),
        "split_half_corr_permutation_p_one_sided": float((null[:, 3] >= obs[3]).mean()),
        "teams_probability_positive_outside_0p025_0p975": int(
            sum(
                e["probability_positive"] > 1 - TAIL or e["probability_positive"] < TAIL
                for e in effects
            )
        ),
        "team_effect_range_pp": [float(means.min()), float(means.max())],
        "team_effects": effects,
        "permutation": "home-oriented residuals shuffled across games within week",
        "permutation_draws": PERM_DRAWS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last-week", type=int, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out_path = args.out or OUT_DIR / f"pool05_ownership_v2_w{args.last_week:02d}.json"
    df = load_games(args.last_week)
    teams = sorted(set(df["home"]) | set(df["away"]))
    slots = sorted(set(df["slot"]) - {"sun_early"})
    wts = (df["n_home"] + df["n_away"]).to_numpy()
    weeks = df["week"].to_numpy()
    outer_folds = len(np.unique(weeks))
    base_pred = baseline_lowo(df)
    base_ll = game_ll(base_pred, df["n_o"].to_numpy(), df["n_t"].to_numpy())
    base_err = PP * (df["n_o"] / (df["n_o"] + df["n_t"]) - base_pred).to_numpy()
    out = {
        "family": FAMILY,
        "games": len(df),
        "last_week": args.last_week,
        "games_per_week": {
            int(w): int(c) for w, c in df["week"].value_counts().sort_index().items()
        },
        "slot_counts": df["slot"].value_counts().to_dict(),
        "baseline_definition": "per-pick log loss, model-side share on line, leave-one-week-out",
        "baseline_log_loss": float(-base_ll.sum() / wts.sum()),
        "baseline_mae_pp": float(np.abs(base_err).mean()),
        "baseline_rmse_pp": float(np.sqrt((base_err**2).mean())),
        "baseline_log_loss_by_week": {
            int(w): float(-base_ll[weeks == w].sum() / wts[weeks == w].sum())
            for w in np.unique(weeks)
        },
        "models": {},
    }
    rng = np.random.default_rng(BOOT_SEED)
    n1, n0 = df["n_home"].to_numpy(), df["n_away"].to_numpy()
    for name, spec in MODELS.items():
        pred, folds = lowo_model(df, spec, teams, slots)
        ll = game_ll(pred, n1, n0)
        X, _, nb = design(df, spec, teams, slots)
        lam_all = inner_select(df, spec, teams, slots)
        b = fit(X, n1, n0, nb, lam_all)
        ll_in = game_ll(expit(X @ b), n1, n0)
        diff = ll - base_ll
        d_games = boot(diff, wts, weeks, rng, False)
        d_weeks = boot(diff, wts, weeks, rng, True)
        rec = {
            "oow_log_loss": float(-ll.sum() / wts.sum()),
            "insample_log_loss": float(-ll_in.sum() / wts.sum()),
            "improvement_vs_baseline": float(diff.sum() / wts.sum()),
            "interval_games_bootstrap": interval(d_games),
            "probability_positive_games": float(probability_positive_from_draws(d_games)),
            "interval_weeks_bootstrap": interval(d_weeks),
            "probability_positive_weeks": float(probability_positive_from_draws(d_weeks)),
            "weeks_improved": int(sum(diff[weeks == w].sum() > 0 for w in np.unique(weeks))),
            "folds": folds,
        }
        rec["gap_oow_minus_insample"] = rec["oow_log_loss"] - rec["insample_log_loss"]
        if spec["team"]:
            lam_med = float(np.median([f["lambda"] for f in folds]))
            rec["split_half_lambda"] = lam_med
            rec["team_effect_split_half_corr_odd_even"] = split_half(
                df, spec, teams, slots, lam_med
            )
        out["models"][name] = rec
    out["team_residual"] = team_residual_unit(df, teams, base_pred, base_ll, wts, weeks)
    out["looks"] = {
        "outer_cells": len(MODELS),
        "outer_folds_each": outer_folds,
        "inner_lambda_fits": len(LAMBDAS) * (outer_folds - 1) * outer_folds * len(MODELS),
        "primary": PRIMARY,
        "team_residual_cells": 5,
        "team_residual_inner_lambda_fits": len(LAMBDAS) * (outer_folds - 1) * outer_folds,
        "team_effect_intervals": len(teams),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    tr = out["team_residual"]
    print(f"games {len(df)} per week {out['games_per_week']} slots {out['slot_counts']}")
    print(
        f"baseline {out['baseline_log_loss']:.4f} mae {out['baseline_mae_pp']:.2f}",
        f"rmse {out['baseline_rmse_pp']:.2f}",
    )
    print(json.dumps({k: v for k, v in tr.items() if k not in ("team_effects", "folds")}, indent=1))
    print("team lams", [f["lambda"] for f in tr["folds"]])
    for e in tr["team_effects"]:
        lo, hi = e["interval"]
        print(
            e["team"],
            e["games"],
            f"{e['effect_pp']:+.2f} [{lo:+.2f},{hi:+.2f}] P+ {e['probability_positive']:.3f}",
        )
    for n, r in out["models"].items():
        lo, hi = r["interval_games_bootstrap"]
        wlo, whi = r["interval_weeks_bootstrap"]
        print(
            n,
            f"oow {r['oow_log_loss']:.4f} in {r['insample_log_loss']:.4f}",
            f"imp {r['improvement_vs_baseline']:+.4f}",
            f"games [{lo:+.4f},{hi:+.4f}] P+ {r['probability_positive_games']:.3f}",
            f"weeks [{wlo:+.4f},{whi:+.4f}] P+ {r['probability_positive_weeks']:.3f}",
            f"wk_imp {r['weeks_improved']}",
            f"sh {r.get('team_effect_split_half_corr_odd_even')}",
            f"lams {[f['lambda'] for f in r['folds']]}",
        )


main()
