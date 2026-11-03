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
FIELD_FILE = Path("data/splash/field/2026_field_best_picks_w01-08.parquet")
BOARD_GLOB = "2026_week0[1-8]_*.json"
BOARD_DIR = Path("data/splash")
OUT = Path("artifacts/pool_rank_replay/pool05_ownership_v2.json")
LAMBDAS = [1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
BOOT_DRAWS = 4000
BOOT_SEED = 5
INNER_FOLDS = 7
OUTER_FOLDS = 8
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


def load_games():
    raw = pd.read_parquet(FIELD_FILE)
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


def main():
    argparse.ArgumentParser().parse_args()
    df = load_games()
    teams = sorted(set(df["home"]) | set(df["away"]))
    slots = sorted(set(df["slot"]) - {"sun_early"})
    wts = (df["n_home"] + df["n_away"]).to_numpy()
    weeks = df["week"].to_numpy()
    base_ll = game_ll(baseline_lowo(df), df["n_o"].to_numpy(), df["n_t"].to_numpy())
    out = {
        "family": FAMILY,
        "games": len(df),
        "slot_counts": df["slot"].value_counts().to_dict(),
        "baseline_definition": "per-pick log loss, model-side share on line, leave-one-week-out",
        "baseline_log_loss": float(-base_ll.sum() / wts.sum()),
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
    out["looks"] = {
        "outer_cells": len(MODELS),
        "outer_folds_each": OUTER_FOLDS,
        "inner_lambda_fits": len(LAMBDAS) * INNER_FOLDS * OUTER_FOLDS * len(MODELS),
        "primary": PRIMARY,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"games {len(df)} slots {out['slot_counts']}")
    print(f"baseline {out['baseline_log_loss']:.4f}")
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
