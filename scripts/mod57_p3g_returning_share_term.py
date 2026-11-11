from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lead98_price_skew import BASE, fit, loso, metrics, predict, summary
from mod57_p2q_market_reaction_speed import EARLY, surprise_values
from mod57_p2r_early_overreaction_term import (
    ARTIFACT,
    FORWARD_COEFFICIENTS,
    FORWARD_GAMES,
    SCHEDULES,
    load_reg,
)
from mod57_u3_deadline_fair_line import block_bootstrap, decisive
from per_flag_weights_eval import SEED

from nfl_ats.pool_hook import crossed_push_mass, tuesday_market_lines

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/mod57_p3g"
FEATURES = ROOT / "artifacts/mod57_p3c/team_change_features.parquet"
FAMILY = "mod57_returning_share_stack_term"
PLACEBO = (7, 12)
PERMUTATIONS = 500
OFF = "diff_ret_snap_share_off"
DEF = "diff_ret_snap_share_def"
T_TERM = "early_surprise"
SHIFT = "early_indicator"
WEEKS_26 = "weeks_2_6"
ARMS = {
    "A_off": ("ret_off_early",),
    "B_off_def": ("ret_off_early", "ret_def_early"),
    "C_off_plus_T": ("ret_off_early", T_TERM),
}
CONTROLS = {
    "placebo_off_weeks_7_12": ("ret_off_placebo",),
    "intercept_shift_only": (SHIFT,),
}


def make_terms(data, early):
    week = data["week"]
    in_early = week.between(*EARLY)
    data["ret_off_early"] = data[OFF].where(in_early, 0.0).fillna(0.0)
    data["ret_def_early"] = data[DEF].where(in_early, 0.0).fillna(0.0)
    data["ret_off_placebo"] = data[OFF].where(week.between(*PLACEBO), 0.0).fillna(0.0)
    data[SHIFT] = in_early.astype(float)
    data[T_TERM] = data["D2"].where(in_early, 0.0).fillna(0.0)
    return data


def improvement(entry, name):
    d = entry[name]
    if name == "accuracy_points":
        return d["delta_mean"], d["ci95"][0], d["ci95"][1], d["probability_positive"]
    return -d["delta_mean"], -d["ci95"][1], -d["ci95"][0], d["probability_positive"]


def permutation_gain(data, y, early_mask, base_probs, rng):
    base_ll = -(
        y * np.log(np.clip(base_probs, 1e-9, 1 - 1e-9))
        + (1 - y) * np.log1p(-np.clip(base_probs, 1e-9, 1 - 1e-9))
    )
    gains, accs = [], []
    work = data.copy()
    base_hit = ((base_probs >= 0.5) == (y == 1)).astype(float)
    for _ in range(PERMUTATIONS):
        shuffled = data[OFF].copy()
        for _, idx in data.loc[early_mask].groupby("season").groups.items():
            shuffled.loc[idx] = rng.permutation(data.loc[idx, OFF].to_numpy())
        work["perm"] = shuffled.where(early_mask, 0.0).fillna(0.0)
        oos, _ = loso(work, (*BASE, "perm"))
        p = np.clip(oos, 1e-9, 1 - 1e-9)
        ll = -(y * np.log(p) + (1 - y) * np.log1p(-p))
        hit = ((p >= 0.5) == (y == 1)).astype(float)
        gains.append(float((base_ll - ll)[early_mask].mean()))
        accs.append(float(100 * (hit - base_hit)[early_mask].mean()))
    return np.array(gains), np.array(accs)


def primary(data, rng):
    y = data["home_covered"].to_numpy(float)
    early = data["week"].between(*EARLY).to_numpy()
    pops = {"all_games": np.ones(len(data), bool), WEEKS_26: early}
    variants = {"four_term": BASE}
    variants.update({k: (*BASE, *v) for k, v in ARMS.items()})
    variants.update({k: (*BASE, *v) for k, v in CONTROLS.items()})
    probs, folds, insample = {}, {}, {}
    for name, feats in variants.items():
        probs[name], folds[name] = loso(data, feats)
        b, m, s = fit(data, feats)
        insample[name] = predict(data, feats, b, m, s)
    out = {"folds": {}, "fold_summary": {}, "pops": {}, "frozen": {}}
    for name in variants:
        if name != "four_term":
            out["folds"][name] = folds[name]
            out["fold_summary"][name] = summary(folds[name])
    for pop, mask in pops.items():
        sub = data.loc[mask].reset_index(drop=True)
        ys = y[mask]
        entry = {"games": int(mask.sum()), "variants": {}, "paired": {}, "decisive": {}}
        for name in variants:
            oos = metrics(probs[name][mask], ys)
            ins = metrics(insample[name][mask], ys)
            entry["variants"][name] = {
                "oos": oos,
                "in_sample": ins,
                "gap_in_minus_oos_accuracy_points": 100.0 * (ins["accuracy"] - oos["accuracy"]),
            }
            if name != "four_term":
                entry["paired"][name] = block_bootstrap(
                    sub, probs[name][mask], probs["four_term"][mask], rng
                )
                entry["decisive"][name] = decisive(probs[name][mask], probs["four_term"][mask], ys)
        out["pops"][pop] = entry
    for name in ARMS:
        b, m, s = fit(data, variants[name])
        out["frozen"][name] = {
            "features": list(variants[name]),
            "beta": [float(x) for x in b],
            "means": m,
            "stds": s,
        }
    gains, accs = permutation_gain(data, y, early, probs["four_term"], rng)
    p = np.clip(probs["A_off"], 1e-9, 1 - 1e-9)
    pb = np.clip(probs["four_term"], 1e-9, 1 - 1e-9)
    real_ll = float(
        (
            (-(y * np.log(pb) + (1 - y) * np.log1p(-pb)))
            - (-(y * np.log(p) + (1 - y) * np.log1p(-p)))
        )[early].mean()
    )
    real_acc = float(
        100
        * (((p >= 0.5) == (y == 1)).astype(float) - ((pb >= 0.5) == (y == 1)).astype(float))[
            early
        ].mean()
    )
    out["permutation"] = {
        "permutations": PERMUTATIONS,
        "real_log_loss_gain_weeks_2_6": real_ll,
        "real_accuracy_points_weeks_2_6": real_acc,
        "share_permuted_log_loss_gain_at_or_above_real": float((gains >= real_ll).mean()),
        "share_permuted_accuracy_at_or_above_real": float((accs >= real_acc).mean()),
        "permuted_log_loss_gain_quantiles": {
            str(q): float(np.quantile(gains, q)) for q in (0.05, 0.5, 0.95, 0.99)
        },
        "permuted_accuracy_quantiles": {
            str(q): float(np.quantile(accs, q)) for q in (0.05, 0.5, 0.95, 0.99)
        },
    }
    return out


def forward(frozen, feats26, d2_26):
    d = pd.read_csv(FORWARD_GAMES)
    coefficients = json.loads(FORWARD_COEFFICIENTS.read_text(encoding="utf-8"))
    c = coefficients["coefficients"]
    push = {float(a): float(m) for a, m in coefficients["hook_push_mass"]}
    anchor = {}
    for week, g in d.groupby("week"):
        anchor.update(tuesday_market_lines(ROOT / "data", 2026, int(week), g["game_id"].tolist()))
    d["anchor"] = d["game_id"].map(anchor)
    masses, dirs = [], []
    for r in d.itertuples():
        m = (
            0.0
            if pd.isna(r.anchor)
            else crossed_push_mass(float(r.anchor), float(r.pool_home_spread), push)
        )
        masses.append(m)
        dirs.append(0.0 if m <= 0 else float(np.sign(r.pool_home_spread - r.anchor)))
    half = [0.5 * (1 - x * m) for x, m in zip(dirs, masses, strict=True)]
    d["hook"] = [0.0 if x == 0 else np.log(h / (1 - h)) for x, h in zip(dirs, half, strict=True)]
    base = d["raw_model_home_prob_final"].clip(1e-6, 1 - 1e-6)
    mv = d["served_move"].fillna(0.0)
    av = d["served_move_available"].fillna(0.0).astype(float)
    z = (
        c["intercept"]
        + c["model_logit"] * np.log(base / (1 - base))
        + c["move_toward_home"] * mv
        + c["move_available"] * av
        + c["pool_hook"] * d["hook"]
    ).to_numpy()
    current = 1.0 / (1.0 + np.exp(-z))
    week = d["week"]
    early = week.between(*EARLY)
    cols = {
        "ret_off_early": d["game_id"].map(feats26[OFF]).where(early, 0.0).fillna(0.0),
        "ret_def_early": d["game_id"].map(feats26[DEF]).where(early, 0.0).fillna(0.0),
        T_TERM: d["game_id"].map(d2_26).where(early, 0.0).fillna(0.0),
    }
    y = d["hc"].to_numpy(float)
    mask = early.to_numpy()
    probs = {"current": current}
    for name, fz in frozen.items():
        contribution = np.zeros(len(d))
        for i, f in enumerate(fz["features"]):
            if f in cols:
                contribution += (
                    fz["beta"][i + 1] * (cols[f].to_numpy() - fz["means"][f]) / fz["stds"][f]
                )
        probs[name] = 1.0 / (1.0 + np.exp(-(z + contribution)))
    sub = d.loc[mask, ["week", "hc"]].assign(season=2026).reset_index(drop=True)
    sub["home_covered"] = sub["hc"]
    rng = np.random.default_rng(SEED)
    out = {
        "games_weeks_2_6": int(mask.sum()),
        "games_all": len(d),
        "weeks_in_file": sorted(int(w) for w in week.unique()),
        "term_nonzero_games_A": int((cols["ret_off_early"].to_numpy()[mask] != 0).sum()),
        "metrics": {k: metrics(p[mask], y[mask]) for k, p in probs.items()},
        "paired": {},
        "decisive": {},
        "changed_games": {},
    }
    for name in frozen:
        out["paired"][name] = block_bootstrap(sub, probs[name][mask], current[mask], rng)
        out["decisive"][name] = decisive(probs[name][mask], current[mask], y[mask])
        ch = ((probs[name] >= 0.5) != (current >= 0.5)) & mask
        out["changed_games"][name] = (
            d.loc[ch, ["game_id", "week", "hc"]]
            .assign(current=current[ch], new=probs[name][ch])
            .to_dict("records")
        )
    return out


def forward_inputs():
    snapshot = sorted(p for p in SCHEDULES.iterdir() if (p / "schedules.parquet").is_file())[-1]
    sched = pd.read_parquet(snapshot / "schedules.parquet")
    played = sched.loc[
        (sched["season"] == 2026) & sched["result"].notna() & sched["spread_line"].notna()
    ]
    reg26 = pd.DataFrame(
        {
            "game_id": played["game_id"],
            "season": played["season"],
            "gameday": pd.to_datetime(played["gameday"]),
            "home_team": played["home_team"],
            "away_team": played["away_team"],
            "margin": played["result"].astype(float),
            "close_line": played["spread_line"].astype(float),
        }
    ).reset_index(drop=True)
    return surprise_values(reg26)


def cells(res, fw):
    out = []
    entry = res["pops"][WEEKS_26]
    for arm in ARMS:
        for metric, units in (
            ("accuracy_points", "accuracy_points"),
            ("log_loss", "log_loss_improvement"),
        ):
            eff, lo, hi, pp = improvement(entry["paired"][arm], metric)
            out.append(
                {
                    "name": f"mod57_p3g_{arm}_{WEEKS_26}_{metric}",
                    "description": (
                        f"Arm {arm}: returning offensive snap share (home minus away) times a "
                        f"weeks 2-6 indicator added as fitted term(s) to the served four-term "
                        f"probability, leave-one-season-out at the opener, weeks 2-6 games, "
                        f"{metric} versus the four-term, week-block bootstrap; improvement "
                        "positive"
                    ),
                    "effect": eff,
                    "effect_units": units,
                    "interval_low": lo,
                    "interval_high": hi,
                    "probability_positive": pp,
                    "season_start": 2020,
                    "season_end": 2025,
                    "sample_games": entry["games"],
                }
            )
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reg = load_reg()
    d2 = surprise_values(reg)
    rng = np.random.default_rng(SEED)
    data = pd.read_parquet(ARTIFACT)
    data = data.loc[data["season"].between(2020, 2025)].reset_index(drop=True)
    feats = pd.read_parquet(FEATURES).set_index("game_id")
    data[OFF] = data["game_id"].map(feats[OFF])
    data[DEF] = data["game_id"].map(feats[DEF])
    data["D2"] = d2.reindex(data["game_id"]).to_numpy()
    early = data["week"].between(*EARLY)
    data = make_terms(data, early)
    res = primary(data, rng)
    feats26 = {OFF: feats[OFF], DEF: feats[DEF]}
    fw = forward(res["frozen"], feats26, forward_inputs())
    results = {
        "games": len(data),
        "early_games_missing_input": int((early & data[OFF].isna()).sum()),
        "primary": res,
        "forward": fw,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    batch = {
        "league": "nfl",
        "classification": "unresolved_below_power",
        "source": "artifacts/mod57_p3g/results.json",
        "category": "market",
        "family": FAMILY,
        "notes": (
            "Returning offensive snap share (home minus away) in weeks 2-6 as fitted term(s) in "
            "the four-term pick probability, leave-one-season-out at the opener 2020-25, "
            "with permutation, placebo, intercept-shift and frozen 2026 reads in the source."
        ),
        "classification_evidence": (
            "Week-block bootstrap, probability_positive reported; no refuted mechanism and no "
            "closing ground."
        ),
        "plain_summary": (
            "Whether how much of last season's playing time a roster keeps adds to the pick "
            "probability in the first weeks."
        ),
        "cells": cells(res, fw),
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print("games", len(data), "missing input early", results["early_games_missing_input"])
    for pop, e in res["pops"].items():
        print("POP", pop, e["games"])
        for name, v in e["variants"].items():
            print(
                f"  {name} oos {v['oos']['record']} acc {v['oos']['accuracy']:.4f} "
                f"ll {v['oos']['log_loss']:.5f} in {v['in_sample']['accuracy']:.4f} "
                f"gap {v['gap_in_minus_oos_accuracy_points']:+.2f}"
            )
        for name, dec in e["decisive"].items():
            p = e["paired"][name]
            print(
                f"  vs four {name} changed {dec['changed_pick_games']} new {dec['new_wins']} "
                f"base {dec['base_wins']} | "
                + " | ".join(
                    f"{m} {p[m]['delta_mean']:+.5f} "
                    f"[{p[m]['ci95'][0]:+.5f},{p[m]['ci95'][1]:+.5f}] "
                    f"P+ {p[m]['probability_positive']:.3f}"
                    for m in ("accuracy_points", "log_loss")
                )
            )
    for k, folds in res["folds"].items():
        print(
            "FOLDS",
            k,
            {s: {n: round(v, 4) for n, v in f.items() if n not in BASE} for s, f in folds.items()},
        )
    print("PERM", json.dumps(res["permutation"]))
    print("FORWARD", json.dumps(fw, default=str))


if __name__ == "__main__":
    main()
