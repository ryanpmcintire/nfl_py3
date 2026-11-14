import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
PICKPROB = ROOT / "artifacts/pick_probability/20261114T184614Z/per_game.parquet"
LIVE = ROOT / "artifacts/prospective_scorecard/20261110T143033Z/predictions.csv"
FEATURES = ROOT / "data/processed/game_features.parquet"
OUT = ROOT / "artifacts/mod47"
LANE = "docs/lanes/mod-47-short-week-confidence-slope.md"
FAMILY = "mod47_short_week_side_term"
NATURAL_WEEK_DAYS = 7
COUNTED_FLAGS = ["flag_division", "flag_protection"]
SERVED_INPUTS = ["model_logit", "flag_sum", "market_move_available", "market_move_toward_home"]
FIRST_SEASON = 2020
LAST_SEASON = 2025
LIVE_SEASON = 2026
LIVE_LAST_WEEK = 9
BOOT = 2000
PERMUTATIONS = 1000
SEED = 47
EPS = 1e-6
NEWTON_STEPS = 100
NEWTON_TOL = 1e-10
METRICS = ["accuracy_points", "log_loss_improvement", "brier_improvement"]


def short_table():
    f = pd.read_parquet(
        FEATURES, columns=["game_id", "season", "gameday", "home_team", "away_team"]
    )
    parts = [
        f[["game_id", "season", "gameday", f"{side}_team"]]
        .rename(columns={f"{side}_team": "team"})
        .assign(side=side)
        for side in ("home", "away")
    ]
    long = pd.concat(parts).sort_values(["team", "season", "gameday", "game_id"])
    prev = long.groupby(["team", "season"], sort=False)["gameday"].shift(1)
    long["rest"] = (long["gameday"] - prev).dt.days.astype(float)
    long["short"] = (long["rest"] < NATURAL_WEEK_DAYS).astype(float)
    wide = long.pivot(index="game_id", columns="side", values=["rest", "short"])
    wide.columns = [f"{side}_{name}" for name, side in wide.columns]
    wide = wide.reset_index().merge(f[["game_id", "gameday"]], on="game_id")
    wide["short_diff"] = wide["home_short"] - wide["away_short"]
    wide["weekday"] = wide["gameday"].dt.day_name()
    return wide.drop(columns=["gameday"])


def fit_offset(x, y, offset):
    b = np.zeros(x.shape[1])
    for _ in range(NEWTON_STEPS):
        p = expit(offset + x @ b)
        grad = x.T @ (y - p)
        hess = (x * (p * (1 - p))[:, None]).T @ x
        step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        b = b + step
        if np.abs(step).max() < NEWTON_TOL:
            break
    return b


def loso(x, y, offset, season):
    p = np.empty(len(y))
    coefs = {}
    for s in np.unique(season):
        te = season == s
        b = fit_offset(x[~te], y[~te], offset[~te])
        p[te] = expit(offset[te] + x[te] @ b)
        coefs[int(s)] = [round(float(v), 4) for v in b]
    return p, coefs


def losses(p, y):
    pc = np.clip(p, EPS, 1 - EPS)
    return {
        "acc": ((p >= 0.5) == (y == 1)).astype(float) * 100,
        "ll": -(y * np.log(pc) + (1 - y) * np.log(1 - pc)),
        "brier": (p - y) ** 2,
    }


def deltas(p_new, p_ref, y):
    a, b = losses(p_new, y), losses(p_ref, y)
    return {
        "accuracy_points": a["acc"] - b["acc"],
        "log_loss_improvement": b["ll"] - a["ll"],
        "brier_improvement": b["brier"] - a["brier"],
    }


def contrast(p_new, p_ref, y, rng):
    d = deltas(p_new, p_ref, y)
    n = len(y)
    idx = rng.integers(0, n, (BOOT, n))
    out = {}
    for k, v in d.items():
        bs = v[idx].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[k] = {
            "effect": float(v.mean()),
            "low": float(lo),
            "high": float(hi),
            "pp": float((bs > 0).mean() + 0.5 * (bs == 0).mean()),
        }
    changed = (p_new >= 0.5) != (p_ref >= 0.5)
    new_right = (p_new >= 0.5) == (y == 1)
    out["decisive"] = {
        "games": int(changed.sum()),
        "arm_correct": int((changed & new_right).sum()),
        "reference_correct": int((changed & ~new_right).sum()),
    }
    out["n"] = int(n)
    return out


def point(p_new, p_ref, y):
    return {k: float(v.mean()) for k, v in deltas(p_new, p_ref, y).items()}


def per_season(p_new, p_ref, y, season):
    d = deltas(p_new, p_ref, y)
    return {
        int(s): {k: round(float(v[season == s].mean()), 5) for k, v in d.items()}
        for s in np.unique(season)
    }


def coefficient_boot(x, y, offset, rng):
    n = len(y)
    bs = np.empty(BOOT)
    for i in range(BOOT):
        j = rng.integers(0, n, n)
        bs[i] = fit_offset(x[j], y[j], offset[j])[0]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"low": float(lo), "high": float(hi), "pp": float((bs > 0).mean())}


def permuted_control(sd, y, z, season, rng):
    rows = []
    served = expit(z)
    for _ in range(PERMUTATIONS):
        perm = sd.copy()
        for s in np.unique(season):
            k = np.flatnonzero(season == s)
            perm[k] = sd[rng.permutation(k)]
        p, _ = loso(perm[:, None], y, z, season)
        rows.append(point(p, served, y))
    return pd.DataFrame(rows)


def counts(d):
    t = d.assign(
        both_short=((d.home_short == 1) & (d.away_short == 1)).astype(int),
        home_only=(d.short_diff == 1).astype(int),
        away_only=(d.short_diff == -1).astype(int),
        thursday_nonzero=((d.weekday == "Thursday") & (d.short_diff != 0)).astype(int),
        thursday_both=(
            (d.weekday == "Thursday") & (d.home_short == 1) & (d.away_short == 1)
        ).astype(int),
        games=1,
    )
    cols = [
        "games",
        "home_short",
        "away_short",
        "both_short",
        "home_only",
        "away_only",
        "thursday_nonzero",
        "thursday_both",
    ]
    return t.groupby("season")[cols].sum().astype(int)


def cell(name, description, units, stat, n, lo, hi, category, summary, notes):
    return {
        "name": name,
        "description": description,
        "source": LANE,
        "effect": round(stat["effect"], 6),
        "effect_units": units,
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": lo,
        "season_end": hi,
        "family": FAMILY,
        "category": category,
        "classification_evidence": (
            "No resolved wrong sign, no split-half reliability measured and no positive "
            "control proven at this size; paired game bootstrap interval reported."
        ),
        "notes": notes,
        "interval_low": round(stat["low"], 6),
        "interval_high": round(stat["high"], 6),
        "probability_positive": round(stat["pp"], 4),
        "sample_games": int(n),
        "sample_blocks": int(hi - lo + 1),
        "plain_summary": summary,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    sched = short_table()
    d = pd.read_parquet(PICKPROB)
    d = d[d.margin_vs_open.ne(0) & d.season.between(FIRST_SEASON, LAST_SEASON)]
    d = d.merge(sched, on="game_id", how="left", validate="one_to_one").reset_index(drop=True)
    assert d.short_diff.notna().all()
    d["flag_sum"] = d[COUNTED_FLAGS].sum(axis=1).astype(float)
    assert d[SERVED_INPUTS].notna().all().all()
    y = d.home_covered.to_numpy(dtype=float)
    season = d.season.to_numpy()
    served = d.out_of_season_home_probability.to_numpy()
    z = logit(np.clip(served, EPS, 1 - EPS))
    sd = d.short_diff.to_numpy(dtype=float)
    zero = np.zeros(len(d))
    report = {
        "family": FAMILY,
        "served_artifact": str(PICKPROB.relative_to(ROOT)),
        "n": len(d),
        "flag_sum_matches_composition": float(np.abs(d.flag_sum - d.composition_flag_sum).max()),
    }
    ct = counts(d)
    report["counts"] = ct.to_dict("index")
    print("n", len(d))
    print(ct.to_string())
    nz = d[d.short_diff != 0]
    report["nonzero_weekdays"] = nz.weekday.value_counts().to_dict()
    report["nonzero_rest_pairs"] = (
        nz.groupby(["home_rest", "away_rest"]).size().sort_values(ascending=False).head(8)
    ).to_string()
    print("nonzero weekdays", report["nonzero_weekdays"])
    print(report["nonzero_rest_pairs"])
    short_cov = np.where(nz.short_diff == 1, nz.home_covered == 1, nz.home_covered == 0)
    served_on_short = np.where(
        nz.short_diff == 1,
        nz.out_of_season_home_probability >= 0.5,
        nz.out_of_season_home_probability < 0.5,
    )
    report["raw_short_side"] = {
        "games": len(nz),
        "short_side_covered": int(short_cov.sum()),
        "short_side_failed": int((~short_cov).sum()),
        "served_pick_on_short_side": int(served_on_short.sum()),
    }
    print("raw short side", report["raw_short_side"])

    xa = sd[:, None]
    pa, ca = loso(xa, y, z, season)
    arm_a = contrast(pa, served, y, rng)
    c_full = fit_offset(xa, y, z)
    arm_a["in_sample"] = point(expit(z + xa @ c_full), served, y)
    arm_a["fold_coefficients"] = {k: v[0] for k, v in ca.items()}
    arm_a["full_coefficient"] = float(c_full[0])
    arm_a["coefficient_boot"] = coefficient_boot(xa, y, z, rng)
    arm_a["per_season"] = per_season(pa, served, y, season)
    report["arm_a_offset"] = arm_a

    base = np.column_stack([np.ones(len(d)), d[SERVED_INPUTS].to_numpy(dtype=float)])
    plus = np.column_stack([base, sd])
    pb0, _ = loso(base, y, zero, season)
    pb1, cb = loso(plus, y, zero, season)
    arm_b = contrast(pb1, pb0, y, rng)
    b0, b1 = fit_offset(base, y, zero), fit_offset(plus, y, zero)
    arm_b["in_sample"] = point(expit(plus @ b1), expit(base @ b0), y)
    arm_b["fold_coefficients"] = {k: v[-1] for k, v in cb.items()}
    arm_b["fold_all_coefficients"] = cb
    arm_b["full_coefficient"] = float(b1[-1])
    arm_b["refit_vs_served"] = {
        "correlation": float(np.corrcoef(pb0, served)[0, 1]),
        "served_minus_refit": point(served, pb0, y),
    }
    arm_b["per_season"] = per_season(pb1, pb0, y, season)
    report["arm_b_joint"] = arm_b

    ones = np.ones((len(d), 1))
    pc, cc = loso(ones, y, z, season)
    arm_c = contrast(pc, served, y, rng)
    arm_c["in_sample"] = point(expit(z + ones @ fit_offset(ones, y, z)), served, y)
    arm_c["fold_coefficients"] = {k: v[0] for k, v in cc.items()}
    report["arm_c_intercept_only"] = arm_c

    perm = permuted_control(sd, y, z, season, rng)
    arm_d = {}
    for k in METRICS:
        lo, hi = np.percentile(perm[k], [2.5, 97.5])
        arm_d[k] = {
            "effect": float(perm[k].mean()),
            "low": float(lo),
            "high": float(hi),
            "pp": float((perm[k] > 0).mean() + 0.5 * (perm[k] == 0).mean()),
            "share_at_or_above_observed": float((perm[k] >= arm_a[k]["effect"]).mean()),
        }
    report["arm_d_permuted"] = arm_d

    lv = pd.read_csv(LIVE)
    lv = lv[lv.y.isin([0, 1]) & lv.season.eq(LIVE_SEASON) & lv.week.le(LIVE_LAST_WEEK)]
    lv = lv.merge(sched, on="game_id", how="left", validate="one_to_one").reset_index(drop=True)
    assert lv.short_diff.notna().all()
    ph = lv.recorded_home_probability.to_numpy()
    yh = np.where(lv.pick_side == "HOME", lv.y, 1 - lv.y).astype(float)
    sdl = lv.short_diff.to_numpy(dtype=float)
    pl = expit(logit(np.clip(ph, EPS, 1 - EPS)) + sdl * c_full[0])
    live = contrast(pl, ph, yh, rng)
    live["counts"] = counts(lv).to_dict("index")
    live["frozen_coefficient"] = float(c_full[0])
    lnz = lv[sdl != 0].assign(after=pl[sdl != 0], home_cover=yh[sdl != 0])
    live["nonzero_games"] = (
        lnz[
            [
                "game_id",
                "weekday",
                "home_rest",
                "away_rest",
                "short_diff",
                "recorded_home_probability",
                "after",
                "home_cover",
            ]
        ]
        .round(4)
        .to_dict("records")
    )
    live_cov = np.where(lnz.short_diff == 1, lnz.home_cover == 1, lnz.home_cover == 0)
    live["raw_short_side"] = {
        "games": len(lnz),
        "short_side_covered": int(live_cov.sum()),
        "short_side_failed": int((~live_cov).sum()),
    }
    report["live_2026"] = live

    for tag in ("arm_a_offset", "arm_b_joint", "arm_c_intercept_only"):
        r = report[tag]
        print(tag, "decisive", r["decisive"])
        for k in METRICS:
            s = r[k]
            gap = r["in_sample"][k] - s["effect"]
            print(
                f"  {k} {s['effect']:+.5f} [{s['low']:+.5f},{s['high']:+.5f}] P+ {s['pp']:.3f}"
                f" | in-sample {r['in_sample'][k]:+.5f} gap {gap:+.5f}"
            )
        print("  fold coefficients", r["fold_coefficients"])
    print(
        "arm_a full coefficient",
        round(arm_a["full_coefficient"], 4),
        arm_a["coefficient_boot"],
    )
    print("arm_a per season", arm_a["per_season"])
    print("arm_b full coefficient", round(arm_b["full_coefficient"], 4))
    print("arm_b refit vs served", arm_b["refit_vs_served"])
    print("arm_b per season", arm_b["per_season"])
    print("arm_d permuted", json.dumps(arm_d))
    print("live decisive", live["decisive"], "raw", live["raw_short_side"])
    for k in METRICS:
        s = live[k]
        print(f"  live {k} {s['effect']:+.5f} [{s['low']:+.5f},{s['high']:+.5f}] P+ {s['pp']:.3f}")
    print(pd.DataFrame(live["nonzero_games"]).to_string())
    print(pd.DataFrame(live["counts"]).T.to_string())

    n = len(d)
    looks = (
        "Family mod47_short_week_side_term declared before the outcome join: 6 graded "
        "contrasts (2 arms x 3 metrics), 2 controls, in-sample versus out-of-sample per arm, "
        "1 frozen read on 2026 Weeks 1-9, 1 raw short-side cover count. Opener grade, "
        "pushes excluded, leave-one-season-out, served out-of-season probability as base."
    )
    fan = (
        "When one team has had less than a full week since its last game and its opponent "
        "has not, leaning away from the more rushed team has not yet shown a clear gain "
        "against the opening line."
    )
    da, db = arm_a["decisive"], arm_b["decisive"]
    rec_a = f"decisive {da['arm_correct']}-{da['reference_correct']} of {da['games']}"
    rec_b = f"decisive {db['arm_correct']}-{db['reference_correct']} of {db['games']}"
    cells = []
    for k in METRICS:
        cells.append(
            cell(
                f"mod47_u3_offset_short_diff_{k}",
                "Home-short minus away-short as one fitted coefficient added to the served "
                f"out-of-season logit, LOSO {FIRST_SEASON}-{LAST_SEASON}, versus served; "
                f"{rec_a}; fold coefficients {list(arm_a['fold_coefficients'].values())}.",
                k,
                arm_a[k],
                n,
                FIRST_SEASON,
                LAST_SEASON,
                "schedule",
                fan,
                looks,
            )
        )
        cells.append(
            cell(
                f"mod47_u3_joint_short_diff_{k}",
                "Home-short minus away-short added to a LOSO refit of the four served inputs "
                f"(no pool-hook stage), {FIRST_SEASON}-{LAST_SEASON}, versus the same refit "
                f"without it; {rec_b}; fold coefficients "
                f"{list(arm_b['fold_coefficients'].values())}.",
                k,
                arm_b[k],
                n,
                FIRST_SEASON,
                LAST_SEASON,
                "schedule",
                fan,
                looks,
            )
        )
    k = "log_loss_improvement"
    dc = arm_c["decisive"]
    cells.append(
        cell(
            "mod47_u3_control_intercept_only_log_loss_improvement",
            "Control: one fitted intercept added to the served out-of-season logit, LOSO "
            f"{FIRST_SEASON}-{LAST_SEASON}, versus served; decisive "
            f"{dc['arm_correct']}-{dc['reference_correct']} of {dc['games']}.",
            k,
            arm_c[k],
            n,
            FIRST_SEASON,
            LAST_SEASON,
            "control",
            "A check that adds nothing but a constant home lean to the served pick "
            "probability, to show how much a meaningless extra term moves the score.",
            looks,
        )
    )
    cells.append(
        cell(
            "mod47_u3_control_permuted_short_diff_log_loss_improvement",
            "Control: the short-week differential shuffled among games within each season "
            f"({PERMUTATIONS} draws) through the offset arm; effect is the mean and the "
            "interval the central 95 percent of shuffled runs; share of shuffles at or above "
            f"the real arm {arm_d[k]['share_at_or_above_observed']:.3f}.",
            k,
            arm_d[k],
            n,
            FIRST_SEASON,
            LAST_SEASON,
            "control",
            "A check that scrambles which games had a rushed team, to show what the "
            "short-week term scores when it carries no information.",
            looks,
        )
    )
    dl = live["decisive"]
    cells.append(
        cell(
            "mod47_u3_frozen_2026_wk1_9_log_loss_improvement",
            "Offset coefficient frozen on 2020-25 "
            f"({arm_a['full_coefficient']:+.4f}) applied to the published 2026 Weeks 1-9 "
            f"home probability; {live['raw_short_side']['games']} games with a nonzero "
            f"differential; decisive {dl['arm_correct']}-{dl['reference_correct']} of "
            f"{dl['games']}.",
            k,
            live[k],
            live["n"],
            LIVE_SEASON,
            LIVE_SEASON,
            "schedule",
            fan,
            looks,
        )
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (OUT / "weak_signals_batch.json").write_text(
        json.dumps({"cells": cells}, indent=1), encoding="utf-8"
    )
    print("cells", len(cells))


main()
