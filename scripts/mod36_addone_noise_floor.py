import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import kurtosis, skew

ROOT = Path(__file__).resolve().parents[1]
PICK_DIR = ROOT / "artifacts/pick_probability"
FEATURES = ROOT / "data/processed/game_features.parquet"
HABIT = ROOT / "artifacts/per16/unit4_games.parquet"
SWING = ROOT / "artifacts/sim09/stakes_teamweek.csv"
OUT = ROOT / "artifacts/mod36"
LANE = "docs/lanes/mod-36-mod36-add-one-noise-null.md"
FAMILY = "mod36_addone_noise_floor"
SERVED_COLUMN = "out_of_season_home_probability"
FIRST_SEASON = 2020
LAST_SEASON = 2025
NATURAL_WEEK_DAYS = 7
LEVERAGE_WEEKS = (10, 16)
ALIAS = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
DECLARED_DRAWS = 200
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else DECLARED_DRAWS
OUT = OUT if DRAWS == DECLARED_DRAWS else OUT / "smoke"
BOOT = 2000
SEED = 36
BOOT_STREAM = 900
LADDER_RUNGS = 10
NOTABLE = (0.90, 0.95)
POWER_TARGET = 0.80
BAND = (5.0, 50.0, 95.0)
INTERVAL = (2.5, 97.5)
EPS = 1e-6
NEWTON_STEPS = 100
NEWTON_TOL = 1e-10
METRICS = ["accuracy_points", "log_loss_improvement", "brier_improvement"]
SHORT = ["acc", "ll", "brier"]
DIRECTION = np.array([1.0, -1.0, -1.0])
TEMPLATES = [
    {
        "name": "short_week_side",
        "column": "short_diff",
        "intercept": False,
        "matched": "served",
        "reported": True,
        "what": "which games had one team on a short week and the other not",
    },
    {
        "name": "rest_difference",
        "column": "rest_diff",
        "intercept": False,
        "matched": "served",
        "reported": False,
        "what": "the gap in rest days between the two teams",
    },
    {
        "name": "questionable_habit",
        "column": "habit_adj",
        "intercept": True,
        "matched": "served",
        "reported": True,
        "what": "each game's reading of how often a coach's doubtful-to-play listings suit up",
    },
    {
        "name": "playoff_leverage",
        "column": "levdiff",
        "intercept": True,
        "matched": "intercept_only",
        "reported": True,
        "what": "the gap in playoff stakes between the two teams in late-season games",
    },
]


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
    wide = long.pivot(index="game_id", columns="side", values="short").reset_index()
    wide["short_diff"] = wide["home"] - wide["away"]
    return wide[["game_id", "short_diff"]]


def leverage_table():
    f = pd.read_parquet(
        FEATURES,
        columns=["game_id", "season", "week", "home_team", "away_team", "game_type"],
    )
    f = f[(f.game_type == "REG") & f.week.between(*LEVERAGE_WEEKS) & (f.season <= LAST_SEASON)]
    f = f.copy()
    for c in ("home_team", "away_team"):
        f[c] = f[c].replace(ALIAS)
    t = pd.read_csv(SWING)[["season", "week", "team", "lev"]]
    for tag, key in (("h", "home_team"), ("a", "away_team")):
        f = f.merge(
            t.rename(columns={"team": key, "lev": f"lev_{tag}"}),
            on=["season", "week", key],
            how="inner",
        )
    f["levdiff"] = f.lev_h - f.lev_a
    return f[["game_id", "levdiff"]]


def games():
    pick = sorted(PICK_DIR.glob("*/per_game.parquet"))[-1]
    d = pd.read_parquet(pick)
    d = d[d.margin_vs_open.ne(0) & d.season.between(FIRST_SEASON, LAST_SEASON)]
    d = d[["game_id", "season", "week", "home_covered", SERVED_COLUMN]]
    extras = [
        short_table(),
        pd.read_parquet(FEATURES, columns=["game_id", "rest_diff"]),
        pd.read_parquet(HABIT, columns=["game_id", "habit_adj"]),
        leverage_table(),
    ]
    for e in extras:
        d = d.merge(e, on="game_id", how="left", validate="one_to_one")
    return pick, d.reset_index(drop=True)


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


def loso(x, y, offset, folds):
    p = np.empty(len(y))
    coefs = []
    for te in folds:
        b = fit_offset(x[~te], y[~te], offset[~te])
        p[te] = expit(offset[te] + x[te] @ b)
        coefs.append(b)
    return p, np.array(coefs)


def metric_matrix(p, y):
    pc = np.clip(p, EPS, 1 - EPS)
    return np.column_stack(
        [
            ((p >= 0.5) == (y == 1)).astype(float) * 100,
            -(y * np.log(pc) + (1 - y) * np.log(1 - pc)),
            (p - y) ** 2,
        ]
    )


def design(x, intercept):
    if intercept:
        return np.column_stack([np.ones(len(x)), x])
    return x[:, None]


def setup(d, tpl, index):
    g = d[d[tpl["column"]].notna()].reset_index(drop=True)
    y = g.home_covered.to_numpy(dtype=float)
    season = g.season.to_numpy()
    z = logit(np.clip(g[SERVED_COLUMN].to_numpy(), EPS, 1 - EPS))
    raw = g[tpl["column"]].to_numpy(dtype=float)
    sd = float(raw.std(ddof=1))
    n = len(g)
    folds = [season == s for s in np.unique(season)]
    served = expit(z)
    refs = {"served": {"p": served, "m": metric_matrix(served, y)}}
    if tpl["intercept"]:
        p0, _ = loso(np.ones((n, 1)), y, z, folds)
        refs["intercept_only"] = {"p": p0, "m": metric_matrix(p0, y)}
    boot = np.random.default_rng([SEED, BOOT_STREAM, index])
    return {
        "n": n,
        "y": y,
        "z": z,
        "sign": 2 * y - 1,
        "xs": raw / sd,
        "sd": sd,
        "folds": folds,
        "groups": [np.flatnonzero(f) for f in folds],
        "refs": refs,
        "w": boot.multinomial(n, np.full(n, 1 / n), size=BOOT).astype(float),
        "shape": {
            "n": n,
            "sd": sd,
            "nonzero_share": float((raw != 0).mean()),
            "distinct_values": len(np.unique(raw)),
            "skew": float(skew(raw)),
            "excess_kurtosis": float(kurtosis(raw)),
            "served_accuracy": float(refs["served"]["m"][:, 0].mean()),
        },
    }


def evaluate(st, p_new, ref, interval=False):
    r = st["refs"][ref]
    v = (metric_matrix(p_new, st["y"]) - r["m"]) * DIRECTION
    bs = st["w"] @ v / st["n"]
    changed = (p_new >= 0.5) != (r["p"] >= 0.5)
    right = (p_new >= 0.5) == (st["y"] == 1)
    out = {
        "effect": v.mean(axis=0),
        "pp": (bs > 0).mean(axis=0) + 0.5 * (bs == 0).mean(axis=0),
        "changed": int(changed.sum()),
        "arm_right": int((changed & right).sum()),
        "ref_right": int((changed & ~right).sum()),
    }
    if interval:
        out["low"], out["high"] = np.percentile(bs, INTERVAL, axis=0)
    return out


def permuted(st, rng):
    out = st["xs"].copy()
    for k in st["groups"]:
        out[k] = st["xs"][rng.permutation(k)]
    return out


def planted(st, beta, rng):
    out = np.empty(st["n"])
    for k in st["groups"]:
        v = st["xs"][k]
        w = expit(st["sign"][k][:, None] * (st["z"][k][:, None] + beta * v[None, :]))
        c = np.cumsum(w, axis=1)
        u = rng.random(len(k)) * c[:, -1]
        j = np.minimum((c < u[:, None]).sum(axis=1), len(k) - 1)
        out[k] = v[j]
    return out


def draw_rows(st, tpl, kind, beta, draw, x, rows):
    xd = design(x, tpl["intercept"])
    p, _ = loso(xd, st["y"], st["z"], st["folds"])
    coef = float(fit_offset(xd, st["y"], st["z"])[-1])
    oracle = np.full(3, np.nan)
    if kind == "planted":
        m = metric_matrix(expit(st["z"] + beta * x), st["y"])
        oracle = ((m - st["refs"]["served"]["m"]) * DIRECTION).mean(axis=0)
    for ref in st["refs"]:
        e = evaluate(st, p, ref)
        row = {
            "template": tpl["name"],
            "reference": ref,
            "kind": kind,
            "beta": beta,
            "draw": draw,
            "coef": coef,
            "x_sd": float(x.std(ddof=1)),
            "changed": e["changed"],
            "arm_right": e["arm_right"],
            "ref_right": e["ref_right"],
        }
        for i, k in enumerate(SHORT):
            row[k] = float(e["effect"][i])
            row[f"pp_{k}"] = float(e["pp"][i])
            row[f"oracle_{k}"] = float(oracle[i])
        rows.append(row)


def real_read(st, tpl):
    xd = design(st["xs"], tpl["intercept"])
    p, coefs = loso(xd, st["y"], st["z"], st["folds"])
    full = fit_offset(xd, st["y"], st["z"])
    out = {
        "full_coefficient_per_sd": float(full[-1]),
        "fold_coefficients_per_sd": [float(v) for v in coefs[:, -1]],
    }
    for ref in st["refs"]:
        e = evaluate(st, p, ref, interval=True)
        out[ref] = {
            "changed": e["changed"],
            "arm_right": e["arm_right"],
            "ref_right": e["ref_right"],
        }
        for i, k in enumerate(SHORT):
            out[ref][k] = {
                "effect": float(e["effect"][i]),
                "low": float(e["low"][i]),
                "high": float(e["high"][i]),
                "pp": float(e["pp"][i]),
            }
    return out


def band(v):
    q = np.percentile(v, BAND)
    return {
        "mean": float(np.mean(v)),
        "q05": float(q[0]),
        "q50": float(q[1]),
        "q95": float(q[2]),
    }


def percentile_of(null, value):
    null = np.asarray(null, dtype=float)
    return float(100 * ((null < value).mean() + 0.5 * (null == value).mean()))


def null_summary(g):
    out = {k: band(g[k]) for k in SHORT}
    out["changed"] = band(g.changed)
    out["arm_right"] = band(g.arm_right)
    out["ref_right"] = band(g.ref_right)
    margin = g.arm_right - g.ref_right
    out["record"] = {
        "losing_share": float((margin < 0).mean()),
        "level_share": float((margin == 0).mean()),
        "winning_share": float((margin > 0).mean()),
    }
    out["coef"] = band(g.coef) | {"sd": float(g.coef.std(ddof=1))}
    for k in SHORT:
        pp = g[f"pp_{k}"]
        out[f"pp_{k}"] = band(pp)
        for level in NOTABLE:
            out[f"pp_{k}"][f"share_at_or_above_{level:.2f}"] = float((pp >= level).mean())
            out[f"pp_{k}"][f"share_at_or_below_{1 - level:.2f}"] = float((pp <= 1 - level).mean())
    return out


def planted_summary(g):
    out = {
        "beta": float(g.beta.iloc[0]),
        "x_sd_ratio": float(g.x_sd.mean()),
        "changed_median": float(g.changed.median()),
    }
    for k in SHORT:
        out[f"oracle_{k}"] = band(g[f"oracle_{k}"])
        out[f"fitted_{k}"] = band(g[k])
        for level in NOTABLE:
            out[f"detect_{k}_{level:.2f}"] = float((g[f"pp_{k}"] >= level).mean())
    return out


def smallest_detected(cells):
    out = {}
    for k in SHORT:
        for level in NOTABLE:
            key = f"detect_{k}_{level:.2f}"
            hit = [c for c in cells if c["beta"] > 0 and c[key] >= POWER_TARGET]
            if not hit:
                out[key] = {
                    "reached": False,
                    "largest_beta_tried": cells[-1]["beta"],
                    "share_at_largest": cells[-1][key],
                }
                continue
            c = hit[0]
            out[key] = {
                "reached": True,
                "beta_per_sd": c["beta"],
                "share": c[key],
                "true_accuracy_points": c["oracle_acc"]["mean"],
                "true_log_loss_improvement": c["oracle_ll"]["mean"],
                "true_brier_improvement": c["oracle_brier"]["mean"],
                "holds_at_every_larger_size": all(
                    x[key] >= POWER_TARGET for x in cells if x["beta"] >= c["beta"]
                ),
            }
    return out


def null_cells(tpl, st, s):
    form = "an intercept and one coefficient" if tpl["intercept"] else "one coefficient"
    versus = "the served probability" if tpl["matched"] == "served" else "an intercept-only fit"
    cells = []
    for k, units in zip(SHORT, METRICS, strict=True):
        cells.append(
            {
                "name": f"mod36_null_{tpl['name']}_{units}",
                "description": (
                    f"Matched-marginal null: the real {tpl['column']} column shuffled among "
                    f"games within each season, {DRAWS} draws (seed {SEED}), entered as {form} "
                    f"on the served out-of-season logit, leave-one-season-out "
                    f"{FIRST_SEASON}-{LAST_SEASON}, versus {versus}; effect is the mean of "
                    "draws, interval the 5th to 95th percentile, probability_positive the "
                    "share of draws above zero; median changed picks "
                    f"{s['changed']['q50']:.0f}, median record "
                    f"{s['arm_right']['q50']:.0f}-{s['ref_right']['q50']:.0f}."
                ),
                "source": LANE,
                "effect": round(s[k]["mean"], 6),
                "effect_units": units,
                "classification": "unresolved_below_power",
                "league": "nfl",
                "season_start": FIRST_SEASON,
                "season_end": LAST_SEASON,
                "family": FAMILY,
                "category": "control",
                "classification_evidence": (
                    "Negative-control null distribution of a term with no game link; it is "
                    "an instrument reading, closes nothing and is no closing ground."
                ),
                "notes": (
                    f"Family {FAMILY}: {len(TEMPLATES)} templates x {len(METRICS)} metrics "
                    "at each template's matched reference; planted-effect cells and "
                    "detection shares stay in the lane."
                ),
                "interval_low": round(s[k]["q05"], 6),
                "interval_high": round(s[k]["q95"], 6),
                "probability_positive": round(
                    float(s[k]["share_positive"]),
                    4,
                ),
                "sample_games": int(st["n"]),
                "sample_blocks": int(LAST_SEASON - FIRST_SEASON + 1),
                "plain_summary": (
                    f"A check that scrambles {tpl['what']}, to show how much a meaningless "
                    "extra ingredient moves the pick probability's score."
                ),
            }
        )
    return cells


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pick, d = games()
    print("base", pick.relative_to(ROOT), "games", len(d), flush=True)
    states, reals, rows = [], [], []
    summary = {
        "family": FAMILY,
        "base_artifact": str(pick.relative_to(ROOT)),
        "base_column": SERVED_COLUMN,
        "draws": DRAWS,
        "boot": BOOT,
        "seed": SEED,
        "notable_levels": list(NOTABLE),
        "power_target": POWER_TARGET,
        "templates": {},
    }
    for t, tpl in enumerate(TEMPLATES):
        st = setup(d, tpl, t)
        real = real_read(st, tpl)
        states.append(st)
        reals.append(real)
        rng = np.random.default_rng([SEED, t, 0])
        for i in range(DRAWS):
            draw_rows(st, tpl, "null", 0.0, i, permuted(st, rng), rows)
        print(tpl["name"], st["shape"], f"{time.time() - started:.0f}s", flush=True)
    null = pd.DataFrame(rows)
    reported = {}
    for tpl, real in zip(TEMPLATES, reals, strict=True):
        if tpl["reported"]:
            reported[f"{tpl['name']}_full"] = abs(real["full_coefficient_per_sd"])
            reported[f"{tpl['name']}_fold_max"] = max(
                abs(v) for v in real["fold_coefficients_per_sd"]
            )
    summary["reported_sizes_logit_per_sd"] = reported
    print("reported sizes", {k: round(v, 4) for k, v in reported.items()}, flush=True)
    planted_rows = []
    for t, (tpl, st) in enumerate(zip(TEMPLATES, states, strict=True)):
        own = null[(null.template == tpl["name"]) & (null.reference == "served")]
        se = float(own.coef.std(ddof=1))
        ladder = [k * se for k in range(1, LADDER_RUNGS + 1)]
        grid = sorted({0.0, *reported.values(), *ladder})
        for c, beta in enumerate(grid):
            rng = np.random.default_rng([SEED, t, 1 + c])
            for i in range(DRAWS):
                draw_rows(st, tpl, "planted", beta, i, planted(st, beta, rng), planted_rows)
        summary["templates"][tpl["name"]] = {
            "column": tpl["column"],
            "intercept": tpl["intercept"],
            "matched_reference": tpl["matched"],
            "shape": st["shape"],
            "null_coefficient_sd": se,
            "grid": grid,
        }
        print(tpl["name"], "grid", len(grid), f"{time.time() - started:.0f}s", flush=True)
    plant = pd.DataFrame(planted_rows)
    pd.concat([null, plant]).to_parquet(OUT / "draws.parquet", index=False)

    cells = []
    for tpl, st, real in zip(TEMPLATES, states, reals, strict=True):
        block = summary["templates"][tpl["name"]]
        block["real_term"] = real
        block["null"], block["planted"], block["detectable"], block["read"] = {}, {}, {}, {}
        for ref in st["refs"]:
            g = null[(null.template == tpl["name"]) & (null.reference == ref)]
            s = null_summary(g)
            for k in SHORT:
                s[k]["share_positive"] = float((g[k] > 0).mean() + 0.5 * (g[k] == 0).mean())
            block["null"][ref] = s
            pg = plant[(plant.template == tpl["name"]) & (plant.reference == ref)]
            pcells = [planted_summary(x) for _, x in pg.groupby("beta", sort=True)]
            block["planted"][ref] = pcells
            block["detectable"][ref] = smallest_detected(pcells)
            r = real[ref]
            read = {
                "record_margin_percentile": percentile_of(
                    g.arm_right - g.ref_right, r["arm_right"] - r["ref_right"]
                )
            }
            for k in SHORT:
                read[f"{k}_percentile"] = percentile_of(g[k], r[k]["effect"])
                read[f"pp_{k}_percentile"] = percentile_of(g[f"pp_{k}"], r[k]["pp"])
            read["coefficient_percentile"] = percentile_of(g.coef, real["full_coefficient_per_sd"])
            block["read"][ref] = read
            tag = f"{tpl['name']} vs {ref}"
            print(f"NULL {tag} n={st['n']} coef sd {s['coef']['sd']:.4f}")
            for k in SHORT:
                b, pp = s[k], s[f"pp_{k}"]
                print(
                    f"  {k} mean {b['mean']:+.5f} q05/50/95 {b['q05']:+.5f}/{b['q50']:+.5f}/"
                    f"{b['q95']:+.5f} | P+ q05/50/95 {pp['q05']:.3f}/{pp['q50']:.3f}/"
                    f"{pp['q95']:.3f} >=.90 {pp['share_at_or_above_0.90']:.3f} >=.95 "
                    f"{pp['share_at_or_above_0.95']:.3f} <=.10 {pp['share_at_or_below_0.10']:.3f}"
                    f" <=.05 {pp['share_at_or_below_0.05']:.3f}"
                )
            ch, ar, rr, rec = s["changed"], s["arm_right"], s["ref_right"], s["record"]
            print(
                f"  changed q05/50/95 {ch['q05']:.0f}/{ch['q50']:.0f}/{ch['q95']:.0f} median "
                f"record {ar['q50']:.0f}-{rr['q50']:.0f} losing {rec['losing_share']:.2f} "
                f"level {rec['level_share']:.2f} winning {rec['winning_share']:.2f}"
            )
            print(f"PLANTED {tag}")
            for c in pcells:
                print(
                    f"  beta {c['beta']:.4f} true acc {c['oracle_acc']['mean']:+.3f} ll "
                    f"{c['oracle_ll']['mean']:+.5f} | fitted acc {c['fitted_acc']['mean']:+.3f}"
                    f" ll {c['fitted_ll']['mean']:+.5f} | det acc "
                    f"{c['detect_acc_0.90']:.2f}/{c['detect_acc_0.95']:.2f} ll "
                    f"{c['detect_ll_0.90']:.2f}/{c['detect_ll_0.95']:.2f} brier "
                    f"{c['detect_brier_0.90']:.2f}/{c['detect_brier_0.95']:.2f} changed "
                    f"{c['changed_median']:.0f} sd ratio {c['x_sd_ratio']:.3f}"
                )
            for key, v in block["detectable"][ref].items():
                if v["reached"]:
                    print(
                        f"  DETECT {key}: beta {v['beta_per_sd']:.4f} share {v['share']:.2f} "
                        f"true acc {v['true_accuracy_points']:+.3f} ll "
                        f"{v['true_log_loss_improvement']:+.5f} monotone "
                        f"{v['holds_at_every_larger_size']}"
                    )
                else:
                    print(
                        f"  DETECT {key}: not reached; largest beta "
                        f"{v['largest_beta_tried']:.4f} share {v['share_at_largest']:.2f}"
                    )
            print(
                f"REAL {tag} coef/sd {real['full_coefficient_per_sd']:+.4f} folds "
                f"{[round(v, 3) for v in real['fold_coefficients_per_sd']]} record "
                f"{r['arm_right']}-{r['ref_right']} of {r['changed']}"
            )
            for k in SHORT:
                e = r[k]
                print(
                    f"  {k} {e['effect']:+.5f} [{e['low']:+.5f},{e['high']:+.5f}] P+ "
                    f"{e['pp']:.3f} | null percentile effect {read[f'{k}_percentile']:.1f} "
                    f"P+ {read[f'pp_{k}_percentile']:.1f}"
                )
            print(
                f"  record margin percentile {read['record_margin_percentile']:.1f} "
                f"coefficient percentile {read['coefficient_percentile']:.1f}"
            )
            if ref == tpl["matched"]:
                cells.extend(null_cells(tpl, st, s))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUT / "weak_signals_batch.json").write_text(
        json.dumps({"cells": cells}, indent=1), encoding="utf-8"
    )
    print("cells", len(cells), f"elapsed {time.time() - started:.0f}s", flush=True)


main()
