import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PICK_DIR = ROOT / "artifacts/pick_probability"
OUT = ROOT / "artifacts/eng54_served" / ("smoke" if "--smoke" in sys.argv else "")
LANE = "docs/lanes/eng-54-outcome-label-default-audit.md"
REGISTRY = ROOT / "registry/weak_signals"
SERVED_COLUMN = "out_of_season_home_probability"
FIRST_SEASON = 2020
LAST_SEASON = 2025
SMOKE = "--smoke" in sys.argv
NULL_DRAWS = 5 if SMOKE else 200
PLANT_DRAWS = 3 if SMOKE else 100
LADDER_RUNGS = 2 if SMOKE else 10
BOOT = 2000
SEED = 54
NOTABLE = (0.90, 0.95)
POWER_TARGET = 0.80
INTERVAL = (2.5, 97.5)
EPS = 1e-6
NEWTON_STEPS = 100
NEWTON_TOL = 1e-10
SHORT = ["acc", "ll", "brier"]
DIRECTION = np.array([1.0, -1.0, -1.0])


class Stop(Exception):
    pass


def run_source(name, patches=(), strip_main=False, call=None):
    path = SCRIPTS / name
    src = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    for old, new in patches:
        if old not in src:
            raise RuntimeError(f"anchor missing in {name}: {old[:50]!r}")
        src = src.replace(old, new, 1)
    if strip_main:
        body = src.rstrip()
        if not body.endswith("\nmain()"):
            raise RuntimeError(f"no trailing main() in {name}")
        src = body[: -len("main()")] + "\n"
    ns = {"__name__": "eng54_" + path.stem, "__file__": str(path), "_Stop": Stop}
    try:
        exec(compile(src, str(path), "exec"), ns)
        if call:
            return call(ns)
    except Stop as e:
        return e.args[0]
    return ns


def identity(raw, z):
    return raw


def interaction(raw, z):
    r2 = raw * z[..., None]
    return np.concatenate([np.broadcast_to(raw, r2.shape), r2], axis=-1)


def clip_logit(p):
    return logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))


def fam_mod42():
    anchor = "d = o.reset_index(drop=True)\n"
    d = run_source("mod42_ol_continuity.py", [(anchor, anchor + "raise _Stop(d)\n")])
    arms = {
        "mod42_ol_cont": (["d_cont"], "the five-starter continuity streak difference"),
        "mod42_ol_new": (["d_new"], "the new-starter count difference"),
        "mod42_ol_cont_x_pressure": (
            ["d_cont", "d_inter"],
            "continuity plus continuity times opponent pass-rush pressure",
        ),
    }
    return d[["game_id", "d_cont", "d_new", "d_inter"]], arms, True, identity


def fam_mod43():
    anchor = (
        '    arms = {\n        "intercept": [],\n        "served": ["z"],\n        "served+career"'
    )
    d = run_source(
        "mod43_qb_learning_curve.py",
        [(anchor, "    raise _Stop(d)\n" + anchor)],
        call=lambda ns: ns["main"](),
    )
    arms = {
        "mod43 career curve term on served logit opener": (
            ["g_career"],
            "the projected quarterback change from career starts",
        ),
        "mod43 team-start curve term on served logit opener": (
            ["g_team"],
            "the projected quarterback change from starts with the current team",
        ),
    }
    return d[["game_id", "g_career", "g_team"]], arms, True, identity


def fam_mod44():
    ns = run_source("mod44_coach_tenure.py", strip_main=True)
    d = ns["attach"](ns["population"](), ns["team_games"]())
    e = d[d.era == "2020-25"].copy()
    e["fy"] = e.h_first_year.astype(float) - e.a_first_year.astype(float)
    e["im"] = e.h_interim.astype(float) - e.a_interim.astype(float)
    e["tf"] = e.h_tenure_feat.fillna(0) - e.a_tenure_feat.fillna(0)
    arms = {
        "mod44_term_served+first_year_2020-25_loso": (["fy"], "the first-year head coach difference"),
        "mod44_term_served+interim_2020-25_loso": (["im"], "the interim head coach difference"),
        "mod44_term_served+both_states_2020-25_loso": (
            ["fy", "im"],
            "the first-year and interim head coach differences",
        ),
        "mod44_term_served+games_into_2020-25_loso": (["tf"], "the head coach tenure difference"),
    }
    return e[["game_id", "fy", "im", "tf"]].drop_duplicates("game_id"), arms, True, identity


def fam_mod47():
    ns = run_source("mod47_short_week_slope.py", strip_main=True)
    d, _ = ns["load"]("opener")
    d = d[d.season.between(FIRST_SEASON, LAST_SEASON)]
    arms = {
        "mod47_short_week_slope_interaction_opener": (
            ["short"],
            "a short-week indicator and its interaction with the served logit",
        ),
    }
    return d[["game_id", "short"]].astype({"short": float}), arms, True, interaction


def fam_mkt27():
    g = pd.read_csv(ROOT / "artifacts/mkt27/grade_games.csv")
    g = g[g.result != g.line].copy()
    g["game_id"] = g.game_id.astype(str)
    ps = clip_logit(g.s_cover / (g.s_cover + g.s_loss))
    for tag in ("t3", "t2"):
        g[f"d_{tag}"] = clip_logit(g[f"{tag}_cover"] / (g[f"{tag}_cover"] + g[f"{tag}_loss"])) - ps
    arms = {
        "mkt27_total_tercile_lattice_loso_logloss": (
            ["d_t3"],
            "the total-tercile-conditioned lattice shift",
        ),
        "mkt27_total_median_lattice_loso_logloss": (
            ["d_t2"],
            "the total-median-conditioned lattice shift",
        ),
    }
    return g[["game_id", "d_t3", "d_t2"]], arms, True, identity


def fam_lead76():
    g = pd.read_parquet(ROOT / "artifacts/lead76_declared/game_level.parquet")
    g = g[g.nonpush].copy()
    g["x_d"] = clip_logit(g.p_rule) - clip_logit(g.p_pool)
    arms = {
        "lead76_decl_term_vs_cal_loso": (["x_d"], "the rule-era minus pooled margin-prior shift"),
    }
    return g[["game_id", "x_d", "p_pool"]], arms, False, identity


FAMILIES = {
    "mod42_ol_continuity": fam_mod42,
    "mod43_qb_learning_curve": fam_mod43,
    "mod44_coach_tenure": fam_mod44,
    "mod47_short_week_slope": fam_mod47,
    "mkt27_total_lattice": fam_mkt27,
    "lead76_rule_prior": fam_lead76,
}


def served_games():
    pick = sorted(PICK_DIR.glob("*/per_game.parquet"))[-1]
    d = pd.read_parquet(
        pick, columns=["game_id", "season", "margin_vs_open", "home_covered", SERVED_COLUMN]
    )
    d = d[d.margin_vs_open.ne(0) & d.season.between(FIRST_SEASON, LAST_SEASON)]
    d = d.assign(game_id=d.game_id.astype(str))
    return pick, d[["game_id", "season", "home_covered", SERVED_COLUMN]]


def fit(x, y):
    b = np.zeros(x.shape[1])
    for _ in range(NEWTON_STEPS):
        p = expit(x @ b)
        grad = x.T @ (y - p)
        hess = (x * (p * (1 - p))[:, None]).T @ x
        step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        b = b + step
        if np.abs(step).max() < NEWTON_TOL:
            break
    return b


def loso(x, y, folds):
    p = np.empty(len(y))
    coefs = []
    for te in folds:
        b = fit(x[~te], y[~te])
        p[te] = expit(x[te] @ b)
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


def setup(g, cols, intercept, builder, index):
    g = g.dropna(subset=cols).reset_index(drop=True)
    y = g.home_covered.to_numpy(dtype=float)
    season = g.season.to_numpy()
    z = clip_logit(g[SERVED_COLUMN])
    raw = g[cols].to_numpy(dtype=float)
    n = len(g)
    folds = [season == s for s in np.unique(season)]
    base = np.column_stack(([np.ones(n)] if intercept else []) + [z])
    p_ref, _ = loso(base, y, folds)
    st = {
        "n": n,
        "y": y,
        "z": z,
        "season": season,
        "sign": 2 * y - 1,
        "raw": raw,
        "base": base,
        "builder": builder,
        "folds": folds,
        "groups": [np.flatnonzero(f) for f in folds],
        "p_ref": p_ref,
        "m_ref": metric_matrix(p_ref, y),
        "w": np.random.default_rng([SEED, index]).multinomial(n, np.full(n, 1 / n), size=BOOT)
        .astype(float),
    }
    terms = builder(raw, z)
    full = fit(np.column_stack([base, terms]), y)
    c = full[base.shape[1] :]
    v = terms @ c
    sd = float(v.std(ddof=1))
    if not np.isfinite(sd) or sd == 0:
        c = np.eye(terms.shape[1])[0]
        v = terms @ c
        sd = float(v.std(ddof=1))
    st.update({"c": c, "mu": float(v.mean()), "vsd": sd, "full": full})
    return st


def design(st, raw):
    return np.column_stack([st["base"], st["builder"](raw, st["z"])])


def evaluate(st, p, interval=False):
    v = (metric_matrix(p, st["y"]) - st["m_ref"]) * DIRECTION
    bs = st["w"] @ v / st["n"]
    changed = (p >= 0.5) != (st["p_ref"] >= 0.5)
    right = (p >= 0.5) == (st["y"] == 1)
    out = {
        "effect": v.mean(axis=0),
        "pp": (bs > 0).mean(axis=0) + 0.5 * (bs == 0).mean(axis=0),
        "changed": int(changed.sum()),
        "arm_right": int((changed & right).sum()),
        "ref_right": int((changed & ~right).sum()),
    }
    if interval:
        out["low"], out["high"] = np.percentile(bs, INTERVAL, axis=0)
        per = pd.Series(v[:, 1]).groupby(st["season"]).mean()
        out["seasons_positive_ll"] = int((per > 0).sum())
    return out


def permuted(st, rng):
    out = st["raw"].copy()
    for k in st["groups"]:
        out[k] = st["raw"][rng.permutation(k)]
    return out


def composite(st, raw, z):
    return (st["builder"](raw, z) @ st["c"] - st["mu"]) / st["vsd"]


def planted(st, beta, rng):
    out = np.empty_like(st["raw"])
    for k in st["groups"]:
        zk = st["z"][k]
        vm = composite(st, st["raw"][k][None, :, :], zk[:, None])
        w = expit(st["sign"][k][:, None] * (zk[:, None] + beta * vm))
        w = np.broadcast_to(w, (len(k), len(k)))
        c = np.cumsum(w, axis=1)
        u = rng.random(len(k)) * c[:, -1]
        j = np.minimum((c < u[:, None]).sum(axis=1), len(k) - 1)
        out[k] = st["raw"][k][j]
    return out


def percentile_of(null, value):
    null = np.asarray(null, dtype=float)
    return float(100 * ((null < value).mean() + 0.5 * (null == value).mean()))


def grade_arm(st, index):
    xr = design(st, st["raw"])
    p, coefs = loso(xr, st["y"], st["folds"])
    real = evaluate(st, p, interval=True)
    nb = st["base"].shape[1]
    p_in = expit(xr @ st["full"])
    ll_in = float(metric_matrix(p_in, st["y"])[:, 1].mean())
    ll_out = float(metric_matrix(p, st["y"])[:, 1].mean())
    rng = np.random.default_rng([SEED, index, 0])
    null = []
    for _ in range(NULL_DRAWS):
        pn, _ = loso(design(st, permuted(st, rng)), st["y"], st["folds"])
        e = evaluate(st, pn)
        null.append([*e["effect"], *e["pp"], e["arm_right"] - e["ref_right"]])
    null = np.array(null)
    info = float(np.sum(expit(st["z"]) * (1 - expit(st["z"]))))
    se = 1 / np.sqrt(info)
    fitted_size = st["vsd"]
    grid = sorted({*(k * se for k in range(1, LADDER_RUNGS + 1)), fitted_size})
    ladder = []
    for r, beta in enumerate(grid):
        prng = np.random.default_rng([SEED, index, 1 + r])
        det, oracle = [], []
        for _ in range(PLANT_DRAWS):
            raw_p = planted(st, beta, prng)
            pp_, _ = loso(design(st, raw_p), st["y"], st["folds"])
            e = evaluate(st, pp_)
            det.append(e["pp"])
            truth = expit(st["z"] + beta * composite(st, raw_p, st["z"]))
            oracle.append(((metric_matrix(truth, st["y"]) - st["m_ref"]) * DIRECTION).mean(axis=0))
        det, oracle = np.array(det), np.array(oracle)
        rung = {"beta_per_sd": float(beta), "is_fitted_size": bool(beta == fitted_size)}
        for i, k in enumerate(SHORT):
            rung[f"true_{k}"] = float(oracle[:, i].mean())
            for level in NOTABLE:
                rung[f"detect_{k}_{level:.2f}"] = float((det[:, i] >= level).mean())
        ladder.append(rung)
    detectable = {}
    for k in ("acc", "ll"):
        key = f"detect_{k}_0.90"
        hit = [x for x in ladder if x[key] >= POWER_TARGET]
        detectable[k] = (
            {
                "beta_per_sd": hit[0]["beta_per_sd"],
                "true_acc": hit[0]["true_acc"],
                "true_ll": hit[0]["true_ll"],
            }
            if hit
            else {"reached": False, "largest_beta": grid[-1], "share": ladder[-1][key]}
        )
    out = {
        "n": st["n"],
        "fisher_se_per_sd": se,
        "fitted_size_logit_per_sd": fitted_size,
        "fold_term_coefficients": coefs[:, nb:].round(4).tolist(),
        "full_term_coefficients": st["full"][nb:].round(4).tolist(),
        "ll_in_sample": ll_in,
        "ll_out_of_sample": ll_out,
        "ll_gap_out_minus_in": ll_out - ll_in,
        "changed": real["changed"],
        "arm_right": real["arm_right"],
        "ref_right": real["ref_right"],
        "seasons_positive_ll": real["seasons_positive_ll"],
        "record_margin_percentile": percentile_of(null[:, 6], real["arm_right"] - real["ref_right"]),
        "ladder": ladder,
        "detectable": detectable,
    }
    for i, k in enumerate(SHORT):
        out[k] = {
            "effect": float(real["effect"][i]),
            "low": float(real["low"][i]),
            "high": float(real["high"][i]),
            "pp": float(real["pp"][i]),
            "null_percentile": percentile_of(null[:, i], real["effect"][i]),
            "pp_null_percentile": percentile_of(null[:, 3 + i], real["pp"][i]),
            "null_q05_q50_q95": np.percentile(null[:, i], [5, 50, 95]).round(6).tolist(),
        }
    return out


def det_text(d):
    if "reached" in d:
        return f"not reached by {d['largest_beta']:.3f} logit per sd (share {d['share']:.2f})"
    return (
        f"{d['beta_per_sd']:.3f} logit per sd, true {d['true_acc']:+.2f} accuracy points, "
        f"{d['true_ll']:+.5f} log loss"
    )


def make_cell(family, raw_name, what, intercept, res, category, looks):
    ll, acc, br = res["ll"], res["acc"], res["brier"]
    form = "an intercept and a free slope on" if intercept else "a free slope on"
    inside = 5 < ll["null_percentile"] < 95
    return {
        "name": f"{raw_name}_served",
        "description": (
            f"Served-base re-grade of {raw_name}: {what} added to {form} the served "
            f"out-of-season logit, leave-one-season-out {FIRST_SEASON}-{LAST_SEASON} opener "
            "games, log loss versus the same fit without the term; matched null "
            f"{NULL_DRAWS} within-season shuffles puts the effect at percentile "
            f"{ll['null_percentile']:.1f} and P+ at {ll['pp_null_percentile']:.1f}."
        ),
        "source": LANE,
        "effect": round(ll["effect"], 6),
        "effect_units": "log_loss_improvement",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": FIRST_SEASON,
        "season_end": LAST_SEASON,
        "family": family,
        "category": category,
        "classification_evidence": (
            "Served base (out_of_season_home_probability); planted detectable size at "
            f"80 percent power, P+ at least 0.90: log loss {det_text(res['detectable']['ll'])}; "
            f"accuracy {det_text(res['detectable']['acc'])}. No refuted mechanism and no "
            "positive control at this size, so no closing ground."
        ),
        "notes": (
            f"scripts/eng54_served_regrade.py, artifacts/eng54_served/{family}.json. Accuracy "
            f"{acc['effect']:+.3f} pts [{acc['low']:+.3f}, {acc['high']:+.3f}] P+ {acc['pp']:.3f} "
            f"(null pct {acc['null_percentile']:.1f}); Brier {br['effect']:+.6f} "
            f"[{br['low']:+.6f}, {br['high']:+.6f}] P+ {br['pp']:.3f}. Decisive record "
            f"{res['arm_right']}-{res['ref_right']} of {res['changed']} changed sides (null pct "
            f"{res['record_margin_percentile']:.1f}). Log loss out of sample minus in sample "
            f"{res['ll_gap_out_minus_in']:+.5f}; seasons positive {res['seasons_positive_ll']}/6; "
            f"fold term coefficients {res['fold_term_coefficients']}. Raw-base cell {raw_name} "
            f"kept as history. Looks: {looks}."
        ),
        "interval_low": round(ll["low"], 6),
        "interval_high": round(ll["high"], 6),
        "probability_positive": round(ll["pp"], 4),
        "sample_games": int(res["n"]),
        "sample_blocks": int(LAST_SEASON - FIRST_SEASON + 1),
        "plain_summary": (
            f"Rechecked against the probability that actually picks each side, {what} "
            + (
                "scored inside the range a meaningless ingredient produces."
                if inside
                else "scored outside the range a meaningless ingredient produces."
            )
        ),
    }


def lead76_direct(g, family, category):
    g = g.dropna(subset=["p_pool"]).reset_index(drop=True)
    y = g.home_covered.to_numpy(dtype=float)
    ms = metric_matrix(g[SERVED_COLUMN].to_numpy(), y)
    mp = metric_matrix(g.p_pool.to_numpy(), y)
    v = (mp - ms) * DIRECTION
    w = np.random.default_rng([SEED, 999]).multinomial(len(y), np.full(len(y), 1 / len(y)), BOOT)
    bs = w @ v / len(y)
    lo, hi = np.percentile(bs[:, 1], INTERVAL)
    pp = float((bs[:, 1] > 0).mean())
    ch = (g.p_pool.to_numpy() >= 0.5) != (g[SERVED_COLUMN].to_numpy() >= 0.5)
    rt = (g.p_pool.to_numpy() >= 0.5) == (y == 1)
    return {
        "name": "lead76_decl_pool_vs_served_cover_served",
        "description": (
            "Served-base re-grade of lead76_decl_pool_vs_served_cover: discrete pooled-prior "
            "read at the model point versus the served out-of-season probability, unfitted, "
            f"{FIRST_SEASON}-{LAST_SEASON} non-push opener games, log loss."
        ),
        "source": LANE,
        "effect": round(float(v[:, 1].mean()), 6),
        "effect_units": "log_loss_improvement",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": FIRST_SEASON,
        "season_end": LAST_SEASON,
        "family": family,
        "category": category,
        "classification_evidence": (
            "Two-forecast comparison against the served base, game bootstrap; no fitted term, "
            "no refuted mechanism, no positive control; no closing ground."
        ),
        "notes": (
            f"Accuracy {v[:, 0].mean():+.3f} pts P+ {(bs[:, 0] > 0).mean():.3f}; decisive record "
            f"{int((ch & rt).sum())}-{int((ch & ~rt).sum())} of {int(ch.sum())}. Raw-base cell "
            "lead76_decl_pool_vs_served_cover kept as history."
        ),
        "interval_low": round(float(lo), 6),
        "interval_high": round(float(hi), 6),
        "probability_positive": round(pp, 4),
        "sample_games": len(y),
        "sample_blocks": int(LAST_SEASON - FIRST_SEASON + 1),
        "plain_summary": (
            "Rechecked against the probability that actually picks each side, the pooled "
            "margin-history read alone."
        ),
    }


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pick, served = served_games()
    names = [a for a in sys.argv[1:] if a != "--smoke"] or list(FAMILIES)
    for index, family in enumerate(names):
        try:
            frame, arms, intercept, builder = FAMILIES[family]()
        except Exception as exc:
            print("FAILED", family, repr(exc), flush=True)
            continue
        frame = frame.assign(game_id=frame.game_id.astype(str))
        dup = int(frame.game_id.duplicated().sum())
        frame = frame.drop_duplicates("game_id")
        print(family, "frame rows", len(frame), "duplicates dropped", dup, flush=True)
        g = served.merge(frame, on="game_id", how="inner", validate="one_to_one")
        shard = json.loads((REGISTRY / f"{family}.json").read_text(encoding="utf-8"))["signals"]
        looks = f"{len(arms)} arms x 3 metrics in {family} served re-grade"
        summary = {"base_artifact": str(pick.relative_to(ROOT)), "games": len(g), "arms": {}}
        cells = []
        for a, (raw_name, (cols, what)) in enumerate(arms.items()):
            st = setup(g, cols, intercept, builder, 100 * index + a)
            res = grade_arm(st, 100 * index + a)
            summary["arms"][raw_name] = res
            cells.append(
                make_cell(family, raw_name, what, intercept, res, shard[raw_name]["category"], looks)
            )
            ll, acc = res["ll"], res["acc"]
            print(
                f"{family} | {raw_name} n={res['n']} ll {ll['effect']:+.5f} "
                f"[{ll['low']:+.5f},{ll['high']:+.5f}] P+ {ll['pp']:.3f} nullpct "
                f"{ll['null_percentile']:.1f} | acc {acc['effect']:+.3f} "
                f"[{acc['low']:+.3f},{acc['high']:+.3f}] P+ {acc['pp']:.3f} nullpct "
                f"{acc['null_percentile']:.1f} | rec {res['arm_right']}-{res['ref_right']} | "
                f"det ll {det_text(res['detectable']['ll'])} | det acc "
                f"{det_text(res['detectable']['acc'])} | {time.time() - started:.0f}s",
                flush=True,
            )
        if family == "lead76_rule_prior":
            cells.append(
                lead76_direct(g, family, shard["lead76_decl_pool_vs_served_cover"]["category"])
            )
            c = cells[-1]
            print(family, "| direct", c["effect"], c["interval_low"], c["interval_high"],
                  c["probability_positive"], c["notes"], flush=True)
        (OUT / f"{family}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        (OUT / f"{family}_batch.json").write_text(
            json.dumps({"cells": cells}, indent=1), encoding="utf-8"
        )
    print("done", f"{time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
