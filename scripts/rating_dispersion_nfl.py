import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from eng62_tuesday_reference_oos import stage_one
from xlg14_fumble_luck_nfl_confirmation import coefficient_draws, load_mod36

from nfl_ats.pick_probability import active_market_move_feature_version
from nfl_ats.pick_probability_fit import build_fit_population

ART = ROOT / "artifacts"
OUT = ART / "rating_dispersion_nfl"
PT = ART / "rating_dispersion" / "nfl_dispersion.parquet"
LANE = "docs/lanes/rating-dispersion-2026-11-16.md"
POWER_LANE = "docs/lanes/xlg-harness-power-2026-11-14.md"
FAMILY = "rating_dispersion_nfl_u1"
STREAM = 42
NAME = "median_open_gap"
MOVE = "close_move"
PUSH_LABEL = 0.5
HALF_SPLIT = 2023
SERVED_MOVE_SEASONS = (2023, 2025)


def population(m36):
    version = active_market_move_feature_version(ART)
    pop, _ = build_fit_population(
        ART, ROOT / "data", market_move_feature_version=version, keep_pushes=True
    )
    pop = pop.reset_index(drop=True)
    outside, _ = stage_one(pop)
    pop["z"] = outside
    pop = pop[pop["season"].between(m36.FIRST_SEASON, m36.LAST_SEASON)].copy()
    pick = sorted((ART / "pick_probability").glob("*/per_game.parquet"))[-1]
    art = pd.read_parquet(pick, columns=["game_id", m36.SERVED_COLUMN])
    chk = pop[["game_id", "z"]].merge(art, on="game_id")
    gap = float((expit(chk["z"]) - chk[m36.SERVED_COLUMN]).abs().max())
    push = pop["margin_vs_open"].eq(0.0)
    pop["y"] = np.where(push, PUSH_LABEL, pop["home_covered"].astype(float))
    pop["push"] = push
    pt = pd.read_parquet(PT, columns=["game_id", "system_median", "systems"])
    before = len(pop)
    pop = pop.merge(pt, on="game_id", how="inner", validate="one_to_one")
    pop = pop[pop["system_median"].notna()]
    pop[NAME] = pop["system_median"] - pop["tue_open_home_spread"]
    pop[MOVE] = pop["close_home_spread"] - pop["tue_open_home_spread"]
    pop = pop.sort_values(["season", "game_id"]).reset_index(drop=True)
    diag = {
        "per_game_artifact": str(pick.relative_to(ROOT)),
        "stage_gap_max_abs_probability": gap,
        "games_before_pt_join": before,
        "games": len(pop),
        "pushes": int(pop["push"].sum()),
        "by_season": {int(k): int(v) for k, v in pop.groupby("season").size().items()},
        "median_systems": float(pop["systems"].median()),
        "corr_median_vs_open": float(np.corrcoef(pop["system_median"], pop["tue_open_home_spread"])[0, 1]),
    }
    return pop, diag


def state(m36, d, index):
    frame = pd.DataFrame(
        {
            "game_id": d["game_id"].to_numpy(),
            "season": d["season"].to_numpy(int),
            "home_covered": d["y"].to_numpy(float),
            m36.SERVED_COLUMN: expit(d["z"].to_numpy(float)),
            NAME: d[NAME].to_numpy(float),
        }
    )
    tpl = {"name": NAME, "column": NAME, "intercept": True}
    st = m36.setup(frame, tpl, index)
    move = d[MOVE].to_numpy(float)
    st["move"] = move / move.std(ddof=1)
    mp, mcoef = m36.loso(st["move"][:, None], st["y"], st["z"], st["folds"])
    st["refs"]["move_only"] = {"p": mp, "m": m36.metric_matrix(mp, st["y"])}
    st["move_fold_coefficients"] = [float(c) for c in mcoef[:, 0]]
    st["push"] = d["push"].to_numpy(bool)
    st["season"] = d["season"].to_numpy(int)
    return st


def predict(m36, st, x, with_move):
    xd = np.column_stack([x, st["move"]]) if with_move else x[:, None]
    p, coefs = m36.loso(xd, st["y"], st["z"], st["folds"])
    full = m36.fit_offset(xd, st["y"], st["z"])
    return p, coefs[:, 0], float(full[0]), xd


def half_coefs(m36, st):
    out = {}
    for name, mask in (("earlier", st["season"] < HALF_SPLIT), ("later", st["season"] >= HALF_SPLIT)):
        if mask.any():
            out[name] = float(m36.fit_offset(st["xs"][mask][:, None], st["y"][mask], st["z"][mask])[0])
    return out


def read(m36, st, with_move, ref, index):
    p, folds, coef, xd = predict(m36, st, st["xs"], with_move)
    e = m36.evaluate(st, p, ref, interval=True)
    inside = ((m36.metric_matrix(expit(st["z"] + xd @ m36.fit_offset(xd, st["y"], st["z"])), st["y"]) - st["refs"][ref]["m"]) * m36.DIRECTION).mean(axis=0)
    rng = np.random.default_rng([m36.SEED, STREAM, index])
    null = []
    for _ in range(m36.DRAWS):
        xp = m36.permuted(st, rng)
        pn, _, cn, _ = predict(m36, st, xp, with_move)
        en = m36.evaluate(st, pn, ref)
        null.append([*en["effect"], cn, en["changed"]])
    null = np.array(null)
    held = {"changed": e["changed"], "arm_right": e["arm_right"], "ref_right": e["ref_right"]}
    ref_p = st["refs"][ref]["p"]
    ch = (p >= 0.5) != (ref_p >= 0.5)
    held["changed_pushes"] = int((ch & st["push"]).sum())
    for i, k in enumerate(m36.SHORT):
        held[k] = {
            "effect": float(e["effect"][i]),
            "low": float(e["low"][i]),
            "high": float(e["high"][i]),
            "pp": float(e["pp"][i]),
        }
    seasons = sorted(int(s) for s in np.unique(st["season"]))
    res = {
        "games": st["n"],
        "seasons": seasons,
        "reference": ref,
        "coefficient_per_sd": coef,
        "fold_coefficients_per_sd": dict(zip((str(s) for s in seasons), (float(c) for c in folds), strict=True)),
        "folds_with_expected_sign": int(sum(c > 0 for c in folds)),
        "held_out": held,
        "in_sample_minus_held_out": {k: float(inside[i] - held[k]["effect"]) for i, k in enumerate(m36.SHORT)},
        "null_percentile": {k: m36.percentile_of(null[:, i], held[k]["effect"]) for i, k in enumerate(m36.SHORT)},
        "null_ll_band": m36.band(null[:, 1]),
        "coefficient_null_percentile": m36.percentile_of(null[:, 3], coef),
        "null_changed_median": float(np.median(null[:, 4])),
    }
    if not with_move:
        draws = coefficient_draws(m36, st)
        lo, hi = (float(v) for v in np.percentile(draws, m36.INTERVAL))
        res["coefficient_interval_per_sd"] = [lo, hi]
        res["coefficient_probability_positive"] = float((draws > 0).mean())
        res["wrong_sign_resolved"] = bool(hi < 0)
        res["halves"] = half_coefs(m36, st)
        h = list(res["halves"].values())
        res["half_signs_agree"] = bool(len(h) == 2 and h[0] * h[1] > 0)
    bar = 100 * m36.NOTABLE[1]
    res["survivor"] = bool(res["null_percentile"]["ll"] > bar and coef > 0 and res.get("half_signs_agree", True))
    return res


def controls(m36, st):
    out = {}
    const = np.empty(st["n"])
    for te in st["folds"]:
        const[te] = st["y"][~te].mean()
    p0, c0 = m36.loso(np.ones((st["n"], 1)), st["y"], st["z"], st["folds"])
    for name, p in (
        ("served", st["refs"]["served"]["p"]),
        ("constant", const),
        ("served_plus_intercept", p0),
        ("served_plus_close_move", st["refs"]["move_only"]["p"]),
    ):
        m = m36.metric_matrix(p, st["y"])
        out[name] = {"accuracy": float(m[:, 0].mean()), "log_loss": float(m[:, 1].mean())}
    out["intercept_fold"] = [float(c) for c in c0[:, 0]]
    out["close_move_fold_coefficients_per_sd"] = st["move_fold_coefficients"]
    return out


def line(tag, r):
    h = r["held_out"]
    s = (
        f"{tag}: games {r['games']} ref {r['reference']} changed {h['changed']} (pushes "
        f"{h['changed_pushes']}) record {h['arm_right']}-{h['ref_right']}; ll {h['ll']['effect']:+.6f} "
        f"[{h['ll']['low']:+.6f},{h['ll']['high']:+.6f}] pp {h['ll']['pp']:.3f} pct "
        f"{r['null_percentile']['ll']:.1f}; acc {h['acc']['effect']:+.3f} pp {h['acc']['pp']:.3f} pct "
        f"{r['null_percentile']['acc']:.1f}; brier {h['brier']['effect']:+.6f} pct "
        f"{r['null_percentile']['brier']:.1f}; in-sample ll gap {r['in_sample_minus_held_out']['ll']:+.6f}; "
        f"coef {r['coefficient_per_sd']:+.4f} (null pct {r['coefficient_null_percentile']:.1f}) folds "
        + " ".join(f"{k}:{v:+.3f}" for k, v in r["fold_coefficients_per_sd"].items())
    )
    if "coefficient_interval_per_sd" in r:
        lo, hi = r["coefficient_interval_per_sd"]
        s += f"; coef CI [{lo:+.4f},{hi:+.4f}] P+ {r['coefficient_probability_positive']:.3f}; halves {r['halves']}"
    return s + f"; survivor {r['survivor']}"


def cells(m36, tag, r, desc, plain):
    h = r["held_out"]
    out = []
    for k, units in zip(m36.SHORT, m36.METRICS, strict=True):
        out.append(
            {
                "name": f"rating_dispersion_nfl_{NAME}_{tag}_{units}",
                "description": (
                    f"NFL confirmation of the college survivor {NAME} (Prediction Tracker system median "
                    f"minus the Tuesday opener, home-margin units): {desc} Changed picks {h['changed']}, "
                    f"record {h['arm_right']}-{h['ref_right']}; coefficient per SD "
                    f"{r['coefficient_per_sd']:+.4f}, positive in {r['folds_with_expected_sign']} of "
                    f"{len(r['seasons'])} folds; percentile {r['null_percentile'][k]:.1f} of {m36.DRAWS} "
                    f"within-season shuffles; in-sample minus held-out {r['in_sample_minus_held_out'][k]:+.6f}."
                ),
                "source": LANE,
                "effect": round(h[k]["effect"], 6),
                "effect_units": units,
                "classification": "unresolved_below_power",
                "league": "nfl",
                "season_start": int(r["seasons"][0]),
                "season_end": int(r["seasons"][-1]),
                "family": FAMILY,
                "category": "market",
                "classification_evidence": (
                    "No resolved wrong sign, split-half reliability not measured, and no positive "
                    f"control bounds an effect this small ({POWER_LANE})."
                ),
                "notes": (
                    f"Family {FAMILY}: 3 reads x 3 metrics = 9 cells, 3 gating log-loss reads, 2 season-half "
                    "coefficients, 2 control reads; declared in the lane before the outcome join; "
                    f"survivor: {'yes' if r['survivor'] else 'no'}. Not served."
                ),
                "interval_low": round(h[k]["low"], 6),
                "interval_high": round(h[k]["high"], 6),
                "probability_positive": round(h[k]["pp"], 4),
                "sample_games": int(r["games"]),
                "sample_blocks": len(r["seasons"]),
                "plain_summary": plain,
            }
        )
    return out


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    m36 = load_mod36()
    d, diag = population(m36)
    print("population", diag, flush=True)
    st = state(m36, d, 0)
    corr = {
        "gap_vs_close_move": float(np.corrcoef(d[NAME], d[MOVE])[0, 1]),
        "gap_vs_served_logit": float(np.corrcoef(d[NAME], d["z"])[0, 1]),
        "close_move_vs_served_logit": float(np.corrcoef(d[MOVE], d["z"])[0, 1]),
    }
    late = d[d["season"].between(*SERVED_MOVE_SEASONS)].reset_index(drop=True)
    corr["gap_vs_close_move_2023_2025"] = float(np.corrcoef(late[NAME], late[MOVE])[0, 1])
    print("correlations", corr, flush=True)
    r1 = read(m36, st, False, "served", 1)
    print(line("read1", r1), flush=True)
    r2 = read(m36, st, True, "move_only", 2)
    print(line("read2", r2), flush=True)
    st3 = state(m36, late, 3)
    r3 = read(m36, st3, False, "served", 3)
    print(line("read3", r3), flush=True)
    ctl = {"all": controls(m36, st), "2023_2025": controls(m36, st3)}
    for k, v in ctl.items():
        print("controls", k, {a: b for a, b in v.items()}, flush=True)
    res = {
        "family": FAMILY,
        "population": diag,
        "draws": m36.DRAWS,
        "correlations": corr,
        "read1_served": r1,
        "read2_beyond_close_move": r2,
        "read3_2023_2025": r3,
        "controls": ctl,
        "elapsed_seconds": time.time() - started,
    }
    batch = (
        cells(m36, "r1", r1, "one offset-logistic coefficient on the leave-one-season-out served logit, opener grade 2020-2025, pushes kept as half.",
              "Checks whether pro games where the computer ratings' middle pick sits away from the opening line beat the opener more often than the model expects. No pick changes.")
        + cells(m36, "r2", r2, "fitted alongside the closing-minus-opening line move and scored against the move-only fit, 2020-2025.",
                "Checks whether the computer ratings add anything once the line's move to kickoff is known. No pick changes.")
        + cells(m36, "r3", r3, "read 1 restricted to 2023-2025, when the served probability already carries a line-move term.",
                "Checks the computer-ratings gap on the three recent pro seasons alone. No pick changes.")
    )
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": batch}, indent=1), encoding="utf-8")
    print(f"elapsed {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
