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
from mod57_p2q_market_reaction_speed import EARLY, RATINGS, surprise_values
from xlg14_fumble_luck_nfl_confirmation import load_mod36

from nfl_ats.pick_probability import active_market_move_feature_version
from nfl_ats.pick_probability_fit import build_fit_population

ART = ROOT / "artifacts"
OUT = ART / "win_totals_anchor"
FEATURES = ART / "win_totals_test" / "features.parquet"
LANE = "docs/lanes/win-totals-2026-11-16.md"
POWER_LANE = "docs/lanes/xlg-harness-power-2026-11-14.md"
FAMILY = "win_totals_anchor_2026_11_16"
NAME = "wt_drift"
WEEKS = (7, 12)
EXPECTED_SIGN = -1.0
PUSH_LABEL = 0.5
HALF_SPLIT = 2023
STREAM = 57


def population(m36):
    version = active_market_move_feature_version(ART)
    pop, _ = build_fit_population(
        ART, ROOT / "data", market_move_feature_version=version, keep_pushes=True
    )
    pop = pop.reset_index(drop=True)
    z, _ = stage_one(pop)
    pop["z"] = z
    pop = pop[pop["season"].between(m36.FIRST_SEASON, m36.LAST_SEASON)].copy()
    pick = sorted((ART / "pick_probability").glob("*/per_game.parquet"))[-1]
    art = pd.read_parquet(pick, columns=["game_id", "week", m36.SERVED_COLUMN])
    chk = pop[["game_id", "z"]].merge(art, on="game_id")
    gap = float((expit(chk["z"]) - chk[m36.SERVED_COLUMN]).abs().max())
    if "week" not in pop.columns:
        sched = pd.read_parquet(FEATURES, columns=["game_id", "week"])
        pop = pop.merge(sched, on="game_id", how="left", validate="one_to_one")
    feat = pd.read_parquet(FEATURES, columns=["game_id", NAME])
    pop = pop.merge(feat, on="game_id", how="left", validate="one_to_one")
    push = pop["margin_vs_open"].eq(0.0)
    pop["y"] = np.where(push, PUSH_LABEL, pop["home_covered"].astype(float))
    pop["push"] = push
    reg = pd.read_parquet(RATINGS)
    reg["gameday"] = pd.to_datetime(reg["gameday"])
    reg = reg.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    pop["surprise"] = surprise_values(reg).reindex(pop["game_id"]).to_numpy()
    pop = pop.sort_values(["season", "game_id"]).reset_index(drop=True)
    diag = {
        "per_game_artifact": str(pick.relative_to(ROOT)),
        "stage_gap_max_abs_probability": gap,
        "games": len(pop),
        "pushes": int(push.sum()),
        "missing_term": int(pop[NAME].isna().sum()),
    }
    return pop, diag


def record(p_new, p_ref, y):
    changed = (p_new >= 0.5) != (p_ref >= 0.5)
    push = y == PUSH_LABEL
    new_right = (p_new >= 0.5) == (y == 1)
    ref_right = (p_ref >= 0.5) == (y == 1)
    return {
        "changed": int(changed.sum()),
        "arm_right": int((changed & ~push & new_right).sum()),
        "ref_right": int((changed & ~push & ref_right).sum()),
        "pushes_changed": int((changed & push).sum()),
        "arm_overall": [int((~push & new_right).sum()), int((~push & ~new_right).sum())],
        "ref_overall": [int((~push & ref_right).sum()), int((~push & ~ref_right).sum())],
    }


def season_block(m36, v, season, rng):
    seasons = np.unique(season)
    idx = [np.flatnonzero(season == s) for s in seasons]
    draws = []
    for _ in range(m36.BOOT):
        pick = rng.integers(0, len(seasons), len(seasons))
        rows = np.concatenate([idx[i] for i in pick])
        draws.append(v[rows].mean(axis=0))
    draws = np.array(draws)
    lo, hi = np.percentile(draws, m36.INTERVAL, axis=0)
    pp = (draws > 0).mean(axis=0) + 0.5 * (draws == 0).mean(axis=0)
    return lo, hi, pp


def gains(m36, p, st):
    return (m36.metric_matrix(p, st["y"]) - st["refs"]["served"]["m"]) * m36.DIRECTION


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    m36 = load_mod36()
    pop, diag = population(m36)
    win = pop[pop["week"].between(*WEEKS) & pop[NAME].notna()].reset_index(drop=True)
    frame = pd.DataFrame(
        {
            "game_id": win["game_id"].to_numpy(),
            "season": win["season"].to_numpy(int),
            "home_covered": win["y"].to_numpy(float),
            m36.SERVED_COLUMN: expit(win["z"].to_numpy(float)),
            NAME: win[NAME].to_numpy(float),
        }
    )
    tpl = {"name": NAME, "column": NAME, "intercept": True}
    st = m36.setup(frame, tpl, STREAM)
    y, z, folds = st["y"], st["z"], st["folds"]
    season = frame["season"].to_numpy(int)
    xd = st["xs"][:, None]
    p, coefs = m36.loso(xd, y, z, folds)
    full = float(m36.fit_offset(xd, y, z)[0])
    inside = gains(m36, expit(z + full * st["xs"]), st).mean(axis=0)
    v = gains(m36, p, st)
    held = v.mean(axis=0)
    rng = np.random.default_rng([m36.SEED, STREAM, 1])
    lo, hi, pp = season_block(m36, v, season, rng)
    halves = {}
    for name, mask in (("2020_22", season < HALF_SPLIT), ("2023_25", season >= HALF_SPLIT)):
        c = float(m36.fit_offset(xd[mask], y[mask], z[mask])[0])
        halves[name] = {"coefficient_per_sd": c, "held_out_ll_gain": float(v[mask, 1].mean())}
    nrng = np.random.default_rng([m36.SEED, STREAM, 0])
    null = []
    for _ in range(m36.DECLARED_DRAWS):
        xp = m36.permuted(st, nrng)
        pn, _ = m36.loso(xp[:, None], y, z, folds)
        cn = float(m36.fit_offset(xp[:, None], y, z)[0])
        null.append([*gains(m36, pn, st).mean(axis=0), cn])
    null = np.array(null)
    pct = {k: m36.percentile_of(null[:, i], held[i]) for i, k in enumerate(m36.SHORT)}
    p_int, _ = m36.loso(np.ones((st["n"], 1)), y, z, folds)
    rec = record(p, expit(z), y)
    allp = pop.copy()
    allp["p_ref"] = expit(allp["z"])
    allp["p_new"] = allp["p_ref"]
    key = allp.set_index("game_id").index
    newp = pd.Series(p, index=frame["game_id"])
    allp.loc[key.isin(newp.index), "p_new"] = newp.reindex(key[key.isin(newp.index)]).to_numpy()
    rec_all = record(allp["p_new"].to_numpy(), allp["p_ref"].to_numpy(), allp["y"].to_numpy())
    early = pop[pop["week"].between(*EARLY) & pop[NAME].notna() & pop["surprise"].notna()]
    wsur = win["surprise"].notna().to_numpy()
    corr = {
        "served_logit": float(np.corrcoef(st["xs"], z)[0, 1]),
        "surprise_to_date_weeks_7_12": float(np.corrcoef(st["xs"][wsur], win["surprise"][wsur])[0, 1]),
        "T_weeks_2_6": float(np.corrcoef(early[NAME], early["surprise"])[0, 1]),
        "T_games": len(early),
    }
    halves_ok = all(h["coefficient_per_sd"] * EXPECTED_SIGN > 0 for h in halves.values())
    passed = bool(pct["ll"] > 100 * m36.NOTABLE[1] and full * EXPECTED_SIGN > 0 and halves_ok)
    seasons = sorted(int(s) for s in np.unique(season))
    res = {
        "family": FAMILY,
        "population": diag,
        "games": st["n"],
        "pushes": int((y == PUSH_LABEL).sum()),
        "nonzero_games": int((st["xs"] != 0).sum()),
        "sd": st["sd"],
        "coefficient_per_sd": full,
        "fold_coefficients_per_sd": dict(zip(map(str, seasons), map(float, coefs[:, 0]), strict=True)),
        "folds_expected_sign": int((coefs[:, 0] * EXPECTED_SIGN > 0).sum()),
        "halves": halves,
        "held_out": {
            k: {"effect": float(held[i]), "low": float(lo[i]), "high": float(hi[i]), "pp": float(pp[i])}
            for i, k in enumerate(m36.SHORT)
        },
        "in_sample": dict(zip(m36.SHORT, map(float, inside), strict=True)),
        "in_sample_minus_held_out": {k: float(inside[i] - held[i]) for i, k in enumerate(m36.SHORT)},
        "null_percentile": pct,
        "null_ll_q05_q95": [float(np.percentile(null[:, 1], 5)), float(np.percentile(null[:, 1], 95))],
        "null_coefficient_sd": float(null[:, 3].std(ddof=1)),
        "intercept_only_ll_gain": float(gains(m36, p_int, st)[:, 1].mean()),
        "record_weeks_7_12": rec,
        "record_all_games": rec_all,
        "correlations": corr,
        "passed": passed,
        "elapsed_s": round(time.time() - started, 1),
    }
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    cells = []
    for i, (k, units) in enumerate(zip(m36.SHORT, m36.METRICS, strict=True)):
        h = res["held_out"][k]
        cells.append(
            {
                "name": f"win_totals_anchor_wt_drift_wk7_12_{units}",
                "description": (
                    "Preseason win-total percentile minus current market-rating percentile, home "
                    "minus away, weeks 7-12, one offset-logistic coefficient per SD on the served "
                    "stage at the opener 2020-25, pushes kept at 0.5, leave-one-season-out. "
                    f"Coefficient {full:+.4f} per SD (expected negative), folds "
                    f"{res['folds_expected_sign']}/{len(seasons)} negative; halves "
                    f"{halves['2020_22']['coefficient_per_sd']:+.4f} / "
                    f"{halves['2023_25']['coefficient_per_sd']:+.4f}; null percentile "
                    f"{pct[k]:.1f} of {m36.DECLARED_DRAWS} within-season shuffles; changed "
                    f"{rec['changed']} record {rec['arm_right']}-{rec['ref_right']}; in-sample "
                    f"minus held-out {res['in_sample_minus_held_out'][k]:+.6f}."
                ),
                "source": LANE,
                "effect": round(h["effect"], 6),
                "effect_units": units,
                "classification": "unresolved_below_power",
                "league": "nfl",
                "season_start": m36.FIRST_SEASON,
                "season_end": m36.LAST_SEASON,
                "family": FAMILY,
                "category": "market",
                "classification_evidence": (
                    "No resolved wrong sign and no positive control bounding an effect this "
                    f"small; the NFL add-one harness floor is in {POWER_LANE}."
                ),
                "notes": (
                    f"Family {FAMILY}: one declared fresh look after the weeks 7-12 placebo of "
                    "win_totals_drift_2026_11_16; 1 gating read, 3 cells; season-block interval; "
                    f"declared pass {passed}. Not served."
                ),
                "interval_low": round(h["low"], 6),
                "interval_high": round(h["high"], 6),
                "probability_positive": round(h["pp"], 4),
                "sample_games": int(st["n"]),
                "sample_blocks": len(seasons),
                "plain_summary": (
                    "Checks whether teams whose market rating has moved away from their preseason "
                    "win total keep beating the opening line in that direction in weeks 7-12."
                ),
            }
        )
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
