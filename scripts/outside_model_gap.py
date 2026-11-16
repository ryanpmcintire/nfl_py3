import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "game_features.parquet"
ELO = ROOT / "data" / "external" / "fivethirtyeight_elo" / "nfl_elo_pregame_joined.parquet"
NFELO = ROOT / "data" / "external" / "nfelo" / "nfelo_games.parquet"
NFELO_LINES = ROOT / "data" / "external" / "nfelo" / "historic_projected_spreads.parquet"
PICK_DIR = ROOT / "artifacts" / "pick_probability"
OUT = ROOT / "artifacts" / "outside_model_gap"
LANE = "docs/lanes/outside-model-gap-2026-11-16.md"
POWER_LANE = "docs/lanes/xlg-harness-power-2026-11-14.md"
FAMILY = "outside_model_gap_2026_11_16"
ELO_SCALE = 400.0
POINTS_PER_ELO_538 = 1 / 25.0
SEED = 20261116
DRAWS = 200
BOOT = 2000
BAR = 95.0
INTERVAL = (2.5, 97.5)
LOOKS = [
    ("qbelo_gap", "close", 2011, 2022),
    ("qbelo_gap", "opener", 2020, 2022),
    ("qb_adjust", "close", 2011, 2022),
    ("qb_adjust", "opener", 2020, 2022),
    ("nfelo_gap", "close", 2011, 2025),
    ("nfelo_gap", "opener", 2020, 2025),
]
WHAT = {
    "qbelo_gap": "538 quarterback-adjusted Elo implied home margin minus the line",
    "qb_adjust": "538 quarterback adjustment alone (QB Elo minus plain Elo implied home margin)",
    "nfelo_gap": "nfelo pre-market model home margin minus the line",
}
PLAIN = {
    "qbelo_gap": "FiveThirtyEight's quarterback-adjusted rating",
    "qb_adjust": "FiveThirtyEight's starting-quarterback adjustment",
    "nfelo_gap": "the nfelo model's own number",
}


def implied_margin(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return ELO_SCALE * np.log10(p / (1 - p)) * POINTS_PER_ELO_538


def terms():
    e = pd.read_parquet(ELO)
    home_first = e["team1"].eq(e["home_team"]).to_numpy()
    qp = np.where(home_first, e["qbelo_prob1"], e["qbelo_prob2"])
    ep = np.where(home_first, e["elo_prob1"], e["elo_prob2"])
    e538 = pd.DataFrame(
        {
            "game_id": e["game_id"],
            "qbelo_margin": implied_margin(qp),
            "qb_adjust": implied_margin(qp) - implied_margin(ep),
        }
    ).drop_duplicates("game_id")
    n = pd.read_parquet(NFELO, columns=["game_id", "nfelo_dif_base"])
    h = pd.read_parquet(NFELO_LINES, columns=["game_id", "home_line_pre_regression"])
    m = h.merge(n, on="game_id").dropna()
    slope = float(np.linalg.lstsq(m[["nfelo_dif_base"]].to_numpy(), m["home_line_pre_regression"].to_numpy(), rcond=None)[0][0])
    nf = pd.DataFrame({"game_id": n["game_id"], "nfelo_margin": -slope * n["nfelo_dif_base"]}).drop_duplicates("game_id")
    return e538, nf, {"nfelo_points_per_elo": -slope, "nfelo_conversion_rows": len(m)}


def frame():
    e538, nf, conv = terms()
    f = pd.read_parquet(FEATURES, columns=["game_id", "season", "week", "spread_line", "home_score", "away_score"])
    pick = sorted(PICK_DIR.glob("*/per_game.parquet"))[-1]
    op = pd.read_parquet(pick, columns=["game_id", "tue_open_home_spread"])
    d = f.merge(op, on="game_id", how="left").merge(e538, on="game_id", how="left").merge(nf, on="game_id", how="left")
    d["result"] = d["home_score"] - d["away_score"]
    both = d[["spread_line", "tue_open_home_spread"]].dropna()
    conv["close_opener_corr"] = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
    conv["opener_rows"] = len(both)
    conv["pick_artifact"] = str(pick.relative_to(ROOT))
    if conv["close_opener_corr"] <= 0:
        raise SystemExit("opener and close disagree in sign convention")
    return d, conv


def look_data(d, term, line, first, last):
    col = "spread_line" if line == "close" else "tue_open_home_spread"
    g = d[d["season"].between(first, last)].copy()
    g["line"] = g[col]
    if term == "qb_adjust":
        g["x"] = g["qb_adjust"]
    else:
        g["x"] = g["qbelo_margin" if term == "qbelo_gap" else "nfelo_margin"] - g["line"]
    g = g.dropna(subset=["x", "line", "result"]).reset_index(drop=True)
    g["ats"] = g["result"] - g["line"]
    return g


def stats(x, ats):
    r = float(np.corrcoef(x, ats)[0, 1])
    ok = (x != 0) & (ats != 0)
    acc = 100 * float((np.sign(x[ok]) == np.sign(ats[ok])).mean()) - 50
    return r, acc, ok


def screen():
    OUT.mkdir(parents=True, exist_ok=True)
    d, conv = frame()
    print("conversion", conv, flush=True)
    results = []
    for i, (term, line, first, last) in enumerate(LOOKS):
        g = look_data(d, term, line, first, last)
        x, ats, season = g["x"].to_numpy(float), g["ats"].to_numpy(float), g["season"].to_numpy()
        r, acc, ok = stats(x, ats)
        rng = np.random.default_rng([SEED, i])
        groups = [np.flatnonzero(season == s) for s in np.unique(season)]
        nr, na = [], []
        for _ in range(DRAWS):
            xs = x.copy()
            for k in groups:
                xs[k] = x[rng.permutation(k)]
            a, b, _ = stats(xs, ats)
            nr.append(a)
            na.append(b)
        nr, na = np.array(nr), np.array(na)
        seasons = sorted(np.unique(season))
        cut = seasons[(len(seasons) + 1) // 2 - 1]
        halves = [stats(x[m], ats[m])[:2] for m in (season <= cut, season > cut)]
        hit = (np.sign(x[ok]) == np.sign(ats[ok])).astype(float)
        brng = np.random.default_rng([SEED, i, 1])
        bs = 100 * brng.multinomial(len(hit), np.full(len(hit), 1 / len(hit)), size=BOOT) @ hit / len(hit) - 50
        lo, hi = (float(v) for v in np.percentile(bs, INTERVAL))
        pct_r = float(100 * ((nr < r).mean() + 0.5 * (nr == r).mean()))
        pct_a = float(100 * ((na < acc).mean() + 0.5 * (na == acc).mean()))
        survivor = bool(pct_r >= BAR and halves[0][0] > 0 and halves[1][0] > 0)
        res = {
            "term": term, "line": line, "seasons": [first, last], "games": len(g),
            "decided_games": int(ok.sum()), "term_sd": float(x.std(ddof=1)),
            "r": r, "r_null_q95": float(np.percentile(nr, 95)), "r_null_percentile": pct_r,
            "accuracy_points": acc, "accuracy_interval": [lo, hi],
            "accuracy_probability_positive": float((bs > 0).mean() + 0.5 * (bs == 0).mean()),
            "accuracy_null_percentile": pct_a, "record": [int(hit.sum()), int(len(hit) - hit.sum())],
            "halves_cut_season": int(cut), "halves_r_acc": halves, "survivor": survivor,
        }
        results.append(res)
        print(json.dumps(res), flush=True)
    (OUT / "screen.json").write_text(json.dumps({"conversion": conv, "looks": results}, indent=1), encoding="utf-8")
    cells = []
    for res in results:
        lo, hi = res["accuracy_interval"]
        line_word = "opening" if res["line"] == "opener" else "closing"
        cells.append({
            "name": f"outside_model_gap_{res['term']}_{res['line']}_accuracy_points",
            "description": (
                f"{WHAT[res['term']]} ({res['line']} line), home minus away, points; forced pick by its sign, "
                f"seasons {res['seasons'][0]}-{res['seasons'][1]}, {res['decided_games']} decided games, record "
                f"{res['record'][0]}-{res['record'][1]}; r with ATS margin {res['r']:+.4f} at null percentile "
                f"{res['r_null_percentile']:.1f} of {DRAWS} within-season shuffles; halves (cut {res['halves_cut_season']}) r "
                f"{res['halves_r_acc'][0][0]:+.4f} / {res['halves_r_acc'][1][0]:+.4f}; accuracy null percentile "
                f"{res['accuracy_null_percentile']:.1f}; survivor {res['survivor']}."
            ),
            "source": LANE,
            "effect": round(res["accuracy_points"], 6),
            "effect_units": "accuracy_points",
            "classification": "unresolved_below_power",
            "league": "nfl",
            "season_start": res["seasons"][0],
            "season_end": res["seasons"][1],
            "family": FAMILY,
            "category": "market",
            "classification_evidence": (
                "No resolved wrong sign (bootstrap interval not wholly below zero), split-half reliability not "
                "measured, and no positive control bounds an effect this small at this sample "
                f"({POWER_LANE}: one-term opener tests see about 1.5 points)."
            ),
            "notes": (
                f"Family {FAMILY}: 6 declared screen looks (3 terms x close/opener), gate r null percentile >= {BAR:.0f} "
                "and positive in both halves; declared in the lane before outcomes were joined. Not served."
            ),
            "interval_low": round(lo, 6),
            "interval_high": round(hi, 6),
            "probability_positive": round(res["accuracy_probability_positive"], 4),
            "sample_games": res["decided_games"],
            "sample_blocks": res["seasons"][1] - res["seasons"][0] + 1,
            "plain_summary": (
                f"Checks whether picking the side {PLAIN[res['term']]} likes more than the {line_word} line beats that "
                "line. It did not stand out from chance; nothing is ruled out and no pick changes."
                if not res["survivor"] else
                f"Checks whether picking the side {PLAIN[res['term']]} likes more than the {line_word} line beats that "
                "line. It stood out in the screen and goes to one test inside the served probability; no pick changes yet."
            ),
        })
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1), encoding="utf-8")


def served():
    from xlg14_fumble_luck_nfl_confirmation import coefficient_draws, controls, load_mod36

    m36 = load_mod36()
    d, conv = frame()
    hook, g = m36.games()
    g = g[["game_id", "season", "home_covered", m36.SERVED_COLUMN]].merge(
        d[["game_id", "tue_open_home_spread", "nfelo_margin"]], on="game_id", how="left", validate="one_to_one"
    )
    g["nfelo_gap"] = (g["nfelo_margin"] - g["tue_open_home_spread"]).fillna(0.0)
    tpl = {"name": "nfelo_gap", "column": "nfelo_gap", "intercept": False, "matched": "served", "reported": False, "what": WHAT["nfelo_gap"]}
    index = len(m36.TEMPLATES) + 1
    st = m36.setup(g, tpl, index)
    real = m36.real_read(st, tpl)
    held = real["served"]
    rng = np.random.default_rng([SEED, index, 0])
    rows = []
    for i in range(DRAWS):
        m36.draw_rows(st, tpl, "null", 0.0, i, m36.permuted(st, rng), rows)
    null = pd.DataFrame(rows)
    draws = coefficient_draws(m36, st)
    lo, hi = (float(v) for v in np.percentile(draws, INTERVAL))
    xd = m36.design(st["xs"], False)
    full = m36.fit_offset(xd, st["y"], st["z"])
    inside = ((m36.metric_matrix(expit(st["z"] + xd @ full), st["y"]) - st["refs"]["served"]["m"]) * m36.DIRECTION).mean(axis=0)
    res = {
        "pick_artifact": str(hook.relative_to(ROOT)),
        "population": "served per_game rows 2020-2025; the 34 opener pushes are absent from this artifact",
        "games": st["n"],
        "nonzero_games": int((st["xs"] != 0).sum()),
        "opener_whole_share": float((g["tue_open_home_spread"] % 1 == 0).mean()),
        "coefficient_per_sd": real["full_coefficient_per_sd"],
        "coefficient_interval_per_sd": [lo, hi],
        "coefficient_probability_positive": float((draws > 0).mean()),
        "coefficient_null_percentile": m36.percentile_of(null["coef"], real["full_coefficient_per_sd"]),
        "fold_coefficients_per_sd": real["fold_coefficients_per_sd"],
        "held_out": held,
        "in_sample_minus_held_out": {k: float(inside[i] - held[k]["effect"]) for i, k in enumerate(m36.SHORT)},
        "null_percentile": {k: m36.percentile_of(null[k], held[k]["effect"]) for k in m36.SHORT},
        "null": {k: m36.band(null[k]) for k in m36.SHORT},
        "controls": controls(m36, st),
    }
    print(json.dumps(res, indent=1, default=float), flush=True)
    (OUT / "served.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    cells = []
    for k, units in zip(m36.SHORT, m36.METRICS, strict=True):
        cells.append({
            "name": f"outside_model_gap_nfelo_gap_served_opener_{units}",
            "description": (
                f"{WHAT['nfelo_gap']} (Tuesday opener), home minus away, points; the one declared served look after "
                "the close-line screen survivor: offset logistic on the two-stage out-of-season served probability, "
                f"opener games 2020-2025 (served artifact excludes the opener pushes), leave-one-season-out. Games {res['games']}, nonzero "
                f"{res['nonzero_games']}; changed {held['changed']}, record {held['arm_right']}-{held['ref_right']}; "
                f"coefficient per SD {res['coefficient_per_sd']:+.4f} [{lo:+.4f}, {hi:+.4f}]; null percentile "
                f"{res['null_percentile'][k]:.1f} of {DRAWS} within-season shuffles; in-sample minus held-out "
                f"{res['in_sample_minus_held_out'][k]:+.6f}."
            ),
            "source": LANE,
            "effect": round(held[k]["effect"], 6),
            "effect_units": units,
            "classification": "unresolved_below_power",
            "league": "nfl",
            "season_start": 2020,
            "season_end": 2025,
            "family": FAMILY,
            "category": "market",
            "classification_evidence": (
                "No resolved wrong sign, split-half reliability not measured, and no positive control bounds an "
                f"effect this small ({POWER_LANE})."
            ),
            "notes": f"Family {FAMILY}: served look 1 of 1, gating read log-loss null percentile >= {BAR:.0f}. Not served.",
            "interval_low": round(held[k]["low"], 6),
            "interval_high": round(held[k]["high"], 6),
            "probability_positive": round(held[k]["pp"], 4),
            "sample_games": int(res["games"]),
            "sample_blocks": 6,
            "plain_summary": (
                "Checks whether adding the nfelo model's own number to the card's probability improves picks "
                "against the opening line. No pick changes from this check."
            ),
        })
    (OUT / "served_weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"screen": screen, "served": served}[sys.argv[1]]()
