from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lead98_price_skew import metrics
from mod57_p2c_uncorrected_error_model import (
    ARTIFACT,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    QUIET,
    SERVED_FEATURES,
    boot,
    paired,
    per_season,
)
from mod57_p2l_availability_ceiling import MOVED, ORACLE_TABLE, run_eval
from mod57_p2m_knowable_ceiling import KNOW, SHUFFLE_SEED, shuffle_within_season, unit_columns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/mod57_p2o"
LABEL = "LEAK_NOT_SERVABLE"
FAMILY_LINE = "mod57_how_wrong_is_the_line"
FAMILY_UNITS = "mod57_availability_ceiling"
MIN_PRIOR_GAMES = 4
FIRST_SEASON = 2009
LAST_SEASON = 2025
THIRD_EDGES = (0, 6, 12, 99)
THIRD_LABELS = ("weeks_1_6", "weeks_7_12", "weeks_13_18")
GAP_EDGES = (-1.0, 1.0, 3.0, 1e9)
GAP_LABELS = ("under_1", "1_to_3", "over_3")
CHECK_GAMES = 12
CHECK_SEED = 20261112
PERTURB = 10.0


def build_table():
    served = pd.read_parquet(SERVED_FEATURES)
    served["gameday"] = pd.to_datetime(served["gameday"])
    reg = served.loc[
        (served["game_type"] == "REG")
        & served["season"].between(FIRST_SEASON, LAST_SEASON)
        & served["home_score"].notna()
        & served["away_score"].notna()
    ].copy()
    reg["margin"] = reg["home_score"] - reg["away_score"]
    reg = reg.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return served, reg


def team_index(reg):
    teams = sorted(set(reg["home_team"]) | set(reg["away_team"]))
    return {t: i for i, t in enumerate(teams)}


def design(frame, index):
    n_teams = len(index)
    x = np.zeros((len(frame), n_teams + 1))
    rows = np.arange(len(frame))
    x[rows, frame["home_team"].map(index).to_numpy()] = 1.0
    x[rows, frame["away_team"].map(index).to_numpy()] = -1.0
    x[:, -1] = 1.0
    return x


def fit_coef(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def hindsight(reg, index):
    pred = np.full(len(reg), np.nan)
    pred_with_own = np.full(len(reg), np.nan)
    checks = {"max_abs_sum_of_ratings": 0.0, "perturbed_games": 0, "perturb_changed_excluded": 0}
    rng = np.random.default_rng(CHECK_SEED)
    for _, ids in reg.groupby("season").indices.items():
        x = design(reg.iloc[ids], index)
        y = reg["margin"].to_numpy(float)[ids]
        full = fit_coef(x, y)
        pred_with_own[ids] = x @ full
        picks = set(rng.choice(len(ids), 2, replace=False).tolist())
        for i in range(len(ids)):
            keep = np.arange(len(ids)) != i
            coef = fit_coef(x[keep], y[keep])
            pred[ids[i]] = x[i] @ coef
            checks["max_abs_sum_of_ratings"] = max(
                checks["max_abs_sum_of_ratings"], float(abs(coef[:-1].sum()))
            )
            if i in picks:
                y_changed = y.copy()
                y_changed[i] += PERTURB
                again = fit_coef(x[keep], y_changed[keep])
                checks["perturbed_games"] += 1
                checks["perturb_changed_excluded"] += int(abs(x[i] @ again - pred[ids[i]]) > 1e-9)
    gap = np.abs(pred - pred_with_own)
    checks["games_where_including_own_changes_prediction"] = int((gap > 1e-9).sum())
    checks["games"] = len(reg)
    return pred, checks


def run_pregame(reg, index):
    pred = pd.Series(np.nan, index=reg["game_id"].to_numpy())
    fallback = 0
    prior_coef = {}
    for season in sorted(reg["season"].unique()):
        past = reg.loc[reg["season"] == season - 1]
        if len(past):
            prior_coef[season] = fit_coef(design(past, index), past["margin"].to_numpy(float))
    for season, sdf in reg.groupby("season"):
        for day, target in sdf.groupby("gameday"):
            train = sdf.loc[sdf["gameday"] < day]
            counts = np.zeros(len(index))
            if len(train):
                counts = np.bincount(
                    train["home_team"].map(index), minlength=len(index)
                ) + np.bincount(train["away_team"].map(index), minlength=len(index))
            coef = None
            xt = design(target, index)
            for j, (_, row) in enumerate(target.iterrows()):
                low = min(counts[index[row["home_team"]]], counts[index[row["away_team"]]])
                if low < MIN_PRIOR_GAMES:
                    use = prior_coef.get(season)
                    fallback += 1
                else:
                    if coef is None:
                        coef = fit_coef(design(train, index), train["margin"].to_numpy(float))
                    use = coef
                if use is not None:
                    pred[row["game_id"]] = float(xt[j] @ use)
    return pred, fallback


def truncation_check(reg, index, full_pred):
    rng = np.random.default_rng(CHECK_SEED)
    pool = reg.loc[reg["season"] > FIRST_SEASON]
    early = pool.loc[pool["week"] == 2].sample(2, random_state=CHECK_SEED)
    rest = pool.drop(early.index).sample(CHECK_GAMES - 2, random_state=int(rng.integers(1 << 30)))
    rows = []
    for _, g in pd.concat([early, rest]).iterrows():
        keep = reg["gameday"] < g["gameday"]
        same = (reg["gameday"] == g["gameday"]) & (reg["season"] == g["season"])
        trunc = pd.concat([reg.loc[keep], reg.loc[same].assign(margin=0.0)])
        trunc = trunc.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        again, _ = run_pregame(trunc, index)
        rows.append(
            {
                "game_id": g["game_id"],
                "full": float(full_pred[g["game_id"]]),
                "truncated": float(again[g["game_id"]]),
                "abs_diff": float(abs(full_pred[g["game_id"]] - again[g["game_id"]])),
            }
        )
    return rows


def slope_boot(x, y, block, rng):
    n_blocks = int(block.max()) + 1
    n = np.bincount(block, minlength=n_blocks).astype(float)
    sx = np.bincount(block, weights=x, minlength=n_blocks)
    sy = np.bincount(block, weights=y, minlength=n_blocks)
    sxx = np.bincount(block, weights=x * x, minlength=n_blocks)
    sxy = np.bincount(block, weights=x * y, minlength=n_blocks)

    def slope(idx):
        nn, a, b, c, d = (v[idx].sum(axis=-1) for v in (n, sx, sy, sxx, sxy))
        return (d - a * b / nn) / (c - a * a / nn)

    point = float(slope(np.arange(n_blocks)))
    draws = slope(rng.integers(0, n_blocks, (BOOTSTRAP_SAMPLES, n_blocks)))
    return {
        "mean": point,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_positive": float((draws > 0).mean()),
    }


def record(hit):
    wins = int(hit.sum())
    return {
        "record": f"{wins}-{len(hit) - wins}",
        "accuracy": float(hit.mean()) if len(hit) else None,
        "games": len(hit),
    }


def pick_accuracy(d, rating, line, rng):
    dec = d.loc[(d["margin"] != d[line]) & (d[rating] != d[line])]
    hit = ((dec[rating] > dec[line]) == (dec["margin"] > dec[line])).to_numpy()
    out = {"overall": record(hit)}
    block = pd.factorize(dec["season"])[0]
    overall = boot(hit.astype(float), block, rng)
    out["overall"]["ci95"] = overall["ci95"]
    out["overall"]["probability_above_half"] = float(
        (rng_draws(hit.astype(float) - 0.5, block, rng) > 0).mean()
    )
    third = pd.cut(dec["week"], THIRD_EDGES, labels=THIRD_LABELS).to_numpy()
    out["by_third"] = {k: record(hit[third == k]) for k in THIRD_LABELS}
    gap = (dec[rating] - dec[line]).abs()
    band = pd.cut(gap, GAP_EDGES, labels=GAP_LABELS, right=False).to_numpy()
    out["by_abs_gap"] = {k: record(hit[band == k]) for k in GAP_LABELS}
    return out


def rng_draws(d, block, rng):
    n_blocks = int(block.max()) + 1
    sums = np.bincount(block, weights=d, minlength=n_blocks)
    sizes = np.bincount(block, minlength=n_blocks)
    idx = rng.integers(0, n_blocks, (BOOTSTRAP_SAMPLES, n_blocks))
    return sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)


def line_stats(frame, rating, line, rng):
    d = frame.dropna(subset=[rating, line, "margin"]).reset_index(drop=True)
    gap = d[rating] - d[line]
    err_line = d["margin"] - d[line]
    err_rating = d["margin"] - d[rating]
    block = pd.factorize(d["season"])[0]
    third = pd.cut(d["week"], THIRD_EDGES, labels=THIRD_LABELS)
    stats = {
        "games": len(d),
        "mean_abs_gap": float(gap.abs().mean()),
        "median_abs_gap": float(gap.abs().median()),
        "gap_by_third": {
            k: {
                "mean": float(gap.abs()[third == k].mean()),
                "median": float(gap.abs()[third == k].median()),
            }
            for k in THIRD_LABELS
        },
        "mae_rating": float(err_rating.abs().mean()),
        "mae_line": float(err_line.abs().mean()),
        "rmse_rating": float(np.sqrt((err_rating**2).mean())),
        "rmse_line": float(np.sqrt((err_line**2).mean())),
        "mae_diff_line_minus_rating": boot(
            (err_line.abs() - err_rating.abs()).to_numpy(), block, rng
        ),
        "mse_diff_line_minus_rating": boot((err_line**2 - err_rating**2).to_numpy(), block, rng),
        "calibration_slope": slope_boot(gap.to_numpy(), err_line.to_numpy(), block, rng),
        "pick_accuracy": pick_accuracy(d, rating, line, rng),
    }
    return stats


def line_cells(results):
    cells = []
    units = {
        "mae_diff_line_minus_rating": "margin_points",
        "mse_diff_line_minus_rating": "squared_margin_points",
        "calibration_slope": "calibration_slope",
    }
    for rating in ("H", "P"):
        for line in ("close", "open"):
            s = results["line_stats"][f"{rating}_{line}"]
            seasons = (FIRST_SEASON if rating == "H" else FIRST_SEASON + 1, LAST_SEASON)
            if line == "open":
                seasons = (2020, LAST_SEASON)
            leak = (
                "POSITIVE CONTROL, LEAK, USES FUTURE GAMES, NOT SERVABLE: "
                if rating == "H"
                else "Pregame rating (games before the date only): "
            )
            for stat, unit in units.items():
                cells.append(
                    {
                        "name": f"mod57_p2o_{rating}_{line}_{stat}",
                        "description": (
                            f"{leak}{stat} of the season rating against the {line} line, "
                            "season-block bootstrap"
                        ),
                        "effect": s[stat]["mean"],
                        "effect_units": unit,
                        "interval_low": s[stat]["ci95"][0],
                        "interval_high": s[stat]["ci95"][1],
                        "probability_positive": s[stat]["probability_positive"],
                        "season_start": seasons[0],
                        "season_end": seasons[1],
                    }
                )
    return cells


def own_removed_units(served, stems):
    parts = []
    for side in ("home", "away"):
        part = served[["game_id", "season", "gameday", f"{side}_team"]].rename(
            columns={f"{side}_team": "team"}
        )
        part["side"] = side
        for stem in stems:
            part[stem] = served[f"{side}_{stem}"].to_numpy(float)
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    long["gameday"] = pd.to_datetime(long["gameday"])
    long = long.sort_values(["season", "team", "gameday", "game_id"]).reset_index(drop=True)
    own = {}
    prior = {}
    for stem in stems:
        val = long[stem]
        filled = val.fillna(0.0)
        present = val.notna().astype(float)
        long["_f"], long["_p"] = filled, present
        group = long.groupby(["season", "team"])
        total = group["_f"].transform("sum")
        count = group["_p"].transform("sum")
        denom = (count - present).replace(0, np.nan)
        own[stem] = ((total - filled) / denom).to_numpy()
        cum_f = group["_f"].cumsum() - filled
        cum_p = group["_p"].cumsum() - present
        prior[stem] = (cum_f / cum_p.replace(0, np.nan)).to_numpy()
    result = {}
    for tag, table in (("own", own), ("prior", prior)):
        wide = long[["game_id", "side"]].copy()
        for stem in stems:
            wide[stem] = table[stem]
        home = wide.loc[wide["side"] == "home"].set_index("game_id")[list(stems)]
        away = wide.loc[wide["side"] == "away"].set_index("game_id")[list(stems)]
        diff = (home - away.reindex(home.index)).add_prefix(f"diff_unit{tag}_")
        result[tag] = diff.reset_index()
    return result, long


def own_check(long, stems, tables):
    rng = np.random.default_rng(CHECK_SEED)
    stem = stems[0]
    column = f"diff_unitown_{stem}"
    table = tables["own"].set_index("game_id")[column]
    sample = (
        long.loc[long["side"] == "home"]
        .dropna(subset=[stem])
        .sample(5, random_state=int(rng.integers(1 << 30)))
    )
    worst = 0.0
    for _, row in sample.iterrows():
        away = long.loc[(long["game_id"] == row["game_id"]) & (long["side"] == "away")].iloc[0]
        means = []
        for team in (row, away):
            mates = long.loc[
                (long["season"] == team["season"])
                & (long["team"] == team["team"])
                & (long["game_id"] != team["game_id"])
            ]
            means.append(mates[stem].mean())
        worst = max(worst, float(abs(means[0] - means[1] - table[row["game_id"]])))
    return worst


def unit_arms(served):
    stems = unit_columns(served)
    tables, long = own_removed_units(served, stems)
    oracle = pd.read_parquet(ORACLE_TABLE)
    own_cols = tuple(c for c in tables["own"].columns if c != "game_id")
    prior_cols = tuple(c for c in tables["prior"].columns if c != "game_id")
    features = (
        served.merge(oracle[["game_id", *KNOW]], on="game_id", how="left")
        .merge(tables["own"], on="game_id", how="left")
        .merge(tables["prior"], on="game_id", how="left")
    )
    for column in (*KNOW, *own_cols, *prior_cols):
        features[column] = features[column].fillna(0.0)
    shuf = shuffle_within_season(features, own_cols, np.random.default_rng(SHUFFLE_SEED))
    arms = {
        "served_base": (),
        "U_A_oracles_plus_own_removed_units": (*KNOW, *own_cols),
        "U_alone_own_removed_units": own_cols,
        "U_prior_games_only_units": prior_cols,
        "SHUFFLED_U_alone_control": shuf,
    }
    return features, arms, stems, long


def unit_section():
    served = pd.read_parquet(SERVED_FEATURES)
    features, arms, stems, long = unit_arms(served)
    tables, _ = own_removed_units(served, stems)
    removal_check = own_check(long, stems, tables)
    per_game = pd.read_parquet(ARTIFACT)
    per_game = per_game.loc[per_game["season"].between(2020, 2025)].reset_index(drop=True)
    ids = per_game["game_id"]
    y = per_game["home_covered"].to_numpy(float)
    move = np.abs(per_game["open_move"].to_numpy(float))
    pops = {"all": np.ones(len(per_game), bool), "quiet": move <= QUIET, "moved": move >= MOVED}
    probs = {}
    for tag, extra in arms.items():
        scored, _ = run_eval(features, extra, tag)
        probs[tag] = (
            scored.set_index("game_id")["home_cover_probability_at_open"]
            .reindex(ids)
            .to_numpy(float)
        )
    brng = np.random.default_rng(BOOTSTRAP_SEED)
    section = {
        "label": f"{LABEL}: own-row-removed unit rates use other games of the same season",
        "games": len(per_game),
        "own_removal_manual_check_max_abs_diff": removal_check,
        "missing": {t: int(np.isnan(p).sum()) for t, p in probs.items()},
        "populations": {},
    }
    cells = []
    primary = "U_A_oracles_plus_own_removed_units"
    for pop, mask in pops.items():
        sub = per_game.loc[mask].reset_index(drop=True)
        entry = {"games": int(mask.sum()), "records": {}, "vs_served_base": {}, "per_season": {}}
        for tag, p in probs.items():
            entry["records"][tag] = metrics(p[mask], y[mask])
            entry["per_season"][tag] = per_season(sub, p[mask])
        for tag in arms:
            if tag == "served_base":
                continue
            d = paired(sub, probs[tag][mask], probs["served_base"][mask], brng)
            entry["vs_served_base"][tag] = d
            if tag.startswith("SHUFFLED"):
                continue
            a = d["accuracy_points"]
            role = "primary" if tag == primary else "descriptive arm"
            cells.append(
                {
                    "name": f"mod57_p2o_{tag}_{pop}_accuracy_points",
                    "description": (
                        f"POSITIVE CONTROL, LEAK, NOT SERVABLE ({role}): base plus {tag} "
                        f"minus served base, {pop} games, opener grade, week-block bootstrap"
                    ),
                    "effect": a["mean"],
                    "effect_units": "accuracy_points",
                    "interval_low": a["ci95"][0],
                    "interval_high": a["ci95"][1],
                    "probability_positive": a["probability_positive"],
                    "season_start": 2020,
                    "season_end": 2025,
                }
            )
        section["populations"][pop] = entry
    return section, cells


def sign_check(reg):
    covered = pd.read_parquet(ARTIFACT)[["game_id", "home_covered"]]
    d = reg.merge(covered, on="game_id").dropna(subset=["open_line"])
    d = d.loc[d["margin"] != d["open_line"]]
    return float(((d["margin"] > d["open_line"]) == (d["home_covered"] == 1)).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, reg = build_table()
    index = team_index(reg)
    pred_h, checks_h = hindsight(reg, index)
    pred_p_series, fallback = run_pregame(reg, index)
    reg["R_H"] = pred_h
    reg["R_P"] = pred_p_series.reindex(reg["game_id"]).to_numpy()
    trunc = truncation_check(reg, index, pred_p_series)
    per_game = pd.read_parquet(ARTIFACT)[["game_id", "tue_open_home_spread"]]
    reg = reg.merge(per_game, on="game_id", how="left")
    reg["close_line"] = reg["spread_line"]
    reg["open_line"] = reg["tue_open_home_spread"]
    reg.loc[reg["season"] < 2020, "open_line"] = np.nan
    saved = reg[
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "margin",
            "close_line",
            "open_line",
            "R_H",
            "R_P",
        ]
    ]
    saved.to_parquet(OUT / f"per_game_ratings_{LABEL}.parquet", index=False)
    brng = np.random.default_rng(BOOTSTRAP_SEED)
    line_results = {}
    for rating in ("H", "P"):
        for line in ("close", "open"):
            line_results[f"{rating}_{line}"] = line_stats(reg, f"R_{rating}", f"{line}_line", brng)
    results = {
        "label": f"{LABEL}: H uses future games on purpose",
        "pregame_fallback_to_prior_season_games": int(fallback),
        "pregame_games_with_no_prediction": int(reg["R_P"].isna().sum()),
        "hindsight_removal_checks": checks_h,
        "truncation_check": trunc,
        "truncation_max_abs_diff": max(r["abs_diff"] for r in trunc),
        "open_sign_check_home_covered_agreement": sign_check(reg),
        "line_stats": line_results,
    }
    (OUT / f"results_lines_{LABEL}.json").write_text(
        json.dumps(results, indent=1, default=str), encoding="utf-8"
    )
    shared = {
        "league": "nfl",
        "classification": "unresolved_below_power",
        "source": f"artifacts/mod57_p2o/results_lines_{LABEL}.json",
        "category": "market",
        "notes": (
            "H is a positive control that uses future games of the same season, not servable. "
            "P uses only earlier games. Season-block bootstrap."
        ),
        "classification_evidence": (
            "Hindsight rating with the game removed; probability_positive reported; "
            "no refuted mechanism and no closing ground."
        ),
        "plain_summary": (
            "Measures how far the betting line sits from a hindsight rating of each team "
            "with the game itself left out."
        ),
    }
    (OUT / f"batch_lines_{LABEL}.json").write_text(
        json.dumps({**shared, "family": FAMILY_LINE, "cells": line_cells(results)}, indent=1),
        encoding="utf-8",
    )
    print("fallback", fallback, "no_pred_P", results["pregame_games_with_no_prediction"])
    print("hindsight checks", checks_h)
    print("truncation max abs diff", results["truncation_max_abs_diff"], len(trunc))
    for key, s in line_results.items():
        print(
            key,
            s["games"],
            f"|gap| mean {s['mean_abs_gap']:.2f} med {s['median_abs_gap']:.2f}",
            {k: round(v["mean"], 2) for k, v in s["gap_by_third"].items()},
        )
        print(
            f"  MAE rating {s['mae_rating']:.3f} line {s['mae_line']:.3f} "
            f"RMSE rating {s['rmse_rating']:.3f} line {s['rmse_line']:.3f}"
        )
        for stat in (
            "mae_diff_line_minus_rating",
            "mse_diff_line_minus_rating",
            "calibration_slope",
        ):
            v = s[stat]
            print(
                f"  {stat} {v['mean']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] "
                f"P+ {v['probability_positive']:.3f}"
            )
        pa = s["pick_accuracy"]
        print(
            "  pick",
            pa["overall"]["record"],
            f"{pa['overall']['accuracy']:.4f}",
            [round(x, 3) for x in pa["overall"]["ci95"]],
            f"P>.5 {pa['overall']['probability_above_half']:.3f}",
        )
        print(
            "  third",
            {k: (v["record"], round(v["accuracy"], 3)) for k, v in pa["by_third"].items()},
        )
        print(
            "  gap",
            {k: (v["record"], round(v["accuracy"], 3)) for k, v in pa["by_abs_gap"].items()},
        )
    section, cells = unit_section()
    (OUT / f"results_units_{LABEL}.json").write_text(
        json.dumps(section, indent=1, default=str), encoding="utf-8"
    )
    unit_shared = {
        **shared,
        "source": f"artifacts/mod57_p2o/results_units_{LABEL}.json",
        "category": "health",
        "sample_games": section["games"],
        "family": FAMILY_UNITS,
        "notes": (
            "positive control, leak, not servable: unit rates averaged over the team's other "
            "games of the season (own row removed); later rows are as-of smoothed and may still "
            "carry the game."
        ),
        "classification_evidence": (
            "Own-row-removed positive controls; probability_positive reported; "
            "no refuted mechanism and no closing ground."
        ),
    }
    (OUT / f"batch_units_{LABEL}.json").write_text(
        json.dumps({**unit_shared, "cells": cells}, indent=1), encoding="utf-8"
    )
    print("units missing", section["missing"])
    for pop, e in section["populations"].items():
        print(pop, e["games"])
        for t, r in e["records"].items():
            line = f"  {t} {r['record']} acc {r['accuracy']:.4f}"
            d = e["vs_served_base"].get(t)
            if d:
                a = d["accuracy_points"]
                line += (
                    f" | d {a['mean']:+.2f} [{a['ci95'][0]:+.2f},{a['ci95'][1]:+.2f}]"
                    f" P+ {a['probability_positive']:.3f}"
                    f" changed {d['new_wins_on_changed']}-{d['old_wins_on_changed']}"
                )
            print(line)


if __name__ == "__main__":
    main()
