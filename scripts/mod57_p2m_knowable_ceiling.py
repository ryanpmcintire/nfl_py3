from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lead98_price_skew import metrics
from mod57_p2c_uncorrected_error_model import (
    ARTIFACT,
    BOOTSTRAP_SEED,
    FULL_COLUMNS,
    QUIET,
    SERVED_FEATURES,
    paired,
    per_season,
)
from mod57_p2l_availability_ceiling import (
    MOVED,
    ORACLE_TABLE,
    PBP_SNAPSHOT,
    run_eval,
)

from nfl_ats.pbp import load_pbp_snapshot
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/mod57_p2m"
FAMILY = "mod57_availability_ceiling"
SHUFFLE_SEED = 20261111
LABEL = "LEAK_NOT_SERVABLE"
KNOW = ("diff_oracle_o1", "diff_oracle_o3")
RATE_PREFIXES = ("off_", "def_")
TURNOVER = "diff_turnover_margin_realised"


def unit_columns(served):
    names = []
    for column in FULL_COLUMNS:
        stem = column[5:]
        if (
            column.startswith("diff_")
            and stem.startswith(RATE_PREFIXES)
            and f"home_{stem}" in served.columns
            and f"away_{stem}" in served.columns
        ):
            names.append(stem)
    return names


def season_unit_ratings(served, stems):
    parts = []
    for side in ("home", "away"):
        part = served[["game_id", "season", f"{side}_team"]].rename(
            columns={f"{side}_team": "team"}
        )
        for stem in stems:
            part[stem] = served[f"{side}_{stem}"].to_numpy(float)
        parts.append(part)
    means = pd.concat(parts).groupby(["season", "team"])[stems].mean()
    home = means.reindex(pd.MultiIndex.from_arrays([served["season"], served["home_team"]]))
    away = means.reindex(pd.MultiIndex.from_arrays([served["season"], served["away_team"]]))
    out = pd.DataFrame(
        {f"diff_unitfin_{stem}": home[stem].to_numpy() - away[stem].to_numpy() for stem in stems}
    )
    out["game_id"] = served["game_id"].to_numpy()
    return out


def turnover_margin(served):
    pbp = load_pbp_snapshot(pbp_snapshot_from_root(PBP_SNAPSHOT))
    plays = pbp.loc[pbp["posteam"].notna()]
    give = (
        plays.assign(g=plays["interception"].fillna(0) + plays["fumble_lost"].fillna(0))
        .groupby(["game_id", "posteam"])["g"]
        .sum()
    )
    home = give.reindex(
        pd.MultiIndex.from_arrays([served["game_id"], served["home_team"]])
    ).to_numpy(float)
    away = give.reindex(
        pd.MultiIndex.from_arrays([served["game_id"], served["away_team"]])
    ).to_numpy(float)
    return pd.DataFrame({"game_id": served["game_id"].to_numpy(), TURNOVER: away - home})


def shuffle_within_season(features, columns, rng):
    names = []
    for column in columns:
        values = features[column].to_numpy().copy()
        for idx in features.groupby("season").indices.values():
            values[idx] = rng.permutation(values[idx])
        name = f"shuf_{column}"
        features[name] = values
        names.append(name)
    return tuple(names)


def season_coefficients(log, week_start):
    if len(log) != len(week_start):
        return f"{len(log)} fits for {len(week_start)} weeks"
    by_season = defaultdict(list)
    for entry_fit, (season, _) in zip(log, week_start.index, strict=True):
        if entry_fit["coef"]:
            by_season[int(season)].append(entry_fit["coef"])
    return {
        str(s): {k: float(np.mean([c[k] for c in v if k in c])) for k in v[0]}
        for s, v in by_season.items()
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    served = pd.read_parquet(SERVED_FEATURES)
    oracle = pd.read_parquet(ORACLE_TABLE)
    stems = unit_columns(served)
    units = season_unit_ratings(served, stems)
    unit_cols = tuple(c for c in units.columns if c != "game_id")
    features = (
        served.merge(oracle[["game_id", *KNOW]], on="game_id", how="left")
        .merge(units, on="game_id", how="left")
        .merge(turnover_margin(served), on="game_id", how="left")
    )
    for column in (*KNOW, *unit_cols, TURNOVER):
        features[column] = features[column].fillna(0.0)
    all_inputs = (*KNOW, *unit_cols)
    shuf = shuffle_within_season(features, all_inputs, np.random.default_rng(SHUFFLE_SEED))

    per_game = pd.read_parquet(ARTIFACT)
    per_game = per_game.loc[per_game["season"].between(2020, 2025)].reset_index(drop=True)
    ids = per_game["game_id"]
    y = per_game["home_covered"].to_numpy(float)
    move = np.abs(per_game["open_move"].to_numpy(float))
    pops = {"all": np.ones(len(per_game), bool), "quiet": move <= QUIET, "moved": move >= MOVED}
    model_alone = per_game["model_probability"].to_numpy(float)

    arms = {
        "served_base": (),
        "A_knowable_together": all_inputs,
        "B_actives_alone": ("diff_oracle_o1",),
        "B_qb_alone": ("diff_oracle_o3",),
        "B_unit_ratings_alone": unit_cols,
        "P_turnover_margin_power_check": (TURNOVER,),
        "SHUFFLED_A_control": shuf,
    }
    probs, fits = {}, {}
    for tag, extra in arms.items():
        scored, log = run_eval(features, extra, tag)
        probs[tag] = (
            scored.set_index("game_id")["home_cover_probability_at_open"]
            .reindex(ids)
            .to_numpy(float)
        )
        fits[tag] = log

    week_start = per_game.groupby(["season", "week"])["gameday"].min().sort_index()
    result = {
        "family": FAMILY,
        "label": f"{LABEL}: deliberate leaks used as positive controls, none servable",
        "games": len(per_game),
        "unit_rate_inputs": list(unit_cols),
        "base_matches_served_artifact_max_abs": float(
            np.nanmax(np.abs(probs["served_base"] - model_alone))
        ),
        "missing": {t: int(np.isnan(p).sum()) for t, p in probs.items()},
        "populations": {},
    }
    brng = np.random.default_rng(BOOTSTRAP_SEED)
    cells = []
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
            if tag == "SHUFFLED_A_control":
                continue
            for metric in ("accuracy_points", "log_loss_improvement", "brier_improvement"):
                cells.append(
                    {
                        "name": f"mod57_p2m_{tag}_{pop}_{metric}",
                        "description": (
                            f"POSITIVE CONTROL, DELIBERATE LEAK, NOT SERVABLE: base plus {tag} "
                            f"minus served base, {pop} games, opener grade, week-block bootstrap"
                        ),
                        "effect": d[metric]["mean"],
                        "effect_units": metric,
                        "interval_low": d[metric]["ci95"][0],
                        "interval_high": d[metric]["ci95"][1],
                        "probability_positive": d[metric]["probability_positive"],
                        "season_start": 2020,
                        "season_end": 2025,
                    }
                )
        result["populations"][pop] = entry
    coefs = {
        tag: season_coefficients(fits[tag], week_start)
        for tag in ("A_knowable_together", "P_turnover_margin_power_check")
    }
    result["season_mean_coefficients_scaled_units"] = coefs
    (OUT / "results_LEAK_NOT_SERVABLE.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8"
    )
    (OUT / "cells_LEAK_NOT_SERVABLE.json").write_text(
        json.dumps({"cells": cells}, indent=1), encoding="utf-8"
    )
    print("base_diff", result["base_matches_served_artifact_max_abs"], result["missing"])
    for pop, e in result["populations"].items():
        print(pop, e["games"])
        for t, r in e["records"].items():
            line = (
                f"  {t} {r['record']} acc {r['accuracy']:.4f} "
                f"ll {r['log_loss']:.4f} br {r['brier']:.4f}"
            )
            d = e["vs_served_base"].get(t)
            if d:
                a = d["accuracy_points"]
                line += (
                    f" | d {a['mean']:+.2f} [{a['ci95'][0]:+.2f},{a['ci95'][1]:+.2f}]"
                    f" P+ {a['probability_positive']:.3f}"
                    f" llP+ {d['log_loss_improvement']['probability_positive']:.3f}"
                    f" changed {d['new_wins_on_changed']}-{d['old_wins_on_changed']}"
                )
            print(line)
    for tag, c in coefs.items():
        if not isinstance(c, dict):
            print(tag, c)
            continue
        for s, v in c.items():
            lead = {k.replace("diff_", ""): round(x, 3) for k, x in v.items() if "unitfin" not in k}
            unit = [x for k, x in v.items() if "unitfin" in k]
            print(tag, s, lead, "unit_pos", sum(x > 0 for x in unit), "of", len(unit))


if __name__ == "__main__":
    main()
