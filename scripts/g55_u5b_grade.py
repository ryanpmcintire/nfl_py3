from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import g55_u3_servable_all as u3
from lead98_price_skew import metrics
from mod57_p2c_uncorrected_error_model import BOOTSTRAP_SEED, SCORE_SEASONS

ROOT = u3.ROOT
OUT = ROOT / "artifacts/g55_u5b"
V2 = ROOT / "artifacts/g55_u5a/expected_snaps_v2.parquet"
V2_MANIFEST = ROOT / "artifacts/g55_u5a/manifest.json"
U3_PER_GAME = ROOT / "artifacts/g55_u3/per_game.parquet"
U3_RESULTS = ROOT / "artifacts/g55_u3/results.json"
ARM = "S_ALL_v2"
PRIOR = "S_ALL"
DECLARED_PRIOR_RECORD = "809-694"


def configure():
    u3.OUT = OUT
    u3.SYNTH_TABLE = OUT / "servable_features.parquet"
    u3.NULL_DIR = OUT / "null"
    u3.EXPECTED = V2
    u3.EXPECTED_MANIFEST = V2_MANIFEST


def guard_v2():
    manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("realized_share_stored") is not False:
        raise SystemExit("v2 manifest does not certify that no realised share is stored")
    exp = pd.read_parquet(V2)
    inactive = ~exp["active"].fillna(False).astype(bool)
    if exp.loc[inactive, "expected_share"].abs().sum() != 0.0:
        raise SystemExit("inactive players carry a nonzero v2 forecast")
    actives = pd.read_parquet(u3.ACTIVES)
    known = actives.loc[actives["known_at_deadline"], ["game_id", "team"]].drop_duplicates()
    known_keys = set(zip(known["game_id"].astype(str), known["team"].astype(str), strict=True))
    v2_keys = set(zip(exp["game_id"].astype(str), exp["team"].astype(str), strict=True))
    if not v2_keys <= known_keys:
        raise SystemExit("v2 forecast rows on a team-game not known at the deadline")
    return {
        "rows": len(exp),
        "inactive_rows_nonzero": 0,
        "team_games": len(v2_keys),
        "realized_share_stored": manifest.get("realized_share_stored"),
    }


def rename(entry):
    if isinstance(entry, dict) and "S_ALL" in entry:
        entry[ARM] = entry.pop("S_ALL")
    return entry


def changed(p_new, p_old, y):
    pick_new = p_new >= 0.5
    pick_old = p_old >= 0.5
    mask = pick_new != pick_old
    new_wins = int((pick_new[mask] == (y[mask] == 1)).sum())
    old_wins = int((pick_old[mask] == (y[mask] == 1)).sum())
    games = int(mask.sum())
    return {
        "games": games,
        "arm_record": f"{new_wins}-{games - new_wins}",
        "other_record": f"{old_wins}-{games - old_wins}",
    }


def grade():
    configure()
    guard = guard_v2()
    u3.grade()
    result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    rows = pd.read_parquet(OUT / "per_game.parquet")
    prior = pd.read_parquet(U3_PER_GAME)[["game_id", "p_base", "p_arm"]].rename(
        columns={"p_base": "p_base_u3", "p_arm": "p_s_all"}
    )
    rows = rows.merge(prior, on="game_id", how="left", validate="one_to_one")
    per_game, y, pops = u3.grade_frame()
    if not per_game["game_id"].astype(str).equals(rows["game_id"].astype(str)):
        raise SystemExit("per-game order differs from the grade frame")
    p_arm = rows["p_arm"].to_numpy(float)
    p_base = rows["p_base"].to_numpy(float)
    p_prior = rows["p_s_all"].to_numpy(float)
    prior_check = {
        "s_all_record": metrics(p_prior, y)["record"],
        "declared": DECLARED_PRIOR_RECORD,
        "max_abs_diff_base_today_v_u3": float(np.nanmax(np.abs(p_base - rows["p_base_u3"]))),
        "missing_s_all": int(np.isnan(p_prior).sum()),
    }
    srng = np.random.default_rng(BOOTSTRAP_SEED)
    for pop, mask in pops.items():
        entry = result["populations"][pop]
        rename(entry["records"])
        rename(entry["per_season"])
        sub = per_game.loc[mask].reset_index(drop=True)
        entry["records"][PRIOR] = metrics(p_prior[mask], y[mask])
        entry["per_season"][PRIOR] = u3.per_season(sub, p_prior[mask])
        entry["vs_s_all_season_block"] = u3.season_paired(sub, p_arm[mask], p_prior[mask], srng)
        entry["changed_picks_vs_s_all"] = changed(p_arm[mask], p_prior[mask], y[mask])
    rename(result["missing"])
    rename(result["eval_seconds"])
    result["arm"] = ARM
    result["v2_guard"] = guard
    result["s_all_check"] = prior_check
    result["score_seasons"] = list(SCORE_SEASONS)
    rows.drop(columns=["p_base_u3"]).to_parquet(OUT / "per_game.parquet", index=False)
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"s_all_check": prior_check, "v2_guard": guard}, indent=1))


def null(start, draws):
    configure()
    u3.null(start, draws)


def summarize():
    result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    draws = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((OUT / "null").glob("draw_*.json"))
    ]
    null_summary = {"draws": len(draws), "populations": {}}
    for pop, entry in result["populations"].items():
        arm = entry["records"][ARM]
        null_wins = np.array([u3.wins(d["populations"][pop]["record"]) for d in draws])
        null_ll = np.array([d["populations"][pop]["log_loss"] for d in draws])
        arm_wins = u3.wins(arm["record"])
        null_summary["populations"][pop] = {
            "arm_wins": arm_wins,
            "null_wins_mean": float(null_wins.mean()),
            "null_wins_range": [int(null_wins.min()), int(null_wins.max())],
            "wins_percentile": float(
                100 * (np.mean(null_wins < arm_wins) + 0.5 * np.mean(null_wins == arm_wins))
            ),
            "arm_log_loss": arm["log_loss"],
            "null_log_loss_mean": float(null_ll.mean()),
            "log_loss_percentile": float(
                100
                * (np.mean(null_ll > arm["log_loss"]) + 0.5 * np.mean(null_ll == arm["log_loss"]))
            ),
        }
    allpop = result["populations"]["all"]
    nall = null_summary["populations"]["all"]
    tests = {
        "wins_at_least_declared": nall["arm_wins"] >= u3.DECLARED_SERVE_WINS,
        "null_wins_percentile_at_least_declared": nall["wins_percentile"]
        >= u3.DECLARED_SERVE_NULL_PCT,
        "log_loss_no_worse_than_base": allpop["records"][ARM]["log_loss"]
        <= allpop["records"]["served_base"]["log_loss"],
    }
    result["null"] = null_summary
    result["serving_candidate_tests"] = tests
    result["serving_candidate"] = all(tests.values())
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    cells = []
    for pop, entry in result["populations"].items():
        d = entry["vs_base_season_block"]["accuracy_points"]
        v = entry["vs_s_all_season_block"]["accuracy_points"]
        npop = null_summary["populations"][pop]
        cells.append(
            {
                "name": f"g55_u5b_s_all_v2_{pop}_accuracy_points",
                "description": (
                    "Servable deadline availability arm S_ALL_v2 (U1 gameday actives O1, U5a v2 "
                    "pregame one-score snap-share forecast O2, projected QB inactive O3, "
                    f"known-at-deadline games only) minus served base, {pop} games, model alone "
                    "at the Tuesday opener, season-block bootstrap; record "
                    f"{entry['records'][ARM]['record']} v base "
                    f"{entry['records']['served_base']['record']} v S_ALL "
                    f"{entry['records'][PRIOR]['record']} (v S_ALL {v['mean']:+.2f} pts, "
                    f"P+ {v['probability_positive']:.3f}); matched null wins percentile "
                    f"{npop['wins_percentile']:.1f} over {len(draws)} within-season shuffles"
                ),
                "effect": d["mean"],
                "effect_units": "accuracy_points",
                "interval_low": d["ci95"][0],
                "interval_high": d["ci95"][1],
                "probability_positive": d["probability_positive"],
                "sample_games": entry["games"],
                "season_start": 2020,
                "season_end": 2025,
            }
        )
    batch = {
        "league": "nfl",
        "classification": "unresolved_below_power",
        "family": u3.FAMILY,
        "source": "artifacts/g55_u5b/results.json",
        "category": "health",
        "sample_blocks": len(SCORE_SEASONS),
        "notes": (
            "G55-U5b declared look family (all primary, quiet, moved = 3 looks); declaration in "
            "docs/lanes/goal-55-backwards-2026-11-21.md section G55-U5b declaration; serving "
            f"candidate {'passed' if result['serving_candidate'] else 'failed'}"
        ),
        "classification_evidence": (
            "Declared grade; no refuted mechanism and no positive-control bound, so the result is "
            "recorded below power with probability_positive and the matched-null percentile."
        ),
        "plain_summary": (
            "Gives the model the gameday inactive list, a sharper pregame forecast of each "
            "active player's playing time in close games, and whether the expected quarterback "
            "is inactive, using only what is posted before picks lock."
        ),
        "cells": cells,
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print(json.dumps({"null": null_summary, "tests": tests}, indent=1))


def main():
    parser = argparse.ArgumentParser(
        description="G55-U5b grade of S_ALL_v2: the U3 harness with the U5a v2 usage forecast."
    )
    parser.add_argument("mode", choices=["build", "grade", "null", "summarize"])
    parser.add_argument("--start", type=int, default=0, help="first null draw index")
    parser.add_argument("--draws", type=int, default=1, help="number of null draws")
    args = parser.parse_args()
    if args.mode == "build":
        configure()
        guard_v2()
        u3.build()
    elif args.mode == "grade":
        grade()
    elif args.mode == "null":
        null(args.start, args.draws)
    else:
        summarize()


if __name__ == "__main__":
    main()
