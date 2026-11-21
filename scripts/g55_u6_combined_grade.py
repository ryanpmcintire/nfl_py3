from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import g55_u3_servable_all as u3
import g55_u5b_grade as u5b
import mod23_unit5 as m5
from lead98_price_skew import metrics
from mod57_p2c_uncorrected_error_model import (
    BOOTSTRAP_SEED,
    SCORE_SEASONS,
    SERVED_FEATURES,
    paired,
    per_season,
)

from nfl_ats.pbp import PbpSnapshot, load_pbp_snapshot, snapshot_from_root

ROOT = u3.ROOT
OUT = ROOT / "artifacts/g55_u6"
NULL_DIR = OUT / "null"
TERMS = OUT / "coverage_terms.parquet"
U5B_TABLE = ROOT / "artifacts/g55_u5b/servable_features.parquet"
U5B_PER_GAME = ROOT / "artifacts/g55_u5b/per_game.parquet"
FROZEN_TERMS = ROOT / "artifacts/mod23_unit5/terms.parquet"
FROZEN_PBP = ROOT / "data/pbp/raw/20260929T191306Z"
FROZEN_PARTICIPATION = "20260813T131635Z"
COV = ("home_cov_mz", "away_cov_mz", "diff_cov_mz")
ARM = "arm_C"
V2 = "S_ALL_v2"
COV_ALONE = "arm_a_man_zone"
FAMILY = "g55_combined"
DECLARED_V2_RECORD = "812-691"
DECLARED_COV_RECORD = "809-694"


def configure():
    u3.OUT = OUT
    u3.SYNTH_TABLE = U5B_TABLE
    u3.NULL_DIR = NULL_DIR


def matchup_terms_exact(state, classes):
    dn = np.column_stack([state[f"dn_{c}"].fillna(0.0).to_numpy() for c in classes])
    l_dn = np.column_stack([state[f"L_dn_{c}"].to_numpy() for c in classes])
    league_share = l_dn / l_dn.sum(axis=1, keepdims=True)
    known = dn.sum(axis=1, keepdims=True)
    rates = (dn + m5.SHRINK_PLAYS * league_share) / (known + m5.SHRINK_PLAYS)
    sums = np.column_stack([state[f"sum_{c}"].fillna(0.0).to_numpy() for c in classes])
    counts = np.column_stack([state[f"count_{c}"].fillna(0.0).to_numpy() for c in classes])
    l_sum = np.column_stack([state[f"L_sum_{c}"].to_numpy() for c in classes])
    l_cnt = np.column_stack([state[f"L_count_{c}"].to_numpy() for c in classes])
    league_epa = l_sum / l_cnt
    deviation = (sums - counts * league_epa) / (counts + m5.SHRINK_PLAYS)
    frame = pd.DataFrame({"game_id": state["game_id"].to_numpy(), "team": state["team"].to_numpy()})
    for j, cls in enumerate(classes):
        frame[f"rate_{cls}"] = rates[:, j]
        frame[f"dev_{cls}"] = deviation[:, j]
    return frame


def terms():
    features = pd.read_parquet(SERVED_FEATURES)
    schedules = pd.read_parquet(m5.u3.SNAPSHOT / "schedules.parquet")
    psnap = m5.latest_participation_snapshot(m5.PARTICIPATION_ROOT)
    if psnap.snapshot_id != FROZEN_PARTICIPATION:
        raise SystemExit(f"participation snapshot {psnap.snapshot_id} is not the frozen one")
    participation = m5.load_participation_snapshot(psnap)
    participation = participation.loc[participation["season"] >= m5.FIRST_DATA_SEASON]
    snap = snapshot_from_root(FROZEN_PBP)
    wanted = tuple(s for s in snap.seasons if s >= m5.FIRST_DATA_SEASON)
    pbp = load_pbp_snapshot(PbpSnapshot(snap.snapshot_id, snap.root, wanted))
    joined = m5.play_classes(participation, pbp, schedules)
    m5.matchup_terms = matchup_terms_exact
    built = m5.build_terms(features, joined)
    out = built[["game_id", *COV]].copy()
    frozen = pd.read_parquet(FROZEN_TERMS)[["game_id", *COV]]
    both = out.merge(frozen, on="game_id", how="outer", suffixes=("", "_frozen"), indicator=True)
    diffs = {
        c: float(np.nanmax(np.abs(both[c].to_numpy() - both[f"{c}_frozen"].to_numpy())))
        for c in COV
    }
    pre = features["season"].to_numpy() < m5.FIRST_DATA_SEASON
    summary = {
        "rows": len(out),
        "rows_only_one_side": int((both["_merge"] != "both").sum()),
        "max_abs_diff_vs_frozen_terms": diffs,
        "nonzero_before_first_data_season": float(out.loc[pre, list(COV)].abs().sum().sum()),
        "missing": int(out[list(COV)].isna().sum().sum()),
        "pbp_snapshot": snap.snapshot_id,
        "participation_snapshot": psnap.snapshot_id,
        "joined_plays": len(joined),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(TERMS, index=False)
    (OUT / "terms_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def load_features():
    features = u3.load_features()
    cov = pd.read_parquet(TERMS)
    features = features.merge(cov, on="game_id", how="left", validate="one_to_one")
    if features[list(COV)].isna().any().any():
        raise SystemExit("coverage term missing on a feature row")
    known = features["deadline_known"].to_numpy(bool)
    leaked = features.loc[~known, list(u3.COLUMNS)].abs().to_numpy().sum()
    if leaked != 0.0:
        raise SystemExit("nonzero servable term on a game not known at the deadline")
    pre = features["season"].to_numpy(int) < m5.FIRST_DATA_SEASON
    if features.loc[pre, list(COV)].abs().to_numpy().sum() != 0.0:
        raise SystemExit("coverage term nonzero before the first charted season")
    return features


def grade():
    configure()
    v2_guard = u5b.guard_v2()
    features = load_features()
    per_game, y, pops = u3.grade_frame()
    ids = per_game["game_id"]
    arms = {
        "served_base": (),
        V2: tuple(u3.COLUMNS),
        COV_ALONE: COV,
        ARM: (*u3.COLUMNS, *COV),
    }
    probs, logs, seconds = {}, {}, {}
    for tag, extra in arms.items():
        start = time.time()
        scored, log = u3.run_eval(features, extra, tag)
        seconds[tag] = time.time() - start
        probs[tag] = u3.probability(scored, ids)
        logs[tag] = log
        print(tag, metrics(probs[tag], y)["record"], flush=True)
        if tag == "served_base" and metrics(probs[tag], y)["record"] != u3.DECLARED_BASE_RECORD:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "results.json").write_text(
                json.dumps({"aborted": "base not reproduced"}, indent=1), encoding="utf-8"
            )
            raise SystemExit("base not reproduced")
    p_base, p_v2, p_cov, p_arm = (probs[k] for k in arms)
    u5b_rows = pd.read_parquet(U5B_PER_GAME).set_index("game_id").reindex(ids)
    checks = {
        "base_record": metrics(p_base, y)["record"],
        "base_declared": u3.DECLARED_BASE_RECORD,
        "s_all_v2_record": metrics(p_v2, y)["record"],
        "s_all_v2_declared": DECLARED_V2_RECORD,
        "s_all_v2_max_abs_diff_vs_u5b": float(
            np.nanmax(np.abs(p_v2 - u5b_rows["p_arm"].to_numpy(float)))
        ),
        "base_max_abs_diff_vs_u5b": float(
            np.nanmax(np.abs(p_base - u5b_rows["p_base"].to_numpy(float)))
        ),
        "arm_a_alone_record": metrics(p_cov, y)["record"],
        "arm_a_alone_declared": DECLARED_COV_RECORD,
        "missing": {k: int(np.isnan(v).sum()) for k, v in probs.items()},
    }
    week_start = per_game.groupby(["season", "week"])["gameday"].min().sort_index()
    terms_graded = features.set_index("game_id").loc[ids, [*u3.COLUMNS, *COV, "deadline_known"]]
    result = {
        "family": FAMILY,
        "label": "POST-SELECTION",
        "arm": ARM,
        "games": len(per_game),
        "score_seasons": list(SCORE_SEASONS),
        "checks": checks,
        "v2_guard": v2_guard,
        "terms_summary": json.loads((OUT / "terms_summary.json").read_text(encoding="utf-8")),
        "eval_seconds": seconds,
        "nonzero_share_graded": {
            c: float((terms_graded[c].abs() > 0).mean()) for c in (*u3.COLUMNS, *COV)
        },
        "season_mean_coefficients_scaled_units": u3.season_coefficients(logs[ARM], week_start),
        "populations": {},
    }
    wrng = np.random.default_rng(BOOTSTRAP_SEED)
    brng = np.random.default_rng(BOOTSTRAP_SEED)
    vrng = np.random.default_rng(BOOTSTRAP_SEED)
    for pop, mask in pops.items():
        sub = per_game.loc[mask].reset_index(drop=True)
        week_block = paired(sub, p_arm[mask], p_base[mask], wrng)
        result["populations"][pop] = {
            "games": int(mask.sum()),
            "records": {k: metrics(v[mask], y[mask]) for k, v in probs.items()},
            "per_season": {
                k: per_season(sub, probs[k][mask]) for k in ("served_base", V2, COV_ALONE, ARM)
            },
            "vs_base_season_block": u3.season_paired(sub, p_arm[mask], p_base[mask], brng),
            "vs_s_all_v2_season_block": u3.season_paired(sub, p_arm[mask], p_v2[mask], vrng),
            "vs_base_week_block": week_block,
            "changed_picks_vs_base": u5b.changed(p_arm[mask], p_base[mask], y[mask]),
            "changed_picks_vs_s_all_v2": u5b.changed(p_arm[mask], p_v2[mask], y[mask]),
        }
    rows = per_game[["game_id", "season", "week", "home_covered"]].copy()
    rows["p_base"] = p_base
    rows["p_s_all_v2"] = p_v2
    rows["p_arm"] = p_arm
    rows.to_parquet(OUT / "per_game.parquet", index=False)
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"checks": checks, "eval_seconds": seconds}, indent=1))


def null(start, draws):
    configure()
    features = load_features()
    per_game, y, pops = u3.grade_frame()
    ids = per_game["game_id"]
    known = features["deadline_known"].to_numpy(bool)
    seasons = features["season"].to_numpy(int)
    charted = seasons >= m5.FIRST_DATA_SEASON
    joint = [*u3.COLUMNS, *COV]
    cov_idx = [joint.index(c) for c in COV]
    values = features[joint].to_numpy(float)
    NULL_DIR.mkdir(parents=True, exist_ok=True)
    for k in range(start, start + draws):
        began = time.time()
        rng = np.random.default_rng([u3.NULL_SEED, k])
        shuffled = values.copy()
        for season in np.unique(seasons[charted]):
            idx = np.flatnonzero(known & (seasons == season))
            if len(idx):
                shuffled[idx] = values[rng.permutation(idx)]
            rest = np.flatnonzero(~known & charted & (seasons == season))
            if len(rest):
                shuffled[np.ix_(rest, cov_idx)] = values[np.ix_(rng.permutation(rest), cov_idx)]
        frame = features.copy()
        frame[joint] = shuffled
        scored, _ = u3.run_eval(frame, tuple(joint), f"null_{k}")
        p = u3.probability(scored, ids)
        entry = {
            "draw": k,
            "seed": [u3.NULL_SEED, k],
            "seconds": time.time() - began,
            "populations": {pop: metrics(p[m], y[m]) for pop, m in pops.items()},
        }
        (NULL_DIR / f"draw_{k:03d}.json").write_text(json.dumps(entry, indent=1), encoding="utf-8")
        print(json.dumps({"draw": k, "seconds": entry["seconds"], **entry["populations"]["all"]}))


def summarize():
    result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    draws = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(NULL_DIR.glob("draw_*.json"))
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
        v = entry["vs_s_all_v2_season_block"]["accuracy_points"]
        npop = null_summary["populations"][pop]
        cells.append(
            {
                "name": f"g55_u6_arm_c_{pop}_accuracy_points",
                "description": (
                    "POST-SELECTION combined arm C: S_ALL_v2 deadline availability inputs plus "
                    "MOD-23 unit 5a man/zone coverage matchup inputs in one walk-forward ridge, "
                    f"minus served base, {pop} games, model alone at the Tuesday opener, "
                    f"season-block bootstrap; record {entry['records'][ARM]['record']} v base "
                    f"{entry['records']['served_base']['record']} v S_ALL_v2 "
                    f"{entry['records'][V2]['record']} (v S_ALL_v2 {v['mean']:+.2f} pts, "
                    f"P+ {v['probability_positive']:.3f}); matched null wins percentile "
                    f"{npop['wins_percentile']:.1f} over {len(draws)} within-season joint shuffles"
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
        "family": FAMILY,
        "source": "artifacts/g55_u6/results.json",
        "category": "modeling",
        "sample_blocks": len(SCORE_SEASONS),
        "notes": (
            "POST-SELECTION: both pieces were chosen after their own grades on these games, so "
            "the history grade is optimistic and only the forward read decides. G55-U6 declared "
            "look family g55_combined (all primary, quiet, moved = 3 looks); declaration in "
            "docs/lanes/goal-55-backwards-2026-11-21.md section G55-U6 declaration; serving "
            f"candidate {'passed' if result['serving_candidate'] else 'failed'}"
        ),
        "classification_evidence": (
            "Declared grade; no refuted mechanism and no positive-control bound, so the result is "
            "recorded below power with probability_positive and the matched-null percentile."
        ),
        "plain_summary": (
            "Gives the model the gameday inactive list, a pregame forecast of each active "
            "player's playing time, whether the expected quarterback is inactive, and how each "
            "offense has fared against the man or zone coverage its opponent tends to play."
        ),
        "cells": cells,
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print(json.dumps({"null": null_summary, "tests": tests}, indent=1))


def main():
    parser = argparse.ArgumentParser(
        description="G55-U6 combined arm C: S_ALL_v2 plus MOD-23 unit 5a man/zone coverage inputs."
    )
    parser.add_argument("mode", choices=["terms", "grade", "null", "summarize"])
    parser.add_argument("--start", type=int, default=0, help="first null draw index")
    parser.add_argument("--draws", type=int, default=1, help="number of null draws")
    args = parser.parse_args()
    if args.mode == "terms":
        terms()
    elif args.mode == "grade":
        grade()
    elif args.mode == "null":
        null(args.start, args.draws)
    else:
        summarize()


if __name__ == "__main__":
    main()
