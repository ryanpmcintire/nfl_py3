from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lead98_price_skew import metrics
from mod57_p2c_uncorrected_error_model import (
    ARTIFACT,
    BASE_ALPHA,
    BASE_CONFIG,
    BOOTSTRAP_SEED,
    FULL_COLUMNS,
    MARKET_ROOT,
    SCORE_SEASONS,
    SERVED_FEATURES,
    SET_NAME,
    boot,
    paired,
    per_season,
)

from nfl_ats import clv, players
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.pbp import load_pbp_snapshot
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root
from nfl_ats.players import (
    _REPLACEMENT_QB_EPA,
    _compile_lineup,
    _position_group,
    attach_snap_player_ids,
    canonicalize_rosters,
    canonicalize_snaps,
    enrich_with_player_features,
    load_player_snapshot,
    load_player_value_snapshot,
    player_snapshot_from_root,
    player_value_snapshot_from_root,
)
from nfl_ats.quarterbacks import build_qb_game_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/g55_u4"
ORACLE_TABLE = OUT / "oracle_features_neutral_LEAK_NOT_SERVABLE.parquet"
P2L_TABLE = ROOT / "artifacts/mod57_p2l/oracle_features_LEAK_NOT_SERVABLE.parquet"
S_ALL_GAMES = ROOT / "artifacts/g55_u3/per_game.parquet"
BASE_FEATURES = ROOT / "data/processed/game_features_pbp.parquet"
PLAYER_SNAPSHOT = ROOT / "data/players/raw/20261108T141611Z"
PBP_SNAPSHOT = ROOT / "data/pbp/raw/20261111T010519Z"
VALUE_SNAPSHOT = ROOT / "data/players/values/raw/20260817T184911Z"
PARTICIPATION = ROOT / "data/players/participation/raw/20260813T131635Z"
SCRIMMAGE = ("pass", "run", "qb_kneel", "qb_spike")
TOUCHDOWN = 6
TWO_POINT_CONVERSION = 2
ONE_SCORE = TOUCHDOWN + TWO_POINT_CONVERSION
ROLE_SPAN = 8
DECISION_HOURS = 24
FAMILY = "mod57_availability_ceiling"
DECLARED_BASE_RECORD = "802-701"
ORACLES = ("diff_oracle_o1", "diff_oracle_o2", "diff_oracle_o2n", "diff_oracle_o3")
ARMS = {
    "ALL_rerun": ("diff_oracle_o1", "diff_oracle_o2", "diff_oracle_o3"),
    "ALL_neutral": ("diff_oracle_o1", "diff_oracle_o2n", "diff_oracle_o3"),
    "O2_neutral": ("diff_oracle_o2n",),
}
RECORDED = ("ALL_neutral", "O2_neutral")
FIT_LOG: list[dict] = []


def participation_shares(pbp):
    plays = pbp.loc[
        pbp["play_type"].isin(SCRIMMAGE) & pbp["posteam"].notna() & pbp["defteam"].notna(),
        ["game_id", "play_id", "posteam", "defteam", "score_differential"],
    ].copy()
    plays["play_id"] = pd.to_numeric(plays["play_id"], errors="coerce")
    plays = plays.dropna(subset=["play_id", "score_differential"])
    plays["play_id"] = plays["play_id"].astype("int64")
    plays["neutral"] = plays["score_differential"].abs() <= ONE_SCORE
    manifest = json.loads((PARTICIPATION / "manifest.json").read_text(encoding="utf-8"))
    frames = [
        pd.read_parquet(
            PARTICIPATION / part["path"].replace("\\", "/"),
            columns=["game_id", "play_id", "offense_players", "defense_players"],
        )
        for part in manifest["partitions"]
    ]
    part = pd.concat(frames, ignore_index=True)
    part["play_id"] = pd.to_numeric(part["play_id"], errors="coerce")
    part = part.dropna(subset=["play_id"])
    part["play_id"] = part["play_id"].astype("int64")
    part = part.loc[
        part["offense_players"].fillna("").str.len().gt(0)
        & part["defense_players"].fillna("").str.len().gt(0)
    ]
    merged = plays.merge(part, on=["game_id", "play_id"], how="inner")
    summary = {
        "scrimmage_plays": len(plays),
        "participation_plays": len(part),
        "joined_plays": len(merged),
        "neutral_share_of_joined": float(merged["neutral"].mean()),
    }
    stores = {}
    for scope in ("neutral", "full"):
        sub = merged if scope == "full" else merged.loc[merged["neutral"]]
        sides = []
        keys = []
        for side, team_col, col in (
            ("offense", "posteam", "offense_players"),
            ("defense", "defteam", "defense_players"),
        ):
            frame = sub[["game_id", team_col, col]].rename(columns={team_col: "team"})
            totals = frame.groupby(["game_id", "team"]).size().rename("plays").reset_index()
            keys.append(set(zip(totals["game_id"], totals["team"], strict=True)))
            frame = frame.assign(gsis_id=frame[col].str.split(";")).explode("gsis_id")
            frame = frame.loc[frame["gsis_id"].fillna("").str.len().gt(0)]
            counts = frame.groupby(["game_id", "team", "gsis_id"]).size().rename("on")
            counts = counts.reset_index().merge(totals, on=["game_id", "team"])
            counts[side] = counts["on"] / counts["plays"]
            sides.append(counts[["game_id", "team", "gsis_id", side]])
        shares = sides[0].merge(sides[1], on=["game_id", "team", "gsis_id"], how="outer")
        shares[["offense", "defense"]] = shares[["offense", "defense"]].fillna(0.0)
        both = keys[0] & keys[1]
        store = {}
        for (game_id, team), grp in shares.groupby(["game_id", "team"], sort=False):
            if (game_id, team) in both:
                store[(str(game_id), str(team))] = dict(
                    zip(
                        grp["gsis_id"],
                        zip(grp["offense"], grp["defense"], strict=True),
                        strict=True,
                    )
                )
        stores[scope] = store
        summary[f"{scope}_team_games"] = len(store)
    return stores, summary


def role_rows(base, rosters, snaps_ids, stores):
    alpha = 2.0 / (ROLE_SPAN + 1.0)
    games = base.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    snaps_ids = snaps_ids.copy()
    snaps_ids["player_key"] = snaps_ids["gsis_id"].fillna(
        "pfr:" + snaps_ids["pfr_player_id"].astype(str)
    )
    snaps_ids["position_group"] = snaps_ids["position"].map(_position_group)
    by_game = {
        (str(g), str(t)): grp for (g, t), grp in snaps_ids.groupby(["game_id", "team"], sort=False)
    }
    roster_sets = {
        (int(s), int(w), str(t)): grp.set_index("gsis_id")["position"].to_dict()
        for (s, w, t), grp in rosters.dropna(subset=["gsis_id"]).groupby(
            ["season", "week", "team"], sort=False
        )
    }
    neutral, full = stores["neutral"], stores["full"]
    role = defaultdict(dict)
    rows = []
    agreement = []
    stats = {
        "team_games": 0,
        "no_roster": 0,
        "scored_team_games": 0,
        "scored_team_games_no_neutral": 0,
        "scored_team_games_no_neutral_2020_25": 0,
    }
    for game in games.itertuples(index=False):
        kickoff = pd.Timestamp(game.kickoff)
        for team in (str(game.home_team), str(game.away_team)):
            stats["team_games"] += 1
            key = (str(game.game_id), team)
            roster = roster_sets.get((int(game.season), int(game.week), team))
            grp = by_game.get(key)
            if roster is None:
                stats["no_roster"] += 1
            if roster is not None and grp is not None:
                stats["scored_team_games"] += 1
                nshare = neutral.get(key)
                if nshare is None:
                    stats["scored_team_games_no_neutral"] += 1
                    if int(game.season) in SCORE_SEASONS:
                        stats["scored_team_games_no_neutral_2020_25"] += 1
                real = {
                    str(r.gsis_id): (float(r.offense_pct), float(r.defense_pct), float(r.st_pct))
                    for r in grp.itertuples(index=False)
                    if pd.notna(r.gsis_id)
                }
                real_any = {
                    str(r.gsis_id): bool(r.offense_snaps + r.defense_snaps + r.st_snaps > 0)
                    for r in grp.itertuples(index=False)
                    if pd.notna(r.gsis_id)
                }
                fshare = full.get(key)
                if fshare is not None:
                    for pid, r in real.items():
                        pfr = r[0] + r[1]
                        if pfr > 0.0:
                            f = fshare.get(pid, (0.0, 0.0))
                            agreement.append((int(game.season), pfr, f[0] + f[1]))
                for pid, state in role[team].items():
                    if pid not in roster:
                        continue
                    prior = float(state["offense_pct"]) + float(state["defense_pct"])
                    if prior <= 0.0:
                        continue
                    r = real.get(pid, (0.0, 0.0, 0.0))
                    sev2 = float(np.clip(1.0 - (r[0] + r[1]) / prior, 0.0, 1.0))
                    sev1 = 0.0 if real_any.get(pid, False) else 1.0
                    if nshare is None:
                        sev2n = 0.0
                    else:
                        n = nshare.get(pid, (0.0, 0.0))
                        sev2n = float(np.clip(1.0 - (n[0] + n[1]) / prior, 0.0, 1.0))
                    if sev1 > 0.0 or sev2 > 0.0 or sev2n > 0.0:
                        rows.append(
                            {
                                "season": int(game.season),
                                "week": int(game.week),
                                "game_type": "REG",
                                "team": team,
                                "gsis_id": pid,
                                "position": roster.get(pid),
                                "report_status": "Out",
                                "practice_status": "Did Not Participate In Practice",
                                "effective_observed_at": kickoff
                                - pd.Timedelta(hours=DECISION_HOURS + 24),
                                "date_modified": kickoff - pd.Timedelta(hours=DECISION_HOURS + 24),
                                "sev1": sev1,
                                "sev2": sev2,
                                "sev2n": sev2n,
                            }
                        )
            if grp is not None:
                lineup = _compile_lineup(grp)
                for gsis_id, offense, defense, special, _group in lineup.roles:
                    if gsis_id is None:
                        continue
                    state = role[team].setdefault(gsis_id, {})
                    for column, value in (
                        ("offense_pct", offense),
                        ("defense_pct", defense),
                        ("st_pct", special),
                    ):
                        previous = state.get(column)
                        state[column] = (
                            value if previous is None else alpha * value + (1.0 - alpha) * previous
                        )
    agree = pd.DataFrame(agreement, columns=["season", "pfr", "participation"])
    stats["source_agreement"] = {
        "player_games": len(agree),
        "correlation": float(agree["pfr"].corr(agree["participation"])),
        "median_ratio_participation_to_pfr": float(
            (agree["participation"] / agree["pfr"]).median()
        ),
        "median_ratio_by_season": {
            str(s): float((g["participation"] / g["pfr"]).median())
            for s, g in agree.groupby("season")
        },
    }
    return pd.DataFrame(rows), stats


def run_enrich(base, injuries, rosters_raw, snaps_raw, pbp, stats_raw, severity_column):
    frame = injuries.loc[injuries[severity_column] > 0].copy()
    frame["oracle_severity"] = frame[severity_column]
    original_canon = players.canonicalize_injuries
    original_sev = players._injury_unavailability
    players.canonicalize_injuries = lambda inj, **kwargs: inj
    players._injury_unavailability = lambda row: float(row["oracle_severity"])
    try:
        enriched = enrich_with_player_features(
            base,
            frame,
            rosters_raw,
            snaps_raw,
            pbp,
            player_stats=stats_raw,
            decision_hours_before_kickoff=DECISION_HOURS,
        )
    finally:
        players.canonicalize_injuries = original_canon
        players._injury_unavailability = original_sev
    return enriched


def build_oracle():
    started = time.time()
    base = pd.read_parquet(BASE_FEATURES)
    _injuries_raw, rosters_raw, snaps_raw = load_player_snapshot(
        player_snapshot_from_root(PLAYER_SNAPSHOT)
    )
    rosters = canonicalize_rosters(rosters_raw)
    snaps_ids = attach_snap_player_ids(canonicalize_snaps(snaps_raw), rosters)
    pbp = load_pbp_snapshot(pbp_snapshot_from_root(PBP_SNAPSHOT))
    stores, part_summary = participation_shares(pbp)
    inj, stats = role_rows(base, rosters, snaps_ids, stores)
    value_stats = load_player_value_snapshot(player_value_snapshot_from_root(VALUE_SNAPSHOT))
    out = base[["game_id"]].copy()
    summary = {
        "participation": part_summary,
        "role_rows_stats": stats,
        "oracle_injury_rows": len(inj),
        "rows_by_severity": {c: int((inj[c] > 0).sum()) for c in ("sev1", "sev2", "sev2n")},
    }
    for tag, column in (("o1", "sev1"), ("o2", "sev2"), ("o2n", "sev2n")):
        enriched = run_enrich(base, inj, rosters_raw, snaps_raw, pbp, value_stats, column)
        part = enriched[
            [
                "game_id",
                "diff_injury_skill_epa_value_lost",
                "diff_injury_defense_disruption_value_lost",
            ]
        ].copy()
        part[f"diff_oracle_{tag}"] = part["diff_injury_skill_epa_value_lost"].fillna(0.0) + part[
            "diff_injury_defense_disruption_value_lost"
        ].fillna(0.0)
        out = out.merge(part[["game_id", f"diff_oracle_{tag}"]], on="game_id", how="left")
    served = pd.read_parquet(SERVED_FEATURES)
    qb = build_qb_game_metrics(pbp)
    team_col = "team" if "team" in qb.columns else "posteam"
    totals = qb.groupby(["game_id", team_col])["qb_dropbacks"].sum().rename("total")
    per_qb = qb.groupby(["game_id", team_col, "player_id"])["qb_dropbacks"].sum()
    parts = {}
    for side in ("home", "away"):
        keys = pd.MultiIndex.from_arrays(
            [
                served["game_id"],
                served[f"{side}_team"],
                served[f"{side}_projected_qb_id"].astype(str),
            ]
        )
        got = per_qb.reindex(keys).to_numpy(float)
        tot = totals.reindex(
            pd.MultiIndex.from_arrays([served["game_id"], served[f"{side}_team"]])
        ).to_numpy(float)
        share = np.where(np.isfinite(tot) & (tot > 0), np.nan_to_num(got, nan=0.0) / tot, np.nan)
        gap = served[f"{side}_qb_starter_epa_per_dropback"].to_numpy(float) - _REPLACEMENT_QB_EPA
        parts[side] = np.nan_to_num((1.0 - share) * gap, nan=0.0)
    o3 = pd.DataFrame(
        {"game_id": served["game_id"], "diff_oracle_o3": parts["home"] - parts["away"]}
    )
    out = out.merge(o3, on="game_id", how="left")
    cached = pd.read_parquet(P2L_TABLE).set_index("game_id")
    mine = out.drop_duplicates("game_id").set_index("game_id")
    common = mine.index.intersection(cached.index)
    summary["max_abs_diff_to_p2l_cache"] = {
        c: float(
            np.nanmax(np.abs(mine.loc[common, c].fillna(0.0) - cached.loc[common, c].fillna(0.0)))
        )
        for c in ("diff_oracle_o1", "diff_oracle_o2", "diff_oracle_o3")
    }
    summary["o2_o2n_correlation"] = float(out["diff_oracle_o2"].corr(out["diff_oracle_o2n"]))
    summary["build_seconds"] = time.time() - started
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ORACLE_TABLE)
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return out


def make_fit(original, columns):
    def wrapped(frame, **kwargs):
        model = original(frame, **kwargs)
        entry = {"max_gameday": str(pd.to_datetime(frame["gameday"]).max().date()), "coef": None}
        try:
            est = model.estimator
            names = list(est[:-1].get_feature_names_out())
            coef = est[-1].coef_
            entry["coef"] = {c: float(coef[names.index(c)]) for c in columns if c in names}
        except Exception:
            entry["coef"] = None
        FIT_LOG.append(entry)
        return model

    return wrapped


def run_eval(features, extra, tag):
    original_fit = clv.fit_margin_model
    original_set = FEATURE_SETS[SET_NAME]
    FEATURE_SETS[SET_NAME] = FULL_COLUMNS + tuple(extra)
    clv.fit_margin_model = make_fit(original_fit, extra)
    del FIT_LOG[:]
    try:
        config = {**BASE_CONFIG, "ridge_alpha": BASE_ALPHA, "g55_u4_arm": tag}
        scored = clv.opener_pick_evaluation(
            MARKET_ROOT,
            features,
            active_model_config=config,
            min_train_games=DEFAULT_MIN_TRAIN_GAMES,
        )
    finally:
        clv.fit_margin_model = original_fit
        FEATURE_SETS[SET_NAME] = original_set
    scored = scored.loc[scored["season"].isin(SCORE_SEASONS)].copy()
    return scored, list(FIT_LOG)


def season_paired(frame, p_new, p_old, rng):
    y = frame["home_covered"].to_numpy(float)
    pn = np.clip(p_new, 1e-9, 1 - 1e-9)
    po = np.clip(p_old, 1e-9, 1 - 1e-9)
    hit_new = ((pn >= 0.5) == (y == 1)).astype(float)
    hit_old = ((po >= 0.5) == (y == 1)).astype(float)
    ll_new = -(y * np.log(pn) + (1 - y) * np.log1p(-pn))
    ll_old = -(y * np.log(po) + (1 - y) * np.log1p(-po))
    block = frame.groupby("season").ngroup().to_numpy()
    acc = boot(hit_new - hit_old, block, rng)
    acc = {**acc, "mean": acc["mean"] * 100, "ci95": [v * 100 for v in acc["ci95"]]}
    return {
        "accuracy_points": acc,
        "log_loss_improvement": boot(ll_old - ll_new, block, rng),
        "brier_improvement": boot((po - y) ** 2 - (pn - y) ** 2, block, rng),
    }


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


def wins(record):
    return int(record.split("-")[0])


def grade():
    oracle = pd.read_parquet(ORACLE_TABLE).drop_duplicates("game_id")
    served = pd.read_parquet(SERVED_FEATURES)
    features = served.merge(oracle, on="game_id", how="left")
    for column in ORACLES:
        features[column] = features[column].fillna(0.0)
    per_game = pd.read_parquet(ARTIFACT)
    per_game = per_game.loc[per_game["season"].isin(SCORE_SEASONS)].reset_index(drop=True)
    ids = per_game["game_id"]
    y = per_game["home_covered"].to_numpy(float)
    probs, fits, seconds = {}, {}, {}
    for tag, extra in ({"served_base": ()} | ARMS).items():
        start = time.time()
        scored, log = run_eval(features, extra, tag)
        seconds[tag] = time.time() - start
        probs[tag] = (
            scored.set_index("game_id")["home_cover_probability_at_open"]
            .reindex(ids)
            .to_numpy(float)
        )
        fits[tag] = log
        if tag == "served_base":
            got = metrics(probs[tag], y)["record"]
            if got != DECLARED_BASE_RECORD:
                raise SystemExit(f"base not reproduced: {got}")
    s_all = pd.read_parquet(S_ALL_GAMES).set_index("game_id")
    probs["S_ALL"] = s_all["p_arm"].reindex(ids).to_numpy(float)
    week_start = per_game.groupby(["season", "week"])["gameday"].min().sort_index()
    games_oracle = features.drop_duplicates("game_id").set_index("game_id").loc[ids, list(ORACLES)]
    records = {tag: metrics(p, y) for tag, p in probs.items()}
    srng = np.random.default_rng(BOOTSTRAP_SEED)
    wrng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = {}
    for new, old in (
        ("ALL_rerun", "served_base"),
        ("ALL_neutral", "served_base"),
        ("O2_neutral", "served_base"),
        ("ALL_neutral", "ALL_rerun"),
        ("ALL_neutral", "S_ALL"),
        ("ALL_rerun", "S_ALL"),
        ("S_ALL", "served_base"),
    ):
        week_block = paired(per_game, probs[new], probs[old], wrng)
        comparisons[f"{new}_v_{old}"] = {
            "season_block": season_paired(per_game, probs[new], probs[old], srng),
            "week_block_accuracy_points": week_block["accuracy_points"],
            "changed_picks": week_block["changed_pick_games"],
            "new_wins_on_changed": week_block["new_wins_on_changed"],
            "old_wins_on_changed": week_block["old_wins_on_changed"],
        }
    w = {tag: wins(r["record"]) for tag, r in records.items()}
    gap = w["ALL_rerun"] - w["S_ALL"]
    result = {
        "family": FAMILY,
        "label": (
            "LEAK: oracle inputs use each game's own realised data; "
            "positive control, deliberate leak, not servable"
        ),
        "games": len(per_game),
        "eval_seconds": seconds,
        "missing": {t: int(np.isnan(p).sum()) for t, p in probs.items()},
        "base_max_abs_diff_to_g55_u3_base": float(
            np.nanmax(np.abs(probs["served_base"] - s_all["p_base"].reindex(ids).to_numpy(float)))
        ),
        "nonzero_share_graded": {c: float((games_oracle[c].abs() > 0).mean()) for c in ORACLES},
        "records": records,
        "per_season": {tag: per_season(per_game, p) for tag, p in probs.items()},
        "comparisons": comparisons,
        "gap_wins_all_rerun_minus_s_all": gap,
        "gap_wins_all_neutral_minus_s_all": w["ALL_neutral"] - w["S_ALL"],
        "gap_share_surviving_in_all_neutral": (
            (w["ALL_neutral"] - w["S_ALL"]) / gap if gap != 0 else None
        ),
        "season_mean_coefficients_scaled_units": {
            tag: season_coefficients(log, week_start) for tag, log in fits.items() if tag in ARMS
        },
    }
    rows = per_game[["game_id", "season", "week", "home_covered", "open_move"]].copy()
    for tag, p in probs.items():
        rows[f"p_{tag}"] = p
    for column in ORACLES:
        rows[column] = games_oracle[column].to_numpy()
    rows.to_parquet(OUT / "per_game.parquet", index=False)
    cells = []
    for tag in RECORDED:
        d = comparisons[f"{tag}_v_served_base"]["season_block"]["accuracy_points"]
        cells.append(
            {
                "name": f"g55_u4_{tag}_all_accuracy_points",
                "description": (
                    f"POSITIVE CONTROL, DELIBERATE LEAK, NOT SERVABLE: base plus {tag} "
                    "(realised snap share counted only while the score is within 8, one "
                    "possession) minus served base, all games, model alone at the Tuesday opener, "
                    f"season-block bootstrap; record {records[tag]['record']} v base "
                    f"{records['served_base']['record']}; full-game ALL rerun "
                    f"{records['ALL_rerun']['record']}"
                ),
                "effect": d["mean"],
                "effect_units": "accuracy_points",
                "interval_low": d["ci95"][0],
                "interval_high": d["ci95"][1],
                "probability_positive": d["probability_positive"],
                "sample_games": len(per_game),
                "season_start": 2020,
                "season_end": 2025,
            }
        )
    batch = {
        "league": "nfl",
        "classification": "unresolved_below_power",
        "family": FAMILY,
        "source": "artifacts/g55_u4/results.json",
        "category": "health",
        "sample_blocks": len(SCORE_SEASONS),
        "notes": (
            "G55-U4 declared looks (ALL_neutral and O2_neutral, all games = 2); declaration in "
            "docs/lanes/g55-u4-neutral-usage-ceiling-2026-11-21.md; ALL rerun is a reference"
        ),
        "classification_evidence": (
            "Deliberate-leak ceiling diagnostic; no refuted mechanism and no positive-control "
            "bound, so recorded below power with probability_positive."
        ),
        "plain_summary": (
            "Tells the model after the fact how much each player played while the game was "
            "still within one score, to see how much of the perfect who-plays ceiling is the "
            "final score leaking through playing time."
        ),
        "cells": cells,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "per_season"}, indent=1, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="G55-U4 one-score usage ceiling on the P2-L opener harness (deliberate leak)."
    )
    parser.add_argument("mode", choices=["build", "grade"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "build":
        build_oracle()
    else:
        grade()


if __name__ == "__main__":
    main()
