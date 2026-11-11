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
    BASE_ALPHA,
    BASE_CONFIG,
    BOOTSTRAP_SEED,
    FULL_COLUMNS,
    MARKET_ROOT,
    QUIET,
    SCORE_SEASONS,
    SERVED_FEATURES,
    SET_NAME,
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
OUT = ROOT / "artifacts/mod57_p2l"
ORACLE_TABLE = OUT / "oracle_features_LEAK_NOT_SERVABLE.parquet"
BASE_FEATURES = ROOT / "data/processed/game_features_pbp.parquet"
PLAYER_SNAPSHOT = ROOT / "data/players/raw/20261108T141611Z"
PBP_SNAPSHOT = ROOT / "data/pbp/raw/20261111T010519Z"
VALUE_SNAPSHOT = ROOT / "data/players/values/raw/20260817T184911Z"
ROLE_SPAN = 8
DECISION_HOURS = 24
MOVED = 1.0
SHUFFLE_SEED = 20261111
FAMILY = "mod57_availability_ceiling"
ORACLES = ("diff_oracle_o1", "diff_oracle_o2", "diff_oracle_o3")
ARMS = {
    "O1_actives": ("diff_oracle_o1",),
    "O2_snap_share": ("diff_oracle_o2",),
    "O3_qb_dropbacks": ("diff_oracle_o3",),
    "ALL_three": ORACLES,
}
FIT_LOG: list[dict] = []


def role_rows(base, rosters, snaps_ids):
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
    role = defaultdict(dict)
    rows = []
    stats = {"team_games": 0, "no_roster": 0}
    for game in games.itertuples(index=False):
        kickoff = pd.Timestamp(game.kickoff)
        for team in (str(game.home_team), str(game.away_team)):
            stats["team_games"] += 1
            roster = roster_sets.get((int(game.season), int(game.week), team))
            grp = by_game.get((str(game.game_id), team))
            if roster is None:
                stats["no_roster"] += 1
            if roster is not None and grp is not None:
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
                for pid, state in role[team].items():
                    if pid not in roster:
                        continue
                    prior = float(state["offense_pct"]) + float(state["defense_pct"])
                    if prior <= 0.0:
                        continue
                    r = real.get(pid, (0.0, 0.0, 0.0))
                    sev2 = float(np.clip(1.0 - (r[0] + r[1]) / prior, 0.0, 1.0))
                    sev1 = 0.0 if real_any.get(pid, False) else 1.0
                    if sev1 > 0.0 or sev2 > 0.0:
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
    base = pd.read_parquet(BASE_FEATURES)
    _injuries_raw, rosters_raw, snaps_raw = load_player_snapshot(
        player_snapshot_from_root(PLAYER_SNAPSHOT)
    )
    rosters = canonicalize_rosters(rosters_raw)
    snaps_ids = attach_snap_player_ids(canonicalize_snaps(snaps_raw), rosters)
    inj, stats = role_rows(base, rosters, snaps_ids)
    pbp = load_pbp_snapshot(pbp_snapshot_from_root(PBP_SNAPSHOT))
    value_stats = load_player_value_snapshot(player_value_snapshot_from_root(VALUE_SNAPSHOT))
    out = base[["game_id"]].copy()
    summary = {"role_rows_stats": stats, "oracle_injury_rows": len(inj)}
    for tag, column in (("o1", "sev1"), ("o2", "sev2")):
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
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ORACLE_TABLE)
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
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
        config = {**BASE_CONFIG, "ridge_alpha": BASE_ALPHA, "mod57_p2l_arm": tag}
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    oracle = pd.read_parquet(ORACLE_TABLE) if ORACLE_TABLE.exists() else build_oracle()
    served = pd.read_parquet(SERVED_FEATURES)
    features = served.merge(oracle, on="game_id", how="left")
    for column in ORACLES:
        features[column] = features[column].fillna(0.0)
    rng = np.random.default_rng(SHUFFLE_SEED)
    for column in ORACLES:
        shuffled = features[column].to_numpy().copy()
        for idx in features.groupby("season").indices.values():
            shuffled[idx] = rng.permutation(shuffled[idx])
        features[column.replace("oracle", "shuf")] = shuffled
    shuf = tuple(c.replace("oracle", "shuf") for c in ORACLES)

    per_game = pd.read_parquet(ARTIFACT)
    per_game = per_game.loc[per_game["season"].between(2020, 2025)].reset_index(drop=True)
    ids = per_game["game_id"]
    y = per_game["home_covered"].to_numpy(float)
    move = np.abs(per_game["open_move"].to_numpy(float))
    pops = {"all": np.ones(len(per_game), bool), "quiet": move <= QUIET, "moved": move >= MOVED}
    model_alone = per_game["model_probability"].to_numpy(float)
    games_oracle = features.set_index("game_id").loc[ids, list(ORACLES)]

    probs, fits = {}, {}
    arms = {"served_base": ()} | ARMS | {"SHUFFLED_ALL_control": shuf}
    for tag, extra in arms.items():
        scored, log = run_eval(features, extra, tag)
        scored = scored.set_index("game_id")
        probs[tag] = scored["home_cover_probability_at_open"].reindex(ids).to_numpy(float)
        fits[tag] = log

    week_start = per_game.groupby(["season", "week"])["gameday"].min().sort_index()
    result = {
        "family": FAMILY,
        "label": (
            "LEAK: oracle inputs use each game's own realised data; "
            "positive control, deliberate leak, not servable"
        ),
        "games": len(per_game),
        "base_matches_served_artifact_max_abs": float(
            np.nanmax(np.abs(probs["served_base"] - model_alone))
        ),
        "missing": {t: int(np.isnan(p).sum()) for t, p in probs.items()},
        "nonzero_share": {c: float((games_oracle[c].abs() > 0).mean()) for c in ORACLES},
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
        for tag in [*ARMS, "SHUFFLED_ALL_control"]:
            d = paired(sub, probs[tag][mask], probs["served_base"][mask], brng)
            entry["vs_served_base"][tag] = d
            if tag in ARMS:
                for metric in ("accuracy_points", "log_loss_improvement", "brier_improvement"):
                    cells.append(
                        {
                            "name": f"mod57_p2l_{tag}_{pop}_{metric}",
                            "description": (
                                f"POSITIVE CONTROL, DELIBERATE LEAK, NOT SERVABLE: base plus {tag} "
                                f"oracle minus served base, {pop} games, opener grade, "
                                "week-block bootstrap"
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
    nz = {}
    base_hit = (probs["served_base"] >= 0.5) == (y == 1)
    for tag, cols in ARMS.items():
        nonzero = (games_oracle[list(cols)].abs() > 0).any(axis=1).to_numpy()
        hit = (probs[tag] >= 0.5) == (y == 1)
        nz[tag] = {
            "games": int(nonzero.sum()),
            "arm_record": f"{int(hit[nonzero].sum())}-{int((~hit[nonzero]).sum())}",
            "base_record": f"{int(base_hit[nonzero].sum())}-{int((~base_hit[nonzero]).sum())}",
        }
    result["nonzero_games_record"] = nz
    coefs = {}
    for tag, log in fits.items():
        if len(log) != len(week_start):
            coefs[tag] = f"{len(log)} fits for {len(week_start)} weeks"
            continue
        by_season = defaultdict(list)
        for entry_fit, (season, _) in zip(log, week_start.index, strict=True):
            if entry_fit["coef"]:
                by_season[int(season)].append(entry_fit["coef"])
        coefs[tag] = {
            str(s): {k: float(np.mean([c[k] for c in v if k in c])) for k in v[0]}
            for s, v in by_season.items()
        }
    result["season_mean_coefficients_scaled_units"] = coefs
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    (OUT / "cells.json").write_text(json.dumps({"cells": cells}, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
