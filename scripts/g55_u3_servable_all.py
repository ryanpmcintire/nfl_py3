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
    QUIET,
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/g55_u3"
SYNTH_TABLE = OUT / "servable_features.parquet"
NULL_DIR = OUT / "null"
ACTIVES = ROOT / "artifacts/g55_u1/deadline_actives.parquet"
EXPECTED = OUT / "u2_rebuilt/expected_snaps.parquet"
EXPECTED_MANIFEST = OUT / "u2_rebuilt/manifest.json"
BASE_FEATURES = ROOT / "data/processed/game_features_pbp.parquet"
PLAYER_SNAPSHOT = ROOT / "data/players/raw/20261108T141611Z"
PBP_SNAPSHOT = ROOT / "data/pbp/raw/20261111T010519Z"
VALUE_SNAPSHOT = ROOT / "data/players/values/raw/20260817T184911Z"
ROLE_SPAN = 8
DECISION_HOURS = 24
MOVED = 1.0
NULL_SEED = 20261121
FAMILY = "g55_deadline_availability"
INACTIVE_STATUS = "INA"
COLUMNS = ("diff_servable_o1", "diff_servable_o2", "diff_servable_o3")
DECLARED_BASE_RECORD = "802-701"
DECLARED_SERVE_WINS = 827
DECLARED_SERVE_NULL_PCT = 95.0
FIT_LOG: list[dict] = []


def deadline_lists(table):
    known = table.loc[table["known_at_deadline"]]
    lists = {}
    for (game_id, team), grp in known.groupby(["game_id", "team"], sort=False):
        lists[(str(game_id), str(team))] = {
            str(p): (None if pd.isna(pos) else str(pos), bool(a))
            for p, pos, a in zip(grp["gsis_id"], grp["position"], grp["active"], strict=True)
        }
    return lists


def role_rows(base, lists, expected, snaps_ids):
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
    role = defaultdict(dict)
    rows = []
    keys = []
    stats = {"team_games": 0, "deadline_list": 0, "no_deadline_list": 0}
    for game in games.itertuples(index=False):
        kickoff = pd.Timestamp(game.kickoff)
        game_id = str(game.game_id)
        for team in (str(game.home_team), str(game.away_team)):
            stats["team_games"] += 1
            listed = lists.get((game_id, team))
            if listed is None:
                stats["no_deadline_list"] += 1
            else:
                stats["deadline_list"] += 1
                for pid, state in role[team].items():
                    entry = listed.get(pid)
                    if entry is None:
                        continue
                    prior = float(state["offense_pct"]) + float(state["defense_pct"])
                    if prior <= 0.0:
                        continue
                    position, active = entry
                    forecast = (game_id, team, pid) in expected
                    if active and not forecast:
                        tag = f"active_no_u2_{int(game.season)}"
                        stats[tag] = stats.get(tag, 0) + 1
                    share = float(expected.get((game_id, team, pid), 0.0))
                    sev2 = float(np.clip(1.0 - share / prior, 0.0, 1.0))
                    if active and not forecast:
                        sev2 = 0.0
                    sev1 = 0.0 if active else 1.0
                    if sev1 > 0.0 or sev2 > 0.0:
                        keys.append((game_id, team))
                        rows.append(
                            {
                                "season": int(game.season),
                                "week": int(game.week),
                                "game_type": "REG",
                                "team": team,
                                "gsis_id": pid,
                                "position": position,
                                "report_status": "Out",
                                "practice_status": "Did Not Participate In Practice",
                                "effective_observed_at": kickoff
                                - pd.Timedelta(hours=DECISION_HOURS + 24),
                                "date_modified": kickoff - pd.Timedelta(hours=DECISION_HOURS + 24),
                                "sev1": sev1,
                                "sev2": sev2,
                            }
                        )
            grp = by_game.get((game_id, team))
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
    key_frame = pd.DataFrame(keys, columns=["game_id", "team"])
    return pd.DataFrame(rows), key_frame, stats


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


def qb_term(served, table):
    known = table.loc[table["known_at_deadline"]].drop_duplicates(["game_id", "team", "gsis_id"])
    status = known.set_index(["game_id", "team", "gsis_id"])["status"].astype(object)
    known_team_games = set(zip(known["game_id"], known["team"], strict=True))
    parts, counts = {}, []
    for side in ("home", "away"):
        team = served[f"{side}_team"].astype(str)
        qb = served[f"{side}_projected_qb_id"].astype(str)
        got = pd.Series(status.reindex(pd.MultiIndex.from_arrays([served["game_id"], team, qb])))
        got = got.reset_index(drop=True)
        is_known = np.array(
            [(g, t) in known_team_games for g, t in zip(served["game_id"], team, strict=True)]
        )
        inactive = is_known & got.eq(INACTIVE_STATUS).fillna(False).to_numpy(bool)
        share = np.where(inactive, 0.0, 1.0)
        gap = served[f"{side}_qb_starter_epa_per_dropback"].to_numpy(float) - _REPLACEMENT_QB_EPA
        parts[side] = np.nan_to_num((1.0 - share) * gap, nan=0.0)
        counts.append(
            pd.DataFrame(
                {
                    "season": served["season"].to_numpy(int),
                    "known": is_known,
                    "projected_ina": inactive,
                    "projected_other_status": is_known & got.notna().to_numpy() & ~inactive,
                    "projected_not_listed": is_known & got.isna().to_numpy(),
                    "projected_act": is_known & got.eq("ACT").fillna(False).to_numpy(bool),
                }
            )
        )
    counts = pd.concat(counts).groupby("season").sum()
    counts["projected_other_status"] -= counts["projected_act"]
    o3 = pd.DataFrame(
        {"game_id": served["game_id"], "diff_servable_o3": parts["home"] - parts["away"]}
    )
    return o3, counts


def build():
    manifest = json.loads(EXPECTED_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("stand_in_for_uncovered") is not False or not str(
        manifest.get("actives", "")
    ).replace("\\", "/").endswith("g55_u1/deadline_actives.parquet"):
        raise SystemExit("U2 rebuild was not built from the U1 actives without the stand-in")
    exp = pd.read_parquet(EXPECTED)
    if exp["active_source"].eq("snaps_gt_zero").any():
        raise SystemExit("U2 rebuild carries realised-snap stand-in actives")
    exp = exp.loc[exp["gsis_id"].notna()]
    expected = {
        (str(g), str(t), str(p)): float(s)
        for g, t, p, s in zip(
            exp["game_id"], exp["team"], exp["gsis_id"], exp["expected_share"], strict=True
        )
    }
    table = pd.read_parquet(ACTIVES)
    lists = deadline_lists(table)
    base = pd.read_parquet(BASE_FEATURES)
    _injuries_raw, rosters_raw, snaps_raw = load_player_snapshot(
        player_snapshot_from_root(PLAYER_SNAPSHOT)
    )
    rosters = canonicalize_rosters(rosters_raw)
    snaps_ids = attach_snap_player_ids(canonicalize_snaps(snaps_raw), rosters)
    inj, keys, stats = role_rows(base, lists, expected, snaps_ids)
    known_games = set(table.loc[table["known_at_deadline"], "game_id"].astype(str))
    if not keys["game_id"].isin(known_games).all():
        raise SystemExit("synthetic rows on a game not known at the deadline")
    pbp = load_pbp_snapshot(pbp_snapshot_from_root(PBP_SNAPSHOT))
    value_stats = load_player_value_snapshot(player_value_snapshot_from_root(VALUE_SNAPSHOT))
    out = base[["game_id"]].copy()
    for tag, column in (("o1", "sev1"), ("o2", "sev2")):
        enriched = run_enrich(base, inj, rosters_raw, snaps_raw, pbp, value_stats, column)
        part = enriched[
            [
                "game_id",
                "diff_injury_skill_epa_value_lost",
                "diff_injury_defense_disruption_value_lost",
            ]
        ].copy()
        part[f"diff_servable_{tag}"] = part["diff_injury_skill_epa_value_lost"].fillna(0.0) + part[
            "diff_injury_defense_disruption_value_lost"
        ].fillna(0.0)
        out = out.merge(part[["game_id", f"diff_servable_{tag}"]], on="game_id", how="left")
    served = pd.read_parquet(SERVED_FEATURES)
    o3, qb_counts = qb_term(served, table)
    out = out.merge(o3, on="game_id", how="left")
    out["deadline_known"] = out["game_id"].astype(str).isin(known_games)
    for column in COLUMNS:
        out[column] = out[column].fillna(0.0)
    leaked = out.loc[~out["deadline_known"], list(COLUMNS)].abs().sum().sum()
    if leaked != 0.0:
        raise SystemExit("nonzero servable term on a game not known at the deadline")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SYNTH_TABLE)
    slots = table[["game_id", "team", "season", "slot"]].drop_duplicates(["game_id", "team"])
    tg = keys.drop_duplicates().merge(slots, on=["game_id", "team"], how="left")
    inj_tg = inj.assign(game_id=keys["game_id"].to_numpy())
    summary = {
        "role_rows_stats": stats,
        "synthetic_rows": len(inj),
        "synthetic_rows_sev1": int((inj["sev1"] > 0).sum()),
        "synthetic_rows_sev2": int((inj["sev2"] > 0).sum()),
        "synthetic_rows_by_season": inj_tg.groupby("season").size().to_dict(),
        "team_games_with_synthetic_rows_by_season_slot": {
            f"{s}|{sl}": int(n) for (s, sl), n in tg.groupby(["season", "slot"]).size().items()
        },
        "team_games_known_by_season_slot": {
            f"{s}|{sl}": int(n)
            for (s, sl), n in table.loc[table["known_at_deadline"]]
            .drop_duplicates(["game_id", "team"])
            .groupby(["season", "slot"])
            .size()
            .items()
        },
        "qb_projected_status_counts_by_season": {
            str(s): {k: int(v) for k, v in row.items()} for s, row in qb_counts.iterrows()
        },
        "games_known": int(out["deadline_known"].sum()),
        "nonzero_games_by_term": {c: int((out[c].abs() > 0).sum()) for c in COLUMNS},
        "nonzero_on_unknown_games": float(leaked),
        "realised_snap_inputs_to_arm": 0,
        "u2_manifest": {k: manifest.get(k) for k in ("actives", "stand_in_for_uncovered")},
    }
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


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
        config = {**BASE_CONFIG, "ridge_alpha": BASE_ALPHA, "g55_u3_arm": tag}
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


def load_features():
    synth = pd.read_parquet(SYNTH_TABLE)
    served = pd.read_parquet(SERVED_FEATURES)
    features = served.merge(synth, on="game_id", how="left")
    for column in COLUMNS:
        features[column] = features[column].fillna(0.0)
    features["deadline_known"] = features["deadline_known"].fillna(False).astype(bool)
    return features


def grade_frame():
    per_game = pd.read_parquet(ARTIFACT)
    per_game = per_game.loc[per_game["season"].isin(SCORE_SEASONS)].reset_index(drop=True)
    y = per_game["home_covered"].to_numpy(float)
    move = np.abs(per_game["open_move"].to_numpy(float))
    pops = {"all": np.ones(len(per_game), bool), "quiet": move <= QUIET, "moved": move >= MOVED}
    return per_game, y, pops


def probability(scored, ids):
    return (
        scored.set_index("game_id")["home_cover_probability_at_open"].reindex(ids).to_numpy(float)
    )


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


def grade():
    features = load_features()
    per_game, y, pops = grade_frame()
    ids = per_game["game_id"]
    seconds = {}
    start = time.time()
    scored, _ = run_eval(features, (), "served_base")
    seconds["served_base"] = time.time() - start
    p_base = probability(scored, ids)
    base_all = metrics(p_base, y)
    model_alone = per_game["model_probability"].to_numpy(float)
    reproduced = {
        "record": base_all["record"],
        "declared": DECLARED_BASE_RECORD,
        "max_abs_diff_to_served_artifact": float(np.nanmax(np.abs(p_base - model_alone))),
    }
    if base_all["record"] != DECLARED_BASE_RECORD:
        (OUT / "results.json").write_text(
            json.dumps({"aborted": "base not reproduced", **reproduced}, indent=1),
            encoding="utf-8",
        )
        raise SystemExit(f"base not reproduced: {base_all['record']}")
    start = time.time()
    scored, log = run_eval(features, COLUMNS, "S_ALL")
    seconds["S_ALL"] = time.time() - start
    p_arm = probability(scored, ids)
    week_start = per_game.groupby(["season", "week"])["gameday"].min().sort_index()
    games_terms = features.set_index("game_id").loc[ids, [*COLUMNS, "deadline_known"]]
    result = {
        "family": FAMILY,
        "games": len(per_game),
        "base_reproduced": reproduced,
        "eval_seconds": seconds,
        "missing": {
            "served_base": int(np.isnan(p_base).sum()),
            "S_ALL": int(np.isnan(p_arm).sum()),
        },
        "graded_games_known_at_deadline": int(games_terms["deadline_known"].sum()),
        "nonzero_share_graded": {c: float((games_terms[c].abs() > 0).mean()) for c in COLUMNS},
        "season_mean_coefficients_scaled_units": season_coefficients(log, week_start),
        "populations": {},
    }
    wrng = np.random.default_rng(BOOTSTRAP_SEED)
    srng = np.random.default_rng(BOOTSTRAP_SEED)
    for pop, mask in pops.items():
        sub = per_game.loc[mask].reset_index(drop=True)
        week_block = paired(sub, p_arm[mask], p_base[mask], wrng)
        result["populations"][pop] = {
            "games": int(mask.sum()),
            "records": {
                "served_base": metrics(p_base[mask], y[mask]),
                "S_ALL": metrics(p_arm[mask], y[mask]),
            },
            "per_season": {
                "served_base": per_season(sub, p_base[mask]),
                "S_ALL": per_season(sub, p_arm[mask]),
            },
            "vs_base_season_block": season_paired(sub, p_arm[mask], p_base[mask], srng),
            "vs_base_week_block": week_block,
            "changed_picks": {
                "games": week_block["changed_pick_games"],
                "arm_record": f"{week_block['new_wins_on_changed']}-"
                f"{week_block['changed_pick_games'] - week_block['new_wins_on_changed']}",
                "base_record": f"{week_block['old_wins_on_changed']}-"
                f"{week_block['changed_pick_games'] - week_block['old_wins_on_changed']}",
            },
        }
    rows = per_game[["game_id", "season", "week", "home_covered", "open_move"]].copy()
    rows["p_base"] = p_base
    rows["p_arm"] = p_arm
    rows["pick_home_base"] = p_base >= 0.5
    rows["pick_home_arm"] = p_arm >= 0.5
    rows["correct_base"] = rows["pick_home_base"] == (y == 1)
    rows["correct_arm"] = rows["pick_home_arm"] == (y == 1)
    for column in [*COLUMNS, "deadline_known"]:
        rows[column] = games_terms[column].to_numpy()
    rows.to_parquet(OUT / "per_game.parquet", index=False)
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps(result, indent=1, default=str))


def null(start, draws):
    features = load_features()
    per_game, y, pops = grade_frame()
    ids = per_game["game_id"]
    known = features["deadline_known"].to_numpy(bool)
    seasons = features["season"].to_numpy(int)
    groups = [np.flatnonzero(known & (seasons == s)) for s in np.unique(seasons[known])]
    values = features[list(COLUMNS)].to_numpy(float)
    NULL_DIR.mkdir(parents=True, exist_ok=True)
    for k in range(start, start + draws):
        began = time.time()
        rng = np.random.default_rng([NULL_SEED, k])
        shuffled = values.copy()
        for idx in groups:
            shuffled[idx] = values[rng.permutation(idx)]
        frame = features.copy()
        frame[list(COLUMNS)] = shuffled
        scored, _ = run_eval(frame, COLUMNS, f"null_{k}")
        p = probability(scored, ids)
        entry = {
            "draw": k,
            "seed": [NULL_SEED, k],
            "seconds": time.time() - began,
            "populations": {pop: metrics(p[m], y[m]) for pop, m in pops.items()},
        }
        (NULL_DIR / f"draw_{k:03d}.json").write_text(json.dumps(entry, indent=1), encoding="utf-8")
        print(json.dumps({"draw": k, "seconds": entry["seconds"], **entry["populations"]["all"]}))


def wins(record):
    return int(record.split("-")[0])


def summarize():
    result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    draws = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(NULL_DIR.glob("draw_*.json"))
    ]
    null_summary = {"draws": len(draws), "populations": {}}
    for pop, entry in result["populations"].items():
        arm = entry["records"]["S_ALL"]
        null_wins = np.array([wins(d["populations"][pop]["record"]) for d in draws])
        null_ll = np.array([d["populations"][pop]["log_loss"] for d in draws])
        arm_wins = wins(arm["record"])
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
        "wins_at_least_declared": nall["arm_wins"] >= DECLARED_SERVE_WINS,
        "null_wins_percentile_at_least_declared": nall["wins_percentile"]
        >= DECLARED_SERVE_NULL_PCT,
        "log_loss_no_worse_than_base": allpop["records"]["S_ALL"]["log_loss"]
        <= allpop["records"]["served_base"]["log_loss"],
    }
    result["null"] = null_summary
    result["serving_candidate_tests"] = tests
    result["serving_candidate"] = all(tests.values())
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    cells = []
    for pop, entry in result["populations"].items():
        d = entry["vs_base_season_block"]["accuracy_points"]
        npop = null_summary["populations"][pop]
        cells.append(
            {
                "name": f"g55_u3_s_all_{pop}_accuracy_points",
                "description": (
                    "Servable deadline availability arm S_ALL (U1 gameday actives O1, U2 expected "
                    "snap share O2, projected QB inactive O3, known-at-deadline games only) minus "
                    f"served base, {pop} games, model alone at the Tuesday opener, season-block "
                    f"bootstrap; record {entry['records']['S_ALL']['record']} v base "
                    f"{entry['records']['served_base']['record']}; matched null wins percentile "
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
        "family": FAMILY,
        "source": "artifacts/g55_u3/results.json",
        "category": "health",
        "sample_blocks": len(SCORE_SEASONS),
        "notes": (
            "G55-U3 declared look family (all primary, quiet, moved = 3 looks); declaration in "
            "docs/lanes/goal-55-backwards-2026-11-21.md; serving candidate "
            f"{'passed' if result['serving_candidate'] else 'failed'}"
        ),
        "classification_evidence": (
            "Declared grade; no refuted mechanism and no positive-control bound, so the result is "
            "recorded below power with probability_positive and the matched-null percentile."
        ),
        "plain_summary": (
            "Gives the model the gameday inactive list, each player's expected playing time and "
            "whether the expected quarterback is inactive, using only what is posted before picks "
            "lock."
        ),
        "cells": cells,
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print(json.dumps({"null": null_summary, "tests": tests}, indent=1))


def main():
    parser = argparse.ArgumentParser(
        description="G55-U3 servable deadline availability arm on the P2-L opener harness."
    )
    parser.add_argument("mode", choices=["build", "grade", "null", "summarize"])
    parser.add_argument("--start", type=int, default=0, help="first null draw index")
    parser.add_argument("--draws", type=int, default=1, help="number of null draws")
    args = parser.parse_args()
    if args.mode == "build":
        build()
    elif args.mode == "grade":
        grade()
    elif args.mode == "null":
        null(args.start, args.draws)
    else:
        summarize()


if __name__ == "__main__":
    main()
