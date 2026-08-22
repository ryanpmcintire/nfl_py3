"""Adversarial red-team audit of four recent edge claims, using attack methods
DIFFERENT from each claim's original screen (different seeds, different
blocking, different estimators). Attribution on already-scored archive data
and fresh-seed resamples only: no rotation-registry window is spent anywhere
in this script.

Claims attacked (all previously measured on 2026-08-21 artifacts):

1. Overlay composition holdout (rho 0.72, ~+1pt fair expectation).
   Attacks: season-blocked-lower-bound-only subset selection;
   leave-one-season-out CV of the subset choice across all six seasons;
   within-week flip-shuffle permutation null for the rank-stability rho.
2. NFL.com Friday out_count>=2 (-2.69 pts full-slate, P(direction) .976).
   Attacks: leave-one-team-out and leave-one-season-out stability; a
   prior-week-win-rate quartile-stratified control for "bad teams have
   injured players"; starter vs non-starter decomposition of the Out
   designations.
3. Night body-clock west-road (P(direction negative) ~0.92).
   Attack: haversine travel-distance join against
   registry/stadium_coordinates.json; distance-controlled linear probability
   models with week-clustered SEs; distance-tercile stratification of the
   night gap; matched long-distance night-vs-day comparison among west-road
   teams.
4. Bye fade post-2011 (P+ 0.870 on the fade-full-slate cell).
   Attack: SHAM bye weeks (each team's true bye shifted +/-2 weeks within its
   season, fresh random draws); a real bye mechanism should vanish under sham
   assignment while a schedule artifact survives it.

Every new cell computed here carries the ``redteam_`` prefix so the owner can
record it via explicit ``nfl-ats weak-signals record`` calls; this script
never writes either registry JSON.

Usage (from the repo root)::

    .\\.tools\\uv.exe run python scripts/edge_audit_redteam.py [--claims 1,2,3,4]
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import body_clock_screen  # noqa: E402
import bye_overvaluation_screen  # noqa: E402
import nflcom_friday_designation_screen  # noqa: E402
from body_clock_screen import (  # noqa: E402
    DEFAULT_COORDS_PATH,
    load_coords,
)
from bye_overvaluation_screen import (  # noqa: E402
    block_bootstrap_two_group,
    build_bye_maps,
)
from nflcom_friday_designation_screen import (  # noqa: E402
    attach_flags,
    build_starter_keys,
    build_tuesday_visible,
    initial_last_key,
    latest,
    load_report_flags,
)
from overlay_stack_backtest import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_PER_GAME_ARTIFACT,
    OVERLAY_NAMES,
    build_predictions_frame,
    load_inputs,
    run_overlays,
    verify_no_direction_conflicts,
)
from overlay_subset_composition import (  # noqa: E402
    ARREST_MEMBER_NAME,
    blocked_bootstrap_matrix,
    build_delta_matrix,
    reconstruct_arrest_flip_set,
)

from nfl_ats.provenance import (  # noqa: E402
    artifact_provenance,
    sha256_file,
    write_experiment_artifact,
)

REDTEAM_SEED = 20260822
DEFAULT_OUTPUT_ROOT = Path("artifacts/edge_audit_redteam")
PERMUTATIONS = 400
PLACEBO_DRAWS = 100
PLACEBO_SAMPLES = 4000


def _pts(value: float) -> float:
    return float(value * 100.0)


def summarize_gap(
    df: pd.DataFrame,
    *,
    flag: np.ndarray | pd.Series,
    value_col: str,
    block_col: str = "week_block",
    samples: int,
    seed: int,
) -> dict[str, Any]:
    work = df.copy()
    flag_arr = np.asarray(flag, dtype=bool)
    n_total = len(work)
    n_flag = int(flag_arr.sum())
    if n_flag == 0 or n_flag == n_total:
        return {"n_total": n_total, "n_flag": n_flag, "insufficient_data": True}
    values = work[value_col].to_numpy(dtype=float)
    valid = np.isfinite(values)
    subset_cover = float(values[valid & flag_arr].mean())
    complement_cover = float(values[valid & ~flag_arr].mean())
    raw_gap_pts = (subset_cover - complement_cover) * 100.0
    fraction_of_slate = float(flag_arr.mean())
    draws = block_bootstrap_two_group(
        work.assign(_flag=flag_arr).loc[valid],
        flag_col="_flag",
        value_col=value_col,
        block_col=block_col,
        samples=samples,
        seed=seed,
    )
    scaled = draws * fraction_of_slate
    lower, upper = np.quantile(scaled, [0.025, 0.975])
    return {
        "n_total": n_total,
        "n_flag": n_flag,
        "subset_cover": subset_cover,
        "complement_cover": complement_cover,
        "raw_gap_pts": raw_gap_pts,
        "fraction_of_slate": fraction_of_slate,
        "full_slate_effect_pts": raw_gap_pts * fraction_of_slate,
        "ci95_scaled": [float(lower), float(upper)],
        "probability_positive": float(np.mean(draws > 0)),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "insufficient_data": False,
    }


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 3958.7613
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def lpm_cluster(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    n, k = x.shape
    xt_x_inv = np.linalg.pinv(x.T @ x)
    beta = xt_x_inv @ (x.T @ y)
    resid = y - x @ beta
    _, inv = np.unique(groups, return_inverse=True)
    g = int(inv.max()) + 1
    meat = np.zeros((k, k))
    for j in range(g):
        idx = inv == j
        score = x[idx].T @ resid[idx]
        meat += np.outer(score, score)
    c = g / (g - 1) * (n - 1) / max(n - k, 1)
    vcov = c * (xt_x_inv @ meat @ xt_x_inv)
    ses = np.sqrt(np.diag(vcov))
    z = [float(b / s) if s > 0 else None for b, s in zip(beta, ses, strict=True)]
    return {
        "coefficients": [float(b) for b in beta],
        "cluster_se": [float(s) for s in ses],
        "z": z,
        "n": int(n),
        "n_clusters": g,
    }


def run_claim1(args: argparse.Namespace) -> dict[str, Any]:
    print("=== claim 1: overlay composition holdout ===")
    per_game, schedules, player_features, snapshot_name, player_feature_path = load_inputs(
        args.per_game_artifact, args.data_root
    )
    predictions = build_predictions_frame(per_game, schedules)
    results = run_overlays(predictions, schedules, player_features)
    flip_sets = {name: {flip.game_id for flip in result.flips} for name, result in results.items()}
    verify_no_direction_conflicts(predictions, results, flip_sets)
    arrest_flip_ids, _scored = reconstruct_arrest_flip_set(per_game, args.features, args.incidents)
    members: tuple[str, ...] = (*OVERLAY_NAMES, ARREST_MEMBER_NAME)
    member_flip_sets = {name: flip_sets[name] for name in OVERLAY_NAMES}
    member_flip_sets[ARREST_MEMBER_NAME] = arrest_flip_ids

    subsets: list[tuple[str, ...]] = []
    for size in range(1, len(members) + 1):
        subsets.extend(tuple(sorted(combo)) for combo in combinations(members, size))

    eval_frame = predictions[["game_id", "season", "week"]].merge(
        per_game[["game_id", "correct_at_open_probability_rule"]], on="game_id", how="left"
    )
    eval_frame = eval_frame.rename(columns={"correct_at_open_probability_rule": "correct_baseline"})
    eval_frame["correct_baseline"] = pd.to_numeric(eval_frame["correct_baseline"], errors="coerce")
    valid_mask = eval_frame["correct_baseline"].notna().to_numpy()
    base_valid = eval_frame.loc[valid_mask, "correct_baseline"].to_numpy(dtype=float)

    deltas_full = build_delta_matrix(
        eval_frame["correct_baseline"], eval_frame["game_id"], member_flip_sets, members, subsets
    )
    deltas = deltas_full[valid_mask]
    seasons = eval_frame.loc[valid_mask, "season"].reset_index(drop=True)
    weeks = eval_frame.loc[valid_mask, "week"].reset_index(drop=True)
    blocks_valid = seasons.to_frame("season").assign(week=weeks.to_numpy())

    sel_mask = seasons.isin((2020, 2021, 2022)).to_numpy()
    eval_mask = ~sel_mask
    means_sel = deltas[sel_mask].mean(axis=0)
    means_eval = deltas[eval_mask].mean(axis=0)
    rho_obs = float(spearmanr(means_sel, means_eval).statistic)
    slope_obs = float(np.polyfit(means_sel, means_eval, 1)[0])

    season_blocked_only: list[dict[str, Any]] = []
    for label, selection_seasons, evaluation_seasons in (
        ("forward_2020_2022_to_2023_2025", (2020, 2021, 2022), (2023, 2024, 2025)),
        ("reverse_2023_2025_to_2020_2022", (2023, 2024, 2025), (2020, 2021, 2022)),
    ):
        s_mask = seasons.isin(selection_seasons).to_numpy()
        e_mask = seasons.isin(evaluation_seasons).to_numpy()
        stats_sel = blocked_bootstrap_matrix(
            deltas[s_mask],
            blocks_valid.loc[s_mask].reset_index(drop=True),
            block="season",
            samples=args.samples,
            seed=args.seed + 11,
        )
        best_column = int(np.argmax(stats_sel["lower"]))
        frozen = subsets[best_column]
        holdout = blocked_bootstrap_matrix(
            deltas[e_mask][:, [best_column]],
            blocks_valid.loc[e_mask].reset_index(drop=True),
            block="week",
            samples=args.samples,
            seed=args.seed + 12,
        )
        season_holdout = blocked_bootstrap_matrix(
            deltas[e_mask][:, [best_column]],
            blocks_valid.loc[e_mask].reset_index(drop=True),
            block="season",
            samples=args.samples,
            seed=args.seed + 13,
        )
        record = {
            "label": label,
            "selection_criterion": (
                "max season-blocked bootstrap LOWER bound on selection half "
                "(conservative estimator; original used raw mean argmax)"
            ),
            "frozen_members": list(frozen),
            "selection_lower_bound_accuracy_points": _pts(stats_sel["lower"][best_column]),
            "holdout_delta_accuracy_points": _pts(holdout["estimate"][0]),
            "holdout_week_blocked_p_plus": float(holdout["probability_positive"][0]),
            "holdout_week_blocked_ci95_accuracy_points": [
                _pts(holdout["lower"][0]),
                _pts(holdout["upper"][0]),
            ],
            "holdout_season_blocked_p_plus": float(season_holdout["probability_positive"][0]),
        }
        season_blocked_only.append(record)
        print(
            f"  season-blocked-only {label}: {'+'.join(frozen)} "
            f"holdout {record['holdout_delta_accuracy_points']:+.4f} pts "
            f"P+(wk) {record['holdout_week_blocked_p_plus']:.4f}"
        )

    folds: list[dict[str, Any]] = []
    pooled_deltas: list[np.ndarray] = []
    pooled_week_blocks: list[pd.DataFrame] = []
    all_seasons = sorted(int(s) for s in seasons.unique())
    for held_out in all_seasons:
        train_mask = seasons.ne(held_out).to_numpy()
        test_mask = seasons.eq(held_out).to_numpy()
        best_column = int(np.argmax(deltas[train_mask].mean(axis=0)))
        fold_deltas = deltas[test_mask][:, best_column]
        pooled_deltas.append(fold_deltas)
        pooled_week_blocks.append(blocks_valid.loc[test_mask].reset_index(drop=True))
        folds.append(
            {
                "held_out_season": held_out,
                "selected_members": list(subsets[best_column]),
                "train_delta_accuracy_points": _pts(deltas[train_mask][:, best_column].mean()),
                "held_out_delta_accuracy_points": _pts(fold_deltas.mean()),
                "n_games": int(test_mask.sum()),
            }
        )
    pooled = np.concatenate(pooled_deltas)
    pooled_blocks = pd.concat(pooled_week_blocks, ignore_index=True)
    cv_stats = blocked_bootstrap_matrix(
        pooled[:, None],
        pooled_blocks,
        block="week",
        samples=args.samples,
        seed=args.seed + 14,
    )
    cv_payload = {
        "protocol": (
            "for each held-out season: select argmax-mean subset on the other five "
            "seasons, score that frozen subset on the held-out season; pooled "
            "held-out deltas get a week-blocked bootstrap"
        ),
        "folds": folds,
        "pooled_held_out_delta_accuracy_points": _pts(pooled.mean()),
        "pooled_week_blocked_ci95_accuracy_points": [
            _pts(cv_stats["lower"][0]),
            _pts(cv_stats["upper"][0]),
        ],
        "pooled_week_blocked_p_plus": float(cv_stats["probability_positive"][0]),
        "positive_folds": int(sum(1 for f in folds if f["held_out_delta_accuracy_points"] > 0)),
        "n_folds": len(folds),
    }
    print(
        f"  LOSO-CV pooled held-out delta "
        f"{cv_payload['pooled_held_out_delta_accuracy_points']:+.4f} pts "
        f"P+ {cv_payload['pooled_week_blocked_p_plus']:.4f} "
        f"({cv_payload['positive_folds']}/{len(folds)} folds positive)"
    )

    rng = np.random.default_rng(args.seed + 15)
    week_ids = (blocks_valid["season"] * 100 + blocks_valid["week"]).to_numpy()
    unique_weeks = np.unique(week_ids)
    membership_float = np.column_stack(
        [
            eval_frame["game_id"]
            .isin(member_flip_sets[name])
            .to_numpy(dtype=np.float64)[valid_mask]
            for name in members
        ]
    )
    bits = np.zeros((len(members), len(subsets)), dtype=np.float64)
    for column, subset in enumerate(subsets):
        for index, name in enumerate(members):
            if name in subset:
                bits[index, column] = 1.0

    def half_means_from(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flipped_any = (m @ bits) > 0
        d = np.where(flipped_any, (1.0 - 2.0 * base_valid)[:, None], 0.0)
        return d[sel_mask].mean(axis=0), d[eval_mask].mean(axis=0)

    check_sel, check_eval = half_means_from(membership_float)
    assert np.allclose(check_sel, means_sel, atol=1e-12)
    assert np.allclose(check_eval, means_eval, atol=1e-12)

    m_perm = membership_float.copy()
    rho_null = np.empty(PERMUTATIONS)
    for perm in range(PERMUTATIONS):
        for col in range(m_perm.shape[1]):
            for wk in unique_weeks:
                idx = np.flatnonzero(week_ids == wk)
                m_perm[idx, col] = m_perm[rng.permutation(idx), col]
        ms, me = half_means_from(m_perm)
        rho_null[perm] = spearmanr(ms, me).statistic
    p_value = float(np.mean(rho_null >= rho_obs))
    perm_payload = {
        "null": (
            "each member's flip indicator independently shuffled among games within "
            "its week (preserves weekly flip counts and weekly slate structure, "
            "destroys all game-level signal)"
        ),
        "permutations": PERMUTATIONS,
        "observed_rho": rho_obs,
        "observed_shrinkage_slope": slope_obs,
        "null_rho_mean": float(rho_null.mean()),
        "null_rho_sd": float(rho_null.std(ddof=1)),
        "null_rho_q95": float(np.quantile(rho_null, 0.95)),
        "empirical_p_one_sided": p_value,
        "permutation_seed": int(args.seed + 15),
    }
    print(
        f"  permutation null: observed rho {rho_obs:.4f} vs null mean {rho_null.mean():.4f} "
        f"sd {rho_null.std(ddof=1):.4f}, one-sided empirical p {p_value:.4f}"
    )

    return {
        "source_artifact": str(args.per_game_artifact),
        "source_artifact_sha256": sha256_file(args.per_game_artifact),
        "schedule_snapshot": snapshot_name,
        "player_feature_table": str(player_feature_path),
        "incidents_table": str(args.incidents),
        "n_scored_games": int(valid_mask.sum()),
        "n_subsets": len(subsets),
        "bootstrap_samples": int(args.samples),
        "redteam_seed_base": int(args.seed),
        "independent_reproduction": {
            "note": (
                "same construction as artifacts/overlay_selection_holdout/"
                "20260821T195512Z rebuilt this session with redteam seeds; the "
                "original artifact reported rho 0.7207 and slope 0.6356"
            ),
            "spearman_rho_selection_vs_holdout": rho_obs,
            "ols_slope_holdout_on_selection": slope_obs,
            "global_max_subset_members": list(subsets[int(np.argmax(deltas.mean(axis=0)))]),
            "global_max_subset_delta_accuracy_points": _pts(deltas.mean(axis=0).max()),
        },
        "attack_season_blocked_only_selection": season_blocked_only,
        "attack_leave_one_season_out_cv": cv_payload,
        "attack_rank_stability_permutation_null": perm_payload,
    }


def run_claim2(args: argparse.Namespace) -> dict[str, Any]:
    print("=== claim 2: nfl.com friday out_count>=2 ===")
    schedules_path = latest(REPO / "data" / "raw", "*/schedules.parquet")
    raw_sched = pd.read_parquet(schedules_path)
    long = nflcom_friday_designation_screen.load_population(schedules_path)
    qa, report_counts = load_report_flags(REPO / "data" / "raw" / "nflcom_injuries")
    snaps_path = latest(REPO / "data" / "players" / "raw", "*/snap_counts.parquet")
    starter_exact, starter_fuzzy = build_starter_keys(snaps_path)
    nflverse_exact, nflverse_fuzzy = build_tuesday_visible(REPO / "data" / "players" / "raw")
    work = attach_flags(long, qa, starter_exact, starter_fuzzy, nflverse_exact, nflverse_fuzzy)

    qa_local = qa.copy()
    starter_status: list[bool] = []
    for season, week, team, name in zip(
        qa_local["season"], qa_local["week"], qa_local["team"], qa_local["norm_name"], strict=True
    ):
        key3 = (int(season), int(week), str(team))
        init_last = initial_last_key(str(name))
        starter_status.append(
            (*key3, str(name)) in starter_exact
            or (init_last != ("", "") and (*key3, *init_last) in starter_fuzzy)
        )
    qa_local["is_starter"] = starter_status
    out_rows = qa_local.loc[qa_local["status_norm"] == "out"].copy()
    out_rows["nonstarter"] = ~out_rows["is_starter"]
    counts = (
        out_rows.groupby(["season", "week", "team"])
        .agg(out_starter=("is_starter", "sum"), out_nonstarter=("nonstarter", "sum"))
        .reset_index()
    )
    work = work.merge(counts, on=["season", "week", "team"], how="left")
    work[["out_starter", "out_nonstarter"]] = work[["out_starter", "out_nonstarter"]].fillna(0)
    work["out_starter"] = work["out_starter"].astype(int)
    work["out_nonstarter"] = work["out_nonstarter"].astype(int)

    scores = raw_sched[
        ["game_id", "home_team", "away_team", "home_score", "away_score"]
    ].drop_duplicates("game_id")
    wl = long.merge(scores, on="game_id", how="left", validate="many_to_one")
    is_home = wl["team"] == wl["home_team"]
    own = np.where(is_home, wl["home_score"], wl["away_score"]).astype(float)
    opp = np.where(is_home, wl["away_score"], wl["home_score"]).astype(float)
    wl["won"] = np.where(np.isfinite(own) & np.isfinite(opp), (own > opp).astype(float), np.nan)
    wl = wl.sort_values(["team", "gameday"]).reset_index(drop=True)
    wl["prior_games"] = wl.groupby("team").cumcount()
    wl["prior_wins"] = (
        wl.groupby("team")["won"].transform(lambda s: s.shift(1).cumsum()).fillna(0.0)
    )
    wl["prior_win_rate"] = wl["prior_wins"] / wl["prior_games"].replace(0, np.nan)
    work = work.merge(
        wl[["game_id", "team", "prior_win_rate"]],
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )

    flag_any = work["out_count"].ge(2).to_numpy()
    repro = summarize_gap(
        work, flag=flag_any, value_col="team_cover", samples=args.samples, seed=args.seed + 21
    )
    print(
        f"  reproduction redteam_nflcom_out2_any: raw gap {repro['raw_gap_pts']:+.3f} pts "
        f"P+ {repro['probability_positive']:.4f} n_flag {repro['n_flag']}"
    )

    loo_team: dict[str, float] = {}
    cover_values = work["team_cover"].to_numpy(dtype=float)

    def gap_excluding(mask: np.ndarray) -> float | None:
        f = flag_any[mask]
        v = cover_values[mask]
        if f.sum() == 0 or (~f & np.isfinite(v)).sum() == 0:
            return None
        return float((v[f].mean() - v[~f].mean()) * 100.0)

    for team in sorted(work["team"].unique()):
        gap = gap_excluding(work["team"].ne(team).to_numpy())
        if gap is not None:
            loo_team[str(team)] = gap
    loo_values = np.array(list(loo_team.values()))
    loo_season: dict[str, float] = {}
    for season in sorted(int(s) for s in work["season"].unique()):
        gap = gap_excluding(work["season"].ne(season).to_numpy())
        if gap is not None:
            loo_season[str(season)] = gap
    print(
        f"  LOO team: gaps {loo_values.min():+.3f}..{loo_values.max():+.3f} pts, "
        f"{int((loo_values > 0).sum())}/{len(loo_values)} positive; "
        f"LOO season: {loo_season}"
    )

    cov = work["prior_win_rate"]
    analyzed = cov.notna().to_numpy()
    sub_cover = cover_values[analyzed]
    sub_flag = flag_any[analyzed]
    quartiles = np.quantile(cov.to_numpy()[analyzed], [0.25, 0.5, 0.75])
    bins = np.digitize(cov.to_numpy()[analyzed], quartiles, right=False)
    total_flag = int(sub_flag.sum())
    strata = []
    adjusted_flag_weighted = 0.0
    for b in range(4):
        mask_b = bins == b
        fb = sub_flag[mask_b]
        if fb.sum() == 0 or (~fb).sum() == 0:
            continue
        vb = sub_cover[mask_b]
        gap = float((vb[fb].mean() - vb[~fb].mean()) * 100.0)
        adjusted_flag_weighted += gap * (fb.sum() / total_flag)
        lo_edge = float(quartiles[b - 1]) if b > 0 else 0.0
        hi_edge = float(quartiles[b]) if b < 3 else 1.0
        strata.append(
            {
                "stratum": b,
                "prior_win_rate_range": [lo_edge, hi_edge],
                "n_flag": int(fb.sum()),
                "flag_rate": float(fb.mean()),
                "gap_pts": gap,
            }
        )
    unadjusted = float((sub_cover[sub_flag].mean() - sub_cover[~sub_flag].mean()) * 100.0)
    strat_payload = {
        "covariate": "prior within-season win rate (pregame-safe); week>=2 team-games only",
        "quartile_edges": [float(q) for q in quartiles],
        "strata": strata,
        "unadjusted_gap_pts_analyzed_subset": unadjusted,
        "flag_weighted_within_stratum_gap_pts": float(adjusted_flag_weighted),
        "interpretation": (
            "if the negative gap persists inside every win-rate stratum, 'bad teams "
            "have injured players' does not explain it away"
        ),
    }
    print(
        f"  bad-team control: unadjusted {unadjusted:+.3f} pts -> "
        f"within-stratum adjusted {adjusted_flag_weighted:+.3f} pts"
    )

    flag_start = work["out_starter"].ge(2).to_numpy()
    flag_nonstart = ((work["out_nonstarter"] >= 2) & (work["out_starter"] == 0)).to_numpy()
    cell_start = summarize_gap(
        work, flag=flag_start, value_col="team_cover", samples=args.samples, seed=args.seed + 22
    )
    cell_nonstart = summarize_gap(
        work, flag=flag_nonstart, value_col="team_cover", samples=args.samples, seed=args.seed + 23
    )
    start_gap = cell_start.get("raw_gap_pts")
    nonstart_gap = cell_nonstart.get("raw_gap_pts")
    print(
        f"  starters-only >=2 outs: gap {start_gap:+.3f} pts P+ "
        f"{cell_start.get('probability_positive'):.4f} n {cell_start['n_flag']}"
    )
    print(
        f"  nonstarters-only >=2 outs: gap {nonstart_gap:+.3f} pts P+ "
        f"{cell_nonstart.get('probability_positive'):.4f} n {cell_nonstart['n_flag']}"
    )

    return {
        "population": {
            "n_games": int(len(work) / 2),
            "n_team_games": len(work),
            "report_counts": report_counts,
            "schedules": str(schedules_path),
        },
        "bootstrap_samples": int(args.samples),
        "redteam_seed_base": int(args.seed),
        "independent_reproduction_redteam_nflcom_out2_any": repro,
        "attack_leave_one_out_stability": {
            "leave_one_team_gaps_pts": loo_team,
            "leave_one_season_gaps_pts": loo_season,
        },
        "attack_bad_team_control_stratified": strat_payload,
        "attack_starter_proxy_decomposition": {
            "redteam_nflcom_out2_starters_only": cell_start,
            "redteam_nflcom_out2_nonstarters_only": cell_nonstart,
        },
    }


def run_claim3(args: argparse.Namespace) -> dict[str, Any]:
    print("=== claim 3: night body-clock west-road vs travel distance ===")
    coords = load_coords(DEFAULT_COORDS_PATH)
    df = body_clock_screen.load_population(bye_overvaluation_screen.DEFAULT_SCHEDULES, coords)
    away_west = df["away_body_tz"].isin({"America/Los_Angeles", "America/Phoenix"})
    true_home = df["location"] == "Home"
    night = df["kick_min"] >= 20 * 60

    def coord_of(stadium: object, key: str) -> Any:
        entry = coords.get(stadium) if isinstance(stadium, str) else None
        return entry[key] if entry else None

    dist = np.full(len(df), np.nan)
    venue_lat = df["stadium"].map(lambda s: coord_of(s, "lat"))
    venue_lon = df["stadium"].map(lambda s: coord_of(s, "lon"))
    home_lat = df["team_home_stadium"].map(lambda s: coord_of(s, "lat"))
    home_lon = df["team_home_stadium"].map(lambda s: coord_of(s, "lon"))
    for i, (la1, lo1, la2, lo2) in enumerate(
        zip(home_lat, home_lon, venue_lat, venue_lon, strict=True)
    ):
        if all(pd.notna(x) for x in (la1, lo1, la2, lo2)):
            dist[i] = haversine_miles(la1, lo1, la2, lo2)
    df["_dist_mi"] = dist
    unresolved_distance = int((df["_dist_mi"].isna()).sum())

    flag_west_night = (away_west & true_home & night).fillna(False).to_numpy()
    repro = summarize_gap(
        df, flag=flag_west_night, value_col="home_cover", samples=args.samples, seed=args.seed + 31
    )
    p_direction_negative = 1.0 - repro["probability_positive"]
    print(
        f"  reproduction redteam_body_clock_night_west_road_ge2000et: raw gap "
        f"{repro['raw_gap_pts']:+.3f} pts P+(home_cover) {repro['probability_positive']:.4f} "
        f"P(direction negative) {p_direction_negative:.4f} n {repro['n_flag']}"
    )

    road_true = true_home.fillna(False).to_numpy()
    night_arr = night.fillna(False).to_numpy()
    west_arr = away_west.fillna(False).to_numpy()
    cover = df["home_cover"].to_numpy(dtype=float)
    dist_arr = df["_dist_mi"].to_numpy()
    ok = np.isfinite(cover) & np.isfinite(dist_arr)

    night_pop = ok & road_true & night_arr
    y = cover[night_pop]
    west_f = west_arr[night_pop].astype(float)
    d = dist_arr[night_pop] / 1000.0
    xa = np.column_stack([np.ones_like(y), west_f, d])
    model_a = lpm_cluster(y, xa, df["week_block"].to_numpy()[night_pop])
    model_a["names"] = ["const", "west_body_clock_visitor", "distance_1000mi"]

    west_road_pop = ok & road_true & west_arr
    yb = cover[west_road_pop]
    night_f = night_arr[west_road_pop].astype(float)
    db = dist_arr[west_road_pop] / 1000.0
    xb = np.column_stack([np.ones_like(yb), night_f, db])
    model_b = lpm_cluster(yb, xb, df["week_block"].to_numpy()[west_road_pop])
    model_b["names"] = ["const", "night_kick_ge2000et", "distance_1000mi"]

    terciles = np.quantile(dist_arr[night_pop], [1 / 3, 2 / 3])
    bins = np.digitize(dist_arr[night_pop], terciles, right=False)
    strata = []
    for b in range(3):
        mask_b = bins == b
        wb = west_f[mask_b] > 0
        if wb.sum() == 0 or (~wb).sum() == 0:
            continue
        vb = y[mask_b]
        strata.append(
            {
                "tercile": b,
                "distance_range_mi": [
                    float(dist_arr[night_pop][mask_b].min()),
                    float(dist_arr[night_pop][mask_b].max()),
                ],
                "n_west": int(wb.sum()),
                "n_other": int((~wb).sum()),
                "home_cover_west": float(vb[wb].mean()),
                "home_cover_other": float(vb[~wb].mean()),
                "gap_pts": float((vb[wb].mean() - vb[~wb].mean()) * 100.0),
            }
        )

    med_all = float(np.median(dist_arr[ok & road_true]))
    long_west = west_road_pop & (dist_arr >= med_all)
    day_long = cover[long_west & ~night_arr]
    night_long = cover[long_west & night_arr]
    has_both = len(day_long) > 0 and len(night_long) > 0
    matched = {
        "definition": (
            f"west-body-clock road games at or above the median road-game distance "
            f"({med_all:.0f} mi): night (>=20:00 ET) vs earlier-kickoff cover rates"
        ),
        "n_day": len(day_long),
        "day_home_cover": float(day_long.mean()) if len(day_long) else None,
        "n_night": len(night_long),
        "night_home_cover": float(night_long.mean()) if len(night_long) else None,
        "night_minus_day_gap_pts": float((night_long.mean() - day_long.mean()) * 100.0)
        if has_both
        else None,
    }

    print(
        f"  LPM night road games: west coef {model_a['coefficients'][1]:+.4f} "
        f"(SE {model_a['cluster_se'][1]:.4f}, z {model_a['z'][1]:+.2f}); "
        f"distance coef {model_a['coefficients'][2]:+.4f}"
    )
    print(
        f"  LPM west-body-clock road games: night coef "
        f"{model_b['coefficients'][1]:+.4f} (SE {model_b['cluster_se'][1]:.4f}, "
        f"z {model_b['z'][1]:+.2f})"
    )
    for s in strata:
        print(
            f"  night distance tercile {s['tercile']} "
            f"({s['distance_range_mi'][0]:.0f}-{s['distance_range_mi'][1]:.0f} mi): "
            f"gap {s['gap_pts']:+.2f} pts (n_west {s['n_west']})"
        )
    if has_both:
        print(f"  matched long-distance west-road: {matched}")

    return {
        "n_scored_with_distance": int(ok.sum()),
        "rows_missing_distance": unresolved_distance,
        "bootstrap_samples": int(args.samples),
        "redteam_seed_base": int(args.seed),
        "coords_table": str(DEFAULT_COORDS_PATH),
        "independent_reproduction_redteam_body_clock_night_west_road_ge2000et": repro,
        "attack_distance_control": {
            "lpm_night_road_games": model_a,
            "lpm_west_body_clock_road_games": model_b,
            "night_distance_terciles": strata,
            "matched_long_distance_night_vs_day_among_west_road": matched,
        },
    }


def build_bye_maps_within_season(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    long_rows = []
    for _, g in df.iterrows():
        for side, team in (("home", g["home_team"]), ("away", g["away_team"])):
            long_rows.append(
                {
                    "game_id": g["game_id"],
                    "team": team,
                    "side": side,
                    "season": int(g["season"]),
                    "gameday_dt": g["gameday_dt"],
                }
            )
    long_df = pd.DataFrame(long_rows).sort_values(["team", "season", "gameday_dt"])
    grp = long_df.groupby(["team", "season"])
    long_df["gap_days"] = grp["gameday_dt"].diff().dt.days
    long_df["post_bye"] = (long_df["gap_days"] >= 12).fillna(False)

    def side_map(side: str) -> pd.Series:
        joined = df[["game_id"]].merge(
            long_df.loc[long_df["side"] == side, ["game_id", "post_bye"]],
            on="game_id",
            how="left",
        )
        return joined.set_index("game_id")["post_bye"].reindex(df["game_id"]).fillna(False)

    return side_map("home"), side_map("away")


def run_claim4(args: argparse.Namespace) -> dict[str, Any]:
    print("=== claim 4: bye fade post-2011 sham-bye placebo ===")
    df = bye_overvaluation_screen.load_population(bye_overvaluation_screen.DEFAULT_SCHEDULES)
    home_pb_cross, away_pb_cross = build_bye_maps(df)
    home_pb, away_pb = build_bye_maps_within_season(df)
    cross_season_openers = int((home_pb_cross.to_numpy() & ~home_pb.to_numpy()).sum()) + int(
        (away_pb_cross.to_numpy() & ~away_pb.to_numpy()).sum()
    )
    df["home_off_bye"] = home_pb.to_numpy()
    df["away_off_bye"] = away_pb.to_numpy()
    era_post = df["season"].ge(2012)

    def fade_cell(
        off_bye_home: pd.Series, off_bye_away: pd.Series, era: pd.Series
    ) -> dict[str, Any]:
        edge_home = (off_bye_home & ~off_bye_away).fillna(False)
        edge_away = (off_bye_away & ~off_bye_home).fillna(False)
        fade_side_cover = np.where(edge_home, 1.0 - df["home_cover"], df["home_cover"])
        work = df.assign(_fade=fade_side_cover)
        blocks = set(work.loc[edge_home | edge_away, "week_block"].unique().tolist())
        pop = (era & work["week_block"].isin(blocks)).to_numpy()
        sub = work.loc[pop]
        flag = (edge_home | edge_away).to_numpy()[pop]
        return summarize_gap(
            sub,
            flag=flag,
            value_col="_fade",
            block_col="week_block",
            samples=PLACEBO_SAMPLES,
            seed=args.seed + 41,
        )

    repro = fade_cell(df["home_off_bye"], df["away_off_bye"], era_post)
    print(
        f"  reproduction redteam_bye_overval_fade_full_slate_post2011: effect "
        f"{repro['full_slate_effect_pts']:+.4f} pts P+ {repro['probability_positive']:.4f} "
        f"n {repro['n_flag']}"
    )

    bye_map: dict[tuple[int, str], int] = {}
    flagged = df.loc[df["home_off_bye"] | df["away_off_bye"]]
    multi_gap_team_seasons = set()
    for _, g in flagged.iterrows():
        if g["home_off_bye"]:
            key_h = (int(g["season"]), str(g["home_team"]))
            if key_h in bye_map:
                multi_gap_team_seasons.add(key_h)
            bye_map.setdefault(key_h, int(g["week"]))
        if g["away_off_bye"]:
            key_a = (int(g["season"]), str(g["away_team"]))
            if key_a in bye_map:
                multi_gap_team_seasons.add(key_a)
            bye_map.setdefault(key_a, int(g["week"]))
    max_week = df.groupby("season")["week"].max().to_dict()
    cross_repro = fade_cell(
        home_pb_cross.set_axis(df.index), away_pb_cross.set_axis(df.index), era_post
    )

    draw_records = []
    effects = []
    p_pluses = []
    flags_counts = []
    for draw in range(PLACEBO_DRAWS):
        rng = np.random.default_rng(args.seed + 1000 + draw)
        sham: dict[tuple[int, str], int] = {}
        for (season, team), wk in bye_map.items():
            delta = int(rng.choice([-2, 2]))
            target = wk + delta
            hi = int(max_week.get(season, 18))
            if target < 2 or target > hi:
                target = wk - delta
            target = int(min(max(target, 2), hi))
            sham[(season, team)] = target
        sham_home = pd.Series(
            [
                sham.get((int(s), t), -1) == int(w)
                for s, t, w in zip(df["season"], df["home_team"], df["week"], strict=True)
            ],
            index=df.index,
        )
        sham_away = pd.Series(
            [
                sham.get((int(s), t), -1) == int(w)
                for s, t, w in zip(df["season"], df["away_team"], df["week"], strict=True)
            ],
            index=df.index,
        )
        cell = fade_cell(sham_home, sham_away, era_post)
        if cell.get("insufficient_data"):
            continue
        effects.append(cell["full_slate_effect_pts"])
        p_pluses.append(cell["probability_positive"])
        flags_counts.append(cell["n_flag"])
        draw_records.append(
            {
                "draw": draw,
                "full_slate_effect_pts": cell["full_slate_effect_pts"],
                "raw_gap_pts": cell["raw_gap_pts"],
                "probability_positive": cell["probability_positive"],
                "n_flag": cell["n_flag"],
            }
        )
    eff = np.array(effects)
    obs_eff = repro["full_slate_effect_pts"]
    placebo_payload = {
        "protocol": (
            "each team's true strict-bye week shifted +/-2 weeks within its season "
            "(random direction per team-season, clipped to the season's week range); "
            "the fade-full-slate cell rebuilt identically under sham assignment"
        ),
        "n_draws": len(eff),
        "samples_per_draw": PLACEBO_SAMPLES,
        "placebo_effect_mean_pts": float(eff.mean()),
        "placebo_effect_sd_pts": float(eff.std(ddof=1)),
        "placebo_effect_q975_pts": float(np.quantile(eff, 0.975)),
        "placebo_abs_at_least_observed_frac": float(np.mean(np.abs(eff) >= abs(obs_eff))),
        "placebo_mean_p_plus": float(np.mean(p_pluses)),
        "mean_placebo_n_flag": float(np.mean(flags_counts)),
        "observed_effect_pts": obs_eff,
        "observed_n_flag": repro["n_flag"],
        "draws": draw_records,
    }
    print(
        f"  placebo over {len(eff)} draws: mean effect {eff.mean():+.4f} pts "
        f"(sd {eff.std(ddof=1):.4f}, q97.5 {np.quantile(eff, 0.975):+.4f}), "
        f"P(|effect| >= observed) {placebo_payload['placebo_abs_at_least_observed_frac']:.3f}"
    )

    return {
        "bootstrap_samples_per_draw": PLACEBO_SAMPLES,
        "placebo_draws": PLACEBO_DRAWS,
        "redteam_seed_base": int(args.seed),
        "instrument_note": (
            "scripts/bye_overvaluation_screen.py build_bye_maps sorts each team's "
            "games by gameday ACROSS seasons, so every season opener inherits a "
            ">=12-day gap from the prior season's finale and counts as off-bye "
            f"({cross_season_openers} opener team-games flagged by the cross-season "
            "map that the within-season map does not flag). The fade-full-slate cell "
            "is insulated because week-1 blocks can only produce BOTH-off-bye games, "
            "never an XOR edge; the both-bye sanity cell is NOT insulated. This audit "
            "rebuilds all bye flags within season before sham assignment."
        ),
        "multi_gap_team_seasons_clipped_to_first": len(multi_gap_team_seasons),
        "cross_season_map_resample_redteam_bye_overval_fade_full_slate_post2011": cross_repro,
        "independent_reproduction_redteam_bye_overval_fade_full_slate_post2011": repro,
        "attack_sham_bye_placebo": placebo_payload,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-game-artifact", type=Path, default=DEFAULT_PER_GAME_ARTIFACT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--features", type=Path, default=Path("data/processed/game_features_pbp.parquet")
    )
    parser.add_argument(
        "--incidents",
        type=Path,
        default=Path("data/raw/player_arrests/20260820T153000Z/incidents_point_in_time.parquet"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=REDTEAM_SEED)
    parser.add_argument("--claims", type=str, default="1,2,3,4")
    args = parser.parse_args(argv)

    wanted = {int(c) for c in args.claims.split(",")}
    payload: dict[str, Any] = {
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "mission": (
            "adversarial red-team audit of four recent edge claims with different "
            "seeds/blocking/estimators than their original screens; attribution on "
            "already-scored archive data and fresh-seed resamples only; no window spend"
        ),
        "redteam_seed_base": int(args.seed),
    }
    if 1 in wanted:
        payload["claim1_overlay_composition"] = run_claim1(args)
    if 2 in wanted:
        payload["claim2_nflcom_friday_out_count_ge2"] = run_claim2(args)
    if 3 in wanted:
        payload["claim3_body_clock_night_west_road"] = run_claim3(args)
    if 4 in wanted:
        payload["claim4_bye_fade_post2011"] = run_claim4(args)

    configuration = {
        "command": "edge-audit-redteam",
        "schedules": str(bye_overvaluation_screen.DEFAULT_SCHEDULES),
        "per_game_artifact": str(args.per_game_artifact),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "claims": sorted(wanted),
        "permutations": PERMUTATIONS,
        "placebo_draws": PLACEBO_DRAWS,
    }
    payload["provenance"] = artifact_provenance(
        configuration, bye_overvaluation_screen.DEFAULT_SCHEDULES, project_root=REPO
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="edge-audit-redteam",
        metrics=payload,
        notes=(
            "Adversarial red-team audit of four recent edge claims (overlay "
            "composition holdout, nfl.com Friday out_count>=2, night body-clock "
            "west-road, post-2011 bye fade) using different seeds/blocking/estimators; "
            "attribution on already-scored archive data and fresh-seed resamples only, "
            "no rotation-registry window spend. New cells carry the redteam_ prefix; "
            "recording happens via explicit nfl-ats weak-signals record calls returned "
            "to the owner. Never pool redteam_ decompositions with their parent "
            "signals as independent."
        ),
    )
    print(f"Wrote {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
