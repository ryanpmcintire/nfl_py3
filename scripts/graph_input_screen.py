"""Two-gate + redundancy screen for candidate graph-rating inputs (owner-directed
exploration, 2026-08-26). Predeclared in ``docs/graph_input_screen.md`` --
read that file for the full design and its justification before touching this
script's constants.

**Binding taxonomy (owned verbatim, per AGENTS.md/CLAUDE.md)**: an interval
or CI that contains zero is NEVER grounds to reject, fail, or close an
experiment -- at this evaluator's ~2-point resolution "contains zero" is the
EXPECTED outcome for a real small signal. Only two closing grounds: (1)
refuted mechanism -- RESOLVED wrong sign (whole interval on the wrong side
of zero) or zero/negative split-half reliability (``reliability <= 0.10``);
(2) bounded by a positive control proven able to detect an effect that size.
Everything else is ``unresolved_below_power``: record with
``probability_positive``, never "contains zero". If a record command errors,
the verdict is wrong, not the validator.

**Gate 1** (does it repeat) reuses
``artifacts/reliability_map/20260826T112507Z/results.json`` unchanged.
**Gate 2** (does it add anything the market does not already have) compares
a zero-feature market baseline (``nfl_ats.margin.fit_market_baseline``,
reused unmodified) against a single-feature ridge on each family's
standardized home-minus-away differential, weekly-refit walk-forward, graded
with production's probability rule, on a predeclared SELECTION window
(2013-2019, close-graded) used only to rank, and a predeclared HOLDOUT window
(2020-2025, opener-graded via ``nfl_ats.clv.build_pairing_table``) whose
delta is the decision-relevant number (rule: grade the decision at the
opener). **Gate 3** clusters survivors by feature-feature correlation.

This script does not modify ``margin.py``, ``constants.py``, or ``clv.py`` --
every reused function is imported, not copied, except a small wrapper that
mirrors ``fit_margin_model`` but accepts an explicit column list instead of a
registered ``MarginFeatureProfile`` (see the module docstring's declared
simplifications).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import reliability_map as relmap  # noqa: E402

from nfl_ats.clv import (  # noqa: E402
    CLOSE_LABEL_PRIORITY,
    HISTORICAL_CAPTURE_KIND,
    build_pairing_table,
    close_reference_table,
    pick_correct,
    week_blocked_bootstrap,
)
from nfl_ats.constants import MIN_FITTABLE_TRAIN_GAMES  # noqa: E402
from nfl_ats.margin import MarginModel, fit_market_baseline, make_margin_estimator  # noqa: E402
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.weak_signals import (  # noqa: E402
    CLOSING_GROUNDS,
    WeakSignal,
    default_registry_path,
    load_registry,
    record_signal,
    save_registry,
)

SEED = 20260826
BOOTSTRAP_SAMPLES = 1000
RIDGE_ALPHA = 10.0
DISTRIBUTION_FRACTION = 0.20
MIN_DISTRIBUTION_ROWS = 10
REDUNDANCY_THRESHOLD = 0.6  # |r| >= this co-clusters two families

SELECTION_SEASONS = tuple(range(2013, 2020))
HOLDOUT_SEASONS = tuple(range(2020, 2026))

RELIABILITY_ARTIFACT = REPO / "artifacts" / "reliability_map" / "20260826T112507Z" / "results.json"
MARKET_ROOT = REPO / "data" / "market" / "raw"

NO_SPLIT_HALF_RELIABILITY_MAX = 0.10

# Explicit per-family category mapping (CATEGORIES from weak_signals.py) --
# every one of the 83 discovered families accounted for, nothing left to a
# silent default. market/onfield/health/schedule/environment/attention/
# offfield/modeling/control per that module's own definitions.
FAMILY_CATEGORY: dict[str, str] = {
    "active_roster_continuity": "onfield",
    "active_roster_mean_experience": "onfield",
    "ats_residual": "market",
    "bias_playoff_holdover": "schedule",
    "bias_prior_week_ats": "market",
    "bias_week2_anchor": "schedule",
    "def_epa_per_play": "onfield",
    "def_pass_epa_per_play": "onfield",
    "def_rush_epa_per_play": "onfield",
    "def_sack_rate": "onfield",
    "def_takeaway_rate": "onfield",
    "def_yards_per_play": "onfield",
    "defense_lineup_continuity": "onfield",
    "drive_plays_per_drive": "onfield",
    "drive_plays_per_drive_allowed": "onfield",
    "drive_points_per_drive": "onfield",
    "drive_points_per_drive_allowed": "onfield",
    "drive_scoring_rate": "onfield",
    "drive_scoring_rate_allowed": "onfield",
    "drive_seconds_per_drive": "onfield",
    "drive_seconds_per_drive_allowed": "onfield",
    "drive_takeaway_rate": "onfield",
    "drive_turnover_rate": "onfield",
    "drive_yards_per_drive": "onfield",
    "drive_yards_per_drive_allowed": "onfield",
    "front_lineup_continuity": "onfield",
    "gap_division_revenge": "schedule",
    "gap_post_blowout_loss_bounce": "schedule",
    "gap_post_blowout_win_letdown": "schedule",
    "gap_sandwich_spot": "schedule",
    "graph_defense": "onfield",
    "graph_offense": "onfield",
    "graph_pagerank": "onfield",
    "injury_defense_disruption_value_lost": "health",
    "injury_defense_unavailability": "health",
    "injury_front_unavailability": "health",
    "injury_offense_unavailability": "health",
    "injury_offensive_line_unavailability": "health",
    "injury_secondary_unavailability": "health",
    "injury_skill_epa_value_lost": "health",
    "injury_skill_unavailability": "health",
    "injury_special_teams_unavailability": "health",
    "off_cpoe": "onfield",
    "off_epa_per_play": "onfield",
    "off_pass_epa_per_play": "onfield",
    "off_rush_epa_per_play": "onfield",
    "off_sack_rate": "onfield",
    "off_turnover_rate": "onfield",
    "off_yards_per_play": "onfield",
    "offense_lineup_continuity": "onfield",
    "offensive_line_continuity": "onfield",
    "pbp_def_early_down_epa": "onfield",
    "pbp_def_epa_per_play": "onfield",
    "pbp_def_explosive_rate_allowed": "onfield",
    "pbp_def_success_rate_allowed": "onfield",
    "pbp_drives": "onfield",
    "pbp_matchup_early_down_epa": "onfield",
    "pbp_matchup_epa_per_play": "onfield",
    "pbp_matchup_explosive_rate": "onfield",
    "pbp_matchup_pressure_allowed_rate": "onfield",
    "pbp_matchup_sack_allowed_rate": "onfield",
    "pbp_matchup_success_rate": "onfield",
    "pbp_off_early_down_epa": "onfield",
    "pbp_off_epa_per_play": "onfield",
    "pbp_off_explosive_rate": "onfield",
    "pbp_off_pass_rate": "onfield",
    "pbp_off_proe": "onfield",
    "pbp_off_success_rate": "onfield",
    "pbp_pressure_allowed_rate": "onfield",
    "pbp_pressure_rate": "onfield",
    "pbp_sack_allowed_rate": "onfield",
    "pbp_sack_rate": "onfield",
    "pbp_start_yardline_100": "onfield",
    "qb_expected_epa_per_dropback": "onfield",
    "qb_start_probability": "onfield",
    "qb_starter_cpoe": "onfield",
    "qb_starter_epa_per_dropback": "onfield",
    "qb_starter_experience_log": "onfield",
    "schedule_rating": "onfield",
    "secondary_lineup_continuity": "onfield",
    "skill_lineup_continuity": "onfield",
    "special_teams_lineup_continuity": "onfield",
    "team_games": "onfield",
}


def fit_single_feature_market_residual_model(
    frame: pd.DataFrame,
    feature_column: str,
    *,
    ridge_alpha: float = RIDGE_ALPHA,
    distribution_fraction: float = DISTRIBUTION_FRACTION,
    min_distribution_rows: int = MIN_DISTRIBUTION_ROWS,
    random_state: int = 42,
) -> MarginModel:
    """``fit_margin_model`` with an explicit single feature column instead of a
    registered ``MarginFeatureProfile`` -- see the module docstring's declared
    simplifications. Mirrors that function's logic line for line.
    """

    required = {"game_id", "gameday", "result", "ats_margin", feature_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Margin training is missing columns: {', '.join(missing)}")

    training = regular_season_rows(frame)
    training = training.loc[training["ats_margin"].notna()].copy()
    training["gameday"] = pd.to_datetime(training["gameday"], errors="raise")
    training = training.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    if len(training) < MIN_FITTABLE_TRAIN_GAMES:
        raise ValueError("Not enough completed games for a margin model")
    distribution_rows = int(len(training) * distribution_fraction)
    if distribution_rows < min_distribution_rows or len(training) - distribution_rows < 40:
        raise ValueError("Not enough rows for an out-of-time residual distribution")

    split = len(training) - distribution_rows
    fit_part = training.iloc[:split]
    distribution_part = training.iloc[split:]
    cols = [feature_column]
    temporary = make_margin_estimator("ridge", random_state, ridge_alpha=ridge_alpha)
    temporary.fit(fit_part.loc[:, cols], fit_part["ats_margin"])
    calibration_prediction = np.asarray(
        temporary.predict(distribution_part.loc[:, cols]), dtype=float
    )
    residuals = np.asarray(
        distribution_part["ats_margin"].to_numpy(dtype=float) - calibration_prediction,
        dtype=np.float64,
    )
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < min_distribution_rows:
        raise ValueError("Out-of-time residual distribution has too few finite values")

    estimator = make_margin_estimator("ridge", random_state, ridge_alpha=ridge_alpha)
    estimator.fit(training.loc[:, cols], training["ats_margin"])
    return MarginModel(
        estimator=estimator,
        residuals=residuals,
        model_name="ridge",
        ridge_alpha=ridge_alpha,
        target="market_residual",
        feature_columns=(feature_column,),
        training_rows=len(training),
        distribution_rows=len(residuals),
        training_max_gameday=training["gameday"].max().date().isoformat(),
    )


def load_features_with_diffs() -> tuple[pd.DataFrame, dict[str, tuple[str, str, str]]]:
    """Reuse ``reliability_map``'s own loader/discovery, then add one
    standardized-nowhere-yet ``diff_<family>`` column per discovered family
    (the ridge pipeline standardizes internally; this is just the raw
    home-minus-away differential).
    """

    features = relmap.load_feature_table()
    dtypes = {c: features[c].dtype for c in features.columns}
    families, _excluded = relmap.discover_family_pairs(list(features.columns), dtypes)
    diff_columns = {
        f"screen_diff_{name}": pd.to_numeric(features[home_col], errors="coerce")
        - pd.to_numeric(features[away_col], errors="coerce")
        for name, (home_col, away_col, _pattern) in families.items()
    }
    features = pd.concat([features, pd.DataFrame(diff_columns, index=features.index)], axis=1)
    return features, families


def run_close_graded_window(
    features: pd.DataFrame,
    families: dict[str, tuple[str, str, str]],
    seasons: tuple[int, ...],
) -> dict[str, pd.DataFrame]:
    """SELECTION window: paired per-game rows graded at the feature table's own
    (closing) ``spread_line``, for every family. Returns ``{family: frame}``
    with columns ``game_id, season, week, baseline_correct, candidate_correct``.
    """

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    completed = frame.loc[frame["result"].notna()].copy()
    window = completed.loc[completed["season"].astype(int).isin(seasons)]

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in families}
    n_weeks = 0
    for (_season, _week), group in window.groupby(["season", "week"], sort=True):
        cutoff = group["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < MIN_FITTABLE_TRAIN_GAMES:
            continue
        n_weeks += 1
        baseline = fit_market_baseline(training)
        baseline_pred = baseline.predict(group)
        settle_margin = pd.to_numeric(group["result"], errors="coerce") - pd.to_numeric(
            group["spread_line"], errors="coerce"
        )
        baseline_correct = pick_correct(
            baseline_pred["home_cover_probability"].ge(0.5), settle_margin
        )
        for name in families:
            diff_col = f"screen_diff_{name}"
            try:
                model = fit_single_feature_market_residual_model(training, diff_col)
            except ValueError:
                continue
            cand_pred = model.predict(group)
            cand_correct = pick_correct(cand_pred["home_cover_probability"].ge(0.5), settle_margin)
            for game_id, s_val, w_val, bc, cc in zip(
                group["game_id"],
                group["season"],
                group["week"],
                baseline_correct,
                cand_correct,
                strict=True,
            ):
                rows[name].append(
                    {
                        "game_id": game_id,
                        "season": int(str(s_val)),
                        "week": int(str(w_val)),
                        "baseline_correct": bc,
                        "candidate_correct": cc,
                    }
                )
    print(f"  close-graded selection window: {n_weeks} weeks fitted")
    return {name: pd.DataFrame(r) for name, r in rows.items()}


def run_opener_graded_window(
    features: pd.DataFrame,
    families: dict[str, tuple[str, str, str]],
    seasons: tuple[int, ...],
) -> dict[str, pd.DataFrame]:
    """HOLDOUT window: paired per-game rows graded at the archived Tuesday
    opener, reusing ``build_pairing_table``/``close_reference_table`` exactly
    as ``nfl_ats.clv.opener_pick_evaluation`` does.
    """

    features_reg = regular_season_rows(features)
    pairing = build_pairing_table(
        MARKET_ROOT,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=features_reg,
    )
    if pairing.empty:
        raise ValueError(f"No {HISTORICAL_CAPTURE_KIND!r} snapshots with decision quotes")
    close = close_reference_table(pairing, features_reg)
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "season", "week", "home_spread", "spread_books"]
    ].rename(columns={"home_spread": "tue_open_home_spread", "spread_books": "opener_books"})
    paired = tue_open.merge(close, on="game_id", how="inner")
    outcomes = features_reg[["game_id", "result"]].drop_duplicates("game_id")
    paired = paired.merge(outcomes, on="game_id", how="inner")
    paired = paired.loc[pd.to_numeric(paired["result"], errors="coerce").notna()].copy()
    paired = paired.loc[paired["season"].astype(int).isin(seasons)]
    if paired.empty:
        raise ValueError("No paired opener/close games in the holdout seasons")

    frame = features_reg.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[frame["result"].notna()].copy()

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in families}
    n_weeks = 0
    for (season, week), group in paired.groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"]))]
        if week_rows.empty:
            continue
        cutoff = week_rows["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < MIN_FITTABLE_TRAIN_GAMES:
            continue
        n_weeks += 1
        baseline = fit_market_baseline(training)
        scoring = week_rows.merge(
            group[["game_id", "tue_open_home_spread", "close_home_spread"]],
            on="game_id",
            how="inner",
        ).copy()
        at_open = scoring.copy()
        at_open["spread_line"] = at_open["tue_open_home_spread"]
        margin_vs_open = pd.to_numeric(scoring["result"], errors="coerce") - pd.to_numeric(
            scoring["tue_open_home_spread"], errors="coerce"
        )
        baseline_pred_open = baseline.predict(at_open)
        baseline_correct_open = pick_correct(
            baseline_pred_open["home_cover_probability"].ge(0.5), margin_vs_open
        )
        for name in families:
            diff_col = f"screen_diff_{name}"
            try:
                model = fit_single_feature_market_residual_model(training, diff_col)
            except ValueError:
                continue
            cand_pred_open = model.predict(at_open)
            cand_correct_open = pick_correct(
                cand_pred_open["home_cover_probability"].ge(0.5), margin_vs_open
            )
            for game_id, bc, cc in zip(
                scoring["game_id"], baseline_correct_open, cand_correct_open, strict=True
            ):
                rows[name].append(
                    {
                        "game_id": game_id,
                        "season": int(str(season)),
                        "week": int(str(week)),
                        "baseline_correct": bc,
                        "candidate_correct": cc,
                    }
                )
    print(f"  opener-graded holdout window: {n_weeks} weeks fitted")
    return {name: pd.DataFrame(r) for name, r in rows.items()}


def _delta_metric(df: pd.DataFrame) -> dict[str, float]:
    valid = df.dropna(subset=["baseline_correct", "candidate_correct"])
    if valid.empty:
        return {
            "delta_accuracy": float("nan"),
            "candidate_accuracy": float("nan"),
            "baseline_accuracy": float("nan"),
        }
    return {
        "delta_accuracy": float((valid["candidate_correct"] - valid["baseline_correct"]).mean()),
        "candidate_accuracy": float(valid["candidate_correct"].mean()),
        "baseline_accuracy": float(valid["baseline_correct"].mean()),
    }


def summarize_window(paired: pd.DataFrame) -> dict[str, Any] | None:
    if paired.empty or paired.dropna(subset=["baseline_correct", "candidate_correct"]).empty:
        return None
    point = _delta_metric(paired)
    week_boot = week_blocked_bootstrap(
        paired, _delta_metric, block="week", samples=BOOTSTRAP_SAMPLES, seed=SEED
    )
    season_boot = week_blocked_bootstrap(
        paired, _delta_metric, block="season", samples=BOOTSTRAP_SAMPLES, seed=SEED
    )
    week_row = week_boot.loc[week_boot["metric"].eq("delta_accuracy")].iloc[0]
    season_row = season_boot.loc[season_boot["metric"].eq("delta_accuracy")].iloc[0]
    n_games = len(paired.dropna(subset=["baseline_correct", "candidate_correct"]))
    n_weeks = int(paired[["season", "week"]].drop_duplicates().shape[0])
    n_seasons = int(paired["season"].nunique())
    return {
        "delta_accuracy": point["delta_accuracy"],
        "candidate_accuracy": point["candidate_accuracy"],
        "baseline_accuracy": point["baseline_accuracy"],
        "week_blocked_ci95": [float(week_row["lower"]), float(week_row["upper"])],
        "week_blocked_probability_positive": float(week_row["probability_positive"]),
        "season_blocked_ci95": [float(season_row["lower"]), float(season_row["upper"])],
        "season_blocked_probability_positive": float(season_row["probability_positive"]),
        "n_games": n_games,
        "n_weeks": n_weeks,
        "n_seasons": n_seasons,
    }


def cluster_families(
    features: pd.DataFrame, family_names: list[str], *, threshold: float
) -> tuple[dict[str, int], list[list[str]]]:
    """Hierarchical clustering of surviving families by |correlation| of their
    standardized diff columns across the full REG-season archive.
    """

    reg = regular_season_rows(features)
    cols = [f"screen_diff_{name}" for name in family_names]
    matrix = reg[cols].apply(pd.to_numeric, errors="coerce")
    corr = matrix.corr(method="pearson", min_periods=30).to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    distance = 1.0 - np.abs(corr)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    if len(family_names) < 2:
        cluster_ids = {family_names[0]: 1} if family_names else {}
        return cluster_ids, [family_names] if family_names else []
    Z = linkage(condensed, method="average")
    cut_distance = 1.0 - threshold
    labels = fcluster(Z, t=cut_distance, criterion="distance")
    cluster_ids = {name: int(label) for name, label in zip(family_names, labels, strict=True)}
    clusters: dict[int, list[str]] = {}
    for name, label in cluster_ids.items():
        clusters.setdefault(label, []).append(name)
    return cluster_ids, list(clusters.values())


def main() -> None:
    started = time.time()
    print("=== loading reliability map ===")
    import json

    reliability_payload = json.loads(RELIABILITY_ARTIFACT.read_text(encoding="utf-8"))
    reliability_by_family = {r["metric"]: r for r in reliability_payload["results"]}

    print("=== loading features + diffs ===")
    features, families = load_features_with_diffs()
    family_names = sorted(families)
    print(f"families: {len(family_names)}")
    missing_category = sorted(set(family_names) - set(FAMILY_CATEGORY))
    if missing_category:
        raise SystemExit(f"FAMILY_CATEGORY is missing: {missing_category}")

    print("=== Gate 1: resolve closures from the reliability map ===")
    gate1_closed: dict[str, str] = {}
    for name in family_names:
        rel = reliability_by_family[name]
        sb = rel["spearman_brown_full_length_reliability"]
        ci = rel["pearson_r_ci95"]
        resolved_below_zero = ci[1] is not None and ci[1] < 0.0
        if sb is not None and sb <= NO_SPLIT_HALF_RELIABILITY_MAX and resolved_below_zero:
            gate1_closed[name] = "no_split_half_reliability"
    print(f"Gate 1 closed: {gate1_closed}")

    print("=== Gate 2: selection window (2013-2019, close-graded) ===")
    selection_windows = run_close_graded_window(features, families, SELECTION_SEASONS)
    print("=== Gate 2: holdout window (2020-2025, opener-graded) ===")
    holdout_windows = run_opener_graded_window(features, families, HOLDOUT_SEASONS)

    gate2: dict[str, dict[str, Any]] = {}
    for name in family_names:
        selection_summary = summarize_window(selection_windows[name])
        holdout_summary = summarize_window(holdout_windows[name])
        gate2[name] = {"selection": selection_summary, "holdout": holdout_summary}
        sel_p = (
            selection_summary["week_blocked_probability_positive"]
            if selection_summary
            else float("nan")
        )
        hold_p = (
            holdout_summary["week_blocked_probability_positive"]
            if holdout_summary
            else float("nan")
        )
        print(f"  {name}: selection P+={sel_p:.3f} holdout P+={hold_p:.3f}")

    print("=== Gate 3: redundancy clustering ===")
    survivors = [name for name in family_names if name not in gate1_closed]
    cluster_ids, clusters = cluster_families(features, survivors, threshold=REDUNDANCY_THRESHOLD)

    def representative_key(name: str) -> tuple[float, float, str]:
        holdout = gate2[name]["holdout"]
        p_plus = holdout["week_blocked_probability_positive"] if holdout else -1.0
        reliability = (
            reliability_by_family[name]["spearman_brown_full_length_reliability"] or -999.0
        )
        return (-p_plus, -reliability, name)

    representatives: dict[int, str] = {}
    for members in clusters:
        cluster_label = cluster_ids[members[0]]
        representatives[cluster_label] = sorted(members, key=representative_key)[0]

    print(f"clusters: {len(clusters)} from {len(survivors)} survivors")

    # ------------------------------------------------------------------
    # Assemble the per-family report
    # ------------------------------------------------------------------
    per_family: dict[str, Any] = {}
    for name in family_names:
        rel = reliability_by_family[name]
        entry: dict[str, Any] = {
            "family": name,
            "category": FAMILY_CATEGORY[name],
            "reliability": rel["spearman_brown_full_length_reliability"],
            "reliability_pearson_r_ci95": rel["pearson_r_ci95"],
            "reliability_probability_positive": rel["probability_positive"],
            "gate2": gate2[name],
        }
        if name in gate1_closed:
            entry["status"] = "closed_gate1"
            entry["closing_ground"] = gate1_closed[name]
            entry["cluster"] = None
            entry["is_cluster_representative"] = False
        else:
            entry["status"] = "survivor"
            entry["closing_ground"] = None
            entry["cluster"] = cluster_ids[name]
            entry["is_cluster_representative"] = representatives.get(cluster_ids[name]) == name
        per_family[name] = entry

    ranked_survivors = sorted(
        survivors,
        key=lambda n: (
            -(
                gate2[n]["holdout"]["week_blocked_probability_positive"]
                if gate2[n]["holdout"]
                else -1.0
            )
        ),
    )

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO / "artifacts" / "graph_input_screen" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    configuration = {
        "command": "graph-input-screen",
        "seed": SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "ridge_alpha": RIDGE_ALPHA,
        "distribution_fraction": DISTRIBUTION_FRACTION,
        "selection_seasons": list(SELECTION_SEASONS),
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
        "predeclaration": "docs/graph_input_screen.md",
        "reliability_artifact": str(RELIABILITY_ARTIFACT),
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        **configuration,
        "n_families": len(family_names),
        "n_gate1_closed": len(gate1_closed),
        "gate1_closed": gate1_closed,
        "n_survivors": len(survivors),
        "n_clusters": len(clusters),
        "clusters": [
            {
                "cluster_id": cluster_ids[members[0]],
                "members": sorted(members),
                "representative": representatives[cluster_ids[members[0]]],
            }
            for members in clusters
        ],
        "ranked_survivors": ranked_survivors,
        "families": per_family,
        "provenance": artifact_provenance(
            configuration,
            REPO / "data" / "processed" / "game_features_weak_stack_v4.parquet",
            project_root=REPO,
        ),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="graph-input-screen",
        metrics={
            "n_families": len(family_names),
            "n_gate1_closed": len(gate1_closed),
            "n_survivors": len(survivors),
            "n_clusters": len(clusters),
        },
        notes=(
            "Two-gate + redundancy screen for candidate graph-rating inputs; "
            "see docs/graph_input_screen.md for the predeclared design."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")

    # ------------------------------------------------------------------
    # Record every family via the same validated path the CLI uses
    # ------------------------------------------------------------------
    print("=== recording weak signals ===")
    registry_path = default_registry_path()
    registry = load_registry(registry_path)
    recorded_at = time.strftime("%Y-%m-%d", time.gmtime())
    for name in family_names:
        entry = per_family[name]
        holdout = entry["gate2"]["holdout"]
        reliability = entry["reliability"]
        if entry["status"] == "closed_gate1":
            classification = "refuted_mechanism"
            closing_ground = entry["closing_ground"]
            evidence = (
                f"Gate 1 (split-half reliability, {RELIABILITY_ARTIFACT.name}): "
                f"Spearman-Brown reliability {reliability:.4f}, pearson_r_ci95 "
                f"{entry['reliability_pearson_r_ci95']} resolved entirely below "
                f"zero and reliability <= {NO_SPLIT_HALF_RELIABILITY_MAX}."
            )
        elif holdout is not None and holdout["week_blocked_ci95"][1] < 0.0:
            classification = "refuted_mechanism"
            closing_ground = "wrong_sign_resolved"
            evidence = (
                "Gate 2 holdout (opener-graded, 2020-2025): week-blocked 95% CI "
                f"{holdout['week_blocked_ci95']} entirely below zero for the "
                "candidate-minus-baseline accuracy delta."
            )
        else:
            classification = "unresolved_below_power"
            closing_ground = None
            evidence = ""

        if holdout is not None:
            effect = holdout["delta_accuracy"] * 100.0
            interval = (
                holdout["week_blocked_ci95"][0] * 100.0,
                holdout["week_blocked_ci95"][1] * 100.0,
            )
            probability_positive = holdout["week_blocked_probability_positive"]
            sample_games = holdout["n_games"]
            sample_blocks = holdout["n_weeks"]
        else:
            effect = 0.0
            interval = None
            probability_positive = None
            sample_games = None
            sample_blocks = None

        selection = entry["gate2"]["selection"]
        notes_parts = [f"Gate 1 reliability {reliability}."]
        if selection is not None:
            notes_parts.append(
                f"Selection window (close-graded, 2013-2019) delta_accuracy "
                f"{selection['delta_accuracy'] * 100.0:+.3f} pts, week-blocked P+ "
                f"{selection['week_blocked_probability_positive']:.3f}, "
                f"n={selection['n_games']}."
            )
        if holdout is not None:
            notes_parts.append(
                f"Holdout window (opener-graded, 2020-2025) is the decision-"
                f"relevant figure recorded as effect/interval above, n="
                f"{holdout['n_games']}."
            )
        else:
            notes_parts.append("Holdout window produced no usable paired games.")

        signal = WeakSignal(
            name=f"graph_input_screen_{name}",
            recorded_at=recorded_at,
            description=(
                f"Graph-rating input screen: single-feature ridge on the "
                f"standardized home-minus-away {name} differential vs a "
                f"zero-feature market baseline, "
                f"probability-rule accuracy, weekly-refit walk-forward."
            ),
            source=str(output_dir / "results.json"),
            effect=float(effect),
            effect_units="accuracy_points",
            classification=classification,
            league="nfl",
            seasons=(HOLDOUT_SEASONS[0], HOLDOUT_SEASONS[-1]),
            interval=interval,
            probability_positive=probability_positive,
            sample_games=sample_games,
            sample_blocks=sample_blocks,
            reliability=reliability,
            family="graph_input_screen",
            classification_evidence=evidence,
            closing_ground=closing_ground,
            notes=" ".join(notes_parts),
            plain_summary=f"Tests whether {name} beats the market alone at picking covers.",
            category=FAMILY_CATEGORY[name],
        )
        registry = record_signal(registry, signal, replace=True)
    save_registry(registry, registry_path)
    print(f"recorded {len(family_names)} signals to {registry_path}")

    admissible = CLOSING_GROUNDS  # noqa: F841 (imported for the predeclared-taxonomy cross-check)
    print(f"\ndone in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
