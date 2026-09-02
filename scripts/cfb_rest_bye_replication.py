"""NFL rest / bye mechanisms, replicated on COLLEGE FOOTBALL.

Predeclared in ``docs/cfb_rest_bye_replication.md`` BEFORE this script was
pointed at any outcome column. Read that document first -- it freezes the
population, the four cells and their NFL sources, the per-side rest derivation,
the comparator, the null, the positive control, the era split and the recording
rules.

**The comparison this harness runs, stated once and plainly**: the frozen
XLG-03 benchmark contract ``nfl_ats.cfb_features.CFB_MODEL_FEATURE_COLUMNS``
ALREADY carries ``rest_diff``, so the baseline arm already prices the linear
rest difference. Every cell scored here is a MARGINAL on top of that -- the
project's own "composition is not the signal" discipline: evaluate on top of
what is played, not on top of a bare baseline.

Cells (``--cell``), each the NFL construct with the league swapped:

* ``home_off_bye``     -- NFL ``travel_rest_home_off_bye``  (home rest >= 13)
* ``away_off_bye``     -- NFL ``travel_rest_away_off_bye``  (away rest >= 13)
* ``bye_edge_home``    -- NFL ``bye_overval_home_edge_post2011``
  (home off a strict >=12-day bye AND opponent NOT off bye)
* ``short_week_road``  -- NFL ``travel_rest_short_week_road`` (away rest <= 5)

plus three DECLARED SENSITIVITY ARMS, frozen with the primaries and never
substituted for them: ``home_off_bye_gap12``, ``away_off_bye_gap12``,
``short_week_road_le6``.

Modes (``--mode``):

* ``coverage``         -- PREDICTOR-ONLY diagnostics (per-season cell coverage,
  rest histograms, the reproduce-``rest_diff`` check, split-half reliability on
  two instruments). Touches no outcome column; this is what freezes the scored
  season set in the predeclaration.
* ``null``             -- settle margins shuffled within each week. A harness
  that reports a real effect here is broken.
* ``positive-control`` -- the candidate's one new column is replaced by the
  realised ``ats_margin``, a deliberate leak. A harness that cannot detect this
  inside the full 36-column ridge fit would be blind.
* ``screen``           -- the real look. No NFL evaluation window and no
  rotation window is spent: CFB is this project's sanctioned free cross-league
  replication ground.

**Overlap disclosure, carried in every artifact and every registry note.**
``registry/weak_signals.json`` already holds ``cfb_bias_battery_bye_week_rest_edge``
(rest edge >= 6 days) and ``cfb_bias_battery_short_week_rest_disadvantage``
(rest edge <= -4 days) from ``scripts/cfb_bias_battery_screen.py``. Those are
SUBSET COVER RATE VS COMPLEMENT measurements on an overlapping population --
a different quantity from this harness's paired accuracy delta of the XLG-03
estimator with one extra column. AGENTS.md's commensurability rule forbids
pooling non-commensurable comparators, so ``cfb_rest_bye_replication`` is a
separate pooling family and is NEVER pooled with ``cfb_bias_battery``.

Closing-grounds taxonomy (binding, restated per AGENTS.md so this file stands
on its own): an interval containing zero is NEVER grounds to reject, fail or
close an experiment. Only a RESOLVED wrong sign (whole interval on the wrong
side of zero), zero split-half reliability, or a positive control proven able
to detect an effect that size closes a line of work. Everything else is
``unresolved_below_power``: record it and report ``probability_positive``,
never the binary "contains zero". If a record command errors, the verdict is
wrong, not the validator. This run is CLOSE-graded (CFB has no verified opener
-- ``docs/cfb_data.md``), and a CFB result never by itself changes an NFL card.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nfl_ats.cfb_benchmark import (  # noqa: E402
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_CLEAN_CORE_SEASONS,
    fit_cfb_residual_model,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS  # noqa: E402
from nfl_ats.cfb_qb_dependence import split_half_reliability  # noqa: E402
from nfl_ats.cfb_rest_bye_feature import (  # noqa: E402
    CFB_AWAY_OFF_BYE_COLUMN,
    CFB_AWAY_OFF_BYE_GAP12_COLUMN,
    CFB_BYE_EDGE_HOME_COLUMN,
    CFB_HOME_OFF_BYE_COLUMN,
    CFB_HOME_OFF_BYE_GAP12_COLUMN,
    CFB_REST_CELL_PANEL_METRIC,
    CFB_REST_PANEL_METRICS,
    CFB_SHORT_WEEK_ROAD_COLUMN,
    CFB_SHORT_WEEK_ROAD_LE6_COLUMN,
    attach_cfb_rest_bye_features,
)
from nfl_ats.clv import pick_correct, week_blocked_bootstrap  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

DEFAULT_FEATURES = REPO_ROOT / "data" / "processed" / "cfb_game_features.parquet"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "cfb_rest_bye_replication"

#: docs/cfb_rest_bye_replication.md section 7 -- 1,000 samples for comparability
#: with the sibling CFB replication harnesses, seed = today's date per repo
#: convention.
BOOTSTRAP_SAMPLES = 1_000
SEED = 20260901
PERMUTATIONS = 200

#: docs/cfb_rest_bye_replication.md section 7 -- the benchmark's own declared
#: 2020 regime gap is the boundary, the same one the FluView CFB replication
#: uses. Magnitudes are reported per era and NEVER averaged across a sign flip
#: (owner rule "era magnitude, not presence").
ERAS: tuple[tuple[str, int, int], ...] = (("2012_2019", 2012, 2019), ("2021_2025", 2021, 2025))

OVERLAP_DISCLOSURE = (
    "Separate pooling family cfb_rest_bye_replication; NEVER pooled with "
    "cfb_bias_battery. registry/weak_signals.json's "
    "cfb_bias_battery_bye_week_rest_edge (rest edge >=6d) and "
    "cfb_bias_battery_short_week_rest_disadvantage (rest edge <=-4d) are SUBSET "
    "COVER RATE VS COMPLEMENT measurements on an overlapping CFB population -- a "
    "different quantity from this harness's paired accuracy delta of the XLG-03 "
    "estimator with exactly one extra column. AGENTS.md's commensurability rule "
    "forbids pooling non-commensurable comparators."
)

CELLS: dict[str, dict[str, str]] = {
    "home_off_bye": {
        "column": CFB_HOME_OFF_BYE_COLUMN,
        "role": "primary",
        "nfl_sibling": "travel_rest_home_off_bye",
        "predicted_direction": "positive home_cover edge (extra home preparation time)",
    },
    "away_off_bye": {
        "column": CFB_AWAY_OFF_BYE_COLUMN,
        "role": "primary",
        "nfl_sibling": "travel_rest_away_off_bye",
        "predicted_direction": "negative home_cover edge (extra-rested visitor)",
    },
    "bye_edge_home": {
        "column": CFB_BYE_EDGE_HOME_COLUMN,
        "role": "primary",
        "nfl_sibling": "bye_overval_home_edge_post2011",
        "predicted_direction": "negative home_cover edge (market overprices the bye)",
    },
    "short_week_road": {
        "column": CFB_SHORT_WEEK_ROAD_COLUMN,
        "role": "primary",
        "nfl_sibling": "travel_rest_short_week_road",
        "predicted_direction": "positive home_cover edge (short rest plus travel compounds)",
    },
    "home_off_bye_gap12": {
        "column": CFB_HOME_OFF_BYE_GAP12_COLUMN,
        "role": "sensitivity",
        "nfl_sibling": "travel_rest_home_off_bye",
        "predicted_direction": "positive home_cover edge",
    },
    "away_off_bye_gap12": {
        "column": CFB_AWAY_OFF_BYE_GAP12_COLUMN,
        "role": "sensitivity",
        "nfl_sibling": "travel_rest_away_off_bye",
        "predicted_direction": "negative home_cover edge",
    },
    "short_week_road_le6": {
        "column": CFB_SHORT_WEEK_ROAD_LE6_COLUMN,
        "role": "sensitivity",
        "nfl_sibling": "travel_rest_short_week_road",
        "predicted_direction": "positive home_cover edge",
    },
}


# ---------------------------------------------------------------------------
# population
# ---------------------------------------------------------------------------


def load_population(features_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The XLG-03 benchmark table with the rest/bye candidate columns attached.

    Per-side rest is derived from the FULL CFB schedules snapshot, never from
    this filtered table -- a team's actual previous game is frequently absent
    from the benchmark subset (docs/cfb_rest_bye_replication.md section 4).
    """

    features = pd.read_parquet(features_path)
    features["gameday"] = pd.to_datetime(features["gameday"], errors="raise")
    attached, diagnostics = attach_cfb_rest_bye_features(features)
    attached = attached.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return attached, diagnostics


def covered_clean_core_seasons(diagnostics: dict[str, Any], column: str) -> tuple[int, ...]:
    """Clean-core seasons whose measured coverage for ``column`` exceeds zero.

    Predictor-only: the coverage fraction is computed from the candidate column
    alone, never from an outcome. The rule is declared in section 2 before it is
    measured, exactly as the FluView CFB replication declares its own.
    """

    coverage = diagnostics["coverage_by_season"][column]
    return tuple(
        season for season in CFB_CLEAN_CORE_SEASONS if float(coverage.get(str(season), 0.0)) > 0.0
    )


def reliability_readings(panel: pd.DataFrame, seed: int) -> dict[str, dict[str, Any]]:
    """Both predeclared split-half instruments, on every panel metric.

    ``within_season_odd_even_week`` is the repo's standard instrument
    (``nfl_ats.cfb_qb_dependence.split_half_reliability`` splits each
    team-season by odd/even WEEK). ``across_season_odd_even_year`` reuses the
    identical function on a re-framed panel -- one pooled "season", the calendar
    year in the ``week`` slot -- so the odd/even split falls between SEASONS of
    the same program instead of between weeks of one season.

    Section 7 declares why both are reported: a team's season contains a fixed
    number of days and a fixed number of games, so extra rest in one part of the
    calendar is arithmetically less rest elsewhere. That compositional
    constraint pushes ANY within-season split-half correlation negative for a
    schedule quantity, which is a property of the instrument, not evidence about
    the trait. The between-season instrument is free of it.
    """

    across = panel.assign(week=panel["season"].astype(int), season=0)
    return {
        "within_season_odd_even_week": {
            metric: split_half_reliability(panel, metric, seed=seed)
            for metric in CFB_REST_PANEL_METRICS
        },
        "across_season_odd_even_year": {
            metric: split_half_reliability(across, metric, seed=seed)
            for metric in CFB_REST_PANEL_METRICS
        },
    }


# ---------------------------------------------------------------------------
# the evaluator
# ---------------------------------------------------------------------------


def run_walk_forward(
    attached: pd.DataFrame,
    scored_seasons: tuple[int, ...],
    *,
    candidate_column: str,
    leak_treatment: bool,
) -> pd.DataFrame:
    """Per-week walk-forward rows for every scored season.

    Every scored week's two models are trained on all completed games in the
    WHOLE table that kicked off strictly before that week's own earliest
    kickoff, with the benchmark's own 500-game floor -- the forward-chaining
    ``nfl_ats.cfb_benchmark.cfb_walk_forward_benchmark`` performs, and the same
    per-week refit the sibling CFB replication harnesses perform.
    """

    completed = attached.loc[
        pd.to_numeric(attached["result"], errors="coerce").notna()
        & pd.to_numeric(attached["ats_margin"], errors="coerce").notna()
    ].copy()

    candidate_source = completed
    if leak_treatment:
        candidate_source = completed.copy()
        candidate_source[candidate_column] = pd.to_numeric(
            candidate_source["ats_margin"], errors="coerce"
        )

    candidate_columns = (*CFB_MODEL_FEATURE_COLUMNS, candidate_column)
    scored = completed.loc[completed["season"].astype(int).isin(scored_seasons)]

    rows: list[dict[str, Any]] = []
    for (season_value, week_value), group in scored.groupby(["season", "week"], sort=True):
        cutoff = group["gameday"].min()
        baseline_training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(baseline_training) < CFB_BENCHMARK_MIN_TRAIN_GAMES:
            continue
        candidate_training = candidate_source.loc[candidate_source["gameday"].lt(cutoff)]

        baseline_model = fit_cfb_residual_model(
            baseline_training,
            ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA,
            feature_columns=CFB_MODEL_FEATURE_COLUMNS,
        )
        candidate_model = fit_cfb_residual_model(
            candidate_training,
            ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA,
            feature_columns=candidate_columns,
        )
        candidate_scoring = (
            group
            if not leak_treatment
            else group.assign(
                **{candidate_column: pd.to_numeric(group["ats_margin"], errors="coerce")}
            )
        )
        settle_margin = pd.to_numeric(group["result"], errors="coerce") - pd.to_numeric(
            group["spread_line"], errors="coerce"
        )
        baseline_probability = baseline_model.predict(group)["home_cover_probability"]
        candidate_probability = candidate_model.predict(candidate_scoring)["home_cover_probability"]

        for game_id, margin, base, cand, feature_value in zip(
            group["game_id"],
            settle_margin,
            baseline_probability,
            candidate_probability,
            group[candidate_column],
            strict=True,
        ):
            rows.append(
                {
                    "game_id": game_id,
                    "season": int(str(season_value)),
                    "week": int(str(week_value)),
                    "settle_margin": margin,
                    "baseline_probability": base,
                    "candidate_probability": cand,
                    "feature_value": feature_value,
                }
            )
        print(
            f"  {int(str(season_value))} wk {int(str(week_value)):>2}: {len(group)} games, "
            f"train={len(baseline_training)}",
            flush=True,
        )
    return pd.DataFrame(rows)


ARM_PROBABILITY = {"baseline": "baseline_probability", "candidate": "candidate_probability"}


def grade(frame: pd.DataFrame, margins: pd.Series | None = None) -> pd.DataFrame:
    settle = frame["settle_margin"] if margins is None else margins
    graded = frame.copy()
    for arm, column in ARM_PROBABILITY.items():
        graded[f"{arm}_correct"] = pick_correct(graded[column].ge(0.5), settle)
    return graded


def week_positions(frame: pd.DataFrame) -> list[np.ndarray]:
    return [
        np.asarray(positions, dtype=np.intp)
        for positions in frame.groupby(["season", "week"], sort=False).indices.values()
    ]


def permuted_margins(
    frame: pd.DataFrame, rng: np.random.Generator, groups: list[np.ndarray]
) -> pd.Series:
    values = frame["settle_margin"].to_numpy(dtype=float, copy=True)
    for positions in groups:
        values[positions] = rng.permutation(values[positions])
    return pd.Series(values, index=frame.index)


def _paired_metric(reference: str, candidate: str) -> Any:
    def metric(df: pd.DataFrame) -> dict[str, float]:
        valid = df.dropna(subset=[reference, candidate])
        if valid.empty:
            return {
                "delta_accuracy": float("nan"),
                "candidate_accuracy": float("nan"),
                "reference_accuracy": float("nan"),
            }
        return {
            "delta_accuracy": float((valid[candidate] - valid[reference]).mean()),
            "candidate_accuracy": float(valid[candidate].mean()),
            "reference_accuracy": float(valid[reference].mean()),
        }

    return metric


def null_distribution(frame: pd.DataFrame, *, permutations: int, seed: int) -> dict[str, Any]:
    """Within-week permutation null.

    Not centred on zero by design -- it preserves each week's realised
    home-cover rate and the two arms carry different home-pick rates. Reported
    ALONGSIDE the bootstrap-vs-zero interval, never instead of it. Both arms'
    models are fit ONCE on the REAL margin; only the grading margin is shuffled,
    so the draws cost no extra fits.
    """

    rng = np.random.default_rng(seed)
    metric = _paired_metric("baseline_correct", "candidate_correct")
    groups = week_positions(frame)
    deltas = [
        metric(grade(frame, permuted_margins(frame, rng, groups)))["delta_accuracy"]
        for _ in range(permutations)
    ]
    values = np.asarray(deltas, dtype=float)
    finite = values[np.isfinite(values)]
    observed = metric(grade(frame))["delta_accuracy"]
    return {
        "permutations": len(finite),
        "null_mean_delta": float(finite.mean()),
        "null_sd_delta": float(finite.std(ddof=1)),
        "null_q025": float(np.quantile(finite, 0.025)),
        "null_q975": float(np.quantile(finite, 0.975)),
        "observed_delta": float(observed),
        "fraction_of_null_below_observed": float((finite < observed).mean()),
    }


def summarize_pair(paired: pd.DataFrame, samples: int, seed: int) -> dict[str, Any] | None:
    if paired.empty or paired.dropna(subset=["baseline_correct", "candidate_correct"]).empty:
        return None
    metric = _paired_metric("baseline_correct", "candidate_correct")
    point = metric(paired)
    week = week_blocked_bootstrap(paired, metric, block="week", samples=samples, seed=seed)
    week_row = week.loc[week["metric"].eq("delta_accuracy")].iloc[0]
    summary: dict[str, Any] = {
        "delta_accuracy": point["delta_accuracy"],
        "candidate_accuracy": point["candidate_accuracy"],
        "baseline_accuracy": point["reference_accuracy"],
        "week_blocked_ci95": [float(week_row["lower"]), float(week_row["upper"])],
        "week_blocked_probability_positive": float(week_row["probability_positive"]),
        "n_games": len(paired.dropna(subset=["baseline_correct", "candidate_correct"])),
        "n_weeks": int(paired[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(paired["season"].nunique()),
        "n_flagged": int(pd.to_numeric(paired["feature_value"], errors="coerce").sum()),
        "n_feature_missing": int(
            pd.to_numeric(paired["feature_value"], errors="coerce").isna().sum()
        ),
    }
    if paired["season"].nunique() >= 2:
        season = week_blocked_bootstrap(paired, metric, block="season", samples=samples, seed=seed)
        season_row = season.loc[season["metric"].eq("delta_accuracy")].iloc[0]
        summary["season_blocked_ci95"] = [float(season_row["lower"]), float(season_row["upper"])]
        summary["season_blocked_probability_positive"] = float(season_row["probability_positive"])
    else:
        summary["season_blocked_ci95"] = None
        summary["season_blocked_probability_positive"] = None
    return summary


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _print_pair(label: str, summary: dict[str, Any] | None) -> None:
    if summary is None:
        print(f"  {label}: no scored games")
        return
    low, high = summary["week_blocked_ci95"]
    print(
        f"  {label}: delta {summary['delta_accuracy'] * 100:+.3f} pts  P+ "
        f"{summary['week_blocked_probability_positive']:.3f}  week 95% "
        f"[{low * 100:+.3f}, {high * 100:+.3f}]  n={summary['n_games']} games, "
        f"{summary['n_weeks']} weeks, flagged={summary['n_flagged']}, "
        f"missing={summary['n_feature_missing']}"
    )


def _coverage_payload(
    diagnostics: dict[str, Any], panel: pd.DataFrame, started: float, seed: int
) -> dict[str, Any]:
    clean_panel = panel.loc[panel["season"].isin(CFB_CLEAN_CORE_SEASONS)]
    return {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "mode": "coverage",
        "note": (
            "PREDICTOR-ONLY diagnostics: no outcome column is read. This is what "
            "freezes the scored season set in docs/cfb_rest_bye_replication.md."
        ),
        "overlap_disclosure": OVERLAP_DISCLOSURE,
        "baseline_already_carries_rest_diff": "rest_diff" in CFB_MODEL_FEATURE_COLUMNS,
        "clean_core_seasons": list(CFB_CLEAN_CORE_SEASONS),
        "covered_clean_core_seasons": {
            name: list(covered_clean_core_seasons(diagnostics, cell["column"]))
            for name, cell in CELLS.items()
        },
        "n_clean_core_panel_rows": len(clean_panel),
        "reliability": reliability_readings(clean_panel, seed),
        **{key: value for key, value in diagnostics.items() if key != "team_panel"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("coverage", "null", "positive-control", "screen"), required=True
    )
    parser.add_argument("--cell", choices=tuple(CELLS), default="home_off_bye")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    started = time.time()

    print(f"=== loading {args.features} ===", flush=True)
    attached, diagnostics = load_population(args.features)
    panel = diagnostics["team_panel"]

    reconstruction = diagnostics["rest_diff_reconstruction"]
    print(
        f"table rows={diagnostics['n_games']}, "
        f"home_rest missing={diagnostics['n_home_rest_missing']}, "
        f"away_rest missing={diagnostics['n_away_rest_missing']}"
    )
    print(
        "rest_diff reconstruction: "
        f"{reconstruction['n_exact_match']}/{reconstruction['n_both_defined']} exact, "
        f"missingness mismatches={reconstruction['n_missingness_pattern_mismatch']}, "
        f"max|diff|={reconstruction['max_abs_difference']}"
    )

    if args.mode == "coverage":
        payload = _coverage_payload(diagnostics, panel, started, args.seed)
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_ROOT / "coverage.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=float), encoding="utf-8"
        )
        print(json.dumps(payload["covered_clean_core_seasons"], indent=2))
        print("flagged / missing per column (whole 12,500-row table):")
        for column, flagged in diagnostics["flagged_by_column"].items():
            print(
                f"  {column}: flagged={flagged} missing={diagnostics['missing_by_column'][column]}"
            )
        print("split-half reliability (clean-core team panel):")
        for instrument, readings in payload["reliability"].items():
            for metric, values in readings.items():
                print(
                    f"  {instrument:<28} {metric:<20} n={values['n_team_seasons']:>5} "
                    f"r={values['pearson_r']:+.4f} "
                    f"SB={values['spearman_brown_full_length_reliability']:+.4f} "
                    f"P+={values['probability_positive']:.4f}"
                )
        print(f"wrote {ARTIFACT_ROOT / 'coverage.json'}")
        return 0

    cell = CELLS[args.cell]
    candidate_column = cell["column"]
    scored_seasons = covered_clean_core_seasons(diagnostics, candidate_column)
    print(f"cell={args.cell} ({cell['role']}) column={candidate_column} mode={args.mode}")
    print(f"NFL sibling: {cell['nfl_sibling']}; predicted: {cell['predicted_direction']}")
    print(f"scored seasons: {scored_seasons}", flush=True)

    fitted = run_walk_forward(
        attached,
        scored_seasons,
        candidate_column=candidate_column,
        leak_treatment=args.mode == "positive-control",
    )

    result: dict[str, Any] = {"status": "no_scored_games"}
    if not fitted.empty:
        if args.mode == "null":
            result = {
                "status": "scored",
                "null": null_distribution(fitted, permutations=args.permutations, seed=args.seed),
            }
        else:
            graded = grade(fitted)
            era_results = {}
            for label, start, end in ERAS:
                era_frame = graded.loc[graded["season"].between(start, end)]
                era_results[label] = (
                    summarize_pair(era_frame, samples=args.bootstrap_samples, seed=args.seed)
                    if not era_frame.empty
                    else None
                )
            season_results = {
                str(int(season)): summarize_pair(
                    group, samples=args.bootstrap_samples, seed=args.seed
                )
                for season, group in graded.groupby("season", sort=True)
            }
            result = {
                "status": "scored",
                "home_pick_rate": {
                    arm: float(graded[column].ge(0.5).mean())
                    for arm, column in ARM_PROBABILITY.items()
                },
                "permutation_null": null_distribution(
                    fitted, permutations=args.permutations, seed=args.seed
                ),
                "candidate_vs_baseline_pooled": summarize_pair(
                    graded, samples=args.bootstrap_samples, seed=args.seed
                ),
                "era_results": era_results,
                "season_results": season_results,
            }

    configuration = {
        "cell": args.cell,
        "role": cell["role"],
        "nfl_sibling": cell["nfl_sibling"],
        "predicted_direction": cell["predicted_direction"],
        "mode": args.mode,
        "scored_seasons": list(scored_seasons),
        "grade": "close_proxy_median_book_spread_line",
        "league": "cfb",
        "baseline_feature_columns": list(CFB_MODEL_FEATURE_COLUMNS),
        "baseline_already_carries_rest_diff": "rest_diff" in CFB_MODEL_FEATURE_COLUMNS,
        "candidate_column": candidate_column,
        "regressor": "ridge",
        "ridge_alpha": CFB_BENCHMARK_RIDGE_ALPHA,
        "min_train_games": CFB_BENCHMARK_MIN_TRAIN_GAMES,
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": args.seed,
        "predeclaration": "docs/cfb_rest_bye_replication.md",
        "features_path": str(args.features),
    }
    clean_panel = panel.loc[panel["season"].isin(CFB_CLEAN_CORE_SEASONS)]
    panel_metric = CFB_REST_CELL_PANEL_METRIC[candidate_column]
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        "overlap_disclosure": OVERLAP_DISCLOSURE,
        **configuration,
        "panel_metric": panel_metric,
        "reliability": {
            instrument: readings[panel_metric]
            for instrument, readings in reliability_readings(clean_panel, args.seed).items()
        },
        "coverage": {
            "coverage_by_season": diagnostics["coverage_by_season"][candidate_column],
            "n_flagged_whole_table": diagnostics["flagged_by_column"][candidate_column],
            "n_missing_whole_table": diagnostics["missing_by_column"][candidate_column],
            "n_home_rest_missing": diagnostics["n_home_rest_missing"],
            "n_away_rest_missing": diagnostics["n_away_rest_missing"],
        },
        "rest_diff_reconstruction": reconstruction,
        "result": result,
        "provenance": artifact_provenance(configuration, args.features, project_root=REPO_ROOT),
    }
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = ARTIFACT_ROOT / timestamp
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="cfb-rest-bye-replication",
        metrics={"cell": args.cell, "mode": args.mode, "status": result.get("status", "unknown")},
        notes=(
            "Cross-league CFB replication of the NFL rest/bye constructs on top of the "
            "frozen XLG-03 benchmark arm, which ALREADY carries rest_diff; no NFL window "
            "spent. See docs/cfb_rest_bye_replication.md. " + OVERLAP_DISCLOSURE
        ),
    )
    print("wrote " + str(output_dir / "results.json"))

    if result.get("status") != "scored":
        return 0
    if args.mode == "null":
        null = result["null"]
        print()
        print(
            f"NULL CHECK ({args.permutations} within-week permutations): the distribution "
            "must be centred near its own closed-form expectation, not necessarily zero."
        )
        print(
            f"null mean {null['null_mean_delta'] * 100:+.3f} pts, sd "
            f"{null['null_sd_delta'] * 100:.3f}, 95% [{null['null_q025'] * 100:+.3f}, "
            f"{null['null_q975'] * 100:+.3f}], observed {null['observed_delta'] * 100:+.3f}"
        )
        return 0

    print()
    print(f"candidate (+{candidate_column}) minus XLG-03 baseline, {args.mode}:")
    _print_pair("pooled", result["candidate_vs_baseline_pooled"])
    null = result["permutation_null"]
    print(
        f"permutation null: mean {null['null_mean_delta'] * 100:+.3f} pts, observed at the "
        f"{null['fraction_of_null_below_observed'] * 100:.1f}th percentile of its own null"
    )
    for label, _start, _end in ERAS:
        _print_pair(f"era {label}", result["era_results"][label])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
