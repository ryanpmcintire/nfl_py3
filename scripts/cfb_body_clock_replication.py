"""Circadian body-clock / timezone cells, replicated on COLLEGE FOOTBALL.

Predeclared in ``docs/cfb_body_clock_replication.md`` BEFORE this script was
pointed at any outcome column. Read that document first -- it freezes the
population, the venue-state -> IANA-timezone map (split-state resolution rule,
neutral-site rule, WEST zone set, past-midnight-ET rule), the comparator, the
four cells, the null, the positive control, the era split and the recording
rules.

Four cells, selected with ``--cell``, each replicating one recorded NFL entry:

* ``west_road_early``              -> NFL ``body_clock_west_road_early``
* ``east_host_west_visitor_early`` -> NFL ``body_clock_east_host_west_visitor_early``
* ``eastbound_multizone``          -> NFL ``travel_rest_eastbound_multizone``
* ``night_west_road``              -> NFL ``body_clock_night_west_road_ge2000et``

Baseline in every case is the frozen XLG-03 benchmark arm --
``nfl_ats.cfb_benchmark.fit_cfb_residual_model`` on
``nfl_ats.cfb_features.CFB_MODEL_FEATURE_COLUMNS``, market-residual target,
ridge alpha 10, minimum 500 training games, weekly refits trained strictly
before each scored week's earliest gameday. The candidate is the identical
estimator with exactly one extra column, using the benchmark's own declared
``feature_columns`` extension point.

Because the four NFL entries are signed as a subset-vs-complement
full-slate-scaled ``home_cover`` gap and NOT as a paired model delta, every
scoring mode ALSO computes that NFL estimator verbatim via
``scripts._common.summarize``, so the "did the NFL direction replicate"
question is answered on commensurable numbers. The primary registered effect
remains the paired model delta.

Modes (``--mode``):

* ``coverage``         -- PREDICTOR-ONLY diagnostics (zone coverage, flagged
  counts, split-half reliability, the per-school zone assignment table).
  Touches no outcome column.
* ``null``             -- settle margins shuffled within each week. A harness
  that reports an effect here is broken.
* ``positive-control`` -- the candidate's one new column is replaced by the
  realised ``ats_margin``, a deliberate leak. A harness that cannot detect this
  inside the full 36-column ridge fit would be blind.
* ``screen``           -- the real look. No NFL evaluation window and no
  rotation window is spent: CFB is this project's sanctioned free cross-league
  replication ground.

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
from nfl_ats.cfb_body_clock_feature import (  # noqa: E402
    CFB_BODY_CLOCK_CELL_COLUMNS,
    CFB_BODY_CLOCK_FEATURE_COLUMNS,
    attach_cfb_body_clock_features,
    body_clock_offset_panel,
    cell_exposure_panel,
    default_team_info_dir,
    load_team_zone_map,
    zone_assignment_table,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS  # noqa: E402
from nfl_ats.cfb_qb_dependence import split_half_reliability  # noqa: E402
from nfl_ats.clv import pick_correct, week_blocked_bootstrap  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from scripts._common import summarize as summarize_two_group  # noqa: E402

DEFAULT_FEATURES = REPO_ROOT / "data" / "processed" / "cfb_game_features.parquet"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "cfb_body_clock_replication"

#: docs/cfb_body_clock_replication.md section 7 -- 1,000 samples and seed
#: 20260901, matching the sibling CFB replication harness for comparability.
BOOTSTRAP_SAMPLES = 1_000
SEED = 20260901
PERMUTATIONS = 200

#: The NFL screens' own bootstrap size for the subset-vs-complement estimator
#: (docs/body_clock_screen.md: "20,000 samples"). Kept at the NFL figure so the
#: commensurable comparison is commensurable in its uncertainty too.
TWO_GROUP_SAMPLES = 20_000

#: docs/cfb_body_clock_replication.md section 7 -- the benchmark's own declared
#: 2020 regime gap. Magnitudes are reported per era and NEVER averaged across a
#: sign flip (owner rule "era magnitude, not presence").
ERAS: tuple[tuple[str, int, int], ...] = (("2012_2019", 2012, 2019), ("2021_2025", 2021, 2025))

#: Cell key -> the NFL registry entry it replicates, so every artifact says so.
NFL_SIBLINGS: dict[str, str] = {
    "west_road_early": "body_clock_west_road_early",
    "east_host_west_visitor_early": "body_clock_east_host_west_visitor_early",
    "eastbound_multizone": "travel_rest_eastbound_multizone",
    "night_west_road": "body_clock_night_west_road_ge2000et",
}


# ---------------------------------------------------------------------------
# population
# ---------------------------------------------------------------------------


def load_population(features_path: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """The XLG-03 benchmark table with all four candidate columns attached."""

    features = pd.read_parquet(features_path)
    features["gameday"] = pd.to_datetime(features["gameday"], errors="raise")
    zones = load_team_zone_map()
    attached, diagnostics = attach_cfb_body_clock_features(features, team_zones=zones)
    attached = attached.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return attached, diagnostics, zones


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
    WHOLE table whose ``gameday`` is strictly before that week's own earliest
    ``gameday``, with the benchmark's own 500-game floor -- the same forward
    chaining ``nfl_ats.cfb_benchmark.cfb_walk_forward_benchmark`` performs.
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
    ALONGSIDE the bootstrap-vs-zero interval, never instead of it.
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
# the NFL-commensurable estimator: subset-vs-complement home_cover gap
# ---------------------------------------------------------------------------


def two_group_gap(
    attached: pd.DataFrame, scored_seasons: tuple[int, ...], column: str, *, seed: int
) -> dict[str, Any]:
    """``(subset_cover - complement_cover) * 100 * fraction_of_slate``.

    The estimator the four NFL registry entries are actually signed with,
    computed verbatim via ``scripts._common.summarize``. NFL population
    convention taken unchanged: ``home_cover`` with pushes dropped, and a row
    whose flag is MISSING has the flag forced False and sits in the complement
    (docs/cfb_body_clock_replication.md section 6).
    """

    scored = attached.loc[attached["season"].astype(int).isin(scored_seasons)].copy()
    scored = scored.loc[pd.to_numeric(scored["home_cover"], errors="coerce").notna()]
    scored["home_cover"] = pd.to_numeric(scored["home_cover"], errors="coerce").astype(float)
    values = pd.to_numeric(scored[column], errors="coerce")
    flag = values.fillna(0.0).astype(bool)
    scored["_week_block"] = scored["season"].astype(int) * 100 + scored["week"].astype(int)
    scored["_season_block"] = scored["season"].astype(int)
    week = summarize_two_group(
        scored, flag=flag, block_col="_week_block", samples=TWO_GROUP_SAMPLES, seed=seed
    )
    season = summarize_two_group(
        scored, flag=flag, block_col="_season_block", samples=TWO_GROUP_SAMPLES, seed=seed
    )
    return {
        "note": (
            "NFL-commensurable estimator (subset-vs-complement full-slate-scaled home_cover "
            "gap, scripts/_common.summarize). Rows with a missing flag are forced False into "
            "the complement, the NFL family's own convention."
        ),
        "n_flag_missing_forced_false": int(values.isna().sum()),
        "week_blocked": week,
        "season_blocked": season,
    }


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


def _reliabilities(diagnostics: dict[str, Any], derived: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Predictor-only split-half reliability reads (section 7)."""

    context = diagnostics["context"]
    out: dict[str, Any] = {}
    for cell, column in CFB_BODY_CLOCK_CELL_COLUMNS.items():
        panel = cell_exposure_panel(context, derived, column)
        out[cell] = split_half_reliability(panel, "in_cell", seed=seed)
    out["body_clock_utc_offset"] = split_half_reliability(
        body_clock_offset_panel(context), "body_offset_hours", seed=seed
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("coverage", "null", "positive-control", "screen"), required=True
    )
    parser.add_argument(
        "--cell", choices=tuple(CFB_BODY_CLOCK_CELL_COLUMNS), default="west_road_early"
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    started = time.time()
    scored_seasons = tuple(CFB_CLEAN_CORE_SEASONS)

    print(f"=== loading {args.features} ===", flush=True)
    attached, diagnostics, zones = load_population(args.features)
    context = diagnostics.pop("context")
    derived = attached.loc[:, ["game_id", *CFB_BODY_CLOCK_FEATURE_COLUMNS]]

    print(
        f"table rows={diagnostics['n_games']}, zones={len(diagnostics['zones_present'])}, "
        f"unresolved_zone={diagnostics['n_unresolved_zone']}, "
        f"split_state_fallback={diagnostics['n_split_state_city_fallback']}, "
        f"neutral_site={diagnostics['n_neutral_site']}, "
        f"past_midnight_et={diagnostics['n_past_midnight_et_adjusted']}"
    )

    if args.mode == "coverage":
        diagnostics["context"] = context
        reliability = _reliabilities(diagnostics, derived, args.seed)
        diagnostics.pop("context")
        in_core = attached.loc[attached["season"].astype(int).isin(scored_seasons)]
        core_derived = derived.loc[in_core.index]
        payload = {
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "mode": "coverage",
            "note": (
                "PREDICTOR-ONLY diagnostics: no outcome column is read. Fills section 5 of "
                "docs/cfb_body_clock_replication.md."
            ),
            "team_info_snapshot": str(default_team_info_dir()),
            "clean_core_seasons": list(scored_seasons),
            "clean_core_rows": len(in_core),
            "clean_core_neutral_site": int(
                pd.to_numeric(in_core["neutral_site"], errors="coerce").fillna(0).eq(1).sum()
            ),
            "clean_core_flagged_total": {
                column: int(pd.to_numeric(core_derived[column], errors="coerce").sum())
                for column in CFB_BODY_CLOCK_FEATURE_COLUMNS
            },
            "clean_core_feature_missing": {
                column: int(pd.to_numeric(core_derived[column], errors="coerce").isna().sum())
                for column in CFB_BODY_CLOCK_FEATURE_COLUMNS
            },
            "reliability": reliability,
            **dict(diagnostics),
        }
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_ROOT / "coverage.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=float), encoding="utf-8"
        )
        (ARTIFACT_ROOT / "zone_assignments.json").write_text(
            json.dumps(zone_assignment_table(zones), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"clean core rows: {payload['clean_core_rows']}")
        print("clean-core flagged / missing per cell:")
        for cell, column in CFB_BODY_CLOCK_CELL_COLUMNS.items():
            print(
                f"  {cell:<30} flagged={payload['clean_core_flagged_total'][column]:>5} "
                f"missing={payload['clean_core_feature_missing'][column]:>4}"
            )
        print("zone coverage by season (home / away):")
        for season in sorted(diagnostics["home_zone_coverage_by_season"]):
            print(
                f"  {season}: {diagnostics['home_zone_coverage_by_season'][season]:.3f} / "
                f"{diagnostics['away_zone_coverage_by_season'][season]:.3f}"
            )
        print(f"zones present: {diagnostics['zones_present']}")
        print(
            f"hawaii body clocks excluded from WEST: "
            f"{diagnostics['n_hawaii_body_clock_excluded']}; mountain excluded: "
            f"{diagnostics['n_mountain_body_clock_excluded']}"
        )
        print("split-half reliability (predictor-only):")
        for key, value in reliability.items():
            print(
                f"  {key:<30} n={value['n_team_seasons']:>5} r={value['pearson_r']:.4f} "
                f"SB={value['spearman_brown_full_length_reliability']:.4f} "
                f"P+={value['probability_positive']:.4f}"
            )
        print(f"wrote {ARTIFACT_ROOT / 'coverage.json'}")
        print(f"wrote {ARTIFACT_ROOT / 'zone_assignments.json'}")
        return 0

    candidate_column = CFB_BODY_CLOCK_CELL_COLUMNS[args.cell]
    print(f"cell={args.cell} column={candidate_column} mode={args.mode}")
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
            era_results: dict[str, Any] = {}
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
            if args.mode == "screen":
                result["nfl_commensurable_two_group"] = two_group_gap(
                    attached, scored_seasons, candidate_column, seed=args.seed
                )
                result["nfl_commensurable_two_group_by_era"] = {
                    label: two_group_gap(
                        attached,
                        tuple(s for s in scored_seasons if start <= s <= end),
                        candidate_column,
                        seed=args.seed,
                    )
                    for label, start, end in ERAS
                }

    configuration = {
        "cell": args.cell,
        "nfl_sibling": NFL_SIBLINGS[args.cell],
        "mode": args.mode,
        "scored_seasons": list(scored_seasons),
        "grade": "close_proxy_median_book_spread_line",
        "league": "cfb",
        "baseline_feature_columns": list(CFB_MODEL_FEATURE_COLUMNS),
        "candidate_column": candidate_column,
        "regressor": "ridge",
        "ridge_alpha": CFB_BENCHMARK_RIDGE_ALPHA,
        "min_train_games": CFB_BENCHMARK_MIN_TRAIN_GAMES,
        "bootstrap_samples": args.bootstrap_samples,
        "two_group_bootstrap_samples": TWO_GROUP_SAMPLES,
        "permutations": args.permutations,
        "seed": args.seed,
        "predeclaration": "docs/cfb_body_clock_replication.md",
        "features_path": str(args.features),
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "coverage": dict(diagnostics),
        "result": result,
        "provenance": artifact_provenance(configuration, args.features, project_root=REPO_ROOT),
    }
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = ARTIFACT_ROOT / timestamp
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="cfb-body-clock-replication",
        metrics={"cell": args.cell, "mode": args.mode, "status": result.get("status", "unknown")},
        notes=(
            "Cross-league CFB replication of the NFL circadian body-clock / timezone cells on "
            "top of the frozen XLG-03 benchmark arm; no NFL window spent. See "
            "docs/cfb_body_clock_replication.md."
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
    if args.mode == "screen":
        gap = result["nfl_commensurable_two_group"]["week_blocked"]
        print(
            f"NFL-commensurable gap (subset vs complement, {NFL_SIBLINGS[args.cell]}): "
            f"{gap['full_slate_effect_pts']:+.4f} pts, 95% "
            f"[{gap['ci95_scaled'][0]:+.4f}, {gap['ci95_scaled'][1]:+.4f}], "
            f"P+ {gap['probability_positive']:.4f}, n_flag={gap['n_flag']}, "
            f"subset_cover={gap['subset_cover']:.4f}, complement={gap['complement_cover']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
