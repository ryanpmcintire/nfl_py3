"""FluView elevated-illness indicators, replicated on COLLEGE FOOTBALL.

Predeclared in ``docs/fluview_cfb_replication.md`` BEFORE this script was
pointed at any outcome column. Read that document first -- it freezes the
population, the school -> venue-state source, the comparator, the two cells,
the null, the positive control, the era split and the recording rules.

Two cells, selected with ``--cell``:

* ``away`` (PRIMARY)   -- XLG-03 benchmark contract plus
  ``cfb_fluview_away_market_elevated``.
* ``home`` (SECONDARY) -- XLG-03 benchmark contract plus
  ``cfb_fluview_home_market_elevated``.

The primary/secondary labels mirror ``docs/fluview_on_production.md`` section 4
verbatim (that NFL family declares away primary, home secondary); both cells
are recorded regardless of sign and reported with equal prominence, and the
decision rule is expected value, so the label fixes only which cell leads the
write-up.

Baseline in both cases is the frozen XLG-03 benchmark arm --
``nfl_ats.cfb_benchmark.fit_cfb_residual_model`` on
``nfl_ats.cfb_features.CFB_MODEL_FEATURE_COLUMNS``, market-residual target,
ridge alpha 10, minimum 500 training games, weekly refits trained strictly
before each scored week's earliest kickoff. The candidate is the identical
estimator with exactly one extra column, using the benchmark's own declared
``feature_columns`` extension point.

Modes (``--mode``):

* ``fetch-inputs``     -- one-off: pin the cfbfastR-data head commit and
  download the ``team_info`` parquets that carry each school's venue STATE (no
  local CFB snapshot has it; no CFBD API credit is spent). Free, no key, same
  host ``src/nfl_ats/cfb.py`` already declares for schedules and lines.
* ``coverage``         -- PREDICTOR-ONLY diagnostics (per-season feature
  coverage, per-state thresholds, split-half reliability). Touches no outcome
  column; this is what freezes the scored season set in the predeclaration.
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
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nfl_ats.cfb import (  # noqa: E402
    CFBFASTR_DATA_REPOSITORY,
    _cfbfastr_raw_url,
    resolve_cfbfastr_commit,
)
from nfl_ats.cfb_benchmark import (  # noqa: E402
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_CLEAN_CORE_SEASONS,
    fit_cfb_residual_model,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS  # noqa: E402
from nfl_ats.cfb_qb_dependence import split_half_reliability  # noqa: E402
from nfl_ats.clv import pick_correct, week_blocked_bootstrap  # noqa: E402
from nfl_ats.fluview_cfb_feature import (  # noqa: E402
    CFB_FLUVIEW_AWAY_ELEVATED_COLUMN,
    CFB_FLUVIEW_HOME_ELEVATED_COLUMN,
    TEAM_INFO_COLUMNS,
    attach_cfb_fluview_features,
)
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

DEFAULT_FEATURES = REPO_ROOT / "data" / "processed" / "cfb_game_features.parquet"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "fluview_cfb_replication"
TEAM_INFO_ROOT = REPO_ROOT / "data" / "cfb" / "team_info" / "raw"

#: docs/fluview_cfb_replication.md section 7 -- 1,000 samples for comparability
#: with the NFL sibling harness, seed = today's date per repo convention.
BOOTSTRAP_SAMPLES = 1_000
SEED = 20260901
PERMUTATIONS = 200

#: docs/fluview_cfb_replication.md section 7 -- the obvious boundary is the
#: benchmark's own declared 2020 regime gap. Magnitudes are reported per era
#: and NEVER averaged across a sign flip (owner rule "era magnitude, not
#: presence").
ERAS: tuple[tuple[str, int, int], ...] = (("2017_2019", 2017, 2019), ("2021_2025", 2021, 2025))

#: Seasons the cfbfastR-data team_info snapshot must cover: every season the
#: XLG-03 table can carry, so the school -> state map never runs short of the
#: training population.
TEAM_INFO_SEASONS = tuple(range(2006, 2026))

CELLS: dict[str, dict[str, str]] = {
    "away": {"column": CFB_FLUVIEW_AWAY_ELEVATED_COLUMN, "role": "primary"},
    "home": {"column": CFB_FLUVIEW_HOME_ELEVATED_COLUMN, "role": "secondary"},
}


# ---------------------------------------------------------------------------
# mode: fetch-inputs
# ---------------------------------------------------------------------------


def fetch_team_info() -> Path:
    """Download cfbfastR-data ``team_info`` (school -> venue STATE) at a pinned commit.

    Free, no key, ``raw.githubusercontent.com``, the same repository
    ``src/nfl_ats/cfb.py`` already declares as a sanctioned CFB source for
    schedules and lines. Deliberately NOT the CFBD ``/venues`` endpoint the
    surface-familiarity replication used: no CFBD API credit is spent here.
    """

    pin = resolve_cfbfastr_commit()
    commit = pin["commit_sha"]
    snapshot = TEAM_INFO_ROOT / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    entries: list[dict[str, Any]] = []
    for season in TEAM_INFO_SEASONS:
        relative = f"team_info/parquet/cfb_team_info_{season}.parquet"
        url = _cfbfastr_raw_url(commit, relative)
        request = urllib.request.Request(url, headers={"User-Agent": "nfl-ats-research/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
        frame = pd.read_parquet(pd.io.common.BytesIO(body))
        missing = sorted(set(TEAM_INFO_COLUMNS).difference(frame.columns))
        if missing:
            raise SystemExit(f"team_info {season} is missing columns: {', '.join(missing)}")
        partition = snapshot / f"season={season}"
        partition.mkdir(parents=True, exist_ok=True)
        out = partition / "team_info.parquet"
        frame.loc[:, list(TEAM_INFO_COLUMNS)].to_parquet(out, index=False)
        entries.append(
            {
                "season": season,
                "url": url,
                "source_bytes": len(body),
                "source_sha256": hashlib.sha256(body).hexdigest(),
                "rows": len(frame),
                "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            }
        )
        print(f"  season={season}: {len(frame)} rows, {len(body)} bytes")
    manifest = {
        "source": f"https://github.com/{CFBFASTR_DATA_REPOSITORY} team_info/parquet",
        "commit_pin": pin,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "columns_kept": list(TEAM_INFO_COLUMNS),
        "why": (
            "school -> venue STATE for the FluView CFB replication; no local CFB "
            "snapshot carries venue state and registry/stadium_coordinates.json is "
            "NFL-only. No CFBD API credit spent."
        ),
        "redistribution_rule": (
            "Private research retention only; raw CFB source tables are never "
            "republished from this repository."
        ),
        "partitions": entries,
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {snapshot / 'manifest.json'}")
    return snapshot


# ---------------------------------------------------------------------------
# population
# ---------------------------------------------------------------------------


def load_population(features_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The XLG-03 benchmark table with both candidate columns attached.

    Thresholds are frozen ONCE on the whole table (docs/fluview_cfb_replication.md
    section 3), before any restriction to a scored window, so a narrower scored
    population can never move the decile that defines "elevated".
    """

    features = pd.read_parquet(features_path)
    features["gameday"] = pd.to_datetime(features["gameday"], errors="raise")
    attached, diagnostics = attach_cfb_fluview_features(features)
    attached = attached.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return attached, diagnostics


def covered_clean_core_seasons(diagnostics: dict[str, Any]) -> tuple[int, ...]:
    """Clean-core seasons whose measured home-side feature coverage exceeds zero.

    Predictor-only (docs/fluview_cfb_replication.md section 2): the coverage
    fraction is computed from the feature column alone, never from an outcome.
    """

    coverage = diagnostics["home_coverage_by_season"]
    return tuple(
        season for season in CFB_CLEAN_CORE_SEASONS if float(coverage.get(str(season), 0.0)) > 0.0
    )


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
    ``nfl_ats.cfb_benchmark.cfb_walk_forward_benchmark`` performs, and the
    per-week refit ``scripts/fluview_elevated_on_production.py::run_leg``
    performs on the NFL side.
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
    """Within-week permutation null. Not centred on zero by design -- it
    preserves each week's realised home-cover rate and the two arms carry
    different home-pick rates. Reported ALONGSIDE the bootstrap-vs-zero
    interval, never instead of it."""

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
        "n_flagged_elevated": int(pd.to_numeric(paired["feature_value"], errors="coerce").sum()),
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
        f"{summary['n_weeks']} weeks, flagged={summary['n_flagged_elevated']}, "
        f"missing={summary['n_feature_missing']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("fetch-inputs", "coverage", "null", "positive-control", "screen"),
        required=True,
    )
    parser.add_argument("--cell", choices=tuple(CELLS), default="away")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--population",
        choices=("covered", "full-clean-core"),
        default="covered",
        help="'covered' is the predeclared primary population (clean-core seasons with "
        "measured non-zero feature coverage); 'full-clean-core' is the disclosed "
        "secondary read that also includes the zero-coverage seasons",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    started = time.time()

    if args.mode == "fetch-inputs":
        snapshot = fetch_team_info()
        print(f"team_info snapshot: {snapshot}")
        return 0

    print(f"=== loading {args.features} ===", flush=True)
    attached, diagnostics = load_population(args.features)
    covered = covered_clean_core_seasons(diagnostics)
    scored_seasons = covered if args.population == "covered" else CFB_CLEAN_CORE_SEASONS
    panel = diagnostics.pop("state_week_panel")

    print(
        f"table rows={diagnostics['n_games']}, states={len(diagnostics['states_present'])}, "
        f"unmapped_state={diagnostics['n_unmapped_state']}, "
        f"neutral_site={diagnostics['n_neutral_site']}"
    )
    print(f"clean-core seasons with non-zero coverage: {covered}")

    reliability = split_half_reliability(
        panel.dropna(subset=["ili"]).assign(team_id=lambda d: d["state"]), "ili", seed=args.seed
    )

    if args.mode == "coverage":
        payload = {
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "mode": "coverage",
            "note": (
                "PREDICTOR-ONLY diagnostics: no outcome column is read. This is what "
                "freezes the scored season set in docs/fluview_cfb_replication.md."
            ),
            "clean_core_seasons": list(CFB_CLEAN_CORE_SEASONS),
            "covered_clean_core_seasons": list(covered),
            "reliability": reliability,
            "n_state_week_panel_rows": len(panel),
            **dict(diagnostics),
        }
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_ROOT / "coverage.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=float), encoding="utf-8"
        )
        print(json.dumps({k: payload[k] for k in ("covered_clean_core_seasons",)}, indent=2))
        print("per-season home-side coverage:")
        for season, value in sorted(diagnostics["home_coverage_by_season"].items()):
            print(f"  {season}: {value:.1%}")
        print(
            f"reliability: n_state_seasons={reliability['n_team_seasons']} "
            f"pearson_r={reliability['pearson_r']:.4f} "
            f"spearman_brown={reliability['spearman_brown_full_length_reliability']:.4f} "
            f"P+={reliability['probability_positive']:.4f}"
        )
        print(f"wrote {ARTIFACT_ROOT / 'coverage.json'}")
        return 0

    cell = CELLS[args.cell]
    candidate_column = cell["column"]
    print(f"cell={args.cell} ({cell['role']}) column={candidate_column} mode={args.mode}")
    print(f"scored seasons ({args.population}): {scored_seasons}", flush=True)

    fitted = run_walk_forward(
        attached,
        tuple(scored_seasons),
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
        "mode": args.mode,
        "population": args.population,
        "scored_seasons": list(scored_seasons),
        "grade": "close_proxy_median_book_spread_line",
        "league": "cfb",
        "baseline_feature_columns": list(CFB_MODEL_FEATURE_COLUMNS),
        "candidate_column": candidate_column,
        "regressor": "ridge",
        "ridge_alpha": CFB_BENCHMARK_RIDGE_ALPHA,
        "min_train_games": CFB_BENCHMARK_MIN_TRAIN_GAMES,
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": args.seed,
        "predeclaration": "docs/fluview_cfb_replication.md",
        "features_path": str(args.features),
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "reliability": reliability,
        "coverage": {
            "covered_clean_core_seasons": list(covered),
            "home_coverage_by_season": diagnostics["home_coverage_by_season"],
            "n_unmapped_state": diagnostics["n_unmapped_state"],
            "n_neutral_site": diagnostics["n_neutral_site"],
            "n_home_missing": diagnostics["n_home_missing"],
            "n_away_missing": diagnostics["n_away_missing"],
        },
        "state_thresholds": diagnostics["state_thresholds"],
        "result": result,
        "provenance": artifact_provenance(configuration, args.features, project_root=REPO_ROOT),
    }
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = ARTIFACT_ROOT / timestamp
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="fluview-cfb-replication",
        metrics={"cell": args.cell, "mode": args.mode, "status": result.get("status", "unknown")},
        notes=(
            "Cross-league CFB replication of the FluView elevated-illness construct on top "
            "of the frozen XLG-03 benchmark arm; no NFL window spent. See "
            "docs/fluview_cfb_replication.md."
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
