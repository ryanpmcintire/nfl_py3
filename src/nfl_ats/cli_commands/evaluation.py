"""Backtests, nested evaluation, experiments, ablations and anytime comparisons."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from nfl_ats.anytime import (
    ANYTIME_METRICS,
    DEFAULT_ALPHA,
    DEFAULT_TARGET_GAMES,
    anytime_summary,
    paired_anytime_comparisons,
)
from nfl_ats.backtest import walk_forward_backtest
from nfl_ats.calibration import RESIDUAL_SMOOTHING_METHODS
from nfl_ats.cli_common import (
    _add_bootstrap_args,
    _add_feature_profile_arg,
    _add_features_arg,
    _add_regressor_args,
    _add_season_range_args,
    _artifacts_root,
    _data_root,
    _load_features,
    _print_json,
    _registry_root,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.dependence import prediction_dependence_audit
from nfl_ats.evaluation import (
    DEFAULT_EVALUATION_CANDIDATES,
    SELECTION_METRICS,
    format_candidates,
    nested_walk_forward_evaluation,
    parse_candidates,
)
from nfl_ats.experiment_runner import run_experiment_cli
from nfl_ats.experiments import (
    DEFAULT_EXPERIMENT_SETS,
    DEFAULT_PLAYER_PROFILE_SETS,
    FROZEN_AVAILABILITY_MIN_TRAIN_GAMES,
    FROZEN_AVAILABILITY_PROFILE,
    FROZEN_AVAILABILITY_RIDGE_ALPHA,
    FROZEN_AVAILABILITY_START_SEASON,
    FROZEN_PARTICIPATION_BASELINE_PROFILE,
    FROZEN_PARTICIPATION_CANDIDATE_PROFILE,
    FROZEN_PARTICIPATION_MIN_TRAIN_GAMES,
    FROZEN_PARTICIPATION_RIDGE_ALPHA,
    FROZEN_PARTICIPATION_START_SEASON,
    FROZEN_PLAYER_CALIBRATIONS,
    FROZEN_PLAYER_EVALUATION_START_SEASON,
    FROZEN_PLAYER_FIRST_TEST_SEASON,
    FROZEN_PLAYER_MIN_CALIBRATION_GAMES,
    FROZEN_PLAYER_MIN_TRAIN_GAMES,
    FROZEN_PLAYER_MODEL_PROFILES,
    FROZEN_PLAYER_RAW_START_SEASON,
    FROZEN_PLAYER_RIDGE_ALPHAS,
    FROZEN_PLAYER_VALIDATION_SEASONS,
    nested_outcome_profile_selection,
    paired_feature_comparisons,
    player_model_candidate_id,
    run_feature_set_experiment,
    run_frozen_player_model_selection,
    run_outcome_profile_experiment,
)
from nfl_ats.io import atomic_csv, atomic_json, atomic_parquet, run_id
from nfl_ats.margin import MARGIN_FEATURE_PROFILES
from nfl_ats.model_card import build_model_card, model_card_markdown
from nfl_ats.modeling import MODEL_NAMES
from nfl_ats.outcomes import OUTCOME_METHODS, outcome_bootstrap_intervals, walk_forward_outcomes
from nfl_ats.portfolio import simulate_bankroll_paths, simulate_paper_bankroll
from nfl_ats.provenance import (
    artifact_provenance,
    sha256_file,
    verify_experiment_links,
    write_experiment_artifact,
)
from nfl_ats.reporting import block_bootstrap_intervals
from nfl_ats.weak_signals import default_registry_path as weak_signal_registry_path


def _cmd_backtest(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    result = walk_forward_backtest(
        features,
        start_season=args.start_season,
        end_season=args.end_season,
        model_name=args.model,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
        feature_set=args.feature_set,
    )
    output = _artifacts_root() / "backtests" / run_id()
    atomic_parquet(result.predictions, output / "predictions.parquet")
    portfolio = simulate_paper_bankroll(
        result.predictions,
        initial_bankroll=args.initial_bankroll,
        kelly_multiplier=args.kelly_multiplier,
        max_bet_fraction=args.max_bet_fraction,
        max_week_fraction=args.max_week_fraction,
        probability_haircut=args.probability_haircut,
    )
    atomic_parquet(portfolio.ledger, output / "paper_ledger.parquet")
    atomic_json(portfolio.metrics, output / "portfolio_metrics.json")
    simulation = simulate_bankroll_paths(
        result.predictions,
        paths=args.bankroll_paths,
        seed=args.bankroll_seed,
        initial_bankroll=args.initial_bankroll,
        kelly_multiplier=args.kelly_multiplier,
        max_bet_fraction=args.max_bet_fraction,
        max_week_fraction=args.max_week_fraction,
        probability_haircut=args.probability_haircut,
    )
    atomic_json(simulation.metrics, output / "bankroll_simulation.json")
    atomic_parquet(simulation.paths, output / "bankroll_paths.parquet")
    metrics = {**result.metrics, "paper_portfolio": portfolio.metrics}
    atomic_json(metrics, output / "metrics.json")
    uncertainty = pd.concat(
        [
            block_bootstrap_intervals(
                result.predictions,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            block_bootstrap_intervals(
                result.predictions,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    atomic_csv(uncertainty, output / "uncertainty.csv")
    configuration = {
        "command": "backtest",
        "start_season": args.start_season,
        "end_season": args.end_season,
        "model": args.model,
        "feature_set": args.feature_set,
        "min_edge": args.min_edge,
        "min_train_games": args.min_train_games,
        "initial_bankroll": args.initial_bankroll,
        "kelly_multiplier": args.kelly_multiplier,
        "max_bet_fraction": args.max_bet_fraction,
        "max_week_fraction": args.max_week_fraction,
        "probability_haircut": args.probability_haircut,
        "bankroll_paths": args.bankroll_paths,
        "bankroll_seed": args.bankroll_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    provenance = artifact_provenance(configuration, args.features)
    write_experiment_artifact(
        output,
        "run.json",
        provenance,
        command="backtest",
        metrics=metrics,
        provenance_key=None,
        registry_root=_registry_root(),
    )
    card = build_model_card(metrics, provenance, result.predictions)
    atomic_json(card, output / "model_card.json")
    (output / "model_card.md").write_text(model_card_markdown(card), encoding="utf-8")
    _print_json({**metrics, "artifact_directory": str(output)})


def _cmd_nested_evaluate(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    candidates = parse_candidates(args.candidates)
    result = nested_walk_forward_evaluation(
        features,
        first_test_season=args.first_test_season,
        last_test_season=args.last_test_season,
        validation_seasons=args.validation_seasons,
        candidates=candidates,
        selection_metric=args.selection_metric,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
    )
    output = _artifacts_root() / "nested_evaluations" / run_id()
    atomic_csv(result.candidate_validation, output / "candidate_validation.csv")
    atomic_csv(result.fold_summary, output / "fold_summary.csv")
    atomic_parquet(result.predictions, output / "predictions.parquet")
    atomic_json(result.metrics, output / "metrics.json")
    uncertainty = pd.concat(
        [
            block_bootstrap_intervals(
                result.predictions,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            block_bootstrap_intervals(
                result.predictions,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    atomic_csv(uncertainty, output / "uncertainty.csv")
    dependence = prediction_dependence_audit(
        result.predictions,
        permutations=args.dependence_permutations,
        seed=args.dependence_seed,
    )
    atomic_json(dependence.summary, output / "dependence_summary.json")
    atomic_csv(dependence.team_summary, output / "dependence_by_team.csv")
    configuration = {
        "command": "nested-evaluate",
        "first_test_season": args.first_test_season,
        "last_test_season": args.last_test_season,
        "validation_seasons": args.validation_seasons,
        "selection_metric": args.selection_metric,
        "candidates": [candidate.candidate_id for candidate in candidates],
        "min_edge": args.min_edge,
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "dependence_permutations": args.dependence_permutations,
        "dependence_seed": args.dependence_seed,
    }
    write_experiment_artifact(
        output,
        "run.json",
        artifact_provenance(configuration, args.features),
        command="nested-evaluate",
        metrics=result.metrics,
        provenance_key=None,
        registry_root=_registry_root(),
    )
    _print_json({**result.metrics, "artifact_directory": str(output)})


def _cmd_dependence_audit(args: argparse.Namespace) -> None:
    if not args.predictions.is_file():
        raise FileNotFoundError(f"Prediction table not found: {args.predictions}")
    predictions = pd.read_parquet(args.predictions)
    audit = prediction_dependence_audit(
        predictions,
        permutations=args.permutations,
        seed=args.seed,
    )
    output = _artifacts_root() / "dependence" / run_id()
    summary = {
        **audit.summary,
        "source_predictions": str(args.predictions.resolve()),
        "source_predictions_sha256": sha256_file(args.predictions),
    }
    atomic_json(summary, output / "summary.json")
    atomic_csv(audit.team_summary, output / "by_team.csv")
    atomic_parquet(audit.team_residuals, output / "team_residuals.parquet")
    _print_json({**summary, "artifact_directory": str(output)})


def _cmd_experiment(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    feature_sets = tuple(part.strip() for part in args.feature_sets.split(",") if part.strip())
    result = run_feature_set_experiment(
        features,
        start_season=args.start_season,
        model_name=args.model,
        feature_sets=feature_sets,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
    )
    output = _artifacts_root() / "experiments" / run_id()
    atomic_csv(result.summary, output / "summary.csv")
    atomic_csv(result.season_summary, output / "season_summary.csv")
    atomic_parquet(result.predictions, output / "predictions.parquet")
    baseline_feature_set = args.baseline_feature_set or feature_sets[0]
    paired = pd.concat(
        [
            paired_feature_comparisons(
                result.predictions,
                baseline_feature_set=baseline_feature_set,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            paired_feature_comparisons(
                result.predictions,
                baseline_feature_set=baseline_feature_set,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    atomic_csv(paired, output / "paired_comparisons.csv")

    configuration = {
        "command": "experiment",
        "start_season": args.start_season,
        "model": args.model,
        "feature_sets": list(feature_sets),
        "min_edge": args.min_edge,
        "min_train_games": args.min_train_games,
        "baseline_feature_set": baseline_feature_set,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="experiment",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_experiment_run(args: argparse.Namespace) -> None:
    """The declarative pipeline: spec in, screen -> bootstrap -> classify ->
    registry record -> provenance stamp out, with zero hand-transcription.
    See ``nfl_ats.experiment_runner`` and ``docs/experiment_pipeline.md``.
    """

    outcome = run_experiment_cli(
        args.spec,
        repo_root=Path.cwd(),
        dry_run=args.dry_run,
        replace=args.replace,
        features_path=args.features,
        market_root=args.market_root,
        artifacts_root=_artifacts_root(),
        registry_root=_registry_root(),
        registry_path=weak_signal_registry_path(_registry_root()),
    )
    if outcome.dry_run:
        _print_json({"dry_run": True, "would_record": outcome.preview})
        return
    _print_json(
        {
            "dry_run": False,
            "artifact_directory": outcome.artifact_directory,
            "registry_record": outcome.registry_record,
            "recorded": outcome.preview,
        }
    )


def _cmd_experiment_verify(args: argparse.Namespace) -> None:
    """Report which registry rows still resolve to a real artifact on disk.

    Read-only audit of ``registry/experiments/``: every row's
    ``artifact_directory`` is resolved (absolute stored paths as-is; relative
    ones against each ``--artifacts-root`` and the repo root) and checked for
    existence, and path/identity inconsistencies are flagged. See
    :func:`nfl_ats.provenance.verify_experiment_links` for the flag meanings.
    """

    verifications = verify_experiment_links(
        artifacts_roots=list(args.artifacts_root) if args.artifacts_root else None
    )
    if args.as_json:
        _print_json(
            {
                "rows": [vars(verification) for verification in verifications],
                "total": len(verifications),
                "present": sum(1 for v in verifications if v.exists),
                "missing": sum(1 for v in verifications if not v.exists),
            }
        )
        if args.require_links and any(not v.exists for v in verifications):
            raise SystemExit(1)
        return

    present = sum(1 for v in verifications if v.exists)
    missing = len(verifications) - present
    flag_counts: dict[str, int] = {}
    for v in verifications:
        for flag in v.flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    print(f"Scanned {len(verifications)} experiment-registry rows.")
    print(f"  linked artifact present: {present}")
    print(
        f"  linked artifact missing: {missing} (expected on a fresh clone: "
        f"artifacts/ is gitignored and local)"
    )
    if flag_counts:
        print("  consistency flags:")
        for flag, count in sorted(flag_counts.items()):
            print(f"    {flag}: {count}")
    if missing:
        print("\nMissing links (measured on this machine, not inferred):")
        for v in verifications:
            if not v.exists:
                tried = ", ".join(v.candidate_paths) or "(no artifact_directory stored)"
                print(f"  - {v.experiment_id}\n      tried: {tried}")
    if args.require_links and missing:
        raise SystemExit(1)


def _cmd_margin_backtest(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = _load_features(args.features)
    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    modeling_started = perf_counter()
    result = walk_forward_outcomes(
        features,
        start_season=args.start_season,
        regressor=args.regressor,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
        feature_profile=args.feature_profile,
        methods=methods,
        ridge_alpha=args.ridge_alpha,
        probability_method=args.probability_method,
    )
    modeling_seconds = perf_counter() - modeling_started
    output = _artifacts_root() / "margins" / run_id()
    atomic_parquet(result.predictions, output / "predictions.parquet")
    atomic_csv(result.summary, output / "summary.csv")
    atomic_csv(result.season_summary, output / "season_summary.csv")
    uncertainty_started = perf_counter()
    uncertainty = pd.concat(
        [
            outcome_bootstrap_intervals(
                result.predictions,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            outcome_bootstrap_intervals(
                result.predictions,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    uncertainty_seconds = perf_counter() - uncertainty_started
    atomic_csv(uncertainty, output / "uncertainty.csv")
    configuration = {
        "command": "margin-backtest",
        "start_season": args.start_season,
        "regressor": args.regressor,
        "ridge_alpha": args.ridge_alpha,
        "calibration_method": "none",
        "min_edge": args.min_edge,
        "min_train_games": args.min_train_games,
        "feature_profile": args.feature_profile,
        "probability_method": args.probability_method,
        "methods": list(methods),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "methods": result.summary["method"].tolist(),
        "timing": {
            "modeling_seconds": modeling_seconds,
            "uncertainty_seconds": uncertainty_seconds,
            "total_seconds": perf_counter() - command_started,
        },
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="margin-backtest",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_player_ablation(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = _load_features(args.features)
    profiles = tuple(part.strip() for part in args.profiles.split(",") if part.strip())
    result = run_outcome_profile_experiment(
        features,
        start_season=args.start_season,
        profiles=profiles,
        regressor=args.regressor,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
        ridge_alpha=args.ridge_alpha,
    )
    output = _artifacts_root() / "player_experiments" / run_id()
    atomic_csv(result.summary, output / "summary.csv")
    atomic_csv(result.season_summary, output / "season_summary.csv")
    atomic_parquet(result.predictions, output / "predictions.parquet")

    comparison_input = result.predictions.rename(columns={"feature_profile": "feature_set"})
    paired = pd.concat(
        [
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set=args.baseline_profile,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set=args.baseline_profile,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    ).rename(
        columns={
            "baseline_feature_set": "baseline_feature_profile",
            "candidate_feature_set": "candidate_feature_profile",
        }
    )
    atomic_csv(paired, output / "paired_comparisons.csv")
    nested = nested_outcome_profile_selection(
        result.predictions,
        first_test_season=args.first_nested_test_season,
        validation_seasons=args.validation_seasons,
    )
    atomic_csv(nested.candidate_validation, output / "nested_candidate_validation.csv")
    atomic_csv(nested.fold_summary, output / "nested_fold_summary.csv")
    atomic_csv(nested.summary, output / "nested_summary.csv")
    atomic_csv(nested.season_summary, output / "nested_season_summary.csv")
    atomic_parquet(nested.predictions, output / "nested_predictions.parquet")
    nested_baseline = result.predictions.loc[
        result.predictions["feature_profile"].eq(args.baseline_profile)
        & result.predictions["season"].isin(nested.predictions["season"].unique())
    ].copy()
    nested_baseline["feature_set"] = args.baseline_profile
    nested_selected = nested.predictions.copy()
    nested_selected["feature_set"] = "nested_selected"
    nested_comparison_input = pd.concat([nested_baseline, nested_selected], ignore_index=True)
    nested_paired = pd.concat(
        [
            paired_feature_comparisons(
                nested_comparison_input,
                baseline_feature_set=args.baseline_profile,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            paired_feature_comparisons(
                nested_comparison_input,
                baseline_feature_set=args.baseline_profile,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    ).rename(
        columns={
            "baseline_feature_set": "baseline_feature_profile",
            "candidate_feature_set": "candidate_feature_profile",
        }
    )
    atomic_csv(nested_paired, output / "nested_paired_comparisons.csv")
    configuration = {
        "command": "player-ablation",
        "start_season": args.start_season,
        "regressor": args.regressor,
        "ridge_alpha": args.ridge_alpha,
        "calibration_method": "none",
        "profiles": list(profiles),
        "baseline_profile": args.baseline_profile,
        "method": "market_residual",
        "min_edge": args.min_edge,
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "first_nested_test_season": args.first_nested_test_season,
        "validation_seasons": args.validation_seasons,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "timing": {"total_seconds": perf_counter() - command_started},
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="player-ablation",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_participation_ablation(args: argparse.Namespace) -> None:
    """Run the predeclared one-candidate participation-value comparison."""

    command_started = perf_counter()
    features = _load_features(args.features)
    profiles = (
        FROZEN_PARTICIPATION_BASELINE_PROFILE,
        FROZEN_PARTICIPATION_CANDIDATE_PROFILE,
    )
    result = run_outcome_profile_experiment(
        features,
        start_season=FROZEN_PARTICIPATION_START_SEASON,
        profiles=profiles,
        regressor="ridge",
        min_edge=0.02,
        min_train_games=FROZEN_PARTICIPATION_MIN_TRAIN_GAMES,
        ridge_alpha=FROZEN_PARTICIPATION_RIDGE_ALPHA,
    )
    output = _artifacts_root() / "participation_experiments" / run_id()
    atomic_csv(result.summary, output / "summary.csv")
    atomic_csv(result.season_summary, output / "season_summary.csv")
    atomic_parquet(result.predictions, output / "predictions.parquet")
    comparison_input = result.predictions.rename(columns={"feature_profile": "feature_set"})
    paired = pd.concat(
        [
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set=FROZEN_PARTICIPATION_BASELINE_PROFILE,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set=FROZEN_PARTICIPATION_BASELINE_PROFILE,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    ).rename(
        columns={
            "baseline_feature_set": "baseline_feature_profile",
            "candidate_feature_set": "candidate_feature_profile",
        }
    )
    atomic_csv(paired, output / "paired_comparisons.csv")
    configuration = {
        "command": "participation-ablation",
        "hypothesis_frozen_before_scoring": True,
        "start_season": FROZEN_PARTICIPATION_START_SEASON,
        "regressor": "ridge",
        "ridge_alpha": FROZEN_PARTICIPATION_RIDGE_ALPHA,
        "calibration_method": "none",
        "profiles": list(profiles),
        "baseline_profile": FROZEN_PARTICIPATION_BASELINE_PROFILE,
        "candidate_profile": FROZEN_PARTICIPATION_CANDIDATE_PROFILE,
        "method": "market_residual",
        "min_edge": 0.02,
        "min_train_games": FROZEN_PARTICIPATION_MIN_TRAIN_GAMES,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "timing": {"total_seconds": perf_counter() - command_started},
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="participation-ablation",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_availability_ablation(args: argparse.Namespace) -> None:
    """Run the predeclared learned-versus-fixed availability comparison."""

    command_started = perf_counter()
    baseline_features = _load_features(args.baseline_features)
    learned_features = _load_features(args.learned_features)
    results = []
    for method, features in (
        ("fixed", baseline_features),
        ("learned", learned_features),
    ):
        result = run_outcome_profile_experiment(
            features,
            start_season=FROZEN_AVAILABILITY_START_SEASON,
            profiles=(FROZEN_AVAILABILITY_PROFILE,),
            regressor="ridge",
            min_edge=0.02,
            min_train_games=FROZEN_AVAILABILITY_MIN_TRAIN_GAMES,
            ridge_alpha=FROZEN_AVAILABILITY_RIDGE_ALPHA,
        )
        result.summary.insert(0, "availability_method", method)
        result.season_summary.insert(0, "availability_method", method)
        result.predictions.insert(0, "availability_method", method)
        results.append(result)
    summary = pd.concat([result.summary for result in results], ignore_index=True)
    season_summary = pd.concat([result.season_summary for result in results], ignore_index=True)
    predictions = pd.concat([result.predictions for result in results], ignore_index=True)
    comparison_input = predictions.rename(columns={"availability_method": "feature_set"})
    paired = pd.concat(
        [
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set="fixed",
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set="fixed",
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    ).rename(
        columns={
            "baseline_feature_set": "baseline_availability_method",
            "candidate_feature_set": "candidate_availability_method",
        }
    )
    output = _artifacts_root() / "availability_experiments" / run_id()
    atomic_csv(summary, output / "summary.csv")
    atomic_csv(season_summary, output / "season_summary.csv")
    atomic_parquet(predictions, output / "predictions.parquet")
    atomic_csv(paired, output / "paired_comparisons.csv")
    configuration = {
        "command": "availability-ablation",
        "hypothesis_frozen_before_scoring": True,
        "start_season": FROZEN_AVAILABILITY_START_SEASON,
        "regressor": "ridge",
        "ridge_alpha": FROZEN_AVAILABILITY_RIDGE_ALPHA,
        "calibration_method": "none",
        "feature_profile": FROZEN_AVAILABILITY_PROFILE,
        "baseline_availability_method": "fixed",
        "candidate_availability_method": "learned",
        "method": "market_residual",
        "min_edge": 0.02,
        "min_train_games": FROZEN_AVAILABILITY_MIN_TRAIN_GAMES,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "baseline_provenance": artifact_provenance(configuration, args.baseline_features),
        "learned_provenance": artifact_provenance(configuration, args.learned_features),
        "timing": {"total_seconds": perf_counter() - command_started},
    }
    atomic_json(metadata, output / "metadata.json")
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_player_model_selection(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = _load_features(args.features)
    result = run_frozen_player_model_selection(
        features,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
    )
    output = _artifacts_root() / "player_model_selection" / run_id()
    atomic_parquet(result.raw_predictions, output / "raw_predictions.parquet")
    atomic_parquet(result.predictions, output / "candidate_predictions.parquet")
    atomic_csv(result.summary, output / "candidate_summary.csv")
    atomic_csv(result.season_summary, output / "candidate_season_summary.csv")
    atomic_csv(result.nested.candidate_validation, output / "nested_candidate_validation.csv")
    atomic_csv(result.nested.fold_summary, output / "nested_fold_summary.csv")
    atomic_csv(result.nested.summary, output / "nested_summary.csv")
    atomic_csv(result.nested.season_summary, output / "nested_season_summary.csv")
    atomic_parquet(result.nested.predictions, output / "nested_predictions.parquet")

    baseline_id = player_model_candidate_id("base", 10.0, "none")
    nested_seasons = result.nested.predictions["season"].unique()
    nested_baseline = result.predictions.loc[
        result.predictions["candidate_id"].eq(baseline_id)
        & result.predictions["season"].isin(nested_seasons)
    ].copy()
    nested_baseline["feature_set"] = baseline_id
    nested_selected = result.nested.predictions.copy()
    nested_selected["feature_set"] = "nested_selected"
    comparison_input = pd.concat([nested_baseline, nested_selected], ignore_index=True)
    nested_paired = pd.concat(
        [
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set=baseline_id,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set=baseline_id,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    atomic_csv(nested_paired, output / "nested_paired_comparisons.csv")

    configuration = {
        "command": "player-model-selection",
        "budget_status": "FROZEN_BEFORE_EVALUATION",
        "raw_start_season": FROZEN_PLAYER_RAW_START_SEASON,
        "evaluation_start_season": FROZEN_PLAYER_EVALUATION_START_SEASON,
        "first_nested_test_season": FROZEN_PLAYER_FIRST_TEST_SEASON,
        "validation_seasons": FROZEN_PLAYER_VALIDATION_SEASONS,
        "profiles": list(FROZEN_PLAYER_MODEL_PROFILES),
        "ridge_alphas": list(FROZEN_PLAYER_RIDGE_ALPHAS),
        "calibration_methods": list(FROZEN_PLAYER_CALIBRATIONS),
        "min_calibration_games": FROZEN_PLAYER_MIN_CALIBRATION_GAMES,
        "baseline_candidate_id": baseline_id,
        "selection_rule": "validation_accuracy_desc,brier_score_asc,candidate_id_asc",
        "min_edge": args.min_edge,
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "raw_configurations": int(result.raw_predictions["raw_candidate_id"].nunique()),
        "candidate_configurations": int(result.predictions["candidate_id"].nunique()),
        "outer_test_games": int(result.nested.predictions["home_cover"].notna().sum()),
        "timing": {"total_seconds": perf_counter() - command_started},
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="player-model-selection",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_anytime_compare(args: argparse.Namespace) -> None:
    predictions = pd.read_parquet(args.predictions)
    if args.feature_set_column != "feature_set":
        predictions = predictions.rename(columns={args.feature_set_column: "feature_set"})
    trace = paired_anytime_comparisons(
        predictions,
        baseline_feature_set=args.baseline_feature_set,
        metric=args.metric,
        block=args.block,
        alpha=args.alpha,
        prior_variance=args.prior_variance,
        target_games=args.target_games,
        per_game_variance_proxy=args.per_game_variance_proxy,
        intraclass_correlation=args.intraclass_correlation,
    )
    summary = anytime_summary(trace)
    output = _artifacts_root() / "anytime" / run_id()
    atomic_csv(trace, output / "trace.csv")
    atomic_csv(summary, output / "summary.csv")
    metadata = {
        "command": "anytime compare",
        "predictions": str(args.predictions),
        "baseline_feature_set": args.baseline_feature_set,
        "metric": args.metric,
        "block": args.block,
        "alpha": args.alpha,
        "target_games": args.target_games,
        "per_game_variance_proxy": args.per_game_variance_proxy,
        "intraclass_correlation": args.intraclass_correlation,
        "artifact": str(output),
        "summary": summary.to_dict(orient="records"),
    }
    atomic_json(metadata, output / "metadata.json")
    _print_json(metadata)


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the evaluation, experiment and ablation commands."""

    backtest = subparsers.add_parser("backtest", help="run expanding weekly evaluation")
    _add_features_arg(backtest)
    _add_season_range_args(backtest, 2018, None)
    backtest.add_argument("--model", choices=MODEL_NAMES, default="logistic")
    backtest.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="full")
    backtest.add_argument("--min-edge", type=float, default=0.02)
    backtest.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    backtest.add_argument("--initial-bankroll", type=float, default=100.0)
    backtest.add_argument("--kelly-multiplier", type=float, default=0.25)
    backtest.add_argument("--max-bet-fraction", type=float, default=0.02)
    backtest.add_argument("--max-week-fraction", type=float, default=0.10)
    backtest.add_argument("--probability-haircut", type=float, default=0.0)
    backtest.add_argument("--bankroll-paths", type=int, default=5_000)
    backtest.add_argument("--bankroll-seed", type=int, default=20260812)
    _add_bootstrap_args(backtest)
    backtest.set_defaults(handler=_cmd_backtest)

    nested = subparsers.add_parser(
        "nested-evaluate",
        help="select configurations on prior seasons and score untouched outer seasons",
    )
    _add_features_arg(nested)
    nested.add_argument("--first-test-season", type=int, default=2018)
    nested.add_argument("--last-test-season", type=int, default=current_year - 1)
    nested.add_argument("--validation-seasons", type=int, default=2)
    nested.add_argument("--selection-metric", choices=SELECTION_METRICS, default="brier_score")
    nested.add_argument(
        "--candidates",
        default=format_candidates(DEFAULT_EVALUATION_CANDIDATES),
        help="comma-separated frozen search budget of model:feature_set entries",
    )
    nested.add_argument("--min-edge", type=float, default=0.02)
    nested.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    _add_bootstrap_args(nested)
    nested.add_argument("--dependence-permutations", type=int, default=1_000)
    nested.add_argument("--dependence-seed", type=int, default=20260812)
    nested.set_defaults(handler=_cmd_nested_evaluate)

    dependence = subparsers.add_parser(
        "dependence-audit", help="measure serial dependence in saved prediction errors"
    )
    dependence.add_argument("--predictions", type=Path, required=True)
    dependence.add_argument("--permutations", type=int, default=1_000)
    dependence.add_argument("--seed", type=int, default=20260812)
    dependence.set_defaults(handler=_cmd_dependence_audit)

    experiment = subparsers.add_parser(
        "experiment", help="feature-set comparisons and the declarative experiment pipeline"
    )
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)

    experiment_compare = experiment_commands.add_parser(
        "compare", help="compare feature sets with identical walk-forward windows"
    )
    _add_features_arg(experiment_compare)
    experiment_compare.add_argument("--start-season", type=int, default=2022)
    experiment_compare.add_argument("--model", choices=MODEL_NAMES, default="logistic")
    experiment_compare.add_argument("--feature-sets", default=",".join(DEFAULT_EXPERIMENT_SETS))
    experiment_compare.add_argument(
        "--baseline-feature-set",
        choices=tuple(FEATURE_SETS),
        help="paired comparison baseline; defaults to the first requested feature set",
    )
    _add_bootstrap_args(experiment_compare)
    experiment_compare.add_argument("--min-edge", type=float, default=0.02)
    experiment_compare.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    experiment_compare.set_defaults(handler=_cmd_experiment)

    experiment_run = experiment_commands.add_parser(
        "run",
        help=(
            "run a declarative experiment spec: reliability check -> screen -> bootstrap -> "
            "mechanical classification -> registry record -> provenance stamp"
        ),
    )
    experiment_run.add_argument("spec", type=Path, help="path to a JSON experiment spec")
    _add_features_arg(experiment_run, help_text="override the feature table")
    experiment_run.add_argument(
        "--market-root",
        type=Path,
        default=_data_root() / "market" / "raw",
        help="historical odds snapshot root used by population.grade='opener'",
    )
    experiment_run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the would-be registry record and exit; writes nothing to disk",
    )
    experiment_run.add_argument(
        "--replace",
        action="store_true",
        help="allow overwriting an existing registry entry of the same name",
    )
    experiment_run.set_defaults(handler=_cmd_experiment_run)

    experiment_verify = experiment_commands.add_parser(
        "verify",
        help=(
            "check each registry row's linked artifact_directory resolves on disk "
            "and surface path/identity inconsistencies (read-only)"
        ),
    )
    experiment_verify.add_argument(
        "--artifacts-root",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "extra root to resolve relative artifact_directory paths against "
            "(repeatable); defaults to $NFL_ATS_ARTIFACTS_DIR or ./artifacts. "
            "Absolute stored paths are always checked as-is."
        ),
    )
    experiment_verify.add_argument(
        "--require-links",
        action="store_true",
        help=(
            "exit non-zero if any linked artifact is missing. Leave unset in CI "
            "on a fresh clone: artifacts/ is gitignored and legitimately absent, "
            "so a missing link is not by itself a defect -- the row's hashes are."
        ),
    )
    experiment_verify.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="emit machine-readable JSON instead of a human table",
    )
    experiment_verify.set_defaults(handler=_cmd_experiment_verify)

    margin_backtest = subparsers.add_parser(
        "margin-backtest",
        help="compare market, fair-margin, residual-margin, straight-up, and direct ATS models",
    )
    _add_features_arg(margin_backtest)
    margin_backtest.add_argument("--start-season", type=int, default=2018)
    _add_regressor_args(margin_backtest)
    _add_feature_profile_arg(margin_backtest, default="base")
    margin_backtest.add_argument(
        "--methods",
        default=",".join(OUTCOME_METHODS),
        help=f"comma-separated subset of: {','.join(OUTCOME_METHODS)}",
    )
    margin_backtest.add_argument("--min-edge", type=float, default=0.02)
    margin_backtest.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    margin_backtest.add_argument(
        "--probability-method",
        choices=RESIDUAL_SMOOTHING_METHODS,
        # Default UNCHANGED (2026-08-19): this backs historical/research
        # backtests broadly, so it stays "ecdf" unless a caller explicitly
        # asks for "gaussian" -- e.g. to build the matching margins/
        # evaluation a --probability-method gaussian margin-predict forecast
        # needs to activate SYNCHRONIZED (nfl_ats.active_model).
        default="ecdf",
        help="how home_cover_probability is read off the out-of-time residual "
        "sample for every margin-model method scored (market_residual, fair_margin)",
    )
    _add_bootstrap_args(margin_backtest, samples=1_000)
    margin_backtest.set_defaults(handler=_cmd_margin_backtest)

    player_ablation = subparsers.add_parser(
        "player-ablation",
        help="compare player feature families on the residual-margin ATS model",
    )
    _add_features_arg(player_ablation, "game_features_player.parquet")
    player_ablation.add_argument("--start-season", type=int, default=2018)
    _add_regressor_args(player_ablation)
    player_ablation.add_argument("--profiles", default=",".join(DEFAULT_PLAYER_PROFILE_SETS))
    player_ablation.add_argument(
        "--baseline-profile", choices=MARGIN_FEATURE_PROFILES, default="base"
    )
    player_ablation.add_argument("--min-edge", type=float, default=0.02)
    player_ablation.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    _add_bootstrap_args(player_ablation)
    player_ablation.add_argument("--first-nested-test-season", type=int, default=2020)
    player_ablation.add_argument("--validation-seasons", type=int, default=2)
    player_ablation.set_defaults(handler=_cmd_player_ablation)

    participation_ablation = subparsers.add_parser(
        "participation-ablation",
        help="run the frozen player-value versus participation-value comparison",
    )
    _add_features_arg(participation_ablation, "game_features_player_participation.parquet")
    _add_bootstrap_args(participation_ablation, seed=20260813)
    participation_ablation.set_defaults(handler=_cmd_participation_ablation)

    availability_ablation = subparsers.add_parser(
        "availability-ablation",
        help="run the frozen learned-versus-fixed availability comparison",
    )
    availability_ablation.add_argument(
        "--baseline-features",
        type=Path,
        default=_data_root() / "processed" / "game_features_player_value.parquet",
    )
    availability_ablation.add_argument(
        "--learned-features",
        type=Path,
        default=_data_root() / "processed" / "game_features_player_learned_availability.parquet",
    )
    _add_bootstrap_args(availability_ablation, seed=20260813)
    availability_ablation.set_defaults(handler=_cmd_availability_ablation)

    player_selection = subparsers.add_parser(
        "player-model-selection",
        help="run the frozen nested player profile, Ridge, and calibration budget",
    )
    player_selection.add_argument(
        "--features",
        type=Path,
        default=_data_root() / "processed" / "game_features_player_value.parquet",
    )
    player_selection.add_argument("--min-edge", type=float, default=0.02)
    # This command runs the FROZEN selection, so its default is pinned to the
    # value that selection was scored under. It must not drift with the live
    # default, or re-running it would stop reproducing the recorded artifact.
    player_selection.add_argument(
        "--min-train-games", type=int, default=FROZEN_PLAYER_MIN_TRAIN_GAMES
    )
    _add_bootstrap_args(player_selection, seed=20260813)
    player_selection.set_defaults(handler=_cmd_player_model_selection)


def register_anytime(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the anytime-valid comparison commands."""

    anytime = subparsers.add_parser(
        "anytime", help="anytime-valid (continuous-monitoring) paired comparisons"
    )
    anytime_commands = anytime.add_subparsers(dest="anytime_command", required=True)
    anytime_compare = anytime_commands.add_parser(
        "compare",
        help=(
            "confidence-sequence/e-value trace for a paired feature-set comparison; same "
            "input contract as experiments.paired_feature_comparisons, valid under peeking "
            "after every week/season instead of only at a single predeclared sample size"
        ),
    )
    anytime_compare.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="parquet with feature_set/game_id/season/week/home_cover/"
        "home_cover_probability columns (or --feature-set-column to rename one)",
    )
    anytime_compare.add_argument(
        "--feature-set-column",
        default="feature_set",
        help="column to rename to 'feature_set' first, e.g. 'method' for a cfb-benchmark "
        "predictions.parquet",
    )
    anytime_compare.add_argument("--baseline-feature-set", required=True)
    anytime_compare.add_argument(
        "--metric", choices=ANYTIME_METRICS, default="accuracy_improvement"
    )
    anytime_compare.add_argument("--block", choices=("week", "season"), default="week")
    anytime_compare.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    anytime_compare.add_argument(
        "--target-games",
        type=int,
        default=DEFAULT_TARGET_GAMES,
        help="horizon the default mixing variance is tuned for (ignored if --prior-variance "
        "is passed); default matches the rotation registry's own 3-season window",
    )
    anytime_compare.add_argument(
        "--prior-variance",
        type=float,
        default=None,
        help="override the mixing variance directly instead of deriving it from --target-games",
    )
    anytime_compare.add_argument(
        "--per-game-variance-proxy",
        type=float,
        default=1.0,
        help="upper bound on one game's own variance; 1.0 is Hoeffding's worst case for a "
        "[-1, 1] variable and needs no assumption. See docs/anytime_valid.md for the "
        "measured, less-conservative value this project uses (0.55) and why",
    )
    anytime_compare.add_argument(
        "--intraclass-correlation",
        type=float,
        default=0.0,
        help="Kish design-effect correlation, 0-1; 0.0 (independence -- disjoint teams, no "
        "shared outcome mechanism) is this project's standing decision, not an estimate. "
        "1.0 (every game in a block moves together) is the assumption-free worst case, "
        "kept available for stress-testing. See docs/anytime_valid.md for the argument "
        "and why an unmeasured 0.10 pad and an auto-estimated value were both rejected",
    )
    anytime_compare.set_defaults(handler=_cmd_anytime_compare)
