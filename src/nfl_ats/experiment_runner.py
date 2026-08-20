"""The agentless experiment pipeline: put a declarative spec in, get an answer out.

Motivation (2026-08-18 session): three separate hand-transcription defects were
caught in one sitting -- a 100x fraction-vs-points scaling bug, a sign bug, and
a corrupted source path, all in numbers a human copied from console output into
``registry/weak_signals.json`` by hand. Every piece needed to avoid that error
class already existed as separate machinery:

- ``nfl_ats.experiments.paired_feature_comparisons`` -- the block-bootstrap
  engine with the D4 degeneracy guard.
- ``nfl_ats.estimation_variance`` -- ``MIN_BLOCKS_FOR_INTERVAL``,
  ``guard_block_count``, and the honest refit-correction band this module
  cites (see :data:`HONEST_REFIT_WIDENING_UPPER_BOUND`).
- ``nfl_ats.weak_signals`` -- ``record_signal``/``validate_closure``, which
  IS the closing-ground taxonomy encoded as a validator, not prose.
- ``nfl_ats.provenance.write_experiment_artifact`` -- the run-provenance
  stamp every CLI command already gets as a side effect of its artifact write.
- ``scripts/penalty_discipline_interval.py`` /
  ``scripts/nfl_bias_battery_screen.py`` -- the subset-vs-complement,
  week-blocked joint bootstrap, full-slate-scaling pattern this module
  generalizes into a registry of named, reusable flag builders.

What was missing was the GLUE: a single entry point that runs the whole loop
(reliability check -> screen -> bootstrap -> mechanical classification ->
registry record -> provenance stamp) from a declarative spec, computing every
registry field directly from data so there is no point where a human retypes
a number.

Both ``experiment_type: "subset_bias"`` (a pregame-safe boolean flag vs. its
complement, cover rate vs. the spread) and ``"feature_arm"`` (profile-vs-profile
or ridge_alpha-vs-ridge_alpha, via ``nfl_ats.outcomes.walk_forward_outcomes`` +
``nfl_ats.experiments.paired_feature_comparisons``, the pattern
``scripts/ridge_alpha_promotion_eval.py``'s ``evaluate_arm`` demonstrated for
the opener grade specifically) are implemented. ``subset_bias`` additionally
supports ``population.grade: "opener"`` (a population loader analogous to
``clv.opener_pick_evaluation``, restricted to the paired Tuesday-opener
archive); ``feature_arm`` supports only ``grade: "close"`` this pass.

**Mechanical classification (AGENTS.md, "An interval crossing zero is NOT
grounds for rejection", binding).** This runner writes exactly one non-default
terminal verdict on its own authority: ``refuted_mechanism`` /
``wrong_sign_resolved``, and ONLY when both hold:

1. the PRIMARY (week-blocked) interval sits entirely below zero, and
2. the inflation factor needed to widen that interval back across zero
   exceeds :data:`HONEST_REFIT_WIDENING_UPPER_BOUND` -- the documented
   one-sided 95%% upper bound on how much an honest, refit-aware interval
   could widen a naive one for a fit-changing comparison
   (``docs/estimation_variance.md``: "...1.293x to 1.003x (one-sided 95%%
   upper bound 1.099x)"; also the reviewer note on
   ``mod06_js_shrinkage_position_prior_cfb`` in
   ``registry/weak_signals.json``, which refused closure at a required
   1.082x widening because it sat *inside* that band).

Every other outcome -- including a naive interval that excludes zero but
would need less than 1.099x widening to re-cross -- is recorded
``unresolved_below_power`` with no ``closing_ground``. The runner NEVER
produces ``bounded_by_control`` or a reliability-grounded
``no_split_half_reliability`` closure; both remain human adjudications, and a
spec cannot request them (there is no field for it).
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.clv import CLOSE_LABEL_PRIORITY, build_pairing_table, close_reference_table
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, TEAM_ABBREVIATION_ALIASES
from nfl_ats.estimation_variance import (
    MIN_BLOCKS_FOR_INTERVAL,
    guard_block_count,
)
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.margin import MARGIN_FEATURE_PROFILES
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND
from nfl_ats.outcomes import walk_forward_outcomes
from nfl_ats.pbp import latest_pbp_snapshot, load_pbp_snapshot
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact
from nfl_ats.weak_signals import (
    LEAGUES,
    WeakSignal,
    default_registry_path,
    load_registry,
    record_signal,
    save_registry,
)

#: One-sided 95% upper bound on how much an honest, refit-aware interval can
#: widen a naive (game-resample-only, single-fit) one for a comparison that
#: changes what gets FIT (as opposed to only how the residual is read).
#: MEASURED, not typed in from memory: ``docs/estimation_variance.md`` Part II
#: reports the flagship real-CFB comparison's honest factor moving "...1.293x
#: to 1.003x (one-sided 95% upper bound 1.099x)" after fixing a double-counted
#: interaction term, and its per-comparison table (sec 11) lists every audited
#: entry at "refit 1.003x" with this same upper bound. This is the SAME
#: constant the registry's own reviewer adjudication cites verbatim on
#: ``mod06_js_shrinkage_position_prior_cfb`` (``registry/weak_signals.json``):
#: closure was refused there because re-crossing zero needed only a 1.082x
#: widening, "inside the documented 1.003-1.099x honest refit-correction band".
HONEST_REFIT_WIDENING_UPPER_BOUND = 1.099

DEFAULT_SAMPLES = 20_000
DEFAULT_CONFIDENCE = 0.95

EXPERIMENT_TYPES = ("subset_bias", "feature_arm")
GRADES = ("close", "opener")
ENDPOINT_METRICS = ("accuracy", "brier", "logloss")
BLOCK_KINDS = ("week", "season")
RELIABILITY_METHODS = ("split_half", "not_applicable")

_BLOCK_COLUMNS = {"week": "week_block", "season": "season"}


class ExperimentSpecError(ValueError):
    """Raised when a declarative experiment spec fails strict validation.

    A ``ValueError`` subclass so the CLI reports a user-facing error rather
    than a traceback, matching ``WeakSignalError``/``ExperimentRecordError``.
    """


class ExperimentRunnerError(ValueError):
    """Raised when a valid spec cannot be run (unknown builder, unsupported
    grade, unimplemented experiment type, empty population after filtering,
    a reliability-check/builder mismatch, or a registry-locking failure).
    """


# ---------------------------------------------------------------------------
# Spec schema and validation
# ---------------------------------------------------------------------------

_TOP_LEVEL_FIELDS = frozenset(
    {
        "name",
        "hypothesis",
        "experiment_type",
        "population",
        "construct",
        "endpoints",
        "blocking",
        "samples",
        "seed",
        "reliability_check",
    }
)
_POPULATION_FIELDS = frozenset({"league", "seasons", "grade"})
_CONSTRUCT_FIELDS_SUBSET_BIAS = frozenset({"flag_builder", "params"})
_CONSTRUCT_FIELDS_FEATURE_ARM = frozenset({"baseline", "candidate"})
_FEATURE_ARM_ARM_FIELDS = frozenset({"feature_profile", "ridge_alpha"})
_ENDPOINTS_FIELDS = frozenset({"primary", "secondary"})
_BLOCKING_FIELDS = frozenset({"primary", "secondary"})
_RELIABILITY_FIELDS = frozenset({"method", "reason"})

#: ``fit_margin_model``'s own default -- not a project-authoritative constant,
#: just what a ``feature_arm`` spec's arm gets if it omits ``ridge_alpha``.
DEFAULT_FEATURE_ARM_RIDGE_ALPHA = 10.0


@dataclass(frozen=True)
class FeatureArmConfig:
    """One ``feature_arm`` arm: a feature profile and a ridge penalty.

    ``feature_profile`` must be a name registered in
    ``margin.MARGIN_FEATURE_PROFILES``; ``ridge_alpha`` defaults to
    ``fit_margin_model``'s own default (10.0) when the spec omits it.
    """

    feature_profile: str
    ridge_alpha: float


@dataclass(frozen=True)
class ExperimentSpec:
    """A validated, immutable declarative experiment spec.

    Mirrors ``weak_signals.WeakSignal``'s validate-once-then-trust-the-
    dataclass shape: every field here has already survived
    :func:`experiment_spec_from_payload`'s strict checks, so downstream code
    never re-validates.

    ``flag_builder``/``construct_params`` are populated only for
    ``experiment_type == "subset_bias"`` (empty string / empty dict
    otherwise); ``feature_arm_baseline``/``feature_arm_candidate`` are
    populated only for ``experiment_type == "feature_arm"`` (``None``
    otherwise) -- which pair is live is entirely determined by
    ``experiment_type``, never guessed downstream.
    """

    name: str
    hypothesis: str
    experiment_type: str
    league: str
    seasons: tuple[int, int]
    grade: str
    flag_builder: str
    construct_params: dict[str, Any]
    feature_arm_baseline: FeatureArmConfig | None
    feature_arm_candidate: FeatureArmConfig | None
    endpoint_primary: str
    endpoint_secondary: tuple[str, ...]
    block_primary: str
    block_secondary: str | None
    samples: int
    seed: int
    reliability_method: str
    reliability_reason: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentSpecError(message)


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field_name!r} must be an object")
    return dict(value)


def _feature_arm_config_from_payload(value: Any, field_name: str) -> FeatureArmConfig:
    payload = _require_dict(value, field_name)
    unknown = sorted(set(payload).difference(_FEATURE_ARM_ARM_FIELDS))
    _require(not unknown, f"{field_name} has unknown fields: {', '.join(unknown)}")
    _require("feature_profile" in payload, f"{field_name}.feature_profile is required")
    feature_profile = str(payload["feature_profile"])
    _require(
        feature_profile in MARGIN_FEATURE_PROFILES,
        f"{field_name}.feature_profile must be one of {', '.join(MARGIN_FEATURE_PROFILES)}, "
        f"got {feature_profile!r}",
    )
    ridge_alpha_raw = payload.get("ridge_alpha", DEFAULT_FEATURE_ARM_RIDGE_ALPHA)
    _require(
        isinstance(ridge_alpha_raw, (int, float)) and not isinstance(ridge_alpha_raw, bool),
        f"{field_name}.ridge_alpha must be a number",
    )
    ridge_alpha = float(ridge_alpha_raw)
    _require(ridge_alpha > 0.0, f"{field_name}.ridge_alpha must be positive")
    return FeatureArmConfig(feature_profile=feature_profile, ridge_alpha=ridge_alpha)


def experiment_spec_from_payload(payload: dict[str, Any]) -> ExperimentSpec:
    unknown = sorted(set(payload).difference(_TOP_LEVEL_FIELDS))
    _require(not unknown, f"Experiment spec has unknown top-level fields: {', '.join(unknown)}")
    for required_field in (
        "name",
        "hypothesis",
        "experiment_type",
        "population",
        "construct",
        "seed",
    ):
        _require(
            required_field in payload,
            f"Experiment spec is missing required field {required_field!r}",
        )

    name = str(payload["name"]).strip()
    _require(bool(name), "Experiment spec 'name' must be non-empty")
    hypothesis = str(payload["hypothesis"]).strip()
    _require(bool(hypothesis), "Experiment spec 'hypothesis' must be non-empty prose")

    experiment_type = payload["experiment_type"]
    _require(
        experiment_type in EXPERIMENT_TYPES,
        f"Unknown experiment_type {experiment_type!r}; "
        f"expected one of {', '.join(EXPERIMENT_TYPES)}",
    )

    population = _require_dict(payload["population"], "population")
    unknown_pop = sorted(set(population).difference(_POPULATION_FIELDS))
    _require(not unknown_pop, f"population has unknown fields: {', '.join(unknown_pop)}")
    league = population.get("league")
    _require(
        league in LEAGUES, f"population.league must be one of {', '.join(LEAGUES)}, got {league!r}"
    )
    seasons = population.get("seasons")
    _require(
        isinstance(seasons, (list, tuple)) and len(seasons) == 2,
        "population.seasons must be a two-element [start, end]",
    )
    assert isinstance(seasons, (list, tuple))
    season_start, season_end = int(seasons[0]), int(seasons[1])
    _require(season_start <= season_end, "population.seasons is out of order")
    grade = population.get("grade", "close")
    _require(grade in GRADES, f"population.grade must be one of {', '.join(GRADES)}, got {grade!r}")

    construct = _require_dict(payload["construct"], "construct")
    flag_builder = ""
    construct_params: dict[str, Any] = {}
    feature_arm_baseline: FeatureArmConfig | None = None
    feature_arm_candidate: FeatureArmConfig | None = None
    if experiment_type == "subset_bias":
        unknown_construct = sorted(set(construct).difference(_CONSTRUCT_FIELDS_SUBSET_BIAS))
        _require(
            not unknown_construct,
            f"construct has unknown fields for subset_bias: {', '.join(unknown_construct)}",
        )
        _require("flag_builder" in construct, "construct.flag_builder is required for subset_bias")
        flag_builder = str(construct["flag_builder"])
        construct_params = _require_dict(construct.get("params", {}), "construct.params")
    else:
        assert experiment_type == "feature_arm"
        unknown_construct = sorted(set(construct).difference(_CONSTRUCT_FIELDS_FEATURE_ARM))
        _require(
            not unknown_construct,
            f"construct has unknown fields for feature_arm: {', '.join(unknown_construct)}",
        )
        _require("baseline" in construct, "construct.baseline is required for feature_arm")
        _require("candidate" in construct, "construct.candidate is required for feature_arm")
        feature_arm_baseline = _feature_arm_config_from_payload(
            construct["baseline"], "construct.baseline"
        )
        feature_arm_candidate = _feature_arm_config_from_payload(
            construct["candidate"], "construct.candidate"
        )

    endpoints = _require_dict(
        payload.get("endpoints", {"primary": "accuracy", "secondary": []}), "endpoints"
    )
    unknown_endpoints = sorted(set(endpoints).difference(_ENDPOINTS_FIELDS))
    _require(not unknown_endpoints, f"endpoints has unknown fields: {', '.join(unknown_endpoints)}")
    endpoint_primary = endpoints.get("primary", "accuracy")
    _require(
        endpoint_primary == "accuracy",
        "endpoints.primary must be 'accuracy' (the project's primary bar)",
    )
    endpoint_secondary_raw = endpoints.get("secondary", [])
    _require(isinstance(endpoint_secondary_raw, list), "endpoints.secondary must be a list")
    for metric in endpoint_secondary_raw:
        _require(
            metric in ENDPOINT_METRICS,
            f"endpoints.secondary has unknown metric {metric!r}; "
            f"expected one of {', '.join(ENDPOINT_METRICS)}",
        )
    _require(
        len(set(endpoint_secondary_raw)) == len(endpoint_secondary_raw),
        "endpoints.secondary has duplicates",
    )
    _require(
        experiment_type == "feature_arm" or not endpoint_secondary_raw,
        "endpoints.secondary must be empty for subset_bias (a raw cover-rate comparison has no "
        "probabilistic prediction to score brier/logloss against); reserved for feature_arm",
    )
    endpoint_secondary = tuple(endpoint_secondary_raw)

    blocking = _require_dict(
        payload.get("blocking", {"primary": "week", "secondary": "season"}), "blocking"
    )
    unknown_blocking = sorted(set(blocking).difference(_BLOCKING_FIELDS))
    _require(not unknown_blocking, f"blocking has unknown fields: {', '.join(unknown_blocking)}")
    block_primary = blocking.get("primary", "week")
    _require(
        block_primary in BLOCK_KINDS, f"blocking.primary must be one of {', '.join(BLOCK_KINDS)}"
    )
    block_secondary = blocking.get("secondary", "season")
    if block_secondary is not None:
        _require(
            block_secondary in BLOCK_KINDS,
            f"blocking.secondary must be one of {', '.join(BLOCK_KINDS)}",
        )
        _require(
            block_secondary != block_primary, "blocking.secondary must differ from blocking.primary"
        )

    samples = int(payload.get("samples", DEFAULT_SAMPLES))
    _require(samples >= 10, "samples must be at least 10")

    # No wall-clock nondeterminism: every run must be reproducible from the
    # spec alone, so 'seed' has no default and must be explicit.
    seed_raw = payload["seed"]
    _require(
        isinstance(seed_raw, int) and not isinstance(seed_raw, bool), "seed must be an integer"
    )
    seed = int(seed_raw)

    reliability_check = _require_dict(payload["reliability_check"], "reliability_check")
    unknown_reliability = sorted(set(reliability_check).difference(_RELIABILITY_FIELDS))
    _require(
        not unknown_reliability,
        f"reliability_check has unknown fields: {', '.join(unknown_reliability)}",
    )
    _require("method" in reliability_check, "reliability_check.method is required")
    reliability_method = reliability_check["method"]
    _require(
        reliability_method in RELIABILITY_METHODS,
        f"reliability_check.method must be one of {', '.join(RELIABILITY_METHODS)}",
    )
    reliability_reason = str(reliability_check.get("reason", ""))
    if reliability_method == "not_applicable":
        _require(
            bool(reliability_reason.strip()),
            "reliability_check.reason is required and must be non-empty "
            "when method is 'not_applicable'",
        )

    return ExperimentSpec(
        name=name,
        hypothesis=hypothesis,
        experiment_type=str(experiment_type),
        league=str(league),
        seasons=(season_start, season_end),
        grade=str(grade),
        flag_builder=flag_builder,
        construct_params=dict(construct_params),
        feature_arm_baseline=feature_arm_baseline,
        feature_arm_candidate=feature_arm_candidate,
        endpoint_primary=str(endpoint_primary),
        endpoint_secondary=endpoint_secondary,
        block_primary=str(block_primary),
        block_secondary=None if block_secondary is None else str(block_secondary),
        samples=samples,
        seed=seed,
        reliability_method=str(reliability_method),
        reliability_reason=reliability_reason,
    )


def load_experiment_spec(path: Path) -> ExperimentSpec:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ExperimentSpecError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ExperimentSpecError(f"{path} must contain a JSON object")
    return experiment_spec_from_payload(payload)


def experiment_spec_to_payload(spec: ExperimentSpec) -> dict[str, Any]:
    """Inverse of :func:`experiment_spec_from_payload`, for provenance hashing."""

    if spec.experiment_type == "subset_bias":
        construct: dict[str, Any] = {
            "flag_builder": spec.flag_builder,
            "params": spec.construct_params,
        }
    else:
        assert spec.feature_arm_baseline is not None
        assert spec.feature_arm_candidate is not None
        construct = {
            "baseline": {
                "feature_profile": spec.feature_arm_baseline.feature_profile,
                "ridge_alpha": spec.feature_arm_baseline.ridge_alpha,
            },
            "candidate": {
                "feature_profile": spec.feature_arm_candidate.feature_profile,
                "ridge_alpha": spec.feature_arm_candidate.ridge_alpha,
            },
        }
    return {
        "name": spec.name,
        "hypothesis": spec.hypothesis,
        "experiment_type": spec.experiment_type,
        "population": {"league": spec.league, "seasons": list(spec.seasons), "grade": spec.grade},
        "construct": construct,
        "endpoints": {"primary": spec.endpoint_primary, "secondary": list(spec.endpoint_secondary)},
        "blocking": {"primary": spec.block_primary, "secondary": spec.block_secondary},
        "samples": spec.samples,
        "seed": spec.seed,
        "reliability_check": {"method": spec.reliability_method, "reason": spec.reliability_reason},
    }


# ---------------------------------------------------------------------------
# subset_bias: the team-game long table shared by every registered builder
# ---------------------------------------------------------------------------


def _canonical_team(team: pd.Series) -> pd.Series:
    return team.map(lambda code: TEAM_ABBREVIATION_ALIASES.get(code, code))


def _base_team_game_table(features: pd.DataFrame) -> pd.DataFrame:
    """REG-season, pushes-dropped, one row per (game, team) side.

    Ported from ``scripts/nfl_bias_battery_screen.py``'s ``build_long_table``
    / ``scripts/penalty_discipline_interval.py``'s ``build_team_game_table``,
    trimmed to the columns every currently-registered builder needs
    (``team_spread``/``spread_line`` for the situational builders,
    ``team``/``season`` for the trait-based ones) and merged into one
    function since both precedent scripts built this table independently.
    """

    required = {
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_cover",
        "spread_line",
        "game_type",
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ExperimentRunnerError(f"Feature table is missing columns: {', '.join(missing)}")

    reg = features.loc[features["game_type"] == "REG"].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["spread_line"] = pd.to_numeric(reg["spread_line"], errors="coerce")

    sides = []
    for is_home, team_col, opponent_col in (
        (True, "home_team", "away_team"),
        (False, "away_team", "home_team"),
    ):
        side = pd.DataFrame(
            {
                "game_id": reg["game_id"],
                "season": reg["season"].astype(int),
                "week": reg["week"].astype(int),
                "team": reg[team_col],
                "opponent": reg[opponent_col],
                "is_home": is_home,
                "team_covered": reg["home_cover"] if is_home else 1.0 - reg["home_cover"],
                "spread_line": reg["spread_line"],
                "team_spread": reg["spread_line"] if is_home else -reg["spread_line"],
            }
        )
        sides.append(side)

    long_df = pd.concat(sides, ignore_index=True)
    # Pushes: home_cover is NaN and must not silently count as a loss/win on
    # either side of any comparison.
    long_df = long_df.loc[long_df["team_covered"].notna()].copy()
    long_df["team"] = _canonical_team(long_df["team"])
    long_df["opponent"] = _canonical_team(long_df["opponent"])
    long_df["week_block"] = long_df["season"] * 100 + long_df["week"]
    return long_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# The named flag-builder registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsetBiasConstruct:
    """What a named flag builder hands back to the generic pipeline.

    ``table`` is the population this construct is defined over (already
    filtered to whatever rows the builder's own trait requires -- e.g. a team
    penalty rate builder drops the first season of local history, which has
    no prior-season rate to lag). ``flag``/``eligible`` are boolean Series
    aligned to ``table.index``.

    ``eligible=None`` means "compare the flag against everyone else in
    ``table``" (the one-sided design ``hc_year_one_fade`` and the bias
    battery use: the fraction of the slate scaling the effect is
    ``n_flag / len(table)``). A non-``None`` ``eligible`` restricts the
    comparison to a named subset of ``table`` that is neither the whole
    population nor just the flag (the two-sided, paired design
    ``penalty_discipline`` uses: quartile 1 vs quartile 4, with quartiles 2-3
    still counted in ``len(table)`` for scaling but excluded from the direct
    comparison). Both are legitimate, already-precedented designs; which one
    applies is a property of the construct, not something the generic
    pipeline guesses.
    """

    table: pd.DataFrame
    flag: pd.Series
    eligible: pd.Series | None
    #: +1 if flag=True favours the stated hypothesis, -1 if it opposes it.
    sign: int
    reliability: float | None
    reliability_pairs: int | None
    reliability_note: str
    population_note: str = ""


@dataclass(frozen=True)
class FlagBuilder:
    name: str
    leagues: tuple[str, ...]
    description: str
    build: Callable[[pd.DataFrame, tuple[int, int], dict[str, Any], Path], SubsetBiasConstruct]


def _flag_home_underdog(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params, repo_root  # unused: no persistent trait, no extra data source
    table = _base_team_game_table(features)
    flag = table["is_home"] & (table["spread_line"] < 0.0)
    return SubsetBiasConstruct(
        table=table,
        flag=flag,
        eligible=None,
        sign=1,
        reliability=None,
        reliability_pairs=None,
        reliability_note=(
            "home_underdog is a per-game situational condition (this game's own line), not a "
            "persistent per-team trait -- there is nothing to split-half."
        ),
    )


def _flag_large_favorite(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, repo_root
    threshold = float(params.get("threshold", 10.0))
    table = _base_team_game_table(features)
    is_favorite = table["team_spread"] > 0.0
    flag = is_favorite & (table["team_spread"] > threshold)
    return SubsetBiasConstruct(
        table=table,
        flag=flag,
        eligible=None,
        sign=-1,  # hypothesis: large favourites are over-priced and cover LESS
        reliability=None,
        reliability_pairs=None,
        reliability_note=(
            "large_favorite is a per-game situational condition (this game's own line), not a "
            "persistent per-team trait -- there is nothing to split-half."
        ),
    )


def _team_season_penalty_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """``mean(penalty)`` over every raw regular-season play where ``posteam == team``.

    Ported verbatim from ``scripts/penalty_discipline_interval.py`` (which
    documents, at length, why this exact definition -- no play-type filter --
    is the one that reproduces the recorded 0.0750 +/- 0.0101 figures).
    """

    plays = pbp.loc[pbp["posteam"].notna()].copy()
    plays["penalty"] = pd.to_numeric(plays["penalty"], errors="coerce").fillna(0.0)
    plays["team"] = _canonical_team(plays["posteam"])
    grouped = plays.groupby(["season", "team"]).agg(
        plays=("penalty", "size"), penalties=("penalty", "sum")
    )
    grouped["rate"] = grouped["penalties"] / grouped["plays"]
    return grouped.reset_index()


def _year_over_year_reliability(rate: pd.DataFrame) -> tuple[float | None, int]:
    ordered = rate.sort_values(["team", "season"]).copy()
    ordered["next_rate"] = ordered.groupby("team")["rate"].shift(-1)
    ordered["next_season"] = ordered.groupby("team")["season"].shift(-1)
    pairs = ordered.loc[ordered["next_season"] == ordered["season"] + 1]
    if pairs.empty:
        return None, 0
    correlation = pairs["rate"].corr(pairs["next_rate"])
    return (None if pd.isna(correlation) else float(correlation)), len(pairs)


def _lag_and_quartile(rate: pd.DataFrame) -> pd.DataFrame:
    ordered = rate.sort_values(["team", "season"]).copy()
    ordered["prev_rate"] = ordered.groupby("team")["rate"].shift(1)
    ordered["prev_season"] = ordered.groupby("team")["season"].shift(1)
    lagged = ordered.loc[ordered["season"] - ordered["prev_season"] == 1].copy()
    lagged["quartile"] = pd.qcut(lagged["prev_rate"], 4, labels=[1, 2, 3, 4]).astype(int)
    return lagged[["team", "season", "prev_rate", "quartile"]]


def _flag_penalty_rate_quartile(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    """Reproduces ``scripts/penalty_discipline_interval.py``'s construct exactly.

    Least-penalized quartile (Q1) vs most-penalized quartile (Q4) of a team's
    PRIOR-season penalty rate (global quartile cut), team-game cover rate.
    Quartiles 2-3 are excluded from the direct comparison but still count
    toward the full-slate scaling denominator (this is the "two-sided,
    paired" design; see :class:`SubsetBiasConstruct`).
    """

    del seasons  # the trait needs the full local history to lag; season-filtering happens after
    pbp_raw_root = Path(params.get("pbp_raw_root", repo_root / "data" / "pbp" / "raw"))
    snapshot = latest_pbp_snapshot(pbp_raw_root)
    pbp = load_pbp_snapshot(snapshot, include_postseason=False)
    rate = _team_season_penalty_rate(pbp)
    reliability, reliability_pairs = _year_over_year_reliability(rate)
    lagged = _lag_and_quartile(rate)

    long_df = _base_team_game_table(features)
    merged = long_df.merge(lagged, on=["team", "season"], how="inner")
    flag = merged["quartile"] == 1
    eligible = merged["quartile"].isin([1, 4])
    return SubsetBiasConstruct(
        table=merged,
        flag=flag,
        eligible=eligible,
        sign=1,  # hypothesis: the least-penalized quartile covers MORE
        reliability=reliability,
        reliability_pairs=reliability_pairs,
        reliability_note=(
            f"Year-over-year Pearson correlation of team-season penalty rate "
            f"(mean(penalty) over every raw regular-season play where posteam==team), "
            f"{reliability_pairs} team-season pairs."
        ),
        population_note=f"PBP snapshot {snapshot.snapshot_id}.",
    )


# ---------------------------------------------------------------------------
# Bias-battery builders: ported from scripts/nfl_bias_battery_screen.py
# ---------------------------------------------------------------------------
#
# ``scripts/nfl_bias_battery_screen.py`` is a measure-only script (never
# writes the registry) that predeclared 17 situational/behavioral cells and
# scored them itself with its own copy of the block-bootstrap machinery this
# module already generalizes. The functions below port its ``build_long_table``
# / ``add_history_features`` / ``build_hypotheses`` flag LOGIC verbatim (same
# masks, same thresholds, same column derivations) into the runner's
# ``SubsetBiasConstruct`` shape, so the already-recorded close-graded
# ``bias_battery_*`` entries can be re-screened at other grades (starting
# with the opener) through this pipeline instead of a second bespoke script.
# Nothing about the CONSTRUCTS is redesigned here -- only the harness they run
# through.

PT_TEAMS = frozenset({"SEA", "SF", "LA", "LAC", "LV"})


def _latest_schedules_snapshot(repo_root: Path) -> Path:
    candidates = sorted((repo_root / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise ExperimentRunnerError(
            f"No data/raw/*/schedules.parquet snapshot found under {repo_root}"
        )
    return candidates[-1]


_BIAS_BATTERY_LONG_HOME = {
    "team": "home_team",
    "opponent": "away_team",
    "own_rest": "home_rest",
    "opp_rest": "away_rest",
    "own_qb_name": "home_qb_name",
}
_BIAS_BATTERY_LONG_AWAY = {
    "team": "away_team",
    "opponent": "home_team",
    "own_rest": "away_rest",
    "opp_rest": "home_rest",
    "own_qb_name": "away_qb_name",
}


def _bias_battery_merged_features(features: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    """REG-season ``features`` inner-joined with the newest schedules snapshot's
    rest/roof/surface/QB-name columns -- ``nfl_bias_battery_screen.load_merged``.
    """

    schedules_path = _latest_schedules_snapshot(repo_root)
    schedules = pd.read_parquet(schedules_path).loc[
        :, ["game_id", "away_rest", "home_rest", "roof", "surface", "away_qb_name", "home_qb_name"]
    ]
    merged = features.merge(schedules, on="game_id", how="inner", validate="one_to_one")
    return merged.loc[merged["game_type"] == "REG"].copy()


def _bias_battery_build_long_table(merged: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, side); pushes dropped -- ``nfl_bias_battery_screen.build_long_table``."""

    merged = merged.copy()
    merged["home_team"] = _canonical_team(merged["home_team"])
    merged["away_team"] = _canonical_team(merged["away_team"])
    merged["gameday"] = pd.to_datetime(merged["gameday"], errors="raise")
    merged["result"] = pd.to_numeric(merged["result"], errors="coerce")
    merged["spread_line"] = pd.to_numeric(merged["spread_line"], errors="coerce")
    merged["temp"] = pd.to_numeric(merged["temp"], errors="coerce")
    merged["gametime_hour"] = pd.to_numeric(
        merged["gametime"].astype(str).str.split(":").str[0], errors="coerce"
    )

    sides = []
    for is_home, cols in ((True, _BIAS_BATTERY_LONG_HOME), (False, _BIAS_BATTERY_LONG_AWAY)):
        side = pd.DataFrame(
            {
                "game_id": merged["game_id"],
                "season": merged["season"].astype(int),
                "week": merged["week"].astype(int),
                "gameday": merged["gameday"],
                "team": merged[cols["team"]],
                "opponent": merged[cols["opponent"]],
                "is_home": is_home,
                "team_covered": merged["home_cover"] if is_home else 1.0 - merged["home_cover"],
                "team_ats_margin": merged["ats_margin"] if is_home else -merged["ats_margin"],
                "team_score_margin": merged["result"] if is_home else -merged["result"],
                "spread_line": merged["spread_line"],
                "team_spread": merged["spread_line"] if is_home else -merged["spread_line"],
                "own_rest": merged[cols["own_rest"]],
                "opp_rest": merged[cols["opp_rest"]],
                "own_qb_name": merged[cols["own_qb_name"]],
                "div_game": merged["div_game"],
                "neutral_site": merged["neutral_site"],
                "weekday": merged["weekday"],
                "gametime_hour": merged["gametime_hour"],
                "temp": merged["temp"],
                "roof": merged["roof"],
            }
        )
        sides.append(side)

    long_df = pd.concat(sides, ignore_index=True)
    long_df = long_df.loc[long_df["team_covered"].notna()].copy()
    long_df["team"] = _canonical_team(long_df["team"])
    long_df["opponent"] = _canonical_team(long_df["opponent"])
    long_df["team_is_favorite"] = long_df["team_spread"] > 0
    long_df["week_block"] = long_df["season"] * 100 + long_df["week"]
    return long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)


def _bias_battery_qb_backup_flag(group: pd.DataFrame) -> pd.Series:
    """Per (team, season) group, sorted by gameday.

    Ported from ``nfl_bias_battery_screen._qb_backup_flag``.
    """

    counts: Counter[str] = Counter()
    flags: list[float] = []
    for qb_name in group["own_qb_name"]:
        total_prior = sum(counts.values())
        if total_prior >= 3:
            modal_qb = counts.most_common(1)[0][0]
            flags.append(1.0 if qb_name != modal_qb else 0.0)
        else:
            flags.append(np.nan)
        if isinstance(qb_name, str) and qb_name:
            counts[qb_name] += 1
    return pd.Series(flags, index=group.index)


def _bias_battery_add_history_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Every within-season, strictly-prior-game derived column the battery needs --
    ``nfl_bias_battery_screen.add_history_features``, ported verbatim.
    """

    long_df = long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)
    grouped = long_df.groupby(["team", "season"], sort=False)

    win = (long_df["team_score_margin"] > 0).astype(float)
    prior_games = grouped.cumcount()
    long_df["_win"] = win
    cum_wins_incl = long_df.groupby(["team", "season"], sort=False)["_win"].cumsum()
    prior_wins = cum_wins_incl - long_df["_win"]
    long_df["prior_games"] = prior_games.to_numpy()
    long_df["prior_win_pct"] = np.where(
        long_df["prior_games"] > 0, prior_wins / long_df["prior_games"], np.nan
    )
    long_df = long_df.drop(columns=["_win"])

    opp_stats = long_df[["team", "season", "week", "prior_win_pct", "prior_games"]].rename(
        columns={
            "team": "opponent",
            "prior_win_pct": "opp_prior_win_pct",
            "prior_games": "opp_prior_games",
        }
    )
    long_df = long_df.merge(opp_stats, on=["opponent", "season", "week"], how="left")

    grouped = long_df.groupby(["team", "season"], sort=False)
    long_df["prior_score_margin"] = grouped["team_score_margin"].shift(1)

    long_df["backup_qb_flag"] = np.nan
    for _, group in long_df.groupby(["team", "season"], sort=False):
        long_df.loc[group.index, "backup_qb_flag"] = _bias_battery_qb_backup_flag(group).to_numpy()

    long_df["is_true_road"] = (~long_df["is_home"]) & (long_df["neutral_site"] == 0)
    grouped = long_df.groupby(["team", "season"], sort=False)
    prev1 = grouped["is_true_road"].shift(1).fillna(False)
    prev2 = grouped["is_true_road"].shift(2).fillna(False)
    long_df["three_plus_road_flag"] = long_df["is_true_road"] & prev1 & prev2

    grouped = long_df.groupby(["team", "season"], sort=False)
    prior_div = grouped["div_game"].shift(1)
    next_div = grouped["div_game"].shift(-1)
    long_df["sandwich_flag"] = (long_df["div_game"] == 0) & (prior_div == 1) & (next_div == 1)

    long_df = long_df.sort_values(["team", "opponent", "season", "gameday"]).reset_index(drop=True)
    grouped = long_df.groupby(["team", "opponent", "season"], sort=False)
    meeting_rank = grouped.cumcount()
    first_margin = grouped["team_score_margin"].transform("first")
    long_df["revenge_flag"] = (meeting_rank >= 1) & (first_margin < 0)

    return long_df.sort_values(["team", "season", "gameday"]).reset_index(drop=True)


def _bias_battery_team_game_table(features: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    """The full bias-battery long table, schedules-merged and history-featured."""

    merged = _bias_battery_merged_features(features, repo_root)
    long_df = _bias_battery_build_long_table(merged)
    return _bias_battery_add_history_features(long_df)


def _bias_battery_construct(
    table: pd.DataFrame,
    flag: pd.Series,
    *,
    sign: int,
    eligible: pd.Series | None = None,
    trait_note: str,
) -> SubsetBiasConstruct:
    return SubsetBiasConstruct(
        table=table,
        flag=flag,
        eligible=eligible,
        sign=sign,
        reliability=None,
        reliability_pairs=None,
        reliability_note=trait_note,
    )


def _flag_division_revenge_game(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params
    table = _bias_battery_team_game_table(features, repo_root)
    return _bias_battery_construct(
        table,
        table["revenge_flag"],
        sign=1,
        trait_note=(
            "division_revenge_game is a per-game situational condition (this season's own "
            "head-to-head history), not a persistent per-team trait -- there is nothing to "
            "split-half."
        ),
    )


def _flag_extra_rest_edge(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params
    table = _bias_battery_team_game_table(features, repo_root)
    flag = (table["own_rest"] - table["opp_rest"]) >= 4
    return _bias_battery_construct(
        table,
        flag,
        sign=1,
        trait_note=(
            "extra_rest_edge is a per-game situational condition (this game's own rest "
            "differential), not a persistent per-team trait -- there is nothing to split-half."
        ),
    )


def _flag_short_week(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params
    table = _bias_battery_team_game_table(features, repo_root)
    flag = table["own_rest"] <= 5
    return _bias_battery_construct(
        table,
        flag,
        sign=-1,
        trait_note=(
            "short_week is a per-game situational condition (this game's own rest), not a "
            "persistent per-team trait -- there is nothing to split-half."
        ),
    )


def _flag_west_coast_early_kickoff(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params
    table = _bias_battery_team_game_table(features, repo_root)
    flag = (
        (~table["is_home"])
        & table["team"].isin(PT_TEAMS)
        & (~table["opponent"].isin(PT_TEAMS))
        & (table["neutral_site"] == 0)
        & (table["gametime_hour"] < 14)
    )
    return _bias_battery_construct(
        table,
        flag,
        sign=-1,
        trait_note=(
            "west_coast_early_kickoff is a per-game situational condition (this game's own "
            "travel/kickoff time), not a persistent per-team trait -- there is nothing to "
            "split-half."
        ),
    )


def _flag_sandwich_spot(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params
    table = _bias_battery_team_game_table(features, repo_root)
    return _bias_battery_construct(
        table,
        table["sandwich_flag"],
        sign=-1,
        trait_note=(
            "sandwich_spot is a per-game situational condition (this week's own division-game "
            "schedule), not a persistent per-team trait -- there is nothing to split-half."
        ),
    )


def _flag_backup_qb_start(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params
    table = _bias_battery_team_game_table(features, repo_root)
    flag = table["backup_qb_flag"] == 1.0
    eligible = table["backup_qb_flag"].notna()
    return _bias_battery_construct(
        table,
        flag,
        sign=1,
        eligible=eligible,
        trait_note=(
            "backup_qb_start is a per-game situational condition (this game's own starter vs. "
            "this team-season's modal QB), not a persistent per-team trait -- there is nothing "
            "to split-half. Rows with fewer than 3 prior starts this season (no modal-QB "
            "baseline) are excluded from both arms via `eligible`."
        ),
    )


def _flag_motivation_mismatch(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params
    table = _bias_battery_team_game_table(features, repo_root)
    flag = (
        (table["prior_win_pct"] >= 0.400)
        & table["week"].between(11, 18)
        & (table["opp_prior_games"] >= 9)
        & (table["opp_prior_win_pct"] <= 0.300)
    )
    return _bias_battery_construct(
        table,
        flag,
        sign=1,
        trait_note=(
            "motivation_mismatch is a per-game situational condition (this week's own record vs. "
            "the opponent's), not a persistent per-team trait -- there is nothing to split-half."
        ),
    )


# ---------------------------------------------------------------------------
# Referee-battery builders: officiating-crew effects on ATS cover rates
# ---------------------------------------------------------------------------
#
# New signal family (2026-08-19), predeclared in docs/referee_battery.md
# BEFORE any cover-rate sign was looked at. Every flag here is pregame-safe:
# crew assignments (who is head referee this game) are public before kickoff,
# and every trait used to bucket a referee is the referee's own PRIOR-season
# officiating history -- never this game's own penalties (see the leakage
# test in tests/test_experiment_runner.py). Data source:
# nflreadpy.load_officials() (nflverse-data officials/officials release,
# 2015-2025) fetched into data/raw/officials/<snapshot>/officials.parquet,
# plus a derived per-game penalty-by-team aggregate
# (data/raw/officials/<snapshot>/game_penalties.parquet, built from nflverse
# PBP's own penalty/penalty_team columns -- NOT the repo's existing trimmed
# local PBP snapshot, whose stored column list omits penalty_team). See
# docs/referee_battery.md for the full predeclaration, data-coverage caveats,
# and mechanisms.
#
# officials.parquet's own game_id is the LEGACY numeric GSIS format, not
# game_features.parquet's game_id (e.g. "2015_01_PIT_NE"); the crosswalk is
# the newest data/raw/*/schedules.parquet snapshot's own old_game_id column
# (the same crosswalk source _latest_schedules_snapshot already serves to
# the bias-battery builders above).

_REFEREE_POSITION = "Referee"
_REFEREE_SEASON_TYPE = "REG"
_DEFAULT_VETERAN_THRESHOLD_SEASONS = 5


def _latest_officials_snapshot(repo_root: Path) -> tuple[Path, Path, str]:
    candidates = sorted((repo_root / "data" / "raw" / "officials").glob("*/officials.parquet"))
    if not candidates:
        raise ExperimentRunnerError(
            f"No data/raw/officials/*/officials.parquet snapshot found under {repo_root}. Fetch "
            "nflverse officials data (nflreadpy.load_officials()) plus a derived "
            "game_penalties.parquet before running a referee_battery spec -- see "
            "docs/referee_battery.md."
        )
    officials_path = candidates[-1]
    game_penalties_path = officials_path.with_name("game_penalties.parquet")
    if not game_penalties_path.is_file():
        raise ExperimentRunnerError(
            f"{game_penalties_path} is missing (expected alongside {officials_path})"
        )
    return officials_path, game_penalties_path, officials_path.parent.name


@dataclass(frozen=True)
class _RefereeTraitData:
    """Per-game_id referee trait table plus the two traits' own split-half reliability.

    ``game_trait`` has one row per (standard-format) game_id that has a
    matched REG-season head-referee assignment: ``official_name``, ``season``,
    ``lag_penalty_rate_quartile``/``lag_home_away_diff_quartile`` (1-4, NaN
    for an official's first dataset-visible season -- no valid year-over-year
    lag), and ``prior_seasons_experience`` (count of distinct PRIOR seasons
    that official appears as ``Referee`` in this dataset; always present).
    """

    game_trait: pd.DataFrame
    penalty_rate_reliability: float | None
    penalty_rate_reliability_pairs: int
    home_away_diff_reliability: float | None
    home_away_diff_reliability_pairs: int
    n_officials: int
    snapshot_id: str


def _referee_year_over_year_reliability(
    name_season: pd.DataFrame, value_col: str
) -> tuple[float | None, int]:
    ordered = name_season.sort_values(["official_name", "season"]).copy()
    ordered["_next_value"] = ordered.groupby("official_name")[value_col].shift(-1)
    ordered["_next_season"] = ordered.groupby("official_name")["season"].shift(-1)
    pairs = ordered.loc[ordered["_next_season"] == ordered["season"] + 1]
    if pairs.empty:
        return None, 0
    correlation = pairs[value_col].corr(pairs["_next_value"])
    return (None if pd.isna(correlation) else float(correlation)), len(pairs)


def _build_referee_trait_data(repo_root: Path) -> _RefereeTraitData:
    officials_path, game_penalties_path, snapshot_id = _latest_officials_snapshot(repo_root)
    officials = pd.read_parquet(officials_path)
    game_penalties = pd.read_parquet(game_penalties_path)

    required_off = {"game_id", "official_name", "position", "season", "season_type"}
    missing_off = sorted(required_off.difference(officials.columns))
    if missing_off:
        raise ExperimentRunnerError(
            f"{officials_path} is missing columns: {', '.join(missing_off)}"
        )
    required_gp = {"game_id", "penalties_total", "penalties_on_home", "penalties_on_away"}
    missing_gp = sorted(required_gp.difference(game_penalties.columns))
    if missing_gp:
        raise ExperimentRunnerError(
            f"{game_penalties_path} is missing columns: {', '.join(missing_gp)}"
        )

    refs = officials.loc[
        (officials["position"] == _REFEREE_POSITION)
        & (officials["season_type"] == _REFEREE_SEASON_TYPE)
    ].copy()

    schedules_path = _latest_schedules_snapshot(repo_root)
    schedules = pd.read_parquet(schedules_path).loc[:, ["game_id", "old_game_id"]]
    # The LEFT frame's overlapping "game_id" (legacy numeric) is suffixed
    # "_legacy"; the RIGHT frame's (schedules' own, standard-format) keeps
    # the bare "game_id" name -- so after this merge "game_id" IS the
    # game_features-shaped id, matching every other builder's join key.
    refs = refs.merge(
        schedules, left_on="game_id", right_on="old_game_id", how="inner", suffixes=("_legacy", "")
    )
    refs = refs.loc[:, ["game_id", "official_name", "season"]]

    merged_games = refs.merge(game_penalties, on="game_id", how="inner", suffixes=("", "_gp"))
    merged_games["_diff"] = merged_games["penalties_on_away"] - merged_games["penalties_on_home"]

    name_season = (
        merged_games.groupby(["official_name", "season"])
        .agg(mean_total=("penalties_total", "mean"), mean_diff=("_diff", "mean"))
        .reset_index()
    )

    penalty_rate_reliability, penalty_rate_pairs = _referee_year_over_year_reliability(
        name_season, "mean_total"
    )
    home_away_diff_reliability, home_away_diff_pairs = _referee_year_over_year_reliability(
        name_season, "mean_diff"
    )

    lag = name_season.sort_values(["official_name", "season"]).copy()
    lag["prev_total"] = lag.groupby("official_name")["mean_total"].shift(1)
    lag["prev_diff"] = lag.groupby("official_name")["mean_diff"].shift(1)
    lag["prev_season"] = lag.groupby("official_name")["season"].shift(1)
    lagged = lag.loc[lag["season"] - lag["prev_season"] == 1].copy()
    lagged["lag_penalty_rate_quartile"] = pd.qcut(
        lagged["prev_total"], 4, labels=[1, 2, 3, 4]
    ).astype(int)
    lagged["lag_home_away_diff_quartile"] = pd.qcut(
        lagged["prev_diff"], 4, labels=[1, 2, 3, 4]
    ).astype(int)

    experience = name_season[["official_name", "season"]].drop_duplicates()
    experience = experience.sort_values(["official_name", "season"]).copy()
    experience["prior_seasons_experience"] = experience.groupby("official_name").cumcount()

    game_trait = (
        merged_games[["game_id", "official_name", "season"]]
        .merge(
            lagged[
                [
                    "official_name",
                    "season",
                    "lag_penalty_rate_quartile",
                    "lag_home_away_diff_quartile",
                ]
            ],
            on=["official_name", "season"],
            how="left",
        )
        .merge(
            experience[["official_name", "season", "prior_seasons_experience"]],
            on=["official_name", "season"],
            how="left",
        )
    )

    return _RefereeTraitData(
        game_trait=game_trait,
        penalty_rate_reliability=penalty_rate_reliability,
        penalty_rate_reliability_pairs=penalty_rate_pairs,
        home_away_diff_reliability=home_away_diff_reliability,
        home_away_diff_reliability_pairs=home_away_diff_pairs,
        n_officials=int(name_season["official_name"].nunique()),
        snapshot_id=snapshot_id,
    )


def _referee_team_game_table(
    features: pd.DataFrame, repo_root: Path
) -> tuple[pd.DataFrame, _RefereeTraitData]:
    table = _base_team_game_table(features)
    trait_data = _build_referee_trait_data(repo_root)
    # table already carries its own "season" (from _base_team_game_table); keep only
    # the trait columns the flag builders need, or "season"/"official_name" would
    # collide and get silently suffixed (season_x/season_y), breaking the runner's
    # own `construct.table["season"]` population filter downstream.
    trait_columns = trait_data.game_trait.loc[
        :,
        [
            "game_id",
            "lag_penalty_rate_quartile",
            "lag_home_away_diff_quartile",
            "prior_seasons_experience",
        ],
    ]
    merged = table.merge(trait_columns, on="game_id", how="inner")
    return merged, trait_data


def _referee_penalty_rate_population_note(
    trait_data: _RefereeTraitData, *, trait_label: str
) -> str:
    return (
        f"Officials snapshot {trait_data.snapshot_id}: head referee (position='Referee', "
        "season_type='REG') joined to game_features via the newest schedules snapshot's "
        f"old_game_id crosswalk, {trait_data.n_officials} distinct referees. Flag uses the home "
        f"team's referee's PRIOR-season {trait_label}, global qcut(4) over every (official, "
        "season) pair with a valid year-over-year lag."
    )


def _flag_referee_penalty_rate_top_quartile(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params
    merged, trait_data = _referee_team_game_table(features, repo_root)
    merged = merged.loc[merged["lag_penalty_rate_quartile"].notna()].copy()
    merged["lag_penalty_rate_quartile"] = merged["lag_penalty_rate_quartile"].astype(int)
    flag = merged["is_home"] & (merged["lag_penalty_rate_quartile"] == 4)
    return SubsetBiasConstruct(
        table=merged,
        flag=flag,
        eligible=None,
        sign=1,
        reliability=trait_data.penalty_rate_reliability,
        reliability_pairs=trait_data.penalty_rate_reliability_pairs,
        reliability_note=(
            "Year-over-year Pearson correlation of referee-season mean total penalties/game "
            f"(both teams combined), {trait_data.penalty_rate_reliability_pairs} referee-season "
            f"pairs, {trait_data.n_officials} distinct referees, officials snapshot "
            f"{trait_data.snapshot_id}."
        ),
        population_note=_referee_penalty_rate_population_note(
            trait_data, trait_label="mean total penalties/game (top quartile)"
        ),
    )


def _flag_referee_penalty_rate_bottom_quartile(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params
    merged, trait_data = _referee_team_game_table(features, repo_root)
    merged = merged.loc[merged["lag_penalty_rate_quartile"].notna()].copy()
    merged["lag_penalty_rate_quartile"] = merged["lag_penalty_rate_quartile"].astype(int)
    flag = merged["is_home"] & (merged["lag_penalty_rate_quartile"] == 1)
    return SubsetBiasConstruct(
        table=merged,
        flag=flag,
        eligible=None,
        sign=-1,
        reliability=trait_data.penalty_rate_reliability,
        reliability_pairs=trait_data.penalty_rate_reliability_pairs,
        reliability_note=(
            "Year-over-year Pearson correlation of referee-season mean total penalties/game "
            f"(both teams combined), {trait_data.penalty_rate_reliability_pairs} referee-season "
            f"pairs, {trait_data.n_officials} distinct referees, officials snapshot "
            f"{trait_data.snapshot_id}."
        ),
        population_note=_referee_penalty_rate_population_note(
            trait_data, trait_label="mean total penalties/game (bottom quartile)"
        ),
    )


def _flag_referee_home_penalty_tilt_top_quartile(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params
    merged, trait_data = _referee_team_game_table(features, repo_root)
    merged = merged.loc[merged["lag_home_away_diff_quartile"].notna()].copy()
    merged["lag_home_away_diff_quartile"] = merged["lag_home_away_diff_quartile"].astype(int)
    flag = merged["is_home"] & (merged["lag_home_away_diff_quartile"] == 4)
    return SubsetBiasConstruct(
        table=merged,
        flag=flag,
        eligible=None,
        sign=1,
        reliability=trait_data.home_away_diff_reliability,
        reliability_pairs=trait_data.home_away_diff_reliability_pairs,
        reliability_note=(
            "Year-over-year Pearson correlation of referee-season mean(penalties_on_away) - "
            f"mean(penalties_on_home), {trait_data.home_away_diff_reliability_pairs} "
            f"referee-season pairs, {trait_data.n_officials} distinct referees, officials "
            f"snapshot {trait_data.snapshot_id}. MEASURED near zero in this window -- report "
            "plainly, not a reason to skip recording."
        ),
        population_note=_referee_penalty_rate_population_note(
            trait_data, trait_label="mean(away penalties) - mean(home penalties) (top quartile)"
        ),
    )


def _flag_referee_home_penalty_tilt_bottom_quartile(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params
    merged, trait_data = _referee_team_game_table(features, repo_root)
    merged = merged.loc[merged["lag_home_away_diff_quartile"].notna()].copy()
    merged["lag_home_away_diff_quartile"] = merged["lag_home_away_diff_quartile"].astype(int)
    flag = merged["is_home"] & (merged["lag_home_away_diff_quartile"] == 1)
    return SubsetBiasConstruct(
        table=merged,
        flag=flag,
        eligible=None,
        sign=-1,
        reliability=trait_data.home_away_diff_reliability,
        reliability_pairs=trait_data.home_away_diff_reliability_pairs,
        reliability_note=(
            "Year-over-year Pearson correlation of referee-season mean(penalties_on_away) - "
            f"mean(penalties_on_home), {trait_data.home_away_diff_reliability_pairs} "
            f"referee-season pairs, {trait_data.n_officials} distinct referees, officials "
            f"snapshot {trait_data.snapshot_id}. MEASURED near zero in this window -- report "
            "plainly, not a reason to skip recording."
        ),
        population_note=_referee_penalty_rate_population_note(
            trait_data, trait_label="mean(away penalties) - mean(home penalties) (bottom quartile)"
        ),
    )


def _flag_referee_veteran_home_cover(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons
    threshold = int(params.get("veteran_threshold", _DEFAULT_VETERAN_THRESHOLD_SEASONS))
    merged, trait_data = _referee_team_game_table(features, repo_root)
    flag = merged["is_home"] & (merged["prior_seasons_experience"] >= threshold)
    return SubsetBiasConstruct(
        table=merged,
        flag=flag,
        eligible=None,
        sign=-1,
        reliability=None,
        reliability_pairs=None,
        reliability_note=(
            "prior_seasons_experience is a monotonically increasing career-stage counter, not a "
            "trait whose value could repeat or correlate year-over-year in the split-half sense "
            "-- there is nothing to split-half (analogous to backup_qb_start's per-game "
            "career-stage condition)."
        ),
        population_note=(
            f"Officials snapshot {trait_data.snapshot_id}, {trait_data.n_officials} distinct "
            f"referees. Flag = home team's referee has >= {threshold} distinct PRIOR "
            "dataset-visible seasons as head referee. See docs/referee_battery.md for the "
            "2015 left-censoring caveat (population.seasons should exclude 2015)."
        ),
    )


def _flag_referee_rookie_home_cover(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params
    merged, trait_data = _referee_team_game_table(features, repo_root)
    flag = merged["is_home"] & (merged["prior_seasons_experience"] == 0)
    return SubsetBiasConstruct(
        table=merged,
        flag=flag,
        eligible=None,
        sign=1,
        reliability=None,
        reliability_pairs=None,
        reliability_note=(
            "prior_seasons_experience is a monotonically increasing career-stage counter, not a "
            "trait whose value could repeat or correlate year-over-year in the split-half sense "
            "-- there is nothing to split-half (analogous to backup_qb_start's per-game "
            "career-stage condition)."
        ),
        population_note=(
            f"Officials snapshot {trait_data.snapshot_id}, {trait_data.n_officials} distinct "
            "referees. Flag = home team's referee has 0 distinct PRIOR dataset-visible seasons "
            "as head referee. See docs/referee_battery.md for the 2015 left-censoring caveat "
            "(population.seasons should exclude 2015 -- ALL referees show 0 prior seasons that "
            "year purely from data coverage, not genuine debuts)."
        ),
    )


# ---------------------------------------------------------------------------
# Penalty-TYPE crew tendencies: widens the referee battery above from total
# penalty counts to type-specific rates (docs/data_source_scout_v4.md lead
# #1, "Penalty-type crew tendencies", predeclared docs/penalty_crew_tendencies.md).
# ---------------------------------------------------------------------------
#
# New signal family (2026-08-20 session). Data source:
# data/raw/officials/<timestamp>/game_penalty_types.parquet, built by
# scripts/fetch_penalty_type_snapshot.py -- a fresh re-pull of
# nflreadpy.load_pbp() (same nflverse pipeline the repo already ingests) that
# retains penalty_type/penalty_team (present upstream, absent from
# nfl_ats.pbp.PBP_SNAPSHOT_COLUMNS and data/pbp/team_style/raw_pbp_narrow.parquet
# alike), aggregated to one row per (game_id, penalty_type) with
# penalties_total/penalties_on_home/penalties_on_away -- same shape and same
# home/away attribution convention (penalty_team == home_team/away_team) as
# the existing game_penalties.parquet, MEASURED-verified to reproduce its
# per-game totals exactly (0 count mismatches, 0 games only in either table,
# all 11 seasons' game counts matched) after summing type counts back up.
#
# Every cell below is pregame-safe for the SAME reason the existing referee
# battery is: crew assignment is public before kickoff, and every trait is
# the referee's PRIOR-season history, never this game's own penalties (the
# existing leakage test's mutation pattern applies identically here since
# these builders reuse the same shift(1)-over-(official, season) lag
# construction). The four cells interact a referee-crew trait with a SECOND,
# already-pregame-safe condition (opener line, prior-rolling team pass rate,
# game total line) -- each implemented as a boolean AND of two top/bottom
# quartile flags, matching this module's existing quartile-cut convention
# throughout (no continuous interaction terms; `subset_bias` is a boolean-flag
# framework project-wide, see docs/experiment_pipeline.md).

_DPI_PENALTY_TYPE = "Defensive Pass Interference"
_HOLDING_PENALTY_TYPE = "Offensive Holding"
_HEAVY_UNDERDOG_THRESHOLD_DEFAULT = 7.0


def _latest_penalty_type_snapshot(repo_root: Path) -> tuple[Path, str]:
    candidates = sorted(
        (repo_root / "data" / "raw" / "officials").glob("*/game_penalty_types.parquet")
    )
    if not candidates:
        raise ExperimentRunnerError(
            f"No data/raw/officials/*/game_penalty_types.parquet snapshot found under "
            f"{repo_root}. Run scripts/fetch_penalty_type_snapshot.py before running a "
            "penalty-type crew-tendency spec -- see docs/penalty_crew_tendencies.md."
        )
    path = candidates[-1]
    return path, path.parent.name


@dataclass(frozen=True)
class _RefereeTypeTraitData:
    """Per-game_id lagged quartile of one referee's PRIOR-season rate of ONE penalty type.

    Same shape and construction as ``_RefereeTraitData``'s ``mean_total``
    trait, restricted to a single ``penalty_type`` value. A game with zero
    penalties of this type is a genuine zero observation (not a missing one)
    -- it is simply absent from the long ``game_penalty_types`` table for
    that type, so the per-referee-season mean is computed over EVERY game
    that referee worked, with absent games filled to 0.0, never dropped.
    """

    game_trait: pd.DataFrame
    reliability: float | None
    reliability_pairs: int
    n_officials: int
    officials_snapshot_id: str
    penalty_type_snapshot_id: str
    penalty_type: str


def _build_referee_type_trait_data(repo_root: Path, penalty_type: str) -> _RefereeTypeTraitData:
    officials_path, _game_penalties_path, officials_snapshot_id = _latest_officials_snapshot(
        repo_root
    )
    officials = pd.read_parquet(officials_path)
    refs = officials.loc[
        (officials["position"] == _REFEREE_POSITION)
        & (officials["season_type"] == _REFEREE_SEASON_TYPE)
    ].copy()

    schedules_path = _latest_schedules_snapshot(repo_root)
    schedules = pd.read_parquet(schedules_path).loc[:, ["game_id", "old_game_id"]]
    refs = refs.merge(
        schedules, left_on="game_id", right_on="old_game_id", how="inner", suffixes=("_legacy", "")
    )
    refs = refs.loc[:, ["game_id", "official_name", "season"]]

    penalty_type_path, penalty_type_snapshot_id = _latest_penalty_type_snapshot(repo_root)
    game_penalty_types = pd.read_parquet(penalty_type_path)
    required = {"game_id", "penalty_type", "penalties_total"}
    missing = sorted(required.difference(game_penalty_types.columns))
    if missing:
        raise ExperimentRunnerError(f"{penalty_type_path} is missing columns: {', '.join(missing)}")

    type_counts = game_penalty_types.loc[
        game_penalty_types["penalty_type"] == penalty_type, ["game_id", "penalties_total"]
    ]
    merged_games = refs.merge(type_counts, on="game_id", how="left")
    merged_games["penalties_total"] = merged_games["penalties_total"].fillna(0.0)

    name_season = (
        merged_games.groupby(["official_name", "season"])
        .agg(mean_total=("penalties_total", "mean"))
        .reset_index()
    )
    reliability, reliability_pairs = _referee_year_over_year_reliability(name_season, "mean_total")

    lag = name_season.sort_values(["official_name", "season"]).copy()
    lag["prev_total"] = lag.groupby("official_name")["mean_total"].shift(1)
    lag["prev_season"] = lag.groupby("official_name")["season"].shift(1)
    lagged = lag.loc[lag["season"] - lag["prev_season"] == 1].copy()
    lagged["lag_type_quartile"] = pd.qcut(lagged["prev_total"], 4, labels=[1, 2, 3, 4]).astype(int)

    game_trait = merged_games[["game_id", "official_name", "season"]].merge(
        lagged[["official_name", "season", "lag_type_quartile"]],
        on=["official_name", "season"],
        how="left",
    )

    return _RefereeTypeTraitData(
        game_trait=game_trait,
        reliability=reliability,
        reliability_pairs=reliability_pairs,
        n_officials=int(name_season["official_name"].nunique()),
        officials_snapshot_id=officials_snapshot_id,
        penalty_type_snapshot_id=penalty_type_snapshot_id,
        penalty_type=penalty_type,
    )


def _merge_home_pass_rate_quartile(table: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    """Attach a GAME-level (not duplicated-row-level) quartile of the home team's
    prior-rolling pregame-safe pass rate (``enrich_with_pbp_features``'s
    ``home_pbp_off_pass_rate``, an EWMA of games strictly before the one being
    scored -- see ``nfl_ats.pbp.enrich_with_pbp_features``'s own docstring).
    Quartile boundaries are computed once over the deduplicated per-game
    population, then merged onto every row of ``table`` (both team-game
    sides carry the same HOME-team value; only ``is_home`` rows are ever
    flagged by a caller of this helper).
    """

    path = repo_root / "data" / "processed" / "game_features_pbp.parquet"
    if not path.is_file():
        raise ExperimentRunnerError(f"{path} not found")
    pbp_features = pd.read_parquet(path).loc[:, ["game_id", "home_pbp_off_pass_rate"]]
    pbp_features = pbp_features.loc[pbp_features["home_pbp_off_pass_rate"].notna()]
    pbp_features = pbp_features.drop_duplicates("game_id").copy()
    pbp_features["home_pass_rate_quartile"] = pd.qcut(
        pbp_features["home_pbp_off_pass_rate"], 4, labels=[1, 2, 3, 4]
    ).astype(int)
    return table.merge(
        pbp_features[["game_id", "home_pass_rate_quartile"]], on="game_id", how="inner"
    )


def _merge_total_line_quartile(table: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Attach a GAME-level quartile of the game's own (pregame) total (over/under) line."""

    total_line = features.loc[:, ["game_id", "total_line"]].copy()
    total_line["total_line"] = pd.to_numeric(total_line["total_line"], errors="coerce")
    total_line = total_line.loc[total_line["total_line"].notna()].drop_duplicates("game_id").copy()
    total_line["total_line_quartile"] = pd.qcut(
        total_line["total_line"], 4, labels=[1, 2, 3, 4]
    ).astype(int)
    return table.merge(total_line[["game_id", "total_line_quartile"]], on="game_id", how="inner")


def _flag_referee_high_flag_heavy_underdog(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    """Cell A: high-total-flag crew (existing mean_total trait) AND home team a heavy underdog.

    Reuses the ALREADY-MEASURED mean_total trait (referee_battery.md cells
    1/2, split-half +0.370) -- no new trait, only a narrower population.
    Mechanism: cell 1's hypothesized road-team communication/tempo
    disruption from extra stoppages is hypothesized to matter MOST when the
    home team is already a big underdog and most needs the extra time
    stoppages buy to control tempo/limit possessions against a stronger
    opponent, so the home-cover edge should concentrate in this subset.
    Sign: +1. Designed to run at ``population.grade="opener"`` per AGENTS.md's
    binding "grade the decision at the opener" rule.
    """

    del seasons
    threshold = float(params.get("underdog_threshold", _HEAVY_UNDERDOG_THRESHOLD_DEFAULT))
    merged, trait_data = _referee_team_game_table(features, repo_root)
    merged = merged.loc[merged["lag_penalty_rate_quartile"].notna()].copy()
    merged["lag_penalty_rate_quartile"] = merged["lag_penalty_rate_quartile"].astype(int)
    high_flag_crew = merged["lag_penalty_rate_quartile"] == 4
    heavy_underdog = merged["spread_line"] <= -threshold
    flag = merged["is_home"] & high_flag_crew & heavy_underdog
    return SubsetBiasConstruct(
        table=merged,
        flag=flag,
        eligible=None,
        sign=1,
        reliability=trait_data.penalty_rate_reliability,
        reliability_pairs=trait_data.penalty_rate_reliability_pairs,
        reliability_note=(
            "Reuses referee_battery.md cell 1/2's existing mean_total trait unchanged -- "
            f"{trait_data.penalty_rate_reliability_pairs} referee-season pairs, "
            f"{trait_data.n_officials} distinct referees, officials snapshot "
            f"{trait_data.snapshot_id}. This cell introduces no new trait, only a narrower "
            "population (top-quartile crew AND home team a heavy underdog)."
        ),
        population_note=(
            _referee_penalty_rate_population_note(
                trait_data, trait_label="mean total penalties/game (top quartile)"
            )
            + f" Additionally restricted to home team getting >= {threshold} points (heavy "
            "underdog, spread_line convention: negative = home not favored). Predeclared "
            "docs/penalty_crew_tendencies.md cell A."
        ),
    )


def _flag_referee_dpi_tilt_pass_heavy_favorite(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    """Cell B: crew's PRIOR-season Defensive Pass Interference rate top quartile AND
    home team is BOTH the favorite AND in the top quartile of prior-rolling pass rate.

    Mechanism: a crew that calls DPI at a high rate is hypothesized to
    disproportionately extend a pass-heavy offense's drives (more DPI flags
    against the defense = more automatic first downs/yardage for the
    offense); a pass-heavy favorite facing such a crew is hypothesized to
    cover MORE. Sign: +1.
    """

    del seasons, params
    type_trait = _build_referee_type_trait_data(repo_root, _DPI_PENALTY_TYPE)
    base = _base_team_game_table(features)
    merged = base.merge(
        type_trait.game_trait.loc[:, ["game_id", "lag_type_quartile"]], on="game_id", how="inner"
    )
    merged = _merge_home_pass_rate_quartile(merged, repo_root)
    merged = merged.loc[merged["lag_type_quartile"].notna()].copy()
    merged["lag_type_quartile"] = merged["lag_type_quartile"].astype(int)
    dpi_tilt_top = merged["lag_type_quartile"] == 4
    home_favorite = merged["is_home"] & (merged["team_spread"] > 0.0)
    pass_heavy = merged["home_pass_rate_quartile"] == 4
    flag = home_favorite & pass_heavy & dpi_tilt_top
    return SubsetBiasConstruct(
        table=merged,
        flag=flag,
        eligible=None,
        sign=1,
        reliability=type_trait.reliability,
        reliability_pairs=type_trait.reliability_pairs,
        reliability_note=(
            f"Year-over-year Pearson correlation of referee-season mean {_DPI_PENALTY_TYPE} "
            f"calls/game, {type_trait.reliability_pairs} referee-season pairs, "
            f"{type_trait.n_officials} distinct referees, officials snapshot "
            f"{type_trait.officials_snapshot_id}, penalty-type snapshot "
            f"{type_trait.penalty_type_snapshot_id}. MEASURED near zero in this window ("
            "docs/penalty_crew_tendencies.md reliability table) -- report plainly, not a "
            "reason to skip recording, mirroring referee_battery.md cells 5/6's treatment of "
            "mean_diff's own near-zero reliability."
        ),
        population_note=(
            f"Officials snapshot {type_trait.officials_snapshot_id}, penalty-type snapshot "
            f"{type_trait.penalty_type_snapshot_id}, {type_trait.n_officials} distinct "
            "referees. Flag = home team is the favorite (team_spread>0) AND home team's "
            "prior-rolling pregame pass rate (game_features_pbp.parquet's "
            "home_pbp_off_pass_rate) in the top quartile AND the game's referee's PRIOR-season "
            f"{_DPI_PENALTY_TYPE} rate in the top quartile, vs. everyone else. Predeclared "
            "docs/penalty_crew_tendencies.md cell B."
        ),
    )


def _flag_referee_holding_tilt_run_heavy(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    """Cell C: crew's PRIOR-season Offensive Holding rate top quartile AND home team
    in the BOTTOM quartile of prior-rolling pass rate (i.e. run-heavy).

    Mechanism: a crew that calls offensive holding at a high rate is
    hypothesized to disproportionately disrupt a run-heavy team's sustained
    run-blocking schemes (more holding scrutiny on run blocks), hurting
    drive sustain for the home team when it is run-heavy. Sign: -1 (home
    cover DECREASES in this subset).
    """

    del seasons, params
    type_trait = _build_referee_type_trait_data(repo_root, _HOLDING_PENALTY_TYPE)
    base = _base_team_game_table(features)
    merged = base.merge(
        type_trait.game_trait.loc[:, ["game_id", "lag_type_quartile"]], on="game_id", how="inner"
    )
    merged = _merge_home_pass_rate_quartile(merged, repo_root)
    merged = merged.loc[merged["lag_type_quartile"].notna()].copy()
    merged["lag_type_quartile"] = merged["lag_type_quartile"].astype(int)
    holding_tilt_top = merged["lag_type_quartile"] == 4
    run_heavy = merged["home_pass_rate_quartile"] == 1
    flag = merged["is_home"] & run_heavy & holding_tilt_top
    return SubsetBiasConstruct(
        table=merged,
        flag=flag,
        eligible=None,
        sign=-1,
        reliability=type_trait.reliability,
        reliability_pairs=type_trait.reliability_pairs,
        reliability_note=(
            f"Year-over-year Pearson correlation of referee-season mean {_HOLDING_PENALTY_TYPE} "
            f"calls/game, {type_trait.reliability_pairs} referee-season pairs, "
            f"{type_trait.n_officials} distinct referees, officials snapshot "
            f"{type_trait.officials_snapshot_id}, penalty-type snapshot "
            f"{type_trait.penalty_type_snapshot_id}. MEASURED a real, moderate persistent "
            "trait in this window, similar magnitude to mean_total's own +0.370 (see "
            "docs/penalty_crew_tendencies.md reliability table)."
        ),
        population_note=(
            f"Officials snapshot {type_trait.officials_snapshot_id}, penalty-type snapshot "
            f"{type_trait.penalty_type_snapshot_id}, {type_trait.n_officials} distinct "
            "referees. Flag = home team's prior-rolling pregame pass rate "
            "(game_features_pbp.parquet's home_pbp_off_pass_rate) in the BOTTOM quartile "
            "(run-heavy) AND the game's referee's PRIOR-season "
            f"{_HOLDING_PENALTY_TYPE} rate in the top quartile, vs. everyone else. Predeclared "
            "docs/penalty_crew_tendencies.md cell C."
        ),
    )


def _flag_referee_flag_rate_high_total_line(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    """Cell D: crew's overall PRIOR-season flag rate (existing mean_total trait) top
    quartile AND the game's own total (over/under) line in the top quartile.

    Mechanism: a high-flag crew's extra stoppages are hypothesized to matter
    most for tempo/possession control in a high-total (shootout-projected,
    typically pass-heavy/up-tempo) game; the home team, which controls the
    game plan at home, is hypothesized to benefit from that extra structure
    more than the visitor. Sign: +1. Implemented as the boolean AND of two
    top-quartile flags (this module's `subset_bias` framework is
    boolean-flag-based project-wide; a continuous z-score interaction term
    is out of scope, see docs/experiment_pipeline.md).
    """

    del seasons, params
    merged, trait_data = _referee_team_game_table(features, repo_root)
    merged = _merge_total_line_quartile(merged, features)
    merged = merged.loc[merged["lag_penalty_rate_quartile"].notna()].copy()
    merged["lag_penalty_rate_quartile"] = merged["lag_penalty_rate_quartile"].astype(int)
    high_flag_crew = merged["lag_penalty_rate_quartile"] == 4
    high_total = merged["total_line_quartile"] == 4
    flag = merged["is_home"] & high_flag_crew & high_total
    return SubsetBiasConstruct(
        table=merged,
        flag=flag,
        eligible=None,
        sign=1,
        reliability=trait_data.penalty_rate_reliability,
        reliability_pairs=trait_data.penalty_rate_reliability_pairs,
        reliability_note=(
            "Reuses referee_battery.md cell 1/2's existing mean_total trait unchanged -- "
            f"{trait_data.penalty_rate_reliability_pairs} referee-season pairs, "
            f"{trait_data.n_officials} distinct referees, officials snapshot "
            f"{trait_data.snapshot_id}. This cell introduces no new trait, only a narrower "
            "population (top-quartile crew AND top-quartile game total line)."
        ),
        population_note=(
            _referee_penalty_rate_population_note(
                trait_data, trait_label="mean total penalties/game (top quartile)"
            )
            + " Additionally restricted to the game's own total_line in the top quartile. "
            "Predeclared docs/penalty_crew_tendencies.md cell D."
        ),
    )


# ---------------------------------------------------------------------------
# Forecast-weather builders: the 2009-2019 archive backward-extension family
# ---------------------------------------------------------------------------
#
# New signal family (2026-08-20 session, backlog item 3), predeclared in the
# "2026-08-20 extension" section of docs/forecast_weather_screen.md BEFORE any
# effect on the extended window was computed. Data source:
# data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet, a
# FRESH single-cutoff-mode (kickoff_nearest, model=GFS) archive spanning the
# full 2009-2025 project window, built this session by reusing
# scripts/ingest_forecast_archive.py's exact walking/cutoff/station-mapping
# machinery unchanged (only the field EXTRACTION was extended, additively, to
# also capture GFS MOS precipitation probability -- see that script's
# nearest_row_with_field). tuesday_noon (the cutoff the ORIGINAL 4
# forecast_weather_* cells in registry/weak_signals.json were scored on) is
# NOT used here: its MOS model (MEX) is measurably absent from the IEM archive
# before 2020-07-12 (docs/forecast_archive_build.md, reconfirmed this session
# by a live probe), so a tuesday_noon-cutoff archive genuinely cannot extend
# backward to 2009-2019 -- kickoff_nearest can (GFS's IEM archive reaches back
# to at least 2005) and is, per the docs/forecast_archive_build.md
# 2026-08-20 owner correction, the MORE pool-relevant cutoff anyway (picks are
# editable up to each game's real deadline, not frozen at Tuesday noon). This
# means these 6 builders are NOT byte-identical reproductions of their
# `forecast_weather_*` (tuesday_noon) namesakes/siblings -- same mechanism,
# different information-timing AND a different population (all use REG
# 2009-2025 archive coverage vs. the originals' REG 2020-2025) -- so every
# registry name below carries a distinguishing `_kn_` (kickoff_nearest)
# infix, and must not be pooled against its tuesday_noon sibling as
# independent evidence (same overlap-disclosure convention the tuesday_noon
# screen already used against ITS actual-weather siblings).
#
# GAME-level construction, not team-long: `home_cover` is a GAME outcome and
# these flags are GAME-level weather/market conditions with no team-relative
# framing of their own, so this section does NOT reuse `_base_team_game_table`
# (which duplicates every game into a home-side/away-side pair via `is_home`
# gating -- correct for a genuinely team-relative situational condition like
# `home_underdog`, but it changes what's measured for a flag that doesn't need
# that duplication: the complement would silently absorb BOTH the flagged
# game's own away-side row (team_covered=1-home_cover) and every other game's
# both sides, which is not the same quantity as "mean(home_cover) over every
# other game", the comparison scripts/nfl_forecast_weather_screen.py's
# ORIGINAL 4 cells used). `_forecast_weather_game_table` below builds one row
# per REG game instead, with `team_covered` set to `home_cover` directly --
# this keeps these builders numerically faithful to that original screen's
# subset-vs-complement design, just run through the standardized pipeline.

_FORECAST_OUTDOOR_ROOFS = frozenset({"outdoors", "open"})
_FORECAST_DOME_CLOSED_ROOFS = frozenset({"dome", "closed"})
#: Reused verbatim from scripts/nfl_forecast_weather_screen.py /
#: scripts/nfl_weather_battery_screen.py (the warm_team_cold_late mechanism's
#: static warm-winter-metro away-team list).
_FORECAST_WARM_METRO_TEAM_CODES = frozenset(
    {"MIA", "TB", "JAX", "ARI", "SF", "OAK", "LA", "LAC", "SD", "HOU", "DAL", "NO", "LV"}
)
_FORECAST_TEMP_GAP_THRESHOLD_F = 25.0
_FORECAST_WARM_TEAM_TEMP_THRESHOLD_F = 35.0
_FORECAST_WARM_TEAM_MIN_WEEK = 13
_FORECAST_DOME_TEAM_TEMP_THRESHOLD_F = 40.0
_FORECAST_HIGH_WIND_THRESHOLD_MPH = 15.0
_FORECAST_DOME_COLD_WINDY_TEMP_THRESHOLD_F = 32.0
_FORECAST_DOME_COLD_WINDY_WIND_THRESHOLD_MPH = 10.0
_FORECAST_PRECIP_PROB_THRESHOLD_PCT = 60.0
_FORECAST_HIGH_TOTAL_THRESHOLD = 47.0
_FORECAST_TEMP_SWING_THRESHOLD_F = 30.0
_DEFAULT_FORECAST_ARCHIVE_PATH = (
    "data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet"
)


def _forecast_weather_archive(repo_root: Path, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    rel_path = str(params.get("forecast_archive_path", _DEFAULT_FORECAST_ARCHIVE_PATH))
    archive_path = repo_root / rel_path
    if not archive_path.is_file():
        raise ExperimentRunnerError(
            f"Forecast archive not found: {archive_path}. Build it with "
            "scripts/ingest_forecast_archive.py --cutoff-mode kickoff_nearest first, or pass "
            "construct.params.forecast_archive_path explicitly."
        )
    archive = pd.read_parquet(
        archive_path,
        columns=[
            "game_id",
            "forecast_temp_f",
            "forecast_wind_mph",
            "forecast_precip_prob_pct",
            "fetch_status",
        ],
    )
    return archive, rel_path


def _team_season_pass_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """``pass_attempt / (pass_attempt + rush_attempt)`` per (season, team), over

    every raw regular-season play where ``posteam == team`` -- the volume
    (play-calling) analogue of ``_team_season_penalty_rate``, same shape
    (columns ``season``, ``team``, ``rate``) so it drops directly into the
    already-reviewed ``_year_over_year_reliability``/``_lag_and_quartile``
    helpers below with no changes to either.
    """

    plays = pbp.loc[pbp["posteam"].notna()].copy()
    plays["pass_attempt"] = pd.to_numeric(plays["pass_attempt"], errors="coerce").fillna(0.0)
    plays["rush_attempt"] = pd.to_numeric(plays["rush_attempt"], errors="coerce").fillna(0.0)
    plays["team"] = _canonical_team(plays["posteam"])
    grouped = plays.groupby(["season", "team"]).agg(
        pass_attempts=("pass_attempt", "sum"), rush_attempts=("rush_attempt", "sum")
    )
    denominator = grouped["pass_attempts"] + grouped["rush_attempts"]
    grouped["rate"] = grouped["pass_attempts"] / denominator
    return grouped.reset_index()


def _forecast_weather_game_table(
    features: pd.DataFrame, repo_root: Path, params: dict[str, Any]
) -> tuple[pd.DataFrame, str]:
    """One row per REG game (pushes dropped), with every column the 6
    forecast-weather builders below need already attached: ``outdoor``,
    ``week_block``, ``team_covered`` (=``home_cover``), the forecast archive's
    own ``forecast_temp_f``/``forecast_wind_mph``/``forecast_precip_prob_pct``,
    ``away_modal_roof`` and ``climate_temp`` (away team's own same-season
    aggregates, ACTUAL weather, same convention as
    scripts/nfl_forecast_weather_screen.py), and ``away_prior_actual_temp``
    (the away team's own immediately-preceding same-season game's actual
    temp, for the temp-swing-vs-prior-week cell).
    """

    # game_features.parquet already carries its OWN temp/wind/gameday columns
    # (unlike the bias-battery builders' source table); only stadium/roof are
    # missing from it, so only those two are pulled from schedules -- pulling
    # temp/wind/gameday too would silently collide and suffix (_x/_y) instead
    # of erroring, which is exactly the bug this comment is here to prevent
    # from being reintroduced.
    schedules_path = _latest_schedules_snapshot(repo_root)
    schedules = pd.read_parquet(schedules_path, columns=["game_id", "stadium", "roof"])

    reg = features.loc[features["game_type"] == "REG"].copy()
    reg["home_team"] = _canonical_team(reg["home_team"])
    reg["away_team"] = _canonical_team(reg["away_team"])
    reg["season"] = reg["season"].astype(int)
    reg["week"] = reg["week"].astype(int)
    reg["spread_line"] = pd.to_numeric(reg["spread_line"], errors="coerce")
    reg["total_line"] = pd.to_numeric(reg["total_line"], errors="coerce")
    reg = reg.merge(schedules, on="game_id", how="left", validate="one_to_one")
    reg["gameday"] = pd.to_datetime(reg["gameday"], errors="raise")
    reg["temp"] = pd.to_numeric(reg["temp"], errors="coerce")
    reg["wind"] = pd.to_numeric(reg["wind"], errors="coerce")

    reg = reg.loc[reg["home_cover"].notna()].copy()  # pushes dropped
    reg["team_covered"] = reg["home_cover"]
    reg["outdoor"] = reg["roof"].isin(_FORECAST_OUTDOOR_ROOFS)
    reg["week_block"] = reg["season"] * 100 + reg["week"]

    archive, archive_rel_path = _forecast_weather_archive(repo_root, params)
    reg = reg.merge(archive, on="game_id", how="left", validate="one_to_one")

    modal_roof = (
        reg.groupby(["home_team", "season"])["roof"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)  # type: ignore[type-var]
        .rename("away_modal_roof")
    )
    reg = reg.merge(modal_roof, left_on=["away_team", "season"], right_index=True, how="left")

    outdoor_home = reg.loc[reg["outdoor"]]
    team_climate = (
        outdoor_home.groupby(["home_team", "season"])
        .agg(climate_temp=("temp", "mean"))
        .reset_index()
    )
    reg = reg.merge(
        team_climate,
        left_on=["away_team", "season"],
        right_on=["home_team", "season"],
        how="left",
        suffixes=("", "_climate"),
    )
    if "home_team_climate" in reg.columns:
        reg = reg.drop(columns=["home_team_climate"])

    home_side = reg[["game_id", "season", "gameday", "home_team", "temp"]].rename(
        columns={"home_team": "team"}
    )
    home_side["is_home"] = True
    away_side = reg[["game_id", "season", "gameday", "away_team", "temp"]].rename(
        columns={"away_team": "team"}
    )
    away_side["is_home"] = False
    team_games = pd.concat([home_side, away_side], ignore_index=True)
    team_games["team"] = _canonical_team(team_games["team"])
    team_games = team_games.sort_values(["team", "season", "gameday"])
    team_games["prior_actual_temp"] = team_games.groupby(["team", "season"])["temp"].shift(1)
    away_prior = team_games.loc[
        ~team_games["is_home"], ["game_id", "team", "prior_actual_temp"]
    ].rename(columns={"team": "away_team", "prior_actual_temp": "away_prior_actual_temp"})
    reg = reg.merge(away_prior, on=["game_id", "away_team"], how="left", validate="one_to_one")

    return reg.reset_index(drop=True), archive_rel_path


def _forecast_weather_construct(
    table: pd.DataFrame,
    flag: pd.Series,
    *,
    sign: int,
    archive_rel_path: str,
    reliability: float | None = None,
    reliability_pairs: int | None = None,
    reliability_note: str,
    extra_population_note: str = "",
) -> SubsetBiasConstruct:
    population_note = (
        f"Forecast archive: {archive_rel_path} (kickoff_nearest cutoff, model=GFS, REG "
        "2009-2025). One row per REG game (pushes dropped), NOT the team-long shape other "
        "builders in this module use -- see the forecast-weather section header for why."
    )
    if extra_population_note:
        population_note = f"{population_note} {extra_population_note}"
    return SubsetBiasConstruct(
        table=table,
        flag=flag.fillna(False).astype(bool),
        eligible=None,
        sign=sign,
        reliability=reliability,
        reliability_pairs=reliability_pairs,
        reliability_note=reliability_note,
        population_note=population_note,
    )


def _flag_forecast_weather_kn_warm_team_cold_late(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons
    table, archive_rel_path = _forecast_weather_game_table(features, repo_root, params)
    flag = (
        table["away_team"].isin(_FORECAST_WARM_METRO_TEAM_CODES)
        & table["outdoor"]
        & (table["forecast_temp_f"] <= _FORECAST_WARM_TEAM_TEMP_THRESHOLD_F)
        & (table["week"] >= _FORECAST_WARM_TEAM_MIN_WEEK)
    )
    return _forecast_weather_construct(
        table,
        flag,
        sign=1,
        archive_rel_path=archive_rel_path,
        reliability_note=(
            "warm_team_cold_late is a per-game situational condition (this week's forecast + "
            "this away team's static warm-winter-metro membership), not a persistent per-team "
            "trait with a year-over-year value to split-half."
        ),
        extra_population_note=(
            "Mirrors forecast_weather_warm_team_cold_late (tuesday_noon, REG 2020-2025) with "
            "kickoff_nearest substituted and the population extended to REG 2009-2025 -- same "
            "flag: away team in the static warm-winter-metro list AND outdoor AND forecast "
            "temp<=35F AND week>=13."
        ),
    )


def _flag_forecast_weather_kn_temp_gap_cold_visitor(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons
    table, archive_rel_path = _forecast_weather_game_table(features, repo_root, params)
    temp_gap = table["climate_temp"] - table["forecast_temp_f"]
    flag = table["outdoor"] & (temp_gap >= _FORECAST_TEMP_GAP_THRESHOLD_F)
    return _forecast_weather_construct(
        table,
        flag,
        sign=1,
        archive_rel_path=archive_rel_path,
        reliability_note=(
            "temp_gap_cold_visitor is a per-game situational condition (this away team's own "
            "same-season climatological-normal home temp minus this week's forecast temp), not "
            "a persistent per-team trait with a year-over-year value to split-half."
        ),
        extra_population_note=(
            "Mirrors forecast_weather_temp_gap_cold_visitor (tuesday_noon, REG 2020-2025) with "
            "kickoff_nearest substituted and the population extended to REG 2009-2025 -- same "
            "flag: away team's own climatological-normal outdoor home temp (ACTUAL, same-season "
            "aggregate) minus this game's forecast temp >= 25F, AND outdoor."
        ),
    )


def _flag_forecast_weather_kn_wind_passing_away_favorite(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    """NEW cell (2026-08-20 backward-extension family): forecast wind >= 15mph

    AND the AWAY team is both the market favorite (spread_line < 0, this
    module's convention: positive spread_line = home favored, see
    ``_flag_large_favorite``'s ``team_spread``) AND that team's PRIOR-season
    pass-rate quartile (global qcut(4) over every (team, season) pair with a
    valid year-over-year lag, mirroring ``_flag_penalty_rate_quartile``'s
    construction exactly but on pass rate instead of penalty rate) is Q4 (most
    pass-heavy). Predicted POSITIVE home_cover edge: a pass-heavy road
    favorite's game plan is disrupted by real wind, benefiting the home
    underdog. Deliberately one-sided (away-favorite only, not
    home-favorite-symmetric) so the flag has a single, unambiguous sign -- see
    the section header for why a mixed-sign construction was avoided.
    """

    del seasons
    pbp_raw_root = Path(params.get("pbp_raw_root", repo_root / "data" / "pbp" / "raw"))
    snapshot = latest_pbp_snapshot(pbp_raw_root)
    pbp = load_pbp_snapshot(snapshot, include_postseason=False)
    rate = _team_season_pass_rate(pbp)
    reliability, reliability_pairs = _year_over_year_reliability(rate)
    lagged = _lag_and_quartile(rate)

    table, archive_rel_path = _forecast_weather_game_table(features, repo_root, params)
    away_lag = lagged.rename(columns={"team": "away_team", "quartile": "away_pass_rate_quartile"})[
        ["away_team", "season", "away_pass_rate_quartile"]
    ]
    table = table.merge(away_lag, on=["away_team", "season"], how="left")

    away_favorite = table["spread_line"] < 0.0
    away_pass_heavy = table["away_pass_rate_quartile"] == 4
    flag = (
        table["outdoor"]
        & (table["forecast_wind_mph"] >= _FORECAST_HIGH_WIND_THRESHOLD_MPH)
        & (away_favorite & away_pass_heavy)
    )
    return _forecast_weather_construct(
        table,
        flag,
        sign=1,
        archive_rel_path=archive_rel_path,
        reliability=reliability,
        reliability_pairs=reliability_pairs,
        reliability_note=(
            f"Year-over-year Pearson correlation of team-season pass rate (pass_attempt / "
            f"(pass_attempt+rush_attempt) over every raw REG play where posteam==team), "
            f"{reliability_pairs} team-season pairs, PBP snapshot {snapshot.snapshot_id}."
        ),
        extra_population_note=(
            f"NEW cell, not a rerun of any tuesday_noon sibling. PBP snapshot "
            f"{snapshot.snapshot_id}. away_pass_rate_quartile is the away team's PRIOR-season "
            "pass-attempt-rate global quartile (Q4=most pass-heavy); rows with no valid "
            "year-over-year lag (a team's first PBP-visible season) have "
            "away_pass_rate_quartile=NaN and the flag is forced False on them."
        ),
    )


def _flag_forecast_weather_kn_precip_high_total(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    """NEW cell: outdoor AND forecast precip probability >= 60% AND total_line

    >= 47. Predicted POSITIVE home_cover edge (consistent with this family's
    other adverse-weather cells: a high total suggests the market has not
    fully priced in precip-driven scoring suppression, and the home team is
    disclosed-conventionally assumed better adapted to its own site's weather
    -- the SAME unverified folk mechanism the sibling cells already carry, not
    a new assumption).
    """

    del seasons
    table, archive_rel_path = _forecast_weather_game_table(features, repo_root, params)
    flag = (
        table["outdoor"]
        & (table["forecast_precip_prob_pct"] >= _FORECAST_PRECIP_PROB_THRESHOLD_PCT)
        & (table["total_line"] >= _FORECAST_HIGH_TOTAL_THRESHOLD)
    )
    return _forecast_weather_construct(
        table,
        flag,
        sign=1,
        archive_rel_path=archive_rel_path,
        reliability_note=(
            "precip_high_total is a per-game situational condition (this week's forecast precip "
            "probability and this game's own market total), not a persistent per-team trait with "
            "a year-over-year value to split-half."
        ),
        extra_population_note=(
            "NEW cell, not a rerun of any tuesday_noon sibling (the tuesday_noon archive never "
            "captured a precipitation-probability field). forecast_precip_prob_pct is GFS MOS "
            "p06 (6h precip probability), falling back to p12 (12h) when p06 is null on the "
            "selected row; see scripts/ingest_forecast_archive.py's nearest_row_with_field."
        ),
    )


def _flag_forecast_weather_kn_temp_swing_prior_week(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    """NEW cell: outdoor AND |forecast_temp_f - away team's own immediately

    preceding same-season game's ACTUAL temp| >= 30F. Predicted POSITIVE
    home_cover edge: a large temperature swing (either direction) since the
    away team's last game is a disruption borne only by the visitor.
    """

    del seasons
    table, archive_rel_path = _forecast_weather_game_table(features, repo_root, params)
    temp_swing = (table["forecast_temp_f"] - table["away_prior_actual_temp"]).abs()
    flag = table["outdoor"] & (temp_swing >= _FORECAST_TEMP_SWING_THRESHOLD_F)
    return _forecast_weather_construct(
        table,
        flag,
        sign=1,
        archive_rel_path=archive_rel_path,
        reliability_note=(
            "temp_swing_prior_week is a per-game situational condition (this away team's own "
            "immediately preceding game's actual temp vs. this week's forecast temp), not a "
            "persistent per-team trait with a year-over-year value to split-half."
        ),
        extra_population_note=(
            "NEW cell, not a rerun of any tuesday_noon sibling. away_prior_actual_temp is the "
            "away team's own immediately preceding SAME-SEASON game's actual temp (either home "
            "or away); a team's first game of a season has no prior game and the flag is forced "
            "False on it (pregame-safe: only already-played games are used)."
        ),
    )


def _flag_forecast_weather_kn_dome_cold_windy(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    """NEW cell: away team's modal home roof this season is dome/closed AND

    this game is outdoor AND forecast_temp_f <= 32F AND forecast_wind_mph >=
    10mph. A compound, stricter version of forecast_weather_dome_team_outdoors_cold
    (temp<=40F alone) -- tests whether COLD+WINDY together compounds the
    dome-team disadvantage beyond cold alone, a distinct predeclared
    hypothesis, not a threshold retune of the sibling cell.
    """

    del seasons
    table, archive_rel_path = _forecast_weather_game_table(features, repo_root, params)
    flag = (
        table["away_modal_roof"].isin(_FORECAST_DOME_CLOSED_ROOFS)
        & table["outdoor"]
        & (table["forecast_temp_f"] <= _FORECAST_DOME_COLD_WINDY_TEMP_THRESHOLD_F)
        & (table["forecast_wind_mph"] >= _FORECAST_DOME_COLD_WINDY_WIND_THRESHOLD_MPH)
    )
    return _forecast_weather_construct(
        table,
        flag,
        sign=1,
        archive_rel_path=archive_rel_path,
        reliability_note=(
            "dome_cold_windy is a per-game situational condition (this away team's own "
            "same-season modal home roof and this week's forecast temp/wind), not a persistent "
            "per-team trait with a year-over-year value to split-half."
        ),
        extra_population_note=(
            "NEW cell, compounds forecast_weather_dome_team_outdoors_cold's cold-only condition "
            "(temp<=40F) with a wind>=10mph requirement and a stricter temp<=32F -- a distinct "
            "predeclared hypothesis (cold+windy compounding), not a threshold retune."
        ),
    )


# ---------------------------------------------------------------------------
# Interim head-coach builders: motivation/effort discontinuity after a
# mid-season coaching change (docs/data_source_scout_v3.md section 5).
# ---------------------------------------------------------------------------
#
# New signal family (2026-08-20), predeclared in full in
# docs/interim_coach_screen.md BEFORE any cover-rate sign was looked at.
# Distinct from -- and explicitly checked for overlap with -- the already-live
# hc_year_one_fade_overlay challenger (nfl_ats.coach_fade_overlay): that
# family flags a team whose CURRENT-season coach is new relative to LAST
# season (a whole-season condition, no in-season discontinuity required);
# THIS family flags a team whose coach changed WITHIN the current season (an
# in-season firing/suspension), a narrower and rarer within-season event. A
# team can be both (a mid-season-hired interim who is also new relative to
# last season is trivially true, since the fired predecessor WAS last
# season's coach too) -- overlap is expected and reported, not a bug.
#
# Source: the Pro Football Rumors "interim coaches since 2000" list
# (data/raw/interim_coaches/<snapshot>/parsed_table.csv; see manifest.json in
# the same directory for fetch provenance and cross-checks -- 3 randomly
# selected entries independently verified via WebSearch, plus 2 more spot
# checks and the 2012 Saints date resolution verified directly against
# schedules.parquet, 6 of 6 agreeing). Joined onto the newest
# data/raw/*/schedules.parquet snapshot's own PER-GAME home_coach/away_coach
# field -- not a takeover-date range -- because that field is strictly more
# precise: three spot-checked entries (BUF 2009, DEN 2010, NYG 2017) showed
# the coach-name transition lands on exactly the week boundary the PFR
# takeover date implies, and it directly resolved the two 2012 Saints entries
# (Kromer/Vitt) that PFR's own article gives no date for at all.
#
# Joinable population is 2009-2025 only (game_features.parquet's own season
# floor, measured); 13 of the 52 listed interim stints (seasons 2000-2008)
# cannot be graded and are excluded from every cell below -- an honest
# coverage limit, not a defect in the source list.

_INTERIM_COACH_SEASON_FLOOR = 2009


def _latest_interim_coaches_snapshot(repo_root: Path) -> Path:
    candidates = sorted((repo_root / "data" / "raw" / "interim_coaches").glob("*/parsed_table.csv"))
    if not candidates:
        raise ExperimentRunnerError(
            f"No data/raw/interim_coaches/*/parsed_table.csv snapshot found under {repo_root}. "
            "Fetch the Pro Football Rumors interim-coach list -- see docs/interim_coach_screen.md."
        )
    return candidates[-1]


@dataclass(frozen=True)
class _InterimCoachTraitData:
    """One row per (game_id, team) that matched an interim-coach stint.

    ``entry_id`` identifies the specific stint (a (interim coach, team,
    season) triple from the source list); ``interim_game_number`` is a
    1-indexed rank within that stint ordered by gameday (the "interim window
    length so far" sub-flag); ``fired_coach_was_year_one``/
    ``fired_coach_year_one_known`` reuse
    ``coach_fade_overlay.team_season_primary_coach``'s EXACT year-1
    definition, applied to the season BEFORE the takeover season, to ask
    whether the coach who got fired was himself in year 1 of his own tenure.
    """

    game_trait: pd.DataFrame
    n_entries_total: int
    n_entries_joinable: int
    snapshot_id: str


def _build_interim_coach_trait_data(repo_root: Path) -> _InterimCoachTraitData:
    from nfl_ats.coach_fade_overlay import team_season_primary_coach

    parsed_path = _latest_interim_coaches_snapshot(repo_root)
    parsed_raw = pd.read_csv(parsed_path)
    required = {
        "entry_id",
        "interim_coach_name",
        "team_abbr",
        "predecessor_coach_name",
        "season",
        "joinable_2009plus",
    }
    missing = sorted(required.difference(parsed_raw.columns))
    if missing:
        raise ExperimentRunnerError(f"{parsed_path} is missing columns: {', '.join(missing)}")
    if "predecessor_status" not in parsed_raw.columns:
        parsed_raw["predecessor_status"] = "fired"
    n_entries_total = int(parsed_raw["entry_id"].nunique())

    parsed = parsed_raw.loc[parsed_raw["joinable_2009plus"].astype(bool)].copy()
    parsed["team_abbr"] = _canonical_team(parsed["team_abbr"].astype(str).str.strip())
    parsed["season"] = parsed["season"].astype(int)
    parsed["interim_coach_name"] = parsed["interim_coach_name"].astype(str).str.strip()
    n_entries_joinable = int(parsed["entry_id"].nunique())

    schedules_path = _latest_schedules_snapshot(repo_root)
    schedules = pd.read_parquet(schedules_path)
    sched_cols = {
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "home_team",
        "away_team",
        "home_coach",
        "away_coach",
    }
    missing_sched = sorted(sched_cols.difference(schedules.columns))
    if missing_sched:
        raise ExperimentRunnerError(
            f"{schedules_path} is missing columns: {', '.join(missing_sched)}"
        )
    reg = schedules.loc[schedules["game_type"].astype(str).eq("REG")].copy()
    reg["gameday"] = pd.to_datetime(reg["gameday"], errors="raise")

    sides = []
    for team_col, coach_col in (("home_team", "home_coach"), ("away_team", "away_coach")):
        side = pd.DataFrame(
            {
                "game_id": reg["game_id"].astype(str),
                "season": reg["season"].astype(int),
                "gameday": reg["gameday"],
                "team": _canonical_team(reg[team_col]),
                "coach": reg[coach_col].astype(str).str.strip(),
            }
        )
        sides.append(side)
    long_sched = pd.concat(sides, ignore_index=True)

    # Stage 1 (primary, preferred): exact (team, season, credited coach name) match
    # against schedules.parquet's own per-game field -- proven exact-to-the-week
    # against 3 spot-checked entries (see the module docstring above this section).
    matched_name = long_sched.merge(
        parsed[
            [
                "entry_id",
                "interim_coach_name",
                "team_abbr",
                "season",
                "predecessor_status",
            ]
        ],
        left_on=["team", "season", "coach"],
        right_on=["team_abbr", "season", "interim_coach_name"],
        how="inner",
    )
    dup = int(matched_name.duplicated(subset=["game_id", "team"]).sum())
    if dup:
        raise ExperimentRunnerError(
            f"{dup} (game_id, team) row(s) matched more than one interim-coach entry -- "
            f"{parsed_path} likely has an ambiguous (team, season, coach_name) triple"
        )

    # Stage 2 (fallback, measured this session to be needed for 10 of 39 joinable
    # entries): schedules.parquet's home_coach/away_coach field does NOT always
    # reflect an in-season interim change -- for some team-seasons it credits the
    # FIRED coach for every remaining game of the season (measured directly:
    # MIA 2015, TEN 2015, LA 2016, CAR 2019, NYJ 2024, NO 2024, CHI 2024, TEN 2025,
    # NYG 2025 never show the interim's name at all). For any joinable entry with
    # ZERO stage-1 matches, fall back to a takeover-date range (team, season,
    # gameday >= takeover_date_iso) -- the PFR list's own stated date, cross-checked
    # in manifest.json.
    matched_via_name = set(matched_name["entry_id"].unique().tolist())
    unmatched = parsed.loc[~parsed["entry_id"].isin(matched_via_name)].copy()
    unmatched["takeover_date"] = pd.to_datetime(unmatched["takeover_date_iso"], errors="raise")

    fallback_frames: list[pd.DataFrame] = []
    for record in unmatched.to_dict("records"):
        team_abbr = str(record["team_abbr"])
        season_value = int(record["season"])
        takeover_date = record["takeover_date"]
        window = long_sched.loc[
            (long_sched["team"] == team_abbr)
            & (long_sched["season"] == season_value)
            & (long_sched["gameday"] >= takeover_date)
        ].copy()
        if window.empty:
            raise ExperimentRunnerError(
                f"Interim-coach entry {record['entry_id']} ({record['interim_coach_name']}, "
                f"{team_abbr} {season_value}) matched NEITHER schedules.parquet's own coach "
                "name field NOR a takeover-date fallback -- no REG-season game for that team/"
                f"season falls on or after {takeover_date.date()}. Fix the entry in "
                f"{parsed_path}."
            )
        window["entry_id"] = record["entry_id"]
        window["predecessor_status"] = record["predecessor_status"]
        fallback_frames.append(window)
    matched_fallback = (
        pd.concat(fallback_frames, ignore_index=True)
        if fallback_frames
        else matched_name.loc[[], ["game_id", "team", "gameday", "entry_id", "predecessor_status"]]
    )

    join_cols = ["game_id", "team", "gameday", "entry_id", "predecessor_status"]
    matched = pd.concat([matched_name[join_cols], matched_fallback[join_cols]], ignore_index=True)
    dup2 = int(matched.duplicated(subset=["game_id", "team"]).sum())
    if dup2:
        raise ExperimentRunnerError(
            f"{dup2} (game_id, team) row(s) matched more than one interim-coach entry after "
            "combining the name-match and date-fallback joins -- an overlapping stint window in "
            f"{parsed_path}"
        )
    matched_entries = int(matched["entry_id"].nunique())
    if matched_entries != n_entries_joinable:
        raise ExperimentRunnerError(
            f"Only {matched_entries} of {n_entries_joinable} joinable interim-coach entries in "
            f"{parsed_path} matched a schedules.parquet row via name or date-fallback -- fix the "
            "mismatched row rather than silently dropping it."
        )

    matched = matched.sort_values(["entry_id", "gameday"]).copy()
    matched["interim_game_number"] = matched.groupby("entry_id").cumcount() + 1
    matched["first_game_under_interim"] = matched["interim_game_number"] == 1

    primary = team_season_primary_coach(schedules)
    primary_lookup = {
        (str(team), int(season)): str(coach)
        for team, season, coach in zip(
            primary["team"], primary["season"], primary["primary_coach"], strict=True
        )
    }
    entries = parsed.drop_duplicates(subset=["entry_id"])
    year_one_flag: dict[int, bool] = {}
    year_one_known: dict[int, bool] = {}
    for record in entries.to_dict("records"):
        entry_id = int(record["entry_id"])
        season_before = int(record["season"]) - 1
        season_before_prior = season_before - 1
        team_abbr = str(record["team_abbr"])
        predecessor_credited = primary_lookup.get((team_abbr, season_before))
        prior_credited = primary_lookup.get((team_abbr, season_before_prior))
        known = predecessor_credited is not None and prior_credited is not None
        year_one_known[entry_id] = bool(known)
        year_one_flag[entry_id] = bool(known and predecessor_credited != prior_credited)

    matched["fired_coach_was_year_one"] = (
        matched["entry_id"].map(year_one_flag).fillna(False).astype(bool)
    )
    matched["fired_coach_year_one_known"] = (
        matched["entry_id"].map(year_one_known).fillna(False).astype(bool)
    )

    game_trait = matched[
        [
            "game_id",
            "team",
            "entry_id",
            "predecessor_status",
            "first_game_under_interim",
            "interim_game_number",
            "fired_coach_was_year_one",
            "fired_coach_year_one_known",
        ]
    ].copy()

    return _InterimCoachTraitData(
        game_trait=game_trait,
        n_entries_total=n_entries_total,
        n_entries_joinable=n_entries_joinable,
        snapshot_id=parsed_path.parent.name,
    )


def _interim_coach_team_game_table(
    features: pd.DataFrame, repo_root: Path
) -> tuple[pd.DataFrame, _InterimCoachTraitData]:
    table = _base_team_game_table(features)
    trait_data = _build_interim_coach_trait_data(repo_root)
    merged = table.merge(trait_data.game_trait, on=["game_id", "team"], how="left")
    merged["under_interim"] = merged["entry_id"].notna()
    merged["predecessor_status"] = merged["predecessor_status"].fillna("")
    merged["first_game_under_interim"] = (
        merged["first_game_under_interim"].fillna(False).astype(bool)
    )
    merged["interim_game_number"] = merged["interim_game_number"].fillna(0).astype(int)
    merged["fired_coach_was_year_one"] = (
        merged["fired_coach_was_year_one"].fillna(False).astype(bool)
    )
    merged["fired_coach_year_one_known"] = (
        merged["fired_coach_year_one_known"].fillna(False).astype(bool)
    )
    return merged, trait_data


def _interim_coach_population_note(trait_data: _InterimCoachTraitData) -> str:
    return (
        f"Interim-coach snapshot {trait_data.snapshot_id}: Pro Football Rumors' interim-coach "
        f"list ({trait_data.n_entries_total} stints, 2000-2025) joined onto the newest schedules "
        "snapshot's own PER-GAME home_coach/away_coach field (team, season, credited coach "
        f"name), not a takeover-date range. {trait_data.n_entries_joinable} of those stints fall "
        f"in this project's {_INTERIM_COACH_SEASON_FLOOR}+ graded population; earlier stints "
        "(2000-2008) cannot be graded against game_features.parquet's own season floor. See "
        "data/raw/interim_coaches/*/manifest.json for fetch provenance and cross-checks, and "
        "docs/interim_coach_screen.md for the predeclared cell family."
    )


def _flag_interim_hc_active(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons
    exclude_suspension = bool(params.get("exclude_suspension_cases", False))
    table, trait_data = _interim_coach_team_game_table(features, repo_root)
    if exclude_suspension:
        table = table.loc[table["predecessor_status"] != "suspended"].copy()
    flag = table["under_interim"]
    note = _interim_coach_population_note(trait_data)
    if exclude_suspension:
        note += (
            " Sensitivity run: predecessor_status=='suspended' cases (2012 Saints, Sean Payton's "
            "Bounty Scandal suspension -- not a firing) excluded from both arms."
        )
    return SubsetBiasConstruct(
        table=table,
        flag=flag,
        eligible=None,
        sign=1,  # hypothesis: motivation/effort discontinuity lifts cover rate under interim HCs
        reliability=None,
        reliability_pairs=None,
        reliability_note=(
            "under_interim is a one-off situational event (a specific mid-season coaching "
            "change), not a persistent per-team trait that would repeat year over year -- there "
            "is nothing to split-half."
        ),
        population_note=note,
    )


def _flag_interim_hc_first_game(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons
    exclude_suspension = bool(params.get("exclude_suspension_cases", False))
    table, trait_data = _interim_coach_team_game_table(features, repo_root)
    if exclude_suspension:
        table = table.loc[table["predecessor_status"] != "suspended"].copy()
    flag = table["first_game_under_interim"]
    return SubsetBiasConstruct(
        table=table,
        flag=flag,
        eligible=None,
        sign=1,  # folklore: teams often cover their FIRST game under a new interim coach
        reliability=None,
        reliability_pairs=None,
        reliability_note=(
            "first_game_under_interim is a one-off situational event, not a persistent per-team "
            "trait -- there is nothing to split-half."
        ),
        population_note=_interim_coach_population_note(trait_data)
        + " Flag = the FIRST REG-season game credited to a specific interim stint, vs. everyone "
        "else in the league (same one-sided design as hc_year_one_fade/the bias battery).",
    )


def _flag_interim_hc_home(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params
    table, trait_data = _interim_coach_team_game_table(features, repo_root)
    eligible = table["under_interim"]
    flag = table["is_home"]
    return SubsetBiasConstruct(
        table=table,
        flag=flag,
        eligible=eligible,
        sign=1,  # arbitrary reporting convention (home > road within interim games); NO a priori
        # mechanism was predeclared for this direction -- see docs/interim_coach_screen.md.
        reliability=None,
        reliability_pairs=None,
        reliability_note=(
            "is_home is a per-game situational condition, not a persistent per-team trait -- "
            "there is nothing to split-half."
        ),
        population_note=_interim_coach_population_note(trait_data)
        + " Restricted to under_interim games only (eligible); flag = home vs. road WITHIN that "
        "population. Sign is an arbitrary reporting convention (positive = home covers more than "
        "road within interim games), not a predeclared directional mechanism -- report the "
        "number honestly regardless of which way it points.",
    )


def _flag_interim_hc_fired_year_one(
    features: pd.DataFrame, seasons: tuple[int, int], params: dict[str, Any], repo_root: Path
) -> SubsetBiasConstruct:
    del seasons, params
    table, trait_data = _interim_coach_team_game_table(features, repo_root)
    eligible = table["under_interim"] & table["fired_coach_year_one_known"]
    flag = table["fired_coach_was_year_one"]
    return SubsetBiasConstruct(
        table=table,
        flag=flag,
        eligible=eligible,
        sign=-1,  # mechanism: firing a coach in his OWN year 1 signals organizational chaos/panic
        # rather than a considered reset, hypothesized to BLUNT the interim cover-rate bump
        # relative to firing a longer-tenured coach.
        reliability=None,
        reliability_pairs=None,
        reliability_note=(
            "fired_coach_was_year_one is a one-off situational event (this specific firing's "
            "circumstances), not a persistent per-team trait -- there is nothing to split-half."
        ),
        population_note=_interim_coach_population_note(trait_data)
        + " Restricted to under_interim games with a KNOWN predecessor tenure (the team's prior "
        "season's primary coach AND the season before that both observed in the "
        f"{_INTERIM_COACH_SEASON_FLOOR}+ data, via nfl_ats.coach_fade_overlay."
        "team_season_primary_coach -- the SAME year-1 definition hc_year_one_fade_overlay uses, "
        "applied to the season BEFORE the takeover). flag = the fired coach was himself in year "
        "1 of his own tenure when he was fired.",
    )


FLAG_BUILDERS: dict[str, FlagBuilder] = {
    "penalty_rate_quartile": FlagBuilder(
        name="penalty_rate_quartile",
        leagues=("nfl",),
        description=(
            "Prior-season team penalty-rate quartile 1 (least penalized) vs quartile 4 (most "
            "penalized), global quartile cut, team-game cover rate. Reproduces "
            "scripts/penalty_discipline_interval.py."
        ),
        build=_flag_penalty_rate_quartile,
    ),
    "home_underdog": FlagBuilder(
        name="home_underdog",
        leagues=("nfl",),
        description="Home team getting points, vs. everyone else.",
        build=_flag_home_underdog,
    ),
    "large_favorite": FlagBuilder(
        name="large_favorite",
        leagues=("nfl",),
        description="Favored by more than params.threshold (default 10) points, vs. everyone else.",
        build=_flag_large_favorite,
    ),
    "division_revenge_game": FlagBuilder(
        name="division_revenge_game",
        leagues=("nfl",),
        description=(
            "2nd meeting this season vs. same opponent; team lost the 1st meeting. Ported from "
            "scripts/nfl_bias_battery_screen.py."
        ),
        build=_flag_division_revenge_game,
    ),
    "extra_rest_edge": FlagBuilder(
        name="extra_rest_edge",
        leagues=("nfl",),
        description=(
            "Team's rest minus opponent's rest >= 4 days. Ported from "
            "scripts/nfl_bias_battery_screen.py."
        ),
        build=_flag_extra_rest_edge,
    ),
    "short_week": FlagBuilder(
        name="short_week",
        leagues=("nfl",),
        description="Team's own rest <= 5 days. Ported from scripts/nfl_bias_battery_screen.py.",
        build=_flag_short_week,
    ),
    "west_coast_early_kickoff": FlagBuilder(
        name="west_coast_early_kickoff",
        leagues=("nfl",),
        description=(
            "Traveling Pacific-timezone team, non-PT opponent, kickoff before 14:00 ET. Ported "
            "from scripts/nfl_bias_battery_screen.py."
        ),
        build=_flag_west_coast_early_kickoff,
    ),
    "sandwich_spot": FlagBuilder(
        name="sandwich_spot",
        leagues=("nfl",),
        description=(
            "Non-division game flanked by a division game last week and next week. Ported from "
            "scripts/nfl_bias_battery_screen.py."
        ),
        build=_flag_sandwich_spot,
    ),
    "backup_qb_start": FlagBuilder(
        name="backup_qb_start",
        leagues=("nfl",),
        description=(
            "Starting QB differs from the team's modal QB this season (>=3 prior starts). Ported "
            "from scripts/nfl_bias_battery_screen.py."
        ),
        build=_flag_backup_qb_start,
    ),
    "motivation_mismatch": FlagBuilder(
        name="motivation_mismatch",
        leagues=("nfl",),
        description=(
            "Competitive team (>=40% prior win pct) facing a bad_team_late opponent. Ported from "
            "scripts/nfl_bias_battery_screen.py."
        ),
        build=_flag_motivation_mismatch,
    ),
    "referee_penalty_rate_top_quartile": FlagBuilder(
        name="referee_penalty_rate_top_quartile",
        leagues=("nfl",),
        description=(
            "Home team's referee's prior-season mean total penalties/game (both teams) in the "
            "top quartile, vs. everyone else. See docs/referee_battery.md."
        ),
        build=_flag_referee_penalty_rate_top_quartile,
    ),
    "referee_penalty_rate_bottom_quartile": FlagBuilder(
        name="referee_penalty_rate_bottom_quartile",
        leagues=("nfl",),
        description=(
            "Home team's referee's prior-season mean total penalties/game (both teams) in the "
            "bottom quartile, vs. everyone else. See docs/referee_battery.md."
        ),
        build=_flag_referee_penalty_rate_bottom_quartile,
    ),
    "referee_home_penalty_tilt_top_quartile": FlagBuilder(
        name="referee_home_penalty_tilt_top_quartile",
        leagues=("nfl",),
        description=(
            "Home team's referee's prior-season (away penalties - home penalties) differential "
            "in the top (most home-protective) quartile, vs. everyone else. See "
            "docs/referee_battery.md."
        ),
        build=_flag_referee_home_penalty_tilt_top_quartile,
    ),
    "referee_home_penalty_tilt_bottom_quartile": FlagBuilder(
        name="referee_home_penalty_tilt_bottom_quartile",
        leagues=("nfl",),
        description=(
            "Home team's referee's prior-season (away penalties - home penalties) differential "
            "in the bottom (least home-protective) quartile, vs. everyone else. See "
            "docs/referee_battery.md."
        ),
        build=_flag_referee_home_penalty_tilt_bottom_quartile,
    ),
    "referee_veteran_home_cover": FlagBuilder(
        name="referee_veteran_home_cover",
        leagues=("nfl",),
        description=(
            "Home team's referee has >= params.veteran_threshold (default 5) distinct prior "
            "dataset-visible seasons as head referee, vs. everyone else. See "
            "docs/referee_battery.md."
        ),
        build=_flag_referee_veteran_home_cover,
    ),
    "referee_rookie_home_cover": FlagBuilder(
        name="referee_rookie_home_cover",
        leagues=("nfl",),
        description=(
            "Home team's referee has 0 distinct prior dataset-visible seasons as head referee, "
            "vs. everyone else. See docs/referee_battery.md."
        ),
        build=_flag_referee_rookie_home_cover,
    ),
    "referee_high_flag_heavy_underdog": FlagBuilder(
        name="referee_high_flag_heavy_underdog",
        leagues=("nfl",),
        description=(
            "Home team's referee's prior-season mean_total penalty rate top quartile AND home "
            "team a heavy underdog (spread_line <= -params.underdog_threshold, default 7), vs. "
            "everyone else. Reuses the existing mean_total trait. See "
            "docs/penalty_crew_tendencies.md cell A."
        ),
        build=_flag_referee_high_flag_heavy_underdog,
    ),
    "referee_dpi_tilt_pass_heavy_favorite": FlagBuilder(
        name="referee_dpi_tilt_pass_heavy_favorite",
        leagues=("nfl",),
        description=(
            "Home team is the favorite AND top-quartile prior-rolling pass rate AND the game's "
            "referee's prior-season Defensive Pass Interference rate in the top quartile, vs. "
            "everyone else. See docs/penalty_crew_tendencies.md cell B."
        ),
        build=_flag_referee_dpi_tilt_pass_heavy_favorite,
    ),
    "referee_holding_tilt_run_heavy": FlagBuilder(
        name="referee_holding_tilt_run_heavy",
        leagues=("nfl",),
        description=(
            "Home team in the bottom quartile of prior-rolling pass rate (run-heavy) AND the "
            "game's referee's prior-season Offensive Holding rate in the top quartile, vs. "
            "everyone else. See docs/penalty_crew_tendencies.md cell C."
        ),
        build=_flag_referee_holding_tilt_run_heavy,
    ),
    "referee_flag_rate_high_total_line": FlagBuilder(
        name="referee_flag_rate_high_total_line",
        leagues=("nfl",),
        description=(
            "Home team's referee's prior-season mean_total penalty rate top quartile AND the "
            "game's own total_line in the top quartile, vs. everyone else. Reuses the existing "
            "mean_total trait. See docs/penalty_crew_tendencies.md cell D."
        ),
        build=_flag_referee_flag_rate_high_total_line,
    ),
    "forecast_weather_kn_warm_team_cold_late": FlagBuilder(
        name="forecast_weather_kn_warm_team_cold_late",
        leagues=("nfl",),
        description=(
            "Away team in the static warm-winter-metro list AND outdoor AND kickoff_nearest "
            "forecast temp<=35F AND week>=13. kickoff_nearest/REG 2009-2025 rerun of "
            "forecast_weather_warm_team_cold_late (tuesday_noon, REG 2020-2025)."
        ),
        build=_flag_forecast_weather_kn_warm_team_cold_late,
    ),
    "forecast_weather_kn_temp_gap_cold_visitor": FlagBuilder(
        name="forecast_weather_kn_temp_gap_cold_visitor",
        leagues=("nfl",),
        description=(
            "Away team's own climatological-normal outdoor home temp (actual) minus "
            "kickoff_nearest forecast temp >= 25F, AND outdoor. kickoff_nearest/REG 2009-2025 "
            "rerun of forecast_weather_temp_gap_cold_visitor (tuesday_noon, REG 2020-2025)."
        ),
        build=_flag_forecast_weather_kn_temp_gap_cold_visitor,
    ),
    "forecast_weather_kn_wind_passing_away_favorite": FlagBuilder(
        name="forecast_weather_kn_wind_passing_away_favorite",
        leagues=("nfl",),
        description=(
            "Outdoor AND kickoff_nearest forecast wind>=15mph AND the away team is both the "
            "market favorite and its prior-season pass-attempt-rate quartile is Q4 (most "
            "pass-heavy). NEW cell, 2026-08-20 backward-extension family."
        ),
        build=_flag_forecast_weather_kn_wind_passing_away_favorite,
    ),
    "forecast_weather_kn_precip_high_total": FlagBuilder(
        name="forecast_weather_kn_precip_high_total",
        leagues=("nfl",),
        description=(
            "Outdoor AND kickoff_nearest forecast precip probability>=60% AND total_line>=47. "
            "NEW cell, 2026-08-20 backward-extension family."
        ),
        build=_flag_forecast_weather_kn_precip_high_total,
    ),
    "forecast_weather_kn_temp_swing_prior_week": FlagBuilder(
        name="forecast_weather_kn_temp_swing_prior_week",
        leagues=("nfl",),
        description=(
            "Outdoor AND |kickoff_nearest forecast temp - away team's own immediately preceding "
            "same-season game's actual temp| >= 30F. NEW cell, 2026-08-20 backward-extension "
            "family."
        ),
        build=_flag_forecast_weather_kn_temp_swing_prior_week,
    ),
    "forecast_weather_kn_dome_cold_windy": FlagBuilder(
        name="forecast_weather_kn_dome_cold_windy",
        leagues=("nfl",),
        description=(
            "Away team's modal home roof is dome/closed AND outdoor AND kickoff_nearest forecast "
            "temp<=32F AND forecast wind>=10mph. NEW cell (compounds "
            "forecast_weather_dome_team_outdoors_cold with a wind requirement), 2026-08-20 "
            "backward-extension family."
        ),
        build=_flag_forecast_weather_kn_dome_cold_windy,
    ),
    "interim_hc_active": FlagBuilder(
        name="interim_hc_active",
        leagues=("nfl",),
        description=(
            "Team is currently playing under an in-season interim head coach (Pro Football "
            "Rumors' interim-coach list joined onto schedules.parquet's own per-game credited "
            "coach field), vs. everyone else. params.exclude_suspension_cases (default False) "
            "drops the 2012 Saints (suspension, not a firing) as a sensitivity check. See "
            "docs/interim_coach_screen.md."
        ),
        build=_flag_interim_hc_active,
    ),
    "interim_hc_first_game": FlagBuilder(
        name="interim_hc_first_game",
        leagues=("nfl",),
        description=(
            "Team's FIRST REG-season game under a new interim head coach, vs. everyone else "
            "(the specific bettor-folklore claim). Same source/join and "
            "params.exclude_suspension_cases as interim_hc_active. See "
            "docs/interim_coach_screen.md."
        ),
        build=_flag_interim_hc_first_game,
    ),
    "interim_hc_home": FlagBuilder(
        name="interim_hc_home",
        leagues=("nfl",),
        description=(
            "Restricted to interim-coached games only: home vs. road. Descriptive/exploratory -- "
            "no predeclared directional mechanism, sign is a reporting convention only. See "
            "docs/interim_coach_screen.md."
        ),
        build=_flag_interim_hc_home,
    ),
    "interim_hc_fired_year_one": FlagBuilder(
        name="interim_hc_fired_year_one",
        leagues=("nfl",),
        description=(
            "Restricted to interim-coached games with a known predecessor tenure: the fired "
            "coach was himself in year 1 of his own tenure, vs. not. Hypothesis: a year-1 firing "
            "signals organizational chaos and BLUNTS the interim cover-rate bump. See "
            "docs/interim_coach_screen.md."
        ),
        build=_flag_interim_hc_fired_year_one,
    ),
}


# ---------------------------------------------------------------------------
# The generic subset-vs-complement bootstrap
# ---------------------------------------------------------------------------


def scale_subset_effect(raw_gap_fraction: float, *, sign: int, fraction_of_slate: float) -> float:
    """Full-slate-scaled effect in accuracy POINTS, positive favours the candidate.

    The one place this module converts a cover-rate FRACTION (e.g. 0.0067,
    not 0.67) into accuracy POINTS -- the exact 100x step a hand-transcription
    got backwards this session (``best_pick_tiebreak_cfb_stage0_ecdf_gaussian``'s
    recorded correction note). ``raw_gap_fraction`` must be a fraction;
    passing points here silently reintroduces that bug.

    Scaling precedent: ``hc_year_one_fade`` (``registry/weak_signals.json``)
    scaled a 4.26-point subset gap by the 17.7% of the slate it applies to
    before recording 0.7528; ``penalty_discipline``
    (``scripts/penalty_discipline_interval.py``) scales a two-sided quartile
    gap by the fraction of the slate the compared quartiles represent.
    """

    if sign not in (1, -1):
        raise ValueError("sign must be 1 or -1")
    if not 0.0 <= fraction_of_slate <= 1.0:
        raise ValueError("fraction_of_slate must be in [0, 1]")
    return sign * raw_gap_fraction * 100.0 * fraction_of_slate


def _block_bootstrap_subset_gap(
    df: pd.DataFrame, *, flag: pd.Series, value_col: str, block_col: str, samples: int, seed: int
) -> npt.NDArray[np.float64]:
    """Vectorized joint block bootstrap of ``100*(flag_mean - complement_mean)``.

    Both arms' means for a given draw come from the SAME resampled set of
    blocks (one multinomial draw over the shared block ids), the correct way
    to jointly bootstrap a two-group comparison sharing a blocking structure.
    A generalization of ``scripts/nfl_bias_battery_screen.py``'s
    ``block_bootstrap_two_group`` / ``scripts/penalty_discipline_interval.py``'s
    ``block_bootstrap_quartile_gap`` from a specific pair of groups to any
    boolean flag -- with the same block-id-derivation order (``np.unique``,
    sorted, order-independent of row order), the same single
    ``rng.multinomial`` call shape, and the same sums/counts-via-bincount
    trick, this reproduces either precedent bit-for-bit given the same
    (already-restricted) population, flag, block column, samples, and seed.
    """

    blocks, block_index = np.unique(df[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    values = df[value_col].to_numpy(dtype=np.float64)
    flag_array = flag.to_numpy(dtype=bool)

    sums: dict[bool, npt.NDArray[np.float64]] = {}
    counts: dict[bool, npt.NDArray[np.float64]] = {}
    for group in (True, False):
        mask = flag_array == group
        sums[group] = np.bincount(
            block_index[mask], weights=values[mask], minlength=block_count
        ).astype(np.float64)
        counts[group] = np.bincount(block_index[mask], minlength=block_count).astype(np.float64)

    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    subset_count = drawn @ counts[True]
    complement_count = drawn @ counts[False]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_subset = (drawn @ sums[True]) / subset_count
        mean_complement = (drawn @ sums[False]) / complement_count
    gap = (mean_subset - mean_complement) * 100.0
    valid = (subset_count > 0) & (complement_count > 0)
    return np.asarray(gap[valid], dtype=np.float64)


@dataclass(frozen=True)
class BlockIntervalResult:
    block_kind: str
    block_count: int
    degenerate: bool
    estimate: float
    lower: float
    upper: float
    standard_error: float
    probability_positive: float
    samples: int


def _interval_summary(
    gap_draws: npt.NDArray[np.float64],
    *,
    block_kind: str,
    block_count: int,
    degenerate: bool,
    sign: int,
    fraction_of_slate: float,
    confidence: float,
) -> BlockIntervalResult:
    signed = sign * gap_draws
    scaled = signed * fraction_of_slate
    tail = (1.0 - confidence) / 2.0
    return BlockIntervalResult(
        block_kind=block_kind,
        block_count=block_count,
        degenerate=degenerate,
        estimate=float(np.mean(scaled)),
        lower=float(np.quantile(scaled, tail)),
        upper=float(np.quantile(scaled, 1.0 - tail)),
        standard_error=float(np.std(scaled, ddof=1)),
        probability_positive=float(np.mean(scaled > 0.0)),
        samples=len(scaled),
    )


# ---------------------------------------------------------------------------
# Mechanical classification (AGENTS.md binding rule; see module docstring)
# ---------------------------------------------------------------------------


def widening_factor_to_recross_zero(estimate: float, upper: float) -> float:
    """Symmetric-about-``estimate`` inflation factor that brings ``upper`` back to zero.

    ``upper`` must be strictly below zero, and ``estimate`` must sit strictly
    below ``upper`` (further from zero) -- the usual shape of a wrong-signed
    interval, where the point estimate is more negative than its own upper
    bound. Scaling the interval by factor ``f`` about the point estimate
    sends ``upper -> estimate + f * (upper - estimate)``; solving that for
    ``f`` at a target of zero gives ``f = -estimate / (upper - estimate)``.

    This is exactly the arithmetic behind the reviewer note on
    ``mod06_js_shrinkage_position_prior_cfb``
    (``registry/weak_signals.json``): effect -0.526, interval upper -0.043 ->
    ``0.526 / (0.526 - 0.043) = 1.089`` (reported there, from slightly
    rounder inputs, as "1.082x").
    """

    if upper >= 0.0:
        raise ValueError("upper must be strictly below zero")
    denominator = upper - estimate
    if denominator <= 0.0:
        raise ValueError("estimate must be strictly below upper")
    return -estimate / denominator


@dataclass(frozen=True)
class ClassificationResult:
    classification: str
    closing_ground: str | None
    note: str
    widening_factor: float | None


def classify_subset_bias_result(
    *, estimate: float, lower: float, upper: float
) -> ClassificationResult:
    """The runner's ONE mechanically-computed terminal verdict; see module docstring.

    Never returns ``bounded_by_control``, and the only admissible
    ``closing_ground`` this can ever emit is ``wrong_sign_resolved`` --
    matching ``weak_signals.validate_closure``'s own requirement that
    ``wrong_sign_resolved`` demands an interval entirely below zero.
    """

    if upper >= 0.0:
        return ClassificationResult(
            classification="unresolved_below_power",
            closing_ground=None,
            note=(
                f"Primary interval [{lower:+.4f}, {upper:+.4f}] does not sit entirely below "
                "zero. Per AGENTS.md, an interval crossing zero is never grounds for "
                "rejection on its own."
            ),
            widening_factor=None,
        )
    try:
        factor = widening_factor_to_recross_zero(estimate, upper)
    except ValueError as error:
        return ClassificationResult(
            classification="unresolved_below_power",
            closing_ground=None,
            note=(
                f"Primary interval sits below zero but the widening factor could not be "
                f"computed ({error})."
            ),
            widening_factor=None,
        )
    if factor > HONEST_REFIT_WIDENING_UPPER_BOUND:
        return ClassificationResult(
            classification="refuted_mechanism",
            closing_ground="wrong_sign_resolved",
            note=(
                f"Primary interval [{lower:+.4f}, {upper:+.4f}] sits entirely below zero; "
                f"re-crossing zero needs a {factor:.3f}x widening, which EXCEEDS the documented "
                f"honest refit-correction upper bound {HONEST_REFIT_WIDENING_UPPER_BOUND}x "
                "(docs/estimation_variance.md: '...1.293x to 1.003x (one-sided 95% upper bound "
                "1.099x)')."
            ),
            widening_factor=factor,
        )
    return ClassificationResult(
        classification="unresolved_below_power",
        closing_ground=None,
        note=(
            f"Primary interval [{lower:+.4f}, {upper:+.4f}] sits entirely below zero, but "
            f"re-crossing needs only a {factor:.3f}x widening -- inside the documented honest "
            f"refit-correction band (up to {HONEST_REFIT_WIDENING_UPPER_BOUND}x), so per "
            "AGENTS.md this stays unresolved, not refuted."
        ),
        widening_factor=factor,
    )


# ---------------------------------------------------------------------------
# Opener-grade population loader (population.grade == "opener")
# ---------------------------------------------------------------------------


def _opener_graded_features(
    features: pd.DataFrame, *, repo_root: Path, market_root: Path | None
) -> tuple[pd.DataFrame, str]:
    """Restrict ``features`` to the paired Tuesday-opener archive and overwrite
    ``spread_line``/``home_cover``/``ats_margin`` to the OPENER line.

    Mirrors ``clv.opener_pick_evaluation``'s population definition exactly:
    every REG-season game with BOTH a ``tue_open`` consensus AND a resolvable
    close (``docs/opener_evaluation.md``'s 1,537-game, 2020-2025 archive,
    ``build_pairing_table`` + ``close_reference_table``). Unlike
    ``opener_pick_evaluation`` this skips the margin-model fit entirely --
    ``subset_bias`` flags are pregame-safe situational constructs, not model
    predictions, so all that is needed is which line to grade cover rate
    against. Every registered flag builder reads ``spread_line``/
    ``home_cover`` off whatever features frame it is handed (see
    ``_base_team_game_table``/the bias-battery table builder below), so
    overwriting those two columns here -- plus recomputing ``ats_margin`` for
    consistency, using the exact ``features.add_ats_outcomes`` convention
    (``ats_margin = result - spread_line``; ``home_cover`` = 1/0/NaN-on-push
    from its sign) -- is enough to make every already-registered NFL builder
    opener-aware with zero builder-side changes.
    """

    required = {"game_id", "season", "week", "gameday", "result", "game_type"}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ExperimentRunnerError(
            f"Feature table is missing columns needed for opener grading: {', '.join(missing)}"
        )

    root = market_root or (repo_root / "data" / "market" / "raw")
    reg = features.loc[features["game_type"] == "REG"].copy()
    reg["gameday"] = pd.to_datetime(reg["gameday"], errors="raise")

    pairing = build_pairing_table(
        root,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=reg,
    )
    if pairing.empty:
        raise ExperimentRunnerError(
            f"No {HISTORICAL_CAPTURE_KIND!r} snapshots with decision quotes under {root}; "
            "population.grade='opener' needs the historical odds snapshot archive."
        )
    close = close_reference_table(pairing, reg)
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "home_spread"]
    ].rename(columns={"home_spread": "opener_home_spread"})
    paired_ids = tue_open[["game_id"]].merge(close[["game_id"]], on="game_id", how="inner")

    merged = reg.merge(paired_ids, on="game_id", how="inner").merge(
        tue_open, on="game_id", how="left"
    )
    merged["result"] = pd.to_numeric(merged["result"], errors="coerce")
    merged = merged.loc[merged["result"].notna()].copy()
    if merged.empty:
        raise ExperimentRunnerError(
            f"No completed REG-season games have both a Tuesday opener and a resolvable close "
            f"under {root}."
        )

    merged["spread_line"] = pd.to_numeric(merged["opener_home_spread"], errors="coerce")
    merged["ats_margin"] = merged["result"] - merged["spread_line"]
    merged["home_cover"] = np.select(
        [merged["ats_margin"] > 0.0, merged["ats_margin"] < 0.0], [1.0, 0.0], default=np.nan
    )
    merged = merged.drop(columns=["opener_home_spread"])

    season_lo, season_hi = int(merged["season"].min()), int(merged["season"].max())
    note = (
        f"Opener-grade population: {len(merged)} REG-season games with both a tue_open consensus "
        f"and a resolvable close (nfl_ats.clv.build_pairing_table/close_reference_table, "
        f"capture_kind={HISTORICAL_CAPTURE_KIND!r}, market_root={root}), seasons "
        f"{season_lo}-{season_hi}. spread_line/home_cover/ats_margin are the OPENER line, not the "
        "close. If population.seasons extends outside this coverage, the comparison is silently "
        "trimmed to the intersection (same convention as every other season-window filter in this "
        "module)."
    )
    return merged.reset_index(drop=True), note


# ---------------------------------------------------------------------------
# Running a subset_bias experiment end to end
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsetBiasRunResult:
    spec: ExperimentSpec
    n_total: int
    n_flag: int
    n_complement: int
    fraction_of_slate: float
    raw_gap_pct: float
    effect: float
    primary: BlockIntervalResult
    secondary: BlockIntervalResult | None
    reliability: float | None
    reliability_pairs: int | None
    reliability_note: str
    population_note: str
    classification: ClassificationResult
    builder_description: str
    sign: int


def run_subset_bias_experiment(
    spec: ExperimentSpec,
    *,
    repo_root: Path,
    features_path: Path | None = None,
    market_root: Path | None = None,
) -> SubsetBiasRunResult:
    if spec.experiment_type != "subset_bias":
        raise ExperimentRunnerError(
            f"run_subset_bias_experiment called on a {spec.experiment_type!r} spec"
        )
    builder = FLAG_BUILDERS.get(spec.flag_builder)
    if builder is None:
        raise ExperimentRunnerError(
            f"Unknown construct.flag_builder {spec.flag_builder!r}; registered builders: "
            f"{', '.join(sorted(FLAG_BUILDERS))}"
        )
    if spec.league not in builder.leagues:
        raise ExperimentRunnerError(
            f"flag_builder {spec.flag_builder!r} supports leagues {builder.leagues}, "
            f"not {spec.league!r}"
        )
    if spec.grade == "opener" and spec.league != "nfl":
        raise ExperimentRunnerError(
            "population.grade='opener' is only wired for league='nfl' -- the historical odds "
            "snapshot archive it reads (data/market/raw) is NFL-only; no CFB opener archive exists."
        )
    if spec.reliability_method == "split_half" and builder.build in (
        _flag_home_underdog,
        _flag_large_favorite,
        _flag_division_revenge_game,
        _flag_extra_rest_edge,
        _flag_short_week,
        _flag_west_coast_early_kickoff,
        _flag_sandwich_spot,
        _flag_backup_qb_start,
        _flag_motivation_mismatch,
        _flag_referee_veteran_home_cover,
        _flag_referee_rookie_home_cover,
        _flag_forecast_weather_kn_warm_team_cold_late,
        _flag_forecast_weather_kn_temp_gap_cold_visitor,
        _flag_forecast_weather_kn_precip_high_total,
        _flag_forecast_weather_kn_temp_swing_prior_week,
        _flag_forecast_weather_kn_dome_cold_windy,
        _flag_interim_hc_active,
        _flag_interim_hc_first_game,
        _flag_interim_hc_home,
        _flag_interim_hc_fired_year_one,
    ):
        raise ExperimentRunnerError(
            f"flag_builder {spec.flag_builder!r} has no persistent per-entity trait to split-half; "
            "set reliability_check.method='not_applicable' with a reason"
        )

    features_file = features_path or (repo_root / "data" / "processed" / "game_features.parquet")
    if not features_file.is_file():
        raise ExperimentRunnerError(f"Feature table not found: {features_file}")
    features = pd.read_parquet(features_file)

    opener_population_note = ""
    if spec.grade == "opener":
        features, opener_population_note = _opener_graded_features(
            features, repo_root=repo_root, market_root=market_root
        )

    construct = builder.build(features, spec.seasons, spec.construct_params, repo_root)

    season_mask = construct.table["season"].between(spec.seasons[0], spec.seasons[1])
    table = construct.table.loc[season_mask]
    flag = construct.flag.loc[season_mask]
    eligible = None if construct.eligible is None else construct.eligible.loc[season_mask]

    if table.empty:
        raise ExperimentRunnerError(
            f"No rows remain for {spec.flag_builder!r} after restricting to seasons {spec.seasons}"
        )
    n_total = len(table)
    comparison = table if eligible is None else table.loc[eligible]
    comparison_flag = flag if eligible is None else flag.loc[eligible]
    n_flag = int(comparison_flag.sum())
    n_complement = len(comparison) - n_flag
    if n_flag == 0 or n_complement == 0:
        raise ExperimentRunnerError(
            f"{spec.flag_builder!r}: subset flag or its complement is empty after population/"
            "eligibility filtering -- cannot compare an empty group"
        )

    subset_cover = float(comparison.loc[comparison_flag, "team_covered"].mean())
    complement_cover = float(comparison.loc[~comparison_flag, "team_covered"].mean())
    raw_gap_fraction = subset_cover - complement_cover
    raw_gap_pct = construct.sign * raw_gap_fraction * 100.0
    # One-sided design (eligible=None, hc_year_one_fade/bias-battery precedent):
    # the complement is "everyone else", so the effect only "fires" on the flag
    # rows -- fraction_of_slate = n_flag / n_total. Two-sided/restricted design
    # (eligible provided, penalty_discipline precedent): BOTH compared arms are
    # exploitable (e.g. back Q1, fade Q4), so fraction_of_slate is the whole
    # compared subset's share of the full population, (n_flag + n_complement) /
    # n_total -- see SubsetBiasConstruct's docstring.
    slate_numerator = n_flag if eligible is None else len(comparison)
    fraction_of_slate = slate_numerator / n_total
    effect = scale_subset_effect(
        raw_gap_fraction, sign=construct.sign, fraction_of_slate=fraction_of_slate
    )

    def _block_result(block_kind: str) -> BlockIntervalResult:
        block_column = _BLOCK_COLUMNS[block_kind]
        block_count = int(comparison[block_column].nunique())
        verdict = guard_block_count(
            block_count,
            min_blocks=MIN_BLOCKS_FOR_INTERVAL,
            on_degenerate="warn",
            context=f"experiment_runner subset_bias {spec.name} ({block_kind}-blocked)",
        )
        draws = _block_bootstrap_subset_gap(
            comparison,
            flag=comparison_flag,
            value_col="team_covered",
            block_col=block_column,
            samples=spec.samples,
            seed=spec.seed,
        )
        return _interval_summary(
            draws,
            block_kind=block_kind,
            block_count=block_count,
            degenerate=verdict.degenerate,
            sign=construct.sign,
            fraction_of_slate=fraction_of_slate,
            confidence=DEFAULT_CONFIDENCE,
        )

    primary = _block_result(spec.block_primary)
    secondary = _block_result(spec.block_secondary) if spec.block_secondary is not None else None

    reliability = construct.reliability if spec.reliability_method == "split_half" else None
    reliability_pairs = (
        construct.reliability_pairs if spec.reliability_method == "split_half" else None
    )

    classification = classify_subset_bias_result(
        estimate=effect, lower=primary.lower, upper=primary.upper
    )

    population_note = " ".join(
        note for note in (opener_population_note, construct.population_note) if note
    )

    return SubsetBiasRunResult(
        spec=spec,
        n_total=n_total,
        n_flag=n_flag,
        n_complement=n_complement,
        fraction_of_slate=fraction_of_slate,
        raw_gap_pct=raw_gap_pct,
        effect=effect,
        primary=primary,
        secondary=secondary,
        reliability=reliability,
        reliability_pairs=reliability_pairs,
        reliability_note=construct.reliability_note,
        population_note=population_note,
        classification=classification,
        builder_description=builder.description,
        sign=construct.sign,
    )


# ---------------------------------------------------------------------------
# feature_arm: profile-vs-profile / ridge_alpha-vs-ridge_alpha
# ---------------------------------------------------------------------------
#
# Pattern named in the module docstring: two ``margin.fit_margin_model`` arms
# (baseline/candidate feature profile and/or ridge_alpha) walked forward with
# ``outcomes.walk_forward_outcomes`` (the ``nflverse_spread``/close grade --
# game_features.parquet's own spread_line across its full history, the same
# grade ``scripts/ridge_alpha_promotion_eval.py.run_nflverse_grade`` uses),
# paired by ``game_id``, scored with ``experiments.paired_feature_comparisons``
# -- the ALREADY-REVIEWED block-bootstrap engine this whole module exists to
# stop hand-transcribing output from, so this arm reuses it rather than
# re-deriving a second bootstrap.

#: 95% interval spans 2 * 1.96 standard errors -- ``weak_signals.WeakSignal.
#: resolved_standard_error``'s own fallback formula, reused here because
#: ``paired_feature_comparisons`` reports only the interval, not a raw SE.
_NORMAL_95_HALF_WIDTH = 1.959963984540054

_PAIRED_METRIC_NAMES = {
    "accuracy": "accuracy_improvement",
    "brier": "brier_improvement",
    "logloss": "log_loss_improvement",
}
#: The ONE 100x fraction-vs-points step, applied only to the accuracy metric
#: (``weak_signals.EFFECT_UNITS``: accuracy_points are percentage points,
#: e.g. record 1.10 not 0.011; brier/log_loss are recorded as the raw,
#: unscaled metric difference). Getting this backwards is exactly the bug
#: class ``scale_subset_effect`` above exists to prevent for subset_bias.
_PAIRED_METRIC_SCALE = {"accuracy": 100.0, "brier": 1.0, "logloss": 1.0}


def _block_result_from_paired_table(
    table: pd.DataFrame, *, metric_name: str, scale: float
) -> BlockIntervalResult:
    rows = table.loc[table["metric"] == metric_name]
    if rows.empty:
        raise ExperimentRunnerError(
            f"paired_feature_comparisons returned no rows for metric {metric_name!r}"
        )
    row = rows.iloc[0]
    lower = float(row["lower"]) * scale
    upper = float(row["upper"]) * scale
    return BlockIntervalResult(
        block_kind=str(row["block"]),
        block_count=int(row["blocks"]),
        degenerate=bool(row["degenerate_blocks"]),
        estimate=float(row["estimate"]) * scale,
        lower=lower,
        upper=upper,
        standard_error=(upper - lower) / (2.0 * _NORMAL_95_HALF_WIDTH),
        probability_positive=float(row["probability_positive"]),
        samples=int(row["samples"]),
    )


@dataclass(frozen=True)
class FeatureArmRunResult:
    spec: ExperimentSpec
    paired_games: int
    baseline_config: FeatureArmConfig
    candidate_config: FeatureArmConfig
    accuracy_primary: BlockIntervalResult
    accuracy_secondary: BlockIntervalResult | None
    brier_primary: BlockIntervalResult | None
    brier_secondary: BlockIntervalResult | None
    logloss_primary: BlockIntervalResult | None
    logloss_secondary: BlockIntervalResult | None
    classification: ClassificationResult


def run_feature_arm_experiment(
    spec: ExperimentSpec, *, repo_root: Path, features_path: Path | None = None
) -> FeatureArmRunResult:
    if spec.experiment_type != "feature_arm":
        raise ExperimentRunnerError(
            f"run_feature_arm_experiment called on a {spec.experiment_type!r} spec"
        )
    if spec.grade != "close":
        raise ExperimentRunnerError(
            "feature_arm currently only supports population.grade='close' -- "
            "outcomes.walk_forward_outcomes reads game_features.parquet's own spread_line (the "
            "close) across the full nflverse_spread history. Opener-graded feature_arm would need "
            "clv.opener_pick_evaluation's weekly-refit/opener-substitution machinery wired in "
            "separately (the pattern scripts/ridge_alpha_promotion_eval.py's run_opener_grade "
            "already demonstrates for one specific comparison); not implemented here."
        )
    if spec.league != "nfl":
        raise ExperimentRunnerError(
            "feature_arm is only wired for league='nfl' -- margin.MARGIN_FEATURE_PROFILES' "
            "registered profiles (including weak_stack) are NFL-only."
        )
    if spec.reliability_method == "split_half":
        raise ExperimentRunnerError(
            "feature_arm has no persistent per-entity trait to split-half (it compares two model "
            "arms' predictions, not a per-team/per-game trait); set "
            "reliability_check.method='not_applicable' with a reason"
        )
    assert spec.feature_arm_baseline is not None
    assert spec.feature_arm_candidate is not None

    features_file = features_path or (repo_root / "data" / "processed" / "game_features.parquet")
    if not features_file.is_file():
        raise ExperimentRunnerError(f"Feature table not found: {features_file}")
    features = pd.read_parquet(features_file)

    def _arm_predictions(label: str, config: FeatureArmConfig) -> pd.DataFrame:
        result = walk_forward_outcomes(
            features,
            start_season=spec.seasons[0],
            end_season=spec.seasons[1],
            regressor="ridge",
            feature_profile=config.feature_profile,  # type: ignore[arg-type]
            methods=("market_residual",),
            ridge_alpha=config.ridge_alpha,
            min_train_games=DEFAULT_MIN_TRAIN_GAMES,
        )
        predictions = result.predictions.copy()
        predictions["feature_set"] = label
        return predictions

    baseline_predictions = _arm_predictions("baseline", spec.feature_arm_baseline)
    candidate_predictions = _arm_predictions("candidate", spec.feature_arm_candidate)
    combined = pd.concat([baseline_predictions, candidate_predictions], ignore_index=True)

    def _paired(block_kind: str) -> pd.DataFrame:
        return paired_feature_comparisons(
            combined,
            baseline_feature_set="baseline",
            samples=spec.samples,
            block=block_kind,  # type: ignore[arg-type]
            seed=spec.seed,
            on_degenerate="warn",
        )

    primary_table = _paired(spec.block_primary)
    secondary_table = _paired(spec.block_secondary) if spec.block_secondary is not None else None
    paired_games = int(primary_table.iloc[0]["paired_games"]) if not primary_table.empty else 0

    accuracy_primary = _block_result_from_paired_table(
        primary_table,
        metric_name=_PAIRED_METRIC_NAMES["accuracy"],
        scale=_PAIRED_METRIC_SCALE["accuracy"],
    )
    accuracy_secondary = (
        None
        if secondary_table is None
        else _block_result_from_paired_table(
            secondary_table,
            metric_name=_PAIRED_METRIC_NAMES["accuracy"],
            scale=_PAIRED_METRIC_SCALE["accuracy"],
        )
    )

    brier_primary = brier_secondary = None
    logloss_primary = logloss_secondary = None
    if "brier" in spec.endpoint_secondary:
        brier_primary = _block_result_from_paired_table(
            primary_table,
            metric_name=_PAIRED_METRIC_NAMES["brier"],
            scale=_PAIRED_METRIC_SCALE["brier"],
        )
        if secondary_table is not None:
            brier_secondary = _block_result_from_paired_table(
                secondary_table,
                metric_name=_PAIRED_METRIC_NAMES["brier"],
                scale=_PAIRED_METRIC_SCALE["brier"],
            )
    if "logloss" in spec.endpoint_secondary:
        logloss_primary = _block_result_from_paired_table(
            primary_table,
            metric_name=_PAIRED_METRIC_NAMES["logloss"],
            scale=_PAIRED_METRIC_SCALE["logloss"],
        )
        if secondary_table is not None:
            logloss_secondary = _block_result_from_paired_table(
                secondary_table,
                metric_name=_PAIRED_METRIC_NAMES["logloss"],
                scale=_PAIRED_METRIC_SCALE["logloss"],
            )

    classification = classify_subset_bias_result(
        estimate=accuracy_primary.estimate,
        lower=accuracy_primary.lower,
        upper=accuracy_primary.upper,
    )

    return FeatureArmRunResult(
        spec=spec,
        paired_games=paired_games,
        baseline_config=spec.feature_arm_baseline,
        candidate_config=spec.feature_arm_candidate,
        accuracy_primary=accuracy_primary,
        accuracy_secondary=accuracy_secondary,
        brier_primary=brier_primary,
        brier_secondary=brier_secondary,
        logloss_primary=logloss_primary,
        logloss_secondary=logloss_secondary,
        classification=classification,
    )


# ---------------------------------------------------------------------------
# Evidence text, the WeakSignal payload, and the orchestrated run
# ---------------------------------------------------------------------------


def _format_block_line(result: BlockIntervalResult, *, label: str, seed: int) -> str:
    degenerate_note = (
        " [DEGENERATE: below the measured block-count floor]" if result.degenerate else ""
    )
    return (
        f"{label}-blocked ({result.block_count} blocks, {result.samples} resamples, "
        f"seed={seed}){degenerate_note}: "
        f"estimate={result.estimate:+.4f} pts, 95% [{result.lower:+.4f}, {result.upper:+.4f}] pts, "
        f"se={result.standard_error:.4f}, P+={result.probability_positive:.4f}"
    )


def build_evidence_text(
    result: SubsetBiasRunResult, *, spec_path: Path, artifact_dir: Path | None
) -> str:
    spec = result.spec
    lines = [
        f"subset_bias construct={spec.flag_builder!r} ({result.builder_description}); "
        f"sign={result.sign:+d} (positive flag=True favours the hypothesis).",
        f"Population: {spec.league} {spec.seasons[0]}-{spec.seasons[1]}, grade={spec.grade}. "
        f"{result.n_flag} flag / {result.n_complement} complement team-games "
        f"({result.n_flag + result.n_complement} compared) out of {result.n_total} classifiable "
        f"team-games (fraction_of_slate={result.fraction_of_slate:.4f}).",
        f"raw_gap={result.raw_gap_pct:+.4f} pts (unscaled); "
        f"full-slate-scaled effect={result.effect:+.4f} pts.",
        _format_block_line(result.primary, label=spec.block_primary.capitalize(), seed=spec.seed),
    ]
    if result.secondary is not None:
        secondary_label = (spec.block_secondary or "secondary").capitalize()
        lines.append(_format_block_line(result.secondary, label=secondary_label, seed=spec.seed))
    if spec.reliability_method == "split_half":
        lines.append(
            f"Split-half reliability: {result.reliability:.4f} ({result.reliability_pairs} pairs). "
            f"{result.reliability_note}"
        )
    else:
        lines.append(f"Reliability check: not_applicable -- {spec.reliability_reason}")
    if result.population_note:
        lines.append(result.population_note)
    lines.append(result.classification.note)
    lines.append(
        f"Generated by `nfl-ats experiment run {spec_path}` (spec name={spec.name!r}), "
        f"seed={spec.seed}." + ("" if artifact_dir is None else f" artifact={artifact_dir}")
    )
    return " ".join(lines)


def build_weak_signal(
    result: SubsetBiasRunResult,
    *,
    spec_path: Path,
    artifact_dir: Path | None,
    recorded_at: str | None = None,
) -> WeakSignal:
    spec = result.spec
    evidence = build_evidence_text(result, spec_path=spec_path, artifact_dir=artifact_dir)
    return WeakSignal(
        name=spec.name,
        recorded_at=recorded_at or datetime.now(UTC).date().isoformat(),
        description=spec.hypothesis,
        source=f"nfl-ats experiment run {spec_path}",
        effect=result.effect,
        effect_units="accuracy_points",
        classification=result.classification.classification,
        league=spec.league,
        seasons=spec.seasons,
        standard_error=result.primary.standard_error,
        interval=(result.primary.lower, result.primary.upper),
        probability_positive=result.primary.probability_positive,
        sample_games=result.n_flag + result.n_complement,
        sample_blocks=result.primary.block_count,
        classification_evidence=evidence,
        closing_ground=result.classification.closing_ground,
        reliability=result.reliability,
        notes=(
            f"seed={spec.seed}; samples={spec.samples}; flag_builder={spec.flag_builder}. "
            + (
                ""
                if result.secondary is None
                else _format_block_line(
                    result.secondary,
                    label=(spec.block_secondary or "?").capitalize(),
                    seed=spec.seed,
                )
                + ". "
            )
            + f"spec={spec_path}"
            + ("" if artifact_dir is None else f"; artifact={artifact_dir}")
        ),
    )


def build_feature_arm_evidence_text(
    result: FeatureArmRunResult, *, spec_path: Path, artifact_dir: Path | None
) -> str:
    spec = result.spec
    lines = [
        f"feature_arm baseline=(feature_profile={result.baseline_config.feature_profile!r}, "
        f"ridge_alpha={result.baseline_config.ridge_alpha:g}) vs "
        f"candidate=(feature_profile={result.candidate_config.feature_profile!r}, "
        f"ridge_alpha={result.candidate_config.ridge_alpha:g}); positive favours the candidate.",
        f"Population: {spec.league} {spec.seasons[0]}-{spec.seasons[1]}, grade={spec.grade} "
        f"(outcomes.walk_forward_outcomes, method=market_residual only). "
        f"{result.paired_games} paired games.",
        _format_block_line(
            result.accuracy_primary,
            label=f"Accuracy {spec.block_primary.capitalize()}",
            seed=spec.seed,
        ),
    ]
    secondary_label = (spec.block_secondary or "secondary").capitalize()
    if result.accuracy_secondary is not None:
        lines.append(
            _format_block_line(
                result.accuracy_secondary, label=f"Accuracy {secondary_label}", seed=spec.seed
            )
        )
    for metric_label, primary_result, secondary_result in (
        ("Brier", result.brier_primary, result.brier_secondary),
        ("LogLoss", result.logloss_primary, result.logloss_secondary),
    ):
        if primary_result is not None:
            lines.append(
                _format_block_line(
                    primary_result,
                    label=f"{metric_label} {spec.block_primary.capitalize()}",
                    seed=spec.seed,
                )
            )
        if secondary_result is not None:
            lines.append(
                _format_block_line(
                    secondary_result, label=f"{metric_label} {secondary_label}", seed=spec.seed
                )
            )
    lines.append(f"Reliability check: not_applicable -- {spec.reliability_reason}")
    lines.append(result.classification.note)
    lines.append(
        f"Generated by `nfl-ats experiment run {spec_path}` (spec name={spec.name!r}), "
        f"seed={spec.seed}." + ("" if artifact_dir is None else f" artifact={artifact_dir}")
    )
    return " ".join(lines)


def build_feature_arm_weak_signal(
    result: FeatureArmRunResult,
    *,
    spec_path: Path,
    artifact_dir: Path | None,
    recorded_at: str | None = None,
) -> WeakSignal:
    spec = result.spec
    evidence = build_feature_arm_evidence_text(
        result, spec_path=spec_path, artifact_dir=artifact_dir
    )
    secondary_label = (spec.block_secondary or "?").capitalize()
    secondary_note = (
        ""
        if result.accuracy_secondary is None
        else _format_block_line(
            result.accuracy_secondary, label=f"Accuracy {secondary_label}", seed=spec.seed
        )
        + ". "
    )
    return WeakSignal(
        name=spec.name,
        recorded_at=recorded_at or datetime.now(UTC).date().isoformat(),
        description=spec.hypothesis,
        source=f"nfl-ats experiment run {spec_path}",
        effect=result.accuracy_primary.estimate,
        effect_units="accuracy_points",
        classification=result.classification.classification,
        league=spec.league,
        seasons=spec.seasons,
        standard_error=result.accuracy_primary.standard_error,
        interval=(result.accuracy_primary.lower, result.accuracy_primary.upper),
        probability_positive=result.accuracy_primary.probability_positive,
        sample_games=result.paired_games,
        sample_blocks=result.accuracy_primary.block_count,
        classification_evidence=evidence,
        closing_ground=result.classification.closing_ground,
        reliability=None,
        notes=(
            f"seed={spec.seed}; samples={spec.samples}; feature_arm "
            f"baseline={result.baseline_config.feature_profile}"
            f"(alpha={result.baseline_config.ridge_alpha:g}) "
            f"candidate={result.candidate_config.feature_profile}"
            f"(alpha={result.candidate_config.ridge_alpha:g}). "
            + secondary_note
            + f"spec={spec_path}"
            + ("" if artifact_dir is None else f"; artifact={artifact_dir}")
        ),
    )


#: Either result type a run can produce; which one is entirely determined by
#: ``spec.experiment_type`` (never guessed downstream).
ExperimentRunResult = SubsetBiasRunResult | FeatureArmRunResult


def _weak_signal_for_result(
    result: ExperimentRunResult,
    *,
    spec_path: Path,
    artifact_dir: Path | None,
    recorded_at: str | None = None,
) -> WeakSignal:
    if isinstance(result, FeatureArmRunResult):
        return build_feature_arm_weak_signal(
            result, spec_path=spec_path, artifact_dir=artifact_dir, recorded_at=recorded_at
        )
    return build_weak_signal(
        result, spec_path=spec_path, artifact_dir=artifact_dir, recorded_at=recorded_at
    )


def run_experiment(
    spec: ExperimentSpec,
    *,
    repo_root: Path,
    features_path: Path | None = None,
    market_root: Path | None = None,
) -> ExperimentRunResult:
    """Dispatch on ``spec.experiment_type``."""

    if spec.experiment_type == "subset_bias":
        return run_subset_bias_experiment(
            spec, repo_root=repo_root, features_path=features_path, market_root=market_root
        )
    if spec.experiment_type == "feature_arm":
        return run_feature_arm_experiment(spec, repo_root=repo_root, features_path=features_path)
    raise ExperimentRunnerError(f"Unhandled experiment_type {spec.experiment_type!r}")


# ---------------------------------------------------------------------------
# Registry locking: the single-writer convention, enforced rather than hoped for
# ---------------------------------------------------------------------------


class _RegistryLock:
    """A cheap filesystem lock so two concurrent runner invocations cannot race
    a load-modify-save of ``weak_signals.json`` into a lost update.

    ``registry/weak_signals.json`` has a documented single-writer convention
    (``nfl_ats.weak_signals`` module docstring context, and every existing
    CLI writer) that this runner is now one more caller of; a lockfile makes
    that convention mechanically enforced for THIS writer rather than merely
    documented. Uses exclusive file creation (``O_CREAT | O_EXCL``), which is
    atomic on both POSIX and Windows -- no new dependency.
    """

    def __init__(self, registry_path: Path, *, timeout: float = 30.0, poll: float = 0.05) -> None:
        self._lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
        self._timeout = timeout
        self._poll = poll
        self._acquired = False

    def __enter__(self) -> _RegistryLock:
        deadline = time.monotonic() + self._timeout
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                self._acquired = True
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise ExperimentRunnerError(
                        f"Could not acquire the weak-signal registry lock at {self._lock_path} "
                        f"within {self._timeout}s; another `nfl-ats experiment run` (or "
                        "`weak-signals record`) invocation may be in progress. Concurrent runner "
                        "invocations must not overlap; if the lock is stale, remove it by hand."
                    ) from None
                time.sleep(self._poll)

    def __exit__(self, *exc_info: object) -> None:
        if self._acquired:
            with contextlib.suppress(FileNotFoundError):
                self._lock_path.unlink()


def record_experiment_signal(
    result: ExperimentRunResult,
    *,
    spec_path: Path,
    artifact_dir: Path | None,
    registry_path: Path | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    """Record the run's WeakSignal under the registry lock. Raises on validation failure."""

    path = registry_path or default_registry_path()
    signal = _weak_signal_for_result(result, spec_path=spec_path, artifact_dir=artifact_dir)
    with _RegistryLock(path):
        registry = load_registry(path)
        registry = record_signal(registry, signal, replace=replace)
        save_registry(registry, path)
    return {
        "registry": str(path),
        "recorded": signal.name,
        "classification": signal.classification,
        "closing_ground": signal.closing_ground,
        "effect": signal.effect,
        "effect_units": signal.effect_units,
        "favours_candidate": signal.favours_candidate,
    }


@dataclass(frozen=True)
class ExperimentRunOutcome:
    dry_run: bool
    preview: dict[str, Any]
    artifact_directory: str | None
    registry_record: dict[str, Any] | None


def run_experiment_cli(
    spec_path: Path,
    *,
    repo_root: Path,
    dry_run: bool,
    replace: bool = False,
    features_path: Path | None = None,
    market_root: Path | None = None,
    artifacts_root: Path | None = None,
    registry_root: Path | None = None,
    registry_path: Path | None = None,
    run_id_value: str | None = None,
) -> ExperimentRunOutcome:
    """The whole loop: load spec, run, classify, and (unless dry-run) stamp provenance and record.

    ``--dry-run`` performs every computation but writes nothing to disk --
    no artifact, no registry row -- and returns the record it WOULD have
    written for inspection.
    """

    spec = load_experiment_spec(spec_path)
    result = run_experiment(
        spec, repo_root=repo_root, features_path=features_path, market_root=market_root
    )

    preview = _weak_signal_for_result(result, spec_path=spec_path, artifact_dir=None)
    preview_payload = {
        "name": preview.name,
        "description": preview.description,
        "classification": preview.classification,
        "closing_ground": preview.closing_ground,
        "effect": preview.effect,
        "effect_units": preview.effect_units,
        "interval": list(preview.interval) if preview.interval else None,
        "probability_positive": preview.probability_positive,
        "standard_error": preview.standard_error,
        "sample_games": preview.sample_games,
        "sample_blocks": preview.sample_blocks,
        "reliability": preview.reliability,
        "league": preview.league,
        "seasons": list(preview.seasons),
        "classification_evidence": preview.classification_evidence,
    }
    if dry_run:
        return ExperimentRunOutcome(
            dry_run=True, preview=preview_payload, artifact_directory=None, registry_record=None
        )

    stamp = run_id_value or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_directory = (artifacts_root or (repo_root / "artifacts")) / "experiment_runner" / stamp
    features_file = features_path or (repo_root / "data" / "processed" / "game_features.parquet")
    configuration = experiment_spec_to_payload(spec)
    if isinstance(result, FeatureArmRunResult):
        result_metadata: dict[str, Any] = {
            "paired_games": result.paired_games,
            "baseline_config": {
                "feature_profile": result.baseline_config.feature_profile,
                "ridge_alpha": result.baseline_config.ridge_alpha,
            },
            "candidate_config": {
                "feature_profile": result.candidate_config.feature_profile,
                "ridge_alpha": result.candidate_config.ridge_alpha,
            },
            "accuracy_primary": vars(result.accuracy_primary),
            "accuracy_secondary": (
                None if result.accuracy_secondary is None else vars(result.accuracy_secondary)
            ),
            "brier_primary": None if result.brier_primary is None else vars(result.brier_primary),
            "brier_secondary": (
                None if result.brier_secondary is None else vars(result.brier_secondary)
            ),
            "logloss_primary": (
                None if result.logloss_primary is None else vars(result.logloss_primary)
            ),
            "logloss_secondary": (
                None if result.logloss_secondary is None else vars(result.logloss_secondary)
            ),
            "classification": result.classification.classification,
            "closing_ground": result.classification.closing_ground,
            "widening_factor": result.classification.widening_factor,
        }
    else:
        result_metadata = {
            "n_total": result.n_total,
            "n_flag": result.n_flag,
            "n_complement": result.n_complement,
            "fraction_of_slate": result.fraction_of_slate,
            "raw_gap_pct": result.raw_gap_pct,
            "effect": result.effect,
            "primary": vars(result.primary),
            "secondary": None if result.secondary is None else vars(result.secondary),
            "reliability": result.reliability,
            "reliability_pairs": result.reliability_pairs,
            "classification": result.classification.classification,
            "closing_ground": result.classification.closing_ground,
            "widening_factor": result.classification.widening_factor,
        }
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": "experiment run",
        "spec_path": str(spec_path),
        **configuration,
        "provenance": artifact_provenance(configuration, features_file, project_root=repo_root),
        "result": result_metadata,
    }
    write_experiment_artifact(
        output_directory,
        "metadata.json",
        metadata,
        command="experiment-run",
        metrics=metadata["result"],
        source=f"nfl-ats experiment run {spec_path}",
        weak_signal_name=spec.name,
        registry_root=registry_root,
    )

    registry_record = record_experiment_signal(
        result,
        spec_path=spec_path,
        artifact_dir=output_directory,
        registry_path=registry_path,
        replace=replace,
    )
    return ExperimentRunOutcome(
        dry_run=False,
        preview=preview_payload,
        artifact_directory=str(output_directory),
        registry_record=registry_record,
    )
