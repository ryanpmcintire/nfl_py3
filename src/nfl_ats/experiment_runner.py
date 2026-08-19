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

Only ``experiment_type: "subset_bias"`` is fully implemented (a pregame-safe
boolean flag vs. its complement, cover rate vs. the spread). ``"feature_arm"``
(profile-vs-profile via ``scripts/ridge_alpha_promotion_eval.py``'s
``evaluate_arm`` pattern) is validated at the schema level but raises a clear
:class:`ExperimentRunnerError` when run -- not implemented this pass; see that
script for the pattern a future implementation should wire up.

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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.estimation_variance import (
    MIN_BLOCKS_FOR_INTERVAL,
    guard_block_count,
)
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
_CONSTRUCT_FIELDS = frozenset({"flag_builder", "params"})
_ENDPOINTS_FIELDS = frozenset({"primary", "secondary"})
_BLOCKING_FIELDS = frozenset({"primary", "secondary"})
_RELIABILITY_FIELDS = frozenset({"method", "reason"})


@dataclass(frozen=True)
class ExperimentSpec:
    """A validated, immutable declarative experiment spec.

    Mirrors ``weak_signals.WeakSignal``'s validate-once-then-trust-the-
    dataclass shape: every field here has already survived
    :func:`experiment_spec_from_payload`'s strict checks, so downstream code
    never re-validates.
    """

    name: str
    hypothesis: str
    experiment_type: str
    league: str
    seasons: tuple[int, int]
    grade: str
    flag_builder: str
    construct_params: dict[str, Any]
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
    unknown_construct = sorted(set(construct).difference(_CONSTRUCT_FIELDS))
    _require(not unknown_construct, f"construct has unknown fields: {', '.join(unknown_construct)}")
    _require("flag_builder" in construct, "construct.flag_builder is required")
    flag_builder = str(construct["flag_builder"])
    construct_params = _require_dict(construct.get("params", {}), "construct.params")

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

    return {
        "name": spec.name,
        "hypothesis": spec.hypothesis,
        "experiment_type": spec.experiment_type,
        "population": {"league": spec.league, "seasons": list(spec.seasons), "grade": spec.grade},
        "construct": {"flag_builder": spec.flag_builder, "params": spec.construct_params},
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
    spec: ExperimentSpec, *, repo_root: Path, features_path: Path | None = None
) -> SubsetBiasRunResult:
    if spec.experiment_type != "subset_bias":
        raise ExperimentRunnerError(
            f"run_subset_bias_experiment called on a {spec.experiment_type!r} spec"
        )
    if spec.grade != "close":
        raise ExperimentRunnerError(
            "subset_bias currently only supports population.grade='close' (game_features.parquet's "
            "own spread_line, no snapshot backfill); 'opener' requires wiring an opener-snapshot "
            "population loader analogous to clv.opener_pick_evaluation, not yet implemented."
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
    if spec.reliability_method == "split_half" and builder.build in (
        _flag_home_underdog,
        _flag_large_favorite,
    ):
        raise ExperimentRunnerError(
            f"flag_builder {spec.flag_builder!r} has no persistent per-entity trait to split-half; "
            "set reliability_check.method='not_applicable' with a reason"
        )

    features_file = features_path or (repo_root / "data" / "processed" / "game_features.parquet")
    if not features_file.is_file():
        raise ExperimentRunnerError(f"Feature table not found: {features_file}")
    features = pd.read_parquet(features_file)

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
        population_note=construct.population_note,
        classification=classification,
        builder_description=builder.description,
        sign=construct.sign,
    )


def run_feature_arm_experiment(spec: ExperimentSpec, *, repo_root: Path) -> NoReturn:
    """Not implemented this pass. See ``scripts/ridge_alpha_promotion_eval.py``'s
    ``evaluate_arm`` for the pattern a future implementation should wire up:
    two ``margin.fit_margin_model`` arms (baseline/candidate feature profile
    or ridge_alpha) walked forward with ``outcomes.walk_forward_outcomes``,
    paired by ``game_id``, scored with
    ``experiments.paired_feature_comparisons``.
    """

    del repo_root
    raise ExperimentRunnerError(
        f"experiment_type='feature_arm' (spec {spec.name!r}) is validated but not implemented. "
        "See scripts/ridge_alpha_promotion_eval.py's evaluate_arm for the profile-vs-profile "
        "pattern a future implementation should wire up."
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


def run_experiment(
    spec: ExperimentSpec, *, repo_root: Path, features_path: Path | None = None
) -> SubsetBiasRunResult:
    """Dispatch on ``spec.experiment_type``. Only ``subset_bias`` returns a result today."""

    if spec.experiment_type == "subset_bias":
        return run_subset_bias_experiment(spec, repo_root=repo_root, features_path=features_path)
    if spec.experiment_type == "feature_arm":
        run_feature_arm_experiment(spec, repo_root=repo_root)
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
    result: SubsetBiasRunResult,
    *,
    spec_path: Path,
    artifact_dir: Path | None,
    registry_path: Path | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    """Record the run's WeakSignal under the registry lock. Raises on validation failure."""

    path = registry_path or default_registry_path()
    signal = build_weak_signal(result, spec_path=spec_path, artifact_dir=artifact_dir)
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
    result = run_experiment(spec, repo_root=repo_root, features_path=features_path)

    preview = build_weak_signal(result, spec_path=spec_path, artifact_dir=None)
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
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": "experiment run",
        "spec_path": str(spec_path),
        **configuration,
        "provenance": artifact_provenance(configuration, features_file, project_root=repo_root),
        "result": {
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
        },
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
