"""ENG-06: prospective evidence scorecards for the active model and every challenger.

A settled-week REPORT, not a verdict. It reads the append-only prospective
ledgers (``artifacts/clv_ledger/decisions.parquet`` for the active model,
``artifacts/prospective/challenger_decisions.parquet`` for every registered
challenger, ``artifacts/prospective/pick_revisions.parquet`` for the active
model's Tuesday-vs-refresh flips) and settles them the same way
``nfl-ats prospective-score`` does, reusing :func:`nfl_ats.prospective_scoring.
settle_prospective_picks` and :func:`nfl_ats.clv.week_blocked_bootstrap` for
every interval and ``probability_positive`` computed here rather than
reimplementing interval math. It never calls ``nfl-ats weak-signals record``
or ``nfl-ats rotation record-look`` and never writes to ``registry/`` --
verdicts flow through those two commands only; this module only reads and
reports.

Binding research invariant this module encodes (AGENTS.md, "An interval
crossing zero is NOT grounds for rejection"), pasted verbatim because every
row's ``classification`` field exists to enforce it mechanically rather than
rely on prose being read:

    An interval or CI that contains zero is NEVER grounds to reject, fail,
    or close an experiment. At this evaluator's ~2-point resolution,
    "contains zero" is the EXPECTED outcome for a real small signal. Only
    two grounds ever close a line of work: (1) refuted mechanism -- a
    RESOLVED wrong sign (whole interval on the wrong side of zero) or zero
    split-half reliability; (2) bounded by a positive control proven able
    to detect an effect that size. Everything else is
    ``unresolved_below_power``: report ``probability_positive``, never the
    binary "contains zero".

Concretely: :data:`nfl_ats.weak_signals.CLASSIFICATIONS` has three admissible
values. This module NEVER emits the two terminal ones (``refuted_mechanism``,
``bounded_by_control``) -- both require a registry-level judgement (a
predeclared closing ground, a split-half reliability check, or a positive
control) that a settled-week report does not perform. Every row this module
produces is therefore classified ``unresolved_below_power``
(:data:`nfl_ats.weak_signals.POOLABLE_CLASSIFICATION`) regardless of which
side of zero its interval falls on -- that is the correct, safe encoding of
the invariant above, not a placeholder. ``interval_crosses_zero`` is reported
alongside it, for transparency, but never changes the classification.

No "games needed" figure is ever computed or reported anywhere in this
module -- only ``settled_games`` counts of what has actually resolved.
Within-week game correlation is treated as zero throughout (the week is the
bootstrap block; no intra-week correlation is estimated or padded), per the
owner mandate recorded in AGENTS.md/team memory.

ENG-33 (ROADMAP.md Phase 13) extends every challenger row with two ADVISORY
fields, computed here but never acted on here: ``closing_ground_candidate``
(``None``, ``"wrong_sign_resolved"``, ``"no_split_half_reliability"``, or
``"positive_control_bound"`` -- the exact strings
:data:`nfl_ats.weak_signals.CLOSING_GROUNDS` admits) and its
``closing_ground_evidence``, plus ``next_admissible_action`` from the fixed
ENG-20 vocabulary (:mod:`nfl_ats.research_queue`), which never includes
"wait". Flagging a row as a *candidate* changes nothing about its
``classification``, which stays ``unresolved_below_power`` regardless -- the
candidate field exists so a reader does not have to recompute the interval
math by hand to see which rows are worth taking to
``nfl-ats weak-signals record`` / ``nfl-ats rotation record-look``, the only
two commands that may actually act on it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats import research_queue, rotation, weak_signals
from nfl_ats.clv import (
    live_close_reference,
    load_paper_decisions,
    pick_correct,
    week_blocked_bootstrap,
)
from nfl_ats.estimation_variance import BlockCountVerdict, guard_block_count
from nfl_ats.pick_refresh import load_pick_revisions
from nfl_ats.prospective_scoring import (
    CLOSE_GRADE,
    DECISION_GRADE,
    load_challenger_decisions,
    load_challenger_registry,
    prospective_accuracy,
    prospective_accuracy_metrics,
    settle_prospective_picks,
)
from nfl_ats.reporting import calibration_table
from nfl_ats.weak_signals import POOLABLE_CLASSIFICATION

SCORECARD_SCHEMA_VERSION = 1

#: Reused everywhere an interval needs a fixed bootstrap RNG; matches the
#: project's convention of a frozen, documented seed rather than the wall
#: clock (see e.g. ``nfl-ats prospective-score``'s ``--bootstrap-seed``).
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20260904

#: The calibration bin width already used everywhere else in the repo
#: (``nfl_ats.reporting.calibration_table``'s own default).
CALIBRATION_BINS = 10

ACTIVE_MODEL_ENTRANT_ID = "active_model"

_METRIC_FN = Callable[[pd.DataFrame], dict[str, float]]

# ENG-33: the two admissible closing grounds under "refuted mechanism",
# reused directly from nfl_ats.weak_signals.CLOSING_GROUNDS rather than
# retyped, so a future rename there cannot silently drift out of sync here.
WRONG_SIGN_RESOLVED, NO_SPLIT_HALF_RELIABILITY = weak_signals.CLOSING_GROUNDS["refuted_mechanism"]
(POSITIVE_CONTROL_BOUND,) = weak_signals.CLOSING_GROUNDS["bounded_by_control"]

#: The fixed ENG-20 next-admissible-action vocabulary, spelled the way
#: ENG-33's own definition of done names it. Four of the six strings are
#: identical to ``nfl_ats.research_queue.NEXT_ACTIONS``; the other two
#: (``test_on_production``, ``run_candidate_sized_positive_control``) are
#: this report's names for ``research_queue``'s
#: ``test_on_top_of_production`` / ``run_positive_control`` and are
#: translated by :data:`_ACTION_TRANSLATION` below. Never includes "wait".
NEXT_ADMISSIBLE_ACTIONS = (
    "run_unspent_window",
    "run_reused_window_with_discount",
    "test_on_production",
    "run_candidate_sized_positive_control",
    "record_pending_look",
    "closed",
)
_ACTION_TRANSLATION = {
    research_queue.ACTION_TEST_ON_TOP_OF_PRODUCTION: "test_on_production",
    research_queue.ACTION_RUN_POSITIVE_CONTROL: "run_candidate_sized_positive_control",
}

#: A challenger's own registered evidence cites the weak-signal(s) it is
#: built from as ``"registry/weak_signals.json:<name>"`` strings (see e.g.
#: ``hc_year_one_fade_overlay``'s ``evidence.registry_source`` in
#: ``artifacts/prospective/challengers.json``). Matched anywhere in the
#: evidence block, not just a fixed field name, because different
#: challengers spell the field differently (``registry_source`` as a bare
#: string, a list of strings, or nested one level under a named cell).
_REGISTRY_SOURCE_RE = re.compile(r"registry/weak_signals\.json:([A-Za-z0-9_]+)")

#: Fields a challenger's registered evidence uses to report a
#: candidate-favouring probability -- reporting ``probability_positive`` (as
#: opposed to a ``probability_negative``) IS the predeclared direction under
#: this repo's "positive favours candidate" convention
#: (``nfl_ats.weak_signals``'s ``EFFECT_UNITS`` block), so presence alone is
#: read as a predeclared positive sign, regardless of the number's value.
_SIGN_PROBABILITY_FIELDS: tuple[str, ...] = ("probability_positive", "source_probability_positive")

#: Fields carrying a signed accuracy-point effect a challenger declared for
#: itself at registration; the sign of a nonzero value is read as the
#: predeclared direction. Covers ``best_pick_big_spread_eligibility``, the
#: one live entry whose own declared effect is negative.
_SIGN_EFFECT_FIELDS: tuple[str, ...] = (
    "effect_accuracy_points",
    "source_effect_accuracy_points",
    "paired_delta_points",
)


def _scope(frame: pd.DataFrame, *, season: int, through_week: int | None) -> pd.DataFrame:
    """Rows for one season, optionally capped at ``through_week`` inclusive."""

    if frame.empty or "season" not in frame.columns:
        return frame
    scoped = frame.loc[frame["season"].astype(int).eq(season)]
    if through_week is not None and "week" in scoped.columns:
        scoped = scoped.loc[scoped["week"].astype(int).le(through_week)]
    return scoped.reset_index(drop=True)


def _bootstrap_interval(
    frame: pd.DataFrame,
    metric_fn: _METRIC_FN,
    *,
    samples: int,
    seed: int,
    context: str,
) -> dict[str, Any]:
    """Week-blocked interval + ``probability_positive`` for every metric ``metric_fn`` returns.

    Reuses :func:`nfl_ats.clv.week_blocked_bootstrap` for the resampling and
    interval math and :func:`nfl_ats.estimation_variance.guard_block_count`
    (``on_degenerate="warn"``) for the block-count degeneracy read -- this
    project's interval machinery, never reimplemented here. Per AGENTS.md, a
    degenerate or narrow block count is reported (``block_count_degenerate``,
    ``block_count_message``), never used to suppress or omit the interval.
    """

    if frame.empty:
        return {"interval_available": False, "interval_note": "no rows to bootstrap"}
    block_count = frame.groupby(["season", "week"], dropna=False).ngroups
    if block_count == 0:
        return {"interval_available": False, "interval_note": "no weeks to block on"}
    verdict: BlockCountVerdict = guard_block_count(
        block_count, on_degenerate="warn", context=context
    )
    table = week_blocked_bootstrap(frame, metric_fn, block="week", samples=samples, seed=seed)
    metrics: dict[str, Any] = {}
    for _, row in table.iterrows():
        metrics[str(row["metric"])] = {
            "estimate": float(row["estimate"]),
            "interval_lower": float(row["lower"]),
            "interval_upper": float(row["upper"]),
            "probability_positive": float(row["probability_positive"]),
        }
    return {
        "interval_available": True,
        "week_blocks": int(block_count),
        "block_count_degenerate": bool(verdict.degenerate),
        "block_count_message": verdict.message,
        "metrics": metrics,
    }


def _paired_delta_metric(name: str) -> _METRIC_FN:
    """A ``metric_fn`` reporting a paired accuracy delta in ACCURACY POINTS.

    Positive favours the candidate column, matching the repo-wide convention
    (``registry/weak_signals.json``'s ``accuracy_points`` scale: percentage
    points, e.g. 1.10 for a 1.1-point gap, not a fraction).
    """

    def metric_fn(frame: pd.DataFrame) -> dict[str, float]:
        delta = (
            pd.to_numeric(frame["candidate_correct"], errors="coerce")
            - pd.to_numeric(frame["baseline_correct"], errors="coerce")
        ) * 100.0
        return {name: float(delta.mean())}

    return metric_fn


def _paired_vs_active(
    candidate_settled: pd.DataFrame,
    active_settled: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired forced-pick accuracy delta on games BOTH entrants recorded and settled."""

    if candidate_settled.empty or active_settled.empty:
        return {"shared_settled_games": 0, "note": "no shared settled games yet"}
    left = candidate_settled[["game_id", "season", "week", f"correct_at_{DECISION_GRADE}"]].rename(
        columns={f"correct_at_{DECISION_GRADE}": "candidate_correct"}
    )
    right = active_settled[["game_id", f"correct_at_{DECISION_GRADE}"]].rename(
        columns={f"correct_at_{DECISION_GRADE}": "baseline_correct"}
    )
    merged = left.merge(right, on="game_id", how="inner")
    resolved = merged.dropna(subset=["candidate_correct", "baseline_correct"])
    result: dict[str, Any] = {"shared_settled_games": len(resolved)}
    if resolved.empty:
        result["note"] = "no shared settled games yet"
        return result
    result.update(
        _bootstrap_interval(
            resolved,
            _paired_delta_metric("paired_delta_accuracy_points"),
            samples=samples,
            seed=seed,
            context="paired_vs_active",
        )
    )
    return result


def _overlay_marginal(
    candidate_settled: pd.DataFrame,
    active_settled: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired accuracy delta restricted to games where this entrant's pick differs.

    "The games it fired on": for a pick-level overlay/challenger this is
    exactly the flip set; for a full alternative model it is the disagreement
    set docs/prospective_evidence.md already frames the same way ("the two
    arms disagree on 3 of 16 games ... which is where all of the paired
    evidence will come from"). Both read the same paired-delta machinery as
    :func:`_paired_vs_active`, just pre-filtered to the disagreement rows.
    """

    if candidate_settled.empty or active_settled.empty:
        return {"disagreement_games": 0, "note": "no shared settled games yet"}
    left = candidate_settled[
        ["game_id", "season", "week", "pick_side", f"correct_at_{DECISION_GRADE}"]
    ].rename(
        columns={
            "pick_side": "candidate_pick_side",
            f"correct_at_{DECISION_GRADE}": "candidate_correct",
        }
    )
    right = active_settled[["game_id", "pick_side", f"correct_at_{DECISION_GRADE}"]].rename(
        columns={
            "pick_side": "active_pick_side",
            f"correct_at_{DECISION_GRADE}": "baseline_correct",
        }
    )
    merged = left.merge(right, on="game_id", how="inner")
    resolved = merged.dropna(subset=["candidate_correct", "baseline_correct"])
    disagreement = resolved.loc[
        resolved["candidate_pick_side"].astype(str).ne(resolved["active_pick_side"].astype(str))
    ]
    result: dict[str, Any] = {
        "shared_settled_games": len(resolved),
        "disagreement_games": len(disagreement),
    }
    if disagreement.empty:
        result["note"] = (
            "no settled games yet where this entrant's pick differs from the active "
            "model's chain pick"
        )
        return result
    result.update(
        _bootstrap_interval(
            disagreement,
            _paired_delta_metric("marginal_paired_delta_accuracy_points"),
            samples=samples,
            seed=seed,
            context="overlay_marginal",
        )
    )
    return result


def _refresh_effect(
    artifacts_root: Path,
    outcomes: pd.DataFrame,
    *,
    season: int,
    through_week: int | None,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Tuesday pick vs. final refresh pick: flip count and paired delta.

    Reads ``artifacts/prospective/pick_revisions.parquet`` via
    :func:`nfl_ats.pick_refresh.load_pick_revisions` -- the ledger
    ``refresh-picks --record-decisions`` writes whenever a late-week refresh
    pass changes (or reconsiders and keeps) the active model's chain pick.
    Both the Tuesday and the refreshed pick are graded against the SAME
    frozen ``decision_home_spread`` (a refresh changes the pick side, not the
    anchor line), using :func:`nfl_ats.clv.pick_correct` for both, exactly as
    the ledger's own ``describe_week_revisions`` documents.

    This is a property of the active model's played chain, not of any one
    challenger -- a handful of challengers have their OWN dedicated refresh
    ledgers (``docs/prospective_evidence.md``, ``scripts/lockday_verify.py``'s
    ``DEDICATED_LEDGERS``); those are out of scope here and are not silently
    folded in.
    """

    revisions = load_pick_revisions(artifacts_root)
    if revisions.empty:
        return {
            "available": False,
            "note": "no rows in artifacts/prospective/pick_revisions.parquet yet",
        }
    scoped = _scope(revisions, season=season, through_week=through_week)
    if scoped.empty:
        return {
            "available": False,
            "note": f"no pick-revision rows recorded for season {season} in scope",
        }
    latest = (
        scoped.sort_values("revision_recorded_at_utc")
        .groupby("game_id", as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    previous = latest["previous_pick_side"].astype(str)
    new = latest["new_pick_side"].astype(str)
    flips = int(previous.ne(new).sum())
    result: dict[str, Any] = {
        "available": True,
        "revised_games": len(latest),
        "flips": flips,
        "kept": len(latest) - flips,
    }
    merged = latest.merge(outcomes, on="game_id", how="left")
    result_values = pd.to_numeric(merged["result"], errors="coerce")
    line = pd.to_numeric(merged["decision_home_spread"], errors="coerce")
    margin = result_values - line
    settled_mask = margin.notna() & margin.ne(0.0)
    settled = merged.loc[settled_mask].copy()
    result["settled_games"] = len(settled)
    if settled.empty:
        result["note"] = "no revised games have settled yet"
        return result
    settled_margin = margin.loc[settled_mask]
    settled["candidate_correct"] = pick_correct(
        settled["new_pick_side"].astype(str).eq("HOME"), settled_margin
    ).to_numpy()
    settled["baseline_correct"] = pick_correct(
        settled["previous_pick_side"].astype(str).eq("HOME"), settled_margin
    ).to_numpy()
    resolved = settled.dropna(subset=["candidate_correct", "baseline_correct"])
    if resolved.empty:
        result["note"] = "every settled revision was a push at one side or the other"
        return result
    result.update(
        _bootstrap_interval(
            resolved,
            _paired_delta_metric("refresh_paired_delta_accuracy_points"),
            samples=samples,
            seed=seed,
            context="refresh_effect",
        )
    )
    return result


def _read_card_probabilities(path: Path) -> pd.DataFrame:
    """``game_id`` -> ``home_cover_probability`` from one ``margin-predict`` card.

    Fails open: a missing or malformed card contributes nothing rather than
    raising -- a settled-week report must not crash because an old weekly
    forecast directory was cleaned up.
    """

    empty = pd.DataFrame(columns=["game_id", "home_cover_probability"])
    if not path.is_file():
        return empty
    try:
        card = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return empty
    if not {"game_id", "home_cover_probability"}.issubset(card.columns):
        return empty
    card = card.loc[:, ["game_id", "home_cover_probability"]].copy()
    card["game_id"] = card["game_id"].astype(str)
    card["home_cover_probability"] = pd.to_numeric(card["home_cover_probability"], errors="coerce")
    return card.drop_duplicates("game_id")


def _attach_home_cover_probability(
    settled: pd.DataFrame, artifacts_root: Path, artifact_column: str
) -> pd.DataFrame:
    """Join each settled row to its own card's recorded ``home_cover_probability``.

    ``artifact_column`` is ``forecast_artifact`` (active model rows, already a
    path relative to ``artifacts_root``) or ``source_artifact`` (challenger
    rows, a bare directory name under ``artifacts/margin_predictions``).
    """

    if settled.empty or artifact_column not in settled.columns:
        return settled.assign(home_cover_probability=np.nan)
    frames: list[pd.DataFrame] = []
    for artifact in sorted(settled[artifact_column].dropna().astype(str).unique()):
        relative = (
            artifact if artifact_column == "forecast_artifact" else f"margin_predictions/{artifact}"
        )
        probabilities = _read_card_probabilities(artifacts_root / relative / "recommendations.csv")
        if probabilities.empty:
            continue
        frames.append(probabilities.assign(**{artifact_column: artifact}))
    if not frames:
        return settled.assign(home_cover_probability=np.nan)
    lookup = pd.concat(frames, ignore_index=True)
    return settled.merge(lookup, on=[artifact_column, "game_id"], how="left")


def _calibration(
    settled: pd.DataFrame, artifacts_root: Path, artifact_column: str
) -> dict[str, Any]:
    """Brier score + reliability bins (``nfl_ats.reporting.calibration_table``, bins=10).

    ``home_cover`` (the actual outcome) is derived from the already-computed
    ``ats_margin_at_decision_line`` column; ``home_cover_probability`` is read
    back from each row's own recorded forecast card (see
    :func:`_attach_home_cover_probability`). Rows whose card cannot be found
    are excluded from calibration, and that exclusion is reported explicitly
    rather than silently shrinking the sample.
    """

    margin_column = f"ats_margin_at_{DECISION_GRADE}"
    if settled.empty or margin_column not in settled.columns:
        return {"available": False, "note": "no settled games yet"}
    margin = pd.to_numeric(settled[margin_column], errors="coerce")
    settled_rows = settled.loc[margin.notna() & margin.ne(0.0)].copy()
    if settled_rows.empty:
        return {"available": False, "note": "no non-push settled games yet"}
    settled_margin = margin.loc[settled_rows.index]
    settled_rows["home_cover"] = np.where(settled_margin.gt(0.0), 1.0, 0.0)
    enriched = _attach_home_cover_probability(settled_rows, artifacts_root, artifact_column)
    with_probability = enriched.dropna(subset=["home_cover_probability"])
    result: dict[str, Any] = {
        "settled_games": len(settled_rows),
        "games_with_recorded_probability": len(with_probability),
    }
    if with_probability.empty:
        result["available"] = False
        result["note"] = (
            f"none of {len(settled_rows)} settled game(s) had a readable recorded "
            "probability card (recommendations.csv not found alongside the ledger row)"
        )
        return result
    result["available"] = True
    actual = with_probability["home_cover"].to_numpy(dtype=float)
    probability = with_probability["home_cover_probability"].to_numpy(dtype=float)
    result["brier_score"] = float(np.mean(np.square(probability - actual)))
    bins = calibration_table(with_probability, bins=CALIBRATION_BINS)
    result["bins"] = bins.to_dict(orient="records")
    if len(with_probability) < len(settled_rows):
        result["note"] = (
            f"{len(settled_rows) - len(with_probability)} of {len(settled_rows)} settled "
            "game(s) had no readable recorded probability card and were excluded"
        )
    return result


def _classification(paired_vs_active: dict[str, Any]) -> tuple[str, bool | None]:
    """The registry-admissible classification for this row, and whether its
    paired-delta interval crosses zero (reported, never used to change the
    classification -- see the module docstring).
    """

    metrics = paired_vs_active.get("metrics") if isinstance(paired_vs_active, dict) else None
    if not metrics:
        return POOLABLE_CLASSIFICATION, None
    delta = metrics.get("paired_delta_accuracy_points")
    if not isinstance(delta, dict):
        return POOLABLE_CLASSIFICATION, None
    crosses_zero = bool(delta["interval_lower"] <= 0.0 <= delta["interval_upper"])
    return POOLABLE_CLASSIFICATION, crosses_zero


# ---------------------------------------------------------------------------
# ENG-33: closing-ground CANDIDATE detection and next_admissible_action.
#
# Everything below is advisory only. It never changes `classification`
# (always `unresolved_below_power`, computed above) and never writes to
# either registry -- the taxonomy pasted in the module docstring applies in
# full: an interval containing zero is never grounds to reject, and only a
# RESOLVED wrong sign, zero split-half reliability, or a proven positive
# control may ever close a line of work. This section only flags which rows
# LOOK like candidates for those two admissible grounds so a reader does not
# have to recompute the interval math by hand before deciding whether to run
# `nfl-ats weak-signals record` / `nfl-ats rotation record-look`.
# ---------------------------------------------------------------------------


def _predeclared_sign(entry: dict[str, Any] | None) -> tuple[int | None, str | None]:
    """The direction, if any, a challenger's OWN registered evidence declares.

    Returns ``(sign, reason)`` where ``sign`` is ``+1``, ``-1``, or ``None``;
    ``reason`` is only set (to ``"no_predeclared_sign"``) when ``sign`` is
    ``None``. ``entry`` is the raw ``challengers.json`` dict for this
    entrant, or ``None`` for the active model row, which is not itself a
    registered challenger and therefore never carries a predeclared sign.
    Most live entries declare no single top-level numeric direction at all
    (their ``evidence`` is prose/classification only) and correctly resolve
    to ``no_predeclared_sign`` here -- that is the literal, honest answer for
    a row that never staked out a direction before seeing data, not a gap in
    this function.
    """

    if entry is None:
        return None, "no_predeclared_sign"
    evidence = entry.get("evidence")
    if not isinstance(evidence, dict):
        return None, "no_predeclared_sign"
    for field in _SIGN_PROBABILITY_FIELDS:
        if isinstance(evidence.get(field), int | float):
            return 1, None
    for field in _SIGN_EFFECT_FIELDS:
        value = evidence.get(field)
        if isinstance(value, int | float) and value != 0:
            return (1 if value > 0 else -1), None
    return None, "no_predeclared_sign"


def _referenced_signal_names(entry: dict[str, Any]) -> tuple[str, ...]:
    """Every ``registry/weak_signals.json:<name>`` reference in ``entry["evidence"]``."""

    texts: list[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                _walk(nested)
        elif isinstance(value, list | tuple):
            for nested in value:
                _walk(nested)

    _walk(entry.get("evidence"))
    seen: set[str] = set()
    ordered: list[str] = []
    for text in texts:
        for name in _REGISTRY_SOURCE_RE.findall(text):
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return tuple(ordered)


def _weekly_paired_deltas(
    candidate_settled: pd.DataFrame, active_settled: pd.DataFrame
) -> pd.Series:
    """One paired accuracy-point delta per SETTLED WEEK (the split-half unit below).

    Same paired merge :func:`_paired_vs_active` performs, aggregated to one
    row per week instead of one row per game, because the split-half check
    below treats the week as the independent unit (AGENTS.md / team memory:
    within-week game correlation is treated as zero throughout this module
    and is never estimated or padded).
    """

    if candidate_settled.empty or active_settled.empty:
        return pd.Series(dtype=float)
    left = candidate_settled[["game_id", "week", f"correct_at_{DECISION_GRADE}"]].rename(
        columns={f"correct_at_{DECISION_GRADE}": "candidate_correct"}
    )
    right = active_settled[["game_id", f"correct_at_{DECISION_GRADE}"]].rename(
        columns={f"correct_at_{DECISION_GRADE}": "baseline_correct"}
    )
    merged = left.merge(right, on="game_id", how="inner")
    resolved = merged.dropna(subset=["candidate_correct", "baseline_correct"])
    if resolved.empty:
        return pd.Series(dtype=float)
    delta = (
        pd.to_numeric(resolved["candidate_correct"], errors="coerce")
        - pd.to_numeric(resolved["baseline_correct"], errors="coerce")
    ) * 100.0
    return delta.groupby(resolved["week"]).mean().sort_index()


def _split_half_reliability(per_week: pd.Series, *, samples: int, seed: int) -> dict[str, Any]:
    """Odd/even-week split-half reliability of the per-week paired-delta series.

    Weeks are the independent unit here (owner mandate, restated above), so
    the split-half unit is the WEEK itself, not the game: sorted settled
    weeks are paired consecutively (week[0] with week[1], week[2] with
    week[3], ...) and this is a minimal Pearson correlation between the two
    positions across pairs, percentile-bootstrapped over pairs for an
    interval. There is no existing week-level split-half helper in the repo
    to reuse -- the two that exist,
    :func:`nfl_ats.durability_prior.split_half_reliability` and
    :func:`nfl_ats.cfb_qb_dependence.split_half_reliability`, both key their
    two halves on a PLAYER or TEAM-SEASON as the repeated-measure unit, which
    this per-week accuracy-delta series does not have -- so this is the
    "else a minimal Pearson" fallback ENG-33 anticipates.
    """

    note = (
        "weeks are the independent unit (within-week game correlation is treated as "
        "zero throughout this module and is never estimated or padded); consecutive "
        "settled weeks are paired and this is the Pearson correlation between the two "
        "positions across pairs"
    )
    weeks_used = [int(week) for week in per_week.index]
    values = per_week.to_numpy(dtype=float)
    pair_count = len(values) // 2
    if pair_count < 3:
        return {
            "available": False,
            "week_pairs": pair_count,
            "weeks_used": weeks_used,
            "note": f"fewer than 3 week-pairs settled ({pair_count}); {note}",
        }
    first = values[0 : 2 * pair_count : 2]
    second = values[1 : 2 * pair_count : 2]
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return {
            "available": False,
            "week_pairs": pair_count,
            "weeks_used": weeks_used,
            "note": f"one side of the split has zero variance, correlation is undefined; {note}",
        }
    reliability = float(np.corrcoef(first, second)[0, 1])
    rng = np.random.default_rng(seed)
    boots = np.empty(samples, dtype=float)
    for draw in range(samples):
        selected = rng.integers(0, pair_count, size=pair_count)
        a, b = first[selected], second[selected]
        boots[draw] = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
    valid = boots[~np.isnan(boots)]
    if valid.size == 0:
        return {
            "available": False,
            "week_pairs": pair_count,
            "weeks_used": weeks_used,
            "reliability": reliability,
            "note": f"bootstrap resamples were all degenerate (zero variance); {note}",
        }
    return {
        "available": True,
        "week_pairs": pair_count,
        "weeks_used": weeks_used,
        "reliability": reliability,
        "interval_lower": float(np.percentile(valid, 2.5)),
        "interval_upper": float(np.percentile(valid, 97.5)),
        "note": note,
    }


def _closing_ground_candidate(
    *,
    predeclared_sign: int | None,
    predeclared_sign_reason: str | None,
    paired_metric: dict[str, Any] | None,
    split_half: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    """The advisory ``(closing_ground_candidate, closing_ground_evidence)`` pair.

    ``wrong_sign_resolved`` only when the WHOLE paired-delta interval sits on
    the side opposite a predeclared sign; ``no_split_half_reliability`` only
    when the split-half reliability's own bootstrap interval upper bound is
    at or below zero; ``positive_control_bound`` is always ``None`` here with
    reason ``no_positive_control_in_report`` -- this scorecard runs no
    positive control. Never returns a TERMINAL classification; the caller
    always keeps ``classification == unresolved_below_power`` regardless of
    what this returns.
    """

    evidence: dict[str, Any] = {
        "predeclared_sign": (
            "positive" if predeclared_sign == 1 else "negative" if predeclared_sign == -1 else None
        ),
        "predeclared_sign_reason": predeclared_sign_reason,
        "split_half_reliability": split_half,
        "positive_control_reason": "no_positive_control_in_report",
    }
    candidate: str | None = None
    if paired_metric is not None:
        lower = float(paired_metric["interval_lower"])
        upper = float(paired_metric["interval_upper"])
        evidence["paired_delta_interval"] = [lower, upper]
        if (predeclared_sign == 1 and upper < 0.0) or (predeclared_sign == -1 and lower > 0.0):
            candidate = WRONG_SIGN_RESOLVED
    else:
        evidence["paired_delta_interval"] = None
    if (
        candidate is None
        and split_half.get("available")
        and float(split_half["interval_upper"]) <= 0.0
    ):
        candidate = NO_SPLIT_HALF_RELIABILITY
    return candidate, evidence


def _closed_via_weak_signal(
    matched_signals: Sequence[weak_signals.WeakSignal],
) -> tuple[str, str] | None:
    """``("closed", detail)`` iff one of ``matched_signals`` is already an
    admissibly-closed terminal verdict -- never guessed, always read from a
    registry entry that ``weak_signals.signal_from_payload`` already
    validated at load time (``validate_closure``), so a "closed" this
    function returns can never be an inadmissible one.
    """

    for signal in matched_signals:
        if (
            signal.classification in weak_signals.TERMINAL_CLASSIFICATIONS
            and signal.closing_ground is not None
        ):
            return (
                "closed",
                f"{signal.name} closed on {signal.classification} ({signal.closing_ground})",
            )
    return None


def _next_admissible_action(
    entry: dict[str, Any] | None,
    *,
    has_settled_shared_data: bool,
    weak_signal_registry: weak_signals.Registry,
    rotation_registry: rotation.Registry,
) -> tuple[str, str]:
    """``(action, detail)``, ``action`` always one of :data:`NEXT_ADMISSIBLE_ACTIONS`.

    1. ``entry is None`` (the active model row: not a registered challenger)
       -> ``record_pending_look`` -- it is already production; the admissible
       action is to keep recording its weekly settlements.
    2. One of this challenger's OWN cited weak signals already carries an
       admissible terminal closure -> ``closed`` (see
       :func:`_closed_via_weak_signal`).
    3. One of those signals' inferred family (:func:`weak_signals.signal_family`)
       is a DECLARED rotation family -> delegate to
       :func:`nfl_ats.research_queue.next_admissible_action`, the existing,
       tested per-family decision procedure, translating its two
       differently-spelled actions via :data:`_ACTION_TRANSLATION`.
    4. No matching rotation family, but this row already has settled shared
       prospective evidence -> ``record_pending_look`` (AGENTS.md: recording
       is the default action for a category-3 result, not an optional
       extra).
    5. No matching rotation family and no settled evidence yet ->
       ``test_on_production`` -- the admissible next step is to keep running
       it as a live prospective challenger against the production chain,
       which is exactly what its ``challengers.json`` registration already
       sets up.
    """

    if entry is None:
        return (
            "record_pending_look",
            "active model: not a registered challenger; keep recording weekly "
            "settlements (refresh_effect and the season interval) as the evidence",
        )
    names = _referenced_signal_names(entry)
    matched_signals = [
        weak_signal_registry.signals[name] for name in names if name in weak_signal_registry.signals
    ]
    closed = _closed_via_weak_signal(matched_signals)
    if closed is not None:
        return closed

    family_name: str | None = None
    for signal in matched_signals:
        candidate_family = weak_signals.signal_family(signal)
        if candidate_family in rotation_registry.families:
            family_name = candidate_family
            break

    if family_name is not None:
        action, detail = research_queue.next_admissible_action(
            rotation_registry,
            family_name=family_name,
            grade_guess="close",
            weak_signal=matched_signals[0] if matched_signals else None,
        )
        return _ACTION_TRANSLATION.get(action, action), detail

    if has_settled_shared_data:
        return (
            "record_pending_look",
            "settled prospective evidence exists for this challenger; AGENTS.md: "
            "recording is the default action for a category-3 result, via "
            "`nfl-ats weak-signals record` / `rotation record-look`",
        )
    return (
        "test_on_production",
        "no declared rotation family and no settled prospective evidence yet; the "
        "admissible next step is to keep running this challenger against the live "
        "production chain, as its challengers.json registration already does",
    )


def _load_registries_for_report(
    registry_root: Path | None,
) -> tuple[weak_signals.Registry, rotation.Registry]:
    """Read-only load of both registries this module consults for advisories.

    Never writes either registry (module contract). A missing rotation
    ledger fails open to an EMPTY registry rather than raising -- matching
    this module's existing fail-open convention for a missing/partial file
    elsewhere (e.g. :func:`_read_card_probabilities`, and
    ``build_season_scorecards``'s own ``FileNotFoundError`` handling for a
    missing challenger registry).
    """

    root = (
        registry_root
        if registry_root is not None
        else Path(os.environ.get("NFL_ATS_REGISTRY_DIR", "registry"))
    )
    weak_signal_registry = weak_signals.load_registry(
        root / weak_signals.WEAK_SIGNAL_REGISTRY_FILENAME
    )
    try:
        rotation_registry = rotation.load_registry(root / rotation.ROTATION_REGISTRY_FILENAME)
    except rotation.RegistryError:
        rotation_registry = rotation.Registry(
            version=rotation.ROTATION_REGISTRY_VERSION, notes=(), families={}
        )
    return weak_signal_registry, rotation_registry


def _coverage(entrant_scope: pd.DataFrame, active_scope: pd.DataFrame) -> dict[str, Any]:
    active_games = set(active_scope["game_id"].astype(str)) if not active_scope.empty else set()
    entrant_games = set(entrant_scope["game_id"].astype(str)) if not entrant_scope.empty else set()
    games_on_card = len(active_games)
    covered = len(entrant_games & active_games) if active_games else len(entrant_games)
    return {
        "games_recorded": len(entrant_games),
        "games_on_card": games_on_card,
        "games_covered_of_card": covered,
        "coverage_ratio": (covered / games_on_card) if games_on_card else None,
    }


def _accuracy_block(settled: pd.DataFrame, *, samples: int, seed: int) -> dict[str, Any]:
    """Point accuracy at both grades plus the entrant's own week-blocked interval.

    Reuses :func:`nfl_ats.prospective_scoring.prospective_accuracy` for the
    point summary and :func:`nfl_ats.prospective_scoring.
    prospective_accuracy_metrics` as the ``metric_fn`` for
    :func:`nfl_ats.clv.week_blocked_bootstrap`, the same pairing
    ``nfl-ats prospective-score`` itself uses.
    """

    summary = prospective_accuracy(settled)
    decision = summary["forced_picks"][DECISION_GRADE]
    close = summary["forced_picks"][CLOSE_GRADE]
    block: dict[str, Any] = {
        "settled_games": decision["games"],
        "pushes": decision["pushes"],
        "pending": decision["pending"],
        "accuracy_decision_line": decision["accuracy"] if decision["games"] else None,
        "accuracy_close_line": close["accuracy"] if close["games"] else None,
    }
    resolved = (
        settled.dropna(subset=[f"correct_at_{DECISION_GRADE}"]) if not settled.empty else settled
    )
    if resolved.empty:
        block["interval"] = {"interval_available": False, "interval_note": "no settled games yet"}
    else:
        block["interval"] = _bootstrap_interval(
            resolved,
            prospective_accuracy_metrics,
            samples=samples,
            seed=seed,
            context="own_accuracy",
        )
    return block


def _entrant_row(
    entrant_id: str,
    *,
    entrant_kind: str,
    challenger_status: str | None,
    registered_evidence: dict[str, Any] | None,
    challenger_entry: dict[str, Any] | None,
    weak_signal_registry: weak_signals.Registry,
    rotation_registry: rotation.Registry,
    entrant_scope: pd.DataFrame,
    active_scope: pd.DataFrame,
    entrant_settled: pd.DataFrame,
    active_settled: pd.DataFrame,
    artifacts_root: Path,
    artifact_column: str,
    is_active_model: bool,
    outcomes: pd.DataFrame,
    season: int,
    through_week: int | None,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "entrant_id": entrant_id,
        "entrant_kind": entrant_kind,
        "challenger_status": challenger_status,
        "registered_evidence": registered_evidence,
        "season": season,
        "through_week": through_week,
    }
    row.update(_coverage(entrant_scope, active_scope))
    row.update(_accuracy_block(entrant_settled, samples=samples, seed=seed))
    row["calibration"] = _calibration(entrant_settled, artifacts_root, artifact_column)

    if is_active_model:
        row["paired_vs_active"] = None
        row["overlay_marginal"] = None
        row["classification"] = POOLABLE_CLASSIFICATION
        row["interval_crosses_zero"] = None
        row["refresh_effect"] = _refresh_effect(
            artifacts_root,
            outcomes,
            season=season,
            through_week=through_week,
            samples=samples,
            seed=seed,
        )
        # ENG-33: the active model is not a registered challenger and has no
        # paired comparison against itself, so it never carries a closing-
        # ground candidate -- only next_admissible_action ("keep recording").
        row["closing_ground_candidate"] = None
        row["closing_ground_evidence"] = {
            "reason": "active_model_is_not_a_challenger_and_has_no_paired_comparison"
        }
        action, action_detail = _next_admissible_action(
            None,
            has_settled_shared_data=False,
            weak_signal_registry=weak_signal_registry,
            rotation_registry=rotation_registry,
        )
        row["next_admissible_action"] = action
        row["next_admissible_action_detail"] = action_detail
    else:
        paired = _paired_vs_active(entrant_settled, active_settled, samples=samples, seed=seed)
        row["paired_vs_active"] = paired
        row["overlay_marginal"] = _overlay_marginal(
            entrant_settled, active_settled, samples=samples, seed=seed
        )
        classification, crosses_zero = _classification(paired)
        row["classification"] = classification
        row["interval_crosses_zero"] = crosses_zero
        row["refresh_effect"] = {
            "available": False,
            "note": (
                "refresh effect is tracked for the active model's played chain only "
                "(artifacts/prospective/pick_revisions.parquet); not computed per challenger here"
            ),
        }
        predeclared_sign, predeclared_sign_reason = _predeclared_sign(challenger_entry)
        per_week_deltas = _weekly_paired_deltas(entrant_settled, active_settled)
        split_half = _split_half_reliability(per_week_deltas, samples=samples, seed=seed)
        paired_metric = (paired.get("metrics") or {}).get("paired_delta_accuracy_points")
        closing_ground_candidate, closing_ground_evidence = _closing_ground_candidate(
            predeclared_sign=predeclared_sign,
            predeclared_sign_reason=predeclared_sign_reason,
            paired_metric=paired_metric,
            split_half=split_half,
        )
        # The candidate field is advisory ONLY: classification above is
        # already fixed at unresolved_below_power regardless of what this
        # says, per the module docstring and AGENTS.md.
        row["closing_ground_candidate"] = closing_ground_candidate
        row["closing_ground_evidence"] = closing_ground_evidence
        has_settled_shared_data = bool(paired.get("shared_settled_games", 0))
        action, action_detail = _next_admissible_action(
            challenger_entry,
            has_settled_shared_data=has_settled_shared_data,
            weak_signal_registry=weak_signal_registry,
            rotation_registry=rotation_registry,
        )
        row["next_admissible_action"] = action
        row["next_admissible_action_detail"] = action_detail
    return row


def build_season_scorecards(
    artifacts_root: Path,
    data_root: Path,
    features: pd.DataFrame,
    *,
    season: int,
    through_week: int | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    now: datetime | None = None,
    registry_root: Path | None = None,
) -> list[dict[str, Any]]:
    """One scorecard row per entrant (the active model, then every registered challenger).

    ``features`` is the canonical feature table (default
    ``data/processed/game_features.parquet``), used exactly as
    ``nfl-ats prospective-score`` uses it: ``game_id``/``result`` for
    settlement, and as the schedule ``nfl_ats.clv.live_close_reference`` reads
    against for the secondary close grade.

    ``registry_root`` (ENG-33) is read-only: it locates
    ``weak_signals.json``/``rotation_registry.json`` for the
    ``closing_ground_candidate``/``next_admissible_action`` advisories,
    defaulting to the same tracked ``registry/`` (honouring
    ``NFL_ATS_REGISTRY_DIR``) every other registry reader in this repo uses.
    This function never writes to either file.
    """

    instant = now or datetime.now(UTC)
    outcomes = features.loc[:, ["game_id", "result"]].copy()
    close_reference = live_close_reference(data_root / "market" / "raw", features, as_of=instant)
    weak_signal_registry, rotation_registry = _load_registries_for_report(registry_root)

    active_all = load_paper_decisions(artifacts_root)
    active_scope = _scope(active_all, season=season, through_week=through_week)
    active_settled = settle_prospective_picks(
        active_scope, outcomes, close_reference=close_reference
    )

    rows: list[dict[str, Any]] = [
        _entrant_row(
            ACTIVE_MODEL_ENTRANT_ID,
            entrant_kind="active_model",
            challenger_status=None,
            registered_evidence=None,
            challenger_entry=None,
            weak_signal_registry=weak_signal_registry,
            rotation_registry=rotation_registry,
            entrant_scope=active_scope,
            active_scope=active_scope,
            entrant_settled=active_settled,
            active_settled=active_settled,
            artifacts_root=artifacts_root,
            artifact_column="forecast_artifact",
            is_active_model=True,
            outcomes=outcomes,
            season=season,
            through_week=through_week,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
    ]

    try:
        registry = load_challenger_registry(artifacts_root)
    except FileNotFoundError:
        registry = {"challengers": []}
    challenger_all = load_challenger_decisions(artifacts_root)

    for entry in registry.get("challengers", []):
        if not isinstance(entry, dict):
            continue
        challenger_id = str(entry.get("challenger_id"))
        if not challenger_all.empty:
            own = challenger_all.loc[challenger_all["challenger_id"].astype(str).eq(challenger_id)]
        else:
            own = challenger_all
        own_scope = _scope(own, season=season, through_week=through_week)
        own_settled = settle_prospective_picks(own_scope, outcomes, close_reference=close_reference)
        evidence = entry.get("evidence")
        registered_evidence = (
            {
                "registered_status": entry.get("status"),
                "probability_positive": (evidence or {}).get("probability_positive")
                if isinstance(evidence, dict)
                else None,
                "registry_verdict": (evidence or {}).get("registry_verdict")
                if isinstance(evidence, dict)
                else None,
            }
            if entry.get("status") is not None or evidence is not None
            else None
        )
        rows.append(
            _entrant_row(
                challenger_id,
                entrant_kind="challenger",
                challenger_status=str(entry.get("status"))
                if entry.get("status") is not None
                else None,
                registered_evidence=registered_evidence,
                challenger_entry=entry,
                weak_signal_registry=weak_signal_registry,
                rotation_registry=rotation_registry,
                entrant_scope=own_scope,
                active_scope=active_scope,
                entrant_settled=own_settled,
                active_settled=active_settled,
                artifacts_root=artifacts_root,
                artifact_column="source_artifact",
                is_active_model=False,
                outcomes=outcomes,
                season=season,
                through_week=through_week,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
        )

    return rows


def _fmt_pct(value: float | None) -> str:
    return f"{value * 100.0:.2f}%" if value is not None and not pd.isna(value) else "--"


def _fmt_pp(value: float | None) -> str:
    return f"{value:+.2f}pp" if value is not None and not pd.isna(value) else "--"


def _fmt_prob(value: float | None) -> str:
    return f"{value:.3f}" if value is not None and not pd.isna(value) else "--"


def _md_cell(text: str) -> str:
    """Escape ``|`` so a challenger id containing one (e.g. a fingerprint-style
    ``a|b|c`` name) cannot be mistaken for a Markdown table cell boundary."""

    return text.replace("|", "\\|")


def scorecards_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """A flat, one-row-per-entrant table for the Markdown render and CSV export."""

    flat: list[dict[str, Any]] = []
    for row in rows:
        paired = row.get("paired_vs_active") or {}
        paired_metric = (paired.get("metrics") or {}).get("paired_delta_accuracy_points", {})
        marginal = row.get("overlay_marginal") or {}
        marginal_metric = (marginal.get("metrics") or {}).get(
            "marginal_paired_delta_accuracy_points", {}
        )
        refresh = row.get("refresh_effect") or {}
        refresh_metric = (refresh.get("metrics") or {}).get(
            "refresh_paired_delta_accuracy_points", {}
        )
        calibration = row.get("calibration") or {}
        flat.append(
            {
                "entrant_id": row["entrant_id"],
                "entrant_kind": row["entrant_kind"],
                "challenger_status": row.get("challenger_status"),
                "classification": row["classification"],
                "interval_crosses_zero": row.get("interval_crosses_zero"),
                "games_on_card": row.get("games_on_card"),
                "games_recorded": row.get("games_recorded"),
                "coverage_ratio": row.get("coverage_ratio"),
                "settled_games": row.get("settled_games"),
                "pushes": row.get("pushes"),
                "pending": row.get("pending"),
                "accuracy_decision_line": row.get("accuracy_decision_line"),
                "accuracy_close_line": row.get("accuracy_close_line"),
                "paired_delta_accuracy_points": paired_metric.get("estimate"),
                "paired_delta_lower": paired_metric.get("interval_lower"),
                "paired_delta_upper": paired_metric.get("interval_upper"),
                "paired_delta_probability_positive": paired_metric.get("probability_positive"),
                "paired_shared_settled_games": paired.get("shared_settled_games"),
                "marginal_delta_accuracy_points": marginal_metric.get("estimate"),
                "marginal_probability_positive": marginal_metric.get("probability_positive"),
                "marginal_disagreement_games": marginal.get("disagreement_games"),
                "refresh_flips": refresh.get("flips"),
                "refresh_delta_accuracy_points": refresh_metric.get("estimate"),
                "refresh_probability_positive": refresh_metric.get("probability_positive"),
                "brier_score": calibration.get("brier_score"),
                "calibration_games": calibration.get("games_with_recorded_probability"),
                # ENG-33: advisory only -- classification (above) is unchanged by these.
                "closing_ground_candidate": row.get("closing_ground_candidate"),
                "next_admissible_action": row.get("next_admissible_action"),
                "next_admissible_action_detail": row.get("next_admissible_action_detail"),
            }
        )
    return pd.DataFrame(flat)


def render_markdown(rows: list[dict[str, Any]], *, season: int, through_week: int | None) -> str:
    """The printed/written Markdown scorecard table plus a short legend."""

    scope_text = f"season {season}" + (
        f" through week {through_week}" if through_week else " (full season)"
    )
    lines = [
        f"# Prospective evidence scorecard -- {scope_text}",
        "",
        (
            "Every row is a REPORT, not a verdict: classifications flow only through "
            "`nfl-ats weak-signals record` / `nfl-ats rotation record-look`, which this "
            "table never calls. Per AGENTS.md, an interval crossing zero is never grounds "
            "to reject a signal, so every row below is classified `unresolved_below_power` "
            "-- `probability_positive` is the number to read, not the interval's sign."
        ),
        "",
        (
            "| Entrant | Status | Coverage | Settled | Decision-line acc. | Close-line acc. | "
            "Paired delta vs active | P+ | Marginal (fired-on) delta | P+ | Refresh flips | "
            "Refresh delta | Brier | Classification | Closing-ground candidate | "
            "Next admissible action |"
        ),
        ("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"),
    ]
    for row in rows:
        paired = row.get("paired_vs_active") or {}
        paired_metric = (paired.get("metrics") or {}).get("paired_delta_accuracy_points", {})
        marginal = row.get("overlay_marginal") or {}
        marginal_metric = (marginal.get("metrics") or {}).get(
            "marginal_paired_delta_accuracy_points", {}
        )
        refresh = row.get("refresh_effect") or {}
        refresh_metric = (refresh.get("metrics") or {}).get(
            "refresh_paired_delta_accuracy_points", {}
        )
        calibration = row.get("calibration") or {}
        coverage_ratio = row.get("coverage_ratio")
        coverage_text = f"{row.get('games_covered_of_card', 0)}/{row.get('games_on_card', 0)}" + (
            f" ({coverage_ratio * 100:.0f}%)" if coverage_ratio is not None else ""
        )
        refresh_text = (
            str(refresh.get("flips"))
            if row["entrant_kind"] == "active_model" and refresh.get("available")
            else "--"
        )
        closing_ground = row.get("closing_ground_candidate")
        closing_ground_text = f"{closing_ground} (candidate)" if closing_ground else "--"
        next_action_text = str(row.get("next_admissible_action") or "--")
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(str(row["entrant_id"])),
                    _md_cell(str(row.get("challenger_status") or "active")),
                    coverage_text,
                    str(row.get("settled_games", 0)),
                    _fmt_pct(row.get("accuracy_decision_line")),
                    _fmt_pct(row.get("accuracy_close_line")),
                    _fmt_pp(paired_metric.get("estimate")),
                    _fmt_prob(paired_metric.get("probability_positive")),
                    _fmt_pp(marginal_metric.get("estimate")),
                    _fmt_prob(marginal_metric.get("probability_positive")),
                    refresh_text,
                    _fmt_pp(refresh_metric.get("estimate"))
                    if row["entrant_kind"] == "active_model"
                    else "--",
                    (
                        f"{calibration.get('brier_score'):.4f}"
                        if calibration.get("brier_score") is not None
                        else "--"
                    ),
                    row["classification"],
                    closing_ground_text,
                    next_action_text,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Decision-line accuracy is the PRIMARY grade (the line the pick was actually made "
        "at, per docs/prospective_evidence.md) and is treated as this report's "
        "opener-equivalent grade; close-line is secondary. `--` means no settled games (or "
        "no applicable data) yet, not zero."
    )
    lines.append(
        "`Closing-ground candidate` (ENG-33) flags rows whose paired-delta interval sits "
        "entirely opposite a predeclared sign, or whose split-half reliability interval's "
        "upper bound is at or below zero -- ADVISORY only, always a *candidate*, never a "
        "verdict: `Classification` stays `unresolved_below_power` regardless. "
        "`Next admissible action` is drawn from the fixed six-item vocabulary "
        '(never "wait"); `closed` appears only when the registry already holds an '
        "admissible closure for a signal this row cites. Neither column is written by this "
        "tool -- both are read-only reports of what `nfl-ats weak-signals record` / "
        "`nfl-ats rotation record-look` would find if run."
    )
    return "\n".join(lines)
