"""Best Pick nomination v2: calibrated-probability distance, filtered by
cross-book opener dispersion (owner decision 2026-08-18, POL-09).

The incumbent Best Pick chooser (``nfl_ats.best_pick.sweep_robustness``,
still selected via ``select_best_pick``) is measured signal-free: 24 of 35
confirmation weeks were ties, and the tie-agnostic edge is +0.92 points, not
the recorded +8.68 (``docs/best_pick_ranker.md``). This module implements the
MEASURED winner from a same-day screen (**read**,
``scratchpad/bestpick_opener/predeclaration.md`` + ``results.md``, session
scratchpad; script ``scripts/best_pick_opener_ranker_eval.py``): chooser 6,
``dispersion_filtered_candidate`` -- an alpha=2000 calibrated probability's
distance from 0.5, restricted to that week's below-median cross-book opener
``spread_std`` games (fallback to the full week on missing/degenerate
dispersion data), scored **+3.92 points vs its unfiltered parent**
(``probability_positive`` 0.813, interval [-3.92, +11.76], 102 paired weeks
-- third reuse of the same 107-week opener population, multiplicity discount
stated explicitly, interval contains zero, this is a lean not a resolution).

**Owner-specified composition, not a single scored chooser.** The
predeclared chooser 6 breaks ties by ascending ``game_id`` (like every other
chooser in that screen); a SEPARATE chooser (8, ``dispersion_tiebreak``,
unfiltered, P+ 0.0 on a 5-week degenerate sample -- see ``results.md``)
tested breaking ties by lower dispersion instead. The owner's 2026-08-18
production directive composes the two: chooser 6's filter, PLUS a
dispersion tie-break applied *within* that filtered pool. That exact
composition was never itself scored as one chooser -- each half is
separately measured, the combination is not. Flagged here and in
``docs/best_pick_ranker.md`` rather than silently presented as identical to
chooser 6.

SIDES NEVER CHANGE. This module only decides WHICH game gets the week's
Best Pick nomination; every game's own forced pick still comes from the
active model, exactly as published. Best Pick selection (both v1 and v2)
deliberately runs on UN-overlaid predictions in ``publishing.py``, so this
lever and the year-1-coach-fade overlay (``nfl_ats.coach_fade_overlay``)
stay independently measured rather than silently composed -- see that
module's docstring for the same property stated the other direction.

Three things live here:

1. :func:`week_dispersion_pool` -- the below-median cross-book opener
   ``spread_std`` eligibility pool for one week, read from the LOCAL market
   snapshot store the weekly pipeline already populates via ``odds-ingest``
   (``nfl_ats.market_data.load_quote_history`` / ``tuesday_opener_quotes``)
   -- this module never calls the odds API. The historical evidence above
   was built from a decision-labeled backfill archive
   (``nfl_ats.clv.build_pairing_table``'s ``spread_std``); production has no
   such labeled snapshot for the live Tuesday capture, so this reads the
   same free-form store ``nfl_ats.cli``'s existing
   ``pool-card-at-lines --use-tuesday-opener`` already reads, extended
   (``market_data.tuesday_opener_quotes``'s new ``opener_std`` column) to
   carry the same dispersion measure the historical evidence used.
2. :func:`fit_candidate_probabilities` / :func:`nominate_v2` -- the
   alpha=2000 ``market_residual`` probability, fit walk-forward (trained on
   completed games strictly before the target week, mirroring
   ``nfl_ats.outcomes.score_outcome_week``'s own cutoff -- the same recipe
   ``scripts/ridge_alpha_promotion_eval.py`` used to name this alpha), and
   the ranking/fallback/tie-break rule itself.
3. :func:`record_nomination_challenger_decisions` -- writes v2's weekly
   nominee to the prospective challenger ledger under
   :data:`CHALLENGER_ID` (registered in
   ``artifacts/prospective/challengers.json`` as ``best_pick_nomination_v2``,
   pinned to the active model's configuration fingerprint), mirroring
   ``nfl_ats.coach_fade_overlay.record_overlay_challenger_decisions``. v1's
   nomination is already tracked, unchanged, via the active model's own
   ``is_best_pick`` flag on the primary paper-decision ledger
   (``nfl_ats.clv.record_paper_decisions`` -- NOT modified by this module);
   together the two ledgers score v1 against v2 weekly without either
   rule's tracking depending on the other.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.margin import MARGIN_FEATURE_PROFILES, MarginFeatureProfile
from nfl_ats.market_data import load_quote_history, tuesday_opener_quotes
from nfl_ats.outcomes import fit_margin_models_for_week
from nfl_ats.prospective_scoring import (
    ACTIVE_CHALLENGER_STATUS,
    CHALLENGER_DECISION_COLUMNS,
    artifact_model_config,
    challenger_ledger_path,
    config_fingerprint,
    find_challenger,
    load_challenger_decisions,
)
from nfl_ats.provenance import sha256_file

#: docs/ridge_alpha.md section 4's named candidate -- the walk-forward Brier
#: optimum, confirmed at the opener grade by
#: artifacts/ridge_alpha_promotion/20260818T221459Z. Not re-derived here.
NOMINATION_RIDGE_ALPHA = 2_000.0

#: Owner decision 2026-08-18: nominate the weekly Best Pick with the measured
#: v2 rule instead of the signal-free ``sweep_robustness`` incumbent.
#: Flipping this off restores the v1 nomination wherever ``publishing.py``
#: checks it. The challenger ledger (:func:`record_nomination_challenger_decisions`)
#: records v2's nomination regardless of this switch, since it reads the
#: flag itself rather than relying on the published card's effect.
NOMINATION_V2_ENABLED = True

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "best_pick_nomination_v2"

#: v3 (POL-09 audit, docs/best_pick_ranker.md "v3 audit", 2026-08-19):
#: registered in artifacts/prospective/challengers.json as a SIDE-LEDGER-ONLY
#: challenger. v3 uses the exact same eligibility pool and primary ranking as
#: v2 (:func:`week_dispersion_pool`, ``candidate_dist``) but breaks ties on
#: ascending ``game_id`` alone -- see :func:`select_nominee_v3`. Never wired
#: into ``publishing.py``; ``NOMINATION_V2_ENABLED`` is untouched and governs
#: the published card exactly as before.
CHALLENGER_ID_V3 = "best_pick_nomination_v3"


# ---------------------------------------------------------------------------
# 1. Dispersion pool
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispersionPool:
    """Below-median cross-book opener ``spread_std`` pool for one week.

    ``frame`` is one row per requested ``game_id`` with ``spread_std`` (NaN
    when the local Tuesday capture has no book coverage for that game) and
    ``pool_pass`` (eligible for nomination this week).

    Fallback rule (predeclared, measured over 107 historical opener weeks in
    ``scratchpad/bestpick_opener/predeclaration.md``'s "Addendum 2026-08-18
    (evening)", mirrored here for one live week): the WHOLE week falls back
    to its full, unfiltered game set (every game passes) if EITHER (a) any
    game that week is missing ``spread_std``, OR (b) a strict
    ``spread_std < week_median`` filter would leave zero eligible games
    (ties at the week's minimum -- common, since many books agree to the
    half-point). Both triggers are counted separately.
    """

    frame: pd.DataFrame
    fallback: bool
    fallback_reason: str | None  # "missing_data" | "empty_filter" | None
    n_games: int
    n_missing: int
    n_pool_pass: int


def week_dispersion_pool(market_root: Path, game_ids: Sequence[str]) -> DispersionPool:
    """The below-median-``spread_std`` eligibility pool for one week's games.

    Reads ONLY the local market snapshot store the weekly pipeline already
    populates via ``odds-ingest`` (``nfl_ats.market_data.load_quote_history``)
    -- never calls the odds API. A missing or empty store degrades to "every
    game missing ``spread_std``", which the fallback rule below already
    turns into a full-week pool rather than a failure.
    """

    ids = [str(game_id) for game_id in game_ids]
    if not ids:
        raise ValueError("week_dispersion_pool needs at least one game_id")

    quotes = load_quote_history(market_root)
    opener = tuesday_opener_quotes(quotes)
    dispersion = (
        opener.rename(columns={"nflverse_game_id": "game_id", "opener_std": "spread_std"})[
            ["game_id", "spread_std"]
        ]
        .assign(game_id=lambda frame: frame["game_id"].astype(str))
        .drop_duplicates("game_id")
    )

    frame = pd.DataFrame({"game_id": ids}).merge(dispersion, on="game_id", how="left")
    n_games = len(frame)
    n_missing = int(frame["spread_std"].isna().sum())

    if n_missing > 0:
        frame["pool_pass"] = True
        return DispersionPool(frame, True, "missing_data", n_games, n_missing, n_games)

    median = frame["spread_std"].median()
    below = frame["spread_std"].lt(median)
    if not bool(below.any()):
        frame["pool_pass"] = True
        return DispersionPool(frame, True, "empty_filter", n_games, n_missing, n_games)

    frame["pool_pass"] = below
    return DispersionPool(frame, False, None, n_games, n_missing, int(below.sum()))


# ---------------------------------------------------------------------------
# 2. Candidate probabilities and the nomination rule
# ---------------------------------------------------------------------------


def fit_candidate_probabilities(
    features: pd.DataFrame,
    *,
    season: int,
    week: int,
    regressor: str,
    feature_profile: MarginFeatureProfile,
    min_train_games: int,
    ridge_alpha: float = NOMINATION_RIDGE_ALPHA,
) -> pd.DataFrame:
    """Alpha=2000 ``market_residual`` cover probability for one week's games.

    Walk-forward, via ``nfl_ats.outcomes.fit_margin_models_for_week`` --
    the same public entry point ``nfl-ats pool-card-at-lines`` uses to refit
    at a custom line, and the same training-cutoff logic (strictly before
    the target week's earliest kickoff) ``score_outcome_week`` uses to build
    the active card's own weekly forecast. Predicts at each game's OWN
    ``spread_line`` from ``features`` -- the same line the active card's own
    probability was computed at -- so "distance from 0.5" is apples-to-apples
    with how the published forecast itself was built; no market re-fetch for
    the LINE is needed (only for dispersion, a separate signal, see
    :func:`week_dispersion_pool`).
    """

    if feature_profile not in MARGIN_FEATURE_PROFILES:
        raise ValueError(f"Unknown feature profile: {feature_profile!r}")
    target, margin_models = fit_margin_models_for_week(
        features,
        season=season,
        week=week,
        regressor=regressor,
        min_train_games=min_train_games,
        feature_profile=feature_profile,
        ridge_alpha=ridge_alpha,
        methods=("market_residual",),
    )
    model = margin_models["market_residual"]
    forecast = model.predict(target)
    result = target[["game_id"]].copy()
    result["game_id"] = result["game_id"].astype(str)
    result["candidate_home_cover_probability"] = forecast["home_cover_probability"].to_numpy()
    result["candidate_dist"] = (result["candidate_home_cover_probability"] - 0.5).abs()
    return result.reset_index(drop=True)


@dataclass(frozen=True)
class NominationV2Result:
    """One week's v2 nomination, plus everything needed to disclose it."""

    game_id: str
    n_tied_at_max: int
    tie_break: str  # "none" | "dispersion" | "game_id"
    probability_table: pd.DataFrame
    dispersion: DispersionPool


def select_nominee(candidates: pd.DataFrame) -> tuple[str, int, str]:
    """The pure ranking/tie-break rule, isolated from fitting and I/O.

    ``candidates`` is the ALREADY-ELIGIBLE pool for the week (every row has
    ``pool_pass`` True, or the caller never restricted it) with ``game_id``,
    ``candidate_dist``, and ``spread_std`` columns. Returns
    ``(nominee_game_id, n_tied_at_max, tie_break_used)``.

    Rule (owner decision 2026-08-18, POL-09 -- composed from two separately
    -measured pieces, see the module docstring): rank by ``candidate_dist``
    descending; among ties at the max, prefer LOWER ``spread_std``
    (``na_position="last"`` -- a candidate with no measurable dispersion
    loses the dispersion tie-break rather than being treated as "lowest");
    any remaining tie breaks on ascending ``game_id``, matching
    ``nfl_ats.best_pick.select_best_pick``'s convention exactly.
    """

    if candidates.empty:
        raise ValueError("select_nominee needs at least one candidate")
    top_value = candidates["candidate_dist"].max()
    tied = candidates.loc[candidates["candidate_dist"].eq(top_value)]
    n_tied = len(tied)
    if n_tied <= 1:
        tie_break = "none"
    elif tied["spread_std"].dropna().nunique() >= 2:
        tie_break = "dispersion"
    else:
        tie_break = "game_id"
    ordered = tied.sort_values(
        ["spread_std", "game_id"], ascending=[True, True], na_position="last"
    )
    return str(ordered.iloc[0]["game_id"]), n_tied, tie_break


def select_nominee_v3(candidates: pd.DataFrame) -> tuple[str, int, str]:
    """v3's pure ranking/tie-break rule (POL-09 audit, docs/best_pick_ranker.md
    "v3 audit", 2026-08-19).

    Identical primary ranking to :func:`select_nominee` (``candidate_dist``
    descending) over the SAME eligible pool (:func:`nominate_v3` restricts to
    the same below-median-dispersion pool :func:`nominate_v2` does) -- but
    ties at the max break on ascending ``game_id`` alone, with no dispersion
    tie-break layer. This is
    ``scripts/best_pick_opener_ranker_eval.py``'s chooser 6 exactly: the
    half of v2's composition that WAS separately measured
    (``probability_positive`` 0.813 vs its unfiltered parent,
    ``registry/weak_signals.json:best_pick_opener_ranker_dispersion_filtered_candidate_vs_unfiltered``).
    The OTHER half of v2's composition (a dispersion tie-break layered
    *inside* the filtered pool -- :func:`select_nominee` above) was audited
    against this rule head-to-head on the same 107-week opener population
    (``scripts/best_pick_nomination_v3_audit.py``) and measured no better:
    +0.97 accuracy points, ``probability_positive`` 0.631, interval
    [0.0, +2.91] -- but the ENTIRE difference traces to one diverging week of
    103 paired weeks (the other nominal divergence produced identical
    lift for both rules). An EV-positive lean on a forced pick per
    ``AGENTS.md`` ("a promotion bar is not a decision bar"), not a resolved
    improvement; report as ``unresolved_below_power``, never "confirmed".
    """

    if candidates.empty:
        raise ValueError("select_nominee_v3 needs at least one candidate")
    top_value = candidates["candidate_dist"].max()
    tied = candidates.loc[candidates["candidate_dist"].eq(top_value)]
    n_tied = len(tied)
    tie_break = "none" if n_tied <= 1 else "game_id"
    ordered = tied.sort_values("game_id", ascending=True)
    return str(ordered.iloc[0]["game_id"]), n_tied, tie_break


def nominate_v2(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    market_root: Path,
    season: int,
    week: int,
    regressor: str,
    feature_profile: MarginFeatureProfile,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    ridge_alpha: float = NOMINATION_RIDGE_ALPHA,
) -> NominationV2Result | None:
    """The v2 rule's Best Pick nominee for one week's card, or ``None``.

    Regular season only (mirrors ``nfl_ats.best_pick``'s v1 gate exactly): a
    ``predictions`` frame carrying any non-REG game, or no games at all,
    returns ``None`` rather than nominate outside the population this rule
    was measured on.

    Raises (never silently mis-nominates) on genuine infrastructure
    problems -- a candidate probability missing for a carded game, or the
    dispersion join dropping/duplicating rows. Callers that want the
    "never fail the publish" degrade contract (``publishing.py``) catch
    those at the call site, matching how ``_apply_overlay`` degrades on
    ``nfl_ats.coach_fade_overlay.apply_coach_fade_overlay``'s own raises.
    """

    if predictions.empty or "game_id" not in predictions.columns:
        return None
    if (
        "game_type" in predictions.columns
        and not predictions["game_type"].astype(str).eq("REG").all()
    ):
        return None

    game_ids = predictions["game_id"].astype(str).tolist()
    probabilities = fit_candidate_probabilities(
        features,
        season=season,
        week=week,
        regressor=regressor,
        feature_profile=feature_profile,
        min_train_games=min_train_games,
        ridge_alpha=ridge_alpha,
    )
    available = set(probabilities["game_id"])
    missing_games = sorted(set(game_ids) - available)
    if missing_games:
        raise DataContractError(
            "Candidate probabilities are missing games on the published card: "
            f"{', '.join(missing_games[:5])}"
        )
    probabilities = probabilities.loc[probabilities["game_id"].isin(game_ids)].reset_index(
        drop=True
    )

    dispersion = week_dispersion_pool(market_root, game_ids)
    table = probabilities.merge(dispersion.frame, on="game_id", how="inner", validate="one_to_one")
    if len(table) != len(game_ids):
        raise DataContractError("Dispersion pool join dropped or duplicated games")

    candidates = table.loc[table["pool_pass"]]
    if candidates.empty:
        # The fallback rule guarantees this cannot happen (a fallback week
        # marks every game pool_pass=True); fail loudly if it somehow does
        # rather than silently nominate from the wrong pool.
        raise DataContractError("Dispersion-pool fallback left zero eligible candidates")

    nominee, n_tied, tie_break = select_nominee(candidates)

    return NominationV2Result(
        game_id=nominee,
        n_tied_at_max=n_tied,
        tie_break=tie_break,
        probability_table=table.sort_values("game_id").reset_index(drop=True),
        dispersion=dispersion,
    )


@dataclass(frozen=True)
class NominationV3Result:
    """One week's v3 nomination -- same fitting and eligibility pool as v2,
    but v3's own dispersion-tiebreak-free ranking rule (:func:`select_nominee_v3`)."""

    game_id: str
    n_tied_at_max: int
    tie_break: str  # "none" | "game_id"
    probability_table: pd.DataFrame
    dispersion: DispersionPool


def nominate_v3(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    market_root: Path,
    season: int,
    week: int,
    regressor: str,
    feature_profile: MarginFeatureProfile,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    ridge_alpha: float = NOMINATION_RIDGE_ALPHA,
) -> NominationV3Result | None:
    """v3's Best Pick nominee for one week's card, or ``None``.

    Deliberately duplicates :func:`nominate_v2`'s small orchestration body
    (rather than refactoring it to share code) so the live production path
    (``NOMINATION_V2_ENABLED``) never changes shape for a challenger that only
    ever writes to the side ledger -- see :func:`select_nominee_v3` for what
    actually differs and why. Reuses :func:`fit_candidate_probabilities` and
    :func:`week_dispersion_pool` byte-for-byte, so the candidate probabilities
    and the eligibility pool are IDENTICAL to v2's; only the tie-break rule
    differs.
    """

    if predictions.empty or "game_id" not in predictions.columns:
        return None
    if (
        "game_type" in predictions.columns
        and not predictions["game_type"].astype(str).eq("REG").all()
    ):
        return None

    game_ids = predictions["game_id"].astype(str).tolist()
    probabilities = fit_candidate_probabilities(
        features,
        season=season,
        week=week,
        regressor=regressor,
        feature_profile=feature_profile,
        min_train_games=min_train_games,
        ridge_alpha=ridge_alpha,
    )
    available = set(probabilities["game_id"])
    missing_games = sorted(set(game_ids) - available)
    if missing_games:
        raise DataContractError(
            "Candidate probabilities are missing games on the published card: "
            f"{', '.join(missing_games[:5])}"
        )
    probabilities = probabilities.loc[probabilities["game_id"].isin(game_ids)].reset_index(
        drop=True
    )

    dispersion = week_dispersion_pool(market_root, game_ids)
    table = probabilities.merge(dispersion.frame, on="game_id", how="inner", validate="one_to_one")
    if len(table) != len(game_ids):
        raise DataContractError("Dispersion pool join dropped or duplicated games")

    candidates = table.loc[table["pool_pass"]]
    if candidates.empty:
        # The fallback rule guarantees this cannot happen (a fallback week
        # marks every game pool_pass=True); fail loudly if it somehow does
        # rather than silently nominate from the wrong pool.
        raise DataContractError("Dispersion-pool fallback left zero eligible candidates")

    nominee, n_tied, tie_break = select_nominee_v3(candidates)

    return NominationV3Result(
        game_id=nominee,
        n_tied_at_max=n_tied,
        tie_break=tie_break,
        probability_table=table.sort_values("game_id").reset_index(drop=True),
        dispersion=dispersion,
    )


# ---------------------------------------------------------------------------
# 3. Disclosure text
# ---------------------------------------------------------------------------


def nomination_v2_tie_note(result: NominationV2Result) -> str:
    """Plain-language tie disclosure, mirroring
    ``nfl_ats.best_pick.best_pick_tie_note`` -- but distinguishing a tie the
    dispersion tie-break actually RESOLVED (a reproducible lean, not luck)
    from one that still fell through to an alphabetical ``game_id``
    tie-break (arbitrary, exactly like v1's disclosure).
    """

    if result.n_tied_at_max <= 1:
        return ""
    if result.tie_break == "dispersion":
        return (
            f"{result.n_tied_at_max} games tied on calibrated-probability distance this week; "
            "the tie was broken by lower cross-book dispersion, not chosen arbitrarily."
        )
    return (
        f"This week {result.n_tied_at_max} games tie at the top of that signal, and the "
        "dispersion tie-break did not discriminate between them either, so choosing between "
        "them is arbitrary -- reproducible, but not a lean."
    )


def nomination_v3_tie_note(result: NominationV3Result) -> str:
    """Plain-language tie disclosure for v3. Unlike
    :func:`nomination_v2_tie_note`, v3 never attempts a dispersion
    tie-break at all (see :func:`select_nominee_v3`), so every multi-way tie
    is arbitrary by construction -- there is no "resolved by dispersion"
    branch to report. Not surfaced on any published card (v3 is never wired
    into ``publishing.py``); used only in the challenger ledger's own
    recording result for debugging.
    """

    if result.n_tied_at_max <= 1:
        return ""
    return (
        f"{result.n_tied_at_max} games tie at the top of that signal in the "
        "dispersion-filtered pool; v3 breaks ties on ascending game_id alone "
        "(no dispersion tie-break), so choosing between them is arbitrary -- "
        "reproducible, but not a lean."
    )


#: Verbatim per the owner's 2026-08-18 decision -- do not paraphrase; other
#: surfaces (tests, docs) match against this exact string.
NOMINATION_V2_METHOD_SENTENCE = "nominated by calibrated probability among low-disagreement games"


def nomination_v2_disclosure_note(result: NominationV2Result) -> str:
    """The card-facing sentence disclosing the v2 nomination method.

    Always leads with :data:`NOMINATION_V2_METHOD_SENTENCE` verbatim, then
    states the dispersion-pool fallback (if this week fell back to the full
    game set) and the tie state (via :func:`nomination_v2_tie_note`) -- so
    the disclosure never implies a filter that did not actually apply.
    """

    sentence = NOMINATION_V2_METHOD_SENTENCE
    dispersion = result.dispersion
    if dispersion.fallback:
        reason = (
            "at least one game was missing cross-book opener data"
            if dispersion.fallback_reason == "missing_data"
            else "no game sat strictly below the week's own median dispersion"
        )
        sentence += f" (fell back to the full, unfiltered week this week: {reason})"
    sentence += "."
    tie_note = nomination_v2_tie_note(result)
    return f"{sentence} {tie_note}".rstrip() if tie_note else sentence


# ---------------------------------------------------------------------------
# 4. Challenger-ledger recording (both nominations tracked)
# ---------------------------------------------------------------------------


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_nomination_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the v2 rule's weekly nominee to the prospective challenger
    ledger under :data:`CHALLENGER_ID`, so the season scores v1 against v2.

    v1's nomination needs no new recording here: it is already tracked,
    unchanged, via the active model's own ``is_best_pick`` flag on the
    primary paper-decision ledger (``nfl_ats.clv.record_paper_decisions``,
    NOT modified by this module). This function supplies the other half --
    mirroring ``nfl_ats.coach_fade_overlay.record_overlay_challenger_decisions``:
    this challenger's "model" IS the active model's own synchronized weekly
    forecast (no separate ``margin-predict`` artifact exists for it), so this
    reads that forecast directly rather than searching
    ``artifacts/margin_predictions/`` by fingerprint.

    Unlike the overlay challenger, which records every game's (possibly
    flipped) pick, this records exactly ONE row per week: the v2 nominee, at
    the active model's own UNCHANGED pick side and line for that one game
    (sides never change -- only which game is nominated does). ``bet_side``
    is ``PASS`` and ``edge`` is NaN: this tracks the nomination's forced-pick
    (Best Pick) accuracy, not an invented paper-bet edge, mirroring the
    overlay challenger's identical rationale.

    Recording follows the same whole-week-pre-kickoff rule
    ``nfl_ats.clv.record_paper_decisions`` uses for the v1 Best Pick flag: a
    nominee is only recorded while EVERY game on the card is still in the
    future, so a backdated nomination (choosing with results already in
    hand) is impossible. Every anti-backdating guarantee of the ordinary
    challenger path still applies otherwise: pre-kickoff only, never
    rewrites an existing row, and the ``RECORDING_LOCK_WINDOW`` rehearsal
    guard.
    """

    entry = find_challenger(artifacts_root, CHALLENGER_ID)
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {CHALLENGER_ID!r} is registered as {status!r}; only "
            f"{ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError(
            "No synchronized active ATS model is available to record nomination decisions from"
        )
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        raise ValueError("Active ATS model has no linked weekly forecast")
    metadata_path = forecast / "metadata.json"
    card_path = forecast / "recommendations.csv"
    if not metadata_path.is_file() or not card_path.is_file():
        raise ValueError(f"Linked weekly forecast is incomplete: {forecast}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("active_model_id") != active.get("model_id"):
        raise ValueError("Weekly forecast model ID does not match the active model")
    if metadata.get("synchronization_status") != "SYNCHRONIZED":
        raise ValueError("Weekly forecast is not synchronized with an evaluation")

    observed_config = artifact_model_config(metadata)
    declared_fingerprint = config_fingerprint(entry.get("model", {}))
    observed_fingerprint = config_fingerprint(observed_config)
    if declared_fingerprint != observed_fingerprint:
        raise DataContractError(
            f"Challenger {CHALLENGER_ID!r} is registered pinned to configuration "
            f"fingerprint {declared_fingerprint}, but the current active forecast "
            f"{forecast} was produced with {observed_fingerprint}; the active model "
            "changed underneath this nomination rule -- re-register before recording"
        )

    card = pd.read_csv(card_path)
    required = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
    }
    missing = sorted(required.difference(card.columns))
    if missing:
        raise DataContractError(f"Active forecast card is missing columns: {', '.join(missing)}")
    if card["game_id"].duplicated().any():
        raise DataContractError("Active forecast card contains duplicate games")
    spreads = pd.to_numeric(card["spread_line"], errors="coerce")
    if not np.isfinite(spreads.to_numpy(dtype=float)).all():
        raise DataContractError("Active forecast card has games without a decision spread")
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Active forecast card has games without a kickoff timestamp")

    feature_table_path = observed_config.get("feature_table")
    if not feature_table_path:
        raise DataContractError(
            "Active forecast metadata carries no feature_table path to refit the "
            "alpha=2000 nomination candidate from"
        )
    features = pd.read_parquet(feature_table_path)
    season = int(card["season"].iloc[0])
    week = int(card["week"].iloc[0])
    min_train_games = metadata.get("min_train_games")
    result = nominate_v2(
        card,
        features,
        market_root=data_root / "market" / "raw",
        season=season,
        week=week,
        regressor=str(metadata.get("regressor", "ridge")),
        # Not str(...)-wrapped: nominate_v2 declares feature_profile as the
        # MarginFeatureProfile Literal, and metadata.get(...) is already
        # typed Any (json.loads), so passing it straight through type-checks
        # -- an explicit str() cast would widen it to plain str and fail.
        feature_profile=metadata.get("feature_profile"),
        min_train_games=int(min_train_games) if min_train_games else DEFAULT_MIN_TRAIN_GAMES,
    )
    nominee_id = result.game_id if result is not None else None

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    whole_week_pre_kickoff = bool(kickoffs.gt(recorded_at).all())
    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID)]
    already_ids = set(mine["game_id"].astype(str))

    already = nominee_id is not None and nominee_id in already_ids
    post_kickoff_skipped = nominee_id is not None and not already and not whole_week_pre_kickoff
    if nominee_id is None or already or not whole_week_pre_kickoff:
        fresh = card.iloc[0:0]
    else:
        fresh = card.loc[card["game_id"].astype(str).eq(nominee_id)]

    decisions = pd.DataFrame(
        {
            "recorded_at_utc": recorded_at,
            "challenger_id": CHALLENGER_ID,
            "config_fingerprint": observed_fingerprint,
            "source_artifact": forecast.name,
            "source_sha256": sha256_file(card_path),
            "forecast_created_at_utc": pd.to_datetime(
                metadata.get("created_at_utc"), utc=True, errors="coerce"
            ),
            "feature_profile": str(metadata.get("feature_profile")),
            "feature_table_sha256": str(observed_config.get("feature_table_sha256")),
            "game_id": fresh["game_id"].astype(str),
            "season": fresh["season"].astype(int),
            "week": fresh["week"].astype(int),
            "kickoff": kickoffs.loc[fresh.index],
            "away_team": fresh["away_team"].astype(str),
            "home_team": fresh["home_team"].astype(str),
            "pick_side": np.where(
                pd.to_numeric(fresh["home_cover_probability"], errors="coerce").ge(0.5),
                "HOME",
                "AWAY",
            ).astype(str),
            "bet_side": "PASS",
            "decision_home_spread": spreads.loc[fresh.index].astype(float),
            "edge": np.nan,
        }
    )
    if not decisions.empty:
        combined = (
            decisions if existing.empty else pd.concat([existing, decisions], ignore_index=True)
        )
        atomic_parquet(
            combined[list(CHALLENGER_DECISION_COLUMNS)], challenger_ledger_path(artifacts_root)
        )
        ledger_rows = len(combined)
    else:
        ledger_rows = len(existing)

    return {
        "challenger_id": CHALLENGER_ID,
        "season": season,
        "week": week,
        "source_artifact": forecast.name,
        "config_fingerprint": observed_fingerprint,
        "nominated_game_id": nominee_id,
        "nomination_tie_note": nomination_v2_tie_note(result) if result is not None else "",
        "recorded": len(decisions),
        "already_recorded": int(already),
        "post_kickoff_skipped": int(post_kickoff_skipped),
        "ledger_rows": int(ledger_rows),
    }


def record_nomination_v3_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append v3's weekly nominee to the prospective challenger ledger under
    :data:`CHALLENGER_ID_V3`, mirroring
    :func:`record_nomination_challenger_decisions` exactly except for which
    rule is fit (:func:`nominate_v3` instead of :func:`nominate_v2`) and
    which challenger id the row is written under.

    v3 is a SIDE-LEDGER-ONLY challenger (POL-09 audit,
    docs/best_pick_ranker.md "v3 audit", 2026-08-19): it is never read by
    ``publishing.py``, never affects ``is_best_pick`` on the primary paper
    -decision ledger, and does not touch ``NOMINATION_V2_ENABLED``. This
    function's only job is to accrue independent 2026 prospective evidence
    for v3 against both v1 (via the primary ledger's ``is_best_pick``) and
    v2 (via :data:`CHALLENGER_ID`'s own rows), so the season can eventually
    settle the tie-break question the historical audit could not: the entire
    measured historical edge over v2 traced to a single week of 103.

    Duplicates :func:`record_nomination_challenger_decisions`'s body rather
    than refactoring it to share code, for the same reason
    :func:`nominate_v3` duplicates :func:`nominate_v2`'s: the v2/live-card
    recording path must not change shape for a side-ledger-only addition.
    """

    entry = find_challenger(artifacts_root, CHALLENGER_ID_V3)
    status = str(entry.get("status"))
    if status != ACTIVE_CHALLENGER_STATUS:
        raise ValueError(
            f"Challenger {CHALLENGER_ID_V3!r} is registered as {status!r}; only "
            f"{ACTIVE_CHALLENGER_STATUS} challengers have picks recorded"
        )

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError(
            "No synchronized active ATS model is available to record nomination decisions from"
        )
    forecast = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast is None:
        raise ValueError("Active ATS model has no linked weekly forecast")
    metadata_path = forecast / "metadata.json"
    card_path = forecast / "recommendations.csv"
    if not metadata_path.is_file() or not card_path.is_file():
        raise ValueError(f"Linked weekly forecast is incomplete: {forecast}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("active_model_id") != active.get("model_id"):
        raise ValueError("Weekly forecast model ID does not match the active model")
    if metadata.get("synchronization_status") != "SYNCHRONIZED":
        raise ValueError("Weekly forecast is not synchronized with an evaluation")

    observed_config = artifact_model_config(metadata)
    declared_fingerprint = config_fingerprint(entry.get("model", {}))
    observed_fingerprint = config_fingerprint(observed_config)
    if declared_fingerprint != observed_fingerprint:
        raise DataContractError(
            f"Challenger {CHALLENGER_ID_V3!r} is registered pinned to configuration "
            f"fingerprint {declared_fingerprint}, but the current active forecast "
            f"{forecast} was produced with {observed_fingerprint}; the active model "
            "changed underneath this nomination rule -- re-register before recording"
        )

    card = pd.read_csv(card_path)
    required = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "away_team",
        "home_team",
        "spread_line",
        "home_cover_probability",
    }
    missing = sorted(required.difference(card.columns))
    if missing:
        raise DataContractError(f"Active forecast card is missing columns: {', '.join(missing)}")
    if card["game_id"].duplicated().any():
        raise DataContractError("Active forecast card contains duplicate games")
    spreads = pd.to_numeric(card["spread_line"], errors="coerce")
    if not np.isfinite(spreads.to_numpy(dtype=float)).all():
        raise DataContractError("Active forecast card has games without a decision spread")
    kickoffs = pd.to_datetime(card["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        raise DataContractError("Active forecast card has games without a kickoff timestamp")

    feature_table_path = observed_config.get("feature_table")
    if not feature_table_path:
        raise DataContractError(
            "Active forecast metadata carries no feature_table path to refit the "
            "alpha=2000 nomination candidate from"
        )
    features = pd.read_parquet(feature_table_path)
    season = int(card["season"].iloc[0])
    week = int(card["week"].iloc[0])
    min_train_games = metadata.get("min_train_games")
    result = nominate_v3(
        card,
        features,
        market_root=data_root / "market" / "raw",
        season=season,
        week=week,
        regressor=str(metadata.get("regressor", "ridge")),
        # Not str(...)-wrapped: nominate_v3 declares feature_profile as the
        # MarginFeatureProfile Literal, and metadata.get(...) is already
        # typed Any (json.loads), so passing it straight through type-checks
        # -- an explicit str() cast would widen it to plain str and fail.
        feature_profile=metadata.get("feature_profile"),
        min_train_games=int(min_train_games) if min_train_games else DEFAULT_MIN_TRAIN_GAMES,
    )
    nominee_id = result.game_id if result is not None else None

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    whole_week_pre_kickoff = bool(kickoffs.gt(recorded_at).all())
    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID_V3)]
    already_ids = set(mine["game_id"].astype(str))

    already = nominee_id is not None and nominee_id in already_ids
    post_kickoff_skipped = nominee_id is not None and not already and not whole_week_pre_kickoff
    if nominee_id is None or already or not whole_week_pre_kickoff:
        fresh = card.iloc[0:0]
    else:
        fresh = card.loc[card["game_id"].astype(str).eq(nominee_id)]

    decisions = pd.DataFrame(
        {
            "recorded_at_utc": recorded_at,
            "challenger_id": CHALLENGER_ID_V3,
            "config_fingerprint": observed_fingerprint,
            "source_artifact": forecast.name,
            "source_sha256": sha256_file(card_path),
            "forecast_created_at_utc": pd.to_datetime(
                metadata.get("created_at_utc"), utc=True, errors="coerce"
            ),
            "feature_profile": str(metadata.get("feature_profile")),
            "feature_table_sha256": str(observed_config.get("feature_table_sha256")),
            "game_id": fresh["game_id"].astype(str),
            "season": fresh["season"].astype(int),
            "week": fresh["week"].astype(int),
            "kickoff": kickoffs.loc[fresh.index],
            "away_team": fresh["away_team"].astype(str),
            "home_team": fresh["home_team"].astype(str),
            "pick_side": np.where(
                pd.to_numeric(fresh["home_cover_probability"], errors="coerce").ge(0.5),
                "HOME",
                "AWAY",
            ).astype(str),
            "bet_side": "PASS",
            "decision_home_spread": spreads.loc[fresh.index].astype(float),
            "edge": np.nan,
        }
    )
    if not decisions.empty:
        combined = (
            decisions if existing.empty else pd.concat([existing, decisions], ignore_index=True)
        )
        atomic_parquet(
            combined[list(CHALLENGER_DECISION_COLUMNS)], challenger_ledger_path(artifacts_root)
        )
        ledger_rows = len(combined)
    else:
        ledger_rows = len(existing)

    return {
        "challenger_id": CHALLENGER_ID_V3,
        "season": season,
        "week": week,
        "source_artifact": forecast.name,
        "config_fingerprint": observed_fingerprint,
        "nominated_game_id": nominee_id,
        "nomination_tie_note": nomination_v3_tie_note(result) if result is not None else "",
        "recorded": len(decisions),
        "already_recorded": int(already),
        "post_kickoff_skipped": int(post_kickoff_skipped),
        "ledger_rows": int(ledger_rows),
    }
