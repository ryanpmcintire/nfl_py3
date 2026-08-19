"""What actually gets submitted for one weekly forecast -- the single shared
implementation of "overlay applied, Best Pick nominated."

A weekly forecast artifact (``recommendations.csv``) holds the model's OWN,
un-overlaid picks. Two levers change what actually gets submitted to the pool
without ever rewriting that file:

1. The year-1-coach-fade overlay (:mod:`nfl_ats.coach_fade_overlay`) flips a
   handful of picks post-prediction (weeks 1-8, "clean case" only).
2. Best Pick nomination v2 (:mod:`nfl_ats.best_pick_nomination`) usually
   replaces the incumbent ``sweep_robustness`` signal (v1) for choosing WHICH
   game gets the week's bonus pick.

Before this module existed, that composition was implemented three times:
``nfl_ats.publishing`` (the tracked Markdown card), ``nfl_ats.public_board``
(the public GitHub Pages site, which had NEITHER lever wired in at all --
2026-08-19 incident: the site showed BAL at IND while the published card had
already flipped that pick to IND, and nominated the wrong Best Pick), and
``nfl_ats.dashboard.data`` (the internal Streamlit dashboard, a near-duplicate
of ``publishing``'s logic). Three copies of the same decision is mirror-drift
waiting to happen. This module is the one place the decision is made;
``publishing.py``, ``public_board.py``, and ``dashboard/data.py`` all call
into it.

Best Pick selection ALWAYS runs on the UN-overlaid predictions (both rules):
the overlay must never influence which game is nominated, only which side a
game's forced pick lands on. That keeps the two levers independently
measured rather than silently composed -- see
:mod:`nfl_ats.best_pick_nomination`'s module docstring for the same property
stated from the other direction.

Both levers degrade to a no-op / the incumbent rule whenever their inputs are
unavailable (no ``data_root``, no local snapshot, a malformed frame): a
missing input must degrade the card, never fail a caller.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.best_pick import best_pick_tie_note, select_best_pick
from nfl_ats.best_pick_nomination import (
    NOMINATION_V2_ENABLED,
    NominationV2Result,
    nominate_v2,
    nomination_v2_disclosure_note,
    nomination_v2_tie_note,
)
from nfl_ats.coach_fade_overlay import (
    OVERLAY_WEEK_MAX,
    OverlayResult,
    apply_coach_fade_overlay,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.prospective_scoring import artifact_model_config
from nfl_ats.snapshots import latest_snapshot, load_snapshot

# ---------------------------------------------------------------------------
# The coach-fade overlay: which side actually gets picked
# ---------------------------------------------------------------------------


def _disabled_overlay(predictions: pd.DataFrame) -> OverlayResult:
    """A no-op overlay result, bypassing ``apply_coach_fade_overlay``'s own
    column contract entirely -- a disabled overlay has nothing to validate,
    and a caller that never asked for the overlay (``data_root=None``) should
    not need overlay-specific columns on ``predictions`` just to render."""

    return OverlayResult(predictions.reset_index(drop=True).copy(), (), (), OVERLAY_WEEK_MAX, False)


def resolve_overlay(predictions: pd.DataFrame, data_root: Path | None) -> OverlayResult:
    """The year-1-coach fade overlay (docs/coach_fade_overlay.md), or a no-op.

    Degrades to a disabled overlay -- never raises -- when ``data_root`` is
    omitted, no local schedule snapshot is available, or the
    predictions/schedule frames do not carry what the overlay needs (e.g. a
    minimal fixture with no ``season``/``week`` columns).
    """

    if data_root is None:
        return _disabled_overlay(predictions)
    try:
        schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    except FileNotFoundError:
        return _disabled_overlay(predictions)
    try:
        return apply_coach_fade_overlay(predictions, schedules)
    except DataContractError:
        return _disabled_overlay(predictions)


# ---------------------------------------------------------------------------
# Best Pick nomination: which GAME gets the week's bonus pick
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BestPickNomination:
    """Both rules' weekly nominations, plus which one is actually played.

    ``v1_game_id``/``v1_tie_note`` and ``v2_result`` are ALWAYS populated
    when computable, regardless of the switch, so a caller can disclose both
    nominations even when only one is played (owner decision 2026-08-18,
    POL-09). ``active_*`` is whichever rule is actually marked: v2 when
    :data:`nfl_ats.best_pick_nomination.NOMINATION_V2_ENABLED` is on AND v2
    could be computed this week, the incumbent v1 rule otherwise.
    """

    v1_game_id: str | None
    v1_tie_note: str
    v2_result: NominationV2Result | None
    active_rule: str  # "v1" | "v2"
    active_game_id: str | None
    active_tie_note: str
    method_note: str


def _v1_nomination(predictions: pd.DataFrame, sweep: pd.DataFrame) -> tuple[str | None, str]:
    """The v1 (``sweep_robustness``) nomination, or ``None`` when it cannot
    be computed, plus a disclosure sentence when that pick is an undisclosed
    tie. Regular season only, and silent when ``sweep`` is empty/missing the
    columns ``select_best_pick`` needs -- a missing Best Pick must degrade
    the card, never fail the caller.
    """

    if (
        "game_type" in predictions.columns
        and not predictions["game_type"].astype(str).eq("REG").all()
    ):
        return None, ""
    if sweep is None or sweep.empty:
        return None, ""
    if "method" in sweep.columns and "method" in predictions.columns:
        sweep = sweep.loc[sweep["method"].isin(set(predictions["method"].astype(str)))]
    best_pick_id = select_best_pick(predictions, sweep)
    if best_pick_id is None:
        return None, ""
    return best_pick_id, best_pick_tie_note(predictions, sweep)


@dataclass(frozen=True)
class V2NominationInputs:
    """Primitive, cacheable inputs for v2 nomination, extracted from a
    weekly forecast's own ``metadata.json``.

    Every field is a plain string/int/float so a caller (the dashboard) can
    use them as a cache key without hashing a ``Path`` or a ``dict`` --
    :func:`nfl_ats.dashboard.data.resolve_best_pick`'s reason for existing as
    a thin wrapper rather than calling :func:`compute_v2_nomination` directly.
    """

    feature_table: str
    market_root: str
    season: int
    week: int
    regressor: str
    feature_profile: Any
    min_train_games: int


def v2_nomination_inputs(
    metadata: Mapping[str, Any], data_root: Path | None
) -> V2NominationInputs | None:
    """Extract v2's inputs from ``metadata``, or ``None`` when v2 cannot run
    this week (no ``data_root``, or the forecast's metadata is missing
    season/week/feature_profile/its own recorded feature table) -- callers
    should treat ``None`` exactly like "no data_root": degrade to v1.
    """

    if data_root is None:
        return None
    season = metadata.get("season")
    week = metadata.get("week")
    feature_profile = metadata.get("feature_profile")
    if season is None or week is None or not feature_profile:
        return None
    feature_table = artifact_model_config(metadata).get("feature_table")
    if not feature_table:
        return None
    min_train_games = metadata.get("min_train_games")
    return V2NominationInputs(
        feature_table=str(feature_table),
        market_root=str(data_root / "market" / "raw"),
        season=int(season),
        week=int(week),
        regressor=str(metadata.get("regressor", "ridge")),
        feature_profile=feature_profile,
        min_train_games=int(min_train_games) if min_train_games else DEFAULT_MIN_TRAIN_GAMES,
    )


def compute_v2_nomination(
    predictions: pd.DataFrame,
    metadata: Mapping[str, Any],
    data_root: Path | None,
    *,
    nominate_v2_fn: Callable[..., NominationV2Result | None] = nominate_v2,
) -> NominationV2Result | None:
    """The plain, uncached v2 computation -- what every one-shot caller
    (publish, static-site render) wants. ``nominate_v2_fn`` is injectable so
    a caller can swap in a cached/mocked implementation (tests patch this at
    the CALLER's own module scope, e.g. ``publishing.nominate_v2``, and pass
    it through here explicitly, rather than patching this module).

    Degrades to ``None`` -- never raises -- when the feature table cannot be
    read, or ``nominate_v2_fn`` itself raises ``ValueError``/
    ``DataContractError`` (e.g. not enough walk-forward training history
    yet).
    """

    inputs = v2_nomination_inputs(metadata, data_root)
    if inputs is None:
        return None
    try:
        features = pd.read_parquet(inputs.feature_table)
    except (OSError, ValueError):
        return None
    try:
        return nominate_v2_fn(
            predictions,
            features,
            market_root=Path(inputs.market_root),
            season=inputs.season,
            week=inputs.week,
            regressor=inputs.regressor,
            feature_profile=inputs.feature_profile,
            min_train_games=inputs.min_train_games,
        )
    except (ValueError, DataContractError):
        return None


_UNSET: Any = object()


def resolve_nomination(
    predictions: pd.DataFrame,
    sweep: pd.DataFrame,
    metadata: Mapping[str, Any],
    data_root: Path | None,
    *,
    v2_result: NominationV2Result | None = _UNSET,
    nominate_v2_fn: Callable[..., NominationV2Result | None] = nominate_v2,
) -> BestPickNomination:
    """Both rules' nominations for one week, plus which one is played.

    ``sweep`` should already be filtered to the active method (every caller
    already does this for its own plotting purposes); v1 defensively
    re-filters to ``predictions``'s own method values in case it is not.

    ``v2_result`` lets a caller supply an ALREADY-COMPUTED v2 nomination
    (the dashboard's cached cross-book dispersion scan -- see
    :mod:`nfl_ats.dashboard.data`) instead of having this function compute
    one fresh. Omit it (the default sentinel) to compute v2 the plain,
    uncached way via :func:`compute_v2_nomination`.
    """

    v1_id, v1_tie = _v1_nomination(predictions, sweep)
    resolved_v2 = (
        compute_v2_nomination(predictions, metadata, data_root, nominate_v2_fn=nominate_v2_fn)
        if v2_result is _UNSET
        else v2_result
    )

    if NOMINATION_V2_ENABLED and resolved_v2 is not None:
        return BestPickNomination(
            v1_game_id=v1_id,
            v1_tie_note=v1_tie,
            v2_result=resolved_v2,
            active_rule="v2",
            active_game_id=resolved_v2.game_id,
            active_tie_note=nomination_v2_tie_note(resolved_v2),
            method_note=nomination_v2_disclosure_note(resolved_v2),
        )
    return BestPickNomination(
        v1_game_id=v1_id,
        v1_tie_note=v1_tie,
        v2_result=resolved_v2,
        active_rule="v1",
        active_game_id=v1_id,
        active_tie_note=v1_tie,
        method_note="",
    )


# ---------------------------------------------------------------------------
# Both levers together -- the convenience entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CardView:
    """What actually gets submitted for one weekly forecast: overlay-applied
    picks, the Best Pick nomination, and disclosure notes."""

    predictions: pd.DataFrame  # overlay.overlaid_predictions, for convenience
    overlay: OverlayResult
    nomination: BestPickNomination

    @property
    def best_pick_game_id(self) -> str | None:
        return self.nomination.active_game_id


def resolve_card_view(
    predictions: pd.DataFrame,
    sweep: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    data_root: Path | None = None,
    nominate_v2_fn: Callable[..., NominationV2Result | None] = nominate_v2,
) -> CardView:
    """Apply the coach-fade overlay and resolve the Best Pick nomination in
    one call -- the single source of truth for "what actually gets played"
    behind the published Markdown card, the public site, and the dashboard.
    """

    nomination = resolve_nomination(
        predictions, sweep, metadata, data_root, nominate_v2_fn=nominate_v2_fn
    )
    overlay = resolve_overlay(predictions, data_root)
    return CardView(overlay.overlaid_predictions, overlay, nomination)


__all__ = [
    "BestPickNomination",
    "CardView",
    "V2NominationInputs",
    "compute_v2_nomination",
    "resolve_card_view",
    "resolve_nomination",
    "resolve_overlay",
    "v2_nomination_inputs",
]
