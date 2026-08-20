"""Era-weighted (half-life 8 seasons) challenger: MOD-14's selected training-recipe arm.

Research chain: ``docs/era_weighting_screen.md`` (MOD-14, predeclared
2026-08-19 before any arm's accuracy was scored). That document screened a
predeclared seven-arm grid (one uniform baseline plus six exponential-decay/
rolling-window candidates) on two independent instruments -- 12,500 free CFB
games (``nfl_ats.cfb_benchmark``) and the real production NFL recipe on 2,047
close-graded games -- varying **only** the per-training-row sample weight fed
to the frozen ``ridge_alpha=10.0`` fit, never the ridge penalty itself (a
different lever from MOD-06's shrinkage-toward-zero work, which this document
does not reopen). Both instruments independently selected the same arm,
**exponential season-decay with an 8-season half-life**:

- CFB, clean-core, week-blocked, paired vs. baseline: **+0.3470 accuracy
  points, 95% [-0.1804, +0.8633], probability_positive 0.8987**, n=8,933
  paired games.
- NFL, close grade, week-blocked, paired vs. baseline: **+0.6839 accuracy
  points, 95% [-0.5416, +1.9380], probability_positive 0.8505** (week-blocked)
  / **0.9533** (season-blocked, 95% [-0.0961, +1.4342]), n=2,047 games.

Both instruments' 95% intervals contain zero. Per AGENTS.md that is the
EXPECTED shape for a real small signal at this evaluator's ~2-point
resolution, never grounds to decline building a no-window-cost prospective
challenger -- neither admissible closing ground applies (no resolved wrong
sign, no positive-control bound was run), so both reads stay
``unresolved_below_power`` in the registry (twelve entries,
``era_weighting_cfb_half_life_8`` / ``era_weighting_nfl_half_life_8`` among
them). ``half_life_8`` was selected because it is the strongest-or-co-
strongest accuracy lean on every one of six cuts measured across both
instruments and never resolves negative on any secondary (Brier/log-loss/
margin-error) metric anywhere in ``docs/era_weighting_screen.md`` -- unlike
``half_life_2`` (resolved worse on NFL Brier/log-loss) and ``rolling_6``
(resolved worse on CFB Brier/log-loss/margin MAE/RMSE). This selection-among-
six is disclosed here, not hidden: this challenger tests ONLY the selected
arm, not the full grid, so its own prospective read should be understood as
confirming (or not) the grid's already-disclosed best-of-six pick, not a
fresh blind draw.

**A third, later look reverses sign at the OPENER grade -- disclosed here in
full, not smoothed over.** ``docs/era_weighting_screen.md`` Section 8
("Opener-grade information read", predeclared before running, measured
2026-08-19/20, ``registry/weak_signals.json:era_weighting_nfl_half_life_8_opener``)
re-scored the identical ``half_life_8`` vs. ``baseline`` pair on
``docs/opener_evaluation.md``'s 1,537-paired-game 2020-2025 archive, at the
production probability rule -- this project's actual decision-grade protocol
(AGENTS.md "grade the decision at the opener"), not the CLOSE grade Section 6
above used. At the opener, ``half_life_8`` leans NEGATIVE on every cut
measured: primary probability-rule accuracy **-0.3992 pts, 95% week-blocked
[-1.9450, +1.1921], probability_positive 0.2990** (n=1,503); secondary sign
rule -0.6653 pts, probability_positive 0.2031; Brier improvement -0.000238
pts (P+ 0.3646), log-loss improvement -0.000479 (P+ 0.3658) -- the OPPOSITE
sign from both the CFB screen and this same archive's own close-grade read
on the identical two arms (+0.1991 pts, P+ 0.5784 at close). No interval on
either side sits entirely below/above zero, so under the binding taxonomy
this is exactly as ``unresolved_below_power`` as the positive-leaning reads
above -- a negative point estimate crossing zero is not evidence of harm any
more than a positive one crossing zero was evidence of benefit, and this
divergence does NOT refute the mechanism or close the line. What it does
mean: this challenger's three predecessor looks (CFB, NFL close, NFL opener)
genuinely disagree on sign, all below power, and the 2026 prospective ledger
this module writes to is the next, independent look, not a formality on an
already-settled question.

**This module is a genuinely different challenger shape from the other
overlay challengers in this file's neighborhood** (``coach_fade_overlay``,
``injury_value_tilt_overlay``, ``surface_switch_tilt_overlay``, etc., which
all transform an already-fitted card's picks post-hoc) and from
``smooth_cdf_mapping_overlay``/``ecdf_mapping_incumbent_overlay`` (which
re-read the SAME fitted residual sample through a different probability
mapping, never refitting the ridge coefficients themselves). This challenger
actually **refits** the active recipe's ridge model every week with different
per-row sample weights -- a training-recipe-level challenger, the first of
its kind in this file. Mirrors ``smooth_cdf_mapping_overlay``'s
verify-reproduction-then-swap-one-thing discipline exactly: before any
half-life-weighted probability is trusted, the SAME leak-safe training rows
are refit with UNIFORM weights (``sample_weight=1`` for every row) and that
refit's Gaussian-read probability is required to reproduce the active card's
own ``home_cover_probability`` to floating-point precision
(``atol=1e-9``) -- proof the reimplementation is fitting the identical row
set, in the identical order, through the identical
``nfl_ats.margin.make_margin_estimator`` pipeline the production card used,
not a drifted reimplementation. Only then is the half-life-8-weighted refit's
Gaussian read trusted and swapped in.

**Weighting arithmetic, ported verbatim from ``scripts/era_weighting_lib.py``**
(a script-local module built for MOD-14's screen; per this task's environment
rules the production fitters are reused by import
(``nfl_ats.margin.make_margin_estimator``, ``MarginModel``) and the
sample-weight hook is copied here rather than imported from ``scripts/``,
which is not part of the installed package and not importable from ``src/``):
:func:`half_life_weights` and :func:`fit_weighted_ridge_margin` below are
byte-for-byte the same functions ``scripts/era_weighting_lib.py`` defines.
Decay is season-granularity only (no within-season decay): every training row
from the SAME season as the week being predicted carries weight 1.0
regardless of which week within that season it came from -- a deliberate
simplification stated in ``docs/era_weighting_screen.md`` Section 2, not
hidden here.

Nothing here touches ``margin.py``, ``outcomes.py``, ``pool.py``, or the
published card. ``apply_era_weighted_half_life_8_overlay`` is a pure function
of (predictions, features); :func:`record_era_weighted_half_life_8_challenger_decisions`
writes the reweighted arm's picks to the SEPARATE prospective challenger
ledger, dual-tracked against the active model, at no rotation-registry window
cost -- it mirrors
``smooth_cdf_mapping_overlay.record_smooth_cdf_mapping_challenger_decisions``
for the write-path guarantees (fingerprint pin, anti-backdating, append-only,
first-write-wins).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.calibration import smoothed_home_cover_probability
from nfl_ats.clv import refuse_if_outside_recording_lock_window
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.margin import (
    MarginFeatureProfile,
    MarginModel,
    make_margin_estimator,
    margin_feature_columns,
)
from nfl_ats.modeling import regular_season_rows
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

#: Registered in artifacts/prospective/challengers.json.
CHALLENGER_ID = "era_weighted_half_life_8"

#: MOD-14's selected arm (docs/era_weighting_screen.md "Walk-forward
#: discipline: which arm does the predeclared grid select?"). Frozen, not a
#: free parameter of this overlay.
HALF_LIFE_SEASONS = 8.0

_REQUIRED_PREDICTION_COLUMNS = frozenset(
    {"game_id", "season", "week", "home_team", "away_team", "spread_line", "home_cover_probability"}
)


# ---------------------------------------------------------------------------
# Ported verbatim from scripts/era_weighting_lib.py (MOD-14) -- see module
# docstring for why this is copied rather than imported.
# ---------------------------------------------------------------------------


def half_life_weights(
    seasons: npt.NDArray[np.float64], predict_season: int, half_life: float
) -> npt.NDArray[np.float64]:
    """Season-granularity exponential decay: weight 1.0 for the predicted season.

    ``elapsed`` is clamped at zero rather than allowed to go negative -- a
    training row can share the predicted season (earlier weeks of the same
    season) but never postdate it, since training is already restricted to
    strictly-earlier gamedays upstream of this function.
    """

    if half_life <= 0.0:
        raise ValueError("half_life must be positive")
    elapsed = np.clip(predict_season - seasons.astype(np.float64), 0.0, None)
    return np.power(0.5, elapsed / half_life)


def fit_weighted_ridge_margin(
    sorted_frame: pd.DataFrame,
    *,
    target: npt.NDArray[np.float64],
    feature_columns: tuple[str, ...],
    weights: npt.NDArray[np.float64],
    ridge_alpha: float = 10.0,
    distribution_fraction: float = 0.20,
    min_distribution_rows: int = 10,
    min_rows: int = 50,
    random_state: int = 42,
    model_name: str = "ridge",
) -> MarginModel:
    """Generic weighted mirror of ``nfl_ats.margin.fit_margin_model``.

    ``sorted_frame`` must already be chronologically sorted and filtered to
    completed, target-notna rows by the caller. ``target``/``weights`` are
    aligned 1:1 with ``sorted_frame``'s row order.

    The weight vector is routed to the Ridge step only
    (``regressor__sample_weight``); the imputer/scaler steps of
    ``make_margin_estimator``'s pipeline are fit unweighted, exactly as the
    frozen production pipeline already does for every existing (unweighted)
    arm.
    """

    if len(sorted_frame) < min_rows:
        raise ValueError(f"At least {min_rows} completed games are required to fit")
    if len(sorted_frame) != len(target) or len(sorted_frame) != len(weights):
        raise ValueError("sorted_frame, target, and weights must be the same length")
    if not 0.10 <= distribution_fraction < 0.5:
        raise ValueError("distribution_fraction must be in [0.10, 0.5)")

    distribution_rows = int(len(sorted_frame) * distribution_fraction)
    if distribution_rows < min_distribution_rows or len(sorted_frame) - distribution_rows < 40:
        raise ValueError("Not enough rows for an out-of-time residual distribution")
    split = len(sorted_frame) - distribution_rows

    columns = list(feature_columns)
    temporary = make_margin_estimator(model_name, random_state, ridge_alpha=ridge_alpha)
    temporary.fit(
        sorted_frame.iloc[:split].loc[:, columns],
        target[:split],
        regressor__sample_weight=weights[:split],
    )
    calibration_prediction = np.asarray(
        temporary.predict(sorted_frame.iloc[split:].loc[:, columns]), dtype=float
    )
    residuals = np.asarray(target[split:] - calibration_prediction, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < min_distribution_rows:
        raise ValueError("Out-of-time residual distribution has too few finite values")

    estimator = make_margin_estimator(model_name, random_state, ridge_alpha=ridge_alpha)
    estimator.fit(
        sorted_frame.loc[:, columns],
        target,
        regressor__sample_weight=weights,
    )
    return MarginModel(
        estimator=estimator,
        residuals=residuals,
        model_name=model_name,
        ridge_alpha=ridge_alpha if model_name == "ridge" else None,
        target="market_residual",
        feature_columns=feature_columns,
        training_rows=len(sorted_frame),
        distribution_rows=len(residuals),
        training_max_gameday=pd.to_datetime(sorted_frame["gameday"]).max().date().isoformat(),
    )


# ---------------------------------------------------------------------------
# Leak-safe training-frame construction, mirroring
# nfl_ats.outcomes._target_and_models_for_week / nfl_ats.margin.fit_margin_model
# exactly (row-for-row, order-for-order) so a uniform-weight refit reproduces
# the active card bit-for-bit.
# ---------------------------------------------------------------------------


def _target_values(frame: pd.DataFrame) -> pd.Series:
    """Mirrors ``nfl_ats.margin._target_values(frame, "market_residual")``."""

    return pd.to_numeric(frame["ats_margin"], errors="coerce")


def _leak_safe_training_frame(
    features: pd.DataFrame, *, season: int, week: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirrors ``nfl_ats.outcomes._target_and_models_for_week``'s target/cutoff
    logic exactly: the target week's games, and every strictly-earlier
    completed regular-season row as the training pool (not yet sorted or
    target-filtered -- that happens in :func:`_prepare_sorted_training`,
    matching ``fit_margin_model``'s own internal order)."""

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    target = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)].copy()
    if target.empty:
        raise ValueError(f"No games found for {season} week {week}")
    cutoff = target["gameday"].min()
    training = regular_season_rows(frame)
    training = training.loc[training["gameday"].lt(cutoff) & training["result"].notna()].copy()
    return target, training


def _prepare_sorted_training(training: pd.DataFrame) -> pd.DataFrame:
    """Mirrors ``nfl_ats.margin.fit_margin_model``'s own internal prep
    (target-notna filter, chronological sort, reset index) so the weighted
    fit sees the identical row set/order the production (unweighted) fit
    used."""

    prepared = training.loc[_target_values(training).notna()].copy()
    prepared["gameday"] = pd.to_datetime(prepared["gameday"], errors="raise")
    prepared = prepared.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return prepared


@dataclass(frozen=True)
class EraWeightedFlip:
    """One game whose forced pick moved sides under the half-life-8 refit."""

    game_id: str
    matchup: str
    from_side: str
    to_side: str
    baseline_probability: float
    era_weighted_probability: float


@dataclass(frozen=True)
class EraWeightedResult:
    """The overlay's effect on one or more weeks' cards.

    ``overlaid_predictions`` is ``predictions`` unchanged except for
    ``home_cover_probability`` on every row (the half-life-8 read replaces
    the baseline read for every game, not only flipped ones) -- every other
    column stays byte-identical, mirroring
    ``smooth_cdf_mapping_overlay.SmoothCdfMappingResult``.
    """

    overlaid_predictions: pd.DataFrame
    flips: tuple[EraWeightedFlip, ...]
    enabled: bool

    @property
    def flip_count(self) -> int:
        return len(self.flips)


def apply_era_weighted_half_life_8_overlay(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    regressor: str = "ridge",
    ridge_alpha: float = 10.0,
    feature_profile: MarginFeatureProfile = "weak_stack",
    min_train_games: int = 500,
    half_life: float = HALF_LIFE_SEASONS,
    enabled: bool = True,
) -> EraWeightedResult:
    """Refit the active recipe with half-life-8 season-decay sample weights.

    ``predictions`` may span more than one (season, week) group; each group's
    training pool is every strictly-earlier completed regular-season row
    (the same leak-safe cutoff ``score_outcome_week`` uses). For each group:

    1. Refit with UNIFORM weights (``sample_weight=1``) and require the
       Gaussian read off that refit to reproduce the supplied card's
       ``home_cover_probability`` to floating-point precision
       (``atol=1e-9``) -- proof this is fitting the identical leak-safe row
       set the active card was built from, not a drifted reimplementation.
       Raises :class:`~nfl_ats.data.DataContractError` rather than silently
       comparing against a moved target if the feature table or
       configuration has changed underneath it.
    2. Only then, refit with :func:`half_life_weights` (half-life 8 seasons)
       and replace ``home_cover_probability`` with THAT refit's Gaussian
       read.
    """

    missing = sorted(_REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise DataContractError(f"predictions is missing overlay columns: {', '.join(missing)}")

    base = predictions.reset_index(drop=True).copy()
    if not enabled or base.empty:
        return EraWeightedResult(base, (), enabled)

    base["game_id"] = base["game_id"].astype(str)
    feature_columns = margin_feature_columns("market_residual", feature_profile)
    era_weighted_probability_by_game: dict[str, float] = {}

    for _, group in base.groupby(["season", "week"], sort=True):
        season = int(group["season"].iloc[0])
        week = int(group["week"].iloc[0])
        target, training = _leak_safe_training_frame(features, season=season, week=week)
        if len(training) < min_train_games:
            raise DataContractError(
                f"Only {len(training)} eligible games precede season {season} week {week}; "
                f"need {min_train_games} to refit the era-weighted arm"
            )
        sorted_frame = _prepare_sorted_training(training)
        target_values = _target_values(sorted_frame).to_numpy(dtype=float)

        target = target.copy()
        target["game_id"] = target["game_id"].astype(str)
        if target["game_id"].duplicated().any():
            raise DataContractError(
                f"Refitting season {season} week {week} produced duplicate game IDs "
                "in the target universe"
            )
        target_indexed = target.set_index("game_id", drop=False)

        group_ids = group["game_id"].tolist()
        missing_games = sorted(set(group_ids).difference(target_indexed.index))
        if missing_games:
            raise DataContractError(
                f"Refitting season {season} week {week} is missing games from the "
                f"target universe: {', '.join(missing_games)} -- the feature table "
                "has likely drifted from the one that produced this card"
            )
        aligned = target_indexed.loc[group_ids]
        spread = aligned["spread_line"].to_numpy(dtype=float)

        # 1. Uniform-weight reproduction check (same discipline as
        #    smooth_cdf_mapping_overlay's ECDF check).
        uniform_weights = np.ones(len(sorted_frame), dtype=float)
        uniform_model = fit_weighted_ridge_margin(
            sorted_frame,
            target=target_values,
            feature_columns=feature_columns,
            weights=uniform_weights,
            ridge_alpha=ridge_alpha,
            model_name=regressor,
        )
        uniform_predicted = uniform_model.predict(aligned)
        uniform_centers = uniform_predicted["predicted_margin"].to_numpy(dtype=float)
        uniform_check = smoothed_home_cover_probability(
            uniform_model.residuals, uniform_centers, spread, method="gaussian"
        )
        supplied = group["home_cover_probability"].to_numpy(dtype=float)
        if not np.allclose(uniform_check, supplied, rtol=0.0, atol=1e-9):
            raise DataContractError(
                f"Uniform-weight refit for season {season} week {week} does not "
                "reproduce the supplied card's home_cover_probability -- the feature "
                "table or configuration has drifted from the one that produced this "
                "card, so the half-life-8 refit would not be a like-for-like comparison"
            )

        # 2. Half-life-8 season-decay weighted refit -- the arm being tested.
        seasons_arr = sorted_frame["season"].to_numpy(dtype=float)
        weights = half_life_weights(seasons_arr, predict_season=season, half_life=half_life)
        weighted_model = fit_weighted_ridge_margin(
            sorted_frame,
            target=target_values,
            feature_columns=feature_columns,
            weights=weights,
            ridge_alpha=ridge_alpha,
            model_name=regressor,
        )
        weighted_predicted = weighted_model.predict(aligned)
        weighted_centers = weighted_predicted["predicted_margin"].to_numpy(dtype=float)
        weighted_probability = smoothed_home_cover_probability(
            weighted_model.residuals, weighted_centers, spread, method="gaussian"
        )
        for game_id, probability in zip(group_ids, weighted_probability, strict=True):
            era_weighted_probability_by_game[game_id] = float(probability)

    overlaid = base.copy()
    overlaid["home_cover_probability"] = base["game_id"].map(era_weighted_probability_by_game)
    if overlaid["home_cover_probability"].isna().any():  # pragma: no cover - defensive
        raise DataContractError("Every game must receive a re-weighted probability")
    overlaid["home_cover_probability"] = overlaid["home_cover_probability"].astype(float)

    flips: list[EraWeightedFlip] = []
    for _, row in base.iterrows():
        game_id = str(row["game_id"])
        original_probability = float(row["home_cover_probability"])
        mapped_probability = era_weighted_probability_by_game[game_id]
        original_side = "HOME" if original_probability >= 0.5 else "AWAY"
        mapped_side = "HOME" if mapped_probability >= 0.5 else "AWAY"
        if mapped_side != original_side:
            flips.append(
                EraWeightedFlip(
                    game_id=game_id,
                    matchup=f"{row['away_team']} at {row['home_team']}",
                    from_side=original_side,
                    to_side=mapped_side,
                    baseline_probability=original_probability,
                    era_weighted_probability=mapped_probability,
                )
            )

    return EraWeightedResult(overlaid, tuple(flips), enabled)


def overlay_disclosure_note(result: EraWeightedResult) -> str:
    """Plain-language provenance sentence, mirroring
    ``smooth_cdf_mapping_overlay.overlay_disclosure_note``.

    Empty when the overlay is off or changed no picks this week. Not
    currently surfaced on the published card -- this challenger is dual-
    tracked only.
    """

    if not result.enabled or result.flip_count == 0:
        return ""
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{flip.matchup}: {flip.from_side} -> {flip.to_side}" for flip in result.flips
    )
    return (
        f"**Era-weighted (half-life 8) refit applied: {result.flip_count} pick{plural} "
        "flipped** (the active recipe refit with exponential season-decay sample "
        "weights, half-life 8 seasons, MOD-14's selected arm). "
        f"{detail}. See docs/era_weighting_screen.md. Prospective evidence only -- not "
        "applied to the published card."
    )


def _record_instant(now: datetime | None) -> pd.Timestamp:
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    return instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")


def record_era_weighted_half_life_8_challenger_decisions(
    artifacts_root: Path,
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append the half-life-8 refit's picks to the prospective challenger ledger.

    Mirrors ``smooth_cdf_mapping_overlay.record_smooth_cdf_mapping_challenger_decisions``
    exactly for the write-path guarantees: this is not a pre-generated
    ``margin-predict`` artifact under its own configuration fingerprint --
    its "model" IS the active model's own recipe, refit weekly with
    different sample weights -- so it reads the active model's own
    synchronized weekly forecast rather than searching
    ``artifacts/margin_predictions/``, and it refuses to record if the
    active model's live fingerprint no longer matches the snapshot this
    challenger was registered against (a promotion under this challenger's
    feet must not silently convert into "prospective evidence" for a
    different base model).
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
            "No synchronized active ATS model is available to record era-weighted decisions from"
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
            "changed underneath this challenger -- re-register before recording"
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

    feature_table = observed_config.get("feature_table")
    if not feature_table:
        raise ValueError("Active forecast metadata has no feature table path recorded")
    feature_path = Path(str(feature_table))
    if not feature_path.is_file():
        feature_path = data_root / "processed" / feature_path.name
    if not feature_path.is_file():
        raise ValueError(f"Feature table for the active model is not built yet: {feature_path}")
    features = pd.read_parquet(feature_path)

    result = apply_era_weighted_half_life_8_overlay(
        card,
        features,
        regressor=str(observed_config.get("regressor")),
        ridge_alpha=float(observed_config.get("ridge_alpha", 10.0)),
        feature_profile=str(observed_config.get("feature_profile")),  # type: ignore[arg-type]
        min_train_games=int(observed_config.get("min_train_games", 500)),
    )
    reweighted_card = result.overlaid_predictions

    recorded_at = _record_instant(now)
    refuse_if_outside_recording_lock_window(kickoffs, recorded_at, ledger="challenger")
    pre_kickoff = kickoffs.gt(recorded_at)
    existing = load_challenger_decisions(artifacts_root)
    mine = existing.loc[existing["challenger_id"].astype(str).eq(CHALLENGER_ID)]
    already = card["game_id"].astype(str).isin(set(mine["game_id"].astype(str)))
    keep = pre_kickoff & ~already
    fresh = reweighted_card.loc[keep]

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
        "season": int(card["season"].iloc[0]),
        "week": int(card["week"].iloc[0]),
        "source_artifact": forecast.name,
        "config_fingerprint": observed_fingerprint,
        "recorded": len(decisions),
        "already_recorded": int(already.sum()),
        "post_kickoff_skipped": int((~pre_kickoff & ~already).sum()),
        "ledger_rows": int(ledger_rows),
        "flip_count": result.flip_count,
        "flipped_game_ids": [flip.game_id for flip in result.flips],
    }
