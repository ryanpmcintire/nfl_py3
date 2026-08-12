"""Release-blocking invariants for weekly prediction artifacts.

Models are allowed to be uncertain.  Prediction artifacts are not allowed to
be internally inconsistent.  These checks independently recompute decisions,
market math, cutoffs, and method relationships before a card is published or
frozen.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.odds import choose_bet, market_hold, no_vig_probabilities

PREDICTION_SAFETY_VERSION = 1
VALID_BET_SIDES = frozenset(("HOME", "AWAY", "PASS"))
VALID_PICKS = frozenset(("HOME", "AWAY"))


class PredictionSafetyError(ValueError):
    """A prediction card failed an invariant and must not be published."""


@dataclass(frozen=True)
class PredictionSafetyAudit:
    version: int
    status: str
    card_type: str
    rows: int
    games: int
    checks_passed: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks_passed"] = list(self.checks_passed)
        payload["warnings"] = list(self.warnings)
        return payload


def _fail(check: str, message: str) -> None:
    raise PredictionSafetyError(f"Prediction safety check {check!r} failed: {message}")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], card_type: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        _fail("schema", f"{card_type} card is missing columns: {', '.join(missing)}")


def _all_finite_numeric(frame: pd.DataFrame, column: str) -> bool:
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().all() and np.isfinite(values.to_numpy(dtype=float)).all())


def _close(actual: Any, expected: Any, *, tolerance: float = 1e-9) -> bool:
    if pd.isna(actual) and pd.isna(expected):
        return True
    try:
        return bool(math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=1e-9))
    except (TypeError, ValueError):
        return False


def _validate_identity_and_cutoff(
    frame: pd.DataFrame,
    *,
    expected_season: int | None,
    expected_week: int | None,
    unique_columns: Sequence[str],
) -> tuple[str, ...]:
    checks: list[str] = []
    if frame.empty:
        _fail("nonempty", "the card contains no rows")
    if frame.duplicated(list(unique_columns)).any():
        examples = frame.loc[frame.duplicated(list(unique_columns), keep=False), "game_id"]
        _fail("unique_predictions", f"duplicate prediction keys include {examples.iloc[0]}")
    checks.append("unique_predictions")

    identifiers = frame["game_id"].astype("string").str.strip()
    if identifiers.isna().any() or identifiers.eq("").any():
        _fail("game_identity", "game_id must be nonempty")
    home = frame["home_team"].astype("string").str.strip()
    away = frame["away_team"].astype("string").str.strip()
    invalid_team = home.isna() | away.isna() | home.eq("") | away.eq("") | home.eq(away)
    if invalid_team.any():
        game = identifiers[invalid_team].iloc[0]
        _fail("game_identity", f"invalid home/away identity for {game}")
    checks.append("game_identity")

    seasons = pd.to_numeric(frame["season"], errors="coerce")
    weeks = pd.to_numeric(frame["week"], errors="coerce")
    if seasons.isna().any() or weeks.isna().any():
        _fail("card_scope", "season and week must be numeric")
    if seasons.nunique() != 1 or weeks.nunique() != 1:
        _fail("card_scope", "a weekly card must contain exactly one season and week")
    if expected_season is not None and not seasons.eq(expected_season).all():
        _fail("card_scope", f"rows do not all belong to expected season {expected_season}")
    if expected_week is not None and not weeks.eq(expected_week).all():
        _fail("card_scope", f"rows do not all belong to expected week {expected_week}")
    checks.append("card_scope")

    gamedays = pd.to_datetime(frame["gameday"], errors="coerce")
    training_cutoffs = pd.to_datetime(frame["train_max_gameday"], errors="coerce")
    if gamedays.isna().any() or training_cutoffs.isna().any():
        _fail("training_cutoff", "gameday and train_max_gameday must be parseable")
    invalid_cutoff = training_cutoffs.dt.normalize().ge(gamedays.dt.normalize())
    if invalid_cutoff.any():
        _fail(
            "training_cutoff",
            f"training is not strictly earlier than {identifiers[invalid_cutoff].iloc[0]}",
        )
    train_rows = pd.to_numeric(frame["train_rows"], errors="coerce")
    if train_rows.isna().any() or train_rows.le(0).any() or (train_rows % 1).ne(0).any():
        _fail("training_cutoff", "train_rows must contain positive integers")
    checks.append("training_cutoff")
    return tuple(checks)


def _validate_market_inputs(
    frame: pd.DataFrame, *, prospective: bool
) -> tuple[list[str], list[str]]:
    checks: list[str] = []
    warnings: list[str] = []
    spreads = pd.to_numeric(frame["spread_line"], errors="coerce")
    if spreads.isna().any() or not np.isfinite(spreads).all():
        _fail("market_inputs", "every spread must be finite")
    if spreads.abs().gt(40).any():
        _fail("market_inputs", "an NFL spread exceeds the 40-point plausibility bound")

    missing_prices = pd.Series(False, index=frame.index)
    for column in ("home_spread_odds", "away_spread_odds"):
        prices = pd.to_numeric(frame[column], errors="coerce")
        missing_prices |= prices.isna() | ~np.isfinite(prices)
        present = prices.loc[prices.notna() & np.isfinite(prices)]
        invalid = present.eq(0) | present.abs().lt(100) | present.abs().gt(100_000)
        if invalid.any():
            _fail(
                "market_inputs",
                f"{column} contains values inconsistent with American odds",
            )
    if prospective and missing_prices.any():
        _fail("market_inputs", "prospective cards require both spread prices for every game")
    if missing_prices.any():
        warnings.append(
            f"{int(missing_prices.sum())} rows use the explicit -110 fallback for missing prices"
        )
    checks.append("market_inputs")
    return checks, warnings


def _validate_decisions(frame: pd.DataFrame, min_edge: float) -> tuple[str, ...]:
    if min_edge < 0:
        _fail("decision_policy", "min_edge cannot be negative")
    probabilities = pd.to_numeric(frame["home_cover_probability"], errors="coerce")
    if probabilities.isna().any() or not np.isfinite(probabilities).all():
        _fail("probabilities", "home cover probabilities must be finite")
    if not probabilities.between(0.0, 1.0, inclusive="both").all():
        _fail("probabilities", "home cover probabilities must lie in [0, 1]")
    if not set(frame["pick"].astype(str)).issubset(VALID_PICKS):
        _fail("decision_policy", "pick contains a value other than HOME or AWAY")
    expected_pick = np.where(probabilities.ge(0.5), "HOME", "AWAY")
    if not np.array_equal(frame["pick"].astype(str).to_numpy(), expected_pick):
        _fail("decision_policy", "pick is inconsistent with home_cover_probability")
    if not set(frame["bet_side"].astype(str)).issubset(VALID_BET_SIDES):
        _fail("decision_policy", "bet_side contains an invalid value")

    for probability, (_, row) in zip(probabilities, frame.iterrows(), strict=True):
        home_odds = pd.to_numeric(pd.Series([row["home_spread_odds"]]), errors="coerce").iloc[0]
        away_odds = pd.to_numeric(pd.Series([row["away_spread_odds"]]), errors="coerce").iloc[0]
        decision = choose_bet(
            float(probability),
            float(home_odds) if pd.notna(home_odds) else None,
            float(away_odds) if pd.notna(away_odds) else None,
            min_edge=min_edge,
        )
        if str(row["bet_side"]) != decision.side:
            _fail("decision_policy", f"bet side is inconsistent for {row['game_id']}")
        if not _close(row["edge"], decision.edge):
            _fail("decision_policy", f"edge is inconsistent for {row['game_id']}")
        if not _close(row["bet_odds"], decision.odds):
            _fail("decision_policy", f"selected price is inconsistent for {row['game_id']}")
        if not _close(row["break_even_probability"], decision.break_even_probability):
            _fail(
                "decision_policy",
                f"break-even probability is inconsistent for {row['game_id']}",
            )
        expected_market = no_vig_probabilities(
            float(home_odds) if pd.notna(home_odds) else None,
            float(away_odds) if pd.notna(away_odds) else None,
        )[0]
        expected_hold = market_hold(
            float(home_odds) if pd.notna(home_odds) else None,
            float(away_odds) if pd.notna(away_odds) else None,
        )
        if not _close(row["market_home_no_vig_probability"], expected_market):
            _fail("market_math", f"no-vig probability is inconsistent for {row['game_id']}")
        if not _close(row["market_hold"], expected_hold):
            _fail("market_math", f"market hold is inconsistent for {row['game_id']}")
    return ("probabilities", "decision_policy", "market_math")


def _prospective_checks(
    frame: pd.DataFrame,
    *,
    created_at: datetime | None,
) -> tuple[list[str], list[str]]:
    checks: list[str] = []
    warnings: list[str] = []
    instant = created_at or datetime.now(UTC)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    instant = instant.astimezone(UTC)
    kickoffs = pd.to_datetime(frame["kickoff"], errors="coerce", utc=True)
    if kickoffs.isna().any():
        _fail("pregame_timing", "kickoff is missing or unparseable")
    if kickoffs.le(instant).any():
        game = frame.loc[kickoffs.le(instant), "game_id"].iloc[0]
        _fail("pregame_timing", f"{game} has already kicked off")
    outcome_columns = [
        column for column in ("home_cover", "result", "ats_margin") if column in frame
    ]
    if outcome_columns and frame[outcome_columns].notna().any(axis=None):
        _fail("outcome_embargo", "a prospective card contains a known outcome")
    checks.extend(("pregame_timing", "outcome_embargo"))

    timestamp_column = next(
        (
            column
            for column in ("market_observed_at_utc", "line_observed_at_utc", "observed_at_utc")
            if column in frame
        ),
        None,
    )
    if timestamp_column is None or frame[timestamp_column].isna().all():
        warnings.append("market quote timestamp is unavailable; line freshness is not verified")
    else:
        observed = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
        if observed.isna().any():
            _fail("market_timing", "some market observation timestamps are missing")
        if observed.gt(instant).any() or observed.ge(kickoffs).any():
            _fail("market_timing", "a market observation occurs after freeze time or kickoff")
        checks.append("market_timing")
    return checks, warnings


def _feature_checks(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    if not feature_columns:
        return [], []
    _require_columns(frame, feature_columns, "ATS")
    original = frame.loc[:, list(feature_columns)]
    features = original.apply(pd.to_numeric, errors="coerce")
    if (original.notna() & features.isna()).any(axis=None):
        _fail("model_inputs", "a nonmissing model input is not numeric")
    finite_or_missing = features.isna() | np.isfinite(features)
    if not finite_or_missing.all(axis=None):
        _fail("model_inputs", "a model input contains positive or negative infinity")
    missing_fraction = features.isna().mean(axis=1)
    if missing_fraction.gt(0.5).any():
        game = frame.loc[missing_fraction.gt(0.5), "game_id"].iloc[0]
        _fail("model_inputs", f"more than half of selected inputs are missing for {game}")
    warnings: list[str] = []
    if missing_fraction.gt(0.25).any():
        warnings.append(
            f"{int(missing_fraction.gt(0.25).sum())} rows are missing more than 25% of model inputs"
        )
    return ["model_inputs"], warnings


def validate_prediction_card(
    predictions: pd.DataFrame,
    *,
    min_edge: float,
    expected_season: int | None = None,
    expected_week: int | None = None,
    feature_columns: Sequence[str] | None = None,
    prospective: bool = False,
    created_at: datetime | None = None,
) -> PredictionSafetyAudit:
    """Validate a direct ATS card and independently recompute its decisions."""

    required = (
        "game_id",
        "season",
        "week",
        "gameday",
        "away_team",
        "home_team",
        "spread_line",
        "away_spread_odds",
        "home_spread_odds",
        "home_cover_probability",
        "pick",
        "bet_side",
        "edge",
        "bet_odds",
        "break_even_probability",
        "market_home_no_vig_probability",
        "market_hold",
        "train_rows",
        "train_max_gameday",
    )
    _require_columns(predictions, required, "ATS")
    if prospective:
        _require_columns(predictions, ("kickoff",), "prospective ATS")
    checks = list(
        _validate_identity_and_cutoff(
            predictions,
            expected_season=expected_season,
            expected_week=expected_week,
            unique_columns=("game_id",),
        )
    )
    market_checks, warnings = _validate_market_inputs(predictions, prospective=prospective)
    checks.extend(market_checks)
    checks.extend(_validate_decisions(predictions, min_edge))
    feature_checks, feature_warnings = _feature_checks(predictions, feature_columns)
    checks.extend(feature_checks)
    warnings.extend(feature_warnings)
    if prospective:
        timing_checks, timing_warnings = _prospective_checks(
            predictions,
            created_at=created_at,
        )
        checks.extend(timing_checks)
        warnings.extend(timing_warnings)
    status = "PASS_WITH_WARNINGS" if warnings else "PASS"
    return PredictionSafetyAudit(
        version=PREDICTION_SAFETY_VERSION,
        status=status,
        card_type="direct_ats",
        rows=len(predictions),
        games=int(predictions["game_id"].nunique()),
        checks_passed=tuple(dict.fromkeys(checks)),
        warnings=tuple(warnings),
    )


def validate_outcome_prediction_card(
    predictions: pd.DataFrame,
    *,
    min_edge: float,
    expected_methods: Sequence[str],
    expected_season: int | None = None,
    expected_week: int | None = None,
) -> PredictionSafetyAudit:
    """Validate the five-method straight-up, margin, and ATS weekly card."""

    required = (
        "game_id",
        "season",
        "week",
        "gameday",
        "away_team",
        "home_team",
        "spread_line",
        "away_spread_odds",
        "home_spread_odds",
        "method",
        "home_win_probability",
        "home_cover_probability",
        "predicted_margin",
        "fair_spread",
        "predicted_market_residual",
        "margin_lower_50",
        "margin_upper_50",
        "margin_lower_80",
        "margin_upper_80",
        "bet_side",
        "edge",
        "bet_odds",
        "break_even_probability",
        "train_rows",
        "train_max_gameday",
    )
    _require_columns(predictions, required, "outcome")
    checks = list(
        _validate_identity_and_cutoff(
            predictions,
            expected_season=expected_season,
            expected_week=expected_week,
            unique_columns=("game_id", "method"),
        )
    )
    market_checks, warnings = _validate_market_inputs(predictions, prospective=False)
    checks.extend(market_checks)

    expected = set(expected_methods)
    observed = set(predictions["method"].astype(str))
    if observed != expected:
        message = f"expected methods {sorted(expected)}, observed {sorted(observed)}"
        _fail("method_coverage", message)
    method_counts = predictions.groupby("game_id")["method"].nunique()
    if not method_counts.eq(len(expected)).all():
        _fail("method_coverage", "not every game has exactly one row for every method")
    checks.append("method_coverage")

    margin_methods = predictions["method"].isin(("market", "fair_margin", "market_residual"))
    margin_rows = predictions.loc[margin_methods]
    for column in (
        "predicted_margin",
        "fair_spread",
        "predicted_market_residual",
        "margin_lower_50",
        "margin_upper_50",
        "margin_lower_80",
        "margin_upper_80",
    ):
        if not _all_finite_numeric(margin_rows, column):
            _fail("margin_distribution", f"{column} must be finite for margin methods")
    ordered = (
        pd.to_numeric(margin_rows["margin_lower_80"]).le(
            pd.to_numeric(margin_rows["margin_lower_50"])
        )
        & pd.to_numeric(margin_rows["margin_lower_50"]).le(
            pd.to_numeric(margin_rows["margin_upper_50"])
        )
        & pd.to_numeric(margin_rows["margin_upper_50"]).le(
            pd.to_numeric(margin_rows["margin_upper_80"])
        )
    )
    if not ordered.all():
        _fail("margin_distribution", "prediction intervals are not monotonically nested")
    if not np.isclose(
        pd.to_numeric(margin_rows["fair_spread"]),
        pd.to_numeric(margin_rows["predicted_margin"]),
    ).all():
        _fail("margin_identity", "fair_spread does not equal predicted_margin")
    residual_identity = pd.to_numeric(margin_rows["predicted_margin"]) - pd.to_numeric(
        margin_rows["spread_line"]
    )
    if not np.isclose(
        pd.to_numeric(margin_rows["predicted_market_residual"]), residual_identity
    ).all():
        _fail("margin_identity", "market residual does not equal prediction minus spread")
    market_rows = predictions.loc[predictions["method"].eq("market")]
    if not np.isclose(
        pd.to_numeric(market_rows["predicted_margin"]),
        pd.to_numeric(market_rows["spread_line"]),
    ).all():
        _fail("market_baseline", "market predicted margin does not equal the market spread")
    checks.extend(("margin_distribution", "margin_identity", "market_baseline"))

    win_required = predictions["method"].ne("direct_ats")
    cover_required = predictions["method"].ne("straight_up")
    for column, required_mask in (
        ("home_win_probability", win_required),
        ("home_cover_probability", cover_required),
    ):
        values = pd.to_numeric(predictions.loc[required_mask, column], errors="coerce")
        if values.isna().any() or not values.between(0.0, 1.0, inclusive="both").all():
            _fail("probabilities", f"{column} is invalid for a method that requires it")
    checks.append("probabilities")

    for _, row in predictions.iterrows():
        probability = pd.to_numeric(
            pd.Series([row["home_cover_probability"]]), errors="coerce"
        ).iloc[0]
        if pd.isna(probability):
            if row["bet_side"] != "PASS" or any(
                pd.notna(row[column]) for column in ("edge", "bet_odds", "break_even_probability")
            ):
                _fail("decision_policy", f"non-ATS method has a wager for {row['game_id']}")
            continue
        decision = choose_bet(
            float(probability),
            row.get("home_spread_odds"),
            row.get("away_spread_odds"),
            min_edge=min_edge,
        )
        if str(row["bet_side"]) != decision.side or not _close(row["edge"], decision.edge):
            _fail("decision_policy", f"decision is inconsistent for {row['game_id']}")
        if not _close(row["bet_odds"], decision.odds) or not _close(
            row["break_even_probability"], decision.break_even_probability
        ):
            _fail("decision_policy", f"decision price is inconsistent for {row['game_id']}")
    checks.append("decision_policy")
    return PredictionSafetyAudit(
        version=PREDICTION_SAFETY_VERSION,
        status="PASS_WITH_WARNINGS" if warnings else "PASS",
        card_type="outcomes",
        rows=len(predictions),
        games=int(predictions["game_id"].nunique()),
        checks_passed=tuple(dict.fromkeys(checks)),
        warnings=tuple(warnings),
    )
