"""Surrogate-outcome validity: does opener-to-close line movement predict
which model is actually better at picking games, or does it just reward
market imitation?

Full predeclared methodology and results: ``docs/surrogate_outcome.md``
(2026-08-18). Headline verdict from that analysis: the movement label
carries a real but weak, market-imitation-prone signal about outcome
quality. It is NOT a free efficiency multiplier over direct forced-pick ATS
accuracy -- transfer-corrected, it needs *more* games than the direct label
to resolve the project's decision-relevant effect sizes, not fewer -- and it
must never be used alone to decide a promotion. See the doc for the
arithmetic; this module owns only the two reusable primitives that analysis
needed and nothing else builds on.

1. :func:`movement_agreement` -- per-game surrogate-outcome label: did a
   model's forced-pick DIRECTION match the direction the closing line
   actually moved? Sign convention matches
   :func:`nfl_ats.clv.opener_pick_evaluation`'s own movement-oracle
   (``open_move > 0`` <=> "pick home"), so any frame carrying that
   function's ``pick_home_at_open``/``open_move`` columns can be scored
   directly, with no extra fitting.
2. :func:`fit_movement_target_model` -- an ADVERSARIAL CONTROL, not a
   candidate ATS model. It is a weekly-refit ridge estimator fit to predict
   ``open_move`` itself (the market's own future adjustment) rather than
   ``market_residual`` (true fair value against the market). It exists
   solely to test whether the surrogate can be gamed by a model that is
   good at tracking movement while adding nothing over the opener --
   confirmed 2026-08-18: it beats every real candidate profile on the
   surrogate (63.6% movement agreement vs the best real candidate's 55.6%,
   probability_positive 1.00 it is higher) while very likely losing to the
   plain ``player`` baseline on real accuracy on the identical 1,380 games
   (51.78% vs 53.11%, probability_positive 0.756 the baseline is better).
   NEVER wire this into weekly production scoring, a candidate feature
   profile, or the active model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.clv import (
    CLOSE_LABEL_PRIORITY,
    HISTORICAL_CAPTURE_KIND,
    build_pairing_table,
    close_reference_table,
    pick_correct,
)
from nfl_ats.data import DataContractError
from nfl_ats.margin import MarginFeatureProfile, make_margin_estimator, margin_feature_columns
from nfl_ats.modeling import regular_season_rows

#: The adversarial control's training pool is restricted to games that
#: themselves carry a resolved opener/close movement label (2020-2025 only),
#: not the full pre-2020 archive every real market-residual model trains on.
#: A lower default gives it a fair within-window start instead of being
#: starved out of most of 2020.
DEFAULT_MOVEMENT_MIN_TRAIN_GAMES = 150


def movement_agreement(scored: pd.DataFrame) -> pd.Series:
    """Per-game surrogate label: did the forced pick match the movement direction?

    ``scored`` must carry ``pick_home_at_open`` (bool-like) and
    ``open_move`` (float, close-minus-open) columns -- exactly what
    :func:`nfl_ats.clv.opener_pick_evaluation` and
    :func:`fit_movement_target_model` both produce, so either can be scored
    with this one function. Returns a float Series of 1.0 (agree) / 0.0
    (disagree) / NaN, NaN where the line never moved (``open_move == 0``),
    matching :func:`nfl_ats.clv.pick_correct`'s push convention -- there is
    nothing to agree or disagree with when nothing moved.
    """

    required = {"pick_home_at_open", "open_move"}
    missing = sorted(required.difference(scored.columns))
    if missing:
        raise DataContractError(f"movement_agreement is missing columns: {', '.join(missing)}")
    move = pd.to_numeric(scored["open_move"], errors="coerce")
    agree = scored["pick_home_at_open"].astype(bool).eq(move.gt(0.0))
    return pd.Series(
        np.where(move.eq(0.0), np.nan, agree.astype(float)), index=scored.index, dtype=float
    )


def movement_agreement_rate(scored: pd.DataFrame) -> dict[str, float]:
    """Aggregate surrogate-outcome rate plus the game count it was computed over."""

    agreement = pd.to_numeric(movement_agreement(scored), errors="coerce").dropna()
    return {
        "movement_agreement_rate": float(agreement.mean()) if len(agreement) else float("nan"),
        "movement_agreement_games": float(len(agreement)),
    }


def fit_movement_target_model(
    root: Path,
    features: pd.DataFrame,
    *,
    feature_profile: MarginFeatureProfile = "player",
    ridge_alpha: float = 10.0,
    min_train_games: int = DEFAULT_MOVEMENT_MIN_TRAIN_GAMES,
    capture_kind: str = HISTORICAL_CAPTURE_KIND,
) -> pd.DataFrame:
    """Adversarial control: weekly-refit ridge fit directly to ``open_move``.

    Mirrors :func:`nfl_ats.clv.opener_pick_evaluation`'s leak-safe weekly
    -refit loop exactly -- training strictly precedes the scored week's
    first kickoff -- except the fitted target is the market's own realized
    opener-to-close movement rather than ``market_residual``. Training rows
    are additionally restricted to games that themselves carry a resolved
    movement label (paired opener + close), so this model can only ever
    train on strictly fewer, strictly later-starting games than the real
    market-residual model, which also sees the full pre-2020 archive.

    Returns one row per scored game: ``game_id``, ``season``, ``week``,
    ``result``, ``tue_open_home_spread``, ``open_move``,
    ``predicted_open_move``, ``pick_home_at_open`` (``predicted_open_move >
    0``, the same "pick home" convention
    :func:`nfl_ats.clv.opener_pick_evaluation` uses), ``margin_vs_open``,
    and ``correct_at_open`` (real ATS accuracy against the actual result --
    the outcome-skill side of the validity check). Pass the result straight
    to :func:`movement_agreement` for the surrogate-skill side.
    """

    feature_columns = margin_feature_columns("market_residual", feature_profile)
    required = {"game_id", "season", "week", "gameday", "result", *feature_columns}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"Movement-target fit is missing columns: {', '.join(missing)}")

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")

    pairing = build_pairing_table(
        root,
        capture_kind=capture_kind,
        labels=("tue_open", *CLOSE_LABEL_PRIORITY),
        schedule=frame,
    )
    if pairing.empty:
        raise ValueError(f"No {capture_kind!r} snapshots with decision quotes under {root}")
    close = close_reference_table(pairing, frame)
    tue_open = pairing.loc[pairing["decision_label"].eq("tue_open")][
        ["game_id", "season", "week", "home_spread"]
    ].rename(columns={"home_spread": "tue_open_home_spread"})
    paired = tue_open.merge(close, on="game_id", how="inner")
    outcomes = frame[["game_id", "result", "gameday"]].drop_duplicates("game_id")
    paired = paired.merge(outcomes, on="game_id", how="inner")
    paired = paired.loc[pd.to_numeric(paired["result"], errors="coerce").notna()].copy()
    if paired.empty:
        raise ValueError("No completed games have both a Tuesday opener and a close")
    paired["open_move"] = paired["close_home_spread"] - paired["tue_open_home_spread"]

    # Only paired games carry a known movement label -- the model literally
    # cannot see a movement outcome for anything else, unlike the real
    # market-residual model, which trains on every completed game back to
    # 2009 regardless of opener/close coverage.
    train_pool = frame.merge(paired[["game_id", "open_move"]], on="game_id", how="inner")

    scored_weeks: list[pd.DataFrame] = []
    for (_season, _week), group in paired.groupby(["season", "week"], sort=True):
        week_rows = frame.loc[frame["game_id"].isin(set(group["game_id"]))]
        if week_rows.empty:
            continue
        cutoff = week_rows["gameday"].min()
        training = train_pool.loc[train_pool["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        estimator = make_margin_estimator("ridge", 42, ridge_alpha=ridge_alpha)
        estimator.fit(training.loc[:, list(feature_columns)], training["open_move"])
        scoring = week_rows.merge(
            group[["game_id", "tue_open_home_spread", "close_home_spread", "open_move"]],
            on="game_id",
            how="inner",
        ).copy()
        predicted_move = np.asarray(
            estimator.predict(scoring.loc[:, list(feature_columns)]), dtype=float
        )
        scored = scoring[
            ["game_id", "season", "week", "result", "tue_open_home_spread", "open_move"]
        ].copy()
        scored["predicted_open_move"] = predicted_move
        scored_weeks.append(scored)
    if not scored_weeks:
        raise ValueError("No paired week had at least min_train_games completed training rows")

    result = pd.concat(scored_weeks, ignore_index=True)
    result["margin_vs_open"] = result["result"] - result["tue_open_home_spread"]
    result["pick_home_at_open"] = result["predicted_open_move"] > 0.0
    result["correct_at_open"] = pick_correct(result["pick_home_at_open"], result["margin_vs_open"])
    return result.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
