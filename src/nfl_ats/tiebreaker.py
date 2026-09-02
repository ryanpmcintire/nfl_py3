"""Pool tiebreaker: a defensible final-score guess for one game.

The pool breaks ties on the final score of the week's LAST game (owner,
2026-09-01; Week 1 that is DEN @ KC on Monday night). This module produces
the guess and, just as importantly, states how accurate such a guess can be.

Method, and why it is deliberately modest
-----------------------------------------
The market's own spread and total are the strong baseline this project holds
every model to, so the guess starts from the market-implied score::

    home = (total + home_expected_margin) / 2
    away = (total - home_expected_margin) / 2

and then calibrates it against every completed game with a recorded spread
and total (4,630 games, 2009-2025): the *neighborhood* of historically
similar market shapes supplies the median actual total, median actual home
margin, and the most common exact final scores.

The neighborhood is KERNEL-WEIGHTED, not a hard window, and that choice is
load-bearing rather than cosmetic. Quoted spreads and totals are quantized
to half points, so a hard +/-w window is a STEP FUNCTION of its centre: a
whole half-point bucket enters or leaves the moment the centre crosses an
edge. Measured 2026-09-01 on the live Week 1 board, that is not theoretical
-- wiring :data:`TOTALS_RESIDUAL_WEIGHT` moved the centre total by +0.042
(43.000 -> 43.042), which pushed the entire ``total_line == 41.5`` bucket
(38 games) outside the old +/-1.5 window, dropped the neighborhood 259 ->
221, moved the median actual total 43 -> 41, and moved the published guess
DOWN from KC 23 - DEN 20 to KC 22 - DEN 19 while the totals model was
arguing the total should be HIGHER (+0.42). A displayed number that moves
the wrong way because of a mechanical window edge is a defect, not a
finding. So each historical game is weighted by a triangular kernel on its
standardized distance from the centre::

    d = sqrt((delta_margin / h_m)**2 + (delta_total / h_t)**2)
    w = max(0, 1 - d)

with base bandwidths ``h_m = 1.0`` and ``h_t = 1.5`` inherited from the
first entry of :data:`_NEIGHBORHOOD_WINDOWS` (the old first window's
half-widths -- no new constant). ``w`` is 1 at the centre, falls linearly,
and reaches 0 exactly AT the bandwidth, so a game on the boundary carries
zero weight instead of a full vote: a sub-half-point blend nudge can no
longer flip the guess. The bandwidth widens along that same schedule --
continuously, by linear interpolation between its entries, so the
bandwidth itself is not a step function of the centre either -- until the
Kish effective sample size ``(sum w)**2 / sum w**2`` reaches
:data:`_MIN_NEIGHBORHOOD` (150, also inherited). Medians are weighted
medians and the exact-final modes are weighted counts. When the active weekly
forecast prices the game, the model's margin disagreement is blended in at
weight :data:`MODEL_RESIDUAL_WEIGHT` (0.2) -- see that constant's docstring
for the measurement showing why the model does NOT simply override the
market here the way it picks sides against it. Median-based numbers are the
right guess when the tiebreak metric is closest-total (median minimizes
absolute error); the exact-score modes are the right guess when the metric
is exact-score matching. Both are reported because the pool's metric is not
recorded anywhere in this repository.

Measured accuracy of the baseline itself (2009-2025, 4,630 games): the
market total misses the actual total by ~10.5 points on average (median 9.0,
bias +0.5 -- actuals run half a point OVER the line); each implied team
score misses by ~7.4 points. A tiebreaker guess is a coin toss weighted a
few points in your favour, not a prediction -- any write-up quoting this
module must keep that framing.

The dedicated over/under training regime that was queued here has now RUN
(:mod:`nfl_ats.totals`, `docs/totals_model.md`, artifact
``artifacts/totals_backtest/20260901T184010Z``). Its verdict is the same
shape as the margin side's: the market total is the better point estimate on
its own, and the model's residual is folded in at
:data:`TOTALS_RESIDUAL_WEIGHT` (0.1) rather than allowed to override it --
see that constant's docstring for the sweep. A second wave
(:mod:`nfl_ats.totals_wave2`, ``docs/totals_model_wave2.md``) screened 24
drive-pace columns on top of wave 1's 41 and came back the favourite,
``probability_positive`` 0.8235 for beating wave 1 -- see
:data:`TOTALS_RESIDUAL_WEIGHT`'s docstring for both sweeps. This module now
serves wave 2's model when the drive-pace feature table exists, falling back
to wave 1's when it does not (a fresh clone).

Spread-sign conventions, stated once
------------------------------------
Two sources, two conventions, converted at the edge and nowhere else:

- ``schedules.parquet`` ``spread_line``: POSITIVE = home favored by that
  many (verified empirically: mean(actual home margin - spread_line) = +0.06
  over 4,630 games). This module's ``home_expected_margin`` equals it.
- Odds-snapshot ``quotes.parquet`` HOME outcome ``line``: NEGATIVE = home
  favored (a home side at -2.5 gives 2.5). ``home_expected_margin`` is its
  negation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.totals import TotalsDataError, TotalsView, model_total_view
from nfl_ats.totals_wave2 import model_total_view_wave2

#: Widening (margin, total) KERNEL BANDWIDTHS for the historical
#: neighborhood, walked in order -- continuously, by linear interpolation
#: between neighbouring entries -- until the effective sample size reaches
#: ``_MIN_NEIGHBORHOOD``. These are the same numbers that were the hard
#: window half-widths before 2026-09-01; the first entry ``(1.0, 1.5)``
#: supplies the base bandwidths ``h_m``/``h_t`` the module docstring
#: describes. The final ``None`` entry still means "all of history,
#: unweighted" and is the fallback when even the widest bandwidth cannot
#: reach the floor (a tiny synthetic history in tests; with 4,630 real
#: games the finite entries always clear it first).
_NEIGHBORHOOD_WINDOWS: tuple[tuple[float, float] | None, ...] = (
    (1.0, 1.5),
    (1.5, 2.5),
    (2.5, 3.5),
    (3.5, 5.0),
    None,
)

#: The finite entries alone -- the interpolation knots for the continuous
#: widening. Index ``k`` is scale position ``float(k)``.
_BANDWIDTH_SCHEDULE: tuple[tuple[float, float], ...] = tuple(
    window for window in _NEIGHBORHOOD_WINDOWS if window is not None
)

#: Floor on the neighborhood's size. Unchanged in value and in intent from
#: the hard-window era; it is now read as a floor on the Kish EFFECTIVE
#: sample size ``(sum w)**2 / sum w**2``, which equals the plain count when
#: every weight is equal and is the standard weighted-sample equivalent
#: otherwise.
_MIN_NEIGHBORHOOD = 150

#: Bisection steps used to find the minimal bandwidth scale whose effective
#: sample size clears ``_MIN_NEIGHBORHOOD``. Derived, not chosen: each step
#: halves the bracket, so 40 steps resolve the scale to 2**-40 ~ 9e-13 of
#: one schedule entry -- roughly twelve orders of magnitude finer than the
#: half-point quantum this whole design exists to be insensitive to, and
#: still far above float64's ~2e-16 relative precision.
_BANDWIDTH_BISECTION_STEPS = 40

#: Relative tolerance for "the cumulative weight lands exactly on the half
#: point" in the weighted median. A float-comparison epsilon, not a model
#: parameter: float64 carries ~1e-16 relative precision and the cumulative
#: sum accumulates at most a few thousand terms, so 1e-12 leaves about four
#: decades of headroom while staying far below any real weight difference.
#: It exists so that uniform weights reproduce pandas' even-count median
#: (average of the two middle values) exactly.
_MEDIAN_TIE_TOLERANCE = 1e-12

#: Weight on the active model's margin disagreement when blending it into the
#: market margin for SCORE-GUESSING (not side-picking). Measured 2026-09-01 on
#: ``artifacts/opener_evaluation/20260819T174244Z`` (1,537 opener-graded
#: games, chronological prediction-level output): home-margin MAE is 9.912 at
#: k=0 (market alone), 9.906 at k=0.2 (the optimum, better in 5 of 6
#: seasons), and 10.003 at k=1 -- the RAW MODEL IS WORSE THAN THE MARKET as a
#: point estimate, even though it beats the market on sides. On the 487 games
#: where the model disagrees by >=2 points, no weight helps at all (9.800 at
#: k=0.3 vs 9.793 at k=0). Side-picking skill is not point-estimate skill:
#: the pick needs only P(margin > line) tilted past 50%, the tiebreaker needs
#: E[margin], and the market wins the latter. This constant is derived from
#: that sweep, not chosen (AGENTS.md: underived constants are defects).
MODEL_RESIDUAL_WEIGHT = 0.2

#: Weight on the TOTALS model's residual when blending it into the market
#: total. Two sweeps measured this, wave 1 then wave 2, and BOTH chose k=0.1
#: -- the weight is unchanged across the wiring switch below; what changes
#: is which fitted model produces the residual being blended in.
#:
#: Wave 1, measured 2026-09-01 by `nfl-ats totals-backtest`
#: (``artifacts/totals_backtest/20260901T184010Z``, 3,935 walk-forward
#: regular-season games 2010-2025, prediction-level output preserved, 41-column
#: allowlist on ``game_features.parquet``). The full MAE sweep over
#: ``total_line + k * predicted_residual``, in total points: k=0.0 10.4249
#: (market alone), k=0.1 10.4241 (the optimum), k=0.2 10.4260, k=0.3 10.4310,
#: k=0.4 10.4387, k=0.5 10.4486, k=0.6 10.4615, k=0.7 10.4785, k=0.8 10.4993,
#: k=0.9 10.5233, k=1.0 10.5495 -- so the RAW MODEL TOTAL IS WORSE THAN THE
#: MARKET TOTAL (10.549 vs 10.425), exactly the shape
#: :data:`MODEL_RESIDUAL_WEIGHT` found on the margin. The k=0.1 improvement is
#: +0.0008 total points, week-blocked bootstrap 95% [-0.0062, +0.0077],
#: ``probability_positive`` 0.583 over 261 week blocks; registry entry
#: ``totals_market_residual_blend``.
#:
#: Wave 2, measured 2026-09-01 by
#: ``scripts/totals_wave2_backtest.py --mode screen`` (``docs/
#: totals_model_wave2.md``, ``artifacts/totals_backtest_wave2/wp18run/screen``,
#: identical 3,935-game population, 65-column allowlist -- wave 1's 41 plus 24
#: drive-pace columns -- on ``game_features_pbp.parquet``). Wave 2's OWN
#: MAE-minimizing k, from its own independently re-swept grid, is also 0.1
#: (10.4221 vs market 10.4249, +0.0028 vs market). The decision metric for
#: ADOPTING wave 2 over wave 1 is the paired wave-2-vs-wave-1 comparison,
#: graded with wave 1 at its frozen k=0.1 and wave 2 at its own k=0.1: +0.0020
#: total points, week-blocked bootstrap 95% [-0.0024, +0.0063],
#: ``probability_positive`` 0.8235 over 261 week blocks; registry entry
#: ``totals_market_residual_wave2_vs_wave1``. Both intervals cross zero --
#: expected at this evaluator's resolution for a real small signal, not
#: grounds to reject either (AGENTS.md). 0.8235 is why the served number below
#: comes from wave 2's model rather than wave 1's: it is the favourite on the
#: project's EV decision rule, consistent everywhere it was checked (vs
#: market, per-season majority, playoffs -- see the doc). 0.583 is, as before,
#: why the weight itself is 0.1 and not 0.0. Derived from those two sweeps,
#: not chosen.
TOTALS_RESIDUAL_WEIGHT = 0.1


@dataclass(frozen=True)
class MarketConsensus:
    """One game's freshest market read: median across books in the newest
    local odds snapshot that quotes it, or the schedules row as fallback."""

    game_id: str
    home_expected_margin: float  # positive = home favored
    total_line: float
    source: str  # "snapshot <stamp> (<n> books)" or "schedules (fallback)"


@dataclass(frozen=True)
class ModelView:
    """The active model's margin opinion for the game, read from the newest
    weekly forecast that prices it -- the same numbers behind the played
    pick, shown so the guess can acknowledge a disagreement (e.g. Week 1
    DEN @ KC: market KC by 2.5, model KC by ~4.3) instead of silently
    ignoring it."""

    predicted_margin: float  # model's expected home margin
    forecast_line: float  # the spread_line the residual was measured against
    residual: float  # predicted_margin - forecast_line
    source: str


@dataclass(frozen=True)
class TiebreakerReport:
    game_id: str
    home: str
    away: str
    consensus: MarketConsensus
    #: The model's view when a forecast prices this game, else ``None``; the
    #: guess margin is then market + MODEL_RESIDUAL_WEIGHT * residual.
    model_view: ModelView | None
    #: The totals model's view when :mod:`nfl_ats.totals` can price this game,
    #: else ``None``; the guess total is then market +
    #: TOTALS_RESIDUAL_WEIGHT * residual. Mirrors ``model_view`` exactly.
    totals_view: TotalsView | None
    #: The margin the guess is actually built from (blended when a model
    #: view exists, the market consensus alone otherwise).
    guess_margin: float
    #: The total the guess is actually built from (blended when a totals view
    #: exists, the market consensus total alone otherwise).
    guess_total_line: float
    implied_home: float
    implied_away: float
    #: Kish EFFECTIVE sample size of the kernel-weighted neighborhood,
    #: rounded -- ``(sum w)**2 / sum w**2``, which is the plain game count
    #: when every weight is equal (the "all history" fallback).
    neighborhood_games: int
    neighborhood_window: str
    #: Weighted medians over that neighborhood.
    median_total: float
    median_home_margin: float
    #: Integer guess: closest-total-optimal (median-based), margin-consistent.
    guess_home: int
    guess_away: int
    #: Most common exact ``(home_score, away_score)`` finals in the
    #: neighborhood, with KERNEL-WEIGHTED counts (a float: a game half a
    #: bandwidth away casts half a vote) -- the exact-score-metric guess.
    common_scores: tuple[tuple[int, int, float], ...]
    #: Whole-history honest error bars for the baseline.
    total_mae: float
    total_median_ae: float
    total_bias: float
    implied_score_mae: float


def newest_schedules_path(data_root: Path) -> Path:
    hits = sorted((data_root / "raw").glob("*/schedules.parquet"))
    if not hits:
        raise FileNotFoundError(f"no schedules.parquet under {data_root / 'raw'}")
    return hits[-1]


def lined_finals(schedules: pd.DataFrame) -> pd.DataFrame:
    """Completed games with a recorded spread and total."""

    mask = (
        schedules["home_score"].notna()
        & schedules["away_score"].notna()
        & schedules["spread_line"].notna()
        & schedules["total_line"].notna()
    )
    return schedules.loc[mask]


def last_game_of_week(schedules: pd.DataFrame, season: int, week: int) -> pd.Series:
    """The week's last kickoff -- the pool's tiebreaker game -- by
    ``(gameday, gametime)``."""

    games = schedules.loc[
        (schedules["season"] == season)
        & (schedules["week"] == week)
        & (schedules["game_type"].astype(str) == "REG")
    ]
    if games.empty:
        raise ValueError(f"no REG games for season {season} week {week}")
    keys = games["gameday"].astype(str)
    if "gametime" in games.columns:
        keys = keys + " " + games["gametime"].astype(str).fillna("")
    last: pd.Series = games.loc[keys.sort_values().index[-1]]
    return last


def upcoming_week(schedules: pd.DataFrame, today: date) -> tuple[int, int]:
    """The (season, week) of the next REG game on or after ``today`` --
    the week whose card is currently in play."""

    regular = schedules.loc[schedules["game_type"].astype(str) == "REG"].copy()
    days = pd.to_datetime(regular["gameday"], errors="coerce")
    ahead = regular.loc[days.dt.date >= today]
    if ahead.empty:
        raise ValueError(f"no REG games on or after {today.isoformat()}")
    first = ahead.loc[pd.to_datetime(ahead["gameday"]).sort_values().index[0]]
    return int(first["season"]), int(first["week"])


def market_implied_scores(home_expected_margin: float, total_line: float) -> tuple[float, float]:
    home = (total_line + home_expected_margin) / 2.0
    away = (total_line - home_expected_margin) / 2.0
    return home, away


def snapshot_consensus(game_id: str, data_root: Path) -> MarketConsensus | None:
    """Median spread/total across books in the NEWEST snapshot quoting the
    game. Walks snapshots newest-first so one capture missing the game (an
    early-week partial board) falls back to the one before it."""

    snapshots = sorted((data_root / "market" / "raw").glob("*/quotes.parquet"), reverse=True)
    for quotes_path in snapshots:
        quotes = pd.read_parquet(
            quotes_path,
            columns=["nflverse_game_id", "market", "outcome_side", "line", "bookmaker_key"],
        )
        rows = quotes.loc[quotes["nflverse_game_id"].astype(str).eq(game_id)]
        if rows.empty:
            continue
        spreads = rows.loc[(rows["market"] == "spreads") & (rows["outcome_side"] == "HOME")]
        totals = rows.loc[(rows["market"] == "totals") & (rows["outcome_side"] == "OVER")]
        if spreads.empty or totals.empty:
            continue
        # One line per book (a book quotes each market once per snapshot,
        # but groupby-first makes that an invariant rather than a hope).
        spread_by_book = spreads.groupby("bookmaker_key")["line"].first()
        total_by_book = totals.groupby("bookmaker_key")["line"].first()
        return MarketConsensus(
            game_id=game_id,
            home_expected_margin=-float(spread_by_book.median()),
            total_line=float(total_by_book.median()),
            source=f"snapshot {quotes_path.parent.name} ({len(spread_by_book)} books)",
        )
    return None


def active_model_view(game_id: str, artifacts_root: Path) -> ModelView | None:
    """The active method's ``predicted_market_residual`` for the game, from
    the newest weekly forecast that prices it. ``None`` when no forecast
    covers the game (a historical query) or the artifact tree is absent (a
    fresh clone) -- the guess then simply uses the market alone."""

    active_path = artifacts_root / "active_ats_model.json"
    if not active_path.is_file():
        return None
    method = str(json.loads(active_path.read_text(encoding="utf-8")).get("method", ""))
    if not method:
        return None
    forecasts = sorted(
        (artifacts_root / "margin_predictions").glob("*/predictions.csv"), reverse=True
    )
    for predictions_path in forecasts:
        frame = pd.read_csv(predictions_path)
        required = {"game_id", "method", "predicted_margin", "predicted_market_residual"}
        if not required.issubset(frame.columns):
            continue
        rows = frame.loc[
            frame["game_id"].astype(str).eq(game_id)
            & frame["method"].astype(str).eq(method)
            & frame["predicted_market_residual"].notna()
        ]
        if rows.empty:
            continue
        row = rows.iloc[0]
        return ModelView(
            predicted_margin=float(row["predicted_margin"]),
            forecast_line=float(row["spread_line"]),
            residual=float(row["predicted_market_residual"]),
            source=f"forecast {predictions_path.parent.name} ({method})",
        )
    return None


@dataclass(frozen=True)
class Neighborhood:
    """The kernel-weighted set of historically similar games behind a guess.

    ``frame`` holds only the rows that carry positive weight, ``weights`` is
    aligned to it positionally, and ``effective_size`` is the Kish effective
    sample size ``(sum w)**2 / sum w**2`` -- the number the report shows and
    the number the widening schedule targets."""

    frame: pd.DataFrame
    weights: npt.NDArray[np.float64]
    label: str
    effective_size: float


def kernel_weights(
    finals: pd.DataFrame,
    home_expected_margin: float,
    total_line: float,
    margin_bandwidth: float,
    total_bandwidth: float,
) -> npt.NDArray[np.float64]:
    """Triangular kernel weight per historical game.

    ``w = max(0, 1 - d)`` on the standardized distance
    ``d = sqrt((delta_margin / h_m)**2 + (delta_total / h_t)**2)``: exactly 1
    at the centre, linearly decreasing, exactly 0 at and beyond the bandwidth
    ellipse. Continuous in the centre by construction -- that is the whole
    point (see the module docstring)."""

    margins = finals["spread_line"].to_numpy(dtype=float)
    totals = finals["total_line"].to_numpy(dtype=float)
    distance = np.sqrt(
        ((margins - home_expected_margin) / margin_bandwidth) ** 2
        + ((totals - total_line) / total_bandwidth) ** 2
    )
    weights: npt.NDArray[np.float64] = np.clip(1.0 - distance, 0.0, None)
    return weights


def effective_sample_size(weights: npt.NDArray[np.float64]) -> float:
    """Kish effective sample size ``(sum w)**2 / sum w**2``.

    Equals the plain count when every weight is equal, which is why it can
    inherit ``_MIN_NEIGHBORHOOD`` unchanged from the hard-window era."""

    total = float(weights.sum())
    squared = float((weights**2).sum())
    if squared <= 0.0:
        return 0.0
    return total * total / squared


def _bandwidths_at(scale: float) -> tuple[float, float]:
    """Bandwidths at a continuous position along ``_BANDWIDTH_SCHEDULE``.

    ``scale`` 0.0 is the first entry, 1.0 the second, and fractional values
    interpolate linearly between neighbours -- so the bandwidth, and hence
    every weight, is a continuous function of how far the schedule has been
    walked."""

    last = len(_BANDWIDTH_SCHEDULE) - 1
    lower_index = min(max(math.floor(scale), 0), last)
    upper_index = min(lower_index + 1, last)
    fraction = min(max(scale - lower_index, 0.0), 1.0)
    lower_margin, lower_total = _BANDWIDTH_SCHEDULE[lower_index]
    upper_margin, upper_total = _BANDWIDTH_SCHEDULE[upper_index]
    return (
        lower_margin + fraction * (upper_margin - lower_margin),
        lower_total + fraction * (upper_total - lower_total),
    )


def _label(margin_bandwidth: float, total_bandwidth: float) -> str:
    return f"±{margin_bandwidth:.2f} margin, ±{total_bandwidth:.2f} total"


def _neighborhood(
    finals: pd.DataFrame, home_expected_margin: float, total_line: float
) -> Neighborhood:
    """Weight history around ``(home_expected_margin, total_line)``.

    Walks ``_BANDWIDTH_SCHEDULE`` for the first entry whose effective sample
    size clears ``_MIN_NEIGHBORHOOD``, then bisects back toward the previous
    entry for the SMALLEST scale that still clears it. Bisecting is what
    keeps the bandwidth continuous in the centre: taking whole schedule steps
    would reintroduce exactly the step function the kernel removed, just at
    the ESS threshold instead of at a bucket edge."""

    def weights_at(scale: float) -> npt.NDArray[np.float64]:
        return kernel_weights(finals, home_expected_margin, total_line, *_bandwidths_at(scale))

    def pack(scale: float, weights: npt.NDArray[np.float64]) -> Neighborhood:
        positive = weights > 0.0
        return Neighborhood(
            frame=finals.loc[positive],
            weights=weights[positive],
            label=_label(*_bandwidths_at(scale)),
            effective_size=effective_sample_size(weights),
        )

    base = weights_at(0.0)
    if effective_sample_size(base) >= _MIN_NEIGHBORHOOD:
        return pack(0.0, base)

    upper: float | None = None
    for index in range(1, len(_BANDWIDTH_SCHEDULE)):
        if effective_sample_size(weights_at(float(index))) >= _MIN_NEIGHBORHOOD:
            upper = float(index)
            break
    if upper is None:
        # The schedule's final ``None`` entry: all of history, unweighted.
        return Neighborhood(
            frame=finals,
            weights=np.ones(len(finals), dtype=float),
            label="all history",
            effective_size=float(len(finals)),
        )

    lower = upper - 1.0
    for _ in range(_BANDWIDTH_BISECTION_STEPS):
        middle = (lower + upper) / 2.0
        if effective_sample_size(weights_at(middle)) >= _MIN_NEIGHBORHOOD:
            upper = middle
        else:
            lower = middle
    return pack(upper, weights_at(upper))


def weighted_median(values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]) -> float:
    """The weighted median of ``values``.

    The smallest value whose cumulative weight reaches half the total; when
    the cumulative weight lands exactly on the half point the two straddling
    values are averaged, so uniform weights reproduce ``pandas.Series.median``
    (including its even-count averaging) exactly."""

    if len(values) == 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    total = float(cumulative[-1])
    if total <= 0.0:
        return float("nan")
    half = total / 2.0
    index = min(int(np.searchsorted(cumulative, half, side="left")), len(ordered_values) - 1)
    on_the_half_point = math.isclose(
        float(cumulative[index]), half, rel_tol=_MEDIAN_TIE_TOLERANCE, abs_tol=0.0
    )
    if on_the_half_point and index + 1 < len(ordered_values):
        return float((ordered_values[index] + ordered_values[index + 1]) / 2.0)
    return float(ordered_values[index])


def weighted_score_counts(
    frame: pd.DataFrame, weights: npt.NDArray[np.float64]
) -> dict[tuple[int, int], float]:
    """Total kernel weight behind each exact ``(home_score, away_score)``
    final in the neighborhood. Sums to ``weights.sum()`` by construction."""

    counts: dict[tuple[int, int], float] = {}
    for home_score, away_score, weight in zip(
        frame["home_score"], frame["away_score"], weights, strict=True
    ):
        key = (int(home_score), int(away_score))
        counts[key] = counts.get(key, 0.0) + float(weight)
    return counts


def build_report(
    game: pd.Series,
    consensus: MarketConsensus,
    finals: pd.DataFrame,
    model_view: ModelView | None = None,
    totals_view: TotalsView | None = None,
) -> TiebreakerReport:
    guess_margin = consensus.home_expected_margin
    if model_view is not None:
        guess_margin += MODEL_RESIDUAL_WEIGHT * model_view.residual
    guess_total_line = consensus.total_line
    if totals_view is not None:
        guess_total_line += TOTALS_RESIDUAL_WEIGHT * totals_view.residual
    implied_home, implied_away = market_implied_scores(guess_margin, guess_total_line)
    neighborhood = _neighborhood(finals, guess_margin, guess_total_line)
    rows, weights = neighborhood.frame, neighborhood.weights
    actual_totals = (rows["home_score"] + rows["away_score"]).to_numpy(dtype=float)
    actual_margins = (rows["home_score"] - rows["away_score"]).to_numpy(dtype=float)
    median_total = weighted_median(actual_totals, weights)
    median_margin = weighted_median(actual_margins, weights)
    guess_total = round(median_total)
    guess_home = round((guess_total + median_margin) / 2.0)
    guess_away = guess_total - guess_home
    score_counts = weighted_score_counts(rows, weights)
    # Ties broken by score rather than by iteration order, so the reported
    # modes are deterministic across pandas/row orderings.
    ranked = sorted(score_counts.items(), key=lambda item: (-item[1], item[0]))
    common = tuple(
        (home_score, away_score, count) for (home_score, away_score), count in ranked[:3]
    )

    total_error = (finals["home_score"] + finals["away_score"]) - finals["total_line"]
    implied_home_all = (finals["total_line"] + finals["spread_line"]) / 2.0
    implied_away_all = (finals["total_line"] - finals["spread_line"]) / 2.0
    implied_mae = float(
        pd.concat(
            [
                (finals["home_score"] - implied_home_all).abs(),
                (finals["away_score"] - implied_away_all).abs(),
            ]
        ).mean()
    )
    return TiebreakerReport(
        game_id=str(game["game_id"]),
        home=str(game["home_team"]),
        away=str(game["away_team"]),
        consensus=consensus,
        model_view=model_view,
        totals_view=totals_view,
        guess_margin=guess_margin,
        guess_total_line=guess_total_line,
        implied_home=implied_home,
        implied_away=implied_away,
        neighborhood_games=round(neighborhood.effective_size),
        neighborhood_window=neighborhood.label,
        median_total=median_total,
        median_home_margin=median_margin,
        guess_home=guess_home,
        guess_away=guess_away,
        common_scores=common,
        total_mae=float(total_error.abs().mean()),
        total_median_ae=float(total_error.abs().median()),
        total_bias=float(total_error.mean()),
        implied_score_mae=implied_mae,
    )


def tiebreaker_report(
    data_root: Path,
    *,
    artifacts_root: Path | None = None,
    season: int | None = None,
    week: int | None = None,
    game_id: str | None = None,
    today: date | None = None,
    features_path: Path | None = None,
    wave2_features_path: Path | None = None,
) -> TiebreakerReport:
    """The full pipeline: resolve the game, read the freshest market, blend
    in the active model's view (weight :data:`MODEL_RESIDUAL_WEIGHT`) and the
    totals model's view (weight :data:`TOTALS_RESIDUAL_WEIGHT`), build the
    calibrated guess. ``game_id`` overrides ``season``/``week``; with neither,
    the week of the next upcoming game is used.

    The totals view now prefers WAVE 2 (:func:`nfl_ats.totals_wave2.
    model_total_view_wave2`, 65-column drive-pace allowlist), falling back to
    WAVE 1 (:func:`nfl_ats.totals.model_total_view`, 41 columns) only when the
    wave-2 feature table is absent -- a fresh clone, or a synthetic data root
    in tests -- never merely because wave 2 declined to price this one game
    (that case is market-only, same as wave 1's own contract; see
    :func:`nfl_ats.totals_wave2.model_total_view_wave2`'s docstring for why).
    ``wave2_features_path`` defaults to ``<data_root>/processed/
    game_features_pbp.parquet`` and is tried first; ``features_path`` defaults
    to ``<data_root>/processed/game_features.parquet`` and is now used only as
    the wave-1 fallback source. With neither table present the guess uses the
    market total alone, exactly as it did before the totals regime existed."""

    schedules = pd.read_parquet(newest_schedules_path(data_root))
    if game_id is not None:
        rows = schedules.loc[schedules["game_id"].astype(str).eq(game_id)]
        if rows.empty:
            raise ValueError(f"game_id {game_id!r} not in schedules")
        game = rows.iloc[0]
    else:
        if season is None or week is None:
            season, week = upcoming_week(schedules, today or date.today())
        game = last_game_of_week(schedules, season, week)

    consensus = snapshot_consensus(str(game["game_id"]), data_root)
    if consensus is None:
        if pd.isna(game.get("spread_line")) or pd.isna(game.get("total_line")):
            raise ValueError(
                f"no odds snapshot quotes {game['game_id']} and schedules has no line for it"
            )
        consensus = MarketConsensus(
            game_id=str(game["game_id"]),
            home_expected_margin=float(game["spread_line"]),
            total_line=float(game["total_line"]),
            source="schedules (fallback -- possibly stale)",
        )
    model_view = (
        active_model_view(str(game["game_id"]), artifacts_root)
        if artifacts_root is not None
        else None
    )
    wave1_features = (
        features_path
        if features_path is not None
        else data_root / "processed" / "game_features.parquet"
    )
    wave2_features = (
        wave2_features_path
        if wave2_features_path is not None
        else data_root / "processed" / "game_features_pbp.parquet"
    )
    if wave2_features.is_file():
        try:
            totals_view = model_total_view_wave2(str(game["game_id"]), data_root, wave2_features)
        except (TotalsDataError, KeyError, OSError, TypeError, ValueError):
            # A present but stale, corrupt, or misaligned PBP table must not
            # turn into a wave-1 view (or a fabricated residual).  The
            # tiebreaker remains usable from the market-only path.  The wave-2
            # model itself keeps raising TotalsDataError for direct callers so
            # data-quality failures remain visible to backtests and tests.
            totals_view = None
    else:
        # The PBP table itself is absent (a fresh clone) -- fall back to
        # wave 1's model, tagged so the report line names the fallback rather
        # than silently looking like a wave-2 number.
        totals_view = model_total_view(str(game["game_id"]), data_root, wave1_features)
        if totals_view is not None:
            totals_view = replace(
                totals_view,
                source=f"wave 1 fallback (PBP table absent) -- {totals_view.source}",
            )
    return build_report(game, consensus, lined_finals(schedules), model_view, totals_view)


def format_report(report: TiebreakerReport) -> str:
    lines = [
        f"tiebreaker guess -- {report.away} at {report.home} ({report.game_id})",
        f"market: {report.home} by {report.consensus.home_expected_margin:g}, "
        f"total {report.consensus.total_line:g}  [{report.consensus.source}]",
    ]
    if report.model_view is not None:
        lines += [
            f"model view: expects {report.home} by {report.model_view.predicted_margin:.2f} "
            f"vs the {report.model_view.forecast_line:g} forecast line "
            f"(disagreement {report.model_view.residual:+.2f})  "
            f"[{report.model_view.source}]",
            f"guess margin blends it at weight {MODEL_RESIDUAL_WEIGHT:g} (measured optimum "
            f"-- the raw model is WORSE than the market as a point estimate): "
            f"{report.home} by {report.guess_margin:.2f}",
        ]
    if report.totals_view is not None:
        lines += [
            f"model total view: expects {report.totals_view.predicted_total:.2f} "
            f"vs the {report.totals_view.market_total:g} market total "
            f"(disagreement {report.totals_view.residual:+.2f})  "
            f"[{report.totals_view.source}]",
            f"guess total blends it at weight {TOTALS_RESIDUAL_WEIGHT:g} (measured optimum "
            f"-- the raw model total is WORSE than the market total): "
            f"{report.guess_total_line:.2f}",
        ]
    lines += [
        f"implied score at the guess margin: {report.home} {report.implied_home:.2f}, "
        f"{report.away} {report.implied_away:.2f}",
        "",
        f"calibration neighborhood: effective {report.neighborhood_games} similar games "
        f"(kernel-weighted; {report.neighborhood_window})",
        f"  median actual total {report.median_total:g}, "
        f"median home margin {report.median_home_margin:+g}",
        "",
        f"GUESS (closest-total metric): {report.home} {report.guess_home}, "
        f"{report.away} {report.guess_away}"
        f"  (total {report.guess_home + report.guess_away})",
        "most common exact finals in the neighborhood (exact-score metric, weighted):",
    ]
    for home_score, away_score, count in report.common_scores:
        lines.append(f"  {report.home} {home_score} - {report.away} {away_score}  ({count:.1f}x)")
    lines += [
        "",
        "honest error bars (all 2009-2025 lined finals): the market total "
        f"misses by {report.total_mae:.1f} on average (median {report.total_median_ae:.1f}, "
        f"bias {report.total_bias:+.1f}); each implied team score misses by "
        f"{report.implied_score_mae:.1f}.",
    ]
    return "\n".join(lines)
